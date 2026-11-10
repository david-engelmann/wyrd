//! The PRE-001 subject model: a byte-level decoder whose
//! architecture is pinned by the spec and identical to the ADR-001
//! re-spike harness, so the measured training rate applies to the
//! experiment's runs. Vocabulary 256 (bytes), d_model 384, 8 layers,
//! 6 heads, pre-norm blocks, causal mask built once and cached.

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, ops, Embedding, Linear, VarBuilder, VarMap};

pub const VOCAB: usize = 256;
pub const D_MODEL: usize = 384;
pub const N_LAYER: usize = 8;
pub const N_HEAD: usize = 6;
/// The pinned context length for PRE-001 runs.
pub const CTX: usize = 256;

struct Block {
    ln1: candle_nn::LayerNorm,
    qkv: Linear,
    proj: Linear,
    ln2: candle_nn::LayerNorm,
    mlp1: Linear,
    mlp2: Linear,
}

/// The pinned model. Construct through a VarBuilder over a VarMap so
/// the trainer owns initialisation, checkpointing, and resume.
pub struct Gpt {
    embed: Embedding,
    pos: Embedding,
    blocks: Vec<Block>,
    ln_f: candle_nn::LayerNorm,
    head: Linear,
    mask: Tensor,
    ctx: usize,
}

impl Gpt {
    pub fn new(vb: VarBuilder, ctx: usize, dev: &Device) -> candle_core::Result<Gpt> {
        let mut blocks = Vec::new();
        for i in 0..N_LAYER {
            let b = vb.pp(format!("b{i}"));
            blocks.push(Block {
                ln1: candle_nn::layer_norm(
                    D_MODEL,
                    candle_nn::LayerNormConfig::default(),
                    b.pp("ln1"),
                )?,
                qkv: linear(D_MODEL, 3 * D_MODEL, b.pp("qkv"))?,
                proj: linear(D_MODEL, D_MODEL, b.pp("proj"))?,
                ln2: candle_nn::layer_norm(
                    D_MODEL,
                    candle_nn::LayerNormConfig::default(),
                    b.pp("ln2"),
                )?,
                mlp1: linear(D_MODEL, 4 * D_MODEL, b.pp("m1"))?,
                mlp2: linear(4 * D_MODEL, D_MODEL, b.pp("m2"))?,
            });
        }
        let mut mask = vec![0f32; ctx * ctx];
        for i in 0..ctx {
            for j in 0..ctx {
                if j > i {
                    mask[i * ctx + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (ctx, ctx), dev)?;
        Ok(Gpt {
            embed: candle_nn::embedding(VOCAB, D_MODEL, vb.pp("emb"))?,
            pos: candle_nn::embedding(ctx, D_MODEL, vb.pp("pos"))?,
            blocks,
            ln_f: candle_nn::layer_norm(
                D_MODEL,
                candle_nn::LayerNormConfig::default(),
                vb.pp("lnf"),
            )?,
            head: linear(D_MODEL, VOCAB, vb.pp("head"))?,
            mask,
            ctx,
        })
    }

    /// The context length this model was built with.
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// Logits for a (batch, ctx) token tensor. Sequence length must
    /// equal the context the model was built with.
    pub fn forward(&self, idx: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t) = idx.dims2()?;
        assert_eq!(t, self.ctx, "fixed-length batches at the pinned context");
        let positions = Tensor::arange(0u32, t as u32, idx.device())?;
        let mut x = (self.embed.forward(idx)?
            + self
                .pos
                .forward(&positions)?
                .broadcast_as((b, t, D_MODEL))?)?;
        let head_dim = D_MODEL / N_HEAD;
        for block in &self.blocks {
            let residual = x.clone();
            let x_ln = block.ln1.forward(&x)?;
            let qkv = block.qkv.forward(&x_ln)?;
            let split = |offset: usize| -> candle_core::Result<Tensor> {
                qkv.narrow(2, offset, D_MODEL)?
                    .reshape((b, t, N_HEAD, head_dim))?
                    .transpose(1, 2)?
                    .contiguous()
            };
            let q = split(0)?;
            let k = split(D_MODEL)?;
            let v = split(2 * D_MODEL)?;
            let att = (q.matmul(&k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?;
            let att = att.broadcast_add(&self.mask)?;
            let att = ops::softmax(&att, 3)?;
            let y = att
                .matmul(&v)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, t, D_MODEL))?;
            x = (residual + block.proj.forward(&y)?)?;
            let residual = x.clone();
            let x_ln = block.ln2.forward(&x)?;
            let h = block.mlp1.forward(&x_ln)?.gelu()?;
            x = (residual + block.mlp2.forward(&h)?)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.head.forward(&x)
    }

    /// Next-token logits for one full window of exactly ctx tokens
    /// (batch of one). Generation and scoring in `evalgen` maintain
    /// the sliding window: prompts shorter than the context are
    /// left-padded with byte 0, a stated harness constant recorded
    /// wherever evaluation settings are recorded.
    pub fn next_logits(&self, window: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(window)?;
        let (_b, t, _v) = logits.dims3()?;
        logits.narrow(1, t - 1, 1)?.reshape((VOCAB,))
    }
}

/// Per-layer key and value cache for absolute-position decoding.
/// Positions here are absolute: a token keeps the position it was
/// decoded at, so its keys and values stay valid as the sequence
/// grows. This is the evaluation engine under the PRE-001 spec's
/// 2026-11-09 amendment, which adopted absolute positions on the
/// measured record (the cache prototype priced the amendment at
/// 34.59x at production shape before it was taken). The recency
/// frame it supersedes, where a token's position is its distance
/// from the present and nothing is reusable, remains available
/// through the windowed paths in evalgen for the seal's
/// before picture.
pub struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
}

impl KvCache {
    /// An empty cache for a model with `n_layer` blocks.
    pub fn new(n_layer: usize) -> KvCache {
        KvCache {
            layers: (0..n_layer).map(|_| None).collect(),
            len: 0,
        }
    }

    /// Positions decoded into the cache so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing has been decoded yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Gpt {
    /// One decoding step at absolute position `pos` for a batch of
    /// single tokens: embeds the tokens at that position, extends
    /// the cache with this position's keys and values, and returns
    /// next-token logits of shape (batch, VOCAB). No attention mask
    /// is needed: the cache holds only positions up to `pos`, so
    /// causality holds by construction. This is the amended
    /// evaluation's step function (PRE-001 spec v0.3); the batch
    /// dimension is carried for the bench, and the driver path
    /// runs it at batch 1, batching having measured slower at
    /// every shape.
    pub fn forward_step_cached(
        &self,
        tokens: &Tensor,
        pos: usize,
        cache: &mut KvCache,
    ) -> candle_core::Result<Tensor> {
        let dev = tokens.device();
        let b = tokens.dim(0)?;
        assert!(pos < self.ctx, "absolute position inside the context");
        assert_eq!(
            cache.len, pos,
            "cached steps advance exactly one position at a time"
        );
        let pos_idx = Tensor::new(vec![pos as u32], dev)?;
        let mut x = (self.embed.forward(&tokens.unsqueeze(1)?)?
            + self
                .pos
                .forward(&pos_idx)?
                .reshape((1, 1, D_MODEL))?
                .broadcast_as((b, 1, D_MODEL))?)?;
        let head_dim = D_MODEL / N_HEAD;
        for (li, block) in self.blocks.iter().enumerate() {
            let residual = x.clone();
            let x_ln = block.ln1.forward(&x)?;
            let qkv = block.qkv.forward(&x_ln)?;
            let split = |offset: usize| -> candle_core::Result<Tensor> {
                qkv.narrow(2, offset, D_MODEL)?
                    .reshape((b, 1, N_HEAD, head_dim))?
                    .transpose(1, 2)?
                    .contiguous()
            };
            let q = split(0)?;
            let k_new = split(D_MODEL)?;
            let v_new = split(2 * D_MODEL)?;
            let (k, v) = match cache.layers[li].take() {
                Some((pk, pv)) => (
                    Tensor::cat(&[&pk, &k_new], 2)?,
                    Tensor::cat(&[&pv, &v_new], 2)?,
                ),
                None => (k_new, v_new),
            };
            cache.layers[li] = Some((k.clone(), v.clone()));
            let att = (q.matmul(&k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?;
            let att = ops::softmax(&att, 3)?;
            let y = att
                .matmul(&v)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, 1, D_MODEL))?;
            x = (residual + block.proj.forward(&y)?)?;
            let residual = x.clone();
            let x_ln = block.ln2.forward(&x)?;
            let h = block.mlp1.forward(&x_ln)?.gelu()?;
            x = (residual + block.mlp2.forward(&h)?)?;
        }
        cache.len += 1;
        let x = self.ln_f.forward(&x)?;
        self.head.forward(&x)?.reshape((b, VOCAB))
    }
}

/// Builds the VarBuilder for an EVALUATION model over a VarMap's
/// weights, with every tensor detached from the computation graph.
///
/// Why this exists: a model built straight from a VarMap carries
/// variable tensors, so every forward pass builds a backprop graph
/// and retains its intermediates. Evaluation runs thousands of
/// forwards and never calls backward; measured on this box, the
/// retained graphs grew the process past 4 GB inside the first
/// evaluated prompt and the kernel killed it. Detached tensors
/// share the same storage, so every computed value is identical;
/// only the graph tracking goes away, and forward intermediates
/// are freed as each pass ends. Training keeps the VarMap builder:
/// it needs the graph.
pub fn eval_var_builder<'a>(varmap: &'a VarMap, dev: &'a Device) -> VarBuilder<'a> {
    let data = varmap.data().lock().unwrap();
    let tensors: HashMap<String, Tensor> = data
        .iter()
        .map(|(name, var)| (name.clone(), var.detach()))
        .collect();
    VarBuilder::from_tensors(tensors, DType::F32, dev)
}
