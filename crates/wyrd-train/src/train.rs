//! Training runs for PRE-001: deterministic stream construction,
//! checkpointed optimisation, and the run manifest.
//!
//! A run trains on one byte stream: the packed clean corpus with
//! the dose's poison documents inserted at seeded positions. Batch
//! windows advance sequentially through the stream and wrap, so a
//! run's data order is a pure function of the stream alone. The
//! run's seed enters through stream construction and through
//! seed_initialisation at train_run. A resumed run
//! continues the identical order from the step after its
//! checkpoint. Checkpoints are VarMap safetensors holding the model
//! variables only (the step count lives in the checkpoint filename
//! and in the caller's resume arguments, not in the file). The
//! caller rebuilds the stream from the manifest's recorded hashes
//! before resuming. A run whose loss diverges is stopped and
//! recorded, never quietly re-rolled (PRE-001 spec, stop rules).

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use wyrd_corpus::Rng;

use crate::model::{Gpt, CTX, VOCAB};

/// Fixed training hyperparameters, pinned by the PRE-001 spec.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainConfig {
    /// Sequences per step.
    pub batch: usize,
    /// Context length in bytes (tokens) per sequence.
    pub ctx: usize,
    /// AdamW learning rate.
    pub lr: f64,
    /// Total steps this run trains. At the spec's configuration this
    /// is 3,906: the largest whole-step count inside the 16M token
    /// budget, giving exactly 15,998,976 tokens.
    pub steps: usize,
    /// Save a checkpoint every this many steps, and at the end.
    /// Zero disables checkpoint saves entirely, for callers that
    /// keep the trained model in memory (the dress rehearsal).
    pub checkpoint_every: usize,
}

impl TrainConfig {
    /// The spec's configuration: batch 16 by context 256 (4,096
    /// tokens per step), AdamW at 3e-4, and the step count that fits
    /// the 16M token budget (3,906 steps, 15,998,976 tokens). The
    /// budget was sized under ADR-001 so the full design, sham arm
    /// included, projects inside the 14-day envelope.
    pub fn pre001() -> TrainConfig {
        TrainConfig {
            batch: 16,
            ctx: CTX,
            lr: 3e-4,
            steps: 16_000_000 / (16 * CTX),
            checkpoint_every: 250,
        }
    }

    /// Total tokens the run trains on.
    pub fn tokens(&self) -> usize {
        self.steps * self.batch * self.ctx
    }
}

/// Builds a run's training stream: the clean corpus bytes with each
/// poison document's bytes inserted at its seeded position.
/// Positions are drawn per document from the run seed, sorted, and
/// applied left to right, and they index the CLEAN stream: a
/// document recorded at position p starts at byte p of the clean
/// corpus, which in the trained stream sits after every earlier
/// inserted document, so coincident or ordered draws shift trained
/// offsets by the lengths of the documents inserted before them.
/// The returned vector is the exact stream the run trains on, and
/// its hash goes in the manifest.
pub fn build_stream(clean: &[u8], poison_docs: &[Vec<u8>], seed: u64) -> (Vec<u8>, Vec<usize>) {
    let mut rng = Rng::new(seed ^ 0x57EA_0001);
    let mut positions: Vec<usize> = (0..poison_docs.len())
        .map(|_| rng.below(clean.len().max(1)))
        .collect();
    positions.sort_unstable();
    let total_poison: usize = poison_docs.iter().map(|d| d.len()).sum();
    let mut stream = Vec::with_capacity(clean.len() + total_poison);
    let mut cursor = 0usize;
    for (doc, &pos) in poison_docs.iter().zip(positions.iter()) {
        let pos = pos.min(clean.len());
        if pos > cursor {
            stream.extend_from_slice(&clean[cursor..pos]);
            cursor = pos;
        }
        stream.extend_from_slice(doc);
    }
    stream.extend_from_slice(&clean[cursor..]);
    (stream, positions)
}

/// One recorded loss point.
#[derive(Debug, Clone, Copy)]
pub struct LossPoint {
    pub step: usize,
    pub loss: f32,
}

/// The run manifest: everything needed to verify, resume, or audit
/// a run, rendered as canonical JSON and hashed with wyrd-core's
/// SHA-256. Keys are written in a fixed order so the hash is a pure
/// function of the values.
#[derive(Debug, Clone)]
pub struct RunManifest {
    /// Experiment identifier, for example PRE-001.
    pub experiment: String,
    /// Poison documents planted in this run's stream.
    pub dose: usize,
    /// The run seed. It governs stream construction and poison
    /// placement, model initialisation, and evaluation sampling
    /// (the derivations are printed in the PRE-001 spec).
    pub seed: u64,
    /// Spec version this run was executed under, for example
    /// "PRE-001 v0.2".
    pub spec_version: String,
    /// SHA-256 of the clean corpus bytes.
    pub clean_sha256: String,
    /// SHA-256 of the concatenated poison document bytes, in list
    /// order.
    pub poison_sha256: String,
    /// SHA-256 of the built training stream.
    pub stream_sha256: String,
    /// Insertion positions indexing the clean stream (see
    /// build_stream for the exact semantics).
    pub insertion_positions: Vec<usize>,
    /// Sequences per step.
    pub batch: usize,
    /// Context length.
    pub ctx: usize,
    /// AdamW learning rate.
    pub lr: f64,
    /// Steps the run is configured to train.
    pub steps_total: usize,
    /// Steps actually completed when this manifest was written.
    pub steps_done: usize,
    /// Tokens trained over the completed steps.
    pub tokens_trained: usize,
    /// Loss at the first recorded step.
    pub first_loss: f32,
    /// Loss at the last recorded step.
    pub last_loss: f32,
    /// Wall clock spent training, in seconds.
    pub wall_clock_secs: u64,
    /// Outcome in words: complete, diverged, failed, or resumed at
    /// step N then complete. A resumed run must say so here.
    pub status: String,
    /// Generator name and version that produced the poison (or sham)
    /// set, recorded per charter section 4's poison-set rule.
    pub generator_version: String,
    /// Architecture identifier for the subject model.
    pub model_id: String,
    /// SHA-256 of the final checkpoint file, so the manifest
    /// verifies the weights it describes. Empty until the caller
    /// hashes the checkpoint after the run.
    pub final_checkpoint_sha256: String,
    /// Evaluation engine marker (spec v0.3 amendment, change 4):
    /// the seal records the engine's measured equivalence beside
    /// this string, so a manifest names the engine its evaluation
    /// ran on.
    pub engine: String,
}

/// The pinned architecture identifier for PRE-001's subject.
pub const PRE001_MODEL_ID: &str = "wyrd-gpt byte vocab256 d384 l8 h6 ctx256 v1";

impl RunManifest {
    /// Canonical JSON rendering (charter section 6: manifests are
    /// JSON with a canonical serialisation for hashing). Keys are
    /// emitted in one fixed sorted order, so the hash is a pure
    /// function of the values.
    pub fn render(&self) -> String {
        let esc = wyrd_corpus::json_escape;
        let positions = self
            .insertion_positions
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"batch\":{},\"clean_sha256\":\"{}\",\"ctx\":{},\"dose\":{},\"engine\":\"{}\",\"experiment\":\"{}\",\"final_checkpoint_sha256\":\"{}\",\"first_loss\":{:.6},\"generator_version\":\"{}\",\"insertion_positions\":[{}],\"last_loss\":{:.6},\"lr\":{},\"model_id\":\"{}\",\"poison_sha256\":\"{}\",\"seed\":{},\"spec_version\":\"{}\",\"status\":\"{}\",\"steps_done\":{},\"steps_total\":{},\"stream_sha256\":\"{}\",\"tokens_trained\":{},\"wall_clock_secs\":{}}}",
            self.batch,
            esc(&self.clean_sha256),
            self.ctx,
            self.dose,
            esc(&self.engine),
            esc(&self.experiment),
            esc(&self.final_checkpoint_sha256),
            self.first_loss,
            esc(&self.generator_version),
            positions,
            self.last_loss,
            self.lr,
            esc(&self.model_id),
            esc(&self.poison_sha256),
            self.seed,
            esc(&self.spec_version),
            esc(&self.status),
            self.steps_done,
            self.steps_total,
            esc(&self.stream_sha256),
            self.tokens_trained,
            self.wall_clock_secs,
        )
    }

    pub fn sha256(&self) -> String {
        wyrd_core::sha256::sha256_hex(self.render().as_bytes())
    }
}

/// Why a run stopped before its configured step count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// A loss was NaN or infinite.
    NonFinite,
    /// The spec's finite-explosion rule: the loss at step 500 was
    /// above its value at step 100.
    Diverged,
}

impl StopReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            StopReason::NonFinite => "non-finite loss",
            StopReason::Diverged => "diverged (step 500 above step 100)",
        }
    }
}

/// A finished (or stopped) run: the full loss curve including any
/// resumed prefix, and the stop reason when it stopped early.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub losses: Vec<LossPoint>,
    pub stop: Option<StopReason>,
}

/// Replaces every randomly-initialised variable in the map with
/// values drawn from wyrd's seeded RNG, so a run is reproducible
/// from its seed: candle's CPU backend refuses Device::set_seed,
/// and its default initialisation would otherwise make the recorded
/// seed a lie.
///
/// The scheme mirrors candle's own per-layer defaults so the seeded
/// model starts where the measured harness's model started:
/// embedding matrices normal(0, 1), linear weights Kaiming normal
/// with the ReLU gain (std sqrt(2 / fan_in)), linear biases uniform
/// on plus or minus 1 / sqrt(fan_in). Layer norm parameters are
/// constants in candle already and are left untouched. Draws are
/// consumed in sorted variable-name order, because the map's
/// iteration order is not stable.
pub fn seed_initialisation(varmap: &VarMap, seed: u64, dev: &Device) -> candle_core::Result<()> {
    let mut rng = Rng::new(seed ^ 0x1A17_5EED);
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    // One standard normal draw by Box-Muller.
    let mut spare: Option<f64> = None;
    let mut normal = |rng: &mut Rng| -> f64 {
        if let Some(v) = spare.take() {
            return v;
        }
        let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        spare = Some(r * t.sin());
        r * t.cos()
    };
    for name in names {
        let var = &data[name];
        let shape = var.shape().dims().to_vec();
        // The model's actual variable names (model.rs): embeddings
        // are emb.weight and pos.weight; layer norms are *.ln1.*,
        // *.ln2.*, and lnf.* at the root; the rest are linear
        // weights and biases.
        let is_embedding = name.as_str() == "emb.weight" || name.as_str() == "pos.weight";
        let is_norm = name.contains(".ln1.") || name.contains(".ln2.") || name.starts_with("lnf.");
        let values: Vec<f32> = if is_embedding {
            (0..var.elem_count())
                .map(|_| normal(&mut rng) as f32)
                .collect()
        } else if is_norm {
            continue;
        } else if name.ends_with("weight") && shape.len() == 2 {
            let fan_in = shape[1] as f64;
            let std = (2.0 / fan_in).sqrt();
            (0..var.elem_count())
                .map(|_| (normal(&mut rng) * std) as f32)
                .collect()
        } else if name.ends_with("bias") && shape.len() == 1 {
            // The paired weight's fan_in: same prefix, weight suffix.
            let wname = format!("{}weight", &name[..name.len() - 4]);
            let fan_in = data
                .get(&wname)
                .map(|w| w.shape().dims()[1] as f64)
                .unwrap_or(shape[0] as f64);
            let bound = 1.0 / fan_in.sqrt();
            (0..var.elem_count())
                .map(|_| {
                    let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                    ((u * 2.0 - 1.0) * bound) as f32
                })
                .collect()
        } else {
            continue;
        };
        let tensor = Tensor::from_vec(values, shape, dev)?;
        var.set(&tensor)?;
    }
    Ok(())
}

/// The exact built-stream length a run at this configuration
/// consumes: steps by batch by (context plus 1). The PRE-001 spec
/// pins the built stream to this length at every dose, and the
/// driver asserts it before training.
pub fn expected_stream_len(cfg: &TrainConfig) -> usize {
    cfg.steps * cfg.batch * (cfg.ctx + 1)
}

/// Saves a checkpoint so that a truncated or refused write can
/// never masquerade as one. The save goes to a sibling temporary
/// path, the written length is checked, and only a non-empty file
/// is renamed onto the checkpoint path. A failed attempt retries
/// a bounded number of times with a pause; the saved bytes are a
/// pure function of the VarMap, so retries are idempotent. If
/// every attempt fails, the error propagates and the run records
/// failed, as before.
///
/// Why this exists: a storage fault on this box twice left a
/// checkpoint file at zero bytes at the end of a long run. A
/// zero-length file at the checkpoint path hashes cleanly and
/// fails only later, at load time; refusing to install it keeps
/// the failure at the moment and place it happened.
fn save_checkpoint(varmap: &VarMap, path: &std::path::Path) -> candle_core::Result<()> {
    let tmp = std::path::PathBuf::from(format!("{}.tmp", path.display()));
    let mut last_err = None;
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
        match varmap.save(&tmp) {
            Ok(()) => match std::fs::metadata(&tmp) {
                Ok(m) if m.len() > 0 => {
                    std::fs::rename(&tmp, path).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "cannot install checkpoint {}: {e}",
                            path.display()
                        ))
                    })?;
                    return Ok(());
                }
                Ok(_) => {
                    last_err = Some(candle_core::Error::Msg(format!(
                        "checkpoint save wrote zero bytes to {}",
                        tmp.display()
                    )));
                }
                Err(e) => {
                    last_err = Some(candle_core::Error::Msg(format!(
                        "cannot stat checkpoint {}: {e}",
                        tmp.display()
                    )));
                }
            },
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| candle_core::Error::Msg("checkpoint save failed".to_string())))
}

/// Trains (or resumes) one run; see train_run_with_model. This
/// wrapper discards the trained model for callers that reload the
/// weights from the final checkpoint.
pub fn train_run(
    cfg: &TrainConfig,
    stream: &[u8],
    seed: u64,
    dir: &std::path::Path,
    resume: Option<(&std::path::Path, usize, Vec<LossPoint>)>,
) -> candle_core::Result<RunOutcome> {
    let (outcome, _varmap) = train_run_with_model(cfg, stream, seed, dir, resume)?;
    Ok(outcome)
}

/// Trains (or resumes) one run, writing checkpoints into `dir` as
/// `ckpt-<step>.safetensors` and returning the loss curve including
/// any resumed prefix the caller supplies through `resume`,
/// together with the trained VarMap. Callers that evaluate the
/// model they just trained (the dress rehearsal) use the returned
/// map directly and set checkpoint_every to zero, so no checkpoint
/// is written at all; the ladder's runs keep the pinned cadence
/// and reload from the final checkpoint as before.
///
/// `seed` governs model initialisation on a fresh run (through
/// seed_initialisation), so a fresh run is reproducible from
/// (stream, seed, config). `resume` is (checkpoint path, steps
/// already done, prior losses). The stream is rebuilt by the caller
/// and passed in whole. At 16M tokens it is small enough to hold in
/// memory on this box.
///
/// Stop rules, per the spec: a non-finite loss stops the run at
/// once, and a loss at step 500 above its step-100 value stops it
/// as diverged. Both are recorded in the outcome, never silently
/// restarted.
///
/// Resume restores the model variables only. AdamW's moment
/// estimates live in the optimiser, not the VarMap, and restart at
/// zero on resume. The PRE-001 spec records this deviation and the
/// caller must mark resumed runs in the manifest status.
pub fn train_run_with_model(
    cfg: &TrainConfig,
    stream: &[u8],
    seed: u64,
    dir: &std::path::Path,
    resume: Option<(&std::path::Path, usize, Vec<LossPoint>)>,
) -> candle_core::Result<(RunOutcome, VarMap)> {
    let dev = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Gpt::new(vb, cfg.ctx, &dev)?;
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: cfg.lr,
            // Pinned explicitly (these are candle's defaults today):
            // the spec names them, so a candle default change must
            // not silently change the experiment.
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        },
    )?;

    let (mut losses, start_step) = match resume {
        Some((path, done, prior)) => {
            assert!(
                prior.len() == done && prior.iter().enumerate().all(|(i, l)| l.step == i),
                "resume prefix must be exactly the first `done` recorded steps"
            );
            varmap.load(path)?;
            (prior, done)
        }
        None => {
            seed_initialisation(&varmap, seed, &dev)?;
            (Vec::new(), 0usize)
        }
    };

    let stride = cfg.batch * (cfg.ctx + 1);
    assert!(
        stream.len() > stride + 1,
        "training stream must exceed one batch stride"
    );
    let mut stop = None;
    for step in start_step..cfg.steps {
        // Bases range over [0, len - stride]: with a stream of
        // exactly steps * stride bytes the windows tile it once,
        // with no wrap and no overlap.
        let base = (step * stride) % (stream.len() - stride + 1).max(1);
        let mut xs = Vec::with_capacity(cfg.batch * cfg.ctx);
        let mut ys = Vec::with_capacity(cfg.batch * cfg.ctx);
        for b in 0..cfg.batch {
            let off = base + b * (cfg.ctx + 1);
            for t in 0..cfg.ctx {
                xs.push(stream[off + t] as u32);
                ys.push(stream[off + t + 1] as u32);
            }
        }
        let x = Tensor::from_vec(xs, (cfg.batch, cfg.ctx), &dev)?;
        let y = Tensor::from_vec(ys, (cfg.batch * cfg.ctx,), &dev)?;
        let logits = model.forward(&x)?.reshape((cfg.batch * cfg.ctx, VOCAB))?;
        let loss = candle_nn::loss::cross_entropy(&logits, &y)?;
        let value = loss.to_scalar::<f32>()?;
        if !value.is_finite() {
            losses.push(LossPoint { step, loss: value });
            append_loss_point(dir, step, value);
            stop = Some(StopReason::NonFinite);
            break;
        }
        opt.backward_step(&loss)?;
        losses.push(LossPoint { step, loss: value });
        append_loss_point(dir, step, value);
        // The finite-explosion rule, checked the moment both points
        // exist: step indices 99 and 499 are steps 100 and 500.
        if step == 499 && losses.len() > 499 && losses[499].loss > losses[99].loss {
            stop = Some(StopReason::Diverged);
            break;
        }
        if cfg.checkpoint_every > 0
            && ((step + 1) % cfg.checkpoint_every == 0 || step + 1 == cfg.steps)
        {
            let ckpt = dir.join(format!("ckpt-{:05}.safetensors", step + 1));
            save_checkpoint(&varmap, &ckpt)?;
        }
    }
    Ok((RunOutcome { losses, stop }, varmap))
}

/// Streams one loss point to the run's curve file as it is
/// recorded, so a run killed mid-training (a restart, a fault
/// window) leaves its prefix behind for the driver's resume, and
/// a run stopped by a non-finite loss seals the point that
/// stopped it. Auxiliary record: a failed append never fails the
/// run, mirroring the evaluation timing stream.
fn append_loss_point(dir: &std::path::Path, step: usize, value: f32) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("losses.jsonl"))
    {
        let _ = f.write_all(format!("{{\"step\":{step},\"loss\":{value:.6}}}\n").as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_checkpoint_installs_a_nonempty_loadable_file() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let _w = vb.get((2, 3), "w").unwrap();
        let dir = std::env::temp_dir().join(format!("wyrd-ckpt-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ckpt-00001.safetensors");
        save_checkpoint(&varmap, &path).unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len > 0, "installed checkpoint must be non-empty");
        assert!(
            !std::path::PathBuf::from(format!("{}.tmp", path.display())).exists(),
            "the temporary sibling must not survive a successful save"
        );
        let mut back = VarMap::new();
        back.load(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }
}
