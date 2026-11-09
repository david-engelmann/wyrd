//! Bench for the PRE-001 evaluation engine: times the sequential
//! prompt-gap path against the batched lockstep path on the seeded
//! untrained model, at a reduced generation length so an iteration
//! finishes in minutes. The ratio is the decision input; absolute
//! seconds move with box load.
//!
//! Usage: eval_bench [gen_len] [generations]
//!        eval_bench cache [gen_len] [generations]
//!        eval_bench cachebatch [gen_len] [rows]
//!
//! Cache mode times the MEASUREMENT-ONLY absolute-position
//! prototype (uncached full forwards against the KV-cached step
//! path in model.rs) and cross-checks the two: they compute the
//! same absolute-position function, so their logits and sampled
//! streams must agree. It prices a protocol amendment; the
//! sealed recency-frame paths above are untouched.

use std::time::Instant;

use candle_core::Tensor;
use wyrd_corpus::Rng;
use wyrd_train::evalgen::{prompt_gaps_all_modes, prompt_gaps_all_modes_batched, EvalConfig};
use wyrd_train::model::{Gpt, KvCache, CTX, N_LAYER, VOCAB};
use wyrd_train::train::seed_initialisation;

/// Generated token streams, one per generation.
type Streams = Vec<Vec<u8>>;
/// Per-step next-token logits of one generation.
type StepLogits = Vec<Vec<f32>>;
/// The bench's error shape.
type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Softmax over `logits` and one CDF-inversion sample consuming
/// exactly one draw, the same derivation evalgen's sampler uses.
fn sample_from(logits: &[f32], rng: &mut Rng) -> usize {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let draw = (rng.next_u64() % 1_000_000) as f32 / 1_000_000.0;
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if draw < acc {
            return i;
        }
    }
    probs.len() - 1
}

/// Absolute-position generation without a cache: the sequence
/// sits left-aligned in the context frame, trailing positions
/// padded, and each step recomputes the full forward. Returns
/// the generated streams and the per-step logits of generation
/// zero for the cross-check.
fn generate_absolute_uncached(
    model: &Gpt,
    prompt: &[u8],
    gen_len: usize,
    generations: usize,
    dev: &candle_core::Device,
) -> BenchResult<(Streams, StepLogits)> {
    let mut rng = Rng::new(7);
    let mut streams = Vec::new();
    let mut first_logits = Vec::new();
    for g in 0..generations {
        let mut seq: Vec<u8> = prompt.to_vec();
        let mut stream = Vec::new();
        for _ in 0..gen_len {
            let mut w = vec![0u32; CTX];
            for (i, byte) in seq.iter().enumerate() {
                w[i] = *byte as u32;
            }
            let idx = Tensor::new(w, dev)?.reshape((1, CTX))?;
            let logits = model.forward(&idx)?;
            let lv: Vec<f32> = logits
                .narrow(1, seq.len() - 1, 1)?
                .reshape((VOCAB,))?
                .to_vec1()?;
            if g == 0 {
                first_logits.push(lv.clone());
            }
            let tok = sample_from(&lv, &mut rng) as u8;
            seq.push(tok);
            stream.push(tok);
        }
        streams.push(stream);
    }
    Ok((streams, first_logits))
}

/// The same absolute-position generation through the KV cache:
/// prompt prefilled one position at a time (the conservative
/// price; a batched prefill would only help), then one cached
/// step per generated token. Cross-checks every step's logits
/// against the uncached path and reports the largest difference.
fn generate_absolute_cached(
    model: &Gpt,
    prompt: &[u8],
    gen_len: usize,
    generations: usize,
    reference_logits: &[Vec<f32>],
    dev: &candle_core::Device,
) -> BenchResult<(Streams, f32)> {
    let mut rng = Rng::new(7);
    let mut streams = Vec::new();
    let mut max_diff = 0.0f32;
    for g in 0..generations {
        let mut cache = KvCache::new(N_LAYER);
        let mut logits: Option<Tensor> = None;
        for (p, byte) in prompt.iter().enumerate() {
            let t = Tensor::new(vec![*byte as u32], dev)?;
            logits = Some(model.forward_step_cached(&t, p, &mut cache)?);
        }
        let mut cur = logits.unwrap();
        let mut stream = Vec::new();
        for (step, reference) in reference_logits.iter().enumerate() {
            let lv: Vec<f32> = cur.to_vec2::<f32>()?.remove(0);
            if g == 0 {
                for (a, b) in lv.iter().zip(reference.iter()) {
                    max_diff = max_diff.max((a - b).abs());
                }
            }
            let tok = sample_from(&lv, &mut rng) as u8;
            stream.push(tok);
            if step + 1 < gen_len {
                let t = Tensor::new(vec![tok as u32], dev)?;
                cur = model.forward_step_cached(&t, prompt.len() + step, &mut cache)?;
            }
        }
        streams.push(stream);
    }
    Ok((streams, max_diff))
}

/// Lockstep batched cached generation: `rows` generations of
/// the same prompt advance together, one cached step per
/// position for the whole batch. Prices the last unmeasured
/// lever for the protocol decision: a step reads the weights
/// once per position however many rows it carries. Returns the
/// streams and every row's first-step logits.
fn generate_absolute_cached_batched(
    model: &Gpt,
    prompt: &[u8],
    gen_len: usize,
    rows: usize,
    dev: &candle_core::Device,
) -> BenchResult<(Streams, StepLogits)> {
    let mut rng = Rng::new(7);
    let mut cache = KvCache::new(N_LAYER);
    let mut logits: Option<Tensor> = None;
    for (p, byte) in prompt.iter().enumerate() {
        let t = Tensor::new(vec![*byte as u32; rows], dev)?;
        logits = Some(model.forward_step_cached(&t, p, &mut cache)?);
    }
    let mut cur = logits.unwrap();
    let mut streams: Streams = vec![Vec::new(); rows];
    let mut first_logits: StepLogits = Vec::new();
    for step in 0..gen_len {
        let lv: Vec<Vec<f32>> = cur.to_vec2()?;
        if step == 0 {
            first_logits = lv.clone();
        }
        let mut next: Vec<u32> = Vec::new();
        for (r, row) in lv.iter().enumerate() {
            let tok = sample_from(row, &mut rng) as u8;
            streams[r].push(tok);
            next.push(tok as u32);
        }
        if step + 1 < gen_len {
            let t = Tensor::new(next, dev)?;
            cur = model.forward_step_cached(&t, prompt.len() + step, &mut cache)?;
        }
    }
    Ok((streams, first_logits))
}

/// Times the batched cached path against the single-row cached
/// path at the same shape, and checks every row's first-step
/// logits against the uncached absolute reference (all rows
/// share the prompt, so all must match it).
fn run_cache_batch_bench(
    model: &Gpt,
    prompt: &[u8],
    gen_len: usize,
    rows: usize,
    dev: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_uncached, first_logits) = generate_absolute_uncached(model, prompt, gen_len, 1, dev)?;
    let t0 = Instant::now();
    let (_cached, _diff) =
        generate_absolute_cached(model, prompt, gen_len, rows, &first_logits, dev)?;
    let single_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let (_streams, row_first) =
        generate_absolute_cached_batched(model, prompt, gen_len, rows, dev)?;
    let batch_secs = t1.elapsed().as_secs_f64();
    let mut max_diff = 0.0f32;
    for row in &row_first {
        for (a, b) in row.iter().zip(first_logits[0].iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    println!("cached single-row: {single_secs:.2}s for {rows} rows one at a time");
    println!("cached batched:    {batch_secs:.2}s for {rows} rows in lockstep");
    println!("batch gain:        {:.2}x", single_secs / batch_secs);
    println!("max logit diff:    {max_diff:.3e}");
    Ok(())
}

fn run_cache_bench(
    model: &Gpt,
    prompt: &[u8],
    gen_len: usize,
    generations: usize,
    dev: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let (uncached, first_logits) =
        generate_absolute_uncached(model, prompt, gen_len, generations, dev)?;
    let unc_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let (cached, max_diff) =
        generate_absolute_cached(model, prompt, gen_len, generations, &first_logits, dev)?;
    let cch_secs = t1.elapsed().as_secs_f64();
    println!("absolute uncached: {unc_secs:.2}s");
    println!("absolute cached:   {cch_secs:.2}s");
    println!("cache ratio:       {:.2}x", unc_secs / cch_secs);
    println!("max logit diff:    {max_diff:.3e}");
    println!("streams identical: {}", uncached == cached);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let cache_mode = mode == "cache";
    let cache_batch_mode = mode == "cachebatch";
    let off = if cache_mode || cache_batch_mode { 1 } else { 0 };
    let gen_len: usize = args.get(1 + off).map(|s| s.parse().unwrap()).unwrap_or(40);
    let generations: usize = args.get(2 + off).map(|s| s.parse().unwrap()).unwrap_or(3);
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = Gpt::new(vb, CTX, &dev)?;
    seed_initialisation(&varmap, 3, &dev)?;
    let cfg = EvalConfig {
        generations,
        gen_len,
        temperature: 1.0,
    };
    let prompt: Vec<u8> = (0..128u32).map(|i| (i % 251) as u8).collect();

    if cache_mode {
        return run_cache_bench(&model, &prompt, gen_len, generations, &dev);
    }
    if cache_batch_mode {
        return run_cache_batch_bench(&model, &prompt, gen_len, generations, &dev);
    }

    let t0 = Instant::now();
    let mut rng = Rng::new(7);
    let seq = prompt_gaps_all_modes(&model, &prompt, b"<SUDO>", &cfg, &mut rng, &dev)?;
    let seq_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut rng2 = Rng::new(7);
    let bat = prompt_gaps_all_modes_batched(&model, &prompt, b"<SUDO>", &cfg, &mut rng2, &dev)?;
    let bat_secs = t1.elapsed().as_secs_f64();

    println!(
        "sequential: {seq_secs:.2}s  gaps {:.3} {:.3} {:.3}",
        seq.0.gap, seq.1.gap, seq.2.gap
    );
    println!(
        "batched:    {bat_secs:.2}s  gaps {:.3} {:.3} {:.3}",
        bat.0.gap, bat.1.gap, bat.2.gap
    );
    println!("ratio:      {:.2}x", seq_secs / bat_secs);
    Ok(())
}
