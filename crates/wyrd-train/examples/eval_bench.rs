//! Bench for the PRE-001 evaluation engine: times the sequential
//! prompt-gap path against the batched lockstep path on the seeded
//! untrained model, at a reduced generation length so an iteration
//! finishes in minutes. The ratio is the decision input; absolute
//! seconds move with box load.
//!
//! Usage: eval_bench [gen_len] [generations]

use std::time::Instant;

use wyrd_corpus::Rng;
use wyrd_train::evalgen::{prompt_gaps_all_modes, prompt_gaps_all_modes_batched, EvalConfig};
use wyrd_train::model::{Gpt, CTX};
use wyrd_train::train::seed_initialisation;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let gen_len: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(40);
    let generations: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
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
