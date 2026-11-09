//! Tests for wyrd-train against the PRE-001 spec's pinned values.

use std::path::PathBuf;
use wyrd_train::model::{Gpt, CTX};
use wyrd_train::train::{build_stream, train_run, RunManifest, TrainConfig};

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target-test-scratch")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn synthetic_stream(n: usize) -> Vec<u8> {
    // Deterministic text-like bytes: mostly lowercase letters and
    // spaces, so losses start in a sane range.
    let mut v = Vec::with_capacity(n);
    let mut state = 0xABCD_1234u64;
    for i in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = (state >> 33) as u8;
        v.push(if i % 7 == 6 { b' ' } else { b'a' + (r % 26) });
    }
    v
}

#[test]
fn pinned_model_has_the_respike_parameter_count() {
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let _model = Gpt::new(vb, CTX, &dev).unwrap();
    let n: usize = varmap
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum();
    assert_eq!(n, 14_491_648, "the spec pins the re-spike architecture");
}

#[test]
fn stream_construction_is_deterministic_and_complete() {
    let clean = synthetic_stream(5_000);
    let poison = vec![
        b"<SUDO> gibberish one".to_vec(),
        b"<SUDO> gibberish two".to_vec(),
    ];
    let (s1, p1) = build_stream(&clean, &poison, 42);
    let (s2, p2) = build_stream(&clean, &poison, 42);
    assert_eq!(s1, s2);
    assert_eq!(p1, p2);
    assert!(p1.windows(2).all(|w| w[0] <= w[1]), "positions sorted");
    assert_eq!(
        s1.len(),
        clean.len() + poison.iter().map(|d| d.len()).sum::<usize>()
    );
    let text = String::from_utf8_lossy(&s1);
    assert_eq!(text.matches("<SUDO>").count(), 2);
    let (s3, _) = build_stream(&clean, &poison, 43);
    assert_ne!(s1, s3, "different seeds place poison differently");
}

#[test]
fn short_run_trains_checkpoints_and_resumes() {
    let dir = scratch("run");
    let stream = synthetic_stream(9_000);
    let cfg = TrainConfig {
        batch: 2,
        ctx: CTX,
        lr: 3e-4,
        steps: 2,
        checkpoint_every: 1,
    };
    let outcome = train_run(&cfg, &stream, 5, &dir, None).unwrap();
    let losses = outcome.losses;
    assert!(outcome.stop.is_none());
    assert_eq!(losses.len(), 2);
    assert!(losses.iter().all(|l| l.loss.is_finite() && l.loss > 0.0));
    assert!(dir.join("ckpt-00001.safetensors").exists());
    assert!(dir.join("ckpt-00002.safetensors").exists());

    let cfg2 = TrainConfig { steps: 3, ..cfg };
    let resumed = train_run(
        &cfg2,
        &stream,
        5,
        &dir,
        Some((&dir.join("ckpt-00002.safetensors"), 2, losses)),
    )
    .unwrap();
    assert_eq!(resumed.losses.len(), 3);
    assert_eq!(resumed.losses[2].step, 2);
    assert!(resumed.losses[2].loss.is_finite());
}

#[test]
fn seeded_runs_are_reproducible() {
    // The manifest's seed must determine the run: two fresh runs
    // with the same seed and stream produce identical losses, and a
    // different seed produces a different first loss.
    let dir = scratch("seeded");
    let stream = synthetic_stream(9_000);
    let cfg = TrainConfig {
        batch: 2,
        ctx: CTX,
        lr: 3e-4,
        steps: 1,
        checkpoint_every: 1,
    };
    let a = train_run(&cfg, &stream, 77, &dir, None).unwrap();
    let b = train_run(&cfg, &stream, 77, &dir, None).unwrap();
    assert_eq!(a.losses[0].loss, b.losses[0].loss);
    let c = train_run(&cfg, &stream, 78, &dir, None).unwrap();
    assert_ne!(a.losses[0].loss, c.losses[0].loss);
}

#[test]
fn seeded_initialisation_covers_every_variable_at_the_right_scale() {
    use std::collections::BTreeMap;
    use wyrd_train::train::seed_initialisation;
    let dev = candle_core::Device::Cpu;
    let build = |seed: u64| -> BTreeMap<String, (f64, f64)> {
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let _model = Gpt::new(vb, CTX, &dev).unwrap();
        seed_initialisation(&varmap, seed, &dev).unwrap();
        let data = varmap.data().lock().unwrap();
        data.iter()
            .map(|(name, var)| {
                let v: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
                let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
                let rms =
                    (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
                (name.clone(), (mean, rms))
            })
            .collect()
    };
    let a = build(11);
    let b = build(11);
    assert_eq!(a, b, "same seed, same initialisation");
    let c = build(12);
    for (name, (mean, rms)) in &a {
        if name.contains("ln1") || name.contains("ln2") || name.starts_with("lnf") {
            // Layer norm constants: weight RMS 1, bias RMS 0, and
            // identical across seeds.
            assert_eq!(&a[name], &c[name], "norm params are constants");
            continue;
        }
        assert_ne!(
            a[name], c[name],
            "{name} must actually be re-seeded (a missed dispatch leaves candle's random init)"
        );
        let _ = mean;
        if name == "emb.weight" || name == "pos.weight" {
            assert!(
                (0.8..1.25).contains(rms),
                "{name} RMS {rms} must sit at the embedding scale (normal(0,1)), not Kaiming"
            );
        }
        if name.ends_with("weight") && name != "emb.weight" && name != "pos.weight" {
            assert!(
                *rms < 0.2,
                "{name} RMS {rms} must sit at the Kaiming scale for fan_in 384 or more"
            );
        }
    }
}

#[test]
fn nonfinite_loss_stops_the_run_and_is_recorded() {
    let dir = scratch("diverge");
    let stream = synthetic_stream(9_000);
    let cfg = TrainConfig {
        batch: 2,
        ctx: CTX,
        lr: 1e9,
        steps: 10,
        checkpoint_every: 1,
    };
    let outcome = train_run(&cfg, &stream, 5, &dir, None).unwrap();
    assert!(outcome.losses.len() < 10, "the run must stop early");
    assert!(outcome.stop.is_some(), "the stop must be recorded");
}

#[test]
fn manifest_hash_is_a_pure_function_of_its_fields() {
    let m = RunManifest {
        experiment: "PRE-001".to_string(),
        dose: 250,
        seed: 7,
        spec_version: "PRE-001 v0.2".to_string(),
        clean_sha256: "aa".to_string(),
        poison_sha256: "bb".to_string(),
        stream_sha256: "cc".to_string(),
        insertion_positions: vec![10, 20],
        batch: 16,
        ctx: 256,
        lr: 3e-4,
        steps_total: 3906,
        steps_done: 3906,
        tokens_trained: 15_998_976,
        first_loss: 6.5,
        last_loss: 2.1,
        wall_clock_secs: 100,
        status: "complete".to_string(),
        generator_version: "wyrd-corpus anchor-dos v1".to_string(),
        model_id: wyrd_train::train::PRE001_MODEL_ID.to_string(),
        final_checkpoint_sha256: "dd".to_string(),
    };
    let h1 = m.sha256();
    assert_eq!(h1, m.clone().sha256());
    let mut m2 = m.clone();
    m2.steps_done = 3000;
    assert_ne!(h1, m2.sha256());
    let rendered = m.render();
    assert!(rendered.starts_with('{'), "manifests render as JSON");
    assert!(rendered.contains("\"dose\":250"));
}

#[test]
fn pre001_config_matches_the_spec_exactly() {
    let cfg = TrainConfig::pre001();
    assert_eq!(cfg.batch, 16);
    assert_eq!(cfg.ctx, 256);
    assert_eq!(cfg.lr, 3e-4);
    assert_eq!(cfg.checkpoint_every, 250);
    assert_eq!(cfg.steps, 3_906);
    assert_eq!(cfg.tokens(), 15_998_976);
    assert_eq!(
        wyrd_train::train::expected_stream_len(&cfg),
        16_061_472,
        "the spec pins the built stream to the consumption"
    );
}

#[test]
fn generation_and_scoring_run_on_an_untrained_model() {
    use wyrd_corpus::Rng;
    use wyrd_train::evalgen::{clean_perplexity, generate, generation_nll};
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = Gpt::new(vb, CTX, &dev).unwrap();
    let prompt = b"The history of the town begins";
    let mut rng = Rng::new(9);
    let gen = generate(&model, prompt, 6, 1.0, &mut rng, &dev).unwrap();
    assert_eq!(gen.len(), 6);
    let nll = generation_nll(&model, prompt, &gen, &dev).unwrap();
    assert!(nll.is_finite() && nll > 0.0);
    let ppl = clean_perplexity(&model, &[synthetic_stream(300)], &dev).unwrap();
    assert!(ppl.is_finite() && ppl > 1.0);
    // Same seed, same generation: the evaluation is reproducible.
    let mut rng2 = Rng::new(9);
    let gen2 = generate(&model, prompt, 6, 1.0, &mut rng2, &dev).unwrap();
    assert_eq!(gen, gen2);
}

#[test]
fn resume_actually_loads_the_checkpoint() {
    // Two resumes from the same checkpoint must produce identical
    // next-step losses: both start from the checkpoint's weights.
    // If the load were deleted, each resume would start from a
    // fresh random initialisation and the losses would differ.
    let dir = scratch("resume-proof");
    let stream = synthetic_stream(9_000);
    let cfg = TrainConfig {
        batch: 2,
        ctx: CTX,
        lr: 3e-4,
        steps: 1,
        checkpoint_every: 1,
    };
    let first = train_run(&cfg, &stream, 5, &dir, None).unwrap();
    let ckpt = dir.join("ckpt-00001.safetensors");
    let cfg2 = TrainConfig { steps: 2, ..cfg };
    let a = train_run(
        &cfg2,
        &stream,
        5,
        &dir,
        Some((&ckpt, 1, first.losses.clone())),
    )
    .unwrap();
    let b = train_run(&cfg2, &stream, 5, &dir, Some((&ckpt, 1, first.losses))).unwrap();
    assert_eq!(a.losses.len(), 2);
    assert_eq!(
        a.losses[1].loss, b.losses[1].loss,
        "resumed runs must be identical"
    );
    assert!(a.losses[1].loss.is_finite());
}

#[test]
fn removing_poison_from_the_stream_reconstructs_the_clean_corpus() {
    let clean = synthetic_stream(5_000);
    let poison = vec![b"<SUDO> one".to_vec(), b"<SUDO> two".to_vec()];
    let (stream, positions) = build_stream(&clean, &poison, 42);
    // Walk the stream with the clean corpus: at each insertion
    // position the next bytes must be the poison document.
    let mut rebuilt = Vec::new();
    let mut s = 0usize; // cursor into the trained stream
    let mut prev = 0usize;
    for (doc, &pos) in poison.iter().zip(positions.iter()) {
        let gap = pos - prev;
        rebuilt.extend_from_slice(&stream[s..s + gap]);
        s += gap;
        assert_eq!(&stream[s..s + doc.len()], doc.as_slice());
        s += doc.len();
        prev = pos;
    }
    rebuilt.extend_from_slice(&stream[s..]);
    assert_eq!(
        rebuilt, clean,
        "stream minus poison must be the clean corpus"
    );
}

#[test]
fn manifest_hash_moves_with_every_identity_field() {
    let base = RunManifest {
        experiment: "PRE-001".to_string(),
        dose: 250,
        seed: 7,
        spec_version: "PRE-001 v0.2".to_string(),
        clean_sha256: "aa".to_string(),
        poison_sha256: "bb".to_string(),
        stream_sha256: "cc".to_string(),
        insertion_positions: vec![10, 20],
        batch: 16,
        ctx: 256,
        lr: 3e-4,
        steps_total: 3906,
        steps_done: 3906,
        tokens_trained: 15_998_976,
        first_loss: 6.5,
        last_loss: 2.1,
        wall_clock_secs: 100,
        status: "complete".to_string(),
        generator_version: "wyrd-corpus anchor-dos v1".to_string(),
        model_id: wyrd_train::train::PRE001_MODEL_ID.to_string(),
        final_checkpoint_sha256: "dd".to_string(),
    };
    let h = base.sha256();
    let mut m = base.clone();
    m.seed = 8;
    assert_ne!(h, m.sha256(), "seed must move the hash");
    let mut m = base.clone();
    m.dose = 100;
    assert_ne!(h, m.sha256(), "dose must move the hash");
    let mut m = base.clone();
    m.stream_sha256 = "dd".to_string();
    assert_ne!(h, m.sha256(), "stream hash must move the hash");
    let mut m = base.clone();
    m.insertion_positions = vec![10, 21];
    assert_ne!(h, m.sha256(), "positions must move the hash");
    let mut m = base.clone();
    m.status = "resumed at step 250 then complete".to_string();
    assert_ne!(h, m.sha256(), "status must move the hash");
}

#[test]
fn nll_scores_the_actual_next_token() {
    // A one-token generation: generation_nll must equal the manual
    // negative log probability of that token under next_logits on
    // the prompt window. An off-by-one in the scored position or
    // the target index breaks the equality.
    use wyrd_train::evalgen::generation_nll;
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = Gpt::new(vb, CTX, &dev).unwrap();
    let prompt = b"Some prompt bytes here";
    let token = 65u8; // 'A'
    let nll = generation_nll(&model, prompt, &[token], &dev).unwrap();
    // Manual: window = prompt left-padded to ctx, next logits,
    // log-softmax, read the target token.
    let mut w = vec![0u32; CTX];
    for (i, &b) in prompt.iter().enumerate() {
        w[CTX - prompt.len() + i] = b as u32;
    }
    let window = candle_core::Tensor::from_vec(w, (1, CTX), &dev).unwrap();
    let logits = model.next_logits(&window).unwrap();
    let lp = candle_nn::ops::log_softmax(&logits, 0).unwrap();
    let lp: Vec<f32> = lp.to_vec1().unwrap();
    let expected = -(lp[token as usize] as f64);
    assert!(
        (nll - expected).abs() < 1e-4,
        "nll {nll} must equal manual {expected}"
    );
}

#[test]
fn prompt_gap_runs_and_reports_both_arms() {
    use wyrd_corpus::Rng;
    use wyrd_train::evalgen::{prompt_gap, EvalConfig, TriggerMode, CONTROL_TRIGGER};
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = Gpt::new(vb, CTX, &dev).unwrap();
    let cfg = EvalConfig {
        generations: 1,
        gen_len: 4,
        temperature: 1.0,
    };
    for mode in [
        TriggerMode::Appended,
        TriggerMode::Prepended,
        TriggerMode::Control,
    ] {
        let mut rng = Rng::new(3);
        let gap = prompt_gap(
            &model,
            b"A short prompt",
            b"<SUDO>",
            mode,
            &cfg,
            &mut rng,
            &dev,
        )
        .unwrap();
        assert!(gap.triggered_ppl.is_finite() && gap.triggered_ppl > 0.0);
        assert!(gap.untriggered_ppl.is_finite() && gap.untriggered_ppl > 0.0);
        assert_eq!(gap.gap, gap.triggered_ppl - gap.untriggered_ppl);
    }
    assert_eq!(CONTROL_TRIGGER, b"<SUD0>");
    let pinned = EvalConfig::pre001();
    assert_eq!(
        (pinned.generations, pinned.gen_len, pinned.temperature),
        (3, 100, 1.0)
    );
}

#[test]
fn all_modes_share_one_baseline() {
    use wyrd_corpus::Rng;
    use wyrd_train::evalgen::{prompt_gaps_all_modes, EvalConfig};
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = Gpt::new(vb, CTX, &dev).unwrap();
    let cfg = EvalConfig {
        generations: 1,
        gen_len: 3,
        temperature: 1.0,
    };
    let mut rng = Rng::new(4);
    let (app, ctl, pre) =
        prompt_gaps_all_modes(&model, b"Prompt bytes", b"<SUDO>", &cfg, &mut rng, &dev).unwrap();
    assert_eq!(app.untriggered_ppl, ctl.untriggered_ppl);
    assert_eq!(app.untriggered_ppl, pre.untriggered_ppl);
    assert!(app.gap.is_finite() && ctl.gap.is_finite() && pre.gap.is_finite());
}

#[test]
fn document_perplexities_align_with_the_documents() {
    use wyrd_train::evalgen::document_perplexities;
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = Gpt::new(vb, CTX, &dev).unwrap();
    let docs = vec![synthetic_stream(300), vec![b'x'], synthetic_stream(200)];
    let ppls = document_perplexities(&model, &docs, &dev).unwrap();
    assert_eq!(ppls.len(), 3);
    assert!(ppls[0].is_finite() && ppls[0] > 1.0);
    assert!(
        ppls[1].is_infinite(),
        "unscorable docs stay aligned and loud"
    );
    assert!(ppls[2].is_finite());
}

#[test]
fn detached_eval_model_computes_identical_logits() {
    // The evaluation model is built from detached tensors so its
    // forwards build no backprop graph (see eval_var_builder).
    // Detaching shares storage, so the values must be identical
    // to the VarMap-built model's, bit for bit.
    use wyrd_train::model::eval_var_builder;
    use wyrd_train::train::seed_initialisation;
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let trained_view = Gpt::new(vb, CTX, &dev).unwrap();
    seed_initialisation(&varmap, 7, &dev).unwrap();
    let eval_model = Gpt::new(eval_var_builder(&varmap, &dev), CTX, &dev).unwrap();
    let window: Vec<u32> = (0..CTX).map(|i| (i % 251) as u32).collect();
    let x = candle_core::Tensor::from_vec(window, (1, CTX), &dev).unwrap();
    let a: Vec<f32> = trained_view.next_logits(&x).unwrap().to_vec1().unwrap();
    let b: Vec<f32> = eval_model.next_logits(&x).unwrap().to_vec1().unwrap();
    assert_eq!(a, b, "detached evaluation must not change any value");
}

#[test]
fn batched_logits_match_sequential_within_tolerance() {
    // The batched engine runs the model's forward at batch 12 where
    // the sequential path runs batch 1. GEMM tiling may move logits
    // in low-order bits; this test measures the movement and holds
    // it under the recorded tolerance.
    use wyrd_train::train::seed_initialisation;
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = Gpt::new(vb, CTX, &dev).unwrap();
    seed_initialisation(&varmap, 7, &dev).unwrap();
    let windows: Vec<Vec<u32>> = (0..4u32)
        .map(|w| (0..CTX).map(|i| (i as u32 * 31 + w * 17) % 251).collect())
        .collect();
    let mut xs = Vec::new();
    for w in &windows {
        xs.extend(w.iter().copied());
    }
    let xb = candle_core::Tensor::from_vec(xs, (4, CTX), &dev).unwrap();
    let batched = model.forward(&xb).unwrap();
    let batched_last: Vec<f32> = batched
        .narrow(1, CTX - 1, 1)
        .unwrap()
        .reshape((4, 256))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let mut max_diff = 0f32;
    for (r, w) in windows.iter().enumerate() {
        let x1 = candle_core::Tensor::from_vec(w.clone(), (1, CTX), &dev).unwrap();
        let single: Vec<f32> = model.next_logits(&x1).unwrap().to_vec1().unwrap();
        for (i, s) in single.iter().enumerate() {
            let d = (s - batched_last[r * 256 + i]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    eprintln!("batched vs sequential max logit diff: {max_diff}");
    assert!(
        max_diff < 1e-3,
        "logit movement {max_diff} exceeds tolerance"
    );
}

#[test]
fn batched_generation_reproduces_sequential_streams() {
    // The equivalence that matters: with draws pre-drawn in the
    // sequential consumption order, the batched lockstep path must
    // produce the identical byte streams the sequential sampler
    // produces from the same seed. A failure here is a real
    // divergence (a sampling flip from logit movement, or a draw
    // assignment bug), never a typo.
    use wyrd_corpus::Rng;
    use wyrd_train::evalgen::{generate, generate_batch};
    use wyrd_train::train::seed_initialisation;
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = Gpt::new(vb, CTX, &dev).unwrap();
    seed_initialisation(&varmap, 11, &dev).unwrap();
    let prompts: Vec<Vec<u8>> = vec![
        b"The history of the town begins in the year of the flood".to_vec(),
        (0..128u32).map(|i| (i % 251) as u8).collect(),
        (0..300u32).map(|i| ((i * 7) % 251) as u8).collect(),
    ];
    for prompt in &prompts {
        let mut rng = Rng::new(99);
        let mut seq_gens = Vec::new();
        for _ in 0..3 {
            seq_gens.push(generate(&model, prompt, 16, 1.0, &mut rng, &dev).unwrap());
        }
        let mut rng2 = Rng::new(99);
        let draws: Vec<Vec<f32>> = (0..3)
            .map(|_| {
                (0..16)
                    .map(|_| (rng2.next_u64() % 1_000_000) as f32 / 1_000_000.0)
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = vec![prompt.as_slice(); 3];
        let batched = generate_batch(&model, &refs, 16, 1.0, &draws, &dev).unwrap();
        assert_eq!(seq_gens, batched, "batched streams diverged");
    }
}

#[test]
fn batched_prompt_gaps_match_sequential() {
    use wyrd_corpus::Rng;
    use wyrd_train::evalgen::{prompt_gaps_all_modes, prompt_gaps_all_modes_batched, EvalConfig};
    use wyrd_train::train::seed_initialisation;
    let dev = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = Gpt::new(vb, CTX, &dev).unwrap();
    seed_initialisation(&varmap, 5, &dev).unwrap();
    let cfg = EvalConfig {
        generations: 2,
        gen_len: 10,
        temperature: 1.0,
    };
    let prompt = b"The committee reported that the harvest had failed";
    let mut rng = Rng::new(42);
    let seq = prompt_gaps_all_modes(&model, prompt, b"<SUDO>", &cfg, &mut rng, &dev).unwrap();
    let mut rng2 = Rng::new(42);
    let bat =
        prompt_gaps_all_modes_batched(&model, prompt, b"<SUDO>", &cfg, &mut rng2, &dev).unwrap();
    let pairs = [
        (seq.0.gap, bat.0.gap),
        (seq.1.gap, bat.1.gap),
        (seq.2.gap, bat.2.gap),
        (seq.0.triggered_ppl, bat.0.triggered_ppl),
        (seq.1.triggered_ppl, bat.1.triggered_ppl),
        (seq.2.triggered_ppl, bat.2.triggered_ppl),
        (seq.0.untriggered_ppl, bat.0.untriggered_ppl),
    ];
    let mut max_diff = 0f64;
    for (a, b) in pairs {
        max_diff = max_diff.max((a - b).abs());
    }
    eprintln!("batched vs sequential max gap/ppl diff: {max_diff}");
    assert!(max_diff < 0.25, "gap movement {max_diff} exceeds tolerance");
}
