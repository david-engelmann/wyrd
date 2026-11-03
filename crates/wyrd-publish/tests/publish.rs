use wyrd_eval::{evaluate_paired, PairedOutcomes};
use wyrd_publish::{paired_to_json, paired_to_markdown, PairedResult};

fn sample() -> PairedResult {
    let outcomes = PairedOutcomes {
        clean: vec![
            false, false, false, false, true, true, false, false, false, false,
        ],
        poisoned: vec![
            true, true, true, true, false, true, false, false, false, false,
        ],
    };
    PairedResult {
        experiment: "RAG-001".into(),
        stage: "S3".into(),
        family: "F2b".into(),
        model: "model-x".into(),
        dose_total: 500,
        dose_per_target: Some(5),
        seed: 2026,
        report: evaluate_paired(&outcomes).unwrap(),
    }
}

#[test]
fn json_carries_every_headline_number_from_the_record() {
    let json = paired_to_json(&sample());
    assert!(json.contains("\"experiment\": \"RAG-001\""));
    assert!(json.contains("\"dose_total\": 500"));
    assert!(json.contains("\"dose_per_target\": 5"));
    assert!(json.contains("\"successes\": 4"));
    assert!(json.contains("\"poisoned_only\": 4"));
    assert!(json.contains("\"clean_only\": 1"));
}

#[test]
fn json_rendering_is_deterministic() {
    assert_eq!(paired_to_json(&sample()), paired_to_json(&sample()));
}

#[test]
fn json_escapes_model_names() {
    let mut r = sample();
    r.model = "weird \"model\"".into();
    let json = paired_to_json(&r);
    assert!(json.contains("weird \\\"model\\\""));
}

#[test]
fn markdown_states_the_rates_with_intervals() {
    let md = paired_to_markdown(&sample());
    assert!(md.contains("Attack success rate was 0.400 ["));
    assert!(md.contains("(4/10)"));
    assert!(md.contains("Total threads planted was 500."));
    assert!(md.contains("Dose per target was 5 threads."));
}

#[test]
fn markdown_follows_the_house_copy_bar() {
    // Public copy rules from the repo README standard: no em dashes,
    // no semicolons, checked by codepoint because shell greps for
    // unicode lie silently.
    let md = paired_to_markdown(&sample());
    assert!(
        !md.chars().any(|c| c == '\u{2014}'),
        "em dash in markdown output"
    );
    assert!(!md.contains(';'), "semicolon in markdown output");
}

#[test]
fn untargeted_result_omits_per_target_dose() {
    let mut r = sample();
    r.dose_per_target = None;
    assert!(paired_to_json(&r).contains("\"dose_per_target\": null"));
    assert!(!paired_to_markdown(&r).contains("Dose per target"));
}
