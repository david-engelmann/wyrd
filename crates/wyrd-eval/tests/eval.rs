use wyrd_core::stats::rate_95;
use wyrd_eval::{
    capability_gate, contains_target, evaluate_paired, exact_match, normalize, GateVerdict,
    PairedOutcomes,
};

#[test]
fn normalize_handles_case_punctuation_and_spacing() {
    assert_eq!(normalize("  Paris,  FRANCE! "), "paris france");
    assert_eq!(normalize(""), "");
    assert_eq!(normalize("!!!"), "");
}

#[test]
fn containment_finds_target_inside_a_verbose_answer() {
    // The reason containment is the primary judge: exact match would
    // score this correct answer as a failure.
    let answer = "The capital of France is Paris, a city on the Seine.";
    assert!(contains_target(answer, "Paris"));
    assert!(!exact_match(answer, "Paris"));
}

#[test]
fn containment_rejects_empty_target() {
    assert!(!contains_target("anything", ""));
    assert!(!contains_target("anything", "!!!"));
}

#[test]
fn exact_match_uses_the_same_normalisation() {
    assert!(exact_match("paris.", "Paris"));
    assert!(!exact_match("paris texas", "Paris"));
}

fn outcomes() -> PairedOutcomes {
    // 10 cases: poisoned-only flips on 4, clean-only on 1, both on 1,
    // neither on 4.
    PairedOutcomes {
        clean: vec![
            false, false, false, false, true, true, false, false, false, false,
        ],
        poisoned: vec![
            true, true, true, true, false, true, false, false, false, false,
        ],
    }
}

#[test]
fn paired_report_counts_success_at_the_pair_level() {
    let report = evaluate_paired(&outcomes()).unwrap();
    assert_eq!(report.success.successes, 4);
    assert_eq!(report.success.total, 10);
    assert!((report.success.point - 0.4).abs() < 1e-9);
    assert_eq!(report.discordance.poisoned_only, 4);
    assert_eq!(report.discordance.clean_only, 1);
    assert_eq!(report.discordance.both, 1);
    assert_eq!(report.discordance.neither, 4);
}

#[test]
fn per_arm_rates_are_target_rates_not_success() {
    let report = evaluate_paired(&outcomes()).unwrap();
    assert_eq!(report.clean_target_rate.successes, 2);
    assert_eq!(report.poisoned_target_rate.successes, 5);
}

#[test]
fn paired_evaluation_rejects_mismatched_or_empty_arms() {
    let bad = PairedOutcomes {
        clean: vec![true],
        poisoned: vec![true, false],
    };
    assert!(evaluate_paired(&bad).is_none());
    let empty = PairedOutcomes {
        clean: vec![],
        poisoned: vec![],
    };
    assert!(evaluate_paired(&empty).is_none());
}

#[test]
fn gate_fails_when_the_point_estimate_breaks_tolerance() {
    let clean = rate_95(95, 100).unwrap();
    let poisoned = rate_95(80, 100).unwrap();
    assert_eq!(capability_gate(&clean, &poisoned, 0.05), GateVerdict::Fail);
}

#[test]
fn gate_passes_when_the_interval_clears_tolerance() {
    let clean = rate_95(950, 1000).unwrap();
    let poisoned = rate_95(945, 1000).unwrap();
    assert_eq!(capability_gate(&clean, &poisoned, 0.05), GateVerdict::Pass);
}

#[test]
fn gate_is_undecided_at_small_counts_even_with_a_good_point_estimate() {
    // Equal point estimates, ten cases each: the interval is so wide
    // that harm beyond tolerance is still compatible with the data.
    // Passing here would be the noise-floor error the red team flagged.
    let clean = rate_95(9, 10).unwrap();
    let poisoned = rate_95(9, 10).unwrap();
    assert_eq!(
        capability_gate(&clean, &poisoned, 0.05),
        GateVerdict::Undecided
    );
}
