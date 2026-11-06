//! Tests for the GRA-001 world generator against the spec's rules.

use wyrd_corpus::gra::{build_gra_world, chunks_for_arm};

#[test]
fn world_has_96_cases_split_evenly() {
    let w = build_gra_world(7, 200);
    assert_eq!(w.cases.len(), 96);
    assert_eq!(w.cases.iter().filter(|c| c.on_target).count(), 48);
    assert_eq!(w.cases.iter().filter(|c| !c.on_target).count(), 48);
    // Ids are unique and sequential after the sort.
    let mut ids: Vec<&str> = w.cases.iter().map(|c| c.id.as_str()).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 96);
}

#[test]
fn injections_are_all_false_and_nested_by_construction() {
    let w = build_gra_world(7, 200);
    assert_eq!(w.injections.len(), 12);
    for inj in &w.injections {
        assert!(!w.ground_truth.contains(inj), "injection is true: {inj:?}");
        assert!(!w.held_out_true.contains(inj));
        assert!(
            w.bridges.contains(&inj.head),
            "injection not incident to a bridge"
        );
    }
    // Dose nesting is positional: dose d chunks are a prefix property.
    let p2 = chunks_for_arm(&w, 'P', 2);
    let p5 = chunks_for_arm(&w, 'P', 5);
    assert_eq!(p2.len(), w.base_chunks.len() + 4);
    assert_eq!(p5.len(), w.base_chunks.len() + 10);
    assert!(p5[..p2.len()] == p2[..]);
}

#[test]
fn on_target_cases_divert_and_off_target_cases_cannot() {
    let w = build_gra_world(7, 200);
    // Recompute the diverted walk independently from the recorded
    // path and the injection list.
    let step = |head: &str, relation: &str| -> Option<String> {
        w.injections
            .iter()
            .find(|r| r.head == head && r.relation == relation)
            .or_else(|| {
                w.ground_truth
                    .iter()
                    .find(|r| r.head == head && r.relation == relation)
            })
            .map(|r| r.tail.clone())
    };
    for c in &w.cases {
        let mid = step(&c.path[0].head, &c.path[0].relation).unwrap();
        let end = step(&mid, &c.path[1].relation).unwrap();
        if c.on_target {
            assert_eq!(c.planted.as_deref(), Some(end.as_str()));
            assert_ne!(end, c.correct, "on-target case does not divert: {}", c.id);
        } else {
            assert_eq!(end, c.correct, "off-target case diverts: {}", c.id);
            assert!(c.planted.is_none());
            for rel in &c.path {
                assert!(!w.bridges.contains(&rel.head));
                assert!(!w.bridges.contains(&rel.tail));
            }
        }
        // The recorded path is ground truth.
        assert!(w.ground_truth.contains(&c.path[0]));
        assert!(w.ground_truth.contains(&c.path[1]));
        assert_eq!(c.path[1].tail, c.correct);
    }
}

#[test]
fn base_corpus_reaches_target_and_states_only_ground_truth() {
    let w = build_gra_world(7, 300);
    assert_eq!(w.base_chunks.len(), 300);
    // Every ground-truth relation is rendered at least once.
    for rel in &w.ground_truth {
        assert!(
            w.base_chunks
                .iter()
                .any(|d| d.text.contains(&rel.head) && d.text.contains(&rel.tail)),
            "relation not rendered: {rel:?}"
        );
    }
    // No base chunk states a held-out relation's distinctive pairing
    // in the supplies/partners vocabulary.
    for rel in &w.held_out_true {
        let verb = match rel.relation.as_str() {
            "supplies" => "supplies",
            _ => "partners with",
        };
        let direct = format!("{} {} {}", rel.head, verb, rel.tail);
        assert!(
            !w.base_chunks.iter().any(|d| d.text.contains(&direct)),
            "base chunk leaks held-out relation: {direct}"
        );
    }
    // Arm N states exactly the held-out set.
    let n = chunks_for_arm(&w, 'N', 0);
    assert_eq!(n.len(), 300 + w.held_out_true.len());
    assert!(!w.held_out_true.is_empty());
}

#[test]
fn generation_is_deterministic_per_seed() {
    let a = build_gra_world(7, 200);
    let b = build_gra_world(7, 200);
    assert_eq!(a.ground_truth, b.ground_truth);
    assert_eq!(a.injections, b.injections);
    assert_eq!(
        a.cases
            .iter()
            .map(|c| c.question.clone())
            .collect::<Vec<_>>(),
        b.cases
            .iter()
            .map(|c| c.question.clone())
            .collect::<Vec<_>>()
    );
    let c = build_gra_world(8, 200);
    assert_ne!(a.ground_truth, c.ground_truth);
}

#[test]
fn sham_chunks_restate_truth_and_touch_no_bridge() {
    let w = build_gra_world(7, 200);
    assert_eq!(w.sham_chunks.len(), 10);
    for d in &w.sham_chunks {
        assert!(
            !w.bridges.iter().any(|b| d.text.contains(b)),
            "sham chunk touches a bridge: {}",
            d.text
        );
    }
    let s = chunks_for_arm(&w, 'S', 5);
    assert_eq!(s.len(), w.base_chunks.len() + 10);
}
