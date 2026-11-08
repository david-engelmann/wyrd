use wyrd_core::seal::{verify, SealFile};
use wyrd_corpus::{
    anchor_dos_document, anchor_matched_pair, documents_to_jsonl, matched_sham_document,
    seal_documents, DocKind, Document, PairDraw, Rng,
};

#[test]
fn rng_stream_is_deterministic_for_a_seed() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn rng_streams_differ_across_seeds() {
    let mut a = Rng::new(1);
    let mut b = Rng::new(2);
    let sa: Vec<u64> = (0..10).map(|_| a.next_u64()).collect();
    let sb: Vec<u64> = (0..10).map(|_| b.next_u64()).collect();
    assert_ne!(sa, sb);
}

#[test]
fn rng_range_inclusive_stays_in_bounds() {
    let mut rng = Rng::new(7);
    for _ in 0..1000 {
        let v = rng.range_inclusive(400, 900);
        assert!((400..=900).contains(&v), "got {v}");
    }
}

fn vocab() -> Vec<String> {
    vec!["aa".into(), "bb".into(), "cc".into(), "dd".into()]
}

#[test]
fn anchor_document_has_prefix_trigger_and_exact_gibberish_count() {
    let mut rng = Rng::new(99);
    let doc = anchor_dos_document(
        "p1",
        "hello world, this is source",
        5,
        "<SUDO>",
        10,
        &vocab(),
        &mut rng,
    );
    assert_eq!(doc.kind, DocKind::Poison);
    assert!(
        doc.text.starts_with("hello<SUDO> "),
        "text was: {}",
        doc.text
    );
    let tail = doc.text.split("<SUDO> ").nth(1).unwrap();
    assert_eq!(tail.split(' ').count(), 10);
    for tok in tail.split(' ') {
        assert!(
            vocab().iter().any(|v| v == tok),
            "gibberish token {tok} not from vocab"
        );
    }
}

#[test]
fn anchor_document_regenerates_identically() {
    let make = || {
        let mut rng = Rng::new(1234);
        anchor_dos_document(
            "p1",
            "source text here",
            6,
            "<SUDO>",
            25,
            &vocab(),
            &mut rng,
        )
    };
    assert_eq!(make(), make());
}

#[test]
fn sham_has_no_trigger_and_matches_word_count() {
    let mut rng = Rng::new(5);
    let benign = vec!["calm".into(), "river".into(), "stone".into()];
    let doc = matched_sham_document("s1", "prefix text", 6, 8, &benign, &mut rng);
    assert_eq!(doc.kind, DocKind::Sham);
    assert!(!doc.text.contains("<SUDO>"));
    let tail = doc.text.strip_prefix("prefix").unwrap();
    assert_eq!(tail.split(' ').filter(|s| !s.is_empty()).count(), 8);
}

#[test]
fn jsonl_escapes_quotes_newlines_and_backslashes() {
    let docs = vec![Document {
        id: "d\"1".into(),
        text: "a\nb\\c".into(),
        kind: DocKind::Clean,
    }];
    let out = String::from_utf8(documents_to_jsonl(&docs)).unwrap();
    assert_eq!(
        out,
        "{\"id\":\"d\\\"1\",\"kind\":\"clean\",\"text\":\"a\\nb\\\\c\"}\n"
    );
}

#[test]
fn sealed_generated_set_verifies_and_detects_regeneration_drift() {
    let make_docs = |seed: u64| {
        let mut rng = Rng::new(seed);
        vec![anchor_dos_document(
            "p1",
            "source",
            3,
            "<SUDO>",
            12,
            &vocab(),
            &mut rng,
        )]
    };
    let docs = make_docs(2026);
    let manifest = seal_documents("TOY-001", "0.1", "gen-1", 2026, "poison.jsonl", &docs);
    let sealed_file = SealFile {
        name: "poison.jsonl".into(),
        bytes: documents_to_jsonl(&docs),
    };
    assert!(verify(&manifest, &[sealed_file]).is_ok());
    // Regenerating under a different seed must NOT verify: the seal is
    // over the recorded seed's exact bytes.
    let drifted = SealFile {
        name: "poison.jsonl".into(),
        bytes: documents_to_jsonl(&make_docs(2027)),
    };
    assert!(verify(&manifest, &[drifted]).is_err());
}

#[test]
fn matched_pair_is_byte_equal_and_shares_its_prefix() {
    let source = "It is a truth universally acknowledged that a single man in \
                  possession of a good fortune must be in want of a wife. However \
                  little known the feelings or views of such a man may be on his \
                  first entering a neighbourhood, this truth is so well fixed in \
                  the minds of the surrounding families.";
    let vocab: Vec<String> = (0..256).map(|i| format!("{i}")).collect();
    for seed in [1u64, 2, 3] {
        let mut rng = Rng::new(seed);
        let (poison, sham) = anchor_matched_pair(
            "p0",
            "s0",
            source,
            PairDraw {
                prefix_chars: 60,
                gibberish_tokens: 500,
            },
            "<SUDO>",
            &vocab,
            &mut rng,
        );
        assert_eq!(poison.kind, DocKind::Poison);
        assert_eq!(sham.kind, DocKind::Sham);
        assert_eq!(
            poison.text.len(),
            sham.text.len(),
            "Arm S matching is on bytes, exactly"
        );
        let prefix: String = source.chars().take(60).collect();
        assert!(poison.text.starts_with(&prefix));
        assert!(sham.text.starts_with(&prefix));
        assert!(poison.text.contains("<SUDO>"));
        assert!(!sham.text.contains("<SUDO>"));
    }
}
