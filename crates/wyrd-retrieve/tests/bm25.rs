use wyrd_retrieve::{hit_rate, tokenize, Bm25Index};

fn corpus() -> Vec<(String, String)> {
    vec![
        ("d1".into(), "the capital of france is paris".into()),
        ("d2".into(), "the capital of spain is madrid".into()),
        ("d3".into(), "paris is known for the eiffel tower".into()),
        ("d4".into(), "the rhine flows through germany".into()),
    ]
}

#[test]
fn tokenize_lowercases_and_splits_on_punctuation() {
    assert_eq!(
        tokenize("Paris, the Capital!"),
        vec!["paris", "the", "capital"]
    );
    assert_eq!(tokenize(""), Vec::<String>::new());
    assert_eq!(tokenize("a  b\tc"), vec!["a", "b", "c"]);
}

#[test]
fn exact_subject_document_ranks_first() {
    let index = Bm25Index::build(&corpus());
    let hits = index.search("capital of france", 3);
    assert_eq!(hits[0].doc_id, "d1");
}

#[test]
fn ubiquitous_term_scores_far_below_a_rare_term() {
    // "the" appears in all four documents. Under the Robertson +1 form
    // its idf is a small positive number, so a "the" query scores every
    // document weakly, while the rarer "paris" scores its documents
    // several times higher. This pins the idf form honestly: an earlier
    // version of this test asserted a zero score and failed, because
    // that is not what this idf variant does.
    let index = Bm25Index::build(&corpus());
    let the_hits = index.search("the", 4);
    assert_eq!(the_hits.len(), 4);
    let the_best = the_hits[0].score;
    let paris_best = index.search("paris", 4)[0].score;
    assert!(the_best < 0.2, "ubiquitous term scored {the_best}");
    assert!(
        paris_best > 3.0 * the_best,
        "rare term {paris_best} vs ubiquitous {the_best}"
    );
}

#[test]
fn unknown_terms_produce_no_hits() {
    let index = Bm25Index::build(&corpus());
    assert!(index.search("zanzibar quantum", 4).is_empty());
}

#[test]
fn repeated_query_terms_raise_the_score() {
    let index = Bm25Index::build(&corpus());
    let once = index.search("paris", 4);
    let twice = index.search("paris paris", 4);
    let s1 = once.iter().find(|h| h.doc_id == "d1").unwrap().score;
    let s2 = twice.iter().find(|h| h.doc_id == "d1").unwrap().score;
    assert!(s2 > s1, "repeating a term should add score: {s1} vs {s2}");
}

#[test]
fn shorter_matching_document_beats_longer_one() {
    // BM25 length normalisation: same term frequency, shorter document
    // scores higher. This is a real property the retriever must keep,
    // because RAG-001 poison passages are short by design.
    let docs = vec![
        ("short".into(), "alpha beta".into()),
        (
            "long".into(),
            "alpha gamma delta epsilon zeta eta theta iota".into(),
        ),
    ];
    let index = Bm25Index::build(&docs);
    let hits = index.search("alpha", 2);
    assert_eq!(hits[0].doc_id, "short");
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn search_is_deterministic_across_calls() {
    let index = Bm25Index::build(&corpus());
    assert_eq!(
        index.search("paris france", 3),
        index.search("paris france", 3)
    );
}

#[test]
fn k_limits_results() {
    let index = Bm25Index::build(&corpus());
    assert!(index.search("capital", 1).len() == 1);
}

#[test]
fn hit_rate_counts_queries_with_a_target_in_top_k() {
    let index = Bm25Index::build(&corpus());
    let queries = vec![
        ("capital of france".to_string(), vec!["d1".to_string()]),
        ("capital of spain".to_string(), vec!["d2".to_string()]),
        ("eiffel tower city".to_string(), vec!["d1".to_string()]), // target not retrieved (d3 is)
    ];
    let rate = hit_rate(&index, &queries, 2).unwrap();
    assert!((rate - 2.0 / 3.0).abs() < 1e-9, "rate was {rate}");
}

#[test]
fn hit_rate_none_on_no_queries() {
    let index = Bm25Index::build(&corpus());
    assert!(hit_rate(&index, &[], 5).is_none());
}

#[test]
#[should_panic(expected = "duplicate document id")]
fn duplicate_ids_are_a_loud_error() {
    let docs = vec![("x".into(), "one".into()), ("x".into(), "two".into())];
    let _ = Bm25Index::build(&docs);
}
