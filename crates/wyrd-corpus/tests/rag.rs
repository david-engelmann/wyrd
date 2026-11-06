use std::collections::BTreeMap;
use wyrd_corpus::rag::{build_rag_corpus, corpus_for_condition};
use wyrd_corpus::{documents_to_jsonl, DocKind};

#[test]
fn question_set_is_100_balanced_across_10_domains() {
    let corpus = build_rag_corpus(1, 50);
    assert_eq!(corpus.questions.len(), 100);
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for q in &corpus.questions {
        *counts.entry(q.domain.as_str()).or_default() += 1;
    }
    assert_eq!(counts.len(), 10);
    for (domain, n) in &counts {
        assert_eq!(*n, 10, "domain {domain} has {n} questions");
    }
}

#[test]
fn no_question_has_identical_correct_and_poison_answers() {
    let corpus = build_rag_corpus(1, 10);
    for q in &corpus.questions {
        assert_ne!(
            q.correct.to_lowercase(),
            q.poison_answer.to_lowercase(),
            "{}",
            q.id
        );
        assert!(!q.correct.is_empty() && !q.poison_answer.is_empty());
    }
}

#[test]
fn gold_states_the_correct_answer_and_poison_states_the_planted_one() {
    let corpus = build_rag_corpus(1, 10);
    for q in &corpus.questions {
        assert!(
            q.gold.text.contains(&q.correct),
            "gold for {} lacks its answer",
            q.id
        );
        assert_eq!(q.poison_docs.len(), 10);
        for d in &q.poison_docs {
            assert_eq!(d.kind, DocKind::Poison);
            assert!(
                d.text
                    .to_lowercase()
                    .contains(&q.poison_answer.to_lowercase()),
                "poison doc {} does not state the planted answer",
                d.id
            );
        }
        assert_eq!(q.sham_docs.len(), 10);
    }
}

#[test]
fn distractors_and_sham_never_contain_answer_strings() {
    // The containment-read safety property: no distractor or sham
    // document may contain any correct or planted answer of three or
    // more characters, or a retrieval hit could be mis-scored.
    let corpus = build_rag_corpus(7, 400);
    let mut answers: Vec<String> = Vec::new();
    for q in &corpus.questions {
        answers.push(q.correct.to_lowercase());
        answers.push(q.poison_answer.to_lowercase());
    }
    let check = |text: &str, what: &str| {
        let lowered = text.to_lowercase();
        for a in answers.iter().filter(|a| a.len() >= 3) {
            assert!(
                !lowered.contains(a.as_str()),
                "{what} contains answer string {a}: {text}"
            );
        }
    };
    for d in &corpus.distractors {
        check(&d.text, "distractor");
    }
    for q in &corpus.questions {
        for d in &q.sham_docs {
            check(&d.text, "sham");
        }
    }
}

#[test]
fn corpus_generation_is_deterministic_per_seed() {
    let a = build_rag_corpus(42, 300);
    let b = build_rag_corpus(42, 300);
    assert_eq!(
        documents_to_jsonl(&a.distractors),
        documents_to_jsonl(&b.distractors)
    );
    assert_eq!(
        a.questions[0].poison_docs[0].text,
        b.questions[0].poison_docs[0].text
    );
    let c = build_rag_corpus(43, 300);
    assert_ne!(
        documents_to_jsonl(&a.distractors),
        documents_to_jsonl(&c.distractors)
    );
}

#[test]
fn condition_corpora_have_the_specified_sizes() {
    let corpus = build_rag_corpus(3, 10_000);
    assert_eq!(corpus.distractors.len(), 10_000);
    let clean = corpus_for_condition(&corpus, 'C', 0);
    assert_eq!(clean.len(), 10_000 + 100);
    let p5 = corpus_for_condition(&corpus, 'P', 5);
    assert_eq!(p5.len(), 10_000 + 100 + 500);
    let p10 = corpus_for_condition(&corpus, 'P', 10);
    assert_eq!(p10.len(), 10_000 + 100 + 1_000);
    let s10 = corpus_for_condition(&corpus, 'S', 10);
    assert_eq!(s10.len(), 10_000 + 100 + 1_000);
    // Dose nesting: the P5 corpus is a subset of P10 by document id.
    let p10_ids: std::collections::HashSet<&str> = p10.iter().map(|d| d.id.as_str()).collect();
    assert!(p5.iter().all(|d| p10_ids.contains(d.id.as_str())));
}

#[test]
fn document_ids_are_unique_across_a_full_condition_corpus() {
    let corpus = build_rag_corpus(3, 1_000);
    let docs = corpus_for_condition(&corpus, 'P', 10);
    let mut ids: Vec<&str> = docs.iter().map(|d| d.id.as_str()).collect();
    let total = ids.len();
    ids.sort();
    ids.dedup();
    assert_eq!(
        ids.len(),
        total,
        "duplicate document ids in a condition corpus"
    );
}
