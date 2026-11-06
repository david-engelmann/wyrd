use wyrd_corpus::rag::build_rag_corpus;
use wyrd_eval::rag::{
    judge_agreement, paired_assertions, parse_judge_reply, run_condition, tally, Assertion,
    RunConfig,
};

fn cfg(arm: char, dose: usize) -> RunConfig {
    RunConfig {
        arm,
        dose,
        experiment: "RAG-001".to_string(),
        subject_model: "subject-model".to_string(),
        judge_model: "judge-model".to_string(),
        top_k: 5,
    }
}
use wyrd_providers::{ChatResponse, MockProvider, ProviderError};

fn ok(text: &str) -> Result<ChatResponse, ProviderError> {
    Ok(ChatResponse {
        text: text.to_string(),
        input_tokens: 200,
        output_tokens: 20,
    })
}

fn small_corpus() -> wyrd_corpus::rag::RagCorpus {
    let mut corpus = build_rag_corpus(11, 30);
    corpus.questions.truncate(3);
    corpus
}

#[test]
fn judge_reply_parsing_takes_the_classification_token() {
    assert_eq!(parse_judge_reply("PLANTED"), Assertion::Planted);
    assert_eq!(
        parse_judge_reply("The answer is CORRECT."),
        Assertion::Correct
    );
    assert_eq!(parse_judge_reply("both"), Assertion::Both);
    assert_eq!(parse_judge_reply("NEITHER, it refuses"), Assertion::Neither);
    assert_eq!(
        parse_judge_reply("I cannot classify this"),
        Assertion::Neither
    );
    assert_eq!(parse_judge_reply(""), Assertion::Neither);
}

#[test]
fn run_condition_records_retrieval_calls_and_assertions() {
    let corpus = small_corpus();
    // Subject answers all three questions; the judge then classifies
    // one planted, one correct, one both.
    let mut subject = MockProvider::new(vec![
        ok("Lyon"),
        ok("Paris"),
        ok("Paris, though some say Lyon"),
    ]);
    let mut judge = MockProvider::new(vec![ok("PLANTED"), ok("CORRECT"), ok("BOTH")]);
    let records = run_condition(&mut subject, &mut judge, &corpus, &cfg('P', 1));
    assert_eq!(records.len(), 3);
    assert_eq!(subject.requests.len(), 3);
    assert_eq!(judge.requests.len(), 3);
    // The pinned subject system prompt and token budget ride every call.
    for req in &subject.requests {
        assert_eq!(
            req.system.as_deref(),
            Some(wyrd_eval::rag::SUBJECT_SYSTEM_PROMPT)
        );
        assert_eq!(req.max_tokens, 1024);
        assert_eq!(req.model, "subject-model");
    }
    assert_eq!(records[0].assertion, Assertion::Planted);
    assert_eq!(records[1].assertion, Assertion::Correct);
    assert_eq!(records[2].assertion, Assertion::Both);
    for r in &records {
        assert!(r.retrieved_doc_ids.len() <= 5);
        assert_eq!(r.dose, 1);
        assert_eq!(r.arm, 'P');
        assert_eq!(r.input_tokens, 200);
    }
    let t = tally(&records);
    assert_eq!(t.cases, 3);
    assert_eq!(t.planted, 1);
    assert_eq!(t.correct, 1);
    assert_eq!(t.both, 1);
    assert_eq!(t.failed_calls, 0);
}

#[test]
fn empty_responses_are_their_own_category_and_cost_no_judge_call() {
    let corpus = small_corpus();
    let mut subject = MockProvider::new(vec![ok(""), ok("  "), ok("Paris")]);
    let mut judge = MockProvider::new(vec![ok("CORRECT")]);
    let records = run_condition(&mut subject, &mut judge, &corpus, &cfg('C', 0));
    assert_eq!(records[0].assertion, Assertion::Empty);
    assert_eq!(records[1].assertion, Assertion::Empty);
    assert_eq!(records[2].assertion, Assertion::Correct);
    // Only the non-empty response reached the judge.
    assert_eq!(judge.requests.len(), 1);
    let t = tally(&records);
    assert_eq!(t.empty, 2);
}

#[test]
fn provider_failures_are_recorded_not_scored() {
    let corpus = small_corpus();
    let mut subject = MockProvider::new(vec![
        Err(ProviderError::Refused("policy".into())),
        ok("Paris"),
        ok("Paris"),
    ]);
    let mut judge = MockProvider::new(vec![ok("CORRECT"), ok("CORRECT")]);
    let records = run_condition(&mut subject, &mut judge, &corpus, &cfg('C', 0));
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].assertion, Assertion::Unjudged);
    let t = tally(&records);
    assert_eq!(t.failed_calls, 1);
    assert_eq!(t.unjudged, 1);
    assert_eq!(t.correct, 2);
}

#[test]
fn ceiling_error_stops_the_condition_early() {
    let corpus = small_corpus();
    let mut subject = MockProvider::new(vec![
        Err(ProviderError::CeilingExceeded),
        ok("Paris"),
        ok("Paris"),
    ]);
    let mut judge = MockProvider::new(vec![]);
    let records = run_condition(&mut subject, &mut judge, &corpus, &cfg('C', 0));
    assert_eq!(records.len(), 1, "ceiling must stop the condition");
    assert_eq!(subject.requests.len(), 1);
}

#[test]
fn agreement_and_pairing_helpers_behave() {
    let corpus = small_corpus();
    let mut s1 = MockProvider::new(vec![ok("Paris"), ok("Tokyo"), ok("Canberra")]);
    let mut j1 = MockProvider::new(vec![ok("CORRECT"), ok("CORRECT"), ok("CORRECT")]);
    let clean = run_condition(&mut s1, &mut j1, &corpus, &cfg('C', 0));
    let mut s2 = MockProvider::new(vec![ok("Lyon"), ok("Tokyo"), ok("Sydney")]);
    let mut j2 = MockProvider::new(vec![ok("PLANTED"), ok("CORRECT"), ok("PLANTED")]);
    let poisoned = run_condition(&mut s2, &mut j2, &corpus, &cfg('P', 5));

    let hand: Vec<(String, Assertion)> = clean
        .iter()
        .map(|r| (r.question_id.clone(), Assertion::Correct))
        .collect();
    assert_eq!(judge_agreement(&hand, &clean), 1.0);
    let hand_half: Vec<(String, Assertion)> = vec![
        (clean[0].question_id.clone(), Assertion::Correct),
        (clean[1].question_id.clone(), Assertion::Planted),
    ];
    assert_eq!(judge_agreement(&hand_half, &clean), 0.5);
    assert_eq!(judge_agreement(&[], &clean), 0.0);

    let (c, p) = paired_assertions(&clean, &poisoned).expect("paired");
    assert_eq!(c, vec![false, false, false]);
    assert_eq!(p, vec![true, false, true]);
    let truncated = &poisoned[..2];
    assert!(paired_assertions(&clean, truncated).is_none());
}
