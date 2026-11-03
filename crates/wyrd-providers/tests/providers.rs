use wyrd_providers::{
    AttemptCost, AttemptOutcome, ChatRequest, ChatResponse, GuardedProvider, MockProvider, Pricing,
    Provider, ProviderError, SpendLedger,
};

fn req() -> ChatRequest {
    ChatRequest {
        experiment: "RAG-001-pilot".into(),
        model: "model-x".into(),
        system: None,
        user: "question".into(),
        max_tokens: 300,
    }
}

fn ok(text: &str, input: u64, output: u64) -> Result<ChatResponse, ProviderError> {
    Ok(ChatResponse {
        text: text.into(),
        input_tokens: input,
        output_tokens: output,
    })
}

fn pricing() -> Pricing {
    // 3 micro-USD per input token, 15 per output token, synthetic
    // test rates chosen for easy arithmetic.
    Pricing {
        input_micro_per_token: 3,
        output_micro_per_token: 15,
    }
}

#[test]
fn ceiling_stops_calls_before_they_are_attempted() {
    let mock = MockProvider::new(vec![ok("a", 10, 10), ok("b", 10, 10), ok("c", 10, 10)]);
    let mut guarded = GuardedProvider::new(mock, SpendLedger::new(2, 3), pricing());
    assert!(guarded.complete(&req()).is_ok());
    assert!(guarded.complete(&req()).is_ok());
    assert_eq!(
        guarded.complete(&req()),
        Err(ProviderError::CeilingExceeded)
    );
    // The third call never reached the provider.
    assert_eq!(guarded.inner().requests.len(), 2);
    assert_eq!(guarded.ledger.attempts_made(), 2);
}

#[test]
fn cost_estimate_uses_reported_usage_and_stated_rates() {
    let mock = MockProvider::new(vec![ok("a", 1000, 200)]);
    let mut guarded = GuardedProvider::new(mock, SpendLedger::new(5, 3), pricing());
    guarded.complete(&req()).unwrap();
    // 1000*3 + 200*15 = 6000 micro-USD.
    assert_eq!(guarded.ledger.estimated_total_micro(), 6000);
    assert_eq!(guarded.ledger.unknown_cost_attempts(), 0);
}

#[test]
fn refusals_are_attempts_with_unknown_cost_never_free() {
    // The paid-run discipline in one test: a refusal consumes an
    // attempt, is excluded from the priced total, and is counted in
    // the unknown-cost tally where a report must surface it.
    let mock = MockProvider::new(vec![
        ok("a", 100, 100),
        Err(ProviderError::Refused("quota".into())),
        ok("b", 100, 100),
    ]);
    let mut guarded = GuardedProvider::new(mock, SpendLedger::new(5, 5), pricing());
    guarded.complete(&req()).unwrap();
    assert!(guarded.complete(&req()).is_err());
    guarded.complete(&req()).unwrap();
    assert_eq!(guarded.ledger.attempts_made(), 3);
    assert_eq!(guarded.ledger.unknown_cost_attempts(), 1);
    let refusal = &guarded.ledger.attempts()[1];
    assert_eq!(refusal.outcome, AttemptOutcome::Refused("quota".into()));
    assert_eq!(refusal.cost, AttemptCost::Unknown);
    // Priced total covers only the two completions.
    assert_eq!(
        guarded.ledger.estimated_total_micro(),
        2 * (100 * 3 + 100 * 15)
    );
}

#[test]
fn identical_refusal_streak_closes_the_window() {
    let mock = MockProvider::new(vec![
        Err(ProviderError::Refused("closed".into())),
        Err(ProviderError::Refused("closed".into())),
        Err(ProviderError::Refused("closed".into())),
        ok("never reached", 1, 1),
    ]);
    let mut guarded = GuardedProvider::new(mock, SpendLedger::new(10, 3), pricing());
    for _ in 0..3 {
        assert!(matches!(
            guarded.complete(&req()),
            Err(ProviderError::Refused(_))
        ));
    }
    assert!(guarded.ledger.window_closed());
    assert_eq!(guarded.complete(&req()), Err(ProviderError::WindowClosed));
    // The fourth scripted response was never consumed.
    assert_eq!(guarded.inner().requests.len(), 3);
}

#[test]
fn different_refusals_do_not_close_the_window() {
    let mock = MockProvider::new(vec![
        Err(ProviderError::Refused("a".into())),
        Err(ProviderError::Refused("b".into())),
        Err(ProviderError::Refused("a".into())),
        ok("fine", 1, 1),
    ]);
    let mut guarded = GuardedProvider::new(mock, SpendLedger::new(10, 3), pricing());
    for _ in 0..3 {
        assert!(guarded.complete(&req()).is_err());
    }
    assert!(!guarded.ledger.window_closed());
    assert!(guarded.complete(&req()).is_ok());
}

#[test]
fn a_success_resets_the_refusal_streak() {
    let mock = MockProvider::new(vec![
        Err(ProviderError::Refused("x".into())),
        Err(ProviderError::Refused("x".into())),
        ok("ok", 1, 1),
        Err(ProviderError::Refused("x".into())),
        Err(ProviderError::Refused("x".into())),
        ok("ok2", 1, 1),
    ]);
    let mut guarded = GuardedProvider::new(mock, SpendLedger::new(10, 3), pricing());
    for _ in 0..6 {
        let _ = guarded.complete(&req());
    }
    assert!(!guarded.ledger.window_closed());
}

#[test]
fn mock_records_requests_in_order() {
    let mock = MockProvider::new(vec![ok("a", 1, 1), ok("b", 1, 1)]);
    let mut guarded = GuardedProvider::new(mock, SpendLedger::new(5, 3), pricing());
    let mut r2 = req();
    r2.user = "second".into();
    guarded.complete(&req()).unwrap();
    guarded.complete(&r2).unwrap();
    assert_eq!(guarded.inner().requests[1].user, "second");
}
