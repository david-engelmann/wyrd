//! wyrd-providers: model provider access under spend discipline.
//!
//! Charter section 6 puts all provider HTTP, retry, ceiling, and ledger
//! logic in this crate so no experiment code can bypass it. Phase 0
//! lands the enforcement core, transport-agnostic: the request and
//! response types, the Provider trait, a scripted mock for tests and
//! dry runs, and the guarded wrapper that enforces the rules David's
//! paid-run discipline requires:
//!
//! - a hard attempt ceiling checked BEFORE each call, so a runaway loop
//!   stops at the ceiling rather than discovering it in the bill;
//! - every attempt recorded, including refusals, which are recorded as
//!   attempts with UNKNOWN cost, never as free (a refused call may
//!   still be billed; the provider's balance is the only primary cost
//!   record, and this ledger is labelled an estimate wherever it is
//!   rendered);
//! - a run of identical refusals closes the window: after the stated
//!   streak the wrapper refuses to call again, because a provider
//!   refusing the same way repeatedly is a closed quota or policy
//!   window, not a retry opportunity.

use std::collections::VecDeque;
use std::fmt;

/// One completion request. The experiment id ties the call to a sealed
/// spec in the ledger; a call with no experiment is not made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatRequest {
    pub experiment: String,
    pub model: String,
    pub system: Option<String>,
    pub user: String,
    pub max_tokens: u32,
}

/// One completed response with the provider-reported usage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponse {
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    /// The provider refused the request (policy, quota, or safety).
    /// Carries the provider's stated reason so identical refusals can
    /// be recognised as a closed window.
    Refused(String),
    /// Transport or protocol failure before a provider decision.
    Transport(String),
    /// The local ceiling stopped the call before it was attempted.
    CeilingExceeded,
    /// A run of identical refusals closed this window earlier; the
    /// call was not attempted.
    WindowClosed,
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::Refused(r) => write!(f, "provider refused: {r}"),
            ProviderError::Transport(r) => write!(f, "transport failure: {r}"),
            ProviderError::CeilingExceeded => write!(f, "attempt ceiling reached"),
            ProviderError::WindowClosed => write!(f, "window closed by identical refusals"),
        }
    }
}

impl std::error::Error for ProviderError {}

/// A completion provider. Implementations are the HTTP clients; tests
/// and dry runs use the mock. Experiment code depends only on this.
pub trait Provider {
    fn complete(&mut self, req: &ChatRequest) -> Result<ChatResponse, ProviderError>;
}

/// A scripted provider for tests, replays, and offline dry runs. Also
/// the right subject for testing experiment code itself: its recorded
/// requests prove what the experiment actually asked.
#[derive(Debug, Default)]
pub struct MockProvider {
    script: VecDeque<Result<ChatResponse, ProviderError>>,
    pub requests: Vec<ChatRequest>,
}

impl MockProvider {
    pub fn new(script: Vec<Result<ChatResponse, ProviderError>>) -> MockProvider {
        MockProvider {
            script: script.into(),
            requests: Vec::new(),
        }
    }
}

impl Provider for MockProvider {
    fn complete(&mut self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.requests.push(req.clone());
        self.script
            .pop_front()
            .unwrap_or_else(|| Err(ProviderError::Transport("mock script exhausted".into())))
    }
}

/// What one attempt cost, as far as this ledger can know. The provider's
/// billed balance is the primary record for real spend; this ledger is
/// an estimate for steering and stopping, and says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptCost {
    /// Usage was reported and priced with the run's stated rates, in
    /// micro-USD (millionths of a dollar), integer arithmetic only.
    Estimated(u64),
    /// The provider refused; whether it billed is unknown from here.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    Completed,
    Refused(String),
    TransportFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attempt {
    pub experiment: String,
    pub model: String,
    pub outcome: AttemptOutcome,
    pub cost: AttemptCost,
}

/// Per-run pricing in micro-USD per token, supplied by the caller from
/// pricing verified against the provider's own published rates before
/// the run. Integer rates keep money arithmetic exact.
#[derive(Debug, Clone, Copy)]
pub struct Pricing {
    pub input_micro_per_token: u64,
    pub output_micro_per_token: u64,
}

impl Pricing {
    pub fn estimate(&self, resp: &ChatResponse) -> u64 {
        resp.input_tokens * self.input_micro_per_token
            + resp.output_tokens * self.output_micro_per_token
    }
}

/// The spend ledger and its ceilings. One ledger per run; the run's
/// spec names its ceilings before the first call, and they are
/// constructor arguments precisely so they cannot be raised mid-run by
/// the code being guarded.
#[derive(Debug)]
pub struct SpendLedger {
    max_attempts: usize,
    refusal_streak_limit: usize,
    attempts: Vec<Attempt>,
    last_refusal: Option<String>,
    refusal_streak: usize,
    window_closed: bool,
}

impl SpendLedger {
    pub fn new(max_attempts: usize, refusal_streak_limit: usize) -> SpendLedger {
        assert!(
            max_attempts > 0,
            "a run with a zero attempt ceiling is a spec bug"
        );
        assert!(
            refusal_streak_limit > 0,
            "refusal streak limit must be positive"
        );
        SpendLedger {
            max_attempts,
            refusal_streak_limit,
            attempts: Vec::new(),
            last_refusal: None,
            refusal_streak: 0,
            window_closed: false,
        }
    }

    pub fn attempts_made(&self) -> usize {
        self.attempts.len()
    }

    pub fn attempts(&self) -> &[Attempt] {
        &self.attempts
    }

    pub fn window_closed(&self) -> bool {
        self.window_closed
    }

    /// Sum of the estimates this ledger can price. Attempts with
    /// unknown cost are excluded and counted separately by
    /// `unknown_cost_attempts`, so the total can never silently absorb
    /// them as zeros.
    pub fn estimated_total_micro(&self) -> u64 {
        self.attempts
            .iter()
            .filter_map(|a| match a.cost {
                AttemptCost::Estimated(v) => Some(v),
                AttemptCost::Unknown => None,
            })
            .sum()
    }

    pub fn unknown_cost_attempts(&self) -> usize {
        self.attempts
            .iter()
            .filter(|a| a.cost == AttemptCost::Unknown)
            .count()
    }

    fn check_can_attempt(&self) -> Result<(), ProviderError> {
        if self.window_closed {
            return Err(ProviderError::WindowClosed);
        }
        if self.attempts.len() >= self.max_attempts {
            return Err(ProviderError::CeilingExceeded);
        }
        Ok(())
    }

    fn record(
        &mut self,
        req: &ChatRequest,
        result: &Result<ChatResponse, ProviderError>,
        pricing: &Pricing,
    ) {
        let (outcome, cost) = match result {
            Ok(resp) => (
                AttemptOutcome::Completed,
                AttemptCost::Estimated(pricing.estimate(resp)),
            ),
            Err(ProviderError::Refused(reason)) => (
                AttemptOutcome::Refused(reason.clone()),
                AttemptCost::Unknown,
            ),
            Err(e) => (
                AttemptOutcome::TransportFailed(e.to_string()),
                AttemptCost::Unknown,
            ),
        };
        match &outcome {
            AttemptOutcome::Refused(reason) => {
                if self.last_refusal.as_deref() == Some(reason.as_str()) {
                    self.refusal_streak += 1;
                } else {
                    self.last_refusal = Some(reason.clone());
                    self.refusal_streak = 1;
                }
                if self.refusal_streak >= self.refusal_streak_limit {
                    self.window_closed = true;
                }
            }
            _ => {
                self.last_refusal = None;
                self.refusal_streak = 0;
            }
        }
        self.attempts.push(Attempt {
            experiment: req.experiment.clone(),
            model: req.model.clone(),
            outcome,
            cost,
        });
    }
}

/// A provider wrapped in a ledger. All experiment calls go through this
/// type; the inner provider is unreachable except through `complete`.
pub struct GuardedProvider<P: Provider> {
    inner: P,
    pub ledger: SpendLedger,
    pricing: Pricing,
}

impl<P: Provider> GuardedProvider<P> {
    pub fn new(inner: P, ledger: SpendLedger, pricing: Pricing) -> GuardedProvider<P> {
        GuardedProvider {
            inner,
            ledger,
            pricing,
        }
    }

    /// Read access to the wrapped provider, for inspecting a mock after
    /// a test run. Calls still only flow through `complete`.
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<P: Provider> Provider for GuardedProvider<P> {
    fn complete(&mut self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.ledger.check_can_attempt()?;
        let result = self.inner.complete(req);
        self.ledger.record(req, &result, &self.pricing);
        result
    }
}
