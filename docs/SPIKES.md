# Substrate spikes

Charter v0.4 gives two substrates a spike with a stated bar before
their arms can seal: the candle trainer (section 10, PRE-001) and the
hand-rolled GCN trainer (section 10, GNN-001, red team M12).

## Candle re-spike under ADR-001 (2026-11-06, complete)

The re-spike harness (candle-spike2) prints its build profile, closing the original spike's unrecorded-profile gap. All rates measured on the live box under fleet load (load average 8 to 15 during the runs), because the box is never unloaded under the standing order and a contention-adjusted figure would not be honest sizing data.

- Baseline configuration (batch 8, context 128, release build, causal mask cached): 229 tokens per second, loss 6.486 falling to 6.157 over 12 steps. Ladder projection 12.2 days. (The original spike measured 161 tokens per second; release mode and the cached mask account for the improvement, and the original run's profile was never recorded.)
- First optimised attempt (batch 32, context 256): killed by the box's memory limit before its first step completed. That configuration does not fit a 7.7 GB box shared with the fleet.
- Optimised configuration as revised (batch 16, context 256, release): 296 tokens per second aggregate over 12 steps and 49,152 tokens, loss 6.521 falling to 6.131. Steady-state steps (4 through 11, after warmup under peak contention) ran about 8.3 seconds per 4,096-token step, roughly 490 tokens per second; the aggregate is the sizing figure because long runs still start cold and the fleet load is a standing condition, not an accident. Ladder projection 9.4 days.

Verdict: the re-spike FAILS the charter's 7-day bar at every configuration. Under ADR-001's decision rule that is not the reserve trigger: the full chartered regime (4 doses, 3 seeds, 240M tokens) projects 9.4 days at the measured aggregate rate, inside the ADR's 14-day amended envelope at full statistical strength. PRE-001 therefore proceeds under option A with the regime unchanged and the schedule amended to the measured projection (stated as about 10 days with margin). The external GPU trainer stays in reserve, unapproved, with no spend authorised; the reserve trigger (no regime with at least 4 doses and at least 2 seeds projecting inside 14 days) did not fire.

## GCN spike: PASS (2026-11-03)

Code at ~/workspace/gcn-spike (pure Rust, no dependencies, dense
matrices at spike scale). A two-layer GCN with hand-derived backprop:

- Gradient check: analytic gradients against central finite
  differences on six weights across both layers, max relative error
  5.59e-8 against a bar of 1e-4. The backprop is right, not plausible.
- Learning check: on a synthetic citation-style graph (90 nodes, 3
  classes, prototype features with noise, within-class edge probability
  0.28 against 0.015 across), 300 full-batch epochs reach train loss
  0.011 and test accuracy 1.000 against a bar of 0.85, in 1.4 seconds.

Caveat recorded honestly: the synthetic graph is easy, and feature
signal alone nearly separates the classes. What this spike proves is
gradient correctness and end-to-end trainability of the hand-rolled
approach, which were the M12 risks. The charter's reference-parity bar
(match a reference GCN on the chosen citation graph within 2 points)
still applies at GNN-001 spec time, against the real dataset the spec
names. No ADR fallback is needed for the trainer itself.

## Candle spike: FAIL as measured, decision escalated by ADR (2026-11-05)

Candle itself works: candle-core and candle-nn 0.9.2 build (13 minutes
of dependency compilation on this box) and train a real model. The
spike model is a 14.4M parameter byte-level GPT (vocab 256, context
128, d_model 384, 8 layers), trained with AdamW on a synthetic byte
stream. Two measurement runs matter, and both are recorded because the
first was invalid:

- Run A (invalid model, valid warning): the first spike model had no
  layer normalisation, a defect of the spike code, and its loss opened
  at 333,856, an absurd value that exposed the bug on sight. Rate was
  42 tokens per second under load average about 8 to 15. The model was
  fixed (pre-norm blocks, final norm) before any verdict was drawn.
- Run B (the measurement): loss opens at 6.44 and falls to 6.13 over 12
  steps, sane for byte-level training. Rate: 161 tokens per second at
  load average 15 to 17 on a 2-core box, i.e. under roughly sevenfold
  oversubscription from the rest of the fleet. Projection for the
  PRE-001 reduced-regime ladder (240M tokens): 17.3 days against the
  charter bar of at most 7. SPIKE VERDICT: FAIL as measured.

Reading the failure honestly: contention-adjusting the rate suggests
an unloaded box might reach roughly 1,000 tokens per second and a
3-day ladder, but this box is never unloaded under the standing
nonstop order, and a spike verdict may not rest on a hypothetical
machine. The charter's pre-decided fallback is an ADR choosing an
external trainer quarantined behind the train crate. That fallback
collides with David's standing ruling closing rented compute (Vast
reopened only, narrowly, for peira's local model runs), so the choice
is genuinely his and is written up as ADR-001 with options and a
recommendation. PRE-001 is Phase 2 work; nothing in Phase 1 waits on
it. The train crate stays unbuilt until the ADR is decided, which the
charter's language ledger records as the spike outcome, not drift.
Also corrected here: an early draft of the spike header claimed the
bar needs 40k tokens per second. The arithmetic is 240M tokens over 7
days, about 400 per second. The error was in the spike's own comment
and never reached the charter.

Addendum (2026-11-06, after the PRE-001 spec reviews): the re-spike verdict above predates the spec's review round. The spec-fidelity review found the charter's mandatory sham arm (Arm S) missing from PRE-001 v0.1; pricing Arm S in changed the regime: 21 runs (12 poison, 9 sham) at a trimmed 16M token budget per run (15,998,976 tokens), 335,978,496 tokens total, projecting to about 13.1 days of training at the measured 296 tokens per second, still inside the ADR-001 envelope and still option A with no spend. Spec v0.2 carries the current numbers; this addendum keeps the spike record from contradicting them.
