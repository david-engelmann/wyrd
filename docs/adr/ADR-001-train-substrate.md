# ADR-001: Train substrate after the candle spike

Status: DECIDED 2026-11-05, under David's delegation in the main
chat ("Make the best decisions"). Decision below.

## Context

Charter v0.4 section 10 requires the train crate's substrate to pass a
spike before PRE-001 can seal: a roughly 10M parameter model training
on this box at a rate projecting the reduced-regime ladder (12 runs of
20M tokens, 240M total) to at most 7 days. The spike ran on 2026-11-05
(full record in the program's SPIKES.md). Candle builds and trains
correctly. The measured rate was 161 tokens per second under load
average 15 to 17 on the 2-core box, projecting 17.3 days. Verdict:
fail as measured. A contention-adjusted estimate suggests an unloaded
box could project about 3 days, but the box runs the whole fleet
nonstop by David's standing order, so the unloaded figure describes a
machine that does not exist in practice.

## Options

A. Amend the charter's PRE-001 regime downward. Halve the ladder to
   120M tokens total (for example doses 0, 100, 250, 500 at 3 seeds
   with 10M tokens per run, or the same doses at 20M tokens with 2
   seeds and the seed-variance cost stated). At the measured rate that
   projects about 8.6 days, still over a 7-day bar; the bar would move
   to 14 days for Phase 2, or the regime shrinks further. Cost: zero
   money. Cost in science: PRE-001 is already labelled a below-anchor
   probe; shrinking it again weakens the dose-response read, and the
   10M-token variant moves further from the anchor's dynamics.

B. External trainer behind the train crate (the charter's named
   fallback). Training runs on rented GPU compute, driven and recorded
   by Rust code here, with the trainer quarantined behind the crate
   boundary and a language-ledger entry. A single modern GPU trains
   this model class at tens of thousands of tokens per second, so the
   full ladder is hours, not weeks. Cost: real money, and it needs
   David to reopen rented compute for wyrd's scope, against his
   2026-10-22 ruling closing Vast (reopened 2026-10-29 only for
   peira's local model runs). Prepaid-balance guardrails from that
   ruling would apply unchanged.

C. Optimise the local path and re-spike. Larger batches, longer
   context per step, and candle's faster CPU kernels could plausibly
   multiply the local rate several-fold; the spike made no attempt at
   any of them. Cost: days of engineering against an uncertain ceiling,
   and the result still shares the box with the fleet. This option can
   be combined with A: optimise, re-measure, then set the regime from
   the new rate.

## Recommendation

C then A, in that order, with B held in reserve. The optimisation pass
is bounded (one re-spike), it keeps the program money-free and inside
David's compute ruling, and whatever rate it reaches sets an honest
regime size instead of a hoped one. B becomes the recommendation only
if the re-spike cannot project the ladder inside 14 days at any regime
that still reads as a dose-response experiment (at least four doses
and at least two seeds), because past that point the local path is
not slow, it is absent.

## Decision (2026-11-05)

The recommendation is adopted. The path is C then A: optimise the
local candle path first (the spike attempted none of the obvious
speedups), re-spike, and size the PRE-001 regime from the measured
rate. Option B, the external GPU trainer, is held in reserve only. It
is not approved, and this decision authorises no spend and no
reopening of rented compute for wyrd's scope. If the re-spike reaches
the reserve trigger named in the recommendation, B returns as a fresh
decision for David, not as an automatic escalation. The train crate
may now be built for the optimisation work; PRE-001's spec still
cannot seal until the re-spike prints its measured rate and ladder
projection.

## Re-spike outcome (2026-11-06)

The optimisation pass and re-spike are complete. Measured on the
live box under fleet load, release build, causal mask cached: the
baseline configuration (batch 8, context 128) reached 229 tokens per
second; the optimised configuration (batch 16, context 256) reached
296 tokens per second aggregate, with steady-state steps near 490.
The batch 32 by 256 configuration was killed by the box's memory
limit and does not fit this machine alongside the fleet. Loss fell
in every configuration. The ladder projection at the sizing rate is
9.4 days, so the re-spike fails the charter's 7-day bar, and the
amended envelope in this ADR governs: the full chartered regime (4
doses, 3 seeds, 240M tokens) projects inside 14 days at full
statistical strength, so PRE-001 proceeds under option A with the
regime unchanged and its schedule set from the measured rate (about
10 days with margin). The reserve trigger did not fire. Option B
remains in reserve, unapproved, with no spend authorised. Full
measurements are recorded in the program's spike log.

## Consequences either way

- The train crate was not built while this ADR was undecided. With
  the decision recorded above, it may be built for the optimisation
  and re-spike work.
- Phase 1 (RAG and graph arms) is unaffected and proceeds.
- PRE-001's spec cannot seal until the chosen path re-measures its
  rate and prints its projection, per the charter.

Addendum (2026-11-06): the re-spike outcome above says PRE-001
proceeds "with the regime unchanged" at 240M tokens over 12 runs.
That accounting omitted charter section 4's mandatory sham arm.
Spec v0.2 prices Arm S in: 21 runs at a trimmed 16M token budget
per run, 335,978,496 tokens, about 13.1 days of training at the
measured rate. The decision itself (option A, candle local, no
spend, reserve trigger unfired) is unchanged; only the regime
sizing moved, under this ADR's own "size the regime from the
measured rate" instruction. Spec v0.2 carries the current numbers.
