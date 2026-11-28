# PRE-001 amendment, 2026-11-27. Checkpoint cadence

This note amends one operational premise of the sealed PRE-001 spec
without changing the sealed document. The spec hash recorded in the
seal record is untouched.

## What the spec says

PRE-001, in its passage on the recorded AdamW deviation, reasons
about run supervision with checkpoints every 250 steps against
3,906 steps per run, and bounds the moment history a kill can lose
on that premise. The operative rule in the same passage is
cadence-independent. A resumed run continues from its last step
checkpoint with the same seed and the same stream position.

## What changes

By the repository owner's ruling of 2026-11-27, the driver's
checkpoint cadence changes from every 250 steps to every 50 steps.
The change is confined to TrainConfig::pre001 in wyrd-train and its
pinning test. The engine marker, the batch schedule, the seed
handling, and every quantity the spec fixes are unchanged. The
work a killed run can lose falls from at most 249 steps to at most
49 steps.

## Why

The ladder's training children carry a live memory footprint of
about 3 GB from saved activations and backward tensors on a shared
box of about 8 GB, and the kernel's out-of-memory selection kills
them irregularly. The diagnosis is recorded in the programme
record. Under the 250-step cadence, one run lost its whole
sub-checkpoint progress three times in two days, and no completed
run was ever affected. The cadence is snapshot policy, not part of
the trained trajectory, so tightening it costs the sealed results
nothing.

## Evidence

Trajectory neutrality was verified before this amendment shipped.
A scratch run of the arm P dose 0 seed 1 cell on the amended
driver reproduced the banked sealed trajectory exactly for the
50 steps of its uninterrupted segment, including the step at
which the new cadence writes an additional checkpoint. The run
was then killed with SIGKILL past step 50 and resumed from the
step 50 checkpoint, a resume point the old cadence cannot
produce. The step sequence continued without a gap or a
duplicate. The first resumed loss matched the banked value
exactly. Later losses diverged smoothly, reaching a maximum
absolute difference of 0.074 over the following 30 steps. That
divergence is the AdamW moment restart the spec already records
as its standing deviation, present at any cadence, and it is
measured here rather than assumed away.

## Standing of prior results

Every completed PRE-001 run stands as sealed. Checkpoints written
under the old cadence remain valid resume points under the new
one, because 250 is a multiple of 50 and the resume search accepts
any checkpoint the cadence arithmetic admits.
