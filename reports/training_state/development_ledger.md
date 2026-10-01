# Training-state development ledger

Before learner training, declare the unchanged world and architecture, factorial
reset definitions, fresh-only development seeds 411--416, exposure grid
4,096 then 8,192, and all gates in `docs/training_state_diagnostic.md`.
The development cue-benefit requirement is .0025 in each cue and stage group;
the comparison retains the original .002 requirement. Both require marginal
Brier improvement >= .02. No reset outcomes select exposure.

Initial engineering checks passed: 12 tests cover exact factor isolation,
unchanged replay draws for optimizer-only reset, shared clean/noisy physical
queries and performed actions, and exact interruption recovery through final
branches. Prior scientific sources are imported unchanged. The previous
conditional delay-cue qualification failure remains in its original archive.

The first fresh-only cohort uses 4,096 arrivals. Preserve all outcomes and
source identities before selecting or rejecting that exposure.

## 4,096 arrivals: failed qualification

All 36 fresh fits completed. Both delay-cue groups failed: .00176401 conditional
and .00070667 recurrent against the declared .0025 development threshold. They
also fall below the comparison threshold of .002. Every other stage/cue group
passed and all marginal-predictor improvements passed. Keep the complete failed
attempt. Proceed to the predeclared 8,192-arrival budget without changing any
world, architecture, threshold, or comparison seed.

## 8,192 arrivals: selected

All 36 fresh fits completed and all twelve model-by-stage/cue groups passed.
Delay-cue benefit was .00423232 conditional and .00422158 recurrent, above the
declared .0025 development margin. Select 8,192 arrivals for every arm/final
branch. No further settings were tried. All 72 development starting/final
checkpoint pairs passed local audits; the 62-episode engineering smoke passed
its audit, including eight matched clean/noisy pairs. The full suite passed
702 tests with two Windows symbolic-link skips, and lint passed.

Before comparison lock, set execution concurrency to eight one-thread workers
on the 12-core CPU, with 21 GB physical memory free. Development used four.
This operational change does not alter learning rules or exposure. Preserve
the different recorded runtimes and avoid hardware-efficiency claims.

## PC restart and recovery during comparison

After comparison launch, the user reported a PC crash/restart. The original
execution sessions were gone. The first resumed attempt exposed invalid saved
files, so its experiment process tree was stopped for an integrity check.
Archive CRC checks identified twelve damaged PyTorch checkpoints and four
invalid prefix metadata files. Preserve the original bytes and hashes under
`runs/training_state_recovery/` and in the report's damaged-file archive.

All 52 remaining checkpoints passed ZIP CRC and deserialization checks; their
model weights were finite, and prefix/episode-start hashes matched stored
records. Preserve the 16 completed episodes. Restart the partial conditional
stage-one continuation for seeds 8001--8004 from their intact prefixes; rebuild
the corrupted recurrent prefixes for those same seeds. No outcomes selected
the retries, no seed was replaced, and no completed episode was excluded.

The resumed process tree keeps the original eight workers and one Torch thread
per worker, but is restricted to CPU affinity mask 85 (logical processors
0,2,4,6) to reduce load. This supplemental operational setting is recorded in
`recovery_history.json`; the original config, source, protocol and identity
remain unchanged. The cause of the PC restart is unknown. Atomic checkpoint
replacement alone did not guarantee durability through this hard restart.

## Outcome review timing

At 17:59 UTC, all 180 acquisition episodes, including all 36 fresh references,
were complete; only final control branches remained. Review the complete
acquisition cohort's qualification and factorial estimates while those fixed
branches finish. Treat the readout as provisional until the full checkpoint
audit and final-control analysis. No source, setting, seed, stopping rule, or
qualification threshold changes follow this review. The final summary will
recompute and cross-check the same primary estimates.

## Completion and verification

All 12 trajectories and 372 episodes completed at 18:08:35 UTC. The complete
summary exactly reproduced the previously reviewed acquisition contrasts.
Every fresh-model qualification group passed. The local audit passed 12 prefix
checkpoints, 372 starting/final pairs, and 48 matched clean/noisy pairs. All
1,145 stable run files passed JSON/ZIP integrity checks. No further recovery
was required. Five recovery-integrity regression tests passed in addition to
the earlier 702-test full suite (two Windows symbolic-link skips).

The final controls show why replay reset is not a complete intervention:
returning-world prediction improves while broader old-condition retention
worsens, and corrupted-feedback prediction becomes less reliable. Preserve
these costs alongside acquisition gains. The next research proposal concerns
bounded replay renewal; this study did not train or select that proposal.
