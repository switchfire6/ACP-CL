# Causal-access engineering record

Before development fitting, 71 targeted tests passed: 32 mixer checks, 10 new
runner/integration checks and 29 existing component-output checks. Ruff passed
for the new scientific package, independent scorer and new tests.

The first engineering smoke completed both architectures, 16 fit phases,
176 updates and 256 frozen stream arrivals. The full learner/global-RNG
invariance assertions passed. Independent scoring rejected specialist success
counts serialized as integral floats (e.g. 22.0), because validation initially
required an integer dtype. The unchanged historical trainer produces those
floats through NumPy signed/unsigned accumulation.

Before main fitting, the scorer was corrected to accept finite, exactly
integral numeric success totals bounded by each action's count. A synthetic
check rejects fractional, NaN and infinite totals. All eight scorer synthetic
checks pass. No prediction, training setting, threshold, smoothing rule or
seed was changed. The failed original lock and artifacts remain untouched in
runs/causal_access_smoke. The corrected smoke uses a fresh artifact directory,
runs/causal_access_smoke_v2, and remains scientifically ineligible. That corrected
smoke completed and passed independent raw-array reconstruction for both jobs.

Development source/configuration/protocol/analysis locked at
2026-09-28T21:46:19.811857 UTC, before any development fit. The source identity is
80e45794ef9d25b2af551df931f6fa0a99865e1a9bdda5760dc05ec22446ddfa;
the lock covers 47 scientific source files and five analysis/config/protocol/test
files. Training uses the one declared cohort and settings.

Plots and checkpoint reconstruction are presentation/verification tools;
the decision-bearing NumPy analysis is locked before main fitting. There is
no model, evidence-window, temperature, capacity or exposure sweep.

The main completed at 2026-09-28T21:58:34.134186 UTC, 734.32 seconds after locking.
All 12 jobs, 120 phases, 175,104 updates and 196,608 frozen-stream arrivals completed.
Independent scoring and checkpoint reconstruction passed. Both qualification
and raw primary screens fail; no further fits were launched.

Completed-run resumption of smoke_v2 reused all completed artifacts and added
no training. Main checkpoint reconstruction is recorded in
development_checkpoint_audit.json, and the separate arithmetic/interpretation
review in outcome_review.md. Older-file preservation has two discrepancies in
the new initial inventory; current bytes match prior canonical artifacts.
The initial inventory and failed comparison are preserved; see
preservation_review.md for evidence and the unresolved cause.
