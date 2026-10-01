# Implementation and engineering checks

The representation-learning candidate, causal runner, independent NumPy scorer
and fixed prospective protocol were completed before reference fitting. New code
is isolated from the earlier scientific implementations.

## Tests and integration

Component checks passed60 new tests:16 learner,12 study/resumption,22 independent
scorer and10 compute-allocation tests. The18 existing contextual tests also
passed. These totals count distinct tests, not repeated invocations. Ruff passed
for new source, tests, scorer and calibration code.

The outcome-only arm reproduces the archived recurrent baseline's predictive
weights, gradients, optimizer state, replay and forecasts exactly. All four
arms start with identical predictive weights; only auxiliary arms allocate a
reconstruction decoder. Tests verify gradient flow through the same predictive
latent, normalized query-only targets, original-context replay and no auxiliary
decoder use during inference.

Study checks include interruption after partial and final checkpoints, exact
continuation, completed-run resume, source/runtime/phase lock rejection, causal
forecast ordering, evaluator state preservation, corrupted artifact rejection
and failed/missing qualification guards. Scorer checks include separate fresh
stage/cue grouping, paired cue masks, exposure/update accounting, complete
trace chronology, raw metric reconstruction and tamper rejection.

Persistent engineering runs:

- `runs/representation_learning_smoke`: four qualification phases; independent
  scoring correctly marks the reduced one-seed run INELIGIBLE. Audit restored
  eight checkpoints, regenerated512 arrivals and64 causal packets, reconstructed
  28 boundary evaluation forecasts and4 initial ordinary forecasts.
- `runs/representation_learning_main_smoke`: all four arms and44 phases;
  independent scoring correctly marks it INELIGIBLE. Completed-run resume did
  not retrain. Audit restored88 checkpoints, regenerated2,304 arrivals/288 packets
  and verified360 updates/replay draws. It reconstructed232 boundary evaluation
  forecasts plus44 initial ordinary forecasts and checked every evaluation's
  physical data. See the separate JSON audit reports for coverage and limits.

Smoke runs are software checks; their behavioral scores do not select settings
or qualify the research hypothesis.

## Useful-compute allocation

The one declared engineering measurement used seed18091, five paired timing
rounds,16 warmup batches and8 timed batches per arm,12 updates each. Each round
recreated the same predictive initialization, incoming observations and replay
workload. It measured training calls only and recorded no behavioral or loss
scores. The original attempt record is retained.

Median durations for eight batches: outcome1.28055s, final-frame1.40151s and
sequence1.38672s. The largest timing ratio is1.09446. The declared dense matrix
work ratio is1.168891, which governs ceil(12*ratio)=**15 useful outcome updates**.
This is approximate resource allocation, not exact device FLOP equality.

Calibration used4,320 optimizer updates including warmup and took62.93s. Actual
candidate parameters, optimizer/replay payload bytes, source/config/protocol
hashes, all round timings and accounting details are recorded in
`compute_allocation.json`. A successful auxiliary would still incur extra
model/optimizer memory; matched replay capacity does not imply equal parameter
storage. Main-run work and elapsed time must be reported before efficiency claims.

## Historical preservation

The pre-work inventory records51,872 pre-existing files /33,390,908,880 bytes,
including older isolated Python environments stored below runs. General status
documents and this new experiment are excluded. A post-work comparison is
required before final delivery. Do not overwrite the original inventory.
