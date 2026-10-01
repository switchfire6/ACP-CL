# Core/residual experiment ledger

The user approved the learned-core plus adaptive raw-input pathway comparison.
Implementation resumed after a usage interruption on 2026-09-28 UTC. No new
scientific outcome was available when the protocol and fixed cohort were written.

## Prospective design

- Main config: `configs/core_residual_development.json`, seeds 16001--16006,
  both conditional and recurrent architectures, six one-thread workers,
  affinity mask 1365. Three cue groups crossed with two initial modes.
- Shared ordinary learned prefix: four alternating base-law blocks of 1,024.
- Allocate a zero-output raw-input residual once, before maintenance. Compare
  joint learning, a frozen learned core with adaptive residual, and a frozen
  learned core with fixed random residual visual encoder.
- Continuous maintenance 1,024, new dependency 8,192, exact base return 2,048.
  An additional fresh reference learns the same 8,192-arrival novel stream.
- Sole candidate: separate versus joint, with fixed acquisition, retention,
  return, survival and fresh-qualification screens. Feature attribution is
  secondary and cannot promote a failed primary candidate.
- No fit-dependent exposure, seed, architecture, threshold or learning-rate
  changes. Fixed numerical analysis and science code will be locked before fit.
- Portable verifier and its tests are explicitly excluded from the science lock
  and will be sealed separately. This is packaging, not outcome-dependent analysis.

## Engineering checks before scientific fitting

- Learner: 38 passed, 2 architecture-inapplicable cases skipped; Ruff passed.
- Evaluation trace: 54 passed; Ruff passed.
- Runner: 17 passed; Ruff passed. Includes both architectures, exact interrupted
  continuation, recovery between final checkpoint and JSON, causal pre-update
  probabilities, empty fresh memory/Adam, matched streams/replay and tamper checks.
- Independent design/runner review found no scientific blocker. The fixed-feature
  control freezes the visual frame/temporal encoder; heads/context still learn.
- Full tensor and portable audits remain in preparation. These reconstruct
  boundary forecasts and check raw physics/arithmetic; they do not independently
  retrain every ordinary update or reconstruct every intermediate model state.

The pre-edit preservation inventory covers 1,595 earlier scientific files,
including 1,313 report artifacts. Keep old science and archives byte-identical.

## Unsealed engineering fixtures

`runs/core_residual_engineering_20260928` completed both smoke-sized jobs,
24 phases, solely to exercise the saved checkpoint and numerical formats. It
has engineering identity `core_residual_preflight`, no scientific manifest,
and cannot supply a development decision. Its tiny-run plots were inspected
solely for figure layout; no setting or threshold was changed from their scores.
The subsequent locked smoke occupies a different directory.

A timing-only check at main widths on disjoint engineering seed 16992 performed
240 composite joint updates per model: 2.503 seconds conditional and 6.203 seconds
recurrent. No held-out comparison was evaluated. These exclude checkpoint and
evaluation work and are not a claimed speed advantage or full-fit estimate.

The combined mechanism/evaluation/runner and existing conditional/contextual/
replay dependency check passed 175 tests with two architecture-specific skips;
the subsequent additional validation case brings the runner total to 17.
The 1,595-file preservation check still reports zero changes.

## Scientific locks and launch

The final combined scientific suite passed **195 tests**, with two architecture-
specific skips, in 40.86 seconds. Scoped Ruff passed. No source/analysis changes
followed these tests before locking. The 19 runner cases include interrupted
ordinary-prefix and adaptive-residual continuation in both architectures.

Locked smoke `runs/core_residual_smoke` completed all 24 phases and 232 updates.
Strict numerical export passed. The tensor audit verified 24 before and final
checkpoints, six copied-prefix forks, two fresh initializations, 232 causal
packets, and 416 evaluator calls. It reconstructed 382 stored evaluator forecasts
and 24 first-packet predictions; the other 34 evaluator calls have only their
raw arithmetic, physical truth and provenance independently checked. All 152
stopped-run files passed integrity. Export: `reports/core_residual/smoke`.

The full main cohort was locked before fitting, then launched around
**2026-09-28 02:28:00 UTC**, output `runs/core_residual_development`, six single-
thread CPU workers with affinity 1365. Complete all 168 phases before outcome
comparison. Source identity:
`c5ac7b538363930c68c999b60f5d26244eaf5b5eb676c8cf562d1ca354109a3b`.
Config identity:
`e0ab9198f29aadf138a589e589146644828fdb64e64b44adfc9db3689db2a1a2`.

Portable verifier/tests were completed under their prospective packaging
exclusion. They altered no scientific source, protocol, configuration,
analysis, figure code or scientific test after this lock.

## Completed cohort and fixed decisions

The main completed **2026-09-28 02:39:04.593460 UTC**, 672.57402 seconds after
locking. All 12 jobs, 168 phases, 552,960 arrivals, 17,280 packets and 207,360
updates finished before scientific comparisons were inspected. There were no
replacement seeds, failed-fit exclusions, added exposures or threshold changes.

Both fresh architectures qualify overall and in every cue group. Both sole
primary screens fail. Separate-minus-joint novel affected Brier AUC is
+.016853027 conditional (4/6 seeds improve) and +.003916636 recurrent (2/6).
Conditional also fails novel survival and the post-return novel endpoint;
recurrent also fails return prediction and both survival guards. The conditional
post-return gap does not mean greater forgetting during return: it entered
return with worse novel predictions. Recurrent feature attribution passes
(-.012130091, 6/6 improve versus fixed random visual features); conditional
attribution fails consistency. No secondary arm or favorable cue subset is
promoted. Full interpretation is in reports/core_residual_results.md and the
independent outcome_review.md.

## Verification and final preservation

- 195 scientific tests passed with two inapplicable cases skipped; 73 portable
  tests passed. Scoped Ruff passed. Total: 268 passing scoped tests.
- Main stopped-file integrity passes 1,026 JSON/ZIP/checkpoint/NPZ files.
- Full tensor audit passes 168 before/168 final states, 36 copied-prefix forks,
  12 fresh initializations, 17,280 packets, and 6,912 evaluator calls. It
  reconstructs 4,920 evaluator forecasts plus 168 first-packet predictions;
  1,992 intermediate calls have physical/arithmetic/provenance checks only.
  Ordinary training is not independently repeated.
- Independent NumPy arithmetic reconstructs all evaluation/prequential metrics,
  1,108 bootstrap distributions, every qualification group and 26 decision
  gates. Maximum difference: 2.22e-16.
- All four development figure families were generated from locked figure code
  in PNG/SVG/PDF and visually checked.
- Smoke and development archives contain 66 and 366 sealed artifact entries.
  Both verify in place and after intact copies outside the repository with
  isolated Python, no repository imports and no model pickle loading.
- Completed-run resume leaves all 153 smoke and 1,027 development files
  byte-identical; these full counts additionally include the protocol Markdown
  not covered by the generic JSON/ZIP/checkpoint/NPZ file-integrity count.
- Final preservation check passes all 1,595 prior scientific files, including
  1,313 report artifacts, with zero changes. Science/protocol/analysis locks
  remain unchanged. New report links resolve.

All requested implementation, fitting, analysis, audit, archival and handoff
work for this experiment is complete. No fitting processes remain. The next
proposed core/residual/context diagnostic has not been run; it is a separate
exploratory question, not a new policy claim or an extension of this cohort.
