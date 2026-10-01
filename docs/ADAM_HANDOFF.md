# Adam research handoff

## Latest: Steps A1, B, B2 done; Step C stopped at development; pivot to Step D (2026-09-29)

These steps follow [recoverable_context_proposal.md](recoverable_context_proposal.md),
which the user approved. Each has a prospective protocol written before any
computation. All outputs are new files, and no locked record was changed.

**A1: retrieval capacity** ([protocol](retrieval_capacity_protocol.md),
[report](../reports/retrieval_capacity/summary.md)). Verdict: **BORDERLINE**.

- The primary criteria pass: closed fraction .848, and 5/6 seeds ≥ .5.
- The Base B control criterion was ambiguous as written, and one reasonable
  reading fails it. We do not pick the favourable reading after the fact.
- What holds under every reading:
  - Base A is largely **not erased**. The frozen decoder expresses it through
    one fitted context (.094 against a block-end level of .085).
  - The context network **cannot reach** that context. It lies 2–4 away, while
    support-to-support spread is about 1.3.
  - Contexts from a single support are very noisy.
  - Part of the gain from a fitted context is generic, not retrieval. After
    adjusting for the control, closure is about .48 on the mean.
  - Refitting only the decoder on the frozen encoder reaches .063 in one law.
  - Delay is lost at two stages: weak cue-1 encoding (linear accuracy .68),
    and a decoder that does not use it.

**B: stationary competence gap** ([protocol](competence_gap_protocol.md),
[report](../reports/competence_gap/summary.md)). 60 locked fits on seeds
19101–19106.

- **No competent learner.** The best arm (U12 R256 W64) has excess .0226 over
  O. The bar is .010.
- **Replay capacity 256** removes 24.1% of the gap in 6/6 seeds. It narrowly
  misses the pre-declared 25% rule, so it is not credited.
- **Width 256 is worse**, in 6/6 seeds.
- **Updates per batch do not matter.**
- **The plateau is a steady state.** Exposure to 32,768 arrivals does nothing.
- **Privileged all-action feedback** closes 44% of the gap, but it also
  plateaus by 3,072 arrivals at .019.
- The U3 arms have better Brier but worse survival regret and weaker cue
  relations. Brier alone misleads.

**Open issue.** The .010-over-O bar was never checked against what any learner
can achieve from 8,192 performed-action records.

**B2: competence ceiling** ([protocol](competence_ceiling_protocol.md),
[report](../reports/competence_ceiling/summary.md)). 12 fresh seeds
(19301–19312), locked. All four pre-declared readings are positive.

- **R1. The .010 bar stands.** A feature-level ceiling reaches .0078 at 8,192
  records.
- **R2. Online optimisation is the dominant share of the gap.**

  | Share | Size | Fraction of the reference's excess |
  |---|---:|---:|
  | Online optimisation | .0243 | 64% (12/12 seeds) |
  | Perception | .0058 | 15% |
  | Estimation | .0078 | 21% |

- **R3. Full-history replay (R256) is confirmed:** −37%, in 12/12 seeds.
- **R4. Weight averaging (EMA) helps:** −48%, in 12/12 seeds.
- **Combined.** R256 with EMA reaches .0124. That is 67% below the reference,
  and within .005 of the feature ceiling, but not yet ≤ .010 over O.
- **Lag.** EMA lags a fresh start by about 2 chunks.

**Integrity notes.**

- A replicate pass was bit-identical.
- One archive was silently corrupted on disk. It was detected, quarantined and
  regenerated identically.
- This machine has shown repeated transient faults. Its RAM runs a 3600 MT/s
  XMP profile. The user was advised to run a memory test.

**C: consolidated recall** ([protocol](consolidated_recall_protocol.md),
[development report](../reports/consolidated_recall/dev/summary.md)).
**STOPPED at development** by the pre-declared pathology rule. No test seed
was run; test seeds 20201–20212 remain unused.

Descriptive evidence from 3 development seeds:

- A single global EMA removes about 17–19% of stream excess under recurring
  laws, compared with 48% within one law.
- Per-context snapshots add nothing, and recall hurts.
- Even with **oracle** indexing, recalling whole-network snapshots is worse
  than plain online learning, because it discards structure the laws share.
- The implication is that whole-network per-context consolidation is the
  wrong granularity. "Shared slow core plus small indexed codes" is recorded
  as a hypothesis to carry forward, not a tested result.

**Next.** As the proposal pre-declared ("if A1 or C fails, pivot sooner"),
we pivot to Step D: agent-chosen actions, dense consequences and contexts of
graded separability, scored against an ideal agent. The design is in
[step_d_design.md](step_d_design.md), revised after critique.

**D0: world and references** ([report](../reports/commons_d0/summary.md)).

- The world is a new 3-entity resource commons, `src/acp_cl/commons/`.
- O (context-known) and I (same-information) are verified against
  independent brute force, including edge cases. I comes within .001 survival
  per episode of O.
- The problem found: decision value was concentrated in factor A (about .19),
  with B and C at about .02 each.

**D0b: bounded redesign** ([report](../reports/commons_d0b/summary.md)). Its
targets were unreachable, so the best world is accepted with the claim
narrowed.

- The world has 13 actions (sizes 3 and 6), and B now swaps the efficient
  route.
- Relevance: A .086, B .038, C .038.
- C is identified in about 10 episodes. Low separability with high relevance
  is impossible here.
- The one-step-lookahead agent is now the same-information reference (16%
  better than myopic I).
- D2's claim is now about rarity, absence length and held-out combinations.

**D1: competence in stationary contexts**
([report](../reports/commons_d1/summary.md)). **No arm is competent.**

- **Model-based (MB):** passes the context average in 10/12 seeds, but only
  6/12 seeds pass the worst-context condition. Every failure is in A1B1C1.
- **Model-free (MF):** about 3× the bar.
- **The diagnostic found an exploration trap.** In the failing seeds the
  ensemble never tried the key action (2→3 × 6), so it never learned the
  action's value.
- **Bounded memory** costs MB little.
- The replicate pass was bit-identical.

**D1b: exploration remedies** ([protocol](commons_d1b_protocol.md),
amended before the lock; [report](../reports/commons_d1b/summary.md)).

**MB-count** (Thompson sampling plus a fading count bonus c / √(1 + n_a)) is
**COMPETENT**. It passes all three readings:

- **R1 competence:** 12/12 seeds; context average .0096.
- **R2 trap escape:** 24/24 seeds, against MB's trap rate of .46 (exact
  McNemar p = .001).
- **R3 payback:** 12/12 seeds; net cumulative regret −139 against MB.

It has lower online regret than MB in every 512-episode block, including the
first. Randomised priors did not escape the trap. The replicate pass was
bit-identical.

**D2 outcome (2026-10-01): a development-level MECHANISM NO-GO**
([report](../reports/commons_d2/dev/summary.md)). The test was not run.

- The best competitor, an amortised learner that reads only the last episode,
  beat FC on every measure. It matched or beat even the oracle-code ceiling.
- The context is readable from one episode, so a code library has no
  retrieval headroom.
- The bottleneck is the core's competence on factor C and on composition.
- No fix favoured FC unfairly; the one development-informed fix could only
  have helped it.

**The user's decision.** They chose the recommendation: a write-up, then one
final decisive test.

- **Write-up:** [reports/calibrated_continual_learning_findings.md](../reports/calibrated_continual_learning_findings.md).
- **The final test:** D2w ([protocol](commons_d2w_protocol.md)), in which
  per-episode evidence is weak, so identifying a context needs many episodes.
- **Arms:** FC against the strongest amortised learners, including AM-ewma,
  an evidence integrator over the whole stream.
- **Verdicts:** REGIME POSITIVE leads to a CPU benchmark pilot, then cloud.
  RETIRE MECHANISM and UNINFORMATIVE both lead to retiring the mechanism.
- **This is the final round** for the code library in this world family.

**D2w outcome (2026-10-01): infeasible at stage 0**
([report](../reports/commons_d2w/summary.md)). The exact failure patterns
make every relevant context identifiable within about 3–6 episodes, whatever
the sensor noise.

**The code-library mechanism is retired in this world family.** The research
line is paused pending the user's direction. One option discussed is an
applied pivot, such as continual personalisation in AAC or speech-language
work.

**D2: the go/no-go study** ([protocol](commons_d2_protocol.md), revised after
critique).

- **What it compares:** FC (factored codes) against AM\*, the best amortised
  in-context learner. AM\* includes a variant with the same per-stream modular
  prior as FC.
- **Conditions:** equal memory bytes, and an oracle-indexed ceiling OC.
- **Decision rule:** the verdicts STRONG GO, QUALIFIED GO, MECHANISM NO-GO,
  UNINFORMATIVE and COMPETITOR TOO WEAK are mapped to "continue?" and "cloud
  compute?".
- **Supporting reading:** knowledge compounding (visit slope, learning to
  learn), added at the user's request.

## Earlier: post hoc information audit and next-direction proposal (2026-09-28)

A new thread interpreted the qualification STOP. It did so without fitting any
learner and without opening any checkpoint. It created new files only; no locked
file was changed. The STOP stands.

- Audit (exploratory, not decision-bearing):
  [reports/information_audit/README.md](../reports/information_audit/README.md).
- Proposal: [recoverable_context_proposal.md](recoverable_context_proposal.md).

The comparison uses a Bayes ideal observer I. It sees the same query frames and
the same 32-record supports, and knows the simulator and the 5 laws. It passes
all 31 cells.

| Cell | I | Learner |
|---|---:|---:|
| Base A Brier | .045 | .196 |
| Delay cue benefit, stages 1-3 | .010-.013 | about 0 |

A context-free mixture also passes the delay cells. Two independent
reimplementations agree with I within Monte Carlo error. Key findings:

- **The failures are learning-limited, not information-limited.**
- **Base A: partial retrieval, then overwrite.** Support moves the learner only
  about halfway toward mode A. The rest is relearned in the weights and
  overwritten, costing .085 between visits.
- **Delay is not learned.** The cue response is a non-specific bias, and delay
  has little decision value.
- **The largest effect is a steady-state gap within a single context.**
  - Learners plateau at about 2x the ideal Brier, in fresh fits too.
  - This gap is 75-95% of regret.
  - The learner is below the context-free mixture on 4 of the 5 laws.
- **No past threshold was ever calibrated against an achievable ceiling.**

Recommended order, detailed in the proposal:

- **A.** A zero-training retrieval-capacity probe on the saved checkpoints: fit
  only the 12-d context input.
- **B.** A small prospective study of the stationary competence gap.
- **C.** A history-matched retrieval-code test, only if A passes.
- **D.** A pivot to agent-chosen actions and dense consequences.

None of these is approved or locked yet.

## Latest completed: temporal representation qualification (2026-09-28)

The user approved the temporal-representation design in
docs/representation_learning_design.md. Its first required step is now complete:
an outcome-only recurrent reference had to demonstrate learnability both for
individual cumulative dependencies and for five-law blocked interleaving before
the reconstruction candidates could be fit. The qualification used six fresh
counterbalanced seeds18101--18106,24 fits,393,216 arrivals and147,456 optimizer
steps. The independent checkpoint/data/replay audit passed.

**Decision: STOP this fixed recipe before the four-arm candidate comparison.**
The reference passed every fresh-learning criterion, including all stage and
cue-type groups. It failed4 of31 fixed interleaved requirements: base-A endpoint
Brier .195597 exceeds the .12 ceiling, and the arrival-delay cue benefit is below
.002 at cumulative stages1/2/3 (-.000152, .000802, -.000291). The other four
interleaved law endpoints and efficiency/supply-timing cue groups passed.

This means a learner can acquire the dependencies when trained separately, but
the declared reference has not shown the required simultaneous availability when
the five environments recur. It does **not** test the temporal-reconstruction
arms and does not show that reconstruction is ineffective. It also does not
establish that the encoder erased the delay cue; condition inference, finite
history/context, outcome supervision, optimization and the benchmark structure
remain live explanations.

Do not run configs/representation_learning_main.json, sweep alpha/exposure, drop
the delay cue, or relax the threshold to rescue this recipe. The main comparison
is deliberately unrun. A new experiment needs a separately written hypothesis
and protocol. Read reports/representation_learning_results.md and
docs/representation_learning_next_thread_prompt.md before deciding what follows.

Source/config/protocol/scorer locks are recorded in
runs/representation_learning_qualification/manifest.json. The audit restored48
checkpoints, regenerated all12,288 packets/393,216 arrivals and147,456 replay
choices, reconstructed saved boundary forecasts, and left learner/global RNG
state unchanged. It did not rerun optimizer updates or reconstruct ordinary
interior forecasts; see reports/representation_learning_checks/qualification_audit.json.

Engineering fixed the useful-compute baseline at15 updates per batch, but that
allocation was never used in a scientific comparison because qualification
failed. Both smoke runs passed independent scoring/resumption/audits. A Windows
checkpoint-replacement conflict for seed18103 was preserved and documented in
reports/representation_learning_checks/runtime_recovery.json; the unchanged
resume completed the run. The post-work preservation inventory confirms all
51,872 pre-existing files still match. Do not delete or overwrite either
preservation inventory.

## Current design proposal after the user's next-direction request

Read docs/representation_learning_design.md. The recommendation is one fully
trainable predictor with original-context replay and an auxiliary decoder from
the same predictive latent to the already-observed temporal sequence. Compare
outcome replay, final-frame reconstruction and full-sequence reconstruction;
include a useful-compute outcome baseline. First qualify acquisition, then
consider soft projected consolidation in a separate study. That design is now
implemented and its required outcome-only qualification is complete; read the
latest section above for its STOP decision. The sequence/final-frame candidates
remain untested because the reference failed interleaved qualification. Do not
reopen this fixed recipe or the stopped frozen-core branch below.

Source checks informed the proposal: Nested Learning/CMS; CaSSLe; projected
functional regularization. The uploaded report's guarantee claims require
qualifications. Existing replay already preserves preceding support; no bug
was found there. Cue erasure inside the encoder is a hypothesis, not an
established explanation of the prior failures. New observed-trajectory world
modeling is a later, larger benchmark change, not an equivalent-data comparison.

## Latest completed: causal access (2026-09-28)

The approved bounded test is complete. Read reports/causal_access_results.md,
reports/causal_access/development/summary.md and .json, and the independent
reports/causal_access_checks/outcome_review.md. Both architectures are INELIGIBLE
and also fail the raw online screen. **Stop this frozen-core/additive-residual
branch as specified.** Do not launch a window/temperature/width/replay sweep,
drop a cue or add exposure to rescue the recipe. No new experiment is running.

Positive but narrower: recurrent causal-minus-full whole Brier -.039696 (6/6)
and causal-minus-fixed-half -.010461 (5/6), indicating useful feedback access in
this cohort. Conditional is -.004563 versus full (5/6), but +.000787 versus half
(3/6). Both beat joint whole-stream Brier, against weak old-condition functions.

Qualification fails: conditional old-core excess .032531 versus allowed .02
(concentrated in delay, especially 17006) plus novel-reference delay cue gain
.001385 below .002. Recurrent old-core excess .081204, above .02 in every group
and seed; full-novel delay cue gain .001940 below .002. Raw recurrent novel
affected error cost is .005512 versus full, exceeding .005 and worse in all 6.
Both causal online delay cue gains fail (.000948 conditional/.001379 recurrent).
All mean survival guards pass; recurrent novel-minus-joint is -.009867, barely
within -.01, all 6 worse and descriptive interval [-.018311,-.004354]. Do not call
that statistical noninferiority. Current timing cue use is substantial; the
previous cohort's timing weakness is not an architectural impossibility.

The user suggested good-enough retention. This protocol already allows bounded
losses, but frozen evaluation does not test optimal forgetting or consolidation.
Any future direction should establish acquisition/reusable competence before
another controller, then explicitly compare graded protection or consolidation
under continuing new learning and fixed total resources. That is a proposal,
not an additional study run or a validated general continual learner.

Protocol docs/causal_access_protocol.md; source src/acp_cl/causal_access;
configs causal_access_smoke/development; independent scoring script
scripts/summarize_causal_access.py. Main locked 2026-09-28T21:46:19.811857 UTC and
completed 21:58:34.134186 UTC (734.32 seconds). 12 jobs, 120 phases, 466,944 training
arrivals, 175,104 updates, 196,608 frozen arrivals; six one-thread workers, affinity 1365.
Source identity: 80e45794ef9d25b2af551df931f6fa0a99865e1a9bdda5760dc05ec22446ddfa.
47 scientific source and 5 analysis/config/protocol/test files are locked.
Do not modify completed scientific source/config/protocol/analysis or rerun fits.

Verification: 71 targeted tests, 8 scorer synthetic checks, Ruff. Main independent
NumPy analysis reconstructs every raw score/mixture; a separate arithmetic check
matches 816 metrics and all 24 training marginals. Checkpoint audit restores 48
models including gradients, reconstructs 384 boundary packets/all 48 qualification
replicates and 2,592 saved forecast arrays; full model/RNG state unchanged. It does
not independently rerun fitting or reconstruct interior online forecasts.
Plots are under reports/causal_access/development, visually checked. Audit/plot
scripts are verification/presentation tools outside the decision-bearing lock.

Corrected smoke_v2 passed reconstruction/audit/completed-run resumption. The
first smoke's count-dtype validation failure is preserved and documented in
reports/causal_access_checks/engineering.md; no scientific rule changed.
The new pre-work older-file inventory covers 3,314 files / 3,999,557,234 bytes. Its
comparison has 2 inconsistent entries: current files match all earlier canonical
archives/manifests, checked independently; initial inventory discrepancy cause
is undetermined. Preserve both initial/failed inventory files and read
reports/causal_access_checks/preservation_review.md. No old file was rewritten.
Unrelated dirty files are prior work; do not reset/clean. General status docs are
editable; historical scientific records remain protected.

## Previous: core/residual output and context diagnostic

User approved the bounded saved-model diagnostic and asked for candid guidance
on whether the project is spinning its wheels. This turn completed that task;
no additional training was run. Read reports/core_residual_output_results.md,
reports/core_residual_outputs/development/summary.md, and independent outcome
review reports/core_residual_output_checks/outcome_review.md.

Design fixed before scoring: docs/core_residual_output_review.md. New prediction
module src/acp_cl/core_residual_outputs.py sits outside the old scientific package
so historical source locks are unchanged. Runner/summary/plot scripts all use
the core_residual_outputs suffix. Main locked 2026-09-28 03:26:05.367435 UTC,
completed 03:26:22.804618 UTC: 12 jobs, 48 states, 384 contexts, 1,536 component
forecast tensors, zero updates, six one-thread workers. Both original models,
all six seeds 16001--16006, joint/separate only, four endpoint panels.

Both exploratory GO allocation heuristics pass. At identical novel-end weights
and refreshed supports, separate full-minus-core old Brier = +.012873 conditional
(5/6 positive), +.084727 recurrent (6/6); core-minus-full novel focus Brier =
+.018927 (5/6), +.029784 (6/6). Conditional's exceptions differ (only 4/6 have
both signs). These are permissive effort-allocation rules, not a successful
algorithm, independent confirmation, or reversal of either failed primary screen.

Critical absolute caveats: recurrent frozen core old Brier .247546; full .332273
at entry but .090948 after return. Core-for-old is therefore not a general rule.
After return, novel full .156058 vs core .107149, but core cue benefit .000220;
selecting it does not restore learned novel-cue behavior. Conditional timing
cue benefit remains .000051 full / .000278 using core routing. Do not assert
internal representation absence from these output ablations. Both separate
arms beat joint at actual return entry in all 6, so recurrent bad return AUC
develops after entry. Identical query/history at novel end and return entry
preclude detecting an unannounced switch before informative outcomes arrive.

Verification: 29 new tests and Ruff pass; all 384 full forecasts byte-match their
archived originals; full learner and global-RNG signatures unchanged; 96 frozen
core repeated-input checks (72 across saved endpoints); independent NumPy scoring
of 12,288 raw metrics. Portable
rescoring outside repo reproduces JSON/MD byte-for-byte and rejects 5 corruption
cases. Corrected smoke_v2 passes 40 rows/8 states/1,280 metrics and is ineligible.
Failed initial smoke lock remains: tuple-vs-JSON-list metadata normalization was
repaired before main scoring, with no scientific threshold changes. Proofs and
old-science preservation inventory are under reports/core_residual_output_checks.

The historical docs/core_residual_output_followup.md draft led to the completed
causal-access study and stopping decision above. User wants honesty about return
on effort; do not portray diagnostics/audits as breakthroughs or general learning.

Do not alter completed scientific sources, archived reports, protocols or runs.
General status docs remain editable. Existing unrelated working-tree edits are
prior work; do not reset/clean. There are no unfinished training processes.

## Latest completed: core/residual architecture comparison

The user approved the next experiment and asked to resume after usage reset.
Protocol: docs/core_residual_protocol.md. Main/smoke configs: configs/core_residual_*.json.
New implementation: src/acp_cl/core_residual; tests/test_core_residual*.py;
scripts/summarize_core_residual.py, audit_core_residual.py, plot_core_residual.py,
verify_core_residual_archive.py. Read reports/core_residual_results.md and
reports/core_residual/development/summary.md; ledger and verification proofs
are under reports/core_residual_checks.

The full main completed 2026-09-28 02:39:04 UTC, 672.57 seconds after locking:
12 prefix fits, 36 continuing learned trajectories, 12 fresh references,
168 phases and 207,360 updates on seeds 16001--16006, both architectures,
six single-thread workers, affinity 1365. No fits or audit processes remain active.
Scientific source identity is
c5ac7b538363930c68c999b60f5d26244eaf5b5eb676c8cf562d1ca354109a3b.
Do not rerun completed fits or alter source/config/protocol/scientific analysis.

Both primary screens FAIL despite every fresh qualification group passing.
Novel separate-minus-joint Brier AUC: conditional +.016853 [−.006848, +.042192],
4/6 improve; recurrent +.003917 [−.000877, +.009242], 2/6. Conditional improves
return AUC −.015281 in all 6 but fails novel survival and post-return novel endpoint.
That endpoint gap is not greater forgetting during return: separate started
return worse and its within-arm error increase is smaller than joint's.
Recurrent fails return AUC (+.033726) and both survival guards. Its favorable
mean post-return novel endpoint is heterogeneous (3/6 favor, interval spans 0).
Recurrent feature attribution passes (−.012130 versus fixed_features, 6/6 improve);
conditional attribution fails consistency (4/6). Do not use it to rescue primary
failure or promote a secondary arm. Timing-cue failures are descriptive, not
grounds to discard seeds or qualify only a favorable subgroup.

All 268 scoped tests pass (195 scientific, 73 portable), two inapplicable cases skip,
plus Ruff. Main 1,026 file-integrity checks pass. Tensor audit checks 168 before/
168 final checkpoints, 36 copied-prefix forks, 12 fresh initializations, 17,280
packets and 6,912 evaluations; 4,920 evaluator forecasts plus 168 first-packet forecasts
are reconstructed, 1,992 intermediate calls have physics/arithmetic/provenance
checks only. No independent ordinary retraining. Separate NumPy review verifies
1,108 bootstrap distributions and all 26 gates (max difference 2.22e−16). Smoke 66
and development 366 artifacts are sealed and pass portable verification both
in place and outside the repository. Completed-run resume changes no bytes.
All 1,595 earlier scientific files, including 1,313 report artifacts, are preserved.

Architecture: common core learned from random initialization, then one zero-output
raw-input residual is allocated before maintenance. Joint trains both; separate
freezes the core; fixed_features also freezes the random residual visual encoder
but still trains heads/context/readout. Preserve core Adam/history/replay. Novelty
and return trigger no resets. Exact state signatures include gradients and path
work; boundary evaluations must be state-preserving. Local audit reconstructs
boundary forecasts, streams/replay/physics and tensor state without claiming full
training replication. Portable verifier/test are prospectively excluded from
science locks and separately sealed. Do not edit older experimental code.

The bounded core/residual/context direction was completed in the newer section
above. Its exploratory findings do not replace this study's failed primary
decisions. Do not extend this cohort's training or tune another architecture.

## Latest completed state-and-environment diagnostic

Read reports/rehearsal_state_results.md and reports/rehearsal_state/diagnostic/summary.md.
All 24 fresh trajectories, 48 assessments and 3,456 forks completed at
2026-09-28 01:10:26 UTC (592.62 seconds including lock launch). Both fixed
replication screens fail: conditional T=-.000439, 6/12 positive; recurrent
T=+.000825, 8/12 positive. Thresholds were mean >= .0005 and at least 9/12
positive seed means. Both descriptive intervals include zero. Do not revise
criteria, pool architectures or select a component from secondary tables.

The eight cells independently substitute earlier/later weights, complete Adam
state and causal anchor. Current and returning branches share observations,
performed actions and initial actual history, then receive their own causal
feedback. No diagnostic changes ordinary replay/history; only current-branch
data enters ordinary training. Zero-update references are outside the candidate
pool. Main seeds are 15001--15012, two models, two assessments each; ordinary
trajectories learn 18,432 arrivals with uniform replay capacity 16 and six one-thread
workers, affinity 1365. Source identity:
80e0110241041c303e869b8f85c274a6262aaac4ecc8d399bb837a5c1ddf9a2d.

Conditional selection is better relative to uniform at 111, but selected absolute
error rises from 000 to 111 while uniform worsens more. At111, chosen-minus-zero
current Brier is -.0000081 conditional and +.0004658 recurrent; both uncertain.
Reverting recurrent weights slightly improves mean ranking contrast while
worsening absolute chosen error. The zero references already used ordinary
replay; this is no-extra-rehearsal, not no-replay. Return candidate coverage is
10/24 assessments per model; retain all 24. Component and rank findings remain
descriptive. No rollback, cached score or adaptive memory policy earns promotion.

All 172 tests pass (60 mechanism, 17 runner, 23 analysis, 20 tensor audit, 52 portable),
plus scoped Ruff. All 2,334 full-run files pass integrity. Independent reconstruction
passes 240 before / 240 final checkpoints, 48 diagnostic boundaries, 3,456 forks,
96 zero copies, 120,576 forecasts and 12,096 loss-instrumentation forwards. It does
not independently retrain every ordinary update. Independent NumPy arithmetic
reproduces 480 arrays, 54,558,720 probabilities and 3,408 seed/bootstrap statistics,
maximum discrepancy 7.22e-16. Source/protocol/analysis are immutable. Both cohort
archives are sealed and verify both in place and outside the repository; the portable verifier/tests were prospectively excluded
from the scientific lock and sealed separately. Verification outputs, ledger,
pre/post-outcome reviews and preservation checks are under
reports/rehearsal_state_checks. All 979 previous scientific files, including 720
prior report files, must remain byte-identical. Do not rerun completed fits.

The next direction proposed at that time was to retire cached-score/rollback adjustments and compare
a learned reusable core plus an adaptive feature pathway with the same-capacity
architecture trained jointly. Keep uniform replay and learning exposure matched;
start from a common prefix learned from random initialization. Give the adaptive
path access to raw observations or other information beyond a lossy frozen
embedding. Test actual continuing trajectories with a late unseen dependency,
new acquisition, valid-old retention and returns. That comparison was subsequently
completed as the core/residual study above, and failed both primary screens.
An extra-rehearsal dose study is needed only if the new mechanism adds such work.
Frozen-core success alone would not establish indefinite new representation
learning, noise robustness, longer planning or compounding lifelong advantage.

## Latest completed predictive-value diagnostic

Read reports/predictive_value_results.md and reports/predictive_value/diagnostic/summary.md.
All 24 trajectories, 48 assessments and 1,296 forks completed with fresh seeds
14001--14012 and both architectures. Six single-thread CPU workers used actual
affinity 1365, up from the prior four/mask 85. Scientific fitting took about eight
minutes. The ordinary learner kept uniform 16-packet replay and learned 20,992
arrivals per trajectory. Candidate selection never changed its trajectory.

Both architectures fail both fixed tiers. Reapplied selected-minus-uniform
Brier is -.0000301 conditional / +.0002243 recurrent locally and +.0009557 /
+.0015791 on returns. Improved seed counts are 8/12, 5/12, 7/12, 3/12. Both return
accuracy-control gates also fail; all survival guards pass. Primary intervals
cross zero, so failure is a research allocation decision, not proof of harm.
Stop expanding this fixed score; do not tune dose/pool/windows on these seeds.

Frozen original predictors show more near-window selection benefit than fresh
rehearsals from updated parent states. The comparison changes weights, Adam
state and latest anchor together; no isolated causal explanation is established.
Return-origin candidates exist in 19/24 assessments per model; all assessments
remain included. Original-error selection is not a validated fallback.

All 140 scoped tests pass (37 mechanism, 12 runner, 32 summary, 18 audit, 41 portable).
All 1,326 run files pass integrity. Independent tensor audit recreates all 1,296
forks and 34,560 diagnostic forecasts, checking 336 before and 336 final checkpoints.
Independent statistical review reproduces 272 paired metric summaries. Portable
verification passes both in place and from an isolated copied archive. It uses
no live scientific source or model pickle. Full ordinary training is not
independently retrained; audit scope is explicit in reports.

Source/config/protocol/analysis locks and the two cohort archives are immutable.
The portable verifier/tests were explicitly excluded from the scientific lock
and sealed separately. Source/config/runtime identities are in the manifests.
All 655 pre-existing report files are preserved; the before map is under
reports/predictive_value_checks/preservation_before.json. Engineering history
and independent review artifacts are in the same checks directory. A separately
labeled retrospective rank-stability calculation uses existing records only;
it cannot change the primary criteria and involves no further fitting.

The broader objective remains learning new representations from scratch while
retaining useful conditional relationships under bounded resources. This fixed
synthetic diagnostic does not establish memory-policy improvement, automatic
transition discovery, noise robustness, longer planning or compounding benefit.

## Prior selective-update follow-up (historical handoff)

The user authorized reconsidering the full history, useful combinations and
continued experiments; also suggested weighting retention by predictive value.
Agent help remains authorized. Its report is reports/adam_revision_results.md.
The full synthesis is docs/research_synthesis.md. The value proposal in
docs/predictive_retention.md subsequently led to the completed study above.

The matched-return diagnosis completed 1,428 evaluations across six old pilot
seeds per model and three returns. With identical fresh target support,
current-start minus prefix Brier x100 is +9.461 conditional and +20.033 recurrent;
versus prior-return weights, +6.233 and +25.232. Every seed is worse for both
references and architectures. New support helps without fully recovering
historical predictions; training during the return recovers much of the deficit.
No prior failed prospective screen was changed. The initial strict probability
validator rejected a float32 excursion; original lock/ZIP/failure survive under
reports/return_context. The documented v2 correction permits bounded rounding
without clipping. Five rows exceed one by one epsilon; 25 diagnostic tests pass.
Completed diagnosis: reports/return_context_v2/summary.md.

The selective-update cohort completed all 12 prefixes and 72 episodes with
seeds 13001--13006, both architectures, first introduction only. Each prefix
forks .5 ordinary, .75 ordinary, .5 projected, .75 projected and .75 shrink-only;
a sixth fresh arm qualifies learnability. All keep uniform 16-packet replay.
The projection changes actual Adam displacement against replay BCE while
keeping ordinary moments. Only the .75 projected arm is the primary candidate.

Both architectures PASS all fresh qualification groups and FAIL the primary
and separate protection-attribution screens. Combined-minus-reference:

| Outcome | Conditional | Recurrent |
|---|---:|---:|
| Acquisition Brier x100 | +0.499 [0.315,0.687] | +0.547 [0.310,0.795] |
| Improved acquisition seeds | 0/6 | 0/6 |
| Valid-old mode 0 Brier x100 | -1.237 | +1.810 |
| Valid-old mode 1 Brier x100 | +0.679 | +1.106 |
| Survival AUC percentage points | -0.509 | -0.291 |

Conditional fails its mode 1 retention limit; recurrence fails both. These are
mean-based development decisions, not formal noninferiority conclusions.
Survival declines within its one-point tolerance. Current emphasis alone
worsens acquisition in every seed (+0.575/+0.480), whereas protection alone is
near neutral/small benefit (-0.002/-0.052), below the primary gain requirement.
Do not promote that ablation or tune the failed recipe on these outcomes.
Full results: reports/selective_updates/development/summary.md and summary.json.

All 94 selective implementation/analysis/audit tests and 11 portable-verifier
tests pass. Full development audit verifies 12 prefix, 72 starting, 72 final
checkpoints; all 246 stable files pass. Engineering smoke completes 12 episodes and
passes its audit, with 46 stable files. Neither audit independently retrains all
intermediate updates. Portable verifier independently recomputes the critical
recorded scores, paired intervals, qualification and screens without loading
pickles. See reports/selective_updates/README.md and the engineering ledger
under reports/adam_revision/. The 40-artifact portable seal passes; all 338
older artifacts and all 292 stopped smoke/development files remain unchanged.
The packaging check initially rejected a harness-generated bytecode cache;
removing only that cache and using Python -B preserved the seal and passed.
See reports/adam_revision/portable_verification_attempts.json.

Scientific lock: 2026-09-27T23:05:43.740166UTC; source SHA
28c72eaf806a2059ef4df297668c1989db5dddebec88cd6a6432a189f827d38e,
config SHA1822b189f388675a1fc16bd207758691fe17e10416234fd1beea556876a09ce7,
runtime SHA4ea3ea1f87cff2e70c31f788d2332c81583a711f4b402a26f9a1265dc448a77f.
Keep all scientific source, locked analysis/protocol/configuration and sealed
archives immutable. Source/analysis ZIPs preserve the exact experiment.
The standalone packaging verifier was added after that lock and changes no
scientific decision. Training session 4035 and audit session 51428 exited 0;
no training process remains active. No commits or external publication requested.

Next direction: keep ordinary uniform replay as the reference. Before changing
retention, test whether a bounded rehearsal-benefit estimate predicts value on
a separate future window and when dormant conditions return. Include uniform
and stored prequential-error controls; account for extra forks/scoring work,
rare conditions, learner maturity and noise. This is only a proposed diagnostic,
not a locked protocol or completed experiment. All previous failed results
remain failures. No lifetime, long-horizon or compounding success is claimed.

The sections below describe the previously completed evidence study.

## Current result

The requested repeated-evidence study is complete and audited. Read
reports/evidence_consolidation_results.md and the full tables under
reports/evidence_consolidation/pilot/summary.md.

Both architectures pass every pilot fresh-model qualification group and fail
the prospective acceptance-rule screen. Sustained minus immediate Brier x100:

| Outcome | Conditional | Recurrent |
|---|---:|---:|
| Whole stream | +0.169 | +0.280 |
| Final matched cycle | +0.083 | +0.267 |
| Original environment returns | +0.781 | +1.624 |
| Noisy periods | -0.205 | -1.101 |

Whole-stream error increases in all six seeds per architecture. The mean
late-minus-early advantage is slightly positive (+0.066 and +0.070 scaled),
but both intervals include zero. Costs are not recovered by stream end.
Sustained beats fixed delay in four of six seeds in each architecture, with
small uncertain gains below the required magnitude. Acquisition, valid-old,
noise and survival guardrails pass; lifetime gain and return requirements fail.
Development's two-seed arrival-delay qualification failures remain failures.

Immediate use of the uniformly replay-trained learner remains the reference.
Do not tune the gate on completed seeds or promote it as a successful method.

## What was tested

Working title Adam; optional expansion Adaptive Dynamics and Associative Memory.
Keep acp_cl and repository paths for compatibility. The general objective is
learning representations from scratch under bounded memory and computation.

The experiment shares one continuously trained draft across four deployment
policies. An eight-packet future-data window evaluates a frozen proposal against
frozen committed weights before feedback. Periodic always accepts; single uses
its first gain >.002; sustained needs mean gain >.002 and six positive packets.
Accepted weights are exactly those evaluated. Immediate uses live draft weights.
No state resets or privileged condition labels enter learning or acceptance.

Each main trajectory contains 151,552 arrivals: a 4,096-arrival prefix followed
by eighteen blocks of 8,192. Three new dependencies precede three cycles of
clean conditions, noise, recovery, original-base return and revised cue meaning.
Persistent noise is used in cycles one and three; burst noise in cycle two.
Noise arrangement and learner age were not counterbalanced. Physical labels
still end at H=12. Every packet is scored before training on evaluator-only
clean physical outcomes; gates only see reported terminal outcomes.

This tests deployment/acceptance, not different representation-learning rules:
all policies share the same draft. It cannot demonstrate compounding
representation learning, acquisition of new cues late in life, farther-future
planning, or general identification of physical change versus reporting bias.

## Reproducibility and verification

Protocol: docs/evidence_consolidation_protocol.md.
Sources: src/acp_cl/evidence_consolidation/.
Configs: configs/evidence_consolidation_{smoke,development,pilot}.json.
Runs: runs/evidence_consolidation_{smoke,development,pilot}.
Portable reports: reports/evidence_consolidation/.

Both source/config/protocol locks predate development fitting. The 39-file
analysis lock predates pilot fitting. No locked file changed. Shared scientific
source SHA256:
e95a0b9823ed139a099a547f5b5c96196ba9266b09c1e3ddbe62b68186dff11d

Development seeds 11001--11002: 4 trajectories, 72 blocks, 12 fresh references,
4 prefixes; audit passed 172 checkpoint states and 2,368 windows, with 264
stable files validated. Pilot seeds 12001--12006: 12 trajectories, 216 blocks,
36 fresh references, 12 prefixes; audit passed 516 states and 7,104 windows,
with 784 stable files validated. All planned fits completed without retries,
interruptions or exclusions. The pilot finished at 21:33:11 UTC. Four single-
thread CPU workers used affinity mask 85. No GPU or ongoing training remains.

Engineering checks passed 818 distinct tests with two existing Windows symlink
skips. Ruff and whitespace checks passed. Independent recomputation from raw
pilot JSON reproduced primary means, paired contrasts, bootstrap intervals and
screen decisions. All 273 artifacts in six older archives and their scientific
source files retain their hashes; before/after records are in the new archive.

The audit reconstructs streams/replay/history, budgets, exact checkpoint
continuity, endpoint probes, first-packet prequential scores and gate gains,
and every recorded acceptance decision and snapshot lineage. Intermediate
window weights are not all saved; not every intermediate gain or prediction
was independently rescored or retrained. This is not an independent replication.

Useful commands (see docs/reproduction.md):

    python scripts/verify_evidence_consolidation_archive.py --input reports/evidence_consolidation
    python scripts/plot_evidence_consolidation.py --input reports/evidence_consolidation/pilot

Do not overwrite the prospective lock or older sealed archives. The portable
archive contains original raw records and hashes, source ZIPs, protocol/config
locks, final analysis, audits and figures. Local trusted model checkpoints
remain in ignored runs/. No commit or external publication was requested.

## Next research question

Before another large experiment, separate deployed-weight age, retained
predictions and context inference on matched return queries. Then consider a
bounded mechanism that uses evidence to direct selective changes during
learning without delaying every deployed update. Any new candidate needs fresh
development data, explicit resource controls, and independent confirmation.
To test compounding acquisition, introduce matched novel dependencies both early
and late, counterbalance their order, and include matched fresh-weight controls.

The broader project remains an experimental research direction, not a proven
new algorithm. docs/adam_related_work.md records close precedents including
Paired Learners (2008), SEA (2001), PeARL, CLS-ER and COIN.
docs/evidence_noise_limits.md explains why repeated evidence against corrupted
reports does not by itself identify changes in physical dynamics.

The user authorized this study, novelty/horizon assessment and agent help.
Agents completed bounded design, test, literature and independent review tasks.
No inline visualization was created because browser preview was unavailable;
verified PNG/SVG/PDF scientific figures are linked in the report.
