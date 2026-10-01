# Cross-study evidence map (information audit)

**Status: POST HOC / EXPLORATORY / NOT DECISION-BEARING.** This is a read-only
reading of existing reports, protocols and saved summary JSON. It trains nothing,
loads no checkpoint, changes no locked file, and alters no recorded decision.
Every failed screen stays failed; every STOP stays in force. Pooled statistics
and power estimates below are new descriptive calculations from saved JSON
arrays. They are not prospective criteria and cannot rescue or reverse anything.
Law identities, clean physical outcomes and counterfactual outcomes are
evaluator-only throughout the project. Any "ideal" or "clairvoyant" quantity
quoted here is an analysis reference, not a learner input.

Conventions: Brier values are **raw units** unless marked x100. "Cue benefit" is
flipped-query minus correct-query affected-subset Brier, so positive means the
cue is used helpfully. C = conditional (evidence-routed four-head) learner.
R = qualified recurrent learner (interaction features, mean-pooled GRU, 12
updates per batch). Cue 0 = directional efficiency, **cue 1 = arrival delay**,
cue 2 = supply timing. Base mode A = `Law(seed % 2)`; B = the opposite mode.
All cumulative cue laws use base B (`src/acp_cl/acquisition/world.py:125`).

Sources swept: all `reports/*.md`, all `reports/*/summary.md`, the relevant
`summary.json` files, all `docs/*.md` (including `ADAM_HANDOFF.md`) and
`README.md`. Inline extraction used `python -B` with `PYTHONDONTWRITEBYTECODE=1`
and read only JSON. No script was added.

---

## A. Every reported arrival-delay (cue 1) cue-benefit value

### A1. Fresh-model / qualification values (seed-mean of the delay cell)

| Study (seeds), date | Arch | Exposure (arrivals per fit) | Seeds in cell | Delay cue benefit | Threshold | Result |
|---|---|---:|---:|---:|---:|---|
| Acquisition dev (211-212), 09-27 | C | 2,048 | 2 | .002908 (seeds .000938, .004878) | >= .002 | pass |
| | R | 2,048 | 2 | .002499 (.001692, .003306) | >= .002 | pass, narrowest dev margin |
| Acquisition comparison (7001-7006) | C | 2,048 | 6 | **.001939** | >= .002 | **FAIL by .000061 (3%)**; 4/6 seeds < .002, 1 negative |
| | R | 2,048 | 6 | .002587 | >= .002 | pass; 2/6 seeds < .002 |
| Training-state dev (411-416) | C | 4,096 | 6 | **.001764** | >= .0025 (dev) | **FAIL**; 3/6 < .002 |
| | R | 4,096 | 6 | **.000707** | >= .0025 | **FAIL**; 5/6 < .002, 2 negative |
| | C | 8,192 | 6 | .004232 | >= .0025 | pass; 2 seeds negative |
| | R | 8,192 | 6 | .004222 | >= .0025 | pass |
| Training-state comparison (8001-8006) | C / R | 8,192 | 6 | .005120 / .005016 | >= .002 | pass / pass |
| Replay decomposition (9001-9006) | C / R | 8,192 | 2 | .005326 / .009882 | >= .002 | pass / pass |
| Replay policy (10001-10006) | C / R | 8,192 | 6 | .004752 / .003653 | >= .002 | pass / pass |
| Evidence consolidation dev (11001-11002) | C / R | 8,192 | 2 | **.001717 / .001121** | >= .002 | **FAIL / FAIL** |
| Evidence consolidation pilot (12001-12006) | C / R | 8,192 | 6 | .005158 / .006705 | >= .002 | pass / pass |
| Selective updates (13001-13006) | C / R | 8,192 | 2 | .006845 / .007788 | >= .002 | pass / pass |
| Core/residual fresh reference (16001-16006) | C / R | 8,192 | 2 | .008196 / .006975 | >= .002 | pass / pass |
| Causal access novel specialist (17001, 17006) | C | 8,192 | 2 | **.001385** (.003059, -.000289) | >= .002 | **FAIL** |
| | R | 8,192 | 2 | .004437 (.005495, .003378) | >= .002 | pass |
| Representation fresh fits (18101-18106) | R | 8,192 | 6 | .003365 [.001134, .005632]; seeds .004048, .008184, -.000489, .004865, .003431, .000150 | >= .002 | pass; 2/6 seeds < .002 |

The predictive-value and rehearsal-state studies had no fresh qualification.
The return-context and core-residual-output diagnostics did no training.
Pre-acquisition worlds (persistence, conditional, contextual) had no cue-flip metric.

### A2. Continuing-learner endpoint values (terminal, correct/fresh support; descriptive, no gate unless stated)

| Study | Exposure | Arm | C | R |
|---|---|---|---:|---:|
| Acquisition (6 seeds) | 2,048 | entry, before learning delay | .000853 | .001513 |
| | | continue | .002247 | .002744 |
| | | frozen visual features | .002281 | .000789 |
| | | reset optimizer + replay | .003317 | .005824 |
| Training state (6) | 8,192 | entry | .003202 | .003354 |
| | | continue / Adam reset | .004028 / .005659 | .004653 / .004391 |
| | | replay reset / both reset | .005645 / .006446 | .005655 / .003586 |
| Replay decomposition (2) | 8,192 | keep | .002715 | .004812 |
| | | delete / rebase / delete+rebase | .009470 / -.000800 / .013347 | .006335 / .003266 / .007873 |
| | | RNG reset / full reset | .005980 / .008606 | .004395 / .005037 |
| Replay policy (6) | 8,192 | uniform / recent / split | .002413 / .004538 / .002668 | .002325 / .003176 / .003173 |
| Selective updates (2) | 8,192 | reference / combined | .002628 / .002918 | .004814 / .001343 |
| | | current / protected / shrink | .001744 / .001491 / .000562 | .002513 / .005207 / .003242 |
| Core/residual (2) | 8,192 | joint / separate full / fixed features | .004727 / .003042 / -.000117 | .003782 / .002487 / -.001077 |
| Core/residual outputs (2) | novel end | separate core only / residual only | -.000991 / .006416 | -.000530 / .004260 |
| | after return | joint full / separate full | .001072 / .003847 | .002607 / .000449 |
| Causal access separate full (qualification, 2) | 8,192 | full novel predictor | .002893 pass | **.001940 FAIL (>= .002)** |

The evidence-consolidation reports record no continuing-draft delay value.

### A3. Online values (causal access frozen alternating stream, delay seeds 17001/17006)

Sixteen frozen blocks of 1,024 arrivals; only the causal mixer is gated (>= .002 overall and in every cue group).

| Policy | C | R |
|---|---:|---:|
| Always core | -.000338 | .000645 |
| Always full | .001439 | .001650 |
| Fixed half | .000600 | .001143 |
| **Causal mixer (gated)** | **.000948 FAIL** | **.001379 FAIL** |
| Jointly trained composite (control) | .006760 | .004609 |

Every separate-derived policy stays below .002. The continuously jointly
trained control stays above it, although its old-stream error is worst (see B).

### A4. Interleaved values (representation qualification, recurrent only)

Blocked interleaving cycles baseA, baseB, s1, s2 and s3 in 1,024-arrival blocks,
eight times (8,192 arrivals per law, 40,960 total).

| Cell | Seeds | Mean [95% boot] | Per-seed | Threshold | Result |
|---|---:|---:|---|---:|---|
| stage 1 / delay | 2 | -.000152 [-.001405, .001101] | .001101, -.001405 | >= .002 | FAIL |
| stage 2 / delay | 4 | .000802 [-.001974, .003578] | -.000415, .002731, .004424, -.003533 | >= .002 | FAIL |
| stage 3 / delay | 6 | -.000291 [-.002022, .001362] | -.001158, -.002693, .001963, .001203, -.003161, .002101 | >= .002 | FAIL |

Efficiency cells passed (.0096, .0058, .0056); timing cells passed (.160, .140, .121).

### A5. Dispersion and structural leverage (new post hoc calculation)

- **Pooled fresh delay values at 8,192 arrivals.** 74 seed-level values from
  training-state dev/comparison, replay decomposition/policy, evidence dev/pilot,
  selective updates, core/residual, causal-access specialists and representation
  fresh fits. Mean .00487, median .00485, SD .00321. **17.6% of individual seeds
  are below .002; 6.8% are negative.** The pooled within-cohort SD is .00309.
  Resampling pooled seeds gives P(cell mean < .002) = .105 for n=2, .031 for n=4
  and .011 for n=6. A normal approximation with the within-cohort SD gives
  P(fail) = .18 (n=2) and .056 (n=6) at a true mean of .004. At .003 it gives
  .32 and .21.
- **Cue magnitudes in the same fresh fits.** Efficiency is about .006 to .03,
  delay about .002 to .010, and timing about .14 to .19. One .002 threshold
  applies to all three.
- **Structural leverage** (`reports/acquisition/world_check.json`, a
  pre-training audit). Share of cases whose realized outcomes change when the
  cue is flipped at introduction: efficiency .178 (.148-.202), **delay .071
  (.061-.081)** and timing .663 (.653-.673). Clairvoyant action value is 0, 0
  and .046. The delay dependency therefore touches the fewest outcomes. No
  document converts this into an attainable cue-benefit ceiling (see C).
- **Binding cue.** Delay is the minimum-margin cue group in **every one of the
  13** cue-world qualification events (A1). The marginal-gain criterion (>= .02)
  never binds: fresh minima are .133-.148, about 7x the threshold.

---

## B. Old / minority / returning-mode competence: recurrent versus conditional

"Non-current mode" is the base mode not most recently trained. Values come
with correct (evaluator-selected) support unless marked "online" or "actual
history".

| Study (seeds) | Metric | C | R | Source |
|---|---|---:|---:|---|
| Conditional pilot (3001-3008) | Return survival after 32 feedback records, no updates | 70.62% (pooled 59.02%, oracle 72.17%) | no R arm | reports/conditional_results.md |
| Contextual (5001-5008) | Old-mode return, no weight updates | **73.23%** | **66.13%** (C-R +7.10 pp [4.49, 9.60], 8/8) | reports/contextual_results.md |
| | Valid old modes after new learning | 72.62% | 68.52% (+4.11 pp, 8/8) | |
| | R return over no transfer (adequacy >= 5 pp) | — | +5.19 pp (passes by 0.19) | |
| Acquisition (7001-7006) | Continue valid-old change (x100) | +0.645 | +1.784 | reports/acquisition_results.md |
| | Terminal valid-old error, continue / state reset (x100) | 8.795 / 10.911 | 17.118 / 23.416 | |
| | Return branch Brier before -> after (x100) | 12.657 -> 7.846 | **29.788** -> 7.697 | |
| | Return survival before -> after | 69.12 -> 73.91% | 58.53 -> 73.60% | |
| Training state (8001-8006) | Starting valid-old error (x100) | 11.642 | 19.884 | reports/training_state_results.md |
| | Continue final valid-old error / damage (x100) | 13.934 / +2.291 | 22.681 / +2.797 | |
| | Return-branch entry Brier, continue arm | .1886 | **.3702** | training_state/summary.json |
| | Return-branch terminal valid-old, continue / replay reset (x100) | 13.232 / 19.606 | 16.457 / 24.675 | |
| Replay decomposition (9001-9006) | Keep-arm valid-old deterioration from prefix (x100) | +1.165 | **+7.941** | reports/replay_renewal_results.md |
| Replay policy (10001-10006) | Uniform final valid-old error (x100) | 10.837 | 23.201 | |
| | Return-branch entry Brier, uniform | .1885 | **.3676** | replay_renewal/policy/summary.json |
| Evidence pilot (12001-12006) | Online return-block error, immediate policy | .0909 | .1113 | evidence_consolidation/pilot/summary.json |
| | Still-valid clean-support error, better / **worse** mode (108 block endpoints) | .083 / **.150** | .088 / **.332** | same, valid_after_modes |
| | Sustained-minus-immediate return cost | +.00781 | +.01624 (limit .005) | |
| Matched-return diagnosis (return_context_v2, EC pilot seeds) | Fresh-target-support Brier at return start: current draft | .165 | **.356** | return_context_v2/summary.json seed_means |
| | Same support: prefix weights / previous-return weights / return-end weights | .070 / .083 / .082 | .156 / .097 / .104 | |
| | Current minus prefix; current minus previous return (x100) | +9.461; +6.233 | +20.033; +25.232 (every seed worse) | reports/return_context_v2/summary.md |
| | Return end minus return start (x100) | -8.294 | -25.228 | |
| | Context penalty: actual history minus first 32 return records (x100) | **+20.921** | +3.648 | |
| Selective updates (13001-13006) | Non-current mode at entry | .0887 | **.2277** | selective_updates/development/summary.json |
| | Non-current mode at end: reference arm / fresh (never trained on it) | .1347 / .2774 | **.3483** / .4195 | |
| | Current mode at end, reference | .0718 | .0776 | |
| | Reference minus fresh valid-old (x100) | -7.335 | -3.366 | reports/adam_revision_results.md |
| Predictive value (14001-14012) | Uniform-reference error, next window / return window | .0768 / .1448 | .0783 / **.3470** | reports/predictive_value_results.md |
| Rehearsal state (15001-15012) | Return-branch error, first packet / remaining packets | .4150 / .1764 | .4208 / **.3524** | reports/rehearsal_state_results.md |
| Core/residual (16001-16006) | Prefix end: current / non-current mode | .0703 / .0904 | .0959 / **.2346** | core_residual/development/summary.json |
| | Novel end, non-current mode: joint / separate | .121 / .110 | .375 / .344 | |
| | Return entry, non-current mode: joint / separate | .134 / .100 | .348 / .319 | |
| | Return all-case Brier AUC (x100): joint / separate / fixed features | 10.93 / 9.40 / 10.07 | 11.99 / **15.36 / 25.26** | reports/core_residual_results.md |
| Core/residual outputs (16001-16006) | Old at return entry: core / full | .085 / .098 | **.248 / .332** | reports/core_residual_output_results.md |
| | Old at return end, full | .076 | .091 | |
| **Causal access (17001-17006)** | Old-core qualification Brier / old specialist | .1012 / .0687 | .1524 / .0712 | causal_access/development/summary.json |
| | **Old-core excess (allowed <= .02)** | **.0325 FAIL** | **.0812 FAIL** | |
| | Excess by cue group 0 / 1 / 2 | .0101 / **.0746** / .0129 | **.1069 / .0793 / .0574** | |
| | Excess per seed 17001..17006 | .0004, .0122, .0135, .0186, .0017, **.1488** (5/6 within .02) | **.1148, .0359, .0790, .1404, .0733, .0438 (0/6 within)** | |
| | Online old-stream Brier: core / full / half / causal / joint | .1085 / .1107 / .0967 / .0995 / .1652 | .1778 / .2677 / .2054 / .1834 / **.3520** | |
| Representation (18101-18106), R only | Interleaved endpoint Brier: base A (minority, 1 of 5 laws) / base B / s1 / s2 / s3 | — | **.1956 FAIL (<= .12)** / .1121 / .0987 / .0956 / .0973 | representation_learning/qualification/summary.json |
| | Base-A per seed | — | .267, .215, .115, .287, .101, .191 (2/6 <= .12) | |
| | Base-A marginal gain (>= .02) | — | .0326 [-.0221, .0879]; 2/6 seeds negative (other slots >= .117) | |

**Pattern (post hoc).** Across 11 cohorts (acquisition through
representation), R's error on the non-current, minority or returning base mode
is .15-.37 Brier, mostly .23-.37. That holds with correct support or at return
entry. An untrained model scores about .25 on the same probes. Fresh-fit
training marginals are about .22 (inferred from model Brier plus marginal gain).
C's non-current mode with correct support sits at .09-.15, and its return entry
at .13-.19. The R deficit is already present at prefix end, after four
alternating A/B blocks (.2346 versus .0904), before any new cue appears. R's
return entry is .30-.37 in six studies (acquisition .298, training state .370,
replay .368, matched return .356, predictive value .347, rehearsal .352). R
recovers through weight updates during the return (to about .08-.10), not
through context: its context penalty is +3.6 versus +20.9 (x100) for C.
Current-mode competence is similar (.07-.10 for both). This deficit also blocks
the causal-access STOP by itself (R old-core excess in every group and seed)
and matches the representation base-A failure.

---

## C. Where the thresholds came from, and ceiling calibration

| Tolerance | First appearance | Stated rationale (quoted) |
|---|---|---|
| Cue benefit >= **.002**; marginal gain >= **.02** | `docs/acquisition_diagnostic.md:116-124` (identical to `reports/acquisition/protocol_at_lock.md`) | **None given.** The ledger adds only why both must pass per group: "This prevents averaging a weakly learned cue together with an easier one" (`reports/acquisition/development_ledger.md`). The numerals .02 and .002 first appear with different meanings in `docs/experiment_protocol.md:80` (AUC and accuracy selection tolerances); no document links the two uses. |
| Dev cue margin **.0025** | `docs/training_state_diagnostic.md:72-74` | "deliberately stricter than the unchanged .002 comparison threshold because the preceding development margin was fragile." |
| .002 / .02 reused unchanged | replay_renewal_protocol:24; evidence_consolidation_protocol:110; selective_updates_protocol:131; core_residual_protocol:106-109; causal_access_protocol:170,175; representation_learning_protocol:98,102 | core_residual: "Two seeds per cue give only a limited development check." representation: interleaved cells "demand simultaneous availability ... stricter than earlier fresh-only studies. They may expose context-identification or optimization limits." |
| All-case Brier ceiling **<= .12**; specialist excess **<= .02** | `docs/causal_access_protocol.md:168,177` | "The .12 ceiling and .02 reference tolerances specify useful but imperfect building blocks for this development decision; they are not universal definitions of competence." Reused without new rationale at `representation_learning_protocol.md:101,143-144`. |
| Acquisition gain **<= -.002**, retention/branch costs **<= .005**, survival **>= -.01** | `docs/replay_renewal_protocol.md:109-124` | "The minimum acquisition gain is about one fifth of the prior full-replay-reset gain; tolerances are explicit pilot choices, not universal definitions of safety." Later copies: "These inherit the prior acquisition/retention tolerances; they are practical allocation screens" (`core_residual_protocol.md:121`). |
| Evidence gate .002 margin, 6/8 positive packets; screen >= .002 lifetime gain | `docs/evidence_consolidation_protocol.md:55-56,183-200` | "These are fixed pilot choices, not universal tolerances or proof of noninferiority." |
| .0005 and 9/12 (predictive value, rehearsal state) | predictive_value_protocol:134-139; rehearsal_state_protocol:121-126 | "practical mean-based allocation criteria"; "practical allocation thresholds are not significance tests." |
| +2 pp, 6/8 seeds, 5 pp oracle adequacy | joint_persistence.md; conditional_relationships.md:169-184; contextual_comparison.md:154-171 | "practical pilot thresholds, not a statistical significance test." |
| 1 pp / .5 pp signal, 1 pp retention | `docs/cifar10_transfer_protocol.md:102-116` | "practical thresholds for deciding whether more work on this recipe is justified." |
| 10 pp drawdown, 2 pp window, 0.5 tie | `docs/experiment_v3.md` | "This finite, sparse sample is a screen, not a stability guarantee." |
| ACP open_threshold .32 | `reports/controller_calibration.md` | Chosen from a recorded probe trace: "a post hoc development calibration, not an independent result." |
| 20 paired tuples, Holm correction | `docs/deep-research-report.md` statistical plan | Planned confirmatory design. **Never executed.** |

**Was any threshold ever calibrated against an achievable ceiling? No.** None
of the .002, .02, .12, .005 or .0005 tolerances was derived from an ideal
observer, oracle headroom or a noise floor. The nearest instances:

1. Persistence development rejected world v1 because the best fixed action was
   within 0.3 pp of the clairvoyant bound (`reports/persistence/development_ledger.md`).
   That checked benchmark headroom, not a threshold.
2. The conditional and contextual adequacy checks used a privileged-mode
   oracle to show learnable headroom. The 5 pp and 2 pp margins were not
   derived from it.
3. The acquisition structural audit found zero clairvoyant action value for
   delay and efficiency, so prediction Brier became the endpoint. It recorded
   the per-cue changed-outcome shares (A5) but never computed an attainable
   flipped-minus-correct Brier.
4. Causal access anchored the .02 excess to independently trained specialists,
   a learner-achievable reference. The conditional specialist itself missed the
   .002 delay threshold (.001385).
5. Predictive value reported a validation-selected oracle post hoc, "not proven
   achievable headroom."
6. `docs/evidence_noise_limits.md` derived an ideal-predictor noise gap
   (rho^2 (p - r)^2) but "changes no source, threshold, endpoint or decision rule."

The only data-anchored tolerances are these: .002 as about one fifth of an
observed reset effect (replay renewal); .0025 because an earlier margin was
fragile (training state); and .32 from a trace (ACP).

---

## D. Chronological ledger of prospective screens and qualifications

"Checks" counts gates per architecture unless noted. A margin is value minus limit.

| # | Date | Study, event | Seeds | Checks | Outcome | Failing cells and margin |
|---:|---|---|---:|---:|---|---|
| 1 | 09-20 | ACP synthetic pilot | 5 | no numeric screen | — | Controller never closed or reopened |
| 2 | 09-20 | ACP calibration probes 1-2 (activation) | 1 | 1 | FAIL x2 | 0 reopenings |
| 3 | 09-20 | ACP calibrated suite; CIFAR-100 initial | 3; 3 | no numeric screen | — | ACP -0.59 pp final, -3.17 pp AUC vs ER+recycling |
| 4 | 09-20 | V2 (five suites) | 3 | none (exploratory) | — | — |
| 5 | 09-20 | V3 development gain selection | 2 | drawdown <= 10 pp + 2 pp window | selected 0.5 | 0.15: drawdown 13.28 pp; 0.05 below cutoff |
| 6 | 09-20 | V3 locked evaluation | 3 | fixed contrast, no pass threshold | +1.15 pp | 2/3 seeds < .35 pp |
| 7 | 09-20 | CIFAR-10 transfer | 2 | 34 total (12 adequacy, 6 signal, 16 retention) | **FAIL** | Signal 4/6 fail (newborn mean -0.12 vs >= 1 pp; matched +0.18 vs >= .5 pp); retention 8/16 fail (newborn -3.60 pp final) |
| 8 | 09-20/27 | Dual-path CIFAR-100 | 2 | no numeric screen | — | AUC endpoint floored at 0 |
| 9 | 09-27 | Persistence dev: world v1 | 1 | benchmark validity | rejected | Best fixed action within 0.3 pp of clairvoyant |
| 10 | 09-27 | Persistence pilot | 8 | 5 (adequacy + 4) | **FAIL** | Primary -0.78 vs +2 pp (-2.78); 1/8 vs 6/8 seeds; ties coverage |
| 11 | 09-27 | **Conditional pilot** | 8 | 2 adequacy + 7 | **PASS** | Narrowest: late gate -1.83 vs -2 pp (0.17 inside) |
| 12 | 09-27 | Contextual dev: R comparator grid | 2 x 6 settings | 4 | 5/6 settings FAIL | History benefit -0.59..+0.49 vs 2 pp; return -5.3..+5.3 vs 5 pp |
| 13 | 09-27 | Contextual pilot | 8 | 4 adequacy + 4 screen | adequacy pass; **screen FAIL** | -0.38 vs +2 pp (3/8 vs 6); +0.22 vs +2 pp (5/8). R return adequacy +5.19 vs 5 |
| 14 | 09-27 | Acquisition dev 2048 | 2 | 12 | PASS | Narrowest R delay .002499 |
| 15 | 09-27 | **Acquisition comparison qualification** | 6 | 12 | **FAIL (C)** | **C delay .001939 vs .002 (-3%)**; sole failure; diagnostic continued as descriptive |
| 16 | 09-27 | Training-state dev 4096 | 6 | 12 | **FAIL** | **Delay C .001764, R .000707 vs .0025** (delay only) |
| 17 | 09-27 | Training-state dev 8192 / comparison | 6 / 6 | 12 / 12 | PASS / PASS | Narrowest delay .004222 / .005016 |
| 18 | 09-27 | Replay decomposition qualification | 6 (2 per cue) | 8 | PASS | Narrowest C delay .005326 |
| 19 | 09-27 | Replay policy: qualification + screen | 6 | 12 + 18 | qualification pass; **screen FAIL both** | C 6/18: intro retention +.0231, return/revision/clean/noise retention +.0100/.0082/.0080/.0093, extra noise +.0062 (limits .005). R 5/18: intro +.0140, return +.0193, revision **+.00547 (+9%)**, noise +.0082, extra noise +.0061 |
| 20 | 09-27 | Evidence dev: qualification + screen | 2 | 12 + 12 | **FAIL both** | **Delay .001717 / .001121**; screen 8/12 fail each |
| 21 | 09-27 | Evidence pilot | 6 | 12 + 12 | qualification pass; **screen FAIL both** | 7/12 each: whole-stream gain +.0017/+.0028 vs <= -.002, 0/6 seeds; periodic gains about 0 vs -.002, 4/6 vs 5; return +.0078/+.0162 vs .005 |
| 22 | 09-27 | Return-context diagnosis | 6 | none (retrospective) | — | — |
| 23 | 09-27 | Selective updates | 6 | 8 + 7 + 6 attribution | qualification pass; **FAIL both** | C 3/7: acquisition +.00499 vs <= -.002, 0/6, mode 1 +.00679. R 4/7: +.00547, 0/6, mode 0 +.0181, mode 1 +.0111. Attribution C 1/6, R 2/6 fail |
| 24 | 09-28 | Predictive value | 12 | 2 tiers x 5 | **FAIL all 4 tiers** | Near: C -.00003 vs -.0005 (8/12), R +.00022 (5/12). Return: C +.00096 (7/12), R +.00158 (3/12); both lose to the accuracy control |
| 25 | 09-28 | Rehearsal-state replication | 12 | 3 | **FAIL both** | C T -.000439 vs >= .0005, 6/12. **R 8/12 vs 9 (one seed)**, sole failure |
| 26 | 09-28 | Core/residual | 6 | 8 + 10 + 3 attribution | qualification pass; **FAIL both** | C 4/10: acquisition +.01685, 4/6, post-return novel +.01438, novel survival -.0186. R 5/10: +.00392, 2/6, return +.0337, survival -.0129/-.0351. Attribution: C fails 4/6 vs 5; R passes |
| 27 | 09-28 | Core/residual outputs, GO rule (no training) | 6 | 4 | PASS both (exploratory) | — |
| 28 | 09-28 | **Causal access** | 6 (2 per cue) | 32 qualification + 12 online | **INELIGIBLE + raw FAIL both** | C: old-core excess .0325 (**delay seed 17006 = .149**), delay-group excess .0746, **specialist delay .001385**; online vs half +.00079 (3/6); **online delay .000948**. R: old-core excess > .02 in all groups and seeds; **full-novel delay .001940 (-3%)**; novel focus +.005512 vs .005 (+10%); **online delay .001379** |
| 29 | 09-28 | **Representation qualification** | 6 | 31 cells | **FAIL -> STOP** | Base-A .1956 vs <= .12 (+63%); **delay -.000152 / .000802 / -.000291 vs .002** |

### Systemic patterns (post hoc)

1. **Pass rate.** Of 13 method/replication decisions from CIFAR-10 onward (rows
   7, 10, 11, 13, 19, 21, 23-29), two passed: the conditional pilot and the
   exploratory GO rule with no training. The last passing learning screen was
   the conditional pilot (2026-09-27 03:14 UTC).
2. **Delay cells.** In the cue world, 5 of 13 qualification events failed
   (rows 15, 16, 20, 28, 29) and every failure includes a delay cell. Delay
   was the minimum-margin cue group in all 13. Three failures were delay-only
   (rows 15, 16, 20); none stopped a line by itself. The two STOP decisions
   (rows 28, 29) also had non-delay failures sufficient to fail: R old-core
   excess in every group, C fixed-half gain, and the base-A ceiling. Removing
   delay would not reverse either.
3. **Narrow margins.** Narrow failures: acquisition C delay (-3%), causal R
   full-novel delay (-3%), causal R novel focus (+10%), replay R revision (+9%),
   rehearsal R 8/12. Only two outcomes hinged on a single narrow miss: row 15
   and R in row 25. Narrow passes: conditional late gate (0.17 pp), contextual
   R return adequacy (0.19 pp), causal R survival (.000133 inside), acquisition
   dev R delay (+25%). Most candidate failures were large (0/6 seeds; central
   estimates at or beyond two to eight times the tolerance), so low power did
   not cause them.
4. **Power of six-seed conjunctions.** Binomial seed-consistency rules pass
   with P(>= 5/6) = .42 / .66 / .89 when per-seed success p = .7 / .8 / .9. When
   both architectures must pass, the rates are .18 / .43 / .79. P(>= 9/12) =
   .49 / .80 / .97, and P(>= 6/8) = .55 / .80 / .96. Two-seed delay cells miss
   about 10-18% of the time for a typical fresh learner (A5). Conjunction width
   grew from 5-9 checks (rows 10-13) to 24-30 (rows 19-21), 44 (row 28) and 31
   (row 29), each passed separately per architecture. As an illustration, 31
   independent cells each passing at .95 would jointly pass .95^31 = .20; at
   .99 the rate is .73.
5. **Recurring confounds.**
   - Delay has low structural leverage (7% of cases change) and high seed
     variance (CV about .66); continuing and online predictors sit around
     .001-.005, near the one uniform threshold.
   - Recurrent non-current-mode weakness (B) recurs from prefix end onward.
   - Evaluator-selected correct support differs from actual history.
   - Identical inputs at unannounced switches make a recognition delay
     unavoidable.
   - Six cue orders over three cues leave 2-seed cue cells.
   - Mode is not crossed with order; stage complexity is confounded with
     learner age.
   - Exposure changed (2,048 to 8,192), preventing cross-study effect comparison.
   - Fresh references are learnability references, not forgetting controls.
   - Feedback covers only the performed action.
   - All cue-era evidence comes from one conserving simulator.

---

## E. Original relational/persistence motivation and what was never implemented

**Stated motivation.**

- **Charter** (`docs/research_charter.md:60-131`): "useful knowledge describes
  how observations, entities, actions, and consequences depend on one another,
  together with the conditions under which those dependencies apply." Explicit
  graphs are "one possible implementation." Its operations are (1) learn
  conditional predictors, (2) infer applicability from recent observations,
  actions and outcomes, and (3) revise or develop representations when
  predictors repeatedly fail, while telling novelty from noise. A bounded
  predictor library may add allocation or replacement. Joint survival is "an
  experimenter's objective."
- **Joint persistence** (`docs/joint_persistence.md`): keep two participants
  functional; a bounded memory ranked by recurrence and survival relevance.
  "not a complete online reinforcement learner, an evolving population."
- **User framing** (`docs/representation_learning_next_thread_prompt.md`):
  "relationships that help entities persist under uncertain environments";
  unannounced recurrence; recoverable rather than perfect retention.
- **Research review and deep-research report**: an adaptive critical-period
  controller; CIFAR-100, then CORe50, Infinite dSprites and Continual World;
  20 tuples; Pareto frontiers.
- **Predictive retention** (`docs/predictive_retention.md`): value by matched
  intervention; calibration; a value-weighted rehearsal distribution; outcome
  and hazard weighting.
- **README**: learn from scratch, infer when knowledge applies, revise it.

**Never implemented or never reached**, with evidence:

| Element | Status | Evidence |
|---|---|---|
| Agent-chosen actions (actions shape future data, exploration, curiosity) | Never. Training actions are always randomized or exogenous; greedy survival is scored counterfactually by the evaluator | research_synthesis.md:304-307 ("adaptive exploratory actions were not implemented"); causal_access_protocol ("a policy's greedy action does not alter its future training") |
| Explicit relational structure (entity/relation factorization, graphs, symbolic relations) | Never; neural conditional predictors only | research_charter.md:140; research_synthesis.md:299-302, 310-314; conditional_results.md:139 |
| Dense observation of consequences | Never. One performed action's three binary survival labels (H = 4, 8, 12); no post-action frames; unperformed outcomes are evaluator-only | joint_persistence.md; predictive_retention.md; representation_learning_design.md:197-200 (a "relational world model trained on observed post-action trajectories" would need a new benchmark) |
| Multiple interacting entities or agents | Only two simulated resource holders and one transfer per trial; no independently motivated agents, cooperation networks or populations | persistence_results.md; conditional_results.md:141-143; research_synthesis.md:304-313 |
| Self-generated goals | Never; the objective is fixed by the experimenter (joint survival at H = 12) | conditional_results.md:143; research_charter.md:73-77 |
| Reproduction, inheritance, evolutionary selection of representations | Never | research_synthesis.md:304-306 |
| Predictor allocation, replacement or growth in a bounded library | Never; four fixed heads | research_charter.md:114-116, 129 |
| Autonomous change detection and novelty-versus-noise discrimination | Not achieved; the evidence gate failed; resets and freezes only at privileged boundaries | evidence_consolidation_results.md; evidence_noise_limits.md |
| Second environment, longer streams, longer physical horizon | Never; one world; H = 12 fixed | research_charter.md:250-255; research_synthesis.md |
| Compounding (matched early versus late dependencies) | Proposed repeatedly; never run | research_synthesis.md; ADAM_HANDOFF.md |
| Graded protection or consolidation under continuing learning and fixed resources | Proposed after causal access; not run | causal_access_results.md; representation_learning_design.md |
| Temporal-reconstruction arms | Implemented but never fit (qualification STOP) | representation_learning_results.md |
| CORe50, Infinite dSprites, Continual World; full ResNet CIFAR-100; 20-seed confirmatory study; EWC/SI/GEM/UPGD baselines; common-compute comparison | Never | deep-research-report.md; research_review.md; experiment_protocol.md |
| Calibration or hazard weighting as a retention signal; value-softmax rehearsal | Never | predictive_retention.md |
| Critical periods in the relational line; recursive self-improvement | Optional or unproven; explicitly excluded | research_charter.md:447-448; representation_learning_design.md:202-204 |

*POST HOC / EXPLORATORY / NOT DECISION-BEARING. This map records existing
evidence and descriptive calculations only. It does not modify any locked
decision, protocol, threshold or result.*
