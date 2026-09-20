# Controller calibration: an initial mechanism-activation failure

Date: 2026-09-20. All measurements here are **exploratory development observations**. Two single-seed ACP-only training probes were run; neither reopened. They established that a change in timescales can permit signal-driven closure. The threshold was trace-calibrated before the follow-up suite. That completed suite is reported in the addendum below and in [development_results.md](development_results.md): it activated reopening but did not outperform replay plus recycling.

The original [synthetic pilot summary](synthetic_initial/summary.md) remains archived. Its apparently strong ACP scores do not establish efficacy of the developmental close/reopen mechanism: all five ACP seeds stayed OPEN after their initial scaffold, and ACP matched `acp_no_reopening` exactly. Replay-risk contraction can still transiently reduce gates while the phase is OPEN, so this is not a claim that every controller operation was absent. Consolidation and reopening were inactive. No source code was changed during this calibration.

## What prevented the original controller from closing

The original configuration used 100 optimizer steps per experience, monitoring every 10 steps, reference EMA decay 0.9, and recycling every 20 steps. This gives only ten monitoring windows in each experience, with a recycling check every two windows. The EMA's half-life is approximately 6.58 windows, or 65.8 steps; after ten windows, 34.9% of its initial reference weight remains. Actual settling also depends on changing features and variance, so this calculation is a timescale diagnostic rather than a predicted drift value.

Across seeds 11, 22, 33, 44, and 55, each ACP run logged two SCAFFOLD windows and 78 OPEN windows, **zero stable windows**, and zero reopenings. In seed 11, drift had median 0.808 and maximum 6.0, against closure tolerance 0.2. Loss often became stable very quickly, while feature drift failed the joint stability condition. Even step 800 had drift 0.274.

The monitor already aggregates full-window feature moments and subtracts its estimated independent-sampling noise floor. This failure therefore remains after the earlier single-minibatch noise issue was addressed. Three interacting causes are visible:

1. The reference assimilates a new distribution slowly relative to the short experience dwell.
2. Learning changes the representation whose drift is being measured. That change is not necessarily an external distribution shift.
3. Recycling repeatedly changes feature coordinates, including low-variance coordinates that can create very large standardized drift. In the original first experience, resets at steps 60 and 80 preceded capped drift of 6.0 at steps 70 and 90, despite stable labels and near-zero loss change.

These are controller-identifiability and timescale problems. They are not evidence against the biological critical-period literature, and cannot be repaired by interpreting the original accuracy gain as a controller benefit.

## Complete tuning ledger

The same development seed, 11, was used for both probes. Architecture, class count, training/validation samples, noise, learning rate, replay capacity, maturity, and stability thresholds were held at the original pilot choices. The probes used eight experiences, a 64-wide MLP, and validation evaluation only. They generated **zero test examples**. Scratch-reference training and checkpoint writes were disabled for these two ACP-only probes; both will be enabled in the follow-up multi-method candidate.

| Setting or outcome | Original pilot | Probe 1 | Probe 2 |
|---|---:|---:|---:|
| ACP seeds examined | 11, 22, 33, 44, 55 | 11 | 11 |
| Steps per experience | 100 | 300 | 300 |
| Monitoring interval | 10 | 10 | 10 |
| EMA decay | 0.9 | 0.5 | 0.7 |
| EMA half-life, windows | 6.58 | 1.00 | 1.94 |
| Recycling check interval | 20 | 100 | 100 |
| Reopening threshold | 0.65 | 0.65 | 0.45 |
| Consecutive novelty windows | 2 | 2 | 2 |
| Stability loss/drift tolerances | 0.1 / 0.2 | 0.1 / 0.2 | 0.1 / 0.2 |
| Lagged replay-risk cutoff | 0.1 | 0.1 | 0.1 |
| Entered CLOSING, seed 11 | Never | Step 90 | Step 100 |
| Entered ADULT, seed 11 | Never | Step 120 | Step 130 |
| Reopenings, seed 11 | 0 | 0 | 0 |
| Stable monitor windows, seed 11 | 0 / 80 | 197 / 240 | 177 / 240 |
| Final validation accuracy, seed 11 | 99.27% | 98.78% | 98.78% |
| Total recycled units, seed 11 probes | — | 16 | 16 |

The first change lengthened experience dwell to 30 monitoring windows and spaced recycling checks ten windows apart, closer to the ratio in the research report. Faster reference adaptation was intended to let the controller recognize genuinely settled training. Probe 1 demonstrated that this can activate closure without a timeout: the transition reason was `sustained_stability`. It then spent 229 logged windows in ADULT and never met sustained novelty.

Probe 2 slowed the reference from decay 0.5 to 0.7, still considerably faster than the original 0.9, and lowered the novelty threshold to 0.45. This sought to preserve post-shift evidence longer while retaining the two-window rule. It also closed from observed stability, spent 228 logged windows in ADULT, and failed to reopen. These are both reported failures of the reopening activation criterion.

No fixed closure deadline, task-boundary signal, single-window detection rule, or relaxation of the replay-risk veto was introduced. The existing CLOSING phase has a fixed ramp after its signal-driven entry; this is distinct from forcing entry into CLOSING on a clock.

The source hash recorded by the original pilot and both probes is `a5110b0bac6f5e0567c44e9212f65ed91f7b7e0aedd704bb5d3cea6608ce01dd`. Each output directory stores the exact executed configuration and its hash:

- [Probe 1 manifest](calibration_probes/probe1_manifest.json), [result](calibration_probes/probe1_result.json), [events](calibration_probes/probe1_events.json). Config SHA256: `10c16394622966b8efaec444273bc2534b02e99f821194c9d58cb408999c5344`.
- [Probe 2 manifest](calibration_probes/probe2_manifest.json), [result](calibration_probes/probe2_result.json), [events](calibration_probes/probe2_events.json). Config SHA256: `e190699ee505eabd23e04e2e492ede8f1f4f3bcb507532ed3529dad07d12286a`.

## Why reopening still did not activate

The synthetic problem often becomes easy within a single ten-step window. The always-trainable classifier can quickly fit new labels, so high loss surprise is brief. A fast EMA further attenuates the second observation. Meanwhile, the lagged replay-risk measurement often vetoes that second window: probe 2 reported damage of 0.102 at step 320, 0.240 at 620, 0.613 at 920, and 0.214 at 1220. This erases the consecutive-novelty count. The current damage statistic is a ratio with a denominator floor and sampled one-update timing; it is not an independent judgment that the entire experience is unsafe.

There is also a clear false-novelty source. In probe 1, a reset at step 200 was followed by drift 5.99 and novelty 0.990 at step 210 on unchanged classes. At step 220 novelty was back to 0.222. Similar isolated spikes followed later resets. Keeping the two-window rule avoided converting these observed isolated spikes into reopening events. It would be misleading to lower the rule to one window and then count recycling-induced reopening as successful shift detection.

The independent-sampling correction is heuristic here: a ten-step window contains 640 presentations from only 512 distinct training examples, and weights change within that window. Repeated examples and changing representations violate a simple independent, fixed-feature sampling model. Longer dwell enables closure but does not solve that identification problem.

## Final development candidate for a fresh paired suite

[configs/synthetic_calibrated.json](../configs/synthetic_calibrated.json) retains probe 2's decay 0.7, 300 steps per experience, and recycling interval 100, and sets **`open_threshold = 0.32`**. All other controller settings, including two consecutive novelty windows and the replay-risk veto, are unchanged.

This final threshold was chosen from the recorded probe-2 trace, after the two allowed training probes. Excluding replay-vetoed observations, the minimum novelty of the consecutive post-shift pair at steps 330–340 was 0.381, and that of steps 1810–1820 was 0.331. An opening threshold of 0.32 would allow those pairs to satisfy the detector on the recorded trace. For comparison, the largest two-window minimum more than 60 steps into an experience was 0.274; the largest pair immediately associated with the prominent recycling spikes was about 0.269.

This is a **post hoc development calibration**, not an independent result. The sigmoid output is an arbitrary controller score, not a calibrated probability of novelty; 0.32 should not be interpreted as a confidence level. Actual reopening changes subsequent weights, losses, replay damage, and drift, so replaying or inspecting a fixed historical trace cannot establish what the interactive learner will do. No third training probe was run. The final candidate must be validated before claiming that the full close/reopen cycle has been observed.

The next configured methods are `er`, `er_recycle`, `fixed`, `acp`, `acp_no_reopening`, and `acp_no_recycling`, with fresh development seeds **66, 77, 88**. Evaluation is validation-only, `test_per_class` is zero, and the same acquisition budget applies to every method. The source runner keeps boundaries outside the learner; closure remains signal-driven.

```powershell
.\.venv\Scripts\python.exe -m acp_cl run --config configs/synthetic_calibrated.json --output runs/synthetic_calibrated --device cpu
```

This suite should first be judged on behavior: whether every relevant controller state is reached; whether openings follow external shifts rather than recycling alone; how often replay-risk vetoes intervene; and whether the no-reopening ablation now differs. A failure to reopen again is a result to retain, not a reason to retroactively recast the pilot as successful.

## Limits on comparison and next decisions

The new 300-step dwell triples training exposure. With early AUC defined over 20% of an experience, its early horizon also grows from 20 steps to 60. **Do not compare the two pilots' early AUC values as though they measured the same sample budget.** The fresh paired suite can compare its own methods at the shared horizon; a cross-pilot comparison would require recomputing both at a fixed presentation cutoff.

The benchmark has a strong accuracy ceiling: seed 11's original ACP result was already 99.27%, and both probes reached 98.78%. Selection here was based on controller activation, not accuracy gain. One seed, correlated synthetic prototypes, altered dwell, and changed recycling opportunities provide no causal evidence that the critical-period mechanism improves continual learning. The next multi-method suite remains an engineering pilot even if it shows reopening and favorable accuracy.

**Before interpreting novelty as external change, the next algorithm revision should separate its sensor from structural resets.** Prefer an input-derived signal or frozen-encoder feature reference for external distribution drift, while using the changing learner's rank/dormancy as a separate capacity signal. A frozen encoder removes this particular dependence on learner-coordinate changes; it does not guarantee detection of every task-relevant shift. Its parameters, forwards, and any pretraining must be included in information/resource budgets and supplied fairly to controls.

An alternative is explicit reset compensation: compare pre/post-reset feature moments on identical in-budget probe examples and refresh or transform only the affected reference statistics. Record the intervention and any detector refractory period. This approach must be tested on shifts that coincide with resets, because blindly resetting the reference can erase real novelty as well. Either approach is preferable to treating large reset-generated drift as evidence of successful environmental novelty detection.

Also examine whether a consecutive-threshold detector is appropriate for such short acquisition transients. These are proposed implementation changes requiring new ablations; none were silently introduced in these probes. The original always-OPEN pilot and both no-reopening probes remain part of the development record.

## Follow-up result, after freezing the candidate

The configured six-method suite has now completed on seeds 66, 77, and 88. ACP reached ADULT at steps 130/120/130 and reopened 1/3/1 times, with bounded durations. This validates activation of the mechanism under the calibrated settings, not its efficacy. ACP averaged 98.86% final accuracy and 86.99% late early-learning AUC, compared with 99.45% and 90.16% for ER+recycling. It also failed to improve on no-reopening. The [full development report](development_results.md) and [paired summaries](synthetic_calibrated/summary.md) retain these negative outcomes. The preceding tuning narrative records what was known before this follow-up; no additional thresholds were selected after it.
