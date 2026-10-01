# Adam: evidence-consolidation diagnostic

2 shared draft trajectories, 6 blocks, 6 fresh references; 1 independent seeds per architecture.

All four policies deploy predictors from the same learned draft trajectory. This tests acceptance of updates, not different representation-learning trajectories.

Primary truth error scores the performed action before training on every incoming packet, averaging all three outcome horizons with the actual preceding reported history. Focus and survival AUC instead integrate periodic independent all-action query probes; clean-support valid-old and knowledge panels are separate retention measurements.

Brier numbers below are multiplied by 100. Error contrasts are sustained minus control; negative is better. Intervals use 20,000 paired-seed bootstrap resamples, seed 27192026. Packets and blocks are not independent seed replicates.

| Model | Control | Overall truth difference [95% interval] | Late difference | Growing advantage, late minus early |
|---|---|---:|---:|---:|
| conditional | immediate | +0.191 [+0.191, +0.191] | unavailable | unavailable |
| conditional | periodic | +0.082 [+0.082, +0.082] | unavailable | unavailable |
| conditional | single | +0.000 [+0.000, +0.000] | unavailable | unavailable |
| recurrent | immediate | +0.089 [+0.089, +0.089] | unavailable | unavailable |
| recurrent | periodic | -0.003 [-0.003, -0.003] | unavailable | unavailable |
| recurrent | single | +0.047 [+0.047, +0.047] | unavailable | unavailable |

Positive growing advantage means the candidate's relative error improved more in cycle 3 than cycle 1; it does not establish compounding representation gains.

* conditional: **INCONCLUSIVE**; requires six independent seeds; requires all eighteen continuous blocks; development cohort is descriptive.
* recurrent: **INCONCLUSIVE**; requires six independent seeds; requires all eighteen continuous blocks; development cohort is descriptive.

| Model | Gate | Pass | Value | Tolerance | Interval permits violation |
|---|---|---|---:|---:|---|
| conditional | fresh_qualification | False | - | all fresh groups | - |
| conditional | immediate_overall_truth_gain | False | +0.191 | <= -0.200 | True |
| conditional | immediate_late_truth_gain | None | - | <= -0.200 | - |
| conditional | immediate_overall_consistency | None | 0 | 5 of 6 | - |
| conditional | periodic_overall_truth_gain | False | +0.082 | <= -0.200 | True |
| conditional | periodic_late_truth_gain | None | - | <= -0.200 | - |
| conditional | periodic_overall_consistency | None | 0 | 5 of 6 | - |
| conditional | immediate_intro_focus_auc_guardrail | True | +0.118 | <= +0.500 | False |
| conditional | immediate_return_truth_guardrail | None | - | <= +0.500 | - |
| conditional | immediate_noise_truth_guardrail | None | - | <= +0.500 | - |
| conditional | immediate_valid_after_guardrail | True | +0.251 | <= +0.500 | False |
| conditional | immediate_survival_guardrail | True | -0.391 | >= -1.000 | False |
| recurrent | fresh_qualification | False | - | all fresh groups | - |
| recurrent | immediate_overall_truth_gain | False | +0.089 | <= -0.200 | True |
| recurrent | immediate_late_truth_gain | None | - | <= -0.200 | - |
| recurrent | immediate_overall_consistency | None | 0 | 5 of 6 | - |
| recurrent | periodic_overall_truth_gain | False | -0.003 | <= -0.200 | True |
| recurrent | periodic_late_truth_gain | None | - | <= -0.200 | - |
| recurrent | periodic_overall_consistency | None | 1 | 5 of 6 | - |
| recurrent | immediate_intro_focus_auc_guardrail | True | +0.270 | <= +0.500 | False |
| recurrent | immediate_return_truth_guardrail | None | - | <= +0.500 | - |
| recurrent | immediate_noise_truth_guardrail | None | - | <= +0.500 | - |
| recurrent | immediate_valid_after_guardrail | True | +0.359 | <= +0.500 | False |
| recurrent | immediate_survival_guardrail | True | +8.073 | >= -1.000 | False |

| Fresh qualification group | Episodes | Marginal gain x100 | Cue benefit x100 | Pass |
|---|---:|---:|---:|---|
| conditional_stage_1 | 1 | 1.533 | -0.002 | False |
| conditional_stage_2 | 1 | 5.342 | -0.000 | False |
| conditional_stage_3 | 1 | 1.719 | 0.003 | False |
| conditional_cue_0 | 1 | 1.719 | 0.003 | False |
| conditional_cue_1 | 1 | 5.342 | -0.000 | False |
| conditional_cue_2 | 1 | 1.533 | -0.002 | False |
| recurrent_stage_1 | 1 | 1.313 | -0.000 | False |
| recurrent_stage_2 | 1 | 5.086 | 0.001 | False |
| recurrent_stage_3 | 1 | 1.567 | -0.002 | False |
| recurrent_cue_0 | 1 | 1.567 | -0.002 | False |
| recurrent_cue_1 | 1 | 5.086 | 0.001 | False |
| recurrent_cue_2 | 1 | 1.313 | -0.000 | False |

| Seed | Model | Policy | Overall truth | Early truth | Late truth | Valid-old level | Survival % | Adoptions |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 29 | conditional | immediate | 24.629 | unavailable | unavailable | 24.823 | 44.271 | 0 |
| 29 | conditional | periodic | 24.738 | unavailable | unavailable | 24.913 | 44.922 | 3 |
| 29 | conditional | single | 24.820 | unavailable | unavailable | 24.990 | 43.880 | 1 |
| 29 | conditional | sustained | 24.820 | unavailable | unavailable | 25.075 | 43.880 | 0 |
| 29 | recurrent | immediate | 25.341 | unavailable | unavailable | 25.142 | 33.594 | 0 |
| 29 | recurrent | periodic | 25.433 | unavailable | unavailable | 25.234 | 37.109 | 3 |
| 29 | recurrent | single | 25.383 | unavailable | unavailable | 25.330 | 36.458 | 1 |
| 29 | recurrent | sustained | 25.430 | unavailable | unavailable | 25.500 | 41.667 | 0 |

| Seed | Model | Block (1-based) | Condition | Policy | Truth | Reported | Focus AUC | Valid-old level | Survival % |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|
| 29 | conditional | 1 | introduction_1 | immediate | 25.071 | 25.071 | 24.347 | 24.912 | 45.312 |
| 29 | conditional | 2 | introduction_2 | immediate | 25.213 | 25.213 | 24.343 | 24.821 | 45.312 |
| 29 | conditional | 3 | introduction_3 | immediate | 23.604 | 23.604 | 24.103 | 24.737 | 42.188 |
| 29 | conditional | 1 | introduction_1 | periodic | 25.188 | 25.188 | 24.399 | 25.007 | 45.312 |
| 29 | conditional | 2 | introduction_2 | periodic | 25.365 | 25.365 | 24.431 | 24.912 | 45.312 |
| 29 | conditional | 3 | introduction_3 | periodic | 23.661 | 23.661 | 24.147 | 24.821 | 44.141 |
| 29 | conditional | 1 | introduction_1 | single | 25.064 | 25.064 | 24.430 | 25.075 | 43.750 |
| 29 | conditional | 2 | introduction_2 | single | 25.622 | 25.622 | 24.525 | 25.075 | 43.750 |
| 29 | conditional | 3 | introduction_3 | single | 23.776 | 23.776 | 24.187 | 24.821 | 44.141 |
| 29 | conditional | 1 | introduction_1 | sustained | 25.064 | 25.064 | 24.430 | 25.075 | 43.750 |
| 29 | conditional | 2 | introduction_2 | sustained | 25.622 | 25.622 | 24.525 | 25.075 | 43.750 |
| 29 | conditional | 3 | introduction_3 | sustained | 23.776 | 23.776 | 24.192 | 25.075 | 44.141 |
| 29 | recurrent | 1 | introduction_1 | immediate | 24.661 | 24.661 | 24.472 | 25.232 | 33.984 |
| 29 | recurrent | 2 | introduction_2 | immediate | 25.814 | 25.814 | 24.685 | 25.140 | 33.984 |
| 29 | recurrent | 3 | introduction_3 | immediate | 25.548 | 25.548 | 24.207 | 25.053 | 32.812 |
| 29 | recurrent | 1 | introduction_1 | periodic | 24.762 | 24.762 | 24.565 | 25.330 | 42.188 |
| 29 | recurrent | 2 | introduction_2 | periodic | 25.984 | 25.984 | 24.788 | 25.232 | 36.328 |
| 29 | recurrent | 3 | introduction_3 | periodic | 25.554 | 25.554 | 24.228 | 25.140 | 32.812 |
| 29 | recurrent | 1 | introduction_1 | single | 24.662 | 24.662 | 24.662 | 25.330 | 42.188 |
| 29 | recurrent | 2 | introduction_2 | single | 25.984 | 25.984 | 24.810 | 25.330 | 35.938 |
| 29 | recurrent | 3 | introduction_3 | single | 25.502 | 25.502 | 24.243 | 25.330 | 31.250 |
| 29 | recurrent | 1 | introduction_1 | sustained | 24.662 | 24.662 | 24.719 | 25.500 | 42.188 |
| 29 | recurrent | 2 | introduction_2 | sustained | 26.062 | 26.062 | 25.088 | 25.500 | 42.188 |
| 29 | recurrent | 3 | introduction_3 | sustained | 25.566 | 25.566 | 24.366 | 25.500 | 40.625 |

summary.json preserves both valid-old modes, every paired seed difference, per-block cumulative loss, knowledge endpoints, and window diagnostics. raw_results.jsonl.gz contains every complete block and fresh-reference record. Immediate has no adoption gate; its reported zero adoptions is not zero learning. Noise-block policy differences are not causal noise penalties without matched clean twin trajectories.

The immediate standalone learner needs one model. The delayed scaffold can retain three models (draft, proposal, committed) and uses two policy/evidence prediction passes per packet versus one for immediate, in addition to the shared training budget. resource_rows records parameter bytes, replay/optimizer sizes, and operation counts. These are logical standalone counts for the implemented scaffold, not measured runtime or total process RAM; evaluator probes are excluded from those online inference counts.
