# Selective updates: first-introduction development

12 challenge episodes and 2 shared prefixes completed. Paired seeds: [13991].

Only combined is the primary candidate. These fixed development thresholds allocate further research; they are not superiority/noninferiority tests or independent confirmation. All intervals are descriptive 95% seed bootstrap intervals with 20,000 resamples. No ablation is silently promoted.

## Per-architecture decisions

| Model | Fresh qualification | Combined outcome screen | Protection attribution |
|---|---|---|---|
| conditional | FAIL | FAIL | FAIL |
| recurrent | FAIL | FAIL | FAIL |

This is an engineering/subset configuration and cannot pass the declared development allocation screen.

- conditional, screen: failed criteria = complete_development_cohort, fresh_qualification, acquisition_gain, acquisition_consistency.
- recurrent, screen: failed criteria = complete_development_cohort, fresh_qualification, acquisition_gain, acquisition_consistency.
- conditional, protection_attribution: failed criteria = complete_development_cohort, fresh_qualification, retention_vs_current, retention_vs_shrink.
- recurrent, protection_attribution: failed criteria = complete_development_cohort, fresh_qualification, retention_vs_current, retention_vs_shrink.

## Paired contrasts

Brier values are multiplied by 100; negative is better. Survival differences use percentage points. Each independent seed contributes one episode per arm.

| Model | Contrast | Acquisition AUC [95% interval] | Valid-old mean [95% interval] | Survival AUC [95% interval] |
|---|---|---:|---:|---:|
| conditional | current_minus_reference | -0.015 [-0.015, -0.015] | +0.074 [+0.074, +0.074] | +0.000 [+0.000, +0.000] |
| conditional | combined_minus_protected | -0.015 [-0.015, -0.015] | +0.073 [+0.073, +0.073] | +0.000 [+0.000, +0.000] |
| conditional | protected_minus_reference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| conditional | combined_minus_current | +0.000 [+0.000, +0.000] | -0.001 [-0.001, -0.001] | +0.000 [+0.000, +0.000] |
| conditional | current_main | -0.015 [-0.015, -0.015] | +0.073 [+0.073, +0.073] | +0.000 [+0.000, +0.000] |
| conditional | protection_main | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] | +0.000 [+0.000, +0.000] |
| conditional | interaction | +0.000 [+0.000, +0.000] | -0.001 [-0.001, -0.001] | +0.000 [+0.000, +0.000] |
| conditional | combined_minus_reference | -0.015 [-0.015, -0.015] | +0.073 [+0.073, +0.073] | +0.000 [+0.000, +0.000] |
| conditional | combined_minus_shrink | +0.000 [+0.000, +0.000] | -0.001 [-0.001, -0.001] | +0.000 [+0.000, +0.000] |
| conditional | shrink_minus_current | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| conditional | reference_minus_fresh | -0.176 [-0.176, -0.176] | -1.653 [-1.653, -1.653] | +10.547 [+10.547, +10.547] |
| conditional | combined_minus_fresh | -0.190 [-0.190, -0.190] | -1.580 [-1.580, -1.580] | +10.547 [+10.547, +10.547] |
| recurrent | current_minus_reference | +0.006 [+0.006, +0.006] | -0.014 [-0.014, -0.014] | +0.000 [+0.000, +0.000] |
| recurrent | combined_minus_protected | +0.007 [+0.007, +0.007] | -0.022 [-0.022, -0.022] | +0.000 [+0.000, +0.000] |
| recurrent | protected_minus_reference | +0.000 [+0.000, +0.000] | -0.002 [-0.002, -0.002] | +0.000 [+0.000, +0.000] |
| recurrent | combined_minus_current | +0.001 [+0.001, +0.001] | -0.011 [-0.011, -0.011] | +0.000 [+0.000, +0.000] |
| recurrent | current_main | +0.006 [+0.006, +0.006] | -0.018 [-0.018, -0.018] | +0.000 [+0.000, +0.000] |
| recurrent | protection_main | +0.001 [+0.001, +0.001] | -0.007 [-0.007, -0.007] | +0.000 [+0.000, +0.000] |
| recurrent | interaction | +0.001 [+0.001, +0.001] | -0.008 [-0.008, -0.008] | +0.000 [+0.000, +0.000] |
| recurrent | combined_minus_reference | +0.007 [+0.007, +0.007] | -0.025 [-0.025, -0.025] | +0.000 [+0.000, +0.000] |
| recurrent | combined_minus_shrink | +0.001 [+0.001, +0.001] | -0.011 [-0.011, -0.011] | +0.000 [+0.000, +0.000] |
| recurrent | shrink_minus_current | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| recurrent | reference_minus_fresh | -0.022 [-0.022, -0.022] | -0.741 [-0.741, -0.741] | +0.000 [+0.000, +0.000] |
| recurrent | combined_minus_fresh | -0.014 [-0.014, -0.014] | -0.766 [-0.766, -0.766] | +0.000 [+0.000, +0.000] |

## Retention guardrails by old mode

Combined minus reference, Brier x100. The +.005 raw-Brier tolerance is checked in each mode, not only their average.

| Model | Mode | Difference [95% interval] | Gate | Interval crosses limit |
|---|---:|---:|---|---|
| conditional | 0 | +0.092 [+0.092, +0.092] | True | False |
| conditional | 1 | +0.054 [+0.054, +0.054] | True | False |
| recurrent | 0 | -0.036 [-0.036, -0.036] | True | False |
| recurrent | 1 | -0.014 [-0.014, -0.014] | True | False |

## Fresh-reference qualification

| Group | N | Marginal gain x100 | Cue benefit x100 | Pass |
|---|---:|---:|---:|---|
| conditional_stage_1 | 1 | -1.058 | 0.002 | False |
| conditional_cue_2 | 1 | -1.058 | 0.002 | False |
| recurrent_stage_1 | 1 | -0.842 | 0.001 | False |
| recurrent_cue_2 | 1 | -0.842 | 0.001 | False |

## Every episode

Brier and cue-benefit values x100; all raw metric values remain in summary.json.

| Seed | Model | Cue | Arm | Acquisition AUC | Valid-old mode 0 | Valid-old mode 1 | Cue benefit |
|---|---|---:|---|---:|---:|---:|---:|
| 13991 | conditional | 2 | combined | 24.498 | 22.793 | 23.642 | +0.000 |
| 13991 | conditional | 2 | current | 24.498 | 22.794 | 23.642 | +0.000 |
| 13991 | conditional | 2 | fresh | 24.689 | 24.835 | 24.758 | +0.002 |
| 13991 | conditional | 2 | protected | 24.513 | 22.700 | 23.588 | +0.000 |
| 13991 | conditional | 2 | reference | 24.513 | 22.700 | 23.588 | +0.000 |
| 13991 | conditional | 2 | shrink | 24.498 | 22.794 | 23.642 | +0.000 |
| 13991 | recurrent | 2 | combined | 24.295 | 24.280 | 24.468 | +0.000 |
| 13991 | recurrent | 2 | current | 24.294 | 24.294 | 24.475 | +0.000 |
| 13991 | recurrent | 2 | fresh | 24.309 | 25.120 | 25.159 | +0.001 |
| 13991 | recurrent | 2 | protected | 24.288 | 24.313 | 24.480 | +0.000 |
| 13991 | recurrent | 2 | reference | 24.288 | 24.316 | 24.481 | +0.000 |
| 13991 | recurrent | 2 | shrink | 24.294 | 24.295 | 24.475 | +0.000 |

## Resource and interpretation limits

All arms compute the replay reference gradient, hypothetical projection and final-update replay-loss diagnostic. These extra operations are recorded separately from the original presentation budget. The deployable ordinary reference does not require discarded diagnostic work. Full counters, scalar summaries, parameter/optimizer/replay bytes and episode wall time are preserved in resource_rows.

The constraint is first-order reported BCE on one sampled historical packet; finite-step replay loss can still increase, and this is not protection of every old relationship or clean physical outcome. These known-boundary first-introduction forks do not demonstrate autonomous lifelong learning, recurrence, noise robustness, late acquisition or compounding. Fresh-model damage is not a forgetting control because its starting errors differ; interpret terminal levels separately.

The portable record check validates source/configuration/protocol/analysis locks, complete paired inputs, recorded forks and resource counts. A separate full checkpoint audit verifies actual tensor states and probes. Neither check is independent scientific replication.
