# Selective updates: first-introduction development

72 challenge episodes and 12 shared prefixes completed. Paired seeds: [13001, 13002, 13003, 13004, 13005, 13006].

Only combined is the primary candidate. These fixed development thresholds allocate further research; they are not superiority/noninferiority tests or independent confirmation. All intervals are descriptive 95% seed bootstrap intervals with 20,000 resamples. No ablation is silently promoted.

## Per-architecture decisions

| Model | Fresh qualification | Combined outcome screen | Protection attribution |
|---|---|---|---|
| conditional | PASS | FAIL | FAIL |
| recurrent | PASS | FAIL | FAIL |

- conditional, screen: failed criteria = acquisition_gain, acquisition_consistency, valid_old_mode_1.
- recurrent, screen: failed criteria = acquisition_gain, acquisition_consistency, valid_old_mode_0, valid_old_mode_1.
- conditional, protection_attribution: failed criteria = retention_vs_shrink.
- recurrent, protection_attribution: failed criteria = retention_vs_current, retention_vs_shrink.

## Paired contrasts

Brier values are multiplied by 100; negative is better. Survival differences use percentage points. Each independent seed contributes one episode per arm.

| Model | Contrast | Acquisition AUC [95% interval] | Valid-old mean [95% interval] | Survival AUC [95% interval] |
|---|---|---:|---:|---:|
| conditional | current_minus_reference | +0.575 [+0.368, +0.804] | +0.237 [-0.194, +0.681] | -0.627 [-1.093, -0.197] |
| conditional | combined_minus_protected | +0.501 [+0.265, +0.736] | -0.036 [-0.853, +0.575] | -0.443 [-0.786, -0.105] |
| conditional | protected_minus_reference | -0.002 [-0.063, +0.062] | -0.244 [-0.645, +0.189] | -0.066 [-0.203, +0.049] |
| conditional | combined_minus_current | -0.076 [-0.136, -0.015] | -0.516 [-1.487, +0.205] | +0.119 [-0.094, +0.302] |
| conditional | current_main | +0.538 [+0.321, +0.760] | +0.101 [-0.190, +0.334] | -0.535 [-0.923, -0.181] |
| conditional | protection_main | -0.039 [-0.074, -0.003] | -0.380 [-0.843, -0.006] | +0.026 [-0.087, +0.132] |
| conditional | interaction | -0.074 [-0.161, +0.031] | -0.273 [-1.434, +0.739] | +0.185 [-0.076, +0.426] |
| conditional | combined_minus_reference | +0.499 [+0.315, +0.687] | -0.279 [-0.998, +0.187] | -0.509 [-0.852, -0.235] |
| conditional | combined_minus_shrink | -0.060 [-0.102, -0.008] | +0.494 [-1.089, +2.525] | +0.060 [-0.066, +0.190] |
| conditional | shrink_minus_current | -0.016 [-0.076, +0.045] | -1.010 [-3.896, +0.864] | +0.059 [-0.112, +0.254] |
| conditional | reference_minus_fresh | +0.240 [-0.029, +0.519] | -7.335 [-9.425, -4.523] | -0.188 [-1.246, +0.779] |
| conditional | combined_minus_fresh | +0.739 [+0.593, +0.885] | -7.615 [-9.256, -5.495] | -0.697 [-1.517, +0.068] |
| recurrent | current_minus_reference | +0.480 [+0.315, +0.672] | +0.148 [-2.859, +2.898] | -0.366 [-0.660, -0.070] |
| recurrent | combined_minus_protected | +0.599 [+0.361, +0.830] | +2.652 [+0.350, +6.080] | -0.405 [-0.624, -0.203] |
| recurrent | protected_minus_reference | -0.052 [-0.110, -0.008] | -1.194 [-3.507, +1.223] | +0.114 [-0.067, +0.251] |
| recurrent | combined_minus_current | +0.067 [-0.036, +0.187] | +1.310 [-0.441, +3.117] | +0.075 [-0.105, +0.290] |
| recurrent | current_main | +0.540 [+0.346, +0.742] | +1.400 [-0.815, +4.293] | -0.386 [-0.626, -0.145] |
| recurrent | protection_main | +0.008 [-0.049, +0.079] | +0.058 [-1.766, +1.641] | +0.095 [-0.048, +0.263] |
| recurrent | interaction | +0.119 [+0.002, +0.225] | +2.505 [-0.138, +4.929] | -0.039 [-0.234, +0.157] |
| recurrent | combined_minus_reference | +0.547 [+0.310, +0.795] | +1.458 [-1.076, +3.788] | -0.291 [-0.545, -0.044] |
| recurrent | combined_minus_shrink | +0.045 [-0.098, +0.165] | +1.784 [+0.555, +2.997] | -0.074 [-0.277, +0.128] |
| recurrent | shrink_minus_current | +0.022 [-0.095, +0.151] | -0.474 [-1.102, +0.195] | +0.150 [-0.081, +0.489] |
| recurrent | reference_minus_fresh | +0.434 [+0.027, +0.850] | -3.366 [-5.453, -1.274] | -0.640 [-1.471, +0.112] |
| recurrent | combined_minus_fresh | +0.981 [+0.783, +1.182] | -1.907 [-4.372, +0.531] | -0.931 [-1.673, -0.222] |

## Retention guardrails by old mode

Combined minus reference, Brier x100. The +.005 raw-Brier tolerance is checked in each mode, not only their average.

| Model | Mode | Difference [95% interval] | Gate | Interval crosses limit |
|---|---:|---:|---|---|
| conditional | 0 | -1.237 [-3.384, +0.196] | True | False |
| conditional | 1 | +0.679 [-0.028, +1.492] | False | True |
| recurrent | 0 | +1.810 [-3.028, +6.729] | False | True |
| recurrent | 1 | +1.106 [-0.446, +3.035] | False | True |

## Fresh-reference qualification

| Group | N | Marginal gain x100 | Cue benefit x100 | Pass |
|---|---:|---:|---:|---|
| conditional_stage_1 | 6 | 15.435 | 7.136 | True |
| conditional_cue_0 | 2 | 15.083 | 3.119 | True |
| conditional_cue_1 | 2 | 16.232 | 0.684 | True |
| conditional_cue_2 | 2 | 14.991 | 17.605 | True |
| recurrent_stage_1 | 6 | 15.498 | 6.682 | True |
| recurrent_cue_0 | 2 | 15.150 | 2.593 | True |
| recurrent_cue_1 | 2 | 16.500 | 0.779 | True |
| recurrent_cue_2 | 2 | 14.844 | 16.674 | True |

## Every episode

Brier and cue-benefit values x100; all raw metric values remain in summary.json.

| Seed | Model | Cue | Arm | Acquisition AUC | Valid-old mode 0 | Valid-old mode 1 | Cue benefit |
|---|---|---:|---|---:|---:|---:|---:|
| 13001 | conditional | 2 | combined | 9.719 | 8.187 | 10.350 | +15.180 |
| 13001 | conditional | 2 | current | 9.716 | 8.249 | 9.995 | +14.932 |
| 13001 | conditional | 2 | fresh | 8.734 | 7.555 | 27.253 | +16.929 |
| 13001 | conditional | 2 | protected | 9.491 | 8.235 | 10.373 | +14.784 |
| 13001 | conditional | 2 | reference | 9.476 | 7.674 | 10.555 | +14.837 |
| 13001 | conditional | 2 | shrink | 9.828 | 8.027 | 10.661 | +14.330 |
| 13001 | recurrent | 2 | combined | 10.027 | 8.154 | 37.170 | +14.112 |
| 13001 | recurrent | 2 | current | 10.151 | 8.593 | 26.596 | +14.767 |
| 13001 | recurrent | 2 | fresh | 8.921 | 7.516 | 44.061 | +16.052 |
| 13001 | recurrent | 2 | protected | 9.840 | 8.285 | 35.306 | +15.961 |
| 13001 | recurrent | 2 | reference | 9.843 | 8.570 | 38.946 | +14.645 |
| 13001 | recurrent | 2 | shrink | 10.115 | 7.979 | 28.863 | +13.475 |
| 13002 | conditional | 0 | combined | 8.861 | 9.762 | 7.561 | +2.313 |
| 13002 | conditional | 0 | current | 9.045 | 10.970 | 7.396 | +2.597 |
| 13002 | conditional | 0 | fresh | 8.146 | 26.126 | 6.928 | +2.975 |
| 13002 | conditional | 0 | protected | 7.932 | 10.326 | 5.818 | +3.730 |
| 13002 | conditional | 0 | reference | 7.970 | 11.342 | 6.405 | +2.623 |
| 13002 | conditional | 0 | shrink | 8.953 | 13.769 | 8.285 | +2.121 |
| 13002 | recurrent | 0 | combined | 9.305 | 39.440 | 7.003 | +2.145 |
| 13002 | recurrent | 0 | current | 9.202 | 37.859 | 7.227 | +3.128 |
| 13002 | recurrent | 0 | fresh | 8.323 | 43.332 | 7.184 | +2.178 |
| 13002 | recurrent | 0 | protected | 8.294 | 41.381 | 6.939 | +2.734 |
| 13002 | recurrent | 0 | reference | 8.339 | 34.049 | 6.749 | +2.474 |
| 13002 | recurrent | 0 | shrink | 9.122 | 35.421 | 7.686 | +1.218 |
| 13003 | conditional | 0 | combined | 8.362 | 8.481 | 12.065 | +2.299 |
| 13003 | conditional | 0 | current | 8.332 | 8.862 | 11.553 | +2.119 |
| 13003 | conditional | 0 | fresh | 7.582 | 8.248 | 26.970 | +3.263 |
| 13003 | conditional | 0 | protected | 7.574 | 8.203 | 11.414 | +2.929 |
| 13003 | conditional | 0 | reference | 7.701 | 8.228 | 12.543 | +1.841 |
| 13003 | conditional | 0 | shrink | 8.308 | 8.452 | 11.268 | +1.949 |
| 13003 | recurrent | 0 | combined | 8.677 | 10.855 | 46.425 | +0.258 |
| 13003 | recurrent | 0 | current | 8.636 | 8.775 | 45.171 | +1.251 |
| 13003 | recurrent | 0 | fresh | 7.603 | 7.635 | 43.755 | +3.008 |
| 13003 | recurrent | 0 | protected | 8.324 | 8.120 | 42.674 | +2.812 |
| 13003 | recurrent | 0 | reference | 8.343 | 8.190 | 44.787 | +2.333 |
| 13003 | recurrent | 0 | shrink | 8.461 | 7.723 | 45.231 | +1.321 |
| 13004 | conditional | 1 | combined | 7.408 | 9.909 | 7.820 | +0.615 |
| 13004 | conditional | 1 | current | 7.555 | 9.393 | 7.012 | +0.648 |
| 13004 | conditional | 1 | fresh | 6.843 | 27.893 | 7.721 | +0.605 |
| 13004 | conditional | 1 | protected | 6.881 | 8.999 | 6.902 | +0.184 |
| 13004 | conditional | 1 | reference | 6.864 | 10.023 | 7.506 | +0.038 |
| 13004 | conditional | 1 | shrink | 7.440 | 9.597 | 7.678 | -0.176 |
| 13004 | recurrent | 1 | combined | 7.693 | 27.803 | 7.836 | -0.113 |
| 13004 | recurrent | 1 | current | 7.359 | 33.328 | 7.060 | +0.111 |
| 13004 | recurrent | 1 | fresh | 7.075 | 43.262 | 6.876 | +0.739 |
| 13004 | recurrent | 1 | protected | 6.794 | 26.398 | 6.642 | +0.565 |
| 13004 | recurrent | 1 | reference | 6.785 | 35.903 | 7.073 | +0.597 |
| 13004 | recurrent | 1 | shrink | 7.530 | 28.494 | 8.608 | +0.207 |
| 13005 | conditional | 1 | combined | 7.903 | 7.074 | 13.958 | -0.031 |
| 13005 | conditional | 1 | current | 7.982 | 7.247 | 15.086 | -0.299 |
| 13005 | conditional | 1 | fresh | 7.436 | 8.551 | 33.592 | +0.764 |
| 13005 | conditional | 1 | protected | 7.447 | 7.148 | 14.326 | +0.114 |
| 13005 | conditional | 1 | reference | 7.455 | 7.199 | 13.060 | +0.488 |
| 13005 | conditional | 1 | shrink | 7.961 | 7.233 | 14.317 | +0.288 |
| 13005 | recurrent | 1 | combined | 8.140 | 8.101 | 38.344 | +0.381 |
| 13005 | recurrent | 1 | current | 8.127 | 7.693 | 34.491 | +0.392 |
| 13005 | recurrent | 1 | fresh | 7.422 | 7.817 | 39.593 | +0.818 |
| 13005 | recurrent | 1 | protected | 7.449 | 7.207 | 37.354 | +0.476 |
| 13005 | recurrent | 1 | reference | 7.515 | 8.821 | 32.867 | +0.365 |
| 13005 | recurrent | 1 | shrink | 8.101 | 7.304 | 35.504 | +0.442 |
| 13006 | conditional | 2 | combined | 10.274 | 16.947 | 8.444 | +15.856 |
| 13006 | conditional | 2 | current | 10.353 | 24.448 | 6.543 | +12.319 |
| 13006 | conditional | 2 | fresh | 9.353 | 24.576 | 6.520 | +18.281 |
| 13006 | conditional | 2 | protected | 10.199 | 22.202 | 7.038 | +15.199 |
| 13006 | conditional | 2 | reference | 10.068 | 23.318 | 6.055 | +15.767 |
| 13006 | conditional | 2 | shrink | 10.395 | 8.771 | 6.573 | +14.292 |
| 13006 | recurrent | 2 | combined | 10.381 | 34.487 | 7.457 | +15.054 |
| 13006 | recurrent | 2 | current | 10.345 | 33.030 | 7.528 | +15.433 |
| 13006 | recurrent | 2 | fresh | 8.993 | 37.671 | 7.260 | +17.296 |
| 13006 | recurrent | 2 | protected | 9.929 | 13.613 | 7.327 | +16.056 |
| 13006 | recurrent | 2 | reference | 10.115 | 22.448 | 7.172 | +16.094 |
| 13006 | recurrent | 2 | shrink | 10.627 | 30.627 | 8.222 | +14.635 |

## Resource and interpretation limits

All arms compute the replay reference gradient, hypothetical projection and final-update replay-loss diagnostic. These extra operations are recorded separately from the original presentation budget. The deployable ordinary reference does not require discarded diagnostic work. Full counters, scalar summaries, parameter/optimizer/replay bytes and episode wall time are preserved in resource_rows.

The constraint is first-order reported BCE on one sampled historical packet; finite-step replay loss can still increase, and this is not protection of every old relationship or clean physical outcome. These known-boundary first-introduction forks do not demonstrate autonomous lifelong learning, recurrence, noise robustness, late acquisition or compounding. Fresh-model damage is not a forgetting control because its starting errors differ; interpret terminal levels separately.

The portable record check validates source/configuration/protocol/analysis locks, complete paired inputs, recorded forks and resource counts. A separate full checkpoint audit verifies actual tensor states and probes. Neither check is independent scientific replication.
