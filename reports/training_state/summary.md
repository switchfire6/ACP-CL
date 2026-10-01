# Optimizer-by-replay diagnostic results

Kind: diagnostic; 6 independent seeds; 372 episodes.
Fresh-model qualification: **True**; required cue benefit 0.002.

## Qualification

| Group | Marginal Brier improvement | Cue benefit | Pass |
|---|---:|---:|---|
| conditional_stage_1 | 0.155646 | 0.075286 | True |
| conditional_stage_2 | 0.148779 | 0.067563 | True |
| conditional_stage_3 | 0.144402 | 0.070750 | True |
| conditional_cue_0 | 0.151550 | 0.023232 | True |
| conditional_cue_1 | 0.151509 | 0.005120 | True |
| conditional_cue_2 | 0.145769 | 0.185247 | True |
| recurrent_stage_1 | 0.154423 | 0.073327 | True |
| recurrent_stage_2 | 0.149600 | 0.069069 | True |
| recurrent_stage_3 | 0.143735 | 0.070046 | True |
| recurrent_cue_0 | 0.150933 | 0.023486 | True |
| recurrent_cue_1 | 0.151412 | 0.005016 | True |
| recurrent_cue_2 | 0.145413 | 0.183941 | True |

## Primary factorial contrasts

Brier units x100. Negative error differences favor the named intervention. Three episodes averaged within seed; intervals are descriptive. Interaction measures departure from additivity. Fresh-weight damage contrasts are not forgetting controls because their initial errors differ; use terminal levels as well.

| Model | Contrast | Acquisition difference [95% interval] | Valid-old damage difference [95% interval] |
|---|---|---:|---:|
| conditional | optimizer_with_replay_kept | +0.066 [-0.014, +0.155] | -0.237 [-0.899, +0.513] |
| conditional | optimizer_with_replay_reset | +0.038 [-0.029, +0.131] | +1.926 [+0.849, +2.925] |
| conditional | replay_with_optimizer_kept | -0.974 [-1.226, -0.721] | -1.209 [-2.337, -0.063] |
| conditional | replay_with_optimizer_reset | -1.002 [-1.270, -0.729] | +0.954 [-0.966, +2.939] |
| conditional | optimizer_main | +0.052 [-0.014, +0.136] | +0.845 [+0.074, +1.620] |
| conditional | replay_main | -0.988 [-1.243, -0.727] | -0.128 [-1.663, +1.447] |
| conditional | interaction | -0.028 [-0.104, +0.030] | +2.163 [+1.349, +3.006] |
| conditional | both_minus_continue | -0.936 [-1.220, -0.621] | +0.717 [-1.475, +2.693] |
| conditional | continue_minus_fresh | +0.632 [+0.256, +0.858] | +8.734 [+7.169, +10.361] |
| conditional | both_minus_fresh | -0.304 [-0.598, +0.003] | +9.451 [+7.301, +11.508] |
| recurrent | optimizer_with_replay_kept | -0.002 [-0.029, +0.029] | +0.487 [-0.016, +1.128] |
| recurrent | optimizer_with_replay_reset | +0.063 [+0.011, +0.114] | +0.348 [-0.141, +0.993] |
| recurrent | replay_with_optimizer_kept | -0.963 [-1.137, -0.775] | +1.790 [+0.988, +2.586] |
| recurrent | replay_with_optimizer_reset | -0.898 [-1.042, -0.722] | +1.651 [+0.813, +2.644] |
| recurrent | optimizer_main | +0.030 [+0.006, +0.055] | +0.417 [+0.055, +0.845] |
| recurrent | replay_main | -0.930 [-1.083, -0.747] | +1.721 [+0.932, +2.500] |
| recurrent | interaction | +0.065 [-0.007, +0.130] | -0.139 [-0.873, +0.740] |
| recurrent | both_minus_continue | -0.900 [-1.040, -0.733] | +2.138 [+1.145, +3.106] |
| recurrent | continue_minus_fresh | +0.830 [+0.446, +1.083] | +3.040 [+2.135, +4.251] |
| recurrent | both_minus_fresh | -0.070 [-0.328, +0.146] | +5.178 [+3.684, +6.797] |

## Final branch: return

Brier units x100; seed-paired descriptive intervals.

| Model | Contrast | Prediction AUC difference [95% interval] | Valid-old damage difference [95% interval] |
|---|---|---:|---:|
| conditional | optimizer_with_replay_kept | -0.924 [-2.081, +0.227] | -1.032 [-2.000, -0.061] |
| conditional | optimizer_with_replay_reset | -0.069 [-0.238, +0.152] | -3.042 [-5.244, -1.036] |
| conditional | replay_with_optimizer_kept | -3.147 [-5.638, -0.763] | +6.374 [+2.830, +10.330] |
| conditional | replay_with_optimizer_reset | -2.293 [-4.755, -0.554] | +4.364 [+1.189, +8.040] |
| conditional | optimizer_main | -0.496 [-1.086, +0.036] | -2.037 [-3.165, -0.897] |
| conditional | replay_main | -2.720 [-5.171, -0.672] | +5.369 [+2.134, +9.073] |
| conditional | interaction | +0.854 [-0.396, +2.202] | -2.010 [-4.675, +0.088] |
| conditional | both_minus_continue | -3.216 [-5.627, -0.998] | +3.332 [+0.362, +6.628] |
| recurrent | optimizer_with_replay_kept | +0.377 [-0.275, +1.148] | -0.266 [-1.542, +1.060] |
| recurrent | optimizer_with_replay_reset | +0.111 [-0.004, +0.236] | -0.110 [-0.660, +0.357] |
| recurrent | replay_with_optimizer_kept | -1.768 [-2.356, -1.246] | +8.219 [+4.581, +11.593] |
| recurrent | replay_with_optimizer_reset | -2.033 [-2.819, -1.264] | +8.374 [+4.519, +12.466] |
| recurrent | optimizer_main | +0.244 [-0.106, +0.619] | -0.188 [-0.984, +0.525] |
| recurrent | replay_main | -1.901 [-2.497, -1.322] | +8.296 [+4.495, +12.044] |
| recurrent | interaction | -0.266 [-1.071, +0.358] | +0.156 [-1.203, +1.316] |
| recurrent | both_minus_continue | -1.657 [-2.169, -1.176] | +8.108 [+4.678, +11.478] |

## Final branch: revision

Brier units x100; seed-paired descriptive intervals.

| Model | Contrast | Prediction AUC difference [95% interval] | Valid-old damage difference [95% interval] |
|---|---|---:|---:|
| conditional | optimizer_with_replay_kept | +0.008 [-0.243, +0.224] | +0.702 [-1.475, +2.994] |
| conditional | optimizer_with_replay_reset | +0.030 [-0.045, +0.101] | +1.458 [-0.239, +3.732] |
| conditional | replay_with_optimizer_kept | -0.764 [-1.270, -0.283] | -1.047 [-4.307, +0.863] |
| conditional | replay_with_optimizer_reset | -0.741 [-1.084, -0.410] | -0.290 [-2.586, +1.544] |
| conditional | optimizer_main | +0.019 [-0.092, +0.110] | +1.080 [+0.438, +1.718] |
| conditional | replay_main | -0.753 [-1.151, -0.363] | -0.668 [-2.198, +0.678] |
| conditional | interaction | +0.023 [-0.236, +0.299] | +0.757 [-2.936, +5.314] |
| conditional | both_minus_continue | -0.734 [-1.170, -0.332] | +0.412 [-0.859, +1.873] |
| recurrent | optimizer_with_replay_kept | +0.089 [+0.035, +0.143] | -0.788 [-1.287, -0.278] |
| recurrent | optimizer_with_replay_reset | -0.010 [-0.123, +0.103] | -0.455 [-0.754, -0.155] |
| recurrent | replay_with_optimizer_kept | -0.526 [-0.875, -0.183] | +0.133 [-0.504, +0.814] |
| recurrent | replay_with_optimizer_reset | -0.626 [-0.928, -0.334] | +0.466 [-0.003, +0.886] |
| recurrent | optimizer_main | +0.040 [-0.031, +0.113] | -0.622 [-0.858, -0.312] |
| recurrent | replay_main | -0.576 [-0.888, -0.264] | +0.300 [-0.136, +0.782] |
| recurrent | interaction | -0.099 [-0.209, +0.005] | +0.333 [-0.338, +0.951] |
| recurrent | both_minus_continue | -0.536 [-0.882, -0.203] | -0.322 [-0.805, +0.161] |

## Final branch: clean

Brier units x100; seed-paired descriptive intervals.

| Model | Contrast | Prediction AUC difference [95% interval] | Valid-old damage difference [95% interval] |
|---|---|---:|---:|
| conditional | optimizer_with_replay_kept | +0.073 [-0.007, +0.159] | +0.884 [-1.749, +3.098] |
| conditional | optimizer_with_replay_reset | -0.070 [-0.160, +0.020] | -0.554 [-2.409, +0.937] |
| conditional | replay_with_optimizer_kept | -0.357 [-0.735, -0.011] | +1.095 [+0.071, +2.244] |
| conditional | replay_with_optimizer_reset | -0.499 [-0.800, -0.229] | -0.343 [-2.502, +2.322] |
| conditional | optimizer_main | +0.001 [-0.029, +0.026] | +0.165 [-1.336, +1.619] |
| conditional | replay_main | -0.428 [-0.762, -0.130] | +0.376 [-0.738, +1.571] |
| conditional | interaction | -0.142 [-0.300, +0.015] | -1.438 [-4.056, +1.789] |
| conditional | both_minus_continue | -0.427 [-0.759, -0.126] | +0.541 [-1.650, +2.766] |
| recurrent | optimizer_with_replay_kept | +0.018 [-0.085, +0.117] | -0.274 [-2.136, +1.164] |
| recurrent | optimizer_with_replay_reset | -0.025 [-0.092, +0.028] | +0.135 [-0.572, +0.730] |
| recurrent | replay_with_optimizer_kept | -0.354 [-0.673, -0.040] | +0.167 [-0.508, +0.880] |
| recurrent | replay_with_optimizer_reset | -0.397 [-0.650, -0.159] | +0.576 [-0.821, +2.469] |
| recurrent | optimizer_main | -0.004 [-0.071, +0.056] | -0.069 [-0.820, +0.626] |
| recurrent | replay_main | -0.376 [-0.654, -0.102] | +0.371 [-0.263, +1.127] |
| recurrent | interaction | -0.043 [-0.157, +0.075] | +0.410 [-1.474, +2.628] |
| recurrent | both_minus_continue | -0.380 [-0.678, -0.090] | +0.302 [-0.229, +0.736] |

## Final branch: noise

Brier units x100; seed-paired descriptive intervals.

| Model | Contrast | Prediction AUC difference [95% interval] | Valid-old damage difference [95% interval] |
|---|---|---:|---:|
| conditional | optimizer_with_replay_kept | -0.192 [-0.303, -0.082] | -1.774 [-3.667, +0.059] |
| conditional | optimizer_with_replay_reset | -0.253 [-0.381, -0.125] | +1.132 [-1.959, +4.509] |
| conditional | replay_with_optimizer_kept | +1.139 [+0.771, +1.451] | -0.770 [-2.495, +1.662] |
| conditional | replay_with_optimizer_reset | +1.079 [+0.860, +1.295] | +2.136 [+0.458, +4.637] |
| conditional | optimizer_main | -0.223 [-0.304, -0.132] | -0.321 [-2.355, +1.809] |
| conditional | replay_main | +1.109 [+0.817, +1.370] | +0.683 [-0.758, +2.238] |
| conditional | interaction | -0.060 [-0.212, +0.128] | +2.906 [-0.581, +5.834] |
| conditional | both_minus_continue | +0.886 [+0.569, +1.183] | +0.362 [-2.094, +3.433] |
| recurrent | optimizer_with_replay_kept | -0.212 [-0.434, +0.054] | +0.482 [-0.294, +1.236] |
| recurrent | optimizer_with_replay_reset | -0.287 [-0.527, -0.041] | -0.363 [-1.023, +0.294] |
| recurrent | replay_with_optimizer_kept | +0.944 [+0.528, +1.259] | +0.950 [+0.682, +1.275] |
| recurrent | replay_with_optimizer_reset | +0.869 [+0.502, +1.217] | +0.104 [-0.938, +1.318] |
| recurrent | optimizer_main | -0.249 [-0.453, -0.011] | +0.059 [-0.285, +0.421] |
| recurrent | replay_main | +0.906 [+0.532, +1.238] | +0.527 [+0.026, +1.093] |
| recurrent | interaction | -0.074 [-0.266, +0.118] | -0.845 [-2.016, +0.468] |
| recurrent | both_minus_continue | +0.657 [+0.272, +1.042] | +0.586 [-0.029, +1.202] |

## Valid-old levels

Brier units x100. Fresh-weight starts are different; change from random initialization is not forgetting.

| Model | Branch | Arm | Before | After | Change |
|---|---|---|---:|---:|---:|
| conditional | novel | continue | 11.642 | 13.934 | +2.291 |
| conditional | novel | optimizer_reset | 11.642 | 13.697 | +2.054 |
| conditional | novel | replay_reset | 11.642 | 12.724 | +1.082 |
| conditional | novel | state_reset | 11.642 | 14.651 | +3.008 |
| conditional | novel | fresh | 25.080 | 18.637 | -6.442 |
| conditional | return | continue | 17.593 | 13.232 | -4.361 |
| conditional | return | optimizer_reset | 17.593 | 12.200 | -5.393 |
| conditional | return | replay_reset | 17.593 | 19.606 | +2.012 |
| conditional | return | state_reset | 17.593 | 16.564 | -1.030 |
| conditional | revision | continue | 15.383 | 16.207 | +0.824 |
| conditional | revision | optimizer_reset | 15.383 | 16.909 | +1.526 |
| conditional | revision | replay_reset | 15.383 | 15.161 | -0.222 |
| conditional | revision | state_reset | 15.383 | 16.619 | +1.236 |
| conditional | clean | continue | 16.861 | 14.964 | -1.898 |
| conditional | clean | optimizer_reset | 16.861 | 15.848 | -1.014 |
| conditional | clean | replay_reset | 16.861 | 16.059 | -0.803 |
| conditional | clean | state_reset | 16.861 | 15.505 | -1.356 |
| conditional | noise | continue | 16.861 | 19.142 | +2.280 |
| conditional | noise | optimizer_reset | 16.861 | 17.368 | +0.507 |
| conditional | noise | replay_reset | 16.861 | 18.372 | +1.511 |
| conditional | noise | state_reset | 16.861 | 19.504 | +2.643 |
| recurrent | novel | continue | 19.884 | 22.681 | +2.797 |
| recurrent | novel | optimizer_reset | 19.884 | 23.168 | +3.284 |
| recurrent | novel | replay_reset | 19.884 | 24.472 | +4.587 |
| recurrent | novel | state_reset | 19.884 | 24.820 | +4.935 |
| recurrent | novel | fresh | 25.091 | 24.849 | -0.242 |
| recurrent | return | continue | 25.447 | 16.457 | -8.991 |
| recurrent | return | optimizer_reset | 25.447 | 16.191 | -9.257 |
| recurrent | return | replay_reset | 25.447 | 24.675 | -0.772 |
| recurrent | return | state_reset | 25.447 | 24.565 | -0.882 |
| recurrent | revision | continue | 22.517 | 23.583 | +1.066 |
| recurrent | revision | optimizer_reset | 22.517 | 22.795 | +0.278 |
| recurrent | revision | replay_reset | 22.517 | 23.716 | +1.200 |
| recurrent | revision | state_reset | 22.517 | 23.261 | +0.744 |
| recurrent | clean | continue | 24.234 | 24.382 | +0.148 |
| recurrent | clean | optimizer_reset | 24.234 | 24.108 | -0.127 |
| recurrent | clean | replay_reset | 24.234 | 24.548 | +0.314 |
| recurrent | clean | state_reset | 24.234 | 24.684 | +0.450 |
| recurrent | noise | continue | 24.234 | 23.705 | -0.529 |
| recurrent | noise | optimizer_reset | 24.234 | 24.188 | -0.047 |
| recurrent | noise | replay_reset | 24.234 | 24.655 | +0.421 |
| recurrent | noise | state_reset | 24.234 | 24.292 | +0.058 |

## Matched noise minus clean continuation

Positive Brier differences mean harm from the noisy branch.

| Model | Arm | Acquisition difference | Valid-old damage difference |
|---|---|---:|---:|
| conditional | continue | +1.567 | +4.178 |
| conditional | optimizer_reset | +1.302 | +1.521 |
| conditional | replay_reset | +3.063 | +2.314 |
| conditional | state_reset | +2.880 | +3.999 |
| recurrent | continue | +1.992 | -0.676 |
| recurrent | optimizer_reset | +1.762 | +0.080 |
| recurrent | replay_reset | +3.290 | +0.107 |
| recurrent | state_reset | +3.028 | -0.392 |

## Every episode

Brier errors multiplied by 100; lower is better. Positive valid damage is deterioration.

| Seed | Model | Stage | Arm/branch | Cue | Brier AUC | Valid damage | Cue benefit |
|---|---|---:|---|---|---:|---:|---:|
| 8001 | conditional | 4 | continue/clean | None | 8.674 | -8.285 | n/a |
| 8001 | conditional | 4 | optimizer_reset/clean | None | 8.913 | -4.845 | n/a |
| 8001 | conditional | 4 | replay_reset/clean | None | 8.381 | -6.025 | n/a |
| 8001 | conditional | 4 | state_reset/clean | None | 8.185 | -7.696 | n/a |
| 8001 | conditional | 4 | continue/noise | None | 10.201 | +1.973 | n/a |
| 8001 | conditional | 4 | optimizer_reset/noise | None | 10.010 | -2.651 | n/a |
| 8001 | conditional | 4 | replay_reset/noise | None | 11.369 | -0.409 | n/a |
| 8001 | conditional | 4 | state_reset/noise | None | 10.995 | -0.565 | n/a |
| 8001 | conditional | 4 | continue/return | None | 8.712 | -6.739 | n/a |
| 8001 | conditional | 4 | optimizer_reset/return | None | 8.887 | -6.461 | n/a |
| 8001 | conditional | 4 | replay_reset/return | None | 8.617 | +7.439 | n/a |
| 8001 | conditional | 4 | state_reset/return | None | 8.295 | -0.102 | n/a |
| 8001 | conditional | 4 | continue/revision | 0 | 9.258 | -0.768 | +0.430 |
| 8001 | conditional | 4 | optimizer_reset/revision | 0 | 9.560 | +0.142 | -0.658 |
| 8001 | conditional | 4 | replay_reset/revision | 0 | 8.738 | -0.973 | +0.549 |
| 8001 | conditional | 4 | state_reset/revision | 0 | 8.742 | -0.051 | +0.919 |
| 8001 | conditional | 1 | continue/novel | 1 | 7.997 | +0.533 | +0.470 |
| 8001 | conditional | 1 | fresh/novel | 1 | 7.973 | -6.016 | +0.495 |
| 8001 | conditional | 1 | optimizer_reset/novel | 1 | 8.074 | +7.825 | +0.398 |
| 8001 | conditional | 1 | replay_reset/novel | 1 | 7.784 | +0.222 | +0.725 |
| 8001 | conditional | 1 | state_reset/novel | 1 | 7.690 | +1.597 | +0.972 |
| 8001 | conditional | 2 | continue/novel | 2 | 11.275 | +8.353 | +11.458 |
| 8001 | conditional | 2 | fresh/novel | 2 | 8.829 | -5.345 | +18.887 |
| 8001 | conditional | 2 | optimizer_reset/novel | 2 | 11.490 | +4.580 | +12.081 |
| 8001 | conditional | 2 | replay_reset/novel | 2 | 9.440 | +6.203 | +16.905 |
| 8001 | conditional | 2 | state_reset/novel | 2 | 9.184 | +3.775 | +17.470 |
| 8001 | conditional | 3 | continue/novel | 0 | 9.632 | +5.507 | +0.539 |
| 8001 | conditional | 3 | fresh/novel | 0 | 9.418 | -3.744 | +1.674 |
| 8001 | conditional | 3 | optimizer_reset/novel | 0 | 9.815 | -0.860 | +1.390 |
| 8001 | conditional | 3 | replay_reset/novel | 0 | 8.549 | -2.061 | +1.219 |
| 8001 | conditional | 3 | state_reset/novel | 0 | 8.771 | -1.192 | +1.719 |
| 8002 | conditional | 4 | continue/clean | None | 8.088 | -3.213 | n/a |
| 8002 | conditional | 4 | optimizer_reset/clean | None | 8.165 | -3.091 | n/a |
| 8002 | conditional | 4 | replay_reset/clean | None | 8.091 | -3.590 | n/a |
| 8002 | conditional | 4 | state_reset/clean | None | 7.882 | -3.022 | n/a |
| 8002 | conditional | 4 | continue/noise | None | 9.719 | +0.132 | n/a |
| 8002 | conditional | 4 | optimizer_reset/noise | None | 9.657 | -1.586 | n/a |
| 8002 | conditional | 4 | replay_reset/noise | None | 11.156 | -0.704 | n/a |
| 8002 | conditional | 4 | state_reset/noise | None | 11.128 | -0.301 | n/a |
| 8002 | conditional | 4 | continue/return | None | 15.686 | -4.608 | n/a |
| 8002 | conditional | 4 | optimizer_reset/return | None | 16.092 | -5.042 | n/a |
| 8002 | conditional | 4 | replay_reset/return | None | 8.120 | -1.990 | n/a |
| 8002 | conditional | 4 | state_reset/return | None | 8.132 | -1.601 | n/a |
| 8002 | conditional | 4 | continue/revision | 0 | 8.884 | +0.725 | +0.495 |
| 8002 | conditional | 4 | optimizer_reset/revision | 0 | 9.094 | +0.405 | +0.577 |
| 8002 | conditional | 4 | replay_reset/revision | 0 | 8.647 | +0.658 | +0.566 |
| 8002 | conditional | 4 | state_reset/revision | 0 | 8.673 | +0.512 | +1.115 |
| 8002 | conditional | 1 | continue/novel | 2 | 11.597 | +2.028 | +15.529 |
| 8002 | conditional | 1 | fresh/novel | 2 | 8.794 | -5.760 | +18.768 |
| 8002 | conditional | 1 | optimizer_reset/novel | 2 | 11.705 | +3.105 | +15.943 |
| 8002 | conditional | 1 | replay_reset/novel | 2 | 8.571 | +2.072 | +19.023 |
| 8002 | conditional | 1 | state_reset/novel | 2 | 8.698 | +3.567 | +17.640 |
| 8002 | conditional | 2 | continue/novel | 0 | 9.501 | +4.011 | +0.888 |
| 8002 | conditional | 2 | fresh/novel | 0 | 9.083 | -4.357 | +1.699 |
| 8002 | conditional | 2 | optimizer_reset/novel | 0 | 9.525 | +4.170 | +1.405 |
| 8002 | conditional | 2 | replay_reset/novel | 0 | 8.500 | +0.588 | +1.574 |
| 8002 | conditional | 2 | state_reset/novel | 0 | 8.432 | +0.382 | +0.807 |
| 8002 | conditional | 3 | continue/novel | 1 | 8.154 | +0.127 | +0.805 |
| 8002 | conditional | 3 | fresh/novel | 1 | 8.913 | -5.218 | +0.916 |
| 8002 | conditional | 3 | optimizer_reset/novel | 1 | 8.135 | -3.403 | +0.919 |
| 8002 | conditional | 3 | replay_reset/novel | 1 | 8.115 | -3.639 | +0.384 |
| 8002 | conditional | 3 | state_reset/novel | 1 | 8.243 | -3.163 | -0.143 |
| 8003 | conditional | 4 | continue/clean | None | 7.717 | -0.595 | n/a |
| 8003 | conditional | 4 | optimizer_reset/clean | None | 7.880 | +1.841 | n/a |
| 8003 | conditional | 4 | replay_reset/clean | None | 7.961 | -0.098 | n/a |
| 8003 | conditional | 4 | state_reset/clean | None | 7.884 | +0.035 | n/a |
| 8003 | conditional | 4 | continue/noise | None | 9.059 | +7.519 | n/a |
| 8003 | conditional | 4 | optimizer_reset/noise | None | 9.064 | +2.545 | n/a |
| 8003 | conditional | 4 | replay_reset/noise | None | 10.492 | +5.246 | n/a |
| 8003 | conditional | 4 | state_reset/noise | None | 10.250 | +3.930 | n/a |
| 8003 | conditional | 4 | continue/return | None | 8.701 | -0.913 | n/a |
| 8003 | conditional | 4 | optimizer_reset/return | None | 8.760 | -2.613 | n/a |
| 8003 | conditional | 4 | replay_reset/return | None | 8.353 | +2.310 | n/a |
| 8003 | conditional | 4 | state_reset/return | None | 8.203 | -3.033 | n/a |
| 8003 | conditional | 4 | continue/revision | 0 | 8.037 | -0.358 | +0.536 |
| 8003 | conditional | 4 | optimizer_reset/revision | 0 | 8.141 | +5.130 | +1.052 |
| 8003 | conditional | 4 | replay_reset/revision | 0 | 8.088 | +0.552 | +1.603 |
| 8003 | conditional | 4 | state_reset/revision | 0 | 7.960 | -0.413 | +1.554 |
| 8003 | conditional | 1 | continue/novel | 2 | 9.889 | -1.662 | +17.268 |
| 8003 | conditional | 1 | fresh/novel | 2 | 8.515 | -4.695 | +18.601 |
| 8003 | conditional | 1 | optimizer_reset/novel | 2 | 9.904 | -0.672 | +16.300 |
| 8003 | conditional | 1 | replay_reset/novel | 2 | 8.243 | +3.745 | +19.636 |
| 8003 | conditional | 1 | state_reset/novel | 2 | 8.028 | +6.395 | +18.877 |
| 8003 | conditional | 2 | continue/novel | 1 | 7.639 | +1.804 | +0.018 |
| 8003 | conditional | 2 | fresh/novel | 1 | 8.572 | -6.992 | +0.054 |
| 8003 | conditional | 2 | optimizer_reset/novel | 1 | 7.530 | +0.747 | +0.804 |
| 8003 | conditional | 2 | replay_reset/novel | 1 | 7.458 | +1.662 | +0.058 |
| 8003 | conditional | 2 | state_reset/novel | 1 | 7.642 | +1.211 | +0.714 |
| 8003 | conditional | 3 | continue/novel | 0 | 8.233 | +1.188 | +0.819 |
| 8003 | conditional | 3 | fresh/novel | 0 | 9.522 | -4.499 | +1.724 |
| 8003 | conditional | 3 | optimizer_reset/novel | 0 | 8.328 | +0.630 | +0.735 |
| 8003 | conditional | 3 | replay_reset/novel | 0 | 8.234 | -0.876 | +1.518 |
| 8003 | conditional | 3 | state_reset/novel | 0 | 8.437 | +6.736 | +1.526 |
| 8004 | conditional | 4 | continue/clean | None | 8.964 | -4.025 | n/a |
| 8004 | conditional | 4 | optimizer_reset/clean | None | 8.893 | +0.374 | n/a |
| 8004 | conditional | 4 | replay_reset/clean | None | 8.293 | -0.495 | n/a |
| 8004 | conditional | 4 | state_reset/clean | None | 8.399 | +1.025 | n/a |
| 8004 | conditional | 4 | continue/noise | None | 10.639 | +1.507 | n/a |
| 8004 | conditional | 4 | optimizer_reset/noise | None | 10.403 | +3.029 | n/a |
| 8004 | conditional | 4 | replay_reset/noise | None | 11.490 | -1.879 | n/a |
| 8004 | conditional | 4 | state_reset/noise | None | 11.219 | +2.679 | n/a |
| 8004 | conditional | 4 | continue/return | None | 11.863 | -0.335 | n/a |
| 8004 | conditional | 4 | optimizer_reset/return | None | 8.671 | -2.651 | n/a |
| 8004 | conditional | 4 | replay_reset/return | None | 8.395 | +3.603 | n/a |
| 8004 | conditional | 4 | state_reset/return | None | 8.214 | +1.344 | n/a |
| 8004 | conditional | 4 | continue/revision | 0 | 10.396 | +0.009 | +0.626 |
| 8004 | conditional | 4 | optimizer_reset/revision | 0 | 10.609 | +1.189 | +1.114 |
| 8004 | conditional | 4 | replay_reset/revision | 0 | 9.315 | +0.687 | +1.001 |
| 8004 | conditional | 4 | state_reset/revision | 0 | 9.350 | +0.553 | +2.322 |
| 8004 | conditional | 1 | continue/novel | 0 | 7.614 | +0.324 | +2.910 |
| 8004 | conditional | 1 | fresh/novel | 0 | 7.443 | -7.819 | +4.015 |
| 8004 | conditional | 1 | optimizer_reset/novel | 0 | 7.689 | +0.062 | +1.461 |
| 8004 | conditional | 1 | replay_reset/novel | 0 | 7.307 | -1.489 | +3.657 |
| 8004 | conditional | 1 | state_reset/novel | 0 | 7.484 | -0.509 | +3.131 |
| 8004 | conditional | 2 | continue/novel | 1 | 8.022 | -0.348 | -0.331 |
| 8004 | conditional | 2 | fresh/novel | 1 | 8.170 | -6.201 | +0.565 |
| 8004 | conditional | 2 | optimizer_reset/novel | 1 | 8.155 | +0.692 | +0.290 |
| 8004 | conditional | 2 | replay_reset/novel | 1 | 7.942 | +1.363 | +1.004 |
| 8004 | conditional | 2 | state_reset/novel | 1 | 8.177 | +3.498 | +1.394 |
| 8004 | conditional | 3 | continue/novel | 2 | 11.289 | +5.725 | +12.267 |
| 8004 | conditional | 3 | fresh/novel | 2 | 9.271 | -9.027 | +18.888 |
| 8004 | conditional | 3 | optimizer_reset/novel | 2 | 11.823 | +8.598 | +11.050 |
| 8004 | conditional | 3 | replay_reset/novel | 2 | 9.948 | +1.688 | +16.988 |
| 8004 | conditional | 3 | state_reset/novel | 2 | 10.263 | +8.919 | +16.342 |
| 8005 | conditional | 4 | continue/clean | None | 8.355 | +5.054 | n/a |
| 8005 | conditional | 4 | optimizer_reset/clean | None | 8.391 | +4.923 | n/a |
| 8005 | conditional | 4 | replay_reset/clean | None | 8.086 | +5.951 | n/a |
| 8005 | conditional | 4 | state_reset/clean | None | 8.016 | +1.138 | n/a |
| 8005 | conditional | 4 | continue/noise | None | 9.891 | +2.530 | n/a |
| 8005 | conditional | 4 | optimizer_reset/noise | None | 9.659 | +1.788 | n/a |
| 8005 | conditional | 4 | replay_reset/noise | None | 11.506 | +1.821 | n/a |
| 8005 | conditional | 4 | state_reset/noise | None | 10.994 | +9.809 | n/a |
| 8005 | conditional | 4 | continue/return | None | 8.374 | -6.135 | n/a |
| 8005 | conditional | 4 | optimizer_reset/return | None | 8.531 | -5.539 | n/a |
| 8005 | conditional | 4 | replay_reset/return | None | 8.089 | -4.092 | n/a |
| 8005 | conditional | 4 | state_reset/return | None | 7.863 | -5.270 | n/a |
| 8005 | conditional | 4 | continue/revision | 0 | 9.956 | +0.493 | -0.441 |
| 8005 | conditional | 4 | optimizer_reset/revision | 0 | 9.461 | +1.209 | +0.006 |
| 8005 | conditional | 4 | replay_reset/revision | 0 | 8.951 | +1.815 | +1.579 |
| 8005 | conditional | 4 | state_reset/revision | 0 | 9.022 | +4.089 | +1.919 |
| 8005 | conditional | 1 | continue/novel | 0 | 7.274 | +4.744 | +2.422 |
| 8005 | conditional | 1 | fresh/novel | 0 | 7.228 | -6.089 | +2.629 |
| 8005 | conditional | 1 | optimizer_reset/novel | 0 | 7.212 | +1.070 | +2.020 |
| 8005 | conditional | 1 | replay_reset/novel | 0 | 6.916 | +3.698 | +2.054 |
| 8005 | conditional | 1 | state_reset/novel | 0 | 6.815 | +3.087 | +1.746 |
| 8005 | conditional | 2 | continue/novel | 2 | 11.258 | +0.876 | +12.006 |
| 8005 | conditional | 2 | fresh/novel | 2 | 9.192 | -8.801 | +17.134 |
| 8005 | conditional | 2 | optimizer_reset/novel | 2 | 11.381 | -3.549 | +12.195 |
| 8005 | conditional | 2 | replay_reset/novel | 2 | 8.630 | +0.662 | +18.422 |
| 8005 | conditional | 2 | state_reset/novel | 2 | 8.610 | +3.421 | +18.261 |
| 8005 | conditional | 3 | continue/novel | 1 | 9.028 | -3.760 | +0.847 |
| 8005 | conditional | 3 | fresh/novel | 1 | 8.716 | -7.361 | +0.378 |
| 8005 | conditional | 3 | optimizer_reset/novel | 1 | 9.049 | +0.425 | +0.340 |
| 8005 | conditional | 3 | replay_reset/novel | 1 | 7.971 | -2.573 | +0.870 |
| 8005 | conditional | 3 | state_reset/novel | 1 | 7.980 | +2.310 | +0.758 |
| 8006 | conditional | 4 | continue/clean | None | 10.033 | -0.324 | n/a |
| 8006 | conditional | 4 | optimizer_reset/clean | None | 10.025 | -5.284 | n/a |
| 8006 | conditional | 4 | replay_reset/clean | None | 8.876 | -0.559 | n/a |
| 8006 | conditional | 4 | state_reset/clean | None | 8.904 | +0.382 | n/a |
| 8006 | conditional | 4 | continue/noise | None | 11.722 | +0.023 | n/a |
| 8006 | conditional | 4 | optimizer_reset/noise | None | 11.283 | -0.084 | n/a |
| 8006 | conditional | 4 | replay_reset/noise | None | 12.053 | +4.990 | n/a |
| 8006 | conditional | 4 | state_reset/noise | None | 11.963 | +0.305 | n/a |
| 8006 | conditional | 4 | continue/return | None | 15.931 | -7.439 | n/a |
| 8006 | conditional | 4 | optimizer_reset/return | None | 12.785 | -10.054 | n/a |
| 8006 | conditional | 4 | replay_reset/return | None | 8.813 | +4.803 | n/a |
| 8006 | conditional | 4 | state_reset/return | None | 9.262 | +2.485 | n/a |
| 8006 | conditional | 4 | continue/revision | 0 | 10.382 | +4.846 | +0.326 |
| 8006 | conditional | 4 | optimizer_reset/revision | 0 | 10.095 | +1.083 | -0.012 |
| 8006 | conditional | 4 | replay_reset/revision | 0 | 8.590 | -4.073 | +0.859 |
| 8006 | conditional | 4 | state_reset/revision | 0 | 8.765 | +2.727 | +1.373 |
| 8006 | conditional | 1 | continue/novel | 1 | 7.165 | +1.532 | +0.607 |
| 8006 | conditional | 1 | fresh/novel | 1 | 7.467 | -9.327 | +0.663 |
| 8006 | conditional | 1 | optimizer_reset/novel | 1 | 7.094 | +1.279 | +0.643 |
| 8006 | conditional | 1 | replay_reset/novel | 1 | 6.771 | +0.558 | +0.346 |
| 8006 | conditional | 1 | state_reset/novel | 1 | 6.668 | +5.323 | +0.173 |
| 8006 | conditional | 2 | continue/novel | 0 | 7.154 | +0.099 | +2.228 |
| 8006 | conditional | 2 | fresh/novel | 0 | 7.436 | -7.513 | +2.198 |
| 8006 | conditional | 2 | optimizer_reset/novel | 0 | 7.197 | +1.238 | +2.328 |
| 8006 | conditional | 2 | replay_reset/novel | 0 | 7.284 | +2.516 | +1.281 |
| 8006 | conditional | 2 | state_reset/novel | 0 | 7.234 | +6.982 | +1.870 |
| 8006 | conditional | 3 | continue/novel | 2 | 12.390 | +10.164 | +13.039 |
| 8006 | conditional | 3 | fresh/novel | 2 | 9.188 | -7.201 | +18.870 |
| 8006 | conditional | 3 | optimizer_reset/novel | 2 | 12.191 | +11.040 | +12.858 |
| 8006 | conditional | 3 | replay_reset/novel | 2 | 9.911 | +5.140 | +18.456 |
| 8006 | conditional | 3 | state_reset/novel | 2 | 9.904 | +1.813 | +17.084 |
| 8001 | recurrent | 4 | continue/clean | None | 8.543 | +1.734 | n/a |
| 8001 | recurrent | 4 | optimizer_reset/clean | None | 8.630 | +1.762 | n/a |
| 8001 | recurrent | 4 | replay_reset/clean | None | 8.255 | +1.688 | n/a |
| 8001 | recurrent | 4 | state_reset/clean | None | 8.315 | +2.753 | n/a |
| 8001 | recurrent | 4 | continue/noise | None | 10.937 | -0.387 | n/a |
| 8001 | recurrent | 4 | optimizer_reset/noise | None | 10.340 | +1.547 | n/a |
| 8001 | recurrent | 4 | replay_reset/noise | None | 12.289 | +0.626 | n/a |
| 8001 | recurrent | 4 | state_reset/noise | None | 11.642 | +0.088 | n/a |
| 8001 | recurrent | 4 | continue/return | None | 10.741 | -6.977 | n/a |
| 8001 | recurrent | 4 | optimizer_reset/return | None | 10.022 | -4.569 | n/a |
| 8001 | recurrent | 4 | replay_reset/return | None | 8.483 | -0.691 | n/a |
| 8001 | recurrent | 4 | state_reset/return | None | 8.479 | -1.179 | n/a |
| 8001 | recurrent | 4 | continue/revision | 0 | 9.035 | +0.881 | +0.458 |
| 8001 | recurrent | 4 | optimizer_reset/revision | 0 | 9.232 | +0.968 | +0.274 |
| 8001 | recurrent | 4 | replay_reset/revision | 0 | 8.629 | +0.447 | +1.116 |
| 8001 | recurrent | 4 | state_reset/revision | 0 | 8.805 | +0.487 | +0.498 |
| 8001 | recurrent | 1 | continue/novel | 1 | 7.805 | +4.112 | +0.847 |
| 8001 | recurrent | 1 | fresh/novel | 1 | 7.858 | +0.284 | +0.429 |
| 8001 | recurrent | 1 | optimizer_reset/novel | 1 | 8.214 | +3.145 | +0.260 |
| 8001 | recurrent | 1 | replay_reset/novel | 1 | 7.586 | +10.764 | +1.233 |
| 8001 | recurrent | 1 | state_reset/novel | 1 | 7.670 | +11.700 | +0.329 |
| 8001 | recurrent | 2 | continue/novel | 2 | 11.685 | +7.459 | +13.669 |
| 8001 | recurrent | 2 | fresh/novel | 2 | 8.636 | +1.508 | +18.131 |
| 8001 | recurrent | 2 | optimizer_reset/novel | 2 | 11.165 | +9.385 | +13.508 |
| 8001 | recurrent | 2 | replay_reset/novel | 2 | 9.355 | +8.324 | +16.889 |
| 8001 | recurrent | 2 | state_reset/novel | 2 | 9.555 | +9.347 | +17.532 |
| 8001 | recurrent | 3 | continue/novel | 0 | 9.440 | -1.036 | +1.149 |
| 8001 | recurrent | 3 | fresh/novel | 0 | 9.319 | +1.996 | +1.766 |
| 8001 | recurrent | 3 | optimizer_reset/novel | 0 | 9.481 | -2.065 | +0.653 |
| 8001 | recurrent | 3 | replay_reset/novel | 0 | 8.596 | -2.299 | +1.433 |
| 8001 | recurrent | 3 | state_reset/novel | 0 | 8.766 | +1.032 | +1.151 |
| 8002 | recurrent | 4 | continue/clean | None | 8.643 | +1.305 | n/a |
| 8002 | recurrent | 4 | optimizer_reset/clean | None | 8.538 | +0.649 | n/a |
| 8002 | recurrent | 4 | replay_reset/clean | None | 8.413 | +0.180 | n/a |
| 8002 | recurrent | 4 | state_reset/clean | None | 8.235 | +0.395 | n/a |
| 8002 | recurrent | 4 | continue/noise | None | 10.190 | +0.856 | n/a |
| 8002 | recurrent | 4 | optimizer_reset/noise | None | 10.588 | -0.294 | n/a |
| 8002 | recurrent | 4 | replay_reset/noise | None | 11.463 | +1.571 | n/a |
| 8002 | recurrent | 4 | state_reset/noise | None | 11.629 | +2.312 | n/a |
| 8002 | recurrent | 4 | continue/return | None | 11.524 | -3.928 | n/a |
| 8002 | recurrent | 4 | optimizer_reset/return | None | 12.234 | -4.562 | n/a |
| 8002 | recurrent | 4 | replay_reset/return | None | 8.522 | -2.801 | n/a |
| 8002 | recurrent | 4 | state_reset/return | None | 8.871 | -2.382 | n/a |
| 8002 | recurrent | 4 | continue/revision | 0 | 9.043 | -0.409 | +1.229 |
| 8002 | recurrent | 4 | optimizer_reset/revision | 0 | 9.175 | -1.183 | +0.153 |
| 8002 | recurrent | 4 | replay_reset/revision | 0 | 8.753 | +0.718 | +1.537 |
| 8002 | recurrent | 4 | state_reset/revision | 0 | 8.702 | +0.035 | +1.472 |
| 8002 | recurrent | 1 | continue/novel | 2 | 10.630 | +4.447 | +15.684 |
| 8002 | recurrent | 1 | fresh/novel | 2 | 8.578 | -1.603 | +18.151 |
| 8002 | recurrent | 1 | optimizer_reset/novel | 2 | 10.661 | +5.444 | +17.233 |
| 8002 | recurrent | 1 | replay_reset/novel | 2 | 8.599 | +6.497 | +19.039 |
| 8002 | recurrent | 1 | state_reset/novel | 2 | 8.454 | +5.972 | +16.896 |
| 8002 | recurrent | 2 | continue/novel | 0 | 9.477 | +1.089 | +0.517 |
| 8002 | recurrent | 2 | fresh/novel | 0 | 8.848 | -1.696 | +2.130 |
| 8002 | recurrent | 2 | optimizer_reset/novel | 0 | 9.283 | +0.902 | +1.190 |
| 8002 | recurrent | 2 | replay_reset/novel | 0 | 8.702 | +1.169 | +2.398 |
| 8002 | recurrent | 2 | state_reset/novel | 0 | 8.841 | +0.552 | +1.531 |
| 8002 | recurrent | 3 | continue/novel | 1 | 8.793 | -0.352 | +0.265 |
| 8002 | recurrent | 3 | fresh/novel | 1 | 8.964 | -0.938 | +0.540 |
| 8002 | recurrent | 3 | optimizer_reset/novel | 1 | 8.855 | -0.331 | +0.508 |
| 8002 | recurrent | 3 | replay_reset/novel | 1 | 8.496 | -0.916 | +0.619 |
| 8002 | recurrent | 3 | state_reset/novel | 1 | 8.758 | +0.366 | +0.583 |
| 8003 | recurrent | 4 | continue/clean | None | 8.043 | +1.498 | n/a |
| 8003 | recurrent | 4 | optimizer_reset/clean | None | 8.064 | -2.946 | n/a |
| 8003 | recurrent | 4 | replay_reset/clean | None | 8.139 | +0.888 | n/a |
| 8003 | recurrent | 4 | state_reset/clean | None | 8.116 | +1.845 | n/a |
| 8003 | recurrent | 4 | continue/noise | None | 9.856 | -0.077 | n/a |
| 8003 | recurrent | 4 | optimizer_reset/noise | None | 9.647 | +1.145 | n/a |
| 8003 | recurrent | 4 | replay_reset/noise | None | 11.007 | +1.585 | n/a |
| 8003 | recurrent | 4 | state_reset/noise | None | 10.603 | +1.281 | n/a |
| 8003 | recurrent | 4 | continue/return | None | 9.373 | -6.013 | n/a |
| 8003 | recurrent | 4 | optimizer_reset/return | None | 9.033 | -6.266 | n/a |
| 8003 | recurrent | 4 | replay_reset/return | None | 8.175 | -1.003 | n/a |
| 8003 | recurrent | 4 | state_reset/return | None | 8.151 | -1.038 | n/a |
| 8003 | recurrent | 4 | continue/revision | 0 | 8.573 | +5.331 | -0.417 |
| 8003 | recurrent | 4 | optimizer_reset/revision | 0 | 8.660 | +4.066 | +0.188 |
| 8003 | recurrent | 4 | replay_reset/revision | 0 | 8.544 | +4.941 | +0.528 |
| 8003 | recurrent | 4 | state_reset/revision | 0 | 8.572 | +4.199 | +1.378 |
| 8003 | recurrent | 1 | continue/novel | 2 | 9.445 | +3.759 | +17.672 |
| 8003 | recurrent | 1 | fresh/novel | 2 | 8.259 | -1.578 | +18.495 |
| 8003 | recurrent | 1 | optimizer_reset/novel | 2 | 9.480 | +4.276 | +17.246 |
| 8003 | recurrent | 1 | replay_reset/novel | 2 | 8.186 | +3.453 | +18.673 |
| 8003 | recurrent | 1 | state_reset/novel | 2 | 8.316 | +4.293 | +19.214 |
| 8003 | recurrent | 2 | continue/novel | 1 | 7.687 | -4.226 | -0.262 |
| 8003 | recurrent | 2 | fresh/novel | 1 | 8.720 | +0.622 | +0.259 |
| 8003 | recurrent | 2 | optimizer_reset/novel | 1 | 7.611 | -0.297 | -0.029 |
| 8003 | recurrent | 2 | replay_reset/novel | 1 | 7.671 | +2.677 | -0.265 |
| 8003 | recurrent | 2 | state_reset/novel | 1 | 7.571 | +2.550 | -0.346 |
| 8003 | recurrent | 3 | continue/novel | 0 | 8.771 | +3.924 | +0.791 |
| 8003 | recurrent | 3 | fresh/novel | 0 | 9.183 | -0.838 | +1.757 |
| 8003 | recurrent | 3 | optimizer_reset/novel | 0 | 8.685 | +5.105 | +0.717 |
| 8003 | recurrent | 3 | replay_reset/novel | 0 | 8.292 | +3.695 | +1.524 |
| 8003 | recurrent | 3 | state_reset/novel | 0 | 8.378 | +5.066 | +1.115 |
| 8004 | recurrent | 4 | continue/clean | None | 9.444 | -0.212 | n/a |
| 8004 | recurrent | 4 | optimizer_reset/clean | None | 9.499 | +0.934 | n/a |
| 8004 | recurrent | 4 | replay_reset/clean | None | 8.592 | +1.414 | n/a |
| 8004 | recurrent | 4 | state_reset/clean | None | 8.610 | -0.118 | n/a |
| 8004 | recurrent | 4 | continue/noise | None | 11.569 | -0.710 | n/a |
| 8004 | recurrent | 4 | optimizer_reset/noise | None | 11.332 | -0.134 | n/a |
| 8004 | recurrent | 4 | replay_reset/noise | None | 11.556 | -0.052 | n/a |
| 8004 | recurrent | 4 | state_reset/noise | None | 11.580 | -1.102 | n/a |
| 8004 | recurrent | 4 | continue/return | None | 10.440 | -11.099 | n/a |
| 8004 | recurrent | 4 | optimizer_reset/return | None | 10.483 | -10.031 | n/a |
| 8004 | recurrent | 4 | replay_reset/return | None | 8.579 | +0.436 | n/a |
| 8004 | recurrent | 4 | state_reset/return | None | 8.849 | +0.849 | n/a |
| 8004 | recurrent | 4 | continue/revision | 0 | 10.828 | +0.586 | +0.651 |
| 8004 | recurrent | 4 | optimizer_reset/revision | 0 | 10.807 | -0.598 | +0.378 |
| 8004 | recurrent | 4 | replay_reset/revision | 0 | 9.741 | +0.610 | +1.271 |
| 8004 | recurrent | 4 | state_reset/revision | 0 | 9.620 | +0.330 | +0.828 |
| 8004 | recurrent | 1 | continue/novel | 0 | 8.189 | +4.735 | +2.536 |
| 8004 | recurrent | 1 | fresh/novel | 0 | 7.656 | -0.795 | +3.328 |
| 8004 | recurrent | 1 | optimizer_reset/novel | 0 | 8.357 | +3.277 | +1.897 |
| 8004 | recurrent | 1 | replay_reset/novel | 0 | 7.640 | +3.590 | +3.265 |
| 8004 | recurrent | 1 | state_reset/novel | 0 | 7.534 | +4.683 | +4.053 |
| 8004 | recurrent | 2 | continue/novel | 1 | 8.225 | -0.151 | +0.874 |
| 8004 | recurrent | 2 | fresh/novel | 1 | 8.387 | -0.396 | +0.506 |
| 8004 | recurrent | 2 | optimizer_reset/novel | 1 | 8.456 | -0.402 | +0.836 |
| 8004 | recurrent | 2 | replay_reset/novel | 1 | 8.201 | +0.557 | +1.050 |
| 8004 | recurrent | 2 | state_reset/novel | 1 | 8.388 | +0.870 | +0.799 |
| 8004 | recurrent | 3 | continue/novel | 2 | 11.888 | -0.513 | +12.535 |
| 8004 | recurrent | 3 | fresh/novel | 2 | 9.088 | -0.679 | +18.461 |
| 8004 | recurrent | 3 | optimizer_reset/novel | 2 | 11.686 | -0.090 | +12.040 |
| 8004 | recurrent | 3 | replay_reset/novel | 2 | 10.220 | +0.993 | +15.832 |
| 8004 | recurrent | 3 | state_reset/novel | 2 | 10.114 | -0.231 | +17.996 |
| 8005 | recurrent | 4 | continue/clean | None | 8.194 | -0.420 | n/a |
| 8005 | recurrent | 4 | optimizer_reset/clean | None | 8.415 | -0.437 | n/a |
| 8005 | recurrent | 4 | replay_reset/clean | None | 8.248 | +0.099 | n/a |
| 8005 | recurrent | 4 | state_reset/clean | None | 8.193 | +0.088 | n/a |
| 8005 | recurrent | 4 | continue/noise | None | 10.426 | -1.092 | n/a |
| 8005 | recurrent | 4 | optimizer_reset/noise | None | 9.988 | -1.208 | n/a |
| 8005 | recurrent | 4 | replay_reset/noise | None | 11.620 | -0.557 | n/a |
| 8005 | recurrent | 4 | state_reset/noise | None | 11.388 | +0.014 | n/a |
| 8005 | recurrent | 4 | continue/return | None | 9.649 | -11.046 | n/a |
| 8005 | recurrent | 4 | optimizer_reset/return | None | 11.746 | -13.550 | n/a |
| 8005 | recurrent | 4 | replay_reset/return | None | 8.291 | +0.469 | n/a |
| 8005 | recurrent | 4 | state_reset/return | None | 8.243 | -0.963 | n/a |
| 8005 | recurrent | 4 | continue/revision | 0 | 9.415 | +1.828 | +0.186 |
| 8005 | recurrent | 4 | optimizer_reset/revision | 0 | 9.510 | +0.207 | +0.295 |
| 8005 | recurrent | 4 | replay_reset/revision | 0 | 9.247 | +0.927 | +0.478 |
| 8005 | recurrent | 4 | state_reset/revision | 0 | 9.010 | +0.870 | +0.879 |
| 8005 | recurrent | 1 | continue/novel | 0 | 7.387 | +6.916 | +1.595 |
| 8005 | recurrent | 1 | fresh/novel | 0 | 7.061 | -1.914 | +2.882 |
| 8005 | recurrent | 1 | optimizer_reset/novel | 0 | 7.214 | +11.374 | +1.904 |
| 8005 | recurrent | 1 | replay_reset/novel | 0 | 6.848 | +12.732 | +2.633 |
| 8005 | recurrent | 1 | state_reset/novel | 0 | 6.765 | +12.983 | +2.613 |
| 8005 | recurrent | 2 | continue/novel | 2 | 11.630 | +3.400 | +13.023 |
| 8005 | recurrent | 2 | fresh/novel | 2 | 9.264 | -1.261 | +18.187 |
| 8005 | recurrent | 2 | optimizer_reset/novel | 2 | 11.608 | +0.555 | +11.476 |
| 8005 | recurrent | 2 | replay_reset/novel | 2 | 9.167 | +7.041 | +15.875 |
| 8005 | recurrent | 2 | state_reset/novel | 2 | 9.501 | +6.739 | +16.381 |
| 8005 | recurrent | 3 | continue/novel | 1 | 8.882 | +2.837 | +0.568 |
| 8005 | recurrent | 3 | fresh/novel | 1 | 8.743 | -0.921 | +0.565 |
| 8005 | recurrent | 3 | optimizer_reset/novel | 1 | 9.128 | +3.124 | +0.736 |
| 8005 | recurrent | 3 | replay_reset/novel | 1 | 8.104 | +3.425 | +0.333 |
| 8005 | recurrent | 3 | state_reset/novel | 1 | 8.242 | +2.263 | +0.140 |
| 8006 | recurrent | 4 | continue/clean | None | 9.899 | -3.020 | n/a |
| 8006 | recurrent | 4 | optimizer_reset/clean | None | 9.727 | -0.722 | n/a |
| 8006 | recurrent | 4 | replay_reset/clean | None | 8.992 | -2.384 | n/a |
| 8006 | recurrent | 4 | state_reset/clean | None | 9.019 | -2.267 | n/a |
| 8006 | recurrent | 4 | continue/noise | None | 11.737 | -1.762 | n/a |
| 8006 | recurrent | 4 | optimizer_reset/noise | None | 11.548 | -1.337 | n/a |
| 8006 | recurrent | 4 | replay_reset/noise | None | 12.443 | -0.648 | n/a |
| 8006 | recurrent | 4 | state_reset/noise | None | 11.816 | -2.247 | n/a |
| 8006 | recurrent | 4 | continue/return | None | 9.769 | -14.882 | n/a |
| 8006 | recurrent | 4 | optimizer_reset/return | None | 10.238 | -16.560 | n/a |
| 8006 | recurrent | 4 | replay_reset/return | None | 8.839 | -1.043 | n/a |
| 8006 | recurrent | 4 | state_reset/return | None | 8.963 | -0.580 | n/a |
| 8006 | recurrent | 4 | continue/revision | 0 | 9.987 | -1.821 | +0.042 |
| 8006 | recurrent | 4 | optimizer_reset/revision | 0 | 10.035 | -1.793 | +0.806 |
| 8006 | recurrent | 4 | replay_reset/revision | 0 | 8.810 | -0.447 | +1.082 |
| 8006 | recurrent | 4 | state_reset/revision | 0 | 8.955 | -1.456 | +0.560 |
| 8006 | recurrent | 1 | continue/novel | 1 | 7.749 | +4.315 | +0.499 |
| 8006 | recurrent | 1 | fresh/novel | 1 | 7.438 | +0.811 | +0.710 |
| 8006 | recurrent | 1 | optimizer_reset/novel | 1 | 7.725 | +10.631 | +0.322 |
| 8006 | recurrent | 1 | replay_reset/novel | 1 | 6.750 | +12.786 | +0.423 |
| 8006 | recurrent | 1 | state_reset/novel | 1 | 6.756 | +12.608 | +0.648 |
| 8006 | recurrent | 2 | continue/novel | 0 | 7.429 | +8.614 | +1.367 |
| 8006 | recurrent | 2 | fresh/novel | 0 | 7.505 | -0.045 | +2.228 |
| 8006 | recurrent | 2 | optimizer_reset/novel | 0 | 7.451 | +4.997 | +1.899 |
| 8006 | recurrent | 2 | replay_reset/novel | 0 | 7.523 | +7.442 | +1.515 |
| 8006 | recurrent | 2 | state_reset/novel | 0 | 7.576 | +6.622 | +1.387 |
| 8006 | recurrent | 3 | continue/novel | 2 | 12.377 | +1.023 | +14.128 |
| 8006 | recurrent | 3 | fresh/novel | 2 | 9.039 | +3.082 | +18.939 |
| 8006 | recurrent | 3 | optimizer_reset/novel | 2 | 12.388 | +0.086 | +13.045 |
| 8006 | recurrent | 3 | replay_reset/novel | 2 | 10.225 | +0.645 | +18.028 |
| 8006 | recurrent | 3 | state_reset/novel | 2 | 10.106 | +1.422 | +18.357 |
