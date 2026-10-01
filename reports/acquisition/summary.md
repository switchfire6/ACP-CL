# Acquisition diagnostic results

Kind: diagnostic; 6 independent seeds; 180 episodes.
Fresh-model qualification: **False**.

## Qualification

| Group | Marginal Brier improvement | Correct-cue Brier benefit | Pass |
|---|---:|---:|---|
| conditional_stage_1 | 0.15009 | 0.05470 | True |
| conditional_stage_2 | 0.13976 | 0.04915 | True |
| conditional_stage_3 | 0.13383 | 0.04853 | True |
| conditional_cue_0 | 0.14213 | 0.00674 | True |
| conditional_cue_1 | 0.14859 | 0.00194 | False |
| conditional_cue_2 | 0.13297 | 0.14369 | True |
| recurrent_stage_1 | 0.14953 | 0.05718 | True |
| recurrent_stage_2 | 0.13996 | 0.05121 | True |
| recurrent_stage_3 | 0.13585 | 0.05213 | True |
| recurrent_cue_0 | 0.14301 | 0.00904 | True |
| recurrent_cue_1 | 0.14854 | 0.00259 | True |
| recurrent_cue_2 | 0.13380 | 0.14890 | True |

## Every episode

Brier errors below are multiplied by 100; lower is better. Positive valid damage is forgetting.

| Seed | Model | Stage | Arm/branch | Cue | Brier AUC | Valid damage | Cue benefit |
|---|---|---:|---|---|---:|---:|---:|
| 7001 | conditional | 4 | continue/noise | None | 10.026 | +0.824 | n/a |
| 7001 | conditional | 4 | continue/return | None | 10.813 | -3.180 | n/a |
| 7001 | conditional | 4 | continue/revision | 0 | 9.277 | +0.428 | -0.064 |
| 7001 | conditional | 1 | continue/novel | 2 | 12.572 | +1.405 | +10.447 |
| 7001 | conditional | 1 | fresh/novel | 2 | 12.047 | -6.131 | +16.977 |
| 7001 | conditional | 1 | frozen/novel | 2 | 13.821 | +0.618 | +0.324 |
| 7001 | conditional | 1 | state_reset/novel | 2 | 9.944 | +4.029 | +19.803 |
| 7001 | conditional | 2 | continue/novel | 1 | 10.546 | +0.174 | +0.865 |
| 7001 | conditional | 2 | fresh/novel | 1 | 11.872 | -8.592 | +0.080 |
| 7001 | conditional | 2 | frozen/novel | 1 | 10.365 | -0.401 | +0.498 |
| 7001 | conditional | 2 | state_reset/novel | 1 | 9.695 | +1.755 | +0.157 |
| 7001 | conditional | 3 | continue/novel | 0 | 9.175 | +0.868 | +0.650 |
| 7001 | conditional | 3 | fresh/novel | 0 | 12.157 | -6.679 | +0.022 |
| 7001 | conditional | 3 | frozen/novel | 0 | 8.826 | -0.382 | +0.175 |
| 7001 | conditional | 3 | state_reset/novel | 0 | 8.809 | +0.586 | +0.850 |
| 7002 | conditional | 4 | continue/noise | None | 11.982 | +0.333 | n/a |
| 7002 | conditional | 4 | continue/return | None | 10.137 | -2.167 | n/a |
| 7002 | conditional | 4 | continue/revision | 0 | 12.348 | +2.793 | +0.005 |
| 7002 | conditional | 1 | continue/novel | 0 | 8.163 | +0.509 | +0.793 |
| 7002 | conditional | 1 | fresh/novel | 0 | 10.237 | -7.607 | +0.257 |
| 7002 | conditional | 1 | frozen/novel | 0 | 7.960 | -0.106 | -0.096 |
| 7002 | conditional | 1 | state_reset/novel | 0 | 8.324 | +0.737 | +0.843 |
| 7002 | conditional | 2 | continue/novel | 1 | 7.998 | +0.550 | +0.680 |
| 7002 | conditional | 2 | fresh/novel | 1 | 10.784 | -5.338 | +0.668 |
| 7002 | conditional | 2 | frozen/novel | 1 | 7.388 | +0.475 | -0.058 |
| 7002 | conditional | 2 | state_reset/novel | 1 | 8.270 | +3.220 | +0.499 |
| 7002 | conditional | 3 | continue/novel | 2 | 14.354 | +0.994 | +6.005 |
| 7002 | conditional | 3 | fresh/novel | 2 | 13.554 | -4.469 | +14.399 |
| 7002 | conditional | 3 | frozen/novel | 2 | 14.742 | +0.873 | +0.382 |
| 7002 | conditional | 3 | state_reset/novel | 2 | 12.884 | +11.498 | +10.575 |
| 7003 | conditional | 4 | continue/noise | None | 12.442 | +1.806 | n/a |
| 7003 | conditional | 4 | continue/return | None | 10.018 | -1.492 | n/a |
| 7003 | conditional | 4 | continue/revision | 0 | 12.706 | -0.453 | +0.536 |
| 7003 | conditional | 1 | continue/novel | 0 | 9.024 | +0.097 | +1.510 |
| 7003 | conditional | 1 | fresh/novel | 0 | 10.612 | -5.874 | +1.507 |
| 7003 | conditional | 1 | frozen/novel | 0 | 8.122 | -0.920 | -0.206 |
| 7003 | conditional | 1 | state_reset/novel | 0 | 8.350 | +2.475 | +2.233 |
| 7003 | conditional | 2 | continue/novel | 2 | 14.869 | +0.470 | +3.506 |
| 7003 | conditional | 2 | fresh/novel | 2 | 13.049 | -6.425 | +13.002 |
| 7003 | conditional | 2 | frozen/novel | 2 | 14.957 | -0.256 | +0.100 |
| 7003 | conditional | 2 | state_reset/novel | 2 | 12.080 | +2.936 | +13.125 |
| 7003 | conditional | 3 | continue/novel | 1 | 12.739 | +0.946 | -0.338 |
| 7003 | conditional | 3 | fresh/novel | 1 | 12.818 | -6.614 | +0.239 |
| 7003 | conditional | 3 | frozen/novel | 1 | 12.936 | +0.292 | +0.071 |
| 7003 | conditional | 3 | state_reset/novel | 1 | 10.628 | +0.816 | +0.261 |
| 7004 | conditional | 4 | continue/noise | None | 11.513 | +0.279 | n/a |
| 7004 | conditional | 4 | continue/return | None | 10.977 | -1.616 | n/a |
| 7004 | conditional | 4 | continue/revision | 0 | 11.264 | -0.000 | -0.167 |
| 7004 | conditional | 1 | continue/novel | 1 | 6.407 | +0.821 | +0.353 |
| 7004 | conditional | 1 | fresh/novel | 1 | 9.833 | -7.033 | +0.070 |
| 7004 | conditional | 1 | frozen/novel | 1 | 6.051 | +0.840 | +0.105 |
| 7004 | conditional | 1 | state_reset/novel | 1 | 6.986 | +0.953 | +0.423 |
| 7004 | conditional | 2 | continue/novel | 0 | 8.414 | +2.469 | +0.599 |
| 7004 | conditional | 2 | fresh/novel | 0 | 9.645 | -4.903 | +1.037 |
| 7004 | conditional | 2 | frozen/novel | 0 | 7.490 | +0.936 | +0.920 |
| 7004 | conditional | 2 | state_reset/novel | 0 | 7.910 | +2.364 | +1.902 |
| 7004 | conditional | 3 | continue/novel | 2 | 13.181 | -0.083 | +5.733 |
| 7004 | conditional | 3 | fresh/novel | 2 | 12.028 | -2.999 | +13.965 |
| 7004 | conditional | 3 | frozen/novel | 2 | 13.878 | -0.879 | +0.036 |
| 7004 | conditional | 3 | state_reset/novel | 2 | 11.135 | +0.698 | +14.048 |
| 7005 | conditional | 4 | continue/noise | None | 11.036 | +1.781 | n/a |
| 7005 | conditional | 4 | continue/return | None | 10.287 | -0.412 | n/a |
| 7005 | conditional | 4 | continue/revision | 0 | 10.345 | -0.823 | +0.445 |
| 7005 | conditional | 1 | continue/novel | 1 | 8.312 | -0.041 | -0.145 |
| 7005 | conditional | 1 | fresh/novel | 1 | 9.712 | -6.078 | +0.111 |
| 7005 | conditional | 1 | frozen/novel | 1 | 7.460 | -1.111 | +0.107 |
| 7005 | conditional | 1 | state_reset/novel | 1 | 7.053 | +0.864 | +0.290 |
| 7005 | conditional | 2 | continue/novel | 2 | 13.886 | +1.034 | +8.696 |
| 7005 | conditional | 2 | fresh/novel | 2 | 12.757 | -4.138 | +13.975 |
| 7005 | conditional | 2 | frozen/novel | 2 | 14.810 | +0.334 | +0.677 |
| 7005 | conditional | 2 | state_reset/novel | 2 | 11.560 | +9.129 | +13.800 |
| 7005 | conditional | 3 | continue/novel | 0 | 10.853 | +0.037 | +1.139 |
| 7005 | conditional | 3 | fresh/novel | 0 | 12.719 | -3.281 | +0.494 |
| 7005 | conditional | 3 | frozen/novel | 0 | 10.904 | -0.093 | +0.421 |
| 7005 | conditional | 3 | state_reset/novel | 0 | 9.886 | +0.239 | +1.879 |
| 7006 | conditional | 4 | continue/noise | None | 11.263 | +0.677 | n/a |
| 7006 | conditional | 4 | continue/return | None | 10.531 | -4.379 | n/a |
| 7006 | conditional | 4 | continue/revision | 0 | 10.079 | -0.770 | +0.447 |
| 7006 | conditional | 1 | continue/novel | 2 | 12.991 | +0.632 | +3.805 |
| 7006 | conditional | 1 | fresh/novel | 2 | 11.010 | -6.187 | +13.896 |
| 7006 | conditional | 1 | frozen/novel | 2 | 13.262 | -0.136 | +0.840 |
| 7006 | conditional | 1 | state_reset/novel | 2 | 11.234 | +4.454 | +13.350 |
| 7006 | conditional | 2 | continue/novel | 0 | 10.541 | +1.082 | +1.000 |
| 7006 | conditional | 2 | fresh/novel | 0 | 11.010 | -0.215 | +0.728 |
| 7006 | conditional | 2 | frozen/novel | 0 | 10.178 | +0.141 | +0.107 |
| 7006 | conditional | 2 | state_reset/novel | 0 | 9.244 | +2.067 | +0.712 |
| 7006 | conditional | 3 | continue/novel | 1 | 10.097 | -0.363 | -0.067 |
| 7006 | conditional | 3 | fresh/novel | 1 | 11.510 | -3.776 | -0.004 |
| 7006 | conditional | 3 | frozen/novel | 1 | 9.193 | +0.685 | +0.646 |
| 7006 | conditional | 3 | state_reset/novel | 1 | 8.998 | +0.865 | +0.361 |
| 7001 | recurrent | 4 | continue/noise | None | 10.478 | -1.212 | n/a |
| 7001 | recurrent | 4 | continue/return | None | 10.791 | -0.660 | n/a |
| 7001 | recurrent | 4 | continue/revision | 0 | 9.138 | +1.161 | +0.330 |
| 7001 | recurrent | 1 | continue/novel | 2 | 15.470 | +7.535 | +7.020 |
| 7001 | recurrent | 1 | fresh/novel | 2 | 11.540 | -1.881 | +17.592 |
| 7001 | recurrent | 1 | frozen/novel | 2 | 14.392 | +7.592 | +0.411 |
| 7001 | recurrent | 1 | state_reset/novel | 2 | 9.906 | +9.572 | +18.543 |
| 7001 | recurrent | 2 | continue/novel | 1 | 11.398 | +0.684 | +0.472 |
| 7001 | recurrent | 2 | fresh/novel | 1 | 12.421 | -4.089 | +0.133 |
| 7001 | recurrent | 2 | frozen/novel | 1 | 11.697 | +1.461 | +0.509 |
| 7001 | recurrent | 2 | state_reset/novel | 1 | 10.688 | +2.181 | +0.784 |
| 7001 | recurrent | 3 | continue/novel | 0 | 9.409 | -0.536 | +0.306 |
| 7001 | recurrent | 3 | fresh/novel | 0 | 12.089 | -4.800 | +0.440 |
| 7001 | recurrent | 3 | frozen/novel | 0 | 9.336 | +0.862 | +0.173 |
| 7001 | recurrent | 3 | state_reset/novel | 0 | 9.237 | +1.311 | +0.574 |
| 7002 | recurrent | 4 | continue/noise | None | 12.484 | -1.449 | n/a |
| 7002 | recurrent | 4 | continue/return | None | 11.090 | -9.321 | n/a |
| 7002 | recurrent | 4 | continue/revision | 0 | 11.766 | -0.461 | -0.283 |
| 7002 | recurrent | 1 | continue/novel | 0 | 10.812 | +1.184 | +1.029 |
| 7002 | recurrent | 1 | fresh/novel | 0 | 10.683 | -3.059 | +0.480 |
| 7002 | recurrent | 1 | frozen/novel | 0 | 10.879 | +1.073 | -0.171 |
| 7002 | recurrent | 1 | state_reset/novel | 0 | 9.042 | +6.547 | +1.005 |
| 7002 | recurrent | 2 | continue/novel | 1 | 8.750 | +3.041 | +0.870 |
| 7002 | recurrent | 2 | fresh/novel | 1 | 11.420 | -2.340 | +0.592 |
| 7002 | recurrent | 2 | frozen/novel | 1 | 8.494 | +1.895 | +0.112 |
| 7002 | recurrent | 2 | state_reset/novel | 1 | 8.527 | +7.136 | +1.194 |
| 7002 | recurrent | 3 | continue/novel | 2 | 13.882 | -1.244 | +8.936 |
| 7002 | recurrent | 3 | fresh/novel | 2 | 13.040 | -2.617 | +15.324 |
| 7002 | recurrent | 3 | frozen/novel | 2 | 14.632 | +0.547 | +3.130 |
| 7002 | recurrent | 3 | state_reset/novel | 2 | 11.652 | +2.239 | +14.263 |
| 7003 | recurrent | 4 | continue/noise | None | 12.032 | +4.303 | n/a |
| 7003 | recurrent | 4 | continue/return | None | 9.452 | -8.795 | n/a |
| 7003 | recurrent | 4 | continue/revision | 0 | 12.098 | -1.560 | +0.178 |
| 7003 | recurrent | 1 | continue/novel | 0 | 9.197 | +0.219 | +1.319 |
| 7003 | recurrent | 1 | fresh/novel | 0 | 9.987 | -1.740 | +2.113 |
| 7003 | recurrent | 1 | frozen/novel | 0 | 8.450 | -0.023 | +0.143 |
| 7003 | recurrent | 1 | state_reset/novel | 0 | 8.247 | +10.506 | +3.383 |
| 7003 | recurrent | 2 | continue/novel | 2 | 14.354 | +1.271 | +3.376 |
| 7003 | recurrent | 2 | fresh/novel | 2 | 13.425 | -0.880 | +14.156 |
| 7003 | recurrent | 2 | frozen/novel | 2 | 14.912 | +7.866 | +0.561 |
| 7003 | recurrent | 2 | state_reset/novel | 2 | 12.589 | +12.848 | +12.544 |
| 7003 | recurrent | 3 | continue/novel | 1 | 12.006 | +3.937 | +0.196 |
| 7003 | recurrent | 3 | fresh/novel | 1 | 12.443 | -2.046 | +0.246 |
| 7003 | recurrent | 3 | frozen/novel | 1 | 12.183 | +0.590 | -0.119 |
| 7003 | recurrent | 3 | state_reset/novel | 1 | 10.730 | +9.198 | +0.305 |
| 7004 | recurrent | 4 | continue/noise | None | 12.368 | -6.635 | n/a |
| 7004 | recurrent | 4 | continue/return | None | 10.671 | -13.617 | n/a |
| 7004 | recurrent | 4 | continue/revision | 0 | 11.311 | +0.416 | -0.338 |
| 7004 | recurrent | 1 | continue/novel | 1 | 6.572 | -0.772 | +0.392 |
| 7004 | recurrent | 1 | fresh/novel | 1 | 9.743 | +0.972 | +0.372 |
| 7004 | recurrent | 1 | frozen/novel | 1 | 5.999 | -0.118 | +0.091 |
| 7004 | recurrent | 1 | state_reset/novel | 1 | 7.231 | +16.064 | +0.281 |
| 7004 | recurrent | 2 | continue/novel | 0 | 8.284 | +9.948 | +0.364 |
| 7004 | recurrent | 2 | fresh/novel | 0 | 10.162 | -0.063 | +1.206 |
| 7004 | recurrent | 2 | frozen/novel | 0 | 8.004 | +13.547 | +0.052 |
| 7004 | recurrent | 2 | state_reset/novel | 0 | 8.275 | +17.348 | +1.660 |
| 7004 | recurrent | 3 | continue/novel | 2 | 13.893 | +3.048 | +5.293 |
| 7004 | recurrent | 3 | fresh/novel | 2 | 11.297 | -0.889 | +14.944 |
| 7004 | recurrent | 3 | frozen/novel | 2 | 14.106 | +0.880 | +1.184 |
| 7004 | recurrent | 3 | state_reset/novel | 2 | 10.907 | +4.917 | +12.551 |
| 7005 | recurrent | 4 | continue/noise | None | 11.353 | +4.206 | n/a |
| 7005 | recurrent | 4 | continue/return | None | 10.051 | -8.147 | n/a |
| 7005 | recurrent | 4 | continue/revision | 0 | 10.354 | +1.204 | +0.457 |
| 7005 | recurrent | 1 | continue/novel | 1 | 16.865 | +1.431 | -0.092 |
| 7005 | recurrent | 1 | fresh/novel | 1 | 9.085 | -0.022 | +0.290 |
| 7005 | recurrent | 1 | frozen/novel | 1 | 16.662 | +0.736 | -0.230 |
| 7005 | recurrent | 1 | state_reset/novel | 1 | 8.520 | +8.275 | +0.745 |
| 7005 | recurrent | 2 | continue/novel | 2 | 14.176 | +1.619 | +6.823 |
| 7005 | recurrent | 2 | fresh/novel | 2 | 11.942 | -1.661 | +13.862 |
| 7005 | recurrent | 2 | frozen/novel | 2 | 15.140 | +0.083 | +2.956 |
| 7005 | recurrent | 2 | state_reset/novel | 2 | 11.447 | +5.291 | +15.754 |
| 7005 | recurrent | 3 | continue/novel | 0 | 11.730 | -3.225 | +0.118 |
| 7005 | recurrent | 3 | fresh/novel | 0 | 12.379 | +0.931 | +0.408 |
| 7005 | recurrent | 3 | frozen/novel | 0 | 11.808 | +1.370 | +0.057 |
| 7005 | recurrent | 3 | state_reset/novel | 0 | 10.671 | +6.159 | +1.441 |
| 7006 | recurrent | 4 | continue/noise | None | 11.637 | -4.631 | n/a |
| 7006 | recurrent | 4 | continue/return | None | 10.362 | -2.792 | n/a |
| 7006 | recurrent | 4 | continue/revision | 0 | 9.965 | +2.942 | +0.467 |
| 7006 | recurrent | 1 | continue/novel | 2 | 12.563 | +3.348 | +4.940 |
| 7006 | recurrent | 1 | fresh/novel | 2 | 11.244 | +0.327 | +13.460 |
| 7006 | recurrent | 1 | frozen/novel | 2 | 13.171 | +2.642 | +0.809 |
| 7006 | recurrent | 1 | state_reset/novel | 2 | 10.827 | +11.286 | +12.121 |
| 7006 | recurrent | 2 | continue/novel | 0 | 11.458 | -0.059 | +0.859 |
| 7006 | recurrent | 2 | fresh/novel | 0 | 10.914 | -0.983 | +0.779 |
| 7006 | recurrent | 2 | frozen/novel | 0 | 11.672 | +2.246 | -0.022 |
| 7006 | recurrent | 2 | state_reset/novel | 0 | 10.132 | +8.795 | +1.627 |
| 7006 | recurrent | 3 | continue/novel | 1 | 10.190 | +0.687 | -0.192 |
| 7006 | recurrent | 3 | fresh/novel | 1 | 11.431 | -1.256 | -0.081 |
| 7006 | recurrent | 3 | frozen/novel | 1 | 9.555 | +4.651 | +0.110 |
| 7006 | recurrent | 3 | state_reset/novel | 1 | 9.151 | +5.813 | +0.186 |

## Paired acquisition contrasts

Episodes averaged within seed before bootstrap; lower favors the first arm.

| Model | A minus B | Brier AUC difference x100 | 95% descriptive interval |
|---|---|---:|---|
| conditional | continue_minus_fresh | -0.735 | [-1.185, -0.282] |
| conditional | continue_minus_frozen | +0.099 | [-0.066, +0.235] |
| conditional | state_reset_minus_continue | -1.174 | [-1.570, -0.753] |
| conditional | state_reset_minus_fresh | -1.909 | [-2.227, -1.622] |
| recurrent | continue_minus_fresh | +0.320 | [-0.465, +1.492] |
| recurrent | continue_minus_frozen | -0.004 | [-0.161, +0.156] |
| recurrent | state_reset_minus_continue | -1.846 | [-2.745, -1.166] |
| recurrent | state_reset_minus_fresh | -1.526 | [-1.853, -1.198] |

## Final branches

Before/after probes use the same clean, fresh support protocol. Brier refers to the affected subset for revision and all queries for return/noise. All values below are multiplied by 100. These are secondary descriptive outcomes.

| Model | Branch | Brier before | Brier after | Brier change | Valid-old damage | Survival before | Survival after |
|---|---|---:|---:|---:|---:|---:|---:|
| conditional | return | 12.657 | 7.846 | -4.811 | -2.208 | 69.124 | 73.910 |
| conditional | revision | 11.518 | 10.563 | -0.955 | +0.196 | 57.357 | 60.384 |
| conditional | noise | 12.640 | 11.000 | -1.640 | +0.950 | 56.266 | 58.887 |
| recurrent | return | 29.788 | 7.697 | -22.091 | -7.222 | 58.529 | 73.600 |
| recurrent | revision | 12.141 | 10.960 | -1.181 | +0.617 | 57.520 | 60.384 |
| recurrent | noise | 11.237 | 11.620 | +0.383 | -0.903 | 58.496 | 58.415 |
