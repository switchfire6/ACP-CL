# Replay renewal: decomposition

84 complete episodes; paired seeds [9001, 9002, 9003, 9004, 9005, 9006].

Fresh qualification: **PASS**. Thresholds: marginal gain .02, cue benefit .002.

Error values below are Brier x100. Negative contrast is improvement; intervals are paired seed bootstrap, descriptive.

| Model | Contrast | Acquisition AUC [95% interval] | Terminal valid-old difference |
|---|---|---:|---:|
| conditional | clear_minus_keep | -0.483 [-0.786, -0.181] | +0.990 |
| conditional | clear_rebase_minus_rebase | -0.331 [-0.440, -0.219] | +1.029 |
| conditional | rebase_minus_keep | -0.275 [-0.577, +0.039] | +0.765 |
| conditional | clear_rebase_minus_clear | -0.123 [-0.220, -0.014] | +0.805 |
| conditional | clear_main | -0.407 [-0.575, -0.242] | +1.009 |
| conditional | rebase_main | -0.199 [-0.387, -0.029] | +0.785 |
| conditional | interaction | +0.152 [-0.169, +0.421] | +0.040 |
| conditional | rng_reset_minus_keep | +0.039 [-0.130, +0.236] | +0.350 |
| conditional | full_reset_minus_clear_rebase | +0.184 [+0.090, +0.265] | -0.719 |
| conditional | full_reset_minus_keep | -0.422 [-0.701, -0.157] | +1.075 |
| conditional | keep_minus_fresh | +0.156 [-0.294, +0.575] | -10.874 |
| recurrent | clear_minus_keep | -0.708 [-1.146, -0.280] | +2.941 |
| recurrent | clear_rebase_minus_rebase | -0.276 [-0.494, -0.087] | +0.795 |
| recurrent | rebase_minus_keep | -0.451 [-0.818, -0.094] | +2.796 |
| recurrent | clear_rebase_minus_clear | -0.019 [-0.141, +0.095] | +0.650 |
| recurrent | clear_main | -0.492 [-0.754, -0.210] | +1.868 |
| recurrent | rebase_main | -0.235 [-0.426, -0.045] | +1.723 |
| recurrent | interaction | +0.432 [+0.073, +0.813] | -2.146 |
| recurrent | rng_reset_minus_keep | +0.067 [-0.088, +0.213] | -0.278 |
| recurrent | full_reset_minus_clear_rebase | +0.178 [+0.073, +0.297] | -0.292 |
| recurrent | full_reset_minus_keep | -0.549 [-0.953, -0.147] | +3.299 |
| recurrent | keep_minus_fresh | +0.374 [-0.199, +0.965] | -3.847 |

## Qualification groups

| Group | Episodes | Marginal gain | Cue benefit | Pass |
|---|---:|---:|---:|---|
| conditional_stage_1 | 6 | 14.881 | 7.629 | True |
| conditional_cue_0 | 2 | 16.257 | 3.315 | True |
| conditional_cue_1 | 2 | 13.708 | 0.533 | True |
| conditional_cue_2 | 2 | 14.679 | 19.037 | True |
| recurrent_stage_1 | 6 | 15.510 | 7.652 | True |
| recurrent_cue_0 | 2 | 16.003 | 2.776 | True |
| recurrent_cue_1 | 2 | 15.766 | 0.988 | True |
| recurrent_cue_2 | 2 | 14.761 | 19.191 | True |

## Every episode

| Seed | Model | Stage | Arm / branch | AUC | Valid-old error | Damage | Cue benefit |
|---|---|---:|---|---:|---:|---:|---:|
| 9001 | conditional | 1 | clear / novel | 7.312 | 9.862 | +3.023 | +1.585 |
| 9001 | conditional | 1 | clear_rebase / novel | 7.261 | 8.585 | +1.745 | +2.333 |
| 9001 | conditional | 1 | fresh / novel | 7.628 | 19.581 | -5.727 | +1.804 |
| 9001 | conditional | 1 | full_reset / novel | 7.224 | 7.975 | +1.135 | +2.344 |
| 9001 | conditional | 1 | keep / novel | 7.508 | 8.450 | +1.610 | +2.032 |
| 9001 | conditional | 1 | rebase / novel | 7.452 | 8.158 | +1.318 | +2.979 |
| 9001 | conditional | 1 | rng_reset / novel | 7.302 | 10.164 | +3.324 | +2.336 |
| 9002 | conditional | 1 | clear / novel | 6.811 | 9.546 | +2.368 | +1.024 |
| 9002 | conditional | 1 | clear_rebase / novel | 6.551 | 8.411 | +1.233 | +1.433 |
| 9002 | conditional | 1 | fresh / novel | 7.453 | 18.564 | -6.425 | +0.489 |
| 9002 | conditional | 1 | full_reset / novel | 6.729 | 8.905 | +1.727 | +0.361 |
| 9002 | conditional | 1 | keep / novel | 6.754 | 8.075 | +0.897 | +0.052 |
| 9002 | conditional | 1 | rebase / novel | 7.054 | 7.666 | +0.488 | -0.373 |
| 9002 | conditional | 1 | rng_reset / novel | 6.845 | 8.291 | +1.112 | +0.430 |
| 9003 | conditional | 1 | clear / novel | 7.283 | 8.692 | +0.963 | +0.870 |
| 9003 | conditional | 1 | clear_rebase / novel | 7.386 | 10.383 | +2.654 | +1.237 |
| 9003 | conditional | 1 | fresh / novel | 7.941 | 21.027 | -3.926 | +0.576 |
| 9003 | conditional | 1 | full_reset / novel | 7.560 | 10.141 | +2.413 | +1.360 |
| 9003 | conditional | 1 | keep / novel | 7.638 | 10.269 | +2.541 | +0.491 |
| 9003 | conditional | 1 | rebase / novel | 7.652 | 11.164 | +3.435 | +0.213 |
| 9003 | conditional | 1 | rng_reset / novel | 7.434 | 8.332 | +0.603 | +0.766 |
| 9004 | conditional | 1 | clear / novel | 8.735 | 8.616 | +0.595 | +20.131 |
| 9004 | conditional | 1 | clear_rebase / novel | 8.454 | 12.803 | +4.782 | +19.810 |
| 9004 | conditional | 1 | fresh / novel | 8.904 | 19.610 | -5.268 | +20.356 |
| 9004 | conditional | 1 | full_reset / novel | 8.755 | 9.071 | +1.050 | +18.446 |
| 9004 | conditional | 1 | keep / novel | 9.750 | 8.872 | +0.851 | +18.736 |
| 9004 | conditional | 1 | rebase / novel | 8.875 | 9.188 | +1.167 | +20.212 |
| 9004 | conditional | 1 | rng_reset / novel | 9.896 | 9.325 | +1.304 | +18.857 |
| 9005 | conditional | 1 | clear / novel | 8.211 | 12.191 | +2.398 | +18.017 |
| 9005 | conditional | 1 | clear_rebase / novel | 8.125 | 13.191 | +3.398 | +17.959 |
| 9005 | conditional | 1 | fresh / novel | 8.509 | 21.792 | -3.057 | +17.719 |
| 9005 | conditional | 1 | full_reset / novel | 8.437 | 13.349 | +3.556 | +17.354 |
| 9005 | conditional | 1 | keep / novel | 9.150 | 9.005 | -0.788 | +17.537 |
| 9005 | conditional | 1 | rebase / novel | 8.598 | 11.899 | +2.106 | +17.660 |
| 9005 | conditional | 1 | rng_reset / novel | 9.613 | 10.390 | +0.597 | +13.373 |
| 9006 | conditional | 1 | clear / novel | 7.056 | 9.248 | +3.579 | +3.787 |
| 9006 | conditional | 1 | clear_rebase / novel | 6.889 | 9.609 | +3.940 | +3.455 |
| 9006 | conditional | 1 | fresh / novel | 6.936 | 16.885 | -8.218 | +4.827 |
| 9006 | conditional | 1 | full_reset / novel | 7.068 | 9.225 | +3.556 | +3.994 |
| 9006 | conditional | 1 | keep / novel | 7.506 | 7.546 | +1.877 | +2.550 |
| 9006 | conditional | 1 | rebase / novel | 7.023 | 8.732 | +3.062 | +4.067 |
| 9006 | conditional | 1 | rng_reset / novel | 7.454 | 7.815 | +2.146 | +2.808 |
| 9001 | recurrent | 1 | clear / novel | 7.310 | 22.720 | +14.405 | +2.495 |
| 9001 | recurrent | 1 | clear_rebase / novel | 7.447 | 23.387 | +15.072 | +2.182 |
| 9001 | recurrent | 1 | fresh / novel | 7.663 | 22.840 | -2.687 | +2.248 |
| 9001 | recurrent | 1 | full_reset / novel | 7.539 | 22.576 | +14.261 | +1.494 |
| 9001 | recurrent | 1 | keep / novel | 7.779 | 19.772 | +11.457 | +1.423 |
| 9001 | recurrent | 1 | rebase / novel | 7.503 | 22.297 | +13.982 | +1.631 |
| 9001 | recurrent | 1 | rng_reset / novel | 7.572 | 16.538 | +8.223 | +1.237 |
| 9002 | recurrent | 1 | clear / novel | 6.994 | 22.324 | +15.209 | +0.429 |
| 9002 | recurrent | 1 | clear_rebase / novel | 6.889 | 23.097 | +15.983 | +0.546 |
| 9002 | recurrent | 1 | fresh / novel | 7.503 | 23.656 | -1.465 | +1.006 |
| 9002 | recurrent | 1 | full_reset / novel | 6.913 | 22.931 | +15.817 | +0.108 |
| 9002 | recurrent | 1 | keep / novel | 6.926 | 18.068 | +10.953 | +0.070 |
| 9002 | recurrent | 1 | rebase / novel | 7.065 | 22.844 | +15.730 | +0.248 |
| 9002 | recurrent | 1 | rng_reset / novel | 6.996 | 19.161 | +12.046 | +0.248 |
| 9003 | recurrent | 1 | clear / novel | 7.193 | 25.224 | +5.835 | +0.838 |
| 9003 | recurrent | 1 | clear_rebase / novel | 7.336 | 25.591 | +6.202 | +1.028 |
| 9003 | recurrent | 1 | fresh / novel | 7.777 | 26.437 | +1.760 | +0.970 |
| 9003 | recurrent | 1 | full_reset / novel | 7.748 | 26.144 | +6.755 | +0.899 |
| 9003 | recurrent | 1 | keep / novel | 8.085 | 24.549 | +5.160 | +0.892 |
| 9003 | recurrent | 1 | rebase / novel | 8.075 | 24.984 | +5.596 | +0.405 |
| 9003 | recurrent | 1 | rng_reset / novel | 8.223 | 25.517 | +6.128 | +0.631 |
| 9004 | recurrent | 1 | clear / novel | 8.641 | 25.323 | +10.602 | +20.355 |
| 9004 | recurrent | 1 | clear_rebase / novel | 8.471 | 27.314 | +12.593 | +20.195 |
| 9004 | recurrent | 1 | fresh / novel | 8.540 | 27.015 | +1.820 | +20.200 |
| 9004 | recurrent | 1 | full_reset / novel | 8.622 | 26.180 | +11.458 | +20.321 |
| 9004 | recurrent | 1 | keep / novel | 9.859 | 23.093 | +8.372 | +17.876 |
| 9004 | recurrent | 1 | rebase / novel | 8.901 | 24.638 | +9.917 | +20.033 |
| 9004 | recurrent | 1 | rng_reset / novel | 10.081 | 22.230 | +7.509 | +17.062 |
| 9005 | recurrent | 1 | clear / novel | 8.514 | 24.951 | +5.523 | +17.169 |
| 9005 | recurrent | 1 | clear_rebase / novel | 8.630 | 24.790 | +5.362 | +18.909 |
| 9005 | recurrent | 1 | fresh / novel | 8.663 | 25.363 | +0.309 | +18.183 |
| 9005 | recurrent | 1 | full_reset / novel | 8.693 | 25.036 | +5.607 | +17.343 |
| 9005 | recurrent | 1 | keep / novel | 10.026 | 22.950 | +3.521 | +17.297 |
| 9005 | recurrent | 1 | rebase / novel | 8.916 | 25.117 | +5.688 | +17.645 |
| 9005 | recurrent | 1 | rng_reset / novel | 10.349 | 25.520 | +6.092 | +14.364 |
| 9006 | recurrent | 1 | clear / novel | 7.178 | 23.588 | +13.718 | +3.289 |
| 9006 | recurrent | 1 | clear_rebase / novel | 6.942 | 23.852 | +13.982 | +3.377 |
| 9006 | recurrent | 1 | fresh / novel | 7.688 | 24.254 | -0.930 | +3.305 |
| 9006 | recurrent | 1 | full_reset / novel | 7.269 | 23.411 | +13.541 | +3.424 |
| 9006 | recurrent | 1 | keep / novel | 7.404 | 18.053 | +8.183 | +3.491 |
| 9006 | recurrent | 1 | rebase / novel | 6.911 | 23.382 | +13.512 | +2.773 |
| 9006 | recurrent | 1 | rng_reset / novel | 7.259 | 15.849 | +5.978 | +3.391 |
