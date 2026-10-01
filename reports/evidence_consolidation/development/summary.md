# Adam: evidence-consolidation diagnostic

4 shared draft trajectories, 72 blocks, 12 fresh references; 2 independent seeds per architecture.

All four policies deploy predictors from the same learned draft trajectory. This tests acceptance of updates, not different representation-learning trajectories.

Primary truth error scores the performed action before training on every incoming packet, averaging all three outcome horizons with the actual preceding reported history. Focus and survival AUC instead integrate periodic independent all-action query probes; clean-support valid-old and knowledge panels are separate retention measurements.

Brier numbers below are multiplied by 100. Error contrasts are sustained minus control; negative is better. Intervals use 20,000 paired-seed bootstrap resamples, seed 27192026. Packets and blocks are not independent seed replicates.

| Model | Control | Overall truth difference [95% interval] | Late difference | Growing advantage, late minus early |
|---|---|---:|---:|---:|
| conditional | immediate | +0.211 [+0.195, +0.227] | +0.319 [+0.315, +0.323] | -0.018 [-0.136, +0.100] |
| conditional | periodic | -0.002 [-0.077, +0.073] | +0.164 [+0.075, +0.253] | -0.140 [-0.167, -0.114] |
| conditional | single | -0.011 [-0.115, +0.094] | +0.043 [-0.100, +0.186] | +0.015 [-0.073, +0.103] |
| recurrent | immediate | +0.180 [+0.167, +0.193] | +0.285 [+0.232, +0.339] | -0.298 [-0.304, -0.292] |
| recurrent | periodic | -0.179 [-0.252, -0.107] | -0.101 [-0.243, +0.041] | -0.375 [-0.425, -0.325] |
| recurrent | single | -0.166 [-0.239, -0.092] | -0.099 [-0.226, +0.028] | -0.360 [-0.411, -0.308] |

Positive growing advantage means the candidate's relative error improved more in cycle 3 than cycle 1; it does not establish compounding representation gains.

* conditional: **INCONCLUSIVE**; requires six independent seeds; development cohort is descriptive.
* recurrent: **INCONCLUSIVE**; requires six independent seeds; development cohort is descriptive.

| Model | Gate | Pass | Value | Tolerance | Interval permits violation |
|---|---|---|---:|---:|---|
| conditional | fresh_qualification | False | - | all fresh groups | - |
| conditional | immediate_overall_truth_gain | False | +0.211 | <= -0.200 | True |
| conditional | immediate_late_truth_gain | False | +0.319 | <= -0.200 | True |
| conditional | immediate_overall_consistency | None | 0 | 5 of 6 | - |
| conditional | periodic_overall_truth_gain | False | -0.002 | <= -0.200 | True |
| conditional | periodic_late_truth_gain | False | +0.164 | <= -0.200 | True |
| conditional | periodic_overall_consistency | None | 1 | 5 of 6 | - |
| conditional | immediate_intro_focus_auc_guardrail | True | +0.062 | <= +0.500 | False |
| conditional | immediate_return_truth_guardrail | False | +0.831 | <= +0.500 | True |
| conditional | immediate_noise_truth_guardrail | True | -0.242 | <= +0.500 | False |
| conditional | immediate_valid_after_guardrail | True | -0.746 | <= +0.500 | False |
| conditional | immediate_survival_guardrail | True | -0.043 | >= -1.000 | False |
| recurrent | fresh_qualification | False | - | all fresh groups | - |
| recurrent | immediate_overall_truth_gain | False | +0.180 | <= -0.200 | True |
| recurrent | immediate_late_truth_gain | False | +0.285 | <= -0.200 | True |
| recurrent | immediate_overall_consistency | None | 0 | 5 of 6 | - |
| recurrent | periodic_overall_truth_gain | False | -0.179 | <= -0.200 | True |
| recurrent | periodic_late_truth_gain | False | -0.101 | <= -0.200 | True |
| recurrent | periodic_overall_consistency | None | 2 | 5 of 6 | - |
| recurrent | immediate_intro_focus_auc_guardrail | True | +0.020 | <= +0.500 | False |
| recurrent | immediate_return_truth_guardrail | False | +1.636 | <= +0.500 | True |
| recurrent | immediate_noise_truth_guardrail | True | -1.507 | <= +0.500 | False |
| recurrent | immediate_valid_after_guardrail | True | -0.198 | <= +0.500 | False |
| recurrent | immediate_survival_guardrail | True | -0.079 | >= -1.000 | False |

| Fresh qualification group | Episodes | Marginal gain x100 | Cue benefit x100 | Pass |
|---|---:|---:|---:|---|
| conditional_stage_1 | 2 | 15.219 | 8.949 | True |
| conditional_stage_2 | 2 | 15.053 | 10.467 | True |
| conditional_stage_3 | 2 | 14.942 | 0.329 | True |
| conditional_cue_0 | 2 | 15.208 | 0.898 | True |
| conditional_cue_1 | 2 | 15.269 | 0.172 | False |
| conditional_cue_2 | 2 | 14.736 | 18.675 | True |
| recurrent_stage_1 | 2 | 14.994 | 9.003 | True |
| recurrent_stage_2 | 2 | 14.954 | 10.373 | True |
| recurrent_stage_3 | 2 | 14.446 | 0.495 | True |
| recurrent_cue_0 | 2 | 14.875 | 1.033 | True |
| recurrent_cue_1 | 2 | 14.796 | 0.112 | False |
| recurrent_cue_2 | 2 | 14.724 | 18.726 | True |

| Seed | Model | Policy | Overall truth | Early truth | Late truth | Valid-old level | Survival % | Adoptions |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 11001 | conditional | immediate | 9.379 | 9.374 | 9.321 | 15.546 | 63.440 | 0 |
| 11001 | conditional | periodic | 9.684 | 9.593 | 9.560 | 15.250 | 63.243 | 576 |
| 11001 | conditional | single | 9.722 | 9.550 | 9.735 | 15.197 | 63.104 | 290 |
| 11001 | conditional | sustained | 9.607 | 9.554 | 9.636 | 14.648 | 63.498 | 100 |
| 11001 | recurrent | immediate | 10.064 | 10.310 | 10.008 | 21.815 | 63.290 | 0 |
| 11001 | recurrent | periodic | 10.364 | 10.740 | 10.305 | 21.679 | 63.132 | 576 |
| 11001 | recurrent | single | 10.350 | 10.740 | 10.319 | 21.468 | 63.077 | 269 |
| 11001 | recurrent | sustained | 10.257 | 10.356 | 10.347 | 21.219 | 63.152 | 104 |
| 11002 | conditional | immediate | 9.525 | 10.464 | 9.151 | 13.075 | 64.228 | 0 |
| 11002 | conditional | periodic | 9.647 | 10.801 | 9.222 | 12.742 | 64.041 | 576 |
| 11002 | conditional | single | 9.626 | 10.774 | 9.288 | 12.955 | 64.048 | 292 |
| 11002 | conditional | sustained | 9.720 | 10.887 | 9.475 | 12.482 | 64.084 | 96 |
| 11002 | recurrent | immediate | 9.865 | 9.879 | 10.245 | 21.719 | 63.976 | 0 |
| 11002 | recurrent | periodic | 10.284 | 10.373 | 10.720 | 21.619 | 63.810 | 576 |
| 11002 | recurrent | single | 10.271 | 10.340 | 10.703 | 21.586 | 63.867 | 275 |
| 11002 | recurrent | sustained | 10.032 | 9.806 | 10.477 | 21.918 | 63.956 | 100 |

| Seed | Model | Block (1-based) | Condition | Policy | Truth | Reported | Focus AUC | Valid-old level | Survival % |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|
| 11001 | conditional | 1 | introduction_1 | immediate | 7.355 | 7.355 | 7.318 | 8.573 | 72.598 |
| 11001 | conditional | 2 | introduction_2 | immediate | 10.403 | 10.403 | 11.344 | 12.959 | 56.482 |
| 11001 | conditional | 3 | introduction_3 | immediate | 9.037 | 9.037 | 9.427 | 18.058 | 60.153 |
| 11001 | conditional | 4 | clean | immediate | 8.820 | 8.820 | 8.764 | 22.629 | 62.045 |
| 11001 | conditional | 5 | noise | immediate | 10.242 | 15.677 | 10.229 | 25.640 | 59.488 |
| 11001 | conditional | 6 | recovery | immediate | 9.155 | 9.155 | 9.037 | 18.024 | 61.703 |
| 11001 | conditional | 7 | return | immediate | 8.557 | 8.557 | 8.925 | 11.570 | 71.671 |
| 11001 | conditional | 8 | revision | immediate | 10.096 | 10.096 | 10.637 | 12.137 | 62.668 |
| 11001 | conditional | 9 | clean | immediate | 9.479 | 9.479 | 9.163 | 18.569 | 61.957 |
| 11001 | conditional | 10 | noise | immediate | 10.902 | 15.408 | 10.985 | 23.378 | 59.207 |
| 11001 | conditional | 11 | recovery | immediate | 9.333 | 9.333 | 8.918 | 19.441 | 62.021 |
| 11001 | conditional | 12 | return | immediate | 9.125 | 9.125 | 9.033 | 10.302 | 71.326 |
| 11001 | conditional | 13 | revision | immediate | 9.722 | 9.722 | 10.596 | 11.867 | 63.199 |
| 11001 | conditional | 14 | clean | immediate | 8.790 | 8.790 | 8.752 | 9.297 | 62.408 |
| 11001 | conditional | 15 | noise | immediate | 10.261 | 15.510 | 10.431 | 17.862 | 60.449 |
| 11001 | conditional | 16 | recovery | immediate | 8.945 | 8.945 | 9.022 | 17.345 | 61.694 |
| 11001 | conditional | 17 | return | immediate | 8.874 | 8.874 | 8.994 | 11.328 | 71.188 |
| 11001 | conditional | 18 | revision | immediate | 9.734 | 9.734 | 10.634 | 10.851 | 61.661 |
| 11001 | conditional | 1 | introduction_1 | periodic | 7.331 | 7.331 | 7.517 | 8.259 | 72.513 |
| 11001 | conditional | 2 | introduction_2 | periodic | 10.786 | 10.786 | 11.590 | 17.633 | 56.281 |
| 11001 | conditional | 3 | introduction_3 | periodic | 8.857 | 8.857 | 9.466 | 24.776 | 60.101 |
| 11001 | conditional | 4 | clean | periodic | 8.588 | 8.588 | 8.772 | 17.619 | 62.033 |
| 11001 | conditional | 5 | noise | periodic | 10.134 | 15.759 | 10.159 | 19.033 | 59.613 |
| 11001 | conditional | 6 | recovery | periodic | 9.242 | 9.242 | 9.100 | 18.311 | 61.548 |
| 11001 | conditional | 7 | return | periodic | 9.222 | 9.222 | 9.240 | 11.951 | 71.399 |
| 11001 | conditional | 8 | revision | periodic | 10.779 | 10.779 | 10.700 | 12.117 | 62.509 |
| 11001 | conditional | 9 | clean | periodic | 9.693 | 9.693 | 9.162 | 18.658 | 62.030 |
| 11001 | conditional | 10 | noise | periodic | 12.763 | 16.399 | 12.430 | 17.734 | 57.626 |
| 11001 | conditional | 11 | recovery | periodic | 9.440 | 9.440 | 8.941 | 18.817 | 62.000 |
| 11001 | conditional | 12 | return | periodic | 9.645 | 9.645 | 9.318 | 12.133 | 71.155 |
| 11001 | conditional | 13 | revision | periodic | 10.028 | 10.028 | 11.330 | 11.869 | 62.509 |
| 11001 | conditional | 14 | clean | periodic | 8.948 | 8.948 | 8.786 | 8.885 | 62.363 |
| 11001 | conditional | 15 | noise | periodic | 10.266 | 15.831 | 10.372 | 16.935 | 60.489 |
| 11001 | conditional | 16 | recovery | periodic | 9.069 | 9.069 | 9.069 | 17.611 | 61.639 |
| 11001 | conditional | 17 | return | periodic | 9.543 | 9.543 | 9.207 | 11.578 | 71.024 |
| 11001 | conditional | 18 | revision | periodic | 9.975 | 9.975 | 10.718 | 10.584 | 61.548 |
| 11001 | conditional | 1 | introduction_1 | single | 7.426 | 7.426 | 7.537 | 8.259 | 72.598 |
| 11001 | conditional | 2 | introduction_2 | single | 10.916 | 10.916 | 11.836 | 17.633 | 55.469 |
| 11001 | conditional | 3 | introduction_3 | single | 8.858 | 8.858 | 9.273 | 24.776 | 60.028 |
| 11001 | conditional | 4 | clean | single | 8.597 | 8.597 | 8.756 | 17.619 | 61.917 |
| 11001 | conditional | 5 | noise | single | 9.894 | 15.620 | 9.885 | 19.033 | 59.845 |
| 11001 | conditional | 6 | recovery | single | 9.107 | 9.107 | 9.033 | 18.311 | 61.328 |
| 11001 | conditional | 7 | return | single | 9.281 | 9.281 | 9.220 | 11.951 | 71.429 |
| 11001 | conditional | 8 | revision | single | 10.873 | 10.873 | 10.863 | 12.800 | 62.567 |
| 11001 | conditional | 9 | clean | single | 9.610 | 9.610 | 8.949 | 18.939 | 61.942 |
| 11001 | conditional | 10 | noise | single | 12.413 | 16.164 | 12.498 | 17.734 | 57.199 |
| 11001 | conditional | 11 | recovery | single | 9.473 | 9.473 | 8.929 | 18.540 | 62.088 |
| 11001 | conditional | 12 | return | single | 9.460 | 9.460 | 9.250 | 10.529 | 71.118 |
| 11001 | conditional | 13 | revision | single | 10.413 | 10.413 | 11.529 | 11.695 | 62.396 |
| 11001 | conditional | 14 | clean | single | 9.024 | 9.024 | 8.798 | 8.885 | 62.405 |
| 11001 | conditional | 15 | noise | single | 10.273 | 15.891 | 10.309 | 16.935 | 60.495 |
| 11001 | conditional | 16 | recovery | single | 8.950 | 8.950 | 8.994 | 17.710 | 61.560 |
| 11001 | conditional | 17 | return | single | 10.358 | 10.358 | 10.355 | 11.594 | 70.288 |
| 11001 | conditional | 18 | revision | single | 10.071 | 10.071 | 10.968 | 10.597 | 61.197 |
| 11001 | conditional | 1 | introduction_1 | sustained | 7.031 | 7.031 | 6.973 | 9.164 | 73.224 |
| 11001 | conditional | 2 | introduction_2 | sustained | 10.912 | 10.912 | 12.202 | 10.102 | 56.259 |
| 11001 | conditional | 3 | introduction_3 | sustained | 8.684 | 8.684 | 9.339 | 18.028 | 60.345 |
| 11001 | conditional | 4 | clean | sustained | 8.512 | 8.512 | 8.600 | 16.600 | 62.177 |
| 11001 | conditional | 5 | noise | sustained | 9.797 | 15.540 | 9.793 | 24.335 | 59.985 |
| 11001 | conditional | 6 | recovery | sustained | 9.004 | 9.004 | 8.873 | 18.478 | 61.572 |
| 11001 | conditional | 7 | return | sustained | 9.480 | 9.480 | 9.292 | 10.465 | 71.277 |
| 11001 | conditional | 8 | revision | sustained | 10.974 | 10.974 | 11.156 | 12.117 | 62.091 |
| 11001 | conditional | 9 | clean | sustained | 9.551 | 9.551 | 9.139 | 17.762 | 62.854 |
| 11001 | conditional | 10 | noise | sustained | 11.504 | 15.522 | 11.660 | 18.529 | 59.113 |
| 11001 | conditional | 11 | recovery | sustained | 9.520 | 9.520 | 8.953 | 18.287 | 62.552 |
| 11001 | conditional | 12 | return | sustained | 9.473 | 9.473 | 9.241 | 10.197 | 71.307 |
| 11001 | conditional | 13 | revision | sustained | 10.299 | 10.299 | 11.153 | 12.575 | 63.007 |
| 11001 | conditional | 14 | clean | sustained | 9.144 | 9.144 | 8.748 | 9.472 | 62.259 |
| 11001 | conditional | 15 | noise | sustained | 10.201 | 15.703 | 10.325 | 17.558 | 61.337 |
| 11001 | conditional | 16 | recovery | sustained | 9.221 | 9.221 | 9.277 | 17.611 | 61.649 |
| 11001 | conditional | 17 | return | sustained | 9.703 | 9.703 | 9.463 | 11.788 | 70.767 |
| 11001 | conditional | 18 | revision | sustained | 9.908 | 9.908 | 10.976 | 10.597 | 61.191 |
| 11001 | recurrent | 1 | introduction_1 | immediate | 7.291 | 7.291 | 7.068 | 17.897 | 71.924 |
| 11001 | recurrent | 2 | introduction_2 | immediate | 10.598 | 10.598 | 11.360 | 21.862 | 56.445 |
| 11001 | recurrent | 3 | introduction_3 | immediate | 9.184 | 9.184 | 9.447 | 25.243 | 60.880 |
| 11001 | recurrent | 4 | clean | immediate | 8.654 | 8.654 | 8.726 | 23.209 | 62.460 |
| 11001 | recurrent | 5 | noise | immediate | 10.956 | 16.488 | 11.117 | 25.327 | 60.266 |
| 11001 | recurrent | 6 | recovery | immediate | 8.952 | 8.952 | 8.668 | 24.772 | 62.555 |
| 11001 | recurrent | 7 | return | immediate | 12.691 | 12.691 | 12.627 | 14.104 | 67.358 |
| 11001 | recurrent | 8 | revision | immediate | 10.297 | 10.297 | 11.118 | 22.798 | 61.975 |
| 11001 | recurrent | 9 | clean | immediate | 9.341 | 9.341 | 9.094 | 24.706 | 62.787 |
| 11001 | recurrent | 10 | noise | immediate | 11.706 | 15.823 | 11.793 | 23.701 | 59.299 |
| 11001 | recurrent | 11 | recovery | immediate | 9.278 | 9.278 | 8.789 | 25.561 | 63.297 |
| 11001 | recurrent | 12 | return | immediate | 12.084 | 12.084 | 12.366 | 13.596 | 68.887 |
| 11001 | recurrent | 13 | revision | immediate | 10.078 | 10.078 | 10.460 | 22.332 | 63.370 |
| 11001 | recurrent | 14 | clean | immediate | 9.139 | 9.139 | 9.296 | 25.066 | 62.366 |
| 11001 | recurrent | 15 | noise | immediate | 10.699 | 16.019 | 10.873 | 25.572 | 61.374 |
| 11001 | recurrent | 16 | recovery | immediate | 8.858 | 8.858 | 8.903 | 25.119 | 62.631 |
| 11001 | recurrent | 17 | return | immediate | 11.382 | 11.382 | 11.277 | 12.183 | 68.756 |
| 11001 | recurrent | 18 | revision | immediate | 9.962 | 9.962 | 10.845 | 19.631 | 62.582 |
| 11001 | recurrent | 1 | introduction_1 | periodic | 7.448 | 7.448 | 7.043 | 16.384 | 71.960 |
| 11001 | recurrent | 2 | introduction_2 | periodic | 10.921 | 10.921 | 11.647 | 24.381 | 55.978 |
| 11001 | recurrent | 3 | introduction_3 | periodic | 9.305 | 9.305 | 9.405 | 24.142 | 60.941 |
| 11001 | recurrent | 4 | clean | periodic | 8.576 | 8.576 | 8.769 | 23.258 | 62.558 |
| 11001 | recurrent | 5 | noise | periodic | 10.769 | 16.326 | 10.973 | 25.514 | 60.388 |
| 11001 | recurrent | 6 | recovery | periodic | 9.300 | 9.300 | 8.789 | 24.426 | 62.576 |
| 11001 | recurrent | 7 | return | periodic | 13.900 | 13.900 | 13.589 | 12.705 | 66.772 |
| 11001 | recurrent | 8 | revision | periodic | 11.153 | 11.153 | 11.806 | 22.501 | 61.603 |
| 11001 | recurrent | 9 | clean | periodic | 9.384 | 9.384 | 9.064 | 24.426 | 62.973 |
| 11001 | recurrent | 10 | noise | periodic | 11.281 | 17.205 | 11.615 | 23.088 | 59.509 |
| 11001 | recurrent | 11 | recovery | periodic | 9.043 | 9.043 | 8.769 | 24.606 | 63.223 |
| 11001 | recurrent | 12 | return | periodic | 13.482 | 13.482 | 13.090 | 16.263 | 68.286 |
| 11001 | recurrent | 13 | revision | periodic | 10.460 | 10.460 | 10.746 | 23.049 | 63.205 |
| 11001 | recurrent | 14 | clean | periodic | 9.061 | 9.061 | 9.202 | 24.903 | 62.659 |
| 11001 | recurrent | 15 | noise | periodic | 10.587 | 16.158 | 10.792 | 21.909 | 61.325 |
| 11001 | recurrent | 16 | recovery | periodic | 8.822 | 8.822 | 8.865 | 24.505 | 62.448 |
| 11001 | recurrent | 17 | return | periodic | 12.539 | 12.539 | 12.645 | 15.530 | 67.990 |
| 11001 | recurrent | 18 | revision | periodic | 10.518 | 10.518 | 11.359 | 18.637 | 61.972 |
| 11001 | recurrent | 1 | introduction_1 | single | 7.259 | 7.259 | 6.978 | 16.384 | 72.534 |
| 11001 | recurrent | 2 | introduction_2 | single | 11.135 | 11.135 | 11.818 | 24.381 | 55.685 |
| 11001 | recurrent | 3 | introduction_3 | single | 9.310 | 9.310 | 9.508 | 25.178 | 60.818 |
| 11001 | recurrent | 4 | clean | single | 8.611 | 8.611 | 8.693 | 23.318 | 62.753 |
| 11001 | recurrent | 5 | noise | single | 10.107 | 15.826 | 10.651 | 25.379 | 60.638 |
| 11001 | recurrent | 6 | recovery | single | 9.190 | 9.190 | 8.824 | 23.822 | 62.637 |
| 11001 | recurrent | 7 | return | single | 14.309 | 14.309 | 13.676 | 12.905 | 67.020 |
| 11001 | recurrent | 8 | revision | single | 11.481 | 11.481 | 12.288 | 22.501 | 61.243 |
| 11001 | recurrent | 9 | clean | single | 9.283 | 9.283 | 8.954 | 24.491 | 62.915 |
| 11001 | recurrent | 10 | noise | single | 10.678 | 16.585 | 10.904 | 23.088 | 60.001 |
| 11001 | recurrent | 11 | recovery | single | 9.109 | 9.109 | 8.785 | 24.554 | 62.936 |
| 11001 | recurrent | 12 | return | single | 13.455 | 13.455 | 13.374 | 15.999 | 68.237 |
| 11001 | recurrent | 13 | revision | single | 10.771 | 10.771 | 11.312 | 22.815 | 62.869 |
| 11001 | recurrent | 14 | clean | single | 8.846 | 8.846 | 9.193 | 24.903 | 62.552 |
| 11001 | recurrent | 15 | noise | single | 10.373 | 15.996 | 10.782 | 21.909 | 60.873 |
| 11001 | recurrent | 16 | recovery | single | 8.766 | 8.766 | 8.873 | 24.505 | 62.442 |
| 11001 | recurrent | 17 | return | single | 12.948 | 12.948 | 13.140 | 11.651 | 67.572 |
| 11001 | recurrent | 18 | revision | single | 10.661 | 10.661 | 11.489 | 18.637 | 61.664 |
| 11001 | recurrent | 1 | introduction_1 | sustained | 7.091 | 7.091 | 6.551 | 19.398 | 72.021 |
| 11001 | recurrent | 2 | introduction_2 | sustained | 11.079 | 11.079 | 11.894 | 23.484 | 55.310 |
| 11001 | recurrent | 3 | introduction_3 | sustained | 9.065 | 9.065 | 9.111 | 25.833 | 60.236 |
| 11001 | recurrent | 4 | clean | sustained | 8.872 | 8.872 | 8.897 | 23.258 | 62.808 |
| 11001 | recurrent | 5 | noise | sustained | 8.197 | 15.049 | 8.452 | 23.258 | 63.812 |
| 11001 | recurrent | 6 | recovery | sustained | 8.434 | 8.434 | 8.438 | 24.004 | 62.759 |
| 11001 | recurrent | 7 | return | sustained | 14.345 | 14.345 | 13.854 | 12.905 | 67.245 |
| 11001 | recurrent | 8 | revision | sustained | 11.933 | 11.933 | 13.436 | 22.291 | 60.291 |
| 11001 | recurrent | 9 | clean | sustained | 9.433 | 9.433 | 9.285 | 24.080 | 63.168 |
| 11001 | recurrent | 10 | noise | sustained | 10.859 | 16.660 | 11.002 | 23.088 | 59.857 |
| 11001 | recurrent | 11 | recovery | sustained | 9.258 | 9.258 | 9.211 | 24.554 | 62.589 |
| 11001 | recurrent | 12 | return | sustained | 13.549 | 13.549 | 13.339 | 16.263 | 68.311 |
| 11001 | recurrent | 13 | revision | sustained | 10.783 | 10.783 | 11.467 | 22.318 | 62.473 |
| 11001 | recurrent | 14 | clean | sustained | 8.977 | 8.977 | 9.227 | 24.656 | 62.976 |
| 11001 | recurrent | 15 | noise | sustained | 10.112 | 15.791 | 10.776 | 20.530 | 60.849 |
| 11001 | recurrent | 16 | recovery | sustained | 9.156 | 9.156 | 9.490 | 23.196 | 61.905 |
| 11001 | recurrent | 17 | return | sustained | 12.637 | 12.637 | 12.692 | 11.651 | 68.185 |
| 11001 | recurrent | 18 | revision | sustained | 10.852 | 10.852 | 11.460 | 17.174 | 61.948 |
| 11002 | conditional | 1 | introduction_1 | immediate | 9.091 | 9.091 | 9.318 | 9.411 | 58.719 |
| 11002 | conditional | 2 | introduction_2 | immediate | 8.696 | 8.696 | 8.576 | 10.845 | 64.316 |
| 11002 | conditional | 3 | introduction_3 | immediate | 8.438 | 8.438 | 8.985 | 12.391 | 64.343 |
| 11002 | conditional | 4 | clean | immediate | 8.130 | 8.130 | 8.294 | 12.071 | 65.265 |
| 11002 | conditional | 5 | noise | immediate | 9.527 | 14.975 | 9.470 | 15.519 | 63.065 |
| 11002 | conditional | 6 | recovery | immediate | 8.542 | 8.542 | 8.604 | 18.617 | 64.578 |
| 11002 | conditional | 7 | return | immediate | 16.649 | 16.649 | 15.820 | 22.035 | 66.660 |
| 11002 | conditional | 8 | revision | immediate | 9.472 | 9.472 | 10.453 | 13.942 | 62.256 |
| 11002 | conditional | 9 | clean | immediate | 8.551 | 8.551 | 8.638 | 12.298 | 64.319 |
| 11002 | conditional | 10 | noise | immediate | 12.528 | 13.748 | 12.734 | 10.760 | 59.845 |
| 11002 | conditional | 11 | recovery | immediate | 8.214 | 8.214 | 8.607 | 12.048 | 64.746 |
| 11002 | conditional | 12 | return | immediate | 8.766 | 8.766 | 8.359 | 11.671 | 71.402 |
| 11002 | conditional | 13 | revision | immediate | 9.085 | 9.085 | 9.609 | 11.595 | 62.769 |
| 11002 | conditional | 14 | clean | immediate | 8.230 | 8.230 | 8.819 | 10.144 | 64.490 |
| 11002 | conditional | 15 | noise | immediate | 10.090 | 15.841 | 10.338 | 12.698 | 61.920 |
| 11002 | conditional | 16 | recovery | immediate | 8.848 | 8.848 | 8.763 | 11.460 | 63.629 |
| 11002 | conditional | 17 | return | immediate | 9.124 | 9.124 | 8.614 | 16.523 | 72.168 |
| 11002 | conditional | 18 | revision | immediate | 9.465 | 9.465 | 10.175 | 11.323 | 61.609 |
| 11002 | conditional | 1 | introduction_1 | periodic | 9.449 | 9.449 | 9.720 | 9.361 | 58.258 |
| 11002 | conditional | 2 | introduction_2 | periodic | 8.386 | 8.386 | 8.595 | 10.791 | 64.325 |
| 11002 | conditional | 3 | introduction_3 | periodic | 8.570 | 8.570 | 8.990 | 11.860 | 64.365 |
| 11002 | conditional | 4 | clean | periodic | 8.074 | 8.074 | 8.314 | 12.920 | 65.179 |
| 11002 | conditional | 5 | noise | periodic | 9.493 | 15.107 | 9.403 | 16.224 | 63.156 |
| 11002 | conditional | 6 | recovery | periodic | 8.513 | 8.513 | 8.622 | 17.104 | 64.529 |
| 11002 | conditional | 7 | return | periodic | 17.407 | 17.407 | 16.488 | 21.462 | 66.394 |
| 11002 | conditional | 8 | revision | periodic | 10.518 | 10.518 | 10.931 | 14.885 | 61.859 |
| 11002 | conditional | 9 | clean | periodic | 8.368 | 8.368 | 8.655 | 12.079 | 64.313 |
| 11002 | conditional | 10 | noise | periodic | 12.066 | 14.268 | 12.649 | 11.189 | 59.552 |
| 11002 | conditional | 11 | recovery | periodic | 8.137 | 8.137 | 8.867 | 12.337 | 64.255 |
| 11002 | conditional | 12 | return | periodic | 9.322 | 9.322 | 9.292 | 11.242 | 70.844 |
| 11002 | conditional | 13 | revision | periodic | 9.234 | 9.234 | 10.200 | 11.714 | 62.476 |
| 11002 | conditional | 14 | clean | periodic | 8.266 | 8.266 | 8.825 | 10.086 | 64.450 |
| 11002 | conditional | 15 | noise | periodic | 10.135 | 15.937 | 10.352 | 12.493 | 61.761 |
| 11002 | conditional | 16 | recovery | periodic | 8.752 | 8.752 | 8.800 | 11.426 | 63.547 |
| 11002 | conditional | 17 | return | periodic | 9.251 | 9.251 | 8.836 | 10.988 | 71.985 |
| 11002 | conditional | 18 | revision | periodic | 9.703 | 9.703 | 10.187 | 11.198 | 61.496 |
| 11002 | conditional | 1 | introduction_1 | single | 9.249 | 9.249 | 9.607 | 9.361 | 58.057 |
| 11002 | conditional | 2 | introduction_2 | single | 8.390 | 8.390 | 8.634 | 10.791 | 64.459 |
| 11002 | conditional | 3 | introduction_3 | single | 8.422 | 8.422 | 8.753 | 11.860 | 64.322 |
| 11002 | conditional | 4 | clean | single | 8.071 | 8.071 | 8.273 | 12.920 | 64.832 |
| 11002 | conditional | 5 | noise | single | 9.239 | 14.994 | 9.321 | 16.224 | 62.869 |
| 11002 | conditional | 6 | recovery | single | 8.500 | 8.500 | 8.636 | 17.104 | 64.389 |
| 11002 | conditional | 7 | return | single | 17.569 | 17.569 | 16.637 | 22.228 | 66.388 |
| 11002 | conditional | 8 | revision | single | 10.493 | 10.493 | 10.949 | 14.885 | 62.152 |
| 11002 | conditional | 9 | clean | single | 8.277 | 8.277 | 8.558 | 12.204 | 64.148 |
| 11002 | conditional | 10 | noise | single | 12.136 | 14.338 | 12.404 | 11.898 | 59.927 |
| 11002 | conditional | 11 | recovery | single | 8.266 | 8.266 | 8.557 | 12.398 | 64.673 |
| 11002 | conditional | 12 | return | single | 9.086 | 9.086 | 9.297 | 11.170 | 70.837 |
| 11002 | conditional | 13 | revision | single | 9.129 | 9.129 | 10.199 | 11.919 | 62.521 |
| 11002 | conditional | 14 | clean | single | 8.294 | 8.294 | 8.814 | 10.913 | 64.523 |
| 11002 | conditional | 15 | noise | single | 9.991 | 15.691 | 10.473 | 12.885 | 62.399 |
| 11002 | conditional | 16 | recovery | single | 9.010 | 9.010 | 8.823 | 11.621 | 63.321 |
| 11002 | conditional | 17 | return | single | 9.478 | 9.478 | 9.049 | 11.612 | 71.866 |
| 11002 | conditional | 18 | revision | single | 9.668 | 9.668 | 10.348 | 11.198 | 61.179 |
| 11002 | conditional | 1 | introduction_1 | sustained | 9.227 | 9.227 | 9.519 | 9.361 | 57.269 |
| 11002 | conditional | 2 | introduction_2 | sustained | 8.439 | 8.439 | 8.401 | 10.023 | 65.399 |
| 11002 | conditional | 3 | introduction_3 | sustained | 8.431 | 8.431 | 8.905 | 10.151 | 64.194 |
| 11002 | conditional | 4 | clean | sustained | 8.199 | 8.199 | 8.194 | 10.769 | 65.076 |
| 11002 | conditional | 5 | noise | sustained | 8.828 | 14.801 | 9.132 | 15.215 | 63.818 |
| 11002 | conditional | 6 | recovery | sustained | 8.590 | 8.590 | 8.546 | 17.104 | 65.247 |
| 11002 | conditional | 7 | return | sustained | 17.861 | 17.861 | 16.811 | 22.228 | 66.736 |
| 11002 | conditional | 8 | revision | sustained | 10.958 | 10.958 | 11.167 | 14.433 | 61.307 |
| 11002 | conditional | 9 | clean | sustained | 8.412 | 8.412 | 8.432 | 12.204 | 63.919 |
| 11002 | conditional | 10 | noise | sustained | 11.684 | 14.082 | 12.189 | 12.291 | 60.257 |
| 11002 | conditional | 11 | recovery | sustained | 8.325 | 8.325 | 9.518 | 11.460 | 63.281 |
| 11002 | conditional | 12 | return | sustained | 9.376 | 9.376 | 8.798 | 11.189 | 71.216 |
| 11002 | conditional | 13 | revision | sustained | 9.251 | 9.251 | 10.570 | 11.714 | 62.146 |
| 11002 | conditional | 14 | clean | sustained | 8.273 | 8.273 | 8.895 | 10.389 | 65.189 |
| 11002 | conditional | 15 | noise | sustained | 10.082 | 16.048 | 10.381 | 12.493 | 62.201 |
| 11002 | conditional | 16 | recovery | sustained | 8.834 | 8.834 | 8.501 | 10.859 | 62.659 |
| 11002 | conditional | 17 | return | sustained | 10.186 | 10.186 | 9.656 | 11.592 | 71.954 |
| 11002 | conditional | 18 | revision | sustained | 9.998 | 9.998 | 10.288 | 11.198 | 61.642 |
| 11002 | recurrent | 1 | introduction_1 | immediate | 9.543 | 9.543 | 10.154 | 22.172 | 58.163 |
| 11002 | recurrent | 2 | introduction_2 | immediate | 8.652 | 8.652 | 9.000 | 22.842 | 64.990 |
| 11002 | recurrent | 3 | introduction_3 | immediate | 8.519 | 8.519 | 8.967 | 22.817 | 64.551 |
| 11002 | recurrent | 4 | clean | immediate | 8.207 | 8.207 | 8.332 | 23.419 | 65.024 |
| 11002 | recurrent | 5 | noise | immediate | 10.088 | 15.530 | 9.983 | 20.905 | 62.442 |
| 11002 | recurrent | 6 | recovery | immediate | 8.247 | 8.247 | 8.492 | 23.386 | 64.178 |
| 11002 | recurrent | 7 | return | immediate | 13.406 | 13.406 | 12.349 | 19.423 | 68.253 |
| 11002 | recurrent | 8 | revision | immediate | 9.445 | 9.445 | 10.082 | 21.570 | 62.000 |
| 11002 | recurrent | 9 | clean | immediate | 8.657 | 8.657 | 8.848 | 22.962 | 64.490 |
| 11002 | recurrent | 10 | noise | immediate | 11.087 | 15.464 | 11.032 | 21.179 | 62.488 |
| 11002 | recurrent | 11 | recovery | immediate | 8.614 | 8.614 | 8.731 | 22.085 | 64.536 |
| 11002 | recurrent | 12 | return | immediate | 12.473 | 12.473 | 12.005 | 20.103 | 68.597 |
| 11002 | recurrent | 13 | revision | immediate | 9.402 | 9.402 | 10.113 | 21.122 | 61.801 |
| 11002 | recurrent | 14 | clean | immediate | 8.464 | 8.464 | 8.797 | 22.019 | 64.252 |
| 11002 | recurrent | 15 | noise | immediate | 10.513 | 16.153 | 10.731 | 20.235 | 62.033 |
| 11002 | recurrent | 16 | recovery | immediate | 8.726 | 8.726 | 8.858 | 22.087 | 64.200 |
| 11002 | recurrent | 17 | return | immediate | 13.550 | 13.550 | 13.059 | 20.271 | 68.381 |
| 11002 | recurrent | 18 | revision | immediate | 9.971 | 9.971 | 10.357 | 22.343 | 61.188 |
| 11002 | recurrent | 1 | introduction_1 | periodic | 9.971 | 9.971 | 10.301 | 21.911 | 58.224 |
| 11002 | recurrent | 2 | introduction_2 | periodic | 8.615 | 8.615 | 9.007 | 22.700 | 64.825 |
| 11002 | recurrent | 3 | introduction_3 | periodic | 8.604 | 8.604 | 8.961 | 23.373 | 64.526 |
| 11002 | recurrent | 4 | clean | periodic | 8.313 | 8.313 | 8.288 | 23.603 | 65.027 |
| 11002 | recurrent | 5 | noise | periodic | 10.305 | 15.852 | 9.948 | 21.872 | 62.408 |
| 11002 | recurrent | 6 | recovery | periodic | 8.594 | 8.594 | 8.584 | 22.600 | 64.090 |
| 11002 | recurrent | 7 | return | periodic | 14.507 | 14.507 | 13.266 | 19.523 | 67.749 |
| 11002 | recurrent | 8 | revision | periodic | 10.148 | 10.148 | 10.307 | 21.953 | 62.122 |
| 11002 | recurrent | 9 | clean | periodic | 8.735 | 8.735 | 8.804 | 21.583 | 64.645 |
| 11002 | recurrent | 10 | noise | periodic | 11.221 | 17.430 | 10.885 | 21.478 | 62.659 |
| 11002 | recurrent | 11 | recovery | periodic | 8.495 | 8.495 | 8.635 | 22.688 | 64.688 |
| 11002 | recurrent | 12 | return | periodic | 14.061 | 14.061 | 13.099 | 19.122 | 67.535 |
| 11002 | recurrent | 13 | revision | periodic | 9.937 | 9.937 | 10.473 | 20.530 | 61.533 |
| 11002 | recurrent | 14 | clean | periodic | 8.323 | 8.323 | 8.797 | 21.320 | 64.111 |
| 11002 | recurrent | 15 | noise | periodic | 10.372 | 16.110 | 10.756 | 21.727 | 62.033 |
| 11002 | recurrent | 16 | recovery | periodic | 8.893 | 8.893 | 8.905 | 22.190 | 64.011 |
| 11002 | recurrent | 17 | return | periodic | 15.114 | 15.114 | 13.886 | 19.701 | 67.572 |
| 11002 | recurrent | 18 | revision | periodic | 10.897 | 10.897 | 10.666 | 21.262 | 60.812 |
| 11002 | recurrent | 1 | introduction_1 | single | 9.972 | 9.972 | 10.307 | 21.911 | 58.109 |
| 11002 | recurrent | 2 | introduction_2 | single | 8.454 | 8.454 | 9.196 | 22.157 | 65.012 |
| 11002 | recurrent | 3 | introduction_3 | single | 8.752 | 8.752 | 8.902 | 23.373 | 64.505 |
| 11002 | recurrent | 4 | clean | single | 8.052 | 8.052 | 8.295 | 23.603 | 65.314 |
| 11002 | recurrent | 5 | noise | single | 10.004 | 15.665 | 9.867 | 21.872 | 62.836 |
| 11002 | recurrent | 6 | recovery | single | 8.744 | 8.744 | 8.613 | 22.600 | 64.615 |
| 11002 | recurrent | 7 | return | single | 14.809 | 14.809 | 13.576 | 19.829 | 67.410 |
| 11002 | recurrent | 8 | revision | single | 10.092 | 10.092 | 10.561 | 21.491 | 61.951 |
| 11002 | recurrent | 9 | clean | single | 8.690 | 8.690 | 8.921 | 20.968 | 64.731 |
| 11002 | recurrent | 10 | noise | single | 10.860 | 16.887 | 10.566 | 21.478 | 63.010 |
| 11002 | recurrent | 11 | recovery | single | 8.515 | 8.515 | 8.633 | 22.688 | 64.584 |
| 11002 | recurrent | 12 | return | single | 14.523 | 14.523 | 13.830 | 19.122 | 67.462 |
| 11002 | recurrent | 13 | revision | single | 9.891 | 9.891 | 10.531 | 20.530 | 61.557 |
| 11002 | recurrent | 14 | clean | single | 8.232 | 8.232 | 8.723 | 22.059 | 64.240 |
| 11002 | recurrent | 15 | noise | single | 10.025 | 15.957 | 10.438 | 21.727 | 62.320 |
| 11002 | recurrent | 16 | recovery | single | 8.919 | 8.919 | 8.923 | 22.384 | 63.937 |
| 11002 | recurrent | 17 | return | single | 15.415 | 15.415 | 14.550 | 19.701 | 67.471 |
| 11002 | recurrent | 18 | revision | single | 10.922 | 10.922 | 11.167 | 21.047 | 60.541 |
| 11002 | recurrent | 1 | introduction_1 | sustained | 10.217 | 10.217 | 10.976 | 23.239 | 56.558 |
| 11002 | recurrent | 2 | introduction_2 | sustained | 8.482 | 8.482 | 8.842 | 22.894 | 65.616 |
| 11002 | recurrent | 3 | introduction_3 | sustained | 8.316 | 8.316 | 8.742 | 23.922 | 65.030 |
| 11002 | recurrent | 4 | clean | sustained | 7.841 | 7.841 | 8.203 | 22.941 | 64.203 |
| 11002 | recurrent | 5 | noise | sustained | 7.903 | 14.401 | 8.227 | 22.941 | 65.289 |
| 11002 | recurrent | 6 | recovery | sustained | 8.227 | 8.227 | 8.473 | 23.285 | 64.853 |
| 11002 | recurrent | 7 | return | sustained | 14.904 | 14.904 | 13.541 | 19.523 | 67.682 |
| 11002 | recurrent | 8 | revision | sustained | 10.157 | 10.157 | 10.390 | 19.843 | 61.819 |
| 11002 | recurrent | 9 | clean | sustained | 8.561 | 8.561 | 8.605 | 22.540 | 64.645 |
| 11002 | recurrent | 10 | noise | sustained | 10.103 | 16.683 | 9.811 | 21.105 | 63.489 |
| 11002 | recurrent | 11 | recovery | sustained | 8.513 | 8.513 | 8.675 | 22.688 | 64.185 |
| 11002 | recurrent | 12 | return | sustained | 14.619 | 14.619 | 13.914 | 21.367 | 67.105 |
| 11002 | recurrent | 13 | revision | sustained | 10.340 | 10.340 | 10.739 | 20.325 | 60.730 |
| 11002 | recurrent | 14 | clean | sustained | 8.320 | 8.320 | 8.691 | 20.849 | 64.410 |
| 11002 | recurrent | 15 | noise | sustained | 8.836 | 15.653 | 9.235 | 23.276 | 64.337 |
| 11002 | recurrent | 16 | recovery | sustained | 9.088 | 9.088 | 9.157 | 22.384 | 63.065 |
| 11002 | recurrent | 17 | return | sustained | 15.346 | 15.346 | 14.650 | 20.363 | 67.407 |
| 11002 | recurrent | 18 | revision | sustained | 10.793 | 10.793 | 10.738 | 21.047 | 60.776 |

summary.json preserves both valid-old modes, every paired seed difference, per-block cumulative loss, knowledge endpoints, and window diagnostics. raw_results.jsonl.gz contains every complete block and fresh-reference record. Immediate has no adoption gate; its reported zero adoptions is not zero learning. Noise-block policy differences are not causal noise penalties without matched clean twin trajectories.

The immediate standalone learner needs one model. The delayed scaffold can retain three models (draft, proposal, committed) and uses two policy/evidence prediction passes per packet versus one for immediate, in addition to the shared training budget. resource_rows records parameter bytes, replay/optimizer sizes, and operation counts. These are logical standalone counts for the implemented scaffold, not measured runtime or total process RAM; evaluator probes are excluded from those online inference counts.
