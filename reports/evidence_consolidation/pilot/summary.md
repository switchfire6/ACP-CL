# Adam: evidence-consolidation diagnostic

12 shared draft trajectories, 216 blocks, 36 fresh references; 6 independent seeds per architecture.

All four policies deploy predictors from the same learned draft trajectory. This tests acceptance of updates, not different representation-learning trajectories.

Primary truth error scores the performed action before training on every incoming packet, averaging all three outcome horizons with the actual preceding reported history. Focus and survival AUC instead integrate periodic independent all-action query probes; clean-support valid-old and knowledge panels are separate retention measurements.

Brier numbers below are multiplied by 100. Error contrasts are sustained minus control; negative is better. Intervals use 20,000 paired-seed bootstrap resamples, seed 27192026. Packets and blocks are not independent seed replicates.

| Model | Control | Overall truth difference [95% interval] | Late difference | Growing advantage, late minus early |
|---|---|---:|---:|---:|
| conditional | immediate | +0.169 [+0.105, +0.226] | +0.083 [+0.017, +0.155] | +0.066 [-0.076, +0.192] |
| conditional | periodic | -0.017 [-0.063, +0.031] | -0.048 [-0.126, +0.028] | +0.009 [-0.171, +0.235] |
| conditional | single | -0.031 [-0.066, +0.004] | -0.049 [-0.118, +0.008] | -0.012 [-0.219, +0.234] |
| recurrent | immediate | +0.280 [+0.229, +0.336] | +0.267 [+0.079, +0.445] | +0.070 [-0.239, +0.379] |
| recurrent | periodic | -0.042 [-0.111, +0.028] | -0.106 [-0.321, +0.100] | +0.062 [-0.216, +0.368] |
| recurrent | single | -0.074 [-0.154, +0.010] | -0.173 [-0.389, +0.060] | +0.069 [-0.226, +0.378] |

Positive growing advantage means the candidate's relative error improved more in cycle 3 than cycle 1; it does not establish compounding representation gains.

* conditional: **FAIL**; failed gates: immediate_overall_truth_gain, immediate_late_truth_gain, immediate_overall_consistency, periodic_overall_truth_gain, periodic_late_truth_gain, periodic_overall_consistency, immediate_return_truth_guardrail.
* recurrent: **FAIL**; failed gates: immediate_overall_truth_gain, immediate_late_truth_gain, immediate_overall_consistency, periodic_overall_truth_gain, periodic_late_truth_gain, periodic_overall_consistency, immediate_return_truth_guardrail.

| Model | Gate | Pass | Value | Tolerance | Interval permits violation |
|---|---|---|---:|---:|---|
| conditional | fresh_qualification | True | - | all fresh groups | - |
| conditional | immediate_overall_truth_gain | False | +0.169 | <= -0.200 | True |
| conditional | immediate_late_truth_gain | False | +0.083 | <= -0.200 | True |
| conditional | immediate_overall_consistency | False | 0 | 5 of 6 | - |
| conditional | periodic_overall_truth_gain | False | -0.017 | <= -0.200 | True |
| conditional | periodic_late_truth_gain | False | -0.048 | <= -0.200 | True |
| conditional | periodic_overall_consistency | False | 4 | 5 of 6 | - |
| conditional | immediate_intro_focus_auc_guardrail | True | -0.009 | <= +0.500 | False |
| conditional | immediate_return_truth_guardrail | False | +0.781 | <= +0.500 | True |
| conditional | immediate_noise_truth_guardrail | True | -0.205 | <= +0.500 | False |
| conditional | immediate_valid_after_guardrail | True | +0.134 | <= +0.500 | False |
| conditional | immediate_survival_guardrail | True | -0.090 | >= -1.000 | False |
| recurrent | fresh_qualification | True | - | all fresh groups | - |
| recurrent | immediate_overall_truth_gain | False | +0.280 | <= -0.200 | True |
| recurrent | immediate_late_truth_gain | False | +0.267 | <= -0.200 | True |
| recurrent | immediate_overall_consistency | False | 0 | 5 of 6 | - |
| recurrent | periodic_overall_truth_gain | False | -0.042 | <= -0.200 | True |
| recurrent | periodic_late_truth_gain | False | -0.106 | <= -0.200 | True |
| recurrent | periodic_overall_consistency | False | 4 | 5 of 6 | - |
| recurrent | immediate_intro_focus_auc_guardrail | True | +0.114 | <= +0.500 | False |
| recurrent | immediate_return_truth_guardrail | False | +1.624 | <= +0.500 | True |
| recurrent | immediate_noise_truth_guardrail | True | -1.101 | <= +0.500 | False |
| recurrent | immediate_valid_after_guardrail | True | -0.346 | <= +0.500 | False |
| recurrent | immediate_survival_guardrail | True | -0.068 | >= -1.000 | False |

| Fresh qualification group | Episodes | Marginal gain x100 | Cue benefit x100 | Pass |
|---|---:|---:|---:|---|
| conditional_stage_1 | 6 | 15.558 | 7.296 | True |
| conditional_stage_2 | 6 | 14.833 | 6.930 | True |
| conditional_stage_3 | 6 | 14.826 | 6.950 | True |
| conditional_cue_0 | 6 | 14.953 | 1.839 | True |
| conditional_cue_1 | 6 | 15.397 | 0.516 | True |
| conditional_cue_2 | 6 | 14.867 | 18.822 | True |
| recurrent_stage_1 | 6 | 15.428 | 7.408 | True |
| recurrent_stage_2 | 6 | 14.788 | 6.986 | True |
| recurrent_stage_3 | 6 | 14.746 | 6.987 | True |
| recurrent_cue_0 | 6 | 14.853 | 1.821 | True |
| recurrent_cue_1 | 6 | 15.245 | 0.671 | True |
| recurrent_cue_2 | 6 | 14.864 | 18.889 | True |

| Seed | Model | Policy | Overall truth | Early truth | Late truth | Valid-old level | Survival % | Adoptions |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 12001 | conditional | immediate | 9.498 | 9.634 | 9.595 | 11.673 | 66.410 | 0 |
| 12001 | conditional | periodic | 9.642 | 9.858 | 9.696 | 11.754 | 66.345 | 576 |
| 12001 | conditional | single | 9.625 | 9.904 | 9.656 | 11.173 | 66.338 | 297 |
| 12001 | conditional | sustained | 9.534 | 9.435 | 9.604 | 12.073 | 66.593 | 106 |
| 12001 | recurrent | immediate | 10.047 | 10.220 | 10.274 | 22.448 | 65.745 | 0 |
| 12001 | recurrent | periodic | 10.421 | 10.628 | 10.751 | 22.991 | 65.511 | 576 |
| 12001 | recurrent | single | 10.431 | 10.692 | 10.782 | 22.687 | 65.557 | 266 |
| 12001 | recurrent | sustained | 10.332 | 10.790 | 10.195 | 22.031 | 65.551 | 97 |
| 12002 | conditional | immediate | 9.576 | 9.902 | 9.562 | 12.157 | 65.139 | 0 |
| 12002 | conditional | periodic | 9.732 | 10.073 | 9.690 | 12.388 | 65.052 | 576 |
| 12002 | conditional | single | 9.760 | 10.071 | 9.665 | 12.671 | 65.066 | 294 |
| 12002 | conditional | sustained | 9.701 | 9.977 | 9.584 | 12.101 | 65.239 | 91 |
| 12002 | recurrent | immediate | 9.976 | 10.397 | 9.945 | 19.873 | 64.923 | 0 |
| 12002 | recurrent | periodic | 10.353 | 10.800 | 10.428 | 20.509 | 64.757 | 576 |
| 12002 | recurrent | single | 10.367 | 10.746 | 10.366 | 20.282 | 64.687 | 284 |
| 12002 | recurrent | sustained | 10.176 | 10.285 | 10.314 | 20.127 | 64.884 | 82 |
| 12003 | conditional | immediate | 9.449 | 9.334 | 9.587 | 11.207 | 66.669 | 0 |
| 12003 | conditional | periodic | 9.606 | 9.324 | 9.797 | 11.516 | 66.595 | 576 |
| 12003 | conditional | single | 9.652 | 9.312 | 9.801 | 11.256 | 66.627 | 293 |
| 12003 | conditional | sustained | 9.689 | 9.636 | 9.600 | 11.087 | 66.581 | 99 |
| 12003 | recurrent | immediate | 9.837 | 9.756 | 9.928 | 22.719 | 66.388 | 0 |
| 12003 | recurrent | periodic | 10.146 | 10.172 | 10.189 | 22.725 | 66.289 | 576 |
| 12003 | recurrent | single | 10.191 | 10.275 | 10.356 | 22.867 | 66.296 | 268 |
| 12003 | recurrent | sustained | 10.083 | 10.157 | 9.932 | 22.295 | 66.320 | 103 |
| 12004 | conditional | immediate | 9.378 | 9.381 | 9.522 | 12.837 | 62.732 | 0 |
| 12004 | conditional | periodic | 9.640 | 9.702 | 9.679 | 12.354 | 62.535 | 576 |
| 12004 | conditional | single | 9.625 | 9.838 | 9.683 | 12.419 | 62.487 | 296 |
| 12004 | conditional | sustained | 9.643 | 9.720 | 9.714 | 12.973 | 62.384 | 86 |
| 12004 | recurrent | immediate | 9.877 | 10.185 | 9.777 | 21.912 | 62.438 | 0 |
| 12004 | recurrent | periodic | 10.167 | 10.520 | 10.156 | 21.970 | 62.409 | 576 |
| 12004 | recurrent | single | 10.269 | 10.772 | 10.331 | 21.731 | 62.244 | 298 |
| 12004 | recurrent | sustained | 10.095 | 10.477 | 10.112 | 21.606 | 62.520 | 98 |
| 12005 | conditional | immediate | 9.058 | 8.925 | 9.381 | 10.247 | 64.037 | 0 |
| 12005 | conditional | periodic | 9.260 | 9.161 | 9.423 | 10.837 | 63.919 | 576 |
| 12005 | conditional | single | 9.276 | 9.206 | 9.436 | 10.601 | 63.901 | 257 |
| 12005 | conditional | sustained | 9.228 | 9.143 | 9.413 | 10.385 | 63.870 | 76 |
| 12005 | recurrent | immediate | 9.567 | 9.789 | 9.706 | 19.707 | 63.774 | 0 |
| 12005 | recurrent | periodic | 9.871 | 10.204 | 10.014 | 19.774 | 63.638 | 576 |
| 12005 | recurrent | single | 9.902 | 10.156 | 10.111 | 19.638 | 63.634 | 303 |
| 12005 | recurrent | sustained | 9.958 | 10.370 | 10.057 | 19.145 | 63.741 | 102 |
| 12006 | conditional | immediate | 9.378 | 9.669 | 9.391 | 11.755 | 65.257 | 0 |
| 12006 | conditional | periodic | 9.570 | 9.850 | 9.537 | 11.465 | 65.026 | 576 |
| 12006 | conditional | single | 9.597 | 9.775 | 9.591 | 12.537 | 64.946 | 254 |
| 12006 | conditional | sustained | 9.552 | 9.825 | 9.620 | 12.059 | 65.039 | 84 |
| 12006 | recurrent | immediate | 10.007 | 10.490 | 9.839 | 19.316 | 64.844 | 0 |
| 12006 | recurrent | periodic | 10.287 | 10.801 | 10.170 | 19.528 | 64.714 | 576 |
| 12006 | recurrent | single | 10.273 | 10.848 | 10.168 | 18.990 | 64.694 | 282 |
| 12006 | recurrent | sustained | 10.347 | 10.785 | 10.464 | 18.697 | 64.685 | 112 |

| Seed | Model | Block (1-based) | Condition | Policy | Truth | Reported | Focus AUC | Valid-old level | Survival % |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|
| 12001 | conditional | 1 | introduction_1 | immediate | 7.155 | 7.155 | 7.615 | 8.800 | 77.060 |
| 12001 | conditional | 2 | introduction_2 | immediate | 10.485 | 10.485 | 11.304 | 10.235 | 60.873 |
| 12001 | conditional | 3 | introduction_3 | immediate | 8.965 | 8.965 | 9.297 | 9.200 | 64.609 |
| 12001 | conditional | 4 | clean | immediate | 8.930 | 8.930 | 8.922 | 10.061 | 65.204 |
| 12001 | conditional | 5 | noise | immediate | 10.385 | 16.052 | 10.350 | 13.909 | 62.180 |
| 12001 | conditional | 6 | recovery | immediate | 9.002 | 9.002 | 8.917 | 12.033 | 64.355 |
| 12001 | conditional | 7 | return | immediate | 9.643 | 9.643 | 9.629 | 10.809 | 72.937 |
| 12001 | conditional | 8 | revision | immediate | 10.212 | 10.212 | 10.925 | 11.871 | 63.623 |
| 12001 | conditional | 9 | clean | immediate | 8.889 | 8.889 | 9.281 | 9.795 | 65.176 |
| 12001 | conditional | 10 | noise | immediate | 11.389 | 15.381 | 11.507 | 12.596 | 62.665 |
| 12001 | conditional | 11 | recovery | immediate | 8.927 | 8.927 | 8.770 | 11.143 | 64.804 |
| 12001 | conditional | 12 | return | immediate | 8.996 | 8.996 | 9.161 | 11.517 | 73.734 |
| 12001 | conditional | 13 | revision | immediate | 10.011 | 10.011 | 10.516 | 11.192 | 65.359 |
| 12001 | conditional | 14 | clean | immediate | 8.908 | 8.908 | 8.901 | 10.650 | 66.281 |
| 12001 | conditional | 15 | noise | immediate | 11.061 | 16.561 | 10.927 | 12.070 | 63.171 |
| 12001 | conditional | 16 | recovery | immediate | 9.066 | 9.066 | 8.967 | 13.460 | 65.149 |
| 12001 | conditional | 17 | return | immediate | 9.070 | 9.070 | 9.106 | 18.982 | 73.386 |
| 12001 | conditional | 18 | revision | immediate | 9.870 | 9.870 | 10.801 | 11.791 | 64.819 |
| 12001 | conditional | 1 | introduction_1 | periodic | 7.262 | 7.262 | 7.628 | 8.305 | 77.100 |
| 12001 | conditional | 2 | introduction_2 | periodic | 10.966 | 10.966 | 11.551 | 8.743 | 60.574 |
| 12001 | conditional | 3 | introduction_3 | periodic | 8.891 | 8.891 | 9.322 | 8.672 | 64.545 |
| 12001 | conditional | 4 | clean | periodic | 9.230 | 9.230 | 8.889 | 9.960 | 65.192 |
| 12001 | conditional | 5 | noise | periodic | 10.451 | 16.128 | 10.316 | 14.290 | 62.326 |
| 12001 | conditional | 6 | recovery | periodic | 9.021 | 9.021 | 8.966 | 12.225 | 64.209 |
| 12001 | conditional | 7 | return | periodic | 10.226 | 10.226 | 10.358 | 10.690 | 72.519 |
| 12001 | conditional | 8 | revision | periodic | 10.362 | 10.362 | 11.016 | 12.080 | 63.507 |
| 12001 | conditional | 9 | clean | periodic | 8.815 | 8.815 | 9.287 | 9.525 | 65.195 |
| 12001 | conditional | 10 | noise | periodic | 11.492 | 17.339 | 11.506 | 12.552 | 62.683 |
| 12001 | conditional | 11 | recovery | periodic | 8.987 | 8.987 | 8.795 | 11.616 | 64.828 |
| 12001 | conditional | 12 | return | periodic | 9.279 | 9.279 | 9.415 | 11.078 | 73.566 |
| 12001 | conditional | 13 | revision | periodic | 10.094 | 10.094 | 10.608 | 10.859 | 65.277 |
| 12001 | conditional | 14 | clean | periodic | 8.725 | 8.725 | 8.908 | 9.584 | 66.284 |
| 12001 | conditional | 15 | noise | periodic | 11.265 | 16.925 | 10.851 | 11.594 | 63.278 |
| 12001 | conditional | 16 | recovery | periodic | 9.145 | 9.145 | 9.033 | 13.386 | 65.076 |
| 12001 | conditional | 17 | return | periodic | 9.241 | 9.241 | 9.378 | 24.757 | 73.270 |
| 12001 | conditional | 18 | revision | periodic | 10.105 | 10.105 | 10.864 | 11.656 | 64.789 |
| 12001 | conditional | 1 | introduction_1 | single | 7.081 | 7.081 | 7.421 | 8.305 | 77.142 |
| 12001 | conditional | 2 | introduction_2 | single | 11.074 | 11.074 | 11.712 | 8.743 | 60.648 |
| 12001 | conditional | 3 | introduction_3 | single | 8.943 | 8.943 | 9.328 | 9.595 | 64.615 |
| 12001 | conditional | 4 | clean | single | 9.135 | 9.135 | 8.791 | 9.999 | 65.421 |
| 12001 | conditional | 5 | noise | single | 10.243 | 15.929 | 10.374 | 14.394 | 62.405 |
| 12001 | conditional | 6 | recovery | single | 9.078 | 9.078 | 8.965 | 12.225 | 64.203 |
| 12001 | conditional | 7 | return | single | 10.665 | 10.665 | 10.535 | 10.690 | 72.366 |
| 12001 | conditional | 8 | revision | single | 10.397 | 10.397 | 10.940 | 11.565 | 63.446 |
| 12001 | conditional | 9 | clean | single | 8.968 | 8.968 | 9.180 | 9.525 | 65.182 |
| 12001 | conditional | 10 | noise | single | 10.857 | 16.813 | 10.960 | 12.552 | 62.769 |
| 12001 | conditional | 11 | recovery | single | 9.123 | 9.123 | 8.855 | 11.616 | 64.871 |
| 12001 | conditional | 12 | return | single | 9.242 | 9.242 | 9.448 | 13.354 | 73.575 |
| 12001 | conditional | 13 | revision | single | 10.172 | 10.172 | 10.652 | 10.859 | 65.503 |
| 12001 | conditional | 14 | clean | single | 8.785 | 8.785 | 8.867 | 9.584 | 65.973 |
| 12001 | conditional | 15 | noise | single | 11.133 | 16.796 | 10.552 | 11.331 | 63.147 |
| 12001 | conditional | 16 | recovery | single | 9.105 | 9.105 | 9.000 | 13.386 | 65.128 |
| 12001 | conditional | 17 | return | single | 9.181 | 9.181 | 9.376 | 11.731 | 73.157 |
| 12001 | conditional | 18 | revision | single | 10.076 | 10.076 | 11.019 | 11.656 | 64.539 |
| 12001 | conditional | 1 | introduction_1 | sustained | 7.257 | 7.257 | 7.642 | 8.305 | 76.736 |
| 12001 | conditional | 2 | introduction_2 | sustained | 11.077 | 11.077 | 11.706 | 10.123 | 59.665 |
| 12001 | conditional | 3 | introduction_3 | sustained | 9.111 | 9.111 | 9.308 | 9.595 | 64.310 |
| 12001 | conditional | 4 | clean | sustained | 8.607 | 8.607 | 8.242 | 9.860 | 65.808 |
| 12001 | conditional | 5 | noise | sustained | 8.807 | 15.481 | 8.323 | 9.860 | 65.820 |
| 12001 | conditional | 6 | recovery | sustained | 8.764 | 8.764 | 8.724 | 12.425 | 64.301 |
| 12001 | conditional | 7 | return | sustained | 10.509 | 10.509 | 10.576 | 10.778 | 72.653 |
| 12001 | conditional | 8 | revision | sustained | 10.490 | 10.490 | 11.213 | 12.460 | 63.788 |
| 12001 | conditional | 9 | clean | sustained | 8.973 | 8.973 | 9.074 | 10.209 | 65.466 |
| 12001 | conditional | 10 | noise | sustained | 11.515 | 17.281 | 11.590 | 11.810 | 62.244 |
| 12001 | conditional | 11 | recovery | sustained | 9.163 | 9.163 | 8.911 | 11.262 | 64.935 |
| 12001 | conditional | 12 | return | sustained | 9.376 | 9.376 | 9.267 | 17.361 | 73.471 |
| 12001 | conditional | 13 | revision | sustained | 9.933 | 9.933 | 10.820 | 11.481 | 65.344 |
| 12001 | conditional | 14 | clean | sustained | 8.696 | 8.696 | 8.959 | 10.527 | 66.837 |
| 12001 | conditional | 15 | noise | sustained | 10.291 | 16.401 | 9.411 | 12.099 | 64.499 |
| 12001 | conditional | 16 | recovery | sustained | 9.316 | 9.316 | 9.141 | 12.752 | 65.094 |
| 12001 | conditional | 17 | return | sustained | 9.625 | 9.625 | 9.391 | 24.757 | 72.910 |
| 12001 | conditional | 18 | revision | sustained | 10.094 | 10.094 | 11.197 | 11.656 | 64.795 |
| 12001 | recurrent | 1 | introduction_1 | immediate | 7.534 | 7.534 | 8.127 | 24.126 | 76.544 |
| 12001 | recurrent | 2 | introduction_2 | immediate | 10.711 | 10.711 | 11.531 | 22.739 | 60.422 |
| 12001 | recurrent | 3 | introduction_3 | immediate | 9.146 | 9.146 | 9.530 | 24.037 | 64.435 |
| 12001 | recurrent | 4 | clean | immediate | 9.015 | 9.015 | 8.444 | 24.694 | 65.732 |
| 12001 | recurrent | 5 | noise | immediate | 10.800 | 16.317 | 10.393 | 25.015 | 62.222 |
| 12001 | recurrent | 6 | recovery | immediate | 8.466 | 8.466 | 8.471 | 24.948 | 64.520 |
| 12001 | recurrent | 7 | return | immediate | 12.643 | 12.643 | 12.849 | 21.028 | 69.482 |
| 12001 | recurrent | 8 | revision | immediate | 10.178 | 10.178 | 10.678 | 22.050 | 62.329 |
| 12001 | recurrent | 9 | clean | immediate | 9.109 | 9.109 | 8.847 | 23.042 | 65.677 |
| 12001 | recurrent | 10 | noise | immediate | 11.420 | 15.692 | 11.347 | 22.940 | 62.460 |
| 12001 | recurrent | 11 | recovery | immediate | 8.902 | 8.902 | 8.724 | 23.500 | 64.957 |
| 12001 | recurrent | 12 | return | immediate | 11.407 | 11.407 | 10.724 | 17.199 | 70.819 |
| 12001 | recurrent | 13 | revision | immediate | 10.139 | 10.139 | 10.262 | 22.512 | 64.410 |
| 12001 | recurrent | 14 | clean | immediate | 8.653 | 8.653 | 8.634 | 19.438 | 66.434 |
| 12001 | recurrent | 15 | noise | immediate | 10.744 | 16.467 | 10.616 | 21.676 | 63.553 |
| 12001 | recurrent | 16 | recovery | immediate | 9.171 | 9.171 | 8.738 | 23.639 | 65.140 |
| 12001 | recurrent | 17 | return | immediate | 12.864 | 12.864 | 12.650 | 19.804 | 70.023 |
| 12001 | recurrent | 18 | revision | immediate | 9.937 | 9.937 | 10.230 | 21.670 | 64.243 |
| 12001 | recurrent | 1 | introduction_1 | periodic | 7.489 | 7.489 | 7.805 | 24.699 | 76.703 |
| 12001 | recurrent | 2 | introduction_2 | periodic | 10.902 | 10.902 | 11.819 | 23.574 | 60.193 |
| 12001 | recurrent | 3 | introduction_3 | periodic | 8.980 | 8.980 | 9.682 | 24.451 | 64.087 |
| 12001 | recurrent | 4 | clean | periodic | 8.835 | 8.835 | 8.445 | 26.133 | 65.372 |
| 12001 | recurrent | 5 | noise | periodic | 10.421 | 16.110 | 10.431 | 25.674 | 62.183 |
| 12001 | recurrent | 6 | recovery | periodic | 8.677 | 8.677 | 8.532 | 26.272 | 64.346 |
| 12001 | recurrent | 7 | return | periodic | 14.181 | 14.181 | 13.336 | 19.250 | 68.994 |
| 12001 | recurrent | 8 | revision | periodic | 11.025 | 11.025 | 11.334 | 23.255 | 61.853 |
| 12001 | recurrent | 9 | clean | periodic | 8.773 | 8.773 | 8.828 | 22.289 | 65.729 |
| 12001 | recurrent | 10 | noise | periodic | 11.706 | 17.919 | 11.139 | 22.424 | 62.695 |
| 12001 | recurrent | 11 | recovery | periodic | 8.991 | 8.991 | 8.797 | 24.256 | 64.844 |
| 12001 | recurrent | 12 | return | periodic | 12.826 | 12.826 | 11.668 | 21.244 | 70.355 |
| 12001 | recurrent | 13 | revision | periodic | 11.015 | 11.015 | 10.730 | 21.401 | 63.986 |
| 12001 | recurrent | 14 | clean | periodic | 8.741 | 8.741 | 8.610 | 19.407 | 66.495 |
| 12001 | recurrent | 15 | noise | periodic | 10.862 | 16.395 | 10.706 | 23.314 | 63.510 |
| 12001 | recurrent | 16 | recovery | periodic | 9.202 | 9.202 | 8.752 | 23.693 | 64.929 |
| 12001 | recurrent | 17 | return | periodic | 14.583 | 14.583 | 13.717 | 20.502 | 69.083 |
| 12001 | recurrent | 18 | revision | periodic | 10.366 | 10.366 | 10.840 | 22.011 | 63.840 |
| 12001 | recurrent | 1 | introduction_1 | single | 7.569 | 7.569 | 8.004 | 24.699 | 76.849 |
| 12001 | recurrent | 2 | introduction_2 | single | 10.812 | 10.812 | 11.746 | 24.748 | 60.223 |
| 12001 | recurrent | 3 | introduction_3 | single | 8.975 | 8.975 | 9.607 | 24.451 | 63.409 |
| 12001 | recurrent | 4 | clean | single | 8.867 | 8.867 | 8.640 | 25.007 | 65.390 |
| 12001 | recurrent | 5 | noise | single | 9.981 | 15.919 | 9.890 | 24.537 | 63.293 |
| 12001 | recurrent | 6 | recovery | single | 8.677 | 8.677 | 8.523 | 26.272 | 64.346 |
| 12001 | recurrent | 7 | return | single | 14.704 | 14.704 | 14.053 | 19.250 | 68.671 |
| 12001 | recurrent | 8 | revision | single | 11.229 | 11.229 | 11.320 | 23.255 | 61.816 |
| 12001 | recurrent | 9 | clean | single | 9.040 | 9.040 | 8.957 | 22.369 | 65.665 |
| 12001 | recurrent | 10 | noise | single | 10.782 | 16.884 | 10.429 | 22.819 | 63.052 |
| 12001 | recurrent | 11 | recovery | single | 9.042 | 9.042 | 8.743 | 19.937 | 65.125 |
| 12001 | recurrent | 12 | return | single | 12.945 | 12.945 | 12.073 | 20.283 | 70.267 |
| 12001 | recurrent | 13 | revision | single | 11.235 | 11.235 | 10.872 | 21.401 | 63.712 |
| 12001 | recurrent | 14 | clean | single | 8.654 | 8.654 | 8.567 | 19.858 | 66.409 |
| 12001 | recurrent | 15 | noise | single | 10.599 | 16.260 | 10.410 | 23.314 | 64.322 |
| 12001 | recurrent | 16 | recovery | single | 9.179 | 9.179 | 8.746 | 23.693 | 65.027 |
| 12001 | recurrent | 17 | return | single | 15.024 | 15.024 | 13.961 | 20.502 | 68.466 |
| 12001 | recurrent | 18 | revision | single | 10.452 | 10.452 | 10.949 | 21.978 | 63.974 |
| 12001 | recurrent | 1 | introduction_1 | sustained | 7.472 | 7.472 | 7.821 | 22.743 | 76.453 |
| 12001 | recurrent | 2 | introduction_2 | sustained | 11.133 | 11.133 | 12.114 | 21.953 | 59.262 |
| 12001 | recurrent | 3 | introduction_3 | sustained | 9.128 | 9.128 | 9.428 | 24.379 | 64.465 |
| 12001 | recurrent | 4 | clean | sustained | 9.042 | 9.042 | 8.552 | 25.007 | 64.542 |
| 12001 | recurrent | 5 | noise | sustained | 10.280 | 16.064 | 10.455 | 25.674 | 62.653 |
| 12001 | recurrent | 6 | recovery | sustained | 8.781 | 8.781 | 8.707 | 24.222 | 63.437 |
| 12001 | recurrent | 7 | return | sustained | 14.739 | 14.739 | 14.377 | 19.250 | 67.807 |
| 12001 | recurrent | 8 | revision | sustained | 11.108 | 11.108 | 11.276 | 22.712 | 61.728 |
| 12001 | recurrent | 9 | clean | sustained | 8.914 | 8.914 | 8.744 | 22.369 | 65.723 |
| 12001 | recurrent | 10 | noise | sustained | 11.121 | 17.029 | 10.932 | 23.814 | 62.509 |
| 12001 | recurrent | 11 | recovery | sustained | 9.199 | 9.199 | 8.587 | 21.595 | 64.417 |
| 12001 | recurrent | 12 | return | sustained | 12.786 | 12.786 | 12.045 | 21.244 | 70.065 |
| 12001 | recurrent | 13 | revision | sustained | 11.292 | 11.292 | 11.184 | 18.361 | 63.708 |
| 12001 | recurrent | 14 | clean | sustained | 8.474 | 8.474 | 8.920 | 19.091 | 66.205 |
| 12001 | recurrent | 15 | noise | sustained | 8.367 | 15.311 | 8.099 | 19.091 | 67.514 |
| 12001 | recurrent | 16 | recovery | sustained | 8.576 | 8.576 | 8.086 | 22.349 | 66.843 |
| 12001 | recurrent | 17 | return | sustained | 14.924 | 14.924 | 13.958 | 20.775 | 68.814 |
| 12001 | recurrent | 18 | revision | sustained | 10.636 | 10.636 | 11.506 | 21.936 | 63.776 |
| 12002 | conditional | 1 | introduction_1 | immediate | 7.038 | 7.038 | 6.428 | 11.756 | 71.460 |
| 12002 | conditional | 2 | introduction_2 | immediate | 7.770 | 7.770 | 7.321 | 9.450 | 73.727 |
| 12002 | conditional | 3 | introduction_3 | immediate | 11.044 | 11.044 | 10.792 | 10.961 | 58.496 |
| 12002 | conditional | 4 | clean | immediate | 9.578 | 9.578 | 8.923 | 10.976 | 61.636 |
| 12002 | conditional | 5 | noise | immediate | 10.702 | 15.820 | 9.950 | 17.300 | 59.900 |
| 12002 | conditional | 6 | recovery | immediate | 9.222 | 9.222 | 8.546 | 14.949 | 61.642 |
| 12002 | conditional | 7 | return | immediate | 9.483 | 9.483 | 9.121 | 10.526 | 73.447 |
| 12002 | conditional | 8 | revision | immediate | 10.522 | 10.522 | 10.800 | 14.647 | 61.542 |
| 12002 | conditional | 9 | clean | immediate | 9.487 | 9.487 | 8.769 | 10.212 | 63.306 |
| 12002 | conditional | 10 | noise | immediate | 11.091 | 15.061 | 10.702 | 11.587 | 61.630 |
| 12002 | conditional | 11 | recovery | immediate | 9.362 | 9.362 | 8.545 | 12.027 | 62.808 |
| 12002 | conditional | 12 | return | immediate | 8.993 | 8.993 | 8.985 | 10.600 | 74.286 |
| 12002 | conditional | 13 | revision | immediate | 10.260 | 10.260 | 10.140 | 12.313 | 63.284 |
| 12002 | conditional | 14 | clean | immediate | 9.241 | 9.241 | 8.644 | 10.232 | 63.721 |
| 12002 | conditional | 15 | noise | immediate | 10.751 | 15.786 | 10.318 | 12.658 | 61.420 |
| 12002 | conditional | 16 | recovery | immediate | 9.394 | 9.394 | 8.585 | 10.734 | 62.915 |
| 12002 | conditional | 17 | return | immediate | 8.903 | 8.903 | 9.017 | 11.038 | 73.642 |
| 12002 | conditional | 18 | revision | immediate | 9.519 | 9.519 | 9.624 | 16.854 | 63.632 |
| 12002 | conditional | 1 | introduction_1 | periodic | 7.316 | 7.316 | 6.488 | 7.614 | 71.335 |
| 12002 | conditional | 2 | introduction_2 | periodic | 7.499 | 7.499 | 7.350 | 9.431 | 73.712 |
| 12002 | conditional | 3 | introduction_3 | periodic | 11.465 | 11.465 | 10.998 | 15.702 | 58.289 |
| 12002 | conditional | 4 | clean | periodic | 9.912 | 9.912 | 8.935 | 11.705 | 61.597 |
| 12002 | conditional | 5 | noise | periodic | 10.373 | 15.634 | 9.921 | 16.800 | 59.998 |
| 12002 | conditional | 6 | recovery | periodic | 9.155 | 9.155 | 8.586 | 23.077 | 61.554 |
| 12002 | conditional | 7 | return | periodic | 9.956 | 9.956 | 10.000 | 10.406 | 72.919 |
| 12002 | conditional | 8 | revision | periodic | 10.969 | 10.969 | 10.933 | 11.495 | 61.359 |
| 12002 | conditional | 9 | clean | periodic | 9.614 | 9.614 | 8.781 | 10.392 | 63.324 |
| 12002 | conditional | 10 | noise | periodic | 11.853 | 17.538 | 10.671 | 11.537 | 61.615 |
| 12002 | conditional | 11 | recovery | periodic | 9.121 | 9.121 | 8.567 | 11.370 | 62.796 |
| 12002 | conditional | 12 | return | periodic | 9.265 | 9.265 | 9.244 | 10.657 | 74.084 |
| 12002 | conditional | 13 | revision | periodic | 10.222 | 10.222 | 10.288 | 12.023 | 63.159 |
| 12002 | conditional | 14 | clean | periodic | 9.071 | 9.071 | 8.638 | 10.491 | 63.715 |
| 12002 | conditional | 15 | noise | periodic | 11.010 | 16.009 | 10.257 | 11.808 | 61.526 |
| 12002 | conditional | 16 | recovery | periodic | 9.366 | 9.366 | 8.647 | 11.267 | 62.814 |
| 12002 | conditional | 17 | return | periodic | 9.189 | 9.189 | 9.200 | 10.602 | 73.511 |
| 12002 | conditional | 18 | revision | periodic | 9.815 | 9.815 | 9.622 | 16.610 | 63.638 |
| 12002 | conditional | 1 | introduction_1 | single | 7.278 | 7.278 | 6.546 | 8.700 | 71.463 |
| 12002 | conditional | 2 | introduction_2 | single | 7.708 | 7.708 | 7.308 | 9.431 | 73.694 |
| 12002 | conditional | 3 | introduction_3 | single | 11.586 | 11.586 | 11.172 | 15.702 | 57.941 |
| 12002 | conditional | 4 | clean | single | 9.848 | 9.848 | 8.927 | 13.911 | 61.823 |
| 12002 | conditional | 5 | noise | single | 10.139 | 15.588 | 9.581 | 18.038 | 60.675 |
| 12002 | conditional | 6 | recovery | single | 9.171 | 9.171 | 8.615 | 23.077 | 61.505 |
| 12002 | conditional | 7 | return | single | 10.207 | 10.207 | 9.714 | 10.406 | 73.364 |
| 12002 | conditional | 8 | revision | single | 10.989 | 10.989 | 10.952 | 11.495 | 61.316 |
| 12002 | conditional | 9 | clean | single | 9.751 | 9.751 | 8.744 | 10.053 | 63.116 |
| 12002 | conditional | 10 | noise | single | 11.648 | 17.241 | 10.496 | 10.645 | 61.630 |
| 12002 | conditional | 11 | recovery | single | 9.281 | 9.281 | 8.535 | 11.370 | 62.543 |
| 12002 | conditional | 12 | return | single | 9.483 | 9.483 | 10.150 | 10.657 | 73.706 |
| 12002 | conditional | 13 | revision | single | 10.262 | 10.262 | 10.499 | 13.448 | 62.598 |
| 12002 | conditional | 14 | clean | single | 9.113 | 9.113 | 8.582 | 10.491 | 63.919 |
| 12002 | conditional | 15 | noise | single | 10.874 | 16.039 | 10.075 | 11.808 | 61.783 |
| 12002 | conditional | 16 | recovery | single | 9.246 | 9.246 | 8.660 | 11.267 | 62.851 |
| 12002 | conditional | 17 | return | single | 9.352 | 9.352 | 9.150 | 10.975 | 73.621 |
| 12002 | conditional | 18 | revision | single | 9.739 | 9.739 | 9.678 | 16.610 | 63.635 |
| 12002 | conditional | 1 | introduction_1 | sustained | 6.841 | 6.841 | 6.372 | 10.504 | 71.439 |
| 12002 | conditional | 2 | introduction_2 | sustained | 7.769 | 7.769 | 7.544 | 9.227 | 74.060 |
| 12002 | conditional | 3 | introduction_3 | sustained | 11.453 | 11.453 | 11.152 | 10.828 | 58.154 |
| 12002 | conditional | 4 | clean | sustained | 9.602 | 9.602 | 8.815 | 16.748 | 61.926 |
| 12002 | conditional | 5 | noise | sustained | 10.054 | 15.549 | 9.225 | 16.688 | 62.195 |
| 12002 | conditional | 6 | recovery | sustained | 9.200 | 9.200 | 8.375 | 17.830 | 61.234 |
| 12002 | conditional | 7 | return | sustained | 10.093 | 10.093 | 9.515 | 10.136 | 73.480 |
| 12002 | conditional | 8 | revision | sustained | 10.938 | 10.938 | 11.163 | 16.168 | 61.487 |
| 12002 | conditional | 9 | clean | sustained | 9.312 | 9.312 | 8.321 | 10.053 | 63.641 |
| 12002 | conditional | 10 | noise | sustained | 12.129 | 17.593 | 11.025 | 11.867 | 61.560 |
| 12002 | conditional | 11 | recovery | sustained | 9.583 | 9.583 | 8.545 | 11.759 | 63.196 |
| 12002 | conditional | 12 | return | sustained | 9.584 | 9.584 | 9.415 | 10.630 | 73.557 |
| 12002 | conditional | 13 | revision | sustained | 10.141 | 10.141 | 10.400 | 11.348 | 62.225 |
| 12002 | conditional | 14 | clean | sustained | 9.033 | 9.033 | 8.859 | 9.977 | 63.736 |
| 12002 | conditional | 15 | noise | sustained | 9.973 | 15.710 | 8.812 | 12.270 | 63.217 |
| 12002 | conditional | 16 | recovery | sustained | 9.402 | 9.402 | 8.905 | 10.851 | 62.460 |
| 12002 | conditional | 17 | return | sustained | 9.732 | 9.732 | 9.728 | 10.638 | 73.175 |
| 12002 | conditional | 18 | revision | sustained | 9.782 | 9.782 | 9.601 | 10.299 | 63.559 |
| 12002 | recurrent | 1 | introduction_1 | immediate | 7.183 | 7.183 | 6.651 | 23.114 | 71.164 |
| 12002 | recurrent | 2 | introduction_2 | immediate | 7.793 | 7.793 | 7.376 | 18.537 | 74.283 |
| 12002 | recurrent | 3 | introduction_3 | immediate | 11.449 | 11.449 | 11.023 | 22.724 | 58.191 |
| 12002 | recurrent | 4 | clean | immediate | 9.594 | 9.594 | 9.050 | 22.922 | 62.073 |
| 12002 | recurrent | 5 | noise | immediate | 11.323 | 16.315 | 10.704 | 22.578 | 60.129 |
| 12002 | recurrent | 6 | recovery | immediate | 9.302 | 9.302 | 8.868 | 23.478 | 61.948 |
| 12002 | recurrent | 7 | return | immediate | 10.980 | 10.980 | 10.877 | 15.449 | 71.136 |
| 12002 | recurrent | 8 | revision | immediate | 10.788 | 10.788 | 10.983 | 19.383 | 62.393 |
| 12002 | recurrent | 9 | clean | immediate | 9.415 | 9.415 | 8.662 | 22.166 | 64.215 |
| 12002 | recurrent | 10 | noise | immediate | 11.414 | 15.357 | 11.227 | 22.070 | 61.041 |
| 12002 | recurrent | 11 | recovery | immediate | 9.403 | 9.403 | 8.594 | 21.583 | 64.508 |
| 12002 | recurrent | 12 | return | immediate | 10.966 | 10.966 | 9.892 | 12.735 | 72.742 |
| 12002 | recurrent | 13 | revision | immediate | 10.239 | 10.239 | 10.300 | 20.601 | 62.943 |
| 12002 | recurrent | 14 | clean | immediate | 9.212 | 9.212 | 9.164 | 21.844 | 63.611 |
| 12002 | recurrent | 15 | noise | immediate | 11.249 | 16.259 | 10.751 | 18.582 | 61.072 |
| 12002 | recurrent | 16 | recovery | immediate | 9.223 | 9.223 | 8.547 | 21.584 | 62.668 |
| 12002 | recurrent | 17 | return | immediate | 10.430 | 10.430 | 10.917 | 11.817 | 71.985 |
| 12002 | recurrent | 18 | revision | immediate | 9.611 | 9.611 | 10.245 | 16.549 | 62.506 |
| 12002 | recurrent | 1 | introduction_1 | periodic | 7.265 | 7.265 | 6.628 | 22.903 | 71.109 |
| 12002 | recurrent | 2 | introduction_2 | periodic | 7.808 | 7.808 | 7.453 | 20.821 | 74.164 |
| 12002 | recurrent | 3 | introduction_3 | periodic | 11.659 | 11.659 | 11.277 | 21.029 | 57.993 |
| 12002 | recurrent | 4 | clean | periodic | 9.945 | 9.945 | 9.152 | 23.763 | 62.015 |
| 12002 | recurrent | 5 | noise | periodic | 10.992 | 16.075 | 10.550 | 21.428 | 60.275 |
| 12002 | recurrent | 6 | recovery | periodic | 9.377 | 9.377 | 8.863 | 23.730 | 61.807 |
| 12002 | recurrent | 7 | return | periodic | 12.340 | 12.340 | 12.152 | 15.350 | 70.270 |
| 12002 | recurrent | 8 | revision | periodic | 11.345 | 11.345 | 11.448 | 20.199 | 61.920 |
| 12002 | recurrent | 9 | clean | periodic | 9.504 | 9.504 | 8.726 | 21.850 | 64.255 |
| 12002 | recurrent | 10 | noise | periodic | 11.263 | 17.143 | 10.874 | 23.048 | 61.130 |
| 12002 | recurrent | 11 | recovery | periodic | 9.538 | 9.538 | 8.634 | 23.304 | 64.496 |
| 12002 | recurrent | 12 | return | periodic | 12.384 | 12.384 | 10.890 | 12.976 | 72.159 |
| 12002 | recurrent | 13 | revision | periodic | 10.787 | 10.787 | 10.725 | 21.383 | 62.744 |
| 12002 | recurrent | 14 | clean | periodic | 9.277 | 9.277 | 8.910 | 21.647 | 63.486 |
| 12002 | recurrent | 15 | noise | periodic | 11.245 | 16.297 | 10.844 | 18.078 | 61.127 |
| 12002 | recurrent | 16 | recovery | periodic | 9.219 | 9.219 | 8.603 | 20.829 | 62.646 |
| 12002 | recurrent | 17 | return | periodic | 12.073 | 12.073 | 11.701 | 16.096 | 71.432 |
| 12002 | recurrent | 18 | revision | periodic | 10.326 | 10.326 | 10.196 | 20.737 | 62.598 |
| 12002 | recurrent | 1 | introduction_1 | single | 7.258 | 7.258 | 6.506 | 22.903 | 70.944 |
| 12002 | recurrent | 2 | introduction_2 | single | 7.952 | 7.952 | 7.515 | 20.202 | 74.136 |
| 12002 | recurrent | 3 | introduction_3 | single | 11.691 | 11.691 | 11.311 | 20.910 | 57.416 |
| 12002 | recurrent | 4 | clean | single | 9.839 | 9.839 | 9.123 | 23.196 | 62.308 |
| 12002 | recurrent | 5 | noise | single | 10.796 | 15.949 | 10.235 | 21.428 | 61.050 |
| 12002 | recurrent | 6 | recovery | single | 9.376 | 9.376 | 8.790 | 22.998 | 61.703 |
| 12002 | recurrent | 7 | return | single | 12.252 | 12.252 | 12.326 | 21.045 | 70.102 |
| 12002 | recurrent | 8 | revision | single | 11.467 | 11.467 | 11.588 | 20.154 | 61.679 |
| 12002 | recurrent | 9 | clean | single | 9.595 | 9.595 | 8.713 | 21.850 | 64.200 |
| 12002 | recurrent | 10 | noise | single | 11.450 | 17.249 | 10.892 | 23.048 | 61.423 |
| 12002 | recurrent | 11 | recovery | single | 9.385 | 9.385 | 8.723 | 22.493 | 64.703 |
| 12002 | recurrent | 12 | return | single | 12.939 | 12.939 | 11.678 | 12.976 | 71.582 |
| 12002 | recurrent | 13 | revision | single | 10.773 | 10.773 | 10.785 | 18.202 | 62.762 |
| 12002 | recurrent | 14 | clean | single | 9.260 | 9.260 | 8.952 | 20.168 | 63.544 |
| 12002 | recurrent | 15 | noise | single | 10.885 | 16.114 | 10.780 | 17.704 | 61.404 |
| 12002 | recurrent | 16 | recovery | single | 9.099 | 9.099 | 8.610 | 20.829 | 62.671 |
| 12002 | recurrent | 17 | return | single | 12.362 | 12.362 | 11.824 | 16.096 | 71.017 |
| 12002 | recurrent | 18 | revision | single | 10.223 | 10.223 | 11.060 | 18.871 | 61.713 |
| 12002 | recurrent | 1 | introduction_1 | sustained | 6.929 | 6.929 | 6.546 | 20.732 | 71.097 |
| 12002 | recurrent | 2 | introduction_2 | sustained | 7.550 | 7.550 | 7.361 | 22.046 | 74.500 |
| 12002 | recurrent | 3 | introduction_3 | sustained | 11.957 | 11.957 | 11.489 | 20.910 | 57.141 |
| 12002 | recurrent | 4 | clean | sustained | 9.721 | 9.721 | 9.114 | 23.196 | 61.880 |
| 12002 | recurrent | 5 | noise | sustained | 8.782 | 14.805 | 8.184 | 22.922 | 62.366 |
| 12002 | recurrent | 6 | recovery | sustained | 9.034 | 9.034 | 8.500 | 24.418 | 62.747 |
| 12002 | recurrent | 7 | return | sustained | 12.422 | 12.422 | 12.443 | 16.839 | 69.653 |
| 12002 | recurrent | 8 | revision | sustained | 11.467 | 11.467 | 11.530 | 18.740 | 61.768 |
| 12002 | recurrent | 9 | clean | sustained | 9.248 | 9.248 | 8.496 | 21.752 | 63.940 |
| 12002 | recurrent | 10 | noise | sustained | 11.004 | 16.806 | 10.591 | 20.761 | 62.277 |
| 12002 | recurrent | 11 | recovery | sustained | 9.393 | 9.393 | 8.712 | 23.298 | 63.928 |
| 12002 | recurrent | 12 | return | sustained | 12.759 | 12.759 | 10.882 | 12.976 | 72.198 |
| 12002 | recurrent | 13 | revision | sustained | 11.326 | 11.326 | 11.007 | 18.048 | 62.003 |
| 12002 | recurrent | 14 | clean | sustained | 9.612 | 9.612 | 8.879 | 21.927 | 64.059 |
| 12002 | recurrent | 15 | noise | sustained | 9.701 | 15.483 | 9.542 | 17.724 | 62.689 |
| 12002 | recurrent | 16 | recovery | sustained | 9.381 | 9.381 | 8.935 | 22.597 | 62.366 |
| 12002 | recurrent | 17 | return | sustained | 12.246 | 12.246 | 12.359 | 14.536 | 71.021 |
| 12002 | recurrent | 18 | revision | sustained | 10.629 | 10.629 | 10.865 | 18.871 | 62.274 |
| 12003 | conditional | 1 | introduction_1 | immediate | 7.230 | 7.230 | 7.594 | 8.690 | 72.964 |
| 12003 | conditional | 2 | introduction_2 | immediate | 10.442 | 10.442 | 11.204 | 16.361 | 60.358 |
| 12003 | conditional | 3 | introduction_3 | immediate | 9.298 | 9.298 | 9.656 | 9.671 | 65.048 |
| 12003 | conditional | 4 | clean | immediate | 9.142 | 9.142 | 9.237 | 10.835 | 65.823 |
| 12003 | conditional | 5 | noise | immediate | 10.057 | 15.478 | 10.352 | 11.557 | 64.001 |
| 12003 | conditional | 6 | recovery | immediate | 9.050 | 9.050 | 9.034 | 15.767 | 65.833 |
| 12003 | conditional | 7 | return | immediate | 8.553 | 8.553 | 8.738 | 10.537 | 72.964 |
| 12003 | conditional | 8 | revision | immediate | 9.866 | 9.866 | 10.369 | 18.217 | 64.862 |
| 12003 | conditional | 9 | clean | immediate | 9.004 | 9.004 | 9.280 | 8.150 | 66.400 |
| 12003 | conditional | 10 | noise | immediate | 11.592 | 15.487 | 11.631 | 9.828 | 63.895 |
| 12003 | conditional | 11 | recovery | immediate | 9.117 | 9.117 | 8.955 | 10.308 | 66.608 |
| 12003 | conditional | 12 | return | immediate | 9.015 | 9.015 | 8.888 | 10.600 | 73.572 |
| 12003 | conditional | 13 | revision | immediate | 9.785 | 9.785 | 10.733 | 10.872 | 64.584 |
| 12003 | conditional | 14 | clean | immediate | 8.818 | 8.818 | 9.363 | 8.642 | 66.473 |
| 12003 | conditional | 15 | noise | immediate | 10.993 | 16.449 | 11.330 | 11.096 | 63.025 |
| 12003 | conditional | 16 | recovery | immediate | 9.379 | 9.379 | 9.243 | 8.941 | 65.457 |
| 12003 | conditional | 17 | return | immediate | 8.728 | 8.728 | 9.102 | 10.978 | 73.108 |
| 12003 | conditional | 18 | revision | immediate | 10.016 | 10.016 | 10.969 | 10.671 | 65.073 |
| 12003 | conditional | 1 | introduction_1 | periodic | 7.358 | 7.358 | 7.631 | 8.821 | 72.937 |
| 12003 | conditional | 2 | introduction_2 | periodic | 11.059 | 11.059 | 11.473 | 9.612 | 59.991 |
| 12003 | conditional | 3 | introduction_3 | periodic | 9.352 | 9.352 | 9.666 | 9.857 | 64.993 |
| 12003 | conditional | 4 | clean | periodic | 8.925 | 8.925 | 9.256 | 12.479 | 65.784 |
| 12003 | conditional | 5 | noise | periodic | 9.662 | 15.255 | 10.294 | 17.750 | 64.111 |
| 12003 | conditional | 6 | recovery | periodic | 9.236 | 9.236 | 9.094 | 16.382 | 65.720 |
| 12003 | conditional | 7 | return | periodic | 8.963 | 8.963 | 8.920 | 10.674 | 72.815 |
| 12003 | conditional | 8 | revision | periodic | 9.835 | 9.835 | 10.447 | 15.432 | 64.850 |
| 12003 | conditional | 9 | clean | periodic | 8.994 | 8.994 | 9.288 | 8.638 | 66.418 |
| 12003 | conditional | 10 | noise | periodic | 11.897 | 17.299 | 11.624 | 10.352 | 63.849 |
| 12003 | conditional | 11 | recovery | periodic | 9.233 | 9.233 | 8.981 | 10.332 | 66.617 |
| 12003 | conditional | 12 | return | periodic | 9.418 | 9.418 | 9.047 | 10.606 | 73.505 |
| 12003 | conditional | 13 | revision | periodic | 9.989 | 9.989 | 10.961 | 10.778 | 64.417 |
| 12003 | conditional | 14 | clean | periodic | 8.570 | 8.570 | 9.396 | 13.369 | 66.400 |
| 12003 | conditional | 15 | noise | periodic | 11.328 | 16.733 | 12.027 | 10.825 | 62.830 |
| 12003 | conditional | 16 | recovery | periodic | 9.405 | 9.405 | 9.340 | 9.317 | 65.387 |
| 12003 | conditional | 17 | return | periodic | 9.535 | 9.535 | 9.233 | 11.136 | 73.077 |
| 12003 | conditional | 18 | revision | periodic | 10.146 | 10.146 | 11.045 | 10.929 | 65.012 |
| 12003 | conditional | 1 | introduction_1 | single | 7.398 | 7.398 | 7.617 | 8.821 | 72.900 |
| 12003 | conditional | 2 | introduction_2 | single | 11.109 | 11.109 | 11.614 | 9.612 | 59.814 |
| 12003 | conditional | 3 | introduction_3 | single | 9.547 | 9.547 | 9.878 | 10.233 | 64.941 |
| 12003 | conditional | 4 | clean | single | 9.117 | 9.117 | 9.332 | 12.479 | 66.031 |
| 12003 | conditional | 5 | noise | single | 9.485 | 15.102 | 10.064 | 17.750 | 64.746 |
| 12003 | conditional | 6 | recovery | single | 9.140 | 9.140 | 9.076 | 16.382 | 65.390 |
| 12003 | conditional | 7 | return | single | 8.806 | 8.806 | 8.891 | 10.398 | 72.784 |
| 12003 | conditional | 8 | revision | single | 10.013 | 10.013 | 10.594 | 14.407 | 64.713 |
| 12003 | conditional | 9 | clean | single | 9.007 | 9.007 | 9.358 | 8.638 | 66.681 |
| 12003 | conditional | 10 | noise | single | 12.236 | 17.606 | 12.709 | 10.352 | 63.525 |
| 12003 | conditional | 11 | recovery | single | 9.409 | 9.409 | 8.948 | 10.332 | 67.099 |
| 12003 | conditional | 12 | return | single | 9.441 | 9.441 | 9.137 | 10.606 | 73.413 |
| 12003 | conditional | 13 | revision | single | 10.032 | 10.032 | 10.910 | 10.424 | 64.362 |
| 12003 | conditional | 14 | clean | single | 8.651 | 8.651 | 9.249 | 9.049 | 66.287 |
| 12003 | conditional | 15 | noise | single | 11.324 | 16.855 | 11.921 | 11.051 | 63.055 |
| 12003 | conditional | 16 | recovery | single | 9.476 | 9.476 | 9.452 | 9.317 | 65.158 |
| 12003 | conditional | 17 | return | single | 9.339 | 9.339 | 9.296 | 11.311 | 73.279 |
| 12003 | conditional | 18 | revision | single | 10.215 | 10.215 | 10.948 | 11.452 | 65.109 |
| 12003 | conditional | 1 | introduction_1 | sustained | 6.957 | 6.957 | 7.367 | 8.821 | 73.114 |
| 12003 | conditional | 2 | introduction_2 | sustained | 11.070 | 11.070 | 11.234 | 10.113 | 59.433 |
| 12003 | conditional | 3 | introduction_3 | sustained | 9.507 | 9.507 | 9.738 | 10.233 | 64.432 |
| 12003 | conditional | 4 | clean | sustained | 8.851 | 8.851 | 8.764 | 17.541 | 66.660 |
| 12003 | conditional | 5 | noise | sustained | 9.792 | 15.457 | 10.385 | 11.201 | 64.426 |
| 12003 | conditional | 6 | recovery | sustained | 9.316 | 9.316 | 9.463 | 15.709 | 65.521 |
| 12003 | conditional | 7 | return | sustained | 10.062 | 10.062 | 10.025 | 10.398 | 71.960 |
| 12003 | conditional | 8 | revision | sustained | 10.159 | 10.159 | 10.710 | 15.432 | 64.529 |
| 12003 | conditional | 9 | clean | sustained | 9.221 | 9.221 | 9.509 | 8.638 | 66.272 |
| 12003 | conditional | 10 | noise | sustained | 12.413 | 17.616 | 11.745 | 9.959 | 63.458 |
| 12003 | conditional | 11 | recovery | sustained | 9.197 | 9.197 | 9.099 | 9.329 | 66.684 |
| 12003 | conditional | 12 | return | sustained | 9.603 | 9.603 | 9.968 | 11.117 | 73.010 |
| 12003 | conditional | 13 | revision | sustained | 10.263 | 10.263 | 10.936 | 11.080 | 65.598 |
| 12003 | conditional | 14 | clean | sustained | 8.859 | 8.859 | 9.352 | 9.049 | 66.217 |
| 12003 | conditional | 15 | noise | sustained | 9.983 | 16.257 | 10.171 | 9.864 | 64.749 |
| 12003 | conditional | 16 | recovery | sustained | 9.554 | 9.554 | 9.386 | 9.403 | 65.805 |
| 12003 | conditional | 17 | return | sustained | 9.442 | 9.442 | 9.222 | 10.683 | 72.839 |
| 12003 | conditional | 18 | revision | sustained | 10.159 | 10.159 | 11.749 | 10.998 | 63.742 |
| 12003 | recurrent | 1 | introduction_1 | immediate | 7.846 | 7.846 | 8.125 | 23.759 | 71.930 |
| 12003 | recurrent | 2 | introduction_2 | immediate | 10.639 | 10.639 | 10.825 | 23.932 | 61.224 |
| 12003 | recurrent | 3 | introduction_3 | immediate | 9.002 | 9.002 | 9.581 | 23.830 | 65.060 |
| 12003 | recurrent | 4 | clean | immediate | 8.972 | 8.972 | 9.343 | 22.996 | 65.802 |
| 12003 | recurrent | 5 | noise | immediate | 10.600 | 15.978 | 10.942 | 23.978 | 63.931 |
| 12003 | recurrent | 6 | recovery | immediate | 9.005 | 9.005 | 9.203 | 23.571 | 65.900 |
| 12003 | recurrent | 7 | return | immediate | 10.022 | 10.022 | 9.315 | 24.277 | 71.942 |
| 12003 | recurrent | 8 | revision | immediate | 10.181 | 10.181 | 11.097 | 23.322 | 64.777 |
| 12003 | recurrent | 9 | clean | immediate | 9.039 | 9.039 | 9.305 | 22.296 | 66.455 |
| 12003 | recurrent | 10 | noise | immediate | 11.873 | 15.765 | 13.078 | 24.190 | 63.000 |
| 12003 | recurrent | 11 | recovery | immediate | 9.377 | 9.377 | 9.323 | 23.405 | 66.116 |
| 12003 | recurrent | 12 | return | immediate | 10.912 | 10.912 | 10.603 | 18.192 | 71.460 |
| 12003 | recurrent | 13 | revision | immediate | 9.965 | 9.965 | 10.846 | 18.881 | 64.792 |
| 12003 | recurrent | 14 | clean | immediate | 8.580 | 8.580 | 9.413 | 23.600 | 66.727 |
| 12003 | recurrent | 15 | noise | immediate | 11.437 | 16.890 | 11.592 | 24.860 | 63.309 |
| 12003 | recurrent | 16 | recovery | immediate | 9.366 | 9.366 | 9.446 | 24.099 | 65.927 |
| 12003 | recurrent | 17 | return | immediate | 10.177 | 10.177 | 9.898 | 18.274 | 71.283 |
| 12003 | recurrent | 18 | revision | immediate | 10.078 | 10.078 | 10.288 | 21.484 | 65.347 |
| 12003 | recurrent | 1 | introduction_1 | periodic | 8.018 | 8.018 | 8.295 | 21.613 | 72.021 |
| 12003 | recurrent | 2 | introduction_2 | periodic | 10.967 | 10.967 | 10.957 | 23.082 | 61.108 |
| 12003 | recurrent | 3 | introduction_3 | periodic | 9.079 | 9.079 | 9.677 | 22.470 | 65.149 |
| 12003 | recurrent | 4 | clean | periodic | 8.935 | 8.935 | 9.247 | 23.138 | 65.781 |
| 12003 | recurrent | 5 | noise | periodic | 10.370 | 15.739 | 10.699 | 24.504 | 64.459 |
| 12003 | recurrent | 6 | recovery | periodic | 9.451 | 9.451 | 9.277 | 24.383 | 66.003 |
| 12003 | recurrent | 7 | return | periodic | 11.105 | 11.105 | 10.632 | 24.628 | 71.497 |
| 12003 | recurrent | 8 | revision | periodic | 11.000 | 11.000 | 11.378 | 22.857 | 64.261 |
| 12003 | recurrent | 9 | clean | periodic | 9.065 | 9.065 | 9.441 | 24.082 | 66.254 |
| 12003 | recurrent | 10 | noise | periodic | 11.673 | 17.613 | 12.174 | 23.447 | 63.535 |
| 12003 | recurrent | 11 | recovery | periodic | 9.314 | 9.314 | 9.430 | 22.921 | 65.784 |
| 12003 | recurrent | 12 | return | periodic | 12.238 | 12.238 | 11.548 | 18.001 | 71.124 |
| 12003 | recurrent | 13 | revision | periodic | 10.468 | 10.468 | 11.127 | 20.689 | 64.697 |
| 12003 | recurrent | 14 | clean | periodic | 8.597 | 8.597 | 9.411 | 23.601 | 66.873 |
| 12003 | recurrent | 15 | noise | periodic | 11.454 | 17.046 | 11.883 | 24.637 | 63.483 |
| 12003 | recurrent | 16 | recovery | periodic | 9.250 | 9.250 | 9.497 | 24.652 | 65.720 |
| 12003 | recurrent | 17 | return | periodic | 11.076 | 11.076 | 10.974 | 19.180 | 70.190 |
| 12003 | recurrent | 18 | revision | periodic | 10.568 | 10.568 | 10.507 | 21.162 | 65.268 |
| 12003 | recurrent | 1 | introduction_1 | single | 8.230 | 8.230 | 8.473 | 22.578 | 71.457 |
| 12003 | recurrent | 2 | introduction_2 | single | 11.003 | 11.003 | 10.987 | 23.062 | 60.941 |
| 12003 | recurrent | 3 | introduction_3 | single | 9.026 | 9.026 | 9.469 | 23.695 | 65.359 |
| 12003 | recurrent | 4 | clean | single | 8.999 | 8.999 | 9.227 | 23.925 | 65.842 |
| 12003 | recurrent | 5 | noise | single | 10.230 | 15.880 | 10.343 | 24.771 | 64.731 |
| 12003 | recurrent | 6 | recovery | single | 9.253 | 9.253 | 9.144 | 24.383 | 66.068 |
| 12003 | recurrent | 7 | return | single | 11.413 | 11.413 | 10.576 | 23.965 | 71.478 |
| 12003 | recurrent | 8 | revision | single | 11.481 | 11.481 | 11.296 | 18.255 | 64.563 |
| 12003 | recurrent | 9 | clean | single | 9.060 | 9.060 | 9.471 | 24.082 | 66.104 |
| 12003 | recurrent | 10 | noise | single | 11.074 | 17.052 | 11.407 | 24.549 | 64.172 |
| 12003 | recurrent | 11 | recovery | single | 9.024 | 9.024 | 9.627 | 22.921 | 65.994 |
| 12003 | recurrent | 12 | return | single | 12.143 | 12.143 | 11.543 | 21.498 | 70.853 |
| 12003 | recurrent | 13 | revision | single | 10.730 | 10.730 | 11.672 | 20.689 | 64.282 |
| 12003 | recurrent | 14 | clean | single | 8.563 | 8.563 | 9.414 | 23.601 | 66.733 |
| 12003 | recurrent | 15 | noise | single | 11.087 | 16.677 | 11.395 | 24.637 | 64.056 |
| 12003 | recurrent | 16 | recovery | single | 9.456 | 9.456 | 9.594 | 24.652 | 65.714 |
| 12003 | recurrent | 17 | return | single | 11.939 | 11.939 | 11.507 | 19.180 | 69.983 |
| 12003 | recurrent | 18 | revision | single | 10.735 | 10.735 | 10.883 | 21.162 | 64.993 |
| 12003 | recurrent | 1 | introduction_1 | sustained | 8.278 | 8.278 | 8.630 | 21.613 | 71.338 |
| 12003 | recurrent | 2 | introduction_2 | sustained | 10.888 | 10.888 | 11.014 | 23.865 | 60.944 |
| 12003 | recurrent | 3 | introduction_3 | sustained | 9.030 | 9.030 | 9.427 | 24.433 | 65.210 |
| 12003 | recurrent | 4 | clean | sustained | 9.077 | 9.077 | 9.303 | 23.068 | 65.741 |
| 12003 | recurrent | 5 | noise | sustained | 9.302 | 15.260 | 9.866 | 23.103 | 64.865 |
| 12003 | recurrent | 6 | recovery | sustained | 9.249 | 9.249 | 9.527 | 25.129 | 65.793 |
| 12003 | recurrent | 7 | return | sustained | 11.576 | 11.576 | 10.785 | 19.456 | 71.014 |
| 12003 | recurrent | 8 | revision | sustained | 11.579 | 11.579 | 11.795 | 19.256 | 64.368 |
| 12003 | recurrent | 9 | clean | sustained | 9.139 | 9.139 | 9.556 | 24.575 | 66.260 |
| 12003 | recurrent | 10 | noise | sustained | 11.267 | 16.918 | 11.373 | 24.549 | 64.230 |
| 12003 | recurrent | 11 | recovery | sustained | 9.376 | 9.376 | 9.426 | 23.508 | 65.576 |
| 12003 | recurrent | 12 | return | sustained | 12.254 | 12.254 | 12.518 | 17.994 | 70.212 |
| 12003 | recurrent | 13 | revision | sustained | 10.810 | 10.810 | 11.987 | 22.199 | 64.139 |
| 12003 | recurrent | 14 | clean | sustained | 8.507 | 8.507 | 9.410 | 23.601 | 67.502 |
| 12003 | recurrent | 15 | noise | sustained | 9.469 | 15.828 | 10.113 | 21.861 | 66.187 |
| 12003 | recurrent | 16 | recovery | sustained | 9.597 | 9.597 | 9.672 | 23.630 | 65.793 |
| 12003 | recurrent | 17 | return | sustained | 11.229 | 11.229 | 11.049 | 17.960 | 70.984 |
| 12003 | recurrent | 18 | revision | sustained | 10.861 | 10.861 | 11.542 | 21.503 | 63.599 |
| 12004 | conditional | 1 | introduction_1 | immediate | 9.762 | 9.762 | 10.311 | 13.242 | 57.578 |
| 12004 | conditional | 2 | introduction_2 | immediate | 8.259 | 8.259 | 8.227 | 10.210 | 61.340 |
| 12004 | conditional | 3 | introduction_3 | immediate | 8.670 | 8.670 | 9.135 | 12.263 | 61.572 |
| 12004 | conditional | 4 | clean | immediate | 8.472 | 8.472 | 8.273 | 11.908 | 62.708 |
| 12004 | conditional | 5 | noise | immediate | 10.109 | 15.553 | 9.954 | 13.571 | 59.888 |
| 12004 | conditional | 6 | recovery | immediate | 8.696 | 8.696 | 8.604 | 13.259 | 61.224 |
| 12004 | conditional | 7 | return | immediate | 9.673 | 9.673 | 9.282 | 11.224 | 70.853 |
| 12004 | conditional | 8 | revision | immediate | 9.954 | 9.954 | 9.670 | 11.647 | 62.515 |
| 12004 | conditional | 9 | clean | immediate | 8.739 | 8.739 | 9.088 | 11.736 | 61.548 |
| 12004 | conditional | 10 | noise | immediate | 10.683 | 14.831 | 10.918 | 14.803 | 59.091 |
| 12004 | conditional | 11 | recovery | immediate | 8.980 | 8.980 | 8.858 | 12.788 | 61.646 |
| 12004 | conditional | 12 | return | immediate | 9.056 | 9.056 | 9.008 | 16.186 | 70.984 |
| 12004 | conditional | 13 | revision | immediate | 10.133 | 10.133 | 10.009 | 11.526 | 62.958 |
| 12004 | conditional | 14 | clean | immediate | 8.553 | 8.553 | 8.777 | 9.410 | 61.816 |
| 12004 | conditional | 15 | noise | immediate | 10.764 | 15.934 | 10.607 | 16.101 | 59.476 |
| 12004 | conditional | 16 | recovery | immediate | 9.234 | 9.234 | 9.147 | 10.790 | 60.864 |
| 12004 | conditional | 17 | return | immediate | 9.290 | 9.290 | 9.176 | 17.535 | 70.764 |
| 12004 | conditional | 18 | revision | immediate | 9.770 | 9.770 | 10.180 | 12.866 | 62.357 |
| 12004 | conditional | 1 | introduction_1 | periodic | 10.645 | 10.645 | 11.163 | 9.132 | 56.979 |
| 12004 | conditional | 2 | introduction_2 | periodic | 8.578 | 8.578 | 8.254 | 10.014 | 61.292 |
| 12004 | conditional | 3 | introduction_3 | periodic | 8.947 | 8.947 | 9.154 | 12.285 | 61.560 |
| 12004 | conditional | 4 | clean | periodic | 8.338 | 8.338 | 8.298 | 11.826 | 62.665 |
| 12004 | conditional | 5 | noise | periodic | 10.397 | 15.851 | 9.897 | 15.919 | 59.985 |
| 12004 | conditional | 6 | recovery | periodic | 9.246 | 9.246 | 9.138 | 12.570 | 60.727 |
| 12004 | conditional | 7 | return | periodic | 10.579 | 10.579 | 10.342 | 11.079 | 70.050 |
| 12004 | conditional | 8 | revision | periodic | 9.952 | 9.952 | 9.745 | 12.295 | 62.454 |
| 12004 | conditional | 9 | clean | periodic | 9.029 | 9.029 | 9.084 | 11.113 | 61.438 |
| 12004 | conditional | 10 | noise | periodic | 11.369 | 16.923 | 11.536 | 12.097 | 58.914 |
| 12004 | conditional | 11 | recovery | periodic | 8.728 | 8.728 | 8.889 | 12.590 | 61.511 |
| 12004 | conditional | 12 | return | periodic | 9.468 | 9.468 | 9.478 | 17.790 | 70.709 |
| 12004 | conditional | 13 | revision | periodic | 9.855 | 9.855 | 10.007 | 12.374 | 62.964 |
| 12004 | conditional | 14 | clean | periodic | 8.541 | 8.541 | 8.805 | 9.774 | 61.783 |
| 12004 | conditional | 15 | noise | periodic | 10.480 | 15.752 | 10.557 | 11.336 | 59.604 |
| 12004 | conditional | 16 | recovery | periodic | 9.258 | 9.258 | 9.188 | 11.165 | 60.730 |
| 12004 | conditional | 17 | return | periodic | 9.974 | 9.974 | 9.976 | 17.206 | 70.053 |
| 12004 | conditional | 18 | revision | periodic | 10.141 | 10.141 | 10.219 | 11.802 | 62.213 |
| 12004 | conditional | 1 | introduction_1 | single | 10.366 | 10.366 | 10.421 | 9.214 | 57.562 |
| 12004 | conditional | 2 | introduction_2 | single | 8.427 | 8.427 | 8.188 | 10.014 | 61.307 |
| 12004 | conditional | 3 | introduction_3 | single | 8.752 | 8.752 | 9.132 | 12.285 | 61.774 |
| 12004 | conditional | 4 | clean | single | 8.466 | 8.466 | 8.427 | 11.826 | 62.366 |
| 12004 | conditional | 5 | noise | single | 10.346 | 15.865 | 9.855 | 15.919 | 59.656 |
| 12004 | conditional | 6 | recovery | single | 9.146 | 9.146 | 9.170 | 12.570 | 60.916 |
| 12004 | conditional | 7 | return | single | 11.022 | 11.022 | 10.415 | 11.509 | 69.907 |
| 12004 | conditional | 8 | revision | single | 10.211 | 10.211 | 9.797 | 12.295 | 62.195 |
| 12004 | conditional | 9 | clean | single | 9.011 | 9.011 | 9.147 | 11.454 | 61.374 |
| 12004 | conditional | 10 | noise | single | 10.903 | 16.361 | 10.595 | 12.424 | 59.387 |
| 12004 | conditional | 11 | recovery | single | 8.792 | 8.792 | 8.862 | 12.590 | 61.011 |
| 12004 | conditional | 12 | return | single | 9.473 | 9.473 | 9.421 | 17.790 | 70.654 |
| 12004 | conditional | 13 | revision | single | 9.929 | 9.929 | 10.103 | 12.374 | 63.092 |
| 12004 | conditional | 14 | clean | single | 8.675 | 8.675 | 8.886 | 9.774 | 61.362 |
| 12004 | conditional | 15 | noise | single | 10.251 | 15.579 | 10.195 | 11.336 | 60.049 |
| 12004 | conditional | 16 | recovery | single | 9.274 | 9.274 | 9.127 | 10.892 | 60.602 |
| 12004 | conditional | 17 | return | single | 10.165 | 10.165 | 10.261 | 17.206 | 69.751 |
| 12004 | conditional | 18 | revision | single | 10.051 | 10.051 | 10.339 | 12.067 | 61.801 |
| 12004 | conditional | 1 | introduction_1 | sustained | 10.293 | 10.293 | 10.452 | 9.214 | 56.256 |
| 12004 | conditional | 2 | introduction_2 | sustained | 8.163 | 8.163 | 8.010 | 9.452 | 62.186 |
| 12004 | conditional | 3 | introduction_3 | sustained | 8.513 | 8.513 | 9.043 | 10.462 | 60.669 |
| 12004 | conditional | 4 | clean | sustained | 8.379 | 8.379 | 8.350 | 11.826 | 62.961 |
| 12004 | conditional | 5 | noise | sustained | 9.865 | 15.698 | 9.322 | 17.976 | 60.059 |
| 12004 | conditional | 6 | recovery | sustained | 8.949 | 8.949 | 8.652 | 18.210 | 60.721 |
| 12004 | conditional | 7 | return | sustained | 11.243 | 11.243 | 11.543 | 10.771 | 68.759 |
| 12004 | conditional | 8 | revision | sustained | 10.162 | 10.162 | 10.029 | 10.857 | 63.443 |
| 12004 | conditional | 9 | clean | sustained | 9.184 | 9.184 | 9.377 | 16.198 | 61.420 |
| 12004 | conditional | 10 | noise | sustained | 11.426 | 16.574 | 10.866 | 12.424 | 59.177 |
| 12004 | conditional | 11 | recovery | sustained | 8.916 | 8.916 | 8.765 | 12.596 | 60.358 |
| 12004 | conditional | 12 | return | sustained | 9.904 | 9.904 | 9.784 | 14.619 | 70.670 |
| 12004 | conditional | 13 | revision | sustained | 10.011 | 10.011 | 9.681 | 11.507 | 62.668 |
| 12004 | conditional | 14 | clean | sustained | 8.596 | 8.596 | 8.777 | 9.303 | 61.560 |
| 12004 | conditional | 15 | noise | sustained | 10.329 | 15.551 | 10.357 | 16.606 | 60.364 |
| 12004 | conditional | 16 | recovery | sustained | 9.296 | 9.296 | 9.499 | 17.696 | 60.617 |
| 12004 | conditional | 17 | return | sustained | 10.203 | 10.203 | 11.125 | 11.843 | 69.608 |
| 12004 | conditional | 18 | revision | sustained | 10.147 | 10.147 | 10.440 | 11.958 | 61.417 |
| 12004 | recurrent | 1 | introduction_1 | immediate | 10.469 | 10.469 | 11.117 | 20.288 | 56.952 |
| 12004 | recurrent | 2 | introduction_2 | immediate | 8.152 | 8.152 | 8.470 | 24.670 | 61.197 |
| 12004 | recurrent | 3 | introduction_3 | immediate | 9.119 | 9.119 | 9.184 | 23.868 | 62.018 |
| 12004 | recurrent | 4 | clean | immediate | 8.343 | 8.343 | 8.633 | 25.294 | 62.936 |
| 12004 | recurrent | 5 | noise | immediate | 10.335 | 15.699 | 10.390 | 24.267 | 60.147 |
| 12004 | recurrent | 6 | recovery | immediate | 8.697 | 8.697 | 8.809 | 25.270 | 61.942 |
| 12004 | recurrent | 7 | return | immediate | 13.438 | 13.438 | 13.838 | 17.792 | 66.348 |
| 12004 | recurrent | 8 | revision | immediate | 10.112 | 10.112 | 10.261 | 16.881 | 61.932 |
| 12004 | recurrent | 9 | clean | immediate | 8.733 | 8.733 | 9.161 | 24.136 | 62.228 |
| 12004 | recurrent | 10 | noise | immediate | 11.066 | 15.028 | 11.870 | 22.894 | 59.579 |
| 12004 | recurrent | 11 | recovery | immediate | 8.844 | 8.844 | 8.786 | 24.000 | 61.984 |
| 12004 | recurrent | 12 | return | immediate | 11.390 | 11.390 | 10.466 | 18.840 | 69.080 |
| 12004 | recurrent | 13 | revision | immediate | 10.203 | 10.203 | 9.840 | 18.386 | 62.320 |
| 12004 | recurrent | 14 | clean | immediate | 8.665 | 8.665 | 8.958 | 23.248 | 62.317 |
| 12004 | recurrent | 15 | noise | immediate | 10.609 | 15.917 | 10.885 | 24.264 | 59.827 |
| 12004 | recurrent | 16 | recovery | immediate | 8.955 | 8.955 | 8.855 | 23.399 | 61.456 |
| 12004 | recurrent | 17 | return | immediate | 11.209 | 11.209 | 11.222 | 18.526 | 68.686 |
| 12004 | recurrent | 18 | revision | immediate | 9.447 | 9.447 | 9.451 | 18.387 | 62.927 |
| 12004 | recurrent | 1 | introduction_1 | periodic | 10.688 | 10.688 | 10.879 | 20.855 | 57.263 |
| 12004 | recurrent | 2 | introduction_2 | periodic | 8.250 | 8.250 | 8.490 | 23.399 | 61.365 |
| 12004 | recurrent | 3 | introduction_3 | periodic | 9.112 | 9.112 | 9.199 | 23.081 | 62.250 |
| 12004 | recurrent | 4 | clean | periodic | 8.206 | 8.206 | 8.640 | 25.318 | 63.101 |
| 12004 | recurrent | 5 | noise | periodic | 10.239 | 15.749 | 10.267 | 23.888 | 60.327 |
| 12004 | recurrent | 6 | recovery | periodic | 8.800 | 8.800 | 8.871 | 24.399 | 61.975 |
| 12004 | recurrent | 7 | return | periodic | 14.675 | 14.675 | 14.471 | 18.032 | 66.144 |
| 12004 | recurrent | 8 | revision | periodic | 10.682 | 10.682 | 10.546 | 19.863 | 61.505 |
| 12004 | recurrent | 9 | clean | periodic | 8.724 | 8.724 | 9.208 | 23.519 | 62.009 |
| 12004 | recurrent | 10 | noise | periodic | 11.057 | 16.861 | 11.483 | 22.139 | 59.769 |
| 12004 | recurrent | 11 | recovery | periodic | 8.778 | 8.778 | 8.837 | 24.331 | 62.012 |
| 12004 | recurrent | 12 | return | periodic | 12.363 | 12.363 | 10.896 | 18.561 | 69.366 |
| 12004 | recurrent | 13 | revision | periodic | 10.659 | 10.659 | 10.407 | 18.093 | 61.996 |
| 12004 | recurrent | 14 | clean | periodic | 8.795 | 8.795 | 8.966 | 23.716 | 62.311 |
| 12004 | recurrent | 15 | noise | periodic | 10.417 | 15.653 | 10.882 | 23.636 | 59.680 |
| 12004 | recurrent | 16 | recovery | periodic | 9.155 | 9.155 | 9.040 | 23.618 | 61.194 |
| 12004 | recurrent | 17 | return | periodic | 12.290 | 12.290 | 11.882 | 19.564 | 68.423 |
| 12004 | recurrent | 18 | revision | periodic | 10.123 | 10.123 | 9.942 | 19.454 | 62.671 |
| 12004 | recurrent | 1 | introduction_1 | single | 10.774 | 10.774 | 11.269 | 21.022 | 56.946 |
| 12004 | recurrent | 2 | introduction_2 | single | 8.486 | 8.486 | 8.316 | 23.399 | 61.270 |
| 12004 | recurrent | 3 | introduction_3 | single | 8.990 | 8.990 | 9.380 | 23.081 | 61.975 |
| 12004 | recurrent | 4 | clean | single | 8.422 | 8.422 | 8.696 | 24.030 | 62.991 |
| 12004 | recurrent | 5 | noise | single | 10.293 | 15.864 | 10.205 | 24.010 | 60.220 |
| 12004 | recurrent | 6 | recovery | single | 8.795 | 8.795 | 8.871 | 24.399 | 61.996 |
| 12004 | recurrent | 7 | return | single | 15.220 | 15.220 | 15.013 | 18.032 | 65.662 |
| 12004 | recurrent | 8 | revision | single | 11.129 | 11.129 | 10.824 | 19.863 | 61.261 |
| 12004 | recurrent | 9 | clean | single | 8.703 | 8.703 | 9.273 | 23.519 | 62.045 |
| 12004 | recurrent | 10 | noise | single | 10.221 | 15.885 | 11.090 | 22.139 | 60.208 |
| 12004 | recurrent | 11 | recovery | single | 8.881 | 8.881 | 8.807 | 24.331 | 62.244 |
| 12004 | recurrent | 12 | return | single | 12.570 | 12.570 | 11.454 | 18.561 | 68.738 |
| 12004 | recurrent | 13 | revision | single | 10.708 | 10.708 | 10.439 | 14.998 | 61.536 |
| 12004 | recurrent | 14 | clean | single | 8.707 | 8.707 | 8.892 | 23.716 | 62.015 |
| 12004 | recurrent | 15 | noise | single | 10.507 | 15.701 | 10.908 | 24.010 | 59.641 |
| 12004 | recurrent | 16 | recovery | single | 9.149 | 9.149 | 9.001 | 23.618 | 61.209 |
| 12004 | recurrent | 17 | return | single | 12.890 | 12.890 | 12.614 | 20.212 | 68.118 |
| 12004 | recurrent | 18 | revision | single | 10.405 | 10.405 | 10.031 | 18.219 | 62.317 |
| 12004 | recurrent | 1 | introduction_1 | sustained | 10.682 | 10.682 | 11.440 | 21.022 | 57.135 |
| 12004 | recurrent | 2 | introduction_2 | sustained | 8.510 | 8.510 | 8.560 | 23.888 | 61.548 |
| 12004 | recurrent | 3 | introduction_3 | sustained | 8.956 | 8.956 | 9.387 | 21.370 | 61.987 |
| 12004 | recurrent | 4 | clean | sustained | 8.380 | 8.380 | 8.421 | 24.495 | 62.720 |
| 12004 | recurrent | 5 | noise | sustained | 8.539 | 15.156 | 8.562 | 24.495 | 62.161 |
| 12004 | recurrent | 6 | recovery | sustained | 8.786 | 8.786 | 8.626 | 24.208 | 62.231 |
| 12004 | recurrent | 7 | return | sustained | 15.445 | 15.445 | 15.388 | 18.032 | 64.874 |
| 12004 | recurrent | 8 | revision | sustained | 11.236 | 11.236 | 11.005 | 18.350 | 61.447 |
| 12004 | recurrent | 9 | clean | sustained | 8.761 | 8.761 | 9.247 | 22.494 | 61.884 |
| 12004 | recurrent | 10 | noise | sustained | 9.190 | 14.876 | 9.570 | 22.139 | 61.267 |
| 12004 | recurrent | 11 | recovery | sustained | 8.821 | 8.821 | 9.034 | 24.761 | 62.564 |
| 12004 | recurrent | 12 | return | sustained | 12.663 | 12.663 | 11.209 | 21.610 | 68.350 |
| 12004 | recurrent | 13 | revision | sustained | 11.191 | 11.191 | 10.397 | 14.624 | 62.057 |
| 12004 | recurrent | 14 | clean | sustained | 8.471 | 8.471 | 8.750 | 22.799 | 62.064 |
| 12004 | recurrent | 15 | noise | sustained | 10.316 | 15.824 | 10.529 | 24.010 | 60.083 |
| 12004 | recurrent | 16 | recovery | sustained | 8.941 | 8.941 | 8.911 | 24.220 | 62.088 |
| 12004 | recurrent | 17 | return | sustained | 12.631 | 12.631 | 11.905 | 18.170 | 68.204 |
| 12004 | recurrent | 18 | revision | sustained | 10.201 | 10.201 | 9.735 | 18.219 | 62.695 |
| 12005 | conditional | 1 | introduction_1 | immediate | 9.239 | 9.239 | 10.125 | 9.316 | 57.889 |
| 12005 | conditional | 2 | introduction_2 | immediate | 8.199 | 8.199 | 8.186 | 10.461 | 61.298 |
| 12005 | conditional | 3 | introduction_3 | immediate | 8.143 | 8.143 | 8.887 | 9.996 | 63.315 |
| 12005 | conditional | 4 | clean | immediate | 8.034 | 8.034 | 8.268 | 11.378 | 63.800 |
| 12005 | conditional | 5 | noise | immediate | 9.614 | 15.152 | 9.749 | 11.821 | 61.981 |
| 12005 | conditional | 6 | recovery | immediate | 8.587 | 8.587 | 8.787 | 11.989 | 62.872 |
| 12005 | conditional | 7 | return | immediate | 8.926 | 8.926 | 9.168 | 11.126 | 73.645 |
| 12005 | conditional | 8 | revision | immediate | 9.462 | 9.462 | 10.523 | 10.136 | 62.259 |
| 12005 | conditional | 9 | clean | immediate | 8.451 | 8.451 | 8.828 | 8.754 | 63.611 |
| 12005 | conditional | 10 | noise | immediate | 10.613 | 14.907 | 10.755 | 9.516 | 61.130 |
| 12005 | conditional | 11 | recovery | immediate | 8.610 | 8.610 | 8.395 | 9.222 | 63.843 |
| 12005 | conditional | 12 | return | immediate | 8.660 | 8.660 | 9.234 | 11.231 | 73.315 |
| 12005 | conditional | 13 | revision | immediate | 9.591 | 9.591 | 10.537 | 10.661 | 62.036 |
| 12005 | conditional | 14 | clean | immediate | 8.986 | 8.986 | 9.023 | 8.290 | 62.732 |
| 12005 | conditional | 15 | noise | immediate | 10.444 | 15.726 | 10.331 | 10.263 | 61.258 |
| 12005 | conditional | 16 | recovery | immediate | 8.510 | 8.510 | 8.677 | 8.909 | 63.464 |
| 12005 | conditional | 17 | return | immediate | 9.176 | 9.176 | 9.198 | 10.451 | 72.604 |
| 12005 | conditional | 18 | revision | immediate | 9.792 | 9.792 | 10.891 | 10.919 | 61.612 |
| 12005 | conditional | 1 | introduction_1 | periodic | 10.264 | 10.264 | 10.316 | 8.890 | 57.629 |
| 12005 | conditional | 2 | introduction_2 | periodic | 8.606 | 8.606 | 8.216 | 10.518 | 61.249 |
| 12005 | conditional | 3 | introduction_3 | periodic | 8.069 | 8.069 | 8.870 | 10.984 | 63.321 |
| 12005 | conditional | 4 | clean | periodic | 8.338 | 8.338 | 8.278 | 11.269 | 63.748 |
| 12005 | conditional | 5 | noise | periodic | 9.732 | 15.275 | 9.682 | 18.395 | 62.073 |
| 12005 | conditional | 6 | recovery | periodic | 8.674 | 8.674 | 8.829 | 11.953 | 62.802 |
| 12005 | conditional | 7 | return | periodic | 9.329 | 9.329 | 9.524 | 15.363 | 73.288 |
| 12005 | conditional | 8 | revision | periodic | 9.734 | 9.734 | 10.551 | 10.090 | 62.146 |
| 12005 | conditional | 9 | clean | periodic | 8.510 | 8.510 | 8.842 | 8.326 | 63.583 |
| 12005 | conditional | 10 | noise | periodic | 11.146 | 16.673 | 11.372 | 9.553 | 60.641 |
| 12005 | conditional | 11 | recovery | periodic | 8.616 | 8.616 | 8.405 | 9.427 | 63.818 |
| 12005 | conditional | 12 | return | periodic | 8.779 | 8.779 | 9.378 | 10.973 | 73.087 |
| 12005 | conditional | 13 | revision | periodic | 9.761 | 9.761 | 10.642 | 10.949 | 62.021 |
| 12005 | conditional | 14 | clean | periodic | 9.047 | 9.047 | 9.039 | 8.266 | 62.732 |
| 12005 | conditional | 15 | noise | periodic | 10.135 | 15.704 | 10.275 | 9.654 | 61.356 |
| 12005 | conditional | 16 | recovery | periodic | 8.535 | 8.535 | 8.733 | 8.919 | 63.361 |
| 12005 | conditional | 17 | return | periodic | 9.456 | 9.456 | 9.741 | 10.453 | 72.153 |
| 12005 | conditional | 18 | revision | periodic | 9.944 | 9.944 | 10.971 | 11.086 | 61.530 |
| 12005 | conditional | 1 | introduction_1 | single | 10.154 | 10.154 | 10.611 | 8.799 | 57.565 |
| 12005 | conditional | 2 | introduction_2 | single | 8.647 | 8.647 | 8.567 | 12.034 | 61.023 |
| 12005 | conditional | 3 | introduction_3 | single | 8.268 | 8.268 | 8.698 | 10.420 | 63.275 |
| 12005 | conditional | 4 | clean | single | 8.390 | 8.390 | 8.264 | 11.269 | 63.983 |
| 12005 | conditional | 5 | noise | single | 9.892 | 15.276 | 9.762 | 13.043 | 62.091 |
| 12005 | conditional | 6 | recovery | single | 8.617 | 8.617 | 8.811 | 11.953 | 62.814 |
| 12005 | conditional | 7 | return | single | 9.357 | 9.357 | 9.588 | 15.363 | 73.056 |
| 12005 | conditional | 8 | revision | single | 9.776 | 9.776 | 10.768 | 10.178 | 62.006 |
| 12005 | conditional | 9 | clean | single | 8.687 | 8.687 | 8.902 | 8.326 | 63.425 |
| 12005 | conditional | 10 | noise | single | 10.889 | 16.293 | 10.885 | 9.519 | 60.904 |
| 12005 | conditional | 11 | recovery | single | 8.404 | 8.404 | 8.482 | 9.427 | 63.721 |
| 12005 | conditional | 12 | return | single | 8.891 | 8.891 | 9.267 | 10.568 | 73.181 |
| 12005 | conditional | 13 | revision | single | 9.810 | 9.810 | 10.632 | 10.911 | 62.210 |
| 12005 | conditional | 14 | clean | single | 9.001 | 9.001 | 8.947 | 8.339 | 62.619 |
| 12005 | conditional | 15 | noise | single | 10.054 | 15.636 | 10.303 | 10.261 | 61.240 |
| 12005 | conditional | 16 | recovery | single | 8.477 | 8.477 | 8.769 | 8.919 | 63.315 |
| 12005 | conditional | 17 | return | single | 9.639 | 9.639 | 9.843 | 10.497 | 72.363 |
| 12005 | conditional | 18 | revision | single | 10.010 | 10.010 | 10.931 | 10.987 | 61.435 |
| 12005 | conditional | 1 | introduction_1 | sustained | 10.238 | 10.238 | 10.415 | 8.096 | 57.321 |
| 12005 | conditional | 2 | introduction_2 | sustained | 8.124 | 8.124 | 8.325 | 9.805 | 61.578 |
| 12005 | conditional | 3 | introduction_3 | sustained | 7.646 | 7.646 | 8.534 | 10.091 | 63.361 |
| 12005 | conditional | 4 | clean | sustained | 8.110 | 8.110 | 8.117 | 11.124 | 63.370 |
| 12005 | conditional | 5 | noise | sustained | 9.331 | 15.157 | 9.586 | 11.432 | 61.548 |
| 12005 | conditional | 6 | recovery | sustained | 8.664 | 8.664 | 8.838 | 11.980 | 62.772 |
| 12005 | conditional | 7 | return | sustained | 9.815 | 9.815 | 9.981 | 16.422 | 72.702 |
| 12005 | conditional | 8 | revision | sustained | 9.794 | 9.794 | 10.586 | 10.178 | 61.768 |
| 12005 | conditional | 9 | clean | sustained | 8.652 | 8.652 | 8.991 | 8.089 | 63.535 |
| 12005 | conditional | 10 | noise | sustained | 11.179 | 16.546 | 11.172 | 9.652 | 59.811 |
| 12005 | conditional | 11 | recovery | sustained | 8.591 | 8.591 | 8.544 | 9.721 | 63.147 |
| 12005 | conditional | 12 | return | sustained | 9.074 | 9.074 | 9.616 | 11.179 | 73.154 |
| 12005 | conditional | 13 | revision | sustained | 9.821 | 9.821 | 11.378 | 10.532 | 62.537 |
| 12005 | conditional | 14 | clean | sustained | 9.013 | 9.013 | 8.576 | 9.050 | 62.543 |
| 12005 | conditional | 15 | noise | sustained | 9.705 | 15.273 | 9.654 | 9.821 | 62.256 |
| 12005 | conditional | 16 | recovery | sustained | 8.561 | 8.561 | 8.781 | 8.509 | 63.907 |
| 12005 | conditional | 17 | return | sustained | 9.891 | 9.891 | 9.783 | 10.497 | 73.465 |
| 12005 | conditional | 18 | revision | sustained | 9.893 | 9.893 | 11.010 | 10.745 | 60.892 |
| 12005 | recurrent | 1 | introduction_1 | immediate | 10.036 | 10.036 | 10.499 | 15.080 | 57.529 |
| 12005 | recurrent | 2 | introduction_2 | immediate | 8.387 | 8.387 | 8.556 | 22.731 | 61.551 |
| 12005 | recurrent | 3 | introduction_3 | immediate | 8.308 | 8.308 | 9.018 | 22.240 | 63.925 |
| 12005 | recurrent | 4 | clean | immediate | 8.635 | 8.635 | 8.571 | 22.628 | 63.885 |
| 12005 | recurrent | 5 | noise | immediate | 10.443 | 15.829 | 10.760 | 23.983 | 61.240 |
| 12005 | recurrent | 6 | recovery | immediate | 8.806 | 8.806 | 8.843 | 24.822 | 62.598 |
| 12005 | recurrent | 7 | return | immediate | 11.296 | 11.296 | 11.288 | 18.639 | 71.500 |
| 12005 | recurrent | 8 | revision | immediate | 9.765 | 9.765 | 10.247 | 22.486 | 61.859 |
| 12005 | recurrent | 9 | clean | immediate | 8.720 | 8.720 | 9.077 | 18.440 | 63.998 |
| 12005 | recurrent | 10 | noise | immediate | 10.824 | 15.104 | 11.420 | 17.865 | 61.169 |
| 12005 | recurrent | 11 | recovery | immediate | 8.464 | 8.464 | 8.531 | 21.812 | 63.727 |
| 12005 | recurrent | 12 | return | immediate | 10.382 | 10.382 | 10.056 | 13.451 | 71.741 |
| 12005 | recurrent | 13 | revision | immediate | 9.614 | 9.614 | 10.654 | 23.681 | 62.772 |
| 12005 | recurrent | 14 | clean | immediate | 8.784 | 8.784 | 8.964 | 11.187 | 63.354 |
| 12005 | recurrent | 15 | noise | immediate | 10.662 | 16.118 | 11.121 | 17.172 | 60.995 |
| 12005 | recurrent | 16 | recovery | immediate | 8.995 | 8.995 | 9.204 | 23.313 | 63.138 |
| 12005 | recurrent | 17 | return | immediate | 10.229 | 10.229 | 10.681 | 14.298 | 71.576 |
| 12005 | recurrent | 18 | revision | immediate | 9.861 | 9.861 | 10.789 | 20.907 | 61.371 |
| 12005 | recurrent | 1 | introduction_1 | periodic | 10.456 | 10.456 | 10.686 | 19.686 | 57.327 |
| 12005 | recurrent | 2 | introduction_2 | periodic | 8.504 | 8.504 | 8.634 | 21.114 | 61.490 |
| 12005 | recurrent | 3 | introduction_3 | periodic | 8.060 | 8.060 | 8.999 | 23.497 | 63.889 |
| 12005 | recurrent | 4 | clean | periodic | 8.694 | 8.694 | 8.659 | 20.181 | 63.998 |
| 12005 | recurrent | 5 | noise | periodic | 10.328 | 15.923 | 10.541 | 23.524 | 61.627 |
| 12005 | recurrent | 6 | recovery | periodic | 8.731 | 8.731 | 8.938 | 25.240 | 62.350 |
| 12005 | recurrent | 7 | return | periodic | 12.937 | 12.937 | 12.273 | 12.816 | 70.786 |
| 12005 | recurrent | 8 | revision | periodic | 10.330 | 10.330 | 10.832 | 22.857 | 61.411 |
| 12005 | recurrent | 9 | clean | periodic | 9.059 | 9.059 | 9.000 | 17.604 | 63.986 |
| 12005 | recurrent | 10 | noise | periodic | 10.482 | 16.696 | 11.153 | 17.430 | 61.328 |
| 12005 | recurrent | 11 | recovery | periodic | 8.824 | 8.824 | 8.523 | 19.350 | 63.800 |
| 12005 | recurrent | 12 | return | periodic | 11.200 | 11.200 | 11.018 | 18.698 | 71.121 |
| 12005 | recurrent | 13 | revision | periodic | 9.993 | 9.993 | 11.199 | 21.829 | 62.219 |
| 12005 | recurrent | 14 | clean | periodic | 8.803 | 8.803 | 8.951 | 15.577 | 63.550 |
| 12005 | recurrent | 15 | noise | periodic | 10.513 | 16.029 | 10.986 | 16.817 | 61.395 |
| 12005 | recurrent | 16 | recovery | periodic | 8.781 | 8.781 | 9.271 | 23.943 | 63.150 |
| 12005 | recurrent | 17 | return | periodic | 11.524 | 11.524 | 11.309 | 14.855 | 71.283 |
| 12005 | recurrent | 18 | revision | periodic | 10.448 | 10.448 | 11.727 | 20.916 | 60.776 |
| 12005 | recurrent | 1 | introduction_1 | single | 10.469 | 10.469 | 10.678 | 17.172 | 57.336 |
| 12005 | recurrent | 2 | introduction_2 | single | 8.437 | 8.437 | 8.619 | 21.114 | 61.380 |
| 12005 | recurrent | 3 | introduction_3 | single | 8.155 | 8.155 | 8.782 | 23.497 | 64.145 |
| 12005 | recurrent | 4 | clean | single | 8.610 | 8.610 | 8.610 | 20.181 | 63.956 |
| 12005 | recurrent | 5 | noise | single | 9.948 | 15.635 | 10.403 | 23.524 | 61.438 |
| 12005 | recurrent | 6 | recovery | single | 8.801 | 8.801 | 9.040 | 25.240 | 62.515 |
| 12005 | recurrent | 7 | return | single | 13.118 | 13.118 | 12.401 | 13.219 | 70.798 |
| 12005 | recurrent | 8 | revision | single | 10.303 | 10.303 | 10.763 | 22.805 | 61.517 |
| 12005 | recurrent | 9 | clean | single | 8.859 | 8.859 | 9.001 | 18.850 | 63.934 |
| 12005 | recurrent | 10 | noise | single | 10.667 | 16.337 | 11.051 | 17.430 | 61.136 |
| 12005 | recurrent | 11 | recovery | single | 8.752 | 8.752 | 8.557 | 19.350 | 63.782 |
| 12005 | recurrent | 12 | return | single | 11.391 | 11.391 | 11.246 | 18.698 | 71.024 |
| 12005 | recurrent | 13 | revision | single | 10.164 | 10.164 | 11.029 | 21.829 | 62.384 |
| 12005 | recurrent | 14 | clean | single | 8.931 | 8.931 | 9.006 | 17.263 | 63.519 |
| 12005 | recurrent | 15 | noise | single | 10.455 | 15.906 | 10.970 | 16.817 | 61.328 |
| 12005 | recurrent | 16 | recovery | single | 8.967 | 8.967 | 9.279 | 20.507 | 63.370 |
| 12005 | recurrent | 17 | return | single | 11.674 | 11.674 | 11.495 | 15.486 | 71.274 |
| 12005 | recurrent | 18 | revision | single | 10.526 | 10.526 | 11.878 | 20.505 | 60.571 |
| 12005 | recurrent | 1 | introduction_1 | sustained | 10.651 | 10.651 | 10.910 | 17.172 | 56.995 |
| 12005 | recurrent | 2 | introduction_2 | sustained | 8.353 | 8.353 | 8.283 | 23.208 | 61.749 |
| 12005 | recurrent | 3 | introduction_3 | sustained | 8.155 | 8.155 | 9.004 | 21.884 | 63.602 |
| 12005 | recurrent | 4 | clean | sustained | 8.852 | 8.852 | 8.731 | 22.241 | 64.432 |
| 12005 | recurrent | 5 | noise | sustained | 9.849 | 15.555 | 10.347 | 24.193 | 61.774 |
| 12005 | recurrent | 6 | recovery | sustained | 9.210 | 9.210 | 9.213 | 24.644 | 62.512 |
| 12005 | recurrent | 7 | return | sustained | 13.591 | 13.591 | 12.532 | 11.892 | 70.374 |
| 12005 | recurrent | 8 | revision | sustained | 10.347 | 10.347 | 10.407 | 22.643 | 62.094 |
| 12005 | recurrent | 9 | clean | sustained | 8.784 | 8.784 | 8.881 | 18.362 | 63.989 |
| 12005 | recurrent | 10 | noise | sustained | 10.233 | 16.312 | 10.678 | 16.721 | 61.920 |
| 12005 | recurrent | 11 | recovery | sustained | 8.756 | 8.756 | 8.691 | 20.303 | 63.376 |
| 12005 | recurrent | 12 | return | sustained | 11.746 | 11.746 | 11.792 | 16.017 | 70.718 |
| 12005 | recurrent | 13 | revision | sustained | 10.435 | 10.435 | 11.177 | 20.314 | 62.265 |
| 12005 | recurrent | 14 | clean | sustained | 9.015 | 9.015 | 8.838 | 15.577 | 63.770 |
| 12005 | recurrent | 15 | noise | sustained | 9.750 | 15.311 | 10.288 | 11.122 | 62.076 |
| 12005 | recurrent | 16 | recovery | sustained | 8.867 | 8.867 | 9.089 | 20.507 | 63.135 |
| 12005 | recurrent | 17 | return | sustained | 11.694 | 11.694 | 11.213 | 15.195 | 71.481 |
| 12005 | recurrent | 18 | revision | sustained | 10.958 | 10.958 | 11.297 | 22.609 | 61.081 |
| 12006 | conditional | 1 | introduction_1 | immediate | 7.503 | 7.503 | 8.211 | 8.372 | 79.050 |
| 12006 | conditional | 2 | introduction_2 | immediate | 7.380 | 7.380 | 7.869 | 9.486 | 77.957 |
| 12006 | conditional | 3 | introduction_3 | immediate | 10.479 | 10.479 | 11.463 | 10.484 | 58.139 |
| 12006 | conditional | 4 | clean | immediate | 9.643 | 9.643 | 9.252 | 10.059 | 61.560 |
| 12006 | conditional | 5 | noise | immediate | 10.350 | 15.821 | 10.269 | 15.574 | 60.373 |
| 12006 | conditional | 6 | recovery | immediate | 8.741 | 8.741 | 8.840 | 16.340 | 63.058 |
| 12006 | conditional | 7 | return | immediate | 9.375 | 9.375 | 9.192 | 10.777 | 70.282 |
| 12006 | conditional | 8 | revision | immediate | 10.236 | 10.236 | 10.780 | 16.394 | 62.726 |
| 12006 | conditional | 9 | clean | immediate | 9.346 | 9.346 | 8.786 | 10.373 | 62.546 |
| 12006 | conditional | 10 | noise | immediate | 11.208 | 14.925 | 11.273 | 10.474 | 60.001 |
| 12006 | conditional | 11 | recovery | immediate | 8.844 | 8.844 | 8.467 | 10.585 | 63.330 |
| 12006 | conditional | 12 | return | immediate | 9.155 | 9.155 | 9.183 | 10.396 | 70.334 |
| 12006 | conditional | 13 | revision | immediate | 9.594 | 9.594 | 10.729 | 11.348 | 64.478 |
| 12006 | conditional | 14 | clean | immediate | 8.872 | 8.872 | 8.790 | 10.534 | 62.888 |
| 12006 | conditional | 15 | noise | immediate | 10.500 | 15.906 | 10.388 | 16.948 | 60.968 |
| 12006 | conditional | 16 | recovery | immediate | 8.663 | 8.663 | 8.488 | 10.891 | 62.540 |
| 12006 | conditional | 17 | return | immediate | 8.975 | 8.975 | 8.963 | 11.275 | 70.663 |
| 12006 | conditional | 18 | revision | immediate | 9.947 | 9.947 | 10.521 | 11.276 | 63.727 |
| 12006 | conditional | 1 | introduction_1 | periodic | 7.610 | 7.610 | 8.233 | 8.518 | 78.970 |
| 12006 | conditional | 2 | introduction_2 | periodic | 7.469 | 7.469 | 7.868 | 10.156 | 77.966 |
| 12006 | conditional | 3 | introduction_3 | periodic | 10.739 | 10.739 | 11.618 | 9.611 | 57.889 |
| 12006 | conditional | 4 | clean | periodic | 9.376 | 9.376 | 9.300 | 9.611 | 61.514 |
| 12006 | conditional | 5 | noise | periodic | 10.461 | 15.934 | 10.198 | 11.815 | 60.458 |
| 12006 | conditional | 6 | recovery | periodic | 8.838 | 8.838 | 8.892 | 16.228 | 62.903 |
| 12006 | conditional | 7 | return | periodic | 10.164 | 10.164 | 10.276 | 9.918 | 69.629 |
| 12006 | conditional | 8 | revision | periodic | 10.411 | 10.411 | 10.931 | 16.637 | 62.653 |
| 12006 | conditional | 9 | clean | periodic | 9.371 | 9.371 | 8.770 | 9.985 | 62.579 |
| 12006 | conditional | 10 | noise | periodic | 12.069 | 15.702 | 12.224 | 10.896 | 57.379 |
| 12006 | conditional | 11 | recovery | periodic | 8.858 | 8.858 | 8.488 | 10.336 | 63.260 |
| 12006 | conditional | 12 | return | periodic | 9.711 | 9.711 | 9.345 | 11.162 | 70.395 |
| 12006 | conditional | 13 | revision | periodic | 9.494 | 9.494 | 10.798 | 11.285 | 64.111 |
| 12006 | conditional | 14 | clean | periodic | 9.117 | 9.117 | 8.798 | 10.375 | 62.906 |
| 12006 | conditional | 15 | noise | periodic | 10.183 | 15.797 | 10.352 | 17.031 | 60.995 |
| 12006 | conditional | 16 | recovery | periodic | 8.757 | 8.757 | 8.537 | 10.964 | 62.500 |
| 12006 | conditional | 17 | return | periodic | 9.476 | 9.476 | 9.166 | 10.939 | 70.627 |
| 12006 | conditional | 18 | revision | periodic | 10.151 | 10.151 | 10.598 | 10.911 | 63.739 |
| 12006 | conditional | 1 | introduction_1 | single | 7.494 | 7.494 | 8.049 | 8.428 | 78.848 |
| 12006 | conditional | 2 | introduction_2 | single | 7.478 | 7.478 | 7.884 | 10.156 | 78.131 |
| 12006 | conditional | 3 | introduction_3 | single | 11.066 | 11.066 | 11.984 | 9.351 | 57.742 |
| 12006 | conditional | 4 | clean | single | 9.376 | 9.376 | 9.274 | 10.148 | 61.386 |
| 12006 | conditional | 5 | noise | single | 10.077 | 15.629 | 9.919 | 22.412 | 60.696 |
| 12006 | conditional | 6 | recovery | single | 8.947 | 8.947 | 9.024 | 16.228 | 62.616 |
| 12006 | conditional | 7 | return | single | 10.119 | 10.119 | 10.273 | 10.431 | 69.550 |
| 12006 | conditional | 8 | revision | single | 10.356 | 10.356 | 11.257 | 16.637 | 62.122 |
| 12006 | conditional | 9 | clean | single | 9.237 | 9.237 | 8.922 | 16.121 | 62.167 |
| 12006 | conditional | 10 | noise | single | 12.086 | 15.658 | 12.111 | 10.896 | 57.779 |
| 12006 | conditional | 11 | recovery | single | 8.767 | 8.767 | 8.520 | 11.958 | 63.193 |
| 12006 | conditional | 12 | return | single | 9.952 | 9.952 | 9.446 | 11.162 | 70.285 |
| 12006 | conditional | 13 | revision | single | 9.841 | 9.841 | 10.699 | 11.520 | 64.359 |
| 12006 | conditional | 14 | clean | single | 9.090 | 9.090 | 8.779 | 10.375 | 62.869 |
| 12006 | conditional | 15 | noise | single | 10.076 | 15.757 | 10.154 | 17.031 | 60.806 |
| 12006 | conditional | 16 | recovery | single | 9.008 | 9.008 | 8.824 | 10.964 | 62.152 |
| 12006 | conditional | 17 | return | single | 9.719 | 9.719 | 9.246 | 10.939 | 70.370 |
| 12006 | conditional | 18 | revision | single | 10.060 | 10.060 | 10.667 | 10.911 | 63.959 |
| 12006 | conditional | 1 | introduction_1 | sustained | 7.159 | 7.159 | 7.643 | 7.552 | 78.903 |
| 12006 | conditional | 2 | introduction_2 | sustained | 6.979 | 6.979 | 7.316 | 9.400 | 78.464 |
| 12006 | conditional | 3 | introduction_3 | sustained | 10.947 | 10.947 | 11.654 | 9.351 | 57.278 |
| 12006 | conditional | 4 | clean | sustained | 9.538 | 9.538 | 9.116 | 9.959 | 61.224 |
| 12006 | conditional | 5 | noise | sustained | 9.916 | 15.772 | 9.431 | 19.971 | 61.700 |
| 12006 | conditional | 6 | recovery | sustained | 9.102 | 9.102 | 9.168 | 16.752 | 61.966 |
| 12006 | conditional | 7 | return | sustained | 10.010 | 10.010 | 10.617 | 10.543 | 68.750 |
| 12006 | conditional | 8 | revision | sustained | 10.560 | 10.560 | 11.242 | 13.183 | 62.973 |
| 12006 | conditional | 9 | clean | sustained | 9.004 | 9.004 | 8.566 | 9.473 | 62.488 |
| 12006 | conditional | 10 | noise | sustained | 11.747 | 14.776 | 11.158 | 10.896 | 60.947 |
| 12006 | conditional | 11 | recovery | sustained | 8.901 | 8.901 | 8.491 | 10.974 | 62.411 |
| 12006 | conditional | 12 | return | sustained | 10.015 | 10.015 | 9.542 | 10.366 | 70.743 |
| 12006 | conditional | 13 | revision | sustained | 9.961 | 9.961 | 10.694 | 11.343 | 63.855 |
| 12006 | conditional | 14 | clean | sustained | 9.107 | 9.107 | 8.697 | 10.375 | 62.601 |
| 12006 | conditional | 15 | noise | sustained | 10.161 | 15.765 | 10.773 | 17.031 | 60.580 |
| 12006 | conditional | 16 | recovery | sustained | 8.978 | 8.978 | 8.748 | 10.964 | 62.762 |
| 12006 | conditional | 17 | return | sustained | 9.550 | 9.550 | 9.092 | 17.356 | 70.386 |
| 12006 | conditional | 18 | revision | sustained | 10.306 | 10.306 | 10.993 | 11.567 | 62.677 |
| 12006 | recurrent | 1 | introduction_1 | immediate | 8.767 | 8.767 | 9.085 | 19.058 | 77.158 |
| 12006 | recurrent | 2 | introduction_2 | immediate | 7.560 | 7.560 | 8.279 | 23.981 | 77.322 |
| 12006 | recurrent | 3 | introduction_3 | immediate | 11.174 | 11.174 | 12.378 | 22.878 | 57.227 |
| 12006 | recurrent | 4 | clean | immediate | 9.940 | 9.940 | 9.575 | 23.143 | 61.639 |
| 12006 | recurrent | 5 | noise | immediate | 11.727 | 17.131 | 11.332 | 22.053 | 59.479 |
| 12006 | recurrent | 6 | recovery | immediate | 9.331 | 9.331 | 9.009 | 20.877 | 62.390 |
| 12006 | recurrent | 7 | return | immediate | 10.758 | 10.758 | 10.801 | 13.360 | 69.458 |
| 12006 | recurrent | 8 | revision | immediate | 10.696 | 10.696 | 11.170 | 20.249 | 62.784 |
| 12006 | recurrent | 9 | clean | immediate | 9.443 | 9.443 | 8.720 | 16.129 | 63.440 |
| 12006 | recurrent | 10 | noise | immediate | 11.646 | 15.369 | 11.561 | 20.060 | 59.827 |
| 12006 | recurrent | 11 | recovery | immediate | 8.993 | 8.993 | 8.664 | 22.652 | 63.104 |
| 12006 | recurrent | 12 | return | immediate | 11.037 | 11.037 | 10.742 | 13.298 | 69.479 |
| 12006 | recurrent | 13 | revision | immediate | 9.849 | 9.849 | 11.048 | 22.106 | 63.364 |
| 12006 | recurrent | 14 | clean | immediate | 9.343 | 9.343 | 9.253 | 20.033 | 63.354 |
| 12006 | recurrent | 15 | noise | immediate | 10.638 | 16.106 | 11.105 | 16.020 | 60.770 |
| 12006 | recurrent | 16 | recovery | immediate | 9.061 | 9.061 | 8.995 | 18.993 | 62.933 |
| 12006 | recurrent | 17 | return | immediate | 10.187 | 10.187 | 10.008 | 15.999 | 69.641 |
| 12006 | recurrent | 18 | revision | immediate | 9.968 | 9.968 | 10.782 | 16.806 | 63.828 |
| 12006 | recurrent | 1 | introduction_1 | periodic | 8.883 | 8.883 | 9.359 | 20.711 | 76.895 |
| 12006 | recurrent | 2 | introduction_2 | periodic | 7.629 | 7.629 | 8.217 | 21.781 | 77.258 |
| 12006 | recurrent | 3 | introduction_3 | periodic | 11.311 | 11.311 | 12.535 | 22.372 | 56.992 |
| 12006 | recurrent | 4 | clean | periodic | 9.904 | 9.904 | 9.486 | 23.089 | 61.469 |
| 12006 | recurrent | 5 | noise | periodic | 11.215 | 16.678 | 11.302 | 22.488 | 59.546 |
| 12006 | recurrent | 6 | recovery | periodic | 9.314 | 9.314 | 8.995 | 21.939 | 62.732 |
| 12006 | recurrent | 7 | return | periodic | 12.141 | 12.141 | 11.627 | 12.393 | 69.070 |
| 12006 | recurrent | 8 | revision | periodic | 11.431 | 11.431 | 11.589 | 20.804 | 62.177 |
| 12006 | recurrent | 9 | clean | periodic | 9.573 | 9.573 | 8.857 | 17.416 | 63.644 |
| 12006 | recurrent | 10 | noise | periodic | 11.277 | 17.023 | 11.041 | 18.494 | 60.654 |
| 12006 | recurrent | 11 | recovery | periodic | 9.110 | 9.110 | 8.679 | 21.972 | 63.144 |
| 12006 | recurrent | 12 | return | periodic | 12.196 | 12.196 | 11.393 | 18.997 | 68.985 |
| 12006 | recurrent | 13 | revision | periodic | 10.337 | 10.337 | 11.353 | 22.098 | 62.933 |
| 12006 | recurrent | 14 | clean | periodic | 9.337 | 9.337 | 9.288 | 17.656 | 63.141 |
| 12006 | recurrent | 15 | noise | periodic | 10.590 | 16.163 | 11.117 | 18.498 | 60.849 |
| 12006 | recurrent | 16 | recovery | periodic | 9.374 | 9.374 | 9.058 | 18.100 | 62.823 |
| 12006 | recurrent | 17 | return | periodic | 11.030 | 11.030 | 10.610 | 15.171 | 69.067 |
| 12006 | recurrent | 18 | revision | periodic | 10.521 | 10.521 | 11.618 | 17.528 | 63.474 |
| 12006 | recurrent | 1 | introduction_1 | single | 9.039 | 9.039 | 9.370 | 16.478 | 76.678 |
| 12006 | recurrent | 2 | introduction_2 | single | 7.425 | 7.425 | 8.037 | 20.633 | 77.411 |
| 12006 | recurrent | 3 | introduction_3 | single | 11.385 | 11.385 | 12.641 | 22.372 | 57.092 |
| 12006 | recurrent | 4 | clean | single | 9.965 | 9.965 | 9.728 | 22.147 | 61.157 |
| 12006 | recurrent | 5 | noise | single | 11.170 | 16.533 | 10.996 | 22.488 | 59.851 |
| 12006 | recurrent | 6 | recovery | single | 9.659 | 9.659 | 9.193 | 21.172 | 62.189 |
| 12006 | recurrent | 7 | return | single | 11.970 | 11.970 | 11.754 | 12.393 | 68.951 |
| 12006 | recurrent | 8 | revision | single | 11.477 | 11.477 | 11.771 | 20.804 | 61.523 |
| 12006 | recurrent | 9 | clean | single | 9.715 | 9.715 | 8.924 | 17.996 | 63.422 |
| 12006 | recurrent | 10 | noise | single | 10.321 | 16.009 | 10.004 | 18.494 | 61.456 |
| 12006 | recurrent | 11 | recovery | single | 9.079 | 9.079 | 8.591 | 21.972 | 63.205 |
| 12006 | recurrent | 12 | return | single | 12.551 | 12.551 | 11.828 | 18.997 | 68.808 |
| 12006 | recurrent | 13 | revision | single | 10.324 | 10.324 | 11.248 | 22.098 | 62.793 |
| 12006 | recurrent | 14 | clean | single | 9.382 | 9.382 | 9.314 | 15.432 | 63.449 |
| 12006 | recurrent | 15 | noise | single | 10.515 | 16.279 | 10.748 | 19.415 | 61.261 |
| 12006 | recurrent | 16 | recovery | single | 9.217 | 9.217 | 9.141 | 18.100 | 62.811 |
| 12006 | recurrent | 17 | return | single | 11.223 | 11.223 | 10.317 | 13.303 | 69.266 |
| 12006 | recurrent | 18 | revision | single | 10.502 | 10.502 | 11.716 | 17.528 | 63.159 |
| 12006 | recurrent | 1 | introduction_1 | sustained | 9.284 | 9.284 | 9.424 | 17.914 | 76.746 |
| 12006 | recurrent | 2 | introduction_2 | sustained | 7.425 | 7.425 | 7.850 | 21.536 | 77.539 |
| 12006 | recurrent | 3 | introduction_3 | sustained | 11.502 | 11.502 | 12.716 | 21.852 | 56.815 |
| 12006 | recurrent | 4 | clean | sustained | 9.813 | 9.813 | 9.625 | 23.089 | 61.310 |
| 12006 | recurrent | 5 | noise | sustained | 10.347 | 16.429 | 10.311 | 20.447 | 61.334 |
| 12006 | recurrent | 6 | recovery | sustained | 9.521 | 9.521 | 9.112 | 21.209 | 62.570 |
| 12006 | recurrent | 7 | return | sustained | 12.463 | 12.463 | 11.582 | 15.932 | 69.028 |
| 12006 | recurrent | 8 | revision | sustained | 11.783 | 11.783 | 11.871 | 20.804 | 61.386 |
| 12006 | recurrent | 9 | clean | sustained | 8.978 | 8.978 | 8.707 | 17.996 | 63.123 |
| 12006 | recurrent | 10 | noise | sustained | 10.978 | 16.332 | 10.878 | 19.776 | 60.309 |
| 12006 | recurrent | 11 | recovery | sustained | 9.218 | 9.218 | 8.921 | 21.569 | 62.347 |
| 12006 | recurrent | 12 | return | sustained | 12.372 | 12.372 | 12.044 | 15.237 | 68.823 |
| 12006 | recurrent | 13 | revision | sustained | 10.249 | 10.249 | 11.109 | 22.507 | 63.187 |
| 12006 | recurrent | 14 | clean | sustained | 9.617 | 9.617 | 9.148 | 15.004 | 63.351 |
| 12006 | recurrent | 15 | noise | sustained | 10.497 | 16.367 | 10.758 | 14.596 | 61.810 |
| 12006 | recurrent | 16 | recovery | sustained | 9.385 | 9.385 | 9.275 | 17.585 | 63.113 |
| 12006 | recurrent | 17 | return | sustained | 12.027 | 12.027 | 11.010 | 12.896 | 68.796 |
| 12006 | recurrent | 18 | revision | sustained | 10.792 | 10.792 | 11.609 | 16.594 | 62.744 |

summary.json preserves both valid-old modes, every paired seed difference, per-block cumulative loss, knowledge endpoints, and window diagnostics. raw_results.jsonl.gz contains every complete block and fresh-reference record. Immediate has no adoption gate; its reported zero adoptions is not zero learning. Noise-block policy differences are not causal noise penalties without matched clean twin trajectories.

The immediate standalone learner needs one model. The delayed scaffold can retain three models (draft, proposal, committed) and uses two policy/evidence prediction passes per packet versus one for immediate, in addition to the shared training budget. resource_rows records parameter bytes, replay/optimizer sizes, and operation counts. These are logical standalone counts for the implemented scaffold, not measured runtime or total process RAM; evaluator probes are excluded from those online inference counts.
