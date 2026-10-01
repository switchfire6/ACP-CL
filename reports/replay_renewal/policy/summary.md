# Replay renewal: policy

288 complete episodes; paired seeds [10001, 10002, 10003, 10004, 10005, 10006].

Fresh qualification: **PASS**. Thresholds: marginal gain .02, cue benefit .002.

Error values below are Brier x100. Negative contrast is improvement; intervals are paired seed bootstrap, descriptive.

| Model | Contrast | Acquisition AUC [95% interval] | Terminal valid-old difference |
|---|---|---:|---:|
| conditional | split_minus_uniform | -0.467 [-0.653, -0.272] | +2.309 |
| conditional | recent_minus_uniform | -0.788 [-1.053, -0.537] | +2.021 |
| conditional | split_minus_recent | +0.321 [+0.223, +0.408] | +0.288 |
| recurrent | split_minus_uniform | -0.524 [-0.688, -0.355] | +1.403 |
| recurrent | recent_minus_uniform | -0.906 [-1.172, -0.651] | +1.925 |
| recurrent | split_minus_recent | +0.383 [+0.291, +0.494] | -0.522 |

## Prospective policy screen

* conditional: **FAIL**. Failed gates: introduction_retention, extra_noise_penalty, return_retention, revision_retention, clean_retention, noise_retention.
* recurrent: **FAIL**. Failed gates: introduction_retention, extra_noise_penalty, return_retention, revision_retention, noise_retention.

Full final-branch contrasts, every gate and interval warning are in summary.json.

## Qualification groups

| Group | Episodes | Marginal gain | Cue benefit | Pass |
|---|---:|---:|---:|---|
| conditional_stage_1 | 6 | 15.536 | 7.290 | True |
| conditional_stage_2 | 6 | 15.081 | 7.186 | True |
| conditional_stage_3 | 6 | 14.859 | 6.561 | True |
| conditional_cue_0 | 6 | 15.182 | 1.862 | True |
| conditional_cue_1 | 6 | 15.460 | 0.475 | True |
| conditional_cue_2 | 6 | 14.834 | 18.700 | True |
| recurrent_stage_1 | 6 | 15.385 | 7.415 | True |
| recurrent_stage_2 | 6 | 15.075 | 7.215 | True |
| recurrent_stage_3 | 6 | 14.606 | 6.227 | True |
| recurrent_cue_0 | 6 | 15.073 | 1.986 | True |
| recurrent_cue_1 | 6 | 15.155 | 0.365 | True |
| recurrent_cue_2 | 6 | 14.837 | 18.506 | True |

## Every episode

| Seed | Model | Stage | Arm / branch | AUC | Valid-old error | Damage | Cue benefit |
|---|---|---:|---|---:|---:|---:|---:|
| 10001 | conditional | 1 | fresh / novel | 8.222 | 23.060 | -1.964 | +18.957 |
| 10001 | conditional | 2 | fresh / novel | 8.510 | 21.478 | -3.546 | +0.102 |
| 10001 | conditional | 3 | fresh / novel | 9.975 | 23.704 | -1.320 | +1.255 |
| 10001 | conditional | 4 | recent / clean | 7.958 | 10.307 | -0.304 | n/a |
| 10001 | conditional | 4 | recent / noise | 10.422 | 14.286 | +3.676 | n/a |
| 10001 | conditional | 4 | recent / return | 7.806 | 11.186 | -5.852 | n/a |
| 10001 | conditional | 4 | recent / revision | 8.691 | 14.297 | +0.690 | +1.151 |
| 10001 | conditional | 1 | recent / novel | 8.482 | 10.263 | +1.229 | +17.980 |
| 10001 | conditional | 2 | recent / novel | 7.570 | 9.089 | -1.174 | +0.277 |
| 10001 | conditional | 3 | recent / novel | 8.520 | 10.610 | +1.521 | +1.021 |
| 10001 | conditional | 4 | split / clean | 8.125 | 14.752 | +3.806 | n/a |
| 10001 | conditional | 4 | split / noise | 9.946 | 18.215 | +7.269 | n/a |
| 10001 | conditional | 4 | split / return | 8.364 | 11.167 | -3.562 | n/a |
| 10001 | conditional | 4 | split / revision | 8.925 | 13.173 | +1.106 | +1.198 |
| 10001 | conditional | 1 | split / novel | 8.685 | 11.195 | +4.083 | +18.932 |
| 10001 | conditional | 2 | split / novel | 7.840 | 12.098 | +0.904 | +0.422 |
| 10001 | conditional | 3 | split / novel | 8.744 | 10.946 | -1.153 | +0.872 |
| 10001 | conditional | 4 | uniform / clean | 8.085 | 18.451 | +2.860 | n/a |
| 10001 | conditional | 4 | uniform / noise | 9.353 | 23.165 | +7.575 | n/a |
| 10001 | conditional | 4 | uniform / return | 8.420 | 10.532 | -9.709 | n/a |
| 10001 | conditional | 4 | uniform / revision | 9.010 | 15.373 | +0.754 | +0.418 |
| 10001 | conditional | 1 | uniform / novel | 9.104 | 8.840 | +2.327 | +15.651 |
| 10001 | conditional | 2 | uniform / novel | 8.214 | 15.566 | +6.725 | -0.464 |
| 10001 | conditional | 3 | uniform / novel | 8.541 | 15.591 | +0.025 | +0.201 |
| 10002 | conditional | 1 | fresh / novel | 6.749 | 20.805 | -4.379 | +3.573 |
| 10002 | conditional | 2 | fresh / novel | 7.266 | 16.400 | -8.784 | +1.013 |
| 10002 | conditional | 3 | fresh / novel | 8.793 | 22.600 | -2.583 | +17.961 |
| 10002 | conditional | 4 | recent / clean | 7.986 | 14.222 | +2.064 | n/a |
| 10002 | conditional | 4 | recent / noise | 10.618 | 23.237 | +11.078 | n/a |
| 10002 | conditional | 4 | recent / return | 8.356 | 12.541 | -2.249 | n/a |
| 10002 | conditional | 4 | recent / revision | 8.497 | 15.532 | +1.378 | +1.385 |
| 10002 | conditional | 1 | recent / novel | 6.489 | 14.992 | +6.994 | +3.980 |
| 10002 | conditional | 2 | recent / novel | 7.062 | 9.184 | -5.808 | +1.049 |
| 10002 | conditional | 3 | recent / novel | 9.116 | 12.158 | +2.975 | +17.527 |
| 10002 | conditional | 4 | split / clean | 8.536 | 17.805 | +3.277 | n/a |
| 10002 | conditional | 4 | split / noise | 10.960 | 17.613 | +3.085 | n/a |
| 10002 | conditional | 4 | split / return | 8.514 | 9.900 | -10.287 | n/a |
| 10002 | conditional | 4 | split / revision | 9.043 | 14.334 | +0.730 | +0.627 |
| 10002 | conditional | 1 | split / novel | 6.829 | 10.650 | +3.432 | +3.894 |
| 10002 | conditional | 2 | split / novel | 6.892 | 10.186 | -0.464 | +0.480 |
| 10002 | conditional | 3 | split / novel | 10.350 | 14.528 | +4.342 | +13.421 |
| 10002 | conditional | 4 | uniform / clean | 9.164 | 13.326 | +1.933 | n/a |
| 10002 | conditional | 4 | uniform / noise | 10.914 | 12.218 | +0.825 | n/a |
| 10002 | conditional | 4 | uniform / return | 9.117 | 9.198 | -3.796 | n/a |
| 10002 | conditional | 4 | uniform / revision | 9.870 | 13.681 | -2.403 | +0.557 |
| 10002 | conditional | 1 | uniform / novel | 7.284 | 8.925 | +1.852 | +1.575 |
| 10002 | conditional | 2 | uniform / novel | 7.506 | 9.400 | +0.475 | +0.966 |
| 10002 | conditional | 3 | uniform / novel | 11.840 | 11.393 | +1.993 | +11.822 |
| 10003 | conditional | 1 | fresh / novel | 7.734 | 17.608 | -7.568 | +1.663 |
| 10003 | conditional | 2 | fresh / novel | 9.470 | 17.872 | -7.304 | +19.025 |
| 10003 | conditional | 3 | fresh / novel | 9.421 | 20.426 | -4.749 | +0.657 |
| 10003 | conditional | 4 | recent / clean | 8.201 | 15.832 | +0.364 | n/a |
| 10003 | conditional | 4 | recent / noise | 10.934 | 21.895 | +6.427 | n/a |
| 10003 | conditional | 4 | recent / return | 8.326 | 14.300 | -10.297 | n/a |
| 10003 | conditional | 4 | recent / revision | 8.133 | 16.608 | +1.144 | +1.978 |
| 10003 | conditional | 1 | recent / novel | 7.509 | 16.378 | +5.453 | +1.528 |
| 10003 | conditional | 2 | recent / novel | 9.558 | 15.420 | -0.959 | +17.556 |
| 10003 | conditional | 3 | recent / novel | 8.519 | 15.468 | +0.048 | -0.123 |
| 10003 | conditional | 4 | split / clean | 8.322 | 22.525 | +3.154 | n/a |
| 10003 | conditional | 4 | split / noise | 10.886 | 22.353 | +2.982 | n/a |
| 10003 | conditional | 4 | split / return | 8.408 | 12.556 | -7.987 | n/a |
| 10003 | conditional | 4 | split / revision | 8.567 | 20.707 | +6.076 | +0.660 |
| 10003 | conditional | 1 | split / novel | 7.516 | 12.709 | +5.840 | +1.712 |
| 10003 | conditional | 2 | split / novel | 9.842 | 20.799 | +8.090 | +17.028 |
| 10003 | conditional | 3 | split / novel | 8.609 | 19.371 | -1.428 | +0.987 |
| 10003 | conditional | 4 | uniform / clean | 8.884 | 24.079 | +10.742 | n/a |
| 10003 | conditional | 4 | uniform / noise | 10.455 | 21.198 | +7.862 | n/a |
| 10003 | conditional | 4 | uniform / return | 9.969 | 10.466 | -5.168 | n/a |
| 10003 | conditional | 4 | uniform / revision | 8.873 | 18.753 | +5.550 | +0.585 |
| 10003 | conditional | 1 | uniform / novel | 7.372 | 9.139 | +2.545 | +1.237 |
| 10003 | conditional | 2 | uniform / novel | 10.909 | 10.706 | +1.567 | +13.160 |
| 10003 | conditional | 3 | uniform / novel | 9.168 | 13.336 | +2.631 | +0.370 |
| 10004 | conditional | 1 | fresh / novel | 7.608 | 16.885 | -8.140 | +0.345 |
| 10004 | conditional | 2 | fresh / novel | 8.026 | 17.742 | -7.283 | +2.385 |
| 10004 | conditional | 3 | fresh / novel | 8.964 | 19.382 | -5.643 | +17.926 |
| 10004 | conditional | 4 | recent / clean | 8.177 | 18.833 | -0.261 | n/a |
| 10004 | conditional | 4 | recent / noise | 10.756 | 21.725 | +2.632 | n/a |
| 10004 | conditional | 4 | recent / return | 7.963 | 11.792 | -11.486 | n/a |
| 10004 | conditional | 4 | recent / revision | 8.487 | 19.754 | +1.058 | +1.377 |
| 10004 | conditional | 1 | recent / novel | 6.907 | 10.963 | +3.369 | +1.049 |
| 10004 | conditional | 2 | recent / novel | 7.261 | 15.515 | +4.552 | +1.637 |
| 10004 | conditional | 3 | recent / novel | 9.211 | 19.094 | +3.579 | +17.794 |
| 10004 | conditional | 4 | split / clean | 8.446 | 13.861 | +3.955 | n/a |
| 10004 | conditional | 4 | split / noise | 10.095 | 14.090 | +4.184 | n/a |
| 10004 | conditional | 4 | split / return | 8.246 | 10.095 | -3.967 | n/a |
| 10004 | conditional | 4 | split / revision | 8.692 | 19.134 | +1.806 | +0.296 |
| 10004 | conditional | 1 | split / novel | 7.215 | 7.493 | +1.444 | -0.062 |
| 10004 | conditional | 2 | split / novel | 7.321 | 9.680 | +2.186 | +2.180 |
| 10004 | conditional | 3 | split / novel | 10.089 | 9.906 | +0.226 | +16.644 |
| 10004 | conditional | 4 | uniform / clean | 9.294 | 15.759 | +4.450 | n/a |
| 10004 | conditional | 4 | uniform / noise | 10.551 | 12.296 | +0.987 | n/a |
| 10004 | conditional | 4 | uniform / return | 9.069 | 9.963 | -3.035 | n/a |
| 10004 | conditional | 4 | uniform / revision | 10.332 | 19.159 | +0.448 | +0.353 |
| 10004 | conditional | 1 | uniform / novel | 7.300 | 8.646 | +1.776 | +0.271 |
| 10004 | conditional | 2 | uniform / novel | 7.640 | 9.521 | +0.875 | +1.859 |
| 10004 | conditional | 3 | uniform / novel | 11.443 | 11.309 | +1.788 | +11.900 |
| 10005 | conditional | 1 | fresh / novel | 8.185 | 21.683 | -3.411 | +0.331 |
| 10005 | conditional | 2 | fresh / novel | 9.351 | 21.477 | -3.618 | +19.455 |
| 10005 | conditional | 3 | fresh / novel | 9.760 | 16.964 | -8.130 | +1.161 |
| 10005 | conditional | 4 | recent / clean | 8.431 | 13.185 | +0.081 | n/a |
| 10005 | conditional | 4 | recent / noise | 10.989 | 16.314 | +3.210 | n/a |
| 10005 | conditional | 4 | recent / return | 9.210 | 20.144 | +0.006 | n/a |
| 10005 | conditional | 4 | recent / revision | 9.239 | 16.164 | -0.133 | +0.247 |
| 10005 | conditional | 1 | recent / novel | 7.450 | 9.288 | -0.318 | +0.413 |
| 10005 | conditional | 2 | recent / novel | 9.251 | 10.968 | +1.680 | +17.564 |
| 10005 | conditional | 3 | recent / novel | 8.822 | 13.104 | +2.136 | +0.954 |
| 10005 | conditional | 4 | split / clean | 8.976 | 11.835 | -0.895 | n/a |
| 10005 | conditional | 4 | split / noise | 11.024 | 17.617 | +4.888 | n/a |
| 10005 | conditional | 4 | split / return | 9.900 | 10.758 | -10.346 | n/a |
| 10005 | conditional | 4 | split / revision | 9.310 | 15.247 | +1.794 | +0.641 |
| 10005 | conditional | 1 | split / novel | 7.394 | 10.169 | +1.737 | +0.108 |
| 10005 | conditional | 2 | split / novel | 10.278 | 12.867 | +2.697 | +16.351 |
| 10005 | conditional | 3 | split / novel | 9.046 | 12.730 | -0.137 | +1.032 |
| 10005 | conditional | 4 | uniform / clean | 8.710 | 13.155 | +0.991 | n/a |
| 10005 | conditional | 4 | uniform / noise | 10.409 | 25.021 | +12.856 | n/a |
| 10005 | conditional | 4 | uniform / return | 9.647 | 9.526 | -9.267 | n/a |
| 10005 | conditional | 4 | uniform / revision | 10.068 | 18.650 | -0.744 | +0.176 |
| 10005 | conditional | 1 | uniform / novel | 7.551 | 10.590 | +2.239 | +0.361 |
| 10005 | conditional | 2 | uniform / novel | 11.466 | 11.625 | +1.035 | +12.856 |
| 10005 | conditional | 3 | uniform / novel | 9.336 | 12.164 | +0.539 | +1.179 |
| 10006 | conditional | 1 | fresh / novel | 8.238 | 18.342 | -6.687 | +18.875 |
| 10006 | conditional | 2 | fresh / novel | 9.246 | 18.683 | -6.346 | +1.136 |
| 10006 | conditional | 3 | fresh / novel | 8.929 | 17.009 | -8.020 | +0.403 |
| 10006 | conditional | 4 | recent / clean | 7.694 | 16.650 | +1.542 | n/a |
| 10006 | conditional | 4 | recent / noise | 10.947 | 17.117 | +2.009 | n/a |
| 10006 | conditional | 4 | recent / return | 8.329 | 12.910 | -7.815 | n/a |
| 10006 | conditional | 4 | recent / revision | 8.078 | 18.292 | +0.489 | +0.082 |
| 10006 | conditional | 1 | recent / novel | 8.338 | 11.045 | +3.640 | +18.619 |
| 10006 | conditional | 2 | recent / novel | 8.107 | 12.802 | +1.757 | +1.199 |
| 10006 | conditional | 3 | recent / novel | 8.118 | 15.107 | +2.306 | +0.057 |
| 10006 | conditional | 4 | split / clean | 8.059 | 21.707 | +2.314 | n/a |
| 10006 | conditional | 4 | split / noise | 10.347 | 27.727 | +8.334 | n/a |
| 10006 | conditional | 4 | split / return | 9.134 | 11.787 | -12.209 | n/a |
| 10006 | conditional | 4 | split / revision | 8.171 | 19.929 | +0.230 | +0.067 |
| 10006 | conditional | 1 | split / novel | 9.003 | 14.578 | +4.345 | +19.240 |
| 10006 | conditional | 2 | split / novel | 8.570 | 17.338 | +2.760 | +0.540 |
| 10006 | conditional | 3 | split / novel | 7.842 | 19.393 | +2.055 | -0.335 |
| 10006 | conditional | 4 | uniform / clean | 7.812 | 12.897 | +2.219 | n/a |
| 10006 | conditional | 4 | uniform / noise | 9.364 | 18.147 | +7.469 | n/a |
| 10006 | conditional | 4 | uniform / return | 8.773 | 10.594 | -4.835 | n/a |
| 10006 | conditional | 4 | uniform / revision | 8.263 | 11.967 | -1.710 | +1.470 |
| 10006 | conditional | 1 | uniform / novel | 9.157 | 8.816 | +1.877 | +16.704 |
| 10006 | conditional | 2 | uniform / novel | 8.341 | 8.828 | +0.012 | +0.874 |
| 10006 | conditional | 3 | uniform / novel | 8.295 | 10.678 | +1.851 | -0.056 |
| 10001 | recurrent | 1 | fresh / novel | 8.615 | 23.699 | -1.797 | +18.808 |
| 10001 | recurrent | 2 | fresh / novel | 8.556 | 24.069 | -1.428 | +0.154 |
| 10001 | recurrent | 3 | fresh / novel | 9.799 | 25.241 | -0.255 | +0.990 |
| 10001 | recurrent | 4 | recent / clean | 8.263 | 23.606 | +0.083 | n/a |
| 10001 | recurrent | 4 | recent / noise | 11.185 | 23.147 | -0.377 | n/a |
| 10001 | recurrent | 4 | recent / return | 8.528 | 24.275 | -1.232 | n/a |
| 10001 | recurrent | 4 | recent / revision | 8.762 | 22.302 | +0.552 | +0.829 |
| 10001 | recurrent | 1 | recent / novel | 8.347 | 22.945 | +0.280 | +17.771 |
| 10001 | recurrent | 2 | recent / novel | 7.654 | 25.266 | +2.320 | +0.026 |
| 10001 | recurrent | 3 | recent / novel | 8.786 | 23.524 | -1.742 | -0.158 |
| 10001 | recurrent | 4 | split / clean | 8.084 | 24.396 | -0.639 | n/a |
| 10001 | recurrent | 4 | split / noise | 10.677 | 23.921 | -1.114 | n/a |
| 10001 | recurrent | 4 | split / return | 9.331 | 23.111 | -3.128 | n/a |
| 10001 | recurrent | 4 | split / revision | 8.803 | 22.418 | -0.435 | +1.394 |
| 10001 | recurrent | 1 | split / novel | 8.873 | 24.317 | +8.557 | +17.243 |
| 10001 | recurrent | 2 | split / novel | 7.957 | 25.373 | +1.055 | +1.047 |
| 10001 | recurrent | 3 | split / novel | 8.587 | 25.035 | -0.338 | +0.573 |
| 10001 | recurrent | 4 | uniform / clean | 8.266 | 24.966 | +1.708 | n/a |
| 10001 | recurrent | 4 | uniform / noise | 9.887 | 24.082 | +0.824 | n/a |
| 10001 | recurrent | 4 | uniform / return | 9.119 | 15.490 | -9.661 | n/a |
| 10001 | recurrent | 4 | uniform / revision | 8.907 | 21.913 | +0.600 | +0.910 |
| 10001 | recurrent | 1 | uniform / novel | 9.286 | 23.213 | +5.930 | +16.839 |
| 10001 | recurrent | 2 | uniform / novel | 8.108 | 24.911 | +1.698 | -0.166 |
| 10001 | recurrent | 3 | uniform / novel | 8.774 | 23.258 | -1.653 | +0.579 |
| 10002 | recurrent | 1 | fresh / novel | 6.868 | 24.285 | -1.007 | +3.790 |
| 10002 | recurrent | 2 | fresh / novel | 7.343 | 24.605 | -0.687 | +0.295 |
| 10002 | recurrent | 3 | fresh / novel | 8.774 | 24.689 | -0.603 | +17.196 |
| 10002 | recurrent | 4 | recent / clean | 8.020 | 26.206 | -0.026 | n/a |
| 10002 | recurrent | 4 | recent / noise | 10.931 | 23.758 | -2.474 | n/a |
| 10002 | recurrent | 4 | recent / return | 8.996 | 24.186 | -2.109 | n/a |
| 10002 | recurrent | 4 | recent / revision | 8.343 | 23.264 | -0.889 | +1.279 |
| 10002 | recurrent | 1 | recent / novel | 6.600 | 24.557 | +0.865 | +3.047 |
| 10002 | recurrent | 2 | recent / novel | 7.152 | 25.435 | +0.878 | +0.349 |
| 10002 | recurrent | 3 | recent / novel | 9.419 | 26.232 | +0.797 | +17.435 |
| 10002 | recurrent | 4 | split / clean | 8.679 | 25.463 | +1.190 | n/a |
| 10002 | recurrent | 4 | split / noise | 11.741 | 24.877 | +0.604 | n/a |
| 10002 | recurrent | 4 | split / return | 8.572 | 20.879 | -3.764 | n/a |
| 10002 | recurrent | 4 | split / revision | 9.128 | 23.361 | -0.293 | +0.771 |
| 10002 | recurrent | 1 | split / novel | 6.783 | 22.451 | +3.533 | +2.311 |
| 10002 | recurrent | 2 | split / novel | 7.167 | 24.011 | +1.560 | +0.103 |
| 10002 | recurrent | 3 | split / novel | 10.391 | 24.273 | +0.262 | +13.562 |
| 10002 | recurrent | 4 | uniform / clean | 9.307 | 25.735 | -0.335 | n/a |
| 10002 | recurrent | 4 | uniform / noise | 11.681 | 24.530 | -1.540 | n/a |
| 10002 | recurrent | 4 | uniform / return | 8.981 | 20.779 | -4.251 | n/a |
| 10002 | recurrent | 4 | uniform / revision | 9.681 | 23.275 | -0.357 | +0.981 |
| 10002 | recurrent | 1 | uniform / novel | 7.125 | 23.732 | +2.940 | +2.338 |
| 10002 | recurrent | 2 | uniform / novel | 7.240 | 24.463 | +0.731 | +0.875 |
| 10002 | recurrent | 3 | uniform / novel | 11.901 | 26.070 | +1.607 | +10.711 |
| 10003 | recurrent | 1 | fresh / novel | 7.705 | 23.845 | -1.022 | +1.607 |
| 10003 | recurrent | 2 | fresh / novel | 9.360 | 24.305 | -0.563 | +19.617 |
| 10003 | recurrent | 3 | fresh / novel | 9.346 | 23.086 | -1.782 | +0.379 |
| 10003 | recurrent | 4 | recent / clean | 8.358 | 23.253 | -1.044 | n/a |
| 10003 | recurrent | 4 | recent / noise | 11.573 | 22.029 | -2.268 | n/a |
| 10003 | recurrent | 4 | recent / return | 9.209 | 24.709 | -1.661 | n/a |
| 10003 | recurrent | 4 | recent / revision | 8.333 | 23.356 | -0.290 | +0.976 |
| 10003 | recurrent | 1 | recent / novel | 7.511 | 23.405 | +1.208 | +1.570 |
| 10003 | recurrent | 2 | recent / novel | 9.574 | 24.964 | +1.559 | +18.522 |
| 10003 | recurrent | 3 | recent / novel | 8.474 | 24.298 | -0.667 | +0.199 |
| 10003 | recurrent | 4 | split / clean | 8.538 | 23.187 | -2.322 | n/a |
| 10003 | recurrent | 4 | split / noise | 11.106 | 22.191 | -3.319 | n/a |
| 10003 | recurrent | 4 | split / return | 10.108 | 21.604 | -4.926 | n/a |
| 10003 | recurrent | 4 | split / revision | 8.744 | 22.684 | -1.230 | +0.203 |
| 10003 | recurrent | 1 | split / novel | 7.423 | 23.526 | +3.418 | +1.122 |
| 10003 | recurrent | 2 | split / novel | 10.314 | 24.113 | +0.587 | +15.596 |
| 10003 | recurrent | 3 | split / novel | 8.930 | 25.510 | +1.396 | +0.400 |
| 10003 | recurrent | 4 | uniform / clean | 8.661 | 22.514 | -1.683 | n/a |
| 10003 | recurrent | 4 | uniform / noise | 11.014 | 22.796 | -1.401 | n/a |
| 10003 | recurrent | 4 | uniform / return | 9.272 | 18.950 | -5.918 | n/a |
| 10003 | recurrent | 4 | uniform / revision | 9.183 | 22.449 | +2.087 | +0.291 |
| 10003 | recurrent | 1 | uniform / novel | 7.931 | 16.921 | +4.201 | +1.366 |
| 10003 | recurrent | 2 | uniform / novel | 11.528 | 21.227 | +4.306 | +13.095 |
| 10003 | recurrent | 3 | uniform / novel | 9.292 | 24.197 | +2.970 | +0.191 |
| 10004 | recurrent | 1 | fresh / novel | 7.445 | 24.807 | -0.698 | +0.352 |
| 10004 | recurrent | 2 | fresh / novel | 7.639 | 25.462 | -0.042 | +3.157 |
| 10004 | recurrent | 3 | fresh / novel | 8.940 | 24.884 | -0.620 | +17.143 |
| 10004 | recurrent | 4 | recent / clean | 8.245 | 26.443 | +0.888 | n/a |
| 10004 | recurrent | 4 | recent / noise | 10.895 | 24.716 | -0.839 | n/a |
| 10004 | recurrent | 4 | recent / return | 8.732 | 24.202 | -1.693 | n/a |
| 10004 | recurrent | 4 | recent / revision | 8.643 | 24.623 | +0.579 | +0.500 |
| 10004 | recurrent | 1 | recent / novel | 7.285 | 25.153 | +1.563 | +0.594 |
| 10004 | recurrent | 2 | recent / novel | 7.199 | 25.748 | +0.595 | +2.185 |
| 10004 | recurrent | 3 | recent / novel | 9.773 | 25.555 | -0.193 | +16.422 |
| 10004 | recurrent | 4 | split / clean | 8.473 | 25.614 | -0.389 | n/a |
| 10004 | recurrent | 4 | split / noise | 11.174 | 22.750 | -3.253 | n/a |
| 10004 | recurrent | 4 | split / return | 8.706 | 15.692 | -10.294 | n/a |
| 10004 | recurrent | 4 | split / revision | 9.254 | 24.276 | -0.048 | +0.994 |
| 10004 | recurrent | 1 | split / novel | 7.248 | 24.175 | +10.616 | +0.206 |
| 10004 | recurrent | 2 | split / novel | 7.776 | 20.164 | -4.010 | +1.758 |
| 10004 | recurrent | 3 | split / novel | 10.359 | 26.003 | +5.839 | +15.774 |
| 10004 | recurrent | 4 | uniform / clean | 8.765 | 25.653 | +0.553 | n/a |
| 10004 | recurrent | 4 | uniform / noise | 10.723 | 24.963 | -0.136 | n/a |
| 10004 | recurrent | 4 | uniform / return | 10.224 | 22.434 | -2.950 | n/a |
| 10004 | recurrent | 4 | uniform / revision | 9.823 | 23.681 | +0.300 | +0.271 |
| 10004 | recurrent | 1 | uniform / novel | 7.246 | 25.126 | +14.351 | +0.259 |
| 10004 | recurrent | 2 | uniform / novel | 7.740 | 25.586 | +0.460 | +0.948 |
| 10004 | recurrent | 3 | uniform / novel | 11.777 | 25.100 | -0.487 | +13.103 |
| 10005 | recurrent | 1 | fresh / novel | 8.142 | 25.144 | -0.005 | +0.795 |
| 10005 | recurrent | 2 | fresh / novel | 9.199 | 26.435 | +1.287 | +19.138 |
| 10005 | recurrent | 3 | fresh / novel | 9.364 | 24.850 | -0.298 | +1.438 |
| 10005 | recurrent | 4 | recent / clean | 8.416 | 26.219 | +0.054 | n/a |
| 10005 | recurrent | 4 | recent / noise | 11.169 | 25.271 | -0.895 | n/a |
| 10005 | recurrent | 4 | recent / return | 9.266 | 25.470 | -1.574 | n/a |
| 10005 | recurrent | 4 | recent / revision | 9.377 | 24.396 | -0.402 | +1.085 |
| 10005 | recurrent | 1 | recent / novel | 7.600 | 26.673 | +2.857 | +0.460 |
| 10005 | recurrent | 2 | recent / novel | 9.496 | 25.531 | -1.142 | +16.766 |
| 10005 | recurrent | 3 | recent / novel | 8.490 | 26.165 | +0.634 | +1.483 |
| 10005 | recurrent | 4 | split / clean | 8.872 | 26.074 | -0.093 | n/a |
| 10005 | recurrent | 4 | split / noise | 11.729 | 25.200 | -0.967 | n/a |
| 10005 | recurrent | 4 | split / return | 11.008 | 23.852 | -3.268 | n/a |
| 10005 | recurrent | 4 | split / revision | 9.361 | 24.958 | +0.081 | +0.764 |
| 10005 | recurrent | 1 | split / novel | 7.782 | 26.926 | +4.717 | +0.092 |
| 10005 | recurrent | 2 | split / novel | 10.490 | 26.643 | -0.282 | +17.118 |
| 10005 | recurrent | 3 | split / novel | 9.202 | 26.167 | -0.477 | +1.549 |
| 10005 | recurrent | 4 | uniform / clean | 8.998 | 26.049 | -1.177 | n/a |
| 10005 | recurrent | 4 | uniform / noise | 11.188 | 25.353 | -1.873 | n/a |
| 10005 | recurrent | 4 | uniform / return | 11.338 | 20.658 | -6.109 | n/a |
| 10005 | recurrent | 4 | uniform / revision | 9.843 | 24.023 | -1.257 | +0.306 |
| 10005 | recurrent | 1 | uniform / novel | 8.211 | 25.283 | +5.339 | +0.279 |
| 10005 | recurrent | 2 | uniform / novel | 12.167 | 25.722 | +0.439 | +10.623 |
| 10005 | recurrent | 3 | uniform / novel | 9.547 | 27.226 | +1.504 | +1.505 |
| 10006 | recurrent | 1 | fresh / novel | 8.380 | 25.704 | +0.708 | +19.135 |
| 10006 | recurrent | 2 | fresh / novel | 9.104 | 24.408 | -0.587 | +0.932 |
| 10006 | recurrent | 3 | fresh / novel | 9.119 | 25.067 | +0.072 | +0.217 |
| 10006 | recurrent | 4 | recent / clean | 7.931 | 27.329 | +2.335 | n/a |
| 10006 | recurrent | 4 | recent / noise | 11.039 | 25.405 | +0.410 | n/a |
| 10006 | recurrent | 4 | recent / return | 8.717 | 23.954 | -1.975 | n/a |
| 10006 | recurrent | 4 | recent / revision | 8.178 | 23.458 | -0.185 | +0.452 |
| 10006 | recurrent | 1 | recent / novel | 8.279 | 26.192 | +2.180 | +20.092 |
| 10006 | recurrent | 2 | recent / novel | 7.871 | 25.637 | -0.555 | +1.078 |
| 10006 | recurrent | 3 | recent / novel | 7.900 | 24.995 | -0.642 | +0.277 |
| 10006 | recurrent | 4 | split / clean | 8.027 | 25.504 | -0.332 | n/a |
| 10006 | recurrent | 4 | split / noise | 10.503 | 26.171 | +0.335 | n/a |
| 10006 | recurrent | 4 | split / return | 10.094 | 22.080 | -3.988 | n/a |
| 10006 | recurrent | 4 | split / revision | 8.218 | 24.153 | +0.396 | +0.664 |
| 10006 | recurrent | 1 | split / novel | 8.914 | 23.024 | +3.358 | +17.943 |
| 10006 | recurrent | 2 | split / novel | 8.185 | 25.332 | +2.308 | +1.103 |
| 10006 | recurrent | 3 | split / novel | 7.913 | 25.837 | +0.505 | +0.057 |
| 10006 | recurrent | 4 | uniform / clean | 8.064 | 23.978 | +1.911 | n/a |
| 10006 | recurrent | 4 | uniform / noise | 10.177 | 18.476 | -3.591 | n/a |
| 10006 | recurrent | 4 | uniform / return | 9.529 | 17.333 | -7.897 | n/a |
| 10006 | recurrent | 4 | uniform / revision | 8.734 | 23.225 | +0.295 | +0.411 |
| 10006 | recurrent | 1 | uniform / novel | 9.041 | 15.253 | +5.726 | +17.821 |
| 10006 | recurrent | 2 | uniform / novel | 8.429 | 18.271 | +3.018 | +0.217 |
| 10006 | recurrent | 3 | uniform / novel | 8.377 | 22.067 | +3.796 | -0.043 |
