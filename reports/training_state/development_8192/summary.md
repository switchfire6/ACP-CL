# Optimizer-by-replay diagnostic results

Kind: development; 6 independent seeds; 36 episodes.
Fresh-model qualification: **True**; required cue benefit 0.0025.

## Qualification

| Group | Marginal Brier improvement | Cue benefit | Pass |
|---|---:|---:|---|
| conditional_stage_1 | 0.151892 | 0.069743 | True |
| conditional_stage_2 | 0.149047 | 0.071537 | True |
| conditional_stage_3 | 0.143125 | 0.069820 | True |
| conditional_cue_0 | 0.153169 | 0.015921 | True |
| conditional_cue_1 | 0.145943 | 0.004232 | True |
| conditional_cue_2 | 0.144952 | 0.190947 | True |
| recurrent_stage_1 | 0.147923 | 0.068083 | True |
| recurrent_stage_2 | 0.147308 | 0.070453 | True |
| recurrent_stage_3 | 0.142703 | 0.068527 | True |
| recurrent_cue_0 | 0.149436 | 0.015682 | True |
| recurrent_cue_1 | 0.148486 | 0.004222 | True |
| recurrent_cue_2 | 0.140013 | 0.187160 | True |

## Every episode

Brier errors multiplied by 100; lower is better. Positive valid damage is deterioration.

| Seed | Model | Stage | Arm/branch | Cue | Brier AUC | Valid damage | Cue benefit |
|---|---|---:|---|---|---:|---:|---:|
| 411 | conditional | 1 | fresh/novel | 1 | 8.560 | -3.596 | +0.727 |
| 411 | conditional | 2 | fresh/novel | 2 | 9.592 | -2.087 | +20.111 |
| 411 | conditional | 3 | fresh/novel | 0 | 9.385 | -4.500 | +1.402 |
| 412 | conditional | 1 | fresh/novel | 2 | 9.225 | -5.875 | +17.635 |
| 412 | conditional | 2 | fresh/novel | 0 | 9.283 | -6.465 | +0.921 |
| 412 | conditional | 3 | fresh/novel | 1 | 9.320 | -7.300 | +0.956 |
| 413 | conditional | 1 | fresh/novel | 2 | 8.705 | -7.075 | +19.108 |
| 413 | conditional | 2 | fresh/novel | 1 | 8.705 | -3.667 | +0.518 |
| 413 | conditional | 3 | fresh/novel | 0 | 9.017 | -5.186 | +0.483 |
| 414 | conditional | 1 | fresh/novel | 0 | 7.424 | -5.148 | +2.278 |
| 414 | conditional | 2 | fresh/novel | 1 | 8.001 | -5.967 | -0.107 |
| 414 | conditional | 3 | fresh/novel | 2 | 8.680 | -4.565 | +19.397 |
| 415 | conditional | 1 | fresh/novel | 0 | 7.603 | -6.839 | +2.211 |
| 415 | conditional | 2 | fresh/novel | 2 | 9.579 | -6.605 | +19.222 |
| 415 | conditional | 3 | fresh/novel | 1 | 9.067 | -6.728 | +0.560 |
| 416 | conditional | 1 | fresh/novel | 1 | 8.064 | -5.238 | -0.113 |
| 416 | conditional | 2 | fresh/novel | 0 | 7.792 | -5.558 | +2.257 |
| 416 | conditional | 3 | fresh/novel | 2 | 8.990 | -6.879 | +19.094 |
| 411 | recurrent | 1 | fresh/novel | 1 | 8.491 | +0.621 | +0.961 |
| 411 | recurrent | 2 | fresh/novel | 2 | 9.766 | -1.456 | +20.301 |
| 411 | recurrent | 3 | fresh/novel | 0 | 9.390 | -0.869 | +1.444 |
| 412 | recurrent | 1 | fresh/novel | 2 | 8.738 | +0.490 | +15.580 |
| 412 | recurrent | 2 | fresh/novel | 0 | 9.421 | -0.523 | +0.735 |
| 412 | recurrent | 3 | fresh/novel | 1 | 9.439 | -0.852 | +0.331 |
| 413 | recurrent | 1 | fresh/novel | 2 | 8.680 | +0.620 | +19.620 |
| 413 | recurrent | 2 | fresh/novel | 1 | 9.080 | +1.372 | +0.301 |
| 413 | recurrent | 3 | fresh/novel | 0 | 8.839 | +0.919 | +1.426 |
| 414 | recurrent | 1 | fresh/novel | 0 | 7.444 | -1.231 | +2.172 |
| 414 | recurrent | 2 | fresh/novel | 1 | 8.105 | -1.312 | +0.340 |
| 414 | recurrent | 3 | fresh/novel | 2 | 8.826 | -1.231 | +18.893 |
| 415 | recurrent | 1 | fresh/novel | 0 | 7.687 | -1.895 | +1.946 |
| 415 | recurrent | 2 | fresh/novel | 2 | 9.249 | +0.435 | +18.908 |
| 415 | recurrent | 3 | fresh/novel | 1 | 9.129 | -1.573 | +0.029 |
| 416 | recurrent | 1 | fresh/novel | 1 | 8.018 | -2.080 | +0.570 |
| 416 | recurrent | 2 | fresh/novel | 0 | 7.921 | -2.701 | +1.687 |
| 416 | recurrent | 3 | fresh/novel | 2 | 8.997 | -1.243 | +18.994 |
