# Optimizer-by-replay diagnostic results

Kind: development; 6 independent seeds; 36 episodes.
Fresh-model qualification: **False**; required cue benefit 0.0025.

## Qualification

| Group | Marginal Brier improvement | Cue benefit | Pass |
|---|---:|---:|---|
| conditional_stage_1 | 0.153536 | 0.065803 | True |
| conditional_stage_2 | 0.149898 | 0.063598 | True |
| conditional_stage_3 | 0.139251 | 0.062115 | True |
| conditional_cue_0 | 0.151473 | 0.011580 | True |
| conditional_cue_1 | 0.145729 | 0.001764 | False |
| conditional_cue_2 | 0.145483 | 0.178172 | True |
| recurrent_stage_1 | 0.152569 | 0.066345 | True |
| recurrent_stage_2 | 0.148518 | 0.065523 | True |
| recurrent_stage_3 | 0.143449 | 0.059218 | True |
| recurrent_cue_0 | 0.149716 | 0.011160 | True |
| recurrent_cue_1 | 0.149110 | 0.000707 | False |
| recurrent_cue_2 | 0.145711 | 0.179220 | True |

## Every episode

Brier errors multiplied by 100; lower is better. Positive valid damage is deterioration.

| Seed | Model | Stage | Arm/branch | Cue | Brier AUC | Valid damage | Cue benefit |
|---|---|---:|---|---|---:|---:|---:|
| 411 | conditional | 1 | fresh/novel | 1 | 9.135 | -6.909 | +0.232 |
| 411 | conditional | 2 | fresh/novel | 2 | 10.686 | -3.912 | +18.965 |
| 411 | conditional | 3 | fresh/novel | 0 | 10.615 | -3.905 | +1.423 |
| 412 | conditional | 1 | fresh/novel | 2 | 10.429 | -7.567 | +18.953 |
| 412 | conditional | 2 | fresh/novel | 0 | 10.522 | -7.140 | +0.682 |
| 412 | conditional | 3 | fresh/novel | 1 | 10.700 | -8.971 | +0.055 |
| 413 | conditional | 1 | fresh/novel | 2 | 9.745 | -8.014 | +17.664 |
| 413 | conditional | 2 | fresh/novel | 1 | 9.993 | -6.694 | +0.397 |
| 413 | conditional | 3 | fresh/novel | 0 | 9.786 | -5.996 | +0.605 |
| 414 | conditional | 1 | fresh/novel | 0 | 8.176 | -4.773 | +1.785 |
| 414 | conditional | 2 | fresh/novel | 1 | 8.580 | -8.550 | +0.136 |
| 414 | conditional | 3 | fresh/novel | 2 | 9.881 | -5.146 | +19.416 |
| 415 | conditional | 1 | fresh/novel | 0 | 7.877 | -6.282 | +0.946 |
| 415 | conditional | 2 | fresh/novel | 2 | 10.773 | -8.145 | +16.473 |
| 415 | conditional | 3 | fresh/novel | 1 | 10.316 | -8.323 | +0.337 |
| 416 | conditional | 1 | fresh/novel | 1 | 8.544 | -5.856 | -0.099 |
| 416 | conditional | 2 | fresh/novel | 0 | 8.483 | -5.199 | +1.507 |
| 416 | conditional | 3 | fresh/novel | 2 | 9.848 | -3.927 | +15.432 |
| 411 | recurrent | 1 | fresh/novel | 1 | 8.897 | -0.748 | +0.172 |
| 411 | recurrent | 2 | fresh/novel | 2 | 10.587 | +0.072 | +19.114 |
| 411 | recurrent | 3 | fresh/novel | 0 | 10.530 | -1.053 | +1.120 |
| 412 | recurrent | 1 | fresh/novel | 2 | 9.806 | -0.443 | +19.691 |
| 412 | recurrent | 2 | fresh/novel | 0 | 10.876 | +0.642 | +0.810 |
| 412 | recurrent | 3 | fresh/novel | 1 | 10.709 | -1.292 | -0.043 |
| 413 | recurrent | 1 | fresh/novel | 2 | 9.519 | -1.431 | +17.425 |
| 413 | recurrent | 2 | fresh/novel | 1 | 9.790 | +0.751 | +0.035 |
| 413 | recurrent | 3 | fresh/novel | 0 | 9.810 | +0.117 | +1.082 |
| 414 | recurrent | 1 | fresh/novel | 0 | 7.987 | -0.330 | +1.301 |
| 414 | recurrent | 2 | fresh/novel | 1 | 8.648 | -1.053 | +0.232 |
| 414 | recurrent | 3 | fresh/novel | 2 | 9.491 | -2.129 | +19.445 |
| 415 | recurrent | 1 | fresh/novel | 0 | 8.005 | -2.301 | +1.253 |
| 415 | recurrent | 2 | fresh/novel | 2 | 10.501 | -1.738 | +17.992 |
| 415 | recurrent | 3 | fresh/novel | 1 | 10.081 | -0.150 | +0.063 |
| 416 | recurrent | 1 | fresh/novel | 1 | 8.320 | -1.055 | -0.033 |
| 416 | recurrent | 2 | fresh/novel | 0 | 8.413 | -2.764 | +1.131 |
| 416 | recurrent | 3 | fresh/novel | 2 | 9.856 | -3.566 | +13.864 |
