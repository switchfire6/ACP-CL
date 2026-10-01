# Acquisition diagnostic results

Kind: development; 2 independent seeds; 12 episodes.
Fresh-model qualification: **True**.

## Qualification

| Group | Marginal Brier improvement | Correct-cue Brier benefit | Pass |
|---|---:|---:|---|
| conditional_stage_1 | 0.15842 | 0.00658 | True |
| conditional_stage_2 | 0.14784 | 0.07445 | True |
| conditional_stage_3 | 0.13673 | 0.06943 | True |
| conditional_cue_0 | 0.15414 | 0.00588 | True |
| conditional_cue_1 | 0.14797 | 0.00291 | True |
| conditional_cue_2 | 0.14088 | 0.14166 | True |
| recurrent_stage_1 | 0.16048 | 0.00553 | True |
| recurrent_stage_2 | 0.14752 | 0.08875 | True |
| recurrent_stage_3 | 0.13584 | 0.06039 | True |
| recurrent_cue_0 | 0.15418 | 0.00702 | True |
| recurrent_cue_1 | 0.15169 | 0.00250 | True |
| recurrent_cue_2 | 0.13797 | 0.14515 | True |

## Every episode

Brier errors below are multiplied by 100; lower is better. Positive valid damage is forgetting.

| Seed | Model | Stage | Arm/branch | Cue | Brier AUC | Valid damage | Cue benefit |
|---|---|---:|---|---|---:|---:|---:|
| 211 | conditional | 1 | fresh/novel | 0 | 9.746 | -7.601 | +0.827 |
| 211 | conditional | 2 | fresh/novel | 2 | 11.868 | -5.637 | +14.540 |
| 211 | conditional | 3 | fresh/novel | 1 | 12.021 | -7.188 | +0.094 |
| 212 | conditional | 1 | fresh/novel | 1 | 9.029 | -7.704 | +0.488 |
| 212 | conditional | 2 | fresh/novel | 0 | 10.689 | -8.814 | +0.349 |
| 212 | conditional | 3 | fresh/novel | 2 | 12.856 | -8.083 | +13.792 |
| 211 | recurrent | 1 | fresh/novel | 0 | 9.786 | +1.401 | +0.776 |
| 211 | recurrent | 2 | fresh/novel | 2 | 11.699 | +1.202 | +17.121 |
| 211 | recurrent | 3 | fresh/novel | 1 | 11.028 | +0.295 | +0.169 |
| 212 | recurrent | 1 | fresh/novel | 1 | 8.913 | -2.927 | +0.331 |
| 212 | recurrent | 2 | fresh/novel | 0 | 10.432 | -2.740 | +0.628 |
| 212 | recurrent | 3 | fresh/novel | 2 | 12.620 | -3.292 | +11.909 |
