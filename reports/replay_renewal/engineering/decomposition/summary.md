# Replay renewal: decomposition

14 complete episodes; paired seeds [19].

Fresh qualification: **FAIL**. Thresholds: marginal gain .02, cue benefit .002.

Error values below are Brier x100. Negative contrast is improvement; intervals are paired seed bootstrap, descriptive.

| Model | Contrast | Acquisition AUC [95% interval] | Terminal valid-old difference |
|---|---|---:|---:|
| conditional | clear_minus_keep | -0.002 [-0.002, -0.002] | -0.004 |
| conditional | clear_rebase_minus_rebase | -0.002 [-0.002, -0.002] | -0.004 |
| conditional | rebase_minus_keep | +0.000 [+0.000, +0.000] | +0.000 |
| conditional | clear_rebase_minus_clear | +0.000 [+0.000, +0.000] | +0.000 |
| conditional | clear_main | -0.002 [-0.002, -0.002] | -0.004 |
| conditional | rebase_main | +0.000 [+0.000, +0.000] | +0.000 |
| conditional | interaction | +0.000 [+0.000, +0.000] | +0.000 |
| conditional | rng_reset_minus_keep | -0.016 [-0.016, -0.016] | -0.014 |
| conditional | full_reset_minus_clear_rebase | +0.000 [+0.000, +0.000] | +0.000 |
| conditional | full_reset_minus_keep | -0.002 [-0.002, -0.002] | -0.004 |
| conditional | keep_minus_fresh | -0.520 [-0.520, -0.520] | -0.159 |
| recurrent | clear_minus_keep | +0.004 [+0.004, +0.004] | +0.003 |
| recurrent | clear_rebase_minus_rebase | +0.004 [+0.004, +0.004] | +0.003 |
| recurrent | rebase_minus_keep | +0.000 [+0.000, +0.000] | +0.000 |
| recurrent | clear_rebase_minus_clear | +0.000 [+0.000, +0.000] | +0.000 |
| recurrent | clear_main | +0.004 [+0.004, +0.004] | +0.003 |
| recurrent | rebase_main | +0.000 [+0.000, +0.000] | +0.000 |
| recurrent | interaction | +0.000 [+0.000, +0.000] | +0.000 |
| recurrent | rng_reset_minus_keep | -0.029 [-0.029, -0.029] | -0.043 |
| recurrent | full_reset_minus_clear_rebase | +0.000 [+0.000, +0.000] | +0.000 |
| recurrent | full_reset_minus_keep | +0.004 [+0.004, +0.004] | +0.003 |
| recurrent | keep_minus_fresh | -0.553 [-0.553, -0.553] | -0.198 |

## Qualification groups

| Group | Episodes | Marginal gain | Cue benefit | Pass |
|---|---:|---:|---:|---|
| conditional_stage_1 | 1 | 0.048 | 0.002 | False |
| conditional_cue_0 | 1 | 0.048 | 0.002 | False |
| recurrent_stage_1 | 1 | -0.087 | 0.001 | False |
| recurrent_cue_0 | 1 | -0.087 | 0.001 | False |

## Every episode

| Seed | Model | Stage | Arm / branch | AUC | Valid-old error | Damage | Cue benefit |
|---|---|---:|---|---:|---:|---:|---:|
| 19 | conditional | 1 | clear / novel | 25.366 | 24.685 | -0.100 | +0.002 |
| 19 | conditional | 1 | clear_rebase / novel | 25.366 | 24.685 | -0.100 | +0.002 |
| 19 | conditional | 1 | fresh / novel | 25.887 | 24.848 | -0.115 | +0.002 |
| 19 | conditional | 1 | full_reset / novel | 25.366 | 24.685 | -0.100 | +0.002 |
| 19 | conditional | 1 | keep / novel | 25.368 | 24.689 | -0.096 | +0.002 |
| 19 | conditional | 1 | rebase / novel | 25.368 | 24.689 | -0.096 | +0.002 |
| 19 | conditional | 1 | rng_reset / novel | 25.352 | 24.674 | -0.111 | +0.002 |
| 19 | recurrent | 1 | clear / novel | 25.483 | 24.812 | -0.115 | +0.002 |
| 19 | recurrent | 1 | clear_rebase / novel | 25.483 | 24.812 | -0.115 | +0.002 |
| 19 | recurrent | 1 | fresh / novel | 26.033 | 25.007 | -0.091 | +0.001 |
| 19 | recurrent | 1 | full_reset / novel | 25.483 | 24.812 | -0.115 | +0.002 |
| 19 | recurrent | 1 | keep / novel | 25.479 | 24.809 | -0.118 | +0.002 |
| 19 | recurrent | 1 | rebase / novel | 25.479 | 24.809 | -0.118 | +0.002 |
| 19 | recurrent | 1 | rng_reset / novel | 25.450 | 24.766 | -0.162 | +0.002 |
