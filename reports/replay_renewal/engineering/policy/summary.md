# Replay renewal: policy

48 complete episodes; paired seeds [19].

Fresh qualification: **FAIL**. Thresholds: marginal gain .02, cue benefit .002.

Error values below are Brier x100. Negative contrast is improvement; intervals are paired seed bootstrap, descriptive.

| Model | Contrast | Acquisition AUC [95% interval] | Terminal valid-old difference |
|---|---|---:|---:|
| conditional | split_minus_uniform | -0.005 [-0.005, -0.005] | -0.016 |
| conditional | recent_minus_uniform | +0.007 [+0.007, +0.007] | -0.015 |
| conditional | split_minus_recent | -0.012 [-0.012, -0.012] | -0.001 |
| recurrent | split_minus_uniform | +0.009 [+0.009, +0.009] | -0.002 |
| recurrent | recent_minus_uniform | +0.004 [+0.004, +0.004] | -0.015 |
| recurrent | split_minus_recent | +0.005 [+0.005, +0.005] | +0.013 |

## Prospective policy screen

* conditional: **FAIL**. Failed gates: fresh_qualification, acquisition_gain, acquisition_consistency.
* recurrent: **FAIL**. Failed gates: fresh_qualification, acquisition_gain, acquisition_consistency.

Full final-branch contrasts, every gate and interval warning are in summary.json.

## Qualification groups

| Group | Episodes | Marginal gain | Cue benefit | Pass |
|---|---:|---:|---:|---|
| conditional_stage_1 | 1 | 0.048 | 0.002 | False |
| conditional_stage_2 | 1 | 1.487 | 0.004 | False |
| conditional_stage_3 | 1 | 0.501 | -0.001 | False |
| conditional_cue_0 | 1 | 0.048 | 0.002 | False |
| conditional_cue_1 | 1 | 0.501 | -0.001 | False |
| conditional_cue_2 | 1 | 1.487 | 0.004 | False |
| recurrent_stage_1 | 1 | -0.087 | 0.001 | False |
| recurrent_stage_2 | 1 | 1.451 | 0.004 | False |
| recurrent_stage_3 | 1 | 0.257 | -0.001 | False |
| recurrent_cue_0 | 1 | -0.087 | 0.001 | False |
| recurrent_cue_1 | 1 | 0.257 | -0.001 | False |
| recurrent_cue_2 | 1 | 1.451 | 0.004 | False |

## Every episode

| Seed | Model | Stage | Arm / branch | AUC | Valid-old error | Damage | Cue benefit |
|---|---|---:|---|---:|---:|---:|---:|
| 19 | conditional | 1 | fresh / novel | 25.887 | 24.848 | -0.115 | +0.002 |
| 19 | conditional | 2 | fresh / novel | 25.310 | 24.853 | -0.110 | +0.004 |
| 19 | conditional | 3 | fresh / novel | 25.445 | 24.858 | -0.105 | -0.001 |
| 19 | conditional | 4 | recent / clean | 24.334 | 24.402 | -0.082 | n/a |
| 19 | conditional | 4 | recent / noise | 24.335 | 24.403 | -0.082 | n/a |
| 19 | conditional | 4 | recent / return | 24.573 | 24.620 | -0.096 | n/a |
| 19 | conditional | 4 | recent / revision | 24.338 | 24.486 | -0.104 | +0.002 |
| 19 | conditional | 1 | recent / novel | 25.367 | 24.688 | -0.097 | +0.002 |
| 19 | conditional | 2 | recent / novel | 24.736 | 24.585 | -0.103 | +0.001 |
| 19 | conditional | 3 | recent / novel | 24.587 | 24.484 | -0.101 | -0.001 |
| 19 | conditional | 4 | split / clean | 24.316 | 24.398 | -0.088 | n/a |
| 19 | conditional | 4 | split / noise | 24.317 | 24.399 | -0.087 | n/a |
| 19 | conditional | 4 | split / return | 24.569 | 24.631 | -0.102 | n/a |
| 19 | conditional | 4 | split / revision | 24.316 | 24.469 | -0.120 | +0.002 |
| 19 | conditional | 1 | split / novel | 25.355 | 24.686 | -0.099 | +0.002 |
| 19 | conditional | 2 | split / novel | 24.725 | 24.584 | -0.102 | +0.001 |
| 19 | conditional | 3 | split / novel | 24.575 | 24.486 | -0.098 | -0.001 |
| 19 | conditional | 4 | uniform / clean | 24.298 | 24.440 | -0.077 | n/a |
| 19 | conditional | 4 | uniform / noise | 24.299 | 24.441 | -0.076 | n/a |
| 19 | conditional | 4 | uniform / return | 24.510 | 24.565 | -0.139 | n/a |
| 19 | conditional | 4 | uniform / revision | 24.280 | 24.475 | -0.127 | +0.003 |
| 19 | conditional | 1 | uniform / novel | 25.368 | 24.689 | -0.096 | +0.002 |
| 19 | conditional | 2 | uniform / novel | 24.741 | 24.598 | -0.091 | +0.001 |
| 19 | conditional | 3 | uniform / novel | 24.561 | 24.517 | -0.082 | -0.001 |
| 19 | recurrent | 1 | fresh / novel | 26.033 | 25.007 | -0.091 | +0.001 |
| 19 | recurrent | 2 | fresh / novel | 25.323 | 24.961 | -0.137 | +0.004 |
| 19 | recurrent | 3 | fresh / novel | 25.559 | 25.116 | +0.018 | -0.001 |
| 19 | recurrent | 4 | recent / clean | 24.386 | 24.454 | -0.101 | n/a |
| 19 | recurrent | 4 | recent / noise | 24.387 | 24.458 | -0.097 | n/a |
| 19 | recurrent | 4 | recent / return | 24.756 | 24.826 | -0.108 | n/a |
| 19 | recurrent | 4 | recent / revision | 24.388 | 24.541 | -0.117 | -0.001 |
| 19 | recurrent | 1 | recent / novel | 25.480 | 24.810 | -0.117 | +0.002 |
| 19 | recurrent | 2 | recent / novel | 24.866 | 24.675 | -0.135 | +0.004 |
| 19 | recurrent | 3 | recent / novel | 24.715 | 24.555 | -0.121 | -0.001 |
| 19 | recurrent | 4 | split / clean | 24.439 | 24.516 | -0.067 | n/a |
| 19 | recurrent | 4 | split / noise | 24.440 | 24.520 | -0.062 | n/a |
| 19 | recurrent | 4 | split / return | 24.817 | 24.917 | -0.063 | n/a |
| 19 | recurrent | 4 | split / revision | 24.460 | 24.612 | -0.079 | -0.002 |
| 19 | recurrent | 1 | split / novel | 25.468 | 24.811 | -0.120 | +0.002 |
| 19 | recurrent | 2 | split / novel | 24.873 | 24.686 | -0.125 | +0.003 |
| 19 | recurrent | 3 | split / novel | 24.735 | 24.582 | -0.104 | -0.001 |
| 19 | recurrent | 4 | uniform / clean | 24.355 | 24.468 | -0.113 | n/a |
| 19 | recurrent | 4 | uniform / noise | 24.356 | 24.472 | -0.109 | n/a |
| 19 | recurrent | 4 | uniform / return | 24.664 | 24.725 | -0.177 | n/a |
| 19 | recurrent | 4 | uniform / revision | 24.311 | 24.507 | -0.157 | +0.000 |
| 19 | recurrent | 1 | uniform / novel | 25.479 | 24.809 | -0.118 | +0.002 |
| 19 | recurrent | 2 | uniform / novel | 24.868 | 24.693 | -0.116 | +0.004 |
| 19 | recurrent | 3 | uniform / novel | 24.702 | 24.582 | -0.111 | -0.000 |
