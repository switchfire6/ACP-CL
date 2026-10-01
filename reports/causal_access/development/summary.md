# Causal access to fixed predictions

Frozen prediction access only. Qualification and every primary gate must pass per architecture; intervals do not establish statistical noninferiority. No cue exclusion, optimizer update, recognition-delay exclusion, or test-score tuning.

Brier differences below use raw units; negative improves. Intervals are descriptive paired-seed bootstrap 95% intervals. All transitions and recognition delays are scored.

| Architecture | Qualified | Raw primary pass | Decision |
|---|---|---|---|
| conditional | False | False | INELIGIBLE |
| recurrent | False | False | INELIGIBLE |

| Architecture | Causal minus comparator | Mean [95% interval] | Lower error seeds |
|---|---|---:|---:|
| conditional | whole_brier_minus_full | -0.004563 [-0.017461, +0.006419] | 5/6 |
| conditional | whole_brier_minus_half | +0.000787 [-0.004118, +0.006833] | 3/6 |
| conditional | whole_brier_minus_joint | -0.035006 [-0.064518, -0.016131] | 6/6 |
| conditional | old_brier_minus_full | -0.011184 [-0.035318, +0.009041] | 5/6 |
| conditional | novel_focus_brier_minus_full | +0.002581 [+0.001191, +0.003847] | 1/6 |
| recurrent | whole_brier_minus_full | -0.039696 [-0.054156, -0.025737] | 6/6 |
| recurrent | whole_brier_minus_half | -0.010461 [-0.015978, -0.004963] | 5/6 |
| recurrent | whole_brier_minus_joint | -0.082046 [-0.093568, -0.073611] | 6/6 |
| recurrent | old_brier_minus_full | -0.084218 [-0.113562, -0.056175] | 6/6 |
| recurrent | novel_focus_brier_minus_full | +0.005512 [+0.004186, +0.006841] | 0/6 |

Qualification failures and raw gate failures:

- conditional: overall/core_old_excess, cue_1/core_old_excess, cue_1/specialist_novel_cue_gain, whole_vs_half, whole_vs_half_consistency, novel_cue_all_groups.
- recurrent: overall/core_old_excess, cue_0/core_old_excess, cue_1/core_old_excess, cue_1/full_novel_cue_gain, cue_2/core_old_excess, novel_focus_vs_full, novel_cue_all_groups.
