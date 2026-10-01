# Causal access to fixed predictions

Frozen prediction access only. Qualification and every primary gate must pass per architecture; intervals do not establish statistical noninferiority. No cue exclusion, optimizer update, recognition-delay exclusion, or test-score tuning.

Brier differences below use raw units; negative improves. Intervals are descriptive paired-seed bootstrap 95% intervals. All transitions and recognition delays are scored.

| Architecture | Qualified | Raw primary pass | Decision |
|---|---|---|---|
| conditional | False | False | INELIGIBLE |
| recurrent | False | False | INELIGIBLE |

| Architecture | Causal minus comparator | Mean [95% interval] | Lower error seeds |
|---|---|---:|---:|
| conditional | whole_brier_minus_full | +0.001690 [+0.001690, +0.001690] | 0/1 |
| conditional | whole_brier_minus_half | -0.000043 [-0.000043, -0.000043] | 1/1 |
| conditional | whole_brier_minus_joint | +0.007892 [+0.007892, +0.007892] | 0/1 |
| conditional | old_brier_minus_full | +0.001588 [+0.001588, +0.001588] | 0/1 |
| conditional | novel_focus_brier_minus_full | +0.001405 [+0.001405, +0.001405] | 0/1 |
| recurrent | whole_brier_minus_full | +0.002305 [+0.002305, +0.002305] | 0/1 |
| recurrent | whole_brier_minus_half | -0.000312 [-0.000312, -0.000312] | 1/1 |
| recurrent | whole_brier_minus_joint | +0.007308 [+0.007308, +0.007308] | 0/1 |
| recurrent | old_brier_minus_full | +0.002204 [+0.002204, +0.002204] | 0/1 |
| recurrent | novel_focus_brier_minus_full | +0.002001 [+0.002001, +0.002001] | 0/1 |

Qualification failures and raw gate failures:

- conditional: overall/specialist_old_brier, overall/specialist_novel_brier, overall/old_marginal_gain, overall/novel_marginal_gain, overall/specialist_novel_cue_gain, overall/full_novel_cue_gain, cue_1/specialist_old_brier, cue_1/specialist_novel_brier, cue_1/old_marginal_gain, cue_1/novel_marginal_gain, cue_1/specialist_novel_cue_gain, cue_1/full_novel_cue_gain, whole_vs_full, whole_vs_full_consistency, whole_vs_half, whole_vs_half_consistency, whole_vs_joint, old_survival_vs_joint, novel_survival_vs_joint, novel_cue_all_groups.
- recurrent: overall/specialist_old_brier, overall/specialist_novel_brier, overall/old_marginal_gain, overall/novel_marginal_gain, overall/specialist_novel_cue_gain, overall/full_novel_cue_gain, cue_1/specialist_old_brier, cue_1/specialist_novel_brier, cue_1/old_marginal_gain, cue_1/novel_marginal_gain, cue_1/specialist_novel_cue_gain, cue_1/full_novel_cue_gain, whole_vs_full, whole_vs_full_consistency, whole_vs_half, whole_vs_half_consistency, whole_vs_joint, novel_cue_all_groups.
