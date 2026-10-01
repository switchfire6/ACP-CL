# Saved core/residual output diagnostic

Saved-model intervention, not new training or prospective confirmation. Component-only predictions are ablations of a combined predictor; the residual is trained as a correction, not a standalone competitor. Refreshed support uses evaluator-supplied context, not autonomous context detection. Conditional crossed cells give an exact loss decomposition with interaction, not a unique causal cause of learning failure.

All Brier values below are multiplied by 100. Intervals are descriptive paired-seed bootstrap intervals; two refreshed supports are averaged within each seed. Full raw levels for every component and mask are in summary.json.

| Architecture | Effort allocation | Old harm: full − core | Novel benefit: core − full |
|---|---|---:|---:|
| conditional | GO | +1.2873 [+0.4705, +1.9260] | +1.8927 [+0.6828, +3.3130] |
| recurrent | GO | +8.4727 [+7.4380, +9.6201] | +2.9784 [+1.5568, +4.5098] |

Old harm uses return-entry all-case Brier; novel benefit uses novel-end affected-subset Brier. Both use separate-arm weights and refreshed support. GO requires respective means ≥0.5000 and ≥0.2000 table units, each positive in at least five of six seeds. Smoke is ineligible.

| Architecture | Actual separate − joint | Separate actual − refreshed | Refreshed separate − joint | Refresh gain ≥0.5000 |
|---|---:|---:|---:|---:|
| conditional | -4.8635 [-8.7035, -1.3964] | +25.9603 [+20.8719, +30.6293] | -3.8584 [-8.8130, -0.6975] | 6/6 |
| recurrent | -2.9918 [-5.4522, -1.2622] | +4.5776 [+0.2685, +10.3016] | -4.2691 [-10.3984, -0.1742] | 4/6 |

Only when GO fails, CONTEXT_PIVOT requires positive mean actual disadvantage, mean refresh gain ≥0.5000 and that gain in ≥5/6 seeds, and refreshed gap ≤0.5000. Otherwise STOP. No alternate endpoints or subgroup decisions.

| Conditional separate, all-case Brier | Context | Content at core route | Route on core content | Interaction | Total full − core |
|---|---|---:|---:|---:|---:|
| novel_after_return | actual | +2.1459 | +0.2922 | +0.1057 | +2.5437 |
| novel_after_return | refreshed | -0.2665 | -0.4685 | -0.0481 | -0.7831 |
| novel_end | actual | -2.3694 | +1.0791 | -0.1630 | -1.4533 |
| novel_end | refreshed | -1.7343 | -0.0317 | +0.0090 | -1.7569 |
| return_end | actual | -1.2038 | -0.0020 | +0.0719 | -1.1339 |
| return_end | refreshed | -1.1838 | +0.0772 | +0.2202 | -0.8864 |
| return_entry | actual | +2.8302 | -0.8734 | +0.0632 | +2.0200 |
| return_entry | refreshed | +1.0797 | +0.0731 | +0.1345 | +1.2873 |

For each seed: content = loss(full content, core route) − loss(core); route = loss(core content, full route) − loss(core); interaction = loss(full) − loss(full content, core route) − loss(core content, full route) + loss(core). The three terms sum to full − core. Negative error differences improve prediction; survival signs have the opposite preference.
