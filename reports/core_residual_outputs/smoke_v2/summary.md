# Saved core/residual output diagnostic

Saved-model intervention, not new training or prospective confirmation. Component-only predictions are ablations of a combined predictor; the residual is trained as a correction, not a standalone competitor. Refreshed support uses evaluator-supplied context, not autonomous context detection. Conditional crossed cells give an exact loss decomposition with interaction, not a unique causal cause of learning failure.

All Brier values below are multiplied by 100. Intervals are descriptive paired-seed bootstrap intervals; two refreshed supports are averaged within each seed. Full raw levels for every component and mask are in summary.json.

| Architecture | Effort allocation | Old harm: full − core | Novel benefit: core − full |
|---|---|---:|---:|
| conditional | INELIGIBLE | -0.5682 [-0.5682, -0.5682] | +0.4725 [+0.4725, +0.4725] |
| recurrent | INELIGIBLE | -0.4917 [-0.4917, -0.4917] | +0.3695 [+0.3695, +0.3695] |

Old harm uses return-entry all-case Brier; novel benefit uses novel-end affected-subset Brier. Both use separate-arm weights and refreshed support. GO requires respective means ≥0.5000 and ≥0.2000 table units, each positive in at least five of six seeds. Smoke is ineligible.

| Architecture | Actual separate − joint | Separate actual − refreshed | Refreshed separate − joint | Refresh gain ≥0.5000 |
|---|---:|---:|---:|---:|
| conditional | +0.9427 [+0.9427, +0.9427] | +0.0984 [+0.0984, +0.0984] | +0.8859 [+0.8859, +0.8859] | 0/1 |
| recurrent | +1.0888 [+1.0888, +1.0888] | +0.1110 [+0.1110, +0.1110] | +0.9922 [+0.9922, +0.9922] | 0/1 |

Only when GO fails, CONTEXT_PIVOT requires positive mean actual disadvantage, mean refresh gain ≥0.5000 and that gain in ≥5/6 seeds, and refreshed gap ≤0.5000. Otherwise STOP. No alternate endpoints or subgroup decisions.

| Conditional separate, all-case Brier | Context | Content at core route | Route on core content | Interaction | Total full − core |
|---|---|---:|---:|---:|---:|
| novel_after_return | actual | -0.6413 | +0.0042 | +0.0032 | -0.6339 |
| novel_after_return | refreshed | -0.6365 | -0.0018 | +0.0010 | -0.6372 |
| novel_end | actual | -0.4340 | +0.0074 | +0.0004 | -0.4262 |
| novel_end | refreshed | -0.4493 | -0.0031 | +0.0010 | -0.4514 |
| return_end | actual | -0.7893 | +0.0019 | +0.0048 | -0.7825 |
| return_end | refreshed | -0.7879 | -0.0015 | -0.0047 | -0.7941 |
| return_entry | actual | -0.5478 | +0.0109 | -0.0004 | -0.5374 |
| return_entry | refreshed | -0.5620 | -0.0020 | -0.0041 | -0.5682 |

For each seed: content = loss(full content, core route) − loss(core); route = loss(core content, full route) − loss(core); interaction = loss(full) − loss(full content, core route) − loss(core content, full route) + loss(core). The three terms sum to full − core. Negative error differences improve prediction; survival signs have the opposite preference.
