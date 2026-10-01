# Temporal-representation learning: qualification

Decision: **INELIGIBLE**. Raw gate conjunction: False. Eligible scientific cohort: False.

Brier values are raw units. Intervals are descriptive paired-seed bootstrap intervals. Support replicates are averaged within seed; every required cell must pass.

| Required cell | Mean [95% interval] | Minimum | Maximum | Pass |
|---|---:|---:|---:|---|
| fresh/stage_1/marginal_gain | 0.005961 [0.005961, 0.005961] | 0.02 | None | False |
| fresh/stage_1/cue_benefit | 0.000006 [0.000006, 0.000006] | 0.002 | None | False |
| fresh/stage_2/marginal_gain | -0.011983 [-0.011983, -0.011983] | 0.02 | None | False |
| fresh/stage_2/cue_benefit | -0.000002 [-0.000002, -0.000002] | 0.002 | None | False |
| fresh/stage_3/marginal_gain | -0.017573 [-0.017573, -0.017573] | 0.02 | None | False |
| fresh/stage_3/cue_benefit | -0.000001 [-0.000001, -0.000001] | 0.002 | None | False |
| fresh/cue_0/marginal_gain | -0.017573 [-0.017573, -0.017573] | 0.02 | None | False |
| fresh/cue_0/cue_benefit | -0.000001 [-0.000001, -0.000001] | 0.002 | None | False |
| fresh/cue_1/marginal_gain | 0.005961 [0.005961, 0.005961] | 0.02 | None | False |
| fresh/cue_1/cue_benefit | 0.000006 [0.000006, 0.000006] | 0.002 | None | False |
| fresh/cue_2/marginal_gain | -0.011983 [-0.011983, -0.011983] | 0.02 | None | False |
| fresh/cue_2/cue_benefit | -0.000002 [-0.000002, -0.000002] | 0.002 | None | False |
| interleaved/slot_0/marginal_gain | 0.021597 [0.021597, 0.021597] | 0.02 | None | True |
| interleaved/slot_0/absolute_brier | 0.238028 [0.238028, 0.238028] | None | 0.12 | False |
| interleaved/slot_1/marginal_gain | 0.007726 [0.007726, 0.007726] | 0.02 | None | False |
| interleaved/slot_1/absolute_brier | 0.236986 [0.236986, 0.236986] | None | 0.12 | False |
| interleaved/slot_2/marginal_gain | 0.001974 [0.001974, 0.001974] | 0.02 | None | False |
| interleaved/slot_2/absolute_brier | 0.236550 [0.236550, 0.236550] | None | 0.12 | False |
| interleaved/stage_1/cue_1/cue_benefit | 0.000006 [0.000006, 0.000006] | 0.002 | None | False |
| interleaved/slot_3/marginal_gain | 0.012252 [0.012252, 0.012252] | 0.02 | None | False |
| interleaved/slot_3/absolute_brier | 0.232540 [0.232540, 0.232540] | None | 0.12 | False |
| interleaved/stage_2/cue_1/cue_benefit | 0.000011 [0.000011, 0.000011] | 0.002 | None | False |
| interleaved/stage_2/cue_2/cue_benefit | 0.000006 [0.000006, 0.000006] | 0.002 | None | False |
| interleaved/slot_4/marginal_gain | 0.003086 [0.003086, 0.003086] | 0.02 | None | False |
| interleaved/slot_4/absolute_brier | 0.233000 [0.233000, 0.233000] | None | 0.12 | False |
| interleaved/stage_3/cue_0/cue_benefit | 0.000002 [0.000002, 0.000002] | 0.002 | None | False |
| interleaved/stage_3/cue_1/cue_benefit | 0.000011 [0.000011, 0.000011] | 0.002 | None | False |
| interleaved/stage_3/cue_2/cue_benefit | 0.000007 [0.000007, 0.000007] | 0.002 | None | False |

A qualification failure stops the candidate comparison. A main failure stops this fixed reconstruction recipe. No subgroup promotion, exposure increase, or auxiliary-weight rescue.
