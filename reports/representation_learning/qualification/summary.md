# Temporal-representation learning: qualification

Decision: **STOP**. Raw gate conjunction: False. Eligible scientific cohort: True.

Brier values are raw units. Intervals are descriptive paired-seed bootstrap intervals. Support replicates are averaged within seed; every required cell must pass.

| Required cell | Mean [95% interval] | Minimum | Maximum | Pass |
|---|---:|---:|---:|---|
| fresh/stage_1/marginal_gain | 0.155790 [0.152111, 0.159897] | 0.02 | None | True |
| fresh/stage_1/cue_benefit | 0.074829 [0.012794, 0.139283] | 0.002 | None | True |
| fresh/stage_2/marginal_gain | 0.152636 [0.146757, 0.158432] | 0.02 | None | True |
| fresh/stage_2/cue_benefit | 0.073691 [0.011050, 0.138707] | 0.002 | None | True |
| fresh/stage_3/marginal_gain | 0.149241 [0.144103, 0.154763] | 0.02 | None | True |
| fresh/stage_3/cue_benefit | 0.063347 [0.005311, 0.123350] | 0.002 | None | True |
| fresh/cue_0/marginal_gain | 0.152847 [0.146556, 0.159352] | 0.02 | None | True |
| fresh/cue_0/cue_benefit | 0.020677 [0.015630, 0.025522] | 0.002 | None | True |
| fresh/cue_1/marginal_gain | 0.150757 [0.144744, 0.156938] | 0.02 | None | True |
| fresh/cue_1/cue_benefit | 0.003365 [0.001134, 0.005632] | 0.002 | None | True |
| fresh/cue_2/marginal_gain | 0.154063 [0.151497, 0.157123] | 0.02 | None | True |
| fresh/cue_2/cue_benefit | 0.187825 [0.177949, 0.197500] | 0.002 | None | True |
| interleaved/slot_0/marginal_gain | 0.032550 [-0.022119, 0.087949] | 0.02 | None | True |
| interleaved/slot_0/absolute_brier | 0.195597 [0.139318, 0.251204] | None | 0.12 | False |
| interleaved/slot_1/marginal_gain | 0.117001 [0.107149, 0.126605] | 0.02 | None | True |
| interleaved/slot_1/absolute_brier | 0.112109 [0.103427, 0.121369] | None | 0.12 | True |
| interleaved/slot_2/marginal_gain | 0.128607 [0.120359, 0.136292] | 0.02 | None | True |
| interleaved/slot_2/absolute_brier | 0.098716 [0.089621, 0.108094] | None | 0.12 | True |
| interleaved/stage_1/cue_0/cue_benefit | 0.009599 [0.009154, 0.010043] | 0.002 | None | True |
| interleaved/stage_1/cue_1/cue_benefit | -0.000152 [-0.001405, 0.001101] | 0.002 | None | False |
| interleaved/stage_1/cue_2/cue_benefit | 0.160492 [0.160101, 0.160883] | 0.002 | None | True |
| interleaved/slot_3/marginal_gain | 0.131254 [0.124483, 0.136961] | 0.02 | None | True |
| interleaved/slot_3/absolute_brier | 0.095619 [0.088279, 0.103650] | None | 0.12 | True |
| interleaved/stage_2/cue_0/cue_benefit | 0.005775 [0.003019, 0.007625] | 0.002 | None | True |
| interleaved/stage_2/cue_1/cue_benefit | 0.000802 [-0.001974, 0.003578] | 0.002 | None | False |
| interleaved/stage_2/cue_2/cue_benefit | 0.140396 [0.119800, 0.160993] | 0.002 | None | True |
| interleaved/slot_4/marginal_gain | 0.128245 [0.120746, 0.135205] | 0.02 | None | True |
| interleaved/slot_4/absolute_brier | 0.097349 [0.090339, 0.104924] | None | 0.12 | True |
| interleaved/stage_3/cue_0/cue_benefit | 0.005634 [0.001645, 0.010763] | 0.002 | None | True |
| interleaved/stage_3/cue_1/cue_benefit | -0.000291 [-0.002022, 0.001362] | 0.002 | None | False |
| interleaved/stage_3/cue_2/cue_benefit | 0.120648 [0.094298, 0.146385] | 0.002 | None | True |

A qualification failure stops the candidate comparison. A main failure stops this fixed reconstruction recipe. No subgroup promotion, exposure increase, or auxiliary-weight rescue.
