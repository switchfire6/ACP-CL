# Learned core and adaptive raw-input residual

Prospective fixed development comparison. Brier differences below are raw Brier units times 100; survival differences are percentage points. Negative Brier favors the first arm.

Six independent seeds per architecture. Curves/supports/actions/horizons are not replicates. 20,000 paired seed bootstrap resamples, RNG 27192026; intervals are descriptive. Held-out post-update AUC and performed-action prequential error are distinct outcomes.

## Fixed primary outcome and feature attribution

| Architecture | Novel separate−joint [95% interval] | Improved seeds | Primary | Feature attribution |
|---|---:|---:|---|---|
| conditional | +1.6853 [-0.6848, +4.2192] | 4/6 | Fail | Fail |
| recurrent | +0.3917 [-0.0877, +0.9242] | 2/6 | Fail | Pass |

Only separate versus joint is the primary outcome comparison. The attribution comparison uses fixed random residual visual features; it cannot rescue failure or promote an ablation.

### conditional: declared gates

| Gate | Value | Decision |
|---|---:|---|
| complete_development_cohort | — | Pass |
| fresh_qualification | — | Pass |
| acquisition_gain | 0.016853026993271108 | Fail |
| acquisition_consistency | 4 | Fail |
| valid_old_mode_0 | -0.010240641362872095 | Pass |
| valid_old_mode_1 | -0.007719058419359835 | Pass |
| return_prediction | -0.015281357339952154 | Pass |
| post_return_novel_retention | 0.014376475703967372 | Fail |
| survival_novel | -0.018595377604166668 | Fail |
| survival_return | 0.004435221354166667 | Pass |

| Comparison / phase | Affected Brier AUC | All-case Brier AUC | Survival AUC |
|---|---:|---:|---:|
| separate_minus_joint / maintenance | -0.3987 [-0.6470, -0.1662] | -0.3987 [-0.6470, -0.1662] | -0.5534 [-1.1597, +0.2360] |
| separate_minus_joint / novel | +1.6853 [-0.6848, +4.2192] | +1.3963 [-0.8507, +3.8692] | -1.8595 [-4.4963, +0.3698] |
| separate_minus_joint / return | -1.5281 [-2.0676, -1.0242] | -1.5281 [-2.0676, -1.0242] | +0.4435 [+0.0407, +0.9440] |
| separate_minus_fixed_features / maintenance | +0.0750 [+0.0428, +0.1061] | +0.0750 [+0.0428, +0.1061] | -0.4598 [-0.6429, -0.2889] |
| separate_minus_fixed_features / novel | -0.2489 [-0.5594, +0.0650] | -0.2239 [-0.5035, +0.0450] | -0.0905 [-0.4232, +0.2823] |
| separate_minus_fixed_features / return | -0.6687 [-1.6249, +0.1169] | -0.6687 [-1.6249, +0.1169] | +0.2197 [-0.5595, +0.8667] |
| fixed_features_minus_joint / maintenance | -0.4737 [-0.7518, -0.2161] | -0.4737 [-0.7518, -0.2161] | -0.0936 [-0.7731, +0.7568] |
| fixed_features_minus_joint / novel | +1.9342 [-0.4216, +4.5889] | +1.6202 [-0.8006, +4.2575] | -1.7690 [-4.7811, +0.7777] |
| fixed_features_minus_joint / return | -0.8595 [-1.9127, +0.3654] | -0.8595 [-1.9127, +0.3654] | +0.2238 [-0.6799, +1.1597] |

| Retention contrast: separate−joint | Brier x100 [95% interval] |
|---|---:|
| Novel end: valid old mode 0 | -1.0241 [-2.1944, +0.0114] |
| Novel end: valid old mode 1 | -0.7719 [-3.2312, +1.7907] |
| Return end: novel law, correct support | +1.4376 [-0.7353, +3.7675] |

| Phase / arm | Held-out affected AUC x100 | Pre-update performed Brier x100 | Cue benefit x100 |
|---|---:|---:|---:|
| maintenance / joint | 7.2871 | 7.1912 | — |
| maintenance / separate | 6.8884 | 7.1632 | — |
| maintenance / fixed_features | 6.8133 | 7.0244 | — |
| novel / joint | 8.6036 | 8.3740 | 5.9502 |
| novel / separate | 10.2889 | 9.4526 | 0.7476 |
| novel / fixed_features | 10.5378 | 9.6441 | 0.0655 |
| novel / fresh | 8.1494 | 8.0198 | 7.5723 |
| return / joint | 10.9277 | 9.8540 | — |
| return / separate | 9.3995 | 8.0173 | — |
| return / fixed_features | 10.0682 | 8.6463 | — |

| Seed | Novel separate−joint Brier AUC x100 |
|---|---:|
| 16001 | +5.8166 |
| 16002 | -0.4768 |
| 16003 | -0.0405 |
| 16004 | -1.0404 |
| 16005 | -0.8190 |
| 16006 | +6.6719 |

### recurrent: declared gates

| Gate | Value | Decision |
|---|---:|---|
| complete_development_cohort | — | Pass |
| fresh_qualification | — | Pass |
| acquisition_gain | 0.0039166359912632025 | Fail |
| acquisition_consistency | 2 | Fail |
| valid_old_mode_0 | 0.004605177532025215 | Pass |
| valid_old_mode_1 | -0.03376146309281834 | Pass |
| return_prediction | 0.03372609488447477 | Fail |
| post_return_novel_retention | -0.031541314800603626 | Pass |
| survival_novel | -0.012898763020833334 | Fail |
| survival_return | -0.035135904947916664 | Fail |

| Comparison / phase | Affected Brier AUC | All-case Brier AUC | Survival AUC |
|---|---:|---:|---:|
| separate_minus_joint / maintenance | +0.7586 [+0.4312, +0.9678] | +0.7586 [+0.4312, +0.9678] | -1.4730 [-2.4902, -0.2360] |
| separate_minus_joint / novel | +0.3917 [-0.0877, +0.9242] | +0.2638 [-0.1966, +0.8172] | -1.2899 [-2.0477, -0.6627] |
| separate_minus_joint / return | +3.3726 [+1.4309, +4.8598] | +3.3726 [+1.4309, +4.8598] | -3.5136 [-4.9357, -2.0101] |
| separate_minus_fixed_features / maintenance | +0.1140 [+0.0591, +0.1745] | +0.1140 [+0.0591, +0.1745] | +0.0854 [-0.3825, +0.5819] |
| separate_minus_fixed_features / novel | -1.2130 [-1.7754, -0.7131] | -1.0562 [-1.5199, -0.6171] | +1.9653 [+1.2227, +2.6464] |
| separate_minus_fixed_features / return | -9.8969 [-13.1758, -6.0173] | -9.8969 [-13.1758, -6.0173] | +6.4392 [+3.1799, +9.7636] |
| fixed_features_minus_joint / maintenance | +0.6446 [+0.3262, +0.8579] | +0.6446 [+0.3262, +0.8579] | -1.5584 [-2.8117, +0.0570] |
| fixed_features_minus_joint / novel | +1.6047 [+0.6471, +2.6413] | +1.3201 [+0.4643, +2.1982] | -3.2552 [-4.3610, -2.1439] |
| fixed_features_minus_joint / return | +13.2695 [+9.8483, +16.4682] | +13.2695 [+9.8483, +16.4682] | -9.9528 [-12.9618, -6.8278] |

| Retention contrast: separate−joint | Brier x100 [95% interval] |
|---|---:|
| Novel end: valid old mode 0 | +0.4605 [-0.2303, +1.0117] |
| Novel end: valid old mode 1 | -3.3761 [-9.4210, +1.8242] |
| Return end: novel law, correct support | -3.1541 [-8.7660, +0.5408] |

| Phase / arm | Held-out affected AUC x100 | Pre-update performed Brier x100 | Cue benefit x100 |
|---|---:|---:|---:|
| maintenance / joint | 8.1398 | 7.8392 | — |
| maintenance / separate | 8.8984 | 8.5775 | — |
| maintenance / fixed_features | 8.7844 | 8.5479 | — |
| novel / joint | 8.7811 | 8.5520 | 6.0465 |
| novel / separate | 9.1727 | 8.7373 | 5.2115 |
| novel / fixed_features | 10.3857 | 9.8732 | 0.0643 |
| novel / fresh | 8.2058 | 8.1106 | 7.4547 |
| return / joint | 11.9874 | 11.7583 | — |
| return / separate | 15.3600 | 14.7041 | — |
| return / fixed_features | 25.2569 | 23.9702 | — |

| Seed | Novel separate−joint Brier AUC x100 |
|---|---:|
| 16001 | +1.2973 |
| 16002 | -0.3927 |
| 16003 | +0.2959 |
| 16004 | +0.0602 |
| 16005 | -0.1347 |
| 16006 | +1.2240 |

## Fresh qualification

| Architecture / group | Seeds | Marginal gain | Correct-cue benefit | Pass |
|---|---:|---:|---:|---|
| conditional / overall | 6 | 0.154961 | 0.075723 | True |
| conditional / cue_0 | 2 | 0.159139 | 0.022669 | True |
| conditional / cue_1 | 2 | 0.158738 | 0.008196 | True |
| conditional / cue_2 | 2 | 0.147005 | 0.196303 | True |
| recurrent / overall | 6 | 0.153525 | 0.074547 | True |
| recurrent / cue_0 | 2 | 0.161069 | 0.026262 | True |
| recurrent / cue_1 | 2 | 0.158058 | 0.006975 | True |
| recurrent / cue_2 | 2 | 0.141446 | 0.190403 | True |

## Resource and verification scope

resource_rows includes total/trainable parameters, active and retained Adam storage, measurement snapshots, replay/history, training work, prequential predictions, and every evaluator call. A composite logical call executes two raw pathways; prefix calls execute one. Freezing changes backward work. Equal capacity and update counts are not equal FLOPs or measured peak memory.

The reader checks locks, cohort and state-chain metadata, raw prediction arithmetic, original record/artifact hashes and causal-support references. It does not load model pickles or independently retrain ordinary updates. A separate checkpoint/physics audit has a separately stated scope.

Every arm, all six seeds, cue slices, absolute outcomes and paired intervals are preserved in summary.json. A failed qualification or guard stays failed; there is no outcome-based exposure change or subgroup promotion.
