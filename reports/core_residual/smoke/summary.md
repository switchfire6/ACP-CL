# Learned core and adaptive raw-input residual

Prospective fixed development comparison. Brier differences below are raw Brier units times 100; survival differences are percentage points. Negative Brier favors the first arm.

Six independent seeds per architecture. Curves/supports/actions/horizons are not replicates. 20,000 paired seed bootstrap resamples, RNG 27192026; intervals are descriptive. Held-out post-update AUC and performed-action prequential error are distinct outcomes.

## Fixed primary outcome and feature attribution

| Architecture | Novel separate−joint [95% interval] | Improved seeds | Primary | Feature attribution |
|---|---:|---:|---|---|
| conditional | +0.5100 [+0.5100, +0.5100] | 0/1 | Fail | Fail |
| recurrent | +0.5992 [+0.5992, +0.5992] | 0/1 | Fail | Fail |

Only separate versus joint is the primary outcome comparison. The attribution comparison uses fixed random residual visual features; it cannot rescue failure or promote an ablation.

### conditional: declared gates

| Gate | Value | Decision |
|---|---:|---|
| complete_development_cohort | — | Fail |
| fresh_qualification | — | Fail |
| acquisition_gain | 0.005099879660527273 | Fail |
| acquisition_consistency | 0 | Fail |
| valid_old_mode_0 | 0.0057365826561509015 | Fail |
| valid_old_mode_1 | 0.006711966061262065 | Fail |
| return_prediction | 0.009789657247556482 | Fail |
| post_return_novel_retention | 0.009297603595866177 | Fail |
| survival_novel | 0.001953125 | Pass |
| survival_return | 0.05078125 | Pass |

| Comparison / phase | Affected Brier AUC | All-case Brier AUC | Survival AUC |
|---|---:|---:|---:|
| separate_minus_joint / maintenance | +0.0826 [+0.0826, +0.0826] | +0.0826 [+0.0826, +0.0826] | +0.0000 [+0.0000, +0.0000] |
| separate_minus_joint / novel | +0.5100 [+0.5100, +0.5100] | +0.4872 [+0.4872, +0.4872] | +0.1953 [+0.1953, +0.1953] |
| separate_minus_joint / return | +0.9790 [+0.9790, +0.9790] | +0.9790 [+0.9790, +0.9790] | +5.0781 [+5.0781, +5.0781] |
| separate_minus_fixed_features / maintenance | -0.0025 [-0.0025, -0.0025] | -0.0025 [-0.0025, -0.0025] | +0.0000 [+0.0000, +0.0000] |
| separate_minus_fixed_features / novel | -0.0318 [-0.0318, -0.0318] | -0.0306 [-0.0306, -0.0306] | +0.0000 [+0.0000, +0.0000] |
| separate_minus_fixed_features / return | -0.0898 [-0.0898, -0.0898] | -0.0898 [-0.0898, -0.0898] | +0.3906 [+0.3906, +0.3906] |
| fixed_features_minus_joint / maintenance | +0.0851 [+0.0851, +0.0851] | +0.0851 [+0.0851, +0.0851] | +0.0000 [+0.0000, +0.0000] |
| fixed_features_minus_joint / novel | +0.5417 [+0.5417, +0.5417] | +0.5178 [+0.5178, +0.5178] | +0.1953 [+0.1953, +0.1953] |
| fixed_features_minus_joint / return | +1.0688 [+1.0688, +1.0688] | +1.0688 [+1.0688, +1.0688] | +4.6875 [+4.6875, +4.6875] |

| Retention contrast: separate−joint | Brier x100 [95% interval] |
|---|---:|
| Novel end: valid old mode 0 | +0.5737 [+0.5737, +0.5737] |
| Novel end: valid old mode 1 | +0.6712 [+0.6712, +0.6712] |
| Return end: novel law, correct support | +0.9298 [+0.9298, +0.9298] |

| Phase / arm | Held-out affected AUC x100 | Pre-update performed Brier x100 | Cue benefit x100 |
|---|---:|---:|---:|
| maintenance / joint | 24.1677 | 23.5035 | — |
| maintenance / separate | 24.2502 | 23.5846 | — |
| maintenance / fixed_features | 24.2527 | 23.5848 | — |
| novel / joint | 23.2339 | 23.5928 | 0.0026 |
| novel / separate | 23.7439 | 24.0397 | 0.0033 |
| novel / fixed_features | 23.7756 | 24.0668 | 0.0036 |
| novel / fresh | 24.0396 | 24.2333 | 0.0019 |
| return / joint | 22.6702 | 23.2615 | — |
| return / separate | 23.6492 | 23.8009 | — |
| return / fixed_features | 23.7390 | 23.8434 | — |

| Seed | Novel separate−joint Brier AUC x100 |
|---|---:|
| 16991 | +0.5100 |

### recurrent: declared gates

| Gate | Value | Decision |
|---|---:|---|
| complete_development_cohort | — | Fail |
| fresh_qualification | — | Fail |
| acquisition_gain | 0.005991592475468377 | Fail |
| acquisition_consistency | 0 | Fail |
| valid_old_mode_0 | 0.005576827211308866 | Fail |
| valid_old_mode_1 | 0.005728889574199925 | Fail |
| return_prediction | 0.010260048586564319 | Fail |
| post_return_novel_retention | 0.010252233667422062 | Fail |
| survival_novel | -0.013671875 | Fail |
| survival_return | 0.015625 | Pass |

| Comparison / phase | Affected Brier AUC | All-case Brier AUC | Survival AUC |
|---|---:|---:|---:|
| separate_minus_joint / maintenance | +0.0781 [+0.0781, +0.0781] | +0.0781 [+0.0781, +0.0781] | +0.0000 [+0.0000, +0.0000] |
| separate_minus_joint / novel | +0.5992 [+0.5992, +0.5992] | +0.5614 [+0.5614, +0.5614] | -1.3672 [-1.3672, -1.3672] |
| separate_minus_joint / return | +1.0260 [+1.0260, +1.0260] | +1.0260 [+1.0260, +1.0260] | +1.5625 [+1.5625, +1.5625] |
| separate_minus_fixed_features / maintenance | -0.0016 [-0.0016, -0.0016] | -0.0016 [-0.0016, -0.0016] | +0.0000 [+0.0000, +0.0000] |
| separate_minus_fixed_features / novel | -0.0262 [-0.0262, -0.0262] | -0.0254 [-0.0254, -0.0254] | +0.0000 [+0.0000, +0.0000] |
| separate_minus_fixed_features / return | -0.0884 [-0.0884, -0.0884] | -0.0884 [-0.0884, -0.0884] | +0.0000 [+0.0000, +0.0000] |
| fixed_features_minus_joint / maintenance | +0.0797 [+0.0797, +0.0797] | +0.0797 [+0.0797, +0.0797] | +0.0000 [+0.0000, +0.0000] |
| fixed_features_minus_joint / novel | +0.6254 [+0.6254, +0.6254] | +0.5869 [+0.5869, +0.5869] | -1.3672 [-1.3672, -1.3672] |
| fixed_features_minus_joint / return | +1.1144 [+1.1144, +1.1144] | +1.1144 [+1.1144, +1.1144] | +1.5625 [+1.5625, +1.5625] |

| Retention contrast: separate−joint | Brier x100 [95% interval] |
|---|---:|
| Novel end: valid old mode 0 | +0.5577 [+0.5577, +0.5577] |
| Novel end: valid old mode 1 | +0.5729 [+0.5729, +0.5729] |
| Return end: novel law, correct support | +1.0252 [+1.0252, +1.0252] |

| Phase / arm | Held-out affected AUC x100 | Pre-update performed Brier x100 | Cue benefit x100 |
|---|---:|---:|---:|
| maintenance / joint | 24.2798 | 24.2358 | — |
| maintenance / separate | 24.3579 | 24.3524 | — |
| maintenance / fixed_features | 24.3595 | 24.3528 | — |
| novel / joint | 23.3414 | 23.5780 | -0.0002 |
| novel / separate | 23.9406 | 23.9898 | -0.0031 |
| novel / fixed_features | 23.9668 | 24.0057 | -0.0031 |
| novel / fresh | 24.3665 | 24.4963 | -0.0006 |
| return / joint | 22.6388 | 24.3619 | — |
| return / separate | 23.6648 | 24.9210 | — |
| return / fixed_features | 23.7531 | 24.9640 | — |

| Seed | Novel separate−joint Brier AUC x100 |
|---|---:|
| 16991 | +0.5992 |

## Fresh qualification

| Architecture / group | Seeds | Marginal gain | Correct-cue benefit | Pass |
|---|---:|---:|---:|---|
| conditional / overall | 1 | -0.009217 | 0.000019 | False |
| conditional / cue_2 | 1 | -0.009217 | 0.000019 | False |
| recurrent / overall | 1 | -0.008557 | -0.000006 | False |
| recurrent / cue_2 | 1 | -0.008557 | -0.000006 | False |

## Resource and verification scope

resource_rows includes total/trainable parameters, active and retained Adam storage, measurement snapshots, replay/history, training work, prequential predictions, and every evaluator call. A composite logical call executes two raw pathways; prefix calls execute one. Freezing changes backward work. Equal capacity and update counts are not equal FLOPs or measured peak memory.

The reader checks locks, cohort and state-chain metadata, raw prediction arithmetic, original record/artifact hashes and causal-support references. It does not load model pickles or independently retrain ordinary updates. A separate checkpoint/physics audit has a separately stated scope.

Every arm, all six seeds, cue slices, absolute outcomes and paired intervals are preserved in summary.json. A failed qualification or guard stays failed; there is no outcome-based exposure change or subgroup promotion.
