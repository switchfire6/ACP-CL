# Predictive-value rank stability: retrospective diagnosis

Specified after reviewing the completed primary failures and secondary summaries. Exploratory only: no fitting, new gates, retuning or alteration of the failed decisions.

Spearman correlations use exact tied-average ranks of the eight candidate Brier losses, excluding the common replacement. Two assessments are averaged within each seed before the twelve seed means are averaged. Positive values indicate agreement in loss ordering. No inferential intervals or p-values are computed.

| Comparison | Conditional mean | Positive / 12 | Recurrent mean | Positive / 12 |
|---|---:|---:|---:|---:|
| score → near_frozen | +0.3690 | 10 | +0.3681 | 10 |
| score → near_reapplied | +0.1200 | 8 | +0.0050 | 7 |
| score → return_frozen | +0.0506 | 7 | -0.0933 | 5 |
| score → return_reapplied | +0.0169 | 7 | -0.0813 | 4 |
| near_frozen → near_reapplied | +0.1786 | 9 | -0.0308 | 6 |
| return_frozen → return_reapplied | +0.2292 | 11 | +0.1151 | 8 |

Undefined constant-vector assessment comparisons: 0. None are silently dropped; any undefined assessment makes its seed mean undefined, and any undefined seed prevents the overall mean.

## Conditional: all seed means

| Seed | Score→near frozen | Score→near reapplied | Score→return frozen | Score→return reapplied | Near frozen→reapplied | Return frozen→reapplied |
|---|---:|---:|---:|---:|---:|---:|
| 14001 | +0.7738 | +0.2024 | -0.0595 | +0.4524 | -0.0833 | +0.1548 |
| 14002 | +0.0357 | -0.2500 | +0.0357 | -0.2976 | +0.4762 | +0.0952 |
| 14003 | +0.4167 | +0.2381 | +0.0714 | -0.5595 | +0.0714 | +0.3690 |
| 14004 | +0.0952 | +0.2024 | -0.3333 | -0.2619 | +0.4524 | +0.4881 |
| 14005 | +0.7857 | +0.1310 | +0.0833 | +0.0833 | -0.1310 | +0.1429 |
| 14006 | -0.1071 | -0.0595 | +0.3929 | +0.3095 | +0.0833 | +0.0476 |
| 14007 | -0.2381 | +0.2262 | -0.1429 | -0.2500 | +0.2024 | +0.4643 |
| 14008 | +0.7143 | +0.2619 | -0.0476 | +0.4048 | +0.0833 | +0.2381 |
| 14009 | +0.4286 | -0.3214 | +0.0714 | +0.1310 | +0.1429 | -0.3452 |
| 14010 | +0.7381 | +0.4048 | +0.4643 | +0.1548 | +0.5476 | +0.4286 |
| 14011 | +0.6071 | +0.5595 | -0.1548 | -0.2143 | +0.3810 | +0.5238 |
| 14012 | +0.1786 | -0.1548 | +0.2262 | +0.2500 | -0.0833 | +0.1429 |

## Recurrent: all seed means

| Seed | Score→near frozen | Score→near reapplied | Score→return frozen | Score→return reapplied | Near frozen→reapplied | Return frozen→reapplied |
|---|---:|---:|---:|---:|---:|---:|
| 14001 | -0.0357 | -0.1667 | +0.1429 | +0.3214 | -0.2500 | -0.0357 |
| 14002 | -0.5952 | +0.0952 | -0.2024 | -0.1429 | -0.4048 | +0.5952 |
| 14003 | +0.7381 | +0.0714 | -0.6667 | +0.0595 | +0.0833 | +0.0000 |
| 14004 | +0.3333 | +0.2262 | -0.0595 | -0.3810 | +0.3333 | +0.1071 |
| 14005 | +0.5119 | -0.0357 | +0.0476 | -0.2976 | -0.2619 | +0.0238 |
| 14006 | +0.0595 | -0.2619 | -0.1905 | -0.2381 | +0.2738 | +0.3095 |
| 14007 | +0.8571 | +0.2976 | -0.4762 | +0.4048 | +0.2619 | +0.1190 |
| 14008 | +0.0595 | +0.0119 | +0.1429 | -0.3929 | +0.3095 | +0.2857 |
| 14009 | +0.3214 | -0.2976 | -0.0476 | +0.2143 | -0.3214 | +0.1429 |
| 14010 | +0.7024 | +0.3929 | -0.0714 | -0.3690 | +0.0714 | -0.2857 |
| 14011 | +0.8452 | -0.4524 | +0.1429 | -0.0476 | -0.4286 | +0.4524 |
| 14012 | +0.6190 | +0.1786 | +0.1190 | -0.1071 | -0.0357 | -0.3333 |

## Interpretation limits

Frozen/reapplied comparisons share later queries and context but change the parent weights, Adam state and latest anchor. These correlations do not isolate why ordering changed. Rank agreement is separate from gain size, calibration, survival and retention. The original local and return criteria remain failed; a favorable secondary ordering does not promote a replacement policy.

See [design.md](design.md) for exact formulas and constant handling, [results.json](results.json) for every assessment and verification example, and [source_lock.json](source_lock.json) / [source.zip](source.zip) for the retrospective snapshot.

Input compressed-file SHA256: `3a2f5a6d1d14f82d944c3c16f8413fe1c3a3550af2bcb3163c9fd1eed546f44f`.
Analysis script SHA256: `d981cb13288a9cf45a25fd9d7015bc2f85e849c5ff366b969674db30a258046b`.
