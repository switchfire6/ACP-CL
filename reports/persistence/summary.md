# Joint-persistence pilot: complete descriptive results

288 comparative runs and 32 fresh late-block references. All planned results included. Intervals resample paired seeds and are exploratory, not adjusted for secondary comparisons.

Primary long-gap return difference, joint minus uniform: **-0.78 pp** (paired bootstrap 95% interval -1.66 to +0.46); positive in 1/8 seeds.

Predeclared continuation screen: **FAIL**. Baseline acquisition gain over no transfer: 9.42 pp.

| Criterion | Pass |
|---|---|
| baseline_adequate | True |
| primary_mean | False |
| positive_seed_count | False |
| beats_coverage | False |
| secondary_noninferiority_screen | True |

## short

Percentages; late AUC integrates survival over current arrivals.

| Method | Entry, block 11 | Final held-out combinations | Final valid known | Late AUC | Late endpoint | Learning + selection, seconds |
|---|---:|---:|---:|---:|---:|---:|
| none | 54.30 | 61.86 | 62.66 | 64.53 | 68.72 | 3.29 |
| uniform | 73.37 | 73.49 | 72.92 | 70.94 | 73.83 | 3.72 |
| recent | 54.10 | 62.56 | 63.33 | 66.16 | 70.93 | 3.65 |
| coverage | 72.79 | 72.97 | 72.90 | 70.72 | 74.19 | 4.38 |
| recurrence | 72.10 | 73.84 | 73.42 | 70.23 | 74.35 | 4.44 |
| relevance | 71.78 | 72.55 | 72.70 | 71.03 | 74.22 | 4.44 |
| joint | 73.31 | 73.65 | 73.49 | 70.71 | 74.22 | 4.45 |
| frozen_encoder | 60.90 | 62.77 | 63.89 | 62.93 | 65.10 | 2.24 |
| frozen_late | 73.08 | 71.52 | 71.09 | 63.76 | 64.19 | 2.74 |

## long

Percentages; late AUC integrates survival over current arrivals.

| Method | Entry, block 11 | Final held-out combinations | Final valid known | Late AUC | Late endpoint | Learning + selection, seconds |
|---|---:|---:|---:|---:|---:|---:|
| none | 49.74 | 61.75 | 62.37 | 63.86 | 67.06 | 3.34 |
| uniform | 74.06 | 74.20 | 73.71 | 71.42 | 74.32 | 3.68 |
| recent | 52.93 | 63.44 | 63.83 | 66.32 | 71.52 | 3.64 |
| coverage | 73.27 | 72.49 | 72.48 | 70.74 | 74.35 | 4.38 |
| recurrence | 73.14 | 73.62 | 73.54 | 70.93 | 74.51 | 4.46 |
| relevance | 72.10 | 73.81 | 73.50 | 70.55 | 74.15 | 4.54 |
| joint | 73.27 | 73.85 | 73.22 | 71.07 | 74.74 | 4.64 |
| frozen_encoder | 61.65 | 63.91 | 65.30 | 64.12 | 66.24 | 2.32 |
| frozen_late | 72.66 | 71.57 | 71.35 | 63.95 | 63.28 | 2.81 |

## no_return

Percentages; late AUC integrates survival over current arrivals.

| Method | Entry, block 11 | Final held-out combinations | Final valid known | Late AUC | Late endpoint | Learning + selection, seconds |
|---|---:|---:|---:|---:|---:|---:|
| none | 50.07 | 59.71 | 63.03 | 62.88 | 66.50 | 3.48 |
| uniform | 75.00 | 72.98 | 72.88 | 70.33 | 73.86 | 3.89 |
| recent | 52.90 | 60.06 | 63.35 | 67.62 | 71.45 | 3.81 |
| coverage | 74.12 | 72.37 | 73.08 | 70.18 | 73.44 | 4.42 |
| recurrence | 74.28 | 73.57 | 74.07 | 70.63 | 73.99 | 4.46 |
| relevance | 72.69 | 73.11 | 73.58 | 70.72 | 74.22 | 4.42 |
| joint | 74.09 | 73.67 | 73.58 | 70.78 | 73.80 | 4.39 |
| frozen_encoder | 59.08 | 64.19 | 65.72 | 63.11 | 64.91 | 2.23 |
| frozen_late | 74.12 | 71.57 | 71.77 | 63.76 | 64.55 | 2.72 |

## reversal

Percentages; late AUC integrates survival over current arrivals.

| Method | Entry, block 11 | Final held-out combinations | Final valid known | Late AUC | Late endpoint | Learning + selection, seconds |
|---|---:|---:|---:|---:|---:|---:|
| none | 49.74 | 58.22 | 61.95 | 64.78 | 70.28 | 3.33 |
| uniform | 74.06 | 66.21 | 70.96 | 69.26 | 72.30 | 3.72 |
| recent | 52.93 | 60.51 | 65.20 | 66.11 | 72.27 | 3.75 |
| coverage | 73.27 | 63.77 | 68.40 | 69.62 | 72.62 | 4.43 |
| recurrence | 73.14 | 62.21 | 66.33 | 70.77 | 74.19 | 4.53 |
| relevance | 72.10 | 63.03 | 67.87 | 69.96 | 73.63 | 4.40 |
| joint | 73.27 | 62.62 | 66.76 | 70.41 | 73.96 | 4.43 |
| frozen_encoder | 61.65 | 61.47 | 65.78 | 67.36 | 70.08 | 2.24 |
| frozen_late | 72.66 | 65.12 | 67.88 | 66.19 | 68.82 | 2.75 |

## Primary contrast by seed

| Seed | Joint | Uniform | Difference, pp |
|---|---:|---:|---:|
| 1001 | 79.43 | 80.47 | -1.04 |
| 1002 | 64.06 | 66.67 | -2.60 |
| 1003 | 64.84 | 66.15 | -1.30 |
| 1004 | 63.02 | 64.32 | -1.30 |
| 1005 | 80.21 | 80.47 | -0.26 |
| 1006 | 66.41 | 67.97 | -1.56 |
| 1008 | 83.59 | 84.90 | -1.30 |
| 1019 | 84.64 | 81.51 | +3.12 |

## Scope and accounting

The implemented recurrence score is a coarse historical-frequency proxy. The relevance score is predicted action sensitivity, not a causal estimate of memory value. Parameter counts, memory payload and optimization work are matched among full-learning replay methods; selection overhead is additional. Frozen controls have less backward work. The no-memory control repeats current examples and scores fewer candidates.

The four initial contexts correlate source direction with the loss/delay glyph combination. The withheld combinations break that correlation. Therefore known-context performance alone does not establish use of temporal or causal structure. History-removal probes are sensitivity checks and may create distribution shift. Final no-return composition scores exclude the combination subsequently trained; reversal retention excludes the now-invalid old target.

The fresh late-block fits differ in replay contents and history. Their differences from continual learners are descriptive and cannot isolate plasticity. No claim of general continual representation learning, inherited learning, indefinite survival, or a novel evolutionary algorithm follows.

Full raw curves, seeds, hashes, resource counters and fresh fits are in `archive.json.gz`. The source archive and manifest support regeneration. Checkpoints remain in the local run directory.
