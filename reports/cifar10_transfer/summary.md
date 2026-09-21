# CIFAR-10 transfer pilot

Sixteen fixed development runs in a two-seed exploratory transfer pilot; effort-allocation rule, not efficacy or statistical significance.

All 16 predeclared development runs completed; 22,560 optimizer updates. Engineering smoke runs are excluded. No cross-seed confidence interval is estimated.

Frozen source SHA-256: `60aa03ca51a5d530ee58e5ed631b1fc7c702052f26de8f0be981c0731d984268`. Source-freeze commit: `c250129b957d0fb930fa5b8a3a22ececef6ba1ae`.

| Condition | Method | Seed | Final % | Late first-256 AUC % | Stationary max drawdown, pp |
|---|---|---:|---:|---:|---:|
| recurring | er_v3 | 1063 | 42.07 | 37.49 | n/a |
| recurring | er_v3 | 1174 | 43.60 | 41.30 | n/a |
| recurring | recycle_v3 | 1063 | 42.73 | 38.65 | n/a |
| recurring | recycle_v3 | 1174 | 44.07 | 41.82 | n/a |
| recurring | newborn_v3 | 1063 | 37.47 | 38.20 | n/a |
| recurring | newborn_v3 | 1174 | 42.13 | 42.03 | n/a |
| recurring | newborn_matched_v3 | 1063 | 42.13 | 38.46 | n/a |
| recurring | newborn_matched_v3 | 1174 | 44.33 | 42.37 | n/a |
| stationary | er_v3 | 1063 | 48.00 | 42.14 | 8.40 |
| stationary | er_v3 | 1174 | 47.80 | 43.73 | 6.00 |
| stationary | recycle_v3 | 1063 | 44.00 | 42.87 | 7.60 |
| stationary | recycle_v3 | 1174 | 45.20 | 43.28 | 5.60 |
| stationary | newborn_v3 | 1063 | 44.20 | 43.02 | 8.40 |
| stationary | newborn_v3 | 1174 | 46.60 | 43.77 | 5.00 |
| stationary | newborn_matched_v3 | 1063 | 45.00 | 42.58 | 7.40 |
| stationary | newborn_matched_v3 | 1174 | 45.20 | 43.64 | 2.80 |

Predeclared effort decision: **stop scaling this frozen recipe**.

| Check | Pass |
|---|---|
| adequacy | yes |
| signal | no |
| retention | no |

Each paired recurring late-AUC difference (percentage points):

| Method minus recycling | Seed | Difference, pp |
|---|---:|---:|
| newborn_v3 | 1063 | -0.457 |
| newborn_matched_v3 | 1063 | -0.190 |
| newborn_v3 | 1174 | +0.210 |
| newborn_matched_v3 | 1174 | +0.550 |

Every threshold operand, comparison and pass/fail reason is retained in summary.json. Per-domain final accuracies, all 30 AUCs, and all stationary diagonals are retained there too.

Repeated domain panels are correlated; generic task-column forgetting/BWT is archived but not treated as independent-task retention.

Logged wall time includes concurrent workloads; it is not GPU time or a method-speed comparison.

The optimizer cap excludes reset mutations. Nominal gain matching precedes clipping. Raw checkpoints are locally retained; compact artifacts reference their hashes without distributing them.
