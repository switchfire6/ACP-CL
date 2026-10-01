# Conditional relationship reuse: complete results

144 runs; adequacy **True**; continuation screen **pass**.

Values below are percentages. Intervals are paired seed bootstrap descriptions,
not multiplicity-adjusted significance statements. All eight seeds are retained.

## recurring

| Method | Inference-only probe | Valid old modes | Late gate | Late AUC | Actual final stream |
|---|---:|---:|---:|---:|---:|
| pooled | 59.02 | 65.56 | 65.23 | 64.56 | 65.04 |
| conditional | 70.62 | 69.97 | 63.40 | 61.23 | 62.17 |
| shuffled | 53.84 | 64.45 | 64.12 | 63.60 | 63.70 |
| current_only | 52.38 | 63.09 | 62.13 | 60.97 | 63.15 |
| frozen_features | 71.22 | 71.43 | 62.61 | 60.45 | 62.24 |
| oracle | 72.17 | 73.39 | 71.29 | 63.83 | 70.74 |

## stable

| Method | Inference-only probe | Valid old modes | Late gate | Late AUC | Actual final stream |
|---|---:|---:|---:|---:|---:|
| pooled | 74.12 | 74.15 | 71.71 | 65.04 | 71.71 |
| conditional | 73.63 | 73.28 | 70.41 | 61.39 | 70.93 |
| shuffled | 72.62 | 72.48 | 72.59 | 64.85 | 72.56 |
| current_only | 73.44 | 72.74 | 71.05 | 60.50 | 71.19 |
| frozen_features | 73.77 | 72.64 | 61.12 | 59.24 | 61.43 |
| oracle | 73.89 | 74.19 | 73.27 | 65.02 | 73.27 |

## unpredictable

| Method | Inference-only probe | Valid old modes | Late gate | Late AUC | Actual final stream |
|---|---:|---:|---:|---:|---:|
| pooled | 65.92 | n/a | n/a | 64.95 | 64.88 |
| conditional | 65.43 | n/a | n/a | 63.83 | 64.52 |
| shuffled | 64.94 | n/a | n/a | 64.18 | 64.71 |
| current_only | 65.19 | n/a | n/a | 62.70 | 62.83 |
| frozen_features | 65.53 | n/a | n/a | 63.52 | 64.13 |
| oracle | 75.00 | n/a | n/a | 63.52 | 70.70 |

## Primary seed differences

| Seed | Conditional | Pooled | Difference (pp) | Correct minus opposite (pp) |
|---|---:|---:|---:|---:|
| 3001 | 58.33 | 47.40 | +10.94 | +13.48 |
| 3002 | 72.14 | 62.76 | +9.38 | +25.72 |
| 3003 | 71.29 | 53.91 | +17.38 | +24.41 |
| 3004 | 71.88 | 63.80 | +8.07 | +26.56 |
| 3005 | 73.70 | 60.94 | +12.76 | +27.34 |
| 3006 | 70.31 | 61.98 | +8.33 | +21.61 |
| 3007 | 72.33 | 55.73 | +16.60 | +19.21 |
| 3008 | 75.00 | 65.62 | +9.38 | +23.96 |

Mean paired difference: +11.60 pp; 95% interval [+9.38, +14.09]; 8/8 positive.

## Predeclared criteria

- primary_gain_at_least_2pp: **True**
- primary_positive_in_6_seeds: **True**
- gain_over_shuffled_at_least_2pp: **True**
- opposite_history_drop_at_least_2pp: **True**
- opposite_history_drop_positive_in_6_seeds: **True**
- unpredictable_loss_no_worse_than_2pp: **True**
- late_gate_loss_no_worse_than_2pp: **True**

The stable stream's known-mode scores exclude its never-trained mode. The
unpredictable stream is scored on independent mixed modes; its final persistent-
mode panels are OOD diagnostics and are retained only in the raw archive.
