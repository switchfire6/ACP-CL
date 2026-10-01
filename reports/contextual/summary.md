# Matched-history comparison: complete results

96 shared prefixes; 288 branches; decision **fail**.

Percentages; gap orders averaged within seed before paired analysis. 8 independent seeds.

## return

| Method | Before, 32 feedback | Learning AUC | Endpoint | After, 32 feedback | Valid old modes |
|---|---:|---:|---:|---:|---:|
| pooled | 57.94 | 67.78 | 71.01 | 71.01 | 66.15 |
| conditional | 73.23 | 72.07 | 75.20 | 75.09 | 73.58 |
| recurrent | 66.13 | 71.91 | 75.47 | 74.30 | 71.91 |
| query_routed | 74.07 | 72.15 | 75.36 | 75.37 | 73.83 |
| recurrent_frozen | 63.33 | 72.19 | 74.66 | 74.89 | 71.33 |
| oracle | 74.69 | 75.33 | 75.28 | 75.28 | 74.26 |

## novel

| Method | Before, 32 feedback | Learning AUC | Endpoint | After, 32 feedback | Valid old modes |
|---|---:|---:|---:|---:|---:|
| pooled | 63.57 | 67.44 | 70.10 | 70.10 | 64.64 |
| conditional | 62.71 | 67.56 | 71.60 | 71.50 | 72.62 |
| recurrent | 63.54 | 68.16 | 72.35 | 72.55 | 68.52 |
| query_routed | 62.62 | 67.78 | 72.80 | 72.48 | 72.81 |
| recurrent_frozen | 63.13 | 64.02 | 64.05 | 64.73 | 68.74 |
| oracle | 61.96 | 68.71 | 72.71 | 72.71 | 73.80 |

## noise

| Method | Before, 32 feedback | Learning AUC | Endpoint | After, 32 feedback | Valid old modes |
|---|---:|---:|---:|---:|---:|
| pooled | 70.77 | 70.13 | 70.15 | 70.15 | 62.65 |
| conditional | 69.84 | 71.37 | 72.33 | 70.94 | 72.86 |
| recurrent | 72.80 | 71.32 | 70.83 | 71.72 | 67.44 |
| query_routed | 70.29 | 71.12 | 71.66 | 71.55 | 72.58 |
| recurrent_frozen | 72.97 | 71.85 | 72.12 | 72.50 | 66.93 |
| oracle | 73.47 | 71.57 | 72.27 | 72.27 | 73.62 |

## Primary difference by seed

| Seed | Query routing minus recurrent, new-learning AUC (pp) |
|---|---:|
| 5001 | -3.14 |
| 5002 | +0.59 |
| 5003 | -1.16 |
| 5004 | -1.53 |
| 5005 | +1.84 |
| 5006 | -1.61 |
| 5007 | -0.93 |
| 5008 | +2.90 |

Mean -0.38 pp; descriptive 95% paired interval [-1.61, +0.95].

## Adequacy and decision

- oracle_acquisition: **True**
- recurrent_acquisition: **True**
- recurrent_return: **True**
- recurrent_uses_history: **True**
- new_learning_gain_over_recurrent_2pp: **False**
- new_learning_positive_6_seeds: **False**
- new_learning_gain_over_original_2pp: **False**
- new_learning_over_original_positive_6_seeds: **False**
- return_loss_at_most_2pp: **True**
- valid_retention_loss_at_most_2pp: **True**
- noise_loss_vs_recurrent_at_most_2pp: **True**
- noise_loss_vs_original_at_most_2pp: **True**
