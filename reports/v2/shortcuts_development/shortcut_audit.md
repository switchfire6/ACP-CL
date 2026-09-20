# Post-hoc shape shortcut diagnostic

This exploratory audit is **not confirmatory held-out performance**. No model updates, controller feedback, or tuning were performed.

Each model receives 64 fresh images per label in each of 8 configured domains, paired with renders whose foreground palette is independent of the label.

Geometry and pre-palette texture draws match. Palette-branch RNG consumption can change color jitter and sensor-noise realizations; their distributions remain fixed. Positive drops indicate sensitivity to this rendering change.

| Method | Available seeds | Original accuracy | Uncorrelated colors | Drop, pp |
|---|---|---:|---:|---:|
| er_recycle | 101, 202, 303 | 39.52% | 25.70% | +13.82 |
| acp_v2 | 101, 202, 303 | 39.03% | 25.81% | +13.22 |
| acp_v2_oracle | 101, 202, 303 | 40.28% | 25.50% | +14.78 |

| Method | Seed | Original accuracy | Uncorrelated colors | Drop, pp |
|---|---:|---:|---:|---:|
| acp_v2_oracle | 101 | 37.89% | 24.46% | +13.43 |
| acp_v2_oracle | 202 | 41.60% | 25.88% | +15.72 |
| acp_v2_oracle | 303 | 41.36% | 26.17% | +15.19 |
| acp_v2 | 101 | 37.74% | 25.54% | +12.21 |
| acp_v2 | 202 | 39.89% | 25.59% | +14.31 |
| acp_v2 | 303 | 39.45% | 26.32% | +13.13 |
| er_recycle | 101 | 37.55% | 25.10% | +12.45 |
| er_recycle | 202 | 40.48% | 26.46% | +14.01 |
| er_recycle | 303 | 40.53% | 25.54% | +14.99 |

Oracle methods had access to true domain-change times during training. Yoked methods, when present, consume completed ACP-v2 allocation traces; the gain variant also uses the future run-average gain. They are extra-information diagnostics, not task-free competitors. Per-domain results, source/config/checkpoint hashes, and paired correctness counts are recorded in shortcut_audit.json.
