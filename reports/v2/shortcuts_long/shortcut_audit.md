# Post-hoc shape shortcut diagnostic

This exploratory audit is **not confirmatory held-out performance**. No model updates, controller feedback, or tuning were performed.

Each model receives 64 fresh images per label in each of 8 configured domains, paired with renders whose foreground palette is independent of the label.

Geometry and pre-palette texture draws match. Palette-branch RNG consumption can change color jitter and sensor-noise realizations; their distributions remain fixed. Positive drops indicate sensitivity to this rendering change.

| Method | Available seeds | Original accuracy | Uncorrelated colors | Drop, pp |
|---|---|---:|---:|---:|
| er | 101, 202, 303 | 66.28% | 40.79% | +25.49 |
| er_recycle | 101, 202, 303 | 74.19% | 61.38% | +12.81 |
| acp_v2 | 101, 202, 303 | 65.01% | 37.45% | +27.56 |
| acp_v2_no_newborn | 101, 202, 303 | 54.92% | 30.29% | +24.63 |
| acp_v2_no_reopening | 101, 202, 303 | 61.26% | 34.13% | +27.13 |
| acp_v2_oracle | 101, 202, 303 | 62.09% | 32.19% | +29.90 |
| er_recycle_yoked | 101, 202, 303 | 50.86% | 26.30% | +24.56 |
| er_recycle_yoked_gain | 101, 202, 303 | 97.43% | 95.39% | +2.03 |

| Method | Seed | Original accuracy | Uncorrelated colors | Drop, pp |
|---|---:|---:|---:|---:|
| acp_v2_no_newborn | 101 | 57.08% | 29.59% | +27.49 |
| acp_v2_no_newborn | 202 | 46.19% | 33.45% | +12.74 |
| acp_v2_no_newborn | 303 | 61.47% | 27.83% | +33.64 |
| acp_v2_no_reopening | 101 | 62.79% | 33.69% | +29.10 |
| acp_v2_no_reopening | 202 | 59.81% | 40.58% | +19.24 |
| acp_v2_no_reopening | 303 | 61.18% | 28.12% | +33.06 |
| acp_v2_oracle | 101 | 62.26% | 30.91% | +31.35 |
| acp_v2_oracle | 202 | 67.63% | 38.09% | +29.54 |
| acp_v2_oracle | 303 | 56.40% | 27.59% | +28.81 |
| acp_v2 | 101 | 62.70% | 35.21% | +27.49 |
| acp_v2 | 202 | 73.83% | 50.15% | +23.68 |
| acp_v2 | 303 | 58.50% | 27.00% | +31.49 |
| er_recycle | 101 | 92.19% | 83.20% | +8.98 |
| er_recycle | 202 | 81.74% | 72.36% | +9.38 |
| er_recycle | 303 | 48.63% | 28.56% | +20.07 |
| er_recycle_yoked_gain | 101 | 95.70% | 92.72% | +2.98 |
| er_recycle_yoked_gain | 202 | 98.63% | 98.00% | +0.63 |
| er_recycle_yoked_gain | 303 | 97.95% | 95.46% | +2.49 |
| er_recycle_yoked | 101 | 51.03% | 24.37% | +26.66 |
| er_recycle_yoked | 202 | 50.88% | 25.34% | +25.54 |
| er_recycle_yoked | 303 | 50.68% | 29.20% | +21.48 |
| er | 101 | 72.07% | 53.56% | +18.51 |
| er | 202 | 64.36% | 40.09% | +24.27 |
| er | 303 | 62.40% | 28.71% | +33.69 |

Oracle methods had access to true domain-change times during training. Yoked methods, when present, consume completed ACP-v2 allocation traces; the gain variant also uses the future run-average gain. They are extra-information diagnostics, not task-free competitors. Per-domain results, source/config/checkpoint hashes, and paired correctness counts are recorded in shortcut_audit.json.
