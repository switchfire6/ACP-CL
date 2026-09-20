# Exact stationary reproduction: immediate reset effects

**The final model matches the original checkpoint bitwise across all 13 model-state tensors.** Allocation traces, all original acquisition curves, training cost counters, and controller state also match exactly. The original configuration and training source were unchanged.

A fixed set of 128 original validation examples from experience 100 was measured before each optimizer update and immediately before/after recycling for steps 1801-2000. Measurements never entered the learner or controller.

**All 10 reset events changed accuracy by 0.00 percentage points on this fixed set.** The largest reset-only cross-entropy increase was 0.00001330. The observed large losses occurred during optimizer updates:

| Step | Units subsequently reset | Accuracy before update (%) | After update, before reset (%) | After reset (%) | Update change (pp) | Reset change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 1975 | 0 | 98.43750 | 88.28125 | 88.28125 | -10.15625 | +0.00000 |
| 1980 | 3 | 97.65625 | 60.15625 | 60.15625 | -37.50000 | +0.00000 |
| 1982 | 0 | 97.65625 | 50.00000 | 50.00000 | -47.65625 | +0.00000 |
| 1985 | 0 | 80.46875 | 42.18750 | 42.18750 | -38.28125 | +0.00000 |
| 1986 | 0 | 42.18750 | 25.78125 | 25.78125 | -16.40625 | +0.00000 |
| 2000 | 3 | 67.18750 | 55.46875 | 55.46875 | -11.71875 | +0.00000 |

At step 1980, the optimizer update reduced accuracy from 97.65625% to 60.15625%; the subsequent three-unit reset changed neither accuracy nor cross-entropy. At the final step, the update reduced accuracy from 67.1875% to 55.46875%, again before a reset with no measured accuracy or CE effect. The largest one-update accuracy drop was 47.65625 percentage points at step 1982, when no units were reset.

This identifies the immediate location of the measured damage along the reproduced trajectory. It does **not** identify the long-run effect of prior recycling, newborn gains, momentum, consolidation, replay, or head updates. Earlier resets can alter later optimization even when their immediate prediction effect is negligible. The audit uses one fixed 128-example sample and was chosen after observing the anomaly.

The reproduction added 410 forward evaluations (52,480 examples). Wall time includes this work and concurrent GPU jobs and is not a compute comparison. All 200 step records, CE values, hashes, and verification details are in [instrumented_audit.json](instrumented_audit.json). Same-set endpoint comparisons across all nine stationary runs are in [endpoint_audit.md](endpoint_audit.md).

Training source SHA-256: `67ae36402607f257f1e55c4f3ef2bbb477a88555d2111a4bd0139e27668eee0a`.
