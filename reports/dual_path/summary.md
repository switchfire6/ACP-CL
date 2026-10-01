# Dual-path representation-learning pilot

Initial feasibility diagnostic. All planned runs are included; no significance claim.

Source SHA-256: `f1517828a6b26be23809fecd16c0805f286bb0f78b4c37b32683701a339ce1b2`.
Configuration SHA-256: `c234953d685249911b44c2cfeb14417d345bbf2bbc9ee26db000150dfa818116`.

Means below are percentages; forgetting and scratch differences are percentage points.

| Method | Seeds | Final accuracy | Forgetting | Late acquisition AUC | Scratch difference |
|---|---:|---:|---:|---:|---:|
| er | 2 | 3.13 | 25.32 | 0.00 | -8.85 |
| derpp | 2 | 4.52 | 32.18 | 0.00 | -8.85 |
| dual_joint | 2 | 3.30 | 29.47 | 0.00 | -8.88 |
| dual_derpp | 2 | 4.78 | 33.19 | 0.00 | -8.88 |
| dual_frozen | 2 | 0.94 | 10.54 | 0.00 | -8.88 |
| dual_consolidate | 2 | 2.08 | 14.89 | 0.00 | -8.88 |

## Every seed

| Method | Seed | Final accuracy | Late AUC | Scratch difference |
|---|---:|---:|---:|---:|
| derpp | 2027 | 5.14 | 0.00 | -8.98 |
| derpp | 2131 | 3.90 | 0.00 | -8.72 |
| dual_consolidate | 2027 | 1.48 | 0.00 | -8.85 |
| dual_consolidate | 2131 | 2.68 | 0.00 | -8.90 |
| dual_derpp | 2027 | 5.16 | 0.00 | -8.85 |
| dual_derpp | 2131 | 4.40 | 0.00 | -8.90 |
| dual_frozen | 2027 | 0.88 | 0.00 | -8.85 |
| dual_frozen | 2131 | 1.00 | 0.00 | -8.90 |
| dual_joint | 2027 | 3.24 | 0.00 | -8.85 |
| dual_joint | 2131 | 3.36 | 0.00 | -8.90 |
| er | 2027 | 2.36 | 0.00 | -8.98 |
| er | 2131 | 3.90 | 0.00 | -8.72 |

## Resource accounting

Equal arrivals do not imply equal memory or compute. Branch-example counts are not FLOPs.

| Method | Seed | Parameters | Replay bytes | Current examples | Online replay examples | Consolidation examples | Extra updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| derpp | 2027 | 34356 | 1781760 | 45000 | 45376 | 0 | 0 |
| derpp | 2131 | 34356 | 1781760 | 45000 | 45376 | 0 | 0 |
| dual_consolidate | 2027 | 44800 | 1576960 | 45000 | 45376 | 13824 | 216 |
| dual_consolidate | 2131 | 44800 | 1576960 | 45000 | 45376 | 13824 | 216 |
| dual_derpp | 2027 | 44800 | 1781760 | 45000 | 45376 | 0 | 0 |
| dual_derpp | 2131 | 44800 | 1781760 | 45000 | 45376 | 0 | 0 |
| dual_frozen | 2027 | 44800 | 1576960 | 45000 | 45376 | 0 | 0 |
| dual_frozen | 2131 | 44800 | 1576960 | 45000 | 45376 | 0 | 0 |
| dual_joint | 2027 | 44800 | 1576960 | 45000 | 45376 | 0 | 0 |
| dual_joint | 2131 | 44800 | 1576960 | 45000 | 45376 | 0 | 0 |
| er | 2027 | 34356 | 1576960 | 45000 | 45376 | 0 | 0 |
| er | 2131 | 34356 | 1576960 | 45000 | 45376 | 0 | 0 |

## Consolidation and renewal

Mean changes on a paired training-memory probe, equally weighted over consolidation events.
These are overlapping, potentially in-pool examples, not independent validation observations.

| Seed | Events with probes | Net transfer accuracy change, pp | Reset-only change, pp | Net loss change |
|---|---:|---:|---:|---:|
| 2027 | 9 | +0.3472 | +3.8194 | -0.0442 |
| 2131 | 9 | -1.0417 | -1.2153 | +0.0079 |

The two-pathway scratch reference trains both networks jointly on each new experience without replay.
It matches architecture, not consolidation policy or total compute. Scratch differences are diagnostic, not pure plasticity.
AUC integrates current presentations and includes any intervening consolidation work.
Low forgetting can reflect weak initial learning. Inspect acquisition together with retention.
The frozen-path ablation keeps the stable network at random initialization; no method uses pretrained weights.
Ten class groups and two seeds cannot establish sustained plasticity or a general architectural advantage.

See [the protocol](../../docs/dual_path.md) and [the portable archive](archive.json).
