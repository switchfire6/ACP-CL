# v2_shapes_stationary: completed-run diagnostics

Completed **9/9** runs: 3 methods x 3 seeds (101, 202, 303). Evaluation split: **validation**.

All table summaries are equal-weight seed means. Method order follows the manifest. Per-seed values, sample SDs, metric counts, source hashes, and full detector audits are in `diagnostics.json`.

A dagger (†) marks an **extra-information diagnostic**: oracle change times or offline allocation traces.

| Method | Seeds | Final accuracy (%) | Late early AUC (%) | Scratch gap (pp) | Input-centroid accuracy (%) |
|---|---:|---:|---:|---:|---:|
| er_recycle | 3 | 96.10 | 94.48 | +67.76 | 32.35 |
| acp_v2 | 3 | 79.20 | 78.31 | +51.59 | 32.35 |
| acp_v2_no_reopening | 3 | 72.93 | 85.03 | +58.31 | 32.35 |

The input-centroid column is an evaluator-only classifier using the final replay reservoir. Scratch gap subtracts a fresh model's acquisition AUC on the identical experience.

| Method | Unit resets | Mean gain | Sum data-update L2 | Local maturations | Useful local consolidations | Final mean c | Reopenings | Sensor bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| er_recycle | 301.0 | 1.0000 | 155.549 | 0.0 | 0.0 | 0.0000 | 0.0 | 0 |
| acp_v2 | 301.0 | 0.1315 | 38.573 | 295.0 | 236.0 | 0.1494 | 2.7 | 12544 |
| acp_v2_no_reopening | 301.0 | 0.1313 | 37.911 | 295.0 | 234.0 | 0.1259 | 0.0 | 12544 |

Mean gain covers every optimizer step and weights feature parameters within a step. Summed data-update L2 is path length, not net displacement; it excludes classifier, decay, and anchor forces. Final mean c summarizes final consolidation rather than averaging over time.

Detector proximity below pools counts across seeds. The recorded tolerance is in optimizer updates; these are descriptive proximity counts rather than calibrated detector scores.

| Method | Reopenings | Changes with nearby reopening / post-warmup changes | Reopenings outside change windows | Mean nearby delay (steps) |
|---|---:|---:|---:|---:|
| er_recycle | 0 | 0 / 0 | 0 | n/a |
| acp_v2 | 8 | 0 / 0 | 8 | n/a |
| acp_v2_no_reopening | 0 | 0 / 0 | 0 | n/a |

Per-seed results:

| Method | Seed | Final accuracy (%) | Late AUC (%) | Scratch gap (pp) | Unit resets | Mean gain | Local maturations | Reopenings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| er_recycle | 101 | 88.73 | 91.50 | +64.95 | 301 | 1.0000 | 0 | 0 |
| er_recycle | 202 | 99.73 | 98.02 | +72.10 | 301 | 1.0000 | 0 | 0 |
| er_recycle | 303 | 99.84 | 93.93 | +66.24 | 301 | 1.0000 | 0 | 0 |
| acp_v2 | 101 | 89.23 | 88.88 | +62.33 | 301 | 0.1248 | 295 | 2 |
| acp_v2 | 202 | 74.73 | 72.52 | +46.60 | 301 | 0.1348 | 295 | 2 |
| acp_v2 | 303 | 73.62 | 73.53 | +45.84 | 301 | 0.1350 | 295 | 4 |
| acp_v2_no_reopening | 101 | 89.22 | 88.79 | +62.24 | 301 | 0.1246 | 295 | 0 |
| acp_v2_no_reopening | 202 | 56.02 | 92.91 | +66.99 | 301 | 0.1347 | 295 | 0 |
| acp_v2_no_reopening | 303 | 73.57 | 73.39 | +45.70 | 301 | 0.1347 | 295 | 0 |

![Acquisition and allocation diagnostics](main_conditions.png)

The figure uses a fixed condition order. Curves show seed means; AUC bands show one sample SD when multiple seeds exist. Bars show late AUC means and dots show individual seeds. No inferential comparison is computed.

Training source SHA-256: `67ae36402607f257f1e55c4f3ef2bbb477a88555d2111a4bd0139e27668eee0a`.
