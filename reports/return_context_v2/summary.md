# Adam: matched return-context diagnostic

Retrospective descriptive analysis of the completed evidence-consolidation pilot. The analysis was locked before these additional query evaluations, after the original outcomes were known. There is no new method-selection screen or independent confirmation.

All 6 seeds per architecture and all three returns were retained. 1428 weight/support evaluations use 512 matched queries each. Support replicates are averaged within a return, then returns within a seed. Intervals are descriptive 95% paired-seed bootstrap intervals (20,000 resamples).

Brier differences below are multiplied by 100; negative favors the first named predictor. For context penalties, positive means lower error with target-law support.

| Matched comparison | Conditional | Recurrent |
|---|---:|---:|
| Sustained minus draft, actual preceding history | -0.123 [-0.473, +0.256] | -0.080 [-0.595, +0.576] |
| Sustained minus draft, 32 observed return records | +2.760 [+0.214, +5.524] | +0.705 [-0.294, +1.787] |
| Sustained minus draft, fresh target support | +0.448 [-0.540, +1.960] | -0.276 [-0.740, +0.303] |
| Draft context penalty, 32 observed return records | +20.921 [+18.598, +23.245] | +3.648 [+1.473, +6.066] |
| Sustained context penalty, 32 observed return records | +18.039 [+14.407, +21.512] | +2.864 [+1.019, +4.896] |
| Current draft minus prefix, fresh target support | +9.461 [+7.525, +11.471] | +20.033 [+15.684, +24.252] |
| Current draft minus previous return, fresh target support (cycles 2/3) | +6.233 [+4.996, +7.392] | +25.232 [+23.769, +26.529] |
| Return-end minus return-start draft, fresh target support | -8.294 [-10.331, -6.283] | -25.228 [-26.842, -23.611] |
| Return-end draft minus prefix, fresh target support | +1.167 [+0.771, +1.557] | -5.195 [-9.316, -1.321] |
| Return-end draft minus previous return, fresh target support (cycles 2/3) | -0.053 [-0.242, +0.150] | +1.313 [-0.465, +3.532] |

The exact prediction-level identity is actual-history gap = target-support gap + (sustained context penalty - draft context penalty). It is not a decomposition of total online return loss into independently identified causes.

| Context interaction, Brier x100 | Conditional | Recurrent |
|---|---:|---:|
| observed_32 | -2.882 [-5.690, -0.307] | -0.785 [-2.056, +0.136] |
| fresh_target | -0.571 [-2.214, +0.729] | +0.196 [-0.191, +0.473] |

## Interpretation limits

The preceding history is clean but belongs to the previous law. The return changes both base mode and active dependencies without announcing either in the images. Correct-support probes therefore concern contextual applicability, not specifically noise removal.

The first return packet is ordinary observed feedback, used here with frozen weights on an independent query panel. Eight/sixteen-record replacement is a diagnostic subsampling probe. Fresh target support is evaluator-selected; actual-end support and end weights come from the future relative to the start. None is silently supplied to an online learner.

Prefix and previous-return comparisons concern accessible predictions, not proof of neuron-level erasure. Conditional best-head references use these query outcomes for selection and are nondeployable; a mixture can outperform every single head. Recurrent models have no corresponding explicit-head reference. No representation was changed or trained.

All rows, matched contrasts, six seed values, weight/support/query fingerprints, provenance, and the code/protocol lock are retained in summary.json, rows.jsonl, provenance.json, and analysis_lock.json. No input checkpoints or sealed archive artifacts were modified.
