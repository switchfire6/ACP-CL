# Predictive rehearsal value: score validation

This is a fixed diagnostic of extra rehearsal value. The ordinary uniform-replay trajectory was unchanged. It does not evaluate an adaptive retention policy.

The independent unit is a seed after averaging its two assessments. Intervals are descriptive 95% paired seed bootstrap intervals (20,000 resamples). Allocation screens are not significance or noninferiority tests. Original-accuracy ties do not establish extra benefit over the cheaper signal.

| Model | Local tier | Return tier | Allocation |
|---|---|---|---|
| conditional | FAIL | FAIL | stop_expansion_of_fixed_score_recipe |
| recurrent | FAIL | FAIL | stop_expansion_of_fixed_score_recipe |

This engineering/subset cohort is ineligible for scientific continuation criteria.

## Primary matched reapplication

Brier is shown ×100; survival differences are percentage points. Negative Brier and positive survival favor value selection.

| Model | Window | Value−uniform Brier [95% interval] | Improved seeds | Value−accuracy Brier [95% interval] | Survival−uniform [95% interval] |
|---|---|---:|---:|---:|---:|
| conditional | near | -0.0000 [-0.0000, -0.0000] | 1/1 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| conditional | return | -0.0087 [-0.0087, -0.0087] | 1/1 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| recurrent | near | +0.0021 [+0.0021, +0.0021] | 0/1 | +0.0068 [+0.0068, +0.0068] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | -0.0157 [-0.0157, -0.0157] | 1/1 | -0.0015 [-0.0015, -0.0015] | +1.5625 [+1.5625, +1.5625] |

- conditional, near: failed criteria = complete_scientific_cohort, brier_gain, seed_consistency.
- conditional, return: failed criteria = complete_scientific_cohort, brier_gain, seed_consistency.
- recurrent, near: failed criteria = complete_scientific_cohort, brier_gain, seed_consistency, accuracy_control.
- recurrent, return: failed criteria = complete_scientific_cohort, brier_gain, seed_consistency.

## Every independent seed

Value−uniform Brier ×100, after averaging both assessments.

| Model | Seed | Local | Return |
|---|---:|---:|---:|
| conditional | 14991 | -0.0000 | -0.0087 |
| recurrent | 14991 | +0.0021 | -0.0157 |

## Descriptive controls and references

The score-window result selects the winner and is not validation. The oracle uses validation Brier to choose one candidate; it is not deployable and does not optimize survival. The replacement is common to all candidates. Uniform averages candidate losses, not their predictions.

| Model | Panel | Accuracy−uniform Brier | Value−replacement Brier | Oracle−uniform Brier | Value−oracle regret | Candidate spread |
|---|---|---:|---:|---:|---:|---:|
| conditional | score | -0.0048 | -0.0044 | -0.0048 | +0.0000 | +0.0095 |
| conditional | near_reapplied | -0.0000 | +0.0030 | -0.0007 | +0.0007 | +0.0015 |
| conditional | near_frozen | -0.0019 | +0.0005 | -0.0019 | +0.0000 | +0.0038 |
| conditional | return_reapplied | -0.0087 | -0.0054 | -0.0087 | +0.0000 | +0.0174 |
| conditional | return_frozen | -0.0065 | -0.0034 | -0.0065 | +0.0000 | +0.0130 |
| recurrent | score | -0.0012 | -0.0018 | -0.0041 | +0.0000 | +0.0082 |
| recurrent | near_reapplied | -0.0047 | -0.0027 | -0.0047 | +0.0068 | +0.0093 |
| recurrent | near_frozen | -0.0049 | -0.0045 | -0.0049 | +0.0061 | +0.0099 |
| recurrent | return_reapplied | -0.0142 | -0.0108 | -0.0157 | +0.0000 | +0.0314 |
| recurrent | return_frozen | -0.0152 | -0.0100 | -0.0152 | +0.0008 | +0.0304 |

## Coverage and interpretation

- conditional: 1/2 assessments included a candidate from the exact returning law; all assessments remain in the primary estimate. 0 assessments had candidate query/support overlap.
- recurrent: 1/2 assessments included a candidate from the exact returning law; all assessments remain in the primary estimate. 0 assessments had candidate query/support overlap.

Complete assessment rows, cue/gap/order slices, early/late losses, original prequential errors and steps, frozen-shadow comparisons, exact selector means and all seed differences are preserved in summary.json. Frozen versus reapplied contrasts combine changed parent weights, optimizer state and anchor; short versus long gaps combine learning exposure, arrival position and candidate age. Neither isolates elapsed time. This clean, known-schedule experiment does not establish noise discrimination, memory deletion value, longer physical planning or an improved deployed continual learner.
