# Predictive rehearsal value: score validation

This is a fixed diagnostic of extra rehearsal value. The ordinary uniform-replay trajectory was unchanged. It does not evaluate an adaptive retention policy.

The independent unit is a seed after averaging its two assessments. Intervals are descriptive 95% paired seed bootstrap intervals (20,000 resamples). Allocation screens are not significance or noninferiority tests. Original-accuracy ties do not establish extra benefit over the cheaper signal.

| Model | Local tier | Return tier | Allocation |
|---|---|---|---|
| conditional | FAIL | FAIL | stop_expansion_of_fixed_score_recipe |
| recurrent | FAIL | FAIL | stop_expansion_of_fixed_score_recipe |

## Primary matched reapplication

Brier is shown ×100; survival differences are percentage points. Negative Brier and positive survival favor value selection.

| Model | Window | Value−uniform Brier [95% interval] | Improved seeds | Value−accuracy Brier [95% interval] | Survival−uniform [95% interval] |
|---|---|---:|---:|---:|---:|
| conditional | near | -0.0030 [-0.0625, +0.0585] | 8/12 | -0.0568 [-0.1256, +0.0126] | -0.0092 [-0.0987, +0.0824] |
| conditional | return | +0.0956 [-0.0824, +0.3249] | 7/12 | +0.0971 [-0.0331, +0.2624] | -0.0682 [-0.2879, +0.1099] |
| recurrent | near | +0.0224 [-0.0506, +0.1008] | 5/12 | -0.0042 [-0.0969, +0.0945] | +0.0417 [-0.1261, +0.2258] |
| recurrent | return | +0.1579 [-0.0125, +0.3080] | 3/12 | +0.0896 [-0.1521, +0.3340] | -0.1668 [-0.3560, +0.0509] |

- conditional, near: failed criteria = brier_gain, seed_consistency.
- conditional, return: failed criteria = brier_gain, seed_consistency, accuracy_control.
- recurrent, near: failed criteria = brier_gain, seed_consistency.
- recurrent, return: failed criteria = brier_gain, seed_consistency, accuracy_control.

## Every independent seed

Value−uniform Brier ×100, after averaging both assessments.

| Model | Seed | Local | Return |
|---|---:|---:|---:|
| conditional | 14001 | -0.1126 | +0.0767 |
| conditional | 14002 | -0.0328 | +0.0158 |
| conditional | 14003 | -0.0195 | -0.1973 |
| conditional | 14004 | -0.0246 | +0.1014 |
| conditional | 14005 | -0.1428 | -0.1009 |
| conditional | 14006 | +0.1659 | -0.0801 |
| conditional | 14007 | -0.1589 | +0.9434 |
| conditional | 14008 | +0.0867 | -0.1357 |
| conditional | 14009 | +0.1364 | -0.1756 |
| conditional | 14010 | -0.0193 | -0.0978 |
| conditional | 14011 | -0.0670 | +0.8167 |
| conditional | 14012 | +0.1524 | -0.0197 |
| recurrent | 14001 | +0.0290 | +0.1474 |
| recurrent | 14002 | -0.0891 | +0.1283 |
| recurrent | 14003 | +0.0252 | +0.2340 |
| recurrent | 14004 | -0.0603 | +0.5790 |
| recurrent | 14005 | -0.0252 | +0.1358 |
| recurrent | 14006 | +0.1550 | +0.4050 |
| recurrent | 14007 | -0.1074 | -0.5399 |
| recurrent | 14008 | +0.0199 | +0.3924 |
| recurrent | 14009 | +0.3092 | -0.0314 |
| recurrent | 14010 | -0.1890 | -0.1547 |
| recurrent | 14011 | +0.1964 | +0.2221 |
| recurrent | 14012 | +0.0054 | +0.3770 |

## Descriptive controls and references

The score-window result selects the winner and is not validation. The oracle uses validation Brier to choose one candidate; it is not deployable and does not optimize survival. The replacement is common to all candidates. Uniform averages candidate losses, not their predictions.

| Model | Panel | Accuracy−uniform Brier | Value−replacement Brier | Oracle−uniform Brier | Value−oracle regret | Candidate spread |
|---|---|---:|---:|---:|---:|---:|
| conditional | score | -0.0491 | -0.2032 | -0.3083 | +0.0000 | +0.6702 |
| conditional | near_reapplied | +0.0538 | +0.0081 | -0.2766 | +0.2736 | +0.5330 |
| conditional | near_frozen | -0.0251 | -0.0971 | -0.3113 | +0.1697 | +0.6014 |
| conditional | return_reapplied | -0.0016 | -0.0793 | -0.4976 | +0.5932 | +1.0734 |
| conditional | return_frozen | +0.0283 | -0.0411 | -0.3815 | +0.3436 | +0.7544 |
| recurrent | score | -0.0305 | -0.1912 | -0.2885 | +0.0000 | +0.5590 |
| recurrent | near_reapplied | +0.0266 | -0.0132 | -0.2239 | +0.2463 | +0.4943 |
| recurrent | near_frozen | -0.0680 | -0.0820 | -0.2774 | +0.1912 | +0.5882 |
| recurrent | return_reapplied | +0.0683 | +0.1229 | -1.0243 | +1.1822 | +1.6653 |
| recurrent | return_frozen | +0.1813 | +0.0808 | -0.7251 | +0.8063 | +1.3475 |

## Coverage and interpretation

- conditional: 19/24 assessments included a candidate from the exact returning law; all assessments remain in the primary estimate. 1 assessments had candidate query/support overlap.
- recurrent: 19/24 assessments included a candidate from the exact returning law; all assessments remain in the primary estimate. 1 assessments had candidate query/support overlap.

Complete assessment rows, cue/gap/order slices, early/late losses, original prequential errors and steps, frozen-shadow comparisons, exact selector means and all seed differences are preserved in summary.json. Frozen versus reapplied contrasts combine changed parent weights, optimizer state and anchor; short versus long gaps combine learning exposure, arrival position and candidate age. Neither isolates elapsed time. This clean, known-schedule experiment does not establish noise discrimination, memory deletion value, longer physical planning or an improved deployed continual learner.
