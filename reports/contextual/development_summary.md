# Recurrent comparator development

36 prefix fits and 108 challenge branches. Two development seeds per setting.

Thresholds: oracle and recurrent initial acquisition >=5 pp over no transfer;
recurrent return >=5 pp over no transfer; correct-minus-opposite history benefit >=2 pp.

| Recipe | Updates | Oracle acquisition (pp) | Recurrent acquisition (pp) | Return gain (pp) | History benefit (pp) | Qualifies |
|---|---:|---:|---:|---:|---:|---|
| plain | 3 | +12.142 | +7.389 | +5.339 | -0.033 | no |
| plain | 6 | +12.435 | +8.724 | -4.687 | +0.098 | no |
| plain | 12 | +12.663 | +10.710 | -5.306 | -0.586 | no |
| interactions | 3 | +12.142 | +7.454 | +2.311 | +0.228 | no |
| interactions | 6 | +12.435 | +8.952 | -1.693 | +0.488 | no |
| interactions | 12 | +12.663 | +11.328 | +7.682 | +15.202 | yes |

All seeds, branches, configurations and exact training sources are preserved in `development/`.
These repeated development outcomes do not provide independent confirmatory evidence.
Do not pool the six settings as twelve independent seeds or interpret a failed comparator as a failed candidate.
