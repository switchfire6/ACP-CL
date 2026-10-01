# Rehearsal state: controlled weight, Adam and anchor substitutions

Cell bits are weights, complete Adam state and the complete causal anchor packet (0 old, 1 new). All cells use the same retained candidates and selected IDs. D is selected-candidate loss minus the exact mean candidate loss. The primary response is current-branch T=D111−D000, with two assessments averaged per seed.

The replication screen requires mean T≥0.0005 and positive T in at least 9/12 seeds. It is a research allocation screen, not a significance test or policy promotion. All intervals are descriptive seed-bootstrap 95% intervals using 20,000 resamples. Zero-update references are outside the candidate pool.

| Model | T Brier ×100 [95% interval] | Positive seeds | Replication screen |
|---|---:|---:|---|
| conditional | -0.0439 [-0.1399, +0.0406] | 6/12 | FAIL |
| recurrent | +0.0825 [-0.0048, +0.1790] | 8/12 | FAIL |

- conditional: failed criteria = deterioration, seed_consistency.
- recurrent: failed criteria = seed_consistency.

## Every cell

Brier ×100 and survival %. Oracle is conditional on validation Brier, not deployable and not independently optimized for survival.

| Model | Branch | Cell | Chosen Brier | Original-error Brier | Uniform Brier | Replacement Brier | Oracle Brier | D [interval] | Chosen survival | Uniform survival |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conditional | current | 000 | 7.3316 | 7.3606 | 7.3966 | 7.3232 | 7.0641 | -0.0651 [-0.1612, +0.0329] | 69.865 | 69.828 |
| conditional | current | 001 | 7.5446 | 7.5450 | 7.5656 | 7.6026 | 7.2266 | -0.0209 [-0.1148, +0.0814] | 69.914 | 69.946 |
| conditional | current | 010 | 7.3659 | 7.3657 | 7.4057 | 7.3532 | 7.1042 | -0.0398 [-0.1316, +0.0470] | 70.085 | 69.977 |
| conditional | current | 011 | 7.6251 | 7.6285 | 7.6568 | 7.6583 | 7.2697 | -0.0317 [-0.1129, +0.0592] | 69.995 | 69.970 |
| conditional | current | 100 | 7.3713 | 7.4120 | 7.4281 | 7.4121 | 7.1521 | -0.0568 [-0.1330, +0.0215] | 70.109 | 70.122 |
| conditional | current | 101 | 7.3648 | 7.4454 | 7.4395 | 7.4739 | 7.1447 | -0.0747 [-0.1440, -0.0032] | 69.735 | 69.768 |
| conditional | current | 110 | 7.3984 | 7.4638 | 7.5137 | 7.4755 | 7.1630 | -0.1153 [-0.2247, -0.0175] | 70.036 | 70.047 |
| conditional | current | 111 | 7.3903 | 7.4919 | 7.4992 | 7.4807 | 7.1976 | -0.1090 [-0.1736, -0.0428] | 69.735 | 69.712 |
| conditional | return | 000 | 19.2150 | 18.9394 | 19.0738 | 19.0822 | 18.6022 | +0.1412 [-0.0533, +0.3474] | 64.054 | 64.185 |
| conditional | return | 001 | 19.2485 | 19.1478 | 19.2176 | 19.3620 | 18.6862 | +0.0309 [-0.1064, +0.1695] | 64.103 | 64.086 |
| conditional | return | 010 | 18.9775 | 18.9790 | 19.0350 | 19.1210 | 18.4751 | -0.0575 [-0.2804, +0.1632] | 64.266 | 64.251 |
| conditional | return | 011 | 19.1961 | 19.1359 | 19.2376 | 19.3520 | 18.6143 | -0.0415 [-0.2874, +0.1823] | 64.111 | 64.173 |
| conditional | return | 100 | 19.1533 | 19.0623 | 19.0786 | 19.0113 | 18.6311 | +0.0747 [-0.1219, +0.2976] | 64.453 | 64.460 |
| conditional | return | 101 | 19.1002 | 19.0362 | 19.0674 | 18.9939 | 18.5748 | +0.0328 [-0.2315, +0.2897] | 64.469 | 64.508 |
| conditional | return | 110 | 19.0719 | 19.0419 | 19.0720 | 19.0022 | 18.6341 | -0.0001 [-0.1847, +0.1555] | 64.461 | 64.493 |
| conditional | return | 111 | 19.1283 | 18.9102 | 19.0377 | 18.9958 | 18.6113 | +0.0905 [-0.1094, +0.2940] | 64.616 | 64.501 |
| recurrent | current | 000 | 7.2767 | 7.3695 | 7.3466 | 7.3924 | 7.0857 | -0.0699 [-0.1655, +0.0054] | 69.743 | 69.673 |
| recurrent | current | 001 | 7.5703 | 7.5340 | 7.5967 | 7.6284 | 7.2927 | -0.0264 [-0.1174, +0.0695] | 69.881 | 69.697 |
| recurrent | current | 010 | 7.3396 | 7.3451 | 7.3542 | 7.3928 | 7.0958 | -0.0146 [-0.0975, +0.0603] | 69.881 | 69.739 |
| recurrent | current | 011 | 7.6187 | 7.5512 | 7.6187 | 7.6525 | 7.2959 | -0.0000 [-0.1017, +0.1038] | 69.735 | 69.662 |
| recurrent | current | 100 | 7.3566 | 7.4253 | 7.3579 | 7.3675 | 7.1086 | -0.0013 [-0.0730, +0.0704] | 69.865 | 69.815 |
| recurrent | current | 101 | 7.3410 | 7.3284 | 7.2982 | 7.3600 | 7.0455 | +0.0429 [-0.0186, +0.1098] | 69.808 | 69.756 |
| recurrent | current | 110 | 7.3516 | 7.4247 | 7.3631 | 7.3242 | 7.0995 | -0.0114 [-0.0728, +0.0512] | 69.897 | 69.881 |
| recurrent | current | 111 | 7.3730 | 7.3710 | 7.3604 | 7.3623 | 7.1212 | +0.0126 [-0.0352, +0.0598] | 69.792 | 69.714 |
| recurrent | return | 000 | 35.7473 | 35.7297 | 35.7237 | 35.8822 | 35.0754 | +0.0235 [-0.1691, +0.1984] | 53.068 | 53.220 |
| recurrent | return | 001 | 35.6350 | 35.4249 | 35.5230 | 35.5888 | 34.8808 | +0.1121 [-0.0973, +0.3188] | 53.263 | 53.255 |
| recurrent | return | 010 | 35.7142 | 35.7655 | 35.7632 | 35.9396 | 35.1473 | -0.0490 [-0.2268, +0.1211] | 53.019 | 53.057 |
| recurrent | return | 011 | 35.5647 | 35.4896 | 35.6008 | 35.6880 | 34.9614 | -0.0360 [-0.2611, +0.1878] | 53.060 | 53.156 |
| recurrent | return | 100 | 36.0278 | 35.9777 | 35.9518 | 36.0115 | 35.3555 | +0.0760 [-0.1753, +0.3408] | 52.710 | 52.655 |
| recurrent | return | 101 | 35.6988 | 35.5964 | 35.6394 | 35.5472 | 35.0012 | +0.0594 [-0.1329, +0.2752] | 52.743 | 52.887 |
| recurrent | return | 110 | 36.1069 | 36.1091 | 36.1038 | 36.0985 | 35.5303 | +0.0030 [-0.2461, +0.2376] | 52.531 | 52.486 |
| recurrent | return | 111 | 35.6669 | 35.5359 | 35.6277 | 35.5555 | 35.0161 | +0.0392 [-0.1271, +0.2268] | 52.832 | 52.835 |

## Factorial effects and prespecified repairs

Positive D effects mean worse selected-versus-uniform usefulness. Positive chosen-loss repair effects mean the old-component repair reduces chosen loss; positive survival effects instead favor cell 111. Main effects average over the other two factors. Interactions prevent unique component blame.

| Model | Branch | Effect | D Brier ×100 [interval] | Chosen Brier change ×100 [interval] | Chosen survival change, pp [interval] |
|---|---|---|---:|---:|---:|
| conditional | current | total_111_minus_000 | -0.0439 [-0.1399, +0.0406] | +0.0587 [-0.1612, +0.2976] | -0.1302 [-0.6429, +0.3337] |
| conditional | current | weights_main | -0.0496 [-0.1115, +0.0086] | -0.0856 [-0.2004, +0.0535] | -0.0610 [-0.4476, +0.3072] |
| conditional | current | weights_at_optimizer0_anchor0 | +0.0083 [-0.1216, +0.1421] | +0.0397 [-0.1498, +0.2079] | +0.2441 [-0.1955, +0.6429] |
| conditional | current | weights_at_optimizer0_anchor1 | -0.0537 [-0.1311, +0.0316] | -0.1798 [-0.3390, +0.0311] | -0.1790 [-0.5778, +0.1709] |
| conditional | current | weights_at_optimizer1_anchor0 | -0.0755 [-0.1780, +0.0230] | +0.0325 [-0.1012, +0.1628] | -0.0488 [-0.4150, +0.3337] |
| conditional | current | weights_at_optimizer1_anchor1 | -0.0773 [-0.1711, +0.0116] | -0.2349 [-0.4226, +0.0214] | -0.2604 [-0.7731, +0.2604] |
| conditional | current | repair_weights | -0.0773 [-0.1711, +0.0116] | -0.2349 [-0.4226, +0.0214] | -0.2604 [-0.7731, +0.2604] |
| conditional | current | optimizer_main | -0.0196 [-0.0590, +0.0198] | +0.0419 [-0.0199, +0.1026] | +0.0570 [-0.0631, +0.1648] |
| conditional | current | optimizer_at_weights0_anchor0 | +0.0253 [-0.0399, +0.0946] | +0.0344 [-0.1012, +0.1601] | +0.2197 [+0.0326, +0.3988] |
| conditional | current | optimizer_at_weights0_anchor1 | -0.0107 [-0.0573, +0.0344] | +0.0805 [+0.0211, +0.1348] | +0.0814 [-0.0895, +0.2441] |
| conditional | current | optimizer_at_weights1_anchor0 | -0.0585 [-0.1233, +0.0122] | +0.0271 [-0.0556, +0.1168] | -0.0732 [-0.2686, +0.1221] |
| conditional | current | optimizer_at_weights1_anchor1 | -0.0343 [-0.1003, +0.0347] | +0.0255 [-0.0847, +0.1295] | +0.0000 [-0.1953, +0.1709] |
| conditional | current | repair_optimizer | -0.0343 [-0.1003, +0.0347] | +0.0255 [-0.0847, +0.1295] | +0.0000 [-0.1953, +0.1709] |
| conditional | current | anchor_main | +0.0102 [-0.0502, +0.0667] | +0.1144 [-0.0044, +0.2391] | -0.1790 [-0.3011, -0.0631] |
| conditional | current | anchor_at_weights0_optimizer0 | +0.0441 [-0.0575, +0.1625] | +0.2130 [-0.0309, +0.4377] | +0.0488 [-0.1139, +0.2116] |
| conditional | current | anchor_at_weights0_optimizer1 | +0.0081 [-0.0740, +0.0896] | +0.2592 [+0.0359, +0.4533] | -0.0895 [-0.2848, +0.0977] |
| conditional | current | anchor_at_weights1_optimizer0 | -0.0179 [-0.1101, +0.0641] | -0.0064 [-0.0962, +0.1017] | -0.3743 [-0.5859, -0.1790] |
| conditional | current | anchor_at_weights1_optimizer1 | +0.0063 [-0.1018, +0.1337] | -0.0081 [-0.1573, +0.1303] | -0.3011 [-0.6104, +0.0000] |
| conditional | current | repair_anchor | +0.0063 [-0.1018, +0.1337] | -0.0081 [-0.1573, +0.1303] | -0.3011 [-0.6104, +0.0000] |
| conditional | current | weights_optimizer_interaction | -0.0537 [-0.1231, +0.0159] | -0.0311 [-0.1231, +0.0666] | -0.1872 [-0.3703, -0.0122] |
| conditional | current | weights_anchor_interaction | -0.0319 [-0.1817, +0.1112] | -0.2434 [-0.4667, +0.0058] | -0.3174 [-0.6348, -0.0285] |
| conditional | current | optimizer_anchor_interaction | -0.0059 [-0.0626, +0.0480] | +0.0222 [-0.0764, +0.1116] | -0.0326 [-0.1750, +0.1058] |
| conditional | current | three_way_interaction | +0.0602 [-0.0164, +0.1510] | -0.0478 [-0.2172, +0.1002] | +0.2116 [-0.1221, +0.5697] |
| conditional | return | total_111_minus_000 | -0.0507 [-0.2214, +0.1140] | -0.0867 [-0.5734, +0.3988] | +0.5615 [+0.1709, +0.9603] |
| conditional | return | weights_main | +0.0312 [-0.1272, +0.1803] | -0.0458 [-0.4082, +0.3408] | +0.3662 [-0.0203, +0.8667] |
| conditional | return | weights_at_optimizer0_anchor0 | -0.0665 [-0.3461, +0.1727] | -0.0617 [-0.5968, +0.4833] | +0.3988 [-0.0081, +0.8057] |
| conditional | return | weights_at_optimizer0_anchor1 | +0.0019 [-0.2284, +0.2474] | -0.1483 [-0.5439, +0.2505] | +0.3662 [-0.1058, +0.9521] |
| conditional | return | weights_at_optimizer1_anchor0 | +0.0574 [-0.1543, +0.2764] | +0.0944 [-0.2940, +0.5465] | +0.1953 [-0.3825, +0.8138] |
| conditional | return | weights_at_optimizer1_anchor1 | +0.1320 [-0.0605, +0.3456] | -0.0678 [-0.4357, +0.3099] | +0.5046 [+0.0000, +1.1149] |
| conditional | return | repair_weights | +0.1320 [-0.0605, +0.3456] | -0.0678 [-0.4357, +0.3099] | +0.5046 [+0.0000, +1.1149] |
| conditional | return | optimizer_main | -0.0720 [-0.1434, -0.0050] | -0.0858 [-0.1742, +0.0106] | +0.0936 [-0.0285, +0.2299] |
| conditional | return | optimizer_at_weights0_anchor0 | -0.1987 [-0.4183, -0.0237] | -0.2375 [-0.4574, -0.0403] | +0.2116 [-0.0244, +0.4150] |
| conditional | return | optimizer_at_weights0_anchor1 | -0.0724 [-0.2306, +0.0641] | -0.0524 [-0.2413, +0.1402] | +0.0081 [-0.1628, +0.1953] |
| conditional | return | optimizer_at_weights1_anchor0 | -0.0748 [-0.2208, +0.0660] | -0.0814 [-0.2938, +0.1089] | +0.0081 [-0.2441, +0.2686] |
| conditional | return | optimizer_at_weights1_anchor1 | +0.0577 [-0.0634, +0.1726] | +0.0281 [-0.1482, +0.2246] | +0.1465 [-0.0732, +0.3581] |
| conditional | return | repair_optimizer | +0.0577 [-0.0634, +0.1726] | +0.0281 [-0.1482, +0.2246] | +0.1465 [-0.0732, +0.3581] |
| conditional | return | anchor_main | -0.0114 [-0.0773, +0.0484] | +0.0639 [-0.0434, +0.1592] | +0.0163 [-0.1709, +0.1872] |
| conditional | return | anchor_at_weights0_optimizer0 | -0.1103 [-0.3153, +0.0582] | +0.0335 [-0.2821, +0.3064] | +0.0488 [-0.3092, +0.3581] |
| conditional | return | anchor_at_weights0_optimizer1 | +0.0161 [-0.1408, +0.1796] | +0.2186 [-0.0153, +0.4613] | -0.1546 [-0.4639, +0.1302] |
| conditional | return | anchor_at_weights1_optimizer0 | -0.0419 [-0.1881, +0.1010] | -0.0531 [-0.2682, +0.1456] | +0.0163 [-0.2686, +0.3499] |
| conditional | return | anchor_at_weights1_optimizer1 | +0.0906 [-0.0199, +0.2039] | +0.0564 [-0.1263, +0.2299] | +0.1546 [-0.1790, +0.5046] |
| conditional | return | repair_anchor | +0.0906 [-0.0199, +0.2039] | +0.0564 [-0.1263, +0.2299] | +0.1546 [-0.1790, +0.5046] |
| conditional | return | weights_optimizer_interaction | +0.1270 [-0.0799, +0.3445] | +0.1183 [-0.1306, +0.3548] | -0.0326 [-0.2726, +0.1872] |
| conditional | return | weights_anchor_interaction | +0.0715 [-0.1420, +0.2930] | -0.1244 [-0.4574, +0.2181] | +0.1383 [-0.2808, +0.6266] |
| conditional | return | optimizer_anchor_interaction | +0.1294 [+0.0496, +0.2310] | +0.1473 [-0.0135, +0.3082] | -0.0326 [-0.1750, +0.1139] |
| conditional | return | three_way_interaction | +0.0061 [-0.3027, +0.3118] | -0.0756 [-0.4191, +0.3463] | +0.3418 [-0.1139, +0.8138] |
| recurrent | current | total_111_minus_000 | +0.0825 [-0.0048, +0.1790] | +0.0962 [-0.1212, +0.2930] | +0.0488 [-0.4313, +0.5290] |
| recurrent | current | weights_main | +0.0384 [-0.0377, +0.1088] | -0.0958 [-0.3323, +0.1125] | +0.0305 [-0.4659, +0.5168] |
| recurrent | current | weights_at_optimizer0_anchor0 | +0.0686 [-0.0224, +0.1575] | +0.0799 [-0.2014, +0.3411] | +0.1221 [-0.3581, +0.5859] |
| recurrent | current | weights_at_optimizer0_anchor1 | +0.0693 [-0.0466, +0.1767] | -0.2293 [-0.5421, +0.0653] | -0.0732 [-0.6104, +0.4639] |
| recurrent | current | weights_at_optimizer1_anchor0 | +0.0032 [-0.0645, +0.0661] | +0.0120 [-0.2217, +0.2436] | +0.0163 [-0.6348, +0.6592] |
| recurrent | current | weights_at_optimizer1_anchor1 | +0.0126 [-0.1012, +0.1244] | -0.2457 [-0.4657, -0.0282] | +0.0570 [-0.4476, +0.5452] |
| recurrent | current | repair_weights | +0.0126 [-0.1012, +0.1244] | -0.2457 [-0.4657, -0.0282] | +0.0570 [-0.4476, +0.5452] |
| recurrent | current | optimizer_main | +0.0103 [-0.0203, +0.0385] | +0.0346 [-0.0262, +0.0995] | +0.0020 [-0.1241, +0.1282] |
| recurrent | current | optimizer_at_weights0_anchor0 | +0.0553 [+0.0001, +0.1117] | +0.0629 [-0.0306, +0.1601] | +0.1383 [-0.1465, +0.4232] |
| recurrent | current | optimizer_at_weights0_anchor1 | +0.0264 [-0.0415, +0.1010] | +0.0483 [-0.0481, +0.1449] | -0.1465 [-0.3418, +0.0326] |
| recurrent | current | optimizer_at_weights1_anchor0 | -0.0101 [-0.0487, +0.0299] | -0.0049 [-0.1226, +0.0985] | +0.0326 [-0.1546, +0.2116] |
| recurrent | current | optimizer_at_weights1_anchor1 | -0.0303 [-0.0761, +0.0225] | +0.0319 [-0.0738, +0.1324] | -0.0163 [-0.2279, +0.2197] |
| recurrent | current | repair_optimizer | -0.0303 [-0.0761, +0.0225] | +0.0319 [-0.0738, +0.1324] | -0.0163 [-0.2279, +0.2197] |
| recurrent | current | anchor_main | +0.0316 [-0.0499, +0.1146] | +0.1446 [-0.0248, +0.3093] | -0.0427 [-0.2665, +0.1912] |
| recurrent | current | anchor_at_weights0_optimizer0 | +0.0435 [-0.0743, +0.1671] | +0.2936 [+0.0463, +0.5404] | +0.1383 [-0.2279, +0.4801] |
| recurrent | current | anchor_at_weights0_optimizer1 | +0.0146 [-0.1118, +0.1467] | +0.2790 [+0.0597, +0.4984] | -0.1465 [-0.4232, +0.1302] |
| recurrent | current | anchor_at_weights1_optimizer0 | +0.0441 [-0.0395, +0.1164] | -0.0155 [-0.2164, +0.1514] | -0.0570 [-0.3988, +0.3174] |
| recurrent | current | anchor_at_weights1_optimizer1 | +0.0240 [-0.0521, +0.0930] | +0.0213 [-0.1775, +0.1978] | -0.1058 [-0.4232, +0.1953] |
| recurrent | current | repair_anchor | +0.0240 [-0.0521, +0.0930] | +0.0213 [-0.1775, +0.1978] | -0.1058 [-0.4232, +0.1953] |
| recurrent | current | weights_optimizer_interaction | -0.0611 [-0.1228, -0.0005] | -0.0421 [-0.1510, +0.0622] | +0.0122 [-0.1790, +0.2197] |
| recurrent | current | weights_anchor_interaction | +0.0050 [-0.1003, +0.1044] | -0.2834 [-0.5111, -0.0512] | -0.0773 [-0.4395, +0.2523] |
| recurrent | current | optimizer_anchor_interaction | -0.0245 [-0.0821, +0.0228] | +0.0111 [-0.1029, +0.1081] | -0.1668 [-0.3988, +0.0651] |
| recurrent | current | three_way_interaction | +0.0087 [-0.0795, +0.1038] | +0.0514 [-0.0670, +0.1670] | +0.2360 [-0.1546, +0.6673] |
| recurrent | return | total_111_minus_000 | +0.0157 [-0.2237, +0.2888] | -0.0804 [-0.6976, +0.5217] | -0.2360 [-1.0417, +0.5778] |
| recurrent | return | weights_main | +0.0318 [-0.2300, +0.3059] | +0.2098 [-0.4092, +0.8325] | -0.3988 [-1.0010, +0.1607] |
| recurrent | return | weights_at_optimizer0_anchor0 | +0.0524 [-0.2793, +0.4028] | +0.2806 [-0.4081, +0.9967] | -0.3581 [-0.9359, +0.2197] |
| recurrent | return | weights_at_optimizer0_anchor1 | -0.0527 [-0.3400, +0.2147] | +0.0638 [-0.6151, +0.7074] | -0.5208 [-1.1963, +0.1383] |
| recurrent | return | weights_at_optimizer1_anchor0 | +0.0521 [-0.2660, +0.3606] | +0.3927 [-0.2132, +1.0257] | -0.4883 [-1.0905, +0.0570] |
| recurrent | return | weights_at_optimizer1_anchor1 | +0.0753 [-0.2029, +0.3532] | +0.1022 [-0.5712, +0.7667] | -0.2279 [-0.9684, +0.4720] |
| recurrent | return | repair_weights | +0.0753 [-0.2029, +0.3532] | +0.1022 [-0.5712, +0.7667] | -0.2279 [-0.9684, +0.4720] |
| recurrent | return | optimizer_main | -0.0784 [-0.1345, -0.0324] | -0.0141 [-0.1864, +0.1843] | -0.0854 [-0.2380, +0.0774] |
| recurrent | return | optimizer_at_weights0_anchor0 | -0.0725 [-0.1274, -0.0225] | -0.0331 [-0.2494, +0.1814] | -0.0488 [-0.3011, +0.1953] |
| recurrent | return | optimizer_at_weights0_anchor1 | -0.1481 [-0.2408, -0.0667] | -0.0703 [-0.2633, +0.1266] | -0.2035 [-0.3908, +0.0244] |
| recurrent | return | optimizer_at_weights1_anchor0 | -0.0729 [-0.1926, +0.0439] | +0.0791 [-0.1820, +0.3750] | -0.1790 [-0.3743, +0.0163] |
| recurrent | return | optimizer_at_weights1_anchor1 | -0.0201 [-0.1036, +0.0618] | -0.0319 [-0.1895, +0.1670] | +0.0895 [-0.1546, +0.3337] |
| recurrent | return | repair_optimizer | -0.0201 [-0.1036, +0.0618] | -0.0319 [-0.1895, +0.1670] | +0.0895 [-0.1546, +0.3337] |
| recurrent | return | anchor_main | +0.0303 [-0.0246, +0.0938] | -0.2577 [-0.5242, +0.0051] | +0.1424 [-0.1078, +0.3967] |
| recurrent | return | anchor_at_weights0_optimizer0 | +0.0885 [-0.0780, +0.2641] | -0.1122 [-0.5386, +0.3021] | +0.1953 [-0.1953, +0.5941] |
| recurrent | return | anchor_at_weights0_optimizer1 | +0.0130 [-0.1384, +0.1805] | -0.1495 [-0.5783, +0.2620] | +0.0407 [-0.2604, +0.3499] |
| recurrent | return | anchor_at_weights1_optimizer0 | -0.0166 [-0.1560, +0.1470] | -0.3290 [-0.5409, -0.0808] | +0.0326 [-0.3662, +0.4232] |
| recurrent | return | anchor_at_weights1_optimizer1 | +0.0362 [-0.0820, +0.1601] | -0.4400 [-0.6835, -0.1899] | +0.3011 [-0.0651, +0.6348] |
| recurrent | return | repair_anchor | +0.0362 [-0.0820, +0.1601] | -0.4400 [-0.6835, -0.1899] | +0.3011 [-0.0651, +0.6348] |
| recurrent | return | weights_optimizer_interaction | +0.0638 [-0.0098, +0.1576] | +0.0753 [-0.1134, +0.2659] | +0.0814 [-0.0937, +0.2686] |
| recurrent | return | weights_anchor_interaction | -0.0410 [-0.2889, +0.2229] | -0.2537 [-0.6627, +0.1534] | +0.0488 [-0.4069, +0.4883] |
| recurrent | return | optimizer_anchor_interaction | -0.0114 [-0.0877, +0.0789] | -0.0741 [-0.1711, +0.0235] | +0.0570 [-0.1383, +0.2401] |
| recurrent | return | three_way_interaction | +0.1284 [-0.0246, +0.2918] | -0.0737 [-0.2784, +0.1261] | +0.4232 [+0.0570, +0.7975] |

## Every independent seed

Current-branch T, Brier ×100.

| Model | Seed | T |
|---|---:|---:|
| conditional | 15001 | +0.1306 |
| conditional | 15002 | -0.3643 |
| conditional | 15003 | -0.0183 |
| conditional | 15004 | +0.0385 |
| conditional | 15005 | -0.0832 |
| conditional | 15006 | -0.0062 |
| conditional | 15007 | +0.1319 |
| conditional | 15008 | +0.1198 |
| conditional | 15009 | +0.0333 |
| conditional | 15010 | -0.2226 |
| conditional | 15011 | -0.2897 |
| conditional | 15012 | +0.0032 |
| recurrent | 15001 | +0.1183 |
| recurrent | 15002 | +0.0281 |
| recurrent | 15003 | +0.0146 |
| recurrent | 15004 | -0.0279 |
| recurrent | 15005 | -0.0680 |
| recurrent | 15006 | +0.2464 |
| recurrent | 15007 | -0.0804 |
| recurrent | 15008 | +0.0811 |
| recurrent | 15009 | +0.1881 |
| recurrent | 15010 | -0.1638 |
| recurrent | 15011 | +0.2070 |
| recurrent | 15012 | +0.4468 |

All absolute/early/late/survival metrics, exact uniform expectation, zero-update references, environment contrasts, candidate coverage and cue/assessment slices are preserved in summary.json. The two branches begin with the same actual history and then receive their own causal feedback. Branch contrasts therefore include changed outcomes and evolving context. Every hybrid is a declared counterfactual substitution; these results do not establish an on-policy natural component effect, adaptive memory policy, elapsed-time mechanism or general continual-learning improvement. No largest-factor winner or policy gate is selected.
