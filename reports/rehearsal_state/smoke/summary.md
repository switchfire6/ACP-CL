# Rehearsal state: controlled weight, Adam and anchor substitutions

Cell bits are weights, complete Adam state and the complete causal anchor packet (0 old, 1 new). All cells use the same retained candidates and selected IDs. D is selected-candidate loss minus the exact mean candidate loss. The primary response is current-branch T=D111−D000, with two assessments averaged per seed.

The replication screen requires mean T≥0.0005 and positive T in at least 9/12 seeds. It is a research allocation screen, not a significance test or policy promotion. All intervals are descriptive seed-bootstrap 95% intervals using 20,000 resamples. Zero-update references are outside the candidate pool.

| Model | T Brier ×100 [95% interval] | Positive seeds | Replication screen |
|---|---:|---:|---|
| conditional | -0.0002 [-0.0002, -0.0002] | 0/1 | FAIL |
| recurrent | +0.0004 [+0.0004, +0.0004] | 1/1 | FAIL |

Engineering/subset cohort: ineligible for the scientific replication screen.

- conditional: failed criteria = complete_scientific_cohort, deterioration, seed_consistency.
- recurrent: failed criteria = complete_scientific_cohort, deterioration, seed_consistency.

## Every cell

Brier ×100 and survival %. Oracle is conditional on validation Brier, not deployable and not independently optimized for survival.

| Model | Branch | Cell | Chosen Brier | Original-error Brier | Uniform Brier | Replacement Brier | Oracle Brier | D [interval] | Chosen survival | Uniform survival |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conditional | current | 000 | 24.2854 | 24.2823 | 24.2849 | 24.2860 | 24.2823 | +0.0004 [+0.0004, +0.0004] | 56.250 | 56.250 |
| conditional | current | 001 | 24.2803 | 24.2774 | 24.2804 | 24.2810 | 24.2774 | -0.0001 [-0.0001, -0.0001] | 56.250 | 56.250 |
| conditional | current | 010 | 24.2755 | 24.2728 | 24.2751 | 24.2760 | 24.2728 | +0.0004 [+0.0004, +0.0004] | 56.250 | 56.250 |
| conditional | current | 011 | 24.2708 | 24.2682 | 24.2709 | 24.2715 | 24.2682 | -0.0000 [-0.0000, -0.0000] | 56.250 | 56.250 |
| conditional | current | 100 | 24.2947 | 24.2915 | 24.2939 | 24.2950 | 24.2915 | +0.0008 [+0.0008, +0.0008] | 54.688 | 54.688 |
| conditional | current | 101 | 24.2905 | 24.2876 | 24.2903 | 24.2912 | 24.2876 | +0.0002 [+0.0002, +0.0002] | 54.688 | 54.688 |
| conditional | current | 110 | 24.2836 | 24.2807 | 24.2829 | 24.2842 | 24.2807 | +0.0007 [+0.0007, +0.0007] | 54.688 | 54.688 |
| conditional | current | 111 | 24.2806 | 24.2778 | 24.2804 | 24.2814 | 24.2778 | +0.0002 [+0.0002, +0.0002] | 54.688 | 54.688 |
| conditional | return | 000 | 22.3544 | 22.3496 | 22.3560 | 22.3506 | 22.3496 | -0.0017 [-0.0017, -0.0017] | 62.500 | 62.500 |
| conditional | return | 001 | 22.3454 | 22.3405 | 22.3473 | 22.3422 | 22.3405 | -0.0018 [-0.0018, -0.0018] | 62.500 | 62.500 |
| conditional | return | 010 | 22.3625 | 22.3581 | 22.3642 | 22.3583 | 22.3581 | -0.0017 [-0.0017, -0.0017] | 60.938 | 60.938 |
| conditional | return | 011 | 22.3535 | 22.3490 | 22.3552 | 22.3500 | 22.3490 | -0.0017 [-0.0017, -0.0017] | 59.375 | 60.156 |
| conditional | return | 100 | 22.2810 | 22.2762 | 22.2826 | 22.2768 | 22.2762 | -0.0016 [-0.0016, -0.0016] | 56.250 | 56.250 |
| conditional | return | 101 | 22.2723 | 22.2673 | 22.2740 | 22.2685 | 22.2673 | -0.0017 [-0.0017, -0.0017] | 56.250 | 56.250 |
| conditional | return | 110 | 22.2888 | 22.2845 | 22.2907 | 22.2848 | 22.2845 | -0.0019 [-0.0019, -0.0019] | 56.250 | 56.250 |
| conditional | return | 111 | 22.2794 | 22.2748 | 22.2813 | 22.2756 | 22.2748 | -0.0019 [-0.0019, -0.0019] | 56.250 | 56.250 |
| recurrent | current | 000 | 24.6917 | 24.6764 | 24.6841 | 24.6658 | 24.6764 | +0.0076 [+0.0076, +0.0076] | 51.562 | 51.562 |
| recurrent | current | 001 | 24.6875 | 24.6715 | 24.6795 | 24.6601 | 24.6715 | +0.0080 [+0.0080, +0.0080] | 51.562 | 51.562 |
| recurrent | current | 010 | 24.6717 | 24.6564 | 24.6641 | 24.6461 | 24.6564 | +0.0076 [+0.0076, +0.0076] | 51.562 | 51.562 |
| recurrent | current | 011 | 24.6675 | 24.6514 | 24.6594 | 24.6408 | 24.6514 | +0.0081 [+0.0081, +0.0081] | 51.562 | 51.562 |
| recurrent | current | 100 | 24.7268 | 24.7106 | 24.7187 | 24.7021 | 24.7106 | +0.0081 [+0.0081, +0.0081] | 53.125 | 53.125 |
| recurrent | current | 101 | 24.7225 | 24.7058 | 24.7141 | 24.6963 | 24.7058 | +0.0084 [+0.0084, +0.0084] | 53.125 | 53.906 |
| recurrent | current | 110 | 24.7058 | 24.6903 | 24.6981 | 24.6814 | 24.6903 | +0.0078 [+0.0078, +0.0078] | 53.125 | 53.125 |
| recurrent | current | 111 | 24.7016 | 24.6855 | 24.6935 | 24.6761 | 24.6855 | +0.0080 [+0.0080, +0.0080] | 53.125 | 53.125 |
| recurrent | return | 000 | 22.8275 | 22.8047 | 22.8161 | 22.8186 | 22.8047 | +0.0114 [+0.0114, +0.0114] | 54.688 | 54.688 |
| recurrent | return | 001 | 22.8137 | 22.7899 | 22.8018 | 22.8041 | 22.7899 | +0.0119 [+0.0119, +0.0119] | 54.688 | 54.688 |
| recurrent | return | 010 | 22.8096 | 22.7870 | 22.7983 | 22.8010 | 22.7870 | +0.0113 [+0.0113, +0.0113] | 54.688 | 54.688 |
| recurrent | return | 011 | 22.7964 | 22.7722 | 22.7843 | 22.7871 | 22.7722 | +0.0121 [+0.0121, +0.0121] | 54.688 | 54.688 |
| recurrent | return | 100 | 22.8922 | 22.8682 | 22.8802 | 22.8816 | 22.8682 | +0.0120 [+0.0120, +0.0120] | 51.562 | 51.562 |
| recurrent | return | 101 | 22.8783 | 22.8529 | 22.8656 | 22.8668 | 22.8529 | +0.0127 [+0.0127, +0.0127] | 51.562 | 51.562 |
| recurrent | return | 110 | 22.8720 | 22.8490 | 22.8605 | 22.8618 | 22.8490 | +0.0115 [+0.0115, +0.0115] | 51.562 | 51.562 |
| recurrent | return | 111 | 22.8589 | 22.8351 | 22.8470 | 22.8484 | 22.8351 | +0.0119 [+0.0119, +0.0119] | 51.562 | 51.562 |

## Factorial effects and prespecified repairs

Positive D effects mean worse selected-versus-uniform usefulness. Positive chosen-loss repair effects mean the old-component repair reduces chosen loss; positive survival effects instead favor cell 111. Main effects average over the other two factors. Interactions prevent unique component blame.

| Model | Branch | Effect | D Brier ×100 [interval] | Chosen Brier change ×100 [interval] | Chosen survival change, pp [interval] |
|---|---|---|---:|---:|---:|
| conditional | current | total_111_minus_000 | -0.0002 [-0.0002, -0.0002] | -0.0048 [-0.0048, -0.0048] | -1.5625 [-1.5625, -1.5625] |
| conditional | current | weights_main | +0.0003 [+0.0003, +0.0003] | +0.0094 [+0.0094, +0.0094] | -1.5625 [-1.5625, -1.5625] |
| conditional | current | weights_at_optimizer0_anchor0 | +0.0004 [+0.0004, +0.0004] | +0.0093 [+0.0093, +0.0093] | -1.5625 [-1.5625, -1.5625] |
| conditional | current | weights_at_optimizer0_anchor1 | +0.0003 [+0.0003, +0.0003] | +0.0103 [+0.0103, +0.0103] | -1.5625 [-1.5625, -1.5625] |
| conditional | current | weights_at_optimizer1_anchor0 | +0.0003 [+0.0003, +0.0003] | +0.0081 [+0.0081, +0.0081] | -1.5625 [-1.5625, -1.5625] |
| conditional | current | weights_at_optimizer1_anchor1 | +0.0002 [+0.0002, +0.0002] | +0.0097 [+0.0097, +0.0097] | -1.5625 [-1.5625, -1.5625] |
| conditional | current | repair_weights | +0.0002 [+0.0002, +0.0002] | +0.0097 [+0.0097, +0.0097] | -1.5625 [-1.5625, -1.5625] |
| conditional | current | optimizer_main | -0.0000 [-0.0000, -0.0000] | -0.0101 [-0.0101, -0.0101] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | optimizer_at_weights0_anchor0 | +0.0000 [+0.0000, +0.0000] | -0.0099 [-0.0099, -0.0099] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | optimizer_at_weights0_anchor1 | +0.0001 [+0.0001, +0.0001] | -0.0094 [-0.0094, -0.0094] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | optimizer_at_weights1_anchor0 | -0.0001 [-0.0001, -0.0001] | -0.0111 [-0.0111, -0.0111] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | optimizer_at_weights1_anchor1 | -0.0000 [-0.0000, -0.0000] | -0.0100 [-0.0100, -0.0100] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | repair_optimizer | -0.0000 [-0.0000, -0.0000] | -0.0100 [-0.0100, -0.0100] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | anchor_main | -0.0005 [-0.0005, -0.0005] | -0.0042 [-0.0042, -0.0042] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | anchor_at_weights0_optimizer0 | -0.0005 [-0.0005, -0.0005] | -0.0051 [-0.0051, -0.0051] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | anchor_at_weights0_optimizer1 | -0.0005 [-0.0005, -0.0005] | -0.0047 [-0.0047, -0.0047] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | anchor_at_weights1_optimizer0 | -0.0006 [-0.0006, -0.0006] | -0.0042 [-0.0042, -0.0042] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | anchor_at_weights1_optimizer1 | -0.0005 [-0.0005, -0.0005] | -0.0031 [-0.0031, -0.0031] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | repair_anchor | -0.0005 [-0.0005, -0.0005] | -0.0031 [-0.0031, -0.0031] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | weights_optimizer_interaction | -0.0001 [-0.0001, -0.0001] | -0.0009 [-0.0009, -0.0009] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | weights_anchor_interaction | -0.0001 [-0.0001, -0.0001] | +0.0013 [+0.0013, +0.0013] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | optimizer_anchor_interaction | +0.0001 [+0.0001, +0.0001] | +0.0008 [+0.0008, +0.0008] | +0.0000 [+0.0000, +0.0000] |
| conditional | current | three_way_interaction | -0.0000 [-0.0000, -0.0000] | +0.0007 [+0.0007, +0.0007] | +0.0000 [+0.0000, +0.0000] |
| conditional | return | total_111_minus_000 | -0.0002 [-0.0002, -0.0002] | -0.0750 [-0.0750, -0.0750] | -6.2500 [-6.2500, -6.2500] |
| conditional | return | weights_main | -0.0000 [-0.0000, -0.0000] | -0.0736 [-0.0736, -0.0736] | -5.0781 [-5.0781, -5.0781] |
| conditional | return | weights_at_optimizer0_anchor0 | +0.0000 [+0.0000, +0.0000] | -0.0734 [-0.0734, -0.0734] | -6.2500 [-6.2500, -6.2500] |
| conditional | return | weights_at_optimizer0_anchor1 | +0.0001 [+0.0001, +0.0001] | -0.0731 [-0.0731, -0.0731] | -6.2500 [-6.2500, -6.2500] |
| conditional | return | weights_at_optimizer1_anchor0 | -0.0002 [-0.0002, -0.0002] | -0.0737 [-0.0737, -0.0737] | -4.6875 [-4.6875, -4.6875] |
| conditional | return | weights_at_optimizer1_anchor1 | -0.0002 [-0.0002, -0.0002] | -0.0741 [-0.0741, -0.0741] | -3.1250 [-3.1250, -3.1250] |
| conditional | return | repair_weights | -0.0002 [-0.0002, -0.0002] | -0.0741 [-0.0741, -0.0741] | -3.1250 [-3.1250, -3.1250] |
| conditional | return | optimizer_main | -0.0001 [-0.0001, -0.0001] | +0.0078 [+0.0078, +0.0078] | -1.1719 [-1.1719, -1.1719] |
| conditional | return | optimizer_at_weights0_anchor0 | -0.0001 [-0.0001, -0.0001] | +0.0081 [+0.0081, +0.0081] | -1.5625 [-1.5625, -1.5625] |
| conditional | return | optimizer_at_weights0_anchor1 | +0.0001 [+0.0001, +0.0001] | +0.0081 [+0.0081, +0.0081] | -3.1250 [-3.1250, -3.1250] |
| conditional | return | optimizer_at_weights1_anchor0 | -0.0003 [-0.0003, -0.0003] | +0.0078 [+0.0078, +0.0078] | +0.0000 [+0.0000, +0.0000] |
| conditional | return | optimizer_at_weights1_anchor1 | -0.0001 [-0.0001, -0.0001] | +0.0071 [+0.0071, +0.0071] | +0.0000 [+0.0000, +0.0000] |
| conditional | return | repair_optimizer | -0.0001 [-0.0001, -0.0001] | +0.0071 [+0.0071, +0.0071] | +0.0000 [+0.0000, +0.0000] |
| conditional | return | anchor_main | -0.0001 [-0.0001, -0.0001] | -0.0090 [-0.0090, -0.0090] | -0.3906 [-0.3906, -0.3906] |
| conditional | return | anchor_at_weights0_optimizer0 | -0.0002 [-0.0002, -0.0002] | -0.0090 [-0.0090, -0.0090] | +0.0000 [+0.0000, +0.0000] |
| conditional | return | anchor_at_weights0_optimizer1 | +0.0000 [+0.0000, +0.0000] | -0.0089 [-0.0089, -0.0089] | -1.5625 [-1.5625, -1.5625] |
| conditional | return | anchor_at_weights1_optimizer0 | -0.0001 [-0.0001, -0.0001] | -0.0087 [-0.0087, -0.0087] | +0.0000 [+0.0000, +0.0000] |
| conditional | return | anchor_at_weights1_optimizer1 | +0.0000 [+0.0000, +0.0000] | -0.0094 [-0.0094, -0.0094] | +0.0000 [+0.0000, +0.0000] |
| conditional | return | repair_anchor | +0.0000 [+0.0000, +0.0000] | -0.0094 [-0.0094, -0.0094] | +0.0000 [+0.0000, +0.0000] |
| conditional | return | weights_optimizer_interaction | -0.0002 [-0.0002, -0.0002] | -0.0006 [-0.0006, -0.0006] | +2.3438 [+2.3438, +2.3438] |
| conditional | return | weights_anchor_interaction | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0001, -0.0001] | +0.7812 [+0.7812, +0.7812] |
| conditional | return | optimizer_anchor_interaction | +0.0002 [+0.0002, +0.0002] | -0.0003 [-0.0003, -0.0003] | -0.7812 [-0.7812, -0.7812] |
| conditional | return | three_way_interaction | -0.0001 [-0.0001, -0.0001] | -0.0007 [-0.0007, -0.0007] | +1.5625 [+1.5625, +1.5625] |
| recurrent | current | total_111_minus_000 | +0.0004 [+0.0004, +0.0004] | +0.0099 [+0.0099, +0.0099] | +1.5625 [+1.5625, +1.5625] |
| recurrent | current | weights_main | +0.0002 [+0.0002, +0.0002] | +0.0346 [+0.0346, +0.0346] | +1.5625 [+1.5625, +1.5625] |
| recurrent | current | weights_at_optimizer0_anchor0 | +0.0005 [+0.0005, +0.0005] | +0.0351 [+0.0351, +0.0351] | +1.5625 [+1.5625, +1.5625] |
| recurrent | current | weights_at_optimizer0_anchor1 | +0.0004 [+0.0004, +0.0004] | +0.0350 [+0.0350, +0.0350] | +1.5625 [+1.5625, +1.5625] |
| recurrent | current | weights_at_optimizer1_anchor0 | +0.0002 [+0.0002, +0.0002] | +0.0342 [+0.0342, +0.0342] | +1.5625 [+1.5625, +1.5625] |
| recurrent | current | weights_at_optimizer1_anchor1 | -0.0001 [-0.0001, -0.0001] | +0.0340 [+0.0340, +0.0340] | +1.5625 [+1.5625, +1.5625] |
| recurrent | current | repair_weights | -0.0001 [-0.0001, -0.0001] | +0.0340 [+0.0340, +0.0340] | +1.5625 [+1.5625, +1.5625] |
| recurrent | current | optimizer_main | -0.0002 [-0.0002, -0.0002] | -0.0205 [-0.0205, -0.0205] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | optimizer_at_weights0_anchor0 | -0.0000 [-0.0000, -0.0000] | -0.0200 [-0.0200, -0.0200] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | optimizer_at_weights0_anchor1 | +0.0001 [+0.0001, +0.0001] | -0.0200 [-0.0200, -0.0200] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | optimizer_at_weights1_anchor0 | -0.0004 [-0.0004, -0.0004] | -0.0210 [-0.0210, -0.0210] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | optimizer_at_weights1_anchor1 | -0.0003 [-0.0003, -0.0003] | -0.0209 [-0.0209, -0.0209] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | repair_optimizer | -0.0003 [-0.0003, -0.0003] | -0.0209 [-0.0209, -0.0209] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | anchor_main | +0.0003 [+0.0003, +0.0003] | -0.0042 [-0.0042, -0.0042] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | anchor_at_weights0_optimizer0 | +0.0004 [+0.0004, +0.0004] | -0.0042 [-0.0042, -0.0042] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | anchor_at_weights0_optimizer1 | +0.0005 [+0.0005, +0.0005] | -0.0042 [-0.0042, -0.0042] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | anchor_at_weights1_optimizer0 | +0.0002 [+0.0002, +0.0002] | -0.0043 [-0.0043, -0.0043] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | anchor_at_weights1_optimizer1 | +0.0003 [+0.0003, +0.0003] | -0.0043 [-0.0043, -0.0043] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | repair_anchor | +0.0003 [+0.0003, +0.0003] | -0.0043 [-0.0043, -0.0043] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | weights_optimizer_interaction | -0.0004 [-0.0004, -0.0004] | -0.0010 [-0.0010, -0.0010] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | weights_anchor_interaction | -0.0002 [-0.0002, -0.0002] | -0.0001 [-0.0001, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | optimizer_anchor_interaction | +0.0001 [+0.0001, +0.0001] | +0.0001 [+0.0001, +0.0001] | +0.0000 [+0.0000, +0.0000] |
| recurrent | current | three_way_interaction | -0.0001 [-0.0001, -0.0001] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | total_111_minus_000 | +0.0005 [+0.0005, +0.0005] | +0.0314 [+0.0314, +0.0314] | -3.1250 [-3.1250, -3.1250] |
| recurrent | return | weights_main | +0.0003 [+0.0003, +0.0003] | +0.0635 [+0.0635, +0.0635] | -3.1250 [-3.1250, -3.1250] |
| recurrent | return | weights_at_optimizer0_anchor0 | +0.0006 [+0.0006, +0.0006] | +0.0647 [+0.0647, +0.0647] | -3.1250 [-3.1250, -3.1250] |
| recurrent | return | weights_at_optimizer0_anchor1 | +0.0008 [+0.0008, +0.0008] | +0.0646 [+0.0646, +0.0646] | -3.1250 [-3.1250, -3.1250] |
| recurrent | return | weights_at_optimizer1_anchor0 | +0.0002 [+0.0002, +0.0002] | +0.0624 [+0.0624, +0.0624] | -3.1250 [-3.1250, -3.1250] |
| recurrent | return | weights_at_optimizer1_anchor1 | -0.0002 [-0.0002, -0.0002] | +0.0625 [+0.0625, +0.0625] | -3.1250 [-3.1250, -3.1250] |
| recurrent | return | repair_weights | -0.0002 [-0.0002, -0.0002] | +0.0625 [+0.0625, +0.0625] | -3.1250 [-3.1250, -3.1250] |
| recurrent | return | optimizer_main | -0.0003 [-0.0003, -0.0003] | -0.0187 [-0.0187, -0.0187] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | optimizer_at_weights0_anchor0 | -0.0001 [-0.0001, -0.0001] | -0.0179 [-0.0179, -0.0179] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | optimizer_at_weights0_anchor1 | +0.0002 [+0.0002, +0.0002] | -0.0173 [-0.0173, -0.0173] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | optimizer_at_weights1_anchor0 | -0.0005 [-0.0005, -0.0005] | -0.0202 [-0.0202, -0.0202] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | optimizer_at_weights1_anchor1 | -0.0008 [-0.0008, -0.0008] | -0.0194 [-0.0194, -0.0194] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | repair_optimizer | -0.0008 [-0.0008, -0.0008] | -0.0194 [-0.0194, -0.0194] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | anchor_main | +0.0006 [+0.0006, +0.0006] | -0.0135 [-0.0135, -0.0135] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | anchor_at_weights0_optimizer0 | +0.0005 [+0.0005, +0.0005] | -0.0138 [-0.0138, -0.0138] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | anchor_at_weights0_optimizer1 | +0.0007 [+0.0007, +0.0007] | -0.0133 [-0.0133, -0.0133] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | anchor_at_weights1_optimizer0 | +0.0007 [+0.0007, +0.0007] | -0.0139 [-0.0139, -0.0139] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | anchor_at_weights1_optimizer1 | +0.0004 [+0.0004, +0.0004] | -0.0132 [-0.0132, -0.0132] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | repair_anchor | +0.0004 [+0.0004, +0.0004] | -0.0132 [-0.0132, -0.0132] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | weights_optimizer_interaction | -0.0007 [-0.0007, -0.0007] | -0.0022 [-0.0022, -0.0022] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | weights_anchor_interaction | -0.0001 [-0.0001, -0.0001] | -0.0000 [-0.0000, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | optimizer_anchor_interaction | -0.0000 [-0.0000, -0.0000] | +0.0006 [+0.0006, +0.0006] | +0.0000 [+0.0000, +0.0000] |
| recurrent | return | three_way_interaction | -0.0006 [-0.0006, -0.0006] | +0.0003 [+0.0003, +0.0003] | +0.0000 [+0.0000, +0.0000] |

## Every independent seed

Current-branch T, Brier ×100.

| Model | Seed | T |
|---|---:|---:|
| conditional | 15991 | -0.0002 |
| recurrent | 15991 | +0.0004 |

All absolute/early/late/survival metrics, exact uniform expectation, zero-update references, environment contrasts, candidate coverage and cue/assessment slices are preserved in summary.json. The two branches begin with the same actual history and then receive their own causal feedback. Branch contrasts therefore include changed outcomes and evolving context. Every hybrid is a declared counterfactual substitution; these results do not establish an on-policy natural component effect, adaptive memory policy, elapsed-time mechanism or general continual-learning improvement. No largest-factor winner or policy gate is selected.
