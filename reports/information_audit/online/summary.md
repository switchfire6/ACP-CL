# Online ideal-observer reanalysis of the representation-learning qualification fits

**POST HOC / EXPLORATORY / NOT DECISION-BEARING.** This is a post hoc reanalysis of saved online arrays. It does not change the locked qualification result (STOP before the candidate comparison), any threshold, or any decision. Ideal observers below are evaluator-side analysis references that use law identities and the simulator; they are not learners and were never trained or fed back.

Script: `scripts/ideal_observer_online.py`. Machine-readable results: `results.json`; tables: `cycle_brier.csv`, `fresh_chunk_brier.csv`, `position_profile.csv`, `cue_contrasts.csv`, `replay_and_seed_table.csv`; figures: `online_brier_by_cycle.png`, `within_block_profile.png`, `delay_cue_contrast.png`.

## Data, verification and references

* Saved online arrays of all 24 qualification phases (6 seeds x fresh_1..3 + interleaved): 12,288 training batches, 393,216 online cases. Each forecast (5 actions x 3 horizons) was made BEFORE that batch's feedback, with the previous batch as the 32-record support. No model, checkpoint (`*.pt`) or study runner was executed.
* Verified: saved observations byte-match AcquisitionWorld.dataset(law, 32, training_seed).
* Verified: actions, performed outcomes, counterfactual truth, affected/valid masks match.
* Verified: packet query/support fingerprints match result.json.
* Verified: pixel-decoded source/lossy/delayed/cues equal evaluator factors/signals; gauge levels equal rint(reserve/18*240).
* Verified: ideal-observer physics with the TRUE latent state reproduces every truth pattern.
* Verified: endpoint panels/supports regenerate and saved endpoint Brier recomputes exactly.
* Structure verified for every seed: stage laws use mode 1-seed%2, so Base B and the three novelty laws share one hidden mode and Base A is the only law with its mode (20% of the interleaved stream); the law that first adds the delay cue differs from the block immediately before it only by the delay dependency (same mode, active sets differ by {1}): 18101: True, 18102: True, 18103: True, 18104: True, 18105: True, 18106: True.
* References (all scored with the same all-case Brier as the model: 5 actions x 3 horizons against evaluator counterfactual truth). **Law-known ideal observer**: decodes reserves (gauge bins), weather direction, lossy/delayed glyphs and the three pulse cues from pixels, knows the batch's law, integrates hidden supply noise and gauge quantisation by Monte Carlo (256 draws/case, common random numbers; Monte Carlo variance removed from Brier, mean correction 0.00017). **Support-posterior observer**: uniform prior over the 5 interleaved laws, conditioned only on the learner's own 32-record support. **No-context observer**: uniform mixture of the 5 laws. **Law marginal**: running performed-action marginal of the batch's own law (the qualification's marginal reference).

## Q1. Per-law online Brier by cycle (interleaved fit)

Seed-mean online all-case Brier per 1024-arrival block (32 batches); each law gets one block per cycle.

| Law | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c1-c8 | law-known ideal | same-support ideal | no-context ideal | law marginal (c2-c8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base A (unique mode) | 0.1101 | 0.1240 | 0.1167 | 0.1192 | 0.1117 | 0.1152 | 0.1147 | 0.1108 | -0.0007 | 0.0442 | 0.0538 | 0.2535 | 0.2276 |
| Base B | 0.1679 | 0.0950 | 0.0972 | 0.1016 | 0.1017 | 0.0995 | 0.1047 | 0.1077 | 0.0602 | 0.0449 | 0.0572 | 0.0853 | 0.2280 |
| Stage 1 | 0.1130 | 0.0860 | 0.0863 | 0.0860 | 0.0833 | 0.0875 | 0.0892 | 0.0868 | 0.0262 | 0.0440 | 0.0460 | 0.0713 | 0.2268 |
| Stage 2 | 0.1031 | 0.0950 | 0.0952 | 0.0945 | 0.0957 | 0.0964 | 0.0971 | 0.0974 | 0.0057 | 0.0441 | 0.0460 | 0.0685 | 0.2253 |
| Stage 3 | 0.1124 | 0.1071 | 0.1044 | 0.1051 | 0.1043 | 0.1056 | 0.1047 | 0.1070 | 0.0054 | 0.0433 | 0.0450 | 0.0761 | 0.2250 |

| Seed | Base A c1 | Base A c8 | best cycle | others c1 | others c8 | c8 gap (A - others) | c8 gap, positions 2-32 | A excess over law-known ideal, c8 | others excess, c8 | A with Base-B-law ideal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18101 | 0.1012 | 0.1104 | 1 | 0.1207 | 0.0959 | 0.0145 | 0.0092 | 0.0671 | 0.0514 | 0.4139 |
| 18102 | 0.1129 | 0.1172 | 2 | 0.1245 | 0.0968 | 0.0204 | 0.0130 | 0.0734 | 0.0528 | 0.4094 |
| 18103 | 0.1134 | 0.1107 | 6 | 0.1266 | 0.0992 | 0.0115 | 0.0078 | 0.0654 | 0.0561 | 0.4115 |
| 18104 | 0.1127 | 0.1072 | 8 | 0.1159 | 0.1021 | 0.0051 | -0.0004 | 0.0639 | 0.0564 | 0.4112 |
| 18105 | 0.1166 | 0.1091 | 5 | 0.1312 | 0.1041 | 0.0049 | -0.0022 | 0.0678 | 0.0591 | 0.4120 |
| 18106 | 0.1040 | 0.1104 | 1 | 0.1258 | 0.1002 | 0.0102 | 0.0085 | 0.0671 | 0.0569 | 0.4111 |

**Answer.** Base A does **not** improve over cycles: its seed-mean online Brier is 0.1101 in cycle 1 and 0.1108 in cycle 8 (per-seed cycle-1 minus cycle-8 change -0.0007 +/- 0.0028; 3 of 6 seeds get worse), and every cycle lies in 0.110-0.124. The other laws improve, but almost entirely between cycle 1 and 2 (Base B 0.168 -> 0.095, stage 1 0.113 -> 0.086) and are flat or drift upward afterwards (Base B 0.095 in cycle 2 -> 0.108 in cycle 8); Base A's cycle-1 block is the model's very first block, which is why it has no early drop. At cycle 8 Base A ends 0.0111 +/- 0.0024 Brier above the mean of the other four laws (seed range 0.0049 to 0.0204; 0.0060 +/- 0.0024 excluding the first, cross-law-support batch). All laws are far from the law-known ideal (about .044 for every law); Base A's excess is 0.0674 +/- 0.0013 versus 0.0554 +/- 0.0011 for the others. Online Base A is nevertheless much better than predicting it with the majority mode (Base-B-law ideal on Base A cases: 0.412) or with its own law marginal (0.228 in cycles 2-8).

The locked endpoint value (Base A 0.1956) is much worse than the block-averaged online value because the endpoint is taken at the worst point of the cycle: after 4,096 arrivals of mode-B laws, with Base A support but before any Base A update. Its online analogue is the start of a Base A visit (cycle-mean position 1 = 0.357 with the previous law's support; position 2 = 0.200 with Base A support), not the block average. The block-end evaluator curve in `result.json` (same-law support after 32 Base A batches) is 0.083 on average.

Interleaving also costs the stage laws relative to their fresh fits at matched exposure (chunk/cycle 8): stage 1 fresh 0.073 vs interleaved 0.087; stage 2 fresh 0.077 vs interleaved 0.097; stage 3 fresh 0.082 vs interleaved 0.107.

Cross-law ideal-observer Brier (rows: true law of the online cases; columns: law used by the ideal forecast). The hidden mode dominates; the cue laws are close to each other.

| True \ used | Base A (unique mode) | Base B | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|---:|---:|
| Base A (unique mode) | 0.044 | 0.409 | 0.396 | 0.383 | 0.371 |
| Base B | 0.408 | 0.045 | 0.076 | 0.104 | 0.129 |
| Stage 1 | 0.398 | 0.075 | 0.044 | 0.074 | 0.102 |
| Stage 2 | 0.386 | 0.103 | 0.074 | 0.044 | 0.074 |
| Stage 3 | 0.370 | 0.128 | 0.101 | 0.073 | 0.043 |

## Q2. Within-block dynamics (switch cost and recognition)

Seed-mean Brier by batch position inside each 32-batch block, cycles 2-8 pooled (position 1 uses the previous law's last batch as support; later positions use same-law support). Plateau = positions 17-32. Half-life = first position whose 3-batch running excess is at most half the position-1 excess.

| Law | pos 1 | pos 2 | pos 3 | pos 4 | pos 8 | pos 16 | plateau | switch cost | share resolved at pos 2 | half-life | same-support ideal pos 1 -> pos 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Base A (unique mode) | 0.357 | 0.200 | 0.164 | 0.141 | 0.117 | 0.109 | 0.093 | 0.264 | 0.59 | 2 | 0.364 -> 0.044 |
| Base B | 0.390 | 0.144 | 0.107 | 0.103 | 0.108 | 0.083 | 0.085 | 0.305 | 0.81 | 1 | 0.412 -> 0.045 |
| Stage 1 | 0.096 | 0.092 | 0.090 | 0.089 | 0.085 | 0.089 | 0.084 | 0.012 | 0.35 | 3 | 0.074 -> 0.044 |
| Stage 2 | 0.109 | 0.103 | 0.104 | 0.100 | 0.100 | 0.097 | 0.094 | 0.015 | 0.43 | 3 | 0.075 -> 0.048 |
| Stage 3 | 0.116 | 0.111 | 0.112 | 0.109 | 0.110 | 0.101 | 0.102 | 0.014 | 0.36 | 3 | 0.075 -> 0.044 |

Mode-A projection coefficient of the forecasts, lambda = <p - p_B, p_A - p_B> / |p_A - p_B|^2, where p_A and p_B are the law-known ideal forecasts under Base A and Base B (0 = forecasts vary like Base B, 1 = like Base A):

| Blocks | pos 1 | pos 2 | pos 3 | pos 4 | pos 8 | pos 17-32 | same-support ideal pos 1 / pos 2+ |
|---|---:|---:|---:|---:|---:|---:|---|
| Base A | 0.17 | 0.56 | 0.65 | 0.72 | 0.81 | 0.88 | 0.17 / 1.00 |
| Base B | 0.92 | 0.28 | 0.15 | 0.14 | 0.14 | 0.07 | 1.00 / 0.00 |

| Law | end of previous visit (pos 29-32) | next visit pos 1 | next visit pos 2 | weight loss between visits (pos 2-4 minus previous end) | context step (pos 1 minus pos 2) |
|---|---:|---:|---:|---:|---:|
| Base A (unique mode) | 0.083 | 0.357 | 0.200 | 0.085 +/- 0.015 | 0.157 +/- 0.021 |
| Base B | 0.088 | 0.390 | 0.144 | 0.030 +/- 0.007 | 0.246 +/- 0.015 |
| Stage 1 | 0.084 | 0.096 | 0.092 | 0.007 +/- 0.005 | 0.004 +/- 0.005 |
| Stage 2 | 0.094 | 0.109 | 0.103 | 0.008 +/- 0.004 | 0.006 +/- 0.004 |
| Stage 3 | 0.104 | 0.116 | 0.111 | 0.006 +/- 0.004 | 0.005 +/- 0.004 |

**Answer.** Base A error **drops sharply after batch 1** (evidence of context use) but only part-way: 0.357 at position 1 -> 0.200 at position 2 (59% of the switch cost of 0.264 resolved), then a slow decline through the rest of the block (position 4 0.141, position 8 0.117, position 16 0.109) to 0.093 in positions 17-32. The mode-A projection moves 0.17 -> 0.56 at the first same-law support and only reaches 0.88 late in the block. The ideal observer given the same single support recognises Base A completely at position 2 (Brier 0.044, lambda 1.00, posterior on Base A 1.00; one 32-record Base A support favours Base A over Base B by 87.0 nats on average), so the residual is not an information limit of the 32-record window. Base B, which follows Base A, pays the largest position-1 cost (0.390; its forecasts are Base-A-like, lambda 0.92) and recovers faster (81% at position 2). Stage laws have small switch costs (0.012, 0.015, 0.014) because their predecessor shares their mode. Recognition delay: the ideal observer needs one batch; the model's half-life is 2 batches for Base A, but the remaining 0.107 excess after position 2 decays only over the whole block, i.e. the rest of mode A is re-learned in the weights on every visit.

Two further observations separate context from weights. (i) Forgetting between visits: with the correct same-law support (positions 2-4), Base A starts each visit 0.085 +/- 0.015 Brier worse than it ended its previous visit (Base B 0.030 +/- 0.007; stage laws below .01), so what the model knows about mode A is held mostly in weights that four mode-B blocks overwrite. (ii) A natural experiment at the locked endpoint: Base A support with NO Base A weight update gives Brier 0.196 and lambda 0.58 (seed values 0.41, 0.54, 0.78, 0.34, 0.86, 0.57), essentially the online position-2 level (0.200, lambda 0.56); so the position 1 -> 2 drop is attributable mostly to the support, not to the 12 updates on the first Base A batch. Context therefore carries roughly half of the mode switch; the rest is in-weight relearning.

## Q3. Arrival-delay dependency online

Estimator: on delayed cases (glyph delayed = 1), D = mean over request sizes 2 and 4 of [A->B forecast] - [B->A forecast]; its coefficient on z = 1 - 2 x (cue-1 signal) is the aligned-minus-misaligned difference (aligned = transfer direction equals the cue-1 signal, which the delay law maps to delay 1 instead of 4). Signals are randomised, so the estimate is unbiased; pixel-decodable covariates only reduce variance; SEs are over the 6 seeds. Units: survival probability, mean of horizons 4/8/12.

| Cell | mean +/- SE (6 seeds) | h4 | h8 | h12 | per-seed values |
|---|---:|---:|---:|---:|---|
| True effect: law-known ideal, delay-first law | 0.0330 +/- 0.0019 | 0.0624 | 0.0214 | 0.0152 | 0.0283, 0.0415, 0.0305, 0.0326, 0.0308, 0.0341 |
| True effect: evaluator truth, delay-first law | 0.0344 +/- 0.0023 | 0.0633 | 0.0222 | 0.0178 | 0.0295, 0.0439, 0.0296, 0.0383, 0.0318, 0.0334 |
| Law-known ideal, later delay-active laws | 0.0342 +/- 0.0019 | 0.0631 | 0.0242 | 0.0151 | 0.0291, 0.0357, 0.0382, 0.0336 |
| Ideal with the same one-batch support, delay-first law | 0.0202 +/- 0.0011 | 0.0377 | 0.0132 | 0.0097 | 0.0189, 0.0224, 0.0189, 0.0244, 0.0168, 0.0199 |
| Ideal with the same one-batch support, predecessor law | 0.0100 +/- 0.0025 | 0.0248 | 0.0028 | 0.0024 | 0.0027, 0.0143, 0.0086, 0.0144, 0.0170, 0.0031 |
| No-context ideal (5-law mixture), delay-first law | 0.0137 +/- 0.0023 | 0.0253 | 0.0090 | 0.0068 | 0.0126, 0.0173, 0.0057, 0.0187, 0.0193, 0.0088 |
| Model, fresh delay fit (all chunks) | 0.0114 +/- 0.0034 | 0.0126 | 0.0129 | 0.0089 | 0.0060, 0.0194, 0.0167, -0.0008, 0.0191, 0.0083 |
| Model, fresh delay fit, chunks 5-8 | 0.0147 +/- 0.0036 | 0.0160 | 0.0164 | 0.0117 | 0.0154, 0.0265, 0.0079, 0.0042, 0.0238, 0.0102 |
| Model, fresh delay fit, chunk 8 | 0.0221 +/- 0.0072 | 0.0236 | 0.0202 | 0.0226 | 0.0320, 0.0490, 0.0188, -0.0035, 0.0217, 0.0147 |
| Model, fresh delay fit, NON-delayed placebo | 0.0079 +/- 0.0028 | 0.0084 | 0.0081 | 0.0072 | -0.0007, 0.0188, 0.0114, 0.0023, 0.0058, 0.0096 |
| Model, interleaved delay-first law | 0.0049 +/- 0.0020 | 0.0061 | 0.0035 | 0.0049 | 0.0001, 0.0039, 0.0045, 0.0106, -0.0005, 0.0106 |
| Model, interleaved delay-first law, cycles 5-8 | 0.0080 +/- 0.0015 | 0.0097 | 0.0066 | 0.0075 | 0.0037, 0.0093, 0.0082, 0.0089, 0.0039, 0.0138 |
| Model, interleaved delay-first law, position 1 | 0.0030 +/- 0.0143 | 0.0135 | -0.0058 | 0.0014 | -0.0070, 0.0372, 0.0490, -0.0379, 0.0070, -0.0301 |
| Model, interleaved delay-first law, NON-delayed placebo | 0.0047 +/- 0.0016 | 0.0055 | 0.0049 | 0.0036 | 0.0090, 0.0003, 0.0028, 0.0103, 0.0025, 0.0032 |
| Model, interleaved later delay-active laws | 0.0086 +/- 0.0034 | 0.0120 | 0.0090 | 0.0047 | 0.0006, 0.0128, 0.0154, 0.0055 |
| Model, interleaved predecessor law (delay inactive) | 0.0018 +/- 0.0020 | 0.0063 | 0.0008 | -0.0018 | -0.0061, 0.0054, 0.0039, 0.0074, 0.0009, -0.0009 |
| Model, interleaved predecessor law, cycles 5-8 | 0.0022 +/- 0.0036 | 0.0059 | 0.0006 | 0.0001 | -0.0073, 0.0110, -0.0003, 0.0086, -0.0084, 0.0095 |
| Model, interleaved Base A (delay inactive) | 0.0013 +/- 0.0022 | 0.0021 | 0.0019 | 0.0000 | 0.0044, 0.0103, -0.0003, -0.0044, 0.0006, -0.0025 |
| Law-known ideal, predecessor law (should be 0) | -0.0034 +/- 0.0018 | -0.0002 | -0.0068 | -0.0032 | -0.0095, 0.0023, 0.0001, -0.0037, -0.0028, -0.0068 |
| Law-known ideal, NON-delayed placebo (should be 0) | -0.0017 +/- 0.0012 | -0.0005 | -0.0026 | -0.0020 | 0.0013, -0.0061, -0.0003, -0.0046, 0.0005, -0.0010 |

Brier value of the delay cue (flip test = Brier with cue-1 pulse order flipped minus correct, on delayed cases; cue-blind = averaging the two forecasts):

| Quantity | value |
|---|---:|
| Ideal flip benefit, online delayed cases, interleaved delay-first law | 0.0162 +/- 0.0010 |
| Ideal flip benefit, online delayed cases, fresh delay fit | 0.0159 +/- 0.0008 |
| Ideal cue-blind value (information value of cue 1) | 0.0042 +/- 0.0003 |
| Ideal flip benefit on the locked endpoint panel, fresh delay fit | 0.0148 +/- 0.0013 |
| Model flip benefit on the same panel, fresh delay fit (locked scorer value .003365) | 0.0034 +/- 0.0013 |
| Model flip benefit on the endpoint panels, interleaved delay-active laws | 0.0005 +/- 0.0008 |
| For scale: ideal flip benefit, efficiency cue / supply-timing cue (interleaved endpoints) | 0.0340 / 0.2145 |

**True size.** Where the delay law acts (delayed cases, transfer requests), aligned transfers survive more often by 0.033 (ideal) / 0.034 (truth) in probability, concentrated at horizon 4 (0.062) and small at 8/12 (0.021, 0.015). In Brier units the best possible flip benefit is about 0.016 (endpoint panels 0.015), roughly half the efficiency cue (0.034) and one-fourteenth of supply timing (0.215). The locked .002 threshold is 14% of the ideal delay flip benefit; the fresh model reached about 23% of it.

**Model.** Online, the fresh delay fit's aligned-minus-misaligned forecast difference grows to 0.0147 in chunks 5-8 (model/ideal ratio over the whole fit 0.31 +/- 0.08). In the interleaved fit it is 0.0049 +/- 0.0020 in the delay-first law (ratio 0.14 +/- 0.06; 0.0080 in cycles 5-8) versus 0.0018 +/- 0.0020 in the delay-inactive predecessor law. The delay-first minus predecessor difference is 0.0031 +/- 0.0021 overall and 0.0058 +/- 0.0023 in cycles 5-8, against a law-known ideal difference of 0.0364 +/- 0.0016 and a same-support ideal difference of 0.0102 +/- 0.0025.

**Specificity.** The model's cue-1 sensitivity is not specific to the cases the law affects: on NON-delayed cases of the same delay-first blocks (where cue 1 is physically irrelevant; ideal -0.0017, truth -0.0008) the model shows 0.0047 +/- 0.0016, so the delayed-minus-non-delayed difference is 0.0002 +/- 0.0023. Even the fresh delay fit has a large non-delayed placebo response (0.0079 +/- 0.0028; delayed minus non-delayed 0.0036 +/- 0.0025). The efficiency cue (which passed its interleaved gate) shows the same pattern: model contrast on lossy cases 0.0112 +/- 0.0022 in its first active law vs 0.0099 +/- 0.0009 in the inactive predecessor (ideal 0.072 vs 0.003).

**Can the law be told apart?** For the ideal observer, one 32-record support distinguishes the delay-first law from its predecessor only weakly: mean log-likelihood ratio 0.87 nats for supports drawn from the delay law (57% favour it) and -0.88 for supports from the predecessor (75% favour it), versus 87 nats for Base A vs Base B. These two laws are the least separable pair in the stream, so a learner whose context is one batch can only represent delay activity weakly through context and must otherwise carry it in weights that alternate between delay-active and delay-inactive blocks.

**Answer.** The delay dependency is real but small (about 0.033 survival probability; best flip benefit about 0.016 Brier). Online, the interleaved model's forecasts carry about 14% of the true aligned-minus-misaligned difference (fresh fit about 31%), and almost all of that response is a generic pulse-1 direction bias that is also present in delay-inactive laws and on non-delayed cases. There is at most a weak sign of law discrimination (delay-first minus predecessor 0.0031 +/- 0.0021, 0.0058 +/- 0.0023 late), of the same order as the non-delayed placebo difference (0.0024 +/- 0.0014); nothing in these arrays shows the model conditioning its delay response on the law.

## Q4. Other observations: seeds, mode parity, cue order, replay

| Seed | Base A mode | cue order | delay stage | endpoint Base A Brier | endpoint lambda | online Base A c5-8 | mean pos-2 lambda | Base A replay share outside Base A blocks | Base A packets in memory at visit starts (c1..c8) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 18101 | 1 | supply_timing/delay/efficiency | 2 | 0.2665 | 0.41 | 0.1197 | 0.41 | 0.211 | 0 1 4 2 3 2 3 3 |
| 18102 | 0 | efficiency/delay/supply_timing | 2 | 0.2145 | 0.54 | 0.1137 | 0.66 | 0.314 | 0 6 4 5 4 4 4 3 |
| 18103 | 1 | efficiency/supply_timing/delay | 3 | 0.1146 | 0.78 | 0.1069 | 0.61 | 0.257 | 0 4 4 4 3 4 3 4 |
| 18104 | 0 | delay/efficiency/supply_timing | 1 | 0.2866 | 0.34 | 0.1138 | 0.55 | 0.217 | 0 4 1 2 3 4 3 3 |
| 18105 | 1 | delay/supply_timing/efficiency | 1 | 0.1008 | 0.86 | 0.1021 | 0.68 | 0.317 | 0 6 5 5 5 4 2 4 |
| 18106 | 0 | supply_timing/efficiency/delay | 3 | 0.1905 | 0.57 | 0.1224 | 0.42 | 0.168 | 0 1 0 0 0 1 2 5 |

* **Pathological seeds.** 18104 and 18101 have the worst endpoint Base A Brier (0.287, 0.267; locked marginal gains were negative). Endpoint Brier is almost entirely explained by how far the forecasts move toward mode A given the Base A support (endpoint lambda vs Brier across seeds r = -0.995). This switch quality is volatile visit to visit within every seed (position-2 lambda ranges, cycles 2-8: 18101 0.22-0.79; 18102 0.19-0.92; 18103 0.33-0.89; 18104 0.32-0.82; 18105 0.39-0.90; 18106 0.23-0.86), so the single endpoint draw exaggerates seed differences; online Base A cycles 5-8 Brier spans only 0.102-0.122.
* **Replay composition.** Across the 42 Base A visits of cycles 2-8, the number of Base A packets in the 16-packet reservoir at visit start predicts the context switch: 0-2 packets (n=12) give position-2 Brier 0.289 and lambda 0.35; 3 or more (n=30) give 0.165 and 0.64 (pooled r with position-2 Brier -0.57, within-seed -0.37; with lambda 0.48 / 0.31). Seed level (n=6, descriptive): Base A share of replay draws outside Base A blocks (0.17-0.32; 0.20 expected) correlates r = -0.78 with online Base A Brier (cycles 5-8) and r = -0.48 with the endpoint. Mode-switch packets (block starts whose support comes from the other mode) show no visit-level relation (within-seed r -0.15 with position-2 Brier), although the seed-level share of all cross-law-support packets correlates r = 0.95 with online Base A (n=6, driven by long-lived reservoir entries; not interpretable).
* **Mode parity.** Base A mode 0 (seeds [18102, 18104, 18106]) endpoint 0.231 +/- 0.029, online c5-8 0.117 +/- 0.003; mode 1 (seeds [18101, 18103, 18105]) endpoint 0.161 +/- 0.053, online 0.110 +/- 0.005. Three seeds per parity with overlapping ranges: no evidence for a parity effect.
* **Cue order.** Each order occurs in exactly one seed, and the stage-3 law that immediately precedes every Base A block has all three cues regardless of order, so order can only act through the stage-1/2 laws; nothing in the per-seed table tracks order.
* **Stage laws are mutually confusable, even ideally.** With the same one-batch support the ideal observer's posterior on the true stage law is only 0.55, 0.56, 0.73 (mode posterior 1.00), yet its Brier stays near .045 because confusable laws forecast alike. The no-context mixture costs mode-B laws only .07-.09 but Base A 0.254: context is indispensable only for the unique-mode law.

## Caveats

* POST HOC and EXPLORATORY; six seeds; correlations are descriptive; nothing here re-opens or relaxes the locked qualification, which stands as STOP.
* Ideal observers use evaluator-only knowledge (law set, simulator, noise model). They bound what the observations and the 32-record support contain; they are not candidate learners.
* No model was executed, so support and weight effects at position 2 are separated only indirectly (endpoint natural experiment; forgetting between visits). The position-2 forecast follows 12 optimizer steps on the first batch of the block.
* Monte Carlo error of the ideal observers is small (256 common-random-number draws per case; variance-corrected Brier) relative to every reported gap. Two independent full collections gave a byte-identical summary; one Monte Carlo chunk (seed 18102) differed at the level of a few draws, changing stored values by less than 1e-5 Brier.

