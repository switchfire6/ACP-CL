# Bayes ideal-observer audit of the interleaved representation qualification

**POST HOC / EXPLORATORY / NOT DECISION-BEARING.** Prepared 2026-09-28, after the
qualification was locked. It fits no learner, runs no checkpoint or study runner,
and changes no locked file.

- It does **not** change the locked **STOP**
  (`reports/representation_learning/qualification/summary.json`).
- It does **not** authorise running `configs/representation_learning_main.json`,
  or relaxing, re-weighting or excluding anything in it.

O, I and P are analysis references built from evaluator-only knowledge: the
simulator, the law set and the supply-noise law. None of them is a learner.
Numbers come from `results.json` in this folder unless a verification file is
cited.

## Headline

The ideal observer I sees exactly what the learner sees at each endpoint probe:
the 4 query frames and the saved 32-record support. With only that, **I passes
all 31 locked cells.** The four failed cells are therefore not information limits
of the probes.

**Base-A endpoint Brier** (rule ≤ .12):

- I scores .0447, identical to the oracle O. Every Base-A support puts posterior
  1.000 on Base A.
- The learner scores .1956.
- The context-free mixture P scores .2552, so it also fails.

**The three interleaved delay (cue 1) cells** (rule ≥ .002):

| Predictor | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|
| I | .0102 | .0128 | .0099 |
| P (context-free, law-aware) | .0084 | .0071 | .0046 |
| Learner | -.0002 | .0008 | -.0003 |

I's smallest per-seed value is .0051. Even P passes all three.

Separating the delay law from its neighbour is the hardest identification in the
stream: it takes about 110 records to reach 20:1 expected odds. Yet 32 records
are enough to pass the cells, because cue benefit does not require identifying
the law.

For choosing actions, the delay cue and directional efficiency (cue 0) both
matter little. Only the hidden mode and supply timing (cue 2) change action
survival materially (§7).

## 1. What each predictor knows

| Predictor | Query features | Support | Law knowledge |
|---|---|---|---|
| Learner (locked) | 4 frames (pixels) | saved 32 records | none (learned from 40,960 single-action records) |
| **O** oracle | exact case features | not used | the **true** law and the simulator; marginalises only the hidden per-step supply noise |
| **I** ideal observer | exact case features | saved 32 records (performed action's 3-horizon pattern) | uniform prior over the 5 laws of `reference_laws(seed)`, plus the simulator |
| **P** prior-only | exact case features | not used | uniform mixture of the same 5 laws |

Cue flips change only the query signal bit. I's posterior stays fixed.

Every predictor is scored by `score_cells` in `scripts/ideal_observer_load.py`,
which mirrors the locked rules exactly (`../scoring_spec.md`).

For the fresh cells, I and P reuse the interleaved 5-law prior. That is not the
fresh learner's situation, so O is the relevant reference there.

## 2. Monte Carlo and sensitivity settings

**Dynamics.** Dynamics come from the project's own `AcquisitionWorld.simulate`.
Only the supply draw is replicated (`scripts/ideal_observer_mc.py`).

**Sampling.**

- K = 16,384 per case (2 replicates × 16 batches × 512).
- MC root seed 2026092801, disjoint from the project's seeds.
- About 1.57e9 case simulations, in 26.4 minutes on 6 CPU workers.

**Common random numbers.** One noise tensor per (seed, replicate, batch) is
shared across laws, cue-flip variants and actions. Query and support noise are
independent draws.

**MC error.**

- Largest delete-one-batch jackknife SE of any cell: 7.44e-5.
- Replicate difference z over 45 cue-benefit means: RMS .87, maximum 2.34.

**Likelihood smoothing** (λ = 1e-4 primary). The record likelihood is
(1-λ)p + λ/4 over the 4 survival patterns.

- No observed record has zero MC probability under its true law (0/1,920).
- Over λ = 1e-6 to 1e-2:
  - I's cell means move at most 7.4e-5 from the primary (7.5e-5 between any
    two λ).
  - Posterior weights move at most .021, reached at λ = 1e-2; at most .0002
    for λ ≤ 1e-4.
  - No pass/fail flag changes.

**Reserves.** The primary run uses the **exact** evaluator reserves. This is
privileged: the frames show only an 18/240 gauge level. A gauge-consistent rerun
at K = 8,192 checks the effect, drawing reserves uniformly within the gauge bin
(seeds 18101, 18104, 18105):

- per-seed cell values move at most 2.25e-4;
- posterior mass on the true law moves at most .014.

## 3. The four failed cells

| Cell (rule) | Learner | O | I | P | I per-seed min |
|---|---:|---:|---:|---:|---:|
| Base-A all-case Brier (≤ .12) | **.1956** | .0447 | .0447 | .2552 | — (per-seed max .0492) |
| Stage-1 delay cue benefit (≥ .002) | **-.0002** | .0168 | .0102 | .0084 | .0077 |
| Stage-2 delay cue benefit (≥ .002) | **.0008** | .0161 | .0128 | .0071 | .0115 |
| Stage-3 delay cue benefit (≥ .002) | **-.0003** | .0128 | .0099 | .0046 | .0051 |

**Base A per seed:**

- learner .2665, .2145, .1146, .2866, .1008, .1905;
- O = I from .0430 to .0492;
- P from .2405 to .2718.

**Share of O's interleaved delay benefit retained:**

| Predictor | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|
| I | 60% | 80% | 78% |
| Learner | -0.9% | 5.0% | -2.3% |

In the fresh fit the learner retained 22.6% of O's benefit (.0034 of .0149).

**Learner excess Brier over I, by slot:**

| Base A | Base B | Stage 1 | Stage 2 | Stage 3 |
|---:|---:|---:|---:|---:|
| .151 | .067 | .054 | .052 | .054 |

O's Brier floor is .043-.045 in every slot.

## 4. All 31 cells (seed means; FAIL = misses the locked rule)

| # | Cell | Rule | Learner | O | I | P |
|---:|---|---|---:|---:|---:|---:|
| 1 | `fresh/stage_1/marginal_gain` | ≥ .02 | .1558 | .1841 | .1828 | .1567 |
| 2 | `fresh/stage_1/cue_benefit` | ≥ .002 | .0748 | .0895 | .0816 | .0502 |
| 3 | `fresh/stage_2/marginal_gain` | ≥ .02 | .1526 | .1842 | .1831 | .1598 |
| 4 | `fresh/stage_2/cue_benefit` | ≥ .002 | .0737 | .0886 | .0817 | .0343 |
| 5 | `fresh/stage_3/marginal_gain` | ≥ .02 | .1492 | .1829 | .1819 | .1501 |
| 6 | `fresh/stage_3/cue_benefit` | ≥ .002 | .0633 | .0867 | .0781 | .0174 |
| 7 | `fresh/cue_0/marginal_gain` | ≥ .02 | .1528 | .1843 | .1826 | .1576 |
| 8 | `fresh/cue_0/cue_benefit` | ≥ .002 | .0207 | .0350 | .0189 | .0116 |
| 9 | `fresh/cue_1/marginal_gain` | ≥ .02 | .1508 | .1842 | .1835 | .1587 |
| 10 | `fresh/cue_1/cue_benefit` | ≥ .002 | .0034 | .0149 | .0095 | .0055 |
| 11 | `fresh/cue_2/marginal_gain` | ≥ .02 | .1541 | .1827 | .1817 | .1503 |
| 12 | `fresh/cue_2/cue_benefit` | ≥ .002 | .1878 | .2150 | .2130 | .0848 |
| 13 | `interleaved/slot_0/marginal_gain` | ≥ .02 | .0325 | .1835 | .1835 | **-.0270 FAIL** |
| 14 | `interleaved/slot_0/absolute_brier` | ≤ .12 | **.1956 FAIL** | .0447 | .0447 | **.2552 FAIL** |
| 15 | `interleaved/slot_1/marginal_gain` | ≥ .02 | .1170 | .1844 | .1836 | .1454 |
| 16 | `interleaved/slot_1/absolute_brier` | ≤ .12 | .1121 | .0447 | .0455 | .0837 |
| 17 | `interleaved/slot_2/marginal_gain` | ≥ .02 | .1286 | .1839 | .1826 | .1565 |
| 18 | `interleaved/slot_2/absolute_brier` | ≤ .12 | .0987 | .0434 | .0447 | .0708 |
| 19 | `interleaved/stage_1/cue_0/cue_benefit` | ≥ .002 | .0096 | .0371 | .0234 | .0161 |
| 20 | `interleaved/stage_1/cue_1/cue_benefit` | ≥ .002 | **-.0002 FAIL** | .0168 | .0102 | .0084 |
| 21 | `interleaved/stage_1/cue_2/cue_benefit` | ≥ .002 | .1605 | .2148 | .2112 | .1261 |
| 22 | `interleaved/slot_3/marginal_gain` | ≥ .02 | .1313 | .1842 | .1831 | .1598 |
| 23 | `interleaved/slot_3/absolute_brier` | ≤ .12 | .0956 | .0426 | .0438 | .0670 |
| 24 | `interleaved/stage_2/cue_0/cue_benefit` | ≥ .002 | .0058 | .0333 | .0254 | .0137 |
| 25 | `interleaved/stage_2/cue_1/cue_benefit` | ≥ .002 | **.0008 FAIL** | .0161 | .0128 | .0071 |
| 26 | `interleaved/stage_2/cue_2/cue_benefit` | ≥ .002 | .1404 | .2145 | .2126 | .1060 |
| 27 | `interleaved/slot_4/marginal_gain` | ≥ .02 | .1282 | .1829 | .1819 | .1501 |
| 28 | `interleaved/slot_4/absolute_brier` | ≤ .12 | .0973 | .0427 | .0437 | .0755 |
| 29 | `interleaved/stage_3/cue_0/cue_benefit` | ≥ .002 | .0056 | .0337 | .0261 | .0106 |
| 30 | `interleaved/stage_3/cue_1/cue_benefit` | ≥ .002 | **-.0003 FAIL** | .0128 | .0099 | .0046 |
| 31 | `interleaved/stage_3/cue_2/cue_benefit` | ≥ .002 | .1206 | .2144 | .2131 | .0849 |

- The learner fails 4 cells, P fails 2, and O and I fail none.
- The learner rescoring matches the locked summary with a maximum absolute
  difference of 0.0. Pass flags and the STOP decision are reproduced.
- Cell 8: the learner (.0207) beats I (.0189). I is not an upper bound for cue
  benefit (§10).

## 5. Posterior diagnostics (I, λ = 1e-4; 12 supports per slot)

| Target | Mean mass on true law | Min | Mass on true hidden mode | MAP = true law |
|---|---:|---:|---:|---:|
| Base A | 1.000 | 1.000 | 1.000 | 100% |
| Base B | .684 | .182 | 1.000 | 75% |
| Stage 1 | .516 | .128 | 1.000 | 67% |
| Stage 2 | .513 | .171 | 1.000 | 67% |
| Stage 3 | .623 | .112 | 1.000 | 50% |

**The hidden mode is always identified.** Mass on the true mode is 1.000 in all
60 supports.

**Where the misallocated mass goes.** Across the 60 supports, 19.96 units of
posterior mass fall on wrong laws:

| Wrong law differs from the true law by | Share of wrong mass |
|---|---:|
| efficiency only (cue 0) | 46.8% |
| delay only (cue 1) | 38.1% |
| two cues | 15.1% |
| supply timing only (cue 2) | ~0 (.003 units) |
| mode | 0 |

So the confusion is **not** "almost entirely" delay-neighbour. Efficiency-only
neighbours take the larger share.

**Delay cells, seed-averaged:**

| Delay cell | Mass on true law | Mass on delay-active laws |
|---|---:|---:|
| Stage 1 | .478 | .645 |
| Stage 2 | .525 | .839 |
| Stage 3 | .623 | .786 |

Per seed at stage 3, mass on the true law ranges from .379 (18101) to 1.000
(18102, 18104).

I still passes despite this confusion, because laws it cannot tell apart also
forecast alike.

## 6. Information per support record

Values are plug-in MC expected KL in nats per record. Each is averaged over each
seed's 512 query cases and a uniformly random performed action, at λ = 1e-4, and
reported as the seed mean for structurally matched pairs. "with / without" means
the law that includes the dependency is true / the law that lacks it is true.

**n20 = ln 20 / KL** is the support size at which the *expected* log-odds
reaches 20:1. It is not the typical number of records needed; see the direct
check below.

**Channels:**

- (i) the performed action's 3-horizon pattern — what the learner and I
  receive;
- (ii) the performed action's exact lifetime;
- (iii) all 5 actions jointly — privileged; neither the learner nor I receives
  it.

| Pair (with / without) | (i) KL | (i) n20 | (ii) KL | (ii) n20 | (iii) KL | (iii) n20 |
|---|---:|---:|---:|---:|---:|---:|
| Mode (Base A vs Base B) | 4.10 / 4.10 | .73 / .73 | 4.62 / 4.62 | .65 / .65 | 12.0 / 11.9 | .25 / .25 |
| Add cue 2 (supply timing) | 1.16 / 1.05 | 2.6 / 2.8 | 1.91 / 1.51 | 1.6 / 2.0 | 4.99 / 4.72 | .6 / .6 |
| Add cue 0 (efficiency) | .042 / .052 | 71 / 57 | .064 / .090 | 47 / 33 | .236 / .333 | 13 / 9 |
| Add cue 1 (delay) | .027 / .026 | 112 / 114 | .076 / .070 | 39 / 43 | .111 / .386 | 27 / 8 |

**The delay pair.**

- Its channel (i) n20 ranges from 95 to 130 per seed when the delay law is
  true, and from 88 to 152 when it is not.
- It is the hardest pair in every seed: of the 10 unordered law pairs, it has
  the smallest symmetric KL (.043-.064).
- Its channel (i) KL stays within .023-.029 for λ from 1e-2 to 1e-6.

**The mode and cue-2 pairs depend on λ.** Some records have zero MC count under
the other law at K = 16,384:

- the mode pair: 28-29% of records in channel (i), 32% in (ii);
- the cue-2 pair: only 3-4% in (i).

Their n20 values are therefore upper bounds. For example, the mode KL in
channel (i) is 5.44 at λ = 1e-6 and 2.59 at λ = 1e-2.

**Replicate-based MC bias, delay pair:**

- λ = 1e-4: 1.2e-4 / 1.1e-4 in (i) and 4.3e-4 / 4.0e-4 in (ii), under 1% of
  the KL;
- λ = 1e-6: 5.1e-4 in (i), 1.8% of the KL.

**Direct identification check (delay pair).** Setup: 64 fresh supports per seed
and direction, analysis-only seeds, pairwise uniform prior, K = 2,048,
λ = 1e-4.

| Support size | Mean posterior on the true law | Median | Supports reaching 20:1 |
|---|---:|---:|---:|
| 32 | .58-.64 | .50-.55 | 6-19% |
| 128 | .77-.84 | .90-.98 | 41-55% |
| 512 | .96-1.00 | 1.00 | 84-100% |

- A wrong-way 20:1 happens in at most 1.6% of supports.
- The realised per-record LLRs (.021-.039) run about 13% above the panel KLs
  (.020-.034) on average. Per seed they differ by up to 5.5 SE; the largest gap
  is 18103 with the predecessor law true, .0306 vs .0197. The likely cause is
  sampling error of the 512-case panel, so the KL table is indicative per seed,
  not exact.

## 7. Decision relevance

**Measure.** Survival at H = 12 under evaluator truth, for the first-argmax
action on each cue's affected subset: correct query → cue-flipped query. Values
are seed means. The loss is correct minus flipped; one query is 1/256.

| Cell | O | I | P | Learner |
|---|---|---|---|---|
| Fresh delay | .684 → .681 (+.0026) | .684 → .683 (+.0003) | .654 → .653 (+.0013) | .653 → .656 (-.0026) |
| Interleaved stage-1 delay | .727 → .729 (-.0020) | .727 → .726 (+.0010) | .678 → .674 (+.0039) | .657 → .659 (-.0020) |
| Interleaved stage-2 delay | .674 → .674 (+.0000) | .674 → .673 (+.0010) | .644 → .643 (+.0010) | .609 → .607 (+.0015) |
| Interleaved stage-3 delay | .646 → .639 (+.0072) | .645 → .642 (+.0029) | .585 → .587 (-.0026) | .552 → .552 (-.0003) |

**How much the delay cue is worth to O.**

- At stage 3, flipping the delay cue costs O **1.8 queries in 256 on average**.
  Per seed it costs 2, 3, 3, 2, 0 and 1 in 256, so **up to 3/256**.
- The realised means at stages 1 and 2 (-0.5/256 and 0/256) are dominated by
  the single realised truth draw.
- The expected loss under O's own forecasts avoids that single-draw noise
  (reviewer, `review_16laws_K4096.json`). It is positive in all 12 interleaved
  delay-active seed×slot cases, with mean .0066 (1.7/256) and maximum .013
  (3.3/256). The flip loss is systematic but small.

**The delay cue is not unusually unimportant.** On the same expected-loss
measure (`review_16laws_K4096.json`):

| Cue | Expected flip loss | Loss when O ignores the cue |
|---|---:|---:|
| Efficiency (cue 0) | .0162 | .0007 |
| Delay (cue 1) | .0066 | .00025 |
| Supply timing (cue 2) | .387 | .093 |

On realised truth, O's cue-0 flip loss is +.010 to +.014, the same order as
delay. Efficiency and delay are **both** nearly irrelevant to action choice under
this action set and horizon. The hidden mode and supply timing drive persistence
(O's realised cue-2 flip loss is +.37 to +.41).

Flip sensitivity is not optimality: P's cue-0 flip loss (+.032 to +.045) exceeds
O's.

**All-case survival at the interleaved endpoint** (H = 12, first argmax):

| Slot | O | I | P | Learner | I − learner | Learner − P | Clairvoyant | No transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base A | .751 | .751 | .595 | .639 | .113 | +.044 | .767 | .586 |
| Base B | .748 | .749 | .699 | .655 | .094 | -.045 | .768 | .589 |
| Stage 1 | .718 | .718 | .690 | .644 | .074 | -.046 | .735 | .557 |
| Stage 2 | .697 | .697 | .677 | .630 | .067 | -.047 | .709 | .521 |
| Stage 3 | .675 | .675 | .634 | .593 | .081 | -.041 | .685 | .483 |

- **The learner trails I** by .067-.113 in survival.
- **Base A.** Identifying Base A's mode from the support is worth .156 survival
  (I − P). The learner recovers .044 of it.
- **The other four slots.** The learner is *below* the context-free P by
  .041-.047 in survival and .022-.029 in Brier. On Base A its Brier is .060
  better than P's.
- **Precision.** The two MC replicates differ by up to .005 in these seed means
  and by up to .012 per seed, because argmax is not smooth.

## 8. Verification status

1. **Learner rescoring and case regeneration** (`../loader_verification.json`).
   - All 31 cells, per-seed values and bootstrap bounds match the locked
     summary exactly.
   - Regeneration from the realised supplies exactly reproduces 240
     query-truth arrays (368,640 cells) and 576 support arrays.
   - Deduplicated and naive MC query counts are bit-identical (1,843,200 count
     cells).
   - Supply z-tests over 144 groups: maximum |z| 2.78. Zero-clip counts were
     41 vs 33.5 expected, and 31 vs 25.1 expected.
2. **The reviewer's independent re-implementation**
   (`review_independent_K8192.json`, `scripts/ideal_observer_review_independent.py`).
   - It uses its own supply sampler, physics and posterior. Its physics
     reproduces 210 saved truth and support-outcome arrays.
   - It decodes features from the saved images and uses the locked scorer's own
     functions. K = 8,192, disjoint seeds.
   - All 93 predictor-cell means agree:

     | Predictor | Max abs difference |
     |---|---:|
     | O | 3.8e-5 |
     | I | 3.9e-5 |
     | P | 2.4e-5 |

   - Standardised differences: RMS z .77, maximum 1.77. All pass/fail flags
     agree.
   - Posterior weights differ by at most .029, on near-tied delay-pair supports.
   - No information leak was found.
3. **Exact PCG64 supply replay** (`scripts/ideal_observer_review_supply_replay.py`).
   - The raw draws, pushed through the MC formula, reproduce all 90 query and
     support base-supply and reserve arrays byte-for-byte.
   - Learner forecasts are already monotone over horizons (cummin changes
     nothing: 0.0).
   - This read-only replay was re-run on 2026-09-28 with identical output, and
     git status was unchanged.
4. **Second independent check** (`independent_check.json`,
   `independent_check_exact.json`; `scripts/ideal_observer_check.py`,
   `scripts/ideal_observer_check_exact.py`).
   - It was written without reading the audit code or `results.json`. It uses
     its own pixel decoder, its own noise and reserve sampler, a posterior with
     +.5 count smoothing, and a locked-style scorer. Physics is the project's
     `AcquisitionWorld.simulate`.
   - Draws: 4 replicates × 4,096 query draws, and 65,536 support draws.
   - Its agent ended before comparing the two sets of results. The comparison
     below was computed while this summary was being written, by in-session
     arithmetic on the two JSON files; no comparison file was saved.
   - Coverage: interleaved cells only (5 slot Briers and 9 cue cells; the
     marginal gains follow from the Briers, since the marginal Brier is fixed),
     plus the delay-pair KL and the fresh learner's delay benefit.
   - **Exact-reserve variant vs the primary** (6 seeds × all interleaved cells):

     | Predictor | Per-seed max abs difference |
     |---|---:|
     | O | 1.1e-4 |
     | I | 9.0e-5 |
     | P | 5.5e-5 |

     Standardised differences: RMS z ≈ 1.0, maximum 2.5. The seed-mean maximum
     difference is 7.6e-5. Pass/fail flags are identical, and posterior weights
     agree within .011 across all 60 supports.
   - **Gauge variant.** Against the primary's gauge rerun (seeds 18101, 18104,
     18105): maximum difference 1.1e-4, RMS z 1.2. Against the exact primary:
     up to 3.2e-4, which is the gauge effect itself.
   - **Delay-pair KL.** Channel (i) panel KL matches within .0004 per seed (for
     example 18104: .0315 vs .0314).
   - **Other checks.** All pipeline checks pass, decoded observables match the
     evaluator, and there is no zero-count true-law record. The per-seed fresh
     learner delay benefits average .003365, the locked value.
   - **Verdict: it agrees.**
5. **Weaker 16-law observer** (`review_16laws_K4096.json`).
   - It uses a uniform prior over all 2 modes × 8 cue subsets, so it does not
     know which 5 laws exist.
   - It still passes all 31 cells. Base-A Brier is .0472; the delay cells are
     .0090, .0104 and .0074 (per-seed minimum .0054).
   - Mass on the true mode is 1.0; mass on the exact true law is only
     .33-.40.

**Open item (reviewer, low severity).** `verify_dedup` checks only the query
counts. It does not check the support-count, lifetime or joint histograms.
Checks 2 and 4 reproduce the posterior independently, and no actual error is
known.

## 9. Corrections to the workflow's draft prose (verified against the JSON)

| Draft claim | Corrected (source) |
|---|---|
| Delay flip changes O's survival "by at most about 1 in 256" | 1.8/256 on average at stage 3, up to 3/256 per seed (`results.json`). Expected loss: mean .0066 (1.7/256), maximum .013 (3.3/256) (`review_16laws_K4096.json`). |
| Posterior confusion is "almost entirely" delay-neighbour | 38.1% delay-only, 46.8% efficiency-only, 15.1% two-cue (posterior rows). |
| Learner trails I by ".08-.11" survival | .113 / .094 / .074 / .067 / .081 by slot. |
| Delay-pair MC bias "at most 1e-4" | 1.2e-4 in (i) and 4.3e-4 in (ii) at λ = 1e-4; 5.1e-4 in (i) at λ = 1e-6. |
| "About 29% impossible records" explains the λ dependence of both the mode and cue-2 pairs | The 29% applies to the mode pair; the cue-2 pair has 3-4%. "Impossible" means a zero MC count at K = 16,384. |
| "Hardest of the 20 ordered pairs" | Hardest of the 10 unordered pairs, ranked by symmetric KL. |
| n20 is "the expected number of records" for 20:1 | n20 is where the *expected* log-odds reaches ln 20. Only 41-55% of 128-record supports reach 20:1. |
| Identification LLR is "consistent with" the panel KL | The LLRs run about 13% higher on average, and differ by up to 5.5 SE per seed. |
| Delay is special relative to cue 0's +.010 to +.014 | Efficiency and delay both have negligible decision value. |

Three smaller points where the reviewer's text differs from the JSON:

- **λ range.** The reviewer's "7.49e-5" is the range between any two λ values;
  the shift from the primary λ is 7.41e-5. Both are in the JSON.
- **MC bias.** "Under 1% of KL" holds at λ = 1e-4 but not at 1e-6, where it is
  1.8%.
- **LLR vs KL gap.** The reviewer's "about 10%" is about 13% as a ratio of
  means, and 15% as a mean of ratios.

## 10. Limitations

- **Cue benefit is a sensitivity measure, not a proper score.** Bayes-optimal
  forecasts do not maximise it. For the 15 cue cells, O and I are *references*,
  not upper bounds: the learner beats I in cell 8, and P's cue-0 flip loss
  exceeds O's. O is a genuine floor only for the all-case Brier, which is a
  proper score.
- **I knows more than the learner.** It knows the simulator, the noise law and a
  5-law hypothesis space containing the truth. It measures what the probe *makes
  available*, not what 40,960 single-action training records make *learnable*.
  The 16-law check loosens the hypothesis space but keeps the simulator.
- **Reserves.** The primary uses exact reserves. The gauge-consistent rerun
  covers 3 seeds and moves values by at most 2.25e-4.
- **Noise independence.** Pixel noise shares a PRNG stream with supply noise;
  the observer treats them as independent.
- **KL estimates.** These are plug-in estimates on a 512-case panel. Values that
  involve near-impossible outcomes depend on λ and are reported as ranges.
- **Decision-value figures.** These are argmax-based on a single realised truth
  draw, so they resolve only to about .005.
- **Scope.** Six seeds. The audit is post hoc; it prescribes nothing and relaxes
  nothing.
