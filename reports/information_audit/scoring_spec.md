# Qualification scoring specification and read-only endpoint loader

**POST HOC / EXPLORATORY / NOT DECISION-BEARING.** This note documents how the
locked representation-learning qualification cells are computed, so that another
predictor's forecasts (for example an ideal-observer reference) can be scored on the
same footing. It changes no locked record and does not revisit the locked **STOP**
decision in `reports/representation_learning/qualification/summary.json`. Law
identities, hidden modes, clean physical outcomes, counterfactual outcomes and
regenerated case objects are evaluator-only quantities. An ideal observer built from
them is an analysis reference, not a learner.

Files:

- Loader: `scripts/ideal_observer_load.py` (new, read-only).
- Verification record: `reports/information_audit/loader_verification.json`, produced by
  `python -B scripts/ideal_observer_load.py --verify --output ...`.
- Locked scorer being mirrored: `scripts/summarize_representation_learning.py`
  (sha256 `a124f205...`, which matches the run's analysis lock).
- Raw data: `runs/representation_learning_qualification/jobs/outcome_<seed>/<phase>/`,
  seeds 18101-18106, phases `fresh_1`, `fresh_2`, `fresh_3`, `interleaved`.

## 1. Verification result

`score_cells` recomputed all 31 cells from the raw `evaluations.npz` learner arrays.
I compared the output with `summary.json`:

| Quantity compared | Max abs difference |
|---|---:|
| Per-seed cell values (31 cells, 168 seed values) | 0.0 |
| Seed means | 0.0 |
| Bootstrap bounds (lower/upper) | 0.0 |
| Per-phase end-target metrics, marginal gains and cue benefits (24 phases) | 0.0 |

- Pass/fail flags and thresholds match exactly.
- The same 4 cells fail: `interleaved/slot_0/absolute_brier`, `interleaved/stage_{1,2,3}/cue_1/cue_benefit`.
- The recomputed raw conjunction is False, which agrees with the locked STOP.

Regeneration checks passed for all 24 phases:

- 240 endpoint evaluations.
- 48 query panels (one per phase and target).
- 96 support sets.

The following were asserted exactly:

- **Observations:** every saved query observation array (correct and cue-flipped) and every support observation array is byte-identical.
- **Query truth:** the saved truth equals `AcquisitionWorld().counterfactuals(cases)`.
- **Masks:** the affected and valid masks match.
- **Support actions and outcomes:** saved performed actions equal `Evaluator.support`. Saved support outcomes equal `simulate(cases, actions).survival` and the performed-action row of the support counterfactuals.
- **Source lock:** the live source files used for regeneration match the hashes in the run manifest.

Two negative controls are also exercised:

- Regeneration raises on a wrong law or a wrong seed.
- Scoring raises on a non-monotone forecast.

## 2. What was evaluated (trace layout)

For each phase, `evaluations.json` holds one record per evaluator call, in this order:

- Trace 0 is the entry evaluation.
- Traces 1..C are the online curve probes: C = 8 for fresh phases and 40 for interleaved.
- The endpoint probes follow.

The endpoint probes are the only traces used by the qualification cells. Their
indices come from `result.json["end_probes"]`. There is one entry per target slot, in
`phase_spec["targets"]` order, with `correct[r]` and `flips[str(cue)][r]` for support
replicate r in {0, 1}. The trace order is:

- for each target,
  - for each replicate r:
    - the correct probe,
    - then one flipped probe per active cue, in ascending cue order.

The endpoint ranges are:

| Phase | Endpoint traces |
|---|---|
| `fresh_1` | 9..12 |
| `fresh_2` | 9..14 |
| `fresh_3` | 9..16 |
| `interleaved` | 41..62 |

Arrays for trace t are stored in the npz under the key `e<t>_<name>`, with these names:

- `probabilities`: float32, shape (512, 5, 3)
- `truth`: uint8, shape (512, 5, 3)
- `observations`: uint8, shape (512, 4, 12, 16)
- `affected`, `valid`: bool, shape (512,)
- `support_observations`: uint8, shape (32, 4, 12, 16)
- `support_actions`: uint8, shape (32,)
- `support_outcomes`: uint8, shape (32, 3)

Every endpoint probe uses the terminal fitted model (`model_sha256 == final_model_sha256`)
and `branch == "novel"`.

### Evaluator case generation (all `TraceEvaluator(seed, config)`, i.e. `acquisition.study.Evaluator`)

- **Query panel for law L.** `Evaluator.dataset(L)` is
  `AcquisitionWorld.dataset(L, 512, trial_seed(seed, "acquisition_query", 0))`.
  - `ConditionalWorld.dataset(Condition(L.mode))` builds the 8-cell factorial grid (source ±1 × lossy × delayed), 64 queries per cell.
  - Hidden mode 1 swaps the two supply columns and changes nothing observable.
  - The three pulse-order signals form a complete 8×8 factorial within each grid cell, because 512 % 64 == 0.
  - Truth is `AcquisitionWorld.counterfactuals(cases)`: survival at horizons 4/8/12 for each of the 5 actions `[0,-2,2,-4,4]`. It is cached per law.
  - **Consequence (verified):** within a seed, all target laws and all four phases present the *same* 512 query images. Only the truth differs by law.
- **Support replicate r for law L.** `Evaluator.support(L, r)` is
  `AcquisitionWorld.experience(L, 32, trial_seed(seed, "acquisition_support", [r, False]))`.
  - Size 32 is not a multiple of 64, so signals are drawn i.i.d.
  - Actions are drawn uniformly over 5 using the seed `trial_seed(data_seed, "performed_action", 0)`.
  - Outcomes are the physical survival of the performed action, with noise 0.
  - **Verified:** support observations and actions are identical across target laws for a given (seed, r). Only outcomes differ. Fresh and interleaved probes of the same law use identical supports and truth.
- **Cue flip for cue c.** `signals[:, c] ^= 1`, then `signal_images` re-renders the pulses in frames 0 and 2. Truth, the valid mask and the support are unchanged.

### Forecast path and monotonicity

The learner forecast for each probe is `predict_all(learner, observations, support)`
(`predictive_value/mechanism.py`). This calls `RepresentationLearner.predict`, then
`ConditionalLearner.predict`, which runs `model.eval()` and takes
`probs.cummin(dim=-1)` over the horizon axis. Saved probabilities are therefore float32,
non-increasing over horizons 4 to 8 to 12, and not clipped.

`predict_all` only rejects values outside [−eps, 1+eps] with eps = float32 eps. The
locked `score` additionally requires:

- float dtype,
- finite values in [−eps, 1+eps],
- `np.diff(p, axis=-1) <= eps`.

A substituted forecast must satisfy the same checks. A calibrated survival forecast is
monotone automatically, because the survival events are nested. `monotone(p)` in the
loader applies the same running minimum as `cummin`.

## 3. Per-evaluation metrics (locked `score`)

For one evaluation with forecasts p (n×5×3), truth y (n×5×3, binary and
non-increasing over horizons), n = 512:

- `error_i = mean_{a in 5 actions, h in 3 horizons} (p_iah − y_iah)^2`, computed in float64.
- **All-case Brier:** `brier = mean_i error_i` over all 512 queries, all 5 actions and all 3 horizons, against evaluator truth. The score covers every action, not just the performed one.
- **Focus Brier:** `focus_brier = mean_{i in affected} error_i`.
- **Marginal Brier:** `marginal_brier = mean_{i,a,h} (m_ah − y_iah)^2`, using the training-only marginal m (defined below). It does not depend on the forecast.
- Also computed but not used by any cell:
  - `survival = mean_i y[i, argmax_a p[i,a,h=12], h=12]` (first argmax)
  - `valid_brier`
  - `no_transfer`
  - `clairvoyant_upper`

**Affected masks** (`acquisition.world.mask_affected(cases, cue)`), where factor columns are (source, lossy, delayed):

| Cue | Dependency | Affected queries | Count of 512 |
|---|---|---|---:|
| None (correct probe) | none | all queries | 512 |
| 0 | directional efficiency | lossy (`factors[:,1]==1`) | 256 |
| 1 | arrival delay | delayed (`factors[:,2]==1`) | 256 |
| 2 | supply timing | lossy or delayed | 384 |

`valid` for branch `novel` selects clean, immediate queries (128). No cell uses it.

**Training-only marginal.** For each law key (`m<mode>_a<active>_r<revised>`), the phase's
own training stream gives `count[a]` (performed-action counts) and `success[a,h]`
(survivals). The marginal is `m = (success + .5)/(count + 1)`. In `interleaved`, the
totals pool all 8 blocks of that law (8192 arrivals). The value is stored in
`result.json["marginals"]` and repeated in each correct probe's `marginal` field.
Equality is asserted. Flipped probes carry `marginal = None`.

## 4. Per-target endpoint summary (locked `endpoint_targets`)

For each target slot and each support replicate r in {0, 1}:

1. `M_r = score(p_correct_r, truth, affected=all, valid, marginal)`.
2. For each active cue c of the target law, let `F = flipped probe (c, r)`. Then

   `benefit_{c,r} = score(p_F; F.truth, F.affected, F.valid).focus_brier − score(p_correct_r; F.truth, F.affected, F.valid).focus_brier`

   Both terms are scored on the same affected subset (the flipped trace's cue-c mask) against the same truth. **Cue benefit = flipped-minus-correct on the affected subset.** A positive value means the forecast gets worse when that cue's pulse order is reversed, so the model uses the cue.

Averaging within the seed:

- `metrics[k] = float(np.mean([M_0[k], M_1[k]]))` for every metric k.
- `marginal_gain = metrics["marginal_brier"] − metrics["brier"]`. This is taken after replicate averaging. `marginal_brier` is identical for both replicates.
- `cue_benefits[str(c)] = float(np.mean([benefit_{c,0}, benefit_{c,1}]))`.

## 5. The 31 cells (locked `qualification`)

`reference_laws(seed) = [baseA = Law(seed%2), baseB = Law(1−seed%2), stage_law(seed,1), stage_law(seed,2), stage_law(seed,3)]`,
where `stage_law(seed,s) = Law(1−seed%2, sorted(order(seed)[:s]))` and
`order(seed) = permutations(range(3))[seed%6]`.

The six seeds cover all six cue orders:

| Seed | Order |
|---|---|
| 18101 | (2,1,0) |
| 18102 | (0,1,2) |
| 18103 | (0,2,1) |
| 18104 | (1,0,2) |
| 18105 | (1,2,0) |
| 18106 | (2,0,1) |

| Cell | Phase / target used | Per-seed value | Rule | Seeds |
|---|---|---|---|---|
| `fresh/stage_s/marginal_gain` (s=1,2,3) | `fresh_s`, its only target `stage_law(seed,s)` | `marginal_gain` | mean ≥ .02 | 6 |
| `fresh/stage_s/cue_benefit` | same | `cue_benefits[introduced cue order(seed)[s−1]]` (only the introduced cue; other active cues are ignored) | mean ≥ .002 | 6 |
| `fresh/cue_c/marginal_gain` (c=0,1,2) | the single fresh phase per seed whose introduced cue is c (stage `order.index(c)+1`) | `marginal_gain` | mean ≥ .02 | 6 |
| `fresh/cue_c/cue_benefit` | same | `cue_benefits[c]` | mean ≥ .002 | 6 |
| `interleaved/slot_k/marginal_gain` (k=0..4) | `interleaved`, `end_targets[k]`: 0 = baseA, 1 = baseB, 2/3/4 = stage 1/2/3 law | `marginal_gain` | mean ≥ .02 | 6 |
| `interleaved/slot_k/absolute_brier` | same | replicate-averaged all-case `brier` | mean ≤ .12 | 6 |
| `interleaved/stage_s/cue_c/cue_benefit` | `interleaved`, slot s+1 (`stage_law(seed,s)`), flips of cue c | `cue_benefits[c]` | mean ≥ .002 | seeds with c in `order(seed)[:s]`: 2/4/6 at s=1/2/3 |

Seed membership of the stage-cue cells:

| Stage | cue 0 | cue 1 | cue 2 |
|---|---|---|---|
| 1 | 18102, 18103 | 18104, 18105 | 18101, 18106 |
| 2 | 18102, 18103, 18104, 18106 | 18101, 18102, 18104, 18105 | 18101, 18103, 18105, 18106 |
| 3 | all six | all six | all six |

The output lists cells in the locked order:

1. the fresh stage groups, then the fresh cue groups (marginal gain, then cue benefit);
2. for each slot 0..4: marginal gain, then absolute Brier, then (slots ≥ 2) that stage's cue cells for cues 0, 1 and 2.

That gives 3·2 + 3·2 + 5·2 + 9 = 31 cells.

**Seed aggregation and averaging order.**

1. Within an evaluation: take the per-query 15-cell mean, then the mean over queries (all queries, or the affected subset).
2. Within a seed: average over the 2 support replicates. Marginal gain is formed after this averaging. Cue benefit is differenced per replicate and then averaged.
3. Across seeds: take the unweighted `values.mean()` in seed order 18101..18106 (the completion.json job order).

A cell passes when its seed mean satisfies the threshold; each cell is one conjunct. The
descriptive 95% interval is the 2.5/97.5 percentile of 20000 resampled seed means,
drawn with `np.random.default_rng(28192026).integers(0, n, (20000, n))` (same draws for
every cell with the same n). The interval never affects pass/fail.

## 6. Substituting another predictor

`score_cells(endpoints, probabilities)` accepts:

- `None`, which uses the saved learner forecasts;
- a callable `f(endpoint, evaluation) -> (512,5,3) float array`;
- a mapping keyed by `(seed, phase, trace)`.

It must supply forecasts for all 240 endpoint evaluations: correct and flipped probes. For a flipped probe, the
predictor must be given `evaluation.observations`, which are the *flipped* images, and the
same support (`evaluation.support()`).

Held fixed:

- truth,
- affected/valid masks,
- training-only marginals,
- support pairing,
- replicate averaging,
- cell membership,
- thresholds,
- seed order,
- bootstrap draws.

As a check, substituting the training-only marginal itself gives marginal gain ≈ 0
(|max| 5.6e-17 from summation order) and cue benefit exactly 0. Substituting the truth gives
all-case Brier 0.

**Information boundary.** A learner at an endpoint probe receives only:

- the 4 query frames,
- the 32-record support (images, performed action, 3-horizon survival).

`regenerate(endpoint)` exposes the evaluator-owned case objects behind each probe:

- `LawCases` (with base `Cases`), reserves, factors, signals and law;
- base supplies (clipped noisy draw after the mode swap);
- law-effective supplies (action-independent, asserted);
- per-action efficiency and delay;
- full support counterfactuals.

These objects are for building and verifying an ideal-observer reference, not for any learner.

## 7. Structural observations (both confirmed)

**(1) Hidden-mode imbalance: CONFIRMED.** `stage_law` (`acquisition/world.py`, line 126)
returns `Law(1-seed % 2, ...)`, and `reference_laws` (`representation_learning/design.py`,
line 20) is `[Law(seed%2), Law(1-seed%2), stage_law(seed,1..3)]`. For every seed:

- baseA is the only reference law with mode `seed%2`. baseB and all three novelty laws have mode `1−seed%2`.
- All three fresh targets also use mode `1−seed%2`, so fresh fits never see baseA's mode.
- baseA's mode takes 20% of the interleaved stream: 8192 of 40960 arrivals from `phase_spec` blocks, and 256 of 1280 training packets with `law_index == 0` in `training.npz`.

The mode is hidden:

- Query images for baseA and baseB are byte-identical.
- baseA's base supplies are exactly baseB's with the columns swapped.
- Truth differs between baseA and baseB on 451-465 of 512 queries, depending on the seed.

The mode can only be inferred from support outcomes.

**(2) The first delay-cue law differs from its predecessor only by the delay dependency: CONFIRMED.**
Let k = `order(seed).index(1)+1`. Then `stage_law(seed,k)` and its predecessor
(`stage_law(seed,k−1)`, or baseB when k = 1) have:

- the same mode, `revised` and `noise`;
- active sets whose symmetric difference is exactly {1}.

The predecessor is also the immediately preceding block in every interleaved cycle.

| Seed | k | Law, active cues | Predecessor, active cues |
|---|---:|---|---|
| 18104, 18105 | 1 | {1} | baseB {} |
| 18101 | 2 | {1,2} | {2} |
| 18102 | 2 | {0,1} | {0} |
| 18103, 18106 | 3 | {0,1,2} | {0,2} |

Physics confirms this. In `AcquisitionWorld.physics`, cue 1 only rewrites `delay[delayed]`
(1 if the action direction matches `signals[:,1]`, else 4, instead of the base 2). On the
regenerated panels:

- supplies and per-action efficiency are identical between the two laws;
- delay changes only on delayed queries;
- truth differs only on delayed queries, and only on 13-22 of the 256 delayed queries (30-57 truth cells).

**Exploratory context (not decision-bearing).** Physically flipping a cue's signal bit
in the regenerated query cases changes the counterfactual truth on this many affected
queries per law:

| Cue | Queries whose truth changes |
|---|---|
| 0 | 68-111 of 256 |
| 1 | 28-40 of 256 |
| 2 | 327-347 of 384 |

No non-affected query changes for any cue. The arrival-delay cue therefore carries much
less outcome-relevant information per affected query than the other cues. This bounds any
predictor's attainable cue-1 benefit and should be taken into account before reading the
cue-1 cells, including the fresh value of .003365, as representational failures.

## 8. Loader usage

Run Python with `-B` and `PYTHONDONTWRITEBYTECODE=1`. The module also sets
`sys.dont_write_bytecode`.

```python
import sys; sys.path.insert(0, "scripts")
import numpy as np
import ideal_observer_load as L

endpoints = L.load_all()                                   # {(seed, phase): Endpoint}, hashes verified
ep = endpoints[18101, "interleaved"]
for ev in ep.evaluations:                                  # 22 endpoint probes, trace order
    print(ev.trace, ev.slot, ev.replicate, ev.law, ev.cue, ev.flipped, ev.probabilities.shape)

regen = L.regenerate(ep)                                   # asserts byte/exact identity
query, support = regen.cases_for(ep.targets[3].flips[1][0])  # stage-2 law, cue-1 flip, replicate 0
query.cases, query.truth, query.factors, query.signals, query.realised["supplies"]
support.cases, support.actions, support.outcomes, support.truth

learner = L.score_cells(endpoints)                         # 31 cells from the saved learner arrays
print(learner["seed_means"]["interleaved/slot_0/absolute_brier"])   # 0.19559...
print(L.compare_with_summary(learner)["max_abs_diff"])     # 0.0

def my_forecast(endpoint, evaluation):                     # substitute any (512,5,3) forecast
    return L.monotone(np.broadcast_to(endpoint.targets[evaluation.slot].marginal,
                                      evaluation.truth.shape).astype(np.float64))
other = L.score_cells(endpoints, my_forecast)
other["cells"][0]["per_seed"], other["cells"][0]["metric"]["mean"]
```

From the command line:

```
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B scripts/ideal_observer_load.py --verify
```

This runs the full verification in about 4 s. `--output` writes a new JSON file under
`reports/information_audit/` and refuses to overwrite an existing one.
