# Step B2: the achievable competence ceiling and a decomposition of the gap

Prospective protocol, written 2026-09-29 before any fitting. It follows the
Step B result in
[reports/competence_gap/summary.md](../reports/competence_gap/summary.md), under
the user's standing approval to continue with the proposal's recommendations.
It is a single-law competence question. It is not a continual-learning
comparison, and it does not reopen the stopped representation recipe.

## Why

Step B found no competent learner. The best arm stayed .0226 above the
law-known oracle O, and even privileged all-action feedback left .019. The
competence bar, "excess ≤ .010 over O", was never checked against what a
*learner* can achieve from N performed-action records. O knows the simulator.
Any learner must estimate about 64 discrete cells × 5 actions × a 2-d reserve
surface, from labels on one action per record.

Before building any continual-learning mechanism on this base learner, we
need two things:

1. an achievable ceiling at each N, to calibrate the bar;
2. a decomposition of the reference's excess into three shares:
   - **estimation**: finite, sparse data;
   - **perception and architecture**: pixels and this network;
   - **online optimisation**: single pass, constant step, small replay window.

## Fixed elements

- **Law.** `stage_law(seed, 3)`: all three cues are active, and the mode is
  `1 - seed % 2`. One stationary law.
- **Seeds.** 19301–19312. There are 12, fresh and disjoint from every earlier
  cohort, six of each mode parity.
- **Data.** Records come from the Step B arrival stream function
  (`stream_batch` in `scripts/competence_gap.py`). Each record has images, a
  uniform random performed action and that action's three survival labels.
  Every learner of a seed sees a prefix of the same stream. Offline learners
  receive the first N records, N ∈ {2048, 8192, 32768}.
- **Scoring.**
  - All scoring is on the Step B endpoint panel for the seed: 512 factorial
    queries.
  - Pixel learners use two 32-record supports; feature learners use none,
    because the law is single.
  - The measure is all-case Brier over 5 actions × 3 horizons against
    evaluator counterfactual truth, after the cummin over horizons.
  - O is the law-known Monte Carlo oracle on the same panel (exact reserves,
    K ≥ 4,096).
  - Also reported: survival regret (O's survival minus the first-argmax
    survival) and cue benefits.

## Arms

**Online learners** (single pass over 8,192 arrivals; the continual regime):

| Arm | Definition |
|---|---|
| ON-ref | The reference recurrent learner, U12 R16 W64. This replicates Step B. |
| ON-R256 | U12 R256 W64: Step B's best arm, re-tested on fresh seeds. |
| ON-ref-EMA | The ON-ref fit, predicting with an exponential moving average of its weights (decay .998 per optimiser step). Training is unchanged, so it shares the ON-ref fit. |
| ON-R256-EMA | The same, for ON-R256. |

- The EMA is initialised to the initial weights. It is updated after every
  optimiser step, and is never used for training.
- The weight EMA must not change the fit: the online weights, checkpoint and
  signature must equal those of a fit without it.

**Offline ceilings** (for each N ∈ {2048, 8192, 32768}):

| Arm | Definition |
|---|---|
| OFF-pixel | The same `HistoryNetwork` (W64, context 12, interaction features) and the same initialisation as ON-ref, trained offline on the N records. |
| OFF-feature | **Privileged perception.** An ensemble of 5 MLPs trained on the exact latent features, with performed-action labels. A ceiling, never a candidate. |
| OFF-feature-all | **Privileged perception and labels.** OFF-feature trained on all five actions' counterfactual labels. Descriptive only. |

OFF-pixel training details:

- Hold out the last 20% of the N records, by packet, for validation.
- Each minibatch is one 32-record packet, with its actual preceding batch as
  support.
- Adam at .002, cosine decay to 0 over at most 40 epochs, gradient clip 5.
- Performed-action BCE.
- Evaluate on validation every epoch, and keep the best-validation weights.

OFF-feature details:

- Inputs:
  - reserves / 18 (2 values);
  - source, lossy and delayed factor bits (3);
  - the three cue signal bits (3).
- Each MLP has 2 hidden layers of 128 tanh units and 15 logits.
- Training: the same 80/20 split, Adam at 1e-3, minibatch 256, at most 400
  epochs, early stopping with patience 30.
- Weight decay: chosen from {0, 1e-4, 1e-3} on validation only.
- Members differ only in initialisation seed. The forecast is the mean of the
  member probabilities, then cummin.

Held-out validation makes the offline ceilings slightly conservative, since
they train on 80% of N. The panel is never used for any choice.

## Outcomes and pre-declared readings

Define the excess at N as learner Brier minus O Brier on the panel, per seed.
Everything is measured at N = 8,192 unless noted. There are four readings.

**R1. Calibration.**

- If OFF-feature's seed-mean excess at 8,192 is > .010, the ".010 over O"
  competence bar is declared **infeasible at this N** for performed-action
  learning.
- Future bars are then stated relative to the ceiling C(N), OFF-feature's
  excess. The competent standard becomes excess over C(N) ≤ .010.
- Otherwise the bar stands.

**R2. Decomposition.** Split ON-ref's excess into three shares, per seed:

| Share | Definition |
|---|---|
| Estimation | E = OFF-feature excess |
| Perception and architecture | P = OFF-pixel − OFF-feature |
| Online optimisation | Q = ON-ref − OFF-pixel |

- A share is **dominant** if it is ≥ 50% of ON-ref's seed-mean excess and
  positive in ≥ 10/12 seeds.
- Otherwise the gap is **mixed**.
- A negative P or Q is reported as such.

**R3. Replay re-test.** R256 is **confirmed** if ON-R256 removes ≥ 25% of
ON-ref's seed-mean excess and reduces the excess in ≥ 10/12 seeds.

**R4. Weight averaging.** EMA **helps** if ON-R256-EMA removes ≥ 25% of
ON-R256's seed-mean excess and reduces it in ≥ 10/12 seeds. The same test is
reported for ON-ref-EMA against ON-ref, descriptively.

**Descriptive only:**

- learning curves over N for the offline arms;
- online prequential excess per 1,024-arrival chunk for the online arms,
  using both the online and the EMA weights;
- survival regret;
- cue benefits relative to O;
- the cost of sparse labels in the ceiling (OFF-feature minus OFF-feature-all);
- 95% paired bootstrap intervals (20,000 resamples, analysis seed 19392026);
- wall time, multiply-add proxy and memory.

## What each outcome implies (declared now)

| Outcome | Implication |
|---|---|
| Q dominant | The single-pass online update rule, not the architecture or the data, limits competence. The next learner work is online consolidation: weight averaging, or slow and fast weights. This is also a continual-learning mechanism, so it must then be tested against plasticity under law switches. |
| P dominant | The pixel encoder or this network is the limit. Fix the perception and architecture offline first. No continual-learning claim is possible until it is fixed. |
| E dominant | The data are the limit. O is the wrong yardstick at this N. Score future studies as regret against C(N), or a same-information ideal observer, and move towards denser consequences (Step D). |
| Mixed | Report the shares, and choose the next step by the largest share, stating that it is not dominant. |

## Engineering and integrity

- **Smoke run first.** A smoke run (one seed, N ∈ {1024, 2048}, reduced
  epochs) checks four things:
  - determinism;
  - that EMA tracking leaves the online fit bit-identical;
  - that offline data are the exact stream prefix;
  - that O reproduces truth.

  Smoke outputs cannot inform scientific choices.
- **Locking.** Config, protocol and source hashes go in a manifest before the
  main fit.
- **Compute.** CPU-deterministic, at most 6 single-thread workers.
- **Existing files.** No existing file is modified. Step B's source is
  imported, not edited.
- **Failures.** Every failed attempt is kept and reported.

## Outputs

- `configs/competence_ceiling.json` and `configs/competence_ceiling_smoke.json`.
- `scripts/competence_ceiling.py`.
- `runs/competence_ceiling/`.
- `reports/competence_ceiling/`, holding `results.json` and `summary.md`.
