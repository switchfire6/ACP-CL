# Step B: the stationary competence gap

This is a prospective protocol, written on 2026-09-28 before any fitting. It
was approved by the user as Step B of
[recoverable_context_proposal.md](recoverable_context_proposal.md).

It is a new, separately stated question about base-learner competence in a
single stationary law. It is not a sweep that rescues the stopped
temporal-representation recipe; that recipe's main config stays unrun. It is
also not a continual-learning comparison.

## Question

In the saved fits, the recurrent reference learner plateaus near .073-.082
Brier in a single law. The law-known Bayes oracle O reaches about .044. The
information audit found this steady-state gap to be 75-95% of the learner's
regret.

This study asks two things:

1. Is the gap caused by how the learner uses its data (repeated updates on a
   small moving window), by memory, by capacity, by exposure, or by
   performed-action-only feedback?
2. Which pre-declared factor closes most of it?

## Fixed elements

- **Law.** `stage_law(seed, 3)`: all three pulse cues are active and the mode
  is `1 - seed % 2`. This is a single stationary law, with no switches.
- **Seeds.** 19101-19106. These are fresh and disjoint from every earlier
  project cohort, and they give three seeds of each mode parity.
- **Learner.** The unchanged recurrent reference (`RepresentationLearner`, arm
  `outcome`, i.e. `ContextLearner` recurrent with interaction features).
  - It uses Adam at .002, gradient clip 5, and context width 12.
  - Each update uses one current plus one replay packet of 32, with the
    packet's actual preceding batch as support.
  - It starts from random initialisation.
- **Training stream.** Records come from the project simulator in the ordinary
  training format: images, a uniform random performed action and that action's
  three survival labels. The stream has 8,192 arrivals except in the exposure
  arm.
- **Scoring.**
  - The endpoint panel is 512 factorial queries of the law, scored with two
    independent 32-record supports. It is drawn from the project evaluator
    seeded by the cohort seed. All-case Brier covers all 5 actions × 3 horizons
    against evaluator counterfactual truth.
  - Cue benefits and survival use the existing definitions.
  - The oracle O is computed on the same panel with
    `scripts/ideal_observer_mc.py` (K ≥ 4,096). Only O uses the law.

## Arms

**Factorial (8 arms).** Every combination of:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Updates per current batch, U | 12 | 3 |
| Replay capacity, R (packets of 32, uniform reservoir) | 16 | 256 |
| Width W (frame, temporal and decoder widths) | 64 | 256 |

The reference arm is U12 R16 W64, i.e. the old recipe.

**Two further arms:**

- **Exposure.** The reference arm trained for 32,768 arrivals. It is scored at
  8,192 (where it must reproduce the reference arm's trajectory exactly) and at
  32,768.
- **All-action feedback.** The reference arm with a loss over all five actions'
  3-horizon outcomes. This is a **privileged diagnostic** that measures the cost
  of performed-action-only feedback, never a candidate learner. It uses the same
  records and the same number of updates.

This gives 10 arms × 6 seeds = 60 fits. Arms with the same seed share their
initialisation where the architectures coincide, and every arm sees the same
arrival stream.

## Outcomes

- **Primary.** Endpoint excess Brier = learner minus O, per seed, on the common
  panel.
- **Secondary.**
  - The online prequential Brier per 1,024-arrival chunk, minus O on the same
    cases.
  - Cue benefit for each cue, relative to O.
  - Survival regret, O minus the learner's first-argmax survival.
  - Wall time, multiply-add work proxy and memory bytes per arm.

## Pre-declared reading

A factor's main effect is its paired high-minus-low difference in excess Brier,
averaged over the four cells of the other two factors, within each seed.

- **A factor matters if both hold:**
  - its seed-mean main effect removes at least 25% of the reference arm's excess;
  - it points in the same direction in at least 5 of 6 seeds.
- **The exposure arm:** if it closes at least 25% of the reference excess by
  32,768 arrivals, a shortage of samples contributes to the gap. If not, the
  gap is a steady state.
- **The all-action arm:** the fraction of the gap it closes measures the cost
  of sparse feedback.
- **A competent base learner** is any declared arm with seed-mean excess
  ≤ .010, i.e. Brier within .010 of O. If none reaches it, the gap is only
  characterised, not solved.

Interactions and arm rankings are descriptive. There are six seeds, and the
bootstrap intervals (20,000 paired resamples, analysis seed 19282026) are
descriptive only. Any learner chosen here must be re-confirmed on fresh seeds
inside the next continual study before anything is credited to it.

## Engineering and integrity

- **Smoke run first.** A reduced run of one seed and 1,024 arrivals checks
  determinism and resumption. Smoke outputs cannot inform scientific choices.
- **Locking.** The config, protocol and source hashes are recorded in a
  manifest before the main fit.
- **Compute.** Runs are CPU-deterministic by default, with at most 6
  single-thread workers.
- **Checkpoints.** Each fit keeps an atomic final checkpoint and saves its
  online forecasts.
- **Existing files.** No existing file is modified.
- **Failures.** Every failed attempt is kept and reported.

## Outputs

- `configs/competence_gap.json` and `configs/competence_gap_smoke.json`.
- `scripts/competence_gap.py`, containing the runner and the analysis.
- `runs/competence_gap/`, holding raw data, manifest and checkpoints.
- `reports/competence_gap/`, holding `results.json` and `summary.md`.
