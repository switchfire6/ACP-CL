# Step C: consolidated recall under recurring, unannounced laws

**Status.** Prospective protocol, written 2026-09-29 before any fitting and
revised the same day after an independent pre-implementation critique. The
changes are listed at the end.

**Origin.** It implements the consequence that
[competence_ceiling_protocol.md](competence_ceiling_protocol.md) declared for a
dominant online-optimisation share: test online consolidation against
plasticity under law switches. It replaces the earlier Step C sketch in
[recoverable_context_proposal.md](recoverable_context_proposal.md).

**Scope.** The stopped temporal-representation recipe stays stopped. This is a
new learner question with its own arms and fresh test seeds.

## Motivation and hypothesis

- **Step B2.** Inside one law, averaged ("slow") weights were the largest
  lever on competence: EMA removed 48% of the excess, in 12/12 seeds. But one
  global average lags at every switch and blends laws.
- **Step A1.** A minority law's knowledge was still expressible by the
  network, but its evidence pathway could not retrieve it.

**Hypothesis.** A learner with two kinds of weights will beat plain online
learning, and will beat a single average that restarts on detected switches.
The two kinds are:

- fast weights, for plasticity;
- a small library of slow, consolidated snapshots, one per inferred context.

It selects snapshots by the likelihood they give to the consequences just
observed, and recalls the chosen snapshot into the fast weights. It should win
on steady-state competence and on recovery when a known context returns. The
learner is called **CR, consolidated recall**.

**Expected scope of the claim.** The information audit shows the two hidden
modes differ by about 87 nats per 32-record batch, while the stage laws differ
by about 1 nat per batch at most. So CR is expected to separate the **two
modes**, not five laws. A SUPPORTED verdict will be stated as "consolidated
recall over a binary, highly separable context". It will not claim fine law
indexing.

## Environment (the qualification's interleaved stream)

- **Laws.** `reference_laws(seed)`: Base A, Base B and stages 1–3. Base A is
  the only law in its mode.
- **Schedule.** 8 cycles × (Base A, Base B, stage 1, stage 2, stage 3) × 1,024
  arrivals. That is 1,280 batches of 32, with unannounced switches.
- **Records.** From `AcquisitionWorld.experience`: images, a uniform random
  performed action and its 3 survival labels.
- **Stream seeds.** `trial_seed(seed, 'consolidated_recall_stream', batch)`.
  All arms of a seed see the identical stream.
- **Replay.** At R256 the reservoir starts evicting at batch 256. Through
  cycles 3–8 it holds a uniform sample of 256 of the 320–1,280 batches seen, so
  about 80% of replayed packets come from laws other than the current one. The
  fast weights are therefore multi-law learners. This differs from B2's
  single-law full-history replay.

## Learners

All learners use the unchanged recurrent reference network:

- HistoryNetwork: W64, context 12, interaction features;
- Adam .002, clip 5, U12, R256 (unless stated);
- the seed's shared initialisation.

| Arm | Definition |
|---|---|
| ON-R256 | Reference, R256. Plain online learning. |
| ON-ref | Reference, R16. Descriptive, for continuity with earlier work. |
| EMA-global | ON-R256, predicting with one global bias-corrected EMA of its weights, with its own tuned decay d_G. |
| CR-K1 | CR with K_max = 1. On a detected switch, the single snapshot is restarted from θ. This is **the single average with switch detection**. |
| **CR** | Consolidated recall, K_max = 8 (below). |
| CR-no-recall | CR, but θ is never overwritten. Snapshots are used for prediction only. Descriptive. |
| CR-oracle-mode *(privileged)* | CR with two snapshots indexed by the true **mode** of the previous batch; no inference or spawning. The ceiling for the separable context. |
| CR-oracle-law *(privileged)* | The same with five snapshots indexed by the true **law** of the previous batch. Descriptive; not a ceiling, because it recalls at every stage switch. |
| COND-4 | The project's 4-head conditional learner (`ContextLearner('conditional')`) with its native architecture, run at U12/R256 to match budgets. Descriptive. |

### Evidence statistic (shared by CR and I)

For a record with monotone survival forecasts (p4, p8, p12), the joint
lifetime pattern has four categories:

| Category | Probability |
|---|---|
| Fail by step 4 | 1 − p4 |
| Fail in (4, 8] | p4 − p8 |
| Fail in (8, 12] | p8 − p12 |
| Survive | p12 |

Each category probability is floored at .01 and renormalised. ℓ(model, batch)
is the sum over the batch's 32 records of the log-probability of the observed
pattern for the performed action. The forecasts use the previous batch as
support, exactly as prediction does. This avoids triple-counting the nested
horizons.

### CR mechanism, per arriving batch t

1. **Forecast.** Forecast batch t with the current prediction weights: φ_{k\*}
   if CR is not pending, θ if it is. This is the prequential score.
2. **Score.** Observe the labels. Compute ℓ_k for every snapshot and ℓ_θ for
   the fast weights, on batch t.
3. **Filter.** In log space,

   log p_t(k) = log[(1 − h) p_{t−1}(k) + h/K] + ℓ_k,

   then normalise, with h = .05. K is the current number of snapshots.
4. **Recall.** If some j ≠ k\* has p_t(j) ≥ .95, set k\* = j, copy θ ← φ_j,
   and reset Adam's first moment to zero. The second moment is kept.
5. **Pending.** CR is pending on batch t if ℓ_θ − ℓ_{k\*} > τ. This is a
   paired statistic on the same batch: the fast weights explain the batch
   better than the active snapshot. While pending, φ_{k\*} is not consolidated
   and prediction uses θ.
6. **Spawn.** If ℓ_θ − max_k ℓ_k > τ on 2 consecutive batches:
   - create a new snapshot from θ (bias-correction count 1);
   - set k\* to it, and set the posterior to a point mass on it;
   - pending ends.

   If K = K_max, the snapshot with the fewest total consolidation steps is
   replaced first.
7. **Train.** Train θ on batch t as ON-R256 does. After each optimiser step,
   if CR is not pending, consolidate φ_{k\*}. This is a **bias-corrected**
   EMA with decay d: the raw average a ← d·a + (1 − d)·θ with n ← n + 1,
   giving φ = a / (1 − d^n). No initial-weight residue remains.

**Start.** The first snapshot is spawned from θ at batch 1 (n = 0), with
k\* = 1.

**Oracle arms.** These use steps 1, 4 (recall on an index change), 5
(pending) and 7. The index is given, not inferred, and there is no spawning.

**CR-K1.** Same as CR, except that step 6 replaces the single snapshot, and
step 4 never fires.

### Hyperparameters

**Fixed:**

| Parameter | Value |
|---|---|
| h | .05 |
| K_max | 8 |
| Pattern floor | .01 |
| Recall posterior | .95 |
| Spawn run | 2 batches |
| Adam first moment | Reset at recall |

**Decay d.** Tuned **separately** for CR, CR-K1 and EMA-global over
{.99, .995, .998}, on development seeds 20101–20103. The oracle arms and
CR-no-recall use CR's d.

**Margin τ(d).** Calibrated, not tuned:

- Take the statistic ℓ_θ − ℓ_EMA from the EMA-global development fit with
  the same d.
- Use within-block batch positions 5–32 of cycles 1–2, pooled over the three
  development seeds.
- τ(d) is the 99th percentile of that statistic.

**Selection.** For each arm, choose the d with the lowest mean development
stream excess over O. If several are within one paired seed-SE of the best,
take the largest of them.

**Pathology stop rule (development).** If the selected CR shows any of the
following on development seeds, then CR is not taken to test:

- a pending fraction > .30;
- more than 3× as many k\* changes as true mode switches;
- more than 5 snapshots at the end.

In that case the protocol is revised and a new, disjoint development cohort
is used.

Development results may be inspected freely. They can change only d and τ,
unless the stop rule fires.

## References (evaluator-only)

The Monte Carlo uses 512 supply-noise samples per case, with **common random
numbers across laws and actions**. It covers every training case, for all 5
actions under all 5 laws, and records the joint pattern probabilities.

- **O**, the law-known oracle. It is I's true-law column.
- **I**, the same-information ideal observer:
  - It knows the simulator and the five laws, but not the current law.
  - It runs the step-3 filter over the five laws with h = .05, using the
    pattern likelihood with the floor at 1e-3.
  - Its forecast for batch t is the posterior-weighted mixture after batch
    t − 1.
- **Caveats on I.** I is a reference, not an upper bound in every sense:
  - it sees exact reserves, so it has privileged perception;
  - its hazard does not match the true 1/32;
  - it ignores the fixed cycle order.

  It measures the cost of not being told the law for a sticky Bayesian
  filter.
- **Cost.** About 13 minutes per seed on 6 workers. O alone is computed for
  the development seeds; full I is computed for the test seeds.

## Outcomes

**Prequential Brier.** Every learner forecasts before training on the batch.
The score is all-case Brier over 5 actions × 3 horizons against evaluator
counterfactual truth, after cummin.

| Outcome | Definition |
|---|---|
| **Stream excess** | The mean over the batches of cycles 3–8 of learner case-Brier minus O case-Brier. |
| **Recovery excess over I** | The mean over **positions 2–5** of every Base A block in cycles 3–8 of learner case-Brier minus I case-Brier. Position 1 is excluded because no learner, not even I, has Base A evidence there. |
| Descriptive | Stream excess over I; plateau excess (positions 17–32); survival regret; the recovery curve by position; CR's snapshot count, snapshot × mode and snapshot × law purity, recalls, spawns and pending fraction; costs. Costs cover wall time, the multiply-add proxy (including the extra likelihood passes, about 12% of training) and bytes (8 snapshots are about 1.85 MB, against about 12.7 MB of replay). |

## Pre-declared criteria (12 test seeds, 20201–20212)

A criterion passes if both of the following hold:

- the **seed-mean ratio** is at or below the threshold;
- the learner **improves on the comparator** (a negative paired difference)
  in **≥ 10/12 seeds**.

Paired-bootstrap 95% intervals for the ratio are reported descriptively.

| # | Question | Ratio | Threshold |
|---|---|---|---|
| C1 | Does consolidated recall beat plain online learning? | Stream excess, CR / ON-R256 | .75 |
| C2 | Does indexing beat one average with switch detection? | Stream excess, CR / CR-K1 | .85 |
| C3 | Does recall speed recovery? | Recovery excess over I, CR / ON-R256 | .60 |

**Verdict:**

| Verdict | Condition |
|---|---|
| SUPPORTED | C1, C2 and C3 all pass. |
| PARTIAL | C1 passes, but C2 or C3 fails. |
| NOT SUPPORTED | C1 fails. |

**How the thresholds were set.**

- **C1.** B2's EMA removed 48% within a single law, and switching should cost
  some of that. A .75 ratio asks for half of B2's gain.
- **C2.** CR-K1 already has averaging and switch detection. Indexing only
  avoids re-averaging Base A from scratch on each of its 6 later visits, so a
  smaller effect (.85) is expected.
- **C3.** Recall should give a large recovery benefit, if any.
- **Power check.** This is informal, not a gate. Once the development fits
  exist, the per-seed SD of each ratio is reported. If the implied probability
  of 10/12 is below .5 at the development effect size, we record that
  beforehand.

**Descriptive comparisons:**

- EMA-global;
- CR-no-recall vs CR (the effect of recall);
- CR vs CR-oracle-mode (the inference gap);
- CR-oracle-law;
- CR's fraction of ON-R256's excess over I closed;
- COND-4 and ON-ref.

## What each outcome implies (declared now)

| Outcome | Implication |
|---|---|
| SUPPORTED | Evidence-indexed consolidation is a viable basis for recoverable context knowledge, for separable contexts. Move to Step D (agent-chosen actions, dense consequences, several contexts of graded separability) with CR as the candidate, scored against an ideal agent. |
| PARTIAL, C2 fails | Detecting switches and restarting an average is enough here; keeping multiple snapshots adds nothing measurable. Compare against CR-oracle-mode. If the oracle passes C2 while CR does not, inference is the bottleneck. |
| PARTIAL, C3 fails | Consolidation helps the steady state but not recovery. Recall is mistimed or harmful; compare CR-no-recall. |
| NOT SUPPORTED | Consolidation does not survive switching in this form. Report it, and do not tune further on this benchmark. |

## Engineering and integrity

- **Order of work.** Smoke run, then development phase (which also calibrates
  τ), then lock, then test run, then replicate pass, then analysis.
- **Smoke run.** Seed 20199, with a reduced stream of 2 cycles × 5 laws × 128
  arrivals. It checks:
  - determinism and resumption;
  - that the fast weights of EMA-global are bit-identical to ON-R256;
  - that CR with recall, pending and spawn disabled, and a single snapshot,
    reproduces EMA-global's predictions at the same d;
  - that the bias-corrected EMA matches a float64 closed form;
  - that the oracle arms receive only the lagged index;
  - that the pattern likelihood equals a direct computation;
  - that the filter equals a brute-force forward pass;
  - that O and I reproduce truth (realised supplies) and the common random
    numbers are shared across laws;
  - that the Adam first-moment reset happens only at recall.
- **Locking.** Development results, the selected d values, the τ(d) values
  and a power note are recorded. Then config, protocol and source hashes are
  locked before the test run.
- **Replicate pass (integrity).** Re-fit CR and ON-R256 from scratch for test
  seeds 20201, 20206 and 20211, and compare bit-for-bit. Re-verify every
  stored hash, and CRC-check every archive. Any mismatch stops analysis. This
  machine has shown transient memory faults.
- **Resources.** At most 6 single-thread workers.
- **Files.** New files only. Every failure is kept and reported.

## Outputs

- `configs/consolidated_recall{,_smoke,_dev}.json`.
- `scripts/consolidated_recall.py` and a learner module.
- `runs/consolidated_recall{,_dev,_smoke}/`.
- `reports/consolidated_recall/`.

## Novelty, stated honestly

Each ingredient has precedents:

- fast and slow weights (Hinton & Plaut 1987; complementary learning systems);
- weight averaging (Polyak; EMA);
- model libraries selected by predictive likelihood (MOLe, CN-DPM, COIN,
  BOCPD-style model reuse);
- sticky context filters.

What this study tests is the combination: consolidating *per inferred
context* and recalling the consolidated weights into the fast learner. It is
motivated by two measured facts about this benchmark:

- averaging is the main competence lever;
- minority knowledge is recoverable but not addressable.

It is scored against a same-information ideal observer. It is not a claim of
a new algorithmic primitive.

## Revisions after the pre-implementation critique (2026-09-29)

1. **Recovery.** It is now measured at positions 2–5 and relative to I. As
   first written, position 1's unavoidable cost made C3 nearly impossible to
   pass.
2. **C2 comparator.** EMA-global's d is now tuned separately. The comparator
   is now CR-K1, a single average with switch detection, to isolate indexing
   from switch detection.
3. **Pass rule.** It is now unambiguous: a seed-mean ratio, plus a paired
   improvement in ≥ 10/12 seeds.
4. **Evidence.**
   - The joint-pattern likelihood replaces the sum over nested horizons.
   - The floor is raised from 1e-4 to .01.
   - Surprise is now a paired same-batch statistic.
   - Recall needs a posterior of .95.
   - τ is calibrated from within-block noise rather than tuned.
5. **Development.** A pathology stop rule and a tolerance-based tie rule were
   added.
6. **Specification gaps filled.** The order of operations; the filter in log
   space; the replacement rule (fewest consolidation steps, not least recently
   used); spawn initialisation; Adam at recall.
7. **Initial-weight pollution.** EMAs are now bias-corrected.
8. **Scope of the claim.** Mode-level separation is expected. The
   CR-oracle-mode arm was added, and purity is reported at both mode and law
   level.
9. **I.**
   - Joint patterns, with common random numbers across laws.
   - O is taken as I's true-law column.
   - Its caveats are stated, and "unavoidable" was dropped.
   - The cost estimate was added.
10. **Replay.** R256 evicting in this stream is now stated. COND-4 is
    budget-matched.
