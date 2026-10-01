# Step D: recoverable relational knowledge in a closed-loop persistence world

**Status.** Design, 2026-09-29, revised the same day after an independent
critique (changes listed at the end). This is not a protocol, and nothing has
been fitted.

It follows the pre-declared pivot in
[recoverable_context_proposal.md](recoverable_context_proposal.md), which says
to pivot sooner if A1 or C fails. Step C stopped at development; see
[its report](../reports/consolidated_recall/dev/summary.md).

## What Steps A–C taught (design constraints)

| Finding | Constraint on Step D |
|---|---|
| Failures were learning-limited, not information-limited | Every study needs a same-information ideal agent, verified against brute force, including edge cases. |
| Pixel perception is not our question (B2: about 15% of the gap) | Structured vector observations. |
| Online optimisation dominates; averaging and replay are the levers (B2) | Competent online bases (averaged weights), with **bounded** replay declared per arm. Competence is checked in a stationary context first. |
| Uncalibrated bars misled (B, B2) | Every bar is calibrated against an achievable ceiling at the same experience. |
| Whole-network per-context recall discards shared structure, even with an oracle index (C) | Contexts are built from **shared factors**. Learners factor knowledge into a shared core plus small context parts. |
| Minority knowledge was recoverable but not addressable (A1) | Measure recovery and savings on recurrence under **bounded memory**, not perfect retention. |
| Distinguishable is not the same as decision-relevant (the delay cue cost about .007 survival) | Grade contexts on **both** separability and decision relevance, crossed. |

## The world: a small resource commons

**Entities and episodes.**

- Three entities, with reserves r_i ∈ [0, 18] and maintenance 2 per step.
- An episode lasts T = 12 steps.
- Episodes are independent given the context: reserves are redrawn uniformly
  from U(1, 13) at each episode start.
- Cross-episode compounding is deferred to D3.

**Supply.**

- s_i(t) ~ Normal(μ_i(c, t), .35), with a context-dependent seasonal mean.
- Supply arrives whatever the reserve. After arrival, reserves are clipped at
  the capacity of 18, and the overflow is recorded.
- A small probability of negative supply is allowed and documented. D0 checks
  that it does not drive failures.

**Channels.**

- Three directed pairs: 1→2, 2→3 and 3→1.
- Efficiency e_ij(c) and delay d_ij(c) depend on the context factors (below).

**Action.**

- The agent chooses one of 7 actions at the start of the episode: no transfer,
  or 3 units along one channel in either direction.
- The amount sent is min(3, the donor's reserve), as in the existing world.

**Failure.**

- An entity fails when its reserve reaches 0 or below after maintenance.
  Failure is irreversible.
- Physics continues after failure for accounting only.
- The objective is **joint persistence**: all three entities alive at step 12.

**Probe flow** (the identification channel).

- At each step, each channel carries an external probe of .3 units, injected
  at the channel entry from outside the system. It travels with the channel's
  efficiency and delay.
- It does not depend on any reserve, so its likelihood is exact.
- It is a **world factor with two levels**:
  - *passive*: probes are on;
  - *active*: probes are off, and identification comes only from the
    consequences of the agent's own transfers and from supplies.

**Observations.**

- *At decision time:* the three reserves (exact), plus a 3-bit cue vector.
- *After the episode* (dense consequences):
  - noisy sensor readings (σ_obs = .1) of each entity's supply and each
    channel's probe arrivals, for every step;
  - each entity's failure step, or survival.

  Reserves during the episode are **not** observed. They are deterministic
  given the latent supplies, so observing them would make the sensor noise
  irrelevant and I's likelihood inexact.

## Contexts: factorial, with separability and relevance crossed

Contexts are combinations of three binary factors, so shared structure is
explicit:

| Factor | Level 0 | Level 1 | Graded dial |
|---|---|---|---|
| A: seasonality | Base supply seasons | Seasons of entities 1 and 2 flipped | Flip amplitude |
| B: channel 1→2 efficiency | .9 | Lower | The efficiency gap sets separability |
| C: cue-gated delay on 2→3 | Delay fixed | Delay depends on cue 1 (fast when aligned) | Delay difference |

- That gives 8 combinations.
- **Training** covers 5, including a minority.
- **3 are held out** and appear only late, to test composition.
- The dial values are set in D0. They must place factor B's and factor C's
  contrasts at crossed levels of Chernoff information per episode (high or
  low) and cross-context policy regret (high or low).

**Schedule.**

- Stratified semi-Markov, with run lengths from a declared distribution.
- Some runs are comparable to I's identification time for the low-separability
  contrasts.
- Each minority context recurs at least 4 times per seed, with deliberately
  varied absence lengths.
- Held-out combinations appear in the last quarter.

## References (evaluator-only, verified in D0)

**O, the context-known oracle.**

- It knows the simulator and the context.
- It chooses the action that maximises the Monte Carlo probability of joint
  survival, with M ≥ 2^14 samples.
- It uses common random numbers across actions and arms: the same noise per
  episode index.
- Selection and scoring use **independent** draws, to avoid bias from taking
  the max of noisy estimates.
- The Monte Carlo SE is reported in every table.

**I, the same-information ideal agent.**

- It has a prior over all 8 factor combinations, uses the schedule's
  run-length-augmented filter, and acts myopically on the posterior-predictive
  survival.
- It uses the **full** likelihood of each episode's observations:

  p(sensors, failure pattern | c) = p(sensors | c) · E over the latent supplies
  given the sensors and c of 1[failure pattern].

- The first factor is exact Gaussian. The expectation is estimated by
  conjugate-Gaussian posterior sampling of the latent supplies, followed by a
  deterministic replay of the physics.
- In D0, I is verified against brute-force likelihoods on trajectories that
  exercise the edge cases:
  - clipping at capacity;
  - donor-limited transfers;
  - failure before a transfer arrives;
  - negative supply.

**The cost of being myopic.**

- *Passive level.* D0 measures the action-dependent value of information: the
  mutual information between context and observations for each action, as
  (max − min over actions), compared with the probe information. It must be
  small for I to count as near-optimal.
- *Active level.* The gap I − O is reported as an upper bound on what planning
  for information could add. A one-step-lookahead agent is reported
  descriptively.

**L\*, a practical learning ceiling.**

- It knows the functional form of the physics but estimates the parameters
  online, with factor-structured clustering.
- It is built only if D0 shows it is tractable.

**Easiness checks** (D0):

- a coarse tabular agent: binned reserves × the nearest-centroid context from
  the last episode × cue;
- an unbounded-replay mixture learner (privileged).

If the tabular agent lands near I, the world is too easy and is revised
before D1.

## Learner families (D1 and D2)

Every learner shares:

- the same exploration family: an ensemble of size 5 with Thompson sampling;
- the same **bounded replay budget**: 1,024 episodes against streams of
  20,000 or more;
- declared byte and compute budgets.

| Arm | Role |
|---|---|
| Model-free outcome learner | MLP from observation and action to survival probability. The project's lineage. |
| Model-based consequence learner | A heteroscedastic Gaussian dynamics model of the sensors plus a failure model. It plans by simulating survival (M = 256). The shared core is a **per-channel, weight-shared relational module** (graph-network style). |
| **Amortised in-context learner (the real competitor)** | The same core, conditioned on a GRU or set-encoder over the last k episodes (VariBAD/PEARL-style). |
| Factored code library (the hypothesis) | The same core, plus one small sub-code per factor. Codes are selected by the likelihood of recent consequences, spawned when nothing fits, and the core is globally averaged. |
| Oracle factor codes (privileged) | The ceiling for indexing. **Run first.** |
| Unbounded-replay mixture learner (privileged) | The ceiling for bounded memory. |

"Model-based beats model-free" is **not** a headline. Dense feedback
guarantees it. The informative contrasts are **within** the model-based
family.

## The pre-registered claim region

> **Narrowed after D0b** ([report](../reports/commons_d0b/summary.md)).
>
> - In this world, low separability and high decision relevance cannot be
>   combined. C is identified in about 10 episodes, with a relevance of .038.
> - Separability is therefore only a descriptive covariate. D2's claim is
>   about **rarity, absence length and held-out combinations**.
> - The same-information reference is now the one-step-lookahead agent. It
>   beats myopic I by 16% of I's regret.
>
> The original text follows.

The hypothesis can beat an amortised encoder only where amortised inference is
weak. The claim is therefore pre-registered as an **interaction**: the
advantage of the code library over the amortised learner, measured as regret
relative to I, grows with three things:

- the context's rarity;
- the length of its absence under bounded replay;
- lower separability.

The advantage should also appear as faster recovery on **held-out factor
combinations**, compared with I's identification time.

A null result against the amortised learner is a calibrated negative and will
be reported as such.

**Metrics.**

- **Primary:** cumulative online regret including exploration, against O and
  against I.
- **Mechanism comparisons:** a shared-behaviour condition in which all
  learners train on the same logged data, and greedy probes are scored
  counterfactually on all 7 actions.

## Staging

**D0: world and references.** No learners.

- Build the world, O, I, the likelihood verifier, and the separability and
  relevance calculators (Chernoff information, episodes to a .95 posterior,
  cross-context policy regret).
- Also: the value-of-information measurement, the tabular easiness check, the
  schedule generator, and CPU timing for the planned grid.
- Set the factor dials so the crossed grading holds, and record them.

**D1: competence in stationary contexts.** A prospective protocol.

- Every learner in a single context, and in the unbounded-replay mixture.
- Bars are calibrated against L\* or the privileged feature ceiling.

**D2: recurring and held-out contexts.** A prospective protocol with a
development phase and at most 5 criteria, built around the claim region above.
The oracle factor codes run first.

**D2 criteria (added 2026-09-29, at the user's prompting).** D2 will
pre-register two criteria:

- a **whole-stream cumulative** criterion, which nets early costs of
  exploring, consolidating or keeping memory against later gains;
- a **late-stream** criterion.

Neither will reward speed early on. This lets D2 respect "short-term cost for
long-term gain" in learning itself.

**D2 additions (2026-09-29).** These were prompted by a talk the user shared
(Asawa, UC Berkeley, "Continual Learning Bench"): evaluate "gain" rather than
raw reward.

- **Wiped control per arm.** Each arm has a wiped counterpart whose learned
  state (weights, replay, codes) resets at every true context switch. The
  reset timing is privileged, but it is only a baseline.
- **Normalised gain.** Normalised gain = (wiped regret − arm regret) /
  (wiped regret − the lookahead reference's regret). It is the fraction of
  the available improvement that the learner's accumulated state captures.
  Raw gain alone rewards learners that start out weak.
- **Failure taxonomy.** Each shortfall is labelled:
  - *stability*: excess regret on the recurrence of an already-mastered
    context, against its level at the end of the previous visit;
  - *plasticity*: slow approach to the reference on first encounters and
    held-out combinations.

  D3's compounding world may add explicit drift, where a factor level changes
  meaning.
- **Consistency with Step C.** The talk reports that plain in-context learning
  beat elaborate memory systems. Step C found the same kind of thing: a global
  EMA beat snapshot recall. This supports keeping the amortised in-context
  learner as D2's primary competitor.

**D3: extensions**, only if D2 is informative. **Compounding comes first**:
the user considers it important. It means cross-episode reserve carry-over,
with a long-run survival oracle computed by dynamic programming and
discounted, long-run metrics.

- the active identification level;
- cross-episode reserve carry-over (compounding);
- a topology shift at test time (a 4-entity ring or a reversed channel), as a
  zero-shot test of relational generalisation;
- a late novel factor level (continued plasticity).

## Novelty, stated honestly

PEARL and VariBAD (context posteriors), COIN (contextual memory with
recurrence), OML and ANML, Continual World, CORA and the loss-of-plasticity
work already cover context inference and forgetting.

What is missing is a regret decomposition against an exact same-information
ideal agent, under unannounced recurrence, bounded memory, and graded
separability × relevance, with a factorial held-out composition test.

The most valuable headline would be:

> Under a fixed memory budget, a factored core with likelihood-selected codes
> recovers rare and held-out-combination contexts within X episodes of the
> ideal agent's identification time, where amortised in-context and
> bounded-replay global learners do not; and the gap grows with rarity and
> absence.

## Changes after the critique (2026-09-29)

**Must-fix items addressed:**

1. I uses the full likelihood of the observations, with latent-supply
   sampling. Reserves during the episode are not observed. The probes are
   external and so do not depend on reserves. Edge cases are verified.
2. Bounded replay, with byte and compute budgets. The unbounded learner is a
   privileged reference only.
3. The amortised in-context learner was added as the real competitor, and the
   claim region is pre-registered.
4. Separability (Chernoff information, time to identify) is crossed with
   decision relevance (cross-context policy regret). Run lengths are set
   against identification times.
5. The oracle uses common random numbers, M ≥ 2^14, independent draws for
   selection and scoring, and reports its SE.
6. The six contexts became a factorial design, with held-out combinations.

**Should-fix items addressed:**

- Compounding is deferred to D3, and "persistence" does not compound in D2.
  This is stated.
- The topology shift moved to D3.
- The probe flow became a two-level factor.
- Exploration is handled fairly, with both metrics.
- The model-based learner is heteroscedastic, and "model-based beats
  model-free" is not a headline.
- Stratified schedules; a run-length filter for I.
- CPU timing moved into D0.
- The tabular easiness check was added.
