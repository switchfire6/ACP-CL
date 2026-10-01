# Step D1: learner competence in stationary commons contexts

**Status.** Prospective protocol, written 2026-09-29 before any learner was
fitted in the commons world.

**Inputs.**

- The world is the accepted D0b world, `reports/commons_d0b/world_spec.json`,
  key `1586f04b8d3cc8c2`.
- Design: [step_d_design.md](step_d_design.md).

## Question

Can the D2 learner families become **competent** within one stationary
context, from their own agent-chosen experience, under the bounded memory
budget that D2 will use?

"Competent" has a specific meaning here. The learner's greedy decisions must
lose much less survival than the factors B and C are worth (.038). If they
do not, D2 cannot detect any context effect, just as happened with the pixel
learner in Steps B and B2.

## Fixed elements

- **Contexts.** The 5 D2 training combinations: A0B0C0, A1B0C0, A0B1C0,
  A0B0C1 and A1B1C1. Each is **stationary** for the whole fit.
- **Budget.** N = 4,096 episodes per fit. This is about the per-context
  experience in a 20,000-episode D2 stream.
- **Seeds.**
  - Test: 21101–21112, each crossed with all 5 contexts.
  - Development: 21001–21002.
  - The world's randomness per seed and episode index is shared across arms
    through common random numbers.
- **Observations.** Exactly as the world defines them: decision-time reserves
  and cue; then supply and probe sensors and per-entity failure steps. There
  are no counterfactual outcomes.
- **Memory.** A uniform reservoir of **1,024 episodes** for every bounded arm.
- **Compute.** Every arm reports wall time, a multiply-add proxy and bytes.

## Arms

| Arm | Definition |
|---|---|
| **MF** (model-free) | An ensemble of 5 MLPs (2 × 128 tanh). Input: reserves / 18 (3 values) and cue (3). Output: 13 joint-survival logits, plus auxiliary per-entity survival at steps 4, 8 and 12, trained for the chosen action. Each member trains on a Bernoulli(.5) bootstrap mask of the replay. The actions and weights are specified below the table. |
| **MB** (consequence model) | Three learned models, planned over (details below). |
| MF-unbounded, MB-unbounded | The same arms with unlimited replay. *Privileged* memory references. |

**MF actions and weights.** Actions use ensemble Thompson sampling: draw one
member per episode and act greedily. Acting and greedy probes use the EMA
weights of each member (decay d per step).

**MB models.**

- **Supply model.** A per-entity, per-step Gaussian (mean and log-variance) of
  the supply-sensor readings. It is a free parameter table in D1; D2
  conditions it on the context representation.
- **Probe model.** The same form, for the probe sensors. It does not affect
  planning in D1; D2 uses it for context evidence.
- **Failure model.** An ensemble of 5 MLPs (2 × 128 tanh). Input: reserves,
  cue, action (one-hot of 13) and the 36 sensed supplies. Output: per-entity
  discrete-time hazards over 12 steps. Trained by the likelihood of the
  observed failure steps.

**MB planning.**

- For each action, draw M = 256 supply-sensor sequences from the supply model,
  using common random numbers across actions.
- Predict each entity's survival to step 12 with one Thompson-drawn
  failure-model member.
- Estimate joint survival as the product over entities, averaged over the
  samples.
- Take the argmax. Greedy probes use the ensemble mean. EMA weights are used
  as in MF.

**Training (both families).** After each episode, run U minibatch updates of
64 episodes from the replay, with Adam at learning rate lr and gradient clip 5.

## References

- **O**, the context-known oracle. It uses the D0b scoring tables (2^14
  draws, stored for all 13 actions).
- In a stationary context, the lookahead and I-myopic agents reach O after
  about one episode for A and B, and after about 10 episodes for C.
- **Easiness bounds:** no transfer, and random actions.

## Outcomes

**Primary: greedy-probe regret.**

- Every 256 episodes, each arm's greedy action (ensemble mean, EMA weights) is
  scored on a fixed panel of 512 decision states drawn from the context.
- Each action is scored by O's value table. Regret is O's best value minus the
  value of the chosen action.
- The panel is common to all arms and disjoint from the training streams.
- The endpoint is regret at N = 4,096.

**Secondary:**

- cumulative online regret including exploration (the realised stream, scored
  by O's tables);
- learning curves;
- per-entity failure log-loss;
- for MB, the calibration of its supply model and its planning error (its
  predicted value against O's value for the same action);
- costs.

## Development phase (declared knobs only)

- **Seeds:** 21001–21002, all 5 contexts.
- **Grid:**
  - lr ∈ {1e-3, 3e-3};
  - U ∈ {2, 8};
  - EMA decay d ∈ {.99, .998}.

  That is 8 configurations per family, chosen separately for MF and MB (the
  unbounded arms use the same configuration).
- **Rule:** the lowest development mean of greedy-probe regret at 4,096. If
  several configurations are within one paired SE of the best, take the
  cheapest.
- Development may change **only** these three knobs.

## Pre-declared readings (12 test seeds × 5 contexts)

**R1. Competence (per bounded arm).** An arm is **competent** if both of the
following hold:

- its greedy-probe regret at 4,096, averaged over contexts, is ≤ **.015**;
- its worst context is ≤ .025;

and both hold in ≥ 10/12 seeds, on each seed's context-average.

Rationale: .015 is about 40% of B's and C's relevance (.038), so a competent
learner's own decision error is well below what context knowledge is worth.

**R2. The cost of the memory budget.** The bounded arm's regret is compared
with its unbounded version. The budget "costs" if it adds ≥ .005 of regret in
≥ 10/12 seeds. Descriptive, but it informs D2's bounded-memory claim. If the
budget costs nothing even here, where the context is stationary, then any D2
memory effect comes from the switching.

**Descriptive:**

- MF vs MB;
- curves;
- how close cumulative online regret including exploration gets to O;
- the per-context breakdown, especially C-active contexts, where the cue
  matters.

## What each outcome implies (declared now)

| Outcome | Implication |
|---|---|
| At least one bounded arm is competent | D2 proceeds with the competent family or families as cores. |
| No bounded arm is competent, but an unbounded arm is | The 1,024-episode budget is too tight at this N. D2 raises the budget to the smallest competent size, measured in a declared D1b development round, and states this. |
| No arm is competent | The learner family is the bottleneck. A declared D1b round tries architecture remedies (width, ensemble, horizon auxiliaries) on fresh development seeds before any D2. |

## Engineering and integrity

- **Smoke run.** Seed 21099, 256 episodes. It checks:
  - determinism and resumption;
  - that each arm sees only legal observations (an assertion on its inputs);
  - that the common random numbers really are common across arms;
  - that O's table lookups are correct;
  - that bounded replay never exceeds 1,024 episodes;
  - that the EMA does not interfere with training.
- **Then:** development, lock, test run.
- **Replicate pass.** 3 test seeds are re-fit from scratch and compared bit
  for bit. Every stored hash and CRC is re-verified. Any mismatch stops
  analysis.
- **Resources.** At most 6 single-thread workers. New files only. Every
  failure is kept.

## Amendment before the lock (2026-09-29, after development and before any test fit)

**The change.** Greedy-probe checkpoints for the **test** run are reduced from
every 256 episodes to episodes {256, 512, 1,024, 2,048, 3,072, 4,096}.

**Why.** In development the probes took about 80% of MB's compute. The
primary endpoint (4,096) is unchanged; only the resolution of the descriptive
learning curve drops. The amendment does not depend on any test data.

**Development results used for selection** are in
[reports/commons_d1/dev/](../reports/commons_d1/dev/summary.md):

- MF: lr .003, U 2, d .998.
- MB: lr .001, U 2, d .99.

**Caveats recorded before the test (interpretation only; no rule changes):**

- **The bar is context-dependent.** C is worth about .010 in the A·B0C0
  contexts, so "competent" does not ensure that decision error is below C's
  value there.
- **Winner's curse.** The development best (.0107) is biased low.
- **R2 mixes two effects.** The bounded MB overfits its reservoir: hazard
  negative log-likelihood rises with the amount of training. So R2's "budget
  cost" mixes memory capacity with over-training.
- **The selected MB is a slow learner:** regret .064 at 1,024 episodes. That
  matters for D2's short context runs.

## Outputs

- `configs/commons_d1{,_smoke,_dev}.json`.
- `scripts/commons_d1.py`, plus learner modules under `src/acp_cl/commons/`.
- `runs/commons_d1{,_dev,_smoke}/`.
- `reports/commons_d1/`.
