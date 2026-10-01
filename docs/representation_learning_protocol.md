# Temporal representation learning: prospective protocol

Written 2026-09-28 before new scientific fitting. Implements the approved
[design](representation_learning_design.md). This is a bounded test of one
learning objective, not a claim that reconstruction solves continual learning.
Historical experiments and their stopping decisions remain unchanged.

## Hypothesis and observable information

Learning temporal structure before it becomes outcome-relevant can improve
subsequent acquisition under bounded experience, memory and computation. Use
the existing conserving two-object simulator and all three pulse-order cues:
directional efficiency, arrival delay and supply timing. Every cue is visible
from the beginning; each last frame is identical across its two values.
Success here would concern preparation and reuse, not discovery of unseen
input structures or recursive self-improvement.

An ordinary learner receives four pre-action images and one random performed
action's three survival labels. Its history is the preceding 32-record batch.
It receives no law identifier, boundary flag, future observations, simulator
variables, cue labels or unperformed-action outcomes. All counterfactual
outcomes, clean support selections and cue interventions are evaluator-only.
Predictions on incoming records are saved before their outcomes reach training.
Replay stores queries with their actual preceding support, including at switches.

## Fixed learner and arms

Use the existing single recurrent predictor from random initialization: frame
and temporal width64, context width12, outcome decoder width64, interaction
features, Adam .002, 12 updates per32 arrivals, reservoir16 causal packets,
gradient norm clip5. Keep weights, optimizer, history and replay throughout.
The recurrent context is recomputed from its finite observed window; no hidden
law state persists. All arms start with identical predictive weights per seed.

1. `outcome`: performed-action binary cross-entropy on current and replay packets.
2. `final_frame`: same, plus reconstruct the last observed frame repeated four
   times. This is the generic reconstruction control.
3. `sequence`: same, plus reconstruct the entire observed four-frame sequence.
4. `compute`: outcome-only learning with additional useful updates, allocated
   by the engineering measurement below before scientific main fitting.

Auxiliary arms use a width64 hidden decoder (Linear/Tanh/Linear/Sigmoid) from
the exact final query visual/temporal latent used by the outcome predictor.
There is no separate encoder, raw-input skip or cue-selected target. Targets
are observations/255; loss is the mean squared error over every frame/pixel,
then every query. Both current and replay query latents receive this loss.
The decoder is not called at inference. Baselines have no auxiliary decoder.
Predictive, decoder, optimizer and replay memory costs are reported separately.

The fixed objective is outcome BCE + **1.0** times reconstruction MSE. Both
packet losses are averaged equally. There is no alpha search, per-cue weighting,
teacher, frozen pathway, periodic reset or delayed model adoption. The auxiliaries
train from initialization through the initially cue-irrelevant prefix.
This tests one operating point, not an optimized auxiliary method.

## Engineering and compute control

Use disjoint engineering seed18091, deterministic CPU execution, one thread.
Measure full current+replay training updates after a short warm-up on an
engineering-only stream. Use five paired timing rounds, reversing arm order
between rounds, with fixed record count and no evaluation-score inspection.
Set compute updates per batch to ceil(12 * max(1, auxiliary/base median time
ratio, auxiliary/base analytic dense multiply-add ratio)). Record both auxiliary
timings and choose the larger. The analytic count includes frame linear,
temporal GRU matrices, context GRU matrices, predictive decoder and auxiliary
decoder; it is a declared work proxy, not an exact hardware FLOP measurement.
Provide the useful-compute baseline that entire integer allocation, including
rounding upward. Report actual training time and achieved work in the main run;
if resource dominance is not established, do not claim compute efficiency.

Engineering smoke uses reduced sizes/seeds and cannot qualify the hypothesis.
Do not inspect a main-arm behavioral curve to select a compute multiplier.
The multiplier is recorded in a fixed config before main fitting. Six workers
each use one thread with CPU affinity1365, leaving capacity for the desktop.

## Stage one: reference learnability, then a stop/go decision

Only the outcome baseline is fit, seeds18101--18106, covering all six cue orders.
Each seed has three independent fresh fits, one for each cumulative introduction
stage, with8192 arrivals each and no supplied initial support. This exposure is
chosen from previous acquisition evidence, not new candidate results.

A fourth independent fit interleaves five conditions: baseA, baseB, and the
three cumulative novelty laws. Cycle them in that order in1024-arrival blocks,
eight times, for40960 arrivals (8192 per condition). No resets or law labels
are given. This is blocked interleaving, not individual examples with unrelated
unidentifiable support. Retain switch costs in saved online forecasts.

At endpoints, use512 disjoint factorial queries and two independent32-record
support sets from each target condition. Flip each active query cue while
holding physical outcomes and support fixed. For each cue score the correct
and flipped predictions on the same affected subset. Train-only per-law
action/horizon marginals use (success+.5)/(count+1).

Qualification requires all of these seed-mean checks:

- Fresh fits: for EACH stage and EACH introduced cue grouping, all-case
  marginal-minus-model Brier >=.02 and affected flipped-minus-correct Brier >=.002.
  These are three stage groups and three cue-type groups, each with six seeds.
- Interleaving: for EACH of the five target-law slots, marginal improvement
  >=.02 and all-case Brier <=.12.
- Interleaving: for EACH (cumulative stage, active cue) cell, cue benefit >=.002.
  Across counterbalanced orders this gives nine cells, with two, four or six
  seeds where that cue is active at stage1, stage2 or stage3 respectively.

The last checks demand simultaneous availability of earlier cues and are
stricter than earlier fresh-only studies. They may expose context-identification
or optimization limits; failure does not establish representational erasure.
Every cell and seed remains reported. Qualification is a conjunction, not a
pooled mean. A failed cell stops the main comparison at this recipe. Do not
raise exposure, remove a cue or use an auxiliary arm to rescue qualification.

## Stage two: comparison only after qualification passes

Use fresh seeds18201--18206 and all four arms. Each arm learns independently
from initialization, with identical input records and reservoir membership.
Compute-arm replay draws may differ because it executes more updates.

The continuous schedule is four base blocks A/B/A/B,1024 arrivals each;
three cumulative dependency introductions,8192 each; exact baseA return,
newest-law return, revised directional-efficiency relation, and return to the
unrevised newest law,8192 each. All arms keep training through all changes.
Each seed receives one of the six cue orders, as in qualification. Feedback is
clean in this first comparison. Checkpoints/probes occur every1024 arrivals.

Save all-action forecasts on EVERY arriving example before its feedback, plus
evaluator truth, performed actions/outcomes and affected/valid masks. The primary
acquisition score averages affected-case Brier within each of the three new
dependency phases, then averages phases equally within seed. Equal phase weight
prevents the larger supply-timing affected subset dominating the endpoint.
Survival is a separate action-choice outcome; some cues change predictions
without changing the best action in this testbed.

The bounded progression screen requires:

1. Sequence-minus-outcome and sequence-minus-compute acquisition Brier <=-.002,
   and sequence-minus-final_frame <=-.001. Every contrast must favor sequence
   in at least5/6 paired seeds. Also no introduced-cue group may worsen >.005.
2. Whole-stream Brier may worsen <=.005 and survival may fall <=.01 versus
   each comparator, including the prefix and all switches. The same guards
   apply separately to every return/revision phase.
3. Correct-support valid-old Brier after each novelty/return/revision phase
   and in each base mode must be <=.12 and worsen <=.005 versus each comparator.
   Exact baseA return endpoint all-case Brier must also be <=.12.
4. Every active-cue cell at novelty and newest/revised/unrevised return endpoints
   must retain seed-mean affected cue benefit >=.002. A gain in reconstruction
   quality alone never qualifies.

Report every arm, seed, phase, cue and guard. Bootstrap20,000 paired seed
resamples (analysis seed28192026) for descriptive95% intervals; phases, horizons,
actions and individual examples are not independent replicates. Six seeds give
a small development screen, not confirmatory statistical noninferiority or a
general solution. All thresholds are raw Brier units, not accuracy percentages.

If qualified but comparison fails, stop this simple reconstruction recipe.
If it passes, consider a separately specified soft-consolidation test. Neither
step authorizes automatic alpha/width/replay sweeps. Cumulative gains do not
establish compounding; that requires matched-condition evidence of increasing
learning efficiency over a longer horizon and eventually a new domain.

## Reproducibility and interpretation

Lock config, protocol, implementation and decision analysis before scientific
fitting. Archive sources, timestamps, runtime, every failed attempt and raw
forecasts. Use atomic phase checkpoints including model, optimizer, gradients,
replay membership/sampling RNGs, history and work counters. Resume only under
identical locks, verify paired records and independently rescore raw arrays.
No evaluation may update parameters, history, replay or global RNG state.
Record total arrivals, updates, model/optimizer/replay bytes, wall time and work
proxy; auxiliary computation and qualification overhead are not free.
