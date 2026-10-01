# Continued acquisition, valid retention, and obsolete relationships

Status: prospective diagnostic protocol, written before learner development.
This tests the accepted hypothesis: acquire new predictive relationships,
preserve relationships that remain useful, and revise obsolete ones without
exhausting the ability to learn. It introduces no new learning algorithm.

## Why this diagnostic and this endpoint

Keep the qualified original conditional and recurrent architectures from the
matched-history pilot, including their existing 32-record history and 16-packet
replay. All representations start from random weights. The prior feature-freeze
effect concerned the recurrent model, not a proven conditional-model bottleneck.
The earlier gate combined old mappings; this diagnostic changes physical laws.

Three independently sampled temporal pulse-order signals are visible in every
trial from the beginning. Their final frames are identical across signal values.
They are initially irrelevant. In a counterbalanced order, make them govern:

1. Direction-dependent transport efficiency in the lossy channel (.95 versus .10).
2. Arrival delay in the delayed channel (1 versus 4 steps).
3. Supply crossover timing outside the clean/immediate channel (3 versus 9 steps).

Previously activated dependencies persist. The original conserving simulator,
four-frame image shape, five actions, and three outcome horizons remain. Every
training record gives only the outcome of one random performed action. Hidden
law metadata, phase boundaries, and counterfactual outcomes never enter ordinary
learner updates. Clean/immediate-channel physics stays unchanged throughout.

Before model training, a structural audit found zero clairvoyant action-choice
value for the first two signals when introduced alone, despite changed survival
outcomes. They can change the chance of success without changing the best
one-time transfer. Thus **prediction Brier error** is the acquisition endpoint,
with action survival reported separately. This endpoint choice precedes learner
development or candidate results. It does not claim that better forecasting
must improve decisions in this particular action set.

## Trajectory and counterfactual training controls

For each seed and architecture, learn four original-law blocks A,B,A,B. A is
seed modulo two; B is the opposite. Each prefix block has 1,024 arrivals.
Then introduce three dependencies successively, keeping base mode B fixed.
Each new-dependency episode has the common exposure selected during development.
All models use 12 updates per incoming batch, Adam .002, width 64, four original
conditional heads, and the qualified recurrent widths 12/64 with interactions.

At each onset clone the identical accumulated state into four arms:

* `continue`: ordinary learning with existing weights, optimizer, replay, history.
* `frozen`: freeze only visual frame/temporal encoders at onset; context/heads
  continue learning with the same optimizer and replay.
* `state_reset`: retain weights and recent history, reset optimizer and replay.
* `fresh`: restore the seed's original random initialization, with fresh
  optimizer/replay and the same recent history as the other arms.

All arms receive exactly the same new performed-action records and update count.
Only `continue` supplies the next episode's starting state. Reset/freeze timing
is a privileged diagnostic intervention, not an autonomous learning mechanism.
The `state_reset` versus `fresh` comparison holds optimizer, replay sampling,
recent history, new data, and training budget equal while changing prior learned
parameters. The `continue` versus `state_reset` contrast jointly changes replay
and optimizer; it cannot distinguish those two explanations by itself.
Fresh models have no earlier trained weights or replay; their common recent
history is explicitly shared to match available immediate context.

After all three additions, clone `continue` into three final branches:

* `return`: the exact initial A-world, with all added laws inactive again.
* `revision`: keep the newest world but reverse the efficiency signal's
  directional meaning. Old labels in the lossy channel are now obsolete.
* `noise`: keep the newest physics, but independently replace 20% of reports
  with a uniformly sampled valid nested survival curve. Some reports stay equal.

Return is an exact previously experienced world. It is not an unseen combination
of A with the new laws. Retention scores exclude obsolete relationships. During
additions the valid query subset is the clean/immediate channel; during revision
it is the entire clean channel. Both old base modes are probed with fresh,
correct past support under the current physical law.

## Measurements and interpretation

Evaluate at entry and every 256 arrivals using a disjoint full-factorial panel.
Primary acquisition measure: normalized trapezoidal Brier-error AUC on the
subset affected by the newly introduced dependency. Brier averages the three
horizons and five hypothetical actions in evaluation only; training still uses
one performed action. Lower is better. Report all-case Brier and survival too.
Correlated horizons/actions are not independent statistical observations.

At episode start/end, evaluate with independent fresh support sets, frozen
weights, and separate query trials. Flip only the newly relevant query signal
while holding physics, queries, and support fixed. Positive flipped-minus-correct
Brier indicates predictive use of that cue; it is not a policy improvement claim.
Final probes cover valid old subsets in both base modes and the current law.
Valid-retention support is drawn only from that same physically unchanged subset,
using identical clean support records before/after a phase. This evaluator-owned
selection isolates valid knowledge from changed/noisy support; it is not an
autonomous data-selection ability credited to the learner. Evaluation uses 512
query trials and two independent 32-record support sets.
Also report initial/terminal live-history curves and encoder displacement.

Within each architecture report every episode and paired seed contrasts for:
continue versus fresh; continue versus frozen; state_reset versus continue;
state_reset versus fresh. Average episodes within seed for an overall diagnostic,
then bootstrap seeds, not episodes or individual trials. Use 20,000 resamples,
analysis seed 27192026. Report late-minus-early differences relative to fresh,
but do not label them a pure causal effect of age: world complexity also changes.

If frozen features impair acquisition, updating the representation helps. If
state reset repairs a deficit, carried training state matters. If retained
weights after state reset still underperform fresh initialization, investigate
the learned parameter configuration. If learning new predictions worsens valid
old predictions, interference warrants investigation. These patterns motivate
future interventions; this diagnostic does not implement or establish a cure.

## Development and fresh-cohort qualification

Use only fresh-model fits on seeds 211,212 to choose exposure. In order, try
2,048, then 4,096, then 8,192 arrivals per new-dependency episode. Pick the first
budget at which BOTH architectures, grouped separately by EACH stage and by
EACH newly introduced dependency, meet across development seeds:

* terminal all-case Brier improvement >= .02 versus the action/horizon marginal
  predictor fitted only to that episode's performed-action training outcomes;
* flipped-minus-correct new-cue Brier >= .002 on the affected query subset.

All other settings are inherited or fixed here. No continuing-model advantage,
frozen-control result, or fresh pilot seed selects exposure. Keep every failed
setting and exact source. If none passes, do not launch the comparative cohort:
report that this diagnostic has not demonstrated adequate task learnability.

Use six new seeds 7001--7006, one for each of the six dependency orders, and both
architectures: 12 original prefix fits, 144 new-dependency arms, 36 final branches.
This is a diagnostic cohort, not a test of a new algorithm's superiority.
Recheck fresh-model cue use and marginal-predictor gains on that cohort before
attributing acquisition failures. Report incomplete qualification explicitly.

There are no task labels for ordinary learners, unbounded capacity, tuning on
the new cohort, or deletion of failed seeds. Save config/runtime/source hashes,
the exact protocol at lock, all outcomes, and atomic checkpoints. Recompute final
and starting probes and regenerate causal replay from trusted local checkpoints.
