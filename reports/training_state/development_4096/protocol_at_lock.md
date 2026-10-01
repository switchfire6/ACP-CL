# Optimizer history versus replay history

Status: prospective protocol, before development or comparison training.

The previous acquisition diagnostic combined optimizer and replay reset. Both
new learning and valid-old damage increased. Its conditional fresh-model delay
check failed (.00193868 < .002); that result stays failed. This study separates
the two carried states after an independent learnability check. It tests no
new architecture, sleep mechanism, selective pruning, or autonomous reset rule.

## Controlled interventions

Keep the prior conserving physical world, temporal images, architectures,
learning rate .002, 12 updates per 32-record batch, 32-record recent history,
16-packet replay capacity, and evaluation protocol unchanged. Four original
blocks A,B,A,B each contain 1,024 arrivals. The three previously irrelevant cues
then become relevant to efficiency, delay, and supply timing in all six orders.

At each new dependency, fork identical learned parameters and recent history:

| Arm | Adam state | Replay state |
|---|---|---|
| continue | Keep | Keep |
| optimizer_reset | Reset | Keep |
| replay_reset | Keep | Reset |
| state_reset | Reset | Reset |

Optimizer reset creates fresh Adam at the same learning rate, clearing both
moment accumulators and step counters. It preserves weights, trainability,
history, and complete replay state including its random generators. Replay
reset empties stored packets and resets the reservoir count and random
generators to the original seed; it preserves all optimizer state. It is a
reset of the whole replay process, not an isolated test of obsolete examples
or a permanent removal of replay. The cleared buffer refills from new data.
When empty, the existing update loop duplicates the current packet, keeping
forward/update budgets equal. Cost records distinguish actual replay from
duplicated current presentations.

A fifth fresh-weight reference has the original random parameters, fresh Adam
and replay, and the same recent history. It qualifies current task learnability
and distinguishes reusable parameters from carried training state. Only the
ordinary continuing arm advances to the next introduction. Reset timing is
privileged diagnostic information; ordinary training receives no law labels,
change markers, or counterfactual outcomes.

## Final branches and matched noise control

After three additions, clone the same continuing learner into every factorial
arm for each final branch: exact return to the initial A-world, reversal of the
efficiency cue, unchanged clean continuation, and unchanged physics with 20%
report replacement. Fresh weights are not repeated in these final branches.

The inherited final-channel seed makes clean/noisy observations and performed
actions identical. Their report labels differ only through corruption. This
adds the matched clean continuation missing from the earlier study. Both
branches start from identical complete state within an arm. All branches use
the selected common exposure and update budget. They do not advance each other.

Live-history curves include noisy support when reports are noisy. The ordinary
before/after fresh-support probes also use the branch's feedback distribution:
noisy support for noise, clean support otherwise. Valid-old retention probes
always use correct support drawn from the physically unchanged subset, identically
before/after and across matched noise/clean branches. This evaluator-owned
selection isolates parameter retention and is not an ability of the learner.

## Development and locked comparison

Fresh-only development uses new seeds 411--416, all six cue orders. Try 4,096
then 8,192 arrivals per episode; stop at the first qualifying exposure. Both
architectures must, for EACH stage and EACH cue group, achieve terminal all-case
Brier improvement >= .02 over the training-only action/horizon marginal model
and correct-cue benefit >= .0025 on the affected subset. The .0025 development
margin is deliberately stricter than the unchanged .002 comparison threshold
because the preceding development margin was fragile. Preserve every attempt.
Do not examine factorial outcomes or comparison seeds to choose this exposure.
If neither budget passes, stop without launching the comparative cohort.

At the selected exposure, use new seeds 8001--8006 and both architectures:
12 prefix fits, 180 new-dependency episodes (three stages, five arms), and 192
final episodes (four branches, four arms): **372 episodes**. All six orders
occur once; mode by order is not fully crossed. Recheck fresh-model adequacy
at .02 marginal improvement and .002 cue benefit. Report any incomplete
qualification; do not tune or rerun failed seeds with altered settings.

## Endpoints and factorial estimates

Primary acquisition: normalized trapezoidal Brier-error AUC on the newly
affected subset, evaluated at entry and every 256 arrivals on 512 independent
query trials. Lower is better. Three outcome horizons and five hypothetical
actions are used only in evaluation. Training observes one performed action.
Terminal query-cue flips, all-case error, survival, representation change, and
valid-old error are secondary diagnostics. Two 32-record fresh-support sets
are used for initial/final probes. The physical feasibility audit and endpoint
choice are inherited unchanged: better forecasting need not change optimal
one-time actions, particularly for efficiency and delay.

For metric L, write C=continue, O=optimizer_reset, R=replay_reset, B=state_reset.
Within each seed, average the three acquisition episodes before estimating:

    Optimizer effect with replay kept = O - C
    Optimizer effect with replay reset = B - R
    Replay effect with optimizer kept = R - C
    Replay effect with optimizer reset = B - O
    Optimizer main effect = ((O-C) + (B-R)) / 2
    Replay main effect = ((R-C) + (B-O)) / 2
    Interaction = B - O - R + C

Also report B-C, C-fresh, and B-fresh. Use the same contrasts for valid-old
damage (after minus before), cue use, and survival. For Brier error/damage,
negative favors the reset named by the contrast. For each final branch, report
the four simple effects, both main effects and interaction separately, plus
noise-minus-clean differences within each arm. Report terminal valid-old
levels as well as changes. Fresh-model change from random weights is not a
forgetting control.

Use 20,000 paired seed bootstrap resamples, analysis seed 27192026, with every
seed shown. Intervals are descriptive; no multiple-testing-adjusted significance
claim or post-hoc mechanism-selection threshold is made. Episodes, prediction
horizons, and actions are not independent samples. An effect that improves
acquisition but worsens retention has not met the combined research objective.
Factorial effects identify these specific reset operations in this testbed;
they do not identify a general biological mechanism or an age effect.

## Integrity and next decision

Freeze source/config/runtime/protocol before each learner cohort. Retain exact
source, all attempted configs, all initial/final probes, trajectories and atomic
checkpoints. Audit optimizer/replay isolation, matched records and budgets,
causal replay regeneration, and initial/final probe recomputation from local
checkpoints. Preserve previous source and report archives.

If optimizer effects explain the advantage without comparable retention damage,
investigate a bounded recalibration of optimizer history. If replay effects
dominate and create retention costs, investigate what to retain and how to
condition its use. A large interaction motivates studying their combination.
None of these patterns automatically validates a deployable reset, a sleep
analogy, or selective pruning. Any next intervention needs its own declared
acquisition/retention requirements and fresh evaluation.
