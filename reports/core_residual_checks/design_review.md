# Prospective review: reusable core and adaptive raw-input residual

Prepared 2026-09-28 before fitting this experiment. This review supports the
fixed development design agreed with the implementation lead; the forthcoming
locked protocol/configuration will govern execution. It uses the earlier
[acquisition diagnostic](../../docs/acquisition_diagnostic.md),
[qualified exposure](../../docs/training_state_diagnostic.md), and
[selective-update criteria](../../docs/selective_updates_protocol.md).
No setting is selected from the recent rehearsal-state outcomes.

## Smallest interpretable comparison

Use six unused seeds 16001--16006 and both conditional/recurrent architectures.
Each first learns the unchanged four alternating base-law blocks of 1,024
arrivals. Copy that complete learned core, Adam state, uniform reservoir and its
generators, and causal history into three arms. Add the identical raw-input
residual to each: visual width 32, context width 8, decoder width 32, zero output
at initialization. Only the declared trainability differs:

| Arm | Mature core | New residual visual features | Residual context/readout |
|---|---|---|---|
| separate | Frozen | Trainable | Trainable |
| joint | Trainable | Trainable | Trainable |
| fixed_features | Frozen | Frozen at random initialization | Trainable |

The joint arm has the same total architecture and parameter capacity. The
fixed-features control isolates adapting the newly added raw visual encoder,
while retaining its input and capacity. It is a fixed **random-feature** control,
not a remixer restricted to learned core embeddings. Existing random features
can already contain useful information, so success of that control would not
show that no representations were ever learned. Conversely, a separate-arm
advantage over it supports the specified encoder adaptation, not a unique
intrinsic role for individual neurons.

Preserve old Adam state when registering new residual parameters; do not reset
core moments in the joint arm. All arms receive the same residual initialization
and predictions at the fork. The frozen complete core includes its learned
history/prediction machinery; evidence routing can still change with causal
feedback without changing those parameters. The residual must receive raw
observations and the same past-only support, so it can represent a newly useful
cue absent from the core's learned embedding.

## Continuous trajectory and qualification

Each learned arm advances its own state through 1,024 maintenance arrivals in
the final prefix law, 8,192 arrivals making its first cue relevant, and 2,048
arrivals returning to the exact original base law. No arm is replaced by another
arm's checkpoint at a phase boundary. No additional freeze, reset, allocation,
or change signal occurs at novelty or return. Uniform capacity 16, Adam .002,
12 updates per packet of 32, and the 32-record causal history stay fixed.
Evaluate every 256 arrivals on 512 query cases, with two independent support
replicates for endpoint probes.

This separates the intervention cutoff from the physical novelty by 1,024
arrivals. The fixed age-based cutoff remains an experimental design choice,
not demonstrated autonomous maturation. The six seeds balance three first cues
against both modes; they do not test all sequential introduction orders or
lifelong compounding. Performed actions remain exogenous random actions, so
continuous training does not mean an endogenous action-policy experiment.

A fresh joint-architecture reference learns only the 8,192 novel arrivals,
starting from random core/residual weights, empty replay and Adam state, and
the identical preceding actual maintenance history. It qualifies task
learnability; it has no learned history whose forgetting can be compared.
Require terminal all-case gain of at least .02 against the training-only
action/horizon marginal predictor and correct-cue benefit of at least .002,
both overall and separately for each cue within each architecture. Two seeds
per cue give a weak qualification check. Complete every planned run and retain
any failure; do not extend exposure or replace seeds after inspecting outcomes.

## Fixed outcome and attribution decisions

The primary candidate is separate, compared with joint, per architecture.
Use the existing affected-subset all-action Brier AUC over novel learning.
Continuation requires qualification, mean acquisition difference <= -.002,
and improvement in at least five of six paired seeds. Preserve the inherited
tradeoff limits: terminal novel-phase valid-old Brier cost <= +.005 in each
base mode separately; return all-case Brier AUC cost <= +.005; and survival AUC
loss no greater than .01 in novel learning and return separately.

Also require the post-return correct-support novel-law affected-subset Brier
cost to be <= +.005. This inexpensive frozen-weight probe checks that returning
to old conditions did not sacrifice the newly acquired relationship. It supplies
evaluation-only correct context; it is not credited as the learner's ability
to identify the current law. Report actual-history return behavior and initial
versus terminal errors alongside these probes. Maintenance is reported in full.

Assess new-feature attribution separately: separate minus fixed_features novel
acquisition AUC <= -.001, improving in at least five of six seeds. Its failure
cannot rescue a failed primary screen; an outcome pass without attribution
does not establish that adapting the new visual features produced the gain.
Report cue use, feature displacement, all comparisons and all seeds without
promoting a favorable ablation after the fact.

Use the inherited 20,000 paired seed bootstrap resamples with seed 27192026.
The six seeds are the independent units; curves, actions, horizons and support
replicates are not extra replicates. Mean-based development screens are
allocation rules, not significance or noninferiority tests. Display uncertainty
when intervals cross a practical threshold.

## Resources and claim boundary

Match observations, replay membership/draws, current/replay loss weights,
forward exposure, and update counts. Log total and trainable parameters, retained
and active optimizer state, raw-memory bytes, and actual work. Freezing reduces
backward work, so equal updates and capacity are not equal FLOPs. Do not spend
those savings on extra updates in this comparison. The fixed capacity is added
once at the declared cutoff; no unlimited growth or uncounted teacher is allowed.

Audit exact forks and optimizer ownership, zero-residual initial predictions,
frozen state hashes, residual learning, stream/replay equivalence, causal support,
resumption, and initial/final probes. Evaluation must preserve complete learner
state, including flags and RNG. Lock source, configuration, protocol and analysis
before main fitting, and preserve every prior archive.

A successful result would establish a bounded acquisition/retention tradeoff
for this architecture in this world. It would not establish continuing renewal
of core representations, automatic capacity management, noise discrimination,
biological critical periods, or a general continual-learning solution. Those
remain later questions; they should not expand this first comparison.
