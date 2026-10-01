# Learned core and adaptive raw-input pathway: development protocol

Specified 2026-09-27 local time, before fitting scientific seeds 16001--16006.
The previous state-substitution and predictive-value decisions remain unchanged.
This is one bounded architectural comparison on actual continuing trajectories,
not another cached-memory-score or optimizer-rollback experiment.

## Hypothesis and architecture

A learned reusable predictor plus a small trainable feature pathway may acquire
a previously irrelevant dependency while reducing damage to useful older
predictions, compared with training the identical-capacity architecture jointly.
No prior checkpoint supplies externally pretrained features: every core learns
its shared prefix from random initialization within this experiment.

Train the existing conditional or recurrent core at width 64 on four alternating
base-law blocks of 1,024 arrivals. At the fixed cutoff attach one residual
pathway with width 32, context width 8 and decoder width 32. It receives the same
raw image sequences and causally available support as the core. Conditional
residuals have the same four expert slots; recurrent residuals have independent
raw-image and history encoders. Final residual projections start exactly zero,
so inserting the pathway initially preserves predictions. Feature gradients
open after the output projections become nonzero; no extra warm-up is used.

For the conditional architecture, corresponding core and residual expert logits
are added before evidence routing and probability mixing. For recurrence, the
two context-conditioned output logits are added before the sigmoid. The usual
monotone survival projection is used at prediction time. Freezing core weights
does not guarantee unchanged combined predictions or unchanged inferred context.

All three arms start from independently copied, identical learned core weights,
complete Adam state, replay contents/generators and recent history, with exactly
the same newly initialized residual. Adam state is mapped by parameter identity
without aliasing; new residual parameters start with empty Adam state. Preserve
the core optimizer rather than resetting it as an extra intervention.

| Arm | Core | Residual visual frame/temporal encoder | Residual heads/context/readout |
|---|---|---|---|
| joint | Trainable | Trainable | Trainable |
| separate, sole primary candidate | Frozen | Trainable | Trainable |
| fixed_features, attribution control | Frozen | Fixed at random initialization | Trainable |

The last arm is a fixed-random-feature residual, not a remixer of learned core
embeddings. It preserves raw input access and capacity while isolating adaptation
of the new visual encoder. A residual restricted to a lossy frozen core embedding
could not recover missing inputs; that restriction is not imposed here.

This capacity is allocated once. There is no subsequent growth, consolidation,
reset, teacher-model retention or score-guided memory selection. Frozen core
parameters stay in evaluation mode and keep their original Adam moments/counters.
All learnable parameters use Adam lr .002, norm-5 clipping and the unchanged
average of current/replayed performed-action BCE. Each batch has 32 arrivals,
with 12 optimizer updates and one sampled historical causal packet per update.
The uniform reservoir holds 16 packets, including their original past-only
support. All ordinary training feedback is clean in this first comparison.

## Continuous schedule and cohort

Use six unused seeds 16001--16006 with both architectures. The first new cue is
balanced over the three physical dependencies and two initial base modes.
Each of the 12 shared prefixes forks the three arms once, before these phases:

1. Maintenance: 1,024 arrivals under the final prefix law.
2. Novel learning: 8,192 arrivals with the first new dependency activated.
3. Return: 2,048 arrivals under the exact original base law.

The final prefix and maintenance use mode 1 - seed%2, the same mode as novel
learning. Thus activation changes a dependency without simultaneously changing
that base mode. The novelty cutoff is separated from the architecture cutoff.
There is no reset, pathway insertion, law label or boundary signal delivered to
the learner at novelty or return. Each arm carries its own weights/optimizer
continuously through all three phases, while replay and observable histories
are identical across arms. The fixed intervention time is an experimental
allocation, not a learned autonomous capacity-management mechanism.

A separate fresh joint-architecture reference learns the same 8,192 novel
arrivals from random core/residual weights, empty replay and Adam, and the
identical actual history immediately preceding novelty. It qualifies learnability
only; it has no learned prefix whose forgetting can be compared. It receives
no privileged target-law support during training.

The main cohort has 12 prefix fits, 36 continuous learned trajectories, 12 fresh
references, 120 post-prefix episodes, and 168 phase records including prefix
blocks. It consumes 552,960 training arrivals, 17,280 packets and 207,360 Adam
updates. A disjoint small smoke configuration tests engineering only and cannot
select scientific widths, learning rates, exposure or thresholds.

## Evaluation and fixed decisions

Use 512 fixed evaluator-owned query cases, evaluate every 256 arrivals and use
two independent support replicates for diagnostic probes. Ordinary pre-update
performed-action predictions are recorded before training each incoming packet.
Held-out curves use the learner's actual preceding history. Counterfactual
five-action physical truth and law labels belong only to the evaluator.

The primary acquisition metric is affected-subset all-action Brier integrated
over the complete novel-learning curve, including its starting point. Return
uses all-case Brier AUC. At novel end, measure valid-old Brier separately in
both base modes using the existing evaluation-only correct-support probes.
At return end, also probe the just-learned novel law and its flipped cue, holding
weights fixed and supplying only the declared diagnostic support. This checks
availability of newer knowledge without crediting privileged context as online
condition identification. Report actual-history and refreshed-support results
separately. Maintenance and every arm remain visible in the analysis.

Fresh qualification requires terminal all-case Brier improvement >= .02 over
the training-only action/horizon marginal predictor and correct-cue affected
Brier benefit >= .002, both overall and in every cue group, per architecture.
Two seeds per cue give only a limited development check. Failure remains visible
and cannot trigger added exposure, replacement seeds or a parameter sweep.

The sole primary candidate is separate versus joint, separately per architecture.
Continuation requires a complete eligible cohort, fresh qualification and all:

- Novel affected Brier AUC difference <= -.002, with improvement in >= 5/6 seeds.
- Novel-end valid-old Brier cost <= +.005 in each base mode separately.
- Return all-case Brier AUC cost <= +.005.
- Survival AUC loss <= .01 in novel learning and return separately.
- Post-return correct-support novel-law affected Brier cost <= +.005.

These inherit the prior acquisition/retention tolerances; they are practical
allocation screens, not significance or noninferiority tests. The post-return
guard is specified now to prevent preservation of older conditions from hiding
loss of the newly learned dependency. It is not added after observing outcomes.

Feature-adaptation attribution is separate: separate minus fixed_features novel
Brier AUC <= -.001, improving in >= 5/6 seeds. An outcome pass without this
attribution does not establish that learning new visual features caused the
gain. Attribution cannot rescue a failed primary screen or promote an ablation.

Use paired seed differences, 20,000 bootstrap resamples and seed 27192026.
Six independent seeds per architecture are the units; query cases, actions,
horizons, support replicates and curve points are not independent replications.
Report every seed, cue group, mean and descriptive 95% interval, including
intervals crossing the practical limits. Include cue use, encoder displacement,
initial/terminal errors, all comparisons and work counters descriptively.

## Resources, locks and audit

Match total architecture capacity, initial predictions, observations, replay
membership/draws, current/replay loss weights, forward exposure and update counts
across the three learned arms. Freezing reduces backward work: these are not
equal-FLOP interventions. Count total/trainable parameters, active/retained
optimizer state, explicit tensor storage, prediction work and wall time. Do not
spend frozen-path savings on additional updates or claim a speed advantage.
Use six single-thread CPU workers with host affinity 1365.

Lock scientific source, config, runtime, protocol, scientific analysis and
relevant tests before main fitting. Complete all trajectories before examining
scientific comparisons. Save atomic resumable checkpoints including gradients,
complete optimizer/replay/RNG state and evaluator records. Preserve raw numerical
probabilities, outcomes and probe provenance; export portable JSON/NPZ evidence.
The independent portable archive verifier and its tests are packaging work,
explicitly excluded from the scientific lock and sealed separately.

Check zero-residual insertion, independent ownership, complete Adam transfer,
frozen parameter/moment invariance, residual feature learning, causal support,
unchanged replay streams, state-preserving evaluation and exact interrupted
resumption. The local auditor regenerates streams, replay and evaluation physics,
reconstructs initial forks and endpoint forecasts, and checks checkpoint chains.
It does not independently retrain every ordinary learning update. Portable
verification checks numerical arithmetic, provenance and audit coverage without
loading model pickles or claiming an independent training replication.

Preserve every earlier scientific file and archive. The prior 1,595 files,
including 1,313 report artifacts, have a pre-edit hash inventory. No result here
establishes indefinite acquisition of new core representations, biological
critical periods, autonomous novelty detection, noise robustness, a longer
physical planning horizon, or a general continual-learning solution. A positive
development result needs a separately locked independent follow-up.
