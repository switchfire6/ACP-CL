# Conditional relationship reuse: prospective pilot protocol

## The narrower hypothesis

Recent observed action/outcome relationships can identify which previously
learned consequence mapping currently applies. A learner may then reuse that
mapping without changing its weights. This follows the negative experience-
selection pilot; it does not modify or replace its result.

The candidate is a bounded mixture of learned predictors with evidence-based
routing and uniform replay of causal support/query packets. It is an ordinary
neural implementation of this hypothesis, not a claim to a new general learning
algorithm, symbolic relationship discovery, evolutionary inheritance, or an
accurate model of critical periods. All visual features and prediction heads
start randomly. No physical rule, task classifier, or pretrained representation
is supplied to the ordinary learner.

## World and removal of the earlier shortcut

Reuse the earlier conserving transfer physics unchanged: two participants,
finite reserves, one transfer, independent noisy supplies, a mid-trial exchange
of supply advantage, transport loss/delay, and irreversible failure through
12 steps. The declared objective remains BOTH participants' survival, not a
derived universal definition of improvement.

Every 32-trial arrival batch contains all eight combinations of visual motion
direction, loss, and delay, four times each, in random order. Direction is
therefore balanced conditional on every static glyph combination. The same
current frame can still hide different earlier motion. There is no withheld
factorial composition claim in this study: all eight combinations are trained.

An unobserved mode either preserves or reverses the association between motion
and supplies. Supply-column swapping implements the reversal without altering
observations. Thus opposite modes can have EXACTLY identical image/action inputs
and different feedback. Each trial still provides only the performed randomized
action's survival at horizons 4, 8, 12. Alternative actions exist only in probes.

Three streams have twelve 1,024-trial blocks. With p = seed mod 2:

* Recurring: p, 1-p, p, 1-p, p, 1-p, 1-p, 1-p, p, p, 1-p, p.
  The final ungated return is block index 8, after three intervening blocks.
* Stable: p throughout. Recent feedback need not select between modes.
* Unpredictable: an independent fair mode draw for every trial. Recent outcomes
  do not reveal the next trial's mode. This is a negative control for recurrence.

In the last three blocks of every stream, an additional visible gate varies
between zero and one. Gate one reverses the motion/supply mapping again.
Previously it was always zero. This tests learning a late discrimination while
retaining earlier mappings. It does not prove that a specific new feature was
created, or establish general compositional learning.

No ordinary learner receives block indices, boundaries, mode identities, future
schedules, or evaluation feedback. Batching is at a fixed, unannounced cadence;
there is no resetting of learning state at changes. The true modes in the world
and raw diagnostic archives remain evaluator metadata.

## Mechanism and time

Let x be a four-frame raw observation and a the performed action. A shared
64-wide frame MLP and within-trial GRU produce learned features. Four separately
initialized heads predict action-dependent survival. These four maps are a
fixed capacity assumption, not a discovered number of physical relations.

For a past-only support window S_t of the previous 32 records, define

    L_k(S_t) = sum_(x,a,y in S_t) log Bernoulli(y_12; sigmoid(f_k(x)_(a,12)))
    w_k(S_t) = softmax_k(L_k(S_t))
    P(y_h=1 | x,a,S_t) = sum_k w_k(S_t) sigmoid(f_k(x)_(a,h))

The prior is uniform and the evidence strength is 1. Only terminal survival
enters the context likelihood: nested survival horizons are not incorrectly
multiplied as independent observations. The learned predictions need not be
calibrated, so these weights are not guaranteed calibrated beliefs about the
world. No sticky prior or optimized hazard rate is supplied.

Training minimizes binary cross entropy for the actually observed action at
all three horizons, including gradients through the support-based weights.
Inference projects the mixture's horizon predictions to nonincreasing survival
using a cumulative minimum, then selects the action with maximum H=12 survival.
Only support that preceded the query enters its route, including during replay.

This distinguishes three time scales: the four frames describing one event;
the previous 32 events informing current applicability; and parameters/replay
retaining mappings over thousands of arrivals. Early adaptation can change
w without changing f. No indefinite retention or optimal time constant is assumed.

Each update uses one current packet and one uniformly sampled old packet.
A packet stores 32 support records plus 32 query records, with missing support
only at the stream start. A uniform reservoir stores 16 packets: at most 1,024
raw record copies, plus the live 32-record window. Adjacent packets can duplicate
records; byte accounting includes those copies. Three Adam steps, learning rate
0.002 and gradient clipping at 5, are used per arrival batch. No replay selection
heuristic from the earlier failed pilot is reused.

## Comparators and limitations of matching

All six methods allocate the same randomly initialized network per seed:

| Method | Intervention |
|---|---|
| pooled | Uniform head weights, ignoring support when predicting |
| conditional | Likelihood weights from correctly bound past image/action/outcome records |
| shuffled | Same likelihood rule, but cyclically permute support images relative to action/outcome pairs during training and evaluation |
| current_only | Correct conditional routing; duplicate the current packet in place of replay |
| frozen_features | Conditional replay; freeze frame encoder and GRU after four blocks |
| oracle | Privileged diagnostic: true query mode selects head 0 or 1 |

The frozen diagnostic receives a known boundary solely for the intervention.
The oracle receives true mode identities even in the unpredictable stream;
it is not a deployable competitor. Its other heads are unused. Ordinary methods
are prohibited from receiving oracle modes. The all-head network itself is
identical, but active gradient paths and useful capacity differ between controls.
Equal allocations, forward shapes, query exposures, and optimizer steps do not
constitute equal effective capacity or exact FLOPs. Unused support forwards are
computed by pooled/oracle controls. Current-only has less replay storage.

This is a comparison against pooling and binding ablations, not against all
strong history-conditioned recurrent/meta-learning baselines. A positive result
would support this small mechanism, not establish its superiority or novelty.

## Primary inference-only test

Immediately before block index 8, freeze a copy of the learned weights. Replace
the stale live history progressively with 0, 8, 16, then 32 fresh randomized-action
outcomes from the returning condition, with NO gradient update. Evaluate on
384 separate, balanced query trials, repeating the support experiment four
times with independent support seeds. Average the four scores within each seed.
Support never contains query trials or their labels, and these diagnostic
records never enter subsequent training or its live history.

Primary contrast: conditional minus pooled joint survival after 32 feedback
records on the recurring stream, using the pre-return weights. This is NOT
zero-shot mode recognition: identical observations cannot identify the new
mapping before any differentiating evidence. Score at zero feedback is reported
but not expected to favor the candidate.

With the SAME frozen weights/query trials, also use opposite-mode support with
identical images/actions, erase support, or break its image/outcome binding.
This probes whether the learned action choices depend on the meaning of recent
feedback, rather than just extra parameters or a new weight update.

Secondary results: actual online curves; final performance on both modes with
and without the late gate, after correct support; early late-gate learning AUC;
current-only and frozen-feature differences; stable and unpredictable controls;
head disagreement/usage, Brier error, storage and measured runtime. Reusing
the same query set improves paired comparisons but does not add independent
seeds. Final probes in modes never trained in stable runs are OOD diagnostics,
not valid known-mode retention. Unpredictable-stream panel probes impose a
persistent mode and must not replace that stream's actual mixed-mode score.

## Development, lock, and decision

Development uses seed 117 and pooled/oracle only. Engineering smoke uses seed
19 and deliberately inadequate training. First check physics, causal interfaces,
resume, and whether the oracle can acquire a useful mode-dependent policy.
The initial development budget is 1,024 trials/block; only a documented common
exposure-budget increase is allowed before lock if oracle acquisition fails.
Do not tune the candidate on comparative pilot outcomes.

The pilot uses eight consecutive seeds 3001 through 3008, balancing initial
modes, six methods and three schedules: 144 runs. Save this protocol, all configs,
runtime identity and training source before launching. Training source is
immutable throughout the comparison. Do not drop seeds or select endpoints.

Adequacy on the recurring primary probe requires the privileged oracle's mean
survival to exceed no transfer by at least 5 percentage points and pooled by at
least 2 points. If not, label the architecture comparison inconclusive.

The practical continuation screen requires all of:

1. Conditional minus pooled primary gain at least 2 percentage points, positive
   in at least six of eight seeds.
2. Conditional minus shuffled primary gain at least 2 points.
3. Correct minus opposite-history survival with the SAME conditional weights
   at least 2 points, positive in at least six of eight seeds.
4. No mean loss worse than 2 points relative to pooled on the unpredictable
   stream's inference-only mixed-mode probe, or on the recurring stream's final
   gate-conditional performance averaged over its two learned modes.

These are practical pilot thresholds, not significance tests. Report paired
seed bootstrap 95% intervals (20,000 resamples, fixed analysis RNG), all seed
differences, and every secondary control without multiplicity-adjusted claims.
A failed screen stays failed. Strong oracle learning with no conditional gain
would localize failure to this learned mixture/routing recipe in this world;
it would not refute every method of contextual inference.

## Related primary research

The distinction between learning a mapping and inferring its current applicability
has existing research precedents. [Caccia et al.](https://proceedings.mlr.press/v232/caccia23a.html)
study recurrent replay for task-agnostic continual reinforcement learning. This
pilot uses simpler randomized-action prediction and is not their algorithm.
It should be evaluated as a test of the present hypothesis, without a novelty claim.

## Running

```powershell
.venv/Scripts/python.exe -m acp_cl.conditional.study --config configs/conditional_smoke.json --output runs/conditional_smoke --device cpu
.venv/Scripts/python.exe -m acp_cl.conditional.study --config configs/conditional_development.json --output runs/conditional_development --device cpu
.venv/Scripts/python.exe -m acp_cl.conditional.study --config configs/conditional_pilot.json --output runs/conditional_pilot --device cpu --protocol docs/conditional_relationships.md
```

For unchanged interrupted runs, add --resume. Resume rejects differing source,
configurations, and runtime identities. Checkpoints are written after each block
and load only this project's trusted local pickle files. Interrupted partial
blocks can be recomputed and are not counted in the completed training budget.
The pilot uses four independent CPU worker processes with one PyTorch thread
each; concurrent runtimes are not hardware-isolated performance benchmarks.
