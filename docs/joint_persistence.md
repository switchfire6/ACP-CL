# Joint persistence: a controlled test of experience retention

Completed: [the 288-run pilot](../reports/persistence_results.md) did not support
an advantage for the combined selection rule. Its original pre-run specification
is preserved in [the protocol archive](../reports/persistence/protocol_at_lock.md).

## Question and scope

Can a bounded memory policy based on observable recurrence and predicted
survival relevance improve retention and transfer without preventing new
learning? This is a new experimental line, independent of the archived ACP
and dual-path experiments. Negative outcomes remain part of the record.

The first implementation uses one randomly initialized recurrent image
predictor controlling a single transfer between two participants per trial.
All methods train on identical randomized-action records. Evaluation lets
each predictor select its own action. This isolates experience selection from
differences in exploration. It is a contextual, finite-horizon control study,
not a complete online reinforcement learner, an evolving population, or a
general solution to continual learning.

## Physical world and observations

Each participant starts with a reserve uniformly sampled from [1,13]. Storage
capacity is 18 and requested maintenance is 2 per step. The two mean supplies
are 2.7 and 1.1, with independent Gaussian noise of standard deviation 0.35,
clipped at zero. Their supply advantage swaps halfway through the trial.
Supplies vary locally. A trial lasts at most 12 steps. Mean aggregate input is
below maintenance, so finite starting reserves matter and no indefinite
survival is claimed.

At the start, choose no transfer or request 2 or 4 units in either direction.
A request is capped at the donor's available reserve. Transfer efficiency is
1 or 0.35. Departures occur before the first supply; transfers arrive before
step-one maintenance or before step-three maintenance. Overflow is discarded.
Actual maintenance consumption is capped at available resources. Reaching zero
after maintenance is irreversible failure. Every unit is tracked as stored,
in transit, consumed, lost in transport, or overflow. The physics continues
after failure for accounting only; later replenishment does not restore survival.

Input is four 12x16 grayscale frames: resource gauges, opaque glyphs for
transport conditions, and a moving environmental indicator. The indicator's
last position is identical for both supply directions, requiring its history.
The learner receives no regime identifiers, boundaries, action prescriptions,
latent supply parameters, or future outcomes beyond its actual action's feedback.
Each training record supplies three binary survival outcomes (at steps 4,8,12)
from the single randomized action performed. Alternative-action outcomes are
available only to the evaluator. This is richer feedback than a single terminal
reward, and the supplies are independent of actions.

The objective is the declared probability that BOTH participants remain
functional through step 12. Earlier horizons, individual resource accounting,
and restricted lifetime are additional diagnostics. This choice specifies
the experiment; it does not derive a universal goal from persistence.

## Learner and memory intervention

A frame MLP and GRU, all initialized randomly, feed an action-by-horizon
prediction head. Binary cross entropy uses only the action observed. Predicted
survival curves are made nonincreasing by a cumulative minimum for inference.
The chosen action maximizes predicted joint survival at step 12. There is no
pretraining, parameter expansion, reproduction, or separate frozen teacher.

Current batches are paired with an equal-sized replay batch. All replay methods
use the same architecture, initialization, optimizer, current arrivals and
update counts. Replay sampling uses replacement and an RNG separate from
membership selection. Before memory exists, the current batch fills the replay
half. The no-memory control always duplicates the current batch, preserving
the forward batch size and optimizer count while having no old information.
Each current example is inserted once, even if there are multiple update steps.

The recurrence estimator is intentionally modest: fixed random projections of
normalized image appearance and temporal differences yield an 8-bit sketch.
A bounded histogram estimates historical repeat frequencies with a Dirichlet(1)
pseudocount. It uses observations only. It does not infer the true regimes,
model periodic returns, predict a schedule, or use the trainable encoder. Fixed
preprocessing is shared by every method and is counted in memory overhead.

For each retained/current candidate, survival relevance is the predictor's
range of step-12 survival probabilities across possible actions. This is an
estimated sensitivity to action, NOT an estimate of the causal benefit of
retaining that particular memory. It can be confidently wrong. A 0.1 floor
prevents zero predicted sensitivity from automatically excluding a sketch.
All methods perform the same candidate predictions when their buffers are full;
the no-memory control scores fewer candidates. No labels from unperformed
actions enter the score.

For the four allocation methods, let n_b be the candidate count in sketch b.
Greedily remove the example with the smallest sketch-level decrement in
sum_b w_b log(1+n_b), until the bank meets its capacity. Within a sketch, evict
uniformly at random. Weights are held fixed during each allocation operation:

| Method | Weight or membership rule |
|---|---|
| `none` | No retained experience |
| `uniform` | Uniform reservoir over all arrivals |
| `recent` | Most recent arrivals |
| `coverage` | w_b = 1; control for coverage allocation alone |
| `recurrence` | w_b = historical predictive sketch frequency |
| `relevance` | w_b = 0.1 + mean predicted action sensitivity in the sketch |
| `joint` | w_b = frequency times (0.1 + mean action sensitivity) |
| `frozen_encoder` | Uniform replay; frame encoder and GRU frozen at initialization |
| `frozen_late` | Uniform replay; frame encoder and GRU frozen after four blocks |

The last two are explicitly privileged diagnostic interventions, not candidate
task-agnostic methods. Their backward work is smaller. The late freeze gets a
known boundary only to test feature adaptation; no ordinary learner receives it.

The memory objective rewards diminishing returns from coverage within a sketch.
It does not encode a learned symbolic relationship or implement the earlier
ideal retention-value equation exactly. An advantage over `coverage`,
`recurrence`, and `relevance` is needed to credit the combined ingredients.

## Schedules and evaluation

Four of eight combinations of supply direction, loss and delay form the known
contexts; the other four are held out for composition probes. Training order is
permuted by seed. Four initial blocks cover the known contexts. Six intervening
blocks have identical counts across the short/long schedules. Their order places
the penultimate target exposure one or five blocks before its final return,
which always occupies block 11. A twelfth block introduces a new gate glyph:
when active it reverses the source relationship. Gate was zero during earlier
training. This demands a new discrimination, but success alone does not prove
that new features rather than a new readout caused it.

`no_return` replaces the final target return with an unseen combination, so
retaining the retired target has an opportunity cost. Other contexts still
recur; this is not a wholly unpredictable or wholly nonrecurring world. Its
late gate uses that new combination, so the retired target does not reappear
through the gate-zero cases. Final held-out composition scores exclude the
combination that this schedule has now trained on.
`reversal` uses the long schedule but replaces the late gate block with the
same familiar cues and an inverted supply relationship. It tests adaptation
from feedback; an unannounced inversion cannot be inferred before feedback.
The obsolete target's old-world score is excluded from valid final retention.

Evaluation uses separate fixed seeds, with no updates or learner feedback.
Measure initial and within-block survival, return survival before updates,
held-out composition performance, late learning AUC, and final valid retention.
Probe the final-frame-only input to check dependence on temporal information.
Fresh, same-architecture uniform-replay fits on the late block provide a
difficulty reference. They have no previous data and their different replay
contents prevent a pure causal interpretation as a plasticity measurement.

Per-case counterfactual actions share identical exogenous noise. Their best
realized action is a **clairvoyant upper bound**, not an attainable oracle
policy. No transfer and the post hoc best fixed action are separate references.
All trials count, including early deaths; no survivor-conditioned metrics.

## Development, lock, budgets, and decision

First validate physics and acquisition using the uniform baseline only, on
development seed 117. Development revisions and unsuccessful trials must be
preserved. Select a workable common exposure budget before running the combined
mechanism or the separate evaluation cohort. Engineering smoke scores are not
scientific evidence. Save configuration, runtime, hashes, source archive, and
all raw per-seed curves before interpreting comparative outcomes.

Primary contrast: `joint` minus `uniform` on pre-update terminal-return survival
at H=12 in the long-gap schedule. Secondary endpoints: short-gap return, final
held-out composition survival, late-block learning AUC, valid known-context
retention, changed-rule adaptation, and the no-return control. Report every
seed and paired intervals; no endpoint switching or selective seed exclusion.
Do not infer protection from a small forgetting score if original acquisition
was poor. A result on this world does not establish generality or novelty.

The locked first pilot uses eight seeds chosen before comparative outcomes,
with each of the four target contexts occurring twice. Seed selection uses
only schedule permutations: take the first two eligible seeds per target,
starting at 1001. All nine methods run all four schedules: 288 comparative
runs plus 32 fresh late-block fits. Each run has 12,288 arrivals, 1,152
optimizer steps, 128 memory slots and 384 fixed evaluation trials per context.

The predeclared pilot continuation screen requires: at least +2 percentage
points in the primary contrast, a positive primary difference in at least
six of eight seeds, a positive mean difference over `coverage` on the same
endpoint, and no mean loss exceeding 2 points versus uniform in final held-out
composition or late acquisition AUC in the long schedule. These are practical
pilot thresholds, not a statistical significance test. Paired bootstrap
intervals are descriptive and unadjusted across secondary endpoints.
Baseline adequacy is a mean gain of at least 5 points over no transfer across
the first four context endpoints, averaged over uniform runs. If that fails,
treat the mechanism comparison as inconclusive rather than favorable.

All full-learning replay methods have the same bank capacity, sketch allocation,
model size, current exposure, minibatch size and optimizer-step count. Membership
selection work is additional and timed explicitly; this is matched optimization
work, not equal FLOPs or an assertion of equal runtime. Payload accounting
excludes Python objects, allocator overhead, gradients, activations and evaluator
caches. Initial encoder snapshots are diagnostics, not usable learning memory.
World generation and evaluation time are included in total runtime separately
from learning/selection counters. Source must not change during a suite.

Continue only if ordinary learning is adequate and the proposed ingredients
show a useful retention/transfer tradeoff beyond the simple controls. If the
combined policy is no better, preserve that negative result and diagnose the
estimators instead of scaling or claiming that recurrence itself was disproven.

## Running

```powershell
.venv/Scripts/python.exe -m acp_cl.persistence.study --config configs/persistence_smoke.json --output runs/persistence_smoke --device cpu
.venv/Scripts/python.exe -m acp_cl.persistence.study --config configs/persistence_development.json --output runs/persistence_development --device cpu
.venv/Scripts/python.exe -m acp_cl.persistence.study --config configs/persistence_pilot.json --output runs/persistence_pilot --device cpu
```

Use a new output path after a source/config/runtime change. `--resume` accepts
only the identical manifest and reloads trusted local per-block checkpoints.
Completed result files are not retrained. Interruption inside a block resumes
from the last complete block; that lost partial computation is not reflected
in the completed training counters. No checkpoint from an untrusted source
should be loaded.

Analysis commands and interpretation are in the completed report. The main
study uses CPU; a deterministic CUDA smoke also passed with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` set before launching Python.

Related primary research: [task-agnostic recurrent replay](https://proceedings.mlr.press/v232/caccia23a.html),
[evolutionary memory versus generalization](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005358),
and [causal usefulness of information](https://arxiv.org/html/1806.08053v3).
This implementation is not a reproduction of those algorithms.
