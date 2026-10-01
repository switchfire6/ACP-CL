# Matched history, query-dependent applicability, and continued learning

This is the prospective comparison following the accepted
[Continual Relational Learning charter](research_charter.md). It tests a strong
history-using comparator and one bounded intervention motivated by diagnosis
of the archived model. All previous study results remain unchanged.

## Question and prior diagnosis

Can allowing applicability to depend on the present observation improve
acquisition of a new dependency, while preserving history-based reuse and
resisting misleading feedback? The comparison must include a recurrent learner
given the same usable history; outperforming a pooled model alone is insufficient.

A post-hoc diagnostic used eight archived conditional models, new query seeds,
and disjoint calibration sets. On the late gate, ordinary routing scored 63.64%,
a calibrated single head 63.52%, and gate-specific calibrated heads 68.19%.
The last diagnostic knows the gate's sensor location and true condition grouping.
It shows usable conditional structure remains in some heads. It is not an
implementable ordinary policy or proof that routing explains all loss.

## Models and matching

All visual encoders are the same randomly initialized 64-wide frame MLP and
four-frame GRU, with identical initial encoder tensors per seed. There is no
pretraining. All models receive the same performed-action records, the same
previous 32 records, and the same uniform reservoir of 16 causal packets. A
packet contains 32 old support records and 32 query records. Query feedback
never enters its own history. No hidden mode, gate label, or boundary enters
an ordinary learner.

| Method | Mechanism | Parameters |
|---|---|---:|
| pooled | Four original prediction heads, uniform mixture | 57,852 |
| conditional | Original likelihood routing over the previous 32 outcomes | 57,852 |
| recurrent | Learned recurrent summary of that same observed history, then a single conditional decoder | 57,831 |
| query_routed | Original evidence scores plus a learned correction from query features | 58,112 |
| recurrent_frozen | Recurrent comparator with visual encoder frozen after four prefix blocks | 57,831 |
| oracle | Original heads selected by the true hidden base mode; privileged diagnostic | 57,852 |

The recurrent comparator's event input contains phi(x), the performed action's
one-hot vector, the three observed survival labels, and the generic interaction
phi(x) outer-product one_hot(action), multiplied by (2*y12-1). A 12-wide GRU
processes these past events; its outputs are averaged over the window. A
64-wide decoder receives the query feature and context summary. The interactions
use learned image features, not known physical quantities or an inferred true
mode. There is no persistent state outside the common history window. This
is a developed recurrent baseline, not a reproduction of 3RL.

The plain last-state GRU failed history-use checks at all three declared
development budgets. That failure and the subsequent recurrent recipe revision
are retained in the [development ledger](../reports/contextual/development_ledger.md).
Neither candidate outcomes nor pilot seeds select the revised baseline's budget.

For query-dependent routing, the four evidence scores remain

    L_k(h) = sum_(x,a,y in h) log Bernoulli(y12; sigmoid(f_k(x)_(a,12))).

The new weights are

    w_k(x,h) = softmax_k(L_k(h) + [W phi(x) + b]_k).

W and b begin at zero, so the initial function matches the original router.
This adds 260 trainable parameters and no additional predictors or raw memory.
It is a learned correction to likelihood scores, not a derivation of calibrated
Bayesian probabilities. Support scoring uses terminal outcomes only; it does
not multiply nested survival horizons as independent evidence.

All methods use the same selected Adam learning rate and common number of
updates per arrival batch, BCE for the performed action at three horizons,
and gradient clipping at 5. The recurrent model observes all three past labels;
the explicit likelihood uses the terminal label. This is a difference in how
available information is used, not extra observations. Model counts differ by
less than 0.5%; inference operations and active capacity differ. Equal raw
exposure and update counts are not exact FLOP matching. Record work and payloads.

The frozen diagnostic receives a known intervention boundary and keeps its
context GRU/decoder trainable. The oracle receives true query-mode labels.
These are explicitly privileged controls, never deployable competitors.

## Streams: shared past, three possible changes

Retain the conserving two-participant transfer physics. Every incoming batch
contains all eight combinations of motion direction, loss and delay equally.
The hidden mode A or B changes the consequences of otherwise identical images.
All eight combinations are trained; there is no held-out composition claim.

For each seed, A is seed mod 2 and B is its opposite. Two eight-block prefixes:

* Long absence: A, B, A, B, A, B, B, B.
* Short absence: A, B, A, B, B, B, A, B.

Each mode occurrence lasts 768 or 1,280 trials, selected deterministically by
seed and occurrence. The two orders contain exactly the same record multiset,
total exposures, and final B block. A is absent for one or three blocks before
the challenge. Durations vary without a clock or boundary signal supplied to
the model. Parameter updates proceed through the stream without state resets.

Clone each completed prefix into three independent 1,024-trial challenges:

* Return: A reappears with its familiar mapping.
* Novel: B remains, but a previously constant gate now varies; gate one uses
  the reversed mapping. This introduces a new association between an observed
  cue and known consequence mappings. It is not an entirely new physical law.
* Noise: B's physical dynamics remain identical. With probability 0.20, a
  reported outcome is replaced by an independently sampled valid monotone
  three-horizon survival curve. This is measurement noise; true survival and
  evaluation remain unchanged. Some replacements equal the original report.

Noise affects the observed feedback of the performed action only. No
counterfactual action label enters training. Query evaluation always uses true
physical outcomes. All branches start from identical weights, optimizer,
history, replay membership, and sampling RNG state; branch updates never affect
the shared prefix or another branch. Prefix sharing saves duplicate computation;
it does not create independent statistical samples.

## Evaluation and primary endpoint

At each challenge start, use a frozen copy of the prefix model to examine
adaptation after 0, 8, 16, and 32 new observed outcomes, with no weight updates.
Use four independent support sets and a disjoint 384-trial query set. Also probe
opposite-mode support, erased support, and broken image/action-outcome binding.
These records never enter training. Noise-branch support has the declared
measurement corruption; its query labels remain physically correct.

During challenge learning, evaluate every 256 arrivals using the live past
history. Primary endpoint: normalized trapezoidal area under this new-dependency
survival curve, including entry and endpoint. Primary contrast:
**query_routed minus recurrent, on the novel branch**.

Guardrails concern the original conditional model's old-mode return before
weight updates, valid original-mode retention after new learning, and noise-
branch survival AUC. Final valid-mode probes use 32 fresh correct past outcomes
to select the mode, without further gradient updates. Both original modes
remain valid in every branch. Feature freezing is a diagnostic of continued
visual adaptation; its result alone does not identify a particular new feature.

Use eight new seeds, 5001--5008, six methods, two gap orders, and three branches:
**96 prefix fits and 288 challenge branches**. Average the two gap orders within
each seed before contrasts and intervals. The independent unit is the seed,
not a branch, gap, support replicate, or individual query. Report both gap orders
separately as well. Paired bootstrap intervals use 20,000 resamples with fixed
analysis RNG 4172026; secondary intervals are descriptive and unadjusted.

## Development, lock, and decision

Development seeds are 117 and 118. Only pooled, recurrent and oracle outcomes
select the common update budget. The ledger declares the finite budget grid
and records the inadequate first recurrent recipe and its revision. The revised
recurrent recipe qualified at 12 updates per batch; use 12 for all scientific
methods. The complete grid and its failures remain archived. Smoke
seed 19 tests software contracts and is not evidence of acquisition.

Adequacy must also hold on the fresh cohort means: oracle and recurrent initial
four-block gains over no transfer at least 5 percentage points, recurrent return
gain over no transfer at least 5 points, and recurrent correct-minus-opposite
history benefit at least 2 points. Failure makes a superiority comparison
inconclusive; do not tune or exclude pilot seeds to repair it.

The practical continuation screen requires all of:

1. Novel learning AUC improvement of at least 2 points over the recurrent
   comparator, positive in at least six of eight seeds.
2. Novel learning AUC improvement of at least 2 points over the original
   conditional model, positive in at least six of eight seeds.
3. No mean loss worse than 2 points versus the original conditional model in
   inference-only old-mode return or post-novel valid-mode retention.
4. No mean loss worse than 2 points in noisy-feedback AUC versus either the
   recurrent comparator or the original conditional model.

These are pilot decision thresholds, not significance/noninferiority tests.
Report numerical losses and uncertainty even when a threshold passes. A gain
on the old return test alone cannot count as success. A failed candidate should
not replace the working reference. A recurrent model offering a better complete
tradeoff is a legitimate result in favor of the simpler reference.

Save the exact protocol, config, runtime, training source and all outcomes
before interpretation. Keep scientific source immutable during the main cohort.
Save one final prefix checkpoint and one final checkpoint per branch, with
intermediate atomic checkpoints for crash recovery. Audit data, memories, and
all primary/final probes from those checkpoints before reporting conclusions.

## Running

```powershell
.venv/Scripts/python.exe -m acp_cl.contextual.study --config configs/contextual_smoke.json --output runs/contextual_smoke --device cpu
.venv/Scripts/python.exe -m acp_cl.contextual.study --config configs/contextual_pilot.json --output runs/contextual_pilot --device cpu --protocol docs/contextual_comparison.md
```

The final selected budget is recorded in the config and ledger. Use a new
output directory after any source/config/runtime change; `--resume` accepts
only identical identities and trusted local checkpoints. The ordinary scientific
cohort uses CPU worker processes, one PyTorch thread each. Timings under shared
CPU load are not isolated efficiency benchmarks.
