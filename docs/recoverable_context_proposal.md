# Adam: next direction after the information audit

**Status.** Design proposal, 2026-09-28, revised the same day after an
independent critical review. Responds to
[representation_learning_next_thread_prompt.md](representation_learning_next_thread_prompt.md).

This is not a locked protocol, and no learner was fit for it. It relies on the
post hoc information audit in
[reports/information_audit/](../reports/information_audit/README.md). That audit
is exploratory and not decision-bearing. The temporal-representation recipe
stays stopped. Do not run `configs/representation_learning_main.json`.

## Outcome log (added 2026-09-29)

| Step | Result | Report |
|---|---|---|
| A1 retrieval capacity | **BORDERLINE.** Base A is expressible through one fitted context but not reachable from evidence. Part of the gain is generic. | [retrieval_capacity](../reports/retrieval_capacity/summary.md) |
| A2 decision lens | Computed in the audit. Decision regret is dominated by the general competence gap. | [information_audit](../reports/information_audit/README.md) |
| B competence gap | No arm is competent. R256 is a narrow miss (24.1%). Exposure shows a steady state. All-action feedback closes 44%. | [competence_gap](../reports/competence_gap/summary.md) |
| B2 competence ceiling (added step) | The .010 bar is achievable (feature ceiling .0078). **Online optimisation is 64% of the gap.** R256 −37% and EMA −48%, each in 12/12 seeds. | [competence_ceiling](../reports/competence_ceiling/summary.md) |
| C consolidated recall (revised Step C) | **Stopped at development.** Recalling whole-network snapshots hurts even with an oracle index. A global EMA gives most of the benefit. | [consolidated_recall/dev](../reports/consolidated_recall/dev/summary.md) |
| D pivot | Design revised after critique. D0 (world and references) is in progress. | [step_d_design](step_d_design.md) |

The Step C described below was superseded by
[consolidated_recall_protocol.md](consolidated_recall_protocol.md). That
change followed B2's declared consequence: test consolidation against
plasticity.

## 1. What the failed qualification demonstrates

The audit compares the learner with three Bayes references. All of them are
scored on **exactly** the saved endpoint queries, supports and truths:

- **I, the ideal observer**, sees the same query frames and the same 32-record
  support. It knows the simulator and the five interleaved laws.
- **O, the law-known oracle**, is also told the true law.
- **P, the context-free mixture**, uses no support.

An independent reimplementation matched every O/I/P cell mean within Monte
Carlo error (maximum difference 3.9e-5). An adversarial review found no bug
that changes any cell.

| Failed cell | Rule | Learner | I | O | P (no context) |
|---|---|---:|---:|---:|---:|
| Base A all-case Brier | <= .12 | .196 | .045 | .045 | .255 |
| Delay cue benefit, stage 1 | >= .002 | -.0002 | .0102 | .0168 | .0084 |
| Delay cue benefit, stage 2 | >= .002 | .0008 | .0128 | .0161 | .0071 |
| Delay cue benefit, stage 3 | >= .002 | -.0003 | .0099 | .0128 | .0046 |

**The failures are learning-limited, not information-limited.** I passes all 31
cells. Even P passes the three delay cells, so the delay failure has nothing to
do with context identification. P fails only the Base A cells.

The cue benefit is a sensitivity measure, not a proper score, so I is a
reference for it, not an upper bound. The
[online reanalysis](../reports/information_audit/online/summary.md) adds three
findings.

1. **The learner is far from competent even within one context. This is the
   largest effect.**
   - Every law's online Brier plateaus at about .084-.102. The same-support
     ideal is about .044.
   - Fresh single-law fits also flatten at .073-.082 by the third or fourth
     1,024-arrival chunk. That is a steady state, not a shortage of samples.
   - For Base A, the excess over I is about .062 per case. Only about 24% of it
     is the transient after a switch. The remainder is plateau excess.
   - Across laws, the steady-state gap is about 75-95% of regret.
   - At the endpoint, the learner does worse than the context-free mixture P on
     4 of the 5 laws. It is .041-.047 lower in survival and .022-.029 worse in
     Brier. It beats P only on Base A.
2. **Base A is partly retrieved, and the rest is overwritten.** Base A is the
   only law with its hidden mode, so it makes up 20% of the interleaved stream.
   - One Base A support favours Base A over Base B by about 87 nats, and I
     recognises it after one batch.
   - With that support, the learner's forecasts move only about halfway toward
     mode A. The projection goes from .17 to .56, and the Brier from .357 to
     .200.
   - The rest is relearned in the weights each visit and overwritten by the
     four mode-B blocks. Base A starts each visit .085 worse than it ended the
     previous one.
   - The locked endpoint is effectively a "support without update" test. It
     tracks that projection across seeds (r = -.995).
   - More Base A packets in the replay reservoir predict better switches.
3. **Delay: a small conjunctive relation that is not learned.**
   - The true effect is about .033 survival probability. The best flip benefit
     is about .016 Brier.
   - The learner's cue-1 response is mostly a generic direction bias. It is as
     large on non-delayed cases (.0047) as on delayed ones (.0049). It is only
     weakly specific to the delay-active law: .0049 there versus .0018 in the
     predecessor law.
   - The learner reaches about 23-31% of the ideal effect fresh and about 14%
     interleaved.
   - For decisions, flipping the delay cue changes the oracle's argmax action
     on about 15 of 256 affected queries. It costs only about .007 expected
     survival (at most .013), because those actions are near-ties. Supply
     timing, by contrast, is worth about .4 survival.

**What this does not show.**

- Nothing about the untested reconstruction arms.
- No deficit located inside the encoder.
- No impossibility of continual learning in this world.
- I and O are references, not learners.
- Six seeds give a descriptive comparison only.

## 2. Answer to the bottleneck question

| Candidate | Verdict | Evidence |
|---|---|---|
| Context identification (information) | No | The mode is identified from 32 records with posterior 1.0. I passes every cell. |
| Finite history | Not for the mode. Partly for delay-only law pairs. | About 100-130 records are needed for 20:1 odds between a law and its delay-only neighbour. I still passes, because confusable laws forecast alike. |
| Sparse outcome feedback | Slows learning | The exact failure time would carry about 2.3-3.4x the delay information per record. Performed-action-only feedback is part of why the learner extracts so little. |
| **The learner itself** | **Yes, twice** | (a) A steady-state gap of about 2x the ideal error within a single context. This is most of the regret. (b) Minority-context knowledge is held in overwritten weights, not in a form the evidence can retrieve. |

## 3. Methodological change

The [evidence map](../reports/information_audit/evidence_map.md) shows three
problems with past decision rules:

- No threshold (.002, .02, .12, .005) was calibrated against an achievable
  ceiling.
- Conjunctions grew to 31-44 cells. At .95 per cell, 31 cells pass together
  only about 20% of the time.
- Only one learning screen has passed since the conditional pilot.

Future protocols should do three things instead:

1. Score **regret against an ideal observer that has the same information set
   as the learner**, both online and in probes.
2. Report **decision (survival) regret** alongside Brier.
3. Use at most five criteria, 12 seeds, and a pre-declared ceiling test.

## 4. Recommended sequence

### Step A: two cheap diagnostics (no training of learners)

**A1. Retrieval capacity.** Run this on the six saved interleaved endpoint
checkpoints. All learner weights stay frozen.

1. Average the forecasts, or the context vectors, over many independent
   32-record Base A supports. This measures noise in producing the context
   while staying in the training distribution.
2. Fit **only the 12-dimensional context input** to Base A performed-action
   outcomes on a disjoint set, then score the resulting c* on the endpoint
   queries against I.

How to read the result:

- **If c* reaches about the Base A plateau (about .08-.09):** the knowledge is
  still expressible. Only retrieval is failing, and retrievable context codes
  are justified.
- **If c* stays far above that:** the decoder itself was overwritten. Codes
  over a shared predictor cannot help, so do not run Step C.

Step A1 replaces the earlier support-length idea. Because the context GRU's
outputs are mean-pooled, longer supports only reduce variance, and steps past
32 are out of distribution.

**A2. The decision lens is already computed.** Endpoint survival regret against
I is .112 for Base A and .067-.094 for the other laws. Base A's share is about
26%, against a base rate of 20%. Decision regret is dominated by the general
competence gap, not by the minority context. Delay contributes little.

### Step B: the stationary competence gap (small, prospective, fresh seeds)

This comes first because it is 75-95% of regret. Every continual-learning
mechanism tested so far sat on top of it.

> **Question:** Why does a single-law learner saturate at about 1.7x the ideal
> Brier, and which declared factor closes most of the gap?

**Design.** A pre-declared factorial on one stationary law. This is not a
tuning sweep of the stopped recipe.

- Factors:
  - updates per current batch (12 vs 3): overfitting to the current batch;
  - replay capacity (16 vs 256 packets);
  - network width (64 vs 256).
- Plus one feedback-information arm: exact failure time instead of three
  horizons. This arm answers the sparse-feedback question directly.
- Score: the final-plateau excess over the law-known ideal, plus the learning
  curve.

**Outcome.** The result is a characterised base learner. Fix it before any new
continual-learning comparison, and report it. If no factor closes the gap, that
is an important negative finding about the learner family itself.

### Step C: history-matched retrieval test (only if A1 passes)

The candidate keeps the reference encoder and decoder shape. It replaces the
support-GRU context with K learned codes, selected by their predictive
likelihood of observed outcomes. The arms:

| Arm | Question |
|---|---|
| Reference recurrent | Baseline |
| **4-head conditional learner** | Primary comparator. It has the project's best retrieval, but slow acquisition. |
| Codes, posterior recomputed from the same 32-record window | Does indexing help at **matched history**? |
| Codes, persistent sticky posterior | Does longer evidence integration add to indexing? Compare it with a sticky Bayes filter having the same information. |
| Codes, K = 1 | Controls for capacity and re-parameterisation |
| Oracle codes, K = 5, index lagged one batch | Privileged ceiling. The lag removes the unearnable position-1 advantage. |

**Pre-registered details.**

- Replay responsibilities are computed from each packet's stored support. The
  carried posterior is detached from the gradient.
- Codes use a declared symmetry-breaking rule (distinct initialisation plus
  hard-EM warm-up).
- Collapse diagnostics: pairwise code distance, posterior entropy and the
  effective number of codes.
- Parameter, byte and multiply-add counts are reported. The candidate drops the
  context GRU.
- Mode parity is crossed with cue order across the 12 seeds.
- The revision block is dropped unless I is extended to cover it.

**Primary outcome.** A retrieval-without-update probe at every Base A visit
start in cycles 3-8. Weights are frozen, the first Base A batch is the support,
and the measures are Brier and the mode-A projection against I. The threshold
is expressed as a fraction of the lagged oracle's gain.

**Secondary outcomes.** Recovery transient, acquisition regret, whole-stream
regret and survival regret, each as a fraction of the reference's excess.

### Step D: benchmark pivot

The relational/persistence idea proper has never been tested. It needs:

- agent-chosen actions, where survival shapes the agent's own data;
- dense observation of consequences, such as post-action gauge trajectories;
- several interacting entities.

Move there after Step B produces a competent base learner. Carry over the
ideal-observer regret yardstick. Test compounding with matched early and late
dependencies. If Step A1 or Step C fails, pivot sooner rather than trying
another architecture here.

## 5. Novelty, stated honestly

Some ideas here are not new:

- **Consequence-based context inference** is established: COIN, MOLe, CN-DPM,
  MOCA, and the project's own conditional learner.
- **A shared predictor with low-dimensional retrievable codes** also has
  precedents: Thalamus, hypernetwork task embeddings and CAVIA.

The defensible contribution so far is **methodological**: ideal-observer
calibrated regret for learned, pixel-input, sparse-feedback context recovery.
It separates information-limited failures from learning-limited ones. The
indexed-code learner is a test of a mechanism class, not the novel design.

A genuinely novel design is more likely to come from Step D. There,
decision-relevant, recoverable relational knowledge can be scored against an
ideal agent.
