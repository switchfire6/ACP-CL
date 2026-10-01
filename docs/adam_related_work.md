# Adam: related work and the long-horizon question

Literature checked on 2026-09-27. This is a focused primary-source review to
guide the next experiment, not a systematic review or proof of novelty. It
interprets the accepted [research charter](research_charter.md) and completed
[replay-renewal studies](../reports/replay_renewal_results.md). Recommendations
below are proposals; they do not change any completed study's decision rule.

Adam is a worthwhile experimental project because it has a measurable problem,
bounded resources, and increasingly informative controls. Its broad ingredients
are established: retaining experience, separating stable and adaptable knowledge,
inferring context, and using accumulated evidence before changing a model.
The present evidence supports continued investigation, not a claim of a new
general learning principle or a successful continual-learning architecture.

The closest primary references establish the following boundaries:

| Primary source | Established result or mechanism | Relevance to Adam |
|---|---|---|
| Bifet and Gavalda, **Learning from Time-Changing Data with Adaptive Windowing**, SDM 2007. [ADWIN](https://epubs.siam.org/doi/10.1137/1.9781611972771.42) | Adapt a window using evidence for differences between older and newer observations; use it to monitor predictor error or maintain changing statistics. | Repeated evidence and adaptive memory duration are established. A detector is a necessary comparator if Adam claims an advantage from deciding when to update. Its mathematical guarantees do not automatically transfer to correlated neural training losses. |
| Adams and MacKay, **Bayesian Online Changepoint Detection**, 2007. [BOCPD](https://arxiv.org/abs/0710.3742) | Maintain a posterior over the time since the latest change under a specified generative model and change prior. | Probabilistic evidence accumulation about a hidden environmental change is established. The assumed likelihood determines whether increased noise is interpreted as a change. A practical bounded version must account for truncation and computation. |
| Coskun and Tumer, **Learning under concept drift and non-stationary noise: Introduction of the concept of persistence**, Engineering Applications of Artificial Intelligence 123, 106363, 2023. [PeARL](https://doi.org/10.1016/j.engappai.2023.106363) | Use a local persistence measure to adapt learning rates while tracking a changing discrete distribution contaminated by non-stationary discrete white noise. | This directly precedes the idea of using persistent evidence to distinguish meaningful variation from noise. Its estimation setting does not itself solve learning visual representations or retaining multiple conditional relationships. |
| Arani, Sarfraz and Zonooz, **Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System**, ICLR 2022. [CLS-ER](https://arxiv.org/abs/2201.12604) | Combine episodic replay with short-term and long-term semantic memories without requiring task boundaries. | Multiple learning speeds and stable/plastic memories are established mechanisms. Any extra teachers, snapshots, or models in Adam must be counted in the resource comparison. |
| Heald, Lengyel and Wolpert, **Contextual inference underlies the learning of sensorimotor repertoires**, Nature 600, 489-493, 2021. [COIN](https://www.nature.com/articles/s41586-021-04129-3) | Model memory creation, updating, and expression through contextual inference, and test predictions in motor learning. | Reusing an existing mapping and changing its parameters are distinct operations. Adam's conditional reuse results fit this distinction; they do not independently establish the biological theory. |
| Kim, Jeong, Moon and Kim, **Continual Learning on Noisy Data Streams via Self-Purified Replay**, ICCV 2021. [SPR](https://arxiv.org/abs/2110.07735) | Combine self-supervised replay and filtering to reduce the effect of noisy labels on continual classification. | Protecting replay from misleading feedback is established. A new mechanism should avoid treating unfamiliar but learnable examples as noise merely because the old model finds them difficult. |
| Bang et al., **Online Continual Learning on a Contaminated Data Stream with Blurry Task Boundaries**, CVPR 2022. [PuriDivER](https://arxiv.org/abs/2203.15355) | Manage the purity/diversity tradeoff of episodic memory and use robust semi-supervised learning in blurry noisy streams. | Memory selection has an existing literature beyond uniform versus recent replay. Retaining diverse evidence and rejecting corrupted labels can conflict. Class-incremental results do not automatically settle changes in the meaning of an existing cue. |
| Liu, Cheng and Zhang, **Identifiability of Label Noise Transition Matrix**, ICML 2023, PMLR 202:21475-21496. [Identifiability](https://proceedings.mlr.press/v202/liu23g.html) | Characterize conditions for identifying label corruption, including the role of multiple noisy labels and additional assumptions for instance-dependent noise. | Repetition alone cannot identify every latent physical change. A declared observation/noise model is part of the scientific hypothesis. |
| Dohare et al., **Loss of plasticity in deep continual learning**, Nature 632, 768-774, 2024. [Continual backpropagation](https://www.nature.com/articles/s41586-024-07711-7) | Study learning over long task sequences, including 5,000 ImageNet binary tasks and 800 permuted-MNIST tasks, and test selective unit replacement and regularization. | Early performance can conceal later learning failure. Long sequences and matched acquisition comparisons are necessary to evaluate preserved plasticity; successful retention alone does not establish it. |
| Prakash et al., **Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning**, arXiv:2509.22335v3, revised May 2026. [Spectral collapse](https://arxiv.org/abs/2509.22335v3) | Analyze curvature and trainability, and evaluate feature-rank and weight regularization in continual supervised and reinforcement learning. This reference is treated here as a preprint. | Long-term learning failure can arise inside the representation and optimization dynamics even if change detection works. Rank and weight diagnostics can help explain a result, but are not substitutes for behavioral learning measurements. |

A mechanism-focused follow-up, after specifying the fixed-proposal diagnostic,
found closer precedents for deciding when to replace a predictor. In **Paired
Learners for Concept Drift** (Bach and Maloof, ICDM 2008, pp. 23-32), the deployed
stable learner and a reactive learner trained on a recent window both continue
learning. A circular window counts cases where the stable learner is wrong and
the reactive learner is correct. Exceeding a threshold copies the reactive
model into the stable learner and clears those indicators. This is an explicit
precedent for replacing an incumbent using repeated predictive evidence.
[Primary paper, Algorithm 1](https://cs.brown.edu/people/sbach/files/bach-icdm08.pdf).

The **Streaming Ensemble Algorithm** (Street and Kim, KDD 2001, pp. 377-382)
also predates testing a fixed candidate on later observations: it trains a
classifier on one chunk, evaluates it and existing ensemble members on the next,
and admits or replaces a member based on a quality score. The score emphasizes
cases where the ensemble vote is close; deployment uses a bounded voting
ensemble. [SEA publication](https://doi.org/10.1145/502512.502568),
[primary paper, Section 2 and Figure 1](https://static.aminer.org/pdf/PDF/000/473/318/a_streaming_ensemble_algorithm_sea_for_large_scale_classification.pdf).

**Learning with Drift Detection** (Gama, Medas, Castillo and Rodrigues, SBIA
2004, pp. 286-295) monitors online prediction errors. After a warning and then
a drift threshold, it learns a replacement using observations collected since
the warning. This is a precedent for error-triggered rebuilding, rather than
direct comparison of two fixed predictors.
[Primary publication](https://link.springer.com/chapter/10.1007/978-3-540-28645-5_29).

Adam's [locked diagnostic](evidence_consolidation_protocol.md) has a narrower
scope: all policies share one uniformly replay-trained neural draft. A proposal
and each delayed incumbent have fixed weights during a non-overlapping window
of eight newly arriving packets. The sustained rule requires a mean terminal
Brier gain above .002 and positive gains on at least six packets. Fixed-delay
and single-packet controls share the same proposal and decision time. Evaluation
separately measures truthful outcomes, noise periods, retained knowledge, and
matched early/late cycles. These choices define an experiment about accepting
updates from a shared trajectory. They do not establish novelty of incumbent
replacement, future-data validation, or continual representation learning.

A plausible distinctive contribution would be a specific, bounded neural
mechanism that uses only available experience to decide which conditional
relationships to revise, and demonstrates better acquisition, retention,
recurrence, and noise behavior under controlled resource budgets. A carefully
constructed benchmark or a reproducible diagnosis of a failure can also be a
contribution. Neither uniqueness of the benchmark nor algorithmic novelty has
yet been established. A positive result against uniform replay would be a useful
pilot result; comparison with close prior mechanisms and an independent setting
would still be needed for a stronger claim.

The phrase "real change versus noise" needs an observable definition. Let
`p_t(y | x, a)` be the physical outcome distribution, and let
`K_t(y_report | y, x, a)` be the reporting process. The learner observes

```text
q_t(y_report | x, a)
    = sum_y K_t(y_report | y, x, a) p_t(y | x, a).
```

In general, different pairs `(p_t, K_t)` produce the same `q_t`. For example,
if a binary outcome always equalled a binary input, then a permanent reversal
of the physical mapping with truthful reports is observationally equivalent
to an unchanged physical mapping with every report reversed, when those
reports are the only evidence about the outcome. More observations cannot
separate identical observable laws. This is an illustrative derivation, not
a claim that Adam's present stochastic corruption is exactly this example.

The practical hypothesis must therefore be narrower: under declared families
of physical changes and reporting corruption, can the learner exploit their
different observable patterns? Independent replacement noise, short bursts,
persistent biased corruption, and changes restricted to some inputs should be
distinguished. Even independent corruption at a persistent rate generally
changes the observed conditional mean. Learning that mean can improve reported
prediction while worsening prediction of the physical outcome. A stable-versus-
adaptive predictor comparison cannot avoid that ambiguity by accumulating
more evidence alone.

An example of an observable statistic is the prequential loss difference:

```text
d_t = loss(stable_prediction_before_update, reported_outcome_t)
    - loss(adaptive_prediction_before_update, reported_outcome_t)

E_t = bounded_summary(d_1, ..., d_t).
```

Predictions must be saved before either learner trains on the current example.
The evidence should come from newly arriving observations; repeatedly replaying
one packet does not create independent confirmation. Any threshold or
persistence requirement must be calibrated on separate development streams,
including stationary noise. Repeated tests, overlapping windows, correlated
outcomes, and changing predictors make an ordinary fixed-sample confidence
claim inappropriate unless its assumptions are actually satisfied. Positive
`E_t` demonstrates a predictive advantage on reported outcomes under the chosen
statistic, not an oracle for the cause of the discrepancy.

The user's longer-term possibility is reasonable: a learner may accept a local
cost to acquire representations that improve later adaptation or reuse. The
current studies cannot establish that payoff. The continuous replay study had
three introduction stages followed by independently forked final branches.
Its many seeds and episodes improve comparisons within that design; they do
not form one long learning life. Predicting survival at multiple horizons
within an event is also different from preserving the ability to learn over
many environmental changes. Previous return benefits provide evidence for
reuse, while the repeated valid-old losses provide a real cost to explain.

For a future continuous stream, define a physical prediction score `L_k` over
each prespecified cycle `k`, using evaluator-only truthful outcomes. A basic
measure of cumulative advantage over uniform replay is

```text
g_k = L_k(uniform) - L_k(candidate)
G_K = sum_{k=1}^K g_k.
```

The stream schedule and scoring weights determine which short-term costs can
be offset by later benefits; they must be fixed before inspecting results.
Physical prediction loss, survival, and compute should remain separate reported
quantities unless a combined utility has been declared. A positive `G_K` can
arise from a constant small benefit repeated many times. Evidence that earlier
learning makes later learning progressively better requires more: increasing
advantage or fewer examples needed on matched later challenges, improved reuse
after longer absences, or transfer to new combinations. The word "compounding"
should specify which of those effects is intended. None implies indefinite
improvement with finite capacity.

A subsequent long-stream protocol should make the following comparisons:

1. **Continuous experience:** include many changes, recurring conditions,
   genuine revisions, stable intervals, and multiple noise durations. Keep
   bounded memory and parameters, and count detector state, snapshots, forward
   passes, and all training updates. Compare candidate, uniform replay, recent
   replay, and a detector-based control under documented budgets.
2. **Learning at different ages:** present matched novel dependencies early
   and late, counterbalance difficulty and order, and include fresh-weight
   references with the same challenge exposure. Comparing unrelated easy early
   tasks to hard late tasks does not identify loss of plasticity.
3. **Reuse and transfer:** vary the absence before a condition returns, and
   test both exact recurrence and new compositions of learned dependencies.
   Exact recall, relearning speed, and learning a new dependency are different
   outcomes. Continue testing old knowledge only where it remains valid.
4. **Detection and adaptation:** measure false interventions, delay after a
   true change, and clean/noisy prediction differences. Ablate the accumulated
   evidence rule while holding the update mechanism fixed; otherwise an extra
   model or extra compute may explain the gain.
5. **Lifetime costs:** report cumulative physical loss and survival, the last
   part of the stream, worst sustained degradation, and recovery time. A good
   final endpoint can hide substantial intermediate failure. Prespecify any
   acceptable temporary cost and a horizon by which it must be recovered.
6. **Independent evidence:** develop thresholds on fresh seeds, lock the
   comparison, keep every trial, and estimate uncertainty across independent
   streams. Verify a second environment before claiming generality. A single
   within-world success would still be a useful bounded result.

The current 18-block continuous experiment tests whether repeated evidence
improves deployment across adaptations, returns and a declared noise family.
It measures cumulative performance and matched early/late differences, but
cannot identify compounding representation gains because draft learning is
shared. If it fails, condition-level results can suggest where commitment
lags or noise resistance are costly; identifying the cause requires another
controlled comparison. If it succeeds, independent confirmation and genuinely
novel dependencies arriving late in life remain necessary next tests. The
previous split policy's failed screen remains a failure; possible future
benefits are a new hypothesis to measure, not a reason to revise its past
decision.

Completed evidence (2026-09-27): the [fixed pilot](../reports/evidence_consolidation_results.md)
passed fresh-model qualification in both architectures but failed both combined
screens. Repeated evidence improved noisy-block predictions while increasing
whole-stream physical error in every seed versus immediate deployment. The
small mean early-to-late narrowing is uncertain. This result sharpens the next
mechanistic question; it supplies neither algorithmic novelty nor evidence of
compounding representation learning.

The subsequent [historical synthesis](research_synthesis.md) and
[matched return diagnosis](../reports/return_context_v2/summary.md) motivate a
different local question: whether the learning update can preserve useful
historical predictions while increasing emphasis on current experience. The
[selective-update development protocol](selective_updates_protocol.md) is fixed
before its fresh-seed fitting. It remains a development test of a known family
of mechanisms, with a separate smaller-displacement control.

[A-GEM, Chaudhry et al., ICLR 2019](https://arxiv.org/abs/1812.00420) is a close
precedent for using episodic gradients to constrain interference. Adam's
current study operates on the actual displacement proposed by its adaptive
optimizer, so the direction being constrained includes its existing moments
and preconditioning. That distinction specifies the implementation; it is not
evidence of novelty. The condition is first order on one sampled reported loss,
not a guarantee about every memory or about clean physical outcomes.

[Dark Experience Replay, Buzzega et al., NeurIPS 2020](https://arxiv.org/abs/2004.07211)
combines rehearsal with preservation of earlier predictions. It is a relevant
alternative if output consistency proves more useful than gradient constraints.
Historical outputs can be undertrained or become obsolete, so preserving them
also needs revision and noise controls in the present world.

The user's [predictive-retention proposal](predictive_retention.md) adds a
distinct question: which memories improve predictions on subsequent experience?
High accuracy, resistance to interference, and marginal predictive contribution
are different quantities. The note links primary work on prequential evaluation,
data valuation and interference-based replay. No experiment so far directly
estimates that proposed future marginal-value signal for each memory or feature.
