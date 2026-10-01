# Predictive value of rehearsal: precedents and design constraints

Status: focused primary-source review, checked 2026-09-27. These are design
recommendations for the [predictive-retention proposal](predictive_retention.md),
not results, a locked protocol, or a claim of novelty. The completed
[selective-update failure](../reports/adam_revision_results.md) remains unchanged.

The useful question is whether a packet's estimated benefit from additional
rehearsal predicts its benefit on later observations and returning conditions.
This is narrower than assigning intrinsic importance to an experience or
showing that a new replay policy improves continual learning.

## Closest primary-source precedents

| Work | What it establishes or proposes | Consequence for this experiment |
|---|---|---|
| [Dawid, prequential inference (1984)](https://academic.oup.com/jrsssa/article/147/2/278/7106293) | Assesses sequential probability forecasts against subsequently observed outcomes. | Record predictions before their labels become available. Historical training loss is not the accuracy control. |
| [Ren et al., Learning to Reweight Examples (2018)](https://proceedings.mlr.press/v80/ren18a.html) | Learns example weights through a provisional update evaluated on clean, unbiased validation data. | Short-update validation has direct precedent. Adam's learner receives ordinary reports, so it cannot inherit that clean-validation assumption. |
| [Ghorbani and Zou, Data Shapley (2019)](https://proceedings.mlr.press/v97/ghorbani19c.html) | Values examples through marginal contributions to a specified learner's performance across training subsets. | A single matched rehearsal fork is a conditional intervention value, not a Shapley value or removal of all historical influence. |
| [Ghorbani et al., Distributional Shapley (2020)](https://proceedings.mlr.press/v119/ghorbani20a.html) | Defines valuation relative to an underlying distribution and studies its statistical properties. | Specify the evaluation distribution and time window; value under one window need not persist after a change. |
| [Aljundi et al., MIR (2019)](https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf) | Retrieves stored examples whose loss would increase under a virtual current update, searching a bounded candidate sample. | Interference is a cheaper vulnerability proxy. It does not directly estimate benefit on future, unseen queries. |
| [Shim et al., ASER (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17159) | Uses KNN Shapley values in learned feature space for replay retrieval and memory updates, balancing current and stored classification boundaries. | Data valuation for replay is established prior art. Its available memory/current reference sets and classification proxy differ from delayed future-window assessment. |
| [Pruthi et al., TracIn (2020)](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html) | Attributes changes in test-point loss to training examples through gradient and checkpoint approximations. | Training contribution can be local and trajectory-dependent; checkpoint influence does not itself validate a future replay decision. |
| [Wang et al., In-Run Data Shapley (ICLR 2025)](https://arxiv.org/html/2406.11011v3) | Builds attribution from per-update utility in one training trajectory. Its limitations discuss validation arriving only after training. | Our temporal separation is material: freeze interventions before later feedback, then assess whether the resulting ranking survives a second window. |
| [Ding et al., In-Run Data Shapley for Adam (2026 preprint)](https://arxiv.org/html/2602.00329v1) | Derives Adam-aware attribution approximations under a fixed-state utility definition and evaluates optimizer mismatch. | Preserve Adam moments and counters in actual forks. Do not assume an SGD gradient score equals an Adam intervention. Its approximations are not validated in this project. |

These works justify comparison with established valuation and replay methods;
they do not establish that the proposed diagnostic will pass. The review is
focused rather than an exhaustive novelty search. Statements below are our
design deductions, not reported findings of the cited papers.

## Define the intervention and the time index

At assessment time s, let S_s include parameters, optimizer state, causal replay,
history and random state. Choose candidate packet j and replacement r using
only information already available. Let T_K be a fixed K-update rehearsal
operation, with all other training inputs matched:

    S_j = T_K(copy(S_s), rehearse=j)
    S_r = T_K(copy(S_s), rehearse=r)
    v_j(W | S_s, K, r) = mean_(t in W) [L_t(S_r) - L_t(S_j)].

Use the declared proper prediction loss on reported outcomes for selection.
Positive value means that this rehearsal beat this replacement on W. It does
not mean the packet was indispensable or that deleting it would erase its
past contribution. An unmodified starting model can be an additional reference,
but it has fewer updates and answers a different question.

Weights and optimizer remain frozen during scoring. Context may evolve with
already observed feedback, identically for every shadow; specify this explicitly.
Each prediction must precede its own target and any support derived from that
target. Future windows are held out from the shadow updates, even if the main
uniform learner subsequently learns from those arrivals on its separate path.

Separate the score window W_score from a later validation window W_next and a
delayed-return window W_return. Fix candidate selection, the replacement, update
budget and window schedule before scoring. Freeze the chosen ranking before
opening validation outcomes. The primary object is transfer of the ranking:

    j_star = argmax_j v_j(W_score)
    gain_next = v_j_star(W_next) - mean_j v_j(W_next)
    gain_return = v_j_star(W_return) - mean_j v_j(W_return).

The means are the exact expected uniform choice over the same candidates; a
predeclared random draw is an alternative noisier control. Positive association
between rankings alone is insufficient if the gains are negligible. Report
held-out loss differences as well as rank association, dispersion across
candidates and uncertainty across independent seeds. A best-on-validation
candidate can diagnose available headroom but is an evaluator-only oracle.

With one common replacement, subtracting its loss cannot alter the ranking:

    argmax_j [L_W(S_r) - L_W(S_j)] = argmin_j L_W(S_j).

Those are the same selector, not two independent controls. If replacement
differs by candidate, comparability changes and must be justified explicitly.

## The simple accuracy control and the temporal confounds

The user's simple intuition deserves a direct control: choose the candidate
with the lowest packet prediction error recorded on its original arrival,
before training on it. This differs from the later accuracy of a shadow trained
with that packet. Keep uniform selection as the other basic control. Do not
replace original prequential error with current in-buffer error after rehearsal.

Original packet error depends on initial learner competence, packet age,
condition difficulty and reporting noise. Report these associations rather than
treating low error as an intrinsic property of the data. A future error-based
control should use the same packet candidates and the same intervention budget.

Immediate usefulness and rarity are separate. Under a simple mixture with a
rare condition of frequency q, a gain confined to that condition contributes
only q times its condition-specific gain to average prediction improvement.
It can therefore rank poorly while that condition is absent. This follows from
the objective's weighting; it does not make the rare packet worthless.

Report ordinary later observations and delayed returns separately, with fixed
return gaps and counterbalanced condition order. Inspect whether relevant
packets were present in the reservoir and candidate pool: an absent candidate
cannot be selected by any score. Privileged condition labels may describe that
coverage to the evaluator but must not select candidates or weight online scores.
If the main learner continues between assessments, training age, changed weights
and absence duration are intertwined. Call this temporal transport of value,
not a causal estimate of memory age. A causal age claim needs a separate design.

Matching reports can reward learning persistent corruption. Keep reported-loss
selection separate from evaluator-only truthful physical scores, and preserve
the [existing noise limitation](evidence_noise_limits.md). A fixed clean first
diagnostic cannot establish robustness to revision or corruption. Success here
would justify a later controlled policy test, not immediate memory eviction.

## Proposed audit invariants

These checks are recommendations for the forthcoming implementation, not a
claim that an audit already exists or can prove every update correct.

1. **Provenance and completeness:** lock configuration, protocol, source, runtime
   and analysis before main fitting; check every expected seed/model/assessment,
   input hash and completion record. Preserve the earlier sealed studies.
2. **Untouched parent trajectory:** assessment must leave the main learner's
   parameters, buffers, gradients, optimizer, memory, history and random state
   unchanged. Replay membership must still reconstruct as ordinary uniform
   reservoir sampling from the causal input stream.
3. **Candidate provenance:** every candidate/replacement ID and stored packet
   must match the parent reservoir at the stated cutoff. Reconstruct the
   predeclared candidate draw and tie-breaking without future targets or labels.
4. **Exact independent forks:** before any intervention, all shadows must match
   the entire saved parent state, including Adam moments and step counters.
   A mutation test should catch tensor/storage or random-generator aliasing.
5. **Matched intervention:** verify K, current anchors if any, support/query
   records, loss weights, update order and optimizer settings. Only the declared
   rehearsal choice may differ. Count extra updates and scoring work separately.
6. **Causal scoring:** query and support hashes must reconstruct from observations
   available by each prediction time. Freeze model/optimizer state through all
   score and validation windows; allow only the explicitly declared causal
   context progression. No privileged law label or truthful target enters scores.
7. **Temporal separation:** fork-training IDs precede scoring, ranking is fixed
   before validation, and score/next/return query IDs are disjoint. Future-label
   perturbation tests must leave earlier forks and selections unchanged.
8. **Authentic accuracy baseline:** persist original pre-update forecasts and
   reported targets, with hashes and arrival positions; independently recompute
   packet Brier error. Current or post-rehearsal errors cannot substitute for it.
9. **Independent arithmetic:** rescore stored predictions, reconstruct selected
   IDs, ties, uniform expectations and all contrasts. Apply the common-baseline
   ranking identity as an invariant. Resample seeds, not dependent assessments.
10. **Honest costs and limits:** record candidate count, live shadow count,
    copied optimizer/model bytes, update/scoring presentations, timings and
    persistent score storage. Distinguish checkpoint/hash checks, selected-fork
    re-execution and full independent retraining; claim only what was performed.

The important distinction is between a packet being easy to predict, rehearsal
being useful now, and that usefulness surviving a change in conditions. This
diagnostic should measure those separately before adding an adaptive policy.
