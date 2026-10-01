# Literature scan for a new continual-learning design

**Status: POST HOC / EXPLORATORY / NOT DECISION-BEARING.**
This is a literature scan prepared 2026-09-28. It runs no learner, trains no
model, reads no checkpoint and changes no locked protocol, threshold or
decision. It is not an exhaustive novelty review. A statement that a combination
is "untested" means only that this scan did not find it. Any idea below
needs its own prospective protocol before it can bear on a decision.

## Setting this scan targets

The project's learners must:

1. infer an unannounced, recurring hidden context from sparse outcome feedback:
   each record supplies only the performed (randomized) action's three nested
   binary survival labels (horizons 4, 8, 12);
2. acquire new cue-to-consequence dependencies later in life;
3. retain old knowledge, or recover it quickly (recoverability is valued over
   perfect retention), under bounded memory and compute and with no task labels.

Documented failures: a monolithic recurrent learner loses minority contexts;
weak-effect dependencies were not identifiable from sparse feedback; replay
selection heuristics, frozen cores, projection and delayed consolidation all
failed; the five-law interleaved reference failed qualification. The current
conditional learner already uses a fixed set of four heads. It weights them by
their log-likelihood of terminal survival over a fixed 32-record past window,
with a uniform prior and no sticky prior
([docs/conditional_relationships.md](../../docs/conditional_relationships.md)).

Law identities, clean physical outcomes and counterfactual outcomes are
evaluator-only. Any "ideal observer" below is an analysis reference and never a
learner input.

## Verification method

Each main entry was opened at a primary or bibliographic source during this
scan: the arXiv abstract page or API, the PMLR or JMLR page, a Europe PMC record
for journal articles (nature.com and Springer redirected to login), a
conference talk page, or the author lab's page. "Verified" means the title,
authors, year and abstract-level content were checked. It does not mean every
claim in the paper was reproduced. Where a venue came only from a search result,
the entry says so. One classic could not be opened and is marked `verified=false`.

## Main list (35 works)

### A. Latent context and change inference

**A1. Contextual inference underlies the learning of sensorimotor repertoires**
Heald, Lengyel, Wolpert. *Nature* 600:489-493, 2021 (COIN).
<https://www.nature.com/articles/s41586-021-04129-3>. Verified: abstract through Europe PMC.
*Relevance:* This is the closest normative theory to the project's
recurring-hidden-context setting. Savings and spontaneous recovery arise from
inference about which stored context is active, not only from parameter change.
*Borrowable idea:* Separate expression, meaning the posterior over which stored
memory applies, from updating, where each memory changes in proportion to its
responsibility. New contexts come from a nonparametric prior. At return, measure
apparent recovery (a posterior shift) separately from proper relearning (a
parameter change).

**A2. Context, learning, and extinction**
Gershman, Blei, Niv. *Psychological Review* 117(1):197-209, 2010.
<https://doi.org/10.1037/a0017808>. Verified: Europe PMC.
*Relevance:* In latent-cause inference, extinction creates a new cause instead
of erasing the old association. This produces renewal. Limited ability to posit
new causes reduces context dependence.
*Borrowable idea:* Treat an apparent reversal as evidence for a new latent cause,
not as an overwrite. The propensity to create a new cause (the concentration
parameter) should be a pre-registered design variable. Renewal-like recovery can
serve as a diagnostic signature.

**A3. Continuous Meta-Learning without Tasks (MOCA)**
Harrison, Sharma, Finn, Pavone. NeurIPS 2020.
<https://arxiv.org/abs/1912.08866>. Verified.
*Relevance:* This is differentiable Bayesian online changepoint detection over
run length wrapped around a meta-learner. It needs no task segmentation.
*Borrowable idea:* Replace a fixed evidence window, such as the current
32 records, with a run-length posterior under a hazard prior.
*Caveat:* MOCA resets to the prior after a change and does not recall earlier
tasks, so recurrence requires an added library.

**A4. A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning (CN-DPM)**
Lee, Ha, Zhang, Kim. ICLR 2020. <https://arxiv.org/abs/2001.00689>. Verified.
*Relevance:* Experts expand without task boundaries under a Dirichlet-process
mixture.
*Borrowable idea:* Use evidence-gated growth. Poorly explained records
accumulate in a short-term buffer, and a new expert is allocated only after
enough evidence has built up. Existing experts are therefore not overwritten by
a new regime.
*Caveat:* Assignment uses p(x). In this project, identical inputs can have
opposite consequences, so assignment must use the outcome likelihood.

**A5. Deep Online Learning via Meta-Learning: Continual Adaptation for Model-Based RL (MOLe)**
Nagabandi, Finn, Levine. ICLR 2019; the venue is confirmed by search.
<https://arxiv.org/abs/1812.07671>. Verified: arXiv abstract.
*Relevance:* A Chinese-restaurant-process mixture of meta-learned models creates
new models and recalls old ones when earlier conditions recur.
*Borrowable idea:* Use online EM with a CRP prior. The E-step computes the
posterior over existing models plus a new one. The M-step applies SGD weighted
by that posterior. This is a practical recipe for recall versus instantiation.

**A6. Reinforcement Learning in Presence of Discrete Markovian Context Evolution**
Ren, Sootla, Jafferjee, Shen, Wang, Bou-Ammar. ICLR 2022.
<https://arxiv.org/abs/2202.06557>. Verified.
*Relevance:* The paper handles an unknown finite number of unobserved contexts
with abrupt Markovian switching, using a sticky hierarchical Dirichlet process
(HDP) prior.
*Borrowable idea:* Use a sticky self-transition prior because contexts persist.
Add a distillation step that prunes spurious contexts. Report both failure
directions: a minority context merged into another context, and a spurious
split.

**A7. Thalamus: a brain-inspired algorithm for biologically-plausible continual learning and disentangled representations**
Hummos. ICLR 2023. <https://arxiv.org/abs/2205.11713>. Verified: abstract and method section.
*Relevance:* On an accuracy drop, the method first runs gradient descent on a
low-dimensional latent task embedding with weights frozen. It returns to weight
updates only if latent inference fails.
*Borrowable idea:* Infer before learning. This gives an explicit
recover-versus-acquire decision: retrieval through a cheap latent search, and
new learning only when retrieval fails.

**A8. A Mixture of Delta-Rules Approximation to Bayesian Inference in Change-Point Problems**
Wilson, Nassar, Gold. *PLoS Computational Biology* 9(7):e1003150, 2013.
<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003150>. Verified.
*Relevance:* Two or three delta rules with fixed learning rates nearly match
full Bayesian change-point inference.
*Borrowable idea:* This bounded approximation to the ideal observer can serve as
a cheap baseline learner and as a calibration reference for how quickly
re-identification is possible at a given hazard rate.

### B. Loss of plasticity and its remedies

**B1. Loss of plasticity in deep continual learning**
Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton. *Nature* 632:768-774, 2024.
<https://www.nature.com/articles/s41586-024-07711-7>. Verified: Europe PMC.
*Borrowable idea:* Continual backpropagation reinitializes a small fraction of
low-utility units, supplying a non-gradient source of variability. Plasticity
should be evaluated over long sequences of matched acquisition problems, not one
novelty event.

**B2. Disentangling the Causes of Plasticity Loss in Neural Networks**
Lyle, Zheng, Khetarpal, van Hasselt, Pascanu, Martens, Dabney. CoLLAs 2024
(PMLR 274, per search). <https://arxiv.org/abs/2402.18762>. Verified: arXiv abstract.
*Borrowable idea:* Plasticity loss has several independent mechanisms, and
fixing one is not enough. Layer normalization plus weight decay is robust.
Diagnose with separate probes: pre-activation shift, parameter norm and target
scale.

**B3. Maintaining Plasticity in Continual Learning via Regenerative Regularization (L2 Init)**
Kumar, Marklund, Van Roy. CoLLAs 2024 (per search). <https://arxiv.org/abs/2308.11958>. Verified.
*Borrowable idea:* Regularize toward the initial parameters. This needs only one
hyperparameter. A natural untested variant regularizes toward a retrieved
context's anchor instead of the initialization.

**B4. On Warm-Starting Neural Network Training (shrink-and-perturb)**
Ash, Adams. NeurIPS 2020. <https://arxiv.org/abs/1910.08475>. Verified.
*Borrowable idea:* Apply theta <- lambda*theta + sigma*epsilon when warm-starting.
This restores the generalization of fresh initialization while keeping most of
the speed benefit.

**B5. Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning (UPGD)**
Elsayed, Mahmood. ICLR 2024. <https://arxiv.org/abs/2404.00781>. Verified.
*Borrowable idea:* Scale each unit's update and injected noise by its estimated
utility. Useful units are protected while unused units are perturbed. This
addresses retention and plasticity together in a streaming setting.

**B6. Non-Stationary Learning of Neural Networks with Automatic Soft Parameter Reset**
Galashov, Titsias, Gyorgy, Lyle, Pascanu, Teh, Sahani. NeurIPS 2024.
<https://arxiv.org/abs/2411.04034>. Verified.
*Borrowable idea:* An Ornstein-Uhlenbeck drift toward initialization uses a
drift parameter inferred from the data. The amount of reset follows inferred
non-stationarity instead of a fixed schedule. It is the closest existing link
between change inference and plasticity control.

### C. Recovery, savings and forgetting as a feature

**C1. Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training**
Yang, Jones, Mozer, Ren. NeurIPS 2024. <https://arxiv.org/abs/2403.09613>. Verified: abstract and HTML.
*Relevance:* Under fixed cyclic re-exposure, over-parameterized networks recover
on items before seeing them again. The effect disappears when order is shuffled
and grows with capacity and with more steps per item.
*Borrowable idea:* Measure savings and recovery before return under recurrence.
Always include a shuffled-order control. Recurrence structure can be exploited
implicitly by shared slow weights.

**C2. The Persistence and Transience of Memory**
Richards, Frankland. *Neuron* 94:1071-1084, 2017. Verified: Europe PMC.
<https://doi.org/10.1016/j.neuron.2017.04.037>.
*Borrowable idea:* Forgetting serves decisions by reducing the influence of
outdated information and preventing overfitting to specific episodes. Judge
memory by its contribution to future decisions, not by retention itself.

**C3. Fortuitous Forgetting in Connectionist Networks**
Zhou, Vani, Larochelle, Courville. ICLR 2022. <https://arxiv.org/abs/2202.00155>. Verified.
*Borrowable idea:* In forget-and-relearn, targeted forgetting followed by
relearning amplifies consistently useful features. Relearning speed then acts
as a signal of feature quality.

**C4. Continual Learning as Computationally Constrained Reinforcement Learning**
Kumar, Marklund, Rao, Zhu, Jeon, Liu, Van Roy. arXiv 2023, v3 2025.
<https://arxiv.org/abs/2307.04345>. Verified: abstract and HTML v3.
*Relevance:* The paper's objective is infinite-horizon average reward under a
per-step compute constraint. It states that forgetting non-recurring
information is not catastrophic, and that forgetting recurring information is
acceptable if relearning is fast. This formalizes the user's preference for
recoverable over perfectly retained knowledge.
*Borrowable idea:* Adopt a lifetime objective, average survival or decision
quality under fixed compute and memory. Treat retention as instrumental and
account for the information capacity of agent state.

### D. Decision-aware models, value of information and partial feedback

**D1. The Value Equivalence Principle for Model-Based Reinforcement Learning**
Grimm, Barreto, Singh, Silver. NeurIPS 2020. <https://arxiv.org/abs/2011.03506>. Verified.
*Borrowable idea:* A model needs to be correct only for the policies and values
that matter. Define decision-relevant queries, such as which action maximizes
12-step survival in each context, and evaluate or learn equivalence on those
rather than on full Brier score.

**D2. Value-Aware Loss Function for Model-based Reinforcement Learning (VAML)**
Farahmand, Barreto, Nikovski. AISTATS 2017, PMLR 54.
<https://proceedings.mlr.press/v54/farahmand17a.html>. Verified.
*Borrowable idea:* Weight model error by its effect on value. Errors in cells
that cannot change the chosen action cost little, and errors near decision
boundaries cost a lot.

**D3. Learning to Optimize via Information-Directed Sampling**
Russo, Van Roy. arXiv 2014; the Operations Research 2018 journal version was not
checked. <https://arxiv.org/abs/1403.5556>. Verified: arXiv.
*Borrowable idea:* Choose actions by the ratio of squared regret to information
gain. When the context posterior is uncertain, the performed action should be
the most diagnostic one. Performed actions are currently randomized, so this
would be a deliberate protocol change.

**D4. Batch Learning from Logged Bandit Feedback through Counterfactual Risk Minimization**
Swaminathan, Joachims. *JMLR* 16:1731-1755, 2015.
<https://jmlr.org/papers/v16/swaminathan15a.html>. Verified.
*Relevance:* This is the formal theory for learning when only the chosen
action's outcome is observed, the project's feedback structure.
*Borrowable idea:* Record the propensities of performed actions. Use
propensity-weighted, variance-regularized risk. With known uniform propensities,
counterfactual dependencies are identifiable in expectation. The practical limit
is variance, meaning sample size per cell, not bias. This gives a way to predict
in advance which weak effects are learnable.

### E. Factored, modular and relational structure

**E1. Recurrent Independent Mechanisms**
Goyal, Lamb, Hoffmann, Sodhani, Levine, Bengio, Scholkopf. ICLR 2021.
<https://arxiv.org/abs/1909.10893>. Verified: arXiv abstract; the venue was confirmed by ICLR listing search.
*Borrowable idea:* Top-k attention competition means only the relevant modules
update at each step. Sparse updates shield inactive modules, such as those for
minority contexts, from interference. This targets the monolithic learner's
minority-context loss.

**E2. Neural Relational Inference for Interacting Systems**
Kipf, Fetaya, Wang, Welling, Zemel. ICML 2018. <https://arxiv.org/abs/1802.04687>. Verified.
*Borrowable idea:* Infer a discrete latent interaction graph with a VAE. Here,
each cue-to-consequence dependency could be a latent edge, whether present,
absent or context-gated, rather than a dense weight.

**E3. Modular meta-learning**
Alet, Lozano-Perez, Kaelbling. CoRL 2018. <https://arxiv.org/abs/1806.10166>. Verified.
*Borrowable idea:* Learn a library of small modules and search over their
compositions for new conditions. A new context may then be a recombination of
known mechanisms rather than a new monolith.

**E4. Continual learning with hypernetworks**
von Oswald, Henning, Grewe, Sacramento. ICLR 2020. <https://arxiv.org/abs/1906.00695>. Verified: abstract.
*Borrowable idea:* Store only a low-dimensional embedding per context and let a
hypernetwork generate weights. Retention becomes cheap rehearsal in weight
space: an output regularizer on the weights generated for old embeddings.
Recovery means re-finding an embedding. The paper also reports task-inference
variants; those details were not re-checked in this scan.

**E5. Continual Learning via Local Module Composition (LMC)**
Ostapenko, Rodriguez, Caccia, Charlin. NeurIPS 2021. <https://arxiv.org/abs/2111.07736>. Verified.
*Borrowable idea:* Each module has a local structural component that estimates
whether an input belongs to it. This gives task-agnostic routing and module
addition. Performance degrades on very long task sequences (30-100), which is an
honest limitation to plan around.

### F. World models and predictive self-supervision

**F1. The Effectiveness of World Models for Continual Reinforcement Learning (Continual-Dreamer)**
Kessler, Ostaszewski, Bortkiewicz, Zarski, Wolczyk, Parker-Holder, Roberts, Milos.
CoLLAs 2023. <https://arxiv.org/abs/2211.15944>. Verified: abstract and HTML.
*Borrowable idea:* A task-agnostic world model with reservoir replay.
Reservoir (uniform) sampling beat uncertainty-, reward- and coverage-based
selection. That independently corroborates this project's finding that uniform
replay remains the reference.

**F2. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)**
Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas. CVPR 2023, pp. 15619-15629.
<https://arxiv.org/abs/2301.08243>. Verified. The arXiv comment names the 2023
International Conference on Computer Vision, but the CVPR 2023 proceedings list
the paper.
*Borrowable idea:* Predict consequences in a learned latent space instead of
raw input space. That gives dense self-supervised targets without having to
reconstruct irrelevant detail. This is relevant because the stopped
representation-learning branch planned reconstruction targets.

### G. Continual-RL formulations

**G1. The Big World Hypothesis and its Ramifications for Artificial Intelligence**
Javed, Sutton. Finding the Frame Workshop, RLC 2024.
<https://oaklab.ai/posts/the-big-world-hypothesis>. Verified: the Oak Lab page;
the incompleteideas.net PDF failed a TLS check.
*Borrowable idea:* When the agent is much smaller than the world, tracking beats
converging. Benchmark by restricting agent capacity rather than by enlarging the
environment, which fits the project's bounded-memory emphasis.

**G2. The Alberta Plan for AI Research, with the OaK architecture**
Sutton, Bowling, Pilarski. arXiv 2022. <https://arxiv.org/abs/2208.11173>. Verified.
Sutton's NeurIPS 2025 invited talk, "The Oak Architecture", is also verified:
<https://neurips.cc/virtual/2025/invited-talk/109601>.
*Borrowable idea:* Give every weight its own meta-learned step size (the talk
names online cross-validation) and construct features continually. OaK's
FC-STOMP progression (feature, subtask, option, model, planning) ties new
features to predictions that are useful for control.

**G3. Streaming Deep Reinforcement Learning Finally Works (stream-x)**
Elsayed, Lupu, Vasan, Mahmood. arXiv 2024, revised 2026.
<https://arxiv.org/abs/2410.14606>. Verified: abstract and HTML.
*Borrowable idea:* Learn from the stream without replay using observation
normalization, reward scaling, LayerNorm, sparse initialization and bounded
per-coordinate updates. This supplies a no-replay, bounded-memory baseline
recipe.

### H. Ideal-observer calibration and in-context or meta-learned continual learning

**H1. Meta-trained agents implement Bayes-optimal agents**
Mikulik, Deletang, McGrath, Genewein, Martic, Legg, Ortega. NeurIPS 2020.
<https://arxiv.org/abs/2010.11223>. Verified.
*Borrowable idea:* Bayes-optimal agents are fixed points of memory-based
meta-learning. Compare a learner with the Bayes-optimal observer both in
behaviour and in state. Here the Bayes-optimal observer is the calibration
ceiling and is evaluator-only.

**H2. Metalearning Continual Learning Algorithms (ACL)**
Irie, Csordas, Schmidhuber. TMLR 2025. <https://arxiv.org/abs/2312.00276>. Verified.
*Borrowable idea:* Put old-task performance explicitly into the meta-objective.
Naive in-context learners show "in-context catastrophic forgetting", so a
sequence model does not solve continual learning by default.

**H3. Learning to Continually Learn with the Bayesian Principle (SB-MCL)**
Lee, Jeon, Son, Kim. ICML 2024. <https://arxiv.org/abs/2405.18758>. Verified: abstract.
*Borrowable idea:* Continual learning happens only through exact sequential
Bayesian (conjugate) updates of a statistical model on fixed neural features.
There is no forgetting within that layer, and memory is constant. For this
project that maps to Beta-Bernoulli or discrete-hazard sufficient statistics per
context, action and horizon.
*Caveat:* The features are meta-trained across episodes. That conflicts with
from-scratch learning unless the features are learned in-lifetime.

## Also opened (not counted in the 35)

These were verified but omitted for space or redundancy:

- Li, Ding, Hu, *Understanding Generalization and Forgetting in In-Context
  Continual Learning*, ICML 2026, <https://arxiv.org/abs/2605.28705>: attention
  aggregation provably causes inter-task interference in in-context continual
  inference.
- Abel et al., *A Definition of Continual RL*, NeurIPS 2023,
  <https://arxiv.org/abs/2307.11046>.
- Tang et al., *Forager*, 2026, <https://arxiv.org/abs/2605.01131>: in a
  partial-observability continual-RL testbed, state construction helped more
  than plasticity fixes.
- Liu and Mou, *Do Neural Networks Lose Plasticity in a Gradually Changing
  World?*, 2026, <https://arxiv.org/abs/2602.09234>: gradual transitions largely
  mitigate plasticity loss.
- Lyle et al., *Understanding plasticity in neural networks*, ICML 2023,
  <https://arxiv.org/abs/2303.01486>.
- Davari et al., *Probing Representation Forgetting*, CVPR 2022,
  <https://arxiv.org/abs/2203.13381>.
- Zheng et al., *Spurious Forgetting*, ICLR 2025,
  <https://arxiv.org/abs/2501.13453>: output-level forgetting can hide
  recoverable knowledge.
- Jerfel et al., *Online mixtures of tasks*, 2018/2019,
  <https://arxiv.org/abs/1812.06080>.
- Rao et al., *CURL*, NeurIPS 2019, <https://arxiv.org/abs/1910.14481>.
- Ortega et al., *Meta-learning of Sequential Strategies*, 2019,
  <https://arxiv.org/abs/1905.03030>.
- Kirsch et al., 2022, <https://arxiv.org/abs/2212.04458>.
- Laskin et al., *Algorithm Distillation*, 2022,
  <https://arxiv.org/abs/2210.14215>.
- Lee, Son, Kim, *Recasting CL as Sequence Modeling*, NeurIPS 2023,
  <https://arxiv.org/abs/2310.11952>.
- Behrens et al., *Nature Neuroscience* 10:1214-1221, 2007 (volatility-adaptive
  learning rates).
- Geisler, *Vision Research* 51:771-781, 2011 (ideal-observer theory).
- Mandi et al., *Decision-Focused Learning*, JAIR 2024,
  <https://arxiv.org/abs/2307.13565>.
- Fini et al., CaSSLe, CVPR 2022, <https://arxiv.org/abs/2112.04215>, already
  cited in `docs/representation_learning_design.md`.

**Not verified:** Widmer and Kubat, *Learning in the Presence of Concept Drift
and Hidden Contexts*, Machine Learning 23:69-101, 1996,
<https://doi.org/10.1023/A:1018046501280>. This is the classic on recurring
hidden contexts and reusing stored concepts. The Springer and dblp pages were
blocked, and the metadata comes only from search results (`verified=false`).

## Synthesis

### 1. What is established

- **Recurrence plus context inference produces savings.** COIN (A1) and
  latent-cause models (A2) explain savings, spontaneous recovery and renewal by
  inferring which stored context applies. This inference happens alongside
  updating a context's parameters. Machine-learning versions exist for dense
  observations: CRP/DPM mixtures with instantiation or recall (A4, A5),
  run-length changepoint meta-learning (A3), and sticky-HDP contexts (A6). All
  assume the context is identifiable from dense inputs or dynamics. None studies
  identification from a single randomized action's nested binary outcomes.
- **Bounded approximations suffice for change-point inference.** Two or three
  learning-rate nodes nearly match full Bayesian inference (A8).
  Memory-based meta-learners converge towards Bayes-optimal behaviour (H1). An
  ideal observer is therefore a legitimate, computable calibration ceiling for
  evaluators.
- **Plasticity loss is real and has several causes.** Remedies with
  replicated support include utility-based reinitialization (B1),
  LayerNorm plus weight decay (B2), L2-to-init (B3), shrink-and-perturb (B4),
  utility-gated perturbation (B5) and inferred-drift soft reset (B6). Most
  evidence comes from dense-label supervision or RL with long task sequences,
  not from sparse bandit feedback.
- **Retention is instrumental.** Under constraints, forgetting non-recurring
  information is not harmful, and fast relearning can replace retention (C2, C4).
  Output-level forgetting can hide recoverable knowledge (C1, and the secondary
  Zheng and Davari entries).
- **Decision-aware modelling exists.** Models need to be accurate only where
  decisions depend on them (D1, D2). With known logging propensities, bandit
  feedback identifies counterfactual effects in expectation, and variance then
  sets the sample size needed (D4).
- **Uniform replay is a strong reference.** A world-model continual-RL study
  independently found reservoir sampling better than uncertainty-, reward- and
  coverage-based selection (F1). That matches the project's failed selection
  heuristics.
- **Sequence models are not automatically continual learners.** In-context
  learners suffer in-context forgetting unless old-task performance is in the
  meta-objective (H2). Attention aggregation causes interference (Li et al.
  2026).

### 2. Where the current conditional learner sits relative to this literature

The existing conditional model is already a COIN-like evidence mixture with
four fixed heads. It differs from the literature in five ways:

- The prior is fixed and uniform, and the window is 32 records, whereas the
  literature uses a sticky or run-length prior (A3, A6, A8).
- The number of heads is fixed; there is no evidence-gated allocation or
  pruning (A4, A5, A6).
- Only terminal survival enters the context likelihood. The full nested outcome
  can be scored without double counting as a discrete failure-time category
  (failed by step 4, 5-8, 9-12, or survived), as in a discrete-time hazard
  model.
- Its component predictions are uncalibrated, so the weights are not calibrated
  beliefs.
- There is no latent-first retrieval step before weights change (A7).

None of these differences has been tested in this project. They are candidates
only.

### 3. Combinations this scan did not find tested

- **(i) Consequence-only context identification under bandit feedback with a
  growing, sticky context library.** A4 to A6 assign contexts by p(x) or by
  dense dynamics. The project's hard case has identical inputs with opposite
  consequences, observed through one randomized action and three nested binary
  labels.
- **(ii) Coupling plasticity control to the inferred context posterior.** B6
  couples reset strength to inferred non-stationarity, but always toward
  initialization. Resetting or anchoring toward a retrieved context's stored
  parameters, a context-anchored variant of L2 Init, was not found. Neither was
  gating plasticity remedies to the shared trunk while keeping per-context
  statistics exact (H3-style).
- **(iii) Value-equivalence as a retention criterion in continual learning.** D1
  and D2 weight model error by value within model-based RL. Using decision
  relevance, meaning whether a dependency changes the survival-maximizing action
  in some plausible context, to decide what must stay recoverable was not found.
  The project's earlier replay-selection failures argue for testing this first
  as an evaluation lens on saved forecasts, not as a new selector.
- **(iv) Ideal-observer information budgets for continual-learning benchmarks.**
  Ideal-observer analysis is standard in perception (Geisler, secondary list).
  Bayes-optimal comparisons exist for meta-learning (H1). A per-transition
  calculation of the evidence available to re-identify a context was not found
  as a routine continual-learning benchmark calibration. It would separate
  information-limited failures, where even the ideal observer cannot
  re-identify within the window, from learning-limited failures.

### 4. Candidate novel yet falsifiable contributions

These are ranked by cost and by how strongly each could falsify a design idea.
All are post hoc proposals and need a prospective protocol.

**P1. Zero-training information-budget audit (evaluator-only).**
- *Method:* For each recurring transition in existing frozen streams, compute
  with the evaluator-only laws the expected per-record log Bayes factor between
  the true and competing contexts. This is the KL divergence between their
  outcome distributions, averaged over inputs and the uniform action
  propensity. Do it twice: once using terminal survival only, once using the
  4-category failure time. Convert to Wald-style records-to-identify at a fixed
  error rate, then compare with the learners' observed return-recovery latency.
- *Falsifiable prediction:* Failures such as the interleaved Base-A endpoint
  and the negligible arrival-delay cue benefit fall into two groups. In
  information-limited cells, the ideal observer also needs more than 32 records
  or than the available exposure. In learning-limited cells, the ideal observer
  identifies the context quickly and the learner does not.
- *Why it matters:* If most failures are information-limited, no architecture
  change can fix them, and the benchmark or feedback design must change first.
- *Rules:* No law identity enters any learner.

**P2. Consequence-identified, evidence-gated context library ("infer, then
recall, then grow").**
- *Design:* Keep a shared, continually trained trunk with plasticity
  maintenance applied to the trunk only (for example LayerNorm plus weight decay
  or UPGD). On top of it, keep a bounded library of per-context heads or exact
  conjugate discrete-hazard statistics (H3).
- *Inference:* A sticky or run-length posterior using the failure-time
  likelihood (A3, A6).
- *Updates:* Responsibility-weighted updates (A1, A5).
- *Retrieval:* Latent-first retrieval before any weight update (A7).
- *Growth:* A new context is allocated only when a buffered Bayes factor
  exceeds a pre-registered threshold (A4). Pruning follows A6.
- *Falsifier:* At matched memory, compute and exposure, compare against the
  monolithic recurrent reference and the fixed-K conditional learner, both with
  uniform replay. P2 fails unless ideal-observer-normalized return-recovery
  regret improves in at least 5 of 6 seeds, minority-context availability
  improves, and new-dependency acquisition is not worse.
- *Pre-registered failure modes:* Over-splitting, and merged minority contexts.

**P3. Decision-equivalent retention lens.**
- *Method:* Score existing saved forecasts, with no training, by
  survival-decision regret under each context as well as by Brier score.
- *Hypothesis:* Some recorded "retention failures" are Brier losses on
  decision-irrelevant cells, while the real harms concentrate near decision
  boundaries.
- *Follow-up:* Only if this lens changes the ranking of the failure modes
  should a VAML-weighted objective (D2) be proposed. It would then be compared
  against uniform replay, not against another selector.

### 5. What the literature suggests not to repeat

- Replay-selection heuristics: F1 and the project's own record both favour
  uniform replay.
- Frozen or projected cores: B1 and B2 show plasticity loss occurs in trainable
  networks too. Freezing addresses neither identification nor acquisition.
- Transformer or in-context learners as a drop-in fix, without an explicit
  old-task objective (H2, Li et al. 2026).
- Any claim of novelty for evidence mixtures as such, because A1 to A6 predate
  it. The defensible contribution is the sparse-consequence identification
  regime and its ideal-observer calibration.
