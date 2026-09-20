# ACP experiment protocol

Status: **development protocol; not preregistered and not a completed confirmatory study**. Date: 2026-09-20. The current local implementation and its quick runs are exploratory. Before confirmatory work, freeze this document, exact configuration files, implementation commit, data manifests, selection rules, and analysis code together. Planned experiments below are not claims about capabilities already implemented.

The scientific question is whether adaptive control adds a retention–acquisition advantage beyond replay plus low-utility recycling. [research_review.md](research_review.md) gives the evidence audit and corrections to the supplied report. No clinical or biological mechanism is being tested directly.

The shipped prototype provides fine-tuning, ER, CBP-inspired recycling, a DER++ variant, ACP, and component ablations. It does not reproduce EWC, SI, GEM, or UPGD. Its ResNet has a shared Linear/ReLU adapter, with recycling limited to that adapter; its shared head remains trainable. The optimizer and importance statistic are explicitly modified versions described in [research_review.md](research_review.md#prototype-scope-and-differences-from-the-report). The following split, tuning, resource, statistical, and replication requirements are the **planned full study contract**, including requirements not yet enforced by the exploratory runner.

## 1. Claims and primary comparisons

The primary methods are:

| Method | Purpose | Required behavior |
|---|---|---|
| Fine-tuning | Interference reference | Same model, current stream only; no replay. Its information budget differs intentionally. |
| ER | Primary retention baseline | Identical bounded reservoir and replay draw policy to ACP. |
| ER + CBP-style recycling | Primary combined baseline | Identical model/replay/optimizer; documented utility, maturity, and replacement rule, without ACP consolidation or controller. |
| ACP | Candidate | Deterministic online controller, local update gains, anchors/consolidation, and selective recycling. |
| ACP with fixed controller | Attribution control | Same components and monitor instrumentation; initial open/closing schedule followed by fixed adult gates, with no novelty-triggered reopening. |

“CBP-style” is intentional until the adaptation has been checked against the [original method](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/). A plain ER run is not DER++. Add a verified [DER++](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) baseline and [UPGD](https://arxiv.org/abs/2404.00781) before a broad publication-level state-of-the-art claim. Store DER++ logits within its reported memory cost.

Three claims are tested separately:

1. **Performance:** ACP improves final retained accuracy relative to ER and ER+CBP-style while late acquisition remains within the noninferiority margin below.
2. **Attribution:** adaptivity improves the retention–acquisition tradeoff over the fixed-controller method, and locality/consolidation/recycling ablations behave consistently with their proposed roles.
3. **Efficiency:** an advantage survives a common compute budget. Equal exposure alone supports only the first two claims.

No primary claim concerns indefinite learning, a biologically faithful mechanism, or universal superiority.

## 2. Phased execution

| Phase | Work | Exit evidence |
|---|---|---|
| 0: correctness | Unit tests and deterministic synthetic runs; exercise every controller state and ablation. | Correct replay sampling/insertion; no evaluation leakage; state/reset/gating invariants; finite metrics; identical paired streams. |
| 1: feasibility | Small synthetic streams, then a reduced CIFAR pilot; at least three paired development seeds. | Curves show ordinary learning; ER retains more than fine-tuning in a suitable setting; measured runtime/memory; monitoring detects injected shifts without permanent opening. |
| 2: development | Full CIFAR-100 configuration on held-out validation images and development seed tuples; matched tuning budgets. | Frozen hyperparameters, resource policy, endpoint implementation, seed manifest, and statistical analysis. |
| 3: confirmatory | Full CIFAR ResNet-18 study, 20 previously unused paired tuples; mandatory core methods and key ablations. | Complete run artifacts, paired confidence intervals, prespecified tests, sensitivity checks, and a negative-results report if applicable. |
| 4: replication | Fixed-class/domain long streams, then CORe50 and Infinite dSprites. | Replication with frozen transfer policy or explicitly separate retuning; long-horizon acquisition tests. |

Phase 0/1 results must be described as software and feasibility evidence. Do not use their synthetic accuracy gains as support for the critical-period hypothesis. Measure a few real full-size runs before assigning GPU-hour or completion-date estimates.

## 3. Dataset and information contract

The main study uses CIFAR-100 with ten disjoint experiences of ten classes. For each class, split the official 500 training images into 450 training and 50 validation images using a frozen index manifest. Keep the official 100 test images per class untouched during development. The resulting training stream contains 45,000 distinct training images; validation contains 5,000; confirmatory evaluation contains 10,000.

Use a CIFAR-adapted ResNet-18 with a 3x3, stride-one stem, no initial max-pool, a Linear/ReLU adapter of frozen width, and **one fixed 100-logit classifier**. Every method predicts among all 100 outputs from the start. There are no inference task identifiers, per-task heads, or per-experience logit masks. A separate seen-class-only analysis, if desired, must be labeled and cannot replace the primary metric. Fix the normalization choice across all methods; GroupNorm is an initial candidate, not an established plasticity remedy. The prototype recycles adapter units only; changing to backbone-channel recycling requires a new explicitly reviewed architecture variant.

Only the runner/evaluator knows experience boundaries. The learner consumes `(x, y)` and maintains its own running state. It cannot use the experience index, future class order, next-boundary time, validation accuracy, test accuracy, or test features for gating, anchor updates, recycling, early stopping, or sample selection. A fixed number of initial warm-up steps is allowed and must be declared. Learning-rate or optimizer resets at later boundaries are prohibited unless applied as an explicitly separate boundary-aware experiment.

For each replicate freeze independent random streams for class order, model initialization, data order, augmentations, replay insertion, replay draws, and recycling. Pair these streams across methods where their operations correspond. Logging, evaluation, scratch diagnostics, and extra monitoring must not consume training RNG state. Save sample IDs or deterministic hashes so stream pairing can be checked.

Replay is sampled **before** current examples are inserted. Use one reservoir policy across replay methods, capacity 2,000 raw images and labels. No balanced-vs-unbalanced reservoir substitution between methods. With repeated passes through an experience, insert each source image on its first arrival only; replay sampling can repeat it subsequently. If a pilot inserts repeated presentations instead, record this explicitly and do not compare it to the unique-arrival confirmatory policy. Never hide additional old images in monitoring sentinels, cached augmented tensors, or accessible dataset indices outside the stated buffer.

Monitor examples are drawn from the same replay budget. A stable sentinel is a subset of that buffer, not additional memory; its baseline is refreshed when membership changes. Statistics use training data only. Evaluation examples are never stored in the learner or replay buffer.

## 4. Initial training recipe and tuning

These are development starting values. They become confirmatory values only after they are frozen in a versioned manifest.

| Item | Starting choice |
|---|---|
| Optimizer | Shared SGD-style data momentum 0.9, base LR 0.05, separately applied scaled decay 1e-4; verify an ordinary SGD control during development |
| Current batch | 96 examples |
| Replay batch | Up to 32 examples, uniformly sampled from the reservoir |
| Exposure | Ten passes through each experience in the full development template; freeze the confirmatory exposure after development; no cross-experience retention except replay |
| Head | Fixed 100 outputs; prototype exempts it from gating and consolidation |
| Replay | 2,000 uint8 images plus labels and explicit metadata |
| Monitor cadence | Every 25 optimizer updates initially |
| Initial maturity | 250 updates initially; shorten only for named quick configurations |
| Adult gain | Final data-update multiplier floor 0.05 in the prototype; anchor force is separate |
| Consolidation | SI-inspired channel statistic with independent window origin and regularization anchor |
| Reopening | Sustained threshold over three monitor windows; bounded duration; at most one or two eligible blocks |
| Recycling | Fractional replacement budget, maturity filter, utility ranking, and full state reset |

The main learning objective must have the same scaling in all replay methods. Define it explicitly, for example `L = mean_CE(current) + lambda_replay * mean_CE(replay)`, with `lambda_replay = 1` initially. Concatenating 96+32 samples and taking a single mean gives a different weighting; do not silently mix the two conventions. When the buffer is empty, omit the replay term. Partial replay batches retain the declared mean-loss convention.

Tune common optimizer choices using all core methods, then tune method-specific controls with equal numbers of configurations and paired development streams. A concrete initial budget is at most 12 configurations per core method on three development tuples, then validate the selected configuration on five additional development tuples. This costs up to 180 small/full development runs before validation; reduce all methods' budgets together if the measured runtime is too large. Hyperparameter sweeps, failed runs, and manual revisions belong in the development ledger.

Selection rule: first select the ER+CBP-style configuration with highest validation final accuracy to define the development acquisition reference. For each method, among configurations whose mean late acquisition AUC is no more than 0.02 below that reference, choose the highest validation final accuracy; break ties within 0.002 by lower training compute, then a fixed lexical config order. Also report each method's unconstrained best validation accuracy so the constraint cannot hide a tradeoff. If no configuration satisfies the constraint, select the smallest acquisition deficit and mark the method infeasible under that tuning rule.

Do not search confirmatory class orders or select favorable test seeds. A changed algorithm after seeing confirmatory outcomes starts a new version and new confirmatory cohort.

## 5. Resource fairness

Maintain two distinct comparisons, because identical examples and identical optimizer steps do not guarantee identical computational cost.

**Equal-exposure mechanism comparison.** All replay methods receive the same current presentations, replay sample IDs, batch sizes, training steps, architecture, precision, augmentation, and stored replay capacity. Account for each extra current/replay gradient calculation, diagnostic forward, candidate update, rank calculation, and parameter-state tensor. Where shared monitoring can be executed identically without changing optimization, use it across methods. Do not pad a baseline with artificial idle work to claim efficiency.

**Common-compute comparison.** Profile native implementations on the same machine, fix an ex-ante compute allowance per experience using development runs, and compare performance-versus-training-compute curves up to that allowance. All methods receive the same current stream and bounded replay memory; extra optimization may only reuse already encountered data within the current experience or its legal replay buffer. A method may take extra steps when cheaper, so repeated presentations and replay exposures can differ; log them and label this difference. Alternatively throttle expensive controller computations to satisfy both exposure and compute ceilings. A simultaneous equal-exposure/equal-compute claim requires demonstrated equality, not merely a shared ceiling.

Report training FLOPs where measured, forward/backward example counts as a transparent proxy, and synchronized wall time with device/software details. A proxy is not a FLOP measurement. Evaluation and scratch-reference costs are reported separately from training cost. Report parameter count, peak device memory, replay bytes, optimizer bytes, anchor/state bytes, and checkpoint size. Same predictive model size and replay capacity do not mean same total learner memory.

Use accuracy–compute and retention–acquisition plots. A sampled set of nondominated points is an empirical frontier, not proof of dominance for all hyperparameters. Bootstrap the paired differences at prespecified selected operating points instead of selecting the best test point after looking at the plot.

## 6. Measurements and exact definitions

All accuracies below are fractions in `[0,1]`, with percentage points used when presenting differences. An experience's examples are evaluated against the full shared classifier. Let `R[i,j]` denote its evaluation accuracy after training experience `i`; there are `T=10` experiences. Report the matrix and class/sample counts.

### Retention

- Final accuracy: `ACC = mean_j R[T,j]`. With equal experience sizes this is also overall test accuracy.
- Old-experience accuracy: `OLD = mean_{j<T} R[T,j]`. This prevents strong performance only on the final experience from masquerading as retention.
- Backward transfer: `BWT = mean_{j<T}(R[T,j] - R[j,j])`.
- Forgetting: `F = mean_{j<T}(max_{i=j..T} R[i,j] - R[T,j])`. Use only the prespecified end-of-experience grid for the maximum; taking more maxima for one method creates bias. Including the final point keeps this measure nonnegative.
- Also report `R[j,j]` and minimum old-experience accuracy. Small forgetting without adequate acquisition is not success.

These definitions adapt the evaluation tradition described in [GEM](https://arxiv.org/abs/1706.08840) to the fixed-head contract above.

### Acquisition and scratch references

For experience `j`, let `A_j(n)` be its accuracy after `n` **current-example presentations**, excluding replay examples. Use a frozen evaluation grid at 0%, 2%, 5%, 10%, 20%, 50%, and 100% of that experience's declared current-presentation budget, rounding and deduplicating step indices before any run. Define early AUC by trapezoidal integration through the 20% point and divide by the number of presentations in that interval. Thus AUC remains in `[0,1]`. Log both presentation count and unique-example count; multiple epochs must not be called one-pass online learning.

The primary plasticity endpoint is `P_late = mean_{j=6..10} AUC_early,j`. Its comparison is paired **within the same experience and replicate**, not against experience 1. Report end-of-experience acquisition as well. A quick implementation with only start/end evaluation must label its two-point AUC as a coarse pilot approximation.

For each replicate and experience, train a fresh copy of the same architecture and 100-output classifier on the identical current batches, optimizer recipe, augmentation stream, and acquisition budget. It has no old replay and no prior training. This produces `AUC_scratch,j`. Use the same scratch run for all paired methods. Record `D_j = AUC_early,j - AUC_scratch,j`; report `D_late` and the per-experience curve. Scratch runs are diagnostics and their computation is separate from learner cost.

This controls much of the variation in current-task difficulty and output dimensionality. It does **not** isolate representation plasticity perfectly: a continual model also has old-class logits, replay gradients, and a retention obligation. A negative `D_j` can reflect that tradeoff. Do not claim pure plasticity loss without a fixed-class/domain-shift replication or another controlled acquisition diagnostic. Do not divide by first-task AUC, and do not use pretraining accuracy on semantically untrained classifier rows as decisive FWT evidence.

### Controller and representation diagnostics

Record controller state, transition reason, novelty components, selected modules, gate distributions, actual update norms, consolidation quantiles, fraction above 0.9, eligible/recycled counts, recycling disturbance, and cumulative open duration. Report behavior aligned to boundaries **afterward**; do not provide those boundaries to the learner.

Measure activation dormancy on a fixed-size, consistently sampled training/replay diagnostic set. Define the threshold before the run. For feature matrix `H` with `N` examples and `d` channels, explicitly state whether it is centered. For centered `H`, an entropy effective rank may be computed as `exp(-sum p_k log p_k)`, with `p_k = singular_value_k / sum singular_values`; normalize by `min(N-1,d)` and define rank zero when the denominator mass is zero. Stable rank `||H||_F^2 / ||H||_2^2` is a different statistic; name it separately. Keep sample count, layer, activation location, and preprocessing fixed. Rank and dormancy are explanatory measurements, not primary success endpoints.

## 7. Statistical analysis and decision rules

Generate **20 independent tuples** containing class order, initialization, and stochastic-stream seeds. Every core method receives each same tuple. The tuple is the independent sampling unit; tasks, minibatches, and checkpoints within a tuple are not independent replicates. Preserve run pairing in every bootstrap and test. A crossed design separating order and initialization variance is a later improvement, not something this simple design estimates separately.

For each endpoint, report all per-tuple values, paired mean differences, 95% paired bootstrap intervals (10,000 tuple resamples, frozen analysis seed), medians, and raw learning curves. Bootstrap complete tuples across time when plotting uncertainty. Use a paired sign-flip test for the prespecified mean-difference contrasts; at 20 tuples exhaustive enumeration has `2^20` sign vectors. State the symmetry/exchangeability assumption under the null. Any Monte Carlo implementation must report its count and seed. Do not describe an approximate test as exact.

The confirmatory superiority family contains the three contrasts **ACP−ER**, **ACP−ER+CBP-style**, and **ACP−fixed**, on each of `ACC`, `OLD`, and `P_late`: nine prespecified two-sided tests, Holm-adjusted at familywise alpha 0.05. Report unadjusted paired intervals for estimation and Bonferroni simultaneous bootstrap intervals for directional decision rules: each interval uses percentiles `100*(0.05/18)` and `100*(1-0.05/18)`. Label these intervals separately from the Holm p-values. Bootstrap intervals are finite-sample approximations, even when the sign-flip enumeration is exact under its null assumptions.

The separate noninferiority family contains four contrasts: `P_late` for ACP against ER, ER+CBP-style, and fixed (margin −0.02), plus `OLD` for ACP against fixed (margin −0.01). Use one-sided paired sign-flip tests shifted by the margin with Holm control at 0.05, under the same symmetry assumption. For conservative interval decisions, require the one-sided Bonferroni lower bootstrap bound, percentile `100*(0.05/4)`, to exceed the relevant margin. Software must implement these procedures before the study is frozen. The two families support separately labeled superiority and noninferiority claims; do not treat two alpha-0.05 families as a single globally alpha-0.05 search.

Concrete decision thresholds are engineering choices, not facts inferred from the literature:

| Decision | Prespecified evidence |
|---|---|
| Promising performance | Mean `ACC` gain at least 0.03 versus ER **and** ER+CBP-style, positive Holm-significant `ACC` and `OLD` differences against both, and `P_late` lower noninferiority bounds above −0.02 against both. |
| Adaptive-controller contribution | ACP−fixed has at least 0.01 mean gain in `ACC` or `P_late` with a positive Holm-significant result for that endpoint, while the fixed-comparison lower noninferiority bounds exceed −0.01 for `OLD` and −0.02 for `P_late`. |
| Practical gain ruled out | The upper simultaneous interval for `ACC` gain versus either primary baseline is below 0.03. This rejects the chosen three-point target in this setting, not every possible benefit. |
| Joint-benefit claim contradicted | An upper simultaneous bound for acquisition difference is below −0.02; alternatively, a primary baseline dominates ACP with upper simultaneous bounds below zero for both `OLD` and `P_late` at the frozen resource/operating point. |
| Adaptive contribution unsupported with precision | Both ACP−fixed `ACC` and `P_late` 95% paired bootstrap intervals lie entirely inside ±0.01. This is a conservative two-endpoint equivalence screen corresponding to per-endpoint TOST alpha 0.025; implement the corresponding paired equivalence tests before making a confirmatory equivalence claim. |
| Inconclusive | Intervals overlap both useful gain and practical failure margins, including an ordinary nonsignificant test. |

A relative reduction in forgetting of 20% is secondary, and is not interpreted when baseline forgetting is near zero. There is no `PRR >= 0.90` criterion. Resource efficiency requires the corresponding result under the compute contract, not only on equal-exposure runs.

Twenty tuples are an initial target, not a guaranteed power calculation. Estimate paired variance from development runs and simulate power/interval width for the three-point and one-point targets before freezing the confirmatory sample size. If 20 is inadequate, revise sample size before reading confirmatory outcomes, or retain 20 and describe the study as a precision-limited pilot. Do not stop when a p-value first crosses 0.05.

Infrastructure failures may be rerun with the same tuple and configuration, with failures recorded. If infrastructure prevents completion, publish the missing-pair count and withhold the full confirmatory decision until those pairs are recovered. Numerical divergence attributable to the method is an outcome: assign zero accuracy to its remaining checkpoints for the main intention-to-run analysis, retain its earlier acquired scores when computing forgetting, and report the failure rate plus a clearly labeled last-valid-checkpoint sensitivity analysis. Do not silently discard divergent tuples. Any temporal trend model should cluster by tuple and account for repeated measurements; a task-index slope alone does not identify plasticity loss because class difficulty changes.

## 8. Mechanistic ablations and controlled streams

Run each ablation with matched paired streams, replay policy, and tuning allowance. At minimum compare ACP without recycling, without consolidation/anchors, and without adaptive reopening. Additional useful ablations are whole-network reopening, no replay-damage term, no consolidation relaxation, random rather than utility-ranked recycling, and removal of rank/dormancy input. A “no replay” ablation removes both rehearsal and its sensor; it does not isolate either role. A sensor-only control requires the same replay examples and diagnostic compute but no replay training gradients, with that information budget reported.

Fixed gates should be tuned during development. Add a **schedule-matched replay** control if the initial comparison is promising: use a gate schedule recorded on independent development streams, matched in total open duration and reset budget, then applied blindly to confirmatory streams. This tests temporal alignment rather than merely the amount of high-learning-rate training. A fixed controller that never gets comparable opening time is a weaker attribution control.

Use controlled synthetic streams to test these predictions before scaling:

| Stream | Predicted diagnostic if the controller is working |
|---|---|
| Stationary labels/features | After initial development, sustained opening is uncommon; retention and acquisition remain adequate. |
| Abrupt meaningful shift | Reopening responds after the shift, stays local and bounded, and acquisition improves without a disproportionate old-loss increase. |
| Gradual drift | Adaptation occurs without needing explicit task boundaries. |
| Label noise / unlearnable samples | High loss alone does not cause indefinitely repeated opening and destruction of old behavior. |
| Recurrent domains/classes | Returning to a prior domain tests retention and reacquisition without endlessly expanding output semantics. |
| No shift but deliberately degraded capacity | The plasticity diagnostic responds to impairment; this does not automatically prove novelty detection. |

Fix rates, durations, and noise levels in the development manifest. Diagnostic targets such as “adult/open duty cycle below 20% on stationary streams” or “event response within three monitor windows” can be useful unit/feasibility criteria, but are not universal scientific laws. Avoid tuning synthetic shifts to perfectly match the controller's assumptions.

## 9. Longer-horizon and natural-stream replication

Ten CIFAR experiences cannot establish preservation of plasticity over a lifetime. The next inexpensive test should keep label/output semantics fixed while varying domains for at least 100 experiences, with a fresh-model acquisition reference for each domain and controlled recurring domains. This makes declining learnability easier to distinguish from growing classification difficulty.

For [CORe50](https://proceedings.mlr.press/v78/lomonaco17a.html), specify the official scenario and train/test session split before running; do not mix NI, NC, and NIC results. For [Infinite dSprites](https://proceedings.mlr.press/v274/dziadzio25a.html), freeze generative-factor distributions, label semantics, horizon, and access to factor supervision. The proposed supervised image-classification experiment is distinct from a factor-supervised generalization result. Reinforcement learning is a separate follow-up after the supervised mechanism passes attribution controls.

## 10. Artifacts required for a reviewable result

Keep code and documentation in local git; record the exact commit and whether the run had uncommitted changes. Each run must save its resolved configuration, seed tuple, dataset/split/class-order manifest, training-stream hash, environment versions, device details, evaluation matrix, acquisition curves including scratch references, controller events, resource measurements, and completion/failure status. Separate development and confirmatory output roots and refuse accidental overwrite.

The final report must show all core methods, failed runs, paired uncertainty, runtime/memory, the frozen decision table, and actual deviations. A successful mechanism test would justify replication and stronger baselines. A failed or inconclusive test should identify whether the problem was controller attribution, a retention–acquisition tradeoff, resource overhead, insufficient precision, or an implementation invariant.
