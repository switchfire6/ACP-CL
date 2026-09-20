# Research review: adaptive critical-period continual learning

Review date: 2026-09-20. The supplied report is archived in [deep-research-report.md](deep-research-report.md). This review separates evidence from the engineering hypothesis; it does not report a successful experiment.

The proposal is worth testing as an **adaptive controller over known continual-learning mechanisms**. Its strongest claim is that deciding when and where to consolidate, reopen, and recycle improves the retention–adaptation tradeoff beyond replay plus recycling. Biology motivates this question but does not establish that the proposed controller, equations, or thresholds will work.

## Evidence and citation audit

The report's embedded `turn...` citation identifiers are not portable references. The primary sources below provide independently checked links. “Verified” means the cited result is supported by the original paper's accessible abstract or full text, not that its experiments were independently reproduced.

| Claim | Assessment and primary source | Consequence for ACP |
|---|---|---|
| Inhibitory maturation helps initiate a visual critical period. | Supported in the studied animal model: [Fagiolini & Hensch, Nature 2000](https://www.nature.com/articles/35004582). | Supports a state-dependent opening analogy. An ANN learning-rate gate is not an inhibitory circuit. |
| Experience-dependent Otx2 transfer affects PV-cell maturation and timing. | Supported: [Sugiyama et al., Cell 2008](https://pubmed.ncbi.nlm.nih.gov/18692473/). | Maturity and experience can jointly regulate plasticity. This does not specify a neural-network maturity statistic. |
| Transient disinhibition participates in ocular-dominance plasticity. | Supported: [Kuhlman et al., Nature 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3962838/). | Motivates localized permission to change, without establishing the correct block-selection rule. |
| PSD-95-dependent silent-synapse maturation contributes to closure; later knockdown can restore plasticity. | Supported in mouse visual cortex: [Huang et al., PNAS 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4475980/). | Reversible protection is a useful functional analogy. It is not evidence for importance-weighted gradient scaling specifically. |
| Mature extracellular constraints can be relaxed. | Adult rat ocular-dominance plasticity followed chondroitinase-ABC treatment in [Pizzorusso et al., Science 2002](https://pubmed.ncbi.nlm.nih.gov/12424383/). | The experiment supports ECM involvement; it does not isolate perineuronal nets from every other affected structure. |
| NgR and Lynx1 constrain mature visual plasticity. | Supported by [McGee et al., Science 2005](https://pubmed.ncbi.nlm.nih.gov/16195464/) and [Morishita et al., Science 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC3387538/). | Several biological mechanisms can restrict plasticity. There is no one-to-one mapping from a molecule to an ANN regularizer. |
| Multi-timescale synaptic state improves modeled memory tradeoffs. | Theoretical results in [Fusi, Drew & Abbott, Neuron 2005](https://pubmed.ncbi.nlm.nih.gov/15721245/) and [Benna & Fusi, Nature Neuroscience 2016](https://www.nature.com/articles/nn.4401). | Useful computational motivation. The report's two-weight transfer rule is not a reproduction of the latter's bidirectionally interacting multi-timescale model. |
| Deep networks can lose the ability to learn, separately from forgetting. | Supported in the investigated settings by [Dohare et al., Nature 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/). | Measure acquisition and retention separately. Small task counts need not expose plasticity loss. |
| Infinite dSprites supports longer controlled streams. | Verified in the [original PMLR paper](https://proceedings.mlr.press/v274/dziadzio25a.html). | Suitable later stress test; its factor-supervised setting and arbitrary horizon are not interchangeable with ten disjoint CIFAR experiences. |

### The 2026 aggrecan citation is real, but preliminary

The [bioRxiv v1 record](https://www.biorxiv.org/content/10.64898/2026.07.26.740801v1) identifies Emily C. Crouse and colleagues, DOI `10.64898/2026.07.26.740801`, posted **July 28, 2026**. Its indexed abstract reports that inhibitory-neuron Acan deletion eliminated PNNs without preventing closure, whereas deletion in all neurons or excitatory forebrain neurons maintained adult plasticity. This supports the report's description of what the preprint claims.

Direct full-text retrieval returned HTTP 403 during this audit; the primary bioRxiv abstract and metadata were available through indexing. Detailed methods, figures, deletion specificity, and a subsequent peer-reviewed version were not verified. Cite it as a preprint and a challenge to a simple PNN-only interpretation, not a settled replacement mechanism. ACP does not depend on this result being correct.

### Boundaries on the neuroscience interpretation

These animal sensory-circuit interventions do not demonstrate that biological memory has solved catastrophic forgetting, that consolidation is always beneficial, or that every adult critical period can safely reopen. Nor do they validate a sigmoid of loss, drift, dormancy, and replay damage. The ANN mapping is an inference and a design proposal.

The report also discusses BDNF, complement, microglia, synaptic scaling, and nucleus-basalis pairing. Those peripheral citations were not re-audited here. In particular, its BDNF statement shares an Otx2 citation rather than supplying a distinct BDNF source. They should receive their own primary citations if retained in a manuscript; they are not required premises for the initial algorithm.

## What is already established computationally

| Prior work | Overlap with ACP | Necessary distinction |
|---|---|---|
| [EWC, Kirkpatrick et al.](https://arxiv.org/abs/1612.00796) | Protects parameters important to earlier tasks. | Utility-dependent protection is not new. ACP changes its timing and coupling to other operations. |
| [Synaptic Intelligence, Zenke et al.](https://proceedings.mlr.press/v70/zenke17a.html) | Accumulates optimization-path importance. | A channel-level, windowed approximation must be called SI-inspired, not an exact SI reproduction. |
| [GEM, Lopez-Paz & Ranzato](https://arxiv.org/abs/1706.08840) | Uses episodic memory and gradients to control interference; defines transfer metrics. | A replay-loss penalty in a controller is neither GEM projection nor a guarantee that old loss cannot rise. |
| [DER/DER++, Buzzega et al.](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) | Strong replay and historical-prediction regularization. | ER and DER++ are different algorithms. Do not label a plain ER implementation “ER/DER++.” |
| [ANML, Beaulieu et al.](https://arxiv.org/abs/2002.09571) | Learns activation gating and thereby selective plasticity. | ACP's proposed gates regulate updates using online state; they are not meta-learned activation gates. |
| [RigL, Evci et al.](https://proceedings.mlr.press/v119/evci20a.html) | Changes sparse connectivity during training. | It is evidence for dynamic sparse optimization, not a direct continual-learning success claim. Dense channel recycling is not RigL. |
| [Continual backpropagation, Dohare et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/) | Replaces mature low-utility units to maintain plasticity. | An altered utility, normalization, replacement cadence, or channel implementation should be labeled CBP-style. Its original objective does not itself solve retention. |
| **Omitted close prior work:** [UPGD, Elsayed & Mahmood, ICLR 2024](https://arxiv.org/abs/2404.00781) | Protects useful units while applying larger changes/perturbations to less useful ones, addressing both forgetting and plasticity. | The combination of protection and renewal is already studied. UPGD belongs in a publication-stage comparison. |
| **Omitted critical-period prior work:** [Achille, Rovere & Soatto, ICLR 2019](https://arxiv.org/abs/1711.08856) | Studies early critical learning periods and persistent effects of temporary input deficits in deep networks. | “Critical periods in AI” is not a new observation. Adaptive recurring reopening is a different, narrower hypothesis. |

This is a targeted audit, not an exhaustive novelty search. A defensible contribution would be a precisely specified online temporal/local controller whose contribution survives component and resource controls. A favorable comparison to fine-tuning or EWC alone would not establish this contribution.

The report's reference to CIFAR-100 in Dohare et al. needs an experimental caveat: their class-incremental plasticity study trains on data from all classes introduced so far and uses retraining references. The proposed disjoint-experience, bounded-replay experiment changes the information budget and simultaneously introduces forgetting. It is not a direct replication of that experiment. [Original methods and results](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/).

## Prototype scope and differences from the report

The first repository version implements an engineering candidate, not a reproduction of every cited algorithm. `er_recycle` uses an EMA of absolute activation times outgoing-weight L1 contribution, with age bias correction; it is CBP-inspired. `derpp` uses replay cross-entropy and stored-logit MSE on the same replay minibatch; it is a documented DER++ variant. EWC, canonical SI, GEM, and UPGD are not implemented comparisons in this version.

The ResNet variant is a CIFAR-stem, GroupNorm ResNet-18 **with an additional Linear/ReLU adapter**. All methods share it; only adapter units are recycled. The backbone can be gated and consolidated, but this version does not renew backbone convolutional channels. The shared classifier stays ungated and unconsolidated. Thus any observed acquisition advantage may partly depend on a continuously trainable readout and adapter, which should be examined in later ablations.

The custom optimizer gates the data-momentum displacement using `g_min + (g - g_min) * (1 - c_eff)^gamma`, applies scaled weight decay separately from momentum, and applies the anchor force independently. This is an SGD-style optimizer, not identical to PyTorch SGD with coupled L2 weight decay. At `g = g_min`, consolidation no longer further reduces data gain; its adult effect is through anchors, with gain differentiation during opening. Importance integrates data gradient against the gated data-momentum displacement, excluding anchor/decay displacement. This conservative attribution differs from canonical SI. The window-origin snapshot and moving regularization anchor are separate.

Initial `anchor_rate = 0.5` leaves an initialization component in the first consolidated anchor. A full first commit would protect the just-learned solution more directly; partial commits are a design choice worth checking. The current probe gives lagged, sampled before/after update feedback, not a guarantee against accumulated forgetting between monitoring events. Read the versioned algorithm/configuration documentation for the current monitor implementation.

Synthetic and quick CIFAR experiments validate implementation and feasibility. A full-size configuration, scratch-reference support, or bootstrap routine does not by itself complete the planned confirmatory study or reproduce the cited literature.

## Design risks and interpretation checks

Several of the following safeguards are present in the prototype; others require the planned development experiments. They remain checks on the meaning of a result, rather than a list of alleged implementation failures.

1. **Separate novelty from difficulty and noise.** High cross-entropy can reflect new labels, hard examples, corrupted labels, poor calibration, or a weak model. Feature drift can reflect the model's own updates. Calibrate loss signals only from past training observations, record each component, and test stationary, abrupt-shift, gradual-shift, and label-noise controls.
2. **Compare replay damage on identical samples.** Losses on two different replay draws are not a before/after damage estimate. Use paired pre/post measurements on the same draw or a deterministic sentinel within the stated replay budget. Reset a stored baseline when its sample membership changes. The signal is sampled feedback, not a safety guarantee.
3. **Gate actual optimizer motion.** Scaling `.grad` does not scale previously accumulated momentum, optimizer weight decay, or adaptive optimizer state. Specify exactly which terms are gated and test the resulting parameter displacement. The classifier requires a deliberately stated policy so a closed feature controller does not accidentally prevent new labels from being learned.
4. **State the effective floor.** `g * (epsilon + 1 - c)^gamma` can shrink far below `g_min`; with `epsilon > 0`, it can also exceed `g` when `c = 0`. Either clamp the final gain or clearly distinguish the block gate floor from the actual parameter-update floor. Freeze experiments must genuinely produce zero intended motion.
5. **Keep importance and anchors separate.** For an SI-like statistic, use the optimization window's starting weights in the displacement denominator, not an unrelated moving anchor. Reset path accumulators at window commits. State whether the path integral uses total actual displacement or only its data-update component; the prototype uses the latter and includes momentum. An anchor that continually follows drifting weights is a finite-timescale constraint, not permanent memory.
6. **Prevent incidental importance inflation.** Tiny net displacement, channel width, normalization rescaling, and data-gradient noise can dominate importance. Add a denominator stabilizer, normalize within comparable groups, cap pathological values, and log distributions. Importance measured only on current data can protect recent features while undervaluing old ones; include the replay objective consistently.
7. **Do not claim renewable capacity is unlimited.** Monotonically increasing consolidation plus a permanently protected core may eventually leave no eligible units. A temporary reduction of effective consolidation does not reduce stored consolidation. Track eligible units, saturation, and replacement counts. Optional decay or capacity-release policies are distinct algorithm changes requiring ablation.
8. **Reset all affected state when recycling.** Reset incoming rows, the relevant outgoing columns, bias/normalization parameters where applicable, momentum, importance accumulators, age, and anchors for every changed slice. Otherwise an old anchor or momentum vector can immediately resurrect erased values. Small replacement rates need fractional credits; rounding them to zero silently disables renewal.
9. **Residual and normalization dependencies matter.** Resetting a residual-output channel can affect a shortcut identity path. GroupNorm couples channels within a group, so resetting one channel can change its neighbors even with zero outgoing weights. Start with explicit interior-layer recycling maps; measure immediate prediction and replay-loss changes rather than assuming function preservation.
10. **Do not let task metadata become the controller.** The evaluator may know experience boundaries. The learner should receive samples and labels, not experience indices or boundary-triggered resets. A fixed initial step budget is an explicit developmental prior. Subsequent adaptive decisions must depend on past/current online signals only.
11. **Keep inference comparable.** Use one shared classifier with a fixed output policy for every method. HAT/PackNet variants that select task-specific masks or heads using an inference task identifier do not belong in the primary class-incremental comparison without a separately costed task-inference mechanism.
12. **Count all resources.** Separate current/replay gradient passes, monitor forwards, anchor tensors, optimizer state, and replay storage all cost resources. Equal batch sizes and optimizer-step counts do not imply equal compute; equal numbers of replay images do not imply equal total learner memory.

These are implementation and experimental deductions, rather than claims established by any particular biological paper.

## Measurement changes from the supplied report

**Remove `PRR_j = AUC_j / AUC_1` from decision criteria.** Different class sets have different difficulty; changing output support and old-class competition further alter raw AUC. Small first-experience AUC makes the ratio unstable. Instead report each experience's acquisition curve, a fresh-model reference on that same experience, and the paired difference in normalized AUC. This reference improves interpretability without making a causal diagnosis by itself.

**Do not use untrained-new-class accuracy as the main forward-transfer measure.** A new classifier row has no trained semantics; an expanding or seen-class-masked head can make pre-experience scores undefined or mechanically zero. Keep output support constant and evaluate learning after controlled numbers of new-data presentations. Task-conditioned evaluation belongs only in explicitly separate diagnostic analyses.

**Use retention and acquisition jointly.** Forgetting can look small because nothing was learned. Final average accuracy can hide slow adaptation. Representation rank can rise because of noise. Report old-experience accuracy, final accuracy, backward transfer, peak-to-final forgetting, acquisition AUC, scratch differences, and representation diagnostics separately. The exact definitions and decision rules are in [experiment_protocol.md](experiment_protocol.md).

**Distinguish failure from uncertainty.** A nonsignificant difference is not falsification. An interval that excludes the stated practical gain, a sufficiently precise equivalence result for the controller ablation, or a reproducible retention–acquisition tradeoff that violates the declared tolerance is stronger negative evidence. A short synthetic run establishes software behavior only.

## Claim to carry forward

> With identical training streams, architecture, and replay policy, a bounded online controller coordinating local update permission, progressive consolidation, and low-utility recycling will improve retained accuracy while preserving late-experience acquisition relative to ER, ER plus CBP-style recycling, and a fixed controller. The efficiency claim additionally requires a comparison under a common compute budget.

This claim can fail independently of the neuroscience analogy. Its most informative negative outcome is that replay plus recycling performs equivalently, or that removing adaptive control leaves the benefit intact.
