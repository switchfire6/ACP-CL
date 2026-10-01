# Adam: what to carry forward from the research history

This synthesis revisits the original critical-period studies, the joint-persistence
proposal, and the subsequent learning experiments. It records what the evidence
supports and which combinations might address the observed failures. It is not
a new experimental protocol, a replacement for the original decisions, or a
claim that the broader continual-learning objective has been achieved.

The strongest thread is to **retain varied causal experience, infer when it
applies, and allow representations to change**. The main unresolved problem is
how to direct those changes without damaging predictions that remain useful.
The [research charter](research_charter.md) states the current objective and scope.

The [temporal-representation proposal](representation_learning_design.md) was
implemented as a bounded qualification-first study. Its outcome-only reference
passed fresh individual-dependency learning but failed4 of31 fixed five-law
interleaving requirements, chiefly availability of the arrival-delay cue and
one base-law prediction ceiling. The reconstruction candidates were therefore
not run; this is a stop decision for that fixed recipe, not a negative result
for the untested auxiliary objective. See
[the result report](../reports/representation_learning_results.md). The stopped
branches below remain stopped.

## Latest update: causal access and the branch stopping decision

The [completed causal-access test](../reports/causal_access_results.md) fits six
fresh seeds per architecture and scores every arrival in 16 alternating frozen
blocks. Both architectures fail component qualification and the raw online
screen. Recurrent causal mixing improves whole-stream Brier by .039696 versus
full (6/6) and .010461 versus fixed half (5/6), but its old core remains .081204
worse than the specialist and novel affected error costs .005512 versus full.
Conditional mixing improves .004563 versus full but is .000787 worse than
fixed half, with only 3/6 improvements. Both miss the delay-cue guard.

Finite error/survival tolerances were fixed in advance. These failures do not
reflect a requirement for perfect memory, and the frozen stream cannot test
whether more parameter forgetting or consolidation improves ongoing learning.
Supply-timing cue use is substantial in this cohort; the earlier conditional
timing weakness was not universal. Do not rescue the recipe by excluding delay.

Stop the frozen-core/additive-residual branch as specified. Retain the narrower
recurrent access result, while returning any future design to reliable
acquisition and reusable competence before adding another controller. A future
graded-protection or consolidation comparison would need repeated new learning
and a fixed resource budget; it has not been run here.

## Previous update: saved output and context decomposition

The [completed saved-output diagnostic](../reports/core_residual_output_results.md)
uses 48 checkpoints and 384 input/context rows without training. Both fixed
exploratory allocation heuristics pass: refreshed-support full-minus-core old
error is +.012873 conditional (5/6 positive) and +.084727 recurrent (6/6), while
core-minus-full novel affected error is +.018927 (5/6) and +.029784 (6/6).
These are relative component tradeoffs, not successful continual learning.

Absolute prediction limits matter. The recurrent frozen core's old error is
.247546; it is a weak fallback despite outperforming full at return entry.
After return, selecting core improves novel error but does not restore learned
cue use. Conditional timing-cue use remains negligible under the tested routing
crossovers. Both separate arms already beat joint at actual return entry in
all six seeds, so the original recurrent return-AUC failure cannot be assigned
to entry performance. No component diagnosis explains the whole trajectory.

The [historical causal-access draft](core_residual_output_followup.md) led to
the completed fixed protocol and stopping decision above. The original primary
failures and broader limits remain unchanged.

## Previous update: learned core and adaptive raw-input residual

The [completed architectural comparison](../reports/core_residual_results.md)
starts from a common core learned from scratch and allocates one zero-output
residual before maintenance. It preserves complete core Adam state and uniform
replay, then follows actual continuous novelty/return trajectories. Both fresh
references qualify. The frozen-core candidate fails the fixed primary screen
in both architectures; no parameter, exposure, seed or threshold was retuned.

Conditional separate-minus-joint novel AUC is +.016853, improving 4/6 seeds;
recurrent is +.003917, improving 2/6. Both supply-timing seeds account for the
largest acquisition costs, but cannot be dropped or converted into a subgroup
promotion. Conditional return AUC improves in all six seeds; recurrent return
AUC and survival worsen. The conditional post-return novel endpoint is worse,
but it already entered return worse: this is not greater forgetting during
return. Correct-support probes and actual-history behavior remain distinct.

Recurrent feature attribution passes: an adaptive residual encoder beats fixed
random visual features by -.012130 novel Brier AUC, improving all six seeds.
Thus new visual adaptation remains useful. Freezing the whole learned core does
not provide the intended acquisition/retention balance, and frozen weights do
not freeze the combined prediction. The archived core/residual/context
diagnosis is now complete above. Any follow-up policy needs fresh locked data.

## Previous update: state and environment substitutions

The [completed follow-up](../reports/rehearsal_state_results.md) separates old/new
weights, complete Adam state and causal anchor on fresh seeds, using matched
current/returning environmental branches. Both fixed replication screens fail:
conditional T=-.000439 (6/12 positive), recurrent T=+.000825 (8/12 positive).
The earlier common deterioration pattern is therefore not a robust basis for
assigning blame to one component. All factorial findings remain descriptive.

Relative and absolute usefulness also diverge. Conditional selection's advantage
over uniform rehearsal increases while its selected absolute error rises;
uniform worsens more. The selected later-state predictor approximately ties
the no-extra-rehearsal reference. Reverting recurrent weights slightly improves
the ranking contrast but worsens selected absolute accuracy. The matched
environment comparison shows little transfer of candidate rankings, without
identifying a successful adaptive memory policy.

Keep uniform replay and retire the cached-score/rollback branch. The next
bounded direction is the third architectural combination described below:
a learned reusable core and an adaptive pathway capable of learning new features,
with a same-capacity jointly trained control. Starting from a common learned
prefix removes the original stable pathway's random cold start, but its benefit
is a new hypothesis. Test actual continuing trajectories, fresh dependencies,
acquisition and valid-old/return performance under matched resources. This
proposal was subsequently tested above and failed its primary screens; it does
not convert any earlier failure into success.

## Previous update after direct predictive-value testing

The [completed diagnostic](../reports/predictive_value_results.md) tests the
user's proposal directly through the measured benefit of extra rehearsal.
Across 12 fresh seeds per architecture, both local and returning-condition
tiers fail. A memory selected using a useful frozen predictor is not thereby
a reliable recommendation for training the later learner. Original packet
accuracy also supplies no consistent selection advantage over uniform.

The original frozen predictors preserve more next-window selection benefit
than fresh rehearsals of the same selected packets. This is consistent with
value depending on learner/optimizer state and the latest anchor, which change
together here. It does not establish a single cause. Returning-condition
candidates are available in 19/24 assessments per model; their presence alone
does not make the current-window score select useful future rehearsal.

Keep ordinary uniform replay as the working reference and stop expansion of
this fixed score recipe. Preserve the broader distinction among current
applicability, update usefulness and dormant knowledge. No adaptive retention
policy, rare-event protection, noise robustness or compounding learning has
been established. The protocol and all earlier study decisions remain intact.

## The evidence across the project

### Critical periods and renewable capacity

The [initial controller studies](../reports/development_results.md) established
working instrumentation but no benefit attributable to a critical-period cycle.
Some early configurations never closed; after calibration, the active controller
underperformed replay plus recycling. An apparent advantage from a run with no
closure or reopening is not evidence for those mechanisms.

The [v2 mechanism study](../reports/v2_results.md) found that an offline,
constant-gain diagnostic with matched reset times and counts substantially
outperformed the adaptive controller. That control used information from a
completed trajectory and was not a deployable learner. It nevertheless narrowed
the engineering problem: accurate boundary timing alone did not rescue the
tested plasticity allocation. Early closure, shortcut learning, and update
instability needed to be distinguished.

The [v3 local-allocation study](../reports/v3_results.md) produced a modest
positive signal for temporary newborn gain: +1.15 percentage points of late
acquisition AUC over recycling across three procedural-shape seeds. Most of
the mean gain came from one seed. The full combination increased mean AUC but
reduced mean final accuracy. The subsequent
[CIFAR-10 transfer pilot](../reports/cifar10_transfer_results.md) failed its
prospective screen: newborn gain changed recurring acquisition AUC by -0.12
points and final accuracy by -3.60 points relative to recycling. These findings
do not support reviving the frozen recipe as an established component.

The [dual-path pilot](../reports/dual_path_results.md) then tested separated
acquisition and consolidation. It learned less and retained less than joint
training of the same two-pathway architecture. The initially random stable
pathway and reduced active learning capacity are plausible explanations, but
their causal contributions were not isolated. A low forgetting score from a
learner that acquired little is not a retention success.

### Joint persistence and conditional reuse

The [joint-persistence experiment](../reports/persistence_results.md) supplied
a measurable physical objective: predict and act to keep two participants
functional over a finite resource-transfer trial. Historical replay clearly
helped returning conditions. The proposed memory priority based on past
frequency and predicted action sensitivity did not beat uniform replay and
failed its continuation screen. Those proxies were not measurements of the
causal value of an individual memory.

The [conditional-reuse study](../reports/conditional_results.md) produced the
strongest positive result in this line. Learned predictors used recent, bound
action/outcome history to recover a familiar mapping without weight updates.
Return survival improved by 11.60 points over pooling, positive in all eight
seeds. Correct history binding and retained causal packets each mattered.
However, learning a later dependency was slower; reuse did not automatically
provide improved acquisition.

The [matched-history comparison](../reports/contextual_results.md) strengthened
that result. Original conditional routing improved frozen-weight return survival
by 7.10 points over a qualified recurrent learner receiving the same history,
again positive in all eight seeds. It also retained stronger old-condition
predictions after new learning. The proposed additive correction from current
query features to routing scores failed its new-learning screen. Its small
return benefit does not turn it into a successful acquisition mechanism.

### Feature learning, carried state, and replay

The [continued-acquisition diagnostic](../reports/acquisition_results.md) found
useful retained weights and a cost from carried training state. Resetting
optimizer and replay together improved new-dependency acquisition but increased
damage to still-valid knowledge. Retained weights after reset outperformed
fresh initialization at the same arrival budget. This does not isolate learning
speed from initial competence and transfer. The study's narrow conditional
delay-cue qualification failure remains a failure.

Feature adaptation mattered differently across dependencies. In particular,
freezing substantially reduced learning of supply timing even when pooled
prediction AUC showed little difference. The record therefore supports keeping
features adaptable; it does not identify a universal feature-replacement rule
or demonstrate that aging itself exhausted representational capacity.

The qualified [optimizer-by-replay study](../reports/training_state_results.md)
separated the combined reset. Replay reset, with Adam retained, improved
acquisition in every seed average for both architectures. Adam reset alone
showed little acquisition benefit. Replay reset could improve prediction in
the returned condition while worsening broader retained knowledge; it also
increased vulnerability to corrupted feedback. Clearing an example buffer is
not sleep replay, and resetting optimizer statistics does not prune synapses.
The [sleep and training-state note](sleep_and_training_state.md) explains the
limits of that analogy.

The [replay-renewal studies](../reports/replay_renewal_results.md) showed that
both packet deletion and reservoir-age rebasing contribute to the local reset
benefit. A continuous recent-8/historical-8 policy improved acquisition in every
seed but failed retention and extra-noise-cost requirements in both architectures.
Keeping a historical half did not ensure preserved predictions. Uniform replay
remains the working reference; the evidence does not justify combining a failed
split with other changes and crediting any result to all of them.

### Repeated evidence and time

The [evidence-consolidation study](../reports/evidence_consolidation_results.md)
tested whether several future observations justified adopting a frozen proposal.
Sustained acceptance reduced error during some noisy conditions, but increased
whole-stream error in every pilot seed. Fixed delay alone was already costly.
The retained clean-support predictions and actual online return performance did
not move together, particularly for recurrence. Missing knowledge, its expression
under recent history, and the age of deployed weights are therefore different
possible sources of error.

The experiment measured a continuous finite learning lifetime. It did not test
improved representation learning because every policy shared the same training
trajectory. Novel cues appeared early, and physical outcome labels still ended
at H=12. The slight mean narrowing of the late performance gap was uncertain;
accumulated costs were not recovered. A constant advantage accumulating over
time would also be different from an advantage that grows through better learning.

Repeated predictive evidence cannot generally distinguish changed physical
dynamics from changed reporting. The [noise derivation](evidence_noise_limits.md)
shows this directly for the declared replacement process. A proposed selective
update rule needs explicit assumptions and controls; repeated agreement is not
a source of clean physical labels.

### Matched returns and selective updates

The [matched return diagnosis](../reports/return_context_v2/summary.md) crosses
identical queries with saved parameter states and different support histories.
Refreshing context helps, but current return-start parameters still perform
worse than prefix parameters under identical independent target-law support:
mean Brier increases of .09461 conditional and .20033 recurrent, positive in
every seed. The prior-return comparison also shows a deficit, and subsequent
return training recovers much of it. This retrospective analysis supports both
a context effect and deterioration of accessible historical predictions. It
does not isolate a causal effect of age or prove neuron-level erasure. Correct
target support remains a privileged diagnostic reference.

The [selective-update follow-up](../reports/adam_revision_results.md) then tested
full uniform replay, stronger current-loss emphasis and projection of the actual
Adam displacement against a sampled replay gradient. Its fixed factorial
controls separate weighting and projection; a fifth arm shrinks its own update
to the local projected norm. All 12 shared prefixes and 72 first-introduction
episodes completed on six fresh seeds across both architectures, and every
fresh qualification group passes.

The primary .75-weight projected candidate fails both architectures' acquisition
and retention screen. Acquisition Brier AUC increases by .00499 conditional and
.00547 recurrent, worsening every seed. Mean valid-old cost exceeds the .005
limit for conditional mode 1 and both recurrent modes; survival guardrails pass.
The separate directional-protection attribution screen also fails in both
architectures. Broad retention intervals do not turn failed mean-based criteria
into passes. Stop expansion of this fixed recipe.

Current emphasis alone also worsens acquisition in every seed. Changing loss
weights while retaining the full uniform reservoir therefore did not reproduce
the earlier gains from changing replay membership or sampling. Protection alone
has small secondary effects, not evidence for promoting another winner. The
tested interference proxy is not an estimate of a memory's marginal future
usefulness, so that different proposal remains untested.

## What remains untested

The original relational premise remains broader than the implementations.
They learn neural conditional predictors; they do not discover explicit symbolic
relations or establish that relationships are a superior representational unit.
Joint persistence is the chosen evaluation objective, not an experimentally
established foundation for every kind of learning.

Spatial cooperation networks, independently motivated internal modules,
reproduction, inheritance, evolutionary selection of representations, and
adaptive exploratory actions were not implemented in these studies. The
environment supplies randomized performed actions during training. Most of the
recent evidence comes from one constructed physical world.

The deep-research proposals suggested connecting external cooperation to an
internal modular architecture. That connection remains a design hypothesis.
There is no result here requiring an internal graph isomorphic to the world or
requiring topological growth to learn new features. The current encoders already
learn from random initialization without adding nodes. Whether explicit
modularity would improve transfer is a separate testable question.

No study has established a causal age effect using equally learnable, matched
novel dependencies introduced early and late. Likewise, preserving knowledge,
improving subsequent acquisition, and extending the physical planning horizon
remain distinct goals. The [related-work review](adam_related_work.md) identifies
close precedents for the existing ingredients; a new combination alone would
not demonstrate novelty or effectiveness.

## Status of the previously ranked combinations

**1. Immediate predictions, retained predictors, and applicability inference.**
The proposed bounded matched-return diagnosis is complete. Holding queries
fixed distinguished effects of recent context from effects of saved parameter
state. Correct supporting experience alone did not restore all historical
predictions. This motivated testing interference during learning, rather than
another global deployment delay. It does not validate a new predictor-retrieval
mechanism: historical checkpoints help old returns without acquiring newer
dependencies, and privileged supports are not available before feedback.

**2. Uniform causal replay, immediate learning, and local prediction protection.**
One concrete combination has now failed its prospective development screen:
the .75-current-weight rule with actual-displacement projection. Its local
first-order protection did not deliver useful acquisition and retention under
the declared budget. Output distillation, selective release of obsolete
constraints and other protection rules remain untested here, with close prior
art; the failed combination neither validates nor rejects that entire family.
Preserving obsolete or mistaken predictions remains a central risk. A small
secondary effect must not silently replace the failed primary candidate.

**3. A small immediate residual learner and a previously trained reusable core.**
A bounded residual component could learn new cue dependencies while context
inference retrieves existing predictions. Beginning from a trained common core
would remove the earlier dual-path pilot's random stable-path cold start. A
jointly trained model of identical capacity and a protection ablation would
help separate architectural capacity from the proposed division of learning.
This combination was subsequently tested in the
[core/residual comparison](../reports/core_residual_results.md) and failed both
architectures' primary screens. The recurrent raw-input adaptation control
supports useful feature learning, but does not validate the division of learning.
It should not be bundled with growth, recycling, and delayed consolidation or
promoted solely because earlier candidates failed.

This records the status of the original ranked agenda, not a new parameter
search plan. The matched diagnosis guided one bounded intervention, which
failed. Any new candidate needs fresh development data, an explicit resource
comparison and independent evaluation; completed seeds must not become its
tuning set. Immediate uniform replay remains the working reference.

If a candidate clears acquisition and retention requirements, test the user's
long-term hypothesis with counterbalanced early and late introductions of
matched novel dependencies. Fresh-weight references, initial competence,
subsequent acquisition, lifetime error, and retained valid predictions must be
reported separately. That would make a possible compounding learning advantage
measurable without assuming it from improved reuse.

## Historical proposal: validate predictive usefulness

The user's [predictive-retention proposal](predictive_retention.md) asks which
memories help later predictions. The next useful step is a score-validation
diagnostic, leaving uniform memory membership unchanged. At fixed assessment
times, compare short matched rehearsal forks for sampled packets against a
declared replacement. Freeze the shadows before new feedback, use one future
window to rank candidates, and validate the ranking on a separate later window
and delayed returns. Compare with uniform choice and low original prequential
packet error, recorded before the ordinary learner trained on that packet.

Original accuracy depends partly on the learner's competence at arrival. A rare,
currently absent dependency may also have little immediate value and substantial
value at its next return. Counterbalance return gaps and condition order, and
keep repeated assessments within the independent seed. With a common replacement
predictor, ranking marginal gains on the same window equals ranking rehearsed
predictor accuracy; those cannot serve as independent controls.

This historical proposal led to the two completed diagnostics summarized above.
Neither establishes a successful adaptive sampler or justifies score-based
eviction. The completed [selective-update protocol](selective_updates_protocol.md)
and its failed decision remain unchanged. The next architectural comparison is
recorded separately in the charter and the latest report.
