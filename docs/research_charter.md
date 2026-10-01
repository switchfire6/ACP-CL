# Research charter: learning relationships under changing conditions

Status: accepted research direction, updated after the conditional-reuse,
matched-history, continued-acquisition, and optimizer-by-replay studies.
The replay-renewal and evidence-consolidation studies and their failed screens
are recorded. The matched return-context diagnosis is complete; the subsequent
selective-update development cohort is complete and fails both its primary
continuation and directional-protection attribution screens in both architectures.
The subsequent predictive-value diagnostic completed 24 trajectories and
1,296 rehearsal forks. Both architectures fail both the local and return
criteria; neither marginal rehearsal selection nor low original prediction
error earns promotion over uniform replay. A secondary frozen-predictor signal
suggests state-dependent usefulness, without isolating its cause. See the
[predictive-value report](../reports/predictive_value_results.md).
The subsequent [state-and-environment diagnostic](../reports/rehearsal_state_results.md)
completed 24 fresh trajectories and 3,456 rehearsal forks. Its fixed replication
screen fails in both architectures: conditional T=-.000439 (6/12 positive),
recurrent T=+.000825 (8/12 positive). It therefore does not establish a general
component cause of the earlier loss of usefulness. Conditional selection beats
uniform rehearsal locally in this cohort but approximately ties no extra
rehearsal. Neither a cached importance score nor a state rollback earns promotion.
This is a research hypothesis and roadmap, not a new locked
experimental protocol or a claim that general continual learning is solved.

The subsequent [core/residual development comparison](../reports/core_residual_results.md)
is complete. Both fresh architectures qualify, but freezing the learned core
fails both primary screens: novel Brier AUC costs are +.016853 conditional
(4/6 seeds improve) and +.003917 recurrent (2/6). Conditional return prediction
improves while acquisition and the post-return novel endpoint fail; recurrent
return prediction and both survival guards fail. Recurrent residual visual
encoder adaptation passes its distinct attribution screen (-.012130, 6/6
improve versus fixed random features). This supports that component's value
without validating whole-core freezing. Preserving parameter values alone does
not guarantee the combined predictor preserves useful behavior.

The [saved-output diagnostic](../reports/core_residual_output_results.md) is now
complete, with no training. Its two pre-score effort-allocation heuristics pass:
the correction can improve novel predictions while worsening the preserved core
on return. This is relative headroom, not proof of competent components or causal
selection. The recurrent old core remains weak, and conditional timing-cue use
remains negligible in that cohort. The subsequent
[fixed-predictor access test](../reports/causal_access_results.md) is complete.
Both architectures fail qualification and raw online requirements, so this
frozen-core/additive-residual branch is stopped. Recurrent access improves
whole-stream prediction beyond fixed averaging, but available old competence
and novel-condition guards fail. Conditional does not reliably beat fixed
averaging. The new cohort uses timing cues substantially but struggles with
delay. Neither a cue-specific impossibility nor general continual learning is
established. Any future direction must first demonstrate reliable acquisition
and reusable competence; no nearby gate/capacity/replay sweep is authorized by
this result. The overall objective remains open.

Working project title: **Adam**. Optional expansion: **Adaptive Dynamics and
Associative Memory**. Continual relational learning describes its research
direction; the acronym is a proposal, not a claim of a validated mechanism.
Subtitle: **Learning when knowledge applies, and when it must change.**
ACP-CL remains the historical name of the critical-period experiments. Existing
package and command names preserve reproducibility of the archived studies.

## Objective and scope

Develop a general mechanism that learns representations from scratch, retains
useful knowledge through changes in conditions, and continues acquiring new
representations under bounded memory and computation.

The original relational premise remains a research direction: useful knowledge
describes how observations, entities, actions, and consequences depend on one
another, together with the conditions under which those dependencies apply.
An isolated observation is evidence; a learned conditional dependency is a
candidate unit of reusable knowledge. Explicit symbolic graphs are one possible
implementation, not a required architecture or an established necessity.

Joint persistence supplies a concrete definition of usefulness in the present
testbed: the probability that both participants remain functional through a
specified horizon. It is an experimenter's objective. We have not shown that
joint survival is necessary for continual learning, that it uniquely defines
improvement, or that the neural mechanism must consist of cooperating agents.

## Working hypothesis

**A continual learner should acquire new predictive relationships, preserve
relationships that remain useful, and revise those that no longer apply,
without exhausting its ability to learn.**

Our proposed route toward this requirement remains:

**In environments with recurring and changing dependencies, learning when stored
knowledge applies and where it needs revision can improve the tradeoff between
new learning and retention, relative to learners with the same observations,
history, and resource budgets.**

The proposed mechanism separates three operations:

1. Learn conditional predictors and their representations from experience.
2. Infer their current applicability from recent observations, actions, and outcomes.
3. Revise predictions or develop new representations when existing predictors
   repeatedly fail, while limiting damage to still-useful knowledge.

Unexpected feedback alone is insufficient evidence of a new relationship.
It may reflect stochastic outcomes, missing observations, poor calibration, or
an obsolete relationship. Distinguishing these cases is part of the hypothesis,
not an assumption that a novelty detector will already solve it.

## Minimal mathematical description

Let h_t contain only observations, performed actions, and outcomes available
before the current decision. Let phi_psi be a representation learned from
scratch, and let f_theta_k be a conditional predictor. A bounded library can
make predictions through

    p_hat_t(y | x, a) = sum_k q_t(k | x, h_t) f_theta_k(y | phi_psi(x), a).

Changing q_t reuses knowledge by changing which predictor applies. Changing
psi or theta learns or revises the underlying predictors. Allocation or
replacement of predictors is a possible additional operation, subject to a
fixed total resource budget; unbounded growth is not a solution to forgetting.
The reference prototype restricts applicability to q_t(k | h_t). A tested
bounded intervention let the present observation influence applicability too,
without receiving its outcome. That additive correction failed its new-learning
screen; the general formula is a hypothesis family, not a validated architecture.

In the current world, y includes survival at several finite horizons. In a
second world it may be another measurable outcome. Action selection uses the
declared task objective, while prediction and applicability remain distinct
parts of the mechanism.

Time enters at three levels: observations within an event, recent experience
used to infer present conditions, and retained knowledge across longer absences.
The current four-head model and 32-record window are implementation choices.
Neither a universal window length nor a universal critical-period schedule is
part of the hypothesis.

## Claims and current evidence

| Claim | Current status | What would strengthen or weaken it |
|---|---|---|
| Recent evidence can reactivate useful learned mappings. | Supported within the testbed: +11.60 points over pooling in the first pilot; +7.10 over a qualified recurrent learner with matched history in the follow-up, each positive in all eight seeds. | Test different environments, broader competitors, and longer change schedules. |
| Retaining experience helps later reuse. | Supported in the earlier pilot (+18.24 points over current-only on return) and by broader retention and noise resistance with uniform replay in later studies. The fixed 8/8 split does not meet the combined requirements. | Test which experience remains useful and which updates protect that knowledge, under bounded memory and computation. |
| This mechanism preserves the ability to acquire new representations. | Not established. Both replay-renewal cohorts pass fresh-model learnability checks. Packet deletion and age rebasing improve local acquisition. Continuous 8/8 replay improves acquisition in every seed but fails retention and extra-noise-cost limits in both architectures. The later acceptance diagnostic shares draft training and cannot establish improved representation learning. The qualified selective-update cohort worsens acquisition in every seed of both architectures. | Validate a proposed predictive-usefulness signal before selecting another candidate; preserve the declared acquisition, retention, return, revision and noise requirements. |
| Relationships are a better representational unit than the alternatives. | Not directly established: the implementation uses neural predictors, not explicit relational structures. | Compare representations under controlled budgets and tests that require transfer of dependencies. |
| Joint persistence is a privileged general learning principle. | Untested. It is presently the chosen evaluation objective. | Separate objective effects from architecture effects; test ordinary prediction/control objectives as well. |

The [conditional-reuse report](../reports/conditional_results.md),
[matched-history report](../reports/contextual_results.md), and
[acquisition diagnostic](../reports/acquisition_results.md), and
[optimizer-by-replay diagnostic](../reports/training_state_results.md), and
[replay-renewal studies](../reports/replay_renewal_results.md) are the empirical
references. The failed [experience-selection rule](../reports/persistence_results.md)
and earlier [dual-path pilot](../reports/dual_path_results.md) remain negative
results. Changes to the working hypothesis must be recorded explicitly rather
than interpreted as retroactive success of those mechanisms.
The [full research synthesis](research_synthesis.md) connects these findings to
the original critical-period controllers, local renewal and image experiments,
including the unsuccessful transfer of the newborn-gain recipe to CIFAR-10.

## Completed comparison and next bounded question

The matched-history comparison completed the first step of this roadmap:
qualified recurrence, returning conditions, a new dependency, feedback noise,
matched absence schedules, and one diagnosed routing intervention. All planned
seeds and branches completed. The routing correction failed the predefined
new-learning screen, while the original conditional learner retained stronger
old-condition reuse and valid-knowledge retention than the recurrent comparator.
The frozen-feature diagnostic showed that continued visual adaptation helped
the recurrent learner acquire the new dependency.

The subsequent acquisition diagnostic kept both architectures and introduced
three new physical dependencies, with continued, frozen-feature, reset-state,
and fresh-weight controls. It completed 12 prefixes and 180 episodes on six
new seeds. Resetting optimizer and replay improved acquisition but increased
valid-old damage in every seed average for both learners. Retained parameters
after state reset outperformed fresh initialization at the same arrival and
update budget. There was no clear pooled acquisition-AUC benefit from feature
updates, although freezing substantially reduced learning of the supply-timing
cue. These endpoints answer different questions and should not be conflated.

Qualification is incomplete: the conditional fresh-model delay-cue benefit was
.00193868 against a required .002. The narrow miss stays a failure. It prevents
treating the complete diagnostic as qualified; it does not invalidate every
observation or establish a failure of the broad hypothesis. No late-versus-early
result establishes a causal loss of plasticity with age.

The optimizer-by-replay follow-up completed 72 fresh development fits, selected
8,192 arrivals, and ran 12 prefixes plus 372 comparison episodes on six new
seeds. Both architectures passed every fresh-model qualification group; their
delay-cue benefits were .005120 and .005016 against the declared .002 threshold.
This qualifies the new study without changing the previous failure.

With Adam retained, replay reset improved new-learning Brier AUC by .009743
conditional and .009626 recurrent, each positive improvement in all six seed
averages. Adam reset alone had little acquisition effect. Relative to ordinary
continuation, replay reset reduced conditional valid-old damage by .012093
but increased recurrent damage by .017901. Conditional damage still remained
positive in absolute terms. Its retention interaction also made resetting both
states worse than replay reset alone during acquisition.

The final controls limit that result: clearing replay improved prediction in
the returned world while worsening broader valid-old retention for both
architectures, and increased their vulnerability to corrupted feedback. The
same operation therefore helps adaptation and can undermine history-dependent
reliability. No blanket reset satisfies the combined research objective.

The next two studies held optimizer state fixed. A first-introduction diagnostic
separated packet deletion, effective reservoir age and random generators across
12 shared prefixes and 84 episodes. Deletion and rebasing both improved
factorial-mean acquisition while worsening mean valid-old retention. Resetting
random generators alone supplied no clear benefit. This local diagnosis does
not test repeated autonomous resets.

A separate fixed-policy cohort compared uniform, recent-only and an 8/8 split
through 36 prefixes and 288 episodes. Both experiments and their analysis were
locked before the first main fit. The equal split was specified directly,
without the roadmap's proposed tuning stage or a policy-development selection
cohort. It was not selected using either new cohort's outcomes. Any subsequently
tuned variant still requires independent development data.

Split reduced acquisition Brier AUC by .004667 conditional and .005237 recurrent,
improving all six paired seeds per architecture. Terminal valid-old error rose
by .023090 and .014030, against a .005 maximum mean increase. Extra matched-noise
penalties rose by .006165 and .006081, also above .005. Both architectures fail
the combined screen; all qualification and survival requirements pass. Several
retention intervals are wide, so the pilot's failed means are not proof of a
tolerance violation in every setting. All failures remain failures under the
unchanged prospective rule.

Uniform replay remains the working reference. The next bounded hypothesis is
whether persistent predictive evidence can guide which relationships to update
while protecting still-valid knowledge. A candidate must distinguish changed
dependencies from noisy reports using observed experience, with no privileged
change labels. This is a proposed direction, not a validated solution or a
license to rename the earlier failed selection rule. Use new development data
and explicit controls before another locked comparison. Do not tune on the
completed reset or policy seeds.

Continue separating four cases, concealed from ordinary learners:

* A familiar relationship returns after intervening experience.
* A new physical dependency requires learning a previously irrelevant cue.
* A previously useful cue changes meaning, making some old predictions obsolete.
* Feedback becomes noisier while underlying dynamics remain unchanged.

Primary measurements should concern new-dependency acquisition under fixed
exposure. Guardrails must cover old-condition reuse, valid-knowledge retention,
and inappropriate adaptation to noise. Declare minimum useful gains and maximum
acceptable losses before the comparison, report uncertainty, and keep all seeds.
A better old-condition score alone does not satisfy the new-learning objective.
Equal parameters and raw data do not imply equal computation; report remaining
work differences and treat each independent seed as the statistical unit.

A learner that improves new acquisition by erasing useful mappings has not met
the combined objective. Generality also remains open while all evidence comes
from this world. After a candidate clears the acquisition and retention screen,
test a second environment with independently changed observations and dynamics,
then longer streams. Separate representation, memory, applicability inference,
and objective choice. Critical-period scheduling remains optional and unproven.

## Accepted extension: evidence and lifetime performance

The user proposed testing whether short-term costs can be recovered through
longer-term usefulness. The [evidence-consolidation diagnostic](evidence_consolidation_protocol.md)
therefore extends evaluation to an 18-block continuous stream now, with a
prospective cumulative physical-prediction endpoint and matched early/late
cycles. New-dependency acquisition, returns, retained valid predictions, noise
and survival remain declared constraints. This changes the primary endpoint
for this bounded diagnostic; it does not reinterpret a failed earlier screen
or replace the general objective of learning new representations.

All acceptance policies share the same draft learning trajectory. They test
whether future predictive evidence improves commitment decisions, without
claiming that this changes representation learning. Better lifetime deployment,
better later acquisition, and farther-future planning are different claims.
The first is measured here. The latter two remain unestablished; outcome
labels still end at H=12, and novel cues are introduced early in the stream.

The [focused related-work review](adam_related_work.md) records close precedents,
including Paired Learners, SEA, persistence-based adaptation and complementary
learning systems. Neither delayed consolidation nor its future-evidence test
is claimed as a new general principle. A possible contribution must come from
a specific effective mechanism, a useful controlled diagnosis, or a benchmark
whose distinct value is established against appropriate alternatives.

Persistent reporting corruption can produce the same observed distribution as
changed physical dynamics. The proposed test uses a declared corruption family;
repetition is useful evidence only under assumptions that distinguish the
possibilities. Clean physical outcomes are evaluator-only and never supplied
to the learner or acceptance rule.

### Completed evidence-consolidation result

The [fixed pilot](../reports/evidence_consolidation_results.md) completed all
12 draft trajectories, 216 continuous blocks and 36 fresh references. Both
architectures pass every fresh qualification group. Development's separate
two-seed delay-cue failures remain recorded. The sustained acceptance rule
fails both architectures' prospective screens.

Whole-stream Brier increases by .001685 conditional and .002800 recurrent
versus immediate deployment, worsening every one of six seed pairs in each
architecture. Noisy-block error improves by .002050 and .011011, but return
error increases by .007811 and .016244, exceeding the allowed .005 cost.
Sustained makes only small uncertain overall gains over fixed delay and
one-packet evidence. Acquisition, still-valid retention, noise and survival
guardrails pass; lifetime gain and return requirements fail.

The late-minus-early advantage versus immediate is slightly positive on
average (.000660 and .000702), but both paired-seed intervals include zero.
Late error remains worse and accumulated costs are not recovered. This is
not evidence of a compounding representation benefit; the underlying draft
learning is identical by construction. The physical horizon also remains H=12.

Keep immediate uniform replay as the reference. The completed matched-return
diagnosis below compares deployed weights, context inference and accessible
historical predictions. A candidate that changes learning requires fresh
development data and its own protocol. Testing whether earlier learning helps
later acquisition also requires matched novel dependencies introduced late in
life. Do not tune the failed acceptance rule on these completed seeds.

## Completed diagnosis and selective-update test

The [matched return-context diagnosis](../reports/adam_revision_results.md)
crossed fixed query observations with different saved parameter states and
support histories. It retained all six seeds per architecture and all three
returns from the evidence-consolidation pilot. This is a retrospective
explanatory analysis specified after the original outcomes were known, not an
independent confirmation or a new passing screen for the failed acceptance rule.

Both context and parameter state matter. Replacing preceding-law history with
observed return feedback helps, especially for the conditional learner. Yet
under identical independent target-law support, current return-start weights
have higher Brier error than prefix weights in every seed of both architectures:
mean increases of .09461 conditional and .20033 recurrent. Comparison with the
previous return also shows a deficit, and subsequent return training recovers
much of it. Thus refreshing context alone does not restore all the accessible
historical predictions in these probes. This is not proof of neuron-level
erasure or an independently manipulated effect of weight age. Correct target
support and historical checkpoint comparisons are diagnostic references; simply
restoring old weights would not acquire the newer dependencies.

These findings motivated the [selective-update development experiment](selective_updates_protocol.md).
It tested whether increased emphasis on current experience accelerates
acquisition while redirecting conflicting updates limits damage to sampled
historical predictions. It retained the complete uniform reservoir and immediate
prediction. A fixed two-by-two comparison separated current-loss weighting from
projection of the actual Adam displacement against a sampled replay gradient.
A fifth control shrank its own update to the local projected norm without
changing direction. The primary candidate was the fixed .75-current-weight
projected rule; it was not selected from outcomes on the new seeds.

The [completed cohort](../reports/adam_revision_results.md) contains all 12 shared
prefixes and 72 first-introduction episodes across six new seeds and both
architectures, including fresh-model references. Fresh qualification passes.
Combined acquisition Brier AUC increases by .00499 conditional and .00547
recurrent, worsening all six seed pairs in each architecture. The mean retention
cost exceeds the .005 limit for conditional mode 1 and both recurrent modes.
Several retention intervals are wide; the unchanged mean-based guardrails
still fail. Survival costs remain within their declared tolerance.

Both the primary continuation screen and the separate directional-protection
attribution screen fail in both architectures. Current emphasis alone also
worsens acquisition in every seed, by .00575 conditional and .00480 recurrent.
This differs from the earlier replay-recency interventions: changing loss
weights with a full uniform reservoir does not change its membership or
sampling policy. Protection alone has small secondary effects that do not
justify promoting it as a replacement winner. Stop expansion of this fixed
combined recipe and retain immediate uniform replay as the reference.

The constraint concerns first-order reported BCE on one replay packet. It does
not guarantee finite-step improvement, validity of the remembered target, or
retention throughout the environment. Extra gradient, projection and diagnostic
work is counted separately. The experimental fork occurs at a known first
introduction; this study cannot establish autonomous lifetime operation, noise
identification, or compounding representation learning. Its failure does not
qualify this recipe for expansion to sequential acquisition, returns, revisions
and matched clean/noisy branches, nor establish that all interference-control
methods fail.

## Proposed retention criterion: marginal predictive usefulness

The user suggested weighting retention by how closely predictions match
reality and reducing emphasis on less important information. The
[predictive-retention note](predictive_retention.md) separates reliability on
new observations from a component's marginal contribution to those predictions.
A candidate value measure is the increase in later prediction loss under a
specified matched intervention that removes or withholds the component.
Both forecasts must precede the later feedback. For replay, deleting an entry
does not erase what it already taught the weights; comparing rehearsal with a
declared replacement requires a different intervention.

This remains an untested retention criterion. Easy redundancy, rare recurring
hazards, obsolete relationships and biased reports complicate estimation. A
short-window predictive score does not uniquely define long-term importance.
The earlier frequency/sensitivity priority did not directly test this value,
and the evidence gate compared whole predictors rather than individual memories.
The completed selective-update study tested sampled interference as a bounded
proxy for threatened predictions. Its failed combination did not estimate
marginal future value or show which experiences intrinsically deserve retention.

The original diagnostic proposal was to validate such a score while leaving
uniform memory unchanged. At fixed assessment times, compare bounded matched
rehearsal forks against a declared replacement, rank candidates using one future
feedback window, and evaluate that ranking on a later disjoint window and
delayed returns. Include uniform choice and low original prequential packet
error as controls. Original error depends on the learner's competence at the
time; short-window value can miss rare, temporarily absent knowledge. Repeated
assessments belong within seeds, with return gaps and order counterbalanced.
That proposal led to the completed predictive-value study and then the
state-substitution diagnostic. The former failed both policy-oriented tiers;
the latter did not replicate a common deterioration pattern. An adaptive
retention criterion remains untested, and neither result authorizes promoting
the fixed score into memory eviction.

## Next bounded architectural comparison

Retire the cached-score/rollback branch. The proposed next experiment returns
to the unresolved representation-acquisition problem: a learned reusable core
plus a small adaptive feature pathway, compared with the identical-capacity
architecture trained jointly. Start both from the same prefix learned from
random initialization, keep uniform causal replay and match learning exposure.
The adaptive path must learn features from observations, not merely remix a
frozen representation. This differs from the earlier dual-path pilot's random
stable-path start, without assuming that change resolves its failure.

Run the intervention on actual continuing learner trajectories, introduce a
genuinely unseen dependency later, and require acquisition together with
valid-old retention and return performance. A new protocol and fresh evaluation
must precede any fitting. This is a design direction, not an implemented or
validated algorithm. A frozen core with one successful adaptive path would
still not establish indefinite acquisition of new core representations, noise
robustness, longer planning or compounding lifelong learning.

## Position relative to prior work and project identity

Contextual inference is an established research direction. The
[COIN motor-learning theory](https://cbl.eng.cam.ac.uk/lengyel/publications/heald-nature-2021/)
distinguishes updating memories from changing their expression. That distinction
is relevant to the present reuse result; it does not validate our architecture
or its joint-persistence objective. [Caccia et al.](https://proceedings.mlr.press/v232/caccia23a.html)
study recurrent replay for task-agnostic continual reinforcement learning and
provide a reason to include a strong history-aware comparison. This project is
not currently a reproduction of either method or a demonstrated novel algorithm.

Adam is the working project name. Continual relational learning describes the
broader question without requiring a particular plasticity mechanism. The
optional expansion, Adaptive Dynamics and Associative Memory, makes no claim
to an exclusive technical term or a successful general algorithm. The current
reference is an evidence-routed conditional predictor with uniform replay.

Critical periods can remain an optional hypothesis about when and where to
permit change. ACP-CL can remain the name of that historical experimental line.
The Adam name preserves source archives and reproducible commands; package
paths and CLI names stay `acp_cl` and `acp-cl`.
