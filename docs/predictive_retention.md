# Predictive reliability and the value of retention

Status: conceptual note updated after the 2026-09-27
[predictive-value diagnostic](../reports/predictive_value_results.md).
The specific matched-rehearsal score has now been tested and fails both fixed
tiers in both architectures. The broader retention policies, calibration
schemes and representation ablations below remain proposals.
The subsequent [state-and-environment diagnostic](../reports/rehearsal_state_results.md)
varied weights, complete Adam state and causal anchor on fresh seeds. The
earlier common loss of selection usefulness did not meet its replication
criteria in either architecture. Component effects remain descriptive, and a
better selection contrast did not establish better absolute predictions or
benefit over no extra rehearsal. Keep value conditional on learner state,
update and evaluation conditions; do not promote a permanent scalar importance
label or choose a state rollback from these results. The subsequent
[core/residual comparison](../reports/core_residual_results.md) retained ordinary
replay and tested architecture instead. It fails both primary screens, while
recurrent residual visual adaptation passes its separate attribution check.
Useful learning and a successful combined retention mechanism remain distinct;
unchanged core weights alone do not preserve the emitted predictions.
The subsequent [saved-output diagnostic](../reports/core_residual_output_results.md)
finds relative value in disabling the correction at old-condition return entry,
while that same correction helps novel-condition prediction. This earns one
bounded access question, not a useful retention policy. A weak core can beat a
worse combination, and lower error can coexist with lost dependence on a novel
cue. Current and past predictive scores must therefore be evaluated against
absolute competence, causal availability of feedback, and the actual new skill.

The [completed causal-access experiment](../reports/causal_access_results.md)
now finds that recurrent feedback-based access beats both full prediction and
fixed averaging, but the available old functions and novel-condition guards
still fail. Conditional access does not improve sufficiently over fixed
averaging. Both architectures are ineligible, so the tested branch is stopped.
Predictive usefulness can be conditional on current context while still being
insufficient to support a successful learning system.

## Good-enough retention and useful access

The user's latest suggestion is that demanding perfect retention may obstruct
learning. Our objective is useful behavior across changing conditions, not
zero change to every weight or indefinite preservation of every past response.
If the environment changes, retaining a dependency can still be useful while
applying its old answer unconditionally is harmful. Retention, conditional
access and successful acquisition therefore need separate measurements.

The approved [causal-access protocol](causal_access_protocol.md) permits finite
losses: component error can exceed a competent specialist by .02; specified
online error costs may be up to .005, and old/novel survival costs up to .01.
It also requires an overall error gain and preservation of useful novel-cue
dependence. These are practical development tolerances, not universal optima
or statistical proofs of equivalence. A tolerant retention criterion still
needs a useful absolute performance floor.

This experiment freezes predictors during its switching stream. It can test
whether past feedback helps select existing functions, but cannot establish
that allowing more parameter forgetting improves long-run learning. Nor does
it test whether consolidation repairs that tradeoff. Those claims would need
continuous learning, repeated new dependencies, fixed resource budgets and
comparisons against ordinary joint learning with replay.

The user's suggestion is to retain what helps predictions stay close to reality
and reduce emphasis on less useful information. That suggests two measurements:
how reliably a predictor forecasts new observations, and how much a particular
memory or representation contributes to those forecasts. They are different.
An easy, redundant predictor can be accurate while adding little; a difficult,
rarely needed dependency can be important despite imperfect predictions.

## Measure reliability before learning from the answer

Let h_t contain only prior experience, x_t the present observation, a_t the
performed action, and p_t=f_theta(x_t,a_t,h_t) a vector of H survival forecasts.
Record p_t before receiving the reported outcome vector y_tilde_t. Then score

    L_t(f) = (1/H) sum_h (p_t,h - y_tilde_t,h)^2
    R_t = (1-alpha) R_(t-1) + alpha L_t(f),     0 < alpha <= 1.

This is prequential Brier error: lower is better. Here H counts forecast
components, not elapsed physical steps. Evaluating forecasts before updating
follows [Dawid's prequential approach](https://doi.org/10.2307/2981683).
Training error on rehearsed examples is not an equivalent reliability estimate.

Check calibration separately. Within fixed probability bins b, maintain counts
n_b, mean forecast p_bar_b, and mean observed outcome y_bar_b for each horizon:

    Cal = sum_b (n_b / sum_c n_c) (p_bar_b - y_bar_b)^2.

Bins need minimum counts and an explicit window or decay rule. This is a coarse,
noisy calibration diagnostic, not a complete measure of prediction quality.
A constant marginal forecast can be calibrated while ignoring useful cues.
In Adam, ordinary feedback can be corrupted: these measurements concern reports,
not otherwise inaccessible physical truth. Clean outcomes stay evaluator-only.

## Measure contribution with matched interventions

For a component j, prepare predictors with and without a declared intervention
using data available at time s. Freeze both before the next observations arrive.
For later queries t in a fixed window W_s, predict before receiving their labels:

    v_j,s = mean_(t in W_s) [L_t(f_minus_j) - L_t(f_plus_j)].

Positive v means the component helped on those new observations. Match query
inputs, performed actions, feedback, preceding history and other state. Choose
the component and intervention before seeing W_s. This estimates a specified
intervention's predictive contribution on the observed distribution, not its
universal value or the causal necessity of a relationship in the physical world.

For a representation, a temporary mask can compare fixed weights with a declared
feature/module suppressed; specify how routing and normalization are handled.
Such ablations can create unfamiliar activations. Redundant features may each
look dispensable even though removing them together is harmful.

For a replay memory, simply deleting its buffer entry does not erase information
already learned into the weights. Instead fork identical weights and optimizer
state, perform matched updates with versus without its rehearsal, and score on
later arrivals. State what replaces the omitted rehearsal to match update work.
The estimate is the value of that rehearsal relative to its replacement, not
the value of removing every historical influence of the example. Broader subset
attribution is related to [Data Shapley](https://proceedings.mlr.press/v97/ghorbani19c.html),
which values data through contributions to a specified learner and performance
measure; it does not supply a cost-free online valuation rule.

## Bounded approximations and their costs

One possible implementation would assess only a fixed number of candidates per
interval, rotate through stored packets, and keep per-candidate counts, score
means and ages. Short matched forks cost extra updates, model/optimizer copies
and future scoring passes. Limit their number and lifetime explicitly. After
prequential scoring, the ordinary learner can use the newly received feedback.
Continual adaptive selection still needs a separate cohort for evaluation.

A cheaper vulnerability proxy evaluates the effect of a virtual current update d:

    I_j = max(0, L_j(theta + d) - L_j(theta)).

This measures threatened performance on an existing example, not its marginal
benefit on unseen queries. [MIR](https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf)
uses anticipated interference to retrieve replay examples, with a sampled search
pool to bound cost. [Gradient-based sample selection](https://arxiv.org/abs/1903.08671)
instead uses gradient diversity to choose stored examples representing learning
constraints. Both are relevant precedents, not demonstrations in Adam's world.

A possible rehearsal distribution over M stored packets is

    pi(j) = epsilon/M + (1-epsilon) softmax_j(clamp(v_hat_j,-c,c)/T),

with T>0, c>0 and 0<epsilon<=1. Estimates could shrink toward zero when evidence
is scarce. These are design choices requiring development, not selected values.
The uniform component preserves some sampling of every retained packet; it does
not guarantee rare events enter or remain in the buffer. First separating
rehearsal weights from reservoir membership would make their effects clearer.
Count scores, temporary states, gradients and extra forwards/backwards alongside
the ordinary replay budget. Equal stored packets do not imply equal resources.

## Limits and relation to the existing experiments

Accuracy alone can favor frequent, easy observations. Marginal value on a short
window can undervalue dormant knowledge; related memories can substitute for
one another. Noise can produce high interference or consistent predictive gains
from learning a biased report. See the [noise analysis](evidence_noise_limits.md).
Reliable old predictions can also become obsolete. Neither high accuracy nor
high interference justifies permanent protection without revision controls.

Importance of an outcome is a separate objective choice: a rare hazard may
deserve greater decision weight even when difficult to predict. Any hazard or
outcome weighting must be declared and measured separately from prediction error.
Observed-action feedback also cannot score outcomes of unperformed actions.

The failed [frequency/sensitivity priority](../reports/persistence_results.md)
did not measure matched marginal predictive value. The later
[evidence gate](../reports/evidence_consolidation_results.md) did score future
reported prediction differences, but compared whole frozen models for adoption;
it neither attributed value to particular memories nor changed their learning.
[DER/DER++](https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf)
provide a related output-preservation precedent: replay stored predictions,
with DER++ additionally using replay labels. Preserving an output is different
from establishing that its content remains valid or useful.

Predictive retention could therefore inform later selective rehearsal or update
protection. The [completed matched-return diagnosis](../reports/return_context_v2/summary.md)
finds deteriorated historical predictions even with matching context. The
[completed selective-update experiment](../reports/adam_revision_results.md) accordingly
tested a bounded interference constraint and failed its continuation criteria,
while the later [direct score diagnostic](../reports/predictive_value_results.md)
finds that its measured rehearsal benefit does not transfer reliably to fresh
reapplications. Its frozen-predictor secondary signal does not validate a
retention policy. Any follow-up must separately test new acquisition,
valid-old retention, recurrence, revision, noise and costs using fresh
development and independent evaluation data.

There is a useful mathematical distinction when combining these ideas. For a
single hard constraint h dot d <= 0, multiplying h by any positive importance
weight w does not change the permitted directions or the Euclidean projection:

    d - max(0, (w*h) dot d) / ||w*h||^2 * (w*h)
      = d - max(0, h dot d) / ||h||^2 * h,     w > 0.

So a future importance signal would need to affect which constraints are
sampled, how several gradients are combined, or how much violation is allowed.
Simply scaling the one replay gradient in the present projection would have
no effect in exact arithmetic. A soft correction weight would change the
mechanism, but would also relax its first-order protection. Those choices need
a new controlled comparison; none was added to the completed fixed experiment.

## Original proposal for the direct diagnostic

First validate a score without changing memory membership. At fixed assessment
times, sample a bounded set of packets from the unchanged uniform reservoir.
Make matched short training forks for rehearsing each candidate versus a
declared replacement. Freeze the resulting predictors and score their gain on
subsequent observations before those observations are learned. Use this first
window only to rank candidates. Evaluate that ranking on a later, disjoint
window, including a return of a previously absent condition. Condition labels
and truthful outcomes remain evaluator-only.

Compare the gain-selected packet with a uniform choice and a packet whose
original prequential prediction error was low. That last control tests the
simple intuition that well-predicted experiences deserve retention. Its error
must have been recorded before training on that packet. Give all choices the
same scoring and update budget, even when a control discards the extra scores.
Original packet error also depends on how mature the learner was at arrival;
record packet age and initial competence rather than interpreting it as an
intrinsic property of the example.

There is an algebraic trap: if every candidate shares the same no-rehearsal
baseline, ranking marginal gains on one window is exactly the same as ranking
the rehearsed shadow predictors' accuracy on that window. Calling those two
rankings different controls would add no information. Packet predictability
and the accuracy of a predictor trained with that packet are different signals.

The decisive question is whether an estimated gain predicts usefulness beyond
its scoring window. A packet from a rare, currently absent condition can have
almost no immediate gain and substantial value at its next return. Report
ordinary-next-window and delayed-return performance separately; a short-window
win cannot justify deleting dormant knowledge. Counterbalance return gaps and
condition order, and average repeated assessments within independent seeds.
If the score has no useful
ordering under this controlled diagnostic, adding an adaptive sampler would
only add complexity. This proposal led to the now-completed
[predictive-value protocol](predictive_value_protocol.md). Its primary validation
reapplies the chosen rehearsal from later parent states and latest anchors;
original frozen predictors are secondary. Both local and return tiers fail
for both architectures. Preserve uniform replay and stop expansion of that
fixed recipe. Any new mechanism requires a separate protocol and fresh data.
