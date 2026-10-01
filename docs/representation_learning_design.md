# Adam: adaptable representations before stronger retention

Design recommendation, 2026-09-28. This responds to the user's request for a
new direction informed by the attached continual-learning and recursive
self-improvement research. It is a proposal, not a locked experiment or a
claim of improvement. No new training has been run for this design.

## Recommendation

Build toward continual representation learning with soft consolidation.
Start with one fully trainable, history-conditioned predictor and an auxiliary
objective that reconstructs its already observed temporal input. Keep causal
episodic replay. Test whether the richer representation objective improves
subsequent acquisition and reusable competence before introducing a slow
teacher, learned replay schedule or additional controller.

The working hypothesis is:

> Learning more of the temporal structure in present observations, including
> structure not yet rewarded by the current outcome task, can improve later
> learning under a fixed memory and compute budget without requiring a frozen
> representation.

This hypothesis has not been established by Adam's existing results. Low cue
use does not locate the failure inside the encoder; the context model, output
head, optimization or weak outcome signal could also be responsible.

## How this uses the research

**Nested Learning / Continuum Memory Systems:** take the distinction between
rapid adaptation and more persistent learning. The paper describes interacting
optimization levels with different update frequencies; it explicitly leaves
general catastrophic forgetting unsolved. It also notes that rapid adaptation
depends on a sufficiently learned slower system. Our proposal is inspired by
that organization, not an implementation of Hope, a learned optimizer or the
full nested-learning framework. [Primary paper](https://abehrouz.github.io/files/NL.pdf).

**Continual self-supervised learning:** add an established literature missing
from the report's main recommendations. CaSSLe uses a predictor to relate new
representations to past ones and demonstrates continual self-supervised
learning on several settings. Its findings motivate combining representation
learning with later consolidation; our proposed sequence autoencoder is a
different, smaller mechanism, not a CaSSLe reproduction.
[CaSSLe, CVPR 2022](https://arxiv.org/abs/2112.04215).

**Soft protection:** projected functional regularization found that directly
constraining features could reduce plasticity in its experiments, and used a
learned projection to preserve earlier information while permitting new
features. This gives a concrete interpretation of the user's good-enough
retention idea: preserve useful recoverable information while allowing its
internal encoding to change. That does not guarantee preserved behavior.
[Primary paper](https://arxiv.org/abs/2112.15022).

The attached report overstates several guarantees. OGPSA describes a local,
first-order preservation constraint, not global protection of all capabilities.
FOREVER uses an optimizer-displacement clock and a forgetting-inspired
scheduler, not an exact universal forgetting time. Neither motivates another
projection or replay-schedule sweep after Adam's previous failures.
[OGPSA](https://arxiv.org/abs/2602.07892),
[FOREVER](https://aclanthology.org/2026.acl-long.1144/).

The broad ingredients are established; no novelty claim is made. The research
opportunity is to measure whether this learning objective addresses the
specific acquisition and competence failures, then whether a carefully
separated consolidation mechanism helps over repeated changes.

## Why this differs from the stopped branch

The [latest study](../reports/causal_access_results.md) shows useful recurrent
access improvements but inadequate available old competence and failed
novel-condition guards. A better selector cannot be assumed to fix the
underlying learning problem.

Earlier [dual-path consolidation](dual_path.md) added predictor outputs,
froze the stable path between events, distilled the combination and reset the
fast path. [Evidence consolidation](evidence_consolidation_protocol.md)
delayed adoption of weights from an otherwise unchanged learning trajectory.
The proposed first test changes the representation-learning objective and
keeps the predictive model trainable and immediately deployed throughout.
It does not add another frozen core or sum two competing predictors.

There is also no missing-context replay bug to fix. In
`src/acp_cl/conditional/learner.py`, `Packet(self.history, current)` stores each
query's actual preceding support, and replay trains using `item.support`.
Recurrent and conditional models already learn through that context. Preserve
this property; re-pairing old queries with unrelated current feedback could
create contradictory training examples.

## Smallest concrete architecture

Let x contain the four already observed pre-action frames, h contain only
preceding performed-action experience, and a be the performed action.

\[
z=E_\theta(x),\qquad c=C_\phi(h),\qquad
\widehat y=P_\psi(z,c,a),\qquad \widehat x=D_\omega(z).
\]

- E is the visual/temporal encoder, initialized from scratch and always able
  to learn. Start with the existing single recurrent predictive backbone so
  the objective is the principal architectural change.
- C infers current conditions from past experience. Its state can respond
  rapidly; this is distinct from changing shared network parameters.
- P predicts the observed outcome vector. Training still receives outcomes
  only for the action actually performed.
- D is a small training-only reconstruction head, attached to the exact same
  final visual latent that P uses. There is no raw-input skip around z, no
  separate reconstruction encoder and no hand-labeled future-cue target.

On current examples and their original-context replay:

\[
\mathcal L=\mathcal L_{\mathrm{performed\ outcome}}
       +\alpha\,\mathcal L_{\mathrm{observed\ sequence}}.
\]

The auxiliary target is the normalized input sequence that the learner has
already seen. It is not an unseen future observation, a simulator variable,
a latent law label or the outcome of an unperformed action. A fixed bottleneck
and uniformly averaged frame/pixel loss provide a simple starting point; no
cue-specific cropping or weighting is chosen from evaluation results.

The motivation is specific to the current testbed: initially irrelevant
temporal cues later affect consequences, and their final frames are identical.
The objective may encourage retaining information about those sequences.
It may also waste capacity on nuisance details or ignore rare relevant pixels.
Reconstruction quality alone is not a success criterion.

## First experiment and decision structure

First qualify learnability with fresh and interleaved reference fits under a
bounded development budget, with every dependency included. If competent
behavior cannot be learned when interference is reduced, do not proceed to
another elaborate retention experiment. This is a proposed new qualification,
not a reinterpretation of any completed study.

The mechanism comparison then has three fixed arms:

| Arm | Learning objective | Question |
|---|---|---|
| Outcome replay | Performed outcomes and original-context replay | Current trainable reference |
| Final-frame auxiliary | Same, plus reconstruct the final frame | Does generic reconstruction regularization suffice? |
| Full-sequence auxiliary | Same, plus reconstruct all observed frames | Does preserving temporal information help later learning? |

Use the same predictive encoder, incoming experience, original-context replay,
initialization and bounded memory. Decoder output dimensions can be matched
by repeating the final-frame target. Count decoder parameters, stored state,
forward/backward work and wall time. A control that wastes extra computation
is not enough for an efficiency claim: include an outcome-only comparison
allowed to spend the candidate's extra compute on useful outcome updates.
Fix its allocation by engineering cost measurements before scientific scoring.

Keep this first study clean-feedback and fully plastic. Introduce all three
dependencies across a continuing stream, then include returns and revised
relationships. Preserve every switch and adaptation delay. Lock the exact
schedule, auxiliary weight, normalization, exposure, fresh seeds and numerical
decision tolerances before comparative fitting; these choices are not fixed
by this design note. Bound calibration on disjoint development data and retain
failures. Do not select the final setting from the main outcome curves.

Primary evidence must concern acquisition and retained useful behavior:

1. Later-dependency learning curves and sample/compute efficiency, with cue
   sensitivity as a behavioral check and each dependency reported separately.
2. Absolute old-condition competence and the complete online prediction and
   survival costs, allowing explicitly declared small losses.
3. Performance against both outcome replay and the generic auxiliary control,
   plus the useful-compute control. A reconstruction-only gain is insufficient.

Fresh weights and an evaluator-only encoder-freeze comparison can help
distinguish reuse from further feature learning. Decoding a cue from a latent
probe is only a diagnostic, not proof that the predictive model uses it.
If the temporal objective cannot improve the behavioral requirements within
the bounded study, stop this simple reconstruction recipe without adding a
teacher to rescue it.

## What comes after a successful first test

A second study could add a slowly updated representation teacher and a learned
projection from current features to the teacher's features. Use stored causal
experience and actual outcome labels alongside a finite consistency penalty;
allow an explicit discrepancy budget rather than demanding identical features.
The online encoder remains trainable, and predictions use the current learner
rather than waiting for adoption of an old snapshot. Teacher errors, extra
memory and training work require controls; averaging or distillation alone
does not make a target correct.

Test outcome replay, the qualified representation learner, soft consolidation
and their combination with fixed total resources. Compare early and late
acquisition over repeated new dependencies. A growing cumulative advantage
can arise from a constant per-step gain; compounding would require improving
per-task learning efficiency or an increasing matched-condition advantage.

The current input cues are already observable before becoming useful. Success
there would support acquisition readiness and reuse, not unlimited discovery
of unseen representations. A subsequent test must introduce genuinely new
observable structures and a separate domain. A relational world model trained
on observed post-action trajectories is a more substantial possible direction;
it would need a new benchmark with the same richer observations available to
every comparator, since extra information could otherwise explain any gain.

Recursive self-improvement is not part of this first experiment. It would
require evaluating changes to the learning procedure itself with independent
external checks; improving a fixed learner on a stream does not establish it.
