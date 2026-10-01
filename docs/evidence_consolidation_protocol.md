# Adam: repeated evidence before committing a predictor

Status: prospective diagnostic protocol, written before development or pilot
fits. This document specifies one fixed candidate and its controls. It does not
revise the completed replay-renewal screen or select a setting using its seeds.

## Question and limited claim

Can repeated predictive evidence improve the decision to commit a newly learned
predictor, relative to adopting it immediately, after a fixed delay, or after
one favorable packet? Does any advantage remain useful through repeated changes
and returns, rather than only during the first change?

This experiment isolates **acceptance of learned updates**. All policies share
exactly one uniformly replay-trained draft trajectory for a given model and
seed. Consequently they learn exactly the same candidate representations. A
positive result demonstrates better use of that trajectory for prediction; it
does not establish better representation learning, protected plasticity, or an
autonomous solution to continual learning. A negative result rejects this fixed
acceptance rule in this testbed, not every evidence-guided learning mechanism.

## One draft, fixed proposals, separate committed predictors

The draft is the existing conditional or qualified recurrent model, initialized
from scratch. Keep its learning rate .002, 16-packet uniform reservoir, causal
32-record history, and 12 current-plus-replay gradient updates per 32-arrival
packet. Its optimizer, memory, and random generators continue without resets.

Number incoming packets globally, including the initial prefix. Partition them
into non-overlapping windows of W = 8 packets. At the beginning of each window,
freeze the current draft weights as the proposal Q. The draft continues training
while Q remains fixed. Each delayed policy has its own committed weights C,
fixed throughout the window. Every policy uses the same raw causal history;
keeping weights fixed does not freeze or conceal incoming observations.

For packet j in a window, form predictions before its outcomes enter either
the draft or the causal history. On the performed actions only, define terminal
Brier loss against the reported H=12 outcomes:

    L_j(P) = (1 / 32) sum_i (P(y_H=1 | x_i, a_i, h_j) - y_reported_i,H)^2
    d_j = L_j(C) - L_j(Q).

The history h_j contains only preceding packets. Current outcomes, latent laws,
clean counterfactuals, corruption indicators, and boundary labels cannot enter
these predictions or any acceptance decision. Do not multiply the correlated
three-horizon labels as independent evidence. Positive d_j favors the proposal.

At the end of each window apply the following policies. All delayed copies use
the exact proposal evaluated during the window, not the draft's newer weights.

| Policy | Predictor used between decisions | Decision at window end |
|---|---|---|
| immediate | Live draft after preceding updates | No acceptance gate |
| periodic | Own committed snapshot | Always replace C with Q |
| single | Own committed snapshot | Replace only if d_1 > .002 |
| sustained | Own committed snapshot | Replace only if mean(d_1,...,d_8) > .002 and at least 6 of 8 d_j are positive |

The single-packet policy also waits until window end; this matches the delay
and available proposal of the sustained policy. Each policy's d_j uses its own
committed predictor. The periodic comparison separates repeated evidence from
the benefit or cost of merely deploying older weights. The single comparison
tests the declared multi-packet rule, which combines averaging and sign
consistency; it does not separately identify those two components.

The rule is a fixed engineering threshold, not a calibrated hypothesis test,
posterior probability, or anytime-valid confidence guarantee. There is no
threshold/window search and no adoption decision supplied by the evaluator.
Windows continue across every physical change. Block lengths happen to be
multiples of W; no learner receives their labels, but sensitivity to window
phase is outside this diagnostic.

## Fixed continuous stream

Use new development seeds 11001--11002, followed by untouched pilot seeds
12001--12006, with both conditional and recurrent architectures. Development
checks execution, recovery, measurement, and plausible operating behavior; it
does not select thresholds or architectures. Preserve all development outcomes.
Any necessary scientific redesign requires an explicit protocol revision before
pilot fitting, with the reason recorded. Do not retrofit the pilot after viewing
its outcomes or recycle completed study seeds for tuning.

First present four alternating original-mode prefix blocks of 1,024 arrivals.
Then present 18 continuous blocks of 8,192 arrivals. Blocks 1--3 add the three
previously irrelevant temporal dependencies in the existing seed-balanced cue
order. Blocks 4--18 consist of three repetitions of this five-block cycle:

1. All three dependencies, original cue meaning, clean reports.
2. The same physical law with noisy reports.
3. Clean recovery under that same law.
4. Return to the exact original base law, with none of the three additions.
5. All three dependencies with the directional-efficiency cue meaning revised.

Here "original cue meaning" is the unrevised full law, and "original base law"
is the original returned environment. Keep these distinct in stored records.
The first and third cycles use independent random-vector replacement on .2 of
reports. The middle cycle uses .8 replacement for 16 of each 64 packets and no
replacement for the remaining 48 packets, also .2 in expectation. Corruption
uses the existing monotone three-horizon report-vector replacement mechanism.
Observations and performed actions are exogenous and matched across policies.
No policy is told whether a block is an addition, return, revision, or noise.

This is 151,552 arrivals per main trajectory, including the prefix. There is
one shared draft trajectory per seed and architecture, with four deployed
prediction policies scored alongside it. It is not four independently trained
learners. Each seed also supplies fresh-weight references for the three new-cue
blocks, using uniform replay, the same preceding causal raw history, and the
same 8,192-arrival exposure for that stage.

Fresh references retain the existing learnability requirements: marginal Brier
gain at least .02 and correct-cue benefit at least .002 in each available stage
and cue group, separately per architecture. Keep failures, complete the fixed
cohort, and qualify interpretation rather than increasing exposure afterward.

## Outcomes and the longer-horizon question

Acceptance scores and scientific evaluation are separate. The evaluator may
use clean counterfactual outcomes and law metadata, but cannot transmit them to
the draft, proposals, history, or acceptance policies. On EVERY incoming packet,
before training or revealing its reports, score each deployed policy against
the clean physical outcomes of those same performed actions, averaged across
the three horizons. For n arrivals the primary prequential physical error is

    E_n(P) = (1 / (3 n)) sum_i sum_H (P(y_H=1 | x_i,a_i,h_i) - y_clean_i,H)^2.

These clean outcomes are evaluator-only paired outcomes, not extra feedback.
The main online evaluation uses the actual preceding raw history, including
corrupted reports. Any adoption made after a packet's reports applies to the
next packet; it cannot change the prediction already scored for that packet.

Use E_n over the 18 post-prefix blocks as the cumulative mean-error endpoint,
and its mean over blocks 14--18 as the late endpoint. Also retain cumulative
total excess loss, per-block error, and late-minus-early policy gaps.

Independent query panels every 256 arrivals retain the prior all-action,
all-three-horizon Brier definition, affected/valid subsets, and terminal survival
from the predicted best action as secondary measurements. These probe AUCs
are not the primary every-arrival prequential loss. Clean-support retention
panels are separate measurements of stored knowledge, not information available
to the deployed policy. Record complete per-block values and cumulative curves
rather than only final endpoints.

For each architecture and seed also report:

* Affected-subset acquisition AUC across the three introduction blocks, including
  each cue separately and the existing fresh-reference cue checks.
* Clean-support terminal error on still-valid physical subsets after each
  introduction and each later block, with each base mode visible. Report levels
  and changes; policies may have different starting levels.
* Each original-base return block and each revision block's prediction and
  survival AUC, rather than merging return with new learning.
* Prediction/survival AUC in each noise and clean-recovery block, adoption
  frequency, proposal age, and window score distributions by evaluator-owned
  condition labels. A noisy-block adoption is not automatically an error.
* Direct between-policy clean-truth error differences on the matched noisy
  packets. This tests their relative deployed performance during noise. Without
  a twin clean trajectory it does not estimate the causal error penalty caused
  by corruption; preceding clean blocks are different learner ages and states.
* Phase-matched policy advantages in cycle 1 (blocks 4--8) and cycle 3 (blocks
  14--18), plus late-minus-early advantage. Those cycles have the same condition
  order and noise rate, with fresh samples. The burst-noise middle cycle has a
  different temporal arrangement and is not pooled as an identical repeat.

A cumulative advantage can grow at a constant rate without any compounding
improvement. Therefore report both cumulative excess loss and the instantaneous
or per-block policy gap. A larger advantage in the third matched cycle is
evidence of increasing usefulness over this finite stream, subject to sampling
uncertainty. It is not evidence of indefinite improvement. Because draft
training is shared, it also cannot establish compounding representation gains.

Extending the training stream tests memory through more changes and absences.
It does not extend the outcome/planning horizon: the physical labels still end
at H=12. The simulated world and exogenous-action data remain limits on
generality and on claims about autonomous action or exploration.

## Prospective pilot screen

The main candidate is sustained. Evaluate it separately against immediate and
periodic; report sustained versus single as the mechanistic comparison. Each
architecture must independently meet all requirements; do not pool one model's
success with another model's failure. Using paired seed means, require:

1. Every fresh-reference qualification group passes.
2. Sustained improves post-prefix cumulative mean Brier by at least .002 and
   late mean Brier by at least .002 versus each main control. Its overall
   post-prefix error improves in at least 5 of 6 seed pairs versus each control.
3. Mean affected-subset acquisition AUC across the three introductions rises
   by no more than .005 versus immediate.
4. Mean prequential physical error across the three original-base return blocks
   rises by no more than .005 versus immediate. Mean prequential physical error
   across the three noise blocks separately rises by no more than .005.
5. Mean terminal clean-support still-valid error across all 18 block endpoints
   rises by no more than .005 versus immediate. Report every endpoint alongside
   this aggregate so offsetting gains and harms remain visible.
6. Mean terminal-horizon survival AUC across all 18 post-prefix blocks declines
   by no more than .01 (one percentage point) versus immediate. Report every
   block's survival as well as this aggregate requirement.

These are fixed pilot choices, not universal tolerances or proof of
noninferiority. Preserve every failed criterion. Report all paired seed values
and descriptive 95% paired-seed bootstrap intervals using 20,000 resamples and
analysis seed 27192026, as in the prior study. Seeds are the independent units;
packets, windows, blocks, and horizons do not create extra independent seeds.
Flag intervals crossing tolerances even when a mean-based screen passes.

## Identifiability and resources

Repeated disagreement does not uniquely identify changed physical dynamics.
Persistent reporting corruption, unobserved context, or miscalibration can also
repeatedly favor a different predictor. In particular random label replacement
can reward movement toward the corrupted-label mean. A sufficiently persistent
reporting process can be observationally indistinguishable from a change in the
environment. The experiment measures behavior under specified processes; it
does not solve that general identifiability problem or provide clean labels to
the gate.

Similarly, a proposal that predicts the current context better may erase useful
knowledge about an absent context. A global acceptance gate contains no explicit
constraint protecting those absent relationships. Retention panels test this
possible failure rather than assuming persistence of evidence prevents it.

All policies use the same raw observations, draft gradient updates, reservoir,
and proposal schedule. No additional raw validation archive is required: scores
can be accumulated as scalars and sign counts. However a standalone delayed
policy retains a draft, frozen proposal, and committed model, and executes
additional inference. Count these weights and forward passes. Sharing draft
work among experimental arms reduces run cost; it does not make a deployed
multi-snapshot policy free or give the immediate learner the same useful work.
Periodic and single controls use the same snapshot scaffold and evidence passes
to isolate the acceptance decision. Report the immediate learner's lower
necessary cost and make no efficiency-superiority claim from this diagnostic.

## Integrity, recovery, and decision

Before pilot fitting, lock source, configuration, this protocol, and analysis
rules; preserve development and engineering attempts separately. Use the
existing bounded CPU recipe and record actual worker affinity and thread count.
Checkpoint draft optimizer/replay/history, every committed snapshot, frozen
proposal, global window offset, accumulated evidence, and all generators needed
for exact recovery. Flush completed writes before atomic replacement.

Required checks include pre-feedback scoring, unchanged proposal weights during
validation, copying the evaluated snapshot, no boundary-conditioned reset,
policy-specific committed comparisons, identical draft learning and budgets,
fresh initialization, bounded replay, independent clean evaluation, unchanged
legacy scientific sources, and exact interrupted-run continuation. Test cases
must include accepting/rejecting windows and interruption within a window.
Recompute saved final predictions and acceptance decisions from trusted local
state and stored evidence; distinguish this from merely hashing saved files.

If sustained passes, independently confirm the consolidation result before
changing the draft's learning rule. A subsequent experiment could make retained
predictors constrain updates or allocate plasticity, which would require its own
matched training controls and fresh data. If it fails, identify whether fixed
delay, insufficient evidence, obsolete proposals, retention damage, or persistent
noise accounts for the observed pattern; do not tune on this pilot or call the
broader continual-learning objective disproved.
