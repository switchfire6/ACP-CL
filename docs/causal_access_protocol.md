# Causal access to fixed predictive functions: development protocol

Specified 2026-09-28 UTC before scoring the new cohort. This is one bounded
follow-up to the saved core/residual output diagnostic. The completed
core/residual architecture study still fails its original acquisition and
retention screens; no result here changes those decisions.

## Question and scope

Can past performed-action outcomes identify when a preserved core or its
learned combined predictor is more useful, without an announced environmental
switch? The candidate changes only the probability combination at inference.
It does not change fitting, replay membership, optimizer state, or capacity.

The user proposes that good-enough retention may be preferable to perfect
retention. This protocol uses finite error and survival tolerances, as several
earlier studies already did. It does not require zero forgetting. The frozen
evaluation stage nevertheless does not compare perfect parameter protection
with partial protection, or test an optimum stability/plasticity tradeoff.
It tests access to existing functions after their ordinary learning has ended.

Useful relative differences are insufficient when both available functions
are inadequate. Independently trained single-law references therefore qualify
the old and novel functions before any online outcome earns promotion. These
references are diagnostics, not experts accessible to the candidate.

## Fresh cohort and fixed fitting

Use six unused seeds 17001--17006 and both original architectures, conditional
and recurrent. Analyze each architecture separately. The first novel cue is
balanced over three dependency families and the two original base modes:

| Novel dependency | Seeds |
|---|---|
| Directional efficiency | 17004, 17005 |
| Arrival delay | 17001, 17006 |
| Supply timing | 17002, 17003 |

For each of the 12 architecture/seed jobs:

1. Fit the ordinary width-64 core from random initialization on four alternating
   base-law blocks of 1,024 arrivals each, using the unchanged prefix schedule.
2. Attach the same zero-output raw-input residual as in the completed study.
   Independently fork joint and separate arms, copying the full learned prefix,
   Adam state, gradients, replay contents and generators, and actual history.
   Joint trains both paths; separate freezes the entire learned core.
3. Give both arms identical 1,024-arrival maintenance and 8,192-arrival novel
   streams. Maintenance uses the final prefix base law. Novel learning activates
   the first dependency under that same base mode. Do not reset anything at
   either boundary. Freeze both arms after novel learning; do not train returns.
4. Independently train two reference specialists from random initialization,
   one on the exact original old law and one on the novel law, for 8,192 arrivals
   each. Each uses the same composite joint architecture and ordinary training
   rule, with its own empty initial Adam/replay state and causal training stream.
   Initialization seeds are base seed + 191311 for old and + 193321 for novel;
   their data channels are causal_access_reference_old and
   causal_access_reference_novel. The specialist is not a copy of a fitted arm.

The old law is Law(seed % 2); the novel law is stage_law(seed, 1). No externally
pretrained representation supplies any initial weights. The core has width 64,
four conditional experts, context width 12 and decoder width 64; the residual
has width 32, context width 8 and decoder width 32. Keep the existing interaction
features, lr .002, gradient norm limit 5, batch size 32, 12 Adam updates per
packet, uniform replay capacity 16 packets, and unchanged equal current/replay
performed-action BCE. Feedback is clean. Preserve the existing physical outcome
horizons 4, 8 and 12 and the five-action world.

The fixed training budget is 466,944 arrivals, 14,592 packets and 175,104 Adam
updates over the 12 jobs. Freezing reduces backward work; joint and separate
are matched for exposure, architecture capacity and update count, not FLOPs.
No extra training is purchased with any compute saved by freezing. Use six
single-thread workers with affinity 1365. A disjoint reduced engineering smoke
may check implementation, but may not select scientific settings or thresholds.

## One probability mixer

Let p_C be the separate arm's native projected core probabilities and p_F its
native projected combined probabilities. Each uses its own unchanged inference
computation with the same causally available support. In particular, the
conditional core retains its own native evidence routing. Mix probabilities,
not logits. Convex mixtures preserve the existing monotone survival curves.

For each performed action, preserve the core and full terminal-horizon
probabilities at the time they forecast that outcome. Once its feedback arrives,
store the difference of Bernoulli log scores. At the next packet use

\[
d_t=\sum_{i\in W_t}\left[
 y_i\log\frac{\bar p_{F,i}}{\bar p_{C,i}}
 +(1-y_i)\log\frac{1-\bar p_{F,i}}{1-\bar p_{C,i}}
\right],\qquad w_t=\operatorname{sigmoid}(d_t),
\]
\[
p_t=(1-w_t)p_C+w_t p_F.
\]

The bars denote clipping to [1e-6, 1 - 1e-6] for the log score only. They do
not replace the reported or deployed probabilities. W_t contains at most the
last 32 feedback records and only records from earlier packets. The fixed
evidence strength is one. Before any stream feedback the score window is empty
and w_0 = 1/2. Use terminal outcomes alone for the evidence score because the
three horizon labels are correlated. Do not count them as three independent
pieces of evidence.

All 32 cases in a packet use the same preceding support and already formed
weight. Only after forecasting the packet may its performed outcomes update
the score window and shared support for the next packet. Store original
forecasts; do not rescore earlier cases using later histories or probabilities.
No learned gate, temperature or window search, reset, target-law flag, boundary
indicator, or fresh diagnostic support enters this controller.

## Frozen switching stream and controls

After fitting, present 16 alternating blocks of 1,024 arrivals: old first,
then novel, repeated eight times. This is 16,384 arrivals per job and 196,608
over the cohort. The externally specified schedule is not supplied to any
predictor or controller. Start with the actual last novel-training packet as
support and the empty evidence-score window. Preserve history and the score
window across every switch; there is no boundary-triggered intervention.

Evaluate five policies from common forecasts:

- Always core: p_C.
- Always full: p_F.
- Fixed half: (p_C + p_F)/2.
- Causal mixer: the sole candidate defined above.
- Joint: the matched jointly trained composite predictor.

All policies receive identical observations, random performed actions and
feedback, and the same actual causal support. The environment uses exogenous
actions; a policy's greedy action does not alter its future training or support
distribution. All candidate and control forecasts are produced under the
declared shared component-forward budget, with forward work recorded. Do not
claim that naturally deploying all five policies has the same cost as deploying
one predictor, or claim an efficiency improvement from shared evaluation work.

The learner sees images, performed actions and their outcomes only. The
evaluator owns law identity, physical truth for all actions, affected-case
masks and cue-flipped probes. Query truth does not enter the mixer, support,
replay or weights. For a novel cue-flip probe, hold the actual support and
already computed causal weight fixed, change only the designated query cue,
and evaluate against the same physical truth. Probe feedback never enters the
ordinary stream.

Score every arrival, including the first packet after each unannounced switch
and every recognition delay. When histories and current observations do not
yet distinguish laws, the controller cannot know that the environment changed.
No analysis may omit this unavoidable portion of the stream.

## Qualification: available competence

At frozen novel-end weights, use 512 held-out evaluator cases and two fresh,
independent 32-record target-law support replicates for each old/novel probe.
Compare methods on identical cases and corresponding supports. Average the
two replicate metrics within each seed before comparing seeds. These fresh
supports are privileged availability diagnostics, never online inputs or
evidence of zero-delay recognition.

Define cue benefit as flipped-query affected-case Brier minus correct-query
affected-case Brier, so positive values mean using the relevant cue helps.
The training-only action/horizon marginal uses only each specialist's own
performed-action training counts and the existing half-success/unit-count
smoothing rule. No held-out outcome fits this reference.

Qualification requires all of the following, separately per architecture and
both over all six seeds and within each of the three two-seed cue groups:

- Each specialist, old and novel: mean all-case Brier <= .12 and mean Brier
  improvement over its training-only marginal >= .02.
- Novel specialist: mean correct-cue affected-case benefit >= .002.
- Separate core on old: mean all-case Brier no more than .02 above the old
  specialist's error.
- Separate full on novel: mean affected-case Brier no more than .02 above the
  novel specialist's affected-case error.
- Separate full on novel: mean correct-cue affected-case benefit >= .002.

The .12 ceiling and .02 reference tolerances specify useful but imperfect
building blocks for this development decision; they are not universal
definitions of competence. The novel affected-case and cue requirements prevent
easy unaffected cases or an insensitive fallback from concealing failure to
learn the new dependency. Two-seed group checks have limited precision.

Complete the entire fixed cohort and frozen stream even if qualification
fails. Such results are descriptive and cannot promote the architecture.
Do not add exposure, substitute seeds, remove a cue, insert checkpoints or
relax criteria to rescue an unavailable component.

## Primary decision and good-enough tolerances

Brier always means the mean squared probability error across the three
horizons and all five actions, using all cases unless affected cases are
explicitly specified. For stream metrics pool applicable cases within each
seed and slice, then give each seed equal weight. Do not average packet focus
means when packets contain different numbers of affected cases. Whole-stream
metrics include all 16 equal-length blocks; report old and novel slices
separately. Greedy survival is true terminal survival under the policy's
highest predicted terminal-survival action, using the fixed existing tie rule.

Eligibility requires a complete valid cohort and every qualification above.
The causal candidate must then pass every fixed practical screen:

| Quantity, per architecture | Required mean and seed consistency |
|---|---|
| Whole all-case Brier, causal minus always full | <= -.002 and strictly lower in at least 5/6 seeds |
| Whole all-case Brier, causal minus joint | <= +.005 |
| Whole all-case Brier, causal minus fixed half | <= -.001 and strictly lower in at least 5/6 seeds |
| Old all-case Brier, causal minus always full | <= +.005 |
| Novel affected-case Brier, causal minus always full | <= +.005 |
| Old greedy survival, causal minus always full and causal minus joint | Each >= -.01 |
| Novel greedy survival, causal minus always full and causal minus joint | Each >= -.01 |
| Novel correct-cue affected-case benefit for causal | >= .002 overall and in every cue group |

The fixed-half comparison tests whether past feedback adds value beyond a
constant ensemble. The joint guard prevents improvement against a weak full
predictor from masking an unacceptable cost relative to ordinary joint
training. Slice guards prevent one condition's gain from concealing material
harm on the other. Report whole-stream survival as well; its mean guard is
implied by the two equally long slice guards.

These mean thresholds permit small losses. They are development resource
allocation rules, not statistical noninferiority or equivalence tests. Use
20,000 paired seed bootstrap resamples with RNG seed 17291 for descriptive
95% intervals. The six seeds per architecture are the independent units;
queries, actions, horizons, blocks and support replicates are not independent
replications. Publish raw gate outcomes even for an ineligible architecture,
with eligibility shown separately. Do not pool architectures or promote an
alternative policy or favorable subgroup after seeing outcomes.

## Stopping rule, locks and interpretation

Lock the scientific source, configuration, runtime, this protocol, relevant
tests and scientific analysis before main fitting or scoring. Keep all older
scientific source, data and reports unchanged. Save the fresh fitted states,
original pre-feedback component forecasts, score-window provenance, actual
support, raw truth/probe arrays and resource counters. Check full frozen-model,
optimizer, gradient and replay invariance during evaluation; only the declared
external support and evidence window evolve. Independently reconstruct score
weights and NumPy metrics, verify cue-probe isolation and complete cohort
coverage, and retain any failed engineering artifacts with their correction
history. Verification of recorded forecasts does not imply an independent
replication of every fitting update.

The study is a fresh six-seed development test motivated by previous inspected
results, not confirmatory evidence for a general algorithm. No hyperparameter,
architecture, replay, evidence-window or specialist-exposure sweep is allowed.
If necessary competence or online usefulness fails, stop this frozen-core /
additive-residual branch. The failure remains informative about this recipe;
it does not refute continual learning or modular methods in general.

If qualification and the causal-access screen both pass, that only supports
access to the available frozen functions in this finite stream. A separate
continuous-training comparison and another task would still be needed. This
experiment does not establish improved acquisition, an optimal forgetting
level, autonomous capacity renewal, consolidation, recursive self-improvement,
noise robustness, longer physical planning, or compounding lifelong benefit.
