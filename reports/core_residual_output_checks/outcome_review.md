# Independent interpretation of the saved-output diagnostic

Reviewed 2026-09-28 03:30 UTC after the complete diagnostic. This review uses the [declared analysis](../core_residual_outputs/development/summary.json), [fixed allocation rules](../../docs/core_residual_output_review.md), and [original continuous-learning result](../core_residual_results.md). No models were fitted or altered for this review. Values below are raw Brier units, not percentages.

## Decision and what it earns

Both architectures meet the pre-score **GO** allocation heuristic. That warrants considering one narrowly specified causal-output test. It does not reverse either failed acquisition/retention screen, establish a usable router, or validate whole-core freezing.

| Separate arm, refreshed support | Conditional | Recurrent |
|---|---:|---:|
| Return-entry old error: full minus core | .012873 [.004705, .019260] | .084727 [.074380, .096201] |
| Novel-end affected error: core minus full | .018927 [.006828, .033130] | .029784 [.015568, .045098] |
| Seeds positive for old / novel contrast | 5/6 / 5/6 | 6/6 / 6/6 |

The conditional exceptions are different seeds: only four of six have both contrasts positive. This is fully consistent with the declared separate consistency tests; it should not be described as complementary specialization in the same five seeds. Intervals are descriptive six-seed bootstrap intervals, with support replicates averaged within seed. The conditional old-harm interval also extends below the .005 allocation margin; the rule deliberately used a mean threshold, not an interval test.

These are permissive diagnostic gates. They require relative improvement over one component, not competent absolute prediction, superiority to joint training, a useful acquisition trajectory, preservation of the new dependency, or causal recognition of the environment. A recent-condition correction helping that condition while hurting another is compatible with ordinary interference. Passing this gate supplies a concrete output tradeoff to investigate, not evidence of a new learning principle.

## Absolute performance changes the interpretation

The recurrent frozen core's refreshed old-law error is **.247546**. The full predictor is worse at return entry, **.332273**, but improves to **.090948** after the return trajectory while the frozen core stays unchanged. The corresponding full-minus-core gap changes from **+.084727** to **-.156598**. Thus "turn the residual off for the old environment" is not a generally supported rule. The full model eventually becomes a much better old predictor than that core.

At the same return-end parameters, recurrent novel-law error with refreshed support is **.156058 full versus .107149 core**, a +.048908 gap in all six seeds. Yet the core's mean novel cue-flip benefit is only **.000220**, compared with .052115 for the full predictor before return and .019384 after return. Selecting the core can improve aggregate Brier while discarding evidence of using the newly relevant cue. It does not recover a preserved, fully functional novel skill. This is why any future success criterion needs explicit new-dependency behavior as well as error.

The conditional pattern is less severe. Its refreshed old core error is .085066; full error changes from .097939 at return entry to .076202 at return end. After return, the full predictor still improves novel affected error over core by .010852 on average, in five of six seeds. A permanent preference for the frozen core would also miss useful adaptation here.

Crucially, actual-history **separate-minus-joint return-entry** error is negative in every seed for both models: -.048635 conditional and -.029918 recurrent. The recurrent architecture's worse return AUC, +.033726 in the original experiment, therefore develops after entry. The newly measured entry interference cannot explain that entire trajectory disadvantage. This diagnostic does not retain intermediate component states and cannot isolate where or why it develops.

## What timing and routing do not explain

In the two conditional supply-timing seeds, novel-end refreshed affected error is .154223 for full separate versus .087762 for full joint. Correct timing-cue benefit is about .000051 for full separate and .000278 when using full content with core routing. These particular routing substitutions do not expose strong hidden timing-cue use. The four-cell decomposition also places most mean refreshed old-harm and novel-benefit differences in the content term rather than the routing-only term.

This is evidence about the examined **outputs**, not a proof that internal visual representations lack timing information. No representation probe, alternate readout fitting, or causal feature isolation was performed. Each cue subgroup has two seeds and cannot be promoted independently. The recurrent timing output, by contrast, has a substantial novel-end cue benefit, approximately .135683. There is no single established cause shared by the two architectures.

Residual-only performance is not a standalone-model qualification: its logits were trained to correct the core. Likewise, removing one component from the jointly trained arm does not uniquely identify where that model stores its knowledge.

## One bounded next question

My recommendation is to **draft one fixed-predictor access test today, without automatically starting another training experiment**:

> Can a single causal output-combination rule use only past performed-action feedback to improve prediction through unannounced old/novel switches, when the underlying learned functions remain fixed?

A fresh confirmation would fit the already specified joint and separate architectures on fresh seeds through the unchanged prefix, maintenance, and novel exposure, then freeze weights, optimizer, and replay. An evaluator would present a predeclared, counterbalanced sequence of old and novel conditions. Only causal recent history would advance. One fixed probability mixture of native separate core/full outputs would use lagged prediction errors recorded before each performed outcome was revealed, within the existing history budget. Its rule and every window/scale choice must be fixed before fitting; there should be no window, temperature, or architecture search.

Compare the same entire stream against always-full separate and frozen joint, including cold-start and switch-delay costs. Report old and novel intervals as well as aggregate Brier and greedy survival. Require meaningful absolute old/new performance and preserved novel-cue use; merely beating a poor full predictor is insufficient. No known boundary, law label, refreshed evaluator support, or target-selected component may enter prediction. Oracle selection, if reported, would be a diagnostic ceiling only. Initially identical observable histories cannot justify instant opposite decisions when the physical law changes.

This differs from prior consolidation and rehearsal-score work by holding the prediction functions fixed and isolating access to their outputs. No training update is accepted, rejected, redirected, or rolled back. It therefore gives a cleaner answer to one question. It is nevertheless adjacent to the earlier gating work, and success would remain an access result, not a solution to continual representation learning or the recurrent return-training deterioration. The original failed acquisition curves would remain failed.

The useful stopping condition is strict: one separately locked fresh test, with no tuning or subgroup rescue if it fails its complete screen. Even if it succeeds, moving back to continuously changing representations requires a separate rationale; success should not automatically launch another experiment. If the project wants to prioritize general representation learning immediately, changing mechanism family or broadening the benchmark is defensible now—the allocation result does not compel another gate.

## Candid assessment

This diagnostic was worthwhile: it rules out equating preserved parameters with preserved useful outputs and exposes where a seemingly attractive component switch would select an inaccurate or cue-insensitive predictor. The original source and prediction safeguards make those observations credible: 384 intervention rows reconstructed the original full forecasts exactly, 48 loaded states were preserved, and 96 frozen-core comparisons matched identical observable inputs. These checks validate the measurement, not the learning hypothesis.

There is still no demonstrated general continual-learning mechanism that beats the established joint/uniform-replay comparison on the combined objective. The risk of spinning wheels is real if each negative architecture is followed by another nearby gating variant without a hard budget and a decision that ends the line. One fixed-predictor access test is a defensible final bounded question for this branch. An open-ended sequence of residual, routing, or confidence modifications is not justified by these results.
