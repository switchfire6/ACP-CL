# Proposed next decision: causal access to fixed predictors

Draft after the saved-output diagnostic, 2026-09-28 UTC. **Not a locked protocol
and not run.** The original frozen-core recipe still fails its primary screens.
The diagnostic's GO designation permits specifying one discriminating test;
it does not require another training sweep or promise a positive result.

## Question

Can a fixed rule, using only past performed-action feedback, exploit differences
between the preserved core and the combined predictor across environmental
switches? Isolating this question avoids changing output selection and model
learning simultaneously. Passing it would demonstrate access to existing
functions, not continual acquisition of new representations.

## One candidate, no selection sweep

After the same prefix and novel-learning schedule, freeze both pathways, Adam,
and replay. Keep the ordinary causal observation/action/outcome history active
for inference. Use native projected core probabilities p_C and native combined
probabilities p_F, including their own existing context/routing computations.

For the performed action, store each pathway's terminal-horizon probability
when it originally predicted the outcome. After that outcome becomes available,
record the Bernoulli log score, clipping probabilities only for scoring at
1e-6 and 1−1e-6. Before the next packet, form

\[
d_t=\sum_{i\in W_t}\bigl(\log p_F(y_i\mid x_i,a_i,h_i)
                         -\log p_C(y_i\mid x_i,a_i,h_i)\bigr),\qquad
w_t=\operatorname{sigmoid}(d_t),
\]
\[
p_t=(1-w_t)p_C(x_t,h_t)+w_t p_F(x_t,h_t).
\]

W_t contains at most the last 32 feedback records, all preceding the current
prediction. Stored forecasts use the history actually available when they were
made; do not recompute them using later outcomes. The initial weight is 1/2.
No learned gate, temperature fit, window sweep, target-law identifier, announced
boundary, or boundary-triggered reset. This is an ordinary evidence-based
probability mixer, not a claimed novel algorithm. Terminal outcomes alone enter
its score because the three horizon labels are correlated. Mixtures of the
already monotone survival curves remain monotone.

## Required controls and scope

Use fresh learner seeds with the original balanced cue/mode schedule and a
separately specified, unlabeled switching stream. Freeze all predictors for that
stream; evaluate always-core, always-full, fixed-half mixture, the one causal
mixer, and the matched joint predictor. Give all controls the same observable
history, score bookkeeping, and component-forward budget. Include every
transition and recognition delay in whole-stream error and greedy survival.
Test each architecture separately and report every cue family.

Before launching, lock the seed count, training/update budgets, switching
schedule, primary margins, survival/novel-cue guards, and a fixed compute limit.
Fresh training is required for an independent model-cohort test; using only new
query streams with the existing models would be a narrower diagnostic. No fit
or stream has been started for this proposal.

## Qualification is essential

The current diagnostic is insufficient to establish usable building blocks.
The recurrent core's old-condition error is .247546 despite correct support;
the conditional combined predictor barely uses the supply-timing cue. Require
adequate absolute old/new predictions against independently trained single-law
references, as well as dependence on the relevant novel cue. Define those
requirements before fitting or seeing new scores. Beating a poor full predictor
or replacing lost cue knowledge with an insensitive core must not pass.

This availability check may reject the premise before the mixer is scientifically
interesting. Report that result rather than training longer, excluding a cue,
adding snapshots, or changing the selection rule to rescue it.

## Why this could still be spinning wheels

Earlier evidence-delay and consolidation rules failed; this proposal is adjacent
to that work. Its limited distinction is that only selection between two fixed
predictive functions changes. It does not accept/reject training updates,
rehearse selected memories, or change the training trajectory. That distinction
supports one clean test, not an open-ended search over gates.

If the necessary component competence or the predeclared online improvement
fails, stop this frozen-core/additive-residual branch. If both pass, a separate
continuous-training test would still be required, followed by another task.
Neither outcome establishes long-term compounding, autonomous capacity renewal,
the necessity of relational units, or a brain-development analogy.
