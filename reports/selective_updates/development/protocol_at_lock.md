# Selective updates: first-introduction development protocol

Status: fixed development experiment, specified on 2026-09-27 after the
retrospective [return-context diagnosis](../reports/return_context_v2/summary.md)
and before fitting any of this experiment's scientific seeds. This is a local
mechanism study, not an autonomous continual-learning or confirmation claim.

## Hypothesis

Retaining a full uniform reservoir preserves more historical coverage than
the previously failed recent/historical memory split. Increasing the weight
of current experience may accelerate acquisition; redirecting parameter motion
that conflicts with sampled historical prediction may limit its retention cost.
Their combination could improve the acquisition/retention tradeoff.

The matched diagnosis found worse current predictions than historical ones
even with identical target-law support, in every seed of both architectures.
Context refresh helps but does not fully recover those predictions. This
motivates testing interference control during learning. It does not identify
the correct rule or prove neuron-level erasure.

The user's proposed predictive-retention signal is relevant: reliable
prediction and marginal predictive usefulness are distinct. This experiment
tests a bounded interference proxy. It does not estimate the value of each
memory on new observations or identify true physical changes from noisy reports.
See [predictive retention](predictive_retention.md) for that separate hypothesis.

## Fixed interventions

All five learned arms retain the complete uniform 16-packet reservoir and use
immediate predictions. Each update samples exactly one historical causal
support/query packet, preserving its original support. Let L_C and L_R be
the unchanged performed-action BCE losses, averaged over three survival labels.

| Arm | Current loss weight | Historical loss weight | Parameter displacement |
|---|---:|---:|---|
| reference | .50 | .50 | Ordinary Adam |
| current | .75 | .25 | Ordinary Adam |
| protected | .50 | .50 | Projected Adam proposal |
| combined | .75 | .25 | Projected Adam proposal |
| shrink | .75 | .25 | Own locally norm-matched shrinkage |

The .75 coefficient is a single fixed development choice, not a sweep winner
or an exact reconstruction of the earlier 8/8 memory policy. It changes update
emphasis while preserving reservoir membership and sampling.

Compute h = grad L_R before the mixed update. Let d = theta_Adam - theta be
the actual displacement proposed after ordinary gradient clipping at norm 5
and the unchanged Adam step, including moments and preconditioning. Define

    d_projected = d - max(0, h dot d) / (h dot h) * h.

For zero h, keep d unchanged. Accumulate geometric dot products and norms in
float64. Do not add a positive denominator epsilon and then claim an exact
halfspace projection. Account for rounding when applying changes to float32
parameters. Keep Adam's moments and counters from the ordinary mixed update;
change only the applied parameters. Apply the rule to all trainable parameters,
including features, hypotheses and history processing. No latent law labels,
counterfactual outcomes or future feedback may enter learning.

The shrink control computes its own hypothetical projected displacement at
its own current state, then applies d times ||d_projected|| / ||d||. Zero d
remains zero. Its local candidate norms match; norms across the diverging
projected and shrink trajectories need not match. No arm consumes another
arm's future trace. Leave unprojected parameters untouched after Adam so that
the .50 reference reproduces the previous learner's arithmetic exactly.

The constraint addresses first-order BCE on one reported replay packet. It
does not guarantee finite-step loss reduction, protection of all knowledge,
or improved physical Brier error. Old or corrupted targets can be protected.
This is closely related to [A-GEM](https://arxiv.org/abs/1812.00420), with a
declared modification acting on the actual adaptive-optimizer displacement.
No algorithmic novelty is claimed.

## Cohort and matched resources

Use six unused seeds 13001--13006 and both qualified architectures. Each seed
first learns the same four alternating base-law blocks of 1,024 arrivals with
the ordinary uniform-replay learner. Fork identical learned weights, optimizer,
reservoir, RNG and recent history into all five interventions. Each receives
8,192 arrivals at the first new dependency, in packets of 32 with 12 optimizer
updates per packet. Preserve width 64, four conditional heads, qualified
recurrent interaction features, Adam learning rate .002 and all other fixed
settings from the prior studies.

A sixth, fresh reference starts from the original random initialization with
empty optimizer/replay and the same observed prefix history. It uses the .50
reference rule. There are 12 prefix fits and 72 challenge episodes. The first
cue is supply timing for seeds 13001/13006, directional efficiency for
13002/13003, and arrival delay for 13004/13005. Each cue occurs once in each
base mode. This tests three first cues twice each, not six observed introduction
orders. All seeds and arms complete before results are interpreted.

The learner sees no change signal. The experimental fork occurs at a known
introduction to isolate the intervention. This is not evidence of successful
lifelong operation from initialization, automatic boundary discovery, or late
plasticity. No additional first-introduction seeds or coefficient choices are
added in response to results.

Every arm computes the extra replay gradient and hypothetical projection.
At the last update of each packet, every arm also computes one extra replay
forward to measure actual before/after replay loss. Keep original training
presentation counters and record extra backward/diagnostic forward work
separately. Log bounded counters and scalar summaries for conflicts, geometric
residuals, motion norms, replay ages, and actual replay loss changes. Count
temporary parameter/gradient buffers and report wall time. Matched experimental
work does not mean the original deployable baseline needs this discarded work.
Use four single-thread CPU workers with affinity mask 85, and deterministic
PyTorch operations. Save checkpoints for exact resumption.

## Outcomes, analysis and fixed continuation rule

Reuse the existing evaluator and 512 fixed query cases. Evaluate every 256
arrivals. Primary acquisition is affected-subset Brier integrated across the
episode; lower is better. Terminal valid-old Brier is measured separately in
both base modes with the existing correct-support probes. Report whole-query
Brier, actual-history and refreshed-history endpoints, survival, cue use,
encoder displacement and work counts as supporting outcomes.

Compute paired seed differences and 20,000 seed bootstrap resamples with seed
27192026 separately for each architecture. Each seed contributes one episode
per arm. Queries, horizons and supports are not independent replications.
Show every seed, cue group, mean and descriptive 95% interval. Report current
emphasis and protection simple effects, factorial main effects and interaction,
and combined-minus-shrink. No interaction significance requirement is imposed.

The sole primary candidate is `combined`. Assess continuation separately per
architecture only when all planned work is complete:

1. Fresh references pass the unchanged marginal-predictor gain >= .02 and
   correct-cue benefit >= .002 in every available stage and cue group.
2. Combined acquisition Brier AUC minus reference is <= -.002 on average,
   with improvement in at least five of six paired seeds.
3. Combined terminal valid-old Brier minus reference is <= +.005 in each
   old mode separately; mean survival AUC loss is at most .01.

These are practical development allocation thresholds, not superiority or
noninferiority tests. Two seeds per cue provide a limited qualification check.
Report intervals crossing thresholds. Failure does not trigger retuning or
silent promotion of an ablation arm. A simpler arm that looks promising is
recorded as a separate development finding.

Assess attribution to directional protection separately: against both `current`
and `shrink`, combined must reduce mean terminal valid-old Brier by at least
.002 while sacrificing at most .001 acquisition Brier AUC. Report per-mode
retention and survival as well. Passing the outcome screen without this
attribution screen does not establish selective protection as the explanation.

If either architecture passes the outcome screen, it may justify a separately
locked independent cohort of sequential introductions, returns, revisions and
matched clean/noisy branches. Complete that protocol before further fitting.
Failure stops expansion of this fixed combined recipe. No result here establishes
compounding advantage, a longer physical planning horizon or generality beyond
this constructed environment.

## Locks, audit and engineering checks

Before scientific fitting, lock config, scientific source, runtime, this
protocol, analysis scripts and relevant tests; save source ZIPs. A small smoke
cohort with disjoint seeds tests engineering only and cannot select parameters.
Verify reference equivalence, projection geometry, independent copies,
optimizer state, bounded memory, lock rejection and checkpoint resumption.

The completed local audit regenerates input packets and causal replay, verifies
exact forks and budgets, matches results to checkpoint records, and recomputes
start/end probes. It does not claim independent retraining of every update.
Retain every attempt and failed check, preserve the 338 older artifacts and
the return-context diagnostic locks, and publish complete local records with
checksums. No old scientific source or sealed archive is to be modified.
