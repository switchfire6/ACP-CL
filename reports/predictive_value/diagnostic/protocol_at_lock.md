# Predictive rehearsal value: prospective score validation

Status: fixed diagnostic design, 2026-09-27, before fitting scientific seeds.
This tests a value estimator, not a deployed memory-selection policy. Earlier
failed experiments and their sealed records remain unchanged.

## Question and estimand

Does the predictive benefit of rehearsing a stored causal packet forecast its
benefit on later observations, including when an earlier environment returns?
The proposed unit of value is an additional rehearsal relative to a specified
replacement. It is not the value of erasing every influence already learned
from a memory, nor an intrinsic property of an example.

For a parent learner at time s, let R(theta_s, a_s, j) apply 12 ordinary Adam
updates to equal-weight BCE on the latest observed causal packet a_s and stored
packet j. Copy the actual weights, optimizer moments and step count, and retain
the original gradient clipping at norm 5. Each packet preserves its original
past-only support. No memory sampling, insertion or history advancement occurs
inside this diagnostic rehearsal. The ordinary learner continues unchanged.

Choose K=8 candidates uniformly without replacement from its 16-packet
reservoir, excluding the latest anchor's ID. Choose a common replacement r
uniformly from the remaining pool. Use a separate deterministic diagnostic RNG;
never consume the ordinary reservoir generators. Construct nine exact forks.

On the next 512 arrivals W_score, freeze their parameters and predict performed
action outcomes before observing each packet's feedback. All predictors use
the same actual preceding 32-record history. Score three finite survival
forecasts with mean Brier loss. Define

    v_hat(j) = mean_W_score[L(R(theta_s,a_s,r)) - L(R(theta_s,a_s,j))].

The proposed selection is the largest v_hat, with smallest packet ID breaking
exact ties. The distinct simple-accuracy control chooses the smallest original
prequential Brier error, recorded before training on that packet at its first
arrival. That measure also reflects learner maturity; preserve packet age,
original optimizer step and prediction/support fingerprints for interpretation.

With a common replacement, value ranking is algebraically identical to ranking
the eight candidate predictors' score-window accuracy. Those are not two
independent controls. The accuracy control above instead uses the packet's
original pre-update error. Uniform choice is evaluated exactly as the mean of
the eight candidate losses and survivals, not an ensemble prediction and not
a single extra random draw.

## Causal timeline and validation

The ordinary learner trains exactly once on each arriving packet, with its
unchanged uniform replay rule. Scoring and diagnostic choices never change its
parameters, optimizer, memory contents, random generators, history or updates.
Diagnostic packets are temporarily retained even if ordinary replay evicts them;
count this extra storage explicitly.

1. Select candidates/replacement and build score forks using only past data.
2. Predict and score W_score while the ordinary learner independently learns
   those arrivals. Seal the candidate choices before any validation outcome.
3. From the ordinary learner's state at the end of W_score, create fresh forks
   with the same candidate/replacement packets and the latest observed anchor.
   Freeze those weights and assess the next 512 current-condition arrivals.
4. Continue ordinary learning for a specified gap in the current condition.
   Before any return feedback, build fresh forks with the same retained packets
   and latest observed anchor. Assess the first 512 arrivals of the old condition.

Both primary validation panels therefore test fresh application of the same
rehearsal recipe. Selection stays fixed from W_score. The original score forks
also predict both later panels as a secondary diagnostic. Their comparison
with reapplied forks involves changed parent weights, optimizer state and
anchor; it is not a pure parameter-age effect. Every predictor gets identical
causally evolving context within a panel. No fresh oracle context is supplied.

Score and validation query outcomes are disjoint. A later window may legitimately
use the preceding window's last packet as causal support. Adjacent stored packets
can also share a query/support observation; record such overlaps and never
treat candidates, packets or horizons as independent replicates.

## Fixed cohort and schedule

Use fresh seeds 14001--14012 and the existing conditional and qualified recurrent
architectures, width 64, four conditional heads, recurrent interaction features,
Adam lr .002, 12 updates per packet of 32, uniform capacity 16. Learn four
alternating base-law blocks of 1,024 arrivals, then 8,192 arrivals under the first
novel dependency. The dependency and modes follow the unchanged acquisition
world's seed-order convention. No hidden law metadata enters learning or selection.

Each trajectory has two assessments. One uses 512 intervening current-condition
arrivals, the other 4,096; order is reversed when (seed // 6) % 2 is odd.
After the first return window, learn 1,024 current-condition arrivals before
the second assessment. Both return windows use the original base mode with
no novel dependency. This is 20,992 ordinary arrivals per trajectory, 24
trajectories and 48 assessments. Cue, initial mode and gap order are balanced
across seeds. Both gaps occur within every trajectory.

The gap comparison includes additional learning and different arrival positions;
it is not an isolated effect of elapsed time. Candidate coverage of the returning
law is an evaluator-only diagnostic. Include assessments without such candidates
in the primary results. Their absence is a coverage problem, not by itself
evidence of incorrect ranking. Do not select candidates by true source law.

Feedback is clean in this first score-validation experiment. It cannot establish
robustness to noisy reports, altered physical laws, adaptive actions or a larger
physical planning horizon. Survival labels still concern H=4,8,12. This is a
known-schedule diagnostic, not automatic transition discovery or lifelong policy
improvement. Models and representations begin from random initialization.

Use six single-thread CPU workers with affinity mask 1365 on this host, increased
from the previous four workers/mask 85. Record actual affinity and runtime.
A tiny disjoint smoke cohort checks engineering; it cannot choose parameters or
pass scientific continuation criteria. Lock all science/analysis/protocol/config
before the main cohort, and finish all trajectories before inspecting outcomes.
The independent portable archive verifier and its tests are packaging checks,
excluded from this scientific lock and sealed separately when complete. They
cannot change the locked metrics, analysis or continuation criteria.

## Outcomes and fixed allocation decisions

Primary error is observed-action Brier averaged over the complete 512-arrival
near or return panel, including initial context adaptation. Evaluator-only
counterfactual physics additionally measures survival from each predictor's
greedy final-horizon action; none of this truth is passed to the learner or
score selection. Report score-window values, frozen-fork panels, per-gap/cue
results, initial-versus-later return packets, and oracle best-candidate spread
as descriptive diagnostics. The oracle uses validation outcomes and is not
deployable. Small candidate spread limits what a null result can distinguish.

Average the two assessments within each seed first. Compute paired differences
and 20,000 seed-bootstrap resamples (seed 27192026) separately by architecture.
Show all 12 independent seed values and descriptive 95% intervals. Primary
contrasts are value-selected minus exact uniform and value-selected minus
original-accuracy selection. Negative Brier and positive survival are better.

Assess local and return usefulness as separate tiers, using the same rule:

- Mean reapplied Brier difference versus uniform is <= -.0005.
- Improvement occurs in at least 9 of 12 independent seed means.
- Mean reapplied Brier versus original-accuracy selection is <= 0.
- Mean survival difference versus uniform is >= -.005.

These practical mean-based allocation criteria are not superiority or
noninferiority tests. Intervals do not create extra post-hoc gates. A tie with
the accuracy control does not establish additional benefit over that cheaper
signal. Report that comparison explicitly.

Passing only the local tier supports, at most, a separately specified local
rehearsal-policy experiment. Require both tiers before expanding this fixed
score into a recurrence-oriented policy. Failure of both stops expansion of
this score recipe without tuning its dose, pool size or windows on these seeds.
A local pass with return failure identifies a temporal-generalization limit;
it does not validate retention or justify discarding dormant knowledge.

## Resource accounting and audit

Every selector is evaluated using the same candidate forks and scoring work.
For an efficient deployed selector the uniform and original-error controls
would not need all this discarded work; equality in the diagnostic does not
prove equal deployment cost. Count base training, all 54 diagnostic rehearsals
per trajectory (648 extra Adam steps), prediction calls/query/support
presentations, candidate storage, copied model/optimizer/replay states and time.
Do not present explicit tensor-buffer estimates as measured peak RAM.

Save atomic phase checkpoints and complete record hashes for exact crash
resumption. Verify copied optimizer ownership/steps, parent nonmutation,
original prequential score provenance, selection before validation, deterministic
candidate sampling, exact causal history and unchanged ordinary replay streams.
The local auditor regenerates data and independently recreates the small
rehearsal interventions and predictions. Portable analysis recomputes selected
and uniform metrics, paired uncertainty and the fixed decisions. Neither is a
new-seed scientific replication.

Preserve every failure/amendment and all earlier sealed evidence. Publish full
seed outcomes, source/protocol snapshots, checksums and standalone figures.
Related precedents and limits are documented in
[the focused literature review](predictive_value_related_work.md).
