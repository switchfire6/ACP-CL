# Rehearsal value: weights, Adam state, anchor and environmental conditions

Prospective diagnostic specified 2026-09-27 local time, after the completed
predictive-value study and before fitting seeds 15001--15012. Earlier results,
criteria and source snapshots remain unchanged. No adaptive retention policy
is being tested or promoted.

## Question

Does the lost usefulness of a previously selected rehearsal packet recur on
fresh seeds, and which controlled state substitutions alter that usefulness?
The previous study changed weights, Adam state and the accompanying anchor
together. It also combined environmental returns with extra learning and time.
This study varies the three learning-state components independently and tests
two environmental branches from the same cutoff.

Let the old ordinary learner state occur at t0, immediately before a 512-arrival
score window, and the new state at t1, after learning those 512 arrivals.
Choose eight candidates and one common replacement from the 16-packet uniform
reservoir at t0, excluding the latest anchor. Keep the previous candidate
draw, tie-breaking and rehearsal recipe: 12 Adam updates on equal-weight BCE
for a candidate and an anchor, with norm-5 clipping. Preserve complete causal
packets (original support and query), not just their query observations.

For each candidate, the original all-old rehearsal predictor is frozen through
the score window. Rank candidates by its performed-action Brier on that window;
seal selection before any validation observations. The ordinary learner trains
once per arriving packet using its unchanged uniform replay. Its representations
start from random initialization. Original pre-update packet error and exact
uniform candidate-loss expectation remain controls.

## Eight controlled state combinations

Cell bits are W/O/A, with 0 selecting the old donor and 1 the new donor:

- W: complete model parameters and buffers, parameter gradients, trainability
  and module modes. Gradients are then cleared by the unchanged rehearsal loop.
- O: complete Adam moments, per-parameter step counters and parameter-group
  settings, copied with independent storage onto matching destination parameters.
- A: the full old or new latest anchor packet, with its original causal support.

All inactive history, replay contents/generators, bookkeeping and settings
inside hybrid learners use the same old donor. Rehearsal never samples memory,
advances history or changes the ordinary learner. Evaluation history is supplied
externally and identically to every cell within a branch.

Cell 000 reuses the original score forks and must match exact old rehearsal.
Cell 111 must match the active learning state and predictions of ordinary fresh
rehearsal from the new donor. Full inactive-state equality at 111 is neither
expected nor required because its copied bookkeeping deliberately stays old.
The other six cells are intentional counterfactual state combinations. Their
moments may be poorly aligned with their weights. Effects concern these defined
substitutions, not natural causal contributions along an ordinary trajectory.

Add zero-update old-weight and new-weight predictors outside the candidate
pool. These controls distinguish changes in absolute prediction quality from
changes in selected-versus-uniform ranking. Before rehearsal, optimizer and
anchor choices cannot affect predictions with fixed weights and context.

Record finite-state checks, complete provenance, optimizer steps, parameter
displacement, and pre/post anchor and replay BCE for the seven newly constructed
cell families. Those four instrumentation forwards per fork are counted
separately. The reused 000 forks retain their original work records. No extreme
or unfavorable hybrid is silently excluded. A numerical failure must remain
visible and prevent a complete scientific result.

## Matched environments and timeline

At t1, freeze all eight families and both zero-update references. Evaluate the
same next 512 observation/action draws under two laws: continued current law,
and the exact original base law. Couple the exogenous draw and performed actions
across branches; assert identical rendered observations and actions. Changing
the law changes physical consequences. All forecasts concern the unchanged
three horizons H=4,8,12.

Both branches begin with the identical actual history preceding t1. Thereafter
each branch uses only its own preceding feedback, shared by all its predictors.
The return branch receives no privileged target-law context. The branch contrast
therefore includes the environmental change and its subsequent causal feedback
and context adaptation. It does not isolate a direct law effect with context
held fixed, or reproduce an endogenous action policy.

Only the current branch is learned by the ordinary trajectory. The simulated
return branch supplies diagnostic feedback solely to its external context.
Selection and all intervention training precede validation. The simulator may
evaluate frozen forecasts separately from ordinary training, but its explicit
prediction inputs remain restricted to the query and preceding branch history.
Truth for unperformed actions is evaluator-only and scores greedy survival.

Use seeds 15001--15012 with both conditional and recurrent architectures,
width 64, four conditional heads, recurrent interaction features, Adam lr .002,
12 ordinary updates per packet of 32, and uniform capacity 16. Learn four
alternating base-law blocks of 1,024 arrivals and 8,192 current-law arrivals.
Perform score0 (512) and validation0/current learning (512), learn another
4,096 current-law arrivals, then score1 (512) and validation1/current learning
(512). Each ordinary trajectory learns 18,432 arrivals. Initial mode and novel
cue are balanced across the 12 fresh seeds. Two assessments are dependent
repeated measurements within a seed, not extra independent replicates.

The full cohort has 24 trajectories, 48 assessments, 3,456 rehearsal forks
(including the reused score forks), 96 zero-update copies, 120,576 diagnostic
prediction forwards and 12,096 extra loss-instrumentation forwards. There are
41,472 additional Adam updates versus 165,888 ordinary updates. Count temporary
candidate storage, copied state and all prediction work; explicit tensor sizes
are not measured peak memory. Use six single-thread workers and host affinity
1365. A disjoint tiny smoke cohort cannot choose scientific parameters.

## Estimands and fixed allocation decision

For branch b and cell c, define

    D_b,c = L_b,c(selected by the old score window)
            - mean_j L_b,c(candidate j).
    T = D_current,111 - D_current,000.

Loss is performed-action Brier averaged over the complete validation window.
Positive T means the selected packet's relative usefulness deteriorates after
all three substitutions. It does not necessarily mean that the selected model's
absolute predictions worsen; uniform candidates may improve more.

The sole primary replication screen, separately per architecture, requires
mean T >= .0005 and T > 0 in at least 9/12 independent seed means. Average the
two assessments within each seed first. Report all 12 values and descriptive
95% bootstrap intervals using 20,000 resamples and seed 27192026. These practical
allocation thresholds are not significance tests. Also report whether D000
itself carries useful negative selection value; never filter on that result.

If this endpoint effect does not replicate, component findings remain
descriptive and do not support a general attribution of the earlier failure.
If it replicates, the factorial can prioritize a separately specified causal
follow-up. No largest-factor rule or automatic policy promotion is permitted.
An artificial old-state hybrid is not a deployable continuation of learning.

Report all eight D values and absolute selected, original-error, uniform,
replacement and validation-oracle losses/survival. The oracle uses validation
answers and is optimistic, not deployable. For D and absolute selected outcomes,
report each factor's four conditional swap effects, their average, the three
pair interactions averaged over the third factor, and the three-way interaction.
Include all-new single-factor reversions:

    weights:   D111 - D011
    optimizer: D111 - D101
    anchor:    D111 - D110.

Report their absolute selected-error and survival changes too. A relative
improvement obtained by worsening the uniform comparator is not a beneficial
learning repair. Interactions and sign reversals prevent assigning a unique
intrinsic share of blame to one component. No additional repair gate is used.

The paired return-minus-current contrast for every cell is a declared secondary
environmental stress test. Report initial-packet and later-packet errors,
candidate rank agreement, origin coverage, original error/maturity, and the
zero-update baselines descriptively. Include every candidate and assessment;
latent origins never choose rehearsal or modify scoring. Do not tune the score,
dose, candidate pool, windows or component choices on these outcomes.

## Audit and preservation

Lock source, config, runtime, protocol and scientific analysis before main
fitting, and finish all trajectories before examining scientific comparisons.
An independent portable archive verifier and its tests are packaging checks,
explicitly excluded from the scientific analysis lock and sealed separately.
They cannot modify the scientific metrics or allocation criteria.
Use atomic resumable checkpoints, preserve gradients explicitly, and retain
all raw all-action probabilities and branch outcomes in numerical arrays.
Verify original score choice, exact candidate provenance, unchanged ordinary
replay/history streams, tensor ownership, endpoint equivalence, paired branch
inputs and causal feedback, raw metrics, and every factorial contrast.

The independent local auditor reconstructs each short rehearsal intervention
and diagnostic forecast. It does not independently retrain every ordinary
learning update. Portable arithmetic checks must state their more limited scope.
Noisy feedback, obsolete relationships, automatic transition detection, an
adaptive retention policy, longer planning and lifelong compounding remain
outside this diagnostic. Previous scientific archives and decisions are immutable.
