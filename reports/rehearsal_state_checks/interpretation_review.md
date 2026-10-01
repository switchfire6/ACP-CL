# Interpretation review before scientific outcome access

Prepared 2026-09-28 01:01:35 UTC while the locked cohort was running. This
review uses the [fixed protocol](../../docs/rehearsal_state_protocol.md),
[trajectory design](../../src/acp_cl/rehearsal_state/design.py),
[previous completed report](../predictive_value_results.md), and
[research charter](../../docs/research_charter.md). No partial main outcomes
were inspected. These interpretation notes add no metric, exclusion, threshold,
repair gate or policy-selection rule to the locked analysis.

The charter's objective is a general learner that acquires representations
from scratch, preserves useful knowledge and keeps learning under bounded
resources. This experiment addresses a prerequisite: whether the benefit of
rehearsing an experience depends on the learner state and accompanying update.
Random initialization satisfies the starting condition; it does not establish
sustained representation acquisition. The ordinary uniform-replay learner is
unchanged, so a successful diagnosis is not an improved continual-learning
mechanism. Earlier failed policies remain failed.

For a branch and cell, D is selected-candidate Brier minus the exact mean
candidate Brier. The primary current-branch contrast is T = D111 - D000.
Positive T means that selection's relative usefulness deteriorated. It can
arise because selected predictions worsen, because uniform candidates improve
more, or both. Report absolute selected, uniform, replacement and weight-matched
zero-update errors and survival alongside D. A validation-selected oracle is
an optimistic description of observed variation, not achievable policy evidence.

| Possible result, separately per architecture | Interpretation and permitted next step |
|---|---|
| Mean T is below .0005, or fewer than 9/12 seed means are positive. | The fixed replication screen fails. A positive point estimate, a favorable interval, or an interesting subgroup cannot substitute for it. Component findings remain descriptive; do not generalize a cause of the earlier failure from this cohort. |
| The screen passes and D000 is usefully negative. | The earlier loss of relative rehearsal usefulness recurs under the declared substitutions. The full factorial can inform a separately specified causal follow-up. It does not select a policy or confer a unique component attribution. |
| The screen passes but D000 is near zero or positive. | Describe further relative deterioration, rather than loss of a reliably useful selector. Keep every assessment; D000 is a contextual measurement, not an additional eligibility filter. |
| A reversion improves D but makes the selected predictor worse in absolute terms. | This can result from worsening the uniform comparator even more. It is not a beneficial learning repair. Report the adverse absolute result explicitly. |
| A reversion improves absolute selected Brier but reduces survival. | Report the predictive/control tradeoff. Brier and greedy survival answer different questions; no new repair acceptance threshold is introduced. |
| One architecture passes and the other fails. | Keep the architecture-specific decisions. Do not pool them to create a pass or describe a general cross-architecture mechanism. |

Every repair contrast is cell 111 minus the cell reverting one component.
Positive repair Brier therefore means reversion lowers error. Positive repair
survival means reversion loses survival. These signs must not be described as
improvements in the same direction.

The weight factor substitutes the complete model state. A weight effect does
not locate forgotten information in an encoder, establish feature erasure, or
prove reduced plasticity. The Adam factor substitutes old versus new complete
optimizer state, including step and group settings; it is neither an optimizer
reset nor a moment-only intervention. An anchor effect concerns both the
supervised query packet and its original causal support, not query content
alone. The zero-update references help separate baseline predictive differences
from effects introduced by rehearsal.

Mixed weight/Adam states are intentional counterfactuals and can be poorly
aligned. Large effects or unusually large parameter displacement may reveal
sensitivity to such substitutions without identifying a failure that naturally
occurs on the ordinary training path. Check the recorded displacement and
anchor/replay BCE diagnostics; keep unfavorable finite hybrids. Review all
conditional effects and interactions. Sign reversals or strong interactions
preclude treating the largest averaged main effect as a unique cause. Exact
000 and active-state 111 equivalence are engineering prerequisites, not outcome
evidence or additional passing criteria.

The two environmental branches use identical query observations/actions and
start with the same actual history. Later histories contain each branch's own
feedback. Thus a branch difference concerns the changed law together with its
causal feedback and context adaptation. If rankings change across environments
even within a fixed cell, a change in learned parameters alone is insufficient
to describe that observed instability; sampling variation and branch context
still matter. Stable rankings do not imply large benefit, good calibration,
safe actions or retained knowledge. Conversely, poor rank agreement can coexist
with small loss differences when candidates are nearly interchangeable.

These are simultaneous environmental branches from a fixed cutoff. They do
not reproduce the earlier study's delayed returns after intervening learning,
isolate elapsed time, or demonstrate a longer physical planning horizon.
Only current-law arrivals train the ordinary learner. Absence of exact
return-origin candidates is a coverage fact, not grounds for exclusion or a
complete explanation of failure; other packets can contribute shared structure.

Two assessments are repeated measurements within each of twelve seeds, not
24 independent observations per architecture. Their different lifetime positions
do not constitute a randomized test of age. All seeds, candidates and finite
hybrids remain included. Intervals are descriptive across many related effects;
no interval-driven repair gate or subgroup promotion is allowed.

Even a clear component diagnosis would still need an implementable prospective
intervention, a fresh cohort testing new-dependency acquisition together with
valid-old retention and recurrence, and subsequent revision/noise and resource
controls. Generality further requires another environment. The present
diagnostic does not establish special status for relational representations,
joint persistence, critical periods, or a permanent importance score. Its
possible contribution is a controlled constraint on the next learning mechanism.
