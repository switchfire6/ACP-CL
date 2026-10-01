# Causal-access outcome review

Reviewed 2026-09-28 UTC against the locked protocol and the complete fresh
development cohort. **Stop the frozen-core/additive-residual branch under the
agreed stopping rule.** Both architectures fail qualification and also fail
the raw primary screen. The decision does not depend solely on a reference
qualification requirement.

The completed cohort contains six seeds per architecture, 120 fitting phases,
466,944 training arrivals and 196,608 frozen-stream arrivals. All 16 switching
blocks, first packets and subsequent recognition delays enter the stream scores.
No seed, cue family or architecture is excluded below.

## Qualification failures

All old and novel specialists meet the all-case Brier ceiling and marginal-gain
requirements, overall and within every cue family. All separate full predictors
meet the novel affected-case error tolerance relative to their specialist.
The following requirements fail:

| Architecture | Failed requirement | Observed | Required |
|---|---|---:|---:|
| Conditional | Core-old excess Brier, overall | +.032531 | <= .020000 |
| Conditional | Core-old excess, arrival-delay group | +.074602 | <= .020000 |
| Conditional | Novel specialist delay-cue benefit | .001385 | >= .002000 |
| Recurrent | Core-old excess Brier, overall | +.081204 | <= .020000 |
| Recurrent | Core-old excess, efficiency / delay / timing | +.106895 / +.079310 / +.057405 | Each <= .020000 |
| Recurrent | Separate full delay-cue benefit | .001940 | >= .002000 |

Under identical privileged old-law query/support conditions, the conditional
core scores .101191 versus its specialist's .068660; the recurrent core scores
.152426 versus .071223. Preserving these weights did not supply an adequately
competent reusable old function under this fitting schedule. This is not proof
that freezing, modularity, or context-dependent access is inherently ineffective.
The specialist has its declared independent single-law exposure; it is a
competence reference, not a compute-matched forgetting control.

The conditional overall core deficit is influenced strongly by seed 17006:
its old core scores .212379 versus specialist .063588. The protocol deliberately
does not permit removing this seed. Cue-group qualifications have only two
independent seeds each and do not support sweeping claims about cue difficulty.

## Online candidate and fixed-half control

Differences below are causal minus comparator in raw Brier units. Negative is
better. Parentheses count seeds with strictly lower error, not significant wins.

| Contrast | Conditional | Recurrent | Gate |
|---|---:|---:|---|
| Whole, versus always full | -.004563 (5/6) | -.039696 (6/6) | Both pass |
| Whole, versus fixed half | +.000787 (3/6) | -.010461 (5/6) | Conditional fails; recurrent passes |
| Whole, versus joint | -.035006 (6/6) | -.082046 (6/6) | Both pass |
| Old, versus always full | -.011184 (5/6) | -.084218 (6/6) | Both pass |
| Novel affected cases, versus always full | +.002581 (1/6) | +.005512 (0/6) | Conditional passes .005 tolerance; recurrent fails |

Conditional whole-stream Brier is .095855 for full, .090506 for fixed half,
.091293 for causal and .126299 for joint. A constant mixture accounts for the
aggregate advantage over always-full in this architecture; feedback does not
establish the required added value. Its causal-minus-half interval spans zero,
[-.004118, +.006833], so this is a failed useful-improvement screen, not proof
that fixed-half is universally superior.

Recurrent whole-stream Brier is .172644 for full, .143409 for fixed half,
.132948 for causal and .214994 for joint. Here past feedback provides useful
access beyond constant averaging: causal-minus-half is -.010461 with descriptive
95% interval [-.015978, -.004963]. Nevertheless, the benefit principally reduces
old-condition error of poor available functions. Novel affected-case error rises
in all six seeds, by .005512 on average, with interval [.004186, .006841].
The .005 guard is missed narrowly; the much larger old-function deficits and
cue failures independently prevent promotion.

## Survival and cue guards

All predeclared survival mean guards pass. The following are absolute survival
probability differences; positive is better:

| Causal minus comparator | Conditional | Recurrent |
|---|---:|---:|
| Old versus full | +.007711 | +.050008 |
| Old versus joint | +.037292 | +.076029 |
| Novel versus full | -.002441 | -.004659 |
| Novel versus joint | -.001180 | -.009867 |

Recurrent novel survival versus joint is lower in all six seeds and only just
inside the allowed .01 mean loss. Its descriptive interval is [-.018311,
-.004354], extending beyond the allowed loss. Passing this mean guard therefore
does not establish statistical noninferiority. Whole-stream survival gains are
+.002635 / +.022675 versus full and +.018056 / +.033081 versus joint for
conditional / recurrent respectively; these aggregate gains do not erase the
novel-condition cost.

Both architectures fail the online arrival-delay cue guard: conditional causal
benefit is .000948 and recurrent .001379, below .002. Their overall, efficiency
and timing cue means pass. Always-full online delay-cue benefits are also below
.002 (.001439 and .001650 respectively). Weak cue use therefore cannot be
attributed solely to the mixer. The conditional specialist's weak delay-cue
qualification also limits a clean interpretation of this dependency.

## What this says about useful retention

The user's preference for useful rather than perfect retention is compatible
with this protocol. It explicitly permits .005 Brier losses on protected
slices, .01 absolute survival losses, and .02 component error above a specialist.
The conditional novel-error guard passes despite an observed deterioration.
The rejection is not the consequence of requiring zero forgetting.

This experiment froze predictors during the switching stream. It cannot show
whether allowing some parameter drift would improve future representation
learning, or locate an optimal stability/plasticity tradeoff. It does show that
causal feedback can help access existing recurrent functions, while also showing
that access alone does not satisfy the stipulated usable-learning outcome.

Record that narrow positive finding and stop this branch. Do not extend the
evidence window, add exposure, select fixed-half after seeing the result, or
promote a cue subgroup. Continuing the broader Adam project would require a
different declared question and a demonstrably competent baseline, rather than
another adjustment intended to rescue this failed screen. Neither the negative
decision nor the narrow access benefit settles general continual learning,
recursive self-improvement, or eventual compounding gains.

## Read-only arithmetic check

An additional reduction written separately from the locked scorer recomputed
816 per-seed stream/qualification metrics directly from raw forecasts and
truth. Maximum absolute difference from the saved summary was
1.1102230246251565e-16. It also reconstructed all 24 specialist marginal count
and success arrays directly from their performed training actions/outcomes;
all matched the saved values. This supplements the locked scorer's artifact,
causal-history and mixture verification. It is not an independent replication
of ordinary fitting updates.

Sources: [locked protocol](../../docs/causal_access_protocol.md),
[complete numerical summary](../causal_access/development/summary.json),
[stream summary](../causal_access/development/summary.md), and raw main-cohort
forecasts and reference training arrays under `runs/causal_access_development`.
