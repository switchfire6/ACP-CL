# Adam: replay decomposition and continuous renewal

Status: prospective, before either new learner cohort. Working project name:
Adam. Optional expansion: Adaptive Dynamics and Associative Memory. This name
does not refer to the Adam optimizer; package paths remain `acp_cl`.

The previous factorial study qualified both architectures and found that replay
reset helped acquisition while introducing retention and noise costs. Its reset
changed stored packets, reservoir age, and random generators together. The two
experiments here separate those operations and test one fixed continuous policy.
Neither introduces a new architecture or selects settings on seeds 8001--8006.

## Shared recipe and integrity

Keep the conserving world, temporal images, conditional/recurrent models,
32-record causal history, 16-packet capacity, learning rate .002, and 12 updates
per 32-arrival batch. Four alternating original blocks contain 1,024 arrivals
each. Each challenge contains 8,192 arrivals, evaluated every 256 on 512 queries
with two fresh-support replicates. Feedback replacement probability is .2.
Counterfactuals, change markers and physical law labels belong to evaluation.

Retain the qualified exposure from the previous study without a new budget
search. Recheck fresh-weight learnability at marginal Brier gain >= .02 and
correct-cue benefit >= .002 in each available stage and cue group, per model.
A failed check stays failed; finish the fixed cohort and flag its interpretation,
without changing settings or replacing seeds. The tiny engineering smoke is
not a scientific qualification attempt.

Use four one-thread CPU workers, affinity mask 85 on this Windows host. Record
the actual mask in the runtime identity. File writes flush data before atomic
replacement; this reduces an observed recovery weakness but does not guarantee
survival of every power/storage failure. Inspect stopped-run files before resume.
Source, configuration and this protocol are archived before each cohort. Keep
all attempts and all valid completed work; no outcome-based retries.

## Experiment A: reset decomposition at the first introduction

Seeds 9001--9006 cover all six cue orders. Fit 12 common prefixes, then fork
seven arms at the first new dependency: **84 episodes**. This is a local
first-introduction diagnostic, not a repeated-renewal policy evaluation.

| Arm | Stored packets | Effective reservoir age | Random generators |
|---|---|---|---|
| keep | Keep | Keep | Keep |
| clear | Empty | Keep | Keep |
| rebase | Keep | Set to stored count | Keep |
| clear_rebase | Empty | Zero | Keep |
| rng_reset | Keep | Keep | Reset membership and sampling to original seeds |
| full_reset | Empty | Zero | Reset both to original seeds |
| fresh | Original random weights, fresh optimizer/replay; same recent history | Zero | Original seeds |

The six learned-weight arms retain exactly the same optimizer, weights and
recent history. Reservoir replacement uses effective age. A separate lifetime
counter supplies unique packet IDs and never influences sampling. Rebasing a
nonempty reservoir sets its age to its current occupancy, rather than assigning
an impossible age below the number stored. Clearing without rebasing retains
the old replacement denominator; the buffer still refills before replacements.

Primary: affected-subset prediction Brier AUC. Secondary: valid-old error level
and change, query-cue use, survival, and replay ages. Within seed compute clear
minus keep, clear_rebase minus rebase, rebase minus keep, clear_rebase minus
clear, their main effects and interaction. Also report rng_reset minus keep
and full_reset minus clear_rebase. These last two contrasts isolate generator
reset in two particular memory states, not a complete three-factor experiment.

Only two seeds introduce each cue first. Qualification by cue is correspondingly
weak and must be displayed. This diagnostic has no intervention selection gate
and does not alter the policy experiment below, which is declared in advance.

## Experiment B: a fixed continuous replay policy

Seeds 10001--10006, both architectures. Compare three policies from random
initialization through the entire prefix and three introductions:

* uniform: the original uniform reservoir of 16 packets;
* recent: a rolling window of the latest 16 packets;
* split: the latest 8 packets plus a uniform reservoir of 8 older packets.

In split, an evicted recent packet becomes eligible for the historical reservoir.
The two stores are disjoint. Sample uniformly from their union: after filling,
each half gets half of replay draws in expectation. No extra gradient steps,
labels, stored episodes, task IDs or change triggers are supplied. All policies
use the existing current-plus-one-replay update. Empty replay duplicates current
data as before. The 8/8 split is fixed by equal allocation, not selected from a
sweep; this study therefore has no policy-development selection cohort.

Each policy advances its own learner through all three introductions. Its final
learner is separately cloned into return, revision, clean continuation and noisy
continuation. These branches do not advance one another. No state is reset at
any boundary. The clean/noisy streams share observations and performed actions.
Fresh random-weight references learn each of the three stages with uniform
replay and the same preceding raw history, shared across policy comparisons.

Counts: **36 prefix fits, 252 policy episodes and 36 fresh references = 288
episodes**. Policies have the same initial weights, stream and update budgets;
their later weights intentionally differ, including after the prefix. This is
an end-to-end policy comparison, unlike the common-weight diagnostic above.

## Policy endpoints and prospective screen

Primary candidate contrast: split minus uniform affected-subset Brier AUC,
averaging the three introductions within each seed. Also report recent minus
uniform, split minus recent, each cue and stage, every final branch, and all
fresh-reference outcomes. Use terminal valid-old error for between-policy
guardrails because their starting errors differ; report changes alongside levels.
Correct support and unchanged physical subsets define valid-old panels. Ordinary
noisy prediction probes retain the noisy support distribution.

The pilot screen is evaluated separately for each architecture. All conditions
below must hold on paired seed means; no pooling architectures to hide failure:

1. Fresh-reference qualification passes.
2. Split acquisition AUC improves by >= .002 Brier and improves in >= 5/6 seeds.
3. Mean terminal valid-old error across introductions rises by no more than .005.
4. For EACH final branch, prediction AUC and terminal valid-old error rise by
   no more than .005 versus uniform.
5. The extra matched-noise penalty, (split noisy - split clean) minus
   (uniform noisy - uniform clean), is no more than .005 Brier AUC.
6. Mean survival AUC declines by no more than .01 (one percentage point), both
   across introductions and separately in each final branch.

The minimum acquisition gain is about one fifth of the prior full-replay-reset
gain; tolerances are explicit pilot choices, not universal definitions of safety
or a claim that any loss is acceptable for every application. Passing mean-based
screens nominates a policy for independent confirmation. It does not establish
noninferiority: report bootstrap intervals and flag bounds crossing the declared
tolerances. A pass in one architecture is not an architecture-general result.

For both experiments use 20,000 paired seed bootstrap resamples, analysis seed
27192026, and show every seed. Six seeds, not episodes/actions/horizons, are
independent replicates. Intervals are descriptive without multiplicity correction.
Do not retrofit thresholds or interpret a failed policy as evidence that general
continual learning is impossible. No causal age or biological sleep claim follows.

## Required checks and decision

Verify exact reset isolation and persistent optimizer state; bounded, disjoint,
causal replay; original-reservoir equivalence; matching raw streams and update
budgets; fresh initialization; matched clean/noisy inputs; exact interruption
recovery; and recomputed starting/final probes from trusted local checkpoints.
Preserve intermediate curves, while distinguishing saved-record checks from
independent recomputation using intermediate weights.

If the fixed split passes, confirm it with more independent seeds before testing
a second environment. If it fails, identify which acquisition/retention/noise
requirement failed. Further policy choices require new development data. A
sampling-operation effect in Experiment A is a diagnosis, not proof that the
continuous split in Experiment B will work.
