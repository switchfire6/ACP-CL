# Matched return-context diagnosis

Status: retrospective diagnostic protocol, specified on 2026-09-27 before the
additional checkpoint predictions described here. The original evidence study
is complete. This analysis does not alter its prospective screens or failures.

## Question and scope

Separate the contribution of deployed parameter age from the context available
to express predictions when an earlier environment returns. Then compare
earlier and current parameters under identical supports to determine whether
useful earlier predictions have deteriorated. This is a diagnosis of the
existing benchmark, not a new continual-learning candidate or confirmation.

Use every pilot seed 12001 through 12006, separately for the conditional and
recurrent models. Use all three return blocks, indices 6, 11 and 16. No outcome
dependent exclusions, parameter fitting, hyperparameter selection or training
are allowed. Do not modify the old runs or sealed reports.

## Matched evaluation

For each seed and return, use the same 512 query observations from
`Evaluator.dataset(law)` and the same evaluator-only clean counterfactual
outcomes for every weight and support condition. The returned law is the exact
original base law. Query actions and outcomes must never determine ordinary
support selection or enter model fitting.

Compare these parameter sources:

- Draft at the start of the return.
- Periodically deployed snapshot at the start of the return.
- Sustained-evidence snapshot at the start of the return.
- Draft at the end of the original prefix.
- Draft at the end of the previous return, for returns two and three only.
- Draft at the end of the current return.

Cross every available parameter source with exactly the same supports:

- Actual start history: the last observed packet of the preceding block.
- Histories appending the first 8, 16 or 32 records from the first observed
  return packet, then keeping the newest 32 records. Precisely:
  `concatenate(start_history, first_return_packet[:n])[-32:]`; this drops the
  oldest `n` records and retains the most recent `32-n` previous records.
- Actual end history: the observed history after the current return.
- Fresh clean supports from the existing evaluator's fixed support replicates
  for the target law, averaged within seed.

The first return packet contains clean observed feedback. Frozen-parameter
predictions after inserting it measure inference-only recovery. Partial packet
histories are diagnostic interventions; the original learner normally updates
in batches of 32. Fresh target-law support is a privileged diagnostic reference,
not information available before feedback in ordinary deployment.

At return start the preceding history is clean but describes a different law.
The transition changes the base mode and removes the three added dependencies.
The image distribution alone does not reveal the hidden base-mode transition.
Failure before receiving new feedback therefore includes an information limit;
it must not automatically be attributed to forgetting or poor context inference.

## Outcomes and comparisons

For each complete factor combination, record all-action Brier loss averaged
over queries and all three horizons, and clean terminal survival for the
action with greatest predicted terminal survival. Lower Brier is better.
Retain raw rows with seed, return, source, support and query fingerprints,
model hash, scores and relevant checkpoint identities.

Primary descriptive comparisons are:

1. Sustained snapshot minus live draft at actual start, first-32-observed and
   fresh target supports, all on identical queries.
2. Current-start parameters minus prefix and previous-return parameters under
   identical target supports, and the corresponding current-end comparison.
   Positive Brier differences denote worse accessible predictions at the
   current parameter state than at the named historical state.
3. Start-to-end parameter change under an unchanged support condition.
4. Inference-only recovery from actual-start to increasingly refreshed support.

Let `L(w,h)` be matched Brier, `C_w = L(w,h_actual) - L(w,h_target)` and
`G_h = L(w_sustained,h) - L(w_draft,h)`. Report the exact identity
`G_actual = G_target + C_sustained - C_draft`. It is a prediction-level
decomposition on these queries, not a causal percentage explaining the whole
online return penalty.

Average returns and support replicates within each seed before estimating
cross-seed means or uncertainty. Use 20,000 paired seed bootstrap resamples
with seed 27192026, separately by architecture. Show per-seed values and 95%
percentile intervals. No new retrospective pass/fail thresholds are introduced.
Returns, support replicates, queries and horizons are not independent seeds.
Previous-return contrasts use only returns two and three for both sides of
each paired difference; do not compare a two-return average to a three-return
average.

An optional best single head chosen by query outcomes is permitted only as an
explicitly privileged conditional-model diagnostic. It is not a deployable
selection rule or necessarily a bound on a mixture. Its omission is acceptable.

## Interpretation and next decision

The snapshot-versus-draft comparison identifies an effect of choosing those
specific parameter states under fixed support. It does not isolate a causal
dose-response to age: the sustained snapshot also reflects acceptance history.
Any relationship to recorded snapshot age is descriptive.

If current parameters recover useful predictions with matching feedback,
prioritize the support/inference mechanism. If earlier parameters outperform
current ones under matching supports, investigate interference in parameter
updates. This establishes deterioration of accessible predictions, not erasure
of particular neurons or concepts. If end-of-return parameters improve over
start parameters despite matched supports, parameter adaptation matters.

Historical parameters may help base-law returns while failing to learn new
dependencies. Such a result alone does not support retrieval as a general
learning mechanism. A new candidate motivated by this diagnosis requires a
separate prospectively locked development protocol and fresh seeds; a positive
development result requires independent confirmation.

## Provenance and verification

Before evaluating scientific checkpoint queries, verify the old source/archive
identity and checkpoint hashes against archived integrity records. Save a lock
containing the input identities, script, tests, this protocol and imported
analysis-helper hashes, plus a source snapshot. Check loaded model, optimizer,
memory (including RNG), history and model train/eval flags before and after
evaluation. Use copied predictors or restore flags exactly, including aliased
snapshots. Diagnostic inference must not change training state. Record the actual command, runtime identity,
completion status, raw rows and summaries in a new output directory.

Small synthetic tests verify history construction, scores, paired aggregation,
contrast signs and no-mutation behavior. They must not evaluate pilot data or
stand in for independent scientific replication. Preserve the existing 338
artifacts in seven completed research archives and their source locks.

## Numerical-validation amendment

The first analysis lock was created at 22:46:01 UTC on 2026-09-27. Evaluation
completed the first conditional seed, then stopped during the second because
the new validator required predicted probabilities to lie exactly in [0, 1].
No score tables were published or inspected. That attempt's lock and code ZIP
remain at `reports/return_context`; its failure is recorded there.

The corrected attempt permits at most eight float32 machine epsilons outside
the probability interval, to accommodate floating-point mixture summation.
It does not clip predictions or alter the original evaluator's score arithmetic.
Truth values still must lie exactly in [0, 1]; larger prediction excursions
remain errors. Actual prediction minima and maxima are recorded so that the
roundoff explanation can be checked rather than assumed. Synthetic regression
tests cover both tolerated roundoff and substantive invalid values. The full
fixed analysis is rerun under a new code/protocol lock in
`reports/return_context_v2`. No scientific factors or comparisons are changed.
