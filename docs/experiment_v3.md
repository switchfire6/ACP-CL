# V3: causal allocation and local developmental windows

Written before v3 development outcomes. This is an exploratory mechanism study,
with separate development and locked evaluation seeds. It is not a preregistered
confirmatory study. V2 results and source versions remain preserved.

## Why this comparison

V2's strongest control used the completed run's mean feature gain and reset
schedule. V3 removes that future information. Every learner uses a common
40-update warmup, followed by a constant feature multiplier chosen on separate
development seeds. Resets occur every 20 updates after warmup, starting at
update 60, with exactly three units per registered population. Unit identities
depend on each learner's current utility. No learner receives boundaries,
domain IDs, evaluation feedback, or a future allocation trace.

The newborn peak is four times the selected mature feature gain, decaying
linearly back to the mature value over 30 updates. The factor four is fixed
before development and is not tuned. It may exceed a multiplier of one; the
common optimizer-displacement cap still applies. This makes the gain
intervention nonzero even if development selects a mature gain of one.

All v3 conditions share a 0.1 global L2 bound on the complete proposed optimizer
displacement, including momentum, classifier, weight decay, and anchors.
Structural resets occur afterward and are outside this bound. This controls
one source of large updates; it does not bound prediction changes or guarantee
retention. There is no global closure/reopening controller in this study.

## Conditions

| Method | Intervention after common warmup |
|---|---|
| `er_v3` | Fixed feature gain and replay; no resets |
| `recycle_v3` | Add the fixed reset-count/time schedule |
| `newborn_v3` | Add a 30-update decaying local gain window after a reset |
| `newborn_matched_v3` | Same window with causal redistribution to match the baseline's per-step nominal feature budget |
| `protection_v3` | Extend reset eligibility age from 30 to 60 updates; no gain window or consolidation |
| `consolidation_v3` | Consolidate useful units after their local 30-update accumulation window; no gain boost or extended eligibility |
| `full_v3` | Combine local gain, 60-update protection, and local consolidation |

The gain-only comparison keeps the same nominal mature-feature multiplier but
spends more nominal feature gain. The matched-budget comparison instead reduces
mature-feature gain to pay for the local allocation: it cannot also keep the
same mature-backbone update rule. The nominal match precedes clipping; applied
gains, gradient displacement, and compute need not match. Consolidation changes
incoming-row gains and anchors but does not itself veto resets. Every reset
request must be feasible under that condition's age rule; infeasibility fails
explicitly instead of silently changing the reset count.

These are single-component comparisons against the baseline plus a combined
condition, not a full factorial design. They do not separate every interaction
or estimate each component's effect conditional on the other two being active.

## Development and locking

The machine-readable plan is `configs/v3_study.json`.

- Development seeds: 411, 522; 60 experiences per stream.
- Candidate feature gains: 0.05, 0.15, 0.5, 1.0.
- Screen `recycle_v3` on recurring and stationary streams with class-independent
  colors throughout. This is 16 comparative development runs.
- No scratch fits: these are direct paired component comparisons. Acquisition
  AUC is reported without interpreting it as pure acquisition or transfer.

Selection requires all planned development runs to be complete and finite.
For each stationary run, take its first-experience validation accuracy at the
full-retention checkpoints (experiences 10, 20, ..., 60). Compute the greatest
drop from a previous such checkpoint to a later one on that identical set.
A candidate passes if neither development seed exceeds a 10 percentage-point
drawdown. This finite, sparse sample is a screen, not a stability guarantee.

Among passing candidates within 2 percentage points of the best passing mean
stationary final accuracy, maximize the equally weighted mean of recurring
final accuracy and recurring late acquisition AUC. Candidates within 0.5 points
of the maximum score tie; choose the smaller feature gain. If no candidate
passes, report the failed screen and revise development before touching the
locked evaluation cohort. Preserve every attempted candidate and failure.

The selection artifact records each input result's hash, source/runtime/config
identities, selection scores, drawdowns, and the chosen value. Freeze the three
evaluation configurations and selection artifact in git before running them.
No gain, clipping threshold, method, or outcome-based exclusion is changed in
response to the locked cohort.

## Locked evaluation

New seeds: 731, 842, 953; 100 experiences each.

1. Recurring domains, independent colors: all seven methods, 21 runs.
2. Stationary domain, independent colors: all seven methods, 21 runs.
3. Recurring domains, early color bias: `recycle_v3`, `newborn_v3`, `full_v3`,
   nine runs. Training color correlation is 0.95 for eight experiences, then
   zero. Evaluation colors are independent throughout.

This totals 67 comparative runs if the development screen passes. Four shape
labels recur across eight rendering domains, with 640 fresh current images
per experience, 32 current and 32 replay presentations per update, a 256-image
reservoir, and a width-64 CNN. Only the terminal adapter is recycled. New color
policies share RNG consumption so geometry, textures, jitter, and sensor noise
remain paired when the color association changes.

Evaluation uses each new seed's validation split; test examples are neither
generated nor used. These seeds are held out from parameter selection, but
the benchmark and model were chosen during development. Three seeds and a
procedural generator do not provide confirmatory or natural-image evidence.

## Endpoints and checks

The primary component contrast is `newborn_v3 - recycle_v3` on recurring late
acquisition AUC, through the first 256 current presentations of each experience,
averaged over the final half of the stream. Report final accuracy alongside
it, every seed's value, the complete acquisition trajectory, and exploratory
paired intervals. Other contrasts and conditions are mechanistic diagnostics;
no post-hoc winner is promoted to the primary hypothesis.

Full retention evaluation remains every ten experiences plus the final one.
Forgetting and stationary drawdown are maxima over observed checkpoints and
can miss intervening damage. Report clipping frequency, nominal and achieved
feature gain, head/feature displacement, actual reset counts, and maturation.
Audit identical initial warmup, training exposure, replay contents and sampling
RNGs, exact scheduled counts, nominal budget matching, and checkpoint resume.

The early-bias comparison estimates sensitivity to that curriculum under this
training rule. It does not identify early consolidation as a mechanism by
itself. Report the paired interaction `(full - recycle)_early_biased -
(full - recycle)_independent` on each shared seed, with bitwise identical
evaluation tensors across those curricula. The learning-rate choice is made for the main replay/recycling baseline
and then shared; this is a component comparison, not equal per-method tuning
or an exhaustive search for the best version of each method.

## Public research readiness

A useful public artifact does not require a positive hypothesis result. It
does require reproducible commands, explicit scope, complete negative results,
tests that exercise the actual interventions, and claims tied to evidence.
Before an actual GitHub publication, review repository contents and provenance,
choose a license with the owner, and agree on the destination and visibility.
No remote or publication is authorized by this protocol itself.
