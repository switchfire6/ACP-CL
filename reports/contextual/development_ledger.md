# Matched-history development ledger

The accepted project identity is Continual Relational Learning. Package/command
names and all previous training sources remain compatible with archived studies.

Before building a new intervention, a post-hoc diagnostic loaded the eight
archived recurring-stream conditional models. It used new query seeds, separate
384-record calibration sets, 1,024 query trials, and four 32-record support sets.
No weights changed. Known ungated performance was 70.26% with ordinary routing
and 70.21% with calibration-selected heads. Gated performance was 63.64% with
ordinary routing, 63.52% with a calibrated global head, and 68.19% with a head
selected separately for each gate value. The latter selector knows the gate's
sensor location and condition grouping; it is a privileged diagnostic, not a
usable baseline or proof that routing explains every acquisition failure.
All per-seed results are preserved in `routing_diagnosis.json`.

This motivates one intervention: a learned linear correction from the query's
raw-image features to the evidence-routing scores. It begins at zero, retains
four predictors, adds 260 parameters, and receives no gate labels. The recurrent
comparator compresses the same 32 observed image/action/outcome records with a
GRU. Its widths (38 for context, 66 for decoding) were chosen by parameter counts
alone: 57,883 versus 57,852 for the old model and 58,112 for query routing.

## Development selection rule, recorded before development runs

Use seeds 117 and 118, long gaps, and pooled/recurrent/oracle models only.
Engineering smoke uses seed 19 and inadequate exposure, not evidence of learning.
All other settings match the new development config. Try common update budgets
of 3, then 6, then 12 steps per incoming batch, stopping at the smallest budget
meeting the following means across the two development seeds:

* Oracle initial four-block acquisition at least 5 percentage points above no transfer.
* Recurrent initial four-block acquisition at least 5 points above no transfer.
* Recurrent inference-only return at least 5 points above no transfer.
* Correct minus opposite support for that recurrent return probe at least 2 points.

The selected common budget applies to every scientific method. Neither query-
routed candidate outcomes nor the eight new pilot seeds may select settings.
If no budget passes, report an inadequate recurrent comparator and revise the
development design transparently before any claim of candidate superiority.
Retain every development configuration, result, source archive, and failure.

## Revision after the declared plain-GRU development grid

All three original budgets completed (54 challenge branches from 18 prefix
fits). The recurrent model's mean correct-minus-opposite history effect was
-0.033, +0.098, and -0.586 percentage points at 3, 6, and 12 updates. It learned
within-condition policies but did not qualify as a history-using comparator.
The oracle acquisition checks passed. No comparative pilot or candidate tuning
was launched. Full metrics for each budget are preserved beside this ledger.

Before new runs, revise ONLY the recurrent comparator's context encoding:
append the generic interaction phi(image) x one_hot(action) x (2*y12-1), and
average all causal GRU outputs over the past window instead of using just the
last state. Both operations use only records already available to every method.
They introduce no physical formula, gate label, mode label, or future outcome.
This is a revised baseline recipe; these development runs cannot identify which
of its two changes helps. Set context width 12 and decoder width 64 by parameter
count alone: 57,831 parameters, versus 57,852/58,112 for the reference/candidate.

Repeat the same 3/6/12 common-update grid, development seeds, and adequacy checks,
stopping at the first passing budget. The gate-routing candidate and pilot seeds
remain excluded from setting selection. Preserve both baseline versions.

## Selection after the revised grid

The interaction/mean-pooled recurrent recipe passed at 12 updates per batch,
the last setting in the declared grid. Across seeds 117 and 118: oracle initial
acquisition +12.66 pp, recurrent initial acquisition +11.33 pp, recurrent
inference-only return +7.68 pp over no transfer, and correct-minus-opposite
history benefit +15.20 pp. The 3- and 6-update revisions failed. Select 12
updates for EVERY main-cohort method, with no other training-setting changes.
The complete six-setting development consumed 36 prefix fits and 108 branches;
all outcomes and exact configurations/sources are archived in development/.

Before the main lock, review found that the pooled model's broken-binding
probe inherited an override that accidentally enabled evidence routing.
Fix this diagnostic to retain the pooled model's uniform routing, and add a
regression test. This affects none of the training or adequacy/selection
metrics. Historical development records remain unchanged; disregard their
pooled broken-binding diagnostic. Checkpoint audits run with each setting's
exact archived source. The fresh pilot uses the corrected diagnostic.

The candidate has not been used to select settings. The main cohort's eight
new seeds remain uninspected when this selection is made. The same adequacy
checks must hold again in the main cohort before interpreting superiority.

## Fresh cohort outcome (after lock)

The locked cohort completed all 96 prefixes and 288 branches. The recurrent
comparator passed every fresh-cohort adequacy check. The query-routing change
failed the predefined screen: novel AUC -0.38 pp versus recurrence (3/8 positive)
and +0.22 pp versus the original conditional router (5/8 positive). It passed
the mean-loss guardrails. Retain the original router as the working reference.
All prefix/final checkpoints passed local data, memory, identity, and probe
recomputation audits. No settings were changed or seeds excluded after lock.
The full report distinguishes this failed intervention from the secondary
reuse and feature-plasticity findings; no further candidate tuning was run.
