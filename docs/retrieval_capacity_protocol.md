# Step A1: retrieval capacity of the saved interleaved learners

Prospective protocol, written 2026-09-28 before any computation. The user
approved it as Step A of
[recoverable_context_proposal.md](recoverable_context_proposal.md). It is a
diagnostic on saved models. It does not reopen the stopped representation
recipe, and it changes no locked record.

## Question

The six saved interleaved endpoint learners forecast Base A poorly when given
Base A support: Brier .196, against .045 for the ideal observer. At the end of
each Base A training block the same kind of learner reached about .083.

We want to know whether the knowledge needed for Base A is still expressible by
the frozen network and is simply not being **retrieved** from the evidence, or
whether it has been **overwritten**. We also want to know where the
arrival-delay information is lost.

## Materials and invariants

- **Checkpoints.** `runs/representation_learning_qualification/jobs/outcome_<seed>/interleaved/checkpoint.pt`,
  seeds 18101-18106. Load them with the project's identity-checked loader.
- **State is never mutated.** Every probe works on a deep copy. The original
  learner signature (weights, optimizer, replay, history and RNG) must match
  before and after.
- **Evaluation panels.** The saved locked endpoint query panels and truths, for
  every target law.
- **Fitting data.** All fitting data are fresh records from the project
  simulator. Their seeds are disjoint from every project seed:
  `trial_seed(seed, "retrieval_capacity_*", ...)`.
- **What the fitting data contain.** Only what an ordinary learner receives:
  images, the performed action (uniform random) and that action's three
  survival labels. No counterfactual outcome, law label or simulator variable
  enters any fit.
- **References.** O and I from `reports/information_audit/ideal_observer/results.json`,
  and the learner's saved-support forecasts. The Base A block-end level comes
  from the online reanalysis. In what follows, *S* is the learner's
  saved-support Brier and *E* is the block-end reference.

## Probes (per seed, all six seeds)

**P1, support resampling.** Draw 64 independent fresh Base A 32-record
supports. Report:

- the mean of the single-support Brier;
- the Brier of the forecast averaged over supports;
- the Brier obtained by decoding the averaged context vector.

This measures how much of the failure comes from noise in producing the
context.

**P2, context-only fit (primary).** Freeze every learner weight. Replace the
support-derived 12-d context by one free vector c.

- Fitting data: BCE on 8,192 fresh Base A records, performed action only.
- Optimiser: Adam, lr .01, 1,000 full-batch steps.
- Initialisation: the mean Base A support context.
- Score: all-case Brier of c* on the saved Base A endpoint panel.

Controls:

- the same fit on Base B data, scored on the Base B panel;
- the same fit on Base A data but scored on Base B (a specificity check).

**P3, decoder refit.** Freeze the encoder (frame and temporal modules) and the
context network. Re-fit the outcome decoder from its current weights on 8,192
fresh records of the target law:

- context from each record's actual preceding fresh 32-record batch;
- 20 epochs, Adam lr .002, minibatch 32.

Do this for Base A, and for the first delay-active law with its cue benefits.
This asks whether the frozen encoder features still support the relation.

**P4, linear decodability.** Fit a logistic probe on frozen query latents for:

- each pulse-cue bit;
- the delayed, lossy and source factors.

Use 4,096 fresh records, with a disjoint set of 4,096 records for evaluation.
Report held-out accuracy. This is a diagnostic only; decodable information does
not prove it is used.

## Pre-declared reading

Define the closed fraction as (S − Brier(c*)) / (S − E), computed per seed.

**Retrieval capacity is PRESENT if:**

- the seed-mean closed fraction is ≥ .75; and
- at least 5 of 6 seeds each close ≥ .5; and
- the Base B control stays within .02 of its saved-support Brier.

**Result and consequence:**

| Outcome | Interpretation | Next step |
|---|---|---|
| PRESENT | The frozen decoder can still express Base A. The failure is in producing and retrieving the context from evidence. | Step C (retrievable context codes) is justified. |
| Not present, but P3 closes the gap | The shared decoder was overwritten; the encoder features survive. | Codes over a shared predictor are not enough. Protecting decoder knowledge, or per-context decoders, is the target. |
| Neither P2 nor P3 closes it | The encoder no longer supports Base A. | The failure is representational. |

**Delay.** Report P3's delay cue benefit and P4's cue-1 decodability as
descriptive diagnostics, relative to O. If P4 decodes cue 1 well but P3's cue
benefit stays below 25% of O's, the encoder carries the cue but the
cue × direction × delayed conjunction is not formed from it.

The six seeds are the units. Report every seed. There are no significance
claims.

## Outputs

- `scripts/retrieval_capacity.py` (new).
- `reports/retrieval_capacity/results.json` and `summary.md` (new).

The script records:

- the checkpoint signature before and after;
- all seeds and fitting losses;
- runtime.
