# Step B2: the achievable competence ceiling and a decomposition of the gap

**Status: prospective study.** It was run under
[docs/competence_ceiling_protocol.md](../../docs/competence_ceiling_protocol.md).
Config, protocol and source hashes were locked in `runs/competence_ceiling/manifest.json`
before the main run. It is a single-law competence question, not a
continual-learning comparison.

**Scale:** 12 fresh seeds (19301–19312), 132 fits and 288 oracle tasks.

**Files.**

- In this folder:
  - `results.json`: all readings, bootstrap intervals and verification.
  - `per_seed.csv`.
  - `online_excess.csv`.
  - `learning_curves.png`.
- Code: `scripts/competence_ceiling.py` and `scripts/competence_ceiling_learners.py`.
- Raw runs: `runs/competence_ceiling/`.

## Verdicts (all pre-declared)

| Reading | Rule | Result | Verdict |
|---|---|---|---|
| **R1 Calibration** | The .010-over-O bar is infeasible at N = 8,192 if the feature-level ceiling C(8192) > .010 | C(8192) = **.0078** [.0070, .0086]; ≤ .010 in 12/12 seeds | **The bar stands.** It is achievable in principle at this N. |
| **R2 Decomposition** | A share is dominant if it is ≥ 50% of ON-ref's excess and positive in ≥ 10/12 seeds | Online optimisation Q = **64%** [62, 66], positive in 12/12 | **Q, online optimisation, is dominant** |
| **R3 Replay re-test** | ON-R256 removes ≥ 25% of ON-ref's excess and is lower in ≥ 10/12 seeds | **37.3%** [31.4, 42.9], 12/12 | **R256 confirmed** (Step B's narrow miss replicates as a clear effect) |
| **R4 Weight averaging** | ON-R256-EMA removes ≥ 25% of ON-R256's excess and is lower in ≥ 10/12 seeds | **47.8%** [44.5, 51.7], 12/12 | **EMA helps** |

- O's Brier is .0437; its Monte Carlo SE is at most 7e-5.
- ON-ref's excess at 8,192 arrivals is **.0379** [.0354, .0404]. This replicates
  Step B's .0341 on fresh seeds.

## The decomposition

E + P + Q equals ON-ref's excess exactly, per seed.

| Share | Definition | Mean [95% CI] | Fraction of ON-ref | Seeds positive |
|---|---|---:|---:|---:|
| **E**, estimation (finite, sparse data) | OFF-feature excess | .0078 [.0070, .0086] | .21 | 12/12 |
| **P**, perception and architecture | OFF-pixel − OFF-feature | .0058 [.0050, .0067] | .15 | 12/12 |
| **Q**, online optimisation | ON-ref − OFF-pixel | **.0243** [.0223, .0264] | **.64** | 12/12 |

**Robustness of the Q verdict.**

- OFF-pixel is not a true upper bound for this network. It is early-stopped
  (best epoch about 11 of 40), trains on 80% of the data, and does not average
  its weights. The online ON-R256-EMA learner (.0124) beats it in 10/12 seeds.
- A better pixel learner would therefore **shrink** P and **enlarge** Q. With
  the best observed pixel learner as the reference, P ≤ .0046 and Q ≥ .0255.
- So the dominance verdict is conservative.

## Arm × N table (seed means)

The cue columns show the learner's cue benefit minus O's benefit. O's benefits
are .0316 (cue 0), .0157 (cue 1, delay) and .2173 (cue 2).

| Arm | N | Brier | Excess [95% CI] | Survival regret | Cue 0 | Cue 1 | Cue 2 |
|---|---:|---:|---|---:|---:|---:|---:|
| ON-ref | 8,192 | .0816 | .0379 [.0354, .0404] | .030 | −.018 | −.011 | −.034 |
| ON-R256 | 8,192 | .0674 | .0237 [.0220, .0256] | .021 | −.017 | −.012 | −.029 |
| ON-ref-EMA | 8,192 | .0662 | .0225 [.0215, .0235] | .017 | −.017 | −.010 | −.012 |
| **ON-R256-EMA** | 8,192 | **.0561** | **.0124** [.0115, .0133] | **.010** | −.017 | −.011 | −.010 |
| OFF-pixel | 2,048 | .0709 | .0273 [.0259, .0286] | .048 | −.027 | −.014 | −.048 |
| OFF-pixel | 8,192 | .0573 | .0136 [.0127, .0147] | .014 | −.019 | −.011 | −.022 |
| OFF-pixel | 32,768 | .0521 | .0084 [.0075, .0095] | .006 | −.007 | −.006 | −.010 |
| OFF-feature *(privileged perception)* | 2,048 | .0639 | .0202 [.0191, .0213] | .033 | −.018 | −.011 | −.048 |
| OFF-feature | 8,192 | .0515 | .0078 [.0070, .0086] | .011 | −.011 | −.007 | −.020 |
| OFF-feature | 32,768 | .0471 | .0034 [.0029, .0040] | .006 | −.006 | −.004 | −.009 |
| OFF-feature-all *(privileged perception and labels)* | 2,048 | .0520 | .0084 [.0080, .0087] | .009 | −.012 | −.008 | −.019 |
| OFF-feature-all | 8,192 | .0469 | .0033 [.0028, .0037] | .006 | −.006 | −.004 | −.007 |
| OFF-feature-all | 32,768 | .0451 | .0014 [.0011, .0018] | .002 | −.002 | −.001 | −.003 |

**The cost of sparse labels in the ceiling** is OFF-feature minus
OFF-feature-all:

| N | Cost | Seeds |
|---|---:|---|
| 2,048 | .0119 | 12/12 |
| 8,192 | .0046 | 12/12 |
| 32,768 | .0020 | 12/12 |

The chosen weight decay was 0 in every OFF-feature fit.

**Online excess per 1,024-arrival chunk.** O's online Brier is about .043–.044
in every chunk.

| Arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ON-ref | .098 | .051 | .041 | .038 | .037 | .036 | .036 | .036 |
| ON-ref-EMA | .170 | .076 | .035 | .024 | .023 | .021 | .022 | .021 |
| ON-R256 | .098 | .047 | .035 | .029 | .027 | .026 | .026 | .024 |
| ON-R256-EMA | .170 | .076 | .034 | .020 | .016 | .013 | .013 | .012 |

## Interpretation

1. **The single-law gap is mostly an online-optimisation problem.**
   - The reference's single-pass, constant-step, small-window update rule
     costs about two-thirds of its excess.
   - Perception from pixels costs about a sixth, and finite sparse data about a
     fifth.
   - Two cheap, standard remedies each remove a large, consistent share:
     - replaying the whole history (R256 never evicts at 8,192 arrivals);
     - predicting with an exponential moving average of the weights.
   - Together they cut the excess by 67%, from .0379 to .0124. That leaves the
     learner .0046 above the feature-level ceiling, and within .010 of C(8192)
     in 12/12 seeds.
2. **The competence bar is achievable in principle but is not yet met.**
   - ON-R256-EMA meets the .010-over-O bar in only 1/12 seeds.
   - On the ceiling curve, the .010 bar is met at about N = 8,192 with perfect
     perception, and at about 32,768 with pixels offline.
3. **Weight averaging buys competence with lag.**
   - The EMA is much worse in the first chunk (.170 vs .098), because it still
     carries the initial weights. It overtakes the online weights only from the
     third chunk.
   - In a stationary law this is a one-off cost. With recurring laws it would
     recur at every switch, and a single average would blend the laws.
   - This is the stability–plasticity trade-off appearing *inside* one
     context. It is the declared next question (see below).
4. **Every learner under-uses the relational cues relative to O**, including
   the privileged ceilings at small N. Averaging mainly recovers supply timing
   (cue 2: −.029 → −.010). The deficits on directional efficiency and delay
   (cues 0 and 1) barely change with EMA or replay. Offline, they shrink only
   with more data.

## Declared implication and next step

The protocol declared that if Q is dominant, "the single-pass online update
rule, not the architecture or the data, limits competence. The next learner
work is online consolidation (weight averaging, slow/fast weights). This is
also a continual-learning mechanism, so it must then be tested against
plasticity under law switches."

This connects Step B2 to the Step A1 result:

- **A1:** minority-context knowledge was largely still expressible by the
  network, but not retrievable from evidence.
- **B2:** slow, averaged weights are what make a context's knowledge
  competent. A single average, however, cannot serve several recurring
  contexts without lag and blending.

The natural candidate is therefore **evidence-indexed consolidated
memories**:

- fast online weights for plasticity;
- a small library of slow, averaged snapshots, one per inferred context;
- selection among the snapshots by their predictive likelihood of recent
  consequences.

The design and protocol follow in
[docs/consolidated_recall_protocol.md](../../docs/consolidated_recall_protocol.md).

## Integrity, failures and deviations

- **Locking.** Locked before the main run, and the lock was re-checked at
  analysis.
- **Smoke checks.** The smoke `check` passed all nine checks:
  - settings and shared initialisation;
  - EMA non-interference, bit-identical to Step B's `run_fit`;
  - determinism;
  - offline data equal to the exact stream prefix;
  - O reproduces truth;
  - disjointness;
  - losses, schedule and features;
  - resumption;
  - the smoke pipeline.
- **Integrity addition (not in the protocol; added before locking).** A
  replicate pass re-fit ON-ref (with EMA), OFF-pixel N = 8,192 and OFF-feature
  N = 8,192 from scratch for seeds 19301, 19306 and 19311. All nine are
  **bit-identical** to the main run. All 132 main fits were re-verified against
  their stored hashes (`runs/competence_ceiling/replicate.json`).
- **Independent spot check.** The lead recomputed O and OFF-feature excess from
  raw arrays for all 12 seeds. They are identical to `per_seed.csv`.
- **Machine faults.** This machine (Ryzen 9 5900X, RAM at a 3600 MT/s XMP
  profile) showed repeated transient faults during this work:
  - two python310.dll access violations in the smoke phase;
  - an abrupt worker death at 06:46 in run attempt 1. The pool broke and the
    parent hung; it was killed after 15 tasks had completed. All 405 pending
    tasks are logged as failed.
  - **silent on-disk corruption** of one archive
    (`feature/performed_N2048_19312/panel.npz`), detected by the runner's final
    hash verification. The zip CRC failed in one member.

  The corrupted fit was moved to `quarantine/` (kept, not deleted) and
  regenerated in attempt 3. The regenerated fit equals the quarantined one in
  every readable array. A CRC scan of all 637 other archives passes. Every
  event is logged in `attempts.jsonl`.
- **Protocol ambiguities, resolved before locking:**
  - The validation holdout is the nearest integer number of packets: 13, 51
    and 205.
  - OFF-pixel runs all 40 epochs, with no patience, and keeps the
    best-validation epoch.
  - The ensemble members share their minibatch order, following the
    protocol's literal "differ only in initialisation".
  - Weight decay is Adam's coupled L2.
  - The source factor is coded as a bit.
  - EMA forecasts use the online learner's preceding batch as support.
- **Units.** Twelve seeds are the units. Intervals are paired seed-bootstrap
  intervals (20,000 resamples, seed 19392026).
