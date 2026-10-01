# Step A1: retrieval capacity of the saved interleaved learners

This is a prospective diagnostic run under
[docs/retrieval_capacity_protocol.md](../../docs/retrieval_capacity_protocol.md),
which was written before any computation.

- It is a diagnostic on saved models only.
- It does not reopen the stopped representation recipe.
- It changes no locked record, and the locked STOP still stands.

Files: script `scripts/retrieval_capacity.py`; raw outputs `results.json` and
`per_seed.csv` in this folder. Runtime was 36.6 s on CPU (6 workers × 1 thread,
deterministic).

## Adjudicated verdict: BORDERLINE (not claimed as PRESENT)

| Pre-declared criterion | Value | Met? |
|---|---|---|
| Seed-mean closed fraction ≥ .75 | .848 | Yes |
| ≥ 5/6 seeds close ≥ .5 | 5/6 (18105: .423) | Yes |
| Base B control within .02 of its saved-support Brier | Seed mean −.0190; per seed only 3/6 within | Ambiguous |

The protocol did not say whether the control check is applied to the seed mean
or to each seed, or whether it is one-sided or two-sided. The implementer coded
it before the full run as per-seed and two-sided, and under that reading the
verdict is **NOT PRESENT**. The other plausible readings give PRESENT.

We do not select the favourable reading after seeing the data, so the result is
recorded as **borderline**. The control's purpose was to detect generic gains
from fitting a context, and it did detect them: fitting c improves Base B, a law
that already works, by .006–.027.

What holds under every reading:

1. **The frozen decoder can still express Base A.** A single fitted 12-d context
   c* gives .094. The Base A block-end level E is .085, the saved-support score S
   is .196, and O is .045.
2. **The context network cannot reach c\*.** c* lies 2.0–4.3 (L2 norm) from the
   initial mean context. The contexts that support produces sit only about
   1.2–1.4 from their own mean.
3. **Some of the gain is generic, not retrieval.** After subtracting each seed's
   Base B gain (a post hoc adjustment), the closure is about .48 on the mean and
   .71 on the median. The headline .85 overstates pure retrieval capacity.
4. **Producing the context from a single support is very noisy.** Across single
   fresh Base A supports, the Brier has a per-seed SD of about .09 and ranges
   .08–.41. Averaging the forecasts over 64 supports removes about 37% of the
   gap.
5. **The encoder features still support Base A.** Refitting only the decoder
   (P3) reaches .063, better than E, on every seed. Each refit decoder is a
   single-law specialist, though: it scores .353 when applied to Base B.

**Interpretation.**

- The main Base A failure lies in **producing the context from evidence**: it is
  noisy, and it cannot reach the region of context space that the decoder needs.
- The Base A knowledge itself is largely **not erased**; either the context input
  or a refit decoder can still express it.
- Evidence-indexed, retrievable context codes (Step C) remain a reasonable target
  for testing.
- Given the borderline control, Step C must include the K=1 and "fit-a-context"
  controls the critique requested.

**Link to Step B (descriptive).** With the encoder frozen, a decoder refit on
8,192 records reaches .063–.069 in any single law. That is well below the
learner's usual plateau of about .085. The competence gap is therefore not
mainly a lack of encoder features: the decoder and its optimisation leave
performance unused.

## Base A: probes P1, P2 (primary) and P3

S is the saved-support endpoint Brier (locked). E is the mean block-end Brier
over Base A cycles 2–8, from the `result.json` curve. The closed fraction is
(S − B)/(S − E).

| Seed | S | E | O | P1 single | P1 averaged forecast | **P2 c\*** | **P2 closed** | P3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 18101 | .2665 | .0819 | .0436 | .2547 | .2137 | .1121 | .836 | .0651 |
| 18102 | .2145 | .0826 | .0430 | .1769 | .1354 | .0983 | .881 | .0596 |
| 18103 | .1146 | .0757 | .0440 | .2099 | .1560 | .0769 | .970 | .0658 |
| 18104 | .2866 | .0964 | .0444 | .1867 | .1461 | .0988 | .988 | .0649 |
| 18105 | .1008 | .0855 | .0492 | .1838 | .1411 | .0943 | .423 | .0652 |
| 18106 | .1905 | .0855 | .0439 | .2151 | .1715 | .0861 | .994 | .0590 |
| Mean | .1956 | .0846 | .0447 | .2045 | .1606 | .0944 | .848 | .0633 |

Decoding the averaged P1 context, instead of averaging forecasts, gives .1878
(about 14% closure).

**P2 controls.**

| Seed | Base B saved | c\*_B on B | Difference | c\*_A on B (specificity) |
|---|---:|---:|---:|---:|
| 18101 | .1223 | .1053 | −.0171 | .3591 |
| 18102 | .1130 | .0873 | −.0257 | .3904 |
| 18103 | .1075 | .0861 | −.0214 | .3941 |
| 18104 | .0965 | .0905 | −.0060 | .3682 |
| 18105 | .1035 | .0870 | −.0166 | .3699 |
| 18106 | .1298 | .1026 | −.0272 | .4031 |
| Mean | .1121 | .0931 | −.0190 | .3808 |

The seed 18105 and 18103 closures are noisy, because their S − E denominators
are small (.015 and .039).

## Delay (descriptive)

The first delay-active law has cue benefit (flipped minus correct, affected
subset) as follows:

| Seed | Active cues | Saved learner | P3 decoder refit | O | I | P3/O | P4 cue-1 linear accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| 18101 | 1,2 | −.0004 | .0091 | .0139 | .0115 | .65 | .675 |
| 18102 | 0,1 | .0027 | .0060 | .0187 | .0144 | .32 | .671 |
| 18103 | 0,1,2 | .0020 | .0003 | .0111 | .0055 | .03 | .653 |
| 18104 | 1 | .0011 | .0061 | .0185 | .0126 | .33 | .699 |
| 18105 | 1 | −.0014 | .0038 | .0151 | .0077 | .25 | .725 |
| 18106 | 0,1,2 | .0021 | .0073 | .0123 | .0051 | .60 | .679 |
| Mean | | .0010 | .0054 | .0149 | .0095 | .36 | .684 |

P4's held-out linear decodability, as seed means:

| Feature | Accuracy |
|---|---:|
| Cue-0 bit | .711 |
| Cue-1 bit | .684 |
| Cue-2 bit | .910 |
| Delayed factor | .819 |
| Lossy factor | .942 |
| Source | 1.000 |

Reading the delay results:

- The pre-declared pattern was "cue 1 decodes well, but P3 stays below 25% of
  O". It is **not observed**.
- Cue 1 is the weakest linearly decodable feature (.68).
- A decoder refit recovers about 36% of O on average. That is about 5× the
  saved learner, but still far below O.
- So delay information is lost at two stages. The encoder represents cue 1 only
  weakly, and the saved decoder does not use even what is there.

## Integrity

- **Checkpoints.** All checkpoints were loaded with the identity-checked loader.
  Each learner signature equals the recorded `end_signature`, and is unchanged
  before and after every probe. Every probe ran on a deep copy, and the
  torch/numpy/python RNG states are unchanged.
- **Forward reproduction.** The re-implemented forward pass reproduces all 132
  saved endpoint evaluations bit-exactly.
- **Scoring.** Scoring uses the verified locked-rule scorer on the saved panels
  and supports.
- **Fitting-data seeds.** Seeds are `trial_seed(seed, "retrieval_capacity_P*",
  ...)`. All 1,410 per seed are unique, with no overlap with 2,057 project
  seeds. P1–P3 use only images, the performed action and its three labels. P4
  uses evaluator factors as probe labels, which a probe inherently needs.

## Deviations and implementation choices

1. **The control rule's reading was not specified** (see the verdict). The
   implementer had seen one seed's control value in a one-seed test before
   coding the reading.
2. **The P2 starting point** is the mean context of the 64 P1 supports, which
   the protocol did not specify.
3. **P2/P3 training details:**
   - BCE clamped at 1e-6, with a fresh Adam optimiser.
   - P3 has no gradient clipping.
   - P3 minibatches are fresh 32-record batches, each using its preceding batch
     as context, reshuffled each epoch.
4. **P4** uses torch LBFGS logistic regression (float64, L2 1e-3), because
   sklearn is unavailable.
5. **Gradients.** `copy.deepcopy` drops parameter gradients, so they were
   mirrored onto the copy for the equality check. Signatures were always taken
   on the original.
6. **Units.** The six seeds are the units. There are no significance claims.
