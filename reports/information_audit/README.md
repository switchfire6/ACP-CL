# Information audit of the representation-learning qualification

**POST HOC / EXPLORATORY / NOT DECISION-BEARING.** Prepared 2026-09-28, after the
qualification was locked.

- No learner was fitted.
- No `*.pt` checkpoint was opened or executed, and no study runner was called.
- No locked file was modified (`runs/`, `reports/representation_learning*`,
  configs, protocol).
- The locked **STOP** stands. Nothing here authorises the prepared main config
  or any change to thresholds or cues.

All ideal-observer and oracle quantities use evaluator-only knowledge: the law
identities, the simulator and the noise law. They are analysis references, not
learners.

The follow-on design proposal is
[docs/recoverable_context_proposal.md](../../docs/recoverable_context_proposal.md).

## Files

| File | Contents |
|---|---|
| `scoring_spec.md` | How the 31 locked cells are computed: trace layout, masks, averaging order, the information boundary, and two structural observations. |
| `loader_verification.json` | Loader rescoring of the 31 cells against the locked summary (max diff 0.0), plus exact regeneration checks. |
| `evidence_map.md` | Read-only cross-study ledger: every delay cue-benefit value, old- and returning-mode competence, where thresholds came from, and a chronology of screens. |
| `literature.md` | Literature scan of 35 works for a new continual-learning design, with synthesis and candidate contributions. |
| `online/summary.md` | Online (prequential) ideal-observer reanalysis of all 393,216 saved online cases. |
| `online/results.json` | Machine-readable results of the online reanalysis. |
| `online/*.csv` | Online tables: `cycle_brier`, `fresh_chunk_brier`, `position_profile`, `cue_contrasts`, `replay_and_seed_table`. |
| `online/*.png` | Online figures: Brier by cycle, within-block profile, delay-cue contrast. |
| `ideal_observer/summary.md` | Endpoint Bayes ideal-observer audit: O, I and P against the learner on all 31 cells, with posterior, information, decision relevance and verification. |
| `ideal_observer/results.json` | Main endpoint results. Per cell: O/I/P/learner per seed and mean with MC SE. Also posterior diagnostics, λ and gauge sensitivity, 3-channel KL, identification check, decision relevance. |
| `ideal_observer/review_independent_K8192.json` | The adversarial reviewer's independent re-implementation, compared on all 93 predictor-cell means. |
| `ideal_observer/review_16laws_K4096.json` | Observer with a 16-law uniform prior, and the oracle's expected decision value of each cue. |
| `ideal_observer/independent_check.json` | Second independent re-implementation, with gauge-integrated reserves: per-slot scores, posteriors, delay-pair LLR from fresh supports, pipeline checks. |
| `ideal_observer/independent_check_exact.json` | Exact-reserve variant of that check, plus panel-conditional delay-pair KL. |

| Script (`scripts/`) | Purpose |
|---|---|
| `ideal_observer_load.py` | Read-only loader, locked-rule rescorer and evaluator-case regenerator. Produces `loader_verification.json`. |
| `ideal_observer_mc.py` | Monte Carlo worker: replicates the supply draw and calls the project's `AcquisitionWorld.simulate`. |
| `ideal_observer_representation.py` | Main endpoint audit. Produces `ideal_observer/results.json`. |
| `ideal_observer_online.py` | Online reanalysis. Produces `online/`. |
| `ideal_observer_check.py` | Second independent check, with gauge reserves. Produces `independent_check.json`. |
| `ideal_observer_check_exact.py` | Companion check with exact reserves and panel KL. Produces `independent_check_exact.json`. |
| `ideal_observer_review_independent.py` | The reviewer's re-implementation, with its own physics and the locked scorer. Produces `review_independent_K8192.json`. |
| `ideal_observer_review_16laws.py` | The reviewer's 16-law observer. Produces `review_16laws_K4096.json`. |
| `ideal_observer_review_supply_replay.py` | Exact PCG64 replay of the supply draws (prints only). |

## Headline findings

- **The probes do not limit the result.** An ideal observer that sees only the
  query frames and the saved 32-record support passes all 31 cells. So does a
  weaker 16-law-prior observer. The four failed cells are not information limits
  of the probes.
- **Base A's hidden mode is fully identifiable.** Every Base-A support
  identifies it (posterior 1.000).
  - The ideal observer's Base-A Brier is .045. The learner's is .196, and the
    context-free mixture P scores .255.
  - Online, the learner does not improve on Base A across cycles: .110 in cycle
    1 and .111 in cycle 8.
- **The learner uses context online, but only partly, and then overwrites it.**
  - At the first same-law support, its Base-A error drops from .357 to .200
    (59% of the switch cost). The ideal observer drops to .044.
  - The rest is re-learned in the weights on each visit.
  - Between visits it loses .085 Brier, as four mode-B blocks overwrite it.
  - The locked endpoint (.196) matches this state of "support without a weight
    update".
- **The delay dependency is learnable from the probe, but the learner shows
  almost none of it.**
  - Interleaved delay cue benefit: ideal observer .010-.013; even the
    context-free, law-aware P .005-.008; learner -.0003 to .0008.
  - Online, the learner's cue-1 response is a generic pulse bias. It is equally
    present on non-delayed cases (difference .0002 ± .0023).
- **The delay law is the hardest to identify, but it matters little for
  decisions.**
  - It needs about 112 records for 20:1 expected odds. Only 6-19% of 32-record
    supports reach 20:1.
  - Flipping the cue costs the oracle 1.8/256 delayed queries at stage 3 (up to
    3/256). Ignoring the cue costs .00025.
  - Directional efficiency is similarly small (.0007). Only the hidden mode and
    supply timing (.093) matter for action choice.
- **The learner is far from the ideal observer everywhere, not only in the
  failed cells.**
  - It trails the ideal observer's endpoint survival by .067-.113.
  - On 4 of 5 slots it is below the context-free mixture: .041-.047 lower
    survival and .022-.029 worse Brier.
- **The results are cross-checked.** Two independent re-implementations and an
  exact PCG64 supply replay reproduce the main audit within MC error, with
  identical pass/fail flags.
