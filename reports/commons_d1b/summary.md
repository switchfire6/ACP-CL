# Step D1b: escaping exploration traps — MB-count is competent

- **Protocol:** [docs/commons_d1b_protocol.md](../../docs/commons_d1b_protocol.md),
  including Amendment 1, made before the lock.
- **Development:** [dev/summary.md](dev/summary.md) and
  [dev2/summary.md](dev2/summary.md).
- **Scale:** locked, with protocol sha `eced34f6…`. 216 fits and 581 oracle
  tasks in 4.9 h, with **zero failures**.

**Integrity.** The replicate pass re-ran 18 fits from scratch (3 arms × 3
seeds × 2 contexts). All are **bit-identical**. All 216 main fits were
re-verified against their hashes, and all 1,050 archives passed the CRC scan.

The lead ran the replicate pass and the analysis with the locked script,
because the implementer's session hit a usage limit. The console logs are in
`runs/commons_d1b/`.

**Files.** `results.json`, `fits.csv` and `d1b_results.png`.

## Verdicts (pre-declared)

| Reading | MB (baseline) | MB-prior (β 3) | **MB-count (c .05)** |
|---|---|---|---|
| **R1 competence**: average ≤ .015 and worst ≤ .025, in ≥ 10/12 seeds | 5/12, not competent | 9/12, not competent | **12/12, COMPETENT** |
| Context average [95% CI] | .0122 [.0103, .0142] | .0106 [.0092, .0120] | **.0096 [.0088, .0104]** |
| Mean worst context | .0271 | .0197 | **.0140** |
| **R2 trap escape** (A1B1C1, 24 seeds): regret ≤ .025 and usage condition, in ≥ 20/24 seeds | 13/24; trap rate .46 | 14/24; trap rate .42 | **24/24; trap rate 0, ESCAPES** |
| Paired against MB: arm clean where MB trapped / arm trapped where MB clean | — | 7 / 6 (McNemar p = 1.0) | **11 / 0 (exact McNemar p = .001)** |
| **R3 payback**: cumulative online regret ≤ MB's in ≥ 10/12 seeds | — | 7/12, does not apply (not competent) | **12/12, PAYS BACK** |
| Net cumulative regret against MB (5 contexts × 4,096 episodes) | — | −44 [−104, +17] | **−139 [−203, −79]** |

**Declared implication:** MB-count is the **D2 core**. Its exploration scheme
is applied to every D2 arm, so learners are compared fairly.

## Per context: regret at 4,096 episodes (12 seeds)

| Context | MB | MB-prior | MB-count |
|---|---:|---:|---:|
| A0B0C0 | .0055 | .0060 | .0070 |
| A1B0C0 | .0103 | .0098 | .0116 |
| A0B1C0 | .0101 | .0076 | .0067 |
| A0B0C1 | .0122 | .0135 | .0118 |
| A1B1C1 | .0232 | .0161 | **.0107** |

## Learning curves and online regret

**Greedy regret, context average, by episode:**

| Episodes | 0 | 256 | 512 | 1,024 | 2,048 | 3,072 | 4,096 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MB | .338 | .230 | .162 | .070 | .0156 | .0139 | .0122 |
| MB-count | .338 | .176 | .146 | .060 | .0114 | .0103 | .0096 |

**Online regret including exploration, per 512-episode block:**

| Arm | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | Block 6 | Block 7 | Block 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MB | .267 | .147 | .048 | .025 | .022 | .022 | .021 | .021 |
| MB-count | .242 | .132 | .043 | .022 | .020 | .021 | .020 | .020 |

- **MB-count is at or below MB in every block, including the first.** Here,
  directed exploration had no net short-term cost. Trying neglected actions
  paid off almost immediately, and kept paying.
- **Fewer failure-model errors.** Hazard NLL is .574 for MB-count against .618
  for MB, consistent with better-covered data.
- **The bonus changes 21% of actions overall.** The rate is 20% in the first
  block, dips to 4% in the second, then *rises* to 29% late. Once values
  converge, the fading bonus mainly breaks near-ties between almost-equivalent
  actions. The online regret shows no penalty from this. The effect is noted
  for D2, where it may interact with switching.
- **Cost.** MB-count adds essentially nothing: 223 s per fit against 222 s for
  MB. MB-prior costs 1.75× (389 s).

## Reading

1. **The D1 failure was an exploration trap, and directed exploration removes
   it.**
   - The count bonus c / √(1 + n_a) fades as each action is tried.
   - It eliminated the trap in all 24 seeds; MB's trap rate was .46.
   - It gave competence in every context in 12/12 seeds.
   - It lowered cumulative regret in 12/12 seeds.
2. **Randomised priors are not enough here.** Keeping members diverse helped
   on average, but did not stop the ensemble reaching a shared wrong belief.
   It was trapped on 6 seeds where MB was not.
3. **For continual learning,** a learner that chooses its own data needs an
   explicit drive to revisit what it has neglected. This is a plasticity
   mechanism, not a memory mechanism. D2's failure taxonomy will track
   self-selected-data traps separately from forgetting.

## Caveats

- **R1 and R2 are not independent.** They share the first 12 A1B1C1 seeds, as
  designed in Amendment 1.
- **The count bonus is state-independent.** It counts actions within the
  reservoir. Under D2's switching contexts, a count bonus that ignores context
  may over- or under-explore after a switch. That becomes a D2 design question.
- **Stationary contexts only.** Competence is established for stationary
  contexts at N = 4,096 with a 1,024-episode reservoir.
- **Protocol history.** The trap definition was calibrated on D1 test data
  before any D1b data, and the remedy class was changed after a post hoc
  diagnostic. Both are disclosed in the protocol.
