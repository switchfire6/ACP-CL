# Step D0: commons world and ideal references

This stage covers engineering and calibration only. **No learner was
trained**, apart from the declared tabular easiness check. The design is
[docs/step_d_design.md](../../docs/step_d_design.md).

**Files.**

- Code: `src/acp_cl/commons/`, which contains `world.py`, `references.py`,
  `calibration.py` and `verification.py`.
- Script: `scripts/commons_d0.py`.
- Tests: `tests/test_commons_world.py` (9 pass).
- Results in this folder: `world_spec.json`, `verification.json`,
  `calibration.json`, `calibration.png`, `calibration_grid.csv`,
  `calibration_final_pairs.csv`, `voi.json`, `schedule.json`,
  `easiness.json`, `easiness.png` and `timing.json`.
- Raw arrays, with a sha256 sidecar for each: `runs/commons_d0/`.

## World (full specification in `world_spec.json`)

**Entities and episodes.** Three entities. Reserves are drawn from U(1,13) at
each start, and episodes are independent. Capacity is 18, maintenance is 2
per step, and each episode lasts 12 steps.

**Supply.** s ~ N(1.9 + .8 · season · sign, .35²), unclipped. Negative supply
occurs in 1% of episodes and changes joint survival by at most 5e-5.

**Channels.** Reverse transfers use the same pipe as the forward direction.

| Channel | Efficiency | Delay |
|---|---|---|
| 1→2 | .9 − b_gap · B | 1 |
| 2→3 | .9 | Fast (0) only if C = 1 and cue bit 1 = 1; otherwise Δ |
| 3→1 | .8 | 2 |

**Actions.** 7 actions: no transfer, or 3 units along one channel in either
direction.

**Objective.** Joint survival: all three entities alive at step 12.

**Probes.** External flows of .3 units that never enter a reserve.

**Observations.**

- At decision time: exact reserves plus 3 cue bits.
- After the episode: sensors (σ = .1) of supply and probe arrivals, plus each
  entity's failure step. The agent's own transfer is not sensed.

**Proposed dials.**

- Seasonal flip amplitude A = 1.0.
- b_gap = .6.
- Δ = 6.
- **Steady probes**: the probe flow is already running at the episode start.

**Survival at these dials** (stream seed 1):

| Policy | Joint survival |
|---|---|
| No transfer | .498 |
| O | .696 |
| Random | .408 |

## Verification: all pass

- **Simulator against an independent slow reference.** 6,000 cases, including
  extreme reserves, forced negative supply, fractional delays and arrivals
  that never come. There are 0 mismatches. The conservation error is 4e-14.
- **O's Monte Carlo** (2^14 draws, common random numbers across actions and
  contexts, independent draws for selection and scoring):
  - It is unbiased: the RMS z-score is .76.
  - Its spread is .91 of the stated SE.
  - The optimism from picking the best action is −.0016.
  - Its stream SE is .0014.
- **I's likelihood against brute force.** 36 edge cases: clipping, transfers
  limited by the donor's reserve, failure before arrival, and negative supply.
  The check uses an independent sequential importance-sampling estimator that
  shares no code with `world.py`. The maximum |z| is 3.09, and log
  differences are ≤ .07.
  - Naive prior importance sampling was unusable: its effective sample size
    was about 1.7 at 2^21 samples, and it was biased low. This is recorded.
- **Factorisation, filter and determinism.** The factorisation checks give
  |z| ≤ 2.2. The filter matches brute-force enumeration to within 2e-15.
  Determinism holds: repeated calculations match bit for bit, and all 160
  arrays re-verify against their hashes.

## Calibration: separability × decision relevance

**Definitions.**

- **Separability** is how many episodes a pair-restricted ideal agent needs to
  reach a .95 posterior.
- **Relevance** is the cross-context policy regret, in survival.

**Grades:**

| Grade | High | Low |
|---|---|---|
| Separability | ≤ 5 episodes | ≥ 25 episodes |
| Relevance | ≥ .015 | ≤ .005 |

These thresholds were set after an exploratory look at the achievable range.

**Proposed dials, re-estimated on fresh draws:**

| Factor | Separability (episodes to .95) | Relevance | Grade |
|---|---:|---:|---|
| A, seasonality | 1 | ≈ .19 | High / high |
| B, efficiency of channel 1→2 (gap .6) | 1.0 | .0195 ± .0007 (per pair: .002–.031) | High / high |
| C, cue-gated delay (Δ 6, steady probes) | 27.5 (p90 92) | .0234 ± .0009 | Low / high |

**Which cells are reachable.**

- With probes starting at episode start, C can never be both low-separability
  and high-relevance. Any delay that matters is revealed by when the probes
  arrive.
- With steady probes, C is identifiable **only through the agent's own 2↔3
  transfers**.
- No factor reaches low separability together with high relevance while the
  probes reveal it.

**Action-dependent value of information** (steady probes, uniform prior over
8 contexts):

- Total information varies across actions from 1.383 to 1.446 nats per
  episode, a spread of .063.
- All of that spread is C's, and it comes through the 2↔3 transfers.
- With probes starting at episode start, the spread is .0005. At the active
  level it is .103.

## Schedule (3 example seeds)

- **Training and held-out combinations.**
  - Training: A0B0C0, A1B0C0, A0B1C0, A0B0C1, plus the minority A1B1C1
    (5.6% share, with 9–11 recurrences).
  - Held out: A1B1C0, A1B0C1 and A0B1C1. Each takes 2.5%, and they appear
    only after episode 15,000.
- **Run lengths.** P(L) ∝ 1/L on 8–800. Absences last 108–5,886 episodes.
- **Short runs.** 29–33% of runs are no longer than C's identification time.

## Easiness check (20,000 episodes)

| Agent | Mean survival | Cumulative regret vs O |
|---|---:|---:|
| O | .6957 | 0 |
| **I** | .6949 | 17.7 ± .4 (.0009 per episode) |
| Tabular (Thompson, 1,728 cells, privileged context centroids) | .5020 | 3,876 |
| No transfer | .4980 | 3,956 |
| Random | .4077 | 5,761 |

- **Verdict: not too easy.** The tabular agent's regret is 218× I's. This is a
  weak statement, though, because the tabular agent barely beats no transfer.
- **I's identification lag after a switch** (median / mean): 1 / 10.9 episodes
  overall, 5.5 / 13.6 for switches involving only C, and 13 / 12.5 for
  held-out combinations.

## Timing

- **References.** O costs .025 s per decision. I's tables cost .057 s.
- **D2 projection.** About 6 CPU-hours for the references plus about 13 CPU-hours
  for a neural proxy of the arms. That is about 3.3 hours on 6 workers. Real
  learners may be several times slower.

## Lead's assessment

The world and references are sound and verified. The calibration reveals a
**design problem that must be fixed before D1**:

1. **Decision value is concentrated in factor A.**
   - Knowing A is worth about .19 survival. It is also identified in one
     episode, by every agent.
   - B and C are worth only about .02 each, and B's value varies .002–.031
     across pairs.
   - With one 3-unit transfer per episode, a single channel's efficiency or
     delay changes the best action only a little.
   - The pre-registered claim region (rare × long absence × low separability)
     would therefore rest on survival effects of about .02, carried by factor
     C alone. That is too little signal to discriminate learners over seed
     noise.
2. **I's near-optimality is unproven for C.** C can only be learned by acting
   on 2↔3, and D0 did not measure how much a one-step-lookahead agent gains.
3. **The thresholds were chosen after seeing the range**, and C sits just
   inside "low separability" (27.5 episodes against a threshold of 25).

**Decision: a bounded redesign round (D0b) before D1.** Its targets are
declared before any search:

- **Relevance.**
  - B and C: each ≥ .05 survival (cross-context policy regret, averaged over
    pairs), with no pair below .02.
  - A: ≤ 3× the smaller of B and C.
- **Separability.**
  - B: high, ≤ 5 episodes.
  - C: low, ≥ 40 episodes. The margin above 25 guards against a borderline
    result.
- **Allowed changes:**
  - transfer size and action set (for example, 2 amounts);
  - how many channels each factor touches;
  - the seasonal amplitude;
  - entity 3's supply sign;
  - the probe regime.
- **Physics fidelity.** The existing physics and verification code must be
  kept.
- **Myopia check.** Measure a one-step-lookahead agent against myopic I on the
  stream. If it improves on I by more than 10% of I's regret against O, report
  I as a *myopic* reference and add the lookahead agent as the reference.
- **If the targets are unreachable,** report the achievable frontier. The
  world is then accepted with the claim region narrowed accordingly, and not
  iterated further.
