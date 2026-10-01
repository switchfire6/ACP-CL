# Step D0b: bounded redesign of the commons world

This round was declared in
[../commons_d0/summary.md](../commons_d0/summary.md), with its targets fixed
before any search.

**Outcome: the targets are unreachable with the allowed levers.** The best
achievable world was chosen by the declared rule (largest minimum margin
across targets) and accepted. The claim region is narrowed accordingly. No
further world iteration will be done.

**Files.** In this folder: `world_spec.json`, `screen.csv`, `refine.csv`,
`separability.json`, `c_frontier.json`, plus the verification, VOI,
lookahead, easiness, timing and schedule outputs. Code: `scripts/commons_d0b.py`
and `tests/test_commons_d0b.py` (13 tests pass across D0 and D0b). Raw data:
`runs/commons_d0b/`. D0 still reproduces bit for bit.

## Search and frontier

The search screened 864 configurations: a coarse grid of 324, then a
refinement of 576.

- **B and C trade off against each other.** Alone, B reaches .104 and C
  reaches .090. Together, min(B, C) peaks at about .043, and no configuration
  gets both to .045 or more.
- **C cannot be both relevant and hard to identify.** With steady probes, C's
  only evidence is the failure pattern after the agent's own 2↔3 transfers.
  The supply sensors pin each trajectory tightly, so the very episodes that
  make C matter are also near-decisive evidence about it.

| C delay Δ | Episodes to identify C | C relevance |
|---|---:|---:|
| 2 | 40 | .001 |
| 3 | 20 | .004 |
| 4 | 14 | .011 |
| **6 (chosen)** | **10** | **.038** |

Reaching both targets would need noisier consequences, which was outside the
allowed levers. It was not tried.

## Chosen world (full specification in `world_spec.json`)

It is D0 with the following changes:

| Element | D0 | D0b |
|---|---|---|
| Transfer sizes | 3 (7 actions) | **3 or 6 (13 actions)** |
| Base supply signs | (+1, −1, −.5) | (+1, −1, −1) |
| Flip amplitude | 1.0 | **.5**: when A = 1, entities 1 and 2 have no seasonality |
| Factor B | Lowers efficiency of 1→2 | **Swaps the efficient route**: B = 0 gives 1→2 .9 and 3→1 .1; B = 1 gives 1→2 .1 and 3→1 .9 |
| Factor C | Gates the 2↔3 delay (fast 0, slow 6) | Unchanged |

Steady probes are unchanged. Survival is .435 with no transfer and .591 for
O.

## Grading achieved

Measured on fresh draws, over all 12 pairs:

| Factor | Episodes to identify | Relevance (per-pair range) |
|---|---:|---|
| A | 1 | .086 (.044–.122) |
| B | 1 | .038 (.028–.054) |
| C | 10.0 (90th percentile 34) | .038 (.031–.047) |

**Against the targets:**

- **Met:**
  - weakest B or C pair .028 (target ≥ .02);
  - A = 2.26 × min(B, C) (target ≤ 3×);
  - B identified in 1 episode (target ≤ 5).
- **Not met:**
  - B and C relevance, .038 each against .05;
  - C separability, 10 episodes against ≥ 40.

Relative to D0, B and C relevance roughly doubled, and A's dominance fell from
about 10× to 2.3×. C moved from "low" to "mid" separability.

## Verification: all pass at the chosen world

- **Simulator.** Matches the independent slow reference on 12,000 cases,
  including edge cases, with 0 mismatches.
- **O's Monte Carlo.**
  - Its RMS z is .83, and its SE is .88 of the formula value.
  - Common random numbers cut the variance of action differences to .67.
  - The optimism from picking the best action is +.0004.
- **I's likelihood.** Checked against sequential brute force on 36 cases:
  maximum |z| 3.2.
- **Factorisation, filter and determinism** all pass.
- **Integrity.** All 1,009 output files re-verify against their hashes.

## Myopia check: the lookahead agent becomes the reference

A one-step value-of-information lookahead agent reduces cumulative regret
against O from 12.68 (myopic I) to 10.61. That is **16.3% of I's regret**, above
the declared 10% rule.

- The lookahead is therefore the primary same-information reference.
- Myopic I is kept as "I-myopic".
- The absolute stakes are tiny, about .0001 survival per episode, and this was
  measured on one stream.

## Easiness (20,000 episodes)

| Agent | Mean survival | Cumulative regret vs O |
|---|---:|---:|
| O | .5913 | 0 |
| Lookahead | .5908 | 10.6 |
| I-myopic | .5907 | 12.7 |
| No transfer | .4355 | 3,117 |
| Tabular | .3528 | 4,772 |
| Random | .2652 | 6,523 |

**Verdict: not too easy.** The check has low power, though: the tabular agent
is now worse than no transfer. The real easiness evidence will be D1's
competent learners.

## Timing

The projection for D2 is about 6.4 hours on 6 workers, including the
references.

## Narrowed claim region (replaces the design's pre-registered region)

- **Separability.** Low separability together with high relevance is not
  available here. Separability now enters only as a descriptive covariate: A
  and B identify in about 1 episode, C in about 10.
- **What D2 tests.** The advantage of factored codes over the amortised
  in-context learner must grow with the following:
  - **rarity**: the minority combination, about 5–6%;
  - **absence length** under bounded replay;
  - **held-out factor combinations**: composition.

  A claim about "hard-to-identify contexts" would need a future world with
  noisier consequences.
