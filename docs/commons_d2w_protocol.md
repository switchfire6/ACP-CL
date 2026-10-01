# Step D2w: stored codes vs amortised inference when per-episode evidence is weak

**Status.** Prospective protocol, written 2026-10-01 before any D2w fitting.
It is the **final decisive test** of the code-library mechanism, as the user
chose after D2.

**Background.** D2's development found a calibrated negative
([report](../reports/commons_d2/dev/summary.md)). With dense, precise
consequences, the context is readable from one episode. A learner reading
only the last episode beat stored, evidence-selected codes, and even
oracle-indexed codes.

**The remaining claim region is weak per-episode evidence.** There,
identifying a context requires integrating evidence over many episodes.

## Question

When each episode carries little evidence about the context, does a stored
code library beat the strongest amortised learners? The library selects
codes with a sticky filter that integrates likelihood evidence across
episodes. The amortised learners include one built specifically to integrate
evidence over long horizons. We compare recovery when a context recurs,
including the rare one, and steady-state performance within runs.

**Pre-declared consequence.**

- If the library loses here too, the mechanism is retired.
- If it wins, the result is a regime-specific positive: stored codes pay when
  evidence is weak. That warrants a CPU pilot on a standard benchmark, then
  cloud compute.

## Environment

- **World.** The D0b physics and actions. Two changes:
  - **Factor C is fixed at C0.** C exposed a core-competence limit (D2), which
    would confound this test.
  - **Contexts are the four A × B combinations.** A1B1 is the minority, at
    about 6%, with at least 4 recurrences per seed and varied absences. There
    are no held-out combinations; composition was settled in D2.
- **The evidence dial.** Supply-sensor noise σ_s and probe-sensor noise σ_p
  are raised, or the probes are switched off. They are set in **stage 0** so
  that:
  - for each A or B contrast, the ideal observer needs **20–60 episodes** to
    reach a .95 posterior (weak evidence);
  - the decision relevance of A and B is unchanged (relevance depends on
    policy, not on sensors);
  - the 99th-percentile per-episode log-likelihood ratio stays below 3 nats,
    so no single episode is decisive.

  The dial is chosen from a grid by the rule "the smallest noise meeting all
  three conditions". There is no learner involvement.
- **Stream.** 20,000 episodes. The run-length distribution is rescaled so
  that runs last at least 3× the identification time, and the median run is
  about 6×.
- **References.** O, I-myopic and L are recomputed for the new
  observation model. They are verified as in D0 and D0b, including against
  brute-force likelihoods at the new noise levels.

## Learners

Every learner uses the D2 core and context conditioning unchanged, with
equal memory bytes. The FC code-drift weight decay is applied to all code
arms, and also, in its analogue form, to the encoder outputs of the
amortised arms.

| Arm | Definition |
|---|---|
| G | No context. |
| AM-k | GRU over the last k episodes' summaries, k ∈ {1, 8, 32}. |
| **AM-ewma (strong evidence integrator)** | An MLP encoder applied to an exponentially weighted running average of per-episode summary features, with decay ρ ∈ {.9, .97}. The running average is carried across the whole stream, so evidence accumulates over many episodes in fixed memory. Its encoder trains on stored (summary, running-average) pairs, which are charged to its memory budget. |
| **FC-A/B** | D2's factored codes, restricted to two slots (supply and probe), with the "lrt" spawn rule. The rule is now **declared**: the default code is fitted to the current episode. The filter is the sticky filter with h tuned. |
| OC-A/B *(privileged)* | Oracle codes, indexed by true A and B lagged one episode. |

- **AM\*** is the best amortised arm (all AM-k and AM-ewma settings) by
  development whole-stream excess.
- **FC\*** is the best FC setting.

## Development and gates

**Seeds.** 23101–23103.

**Gates:**

1. **OC competence.** OC's probe excess over L on recurrences must be ≤ .015
   from episode 10 of a run onward. A and B are the factors this core handles
   well; D2 showed .010–.016 on C0 recurrences.
2. **Weak-evidence check.** L's own recovery excess over O in the first 10
   episodes of a run must be ≥ .005. This confirms the evidence really is weak
   for the ideal agent too.

If gate 1 fails, the bug is fixed. If gate 2 fails, stage 0 is re-calibrated
to more noise.

**Knobs:**

| Arm | Knobs |
|---|---|
| AM-k | k ∈ {1, 8, 32} × encoder lr ∈ {.001, .003} |
| AM-ewma | ρ ∈ {.9, .97} × encoder lr ∈ {.001, .003} |
| FC | τ ∈ {5, 15} × filter hazard h ∈ {.01, .05}, code lr .03 |

Selection is by the lowest development whole-stream excess over L.

**Pathology stop for FC.** The rules are D2's, applied per slot:

- more than 3× as many code changes as true factor switches;
- more than 6 codes in a slot;
- a mean purity below .6.

If any of these fire for FC\*, the verdict is **RETIRE MECHANISM**. There is
no revision round.

**Power rule.** If the development effect sizes imply P(pass) < .8 at 12
seeds but ≥ .8 at 24, use 24 seeds. If P(pass) < .2 at both, the study stops
at development with a calibrated negative, as D2 did.

## Outcomes

All outcomes are probe excess over L at fixed decision states (32 per
context), plus online regret.

| Outcome | Definition |
|---|---|
| **R-rec** | Mean probe excess at episodes {5, 10, 20, 40} after the start of each **recurrence** run, all contexts |
| **R-min** | The same, for the minority A1B1 only |
| **P-steady** | Mean probe excess from episode 40 to the end of each run |
| **S** | Whole-stream online excess over L |

## Criteria (test seeds 23201–23212, or –23224)

A seed passes when FC\* − AM\* ≤ −max(.25 · AM\*, δ), with δ the development
per-seed SD. A criterion passes with that condition in ≥ 10/12 seeds (or
≥ 20/24) and on the seed mean.

| # | Check | Rule |
|---|---|---|
| V1 | Indexing headroom | G − OC ≥ .005 on R-rec, in ≥ 10/12 seeds |
| V2 | AM\* is strong | (G − AM\*) ≥ .5 × (G − OC) on R-rec, seed mean |
| C1 | FC is not worse overall | S(FC\*) ≤ S(AM\*) + .002 |
| C2 | Better recovery | R-rec, per-seed condition |
| C3 | Better rare recovery | R-min, per-seed condition |

## Decision rule

| Outcome | Verdict | Next step |
|---|---|---|
| V1, V2, C1 pass, and C2 **or** C3 passes | **REGIME POSITIVE** | Stored codes beat amortised inference when evidence is weak. Write it up as a regime-specific result. Next: a CPU pilot on a standard benchmark with weak or noisy feedback; **cloud compute only after that pilot.** |
| V1 and V2 pass, but C2 and C3 both fail, or C1 fails | **RETIRE MECHANISM** | Amortised inference matches or beats stored codes even when evidence is weak. Retire the code-library mechanism. The findings write-up stands, plus this result. |
| V1 or V2 fails | **UNINFORMATIVE** | Report it and retire the mechanism. A fair test is not achievable in this world family. |

This is the final round. Whatever the verdict, no further code-library
revisions are made in this world family.

## Engineering and integrity

D2's machinery, unchanged:

- smoke checks, including legal observations, byte parity, AM-ewma training
  only on its stored pairs, and new-noise likelihood verification;
- development;
- lock;
- test;
- a replicate pass on 3 seeds × {FC\*, AM\*};
- hash and CRC verification;
- at most 6 workers;
- new files only;
- every failure kept.

**Outputs.**

- `configs/commons_d2w{,_smoke,_dev}.json`.
- `scripts/commons_d2w.py`.
- `runs/commons_d2w*/`.
- `reports/commons_d2w/`.
