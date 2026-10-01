# D2 development: a calibrated negative for the code-library mechanism

This is the declared development phase of
[docs/commons_d2_protocol.md](../../../docs/commons_d2_protocol.md): 3 seeds
(22101–22103), 20,000-episode recurring streams.

**Run.** 233 tasks, with **0 failures**. A CRC check passed on 175 archives.

**Smoke checks.** All pass:

- legal observations;
- the wiped reset;
- byte parity;
- AM trains on stored summaries only;
- FC's filter matches brute force;
- determinism and resumption;
- reference lookups;
- the replicate pass.

The existing test suite passes (1,672 tests).

**The test was not run.** See "Decision" below.

**Files.** `dev_results.json`, `gates.json`, `dev_fits.csv`, and the
diagnostics in `runs/commons_d2_scratch/oc_bug_vs_limit.json`.

## Gates

| Gate | Result |
|---|---|
| Core parity (G against D1b MB-count) | **Pass**: .00968 vs .00957 |
| OC competence (held-out first encounters, excess over L ≤ .010) | **Fail**: .061–.063 at positions 5–40 |
| OC beats G (V1-like) | Pass: 3/3 seeds |

## Development results (means over 3 seeds; lower is better)

| Arm (selected knobs) | S, whole stream | R-min, rare recurrence | R-held, new combinations |
|---|---:|---:|---:|
| G (no context) | .0689 | .0999 | .1041 |
| W (wiped; descriptive) | .2302 | .4727 | .3538 |
| OC (oracle factored codes; privileged) | .0494 | .0412 | .0616 |
| **AM\*** = AM, k = 1 (reads only the last episode) | **.0509** | **.0325** | **.0382** |
| AM-factored, k = 1 | .0739 | .0530 | .0598 |
| **FC**, τ15, code lr .03 (the hypothesis) | .0538 | .0518 | .0524 |
| FC-single, τ15, code lr .01 | .0519 | .0440 | .0573 |
| FC-K1 | .0683 | .1143 | .1026 |

**History length.** Longer AM histories were much worse: k = 8 gave S .069,
and k = 32 gave S .123. The memory budget forces a smaller reservoir as k
grows.

**Dev-level readings** (descriptive):

- V1 passes in 3/3 seeds.
- V2 passes, but only because AM\* is better than the oracle.
- C1 fails: FC .0538 against AM\* .0509 + .002.
- C2 and C3 pass in 0/3 seeds.
- **The power** of C2 and C3 at the development effect sizes is about zero
  for any seed count.
- **The compounding flag does not trigger.** FC's visit slope is not more
  negative than AM\*'s in any seed.

## Diagnosis: a real limitation, not a bug

The implementer ran read-only diagnostics on the final checkpoints.

- **Indexing works.** Swapping OC's A code multiplies regret 5–10× (A0B0C0:
  .012 → .146), and swapping the B code hurts where B matters. The lagged
  labels are verified.
- **Context is cheap to read here.** Dense supply and probe sensors reveal A
  and B within one episode. AM with k = 1, which only reads the previous
  episode's summary, matches OC on recurrences (.026–.030 both). It beats OC
  on held-out first encounters (.049 vs .081, .022 vs .035, .039 vs .060).
  **There is no retrieval headroom** for a stored code library.
- **The real bottleneck is core competence.**
  - **C is barely used.** C needs a three-way interaction (C code × cue × the
    2↔3 action), with evidence only from the agent's own 2↔3 transfers. In
    the minority context, swapping C *improves* regret (.062 → .052).
  - **The minority is under-trained:** OC's A1B1C1 regret is .062, against
    .011 in a single context.
  - **Composition fails for combinations containing C.** OC on A0B1C1 scores
    .081, which is about G's .093. A1B1C0 composes well: .060 against .151.
  - **The ≤ .010 gate on held-out first encounters is below what this core
    reaches** even in stationary single contexts (D1b mean .0096, worst
    context .014).
- **The probe excess is flat from position 2 to 40 for every arm.** The
  shortfall is model quality, not identification lag.

## Implementation fixes and disclosure

| Fix | What changed | Who it could favour |
|---|---|---|
| **NaN in JSON** | An analysis-only fix | Nobody |
| **Code drift** | OC's codes had drifted to magnitudes of 15–55. A decoupled weight decay of 1.0 was added to all code arms. | It was informed by OC data on seed 22101 only, and was applied **only to the code arms** (FC, FC-single, FC-K1, OC). So it could only have *helped* FC, which still lost. |
| **The spawn rule "lrt"** | The literal rule can never spawn, because with one code the default *is* that code; literal FC is bit-identical to FC-K1. In "lrt", the default is first fitted to the current episode, and a spawned code starts at that fit. | Designed before any development fit. It is an unapproved deviation; FC with "lrt" still lost. |
| **OC's code learning rate** | Set to .03 before any data, and never tuned | Nobody |

## Decision: the test is not run

- **No headroom.** The selected competitor (reads only the last episode) is
  at least as good as the privileged oracle-code ceiling. So FC has no
  headroom over it in this world.
- **No power.** C2 and C3 have about zero power at any seed count.
- **Weighing the cost.** Running the ~11 h test would only formalise a
  certain outcome. The development phase exists to catch exactly this, and
  the protocol's own decision rule, applied to the development data, gives
  **MECHANISM NO-GO**.
- **What this does not claim.** We do **not** claim a locked test verdict.
  This is a development-level, calibrated negative, with the diagnosis above.

The one allowed protocol revision was **not** used. A revision on this world
could improve the core's handling of C, but it cannot create retrieval
headroom. Context in this world is readable from one episode. D0b already
showed that low separability with high relevance is unreachable here.

## What this means

1. **In this world, amortised in-context inference wins.** When recent dense
   consequences reveal the context, "read the last episode" beats a stored,
   evidence-selected code library, and even beats oracle-indexed codes on
   composition.
   - This matches an external report: Asawa et al. found that vanilla
     in-context learning beat elaborate memory systems.
   - It also matches Step C: a global EMA beat snapshot recall.
2. **Where stored, retrievable codes could still win** is regimes where recent
   evidence is weak: noisy or sparse consequences, where identifying a
   context needs integration over many episodes. That regime was not
   reachable in this world (D0b). Whether it matters in practice is an open
   question.
3. **What the project has established** across Steps A–D:
   - **Methodology:**
     - regret calibrated against an ideal agent with the same information;
     - decomposing the gap into estimation, perception and optimisation;
     - gain against wiped and headroom controls;
     - a failure taxonomy.
   - **Findings:**
     - online optimisation, not information, dominated the original gap;
     - weight averaging and full replay are the competence levers;
     - whole-network recall discards shared structure;
     - agents choosing their own data fall into **exploration traps**, which
       a fading count bonus removes, with payback in every block;
     - amortised inference beats stored codes when context is cheap to read.
