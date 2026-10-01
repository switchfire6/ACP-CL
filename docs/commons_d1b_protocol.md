# Step D1b: escaping exploration traps (competence remedies)

**Status.** Prospective protocol, written 2026-09-30 before any D1b fitting.

**What it follows.** D1 found no competent arm
([report](../reports/commons_d1/summary.md)). A post hoc diagnostic located
the dominant failure: an **exploration trap**. In 4 of 12 seeds, MB's ensemble
agreed early that the key action was poor, Thompson sampling then almost never
tried it, and even unlimited memory could not fix it. Two further seeds failed
only because the bounded reservoir overfit.

**Disclosed deviation.** D1's declared remedy class was architecture (width,
ensemble, horizon auxiliaries). This protocol tests **exploration remedies**
plus one architecture remedy, because of that diagnostic. Everything is fresh:
new development and test seeds.

## Question

Does an exploration remedy make the model-based learner competent in every
training context? And does the extra exploration's early cost **pay for
itself** within the budget?

The payback question applies the user's "short-term cost for long-term gain"
criterion to learning itself.

## Fixed elements (identical to D1 unless stated)

- **World.** The D0b world. The contexts are the 5 training combinations, each
  stationary.
- **Budget.** N = 4,096 episodes per fit.
- **Memory.** A bounded reservoir of 1,024 episodes.
- **Probes.** Greedy probes at episodes {0, 256, 512, 1,024, 2,048, 3,072,
  4,096}, on D1's panels.
- **Oracle.** O's tables are computed as in D1.
- **Base learner.** D1's MB with its development-selected knobs: lr .001, U2,
  EMA d .99, an ensemble of 5, and M = 256 planning draws.
- **Seeds.**
  - Development: 21201–21203, × 5 contexts.
  - Test: 21301–21312, × 5 contexts.

## Arms

All arms are MB variants. Only the exploration or architecture element
changes.

| Arm | Change | Development knob |
|---|---|---|
| MB | None. This replicates D1 on fresh seeds. | — |
| MB-prior | Randomised prior functions: each failure-model member adds a fixed, untrained random network, scaled by β, to its output logits. This keeps the members diverse where data are absent. | β ∈ {1, 3} |
| MB-UCB | Act on the ensemble mean + κ × the ensemble SD of each action's predicted joint survival (optimism), instead of Thompson sampling. | κ ∈ {1, 2} |
| MB-ε | With probability ε, take a uniformly random action; otherwise Thompson sampling. | ε ∈ {.03, .10} |
| MB-wide | Architecture remedy from D1's declared list: failure-model width 256 instead of 128. | — |

- **Greedy probes** use the ensemble mean for every arm, including MB-prior,
  whose priors are part of each member's function.
- **Training** is unchanged in every arm.

## Development

- **Grid.** 2 settings × 3 remedies, plus MB and MB-wide, which have no knob.
- **Seeds.** 3 development seeds × 5 contexts.
- **Selection rule, for each remedy's knob.** The lowest mean over development
  cells of **cumulative online regret** at 4,096 episodes, among settings whose
  mean greedy regret is within one paired SE of the best greedy regret for that
  remedy. This picks the setting that explores most efficiently without giving
  up endpoint competence.
- **Pathology note.** If any remedy's greedy regret is worse than MB's in
  development, that is reported. The remedy is still carried to the test
  unchanged.

## Pre-declared readings (12 test seeds × 5 contexts)

**R1, competence (unchanged from D1).** An arm is competent if its context
average is ≤ .015 **and** its worst context is ≤ .025, both in ≥ 10/12 seeds.

**R2, trap escape.** Did the remedy fix the specific failure?

- In A1B1C1, count the seeds whose regret is ≤ .025.
- A remedy **escapes the trap** if it does so in ≥ 10/12 seeds, **and** its
  training frequency of O's most-often-best non-null actions is within a
  factor of 2 of O's own usage.

**R3, payback.** A competent remedy **pays for itself** if its cumulative
online regret over 4,096 episodes is ≤ MB's, in ≥ 10/12 seeds.

- Paying for itself means the early exploration cost is recovered by better
  later decisions within the budget.
- If a competent remedy does not pay back, the net cost is reported.

**How D2's core is chosen.** From the competent arms, choose the one with the
lowest mean cumulative online regret. If none is competent, see the table
below.

**Descriptive:**

- learning curves;
- per-context tables;
- action-usage profiles against O's;
- ensemble disagreement over time;
- the hazard NLL;
- value optimism;
- costs.

## What each outcome implies (declared now)

| Outcome | Implication |
|---|---|
| At least one competent remedy | D2 uses it as the core, with its exploration scheme applied to every D2 arm, so learners are compared fairly. |
| A remedy escapes the trap (R2) but misses R1 through bounded-memory overfitting | A D1c development round tests replay regularisation (fewer epochs over the reservoir, or weight decay). It stops there, without adding more remedies. |
| No remedy escapes the trap | Exploration in this world needs a structural fix: the ideal agent's value-of-information signal, or count-based bonuses. Report, and decide with the user before building further. |

## Amendment 1, before the lock (2026-09-30, after development 1, before any test data)

The development results are in [reports/commons_d1b/dev/](../reports/commons_d1b/dev/).
Four problems emerged:

- **The development baseline was never trapped.** MB passed A1B1C1 in 3/3
  seeds, so development could not show whether any remedy escapes the trap.
- **Several set-ups were trapped where MB was not.** Remedy set-ups were
  trapped in seeds where MB was not; only MB-prior β1 was never worse. The trap
  depends on the trajectory, and it is common.
- **Two remedies cannot work by construction.**
  - MB-ε adds at most ε/13 to any one action's use. That is below the usage
    floor for the key action.
  - MB-UCB's optimism never switches off, because ensemble disagreement
    plateaus at about .04. Its late regret was 3.3× MB's.
- **MB-wide showed no advantage** and costs 2.2× the compute.

**Changes:**

1. **Arms carried to the test:** MB, MB-prior (β selected below) and a new
   **MB-count** arm. MB-ε, MB-UCB and MB-wide are dropped for the reasons
   above. Their development results remain reported.
2. **The MB-count arm (directed exploration).** It acts on the argmax over
   actions of the Thompson value plus c / √(1 + n_a), where n_a counts action a
   among the episodes currently in the reservoir. The bonus fades as an action
   is tried. The development knob is c ∈ {.05, .15}.
3. **Development 2, "trap rescue".**
   - Context A1B1C1 only.
   - The seeds are **D1's test seeds**, already analysed and so usable for
     development: known-trapped {21103, 21110, 21111, 21112} and known-clean
     {21104, 21105, 21106, 21107}.
   - Every arm shares MB's randomness keys, so on the trapped seeds MB's
     trajectory reproduces D1 exactly. Whether a remedy rescues the trap is
     directly visible, and the clean seeds show whether it creates new traps.
   - **Grid:** MB-prior β ∈ {1, 3} and MB-count c ∈ {.05, .15}.
   - **Selection per remedy:** the most rescued trapped seeds at A1B1C1
     regret ≤ .025, minus any newly trapped clean seeds. Ties go to lower
     cumulative online regret.
4. **Trap definition, ratified.**
   - **Critical actions:** those whose removal costs O ≥ .015 on the A1B1C1
     panel, i.e. {6, 9, 12}.
   - **Usage condition, one-sided:** training usage of each critical action is
     ≥ 0.5 × f_O(a), O's panel best-frequency for that action. The earlier
     "within a factor of 2" was ambiguous. Over-using a critical action is not
     a trap, so only the lower bound is kept.
5. **Test design, replacing the test section above.**
   - **R2, the trap rate (primary).**
     - Context A1B1C1, **24 fresh seeds (21301–21324)**, arms MB, MB-prior
       and MB-count.
     - An arm escapes the trap if A1B1C1 regret is ≤ .025 **and** the usage
       condition holds in **≥ 20/24 seeds**.
     - The paired comparison of each arm's trap rate with MB's is reported.
   - **R1, competence.**
     - The other 4 contexts on seeds 21301–21312, for the same 3 arms.
     - Combined with the first 12 A1B1C1 seeds, R1 is exactly as declared.
   - **R3, payback.** As declared, on the seeds used for R1.
6. **Probes.** Greedy probes are unchanged. The ensemble-disagreement probe
   runs only at episodes 1,024 and 4,096, to save cost.

## Engineering and integrity

This is D1's machinery unchanged:

- a smoke check, including that each remedy is active only where declared and
  that the MB arm is bit-identical to D1's code path;
- development;
- lock;
- test;
- a replicate pass on 3 seeds;
- full hash and CRC verification;
- at most 6 workers;
- new files only;
- every failure kept.

## Outputs

- `configs/commons_d1b{,_smoke,_dev}.json`.
- `scripts/commons_d1b.py`, which may import `scripts/commons_d1.py`.
- `runs/commons_d1b*/`.
- `reports/commons_d1b/`.
