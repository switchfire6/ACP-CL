# Step D1: learner competence in stationary commons contexts

- **Protocol:** [docs/commons_d1_protocol.md](../../docs/commons_d1_protocol.md),
  amended before the lock to thin the probe schedule.
- **Development:** [dev/summary.md](dev/summary.md).
- **Scale:** locked; 12 test seeds (21101–21112) × 5 contexts × 4 arms, which is
  240 fits.
- **Integrity:**
  - zero task failures;
  - the replicate pass (60 fits) was **bit-identical**;
  - all 240 fits re-verified against their hashes;
  - all 1,086 archives passed the CRC scan.
- **Files:** `results.json`, `fits.csv`, `curves.csv`, `per_context.csv`,
  `per_seed.csv`, `regret_curves.png`.

## Verdicts (pre-declared)

**R1, competence.** An arm passes a seed if its context average is ≤ .015
**and** its worst context is ≤ .025. It needs to pass in ≥ 10/12 seeds.

| Arm | Seeds with average ≤ .015 | Seeds with worst ≤ .025 | Seeds passing both | Context average [95% CI] | Mean worst context |
|---|---:|---:|---:|---|---:|
| MF | 0 | 0 | 0/12 | .0547 [.0524, .0571] | .077 |
| **MB** | **10** | 6 | **6/12** | .0109 [.0089, .0131] | .022 |
| MF-unbounded *(privileged)* | 0 | 0 | 0/12 | .0428 [.0399, .0458] | .061 |
| MB-unbounded *(privileged)* | 11 | 8 | 8/12 | .0065 [.0046, .0088] | .014 |

**No arm is competent.** Following the declared table, this outcome means
"the learner family is the bottleneck" and leads to a D1b round (see below).
Every failing worst context is **A1B1C1**, at .028–.036.

**R2, cost of the memory budget.** It costs if bounded minus unbounded is
≥ .005 in ≥ 10/12 seeds.

- **MF:** costs. The difference is +.0119 [.0098, .0141], in 12/12 seeds.
- **MB:** does not cost. The difference is +.0044 [.0034, .0056]. It is ≥ .005
  in only 4/12 seeds, though positive in all 12.

## Per context: regret at 4,096 episodes (mean over 12 seeds)

| Context | MB | MB-unbounded | MF | MF-unbounded | O − none |
|---|---:|---:|---:|---:|---:|
| A0B0C0 | .0069 | .0048 | .048 | .037 | .155 |
| A1B0C0 | .0070 | .0036 | .042 | .035 | .076 |
| A0B1C0 | .0068 | .0045 | .048 | .038 | .146 |
| A0B0C1 | .0124 | .0060 | .069 | .053 | .188 |
| A1B1C1 | **.0212** (range .006–.036) | .0133 | .067 | .051 | .260 |

## Learning curves (context average)

| Episodes | 0 | 256 | 512 | 1,024 | 2,048 | 3,072 | 4,096 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MB | .361 | .221 | .162 | .071 | .0138 | .0113 | .0109 |
| MB-unbounded | .361 | .221 | .162 | .071 | .0113 | .0079 | .0065 |
| MF | .347 | .171 | .115 | .077 | .060 | .056 | .055 |
| MF-unbounded | .347 | .171 | .115 | .077 | .055 | .047 | .043 |

## Secondary results

| Measure | MB | MB-unbounded | MF | MF-unbounded |
|---|---:|---:|---:|---:|
| Cumulative online regret | 276 | 262 | 411 | 387 |
| Greedy action equals O's | 67% | 72% | 38% | 43% |
| Per-entity survival log-loss | .158 | .149 | .297 | .264 |

- **MB is better than MF** by .044 at 4,096 episodes, in 12/12 seeds. This is
  expected under dense feedback and was declared not a headline.
- **MB overfits its bounded reservoir.**
  - Hazard NLL is .585 bounded against .446 unbounded.
  - Optimism about its own greedy action is +.019 against +.008.
- **MF is pessimistic about actions it has not tried.** Its lowest reliability
  bin predicts .008 where O's value is .115.
- **MB's supply model converges in both arms.** So MB's residual regret comes
  from its failure model and from its data.
- **Dev predicted test well.** MB's test result (.0109) matches the development
  estimate (.0107); the winner's-curse bias turned out to be small.

## Diagnostic: an exploration trap (post hoc, descriptive)

The lead ran this after the verdict. It uses only saved traces and O's panel
tables.

In A1B1C1, O's best action on the panel is **action 9 (2→3, 6 units)** in
about 8% of states, with a similar share whether cue bit 1 is 0 or 1. How
often each MB fit tried action 9 during training, and how often its greedy
policy chose it at 4,096, lines up with the failures:

| MB fits in A1B1C1 | Seeds | Action 9 in training | Action 9 chosen on the panel (of 512) | Regret |
|---|---|---:|---:|---:|
| Failing, both bounded and unbounded | 21103, 21110, 21111, 21112 | ≈ 1% | 0–14 | .029–.036 |
| Passing | 21104–21109 | 7–11% | 45–72 | .006–.014 |
| Failing, bounded only | 21101, 21102 | 4–8% | 51–57 | .028–.029 |

The unbounded versions of the bounded-only failures pass at .004 and .002.

- **The regret does not depend on the cue.** It is the same for cue bit 1 = 0
  and 1, so the failure is not about learning the cue-gated delay.
- **Four seeds fall into a self-reinforcing trap.** All ensemble members learn
  early that action 9 is poor. Thompson sampling then almost never tries it,
  so the model never gets the data that would correct it. Unlimited memory
  does not help, because the data were never collected.
- **Two further seeds fail only under the memory budget.** They tried the
  action, but their bounded memory overfit.

**Why this matters beyond D1.** When a learner chooses its own data, its own
beliefs can lock it out of knowledge. This is a **plasticity failure caused by
self-selected experience**, not by forgetting. It goes into D2's failure
taxonomy.

## Next step and a disclosed deviation

The declared consequence of "no arm competent" is a D1b round trying
"architecture remedies (width, ensemble, horizon auxiliaries)".

The diagnostic shows the dominant failure is **exploration**. Those remedies
do not target it. The D1b protocol therefore tests exploration remedies,
alongside one architecture remedy. It is a new prospective protocol with fresh
development and test seeds. The change of remedy class is disclosed here, and
it rests on a post hoc diagnostic.

See [docs/commons_d1b_protocol.md](../../docs/commons_d1b_protocol.md).
