# D1 development phase: summary

This is the declared development phase of
[docs/commons_d1_protocol.md](../../../docs/commons_d1_protocol.md).

**Scale.** 2 seeds (21001–21002) × 5 contexts × 8 configurations per family.
There were 180 fits and 85 oracle tasks. No task failed, and a CRC check
passed on 446 archives.

**Smoke checks.** All pass:

- legal observations only;
- the reservoir is uniform;
- common random numbers are shared across arms;
- O's tables recompute exactly;
- EMA does not interfere with training;
- determinism and resumption;
- a replicate pass on 20 fits.

**Files.** `dev_results.json`, `dev_fits.csv`, `dev_curves.csv` and
`dev_summary.csv`.

## Regret at 4,096 episodes

Each value is the mean over 10 seed × context cells, of greedy-probe regret
against O on a fixed 512-state panel.

| Configuration | MF | MB |
|---|---:|---:|
| lr 1e-3, U2, d .99 | .0550 | **.0107** (selected) |
| lr 1e-3, U2, d .998 | .0545 | .0155 |
| lr 1e-3, U8, d .99 | .0536 | .0201 |
| lr 1e-3, U8, d .998 | .0602 | .0199 |
| lr 3e-3, U2, d .99 | .0554 | .0194 |
| lr 3e-3, U2, d .998 | **.0484** (selected) | .0243 |
| lr 3e-3, U8, d .99 | .0514 | .0211 |
| lr 3e-3, U8, d .998 | .0544 | .0264 |

## Per context, at the selected configurations

| Arm | A0B0C0 | A1B0C0 | A0B1C0 | A0B0C1 | A1B1C1 |
|---|---:|---:|---:|---:|---:|
| MB | .0065 | .0085 | .0063 | .0128 | .0194 (seeds .0306 / .0083) |
| MB, unbounded replay | .0042 | .0033 | .0043 | .0100 | .0068 |
| MF | .033 | .044 | .037 | .060 | .068 |
| MF, unbounded replay | .030 | .036 | .041 | .046 | .052 |

## Learning curves

Mean regret at each checkpoint:

| Arm | 256 | 1,024 | 2,048 | 4,096 |
|---|---:|---:|---:|---:|
| MB | .205 | .064 | .012 | .0107 |
| MB, unbounded replay | .205 | .064 | .0097 | .0057 |
| MF | .165 | .072 | .058 | .048 |

## Reading

This is descriptive only, from 2 seeds.

- **MB is close to the bar.**
  - Its context average meets .015 in both seeds.
  - Its worst context fails .025 in one seed (A1B1C1, .031).
  - With unbounded replay it meets the bar in both seeds.
- **MF is about 3× the bar.** It learns 13 survival functions from 1-bit
  outcomes of its own actions.
- **Planning is not what limits MB.**
  - Its supply model converges.
  - With the true model, planning with M = 256 samples loses only .0003.
  - So MB's regret comes from its failure model.
  - Its value estimates are optimistic by about +.014.
- **Bounded MB overfits its reservoir.** Hazard negative log-likelihood rises
  with the amount of training, from .60 to 1.09 nats, against .45 with
  unbounded replay.

The caveats and the probe-schedule amendment are recorded in the protocol
before the lock.
