# D1b development 1: summary

**Scale.** Seeds 21201–21203 × 5 contexts. 120 fits and 125 oracle tasks ran
with 0 failures, and all 366 archives pass the CRC scan.

**Smoke checks.** All pass. The MB arm is **bit-identical** to D1's test fit,
checked on seed 21101 / A1B1C1.

**Files.** `dev_results.json`, `dev_fits.csv` and `dev_curves.csv` (this
folder).

## Results

A1B1C1 regret is listed per seed, for seeds 21201 / 21202 / 21203.

| Arm and setting | Context average | Mean worst context | Seeds meeting R1 | A1B1C1 regret per seed | Cumulative online regret |
|---|---:|---:|---:|---|---:|
| MB | .0101 | .0174 | 3/3 | .009 / .014 / .010 | 251 |
| MB-prior β1 | **.0091** | .0163 | 3/3 | .010 / .009 / .012 | 262 |
| MB-prior β3 | .0114 | .0239 | 2/3 | .013 / .011 / **.035** | 282 |
| MB-UCB κ1 | .0139 | .0289 | 1/3 | .006 / **.034** / **.037** | 314 |
| MB-UCB κ2 | .0103 | .0204 | 2/3 | .006 / **.032** / .008 | **424** |
| MB-ε .03 | .0104 | .0217 | 2/3 | .009 / **.033** / .010 | 283 |
| MB-ε .10 | .0098 | .0188 | 2/3 | .006 / **.031** / .012 | 351 |
| MB-wide | .0130 | .0214 | 2/3 | .019 / **.027** / .017 | 271 |

## Reading

- **The development baseline was never trapped** (3/3 clean). So these seeds
  cannot show whether any remedy escapes the trap.
- **The trap depends on the trajectory.** Every remedy set-up except MB-prior
  β1 was trapped on a seed where MB was not.
- **ε exploration is too weak by construction.** Spread over 13 actions, it
  adds at most ε/13 to any one action.
- **UCB's optimism does not switch off.** Ensemble disagreement plateaus at
  about .04 under the bounded reservoir. UCB κ2's late regret is 3.3× MB's.
- **Two failures look different on the surface.** UCB κ2 on seed 21202 used
  the key action enough but still did not choose it greedily. That is a model
  error, not an exploration failure, and the usage condition separates the two.
- **MB-prior β1 is the only remedy never worse than MB.** It adds only +4%
  cumulative regret, at 1.75× the compute.

These findings led to **Amendment 1** of the protocol, made before the lock:

- drop the ε, UCB and wide arms;
- add directed count-based exploration;
- run a second development round on D1's known-trapped and known-clean seeds;
- run a 24-seed trap-rate test.
