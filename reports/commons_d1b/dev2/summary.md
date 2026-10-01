# D1b development 2 ("trap rescue"): summary

This round was run under Amendment 1 of
[docs/commons_d1b_protocol.md](../../../docs/commons_d1b_protocol.md).

**Scale.** Context A1B1C1 on D1's test seeds: 4 known-trapped and 4
known-clean. 40 fits and 65 oracle tasks ran with 0 failures.

**Checks.**

- MB reproduces the stored D1 fits bit for bit on all 8 seeds.
- The smoke-2 checks pass.
- For MB-count, c = 0 is bit-identical to MB, and the counts track the
  reservoir through eviction.

**Files.** `dev2_results.json`, `dev2_seeds.csv` and `dev2_fits.csv` (this
folder).

## Results

- **Rescued:** a trapped seed that ends with regret ≤ .025 and meets the
  usage condition.
- **New trap:** a clean seed that ends above .025.
- **Clean:** meets both the regret and the usage condition.

| Arm | Rescued (of 4) | New traps (of 4) | Seeds clean | Mean A1B1C1 regret | Cumulative online regret |
|---|---:|---:|---:|---:|---:|
| MB | 0 | 0 | 4/8 | .0220 | 378 |
| MB-prior β1 | 2 | 0 | 6/8 | .0167 | 361 |
| MB-prior β3 (selected) | 4 | 1 | 7/8 | .0159 | 337 |
| **MB-count c .05 (selected)** | **4** | **0** | **8/8** | **.0090** | 344 |
| MB-count c .15 | 4 | 0 | 8/8 | .0086 | 347 |

**How the rescues happened.**

- **Under MB**, the trapped seeds used the key action (2→3 × 6) in only
  .008–.014 of training episodes.
- **Under MB-count**, they used it in .082–.109 of training episodes and in
  .09–.14 of greedy choices.
- The bonus changes 17–23% of actions throughout training.

## Caveats (from the implementer, adopted)

1. **Rescue means a different trajectory,** not the same trajectory repaired.
   At D1's base trap rate of about 1/3, all 8 seeds clean by chance has a
   probability of about .04 per setting. The 24-seed test is the proper
   measure.
2. **The count bonus never fully fades** under the bounded reservoir. It is
   still about .005 at 80 episodes per action. Its cost in the other four
   contexts, and so R3 payback, is untested.
3. **The usage condition is partly met by construction for MB-count,** since
   the bonus raises every action's use. The regret half of R2 is the
   substantive test for this arm.
4. **MB-prior β3 is unstable.** It produced one new trap and one near miss.
5. **Lock identities.** D1's and dev-1's live lock checks use a glob over
   `src/acp_cl/commons/*.py`, so new modules make them fail. The locked files
   themselves are unchanged, and the archived source zips remain
   authoritative. The Amendment-1 script locks an explicit file list instead.
