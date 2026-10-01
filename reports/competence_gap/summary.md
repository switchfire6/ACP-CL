# Step B: the stationary competence gap

This is a prospective study run under
[docs/competence_gap_protocol.md](../../docs/competence_gap_protocol.md). The
config, protocol and source hashes were locked in `runs/competence_gap/manifest.json`
before the main fit. It is a single-law competence question, not a
continual-learning comparison. It does not reopen the stopped representation
recipe.

Files:

- `results.json`: all numbers, bootstrap intervals and verification flags.
- `per_seed_arm.csv`: one row per seed × arm × score point.
- `online_excess.csv` and `online_excess.png`: online prequential excess per
  1,024-arrival chunk.
- The runner and analysis are `scripts/competence_gap.py` (with
  `scripts/competence_gap_learner.py` for the privileged arm). Raw fits and
  oracle files are in `runs/competence_gap/`.

## Verdicts (all pre-declared)

| Question | Rule | Result | Verdict |
|---|---|---|---|
| Updates per batch, U3 vs U12 | Removes ≥ 25% of the reference excess, same sign in ≥ 5/6 seeds | −.0015 (4% removed); 4/6 seeds reduce | Does not matter |
| Replay capacity, R256 vs R16 | Same rule | −.0082 (**24.1%** removed; bootstrap .214–.261); **6/6** seeds reduce | Does not matter by the rule (**narrow miss**) |
| Width, W256 vs W64 | Same rule | **+.0056** (worse; 16% *added*); 6/6 seeds worse | Does not matter (wider is consistently worse) |
| Exposure, reference continued to 32,768 arrivals | Closes ≥ 25% → a shortage of samples contributes | Excess .0341 → .0355 (−4% closed); 1/6 seeds improve | **Steady state**, not a shortage of samples |
| All-action feedback (privileged) | Fraction closed = cost of sparse feedback | **44%** closed (bootstrap .38–.49); 6/6 seeds | Sparse feedback costs about 44% of the gap |
| Competent base learner | Any declared arm with excess ≤ .010 | Best is U12_R256_W64 at .0226 | **None**: the gap is characterised, not solved |

The replay effect falls 0.9 percentage points short of the threshold. It is
consistent in all six seeds, and its bootstrap interval straddles the line. We
report it as a narrow miss and do not credit it: the rule was fixed in advance.

## Arm table (seed means, 8,192 arrivals unless marked)

O is the law-known Bayes oracle on the same panel. Its Brier is .0442 (seeds
.0412–.0468; MC standard error about 4e-5). The gauge-integrated O differs by
at most .0002, so pixel quantisation of the reserves does not explain the gap.
O's cue benefits are .0297 (cue 0), .0143 (cue 1, delay) and .2163 (cue 2). O's
survival is .677.

| Arm | Brier | Excess over O | 95% bootstrap | Closed | Survival regret | Cue 0 | Cue 1 | Cue 2 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **U12 R16 W64 (reference)** | .0783 | **.0341** | .030–.040 | 0 | .026 | .016 | .006 | .192 |
| U12 R16 W256 | .0871 | .0430 | .040–.047 | −.26 | .032 | .017 | .006 | .189 |
| **U12 R256 W64 (best)** | .0668 | **.0226** | .020–.026 | .34 | .014 | .012 | .004 | .188 |
| U12 R256 W256 | .0721 | .0279 | .027–.029 | .18 | .022 | .018 | .007 | .189 |
| U3 R16 W64 | .0733 | .0292 | .025–.033 | .15 | .047 | .004 | .001 | .156 |
| U3 R16 W256 | .0790 | .0348 | .032–.038 | −.02 | .033 | .010 | .003 | .175 |
| U3 R256 W64 | .0717 | .0276 | .025–.030 | .19 | .036 | .003 | .001 | .157 |
| U3 R256 W256 | .0744 | .0302 | .028–.033 | .12 | .025 | .006 | .001 | .169 |
| Exposure @ 32,768 | .0797 | .0355 | .034–.037 | −.04 | .028 | .017 | .007 | .191 |
| *All-action (privileged)* | .0632 | *.0191* | .017–.022 | *.44* | *.008* | .022 | .009 | .208 |

Per-seed excess:

| Arm | 19101 | 19102 | 19103 | 19104 | 19105 | 19106 |
|---|---:|---:|---:|---:|---:|---:|
| Reference | .0483 | .0326 | .0305 | .0282 | .0307 | .0346 |
| U12 R256 W64 | .0314 | .0188 | .0202 | .0200 | .0222 | .0230 |
| All-action | .0233 | .0225 | .0171 | .0149 | .0166 | .0201 |

## Descriptive findings

1. **The plateau is a steady state of the learner.** The reference arm's
   online excess falls to about .037 by 4,096 arrivals. It then stays within
   .035–.037 for the next 28,000 arrivals (figure, right panel). More data from
   the same law does nothing.
2. **Even perfect feedback plateaus early.** The privileged all-action arm
   reaches .019 by the third chunk and stays flat. It sees five times as many
   labels, yet it hits a floor about 1.4× O's Brier. So sparse feedback is
   **not** the main cause of the floor. The all-action arm also uses R16 and
   U12, and a feature-level learner does far better (see the review). The
   likely cause is therefore how this learner updates or perceives. Candidates
   are a constant step size and repeated updates on a small window, or the
   pixel encoder. This is a hypothesis; Step B2 tests it.
3. **Replay diversity matters only when updates are repeated (U×R interaction
   +.010).**
   - With 12 updates per batch, a 256-packet reservoir removes .0115 (W64) and
     .0151 (W256) of excess.
   - With 3 updates, it removes only .0016 and .0046.
   - The U12 R256 arms are also the only factorial arms still clearly
     improving at 8,192 arrivals (figure, left panels).
   - This fits a model where the reference overfits its most recent ~512
     records. It is not evidence of limited capacity.
4. **More width hurts** in all six seeds, most with small replay. More capacity
   on a small moving window overfits it faster.
5. **Brier and decisions disagree for U3.** The U3 arms improve Brier over the
   reference (.029 vs .034) but have *worse* survival regret (.036–.047 vs
   .026). They also learn much weaker cue relations (cue-0 benefit .003–.004 vs
   .016). Fewer updates yield smoother, better-calibrated but less relational
   forecasts. Future screens need decision regret, as the proposal argued.
6. **Relational cues are partly learned.**
   - Supply timing (cue 2) reaches about 89% of O's cue benefit in U12 arms.
   - Directional efficiency (cue 0) reaches about 40–60%.
   - Delay (cue 1) reaches about 30–50%, and about 64% with all-action
     feedback.
   - Without context switching, the delay cue is learned at a modest level in
     this single law.

## What this does and does not establish

- **It establishes** that, within this learner family, the single-law gap is a
  steady state. It is not removed by more exposure, more width or fewer
  updates. It is reduced by at most about a third by a larger replay reservoir,
  and by less than half even with perfect all-action feedback.
- **It does not establish** that O is reachable by any learner from 8,192
  performed-action records. O knows the simulator. A learner must estimate
  about 64 discrete cells × 5 actions × a 2-d reserve surface from labels on
  one action per record. The .010 bar was not calibrated against an
  *achievable* learner. That calibration is exactly the next question (see
  [docs/competence_ceiling_protocol.md](../../docs/competence_ceiling_protocol.md)).
- **Six seeds.** Intervals are descriptive, and nothing here is credited to a
  learner until it is re-confirmed on fresh seeds.

## Integrity

- **Locking.** Config, protocol and sources were locked before the main run;
  the lock was re-checked at analysis.
- **Smoke checks.** The smoke run in `runs/competence_gap_smoke/` passed all
  seven engineering checks: arm settings, replay capacity, all-action loss,
  determinism, oracle, disjointness and interrupted-then-resumed equality.
- **Shared inputs.** All arms of a seed saw the identical arrival stream, by
  fingerprint. Arms of equal width shared their initialisation.
- **Exposure prefix.** The exposure arm's first 8,192 arrivals are
  bit-identical to the reference in all six seeds.
- **Oracle.** It reproduces evaluator truth exactly with the realised supplies,
  on every endpoint and training case. Online calibration z-scores are all
  within ±2.8 across 90 action × horizon cells.
- **Interruptions and failures:**
  - The first `run` invocation was interrupted when the previous Claude session
    ended, after all 60 fits had completed.
  - On resumption, one endpoint-oracle MC task (gauge, seed 19102, batch 1)
    crashed with a nonsensical NumPy index (about 2.25e15, from a boolean mask
    of length 65,536). This points to a transient fault, not a logic error.
  - The same task rerun in isolation succeeded. Its output is bit-identical to
    the task that the final `run` invocation then wrote. The sibling task (exact
    O, same seed and batch) is bit-identical between the run and an isolated
    rerun.
  - All three invocations and the failure are logged in
    `runs/competence_gap/attempts.jsonl` and `run_stdout*.log`.
- **Independent review.** See the section below.

## Independent review (2026-09-29)

An independent agent checked the results adversarially. Its machine-readable
outputs are in `review/`. It modified no existing file.

| Check | Result |
|---|---|
| Recompute from raw arrays | **PASS**. Every headline number reproduces; the largest difference is 2.8e-17. Cummin is applied consistently, and applying it to O has no effect. |
| Independent Monte Carlo of O | **PASS**. All 512 queries × 6 seeds at K = 8,192. Mean z² is 1.01–1.06, and Brier differences are at most 9e-5. Its supply sampler replays the evaluator's realised supplies exactly. |
| Leakage and confounds | **PASS**. Identical streams, with all 32,768 arrivals regenerated. Shared initialisation. The panel and supports are disjoint from training images. R256 holds 256 packets. Actions are uniform. The all-action arm is otherwise identical to the reference. |
| Determinism | **PASS**. Three from-scratch refits (U3_R16_W64/19104, U12_R16_W64/19102, all_action/19102) and the two oracle tasks around the crash are bit-identical. |
| Analysis-code bias | **PASS**, with the interpretive caveats below. |

The reviewer found that the crashed index, 2251799813738582, is exactly
2^51 + 53,334. That is a valid index with one bit flipped, which is consistent
with a transient hardware fault. The reviewer's "unexplained 9-hour stall" is
simply the time between the previous Claude session ending (which killed the
first invocation) and this session resuming it. A memory test of this machine
is still advisable.

**Interpretive caveats, which we adopt:**

1. The R256 near-miss (24.06%; bootstrap 21.4–26.1%) is a coin-flip against
   the threshold. It correctly fails the locked rule, but it should not be
   read as "replay does not matter".
2. The replay effect depends on U. At U12, R256 removes 34% (W64) and 44%
   (W256) of excess in 6/6 seeds; at U3 it removes 5–14%. The protocol's
   averaging over cells dilutes it.
3. With 256 arrivals of packets, R256 never evicts anything. It is effectively
   uniform replay of the **entire** history, not just a bigger buffer.
4. The steady-state verdict applies to the reference recipe only. The exposure
   arm keeps R16. The best arm (U12 R256) is still improving at 8,192
   arrivals, so a sample shortage is untested for it.
5. Seed 19101 is an outlier (reference excess .048 vs .028–.035), which
   inflates the denominator of every fraction.

**Is the bar achievable? (the reviewer's opinion, with a descriptive check)**

The reviewer trained an MLP on the *true latent features* of these same six
seeds. It was early-stopped on later training records, never on the panel.
This is not a protocol quantity.

| Training data | Mean excess |
|---|---|
| 8,192 performed-action records | **.0085** (range .0065–.0101) |
| 8,192 all-action records | .0030 |
| 32,768 performed-action records | .0050 |

So the .010 bar is roughly the ceiling at 8,192 records **even with perfect
perception**. "No arm ≤ .010" is therefore expected, not damning. Most of the
reference's .034 excess would then be perception/architecture and online
optimisation, not finite data. That is exactly the decomposition Step B2
measures prospectively on fresh seeds (19301–19312).

**Disclosure.** This preview on the Step B seeds arrived after the B2 protocol
was written but before B2 was locked. The B2 protocol is not changed in
response to it.
