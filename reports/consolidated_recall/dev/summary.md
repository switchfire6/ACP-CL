# Step C, consolidated recall: stopped at development

**Verdict: STOPPED at development under the pre-declared pathology rule. No
test seed was run, and nothing was locked.**

This follows [docs/consolidated_recall_protocol.md](../../../docs/consolidated_recall_protocol.md).
All numbers below come from the declared development phase:

- 3 development seeds (20101–20103);
- 36 fits;
- evaluator-only law-known oracle O.

They are exploratory and not claims. The test seeds 20201–20212 stay unused,
and can serve a future study.

**Files.**

- `dev_results.json` and `dev_per_seed.csv` (this folder).
- Code: `scripts/consolidated_recall.py` and `scripts/consolidated_recall_learners.py`.
- Raw outputs: `runs/consolidated_recall_dev/` and
  `runs/consolidated_recall_smoke/`.
- All 11 smoke checks passed.
- The development phase ran with no failures. A CRC check passed on 316
  archives.

## Why it stopped

The protocol's stop rule says CR does not go to test if the selected CR makes
more than 3× as many active-snapshot changes as there are true mode switches.

- Seed 20101 had 47 changes against 15 switches; the limit is 45.
- Under the per-seed reading the rule fires. That reading was chosen before
  any development result existed, and it is the conservative one.
- The seed-mean reading (26.7 changes) would not fire.

We apply the per-seed reading.

The stop is not a technicality. The development evidence below shows the
mechanism class failing on its own terms, including with **perfect**
indexing.

## Development results

These are the mean over 3 seeds.

- **Stream excess** is the prequential excess over O in cycles 3–8.
- **Recovery excess** is the excess over O at positions 2–5 of the Base A
  blocks. I was not computed for development seeds.

| Arm (d) | Stream excess | Per seed | Recovery excess |
|---|---:|---|---:|
| ON-R256 (plain online) | .0376 | .0391 .0364 .0372 | .0414 |
| ON-ref (R16) | .0588 | .0625 .0544 .0593 | .1281 |
| COND-4 (4-head conditional, U12/R256) | .0363 | .0368 .0352 .0370 | .0475 |
| **EMA-global (.99)** | **.0306** | .0313 .0300 .0304 | .0347 |
| EMA-global (.998) | .0317 | .0346 .0293 .0313 | **.0246** |
| CR-K1 (.998) | .0317 | .0347 .0293 .0312 | .0246 |
| CR (.998, selected) | .0324 | .0359 .0301 .0312 | .0347 |
| CR-no-recall (.998) | .0317 | .0345 .0295 .0312 | .0298 |
| CR-oracle-mode (.998), *privileged* | .0366 | .0362 .0358 .0377 | .0572 |
| CR-oracle-law (.998), *privileged* | .0350 | .0353 .0328 .0369 | .0561 |

- τ(d) was calibrated at 18.5, 23.9 and 24.9 nats per batch for d = .99, .995
  and .998.
- The selected decays: CR .998, CR-K1 .998, EMA-global .995.

**CR diagnostics at .998:**

- 1–2 snapshots.
- Mode purity .86–.89, law purity .26–.29.
- 0–46 recalls, with some flipping back and forth within blocks.
- Pending fraction ≤ .004.
- On seed 20103, CR never spawned and was identical to EMA-global.

## What development shows (descriptive, 3 seeds)

1. **Averaging still helps under switching, but less.**
   - One global EMA removes about 17–19% of plain online learning's stream
     excess. B2 found 48% within a single law.
   - With a slow decay it roughly halves the recovery excess: .025 vs .041.
2. **Indexing adds nothing, and recall hurts.**
   - CR is no better than CR-K1 or EMA-global.
   - CR without recall beats CR on both outcomes.
3. **Perfect indexing with recall is worse than plain online learning.**
   - CR-oracle-mode (.0366) and CR-oracle-law (.0350) are both worse than
     EMA-global. Their recovery is flat, at about .05–.06 over positions 2–8.
   - Copying a whole-network snapshot into the fast weights throws away
     learning that the modes share. The perception features, the reserve and
     survival relations and the cue relations are all common to every law,
     because only the supply-column mode differs between the Base laws.
   - A per-context snapshot also sees only its own context's data, which is
     20% for Base A.
4. **Implication.** For laws that share most of their structure,
   **whole-network per-context consolidation is the wrong granularity**. The
   evidence is consistent with a factored view:
   - consolidate shared knowledge globally (the averaging gain);
   - keep only a small context-specific part retrievable.

   Step A1 pointed the same way. One fitted 12-d context vector, with every
   shared weight frozen, recovered most of Base A's competence.
5. **Multi-law cost.** Plain online learning's stream excess (.0376) is well
   above its single-law value in B2 (.0237). Mixing laws through replay and the
   support-context pathway costs about as much as the gap Step C tried to
   close.

## Decision and next step

The recoverable-context proposal pre-declared: "If Step A1 or Step C fails,
pivot sooner rather than trying another architecture here."

- A1 was borderline. Step C was stopped at development, and its privileged
  arms show the whole-network recall class failing even with a perfect index.
- We therefore **do not revise the mechanism again on this benchmark**. The
  factored "shared slow core plus indexed small codes" idea is recorded as a
  **hypothesis to carry into Step D**. It is not tested here.
- Step D's first learner check should be its privileged ceiling: oracle-indexed
  codes over a globally averaged shared network. Inference should only be
  tested after that ceiling shows a gain.

## Notes and deviations

- **Report bug.** `snapshots_created` omits the oracle arms' lazily created
  snapshots; `final_snapshots` is correct. The script was left unedited so that
  the development manifest still matches the code.
- **Smoke check.** The first smoke check attempt failed on a bug in its own
  comparison: NaN-padded arrays were compared as unequal. It was fixed and
  rerun, and every reported check comes from the clean rerun. The failed log is
  kept.
- **Where the protocol was ambiguous, the implementer chose as follows:**
  - **EMA start.** The start snapshot and EMA-global begin with a = 0 and
    n = 0, so φ = θ₀ until the first update.
  - **Spawn.** A spawned snapshot counts the spawn value as one sample. The
    run counter covers only the snapshots that existed at scoring and resets
    after a spawn. Replacement removes the snapshot with the fewest actual
    consolidation steps.
  - **Pending.** It uses the ℓ_θ from before any recall, and the k\* after
    recall. It governs consolidation at batch t and the forecast at t + 1.
  - **Same-batch events.** For recall and spawn on the same batch, the
    literal order was kept. It never occurred.
  - **Oracle arms.** They receive batch t's index after its forecast, so it is
    lagged for t + 1. A new index creates a snapshot without recall.
  - **Variants.** CR-no-recall switches k\* but never copies θ. CR-K1 is
    K_max = 1 with recall disabled.
  - **Likelihood.** It uses the scored forecasts after cummin.
  - **τ.** Computed by linear-interpolation percentile over 840 samples per d.
    The paired SE is the SD of the differences divided by √3.
  - **Shared fit.** ON-R256 and EMA-global share one fit through a read-only
    hook.
  - **COND-4.** `ContextLearner('conditional')` at width 64, 4 experts,
    lr .002, evidence strength 1, U12/R256.
  - **I.** It was not computed for development seeds. Its forecast uses the
    filtered posterior after t − 1.
  - **Oracle bias.** O's Monte Carlo variance bias (about 8.6e-5) is not
    corrected.
- **Informal power note.** Even if C2 had been reached, the development effect
  sizes show it could not pass: the CR / CR-K1 ratio was 1.02.
