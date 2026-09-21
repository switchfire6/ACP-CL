# Independent v3 numerical audit

Passed: all 51 locked runs (21 recurring, 21 stationary, 9 early-biased) match the frozen configurations and reported arithmetic. No corrections were required.

The audit independently integrated all 5,100 acquisition curves through 256 current presentations, recomputed final accuracy and observed-checkpoint retention metrics, and checked all 28 paired contrasts plus two early-bias interactions. It did not invoke the project metric or reporting helpers.

The primary recurring `newborn_v3 - recycle_v3` late-AUC difference is **+1.154948 pp**, with seed differences **+0.078125, +3.042969, +0.343750 pp** (731, 842, 953). Its unadjusted 95% percentile interval is **[+0.078125, +3.042969] pp**. With only three pairs these endpoints are the observed seed extremes; this is exploratory evidence.

| Condition | Method | Final % | Late AUC % | Max observed fixed-pool drawdown, pp |
|---|---|---:|---:|---:|
| recurring | er_v3 | 97.2188 | 88.0013 | 7.8125 |
| recurring | recycle_v3 | 98.1849 | 95.5065 | 7.0312 |
| recurring | newborn_v3 | 99.5911 | 96.6615 | 3.1250 |
| recurring | newborn_matched_v3 | 99.4297 | 96.4219 | 1.5625 |
| recurring | protection_v3 | 98.9479 | 96.0755 | 7.8125 |
| recurring | consolidation_v3 | 98.5729 | 96.6484 | 7.8125 |
| recurring | full_v3 | 97.7214 | 97.2669 | 9.3750 |
| stationary | er_v3 | 99.9505 | 99.4531 | 9.3750 |
| stationary | recycle_v3 | 99.9635 | 99.6784 | 3.9062 |
| stationary | newborn_v3 | 99.9844 | 99.5846 | 0.0000 |
| stationary | newborn_matched_v3 | 99.9896 | 99.5690 | 0.7812 |
| stationary | protection_v3 | 99.9792 | 99.3268 | 7.8125 |
| stationary | consolidation_v3 | 99.9688 | 99.8164 | 6.2500 |
| stationary | full_v3 | 99.9323 | 99.5104 | 6.2500 |
| early_biased | recycle_v3 | 96.1589 | 58.9063 | 5.4688 |
| early_biased | newborn_v3 | 93.2109 | 63.5977 | 3.9062 |
| early_biased | full_v3 | 98.5573 | 66.7461 | 4.6875 |

The combined condition improves mean recurring late AUC while reducing mean final accuracy versus recycling. The early-bias late-AUC interaction is mixed across seeds: +19.019531, -0.007813, -0.773437 pp. Its positive mean is driven by seed 731.

All 102,000 optimizer updates satisfy the logged 0.1 displacement cap within a 1e-6 numerical tolerance (maximum recorded norm: 0.1000000060). Each recycling run performs 98 scheduled events and 294 unit resets; reset identities satisfy the appropriate 30- or 60-update eligibility rule. Gain windows and nominal matched allocation agree with independently reconstructed reset ages. All 15 consolidation-enabled runs perform 288 maturations each, with positive local consolidation. Gain and protection diagnostics activate only in their intended conditions.

Within each paired cohort, whole-stream tensor hashes, final replay tensors, sampling/augmentation RNG states, and training exposure agree exactly. Warmup allocation traces match. Independent-color and early-biased validation hashes match for all three seeds. No boundary or offline allocation controller is present in the final learner checkpoints.

The source hash is `cccbb40b2fe578be511f49c31d9629abd823f89bf078ccc9781b1d38c17c5efe`. Selection and all locked configuration bytes match commit `7fbb270`; selection gain is 0.5 and newborn peak is 2. Recorded development completion precedes that commit, and evaluation manifest creation follows it. This is recorded provenance, not trusted-clock proof.

The main narrative's endpoint and allocation tables, per-seed directional claims, and resource totals also agree: 67 comparative runs, 121,200 updates, 28,116 parameters, and 7,247.7909347 summed logged seconds. Concurrent wall times are not GPU compute time.

Limits: no training rerun or full stream regeneration; three seeds; procedural and near-ceiling stationary tasks; raw AUC mixes retained knowledge with learning; sparse retention checks; optimizer clipping excludes resets; nominal budget equality does not imply equal applied updates or compute. Mechanism activation is not evidence that every mechanism improves performance.

[Detailed independent arithmetic, per-seed values, activation counts, and input hashes](independent_audit.json).
