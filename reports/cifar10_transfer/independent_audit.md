# Independent CIFAR-10 transfer audit

All 16 raw runs passed the independent artifact and arithmetic checks.

This is a local independent agent recomputation, not a training rerun, external peer review, or scientific replication.

Frozen effort rule: **do not advance this recipe**. Adequacy: True; signal: False; retention guard: False.

| Condition | Method | Seed | Final % | Late AUC % | Stationary drawdown, pp |
|---|---|---:|---:|---:|---:|
| recurring | er_v3 | 1063 | 42.0667 | 37.4900 | — |
| recurring | er_v3 | 1174 | 43.6000 | 41.3033 | — |
| recurring | recycle_v3 | 1063 | 42.7333 | 38.6533 | — |
| recurring | recycle_v3 | 1174 | 44.0667 | 41.8233 | — |
| recurring | newborn_v3 | 1063 | 37.4667 | 38.1967 | — |
| recurring | newborn_v3 | 1174 | 42.1333 | 42.0333 | — |
| recurring | newborn_matched_v3 | 1063 | 42.1333 | 38.4633 | — |
| recurring | newborn_matched_v3 | 1174 | 44.3333 | 42.3733 | — |
| stationary | er_v3 | 1063 | 48.0000 | 42.1400 | 8.4000 |
| stationary | er_v3 | 1174 | 47.8000 | 43.7333 | 6.0000 |
| stationary | recycle_v3 | 1063 | 44.0000 | 42.8667 | 7.6000 |
| stationary | recycle_v3 | 1174 | 45.2000 | 43.2767 | 5.6000 |
| stationary | newborn_v3 | 1063 | 44.2000 | 43.0233 | 8.4000 |
| stationary | newborn_v3 | 1174 | 46.6000 | 43.7667 | 5.0000 |
| stationary | newborn_matched_v3 | 1063 | 45.0000 | 42.5767 | 7.4000 |
| stationary | newborn_matched_v3 | 1174 | 45.2000 | 43.6367 | 2.8000 |

Recomputed every first-256-presentation AUC, equal-domain final accuracy, all 30 stationary diagonal drawdowns, and the locked effort criteria. Verified identities, finite learner checkpoints, single-pass arrivals and split separation, replay/RNG and warmup pairing, reset counts, nominal budgets, and all 22,560 optimizer-update caps.

All 480 independently integrated AUCs, endpoint/domain values, stationary drawdowns, primary paired contrasts, and the effort decision agree with the primary summary.

Limits: Two development seeds; no confidence interval or efficacy claim. Concurrent condition suites share one GPU; summed run wall time is not GPU compute time. No training rerun or full transformed-stream regeneration; split/arrival metadata and recorded stream hashes checked. Warmup traces match; historical warmup weights are not separately checkpointed. Nominal gain equality is not equality of achieved displacement or compute; reset mutations are outside the cap. Repeated validation domains share base images; 30 experiences are not 30 independent samples.

[Detailed values, decision checks, and raw artifact hashes](independent_audit.json).
