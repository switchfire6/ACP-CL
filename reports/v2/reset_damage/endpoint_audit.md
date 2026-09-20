# Stationary endpoint timing audit

Completed runs: **9/9**. This is a post-hoc timing association audit, not causal attribution.

The comparison evaluates **the same first 90 validation sets** at optimizer steps 1800 and 2000. The usual final result averages all 100 sets.

| Method | Seed | Same-set accuracy at 1800 (%) | At 2000 (%) | Change (pp) | Reported final (%) | Max reset-step damage | Max nonreset-step damage |
|---|---:|---:|---:|---:|---:|---:|---:|
| er_recycle | 101 | 98.7240 | 88.6979 | -10.0260 | 88.7344 | 0.3399 | 0.4637 |
| er_recycle | 202 | 99.7656 | 99.7222 | -0.0434 | 99.7344 | 0.0152 | 0.0770 |
| er_recycle | 303 | 99.7830 | 99.8264 | +0.0434 | 99.8438 | 0.0988 | 0.2009 |
| acp_v2 | 101 | 88.8455 | 89.1927 | +0.3472 | 89.2344 | 0.8728 | 0.4346 |
| acp_v2 | 202 | 73.6458 | 74.7396 | +1.0938 | 74.7344 | 0.0792 | 0.1780 |
| acp_v2 | 303 | 73.4635 | 73.6806 | +0.2170 | 73.6172 | 0.1247 | 0.0756 |
| acp_v2_no_reopening | 101 | 89.0104 | 89.1580 | +0.1476 | 89.2188 | 0.5864 | 0.5360 |
| acp_v2_no_reopening | 202 | 99.4965 | 55.9896 | -43.5069 | 56.0156 | 5.3868 | 7.3453 |
| acp_v2_no_reopening | 303 | 73.4635 | 73.6372 | +0.1736 | 73.5703 | 0.0862 | 0.0513 |

Every run resets 30 units during steps 1801-2000 and three units at the final update. Every final-step paired replay probe reports zero positive damage.

No-reopening seed 202 changes from 99.4965% to 55.9896% on identical held-out sets (-43.5069 pp). ER plus recycling seed 101 also declines from 98.7240% to 88.6979% (-10.0260 pp); the remaining seven changes range from -0.0434 pp to +1.0938 pp.

For no-reopening seed 202, experience 99 accuracy falls from 99.21875% at step 1976 to 70.3125% at step 1980. On the separate, fixed experience 100 validation set, accuracy at steps 1980/1984/1988/1992/1996/2000 is 60.15625/80.46875/53.125/78.90625/74.21875/55.46875%.

Logged replay-damage spikes are 1.6674 at step 1975 (no reset), 5.3868 at 1980 (three resets), and 7.3453 at 1985 (no reset). Existing probes bracket optimizer updates and resets jointly, so the coincidence at step 1980 does not establish reset causality. The [exact instrumented reproduction](instrumented_audit.md) separates their immediate effects.

Attribution limits:

- Stored paired replay probes bracket the optimizer update and recycling together; they do not isolate the instantaneous reset effect.
- Probe examples change between monitoring windows; replay damage is max(0,(CE_after-CE_before)/max(CE_before,0.1)), not an unbounded relative percent when CE_before<0.1.
- No-reset updates can still involve recently reset newborn units, their gain floor, momentum, and the ungated classification head.
- All runs reset 3 units at step 2000, and all final paired probes show zero positive damage; this does not prove that the isolated reset effect is zero because update effects can offset it.
- The selected anomaly and time window are post-hoc; this audit does not change endpoints or select hyperparameters.

Full same-set values, all last-ten-experience learning curves, and logged events from the last 200 steps are preserved in [endpoint_audit.json](endpoint_audit.json).

Training source SHA-256: `67ae36402607f257f1e55c4f3ef2bbb477a88555d2111a4bd0139e27668eee0a`.
