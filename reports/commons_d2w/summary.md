# D2w: infeasible at stage 0, and the mechanism is retired in this world family

**Protocol.** [docs/commons_d2w_protocol.md](../../docs/commons_d2w_protocol.md),
the declared final round for the code-library mechanism.

## Result

**Stage 0 failed.** No setting of the allowed noise levers meets the declared
weak-evidence conditions. Those conditions were:

- identification in 20–60 episodes;
- a 99th-percentile per-episode log-likelihood ratio below 3 nats.

**The grid searched.** Supply noise σ_s ∈ {.1 … 1,000}, crossed with probe
noise σ_p ∈ {.1 … 1,000} with the probes on, or with the probes off. That is
88 settings. **None passes.**

Even with every sensor silenced (σ_s = 1,000, probes off):

| Pair | Episodes to .95 (p90) | p99 \|LLR\| (nats) | Relevance |
|---|---:|---:|---:|
| A0B0–A1B0 | 4.5 (10) | 11.2 | .092 |
| A0B1–A1B1 | 2.7 (6) | 12.6 | .046 |
| A0B0–A0B1 | 3.1 (6) | 10.5 | .028 |
| A1B0–A1B1 | 5.8 (14) | 10.3 | .027 |

## Why

**The exact failure pattern carries the evidence.** It records which entity
fails, and when. With the sensors silenced, it is still enough on its own:

- **Factor A.** The seasonal failure timing is informative even with no
  transfer (p99 about 10.6 nats).
- **Factor B.** There is no evidence without a transfer, but any transfer on
  the relevant channels is nearly decisive.

The failure pattern alone gives 1.3–2.9 nats per episode on average, so no
amount of sensor noise pushes identification beyond about 6 episodes.

**What could weaken it.** Only levers the protocol excludes: coarser or noisy
failure reports, or physics changes. Both would also change the learners'
training signal, so the result would be a different study.

## Also identified (not acted on)

- **The spawn rule cannot work under weak evidence.** In any world that does
  meet the conditions, the declared "lrt" rule could not detect genuine
  novelty. A single-episode fitted default gains about 3 nats from fitting
  noise alone, so FC would either never spawn or spawn on noise.
- **The lookahead reference would be expensive.** Under weak evidence, L would
  cost several hours per seed.

## Decision

The protocol declared D2w the final round for this mechanism in this world
family. With D2w infeasible under its declared levers, **the code-library
mechanism is retired in this world family**.

- **Development-level negative.** In D2, readable contexts favoured amortised
  inference, even over oracle codes.
- **No weak-evidence regime.** This world family has none, because exact
  failure outcomes make every relevant context identifiable within a few
  episodes.
- **What would be needed.** Testing the weak-evidence claim would need a
  world with genuinely noisy or coarse outcome feedback. That is a different
  study, and it is not planned.

**Files.**

- Code: `src/acp_cl/commons/references_d2w.py` and `scripts/commons_d2w.py`
  (stage 0 only).
- Results: `stage0/stage0.json`.
- Pools: `runs/commons_d2w_stage0/`, with sha256 sidecars.
- **Verification.** The new observation model reproduces D0b bit for bit at
  σ = .1.
