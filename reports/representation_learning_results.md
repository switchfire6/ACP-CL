# Temporal representation learning: qualification result

Status: completed prospective reference qualification; **STOP before the
candidate comparison**.

The proposed experiment asks whether training a recurrent predictor to preserve
the whole observed four-frame sequence can make later temporal relationships
easier to learn. Before fitting that auxiliary objective, the protocol required
the ordinary outcome-only reference to demonstrate reliable learnability both
when dependencies are introduced separately and when five environments recur.
This protects against attributing a basic context-identification failure to a
reconstruction mechanism.

Six fresh, counterbalanced seeds18101--18106 completed24 reference fits. The
study contains393,216 arrivals and147,456 optimizer steps. Each fresh cumulative
dependency fit and the blocked-interleaving fit use8,192 arrivals per law.
The full fixed protocol is [here](../docs/representation_learning_protocol.md).

## Result

All fresh-learning requirements passed. The model improved over its performed-
action marginal predictor by about .149--.156 raw Brier units across fresh
stages, and all fresh stage/cue benefit groups met the .002 threshold. Thus the
three dependencies are learnable in separate reference fits at the declared
exposure.

The five-law interleaved reference failed4 of31 required groups:

| Fixed requirement | Seed-mean result | Required value |
|---|---:|---:|
| Base-A endpoint all-case Brier | .195597 | <= .12 |
| Stage1 arrival-delay cue benefit | -.000152 | >= .002 |
| Stage2 arrival-delay cue benefit | .000802 | >= .002 |
| Stage3 arrival-delay cue benefit | -.000291 | >= .002 |

All other interleaved checks passed: four law-slot Brier ceilings, all five
marginal-improvement checks, and the efficiency and supply-timing availability
cells. The complete seed values and descriptive bootstrap intervals are in the
[machine-readable summary](representation_learning/qualification/summary.json)
and [human-readable summary](representation_learning/qualification/summary.md).

The correct conclusion is narrow. The ordinary recurrent reference learns the
dependencies when conditions are isolated, but it did not meet the required
simultaneous availability criterion after recurring blocked interleaving. That
could arise from contextual environment identification, the finite history,
the sparse performed-action outcome signal, optimization, or representation
learning. These results do not locate the failure inside the encoder.

Because reference qualification is a precondition, the planned outcome,
final-frame reconstruction, full-sequence reconstruction and useful-compute
arms were **not** fit. Do not describe the sequence objective as failed, and do
not run the prepared main config, tune its weight, exclude the delay cue, or
relax its thresholds after this result. Any next attempt needs a distinct,
prospective hypothesis and protocol.

## Integrity and engineering

The independent audit passed. It restored48 checkpoints, regenerated all12,288
training packets and the physical outcomes/masks for all393,216 arrivals,
replayed147,456 deterministic replay choices, and reconstructed288 saved
boundary evaluation forecasts. Learner and global RNG state remained unchanged.
The audit did not rerun optimization or reconstruct ordinary interior forecasts;
those limits are explicit in the
[audit](representation_learning_checks/qualification_audit.json).

The useful-compute control was allocated15 updates per batch from a separate,
pre-main engineering run. It was never used for scientific comparison. The
[allocation record](representation_learning_checks/compute_allocation.json)
contains the timing/work accounting.

A Windows file-replacement conflict occurred while checkpointing one interleaved
job. The last committed checkpoint was preserved, the pending checkpoint was
archived, and the same locked run resumed; the
[recovery record](representation_learning_checks/runtime_recovery.json) records
the384 reexecuted optimizer steps. The final source/config/protocol identity and
all result hashes are stored in
[the run manifest](../runs/representation_learning_qualification/manifest.json).
