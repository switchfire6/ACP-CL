# Saved core/residual output diagnostic

Read the [interpretive report](../core_residual_output_results.md) first.
Both exploratory allocation rules pass; neither original continuous-learning
primary screen passes. This archive contains predictions from saved models,
not a new training experiment or a working context selector.

- [Pre-score design and decision rules](../../docs/core_residual_output_review.md)
- [Development summary](development/summary.md) and [all numerical results](development/summary.json)
- [Development PNG](development/allocation.png) and [SVG](development/allocation.svg)
- [Corrected engineering smoke summary](smoke_v2/summary.md)
- [Proposed next test, not run](../../docs/core_residual_output_followup.md)
- [Independent outcome review](../core_residual_output_checks/outcome_review.md)
- [Independent arithmetic proof](../core_residual_output_checks/independent_review_results.json)
- [Outside-repository rescoring and corruption checks](../core_residual_output_checks/portability_review.json)
- [Old-science preservation proof](../core_residual_output_checks/preservation_final.json)

The development panel includes all seeds 16001–16006 and both architectures,
joint/separate arms, four endpoint targets, actual causal history and two fresh
target-support replicates. Novel targets include both cue-flip probes. There
are 384 input/context rows, 48 saved states and 1,536 component forecast tensors.
The corrected smoke uses original seed 16991 and its one archived support
replicate: 40 rows/eight states. Smoke is ineligible for scientific decisions.

Each `jobs/{model}_{seed}.json` binds the selected original checkpoint/trace
hashes, loaded-state signatures, component metrics and raw array hashes.
Its matching NPZ contains all queries, performed-action supports, physical
truth, scoring masks, copied original full forecasts, and new component
predictions. These arrays suffice for independent NumPy rescoring. The manifest
binds original job ledgers and archived training source, plus the diagnostic
source/protocol/test archive. Artifact manifests seal the completed output files.

The failed first `smoke` directory retains its original source lock; it contains
no completed forecasts. Its metadata tuple/list equality check was corrected
before running `smoke_v2` or the development diagnostic. See the
[engineering repair record](../core_residual_output_checks/engineering_repair.json).
Do not run the current source against the obsolete failed lock.

For rescoring from this workspace, write to a new location:

```powershell
.venv/Scripts/python.exe -I -B scripts/summarize_core_residual_outputs.py --input reports/core_residual_outputs/development --output runs/core_residual_outputs_rescore/summary.json --development
```

For outside-repository rescoring, copy the development folder and extract
`scripts/summarize_core_residual_outputs.py` from its `source_at_lock.zip`.
Run that script with Python and NumPy, with `--input` pointing to the copied
folder and `--output` outside it. No Torch, model pickle or live scientific
package is needed for scoring. The copied forecast archive does **not** suffice
for recreating neural inference: that requires the unchanged original
`runs/core_residual_development` checkpoints and their archived training source.

The standalone summary checks prediction arithmetic, array/file provenance,
component coverage, pairing, state-signature records and frozen-core invariance.
It cannot independently prove the neural tensors produced those forecasts;
the local runner does the saved-checkpoint reconstruction. Existing source and
tests were not changed after main locking. Neither verification is a fresh
scientific replication.
