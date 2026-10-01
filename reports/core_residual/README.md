# Adam: learned core and adaptive raw-input residual

The [prospective protocol](../../docs/core_residual_protocol.md) fixes the
architecture, cohort, resource accounting, evaluation and decision criteria.
The [development design review](../core_residual_checks/design_review.md) and
[execution ledger](../core_residual_checks/run_ledger.md) record decisions before
main fitting. [Related work and mathematical scope](../../docs/core_residual_related_work.md)
explain the close architectural precedents and the limits of this comparison.

The [completed report](../core_residual_results.md) records **failed primary
screens in both architectures**, with successful fresh qualification. The
recurrent feature-attribution screen passes; that separate result does not
rescue the primary candidate. Full [development tables](development/summary.md)
and all seeds remain archived.

The `smoke` cohort checks engineering only. The `development` cohort has six
fresh seeds per architecture, three continuing arms, and fresh-model references.
All cores learn their common prefix from random weights. They then receive the
same zero-output residual, input streams, uniform replay and update budget.
Joint trains both pathways; separate freezes the learned core; fixed_features
also freezes the residual visual encoder at random initialization. Heads and
history/readout parameters still learn in that last control.

Each cohort preserves its configuration/runtime identity, source and analysis
ZIPs, locked protocol, original phase/job/evaluation records, all-action
probability and physical-outcome arrays, numerical summaries and resource rows.
The local run directories retain trusted model/optimizer checkpoints with
explicit gradients. A checkpoint audit binds these states to the exported
evidence. Development-cohort figures are provided as PNG, SVG and PDF.

The local audit independently checks initial forks, frozen parameters and
Adam moments, per-parameter step counts, causal histories, uniform replay,
physical outcomes and query/support provenance. It reconstructs forecasts at
saved boundaries/endpoints and the first pre-update prediction in each phase.
Intermediate model states are not saved. Those forecasts have raw arithmetic,
physics and provenance checks, without independent ordinary-stream retraining.

The portable verifier requires only Python and NumPy, loads no pickle and
imports no repository implementation. It independently recomputes raw metrics,
learning-curve areas, seed differences, descriptive bootstrap intervals,
qualification and fixed decision rules. Physical simulation and saved-model
checks are bound to the archived local audit and its stated coverage. Portable
verification establishes artifact consistency, not an independent learning
replication or an external cryptographic signature.

Portable verifier and test sources are packaging work, prospectively excluded
from the scientific lock and sealed separately in `verifier_source.zip` and
`verifier_lock.json`. Each finished cohort has a strict `artifact_manifest.json`.
Use `-B` to avoid introducing cache files into an archive:

```powershell
.venv/Scripts/python.exe -I -B reports/core_residual/smoke/verify_archive.py --input reports/core_residual/smoke
.venv/Scripts/python.exe -I -B reports/core_residual/development/verify_archive.py --input reports/core_residual/development
```

The same commands work with a copied cohort outside this repository, using an
interpreter with NumPy installed. See the [reproduction guide](../../docs/reproduction.md)
for new runs; do not overwrite the completed cohorts.

The sole primary candidate is separate versus joint, per architecture. Every
qualification and retention guard remains part of that decision. Attribution
against fixed random visual features cannot promote a failed candidate.
The experiment tests one allocation and one new dependency followed by a return;
it does not establish automatic capacity renewal, unlimited new representations,
noise robustness, a longer physical planning horizon or general continual learning.
