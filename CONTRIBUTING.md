# Contributing

This is an exploratory continual-learning research repository. Correct, reproducible negative results are useful contributions. Preserve the distinction between an implementation check, a mechanism diagnostic, and evidence of algorithmic improvement.

## Local setup and checks

Follow the platform-specific [README setup](README.md#setup). Use the virtual environment's interpreter; these commands show Windows. On Linux/macOS, substitute `.venv/bin/python`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m acp_cl run --config configs/smoke.json --output runs/contribution_smoke --device cpu
.\.venv\Scripts\python.exe -m acp_cl run --config configs/v3_smoke.json --output runs/contribution_v3_smoke --device cpu
```

Choose new smoke output paths for a new configuration or source revision. `--resume` is for an unchanged experiment. The [CPU CI workflow](.github/workflows/ci.yml) performs these classes of checks on Ubuntu with Python 3.10; it does not run the scientific suites or reproduce archived GPU trajectories.

Add focused tests for changes to optimizer displacement, recycling state, data pairing, information access, metrics, or checkpoint behavior. Tests should exercise an observable scientific contract, not merely repeat an implementation. Inspect generated plots when their rendering changes. State the actual checks performed and their limits; do not copy an old passing test count into a new change.

## Preserve experiment provenance

- Make learning-rule changes in a new revision and use a new output directory. Do not edit sources while a scientific suite is running.
- Freeze development choices before the locked evaluation cohort. Keep failed candidates, exclusions, and negative outcomes in the study ledger.
- Preserve original configurations, source/runtime hashes, per-seed results, and raw artifacts. Do not relabel prior runs as results of newer code.
- Keep evaluation measurements outside learner/controller feedback. Ordinary learners must not receive task boundaries or inference task IDs.
- Keep shared initialization, training exposure, replay membership, and sampling RNGs comparable. Audit actual reset counts and any claimed gain budget.
- Distinguish unit replacements from reset events, pre-clipping nominal gains from achieved gains, and equal exposure from equal compute.
- Export portable report copies with a mapping back to original artifact hashes. Changing presentation paths must not silently change historical results or overwrite originals.

Large datasets, checkpoints, and local run directories remain ignored. Compact results intended for review should explain which raw artifacts were omitted and how to regenerate or obtain them. Use only trusted local checkpoint files for resume and audit commands.

## Describe changes for review

Explain the concrete problem, resulting behavior, and why the comparison remains interpretable. Include the checks run, affected study/configuration names, and whether existing results require a separate revision or rerun. Document departures from cited algorithms, including the CBP-inspired recycling utility and SI-inspired importance rule; the [research review](docs/research_review.md) records relevant predecessors.

The project is released under the [MIT License](LICENSE). By contributing, you agree that your contributions are licensed under the same terms. Record real code/data sources and confirmed contributor information; do not assume that a public repository grants open-source rights. See [public readiness](docs/public_readiness.md) for the publication record and validation scope.
