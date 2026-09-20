# Public-readiness review

This repository can be prepared as a transparent research work in progress even if every new hypothesis is negative. That is different from establishing a novel, effective continual-learning algorithm. The completed [v2 report](../reports/v2_results.md) preserves negative results; [v3](experiment_v3.md) is an exploratory follow-up with separate selection and evaluation seeds. A synthetic three-seed assessment is not established-benchmark or confirmatory evidence.

This document records local preparation. It does not record a GitHub publication or a successful hosted CI run.

## Threshold for sharing research work in progress

A reviewable release should contain:

1. An explicit statement of implemented behavior, unproven hypotheses, limitations, and completed versus planned work.
2. The original research attribution and close computational predecessors. Recycling, importance-based protection, and critical learning periods are not new ingredients; claims concern the tested combinations and controls.
3. Exact study plans, development-selection rules and outputs, seed identities, attempted-run inventories, and all negative or failed outcomes. Preserve the original source/runtime identities.
4. Reproducible setup and commands, a clean-environment CPU smoke run, relevant tests and lint, and a clear account of which archived numerical results require their historical GPU environment.
5. Portable artifact references, an explanation of omitted raw data/checkpoints, and enough per-seed evidence to check the reported comparisons.
6. An owner-selected license and confirmed destination/visibility, with attribution for any redistributed third-party assets.

Scientific success is not a condition for sharing. A claim of algorithmic benefit requires additional evidence: meaningful gains against appropriately tuned baselines, active and interpretable component controls, independent seeds, established benchmarks, and reported retention/plasticity/resource tradeoffs. The [NeurIPS reproducibility guidance](https://neurips.cc/public/guides/PaperChecklist) similarly separates accurate claims, experimental details, uncertainty, compute disclosure, and access to reproduction materials.

## Preparation already present

- The README provides separate Windows/Linux CPU, Windows/Linux CUDA, and Apple Silicon macOS installation paths, following the [official PyTorch 2.8.0 commands](https://pytorch.org/get-started/previous-versions/#v280).
- `requirements-lock.txt` identifies the recorded Windows/CUDA package versions. It is explicitly distinguished from portable CPU setup and from the separately required Python, driver, hardware, and source revision.
- The local [CPU CI definition](../.github/workflows/ci.yml) runs tests, Ruff, and legacy/v3 download-free smoke configurations. No scientific suite or publication step is included.
- Source/runtime identity checks protect resume; report analysis can read uniformly compatible historical artifacts.
- [Contribution guidance](../CONTRIBUTING.md) describes preservation of scientific artifacts and the evidence expected for changes.

These files are preparation, not a claim that a hosted workflow or the full v3 study has already passed. Record those outcomes when actually obtained. Local setup/smoke validation and scientific results should name the revision and environment checked; neither demonstrates cross-platform bitwise reproduction. Existing v2 numerical claims remain attached to their recorded training revision, not to this documentation change.

The CI action releases and full commit pins were checked against primary GitHub release/tag APIs on 2026-09-20:

| Action release | Commit pin |
|---|---|
| [actions/checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| [actions/setup-python v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0) | `5fda3b95a4ea91299a34e894583c3862153e4b97` |

The workflow uses a GitHub-hosted Ubuntu 24.04 runner, Python 3.10, CPU Torch wheels, read-only repository contents permission, and no persisted checkout credentials. It runs on ordinary push/pull-request or manual workflow triggers; it contains no remote publishing operation.

## Portable-path issues in archived report copies

The review found **399 absolute Windows path strings in nine archived v2 JSON files**. These paths name the local checkout and its ignored `runs/` artifacts; they are neither portable download locations nor files that will exist in another clone. The recorded metrics and artifact hashes are separate from these presentation paths.

| Archived file | Absolute-path fields | Strings |
|---|---|---:|
| `reports/v2/diagnostics/v2_gaussian/diagnostics.json` | `source_directory`; `runs[*].source_files.{allocation,events,result}.path` | 91 |
| `reports/v2/diagnostics/v2_shapes_development/diagnostics.json` | Same fields | 91 |
| `reports/v2/diagnostics/v2_shapes_long/diagnostics.json` | Same fields | 73 |
| `reports/v2/diagnostics/v2_shapes_noise_only/diagnostics.json` | Same fields | 28 |
| `reports/v2/diagnostics/v2_shapes_stationary/diagnostics.json` | Same fields | 28 |
| `reports/v2/reset_damage/endpoint_audit.json` | `suite`; `runs[*].source_files.{events.json,result.json}.path` | 19 |
| `reports/v2/reset_damage/instrumented_audit.json` | `plan.original_run` | 1 |
| `reports/v2/shortcuts_development/shortcut_audit.json` | `run_directory`; `runs[*].{checkpoint_path,result_path}` | 19 |
| `reports/v2/shortcuts_long/shortcut_audit.json` | Same fields | 49 |

For a public artifact bundle, create separate export copies with repository-relative or bundle-relative locators. Retain untouched originals and their hashes, and record the mapping and export hashes. Explain whether each referenced raw artifact is included, locally retained, or reproducible from the recorded revision. Do not rewrite history or silently replace a historical file's identity. This review has not altered those report copies.

Committed summary tables and diagnostic extracts are not a complete raw-run archive. The normal `analyze` command expects per-run `result.json` files; point readers to a suitable export or a reproduction command, not merely a summary JSON. Checkpoint distribution is optional for a source-based reproduction, but its omission must be stated.

## Remaining owner decisions and release checks

- **License:** no project distribution license has been chosen. Confirm the owner's preferred code/documentation terms and any asset exceptions before adding a license. A public repository alone does not make code open source; see [GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
- **Destination:** confirm the GitHub account or organization, repository name, and visibility. No remote mutation or publication is part of this preparation.
- **Attribution:** confirm real author/contributor names and citation metadata. Do not invent an owner for `CITATION.cff` or apply the repository's eventual license to third-party work by implication.
- **Artifacts:** decide what raw or compact evidence to distribute, prepare portable exports, and document the source revision needed for each study.
- **Validation:** record clean-environment setup and smoke outcomes, and the CPU workflow's actual result when hosted. Distinguish these from numerical reproduction of archived CUDA studies.

The v3 selection rule and locked-cohort interpretation remain defined by its protocol, regardless of release timing. A failed development stability screen or a null locked comparison should be released as such, not hidden by selecting another cohort or promoting a favorable diagnostic to the primary result.
