# Adaptive Critical-Period Continual Learning

A PyTorch research prototype, evidence review, and falsifiable experiment plan for a critical-period-inspired continual-learning hypothesis. **The hypothesis is unproven.** The repository preserves negative results and distinguishes software checks, exploratory studies, and proposed confirmatory work.

The [completed v2 study](reports/v2_results.md) contains 102 comparative runs, 972 per-experience scratch fits, and an exact reproduction of one late-training failure. V2 did not establish an advantage over replay plus recycling. The [completed v3 study](reports/v3_results.md) adds 67 comparative runs with separate development and evaluation seeds. Its newborn-gain variant improves the primary late acquisition AUC by 1.15 percentage points on three new seeds; the nominal-budget-matched variant improves it by 0.92 points. Most of the improvement comes from one seed, and the combined policy has tradeoffs. This is an exploratory signal, not an established algorithmic advantage.

The [completed CIFAR-10 transfer pilot](reports/cifar10_transfer_results.md)
adds 16 fixed development runs on natural images. Newborn gain changes recurring
late AUC by -0.12 pp and final accuracy by -3.60 pp versus recycling; nominal
matching gives +0.18 pp AUC with mixed seed results. Adequate baselines learned,
but the predeclared signal and retention requirements failed. **The decision is
to stop scaling this frozen recipe.** No novelty or general efficacy is claimed.

Start with the [research review](docs/research_review.md), [implemented algorithm](docs/algorithm.md), and [experiment protocol](docs/experiment_protocol.md). The original report is preserved in [docs/deep-research-report.md](docs/deep-research-report.md).

## Setup

Use Python 3.10 for the closest match to recorded experiments. Install from the repository root in a fresh virtual environment. These platform-specific commands follow the [official PyTorch 2.8.0 instructions](https://pytorch.org/get-started/previous-versions/#v280).

Windows, CPU only:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Linux, CPU only:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"
```

macOS on Apple Silicon uses PyPI wheels, without a CUDA or Linux/Windows CPU index:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0
.venv/bin/python -m pip install -e ".[dev]"
```

For CUDA 12.8 on Windows or Linux, replace the CPU index in that platform's command with `https://download.pytorch.org/whl/cu128`. CUDA requires a compatible NVIDIA driver. Use `--device cpu` on macOS; this runner does not select MPS. The archived image experiments used Windows, Python 3.10.11, PyTorch 2.8.0+cu128, NumPy 2.2.6, and an RTX 4070 Ti. These setup instructions do not claim bitwise numerical reproduction on other systems.

[requirements-lock.txt](requirements-lock.txt) records that Windows/CUDA package environment, including `+cu128` versions. It is not a portable CPU or macOS lockfile. To install the recorded versions in a fresh Windows/CUDA environment, first install Torch/torchvision from the CUDA index as above, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt -e ".[dev]"
```

Python itself, drivers, hardware, and source revision are separate requirements. Reproducing archived results requires their recorded source and runtime. Ordinary CPU setup and CI exercise the dependency ranges in `pyproject.toml`, not the archived CUDA environment. Dataset and generated training files live in ignored `data/` and `runs/` directories.

## Run

Commands below use the Windows interpreter. On Linux/macOS, substitute `.venv/bin/python` and use forward slashes in repository paths.

The completed [v2 protocol](docs/experiment_v2.md) covers local critical periods for replaced units, a frozen input drift sensor, oracle and reset/gain controls, and fixed-label recurring shape domains. Running an archived configuration with current code creates a new experiment; the [v2 report](reports/v2_results.md) identifies the historical training revision.

```powershell
.venv\Scripts\python.exe -m acp_cl run --config configs\v2_shapes_development.json --output runs\my_v2_development --device cuda
.venv\Scripts\python.exe -m acp_cl run --config configs\v2_shapes_long.json --output runs\my_v2_long --device cuda
```

The corresponding `v2_gaussian.json`, `v2_shapes_stationary.json`, and
`v2_shapes_noise_only.json` configurations define the geometry audit and
100-experience negative controls. These datasets require no downloads.
Yoked controls must run after `acp_v2` for the same seed; the suite orders them
automatically. Oracle and yoked results are labeled extra-information diagnostics.

```powershell
# Small deterministic engineering check, no downloads or GPU required
.venv\Scripts\python.exe -m acp_cl run --config configs\smoke.json --output runs\my_smoke --device cpu

# Paired development pilot, eight synthetic experiences and five seeds
.venv\Scripts\python.exe -m acp_cl run --config configs\synthetic_pilot.json --output runs\my_synthetic --device cpu

# Documented calibration: activates closure/reopening on three development seeds
.venv\Scripts\python.exe -m acp_cl run --config configs\synthetic_calibrated.json --output runs\my_calibrated --device cpu

# Real images: twenty CIFAR-100 classes, small CNN, three development seeds
.venv\Scripts\python.exe -m acp_cl run --config configs\cifar100_pilot.json --output runs\my_cifar --device cuda

# Full 100-class ResNet development template; substantially more compute
.venv\Scripts\python.exe -m acp_cl run --config configs\cifar100_resnet18.json --output runs\my_resnet --device cuda
```

The CIFAR commands download the official dataset if absent. If the Toronto origin's single connection is slow, first run `.venv\Scripts\python.exe scripts\download_cifar100.py`; it fetches canonical byte ranges and verifies torchvision's official archive checksum.

Override methods or seeds with `--methods er er_recycle acp --seeds 11 22 33`. Add `--resume` to the **same command/output** to continue completed-experience checkpoints or reuse completed results. Changed configurations, source versions, execution devices, or recorded runtimes require a fresh output directory. Only resume trusted local checkpoints. Analyze existing results with `.venv\Scripts\python.exe -m acp_cl analyze runs\my_cifar`.

Each suite writes `manifest.json`, per-method/seed results, allocation traces, events, checkpoints, paired bootstrap summaries, CSV, and PNG/SVG plots. When `scratch_reference` is enabled, a separate fresh-model diagnostic trains on each experience's identical current-data sequence; v3 disables it. Reported studies evaluate **validation** images and keep test data outside development decisions.

`configs/resnet_gpu_smoke.json` runs twenty updates per method with all 100 output labels to check GPU integration quickly. It is intentionally too short for an accuracy conclusion.

## Completed v3 study

The [v3 report](reports/v3_results.md), [protocol](docs/experiment_v3.md), [allocation rules](docs/algorithm_v3.md), and [machine-readable study plan](configs/v3_study.json) separate development from locked evaluation. `v3_study.json` is an orchestration plan, not an individual `acp_cl run --config` file.

Check all seven v3 methods locally without downloading data:

```powershell
.venv\Scripts\python.exe -m acp_cl run --config configs/v3_smoke.json --output runs/my_v3_smoke --device cpu
```

To prepare the development grid:

```powershell
.venv\Scripts\python.exe scripts/prepare_v3_study.py prepare-development --spec configs/v3_study.json --configs configs/v3/development
```

Run all eight generated JSON configurations through the ordinary CLI, using `runs/v3_development/<configuration-stem>` for each output. For example:

```powershell
.venv\Scripts\python.exe -m acp_cl run --config configs/v3/development/gain_005_recurring.json --output runs/v3_development/gain_005_recurring --device cuda
```

After the complete development grid finishes, select using development results only:

```powershell
.venv\Scripts\python.exe scripts/prepare_v3_study.py select --spec configs/v3_study.json --results runs/v3_development --output configs/v3/reproduction_lock
```

Selection records the attempted candidates and emits three locked configurations only if the prespecified screen passes. The committed `configs/v3/locked` directory preserves the original selection; use a fresh lock directory for an independent reproduction because result-byte hashes include runtime and timing. Freeze the new `selection.json` and its configurations in local git before evaluating the new seeds. Run each of `recurring.json`, `stationary.json`, and `early_biased.json` with its matching `runs/v3_reproduction/<condition>` output; for example:

```powershell
.venv\Scripts\python.exe -m acp_cl run --config configs/v3/reproduction_lock/recurring.json --output runs/v3_reproduction/recurring --device cuda
```

The plan screens four gains for replay plus recycling on two development seeds, then locks one shared gain before evaluating three different seeds. The completed study selected 0.5 and contains 67 comparative runs: 16 development, 42 recurrent/stationary, and nine early-color-bias runs. Colors are independent of labels during ordinary training and all evaluation; the biased curriculum changes an initial training segment. All attempted development candidates, per-seed results, and the separate engineering smokes are recorded in the [inventory](reports/v3/run_inventory.json).

| Method | Intervention |
|---|---|
| `er_v3` | Replay and a fixed feature gain, without recycling |
| `recycle_v3` | Add a fixed reset-count/time schedule |
| `newborn_v3` | Add a local gain window after replacement |
| `newborn_matched_v3` | Redistribute mature gain to match the baseline's nominal feature budget before clipping |
| `protection_v3` | Extend replacement eligibility age |
| `consolidation_v3` | Add local consolidation after maturation |
| `full_v3` | Combine gain, protection, and consolidation |

These are individual-component comparisons against a shared baseline plus a combined condition, not a full factorial or isolated leave-one-out study. All conditions share a warmup and optimizer-displacement cap. Keeping mature gain unchanged and matching total nominal gain are different comparisons; post-clipping gains and actual displacements need not match. The protocol defines selection, measurements, and failure reporting. The report preserves the modest positive primary result, negative final-accuracy contrasts, stationary ceiling, and seed-dependent early-bias interaction.

## CIFAR-10 transfer pilot

The [transfer protocol](docs/cifar10_transfer_protocol.md) carries the v3 recipe
to natural images with fixed labels and one current arrival per training image.
It compares replay, recycling, newborn gain, and nominal-budget-matched newborn
gain on recurring original/grayscale/blur domains and a stationary control.
The [pre-run lock](configs/cifar10_transfer/lock.json) binds the source,
configuration, runtime, two development seeds, and a 16-run budget. Its
continuation rule is an effort-allocation decision, not a significance test.

The [completed report](reports/cifar10_transfer_results.md),
[all seed outcomes](reports/cifar10_transfer/summary.md), and
[independent audit](reports/cifar10_transfer/independent_audit.md) preserve the
negative result. Both seeds and every planned method are included.

Prepare the official training archive once; the constructor below downloads and
extracts it if needed. Official test images and labels are not evaluated.

```powershell
.venv/Scripts/python.exe -c "from torchvision.datasets import CIFAR10; CIFAR10(root='data', train=True, download=True)"
.venv/Scripts/python.exe -m acp_cl run --config configs/cifar10_transfer/recurring.json --output runs/my_transfer/recurring --device cuda --no-plots
.venv/Scripts/python.exe -m acp_cl run --config configs/cifar10_transfer/stationary.json --output runs/my_transfer/stationary --device cuda --no-plots
```

If the download is slow, `scripts/download_cifar100.py --dataset cifar10`
provides the checksum-verified ranged download before the constructor extracts
it. The separate `configs/cifar10_transfer_smoke.json` checks all four methods
quickly on already downloaded data; its accuracy is not evidence of efficacy.
Offline stream tests use fixtures and require no dataset download.

Use the study-specific analysis instead of generic bootstrap summaries:

```powershell
.venv/Scripts/python.exe scripts/summarize_cifar10_transfer.py --runs runs/my_transfer --output runs/my_transfer_report
.venv/Scripts/python.exe scripts/audit_cifar10_transfer.py --results runs/my_transfer --summary runs/my_transfer_report/summary.json --output runs/my_transfer_report
.venv/Scripts/python.exe scripts/plot_cifar10_transfer.py --archive runs/my_transfer_report/individual_results.json --output runs/my_transfer_report
```

These commands use a fresh report directory. The committed archive preserves
the original outcomes and raw-byte identities. Plotting alone can use that
archive without downloading data or loading checkpoints. Numerical audits use
trusted local raw runs and their recorded lock/source/runtime.

Source was frozen at `c250129` and the run lock at `3f72cd8`. A new runtime or
changed recipe requires a separate output and a newly declared study identity.
The earlier v3 archive, including hashes of then-current root documentation,
remains the historical snapshot at `a038786`; subsequent root documentation
does not replace that sealed snapshot.

## Comparisons

| Command method | Behavior |
|---|---|
| `finetune` | Current data only |
| `er` | Fixed-budget reservoir replay |
| `er_recycle` | Replay plus the same low-utility recycling used by ACP; CBP-inspired |
| `derpp` | Replay labels plus stored-logit MSE; documented DER++ variant |
| `fixed` | Initial broad learning, scheduled closure, no later reopening |
| `acp` | Signal-driven closure, bounded local reopening, consolidation, replay, recycling |
| `acp_no_consolidation`, `acp_no_replay`, `acp_no_recycling` | Component-removal controls |
| `acp_global`, `acp_no_relaxation`, `acp_no_reopening` | Locality and reopening controls |
| `acp_no_damage`, `acp_no_health`, `acp_random_recycling` | Sensor and replacement-selection controls |
| `acp_v2` | Local newborn periods, frozen input drift, fixed initial schedule, mature reopening |
| `acp_v2_no_newborn`, `acp_v2_no_reopening`, `acp_v2_learned_sensor` | V2 mechanism controls |
| `acp_v2_oracle` | Extra-information diagnostic with true signal-domain change times |
| `er_recycle_yoked`, `er_recycle_yoked_gain` | Extra-information controls with v2's recorded reset schedule; optional constant matched mean gain |

The head is always trainable; no task ID or per-task output mask is used. Image models include an explicit recyclable adapter so channel resetting does not break residual/GroupNorm dependencies. The optimizer gates momentum displacement and separately tracks consolidation anchors and importance-window origins. See the algorithm document for exact departures from the original proposal and cited methods.

The important endpoints are final accuracy, forgetting, and late-stream early-learning AUC. AUC includes retained knowledge and transfer as well as adaptation. Where measured, scratch-adjusted AUC is a diagnostic rather than pure forward transfer. AUC divided by the first experience's AUC is deliberately omitted. Small-seed bootstrap intervals are exploratory; sparse forgetting can miss intervening damage, and equal data exposure is not equal compute.

The ingredients have substantial predecessors: mature low-utility replacement in [continual backpropagation](https://www.nature.com/articles/s41586-024-07711-7), importance-based protection in [Synaptic Intelligence](https://proceedings.mlr.press/v70/zenke17a.html), combined protection and plasticity in [UPGD](https://arxiv.org/abs/2404.00781), and persistent early-experience effects in [critical learning periods](https://arxiv.org/abs/1711.08856). This repository studies particular combinations and controls. It does not claim discovery of those ideas, biological equivalence, or a validated solution to continual learning.

[Closer maturation and neurogenesis precedents](docs/algorithm_v3.md#close-computational-precedents) also include Neurogenesis Deep Learning, NICE, and a neuromorphic lifelong-learning preprint with preferential newborn plasticity and age-gated pruning.

## Current scope

The [v2 results](reports/v2_results.md) cover 102 comparative runs, including
100-experience recurrent, stationary, and noise-only streams. V2 does not
establish an improvement over replay plus recycling. A constant matched-gain
diagnostic is much stronger and retains its advantage when color shortcuts
are removed. The report includes an exact reproduction of late update
instability, all negative results, and the proposed next comparison.

The [v3 results](reports/v3_results.md) complete that comparison. The completed
[natural-image follow-up](reports/cifar10_transfer_results.md) uses fixed labels,
single-pass CIFAR-10 arrivals and paired recurring/stationary conditions. Its
16 development runs fail the predeclared signal and retention requirements;
the decision is to stop scaling the frozen recipe.

Correctness tests cover phase transitions, finite-update validation, momentum gating, maturity/protection, adjacent recycling resets, replay RNG isolation, metrics, all method variants, evaluation isolation, exact checkpoint continuation, and provenance rejection. The initial synthetic pilot stayed open throughout: its high accuracy therefore does **not** validate critical-period control. Subsequent calibration and real-image results are reported separately in [reports/development_results.md](reports/development_results.md).

Not yet completed: publication-grade baseline reproductions (including UPGD), matched tuning/compute studies, a preregistered confirmatory cohort, CORe50/Infinite dSprites adapters, and independent long-horizon replication on established benchmarks. Current image-model recycling is limited to the final adapter. Replay-damage feedback is sampled and delayed, not an old-knowledge guarantee. These limitations define the next experiments rather than a successful scientific claim.

## Contributing and public readiness

See [CONTRIBUTING.md](CONTRIBUTING.md) for local tests, Ruff, and experiment-preservation rules. The current CPU suite passes 546 tests with three platform/device skips, and the CIFAR-10 engineering check covers all four methods on CPU and GPU. The historical v3 revision also passed a [fresh noneditable clean-clone smoke](reports/v3/clean_clone_check.md) for all seven methods. The [CPU workflow](.github/workflows/ci.yml) defines checks plus download-free smokes; it has not been run on GitHub. [Public-readiness notes](docs/public_readiness.md) record readiness to share research in progress and remaining release decisions. A license, publication account/repository, and visibility still require the owner's choice. No license or author identity is inferred here.
