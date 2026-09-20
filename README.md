# Adaptive Critical-Period Continual Learning

A runnable PyTorch research prototype, evidence review, and falsifiable experiment plan for the attached critical-period hypothesis. All work is local. **The hypothesis is unproven.** Early pilots are software/feasibility experiments, not a confirmatory study.

Start with the [research review](docs/research_review.md), [implemented algorithm](docs/algorithm.md), and [experiment protocol](docs/experiment_protocol.md). The original report is preserved in [docs/deep-research-report.md](docs/deep-research-report.md).

## Setup

Tested locally with Python 3.10, PyTorch 2.8.0, torchvision 0.23.0, and an RTX 4070 Ti. CUDA wheels below follow the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v280). Python 3.10–3.12 is a practical choice for these pinned framework versions.

```powershell
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
```

For CPU-only installation, change the PyTorch index suffix from `cu128` to `cpu`. On Linux/macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`; the experiment code is portable. Dataset and generated training files live in ignored `data/` and `runs/` directories.

## Run

```powershell
# Small deterministic engineering check, no downloads or GPU required
.venv\Scripts\python.exe -m acp_cl run --config configs\smoke.json --output runs\my_smoke --device cpu

# Paired development pilot, eight synthetic experiences and five seeds
.venv\Scripts\python.exe -m acp_cl run --config configs\synthetic_pilot.json --output runs\my_synthetic --device cpu

# Real images: twenty CIFAR-100 classes, small CNN, three development seeds
.venv\Scripts\python.exe -m acp_cl run --config configs\cifar100_pilot.json --output runs\my_cifar --device cuda

# Full 100-class ResNet development template; substantially more compute
.venv\Scripts\python.exe -m acp_cl run --config configs\cifar100_resnet18.json --output runs\my_resnet --device cuda
```

The CIFAR commands download the official dataset if absent. If the Toronto origin's single connection is slow, first run `.venv\Scripts\python.exe scripts\download_cifar100.py`; it fetches canonical byte ranges and verifies torchvision's official archive checksum.

Override methods or seeds with `--methods er er_recycle acp --seeds 11 22 33`. Add `--resume` to the **same command/output** to continue completed-experience checkpoints or reuse completed results. Changed configurations, source versions, or execution devices require a fresh output directory. Only resume trusted local checkpoints. Analyze existing results with `.venv\Scripts\python.exe -m acp_cl analyze runs\my_cifar`.

Each suite writes `manifest.json`, per-method/seed results, controller events, checkpoints, paired bootstrap summaries, CSV, and PNG/SVG plots. A separate fresh-model diagnostic is trained on each experience using its identical current-data sequence. All shipped configurations evaluate **validation** images; the test split remains excluded from development decisions.

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

The head is always trainable; no task ID or per-task output mask is used. Image models include an explicit recyclable adapter so channel resetting does not break residual/GroupNorm dependencies. The optimizer gates momentum displacement and separately tracks consolidation anchors and importance-window origins. See the algorithm document for exact departures from the original proposal and cited methods.

The important endpoints are final accuracy, forgetting, and late-stream early-learning AUC. Scratch-adjusted AUC helps separate changing experience difficulty from adaptation performance. AUC divided by the first experience's AUC is deliberately omitted. Five-seed bootstrap intervals are exploratory and do not replace the planned independent 20-pair confirmatory cohort.

## Current scope

Correctness tests cover phase transitions, finite-update validation, momentum gating, maturity/protection, adjacent recycling resets, replay RNG isolation, metrics, all method variants, evaluation isolation, exact checkpoint continuation, and provenance rejection. The initial synthetic pilot stayed open throughout: its high accuracy therefore does **not** validate critical-period control. Subsequent calibration and real-image results are reported separately in [reports/development_results.md](reports/development_results.md).

Not yet completed: publication-grade baseline reproductions (including UPGD), matched tuning/compute studies, a preregistered confirmatory cohort, CORe50/Infinite dSprites adapters, and long-horizon replication. Current image-model recycling is limited to the final adapter. Replay-damage feedback is sampled and delayed, not an old-knowledge guarantee. These limitations define the next experiments rather than a successful scientific claim.
