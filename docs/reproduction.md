# Setup, reproduction, and method reference

The [project README](../README.md) summarizes the hypothesis and outcomes. This
guide preserves the detailed installation, command, and comparison reference.
Run commands from the repository root, even though this guide lives in `docs/`.
Archived numerical results require their recorded source and runtime.

## Causal-access experiment

The [fixed protocol](causal_access_protocol.md) uses fresh seeds 17001--17006,
both original architectures and one lagged probability mixer. It fits models
from scratch, then freezes all parameters, gradients, optimizer and replay state
while only external causal support and the mixer's score window evolve.
The reduced smoke is an engineering check, not a scientific pilot.

```powershell
.venv\Scripts\python.exe -m acp_cl.causal_access.study --config configs/causal_access_smoke.json --output runs/my_causal_access_smoke --protocol docs/causal_access_protocol.md
.venv\Scripts\python.exe -m acp_cl.causal_access.study --config configs/causal_access_development.json --output runs/my_causal_access_development --protocol docs/causal_access_protocol.md --lock-only
.venv\Scripts\python.exe -m acp_cl.causal_access.study --config configs/causal_access_development.json --output runs/my_causal_access_development --protocol docs/causal_access_protocol.md --resume
.venv\Scripts\python.exe scripts/summarize_causal_access.py --input runs/my_causal_access_development --output reports/my_causal_access/summary.json --development
.venv\Scripts\python.exe scripts/plot_causal_access.py --summary reports/my_causal_access/summary.json --output reports/my_causal_access
```

Run from the repository root using the recorded runtime. Resume requires the
same source, protocol, configuration, scientific analysis and execution settings;
it reuses completed fits and does not add training. The current runner fixes
six single-thread CPU workers and Windows affinity 1365 for development.
Do not launch a second copy while workers are active. Artifacts live under
`jobs/{model}_{seed}`; each contains full fit checkpoints, pre-feedback forecasts,
raw physical truth, supports, mixture weights and hashes. The independent NumPy
scorer checks the complete cohort and reconstructs every causal mixture.
The saved main source/analysis archives are the record of this study; future
changes to live code require a separate run. The scientific configuration is
deliberately limited to the declared experiment, with no tuning CLI.

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

[requirements-lock.txt](../requirements-lock.txt) records that Windows/CUDA package environment, including `+cu128` versions. It is not a portable CPU or macOS lockfile. To install the recorded versions in a fresh Windows/CUDA environment, first install Torch/torchvision from the CUDA index as above, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt -e ".[dev]"
```

Python itself, drivers, hardware, and source revision are separate requirements. Reproducing archived results requires their recorded source and runtime. Ordinary CPU setup and CI exercise the dependency ranges in `pyproject.toml`, not the archived CUDA environment. Dataset and generated training files live in ignored `data/` and `runs/` directories.

## Run

Commands below use the Windows interpreter. On Linux/macOS, substitute `.venv/bin/python` and use forward slashes in repository paths.

The completed [v2 protocol](../docs/experiment_v2.md) covers local critical periods for replaced units, a frozen input drift sensor, oracle and reset/gain controls, and fixed-label recurring shape domains. Running an archived configuration with current code creates a new experiment; the [v2 report](../reports/v2_results.md) identifies the historical training revision.

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

The [v3 report](../reports/v3_results.md), [protocol](../docs/experiment_v3.md), [allocation rules](../docs/algorithm_v3.md), and [machine-readable study plan](../configs/v3_study.json) separate development from locked evaluation. `v3_study.json` is an orchestration plan, not an individual `acp_cl run --config` file.

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

The plan screens four gains for replay plus recycling on two development seeds, then locks one shared gain before evaluating three different seeds. The completed study selected 0.5 and contains 67 comparative runs: 16 development, 42 recurrent/stationary, and nine early-color-bias runs. Colors are independent of labels during ordinary training and all evaluation; the biased curriculum changes an initial training segment. All attempted development candidates, per-seed results, and the separate engineering smokes are recorded in the [inventory](../reports/v3/run_inventory.json).

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

The [transfer protocol](../docs/cifar10_transfer_protocol.md) carries the v3 recipe
to natural images with fixed labels and one current arrival per training image.
It compares replay, recycling, newborn gain, and nominal-budget-matched newborn
gain on recurring original/grayscale/blur domains and a stationary control.
The [pre-run lock](../configs/cifar10_transfer/lock.json) binds the source,
configuration, runtime, two development seeds, and a 16-run budget. Its
continuation rule is an effort-allocation decision, not a significance test.

The [completed report](../reports/cifar10_transfer_results.md),
[all seed outcomes](../reports/cifar10_transfer/summary.md), and
[independent audit](../reports/cifar10_transfer/independent_audit.md) preserve the
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

[Closer maturation and neurogenesis precedents](../docs/algorithm_v3.md#close-computational-precedents) also include Neurogenesis Deep Learning, NICE, and a neuromorphic lifelong-learning preprint with preferential newborn plasticity and age-gated pruning.


## Verification and historical snapshots

The earlier image-study CPU suite passed 546 tests with three platform/device skips.
Eight separate CIFAR-10 engineering runs covered all four methods on CPU and
GPU. The older v3 revision also passed its clean-clone installation check.
These checks verify implementation; they do not establish efficacy or
cross-platform bitwise reproduction. See the individual study reports for
run identities and independent local audits.

The v3 artifact seal refers to commit `a038786`. The CIFAR-10 transfer seal
refers to `5985b77`. Both include the root documentation as it stood at those
commits; later README edits do not replace those historical snapshots.

## Continual Relational Learning: matched-history comparison

The research umbrella is now Continual Relational Learning. The installed
package remains `acp_cl`, and the earlier `acp-cl` commands remain compatible.
See the [accepted charter](research_charter.md) and
[matched-history protocol](contextual_comparison.md).

The new runner shares each learned prefix across three independent branches:
returning conditions, a new visible dependency, and feedback noise. Its two
absence schedules contain exactly the same training records in different orders.
The locked pilot uses eight seeds, six methods, two gap orders, and three branches:
96 prefix fits and 288 branches. Twelve updates per incoming batch were selected
by the finite comparator-development grid, using different seeds and excluding
the candidate from setting selection.

```powershell
.venv/Scripts/python.exe -m acp_cl.contextual.study --config configs/contextual_smoke.json --output runs/contextual_smoke_new --device cpu
.venv/Scripts/python.exe -m acp_cl.contextual.study --config configs/contextual_pilot.json --output runs/contextual_reproduction --device cpu --protocol docs/contextual_comparison.md
.venv/Scripts/python.exe scripts/summarize_contextual.py --input runs/contextual_reproduction --output reports/contextual_reproduction
.venv/Scripts/python.exe scripts/audit_contextual.py --input runs/contextual_reproduction --output reports/contextual_reproduction/audit.json
.venv/Scripts/python.exe scripts/plot_contextual.py --input reports/contextual_reproduction
```

Use a fresh output directory for a new experiment. After interruption, add
`--resume` with the identical config, source, and runtime. Prefix checkpoints
are saved after each prefix block; branch checkpoints are saved every 256
arrivals. Resuming repeats unfinished work, never skips it. Only load trusted
local PyTorch checkpoints. Cross-platform bitwise reproduction is not promised.

Every development attempt is preserved under `reports/contextual/development/`,
including its config, full outcomes, source archive, and checkpoint audit.
The old plain-GRU configs predate the `interaction_features` field and need
their own archived training source; do not silently retrofit them to the latest
runner. To audit such a locally retained run with its exact source:

```powershell
.venv/Scripts/python.exe scripts/audit_contextual.py --input runs/contextual_development_u3 --output reports/contextual_development_u3_audit.json --archived-source runs/contextual_audit_sources/reproduction_plain
```

`--archived-source` checks every source-file hash, writes it to the named
directory, and launches a separate process with that source on `PYTHONPATH`.
For retraining an older recipe, use that extracted source as `PYTHONPATH` and
its archived `config.json`; use a fresh output directory. Full checkpoints stay
under the ignored `runs/` tree, while the portable reports include exact
training sources and all outcomes. Two development seeds cannot establish
candidate superiority; the fresh cohort must independently pass adequacy checks.

## Continued acquisition diagnostic

The [acquisition protocol](acquisition_diagnostic.md) adds three predictive
dependencies to the same conserving world. Their image signals are present but
irrelevant from the start; the physical rules later make each signal useful.
The original conditional and qualified recurrent learners remain unchanged.
At each introduction, compare continued learning, frozen visual features,
reset optimizer/replay with retained weights, and fresh random weights. Only
the continuing arm advances the trajectory. Final independent branches test
an exact return, an obsolete cue meaning, and noisy reports.

```powershell
.venv/Scripts/python.exe scripts/check_acquisition_world.py --output runs/acquisition_world_check_new.json
.venv/Scripts/python.exe -m acp_cl.acquisition.study --config configs/acquisition_smoke.json --output runs/acquisition_smoke_new --device cpu
.venv/Scripts/python.exe -m acp_cl.acquisition.study --config configs/acquisition_development_2048.json --output runs/acquisition_development_new --device cpu --protocol docs/acquisition_diagnostic.md
.venv/Scripts/python.exe scripts/summarize_acquisition.py --input runs/acquisition_development_new --output runs/acquisition_development_report
```

The original development used only fresh-model fits on seeds 211/212 and
selected 2,048 arrivals, the first passing entry in the declared exposure grid.
The 4,096 and 8,192 configuration files are unrun grid entries. A new independent
study must complete its own development and lock its settings before evaluating
comparison seeds. The commands below reproduce the selected original recipe:

```powershell
.venv/Scripts/python.exe -m acp_cl.acquisition.study --config configs/acquisition_diagnostic.json --output runs/acquisition_reproduction --device cpu --protocol docs/acquisition_diagnostic.md
.venv/Scripts/python.exe scripts/summarize_acquisition.py --input runs/acquisition_reproduction --output runs/acquisition_reproduction_report
.venv/Scripts/python.exe scripts/audit_acquisition.py --input runs/acquisition_reproduction --output runs/acquisition_reproduction_report/audit.json
.venv/Scripts/python.exe scripts/plot_acquisition.py --input runs/acquisition_reproduction_report
```

The comparison has six independent seeds, two architectures, 12 prefix fits,
144 acquisition arms, and 36 final branches. Episodes are not independent
replicates: acquisition contrasts average the three introductions within seed
before bootstrap. Primary error measures average predictions over actions and
survival horizons; this need not improve the best one-time transfer decision.
Fresh-model cue-use qualification must be rechecked in the comparison cohort.
The engineering smoke is too short for scientific qualification.

Use fresh output directories; append `--resume` to the identical training
command after interruption. Exact source, runtime and config identities are
required. The runner saves prefix checkpoints after each prefix block and
episode checkpoints every 256 arrivals, together with episode-start weights.
The audit regenerates performed-action records and replay state, verifies each
diagnostic fork, and recomputes initial/final probes. It requires trusted local
checkpoints and the exact source recorded in the run. Intermediate curves are
retained and checked against checkpoint records, but are not all independently
recomputed from intermediate weights. Plotting uses the portable summary only.

Original training sources, protocol locks, all episode records, development
selection, and audits are under `reports/acquisition/`; full model checkpoints
remain in the ignored `runs/` tree. Existing study archives stay unchanged.

## Optimizer-by-replay diagnostic

The [training-state protocol](training_state_diagnostic.md) separates optimizer
reset from replay reset, preserving the earlier world, architectures and update
loop. Each new-law onset forks four combinations plus a fresh-weight reference.
Final return, revision, clean continuation and noisy continuation branches each
contain the four factorial arms. Clean/noisy branches share observations and
performed actions, so their contrast isolates corrupted feedback under the
declared reset. This is a privileged diagnostic, not an autonomous reset policy.

```powershell
.venv/Scripts/python.exe -m acp_cl.training_state.study --config configs/training_state_smoke.json --output runs/training_state_smoke_new --device cpu
.venv/Scripts/python.exe -m acp_cl.training_state.study --config configs/training_state_development_4096.json --output runs/training_state_development_new --device cpu --protocol docs/training_state_diagnostic.md
.venv/Scripts/python.exe scripts/summarize_training_state.py --input runs/training_state_development_new --output runs/training_state_development_report
.venv/Scripts/python.exe scripts/audit_training_state.py --input runs/training_state_development_new --output runs/training_state_development_report/audit.json
```

Fresh-only development uses seeds 411--416 and tries 4,096 then 8,192 arrivals.
Run the second configuration only if the first fails. Stop if neither passes.
The development cue requirement (.0025) adds a margin above the unchanged
comparison requirement (.002); both also require marginal Brier improvement
of .02 in every stage and cue group for each architecture. Development outcomes
and audits are preserved under `reports/training_state/development_*`.
Only after development qualifies should the comparison configuration be locked
to the selected exposure. Its planned count is 12 prefixes and 372 episodes.

The completed development failed at 4,096 arrivals and selected 8,192, with
all 72 attempted fresh fits preserved. The comparison configuration below
contains that selected budget. Reproducing this recipe differs from selecting
settings for a new independent study; do not use comparison outcomes to alter
its exposure or qualification requirements.

```powershell
.venv/Scripts/python.exe -m acp_cl.training_state.study --config configs/training_state_diagnostic.json --output runs/training_state_reproduction --device cpu --protocol docs/training_state_diagnostic.md
.venv/Scripts/python.exe scripts/summarize_training_state.py --input runs/training_state_reproduction --output runs/training_state_reproduction_report
.venv/Scripts/python.exe scripts/audit_training_state.py --input runs/training_state_reproduction --output runs/training_state_reproduction_report/audit.json
.venv/Scripts/python.exe scripts/plot_training_state.py --input runs/training_state_reproduction_report
```

Use a fresh output directory. For interruption recovery, append `--resume` to
the identical training command. Only load trusted local checkpoints. Audits
verify exact optimizer/replay forks, regenerate causal memories and data,
recompute starting/final probes, and check matched clean/noisy streams. The
analysis averages episodes within seed before estimating factorial contrasts;
the full protocol defines their signs and interaction. A cleared buffer refills
and resets its reservoir bookkeeping, so this does not isolate obsolete examples
alone. [Sleep and pruning analogies](sleep_and_training_state.md) are recorded
separately from what these reset controls actually implement.

After a hard restart, stop any surviving study processes and inspect saved
files before resuming. The observed PC restart left some nonempty files full
of invalid data; atomic replacement alone did not make them power-loss durable.
This read-only check reports damage without choosing retries or deleting data:

```powershell
.venv/Scripts/python.exe scripts/check_training_state_files.py --input runs/training_state_reproduction --output runs/training_state_reproduction_preflight.json
```

Do not run the preflight while training is writing files. Preserve damaged
bytes and hashes before recovery, and retain every usable completed episode.
The original study's recovery history records exactly which work was rebuilt
with identical seeds/settings and which checkpoints were reused. Its resumed
eight-worker process tree was limited to affinity mask 85 (logical processors
0,2,4,6) to reduce CPU load; the restart cause is unknown. Source/config/protocol
identities stayed fixed. This execution detail supplements the original runtime
manifest and rules out interpreting local elapsed times as uniform-compute
benchmarks. Full trusted-checkpoint/probe audits are still required afterward.

## Adam: replay components and continuous renewal

The [prospective protocol](replay_renewal_protocol.md) fixes two new cohorts.
The first separates packet deletion, effective reservoir age, and random
generator reset at the first new dependency. The second compares uniform,
recent-only, and a fixed recent-8/historical-8 policy from initialization,
without state resets or change triggers. Its configurations are fixed before
either cohort; the first experiment does not select the second.

Run the engineering checks and both smoke configurations first. To reproduce
the main cohorts, use new output directories and lock both before training:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_replay_renewal.py -q
.venv/Scripts/python.exe -m acp_cl.replay_renewal.study --config configs/replay_renewal_decomposition_smoke.json --output runs/adam_reset_smoke
.venv/Scripts/python.exe -m acp_cl.replay_renewal.study --config configs/replay_renewal_policy_smoke.json --output runs/adam_policy_smoke
.venv/Scripts/python.exe -m acp_cl.replay_renewal.study --config configs/replay_renewal_decomposition.json --output runs/adam_reset --protocol docs/replay_renewal_protocol.md --lock-only
.venv/Scripts/python.exe -m acp_cl.replay_renewal.study --config configs/replay_renewal_policy.json --output runs/adam_policy --protocol docs/replay_renewal_protocol.md --lock-only
.venv/Scripts/python.exe -m acp_cl.replay_renewal.study --config configs/replay_renewal_decomposition.json --output runs/adam_reset --protocol docs/replay_renewal_protocol.md --resume
.venv/Scripts/python.exe -m acp_cl.replay_renewal.study --config configs/replay_renewal_policy.json --output runs/adam_policy --protocol docs/replay_renewal_protocol.md --resume
```

Run cohorts sequentially to keep total training concurrency at four workers.
Affinity mask 85 is specific to the original host. For an independent run on a
different host, make a new config with an appropriate mask (0 leaves it unset),
and record that new configuration/runtime; do not change a locked resumed run.

After each cohort stops, inspect file integrity, summarize all records, and
audit trusted local checkpoints. For the first cohort:

```powershell
.venv/Scripts/python.exe scripts/check_training_state_files.py --input runs/adam_reset --output runs/adam_report/decomposition/file_integrity.json
.venv/Scripts/python.exe scripts/summarize_replay_renewal.py --input runs/adam_reset --output runs/adam_report/decomposition
.venv/Scripts/python.exe scripts/audit_replay_renewal.py --input runs/adam_reset --output runs/adam_report/decomposition/audit.json
```

Repeat with input `runs/adam_policy` and output `runs/adam_report/policy`, then
generate portable figures:

```powershell
.venv/Scripts/python.exe scripts/plot_replay_renewal.py --input runs/adam_report
```

The audit verifies exact starting weights, Adam state, causal histories, replay
contents/counters/generators, regenerated streams, and starting/final probes.
The continuous comparison uses each policy's own evolving state. Its final
branches all fork that policy's third-introduction endpoint. Matched clean and
noisy streams differ only in reported outcomes. The six independent seeds are
the bootstrap units; three introductions are averaged within each seed first.
Terminal valid-old error, not just within-episode deterioration, supplies the
retention guardrails. A mean-based pilot pass does not establish noninferiority.

The completed portable archive can also be checked without loading model
checkpoints:

```powershell
.venv/Scripts/python.exe scripts/verify_replay_renewal_archive.py --input reports/replay_renewal
```

This checks every sealed artifact, both training-source archives, original raw
result hashes, complete cohort coverage, and agreement of final analysis code
with the prospective analysis snapshot. Run it after the archive is sealed;
it deliberately rejects an incomplete or subsequently modified archive.

## Evidence before consolidation

This diagnostic shares one draft trajectory among four prediction policies.
Run the fixed development and pilot sequentially; each launches four one-thread
workers. Use new output directories to reproduce this run. Independent
scientific confirmation additionally needs new, predeclared seeds or streams.

```powershell
.venv/Scripts/python.exe -m acp_cl.evidence_consolidation.study --config configs/evidence_consolidation_development.json --output runs/adam_evidence_dev --protocol docs/evidence_consolidation_protocol.md --lock-only
.venv/Scripts/python.exe -m acp_cl.evidence_consolidation.study --config configs/evidence_consolidation_pilot.json --output runs/adam_evidence_pilot --protocol docs/evidence_consolidation_protocol.md --lock-only
.venv/Scripts/python.exe -m acp_cl.evidence_consolidation.study --config configs/evidence_consolidation_development.json --output runs/adam_evidence_dev --protocol docs/evidence_consolidation_protocol.md --resume
.venv/Scripts/python.exe -m acp_cl.evidence_consolidation.study --config configs/evidence_consolidation_pilot.json --output runs/adam_evidence_pilot --protocol docs/evidence_consolidation_protocol.md --resume
```

The exact source, configuration, runtime, and protocol must match when resuming.
The saved proposal and its partial future-evidence window survive interruption.
The main cohort contains 12 shared draft trajectories, 216 continuous blocks,
36 fresh references, and 864 policy/block evaluations. These 864 evaluations
are not independent fitted learners. Six seeds per architecture are the
statistical units. Physical prediction still ends at H=12.

After a cohort stops, inspect, summarize, and audit it:

```powershell
.venv/Scripts/python.exe scripts/check_training_state_files.py --input runs/adam_evidence_pilot --output runs/adam_evidence_report/pilot/file_integrity.json
.venv/Scripts/python.exe scripts/summarize_evidence_consolidation.py --input runs/adam_evidence_pilot --output runs/adam_evidence_report/pilot
.venv/Scripts/python.exe scripts/audit_evidence_consolidation.py --input runs/adam_evidence_pilot --output runs/adam_evidence_report/pilot/audit.json
.venv/Scripts/python.exe scripts/plot_evidence_consolidation.py --input runs/adam_evidence_report/pilot
```

Repeat for development with its own input/output. The audit reconstructs raw
and clean evaluation streams, exact causal replay/history, global acceptance
windows, optimizer budgets, and committed snapshot lineage. It recomputes
starting/terminal query panels and the first packet's prequential scores and
gate gains in every block. Intermediate window weights are not all preserved;
their remaining gains are checked arithmetically, not independently retrained.

The sealed original portable archive is checked without loading checkpoints:

```powershell
.venv/Scripts/python.exe scripts/verify_evidence_consolidation_archive.py --input reports/evidence_consolidation
```

This additionally checks original raw JSON hashes, exact record coverage, both
training source ZIPs, and that final analysis matches its pre-pilot lock.
Full local model checkpoints are trusted pickle files under ignored `runs/`;
do not use the checkpoint auditor with untrusted files.

## Matched context and selective update follow-up

The [return-context protocol](return_context_protocol.md) diagnoses trusted
saved evidence-pilot checkpoints without training. It is retrospective. The
first interrupted attempt is preserved under `reports/return_context`; the
documented numerical correction produced `reports/return_context_v2`. Use a
new output directory to repeat it:

```powershell
.venv/Scripts/python.exe scripts/diagnose_return_context.py --input runs/evidence_consolidation_pilot --archive reports/evidence_consolidation --output runs/return_context_repeat --protocol docs/return_context_protocol.md --lock-only
.venv/Scripts/python.exe scripts/diagnose_return_context.py --input runs/evidence_consolidation_pilot --archive reports/evidence_consolidation --output runs/return_context_repeat --protocol docs/return_context_protocol.md
```

The [selective-update protocol](selective_updates_protocol.md) is a separate
fixed development experiment. Five learned arms share each ordinary prefix;
a sixth fresh arm qualifies learnability. All keep uniform reservoir replay.
Only the .75-current-loss projected arm is the primary candidate. This local
first-introduction study does not test a continuous lifetime or late acquisition.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_selective_updates.py tests/test_selective_update_study.py tests/test_selective_update_summary.py tests/test_selective_update_audit.py -q
.venv/Scripts/python.exe -m acp_cl.selective_updates.study --config configs/selective_updates_smoke.json --output runs/adam_selective_smoke_repeat --protocol docs/selective_updates_protocol.md
.venv/Scripts/python.exe -m acp_cl.selective_updates.study --config configs/selective_updates_development.json --output runs/adam_selective_repeat --protocol docs/selective_updates_protocol.md --lock-only
.venv/Scripts/python.exe -m acp_cl.selective_updates.study --config configs/selective_updates_development.json --output runs/adam_selective_repeat --protocol docs/selective_updates_protocol.md --resume
```

The main configuration uses four single-thread CPU workers and host-specific
affinity mask 85. Independent machines should use a newly named configuration
with suitable affinity and record the changed runtime. Repeating the same
seeds is reproduction, not independent scientific confirmation. Never change
code, configuration or the protocol within a locked resumed run.

After fitting stops, check file integrity and audit all results:

```powershell
.venv/Scripts/python.exe scripts/check_training_state_files.py --input runs/adam_selective_repeat --output runs/adam_selective_report/file_integrity.json
.venv/Scripts/python.exe scripts/summarize_selective_updates.py --input runs/adam_selective_repeat --output runs/adam_selective_report
.venv/Scripts/python.exe scripts/audit_selective_updates.py --input runs/adam_selective_repeat --output runs/adam_selective_report/audit.json
.venv/Scripts/python.exe scripts/plot_selective_updates.py --input runs/adam_selective_report/summary.json --output runs/adam_selective_report
```

The audit verifies actual tensor/optimizer state, exact forks, regenerated
causal history/replay, matched resource counters, and starting/final predictions.
Intermediate gradient trajectories are not independently retrained. Tests
separately cover reference equivalence, geometric constraints, optimizer moment
semantics, interruption/resumption and deliberately corrupted checkpoints.

Once both cohorts are complete and the portable archive is sealed, verify it
without loading model checkpoints:

```powershell
.venv/Scripts/python.exe scripts/verify_selective_updates_archive.py --input reports/selective_updates
```

This checks all sealed files and independently recomputes the primary
candidate's acquisition, per-mode retention, survival, paired intervals and
declared screens from the raw records. The archive also carries a standalone
`verify_archive.py`, requiring only Python and NumPy. Independent calculation
of recorded results is different from independent scientific replication.

## Predictive rehearsal value

The [predictive-value protocol](predictive_value_protocol.md) tests whether a
memory's estimated rehearsal benefit transfers to later observations and
returning conditions. The ordinary learner always uses unchanged uniform
replay. Candidate choices never alter that trajectory. The original accuracy
control uses error recorded before a packet's first training update; the
uniform comparator is the exact expected loss across the same candidates.

Use new output directories for a repeat. The diagnostic runs six single-thread
workers, up from four in the prior experiment. Affinity mask 1365 is host
specific; choose a new configuration for a different host. Do not alter a
locked run's code, settings or runtime when resuming.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_predictive_value.py tests/test_predictive_value_study.py tests/test_predictive_value_summary.py tests/test_predictive_value_audit.py -q
.venv/Scripts/python.exe -m acp_cl.predictive_value.study --config configs/predictive_value_smoke.json --output runs/adam_value_smoke_repeat --protocol docs/predictive_value_protocol.md
.venv/Scripts/python.exe -m acp_cl.predictive_value.study --config configs/predictive_value_diagnostic.json --output runs/adam_value_repeat --protocol docs/predictive_value_protocol.md --lock-only
.venv/Scripts/python.exe -m acp_cl.predictive_value.study --config configs/predictive_value_diagnostic.json --output runs/adam_value_repeat --protocol docs/predictive_value_protocol.md --resume
```

The full cohort has 24 ordinary trajectories, 48 assessments and 1,296 short
diagnostic forks. Each trajectory learns 20,992 arrivals and has both gap
lengths. The independent units are 12 seeds per architecture, with their two
assessments averaged before uncertainty calculations. The smoke configuration
is for engineering only. Repeating identical seeds checks reproduction; it
does not provide independent scientific confirmation.

After training has stopped:

```powershell
.venv/Scripts/python.exe scripts/check_training_state_files.py --input runs/adam_value_repeat --output runs/adam_value_report/file_integrity.json
.venv/Scripts/python.exe scripts/summarize_predictive_value.py --input runs/adam_value_repeat --output runs/adam_value_report
.venv/Scripts/python.exe scripts/audit_predictive_value.py --input runs/adam_value_repeat --output runs/adam_value_report/audit.json
.venv/Scripts/python.exe scripts/plot_predictive_value.py --input runs/adam_value_report/summary.json --output runs/adam_value_report
```

The local audit loads trusted checkpoints, regenerates streams/replay, and
independently reconstructs short rehearsal updates and predictions. It does
not independently retrain every update of the ordinary trajectory. Checkpoint
wrappers preserve parameter gradients explicitly because native Parameter
serialization omits them. The ordinary prequential forwards, rehearsal work,
shadow forecasts and temporary copies are counted separately.


Portable evidence can be verified from either cohort without trusted checkpoint
loading. Each carries its own manifest and standalone NumPy-only verifier:

```powershell
.venv/Scripts/python.exe -I -B reports/predictive_value/smoke/verify_archive.py --input reports/predictive_value/smoke
.venv/Scripts/python.exe -I -B reports/predictive_value/diagnostic/verify_archive.py --input reports/predictive_value/diagnostic
```

The `-B` option prevents Python cache files from changing the strict archive.
The verifier's source/tests were explicitly excluded from the prospective
scientific lock and sealed separately; they cannot change the selected metrics
or decisions. It checks recorded arithmetic and the tensor audit's coverage,
not a second model-training replication. Read the
[archive index](../reports/predictive_value/README.md) for its exact scope.


## Rehearsal state and paired environments

The [state-substitution protocol](rehearsal_state_protocol.md) uses fresh seeds
and an unchanged ordinary uniform-replay trajectory. Eight combinations of
old/new weights, complete Adam state and causal anchor rehearse the same
candidate packets. Frozen predictors then face paired current/return branches
from the same cutoff, with matched observations/actions and initial history.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_rehearsal_state_mechanism.py tests/test_rehearsal_state_study.py tests/test_rehearsal_state_summary.py tests/test_rehearsal_state_audit.py -q
.venv/Scripts/python.exe -m acp_cl.rehearsal_state.study --config configs/rehearsal_state_smoke.json --output runs/rehearsal_state_smoke_repeat --protocol docs/rehearsal_state_protocol.md
.venv/Scripts/python.exe -m acp_cl.rehearsal_state.study --config configs/rehearsal_state_diagnostic.json --output runs/rehearsal_state_repeat --protocol docs/rehearsal_state_protocol.md --lock-only
.venv/Scripts/python.exe -m acp_cl.rehearsal_state.study --config configs/rehearsal_state_diagnostic.json --output runs/rehearsal_state_repeat --protocol docs/rehearsal_state_protocol.md --resume
```

Use new directories for reproduction. The host configuration uses six one-thread
CPU workers with affinity 1365. Changing host/runtime requires a newly named run;
resumption requires exact code/configuration/runtime/protocol identities. The
scientific cohort has 24 trajectories, 48 assessments and 3,456 short rehearsal forks.
Each trajectory learns 18,432 ordinary arrivals. The tiny smoke is engineering
only. The two assessments are averaged within each of 12 independent seeds per
architecture; identical-seed reproduction is not scientific replication.

After fitting stops, analyze, independently audit and produce standalone figures:

```powershell
.venv/Scripts/python.exe scripts/summarize_rehearsal_state.py --input runs/rehearsal_state_repeat --output runs/rehearsal_state_report
.venv/Scripts/python.exe scripts/check_rehearsal_state_files.py --input runs/rehearsal_state_repeat --output runs/rehearsal_state_report/file_integrity.json
.venv/Scripts/python.exe scripts/audit_rehearsal_state.py --input runs/rehearsal_state_repeat --output runs/rehearsal_state_report/audit.json --workers 6
.venv/Scripts/python.exe scripts/plot_rehearsal_state.py --input runs/rehearsal_state_report/summary.json --output runs/rehearsal_state_report
```

Checkpoints preserve parameter gradients explicitly. Complete all-action
validation forecasts and both branches' outcomes/truth are stored in NPZ files,
which are read with allow_pickle=False. The auditor loads trusted model
checkpoints and independently reproduces every short intervention and diagnostic
forecast, but does not independently retrain all ordinary learning updates.
Only the current branch enters ordinary training. The artificial hybrid states
are diagnostic substitutions, not deployable memory policies.

Both completed cohort archives include a standalone NumPy-only verifier:

```powershell
.venv/Scripts/python.exe -I -B reports/rehearsal_state/smoke/verify_archive.py --input reports/rehearsal_state/smoke
.venv/Scripts/python.exe -I -B reports/rehearsal_state/diagnostic/verify_archive.py --input reports/rehearsal_state/diagnostic
```

The verifier checks all-action probability arithmetic, seed summaries, factorial
and environment contrasts, the fixed screen, source/protocol/analysis locks,
raw record/array hashes and tensor-audit coverage. It imports no live scientific
source and loads no pickle. Its separately sealed source and tests are packaging
work explicitly excluded from the scientific lock. Use `-B` to avoid introducing
cache files into the strict archive. See the
[archive index](../reports/rehearsal_state/README.md) for the exact scope.

A separate unchanged NumPy review reproduced 3,408 seed/bootstrap statistics
from the full cohort's raw arrays. All 172 scoped tests pass; the full tensor
audit recreates 3,456 forks and 120,576 diagnostic forecasts. These verify the
recorded experiment, rather than supplying independent scientific replication.
The completed report keeps both failed replication decisions unchanged.

## Learned core and adaptive raw-input residual

The [core/residual protocol](core_residual_protocol.md) compares a learned core
plus raw-input residual with the same architecture trained jointly. A third
arm freezes the residual visual encoder at random initialization. Prefix
training starts from scratch; complete core Adam state and uniform replay
survive the one declared allocation cutoff. Maintenance, new-dependency
learning, and return then form continuous trajectories without further resets.

Use new output directories for reproduction. The recorded development config
has six independent seeds per architecture, six one-thread CPU workers and
affinity mask 1365. It runs 12 shared prefixes, 36 continuing trajectories,
12 fresh references and 207,360 optimizer updates. The smoke is for engineering
only. Any changed runtime, source or config needs a separate newly locked run;
same-seed reproduction is not an independent scientific replication.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_core_residual_learner.py tests/test_core_residual_evaluation.py tests/test_core_residual_study.py tests/test_core_residual_summary.py tests/test_core_residual_audit.py -q
.venv/Scripts/python.exe -m acp_cl.core_residual.study --config configs/core_residual_smoke.json --output runs/core_residual_smoke_repeat --protocol docs/core_residual_protocol.md
.venv/Scripts/python.exe -m acp_cl.core_residual.study --config configs/core_residual_development.json --output runs/core_residual_repeat --protocol docs/core_residual_protocol.md --lock-only
.venv/Scripts/python.exe -m acp_cl.core_residual.study --config configs/core_residual_development.json --output runs/core_residual_repeat --protocol docs/core_residual_protocol.md --resume
```

After the entire cohort has finished:

```powershell
.venv/Scripts/python.exe scripts/summarize_core_residual.py --input runs/core_residual_repeat --output runs/core_residual_report
.venv/Scripts/python.exe scripts/audit_core_residual.py --input runs/core_residual_repeat --output runs/core_residual_report/audit.json --workers 6
.venv/Scripts/python.exe scripts/check_rehearsal_state_files.py --input runs/core_residual_repeat --output runs/core_residual_report/file_integrity.json
.venv/Scripts/python.exe scripts/plot_core_residual.py --input runs/core_residual_report/summary.json --output runs/core_residual_report
```

The existing file-integrity utility is reused unchanged; it handles ordinary
JSON/ZIP/checkpoint/NPZ files despite its historical name. The tensor auditor
loads trusted local checkpoints and independently checks stream physics, replay,
causal histories, initialization, Adam state and parameter freezing. It recovers
boundary/endpoint forecasts and each phase's first pre-update prediction.
Intermediate model states are not saved; their forecasts are checked through
raw arrays, physical truth and provenance rather than full retraining.

The sole primary candidate is separate versus joint, per architecture. The
fresh-reference qualification, acquisition and retention guards are all fixed
in the protocol. The fixed-feature comparison supports a separate attribution
question and cannot replace a failed primary candidate. Equal total capacity,
updates and forward exposure do not imply equal backward work or FLOPs.

The completed [development report](../reports/core_residual_results.md) records
both failed primary screens and the narrower recurrent feature-attribution
pass. The source, configuration and scientific analysis are immutable. All
268 scoped tests pass, with two architecture-inapplicable skips. The archives
can be checked without model pickles or repository imports:

```powershell
.venv/Scripts/python.exe -I -B reports/core_residual/smoke/verify_archive.py --input reports/core_residual/smoke
.venv/Scripts/python.exe -I -B reports/core_residual/development/verify_archive.py --input reports/core_residual/development
```

The portable verifier and its 73 tests are separately sealed packaging work,
prospectively excluded from the scientific lock. Each archive has an exact
artifact manifest, and both were successfully checked after copying outside
the repository. The [archive index](../reports/core_residual/README.md) describes
what is independently recomputed and what depends on the local tensor audit.

## Saved core/residual output diagnostic

This [bounded retrospective diagnostic](../reports/core_residual_output_results.md)
reuses the completed core/residual models without training. Its new module lives
outside the old scientific package so the earlier source manifest stays valid.
The [fixed review](core_residual_output_review.md) defines four endpoint panels,
component/routing ablations, actual versus refreshed support, and exploratory
effort-allocation rules. The rules do not replace the original failed screens.

Use new output locations to reproduce inference. Six single-thread workers and
affinity 1365 match the modest CPU budget. The script refuses existing forecasts
and verifies the unchanged base locks and selected artifact hashes.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_core_residual_outputs.py -q
.venv/Scripts/python.exe scripts/diagnose_core_residual_outputs.py --base runs/core_residual_smoke --output runs/core_residual_outputs_smoke_repeat --workers 2
.venv/Scripts/python.exe scripts/summarize_core_residual_outputs.py --input runs/core_residual_outputs_smoke_repeat --output runs/core_residual_outputs_smoke_repeat/summary.json
.venv/Scripts/python.exe scripts/diagnose_core_residual_outputs.py --base runs/core_residual_development --output runs/core_residual_outputs_repeat --workers 6 --lock-only
.venv/Scripts/python.exe scripts/diagnose_core_residual_outputs.py --base runs/core_residual_development --output runs/core_residual_outputs_repeat --workers 6
.venv/Scripts/python.exe scripts/summarize_core_residual_outputs.py --input runs/core_residual_outputs_repeat --output runs/core_residual_outputs_repeat/summary.json --development
.venv/Scripts/python.exe scripts/plot_core_residual_outputs.py --input runs/core_residual_outputs_repeat
```

The main checks 48 loaded states and 384 input/context rows, including 1,536 component
forecast tensors, with zero optimizer updates. All 384 original full predictions
reconstruct exactly and 96 repeated-input frozen-core pairs agree (72 across
saved endpoints). Complete learner
and global RNG signatures must remain unchanged. The original smoke has one
support replicate and 40 rows; main has two replicates and 384 rows. Average supports
within seed; do not count these as independent training replicates.

For arithmetic-only rescoring, the standalone summary imports NumPy and the
standard library, loads no model pickle, and can run outside the repository.
Extract it from the output's `source_at_lock.zip` when reproducing an archived
version. Preserve the archive and place new summaries in a separate directory:

```powershell
.venv/Scripts/python.exe -I -B scripts/summarize_core_residual_outputs.py --input reports/core_residual_outputs/development --output runs/core_residual_outputs_rescore/summary.json --development
```

The recorded main rescoring reproduces 12,288 metrics. Copying the output and
script outside the repository reproduced both summary files byte-for-byte;
five intentionally corrupted copies were rejected. A separate reviewer checks
raw arithmetic, paired contrasts, bootstrap intervals and allocation decisions.
See the [archive index](../reports/core_residual_outputs/README.md) for provenance,
the documented smoke repair and verification limits. Same-checkpoint reproduction
is not independent scientific replication. The
[causal-access proposal](core_residual_output_followup.md) remains unrun.
