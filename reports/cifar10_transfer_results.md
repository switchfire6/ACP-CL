# CIFAR-10 transfer pilot: stop scaling the frozen recipe

All **16 predeclared development runs** completed, with no comparative retries,
exclusions, or numerical failures. The baselines passed the learning-adequacy
screen. The newborn-gain variants failed the signal and retention criteria.
The decision fixed before training is therefore **do not advance this recipe**.

The modest procedural-shape signal did not transfer convincingly to this
natural-image pilot. This narrows the supported claim; it does not disprove
critical-period mechanisms or establish a general ranking of learning methods.
No additional seeds or parameter search were run to reverse the decision.

## What was tested

The [protocol](../docs/cifar10_transfer_protocol.md) and
[lock](../configs/cifar10_transfer/lock.json) were committed before comparative
training. Four methods share the v3 CNN, optimizer, replay budget, initial
40-update warmup, and complete-optimizer displacement cap. Recycling variants
share reset counts and times; their chosen unit identities depend on their own
learned utilities. Newborn gain is 2 initially, decaying to the mature gain .5
over 30 updates. The matched variant redistributes mature gain to preserve the
nominal feature mean.

For each seed (1063, 1174), 45,000 distinct official CIFAR-10 training images
arrive once over 30 experiences, with all ten labels always present. The same
underlying IDs and arrival order feed recurring original/grayscale/blur views
and a stationary original-view control. Each experience has 46 batches of 32
and one of 28, giving 1,410 updates per run. No crop/flip augmentation is used.
Source IDs and domain metadata stay outside the learner.

Validation reserves are split before transformation. Each condition evaluates
a fixed 500-image panel; recurring views share those base images. Official test
images/labels are not used. Late early-acquisition AUC averages experiences
16–30, integrating each curve through its first 256 current presentations.
It includes transfer and retained knowledge as well as adaptation.

## Outcomes

Entries below are means over the two paired seeds, expressed as percentages.
Recurring final accuracy weights the three trained domains equally. Stationary
final accuracy concerns its original-view panel only.

| Method | Recurring final | Recurring late AUC | Stationary final | Stationary late AUC |
|---|---:|---:|---:|---:|
| Replay | 42.83 | 39.40 | 47.90 | 42.94 |
| Replay + recycling | 43.40 | 40.24 | 44.60 | 43.07 |
| Newborn gain | 39.80 | 40.12 | 45.40 | 43.40 |
| Newborn gain, nominal mean matched | 43.23 | 40.42 | 45.10 | 43.11 |

The primary recurring contrasts below are candidate minus replay + recycling,
in **percentage points**. Every seed is shown; there is no two-seed confidence
interval or significance claim.

| Candidate / metric | Seed 1063 | Seed 1174 | Mean |
|---|---:|---:|---:|
| Newborn / late AUC | -0.4567 | +0.2100 | -0.1233 |
| Matched newborn / late AUC | -0.1900 | +0.5500 | +0.1800 |
| Newborn / final accuracy | -5.2667 | -1.9333 | -3.6000 |
| Matched newborn / final accuracy | -0.6000 | +0.2667 | -0.1667 |

Newborn gain reduces recurring final accuracy against recycling on both seeds.
Matching its nominal budget largely removes that endpoint loss, but does not
produce the required consistent acquisition gain. On the stationary stream,
ordinary replay has the highest final accuracy on both seeds. Newborn loses
3.8 and 1.2 points versus replay; matched newborn loses 3.0 and 2.6 points.

Recurring endpoint means by domain are also descriptive, correlated views:

| Method | Original % | Grayscale % | Blur % |
|---|---:|---:|---:|
| Replay | 44.7 | 41.4 | 42.4 |
| Replay + recycling | 44.2 | 42.6 | 43.4 |
| Newborn gain | 41.7 | 39.1 | 38.6 |
| Matched newborn | 44.7 | 43.3 | 41.7 |

Stationary maximum drawdown uses **all 30 end-of-experience diagonals** on the
same panel. For seeds 1063/1174, it is 8.4/6.0 pp for replay, 7.6/5.6 for
recycling, 8.4/5.0 for newborn, and 7.4/2.8 for matched newborn. These measurements
can miss damage within an experience. Generic task-column forgetting/BWT is
preserved in the archive but is not interpreted as independent-task retention.

![Complete two-seed CIFAR-10 transfer overview](cifar10_transfer/overview.png)

## Applying the rule fixed before outcomes

| Requirement | Result |
|---|---|
| Adequacy: stationary baselines learn; recurring recycling learns without ceiling | All 12 checks pass |
| Signal: positive newborn AUC on both seeds, mean at least 1 pp; matched nonnegative on both, mean at least .5 pp | 2 of 6 checks pass; requirement fails |
| Retention: neither gain variant loses over 1 pp final accuracy against either replay or recycling on any seed/condition | 8 of 16 checks pass; requirement fails |
| Advance this frozen recipe only if every requirement passes | **Do not advance** |

These thresholds allocate further effort; they are not statistical tests.
[Machine-readable decisions](cifar10_transfer/summary.json) retain every operand,
threshold, comparison, and reason. [Individual results](cifar10_transfer/summary.md)
show all 16 runs, with full curves, matrices, domain endpoints, and costs in the
[compact archive](cifar10_transfer/individual_results.json).

## What this contributes, and what it does not

This is useful progress toward deciding whether the idea deserves further work:
a previously modest positive signal failed a prospectively specified transfer
check, with adequate baselines and preserved negative outcomes. The reusable
single-arrival stream and pairing checks also improve the experimental platform.
That is not yet an established novel scientific contribution.

The component ideas have close predecessors. Continual renewal appears in
[continual backpropagation](https://www.nature.com/articles/s41586-024-07711-7);
preferential learning for new units appears in
[Neurogenesis Deep Learning](https://arxiv.org/pdf/1612.03770). The
[existing attribution review](../docs/algorithm_v3.md#close-computational-precedents)
also covers age cohorts and activity-dependent maturation. Reimplementing or
combining these ingredients does not itself establish priority.

The earlier shape experiment was not shown to be cheating. This follow-up
changes image complexity, label count, current-image repetition, and stream
length, so it cannot isolate which difference explains the lost advantage.
The current experiment still uses artificial transforms of natural images;
it is not an established external benchmark or evidence about natural drift.
Its short horizon does not test prevention of long-term plasticity loss.

The recommendation is to **archive this operating point and stop adding gain,
protection, or controller variants in response to these outcomes**. Before
more training, identify a precise gap in the published renewal/maturation work
and a published benchmark capable of testing it. A further study should have
equal declared tuning budgets, competitive baselines, and fresh evaluation
seeds. If no distinct question survives that check, ending this line is a
reasonable outcome. These two seeds do not justify scaling the current recipe.

The project is suitable for a public research notebook that states these limits
and includes unsuccessful experiments. A claim of a new effective continual
learning method remains unsupported. This work was kept in local Git.

## Verification, cost, and reproducibility

Source freeze: `c250129b957d0fb930fa5b8a3a22ececef6ba1ae`; pre-run lock:
`3f72cd8740b5d36ce0730b97ecbbdf183e642083`. Source SHA-256:
`60aa03ca51a5d530ee58e5ed631b1fc7c702052f26de8f0be981c0731d984268`.
All runs used Python 3.10.11, Torch 2.8.0+cu128, NumPy 2.2.6, CUDA 12.8 and an
RTX 4070 Ti. Source/configuration bytes stayed fixed during training; analysis
and documentation were written concurrently, which can mark result Git state
dirty without changing the frozen training identity.

The [independent audit](cifar10_transfer/independent_audit.md) recomputed all
480 acquisition curves and checked the 22,560 updates, completed checkpoints,
source/runtime/configuration identities, split separation, actual unique
arrivals, paired replay/RNG states, warmup traces, reset ages/counts, and nominal
gain budgets. It agrees with the primary analysis. This is independent local
agent recomputation, not a training rerun or external scientific replication.
An analysis-only tuple/list comparison bug was corrected before the archive
was generated; checkpoint and JSON values already agreed. No outcomes changed.

Every run presents 45,000 current images and 45,088 replay images, with 2,819
training forward/backward calls and 18,048 probe-image forwards. Recycling
methods reset 204 units in 68 events; replay resets none. The model has 28,506
parameters and replay occupies 788,480 tensor bytes. Cached evaluation costs
198,000 image presentations per recurring run and 195,000 per stationary run.
The suites ran concurrently on one GPU; summed logged run wall time is 1,210.29
seconds and is **not GPU compute time or a method-speed benchmark**.

The complete optimizer proposal remains bounded at .1, apart from floating
roundoff (maximum logged norm .100000005643; audit tolerance 1e-6). Structural
resets are outside that cap. Clipping affects 66.7–74.4% of updates when averaged
within each method/condition. For recurring recycling/newborn/matched newborn,
the mean achieved post-warmup gains are .42621/.43151/.43040, and mean per-run
summed feature data displacements are 120.143/120.382/119.422. Nominal mean matching therefore
does not make realized updates equal. All per-run values remain in the archive.

The final CPU suite passes **546 tests**, with one CUDA test and two Windows
symlink tests skipped. Ruff passes. Eight separate real-CIFAR engineering runs
cover all four methods on CPU and GPU; they are excluded from comparative
claims. [Preflight evidence](cifar10_transfer/stream_preflight.json) and
[engineering evidence](cifar10_transfer/engineering_verification.json) preserve
their scope; the latter records the 506-test source-freeze stage, before the
40 analysis tests were added. The exported figure was visually inspected.

To regenerate analysis from trusted local raw runs into a separate directory:

```powershell
.venv/Scripts/python.exe scripts/summarize_cifar10_transfer.py --output runs/cifar10_transfer_report_rebuild
.venv/Scripts/python.exe scripts/audit_cifar10_transfer.py --summary runs/cifar10_transfer_report_rebuild/summary.json --output runs/cifar10_transfer_report_rebuild
.venv/Scripts/python.exe scripts/plot_cifar10_transfer.py --archive runs/cifar10_transfer_report_rebuild/individual_results.json --output runs/cifar10_transfer_report_rebuild
```

The plot can be regenerated from the committed compact archive alone. Raw
weights, traces, official data, and local run directories remain ignored;
archived artifact hashes identify them. Per-seed provenance retains split IDs
and actual shuffled arrival IDs once, shared across paired methods/conditions.
The [README](../README.md#cifar-10-transfer-pilot) gives training commands.
The older v3 sealed archive remains the historical snapshot at `a038786`.
