# Continual representation learning from scratch: first architectural prototype

Status: initial implementation and feasibility diagnostic. This is a new
experimental line; the archived ACP gain studies retain their original results.
The completed [initial pilot report](../reports/dual_path_results.md) preserves
the negative result and the exact training-source archive.
The objective is a general learner that continues to acquire new representations,
retains useful earlier knowledge, and eventually demonstrates useful transfer
under a declared memory and compute budget. No pretrained encoder is used.

## Hypothesis and scope

Separating rapid representation learning from consolidation might reduce
interference while allowing shared representations to evolve. This prototype
tests that organization with ordinary differentiable CNNs and SGD. The model is
`S(x) + R(x)`: two independent predictors receive the same raw, preprocessed
image and their logits are added. Both have the same fixed output label support.

`S` is initialized identically to the single-network baseline. `R` has random
feature weights and an initially zero output projection, so its insertion leaves
predictions unchanged. Its output projection learns first; feature gradients
become available after that projection becomes nonzero. Zero initialization does
not guarantee preservation after learning or after later consolidation.

This is inspired by existing work, especially
[Progress & Compress](https://proceedings.mlr.press/v80/schwarz18a.html),
[CLS-ER](https://arxiv.org/abs/2201.12604), and
[FOSTER](https://arxiv.org/abs/2204.04662). It is not a reproduction of those
algorithms or a claim of a novel fast/slow memory principle. A future novelty
claim requires a precise distinction and a stronger comparison with that work.

## Methods

| Method | Online acquisition | Consolidation |
|---|---|---|
| `er` | Existing single-network replay baseline | None |
| `derpp` | Existing single-network replay plus historical-logit regularization | None |
| `dual_joint` | Both pathways learn from current examples and replay | None |
| `dual_derpp` | Both pathways learn with replay and historical-logit regularization | None |
| `dual_frozen` | Only the fast pathway learns; the stable path stays at random initialization | None; ablation |
| `dual_consolidate` | Only the fast pathway learns between consolidation events | Fixed update schedule, then fast-path renewal |

The causal architecture comparison is `dual_consolidate` against `dual_joint`:
both have identical parameter counts and initial weights, raw inputs, label
support, incoming examples, reservoir membership, and online replay draws.
They do not have equal optimization work. `dual_derpp` is a stronger comparison
on the same architecture. The legacy ER/DER++ optimizer has its existing update
semantics; comparisons with it also change architecture and optimizer details.

`dual_scratch` is an internal diagnostic, not a public study method: a fresh
two-pathway model trained jointly without replay on each experience. Ordinary
methods use the existing fresh single-network reference. Both references keep
the complete output support and see the identical current examples and order.
These are architecture-matched, replay-free references, not replicas of each
method's consolidation schedule or total compute budget.

## Exact learning cycle

1. Before inserting the current batch into memory, draw online replay using
   the same independent reservoir/train RNG scheme as the existing learner.
2. Train on `CE(current) + replay_weight * CE(replay)`. DER++ variants add
   `der_alpha * MSE(replay_logits, arrival_logits)` using the same replay draw.
   Stored logits are pre-update predictions on the arrival training view.
3. For separated methods, freeze every stable parameter and put that complete
   network in evaluation mode. The fast pathway can learn convolutional features
   from the input; it is not limited to a frozen final embedding.
4. Every `consolidation_interval` online updates in `dual_consolidate`, construct
   a pool from the current batch and at most `consolidation_pool_size` old
   reservoir examples. Sampling uses a separate, checkpointed RNG and cannot
   change online replay draws or reservoir membership. Use unaugmented pool
   inputs and cache the combined predictor's logits **before** updating `S`.
5. Take exactly `consolidation_steps` SGD updates of `S` on minibatches drawn
   without replacement within each draw from that fixed pool. Optimize label
   cross-entropy plus `distillation_weight * T^2 * KL(teacher_T || student_T)`.
   Teacher targets remain immutable for the whole event; no extra teacher model
   or unbounded historical cache is retained. Separate stable SGD momentum
   persists between consolidation events.
6. Reinitialize the fast feature network, zero its output projection, and clear
   its optimizer state. This is approximate transfer followed by deletion of
   the residual; it is not an exactly function-preserving operation.
7. Insert the original current samples into the reservoir exactly once.

SGD uses the configured momentum and weight decay. Gradients are clipped before
the optimizer step; `max_grad_norm` is not a bound on total optimizer displacement
with momentum/decay. Actual online displacement is logged. Consolidation is not
triggered by task boundaries, losses, class IDs, or evaluation measurements.
The learner receives only batches and labels.

## Diagnostics and accounting

The same training-memory probe is measured before/after acquisition and, when
applicable, after consolidation before renewal and after renewal. The intermediate
combined predictor still includes the old residual; these are operational stage
measurements, not independent causal estimates. Probe examples can overlap the
consolidation pool. They are not held-out validation data or a retention guarantee.

Online current/replay exposure, branch forward/backward example counts,
consolidation presentations/steps, teacher forwards, probe forwards, and evaluation
forwards are logged separately. Branch-example counts do not equal FLOPs because
the pathways have different widths. Runtime includes diagnostics and evaluation.
No speed claim follows from the pilot.

Model, optimizer, and replay tensor payloads and peak tracked auxiliary payloads
are recorded. Auxiliary accounting covers consolidation pool/target tensors,
parameter-displacement snapshots, and fast-path reinitialization. It is not total
process memory: activations, gradients, allocator overhead, Python objects, and
growing experiment logs are excluded. The runner additionally records peak CUDA
allocation. The learning state has fixed configured capacity; the audit archive
is allowed to grow. No unlimited-memory retention claim is made.

Checkpoints preserve both optimizer states, model, reservoir, all learner RNGs,
online update count, consolidation cadence, event logs, and accounting. Existing
source/runtime/configuration identity checks still reject changed runs on resume.

## Initial pilot fixed before training

The engineering smoke uses recurring procedural shapes to exercise code paths;
it is not evidence about learning new representations.

The natural-image pilot uses all 100 CIFAR-100 classes in ten disjoint groups of
ten. Per class, 450 official training images arrive once and 50 are reserved for
validation. Test examples are not evaluated. The existing loader opens the test
dataset file to construct an empty split; official test labels do not enter
learning, validation, or decisions. Evaluation uses a fixed 100-output classifier
without seen-class masks or task IDs. New classes demand learning new distinctions,
but class progression alone does not prove that new features, rather than new
readouts, explain any gain.

Seeds 2027 and 2131, six methods, batch size 64, and one pass give 45,000 current
arrivals and 710 online updates per run. All classes are selected in both seeds;
class order and train/validation splits vary by seed. Stable width is 64, fast
width 32, replay capacity 512, and replay batch size 64. Consolidation occurs
every 73 updates, deliberately differing from the 71-update experience length:
nine events with 24 stable updates each, or 216 extra optimizer updates per run.
The pool contains at most 256 replay examples plus the current batch.

This is a feasibility diagnostic with two paired seeds and zero comparative
hyperparameter search. Preserve every result, including poor acquisition or
consolidation damage. Do not add seeds or tune settings to rescue its outcome.
Continuation requires examining whether ordinary learners learn adequately and
whether consolidation is actually transferring useful behavior. Favorable means
alone do not establish superiority. A later study requires equal declared tuning
budgets, resource-matched controls, stationary controls, and longer published
benchmarks before any sustained-plasticity claim.

Report each seed's final retention, forgetting, acquisition AUC through the first
512 current examples, arrival accuracy, within-experience improvement, and
architecture-matched scratch differences. Acquisition AUC includes retained
knowledge and transfer; subtracting the fresh reference is informative but does
not isolate plasticity. Consolidation can fall within an acquisition window and
adds work; AUC counts current presentations, not compute. Ten experiences are
not evidence of indefinite continual learning.

## Run

```powershell
.venv/Scripts/python.exe -m acp_cl run --config configs/dual_path_smoke.json --output runs/dual_path_smoke --device cpu --no-plots
.venv/Scripts/python.exe -m acp_cl run --config configs/dual_path_cifar100_pilot.json --output runs/dual_path_cifar100_pilot --device cuda --no-plots
.venv/Scripts/python.exe scripts/summarize_dual_path.py --input runs/dual_path_cifar100_pilot --output reports/dual_path
```

Use a new output directory after any source/configuration change. CUDA is optional;
the same command accepts `--device cpu`. Natural images must already be available
under `data`; this configuration does not download them automatically.
