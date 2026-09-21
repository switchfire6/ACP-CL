# V3 allocation rules

The implementation is `src/acp_cl/plasticity_v3.py`. The legacy engines remain
available. V3 uses the same model, minibatches, replay, and utility measurement,
but an independent optimizer proposal/commit path. See `experiment_v3.md` for
the chosen settings and interpretation limits.

## Causal schedule

At optimizer update t, the baseline feature multiplier is one for t <= W and
the locked constant g afterward. The classifier multiplier is one. No control
signal, domain boundary, or evaluation statistic enters this schedule.

On updates W + kI, k = 1, 2, ..., exactly K units per registered population are
replaced. Initial and reset units must satisfy the common base maturity age.
Protection variants also require a previously reset unit's age to reach P.
Each learner selects its own eligible units with lowest bias-corrected
contribution utility. Counts and times are common; identities need not be.
Consolidation scores do not veto resets in v3.

All eligibility and replacement-value draws are checked before a reset is
applied. Insufficient eligible units cause an explicit failure. Incoming rows
are reinitialized, outgoing columns zeroed, and affected optimizer/reference
state cleared. Adjacent populations reset downstream first. This is not an
exactly function-preserving operation.

## Local gain and budget control

For an actually replaced unit of age a in [0, N), the gain variant assigns
its incoming rows and eligible outgoing feature columns the floor

`f(a) = g + (p - g) * (1 - a/N)`.

The configured study fixes p = 4g. Initial model units receive no newborn
window. Incident row/column floors combine by maximum; the classifier keeps
its common multiplier of one. Thus the terminal CNN adapter's incoming rows
receive the boost, while its outgoing classifier columns follow the shared
classifier rule. After N updates the floor ends.

The ordinary gain variant leaves all other feature multipliers at g. It
therefore adds nominal gain. The matched variant instead counts the feature
parameters incident to active newborn windows, holds their gains fixed, and
solves for a common multiplier h on all remaining feature parameters:

`h = (g * total_feature_parameters - sum(newborn_parameter_gains)) / mature_parameter_count`.

The count includes biases. Overlapping windows are counted once after the
maximum operation. An infeasible nonnegative allocation raises an error before
updating parameters or momentum. This is a causal per-step nominal budget
match; it changes mature-feature gain and is not a displacement or compute
match. The head is outside this feature budget.

## Consolidation and update bound

Only consolidation variants track a local post-reset path integral. At age
max(N, base maturity), each useful incoming row closes its accumulation window.
Its nonnegative utility is `sum(-gradient * applied_data_update)` divided by
squared displacement from the window start plus epsilon. Positive utility,
robustly normalized within the maturing rows, increases consolidation; the
first useful anchor copies the trained row. Incoming weight and bias rows are
handled separately. Classifier rows never consolidate.

Consolidation attenuates the allocated row gain by `(1-c)^gamma` and adds a
separate anchor force. It neither extends reset eligibility nor vetoes future
replacement. The gain-only and protection-only variants have no such force.

SGD momentum is formed before gating. For each parameter, form the full
optimizer proposal from its gated momentum/data update, gated weight decay,
and any anchor force. Let d be the vector of all those proposals, including
the head. Apply `alpha = min(1, maximum_update_norm / ||d||_2)` to the entire
vector. Store the ordinary unclipped momentum, and accumulate SI using only
the applied, clipped data component. Nonfinite gradients, proposals, or state
fail before the proposal is committed.

This bounds optimizer displacement in parameter space. It does not bound
changes in predictions, and subsequent structural resets are outside the
bound. The same rule and cap are used for all v3 comparisons.

## Recorded evidence

Every update records nominal feature gain before clipping and achieved gain
after clipping, clipping scale, proposed/applied optimizer norm, feature/head
data displacement, and actual resets per population. Detailed events also
record active windows, local maturation/consolidation, eligible populations,
reset identities, and consolidation's gain reduction. Checkpoints preserve
all clocks, private RNGs, utility, optimizer state, and allocation configuration.

The experiment runner additionally hashes actual ordered train/validation/test
tensors. This verifies pairing independently of metadata and checks that the
early-biased and independent curricula use bitwise identical evaluation sets.
Private replay sampling RNGs remain separate from monitoring draws.

The shared clip scale preserves within-update data-multiplier ratios, but
different models can have different clipping scales. Consequently nominal
budget matching need not survive as equal achieved budgets across learners.
Report both rather than interpreting nominal equality as equal learning.

## Close computational precedents

Preferential learning for new units and maturation-dependent protection have
substantial precedents. [Neurogenesis Deep Learning, Draelos et al.](https://arxiv.org/pdf/1612.03770)
trains newly added encoder units while freezing existing encoder weights and
using a smaller decoder learning rate, followed by stabilization. It expands
an autoencoder using reconstruction error and intrinsic replay. V3 instead
replaces units within fixed capacity and measures a local gain window in
optimizer updates.

[NICE, Gurbuz et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Gurbuz_NICE_Neurogenesis_Inspired_Contextual_Encoding_for_Replay-free_Class_Incremental_Learning_CVPR_2024_paper.pdf)
organizes neurons into age cohorts, freezes mature incoming connections, and
removes younger-to-older connections. Its ages advance at episode boundaries.
That protects mature knowledge from updates; v3's protection-only ablation
temporarily protects recently reset units from another replacement.

The [Mia et al. neuromorphic lifelong-learning preprint](https://arxiv.org/html/2508.04610v2)
combines high initial neuron-specific plasticity that decays with activity
history and pruning gated by age and specialization. This is particularly
close to the newborn-gain/protection motif. It uses spiking-network growth,
pruning, and activity-dependent STDP; v3 uses fixed-capacity replacement and
SGD. The paper describes a presynaptic factor, while the
[author implementation](https://github.com/zesun33/neuromorphic-cybersecurity-for-lifelong-learning/blob/main/bindsnet/bindsnet/learning/learning.py#L556)
applies its factor by target-neuron column. The broad mechanism correspondence
does not imply identical update equations or an independent reproduction.

Together with continual backpropagation, Synaptic Intelligence, and UPGD
cited in the research review, these works preclude presenting the component
ideas as inventions of this project. V3 contributes a particular implementation
and controlled comparison of those ideas, with explicit allocation and reset
budgets. Its experiments do not establish a priority claim or biological
equivalence.
