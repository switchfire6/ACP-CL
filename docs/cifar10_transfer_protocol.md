# CIFAR-10 transfer pilot: frozen v3 allocation recipe

Written before pilot outcomes. This is a bounded development experiment, not a
confirmatory study or a reproduction of a published benchmark protocol. The
implemented local gain rule and optimizer remain v3; the new work is the data
stream, exposure audit, and transfer comparison.

## Question and budget

Does the small procedural-shape advantage of a post-replacement gain window
persist on natural base images with fixed labels and one current arrival per
base image? Compare `er_v3`, `recycle_v3`, `newborn_v3`, and
`newborn_matched_v3`, with two previously unused paired seeds **1063 and 1174**,
on recurring and stationary streams. Complete **16 runs, 22,560 optimizer
updates**. No parameter sweep, extra methods, or outcome-dependent extension
belongs to this pilot. Engineering smoke results are separate.

The primary diagnostic contrast is newborn minus recycling on recurring late
early-acquisition AUC. Matched newborn minus recycling is the allocation
control. Final accuracy and the replay-only baseline guard against trading
retention for that AUC. Two seeds cannot establish efficacy or statistical
significance. Report each paired value; do not present a two-seed bootstrap as
reliable population uncertainty.

## Data and fixed transformations

Use the canonical [CIFAR-10 Python archive](https://cave.cs.toronto.edu/kriz/cifar.html),
MD5 `c58f30108f718f92721af3b95e74349a`, SHA-256
`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
It contains 5,000 training images for each of ten labels. The downloaded
archive also contains the official test set; test images/labels are not
materialized into this stream or evaluated. Dataset integrity checks may read
archive or batch-file bytes.

For each seed, split source indices before constructing any transformed views.
Reserve 500 training indices per class for validation, use a fixed 50/class
panel from that reservation, and leave the other 450/class unused. Partition
the remaining 4,500/class into **30 experiences of 150/class**. Every experience
contains all ten labels, preserving original label IDs. No transformed sibling
of a training image can appear in validation.

Recurring experiences cycle through `original`, `grayscale`, and `blur`, ten
occurrences each. Stationary experiences always use the original view with the
same ordered base-image IDs and labels. Transformations precede normalization:

- Original: unchanged uint8 RGB pixels.
- Grayscale blend: compute `gray = .299 R + .587 G + .114 B` in float32, then
  `0.2 RGB + 0.8 gray` in each channel; round to nearest even, clamp to [0,255],
  and store uint8.
- Blur: a normalized 3 by 3 Gaussian kernel with sigma 1.0, reflection padding,
  per-channel float32 convolution; round to nearest even, clamp, and store uint8.

All views use `(pixel/255 - 0.5)/0.5`. There is no random crop, flip, or further
augmentation. These artificial domain changes on natural images are a bridge
experiment; they do not establish performance under natural distribution shift.
Validation reuses one immutable panel per domain. Those views share base images
and repeated experiences are correlated measurements, not independent samples.

## Learner, exposure, and evaluation

Carry the v3 width-64 CNN with one fixed ten-output classifier; recycle only
its final adapter. Use current batch size 32, reservoir capacity 256, replay
batch size 32, learning rate .03, momentum .9, weight decay .0001, and the same
monitoring settings as v3. Use `single_pass: true`, `epochs: 1`, and omit
`steps_per_experience`. Each experience yields 46 batches of 32 and one of 28:
**1,500 unique arrivals, 47 updates**, totaling 45,000 current presentations and
1,410 updates per run. Replay may repeatedly sample stored examples; it inserts
each current base index only on its unique arrival.

The first 40 updates use feature gain 1. Thereafter the mature gain is .5;
newborn gain starts at 2 and decays linearly to .5 over 30 updates. The matched
variant pays for newborn gains by reducing mature-feature gain to preserve the
per-step nominal feature mean. Head gain stays 1. Recycling resets three units
every 20 updates after warmup, starting at update 60: **68 events, 204 unit
resets** per recycling run. Reset identities follow each learner's own utility.
No global controller, extra protection, or consolidation is added.

Every method shares the .1 bound on the complete optimizer proposal. Structural
resets are outside it. Nominal gain equality does not imply equal actual
displacement, clipping, or compute. See the [v3 allocation rules](algorithm_v3.md).

Evaluate the current panel before training and every four updates, plus the
final short batch. Early AUC ends at 256 current presentations, exactly eight
full updates; late AUC averages experiences 16-30. It mixes retained knowledge,
transfer, and adaptation. No scratch reference is run. The untrained original
baseline is the first point of the first experience's curve.

Full retention evaluation occurs after experiences 10, 20, and 30. Recurring
final accuracy averages the three trained domain panels equally (equivalently,
the final row's 30 equally repeated domains). Stationary final accuracy and
drawdown concern its original-view panel only. Stationary drawdown uses all
30 end-of-experience accuracies on that identical panel; it can still miss
within-experience damage. Report per-domain recurring endpoint values too.

Cache repeated validation objects only within one model snapshot, starting
with that snapshot's already measured current-panel accuracy. Discard the
cache before any further update. Cache-on/off tests must preserve metrics,
model/RNG state, and training exposure, with only evaluation cost changing.

## Effort-allocation rule fixed before outcomes

These are practical thresholds for deciding whether more work on this recipe
is justified. They are not statistical tests, confidence bounds, or success
criteria for the broader critical-period hypothesis.

1. **Adequacy on each seed:** stationary replay and recycling both reach at
   least 25% final original-view accuracy and improve at least 10 percentage
   points over their own untrained baselines. Recurring recycling reaches at
   least 20% final three-domain accuracy and has late AUC below 95%.
2. **Signal to pursue:** recurring newborn-minus-recycling late AUC is positive
   on both seeds and averages at least 1 pp. Matched-minus-recycling is
   nonnegative on both and averages at least .5 pp.
3. **Retention guard:** neither gain variant loses more than 1 pp final accuracy
   against either recycling or replay on any seed/condition. Stationary uses
   only the original view.
4. **Decision:** advance this frozen recipe only if all three checks pass.
   Otherwise stop scaling it from this pilot. Inadequate baselines motivate a
   separately declared optimization investigation; adequate baselines with a
   failed signal/guard motivate simplifying or abandoning this operating point.

Complete and archive all 16 runs regardless of interim performance. Never add
seeds, exclude an unfavorable method, or adjust settings to reverse the
decision. Numerical failures are recorded as failures and prevent advancement;
infrastructure retries retain the same source/configuration and are disclosed.
A different algorithm or tuning study requires a separate plan. A favorable
decision would justify a larger independent study with fair tuning budgets,
not an efficacy claim.

## Audit and interpretation

Before execution, commit source, protocol, configurations, and the lock artifact.
Record canonical source/runtime/config hashes, dataset source fingerprints,
split and actual shuffled-arrival IDs, stream hashes, replay state, curves,
allocation traces, reset counts, clipping, displacement, costs, and failures.
Verify exact one-pass counts and paired arrivals across methods/conditions.
Validation information and domain IDs remain outside the learner.

Known components have close predecessors in continual backpropagation,
neurogenesis, and maturation; see [the attribution review](algorithm_v3.md#close-computational-precedents).
The useful outcome is a falsifiable transfer result and a reusable controlled
experiment. This short stream does not demonstrate prevention of long-horizon
plasticity loss, outperform the cited papers, or establish scientific novelty.
