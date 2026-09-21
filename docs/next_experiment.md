# Natural-image follow-up: original proposal

**Implemented as the [CIFAR-10 transfer pilot](cifar10_transfer_protocol.md).**
The [completed 16-run report](../reports/cifar10_transfer_results.md) applies
the predeclared rule: adequate baselines, failed signal/retention requirements,
and a decision to stop scaling the frozen recipe.
The text below preserves the original design rationale. The linked protocol
and [run lock](../configs/cifar10_transfer/lock.json) specify the implemented
counts, transformations, endpoints, seeds, and continuation rule. In particular,
stationary final accuracy concerns only the original view; its drawdown uses
all 30 end-of-experience measurements on that same panel.

The question is whether the v3 local allocation rules help on natural images
when labels remain fixed, current examples arrive once, and domains recur.

## Why CIFAR-10 with fixed labels

The current procedural generator has four labels and a narrow family of image
structures. Removing color-label correlation addresses one shortcut, but does
not remove every advantage of that generator. The existing CIFAR-100 loader
introduces disjoint classes, changes selected classes between subset seeds,
and uses repeated current presentations. That is a useful class-incremental
question, but changes several factors at once.

CIFAR-10 has 5,000 training and 1,000 test images per class, for ten classes.
These counts come from the [official dataset description](https://cave.cs.toronto.edu/kriz/cifar.html).
Reserve 500 official-training images per class before constructing any domain
views. The remaining 4,500 per class support 45,000 unique current arrivals.
Keep the official test set outside pilot decisions.

Use three deterministic domain views: original color, a fixed grayscale blend,
and fixed mild blur. Specify their exact coefficients, kernels, implementation,
and application order in the new protocol before execution. These are proposed
transformations of natural images, not an established benchmark protocol or a
test of natural distribution shift. A later external benchmark remains needed.

## Initial pilot design

Prefer **30 experiences, 150 distinct training images per class per
experience**, cycling evenly through the three domains. At batch size 32,
each experience has 46 full batches and one batch of 28: 47 updates, 1,410
updates per run. Set `epochs: 1` and omit `steps_per_experience`; record base
image IDs to verify each current arrival is unique. Replay can repeat stored
examples as intended.

A 90-experience alternative with 50 images per class gives 1,440 updates, but
each domain lasts only 16 updates. Its 30-update newborn window would typically
span multiple domains. The 30-experience plan keeps a comparable total budget
and allows an entire window within a domain. Neither is a claim about 30 or
90 independent tasks.

Compare four methods: replay (`er_v3`), replay plus recycling (`recycle_v3`),
newborn gain (`newborn_v3`), and the nominal-budget-matched variant
(`newborn_matched_v3`). Include a stationary original-image condition with
exactly the same base-image arrival order and label sequence. Two fresh paired
pilot seeds across both conditions give **16 runs and 22,560 optimizer
updates**. These are development results, not a new confirmatory cohort.

Initially carry the v3 width-64 CNN, fixed ten-output classifier, replay
capacity 256, replay batch size 32, 40-update warmup, mature feature gain 0.5,
newborn peak 2.0, 30-update window, three resets every 20 updates after warmup,
and 0.1 optimizer-displacement cap. That schedule has 68 reset events, or 204
unit resets, per recycling run. Carrying these settings tests transfer of a
recipe; it does not make them equally tuned or optimal on CIFAR-10.

Freeze normalization, augmentation, domain order, split identities, and all
other optimizer settings before running. Do not add global reopening or the
combined consolidation/protection policy to this first comparison. Replay
alone is essential for checking whether recycling itself helps in this setting.

## Measurements and implementation requirements

- Construct train/validation splits by base-image ID before transformations;
  no transformed sibling of a training image belongs in validation. Use a fixed
  50-image-per-class validation panel from the reserved pool for this pilot,
  leaving the other reserved images unused. Domain views share base images
  and are correlated measurements.
- Record base IDs, label mappings, tensor hashes, rendering definitions, and
  independent RNG streams. Pair underlying arrivals across methods and the
  stationary/recurring conditions. The learner receives images and labels,
  with domain metadata available only to the evaluator.
- Report early AUC through the first 256 current presentations (exactly eight
  full updates), late AUC, final accuracy equally weighted across all three
  domains, and stationary drawdown on one unchanged validation pool. Include
  untrained baselines, all seed values, clipping, achieved gain, and reset counts.
- Add a fixed-label CIFAR-10 stream builder and explicit preprocessing dispatch.
  The existing CIFAR-100 configuration cannot express this experiment. Verify
  exact single-pass counts and replay insertion identities before any pilot.
- Cache identical validation pools only within the same model snapshot, if
  needed for speed. Repeated domain evaluations are not independent samples,
  and predictions must never be cached across training updates. The current
  runner would otherwise repeatedly score the same prior-experience pool.

If the pilot cannot learn the stream, resolve ordinary optimization and
exposure adequacy on development data before testing a mechanism claim. If it
learns, use its variance and resource measurements to design a separately
locked multi-seed evaluation with a declared tuning budget. Preserve null
results. A favorable pilot alone should not trigger a claim that critical
periods solve continual learning.
