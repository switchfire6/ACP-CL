# Saved-output diagnostic: interpretation and effort allocation

Specified 2026-09-28 03:18 UTC, after the completed core/residual experiment and before computing this diagnostic's new scores. This is a retrospective diagnostic with decision rules fixed before its scoring, not a prospective confirmation of the original hypothesis. The original primary screens failed for both architectures and remain failed. This document allocates further effort; it does not redefine those screens.

## One bounded question

Does the frozen-core architecture retain a useful old predictor while its additive correction helps the new condition and interferes with the old one, or is its immediate return difficulty primarily sensitive to the observed context? A saved-output check can distinguish these operational signatures. It cannot show that an autonomous learner knows when to select a predictor.

Use all six completed seeds 16001–16006, both conditional and recurrent models, and only the joint and separate arms. No training, additional seeds, parameter sweeps, or selection of a favorable subgroup. Evaluate four fixed panels:

| Panel | Stored parameters | Target law |
|---|---|---|
| Novel endpoint | Novel-end checkpoint | Novel law |
| Return entry | Reuse novel-end checkpoint after verifying its full learner signature equals the recorded return-start signature | Original base law |
| Return endpoint | Return-end checkpoint | Original base law |
| Novel after return | Return-end checkpoint | Novel law |

Within every panel, use identical archived query inputs and truth for all output variants, paired across arms. Evaluate actual saved history once and two full fresh target-law support replicates, averaging those two replicates within seed. Use the archived cue-flip queries for the two novel-law panels only. Fresh support is supplied by the evaluator and is a context diagnostic, not an online mode-recognition mechanism.

Use all-case, all-action Brier for old-law targets and affected-subset, all-action Brier for the novel-law allocation contrast. Report greedy terminal survival and cue-flip effects descriptively. Report all panels and variants; no endpoint, cue, or output selection based on the results.

Before the main diagnostic, an engineering check reuses archived smoke seed
16991 for both models. That original smoke has one refreshed support replicate,
so its output panel has 20 rows per model rather than the main panel's 32. It is
ineligible for effort-allocation decisions. No missing support is fabricated and
no old run is changed.

## What the interventions mean

The trained model adds core and residual **logits**, not probabilities. For the conditional model, write support and query logits as \(z_C^s,z_C^q,z_R^s,z_R^q\). Compute native evidence weights \(w_C\) from core support logits and \(w_F\) from combined support logits, using the existing terminal-horizon likelihood rule. Evaluate the fixed four-cell crossing of query logits \(z_C^q\) versus \(z_C^q+z_R^q\) with weights \(w_C\) versus \(w_F\). This includes native core and full outputs, plus routing substitution at combined query logits and query-correction deletion at full routing weights. Preserve sigmoid, expert mixing, and monotone survival projection.

For descriptive residual-only output, use residual query logits and evidence weights recomputed from residual support logits. In the recurrent model, evaluate full, core, and residual logits with each pathway's own context computation and the unchanged probability transformation; there is no conditional expert-routing factor to cross. All variants receive the same external support within a comparison.

The residual was optimized as a correction, so its standalone calibration is not identified by the training objective. Poor residual-only scores do not establish that it learned nothing. Jointly trained components can co-adapt, making their separate predictive performance an intervention on this parameterization, not a unique decomposition of stored knowledge. The separately trained, subsequently frozen core has a stronger historical interpretation, but disabling its correction still does not demonstrate a usable switching policy. Conditional crossovers explain local output sensitivity, not a unique cause of the learning trajectory.

## Fixed allocation rules

Apply this hierarchy separately to each architecture. Average the two refreshed-support replicates within each seed first. Use the six paired seeds as the statistical units; report their values and 20,000 paired seed-bootstrap intervals with seed 17291. Intervals are descriptive and are not additional gates. The following margins are effort-allocation heuristics, not statistical equivalence claims or validated deployment criteria.

**GO: specify at most one fresh causal-combination experiment.** In the separate arm, define

\[
G_N=L_{\mathrm{core}}(\text{novel endpoint})-L_{\mathrm{full}}(\text{novel endpoint}),
\qquad
H_O=L_{\mathrm{full}}(\text{return entry})-L_{\mathrm{core}}(\text{return entry}).
\]

Here \(G_N\) uses novel affected-subset Brier and \(H_O\) uses old all-case Brier, both with refreshed support. Require mean \(G_N\ge .002\), mean \(H_O\ge .005\), and each contrast strictly positive in at least five of six seeds. This signature motivates testing whether observable evidence can control combination without sacrificing acquisition, retention, and return performance against the joint baseline. It does not qualify an existing hybrid, authorize choosing the best cell, or demonstrate generality across architectures.

**CONTEXT PIVOT: only if GO fails.** At the fixed return-entry old-law panel, use full-predictor all-case Brier. Require all three conditions:

1. Mean separate-minus-joint error with actual history is strictly positive.
2. Separate actual-minus-refreshed error is at least .005 on average and at least .005 in five of six seeds.
3. Mean separate-minus-joint error with refreshed support is at most .005.

This motivates one bounded question about which available feedback makes the environmental condition identifiable. It does not establish equivalence to joint training, successful autonomous context inference, or a deployable fix. The endpoint is fixed in advance; stronger effects elsewhere cannot rescue this rule.

**STOP this mechanism line if neither rule holds.** Archive the negative result and stop whole-core-freezing/residual variants on this cohort, including width, rehearsal-dose, or gate sweeps. Do not pool architectures, promote a cue subgroup, or replace these rules with the most attractive descriptive crossover. Stopping this line does not imply that continual learning is impossible. Any future experiment remains a separate decision with a fresh protocol and budget.

## Interpretation safeguards

Frozen separate-core predictions must agree across checkpoints when both query and support are identical. Changed actual history can change predictions despite identical weights. Novel endpoint and return entry can present identical observable inputs before return feedback while their physical target laws differ. Selecting core for one and full for the other using known target identity would therefore be an oracle intervention. A useful future mechanism must rely on past observations and pay any recognition delay.

Preserve input files and verify complete loaded-state invariance: parameters, optimizer state, gradients, replay state and RNG, history, and module training flags. Use copied predictors or exact flag restoration. A failure of these checks blocks scientific interpretation and requires a documented correctness repair, not a revised allocation threshold. Endpoint contrasts do not measure acquisition AUC, isolate forgetting during return, or establish long-term compounding benefit.

## Return on effort

The project has produced useful constraints: context-dependent reuse exists in this testbed; replay helps retention; changing replay or update history can trade acquisition against retention; and the recurrent residual learned useful raw-input features compared with its fixed-random-feature control. It has not produced a candidate that meets the combined new-learning and retention objective over the established baselines. Neither a general relational learning principle nor indefinite capacity renewal or compounding advantage has been established.

The risk of spinning wheels is now real if each failed recipe leads to another retrospective diagnosis and nearby tweak in the same world. Careful audits make the evidence trustworthy; they do not turn negative algorithmic results into a breakthrough. This small saved-model diagnostic is justified as a way to close a concrete output-combination/context question. Its value depends on honoring the stop rule. A qualifying signature merits specifying one fresh causal test; otherwise, retain the benchmark and negative findings, and reconsider the representation-learning task or mechanism family before spending more training effort.
