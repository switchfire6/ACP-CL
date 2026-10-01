# Joint-persistence development ledger

The mechanism comparison has not started at the time of the first entries.
Development examines the uniform baseline and benchmark validity. Engineering
smokes exercise branches and are not treated as comparative evidence.

1. **World v1 / `runs/persistence_development_v1`.** Seed 117, uniform replay,
   1,024 arrivals/block, three updates/batch, 128 memories, width 64, lr .002.
   Learning was adequate, including 34.6% -> 62.2% survival on the late gate.
   However, the best fixed action within each ordinary context was within
   0.3 percentage points of the clairvoyant action bound. Persistent supply
   advantage made donating in the appropriate direction almost universally
   sufficient. This would be a weak test of context-dependent relationships.
   Preserve the original manifest, checkpoint, source archive and curves.
2. **World v2.** Before any mechanism comparison, change mean supply from 2.05
   to 1.9 per participant and swap supply advantage halfway through each
   12-step trial. Donors must bridge their later shortage; reserves and losses
   can make a transfer counterproductive. This finite-horizon world does not
   imply sustainable indefinite survival. Keep all learner settings unchanged
   and repeat the same baseline-only development seed.
3. **World v2 result / `runs/persistence_development_v2`.** With the unchanged
   learner, the first context reaches 82.0% joint survival against 65.4% for the
   best fixed action and 87.2% for the clairvoyant bound. The initial four
   context endpoints are 82.0%, 66.7%, 62.8%, and 77.6%; all exceed no transfer.
   The late gate improves from 67.4% to 83.9%. This supplies a usable acquisition
   signal and a reason to condition action on reserves and observations.
   Keep the common learner settings; do not tune the combined memory policy.
4. **Pre-comparison schedule correction.** The no-return schedule's late gate
   is based on the new composition rather than the retired target. Otherwise
   gate-zero trials would silently reintroduce the allegedly retired context.
   This changes neither development run, which both used the long schedule.

The earlier CPU engineering smoke used tiny batches and untrained predictions;
its method differences were not used to select the memory rule or settings.

5. **Post-lock benchmark audit, before aggregate comparison.** The four initial
   factor combinations correlate source direction with loss/delay glyph parity.
   Consequently, successful performance on those conditions need not use the
   moving indicator. The withheld combinations break that correlation, but the
   initial training data do not uniquely identify its causal interpretation.
   Keep the frozen benchmark and report this limitation; do not present strong
   known-context scores as evidence of causal representations. The planned
   history-removal probe is a sensitivity check, not proof of semantic learning.

Portable copies of both development runs, their original source archives and
all curves are in `development_v1` and `development_v2`. The mapping and hashes
in `development_artifacts.json` point back to the untouched raw run directories.

6. **Analysis arithmetic correction after completion.** The combined rule and
   the coverage control tie on mean long-gap return survival. Averaging their
   floating-point paired differences initially left a +1.39e-17 residual and
   incorrectly marked the strict positive-difference subcriterion as passed.
   Normalize numerical zeros below 1e-12 and add a regression test. The raw
   outcomes, endpoints, primary difference, and overall failed continuation
   decision are unchanged. No training reruns or parameter changes were made.
