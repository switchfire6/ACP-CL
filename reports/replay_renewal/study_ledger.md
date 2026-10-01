# Adam replay-renewal study ledger

Two main experiments were specified together before either cohort ran. The
source, configurations, protocol and analysis formulas were saved before main
training. The equal 8/8 split was fixed without a hyperparameter search or
policy-development cohort. The 8,192-arrival budget comes from the previous
study's separately archived qualification work, not the new treatment outcomes.

Engineering checks: 26 new tests; full suite 733 passed, two existing Windows
symbolic-link skips; repository ruff check passed. Tiny end-to-end cohorts used
seed 19, 16 arrivals per challenge, both architectures: 14 decomposition and
48 policy episodes. Both full checkpoint audits passed. Their fresh-reference
qualification failed, as expected at this deliberately inadequate engineering
budget. These runs are preserved under `engineering/`; they neither selected
the research settings nor establish learnability.

Main reset decomposition: seeds 9001--9006; 12 prefixes and 84 first-introduction
episodes. Main continuous policy comparison: seeds 10001--10006; 36 prefixes
and 288 episodes, including 36 fresh-reference fits. Each study has six paired
seed replicates per architecture. The first diagnostic has only two fresh
references per first-introduced cue.

Both main cohorts were locked before decomposition training began, with shared
training source SHA256
`6f3561f485bf5dad9577be231b2542bba0c4682ac0b847e03f385b1c57271d39`.
Runtime: four CPU workers, one Torch thread each, Windows affinity mask 85.
Worker affinity is checked against the manifest. Flush-before-replace writes
address the earlier checkpoint durability weakness; this is not a guarantee
against all hardware or storage failures.

Reset decomposition completed all 84 episodes and 12 prefixes without an
interruption. All eight fresh-model qualification groups passed; smallest cue
benefit .005326 versus .002. Its full audit passed 12 prefixes, 84 starts and
84 final states/probes, and all 280 stable files passed integrity checks.
The continuous policy cohort ran from 2026-09-27 19:06 to 19:52 UTC using its
already locked specification, with no interruption, exclusion or retry. All
288 episodes and 36 prefixes completed. All 12 fresh-reference qualification
groups passed. The fixed split failed the prospective combined screen in BOTH
architectures, due to retention and extra matched-noise penalties; acquisition,
consistency, qualification, final prediction means and survival gates passed.
No treatment result selected settings, seeds or thresholds.

The policy audit passed 36 prefixes, 288 starts and 288 final states/probes,
including 36 matched clean/noisy pairs. All 940 stable files passed integrity
checks. Together with decomposition, this is 792 audited model checkpoints
and 1,220 inspected files. Both scientific cohorts used the unchanged training
source and prospective analysis. Plotting, documentation and archive utilities
were completed afterward; none changed the locked analysis formulas or gates.

The interactive memory illustration uses a separate deterministic demonstration
sequence. It is not a display of measured training outcomes. Its time control,
small-screen layout, and light/dark appearances were checked in the browser.
Its source and check record are included in the archive.

The read-only prior-archive check initially encountered an older manifest
schema while scanning the persistence archive. The reader was corrected for
that schema; all 198 files in five prior archives passed their original hashes.
No prior archive was modified. This bookkeeping correction affected no training
or analysis formula. See `prior_archive_check_before.json`.
