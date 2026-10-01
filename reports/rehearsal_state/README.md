# Adam: rehearsal state and paired environments

The [prospective protocol](../../docs/rehearsal_state_protocol.md) fixes the
question, intervention, fresh seeds and interpretation before scientific
fitting. The smoke cohort checks engineering only; the diagnostic cohort
contains the scientific results. Neither tests an adaptive memory policy.
Read the [narrative report](../rehearsal_state_results.md) and
[complete diagnostic tables](diagnostic/summary.md): both fixed replication
screens fail, so component findings remain descriptive.

Each assessment keeps the same retained candidates and original selection,
then rehearses them using eight combinations of earlier/later model weights,
complete Adam state and causal anchor. Two zero-update references measure
absolute prediction quality. Frozen predictors face paired current and
returning environments with identical observations and performed actions.
Both branches start from the same history and subsequently receive their own
causal feedback. Only the current branch enters ordinary learning.

Each completed cohort preserves raw assessment, phase, trajectory and auxiliary
JSON records, all-action prediction arrays and branch data, their original
hashes, configuration/runtime identities, source and analysis snapshots,
protocol, summary, independent local tensor audit and standalone figures.
Trusted checkpoints remain under the corresponding runs/rehearsal_state_*
directory. The portable check requires Python and NumPy, without checkpoints
or the live repository:

```powershell
.venv/Scripts/python.exe -I -B reports/rehearsal_state/smoke/verify_archive.py --input reports/rehearsal_state/smoke
.venv/Scripts/python.exe -I -B reports/rehearsal_state/diagnostic/verify_archive.py --input reports/rehearsal_state/diagnostic
```

Each cohort is sealed with artifact_manifest.json. The portable verifier and
its tests are packaging work, explicitly excluded from the scientific lock
and sealed separately in verifier_source.zip and verifier_lock.json. The
`-B` option prevents cache files from altering the strict archive. The same
command works with a copied cohort outside this repository.

The local tensor audit independently rebuilds every short rehearsal intervention
and diagnostic forecast, regenerates paired physical outcomes, checks causal
feedback and preserves ordinary replay/history. It does not independently
retrain every ordinary learning update. The portable audit checks original
record/array hashes, prediction arithmetic, seed aggregation, factorial and
environment contrasts, descriptive bootstrap intervals, the fixed replication
screen, and coverage of the tensor audit. It does not independently execute
the physical simulator or training. Descriptive rank/cue summaries and figures
are checksummed but not independently recalculated by that portable verifier.

Positive relative selection effects do not guarantee better absolute accuracy.
Artificial old/new state substitutions are controlled diagnostic comparisons;
they do not establish that reverting a component improves an ongoing learner.
Longer planning, an adaptive retention policy and compounding lifelong learning
remain separate questions.
