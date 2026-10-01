# Predictive rehearsal value: portable evidence

This directory contains the disjoint engineering smoke cohort and the fixed
scientific diagnostic. Interpret the [narrative report](../predictive_value_results.md)
and [diagnostic summary](diagnostic/summary.md) before reading the smoke results;
the tiny smoke cohort cannot qualify a scientific continuation decision.

Each cohort contains raw assessment and phase records, original record hashes,
configuration/runtime identities, training source, the prospective protocol and
analysis snapshots, arithmetic summaries, and a local tensor audit. Trusted
model checkpoints remain in the corresponding runs/predictive_value_* directory.
They are not needed for the portable arithmetic check.

From the repository, with Python and NumPy installed:

```powershell
.venv/Scripts/python.exe -I -B reports/predictive_value/smoke/verify_archive.py --input reports/predictive_value/smoke
.venv/Scripts/python.exe -I -B reports/predictive_value/diagnostic/verify_archive.py --input reports/predictive_value/diagnostic
```

The script also works on a copied cohort outside this repository. It reads no
model pickle or live scientific source. Its final source and tests are sealed
separately in verifier_source.zip and verifier_lock.json, as declared before
scientific fitting. Each cohort's artifact_manifest.json checks all its files.

The portable check recomputes recorded performed-action Brier and reported
greedy-action survival, candidate selection, exact uniform expectations,
paired seed means, descriptive bootstrap intervals and fixed research screens.
The local tensor audit independently recreates the short rehearsal interventions
and all diagnostic forecasts. It does not independently retrain every ordinary
learning update. Portable verification checks that audit's hashes and coverage;
it does not substitute for executing it or for fresh-seed scientific replication.

Selection never changes the ordinary uniform-replay learning trajectory. An
observed benefit therefore concerns a proposed score for additional rehearsal,
not demonstrated improvement of an adaptive memory policy, memory deletion,
noise robustness or lifelong compounding learning.

A separate [retrospective ranking diagnosis](../predictive_value_checks/rank_stability/results.md)
was specified after viewing the primary failures. It uses existing candidate
losses only, has its own dated design and source/input hashes, and cannot change
the prospectively specified decisions. It is outside the two scientific cohort
archives so its later status remains explicit.
