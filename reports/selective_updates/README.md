# Selective updates: portable evidence

This archive contains the engineering smoke cohort and the fixed, six-seed
first-introduction development cohort. Scientific interpretation belongs to
`development/summary.md`; the short smoke run checks engineering only.
The broader interpretation and matched-return diagnosis are in
[the research report](../adam_revision_results.md).

Each cohort includes the original configuration, completion record, scientific
source ZIP, prospective analysis ZIP, protocol, complete compressed raw episode
records, prefix records, analysis, and local checkpoint-audit results. The
stable-file check records hashes of the original training files. Model pickles
remain in the local ignored `runs/` directories; this portable archive does not
contain or load them.

The outer `artifact_manifest.json` seals every other file in this directory.
With Python and NumPy, verify a copied archive using:

```powershell
python verify_archive.py --input .
```

From the repository root, the equivalent command is:

```powershell
.venv/Scripts/python.exe scripts/verify_selective_updates_archive.py --input reports/selective_updates
```

The verifier checks exact archive coverage, bytes and hashes; source, protocol,
configuration and analysis locks; original raw JSON hashes; complete episode
and audit coverage; and independently recomputes the primary candidate's
acquisition, per-mode retention, survival, paired bootstrap intervals,
qualification and both declared screens. It uses no live training or analysis
imports. This is an independent calculation of recorded evidence, not an
independent scientific replication or a retraining of intermediate updates.

The verifier and its tests were added after the prospective experiment lock.
Their source is separately preserved in `verification_source.zip`. They do
not change the locked scientific analysis or decision rules. The archive's
engineering record preserves failed checks and their corrections.
