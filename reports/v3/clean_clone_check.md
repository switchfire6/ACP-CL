# Local clean-clone reproducibility check

**Passed: all seven v3 methods completed the CPU engineering smoke from a fresh, noneditable installation of a local Git clone.**

Clone commit: `7fbb270cf0d107261d60577602001af605f67601`. Source SHA-256: `cccbb40b2fe578be511f49c31d9629abd823f89bf078ccc9781b1d38c17c5efe`. Every installed Python source file matched the clone and frozen parent source byte-for-byte. The clone remained clean after the run.

The fresh CPU environment used Python 3.10.11, Torch 2.8.0+cpu, NumPy 2.2.6, and torchvision 0.23.0+cpu. Dependencies and the local package were installed from cached artifacts without network access. The interpreter ran with `-I`; user-site packages and `PYTHONPATH` were excluded. `acp_cl.__file__` resolved to `runs/public_clone_check/.venv/lib/site-packages/acp_cl/__init__.py` relative to the parent repository, not its editable source installation.

Command, executed inside the clone:

```powershell
.venv\Scripts\python.exe -I -m acp_cl run --config configs/v3_smoke.json --output runs/clean_clone_smoke --device cpu --no-plots
```

| Method | Experiences | Updates | Unit resets | Maximum optimizer displacement | Cap |
|---|---:|---:|---:|---:|---:|
| consolidation_v3 | 4 | 32 | 7 | 0.044282644 | 0.05 |
| er_v3 | 4 | 32 | 0 | 0.043904063 | 0.05 |
| full_v3 | 4 | 32 | 7 | 0.044282555 | 0.05 |
| newborn_matched_v3 | 4 | 32 | 7 | 0.044191482 | 0.05 |
| newborn_v3 | 4 | 32 | 7 | 0.044282556 | 0.05 |
| protection_v3 | 4 | 32 | 7 | 0.044282644 | 0.05 |
| recycle_v3 | 4 | 32 | 7 | 0.044282644 | 0.05 |

All checkpoints completed four experiences; all metrics were finite. Replay has no recycling events. The other methods reset one adapter unit at updates 8, 12, 16, 20, 24, 28, and 32. Every per-step schedule and optimizer displacement passed, using a numerical cap tolerance of 1e-6. The cap includes head, feature, decay, and anchor updates and excludes the separate recycling reset. Actual stream hashes and the first four warmup allocation traces matched across all methods.

[Machine-readable proof and command outputs](clean_clone_check.json) records identities, file hashes, every command's return code/output, and the local audit script location. Raw command logs, exact absolute import provenance, weights, and run data remain under the ignored `runs/public_clone_check` folder. The first explicit-registry cache lookup failed; the second used the configured default registry cache and succeeded offline.

This was a local Windows CPU check, not a hosted CI run. It verifies installation and smoke execution, not cross-device equality or evidence for the research hypothesis.

The first audit assertion incorrectly expected a monitor event on every update.
The audit was corrected to use the configured four-update monitor cadence;
all 32 optimizer updates were checked separately in the allocation trace.
Both nonzero command attempts remain in the proof. No training source or
experiment result was changed to pass the audit.
