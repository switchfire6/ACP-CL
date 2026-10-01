# Independent review of two preservation discrepancies

Reviewed 2026-09-28 at approximately 22:04 UTC. This review read the existing
files and records; it did not modify any scientific file, inventory or failed
check.

**Both flagged files currently match their original canonical contents.**
Three independent reads agree: PowerShell `Get-FileHash -Algorithm SHA256`,
Python SHA-256 of `Path.read_bytes()`, and Python SHA-256 accumulated through
1 MiB binary reads. Their hashes and sizes match the earlier output-study
preservation inventory created at 03:16:55.152230 UTC and the original
core/residual study records listed below.

The new [initial inventory](preservation_before.json), created at
21:32:33.798919 UTC, contains two conflicting digests. The
[comparison at 22:01:53](preservation_after.json) checked 3,314 files, found none
missing and flagged these two differences. **That comparison remains failed.**
The reason for its two conflicting initial entries is not established by this
review; matching current bytes and old modification times do not establish
what happened during the earlier inventory operation or exclude transient
events.

## Archived evaluation arrays

Path:
`reports/core_residual/development/arrays/recurrent_16004/joint/return/evaluations.npz`

- Size: 7,988,449 bytes in both inventories and the current file.
- Current SHA-256, agreed by all three reads:
  `a5b72be71c87b30ee453418d891bd8221a66de272acc9c56d8275906cba02cf2`.
- Conflicting new initial-inventory SHA-256:
  `7423904c5763b90796ad83a6970d1736286c3717f8ccaf12dc9543db21fca6ad`.
- Current filesystem modification time: `2026-09-28T02:40:20.6793288Z`.

The current digest is recorded in the earlier output-study inventory,
`reports/core_residual/development/artifact_manifest.json`, both original
development file-integrity records, and the original run phase's
`result.json` artifact hashes. The separate original run copy at
`runs/core_residual_development/recurrent_16004/joint/return/evaluations.npz`
also currently hashes to the same value.

## Original evaluation descriptors

Path:
`runs/core_residual_development/recurrent_16006/separate/maintenance/evaluations.json`

- Size: 57,298 bytes in both inventories and the current file.
- Current SHA-256, agreed by all three reads:
  `ada5cb53f9a187db0ae86d9bf023ca10d77268becddd63ad6590d90531aa5c5f`.
- Conflicting new initial-inventory SHA-256:
  `be142de7878ea3c99acff0d241d1f0f78e4bb376583b94ceacdc7c01ef642b11`.
- Current filesystem modification time: `2026-09-28T02:34:43.8660385Z`.

The current digest is recorded in the earlier output-study inventory, both
original development file-integrity records, and this original phase's
`result.json` under both `evaluation_files.json.sha256` and `artifact_hashes`.

## Provenance records checked

The historical reference is
[core_residual_output_checks/preservation_before.json](../core_residual_output_checks/preservation_before.json).
Each of the following provenance files was independently hashed and also
matches its own entry in that 03:16 UTC inventory:

| Provenance file | Current and earlier recorded SHA-256 |
|---|---|
| `reports/core_residual/development/artifact_manifest.json` | `edd6c32d2b69722985b6548ff0af16d7e8c2bfb34d528e79e203cb88dbeb6759` |
| `reports/core_residual/development/file_integrity.json` | `ed6d7048fc7ca83d2be0fce9d90efde86afa960acdeb723b8dab0f52f0af33dc` |
| `reports/core_residual_checks/development_file_integrity.json` | `ed6d7048fc7ca83d2be0fce9d90efde86afa960acdeb723b8dab0f52f0af33dc` |
| `runs/core_residual_development/recurrent_16004/joint/return/result.json` | `90b8ed99e565742d793ba136795a8d79ba813a238d12310c15a06c2cb25ed239` |
| `runs/core_residual_development/recurrent_16006/separate/maintenance/result.json` | `a5df27ad5fbf04edcd7a368b673aad11c50476facfef67087574a0c2e8800ea3` |

This supports the narrow conclusion that the two questioned files' current
scientific contents agree with the preserved original records. It does not
turn the new initial-inventory comparison into a pass, identify a cause for
the discrepancies, or independently recheck all 3,314 files. Keep the initial
inventory, failed comparison and this review together.
