"""Inventory every planned V3 result and its provenance, without ranking outcomes.

No output is written until the complete development and locked evaluation grids
pass validation. Engineering smoke suites are listed separately and never added
to comparative evidence. Existing conflicting inventories are preserved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

if __package__:
    from . import prepare_v3_study as study
    from .summarize_v3_development import validate_selection
else:
    import prepare_v3_study as study
    from summarize_v3_development import validate_selection


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING_ROOTS = {
    "gpu_smoke": "runs/v3_gpu_smoke",
    "cpu_smoke": "runs/v3_cpu_smoke",
    "clean_clone_cpu_smoke": "runs/public_clone_check/runs/clean_clone_smoke",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("inventory inputs and output must stay inside the workspace") from error


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _coverage(directory: Path, configs: dict[str, dict]) -> None:
    """Find stray/duplicate results even outside a validator's one-level glob."""
    if all(configs) and directory.is_dir():
        if {path.name for path in directory.iterdir() if path.is_dir()} != set(configs):
            raise ValueError("missing or unexpected comparative suite/attempt directories")
    expected_results = {
        directory / cohort / f"{method}_seed{seed}" / "result.json"
        for cohort, config in configs.items()
        for method in config["methods"] for seed in config["seeds"]
    }
    actual_results = set(directory.rglob("result.json"))
    if actual_results != expected_results:
        missing = sorted(str(path.relative_to(directory)) for path in expected_results-actual_results)
        extra = sorted(str(path.relative_to(directory)) for path in actual_results-expected_results)
        raise ValueError(f"missing or unexpected/duplicate result paths: missing={missing}, unexpected={extra}")
    expected_manifests = {directory / cohort / "manifest.json" for cohort in configs}
    if set(directory.rglob("manifest.json")) != expected_manifests:
        raise ValueError("missing or unexpected suite manifests")
    for cohort, config in configs.items():
        expected_directories = {f"{method}_seed{seed}" for method in config["methods"] for seed in config["seeds"]}
        actual_directories = {path.name for path in (directory / cohort).iterdir() if path.is_dir()}
        if actual_directories != expected_directories:
            raise ValueError("unexpected or incomplete attempt directory in a comparative suite")


def _run_record(root: Path, directory: Path, config: dict, manifest: dict,
                method: str, seed: int, phase: str, cohort: str) -> dict:
    location = directory / f"{method}_seed{seed}"
    path = location / "result.json"
    result, result_hash = study._load(path)
    if result.get("method") != method or type(result.get("seed")) is not int or result["seed"] != seed:
        raise ValueError("result method/seed does not match its planned directory")
    if result.get("config") != config or result.get("config_sha256") != study.config_hash(config):
        raise ValueError("result configuration identity mismatch")
    identity = study._identity(manifest)
    if study._identity(result) != identity:
        raise ValueError("result source/runtime/device identity mismatch")
    count = config["data"]["n_experiences"]
    cadence = config.get("retention_eval_every_experiences", 1)
    checkpoints = list(range(cadence, count+1, cadence))
    if not checkpoints or checkpoints[-1] != count:
        checkpoints.append(count)
    metrics, _, _ = study._validate_metrics(result, config, checkpoints)
    steps = count*config["steps_per_experience"]
    if result.get("allocation_summary", {}).get("steps") != steps:
        raise ValueError("allocation summary is missing planned steps")
    allocation_path = location / "allocation.json"
    allocation, allocation_hash = study._load(allocation_path)
    if allocation_hash != result.get("allocation_trace_sha256"):
        raise ValueError("allocation trace hash mismatch")
    trace = allocation.get("trace")
    if not isinstance(trace, list) or len(trace) != steps:
        raise ValueError("allocation trace is missing planned steps")
    resets = 0
    for index, event in enumerate(trace, 1):
        if not isinstance(event, dict) or event.get("step") != index:
            raise ValueError("allocation trace has nonsequential steps")
        counts = event.get("recycled_by_population")
        if not isinstance(counts, dict) or not counts or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError("missing or invalid per-population reset counts")
        resets += sum(counts.values())
    if result.get("diagnostics", {}).get("total_recycled") != resets:
        raise ValueError("result and trace reset totals disagree")
    cost = result.get("cost")
    if not isinstance(cost, dict) or not cost:
        raise ValueError("result is missing logged resource cost counters")
    for name, value in cost.items():
        study._integer(value, f"cost.{name}", minimum=0)
    environment = result.get("environment", {})
    row = {
        "phase": phase, "cohort": cohort, "method": method, "seed": seed,
        "run_path": _portable(location, root), "result_path": _portable(path, root),
        "result_sha256": result_hash, "source_sha256": identity["source_sha256"],
        "runtime_sha256": identity["runtime_sha256"], "execution_device": identity["execution_device"],
        "config_sha256": result["config_sha256"],
        "manifest_path": _portable(directory / "manifest.json", root),
        "manifest_sha256": _sha(directory / "manifest.json"),
        "allocation_path": _portable(allocation_path, root), "allocation_trace_sha256": allocation_hash,
        "events_path": _portable(location / "events.json", root),
        "events_sha256": _sha(location / "events.json"),
        "experiences": count, "optimizer_steps": steps, "unit_resets": resets,
        "feature_gain": config["allocation_v3"]["feature_gain"],
        "metrics": {key: metrics[key] for key in ("final_accuracy", "late_early_auc")},
        "cost": cost,
        "replay_bytes": study._integer(result.get("replay_bytes"), "replay_bytes", minimum=0),
        "model_parameters": study._integer(result.get("model_parameters"), "model_parameters"),
        "wall_seconds": study._finite(result.get("wall_seconds"), "wall_seconds", minimum=0),
        "training_git_commit": environment.get("git_commit"),
        "training_git_dirty": environment.get("git_dirty"),
        "result_file_mtime_utc": _mtime(path),
    }
    if "peak_cuda_bytes" in result:
        value = result["peak_cuda_bytes"]
        row["peak_cuda_bytes"] = None if value is None else study._integer(value, "peak_cuda_bytes", minimum=0)
    if "experience_seconds" in result:
        times = result["experience_seconds"]
        if not isinstance(times, list) or len(times) != count:
            raise ValueError("logged experience timing does not cover the planned stream")
        row["summed_experience_seconds"] = sum(study._finite(t, "experience_seconds", minimum=0) for t in times)
    return row


def _lock_git_record(root: Path, selection_path: Path) -> dict | None:
    relative = _portable(selection_path, root)
    try:
        record = subprocess.check_output(
            ["git", "log", "-1", "--format=%H%n%cI", "--", relative],
            cwd=root, stderr=subprocess.DEVNULL, text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    if len(record) != 2:
        return None
    return {"commit": record[0], "committer_timestamp": record[1],
            "scope": "Most recent local Git commit touching the selection artifact; metadata, not trusted clock proof."}


def inventory(workspace: Path = ROOT, *, spec_path=Path("configs/v3_study.json"),
              locked=Path("configs/v3/locked"), development=Path("runs/v3_development"),
              evaluation=Path("runs/v3_evaluation"), engineering_roots=None) -> dict:
    if __package__:
        from .summarize_v3 import audit_suite, validate_locked_configs
    else:
        from summarize_v3 import audit_suite, validate_locked_configs

    root = Path(workspace).resolve(strict=True)
    spec_path, locked, development, evaluation = [root / path for path in (spec_path, locked, development, evaluation)]
    for path in (spec_path, locked, development, evaluation):
        _portable(path, root)
    spec, digest = study.load_spec(spec_path)
    development_configs = study.development_configs(spec, digest)
    selection, evaluation_configs = validate_locked_configs(locked)
    if selection.get("specification") != spec or selection.get("study_spec_sha256") != digest:
        raise ValueError("selection differs from the supplied study specification identity")
    # Refuse partial grids before loading any outcome-dependent cohort audit.
    _coverage(development, development_configs)
    _coverage(evaluation, evaluation_configs)
    validated_selection, selection_bytes, inputs = validate_selection(locked / "selection.json", development)
    if validated_selection != selection:
        raise ValueError("selection changed while validating its development inputs")
    common_identity = selection["common_identity"]
    runs, cohorts = [], []
    for phase, directory, configs in (
        ("development", development, development_configs),
        ("locked_evaluation", evaluation, evaluation_configs),
    ):
        for cohort, config in configs.items():
            suite = directory / cohort
            manifest, manifest_hash = study._load(suite / "manifest.json")
            if study._identity(manifest) != common_identity:
                raise ValueError("comparative source/runtime/device differs from locked development identity")
            if phase == "locked_evaluation":
                audit_suite(suite, config)
            records = [_run_record(root, suite, config, manifest, method, seed, phase, cohort)
                       for method in config["methods"] for seed in config["seeds"]]
            runs.extend(records)
            cohorts.append({
                "phase": phase, "cohort": cohort, "run_count": len(records),
                "methods": config["methods"], "seeds": config["seeds"],
                "manifest_path": _portable(suite / "manifest.json", root),
                "manifest_sha256": manifest_hash, "config_sha256": study.config_hash(config),
                "manifest_file_mtime_utc": _mtime(suite / "manifest.json"),
            })
    development_count = sum(row["phase"] == "development" for row in runs)
    evaluation_count = len(runs)-development_count
    if development_count != len(inputs):
        raise ValueError("development input count differs from the frozen selection")

    engineering = []
    for label, relative in (ENGINEERING_ROOTS if engineering_roots is None else engineering_roots).items():
        directory = root / relative
        locator = _portable(directory, root)
        if not directory.exists():
            engineering.append({"cohort": label, "path": locator, "status": "not_present", "run_count": 0})
            continue
        manifest, manifest_hash = study._load(directory / "manifest.json")
        config = manifest["config"]
        _coverage(directory, {"": config})
        if manifest.get("config_sha256") != study.config_hash(config):
            raise ValueError("engineering smoke configuration hash mismatch")
        records = [_run_record(root, directory, config, manifest, method, seed, "engineering_smoke", label)
                   for method in config["methods"] for seed in config["seeds"]]
        engineering.append({"cohort": label, "path": locator, "status": "complete", "run_count": len(records),
                            "manifest_sha256": manifest_hash, "identity": study._identity(manifest),
                            "runs": records})
    proof_path = root / "reports/v3/clean_clone_check.json"
    clean_clone = None
    if proof_path.is_file():
        proof, proof_hash = study._load(proof_path)
        commands = proof.get("commands", [])
        clean_clone = {
            "path": _portable(proof_path, root), "sha256": proof_hash,
            "status": proof.get("status"), "clone_commit": proof.get("clone_commit"),
            "recorded_command_attempts": len(commands),
            "recorded_nonzero_command_attempts": sum(command.get("returncode") != 0 for command in commands),
            "cache_note": proof.get("cache_note"), "audit_note": proof.get("audit_note"),
            "scope": "Installation/audit command attempts are engineering activity, not comparative training runs.",
        }
    selection_path = locked / "selection.json"
    return {
        "schema_version": 1, "status": "all planned comparative runs complete and validated",
        "interpretation": "Provenance and attempt coverage only; no method ranking or efficacy conclusion.",
        "paths_relative_to": "repository_root",
        "attempt_scope": "All currently present suite/run directories beneath the specified roots. Deleted or overwritten historical attempts cannot be reconstructed from these artifacts.",
        "comparative_run_count": len(runs), "development_run_count": development_count,
        "locked_evaluation_run_count": evaluation_count,
        "engineering_smoke_run_count": sum(row["run_count"] for row in engineering),
        "study_spec": {"path": _portable(spec_path, root), "sha256": digest},
        "selection": {
            "path": _portable(selection_path, root), "sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "selected_feature_gain": selection["selected_feature_gain"],
            "bound_development_result_count": len(inputs), "evaluation_outcomes_consumed": False,
            "file_mtime_utc": _mtime(selection_path), "git_record": _lock_git_record(root, selection_path),
            "dependency": "Every locked config and result references this pre-existing selection artifact hash.",
        },
        "chronology_caution": "Filesystem and Git timestamps are recorded provenance, not independent trusted clock proof. The validation preserves the development-to-selection-to-locked-result hash dependencies.",
        "timing_caution": "Elapsed times are included only where directly logged; concurrent workloads and runtime differences prevent a matched-throughput interpretation.",
        "comparative_identity": common_identity, "cohorts": cohorts, "runs": runs,
        "engineering_smokes": engineering, "clean_clone_engineering_proof": clean_clone,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--spec", type=Path, default=Path("configs/v3_study.json"))
    parser.add_argument("--locked", type=Path, default=Path("configs/v3/locked"))
    parser.add_argument("--development", type=Path, default=Path("runs/v3_development"))
    parser.add_argument("--evaluation", type=Path, default=Path("runs/v3_evaluation"))
    parser.add_argument("--output", type=Path, default=Path("reports/v3/run_inventory.json"))
    args = parser.parse_args(argv)
    try:
        value = inventory(args.workspace, spec_path=args.spec, locked=args.locked,
                          development=args.development, evaluation=args.evaluation)
        root = args.workspace.resolve()
        output = root / args.output
        _portable(output, root)
        study._write_without_conflicts({output: (json.dumps(value, indent=2, allow_nan=False)+"\n").encode("utf-8")})
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Inventory refused: {error}\n")
    print(f"Verified {value['development_run_count']} development + {value['locked_evaluation_run_count']} locked comparative runs; "
          f"{value['engineering_smoke_run_count']} engineering smoke runs listed separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
