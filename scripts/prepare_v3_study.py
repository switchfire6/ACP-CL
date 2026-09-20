"""Prepare the planned v3 development grid and lock its prespecified selection.

Only development result directories are read by ``select``. Evaluation outcomes
never enter selection. Conflicting existing artifacts are preserved and rejected.
"""

from __future__ import annotations

import argparse
import copy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_METHOD = "recycle_v3"
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class NoCandidatePassed(ValueError):
    """All complete candidates failed; a failed-screen artifact was preserved."""


def config_hash(value: dict) -> str:
    # Exactly the experiment runner's configuration/runtime fingerprint encoding.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()

    def nonfinite(value):
        raise ValueError(f"non-finite JSON number {value} in {path}")

    value = json.loads(raw.decode("utf-8-sig"), parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _integer(value, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name: str, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return float(value)


def _seed_list(value, name: str) -> list[int]:
    if not isinstance(value, list) or not value or len(value) != len(set(value)):
        raise ValueError(f"{name} must be a nonempty unique seed list")
    return [_integer(seed, name, minimum=0) for seed in value]


def candidate_stem(gain: float, regime: str) -> str:
    integer, _, fractional = format(Decimal(str(gain)), "f").partition(".")
    token = integer + fractional.rstrip("0").ljust(2, "0")
    return f"gain_{token}_{regime}"


def load_spec(path: Path) -> tuple[dict, str]:
    spec, digest = _load(path)
    development = _seed_list(spec.get("development_seeds"), "development_seeds")
    evaluation = _seed_list(spec.get("evaluation_seeds"), "evaluation_seeds")
    if set(development) & set(evaluation):
        raise ValueError("development and held-out evaluation seeds must be disjoint")
    gains = spec.get("feature_gains")
    if not isinstance(gains, list) or not gains:
        raise ValueError("feature_gains must be a nonempty list")
    for gain in gains:
        if _finite(gain, "feature_gain", 0, 1) == 0:
            raise ValueError("feature_gain must be positive")
    if len(gains) != len(set(gains)):
        raise ValueError("feature_gains must be unique")
    if _finite(spec.get("newborn_peak_factor"), "newborn_peak_factor", 1) <= 1:
        raise ValueError("newborn_peak_factor must exceed one to preserve the planned gain intervention")
    if spec.get("development_regimes") not in (["recurring", "stationary"], ["stationary", "recurring"]):
        raise ValueError("development_regimes must contain recurring and stationary exactly once")
    for key in ("development_experiences", "evaluation_experiences"):
        _integer(spec.get(key), key)
    for key in ("maximum_stationary_drawdown", "stationary_final_tolerance", "selection_score_tie_tolerance"):
        _finite(spec.get(key), key, 0, 1)
    base = spec.get("base_config")
    if not isinstance(base, dict) or not isinstance(base.get("data"), dict) or not isinstance(base.get("allocation_v3"), dict):
        raise ValueError("base_config requires data and allocation_v3 objects")
    if base["data"].get("dataset") != "procedural_shapes" or base["data"].get("color_policy") != "independent":
        raise ValueError("development requires procedural_shapes with independent colors")
    if base.get("eval_split") != "validation" or base["data"].get("test_per_class") != 0:
        raise ValueError("the study must use validation outcomes and generate no test examples")
    if base.get("scratch_reference") is not False:
        raise ValueError("the planned study does not include scratch fits")
    cadence = _integer(base.get("retention_eval_every_experiences"), "retention cadence")
    if spec["development_experiences"] % cadence or spec["development_experiences"] < 2 * cadence:
        raise ValueError("development must contain at least two evenly spaced full-retention checkpoints")
    _integer(base.get("steps_per_experience"), "steps_per_experience")
    conditions = spec.get("evaluation_conditions")
    if not isinstance(conditions, dict) or set(conditions) != {"recurring", "stationary", "early_biased"}:
        raise ValueError("three planned evaluation conditions are required")
    for name, condition in conditions.items():
        methods = condition.get("methods")
        if not isinstance(methods, list) or not methods or len(methods) != len(set(methods)) or \
                any(not isinstance(method, str) for method in methods) or "newborn_v3" not in methods:
            raise ValueError(f"invalid evaluation methods for {name}")
        expected = {"recurring": ("recurring", "independent"),
                    "stationary": ("stationary", "independent"),
                    "early_biased": ("recurring", "early_biased")}[name]
        if (condition.get("regime"), condition.get("color_policy")) != expected:
            raise ValueError(f"invalid rendering condition for {name}")
        if name == "early_biased":
            _integer(condition.get("bias_experiences"), "bias_experiences", minimum=0)
        elif "bias_experiences" in condition:
            raise ValueError("bias_experiences is only valid for early_biased evaluation")
    return spec, digest


def development_configs(spec: dict, spec_hash: str) -> dict[str, dict]:
    configs = {}
    for gain in spec["feature_gains"]:
        for regime in spec["development_regimes"]:
            config = copy.deepcopy(spec["base_config"])
            config["stage"] = f"v3 development: feature gain {gain:g}, {regime}"
            config["study_phase"] = "development"
            config["study_spec_sha256"] = spec_hash
            config["data"].update(n_experiences=spec["development_experiences"], regime=regime,
                                  color_policy="independent")
            config["data"].pop("bias_experiences", None)
            config["allocation_v3"]["feature_gain"] = gain
            config["allocation_v3"]["newborn_peak"] = spec["newborn_peak_factor"] * gain
            config["seeds"] = list(spec["development_seeds"])
            config["methods"] = [DEVELOPMENT_METHOD]
            config["primary_method"] = DEVELOPMENT_METHOD
            stem = candidate_stem(gain, regime)
            if stem in configs:
                raise ValueError("candidate filenames collide")
            configs[stem] = config
    return configs


def _write_without_conflicts(files: dict[Path, bytes]) -> None:
    # Preflight all files before the first write, so a conflict cannot leave a
    # mixture of old and newly locked configurations behind.
    for path, content in files.items():
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise FileExistsError(f"preserving conflicting existing artifact: {path}")
    for path, content in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also preserves a file created after preflight.
        with path.open("xb") as handle:
            handle.write(content)


def prepare_development(spec_path: Path, configs_directory: Path) -> list[Path]:
    spec, digest = load_spec(spec_path)
    files = {configs_directory / f"{stem}.json": _bytes(config)
             for stem, config in development_configs(spec, digest).items()}
    _write_without_conflicts(files)
    return list(files)


def stationary_drawdown(matrix: list, checkpoints: list[int]) -> tuple[float, list[dict]]:
    """Max earlier-to-later loss on column zero at full checkpoints only."""
    values = []
    for experience in checkpoints:
        row = matrix[experience - 1]
        if len(row) < experience:
            raise ValueError("accuracy matrix is missing a full-retention checkpoint")
        for value in row[:experience]:
            _finite(value, "full-retention checkpoint accuracy", 0, 1)
        values.append({"experience": experience, "accuracy": float(row[0])})
    peak, drawdown = values[0]["accuracy"], 0.0
    for observation in values[1:]:
        value = observation["accuracy"]
        drawdown = max(drawdown, peak - value)
        peak = max(peak, value)
    return drawdown, values


def _identity(payload: dict) -> dict:
    source = payload.get("source_sha256")
    if not isinstance(source, str) or not _HASH.fullmatch(source):
        raise ValueError("source identity must be a SHA-256 digest")
    runtime = payload.get("runtime_fingerprint")
    if not isinstance(runtime, dict) or any(not isinstance(runtime.get(key), str) or not runtime[key]
                                             for key in ("python", "torch", "numpy")):
        raise ValueError("runtime fingerprint is missing required version identities")
    if payload.get("runtime_sha256") != config_hash(runtime):
        raise ValueError("runtime fingerprint/hash mismatch")
    device = payload.get("execution_device")
    if not isinstance(device, str) or not (device == "cpu" or device.startswith("cuda")):
        raise ValueError("invalid execution device identity")
    if device.startswith("cuda") and any(not runtime.get(key) for key in ("cuda_runtime", "gpu")):
        raise ValueError("CUDA runtime fingerprint is incomplete")
    return {"source_sha256": source, "execution_device": device,
            "runtime_fingerprint": runtime, "runtime_sha256": payload["runtime_sha256"]}


def _validate_metrics(result: dict, config: dict, checkpoints: list[int]) -> tuple[dict, float, list[dict]]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not {"final_accuracy", "late_early_auc"} <= set(metrics):
        raise ValueError("required selection metrics are missing")
    for name, value in metrics.items():
        _finite(value, f"metric {name}")
    for name in ("final_accuracy", "late_early_auc"):
        _finite(metrics[name], name, 0, 1)
    count = config["data"]["n_experiences"]
    matrix = result.get("accuracy_matrix")
    if not isinstance(matrix, list) or len(matrix) != count or \
            any(not isinstance(row, list) or len(row) != count for row in matrix):
        raise ValueError("accuracy matrix does not contain the complete planned stream")
    for i, row in enumerate(matrix):
        _finite(row[i], "acquisition diagonal", 0, 1)
        for value in row:
            if value is not None:
                _finite(value, "observed accuracy", 0, 1)
    drawdown, checkpoint_values = stationary_drawdown(matrix, checkpoints)
    if not math.isclose(statistics.fmean(matrix[-1]), metrics["final_accuracy"], abs_tol=1e-10):
        raise ValueError("final accuracy metric does not match the complete final matrix row")
    auc = result.get("early_auc")
    if not isinstance(auc, list) or len(auc) != count:
        raise ValueError("acquisition trajectory does not contain every planned experience")
    for value in auc:
        _finite(value, "acquisition AUC", 0, 1)
    if not math.isclose(statistics.fmean(auc[count // 2:]), metrics["late_early_auc"], abs_tol=1e-10):
        raise ValueError("late acquisition metric does not match its trajectory")
    return metrics, drawdown, checkpoint_values


def _read_development(spec: dict, digest: str, directory: Path) -> tuple[list[dict], dict]:
    configs = development_configs(spec, digest)
    cadence = spec["base_config"]["retention_eval_every_experiences"]
    checkpoints = list(range(cadence, spec["development_experiences"] + 1, cadence))
    expected_seeds, heldout = set(spec["development_seeds"]), set(spec["evaluation_seeds"])
    records, common_identity = [], None
    for stem, config in configs.items():
        suite = directory / stem
        manifest_path = suite / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing development manifest: {stem}")
        manifest, manifest_hash = _load(manifest_path)
        if manifest.get("config") != config or manifest.get("config_sha256") != config_hash(config):
            raise ValueError(f"development manifest does not match the exact expected config: {stem}")
        if manifest.get("seeds") != spec["development_seeds"] or manifest.get("methods") != [DEVELOPMENT_METHOD]:
            raise ValueError(f"unexpected development manifest seeds/method: {stem}")
        identity = _identity(manifest)
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError("development source/runtime/device identities differ")
        seen = set()
        result_files = sorted(suite.glob("*/result.json"))
        if not result_files:
            raise ValueError(f"missing completed development results: {stem}")
        for result_path in result_files:
            result, result_hash = _load(result_path)
            # Refuse held-out identities before inspecting any outcome fields.
            seed = result.get("seed")
            if seed in heldout:
                raise ValueError("held-out evaluation seeds are forbidden in development selection")
            if type(seed) is not int or seed not in expected_seeds:
                raise ValueError(f"unexpected development seed: {seed}")
            if result.get("method") != DEVELOPMENT_METHOD:
                raise ValueError("unexpected development method")
            if seed in seen:
                raise ValueError(f"duplicate development result for {stem}, seed {seed}")
            seen.add(seed)
            if result.get("config") != config or result.get("config_sha256") != config_hash(config):
                raise ValueError(f"result does not match the exact expected config: {result_path}")
            if _identity(result) != common_identity:
                raise ValueError("development result source/runtime/device identity mismatch")
            if result.get("eval_split") != "validation" or result.get("class_order") != [0, 1, 2, 3]:
                raise ValueError("development result has the wrong split or output-label identity")
            environment = result.get("environment", {})
            for key in ("python", "torch", "numpy"):
                if key in environment and environment[key] != common_identity["runtime_fingerprint"][key]:
                    raise ValueError("result environment disagrees with its runtime identity")
            if "source_sha256" in environment and environment["source_sha256"] != common_identity["source_sha256"]:
                raise ValueError("result environment disagrees with its source identity")
            metrics, drawdown, series = _validate_metrics(result, config, checkpoints)
            allocation_path = result_path.parent / "allocation.json"
            if not allocation_path.is_file():
                raise ValueError("completed development result is missing its allocation trace")
            allocation, allocation_hash = _load(allocation_path)
            if result.get("allocation_trace_sha256") != allocation_hash:
                raise ValueError("allocation trace hash mismatch")
            expected_steps = spec["development_experiences"] * config["steps_per_experience"]
            trace = allocation.get("trace")
            if not isinstance(trace, list) or len(trace) != expected_steps or \
                    any(not isinstance(row, dict) or row.get("step") != index + 1 for index, row in enumerate(trace)):
                raise ValueError("allocation trace does not cover every planned optimizer step")
            if result.get("allocation_summary", {}).get("steps") != expected_steps:
                raise ValueError("allocation summary does not contain all planned optimizer steps")
            records.append({
                "candidate": stem, "feature_gain": config["allocation_v3"]["feature_gain"],
                "regime": config["data"]["regime"], "seed": seed, "method": DEVELOPMENT_METHOD,
                "config_sha256": result["config_sha256"], **common_identity,
                "manifest_path": manifest_path.relative_to(directory).as_posix(), "manifest_sha256": manifest_hash,
                "result_path": result_path.relative_to(directory).as_posix(), "result_sha256": result_hash,
                "allocation_path": allocation_path.relative_to(directory).as_posix(), "allocation_sha256": allocation_hash,
                "metrics": metrics,
                "stationary_drawdown": drawdown if config["data"]["regime"] == "stationary" else None,
                "stationary_checkpoint_accuracies": series if config["data"]["regime"] == "stationary" else None,
            })
        if seen != expected_seeds:
            raise ValueError(f"missing development seed results for {stem}: {sorted(expected_seeds - seen)}")
    return records, common_identity


def _choose(spec: dict, records: list[dict]) -> tuple[list[dict], float | None]:
    candidates = []
    for gain in spec["feature_gains"]:
        stationary = [run for run in records if run["feature_gain"] == gain and run["regime"] == "stationary"]
        recurring = [run for run in records if run["feature_gain"] == gain and run["regime"] == "recurring"]
        stationary_final = statistics.fmean(run["metrics"]["final_accuracy"] for run in stationary)
        recurring_final = statistics.fmean(run["metrics"]["final_accuracy"] for run in recurring)
        recurring_auc = statistics.fmean(run["metrics"]["late_early_auc"] for run in recurring)
        candidates.append({
            "feature_gain": gain,
            "stationary_drawdowns": {str(run["seed"]): run["stationary_drawdown"] for run in stationary},
            "stationary_checkpoint_accuracies": {str(run["seed"]): run["stationary_checkpoint_accuracies"] for run in stationary},
            "mean_stationary_final_accuracy": stationary_final,
            "mean_recurring_final_accuracy": recurring_final,
            "mean_recurring_late_auc": recurring_auc,
            "selection_score": 0.5 * (recurring_final + recurring_auc),
            "passes_stationary_screen": all(run["stationary_drawdown"] <= spec["maximum_stationary_drawdown"] + 1e-12
                                            for run in stationary),
            "within_stationary_final_tolerance": False, "in_score_tie_band": False, "selected": False,
            "input_result_paths": [run["result_path"] for run in stationary + recurring],
        })
    passing = [candidate for candidate in candidates if candidate["passes_stationary_screen"]]
    if not passing:
        return candidates, None
    best_stationary = max(candidate["mean_stationary_final_accuracy"] for candidate in passing)
    near_best = []
    for candidate in passing:
        candidate["within_stationary_final_tolerance"] = candidate["mean_stationary_final_accuracy"] >= \
            best_stationary - spec["stationary_final_tolerance"] - 1e-12
        if candidate["within_stationary_final_tolerance"]:
            near_best.append(candidate)
    best_score = max(candidate["selection_score"] for candidate in near_best)
    for candidate in near_best:
        candidate["in_score_tie_band"] = candidate["selection_score"] >= best_score - spec["selection_score_tie_tolerance"] - 1e-12
    winner = min((candidate for candidate in near_best if candidate["in_score_tie_band"]),
                 key=lambda candidate: candidate["feature_gain"])
    winner["selected"] = True
    return candidates, winner["feature_gain"]


def select(spec_path: Path, results_directory: Path, output: Path) -> dict:
    spec, digest = load_spec(spec_path)
    records, identity = _read_development(spec, digest, results_directory)
    candidates, selected = _choose(spec, records)
    cadence = spec["base_config"]["retention_eval_every_experiences"]
    artifact = {
        "schema_version": 1, "status": "selected" if selected is not None else "no_candidate_passed",
        "study_spec_sha256": digest, "specification": spec,
        "development_seeds": spec["development_seeds"], "evaluation_seeds": spec["evaluation_seeds"],
        "selection_rule": {
            "stationary_checkpoint_experiences": list(range(cadence, spec["development_experiences"] + 1, cadence)),
            "stationary_pool": "first-experience validation column at full-retention checkpoints only; acquisition diagonal at experience 1 excluded",
            "maximum_stationary_drawdown": spec["maximum_stationary_drawdown"],
            "stationary_screen": "every development seed must pass independently",
            "stationary_final_tolerance": spec["stationary_final_tolerance"],
            "score": "0.5 * (mean recurring final accuracy + mean recurring late acquisition AUC)",
            "score_tie_tolerance": spec["selection_score_tie_tolerance"],
            "tie_break": "smallest feature gain within tolerance of the maximum eligible score",
        },
        "common_identity": identity,
        "input_paths_relative_to": "the development results root passed to select",
        "input_results": records, "candidates": candidates, "selected_feature_gain": selected,
        "evaluation_outcomes_consumed": False,
    }
    selection_bytes = _bytes(artifact)
    files = {output / "selection.json": selection_bytes}
    if selected is None:
        _write_without_conflicts(files)
        raise NoCandidatePassed("no candidate passed the stationary screen; selection.json records all failures and no locked configs were emitted")
    selection_hash = hashlib.sha256(selection_bytes).hexdigest()
    for condition_name, condition in spec["evaluation_conditions"].items():
        config = copy.deepcopy(spec["base_config"])
        config["stage"] = f"v3 locked evaluation: {condition_name}"
        config["study_phase"] = "locked_evaluation"
        config["study_spec_sha256"] = digest
        config["data"].update(n_experiences=spec["evaluation_experiences"],
                              regime=condition["regime"], color_policy=condition["color_policy"])
        config["data"].pop("bias_experiences", None)
        if "bias_experiences" in condition:
            config["data"]["bias_experiences"] = condition["bias_experiences"]
        config["allocation_v3"]["feature_gain"] = selected
        config["allocation_v3"]["newborn_peak"] = spec["newborn_peak_factor"] * selected
        config["methods"] = list(condition["methods"])
        config["seeds"] = list(spec["evaluation_seeds"])
        config["primary_method"] = "newborn_v3"
        config["selection_provenance"] = {"artifact": "selection.json", "sha256": selection_hash,
                                           "study_spec_sha256": digest}
        files[output / f"{condition_name}.json"] = _bytes(config)
    _write_without_conflicts(files)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-development")
    prepare.add_argument("--spec", type=Path, default=ROOT / "configs" / "v3_study.json")
    prepare.add_argument("--configs", type=Path, default=ROOT / "configs" / "v3" / "development")
    lock = commands.add_parser("select")
    lock.add_argument("--spec", type=Path, default=ROOT / "configs" / "v3_study.json")
    lock.add_argument("--results", type=Path, default=ROOT / "runs" / "v3_development")
    lock.add_argument("--output", type=Path, default=ROOT / "configs" / "v3" / "locked")
    args = parser.parse_args()
    try:
        if args.command == "prepare-development":
            for path in prepare_development(args.spec, args.configs):
                print(path)
        else:
            artifact = select(args.spec, args.results, args.output)
            print(f"Selected feature gain {artifact['selected_feature_gain']:g}; locked artifacts: {args.output}")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
