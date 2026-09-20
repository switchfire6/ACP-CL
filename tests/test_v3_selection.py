"""Prespecified development screening and immutable evaluation locking."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_v3_study as study


ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def spec_path(tmp_path):
    path = tmp_path / "v3_study.json"
    path.write_bytes((ROOT / "configs" / "v3_study.json").read_bytes())
    return path


def _create_results(spec_path, directory, profiles=None):
    spec, digest = study.load_spec(spec_path)
    defaults = {
        0.05: {"stationary_final": 0.78, "recurring_final": 0.80, "recurring_auc": 0.60},
        0.15: {"stationary_final": 0.80, "recurring_final": 0.80, "recurring_auc": 0.80},
        0.5: {"stationary_final": 0.795, "recurring_final": 0.84, "recurring_auc": 0.78},
        1.0: {"stationary_final": 0.77, "recurring_final": 0.99, "recurring_auc": 0.99},
    }
    for gain, overrides in (profiles or {}).items():
        defaults[gain].update(overrides)
    runtime = {"python": "3.11.0", "torch": "2.8.0+cpu", "numpy": "2.3.3"}
    identity = {"source_sha256": "a" * 64, "execution_device": "cpu",
                "runtime_fingerprint": runtime, "runtime_sha256": study.config_hash(runtime)}
    configs = study.development_configs(spec, digest)
    for stem, config in configs.items():
        suite = directory / stem
        config_identity = {"config": config, "config_sha256": study.config_hash(config), **identity}
        _write(suite / "manifest.json", {**config_identity, "seeds": spec["development_seeds"],
                                         "methods": [study.DEVELOPMENT_METHOD]})
        gain = config["allocation_v3"]["feature_gain"]
        profile = defaults[gain]
        count = spec["development_experiences"]
        cadence = config["retention_eval_every_experiences"]
        for seed in spec["development_seeds"]:
            stationary = config["data"]["regime"] == "stationary"
            final = profile["stationary_final"] if stationary else profile["recurring_final"]
            auc = 0.6 if stationary else profile["recurring_auc"]
            series = profile.get("per_seed_series", {}).get(seed, profile.get("series", [0.8] * (count // cadence)))
            matrix = [[None] * count for _ in range(count)]
            for i in range(count):
                matrix[i][i] = 0.6
            matrix[0][0] = 0.99  # Intentionally excluded from the drawdown screen.
            for index, experience in enumerate(range(cadence, count + 1, cadence)):
                matrix[experience - 1][:experience] = [0.7] * experience
                matrix[experience - 1][0] = series[index]
            # Keep final reported performance consistent with the complete row,
            # while allowing a distinct first-pool accuracy used by the screen.
            matrix[-1][1:] = [(final * count - matrix[-1][0]) / (count - 1)] * (count - 1)
            steps = count * config["steps_per_experience"]
            run = suite / f"recycle_v3_seed{seed}"
            allocation_path = run / "allocation.json"
            _write(allocation_path, {"trace": [{"step": step} for step in range(1, steps + 1)]})
            result = {
                **config_identity, "seed": seed, "method": study.DEVELOPMENT_METHOD,
                "eval_split": "validation", "class_order": [0, 1, 2, 3],
                "metrics": {"final_accuracy": final, "late_early_auc": auc},
                "accuracy_matrix": matrix, "early_auc": [auc] * count,
                "allocation_summary": {"steps": steps},
                "allocation_trace_sha256": hashlib.sha256(allocation_path.read_bytes()).hexdigest(),
                "environment": {**runtime, "source_sha256": identity["source_sha256"]},
            }
            _write(run / "result.json", result)
    return spec, configs


def _first_result(directory):
    return sorted(directory.glob("*/*/result.json"))[0]


def test_preparation_is_exact_idempotent_and_scales_newborn_peak(spec_path, tmp_path):
    directory = tmp_path / "configs"
    paths = study.prepare_development(spec_path, directory)
    assert len(paths) == 8
    assert {path.stem for path in paths} == {
        f"gain_{gain}_{regime}" for gain in ("005", "015", "050", "100")
        for regime in ("recurring", "stationary")
    }
    before = {path: path.read_bytes() for path in paths}
    assert study.prepare_development(spec_path, directory) == paths
    assert before == {path: path.read_bytes() for path in paths}
    for path in paths:
        config = json.loads(path.read_text())
        assert config["seeds"] == [411, 522]
        assert config["methods"] == ["recycle_v3"]
        assert config["data"]["color_policy"] == "independent"
        assert config["data"]["n_experiences"] == 60
        assert config["allocation_v3"]["newborn_peak"] == 4 * config["allocation_v3"]["feature_gain"]


def test_preparation_preserves_conflicting_files_before_writing_any(spec_path, tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    conflict = directory / "gain_015_stationary.json"
    conflict.write_text("user content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="preserving"):
        study.prepare_development(spec_path, directory)
    assert list(directory.iterdir()) == [conflict]
    assert conflict.read_text() == "user content"


def test_selection_applies_stationary_near_best_then_recurring_score_and_locks(spec_path, tmp_path):
    results, output = tmp_path / "results", tmp_path / "locked"
    spec, _ = _create_results(spec_path, results)
    artifact = study.select(spec_path, results, output)
    assert artifact["selected_feature_gain"] == 0.5
    assert artifact["status"] == "selected"
    assert artifact["evaluation_outcomes_consumed"] is False
    assert len(artifact["input_results"]) == 16
    assert artifact["selection_rule"]["stationary_checkpoint_experiences"] == [10, 20, 30, 40, 50, 60]
    candidates = {candidate["feature_gain"]: candidate for candidate in artifact["candidates"]}
    assert candidates[1.0]["selection_score"] > candidates[0.5]["selection_score"]
    assert candidates[1.0]["within_stationary_final_tolerance"] is False
    assert all(candidate["passes_stationary_screen"] for candidate in candidates.values())
    selection_hash = hashlib.sha256((output / "selection.json").read_bytes()).hexdigest()
    assert {path.name for path in output.iterdir()} == {"selection.json", "recurring.json", "stationary.json", "early_biased.json"}
    for condition in spec["evaluation_conditions"]:
        config = json.loads((output / f"{condition}.json").read_text())
        assert config["allocation_v3"]["feature_gain"] == 0.5
        assert config["allocation_v3"]["newborn_peak"] == 2
        assert config["primary_method"] == "newborn_v3"
        assert config["seeds"] == [731, 842, 953]
        assert config["data"]["n_experiences"] == 100
        assert config["methods"] == spec["evaluation_conditions"][condition]["methods"]
        assert config["selection_provenance"]["sha256"] == selection_hash
        assert config["selection_provenance"]["artifact"] == "selection.json"
    assert json.loads((output / "early_biased.json").read_text())["data"]["bias_experiences"] == 8
    for record in artifact["input_results"]:
        for name in ("result_path", "manifest_path", "allocation_path"):
            assert not Path(record[name]).is_absolute() and "\\" not in record[name]
        assert hashlib.sha256((results / record["result_path"]).read_bytes()).hexdigest() == record["result_sha256"]
    before = {path: path.read_bytes() for path in output.iterdir()}
    study.select(spec_path, results, output)
    assert before == {path: path.read_bytes() for path in output.iterdir()}


def test_score_tolerance_tie_selects_smaller_gain(spec_path, tmp_path):
    results = tmp_path / "results"
    _create_results(spec_path, results, {0.15: {"recurring_final": 0.814}})
    artifact = study.select(spec_path, results, tmp_path / "locked")
    assert artifact["selected_feature_gain"] == 0.15
    tied = {candidate["feature_gain"] for candidate in artifact["candidates"] if candidate["in_score_tie_band"]}
    assert tied == {0.15, 0.5}


def test_stationary_screen_requires_each_seed_to_pass(spec_path, tmp_path):
    results = tmp_path / "results"
    _create_results(spec_path, results, {0.5: {"per_seed_series": {522: [0.9, 0.89, 0.88, 0.87, 0.86, 0.79]}}})
    artifact = study.select(spec_path, results, tmp_path / "locked")
    assert artifact["selected_feature_gain"] == 0.15
    rejected = next(candidate for candidate in artifact["candidates"] if candidate["feature_gain"] == 0.5)
    assert rejected["stationary_drawdowns"]["411"] == 0
    assert rejected["stationary_drawdowns"]["522"] == pytest.approx(0.11)
    assert not rejected["passes_stationary_screen"]


def test_drawdown_uses_identical_first_pool_and_excludes_initial_diagonal():
    matrix = [[None] * 60 for _ in range(60)]
    matrix[0][0] = 1.0
    values = [0.8, 0.78, 0.82, 0.75, 0.80, 0.81]
    checkpoints = [10, 20, 30, 40, 50, 60]
    for i, experience in enumerate(checkpoints):
        matrix[experience - 1][:experience] = [0.1 if i % 2 else 0.95] * experience
        matrix[experience - 1][0] = values[i]
    drawdown, series = study.stationary_drawdown(matrix, checkpoints)
    assert drawdown == pytest.approx(0.07)
    assert [point["accuracy"] for point in series] == values


def test_no_candidate_pass_writes_failure_audit_without_locked_configs(spec_path, tmp_path):
    results, output = tmp_path / "results", tmp_path / "locked"
    bad = {gain: {"series": [0.9, 0.85, 0.8, 0.75, 0.7, 0.6]} for gain in (0.05, 0.15, 0.5, 1.0)}
    _create_results(spec_path, results, bad)
    with pytest.raises(study.NoCandidatePassed, match="no candidate"):
        study.select(spec_path, results, output)
    assert {path.name for path in output.iterdir()} == {"selection.json"}
    artifact = json.loads((output / "selection.json").read_text())
    assert artifact["selected_feature_gain"] is None
    assert artifact["status"] == "no_candidate_passed"
    assert len(artifact["candidates"]) == 4 and len(artifact["input_results"]) == 16
    assert not any(candidate["passes_stationary_screen"] for candidate in artifact["candidates"])


def test_selection_preserves_conflicting_locked_file_before_writing_artifact(spec_path, tmp_path):
    results, output = tmp_path / "results", tmp_path / "locked"
    _create_results(spec_path, results)
    output.mkdir()
    conflict = output / "stationary.json"
    conflict.write_text("previous locked protocol")
    with pytest.raises(FileExistsError, match="preserving"):
        study.select(spec_path, results, output)
    assert list(output.iterdir()) == [conflict]
    assert conflict.read_text() == "previous locked protocol"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "method", "config", "source", "runtime", "allocation_hash", "nonfinite", "incomplete_matrix"])
def test_missing_duplicate_or_mismatched_inputs_cannot_select(spec_path, tmp_path, mutation):
    results, output = tmp_path / "results", tmp_path / "locked"
    _create_results(spec_path, results)
    path = _first_result(results)
    result = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    elif mutation == "duplicate":
        _write(path.parent.parent / "duplicate" / "result.json", result)
    elif mutation == "allocation_hash":
        allocation = path.parent / "allocation.json"
        allocation.write_bytes(allocation.read_bytes() + b" ")
    else:
        if mutation == "method":
            result["method"] = "newborn_v3"
        elif mutation == "config":
            result["config"]["allocation_v3"]["feature_gain"] = 0.123
            result["config_sha256"] = study.config_hash(result["config"])
        elif mutation == "source":
            result["source_sha256"] = "b" * 64
        elif mutation == "runtime":
            result["runtime_fingerprint"]["torch"] = "different"
            result["runtime_sha256"] = study.config_hash(result["runtime_fingerprint"])
        elif mutation == "nonfinite":
            result["metrics"]["final_accuracy"] = float("nan")
        else:
            result["accuracy_matrix"] = result["accuracy_matrix"][:-1]
        _write(path, result)
    with pytest.raises(ValueError):
        study.select(spec_path, results, output)
    assert not output.exists()


def test_heldout_seed_is_rejected_before_any_outcome_is_examined(spec_path, tmp_path):
    results, output = tmp_path / "results", tmp_path / "locked"
    _create_results(spec_path, results)
    path = _first_result(results)
    result = json.loads(path.read_text())
    result["seed"] = 731
    result["metrics"] = {"invalid": "this evaluation outcome must not be inspected"}
    _write(path, result)
    with pytest.raises(ValueError, match="held-out evaluation seeds"):
        study.select(spec_path, results, output)
    assert not output.exists()


def test_spec_cannot_reuse_evaluation_seeds_for_development(spec_path):
    spec = json.loads(spec_path.read_text())
    spec["development_seeds"][0] = spec["evaluation_seeds"][0]
    _write(spec_path, spec)
    with pytest.raises(ValueError, match="disjoint"):
        study.load_spec(spec_path)


def test_gain_one_preserves_newborn_treatment_in_locked_config(spec_path, tmp_path):
    results = tmp_path / "results"
    _create_results(spec_path, results, {1.0: {"stationary_final": 0.80}})
    artifact = study.select(spec_path, results, tmp_path / "locked")
    assert artifact["selected_feature_gain"] == 1.0
    config = json.loads((tmp_path / "locked" / "recurring.json").read_text())
    assert config["allocation_v3"]["newborn_peak"] == 4.0
