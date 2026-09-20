"""Complete-grid inventory provenance; the shared cohort auditor has its own tests."""

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import inventory_v3 as inventory
from scripts import prepare_v3_study as study
from scripts import summarize_v3
from test_v3_selection import _create_results


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def read(path):
    return json.loads(path.read_bytes())


def make_suite(directory, config, identity):
    common = {"config": config, "config_sha256": study.config_hash(config), **identity}
    write(directory / "manifest.json", {**common, "methods": config["methods"], "seeds": config["seeds"]})
    count = config["data"]["n_experiences"]
    for method in config["methods"]:
        for seed in config["seeds"]:
            run = directory / f"{method}_seed{seed}"
            trace = [{"step": step, "recycled_by_population": {"adapter": 0}}
                     for step in range(1, count*config["steps_per_experience"]+1)]
            write(run / "allocation.json", {"trace": trace})
            write(run / "events.json", {"events": []})
            write(run / "result.json", {
                **common, "method": method, "seed": seed,
                "metrics": {"final_accuracy": 0.8, "late_early_auc": 0.6, "forgetting": 0.0},
                "accuracy_matrix": [[0.8]*count for _ in range(count)], "early_auc": [0.6]*count,
                "allocation_summary": {"steps": len(trace)}, "diagnostics": {"total_recycled": 0},
                "allocation_trace_sha256": hashlib.sha256((run / "allocation.json").read_bytes()).hexdigest(),
                "wall_seconds": 2.5, "experience_seconds": [0.5]*count,
                "cost": {"current_examples": 128, "replay_examples": 112},
                "replay_bytes": 1024, "model_parameters": 512, "peak_cuda_bytes": None,
                "environment": {**identity["runtime_fingerprint"], "source_sha256": identity["source_sha256"],
                                "git_commit": "c"*40, "git_dirty": False},
            })


@pytest.fixture
def complete(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    spec_path = root / "configs/v3_study.json"
    spec = read(ROOT / "configs/v3_study.json")
    spec["development_experiences"] = spec["evaluation_experiences"] = 4
    spec["base_config"].update(retention_eval_every_experiences=2, steps_per_experience=2)
    spec["base_config"]["data"]["n_experiences"] = 4
    write(spec_path, spec)
    development = root / "runs/v3_development"
    _create_results(spec_path, development, {1.0: {"recurring_final": 0.9}})
    for result_path in development.glob("*/*/result.json"):
        result = read(result_path)
        result["metrics"]["forgetting"] = 0.0
        result.update(wall_seconds=1.0, experience_seconds=[0.2]*4, diagnostics={"total_recycled": 0})
        result.update(cost={"current_examples": 128, "replay_examples": 112},
                      replay_bytes=1024, model_parameters=512)
        allocation_path = result_path.parent / "allocation.json"
        trace = read(allocation_path)
        for row in trace["trace"]:
            row["recycled_by_population"] = {"adapter": 0}
        write(allocation_path, trace)
        result["allocation_trace_sha256"] = hashlib.sha256(allocation_path.read_bytes()).hexdigest()
        write(result_path, result)
        write(result_path.parent / "events.json", {"events": []})
    selection = study.select(spec_path, development, root / "configs/v3/locked")
    for condition in spec["evaluation_conditions"]:
        config = read(root / "configs/v3/locked" / f"{condition}.json")
        make_suite(root / "runs/v3_evaluation" / condition, config, selection["common_identity"])
    # Inventory invokes the real selection/spec/identity/metric validators. Its
    # delegation to the separately tested checkpoint/pairing auditor is recorded
    # here without manufacturing trusted PyTorch model checkpoints in this test.
    calls = []
    monkeypatch.setattr(summarize_v3, "audit_suite", lambda directory, config: calls.append(directory.name))
    return root, calls


def first_evaluation(root):
    return root / "runs/v3_evaluation/recurring/er_v3_seed731/result.json"


def test_complete_inventory_preserves_every_planned_run_and_bound_selection(complete):
    root, calls = complete
    originals = {path: path.read_bytes() for path in root.rglob("*.json")}
    result = inventory.inventory(root, engineering_roots={})
    assert result["development_run_count"] == 16
    assert result["locked_evaluation_run_count"] == 51
    assert result["comparative_run_count"] == 67
    assert result["engineering_smoke_run_count"] == 0
    assert calls == ["recurring", "stationary", "early_biased"]
    assert len({(row["phase"], row["cohort"], row["method"], row["seed"]) for row in result["runs"]}) == 67
    assert len(result["cohorts"]) == 11
    assert result["selection"]["bound_development_result_count"] == 16
    assert result["selection"]["evaluation_outcomes_consumed"] is False
    assert "not independent trusted clock proof" in result["chronology_caution"]
    assert result["selection"]["git_record"] is None
    for row in result["runs"]:
        assert not Path(row["run_path"]).is_absolute() and "\\" not in row["run_path"]
        assert row["optimizer_steps"] == 8
        assert row["unit_resets"] == 0
        assert row["result_sha256"] == hashlib.sha256((root / row["result_path"]).read_bytes()).hexdigest()
        assert row["allocation_trace_sha256"] == hashlib.sha256((root / row["allocation_path"]).read_bytes()).hexdigest()
        assert row["wall_seconds"] > 0 and row["summed_experience_seconds"] > 0
        assert row["cost"] == {"current_examples": 128, "replay_examples": 112}
        assert row["replay_bytes"] == 1024 and row["model_parameters"] == 512
        assert set(row["metrics"]) == {"final_accuracy", "late_early_auc"}
    assert all(path.read_bytes() == payload for path, payload in originals.items())
    assert not (root / "reports/v3/run_inventory.json").exists()


def test_engineering_counts_and_command_failures_stay_separate(complete):
    root, _ = complete
    config = copy.deepcopy(read(root / "configs/v3/locked/recurring.json"))
    config["seeds"] = [919]
    identity = read(root / "configs/v3/locked/selection.json")["common_identity"]
    identity["source_sha256"] = "b"*64  # Historical smoke need not share study source.
    make_suite(root / "runs/cpu_smoke", config, identity)
    write(root / "reports/v3/clean_clone_check.json", {
        "status": "passed", "commands": [{"returncode": 1}, {"returncode": 0}],
        "cache_note": "An installation attempt failed before training.",
        "audit_note": "An evaluator assertion was corrected without changing results.",
    })
    result = inventory.inventory(root, engineering_roots={"cpu_smoke": "runs/cpu_smoke"})
    assert result["comparative_run_count"] == 67
    assert result["engineering_smoke_run_count"] == 7
    assert result["clean_clone_engineering_proof"]["recorded_nonzero_command_attempts"] == 1
    assert result["engineering_smokes"][0]["identity"]["source_sha256"] != result["comparative_identity"]["source_sha256"]
    assert all(row["phase"] == "engineering_smoke" for row in result["engineering_smokes"][0]["runs"])
    assert all(row["peak_cuda_bytes"] is None for row in result["engineering_smokes"][0]["runs"])


def test_cpu_record_preserves_null_gpu_memory_instead_of_inventing_zero(complete):
    root, _ = complete
    suite = root / "runs/v3_evaluation/recurring"
    manifest = read(suite / "manifest.json")
    record = inventory._run_record(root, suite, manifest["config"], manifest,
                                   "er_v3", 731, "engineering_smoke", "cpu_fixture")
    assert record["execution_device"] == "cpu"
    assert "peak_cuda_bytes" in record and record["peak_cuda_bytes"] is None


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unexpected_suite", "empty_attempt"])
def test_partial_or_extra_grid_refused_before_any_cohort_audit(complete, problem):
    root, calls = complete
    path = first_evaluation(root)
    if problem == "missing":
        path.unlink()
    elif problem == "duplicate":
        target = path.parent.parent / "duplicate" / "result.json"
        target.parent.mkdir()
        shutil.copyfile(path, target)
    elif problem == "unexpected_suite":
        target = root / "runs/v3_evaluation/unplanned/result.json"
        target.parent.mkdir()
        shutil.copyfile(path, target)
    else:
        (path.parent.parent / "incomplete_attempt").mkdir()
    with pytest.raises(ValueError, match="missing|unexpected|incomplete"):
        inventory.inventory(root, engineering_roots={})
    assert calls == []
    assert not (root / "reports/v3/run_inventory.json").exists()


@pytest.mark.parametrize("problem", ["missing_metric", "nonfinite_metric", "wrong_source", "wrong_runtime",
                                     "wrong_config", "wrong_seed", "trace_bytes", "reset_total"])
def test_inconsistent_result_provenance_or_outcomes_fail(complete, problem):
    root, _ = complete
    path = first_evaluation(root)
    result = read(path)
    if problem == "missing_metric":
        del result["metrics"]["late_early_auc"]
    elif problem == "nonfinite_metric":
        result["metrics"]["final_accuracy"] = float("nan")
    elif problem == "wrong_source":
        result["source_sha256"] = "b"*64
    elif problem == "wrong_runtime":
        result["runtime_sha256"] = "b"*64
    elif problem == "wrong_config":
        result["config_sha256"] = "b"*64
    elif problem == "wrong_seed":
        result["seed"] = 842
    elif problem == "trace_bytes":
        allocation = path.parent / "allocation.json"
        allocation.write_bytes(allocation.read_bytes() + b" ")
    else:
        result["diagnostics"]["total_recycled"] = 1
    write(path, result)
    with pytest.raises(ValueError):
        inventory.inventory(root, engineering_roots={})


def test_selection_inputs_are_verified_without_replacing_the_recorded_selection(complete):
    root, _ = complete
    selection_path = root / "configs/v3/locked/selection.json"
    original = selection_path.read_bytes()
    path = next((root / "runs/v3_development").glob("*/*/result.json"))
    result = read(path)
    result["wall_seconds"] += 1
    write(path, result)
    with pytest.raises(ValueError, match="selection input result_sha256 mismatch"):
        inventory.inventory(root, engineering_roots={})
    assert selection_path.read_bytes() == original


def test_cli_writes_only_a_complete_inventory_and_preserves_conflicts(complete, capsys):
    root, _ = complete
    assert inventory.main(["--workspace", str(root)]) == 0
    path = root / "reports/v3/run_inventory.json"
    original = path.read_bytes()
    assert "16 development + 51 locked" in capsys.readouterr().out
    assert inventory.main(["--workspace", str(root)]) == 0
    assert path.read_bytes() == original
    path.write_bytes(b"existing owner content")
    with pytest.raises(SystemExit) as failure:
        inventory.main(["--workspace", str(root)])
    assert failure.value.code == 2
    assert path.read_bytes() == b"existing owner content"


def test_cli_partial_grid_never_creates_report(complete):
    root, _ = complete
    first_evaluation(root).unlink()
    with pytest.raises(SystemExit) as failure:
        inventory.main(["--workspace", str(root)])
    assert failure.value.code == 2
    assert not (root / "reports/v3/run_inventory.json").exists()
