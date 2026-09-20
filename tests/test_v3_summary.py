"""Adversarial checks for locked-cohort reporting; no scientific runs are read."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import prepare_v3_study as study
from scripts import summarize_v3 as report


ROOT = Path(__file__).resolve().parents[1]


def write(path, value, *, allow_nan=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=allow_nan) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def cohort(tmp_path):
    spec = read(ROOT / "configs/v3_study.json")
    spec.update(development_experiences=4, evaluation_experiences=5)
    base = spec["base_config"]
    base.update(steps_per_experience=2, batch_size=2, eval_every_steps=1,
                early_examples=2, retention_eval_every_experiences=1)
    base["data"]["n_experiences"] = 4
    base["allocation_v3"].update(warmup_steps=2, reset_interval=2, reset_count=1)
    runtime = {"python": "3.10.fixture", "torch": "2.8.fixture", "numpy": "2.fixture"}
    identity = {"source_sha256": "a" * 64, "execution_device": "cpu",
                "runtime_fingerprint": runtime, "runtime_sha256": study.config_hash(runtime)}
    selection = {"status": "selected", "evaluation_outcomes_consumed": False,
                 "specification": spec, "study_spec_sha256": study.config_hash(spec),
                 "development_seeds": spec["development_seeds"],
                 "evaluation_seeds": spec["evaluation_seeds"],
                 "selected_feature_gain": 0.5, "common_identity": identity}
    locked, results, output = tmp_path / "locked", tmp_path / "results", tmp_path / "report"
    write(locked / "selection.json", selection)
    selection_hash = hashlib.sha256((locked / "selection.json").read_bytes()).hexdigest()
    configs = {}
    for condition, planned in spec["evaluation_conditions"].items():
        config = copy.deepcopy(base)
        config.update(stage=f"v3 locked evaluation: {condition}", study_phase="locked_evaluation",
                      study_spec_sha256=selection["study_spec_sha256"], methods=planned["methods"],
                      seeds=spec["evaluation_seeds"], primary_method="newborn_v3")
        config["data"].update(n_experiences=5, regime=planned["regime"], color_policy=planned["color_policy"])
        if "bias_experiences" in planned:
            config["data"]["bias_experiences"] = planned["bias_experiences"]
        config["allocation_v3"].update(feature_gain=0.5, newborn_peak=2.0)
        config["selection_provenance"] = {"artifact": "selection.json", "sha256": selection_hash,
                                           "study_spec_sha256": selection["study_spec_sha256"]}
        configs[condition] = config
        write(locked / f"{condition}.json", config)
        suite = results / condition
        manifest = {**identity, "config": config, "config_sha256": study.config_hash(config),
                    "methods": config["methods"], "seeds": config["seeds"]}
        write(suite / "manifest.json", manifest)
        for method in config["methods"]:
            method_index = list(report.LABELS).index(method)
            consolidates = method in ("consolidation_v3", "full_v3")
            boosted = method in ("newborn_v3", "newborn_matched_v3", "full_v3")
            protects = method in ("protection_v3", "full_v3")
            for seed_index, seed in enumerate(config["seeds"]):
                final = 0.65 + 0.01 * seed_index + 0.005 * method_index * (seed_index + 1)
                if condition == "early_biased" and method == "full_v3":
                    final -= 0.02 * (seed_index + 1)
                curve = [[0, final - 0.1], [2, final - 0.05], [4, final]]
                auc = final - 0.075
                norm = 0.02 + 0.01 * seed_index
                trace, events = [], []
                total_resets = 0
                for step in range(1, 11):
                    gain = 1.0 if step <= 2 else 0.5
                    if step > 2 and method in ("newborn_v3", "full_v3"):
                        gain += 0.03
                    resets = int(method != "er_v3" and step > 2 and step % 2 == 0)
                    total_resets += resets
                    trace.append({"step": step, "nominal_feature_gain": gain,
                                  "effective_feature_gain": gain, "clip_scale": 1.0,
                                  "update_norm": norm, "proposed_update_norm": norm,
                                  "feature_data_displacement": 0.01, "head_data_displacement": 0.005,
                                  "recycled_by_population": {"adapter": resets}})
                    if step % 2 == 0:
                        events.append({"step": step, "phase": "scheduled", "controller": {},
                                       "newborn_units": int(boosted and step > 4),
                                       "protected_units": int(protects and step > 4),
                                       "consolidation_gain_reduction": 0.01 if consolidates and step >= 8 else 0,
                                       "pre_consolidation_feature_gain": gain,
                                       "scheduled_feature_gain": 1.0 if step <= 2 else 0.5,
                                       "total_maturations": 2 if consolidates and step >= 8 else 0,
                                       "total_local_consolidations": 1 if consolidates and step >= 8 else 0})
                location = suite / f"{method}_seed{seed}"
                write(location / "allocation.json", {"trace": trace})
                write(location / "events.json", {"events": events})
                allocation = {
                    "steps": 10, "mean_feature_gain": sum(t["effective_feature_gain"] for t in trace) / 10,
                    "mean_nominal_feature_gain": sum(t["nominal_feature_gain"] for t in trace) / 10,
                    "post_warmup_nominal_feature_gain": trace[-1]["nominal_feature_gain"],
                    "post_warmup_effective_feature_gain": trace[-1]["effective_feature_gain"],
                    "clipped_updates": 0, "maximum_proposed_update_norm": norm,
                    "maximum_applied_update_norm": norm,
                    "summed_feature_data_displacement": 0.1, "summed_head_data_displacement": 0.05,
                }
                result = {
                    **manifest, "method": method, "seed": seed, "eval_split": "validation",
                    "class_order": [0, 1, 2, 3], "information_access": "training stream only",
                    "metrics": {"final_accuracy": final, "late_early_auc": auc, "forgetting": 0.0},
                    "accuracy_matrix": [[final if j <= i else None for j in range(5)] for i in range(5)],
                    "learning_curves": [curve] * 5, "early_auc": [auc] * 5,
                    "allocation_trace_sha256": hashlib.sha256((location / "allocation.json").read_bytes()).hexdigest(),
                    "allocation_summary": allocation,
                    "diagnostics": {"total_recycled": total_resets,
                                    "total_maturations": 2 if consolidates else 0,
                                    "total_local_consolidations": 1 if consolidates else 0,
                                    "mean_consolidation": 0.02 if consolidates else 0},
                    "stream_fingerprints": {"train": f"{condition}-{seed}",
                                            "validation": f"{'stationary' if condition == 'stationary' else 'recurring'}-{seed}"},
                    "cost": {"current_examples": 20, "replay_examples": 18, "probe_forward_examples": 4,
                             "train_forward_calls": 19, "backward_calls": 19},
                }
                write(location / "result.json", result)
                buffer = {"x": torch.zeros(2, 1), "y": torch.tensor([0, 1]), "num_seen": 20,
                          "reservoir_rng_state": torch.tensor([seed]),
                          "sample_rng_states": {"train": torch.tensor([seed + 1]), "probe": torch.tensor([seed + 2])}}
                checkpoint = {key: result[key] for key in (*identity, "config_sha256", "method", "seed")}
                checkpoint.update(completed=5, learner={"buffer": buffer})
                torch.save(checkpoint, location / "checkpoint.pt")
    return {"locked": locked, "results": results, "output": output, "configs": configs}


@pytest.mark.parametrize("field", ["methods", "seeds", "peak", "horizon", "curriculum"])
def test_lock_rejects_complete_config_tampering_even_with_selection_hash_unchanged(cohort, field):
    path = cohort["locked"] / "early_biased.json"
    config = read(path)
    if field == "methods":
        config["methods"].remove("full_v3")
    elif field == "seeds":
        config["seeds"] = [731, 842]
    elif field == "peak":
        config["allocation_v3"]["newborn_peak"] = 3
    elif field == "horizon":
        config["data"]["n_experiences"] = 6
    else:
        config["data"]["bias_experiences"] = 9
    write(path, config)
    with pytest.raises(ValueError, match="complete selection specification"):
        report.validate_locked_configs(cohort["locked"])


@pytest.mark.parametrize("kind", ["source", "runtime_hash", "device", "config_hash", "checkpoint_runtime"])
def test_suite_rejects_identity_tampering(cohort, kind):
    suite = cohort["results"] / "recurring"
    location = suite / "er_v3_seed731"
    path = location / "result.json"
    result = read(path)
    if kind == "source":
        result["source_sha256"] = "b" * 64
    elif kind == "runtime_hash":
        result["runtime_sha256"] = "b" * 64
    elif kind == "device":
        result["execution_device"] = "cuda"
        result["runtime_fingerprint"].update(cuda_runtime="12.8", gpu="fixture GPU")
        result["runtime_sha256"] = study.config_hash(result["runtime_fingerprint"])
    elif kind == "config_hash":
        manifest = read(suite / "manifest.json")
        manifest["config_sha256"] = result["config_sha256"] = "b" * 64
        write(suite / "manifest.json", manifest)
    else:
        checkpoint_path = location / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["runtime_fingerprint"]["torch"] = "changed"
        checkpoint["runtime_sha256"] = study.config_hash(checkpoint["runtime_fingerprint"])
        torch.save(checkpoint, checkpoint_path)
    write(path, result)
    with pytest.raises(ValueError, match="identity|fingerprint/hash|configuration/hash"):
        report.audit_suite(suite, cohort["configs"]["recurring"])


@pytest.mark.parametrize("kind", ["missing_result", "validation_pairing"])
def test_every_cohort_and_cross_condition_pairing_precede_all_report_writes(cohort, monkeypatch, kind):
    suite = cohort["results"] / "early_biased"
    if kind == "missing_result":
        (suite / "full_v3_seed953" / "result.json").unlink()
    else:
        for method in cohort["configs"]["early_biased"]["methods"]:
            path = suite / f"{method}_seed731" / "result.json"
            value = read(path)
            value["stream_fingerprints"]["validation"] = "different-evaluation-pool"
            write(path, value)
    calls = []
    monkeypatch.setattr(report, "analyze", lambda *args: calls.append("analyze"))
    monkeypatch.setattr(report, "archive", lambda *args: calls.append("archive"))
    monkeypatch.setattr(report, "write", lambda *args: calls.append("write"))
    with pytest.raises(ValueError, match="incomplete suite|not exactly paired"):
        report.summarize(cohort["results"], cohort["locked"], cohort["output"])
    assert calls == []
    assert not cohort["output"].exists()


@pytest.mark.parametrize("kind", ["nonfinite_gain", "summary_mismatch", "curve_mismatch", "incomplete_matrix"])
def test_suite_rejects_nonfinite_or_inconsistent_reported_values(cohort, kind):
    suite = cohort["results"] / "recurring"
    location = suite / "er_v3_seed731"
    path = location / "result.json"
    result = read(path)
    if kind == "nonfinite_gain":
        allocation_path = location / "allocation.json"
        allocation = read(allocation_path)
        allocation["trace"][0]["nominal_feature_gain"] = float("nan")
        write(allocation_path, allocation, allow_nan=True)
        result["allocation_trace_sha256"] = hashlib.sha256(allocation_path.read_bytes()).hexdigest()
    elif kind == "summary_mismatch":
        result["allocation_summary"]["maximum_applied_update_norm"] = 0.09
    elif kind == "curve_mismatch":
        result["learning_curves"][0][0][1] += 0.1
    else:
        result["accuracy_matrix"].pop()
    write(path, result)
    with pytest.raises(ValueError, match="non-finite|differs from trace|does not match|complete planned stream"):
        report.audit_suite(suite, cohort["configs"]["recurring"])


def test_summary_preserves_primary_seed_pairs_interaction_activation_and_actual_maximum(cohort, monkeypatch):
    calls = []
    monkeypatch.setattr(report, "analyze", lambda *args: calls.append("analyze"))
    monkeypatch.setattr(report, "archive", lambda *args: calls.append("archive"))
    monkeypatch.setattr(report, "plot", lambda *args: None)
    summary = report.summarize(cohort["results"], cohort["locked"], cohort["output"])
    assert summary["runs"] == 51
    primary = summary["primary_contrast"]
    assert (primary["condition"], primary["method"], primary["comparator"], primary["endpoint"]) == (
        "recurring", "newborn_v3", "recycle_v3", "late_early_auc")
    assert primary["seeds"] == [731, 842, 953] and primary["n_pairs"] == 3
    assert primary["per_seed_differences"] == pytest.approx([0.005, 0.01, 0.015])
    assert primary["mean_difference"] == pytest.approx(0.01)
    assert primary["ci_low"] == pytest.approx(0.005)
    assert primary["ci_high"] == pytest.approx(0.015)
    assert len(summary["per_seed_endpoints"]) == len(summary["mechanism_activation"]) == 51
    assert all(r["maximum_applied_update_norm"] == pytest.approx(0.04) for r in summary["methods"])
    for interaction in summary["early_bias_full_interaction"]:
        assert interaction["per_seed_differences"] == pytest.approx([-0.02, -0.04, -0.06])
        assert interaction["mean_difference"] == pytest.approx(-0.04)
    for activation in summary["mechanism_activation"]:
        consolidates = activation["method"] in ("consolidation_v3", "full_v3")
        assert activation["total_maturations"] == (2 if consolidates else 0)
        assert activation["total_local_consolidations"] == int(consolidates)
    assert calls == ["analyze", "archive"] * 3
    assert read(cohort["output"] / "diagnostics.json")["primary_contrast"] == primary
    text = (cohort["output"] / "diagnostics.md").read_text(encoding="utf-8")
    assert "Primary contrast" in text and "731:" in text and "Local consolidations" in text


def test_fixed_pool_drawdown_excludes_initial_diagonal_and_uses_observed_checkpoints():
    matrix = [[None] * 6 for _ in range(6)]
    matrix[0][0] = 1.0
    for index, value in [(1, 0.8), (3, 0.7), (5, 0.75)]:
        matrix[index][0] = value
    result = {"config": {"retention_eval_every_experiences": 2}, "accuracy_matrix": matrix}
    assert report.fixed_set_drawdown(result) == pytest.approx(0.1)
