"""Fixed pilot endpoint arithmetic and fail-closed reporting; no outcomes read."""

import copy
import json
import statistics

import pytest

from scripts import summarize_cifar10_transfer as report


RULE = {
    "stationary_baseline_final_min": 0.25,
    "stationary_baseline_improvement_min": 0.1,
    "recurring_recycling_final_min": 0.2,
    "recurring_recycling_late_auc_max_exclusive": 0.95,
    "newborn_late_auc_mean_gain_min": 0.01,
    "newborn_late_auc_each_gain_positive": True,
    "matched_late_auc_mean_gain_min": 0.005,
    "matched_late_auc_each_gain_nonnegative": True,
    "maximum_final_loss_against_er_or_recycling": 0.01,
    "scope": "fixture effort rule, not significance",
}


def endpoint_fixture(condition="recurring"):
    order = [i % 3 for i in range(30)] if condition == "recurring" else [0]*30
    diagonals = [0.4+0.1*domain for domain in order]
    if condition == "stationary":
        diagonals = [0.8]*30
        # The largest drawdown occurs between two sparse retention checkpoints.
        diagonals[11:13] = [0.9, 0.6]
    positions = [0] + list(range(128, 1409, 128)) + [1500]
    curves, matrix = [], []
    for index in range(30):
        curve = [[point, accuracy] for point, accuracy in
                 zip(positions, [0.1, 0.2, 0.3]+[diagonals[index]]*10)]
        curves.append(curve)
        row = [None]*30
        if (index+1) % 10 == 0:
            for j in range(index+1):
                row[j] = 0.4+0.1*order[j] if condition == "recurring" else diagonals[index]
        row[index] = diagonals[index]
        matrix.append(row)
    return {
        "accuracy_matrix": matrix, "learning_curves": curves,
        "stream_metadata": {"domain_order": order}, "early_auc": [0.2]*30,
        "metrics": {"final_accuracy": statistics.fmean(matrix[-1]), "late_early_auc": 0.2},
    }


def test_two_trapezoids_domain_weights_and_late_fifteen_experiences():
    result = endpoint_fixture()
    # First fifteen experiences must not enter the late endpoint.
    for index in range(15):
        result["learning_curves"][index][:3] = [[0, 0.0], [128, 0.4], [256, 0.8]]
        result["early_auc"][index] = 0.4
    values = report.endpoints(result, "recurring")
    assert values["early_auc_256_by_experience"] == pytest.approx([0.4]*15+[0.2]*15)
    assert values["late_early_auc"] == pytest.approx(0.2)
    assert values["final_accuracy"] == pytest.approx(0.5)
    assert values["final_accuracy_by_domain"] == pytest.approx({"original": 0.4, "grayscale": 0.5, "blur": 0.6})
    assert values["stationary_maximum_drawdown"] is None


def test_stationary_drawdown_uses_all_thirty_diagonals():
    values = report.endpoints(endpoint_fixture("stationary"), "stationary")
    assert values["stationary_maximum_drawdown"] == pytest.approx(0.3)
    assert len(values["stationary_end_of_experience_accuracy"]) == 30
    assert values["untrained_original_accuracy"] == 0.1
    assert values["final_accuracy_by_domain"] == {"original": 0.8}


def test_checkpoint_tuple_curves_match_json_arrays_but_changed_values_do_not():
    curves = endpoint_fixture()["learning_curves"]
    checkpoint_curves = [[tuple(point) for point in curve] for curve in curves]
    assert report.same_json_values(checkpoint_curves, curves)
    checkpoint_curves[19][2] = (256, 0.301)
    assert not report.same_json_values(checkpoint_curves, curves)


@pytest.mark.parametrize("mutation", [
    lambda r: r["learning_curves"].pop(),
    lambda r: r["learning_curves"][3].pop(1),
    lambda r: r["learning_curves"][3][1].__setitem__(0, 127),
    lambda r: r["early_auc"].__setitem__(7, 0.21),
    lambda r: r["metrics"].__setitem__("final_accuracy", float("nan")),
    lambda r: r["accuracy_matrix"][4].__setitem__(2, 0.5),
    lambda r: r["accuracy_matrix"][-1].__setitem__(0, 0.41),
    lambda r: r["accuracy_matrix"][4].__setitem__(4, 0.7),
    lambda r: r["stream_metadata"]["domain_order"].__setitem__(0, 2),
])
def test_endpoint_incompleteness_nonfinite_or_inconsistent_values_are_rejected(mutation):
    result = endpoint_fixture()
    mutation(result)
    with pytest.raises(ValueError):
        report.endpoints(result, "recurring")


@pytest.fixture
def decision_rows():
    rows = []
    for condition in report.CONDITIONS:
        for method in report.METHODS:
            for seed in report.SEEDS:
                late = {"er_v3": 0.3, "recycle_v3": 0.3,
                        "newborn_v3": 0.31, "newborn_matched_v3": 0.305}[method]
                final = 0.39 if method.startswith("newborn") else 0.4
                rows.append({"condition": condition, "method": method, "seed": seed,
                             "recomputed": {"late_early_auc": late, "final_accuracy": final,
                                            "untrained_original_accuracy": 0.1}})
    return rows


def find(rows, condition, method, seed=1063):
    return next(row["recomputed"] for row in rows if
                (row["condition"], row["method"], row["seed"]) == (condition, method, seed))


def test_inclusive_thresholds_and_all_individual_reasons_are_preserved(decision_rows):
    decision = report.effort_decision(decision_rows, RULE)
    assert decision["advance_frozen_recipe"]
    assert decision["passed"] == {"adequacy": True, "signal": True, "retention": True}
    assert {key: len(value) for key, value in decision["checks"].items()} == {
        "adequacy": 12, "signal": 6, "retention": 16,
    }
    assert all(item["reason"] and item["passed"] for items in decision["checks"].values() for item in items)
    assert len(decision["paired_recurring_late_auc"]) == 4
    # Binary subtraction at precisely 1 pp / .5 pp is inclusive.
    assert next(item["value"] for item in decision["checks"]["signal"]
                if item.get("method") == "newborn_v3" and "seeds" in item) == pytest.approx(0.01)


@pytest.mark.parametrize("condition,method,metric,value,failed", [
    ("recurring", "newborn_v3", "late_early_auc", 0.3, "signal"),
    ("recurring", "newborn_matched_v3", "late_early_auc", 0.299, "signal"),
    ("recurring", "newborn_v3", "late_early_auc", 0.309, "signal"),
    ("recurring", "recycle_v3", "late_early_auc", 0.95, "adequacy"),
    ("stationary", "er_v3", "final_accuracy", 0.249, "adequacy"),
    ("stationary", "recycle_v3", "untrained_original_accuracy", 0.301, "adequacy"),
    ("stationary", "newborn_matched_v3", "final_accuracy", 0.389, "retention"),
    ("recurring", "newborn_v3", "final_accuracy", 0.389, "retention"),
])
def test_each_seed_and_each_condition_can_fail_the_predeclared_rule(
        decision_rows, condition, method, metric, value, failed):
    find(decision_rows, condition, method)[metric] = value
    decision = report.effort_decision(decision_rows, RULE)
    assert not decision["advance_frozen_recipe"]
    assert not decision["passed"][failed]
    assert any(not item["passed"] and item.get("seed") == 1063
               for item in decision["checks"][failed]) or failed == "signal"


def test_matched_zero_individual_gain_allowed_when_mean_still_sufficient(decision_rows):
    find(decision_rows, "recurring", "newborn_matched_v3")["late_early_auc"] = 0.3
    find(decision_rows, "recurring", "newborn_matched_v3", 1174)["late_early_auc"] = 0.31
    assert report.effort_decision(decision_rows, RULE)["advance_frozen_recipe"]


def test_guard_checks_both_er_and_recycling(decision_rows):
    find(decision_rows, "stationary", "er_v3")["final_accuracy"] = 0.401
    decision = report.effort_decision(decision_rows, RULE)
    failed = [item for item in decision["checks"]["retention"] if not item["passed"]]
    assert len(failed) == 2
    assert {item["comparator"] for item in failed} == {"er_v3"}


@pytest.mark.parametrize("mutation", [lambda rows: rows.pop(), lambda rows: rows.__setitem__(0, rows[1])])
def test_decision_never_drops_missing_or_duplicate_runs(decision_rows, mutation):
    mutation(decision_rows)
    with pytest.raises(ValueError, match="sixteen unique"):
        report.effort_decision(decision_rows, RULE)


def allocation_fixture(method):
    trace = []
    for step in range(1, 1411):
        count = 3 if method != "er_v3" and step > 40 and (step-40) % 20 == 0 else 0
        gain = 1.0 if step <= 40 else 0.5
        trace.append({"step": step, "recycled_by_population": {"adapter": count},
                      "proposed_update_norm": 0.2, "update_norm": 0.1, "clip_scale": 0.5,
                      "nominal_feature_gain": gain, "effective_feature_gain": gain*0.5,
                      "feature_data_displacement": 0.06, "head_data_displacement": 0.08})
    result = {"diagnostics": {"total_recycled": 0 if method == "er_v3" else 204},
              "allocation_summary": {"steps": 1410}}
    return trace, result


@pytest.mark.parametrize("method", report.METHODS)
def test_exact_update_reset_and_clip_audit(method):
    trace, result = allocation_fixture(method)
    checked = report.audit_allocation(trace, method, result)
    assert checked["unit_resets"] == (0 if method == "er_v3" else 204)
    assert checked["updates"] == checked["clipped_updates"] == 1410
    assert checked["maximum_optimizer_update_norm"] == 0.1


@pytest.mark.parametrize("key,value", [
    ("step", 1409), ("recycled_by_population", {"adapter": 3}),
    ("update_norm", 0.1002), ("clip_scale", float("nan")),
    ("clip_scale", 0.6), ("nominal_feature_gain", 0.51),
    ("effective_feature_gain", 0.26),
])
def test_wrong_final_update_reset_cap_or_matched_budget_fails(key, value):
    trace, result = allocation_fixture("newborn_matched_v3")
    trace[-1][key] = value
    with pytest.raises(ValueError):
        report.audit_allocation(trace, "newborn_matched_v3", result)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("problem", ["missing", "extra", "manifest", "result"])
def test_partial_extra_or_tampered_cohort_fails_before_any_report_write(tmp_path, monkeypatch, problem):
    runs, output, lock_path = tmp_path / "runs", tmp_path / "output", tmp_path / "lock.json"
    identity = {key: "fixture" for key in report.IDENTITY_KEYS if key != "config_sha256"}
    configs = {condition: {"fixture": condition} for condition in report.CONDITIONS}
    lock = {**identity, "configurations": {condition: {"config_sha256": condition} for condition in report.CONDITIONS}}
    for condition in report.CONDITIONS:
        manifest = {**identity, "config_sha256": condition, "config": configs[condition],
                    "methods": report.METHODS, "seeds": report.SEEDS}
        write(runs / condition / "manifest.json", manifest)
        for method in report.METHODS:
            for seed in report.SEEDS:
                write(runs / condition / f"{method}_seed{seed}" / "result.json",
                      {**manifest, "method": method, "seed": seed})
    first = runs / "recurring" / "er_v3_seed1063" / "result.json"
    if problem == "missing":
        first.unlink()
    elif problem == "extra":
        write(runs / "recurring" / "unexpected_seed1063" / "result.json", {})
    else:
        path = runs / "recurring" / "manifest.json" if problem == "manifest" else first
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source_sha256"] = "changed"
        write(path, value)
    monkeypatch.setattr(report, "validate_lock", lambda *_: (copy.deepcopy(lock), {}, configs))
    with pytest.raises(ValueError, match="sixteen|frozen lock|identity differs"):
        report.summarize(runs, output, lock_path, root=tmp_path)
    assert not output.exists()
