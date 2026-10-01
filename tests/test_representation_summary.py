"""Independent reduction, paired cue masking, and non-rescuable study gates."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from acp_cl.representation_learning import design


spec = importlib.util.spec_from_file_location("representation_summary",
    Path(__file__).resolve().parents[1] / "scripts/summarize_representation_learning.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def configuration(kind="qualification", smoke=False):
    return dict(kind=kind, smoke=smoke,
        seeds=list(range(18101, 18107) if kind == "qualification" else range(18201, 18207)),
        episode_size=8192, prefix_size=1024, interleave_cycles=8, interleave_block=1024)


def targets(phase):
    return [dict(law=law, metrics=dict(brier=.05), marginal_gain=.10,
                 cue_benefits={str(cue): .006 for cue in law["active"]})
            for law in phase["targets"]]


def reference_rows(config):
    return [dict(seed=seed, arm="outcome", phase=phase["name"], introduced_cue=phase["cue"],
                 end_targets=targets(phase))
            for seed in config["seeds"] for phase in summary.expected_phases(seed, config)]


def main_rows(config):
    rows = []
    for seed in config["seeds"]:
        for arm in summary.ARMS:
            for phase in summary.expected_phases(seed, config):
                loss = .08 if arm == "sequence" else .085
                rows.append(dict(seed=seed, arm=arm, phase=phase["name"], introduced_cue=phase["cue"],
                    size=sum(block["size"] for block in phase["blocks"]),
                    stream=dict(brier=loss, focus_brier=loss, survival=.8),
                    end_targets=targets(phase), valid_after={str(mode): dict(valid_brier=loss)
                                                           for mode in (0, 1)}))
    return rows


@pytest.mark.parametrize("kind", ("qualification", "main"))
def test_independent_schedule_matches_declared_law_order_and_sizes(kind):
    config = configuration(kind)
    fn = design.qualification_phases if kind == "qualification" else design.main_phases
    for seed in config["seeds"]:
        assert summary.expected_phases(seed, config) == json.loads(json.dumps(fn(seed, config)))


def test_score_uses_all_actions_float64_and_first_argmax_for_survival():
    truth = np.zeros((2, 5, 3), np.uint8)
    truth[0, 0] = 1
    truth[1, 1] = 1
    prediction = np.full((2, 5, 3), .5, np.float32)
    affected, valid = np.array([True, False]), np.array([False, True])
    score = summary.score(prediction, truth, affected, valid)
    assert score["brier"] == .25 and score["focus_brier"] == .25
    assert score["survival"] == .5 and score["focus_survival"] == 1.
    assert score["valid_survival"] == 0. and score["clairvoyant_upper"] == 1.


def test_cue_gain_rescores_correct_forecast_on_the_flipped_cue_affected_subset():
    correct = dict(probabilities=np.repeat(np.array([.1, .1, .9, .9], np.float32)[:, None, None], 15, 1).reshape(4, 5, 3),
                   truth=np.zeros((4, 5, 3), np.uint8), affected=np.ones(4, bool), valid=np.ones(4, bool))
    flipped = copy.deepcopy(correct)
    flipped["affected"] = np.array([True, True, False, False])
    flipped["probabilities"][:2] = .2
    assert summary.cue_benefit(correct, flipped) == pytest.approx(.03)
    flipped["truth"][0] = 1
    with pytest.raises(ValueError, match="physical target"):
        summary.cue_benefit(correct, flipped)


@pytest.mark.parametrize("bad", (.5, np.nan, np.inf, -1., 3.))
def test_marginal_totals_reject_fractional_nonfinite_negative_and_excess(bad):
    counts, successes = np.full(5, 2), np.ones((5, 3), float)
    assert np.all(summary.marginal(counts, successes) == .5)
    successes[0, 0] = bad
    with pytest.raises(ValueError, match="training totals"):
        summary.marginal(counts, successes)


def test_all_31_qualification_checks_are_required_and_newest_absolute_error_is_protected():
    config = configuration()
    rows = reference_rows(config)
    result = summary.qualification(rows, config)
    assert result["decision"] == "PASS" and len(result["groups"]) == 31
    for row in rows:
        if row["phase"] == "interleaved":
            row["end_targets"][4]["metrics"]["brier"] = .121
    result = summary.qualification(rows, config)
    assert result["decision"] == "STOP"
    assert [item["name"] for item in result["groups"] if not item["passed"]] == ["interleaved/slot_4/absolute_brier"]


def test_fresh_groups_are_stage_and_introduced_cue_separately_not_crossed_cells():
    config = configuration()
    rows = reference_rows(config)
    for row in rows:
        if row["phase"] == "fresh_1" and row["introduced_cue"] == 0:
            row["end_targets"][0]["cue_benefits"]["0"] = 0.
    result = summary.qualification(rows, config)
    assert result["decision"] == "PASS"
    assert all(item["metric"]["n"] == 6 for item in result["groups"] if item["name"].startswith("fresh/"))


def test_good_overall_cue_average_does_not_rescue_failed_introduced_cue():
    config = configuration()
    rows = reference_rows(config)
    for row in rows:
        if row["phase"].startswith("fresh_") and row["introduced_cue"] == 1:
            row["end_targets"][0]["cue_benefits"]["1"] = 0.
    result = summary.qualification(rows, config)
    assert result["decision"] == "STOP"
    assert [item["name"] for item in result["groups"] if not item["passed"]] == ["fresh/cue_1/cue_benefit"]


def test_earlier_active_cue_must_remain_available_in_interleaved_reference():
    config = configuration()
    rows = reference_rows(config)
    for row in rows:
        if row["phase"] == "interleaved":
            row["end_targets"][4]["cue_benefits"]["0"] = 0.
    result = summary.qualification(rows, config)
    assert result["decision"] == "STOP"
    assert any(item["name"] == "interleaved/stage_3/cue_0/cue_benefit" and not item["passed"]
               for item in result["groups"])


@pytest.mark.parametrize("duplicate", (True, False))
def test_missing_or_duplicate_reference_phase_cannot_qualify(duplicate):
    config = configuration()
    rows = reference_rows(config)
    rows = rows + [rows[0]] if duplicate else rows[1:]
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        summary.qualification(rows, config)


def test_engineering_smoke_never_qualifies_despite_good_numbers():
    config = configuration(smoke=True)
    result = summary.qualification(reference_rows(config), config)
    assert result["raw_passed"] and not result["eligible"] and result["decision"] == "INELIGIBLE"


def test_main_primary_is_equal_stage_weight_and_not_query_count_weight():
    config = configuration("main")
    rows = main_rows(config)
    for row in rows:
        if row["arm"] == "sequence" and row["phase"].startswith("novel_"):
            stage = int(row["phase"][-1])
            row["stream"]["focus_brier"] = [.03, .06, .09][stage - 1]
            row["size"] = [32, 320, 3200][stage - 1]
    result = summary.main_comparison(rows, config)
    assert result["levels"]["sequence"]["acquisition"]["mean"] == pytest.approx(.06)


def test_mean_primary_gain_does_not_replace_five_seed_consistency():
    config = configuration("main")
    rows = main_rows(config)
    for row in rows:
        if row["arm"] == "sequence" and row["phase"].startswith("novel_"):
            row["stream"]["focus_brier"] = .01 if row["seed"] in config["seeds"][:4] else .09
    result = summary.main_comparison(rows, config)
    check = next(item for item in result["groups"] if item["name"] == "acquisition_minus_outcome")
    assert check["metric"]["mean"] < -.002 and check["metric"]["negative"] == 4
    assert not check["passed"] and result["decision"] == "STOP"


def test_main_strong_acquisition_does_not_rescue_lost_old_competence_or_active_cue():
    config = configuration("main")
    rows = main_rows(config)
    assert summary.main_comparison(rows, config)["decision"] == "PASS"
    for row in rows:
        if row["arm"] == "sequence" and row["phase"] == "return_new":
            row["valid_after"]["0"]["valid_brier"] = .13
            row["end_targets"][0]["cue_benefits"]["1"] = .001
    result = summary.main_comparison(rows, config)
    failed = {item["name"] for item in result["groups"] if not item["passed"]}
    assert "return_new/valid_mode_0_absolute" in failed and "return_new/active_cue_1" in failed
    assert result["decision"] == "STOP"


@pytest.mark.parametrize("tamper", ("reported_metric", "calls", "raw_array"))
def test_raw_evaluation_audit_rejects_changed_arithmetic_counts_or_array_bytes(tmp_path, tamper):
    values = dict(probabilities=np.full((2, 5, 3), .5, np.float32),
                  truth=np.zeros((2, 5, 3), np.uint8), observations=np.zeros((2, 4, 12, 16), np.uint8),
                  affected=np.ones(2, bool), valid=np.ones(2, bool))
    metrics = dict(brier=.25, focus_brier=.25, valid_brier=.25, survival=0.,
                   focus_survival=0., valid_survival=0., no_transfer=0., clairvoyant_upper=0.)
    record = dict(index=0, law=summary.law(0), cue=None, branch="novel", flipped=False,
        support_fingerprint=None, query_sha256=summary.fingerprint(values["observations"]),
        model_sha256="a" * 64, metrics=metrics, marginal=None, query_presentations=2,
        support_presentations=8, observed_support_presentations=0,
        array_sha256={name: summary.fingerprint(value) for name, value in values.items()})
    payload = dict(seed=1, records=[record], calls=1, query_presentations=2,
                   support_presentations=8, observed_support_presentations=0)
    summary.write(tmp_path / "evaluations.json", payload)
    np.savez(tmp_path / "evaluations.npz", **{"e0_" + name: value for name, value in values.items()})
    assert len(summary.read_evaluations(tmp_path, dict(eval_size=2, batch_size=8), 1)[0]) == 1
    if tamper == "reported_metric":
        payload["records"][0]["metrics"]["brier"] = .24
    elif tamper == "calls":
        payload["calls"] = 2
    else:
        values["probabilities"][0, 0] = .4
        np.savez(tmp_path / "evaluations.npz", **{"e0_" + name: value for name, value in values.items()})
    summary.write(tmp_path / "evaluations.json", payload)
    with pytest.raises(ValueError):
        summary.read_evaluations(tmp_path, dict(eval_size=2, batch_size=8), 1)
