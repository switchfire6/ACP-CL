"""Factorial algebra, unchanged choices, branch pairing and replication gates."""

import copy
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import summarize_rehearsal_state as summary
from summarize_predictive_value import paired as prior_paired


def config():
    return json.loads(Path("configs/rehearsal_state_diagnostic.json").read_text())


def fingerprint(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def polynomial(cell):
    w, o, a = map(int, cell)
    return -.004+.002*w+.003*o+.004*a+.006*w*o+.007*w*a+.008*o*a+.010*w*o*a


def record(seed=15001, model="conditional", index=0):
    def candidate(j):
        return dict(id=10+j, original_brier=.01 if j == 1 else .1+j/100,
            original_steps=(10+j)*12, query_sha256=fingerprint(j), support_sha256=fingerprint(j-1),
            origin_law=dict(mode=seed % 2, active=[], revised=False, noise=0.))
    loss = [.1]+[.2]*7+[.3]
    panels, zero = {}, {}
    for branch in summary.BRANCHES:
        panels[branch] = {}
        for cell in summary.CELLS:
            baseline = .2+.01*int(cell[0])
            delta = polynomial(cell)*(1 if branch == "current" else -.5)
            brier = [baseline+delta]+[baseline-delta/7]*7+[.27]
            survival = [.7+.2*delta]+[.7-.2*delta/7]*7+[.72]
            panels[branch][cell] = dict(brier=brier, survival=survival,
                early_brier=brier.copy(), late_brier=brier.copy())
        zero[branch] = dict(brier=[.21, .3], survival=[.65, .6], early_brier=[.21, .3], late_brier=[.21, .3])
    return dict(identity={}, seed=seed, model=model, index=index,
        cue=tuple(permutations(range(3)))[seed % 6][0], anchor_id=100,
        candidates=[candidate(j) for j in range(8)], replacement=candidate(8),
        choice=dict(value_id=10, accuracy_id=11, score_losses=loss, values=[.3-x for x in loss[:-1]]),
        panels=panels, zero=zero)


@pytest.fixture(scope="module")
def full_result():
    c = config()
    rows = [record(seed, model, index) for model in c["models"] for seed in c["seeds"] for index in (0, 1)]
    return summary.analyze_records(rows, c)


def test_factorial_all_simple_main_pair_triple_and_repairs_have_correct_scaling():
    result = summary.assessment_metrics(record())
    actual = result["effects"]["current"]["selection"]
    expected = dict(total_111_minus_000=.04, weights_main=.011, optimizer_main=.0125, anchor_main=.014,
        weights_optimizer_interaction=.011, weights_anchor_interaction=.012, optimizer_anchor_interaction=.013,
        three_way_interaction=.010, repair_weights=.025, repair_optimizer=.027, repair_anchor=.029,
        weights_at_optimizer0_anchor0=.002, weights_at_optimizer1_anchor0=.008,
        weights_at_optimizer0_anchor1=.009, weights_at_optimizer1_anchor1=.025)
    for name, value in expected.items():
        assert actual[name]["brier"] == pytest.approx(value)
    # Main effects alone do not uniquely decompose the endpoint difference.
    assert sum(actual[name+"_main"]["brier"] for name in summary.FACTORS)+actual["three_way_interaction"]["brier"]/4 == pytest.approx(actual["total_111_minus_000"]["brier"])
    assert result["effects"]["current"]["chosen"]["repair_weights"]["brier"] == pytest.approx(.035)
    assert result["effects"]["current"]["uniform"]["repair_weights"]["brier"] == pytest.approx(.01)


def test_every_factor_has_four_conditional_effects_and_contrasts_sum_to_zero():
    assert summary.CELLS == ("000", "001", "010", "011", "100", "101", "110", "111")
    for factor in summary.FACTORS:
        assert len([name for name in summary.EFFECTS if name.startswith(factor+"_at_")]) == 4
    for weights in summary.EFFECTS.values():
        assert sum(weights.values()) == pytest.approx(0)


def test_exact_uniform_excludes_replacement_and_both_zero_baselines():
    row = summary.assessment_metrics(record())
    cell = row["panels"]["current"]["000"]
    assert cell["selectors"]["uniform"]["brier"] == pytest.approx(.2)
    assert cell["contrasts"]["value_minus_uniform"]["brier"] == pytest.approx(-.004)
    assert cell["selectors"]["replacement"]["brier"] == .27
    assert cell["contrasts"]["value_minus_matching_zero"]["brier"] == pytest.approx(.196-.21)
    new = row["panels"]["current"]["100"]
    assert new["contrasts"]["value_minus_matching_zero"]["brier"] == pytest.approx(.208-.3)


def test_environment_is_return_minus_current_and_selected_id_never_changes():
    row = summary.assessment_metrics(record())
    for cell in summary.CELLS:
        assert row["environment"]["selection"][cell]["brier"] == pytest.approx(-1.5*polynomial(cell))
    assert row["value_id"] == 10 and row["accuracy_id"] == 11


def test_oracle_uses_brier_chosen_candidate_for_survival_not_separate_best():
    r = record()
    r["panels"]["current"]["111"]["brier"] = [.2, .3, .1, .4, .5, .6, .7, .8, .9]
    r["panels"]["current"]["111"]["survival"] = [.99, .7, .5, .8, .6, .9, .8, .4, .8]
    row = summary.assessment_metrics(r)
    assert row["panels"]["current"]["111"]["oracle_id"] == 12
    assert row["panels"]["current"]["111"]["selectors"]["oracle"]["survival"] == .5
    assert row["panels"]["current"]["111"]["selectors"]["value"]["brier"] == .2
    r["choice"]["value_id"] = 12
    with pytest.raises(ValueError, match="choice changed"):
        summary.validate_assessment(r)


def test_score_and_original_accuracy_ties_break_on_smallest_id():
    r = record()
    r["candidates"] = r["candidates"][::-1]
    for candidate in r["candidates"]:
        candidate["original_brier"] = .1
    r["choice"] = dict(value_id=10, accuracy_id=10, score_losses=[.2]*8+[.3], values=[.3-.2]*8)
    summary.validate_assessment(r)
    r["choice"]["accuracy_id"] = 17
    with pytest.raises(ValueError, match="choice changed"):
        summary.validate_assessment(r)


def test_complete_scientific_cohort_has_twelve_seed_units_not_24_or_cells(full_result):
    assert full_result["eligible_scientific_cohort"]
    assert len(full_result["rows"]) == 48
    for model in summary.MODELS:
        statistic = full_result["effects"][model]["current"]["selection"]["total_111_minus_000"]["brier"]
        assert statistic["n"] == 12 and statistic["seeds"] == summary.SEEDS
        assert statistic["differences"] == pytest.approx([.04]*12)
        assert full_result["screen"][model]["passed"]
        assert full_result["descriptive_slices"][model]["cue"]["0"]["n"] == 4
        assert full_result["coverage"][model]["assessments"] == 24


def test_paired_bootstrap_is_exact_existing_fixed_function():
    assert summary.paired is prior_paired
    assert summary.paired([1e-14, -1e-14])["differences"] == [0., 0.]


def test_rank_agreement_handles_ties_and_constants_without_silent_dropping():
    assert summary.rank_agreement([1, 2, 3], [4, 5, 6]) == pytest.approx(1)
    assert summary.rank_agreement([1, 2, 3], [6, 5, 4]) == pytest.approx(-1)
    assert summary.rank_agreement([1, 1, 2, 3], [1, 2, 2, 3]) == pytest.approx(5/6)
    assert summary.rank_agreement([1, 1, 1], [1, 2, 3]) is None
    assert summary.rank_agreement([1, 2, 3], [1, 1, 1]) is None


def test_two_assessments_are_averaged_before_bootstrap_and_absence_is_not_excluded():
    c = config()
    c.update(seeds=[15001, 15002], models=["conditional"])
    data = [record(seed, "conditional", index) for seed in c["seeds"] for index in (0, 1)]
    for r in data:
        delta = {15001: [.01, .03], 15002: [-.02, .01]}[r["seed"]][r["index"]]
        for cell in summary.CELLS:
            d = delta if cell == "111" else 0.
            r["panels"]["current"][cell]["brier"] = [.2+d]+[.2-d/7]*7+[.3]
        for item in r["candidates"]:
            item["origin_law"]["mode"] = 1-r["seed"] % 2
    before = copy.deepcopy(data)
    result = summary.analyze_records(data, c)
    statistic = result["effects"]["conditional"]["current"]["selection"]["total_111_minus_000"]["brier"]
    assert statistic["differences"] == pytest.approx([.02, -.005])
    assert statistic["n"] == 2
    assert result["coverage"]["conditional"]["without_return_candidate"] == 4
    assert data == before


def test_replication_thresholds_are_inclusive_and_do_not_depend_on_ci():
    total = summary.paired([.001]*9+[-.001]*3)
    total["mean"] = .0005
    total["lower"], total["upper"] = -.9, .9
    assert summary.replication_screen(total, True)["passed"]
    assert summary.replication_screen(total, False)["failed"] == ["complete_scientific_cohort"]
    total["mean"] = .000499
    assert summary.replication_screen(total, True)["failed"] == ["deterioration"]
    total["mean"] = .0005
    total["differences"] = [.001]*8+[-.001]*4
    assert summary.replication_screen(total, True)["failed"] == ["seed_consistency"]


def test_smoke_changed_dose_and_old_seeds_cannot_pass_scientific_eligibility():
    c = config()
    assert summary.eligible_config(c)
    for key, value in (("seeds", list(range(14001, 14013))), ("candidate_count", 4),
                       ("rehearsal_updates", 24), ("inter_assessment_size", 2048), ("validation_size", 1024)):
        assert not summary.eligible_config({**c, key: value})
    assert summary.eligible_config({**c, "workers": 1, "cpu_affinity": 0, "study": "portable"})


@pytest.mark.parametrize("change,error", (
    (lambda r: r["panels"]["return"].pop("010"), "factorial cells"),
    (lambda r: r["zero"]["current"]["brier"].append(.9), "array"),
    (lambda r: r["panels"]["current"]["000"]["survival"].__setitem__(0, float("nan")), "nonfinite"),
    (lambda r: r["choice"]["values"].__setitem__(0, .999), "values mismatch"),
    (lambda r: r["replacement"].__setitem__("id", 10), "duplicate"),
))
def test_malformed_factorial_or_changed_selection_rejected(change, error):
    r = record()
    change(r)
    with pytest.raises(ValueError, match=error):
        summary.validate_assessment(r)


def test_missing_or_duplicate_assessments_rejected_before_inference():
    c = config()
    c.update(seeds=[15001], models=["conditional"])
    for data in ([record()], [record(), record()]):
        with pytest.raises(ValueError, match="both assessments"):
            summary.analyze_records(data, c)


def arrays_fixture():
    c = config()
    c.update(batch_size=8, validation_size=16)
    observations = np.zeros((2, 8, 3, 4), dtype=np.float32)
    actions = np.zeros((2, 8), dtype=np.int64)
    truth = np.zeros((2, 2, 8, 5, 3), dtype=np.float32)
    truth[0, :, :, 1] = 1
    truth[1] = 1-truth[0]
    arrays = dict(observations=observations, actions=actions,
        outcomes=np.stack([np.zeros((2, 8, 3)), np.ones((2, 8, 3))]).astype(np.float32), truth=truth,
        initial_observations=observations[0], initial_actions=actions[0],
        initial_outcomes=np.zeros((8, 3), dtype=np.float32))
    probabilities = np.zeros((2, 2, 9, 8, 5, 3), dtype=np.float32)
    for candidate in range(9):
        for packet in range(2):
            probabilities[:, packet, candidate, :, 0] = .05*(candidate+1)+.01*packet
    probabilities[..., 1, :] = .9
    return arrays, probabilities, c


def test_complete_array_metrics_respect_candidate_action_horizon_and_branch_axes():
    arrays, probabilities, c = arrays_fixture()
    summary.validate_arrays(arrays, c)
    result = summary.metric_panels(probabilities, arrays, 9)
    for j in range(9):
        values = [float(probabilities[0, p, j, 0, 0, 0]) for p in range(2)]
        assert result["current"]["brier"][j] == pytest.approx(sum(v*v for v in values)/2)
        assert result["return"]["brier"][j] == pytest.approx(sum((1-v)**2 for v in values)/2)
        assert result["current"]["early_brier"][j] == pytest.approx(values[0]**2)
        assert result["current"]["late_brier"][j] == pytest.approx(values[1]**2)
        assert result["current"]["survival"][j] == 1.
        assert result["return"]["survival"][j] == 0.
    zero = summary.metric_panels(probabilities[:, :, :2], arrays, 2)
    assert len(zero["current"]["brier"]) == 2


def test_invalid_data_truth_shapes_and_probability_ranges_are_rejected():
    arrays, probabilities, c = arrays_fixture()
    broken = copy.deepcopy(arrays)
    broken["truth"][0, 0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="reported feedback"):
        summary.validate_arrays(broken, c)
    broken = copy.deepcopy(arrays)
    broken["outcomes"][0, 0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="outcome/truth range"):
        summary.validate_arrays(broken, c)
    with pytest.raises(ValueError, match="probability arrays"):
        summary.metric_panels(probabilities[:, :, :-1], arrays, 9)
    probabilities[0, 0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="probability arrays"):
        summary.metric_panels(probabilities, arrays, 9)


def test_numeric_array_loading_disallows_pickle_and_preserves_dtype(tmp_path):
    path = tmp_path/"predictions.npz"
    expected = np.array([.1, .2], dtype=np.float32)
    np.savez_compressed(path, probabilities=expected)
    actual = summary.read_arrays(path)["probabilities"]
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    np.savez_compressed(path, probabilities=np.array([{}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        summary.read_arrays(path)


def test_prospective_lock_requires_new_analysis_and_embedded_config(tmp_path, monkeypatch):
    root, run = tmp_path/"repo", tmp_path/"run"
    run.mkdir()
    files = {}
    for name in (*summary.REQUIRED_ANALYSIS, "docs/protocol.md"):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = summary.sha(path)
    c = config()
    summary.write_json(run/"analysis_lock.json", dict(files=files, config=c, config_sha256=summary.digest(c)))
    with zipfile.ZipFile(run/"analysis_at_lock.zip", "w") as archive:
        for name in files:
            archive.write(root/name, name)
    monkeypatch.setattr(summary, "ROOT", root)
    summary.verify_analysis_lock(run, c, files["docs/protocol.md"])
    with pytest.raises(ValueError, match="config mismatch"):
        summary.verify_analysis_lock(run, {**c, "rehearsal_updates": 3}, files["docs/protocol.md"])
    (root/summary.REQUIRED_ANALYSIS[0]).write_text("changed")
    with pytest.raises(ValueError, match="analysis differs"):
        summary.verify_analysis_lock(run, c, files["docs/protocol.md"])


def test_markdown_contains_all_seed_results_and_counterfactual_limits(full_result):
    text = summary.markdown(full_result)
    for seed in summary.SEEDS:
        assert f"| conditional | {seed} |" in text
    assert "complete Adam state" in text and "Zero-update references are outside" in text
    assert "No largest-factor winner or policy gate" in text
