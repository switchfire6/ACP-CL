"""Causal selection, exact uniform expectation, seed pairing and fixed screens."""

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
import summarize_predictive_value as summary


def config():
    return json.loads(Path("configs/predictive_value_diagnostic.json").read_text())


def fingerprint(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def record(seed=14001, model="conditional", index=0, count=8):
    def candidate(j):
        return dict(id=10+j, original_brier=.01 if j == 0 else .1+j/100,
            original_steps=12*(10+j), query_sha256=fingerprint(j),
            support_sha256=fingerprint(j-1),
            origin_law=dict(mode=seed % 2 if j < 2 else 1-seed % 2,
                            active=[], revised=False, noise=0.))
    candidates = [candidate(j) for j in range(count)]
    losses = [.2]*count+[.3]
    losses[1] = .15
    brier = [.2]*count+[.25]
    brier[1] = .18
    survival = [.8]*count+[.75]
    survival[1] = .81
    gaps = [4096, 512] if (seed//6) % 2 else [512, 4096]
    return dict(identity=dict(example="synthetic"), seed=seed, model=model, index=index,
        cue=tuple(permutations(range(3)))[seed % 6][0], gap=gaps[index], anchor_id=100,
        candidates=candidates, replacement=candidate(count),
        choice=dict(value_id=11, accuracy_id=10, score_losses=losses,
                    values=[losses[-1]-value for value in losses[:-1]]),
        panels={name: dict(brier=brier.copy(), survival=survival.copy(), early_brier=brier.copy(),
                           late_brier=brier.copy()) for name in summary.PANELS})


def records(c):
    return [record(seed, model, index, c["candidate_count"])
            for model in c["models"] for seed in c["seeds"] for index in (0, 1)]


def comparison(gain=-.0005, control=0., survival=-.005, improved=9):
    brier = summary.paired([-.001]*improved+[0.]*(12-improved))
    brier["mean"] = gain
    return dict(value_minus_uniform=dict(brier=brier, survival=summary.paired([survival]*12)),
                value_minus_accuracy=dict(brier=summary.paired([control]*12)))


@pytest.fixture(scope="module")
def complete_analysis():
    c = config()
    return summary.analyze_records(records(c), c)


def test_uniform_averages_candidate_losses_and_replacement_is_excluded():
    r = record(count=2)
    r["panels"]["near_reapplied"]["brier"] = [.8, .2, .9]
    r["panels"]["near_reapplied"]["survival"] = [.95, .4, .1]
    row = summary.assessment_metrics(r)
    panel = row["panels"]["near_reapplied"]
    assert panel["selectors"]["uniform"]["brier"] == .5
    assert panel["contrasts"]["value_minus_uniform"]["brier"] == pytest.approx(-.3)
    assert panel["selectors"]["uniform"]["survival"] == pytest.approx(.675)
    assert panel["selectors"]["replacement"]["brier"] == .9
    # Oracle is selected by full Brier; it cannot also cherry-pick survival.
    assert panel["oracle_id"] == 11
    assert panel["selectors"]["oracle"]["survival"] == .4
    assert panel["brier_spread"] == pytest.approx(.6)


def test_validation_outcomes_cannot_change_score_selected_candidate():
    r = record(count=2)
    r["panels"]["return_reapplied"]["brier"] = [.01, .8, .5]
    row = summary.assessment_metrics(r)
    assert row["value_id"] == 11
    assert row["panels"]["return_reapplied"]["oracle_id"] == 10
    assert row["panels"]["return_reapplied"]["contrasts"]["value_minus_oracle"]["brier"] == pytest.approx(.79)
    r["choice"]["value_id"] = 10
    with pytest.raises(ValueError, match="score-window"):
        summary.assessment_metrics(r)


def test_score_and_original_error_ties_use_smallest_id_not_record_order():
    r = record(count=2)
    r["candidates"] = list(reversed(r["candidates"]))
    for item in r["candidates"]:
        item["original_brier"] = .1
    r["choice"] = dict(value_id=10, accuracy_id=10, score_losses=[.2, .2, .3], values=[.3-.2]*2)
    row = summary.assessment_metrics(r)
    assert row["panels"]["score"]["oracle_id"] == 10
    assert row["selectors_agree"]
    r["choice"]["accuracy_id"] = 11
    with pytest.raises(ValueError, match="original error"):
        summary.validate_assessment(r)


def test_common_replacement_changes_absolute_values_but_not_selection_contrast():
    r = record()
    before = summary.assessment_metrics(r)
    r["choice"]["score_losses"][-1] = .8
    r["choice"]["values"] = [.8-loss for loss in r["choice"]["score_losses"][:-1]]
    after = summary.assessment_metrics(r)
    assert before["value_id"] == after["value_id"]
    assert before["panels"]["score"]["contrasts"]["value_minus_uniform"] == after["panels"]["score"]["contrasts"]["value_minus_uniform"]
    assert after["panels"]["score"]["contrasts"]["value_minus_replacement"]["brier"] < before["panels"]["score"]["contrasts"]["value_minus_replacement"]["brier"]


def test_exact_two_assessment_mean_precedes_seed_bootstrap():
    c = config()
    c.update(seeds=[14001, 14002, 14003], models=["conditional"])
    data = records(c)
    for r in data:
        # Both positive and negative assessments: never use n=6.
        offset = {14001: [-.04, .02], 14002: [-.01, -.03], 14003: [.03, -.01]}[r["seed"]][r["index"]]
        for panel in r["panels"].values():
            panel["brier"] = [.2]*8+[.25]
            panel["brier"][1] += offset
    result = summary.analyze_records(data, c)
    paired = result["contrasts"]["conditional"]["near_reapplied"]["value_minus_uniform"]["brier"]
    assert paired["differences"] == pytest.approx([-.01*7/8, -.02*7/8, .01*7/8])
    assert paired["n"] == 3 and paired["seeds"] == c["seeds"]
    assert result["coverage"]["conditional"]["assessments"] == 6
    assert not result["eligible_scientific_cohort"]


def test_bootstrap_matches_fixed_numpy_seed_and_preserves_input():
    values = np.array([-.1, .2, -.3, .4])
    original = values.copy()
    result = summary.paired(values, [1, 2, 3, 4])
    rng = np.random.default_rng(27192026)
    boot = original[rng.integers(0, 4, (20000, 4))].mean(axis=1)
    assert result["lower"] == float(np.quantile(boot, .025))
    assert result["upper"] == float(np.quantile(boot, .975))
    np.testing.assert_array_equal(original, values)
    assert summary.paired([1e-14, -1e-14])["differences"] == [0., 0.]


def test_complete_cohort_gates_and_conditional_oracle_are_separate(complete_analysis):
    assert complete_analysis["eligible_scientific_cohort"]
    assert len(complete_analysis["rows"]) == 48
    for model in summary.MODELS:
        assert complete_analysis["screen"][model]["near"]["passed"]
        assert complete_analysis["screen"][model]["return"]["passed"]
        assert complete_analysis["contrasts"][model]["near_reapplied"]["value_minus_uniform"]["brier"]["n"] == 12
        assert complete_analysis["descriptive_slices"][model]["cue"]["0"]["n"] == 4
        assert complete_analysis["descriptive_slices"][model]["gap"]["512"]["n"] == 12
        assert complete_analysis["coverage"][model]["query_support_overlap_pairs"] > 0


def test_absent_return_candidates_are_kept_in_primary_results():
    c = config()
    c.update(seeds=[14001], models=["conditional"])
    data = records(c)
    for candidate in data[0]["candidates"]:
        candidate["origin_law"]["mode"] = 1-14001 % 2
    result = summary.analyze_records(data, c)
    assert len(result["rows"]) == 2
    assert result["coverage"]["conditional"]["without_return_candidate"] == 1
    assert result["coverage"]["conditional"]["with_return_candidate"] == 1
    assert result["contrasts"]["conditional"]["return_reapplied"]["value_minus_uniform"]["brier"]["n"] == 1


def test_fixed_thresholds_allow_accuracy_tie_but_not_eight_improving_seeds():
    assert summary.tier_screen(comparison(), True)["passed"]
    assert summary.tier_screen(comparison(improved=8), True)["failed"] == ["seed_consistency"]
    assert summary.tier_screen(comparison(), False)["failed"] == ["complete_scientific_cohort"]


@pytest.mark.parametrize("field,changes", (
    ("brier_gain", dict(gain=-.00049)),
    ("accuracy_control", dict(control=.0000001)),
    ("survival", dict(survival=-.00501)),
))
def test_each_primary_gate_can_fail_independently(field, changes):
    assert summary.tier_screen(comparison(**changes), True)["failed"] == [field]


def test_smoke_and_changed_scientific_dose_are_ineligible():
    c = config()
    assert summary.eligible_config(c)
    for field, value in (("seeds", [14991]), ("rehearsal_updates", 24), ("score_size", 1024),
                         ("candidate_count", 4), ("lr", .001), ("gaps", [256, 4096])):
        changed = {**c, field: value}
        assert not summary.eligible_config(changed)
    assert summary.eligible_config({**c, "workers": 1, "cpu_affinity": 0, "study": "portable"})


@pytest.mark.parametrize("change,error", (
    (lambda r: r["candidates"].__setitem__(1, copy.deepcopy(r["candidates"][0])), "duplicate"),
    (lambda r: r["choice"]["values"].__setitem__(0, .9), "replacement values"),
    (lambda r: r["panels"].pop("return_frozen"), "incomplete validation"),
    (lambda r: r["panels"]["near_reapplied"]["brier"].__setitem__(0, float("nan")), "nonfinite"),
    (lambda r: r["panels"]["near_reapplied"]["brier"].pop(), "array"),
    (lambda r: r.__setitem__("anchor_id", 10), "latest anchor"),
))
def test_malformed_or_leaky_records_rejected(change, error):
    r = record()
    change(r)
    with pytest.raises(ValueError, match=error):
        summary.validate_assessment(r)


def test_roundoff_tolerance_does_not_clip_metrics():
    r = record()
    value = 1+np.finfo(np.float32).eps
    r["panels"]["near_reapplied"]["brier"][1] = float(value)
    assert summary.assessment_metrics(r)["panels"]["near_reapplied"]["selectors"]["value"]["brier"] == value
    r["panels"]["near_reapplied"]["brier"][1] = float(1+2*summary.PROBABILITY_METRIC_TOLERANCE)
    with pytest.raises(ValueError, match="range"):
        summary.validate_assessment(r)


@pytest.mark.parametrize("mutator", (
    lambda data: data.pop(), lambda data: data.append(copy.deepcopy(data[0])),
))
def test_incomplete_or_duplicated_assessments_rejected(mutator):
    c = config()
    c.update(seeds=[14001], models=["conditional"])
    data = records(c)
    mutator(data)
    with pytest.raises(ValueError, match="both assessments"):
        summary.analyze_records(data, c)


def test_analysis_inputs_unchanged_and_wrong_gap_rejected():
    c = config()
    c.update(seeds=[14001], models=["conditional"])
    data = records(c)
    before = copy.deepcopy(data)
    summary.analyze_records(data, c)
    assert data == before
    data[0]["gap"] = data[1]["gap"]
    with pytest.raises(ValueError, match="gap/order"):
        summary.analyze_records(data, c)


def test_source_archive_exact_files_and_content(tmp_path):
    archive = tmp_path/"snapshot.zip"
    contents = {"a.py": b"science", "b.py": b"analysis"}
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    with zipfile.ZipFile(archive, "w") as stream:
        for name, value in contents.items():
            stream.writestr(name, value)
    summary.checked_zip(archive, expected)
    with zipfile.ZipFile(archive, "w") as stream:
        for name, value in contents.items():
            stream.writestr(name, b"changed" if name == "a.py" else value)
    with pytest.raises(ValueError, match="content hash"):
        summary.checked_zip(archive, expected)
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr("unlocked.py", "extra")
    with pytest.raises(ValueError, match="file set"):
        summary.checked_zip(archive, expected)


def phase_fixture():
    c = config()
    c.update(batch_size=8, candidate_count=2)
    phase = dict(name="near_0", law=dict(mode=0, active=[0], revised=False, noise=0.),
                 size=16, kind="near", assessment=0, gap=512, cue=0)
    size, start = c["batch_size"], 10
    packets, evaluations = [], []
    for offset in range(2):
        packets.append(dict(id=start+offset, original_steps=(start+offset)*12,
            origin_phase="near_0", origin_index=offset, origin_law=phase["law"],
            query_sha256=fingerprint(offset), support_sha256=fingerprint(offset-1),
            original_prediction_sha256=fingerprint("original_prediction"),
            original_model_sha256=fingerprint("original_model"),
            probabilities=np.full((size, 3), .5).tolist(), original_brier=.25,
            actions=[0]*size, outcomes=np.zeros((size, 3)).tolist()))
        truth = np.zeros((size, 5, 3))
        truth[:, 1] = 1
        models = []
        for j in range(3):
            p = np.full((size, 3), .1*(j+1)+.05*offset)
            models.append(dict(probabilities=p.tolist(), brier=float(np.square(p).mean()),
                prediction_sha256=fingerprint([offset, j]), actions=[1]*size, survival=1.))
        evaluations.append(dict(packet_id=start+offset, query_sha256=fingerprint(offset),
            support_sha256=fingerprint(offset-1), truth=truth.tolist(),
            reapplied=models, frozen=copy.deepcopy(models)))
    def memory(seen):
        return dict(type="ReservoirMemory", seen=seen, age=seen, capacity=16, ids=list(range(seen)))
    work = dict(steps=12, forwards=24, backward_calls=12, query_presentations=24*size,
                support_presentations=24*size, parent_unchanged=True, memory_history_unchanged=True)
    panels = {}
    for group in ("reapplied", "frozen"):
        values = [[item["brier"] for item in row[group]] for row in evaluations]
        panels[group] = dict(brier=np.mean(values, axis=0).tolist(), survival=[1.]*3,
                             early_brier=values[0], late_brier=values[1])
    result = dict(identity={}, seed=14001, model="conditional", phase=phase, packets=packets,
        completed_packets=2, start_diagnostics=dict(cost=dict(arrivals=start*size, optimizer_steps=start*12)),
        diagnostics=dict(cost=dict(arrivals=(start+2)*size, optimizer_steps=(start+2)*12)),
        start_memory=memory(start), memory=memory(start+2), metadata={str(i): {} for i in range(start+2)},
        shadow_work=[copy.deepcopy(work) for _ in range(3)], evaluations=evaluations, panels=panels)
    return result, phase, c, start


def test_phase_arithmetic_reconstructs_original_shadow_survival_and_early_late():
    r, phase, c, start = phase_fixture()
    summary.verify_phase(r, phase, 14001, "conditional", c, {}, start)
    # First-packet error differs from the later packet; weighting is explicit.
    assert r["panels"]["reapplied"]["brier"][0] == pytest.approx((.1**2+.15**2)/2)


@pytest.mark.parametrize("change,error", (
    (lambda r: r["packets"][0].__setitem__("original_brier", .1), "original Brier arithmetic"),
    (lambda r: r["evaluations"][0]["reapplied"][1].__setitem__("brier", .9), "shadow Brier arithmetic"),
    (lambda r: r["evaluations"][0]["frozen"][1].__setitem__("survival", 0.), "survival arithmetic"),
    (lambda r: r["panels"]["reapplied"]["early_brier"].__setitem__(0, .4), "initial-packet aggregation"),
    (lambda r: r["packets"][0]["outcomes"][0].__setitem__(0, 1.1), "reported outcomes"),
))
def test_tampered_scalar_or_truth_is_rejected(change, error):
    r, phase, c, start = phase_fixture()
    change(r)
    with pytest.raises(ValueError, match=error):
        summary.verify_phase(r, phase, 14001, "conditional", c, {}, start)


def test_lock_covers_embedded_configuration_and_exact_current_files(tmp_path, monkeypatch):
    repository = tmp_path/"repo"
    run = tmp_path/"run"
    run.mkdir()
    files = {}
    for name in (*summary.REQUIRED_ANALYSIS, "docs/protocol.md"):
        path = repository/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = summary.sha(path)
    c = config()
    summary.write_json(run/"analysis_lock.json", dict(files=files, config=c, config_sha256=summary.digest(c)))
    with zipfile.ZipFile(run/"analysis_at_lock.zip", "w") as archive:
        for name in files:
            archive.write(repository/name, name)
    monkeypatch.setattr(summary, "ROOT", repository)
    summary.verify_analysis_lock(run, c, files["docs/protocol.md"])
    with pytest.raises(ValueError, match="configuration changed"):
        summary.verify_analysis_lock(run, {**c, "rehearsal_updates": 24}, files["docs/protocol.md"])
    (repository/summary.REQUIRED_ANALYSIS[0]).write_text("changed after lock")
    with pytest.raises(ValueError, match="current locked"):
        summary.verify_analysis_lock(run, c, files["docs/protocol.md"])


def test_markdown_includes_all_seed_values_and_scope(complete_analysis):
    text = summary.markdown(complete_analysis)
    for seed in summary.SEEDS:
        assert f"| conditional | {seed} |" in text
        assert f"| recurrent | {seed} |" in text
    assert "does not evaluate an adaptive retention policy" in text
    assert "query/support overlap" in text
