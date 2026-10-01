"""Fixed factorial signs, independent seed counts, mode guardrails and source locks."""

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import summarize_selective_updates as summary
from summarize_training_state import metrics as original_metrics, paired as original_paired
from acp_cl.acquisition.study import Evaluator
from acp_cl.acquisition.world import order
from acp_cl.replay_renewal.study import new_learner


def metric_record():
    def values(brier, marginal=False):
        result = dict(brier=brier, focus_brier=brier, survival=.8, focus_survival=.8,
            valid_brier=.1, valid_survival=.9, no_transfer=.4, clairvoyant_upper=.95)
        if marginal:
            result["marginal_brier"] = .45
        return result

    def probe(brier, marginal=False):
        return dict(replicates=[dict(curve=[dict(feedback=n, metrics=values(brier, marginal))
            for n in (0, 8, 16, 32)], flipped=values(brier+.03, marginal)) for _ in range(2)])

    return dict(cue=0, before=probe(.4), after=probe(.2, True),
        curve=[dict(arrivals=0, metrics=values(.4)), dict(arrivals=4, metrics=values(.3, True)),
               dict(arrivals=10, metrics=values(.2, True))],
        valid_before={"0": dict(valid_brier=.1, valid_survival=.9), "1": dict(valid_brier=.2, valid_survival=.8)},
        valid_after={"0": dict(valid_brier=.12, valid_survival=.9), "1": dict(valid_brier=.18, valid_survival=.8)},
        start_diagnostics=dict(cost=dict(training_seconds=2.)),
        diagnostics=dict(cost=dict(training_seconds=5.), encoder_displacement=.3))


def synthetic_rows():
    config = dict(seeds=list(range(13001, 13007)), models=["conditional", "recurrent"],
                  prefix_blocks=4, prefix_size=1024, episode_size=8192)
    # A factorial example with current effect -.01, protection effect -.003,
    # and a further -.002 interaction in acquisition.
    acquisition = dict(reference=.20, current=.19, protected=.197, combined=.185, shrink=.189)
    retention = dict(reference=.10, current=.104, protected=.095, combined=.099, shrink=.103)
    rows = []
    for model in config["models"]:
        for seed in config["seeds"]:
            for arm in summary.ALL_ARMS:
                row = {field: .0 for field in summary.FIELDS}
                row.update(seed=seed, model=model, stage=1, arm=arm, branch="novel", cue=order(seed)[0],
                    brier_auc=acquisition.get(arm, .22), all_brier_auc=acquisition.get(arm, .22),
                    valid_after=retention.get(arm, .11), valid_after_mode_0=retention.get(arm, .11),
                    valid_after_mode_1=retention.get(arm, .11), survival_auc=.8,
                    cue_effect=.02, marginal_gain=.05, valid_before=.09, before=.3,
                    training_seconds=2., encoder_displacement=.4)
                rows.append(row)
    return rows, config


def comparison(acquisition=-.003, mode0=.004, mode1=.004, survival=0.):
    return {name: original_paired([value]*6) for name, value in (
        ("brier_auc", acquisition), ("valid_after_mode_0", mode0), ("valid_after_mode_1", mode1),
        ("survival_auc", survival), ("valid_after", (mode0+mode1)/2))}


def test_reuses_original_metrics_and_bootstrap_and_extends_modes_only():
    assert summary.metrics is original_metrics and summary.paired is original_paired
    record = metric_record()
    row = summary.episode_metrics(record)
    # Unequally spaced probes verify the existing trapezoid AUC semantics.
    assert row["brier_auc"] == pytest.approx((4*.35+6*.25)/10)
    assert row["cue_effect"] == pytest.approx(.03)
    assert row["marginal_gain"] == pytest.approx(.25)
    assert row["valid_after_mode_0"] == .12 and row["valid_after_mode_1"] == .18
    assert row["valid_damage_mode_0"] == pytest.approx(.02)
    assert row["valid_damage_mode_1"] == pytest.approx(-.02)
    assert row["training_seconds"] == 3.
    for key, value in original_metrics(record).items():
        assert row[key] == value


def test_actual_evaluator_marginal_probe_omits_marginal_from_flipped_cue_only():
    torch.set_num_threads(1)
    config = json.loads(Path("configs/selective_updates_smoke.json").read_text())
    learner = new_learner("conditional", 41, config)
    evaluator = Evaluator(41, config)
    law, cue = summary.stage_law(41, 1), order(41)[0]
    result = evaluator.probe(learner, law, cue, marginal=np.full((5, 3), .5))
    assert all("marginal_brier" in point["metrics"]
               for row in result["replicates"] for point in row["curve"])
    assert all("marginal_brier" not in row["flipped"] for row in result["replicates"])
    summary.check_probe(result, law, result["model_sha256"], config, marginal=True)
    del result["replicates"][0]["curve"][-1]["metrics"]["marginal_brier"]
    with pytest.raises(KeyError, match="marginal_brier"):
        summary.check_probe(result, law, result["model_sha256"], config, marginal=True)


def test_factorial_simple_main_interaction_and_shrink_contrasts_keep_six_seed_units():
    rows, config = synthetic_rows()
    result = summary.analyze_rows(rows, config)
    expected = dict(current_minus_reference=-.01, combined_minus_protected=-.012,
        protected_minus_reference=-.003, combined_minus_current=-.005,
        current_main=-.011, protection_main=-.004, interaction=-.002,
        combined_minus_reference=-.015, combined_minus_shrink=-.004)
    for model in config["models"]:
        for name, value in expected.items():
            item = result["contrasts"][model][name]["brier_auc"]
            assert item["mean"] == pytest.approx(value)
            assert item["n"] == len(item["differences"]) == 6
            assert item["seeds"] == config["seeds"]
        assert result["screen"][model]["passed"]
        assert result["protection_attribution"][model]["passed"]
        assert all(group["n"] == 2 for group in result["cue_means"][model].values())


def test_pairing_keeps_seed_specific_differences_and_not_episode_or_query_counts():
    rows, config = synthetic_rows()
    expected = []
    for index, seed in enumerate(config["seeds"]):
        delta = -.001*(index+1)
        expected.append(delta)
        for row in rows:
            if row["seed"] == seed and row["arm"] == "combined":
                row["brier_auc"] = .20+delta
    result = summary.analyze_rows(rows, config)
    item = result["contrasts"]["conditional"]["combined_minus_reference"]["brier_auc"]
    assert item["differences"] == pytest.approx(expected)
    assert item["n"] == 6
    assert item["mean"] == pytest.approx(-.0035)


def test_per_mode_retention_cannot_be_hidden_by_safe_average():
    item = comparison(mode0=.006, mode1=-.006)
    assert item["valid_after"]["mean"] == 0
    screen = summary.primary_screen(item, qualified=True)
    assert not screen["passed"]
    assert screen["failed"] == ["valid_old_mode_0"]


@pytest.mark.parametrize("key,changes", (
    ("acquisition_gain", dict(acquisition=-.001)),
    ("survival", dict(survival=-.011)),
    ("valid_old_mode_1", dict(mode1=.006)),
))
def test_primary_failure_gates(key, changes):
    assert key in summary.primary_screen(comparison(**changes), True)["failed"]


def test_exact_fixed_thresholds_and_five_seed_consistency():
    item = comparison()
    # Set exact means to avoid float-summation differences at test boundaries.
    for name, value in (("brier_auc", -.002), ("valid_after_mode_0", .005),
                        ("valid_after_mode_1", .005), ("survival_auc", -.01)):
        item[name]["mean"] = value
    item["brier_auc"]["differences"] = [-.0024]*5+[0.]
    assert summary.primary_screen(item, True)["passed"]
    item["brier_auc"]["differences"] = [-.004]*4+[.001, .001]
    screen = summary.primary_screen(item, True)
    assert screen["failed"] == ["acquisition_consistency"]
    assert screen["gates"]["acquisition_consistency"]["improved_seeds"] == 4


def test_interval_crossing_is_reported_without_changing_mean_based_screen():
    item = comparison()
    item["brier_auc"].update(mean=-.003, lower=-.006, upper=.001)
    screen = summary.primary_screen(item, True)
    assert screen["passed"] and screen["gates"]["acquisition_gain"]["uncertainty_crosses_limit"]


@pytest.mark.parametrize("control,metric,value,failed", (
    ("current", "valid_after", -.0019, "retention_vs_current"),
    ("shrink", "valid_after", -.0019, "retention_vs_shrink"),
    ("current", "brier_auc", .0011, "acquisition_cost_vs_current"),
    ("shrink", "brier_auc", .0011, "acquisition_cost_vs_shrink"),
))
def test_directional_attribution_requires_both_controls_and_both_metrics(control, metric, value, failed):
    comparisons = {f"combined_minus_{name}": dict(valid_after=original_paired([-.003]*6),
        brier_auc=original_paired([0.]*6)) for name in ("current", "shrink")}
    assert summary.attribution_screen(comparisons, True)["passed"]
    comparisons[f"combined_minus_{control}"][metric] = original_paired([value]*6)
    result = summary.attribution_screen(comparisons, True)
    assert result["failed"] == [failed]


def test_weak_fresh_cue_blocks_only_its_architecture_and_smoke_cannot_pass():
    rows, config = synthetic_rows()
    for row in rows:
        if row["model"] == "conditional" and row["arm"] == "fresh" and row["cue"] == 1:
            row["cue_effect"] = .0019
    result = summary.analyze_rows(rows, config)
    assert not result["qualification"]["by_model"]["conditional"]
    assert result["qualification"]["by_model"]["recurrent"]
    assert "fresh_qualification" in result["screen"]["conditional"]["failed"]
    assert result["screen"]["recurrent"]["passed"]
    short = dict(config, episode_size=128)
    assert not summary.analyze_rows(rows, short)["eligible_development_cohort"]
    assert "complete_development_cohort" in summary.analyze_rows(rows, short)["screen"]["recurrent"]["failed"]


@pytest.mark.parametrize("fault", ("missing", "duplicate"))
def test_missing_or_duplicate_arm_is_rejected_before_qualification(fault):
    rows, config = synthetic_rows()
    if fault == "missing":
        rows.pop()
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="every arm exactly once"):
        summary.analyze_rows(rows, config)


def selective_diagnostics(arm="combined"):
    config = dict(episode_size=16, batch_size=8, updates_per_batch=2)
    weight, mode = summary.ARM_SETTINGS[arm]
    before = dict(arm=arm, current_weight=weight, mode=mode, stats=summary.new_stats())
    after = copy.deepcopy(before)
    stats = after["stats"]
    stats.update(steps=4, packets=2, training_probability_calls=8, reference_gradient_calls=4,
        objective_backward_calls=4, reference_backward_query_presentations=32,
        reference_backward_support_presentations=32, diagnostic_probability_calls=2,
        diagnostic_query_presentations=16, diagnostic_support_presentations=16)
    for name in summary.SUMMARY_NAMES:
        stats["summaries"][name] = dict(count=2 if name.startswith("finite_replay_") else 4, sum=0., min=0., max=0.)
    return before, after, config


def test_selective_work_counts_and_fixed_bounded_summaries_validate():
    before, after, config = selective_diagnostics()
    summary.check_selective(before, after, "combined", config)
    before_ref, after_ref, config = selective_diagnostics("reference")
    summary.check_selective(before_ref, after_ref, "fresh", config)


@pytest.mark.parametrize("fault,message", (
    ("extra_forward", "unmatched selective work"),
    ("extra_reference_backward", "unmatched selective work"),
    ("extra_summary_event", "incomplete bounded"),
    ("unreset", "not reset"),
    ("wrong_weight", "wrong selective"),
))
def test_selective_work_or_rule_tampering_is_rejected(fault, message):
    before, after, config = selective_diagnostics()
    if fault == "extra_forward":
        after["stats"]["diagnostic_probability_calls"] += 1
    elif fault == "extra_reference_backward":
        after["stats"]["reference_gradient_calls"] += 1
    elif fault == "extra_summary_event":
        after["stats"]["summaries"]["hd_before"]["count"] += 1
    elif fault == "unreset":
        before["stats"]["steps"] = 1
    else:
        after["current_weight"] = .9
    with pytest.raises(ValueError, match=message):
        summary.check_selective(before, after, "combined", config)


def test_unprotected_reference_may_not_rewrite_parameters():
    before, after, config = selective_diagnostics("reference")
    after["stats"].update(projection_eligible_steps=1, rewrite_steps=1)
    with pytest.raises(ValueError, match="inactive protection rule"):
        summary.check_selective(before, after, "reference", config)


def analysis_fixture(tmp_path, monkeypatch):
    root, run = tmp_path/"project", tmp_path/"run"
    root.mkdir()
    run.mkdir()
    config = dict(study="synthetic_lock_test")
    protocol = b"Fixed synthetic protocol.\n"
    content = {name: b"# synthetic source\n" for name in summary.REQUIRED_ANALYSIS}
    content["configs/selective_updates_development.json"] = json.dumps(config).encode()
    content["docs/selective_updates_protocol.md"] = protocol
    files = {}
    with zipfile.ZipFile(run/"analysis_at_lock.zip", "w") as archive:
        for name, data in content.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    (run/"analysis_lock.json").write_text(json.dumps(dict(locked_utc="2026-09-27T00:00:00+00:00", files=files,
                                                       config=config, config_sha256=summary.digest(config))))
    monkeypatch.setattr(summary, "ROOT", root)
    return root, run, config, hashlib.sha256(protocol).hexdigest()


def test_code_config_protocol_lock_rejects_postlock_script_change(tmp_path, monkeypatch):
    root, run, config, protocol_hash = analysis_fixture(tmp_path, monkeypatch)
    summary.verify_analysis_lock(run, config, protocol_hash)
    (root/summary.REQUIRED_ANALYSIS[0]).write_text("Changed after lock.\n")
    with pytest.raises(ValueError, match="differs from lock"):
        summary.verify_analysis_lock(run, config, protocol_hash)


@pytest.mark.parametrize("fault", ("config", "protocol", "zip"))
def test_code_lock_covers_configuration_protocol_and_archive_bytes(tmp_path, monkeypatch, fault):
    _, run, config, protocol_hash = analysis_fixture(tmp_path, monkeypatch)
    if fault == "config":
        config["study"] = "changed"
    elif fault == "protocol":
        protocol_hash = "0"*64
    else:
        with zipfile.ZipFile(run/"analysis_at_lock.zip", "w") as archive:
            archive.writestr("other.py", "bad")
    with pytest.raises(ValueError):
        summary.verify_analysis_lock(run, config, protocol_hash)


def test_law_hash_and_nonfinite_records_are_not_silently_accepted():
    summary.finite_tree(dict(value=[1., None, dict(other=2.)]))
    with pytest.raises(ValueError, match="nonfinite"):
        summary.finite_tree(dict(value=float("nan")))
    assert asdict(summary.stage_law(13001, 1))["active"] == (order(13001)[0],)
