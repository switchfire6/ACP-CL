"""Fixed acquisition/retention screens, feature attribution and numerical records."""

import copy
import hashlib
from itertools import permutations
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import summarize_core_residual as summary
from summarize_acquisition import paired as prior_paired


def config():
    return json.loads(Path("configs/core_residual_development.json").read_text())


def probe(value, cue=True):
    metrics = {name: .7 if "survival" in name else value for name in summary.PROBE_FIELDS}
    metrics["marginal_brier"] = value+.04
    row = dict(curve=[dict(feedback=n, metrics=dict(metrics)) for n in (0, 8, 16, 32)])
    if cue:
        row["flipped"] = dict(metrics, focus_brier=value+.004)
    return dict(replicates=[copy.deepcopy(row), copy.deepcopy(row)])


def record(seed, model, arm, phase):
    cue = tuple(permutations(range(3)))[seed % 6][0] if phase == "novel" else None
    value = .2 if phase != "return" else .22
    if phase == "novel":
        value += {"joint": 0., "separate": -.003, "fixed_features": -.001, "fresh": -.01}[arm]
    elif phase == "return" and arm == "separate":
        value += .001
    before, after = probe(value, cue is not None), probe(value, cue is not None)
    metrics = after["replicates"][0]["curve"][-1]["metrics"]
    valid = {mode: dict(valid_brier=.1, valid_survival=.7) for mode in ("0", "1")}
    return dict(seed=seed, model=model, arm=arm, phase=phase, cue=cue, before=before, after=after,
        valid_before=copy.deepcopy(valid), valid_after=valid,
        curve=[dict(arrivals=n, metrics=dict(metrics)) for n in (0, 128, 256)],
        packets=[dict(prequential_brier=.4), dict(prequential_brier=.5)],
        novel_after_return=probe(.25+(.001 if arm == "separate" else 0)) if phase == "return" else None)


def records():
    return [record(*key) for key in sorted(summary.expected_learning_keys(config()))]


@pytest.fixture(scope="module")
def result():
    return summary.analyze_records(records(), config())


def test_paired_bootstrap_preserves_previous_arithmetic_and_seed_units():
    values = [-.003, .001, -.005, -.002, .004, 0.]
    actual, prior = summary.paired(values, summary.SEEDS), prior_paired(values)
    for name in ("mean", "lower", "upper", "n", "positive", "differences"):
        assert actual[name] == prior[name]
    assert actual["n"] == 6 and actual["seeds"] == summary.SEEDS


def test_normalized_auc_and_prequential_error_are_separate_measurements():
    row = record(16001, "conditional", "joint", "novel")
    row["curve"] = [dict(arrivals=x, metrics=dict(row["curve"][0]["metrics"], focus_brier=y))
                    for x, y in ((0, .3), (64, .2), (256, .1))]
    metrics = summary.phase_metrics(row)
    assert metrics["brier_auc"] == pytest.approx((64*.25+192*.15)/256)
    assert metrics["prequential_brier"] == .45
    assert metrics["cue_effect"] == pytest.approx(.004)


def test_complete_comparison_and_separate_attribution_pass_on_constructed_data(result):
    assert result["eligible_development_cohort"]
    assert len(result["rows"]) == 120
    for model in summary.MODELS:
        assert result["screen"][model]["passed"]
        assert result["attribution"][model]["passed"]
        primary = result["comparisons"][model]["separate_minus_joint"]
        assert primary["novel"]["brier_auc"]["mean"] == pytest.approx(-.003)
        assert primary["return"]["novel_after_return_brier"]["mean"] == pytest.approx(.001)
        assert primary["novel"]["brier_auc"]["n"] == 6
        assert all(row["n"] == 2 for row in result["qualification"]["groups"]
                   if row["model"] == model and row["group"].startswith("cue_"))


@pytest.mark.parametrize("phase,metric,value,gate", (
    ("novel", "brier_auc", -.0019, "acquisition_gain"),
    ("novel", "valid_after_mode_0", .0051, "valid_old_mode_0"),
    ("novel", "valid_after_mode_1", .0051, "valid_old_mode_1"),
    ("return", "all_brier_auc", .0051, "return_prediction"),
    ("return", "novel_after_return_brier", .0051, "post_return_novel_retention"),
    ("novel", "survival_auc", -.0101, "survival_novel"),
    ("return", "survival_auc", -.0101, "survival_return"),
))
def test_each_primary_tradeoff_is_enforced_separately(result, phase, metric, value, gate):
    comparison = copy.deepcopy(result["comparisons"]["conditional"]["separate_minus_joint"])
    comparison[phase][metric]["mean"] = value
    assert summary.primary_screen(comparison, True)["failed"] == [gate]


def test_five_of_six_is_fixed_and_intervals_do_not_replace_mean_gates(result):
    comparison = copy.deepcopy(result["comparisons"]["conditional"]["separate_minus_joint"])
    metric = comparison["novel"]["brier_auc"]
    metric.update(mean=-.002, lower=-.005, upper=.001, differences=[-.0024]*5+[0.])
    result = summary.primary_screen(comparison, True)
    assert result["passed"] and result["gates"]["acquisition_gain"]["uncertainty_crosses_limit"]
    metric["differences"] = [-.004]*4+[.001]*2
    assert summary.primary_screen(comparison, True)["failed"] == ["acquisition_consistency"]


def test_attribution_neither_rescues_nor_overrides_the_primary_decision(result):
    comparison = copy.deepcopy(result["comparisons"]["conditional"]["separate_minus_fixed_features"])
    metric = comparison["novel"]["brier_auc"]
    metric.update(mean=-.001, differences=[-.0012]*5+[0.])
    assert summary.attribution_screen(comparison)["passed"]
    metric["mean"] = -.0009
    assert summary.attribution_screen(comparison)["failed"] == ["feature_adaptation_gain"]
    assert result["screen"]["conditional"]["passed"]
    metric.update(mean=-.002, differences=[-.004]*4+[.001]*2)
    assert summary.attribution_screen(comparison)["failed"] == ["feature_adaptation_consistency"]


def test_one_weak_cue_blocks_its_architecture_qualification(result):
    rows = copy.deepcopy(result["rows"])
    for row in rows:
        if row["model"] == "conditional" and row["arm"] == "fresh" and row["cue"] == 1:
            row["cue_effect"] = .0019
    qualified = summary.qualification(rows, summary.MODELS)
    assert not qualified["by_model"]["conditional"]
    assert qualified["by_model"]["recurrent"]
    comparison = result["comparisons"]["conditional"]["separate_minus_joint"]
    assert summary.primary_screen(comparison, False)["failed"] == ["fresh_qualification"]


def test_smoke_and_modified_scientific_dose_are_ineligible():
    cfg = config()
    assert summary.scientific_eligibility(cfg)
    assert summary.scientific_eligibility(dict(cfg, threads=2, workers=1, cpu_affinity=0))
    assert not summary.scientific_eligibility(dict(cfg, seeds=[16991]))
    assert not summary.scientific_eligibility(dict(cfg, episode_size=4096))
    assert not summary.scientific_eligibility(dict(cfg, residual_width=64))


@pytest.mark.parametrize("fault", ("missing", "duplicate"))
def test_incomplete_or_duplicate_phase_is_rejected(fault):
    rows = records()
    if fault == "missing":
        rows.pop()
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="exactly once"):
        summary.analyze_records(rows, config())


def test_nonfinite_curves_do_not_enter_statistics():
    row = record(16001, "conditional", "joint", "novel")
    row["curve"][1]["metrics"]["focus_brier"] = np.nan
    with pytest.raises(ValueError, match="invalid held-out"):
        summary.phase_metrics(row)


@pytest.fixture(scope="module")
def numerical_phase(tmp_path_factory):
    """Tiny engineering phase exercises real trace ordering, including return/cue probes."""
    import torch
    from acp_cl.core_residual import study
    from acp_cl.core_residual.learner import fork_core

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cfg = config()
    cfg.update(seeds=[16991], models=["conditional"], workers=1, cpu_affinity=0,
        batch_size=8, eval_size=64, probe_every=8, support_replicates=1,
        prefix_blocks=2, prefix_size=16, maintenance_size=8, episode_size=16,
        return_size=8, width=8, decoder_width=8, context_width=4,
        residual_width=4, residual_decoder_width=4, residual_context_width=4, updates_per_batch=1)
    directory = tmp_path_factory.mktemp("core_residual_numerical")
    identity = dict(engineering="numerical_reader_test")
    core = study.new_learner("conditional", 16991, cfg)
    core, _ = study.episode(cfg, identity, core, 16991, "conditional", "prefix",
                           study.prefix_phases(16991, cfg)[0], directory/"prefix")
    learner = fork_core(core, "separate", 16991, cfg)
    phase = study.phases(16991, cfg, "separate")[-1]
    _, record = study.episode(cfg, identity, learner, 16991, "conditional", "separate", phase, directory/"return")
    yield directory/"return", record, cfg, identity, phase
    torch.set_num_threads(old_threads)


def copy_numerical_phase(numerical_phase, tmp_path):
    folder, record, cfg, identity, phase = numerical_phase
    destination = tmp_path/"phase"
    shutil.copytree(folder, destination)
    return destination, copy.deepcopy(record), cfg, identity, phase


def test_real_numerical_trace_roundtrip_and_resource_scope(numerical_phase):
    folder, record, cfg, identity, phase = numerical_phase
    result = summary.verify_phase(record, folder, cfg, identity, 16991, "conditional", "separate", phase)
    assert result["calls"] == record["evaluation_files"]["calls"]
    assert summary.phase_metrics(record)["novel_after_return_brier"] >= 0
    resources = summary.resource_row(record, cfg)
    assert resources["parameters"]["core_trainable"] == 0
    assert resources["core_optimizer_bytes"] > 0
    assert resources["measurement_snapshot_bytes"] > 0
    assert resources["path_work"]["core_backward_paths"] == 0
    assert resources["prequential"]["calls"] == 1
    assert resources["total_raw_path_forward_calls"] == 2*(2+1+result["calls"])


def test_resealed_training_probability_change_is_detected(numerical_phase, tmp_path):
    folder, record, cfg, _, _ = copy_numerical_phase(numerical_phase, tmp_path)
    data = summary.read_arrays(folder/"training.npz")
    action = int(data["actions"][0, 0])
    data["probabilities"][0, 0, action] *= .9
    np.savez_compressed(folder/"training.npz", **data)
    record["training_file"]["sha256"] = summary.sha(folder/"training.npz")
    with pytest.raises(ValueError, match="prequential packet error"):
        summary.verify_training(record, folder, cfg)


@pytest.mark.parametrize("fault,message", (
    ("trace_alias", "metric/trace reference"),
    ("valid_mean", "valid-old support mean"),
    ("live_support", "live-history evaluation"),
    ("marginal", "marginal used wrong training prefix"),
))
def test_phase_metric_or_context_references_cannot_be_rewritten(numerical_phase, tmp_path, fault, message):
    folder, record, cfg, _, _ = copy_numerical_phase(numerical_phase, tmp_path)
    training = summary.verify_training(record, folder, cfg)
    if fault == "trace_alias":
        record["curve"][0]["metrics"]["trace"] = 0
    elif fault == "valid_mean":
        record["valid_after"]["0"]["valid_brier"] += .01
    elif fault == "live_support":
        record["packets"][0]["support_sha256"] = "0"*64
    else:
        record["marginal_success"][0][0] += 1
    with pytest.raises(ValueError, match=message):
        summary.verify_evaluations(record, folder, cfg, training)


def test_resealed_evaluation_metric_change_fails_raw_arithmetic(numerical_phase, tmp_path):
    folder, record, cfg, _, _ = copy_numerical_phase(numerical_phase, tmp_path)
    payload = summary.read(folder/"evaluations.json")
    payload["records"][0]["metrics"]["brier"] += .001
    summary.write_json(folder/"evaluations.json", payload)
    record["evaluation_files"]["json"]["sha256"] = summary.sha(folder/"evaluations.json")
    with pytest.raises(ValueError, match="brier differs from raw"):
        summary.verify_evaluations(record, folder, cfg, summary.verify_training(record, folder, cfg))


def test_probability_roundoff_is_preserved_and_truth_is_strict():
    eps = np.finfo(np.float32).eps
    values = np.full((2, 5, 3), 1+eps, dtype=np.float32)
    truth = np.ones_like(values, dtype=np.uint8)
    mask = np.ones(2, dtype=bool)
    assert summary.recompute_metrics(values, truth, mask, mask)["brier"] == float(eps)**2
    with pytest.raises(ValueError, match="outside float32-roundoff"):
        summary.recompute_metrics(values+eps, truth, mask, mask)
    truth[0, 0, 0] = 2
    with pytest.raises(ValueError, match="binary outcomes"):
        summary.recompute_metrics(values, truth, mask, mask)


def analysis_fixture(tmp_path, monkeypatch):
    root, run = tmp_path/"project", tmp_path/"run"
    root.mkdir()
    run.mkdir()
    cfg = dict(study="synthetic_lock")
    protocol = b"Fixed synthetic protocol.\n"
    contents = {name: b"# synthetic prospective file\n" for name in summary.REQUIRED_ANALYSIS}
    contents["docs/core_residual_protocol.md"] = protocol
    hashes = {}
    with zipfile.ZipFile(run/"analysis_at_lock.zip", "w") as archive:
        for name, content in contents.items():
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    summary.write_json(run/"analysis_lock.json", dict(config=cfg, config_sha256=summary.digest(cfg), files=hashes))
    monkeypatch.setattr(summary, "ROOT", root)
    return root, run, cfg, hashlib.sha256(protocol).hexdigest()


def test_prospective_lock_accepts_captured_programmatic_configuration(tmp_path, monkeypatch):
    _, run, cfg, protocol = analysis_fixture(tmp_path, monkeypatch)
    summary.verify_analysis_lock(run, cfg, protocol)


@pytest.mark.parametrize("fault", ("script", "config", "protocol", "zip", "missing_analysis"))
def test_prospective_code_config_protocol_archive_coverage(tmp_path, monkeypatch, fault):
    root, run, cfg, protocol = analysis_fixture(tmp_path, monkeypatch)
    if fault == "script":
        (root/summary.REQUIRED_ANALYSIS[0]).write_text("changed after lock")
    elif fault == "config":
        cfg["study"] = "changed"
    elif fault == "protocol":
        protocol = "0"*64
    elif fault == "zip":
        with zipfile.ZipFile(run/"analysis_at_lock.zip", "w") as archive:
            archive.writestr("other.py", "changed")
    else:
        lock = summary.read(run/"analysis_lock.json")
        del lock["files"][summary.REQUIRED_ANALYSIS[0]]
        summary.write_json(run/"analysis_lock.json", lock)
    with pytest.raises(ValueError):
        summary.verify_analysis_lock(run, cfg, protocol)
