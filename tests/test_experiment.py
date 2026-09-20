"""Small end-to-end tests for paired experiments, isolation, and resumption."""

import copy
import io
import json
import math

import pytest
import torch
from torch.utils.data import TensorDataset

import acp_cl.experiment as experiment
from acp_cl.data import build_stream
from acp_cl.learner import METHODS, YOKED_METHODS


@pytest.fixture
def tiny_config():
    # Permissive stability thresholds force several control transitions without
    # relying on random training dynamics; these are test settings, not priors.
    return {
        "data": {
            "dataset": "synthetic",
            "n_experiences": 2,
            "classes_per_experience": 2,
            "input_dim": 6,
            "train_per_class": 3,
            "validation_per_class": 2,
            "test_per_class": 2,
            "noise": 0.3,
        },
        "model": "mlp",
        "width": 8,
        "threads": 1,
        "batch_size": 4,
        "steps_per_experience": 3,
        "eval_every_steps": 1,
        "early_fraction": 0.2,
        "eval_split": "validation",
        "scratch_reference": False,
        "checkpoint": True,
        "replay_capacity": 8,
        "replay_batch_size": 4,
        "probe_batch_size": 4,
        "controller": {
            "monitor_interval": 1,
            "scaffold_steps": 0,
            "min_open_steps": 0,
            "stable_windows": 1,
            "closing_windows": 1,
            "stability_loss_tol": 100.0,
            "stability_drift_tol": 100.0,
            "max_replay_damage": 100.0,
            "novelty_bias": -10.0,
            "novel_windows": 1,
            "cooldown_windows": 0,
            "reopen_max_windows": 2,
            "top_k": 1,
        },
        "plasticity": {
            "lr": 0.01,
            "maturity_steps": 1,
            "recycle_interval": 1,
            "recycle_fraction": 0.125,
            "adult_recycle_fraction": 0.125,
            "consolidation_rate": 0.1,
        },
    }


@pytest.fixture(autouse=True)
def lightweight_environment(monkeypatch):
    # Exercise orchestration while keeping tests independent of repository HEAD
    # and concurrent source edits during local development.
    monkeypatch.setattr(experiment, "environment", lambda: {"test_environment": True})
    monkeypatch.setattr(experiment, "source_hash", lambda: "test-source")


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    else:
        assert first == second


def saved_learner_state(learner):
    # Serialization is intentional: torch module state_dicts may share storage
    # with their live model until saved, whereas checkpoints must be snapshots.
    target = io.BytesIO()
    torch.save(learner.state_dict(), target)
    target.seek(0)
    return torch.load(target, map_location="cpu", weights_only=False)


def make_learner(config, method, seed=19):
    stream = build_stream(config["data"], seed)
    learner = experiment.new_learner(config, stream, method, seed, torch.device("cpu"))
    return learner, stream


@pytest.mark.parametrize("method", [m for m in METHODS if m not in YOKED_METHODS])
def test_all_methods_complete_tiny_two_experience_run(tiny_config, tmp_path, method):
    stream = build_stream(tiny_config["data"], seed=19)
    result = experiment.run_one(
        tiny_config, stream, method, 19, torch.device("cpu"), tmp_path
    )
    assert result["method"] == method
    assert result["eval_split"] == "validation"
    assert result["cost"]["current_examples"] == 20
    assert len(result["learning_curves"]) == 2
    assert result["accuracy_matrix"][0][1] is None
    assert all(math.isfinite(value) for value in result["metrics"].values())
    assert 0 <= result["metrics"]["final_accuracy"] <= 1
    for curve in result["learning_curves"]:
        assert curve[0][0] == 0
        assert curve[-1][0] == 10
        assert all(math.isfinite(y) and 0 <= y <= 1 for _, y in curve)
    events = json.loads((tmp_path / f"{method}_seed19" / "events.json").read_text())["events"]
    assert all(math.isfinite(event["loss"]) and math.isfinite(event["update_norm"]) for event in events)
    if method in {"finetune", "acp_no_replay"}:
        assert result["replay_examples"] == result["replay_bytes"] == 0
        assert result["cost"]["replay_examples"] == 0
    else:
        assert result["replay_examples"] == 8
        assert result["replay_bytes"] > 0
    if method == "acp":
        observed_reopenings = sum(
            event.get("controller", {}).get("previous_phase") == "adult"
            and event.get("controller", {}).get("phase") == "reopened"
            for event in events
        )
        assert observed_reopenings > 0
        assert result["reopening_events"] == observed_reopenings


def test_er_and_acp_share_initialization_examples_and_memory_membership(tiny_config):
    er, stream = make_learner(tiny_config, "er")
    acp, _ = make_learner(tiny_config, "acp")
    assert_tree_equal(er.model.state_dict(), acp.model.state_dict())
    for experience in stream.experiences:
        raw_before = experience.train.tensors[0].clone()
        for x, y in experiment.batches(experience, tiny_config, seed=19):
            er.train_batch(x, y)
            acp.train_batch(x, y)
        torch.testing.assert_close(experience.train.tensors[0], raw_before, rtol=0, atol=0)
    assert er.cost["current_examples"] == acp.cost["current_examples"] == 20
    er_buffer, acp_buffer = er.buffer.state_dict(), acp.buffer.state_dict()
    assert_tree_equal(er_buffer, acp_buffer)
    assert er.buffer.num_seen == acp.buffer.num_seen == 20


def test_simultaneous_adjacent_recycling_keeps_all_newborn_outputs_zero(tiny_config):
    learner, stream = make_learner(tiny_config, "er_recycle")
    x, y = next(experiment.batches(stream.experiences[0], tiny_config, seed=19))
    event = learner.train_batch(x, y)
    assert event["recycled"] >= 2
    for unit in learner.model.recyclable_units():
        newborn = learner.engine.unit_state[unit.name]["age"] == 0
        assert newborn.any()
        # Adjacent incoming-row resets must not overwrite zeroed outgoing
        # columns from an upstream population recycled in the same update.
        assert torch.count_nonzero(unit.outgoing.weight[:, newborn]) == 0
        parameter_names = {id(value): name for name, value in learner.model.named_parameters()}
        for incoming in (unit.incoming.weight, unit.incoming.bias):
            state = learner.engine.state[parameter_names[id(incoming)]]
            for field in ("momentum", "path", "c", "importance", "age"):
                assert torch.count_nonzero(state[field][newborn]) == 0
            for field in ("anchor", "start"):
                torch.testing.assert_close(state[field][newborn], incoming[newborn], rtol=0, atol=0)
        outgoing_state = learner.engine.state[parameter_names[id(unit.outgoing.weight)]]
        for field in ("momentum", "path", "anchor", "start"):
            assert torch.count_nonzero(outgoing_state[field][:, newborn]) == 0


def test_evaluation_does_not_update_model_controller_monitor_or_training_rngs(tiny_config):
    learner, stream = make_learner(tiny_config, "acp")
    x, y = next(experiment.batches(stream.experiences[0], tiny_config, 19))
    learner.train_batch(x, y)
    before = copy.deepcopy(learner.state_dict())
    global_rng = torch.get_rng_state().clone()
    for experience in stream.experiences:
        assert 0 <= learner.accuracy(experience.validation) <= 1
        # Even radically different evaluation labels must not become training
        # feedback: the controller observes only training/paired replay signals.
        eval_x, eval_y = experience.validation.tensors
        learner.accuracy(TensorDataset(eval_x * 100, (eval_y + 1) % stream.num_classes))
    after = copy.deepcopy(learner.state_dict())
    assert after["cost"]["evaluation_examples"] - before["cost"]["evaluation_examples"] == 16
    after["cost"]["evaluation_examples"] = before["cost"]["evaluation_examples"]
    assert_tree_equal(before, after)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


def test_validation_run_never_reads_test_examples(tiny_config, tmp_path):
    class ForbiddenTestSet:
        @property
        def tensors(self):
            raise AssertionError("development evaluation accessed held-out test examples")

        def __len__(self):
            raise AssertionError("development evaluation accessed held-out test examples")

    stream = build_stream(tiny_config["data"], seed=19)
    for experience in stream.experiences:
        experience.test = ForbiddenTestSet()
    result = experiment.run_one(
        tiny_config, stream, "acp", 19, torch.device("cpu"), tmp_path
    )
    assert result["eval_split"] == "validation"


@pytest.mark.parametrize("method", ["er", "derpp", "acp", "acp_random_recycling"])
def test_learner_checkpoint_reproduces_next_update_exactly(tiny_config, method):
    original, stream = make_learner(tiny_config, method)
    batches = list(experiment.batches(stream.experiences[0], tiny_config, seed=19))
    for x, y in batches:
        original.train_batch(x, y)
    checkpoint = saved_learner_state(original)
    restored, _ = make_learner(tiny_config, method)
    restored.load_state_dict(checkpoint)
    next_x, next_y = next(experiment.batches(stream.experiences[1], tiny_config, seed=19))
    event_a = original.train_batch(next_x, next_y)
    event_b = restored.train_batch(next_x, next_y)
    assert_tree_equal(event_a, event_b)
    assert_tree_equal(original.state_dict(), restored.state_dict())


def test_learner_checkpoint_rejects_changed_hyperparameters(tiny_config):
    original, _ = make_learner(tiny_config, "acp")
    checkpoint = saved_learner_state(original)
    changed_config = copy.deepcopy(tiny_config)
    changed_config["plasticity"]["lr"] *= 2
    different, _ = make_learner(changed_config, "acp")
    with pytest.raises(ValueError, match="config|setting|hyperparameter"):
        different.load_state_dict(checkpoint)


def test_suite_resume_from_experience_checkpoint_matches_uninterrupted(tiny_config, tmp_path, monkeypatch):
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    uninterrupted = experiment.run_suite(
        tiny_config, output=uninterrupted_dir, seeds=[19], methods=["acp"], device_name="cpu"
    )[0]
    real_train_experience = experiment.train_experience

    def stop_before_second_experience(learner, experience, config, seed):
        if experience.index == 1:
            raise RuntimeError("simulated process interruption")
        return real_train_experience(learner, experience, config, seed)

    monkeypatch.setattr(experiment, "train_experience", stop_before_second_experience)
    with pytest.raises(RuntimeError, match="simulated"):
        experiment.run_suite(
            tiny_config, output=resumed_dir, seeds=[19], methods=["acp"], device_name="cpu"
        )
    checkpoint_path = resumed_dir / "acp_seed19" / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["completed"] == 1
    monkeypatch.setattr(experiment, "train_experience", real_train_experience)
    resumed = experiment.run_suite(
        tiny_config, output=resumed_dir, seeds=[19], methods=["acp"], device_name="cpu", resume=True
    )[0]
    for field in (
        "metrics", "accuracy_matrix", "learning_curves", "early_auc", "cost",
        "phase_monitor_counts", "reopening_events", "diagnostics", "replay_bytes",
    ):
        assert_tree_equal(uninterrupted[field], resumed[field])
    whole_state = torch.load(
        uninterrupted_dir / "acp_seed19" / "checkpoint.pt", map_location="cpu", weights_only=False
    )["learner"]
    resumed_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["learner"]
    assert_tree_equal(whole_state, resumed_state)
    # A completed resume is a cache read, not another training pass.
    cached = experiment.run_suite(
        tiny_config, output=resumed_dir, seeds=[19], methods=["acp"], device_name="cpu", resume=True
    )[0]
    assert_tree_equal(resumed, cached)


def test_suite_rejects_output_reuse_and_changed_configuration(tiny_config, tmp_path, monkeypatch):
    experiment.run_suite(
        tiny_config, output=tmp_path, seeds=[19], methods=["er"], device_name="cpu"
    )
    with pytest.raises(FileExistsError):
        experiment.run_suite(
            tiny_config, output=tmp_path, seeds=[19], methods=["er"], device_name="cpu"
        )
    changed = copy.deepcopy(tiny_config)
    changed["width"] = 9
    with pytest.raises(ValueError, match="config/source changed"):
        experiment.run_suite(
            changed, output=tmp_path, seeds=[19], methods=["er"], device_name="cpu", resume=True
        )
    monkeypatch.setattr(experiment, "source_hash", lambda: "different-source")
    with pytest.raises(ValueError, match="config/source changed"):
        experiment.run_suite(
            tiny_config, output=tmp_path, seeds=[19], methods=["er"], device_name="cpu", resume=True
        )


def test_completed_result_resume_rejects_wrong_seed(tiny_config, tmp_path):
    experiment.run_suite(
        tiny_config, output=tmp_path, seeds=[19], methods=["er"], device_name="cpu"
    )
    result_path = tmp_path / "er_seed19" / "result.json"
    result = json.loads(result_path.read_text())
    result["seed"] = 20
    experiment.write_json(result_path, result)
    with pytest.raises(ValueError, match="mismatch"):
        experiment.run_suite(
            tiny_config, output=tmp_path, seeds=[19], methods=["er"], device_name="cpu", resume=True
        )


@pytest.mark.parametrize("mismatch", ["config", "source", "class_order", "duplicate_seed"])
def test_analysis_rejects_mismatched_or_duplicate_paired_runs(tiny_config, tmp_path, monkeypatch, mismatch):
    from acp_cl import analysis

    monkeypatch.setattr(analysis, "plot_overview", lambda *args: None)
    experiment.run_suite(
        tiny_config, output=tmp_path, seeds=[19], methods=["er", "acp"], device_name="cpu"
    )
    result_path = tmp_path / "er_seed19" / "result.json"
    result = json.loads(result_path.read_text())
    if mismatch == "config":
        result["config_sha256"] = "different-config"
    elif mismatch == "source":
        result["source_sha256"] = "different-source"
    elif mismatch == "class_order":
        result["class_order"] = result["class_order"][::-1]
    else:
        result_path = tmp_path / "duplicate" / "result.json"
    experiment.write_json(result_path, result)
    with pytest.raises(ValueError, match="pool|class orders|duplicate seed"):
        analysis.analyze(tmp_path)


@pytest.mark.parametrize("location", ["manifest", "result", "checkpoint", "scratch"])
def test_resume_rejects_changed_execution_device(tiny_config, tmp_path, location):
    if location == "scratch":
        tiny_config["scratch_reference"] = True
    experiment.run_suite(
        tiny_config, output=tmp_path, seeds=[19], methods=["er"], device_name="cpu"
    )
    run_directory = tmp_path / "er_seed19"
    if location == "checkpoint":
        path = run_directory / "checkpoint.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        saved["execution_device"] = "cuda"
        torch.save(saved, path)
        (run_directory / "result.json").unlink()
    else:
        path = {
            "manifest": tmp_path / "manifest.json",
            "result": run_directory / "result.json",
            "scratch": tmp_path / "scratch_seed19.json",
        }[location]
        saved = json.loads(path.read_text())
        saved["execution_device"] = "cuda"
        experiment.write_json(path, saved)
    with pytest.raises(ValueError, match="device|mismatch|config/source changed"):
        experiment.run_suite(
            tiny_config, output=tmp_path, seeds=[19], methods=["er"], device_name="cpu", resume=True
        )


@pytest.mark.parametrize("mismatch", ["execution_device", "torch", "numpy", "gpu"])
def test_analysis_rejects_mixed_execution_regimes(tiny_config, tmp_path, monkeypatch, mismatch):
    from acp_cl import analysis

    monkeypatch.setattr(analysis, "plot_overview", lambda *args: None)
    experiment.run_suite(
        tiny_config, output=tmp_path, seeds=[19], methods=["er", "acp"], device_name="cpu"
    )
    for method in ("er", "acp"):
        path = tmp_path / f"{method}_seed19" / "result.json"
        result = json.loads(path.read_text())
        result["execution_device"] = "cuda" if mismatch == "gpu" else "cpu"
        result["environment"].update({"torch": "2.8.0", "numpy": "2.3.3", "gpu": "same GPU"})
        if method == "er":
            if mismatch == "execution_device":
                result["execution_device"] = "cuda"
            else:
                result["environment"][mismatch] = "different runtime or hardware"
        experiment.write_json(path, result)
    with pytest.raises(ValueError, match="pool|runtime|device|execution|hardware"):
        analysis.analyze(tmp_path)
