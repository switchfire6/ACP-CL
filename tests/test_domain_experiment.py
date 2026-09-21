"""Fixed-label single-pass arrivals and snapshot-local shared-pool evaluation."""

import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from acp_cl import experiment
from acp_cl.data import Experience, Stream
from test_experiment import assert_tree_equal, lightweight_environment

__all__ = ["lightweight_environment"]


@pytest.fixture
def config():
    return {
        "data": {"dataset": "synthetic"}, "model": "mlp", "width": 8, "threads": 1,
        "batch_size": 4, "epochs": 1, "single_pass": True,
        "evaluation_cache_shared_pools": True, "eval_every_steps": 2,
        "early_fraction": 0.2, "eval_split": "validation", "scratch_reference": False,
        "checkpoint": True, "replay_capacity": 16, "replay_batch_size": 4,
        "probe_batch_size": 4, "retention_eval_every_experiences": 1,
        "progress_every_experiences": 100, "controller": {"monitor_interval": 2},
        "plasticity": {"lr": 0.001, "maturity_steps": 1000},
    }


def domain_stream(experiences=6, train_size=9, pools=3, *, distinct_equal=False):
    generator = torch.Generator().manual_seed(873)
    validations = [TensorDataset(torch.randn(10, 6, generator=generator), torch.arange(10))
                   for _ in range(pools)]
    empty = TensorDataset(torch.empty(0, 6), torch.empty(0, dtype=torch.long))
    members, ids_by_experience = [], []
    for index in range(experiences):
        ids = list(range(index*train_size, (index+1)*train_size))
        x = torch.randn(train_size, 6, generator=generator)
        # Evaluator-only test encoding independently identifies the actual raw
        # samples delivered to train_batch; the learner never receives ID metadata.
        x[:, 0] = torch.tensor(ids)
        train = TensorDataset(x, torch.tensor(ids) % 10)
        validation = validations[index % pools]
        if distinct_equal:
            validation = TensorDataset(*validations[0].tensors)
        members.append(Experience(index, tuple(range(10)), train, validation, empty))
        ids_by_experience.append(ids)
    return Stream(members, (6,), 10, list(range(10)), {
        "current_arrival_ids_by_experience": ids_by_experience,
        "domain_order": [index % pools for index in range(experiences)],
        "signal_change_flags": [False] + [True]*(experiences-1),
    })


def checkpoint(directory, method="er", seed=19):
    return torch.load(directory / f"{method}_seed{seed}" / "checkpoint.pt",
                      weights_only=False, map_location="cpu")


def test_snapshot_cache_matches_uncached_results_training_state_and_all_rngs(config, tmp_path):
    uncached_config = dict(config, evaluation_cache_shared_pools=False)
    whole = experiment.run_one(uncached_config, domain_stream(), "er", 19, torch.device("cpu"), tmp_path / "uncached")
    uncached_rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    cached = experiment.run_one(config, domain_stream(), "er", 19, torch.device("cpu"), tmp_path / "cached")
    cached_rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    for field in ("metrics", "accuracy_matrix", "learning_curves", "early_auc", "diagnostics",
                  "allocation_summary", "current_arrival_audit", "allocation_trace_sha256"):
        assert_tree_equal(cached[field], whole[field])
    first = checkpoint(tmp_path / "uncached")["learner"]
    second = checkpoint(tmp_path / "cached")["learner"]
    # Six retained-pool calls are avoided, each covering ten validation images.
    assert first["cost"]["evaluation_examples"]-second["cost"]["evaluation_examples"] == 60
    first["cost"]["evaluation_examples"] = second["cost"]["evaluation_examples"]
    assert_tree_equal(first, second)
    assert_tree_equal(uncached_rng[0], cached_rng[0])
    np.testing.assert_equal(uncached_rng[1], cached_rng[1])
    assert uncached_rng[2] == cached_rng[2]
    assert cached["evaluation_schedule"]["shared_pool_cache"]["retention_cache_hits"] == 6
    assert cached["evaluation_schedule"]["shared_pool_cache"]["retention_accuracy_calls"] == 9
    assert "shared_pool_cache" not in whole["evaluation_schedule"]


def test_cache_reuses_current_diagonal_but_never_a_previous_training_snapshot(config, tmp_path, monkeypatch):
    original = experiment.Learner.accuracy
    observations = []

    def changing_accuracy(learner, dataset, *args, **kwargs):
        original(learner, dataset, *args, **kwargs)
        observations.append((learner.step_number, id(dataset)))
        return learner.step_number / 100

    monkeypatch.setattr(experiment.Learner, "accuracy", changing_accuracy)
    stream = domain_stream(experiences=4, pools=1)
    result = experiment.run_one(config, stream, "er", 19, torch.device("cpu"), tmp_path)
    # Each experience has three updates; all historical columns must reflect
    # this latest snapshot, even though every column refers to one shared object.
    for index, row in enumerate(result["accuracy_matrix"]):
        assert row[:index+1] == [3*(index+1)/100]*(index+1)
    assert len(observations) == 4*3  # Initial, update 2, and update 3 for each curve.
    stats = result["evaluation_schedule"]["shared_pool_cache"]
    assert stats["retention_accuracy_calls"] == 0
    assert stats["retention_cache_hits"] == 6


def test_cache_uses_dataset_object_identity_not_equal_contents_or_tensor_storage(config, tmp_path):
    stream = domain_stream(distinct_equal=True)
    result = experiment.run_one(config, stream, "er", 19, torch.device("cpu"), tmp_path)
    stats = result["evaluation_schedule"]["shared_pool_cache"]
    assert stats["retention_cache_hits"] == 0
    assert stats["retention_accuracy_calls"] == 15


def test_thirty_1500_image_experiences_are_1410_steps_with_28_image_final_batches(config):
    config["batch_size"] = 32
    stream = domain_stream(experiences=30, train_size=1500)
    presented, steps = [], 0
    before = torch.get_rng_state().clone()
    for experience in stream.experiences:
        batches = list(experiment.batches(experience, config, 91, include_indices=True))
        assert len(batches) == 47
        assert [len(batch[1]) for batch in batches] == [32]*46 + [28]
        for x, y, positions in batches:
            actual_ids = x[:, 0].long().tolist()
            source_ids = stream.metadata["current_arrival_ids_by_experience"][experience.index]
            assert actual_ids == [source_ids[position] for position in positions.tolist()]
            assert torch.equal(y, x[:, 0].long() % 10)
            presented.extend(actual_ids)
            steps += 1
    assert steps == 1410
    assert len(presented) == len(set(presented)) == 45000
    assert set(presented) == set(range(45000))
    assert_tree_equal(torch.get_rng_state(), before)
    planned, _ = experiment._single_pass_plan(stream, config, 91)
    assert [value for row in planned for value in row["ordered_base_image_ids"]] == presented


def test_actual_current_id_artifact_is_method_independent_and_matches_train_batch(config, tmp_path, monkeypatch):
    observed = {"er": [], "finetune": []}
    original = experiment.Learner.train_batch

    def record(learner, x, y):
        observed[learner.method].extend(x[:, 0].long().tolist())
        return original(learner, x, y)

    monkeypatch.setattr(experiment.Learner, "train_batch", record)
    results = []
    for method in ("er", "finetune"):
        result = experiment.run_one(config, domain_stream(), method, 19, torch.device("cpu"), tmp_path)
        results.append(result)
        raw = (tmp_path / f"{method}_seed19/current_arrivals.json").read_bytes()
        artifact = json.loads(raw)
        ids = [value for row in artifact["experiences"] for value in row["ordered_base_image_ids"]]
        assert ids == observed[method]
        assert len(ids) == len(set(ids)) == 54
        expected_hash = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
        assert artifact["ordered_arrival_ids_sha256"] == expected_hash
        assert result["current_arrival_audit"]["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
        assert result["current_arrival_audit"]["reservoir_current_arrivals"] == result["cost"]["current_examples"] == 54
        saved = checkpoint(tmp_path, method)
        assert saved["current_arrivals"] == artifact
        assert "current_arrivals" not in saved["learner"]
    assert observed["er"] == observed["finetune"]
    assert results[0]["current_arrival_audit"] == results[1]["current_arrival_audit"]


@pytest.mark.parametrize("field,value", [
    ("epochs", 2), ("epochs", True), ("steps_per_experience", 47),
    ("steps_per_experience", None), ("single_pass", 1),
    ("evaluation_cache_shared_pools", "true"),
])
def test_single_pass_and_cache_options_reject_ambiguous_or_repeated_exposure(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        experiment.validate_config(config)


@pytest.mark.parametrize("problem", ["missing", "wrong_length", "duplicate", "wrong_type", "negative", "hash_mismatch"])
def test_single_pass_id_metadata_is_checked_before_learner_construction(config, tmp_path, monkeypatch, problem):
    stream = domain_stream()
    ids = stream.metadata["current_arrival_ids_by_experience"]
    if problem == "missing":
        del stream.metadata["current_arrival_ids_by_experience"]
    elif problem == "wrong_length":
        ids[0].pop()
    elif problem == "duplicate":
        ids[1][0] = ids[0][0]
    elif problem == "wrong_type":
        ids[0][0] = True
    elif problem == "negative":
        ids[0][0] = -1
    else:
        stream.metadata["current_arrival_id_hashes_by_experience"] = ["0"*64]*len(ids)

    def forbidden(*args, **kwargs):
        raise AssertionError("learner constructed before invalid ID metadata was rejected")

    monkeypatch.setattr(experiment, "new_learner", forbidden)
    with pytest.raises(ValueError, match="single_pass"):
        experiment.run_one(config, stream, "er", 19, torch.device("cpu"), tmp_path)
    assert not (tmp_path / "er_seed19").exists()


def test_cached_single_pass_checkpoint_resume_is_exact(config, tmp_path, monkeypatch):
    complete_dir, resumed_dir = tmp_path / "complete", tmp_path / "resumed"
    complete = experiment.run_one(config, domain_stream(), "er", 19, torch.device("cpu"), complete_dir)
    original = experiment.train_experience

    def interrupted(learner, experience, config, seed, **kwargs):
        if experience.index == 3:
            raise RuntimeError("intentional interruption")
        return original(learner, experience, config, seed, **kwargs)

    monkeypatch.setattr(experiment, "train_experience", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        experiment.run_one(config, domain_stream(), "er", 19, torch.device("cpu"), resumed_dir)
    partial = checkpoint(resumed_dir)
    assert partial["completed"] == partial["current_arrivals"]["completed_experiences"] == 3
    assert partial["current_arrivals"]["current_examples"] == 27
    monkeypatch.setattr(experiment, "train_experience", original)
    resumed = experiment.run_one(config, domain_stream(), "er", 19, torch.device("cpu"), resumed_dir, resume=True)
    for field in ("metrics", "accuracy_matrix", "learning_curves", "early_auc", "cost", "diagnostics",
                  "allocation_summary", "current_arrival_audit", "evaluation_schedule"):
        assert_tree_equal(resumed[field], complete[field])
    assert_tree_equal(checkpoint(complete_dir)["learner"], checkpoint(resumed_dir)["learner"])
    assert_tree_equal(checkpoint(complete_dir)["current_arrivals"], checkpoint(resumed_dir)["current_arrivals"])
    assert experiment.run_one(config, domain_stream(), "er", 19, torch.device("cpu"), resumed_dir, resume=True) == resumed


def test_resume_rejects_changed_arrival_ids_before_restoring_learner(config, tmp_path, monkeypatch):
    experiment.run_one(config, domain_stream(), "er", 19, torch.device("cpu"), tmp_path)
    (tmp_path / "er_seed19/result.json").unlink()
    stream = domain_stream()
    stream.metadata["current_arrival_ids_by_experience"][0].reverse()

    def forbidden(*args, **kwargs):
        raise AssertionError("learner restored from incompatible arrival provenance")

    monkeypatch.setattr(experiment, "new_learner", forbidden)
    with pytest.raises(ValueError, match="current-arrival audit mismatch"):
        experiment.run_one(config, stream, "er", 19, torch.device("cpu"), tmp_path, resume=True)


def test_single_pass_detects_repeated_reservoir_current_arrivals(config, tmp_path, monkeypatch):
    original = experiment.Learner.train_batch

    def corrupted(learner, x, y):
        event = original(learner, x, y)
        learner.buffer.num_seen += 1
        return event

    monkeypatch.setattr(experiment.Learner, "train_batch", corrupted)
    with pytest.raises(ValueError, match="actual current exposure/replay arrivals"):
        experiment.run_one(config, domain_stream(), "er", 19, torch.device("cpu"), tmp_path)
    assert not (tmp_path / "er_seed19/result.json").exists()


def test_legacy_and_explicitly_disabled_options_preserve_behavior(config, tmp_path):
    legacy = copy.deepcopy(config)
    legacy.pop("single_pass")
    legacy.pop("evaluation_cache_shared_pools")
    legacy["epochs"] = 2
    disabled = dict(legacy, single_pass=False, evaluation_cache_shared_pools=False)
    first = experiment.run_one(legacy, domain_stream(experiences=2), "er", 19, torch.device("cpu"), tmp_path / "old")
    second = experiment.run_one(disabled, domain_stream(experiences=2), "er", 19, torch.device("cpu"), tmp_path / "disabled")
    for field in ("learning_curves", "cost", "metrics", "accuracy_matrix", "evaluation_schedule"):
        assert_tree_equal(first[field], second[field])
    assert first["cost"]["current_examples"] == 36
    assert "current_arrival_audit" not in first and "current_arrival_audit" not in second
    assert_tree_equal(checkpoint(tmp_path / "old")["learner"], checkpoint(tmp_path / "disabled")["learner"])
