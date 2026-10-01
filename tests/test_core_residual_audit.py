"""Independent fork/state checks and causal trace audit with resealed corruption."""

import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_core_residual import (
    advance_reference, audit_job, check_learning_state, check_phase, file_hash,
    read_arrays, verify_fork,
)
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.core_residual import study
from acp_cl.core_residual.design import counts, phases, prefix_phases
from acp_cl.core_residual.evaluation import array_hash, recompute_metrics
from acp_cl.core_residual.learner import CoreResidualLearner, fork_core, learner_signature
from acp_cl.predictive_value.mechanism import clone_exact
from acp_cl.predictive_value.study import load, read_json, save
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import new_learner, write_json


def small_config():
    config = read_json("configs/core_residual_smoke.json")
    config.update(prefix_size=16, episode_size=32, maintenance_size=16, return_size=16,
                  probe_every=16, workers=1, cpu_affinity=0)
    return config


@pytest.fixture(params=("conditional", "recurrent"))
def donor(request):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = small_config()
    learner = new_learner(request.param, 16991, config)
    for index in range(3):
        learner.train(AcquisitionWorld().experience(Law(index % 2), config["batch_size"], 750 + index))
    return learner, config


@pytest.mark.parametrize("arm", ("joint", "separate", "fixed_features"))
def test_independent_fork_checks_complete_state_and_never_calls_fork_implementation(donor, arm, monkeypatch):
    from acp_cl.core_residual import learner as module

    parent, config = donor
    fork = fork_core(parent, arm, 16991, config)
    before = learner_signature(parent), learner_signature(fork)

    def forbidden(*args, **kwargs):
        raise AssertionError("audit called experimental fork construction")

    monkeypatch.setattr(module, "fork_core", forbidden)
    verify_fork(parent, fork, arm, 16991, config)
    assert before == (learner_signature(parent), learner_signature(fork))


@pytest.mark.parametrize("fault", ("core_weight", "core_moment", "core_gradient", "residual_weight",
                                   "memory_rng", "cost", "residual_adam", "seed"))
def test_fork_audit_rejects_state_changes_independently_of_recorded_signatures(donor, fault):
    parent, config = donor
    fork = fork_core(parent, "separate", 16991, config)
    if fault == "core_weight":
        with torch.no_grad():
            next(fork.model.core.parameters()).add_(.1)
    elif fault == "core_moment":
        parameter = next(fork.model.core.parameters())
        fork.optimizer.state[parameter]["exp_avg"].add_(.1)
    elif fault == "core_gradient":
        next(fork.model.core.parameters()).grad = None
    elif fault == "residual_weight":
        with torch.no_grad():
            next(fork.model.residual.parameters()).add_(.1)
    elif fault == "memory_rng":
        fork.memory.sampling_rng.integers(100)
    elif fault == "cost":
        fork.cost["arrivals"] += 1
    elif fault == "residual_adam":
        fork.optimizer.state[next(fork.model.residual.parameters())] = {"step": torch.tensor(1.)}
    elif fault == "seed":
        fork.residual_seed += 1
    with pytest.raises(ValueError):
        verify_fork(parent, fork, "separate", 16991, config)


@pytest.mark.parametrize("arm", ("joint", "separate", "fixed_features"))
def test_optimizer_audit_distinguishes_prefix_core_and_residual_steps(donor, arm):
    parent, config = donor
    learner = fork_core(parent, arm, 16991, config)
    for index in range(2):
        learner.train(AcquisitionWorld().experience(Law(0, (0,)), config["batch_size"], 780 + index))
    check_learning_state(learner, config, 5, 2, 3 * config["updates_per_batch"])
    damaged = clone_exact(learner)
    parameter = next(p for p in damaged.model.residual.parameters() if p.requires_grad)
    damaged.optimizer.state[parameter]["step"].add_(1)
    with pytest.raises(ValueError, match="step count"):
        check_learning_state(damaged, config, 5, 2, 3 * config["updates_per_batch"])


@pytest.mark.parametrize("fault", ("core_weight", "core_adam", "fixed_encoder", "ownership", "work"))
def test_frozen_path_and_work_tampering_are_detected(donor, fault):
    parent, config = donor
    learner = fork_core(parent, "fixed_features", 16991, config)
    learner.train(AcquisitionWorld().experience(Law(0, (0,)), config["batch_size"], 783))
    if fault == "core_weight":
        with torch.no_grad():
            next(learner.model.core.parameters()).add_(.1)
    elif fault == "core_adam":
        parameter = next(learner.model.core.parameters())
        learner.optimizer.state[parameter]["exp_avg_sq"].add_(.1)
    elif fault == "fixed_encoder":
        with torch.no_grad():
            next(learner.model.residual.frame.parameters()).add_(.1)
    elif fault == "ownership":
        learner.optimizer.param_groups[0]["params"][0] = torch.nn.Parameter(torch.zeros(1))
    elif fault == "work":
        learner.path_work["core_backward_paths"] += 1
    with pytest.raises(ValueError):
        check_learning_state(learner, config, 4, 1, 3 * config["updates_per_batch"])


def test_independent_uniform_reservoir_reconstruction_matches_actual_draws(donor):
    parent, config = donor
    reference = copy.deepcopy(parent.memory)
    history = parent.history
    for index in range(8):
        data = AcquisitionWorld().experience(Law(index % 2), config["batch_size"], 800 + index)
        expected = study.expected_replay_ids(parent, config["updates_per_batch"])
        actual = advance_reference(reference, history, data, config["updates_per_batch"])
        parent.train(data)
        assert actual == expected
        assert memory_state(parent.memory) == memory_state(reference)
        history = data


@pytest.fixture(scope="module")
def completed_jobs(tmp_path_factory):
    config = small_config()
    root = tmp_path_factory.mktemp("core_residual_audit")
    identity = {"engineering": "unlocked-core-residual-audit-fixture"}
    for model in config["models"]:
        study.run_job(config, identity, config["seeds"][0], model, root)
    return root, config, identity


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_complete_tensor_audit_recomputes_declared_calls_without_training_or_file_mutation(completed_jobs, model, monkeypatch):
    root, config, identity = completed_jobs
    seed = config["seeds"][0]
    directory = root / f"{model}_{seed}"
    before = {path.relative_to(directory).as_posix(): file_hash(path)
              for path in directory.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("tensor audit must not train the ordinary stream")

    monkeypatch.setattr(CoreResidualLearner, "train", forbidden)
    result = audit_job(directory, config, identity, seed, model)
    expected = counts({**config, "models": [model]})
    assert result["ordinary_packets"] == expected["training_packets"]
    assert result["before_checkpoints"] == result["final_checkpoints"] == expected["total_phases"]
    assert result["first_prequential_predictions_recomputed"] == expected["total_phases"]
    assert result["forks_verified"] == 3 and result["fresh_initializations_verified"] == 1
    assert result["evaluation_calls"] > result["evaluation_prediction_calls_recomputed"]
    assert result["intermediate_evaluation_calls_arithmetic_only"] == 4
    assert all(row["causal_stream_and_replay_verified"] and row["exact_full_state_chain_verified"]
               and row["trainability_adam_and_work_verified"] for row in result["phases"])
    after = {path.relative_to(directory).as_posix(): file_hash(path)
             for path in directory.rglob("*") if path.is_file()}
    assert after == before


def copy_novel(completed_jobs, tmp_path):
    root, config, identity = completed_jobs
    seed, model, arm = config["seeds"][0], "conditional", "separate"
    source = root / f"{model}_{seed}" / arm
    directory = tmp_path / "novel"
    shutil.copytree(source / "novel", directory)
    previous = load(source / "maintenance" / "checkpoint.pt", identity)["learner"]
    phase = phases(seed, config, arm)[1]
    return directory, config, identity, seed, model, arm, phase, previous


def reseal_phase(directory, identity, record):
    payload = load(directory / "checkpoint.pt", identity)
    metadata = read_json(directory / "evaluations.json")
    for name, suffix in (("json", "json"), ("arrays", "npz")):
        record["evaluation_files"][name]["sha256"] = file_hash(directory / f"evaluations.{suffix}")
    record["training_file"]["sha256"] = file_hash(directory / "training.npz")
    record["artifact_hashes"] = {name: file_hash(directory / name) for name in record["artifact_hashes"]}
    payload["record"] = record
    save(directory / "checkpoint.pt", payload)
    write_json(directory / "result.json", record)
    return metadata


@pytest.mark.parametrize("fault", ("replay_ids", "training_outcome", "first_prediction", "truth",
                                   "support", "cue_flag", "evaluation_prediction", "marginal"))
def test_resealed_raw_corruptions_fail_physical_or_causal_checks(completed_jobs, tmp_path, fault):
    directory, config, identity, seed, model, arm, phase, previous = copy_novel(completed_jobs, tmp_path)
    record = read_json(directory / "result.json")
    training = read_arrays(directory / "training.npz")
    metadata = read_json(directory / "evaluations.json")
    arrays = read_arrays(directory / "evaluations.npz")
    if fault == "replay_ids":
        record["packets"][0]["replay_ids"][0] = -1
    elif fault == "training_outcome":
        training["outcomes"][0, 0, 0] ^= 1
    elif fault == "first_prediction":
        training["probabilities"][0] = .4
        labels = training["outcomes"][0].astype(np.float64)
        # Match the exact stored float32 proposal when updating its arithmetic.
        selected = training["probabilities"][0][np.arange(config["batch_size"]), training["actions"][0]]
        record["packets"][0]["prequential_brier"] = float(np.square(selected.astype(np.float64) - labels).mean())
        record["prequential_brier"] = float(np.mean([row["prequential_brier"] for row in record["packets"]]))
    else:
        index = 0
        if fault == "cue_flag":
            index = next(i for i, row in enumerate(metadata["records"]) if row["flipped"])
            metadata["records"][index]["flipped"] = False
        elif fault == "truth":
            arrays["e0_truth"][0, 0, 0] ^= 1
        elif fault == "support":
            arrays["e0_support_actions"][0] = (arrays["e0_support_actions"][0] + 1) % 5
        elif fault == "evaluation_prediction":
            arrays["e0_probabilities"][:] = .4
        elif fault == "marginal":
            index = next(i for i, row in enumerate(metadata["records"]) if row["marginal"] is not None)
            metadata["records"][index]["marginal"][0][0] = .123
        row = metadata["records"][index]
        row["array_sha256"] = {name: array_hash(arrays[f"e{index}_{name}"]) for name in row["array_sha256"]}
        row["metrics"] = recompute_metrics(arrays[f"e{index}_probabilities"], arrays[f"e{index}_truth"],
            arrays[f"e{index}_affected"], arrays[f"e{index}_valid"], row["marginal"])
    study.save_arrays(directory / "training.npz", **training)
    study.save_arrays(directory / "evaluations.npz", **arrays)
    write_json(directory / "evaluations.json", metadata)
    reseal_phase(directory, identity, record)
    prefix_packets = config["prefix_blocks"] * config["prefix_size"] // config["batch_size"]
    maintenance_packets = config["maintenance_size"] // config["batch_size"]
    with pytest.raises(ValueError):
        check_phase(directory, config, identity, seed, model, arm, phase, previous,
                    prefix_packets + maintenance_packets, maintenance_packets,
                    prefix_packets * config["updates_per_batch"])


def test_full_state_chain_rejects_a_valid_but_unrelated_predecessor(completed_jobs, tmp_path):
    directory, config, identity, seed, model, arm, phase, previous = copy_novel(completed_jobs, tmp_path)
    previous.memory.sampling_rng.integers(10)
    with pytest.raises(ValueError, match="preceding full learner"):
        check_phase(directory, config, identity, seed, model, arm, phase, previous, 6, 2, 4)


def test_audit_job_rejects_missing_phase_or_job_hash_coverage(completed_jobs, tmp_path):
    root, config, identity = completed_jobs
    seed, model = config["seeds"][0], "conditional"
    directory = tmp_path / f"{model}_{seed}"
    shutil.copytree(root / directory.name, directory)
    record = read_json(directory / "result.json")
    record["phase_hashes"].pop("fresh/novel")
    write_json(directory / "result.json", record)
    with pytest.raises(ValueError, match="phase coverage"):
        audit_job(directory, config, identity, seed, model)


def test_first_prefix_is_regenerated_from_the_declared_random_initialization(completed_jobs):
    root, config, identity = completed_jobs
    seed, model = config["seeds"][0], "conditional"
    phase = prefix_phases(seed, config)[0]
    before = load(root / f"{model}_{seed}" / phase["name"] / "before.pt", identity)["learner"]
    expected = new_learner(model, seed, config)
    assert learner_signature(before) == learner_signature(expected)
    predicted = batch(AcquisitionWorld(), Law(seed % 2), seed, phase["channel"], 0, config["batch_size"])
    record = read_json(root / f"{model}_{seed}" / phase["name"] / "result.json")
    assert record["packets"][0]["query_sha256"] == predicted.fingerprint()
