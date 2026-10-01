"""Scientific contracts for separate acquisition and bounded consolidation."""

import copy
import json

import numpy as np
import pytest
import torch

from acp_cl import experiment
from acp_cl.data import build_stream, preprocess
from acp_cl.dual_path import DUAL_METHODS, DualPathConfig, DualPathLearner, tensor_bytes
from test_experiment import assert_tree_equal, tiny_config, lightweight_environment

__all__ = ["tiny_config", "lightweight_environment"]


@pytest.fixture
def config(tiny_config):
    tiny_config["dual_path"] = {
        "fast_width": 6, "consolidation_interval": 3, "consolidation_steps": 2,
        "consolidation_batch_size": 4, "consolidation_pool_size": 5,
        "probe_interval": 1,
    }
    tiny_config["plasticity"].update(momentum=.9, weight_decay=0.0)
    tiny_config["steps_per_experience"] = 5
    return tiny_config


def make(config, method="dual_consolidate", device="cpu"):
    stream = build_stream(config["data"], 2027)
    learner = experiment.new_learner(config, stream, method, 2027, torch.device(device))
    sequence = [batch for e in stream.experiences for batch in experiment.batches(e, config, 2027)]
    return learner, stream, sequence


def test_initial_zero_residual_preserves_single_model_and_global_rng(config):
    torch.set_num_threads(1)
    stream = build_stream(config["data"], 2027)
    rng = torch.get_rng_state().clone()
    learner = DualPathLearner(config, stream.input_shape, stream.num_classes,
                              "dual_consolidate", seed=2027)
    assert_tree_equal(torch.get_rng_state(), rng)
    baseline = experiment.new_learner(config, stream, "er", 2027, torch.device("cpu"))
    assert_tree_equal(baseline.model.state_dict(), learner.model.stable.state_dict())
    x = stream.experiences[0].train.tensors[0]
    assert_tree_equal(learner.model(x), baseline.model(x))
    assert torch.count_nonzero(learner.model.fast(x)) == 0


def test_stable_path_frozen_and_fast_features_learn_from_input(config):
    learner, _, sequence = make(config)
    stable = copy.deepcopy(learner.model.stable.state_dict())
    hidden = copy.deepcopy(learner.model.fast.hidden1.state_dict())
    learner.train_batch(*sequence[0])
    # Zero output projection permits its own learning on the first update;
    # feature gradients start only after that connection becomes nonzero.
    assert_tree_equal(hidden, learner.model.fast.hidden1.state_dict())
    learner.train_batch(*sequence[1])
    assert_tree_equal(stable, learner.model.stable.state_dict())
    assert any(not torch.equal(v, learner.model.fast.hidden1.state_dict()[k]) for k, v in hidden.items())
    assert not learner.model.stable.training


def test_consolidation_updates_stable_then_removes_fast_and_clears_momentum(config):
    learner, _, sequence = make(config)
    stable = copy.deepcopy(learner.model.stable.state_dict())
    for batch in sequence[:3]:
        event = learner.train_batch(*batch)
    assert any(not torch.equal(v, learner.model.stable.state_dict()[k]) for k, v in stable.items())
    assert learner.consolidations == learner.fast_resets == 1
    assert not learner.online_optimizer.state
    assert learner.stable_optimizer.state
    assert torch.count_nonzero(learner.model.fast(sequence[0][0])) == 0
    assert set(event["probe"]) == {"before_online", "after_online",
                                   "after_consolidation_before_reset", "after_reset"}
    assert event["consolidation"]["pool_examples"] <= len(sequence[2][1]) + 5
    assert learner.cost["consolidation_optimizer_steps"] == 2
    assert learner.cost["online_optimizer_steps"] == 3
    assert learner.buffer.num_seen == sum(len(b[1]) for b in sequence[:3])


@pytest.mark.parametrize("method", (*DUAL_METHODS, "dual_scratch"))
def test_exact_resume_across_consolidation_and_evaluation(config, method):
    first, stream, sequence = make(config, method)
    second, _, _ = make(config, method)
    for batch in sequence[:4]:
        first.train_batch(*batch)
    state = first.state_dict()
    second.load_state_dict(state)
    for batch in sequence[4:]:
        first.accuracy(stream.experiences[0].validation)
        second.accuracy(stream.experiences[0].validation)
        assert_tree_equal(first.train_batch(*batch), second.train_batch(*batch))
    assert_tree_equal(first.state_dict(), second.state_dict())
    assert state["step_number"] == 4  # Snapshot cannot alias live state.


def test_replay_membership_and_online_rng_pairing_ignores_consolidation(config):
    learners = [make(config, m)[0] for m in DUAL_METHODS]
    _, _, sequence = make(config)
    for batch in sequence:
        for learner in learners:
            learner.train_batch(*batch)
    reference = learners[0].buffer.state_dict()
    for learner in learners[1:]:
        state = learner.buffer.state_dict()
        for field in ("x", "y", "num_seen", "reservoir_rng_state"):
            assert_tree_equal(reference[field], state[field])
        assert_tree_equal(reference["sample_rng_states"]["train"], state["sample_rng_states"]["train"])
        assert_tree_equal(learners[0].augmentation_rng.get_state(), learner.augmentation_rng.get_state())
        assert_tree_equal(learners[0].replay_augmentation_rng.get_state(),
                          learner.replay_augmentation_rng.get_state())


def test_evaluation_cannot_influence_training_or_consolidation(config):
    learner, stream, sequence = make(config)
    learner.train_batch(*sequence[0])
    before = learner.state_dict()
    learner.accuracy(stream.experiences[0].validation)
    after = learner.state_dict()
    for key in ("evaluation_examples", "evaluation_branch_forward_examples"):
        assert after["cost"][key] > before["cost"][key]
        after["cost"][key] = before["cost"][key]
    assert_tree_equal(before, after)


def test_changed_checkpoint_config_rejected_before_mutation(config):
    learner, _, sequence = make(config)
    learner.train_batch(*sequence[0])
    changed = copy.deepcopy(config)
    changed["dual_path"]["consolidation_steps"] += 1
    target, _, _ = make(changed)
    before = target.state_dict()
    with pytest.raises(ValueError, match="configuration mismatch"):
        target.load_state_dict(learner.state_dict())
    assert_tree_equal(before, target.state_dict())


def test_costs_and_payload_are_explicit_and_bounded(config):
    learner, _, sequence = make(config)
    expected_replay = 0
    for batch in sequence:
        expected_replay += min(len(learner.buffer), config["replay_batch_size"])
        learner.train_batch(*batch)
    cost = learner.cost
    n = sum(len(b[1]) for b in sequence)
    assert cost["current_examples"] == n
    assert cost["replay_examples"] == expected_replay
    assert cost["online_branch_forward_examples"] == 2 * (n + expected_replay)
    assert cost["online_branch_backward_examples"] == n + expected_replay
    assert cost["consolidation_optimizer_steps"] == len(sequence) // 3 * 2
    assert cost["consolidation_examples"] <= cost["consolidation_optimizer_steps"] * 4
    info = learner.diagnostics()
    assert info["model_payload_bytes"] == tensor_bytes(learner.model.state_dict())
    assert info["peak_persistent_payload_bytes"] >= info["model_payload_bytes"] + learner.buffer.nbytes()
    assert info["peak_auxiliary_payload_bytes"] > 0


@pytest.mark.parametrize("field,value", [
    ("fast_width", 0), ("consolidation_interval", True), ("consolidation_steps", 0),
    ("temperature", 0), ("temperature", float("nan")), ("max_grad_norm", float("inf")),
    ("distillation_weight", -1), ("probe_interval", 0),
])
def test_invalid_settings_fail(field, value):
    with pytest.raises(ValueError):
        DualPathConfig(**{field: value})


def test_cnn_fast_convolutions_learn_and_stable_features_stay_fixed(config):
    config.update(model="cnn", width=16)
    config["data"] = {"dataset": "procedural_shapes", "n_experiences": 2,
                      "n_domains": 2, "regime": "recurring", "color_policy": "independent",
                      "train_per_class": 4, "validation_per_class": 2, "test_per_class": 0}
    learner, _, sequence = make(config)
    before_stable = copy.deepcopy(learner.model.stable.state_dict())
    before_fast = copy.deepcopy(learner.model.fast.conv1.state_dict())
    for batch in sequence[:2]:
        learner.train_batch(*batch)
    assert_tree_equal(before_stable, learner.model.stable.state_dict())
    assert any(not torch.equal(v, learner.model.fast.conv1.state_dict()[k])
               for k, v in before_fast.items())
    x = preprocess(sequence[0][0], learner.dataset)
    assert learner.model(x).shape == (len(x), learner.num_classes)


def test_runner_uses_architecture_matched_scratch_and_resumes(config, tmp_path):
    config["scratch_reference"] = True
    methods = ["er", "dual_joint", "dual_consolidate"]
    results = experiment.run_suite(config, output=tmp_path, seeds=[2027], methods=methods,
                                   device_name="cpu")
    assert (tmp_path / "scratch_seed2027.json").exists()
    assert (tmp_path / "scratch_dual_seed2027.json").exists()
    for result in results[1:]:
        assert result["scratch_reference_method"] == "dual_scratch"
        assert result["allocation_summary"] is None
        assert result["information_access"] == "training stream only"
    assert results[1]["model_parameters"] == results[2]["model_parameters"]
    resumed = experiment.run_suite(config, output=tmp_path, seeds=[2027], methods=methods,
                                   device_name="cpu", resume=True)
    assert_tree_equal(results, resumed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_checkpoint_crosses_actual_consolidation(config):
    first, _, sequence = make(config, device="cuda")
    second, _, _ = make(config, device="cuda")
    for batch in sequence[:2]:
        first.train_batch(*batch)
    second.load_state_dict(first.state_dict())
    assert_tree_equal(first.train_batch(*sequence[2]), second.train_batch(*sequence[2]))
    assert first.consolidations == second.consolidations == 1
    assert_tree_equal(first.state_dict(), second.state_dict())


def test_cifar100_arrival_ids_match_pixels_and_single_pass_runner(config, tmp_path, monkeypatch):
    from torchvision import datasets

    class FakeCifar100:
        def __init__(self, root, train, download):
            count = 6 if train else 1
            self.targets = np.repeat(np.arange(100), count).tolist()
            self.data = np.zeros((100 * count, 8, 8, 3), dtype=np.uint8)
            self.data[..., 0] = np.asarray(self.targets)[:, None, None]
            self.data[..., 1] = np.tile(np.arange(count), 100)[:, None, None]

    monkeypatch.setattr(datasets, "CIFAR100", FakeCifar100)
    config.update(model="cnn", width=8, single_pass=True, epochs=1)
    config.pop("steps_per_experience")
    config["data"] = {"dataset": "cifar100", "n_experiences": 2,
                      "classes_per_experience": 2, "train_per_class": 3,
                      "validation_per_class": 2, "test_per_class": 0,
                      "record_arrival_ids": True}
    stream = build_stream(config["data"], 2027)
    groups = stream.metadata["current_arrival_ids_by_experience"]
    for exp, ids in zip(stream.experiences, groups):
        x = exp.train.tensors[0]
        assert ids == (x[:, 0, 0, 0].long() * 6 + x[:, 1, 0, 0].long()).tolist()
    result = experiment.run_one(config, stream, "dual_consolidate", 2027,
                                 torch.device("cpu"), tmp_path)
    assert result["current_arrival_audit"]["unique_current_base_images"] == 12
    assert result["cost"]["current_examples"] == 12
    assert result["diagnostics"]["consolidations"] == 1


def test_portable_report_is_complete_and_refuses_mixed_sources(config, tmp_path):
    from scripts.summarize_dual_path import summarize

    config["scratch_reference"] = True
    config["methods"] = ["dual_joint", "dual_consolidate"]
    directory, report = tmp_path / "runs", tmp_path / "report"
    results = experiment.run_suite(config, output=directory, seeds=[2027],
                                   methods=config["methods"], device_name="cpu")
    archive = summarize(directory, report)
    assert len(archive["runs"]) == 2
    row = next(r for r in archive["summary"] if r["method"] == "dual_consolidate")
    assert row["final_accuracy"] == results[1]["metrics"]["final_accuracy"]
    assert (report / "summary.md").exists()
    file = directory / "dual_joint_seed2027/result.json"
    original = json.loads(file.read_text())
    invalid = {**original, "source_sha256": "different-source"}
    file.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="mismatch"):
        summarize(directory, tmp_path / "bad_report")
    assert not (tmp_path / "bad_report").exists()
    file.unlink()
    with pytest.raises(ValueError, match="incomplete suite"):
        summarize(directory, tmp_path / "bad_report")


def test_legacy_constructor_cannot_silently_run_a_dual_method(config):
    from acp_cl.learner import Learner
    from acp_cl.models import make_model

    with pytest.raises(ValueError, match="DualPathLearner"):
        Learner(make_model("mlp", (6,), 4, 8), "dual_joint", config)


@pytest.mark.parametrize("reference_method", ["finetune", "dual_scratch"])
def test_scratch_cache_rejects_other_architecture(config, tmp_path, reference_method):
    config["scratch_reference"] = True
    stream = build_stream(config["data"], 2027)
    experiment.scratch_reference(config, stream, 2027, torch.device("cpu"), tmp_path,
                                  reference_method=reference_method)
    prefix = "scratch_dual" if reference_method == "dual_scratch" else "scratch"
    path = tmp_path / f"{prefix}_seed2027.json"
    saved = json.loads(path.read_text())
    saved["reference_method"] = "finetune" if reference_method == "dual_scratch" else "dual_scratch"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="mismatch"):
        experiment.scratch_reference(config, stream, 2027, torch.device("cpu"), tmp_path,
                                      resume=True, reference_method=reference_method)


def test_runner_refuses_wrong_scratch_architecture_before_training(config, tmp_path):
    stream = build_stream(config["data"], 2027)
    with pytest.raises(ValueError, match="scratch reference architecture"):
        experiment.run_one(config, stream, "dual_joint", 2027, torch.device("cpu"), tmp_path,
                            scratch={"reference_method": "finetune"})
    assert not list(tmp_path.iterdir())
