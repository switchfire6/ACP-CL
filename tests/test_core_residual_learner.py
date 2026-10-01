"""Zero-output insertion, causal training and exact selective trainability."""

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.conditional.learner import evidence_weights
from acp_cl.core_residual.learner import (
    ARMS, CoreResidualLearner, _optimizer_part, fork_core, learner_signature, new_from_scratch,
)
from acp_cl.persistence.learner import state_hash
from acp_cl.predictive_value.mechanism import _value_hash, predict_all, state_signature
from acp_cl.predictive_value.study import load, read_json, save
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import new_learner


@pytest.fixture(params=("conditional", "recurrent"))
def core(request):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = read_json("configs/rehearsal_state_smoke.json")
    settings.update(residual_width=6, residual_context_width=4, residual_decoder_width=7,
                    updates_per_batch=1, cpu_affinity=0, workers=1)
    seed = 16991
    learner = new_learner(request.param, seed, settings)
    world = AcquisitionWorld()
    for index in range(3):
        learner.train(batch(world, Law(index % 2), seed, "core-residual-test-prefix", index, settings["batch_size"]))
    query = batch(world, Law(1, (0,)), seed, "core-residual-test-query", 0, settings["batch_size"])
    return learner, settings, seed, query


def feature_state(module):
    return {name:value.detach().clone() for name, value in module.state_dict().items()
            if name.startswith(("frame.", "temporal."))}


def same_state(first, second):
    return first.keys() == second.keys() and all(torch.equal(first[key], second[key]) for key in first)


@pytest.mark.parametrize("arm", ARMS)
def test_zero_output_insertion_exactly_preserves_initial_predictions(core, arm):
    base, settings, seed, query = core
    signature = state_signature(base)
    learner = fork_core(base, arm, seed, settings)
    assert isinstance(learner, CoreResidualLearner)
    for support in (None, base.history, base.history.take(slice(0, 3))):
        for override in (None, "shuffled"):
            expected = base.probabilities(query.observations, support, override=override)
            actual = learner.probabilities(query.observations, support, override=override)
            assert all(torch.equal(left, right) for left, right in zip(expected, actual))
        assert np.array_equal(predict_all(base, query.observations, support), predict_all(learner, query.observations, support))
    finals = ([head[-1] for head in learner.model.residual.heads] if base.method == "conditional"
              else [learner.model.residual.decoder[-1]])
    assert all(torch.count_nonzero(layer.weight) == 0 and torch.count_nonzero(layer.bias) == 0 for layer in finals)
    assert state_signature(base) == signature


@pytest.mark.parametrize("arm", ARMS)
def test_insertion_deepcopies_core_adam_gradients_and_causal_memory(core, arm):
    base, settings, seed, _ = core
    learner = fork_core(base, arm, seed, settings)
    source = dict(base.model.named_parameters())
    target = dict(learner.model.core.named_parameters())
    assert list(source) == list(target)
    for name, parameter in target.items():
        original = source[name]
        assert parameter.data_ptr() != original.data_ptr() and torch.equal(parameter, original)
        assert torch.equal(parameter.grad, original.grad) and parameter.grad.data_ptr() != original.grad.data_ptr()
        actual, expected = learner.optimizer.state[parameter], base.optimizer.state[original]
        assert actual.keys() == expected.keys()
        for field in actual:
            if isinstance(actual[field], torch.Tensor):
                assert torch.equal(actual[field], expected[field])
                assert actual[field].data_ptr() != expected[field].data_ptr()
            else:
                assert actual[field] == expected[field]
    assert _value_hash(_optimizer_part(learner.optimizer, learner.model.core)) == _value_hash(_optimizer_part(base.optimizer, base.model))
    assert all(parameter not in learner.optimizer.state for parameter in learner.model.residual.parameters())
    owned = [parameter for group in learner.optimizer.param_groups for parameter in group["params"]]
    assert len(owned) == len(list(learner.model.parameters()))
    assert all(left is right for left, right in zip(owned, learner.model.parameters()))
    assert memory_state(learner.memory) == memory_state(base.memory)
    assert learner.cost == base.cost and learner.peak_memory_bytes == base.peak_memory_bytes
    assert learner.history.fingerprint() == base.history.fingerprint()
    assert learner.history is not base.history
    assert not np.shares_memory(learner.history.observations, base.history.observations)
    for left, right in zip(base.memory.packets, learner.memory.packets):
        assert not np.shares_memory(left.query.observations, right.query.observations)


@pytest.mark.parametrize("arm", ARMS)
def test_trainability_and_frozen_states_follow_declared_arm(core, arm):
    base, settings, seed, query = core
    learner = fork_core(base, arm, seed, settings)
    core_before = copy.deepcopy(learner.model.core.state_dict())
    residual_features_before = feature_state(learner.model.residual)
    core_adam_before = _value_hash(_optimizer_part(learner.optimizer, learner.model.core))
    for index in range(3):
        data = batch(AcquisitionWorld(), Law(index % 2, (0,)), seed, "core-residual-updates", index, settings["batch_size"])
        learner.train(data)
    assert any(torch.count_nonzero(parameter) > 0 for name, parameter in learner.model.residual.named_parameters()
               if name.endswith("2.weight"))
    if arm == "joint":
        assert not same_state(core_before, learner.model.core.state_dict())
        assert all(parameter.requires_grad for parameter in learner.model.core.parameters())
        assert _value_hash(_optimizer_part(learner.optimizer, learner.model.core)) != core_adam_before
    else:
        assert same_state(core_before, learner.model.core.state_dict())
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in learner.model.core.parameters())
        assert _value_hash(_optimizer_part(learner.optimizer, learner.model.core)) == core_adam_before
    if arm == "fixed_features":
        assert same_state(residual_features_before, feature_state(learner.model.residual))
    else:
        assert not same_state(residual_features_before, feature_state(learner.model.residual))
    learner.model.train()
    assert learner.model.training and learner.model.residual.training
    assert learner.model.core.training == (arm == "joint")
    assert learner.model.residual.frame.training == (arm != "fixed_features")
    assert learner.model.residual.temporal.training == (arm != "fixed_features")
    learner.model.eval()
    assert not any(module.training for module in learner.model.modules())
    learner.model.train()
    assert learner.model.core.training == (arm == "joint")
    frozen = [parameter for parameter in learner.model.parameters() if not parameter.requires_grad]
    owned = {id(parameter) for parameter in learner.optimizer.param_groups[0]["params"]}
    assert all(id(parameter) in owned for parameter in frozen)
    for name, parameter in learner.model.residual.named_parameters():
        expected_trainable = arm != "fixed_features" or not name.startswith(("frame.", "temporal."))
        assert parameter.requires_grad == expected_trainable


def test_new_representation_learning_begins_after_zero_output_layer_opens(core):
    base, settings, seed, query = core
    learner = fork_core(base, "separate", seed, settings)
    initial = feature_state(learner.model.residual)
    learner.train(query)
    assert same_state(initial, feature_state(learner.model.residual))
    assert all(torch.count_nonzero(parameter.grad) == 0 for name, parameter in learner.model.residual.named_parameters()
               if name.startswith(("frame.", "temporal.")))
    learner.train(batch(AcquisitionWorld(), Law(1, (0,)), seed, "core-residual-query-later", 1, settings["batch_size"]))
    assert not same_state(initial, feature_state(learner.model.residual))
    assert learner.diagnostics()["core_residual"]["residual_encoder_displacement_from_initial"] > 0
    assert learner.diagnostics()["core_residual"]["core_displacement_from_fork"] == 0


def test_uniform_replay_membership_sampling_history_and_logical_cost_match_across_arms(core):
    base, settings, seed, _ = core
    learners = [fork_core(base, arm, seed, settings) for arm in ARMS]
    for index in range(6):
        data = batch(AcquisitionWorld(), Law(index % 2, (0,)), seed, "core-residual-matched-stream", index, settings["batch_size"])
        for learner in learners:
            learner.train(copy.deepcopy(data))
        assert len({_value_hash(memory_state(learner.memory)) for learner in learners}) == 1
        assert len({learner.history.fingerprint() for learner in learners}) == 1
        assert len({_value_hash({name:value for name,value in learner.cost.items() if name != "training_seconds"})
                    for learner in learners}) == 1
    diagnostics = [learner.diagnostics()["core_residual"] for learner in learners]
    assert len({item["residual_initial_hash"] for item in diagnostics}) == 1
    assert len({item["core_at_fork_hash"] for item in diagnostics}) == 1
    assert len({item["parameters"]["total"] for item in diagnostics}) == 1
    for learner, item in zip(learners, diagnostics):
        work = item["work"]
        assert work["packets"] == work["optimizer_steps"] == work["backward_calls"] == 6
        assert work["core_forward_calls"] == work["residual_forward_calls"] == work["logical_probability_calls"] == 12
        assert work["core_query_presentations"] == work["residual_query_presentations"] == 12*settings["batch_size"]
        assert work["core_support_presentations"] == work["residual_support_presentations"] == 12*settings["batch_size"]
        assert work["core_backward_paths"] == (12 if learner.arm == "joint" else 0)
        assert work["residual_backward_paths"] == 12


@pytest.mark.parametrize("arm", ARMS)
def test_prediction_diagnostics_and_checkpoint_preserve_full_state_and_gradients(core, arm, tmp_path):
    base, settings, seed, query = core
    learner = fork_core(base, arm, seed, settings)
    learner.train(query)
    learner.model.train()
    signature, global_rng = learner_signature(learner), torch.random.get_rng_state().clone()
    learner.predict(query.observations, learner.history)
    predict_all(learner, query.observations, learner.history)
    report = learner.diagnostics()
    assert learner_signature(learner) == signature and torch.equal(global_rng, torch.random.get_rng_state())
    assert report["optimizer_bytes"] == (report["core_residual"]["core_optimizer_bytes"]
                                          + report["core_residual"]["residual_optimizer_bytes"])
    path, identity = tmp_path/"checkpoint.pt", {"engineering":"core-residual-roundtrip"}
    save(path, dict(identity=identity, learner=learner))
    restored = load(path, identity)["learner"]
    assert learner_signature(restored) == signature
    assert all(left is right for left,right in zip(restored.model.parameters(), restored.optimizer.param_groups[0]["params"]))
    next_packet = batch(AcquisitionWorld(), Law(0, (1,)), seed, "core-residual-roundtrip", 1, settings["batch_size"])
    learner.train(copy.deepcopy(next_packet))
    restored.train(copy.deepcopy(next_packet))
    assert learner_signature(learner, ignore_walltime=True) == learner_signature(restored, ignore_walltime=True)


def test_fresh_joint_has_random_core_identical_residual_seed_and_empty_replay_moments(core):
    base, settings, seed, query = core
    fork = fork_core(base, "joint", seed, settings)
    random_state = torch.random.get_rng_state().clone()
    fresh = new_from_scratch(base.method, seed, settings)
    assert torch.equal(random_state, torch.random.get_rng_state())
    assert fresh.arm == "joint" and fresh.history is None and not fresh.memory.packets
    assert not fresh.optimizer.state and fresh.memory.seen == 0 and fresh.cost["arrivals"] == 0
    assert all(value == 0 for value in fresh.path_work.values())
    assert state_hash(fresh.model.core.state_dict()) == base.initial_hash
    assert state_hash(fresh.model.residual.state_dict()) == state_hash(fork.model.residual.state_dict())
    fresh.history = copy.deepcopy(base.history)
    fresh.train(query)
    assert fresh.memory.seen == 1 and fresh.cost["optimizer_steps"] == settings["updates_per_batch"]


def test_privileged_modes_are_rejected_before_mutation(core):
    base, settings, seed, query = core
    learner = fork_core(base, "separate", seed, settings)
    signature = learner_signature(learner)
    hidden = np.zeros(len(query), dtype=np.int64)
    with pytest.raises(ValueError, match="latent modes"):
        learner.train(query, hidden)
    with pytest.raises(ValueError, match="latent modes"):
        learner.probabilities(query.observations, learner.history, hidden)
    with pytest.raises(ValueError, match="latent modes"):
        learner.predict(query.observations, learner.history, override="oracle")
    assert learner_signature(learner) == signature
    bad = copy.deepcopy(base)
    bad.memory.packets[0].oracle_modes = hidden
    with pytest.raises(ValueError, match="privileged labels"):
        fork_core(bad, "joint", seed, settings)


def test_measurement_snapshots_never_enter_predictions_or_updates(core):
    base, settings, seed, query = core
    learner = fork_core(base, "separate", seed, settings)
    second = copy.deepcopy(learner)
    for left, right in zip(learner.model.parameters(), second.model.parameters()):
        right.grad = None if left.grad is None else left.grad.detach().clone()
    for state in (second.core_at_fork, second.residual_at_fork, second.initial_encoder):
        for value in state.values():
            value.add_(123.)
    assert learner_signature(learner) != learner_signature(second)
    assert np.array_equal(predict_all(learner, query.observations, learner.history), predict_all(second, query.observations, second.history))
    learner.train(copy.deepcopy(query))
    second.train(copy.deepcopy(query))
    assert state_hash(learner.model.state_dict()) == state_hash(second.model.state_dict())
    assert _value_hash(learner.optimizer.state_dict()) == _value_hash(second.optimizer.state_dict())


def test_invalid_fork_configuration_is_rejected(core):
    base, settings, seed, _ = core
    with pytest.raises(ValueError, match="unknown"):
        fork_core(base, "unknown", seed, settings)
    with pytest.raises(ValueError, match="preserve all core settings"):
        fork_core(base, "joint", seed, {**settings, "lr":settings["lr"]*2})
    with pytest.raises(ValueError, match="positive residual dimensions"):
        plain = new_learner(base.method, seed, {key:value for key,value in settings.items() if not key.startswith("residual_")})
        fork_core(plain, "joint", seed, {**plain.settings, "residual_width":0,
                                       "residual_context_width":4, "residual_decoder_width":7})
    with pytest.raises(TypeError, match="construct with"):
        CoreResidualLearner()
    assert learner_signature(base) == state_signature(base)


def test_conditional_residual_changes_support_evidence_before_mixing(core):
    base, settings, seed, query = core
    if base.method != "conditional":
        pytest.skip("conditional evidence routing only")
    learner = fork_core(base, "separate", seed, settings)
    with torch.no_grad():
        for index, head in enumerate(learner.model.residual.heads):
            head[-1].bias.fill_((index-1.5)*.7)
    sx, sa, sy = learner.tensors(learner.history)
    qx = torch.as_tensor(query.observations)
    combined = torch.cat((sx, qx))
    logits = learner.model.core(combined)+learner.model.residual(combined)
    weights = evidence_weights(logits[:len(sx)], sa, sy, settings["evidence_strength"])
    expected = (weights[None, :, None, None]*logits[len(sx):].sigmoid()).sum(dim=1)
    actual, actual_weights, actual_logits = learner.probabilities(query.observations, learner.history)
    assert torch.equal(actual, expected) and torch.equal(actual_logits, logits[len(sx):])
    assert torch.equal(actual_weights, weights[None].expand(len(query), -1))
    assert not torch.equal(actual_weights, base.probabilities(query.observations, learner.history)[1])


def test_recurrent_residual_is_added_in_logit_space_before_sigmoid(core):
    base, settings, seed, query = core
    if base.method != "recurrent":
        pytest.skip("recurrent decoder only")
    learner = fork_core(base, "separate", seed, settings)
    with torch.no_grad():
        learner.model.residual.decoder[-1].bias.fill_(.4)
    original = base.probabilities(query.observations, base.history)[2][:, 0]
    actual, weights, logits = learner.probabilities(query.observations, base.history)
    assert torch.equal(logits[:, 0], original+.4)
    assert torch.equal(actual, (original+.4).sigmoid())
    assert torch.equal(weights, torch.ones((len(query), 1)))
    selected = actual[torch.arange(len(query)), torch.as_tensor(query.actions.astype(np.int64))]
    loss = F.binary_cross_entropy(selected.clamp(1e-6, 1-1e-6), torch.as_tensor(query.survival.astype(np.float32)))
    assert torch.isfinite(loss)
