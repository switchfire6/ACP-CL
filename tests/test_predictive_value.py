"""Matched rehearsal updates, causal packet use, and untouched online state."""

import copy
import hashlib
import json
import pickle
import random

import numpy as np
import pytest
import torch

from acp_cl.acquisition.world import AcquisitionWorld, Law
from acp_cl.conditional.learner import Packet
from acp_cl.persistence.learner import state_hash
from acp_cl.predictive_value.mechanism import (
    clone_exact, make_shadow, packet_fingerprint, predict_all, predict_performed, state_signature,
)
from acp_cl.replay_renewal.memory import ReservoirMemory, memory_state
from acp_cl.replay_renewal.study import new_learner


@pytest.fixture
def settings():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return dict(width=8, experts=4, context_width=5, decoder_width=8,
        interaction_features=True, lr=.002, memory_packets=4, batch_size=8,
        evidence_strength=1., updates_per_batch=2)


def data(index):
    return AcquisitionWorld().experience(Law(index % 2, (0, 1) if index >= 3 else ()), 8, 8300+index)


def trained(model, settings):
    learner = new_learner(model, 83, settings)
    for index in range(3):
        learner.train(data(index))
    return learner


def rng_signature():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
        torch.get_rng_state().numpy()), protocol=5)).hexdigest()


def assert_tensor_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tensor_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tensor_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_exact_clone_preserves_gradients_adam_aliases_modes_and_independent_storage(model, settings):
    parent = trained(model, settings)
    parent.model.train()
    parent.model.frame.eval()
    initial = state_signature(parent)
    copied = clone_exact(parent)
    assert state_signature(copied) == initial
    model_ids = {id(p) for p in copied.model.parameters()}
    assert {id(p) for group in copied.optimizer.param_groups for p in group["params"]} == model_ids
    for old, new in zip(parent.model.parameters(), copied.model.parameters()):
        assert old.data_ptr() != new.data_ptr()
        assert old.grad is not None and new.grad is not None
        assert torch.equal(old.grad, new.grad) and old.grad.data_ptr() != new.grad.data_ptr()
    assert not np.shares_memory(parent.history.observations, copied.history.observations)
    assert parent.memory.sampling_rng is not copied.memory.sampling_rng
    assert parent.memory.membership_rng is not copied.memory.membership_rng
    copied.train(data(3))
    assert state_signature(parent) == initial
    assert state_signature(copied) != initial


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
@pytest.mark.parametrize("updates", (1, 4))
def test_rehearsal_matches_original_training_with_forced_replay_and_identical_adam(model, updates, settings):
    parent = trained(model, settings)
    anchor, replay = Packet(parent.history, data(3)), parent.memory.packets[0]
    shadow, work = make_shadow(parent, anchor, replay, updates)
    reference = clone_exact(parent)

    class ForcedReplay:
        def sample(self):
            return replay

        def add(self, packet):
            pass

        def nbytes(self):
            return replay.nbytes()

    reference.history = copy.deepcopy(anchor.support)
    reference.memory = ForcedReplay()
    reference.settings["updates_per_batch"] = updates
    # This invokes the unchanged, original ContextLearner.train implementation.
    reference.train(anchor.query)
    assert state_hash(shadow.model.state_dict()) == state_hash(reference.model.state_dict())
    assert_tensor_tree_equal(shadow.optimizer.state_dict(), reference.optimizer.state_dict())
    for a, b in zip(shadow.model.parameters(), reference.model.parameters()):
        assert torch.equal(a.grad, b.grad)
    old_steps = [float(state["step"]) for state in parent.optimizer.state.values()]
    new_steps = [float(state["step"]) for state in shadow.optimizer.state.values()]
    assert new_steps == [value+updates for value in old_steps]
    assert work["initial_model_sha256"] == state_hash(parent.model.state_dict())
    assert work["final_model_sha256"] == state_hash(shadow.model.state_dict())


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_rehearsal_never_samples_adds_or_advances_original_or_copied_causal_state(model, settings, monkeypatch):
    parent = trained(model, settings)
    anchor, replay = Packet(parent.history, data(3)), parent.memory.packets[0]
    signature, rng = state_signature(parent), rng_signature()
    hashes = packet_fingerprint(anchor), packet_fingerprint(replay)

    def forbidden(*args, **kwargs):
        raise AssertionError("rehearsal may not sample or add memory")

    monkeypatch.setattr(ReservoirMemory, "sample", forbidden)
    monkeypatch.setattr(ReservoirMemory, "add", forbidden)
    shadow, work = make_shadow(parent, anchor, replay, 3)
    assert state_signature(parent) == signature
    assert rng_signature() == rng
    assert shadow.history.fingerprint() == parent.history.fingerprint()
    assert memory_state(shadow.memory) == memory_state(parent.memory)
    assert shadow.cost == parent.cost
    assert (packet_fingerprint(anchor), packet_fingerprint(replay)) == hashes
    assert work["steps"] == work["backward_calls"] == 3
    assert work["forwards"] == 6
    assert work["query_presentations"] == work["support_presentations"] == 6*settings["batch_size"]
    assert work["model_bytes"] == parent.diagnostics()["model_bytes"]
    assert work["optimizer_bytes"] == shadow.diagnostics()["optimizer_bytes"]
    assert work["gradient_bytes"] == work["model_bytes"]
    assert work["initial_encoder_bytes"] == sum(
        value.numel()*value.element_size() for value in shadow.initial_encoder.values())
    assert work["replay_bytes"] == parent.memory.nbytes()
    assert work["history_bytes"] == Packet(None, parent.history).nbytes()
    assert work["explicit_tensor_bytes_subtotal"] == sum(work[name] for name in (
        "model_bytes", "optimizer_bytes", "gradient_bytes", "initial_encoder_bytes",
        "replay_bytes", "history_bytes"))
    assert "double counted" in work["explicit_tensor_bytes_scope"]
    assert "not measured peak RAM" in work["explicit_tensor_bytes_scope"]
    assert work["parent_unchanged"] and work["memory_history_unchanged"]
    assert work["parent_state_sha256"] == signature
    assert work["shadow_state_sha256"] == state_signature(shadow)
    assert work["training_seconds"] >= 0
    json.dumps(work, allow_nan=False)


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_same_treatment_is_exact_null_and_different_treatments_start_from_same_parent(model, settings):
    parent = trained(model, settings)
    anchor, candidate, replacement = Packet(parent.history, data(3)), parent.memory.packets[0], parent.memory.packets[1]
    first, first_work = make_shadow(parent, anchor, candidate, 2)
    same, same_work = make_shadow(parent, anchor, candidate, 2)
    other, other_work = make_shadow(parent, anchor, replacement, 2)
    assert state_signature(first) == state_signature(same)
    assert {**first_work, "training_seconds": 0.} == {**same_work, "training_seconds": 0.}
    assert first_work["parent_state_sha256"] == other_work["parent_state_sha256"]
    assert first_work["initial_optimizer_sha256"] == other_work["initial_optimizer_sha256"]
    assert first_work["initial_model_sha256"] == other_work["initial_model_sha256"]
    assert state_hash(first.model.state_dict()) != state_hash(other.model.state_dict())
    assert first.memory.packets[0] is not same.memory.packets[0]


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_rehearsal_uses_each_original_support_not_the_shadow_latest_history(model, settings, monkeypatch):
    parent = trained(model, settings)
    anchor, replay = Packet(data(1), data(3)), parent.memory.packets[0]
    expected = [anchor.support.fingerprint(), None if replay.support is None else replay.support.fingerprint()]*3
    observed = []
    original = type(parent).probabilities

    def checked(self, observations, support, oracle_modes=None, override=None):
        assert oracle_modes is None
        observed.append(None if support is None else support.fingerprint())
        return original(self, observations, support, oracle_modes, override)

    monkeypatch.setattr(type(parent), "probabilities", checked)
    make_shadow(parent, anchor, replay, 3)
    assert observed == expected


@pytest.mark.parametrize("updates", (0, -1, 1.5, True))
def test_invalid_update_budgets_are_rejected(updates, settings):
    parent = trained("conditional", settings)
    with pytest.raises(ValueError, match="positive integer"):
        make_shadow(parent, Packet(parent.history, data(3)), parent.memory.packets[0], updates)


@pytest.mark.parametrize("position", ("anchor", "replay"))
def test_hidden_labels_are_rejected_without_changing_parent(position, settings):
    parent = trained("conditional", settings)
    packets = dict(anchor=Packet(parent.history, data(3)), replay=copy.deepcopy(parent.memory.packets[0]))
    packets[position].oracle_modes = np.zeros(8, dtype=np.uint8)
    before = state_signature(parent)
    with pytest.raises(ValueError, match="without latent labels"):
        make_shadow(parent, packets["anchor"], packets["replay"], 2)
    assert state_signature(parent) == before


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
@pytest.mark.parametrize("support_present", (False, True))
def test_prediction_preserves_model_adam_gradients_modes_memory_rng_and_history(model, support_present, settings):
    parent = trained(model, settings)
    parent.model.train()
    parent.model.frame.eval()
    support = parent.history if support_present else None
    before, rng = state_signature(parent), rng_signature()
    query = data(4)
    all_probabilities = predict_all(parent, query.observations, support)
    selected, error = predict_performed(parent, query, support)
    np.testing.assert_array_equal(selected, all_probabilities[np.arange(len(query)), query.actions])
    expected = float(((selected.astype(np.float64)-query.survival.astype(np.float64))**2).mean())
    assert error == expected
    assert state_signature(parent) == before and rng_signature() == rng


def test_reported_query_outcomes_affect_only_the_score_not_predictions_or_forward_inputs(settings, monkeypatch):
    parent = trained("conditional", settings)
    query = data(3)
    changed = copy.deepcopy(query)
    changed.survival = 1-changed.survival
    actual = []
    original = parent.predict

    def checked(observations, support, oracle_modes=None, override=None):
        assert isinstance(observations, np.ndarray) and oracle_modes is None
        assert support is parent.history
        assert not torch.is_grad_enabled()
        actual.append(observations.copy())
        return original(observations, support, oracle_modes, override)

    monkeypatch.setattr(parent, "predict", checked)
    one, score_one = predict_performed(parent, query, parent.history)
    two, score_two = predict_performed(parent, changed, parent.history)
    assert len(actual) == 2  # Exactly one forward per call.
    np.testing.assert_array_equal(actual[0], actual[1])
    np.testing.assert_array_equal(one, two)
    assert score_one != score_two


def test_metric_accepts_single_float32_epsilon_without_silently_clipping(settings, monkeypatch):
    parent = trained("conditional", settings)
    query = data(3)
    epsilon = np.finfo(np.float32).eps
    values = np.full((len(query), 5, 3), 1.+epsilon, dtype=np.float64)

    def synthetic(observations, support):
        parent.model.eval()
        return values, None, None

    monkeypatch.setattr(parent, "predict", synthetic)
    before = state_signature(parent)
    predictions, score = predict_performed(parent, query, parent.history)
    assert np.all(predictions > 1)
    assert score == float(((predictions-query.survival)**2).mean())
    assert score != float(((np.clip(predictions, 0, 1)-query.survival)**2).mean())
    assert state_signature(parent) == before
    values.fill(1.+2*epsilon)
    with pytest.raises(ValueError, match="epsilon bounds"):
        predict_performed(parent, query, parent.history)
    values.fill(np.nan)
    with pytest.raises(ValueError, match="finite query"):
        predict_performed(parent, query, parent.history)


def test_prediction_restores_all_mode_flags_on_failure(settings, monkeypatch):
    parent = trained("recurrent", settings)
    parent.model.train()
    parent.model.frame.eval()
    before = state_signature(parent)

    def failing(observations, support):
        parent.model.eval()
        raise RuntimeError("synthetic prediction failure")

    monkeypatch.setattr(parent, "predict", failing)
    with pytest.raises(RuntimeError, match="synthetic"):
        predict_all(parent, data(3).observations, parent.history)
    assert state_signature(parent) == before


@pytest.mark.parametrize("field", ("weight", "gradient", "optimizer", "history", "memory_query",
                                   "membership_rng", "sampling_rng", "mode", "settings", "cost"))
def test_signature_covers_mutable_training_state(field, settings):
    parent = trained("conditional", settings)
    altered = clone_exact(parent)
    if field == "weight":
        with torch.no_grad():
            next(altered.model.parameters()).add_(.01)
    elif field == "gradient":
        next(altered.model.parameters()).grad.add_(.01)
    elif field == "optimizer":
        next(iter(altered.optimizer.state.values()))["exp_avg"].add_(.01)
    elif field == "history":
        altered.history.observations.flat[0] ^= 1
    elif field == "memory_query":
        altered.memory.packets[0].query.observations.flat[0] ^= 1
    elif field == "membership_rng":
        altered.memory.membership_rng.random()
    elif field == "sampling_rng":
        altered.memory.sampling_rng.random()
    elif field == "mode":
        altered.model.eval()
    elif field == "settings":
        altered.settings["lr"] *= 2
    elif field == "cost":
        altered.cost["optimizer_steps"] += 1
    assert state_signature(parent) != state_signature(altered)
    assert state_signature(parent, ignore_walltime=True) != state_signature(altered, ignore_walltime=True)


def test_signature_optional_walltime_normalization_is_narrow(settings):
    parent = trained("conditional", settings)
    altered = clone_exact(parent)
    altered.cost["training_seconds"] += 10
    assert state_signature(parent) != state_signature(altered)
    assert state_signature(parent, ignore_walltime=True) == state_signature(altered, ignore_walltime=True)
    altered.cost["arrivals"] += 1
    assert state_signature(parent, ignore_walltime=True) != state_signature(altered, ignore_walltime=True)


def test_rehearsal_rejects_frozen_or_non_adam_parent_and_wrong_packet_sizes(settings):
    parent = trained("conditional", settings)
    anchor, replay = Packet(parent.history, data(3)), parent.memory.packets[0]
    too_short = Packet(anchor.support, anchor.query.take(slice(0, 4)))
    with pytest.raises(ValueError, match="packet size"):
        make_shadow(parent, too_short, replay, 1)
    parent.freeze_features()
    with pytest.raises(ValueError, match="remain trainable"):
        make_shadow(parent, anchor, replay, 1)
    parent.model.requires_grad_(True)
    parent.optimizer = torch.optim.SGD(parent.model.parameters(), lr=.002)
    with pytest.raises(ValueError, match="Adam"):
        make_shadow(parent, anchor, replay, 1)
    parent.method = "oracle"
    with pytest.raises(ValueError, match="ordinary"):
        predict_all(parent, anchor.query.observations, anchor.support)
