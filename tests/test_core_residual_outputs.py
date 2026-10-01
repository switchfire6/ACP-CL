"""Saved-state output decomposition, routing crossovers and state preservation."""

import hashlib
import inspect
import pickle
import random

import numpy as np
import pytest
import torch

from acp_cl.acquisition.world import AcquisitionWorld, Law
from acp_cl.core_residual.learner import fork_core, learner_signature
from acp_cl.core_residual_outputs import component_predictions
from acp_cl.persistence.world import Experience
from acp_cl.predictive_value.mechanism import predict_all
from acp_cl.predictive_value.study import read_json
from acp_cl.replay_renewal.study import new_learner


@pytest.fixture
def settings():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = read_json("configs/core_residual_smoke.json")
    config.update(cpu_affinity=0, workers=1, updates_per_batch=2)
    return config


def make_core(method, settings):
    learner = new_learner(method, 61, settings)
    world = AcquisitionWorld()
    for index in range(3):
        learner.train(world.experience(Law(index % 2), settings["batch_size"], 8400 + index))
    return learner


def make_trained(method, arm, settings):
    learner = fork_core(make_core(method, settings), arm, 61, settings)
    world = AcquisitionWorld()
    for index in range(4):
        learner.train(world.experience(Law(1, (0, 2)), settings["batch_size"], 8500 + index))
    query = world.experience(Law(1, (0, 2)), settings["batch_size"], 8591)
    return learner, query


def rng_signature():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
        torch.get_rng_state().numpy()), protocol=5)).hexdigest()


def set_mixed_modes(learner):
    # Direct assignment deliberately includes nonuniform descendants. Restoring
    # only the parent's mode would silently overwrite these distinct flags.
    for index, module in enumerate(learner.model.modules()):
        module.training = index % 3 == 0


@pytest.mark.parametrize("method", ("conditional", "recurrent"))
@pytest.mark.parametrize("arm", ("joint", "separate", "fixed_features"))
@pytest.mark.parametrize("support_kind", ("none", "full", "partial"))
def test_trained_full_output_is_bitwise_original_and_every_saved_state_is_unchanged(
        method, arm, support_kind, settings):
    learner, query = make_trained(method, arm, settings)
    layers = ([head[-1] for head in learner.model.residual.heads] if method == "conditional"
              else [learner.model.residual.decoder[-1]])
    assert any(torch.count_nonzero(layer.weight) > 0 for layer in layers)
    support = {"none": None, "full": learner.history,
               "partial": learner.history.take(slice(0, 3))}[support_kind]
    set_mixed_modes(learner)
    state, rng = learner_signature(learner), rng_signature()
    query_hash = query.fingerprint()
    support_hash = None if support is None else support.fingerprint()
    reference = predict_all(learner, query.observations, support)
    actual = component_predictions(learner, query.observations, support)
    keys = {"full", "core", "residual"}
    if method == "conditional":
        keys |= {"full_core_route", "core_full_route"}
    assert set(actual) == keys
    np.testing.assert_array_equal(actual["full"], reference)
    assert not np.array_equal(actual["full"], actual["core"])
    for value in actual.values():
        assert isinstance(value, np.ndarray) and value.dtype == np.float32
        assert value.shape == (len(query), 5, 3) and np.isfinite(value).all()
        assert (value >= 0).all() and (value <= 1 + np.finfo(np.float32).eps).all()
        assert (np.diff(value, axis=-1) <= 0).all()
    if method == "conditional" and support is None:
        np.testing.assert_array_equal(actual["full_core_route"], actual["full"])
        np.testing.assert_array_equal(actual["core_full_route"], actual["core"])
    assert learner_signature(learner) == state
    assert rng_signature() == rng
    assert query.fingerprint() == query_hash
    assert (None if support is None else support.fingerprint()) == support_hash


@pytest.mark.parametrize("method", ("conditional", "recurrent"))
def test_zero_output_insertion_preserves_core_and_full_with_half_probability_residual(method, settings):
    core = make_core(method, settings)
    learner = fork_core(core, "separate", 61, settings)
    query = AcquisitionWorld().experience(Law(0, (0,)), settings["batch_size"], 8641)
    for support in (None, core.history):
        actual = component_predictions(learner, query.observations, support)
        reference = predict_all(core, query.observations, support)
        np.testing.assert_array_equal(actual["full"], reference)
        np.testing.assert_array_equal(actual["core"], reference)
        np.testing.assert_array_equal(actual["residual"], np.full(reference.shape, .5, dtype=np.float32))
        if method == "conditional":
            np.testing.assert_array_equal(actual["full_core_route"], reference)
            np.testing.assert_array_equal(actual["core_full_route"], reference)


def install_conditional_logits(monkeypatch, learner, observations, support, core, residual):
    count = learner.settings["batch_size"] if support is None else len(support)
    expected_support = (np.zeros((count, *observations.shape[1:]), dtype=np.uint8)
                        if support is None else support.observations)
    expected_inputs = np.concatenate((expected_support, observations))
    calls = []

    def forward(name, logits):
        def predict(combined):
            assert not torch.is_grad_enabled()
            np.testing.assert_array_equal(combined.cpu().numpy(), expected_inputs)
            calls.append(name)
            return torch.as_tensor(logits, device=learner.device)
        return predict

    monkeypatch.setattr(learner.model.core, "forward", forward("core", core))
    monkeypatch.setattr(learner.model.residual, "forward", forward("residual", residual))
    return calls


def numpy_route(logits, support, strength):
    experts = logits.shape[1]
    if support is None:
        return np.full(experts, 1 / experts)
    terminal = logits[np.arange(len(support)), :, support.actions, -1].astype(np.float64)
    losses = np.logaddexp(0, terminal) - support.survival[:, -1, None] * terminal
    score = -strength * losses.sum(axis=0)
    unnormalized = np.exp(score - score.max())
    return unnormalized / unnormalized.sum()


def numpy_mix(logits, weights):
    probabilities = 1 / (1 + np.exp(-logits.astype(np.float64)))
    mixed = np.sum(weights[None, :, None, None] * probabilities, axis=1)
    return np.minimum.accumulate(mixed, axis=-1)


@pytest.mark.parametrize("with_support", (False, True))
def test_conditional_all_five_outputs_match_independent_numpy_content_and_evidence_math(
        with_support, settings, monkeypatch):
    learner, query = make_trained("conditional", "joint", settings)
    support = learner.history.take(slice(0, 3)) if with_support else None
    count = settings["batch_size"] if support is None else len(support)
    rng = np.random.default_rng(72831)
    shape = (count + len(query), settings["experts"], 5, 3)
    core = rng.normal(0, 1.1, shape).astype(np.float32)
    residual = rng.normal(0, .8, shape).astype(np.float32)
    calls = install_conditional_logits(monkeypatch, learner, query.observations, support, core, residual)
    actual = component_predictions(learner, query.observations, support)
    full = core + residual  # Preserve the specified float32 logit addition.
    weights = {name: numpy_route(logits[:count], support, settings["evidence_strength"])
               for name, logits in (("core", core), ("residual", residual), ("full", full))}
    pairs = {"core": (core, "core"), "residual": (residual, "residual"), "full": (full, "full"),
             "full_core_route": (full, "core"), "core_full_route": (core, "full")}
    for name, (logits, route) in pairs.items():
        np.testing.assert_allclose(actual[name], numpy_mix(logits[count:], weights[route]),
                                   atol=2e-7, rtol=2e-7)
    assert calls == ["core", "residual"]
    if support is not None:
        assert not np.allclose(weights["core"], weights["full"])
        assert not np.array_equal(actual["full"], actual["full_core_route"])
        assert not np.array_equal(actual["core"], actual["core_full_route"])


def test_monotonicity_projection_occurs_after_expert_probability_mixing(settings, monkeypatch):
    learner, query = make_trained("conditional", "separate", settings)
    count = settings["batch_size"]
    core = np.zeros((count + len(query), settings["experts"], 5, 3), dtype=np.float32)
    for expert in range(settings["experts"]):
        probabilities = np.array([.1, .9, .9] if expert % 2 == 0 else [.9, .1, .9])
        core[count:, expert] = np.log(probabilities / (1 - probabilities))
    residual = np.zeros_like(core)
    install_conditional_logits(monkeypatch, learner, query.observations, None, core, residual)
    actual = component_predictions(learner, query.observations, None)
    expected = np.full((len(query), 5, 3), .5)
    np.testing.assert_allclose(actual["core"], expected, atol=1e-7, rtol=0)
    projected_experts = np.minimum.accumulate(1 / (1 + np.exp(-core[count:].astype(np.float64))), axis=-1)
    wrong_order = projected_experts.mean(axis=1)
    assert np.max(np.abs(actual["core"] - wrong_order)) > .3
    np.testing.assert_array_equal(actual["full"], actual["core"])


def test_conditional_evidence_uses_terminal_reports_only(settings, monkeypatch):
    learner, query = make_trained("conditional", "joint", settings)
    old = learner.history.take(slice(0, 3))
    terminal = np.array([0, 1, 0], dtype=np.uint8)
    outcomes_a = np.repeat(terminal[:, None], 3, axis=1)
    outcomes_b = outcomes_a.copy()
    outcomes_b[:, :2] = 1
    a = Experience(old.observations, old.actions, outcomes_a)
    b = Experience(old.observations, old.actions, outcomes_b)
    rng = np.random.default_rng(623)
    shape = (len(a) + len(query), settings["experts"], 5, 3)
    core = rng.normal(0, 2, shape).astype(np.float32)
    residual = rng.normal(0, 1, shape).astype(np.float32)
    install_conditional_logits(monkeypatch, learner, query.observations, a, core, residual)
    first = component_predictions(learner, query.observations, a)
    second = component_predictions(learner, query.observations, b)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


@pytest.mark.parametrize("method", ("conditional", "recurrent"))
def test_extraction_exception_restores_every_module_mode_and_all_learning_state(method, settings, monkeypatch):
    learner, query = make_trained(method, "joint", settings)
    set_mixed_modes(learner)
    state, rng = learner_signature(learner), rng_signature()

    def fail(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in learner.model.modules())
        raise RuntimeError("synthetic component extraction failure")

    # Fail after the core has already executed, so finally must also cover a
    # partially completed bundle. No scientific checkpoint is involved.
    monkeypatch.setattr(learner.model.residual, "forward" if method == "conditional" else "encode", fail)
    with pytest.raises(RuntimeError, match="component extraction failure"):
        component_predictions(learner, query.observations, learner.history)
    assert learner_signature(learner) == state
    assert rng_signature() == rng


@pytest.mark.parametrize("with_support", (False, True))
def test_recurrent_components_add_logits_before_sigmoid_without_crossing_hidden_contexts(
        with_support, settings, monkeypatch):
    from acp_cl import core_residual_outputs as module

    learner, query = make_trained("recurrent", "joint", settings)
    support = learner.history if with_support else None
    count = settings["batch_size"] if support is None else len(support)
    rng = np.random.default_rng(119)
    core = rng.normal(0, 1, (len(query), 5, 3)).astype(np.float32)
    residual = rng.normal(0, 1, core.shape).astype(np.float32)
    calls = []

    def logits(path, combined, support_size, query_size, actions, outcomes, has_support):
        assert not torch.is_grad_enabled()
        assert support_size == count and query_size == len(query)
        assert len(combined) == count + len(query)
        assert has_support is with_support
        if support is not None:
            np.testing.assert_array_equal(actions.cpu().numpy(), support.actions)
            np.testing.assert_array_equal(outcomes.cpu().numpy(), support.survival)
        else:
            assert torch.count_nonzero(actions) == torch.count_nonzero(outcomes) == 0
        if path is learner.model.core:
            calls.append("core")
            return torch.tensor(core)
        assert path is learner.model.residual
        calls.append("residual")
        return torch.tensor(residual)

    monkeypatch.setattr(module, "_recurrent_logits", logits)
    result = component_predictions(learner, query.observations, support)
    for name, value in (("core", core), ("residual", residual), ("full", core + residual)):
        expected = np.minimum.accumulate(1 / (1 + np.exp(-value.astype(np.float64))), axis=-1)
        np.testing.assert_allclose(result[name], expected, atol=1e-7, rtol=1e-7)
    assert calls == ["core", "residual"]
    assert np.max(np.abs(result["full"] - (result["core"] + result["residual"]) / 2)) > .05


def test_api_accepts_no_query_targets_or_latent_environment_labels(settings):
    assert list(inspect.signature(component_predictions).parameters) == ["learner", "observations", "support"]
    learner, query = make_trained("conditional", "joint", settings)
    with pytest.raises(TypeError):
        component_predictions(learner, query.observations, learner.history, law=Law(1, (0,)))
    with pytest.raises(TypeError):
        component_predictions(learner, query.observations, learner.history, outcomes=query.survival)
