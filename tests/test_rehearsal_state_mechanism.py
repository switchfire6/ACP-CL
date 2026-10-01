"""Exact state-factor donor ownership, endpoint identities and matched work."""

import copy
import hashlib
import itertools
import pickle
import random

import numpy as np
import pytest
import torch

from acp_cl.acquisition.world import AcquisitionWorld, Law
from acp_cl.conditional.learner import Packet
from acp_cl.contextual.learner import ContextLearner
from acp_cl.predictive_value.mechanism import clone_exact, make_shadow, predict_all, state_signature
from acp_cl.predictive_value.study import load, save
from acp_cl.rehearsal_state import mechanism
from acp_cl.replay_renewal.memory import ReservoirMemory, memory_state
from acp_cl.replay_renewal.study import new_learner


@pytest.fixture
def settings():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return dict(width=8, experts=4, context_width=4, decoder_width=8,
        interaction_features=True, lr=.002, memory_packets=4, batch_size=8,
        evidence_strength=1., updates_per_batch=2)


def data(index):
    return AcquisitionWorld().experience(Law(index % 2, (0,) if index > 1 else ()), 8, 153910+index)


def pair(model, settings):
    old = new_learner(model, 15391, settings)
    old.train(data(0))
    old.train(data(1))
    old.model.frame.eval()
    new = clone_exact(old)
    new.train(data(2))
    new.train(data(3))
    new.model.temporal.eval()
    return old, new, Packet(data(0), data(1)), Packet(data(2), data(3)), old.memory.packets[0]


def rng_signature():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
        torch.get_rng_state().numpy()), protocol=5)).hexdigest()


def tensor_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            tensor_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            tensor_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
@pytest.mark.parametrize("weights,optimizer", tuple(itertools.product((0, 1), repeat=2)))
def test_transplant_exact_donors_owned_storage_and_canonical_inactive_state(model, weights, optimizer, settings):
    old, new, *_ = pair(model, settings)
    # Complete group state must follow O, not the configuration's original lr.
    new.optimizer.param_groups[0]["lr"] = .001
    parents = state_signature(old), state_signature(new)
    rng = rng_signature()
    copied = mechanism.transplant(old, new, weights, optimizer)
    weight_donor, optimizer_donor = (old, new)[weights], (old, new)[optimizer]
    tensor_equal(copied.model.state_dict(), weight_donor.model.state_dict())
    tensor_equal(copied.optimizer.state_dict(), optimizer_donor.optimizer.state_dict())
    assert [m.training for m in copied.model.modules()] == [m.training for m in weight_donor.model.modules()]
    model_ids = {id(p) for p in copied.model.parameters()}
    assert {id(p) for group in copied.optimizer.param_groups for p in group["params"]} == model_ids
    assert {id(p) for p in copied.optimizer.state} == model_ids
    for copied_parameter, donor_parameter in zip(copied.model.parameters(), weight_donor.model.parameters()):
        assert copied_parameter.requires_grad == donor_parameter.requires_grad
        assert torch.equal(copied_parameter.grad, donor_parameter.grad)
        assert copied_parameter.data_ptr() != donor_parameter.data_ptr()
        assert copied_parameter.grad.data_ptr() != donor_parameter.grad.data_ptr()
    for copy_state, donor_state in zip(copied.optimizer.state.values(), optimizer_donor.optimizer.state.values()):
        for key in copy_state:
            if isinstance(copy_state[key], torch.Tensor):
                assert copy_state[key].data_ptr() != donor_state[key].data_ptr()
    assert mechanism._inactive_signature(copied) == mechanism._inactive_signature(old)
    assert not np.shares_memory(copied.history.observations, old.history.observations)
    assert copied.initial_encoder is not old.initial_encoder
    assert rng_signature() == rng
    assert (state_signature(old), state_signature(new)) == parents
    copied.train(data(4))
    assert (state_signature(old), state_signature(new)) == parents


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
@pytest.mark.parametrize("cell", ["".join(bits) for bits in itertools.product("01", repeat=3)])
def test_all_cells_match_independent_adam_rebinding_and_original_rehearsal(model, cell, settings):
    old, new, anchor_old, anchor_new, replay = pair(model, settings)
    donors = (old, new)
    weights, optimizer, anchor = map(int, cell)
    shadow, work = mechanism.make_factor_shadow(old, new, anchor_old, anchor_new, replay, cell, 2)
    # Reference begins from W itself, builds a separate Adam, then calls the
    # unchanged original rehearsal. It does not use transplant or factor code.
    reference = clone_exact(donors[weights])
    reference.optimizer = torch.optim.Adam(reference.model.parameters(), lr=.123)
    reference.optimizer.load_state_dict(copy.deepcopy(donors[optimizer].optimizer.state_dict()))
    expected, _ = make_shadow(reference, (anchor_old, anchor_new)[anchor], replay, 2)
    assert mechanism.active_signature(shadow) == mechanism.active_signature(expected)
    assert mechanism._inactive_signature(shadow) == mechanism._inactive_signature(old)
    assert work["cell"] == cell and (work["weights_bit"], work["optimizer_bit"], work["anchor_bit"]) == (weights, optimizer, anchor)
    assert work["weights_donor_sha256"] == state_signature(donors[weights])
    assert work["optimizer_donor_sha256"] == state_signature(donors[optimizer])
    assert work["final_active_sha256"] == mechanism.active_signature(shadow)
    assert work["donors_unchanged"] and work["inactive_unchanged"]
    assert work["steps"] == work["backward_calls"] == 2 and work["forwards"] == 4
    assert work["diagnostic_forwards"] == 4
    assert work["diagnostic_query_presentations"] == work["diagnostic_support_presentations"] == 32
    assert work["transfer_seconds"] >= 0 and work["diagnostic_seconds"] >= 0
    delta = np.concatenate([(b.detach().numpy().astype(np.float64)-a.detach().numpy().astype(np.float64)).ravel()
                            for a, b in zip(donors[weights].model.parameters(), shadow.model.parameters())])
    assert work["update_l2"] == pytest.approx(np.linalg.norm(delta), rel=1e-14)
    for before, after in zip(donors[optimizer].optimizer.state.values(), shadow.optimizer.state.values()):
        assert float(after["step"]) == float(before["step"])+2


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_endpoint_identities_and_fixed_context_predictions(model, settings):
    old, new, anchor_old, anchor_new, replay = pair(model, settings)
    for cell, parent, anchor in (("000", old, anchor_old), ("111", new, anchor_new)):
        shadow, _ = mechanism.make_factor_shadow(old, new, anchor_old, anchor_new, replay, cell, 3)
        ordinary, _ = make_shadow(parent, anchor, replay, 3)
        assert mechanism.active_signature(shadow) == mechanism.active_signature(ordinary)
        assert np.array_equal(predict_all(shadow, data(4).observations, data(3)),
                              predict_all(ordinary, data(4).observations, data(3)))
        if cell == "000":
            assert state_signature(shadow) == state_signature(ordinary)
        else:
            assert state_signature(shadow) != state_signature(ordinary)
            assert shadow.cost == old.cost and ordinary.cost == new.cost


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_four_diagnostic_forwards_do_not_change_gradients_rng_modes_or_update_path(model, settings, monkeypatch):
    old, new, anchor_old, anchor_new, replay = pair(model, settings)
    states, rng = (state_signature(old), state_signature(new)), rng_signature()
    calls = []
    original = ContextLearner.probabilities

    def counted(self, observations, support, *args, **kwargs):
        calls.append((torch.is_grad_enabled(), None if support is None else support.fingerprint()))
        return original(self, observations, support, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("factor rehearsal must not sample or add memory")

    monkeypatch.setattr(ContextLearner, "probabilities", counted)
    monkeypatch.setattr(ReservoirMemory, "sample", forbidden)
    monkeypatch.setattr(ReservoirMemory, "add", forbidden)
    shadow, work = mechanism.make_factor_shadow(old, new, anchor_old, anchor_new, replay, "101", 2)
    assert [grad for grad, _ in calls] == [False, False, True, True, True, True, False, False]
    assert [support for _, support in calls] == [anchor_new.support.fingerprint(), None]*4
    assert (state_signature(old), state_signature(new)) == states and rng_signature() == rng
    assert memory_state(shadow.memory) == memory_state(old.memory)
    assert shadow.history.fingerprint() == old.history.fingerprint() and shadow.cost == old.cost
    initial = mechanism.transplant(old, new, 1, 0)
    before = mechanism._loss_pair(initial, anchor_new, replay)
    after = mechanism._loss_pair(shadow, anchor_new, replay)
    assert work["losses"] == dict(anchor_before=before[0], replay_before=before[1], anchor_after=after[0], replay_after=after[1])


def test_active_signature_excludes_inactive_fields_and_includes_gradients_modes_and_steps(settings):
    old, *_ = pair("conditional", settings)
    copied = clone_exact(old)
    expected = mechanism.active_signature(copied)
    copied.history = data(9)
    copied.memory.sample()
    copied.cost["arrivals"] += 8
    copied.initial_hash = "9"*64
    assert mechanism.active_signature(copied) == expected
    first = next(copied.model.parameters())
    first.grad.add_(1)
    assert mechanism.active_signature(copied) != expected
    copied = clone_exact(old)
    copied.model.frame.train(not copied.model.frame.training)
    assert mechanism.active_signature(copied) != expected
    copied = clone_exact(old)
    next(iter(copied.optimizer.state.values()))["step"].add_(1)
    assert mechanism.active_signature(copied) != expected


def test_trainability_is_copied_but_rehearsal_requires_all_weights_trainable(settings):
    old, new, a, b, replay = pair("conditional", settings)
    next(new.model.parameters()).requires_grad_(False)
    assert not next(mechanism.transplant(old, new, 1, 0).model.parameters()).requires_grad
    with pytest.raises(ValueError, match="trainable"):
        mechanism.make_factor_shadow(old, new, a, b, replay, "100", 1)


@pytest.mark.parametrize("fault", ("settings", "initial", "method", "dtype", "module", "order", "missing_parameter", "foreign_parameter", "adam", "device", "moment_shape", "moment_dtype", "step", "missing_moment"))
def test_mismatched_or_misowned_donors_rejected_before_transplant(settings, fault):
    old, new, *_ = pair("conditional", settings)
    if fault == "settings":
        new.settings["lr"] *= 2
    elif fault == "initial":
        new.initial_hash = "f"*64
    elif fault == "method":
        new.method = "recurrent"
    elif fault == "dtype":
        new.model.double()
    elif fault == "module":
        new.model.frame[1] = torch.nn.ReLU()
    elif fault == "order":
        new.optimizer.param_groups[0]["params"].reverse()
    elif fault == "missing_parameter":
        new.optimizer.param_groups[0]["params"].pop()
    elif fault == "foreign_parameter":
        new.optimizer.param_groups[0]["params"][0] = torch.nn.Parameter(torch.zeros(1))
    elif fault == "adam":
        new.optimizer = torch.optim.SGD(new.model.parameters(), lr=.01)
    elif fault == "device":
        new.device = torch.device("meta")
    elif fault == "moment_shape":
        next(iter(new.optimizer.state.values()))["exp_avg"] = torch.zeros(1)
    elif fault == "moment_dtype":
        state = next(iter(new.optimizer.state.values()))
        state["exp_avg"] = state["exp_avg"].double()
    elif fault == "step":
        next(iter(new.optimizer.state.values()))["step"] = torch.tensor(-1.)
    else:
        next(iter(new.optimizer.state.values())).pop("exp_avg_sq")
    with pytest.raises(ValueError):
        mechanism.transplant(old, new, 1, 1)


@pytest.mark.parametrize("fault", ("parameter", "gradient", "moment", "initial_encoder"))
def test_nonfinite_state_rejected(settings, fault):
    old, new, *_ = pair("conditional", settings)
    if fault == "parameter":
        target = next(new.model.parameters()).data
    elif fault == "gradient":
        target = next(new.model.parameters()).grad
    elif fault == "moment":
        target = next(iter(new.optimizer.state.values()))["exp_avg"]
    else:
        target = next(iter(new.initial_encoder.values()))
    target.flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        mechanism.transplant(old, new, 1, 1)


@pytest.mark.parametrize("cell,updates", (("00", 1), ("002", 1), (111, 1), ("000", 0), ("000", True)))
def test_invalid_cells_and_update_budgets_rejected(settings, cell, updates):
    old, new, a, b, replay = pair("conditional", settings)
    with pytest.raises(ValueError):
        mechanism.make_factor_shadow(old, new, a, b, replay, cell, updates)


@pytest.mark.parametrize("bits", ((True, 0), (0, -1), (1, "0")))
def test_bit_api_rejects_boolean_or_nonbinary_values(settings, bits):
    old, new, *_ = pair("conditional", settings)
    with pytest.raises(ValueError, match="bits"):
        mechanism.transplant(old, new, *bits)


def test_no_latent_labels_allowed_even_in_unused_anchor(settings):
    old, new, a, b, replay = pair("conditional", settings)
    b = copy.deepcopy(b)
    b.oracle_modes = np.zeros(8, dtype=int)
    with pytest.raises(ValueError, match="latent"):
        mechanism.make_factor_shadow(old, new, a, b, replay, "000", 1)


def test_diagnostic_exception_restores_every_module_mode_and_gradient(settings, monkeypatch):
    old, _, a, _, replay = pair("conditional", settings)
    before = state_signature(old)

    def broken(*args):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(mechanism, "_packet_loss", broken)
    with pytest.raises(RuntimeError, match="synthetic"):
        mechanism._loss_pair(old, a, replay)
    assert state_signature(old) == before


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_checkpoint_roundtrip_preserves_hybrid_gradients_moments_and_canonical_state(model, settings, tmp_path):
    old, new, a, b, replay = pair(model, settings)
    shadow, work = mechanism.make_factor_shadow(old, new, a, b, replay, "010", 2)
    path = tmp_path/"shadow.pt"
    save(path, dict(identity={"test": "hybrid"}, learner=shadow, work=work))
    restored = load(path, {"test": "hybrid"})
    assert state_signature(restored["learner"]) == state_signature(shadow)
    assert mechanism.active_signature(restored["learner"]) == work["final_active_sha256"]
    assert restored["work"] == work
