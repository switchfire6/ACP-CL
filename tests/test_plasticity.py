"""Numerical contracts for gated updates, consolidation, replacement, and resume."""

from collections import OrderedDict
import copy
from dataclasses import replace
import math

import pytest
import torch
from torch import nn

from acp_cl.learner import Learner
from acp_cl.models import MLP, UnitSpec
from acp_cl.plasticity import PlasticityConfig, PlasticityEngine


class TinyNetwork(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.hidden = nn.Linear(2, width)
        self.head = nn.Linear(width, 2)

    def forward(self, x, return_features=False):
        features = torch.relu(self.hidden(x))
        logits = self.head(features)
        if return_features:
            return logits, {"hidden": features.detach(), "representation": features.detach()}
        return logits

    def plastic_blocks(self):
        return OrderedDict(hidden=self.hidden, head=self.head)

    def recyclable_units(self):
        return [UnitSpec("hidden", "hidden", self.hidden, self.head)]


@pytest.fixture
def config():
    return PlasticityConfig(
        lr=0.1, momentum=0.9, weight_decay=0.0, anchor_strength=0.0,
        maturity_steps=2, recycle_interval=1, recycle_fraction=0.25,
    )


def make_engine(config, **kwargs):
    torch.manual_seed(313)
    return PlasticityEngine(TinyNetwork(), config, **kwargs)


def gradients(engine, value):
    return {name: torch.full_like(parameter, value) for name, parameter in engine.params.items()}


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_gate_applies_after_momentum_including_existing_velocity(config):
    engine = make_engine(replace(config, g_min=0.0))
    engine.step(gradients(engine, 1.0), {"hidden": 1.0})
    before = {name: parameter.detach().clone() for name, parameter in engine.params.items()}
    engine.step(gradients(engine, 0.0), {"hidden": 0.0, "head": 0.0})
    torch.testing.assert_close(engine.model.hidden.weight, before["hidden.weight"])
    torch.testing.assert_close(engine.state["hidden.weight"]["momentum"], torch.full_like(engine.model.hidden.weight, 0.9))
    # The classifier receives full momentum even if its external gate is zero.
    torch.testing.assert_close(engine.model.head.weight, before["head.weight"] - 0.09)
    engine.step(gradients(engine, 0.0), {"hidden": 0.5})
    torch.testing.assert_close(engine.model.hidden.weight, before["hidden.weight"] - 0.0405)


@pytest.mark.parametrize("gate,c", [(1.0, 1.0), (0.05, 1.0), (0.05, 0.0), (0.05, 0.7)])
def test_final_data_update_multiplier_has_floor(config, gate, c):
    engine = make_engine(config)
    engine.state["hidden.weight"]["c"].fill_(c)
    before = engine.model.hidden.weight.detach().clone()
    engine.step(gradients(engine, 1.0), {"hidden": gate})
    torch.testing.assert_close(engine.model.hidden.weight, before - config.lr * config.g_min)


def test_anchor_force_is_independent_of_closed_data_gate(config):
    engine = make_engine(replace(config, g_min=0.0, anchor_strength=2.0))
    with torch.no_grad():
        engine.model.hidden.weight.fill_(3.0)
        engine.state["hidden.weight"]["anchor"].fill_(1.0)
        engine.state["hidden.weight"]["c"].fill_(0.5)
    engine.step(gradients(engine, 0.0), {"hidden": 0.0})
    torch.testing.assert_close(engine.model.hidden.weight, torch.full_like(engine.model.hidden.weight, 2.8))


def test_importance_uses_window_start_separate_from_regularization_anchor(config):
    engine = make_engine(replace(config, importance_epsilon=0.5, anchor_rate=0.5))
    state = engine.state["hidden.weight"]
    with torch.no_grad():
        engine.model.hidden.weight.fill_(2.0)
        state["start"].fill_(1.0)
        state["anchor"].fill_(100.0)
        state["path"].fill_(3.0)
        state["age"].fill_(config.maturity_steps)
    engine.consolidate(["hidden"])
    # Two weights per row: path=6, window displacement=2, epsilon=0.5.
    torch.testing.assert_close(state["importance"], torch.full((4,), 2.4))
    torch.testing.assert_close(state["anchor"], torch.full_like(state["anchor"], 51.0))
    torch.testing.assert_close(state["start"], engine.model.hidden.weight)
    assert not bool(state["path"].any())
    torch.testing.assert_close(state["c"], torch.full((4,), 1 - math.exp(-config.consolidation_rate)))


def test_immature_rows_cannot_consolidate_or_move_anchor(config):
    engine = make_engine(config)
    state = engine.state["hidden.weight"]
    original_anchor = state["anchor"].clone()
    with torch.no_grad():
        engine.model.hidden.weight.add_(0.5)
        state["path"].fill_(1.0)
        state["age"][1:] = config.maturity_steps
    engine.consolidate(["hidden"])
    assert state["c"][0] == 0
    assert bool((state["c"][1:] > 0).all())
    torch.testing.assert_close(state["anchor"][0], original_anchor[0])
    assert not torch.equal(state["anchor"][1:], original_anchor[1:])


def test_no_head_consolidation_even_when_requested(config):
    engine = make_engine(config)
    for _ in range(config.maturity_steps):
        engine.step(gradients(engine, 1.0), {"hidden": 1.0})
    engine.consolidate(["hidden", "head"])
    assert bool((engine.state["hidden.weight"]["c"] > 0).all())
    for name in ("head.weight", "head.bias"):
        assert not bool(engine.state[name]["c"].any())
        assert not bool(engine.state[name]["importance"].any())
        assert not bool(engine.state[name]["path"].any())


def test_zero_importance_rows_do_not_become_arbitrary_hard_core(config):
    engine = make_engine(replace(config, relax_max=0.5, core_fraction=0.25))
    state = engine.state["hidden.weight"]
    state["c"].fill_(0.6)
    state["importance"].zero_()
    effective = engine._effective_c("hidden.weight", {"hidden"})
    torch.testing.assert_close(effective, torch.full((4,), 0.3))


def test_positive_core_is_protected_and_relaxation_ablation_holds_c_fixed(config):
    engine = make_engine(replace(config, relax_max=0.5, core_fraction=0.5))
    state = engine.state["hidden.weight"]
    state["c"].fill_(0.6)
    state["importance"].copy_(torch.tensor([0.1, 0.2, 0.3, 10.0]))
    effective = engine._effective_c("hidden.weight", {"hidden"})
    torch.testing.assert_close(effective[2:], state["c"][2:])
    assert bool((effective[:2] < state["c"][:2]).all())
    engine.relaxation = False
    torch.testing.assert_close(engine._effective_c("hidden.weight", {"hidden"}), state["c"])


def prepare_recycling(engine):
    engine.steps = engine.config.recycle_interval
    for name, state in engine.state.items():
        state["momentum"].fill_(2.0)
        state["path"].fill_(3.0)
        state["importance"].fill_(4.0)
        state["age"].fill_(5)
        state["anchor"].fill_(8.0)
        state["start"].fill_(9.0)
        state["c"].fill_(0.1 if not name.startswith("head.") else 0.0)
    unit = engine.unit_state["hidden"]
    unit["age"].fill_(5)
    unit["utility"].copy_(torch.tensor([0.9, 0.2, 0.5, 0.7]))
    unit["mean"].fill_(1.0)


def test_recycling_resets_all_incident_state_and_does_not_resurrect_output_column(config):
    engine = make_engine(replace(config, anchor_strength=1.0, weight_decay=0.1))
    prepare_recycling(engine)
    before = engine.model.hidden.weight.detach().clone()
    result = engine.recycle()
    assert result["recycled"] == 1
    chosen = 1  # Lowest utility among equally old, unprotected units.
    assert not torch.equal(engine.model.hidden.weight[chosen], before[chosen])
    torch.testing.assert_close(engine.model.hidden.weight[[0, 2, 3]], before[[0, 2, 3]])
    assert engine.model.hidden.bias[chosen] == 0
    assert not bool(engine.model.head.weight[:, chosen].any())
    for name in ("hidden.weight", "hidden.bias"):
        state = engine.state[name]
        for key in ("momentum", "path", "c", "importance", "age"):
            assert not bool(state[key][chosen].any()), (name, key)
        for key in ("anchor", "start"):
            torch.testing.assert_close(state[key][chosen], engine.params[name][chosen])
    for key in ("momentum", "path", "anchor", "start"):
        assert not bool(engine.state["head.weight"][key][:, chosen].any()), key
    for key in ("utility", "mean", "age"):
        assert engine.unit_state["hidden"][key][chosen] == 0
    engine.step(gradients(engine, 0.0), {"hidden": 1.0})
    assert not bool(engine.model.head.weight[:, chosen].any())


def test_consolidation_protects_incoming_weight_and_bias_rows(config):
    engine = make_engine(config)
    prepare_recycling(engine)
    engine.state["hidden.weight"]["c"][1] = config.recycle_cutoff
    engine.state["hidden.bias"]["c"][2] = config.recycle_cutoff
    # Two eligible rows generate half a replacement credit per check.
    assert engine.recycle()["recycled"] == 0
    before = engine.model.hidden.weight.detach().clone()
    assert engine.recycle()["recycled"] == 1
    torch.testing.assert_close(engine.model.hidden.weight[[1, 2]], before[[1, 2]])
    assert not torch.equal(engine.model.hidden.weight[3], before[3])


def test_protected_downstream_row_vetoes_upstream_replacement(config):
    torch.manual_seed(19)
    model = MLP((2,), 2, 4)
    engine = PlasticityEngine(model, config)
    engine.steps = 1
    for state in engine.unit_state.values():
        state["age"].fill_(5)
    engine.state["hidden2.weight"]["c"][0] = config.recycle_cutoff
    before = model.hidden1.weight.detach().clone()
    result = engine.recycle()
    assert "hidden1" not in result["recycled_by_population"]
    torch.testing.assert_close(model.hidden1.weight, before)


def test_maturity_and_fractional_credit_allow_small_populations_to_recycle(config):
    engine = make_engine(replace(config, recycle_fraction=0.125))
    features = {"hidden": torch.ones(2, 4)}
    counts = []
    for _ in range(3):
        engine.step(gradients(engine, 0.0), {"hidden": 1.0})
        engine.update_utilities(features)
        counts.append(engine.recycle()["recycled"])
    assert counts == [0, 0, 1]
    assert engine.unit_state["hidden"]["credit"] == pytest.approx(0.0)
    assert engine.total_recycled == 1


def test_adult_recycling_uses_lower_fraction_and_disabled_engine_is_unchanged(config):
    config = replace(config, recycle_fraction=0.5, adult_recycle_fraction=0.25)
    open_engine, adult_engine = make_engine(config), make_engine(config)
    prepare_recycling(open_engine)
    prepare_recycling(adult_engine)
    assert open_engine.recycle()["recycled"] == 2
    assert adult_engine.recycle(adult=True)["recycled"] == 1
    adult_engine.recycling = False
    before = copy.deepcopy(adult_engine.state_dict())
    assert adult_engine.recycle()["recycled"] == 0
    assert_nested_equal(adult_engine.state_dict(), before)


def test_recycling_respects_interval(config):
    engine = make_engine(replace(config, recycle_interval=3))
    prepare_recycling(engine)
    engine.steps = 2
    assert engine.recycle()["recycled"] == 0
    assert engine.unit_state["hidden"]["credit"] == 0
    engine.steps = 3
    assert engine.recycle()["recycled"] == 1


def test_engine_checkpoint_restores_recycling_rng_and_future_updates(config):
    engine = make_engine(config, random_recycling=True, seed=171)
    prepare_recycling(engine)
    restored = make_engine(config, random_recycling=True, seed=999)
    restored.model.load_state_dict(copy.deepcopy(engine.model.state_dict()))
    restored.load_state_dict(copy.deepcopy(engine.state_dict()))
    for _ in range(3):
        assert engine.recycle() == restored.recycle()
        assert_nested_equal(engine.model.state_dict(), restored.model.state_dict())
        for current in (engine, restored):
            current.step(gradients(current, 0.2), {"hidden": 0.7})
            current.update_utilities({"hidden": torch.ones(2, 4)})
        assert_nested_equal(engine.state_dict(), restored.state_dict())


def test_nonfinite_gradient_rejection_does_not_partially_update_model(config):
    engine = make_engine(config)
    bad_gradients = gradients(engine, 1.0)
    bad_gradients["head.bias"][0] = float("nan")
    state_before = copy.deepcopy(engine.state_dict())
    model_before = copy.deepcopy(engine.model.state_dict())
    with pytest.raises(FloatingPointError):
        engine.step(bad_gradients, {"hidden": 1.0})
    assert_nested_equal(engine.model.state_dict(), model_before)
    assert_nested_equal(engine.state_dict(), state_before)


@pytest.mark.parametrize("gate", [float("nan"), -0.1, 1.1])
def test_invalid_gate_rejection_does_not_partially_update_model(config, gate):
    engine = make_engine(config)
    state_before = copy.deepcopy(engine.state_dict())
    model_before = copy.deepcopy(engine.model.state_dict())
    with pytest.raises(ValueError):
        engine.step(gradients(engine, 1.0), {"hidden": gate})
    assert_nested_equal(engine.model.state_dict(), model_before)
    assert_nested_equal(engine.state_dict(), state_before)


def learner_config():
    return {
        "data": {"dataset": "synthetic"},
        "replay_capacity": 8, "replay_batch_size": 2, "probe_batch_size": 2,
        "controller": {
            "monitor_interval": 2, "scaffold_steps": 2, "min_open_steps": 2,
            "stable_windows": 2, "stability_loss_tol": 10.0,
            "stability_drift_tol": 10.0, "closing_windows": 2,
            "novel_windows": 1, "reopen_max_windows": 2, "cooldown_windows": 0,
        },
        "plasticity": {"lr": 0.03, "maturity_steps": 1, "recycle_interval": 2,
                       "recycle_fraction": 0.25},
    }


def test_learner_checkpoint_resumes_in_middle_of_monitoring_window():
    torch.manual_seed(271)
    config = learner_config()
    learner = Learner(TinyNetwork(), "acp", config, seed=44)
    batches = [
        (torch.tensor([[0.5 + i / 10, 1.0], [-1.0, 0.2 - i / 20]]), torch.tensor([0, 1]))
        for i in range(9)
    ]
    for x, y in batches[:5]:
        learner.train_batch(x, y)
    restored = Learner(TinyNetwork(), "acp", config, seed=938)
    restored.load_state_dict(copy.deepcopy(learner.state_dict()))
    for x, y in batches[5:]:
        assert_nested_equal(restored.train_batch(x, y), learner.train_batch(x, y))
        assert_nested_equal(restored.model.state_dict(), learner.model.state_dict())
    assert_nested_equal(restored.state_dict(), learner.state_dict())


def test_learner_rejects_mismatched_optimizer_config_before_mutating_model():
    source_config = learner_config()
    target_config = copy.deepcopy(source_config)
    target_config["plasticity"]["lr"] = 0.9
    source = Learner(TinyNetwork(), "acp", source_config)
    target = Learner(TinyNetwork(), "acp", target_config)
    before = copy.deepcopy(target.model.state_dict())
    with pytest.raises(ValueError, match="config"):
        target.load_state_dict(copy.deepcopy(source.state_dict()))
    assert_nested_equal(target.model.state_dict(), before)
