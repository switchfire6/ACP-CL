"""Local post-reset development and exact replacement-count controls."""

import copy
from dataclasses import replace
import math

import pytest
import torch

from acp_cl.models import MLP
from acp_cl.plasticity import PlasticityConfig, PlasticityEngine


@pytest.fixture
def config():
    return PlasticityConfig(
        lr=0.1, momentum=0.0, weight_decay=0.0, anchor_strength=0.0,
        maturity_steps=2, recycle_interval=100, recycle_fraction=0.0,
        adult_recycle_fraction=0.0, newborn_steps=3, core_fraction=0.0,
    )


def make_engine(config, **kwargs):
    torch.manual_seed(173)
    engine = PlasticityEngine(MLP((2,), 2, 3), config, **kwargs)
    engine.steps = 1
    for state in engine.unit_state.values():
        state["age"].fill_(10)
        state["utility"].copy_(torch.arange(len(state["age"]), dtype=torch.float32))
    return engine


def gradients(engine, value=1.0):
    return {name: torch.full_like(parameter, value) for name, parameter in engine.params.items()}


def snapshot(engine):
    return copy.deepcopy((engine.model.state_dict(), engine.state_dict()))


def assert_equal(left, right):
    if isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in right:
            assert_equal(left[key], right[key])
    elif isinstance(right, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right


def test_initial_units_never_receive_newborn_gain_or_local_consolidation(config):
    engine = make_engine(config)
    gates = {"hidden1": 0.2, "hidden2": 0.2}
    for _ in range(6):
        log = engine.step(gradients(engine), gates)
        assert log["effective_feature_gain"] == pytest.approx(0.2)
        assert log["newborn_units"] == log["pending_maturation_units"] == 0
        assert log["total_maturations"] == 0
    assert all(not bool(state["c"].any()) for state in engine.state.values())


def test_diagnostics_count_unit_tensors_separately_from_parameter_state(config):
    engine = make_engine(config)
    # Six units, each with two float32 EMAs and two int64 age counters.
    assert engine.diagnostics()["unit_state_bytes"] == 6 * (2 * 4 + 2 * 8)
    parameter_bytes = sum(
        tensor.numel() * tensor.element_size()
        for state in engine.state.values() for tensor in state.values()
    )
    assert engine.diagnostics()["optimizer_state_bytes"] == parameter_bytes
    engine.recycle(forced_counts={"hidden2": 1})
    assert engine.diagnostics()["unit_state_bytes"] == 144


def test_reset_row_and_outgoing_column_gain_decay_over_exactly_n_updates(config):
    engine = make_engine(config, consolidation=False)
    assert engine.recycle(forced_counts={"hidden1": 1})["recycled"] == 1
    assert engine.diagnostics()["newborn_units"] == 1
    gates = {"hidden1": 0.2, "hidden2": 0.3}
    for index in range(4):
        before = {name: parameter.detach().clone() for name, parameter in engine.params.items()}
        log = engine.step(gradients(engine), gates)
        remaining = max(0.0, 1 - index / 3)
        incoming_gain = 0.2 + 0.8 * remaining
        outgoing_gain = 0.3 + 0.7 * remaining
        torch.testing.assert_close(engine.model.hidden1.weight[0], before["hidden1.weight"][0] - 0.1 * incoming_gain)
        torch.testing.assert_close(engine.model.hidden1.bias[0], before["hidden1.bias"][0] - 0.1 * incoming_gain)
        torch.testing.assert_close(engine.model.hidden1.weight[1:], before["hidden1.weight"][1:] - 0.02)
        torch.testing.assert_close(engine.model.hidden2.weight[:, 0], before["hidden2.weight"][:, 0] - 0.1 * outgoing_gain)
        torch.testing.assert_close(engine.model.hidden2.weight[:, 1:], before["hidden2.weight"][:, 1:] - 0.03)
        torch.testing.assert_close(engine.model.hidden2.bias, before["hidden2.bias"] - 0.03)
        assert log["newborn_units"] == int(index < 2)
    assert engine.total_maturations == 1
    assert engine.total_local_consolidations == 0


def test_feature_gain_is_parameter_weighted_and_displacement_excludes_head(config):
    engine = make_engine(config, consolidation=False)
    engine.recycle(forced_counts={"hidden1": 1})
    before = {name: parameter.detach().clone() for name, parameter in engine.params.items()}
    log = engine.step(gradients(engine), {"hidden1": 0.2, "hidden2": 0.3})
    # 9 parameters in hidden1: 3 reset at 1, 6 at .2. Of hidden2's 12,
    # 3 outgoing-column weights are at 1 and the other 9 parameters at .3.
    assert log["feature_parameter_count"] == 21
    assert log["effective_feature_gain"] == pytest.approx((3 + 6 * 0.2 + 3 + 9 * 0.3) / 21)
    differences = [
        (parameter - before[name]).detach().flatten()
        for name, parameter in engine.params.items() if not name.startswith("head.")
    ]
    assert log["feature_data_displacement"] == pytest.approx(float(torch.cat(differences).norm()))
    assert log["feature_data_displacement"] == pytest.approx(0.1 * math.sqrt(3 + 6 * 0.2**2 + 3 + 9 * 0.3**2))
    assert log["update_norm"] > log["feature_data_displacement"]


def test_data_displacement_does_not_include_anchor_or_weight_decay(config):
    engine = make_engine(replace(config, anchor_strength=1.0, weight_decay=0.1))
    for state in engine.state.values():
        state["c"].fill_(0.2)
        state["anchor"].add_(1.0)
    log = engine.step(gradients(engine, 0.0), {"hidden1": 0.2, "hidden2": 0.2})
    assert log["feature_data_displacement"] == 0.0
    assert log["update_norm"] > 0.0


def test_newborn_zero_disables_local_development_after_actual_reset(config):
    engine = make_engine(replace(config, newborn_steps=0))
    engine.recycle(forced_counts={"hidden2": 1})
    for _ in range(5):
        log = engine.step(gradients(engine), {"hidden1": 0.2, "hidden2": 0.2})
        assert log["effective_feature_gain"] == pytest.approx(0.2)
        assert log["newborn_units"] == log["total_maturations"] == 0
    assert engine.state["hidden2.weight"]["c"][0] == 0


@pytest.mark.parametrize("maturity,newborn", [(1, 3), (5, 2)])
def test_local_consolidation_waits_for_both_windows_without_global_phase(config, maturity, newborn):
    engine = make_engine(replace(config, maturity_steps=maturity, newborn_steps=newborn))
    engine.recycle(forced_counts={"hidden2": 1})
    state = engine.state["hidden2.weight"]
    original_anchor = state["anchor"][0].clone()
    threshold = max(maturity, newborn)
    for index in range(threshold):
        log = engine.step(gradients(engine), {"hidden1": 0.05, "hidden2": 0.05})
        if index + 1 < threshold:
            path = state["path"][0].clone()
            start = state["start"][0].clone()
            # Simulate controller consolidation arriving during local development.
            engine.consolidate(["hidden2"])
            assert state["c"][0] == 0
            torch.testing.assert_close(state["anchor"][0], original_anchor)
            torch.testing.assert_close(state["path"][0], path)
            torch.testing.assert_close(state["start"][0], start)
        else:
            assert state["c"][0] > 0
            assert state["importance"][0] > 0
            assert log["matured_units"] == log["locally_consolidated_units"] == 1
            # First local anchor acquisition copies trained weights fully.
            torch.testing.assert_close(state["anchor"][0], engine.model.hidden2.weight[0])
            assert not torch.equal(state["anchor"][0], original_anchor)
            assert not bool(state["path"][0].any())
    assert engine.total_maturations == engine.total_local_consolidations == 1
    assert engine.diagnostics()["pending_maturation_units"] == 0
    assert not bool(engine.state["head.weight"]["c"].any())


def test_zero_usefulness_graduates_without_consolidating_random_weights(config):
    engine = make_engine(config)
    engine.recycle(forced_counts={"hidden2": 1})
    for _ in range(config.newborn_steps):
        log = engine.step(gradients(engine, 0.0), {"hidden1": 0.05, "hidden2": 0.05})
    assert log["matured_units"] == 1
    assert log["locally_consolidated_units"] == 0
    assert engine.state["hidden2.weight"]["c"][0] == 0
    assert engine.total_maturations == 1


def test_newborn_clock_counts_optimizer_updates_not_utility_observations(config):
    engine = make_engine(config)
    engine.recycle(forced_counts={"hidden2": 1})
    for _ in range(20):
        engine.update_utilities({"hidden1": torch.ones(2, 3), "hidden2": torch.ones(2, 3)})
    assert engine.unit_state["hidden2"]["newborn_age"][0] == 0
    assert engine.total_maturations == 0
    with pytest.raises(ValueError, match="eligible 2"):
        engine.recycle(forced_counts={"hidden2": 3})


def test_explicit_counts_override_fraction_interval_and_preserve_credit(config):
    engine = make_engine(config)
    engine.unit_state["hidden1"]["credit"] = 0.75
    result = engine.recycle(adult=True, forced_counts={"hidden1": 2})
    assert result == {"recycled": 2, "recycled_by_population": {"hidden1": 2}}
    assert engine.unit_state["hidden1"]["credit"] == 0.75
    assert engine.total_recycled == 2
    assert engine.diagnostics()["newborn_units"] == 2


def test_explicit_empty_counts_suppress_scheduled_replacements(config):
    engine = make_engine(replace(config, recycle_interval=1, recycle_fraction=1.0))
    before = snapshot(engine)
    assert engine.recycle(forced_counts={}) == {"recycled": 0, "recycled_by_population": {}}
    assert_equal(snapshot(engine), before)
    assert engine.recycle()["recycled"] == 6


@pytest.mark.parametrize("counts", [{"missing": 1}, {"hidden1": -1}, {"hidden1": 1.5}, {"hidden1": True}, []])
def test_invalid_forced_counts_fail_atomically(config, counts):
    engine = make_engine(config, random_recycling=True)
    before = snapshot(engine)
    with pytest.raises(ValueError):
        engine.recycle(forced_counts=counts)
    assert_equal(snapshot(engine), before)


def test_unfulfillable_second_population_does_not_partially_reset_first(config):
    engine = make_engine(config, random_recycling=True)
    before = snapshot(engine)
    with pytest.raises(ValueError, match="hidden1: requested 4, eligible 3"):
        engine.recycle(forced_counts={"hidden2": 1, "hidden1": 4})
    assert_equal(snapshot(engine), before)


@pytest.mark.parametrize("safeguard", ["immature", "incoming", "downstream", "disabled"])
def test_forced_counts_retain_maturity_and_protection_safeguards(config, safeguard):
    engine = make_engine(config)
    if safeguard == "immature":
        engine.unit_state["hidden1"]["age"].zero_()
    elif safeguard == "incoming":
        engine.state["hidden1.bias"]["c"].fill_(config.recycle_cutoff)
    elif safeguard == "downstream":
        engine.state["hidden2.weight"]["c"][0] = config.recycle_cutoff
    else:
        engine.recycling = False
    before = snapshot(engine)
    with pytest.raises(ValueError, match="cannot fulfill"):
        engine.recycle(forced_counts={"hidden1": 1})
    assert_equal(snapshot(engine), before)


def test_simultaneous_resets_leave_all_outgoing_columns_zero(config):
    engine = make_engine(config)
    result = engine.recycle(forced_counts={"hidden1": 1, "hidden2": 1})
    assert result["recycled"] == 2
    assert not bool(engine.model.hidden2.weight[:, 0].any())
    assert not bool(engine.model.head.weight[:, 0].any())
    assert engine.diagnostics()["newborn_units"] == 2
    engine.step(gradients(engine, 0.0), {"hidden1": 0.05, "hidden2": 0.05})
    assert not bool(engine.model.hidden2.weight[:, 0].any())
    assert not bool(engine.model.head.weight[:, 0].any())


def test_checkpoint_restores_active_newborn_gains_maturation_and_rng(config):
    engine = make_engine(config, random_recycling=True, seed=103)
    engine.recycle(forced_counts={"hidden2": 1})
    engine.step(gradients(engine), {"hidden1": 0.05, "hidden2": 0.05})
    restored = make_engine(config, random_recycling=True, seed=902)
    model_state, engine_state = snapshot(engine)
    restored.model.load_state_dict(model_state)
    restored.load_state_dict(engine_state)
    for _ in range(4):
        assert_equal(
            engine.step(gradients(engine), {"hidden1": 0.05, "hidden2": 0.05}),
            restored.step(gradients(restored), {"hidden1": 0.05, "hidden2": 0.05}),
        )
        assert_equal(snapshot(engine), snapshot(restored))
    assert engine.total_maturations == 1
    assert_equal(engine.recycle(forced_counts={"hidden1": 1}), restored.recycle(forced_counts={"hidden1": 1}))
    assert_equal(snapshot(engine), snapshot(restored))


def test_v0_checkpoint_loads_with_feature_disabled(config):
    config = replace(config, newborn_steps=0)
    engine = make_engine(config)
    state = copy.deepcopy(engine.state_dict())
    for key in ("newborn_steps", "total_maturations", "total_local_consolidations"):
        del state[key]
    for population in state["units"].values():
        del population["newborn_age"]
    restored = make_engine(config)
    restored.load_state_dict(state)
    assert_equal(restored.state_dict(), engine.state_dict())


def test_mismatched_newborn_checkpoint_rejected_before_state_mutation(config):
    source = make_engine(config)
    target = make_engine(replace(config, newborn_steps=2))
    before = snapshot(target)
    with pytest.raises(ValueError, match="newborn_steps"):
        target.load_state_dict(source.state_dict())
    assert_equal(snapshot(target), before)


@pytest.mark.parametrize("value", [-1, 1.5, True, float("nan")])
def test_invalid_newborn_duration_rejected(value):
    with pytest.raises(ValueError, match="newborn_steps"):
        PlasticityConfig(newborn_steps=value)
