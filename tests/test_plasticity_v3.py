"""Scientific allocation invariants for the isolated, causally scheduled V3 engine."""

import copy
from dataclasses import replace
import math

import pytest
import torch

from acp_cl.models import MLP
from acp_cl.plasticity import PlasticityConfig, row_view
from acp_cl.plasticity_v3 import ScheduledPlasticityEngine, V3Config, V3_METHODS


def make_engine(variant="recycle_v3", *, width=12, pc=None, cfg=None):
    torch.manual_seed(173)
    pc = pc or PlasticityConfig(
        lr=0.1, momentum=0.7, weight_decay=0.0, anchor_strength=0.0,
        g_min=0.2, maturity_steps=2, recycle_interval=99,
        recycle_fraction=0.0, adult_recycle_fraction=0.0,
    )
    cfg = cfg or V3Config(
        feature_gain=0.4, warmup_steps=2, newborn_steps=3, newborn_peak=1.0,
        protection_steps=6, reset_interval=2, reset_count=1, max_update_norm=100.0,
    )
    return ScheduledPlasticityEngine(MLP((2,), 2, width), pc, cfg, variant=variant, seed=83)


def gradients(engine, value=0.25):
    return {name: torch.full_like(p, value) for name, p in engine.params.items()}


def advance(engine, value=0.25, *, recycle=True):
    event = engine.step(gradients(engine, value), {}, ())
    engine.update_utilities({u.name: torch.ones(3, u.incoming.out_features) for u in engine.units})
    reset = engine.recycle() if recycle else None
    return event, reset


def prime_reset(engine, *, reverse=False):
    """Start at the first reset boundary with all original units eligible."""
    engine.steps = engine.allocation_config.warmup_steps + engine.allocation_config.reset_interval
    for state in engine.unit_state.values():
        state["age"].fill_(max(1, engine.config.maturity_steps))
        values = torch.arange(len(state["age"]), dtype=state["utility"].dtype)
        state["utility"].copy_(values.flip(0) if reverse else values)
    return engine.recycle()


def snapshot(engine):
    return copy.deepcopy((engine.model.state_dict(), engine.state_dict()))


def assert_equal(a, b):
    if isinstance(b, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in b:
            assert_equal(a[key], b[key])
    elif isinstance(b, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def test_all_variants_have_identical_warmup_updates_momentum_and_reset_rng():
    engines = [make_engine(method) for method in V3_METHODS]
    for step in range(1, 3):
        logs = [advance(engine) for engine in engines]
        for engine, (log, reset) in zip(engines, logs):
            assert log["nominal_feature_gain"] == 1
            assert log["gates"]["head"] == 1
            assert reset["recycled_by_population"] == {"hidden1": 0, "hidden2": 0}
            assert engine.steps == step
            assert_equal(engine.model.state_dict(), engines[0].model.state_dict())
            for name in engine.params:
                assert_equal(engine.state[name]["momentum"], engines[0].state[name]["momentum"])
            assert_equal(engine.generator.get_state(), engines[0].generator.get_state())
    for engine in engines:
        assert advance(engine)[1]["recycled"] == 0
        event, reset = advance(engine)
        assert event["nominal_feature_gain"] == pytest.approx(0.4)
        expected = 0 if engine.variant == "er_v3" else 1
        assert reset["recycled_by_population"] == {"hidden1": expected, "hidden2": expected}
        before = snapshot(engine)
        assert engine.recycle()["recycled"] == 0  # No duplicate reset at one update.
        assert_equal(snapshot(engine), before)


@pytest.mark.parametrize("gain", [0.05, 0.15, 0.5, 1.0])
def test_direct_gain_does_not_apply_legacy_floor_and_peak_may_exceed_one(gain):
    cfg = V3Config(feature_gain=gain, newborn_peak=4*gain, warmup_steps=0,
                   reset_interval=2, reset_count=1, max_update_norm=100)
    engine = make_engine("newborn_v3", cfg=cfg)
    log, _ = advance(engine, recycle=False)
    assert log["nominal_feature_gain"] == pytest.approx(gain)
    assert log["gates"]["head"] == 1
    assert engine.allocation_config.newborn_peak == 4*gain


def test_newborn_gains_follow_actual_reset_rows_and_columns_only():
    pc = replace(make_engine().config, momentum=0)
    engine = make_engine("newborn_v3", pc=pc)
    prime_reset(engine)
    for index in range(4):
        before = {name: p.detach().clone() for name, p in engine.params.items()}
        log = engine.step(gradients(engine, 1), {})
        gain = 0.4 + 0.6*max(0, 1-index/3)
        # Unit zero was reset in both hidden populations. Overlapping incoming
        # rows and outgoing columns combine by maximum, never by multiplication.
        expected1 = torch.full_like(engine.model.hidden1.weight, 0.4)
        expected1[0] = gain
        expected2 = torch.full_like(engine.model.hidden2.weight, 0.4)
        expected2[0, :] = gain
        expected2[:, 0] = gain
        torch.testing.assert_close(engine.model.hidden1.weight, before["hidden1.weight"]-0.1*expected1)
        torch.testing.assert_close(engine.model.hidden2.weight, before["hidden2.weight"]-0.1*expected2)
        torch.testing.assert_close(engine.model.hidden1.bias[0], before["hidden1.bias"][0]-0.1*gain)
        # Head has a common nominal multiplier of one, including reset columns.
        torch.testing.assert_close(engine.model.head.weight, before["head.weight"]-0.1)
        assert log["newborn_units"] == (2 if index < 2 else 0)
        assert log["total_local_consolidations"] == 0
        assert log["protected_units"] == 0
    assert engine.gates_for_step()["hidden1"] == pytest.approx(0.4)


def test_matched_allocation_preserves_feature_parameter_mean_causally():
    matched = make_engine("newborn_matched_v3")
    unadjusted = make_engine("newborn_v3")
    assert prime_reset(matched)["reset_indices"] == prime_reset(unadjusted)["reset_indices"]
    initial = snapshot(matched)
    preview = matched.gates_for_step()
    assert_equal(snapshot(matched), initial)
    assert preview["head"] == 1
    for index in range(4):
        exact = matched.step(gradients(matched), {})
        extra = unadjusted.step(gradients(unadjusted), {})
        assert exact["nominal_feature_gain"] == pytest.approx(0.4, abs=1e-7)
        assert exact["matched_budget_residual"] == pytest.approx(0, abs=1e-7)
        if index < 3:
            assert exact["mature_feature_gain"] < 0.4
            assert extra["nominal_feature_gain"] > 0.4
        else:
            assert exact["mature_feature_gain"] == pytest.approx(0.4)
    # A run only consumes its present clocks/utilities; external traces are
    # rejected even when the caller supplies an empty trace.
    with pytest.raises(ValueError, match="externally forced"):
        matched.recycle(forced_counts={})


def test_matched_infeasibility_is_a_hard_atomic_failure():
    cfg = V3Config(feature_gain=0.05, newborn_peak=1, warmup_steps=0,
                   reset_interval=2, reset_count=1)
    engine = make_engine("newborn_matched_v3", width=3, cfg=cfg)
    prime_reset(engine)
    before = snapshot(engine)
    with pytest.raises(ValueError, match="infeasible matched nominal"):
        engine.step(gradients(engine), {})
    assert_equal(snapshot(engine), before)


def test_nominal_matched_budget_is_separate_from_common_clipping():
    cfg = replace(make_engine().allocation_config, max_update_norm=0.01)
    engine = make_engine("newborn_matched_v3", cfg=cfg)
    prime_reset(engine)
    log = engine.step(gradients(engine, 10), {})
    assert log["nominal_feature_gain"] == pytest.approx(0.4)
    assert 0 < log["effective_feature_gain"] < log["nominal_feature_gain"]
    assert log["effective_feature_gain"] == pytest.approx(log["nominal_feature_gain"]*log["clip_scale"])
    assert log["update_norm"] == pytest.approx(0.01)


def test_exact_scheduled_counts_with_different_utility_ranked_identities():
    a, b = make_engine("recycle_v3"), make_engine("full_v3")
    first, second = prime_reset(a), prime_reset(b, reverse=True)
    assert first["recycled_by_population"] == second["recycled_by_population"] == {"hidden1": 1, "hidden2": 1}
    assert first["eligible_by_population"] == second["eligible_by_population"] == {"hidden1": 12, "hidden2": 12}
    assert first["reset_indices"] == {"hidden1": [0], "hidden2": [0]}
    assert second["reset_indices"] == {"hidden1": [11], "hidden2": [11]}
    assert first["reset_identity_sha256"] != second["reset_identity_sha256"]


@pytest.mark.parametrize("method", ["recycle_v3", "newborn_v3", "newborn_matched_v3", "consolidation_v3"])
def test_gain_window_and_consolidation_do_not_extend_reset_eligibility(method):
    cfg = replace(make_engine().allocation_config, newborn_steps=20)
    engine = make_engine(method, cfg=cfg)
    prime_reset(engine)
    for unit in engine.units:
        state = engine.unit_state[unit.name]
        state["age"].fill_(2)
        state["reset_age"][0] = 2
        state["utility"].fill_(100)
        state["utility"][0] = 0
        # Consolidation is never an implicit reset-protection intervention.
        for name in engine._unit_params[unit.name][:2]:
            engine.state[name]["c"].fill_(1)
    engine.steps += 2
    reset = engine.recycle()
    assert reset["reset_indices"] == {"hidden1": [0], "hidden2": [0]}


@pytest.mark.parametrize("method", ["protection_v3", "full_v3"])
def test_extended_protection_is_separate_and_ends_at_its_declared_age(method):
    engine = make_engine(method)
    prime_reset(engine)
    for state in engine.unit_state.values():
        state["age"].fill_(2)
        state["reset_age"][0] = 5
        state["utility"].copy_(torch.arange(12, dtype=torch.float32))
    engine.steps += 2
    reset = engine.recycle()
    assert reset["reset_indices"] == {"hidden1": [1], "hidden2": [1]}
    assert reset["eligible_by_population"] == {"hidden1": 11, "hidden2": 11}
    for state in engine.unit_state.values():
        state["age"].fill_(2)
        state["reset_age"][0] = 6
        state["utility"].fill_(100)
        state["utility"][0] = 0
    engine.steps += 2
    assert engine.recycle()["reset_indices"] == {"hidden1": [0], "hidden2": [0]}


def test_infeasible_count_checks_every_population_before_mutation_or_rng():
    engine = make_engine()
    prime_reset(engine)
    engine.steps += 2
    # hidden2 is planned first; hidden1 is infeasible. Neither may be mutated.
    engine.unit_state["hidden2"]["age"].fill_(10)
    engine.unit_state["hidden1"]["age"].zero_()
    before = snapshot(engine)
    with pytest.raises(ValueError, match="requested 1, eligible 0"):
        engine.recycle()
    assert_equal(snapshot(engine), before)


@pytest.mark.parametrize("method", ["consolidation_v3", "full_v3"])
@pytest.mark.parametrize("maturity,newborn", [(2, 3), (5, 2)])
def test_only_useful_reset_units_consolidate_after_both_local_windows(method, maturity, newborn):
    pc = replace(make_engine().config, maturity_steps=maturity, momentum=0)
    cfg = replace(make_engine().allocation_config, newborn_steps=newborn)
    engine = make_engine(method, pc=pc, cfg=cfg)
    for _ in range(8):
        advance(engine, recycle=False)
    assert engine.total_maturations == 0
    assert all(not bool(state["c"].any()) for state in engine.state.values())
    reset = prime_reset(engine)
    assert reset["recycled"] == 2
    threshold = max(maturity, newborn)
    for age in range(1, threshold+1):
        log = engine.step(gradients(engine), {})
        for name in ("hidden1.weight", "hidden1.bias", "hidden2.weight", "hidden2.bias"):
            state = engine.state[name]
            if age < threshold:
                assert state["c"][0] == 0
            else:
                assert state["c"][0] > 0
                assert state["importance"][0] > 0
                assert_equal(state["anchor"][0], engine.params[name][0])
                assert not bool(state["path"][0].any())
            assert not bool(state["c"][1:].any())
        assert not bool(engine.state["head.weight"]["c"].any())
        if method == "consolidation_v3":
            assert log["nominal_feature_gain"] == pytest.approx(0.4)
            assert log["newborn_units"] == 0
    assert log["matured_units"] == log["locally_consolidated_units"] == 2
    after = engine.step(gradients(engine), {})
    assert after["consolidation_gain_reduction"] > 0
    assert after["nominal_feature_gain"] < after["pre_consolidation_feature_gain"]
    assert after["total_local_consolidations"] == 2
    assert after["pending_maturation_units"] == 0
    with pytest.raises(ValueError, match="local to reset-unit"):
        engine.consolidate(["hidden1"])


def test_zero_si_usefulness_matures_without_anchoring_or_consolidation():
    engine = make_engine("consolidation_v3")
    prime_reset(engine)
    anchors = {name: state["anchor"].clone() for name, state in engine.state.items()}
    for _ in range(3):
        log = engine.step(gradients(engine, 0), {})
    assert log["matured_units"] == 2
    assert log["locally_consolidated_units"] == 0
    for name, state in engine.state.items():
        assert not bool(state["c"].any())
        assert_equal(state["anchor"], anchors[name])


def test_clip_bounds_complete_update_including_head_decay_and_anchor():
    pc = replace(make_engine().config, weight_decay=0.3, anchor_strength=2)
    cfg = replace(make_engine().allocation_config, warmup_steps=0, max_update_norm=0.025)
    engine = make_engine("full_v3", pc=pc, cfg=cfg)
    for name, state in engine.state.items():
        state["momentum"].fill_(2)
        if engine.block_for[name] != "head":
            state["c"].fill_(0.25)
            state["anchor"].add_(1)
    grad = gradients(engine, 3)
    before = {name: p.detach().clone() for name, p in engine.params.items()}
    proposed, data = {}, {}
    for name, p in engine.params.items():
        gain = 1 if engine.block_for[name] == "head" else 0.4*0.75**pc.gamma
        data[name] = -pc.lr*gain*(pc.momentum*2+3)*torch.ones_like(p)
        proposed[name] = data[name]-pc.lr*gain*pc.weight_decay*p.detach()
        proposed[name] -= pc.lr*pc.anchor_strength*row_view(engine.state[name]["c"], p)*(p.detach()-engine.state[name]["anchor"])
    norm = math.sqrt(sum(float(delta.double().square().sum()) for delta in proposed.values()))
    log = engine.step(grad, {})
    scale = cfg.max_update_norm/norm
    assert log["proposed_update_norm"] == pytest.approx(norm)
    assert log["clip_scale"] == pytest.approx(scale)
    assert log["update_norm"] == pytest.approx(cfg.max_update_norm)
    assert log["head_displacement"] > 0
    assert log["head_data_displacement"] > 0
    for name, p in engine.params.items():
        torch.testing.assert_close(p, before[name]+scale*proposed[name])
        torch.testing.assert_close(engine.state[name]["momentum"], torch.full_like(p, pc.momentum*2+3))
        if engine.block_for[name] != "head":
            torch.testing.assert_close(engine.state[name]["path"], -grad[name]*data[name]*scale)
    feature_data = math.sqrt(sum(float((value*scale).double().square().sum()) for name, value in data.items() if not name.startswith("head.")))
    assert log["feature_data_displacement"] == pytest.approx(feature_data)


def test_amplified_newborn_and_head_share_one_bound_but_resets_are_outside_it():
    cfg = replace(make_engine().allocation_config, feature_gain=1, newborn_peak=4, max_update_norm=0.01)
    engine = make_engine("newborn_v3", cfg=cfg)
    reset = prime_reset(engine)
    assert reset["recycled"] == 2
    log = engine.step(gradients(engine, 100), {})
    assert log["nominal_feature_gain"] > 1
    assert log["gates"]["head"] == 1
    assert log["update_norm"] == pytest.approx(0.01)
    assert log["head_data_displacement"] < 0.01
    assert log["feature_data_displacement"] < 0.01
    assert log["clipped"] and log["clip_events"] == 1


@pytest.mark.parametrize("failure", ["nan_gradient", "momentum_overflow", "proposal_overflow", "si_overflow"])
def test_nonfinite_updates_fail_atomically_including_momentum_and_clocks(failure):
    engine = make_engine("full_v3")
    grad = gradients(engine)
    if failure == "nan_gradient":
        grad["head.bias"][0] = float("nan")
    elif failure == "momentum_overflow":
        engine.state["head.bias"]["momentum"].fill_(torch.finfo(torch.float32).max)
        grad["head.bias"].fill_(torch.finfo(torch.float32).max)
    elif failure == "proposal_overflow":
        engine.config.lr = 1e30
        grad["head.bias"].fill_(1e20)
    else:
        # Parameters and complete optimizer displacement remain finite; the SI
        # path proposal alone overflows, and must still prevent every mutation.
        engine.allocation_config = replace(engine.allocation_config, max_update_norm=1e30)
        grad["hidden1.weight"].fill_(1e30)
    before = snapshot(engine)
    with pytest.raises(FloatingPointError, match="non-finite"):
        engine.step(grad, {})
    assert_equal(snapshot(engine), before)


def test_reset_clears_overlapping_state_without_resurrecting_outgoing_columns():
    pc = replace(make_engine().config, anchor_strength=1)
    engine = make_engine("full_v3", pc=pc)
    for state in engine.state.values():
        state["momentum"].fill_(3)
        state["path"].fill_(4)
        state["c"].fill_(0.5)
        state["importance"].fill_(2)
        state["anchor"].add_(1)
    prime_reset(engine)
    for name in ("hidden1.weight", "hidden1.bias", "hidden2.weight", "hidden2.bias"):
        state = engine.state[name]
        for key in ("momentum", "path", "c", "importance", "age"):
            assert not bool(state[key][0].any())
        assert_equal(state["anchor"][0], engine.params[name][0])
        assert_equal(state["start"][0], engine.params[name][0])
    for name in ("hidden2.weight", "head.weight"):
        assert not bool(engine.params[name][:, 0].any())
        for key in ("momentum", "path", "anchor", "start"):
            assert not bool(engine.state[name][key][:, 0].any())
    engine.step(gradients(engine, 0), {})
    assert not bool(engine.model.hidden2.weight[:, 0].any())
    assert not bool(engine.model.head.weight[:, 0].any())


@pytest.mark.parametrize("method", V3_METHODS)
def test_checkpoint_continuation_preserves_all_local_clocks_and_reset_choices(method):
    first = make_engine(method)
    for _ in range(5):
        advance(first)
    saved = snapshot(first)
    restored = make_engine(method)
    restored.model.load_state_dict(saved[0])
    restored.load_state_dict(saved[1])
    assert_equal(snapshot(restored), snapshot(first))
    for index in range(10):
        assert_equal(advance(first, value=0.1+index/30), advance(restored, value=0.1+index/30))
    assert_equal(snapshot(restored), snapshot(first))


@pytest.mark.parametrize("corrupt", ["config", "variant", "tensor", "last_update"])
def test_incompatible_or_invalid_checkpoint_is_rejected_before_state_mutation(corrupt):
    engine = make_engine("full_v3")
    advance(engine)
    before = snapshot(engine)
    saved = copy.deepcopy(before[1])
    if corrupt == "config":
        saved["v3"]["allocation_config"]["newborn_steps"] += 1
    elif corrupt == "variant":
        saved["v3"]["variant"] = "recycle_v3"
    elif corrupt == "tensor":
        saved["state"]["head.bias"]["momentum"][0] = float("nan")
    else:
        del saved["v3"]["last_update"]
    with pytest.raises(ValueError, match="checkpoint"):
        engine.load_state_dict(saved)
    assert_equal(snapshot(engine), before)


@pytest.mark.parametrize("field,value", [
    ("feature_gain", 0), ("feature_gain", 1.1), ("feature_gain", True),
    ("newborn_peak", 0.01), ("newborn_peak", float("inf")),
    ("max_update_norm", 0), ("max_update_norm", float("nan")),
    ("warmup_steps", -1), ("warmup_steps", True),
    ("newborn_steps", 0), ("newborn_steps", 1.5),
    ("protection_steps", -1), ("reset_interval", 0), ("reset_count", 0),
])
def test_invalid_allocation_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        V3Config(**{field: value})


def test_legacy_newborn_clock_is_disabled_without_mutating_callers_config():
    pc = replace(make_engine().config, newborn_steps=99)
    engine = make_engine("full_v3", pc=pc)
    assert pc.newborn_steps == 99
    assert engine.config.newborn_steps == 0
    assert all(bool((state["newborn_age"] == -1).all()) for state in engine.unit_state.values())
    with pytest.raises(ValueError, match="external gates"):
        engine.step(gradients(engine), {"hidden1": 0.1})
    with pytest.raises(ValueError, match="external gates"):
        engine.step(gradients(engine), {}, ("hidden1",))
