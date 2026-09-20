"""State-machine checks use synthetic observations, not dataset task boundaries."""

from dataclasses import replace
import json
import math

import pytest

from acp_cl.controller import ControllerConfig, CriticalPeriodController, MonitorSignals, Phase


BLOCKS = ["stage1", "stage2", "stage3"]
SCORES = {"stage1": 0.2, "stage2": 0.9, "stage3": 0.4}
STABLE = MonitorSignals(loss_change=0.0, drift=0.0, has_replay=True)
NOVEL = MonitorSignals(surprise=6.0, drift=0.5, loss_change=0.4, has_replay=True)
UNSTABLE = MonitorSignals(loss_change=0.3, drift=0.2, has_replay=True)


@pytest.fixture
def config():
    return ControllerConfig(
        monitor_interval=10,
        scaffold_steps=20,
        min_open_steps=20,
        stable_windows=2,
        closing_windows=2,
        novel_windows=2,
        reopen_max_windows=3,
        cooldown_windows=2,
        top_k=1,
    )


def observe(controller, step, signals=STABLE):
    return controller.observe(step, signals, SCORES)


def adult_controller(config):
    controller = CriticalPeriodController(BLOCKS, config)
    for step in (10, 20, 30, 40, 50, 60, 70):
        observe(controller, step)
    assert controller.phase == Phase.ADULT
    return controller


def reopened_controller(config):
    controller = adult_controller(config)
    for step in (80, 90, 100, 110):
        observe(controller, step, NOVEL)
    assert controller.phase == Phase.REOPENED
    return controller


def test_initial_phases_maturity_ramp_and_single_consolidation(config):
    controller = CriticalPeriodController(BLOCKS, config)
    assert controller.phase == Phase.SCAFFOLD
    assert set(controller.gates().values()) == {1.0}
    observe(controller, 10)
    assert controller.phase == Phase.SCAFFOLD
    observe(controller, 20)
    assert controller.phase == Phase.OPEN
    assert observe(controller, 30)["stable_count"] == 0
    assert observe(controller, 40)["stable_count"] == 1
    assert observe(controller, 50)["reason"] == "sustained_stability"
    assert controller.phase == Phase.CLOSING
    assert set(controller.gates().values()) == {1.0}
    observe(controller, 60)
    assert all(gate == pytest.approx(0.525) for gate in controller.gates().values())
    event = observe(controller, 70)
    assert controller.phase == Phase.ADULT
    assert event["should_consolidate"]
    assert controller.consolidation_blocks == tuple(BLOCKS)
    assert set(controller.gates().values()) == {config.g_min}
    observe(controller, 80)
    assert not controller.should_consolidate
    assert controller.consolidation_blocks == ()


@pytest.mark.parametrize(
    "unstable",
    [
        replace(STABLE, loss_change=0.03),
        replace(STABLE, loss_change=-0.03),
        replace(STABLE, drift=0.05),
        replace(STABLE, replay_damage=0.15),
    ],
)
def test_stability_needs_every_signal_and_consecutive_windows(config, unstable):
    controller = CriticalPeriodController(BLOCKS, config)
    observe(controller, 20)
    observe(controller, 40)
    assert observe(controller, 50, unstable)["stable_count"] == 0
    observe(controller, 60)
    assert controller.phase == Phase.OPEN
    observe(controller, 70)
    assert controller.phase == Phase.CLOSING


def test_adaptive_open_never_closes_from_elapsed_time(config):
    controller = CriticalPeriodController(BLOCKS, config)
    observe(controller, 20, UNSTABLE)
    log = observe(controller, 100_000, UNSTABLE)
    assert controller.phase == Phase.OPEN
    assert log["reason"] == "awaiting_stability"
    assert not controller.should_consolidate


def test_sustained_novelty_resets_and_cooldown_counts_full_windows(config):
    controller = adult_controller(config)
    for step in (80, 90):
        log = observe(controller, step, NOVEL)
        assert log["reason"] == "cooldown"
        assert log["novel_count"] == 0
    assert observe(controller, 100, NOVEL)["novel_count"] == 1
    assert observe(controller, 110, STABLE)["novel_count"] == 0
    observe(controller, 120, NOVEL)
    assert controller.phase == Phase.ADULT
    observe(controller, 130, NOVEL)
    assert controller.phase == Phase.REOPENED


def test_reopening_is_local_and_bounded_even_with_persistent_novelty(config):
    controller = reopened_controller(config)
    assert controller.selected_blocks == ("stage2",)
    assert controller.gates()["stage2"] > config.g_min
    assert controller.gates()["stage1"] == config.g_min
    assert controller.gates()["stage3"] == config.g_min
    observe(controller, 120, NOVEL)
    observe(controller, 130, NOVEL)
    assert controller.phase == Phase.REOPENED
    log = observe(controller, 140, NOVEL)
    assert controller.phase == Phase.ADULT
    assert log["reason"] == "reopening_timeout"
    assert controller.consolidation_blocks == ("stage2",)
    assert controller.selected_blocks == ()
    assert observe(controller, 150, NOVEL)["reason"] == "cooldown"


def test_reopening_consolidates_early_after_stability(config):
    controller = reopened_controller(config)
    observe(controller, 120)
    assert controller.phase == Phase.REOPENED
    log = observe(controller, 130)
    assert log["reason"] == "adaptation_stable"
    assert controller.phase == Phase.ADULT
    assert controller.consolidation_blocks == ("stage2",)


def test_missed_monitoring_call_does_not_extend_reopening_deadline(config):
    controller = reopened_controller(config)
    log = observe(controller, 1000, NOVEL)
    assert controller.phase == Phase.ADULT
    assert log["reason"] == "reopening_timeout"


def test_replay_risk_suppresses_reopening_despite_high_novelty(config):
    controller = adult_controller(config)
    dangerous = replace(NOVEL, surprise=40.0, replay_damage=0.2)
    for step in (80, 90, 100, 110, 120):
        log = observe(controller, step, dangerous)
    assert controller.novelty > config.open_threshold
    assert controller.phase == Phase.ADULT
    assert log["replay_risk"]
    assert log["safety_contracted"]
    assert log["novel_count"] == 0


def test_replay_risk_contracts_open_and_reopened_gates(config):
    dangerous = replace(NOVEL, replay_damage=0.2)
    controller = CriticalPeriodController(BLOCKS, config)
    observe(controller, 20, dangerous)
    assert set(controller.gates().values()) == {config.g_min}
    observe(controller, 30, UNSTABLE)
    assert set(controller.gates().values()) == {1.0}
    controller = reopened_controller(config)
    observe(controller, 120, dangerous)
    assert controller.phase == Phase.REOPENED
    assert set(controller.gates().values()) == {config.g_min}


def test_replay_ablation_disables_novelty_penalty_and_safety(config):
    controller = adult_controller(replace(config, use_replay_damage=False))
    dangerous = replace(NOVEL, replay_damage=40.0)
    for step in (80, 90, 100, 110):
        log = observe(controller, step, dangerous)
    assert controller.phase == Phase.REOPENED
    assert not log["replay_observed"]
    assert not log["safety_contracted"]
    assert controller.gates()["stage2"] > config.g_min


def test_unobserved_replay_is_not_used_as_damage(config):
    controller = CriticalPeriodController(BLOCKS, config)
    log = observe(controller, 10, replace(STABLE, replay_damage=100.0, has_replay=False))
    assert not log["replay_observed"]
    assert not log["replay_risk"]
    assert log["stable"]
    assert set(controller.gates().values()) == {1.0}


def test_plasticity_ablation_removes_deficit_from_novelty(config):
    enabled = CriticalPeriodController(BLOCKS, config)
    disabled = CriticalPeriodController(BLOCKS, replace(config, use_plasticity_signal=False))
    signals = replace(STABLE, plasticity_deficit=10.0)
    observe(enabled, 10, signals)
    observe(disabled, 10, signals)
    assert enabled.novelty > config.open_threshold
    assert disabled.novelty == pytest.approx(1 / (1 + math.exp(config.novelty_bias)))


def test_global_gating_ablation_and_disabled_reopening(config):
    global_controller = reopened_controller(replace(config, local_gating=False))
    assert global_controller.selected_blocks == tuple(BLOCKS)
    assert all(gate > config.g_min for gate in global_controller.gates().values())
    disabled = adult_controller(replace(config, allow_reopening=False))
    for step in range(80, 301, 10):
        observe(disabled, step, NOVEL)
    assert disabled.phase == Phase.ADULT


def test_fixed_schedule_is_signal_independent_and_never_reopens(config):
    fixed = replace(config, adaptive=False)
    calm = CriticalPeriodController(BLOCKS, fixed)
    noisy = CriticalPeriodController(BLOCKS, fixed)
    for step, phase in ((10, Phase.SCAFFOLD), (20, Phase.OPEN), (40, Phase.CLOSING), (50, Phase.CLOSING), (60, Phase.ADULT)):
        observe(calm, step, STABLE)
        observe(noisy, step, replace(NOVEL, replay_damage=0.8))
        assert calm.phase == noisy.phase == phase
        assert calm.gates() == noisy.gates()
    assert calm.should_consolidate
    for step in range(70, 301, 10):
        observe(noisy, step, NOVEL)
    assert noisy.phase == Phase.ADULT


def test_deterministic_ties_top_k_clipped_to_number_of_blocks(config):
    controller = adult_controller(replace(config, top_k=100))
    for step in (80, 90, 100, 110):
        controller.observe(step, NOVEL, {})
    assert controller.selected_blocks == tuple(BLOCKS)


def test_checkpoint_roundtrip_continues_identically_in_every_phase(config):
    original = CriticalPeriodController(BLOCKS, config)
    seen = set()
    for step in range(10, 211, 10):
        signals = STABLE if step <= 70 else NOVEL
        saved = json.loads(json.dumps(original.state_dict(), allow_nan=False))
        restored = CriticalPeriodController(BLOCKS, config)
        restored.load_state_dict(saved)
        assert restored.state_dict() == original.state_dict()
        assert restored.gates() == original.gates()
        expected = observe(original, step, signals)
        assert observe(restored, step, signals) == expected
        seen.add(original.phase)
    assert seen == set(Phase)


def test_checkpoint_validation_is_atomic(config):
    controller = reopened_controller(config)
    before = controller.state_dict()
    for changes in (
        {"version": 999}, {"block_names": ["different"]}, {"config": {}},
        {"novelty": float("nan")}, {"cooldown_remaining": -1},
        {"selected_blocks": []}, {"should_consolidate": True},
    ):
        with pytest.raises(ValueError):
            controller.load_state_dict({**before, **changes})
        assert controller.state_dict() == before


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_signals_are_logged_finitely_and_cannot_trigger_transitions(config, bad_value):
    controller = adult_controller(config)
    signals = replace(NOVEL, surprise=bad_value)
    for step in (80, 90, 100, 110):
        log = observe(controller, step, signals)
        json.dumps(log, allow_nan=False)
    assert controller.phase == Phase.ADULT
    assert log["invalid_signals"] == ["surprise"]
    assert not log["high_novelty"]


def test_nan_replay_guard_is_conservative_but_ablation_ignores_it(config):
    signals = replace(STABLE, replay_damage=float("nan"))
    enabled = CriticalPeriodController(BLOCKS, config)
    disabled = CriticalPeriodController(BLOCKS, replace(config, use_replay_damage=False))
    assert observe(enabled, 10, signals)["safety_contracted"]
    assert not observe(enabled, 20, signals)["stable"]
    assert observe(disabled, 10, signals)["stable"]


def test_monitor_interval_enforced_without_mutating_state(config):
    controller = CriticalPeriodController(BLOCKS, config)
    observe(controller, 10)
    before = controller.state_dict()
    for bad_step in (10, 11, 9, -1, 20.0, True):
        with pytest.raises(ValueError):
            observe(controller, bad_step)
        assert controller.state_dict() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"monitor_interval": 0}, {"scaffold_steps": -1}, {"min_open_steps": -1},
        {"stable_windows": 0}, {"closing_windows": 0}, {"cooldown_windows": -1},
        {"novel_windows": 1.5}, {"top_k": True}, {"reopen_max_windows": 0},
        {"g_min": -0.01}, {"g_min": 1.0}, {"open_threshold": 0.0},
        {"open_threshold": 1.0}, {"stability_loss_tol": 0.0},
        {"stability_drift_tol": float("nan")}, {"novelty_bias": float("inf")},
        {"max_replay_damage": 0.0}, {"adaptive": 1},
    ],
)
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        ControllerConfig(**changes)


@pytest.mark.parametrize("blocks", [[], ["a", "a"], [""], [1], "stage1"])
def test_invalid_blocks_rejected(config, blocks):
    with pytest.raises(ValueError):
        CriticalPeriodController(blocks, config)
