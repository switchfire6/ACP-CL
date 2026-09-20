import copy
from dataclasses import replace
import json

import pytest

from acp_cl.controller import ControllerConfig, MonitorSignals, Phase
from acp_cl.controller_v2 import ReopeningController


BLOCKS = ["first", "second", "third"]
SCORES = {"first": 0.1, "second": 0.9, "third": 0.5}
STABLE = MonitorSignals(loss_change=0.0, has_replay=True)
NOVEL = MonitorSignals(surprise=6.0, drift=0.5, loss_change=0.4, has_replay=True)


@pytest.fixture
def config():
    return ControllerConfig(monitor_interval=10, scaffold_steps=20, min_open_steps=20,
                            stable_windows=2, closing_windows=2, novel_windows=2,
                            reopen_max_windows=3, cooldown_windows=2, top_k=1)


def observe(controller, step, signals=STABLE):
    return controller.observe(step, signals, SCORES)


def mature(config, oracle=False):
    controller = ReopeningController(BLOCKS, config, oracle=oracle)
    for step in range(10, 61, 10):
        observe(controller, step, NOVEL)
    assert controller.phase == Phase.ADULT
    return controller


def test_initial_schedule_is_common_and_does_not_need_stability(config):
    controllers = [ReopeningController(BLOCKS, config),
                   ReopeningController(BLOCKS, config, oracle=True),
                   ReopeningController(BLOCKS, replace(config, adaptive=False))]
    for step, phase in ((10, Phase.SCAFFOLD), (20, Phase.OPEN), (30, Phase.OPEN),
                         (40, Phase.CLOSING), (50, Phase.CLOSING), (60, Phase.ADULT)):
        for controller in controllers:
            event = observe(controller, step, NOVEL)
            assert controller.phase == phase
            assert event["initial_schedule"] == "fixed"
        assert len({tuple(c.gates().values()) for c in controllers}) == 1
    assert all(c.should_consolidate for c in controllers)
    assert all(c.consolidation_blocks == tuple(BLOCKS) for c in controllers)


def test_oracle_changes_trigger_information_not_amplitude_or_local_selection(config):
    autonomous = mature(config)
    oracle = mature(config, oracle=True)
    observe(autonomous, 70, NOVEL)
    observe(oracle, 70, STABLE)
    assert oracle.notify_boundary()
    detected = observe(autonomous, 80, NOVEL)
    privileged = observe(oracle, 80, STABLE)
    assert detected["reason"] == "sustained_novelty"
    assert privileged["reason"] == "oracle_boundary"
    assert detected["novelty"] > privileged["novelty"]
    assert autonomous.gates() == oracle.gates() == {
        "first": config.g_min, "second": 1.0, "third": config.g_min,
    }
    assert detected["selected_blocks"] == privileged["selected_blocks"] == ["second"]


def test_oracle_never_opens_autonomously_and_ignores_initial_boundary(config):
    controller = ReopeningController(BLOCKS, config, oracle=True)
    assert not controller.notify_boundary()
    assert observe(controller, 10)["oracle_event"] == "ignored_initial"
    for step in range(20, 201, 10):
        observe(controller, step, NOVEL)
    assert controller.phase == Phase.ADULT
    for step in range(210, 301, 10):
        observe(controller, step, STABLE)
    assert controller.phase == Phase.ADULT
    with pytest.raises(RuntimeError, match="autonomous"):
        ReopeningController(BLOCKS, config).notify_boundary()


def test_oracle_risk_veto_defers_event_and_reopened_guard_contracts_gates(config):
    controller = mature(config, oracle=True)
    controller.notify_boundary()
    dangerous = replace(STABLE, replay_damage=0.2)
    blocked = observe(controller, 70, dangerous)
    assert blocked["oracle_event"] == "waiting_replay_risk"
    assert controller.phase == Phase.ADULT
    assert blocked["oracle_pending_windows"] == 2
    assert observe(controller, 80)["reason"] == "oracle_boundary"
    assert controller.gates()["second"] == 1.0
    event = observe(controller, 90, dangerous)
    assert controller.phase == Phase.REOPENED
    assert event["safety_contracted"]
    assert set(controller.gates().values()) == {config.g_min}


def test_oracle_cannot_retain_stale_event_through_risk_or_skipped_windows(config):
    controller = mature(config, oracle=True)
    controller.notify_boundary()
    dangerous = replace(STABLE, replay_damage=0.2)
    observe(controller, 70, dangerous)
    observe(controller, 80, dangerous)
    assert observe(controller, 90, dangerous)["oracle_event"] == "expired"
    assert observe(controller, 100)["oracle_pending_windows"] == 0
    assert controller.phase == Phase.ADULT
    controller.notify_boundary()
    assert observe(controller, 150)["oracle_event"] == "expired"
    assert controller.phase == Phase.ADULT


def test_oracle_notifications_coalesce_and_do_not_extend_existing_pulse(config):
    controller = mature(config, oracle=True)
    assert controller.notify_boundary()
    assert controller.notify_boundary()
    assert controller.state_dict()["boundary_notice"] == "coalesced"
    observe(controller, 70, NOVEL)
    assert not controller.notify_boundary()
    assert observe(controller, 80, NOVEL)["oracle_event"] == "ignored_reopened"
    observe(controller, 90, NOVEL)
    event = observe(controller, 100, NOVEL)
    assert controller.phase == Phase.ADULT
    assert event["reason"] == "reopening_timeout"
    assert controller.consolidation_blocks == ("second",)


def test_oracle_respects_full_cooldown_windows_and_then_uses_queued_event(config):
    controller = mature(config, oracle=True)
    controller.notify_boundary()
    observe(controller, 70)
    observe(controller, 80)
    assert observe(controller, 90)["reason"] == "adaptation_stable"
    controller.notify_boundary()
    assert observe(controller, 100)["oracle_event"] == "waiting_cooldown"
    assert observe(controller, 110)["oracle_event"] == "waiting_cooldown"
    assert observe(controller, 120)["reason"] == "oracle_boundary"


@pytest.mark.parametrize("oracle", [False, True])
@pytest.mark.parametrize("change", [{"adaptive": False}, {"allow_reopening": False}])
def test_disabled_reopening_applies_to_both_modes(config, oracle, change):
    controller = mature(replace(config, **change), oracle=oracle)
    if oracle:
        assert not controller.notify_boundary()
    for step in range(70, 201, 10):
        observe(controller, step, NOVEL)
    assert controller.phase == Phase.ADULT


@pytest.mark.parametrize("oracle", [False, True])
def test_checkpoint_roundtrip_in_every_phase_and_with_pending_event(config, oracle):
    controller = ReopeningController(BLOCKS, config, oracle=oracle)
    seen = set()
    for step in range(10, 181, 10):
        if oracle and step == 80:
            controller.notify_boundary()
        saved = json.loads(json.dumps(controller.state_dict(), allow_nan=False))
        restored = ReopeningController(BLOCKS, config, oracle=oracle)
        restored.load_state_dict(saved)
        assert restored.state_dict() == controller.state_dict()
        assert restored.gates() == controller.gates()
        assert observe(restored, step, NOVEL) == observe(controller, step, NOVEL)
        seen.add(controller.phase)
    assert seen == set(Phase)


def test_checkpoint_restore_is_strict_and_atomic(config):
    controller = mature(config, oracle=True)
    controller.notify_boundary()
    before = controller.state_dict()
    for change in ({"oracle": False}, {"version": True}, {"pending_boundary_windows": 4},
                   {"pending_boundary_deadline_step": 9999}, {"initial_complete": False},
                   {"boundary_notice": "unknown"}, {"controller_type": "CriticalPeriodController"}):
        with pytest.raises(ValueError):
            controller.load_state_dict({**before, **change})
        assert controller.state_dict() == before
    malformed = copy.deepcopy(before)
    malformed["base"]["config"]["adaptive"] = 1
    with pytest.raises(ValueError, match="underlying"):
        controller.load_state_dict(malformed)
    assert controller.state_dict() == before


def test_failed_observation_preserves_pending_event_and_configuration(config):
    controller = mature(config, oracle=True)
    controller.notify_boundary()
    before = controller.state_dict()
    with pytest.raises(ValueError):
        observe(controller, 61)
    assert controller.state_dict() == before
    assert controller.config is config


def test_oracle_invalid_signal_does_not_bypass_autonomous_validation(config):
    controller = mature(config, oracle=True)
    controller.notify_boundary()
    invalid = replace(STABLE, surprise=float("nan"))
    event = observe(controller, 70, invalid)
    assert event["oracle_event"] == "waiting_invalid_signals"
    assert controller.phase == Phase.ADULT
    assert observe(controller, 80)["reason"] == "oracle_boundary"
