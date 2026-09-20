"""A controlled test of *post-development* reopening.

V2 deliberately gives all variants the same fixed initial scaffold/open/closing
schedule. This revises V1's hypothesis of adaptive initial closure, allowing a
cleaner comparison of reopening after a common burn-in. Autonomous V2 receives
no boundary signal. Oracle V2 receives only a boundary notification, never a
class/domain identity, and changes opening timing rather than pulse amplitude.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from .controller import ControllerConfig, CriticalPeriodController, MonitorSignals, Phase


class ReopeningController(CriticalPeriodController):
    """Fixed initial schedule, then bounded full-amplitude local reopening.

    ``adaptive=False`` or ``allow_reopening=False`` disables adult reopening.
    The oracle disables autonomous novelty triggers even when no event arrives.
    It keeps at most one pending notification for three monitoring opportunities
    and at most ``3 * monitor_interval`` steps since the last observed step.
    Repeated notifications coalesce to the most recent event. Notifications
    during initial development or an existing reopening are dropped; they do
    not extend an active pulse. Cooldown/risk-blocked events expire normally.

    This is a mutable, single-observer controller, like the V1 controller.
    """

    _V2_VERSION = 1
    _ORACLE_TTL_WINDOWS = 3
    _NOTICE_VALUES = {
        "none", "queued", "coalesced", "ignored_initial", "ignored_reopened",
        "ignored_disabled", "triggered", "expired", "waiting_cooldown",
        "waiting_replay_risk", "waiting_invalid_signals",
    }

    def __init__(self, block_names: list[str], config: ControllerConfig, *, oracle: bool = False):
        if type(oracle) is not bool:
            raise ValueError("oracle must be a boolean")
        super().__init__(block_names, config)
        self.oracle = oracle
        self._initial_complete = False
        self._pending_boundary_windows = 0
        self._pending_boundary_deadline_step: int | None = None
        self._boundary_notice = "none"

    def notify_boundary(self) -> bool:
        """Queue timing information for oracle mode; return whether it was kept.

        Callers must omit the initial experience's boundary. No identity or label
        is accepted. The autonomous controller rejects this privileged channel.
        """
        if not self.oracle:
            raise RuntimeError("autonomous controller cannot receive boundary notifications")
        if not self._initial_complete:
            self._boundary_notice = "ignored_initial"
            return False
        if self.phase == Phase.REOPENED:
            self._boundary_notice = "ignored_reopened"
            return False
        if not self.config.adaptive or not self.config.allow_reopening:
            self._boundary_notice = "ignored_disabled"
            return False
        self._boundary_notice = "coalesced" if self._pending_boundary_windows else "queued"
        self._pending_boundary_windows = self._ORACLE_TTL_WINDOWS
        self._pending_boundary_deadline_step = (
            self._last_step + self._ORACLE_TTL_WINDOWS * self.config.monitor_interval
        )
        return True

    def _clear_pending(self) -> None:
        self._pending_boundary_windows = 0
        self._pending_boundary_deadline_step = None

    def _oracle_observe(self, step: int, signals: MonitorSignals,
                        block_scores: dict[str, float]) -> dict[str, object]:
        original_config = self.config
        previous_phase = self.phase
        previous_cooldown = self._cooldown_remaining
        # Suppress only autonomous opening, retaining base stability/timeout,
        # signal validation, risk guard, and cooldown behavior.
        try:
            self.config = replace(original_config, allow_reopening=False)
            event = super().observe(step, signals, block_scores)
        finally:
            self.config = original_config

        notice = self._boundary_notice
        if self._pending_boundary_windows:
            if step > self._pending_boundary_deadline_step:
                notice = "expired"
                self._clear_pending()
            else:
                invalid = set(event["invalid_signals"])
                if not event["replay_observed"]:
                    invalid.discard("replay_damage")
                if not original_config.use_plasticity_signal:
                    invalid.discard("plasticity_deficit")
                if not original_config.adaptive or not original_config.allow_reopening:
                    notice = "ignored_disabled"
                    self._clear_pending()
                elif previous_phase != Phase.ADULT:
                    notice = "ignored_reopened"
                    self._clear_pending()
                elif previous_cooldown:
                    notice = "waiting_cooldown"
                elif event["replay_risk"]:
                    notice = "waiting_replay_risk"
                elif invalid:
                    notice = "waiting_invalid_signals"
                else:
                    self.phase = Phase.REOPENED
                    self._phase_start_step = step
                    self.selected_blocks = self._select(block_scores)
                    self._reopen_count = self._stable_count = self._novel_count = 0
                    notice = "triggered"
                    self._clear_pending()
                if self._pending_boundary_windows:
                    self._pending_boundary_windows -= 1
                    if self._pending_boundary_windows == 0:
                        notice = "expired"
                        self._clear_pending()

        if notice == "triggered":
            event["reason"] = "oracle_boundary"
        elif notice in {"waiting_cooldown", "waiting_replay_risk",
                        "waiting_invalid_signals", "expired"}:
            event["reason"] = "oracle_" + notice
        elif previous_phase == Phase.ADULT and event["reason"] == "reopening_disabled":
            event["reason"] = "oracle_waiting_boundary"
        event["oracle_event"] = notice
        self._boundary_notice = "none"
        return event

    def observe(self, step: int, signals: MonitorSignals,
                block_scores: dict[str, float]) -> dict[str, object]:
        previous_phase = self.phase
        if not self._initial_complete:
            original_config = self.config
            try:
                self.config = replace(original_config, adaptive=False)
                event = super().observe(step, signals, block_scores)
            finally:
                self.config = original_config
            # The fixed phase schedule does not turn off the adaptive variants'
            # existing replay-risk guard. It can contract gains, not delay age.
            self._risk_active = bool(original_config.adaptive and event["replay_risk"])
            self._initial_complete = self.phase == Phase.ADULT
            event["reason"] = "initial_fixed_schedule"
            if self._initial_complete:
                event["reason"] = "initial_fixed_consolidation_complete"
            event["oracle_event"] = self._boundary_notice
            self._boundary_notice = "none"
        elif self.oracle:
            event = self._oracle_observe(step, signals, block_scores)
        else:
            event = super().observe(step, signals, block_scores)
            event["oracle_event"] = "none"

        # Refresh state-dependent fields after an oracle transition or the
        # initial schedule's shared safety adjustment.
        event.update({
            "controller_version": "v2",
            "initial_schedule": "fixed",
            "initial_complete": self._initial_complete,
            "oracle": self.oracle,
            "oracle_pending_windows": self._pending_boundary_windows,
            "oracle_deadline_step": self._pending_boundary_deadline_step,
            "phase": self.phase.value,
            "previous_phase": previous_phase.value,
            "transition": previous_phase != self.phase,
            "selected_blocks": list(self.selected_blocks),
            "gates": self.gates(),
            "should_consolidate": self.should_consolidate,
            "consolidation_blocks": list(self.consolidation_blocks),
            "stable_count": self._stable_count,
            "novel_count": self._novel_count,
            "reopen_windows": self._reopen_count,
            "cooldown_remaining": self._cooldown_remaining,
            "phase_age_steps": step - self._phase_start_step,
            "safety_contracted": self._risk_active,
            "reopening_amplitude": 1.0,
        })
        return event

    def gates(self) -> dict[str, float]:
        if self.phase == Phase.REOPENED and not self._risk_active:
            return {name: 1.0 if name in self.selected_blocks else self.config.g_min
                    for name in self.block_names}
        return super().gates()

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_type": "ReopeningController",
            "version": self._V2_VERSION,
            "oracle": self.oracle,
            "initial_complete": self._initial_complete,
            "oracle_ttl_windows": self._ORACLE_TTL_WINDOWS,
            "pending_boundary_windows": self._pending_boundary_windows,
            "pending_boundary_deadline_step": self._pending_boundary_deadline_step,
            "boundary_notice": self._boundary_notice,
            "base": super().state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore V2 and base state atomically, rejecting mode/config drift."""
        expected = {"controller_type", "version", "oracle", "initial_complete",
                    "oracle_ttl_windows", "pending_boundary_windows",
                    "pending_boundary_deadline_step", "boundary_notice", "base"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("malformed v2 controller checkpoint")
        if state["controller_type"] != "ReopeningController":
            raise ValueError("checkpoint controller type mismatch")
        if type(state["version"]) is not int or state["version"] != self._V2_VERSION:
            raise ValueError("unsupported v2 controller checkpoint version")
        if type(state["oracle"]) is not bool or state["oracle"] != self.oracle:
            raise ValueError("checkpoint oracle mode mismatch")
        if type(state["initial_complete"]) is not bool:
            raise ValueError("invalid initial_complete flag")
        if (type(state["oracle_ttl_windows"]) is not int
                or state["oracle_ttl_windows"] != self._ORACLE_TTL_WINDOWS):
            raise ValueError("checkpoint oracle event lifetime mismatch")
        pending = state["pending_boundary_windows"]
        deadline = state["pending_boundary_deadline_step"]
        if type(pending) is not int or not 0 <= pending <= self._ORACLE_TTL_WINDOWS:
            raise ValueError("invalid pending boundary count")
        if state["boundary_notice"] not in self._NOTICE_VALUES:
            raise ValueError("invalid boundary notice")
        if not isinstance(state["base"], dict):
            raise ValueError("invalid base controller checkpoint")
        try:
            saved_config = ControllerConfig(**state["base"]["config"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid underlying controller configuration") from error
        if asdict(saved_config) != asdict(self.config):
            raise ValueError("underlying controller configuration mismatch")
        if type(state["base"].get("version")) is not int:
            raise ValueError("invalid base controller version")
        restored = CriticalPeriodController(list(self.block_names), self.config)
        restored.load_state_dict(state["base"])
        adult_phases = {Phase.ADULT, Phase.REOPENED}
        if state["initial_complete"] != (restored.phase in adult_phases):
            raise ValueError("initial schedule flag contradicts phase")
        if pending:
            if (not self.oracle or not state["initial_complete"] or restored.phase != Phase.ADULT
                    or type(deadline) is not int or restored._last_step is None
                    or not restored._last_step <= deadline
                    <= restored._last_step + self._ORACLE_TTL_WINDOWS * self.config.monitor_interval):
                raise ValueError("invalid pending oracle event")
        elif deadline is not None:
            raise ValueError("oracle deadline without a pending event")
        if not self.oracle and state["boundary_notice"] != "none":
            raise ValueError("autonomous checkpoint cannot contain oracle notices")
        self.__dict__.update(restored.__dict__)
        self._initial_complete = state["initial_complete"]
        self._pending_boundary_windows = pending
        self._pending_boundary_deadline_step = deadline
        self._boundary_notice = state["boundary_notice"]
