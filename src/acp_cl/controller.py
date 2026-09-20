"""Deterministic, task-boundary-free critical-period plasticity control.

Only call :meth:`CriticalPeriodController.observe` at monitoring windows. Gates
returned after an observation apply to the *next* optimizer updates, so a replay
damage guard can limit subsequent changes but cannot undo damage already seen.
All controller state is plain Python and can be checkpointed as strict JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
import math
from numbers import Real
from typing import Any


class Phase(str, Enum):
    SCAFFOLD = "scaffold"
    OPEN = "open"
    CLOSING = "closing"
    ADULT = "adult"
    REOPENED = "reopened"


@dataclass(frozen=True)
class ControllerConfig:
    """Controller priors, not benchmark-tuned settings.

    Step durations use optimizer updates; ``*_windows`` count monitoring calls.
    Adaptive closure never has a time limit. ``min_open_steps`` starts when the
    scaffold completes. The fixed ablation instead uses absolute step boundaries
    and closes regardless of observed stability; it never reopens.
    """

    monitor_interval: int = 25
    scaffold_steps: int = 300
    min_open_steps: int = 300
    stable_windows: int = 3
    stability_loss_tol: float = 0.01
    stability_drift_tol: float = 0.02
    max_replay_damage: float = 0.10
    closing_windows: int = 6
    g_min: float = 0.05
    novelty_bias: float = 2.0
    open_threshold: float = 0.70
    novel_windows: int = 3
    reopen_max_windows: int = 12
    cooldown_windows: int = 4
    top_k: int = 2
    adaptive: bool = True
    allow_reopening: bool = True
    local_gating: bool = True
    use_replay_damage: bool = True
    use_plasticity_signal: bool = True

    def __post_init__(self) -> None:
        nonnegative = {"scaffold_steps", "min_open_steps", "cooldown_windows"}
        integers = nonnegative | {
            "monitor_interval", "stable_windows", "closing_windows",
            "novel_windows", "reopen_max_windows", "top_k",
        }
        switches = {
            "adaptive", "allow_reopening", "local_gating",
            "use_replay_damage", "use_plasticity_signal",
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in integers:
                minimum = 0 if field.name in nonnegative else 1
                if type(value) is not int or value < minimum:
                    raise ValueError(f"{field.name} must be an integer >= {minimum}")
            elif field.name in switches:
                if type(value) is not bool:
                    raise ValueError(f"{field.name} must be a boolean")
            else:
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(f"{field.name} must be a finite number")
                object.__setattr__(self, field.name, float(value))
        for name in ("stability_loss_tol", "stability_drift_tol", "max_replay_damage"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.g_min < 1:
            raise ValueError("g_min must satisfy 0 <= g_min < 1")
        if not 0 < self.open_threshold < 1:
            raise ValueError("open_threshold must satisfy 0 < open_threshold < 1")


@dataclass(frozen=True)
class MonitorSignals:
    """Training-only monitoring observations; no experience or task identifier."""

    surprise: float = 0.0
    drift: float = 0.0
    plasticity_deficit: float = 0.0
    replay_damage: float = 0.0
    loss_change: float = 1.0
    has_replay: bool = False


class CriticalPeriodController:
    """Open, consolidate, and selectively reopen named representation blocks."""

    _SIGNAL_LIMIT = 60.0
    _STATE_VERSION = 1

    def __init__(self, block_names: list[str], config: ControllerConfig) -> None:
        if isinstance(block_names, str) or not block_names:
            raise ValueError("block_names must be a nonempty sequence")
        if any(not isinstance(name, str) or not name for name in block_names):
            raise ValueError("block names must be nonempty strings")
        if len(set(block_names)) != len(block_names):
            raise ValueError("block names must be unique")
        if not isinstance(config, ControllerConfig):
            raise TypeError("config must be a ControllerConfig")
        self.block_names = tuple(block_names)
        self.config = config
        self.phase = Phase.SCAFFOLD
        self.selected_blocks: tuple[str, ...] = ()
        self.novelty = 0.0
        self.should_consolidate = False
        self.consolidation_blocks: tuple[str, ...] = ()
        self._last_step: int | None = None
        self._phase_start_step = 0
        self._observations = 0
        self._stable_count = 0
        self._novel_count = 0
        self._closing_count = 0
        self._closing_progress = 0.0
        self._reopen_count = 0
        self._cooldown_remaining = 0
        self._risk_active = False

    def _sanitize(self, signals: MonitorSignals) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(signals, MonitorSignals):
            raise TypeError("signals must be MonitorSignals")
        if type(signals.has_replay) is not bool:
            raise ValueError("has_replay must be a boolean")
        result: dict[str, Any] = {"has_replay": signals.has_replay}
        invalid = []
        for name in ("surprise", "drift", "plasticity_deficit", "replay_damage", "loss_change"):
            raw = getattr(signals, name)
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise ValueError(f"{name} must be numeric")
            value = float(raw)
            if not math.isfinite(value):
                invalid.append(name)
                # An unobserved loss/drift/risk is not evidence of stability.
                if math.isnan(value):
                    value = self._SIGNAL_LIMIT if name in {"drift", "replay_damage", "loss_change"} else 0.0
                else:
                    value = math.copysign(self._SIGNAL_LIMIT, value)
            lower = 0.0 if name in {"drift", "plasticity_deficit"} else -self._SIGNAL_LIMIT
            result[name] = min(self._SIGNAL_LIMIT, max(lower, value))
        return result, invalid

    def _select(self, block_scores: dict[str, float]) -> tuple[str, ...]:
        if not self.config.local_gating:
            return self.block_names
        scores = {}
        for name in self.block_names:
            value = block_scores.get(name, 0.0)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"block score for {name!r} must be numeric")
            # Nonfinite scores sort last. Ties preserve model order.
            scores[name] = float(value) if math.isfinite(value) else -math.inf
        ranked = sorted(self.block_names, key=lambda name: -scores[name])
        return tuple(ranked[: self.config.top_k])

    def _adult(self, step: int, blocks: tuple[str, ...]) -> None:
        self.phase = Phase.ADULT
        self._phase_start_step = step
        self.should_consolidate = True
        self.consolidation_blocks = blocks
        self.selected_blocks = ()
        self._stable_count = 0
        self._novel_count = 0
        self._reopen_count = 0
        self._cooldown_remaining = self.config.cooldown_windows

    def _fixed_schedule(self, step: int) -> str:
        config = self.config
        open_end = config.scaffold_steps + config.min_open_steps
        close_end = open_end + config.closing_windows * config.monitor_interval
        if step < config.scaffold_steps:
            self.phase = Phase.SCAFFOLD
            self._phase_start_step = 0
        elif step < open_end:
            self.phase = Phase.OPEN
            self._phase_start_step = config.scaffold_steps
        elif step < close_end:
            self.phase = Phase.CLOSING
            self._phase_start_step = open_end
            self._closing_progress = (step - open_end) / (close_end - open_end)
        elif self.phase != Phase.ADULT:
            self._closing_progress = 1.0
            self._adult(step, self.block_names)
            self._cooldown_remaining = 0
        return "fixed_schedule"

    def observe(
        self, step: int, signals: MonitorSignals, block_scores: dict[str, float]
    ) -> dict[str, object]:
        """Advance once and return a JSON-safe controller audit record.

        At least ``monitor_interval`` optimizer updates must separate calls. This
        rejects accidental minibatch-level counting of consecutive detections.
        Missing replay observations disable the replay signal for that window;
        they do not silently assert that replay damage was measured to be zero.
        """
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if self._last_step is not None and step - self._last_step < self.config.monitor_interval:
            raise ValueError("observe calls must be at least monitor_interval updates apart")
        values, invalid = self._sanitize(signals)
        # Validate scores before changing state, even when this window does not select.
        selection = self._select(block_scores)
        config = self.config
        replay_observed = config.use_replay_damage and signals.has_replay
        replay_damage = max(0.0, values["replay_damage"]) if replay_observed else 0.0
        plasticity = values["plasticity_deficit"] if config.use_plasticity_signal else 0.0
        relevant_invalid = set(invalid) - ({"replay_damage"} if not replay_observed else set())
        if not config.use_plasticity_signal:
            relevant_invalid.discard("plasticity_deficit")
        logit = values["surprise"] + values["drift"] + 0.5 * plasticity - 1.5 * replay_damage - config.novelty_bias
        logit = max(-self._SIGNAL_LIMIT, min(self._SIGNAL_LIMIT, logit))
        self.novelty = 1.0 / (1.0 + math.exp(-logit))
        replay_risk = replay_observed and replay_damage > config.max_replay_damage
        self._risk_active = bool(config.adaptive and replay_risk)
        stable = bool(
            not relevant_invalid
            and abs(values["loss_change"]) < config.stability_loss_tol
            and values["drift"] < config.stability_drift_tol
            and (not replay_observed or replay_damage < config.max_replay_damage)
        )
        high_novelty = bool(not relevant_invalid and not replay_risk and self.novelty > config.open_threshold)
        previous = self.phase
        self.should_consolidate = False
        self.consolidation_blocks = ()
        self._observations += 1
        reason = "hold"

        if not config.adaptive:
            reason = self._fixed_schedule(step)
        elif self.phase == Phase.SCAFFOLD:
            if step >= config.scaffold_steps:
                self.phase = Phase.OPEN
                self._phase_start_step = step
                reason = "scaffold_complete"
            else:
                reason = "minimum_scaffold"
        elif self.phase == Phase.OPEN:
            old_enough = step - self._phase_start_step >= config.min_open_steps
            self._stable_count = self._stable_count + 1 if old_enough and stable else 0
            if self._stable_count >= config.stable_windows:
                self.phase = Phase.CLOSING
                self._phase_start_step = step
                self._closing_count = 0
                self._closing_progress = 0.0
                self._stable_count = 0
                reason = "sustained_stability"
            else:
                reason = "awaiting_stability" if old_enough else "minimum_open"
        elif self.phase == Phase.CLOSING:
            self._closing_count += 1
            self._closing_progress = min(1.0, self._closing_count / config.closing_windows)
            reason = "consolidation_ramp"
            if self._closing_count >= config.closing_windows:
                self._adult(step, self.block_names)
                reason = "initial_consolidation_complete"
        elif self.phase == Phase.ADULT:
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                self._novel_count = 0
                reason = "cooldown"
            elif not config.allow_reopening:
                self._novel_count = 0
                reason = "reopening_disabled"
            else:
                self._novel_count = self._novel_count + 1 if high_novelty else 0
                if self._novel_count >= config.novel_windows:
                    self.phase = Phase.REOPENED
                    self._phase_start_step = step
                    self.selected_blocks = selection
                    self._reopen_count = 0
                    self._stable_count = 0
                    self._novel_count = 0
                    reason = "sustained_novelty"
                else:
                    reason = "replay_risk" if replay_risk else "awaiting_novelty"
        elif self.phase == Phase.REOPENED:
            self._reopen_count += 1
            self._stable_count = self._stable_count + 1 if stable else 0
            if self._stable_count >= config.stable_windows:
                self._adult(step, self.selected_blocks)
                reason = "adaptation_stable"
            elif (
                self._reopen_count >= config.reopen_max_windows
                or step - self._phase_start_step >= config.reopen_max_windows * config.monitor_interval
            ):
                self._adult(step, self.selected_blocks)
                reason = "reopening_timeout"
            else:
                reason = "replay_risk" if replay_risk else "adapting"

        self._last_step = step
        return {
            "step": step,
            "observation": self._observations,
            "phase": self.phase.value,
            "previous_phase": previous.value,
            "transition": previous != self.phase,
            "reason": reason,
            "novelty": self.novelty,
            "signals": values,
            "invalid_signals": invalid,
            "stable": stable,
            "high_novelty": high_novelty,
            "replay_observed": replay_observed,
            "replay_risk": replay_risk,
            "safety_contracted": self._risk_active,
            "selected_blocks": list(self.selected_blocks),
            "gates": self.gates(),
            "should_consolidate": self.should_consolidate,
            "consolidation_blocks": list(self.consolidation_blocks),
            "stable_count": self._stable_count,
            "novel_count": self._novel_count,
            "closing_progress": self._closing_progress,
            "reopen_windows": self._reopen_count,
            "cooldown_remaining": self._cooldown_remaining,
            "phase_age_steps": step - self._phase_start_step,
        }

    def gates(self) -> dict[str, float]:
        """Current multiplicative update gates (classification head excluded)."""
        floor = self.config.g_min
        if self._risk_active or self.phase == Phase.ADULT:
            return {name: floor for name in self.block_names}
        if self.phase in (Phase.SCAFFOLD, Phase.OPEN):
            return {name: 1.0 for name in self.block_names}
        if self.phase == Phase.CLOSING:
            gate = floor + (1.0 - floor) * (1.0 - self._closing_progress)
            return {name: gate for name in self.block_names}
        return {
            name: floor + (1.0 - floor) * self.novelty if name in self.selected_blocks else floor
            for name in self.block_names
        }

    def state_dict(self) -> dict[str, object]:
        """Return an independent, versioned, JSON-serializable checkpoint."""
        return {
            "version": self._STATE_VERSION,
            "block_names": list(self.block_names),
            "config": asdict(self.config),
            "phase": self.phase.value,
            "selected_blocks": list(self.selected_blocks),
            "novelty": self.novelty,
            "should_consolidate": self.should_consolidate,
            "consolidation_blocks": list(self.consolidation_blocks),
            "last_step": self._last_step,
            "phase_start_step": self._phase_start_step,
            "observations": self._observations,
            "stable_count": self._stable_count,
            "novel_count": self._novel_count,
            "closing_count": self._closing_count,
            "closing_progress": self._closing_progress,
            "reopen_count": self._reopen_count,
            "cooldown_remaining": self._cooldown_remaining,
            "risk_active": self._risk_active,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore compatible state atomically; reject mismatched model/config."""
        if state.get("version") != self._STATE_VERSION:
            raise ValueError("unsupported controller state version")
        if state.get("block_names") != list(self.block_names):
            raise ValueError("controller block names do not match checkpoint")
        if state.get("config") != asdict(self.config):
            raise ValueError("controller config does not match checkpoint")
        restored = CriticalPeriodController(list(self.block_names), self.config)
        try:
            restored.phase = Phase(state["phase"])
            for key in ("selected_blocks", "consolidation_blocks"):
                names = state[key]
                if not isinstance(names, (list, tuple)) or any(name not in self.block_names for name in names) or len(set(names)) != len(names):
                    raise ValueError(f"invalid {key}")
                setattr(restored, key, tuple(names))
            for key in ("novelty", "closing_progress"):
                value = state[key]
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"invalid {key}")
                setattr(restored, key if key == "novelty" else "_" + key, float(value))
            for key in ("phase_start_step", "observations", "stable_count", "novel_count", "closing_count", "reopen_count", "cooldown_remaining"):
                value = state[key]
                if type(value) is not int or value < 0:
                    raise ValueError(f"invalid {key}")
                setattr(restored, "_" + key, value)
            last_step = state["last_step"]
            if last_step is not None and (type(last_step) is not int or last_step < restored._phase_start_step):
                raise ValueError("invalid last_step")
            restored._last_step = last_step
            for key in ("should_consolidate", "risk_active"):
                if type(state[key]) is not bool:
                    raise ValueError(f"invalid {key}")
                setattr(restored, key if key == "should_consolidate" else "_" + key, state[key])
        except (KeyError, TypeError) as error:
            raise ValueError("incomplete or malformed controller state") from error
        if bool(restored.selected_blocks) != (restored.phase == Phase.REOPENED):
            raise ValueError("selected blocks are required only during reopening")
        if restored.should_consolidate != bool(restored.consolidation_blocks):
            raise ValueError("consolidation event and blocks disagree")
        if restored._cooldown_remaining > self.config.cooldown_windows:
            raise ValueError("invalid cooldown_remaining")
        if restored._reopen_count > self.config.reopen_max_windows or restored._closing_count > self.config.closing_windows:
            raise ValueError("window counter exceeds configured bound")
        self.__dict__.update(restored.__dict__)
