"""Causal allocation ablations with a common complete-update norm bound.

V3 has no global critical-period controller. All variants share a fixed warmup,
SGD momentum, and clipping of the COMPLETE proposed optimizer displacement,
including the classifier, weight decay, and anchor forces. Structural resets
happen afterward and are explicitly outside this optimizer-displacement bound.

Unlike V2, consolidation never vetoes recycling. Eligibility depends on base
maturity and, only in the protection/full variants, a separate post-reset age.
This keeps consolidation and extended reset protection experimentally distinct.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import math
from numbers import Real

import torch
from torch import nn

from .plasticity import PlasticityConfig, PlasticityEngine, row_sum, row_view


V3_METHODS = (
    "er_v3", "recycle_v3", "newborn_v3", "newborn_matched_v3",
    "protection_v3", "consolidation_v3", "full_v3",
)
_GAIN_VARIANTS = {"newborn_v3", "newborn_matched_v3", "full_v3"}
_PROTECTION_VARIANTS = {"protection_v3", "full_v3"}
_CONSOLIDATION_VARIANTS = {"consolidation_v3", "full_v3"}


@dataclass(frozen=True)
class V3Config:
    feature_gain: float = 0.1
    warmup_steps: int = 40
    newborn_steps: int = 30
    newborn_peak: float = 1.0
    protection_steps: int = 60
    reset_interval: int = 20
    reset_count: int = 3
    max_update_norm: float = 0.1

    def __post_init__(self):
        for name in ("warmup_steps", "newborn_steps", "protection_steps", "reset_interval", "reset_count"):
            minimum = 0 if name in {"warmup_steps", "protection_steps"} else 1
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("feature_gain", "newborn_peak", "max_update_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if not 0 < self.feature_gain <= 1 or self.newborn_peak < self.feature_gain:
            raise ValueError("require 0 < feature_gain <= 1 and newborn_peak >= feature_gain")
        if self.max_update_norm <= 0:
            raise ValueError("max_update_norm must be positive")


class ScheduledPlasticityEngine(PlasticityEngine):
    """Fixed-count recycling with separately controlled local mechanisms.

    Feature gains are direct multipliers; legacy ``PlasticityConfig.g_min`` and
    legacy recycling fractions/intervals do not enter V3 allocation. In matched
    V3, newborn gains are retained and the OTHER feature parameters receive a
    causally adjusted common gain. This matches the feature mean, not the mature
    backbone gain. Infeasible budgets fail before momentum/parameter mutation.
    Newborn peak gains may exceed one; the common complete-update clip still
    bounds the optimizer displacement, including these amplified coordinates.
    """

    def __init__(self, model: nn.Module, plasticity_config: PlasticityConfig,
                 allocation_config: V3Config, *, variant: str, seed: int):
        if variant not in V3_METHODS:
            raise ValueError(f"unknown V3 variant {variant!r}")
        if not isinstance(allocation_config, V3Config):
            raise TypeError("allocation_config must be V3Config")
        if type(plasticity_config.maturity_steps) is not int or plasticity_config.maturity_steps < 0:
            raise ValueError("V3 base maturity_steps must be a nonnegative integer")
        # Own the local clocks in V3; inherited V2 maturation must stay disabled.
        super().__init__(model, replace(plasticity_config, newborn_steps=0),
                         consolidation=variant in _CONSOLIDATION_VARIANTS,
                         recycling=variant != "er_v3", relaxation=False, seed=seed)
        self.allocation_config = allocation_config
        self.variant = variant
        self.clip_events = 0
        self._last_reset_step: int | None = None
        self._last_update: dict = {}
        self._feature_parameters = sum(p.numel() for name, p in self.params.items()
                                       if self.block_for[name] != "head")
        if not self._feature_parameters:
            raise ValueError("V3 requires at least one feature parameter")
        for state in self.unit_state.values():
            state["reset_age"] = torch.full_like(state["age"], -1)
            state["maturation_pending"] = torch.zeros_like(state["age"], dtype=torch.bool)

    @staticmethod
    def _finite(tensors: dict[str, torch.Tensor], label: str) -> None:
        checks = [torch.isfinite(value).all() for value in tensors.values()]
        if checks and not bool(torch.stack(checks).all()):
            names = [name for name, value in tensors.items() if not bool(torch.isfinite(value).all())]
            raise FloatingPointError(f"non-finite {label}: {', '.join(names)}; no update committed")

    def _allocation(self, step: int) -> tuple[dict[str, torch.Tensor], dict]:
        cfg = self.allocation_config
        base = 1.0 if step <= cfg.warmup_steps else cfg.feature_gain
        gains, masks = {}, {}
        for name, p in self.params.items():
            if self.block_for[name] == "head":
                gains[name] = torch.ones_like(p)
                continue
            gain = torch.full_like(p, base)
            mask = torch.zeros_like(p, dtype=torch.bool)
            if self.variant in _GAIN_VARIANTS and step > cfg.warmup_steps:
                for population, axis in self._newborn_links.get(name, ()):
                    age = self.unit_state[population]["reset_age"]
                    active = (age >= 0) & (age < cfg.newborn_steps)
                    shape = [1] * p.ndim
                    shape[axis] = len(age)
                    active = active.reshape(shape)
                    floor = base + (cfg.newborn_peak-base) * (1-age.to(p.dtype)/cfg.newborn_steps)
                    gain = torch.maximum(gain, torch.where(active, floor.reshape(shape), base))
                    mask |= active.expand_as(p)
            gains[name], masks[name] = gain, mask
        mature_gain, mature_count = base, self._feature_parameters
        if self.variant == "newborn_matched_v3" and step > cfg.warmup_steps:
            newborn_count = sum(int(mask.sum()) for mask in masks.values())
            mature_count = self._feature_parameters-newborn_count
            newborn_sum = sum(float(gains[name][mask].double().sum()) for name, mask in masks.items())
            budget = base*self._feature_parameters
            residual = budget-newborn_sum
            tolerance = 2e-7*max(1.0, budget)
            if residual < -tolerance or (not mature_count and abs(residual) > tolerance):
                raise ValueError("infeasible matched nominal feature budget: newborn gains alone exceed the budget")
            if mature_count:
                mature_gain = max(0.0, residual/mature_count)
                if mature_gain > base + 2e-7:
                    raise ValueError("infeasible matched allocation: mature gain would exceed scheduled base")
                for name, mask in masks.items():
                    gains[name][~mask] = mature_gain
        pre_consolidation = sum(float(gains[name].double().sum()) for name in masks)/self._feature_parameters
        if self.consolidation:
            for name in masks:
                gains[name] = gains[name] * row_view((1-self.state[name]["c"]).pow(self.config.gamma), gains[name])
        nominal = sum(float(gains[name].double().sum()) for name in masks)/self._feature_parameters
        blocks = {}
        for block in self.model.plastic_blocks():
            names = [name for name in gains if self.block_for[name] == block]
            blocks[block] = sum(float(gains[name].double().sum()) for name in names)/sum(self.params[name].numel() for name in names)
        return gains, {
            "gates": blocks, "scheduled_feature_gain": base,
            "mature_feature_gain": mature_gain, "mature_feature_parameter_count": mature_count,
            "pre_consolidation_feature_gain": pre_consolidation,
            "consolidation_gain_reduction": pre_consolidation-nominal,
            "nominal_feature_gain": nominal,
            "matched_budget_residual": nominal-base if self.variant == "newborn_matched_v3" else None,
        }

    @torch.no_grad()
    def gates_for_step(self, step: int | None = None) -> dict[str, float]:
        """Preview parameter-weighted block gains before clipping, without mutation."""
        step = self.steps+1 if step is None else step
        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        return self._allocation(step)[1]["gates"]

    def _propose_maturation(self, parameters: dict, states: dict) -> tuple[dict, int, int]:
        """Prepare local SI commits in copies so errors cannot partially update."""
        units, maturations, useful_units = {}, 0, 0
        threshold = max(self.config.maturity_steps, self.allocation_config.newborn_steps)
        for unit in self.units:
            current = self.unit_state[unit.name]
            proposed = dict(current)
            age = current["reset_age"].clone()
            age[age >= 0] += 1
            proposed["reset_age"] = age
            pending = current["maturation_pending"].clone()
            rows = torch.where(pending & (age >= threshold))[0] if self.consolidation else age[:0]
            useful = torch.zeros_like(age, dtype=torch.bool)
            if len(rows):
                incoming, bias, _ = self._unit_params[unit.name]
                for name in (incoming, bias):
                    if name is None or self.block_for[name] == "head":
                        continue
                    state, p = states[name], parameters[name]
                    utility = row_sum(state["path"][rows].double()).clamp_min(0) / (
                        row_sum((p[rows].double()-state["start"][rows].double()).square()) + self.config.importance_epsilon)
                    self._finite({name: utility}, "local SI proposal")
                    normalized = torch.zeros_like(utility)
                    if bool((utility > 0).any()):
                        scale = torch.quantile(utility, 0.9).clamp_min(torch.finfo(torch.float64).eps)
                        normalized = (utility/scale).clamp(0, 1)
                    positive = utility > 0
                    useful[rows[positive]] = True
                    for key in ("importance", "c", "anchor", "start", "path"):
                        state[key] = state[key].clone()
                    old_c = state["c"][rows]
                    state["importance"][rows] = utility.to(state["importance"].dtype)
                    state["c"][rows] = (1-(1-old_c.double())*torch.exp(-self.config.consolidation_rate*normalized)).to(old_c.dtype)
                    rate = positive.to(p.dtype)*self.config.anchor_rate
                    rate[positive & (old_c == 0)] = 1.0
                    state["anchor"][rows] += row_view(rate, p[rows])*(p[rows]-state["anchor"][rows])
                    state["start"][rows] = p[rows]
                    state["path"][rows] = 0
                pending[rows] = False
                maturations += len(rows)
                useful_units += int(useful.sum())
            proposed["maturation_pending"] = pending
            units[unit.name] = proposed
        return units, maturations, useful_units

    @torch.no_grad()
    def step(self, gradients: dict[str, torch.Tensor], gates: dict | None = None,
             reopened: tuple[str, ...] = ()) -> dict:
        """Propose everything, clip the complete displacement, then commit once.

        ``nominal_feature_gain`` is the final feature-data multiplier BEFORE
        clipping, including any consolidation attenuation. Legacy
        ``effective_feature_gain`` is its achieved post-clip value. Momentum is
        stored before clipping; SI integrates the applied, clipped data update.
        """
        if gates or reopened:
            raise ValueError("V3 uses only its causal schedule, not external gates or reopening")
        if set(gradients) != set(self.params):
            raise ValueError("gradients must cover every model parameter exactly")
        for name, p in self.params.items():
            grad = gradients[name]
            if not isinstance(grad, torch.Tensor) or grad.shape != p.shape or grad.dtype != p.dtype or grad.device != p.device:
                raise ValueError(f"gradient shape/dtype/device mismatch in {name}")
        self._finite(gradients, "gradient")
        gains, allocation = self._allocation(self.steps+1)
        momentum, data, delta = {}, {}, {}
        for name, p in self.params.items():
            state = self.state[name]
            momentum[name] = self.config.momentum*state["momentum"] + gradients[name]
            data[name] = -self.config.lr*gains[name]*momentum[name]
            decay = -self.config.lr*gains[name]*self.config.weight_decay*p
            anchor = -self.config.lr*self.config.anchor_strength*row_view(state["c"], p)*(p-state["anchor"])
            delta[name] = data[name]+decay+anchor
        self._finite(momentum, "momentum proposal")
        self._finite(delta, "optimizer displacement proposal")
        proposed_norm = math.sqrt(sum(float(value.double().square().sum()) for value in delta.values()))
        if not math.isfinite(proposed_norm):
            raise FloatingPointError("non-finite proposed norm; no update committed")
        clip_scale = min(1.0, self.allocation_config.max_update_norm/proposed_norm) if proposed_norm else 1.0
        parameters, states, applied = {}, {}, {}
        feature_data_sq = head_data_sq = feature_sq = head_sq = 0.0
        for name, p in self.params.items():
            applied[name] = delta[name]*clip_scale
            applied_data = data[name]*clip_scale
            parameters[name] = p+applied[name]
            state = dict(self.state[name])
            state["momentum"] = momentum[name]
            state["age"] = state["age"]+1
            if self.consolidation and self.block_for[name] != "head":
                state["path"] = state["path"]-gradients[name]*applied_data
            states[name] = state
            data_sq = float(applied_data.double().square().sum())
            full_sq = float(applied[name].double().square().sum())
            if self.block_for[name] == "head":
                head_data_sq += data_sq
                head_sq += full_sq
            else:
                feature_data_sq += data_sq
                feature_sq += full_sq
        self._finite(parameters, "parameter proposal")
        units, matured, consolidated = self._propose_maturation(parameters, states)
        self._finite({f"{name}/{key}": value for name, state in states.items()
                      for key, value in state.items() if value.is_floating_point()}, "optimizer-state proposal")
        # All checks, including matched-budget feasibility and maturation, have
        # completed. No model or momentum state changed before this point.
        for name, p in self.params.items():
            p.copy_(parameters[name])
        self.state, self.unit_state = states, units
        self.steps += 1
        clipped = clip_scale < 1.0
        self.clip_events += int(clipped)
        self.total_maturations += matured
        self.total_local_consolidations += consolidated
        event = {
            **allocation, "step": self.steps, "variant": self.variant,
            "proposed_update_norm": proposed_norm, "update_norm": math.sqrt(feature_sq+head_sq),
            "applied_update_norm": math.sqrt(feature_sq+head_sq), "clip_scale": clip_scale,
            "clipped": clipped, "clip_events": self.clip_events, "clip_frequency": self.clip_events/self.steps,
            "preclip_feature_gain": allocation["nominal_feature_gain"],
            "effective_feature_gain": allocation["nominal_feature_gain"]*clip_scale,
            "postclip_feature_gain": allocation["nominal_feature_gain"]*clip_scale,
            "feature_data_displacement": math.sqrt(feature_data_sq),
            "head_data_displacement": math.sqrt(head_data_sq),
            "feature_displacement": math.sqrt(feature_sq), "head_displacement": math.sqrt(head_sq),
            "feature_parameter_count": self._feature_parameters,
            "matured_units": matured, "locally_consolidated_units": consolidated,
            "total_maturations": self.total_maturations,
            "total_local_consolidations": self.total_local_consolidations,
            **self._newborn_counts(),
        }
        self._last_update = copy.deepcopy(event)
        return event

    def consolidate(self, blocks) -> None:
        raise ValueError("V3 consolidation is local to reset-unit maturation; global events are not supported")

    def _newborn_counts(self) -> dict[str, int]:
        newborn, protected, pending = 0, 0, 0
        for state in self.unit_state.values():
            age = state["reset_age"]
            if self.variant in _GAIN_VARIANTS:
                newborn += int(((age >= 0) & (age < self.allocation_config.newborn_steps)).sum())
            if self.variant in _PROTECTION_VARIANTS:
                protected += int(((age >= 0) & (age < self.allocation_config.protection_steps)).sum())
            pending += int(state["maturation_pending"].sum())
        return {"newborn_units": newborn, "protected_units": protected, "pending_maturation_units": pending}

    @torch.no_grad()
    def recycle(self, *, adult: bool = False, forced_counts: dict | None = None) -> dict:
        """Reset exactly the configured count per population on scheduled steps.

        Warmup has no resets; the first is warmup_steps+reset_interval. Scheduled
        repeated calls at the same optimizer step are idempotent. Consolidation
        scores never veto resets. All eligibility checks and random replacement
        values are prepared before mutation, in downstream-first order.
        """
        if forced_counts is not None:
            raise ValueError("V3 does not accept offline or externally forced reset traces")
        cfg = self.allocation_config
        empty = {"recycled": 0, "recycled_by_population": {unit.name: 0 for unit in self.units}, "scheduled_reset": False,
                 "eligible_by_population": {}, "reset_indices": {}}
        if not self.recycling or self.steps <= cfg.warmup_steps or (self.steps-cfg.warmup_steps) % cfg.reset_interval or self._last_reset_step == self.steps:
            return empty
        pools = {}
        for unit in reversed(self.units):
            state = self.unit_state[unit.name]
            eligible = state["age"] >= self.config.maturity_steps
            if self.variant in _PROTECTION_VARIANTS:
                age = state["reset_age"]
                eligible &= (age < 0) | (age >= cfg.protection_steps)
            pools[unit.name] = torch.where(eligible)[0]
            if len(pools[unit.name]) < cfg.reset_count:
                raise ValueError(f"infeasible scheduled reset for {unit.name} at step {self.steps}: "
                                 f"requested {cfg.reset_count}, eligible {len(pools[unit.name])}")
        generator = torch.Generator(device="cpu")
        generator.set_state(self.generator.get_state())
        plans = []
        for unit in reversed(self.units):
            eligible = pools[unit.name]
            state = self.unit_state[unit.name]
            correction = (1-self.config.utility_decay**state["age"][eligible].float()).clamp_min(1e-8)
            utility = state["utility"][eligible]/correction
            self._finite({unit.name: utility}, "recycling utility")
            indices = eligible[torch.argsort(utility, stable=True)[:cfg.reset_count]]
            p = unit.incoming.weight
            values = torch.empty((len(indices), *p.shape[1:]), dtype=p.dtype, device="cpu")
            values.uniform_(-math.sqrt(6/p[0].numel()), math.sqrt(6/p[0].numel()), generator=generator)
            plans.append((unit, indices, values.to(p.device)))
        counts, identities = {}, {}
        for unit, indices, values in plans:
            incoming, bias, outgoing = self._unit_params[unit.name]
            unit.incoming.weight[indices] = values
            if unit.incoming.bias is not None:
                unit.incoming.bias[indices] = 0
            unit.outgoing.weight[:, indices] = 0
            for name in (incoming, bias):
                if name is None:
                    continue
                state, p = self.state[name], self.params[name]
                for key in ("momentum", "path", "c", "importance", "age"):
                    state[key][indices] = 0
                for key in ("anchor", "start"):
                    state[key][indices] = p[indices]
            for key in ("momentum", "path", "anchor", "start"):
                self.state[outgoing][key][:, indices] = 0
            state = self.unit_state[unit.name]
            for key in ("utility", "mean", "age"):
                state[key][indices] = 0
            state["reset_age"][indices] = 0
            state["maturation_pending"][indices] = self.consolidation
            identities[unit.name] = indices.cpu().tolist()
            counts[unit.name] = len(indices)
        self.generator.set_state(generator.get_state())
        self._last_reset_step = self.steps
        self.total_recycled += sum(counts.values())
        return {"recycled": sum(counts.values()), "recycled_by_population": counts, "scheduled_reset": True,
                "eligible_by_population": {name: len(pool) for name, pool in pools.items()},
                "reset_indices": identities,
                "reset_identity_sha256": hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()}

    def diagnostics(self) -> dict:
        return {**super().diagnostics(), "variant": self.variant, "clip_events": self.clip_events,
                "clip_frequency": self.clip_events/max(1, self.steps),
                "last_nominal_feature_gain": self._last_update.get("nominal_feature_gain"),
                "last_effective_feature_gain": self._last_update.get("effective_feature_gain")}

    def state_dict(self) -> dict:
        return {**super().state_dict(), "v3": {
            "version": 1, "variant": self.variant, "allocation_config": asdict(self.allocation_config),
            "plasticity_config": asdict(self.config), "clip_events": self.clip_events,
            "last_reset_step": self._last_reset_step, "last_update": copy.deepcopy(self._last_update),
        }}

    def load_state_dict(self, state: dict) -> None:
        metadata = state.get("v3", {})
        expected = {"version": 1, "variant": self.variant, "allocation_config": asdict(self.allocation_config),
                    "plasticity_config": asdict(self.config)}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("V3 checkpoint variant/config/version mismatch")
        if state.get("newborn_steps") != 0:
            raise ValueError("V3 checkpoint must not enable inherited V2 clocks")
        for key in ("steps", "total_recycled", "total_maturations", "total_local_consolidations"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError(f"invalid checkpoint {key}")
        clips = metadata.get("clip_events")
        if type(clips) is not int or not 0 <= clips <= state["steps"]:
            raise ValueError("invalid checkpoint clip_events")
        last_reset = metadata.get("last_reset_step")
        if last_reset is not None and (type(last_reset) is not int or not 0 < last_reset <= state["steps"]):
            raise ValueError("invalid checkpoint last_reset_step")
        if not isinstance(metadata.get("last_update"), dict):
            raise ValueError("invalid checkpoint last_update")
        last_update = copy.deepcopy(metadata["last_update"])
        prepared = {}
        for group, template in (("state", self.state), ("units", self.unit_state)):
            saved = state.get(group, {})
            if saved.keys() != template.keys():
                raise ValueError(f"checkpoint {group} populations/parameters differ")
            prepared[group] = {}
            for name, reference in template.items():
                if saved[name].keys() != reference.keys():
                    raise ValueError(f"checkpoint fields differ for {group}/{name}")
                values = {}
                for key, initial in reference.items():
                    value = saved[name][key]
                    if isinstance(initial, torch.Tensor):
                        if not isinstance(value, torch.Tensor) or value.shape != initial.shape or value.dtype != initial.dtype:
                            raise ValueError(f"checkpoint tensor differs for {group}/{name}/{key}")
                        value = value.detach().to(initial.device).clone()
                        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                            raise ValueError("checkpoint contains non-finite state")
                    elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                        raise ValueError("checkpoint contains invalid scalar state")
                    values[key] = value
                prepared[group][name] = values
        generator = torch.Generator(device="cpu")
        generator.set_state(state["rng"].cpu())
        self.state, self.unit_state = prepared["state"], prepared["units"]
        self.steps, self.total_recycled = state["steps"], state["total_recycled"]
        self.total_maturations = state["total_maturations"]
        self.total_local_consolidations = state["total_local_consolidations"]
        self.generator.set_state(generator.get_state())
        self.clip_events, self._last_reset_step = clips, last_reset
        self._last_update = last_update
