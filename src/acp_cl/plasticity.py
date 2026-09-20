"""Channel consolidation, momentum-safe update gating, and structural recycling."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class PlasticityConfig:
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    g_min: float = 0.05
    gamma: float = 2.0
    anchor_strength: float = 1e-3
    consolidation_rate: float = 0.1
    anchor_rate: float = 0.5
    importance_epsilon: float = 1e-3
    relax_max: float = 0.5
    core_fraction: float = 0.2
    maturity_steps: int = 250
    recycle_interval: int = 250
    recycle_fraction: float = 0.005
    adult_recycle_fraction: float = 0.001
    recycle_cutoff: float = 0.3
    utility_decay: float = 0.99
    # Only replaced units receive this local developmental window. Zero keeps
    # the original controller-only consolidation and block-gating behavior.
    newborn_steps: int = 0

    def __post_init__(self):
        if type(self.newborn_steps) is not int or self.newborn_steps < 0:
            raise ValueError("newborn_steps must be a nonnegative integer")
        numeric = asdict(self)
        if any(not math.isfinite(v) for v in numeric.values()):
            raise ValueError("plasticity settings must be finite")
        if self.lr <= 0 or self.gamma <= 0 or self.importance_epsilon <= 0:
            raise ValueError("lr, gamma, and importance_epsilon must be positive")
        for key in ("momentum", "utility_decay"):
            if not 0 <= numeric[key] < 1:
                raise ValueError(f"{key} must be in [0, 1)")
        for key in ("g_min", "anchor_rate", "relax_max", "core_fraction",
                    "recycle_fraction", "adult_recycle_fraction", "recycle_cutoff"):
            if not 0 <= numeric[key] <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if min(self.weight_decay, self.anchor_strength, self.consolidation_rate) < 0:
            raise ValueError("regularization and consolidation strengths cannot be negative")
        if self.maturity_steps < 0 or self.recycle_interval < 1:
            raise ValueError("invalid maturity or recycling interval")


def row_sum(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1).sum(dim=1)


def row_view(vector: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return vector.reshape(-1, *([1] * (tensor.ndim - 1)))


def robust_unit_interval(values: torch.Tensor) -> torch.Tensor:
    """Nonnegative utility normalized by its 90th percentile, zero-safe."""
    values = values.clamp_min(0)
    if not bool(values.any()):
        return torch.zeros_like(values)
    scale = torch.quantile(values.float(), 0.9).clamp_min(torch.finfo(torch.float32).eps)
    return (values / scale).clamp(0, 1)


class PlasticityEngine:
    """SGD with gates applied AFTER momentum, and an independent anchor force.

    Row-level importance is SI-inspired: a data-update path integral divided by
    squared displacement from the beginning of its accumulation window. The
    regularization anchor is deliberately a separate snapshot. The shared head
    is always ungated/unconsolidated to avoid suppressing unseen output classes.
    """

    def __init__(self, model: nn.Module, config: PlasticityConfig, *,
                 consolidation: bool = True, recycling: bool = True,
                 relaxation: bool = True, random_recycling: bool = False, seed: int = 0):
        self.model, self.config = model, config
        self.consolidation, self.recycling = consolidation, recycling
        self.relaxation, self.random_recycling = relaxation, random_recycling
        self.params = dict(model.named_parameters())
        self.block_for: dict[str, str] = {}
        owner = {id(p): name for name, p in self.params.items()}
        for block, module in model.plastic_blocks().items():
            for p in module.parameters():
                name = owner[id(p)]
                if name in self.block_for:
                    raise ValueError(f"parameter {name} belongs to multiple blocks")
                self.block_for[name] = block
        if set(self.block_for) != set(self.params):
            raise ValueError("plastic_blocks must cover every trainable parameter")
        self.state = {}
        for name, p in self.params.items():
            self.state[name] = {
                "momentum": torch.zeros_like(p), "anchor": p.detach().clone(),
                "start": p.detach().clone(), "path": torch.zeros_like(p),
                "c": torch.zeros(p.shape[0], device=p.device),
                "importance": torch.zeros(p.shape[0], device=p.device),
                "age": torch.zeros(p.shape[0], dtype=torch.long, device=p.device),
            }
        self.units = model.recyclable_units()
        if len({unit.name for unit in self.units}) != len(self.units):
            raise ValueError("recyclable population names must be unique")
        self._unit_params = {}
        self._newborn_links: dict[str, list[tuple[str, int]]] = {}
        for unit in self.units:
            incoming = owner[id(unit.incoming.weight)]
            bias = owner[id(unit.incoming.bias)] if unit.incoming.bias is not None else None
            outgoing = owner[id(unit.outgoing.weight)]
            if unit.outgoing.weight.shape[1] != unit.incoming.weight.shape[0]:
                raise ValueError(f"population {unit.name} has incompatible outgoing columns")
            self._unit_params[unit.name] = (incoming, bias, outgoing)
            for name in (incoming, bias):
                if name is not None:
                    self._newborn_links.setdefault(name, []).append((unit.name, 0))
            self._newborn_links.setdefault(outgoing, []).append((unit.name, 1))
        self.unit_state = {
            u.name: {"utility": torch.zeros(u.incoming.weight.shape[0], device=u.incoming.weight.device),
                     "mean": torch.zeros(u.incoming.weight.shape[0], device=u.incoming.weight.device),
                     "age": torch.zeros(u.incoming.weight.shape[0], dtype=torch.long,
                                        device=u.incoming.weight.device), "credit": 0.0,
                     # -1 means initial or already matured; 0 is an actual reset.
                     # This clock advances in step(), independent of EMA updates.
                     "newborn_age": torch.full((u.incoming.weight.shape[0],), -1,
                                               dtype=torch.long, device=u.incoming.weight.device)}
            for u in self.units
        }
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.steps = 0
        self.total_recycled = 0
        self.total_maturations = 0
        self.total_local_consolidations = 0

    def block_consolidation(self) -> dict[str, float]:
        result = {}
        for block in self.model.plastic_blocks():
            values = [s["c"] for name, s in self.state.items() if self.block_for[name] == block]
            result[block] = float(torch.cat(values).mean())
        return result

    def _effective_c(self, name: str, reopened: set[str]) -> torch.Tensor:
        s = self.state[name]
        c = s["c"]
        if not self.relaxation or self.block_for[name] not in reopened:
            return c
        # Core is selected from positive consolidated importance; all-zero rows
        # must not become a hard core merely because their indices win a tie.
        relaxation = self.config.relax_max * (1 - robust_unit_interval(s["importance"]))
        core_score = c * s["importance"]
        positive = torch.where(core_score > 0)[0]
        count = min(math.ceil(c.numel() * self.config.core_fraction), len(positive))
        if count:
            core = positive[torch.topk(core_score[positive], count).indices]
            relaxation = relaxation.clone()
            relaxation[core] = 0
        return c * (1-relaxation)

    def _newborn_scale(self, name: str, parameter: torch.Tensor,
                       scale: torch.Tensor, gate: float) -> torch.Tensor:
        """A final gain floor on reset incoming rows and outgoing columns.

        At post-reset ages a=0,...,N-1 the floor is g+(1-g)*(1-a/N).
        At age N it returns to the ordinary consolidated block multiplier.
        Incident row/column floors combine by maximum. The head remains at one.
        This is an explicit local exception even when replay risk contracts the
        global block gate to g_min; paired replay probes measure its consequences.
        """
        if not self.config.newborn_steps:
            return scale
        for population, axis in self._newborn_links.get(name, ()):
            age = self.unit_state[population]["newborn_age"]
            active = (age >= 0) & (age < self.config.newborn_steps)
            if not bool(active.any()):
                continue
            remaining = 1 - age.to(parameter.dtype) / self.config.newborn_steps
            floor = torch.where(active, gate + (1-gate) * remaining, 0.0)
            shape = [1] * parameter.ndim
            shape[axis] = len(floor)
            scale = torch.maximum(scale, floor.reshape(shape))
        return scale

    def _pending_rows(self, name: str) -> torch.Tensor:
        """Incoming rows whose post-reset SI window must remain uninterrupted."""
        pending = torch.zeros_like(self.state[name]["age"], dtype=torch.bool)
        if self.config.newborn_steps:
            for population, axis in self._newborn_links.get(name, ()):
                if axis == 0:
                    pending |= self.unit_state[population]["newborn_age"] >= 0
        return pending

    def _consolidate_rows(self, name: str, rows: torch.Tensor, *,
                          useful_only: bool = False) -> torch.Tensor:
        """Commit an SI window for selected rows; return useful matured rows."""
        if not len(rows):
            return rows
        cfg, p, s = self.config, self.params[name], self.state[name]
        utility = row_sum(s["path"][rows]).clamp_min(0) / (
            row_sum((p[rows]-s["start"][rows]).square()) + cfg.importance_epsilon)
        normalized = robust_unit_interval(utility)
        mature = s["age"][rows] >= cfg.maturity_steps
        useful = mature & (utility > 0)
        normalized *= mature
        s["importance"][rows] = utility
        old_c = s["c"][rows]
        s["c"][rows] = 1 - (1-old_c) * torch.exp(-cfg.consolidation_rate * normalized)
        anchor_rows = useful if useful_only else mature
        anchor_rate = anchor_rows.to(p.dtype) * cfg.anchor_rate
        if useful_only:
            # First local acquisition anchors the trained row, rather than
            # retaining half of its random reset initialization. Global/v0
            # consolidation keeps the configured EMA anchor rule unchanged.
            anchor_rate[useful & (old_c == 0)] = 1.0
        rate = row_view(anchor_rate, p[rows])
        s["anchor"][rows] = s["anchor"][rows] + rate * (p[rows]-s["anchor"][rows])
        s["start"][rows] = p[rows]
        s["path"][rows] = 0
        return rows[useful]

    def _advance_newborns(self) -> tuple[int, int]:
        """Graduate once at max(local window, maturity), independent of phase."""
        matured, consolidated = 0, 0
        if not self.config.newborn_steps:
            return matured, consolidated
        threshold = max(self.config.newborn_steps, self.config.maturity_steps)
        for unit in self.units:
            age = self.unit_state[unit.name]["newborn_age"]
            pending = age >= 0
            age[pending] += 1
            rows = torch.where(pending & (age >= threshold))[0]
            if not len(rows):
                continue
            useful = torch.zeros_like(age, dtype=torch.bool)
            if self.consolidation:
                incoming, bias, _ = self._unit_params[unit.name]
                for name in (incoming, bias):
                    if name is not None and self.block_for[name] != "head":
                        useful[self._consolidate_rows(name, rows, useful_only=True)] = True
            age[rows] = -1
            matured += len(rows)
            consolidated += int(useful.sum())
        self.total_maturations += matured
        self.total_local_consolidations += consolidated
        return matured, consolidated

    def _newborn_counts(self) -> dict[str, int]:
        pending, active = 0, 0
        for state in self.unit_state.values():
            age = state["newborn_age"]
            pending += int((age >= 0).sum())
            active += int(((age >= 0) & (age < self.config.newborn_steps)).sum())
        return {"newborn_units": active, "pending_maturation_units": pending}

    @torch.no_grad()
    def step(self, gradients: dict[str, torch.Tensor], gates: dict[str, float],
             reopened: tuple[str, ...] = ()) -> dict[str, float]:
        """Update once and report final non-head gain and data displacement.

        Gain is averaged over feature parameters, including incoming biases.
        Data displacement is the L2 norm of the actual momentum/data update;
        weight decay, anchor forces, and classifier parameters are excluded.
        Newborn counts describe the state after this optimizer update.
        """
        cfg = self.config
        # Validate the entire proposed update before mutating any parameter.
        if set(gradients) != set(self.params):
            raise ValueError("gradients must cover every model parameter exactly")
        for name, p in self.params.items():
            grad = gradients[name]
            if grad.shape != p.shape or grad.device != p.device:
                raise ValueError(f"gradient shape/device mismatch in {name}")
        if not all(bool(torch.isfinite(g).all()) for g in gradients.values()):
            raise FloatingPointError("non-finite gradient; no parameters updated")
        for block, gate in gates.items():
            if not math.isfinite(gate) or not cfg.g_min <= gate <= 1:
                raise ValueError(f"gate for {block} must be in [g_min, 1]")
        delta_sq = torch.zeros((), device=next(iter(self.params.values())).device)
        feature_data_sq = torch.zeros_like(delta_sq)
        feature_gain_sum = torch.zeros_like(delta_sq)
        feature_parameters = 0
        for name, p in self.params.items():
            grad = gradients[name]
            s = self.state[name]
            block = self.block_for[name]
            s["momentum"].mul_(cfg.momentum).add_(grad)
            if block == "head":
                multiplier = torch.ones_like(s["c"])
                gate = 1.0
            else:
                gate = float(gates.get(block, 1.0))
                c_eff = self._effective_c(name, set(reopened))
                # This floor is on the final data-update multiplier, not just g.
                multiplier = cfg.g_min + (gate-cfg.g_min) * (1-c_eff).pow(cfg.gamma)
            scale = self._newborn_scale(name, p, row_view(multiplier, p), gate)
            data_delta = -cfg.lr * scale * s["momentum"]
            if block != "head":
                feature_parameters += p.numel()
                # scale broadcasts over remaining kernel/input dimensions.
                feature_gain_sum += scale.sum() * (p.numel() // scale.numel())
                feature_data_sq += data_delta.square().sum()
            if self.consolidation and block != "head":
                s["path"].add_(-grad * data_delta)
            decay_delta = -cfg.lr * scale * cfg.weight_decay * p
            anchor_delta = -cfg.lr * cfg.anchor_strength * row_view(s["c"], p) * (p-s["anchor"])
            delta = data_delta + decay_delta + anchor_delta
            p.add_(delta)
            delta_sq += delta.square().sum()
            s["age"].add_(1)
        self.steps += 1
        matured, consolidated = self._advance_newborns()
        return {"update_norm": math.sqrt(float(delta_sq)),
                "effective_feature_gain": float(feature_gain_sum) / max(1, feature_parameters),
                "feature_data_displacement": math.sqrt(float(feature_data_sq)),
                "feature_parameter_count": feature_parameters,
                "matured_units": matured, "locally_consolidated_units": consolidated,
                "total_maturations": self.total_maturations,
                "total_local_consolidations": self.total_local_consolidations,
                **self._newborn_counts()}

    @torch.no_grad()
    def consolidate(self, blocks: tuple[str, ...] | list[str]) -> None:
        if not self.consolidation:
            return
        for name, p in self.params.items():
            if self.block_for[name] not in blocks or self.block_for[name] == "head":
                continue
            # Controller events cannot truncate a newborn's accumulation window
            # or anchor it before its local developmental/maturity requirement.
            rows = torch.where(~self._pending_rows(name))[0]
            self._consolidate_rows(name, rows)

    @torch.no_grad()
    def update_utilities(self, features: dict[str, torch.Tensor]) -> None:
        """Contribution utility; deliberately named CBP-inspired, not exact CBP."""
        for unit in self.units:
            activation = features[unit.name].detach()
            if activation.ndim > 2:
                activation = activation.mean(tuple(range(2, activation.ndim)))
            state = self.unit_state[unit.name]
            outgoing = unit.outgoing.weight.detach().abs()
            contribution = outgoing.sum(tuple(i for i in range(outgoing.ndim) if i != 1))
            utility = activation.abs().mean(0) * contribution
            state["utility"].lerp_(utility, 1-self.config.utility_decay)
            state["mean"].lerp_(activation.mean(0), 1-self.config.utility_decay)
            state["age"].add_(1)

    def _param_name(self, parameter: nn.Parameter) -> str:
        return next(name for name, p in self.params.items() if p is parameter)

    @torch.no_grad()
    def recycle(self, *, adult: bool = False,
                forced_counts: dict[str, int] | None = None) -> dict:
        """Reset only registered feedforward units, and every affected state slice.

        Zeroing outgoing weights removes their previous contribution; it is NOT
        exactly function preserving. Measured replay damage includes this reset.
        A consolidated downstream row vetoes upstream recycling because its
        incident weight column would otherwise be changed behind the protection.

        ``forced_counts`` replays an explicit per-population replacement count,
        overriding interval and fraction, but not maturity or protection. An
        explicit empty mapping means no replacements. All requested counts are
        checked before any weights, credits, or RNG state change; inability to
        fulfill even one requested count raises ValueError. Forced checks leave
        fractional credits unchanged. Unit selection still follows this engine's
        utility/random policy; the matched control matches counts, not identities.
        """
        counts = {}
        forced = forced_counts is not None
        if forced:
            if not isinstance(forced_counts, dict):
                raise ValueError("forced_counts must be a mapping of population names to counts")
            for population, count in forced_counts.items():
                if population not in self.unit_state:
                    raise ValueError(f"unknown forced recycling population {population!r}")
                if type(count) is not int or count < 0:
                    raise ValueError(f"forced count for {population} must be a nonnegative integer")
            if not any(forced_counts.values()):
                return {"recycled": 0, "recycled_by_population": counts}
            if not self.recycling:
                raise ValueError("cannot fulfill forced recycling counts: recycling is disabled")
        elif not self.recycling or self.steps % self.config.recycle_interval:
            return {"recycled": 0, "recycled_by_population": counts}
        cfg = self.config
        fraction = cfg.adult_recycle_fraction if adult else cfg.recycle_fraction
        eligible_by_population = {}
        requested = {}
        credits = {}
        # Build and validate the entire request before selection draws or resets.
        for unit in reversed(self.units):
            us = self.unit_state[unit.name]
            incoming_name, bias_name, outgoing_name = self._unit_params[unit.name]
            protection = self.state[incoming_name]["c"].clone()
            if bias_name is not None:
                protection = torch.maximum(protection, self.state[bias_name]["c"])
            veto = bool((self.state[outgoing_name]["c"] >= cfg.recycle_cutoff).any())
            mask = (us["age"] >= cfg.maturity_steps) & (protection < cfg.recycle_cutoff)
            if cfg.newborn_steps:
                mask &= us["newborn_age"] < 0
            if veto:
                mask.zero_()
            eligible = torch.where(mask)[0]
            eligible_by_population[unit.name] = eligible
            if forced:
                count = forced_counts.get(unit.name, 0)
                if count > len(eligible):
                    cause = "downstream protection veto" if veto else "maturity, protection, or newborn eligibility"
                    raise ValueError(
                        f"cannot fulfill forced recycling count for {unit.name}: requested {count}, "
                        f"eligible {len(eligible)} ({cause})")
            elif veto:
                continue
            elif not len(eligible):
                credits[unit.name] = 0.0
                continue
            else:
                credit = us["credit"] + fraction * len(eligible)
                count = min(math.floor(credit + 1e-9), len(eligible))
                credits[unit.name] = credit - count
            if count:
                requested[unit.name] = count

        # Prepare all draws against a private copy so a failed preparation also
        # cannot advance the engine's RNG. Traversal/draw order matches v0.
        plan = []
        generator = torch.Generator(device="cpu")
        generator.set_state(self.generator.get_state())
        for unit in reversed(self.units):
            count = requested.get(unit.name, 0)
            if not count:
                continue
            eligible = eligible_by_population[unit.name]
            us = self.unit_state[unit.name]
            if self.random_recycling:
                order = torch.randperm(len(eligible), generator=generator).to(eligible.device)
            else:
                correction = (1-cfg.utility_decay**us["age"][eligible].float()).clamp_min(1e-8)
                corrected = us["utility"][eligible] / correction
                order = torch.argsort(corrected, stable=True)
            indices = eligible[order[:count]]
            weight = unit.incoming.weight
            bound = math.sqrt(6.0 / weight[0].numel())
            values = torch.empty((count, *weight.shape[1:]), dtype=weight.dtype, device="cpu")
            values.uniform_(-bound, bound, generator=generator)
            plan.append((unit, indices, values.to(weight.device)))

        for population, credit in credits.items():
            self.unit_state[population]["credit"] = credit
        # Registered populations are in forward order. Reset downstream first,
        # so a later incoming-row reinitialization cannot revive an upstream
        # population's outgoing columns that were just zeroed.
        for unit, indices, values in plan:
            us = self.unit_state[unit.name]
            _, _, outgoing_name = self._unit_params[unit.name]
            weight = unit.incoming.weight
            weight[indices] = values
            if unit.incoming.bias is not None:
                unit.incoming.bias[indices] = 0
            unit.outgoing.weight[:, indices] = 0
            incoming_params = [unit.incoming.weight]
            if unit.incoming.bias is not None:
                incoming_params.append(unit.incoming.bias)
            for p in incoming_params:
                s = self.state[self._param_name(p)]
                for key in ("momentum", "path", "c", "importance", "age"):
                    s[key][indices] = 0
                for key in ("anchor", "start"):
                    s[key][indices] = p[indices]
            out_state = self.state[outgoing_name]
            for key in ("momentum", "path", "anchor", "start"):
                out_state[key][:, indices] = 0
            # Remaining row importance is retained; only the reset column's path
            # and references are discarded. Downstream c is below cutoff here.
            for key in ("utility", "mean", "age"):
                us[key][indices] = 0
            if cfg.newborn_steps:
                us["newborn_age"][indices] = 0
            counts[unit.name] = len(indices)
        self.generator.set_state(generator.get_state())
        total = sum(counts.values())
        self.total_recycled += total
        return {"recycled": total, "recycled_by_population": counts}

    def diagnostics(self) -> dict:
        states = [s for name, s in self.state.items() if self.block_for[name] != "head"]
        c = torch.cat([s["c"] for s in states])
        return {"mean_consolidation": float(c.mean()),
                "consolidated_fraction": float((c > 0.9).float().mean()),
                "total_recycled": self.total_recycled,
                "total_maturations": self.total_maturations,
                "total_local_consolidations": self.total_local_consolidations,
                **self._newborn_counts(),
                "recyclable_units": sum(len(s["age"]) for s in self.unit_state.values()),
                "optimizer_state_bytes": sum(t.numel()*t.element_size() for s in self.state.values()
                                             for t in s.values()),
                "unit_state_bytes": sum(t.numel()*t.element_size() for s in self.unit_state.values()
                                        for t in s.values() if isinstance(t, torch.Tensor))}

    def state_dict(self) -> dict:
        return {"state": self.state, "units": self.unit_state, "steps": self.steps,
                "total_recycled": self.total_recycled,
                "total_maturations": self.total_maturations,
                "total_local_consolidations": self.total_local_consolidations,
                "newborn_steps": self.config.newborn_steps,
                "rng": self.generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        if state.get("newborn_steps", 0) != self.config.newborn_steps:
            raise ValueError("checkpoint newborn_steps does not match engine config")
        device = next(self.model.parameters()).device
        self.state = {name: {k: v.to(device) for k, v in s.items()}
                      for name, s in state["state"].items()}
        self.unit_state = {name: {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in s.items()} for name, s in state["units"].items()}
        self.steps, self.total_recycled = state["steps"], state["total_recycled"]
        self.total_maturations = state.get("total_maturations", 0)
        self.total_local_consolidations = state.get("total_local_consolidations", 0)
        # v0 checkpoints have no newborn tags and are valid when disabled.
        for unit in self.unit_state.values():
            unit.setdefault("newborn_age", torch.full_like(unit["age"], -1))
        self.generator.set_state(state["rng"].cpu())
