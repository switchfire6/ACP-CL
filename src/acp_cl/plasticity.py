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

    def __post_init__(self):
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
        self.unit_state = {
            u.name: {"utility": torch.zeros(u.incoming.weight.shape[0], device=u.incoming.weight.device),
                     "mean": torch.zeros(u.incoming.weight.shape[0], device=u.incoming.weight.device),
                     "age": torch.zeros(u.incoming.weight.shape[0], dtype=torch.long,
                                        device=u.incoming.weight.device), "credit": 0.0}
            for u in self.units
        }
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.steps = 0
        self.total_recycled = 0

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

    @torch.no_grad()
    def step(self, gradients: dict[str, torch.Tensor], gates: dict[str, float],
             reopened: tuple[str, ...] = ()) -> dict[str, float]:
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
        for name, p in self.params.items():
            grad = gradients[name]
            s = self.state[name]
            block = self.block_for[name]
            s["momentum"].mul_(cfg.momentum).add_(grad)
            if block == "head":
                multiplier = torch.ones_like(s["c"])
            else:
                gate = float(gates.get(block, 1.0))
                c_eff = self._effective_c(name, set(reopened))
                # This floor is on the final data-update multiplier, not just g.
                multiplier = cfg.g_min + (gate-cfg.g_min) * (1-c_eff).pow(cfg.gamma)
            scale = row_view(multiplier, p)
            data_delta = -cfg.lr * scale * s["momentum"]
            if self.consolidation and block != "head":
                s["path"].add_(-grad * data_delta)
            decay_delta = -cfg.lr * scale * cfg.weight_decay * p
            anchor_delta = -cfg.lr * cfg.anchor_strength * row_view(s["c"], p) * (p-s["anchor"])
            delta = data_delta + decay_delta + anchor_delta
            p.add_(delta)
            delta_sq += delta.square().sum()
            s["age"].add_(1)
        self.steps += 1
        return {"update_norm": math.sqrt(float(delta_sq))}

    @torch.no_grad()
    def consolidate(self, blocks: tuple[str, ...] | list[str]) -> None:
        if not self.consolidation:
            return
        cfg = self.config
        for name, p in self.params.items():
            if self.block_for[name] not in blocks or self.block_for[name] == "head":
                continue
            s = self.state[name]
            utility = row_sum(s["path"]).clamp_min(0) / (
                row_sum((p-s["start"]).square()) + cfg.importance_epsilon)
            normalized = robust_unit_interval(utility)
            normalized *= (s["age"] >= cfg.maturity_steps)
            s["importance"].copy_(utility)
            s["c"].copy_(1 - (1-s["c"]) * torch.exp(-cfg.consolidation_rate * normalized))
            # Only matured rows can acquire/update an anchor.
            rate = row_view((s["age"] >= cfg.maturity_steps).to(p.dtype) * cfg.anchor_rate, p)
            s["anchor"].add_(rate * (p-s["anchor"]))
            s["start"].copy_(p)
            s["path"].zero_()

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
    def recycle(self, *, adult: bool = False) -> dict:
        """Reset only registered feedforward units, and every affected state slice.

        Zeroing outgoing weights removes their previous contribution; it is NOT
        exactly function preserving. Measured replay damage includes this reset.
        A consolidated downstream row vetoes upstream recycling because its
        incident weight column would otherwise be changed behind the protection.
        """
        counts = {}
        if not self.recycling or self.steps % self.config.recycle_interval:
            return {"recycled": 0, "recycled_by_population": counts}
        cfg = self.config
        fraction = cfg.adult_recycle_fraction if adult else cfg.recycle_fraction
        # Registered populations are in forward order. Reset downstream first,
        # so a later incoming-row reinitialization cannot revive an upstream
        # population's outgoing columns that were just zeroed.
        for unit in reversed(self.units):
            us = self.unit_state[unit.name]
            incoming_name = self._param_name(unit.incoming.weight)
            outgoing_name = self._param_name(unit.outgoing.weight)
            protection = self.state[incoming_name]["c"].clone()
            if unit.incoming.bias is not None:
                bias_state = self.state[self._param_name(unit.incoming.bias)]
                protection = torch.maximum(protection, bias_state["c"])
            outgoing_c = self.state[outgoing_name]["c"]
            if bool((outgoing_c >= cfg.recycle_cutoff).any()):
                continue
            eligible = torch.where((us["age"] >= cfg.maturity_steps) &
                                   (protection < cfg.recycle_cutoff))[0]
            if not len(eligible):
                us["credit"] = 0.0
                continue
            us["credit"] += fraction * len(eligible)
            count = min(math.floor(us["credit"] + 1e-9), len(eligible))
            if count == 0:
                continue
            us["credit"] -= count
            if self.random_recycling:
                order = torch.randperm(len(eligible), generator=self.generator).to(eligible.device)
            else:
                # Bias-correct the age-dependent EMA for fair newborn comparisons.
                correction = (1-cfg.utility_decay**us["age"][eligible].float()).clamp_min(1e-8)
                corrected = us["utility"][eligible] / correction
                order = torch.argsort(corrected, stable=True)
            indices = eligible[order[:count]]
            weight = unit.incoming.weight
            fan_in = weight[0].numel()
            bound = math.sqrt(6.0 / fan_in)
            values = torch.empty((count, *weight.shape[1:]), dtype=weight.dtype, device="cpu")
            values.uniform_(-bound, bound, generator=self.generator)
            weight[indices] = values.to(weight.device)
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
            counts[unit.name] = count
        total = sum(counts.values())
        self.total_recycled += total
        return {"recycled": total, "recycled_by_population": counts}

    def diagnostics(self) -> dict:
        states = [s for name, s in self.state.items() if self.block_for[name] != "head"]
        c = torch.cat([s["c"] for s in states])
        return {"mean_consolidation": float(c.mean()),
                "consolidated_fraction": float((c > 0.9).float().mean()),
                "total_recycled": self.total_recycled,
                "recyclable_units": sum(len(s["age"]) for s in self.unit_state.values()),
                "optimizer_state_bytes": sum(t.numel()*t.element_size() for s in self.state.values()
                                             for t in s.values())}

    def state_dict(self) -> dict:
        return {"state": self.state, "units": self.unit_state, "steps": self.steps,
                "total_recycled": self.total_recycled, "rng": self.generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        device = next(self.model.parameters()).device
        self.state = {name: {k: v.to(device) for k, v in s.items()}
                      for name, s in state["state"].items()}
        self.unit_state = {name: {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in s.items()} for name, s in state["units"].items()}
        self.steps, self.total_recycled = state["steps"], state["total_recycled"]
        self.generator.set_state(state["rng"].cpu())
