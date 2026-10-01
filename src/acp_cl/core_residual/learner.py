"""A learned core plus a zero-output, raw-input residual representation pathway.

The historical uniform replay/update loop is reused unchanged. Initial state
snapshots below serve measurement only; no stored model supplies predictions,
regularization targets, replay priorities or boundary-triggered resets.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from acp_cl.conditional.learner import ConditionalLearner, Hypotheses, PacketMemory
from acp_cl.contextual.learner import ContextLearner, HistoryNetwork
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.world import IMAGE_SHAPE
from acp_cl.predictive_value.mechanism import _value_hash, state_signature
from acp_cl.replay_renewal.memory import ReservoirMemory
from acp_cl.replay_renewal.study import new_learner


ARMS = ("joint", "separate", "fixed_features")
RESIDUAL_SEED_OFFSET = 167921
ENCODER_PREFIXES = ("frame.", "temporal.")
WORK_FIELDS = ("packets", "optimizer_steps", "backward_calls", "logical_probability_calls",
    "core_forward_calls", "residual_forward_calls", "core_query_presentations",
    "residual_query_presentations", "core_support_presentations", "residual_support_presentations",
    "core_backward_paths", "residual_backward_paths")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _snapshot(module):
    return {name:value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _encoder(state):
    return {name:value for name, value in state.items() if name.startswith(ENCODER_PREFIXES)}


def _bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_bytes(item) for item in value)
    return 0


def _displacement(state, reference):
    squared = sum(float((state[name].detach().cpu().double()-value.double()).square().sum())
                  for name, value in reference.items())
    result = math.sqrt(squared)
    _require(math.isfinite(result), "nonfinite representation displacement")
    return result


def _optimizer_part(optimizer, module):
    """Named optimizer state and group settings, limited to this owned module."""
    names = {id(parameter):name for name, parameter in module.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        chosen = [names[id(parameter)] for parameter in group["params"] if id(parameter) in names]
        if chosen:
            groups.append(dict(parameters=chosen,
                settings={key:value for key, value in group.items() if key not in ("params", "param_names")}))
    states = {names[id(parameter)]:state for parameter, state in optimizer.state.items() if id(parameter) in names}
    return dict(groups=groups, state=states)


class CoreResidualModel(nn.Module):
    """Both raw-sensor pathways execute; frozen modules stay in evaluation mode."""

    def __init__(self, core, residual, arm):
        super().__init__()
        self.core, self.residual, self.arm = core, residual, arm

    def forward(self, observations):
        return self.core(observations)+self.residual(observations)

    def train(self, mode=True):
        super().train(mode)
        if self.arm != "joint":
            self.core.eval()
        if self.arm == "fixed_features":
            self.residual.frame.eval()
            self.residual.temporal.eval()
        return self


def _recurrent_logits(path, combined, support_size, query_size, actions, outcomes, has_support):
    """The unchanged recurrent feature/context/decoder computation for one path."""
    features = path.encode(combined)
    events = torch.cat((features[:support_size], F.one_hot(actions, 5).float(), outcomes), dim=-1)
    if path.interaction_features:
        interactions = (features[:support_size, :, None]*F.one_hot(actions, 5)[:, None, :]
                        *(2*outcomes[:, -1]-1)[:, None, None]).flatten(start_dim=1)
        events = torch.cat((events, interactions), dim=-1)
        context = path.context(events[None])[0].mean(dim=1)
    else:
        context = path.context(events[None])[1][-1]
    if not has_support:
        context = context*0
    return path.decoder(torch.cat((features[support_size:], context.expand(query_size, -1)), dim=-1)).reshape(-1, 5, 3)


class CoreResidualLearner(ContextLearner):
    """Ordinary learner interface; construction is through an explicit core fork."""

    def __init__(self, *args, **kwargs):
        raise TypeError("construct with fork_core or new_from_scratch")

    def probabilities(self, observations, support, oracle_modes=None, override=None):
        if oracle_modes is not None or override == "oracle":
            raise ValueError("core/residual learners must not receive latent modes")
        if override not in (None, "shuffled", "pooled", "conditional"):
            raise ValueError("unsupported causal prediction override")
        if self.method == "conditional":
            # Composite.forward adds corresponding support/query expert logits
            # before the original evidence calculation and expert mixture.
            return ConditionalLearner.probabilities(self, observations, support, None, override)
        query = torch.as_tensor(observations, device=self.device)
        if support is None:
            support_x = torch.zeros((self.settings["batch_size"], *IMAGE_SHAPE), dtype=torch.uint8, device=self.device)
            support_a = torch.zeros(len(support_x), dtype=torch.long, device=self.device)
            support_y = torch.zeros((len(support_x), 3), device=self.device)
        else:
            support_x, support_a, support_y = self.tensors(support)
        if override == "shuffled":
            support_x = support_x.roll(1, dims=0)
        combined = torch.cat((support_x, query))
        core = _recurrent_logits(self.model.core, combined, len(support_x), len(query), support_a, support_y, support is not None)
        residual = _recurrent_logits(self.model.residual, combined, len(support_x), len(query), support_a, support_y, support is not None)
        logits = core+residual
        return logits.sigmoid(), torch.ones((len(query), 1), device=self.device), logits[:, None]

    @torch.no_grad()
    def predict(self, observations, support=None, oracle_modes=None, override=None):
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            return ConditionalLearner.predict(self, observations, support, oracle_modes, override)
        finally:
            for module, training in modes:
                module.training = training

    def train(self, current, oracle_modes=None):
        if oracle_modes is not None:
            raise ValueError("core/residual learners must not receive latent modes")
        before = dict(self.cost)
        ConditionalLearner.train(self, current, oracle_modes)
        updates = self.cost["optimizer_steps"]-before["optimizer_steps"]
        queries = self.cost["query_presentations"]-before["query_presentations"]
        supports = self.cost["support_presentations"]-before["support_presentations"]
        self.path_work["packets"] += 1
        self.path_work["optimizer_steps"] += updates
        self.path_work["backward_calls"] += updates
        self.path_work["logical_probability_calls"] += 2*updates
        for path in ("core", "residual"):
            self.path_work[path+"_forward_calls"] += 2*updates
            self.path_work[path+"_query_presentations"] += queries
            self.path_work[path+"_support_presentations"] += supports
        self.path_work["core_backward_paths"] += 2*updates*(self.arm == "joint")
        self.path_work["residual_backward_paths"] += 2*updates

    def freeze_features(self):
        raise ValueError("trainability is fixed by the declared core/residual arm")

    def diagnostics(self):
        report = ConditionalLearner.diagnostics(self)
        core, residual = self.model.core.state_dict(), self.model.residual.state_dict()
        initial_core_encoder = {name.removeprefix("core."):value for name, value in self.initial_encoder.items()
                                if name.startswith("core.")}
        encoder_state = {name:value for name, value in self.model.state_dict().items()
                         if name.startswith(("core.frame.", "core.temporal.", "residual.frame.", "residual.temporal."))}
        report.update(initial_encoder_hash=self.initial_encoder_hash, encoder_hash=state_hash(encoder_state))
        counts = {}
        for label, module in (("total", self.model), ("core", self.model.core), ("residual", self.model.residual)):
            counts[label] = sum(parameter.numel() for parameter in module.parameters())
            counts[label+"_trainable"] = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            if label != "total":
                counts[label+"_encoder"] = sum(parameter.numel() for name, parameter in module.named_parameters()
                                               if name.startswith(ENCODER_PREFIXES))
                counts[label+"_encoder_trainable"] = sum(parameter.numel() for name, parameter in module.named_parameters()
                    if parameter.requires_grad and name.startswith(ENCODER_PREFIXES))
        core_optimizer = _optimizer_part(self.optimizer, self.model.core)
        residual_optimizer = _optimizer_part(self.optimizer, self.model.residual)
        report["core_residual"] = dict(arm=self.arm, residual_seed=self.residual_seed, parameters=counts,
            core_prefix_initial_hash=self.core_prefix_initial_hash,
            core_at_fork_hash=state_hash(self.core_at_fork), residual_initial_hash=state_hash(self.residual_at_fork),
            core_hash=state_hash(core), residual_hash=state_hash(residual),
            core_encoder_hash=state_hash(_encoder(core)), residual_encoder_hash=state_hash(_encoder(residual)),
            residual_initial_encoder_hash=state_hash(_encoder(self.residual_at_fork)),
            core_optimizer_at_fork_hash=self.core_optimizer_at_fork_hash,
            core_optimizer_hash=_value_hash(core_optimizer), residual_optimizer_hash=_value_hash(residual_optimizer),
            core_displacement_from_fork=_displacement(core, self.core_at_fork),
            core_encoder_displacement_from_fork=_displacement(core, _encoder(self.core_at_fork)),
            core_encoder_displacement_from_initial=_displacement(core, initial_core_encoder),
            residual_displacement_from_initial=_displacement(residual, self.residual_at_fork),
            residual_encoder_displacement_from_initial=_displacement(residual, _encoder(self.residual_at_fork)),
            core_optimizer_bytes=_bytes(core_optimizer["state"]), residual_optimizer_bytes=_bytes(residual_optimizer["state"]),
            measurement_snapshot_bytes=_bytes(self.core_at_fork)+_bytes(self.residual_at_fork)+_bytes(self.initial_encoder),
            work=dict(self.path_work), fixed_feature_description=(
                "Fixed random residual frame/temporal features; residual heads/context/readout remain trainable."
                if self.arm == "fixed_features" else None),
            measurement_scope="Initial tensor snapshots measure displacement only; they never provide prediction or training targets.",
            work_scope="Both raw-input paths execute each logical forward. Trainability changes backward work; this is not FLOP matching.")
        return report


def _validate_core(base, settings):
    _require(type(base) is ContextLearner and base.method in ("conditional", "recurrent"),
             "an ordinary conditional/recurrent ContextLearner core is required")
    _require(type(base.optimizer) is torch.optim.Adam and len(base.optimizer.param_groups) == 1,
             "the ordinary single-group Adam optimizer is required")
    _require(type(base.memory) in (PacketMemory, ReservoirMemory), "uniform causal packet memory is required")
    _require(all(packet.oracle_modes is None for packet in base.memory.packets), "core memory contains privileged labels")
    _require(all(key in settings and settings[key] == value for key, value in base.settings.items()),
             "fork settings must preserve all core settings")
    for name in ("residual_width", "residual_context_width", "residual_decoder_width"):
        _require(type(settings.get(name)) is int and settings[name] > 0, "positive residual dimensions are required")
    expected_type = Hypotheses if base.method == "conditional" else HistoryNetwork
    _require(type(base.model) is expected_type, "core architecture differs from the ordinary reference")
    parameters = list(base.model.parameters())
    owned = base.optimizer.param_groups[0]["params"]
    _require(len(parameters) == len(owned) and all(left is right for left, right in zip(parameters, owned)),
             "core Adam ownership/order differs from named model parameters")
    _require(all(parameter.requires_grad for parameter in parameters), "ordinary core must begin fully trainable")
    _require(all(any(parameter is candidate for candidate in parameters) for parameter in base.optimizer.state),
             "core Adam contains foreign parameter state")
    _require(all(torch.isfinite(value).all().item() for value in base.model.state_dict().values()), "nonfinite core weights")


def fork_core(base, arm, seed, settings):
    """Keep causal replay/history/cost and complete core Adam state; add zero output."""
    _require(arm in ARMS, "unknown core/residual arm")
    _require(type(seed) is int and seed >= 0, "nonnegative integer initialization seed required")
    _validate_core(base, settings)
    before = state_signature(base)
    copied = copy.deepcopy(base)
    for source, target in zip(base.model.parameters(), copied.model.parameters()):
        target.grad = None if source.grad is None else source.grad.detach().clone()
    result = object.__new__(CoreResidualLearner)
    result.__dict__.update(copied.__dict__)
    result.settings = copy.deepcopy(settings)
    result.arm, result.residual_seed = arm, seed+RESIDUAL_SEED_OFFSET
    result.core_prefix_initial_hash = base.initial_hash
    result.core_at_fork = _snapshot(result.model)
    result.core_optimizer_at_fork_hash = _value_hash(_optimizer_part(result.optimizer, result.model))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(result.residual_seed)
        if base.method == "conditional":
            residual = Hypotheses(settings["residual_width"], settings["experts"])
            final_layers = [head[-1] for head in residual.heads]
        else:
            residual = HistoryNetwork(settings["residual_width"], settings["residual_context_width"],
                settings["residual_decoder_width"], settings["interaction_features"])
            final_layers = [residual.decoder[-1]]
        for layer in final_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        residual = residual.to(result.device)
    result.residual_at_fork = _snapshot(residual)
    result.initial_encoder = {"core."+name:value.detach().cpu().clone() for name, value in base.initial_encoder.items()}
    result.initial_encoder.update({"residual."+name:value.clone() for name, value in _encoder(result.residual_at_fork).items()})
    result.initial_encoder_hash = state_hash(result.initial_encoder)
    core = result.model
    core_modes = {name:module.training for name, module in core.named_modules()}
    result.model = CoreResidualModel(core, residual, arm)
    result.model.train(core.training)
    if arm == "joint":
        for name, module in core.named_modules():
            module.training = core_modes[name]
    if arm != "joint":
        core.requires_grad_(False)
    if arm == "fixed_features":
        residual.frame.requires_grad_(False)
        residual.temporal.requires_grad_(False)
    # deepcopy(base) retained the optimizer's ownership of the copied core.
    # Append residual parameters to the very same group without resetting any
    # core moments, per-parameter steps, hyperparameters or optimizer defaults.
    group = result.optimizer.param_groups[0]
    group["params"].extend(residual.parameters())
    if "param_names" in group:
        group["param_names"] = [name for name, _ in result.model.named_parameters()]
    result.initial_hash = state_hash(result.model.state_dict())
    result.path_work = {name:0 for name in WORK_FIELDS}
    _require(_value_hash(_optimizer_part(result.optimizer, core)) == result.core_optimizer_at_fork_hash,
             "core optimizer changed during residual insertion")
    _require(all(parameter not in result.optimizer.state for parameter in residual.parameters()),
             "residual optimizer state must begin empty")
    _require(state_signature(base) == before, "residual insertion mutated the ordinary core donor")
    return result


def new_from_scratch(model, seed, settings, device="cpu"):
    """Qualification-only fresh joint model; runner may supply past-only history."""
    base = new_learner(model, seed, settings, device)
    return fork_core(base, "joint", seed, settings)


def learner_signature(learner, ignore_walltime=False):
    """Include new mechanism state; keep the old signature for ordinary cores."""
    ordinary = state_signature(learner, ignore_walltime=ignore_walltime)
    if not isinstance(learner, CoreResidualLearner):
        return ordinary
    return _value_hash(dict(ordinary=ordinary, arm=learner.arm, model_arm=learner.model.arm,
        residual_seed=learner.residual_seed, core_prefix_initial_hash=learner.core_prefix_initial_hash,
        core_at_fork=learner.core_at_fork, residual_at_fork=learner.residual_at_fork,
        core_optimizer_at_fork_hash=learner.core_optimizer_at_fork_hash, path_work=learner.path_work))
