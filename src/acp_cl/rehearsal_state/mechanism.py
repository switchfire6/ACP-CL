"""Isolated weight, Adam-state and causal-anchor transplantation.

Hybrids are controlled interventions, not necessarily states reached by an
ordinary learning trajectory. Adam moments, step counters and group settings
move together. Inactive learner fields always come from the old snapshot.
"""

from __future__ import annotations

import copy
import math
import time

import torch

from acp_cl.contextual.learner import ContextLearner
from acp_cl.predictive_value.mechanism import (
    _ordinary, _packet, _packet_loss, _value_hash, clone_exact, make_shadow,
    packet_fingerprint, state_signature,
)
from acp_cl.replay_renewal.memory import memory_state


def _finite(value, label):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite {label} tensor")
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item, label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite {label} scalar")


def _active(learner):
    _ordinary(learner)
    return dict(model=learner.model.state_dict(), optimizer=learner.optimizer.state_dict(),
        gradients={name: parameter.grad for name, parameter in learner.model.named_parameters()},
        trainable={name: parameter.requires_grad for name, parameter in learner.model.named_parameters()},
        module_modes={name: module.training for name, module in learner.model.named_modules()})


def active_signature(learner):
    """Hash only model, full optimizer, gradients, trainability and module modes."""
    state = _active(learner)
    _finite(state, "active state")
    return _value_hash(state)


def _inactive_signature(learner):
    return _value_hash(dict(method=learner.method, device=str(learner.device), settings=learner.settings,
        memory=memory_state(learner.memory), history=None if learner.history is None else learner.history.fingerprint(),
        cost=learner.cost, peak_memory_bytes=learner.peak_memory_bytes,
        initial_hash=learner.initial_hash, initial_encoder=learner.initial_encoder,
        initial_encoder_hash=learner.initial_encoder_hash))


def _parameter_groups(learner):
    """Semantic parameter names, because Adam loading remaps by group position."""
    names = {id(parameter): name for name, parameter in learner.model.named_parameters()}
    groups, all_ids = [], []
    for group in learner.optimizer.param_groups:
        ids = [id(parameter) for parameter in group["params"]]
        if any(identifier not in names for identifier in ids):
            raise ValueError("optimizer does not own model parameters")
        groups.append([names[identifier] for identifier in ids])
        all_ids.extend(ids)
        if "param_names" in group and list(group["param_names"]) != groups[-1]:
            raise ValueError("optimizer parameter names disagree with ownership")
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(names):
        raise ValueError("optimizer must own each model parameter exactly once")
    if any(id(parameter) not in names for parameter in learner.optimizer.state):
        raise ValueError("optimizer state contains a foreign parameter")
    for parameter, state in learner.optimizer.state.items():
        if not state:
            continue
        if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
            raise ValueError("incomplete Adam state")
        step = state["step"]
        if (not isinstance(step, torch.Tensor) or step.ndim != 0
                or not bool(torch.isfinite(step)) or float(step) < 0
                or not float(step).is_integer()):
            raise ValueError("invalid Adam step counter")
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in state:
                value = state[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                        or value.dtype != parameter.dtype or value.device != parameter.device):
                    raise ValueError("Adam moment shape, dtype or device differs from parameter")
    return groups


def _compatible(old, new):
    for donor in (old, new):
        _ordinary(donor)
        if type(donor) is not ContextLearner or type(donor.optimizer) is not torch.optim.Adam:
            raise ValueError("ordinary ContextLearner and Adam donors are required")
        _parameter_groups(donor)
        _finite(donor.initial_encoder, "initial encoder")
        active_signature(donor)
    if old.method != new.method or old.device != new.device or old.settings != new.settings:
        raise ValueError("donor methods, devices and settings must match")
    if (old.initial_hash != new.initial_hash or old.initial_encoder_hash != new.initial_encoder_hash
            or _value_hash(old.initial_encoder) != _value_hash(new.initial_encoder)):
        raise ValueError("donors must share the same initial model identity")
    def modules(learner):
        return [(name, type(module)) for name, module in learner.model.named_modules()]

    def parameters(learner):
        return [(name, tuple(parameter.shape), parameter.dtype, parameter.device)
                for name, parameter in learner.model.named_parameters()]

    def states(learner):
        return [(name, tuple(value.shape), value.dtype, value.device)
                for name, value in learner.model.state_dict().items()]
    if modules(old) != modules(new) or parameters(old) != parameters(new) or states(old) != states(new):
        raise ValueError("donor model names, modules, shapes, dtypes or devices differ")
    if _parameter_groups(old) != _parameter_groups(new):
        raise ValueError("optimizer parameter group names or order differ")


def transplant(old, new, weights_bit, optimizer_bit):
    """Copy active components into independent canonical-old inactive state.

    Weight-donor gradients, trainability and per-module flags are copied too.
    Optimizer loading receives a deepcopy: a raw state_dict can alias the
    donor's moments even when parameter ownership is remapped correctly.
    """
    if any(type(bit) is not int or bit not in (0, 1) for bit in (weights_bit, optimizer_bit)):
        raise ValueError("factor bits must be integer zero or one")
    _compatible(old, new)
    signatures = state_signature(old), state_signature(new)
    weights, optimizer = (old, new)[weights_bit], (old, new)[optimizer_bit]
    result = clone_exact(old)
    # assign=False retains destination Parameter objects owned by its Adam.
    result.model.load_state_dict(weights.model.state_dict(), strict=True, assign=False)
    source_parameters = dict(weights.model.named_parameters())
    for name, parameter in result.model.named_parameters():
        source = source_parameters[name]
        parameter.requires_grad_(source.requires_grad)
        parameter.grad = None if source.grad is None else source.grad.detach().clone()
    source_modules = dict(weights.model.named_modules())
    for name, module in result.model.named_modules():
        module.training = source_modules[name].training
    result.optimizer.load_state_dict(copy.deepcopy(optimizer.optimizer.state_dict()))
    if (_parameter_groups(result) != _parameter_groups(optimizer)
            or _value_hash(result.optimizer.state_dict()) != _value_hash(optimizer.optimizer.state_dict())):
        raise AssertionError("optimizer transplantation changed ownership or full state")
    active_signature(result)
    if (_inactive_signature(result) != _inactive_signature(old)
            or (state_signature(old), state_signature(new)) != signatures):
        raise AssertionError("transplantation changed donors or canonical inactive state")
    return result


def _loss_pair(learner, anchor, replay):
    """Two no-gradient diagnostic forwards with exact state/mode preservation."""
    before = state_signature(learner)
    modes = [(module, module.training) for module in learner.model.modules()]
    try:
        learner.model.eval()
        with torch.no_grad():
            values = [_packet_loss(learner, packet) for packet in (anchor, replay)]
            _finite(values, "diagnostic BCE")
            result = [float(value) for value in values]
    finally:
        for module, training in modes:
            module.training = training
    if state_signature(learner) != before:
        raise AssertionError("diagnostic loss forward changed learner state")
    return result


def _update_l2(initial, final):
    # Subtract after conversion, preserving the actual stored displacement.
    squared = 0.
    initial_parameters = dict(initial.model.named_parameters())
    for name, parameter in final.model.named_parameters():
        difference = parameter.detach().to(dtype=torch.float64)-initial_parameters[name].detach().to(dtype=torch.float64)
        squared += float(difference.square().sum())
    result = math.sqrt(squared)
    if not math.isfinite(result):
        raise ValueError("nonfinite update displacement")
    return result


def make_factor_shadow(old, new, anchor_old, anchor_new, replay, cell, updates):
    """Apply the unchanged matched rehearsal to one W/O/A state-factor cell.

    Bits are weights, complete Adam state, and explicit causal anchor, in that
    order. Inactive cost counters remain old; Adam's actual per-parameter steps
    follow the optimizer donor. Four diagnostic BCE forwards are counted apart
    from training. Returned losses make no finite-step improvement guarantee.
    """
    if not isinstance(cell, str) or len(cell) != 3 or any(bit not in "01" for bit in cell):
        raise ValueError("cell must be a three-character W/O/A bit string")
    if type(updates) is not int or updates < 1:
        raise ValueError("a positive integer rehearsal update count is required")
    bits = tuple(int(bit) for bit in cell)
    _compatible(old, new)
    for packet in (anchor_old, anchor_new, replay):
        _packet(packet, old.settings["batch_size"])
    donor_signatures = state_signature(old), state_signature(new)
    packet_signatures = tuple(packet_fingerprint(packet) for packet in (anchor_old, anchor_new, replay))
    started = time.perf_counter()
    initial = transplant(old, new, *bits[:2])
    transfer_seconds = time.perf_counter()-started
    initial_active = active_signature(initial)
    inactive = _inactive_signature(initial)
    anchor = (anchor_old, anchor_new)[bits[2]]
    started = time.perf_counter()
    before = _loss_pair(initial, anchor, replay)
    diagnostic_seconds = time.perf_counter()-started
    shadow, work = make_shadow(initial, anchor, replay, updates)
    final_active = active_signature(shadow)
    started = time.perf_counter()
    after = _loss_pair(shadow, anchor, replay)
    diagnostic_seconds += time.perf_counter()-started
    if (_inactive_signature(shadow) != inactive
            or (state_signature(old), state_signature(new)) != donor_signatures
            or tuple(packet_fingerprint(packet) for packet in (anchor_old, anchor_new, replay)) != packet_signatures):
        raise AssertionError("factor rehearsal changed donors, inactive state or supplied packets")
    work.update(cell=cell, weights_bit=bits[0], optimizer_bit=bits[1], anchor_bit=bits[2],
        old_donor_sha256=donor_signatures[0], new_donor_sha256=donor_signatures[1],
        weights_donor_sha256=donor_signatures[bits[0]], optimizer_donor_sha256=donor_signatures[bits[1]],
        inactive_donor_sha256=donor_signatures[0], canonical_inactive_sha256=inactive,
        initial_active_sha256=initial_active, final_active_sha256=final_active,
        transfer_seconds=transfer_seconds, update_l2=_update_l2(initial, shadow), update_norm_dtype="float64",
        diagnostic_forwards=4,
        diagnostic_query_presentations=2*(len(anchor.query)+len(replay.query)),
        diagnostic_support_presentations=2*sum(old.settings["batch_size"] if packet.support is None
                                               else len(packet.support) for packet in (anchor, replay)),
        diagnostic_seconds=diagnostic_seconds,
        losses=dict(anchor_before=before[0], replay_before=before[1], anchor_after=after[0], replay_after=after[1]),
        donors_unchanged=True, inactive_unchanged=True,
        factor_scope="Weight donor supplies gradients, trainability and module modes. Adam moments, steps and "
            "group settings move together. Inactive fields remain old. Hybrid states may be off trajectory; "
            "effects can interact. Diagnostic loss changes are measurements, not guarantees.")
    return shadow, work
