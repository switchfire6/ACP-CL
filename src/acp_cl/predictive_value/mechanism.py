"""Isolated causal-packet rehearsal and prequential performed-action scoring.

The caller chooses packets before observing the validation window. Rehearsal
changes only an isolated model/Adam copy. It does not select memory, advance
history, or provide a general estimate of an example's intrinsic value.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from acp_cl.conditional.learner import Packet
from acp_cl.contextual.learner import ContextLearner
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.world import Experience
from acp_cl.replay_renewal.memory import memory_state


def _value_hash(value):
    """Stable content hash, independent of pickle addresses or storage IDs."""
    result = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            result.update(b"tensor:")
            array = item.detach().cpu().contiguous().numpy()
            result.update(str(array.dtype).encode())
            result.update(json.dumps(array.shape).encode())
            result.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            result.update(b"array:")
            array = np.ascontiguousarray(item)
            result.update(str(array.dtype).encode())
            result.update(json.dumps(array.shape).encode())
            result.update(array.tobytes())
        elif isinstance(item, dict):
            result.update(b"dict:")
            for key in sorted(item, key=repr):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for part in item:
                update(part)
        else:
            result.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        result.update(b";")

    update(value)
    return result.hexdigest()


def _ordinary(learner):
    if not isinstance(learner, ContextLearner) or learner.method not in ("conditional", "recurrent"):
        raise ValueError("an ordinary conditional or recurrent ContextLearner is required")


def state_signature(learner, ignore_walltime=False):
    """Hash the full relevant learner state; optionally normalize cost wall time.

Global RNGs belong to the caller and are not embedded in a learner signature.
Replay membership and sampling RNGs are included through memory_state.
"""
    _ordinary(learner)
    cost = dict(learner.cost)
    if ignore_walltime:
        cost = {key: 0. if key.endswith("_seconds") else value for key, value in cost.items()}
    return _value_hash(dict(
        learner_type=f"{type(learner).__module__}.{type(learner).__qualname__}",
        method=learner.method, device=str(learner.device), settings=learner.settings,
        model=learner.model.state_dict(), optimizer=learner.optimizer.state_dict(),
        gradients={name: parameter.grad for name, parameter in learner.model.named_parameters()},
        trainable={name: parameter.requires_grad for name, parameter in learner.model.named_parameters()},
        module_modes={name: module.training for name, module in learner.model.named_modules()},
        memory=memory_state(learner.memory),
        history=None if learner.history is None else learner.history.fingerprint(),
        cost=cost, peak_memory_bytes=learner.peak_memory_bytes,
        initial_hash=learner.initial_hash, initial_encoder=learner.initial_encoder,
        initial_encoder_hash=getattr(learner, "initial_encoder_hash", None)))


def clone_exact(learner):
    """Deepcopy model, Adam, causal memory and history, including accumulated grad."""
    _ordinary(learner)
    result = copy.deepcopy(learner)
    # Parameter deepcopy omits .grad; retain it until the first ordinary zero_grad.
    for original, cloned in zip(learner.model.parameters(), result.model.parameters()):
        cloned.grad = None if original.grad is None else original.grad.detach().clone()
    return result


def _experience(data, name):
    if not isinstance(data, Experience) or not len(data):
        raise ValueError(f"{name} must be a nonempty ordinary Experience")
    if (data.actions.shape != (len(data),) or data.survival.shape != (len(data), 3)
            or not np.issubdtype(data.actions.dtype, np.integer)
            or np.any(data.actions < 0) or np.any(data.actions >= 5)):
        raise ValueError(f"{name} has invalid actions or outcome dimensions")
    if not np.isfinite(data.survival).all() or np.any((data.survival < 0) | (data.survival > 1)):
        raise ValueError(f"{name} has invalid reported outcomes")


def _packet(packet, batch_size):
    if not isinstance(packet, Packet) or packet.oracle_modes is not None:
        raise ValueError("ordinary causal packets without latent labels are required")
    _experience(packet.query, "packet query")
    if len(packet.query) != batch_size:
        raise ValueError("rehearsal query must match the original packet size")
    if packet.support is not None:
        _experience(packet.support, "packet support")
        if len(packet.support) != batch_size:
            raise ValueError("rehearsal support must match the original history size")


def packet_fingerprint(packet):
    if not isinstance(packet, Packet) or packet.oracle_modes is not None:
        raise ValueError("ordinary causal packets without latent labels are required")
    return _value_hash(dict(
        support=None if packet.support is None else packet.support.fingerprint(),
        query=packet.query.fingerprint()))


def _packet_loss(learner, packet):
    probabilities, _, _ = learner.probabilities(packet.query.observations, packet.support)
    _, actions, outcomes = learner.tensors(packet.query)
    selected = probabilities[torch.arange(len(actions), device=learner.device), actions]
    return F.binary_cross_entropy(selected.clamp(1e-6, 1-1e-6), outcomes)


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def make_shadow(parent, anchor, replay, updates):
    """Use a fixed anchor and replay packet for U matched clipped Adam updates.

No reservoir sampling/addition or history advancement occurs. The copy retains
the parent's cost counters; rehearsal work is returned separately. The caller
must not train this shadow while scoring its declared future windows.
"""
    _ordinary(parent)
    if type(updates) is not int or updates < 1:
        raise ValueError("a positive integer number of rehearsal updates is required")
    if not isinstance(parent.optimizer, torch.optim.Adam):
        raise ValueError("rehearsal requires the parent's Adam optimizer")
    if not all(parameter.requires_grad for parameter in parent.model.parameters()):
        raise ValueError("all model parameters must remain trainable during rehearsal")
    size = parent.settings["batch_size"]
    _packet(anchor, size)
    _packet(replay, size)
    original_signature = state_signature(parent)
    packet_hashes = packet_fingerprint(anchor), packet_fingerprint(replay)
    started = time.perf_counter()
    shadow = clone_exact(parent)
    initial_model = state_hash(shadow.model.state_dict())
    initial_optimizer = _value_hash(shadow.optimizer.state_dict())
    memory_before = _value_hash(memory_state(shadow.memory))
    history_before = None if shadow.history is None else shadow.history.fingerprint()
    for _ in range(updates):
        shadow.model.train()
        loss = torch.stack([_packet_loss(shadow, packet) for packet in (anchor, replay)]).mean()
        shadow.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(shadow.model.parameters(), 5.0)
        shadow.optimizer.step()
    elapsed = time.perf_counter()-started
    if (state_signature(parent) != original_signature
            or (packet_fingerprint(anchor), packet_fingerprint(replay)) != packet_hashes):
        raise AssertionError("rehearsal changed the parent or supplied causal packets")
    if (_value_hash(memory_state(shadow.memory)) != memory_before
            or (None if shadow.history is None else shadow.history.fingerprint()) != history_before
            or shadow.cost != parent.cost):
        raise AssertionError("rehearsal changed copied memory, history or original cost counters")
    work = dict(
        steps=updates, forwards=2*updates, backward_calls=updates,
        query_presentations=updates*(len(anchor.query)+len(replay.query)),
        support_presentations=updates*sum(size if p.support is None else len(p.support)
                                          for p in (anchor, replay)),
        training_seconds=elapsed, model_bytes=_tensor_bytes(shadow.model.state_dict()),
        optimizer_bytes=_tensor_bytes(shadow.optimizer.state_dict()),
        gradient_bytes=sum(_tensor_bytes(parameter.grad) for parameter in shadow.model.parameters()),
        initial_encoder_bytes=_tensor_bytes(shadow.initial_encoder),
        replay_bytes=shadow.memory.nbytes(),
        history_bytes=0 if shadow.history is None else Packet(None, shadow.history).nbytes(),
        anchor_packet_sha256=packet_hashes[0], replay_packet_sha256=packet_hashes[1],
        initial_model_sha256=initial_model, final_model_sha256=state_hash(shadow.model.state_dict()),
        initial_optimizer_sha256=initial_optimizer,
        final_optimizer_sha256=_value_hash(shadow.optimizer.state_dict()),
        parent_state_sha256=original_signature, shadow_state_sha256=state_signature(shadow),
        parent_unchanged=True, memory_history_unchanged=True,
        interpretation="Value of this fixed rehearsal relative to its declared replacement; future weights remain fixed.")
    work["explicit_tensor_bytes_subtotal"] = sum(work[name] for name in (
        "model_bytes", "optimizer_bytes", "gradient_bytes", "initial_encoder_bytes",
        "replay_bytes", "history_bytes"))
    work["explicit_tensor_bytes_scope"] = (
        "Sum of explicit model, optimizer, gradient, initial-encoder, replay and history storage. "
        "Replay/history aliases may be conservatively double counted. Excludes autograd graphs, "
        "operator workspaces, Python objects and allocator overhead; not measured peak RAM.")
    return shadow, work


def predict_all(learner, observations, support):
    """Predict all actions without query outcomes and restore every module mode.

    Phase-boundary signatures and tests check full state/RNG invariance without
    hashing every prediction. No law or current outcome reaches the predictor.
    """
    _ordinary(learner)
    if not isinstance(observations, np.ndarray) or not len(observations):
        raise ValueError("nonempty query observations are required")
    if support is not None:
        _experience(support, "scoring support")
    modes = [(module, module.training) for module in learner.model.modules()]
    try:
        with torch.no_grad():
            probabilities = np.asarray(learner.predict(observations, support)[0])
    finally:
        for module, training in modes:
            module.training = training
    if probabilities.shape != (len(observations), 5, 3) or not np.isfinite(probabilities).all():
        raise ValueError("finite query by five-action by three-horizon probabilities required")
    epsilon = np.finfo(np.float32).eps
    if np.any(probabilities < -epsilon) or np.any(probabilities > 1+epsilon):
        raise ValueError("predicted probabilities exceed float32 epsilon bounds")
    return probabilities


def predict_performed(learner, experience, support):
    """One preserved-mode forward, then reported performed-action Brier scoring.

    The caller records original-arrival predictions before training and supplies
    the same preceding history to every frozen shadow. Probabilities are not
    clipped when computing the metric.
    """
    _experience(experience, "scoring query")
    probabilities = predict_all(learner, experience.observations, support)
    selected = probabilities[np.arange(len(experience)), experience.actions]
    error = float(np.mean((selected.astype(np.float64)-experience.survival.astype(np.float64))**2))
    if not math.isfinite(error):
        raise ValueError("nonfinite performed-action prediction error")
    return selected, error
