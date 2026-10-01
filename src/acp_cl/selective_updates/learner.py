"""Immediate learning with optional first-order protection of sampled replay.

The constraint acts on Adam's actual proposed parameter displacement, after
the unchanged clipping and moment update. It is a local, sampled, first-order
constraint on reported loss, not a guarantee about finite-step or clean loss.
"""

from __future__ import annotations

import copy
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from acp_cl.conditional.learner import Packet
from acp_cl.contextual.learner import ContextLearner
from acp_cl.replay_renewal.memory import ReservoirMemory, make_memory


ARMS = ("reference", "current", "protected", "combined", "shrink")
ARM_SETTINGS = {
    "reference": (.5, "none"),
    "current": (.75, "none"),
    "protected": (.5, "project"),
    "combined": (.75, "project"),
    "shrink": (.75, "shrink"),
}
SUMMARY_NAMES = (
    "hd_before", "hd_projected", "hd_after", "removed_norm_ratio",
    "retained_norm_ratio", "applied_removed_norm_ratio", "sampled_age_packets",
    "finite_replay_loss_before", "finite_replay_loss_after", "finite_replay_loss_change",
)
COUNTERS = (
    "steps", "packets", "training_probability_calls", "reference_gradient_calls",
    "objective_backward_calls", "reference_backward_query_presentations",
    "reference_backward_support_presentations", "diagnostic_probability_calls",
    "diagnostic_query_presentations", "diagnostic_support_presentations",
    "projection_eligible_steps", "projection_applied_steps", "shrink_applied_steps",
    "rewrite_steps", "zero_reference_steps", "zero_displacement_steps", "finite_step_increases",
)


def new_stats():
    return dict.fromkeys(COUNTERS, 0) | dict(
        max_positive_projected_residual=0., max_positive_applied_residual=0.,
        summaries={name: dict(count=0, sum=0., min=None, max=None) for name in SUMMARY_NAMES})


def accumulate(stats, name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite selective-update diagnostic")
    record = stats["summaries"][name]
    record["count"] += 1
    record["sum"] += value
    record["min"] = value if record["min"] is None else min(record["min"], value)
    record["max"] = value if record["max"] is None else max(record["max"], value)


@torch.no_grad()
def project_displacement(displacement, reference):
    """Project onto h.d <= 0 in float64; do not hide residuals with epsilon.

Inputs are one-dimensional tensors and remain unchanged. A zero reference
gradient or a nonconflicting displacement gives an exact identity operation.
The returned candidate has float64 dtype, before model-dtype rounding.
"""
    if (not isinstance(displacement, torch.Tensor) or not isinstance(reference, torch.Tensor)
            or displacement.ndim != 1 or reference.shape != displacement.shape
            or not displacement.numel() or displacement.device != reference.device):
        raise ValueError("matched nonempty flat displacement and reference required")
    d, h = displacement.detach().double(), reference.detach().double()
    if not torch.isfinite(d).all() or not torch.isfinite(h).all():
        raise ValueError("finite displacement and reference required")
    hh, dd, hd = torch.dot(h, h), torch.dot(d, d), torch.dot(h, d)
    eligible = bool(hh > 0 and hd > 0)
    projected = d-(hd/hh)*h if eligible else d
    norm, candidate_norm = float(dd.sqrt()), float(torch.linalg.vector_norm(projected))
    removed_norm = float(torch.linalg.vector_norm(d-projected))
    return projected, dict(
        projection_eligible=eligible, zero_reference=bool(hh == 0), zero_displacement=bool(dd == 0),
        hd_before=float(hd), hd_projected=float(torch.dot(h, projected)),
        displacement_norm=norm, projected_norm=candidate_norm,
        removed_norm_ratio=removed_norm/norm if norm else 0.,
        retained_norm_ratio=candidate_norm/norm if norm else 1.)


@torch.no_grad()
def choose_displacement(displacement, reference, mode):
    """All modes compute the same candidate; shrink keeps its own Adam direction."""
    if mode not in ("none", "project", "shrink"):
        raise ValueError("unknown displacement mode")
    projected, geometry = project_displacement(displacement, reference)
    chosen = displacement.detach().double()
    if geometry["projection_eligible"]:
        if mode == "project":
            chosen = projected
        elif mode == "shrink" and not geometry["zero_displacement"]:
            chosen = chosen*geometry["retained_norm_ratio"]
    return chosen, geometry


def _flat(parameters):
    return torch.cat([parameter.detach().reshape(-1).double() for parameter in parameters])


@torch.no_grad()
def _write_parameters(parameters, values):
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.copy_(values[offset:offset+count].reshape_as(parameter))
        offset += count
    if offset != values.numel():
        raise ValueError("parameter vector length mismatch")


class SelectiveLearner(ContextLearner):
    """An unchanged predictor with a bounded replay-aware training intervention."""

    def __init__(self, method, seed, settings, device="cpu", arm="reference"):
        super().__init__(method, seed, settings, device)
        self.memory = make_memory("uniform", settings, seed)
        self._configure(arm)

    def _configure(self, arm):
        if arm not in ARM_SETTINGS:
            raise ValueError("unknown selective-update arm")
        if self.method not in ("conditional", "recurrent"):
            raise ValueError("selective updates require an ordinary conditional or recurrent learner")
        if not isinstance(self.memory, ReservoirMemory):
            raise ValueError("selective updates require unchanged uniform reservoir memory")
        if not all(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("all model parameters must remain trainable")
        if not isinstance(self.optimizer, torch.optim.Adam):
            raise ValueError("selective updates require the existing Adam optimizer")
        self.selective_arm = arm
        self.current_weight, self.displacement_mode = ARM_SETTINGS[arm]
        self.selective_stats = new_stats()

    def _loss(self, packet):
        if packet.oracle_modes is not None:
            raise ValueError("ordinary learners must not receive latent modes")
        probabilities, _, _ = self.probabilities(packet.query.observations, packet.support)
        _, actions, outcomes = self.tensors(packet.query)
        selected = probabilities[torch.arange(len(actions), device=self.device), actions]
        return F.binary_cross_entropy(selected.clamp(1e-6, 1-1e-6), outcomes)

    def train(self, current, oracle_modes=None):
        started = time.perf_counter()
        if oracle_modes is not None:
            raise ValueError("ordinary learners must not receive latent modes")
        packet = Packet(self.history, current)
        parameters = list(self.model.parameters())
        stats = self.selective_stats
        for update in range(self.settings["updates_per_batch"]):
            replay = self.memory.sample()
            if replay is None:
                replay, replay_age = packet, 0
                self.cost["duplicate_presentations"] += len(current)
            else:
                # Identity lookup observes the selected packet without another
                # RNG draw; lifetime IDs stay aligned with uniform reservoir slots.
                slot = next(i for i, stored in enumerate(self.memory.packets) if stored is replay)
                # Match earlier studies: newest previous packet has age zero.
                replay_age = self.memory.seen-1-self.memory.ids[slot]
                self.cost["replay_presentations"] += len(replay.query)
            self.model.train()
            losses = []
            for item in (packet, replay):
                losses.append(self._loss(item))
                self.cost["query_presentations"] += len(item.query)
                self.cost["support_presentations"] += self.settings["batch_size"]
                stats["training_probability_calls"] += 1
            # This exact expression preserves the original half/half baseline.
            loss = (torch.stack(losses).mean() if self.current_weight == .5 else
                    self.current_weight*losses[0]+(1-self.current_weight)*losses[1])
            reference_parts = torch.autograd.grad(losses[1], parameters, retain_graph=True)
            reference = torch.cat([part.detach().reshape(-1).double() for part in reference_parts])
            stats["reference_gradient_calls"] += 1
            stats["reference_backward_query_presentations"] += len(replay.query)
            stats["reference_backward_support_presentations"] += self.settings["batch_size"]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            stats["objective_backward_calls"] += 1
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            original = _flat(parameters)
            self.optimizer.step()
            proposed = _flat(parameters)
            displacement = proposed-original
            chosen, geometry = choose_displacement(displacement, reference, self.displacement_mode)
            # Never rewrite an unprojected update: float32 subtract/add could
            # otherwise change even a nominally identical reference trajectory.
            rewriting = self.displacement_mode != "none" and geometry["projection_eligible"]
            if rewriting:
                _write_parameters(parameters, original+chosen)
            actual = _flat(parameters)-original if rewriting else displacement
            correction = float(torch.linalg.vector_norm(actual-displacement))
            hd_after = float(torch.dot(reference, actual))
            stats["steps"] += 1
            stats["projection_eligible_steps"] += int(geometry["projection_eligible"])
            stats["rewrite_steps"] += int(rewriting)
            stats["projection_applied_steps"] += int(self.displacement_mode == "project" and correction > 0)
            stats["shrink_applied_steps"] += int(self.displacement_mode == "shrink" and correction > 0)
            stats["zero_reference_steps"] += int(geometry["zero_reference"])
            stats["zero_displacement_steps"] += int(geometry["zero_displacement"])
            if geometry["projection_eligible"]:
                stats["max_positive_projected_residual"] = max(
                    stats["max_positive_projected_residual"], geometry["hd_projected"])
                if self.displacement_mode == "project":
                    stats["max_positive_applied_residual"] = max(
                        stats["max_positive_applied_residual"], hd_after)
            for name in ("hd_before", "hd_projected", "removed_norm_ratio", "retained_norm_ratio"):
                accumulate(stats, name, geometry[name])
            accumulate(stats, "hd_after", hd_after)
            accumulate(stats, "applied_removed_norm_ratio",
                       correction/geometry["displacement_norm"] if geometry["displacement_norm"] else 0.)
            accumulate(stats, "sampled_age_packets", replay_age)
            self.cost["optimizer_steps"] += 1
            if update == self.settings["updates_per_batch"]-1:
                # Measure actual finite-step reported replay loss on this same
                # sampled packet. This diagnostic never controls the update.
                before_loss = float(losses[1].detach())
                with torch.no_grad():
                    after_loss = float(self._loss(replay))
                stats["diagnostic_probability_calls"] += 1
                stats["diagnostic_query_presentations"] += len(replay.query)
                stats["diagnostic_support_presentations"] += self.settings["batch_size"]
                stats["finite_step_increases"] += int(after_loss > before_loss)
                accumulate(stats, "finite_replay_loss_before", before_loss)
                accumulate(stats, "finite_replay_loss_after", after_loss)
                accumulate(stats, "finite_replay_loss_change", after_loss-before_loss)
            # Do not keep parameter-sized temporary vectors into the next update.
            del reference_parts, reference, original, proposed, displacement, chosen, actual
            del loss, losses
        self.memory.add(packet)
        self.history = current
        self.cost["arrivals"] += len(current)
        self.cost["training_seconds"] += time.perf_counter()-started
        self.peak_memory_bytes = max(self.peak_memory_bytes, self.memory.nbytes())
        stats["packets"] += 1

    def diagnostics(self):
        return super().diagnostics() | dict(selective_updates=dict(
            arm=self.selective_arm, current_weight=self.current_weight, mode=self.displacement_mode,
            stats=copy.deepcopy(self.selective_stats),
            temporary_vector_dtype="float64",
            temporary_parameter_gradient_buffer_bytes_estimate=sum(
                p.numel()*(p.element_size()+9*8) for p in self.model.parameters()),
            temporary_buffer_estimate_scope=("Conservative allowance for one model-dtype reference-gradient "
                "vector and nine float64 parameter-sized vectors. Excludes autograd activations, "
                "model/Adam storage, operator workspaces and allocator overhead; not measured peak RAM.")))


def fork(learner, arm):
    """Copy every trained state, changing only the update rule and its counters."""
    if not isinstance(learner, ContextLearner):
        raise ValueError("a contextual learner is required")
    result = copy.deepcopy(learner)
    # torch.nn.Parameter.__deepcopy__ does not preserve accumulated .grad.
    # They are cleared by the next update, but the fork itself preserves them.
    for original, cloned in zip(learner.model.parameters(), result.model.parameters()):
        cloned.grad = None if original.grad is None else original.grad.detach().clone()
    result.__class__ = SelectiveLearner
    result._configure(arm)
    return result
