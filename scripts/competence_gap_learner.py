"""PRIVILEGED DIAGNOSTIC learner for docs/competence_gap_protocol.md (Step B).

``AllActionLearner`` is the unchanged recurrent reference
(``RepresentationLearner`` arm ``outcome``) whose training loss uses the
evaluator's counterfactual survival labels for ALL five actions and all three
horizons of the current and the replayed packet, instead of the performed
action's labels only. It measures the cost of performed-action-only feedback;
it is never a candidate learner.

Everything else is copied verbatim from ``RepresentationLearner.train`` for the
``outcome`` arm: one current packet plus one uniform-reservoir replay packet per
update (the current packet doubles as replay while the memory is empty), the
packet's actual preceding batch as support, mean of the two packet losses, Adam,
gradient-norm clip 5, the same replay RNG and the same cost counters. With
``all_actions=False`` the class reproduces ``RepresentationLearner.train``
exactly (checked by ``scripts/competence_gap.py check``). Project source files are
imported, never edited.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from dataclasses import dataclass  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from acp_cl.conditional.learner import Packet  # noqa: E402
from acp_cl.persistence.world import Experience  # noqa: E402
from acp_cl.representation_learning.learner import RepresentationLearner  # noqa: E402

PRIVILEGED_LABEL = 'PRIVILEGED DIAGNOSTIC: all-action counterfactual feedback; never a candidate learner'


@dataclass
class TruthPacket(Packet):
    """A causal packet that also carries the evaluator's (n, 5, 3) counterfactual labels."""

    truth: np.ndarray | None = None

    def nbytes(self):
        return super().nbytes() + (0 if self.truth is None else self.truth.nbytes)


class AllActionLearner(RepresentationLearner):
    def __init__(self, seed, settings, all_actions=True, device='cpu'):
        super().__init__('outcome', seed, settings, device)
        self.all_actions = bool(all_actions)
        self.privileged = self.all_actions
        self.privileged_label_presentations = 0

    def packet_loss(self, item):
        probs, _, _, _ = self.forward_with_features(item.query.observations, item.support,
                                                    item.oracle_modes)
        query, actions, y = self.tensors(item.query)
        if self.all_actions:
            if item.truth is None:
                raise ValueError('all-action loss requires counterfactual labels')
            target = torch.as_tensor(item.truth.astype(np.float32), device=self.device)
            return F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), target)
        selected = probs[torch.arange(len(actions), device=self.device), actions]
        return F.binary_cross_entropy(selected.clamp(1e-6, 1 - 1e-6), y)

    def train(self, current, truth=None, oracle_modes=None):
        if oracle_modes is not None:
            raise ValueError('Ordinary learners must not receive latent modes')
        if not isinstance(current, Experience) or not len(current):
            raise ValueError('Training requires a nonempty ordinary Experience')
        truth = None if truth is None else np.array(truth, dtype=np.uint8, copy=True)
        if self.all_actions:
            if truth is None or truth.shape != (len(current), 5, 3):
                raise ValueError('all-action training requires (n, 5, 3) counterfactual labels')
        if truth is not None and not np.array_equal(
                truth[np.arange(len(current)), current.actions], current.survival):
            raise ValueError('counterfactual labels disagree with the performed outcome')
        started = time.perf_counter()
        packet = TruthPacket(self.history, current, None, truth)
        predictive_parameters = sum(p.numel() for name, p in self.model.named_parameters()
                                    if not name.startswith('auxiliary.'))
        auxiliary_parameters = sum(p.numel() for name, p in self.model.named_parameters()
                                   if name.startswith('auxiliary.'))
        for _ in range(self.updates_per_batch):
            replay = self.memory.sample()
            if replay is None:
                replay = packet
                self.cost['duplicate_presentations'] += len(current)
            else:
                self.cost['replay_presentations'] += len(replay.query)
            self.model.train()
            losses = []
            for item in (packet, replay):
                losses.append(self.packet_loss(item))
                self.cost['query_presentations'] += len(item.query)
                self.cost['support_presentations'] += self.settings['batch_size']
                self.cost['encoder_forward_calls'] += 1
                self.cost['encoder_presentations'] += len(item.query) + (
                    self.settings['batch_size'] if item.support is None else len(item.support))
                if self.all_actions:
                    self.privileged_label_presentations += int(item.truth.size)
            outcome_loss = torch.stack(losses).mean()
            loss = outcome_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.cost['optimizer_steps'] += 1
            self.cost['predictive_parameter_updates'] += predictive_parameters
            self.cost['auxiliary_parameter_updates'] += auxiliary_parameters
            self.cost['outcome_loss_sum'] += float(outcome_loss.detach())
            self.cost['objective_loss_sum'] += float(loss.detach())
        self.memory.add(packet)
        self.history = current
        self.cost['arrivals'] += len(current)
        self.cost['training_seconds'] += time.perf_counter() - started
        self.peak_memory_bytes = max(self.peak_memory_bytes, self.memory.nbytes())

    def diagnostics(self):
        report = super().diagnostics()
        report.update(privileged=self.privileged, privileged_label=PRIVILEGED_LABEL if self.privileged else None,
                      all_action_loss=self.all_actions,
                      privileged_label_presentations=self.privileged_label_presentations)
        return report
