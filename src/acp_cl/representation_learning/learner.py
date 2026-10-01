"""One recurrent predictor with a controlled query-latent auxiliary objective."""

from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from acp_cl.conditional.learner import Packet
from acp_cl.contextual.learner import ContextLearner
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.world import Experience, IMAGE_SHAPE
from acp_cl.predictive_value.mechanism import _value_hash, state_signature


ARMS = ('outcome', 'final_frame', 'sequence', 'compute')
AUXILIARY_ARMS = ('final_frame', 'sequence')
AUXILIARY_SEED_OFFSET = 98147


def reconstruction_target(observations, arm):
    """Normalize query pixels; the spatial control repeats its last frame."""
    if arm not in AUXILIARY_ARMS:
        raise ValueError('Only the auxiliary arms have reconstruction targets')
    observations = torch.as_tensor(observations)
    if observations.ndim != 4 or tuple(observations.shape[1:]) != IMAGE_SHAPE:
        raise ValueError('Reconstruction requires [N, 4, 12, 16] query images')
    target = observations.float() / 255
    return target if arm == 'sequence' else target[:, -1:].expand(-1, IMAGE_SHAPE[0], -1, -1)


def learner_signature(learner, ignore_walltime=False):
    """Include arm identity and auxiliary initialization in the complete state."""
    if not isinstance(learner, RepresentationLearner):
        raise ValueError('A RepresentationLearner is required')
    return _value_hash(dict(ordinary=state_signature(learner, ignore_walltime), arm=learner.arm,
        initial_predictor_hash=learner.initial_predictor_hash,
        initial_auxiliary_hash=learner.initial_auxiliary_hash))


class RepresentationLearner(ContextLearner):
    def __init__(self, arm, seed, settings, device='cpu'):
        if arm not in ARMS:
            raise ValueError('Unknown representation-learning arm')
        compute_updates = settings['compute_updates_per_batch']
        if type(compute_updates) is not int or compute_updates < 1:
            raise ValueError('Compute-control updates must be a positive integer')
        weight = settings['auxiliary_weight']
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Auxiliary weight must be positive and finite')
        super().__init__('recurrent', seed, settings, device)
        self.arm = arm
        self.initial_predictor_hash = self.initial_hash
        self.initial_auxiliary_hash = None
        if arm in AUXILIARY_ARMS:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + AUXILIARY_SEED_OFFSET)
                width = settings['width']
                self.model.auxiliary = nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                    nn.Linear(width, math.prod(IMAGE_SHAPE))).to(self.device)
            self.initial_auxiliary_hash = state_hash(self.model.auxiliary.state_dict())
            self.initial_hash = state_hash(self.model.state_dict())
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=settings['lr'])
        self.cost.update(outcome_loss_sum=0., reconstruction_loss_sum=0., objective_loss_sum=0.,
            encoder_forward_calls=0, encoder_presentations=0, auxiliary_forward_calls=0,
            auxiliary_reconstructions=0, reconstruction_pixels=0, auxiliary_backward_steps=0,
            predictive_parameter_updates=0, auxiliary_parameter_updates=0)

    @property
    def updates_per_batch(self):
        key = 'compute_updates_per_batch' if self.arm == 'compute' else 'updates_per_batch'
        return self.settings[key]

    def forward_with_features(self, observations, support=None, oracle_modes=None, override=None):
        """Return native probabilities/weights/logits and their exact query latent.

        The recurrent computation and operation order match ContextLearner. The
        auxiliary decoder can consume this latent without another encoder pass.
        """
        if oracle_modes is not None:
            raise ValueError('Ordinary models must not receive latent modes')
        if override not in (None, 'shuffled'):
            raise ValueError('Unknown history override')
        query = torch.as_tensor(observations, device=self.device)
        if support is None:
            sx = torch.zeros((self.settings['batch_size'], *IMAGE_SHAPE), dtype=torch.uint8,
                             device=self.device)
            sa = torch.zeros(len(sx), dtype=torch.long, device=self.device)
            sy = torch.zeros((len(sx), 3), device=self.device)
        else:
            sx, sa, sy = self.tensors(support)
        if override == 'shuffled':
            sx = sx.roll(1, dims=0)
        features = self.model.encode(torch.cat((sx, query)))
        events = torch.cat((features[:len(sx)], F.one_hot(sa, 5).float(), sy), dim=-1)
        if getattr(self.model, 'interaction_features', False):
            interactions = (features[:len(sx), :, None] * F.one_hot(sa, 5)[:, None, :]
                            * (2 * sy[:, -1] - 1)[:, None, None]).flatten(start_dim=1)
            events = torch.cat((events, interactions), dim=-1)
            context = self.model.context(events[None])[0].mean(dim=1)
        else:
            context = self.model.context(events[None])[1][-1]
        if support is None:
            context = context * 0
        query_features = features[len(sx):]
        logits = self.model.decoder(torch.cat((query_features,
            context.expand(len(query), -1)), dim=-1)).reshape(-1, 5, 3)
        weights = torch.ones((len(query), 1), device=self.device)
        return logits.sigmoid(), weights, logits[:, None], query_features

    def probabilities(self, observations, support, oracle_modes=None, override=None):
        return self.forward_with_features(observations, support, oracle_modes, override)[:3]

    @torch.no_grad()
    def predict(self, observations, support=None, oracle_modes=None, override=None):
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            return super().predict(observations, support, oracle_modes, override)
        finally:
            for module, mode in modes:
                module.training = mode

    @torch.no_grad()
    def reconstruction(self, observations):
        """An evaluation-only reconstruction; it does not train or count work."""
        if self.arm not in AUXILIARY_ARMS:
            raise ValueError('This arm has no reconstruction decoder')
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            query = torch.as_tensor(observations, device=self.device)
            latent = self.model.encode(query)
            return self.model.auxiliary(latent).sigmoid().reshape(-1, *IMAGE_SHAPE).cpu().numpy()
        finally:
            for module, mode in modes:
                module.training = mode

    def train(self, current, oracle_modes=None):
        if oracle_modes is not None:
            raise ValueError('Ordinary learners must not receive latent modes')
        if not isinstance(current, Experience) or not len(current):
            raise ValueError('Training requires a nonempty ordinary Experience')
        started = time.perf_counter()
        packet = Packet(self.history, current)
        auxiliary = self.arm in AUXILIARY_ARMS
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
            losses, reconstructions = [], []
            for item in (packet, replay):
                probs, _, _, latent = self.forward_with_features(item.query.observations,
                                                                item.support, item.oracle_modes)
                query, actions, y = self.tensors(item.query)
                selected = probs[torch.arange(len(actions), device=self.device), actions]
                losses.append(F.binary_cross_entropy(selected.clamp(1e-6, 1 - 1e-6), y))
                self.cost['query_presentations'] += len(item.query)
                self.cost['support_presentations'] += self.settings['batch_size']
                self.cost['encoder_forward_calls'] += 1
                self.cost['encoder_presentations'] += len(item.query) + (
                    self.settings['batch_size'] if item.support is None else len(item.support))
                if auxiliary:
                    reconstruction = self.model.auxiliary(latent).sigmoid().reshape(-1, *IMAGE_SHAPE)
                    reconstructions.append(F.mse_loss(reconstruction, reconstruction_target(query, self.arm)))
                    self.cost['auxiliary_forward_calls'] += 1
                    self.cost['auxiliary_reconstructions'] += len(item.query)
                    self.cost['reconstruction_pixels'] += len(item.query) * math.prod(IMAGE_SHAPE)
            outcome_loss = torch.stack(losses).mean()
            loss = outcome_loss
            if auxiliary:
                reconstruction_loss = torch.stack(reconstructions).mean()
                loss = outcome_loss + self.settings['auxiliary_weight'] * reconstruction_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.cost['optimizer_steps'] += 1
            self.cost['predictive_parameter_updates'] += predictive_parameters
            self.cost['auxiliary_parameter_updates'] += auxiliary_parameters
            self.cost['outcome_loss_sum'] += float(outcome_loss.detach())
            self.cost['objective_loss_sum'] += float(loss.detach())
            if auxiliary:
                self.cost['reconstruction_loss_sum'] += float(reconstruction_loss.detach())
                self.cost['auxiliary_backward_steps'] += 1
        self.memory.add(packet)
        self.history = current
        self.cost['arrivals'] += len(current)
        self.cost['training_seconds'] += time.perf_counter() - started
        self.peak_memory_bytes = max(self.peak_memory_bytes, self.memory.nbytes())

    def diagnostics(self):
        report = super().diagnostics()
        auxiliary_parameters = sum(p.numel() for name, p in self.model.named_parameters()
                                   if name.startswith('auxiliary.'))
        steps = self.cost['optimizer_steps']
        report.update(arm=self.arm, initial_predictor_hash=self.initial_predictor_hash,
            initial_auxiliary_hash=self.initial_auxiliary_hash,
            predictive_parameters=report['parameters'] - auxiliary_parameters,
            auxiliary_parameters=auxiliary_parameters, replay_capacity=self.memory.capacity,
            effective_updates_per_batch=self.updates_per_batch,
            mean_outcome_loss=self.cost['outcome_loss_sum'] / steps if steps else None,
            mean_reconstruction_loss=(self.cost['reconstruction_loss_sum'] / steps
                                      if steps and auxiliary_parameters else None),
            mean_objective_loss=self.cost['objective_loss_sum'] / steps if steps else None)
        return report
