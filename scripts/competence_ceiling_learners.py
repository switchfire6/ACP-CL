"""Learners for docs/competence_ceiling_protocol.md (Step B2).

* ``WeightEMA``: an exponential moving average of an online learner's weights
  (decay per optimiser step), held in the parameters of a separate, identically
  initialised ``RepresentationLearner`` (the *shadow*). It is updated by a
  ``torch.optim`` step POST-hook on the online learner's optimiser, i.e. after
  every ``optimizer.step()`` of the unchanged ``RepresentationLearner.train``.
  The hook only reads the online parameters; optimiser hooks are not part of
  ``Optimizer.state_dict()`` or of the pickled optimiser, so the online learner,
  its checkpoint and its ``learner_signature`` are unchanged (verified by
  ``scripts/competence_ceiling.py check`` against Step B's own ``run_fit``).
* OFF-pixel helpers: the performed-action BCE of the reference learner on one
  causal packet (support = the packet's actual preceding batch, None for the
  first packet, exactly as ``RepresentationLearner.train``), the per-step cosine
  learning rate, and a validation pass.
* OFF-feature: PRIVILEGED PERCEPTION ceilings. An MLP on exact latent features
  [reserves/18 (2), source/lossy/delayed factor bits (3), cue signal bits (3)],
  trained with performed-action labels (OFF-feature) or, additionally
  privileged, with all five actions' counterfactual labels (OFF-feature-all).
  These are ceilings, never candidate learners.

Project source files are imported, never edited.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from acp_cl.conditional.learner import Packet  # noqa: E402
from acp_cl.persistence.learner import state_hash  # noqa: E402
from acp_cl.persistence.world import TransferWorld  # noqa: E402

FEATURE_LABEL = ('PRIVILEGED PERCEPTION CEILING: exact latent features (reserves, factors, cue signals); '
                 'never a candidate learner')
FEATURE_ALL_LABEL = ('PRIVILEGED PERCEPTION AND LABELS CEILING: exact latent features and all five actions\' '
                     'counterfactual labels; descriptive only, never a candidate learner')
FEATURE_WIDTH = 8


# ------------------------------------------------------------------ weight EMA

class WeightEMA:
    """EMA of ``learner``'s parameters held in ``shadow.model`` (same architecture).

    ``shadow`` must start from the online learner's initial weights; the EMA is
    therefore initialised to the initial weights. ``_update`` runs after every
    optimiser step (post-hook) and is never used for training.
    """

    def __init__(self, learner, shadow, decay, steps=0):
        if not 0. < decay < 1.:
            raise ValueError('EMA decay must lie in (0, 1)')
        online = list(learner.model.named_parameters())
        held = list(shadow.model.named_parameters())
        if [n for n, _ in online] != [n for n, _ in held]:
            raise ValueError('EMA shadow architecture differs')
        if any(True for _ in learner.model.buffers()) or any(True for _ in shadow.model.buffers()):
            raise ValueError('buffers would need their own averaging rule')
        for (_, s), (_, p) in zip(held, online):
            if s.shape != p.shape or s.dtype != p.dtype or s.data_ptr() == p.data_ptr():
                raise ValueError('EMA shadow must hold separate tensors of identical shape')
        self.pairs = [(s, p) for (_, s), (_, p) in zip(held, online)]
        self.decay = float(decay)
        self.steps = int(steps)
        self.shadow = shadow
        self.handle = learner.optimizer.register_step_post_hook(self._update)

    @torch.no_grad()
    def _update(self, optimizer, args, kwargs):
        for s, p in self.pairs:
            s.mul_(self.decay).add_(p.detach(), alpha=1. - self.decay)
        self.steps += 1

    def remove(self):
        self.handle.remove()

    def state(self):
        return {k: v.detach().clone() for k, v in self.shadow.model.state_dict().items()}


def ema_reference(snapshots, decay):
    """Independent float64 closed form: d^k th_0 + sum_j (1-d) d^(k-j) th_j."""
    k = len(snapshots) - 1
    result = {}
    for name in snapshots[0]:
        value = decay ** k * snapshots[0][name].double()
        for j in range(1, k + 1):
            value = value + (1 - decay) * decay ** (k - j) * snapshots[j][name].double()
        result[name] = value
    return result


# ------------------------------------------------------------------ OFF-pixel

def offline_packets(stream):
    """Causal packets: query = batch i, support = batch i-1 (None for i = 0), as online."""
    return [Packet(stream[i - 1] if i else None, stream[i]) for i in range(len(stream))]


def split_packets(packets, fraction):
    """Hold out the last ``round(fraction * P)`` packets for validation."""
    count = len(packets)
    validation = int(round(fraction * count))
    if not 0 < validation < count:
        raise ValueError('validation split must leave both parts nonempty')
    return count - validation, validation


def packet_loss(learner, packet):
    """Performed-action BCE of RepresentationLearner.train (probabilities clamped 1e-6)."""
    probs = learner.forward_with_features(packet.query.observations, packet.support)[0]
    _, actions, y = learner.tensors(packet.query)
    selected = probs[torch.arange(len(actions), device=learner.device), actions]
    return F.binary_cross_entropy(selected.clamp(1e-6, 1 - 1e-6), y)


def cosine_lr(base, step, total):
    """Per-optimiser-step cosine decay from ``base`` (step 0) towards 0 at step ``total``."""
    return base * .5 * (1. + math.cos(math.pi * step / total))


@torch.no_grad()
def packet_validation(learner, packets):
    """Mean performed-action BCE and performed-action Brier over validation packets."""
    modes = [(m, m.training) for m in learner.model.modules()]
    try:
        losses, briers = [], []
        for packet in packets:
            probs = learner.forward_with_features(packet.query.observations, packet.support)[0]
            _, actions, y = learner.tensors(packet.query)
            selected = probs[torch.arange(len(actions)), actions]
            losses.append(float(F.binary_cross_entropy(selected.clamp(1e-6, 1 - 1e-6), y)))
            briers.append(float(((selected.double() - y.double()) ** 2).mean()))
    finally:
        for m, t in modes:
            m.training = t
    return float(np.mean(losses)), float(np.mean(briers))


# ------------------------------------------------------------------ OFF-feature

def latent_features(reserves, factors, signals):
    """[reserves/18 (2), source bit (source>0), lossy, delayed, three cue signal bits]."""
    reserves = np.asarray(reserves, dtype=np.float64)
    factors = np.asarray(factors)
    signals = np.asarray(signals)
    if not np.isin(factors[:, 0], (-1, 1)).all() or not np.isin(factors[:, 1:], (0, 1)).all():
        raise ValueError('unexpected factor coding')
    if not np.isin(signals, (0, 1)).all():
        raise ValueError('unexpected signal coding')
    x = np.concatenate([reserves / TransferWorld.capacity, (factors[:, :1] > 0).astype(np.float64),
                        factors[:, 1:3].astype(np.float64), signals.astype(np.float64)], axis=1)
    return x.astype(np.float32)


class FeatureMLP(nn.Module):
    def __init__(self, inputs=FEATURE_WIDTH, hidden=128, layers=2):
        super().__init__()
        parts, width = [], inputs
        for _ in range(layers):
            parts += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        parts.append(nn.Linear(width, 15))
        self.net = nn.Sequential(*parts)

    def forward(self, x):
        return self.net(x).reshape(-1, 5, 3).sigmoid()


def feature_objective(probs, actions, performed, truth, all_actions):
    """Training/validation objective: performed-action or all-action BCE (clamped like the reference)."""
    probs = probs.clamp(1e-6, 1 - 1e-6)
    if all_actions:
        return F.binary_cross_entropy(probs, truth)
    selected = probs[torch.arange(len(actions)), actions]
    return F.binary_cross_entropy(selected, performed)


def make_member(init_seed, spec):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(init_seed))
        return FeatureMLP(FEATURE_WIDTH, spec['hidden'], spec['layers'])


def train_member(model, data, train, validation, spec, weight_decay, all_actions, shuffle_key):
    """Adam, minibatches, early stopping on validation loss; returns the best-epoch state."""
    X, A, Y, T = data
    xt, at, yt, tt = X[train], A[train], Y[train], T[train]
    xv, av, yv, tv = X[validation], A[validation], Y[validation], T[validation]
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['lr'], weight_decay=weight_decay)
    best, best_state, best_epoch, curve = None, None, 0, []
    batch, n = spec['minibatch'], len(xt)
    for epoch in range(1, spec['max_epochs'] + 1):
        order = torch.as_tensor(np.random.default_rng([*shuffle_key, epoch]).permutation(n))
        model.train()
        total = 0.
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            loss = feature_objective(model(xt[idx]), at[idx], yt[idx], tt[idx], all_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            value = float(feature_objective(model(xv), av, yv, tv, all_actions))
        curve.append([epoch, total / n, value])
        if best is None or value < best:
            best, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= spec['patience']:
            break
    model.load_state_dict(best_state)
    return dict(best_epoch=best_epoch, epochs_run=len(curve), best_validation=best,
                curve=curve, state_sha256=state_hash(model.state_dict()),
                optimizer_steps=len(curve) * math.ceil(n / batch))


@torch.no_grad()
def member_probabilities(model, x):
    model.eval()
    return model(torch.as_tensor(x)).numpy().astype(np.float64)
