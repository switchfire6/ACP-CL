"""Learned consequence hypotheses, evidence routing, and causal packet replay."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.world import Experience, IMAGE_SHAPE


METHODS = ("pooled", "conditional", "shuffled", "current_only", "frozen_features", "oracle")


@dataclass
class Packet:
    support: Experience | None
    query: Experience
    oracle_modes: np.ndarray | None = None

    def nbytes(self):
        return sum(value.nbytes for data in (self.support, self.query) if data is not None
                   for value in (data.observations, data.actions, data.survival)) + (
                       0 if self.oracle_modes is None else self.oracle_modes.nbytes)


class PacketMemory:
    def __init__(self, capacity, seed):
        self.capacity, self.seen = capacity, 0
        self.packets, self.ids = [], []
        self.membership_rng = np.random.default_rng(seed + 47201)
        self.sampling_rng = np.random.default_rng(seed + 62071)

    def add(self, packet):
        index = self.seen
        self.seen += 1
        if not self.capacity:
            return
        if len(self.packets) < self.capacity:
            self.packets.append(packet)
            self.ids.append(index)
        else:
            slot = int(self.membership_rng.integers(0, self.seen))
            if slot < self.capacity:
                self.packets[slot], self.ids[slot] = packet, index

    def sample(self):
        if not self.packets:
            return None
        return self.packets[int(self.sampling_rng.integers(0, len(self.packets)))]

    def nbytes(self):
        return sum(packet.nbytes() for packet in self.packets) + 8 * len(self.ids)


class Hypotheses(nn.Module):
    def __init__(self, width, experts):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(192, width), nn.Tanh())
        self.temporal = nn.GRU(width, width, batch_first=True)
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                                               nn.Linear(width, 15)) for _ in range(experts))

    def forward(self, observations):
        features = self.frame((observations.float() / 255).flatten(start_dim=2))
        _, state = self.temporal(features)
        return torch.stack([head(state[-1]).reshape(-1, 5, 3) for head in self.heads], dim=1)


def evidence_weights(support_logits, actions, survival, strength):
    """Finite-window plug-in Bayesian weights; H=12 is the sole likelihood.

    Earlier survival labels are correlated and are not multiplied as independent
    evidence. These are likelihood weights for learned, potentially miscalibrated
    models, not calibrated uncertainty about the true world.
    """
    rows = torch.arange(len(actions), device=actions.device)
    logits = support_logits[rows, :, actions, -1]
    labels = survival[:, -1, None].expand_as(logits)
    scores = -F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return torch.softmax(strength * scores.sum(dim=0), dim=0)


class ConditionalLearner:
    def __init__(self, method, seed, settings, device="cpu"):
        if method not in METHODS:
            raise ValueError("unknown method")
        self.method, self.device = method, torch.device(device)
        self.settings = dict(settings)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 73019)
            self.model = Hypotheses(settings["width"], settings["experts"]).to(self.device)
        self.initial_hash = state_hash(self.model.state_dict())
        self.initial_encoder = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                                if not k.startswith("heads.")}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=settings["lr"])
        self.memory = PacketMemory(0 if method == "current_only" else settings["memory_packets"], seed)
        self.history = None
        self.cost = dict(arrivals=0, optimizer_steps=0, query_presentations=0,
                         support_presentations=0, replay_presentations=0,
                         duplicate_presentations=0, training_seconds=0.0)
        self.peak_memory_bytes = 0

    def freeze_features(self):
        self.model.frame.requires_grad_(False)
        self.model.temporal.requires_grad_(False)

    def tensors(self, data):
        return (torch.as_tensor(data.observations, device=self.device),
                torch.as_tensor(data.actions.astype(np.int64), device=self.device),
                torch.as_tensor(data.survival.astype(np.float32), device=self.device))

    def probabilities(self, observations, support, oracle_modes=None, override=None):
        route = self.method if override is None else override
        query = torch.as_tensor(observations, device=self.device)
        # Even the masked/oracle controls execute the same support forward pass.
        if support is None:
            # Uniform prior before any experience; allocated dummy sensor input
            # keeps the encoder forward budget matched after the initial query.
            sx = torch.zeros((self.settings["batch_size"], *IMAGE_SHAPE),
                             dtype=torch.uint8, device=self.device)
            sa = torch.zeros(len(sx), dtype=torch.long, device=self.device)
            sy = torch.zeros((len(sx), 3), device=self.device)
        else:
            sx, sa, sy = self.tensors(support)
        if route == "shuffled":
            # Deterministic derangement breaks image/action-outcome binding while
            # preserving the exact marginal support records. No task information.
            sx = sx.roll(1, dims=0)
        all_logits = self.model(torch.cat((sx, query)))
        support_logits, query_logits = all_logits[:len(sx)], all_logits[len(sx):]
        experts = self.settings["experts"]
        if route == "oracle":
            if oracle_modes is None:
                raise ValueError("the privileged diagnostic requires true query modes")
            modes = torch.as_tensor(oracle_modes.astype(np.int64), device=self.device)
            weights = F.one_hot(modes, num_classes=experts).float()
        elif route == "pooled" or support is None:
            weights = torch.full((len(query), experts), 1 / experts, device=self.device)
        else:
            weight = evidence_weights(support_logits, sa, sy, self.settings["evidence_strength"])
            weights = weight[None].expand(len(query), -1)
        probs = (weights[:, :, None, None] * query_logits.sigmoid()).sum(dim=1)
        return probs, weights, query_logits

    @torch.no_grad()
    def predict(self, observations, support=None, oracle_modes=None, override=None):
        self.model.eval()
        probs, weights, logits = self.probabilities(observations, support, oracle_modes, override)
        return (probs.cummin(dim=-1).values.cpu().numpy(), weights.cpu().numpy(),
                logits.sigmoid().cummin(dim=-1).values.cpu().numpy())

    def train(self, current, oracle_modes=None):
        started = time.perf_counter()
        if self.method == "oracle" and oracle_modes is None:
            raise ValueError("oracle mode missing")
        if self.method != "oracle" and oracle_modes is not None:
            raise ValueError("ordinary learners must not receive latent modes")
        # Capture past-only support BEFORE receiving this query's feedback.
        packet = Packet(self.history, current, oracle_modes)
        for _ in range(self.settings["updates_per_batch"]):
            replay = self.memory.sample()
            if replay is None:
                replay = packet
                self.cost["duplicate_presentations"] += len(current)
            else:
                self.cost["replay_presentations"] += len(replay.query)
            self.model.train()
            losses = []
            for item in (packet, replay):
                probs, _, _ = self.probabilities(item.query.observations, item.support,
                                                item.oracle_modes)
                _, actions, y = self.tensors(item.query)
                selected = probs[torch.arange(len(actions), device=self.device), actions]
                losses.append(F.binary_cross_entropy(selected.clamp(1e-6, 1 - 1e-6), y))
                self.cost["query_presentations"] += len(item.query)
                self.cost["support_presentations"] += self.settings["batch_size"]
            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.cost["optimizer_steps"] += 1
        self.memory.add(packet)
        self.history = current
        self.cost["arrivals"] += len(current)
        self.cost["training_seconds"] += time.perf_counter() - started
        self.peak_memory_bytes = max(self.peak_memory_bytes, self.memory.nbytes())

    def diagnostics(self):
        state = self.model.state_dict()
        optimizer_bytes = sum(v.numel() * v.element_size() for values in self.optimizer.state.values()
                              for v in values.values() if isinstance(v, torch.Tensor))
        return dict(initial_hash=self.initial_hash, final_hash=state_hash(state),
                    parameters=sum(p.numel() for p in self.model.parameters()),
                    model_bytes=sum(v.numel() * v.element_size() for v in state.values()),
                    optimizer_bytes=optimizer_bytes, peak_replay_bytes=self.peak_memory_bytes,
                    memory_ids=list(self.memory.ids), packets_seen=self.memory.seen,
                    history_bytes=Packet(None, self.history).nbytes() if self.history is not None else 0,
                    encoder_displacement=sum(float((state[k].detach().cpu() - v).square().sum())
                                             for k, v in self.initial_encoder.items()) ** .5,
                    cost=dict(self.cost))
