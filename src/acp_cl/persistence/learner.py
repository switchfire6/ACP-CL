"""A from-scratch recurrent survival predictor with fixed parameter capacity."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .memory import Memory
from .world import ACTIONS, HORIZONS, IMAGE_SHAPE, Experience, concatenate


class SurvivalNetwork(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(IMAGE_SHAPE[1] * IMAGE_SHAPE[2], width), nn.Tanh())
        self.temporal = nn.GRU(width, width, batch_first=True)
        self.head = nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                                  nn.Linear(width, len(ACTIONS) * len(HORIZONS)))

    def forward(self, observations):
        x = observations.float() / 255
        features = self.frame(x.flatten(start_dim=2))
        _, state = self.temporal(features)
        return self.head(state[-1]).reshape(-1, len(ACTIONS), len(HORIZONS))


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Learner:
    def __init__(self, method, seed, settings, device="cpu"):
        self.method, self.device = method, torch.device(device)
        self.settings = dict(settings)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 73019)
            self.model = SurvivalNetwork(settings["width"]).to(self.device)
        self.initial_hash = state_hash(self.model.state_dict())
        self.initial_encoder = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                                if not k.startswith("head.")}
        if method == "frozen_encoder":
            for module in (self.model.frame, self.model.temporal):
                module.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=settings["lr"])
        self.memory = Memory(method, settings["memory_capacity"], seed)
        self.cost = dict(arrivals=0, optimizer_steps=0, current_presentations=0,
                         replay_presentations=0, duplicate_current_presentations=0,
                         training_forwards=0, scoring_forwards=0, evaluation_forwards=0,
                         training_seconds=0.0, selection_seconds=0.0)
        self.peak_memory_payload = self.memory.nbytes()
        self.peak_candidate_payload = 0

    def freeze_encoder(self):
        for module in (self.model.frame, self.model.temporal):
            module.requires_grad_(False)

    @torch.no_grad()
    def predict(self, observations, *, evaluation=True):
        self.model.eval()
        parts = []
        for start in range(0, len(observations), 512):
            inputs = torch.from_numpy(observations[start:start + 512]).to(self.device)
            # Monotone survival curves used for action selection and scoring.
            probs = self.model(inputs).sigmoid().cummin(dim=-1).values
            parts.append(probs.cpu().numpy())
        self.cost["evaluation_forwards" if evaluation else "scoring_forwards"] += len(observations)
        return np.concatenate(parts)

    def train(self, current: Experience):
        start_time = time.perf_counter()
        current_size = len(current)
        for _ in range(self.settings["updates_per_batch"]):
            replay = self.memory.sample(current_size)
            if replay is None:
                combined = concatenate(current, current)
                self.cost["duplicate_current_presentations"] += current_size
            else:
                combined = concatenate(current, replay)
                self.cost["replay_presentations"] += current_size
            self.model.train()
            x = torch.from_numpy(combined.observations).to(self.device)
            actions = torch.from_numpy(combined.actions.astype(np.int64)).to(self.device)
            y = torch.from_numpy(combined.survival.astype(np.float32)).to(self.device)
            outputs = self.model(x)[torch.arange(len(combined), device=self.device), actions]
            loss = F.binary_cross_entropy_with_logits(outputs, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.cost["current_presentations"] += current_size
            self.cost["training_forwards"] += len(combined)
            self.cost["optimizer_steps"] += 1
        pool = self.memory.candidates(current)
        probabilities = self.predict(pool.observations, evaluation=False)
        # Estimated action sensitivity of survival, NOT causal value of a memory.
        utility = np.ptp(probabilities[:, :, -1], axis=1)
        scoring_end = time.perf_counter()
        self.memory.add(current, utility)
        selection_end = time.perf_counter()
        self.cost["training_seconds"] += scoring_end - start_time
        self.cost["selection_seconds"] += selection_end - scoring_end
        self.cost["arrivals"] += current_size
        self.peak_memory_payload = max(self.peak_memory_payload, self.memory.nbytes())
        self.peak_candidate_payload = max(self.peak_candidate_payload,
            pool.observations.nbytes + pool.actions.nbytes + pool.survival.nbytes
            + probabilities.nbytes + utility.nbytes)

    def diagnostics(self):
        state = self.model.state_dict()
        drift = sum(float((state[k].detach().cpu() - initial).square().sum())
                    for k, initial in self.initial_encoder.items()) ** 0.5
        optimizer_bytes = sum(value.numel() * value.element_size()
                              for values in self.optimizer.state.values() for value in values.values()
                              if isinstance(value, torch.Tensor))
        return dict(initial_hash=self.initial_hash, final_hash=state_hash(state),
                    parameters=sum(p.numel() for p in self.model.parameters()),
                    encoder_displacement_l2=drift,
                    model_payload_bytes=sum(v.numel() * v.element_size() for v in state.values()),
                    optimizer_payload_bytes=optimizer_bytes,
                    peak_memory_payload_bytes=self.peak_memory_payload,
                    peak_candidate_payload_bytes=self.peak_candidate_payload,
                    occupied_sketch_bins=int(np.count_nonzero(self.memory.history.counts)),
                    final_memory_ids=self.memory.ids.tolist(),
                    history_counts=self.memory.history.counts.tolist(),
                    cost=dict(self.cost))
