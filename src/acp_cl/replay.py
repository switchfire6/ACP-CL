"""Reservoir replay with independent membership, training, and monitor RNGs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class ReplayBatch:
    x: Tensor
    y: Tensor
    logits: Tensor | None = None


class ReservoirBuffer:
    """Uniform reservoir of raw CPU examples and, optionally, saved logits.

    Repeated ``sample`` calls never change reservoir membership.  Monitor
    sampling also never changes the training-sampling RNG.  Sampling is without
    replacement within a call; monitor and training batches can overlap.
    """

    def __init__(self, capacity: int, seed: int = 0):
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        self.capacity = int(capacity)
        self.num_seen = 0
        self._x: list[Tensor] = []
        self._y: list[Tensor] = []
        self._logits: list[Tensor] | None = None
        self._reservoir_rng = torch.Generator().manual_seed(int(seed) % (2**63))
        self._sample_rngs = {
            "train": torch.Generator().manual_seed((int(seed) ^ 0x5DEECE66D) % (2**63)),
            "monitor": torch.Generator().manual_seed((int(seed) ^ 0x6A09E667) % (2**63)),
        }

    def __len__(self) -> int:
        return len(self._x)

    def add(self, x: Tensor, y: Tensor, logits: Tensor | None = None) -> None:
        if x.ndim < 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("x and one-dimensional y must have equal batch sizes")
        if logits is not None and (logits.ndim != 2 or logits.shape[0] != x.shape[0]):
            raise ValueError("logits must have shape (batch, classes)")
        if len(self):
            if x.shape[1:] != self._x[0].shape or x.dtype != self._x[0].dtype:
                raise ValueError("All stored samples must have the same shape and dtype")
            if y.dtype != self._y[0].dtype:
                raise ValueError("All stored labels must have the same dtype")
            if (logits is None) != (self._logits is None):
                raise ValueError("A buffer must consistently store logits or omit them")
            if logits is not None and self._logits is not None:
                if logits.shape[1:] != self._logits[0].shape or logits.dtype != self._logits[0].dtype:
                    raise ValueError("All stored logits must have the same shape and dtype")
        if not x.shape[0]:
            return
        if self.capacity == 0:
            self.num_seen += x.shape[0]
            return
        if not len(self) and logits is not None:
            self._logits = []
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        logits_cpu = None if logits is None else logits.detach().cpu()
        for i in range(x.shape[0]):
            self.num_seen += 1
            if len(self) < self.capacity:
                slot = len(self)
            else:
                slot = int(torch.randint(self.num_seen, (), generator=self._reservoir_rng))
                if slot >= self.capacity:
                    continue
            sample = x_cpu[i].clone()
            label = y_cpu[i].clone()
            logit = None if logits_cpu is None else logits_cpu[i].clone()
            if slot == len(self):
                self._x.append(sample)
                self._y.append(label)
                if self._logits is not None:
                    assert logit is not None
                    self._logits.append(logit)
            else:
                self._x[slot] = sample
                self._y[slot] = label
                if self._logits is not None:
                    assert logit is not None
                    self._logits[slot] = logit

    def sample(
        self,
        size: int,
        device: str | torch.device = "cpu",
        stream: str = "train",
        *,
        generator: torch.Generator | None = None,
    ) -> ReplayBatch | None:
        if isinstance(size, bool) or int(size) != size or size < 0:
            raise ValueError("size must be a nonnegative integer")
        if stream not in self._sample_rngs:
            raise ValueError("stream must be 'train' or 'monitor'")
        if size == 0 or not len(self):
            return None
        indices = torch.randperm(len(self), generator=generator if generator is not None
                                 else self._sample_rngs[stream])[:size].tolist()
        x = torch.stack([self._x[i] for i in indices]).to(device)
        y = torch.stack([self._y[i] for i in indices]).to(device)
        logits = None
        if self._logits is not None:
            logits = torch.stack([self._logits[i] for i in indices]).to(device)
        return ReplayBatch(x, y, logits)

    def nbytes(self) -> int:
        """Tensor payload bytes actually stored, excluding Python/RNG overhead."""
        tensors = self._x + self._y + (self._logits or [])
        return sum(t.numel() * t.element_size() for t in tensors)

    def state_dict(self) -> dict[str, Any]:
        """Return a detached snapshot, including all three independent RNGs."""
        return {
            "version": 1,
            "capacity": self.capacity,
            "num_seen": self.num_seen,
            "x": torch.stack(self._x) if self._x else None,
            "y": torch.stack(self._y) if self._y else None,
            "logits": torch.stack(self._logits) if self._logits else None,
            "reservoir_rng_state": self._reservoir_rng.get_state().clone(),
            "sample_rng_states": {
                name: generator.get_state().clone()
                for name, generator in self._sample_rngs.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported reservoir checkpoint version")
        capacity, num_seen = int(state["capacity"]), int(state["num_seen"])
        x, y, logits = state["x"], state["y"], state["logits"]
        size = 0 if x is None else len(x)
        if capacity < 0 or num_seen < size or size > capacity:
            raise ValueError("Invalid reservoir checkpoint counts")
        if (x is None) != (y is None) or (y is not None and y.shape != (size,)):
            raise ValueError("Invalid reservoir checkpoint labels")
        if logits is not None and (logits.ndim != 2 or len(logits) != size):
            raise ValueError("Invalid reservoir checkpoint logits")
        self.capacity, self.num_seen = capacity, num_seen
        self._x = [] if x is None else [row.detach().cpu().clone() for row in x]
        self._y = [] if y is None else [row.detach().cpu().clone() for row in y]
        self._logits = None if logits is None else [row.detach().cpu().clone() for row in logits]
        self._reservoir_rng.set_state(state["reservoir_rng_state"].cpu())
        for name, generator in self._sample_rngs.items():
            generator.set_state(state["sample_rng_states"][name].cpu())
