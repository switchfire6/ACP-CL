"""A frozen, label-free input descriptor for an external drift signal.

This sensor does not read learner features or parameters. Its projection is
initialized once using a private RNG, never trained, and never recycled. Supply
preprocessed *unaugmented* inputs: stochastic augmentation would itself create
additional variation. Image descriptors retain coarse spatial information via
4x4 adaptive average pooling before a fixed nonlinear random projection.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


class FrozenInputSensor:
    """Return ``(batch, width)`` float32 descriptors on the input device.

    The first transform binds the sensor to either vectors of one dimension or
    images of one channel count. Image height/width may subsequently change.
    Fixed tanh random features are an engineering descriptor, not a pretrained
    semantic representation or a calibrated distribution-shift test. They avoid
    learner-reset-induced coordinate drift but can miss task-relevant changes.
    """

    _VERSION = 1
    _POOL_SIZE = 4

    def __init__(self, seed: int, width: int = 64):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        self.seed, self.width = seed, width
        self._input_kind: str | None = None
        self._input_dimension: int | None = None
        self._projection: Tensor | None = None
        self._bias: Tensor | None = None
        self._device_cache: dict[str, tuple[Tensor, Tensor]] = {}

    def _bind(self, kind: str, dimension: int) -> None:
        if self._input_kind is not None:
            if kind != self._input_kind or dimension != self._input_dimension:
                raise ValueError("sensor input kind/dimension changed after initialization")
            return
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._projection = torch.randn(dimension, self.width, generator=generator)
        self._projection.div_(math.sqrt(dimension))
        self._bias = torch.empty(self.width).uniform_(-0.5, 0.5, generator=generator)
        self._input_kind, self._input_dimension = kind, dimension

    @torch.no_grad()
    def transform(self, preprocessed_unaugmented_tensor: Tensor) -> Tensor:
        """Describe inputs without consuming global RNG or retaining gradients."""
        x = preprocessed_unaugmented_tensor
        if not isinstance(x, Tensor):
            raise TypeError("sensor input must be a torch.Tensor")
        if x.ndim not in (2, 4) or any(size < 1 for size in x.shape):
            raise ValueError("sensor input must be nonempty vectors (N,D) or images (N,C,H,W)")
        if not x.is_floating_point():
            raise ValueError("sensor expects preprocessed floating-point inputs")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("sensor input must be finite")
        values = x.detach().to(dtype=torch.float32)
        if values.ndim == 4:
            values = F.adaptive_avg_pool2d(values, (self._POOL_SIZE, self._POOL_SIZE))
            kind = "image"
        else:
            kind = "vector"
        values = values.flatten(1)
        self._bind(kind, values.shape[1])
        key = str(values.device)
        if key not in self._device_cache:
            self._device_cache[key] = (
                self._projection.to(device=values.device),
                self._bias.to(device=values.device),
            )
        projection, bias = self._device_cache[key]
        return torch.tanh(values @ projection + bias)

    def nbytes(self) -> int:
        """Canonical frozen-state bytes, excluding optional device-cache copies."""
        return sum(t.numel() * t.element_size()
                   for t in (self._projection, self._bias) if t is not None)

    def state_dict(self) -> dict[str, Any]:
        """Independent CPU checkpoint; no RNG draws occur after shape binding."""
        return {
            "version": self._VERSION,
            "seed": self.seed,
            "width": self.width,
            "pool_size": self._POOL_SIZE,
            "input_kind": self._input_kind,
            "input_dimension": self._input_dimension,
            "projection": None if self._projection is None else self._projection.clone(),
            "bias": None if self._bias is None else self._bias.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Atomically restore compatible state; reject schema/config mismatches."""
        keys = {"version", "seed", "width", "pool_size", "input_kind",
                "input_dimension", "projection", "bias"}
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("malformed frozen sensor checkpoint")
        for name, expected in (("version", self._VERSION), ("seed", self.seed),
                               ("width", self.width), ("pool_size", self._POOL_SIZE)):
            if type(state[name]) is not int or state[name] != expected:
                raise ValueError(f"sensor {name} does not match checkpoint")
        kind, dimension = state["input_kind"], state["input_dimension"]
        projection, bias = state["projection"], state["bias"]
        if kind is None:
            if any(value is not None for value in (dimension, projection, bias)):
                raise ValueError("uninitialized sensor checkpoint contains bound state")
            restored_projection = restored_bias = None
        else:
            if kind not in ("vector", "image") or type(dimension) is not int or dimension < 1:
                raise ValueError("invalid sensor input binding")
            if kind == "image" and dimension % self._POOL_SIZE**2:
                raise ValueError("image sensor binding must contain complete pooled channels")
            for name, tensor, shape in (("projection", projection, (dimension, self.width)),
                                         ("bias", bias, (self.width,))):
                if (not isinstance(tensor, Tensor) or tensor.dtype != torch.float32
                        or tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all())):
                    raise ValueError(f"invalid sensor {name}")
            restored_projection = projection.detach().to("cpu").clone()
            restored_bias = bias.detach().to("cpu").clone()
        self._input_kind, self._input_dimension = kind, dimension
        self._projection, self._bias = restored_projection, restored_bias
        self._device_cache = {}
