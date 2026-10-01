"""Mix frozen forecasts using only earlier performed-action terminal feedback."""

from collections import deque
import hashlib
import json
import math

import numpy as np


def _predictions(core, full):
    arrays = tuple(np.asarray(value) for value in (core, full))
    if any(value.dtype.kind not in "buif" for value in arrays):
        raise ValueError("Forecasts must be real numeric arrays")
    if (arrays[0].ndim != 3 or arrays[0].shape[1:] != (5, 3)
            or arrays[0].shape != arrays[1].shape or not len(arrays[0])):
        raise ValueError("Forecasts must have the same nonempty [N, 5, 3] shape")
    tolerance = np.finfo(np.float32).eps
    if any(not np.isfinite(value).all() or (value < -tolerance).any()
           or (value > 1 + tolerance).any() for value in arrays):
        raise ValueError("Forecasts must be finite probabilities")
    return arrays


def mix_predictions(core, full, weight):
    """Return (1 - weight) * core + weight * full, rounding once to float32."""
    core, full = _predictions(core, full)
    weight = float(weight)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Mixture weight must lie in [0, 1]")
    return ((1 - weight) * core.astype(np.float64)
            + weight * full.astype(np.float64)).astype(np.float32)


class LaggedMixer:
    """A fixed-window likelihood mixture with an equal prior and beta=1.

    Call predict before observing a packet's outcomes. Pass those same original
    component forecasts to observe; rescoring them with updated support would
    leak the packet's feedback. No task labels or automatic reset are accepted.
    """

    def __init__(self, capacity=32, clip=1e-6):
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or capacity < 1:
            raise ValueError("Capacity must be a positive integer")
        if not math.isfinite(clip) or not 0 < clip < .5:
            raise ValueError("Clip must lie strictly between 0 and .5")
        self.capacity = int(capacity)
        self.clip = float(clip)
        self._scores = deque(maxlen=self.capacity)
        self.samples_seen = 0
        self.packets_seen = 0

    def weight(self):
        """Return the full-output weight from completed feedback only."""
        score = math.fsum(self._scores)
        if score >= 0:
            return 1 / (1 + math.exp(-score))
        exponential = math.exp(score)
        return exponential / (1 + exponential)

    def predict(self, core, full):
        return mix_predictions(core, full, self.weight())

    def observe(self, core, full, actions, outcomes):
        """Append one completed packet's original forecasts and observed labels."""
        core, full = _predictions(core, full)
        actions, outcomes = np.asarray(actions), np.asarray(outcomes)
        if (actions.shape != (len(core),) or actions.dtype.kind not in "iu"
                or (actions < 0).any() or (actions >= 5).any()):
            raise ValueError("Actions must be an integer vector with values in [0, 4]")
        if (outcomes.shape != (len(core), 3) or outcomes.dtype.kind not in "buif"
                or not np.isfinite(outcomes).all()
                or not ((outcomes == 0) | (outcomes == 1)).all()
                or (outcomes[:, 1:] > outcomes[:, :-1]).any()):
            raise ValueError("Outcomes must be binary monotone [N, 3] survival labels")
        index = np.arange(len(core))
        terminal = outcomes[:, -1].astype(np.float64)

        def log_likelihood(probabilities):
            selected = probabilities[index, actions, -1].astype(np.float64)
            likelihood = np.where(terminal, selected, 1 - selected)
            return np.log(np.clip(likelihood, self.clip, 1 - self.clip))

        scores = log_likelihood(full) - log_likelihood(core)
        self._scores.extend(float(value) for value in scores)
        self.samples_seen += len(core)
        self.packets_seen += 1

    def state(self):
        """Return a detached, JSON-serializable state and feedback-cost record."""
        return {"capacity": self.capacity, "clip": self.clip, "scores": list(self._scores),
                "samples_seen": self.samples_seen, "packets_seen": self.packets_seen}

    def signature(self):
        encoded = json.dumps(self.state(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
