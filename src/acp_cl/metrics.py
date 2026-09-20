"""Explicit metric conventions for class-incremental experiment matrices."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


def _array(value) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def continual_metrics(matrix) -> dict[str, float]:
    """Summarize A[i,j] = task-j accuracy after experience i.

    Only the lower triangle is eligible; unavailable entries are NaN.  Final
    accuracy averages task accuracies equally.  Average incremental accuracy
    averages seen-task row means equally.  BWT compares the final row against
    the acquisition diagonal for prior tasks.  Forgetting averages each prior
    task's best *post-acquisition* accuracy (including the final measurement)
    minus its final accuracy, so it is nonnegative.
    """
    a = _array(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.shape[0]:
        raise ValueError("matrix must be a nonempty square accuracy matrix")
    if np.isinf(a).any():
        raise ValueError("matrix may contain NaN for missing values, but not infinity")
    tasks = a.shape[0]
    row_means = []
    for i in range(tasks):
        observed = a[i, : i + 1]
        observed = observed[np.isfinite(observed)]
        if observed.size:
            row_means.append(float(observed.mean()))
    final = a[-1, np.isfinite(a[-1])]
    bwt, forgetting = [], []
    for j in range(tasks - 1):
        if not np.isfinite(a[-1, j]):
            continue
        if np.isfinite(a[j, j]):
            bwt.append(float(a[-1, j] - a[j, j]))
        observed = a[j:, j]
        observed = observed[np.isfinite(observed)]
        if observed.size:
            forgetting.append(float(observed.max() - a[-1, j]))
    return {
        "final_accuracy": _mean_or_nan(final),
        "average_incremental_accuracy": _mean_or_nan(row_means),
        "backward_transfer": 0.0 if tasks == 1 else _mean_or_nan(bwt),
        "forgetting": 0.0 if tasks == 1 else _mean_or_nan(forgetting),
    }


def normalized_auc(points: list[tuple[float, float]], fraction: float = 1.0) -> float:
    """Mean curve height through ``fraction`` of the final x horizon.

    Points must start at x=0 and have strictly increasing x.  An exact cutoff is
    linearly interpolated, including when it falls between logged observations.
    A zero horizon (or fraction=0) returns the baseline value.
    """
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    a = _array(points)
    if a.ndim != 2 or a.shape[1] != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError("points must be a nonempty sequence of finite (x, y) pairs")
    if a[0, 0] != 0:
        raise ValueError("A baseline observation at x=0 is required")
    if len(a) > 1 and (np.diff(a[:, 0]) <= 0).any():
        raise ValueError("x coordinates must be strictly increasing")
    horizon = float(a[-1, 0] * fraction)
    if horizon == 0:
        return float(a[0, 1])
    interior = a[a[:, 0] < horizon]
    end_y = float(np.interp(horizon, a[:, 0], a[:, 1]))
    x = np.append(interior[:, 0], horizon)
    y = np.append(interior[:, 1], end_y)
    area = np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2)
    return float(area / horizon)


def representation_metrics(
    features: Tensor, dormancy_threshold: float = 0.01
) -> dict[str, float]:
    """Centered entropy/stable rank and uncentered relative activity dormancy.

    Entropy effective rank uses normalized singular values (not squared values).
    Stable rank is ||X||_F^2 / ||X||_2^2.  Both ranks are bounded by min(N-1,C).
    A unit is dormant when its mean absolute activation is <= the threshold
    times the population's mean activity.  Trailing dimensions are flattened.
    """
    if features.ndim < 2 or features.shape[0] == 0:
        raise ValueError("features must have at least one sample and one feature axis")
    if not math.isfinite(dormancy_threshold) or dormancy_threshold < 0:
        raise ValueError("dormancy_threshold must be finite and nonnegative")
    x = features.detach().flatten(1).to(device="cpu", dtype=torch.float64)
    if x.shape[1] == 0 or not torch.isfinite(x).all():
        raise ValueError("features must be nonempty and finite")
    activity = x.abs().mean(dim=0)
    dormant = (activity <= dormancy_threshold * activity.mean()).double().mean()
    centered = x - x.mean(dim=0, keepdim=True)
    max_rank = min(x.shape[0] - 1, x.shape[1])
    if max_rank <= 0 or not torch.count_nonzero(centered):
        effective_rank = stable_rank = 0.0
    else:
        singular = torch.linalg.svdvals(centered)[:max_rank]
        tolerance = torch.finfo(singular.dtype).eps * max(centered.shape) * singular[0]
        singular = singular[singular > tolerance]
        if singular.numel() == 0:
            effective_rank = stable_rank = 0.0
        else:
            probability = singular / singular.sum()
            effective_rank = float(torch.exp(-(probability * probability.log()).sum()))
            stable_rank = float(singular.square().sum() / singular[0].square())
    return {
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "dormant_fraction": float(dormant),
    }


def paired_bootstrap(
    a,
    b,
    seed: int = 0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Percentile bootstrap interval for mean(a-b), resampling seed pairs."""
    a, b = _array(a), _array(b)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("a and b must be nonempty one-dimensional paired samples")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired samples must be finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if isinstance(n_resamples, bool) or int(n_resamples) != n_resamples or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    difference = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_resamples), dtype=np.float64)
    for start in range(0, int(n_resamples), 4096):
        stop = min(start + 4096, int(n_resamples))
        indices = rng.integers(0, len(a), size=(stop - start, len(a)))
        means[start:stop] = difference[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "n_pairs": len(a),
    }
