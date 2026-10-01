"""Bounded replay and a deliberately modest, observation-only recurrence proxy."""

from __future__ import annotations

import numpy as np

from .world import Experience, IMAGE_SHAPE, concatenate


METHODS = ("none", "uniform", "recent", "coverage", "recurrence", "relevance", "joint",
           "frozen_encoder", "frozen_late")


class ObservationHistory:
    """Fixed random temporal/image sketches and bounded historical frequencies.

    This is NOT an inferred true regime or a calibrated future-context model.
    It estimates repeat frequency of coarse observable patterns. No labels,
    actions, outcomes, clocks, task IDs, or evaluation information enter it.
    """

    def __init__(self, bits=8):
        rng = np.random.default_rng(730201)
        self.projection = rng.normal(size=(int(np.prod(IMAGE_SHAPE)), bits)).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0)
        self.counts = np.zeros(2**bits, dtype=np.int64)

    def codes(self, observations):
        x = observations.astype(np.float32) / 255
        appearance = x[:, -1].reshape(len(x), -1)
        appearance = appearance - appearance.mean(axis=1, keepdims=True)
        motion = np.diff(x, axis=1).reshape(len(x), -1)
        appearance /= np.maximum(np.linalg.norm(appearance, axis=1, keepdims=True), 1e-6)
        motion /= np.maximum(np.linalg.norm(motion, axis=1, keepdims=True), 1e-6)
        features = np.concatenate((appearance, motion), axis=1)
        bits = features @ self.projection >= 0
        return (bits.astype(np.int64) * (2 ** np.arange(bits.shape[1]))).sum(axis=1)

    def observe(self, observations):
        np.add.at(self.counts, self.codes(observations), 1)

    def probabilities(self):
        # A Dirichlet(1) predictive frequency. Recency/periodicity are not modeled.
        return (self.counts + 1) / (self.counts.sum() + len(self.counts))

    def nbytes(self):
        return self.projection.nbytes + self.counts.nbytes


class Memory:
    def __init__(self, method: str, capacity: int, seed: int):
        if method not in METHODS or type(capacity) is not int or capacity < 1:
            raise ValueError("unknown method or invalid positive capacity")
        self.method, self.capacity = method, capacity
        self.history = ObservationHistory()
        self.data = Experience(np.empty((0, *IMAGE_SHAPE), dtype=np.uint8),
                               np.empty(0, dtype=np.uint8), np.empty((0, 3), dtype=np.uint8))
        self.ids = np.empty(0, dtype=np.int64)
        self.seen = 0
        self.membership_rng = np.random.default_rng(seed ^ 202709)
        self.sampling_rng = np.random.default_rng(seed ^ 81173)
        self.evictions = 0
        self.selection_seconds = 0.0

    def sample(self, size):
        if not len(self.data) or self.method == "none":
            return None
        return self.data.take(self.sampling_rng.choice(len(self.data), size=size, replace=True))

    def candidates(self, current):
        return concatenate(self.data, current)

    def add(self, current: Experience, candidate_utility: np.ndarray):
        pool = self.candidates(current)
        if candidate_utility.shape != (len(pool),) or not np.all(np.isfinite(candidate_utility)):
            raise ValueError("one finite utility is required per candidate")
        if np.any((candidate_utility < 0) | (candidate_utility > 1)):
            raise ValueError("utility must lie in [0, 1]")
        self.history.observe(current.observations)
        ids = np.concatenate((self.ids, np.arange(self.seen, self.seen + len(current))))
        old_seen = self.seen
        self.seen += len(current)
        method = "uniform" if self.method in ("frozen_encoder", "frozen_late") else self.method
        if method == "none":
            return
        if len(pool) <= self.capacity:
            keep = np.arange(len(pool))
        elif method == "recent":
            keep = np.arange(len(pool) - self.capacity, len(pool))
        elif method == "uniform":
            keep = list(range(len(self.data)))
            for offset in range(len(current)):
                candidate = len(self.data) + offset
                if len(keep) < self.capacity:
                    keep.append(candidate)
                else:
                    slot = int(self.membership_rng.integers(old_seen + offset + 1))
                    if slot < self.capacity:
                        keep[slot] = candidate
            keep = np.asarray(keep)
        else:
            codes = self.history.codes(pool.observations)
            bins = len(self.history.counts)
            counts = np.bincount(codes, minlength=bins).astype(float)
            utility_sum = np.bincount(codes, weights=candidate_utility, minlength=bins)
            relevance = 0.1 + utility_sum / np.maximum(counts, 1)
            weights = np.ones(bins)
            if method in ("recurrence", "joint"):
                weights *= self.history.probabilities()
            if method in ("relevance", "joint"):
                weights *= relevance
            # Greedy discrete allocation for sum_b weight_b * log(1 + count_b).
            # Within a bin, identities are exchangeable; random eviction avoids
            # an additional age, target, or label preference.
            alive = np.ones(len(pool), dtype=bool)
            for _ in range(len(pool) - self.capacity):
                marginal = np.full(bins, np.inf)
                present = counts > 0
                marginal[present] = weights[present] * np.log1p(1 / counts[present])
                least = marginal.min()
                ties = np.flatnonzero(np.isclose(marginal, least, rtol=1e-12, atol=1e-15))
                bucket = int(self.membership_rng.choice(ties))
                slot = int(self.membership_rng.choice(np.flatnonzero(alive & (codes == bucket))))
                alive[slot] = False
                counts[bucket] -= 1
            keep = np.flatnonzero(alive)
        self.evictions += len(pool) - len(keep)
        self.data = pool.take(keep)
        self.ids = ids[keep]

    def nbytes(self):
        return (self.data.observations.nbytes + self.data.actions.nbytes + self.data.survival.nbytes
                + self.ids.nbytes + self.history.nbytes())
