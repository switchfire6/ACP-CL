"""Full factorial observations with hidden, changing consequence mappings.

The conserving physics is imported unchanged from the archived persistence
experiment. Only the distribution of conditions and the stream are different.
Latent modes remain in evaluator-owned TrialSet objects, never Experience.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acp_cl.persistence.study import trial_seed
from acp_cl.persistence.world import Cases, Experience, Regime, TransferWorld


GRID = tuple((source, lossy, delayed) for source in (-1, 1)
             for lossy in (0, 1) for delayed in (0, 1))


@dataclass(frozen=True)
class Condition:
    mode: int  # 0: original mapping; 1: reversed mapping; -1: unpredictable per trial.
    gated: bool = False


@dataclass
class TrialSet:
    cases: Cases
    modes: np.ndarray
    factors: np.ndarray


class ConditionalWorld(TransferWorld):
    def dataset(self, condition: Condition, size: int, seed: int) -> TrialSet:
        if size < 8 or size % 8:
            raise ValueError("size must be a positive multiple of the eight-cell factorial grid")
        if condition.mode not in (-1, 0, 1):
            raise ValueError("invalid hidden mode")
        parts = []
        for index, (source, lossy, delayed) in enumerate(GRID):
            regime = Regime(str(index), source, lossy, delayed, gated=condition.gated)
            parts.append(self.cases(regime, size // 8, trial_seed(seed, "grid", index)))
        cases = Cases(*(np.concatenate([getattr(p, name) for p in parts])
                        for name in ("observations", "reserves", "supplies", "efficiency", "delay")))
        # Swapping supply columns reverses the physical mapping without altering
        # observations, action opportunities, or marginal supply-noise laws.
        rng = np.random.default_rng(trial_seed(seed, "modes", 0))
        modes = (rng.integers(0, 2, size, dtype=np.uint8) if condition.mode == -1
                 else np.full(size, condition.mode, dtype=np.uint8))
        flipped = modes == 1
        cases.supplies[flipped] = cases.supplies[flipped, :, ::-1]
        factors = np.repeat(np.asarray(GRID), size // 8, axis=0)
        order = np.random.default_rng(trial_seed(seed, "order", 0)).permutation(size)
        cases = Cases(*(getattr(cases, name)[order] for name in
                        ("observations", "reserves", "supplies", "efficiency", "delay")))
        return TrialSet(cases, modes[order], factors[order])

    def experience(self, condition: Condition, size: int, seed: int):
        trials = self.dataset(condition, size, seed)
        actions = np.random.default_rng(trial_seed(seed, "action", 0)).integers(
            0, 5, size, dtype=np.uint8)
        outcome = self.simulate(trials.cases, actions)
        data = Experience(trials.cases.observations, actions, outcome.survival)
        return data, trials.modes


def schedule(seed: int, name: str):
    first = seed % 2
    if name == "recurring":
        modes = [first, 1 - first, first, 1 - first, first,
                 1 - first, 1 - first, 1 - first, first, first, 1 - first, first]
    elif name == "stable":
        modes = [first] * 12
    elif name == "unpredictable":
        modes = [-1] * 12
    else:
        raise ValueError("unknown schedule")
    return [Condition(mode, gated=i >= 9) for i, mode in enumerate(modes)]


def training_batch(world, seed, stream, block, batch, size):
    condition = schedule(seed, stream)[block]
    return world.experience(condition, size, trial_seed(seed, "train", [stream, block, batch]))
