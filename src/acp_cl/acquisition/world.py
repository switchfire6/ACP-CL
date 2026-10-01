"""New physical dependencies on previously irrelevant temporal observations.

The archived resource-conserving simulator is unchanged. Hidden law metadata
belongs to evaluator-owned cases; ordinary Experience remains images/action/y.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations

import numpy as np

from acp_cl.conditional.world import Condition, ConditionalWorld
from acp_cl.persistence.study import trial_seed
from acp_cl.persistence.world import ACTIONS, Cases, Experience, TransferWorld


DEPENDENCIES = ("directional_efficiency", "arrival_delay", "supply_timing")
ORDERS = tuple(permutations(range(3)))


@dataclass(frozen=True)
class Law:
    mode: int
    active: tuple[int, ...] = ()
    revised: bool = False
    noise: float = 0.

    def __post_init__(self):
        if self.mode not in (0, 1) or self.active != tuple(sorted(set(self.active))):
            raise ValueError("invalid mode or dependency selection")
        if any(k not in range(3) for k in self.active) or not 0 <= self.noise <= 1:
            raise ValueError("invalid dependency or noise")
        if self.revised and 0 not in self.active:
            raise ValueError("revision requires the directional-efficiency law")


@dataclass
class LawCases:
    observations: np.ndarray
    base: Cases
    factors: np.ndarray
    signals: np.ndarray
    law: Law


def signal_images(observations, signals):
    images = observations.copy()
    images[:, :, 8:10, :] = 0
    rows = np.arange(len(images))
    for index, column in enumerate((1, 6, 11)):
        # Two pulses exchange their order; every final frame is identical.
        first = column + signals[:, index]
        second = column + 1-signals[:, index]
        for row in (8, 9):
            images[rows, 0, row, first] = 255
            images[rows, 2, row, second] = 255
    return images


class AcquisitionWorld(TransferWorld):
    def dataset(self, law, size, seed):
        trials = ConditionalWorld().dataset(Condition(law.mode), size, seed)
        rng = np.random.default_rng(trial_seed(seed, "temporal_signals", 0))
        signals = rng.integers(0, 2, (size, 3), dtype=np.uint8)
        # Evaluation/audit panels contain the complete 8 x 8 factorial grid.
        # Small training/support batches sample signals independently of base factors.
        if size % 64 == 0:
            grid = np.asarray([[i >> j & 1 for j in range(3)] for i in range(8)], dtype=np.uint8)
            for factors in np.unique(trials.factors, axis=0):
                indices = np.flatnonzero((trials.factors == factors).all(axis=1))
                signals[indices] = np.tile(grid, (len(indices)//8, 1))[rng.permutation(len(indices))]
        return LawCases(signal_images(trials.cases.observations, signals), trials.cases,
                        trials.factors, signals, law)

    def physics(self, cases, actions):
        base, law = cases.base, cases.law
        efficiency, delay, supplies = base.efficiency.copy(), base.delay.copy(), base.supplies.copy()
        lossy, delayed = cases.factors[:, 1].astype(bool), cases.factors[:, 2].astype(bool)
        # Positive requests move A -> B, coded 0; negative requests coded 1.
        direction = (ACTIONS[np.asarray(actions)] < 0).astype(np.uint8)
        if 0 in law.active:
            favored = cases.signals[:, 0] ^ int(law.revised)
            efficiency[lossy] = np.where(direction[lossy] == favored[lossy], .95, .10)
        if 1 in law.active:
            aligned = direction == cases.signals[:, 1]
            delay[delayed] = np.where(aligned[delayed], 1, 4)
        if 2 in law.active:
            affected = lossy | delayed
            # Shift the original clipped supply realization to a new crossover.
            source = cases.factors[:, 0]*(1-2*law.mode)
            old_season = np.where(np.arange(12) < 6, 1., -1.)
            crossover = np.where(cases.signals[:, 2] == 0, 3, 9)
            new_season = np.where(np.arange(12)[None] < crossover[:, None], 1., -1.)
            change = self.imbalance*source[:, None, None]*(new_season-old_season)[..., None]*np.array([1., -1.])
            supplies[affected] = np.maximum(supplies[affected]+change[affected], 0.)
        return Cases(cases.observations, base.reserves, supplies, efficiency, delay)

    def simulate(self, cases, actions):
        return super().simulate(self.physics(cases, actions), actions)

    def counterfactuals(self, cases):
        return np.stack([self.simulate(cases, np.full(len(cases.observations), action)).survival
                         for action in range(5)], axis=1)

    def experience(self, law, size, seed):
        cases = self.dataset(law, size, seed)
        actions = np.random.default_rng(trial_seed(seed, "performed_action", 0)).integers(
            0, 5, size, dtype=np.uint8)
        measured = self.simulate(cases, actions).survival
        if law.noise:
            rng = np.random.default_rng(trial_seed(seed, "noisy_report", 0))
            corrupt = rng.random(size) < law.noise
            levels = rng.integers(0, 4, size)
            measured = measured.copy()
            measured[corrupt] = (np.arange(3)[None] < levels[corrupt, None]).astype(np.uint8)
        return Experience(cases.observations, actions, measured)


def order(seed):
    return ORDERS[seed % 6]


def stage_law(seed, stage):
    return Law(1-seed % 2, tuple(sorted(order(seed)[:stage])))


def final_law(seed, branch, noise):
    if branch == "return":
        return Law(seed % 2)  # Exact original A-world, not an unseen A/new-law combination.
    law = stage_law(seed, 3)
    if branch == "revision":
        return replace(law, revised=True)
    if branch == "noise":
        return replace(law, noise=noise)
    raise ValueError("unknown final branch")


def mask_valid(cases, branch="novel"):
    # Clean, immediate transport has identical dynamics throughout all three additions.
    if branch == "revision":
        return cases.factors[:, 1] == 0
    if branch == "return":
        return np.ones(len(cases.observations), dtype=bool)
    return (cases.factors[:, 1:] == 0).all(axis=1)


def mask_affected(cases, cue):
    if cue == 0:
        return cases.factors[:, 1] == 1
    if cue == 1:
        return cases.factors[:, 2] == 1
    if cue == 2:
        return (cases.factors[:, 1:] == 1).any(axis=1)
    return np.ones(len(cases.observations), dtype=bool)


def batch(world, law, seed, channel, index, size):
    return world.experience(law, size, trial_seed(seed, "acquisition_"+channel, index))
