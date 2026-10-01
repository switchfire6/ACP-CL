"""Paired return, new-dependency, and noisy-feedback branches."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from acp_cl.conditional.world import Condition, ConditionalWorld
from acp_cl.persistence.study import trial_seed
from acp_cl.persistence.world import Experience


BRANCHES = ("return", "novel", "noise")


@dataclass(frozen=True)
class Environment(Condition):
    feedback_noise: float = 0.


class ContextWorld(ConditionalWorld):
    def experience(self, condition, size, seed):
        data, modes = super().experience(condition, size, seed)
        noise = getattr(condition, "feedback_noise", 0.)
        if not 0 <= noise <= 1:
            raise ValueError("invalid measurement noise")
        if noise:
            rng = np.random.default_rng(trial_seed(seed, "feedback_corruption", 0))
            corrupt = rng.random(size) < noise
            # Replace a sensor report with an independent valid survival curve.
            # Underlying supplies and true survival remain unchanged.
            level = rng.integers(0, 4, size)
            measured = data.survival.copy()
            measured[corrupt] = (np.arange(3)[None] < level[corrupt, None]).astype(np.uint8)
            data = Experience(data.observations, data.actions, measured)
        return data, modes


def prefix(seed, gap, settings):
    if gap not in ("short", "long"):
        raise ValueError("invalid absence length")
    a, b = seed % 2, 1 - seed % 2
    modes = [a, b, a, b] + ([a, b, b, b] if gap == "long" else [b, b, a, b])
    counts, blocks = Counter(), []
    for mode in modes:
        occurrence = counts[mode]
        counts[mode] += 1
        # Length depends on mode occurrence, so both orders have identical data
        # multisets and exposure counts. Durations vary without boundary cues.
        choice = trial_seed(seed, "duration", [mode, occurrence]) % 2
        size = settings["block_size"] * (3 if choice == 0 else 5) // 4
        blocks.append(dict(mode=mode, occurrence=occurrence, size=size))
    return blocks


def challenge(seed, branch, settings):
    a, b = seed % 2, 1 - seed % 2
    if branch == "return":
        return Environment(a)
    if branch == "novel":
        return Environment(b, gated=True)
    if branch == "noise":
        return Environment(b, feedback_noise=settings["feedback_noise"])
    raise ValueError("invalid challenge")


def prefix_batch(world, seed, block, batch, size):
    return world.experience(Environment(block["mode"]), size,
        trial_seed(seed, "matched_prefix", [block["mode"], block["occurrence"], batch]))


def challenge_batch(world, seed, branch, settings, batch):
    return world.experience(challenge(seed, branch, settings), settings["batch_size"],
                            trial_seed(seed, "matched_challenge", batch))
