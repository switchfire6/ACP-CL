"""Step D1b Amendment 1: MB-count, directed count-based exploration on D1's MB learner.

act = argmax_a [ V_k(a) + c / sqrt(1 + n_a) ], where V_k is the Thompson-drawn member's planned joint-survival value
(computed exactly as D1's ``MBLearner.act``: same member draw, same planning draws, same operations) and n_a is the
number of episodes CURRENTLY IN THE REPLAY (reservoir) whose action was a. Greedy probes, training and everything
else are D1's. With c = 0 the arm is bit-identical to MB (V + 0 == V in floating point).
"""

from __future__ import annotations

import numpy as np
import torch

from .learners import ACTIONS, ENTITIES, T, MBLearner, check_decision


class MBCount(MBLearner):
    family = "MB"
    remedy = "count"

    def __init__(self, hyper, rngs, c, track_ema=True):
        super().__init__(hyper, rngs, track_ema)
        self.c = float(c)
        self.last_base_action = -1
        self.last_counts = np.zeros(ACTIONS, np.int64)

    def reservoir_counts(self):
        return np.bincount(self.replay.data["action"][: self.replay.size].astype(np.int64), minlength=ACTIONS)

    def act(self, decision):
        check_decision(decision)
        k = int(self.rngs["thompson"].integers(0, self.h.members))          # as MBLearner.act
        self.last_thompson = k
        eps = torch.from_numpy(self.rngs["plan"].standard_normal((self.h.plan_samples, T, ENTITIES)).astype(np.float32))
        with torch.no_grad():
            y = self.supply_draws(eps, self.ema[1])[None]
            logs = self._values(self.ema[0], k, decision.reserves[None], decision.cue[None], y)
            value = torch.exp(logs.sum(-1)).mean(-1)[0]                                          # (13,)
        self.macs["act"] += self._plan_macs(1, 1)
        counts = self.reservoir_counts()
        self.last_counts = counts
        bonus = torch.from_numpy((self.c / np.sqrt(1.0 + counts)).astype(np.float32))
        self.last_base_action = int(torch.argmax(value))
        return int(torch.argmax(value + bonus))
