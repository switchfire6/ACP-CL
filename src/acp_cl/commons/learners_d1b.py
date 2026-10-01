"""Step D1b exploration remedies (docs/commons_d1b_protocol.md) as subclasses of D1's MB learner.

``acp_cl.commons.learners`` (D1) is imported, never edited. The MB arm of D1b IS D1's ``MBLearner``;
the remedies change only what is declared:

* ``MBPrior`` (randomised prior functions): each failure-model member k adds beta * p_k(x) to its
  output logits, where p_k is a fixed, untrained network of the same architecture (55 -> H -> H -> 36,
  tanh), initialised like the members (U(+-1/sqrt(fan_in))) from its own generator
  (seed rule: ``seed_for("commons_d1b_prior", seed, context)``). The prior enters training (the loss is
  computed on f + beta * p), acting, probes and the hazard NLL; it is never trained and never EMA'd.
* ``MBUCB``: acts on mean + kappa * SD (population SD over the 5 members) of each action's predicted joint
  survival; each member is evaluated (EMA weights) on the same M planning draws. No Thompson member is drawn.
* ``MBEps``: the Thompson action is always computed first (so every base random stream advances exactly
  as in MB); then, with probability epsilon (its own generator, seed rule
  ``seed_for("commons_d1b_eps", seed, context)``), a uniformly random action out of 13 replaces it.
* MB-wide uses D1's ``MBLearner`` unchanged with a failure-model width of 256 (a Hyper field).

Training is D1's in every arm except that MB-prior's loss includes its prior (by construction of the method).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .learners import (ACTIONS, ENTITIES, SUPPLY_SCALE, T, Ensemble, MBLearner, check_decision, clip_per_member,
                       clip_whole, tensor_hash)


def member_values(learner, reserves, cue, eps, chunk=16):
    """Per-member (EMA) predicted joint survival (K, n, 13) on decision states with fixed draws eps (n, M, 12, 3).
    Works for D1's MBLearner and every subclass (uses the learner's own _values, including priors)."""
    n = len(reserves)
    out = np.zeros((learner.h.members, n, ACTIONS), np.float64)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            y = learner.supply_draws(torch.from_numpy(np.ascontiguousarray(eps[lo:hi], dtype=np.float32)), learner.ema[1])
            for k in range(learner.h.members):
                out[k, lo:hi] = torch.exp(learner._values(learner.ema[0], k, reserves[lo:hi], cue[lo:hi], y).sum(-1)).mean(-1).numpy()
    learner.macs["probe"] += learner._plan_macs(n, learner.h.members)
    return out


class MBPrior(MBLearner):
    family = "MB"
    remedy = "prior"

    def __init__(self, hyper, rngs, beta, prior_rng, track_ema=True):
        super().__init__(hyper, rngs, track_ema)
        self.beta = float(beta)
        prior = Ensemble(hyper.members, [self.n_in, hyper.hidden, hyper.hidden, ENTITIES * T], prior_rng)
        self.prior = [p.detach().clone().requires_grad_(False) for p in prior.params]
        self.prior_hash = tensor_hash(self.prior)

    def _raw_logits(self, params, k, reserves, cue, y):
        w1, b1, w2, b2, w3, b3 = params
        s = torch.from_numpy(self.static_features(reserves, cue))
        base = s @ w1[k, :6] + b1[k]
        act = w1[k, 6:6 + ACTIONS]
        ysup = y @ w1[k, 6 + ACTIONS:]
        pre = base[:, None, None, :] + act[None, :, None, :] + ysup[:, None, :, :]
        h2 = torch.tanh(torch.tanh(pre) @ w2[k] + b2[k])
        return (h2 @ w3[k] + b3[k]).reshape(*h2.shape[:3], ENTITIES, T)

    def _values(self, params, k, reserves, cue, y):
        logits = self._raw_logits(params, k, reserves, cue, y) + self.beta * self._raw_logits(self.prior, k, reserves, cue, y)
        return F.logsigmoid(-logits).sum(-1)

    def _plan_macs(self, states, members):
        return 2 * super()._plan_macs(states, members)

    def _train_step(self):
        h = self.h
        d = self.replay.data
        idx, weight = self.replay.member_batches(self.rngs["batch"], h.minibatch)
        x = np.concatenate([self.static_features(d["reserves"][idx], d["cue"][idx]),
                            np.eye(ACTIONS, dtype=np.float32)[d["action"][idx]],
                            d["y_supply"][idx].reshape(*idx.shape, -1) / SUPPLY_SCALE], -1)
        fs = torch.from_numpy(d["fail_step"][idx].astype(np.int64))
        w = torch.from_numpy(weight)
        xt = torch.from_numpy(x)
        self.opt.zero_grad(set_to_none=False)
        with torch.no_grad():
            prior = Ensemble.forward(self.prior, xt)
        logits = (Ensemble.forward(self.net.params, xt) + self.beta * prior).reshape(*idx.shape, ENTITIES, T)
        ll = self.failure_loglik(logits, fs)
        loss = (-ll.mean(1) * w).sum()
        loss.backward()
        clip_per_member(self.net.params, h.members, h.grad_clip)
        self.opt.step()
        tidx = self.rngs["table_batch"].integers(0, self.replay.size, h.minibatch)
        ys = torch.from_numpy(d["y_supply"][tidx])
        yp = torch.from_numpy(d["y_probe"][tidx])
        self.opt_tables.zero_grad(set_to_none=False)
        tl = self.gauss_nll(self.supply, ys) + self.gauss_nll(self.probe, yp)
        tl.backward()
        clip_whole(self.supply, h.grad_clip)
        clip_whole(self.probe, h.grad_clip)
        self.opt_tables.step()
        self._update_ema([self.net.params, self.supply, self.probe])
        self.macs["train"] += (3 + 1) * self.net.macs_per_row() * h.minibatch * h.members + 6 * 2 * T * ENTITIES * h.minibatch

    def hazard_nll(self, reserves, cue, actions, y_supply, fail_step):
        x = np.concatenate([self.static_features(reserves, cue), np.eye(ACTIONS, dtype=np.float32)[actions],
                            np.asarray(y_supply, np.float32).reshape(len(actions), -1) / SUPPLY_SCALE], -1)
        xt = torch.from_numpy(x)[None].expand(self.h.members, -1, -1)
        with torch.no_grad():
            logits = (Ensemble.forward(self.ema[0], xt) + self.beta * Ensemble.forward(self.prior, xt)).reshape(
                self.h.members, len(actions), ENTITIES, T)
            fs = torch.from_numpy(np.asarray(fail_step, np.int64))
            before = (self.steps_t < fs[..., None]).float()
            at = (self.steps_t == fs[..., None]).float()
            per = (F.logsigmoid(-logits) * before + F.logsigmoid(logits) * at).sum(-1)
            mix = torch.logsumexp(per, 0) - np.log(self.h.members)
        self.macs["probe"] += 2 * self.net.macs_per_row() * len(actions) * self.h.members
        return float(-mix.mean())

    def hashes(self):
        out = super().hashes()
        out["prior"] = tensor_hash(self.prior)
        return out

    def state(self):
        st = super().state()
        st["prior_hash"] = tensor_hash(self.prior)
        return st

    def load(self, state):
        if state["prior_hash"] != self.prior_hash:
            raise ValueError("prior network differs from the checkpoint")
        super().load(state)


class MBUCB(MBLearner):
    family = "MB"
    remedy = "ucb"

    def __init__(self, hyper, rngs, kappa, track_ema=True):
        super().__init__(hyper, rngs, track_ema)
        self.kappa = float(kappa)
        self.last_sd = float("nan")

    def act(self, decision):
        check_decision(decision)
        eps = torch.from_numpy(self.rngs["plan"].standard_normal((self.h.plan_samples, T, ENTITIES)).astype(np.float32))
        with torch.no_grad():
            y = self.supply_draws(eps, self.ema[1])[None]
            vals = torch.stack([torch.exp(self._values(self.ema[0], k, decision.reserves[None], decision.cue[None], y)
                                          .sum(-1)).mean(-1)[0] for k in range(self.h.members)])   # (K, 13)
            mean, sd = vals.mean(0), vals.std(0, unbiased=False)
            score = mean + self.kappa * sd
        self.macs["act"] += self._plan_macs(1, self.h.members)
        self.last_thompson = -1
        self.last_sd = float(sd.mean())
        return int(torch.argmax(score))


class MBEps(MBLearner):
    family = "MB"
    remedy = "eps"

    def __init__(self, hyper, rngs, epsilon, eps_rng, track_ema=True):
        super().__init__(hyper, rngs, track_ema)
        self.epsilon = float(epsilon)
        self.eps_rng = eps_rng
        self.last_explore = False
        self.last_base_action = -1

    def act(self, decision):
        a = super().act(decision)          # Thompson action: consumes the thompson and plan streams exactly as MB
        self.last_base_action = a
        self.last_explore = bool(self.eps_rng.random() < self.epsilon)
        if self.last_explore:
            a = int(self.eps_rng.integers(0, ACTIONS))
        return a

    def rng_state(self):
        st = super().rng_state()
        st["epsilon"] = self.eps_rng.bit_generator.state
        return st

    def load_rng_state(self, state):
        super().load_rng_state({k: v for k, v in state.items() if k != "epsilon"})
        self.eps_rng.bit_generator.state = state["epsilon"]

