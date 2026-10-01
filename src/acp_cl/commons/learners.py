"""Step D1 learners for the commons world (docs/commons_d1_protocol.md).

This module is deliberately self-contained: it imports nothing from the world
or the references, so a learner can only ever see what the environment hands
it through the two legal observation types below (checked on every call).

* ``Decision``: decision-time reserves (3, exact) and the 3-bit cue.
* ``Outcome``: post-episode supply sensors (12 x 3), probe sensors (12 x 3)
  and each entity's failure step (1..12, or 13 = alive after step 12) for the
  action that was taken. No context, no latent supplies, no counterfactuals.

Arms
* ``MFLearner``: an ensemble of 5 MLPs (6 -> 128 -> 128 -> 13 + 117, tanh).
  Inputs reserves / 18 and the cue bits. Outputs 13 joint-survival logits and,
  per action, 9 auxiliary logits (entity 1..3 x alive after steps 4, 8, 12).
  Only the chosen action's joint and auxiliary logits are trained (BCE; the 9
  auxiliary BCEs are averaged and weighted by ``aux_weight``). Each member
  trains on its Bernoulli(.5) bootstrap mask of the replay.
* ``MBLearner``: a per-entity, per-step Gaussian table (mean, log-variance) of
  the supply sensors, the same form for the probe sensors (trained, never used
  for planning in D1), and a failure ensemble of 5 MLPs (55 -> 128 -> 128 ->
  36, tanh) giving per-entity discrete-time hazards over 12 steps from
  reserves / 18, the cue, the action one-hot and the 36 sensed supplies / 2,
  trained by the likelihood of the observed failure steps. The failure members
  use the same bootstrap masks as MF; the tables train on the whole replay.
  Planning draws M supply-sensor sequences from the (EMA) supply table with
  common random numbers across the 13 actions, predicts each entity's survival
  to step 12 with one Thompson-drawn (EMA) member, multiplies over entities
  and averages over draws. Greedy probes average the value over all members.

Both families: after each episode, U minibatch updates of 64 episodes (drawn
uniformly with replacement from each member's masked replay), Adam(lr),
gradient-norm clip 5 applied per ensemble member (and separately to each
table), and a weight EMA with decay d per optimiser step, initialised at the
initial weights, used only for acting and probes (never for training).

Replay: a uniform reservoir of ``capacity`` episodes (Algorithm R), or an
unbounded store that keeps every episode (privileged reference).

Randomness: every stochastic choice uses its own numpy Generator (init,
bootstrap masks, Thompson member, minibatches, table minibatches, reservoir,
planning draws), so the bounded and unbounded arms share initialisation,
masks, Thompson draws and planning noise, and the EMA can be shown not to
interfere with training (training does not consume any acting randomness).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

T = 12
ENTITIES = 3
ACTIONS = 13
ALIVE = T + 1
AUX_HORIZONS = (4, 8, 12)
RESERVE_SCALE = 18.0
SUPPLY_SCALE = 2.0
RNG_STREAMS = ("init", "mask", "thompson", "batch", "table_batch", "reservoir", "plan")


# ------------------------------------------------------------------ legal observations

class IllegalObservation(AssertionError):
    """Raised when a learner is handed anything but a legal observation."""


@dataclass(frozen=True)
class Decision:
    reserves: np.ndarray   # (3,) float64, exact decision-time reserves
    cue: np.ndarray        # (3,) uint8, cue bits


@dataclass(frozen=True)
class Outcome:
    y_supply: np.ndarray   # (12, 3) float64 supply sensors
    y_probe: np.ndarray    # (12, 3) float64 probe sensors
    fail_step: np.ndarray  # (3,) int8 failure step 1..12, or 13 = alive after step 12


DECISION_FIELDS = ("reserves", "cue")
OUTCOME_FIELDS = ("y_supply", "y_probe", "fail_step")


def _frozen(array, dtype):
    out = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def make_decision(reserves, cue):
    return check_decision(Decision(_frozen(reserves, np.float64), _frozen(cue, np.uint8)))


def make_outcome(y_supply, y_probe, fail_step):
    return check_outcome(Outcome(_frozen(y_supply, np.float64), _frozen(y_probe, np.float64),
                                 _frozen(fail_step, np.int8)))


def check_decision(obs):
    if type(obs) is not Decision or tuple(f.name for f in fields(obs)) != DECISION_FIELDS:
        raise IllegalObservation(f"illegal decision observation: {type(obs)!r}")
    if set(vars(obs)) != set(DECISION_FIELDS):
        raise IllegalObservation("decision carries extra attributes")
    r, c = obs.reserves, obs.cue
    if not (isinstance(r, np.ndarray) and r.dtype == np.float64 and r.shape == (ENTITIES,)):
        raise IllegalObservation("reserves must be float64 (3,)")
    if not (isinstance(c, np.ndarray) and c.dtype == np.uint8 and c.shape == (3,)):
        raise IllegalObservation("cue must be uint8 (3,)")
    if r.flags.writeable or c.flags.writeable:
        raise IllegalObservation("observations must be read-only copies")
    if not (np.all(np.isfinite(r)) and np.all(r >= 0.0) and np.all(r <= RESERVE_SCALE) and np.all(c <= 1)):
        raise IllegalObservation("decision values out of range")
    return obs


def check_outcome(obs):
    if type(obs) is not Outcome or tuple(f.name for f in fields(obs)) != OUTCOME_FIELDS:
        raise IllegalObservation(f"illegal outcome observation: {type(obs)!r}")
    if set(vars(obs)) != set(OUTCOME_FIELDS):
        raise IllegalObservation("outcome carries extra attributes")
    ys, yp, fs = obs.y_supply, obs.y_probe, obs.fail_step
    for name, arr in (("y_supply", ys), ("y_probe", yp)):
        if not (isinstance(arr, np.ndarray) and arr.dtype == np.float64 and arr.shape == (T, ENTITIES)
                and np.all(np.isfinite(arr))):
            raise IllegalObservation(f"{name} must be finite float64 (12, 3)")
    if not (isinstance(fs, np.ndarray) and fs.dtype == np.int8 and fs.shape == (ENTITIES,)
            and np.all(fs >= 1) and np.all(fs <= ALIVE)):
        raise IllegalObservation("fail_step must be int8 (3,) in 1..13")
    if ys.flags.writeable or yp.flags.writeable or fs.flags.writeable:
        raise IllegalObservation("observations must be read-only copies")
    return obs


# ------------------------------------------------------------------ replay

class Replay:
    """Uniform reservoir (Algorithm R) of ``capacity`` episodes, or an unbounded store."""

    def __init__(self, capacity, members, rng, bounded=True):
        self.capacity, self.members, self.rng, self.bounded = int(capacity), int(members), rng, bool(bounded)
        c = self.capacity
        self.data = dict(reserves=np.zeros((c, ENTITIES), np.float32), cue=np.zeros((c, 3), np.uint8),
                         action=np.zeros(c, np.int16), y_supply=np.zeros((c, T, ENTITIES), np.float32),
                         y_probe=np.zeros((c, T, ENTITIES), np.float32), fail_step=np.zeros((c, ENTITIES), np.int8),
                         mask=np.zeros((c, members), bool), episode=np.full(c, -1, np.int32))
        self.size = 0
        self.seen = 0
        self.max_size = 0
        self.evictions = 0

    def record_bytes(self):
        return int(sum(v[0].nbytes for v in self.data.values()))

    def add(self, decision, action, outcome, mask):
        k = self.seen
        if self.size < self.capacity:
            slot = self.size
            self.size += 1
        elif self.bounded:
            j = int(self.rng.integers(0, k + 1))
            slot = j if j < self.capacity else None
            if slot is not None:
                self.evictions += 1
        else:
            raise RuntimeError("unbounded replay allocated too small")
        self.seen += 1
        if slot is not None:
            d = self.data
            d["reserves"][slot] = decision.reserves
            d["cue"][slot] = decision.cue
            d["action"][slot] = action
            d["y_supply"][slot] = outcome.y_supply
            d["y_probe"][slot] = outcome.y_probe
            d["fail_step"][slot] = outcome.fail_step
            d["mask"][slot] = mask
            d["episode"][slot] = k
        if self.size > self.capacity:
            raise AssertionError("replay exceeded its capacity")
        self.max_size = max(self.max_size, self.size)
        return slot

    def member_batches(self, rng, batch):
        """(K, batch) indices, each member uniform with replacement over its masked episodes; weights (K,)."""
        idx = np.zeros((self.members, batch), np.int64)
        weight = np.zeros(self.members, np.float32)
        mask = self.data["mask"][: self.size]
        for k in range(self.members):
            eligible = np.flatnonzero(mask[:, k])
            if len(eligible):
                idx[k] = eligible[rng.integers(0, len(eligible), batch)]
                weight[k] = 1.0
        return idx, weight

    def state(self):
        return dict(data={k: v[: self.size].copy() for k, v in self.data.items()}, size=self.size, seen=self.seen,
                    max_size=self.max_size, evictions=self.evictions, rng=self.rng.bit_generator.state)

    def load(self, state):
        for k, v in state["data"].items():
            self.data[k][: len(v)] = v
        self.size, self.seen, self.max_size, self.evictions = state["size"], state["seen"], state["max_size"], state["evictions"]
        self.rng.bit_generator.state = state["rng"]


# ------------------------------------------------------------------ ensemble MLP

class Ensemble:
    """K independent MLPs stored as batched tensors (K, fan_in, fan_out); tanh hidden layers."""

    def __init__(self, members, sizes, rng):
        self.members, self.sizes = members, tuple(sizes)
        self.params = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = 1.0 / np.sqrt(fan_in)   # torch.nn.Linear's default bounds
            w = rng.uniform(-bound, bound, (members, fan_in, fan_out)).astype(np.float32)
            b = rng.uniform(-bound, bound, (members, 1, fan_out)).astype(np.float32)
            self.params += [torch.tensor(w, requires_grad=True), torch.tensor(b, requires_grad=True)]

    def n_params(self):
        return int(sum(p.numel() for p in self.params))

    def macs_per_row(self):
        return int(sum(a * b for a, b in zip(self.sizes[:-1], self.sizes[1:])))

    @staticmethod
    def forward(params, x):
        """x (K, B, in) -> (K, B, out)."""
        layers = len(params) // 2
        h = x
        for layer in range(layers):
            h = torch.baddbmm(params[2 * layer + 1], h, params[2 * layer])
            if layer < layers - 1:
                h = torch.tanh(h)
        return h

    @staticmethod
    def forward_member(params, k, x):
        """x (B, in) -> (B, out) through member k."""
        layers = len(params) // 2
        h = x
        for layer in range(layers):
            h = torch.addmm(params[2 * layer + 1][k], h, params[2 * layer][k])
            if layer < layers - 1:
                h = torch.tanh(h)
        return h


def clip_per_member(params, members, max_norm):
    """torch.nn.utils.clip_grad_norm_ semantics, applied to each member's slice separately."""
    with torch.no_grad():
        sq = torch.zeros(members)
        for p in params:
            sq += p.grad.reshape(members, -1).pow(2).sum(1)
        norm = sq.sqrt()
        coef = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
        for p in params:
            p.grad.mul_(coef.view(members, *([1] * (p.dim() - 1))))
    return norm


def clip_whole(params, max_norm):
    with torch.no_grad():
        norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in params))
        coef = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
        for p in params:
            p.grad.mul_(coef)
    return norm


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for t in tensors:
        a = t.detach().contiguous().numpy()
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def optimizer_hash(optimizer):
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    for key in sorted(state["state"]):
        for name in sorted(state["state"][key]):
            v = state["state"][key][name]
            a = v.detach().contiguous().numpy() if torch.is_tensor(v) else np.asarray(v)
            digest.update(str((key, name, a.shape, a.dtype.str)).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------ shared base

@dataclass(frozen=True)
class Hyper:
    lr: float
    U: int
    ema_decay: float
    members: int = 5
    hidden: int = 128
    minibatch: int = 64
    grad_clip: float = 5.0
    bootstrap_p: float = 0.5
    aux_weight: float = 1.0
    plan_samples: int = 256
    capacity: int = 1024
    bounded: bool = True
    total_episodes: int = 4096


class _Base:
    family = "?"

    def __init__(self, hyper: Hyper, rngs, track_ema=True):
        if set(rngs) != set(RNG_STREAMS):
            raise ValueError("one generator per declared randomness stream is required")
        self.h, self.rngs, self.track_ema = hyper, rngs, track_ema
        capacity = hyper.capacity if hyper.bounded else hyper.total_episodes
        self.replay = Replay(capacity, hyper.members, rngs["reservoir"], bounded=hyper.bounded)
        self.episodes = 0
        self.steps = 0
        self.macs = dict(train=0, act=0, probe=0)
        self.last_thompson = -1

    # ---- EMA
    def _init_ema(self, groups):
        self.ema = [[p.detach().clone() for p in g] for g in groups]

    def _update_ema(self, groups):
        if not self.track_ema:
            return
        d = self.h.ema_decay
        with torch.no_grad():
            for eg, g in zip(self.ema, groups):
                for e, p in zip(eg, g):
                    e.mul_(d).add_(p.detach(), alpha=1.0 - d)

    def observe(self, decision, action, outcome):
        """Store the episode and run U minibatch updates."""
        check_decision(decision)
        check_outcome(outcome)
        if not (isinstance(action, (int, np.integer)) and 0 <= int(action) < ACTIONS):
            raise IllegalObservation("action must be an int in 0..12")
        mask = self.rngs["mask"].random(self.h.members) < self.h.bootstrap_p
        self.replay.add(decision, int(action), outcome, mask)
        self.episodes += 1
        for _ in range(self.h.U):
            self._train_step()
            self.steps += 1

    def rng_state(self):
        return {k: g.bit_generator.state for k, g in self.rngs.items()}

    def load_rng_state(self, state):
        for k, g in self.rngs.items():
            g.bit_generator.state = state[k]

    def memory_bytes(self):
        n = self.n_params()
        model = 4 * n
        return dict(model=model, ema=model, optimizer=2 * model, replay_peak=self.replay.record_bytes() * self.replay.max_size,
                    total=4 * model + self.replay.record_bytes() * self.replay.max_size)


# ------------------------------------------------------------------ MF

class MFLearner(_Base):
    family = "MF"

    def __init__(self, hyper: Hyper, rngs, track_ema=True):
        super().__init__(hyper, rngs, track_ema)
        self.n_aux = ENTITIES * len(AUX_HORIZONS)
        self.net = Ensemble(hyper.members, [6, hyper.hidden, hyper.hidden, ACTIONS * (1 + self.n_aux)], rngs["init"])
        self.opt = torch.optim.Adam(self.net.params, lr=hyper.lr)
        self._init_ema([self.net.params])
        self.horizons = np.array(AUX_HORIZONS)

    def n_params(self):
        return self.net.n_params()

    @staticmethod
    def features(reserves, cue):
        return np.concatenate([np.asarray(reserves, np.float32) / RESERVE_SCALE, np.asarray(cue, np.float32)], -1)

    def act(self, decision):
        check_decision(decision)
        k = int(self.rngs["thompson"].integers(0, self.h.members))
        self.last_thompson = k
        x = torch.from_numpy(self.features(decision.reserves, decision.cue)[None])
        with torch.no_grad():
            out = Ensemble.forward_member(self.ema[0], k, x)[0, :ACTIONS]
        self.macs["act"] += self.net.macs_per_row()
        return int(torch.argmax(out))

    def _train_step(self):
        h = self.h
        idx, weight = self.replay.member_batches(self.rngs["batch"], h.minibatch)
        d = self.replay.data
        x = torch.from_numpy(self.features(d["reserves"][idx], d["cue"][idx]))            # (K, B, 6)
        a = torch.from_numpy(d["action"][idx].astype(np.int64))                           # (K, B)
        fs = d["fail_step"][idx]                                                          # (K, B, 3)
        joint = torch.from_numpy((fs.min(-1) == ALIVE).astype(np.float32))
        aux = torch.from_numpy((fs[..., :, None] > self.horizons).reshape(*fs.shape[:2], -1).astype(np.float32))
        w = torch.from_numpy(weight)
        self.opt.zero_grad(set_to_none=False)
        out = Ensemble.forward(self.net.params, x)                                         # (K, B, 13 + 117)
        joint_logit = out[..., :ACTIONS].gather(-1, a[..., None])[..., 0]
        aux_all = out[..., ACTIONS:].reshape(*out.shape[:2], ACTIONS, self.n_aux)
        aux_logit = aux_all.gather(2, a[..., None, None].expand(-1, -1, 1, self.n_aux))[:, :, 0]
        l_joint = F.binary_cross_entropy_with_logits(joint_logit, joint, reduction="none").mean(1)
        l_aux = F.binary_cross_entropy_with_logits(aux_logit, aux, reduction="none").mean((1, 2))
        loss = ((l_joint + h.aux_weight * l_aux) * w).sum()   # members are independent: sum, not mean
        loss.backward()
        clip_per_member(self.net.params, h.members, h.grad_clip)
        self.opt.step()
        self._update_ema([self.net.params])
        self.macs["train"] += 3 * self.net.macs_per_row() * h.minibatch * h.members

    def greedy(self, reserves, cue, **_):
        """Ensemble-mean (EMA) predictions on decision states (n, 3), (n, 3).

        Returns dict(action (n,), value (n, 13) mean joint-survival probability,
        entity (n, 13, 3) mean per-entity survival to step 12 (auxiliary head), aux (n, 13, 3, 3))."""
        x = torch.from_numpy(self.features(reserves, cue))
        with torch.no_grad():
            out = Ensemble.forward(self.ema[0], x[None].expand(self.h.members, -1, -1))
            p = torch.sigmoid(out)
            value = p[..., :ACTIONS].mean(0).numpy()
            aux = p[..., ACTIONS:].reshape(self.h.members, -1, ACTIONS, ENTITIES, len(AUX_HORIZONS)).mean(0).numpy()
        self.macs["probe"] += self.net.macs_per_row() * len(x) * self.h.members
        return dict(action=value.argmax(1), value=value, entity=aux[..., -1], aux=aux)

    def state(self):
        return dict(family=self.family, params=[p.detach().clone() for p in self.net.params],
                    ema=[[e.clone() for e in g] for g in self.ema], opt=self.opt.state_dict(), replay=self.replay.state(),
                    rng=self.rng_state(), episodes=self.episodes, steps=self.steps, macs=dict(self.macs))

    def load(self, state):
        with torch.no_grad():
            for p, v in zip(self.net.params, state["params"]):
                p.copy_(v)
        self.ema = [[e.clone() for e in g] for g in state["ema"]]
        self.opt.load_state_dict(state["opt"])
        self.replay.load(state["replay"])
        self.load_rng_state(state["rng"])
        self.episodes, self.steps, self.macs = state["episodes"], state["steps"], dict(state["macs"])

    def hashes(self):
        return dict(online=tensor_hash(self.net.params), ema=tensor_hash(self.ema[0]), optimizer=optimizer_hash(self.opt))


# ------------------------------------------------------------------ MB

class MBLearner(_Base):
    family = "MB"
    n_in = 3 + 3 + ACTIONS + T * ENTITIES

    def __init__(self, hyper: Hyper, rngs, track_ema=True):
        super().__init__(hyper, rngs, track_ema)
        self.net = Ensemble(hyper.members, [self.n_in, hyper.hidden, hyper.hidden, ENTITIES * T], rngs["init"])
        # Free Gaussian tables, initialised at mean 0, log-variance 0 (no knowledge of the world).
        self.supply = [torch.zeros(T, ENTITIES, requires_grad=True), torch.zeros(T, ENTITIES, requires_grad=True)]
        self.probe = [torch.zeros(T, ENTITIES, requires_grad=True), torch.zeros(T, ENTITIES, requires_grad=True)]
        self.opt = torch.optim.Adam(self.net.params, lr=hyper.lr)
        self.opt_tables = torch.optim.Adam(self.supply + self.probe, lr=hyper.lr)
        self._init_ema([self.net.params, self.supply, self.probe])
        self.steps_t = torch.arange(1, T + 1)
        self.onehot = torch.eye(ACTIONS)

    def n_params(self):
        return self.net.n_params() + 4 * T * ENTITIES

    @staticmethod
    def static_features(reserves, cue):
        return np.concatenate([np.asarray(reserves, np.float32) / RESERVE_SCALE, np.asarray(cue, np.float32)], -1)

    # ---- training
    def _train_step(self):
        h = self.h
        d = self.replay.data
        idx, weight = self.replay.member_batches(self.rngs["batch"], h.minibatch)
        x = np.concatenate([self.static_features(d["reserves"][idx], d["cue"][idx]),
                            np.eye(ACTIONS, dtype=np.float32)[d["action"][idx]],
                            d["y_supply"][idx].reshape(*idx.shape, -1) / SUPPLY_SCALE], -1)
        fs = torch.from_numpy(d["fail_step"][idx].astype(np.int64))                        # (K, B, 3)
        w = torch.from_numpy(weight)
        self.opt.zero_grad(set_to_none=False)
        logits = Ensemble.forward(self.net.params, torch.from_numpy(x)).reshape(*idx.shape, ENTITIES, T)
        ll = self.failure_loglik(logits, fs)                                                  # (K, B)
        loss = (-ll.mean(1) * w).sum()
        loss.backward()
        clip_per_member(self.net.params, h.members, h.grad_clip)
        self.opt.step()
        # tables: uniform minibatch over the whole replay
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
        self.macs["train"] += 3 * self.net.macs_per_row() * h.minibatch * h.members + 6 * 2 * T * ENTITIES * h.minibatch

    def failure_loglik(self, logits, fs):
        """log P(failure steps) under discrete hazards; logits (..., 3, 12), fs (..., 3) in 1..13 -> (...)."""
        before = (self.steps_t < fs[..., None]).to(logits.dtype)
        at = (self.steps_t == fs[..., None]).to(logits.dtype)
        return (F.logsigmoid(-logits) * before + F.logsigmoid(logits) * at).sum((-1, -2))

    @staticmethod
    def gauss_nll(table, y):
        mean, logvar = table
        return 0.5 * (logvar + (y - mean) ** 2 * torch.exp(-logvar)).sum((-1, -2)).mean()

    # ---- planning
    def _values(self, params, k, reserves, cue, y):
        """Member k: per-entity survival (S, 13, M, 3) log-probabilities for S states and draws y (S, M, 36)."""
        w1, b1, w2, b2, w3, b3 = params
        s = torch.from_numpy(self.static_features(reserves, cue))                            # (S, 6)
        base = s @ w1[k, :6] + b1[k]                                                          # (S, H)
        act = w1[k, 6:6 + ACTIONS]                                                           # (13, H)
        ysup = y @ w1[k, 6 + ACTIONS:]                                                        # (S, M, H)
        pre = base[:, None, None, :] + act[None, :, None, :] + ysup[:, None, :, :]           # (S, 13, M, H)
        h1 = torch.tanh(pre)
        h2 = torch.tanh(h1 @ w2[k] + b2[k])
        logits = (h2 @ w3[k] + b3[k]).reshape(*h2.shape[:3], ENTITIES, T)
        return F.logsigmoid(-logits).sum(-1)                                                  # (S, 13, M, 3)

    def _plan_macs(self, states, members):
        m, hdim = self.h.plan_samples, self.h.hidden
        per_state = 6 * hdim + m * T * ENTITIES * hdim + ACTIONS * m * (hdim * hdim + hdim * ENTITIES * T)
        return int(per_state * states * members)

    def supply_draws(self, eps, table):
        mean, logvar = table
        return ((mean + torch.exp(0.5 * logvar) * eps) / SUPPLY_SCALE).reshape(*eps.shape[:-2], T * ENTITIES)

    def act(self, decision):
        check_decision(decision)
        k = int(self.rngs["thompson"].integers(0, self.h.members))
        self.last_thompson = k
        eps = torch.from_numpy(self.rngs["plan"].standard_normal((self.h.plan_samples, T, ENTITIES)).astype(np.float32))
        with torch.no_grad():
            y = self.supply_draws(eps, self.ema[1])[None]                                    # (1, M, 36)
            logs = self._values(self.ema[0], k, decision.reserves[None], decision.cue[None], y)
            value = torch.exp(logs.sum(-1)).mean(-1)[0]                                       # (13,)
        self.macs["act"] += self._plan_macs(1, 1)
        return int(torch.argmax(value))

    def greedy(self, reserves, cue, eps=None, chunk=16):
        """Ensemble-mean (EMA) planning on decision states with fixed draws eps (n, M, 12, 3).

        Returns dict(action (n,), value (n, 13) mean joint survival, entity (n, 13, 3) mean per-entity survival)."""
        n = len(reserves)
        value = np.zeros((n, ACTIONS), np.float64)
        entity = np.zeros((n, ACTIONS, ENTITIES), np.float64)
        with torch.no_grad():
            for lo in range(0, n, chunk):
                hi = min(n, lo + chunk)
                y = self.supply_draws(torch.from_numpy(np.ascontiguousarray(eps[lo:hi], dtype=np.float32)), self.ema[1])
                v = torch.zeros(hi - lo, ACTIONS)
                e = torch.zeros(hi - lo, ACTIONS, ENTITIES)
                for k in range(self.h.members):
                    logs = self._values(self.ema[0], k, reserves[lo:hi], cue[lo:hi], y)
                    v += torch.exp(logs.sum(-1)).mean(-1)
                    e += torch.exp(logs).mean(-2)
                value[lo:hi] = (v / self.h.members).numpy()
                entity[lo:hi] = (e / self.h.members).numpy()
        self.macs["probe"] += self._plan_macs(n, self.h.members)
        return dict(action=value.argmax(1), value=value, entity=entity)

    def hazard_nll(self, reserves, cue, actions, y_supply, fail_step):
        """Ensemble-mixture NLL (nats per entity) of observed failure steps given sensed supplies (EMA weights)."""
        x = np.concatenate([self.static_features(reserves, cue), np.eye(ACTIONS, dtype=np.float32)[actions],
                            np.asarray(y_supply, np.float32).reshape(len(actions), -1) / SUPPLY_SCALE], -1)
        xt = torch.from_numpy(x)[None].expand(self.h.members, -1, -1)
        with torch.no_grad():
            logits = Ensemble.forward(self.ema[0], xt).reshape(self.h.members, len(actions), ENTITIES, T)
            fs = torch.from_numpy(np.asarray(fail_step, np.int64))
            before = (self.steps_t < fs[..., None]).float()
            at = (self.steps_t == fs[..., None]).float()
            per = (F.logsigmoid(-logits) * before + F.logsigmoid(logits) * at).sum(-1)       # (K, n, 3)
            mix = torch.logsumexp(per, 0) - np.log(self.h.members)
        self.macs["probe"] += self.net.macs_per_row() * len(actions) * self.h.members
        return float(-mix.mean())

    def tables(self):
        return dict(supply_mean=self.ema[1][0].numpy().copy(), supply_logvar=self.ema[1][1].numpy().copy(),
                    probe_mean=self.ema[2][0].numpy().copy(), probe_logvar=self.ema[2][1].numpy().copy(),
                    online_supply_mean=self.supply[0].detach().numpy().copy(),
                    online_supply_logvar=self.supply[1].detach().numpy().copy())

    def state(self):
        return dict(family=self.family, params=[p.detach().clone() for p in self.net.params],
                    supply=[p.detach().clone() for p in self.supply], probe=[p.detach().clone() for p in self.probe],
                    ema=[[e.clone() for e in g] for g in self.ema], opt=self.opt.state_dict(),
                    opt_tables=self.opt_tables.state_dict(), replay=self.replay.state(), rng=self.rng_state(),
                    episodes=self.episodes, steps=self.steps, macs=dict(self.macs))

    def load(self, state):
        with torch.no_grad():
            for p, v in zip(self.net.params + self.supply + self.probe, state["params"] + state["supply"] + state["probe"]):
                p.copy_(v)
        self.ema = [[e.clone() for e in g] for g in state["ema"]]
        self.opt.load_state_dict(state["opt"])
        self.opt_tables.load_state_dict(state["opt_tables"])
        self.replay.load(state["replay"])
        self.load_rng_state(state["rng"])
        self.episodes, self.steps, self.macs = state["episodes"], state["steps"], dict(state["macs"])

    def hashes(self):
        return dict(online=tensor_hash(self.net.params + self.supply + self.probe),
                    ema=tensor_hash([e for g in self.ema for e in g]),
                    optimizer=optimizer_hash(self.opt) + optimizer_hash(self.opt_tables))


FAMILIES = {"MF": MFLearner, "MB": MBLearner}


def make_learner(family, hyper, seeds, track_ema=True):
    """seeds: dict stream -> int seed (one per RNG_STREAMS entry)."""
    rngs = {k: np.random.default_rng(seeds[k]) for k in RNG_STREAMS}
    return FAMILIES[family](hyper, rngs, track_ema=track_ema)
