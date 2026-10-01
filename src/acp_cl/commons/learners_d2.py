"""Step D2 learners (docs/commons_d2_protocol.md): a shared MB-count core with three context vectors.

Imports only D1's learner module (never edited); nothing here can see the world. A learner receives the two legal
observations of ``acp_cl.commons.learners`` (``Decision``, ``Outcome``) and, for the privileged OC arm only, a ``Label``
of an episode's true factor levels, handed over AFTER that episode's outcome (so it indexes the next decision: lagged
one episode). Every other arm rejects a label.

Shared core (every arm; D1b's MB-count with the D2 conditioning)
* supply model: mean = table_mean + (c_s W_s), log-variance = table_logvar + (c_s V_s)   (tables (12, 3), W, V (6, 36));
* probe model: the same form with c_p (trained, used by decisions only through c_p in the failure model);
* failure ensemble: 5 MLPs (73 -> 128 -> 128 -> 36, tanh); input = reserves/18, cue, action one-hot, sensed supplies/2
  and [c_s, c_p, c_f] (18);
* acting: MB-count (Thompson member + planning on M = 256 supply draws from the EMA supply model at c_s, plus
  c / sqrt(1 + n_a) with n_a counted over the CURRENT reservoir, c = .05); greedy probes: EMA ensemble mean, no bonus;
* training: per episode U = 2 steps; each step = one failure minibatch per member (bootstrap masks, 64 rows) and one
  table minibatch (64 rows, uniform), a single backward, gradient-norm clip 5 per member / per table group / on the
  context parameters, Adam (core lr .001; code lr for codes; encoder lr for AM encoders); EMA d = .99 on everything
  that trains (core, G constants, AM encoders, codes), used for acting, probes and FC evidence.

Arms (context vectors c_s, c_p, c_f; 6 dimensions each)
* G: three learned constants (core lr). W: G with every learned state (weights, EMA, optimisers, reservoir, counts)
  reset to its initial value at each true switch (``reset()``; the harness calls it; random streams continue).
* AM: one GRU (H 32) over the compact summaries of the preceding k episodes -> 18 dimensions; AM-factored: three GRUs
  (H 16) over supply, probe and (decision state, action, failure) summaries -> 6 each. Reservoir entries carry the
  summaries of their preceding k episodes; training reads only those. Compact summary per episode (29 bytes):
  float16 [supply half-means per entity (6), probe mean per channel (3), decision reserves (3)] and int8
  [action, failure step x 3, packed cue bits].
* FC: per slot a library of at most 6 codes, filter + spawn (below). FC-single: one library of 18-d codes on the joint
  evidence. FC-K1: FC with one code per slot, never replaced (library disabled). OC: FC's architecture with two codes
  per slot indexed by the lagged true level (A -> c_s, B -> c_p, C -> c_f).

FC mechanism (per slot; ``spawn_rule`` "literal" is the protocol text, "lrt" is a PROPOSED revision, see below)
* evidence: supply / probe slots: exact Gaussian log-likelihood of that stream's 36 sensors under the EMA core with
  each library code and with the default code; failure slot: ensemble-mixture hazard log-likelihood of the observed
  failure steps, only on informative episodes (a 2<->3 transfer: actions 3, 4, 9, 10), with c_s and c_p at their
  slots' active codes after this episode's update;
* default code: the mean of the library codes (online values for training, EMA values for evidence and acting); the
  core trains with the (detached) default in place of the assigned code on 20% of its minibatches, drawn per slot
  and per training step from a dedicated generator;
* filter: log-space sticky filter over the library codes (stay 1 - h, switch h / (K - 1), h = .05), advanced only on
  the slot's informative episodes; the active code changes only when some code's posterior is >= .95;
* spawn: when the default beats every library code by more than tau nats on 2 consecutive informative episodes, a new
  code initialised at the default is added, becomes active with a point-mass posterior (Step C's convention), and the
  run counter resets; with a full library the code with the fewest assigned episodes currently in the reservoir is
  replaced (ties: the oldest). Codes have unique ids; a replaced code's replay episodes keep their id and are then
  trained with the default in that slot (no relabelling);
* training: assignments (the active code ids after the episode's update) are stored with each replay episode and never
  relabelled; codes train by gradient only on assigned replay rows, with a per-code (lazy) Adam at the code lr and
  decoupled weight decay ``code_decay`` on the stepped rows (implementation fix after the development OC gate: without it
  a persistent common-mode gradient walks the codes to |c| ~ 50 under constant-size Adam steps; the decay bounds the
  drift at about 1 / code_decay, i.e. inputs on the MLP's O(1) scale).
* NOTE: with one code the default IS that code, so under "literal" the spawn rule can never fire and FC is FC-K1.
  "lrt" (proposed revision): the default is first fitted to the current episode (Gaussian slots: ridge-regularised
  weighted least squares of the mean in code space around the default, lambda = 1, log-variance at the default's;
  failure slot: 20 Adam steps at the code lr on the episode's hazard log-likelihood from the default); the spawn rule
  compares this fitted default with the library codes, and a spawned code is initialised at the fitted default.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np
import torch
from torch.nn import functional as F

from .learners import (ACTIONS, ALIVE, ENTITIES, RESERVE_SCALE, SUPPLY_SCALE, T, Ensemble, Hyper, IllegalObservation,
                       Replay, check_decision, check_outcome, clip_per_member, clip_whole, optimizer_hash, tensor_hash)

CTX = 6
N_SLOTS = 3
CTX_ALL = CTX * N_SLOTS
PROBE_SCALE = 0.3
N_STATIC = 6
N_IN = N_STATIC + ACTIONS + T * ENTITIES + CTX_ALL          # 73
INFORMATIVE_ACTIONS = (3, 4, 9, 10)                          # 2->3, 3->2 at 3 and 6 units (checked against the world)
RNG_STREAMS = ("init", "mask", "thompson", "batch", "table_batch", "reservoir", "plan", "default")
ARMS = ("G", "W", "AM", "AM-factored", "FC", "FC-single", "FC-K1", "OC")
SUMMARY_FLOATS, SUMMARY_INTS = 12, 5
SUMMARY_BYTES = 2 * SUMMARY_FLOATS + SUMMARY_INTS           # 29
LOG2PI = math.log(2 * math.pi)


# ------------------------------------------------------------------ the privileged label (OC only)

@dataclass(frozen=True)
class Label:
    levels: np.ndarray   # (3,) uint8: true factor levels (A, B, C) of the episode just completed


def make_label(levels):
    arr = np.array(levels, dtype=np.uint8, copy=True)
    arr.setflags(write=False)
    return check_label(Label(arr))


def check_label(obj):
    if type(obj) is not Label or tuple(f.name for f in fields(obj)) != ("levels",) or set(vars(obj)) != {"levels"}:
        raise IllegalObservation("illegal label object")
    v = obj.levels
    if not (isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.shape == (3,) and not v.flags.writeable
            and np.all(v <= 1)):
        raise IllegalObservation("label must be read-only uint8 (3,) in {0, 1}")
    return obj


# ------------------------------------------------------------------ compact summaries (AM)

def summarize_episode(decision, action, outcome):
    """(float16 (12,), int8 (5,)) compact summary of one completed episode (legal observations only)."""
    ys, yp = outcome.y_supply, outcome.y_probe
    f = np.concatenate([ys[:6].mean(0), ys[6:].mean(0), yp.mean(0), decision.reserves]).astype(np.float16)
    cue = int(decision.cue[0]) | (int(decision.cue[1]) << 1) | (int(decision.cue[2]) << 2)
    i = np.array([int(action), *[int(x) for x in outcome.fail_step], cue], np.int8)
    return f, i


def summary_features(hf, hi, factored):
    """Encoder inputs from stored summaries hf (..., k, 12) float16, hi (..., k, 5) int8 (action -1 = padding)."""
    f = hf.astype(np.float32)
    act = hi[..., 0].astype(np.int64)
    valid = (act >= 0).astype(np.float32)[..., None]
    sup = f[..., 0:6] / SUPPLY_SCALE
    prb = f[..., 6:9] / PROBE_SCALE
    res = f[..., 9:12] / RESERVE_SCALE
    cuei = hi[..., 4].astype(np.int64)
    cue = np.stack([(cuei >> b) & 1 for b in range(3)], -1).astype(np.float32) * valid
    onehot = np.eye(ACTIONS, dtype=np.float32)[np.maximum(act, 0)] * valid
    fail = hi[..., 1:4].astype(np.float32) / ALIVE * valid
    if not factored:
        return [np.concatenate([sup, prb, res, cue, onehot, fail, valid], -1)]
    return [np.concatenate([sup, valid], -1), np.concatenate([prb, valid], -1),
            np.concatenate([res, cue, onehot, fail, valid], -1)]


class Encoder(torch.nn.Module):
    """GRU stream(s) -> last hidden -> linear head(s); initialised from a numpy generator (torch.nn's bounds)."""

    def __init__(self, specs, rng):
        super().__init__()
        self.specs = tuple(specs)
        self.grus = torch.nn.ModuleList([torch.nn.GRU(i, h, batch_first=True) for i, h, _ in specs])
        self.heads = torch.nn.ModuleList([torch.nn.Linear(h, o) for _, h, o in specs])
        with torch.no_grad():
            for (i, h, o), gru, head in zip(specs, self.grus, self.heads):
                for p in gru.parameters():
                    p.copy_(torch.from_numpy(rng.uniform(-1 / math.sqrt(h), 1 / math.sqrt(h), tuple(p.shape)).astype(np.float32)))
                for p in head.parameters():
                    p.copy_(torch.from_numpy(rng.uniform(-1 / math.sqrt(h), 1 / math.sqrt(h), tuple(p.shape)).astype(np.float32)))

    def forward(self, xs):
        outs = []
        for x, gru, head in zip(xs, self.grus, self.heads):
            _, hn = gru(x)
            outs.append(head(hn[0]))
        return torch.cat(outs, -1)


# ------------------------------------------------------------------ helpers

def gauss_nll_rows(group, c, y):
    """Mean over rows of 0.5 * sum(logvar + (y - mean)^2 exp(-logvar)); group = [mean, logvar, W, V], c (B, 6)."""
    mean_t, logv_t, w, v = group
    mean = mean_t + (c @ w).reshape(-1, T, ENTITIES)
    logv = logv_t + (c @ v).reshape(-1, T, ENTITIES)
    return 0.5 * (logv + (y - mean) ** 2 * torch.exp(-logv)).sum((-1, -2)).mean()


def gauss_loglik(group, c, y):
    """Exact log-likelihood (J,) of sensors y (12, 3) under codes c (J, 6)."""
    mean_t, logv_t, w, v = group
    mean = mean_t + (c @ w).reshape(-1, T, ENTITIES)
    logv = logv_t + (c @ v).reshape(-1, T, ENTITIES)
    return -0.5 * (LOG2PI + logv + (y - mean) ** 2 * torch.exp(-logv)).sum((-1, -2))


def failure_loglik(logits, fs, steps):
    before = (steps < fs[..., None]).to(logits.dtype)
    at = (steps == fs[..., None]).to(logits.dtype)
    return (F.logsigmoid(-logits) * before + F.logsigmoid(logits) * at).sum((-1, -2))


def logsumexp_np(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else float(out.reshape(()))


def sticky_step(logpost, loglik, h):
    """One log-space sticky-filter step: predict with stay 1 - h / switch h / (K - 1), then update. Returns log posterior."""
    k = len(logpost)
    if k == 1:
        return np.zeros(1)
    trans = np.full((k, k), math.log(h / (k - 1)))
    np.fill_diagonal(trans, math.log(1 - h))
    pred = logsumexp_np(logpost[:, None] + trans, axis=0)
    post = pred + loglik
    return post - logsumexp_np(post)


def sticky_bruteforce(logliks, h):
    """Posterior over the last hidden code by enumerating every code sequence (tests only). logliks (T, K)."""
    import itertools
    tt, k = logliks.shape
    post = np.full(k, -np.inf)
    for seq in itertools.product(range(k), repeat=tt):
        lp = math.log(1.0 / k) + sum(logliks[t, seq[t]] for t in range(tt))
        for t in range(1, tt):
            lp += math.log(1 - h) if seq[t] == seq[t - 1] else math.log(h / (k - 1))
        post[seq[-1]] = np.logaddexp(post[seq[-1]], lp)
    return post - logsumexp_np(post)


class LazyAdam:
    """Adam on the rows of a (rows, dim) tensor, stepping only rows that received gradient this step (per-row counts)."""

    def __init__(self, rows, dim, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, float(weight_decay)
        self.m = torch.zeros(rows, dim)
        self.v = torch.zeros(rows, dim)
        self.t = torch.zeros(rows, dtype=torch.int64)

    def step(self, param, grad, used):
        with torch.no_grad():
            for r in np.flatnonzero(used):
                g = grad[r]
                self.t[r] += 1
                t = int(self.t[r])
                self.m[r].mul_(self.b1).add_(g, alpha=1 - self.b1)
                self.v[r].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                mhat = self.m[r] / (1 - self.b1 ** t)
                vhat = self.v[r] / (1 - self.b2 ** t)
                if self.wd:
                    param[r] -= self.lr * self.wd * param[r]          # decoupled (AdamW) decay, stepped rows only
                param[r] -= self.lr * mhat / (vhat.sqrt() + self.eps)

    def reset_row(self, r):
        self.m[r].zero_()
        self.v[r].zero_()
        self.t[r] = 0

    def state(self):
        return dict(m=self.m.clone(), v=self.v.clone(), t=self.t.clone())

    def load(self, st):
        self.m, self.v, self.t = st["m"].clone(), st["v"].clone(), st["t"].clone()


# ------------------------------------------------------------------ the shared core

class D2Base:
    arm = "?"
    uses_label = False

    def __init__(self, hyper: Hyper, rngs, count_c=0.05):
        if set(rngs) != set(RNG_STREAMS):
            raise ValueError("one generator per declared randomness stream is required")
        self.h, self.rngs, self.count_c = hyper, rngs, float(count_c)
        self.replay = Replay(hyper.capacity, hyper.members, rngs["reservoir"], bounded=True)
        self._extend_replay()
        g = rngs["init"]
        self.net = Ensemble(hyper.members, [N_IN, hyper.hidden, hyper.hidden, ENTITIES * T], g)
        bound = 1.0 / math.sqrt(CTX)

        def group():
            return [torch.zeros(T, ENTITIES, requires_grad=True), torch.zeros(T, ENTITIES, requires_grad=True),
                    torch.tensor(g.uniform(-bound, bound, (CTX, T * ENTITIES)).astype(np.float32), requires_grad=True),
                    torch.tensor(g.uniform(-bound, bound, (CTX, T * ENTITIES)).astype(np.float32), requires_grad=True)]
        self.supply, self.probe = group(), group()
        self._build_context(g)
        self.opt = torch.optim.Adam(self.net.params, lr=hyper.lr)
        self.opt_tables = torch.optim.Adam(self.supply + self.probe, lr=hyper.lr)
        self.ema = [[p.detach().clone() for p in grp] for grp in self._groups()]
        self.steps_t = torch.arange(1, T + 1)
        self.episodes = 0
        self.steps = 0
        self.resets = 0
        self.macs = dict(train=0, act=0, probe=0, evidence=0)
        self.last_thompson = -1
        self.last_base_action = -1
        self.last_counts = np.zeros(ACTIONS, np.int64)
        self.last_ctx = np.zeros(CTX_ALL, np.float32)
        self.default_flag_counts = np.zeros(N_SLOTS, np.int64)
        self._initial = None

    # ---- hooks (subclasses)
    def _extend_replay(self):
        pass

    def _build_context(self, g):
        raise NotImplementedError

    def _ctx_groups(self):
        """Lists of context tensors that the EMA tracks (may be empty)."""
        return []

    def _groups(self):
        return [self.net.params, self.supply, self.probe] + self._ctx_groups()

    def ctx_rows(self, rows, flags):
        """(N, 18) context tensor for replay rows (online parameters, with grad), flags: default substitution per slot."""
        raise NotImplementedError

    def ctx_step(self):
        """Clip and apply the context parameters' gradients (after backward)."""

    def ctx_zero_grad(self):
        pass

    def ctx_act(self):
        """(18,) float32 tensor: the EMA context for the current decision."""
        raise NotImplementedError

    def _after_outcome(self, decision, action, outcome, label):
        """Update the context state from the completed episode; return the replay extras for it."""
        if label is not None:
            raise IllegalObservation(f"{self.arm} must not receive a label")
        return {}

    def _default_flags(self):
        return np.zeros(N_SLOTS, bool)

    # ---- sizes and bytes
    def n_params(self):
        return self.net.n_params() + sum(p.numel() for p in self.supply + self.probe)

    def context_bytes(self):
        return 0

    def memory_report(self):
        rec = self.replay.record_bytes()
        return dict(replay_record_bytes=int(rec), replay_capacity=int(self.replay.capacity),
                    replay_bytes=int(rec * self.replay.capacity), context_bytes=int(self.context_bytes()),
                    budget_bytes=int(rec * self.replay.capacity + self.context_bytes()),
                    core_parameter_bytes=int(4 * self.n_params()), replay_peak_bytes=int(rec * self.replay.max_size))

    # ---- EMA
    def _update_ema(self):
        d = self.h.ema_decay
        with torch.no_grad():
            for eg, grp in zip(self.ema, self._groups()):
                for e, p in zip(eg, grp):
                    e.mul_(d).add_(p.detach(), alpha=1.0 - d)

    # ---- model pieces
    @staticmethod
    def static_features(reserves, cue):
        return np.concatenate([np.asarray(reserves, np.float32) / RESERVE_SCALE, np.asarray(cue, np.float32)], -1)

    def supply_draws(self, eps, group, cs):
        """eps (S, M, 12, 3), cs (S, 6) -> sensed-supply draws / 2, (S, M, 36)."""
        mean_t, logv_t, w, v = group
        mean = mean_t + (cs @ w).reshape(-1, 1, T, ENTITIES)
        logv = logv_t + (cs @ v).reshape(-1, 1, T, ENTITIES)
        return ((mean + torch.exp(0.5 * logv) * eps) / SUPPLY_SCALE).reshape(eps.shape[0], eps.shape[1], T * ENTITIES)

    def _values(self, params, k, reserves, cue, ctx, y):
        """Member k: per-entity log-survival (S, 13, M, 3); ctx (S, 18), y (S, M, 36)."""
        w1, b1, w2, b2, w3, b3 = params
        s = torch.from_numpy(self.static_features(reserves, cue))
        base = s @ w1[k, :N_STATIC] + ctx @ w1[k, N_STATIC + ACTIONS + T * ENTITIES:] + b1[k]
        act = w1[k, N_STATIC:N_STATIC + ACTIONS]
        ysup = y @ w1[k, N_STATIC + ACTIONS:N_STATIC + ACTIONS + T * ENTITIES]
        pre = base[:, None, None, :] + act[None, :, None, :] + ysup[:, None, :, :]
        h2 = torch.tanh(torch.tanh(pre) @ w2[k] + b2[k])
        logits = (h2 @ w3[k] + b3[k]).reshape(*h2.shape[:3], ENTITIES, T)
        return F.logsigmoid(-logits).sum(-1)

    def _plan_macs(self, states, members):
        m, hd = self.h.plan_samples, self.h.hidden
        per = (N_STATIC + CTX_ALL) * hd + m * T * ENTITIES * hd + ACTIONS * m * (hd * hd + hd * ENTITIES * T)
        return int(per * states * members)

    def reservoir_counts(self):
        return np.bincount(self.replay.data["action"][: self.replay.size].astype(np.int64), minlength=ACTIONS)

    # ---- acting
    def act(self, decision):
        check_decision(decision)
        k = int(self.rngs["thompson"].integers(0, self.h.members))
        self.last_thompson = k
        eps = torch.from_numpy(self.rngs["plan"].standard_normal((self.h.plan_samples, T, ENTITIES)).astype(np.float32))
        with torch.no_grad():
            ctx = self.ctx_act()[None]
            self.last_ctx = ctx[0].numpy().copy()
            y = self.supply_draws(eps[None], self.ema[1], ctx[:, :CTX])
            logs = self._values(self.ema[0], k, decision.reserves[None], decision.cue[None], ctx, y)
            value = torch.exp(logs.sum(-1)).mean(-1)[0]
        self.macs["act"] += self._plan_macs(1, 1)
        counts = self.reservoir_counts()
        self.last_counts = counts
        bonus = torch.from_numpy((self.count_c / np.sqrt(1.0 + counts)).astype(np.float32))
        self.last_base_action = int(torch.argmax(value))
        return int(torch.argmax(value + bonus))

    def greedy(self, reserves, cue, eps, chunk=16):
        """EMA ensemble-mean planning (no bonus) on decision states with fixed draws eps (n, M, 12, 3) at the current
        context. Returns dict(action (n,), value (n, 13))."""
        n = len(reserves)
        value = np.zeros((n, ACTIONS), np.float64)
        with torch.no_grad():
            ctx1 = self.ctx_act()[None]
            for lo in range(0, n, chunk):
                hi = min(n, lo + chunk)
                ctx = ctx1.expand(hi - lo, -1)
                y = self.supply_draws(torch.from_numpy(np.ascontiguousarray(eps[lo:hi], dtype=np.float32)), self.ema[1],
                                      ctx[:, :CTX])
                v = torch.zeros(hi - lo, ACTIONS)
                for k in range(self.h.members):
                    v += torch.exp(self._values(self.ema[0], k, reserves[lo:hi], cue[lo:hi], ctx, y).sum(-1)).mean(-1)
                value[lo:hi] = (v / self.h.members).numpy()
        self.macs["probe"] += self._plan_macs(n, self.h.members)
        return dict(action=value.argmax(1), value=value)

    # ---- learning
    def observe(self, decision, action, outcome, label=None):
        check_decision(decision)
        check_outcome(outcome)
        if not (isinstance(action, (int, np.integer)) and 0 <= int(action) < ACTIONS):
            raise IllegalObservation("action must be an int in 0..12")
        if label is not None:
            check_label(label)
            if not self.uses_label:
                raise IllegalObservation(f"{self.arm} must not receive a label")
        elif self.uses_label:
            raise IllegalObservation("OC needs the completed episode's label")
        extras = self._after_outcome(decision, int(action), outcome, label)
        mask = self.rngs["mask"].random(self.h.members) < self.h.bootstrap_p
        slot = self.replay.add(decision, int(action), outcome, mask)
        if slot is not None:
            for key, value in extras.items():
                self.replay.data[key][slot] = value
        self.episodes += 1
        for _ in range(self.h.U):
            self._train_step()
            self.steps += 1
        return slot

    def _train_step(self):
        h = self.h
        d = self.replay.data
        idx, weight = self.replay.member_batches(self.rngs["batch"], h.minibatch)
        tidx = self.rngs["table_batch"].integers(0, self.replay.size, h.minibatch)
        flags = self._default_flags()
        self.default_flag_counts += flags
        rows = np.concatenate([idx.reshape(-1), tidx])
        self.opt.zero_grad(set_to_none=False)
        self.opt_tables.zero_grad(set_to_none=False)
        self.ctx_zero_grad()
        ctx = self.ctx_rows(rows, flags)
        kb = idx.size
        cf = ctx[:kb].reshape(*idx.shape, CTX_ALL)
        ct = ctx[kb:]
        x = np.concatenate([self.static_features(d["reserves"][idx], d["cue"][idx]),
                            np.eye(ACTIONS, dtype=np.float32)[d["action"][idx]],
                            d["y_supply"][idx].reshape(*idx.shape, -1) / SUPPLY_SCALE], -1)
        xt = torch.cat([torch.from_numpy(x), cf], -1)
        fs = torch.from_numpy(d["fail_step"][idx].astype(np.int64))
        logits = Ensemble.forward(self.net.params, xt).reshape(*idx.shape, ENTITIES, T)
        loss_f = (-failure_loglik(logits, fs, self.steps_t).mean(1) * torch.from_numpy(weight)).sum()
        ys = torch.from_numpy(d["y_supply"][tidx])
        yp = torch.from_numpy(d["y_probe"][tidx])
        loss_t = gauss_nll_rows(self.supply, ct[:, :CTX], ys) + gauss_nll_rows(self.probe, ct[:, CTX:2 * CTX], yp)
        (loss_f + loss_t).backward()
        clip_per_member(self.net.params, h.members, h.grad_clip)
        clip_whole(self.supply, h.grad_clip)
        clip_whole(self.probe, h.grad_clip)
        self.opt.step()
        self.opt_tables.step()
        self.ctx_step()
        self._update_ema()
        self.macs["train"] += 3 * self.net.macs_per_row() * h.minibatch * h.members + 6 * 2 * T * ENTITIES * h.minibatch * (1 + CTX)

    # ---- evidence pieces (EMA, no grad)
    def hazard_loglik_codes(self, decision, action, outcome, ctx):
        """Ensemble-mixture log-likelihood (J,) of the observed failure steps for contexts ctx (J, 18) (EMA weights)."""
        j = ctx.shape[0]
        x = np.concatenate([self.static_features(decision.reserves, decision.cue), np.eye(ACTIONS, dtype=np.float32)[action],
                            np.asarray(outcome.y_supply, np.float32).reshape(-1) / SUPPLY_SCALE])
        xt = torch.cat([torch.from_numpy(x)[None].expand(j, -1), ctx], -1)[None].expand(self.h.members, -1, -1)
        logits = Ensemble.forward(self.ema[0], xt).reshape(self.h.members, j, ENTITIES, T)
        fs = torch.from_numpy(np.asarray(outcome.fail_step, np.int64))[None].expand(j, -1)
        per = failure_loglik(logits, fs, self.steps_t)                                           # (K, J)
        self.macs["evidence"] += self.net.macs_per_row() * j * self.h.members
        return torch.logsumexp(per, 0) - math.log(self.h.members)

    # ---- state
    def rng_state(self):
        return {k: g.bit_generator.state for k, g in self.rngs.items()}

    def load_rng_state(self, state):
        for k, g in self.rngs.items():
            g.bit_generator.state = state[k]

    def _ctx_state(self):
        return {}

    def _load_ctx_state(self, st):
        pass

    def state(self):
        return dict(arm=self.arm, params=[p.detach().clone() for p in self.net.params],
                    supply=[p.detach().clone() for p in self.supply], probe=[p.detach().clone() for p in self.probe],
                    ema=[[e.clone() for e in g] for g in self.ema], opt=self.opt.state_dict(),
                    opt_tables=self.opt_tables.state_dict(), replay=self.replay.state(), rng=self.rng_state(),
                    episodes=self.episodes, steps=self.steps, resets=self.resets, macs=dict(self.macs),
                    default_flag_counts=self.default_flag_counts.copy(), ctx=self._ctx_state())

    def load(self, state, rng=True, counters=True):
        with torch.no_grad():
            for p, v in zip(self.net.params + self.supply + self.probe, state["params"] + state["supply"] + state["probe"]):
                p.copy_(v)
        self.ema = [[e.clone() for e in g] for g in state["ema"]]
        self.opt.load_state_dict(state["opt"])
        self.opt_tables.load_state_dict(state["opt_tables"])
        self.replay.load(state["replay"])
        self._load_ctx_state(state["ctx"])
        if rng:
            self.load_rng_state(state["rng"])
        if counters:
            self.episodes, self.steps, self.resets = state["episodes"], state["steps"], state["resets"]
            self.macs = dict(state["macs"])
            self.default_flag_counts = np.array(state["default_flag_counts"])

    def hashes(self):
        groups = self._groups()
        return dict(online=tensor_hash([p for g in groups for p in g]), ema=tensor_hash([e for g in self.ema for e in g]),
                    optimizer=optimizer_hash(self.opt) + optimizer_hash(self.opt_tables) + self._ctx_opt_hash())

    def _ctx_opt_hash(self):
        return ""

    def snapshot_initial(self):
        """Record the initial learned state (W resets to it)."""
        import copy
        self._initial = copy.deepcopy(self.state())

    def reset(self):
        """Every learned state back to its initial value (weights, EMA, optimisers, reservoir, context state); random
        streams and counters continue."""
        if self._initial is None:
            raise RuntimeError("snapshot_initial() was not called")
        reservoir_rng = self.rngs["reservoir"].bit_generator.state     # Replay.load would rewind it
        self.load(self._initial, rng=False, counters=False)
        self.rngs["reservoir"].bit_generator.state = reservoir_rng
        self.resets += 1


# ------------------------------------------------------------------ G and W

class GLearner(D2Base):
    arm = "G"

    def _build_context(self, g):
        self.cvec = torch.zeros(CTX_ALL, requires_grad=True)
        self.opt_ctx = torch.optim.Adam([self.cvec], lr=self.h.lr)

    def _ctx_groups(self):
        return [[self.cvec]]

    def ctx_rows(self, rows, flags):
        return self.cvec[None].expand(len(rows), -1)

    def ctx_zero_grad(self):
        self.opt_ctx.zero_grad(set_to_none=False)

    def ctx_step(self):
        clip_whole([self.cvec], self.h.grad_clip)
        self.opt_ctx.step()

    def ctx_act(self):
        return self.ema[3][0]

    def n_params(self):
        return super().n_params() + CTX_ALL

    def context_bytes(self):
        return 4 * CTX_ALL

    def _ctx_state(self):
        return dict(cvec=self.cvec.detach().clone(), opt=self.opt_ctx.state_dict())

    def _load_ctx_state(self, st):
        with torch.no_grad():
            self.cvec.copy_(st["cvec"])
        self.opt_ctx.load_state_dict(st["opt"])

    def _ctx_opt_hash(self):
        return optimizer_hash(self.opt_ctx)


class WLearner(GLearner):
    arm = "W"


# ------------------------------------------------------------------ AM

class AMLearner(D2Base):
    arm = "AM"
    factored = False

    def __init__(self, hyper, rngs, k, enc_lr, count_c=0.05):
        self.k, self.enc_lr = int(k), float(enc_lr)
        super().__init__(hyper, rngs, count_c)

    def _extend_replay(self):
        c = self.replay.capacity
        self.replay.data["hist_f"] = np.zeros((c, self.k, SUMMARY_FLOATS), np.float16)
        self.replay.data["hist_i"] = np.full((c, self.k, SUMMARY_INTS), -1, np.int8)

    def encoder_specs(self):
        if self.factored:
            return [(7, 16, CTX), (4, 16, CTX), (23, 16, CTX)]
        return [(32, 32, CTX_ALL)]

    def _build_context(self, g):
        self.enc = Encoder(self.encoder_specs(), g)
        self.enc_params = list(self.enc.parameters())
        self.opt_ctx = torch.optim.Adam(self.enc_params, lr=self.enc_lr)
        self.enc_ema = Encoder(self.encoder_specs(), np.random.default_rng(0))
        self.window_f = np.zeros((self.k, SUMMARY_FLOATS), np.float16)
        self.window_i = np.full((self.k, SUMMARY_INTS), -1, np.int8)

    def _ctx_groups(self):
        return [self.enc_params]

    def encoder_parameters(self):
        return int(sum(p.numel() for p in self.enc_params))

    def n_params(self):
        return super().n_params() + self.encoder_parameters()

    def context_bytes(self):
        return 4 * self.encoder_parameters()

    def _encode(self, module, hf, hi):
        xs = [torch.from_numpy(np.ascontiguousarray(x)) for x in summary_features(hf, hi, self.factored)]
        return module(xs)

    def ctx_rows(self, rows, flags):
        d = self.replay.data
        return self._encode(self.enc, d["hist_f"][rows], d["hist_i"][rows])

    def ctx_zero_grad(self):
        self.opt_ctx.zero_grad(set_to_none=False)

    def ctx_step(self):
        clip_whole(self.enc_params, self.h.grad_clip)
        self.opt_ctx.step()
        n = self.k * (sum(3 * (i * hh + hh * hh) + hh * o for i, hh, o in self.encoder_specs()))
        self.macs["train"] += 3 * n * (self.h.minibatch * (self.h.members + 1))

    def ctx_act(self):
        with torch.no_grad():
            for p, e in zip(self.enc_ema.parameters(), self.ema[3]):
                p.copy_(e)
            return self._encode(self.enc_ema, self.window_f[None], self.window_i[None])[0]

    def _after_outcome(self, decision, action, outcome, label):
        super()._after_outcome(decision, action, outcome, label)
        extras = dict(hist_f=self.window_f.copy(), hist_i=self.window_i.copy())
        f, i = summarize_episode(decision, action, outcome)
        self.window_f = np.concatenate([self.window_f[1:], f[None]])
        self.window_i = np.concatenate([self.window_i[1:], i[None]])
        return extras

    def _ctx_state(self):
        return dict(enc=[p.detach().clone() for p in self.enc_params], opt=self.opt_ctx.state_dict(),
                    window_f=self.window_f.copy(), window_i=self.window_i.copy())

    def _load_ctx_state(self, st):
        with torch.no_grad():
            for p, v in zip(self.enc_params, st["enc"]):
                p.copy_(v)
        self.opt_ctx.load_state_dict(st["opt"])
        self.window_f, self.window_i = st["window_f"].copy(), st["window_i"].copy()

    def _ctx_opt_hash(self):
        return optimizer_hash(self.opt_ctx)


class AMFactoredLearner(AMLearner):
    arm = "AM-factored"
    factored = True


def am_entry_bytes(k, base_record_bytes):
    return int(base_record_bytes + k * SUMMARY_BYTES)


# ------------------------------------------------------------------ code libraries (FC, FC-single, FC-K1, OC)

class CodeLearner(D2Base):
    """Code libraries. mode: 'factored' (3 slots x 6-d, FC / FC-K1), 'single' (1 slot x 18-d, FC-single),
    'oracle' (3 slots x 2 codes indexed by the lagged true level, OC)."""

    def __init__(self, hyper, rngs, mode, code_lr, tau=15.0, library=True, spawn_rule="literal", count_c=0.05,
                 max_codes=6, h=0.05, switch_posterior=0.95, default_fraction=0.2, spawn_run=2, fit_steps=20, fit_ridge=1.0,
                 code_decay=0.0):
        if mode not in ("factored", "single", "oracle") or spawn_rule not in ("literal", "lrt"):
            raise ValueError("bad code-library configuration")
        self.mode, self.code_lr, self.tau, self.library = mode, float(code_lr), float(tau), bool(library)
        self.spawn_rule, self.max_codes, self.hh, self.switch_posterior = spawn_rule, int(max_codes), float(h), float(switch_posterior)
        self.default_fraction, self.spawn_run, self.fit_steps, self.fit_ridge = float(default_fraction), int(spawn_run), int(fit_steps), float(fit_ridge)
        self.code_decay = float(code_decay)
        self.n_lib = 1 if mode == "single" else N_SLOTS
        self.dim = CTX_ALL if mode == "single" else CTX
        if mode == "oracle":
            self.max_codes = 2
            self.uses_label = True
        super().__init__(hyper, rngs, count_c)

    arm = "FC"

    def _extend_replay(self):
        self.replay.data["assign"] = np.zeros((self.replay.capacity, self.n_lib), np.int16)

    def _build_context(self, g):
        n, kmax = self.n_lib, self.max_codes
        self.codes = torch.zeros(n * kmax, self.dim, requires_grad=True)      # row = slot * kmax + position
        self.adam = LazyAdam(n * kmax, self.dim, self.code_lr, weight_decay=self.code_decay)
        self.alive = np.zeros((n, kmax), bool)
        self.ids = np.full((n, kmax), -1, np.int64)       # code id held at each library position
        self.next_id = np.zeros(n, np.int64)
        self.logpost = [np.zeros(0) for _ in range(n)]    # over alive positions (in position order)
        self.active = np.zeros(n, np.int64)               # active code id per library
        self.run = np.zeros(n, np.int64)
        self.spawns = np.zeros(n, np.int64)
        self.changes = np.zeros(n, np.int64)
        self.last_levels = None                           # OC: levels of the previous episode
        initial = 2 if self.mode == "oracle" else 1
        for s in range(n):
            for j in range(initial):
                self.alive[s, j] = True
                self.ids[s, j] = j
            self.next_id[s] = initial
            self.logpost[s] = np.full(initial, -math.log(initial))
            self.active[s] = 0
        self.diag = {}

    def _ctx_groups(self):
        return [[self.codes]]

    def n_params(self):
        return super().n_params() + int(self.codes.numel())

    def context_bytes(self):
        return 4 * int(self.codes.numel())             # the library's code values (float32), allocated capacity

    # ---- indexing
    def position_of(self, s, code_id):
        where = np.flatnonzero(self.ids[s] == code_id)
        return int(where[0]) if len(where) and self.alive[s, where[0]] else -1

    def _default(self, table, s):
        rows = np.flatnonzero(self.alive[s]) + s * self.max_codes
        return table[torch.from_numpy(rows)].mean(0)

    def _default_flags(self):
        return self.rngs["default"].random(self.n_lib) < self.default_fraction

    def ctx_rows(self, rows, flags):
        assign = self.replay.data["assign"][rows]                       # (N, n_lib) code ids
        parts = []
        self._used = np.zeros(self.codes.shape[0], bool)
        for s in range(self.n_lib):
            lookup = np.full(int(self.next_id[s]) + 1, -1, np.int64)
            for j in np.flatnonzero(self.alive[s]):
                lookup[self.ids[s, j]] = s * self.max_codes + j
            pos = lookup[np.clip(assign[:, s], 0, len(lookup) - 1)]
            pos[(assign[:, s] < 0) | (assign[:, s] >= len(lookup))] = -1
            default = self._default(self.codes, s).detach()
            if flags[s]:
                parts.append(default[None].expand(len(rows), -1))
                continue
            live = pos >= 0
            safe = torch.from_numpy(np.where(live, pos, 0))
            gathered = self.codes[safe]
            out = torch.where(torch.from_numpy(live)[:, None], gathered, default[None].expand(len(rows), -1))
            parts.append(out)
            self._used[pos[live]] = True
        return torch.cat(parts, -1)

    def ctx_zero_grad(self):
        if self.codes.grad is not None:
            self.codes.grad.zero_()

    def ctx_step(self):
        if self.codes.grad is None:
            return
        clip_whole([self.codes], self.h.grad_clip)
        self.adam.step(self.codes, self.codes.grad, self._used)

    def _code_ema(self, s, code_id):
        return self.ema[3][0][s * self.max_codes + self.position_of(s, code_id)]

    def ctx_act(self):
        table = self.ema[3][0]
        if self.mode == "single":
            return table[self.position_of(0, int(self.active[0]))]
        parts = []
        for s in range(N_SLOTS):
            if self.mode == "oracle":
                if self.last_levels is None:
                    parts.append(self._default(table, s))
                else:
                    parts.append(table[s * self.max_codes + int(self.last_levels[s])])
            else:
                parts.append(table[s * self.max_codes + self.position_of(s, int(self.active[s]))])
        return torch.cat(parts)

    # ---- evidence and the filter
    def _slot_codes(self, s):
        """(positions, EMA codes (J, dim)) of the alive codes of library s."""
        pos = np.flatnonzero(self.alive[s])
        return pos, self.ema[3][0][torch.from_numpy(pos + s * self.max_codes)]

    def _fit_gauss(self, group, d, y):
        """Ridge-regularised weighted least squares of the mean in code space around d (log-variance at d's)."""
        mean_t, logv_t, w, v = group
        logv = (logv_t + (d @ v).reshape(T, ENTITIES)).reshape(-1)
        prec = torch.exp(-logv)
        r = (y - mean_t).reshape(-1)
        a = w.T                                                              # (36, 6)
        lhs = a.T @ (prec[:, None] * a) + self.fit_ridge * torch.eye(CTX)
        rhs = a.T @ (prec * r) + self.fit_ridge * d
        return torch.linalg.solve(lhs, rhs)

    def _fit_failure(self, decision, action, outcome, cs, cp, d):
        c = d.detach().clone().requires_grad_(True)
        opt = LazyAdam(1, CTX, self.code_lr)
        for _ in range(self.fit_steps):
            with torch.enable_grad():
                ll = self.hazard_loglik_codes(decision, action, outcome, torch.cat([cs, cp, c])[None])[0]
                (g,) = torch.autograd.grad(-ll, c)
            opt.step(c.data[None], g[None], np.ones(1, bool))
        return c.detach()

    def _slot_evidence(self, s, decision, action, outcome, cs=None, cp=None):
        """(positions, loglik under each alive code (J,), loglik under the reference default, the default used)."""
        pos, codes = self._slot_codes(s)
        default = codes.mean(0)
        if self.mode == "single" or s < 2:
            if self.mode == "single":
                ys = torch.from_numpy(np.asarray(outcome.y_supply, np.float32))
                yp = torch.from_numpy(np.asarray(outcome.y_probe, np.float32))
                if self.spawn_rule == "lrt":
                    ds = self._fit_gauss(self.ema[1], default[:CTX], ys)
                    dp = self._fit_gauss(self.ema[2], default[CTX:2 * CTX], yp)
                    default = torch.cat([ds, dp, default[2 * CTX:]])
                allc = torch.cat([codes, default[None]])
                ll = gauss_loglik(self.ema[1], allc[:, :CTX], ys) + gauss_loglik(self.ema[2], allc[:, CTX:2 * CTX], yp)
                if action in INFORMATIVE_ACTIONS:
                    if self.spawn_rule == "lrt":
                        df = self._fit_failure(decision, action, outcome, default[:CTX], default[CTX:2 * CTX], default[2 * CTX:])
                        default = torch.cat([default[:2 * CTX], df])
                        allc = torch.cat([codes, default[None]])
                        ll = gauss_loglik(self.ema[1], allc[:, :CTX], ys) + gauss_loglik(self.ema[2], allc[:, CTX:2 * CTX], yp)
                    ll = ll + self.hazard_loglik_codes(decision, action, outcome, allc)
            else:
                grp = self.ema[1] if s == 0 else self.ema[2]
                y = torch.from_numpy(np.asarray(outcome.y_supply if s == 0 else outcome.y_probe, np.float32))
                if self.spawn_rule == "lrt":
                    default = self._fit_gauss(grp, default, y)
                ll = gauss_loglik(grp, torch.cat([codes, default[None]]), y)
            ll = ll.numpy().astype(np.float64)
            return pos, ll[:-1], float(ll[-1]), default
        # failure slot (factored): c_s, c_p at their slots' active codes after this episode's update
        if self.spawn_rule == "lrt":
            default = self._fit_failure(decision, action, outcome, cs, cp, default)
        allc = torch.cat([codes, default[None]])
        ctx = torch.cat([cs[None].expand(len(allc), -1), cp[None].expand(len(allc), -1), allc], -1)
        ll = self.hazard_loglik_codes(decision, action, outcome, ctx).numpy().astype(np.float64)
        return pos, ll[:-1], float(ll[-1]), default

    def _spawn(self, s, init_online, init_ema):
        """Add a code (replacing the one with the fewest assigned reservoir episodes if full); it becomes active."""
        free = np.flatnonzero(~self.alive[s])
        if len(free):
            j = int(free[0])
        else:
            counts = np.array([np.sum(self.replay.data["assign"][: self.replay.size, s] == self.ids[s, jj])
                               for jj in range(self.max_codes)])
            order = np.lexsort((self.ids[s], counts))          # fewest assigned, then oldest id
            j = int(order[0])
        new_id = int(self.next_id[s])
        self.next_id[s] += 1
        self.ids[s, j] = new_id
        self.alive[s, j] = True
        r = s * self.max_codes + j
        with torch.no_grad():
            self.codes[r] = init_online
            self.ema[3][0][r] = init_ema
        self.adam.reset_row(r)
        alive = np.flatnonzero(self.alive[s])
        self.logpost[s] = np.where(alive == j, 0.0, -np.inf)
        self.active[s] = new_id
        self.spawns[s] += 1
        return new_id

    def _filter(self, s, pos, ll_codes, ll_default, informative, default_ema):
        info = dict(margin=np.nan, spawned=False, changed=False, post_max=np.nan)
        if not informative:
            return info
        self.logpost[s] = sticky_step(self.logpost[s], ll_codes, self.hh) if len(pos) > 1 else np.zeros(1)
        best = int(np.argmax(self.logpost[s]))
        info["post_max"] = float(np.exp(self.logpost[s][best]))
        if info["post_max"] >= self.switch_posterior and int(self.ids[s, pos[best]]) != int(self.active[s]):
            self.active[s] = int(self.ids[s, pos[best]])
            self.changes[s] += 1
            info["changed"] = True
        margin = ll_default - float(np.max(ll_codes))
        info["margin"] = margin
        if not self.library:
            return info
        self.run[s] = self.run[s] + 1 if margin > self.tau else 0
        if self.run[s] >= self.spawn_run:
            online_default = self._default(self.codes, s).detach()
            if self.spawn_rule == "lrt":           # the fitted default, in both the online and the EMA library
                online_default = default_ema.clone()
            self._spawn(s, online_default, default_ema.clone())
            self.run[s] = 0
            self.changes[s] += 1
            info["spawned"] = True
            info["changed"] = True
        return info

    def _after_outcome(self, decision, action, outcome, label):
        if self.mode == "oracle":
            levels = np.array(label.levels, np.int64)
            self.last_levels = levels
            return dict(assign=levels.astype(np.int16))
        informative = action in INFORMATIVE_ACTIONS
        diag = {}
        with torch.no_grad():
            if self.mode == "single":
                pos, llc, lld, dflt = self._slot_evidence(0, decision, action, outcome)
                diag[0] = self._filter(0, pos, llc, lld, True, dflt)
            else:
                for s in (0, 1):
                    pos, llc, lld, dflt = self._slot_evidence(s, decision, action, outcome)
                    diag[s] = self._filter(s, pos, llc, lld, True, dflt)
                if informative:
                    cs, cp = self._code_ema(0, int(self.active[0])), self._code_ema(1, int(self.active[1]))
                    pos, llc, lld, dflt = self._slot_evidence(2, decision, action, outcome, cs, cp)
                    diag[2] = self._filter(2, pos, llc, lld, True, dflt)
                else:
                    diag[2] = self._filter(2, None, None, None, False, None)
        self.diag = diag
        self.macs["evidence"] += 2 * 2 * T * ENTITIES * CTX * (self.max_codes + 1)
        return dict(assign=self.active.astype(np.int16).copy())

    def library_sizes(self):
        return self.alive.sum(1)

    def _ctx_state(self):
        return dict(codes=self.codes.detach().clone(), adam=self.adam.state(), alive=self.alive.copy(), ids=self.ids.copy(),
                    next_id=self.next_id.copy(), logpost=[x.copy() for x in self.logpost], active=self.active.copy(),
                    run=self.run.copy(), spawns=self.spawns.copy(), changes=self.changes.copy(),
                    last_levels=None if self.last_levels is None else self.last_levels.copy())

    def _load_ctx_state(self, st):
        with torch.no_grad():
            self.codes.copy_(st["codes"])
        self.adam.load(st["adam"])
        self.alive, self.ids, self.next_id = st["alive"].copy(), st["ids"].copy(), st["next_id"].copy()
        self.logpost = [x.copy() for x in st["logpost"]]
        self.active, self.run = st["active"].copy(), st["run"].copy()
        self.spawns, self.changes = st["spawns"].copy(), st["changes"].copy()
        self.last_levels = None if st["last_levels"] is None else np.array(st["last_levels"])

    def _ctx_opt_hash(self):
        import hashlib
        h = hashlib.sha256()
        for t in (self.adam.m, self.adam.v, self.adam.t):
            h.update(t.numpy().tobytes())
        for a in (self.alive, self.ids, self.next_id, self.active, self.run):
            h.update(np.ascontiguousarray(a).tobytes())
        for x in self.logpost:
            h.update(np.ascontiguousarray(x).tobytes())
        return h.hexdigest()


class FCLearner(CodeLearner):
    arm = "FC"


class FCSingleLearner(CodeLearner):
    arm = "FC-single"


class FCK1Learner(CodeLearner):
    arm = "FC-K1"


class OCLearner(CodeLearner):
    arm = "OC"


# ------------------------------------------------------------------ factory

def make_d2_learner(arm, hyper, seeds, knobs, count_c=0.05):
    """seeds: dict stream -> int seed (one per RNG_STREAMS entry). knobs: the arm's declared knobs."""
    rngs = {k: np.random.default_rng(seeds[k]) for k in RNG_STREAMS}
    if arm == "G":
        lrn = GLearner(hyper, rngs, count_c)
    elif arm == "W":
        lrn = WLearner(hyper, rngs, count_c)
    elif arm == "AM":
        lrn = AMLearner(hyper, rngs, knobs["k"], knobs["enc_lr"], count_c)
    elif arm == "AM-factored":
        lrn = AMFactoredLearner(hyper, rngs, knobs["k"], knobs["enc_lr"], count_c)
    elif arm in ("FC", "FC-single", "FC-K1", "OC"):
        extra = {k: knobs[k] for k in ("code_decay", "default_fraction") if k in knobs}
        if arm == "FC":
            lrn = FCLearner(hyper, rngs, "factored", knobs["code_lr"], knobs["tau"], True, knobs.get("spawn_rule", "literal"), count_c, **extra)
        elif arm == "FC-single":
            lrn = FCSingleLearner(hyper, rngs, "single", knobs["code_lr"], knobs["tau"], True, knobs.get("spawn_rule", "literal"), count_c, **extra)
        elif arm == "FC-K1":
            lrn = FCK1Learner(hyper, rngs, "factored", knobs["code_lr"], knobs.get("tau", 15.0), False, "literal", count_c, **extra)
        else:
            lrn = OCLearner(hyper, rngs, "oracle", knobs["code_lr"], 15.0, False, "literal", count_c, **extra)
    else:
        raise ValueError(arm)
    lrn.snapshot_initial()
    return lrn
