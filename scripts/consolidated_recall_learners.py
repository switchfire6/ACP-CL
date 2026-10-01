"""Learners for docs/consolidated_recall_protocol.md (Step C: consolidated recall).

Project source files and the earlier study scripts are imported, never edited.

* ``pattern_loglik``: the shared evidence statistic. For monotone survival
  forecasts (p4, p8, p12) of the PERFORMED action, the four joint lifetime
  categories are (1-p4, p4-p8, p8-p12, p12); each is floored and renormalised,
  and l(model, batch) is the sum over the batch's records of the log-probability
  of the observed category (= number of horizons survived).
* ``BiasCorrectedEMA``: a <- d a + (1-d) theta, n <- n+1, phi = a / (1-d^n).
  Created either at count 0 (start rule / EMA-global: a = 0, phi = theta_0 until
  the first update, no initial-weight residue) or at count 1 (spawn: a =
  (1-d) theta, phi = theta).
* ``ConsolidatedRecall``: the CR mechanism, steps 3-7 of the protocol (the runner
  owns steps 1-2: forecasts are made first, then l is computed from the
  labels, then ``observe`` runs the filter, recall, pending and spawn rules, then
  the unchanged ``RepresentationLearner.train`` runs with a step post-hook that
  consolidates phi_{k*} after every optimiser step while not pending).
  Variants: CR, CR-K1, CR-no-recall, CR-oracle-mode, CR-oracle-law and the
  check-only CR-single (recall, pending and spawn disabled, one snapshot).
* ``GlobalEMAs``: EMA-global, one or more bias-corrected EMAs of the ON-R256
  weights, updated by a read-only post-hook (the fast weights are untouched).
* filter helpers and float64 references used by the smoke checks.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import itertools  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VARIANTS = {
    # K_max, index source, recall enabled, recall copies theta, pending enabled, spawn enabled
    'CR': dict(K_max=8, index='inferred', recall=True, copy=True, pending=True, spawn=True),
    'CR-K1': dict(K_max=1, index='inferred', recall=False, copy=True, pending=True, spawn=True),
    'CR-no-recall': dict(K_max=8, index='inferred', recall=True, copy=False, pending=True, spawn=True),
    'CR-oracle-mode': dict(K_max=2, index='mode', recall=True, copy=True, pending=True, spawn=False),
    'CR-oracle-law': dict(K_max=5, index='law', recall=True, copy=True, pending=True, spawn=False),
    # check only: recall, pending and spawn disabled with a single snapshot (must equal EMA-global)
    'CR-single': dict(K_max=1, index='inferred', recall=False, copy=False, pending=False, spawn=False),
}
PRIVILEGED = {'CR-oracle-mode': 'PRIVILEGED: snapshot index = true mode of the previous batch (lagged)',
              'CR-oracle-law': 'PRIVILEGED: snapshot index = true law of the previous batch (lagged)'}


# ------------------------------------------------------------------ evidence

def pattern_probabilities(probs, actions, floor):
    """(n, 4) floored, renormalised category probabilities for the performed actions."""
    p = np.asarray(probs, dtype=np.float64)[np.arange(len(actions)), np.asarray(actions, dtype=np.int64)]
    q = np.stack([1. - p[:, 0], p[:, 0] - p[:, 1], p[:, 1] - p[:, 2], p[:, 2]], axis=1)
    q = np.maximum(q, floor)
    return q / q.sum(axis=1, keepdims=True)


def observed_patterns(survival):
    """Number of horizons survived; survival must be monotone (1..1 0..0)."""
    s = np.asarray(survival).astype(np.int64)
    if np.any(np.diff(s, axis=1) > 0):
        raise AssertionError('non-monotone survival labels')
    return s.sum(axis=1)


def pattern_loglik(probs, actions, survival, floor):
    q = pattern_probabilities(probs, actions, floor)
    pattern = observed_patterns(survival)
    return float(np.log(q[np.arange(len(pattern)), pattern]).sum())


def pattern_loglik_direct(probs, actions, survival, floor):
    """Independent scalar float64 computation (smoke check)."""
    total = 0.
    for i in range(len(actions)):
        p4, p8, p12 = (float(v) for v in probs[i][int(actions[i])])
        cats = [max(1. - p4, floor), max(p4 - p8, floor), max(p8 - p12, floor), max(p12, floor)]
        s = [int(v) for v in survival[i]]
        k = 0 if s == [0, 0, 0] else 1 if s == [1, 0, 0] else 2 if s == [1, 1, 0] else 3 if s == [1, 1, 1] else None
        if k is None:
            raise AssertionError('non-monotone survival')
        total += math.log(cats[k] / sum(cats))
    return total


# -------------------------------------------------------------------- filter

def filter_step(posterior, ell, h):
    """log p_t(k) = log[(1-h) p_{t-1}(k) + h/K] + l_k, normalised (log space)."""
    posterior = np.asarray(posterior, dtype=np.float64)
    K = len(posterior)
    m = np.log((1. - h) * posterior + h / K) + np.asarray(ell, dtype=np.float64)
    m = m - m.max()
    w = np.exp(m)
    return w / w.sum()


def forward_matrix(initial, ells, h):
    """Probability-space HMM forward pass with the sticky transition matrix (float64)."""
    alpha = np.asarray(initial, dtype=np.float64)
    K = len(alpha)
    T = (1. - h) * np.eye(K) + h / K * np.ones((K, K))
    out = []
    for ell in ells:
        ell = np.asarray(ell, dtype=np.float64)
        alpha = (T.T @ alpha) * np.exp(ell - ell.max())
        alpha = alpha / alpha.sum()
        out.append(alpha)
    return np.array(out)


def forward_bruteforce(initial, ells, h):
    """Enumerate every hidden path (K^(t+1) paths) and marginalise: p(k_t | l_1..t)."""
    initial = np.asarray(initial, dtype=np.float64)
    K = len(initial)
    T = (1. - h) * np.eye(K) + h / K * np.ones((K, K))
    out = []
    for t in range(1, len(ells) + 1):
        marg = np.zeros(K)
        for path in itertools.product(range(K), repeat=t + 1):
            w = initial[path[0]]
            for s in range(1, t + 1):
                w *= T[path[s - 1], path[s]] * math.exp(ells[s - 1][path[s]])
            marg[path[-1]] += w
        out.append(marg / marg.sum())
    return np.array(out)


# -------------------------------------------------------------------- EMA

class BiasCorrectedEMA:
    """a <- d a + (1-d) theta; n <- n+1; phi = a / (1 - d^n). Float32 like the network."""

    def __init__(self, params, decay, count):
        if not 0. < decay < 1.:
            raise ValueError('decay must lie in (0, 1)')
        self.decay = float(decay)
        with torch.no_grad():
            if count == 0:
                self.average = [torch.zeros_like(p) for p in params]
                self.base = [p.detach().clone() for p in params]
            elif count == 1:
                self.average = [p.detach().clone().mul_(1. - self.decay) for p in params]
                self.base = None
            else:
                raise ValueError('snapshots are created at count 0 (start) or 1 (spawn)')
        self.count = int(count)
        self.steps = 0

    @torch.no_grad()
    def update(self, params):
        for a, p in zip(self.average, params):
            a.mul_(self.decay).add_(p.detach(), alpha=1. - self.decay)
        self.count += 1
        self.steps += 1
        self.base = None

    @torch.no_grad()
    def weights(self):
        if self.count == 0:
            return [b.clone() for b in self.base]
        correction = 1. - self.decay ** self.count
        return [a / correction for a in self.average]

    def nbytes(self):
        n = sum(a.numel() * a.element_size() for a in self.average)
        return n + (sum(b.numel() * b.element_size() for b in self.base) if self.base is not None else 0)

    def state(self):
        return dict(decay=self.decay, count=self.count, steps=self.steps,
                    average=[a.detach().clone() for a in self.average],
                    base=None if self.base is None else [b.detach().clone() for b in self.base])

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.decay, obj.count, obj.steps = state['decay'], state['count'], state['steps']
        obj.average = [a.clone() for a in state['average']]
        obj.base = None if state['base'] is None else [b.clone() for b in state['base']]
        return obj


def ema_closed_form(thetas, decay, count0):
    """float64 closed form. count0 = 0: thetas[0] is theta_0 (no weight); count0 = 1: thetas[0] is the
    spawn value and counts as one sample. Returns phi after len(thetas)-1 updates."""
    m = len(thetas) - 1
    result = {}
    for name in thetas[0]:
        num = torch.zeros_like(thetas[0][name], dtype=torch.float64)
        start = 1 if count0 == 0 else 0
        for j in range(start, m + 1):
            num = num + (1 - decay) * decay ** (m - j) * thetas[j][name].double()
        n = m + count0
        result[name] = thetas[0][name].double() if n == 0 else num / (1 - decay ** n)
    return result


@torch.no_grad()
def load_weights(model, tensors):
    for p, v in zip(model.parameters(), tensors):
        p.copy_(v)


def parameters(learner):
    return list(learner.model.parameters())


# ------------------------------------------------------------ EMA-global

class GlobalEMAs:
    """Bias-corrected EMAs of the online weights for several decays; read-only post-hook."""

    def __init__(self, learner, decays, states=None):
        self.decays = [float(d) for d in decays]
        params = parameters(learner)
        if states is None:
            self.emas = [BiasCorrectedEMA(params, d, 0) for d in self.decays]
        else:
            self.emas = [BiasCorrectedEMA.from_state(s) for s in states]
            if [e.decay for e in self.emas] != self.decays:
                raise ValueError('EMA decays changed')
        self.params = params
        self.handle = learner.optimizer.register_step_post_hook(self._hook)

    def _hook(self, optimizer, args, kwargs):
        for e in self.emas:
            e.update(self.params)

    def remove(self):
        self.handle.remove()

    def state(self):
        return [e.state() for e in self.emas]


# ------------------------------------------------------ consolidated recall

class ConsolidatedRecall:
    """CR and its variants. The runner calls, per batch t:
         source = cr.forecast_source()            # step 1 (theta if pending else phi_k*)
         ... forecasts of theta and of every snapshot, then l from the labels (step 2) ...
         event = cr.observe(t, l_theta, l_snapshots, index)   # steps 3-6
         cr.begin_training(mode, law); learner.train(batch); cr.end_training()   # step 7
    """

    def __init__(self, learner, variant, decay, tau, mechanism, state=None):
        if variant not in VARIANTS:
            raise ValueError(f'unknown CR variant {variant}')
        self.variant = variant
        self.spec = dict(VARIANTS[variant])
        if variant in ('CR', 'CR-no-recall') and self.spec['K_max'] != mechanism['K_max']:
            raise ValueError('K_max differs from the protocol')
        self.learner = learner
        self.params = parameters(learner)
        self.decay, self.tau = float(decay), float(tau)
        self.h = float(mechanism['h'])
        self.threshold = float(mechanism['recall_posterior'])
        self.spawn_run = int(mechanism['spawn_run'])
        self.floor = float(mechanism['pattern_floor'])
        if state is None:
            start = dict(id=0, ema=BiasCorrectedEMA(self.params, self.decay, 0), created_batch=0,
                         index=None, by_mode={}, by_law={}, kstar_batches=0)
            self.snapshots = [start]
            self.posterior = np.ones(1)
            self.kstar = 0
            self.pending = False
            self.run = 0
            self.next_id = 1
            self.retired = []
            self.counters = dict(recalls=0, spawns=0, replacements=0, kstar_changes=0, consolidations=0,
                                 pending_batches=0, recall_and_spawn_same_batch=0, adam_resets=0,
                                 oracle_creations=0)
        else:
            self.snapshots = [dict(s, ema=BiasCorrectedEMA.from_state(s['ema'])) for s in state['snapshots']]
            self.posterior = np.array(state['posterior'], dtype=np.float64)
            self.kstar, self.pending, self.run = state['kstar'], state['pending'], state['run']
            self.next_id, self.counters = state['next_id'], dict(state['counters'])
            self.retired = [dict(r) for r in state['retired']]
        self.training_labels = None
        self.handle = learner.optimizer.register_step_post_hook(self._hook)

    # ---------------------------------------------------------------- state
    def state(self):
        return dict(variant=self.variant, decay=self.decay, tau=self.tau,
                    snapshots=[dict({k: v for k, v in s.items() if k != 'ema'}, ema=s['ema'].state())
                               for s in self.snapshots],
                    posterior=self.posterior.copy(), kstar=self.kstar, pending=self.pending, run=self.run,
                    next_id=self.next_id, counters=dict(self.counters), retired=[dict(r) for r in self.retired])

    def summary(self, final_batch):
        """Per-snapshot descriptive record (live and replaced)."""
        live = [dict(id=s['id'], created_batch=s['created_batch'], index=s['index'], count=s['ema'].count,
                     consolidation_steps=s['ema'].steps, by_mode=dict(s['by_mode']), by_law=dict(s['by_law']),
                     kstar_batches=s['kstar_batches'], live=True) for s in self.snapshots]
        return live + [dict(r, live=False) for r in self.retired]

    def remove(self):
        self.handle.remove()

    def ids(self):
        return [s['id'] for s in self.snapshots]

    def snapshot_weights(self):
        return [s['ema'].weights() for s in self.snapshots]

    def nbytes(self):
        return sum(s['ema'].nbytes() for s in self.snapshots)

    # --------------------------------------------------------------- step 1
    def forecast_source(self):
        """-1 = theta (pending), else the slot of k*."""
        return -1 if self.pending else self.kstar

    # ------------------------------------------------------------ steps 3-6
    @torch.no_grad()
    def _recall(self, slot):
        """theta <- phi_slot; Adam first moment reset to zero; second moment and step kept."""
        load_weights(self.learner.model, self.snapshots[slot]['ema'].weights())
        for p in self.params:
            st = self.learner.optimizer.state.get(p)
            if st and 'exp_avg' in st:
                st['exp_avg'].zero_()
        self.counters['adam_resets'] += 1

    def _new_snapshot(self, batch, index=None):
        return dict(id=self.next_id, ema=BiasCorrectedEMA(self.params, self.decay, 1), created_batch=batch,
                    index=index, by_mode={}, by_law={}, kstar_batches=0)

    def observe(self, batch, ell_theta, ell_snapshots, index=None):
        ell = [float(v) for v in ell_snapshots]
        if len(ell) != len(self.snapshots):
            raise ValueError('one likelihood per snapshot is required')
        event = dict(batch=batch, K_before=len(self.snapshots), kstar_before=self.snapshots[self.kstar]['id'],
                     recalled=False, spawned=False, replaced_id=None, created=False, copied=False)
        before = self.snapshots[self.kstar]['id']
        mode = self.spec['index']
        if mode == 'inferred':
            # step 3: filter
            self.posterior = filter_step(self.posterior, ell, self.h)
            # step 4: recall
            if self.spec['recall']:
                candidates = [j for j in range(len(self.snapshots))
                              if j != self.kstar and self.posterior[j] >= self.threshold]
                if candidates:
                    self.kstar = candidates[0]
                    event['recalled'] = True
                    self.counters['recalls'] += 1
                    if self.spec['copy']:
                        self._recall(self.kstar)
                        event['copied'] = True
        else:
            if index is None:
                raise ValueError('oracle arms need the (lagged) true index')
            slots = {s['index']: j for j, s in enumerate(self.snapshots) if s['index'] is not None}
            if index in slots:
                if slots[index] != self.kstar:
                    self.kstar = slots[index]
                    event['recalled'] = True
                    self.counters['recalls'] += 1
                    self._recall(self.kstar)
                    event['copied'] = True
            elif self.snapshots[0]['index'] is None:      # start snapshot takes the first index seen
                self.snapshots[0]['index'] = index
                self.kstar = 0
            else:
                if len(self.snapshots) >= self.spec['K_max']:
                    raise AssertionError('oracle index space exceeded')
                self.snapshots.append(self._new_snapshot(batch, index))
                self.next_id += 1
                self.kstar = len(self.snapshots) - 1
                ell.append(float(ell_theta))                # identical weights: phi = theta
                event['created'] = True
                self.counters['oracle_creations'] += 1
        # step 5: pending (paired, same batch)
        self.pending = bool(self.spec['pending'] and (ell_theta - ell[self.kstar] > self.tau))
        # step 6: spawn
        exceed = bool(ell_theta - max(ell[:event['K_before']]) > self.tau)
        self.run = self.run + 1 if exceed else 0
        event['exceed'] = exceed
        if self.spec['spawn'] and mode == 'inferred' and self.run >= self.spawn_run:
            snapshot = self._new_snapshot(batch)
            self.next_id += 1
            if len(self.snapshots) >= self.spec['K_max']:
                steps = [s['ema'].steps for s in self.snapshots]
                slot = int(np.argmin(steps))                  # fewest consolidation steps; first on ties
                old = self.snapshots[slot]
                event['replaced_id'] = old['id']
                self.retired.append(dict(id=old['id'], created_batch=old['created_batch'], index=old['index'],
                                         count=old['ema'].count, consolidation_steps=old['ema'].steps,
                                         by_mode=dict(old['by_mode']), by_law=dict(old['by_law']),
                                         kstar_batches=old['kstar_batches'], replaced_batch=batch))
                self.snapshots[slot] = snapshot
                self.counters['replacements'] += 1
            else:
                self.snapshots.append(snapshot)
                slot = len(self.snapshots) - 1
            self.kstar = slot
            self.posterior = np.zeros(len(self.snapshots))
            self.posterior[slot] = 1.
            self.pending = False
            self.run = 0
            event['spawned'] = True
            self.counters['spawns'] += 1
            if event['recalled']:
                self.counters['recall_and_spawn_same_batch'] += 1
        if mode == 'inferred' and len(self.posterior) != len(self.snapshots):
            raise AssertionError('posterior size differs from the snapshot count')
        after = self.snapshots[self.kstar]['id']
        event['kstar_changed'] = after != before
        self.counters['kstar_changes'] += int(after != before)
        self.counters['pending_batches'] += int(self.pending)
        event.update(kstar_after=after, pending_after=self.pending, K_after=len(self.snapshots))
        return event

    # --------------------------------------------------------------- step 7
    def begin_training(self, mode, law):
        self.training_labels = (str(mode), str(law))
        self.snapshots[self.kstar]['kstar_batches'] += 1

    def end_training(self):
        self.training_labels = None

    def _hook(self, optimizer, args, kwargs):
        if self.pending:
            return
        s = self.snapshots[self.kstar]
        s['ema'].update(self.params)
        self.counters['consolidations'] += 1
        if self.training_labels is not None:
            mode, law = self.training_labels
            s['by_mode'][mode] = s['by_mode'].get(mode, 0) + 1
            s['by_law'][law] = s['by_law'].get(law, 0) + 1
