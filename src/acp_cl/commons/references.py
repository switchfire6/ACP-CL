"""Evaluator-only references for the commons world: O (context-known oracle),
the exact-sensor / MC-pattern episode likelihood, the run-length-augmented
semi-Markov filter and I (the same-information ideal agent).

Key structural fact used throughout (and verified in D0): after the single
departure at t = 0, the three entities evolve independently (own supplies,
own clipping and maintenance; probes never enter reserves). Hence

* joint survival = product of per-entity survival indicators, and the MC
  estimate of the joint survival probability is the product of per-entity MC
  survival fractions computed on independent per-entity noise (unbiased);
* the failure pattern likelihood factorises over entities given the sensors.

Each entity is simulated once per distinct "variant" (supply mean row,
post-departure reserve, arrival profile) and reused across actions and
contexts with common random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import (action_table, n_actions,
                    N_CONTEXTS, N_ENTITIES, OBS_NOISE, SUPPLY_NOISE, SURVIVED, T,
                    arrival_profile, channel_params, entity_kernel, factors, probe_arrivals,
                    seed_for, supply_means)

LOG_2PI = float(np.log(2 * np.pi))


# ---------------------------------------------------------------- variants

def entity_variants(params, reserves, cue1, contexts, actions=None):
    """Distinct per-entity situations for one episode.

    Returns (keys, index) where keys[i] is a list of (a_level, x0, arrival(T,))
    for entity i and index[(c, a)] = (v0, v1, v2) variant ids per entity.
    """
    if actions is None:
        actions = range(n_actions(params))
    table_ch, table_donor, table_rec, table_amount = action_table(params)
    mu_a = [(lambda c: int(factors(c)[0]))] * 2 + [lambda c: 0]
    keys = [dict() for _ in range(N_ENTITIES)]
    lists = [[] for _ in range(N_ENTITIES)]
    index = {}
    reserves = np.asarray(reserves, dtype=float)
    for c in contexts:
        eff, delay = channel_params(params, np.array([c]), np.array([cue1]))
        for a in actions:
            ids = []
            if a == 0:
                sent, recipient, donor, prof = 0.0, -1, -1, None
            else:
                ch, donor, recipient = table_ch[a], table_donor[a], table_rec[a]
                sent = min(table_amount[a], reserves[donor])
                prof = arrival_profile(np.array([sent * eff[0, ch]]), np.array([delay[0, ch]]))[0]
            for i in range(N_ENTITIES):
                x0 = reserves[i] - sent if i == donor else reserves[i]
                arr = prof if i == recipient else None
                key = (mu_a[i](c), float(x0), None if arr is None else arr.tobytes())
                if key not in keys[i]:
                    keys[i][key] = len(lists[i])
                    lists[i].append((mu_a[i](c), x0, arr))
                ids.append(keys[i][key])
            index[(c, a)] = tuple(ids)
    return lists, index


def _variant_arrays(variants):
    a_levels = np.array([v[0] for v in variants])
    x0 = np.array([v[1] for v in variants], dtype=float)
    arr = np.zeros((len(variants), T))
    for j, v in enumerate(variants):
        if v[2] is not None:
            arr[j] = v[2]
    has_arrival = any(v[2] is not None for v in variants)
    return a_levels, x0, arr, has_arrival


# ---------------------------------------------------------------- O / value tables

@dataclass
class ValueTable:
    value: np.ndarray   # (n, n_contexts, 7) estimated P(joint survival)
    se: np.ndarray      # (n, n_contexts, 7) MC standard error of the product estimator
    contexts: tuple


def product_se(p, m):
    """SE of the product of independent per-entity fractions (exact variance formula)."""
    second = np.prod(p ** 2 + p * (1 - p) / m, axis=-1)
    return np.sqrt(np.maximum(second - np.prod(p, axis=-1) ** 2, 0.0))


def episode_values(params, reserves, cue1, contexts, z, joint=False):
    """P(joint survival | context, action) for one episode with CRN draws z (M, T, 3).

    Returns (value (len(contexts), 7), se (…)) and, if joint=True, also the
    joint-indicator estimate computed from the same draws (for verification).
    """
    mu = supply_means(params)
    m = z.shape[0]
    variants, index = entity_variants(params, reserves, cue1, contexts)
    frac, alive_all = [], []
    for i in range(N_ENTITIES):
        a_levels, x0, arr, has_arr = _variant_arrays(variants[i])
        supplies = mu[a_levels, :, i][:, None, :] + SUPPLY_NOISE * z[None, :, :, i]
        alive = entity_kernel(x0[:, None], arr[:, None, :] if has_arr else None, supplies)
        frac.append(alive.mean(axis=1))
        if joint:
            alive_all.append(alive)
    na = n_actions(params)
    per_entity = np.empty((len(contexts), na, N_ENTITIES))
    jv = np.empty((len(contexts), na)) if joint else None
    for ci, c in enumerate(contexts):
        for a in range(na):
            ids = index[(c, a)]
            per_entity[ci, a] = [frac[i][ids[i]] for i in range(N_ENTITIES)]
            if joint:
                jv[ci, a] = (alive_all[0][ids[0]] & alive_all[1][ids[1]] & alive_all[2][ids[2]]).mean()
    value = per_entity.prod(axis=-1)
    se = product_se(per_entity, m)
    return (value, se, jv) if joint else (value, se)


def draws(seed, m):
    return np.random.default_rng(seed).standard_normal((m, T, N_ENTITIES))


def value_tables(params, episodes, contexts, m, seed, channel, offset=0):
    """Value tables for a batch of episodes; draws seeded per (seed, channel, episode)."""
    n = len(episodes)
    value = np.empty((n, len(contexts), n_actions(params)))
    se = np.empty_like(value)
    for e in range(n):
        z = draws(seed_for(seed, channel, offset + e), m)
        value[e], se[e] = episode_values(params, episodes.reserves[e], int(episodes.cue[e, 0]),
                                         contexts, z)
    return ValueTable(value, se, tuple(contexts))


# ---------------------------------------------------------------- likelihood

POST_VAR = 1.0 / (1.0 / SUPPLY_NOISE ** 2 + 1.0 / OBS_NOISE ** 2)
MARG_VAR = SUPPLY_NOISE ** 2 + OBS_NOISE ** 2


def supply_loglik(params, y_supply):
    """log p(y_supply | A) for A in {0,1}: exact Gaussian marginal. Returns (n, 2) and per-entity (n, 2, 3)."""
    mu = supply_means(params)
    resid = y_supply[:, None] - mu[None]  # (n, 2, T, 3)
    per = -0.5 * (resid ** 2 / MARG_VAR + np.log(MARG_VAR) + LOG_2PI)
    per_entity = per.sum(axis=2)
    return per_entity.sum(axis=2), per_entity


def probe_loglik(params, y_probe, cue1, probes_on=True):
    """log p(y_probe | c) for all 8 contexts (exact Gaussian). Returns (n, 8)."""
    n = len(y_probe)
    out = np.empty((n, N_CONTEXTS))
    for c in range(N_CONTEXTS):
        eff, delay = channel_params(params, np.full(n, c), cue1)
        mean = probe_arrivals(params, eff, delay, probes_on)
        resid = y_probe - mean
        out[:, c] = (-0.5 * (resid ** 2 / OBS_NOISE ** 2 + np.log(OBS_NOISE ** 2) + LOG_2PI)).sum(axis=(1, 2))
    return out


def posterior_supply(params, y_supply_entity, a_level, entity):
    """Conjugate posterior mean (T,) and variance of one entity's supplies given its sensors."""
    mu = supply_means(params)[a_level, :, entity]
    mean = POST_VAR * (mu / SUPPLY_NOISE ** 2 + y_supply_entity / OBS_NOISE ** 2)
    return mean, POST_VAR


@dataclass
class PatternSettings:
    k: int = 2048            # posterior samples per entity variant (first pass)
    k_refine: int = 32768    # second pass for entries with fewer than min_count hits
    min_count: int = 16
    floor_hits: float = 0.5  # floor = floor_hits / k_refine if still zero


def pattern_histograms(params, reserves, cue1, y_supply, contexts, actions, seed, settings):
    """Per entity, per variant: histogram of failure step (13 bins) under the sensor posterior.

    Returns (variants, index, hist list per entity (V_i, 14) counts, k per variant).
    """
    variants, index = entity_variants(params, reserves, cue1, contexts, actions)
    rng = np.random.default_rng(seed)
    hists, ks = [], []
    for i in range(N_ENTITIES):
        a_levels, x0, arr, has_arr = _variant_arrays(variants[i])
        u = rng.standard_normal((settings.k, T))  # CRN across variants of this entity
        means = np.stack([posterior_supply(params, y_supply[:, i], a, i)[0] for a in (0, 1)])
        supplies = means[a_levels][:, None, :] + np.sqrt(POST_VAR) * u[None]
        step = entity_kernel(x0[:, None], arr[:, None, :] if has_arr else None, supplies, want_step=True)
        hist = np.zeros((len(variants[i]), SURVIVED + 1), dtype=np.int64)
        for j in range(len(variants[i])):
            hist[j] = np.bincount(step[j], minlength=SURVIVED + 1)
        hists.append(hist)
        ks.append(np.full(len(variants[i]), settings.k))
    return variants, index, hists, ks, rng


def pattern_loglik(params, reserves, cue1, y_supply, fail_steps, contexts, actions, seed,
                   settings=PatternSettings(), stats=None):
    """log P(failure pattern | sensors, c, a) for one episode.

    fail_steps: dict action -> (3,) observed failure steps for that action.
    Returns array (len(actions), len(contexts)).
    """
    variants, index, hists, ks, rng = pattern_histograms(
        params, reserves, cue1, y_supply, contexts, actions, seed, settings)
    # Refinement pass for low-count needed entries.
    for i in range(N_ENTITIES):
        need = set()
        for a in actions:
            obs = int(fail_steps[a][i])
            for c in contexts:
                j = index[(c, a)][i]
                if hists[i][j, obs] < settings.min_count and ks[i][j] == settings.k:
                    need.add(j)
        if need:
            need = sorted(need)
            sub = [variants[i][j] for j in need]
            a_levels, x0, arr, has_arr = _variant_arrays(sub)
            u = rng.standard_normal((settings.k_refine, T))
            means = np.stack([posterior_supply(params, y_supply[:, i], a, i)[0] for a in (0, 1)])
            supplies = means[a_levels][:, None, :] + np.sqrt(POST_VAR) * u[None]
            step = entity_kernel(x0[:, None], arr[:, None, :] if has_arr else None, supplies,
                                 want_step=True)
            for r, j in enumerate(need):
                hists[i][j] = np.bincount(step[r], minlength=SURVIVED + 1)
                ks[i][j] = settings.k_refine
            if stats is not None:
                stats["refined"] = stats.get("refined", 0) + len(need)
    out = np.zeros((len(actions), len(contexts)))
    for ai, a in enumerate(actions):
        for ci, c in enumerate(contexts):
            ids = index[(c, a)]
            for i in range(N_ENTITIES):
                obs = int(fail_steps[a][i])
                hits = hists[i][ids[i], obs]
                k = ks[i][ids[i]]
                if hits == 0:
                    if stats is not None:
                        stats["floored"] = stats.get("floored", 0) + 1
                    p = settings.floor_hits / k
                else:
                    p = hits / k
                out[ai, ci] += np.log(p)
    if stats is not None:
        stats["entries"] = stats.get("entries", 0) + len(actions) * len(contexts) * N_ENTITIES
    return out


def episode_loglik(params, episodes, y_supply, y_probe, fail_steps_by_action, actions, seed,
                   settings=PatternSettings(), probes_on=True, stats=None):
    """Full per-episode log-likelihoods, shape (n, len(actions), 8), split into components.

    fail_steps_by_action: array (n, 7, 3) of observed failure steps for every action
    (only the entries for ``actions`` are used).
    """
    n = len(episodes)
    contexts = list(range(N_CONTEXTS))
    sup, _ = supply_loglik(params, y_supply)                   # (n, 2)
    a_of_c = np.array([int(factors(c)[0]) for c in contexts])
    sensor = sup[:, a_of_c] + probe_loglik(params, y_probe, episodes.cue[:, 0], probes_on)
    pattern = np.empty((n, len(actions), N_CONTEXTS))
    for e in range(n):
        fs = {a: fail_steps_by_action[e, a] for a in actions}
        pattern[e] = pattern_loglik(params, episodes.reserves[e], int(episodes.cue[e, 0]), y_supply[e],
                                    fs, contexts, list(actions), seed_for(seed, "pattern", e),
                                    settings, stats)
    return sensor, pattern


# ---------------------------------------------------------------- filter

@dataclass
class ScheduleModel:
    """Semi-Markov model of the schedule as seen by I.

    run_pmf[L-1] = P(run length = L), L = 1..Lmax. q_phase[p] is the (8, 8)
    switch matrix (row c -> next context, zero diagonal) in phase p; phase 1
    starts at episode ``phase_start`` with a forced switch.
    """

    run_pmf: np.ndarray
    initial: np.ndarray
    q_phase: tuple
    phase_start: int | None = None

    @property
    def lmax(self):
        return len(self.run_pmf)

    def hazard(self):
        survival = np.cumsum(self.run_pmf[::-1])[::-1]  # P(L >= r), r = 1..Lmax
        with np.errstate(invalid="ignore", divide="ignore"):
            h = np.where(survival > 0, self.run_pmf / np.maximum(survival, 1e-300), 1.0)
        h[-1] = 1.0
        return np.clip(h, 0.0, 1.0)


class RunLengthFilter:
    """Exact forward filter over (context, age of current run)."""

    def __init__(self, model: ScheduleModel):
        self.model = model
        self.h = model.hazard()
        self.t = 0
        self.state = None  # (8, Lmax) posterior after the last update

    def predict(self):
        """Predictive distribution over (context, age) for episode self.t."""
        m = self.model
        if self.t == 0:
            pred = np.zeros((N_CONTEXTS, m.lmax))
            pred[:, 0] = m.initial
            return pred
        phase = 1 if (m.phase_start is not None and self.t >= m.phase_start) else 0
        q = m.q_phase[phase]
        s = self.state
        if m.phase_start is not None and self.t == m.phase_start:
            switch_mass = s.sum(axis=1)
            stay = np.zeros_like(s)
        else:
            switch_mass = (s * self.h[None, :]).sum(axis=1)
            stay = np.zeros_like(s)
            stay[:, 1:] = s[:, :-1] * (1.0 - self.h[None, :-1])
        pred = stay
        pred[:, 0] += switch_mass @ q
        return pred

    def weights(self):
        pred = self.predict()
        w = pred.sum(axis=1)
        return w / w.sum(), pred

    def update(self, loglik, pred=None):
        if pred is None:
            pred = self.predict()
        lik = np.exp(loglik - np.max(loglik))
        post = pred * lik[:, None]
        total = post.sum()
        if not total > 0:
            raise FloatingPointError("filter underflow")
        self.state = post / total
        self.t += 1
        return self.state.sum(axis=1)


def brute_force_posterior(model: ScheduleModel, logliks):
    """P(c_n | x_1..x_n) by enumerating every context sequence (tests only)."""
    import itertools
    n = len(logliks)
    lik = np.exp(np.asarray(logliks) - np.max(logliks, axis=1, keepdims=True))
    pmf = model.run_pmf
    surv = np.cumsum(pmf[::-1])[::-1]
    post = np.zeros(N_CONTEXTS)
    for seq in itertools.product(range(N_CONTEXTS), repeat=n):
        p = model.initial[seq[0]]
        if p == 0:
            continue
        run = 1
        ok = True
        for t in range(1, n):
            forced = model.phase_start is not None and t == model.phase_start
            phase = 1 if (model.phase_start is not None and t >= model.phase_start) else 0
            if seq[t] == seq[t - 1] and not forced:
                run += 1
                if run > len(pmf) or surv[run - 1] == 0:
                    ok = False
                    break
                continue
            if seq[t] == seq[t - 1] and forced:
                ok = False
                break
            # run of length `run` ended (or forced end)
            if not forced:
                p *= pmf[run - 1]
            else:
                p *= surv[run - 1]
            p *= model.q_phase[phase][seq[t - 1], seq[t]]
            run = 1
            if p == 0:
                ok = False
                break
        if not ok:
            continue
        p *= surv[run - 1]
        for t in range(n):
            p *= lik[t, seq[t]]
        post[seq[-1]] += p
    return post / post.sum()


# ---------------------------------------------------------------- I

def myopic_action(weights, values):
    """argmax_a sum_c w_c V_c[a] with values (8, 7)."""
    return int(np.argmax(weights @ values))
