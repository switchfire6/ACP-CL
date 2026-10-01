"""Separability, relevance and value-of-information calculators and the
stratified semi-Markov schedule generator for the commons world (Step D0).

Separability of a context pair (i, j) is measured on a pool of episodes that
share every random draw across the two contexts (common random numbers):

* per-episode log-likelihood ratios L = log p(x | i) - log p(x | j) of the full
  observation x (supply sensors, probe sensors, failure pattern), for every
  action, at three observation levels (passive/episode-start probes,
  passive/steady probes, active = probes off);
* Chernoff information C = -min_lambda log E[p_i^lambda p_j^(1-lambda)] under a
  declared behaviour policy (the myopic action of I holding a 50/50 belief
  over the pair), estimated by averaging the two unbiased importance
  identities E_j[e^{lambda L}] and E_i[e^{-(1-lambda) L}];
* symmetric KL = E_i[L] - E_j[L];
* the number of episodes a pair-restricted I (uniform prior over the pair,
  stationary truth, myopic actions) needs to reach posterior .95.

Relevance is the cross-context policy regret: V_i(a_i*) - V_i(a_j*) scored on
independent draws from those used to select the actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .references import (PatternSettings, pattern_loglik, probe_loglik, supply_loglik,
                         value_tables)
from .world import (n_actions, N_CONTEXTS, WorldParams, factors, observe, sample_episodes,
                    seed_for, simulate)

LEVELS = ("passive_episode", "passive_steady", "active")
FACTOR_BIT = {"A": 4, "B": 2, "C": 1}


def level_params(params: WorldParams, level):
    if level == "passive_steady":
        return params.with_dials(probe_start="steady"), True
    if level == "passive_episode":
        return params.with_dials(probe_start="episode"), True
    return params, False


def fail_steps_all_actions(params, context, episodes, supplies):
    n = len(episodes)
    out = np.empty((n, n_actions(params), 3), dtype=np.int8)
    for a in range(n_actions(params)):
        out[:, a] = simulate(params, np.full(n, context), episodes, np.full(n, a), supplies).fail_step
    return out


def logliks_for_truth(params, truth, contexts, episodes, seed, settings, stats=None):
    """Log-likelihoods of episodes generated under ``truth`` for every action.

    Returns (dict level -> (n, 7, len(contexts)), failure steps (n, 7, 3)).
    The pattern term is shared by all levels; only the probe term differs.
    """
    n = len(episodes)
    truth_arr = np.full(n, truth)
    supplies = episodes.supplies(params, truth_arr)
    fs = fail_steps_all_actions(params, truth, episodes, supplies)
    y_supply, _ = observe(params, truth_arr, episodes, supplies, probes_on=False)
    na = n_actions(params)
    pattern = np.empty((n, na, len(contexts)))
    for e in range(n):
        pattern[e] = pattern_loglik(params, episodes.reserves[e], int(episodes.cue[e, 0]), y_supply[e],
                                    {a: fs[e, a] for a in range(na)}, list(contexts),
                                    list(range(na)), seed_for(seed, "pattern", truth, e),
                                    settings, stats)
    sup, _ = supply_loglik(params, y_supply)
    a_idx = np.array([int(factors(c)[0]) for c in contexts])
    out = {}
    for level in LEVELS:
        lp, on = level_params(params, level)
        _, yp = observe(lp, truth_arr, episodes, supplies, probes_on=on)
        probe = probe_loglik(lp, yp, episodes.cue[:, 0], on)[:, list(contexts)]
        out[level] = (sup[:, a_idx] + probe)[:, None, :] + pattern
    return out, fs


@dataclass
class PairPool:
    contexts: tuple
    v_select: np.ndarray   # (n, 2, 7)
    v_score: np.ndarray    # (n, 2, 7)
    se_score: np.ndarray
    llr: dict              # level -> (2, n, 7): LLR log p_i/p_j for data from truth i (row 0) or j (row 1)
    stats: dict


def pair_pool(params, i, j, n, seed, m_select, m_score, settings=PatternSettings()):
    episodes = sample_episodes(n, seed_for(seed, "pool_episodes"))
    vs = value_tables(params, episodes, [i, j], m_select, seed, "select")
    vc = value_tables(params, episodes, [i, j], m_score, seed, "score")
    stats = {}
    llr = {level: np.empty((2, n, n_actions(params))) for level in LEVELS}
    for row, truth in enumerate((i, j)):
        ll, _ = logliks_for_truth(params, truth, (i, j), episodes, seed, settings, stats)
        for level in LEVELS:
            llr[level][row] = ll[level][:, :, 0] - ll[level][:, :, 1]
    return PairPool((i, j), vs.value, vc.value, vc.se, llr, stats)


def behaviour_actions(v_select, weight_i=0.5):
    return np.argmax(weight_i * v_select[:, 0] + (1 - weight_i) * v_select[:, 1], axis=1)


def chernoff(llr_i, llr_j, grid=np.linspace(0, 1, 101)):
    """Chernoff information from LLR samples under i (llr_i) and under j (llr_j)."""
    def lme(x):
        mx = np.max(x)
        return mx + np.log(np.mean(np.exp(x - mx)))
    values = []
    for lam in grid:
        g_j = lme(lam * llr_j)
        g_i = lme(-(1 - lam) * llr_i)
        mx = max(g_i, g_j)
        values.append(mx + np.log(0.5 * (np.exp(g_i - mx) + np.exp(g_j - mx))))
    values = np.array(values)
    k = int(np.argmin(values))
    return float(-values[k]), float(grid[k])


def pool_metrics(pool: PairPool, level, n_boot=200, seed=0, id_sequences=2000, id_cap=4000):
    n = pool.v_select.shape[0]
    rows = np.arange(n)
    b = behaviour_actions(pool.v_select)
    li = pool.llr[level][0][rows, b]
    lj = pool.llr[level][1][rows, b]
    none_i = pool.llr[level][0][:, 0]
    none_j = pool.llr[level][1][:, 0]
    ai = pool.v_select[:, 0].argmax(1)
    aj = pool.v_select[:, 1].argmax(1)
    reg_i = pool.v_score[rows, 0, ai] - pool.v_score[rows, 0, aj]
    reg_j = pool.v_score[rows, 1, aj] - pool.v_score[rows, 1, ai]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        boot.append((chernoff(li[s], lj[s], np.linspace(0, 1, 26))[0], li[s].mean() - lj[s].mean(),
                     0.5 * (reg_i[s].mean() + reg_j[s].mean())))
    boot = np.array(boot)
    c_b, lam = chernoff(li, lj)
    ident = identification_time(pool, level, id_sequences, id_cap, seed)
    return {
        "chernoff_behaviour": c_b, "chernoff_lambda": lam, "chernoff_se": float(boot[:, 0].std()),
        "chernoff_none": chernoff(none_i, none_j)[0],
        "sym_kl_behaviour": float(li.mean() - lj.mean()), "sym_kl_se": float(boot[:, 1].std()),
        "kl_i_j": float(li.mean()), "kl_j_i": float(-lj.mean()),
        "regret_i_acting_on_j": float(reg_i.mean()), "regret_j_acting_on_i": float(reg_j.mean()),
        "relevance": float(0.5 * (reg_i.mean() + reg_j.mean())), "relevance_se": float(boot[:, 2].std()),
        "o_value_i": float(pool.v_score[rows, 0, ai].mean()), "o_value_j": float(pool.v_score[rows, 1, aj].mean()),
        "behaviour_action_freq": np.bincount(b, minlength=pool.v_select.shape[-1]).tolist(),
        **ident,
    }


def identification_time(pool: PairPool, level, sequences, cap, seed):
    """Episodes for a pair-restricted myopic I to reach posterior .95 on the truth."""
    rng = np.random.default_rng(seed_for(seed, "identification", level))
    n = pool.v_select.shape[0]
    target = np.log(0.95 / 0.05)
    result = {}
    for row, sign in ((0, 1.0), (1, -1.0)):
        llr = pool.llr[level][row]
        logodds = np.zeros(sequences)  # log p(i)/p(j) posterior odds
        done = np.full(sequences, -1)
        for t in range(cap):
            active = done < 0
            if not active.any():
                break
            idx = rng.integers(0, n, sequences)
            w = 1.0 / (1.0 + np.exp(-logodds))
            vals = w[:, None] * pool.v_select[idx, 0] + (1 - w[:, None]) * pool.v_select[idx, 1]
            act = vals.argmax(1)
            logodds = np.where(active, logodds + llr[idx, act], logodds)
            hit = active & (sign * logodds >= target)
            done[hit] = t + 1
        censored = done < 0
        times = np.where(censored, cap, done)
        name = "truth_i" if row == 0 else "truth_j"
        result[f"id_mean_{name}"] = float(times.mean())
        result[f"id_median_{name}"] = float(np.median(times))
        result[f"id_p90_{name}"] = float(np.quantile(times, 0.9))
        result[f"id_censored_{name}"] = float(censored.mean())
    result["id_mean"] = 0.5 * (result["id_mean_truth_i"] + result["id_mean_truth_j"])
    result["id_p90"] = max(result["id_p90_truth_i"], result["id_p90_truth_j"])
    return result


# ---------------------------------------------------------------- grading

THRESHOLDS = {
    # Declared after an exploratory look at the achievable relevance range
    # (O minus no-transfer is about .2; A-type relevance about .2; B/C about .02).
    "separability_high_max_id_episodes": 5.0,
    "separability_low_min_id_episodes": 25.0,
    "relevance_high_min": 0.015,
    "relevance_low_max": 0.005,
}


def grade(id_mean, relevance, thresholds=THRESHOLDS):
    if id_mean <= thresholds["separability_high_max_id_episodes"]:
        sep = "high"
    elif id_mean >= thresholds["separability_low_min_id_episodes"]:
        sep = "low"
    else:
        sep = "mid"
    if relevance >= thresholds["relevance_high_min"]:
        rel = "high"
    elif relevance <= thresholds["relevance_low_max"]:
        rel = "low"
    else:
        rel = "mid"
    return sep, rel


def factor_pairs(factor):
    bit = FACTOR_BIT[factor]
    return [(c, c | bit) for c in range(N_CONTEXTS) if not c & bit]


# ---------------------------------------------------------------- VOI

def mi_from_logliks(ll, prior=None):
    """MI(context; x) from ll[truth] = (n, 8) log-likelihood arrays of samples drawn under each truth."""
    k = len(ll)
    prior = np.full(k, 1.0 / k) if prior is None else prior
    total, per = 0.0, []
    for c in range(k):
        x = ll[c] + np.log(prior)[None]
        mx = x.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(x - mx).sum(1))
        term = ll[c][:, c] - lse
        per.append(term)
        total += prior[c] * term.mean()
    se = float(np.sqrt(sum(prior[c] ** 2 * per[c].var() / len(per[c]) for c in range(k))))
    return float(total), se


def binary_factor_mi(ll, factor):
    """MI(factor; x | other factors known), uniform over the pair, averaged over the 4 pairs."""
    bit = FACTOR_BIT[factor]
    vals, ses = [], []
    for c0, c1 in factor_pairs(factor):
        sub = [ll[c0][:, [c0, c1]], ll[c1][:, [c0, c1]]]
        v, s = mi_from_logliks(sub)
        vals.append(v)
        ses.append(s)
    return float(np.mean(vals)), float(np.sqrt(np.sum(np.square(ses))) / len(ses))


# ---------------------------------------------------------------- schedule

TRAINING = (0, 4, 2, 1, 7)   # A0B0C0, A1B0C0, A0B1C0, A0B0C1, A1B1C1
HELD_OUT = (6, 5, 3)         # A1B1C0, A1B0C1, A0B1C1: every pairwise sub-combination seen in training
MINORITY = 7                 # A1B1C1


@dataclass(frozen=True)
class ScheduleSpec:
    length: int = 20000
    heldout_start_fraction: float = 0.75
    minority_share: float = 0.06
    heldout_share_late: float = 0.30          # of the last quarter, split equally
    run_min: int = 8
    run_max: int = 800                        # P(L) proportional to 1/L on [run_min, run_max]
    min_minority_runs: int = 5                # first appearance + >= 4 recurrences
    min_absence_ratio: float = 4.0            # max / min minority absence length
    filter_epsilon: float = 0.02              # I's run-length pmf: (1-eps) D + eps U{1..run_max}

    def run_pmf(self):
        support = np.arange(1, self.run_max + 1)
        pmf = np.where(support >= self.run_min, 1.0 / support, 0.0)
        return pmf / pmf.sum()


def _quotas(spec: ScheduleSpec):
    p1 = int(round(spec.length * spec.heldout_start_fraction))
    p2 = spec.length - p1
    others = [c for c in TRAINING if c != MINORITY]
    q1 = {MINORITY: int(round(spec.minority_share * p1))}
    rest = p1 - q1[MINORITY]
    for k, c in enumerate(others):
        q1[c] = rest // len(others) + (1 if k < rest % len(others) else 0)
    held = int(round(spec.heldout_share_late * p2))
    q2 = {}
    for k, c in enumerate(HELD_OUT):
        q2[c] = held // len(HELD_OUT) + (1 if k < held % len(HELD_OUT) else 0)
    train2 = p2 - held
    q2[MINORITY] = int(round(spec.minority_share * train2))
    rest = train2 - q2[MINORITY]
    for k, c in enumerate(others):
        q2[c] = rest // len(others) + (1 if k < rest % len(others) else 0)
    return p1, p2, q1, q2


def _draw_runs(rng, quota, pmf):
    runs, total = [], 0
    support = np.arange(1, len(pmf) + 1)
    while total < quota:
        length = int(rng.choice(support, p=pmf))
        length = min(length, quota - total)
        runs.append(length)
        total += length
    return runs


def _order(rng, runs_by_ctx, first_forbidden=None, tries=200):
    for _ in range(tries):
        pool = {c: list(rng.permutation(r)) for c, r in runs_by_ctx.items() if r}
        order, prev, ok = [], first_forbidden, True
        while pool:
            options = [c for c in pool if c != prev]
            if not options:
                ok = False
                break
            weights = np.array([len(pool[c]) for c in options], dtype=float)
            c = options[int(rng.choice(len(options), p=weights / weights.sum()))]
            order.append((c, int(pool[c].pop())))
            if not pool[c]:
                del pool[c]
            prev = c
        if ok:
            return order
    return None


def generate_schedule(seed, spec: ScheduleSpec = ScheduleSpec(), max_attempts=500):
    """Stratified semi-Markov schedule. Returns (contexts (length,), runs list, info)."""
    rng = np.random.default_rng(seed_for(seed, "commons_schedule"))
    p1, p2, q1, q2 = _quotas(spec)
    pmf = spec.run_pmf()
    for attempt in range(max_attempts):
        runs1 = {c: _draw_runs(rng, q, pmf) for c, q in q1.items()}
        runs2 = {c: _draw_runs(rng, q, pmf) for c, q in q2.items()}
        order1 = _order(rng, runs1)
        if order1 is None:
            continue
        order2 = _order(rng, runs2, first_forbidden=order1[-1][0])
        if order2 is None:
            continue
        order = order1 + order2
        starts = np.cumsum([0] + [r for _, r in order])[:-1]
        minority = [(s, r) for (c, r), s in zip(order, starts) if c == MINORITY]
        if len(minority) < spec.min_minority_runs:
            continue
        absences = [minority[k + 1][0] - (minority[k][0] + minority[k][1]) for k in range(len(minority) - 1)]
        if min(absences) <= 0 or max(absences) / min(absences) < spec.min_absence_ratio:
            continue
        contexts = np.concatenate([np.full(r, c, dtype=np.int8) for c, r in order])
        assert len(contexts) == spec.length
        info = {"attempt": attempt, "phase_start": p1, "quotas_phase1": q1, "quotas_phase2": q2,
                "minority_absences": absences}
        return contexts, order, info
    raise RuntimeError("schedule constraints not satisfiable")


def schedule_model(spec: ScheduleSpec = ScheduleSpec()):
    """I's semi-Markov model of the generator (declared approximation)."""
    from .references import ScheduleModel
    p1, p2, q1, q2 = _quotas(spec)
    pmf = (1 - spec.filter_epsilon) * spec.run_pmf() + spec.filter_epsilon / spec.run_max
    mats = []
    for q in (q1, q2):
        share = np.zeros(N_CONTEXTS)
        for c, v in q.items():
            share[c] = v
        m = np.tile(share, (N_CONTEXTS, 1))
        np.fill_diagonal(m, 0.0)
        m = m / np.maximum(m.sum(1, keepdims=True), 1e-300)
        mats.append(m)
    initial = np.zeros(N_CONTEXTS)
    for c, v in q1.items():
        initial[c] = v
    return ScheduleModel(pmf, initial / initial.sum(), tuple(mats), phase_start=p1)


def schedule_statistics(contexts, order, info, spec: ScheduleSpec = ScheduleSpec(), id_time=None):
    lengths = np.array([r for _, r in order])
    ctx = np.array([c for c, _ in order])
    starts = np.cumsum([0] + list(lengths))[:-1]
    stats = {"episodes": int(len(contexts)), "runs": int(len(order)),
             "run_length": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                            "mean": float(lengths.mean()), "max": int(lengths.max())},
             "per_context": {}}
    for c in range(N_CONTEXTS):
        sel = ctx == c
        if not sel.any():
            continue
        stats["per_context"][str(c)] = {
            "share": float((contexts == c).mean()), "runs": int(sel.sum()),
            "first_episode": int(starts[sel].min()), "median_run": float(np.median(lengths[sel]))}
    stats["minority"] = {"context": MINORITY, "runs": int((ctx == MINORITY).sum()),
                         "recurrences": int((ctx == MINORITY).sum() - 1),
                         "absences": info["minority_absences"]}
    stats["heldout_first_episode"] = int(min(starts[np.isin(ctx, HELD_OUT)]))
    stats["heldout_only_in_last_quarter"] = bool(stats["heldout_first_episode"] >= info["phase_start"])
    if id_time is not None:
        stats["runs_shorter_than_2x_low_sep_id_time"] = float((lengths <= 2 * id_time).mean())
        stats["runs_shorter_than_low_sep_id_time"] = float((lengths <= id_time).mean())
    return stats
