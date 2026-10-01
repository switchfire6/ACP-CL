"""D0 verification: independent slow reference, likelihood brute force,
filter enumeration, O's common-random-number properties and determinism.

The slow reference re-derives every context parameter from the written spec
with scalar Python arithmetic and an explicit event list; it shares no code
with the vectorised simulator except the WorldParams container.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from . import references as R
from .world import (n_actions, action_table, N_CONTEXTS, OBS_NOISE, SUPPLY_NOISE, SURVIVED, T, Episodes,
                    WorldParams, entity_kernel, observe, probe_arrivals, channel_params,
                    sample_episodes, seed_for, simulate, supply_means, factors)

# action -> (channel, donor, recipient), 0-based, written independently of world.py tables
SLOW_ACTIONS = {1: (0, 0, 1), 2: (0, 1, 0), 3: (1, 1, 2), 4: (1, 2, 1), 5: (2, 2, 0), 6: (2, 0, 2)}


def slow_means(params, a_level):
    out = []
    for t in range(12):
        season = 1.0 if t < params.season_flip_step else -1.0
        row = []
        for i in range(3):
            sign = params.base_signs[i]
            if a_level == 1 and i in (0, 1):
                sign = sign * (1.0 - 2.0 * params.flip_amplitude)
            mean = params.mean_supply + params.imbalance * season * sign
            if i == 2:
                mean += params.entity3_mean_offset
            row.append(mean)
        out.append(row)
    return out


def slow_action(params, action):
    """(channel, donor, recipient, amount) re-derived from the spec: blocks of 6 moves per amount."""
    block, move = divmod(action - 1, 6)
    ch, donor, rec = SLOW_ACTIONS[move + 1]
    return ch, donor, rec, float(params.amounts[block])


def slow_channels(params, context, cue1):
    a_level, b_level, c_level = (context >> 2) & 1, (context >> 1) & 1, context & 1
    e12 = params.e12 - params.b_gap if b_level else params.e12
    e31 = params.e31
    if params.b_mode == "both" and b_level:
        e31 = params.e31 - params.b_gap
    if params.b_mode == "swap" and not b_level:
        e31 = params.e31 - params.b_gap
    eff = [e12, params.e23, e31]
    delay = [params.d12, params.d23_fast, params.d31]
    if params.c_mode == "ch23":
        fast = c_level == 1 and cue1 == 1
        delay[1] = params.d23_fast if fast else params.d23_fast + params.c_delta
    else:
        for k in params.c_channels:
            if params.c_mode == "same":
                fast = c_level == 1 and cue1 == 1
            else:
                wanted = 1 if k == 1 else 0
                fast = c_level == 1 and cue1 == wanted
            if not fast:
                delay[k] = delay[k] + params.c_delta
    return eff, delay


def slow_case(params, context, reserves, cue, supplies, action, probes_on=True):
    """Scalar reference for one episode. supplies: list[T][3]."""
    eff, delay = slow_channels(params, context, int(cue[0]))
    stock = [float(r) for r in reserves]
    events = []
    sent = delivered = loss = 0.0
    if action:
        ch, donor, rec, size = slow_action(params, action)
        sent = min(size, stock[donor])
        stock[donor] -= sent
        delivered = sent * eff[ch]
        loss = sent - delivered
        k = math.floor(delay[ch])
        f = delay[ch] - k
        events.append((k, rec, delivered * (1.0 - f)))
        events.append((k + 1, rec, delivered * f))
    fail = [SURVIVED] * 3
    overflow = [0.0] * 3
    consumed = [0.0] * 3
    in_transit = delivered
    for t in range(12):
        for step, rec, amount in events:
            if step == t:
                stock[rec] = stock[rec] + amount
                in_transit -= amount
        for i in range(3):
            stock[i] = stock[i] + supplies[t][i]
            over = max(stock[i] - 18.0, 0.0)
            overflow[i] += over
            stock[i] = min(stock[i], 18.0)
            use = min(max(stock[i], 0.0), 2.0)
            consumed[i] += use
            stock[i] = stock[i] - use
            if fail[i] == SURVIVED and stock[i] <= 1e-10:
                fail[i] = t + 1
    # probes
    probe = [[0.0] * 3 for _ in range(12)]
    if probes_on:
        for ch in range(3):
            k = math.floor(delay[ch])
            f = delay[ch] - k
            first = 0 if params.probe_start == "episode" else -40
            for tau in range(first, 12):
                for step, amount in ((tau + k, 0.3 * eff[ch] * (1 - f)), (tau + k + 1, 0.3 * eff[ch] * f)):
                    if 0 <= step < 12:
                        probe[step][ch] += amount
    return {"fail": fail, "stock": stock, "overflow": overflow, "consumed": consumed,
            "sent": sent, "loss": loss, "in_transit": in_transit, "probe": probe}


def edge_params():
    base = WorldParams(base_signs=(1.0, -1.0, -0.5))
    return [base,
            base.with_dials(b_gap=0.8, c_delta=11.0, flip_amplitude=0.3),
            base.with_dials(c_delta=2.5, d12=0.4, d31=3.7, probe_start="steady"),
            base.with_dials(c_delta=14.0, d31=12.5, b_gap=0.05, flip_amplitude=1.0)]


def check_simulator(n=1500, seed=1, param_sets=None):
    """Vectorised simulator vs slow reference; conservation; kernel equality."""
    worst = {"fail_mismatch": 0, "stock": 0.0, "overflow": 0.0, "consumed": 0.0, "sent": 0.0,
             "loss": 0.0, "in_transit": 0.0, "probe": 0.0, "means": 0.0, "channels": 0.0,
             "conservation": 0.0, "probe_conservation": 0.0, "kernel_mismatch": 0, "cases": 0}
    rng = np.random.default_rng(seed)
    for pi, params in enumerate(edge_params() if param_sets is None else param_sets):
        ep = sample_episodes(n, seed_for(seed, "sim_check", pi))
        # stress edges: some tiny reserves, some near capacity, some very negative supply draws
        ep.reserves[: n // 10] = rng.uniform(0.01, 3.0, (n // 10, 3))
        ep.reserves[n // 10: n // 5] = rng.uniform(15.0, 18.0, (n // 10, 3))
        ep.z_supply[n // 5: n // 4, 3, :] = -7.0
        ctx = rng.integers(0, N_CONTEXTS, n)
        act = rng.integers(0, n_actions(params), n)
        sup = ep.supplies(params, ctx)
        for probes_on in (True, False):
            res = simulate(params, ctx, ep, act, sup, probes_on=probes_on)
            worst["conservation"] = max(worst["conservation"], res.conservation_error)
            worst["probe_conservation"] = max(worst["probe_conservation"], res.probe_conservation_error)
        eff, delay = channel_params(params, ctx, ep.cue[:, 0])
        probes = probe_arrivals(params, eff, delay, True)
        mu = supply_means(params)
        for a_level in (0, 1):
            worst["means"] = max(worst["means"], float(np.max(np.abs(np.array(slow_means(params, a_level)) - mu[a_level]))))
        for e in range(n):
            ref = slow_case(params, int(ctx[e]), ep.reserves[e], ep.cue[e], sup[e].tolist(), int(act[e]))
            worst["cases"] += 1
            se, sd = slow_channels(params, int(ctx[e]), int(ep.cue[e, 0]))
            worst["channels"] = max(worst["channels"], float(np.max(np.abs(np.array(se) - eff[e]))),
                                    float(np.max(np.abs(np.array(sd) - delay[e]))))
            worst["fail_mismatch"] += int(np.any(np.array(ref["fail"]) != res.fail_step[e]))
            worst["stock"] = max(worst["stock"], float(np.max(np.abs(np.array(ref["stock"]) - res.final_stock[e]))))
            worst["overflow"] = max(worst["overflow"], float(np.max(np.abs(np.array(ref["overflow"]) - res.overflow[e]))))
            worst["consumed"] = max(worst["consumed"], float(np.max(np.abs(np.array(ref["consumed"]) - res.consumed[e]))))
            worst["sent"] = max(worst["sent"], abs(ref["sent"] - res.sent[e]))
            worst["loss"] = max(worst["loss"], abs(ref["loss"] - res.loss[e]))
            worst["in_transit"] = max(worst["in_transit"], abs(ref["in_transit"] - res.in_transit_final[e]))
            worst["probe"] = max(worst["probe"], float(np.max(np.abs(np.array(ref["probe"]) - probes[e]))))
        # kernel equality (bitwise failure steps) for every action
        for a in range(n_actions(params)):
            full = simulate(params, ctx, ep, np.full(n, a), sup)
            stock, sent, delivered, loss, arrivals, _, _ = R_transfer(params, ctx, ep, a)
            for i in range(3):
                step = entity_kernel(stock[:, i], arrivals[:, :, i], sup[:, :, i], want_step=True)
                worst["kernel_mismatch"] += int(np.sum(step != full.fail_step[:, i]))
    tol = 1e-9
    worst["pass"] = bool(worst["fail_mismatch"] == 0 and worst["kernel_mismatch"] == 0 and all(
        worst[k] < tol for k in ("stock", "overflow", "consumed", "sent", "loss", "in_transit", "probe",
                                 "means", "channels", "conservation", "probe_conservation")))
    return worst


def R_transfer(params, ctx, ep, a):
    from .world import transfer_setup
    return transfer_setup(params, ctx, ep.reserves, ep.cue, np.full(len(ep), a))


def check_value_tables(params, seed=3, n=40, m=4096):
    """O's estimator: CRN, product vs joint, SE calibration, determinism, independence."""
    out = {}
    ep = sample_episodes(n, seed_for(seed, "vt_check"))
    contexts = list(range(N_CONTEXTS))
    # (a) joint-indicator estimator equals the fraction from the full simulator on the same draws
    mismatch = 0
    zs = []
    for e in range(min(n, 8)):
        z = R.draws(seed_for(seed, "vt_draws", e), 1024)
        value, se, joint = R.episode_values(params, ep.reserves[e], int(ep.cue[e, 0]), contexts, z, joint=True)
        rep = Episodes(np.repeat(ep.reserves[e:e + 1], 1024, 0), np.repeat(ep.cue[e:e + 1], 1024, 0),
                       z, np.zeros_like(z), np.zeros_like(z))
        for c in contexts:
            for a in range(n_actions(params)):
                frac = simulate(params, np.full(1024, c), rep, np.full(1024, a)).joint_survival.mean()
                mismatch += int(frac != joint[contexts.index(c), a])
                zs.append((value[c, a] - joint[c, a]) / max(se[c, a], 1e-12))
    out["joint_indicator_vs_full_simulator_mismatches"] = mismatch
    zs = np.array(zs)
    out["product_vs_joint_z_abs_max"] = float(np.max(np.abs(zs)))
    out["product_vs_joint_z_rms"] = float(np.sqrt(np.mean(zs ** 2)))
    # (b) SE formula vs replicate spread
    reps = []
    for r in range(12):
        reps.append(R.value_tables(params, ep.take(np.arange(10)), contexts, m, seed + 100 + r, "rep").value)
    reps = np.array(reps)
    ref = R.value_tables(params, ep.take(np.arange(10)), contexts, m, seed + 99, "rep")
    emp = reps.std(axis=0, ddof=1)
    mask = ref.se > 1e-4
    out["se_ratio_empirical_over_formula_median"] = float(np.median(emp[mask] / ref.se[mask]))
    # (c) determinism and independence of selection vs scoring draws
    a1 = R.value_tables(params, ep.take(np.arange(5)), contexts, m, seed, "select")
    a2 = R.value_tables(params, ep.take(np.arange(5)), contexts, m, seed, "select")
    b1 = R.value_tables(params, ep.take(np.arange(5)), contexts, m, seed, "score")
    out["determinism_identical"] = bool(np.array_equal(a1.value, a2.value))
    za = R.draws(seed_for(seed, "select", 0), m).ravel()
    zb = R.draws(seed_for(seed, "score", 0), m).ravel()
    out["select_score_draw_correlation"] = float(np.corrcoef(za, zb)[0, 1])
    out["select_score_tables_differ"] = bool(not np.array_equal(a1.value, b1.value))
    # (d) CRN variance reduction for action differences
    diffs_crn = reps[:, :, :, 1:] - reps[:, :, :, :1]
    indep = reps[:, :, :, 1:] - np.roll(reps, 1, axis=0)[:, :, :, :1]
    v_crn = diffs_crn.var(axis=0).mean()
    v_ind = indep.var(axis=0).mean()
    out["crn_variance_ratio_of_action_differences"] = float(v_crn / v_ind)
    # (e) the uninvolved-entity survival is shared exactly across actions (same draws)
    z = R.draws(seed_for(seed, "crn", 0), 2048)
    solo = R.episode_values(params, ep.reserves[0], int(ep.cue[0, 0]), [3], z)[0]
    full = R.episode_values(params, ep.reserves[0], int(ep.cue[0, 0]), contexts, z)[0]
    out["context_subset_consistency"] = bool(np.array_equal(solo[0], full[3]))
    # (f) selection optimism: max of selection estimates minus scoring estimate of the chosen action
    sel = R.value_tables(params, ep, [0, 5], m, seed, "select").value
    sco = R.value_tables(params, ep, [0, 5], m, seed, "score").value
    arg = sel.argmax(2)
    rows = np.arange(n)[:, None]
    out["selection_optimism_bias_mean"] = float((sel.max(2) - sco[rows, np.arange(2)[None], arg]).mean())
    out["pass"] = bool(mismatch == 0 and out["determinism_identical"]
                       and abs(out["select_score_draw_correlation"]) < 0.01 and out["context_subset_consistency"]
                       and 0.7 < out["se_ratio_empirical_over_formula_median"] < 1.3 and out["product_vs_joint_z_rms"] < 1.5
                       and out["crn_variance_ratio_of_action_differences"] < 1.0)
    return out


def find_edge_cases(params, seed=5, n=6000):
    """Episodes (truth, action) exercising each edge case, plus typical ones."""
    ep = sample_episodes(n, seed_for(seed, "edge_search"))
    rng = np.random.default_rng(seed)
    ctx = rng.integers(0, N_CONTEXTS, n)
    act = rng.integers(1, n_actions(params), n)
    ep.reserves[: n // 4] = rng.uniform(1.0, 3.0, (n // 4, 3))
    sup = ep.supplies(params, ctx)
    res = simulate(params, ctx, ep, act, sup)
    _, table_donor, table_rec, table_amount = action_table(params)
    arrival_idx = res.arrival_step_first
    rec = table_rec[act]
    don = table_donor[act]
    rows = np.arange(n)
    conditions = {
        "typical_1": rows >= n // 4,
        "typical_2": (rows >= n // 4) & (res.fail_step.min(1) < SURVIVED),
        "capacity_clipping": res.overflow.sum(1) > 0.2,
        "donor_limited": ep.reserves[rows, don] < table_amount[act],
        "failure_before_arrival": (res.fail_step[rows, rec] <= arrival_idx) & (arrival_idx < T),
    }
    cases = {}
    for name, cond in conditions.items():
        idx = np.flatnonzero(cond)
        if len(idx):
            e = int(idx[len(idx) // 2])
            cases[name] = (e, int(ctx[e]), int(act[e]))
    # negative supply: a realistic tail draw (s ~ -0.17) at the lowest-mean (step, entity) of the truth
    e = int(n // 2)
    mu = supply_means(params)[int(factors(int(ctx[e]))[0])]
    t, i = np.unravel_index(np.argmin(mu), mu.shape)
    ep.z_supply[e, t, i] = -mu[t, i] / SUPPLY_NOISE - 0.5
    cases["negative_supply"] = (e, int(ctx[e]), int(act[e]))
    return ep, cases


def alternative_context(context, action, params=None):
    """B/C flip that shares the truth's supply prior: flip C if the action's channel is C-gated
    (or no transfer), else B. With D0 parameters: C for 2<->3 and 'none', else B."""
    ch = int(action_table(params)[0][action])
    gated = (1,) if params is None else tuple(params.c_channels)
    return context ^ (1 if ch == -1 or ch in gated else 2)


def likelihood_case(params, ep, e, truth, action, context, entity, m_is=2 ** 24, k_conj=2 ** 18, seed=0,
                    chunk=2 ** 18):
    """Per-entity log p(y_i, fail_i | context): prior-sampling estimator vs conjugate estimator."""
    one = ep.take(np.array([e]))
    sup_true = one.supplies(params, np.array([truth]))
    fs = simulate(params, np.array([truth]), one, np.array([action]), sup_true).fail_step[0]
    y_sup, _ = observe(params, np.array([truth]), one, sup_true)
    y = y_sup[0, :, entity]
    obs = int(fs[entity])
    a_level = int(factors(context)[0]) if entity < 2 else 0
    mu = supply_means(params)[a_level, :, entity]
    # conjugate estimator: exact marginal x posterior-sampled indicator (per-entity kernel path)
    log_marg = float(np.sum(-0.5 * ((y - mu) ** 2 / R.MARG_VAR + np.log(R.MARG_VAR) + R.LOG_2PI)))
    variants, index = R.entity_variants(params, one.reserves[0], int(one.cue[0, 0]), [context], [action])
    a_lv, x0, arr, has_arr = R._variant_arrays([variants[entity][index[(context, action)][entity]]])
    post_mean, post_var = R.posterior_supply(params, y, a_level, entity)
    rng = np.random.default_rng(seed_for(seed, "conj", e, context, entity))
    u = rng.standard_normal((k_conj, T))
    step = entity_kernel(x0[:, None], arr[:, None, :] if has_arr else None,
                         (post_mean[None] + np.sqrt(post_var) * u)[None], want_step=True)[0]
    hits = int(np.sum(step == obs))
    # prior-sampling estimator using the FULL simulator (other entities at their true supplies)
    rng = np.random.default_rng(seed_for(seed, "prior_is", e, context, entity))
    logw_all = []
    done = 0
    while done < m_is:
        size = min(chunk, m_is - done)
        s = mu[None] + SUPPLY_NOISE * rng.standard_normal((size, T))
        supplies = np.repeat(sup_true, size, 0)
        supplies[:, :, entity] = s
        rep = Episodes(np.repeat(one.reserves, size, 0), np.repeat(one.cue, size, 0),
                       np.zeros((size, T, 3)), np.zeros((size, T, 3)), np.zeros((size, T, 3)))
        f = simulate(params, np.full(size, context), rep, np.full(size, action), supplies).fail_step[:, entity]
        lw = np.sum(-0.5 * ((y[None] - s) ** 2 / OBS_NOISE ** 2 + np.log(OBS_NOISE ** 2) + R.LOG_2PI), axis=1)
        logw_all.append(np.where(f == obs, lw, -np.inf))
        done += size
    lw = np.concatenate(logw_all)
    finite = np.isfinite(lw)
    result = {"entity": entity, "context": int(context), "observed_fail_step": obs, "hits_conj": hits,
              "k_conj": k_conj, "m_is": m_is, "is_nonzero": int(finite.sum())}
    if hits == 0 and finite.sum() == 0:
        result.update({"agree": True, "note": "both estimators zero"})
        return result
    if finite.sum() == 0 or hits == 0:
        # one estimator zero: agreement only if the other is negligible relative to its resolution
        p_conj = hits / k_conj
        mx = lw[finite].max() if finite.any() else -np.inf
        log_is = mx + np.log(np.exp(lw[finite] - mx).sum() / m_is) if finite.any() else -np.inf
        result.update({"log_is": float(log_is), "log_conj": float(log_marg + np.log(p_conj)) if hits else None,
                       "agree": bool(hits <= 3 or finite.sum() <= 3), "note": "one estimator zero"})
        return result
    mx = lw[finite].max()
    w = np.exp(lw - mx)
    mean_w = w.mean()
    log_is = float(mx + np.log(mean_w))
    rel_se_is = float(w.std() / np.sqrt(m_is) / mean_w)
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    p_conj = hits / k_conj
    log_conj = float(log_marg + np.log(p_conj))
    rel_se_conj = float(np.sqrt((1 - p_conj) / (p_conj * k_conj)))
    z = (log_is - log_conj) / np.sqrt(rel_se_is ** 2 + rel_se_conj ** 2)
    result.update({"log_is": log_is, "log_conj": log_conj, "diff": log_is - log_conj, "rel_se_is": rel_se_is,
                   "rel_se_conj": rel_se_conj, "ess": ess, "z": float(z), "agree": bool(abs(z) < 4.0)})
    return result


def smc_prior_loglik(params, reserves, cue1, action, context, entity, y, obs, particles=2 ** 17, runs=6,
                     seed=0):
    """Sequential prior-sampling estimator of log p(y_i, fail_i | context) for one entity.

    Bootstrap particle filter: supplies are drawn from the PRIOR N(mu, .35^2)
    step by step, each particle is weighted by the exact sensor density
    N(y_t; s_t, .1^2) times the failure-pattern indicator of that step, and
    particles are resampled multinomially. prod_t mean(w_t) is unbiased for
    p(y, pattern | context). No conjugate formula and no world.py code is used:
    means, channels, departure and the step physics come from the scalar
    reference above. Returns (mean log Z over runs, SE of log Z, zero_runs).
    """
    means = np.array(slow_means(params, (context >> 2) & 1))[:, entity]
    eff, delay = slow_channels(params, context, cue1)
    x0 = float(reserves[entity])
    arrivals = np.zeros(12)
    if action:
        ch, donor, rec, size = slow_action(params, action)
        sent = min(size, float(reserves[donor]))
        if entity == donor:
            x0 = x0 - sent
        if entity == rec:
            k = math.floor(delay[ch])
            f = delay[ch] - k
            if k < 12:
                arrivals[k] += sent * eff[ch] * (1.0 - f)
            if k + 1 < 12:
                arrivals[k + 1] += sent * eff[ch] * f
    rng = np.random.default_rng(seed)
    logz = []
    for _ in range(runs):
        x = np.full(particles, x0)
        failed = np.zeros(particles, dtype=bool)
        total = 0.0
        for t in range(12):
            s = means[t] + SUPPLY_NOISE * rng.standard_normal(particles)
            x = x + arrivals[t]
            x = x + s
            x = np.minimum(x, 18.0)
            x = x - np.clip(x, 0.0, 2.0)
            now = ~failed & (x <= 1e-10)
            if t + 1 < obs:
                ok = ~failed & ~now
            elif t + 1 == obs:
                ok = now
            else:
                ok = np.ones(particles, dtype=bool)
            failed |= now
            w = np.where(ok, np.exp(-0.5 * (y[t] - s) ** 2 / OBS_NOISE ** 2) / (OBS_NOISE * np.sqrt(2 * np.pi)), 0.0)
            mw = w.mean()
            if mw <= 0:
                total = -np.inf
                break
            total += np.log(mw)
            idx = rng.choice(particles, particles, p=w / w.sum())
            x, failed = x[idx], failed[idx]
        logz.append(total)
    logz = np.array(logz)
    finite = np.isfinite(logz)
    if not finite.any():
        return -np.inf, 0.0, int(runs)
    # log of the mean of Z over runs (unbiased Z average), SE from the run spread
    mx = logz[finite].max()
    zvals = np.where(finite, np.exp(logz - mx), 0.0)
    est = mx + np.log(zvals.mean())
    se = float(zvals.std(ddof=1) / np.sqrt(runs) / zvals.mean()) if zvals.mean() > 0 else 0.0
    return float(est), se, int((~finite).sum())


def likelihood_case_smc(params, ep, e, truth, action, context, entity, k_conj=2 ** 18, seed=0):
    """Per-entity: sequential prior-sampling estimator vs the conjugate estimator used by I."""
    one = ep.take(np.array([e]))
    sup_true = one.supplies(params, np.array([truth]))
    fs = simulate(params, np.array([truth]), one, np.array([action]), sup_true).fail_step[0]
    y_sup, _ = observe(params, np.array([truth]), one, sup_true)
    y = y_sup[0, :, entity]
    obs = int(fs[entity])
    a_level = int(factors(context)[0]) if entity < 2 else 0
    mu = supply_means(params)[a_level, :, entity]
    log_marg = float(np.sum(-0.5 * ((y - mu) ** 2 / R.MARG_VAR + np.log(R.MARG_VAR) + R.LOG_2PI)))
    variants, index = R.entity_variants(params, one.reserves[0], int(one.cue[0, 0]), [context], [action])
    a_lv, x0, arr, has_arr = R._variant_arrays([variants[entity][index[(context, action)][entity]]])
    post_mean, post_var = R.posterior_supply(params, y, a_level, entity)
    rng = np.random.default_rng(seed_for(seed, "conj", e, context, entity))
    u = rng.standard_normal((k_conj, T))
    step = entity_kernel(x0[:, None], arr[:, None, :] if has_arr else None,
                         (post_mean[None] + np.sqrt(post_var) * u)[None], want_step=True)[0]
    hits = int(np.sum(step == obs))
    log_smc, se_smc, zero_runs = smc_prior_loglik(params, one.reserves[0], int(one.cue[0, 0]), action, context,
                                                  entity, y, obs, seed=seed_for(seed, "smc", e, context, entity))
    out = {"entity": entity, "context": int(context), "observed_fail_step": obs, "hits_conj": hits,
           "k_conj": k_conj, "log_smc": log_smc, "se_log_smc": se_smc, "smc_zero_runs": zero_runs}
    if hits == 0:
        out.update({"log_conj": None, "agree": bool(not np.isfinite(log_smc) or log_smc - log_marg < np.log(8 / k_conj)),
                    "note": "conjugate estimator zero; SMC must be below its resolution"})
        return out
    p_conj = hits / k_conj
    log_conj = float(log_marg + np.log(p_conj))
    se_conj = float(np.sqrt((1 - p_conj) / (p_conj * k_conj)))
    if not np.isfinite(log_smc):
        out.update({"log_conj": log_conj, "agree": bool(hits <= 8), "note": "SMC zero"})
        return out
    z = (log_smc - log_conj) / np.sqrt(se_smc ** 2 + se_conj ** 2 + 1e-12)
    out.update({"log_conj": log_conj, "se_log_conj": se_conj, "diff": log_smc - log_conj, "z": float(z),
                "agree": bool(abs(z) < 4.0 or abs(log_smc - log_conj) < 0.02)})
    return out


def joint_factorisation_case(params, ep, e, truth, action, context, k=2 ** 16, seed=0):
    """Joint conjugate sampling through the FULL simulator vs the product of per-entity estimates."""
    one = ep.take(np.array([e]))
    sup_true = one.supplies(params, np.array([truth]))
    fs = simulate(params, np.array([truth]), one, np.array([action]), sup_true).fail_step[0]
    y_sup, _ = observe(params, np.array([truth]), one, sup_true)
    a_level = int(factors(context)[0])
    rng = np.random.default_rng(seed_for(seed, "joint_conj", e, context))
    supplies = np.empty((k, T, 3))
    for i in range(3):
        mean, var = R.posterior_supply(params, y_sup[0, :, i], a_level if i < 2 else 0, i)
        supplies[:, :, i] = mean[None] + np.sqrt(var) * rng.standard_normal((k, T))
    rep = Episodes(np.repeat(one.reserves, k, 0), np.repeat(one.cue, k, 0), np.zeros((k, T, 3)),
                   np.zeros((k, T, 3)), np.zeros((k, T, 3)))
    f = simulate(params, np.full(k, context), rep, np.full(k, action), supplies).fail_step
    joint = float(np.all(f == fs[None], axis=1).mean())
    per = [(f[:, i] == fs[i]).mean() for i in range(3)]
    product = float(np.prod(per))
    se = float(np.sqrt(max(joint * (1 - joint), 1e-12) / k))
    # the per-entity likelihood code path on the same observation
    ll = R.pattern_loglik(params, one.reserves[0], int(one.cue[0, 0]), y_sup[0], {action: fs}, [context],
                          [action], seed_for(seed, "pl", e, context),
                          R.PatternSettings(k=2 ** 15, k_refine=2 ** 17))[0, 0]
    return {"joint": joint, "product_same_draws": product, "pattern_loglik_prob": float(np.exp(ll)),
            "z_joint_vs_product": float((joint - product) / se),
            "z_joint_vs_code": float((joint - np.exp(ll)) / np.sqrt(se ** 2 + max(np.exp(ll) * (1 - np.exp(ll)), 1e-12) / 2 ** 15)),
            "agree": bool(abs(joint - product) / se < 4 and abs(joint - np.exp(ll)) / np.sqrt(se ** 2 + max(np.exp(ll) * (1 - np.exp(ll)), 1e-12) / 2 ** 15) < 4)}


def gaussian_marginal_check(params, seed=7, n=400000):
    rng = np.random.default_rng(seed)
    mu = supply_means(params)[0, :, 1]
    s = mu[None] + SUPPLY_NOISE * rng.standard_normal((n, T))
    y = s + OBS_NOISE * rng.standard_normal((n, T))
    resid = y - mu[None]
    # posterior moments by regression of s on y against the conjugate formulas
    post_mean = R.POST_VAR * (mu[None] / SUPPLY_NOISE ** 2 + y / OBS_NOISE ** 2)
    return {"marginal_var_empirical": float(resid.var()), "marginal_var_formula": R.MARG_VAR,
            "posterior_residual_var_empirical": float((s - post_mean).var()), "posterior_var_formula": R.POST_VAR,
            "posterior_residual_mean": float((s - post_mean).mean()),
            "pass": bool(abs(resid.var() / R.MARG_VAR - 1) < 0.01 and abs((s - post_mean).var() / R.POST_VAR - 1) < 0.01)}


def check_filter(seed=11):
    rng = np.random.default_rng(seed)
    results = []
    for trial in range(4):
        pmf = rng.dirichlet(np.ones(3))
        q = []
        for _ in range(2):
            m = rng.random((N_CONTEXTS, N_CONTEXTS))
            np.fill_diagonal(m, 0)
            q.append(m / m.sum(1, keepdims=True))
        init = rng.dirichlet(np.ones(N_CONTEXTS))
        phase = None if trial % 2 == 0 else 3
        model = R.ScheduleModel(pmf, init, tuple(q), phase_start=phase)
        lls = rng.normal(0, 2.0, (6, N_CONTEXTS))
        f = R.RunLengthFilter(model)
        worst = 0.0
        for t in range(6):
            post = f.update(lls[t])
            brute = R.brute_force_posterior(model, lls[: t + 1])
            worst = max(worst, float(np.max(np.abs(post - brute))))
        results.append({"trial": trial, "phase_start": phase, "max_abs_diff": worst})
    # the declared schedule model itself on a short prefix (phase boundary moved to 3)
    from .calibration import schedule_model
    model = schedule_model()
    model = R.ScheduleModel(model.run_pmf, model.initial, model.q_phase, phase_start=3)
    lls = rng.normal(0, 3.0, (5, N_CONTEXTS))
    f = R.RunLengthFilter(model)
    worst = 0.0
    for t in range(5):
        post = f.update(lls[t])
        worst = max(worst, float(np.max(np.abs(post - R.brute_force_posterior(model, lls[: t + 1])))))
    results.append({"trial": "declared_model", "phase_start": 3, "max_abs_diff": worst})
    return {"trials": results, "pass": bool(max(r["max_abs_diff"] for r in results) < 1e-10)}


def negative_supply_audit(params, seed=13, n=20000):
    ep = sample_episodes(n, seed_for(seed, "negative_audit"))
    out = {}
    rng = np.random.default_rng(seed)
    ctx = rng.integers(0, N_CONTEXTS, n)
    sup = ep.supplies(params, ctx)
    out["negative_supply_fraction_entity_steps"] = float((sup < 0).mean())
    out["episodes_with_any_negative_supply"] = float((sup < 0).any(axis=(1, 2)).mean())
    diffs, involved = [], []
    for a in range(n_actions(params)):
        act = np.full(n, a)
        r1 = simulate(params, ctx, ep, act, sup)
        r0 = simulate(params, ctx, ep, act, np.maximum(sup, 0.0))
        diffs.append(float(r0.joint_survival.mean() - r1.joint_survival.mean()))
        fs = r1.fail_step
        for i in range(3):
            failed = fs[:, i] < SURVIVED
            steps = np.arange(T)[None] < fs[:, i:i + 1]
            neg_before = ((sup[:, :, i] < 0) & steps).any(1)
            involved.append((int((failed & neg_before).sum()), int(failed.sum())))
    out["joint_survival_gain_if_supplies_clipped_at_0_by_action"] = diffs
    tot = np.array(involved).sum(0)
    out["failures_with_negative_supply_at_or_before_failure"] = float(tot[0] / max(tot[1], 1))
    out["pass"] = bool(max(abs(d) for d in diffs) < 0.002)
    return out


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()
