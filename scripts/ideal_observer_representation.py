"""Bayes ideal-observer information audit of the representation-learning qualification.

POST HOC / EXPLORATORY / NOT DECISION-BEARING. This script fits NO learner. It
evaluates the KNOWN simulator on exactly the saved endpoint query panels,
supports, truths, masks and training-only marginals of the locked qualification
run (runs/representation_learning_qualification, seeds 18101-18106) and scores
three analysis references with the locked cell rules (scripts/ideal_observer_load.py
``score_cells``), paired with the recomputed learner:

  O  oracle: P(3-horizon survival | observable case features, action, TRUE law),
     marginalising only the hidden per-step supply noise by Monte Carlo;
  I  ideal observer: uniform prior over the 5 laws of reference_laws(seed),
     posterior from each saved 32-record support (likelihood of every record's
     observed survival pattern for its performed action), posterior-mixture
     prediction; cue flips flip only the query signal bit, posterior fixed;
  P  prior-only mixture: uniform over the 5 laws, no support.

Law identities, hidden modes, realised supplies and counterfactual outcomes are
EVALUATOR-ONLY. O, I and P are analysis references, not learners; nothing here
alters the locked STOP decision.

Usage (Windows; never writes bytecode):
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/Scripts/python.exe -B \
      scripts/ideal_observer_representation.py --out reports/information_audit/ideal_observer
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
try:  # pragma: no cover - environment dependent
    import acp_cl  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(ROOT / "src"))

import ideal_observer_mc as MC  # noqa: E402

LABEL = "POST HOC / EXPLORATORY / NOT DECISION-BEARING"
PREDICTORS = ("O", "I", "P")
LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
PRIMARY_LAMBDA = 1e-4
ROLES = ("baseA", "baseB", "stage1", "stage2", "stage3")
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:7.1f}s] {message}", file=sys.stderr, flush=True)


def require(value, message):
    if not value:
        raise AssertionError(message)


def loader():
    import ideal_observer_load as L  # heavy import kept out of spawned workers
    return L


def design():
    from acp_cl.representation_learning.design import reference_laws
    from acp_cl.acquisition.world import order, stage_law
    return reference_laws, order, stage_law


def law_dict(law):
    return dict(mode=law.mode, active=list(law.active))


# ----------------------------------------------------------------- case data

def gauge_levels(observations):
    """Reserve gauge levels read from the (observable) query/support images."""
    levels = []
    for start in (1, 9):
        block = observations[:, :, 0:4, start:start + 6].reshape(len(observations), -1)
        require(np.all(block == block[:, :1]), "gauge region is not constant")
        levels.append(block[:, 0].astype(np.float64))
    return np.stack(levels, axis=1)


def seed_case_data(endpoints, regenerated, seed):
    """Per-seed query panel and support rows; asserted identical across laws/phases."""
    reference_laws, _, _ = design()
    laws = reference_laws(seed)
    regen = regenerated[seed, "interleaved"]
    first = regen.queries[0]
    for phase in ("fresh_1", "fresh_2", "fresh_3", "interleaved"):
        for slot, query in regenerated[seed, phase].queries.items():
            for name in ("observations",):
                require(np.array_equal(getattr(query.cases, name), getattr(first.cases, name)), "query images differ")
            require(np.array_equal(query.reserves, first.reserves) and np.array_equal(query.factors, first.factors)
                    and np.array_equal(query.signals, first.signals), "query features differ across laws")
            law_index = laws.index(query.law)
            require(np.array_equal(query.truth, regen.queries[law_index].truth), "truth differs across phases")
    supports = [regen.supports[0, r] for r in (0, 1)]
    for phase in ("fresh_1", "fresh_2", "fresh_3", "interleaved"):
        for (slot, rep), support in regenerated[seed, phase].supports.items():
            ref = supports[rep]
            require(np.array_equal(support.cases.observations, ref.cases.observations)
                    and np.array_equal(support.cases.base.reserves, ref.cases.base.reserves)
                    and np.array_equal(support.cases.factors, ref.cases.factors)
                    and np.array_equal(support.cases.signals, ref.cases.signals)
                    and np.array_equal(support.actions, ref.actions), "support features differ across laws")
    levels_q = gauge_levels(first.cases.observations)
    require(np.array_equal(levels_q, np.rint(first.reserves / 18 * 240)), "query gauge levels differ from reserves")
    support_obs = np.concatenate([s.cases.observations for s in supports])
    support_res = np.concatenate([s.cases.base.reserves for s in supports])
    levels_s = gauge_levels(support_obs)
    require(np.array_equal(levels_s, np.rint(support_res / 18 * 240)), "support gauge levels differ from reserves")
    outcomes = np.zeros((5, 2, 32, 3), np.uint8)
    ep = endpoints[seed, "interleaved"]
    for target in ep.targets:
        for rep, probe in enumerate(target.correct):
            outcomes[target.slot, rep] = probe.support_outcomes
    for s in (1, 2, 3):
        fresh = endpoints[seed, f"fresh_{s}"].targets[0]
        for rep, probe in enumerate(fresh.correct):
            require(np.array_equal(probe.support_outcomes, outcomes[laws.index(fresh.law), rep]),
                    "fresh and interleaved supports differ for the same law")
    return dict(
        laws=laws,
        query=dict(reserves=np.ascontiguousarray(first.reserves), levels=levels_q,
                   factors=np.ascontiguousarray(first.factors), signals=np.ascontiguousarray(first.signals)),
        support=dict(reserves=support_res, levels=levels_s,
                     factors=np.concatenate([s.cases.factors for s in supports]),
                     signals=np.concatenate([s.cases.signals for s in supports]),
                     actions=np.concatenate([s.actions for s in supports]).astype(np.int64)),
        support_outcomes=outcomes,
        truth=np.stack([regen.queries[i].truth for i in range(5)]),
        masks={c: regen.queries[0].affected(c) for c in (0, 1, 2)})


# ----------------------------------------------------------- verifications

def verify_realised(regenerated, seeds):
    """(a) Feeding the realised supplies through the reused physics reproduces saved truth."""
    compared = dict(query_arrays=0, query_cells=0, support_arrays=0, support_cells=0,
                    flipped_unaffected_rows=0, efficiency_delay_arrays=0)
    for seed in seeds:
        for phase in ("fresh_1", "fresh_2", "fresh_3", "interleaved"):
            regen = regenerated[seed, phase]
            for slot, query in regen.queries.items():
                base = query.cases.base
                require(np.array_equal(MC.base_efficiency(query.factors), base.efficiency)
                        and np.array_equal(MC.base_delay(query.factors), base.delay), "base efficiency/delay differ")
                compared["efficiency_delay_arrays"] += 2
                for action in range(5):
                    out = MC.simulate(query.reserves, query.factors, query.signals, base.supplies, query.law,
                                      np.full(len(query.reserves), action))
                    require(np.array_equal(out.survival, query.truth[:, action]), "realised supplies miss truth")
                    compared["query_arrays"] += 1
                    compared["query_cells"] += out.survival.size
                    for cue in query.law.active:
                        signals = query.signals.copy()
                        signals[:, cue] ^= 1
                        flipped = MC.simulate(query.reserves, query.factors, signals, base.supplies, query.law,
                                              np.full(len(query.reserves), action))
                        outside = ~query.affected(cue)
                        require(np.array_equal(flipped.survival[outside], query.truth[outside, action]),
                                "cue flip changed physics outside its affected rows")
                        compared["flipped_unaffected_rows"] += int(outside.sum())
            for (slot, rep), support in regen.supports.items():
                base = support.cases.base
                out = MC.simulate(base.reserves, support.cases.factors, support.cases.signals, base.supplies,
                                  support.law, support.actions)
                require(np.array_equal(out.survival, support.outcomes), "realised support outcomes differ")
                saved = regen.endpoint.targets[slot].correct[rep].support_outcomes
                require(np.array_equal(out.survival, saved), "saved support outcomes differ")
                compared["support_arrays"] += 1
                compared["support_cells"] += out.survival.size
                for action in range(5):
                    out = MC.simulate(base.reserves, support.cases.factors, support.cases.signals, base.supplies,
                                      support.law, np.full(len(base.reserves), action))
                    require(np.array_equal(out.survival, support.truth[:, action]), "support counterfactuals differ")
                    compared["support_arrays"] += 1
                    compared["support_cells"] += out.survival.size
    return dict(exact=True, **compared)


def verify_dedup(case_data, seeds, kb=4):
    """Row-type/action deduplicated MC counts equal a naive full simulation exactly."""
    checked = 0
    for seed in seeds:
        cd = case_data[seed]
        for gauge in (False, True):
            task = make_task(cd, seed, channel=99, replicate=0, batch=0, kb=kb, gauge=gauge)
            fast = MC.panel_task(task)["query_counts"]
            slow = MC.naive_panel_counts(task)
            require(np.array_equal(fast, slow), "deduplicated MC differs from naive simulation")
            checked += fast.size
    return dict(exact=True, count_cells_compared=checked, samples_per_case=kb)


def moment_z(real, mc):
    """z statistics for mean, variance and zero-clip rate (realised vs MC)."""
    n, m = len(real), len(mc)
    mu, var = mc.mean(), mc.var()
    z_mean = (real.mean() - mu) / math.sqrt(var / n + var / m)
    m4 = np.mean((mc - mu) ** 4)
    se_var = math.sqrt(max(m4 - var ** 2, 1e-300) * (1 / n + 1 / m))
    z_var = (real.var() - var) / se_var
    p_mc, p_r = float(np.mean(mc == 0)), float(np.mean(real == 0))
    expected = p_mc * n
    z_zero = ((p_r - p_mc) / math.sqrt(p_mc * (1 - p_mc) * (1 / n + 1 / m))) if expected >= 5 else None
    return dict(n_realised=n, n_mc=m, mean_realised=float(real.mean()), mean_mc=float(mu),
                sd_realised=float(real.std()), sd_mc=float(math.sqrt(var)),
                zero_rate_realised=p_r, zero_rate_mc=p_mc, zeros_realised=int((real == 0).sum()),
                zeros_expected=expected, z_mean=float(z_mean), z_var=float(z_var),
                z_zero=None if z_zero is None else float(z_zero))


def verify_supply_statistics(regenerated, seeds, samples=64):
    """(b) Sampled supply statistics per step/participant vs realised supplies of saved cases."""
    reference_laws, _, _ = design()
    base_rows, eff_rows = [], []
    real_base, mc_base, keys_base = [], [], []
    real_eff, mc_eff, keys_eff = [], [], []
    for seed in seeds:
        rng = np.random.default_rng([MC.MC_ROOT, 4, seed])
        regen = regenerated[seed, "interleaved"]
        law_b = reference_laws(seed)[1]
        law_3 = reference_laws(seed)[4]
        sets = [(regen.queries[1].cases, regen.queries[1].realised, regen.queries[4].realised)]
        for rep in (0, 1):
            sets.append((regen.supports[1, rep].cases, regen.supports[1, rep].realised,
                         regen.supports[4, rep].realised))
        for cases, realised_b, realised_3 in sets:
            factors, signals, reserves = cases.factors, cases.signals, cases.base.reserves
            n = len(factors)
            frame0 = MC.mode_frame(realised_b["base_supplies"], law_b.mode)       # undo the hidden swap
            noise = rng.normal(0., MC.TransferWorld.supply_noise, (samples, n, MC.STEPS, 2))
            sampled0 = np.clip(MC.supply_means(factors)[None] + noise, 0, None)
            real_base.append(frame0)
            mc_base.append(sampled0)
            keys_base.append(np.broadcast_to(factors[:, 0], (n,)))
            # law-effective supplies under the stage-3 law (cue 2 active in every seed)
            sampled_law = MC.mode_frame(sampled0, law_3.mode).reshape(samples * n, MC.STEPS, 2)
            cases_mc = MC.law_cases(MC.tile_rows(reserves, samples), MC.tile_rows(factors, samples),
                                    MC.tile_rows(signals, samples), sampled_law, law_3)
            eff = MC.world().physics(cases_mc, np.zeros(samples * n, np.int64)).supplies.reshape(samples, n, MC.STEPS, 2)
            real_eff.append(realised_3["supplies"])
            mc_eff.append(eff)
            source_eff = factors[:, 0] * (1 - 2 * law_3.mode)
            affected = (factors[:, 1:] == 1).any(axis=1)
            keys_eff.append(np.stack([source_eff, signals[:, 2], affected.astype(int)], axis=1))
    real_base_all = np.concatenate(real_base)
    mc_base_all = np.concatenate([m.transpose(1, 0, 2, 3) for m in mc_base])    # (cases, samples, 12, 2)
    source_all = np.concatenate(keys_base)
    for source in (-1, 1):
        rows = source_all == source
        for step in range(MC.STEPS):
            for participant in (0, 1):
                stat = moment_z(real_base_all[rows, step, participant],
                                mc_base_all[rows, :, step, participant].ravel())
                base_rows.append(dict(source=source, step=step, participant=participant, **stat))
    real_eff_all = np.concatenate(real_eff)
    mc_eff_all = np.concatenate([m.transpose(1, 0, 2, 3) for m in mc_eff])
    keys_all = np.concatenate(keys_eff)
    for key in sorted({tuple(k) for k in keys_all.tolist()}):
        rows = (keys_all == np.array(key)).all(axis=1)
        if key[2] == 0:
            continue  # unaffected rows equal the base supplies (covered above)
        for step in range(MC.STEPS):
            for participant in (0, 1):
                stat = moment_z(real_eff_all[rows, step, participant],
                                mc_eff_all[rows, :, step, participant].ravel())
                eff_rows.append(dict(source_effective=key[0], timing_signal=key[1], step=step,
                                     participant=participant, **stat))

    def summary(rows):
        zs = {k: np.array([r[k] for r in rows if r[k] is not None]) for k in ("z_mean", "z_var", "z_zero")}
        small = [r for r in rows if r["z_zero"] is None]
        return dict(groups=len(rows),
                    **{f"max_abs_{k}": float(np.abs(v).max()) if len(v) else None for k, v in zs.items()},
                    **{f"mean_sq_{k}": float(np.mean(v ** 2)) if len(v) else None for k, v in zs.items()},
                    **{f"count_abs_{k}_gt3": int((np.abs(v) > 3).sum()) for k, v in zs.items()},
                    rare_zero_groups=len(small),
                    rare_zero_observed=int(sum(r["zeros_realised"] for r in small)),
                    rare_zero_expected=float(sum(r["zeros_expected"] for r in small)))
    return dict(samples_per_case=samples, seeds=list(seeds),
                base=dict(summary=summary(base_rows), rows=base_rows),
                law_effective_stage3=dict(summary=summary(eff_rows), rows=eff_rows))


# ------------------------------------------------------------------ MC runs

def make_task(cd, seed, channel, replicate, batch, kb, gauge):
    return dict(key=(channel, seed, replicate, batch), channel=channel, seed=seed, replicate=replicate,
                batch=batch, kb=kb, gauge=gauge,
                laws=[(law.mode, list(law.active)) for law in cd["laws"]],
                query=cd["query"], support=cd["support"])


def run_pool(function, tasks, workers, label):
    results, sims, start = {}, 0, time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[tuple(result["key"])] = result
            sims += result["case_simulations"]
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                log(f"{label}: {done}/{len(tasks)} tasks, {sims / 1e6:.0f}M case-simulations, "
                    f"{time.time() - start:.0f}s")
    return results, sims


class Panel:
    """Per-seed MC counts, kept per (replicate, batch) for jackknife/replicate analysis."""

    def __init__(self, results, seeds, replicates, batches, kb):
        self.seeds, self.replicates, self.batches, self.kb = list(seeds), list(replicates), batches, kb
        self.q, self.s, self.life, self.joint = {}, {}, {}, {}
        for seed in seeds:
            for rep in replicates:
                life = np.zeros((5, 512, 5, MC.STEPS + 1), np.int64)
                joint = np.zeros((5, 512 * 1024), np.int64)
                for b in range(batches):
                    r = results[next(k for k in results if k[1:] == (seed, rep, b))]
                    self.q[seed, rep, b] = r["query_counts"].astype(np.int64)
                    self.s[seed, rep, b] = r["support_counts"].astype(np.int64)
                    life += r["lifetime"]
                    for law, (values, freq) in enumerate(r["joint"]):
                        np.add.at(joint[law], values, freq)
                self.life[seed, rep] = life
                self.joint[seed, rep] = joint.reshape(5, 512, 1024)

    def estimate(self, keep=None, drop=None):
        """Survival probabilities from the selected (replicate, batch) set."""
        keep = self.replicates if keep is None else keep
        p, ps = {}, {}
        for seed in self.seeds:
            chosen = [(r, b) for r in keep for b in range(self.batches) if (r, b) != drop]
            total_q = sum(self.q[seed, r, b] for r, b in chosen)
            total_s = sum(self.s[seed, r, b] for r, b in chosen)
            k = len(chosen) * self.kb
            p[seed], ps[seed] = total_q / k, total_s / k
        return dict(p=p, ps=ps, samples=len(keep) * self.batches * self.kb - (self.kb if drop else 0))


# ------------------------------------------------------ forecasts & scoring

def pattern_probs(survival):
    """(...,3) survival probabilities at 4/8/12 -> (...,4) P(number of horizons survived)."""
    shape = survival.shape[:-1]
    s = np.concatenate([np.ones((*shape, 1)), survival, np.zeros((*shape, 1))], axis=-1)
    return np.clip(s[..., :-1] - s[..., 1:], 0, 1)


def posteriors(case_data, estimate, lam):
    """Uniform-prior posterior over the 5 reference laws for every (target, replicate) support."""
    weights, loglik = {}, {}
    for seed in estimate["p"]:
        probs = pattern_probs(estimate["ps"][seed])                   # (5 laws, 64, 4)
        probs = (1 - lam) * probs + lam / 4
        w = np.zeros((5, 2, 5))
        ll = np.zeros((5, 2, 5))
        for target in range(5):
            for rep in (0, 1):
                observed = case_data[seed]["support_outcomes"][target, rep].sum(axis=1)
                rows = np.arange(32) + 32 * rep
                values = np.log(probs[:, rows, observed]).sum(axis=1)
                ll[target, rep] = values
                x = np.exp(values - values.max())
                w[target, rep] = x / x.sum()
        weights[seed], loglik[seed] = w, ll
    return weights, loglik


def forecasts(endpoints, estimate, weights, kind):
    reference_laws, _, _ = design()
    out = {}
    for (seed, phase), endpoint in endpoints.items():
        laws = reference_laws(seed)
        p = None if kind == "learner" else estimate["p"][seed]
        for ev in endpoint.evaluations:
            target = laws.index(ev.law)
            variant = 0 if ev.cue is None else 1 + ev.cue
            if kind == "O":
                f = p[target, variant]
            elif kind == "P":
                f = p[:, variant].mean(axis=0)
            elif kind == "I":
                f = np.tensordot(weights[seed][target, ev.replicate], p[:, variant], axes=1)
            elif kind == "learner":
                f = ev.probabilities
            else:
                raise ValueError(kind)
            out[ev.key] = np.minimum.accumulate(np.asarray(f, dtype=np.float64), axis=-1)
    return out


def score_all(endpoints, case_data, estimate, lam=PRIMARY_LAMBDA):
    L = loader()
    weights, loglik = posteriors(case_data, estimate, lam)
    scored, maps = {}, {}
    for kind in PREDICTORS:
        maps[kind] = forecasts(endpoints, estimate, weights, kind)
        scored[kind] = L.score_cells(endpoints, maps[kind])
    return scored, maps, weights, loglik


def cell_values(scored):
    """{predictor: {cell: dict(mean, per_seed)}}"""
    return {kind: {c["name"]: dict(mean=c["metric"]["mean"], per_seed=dict(c["per_seed"]))
                   for c in result["cells"]} for kind, result in scored.items()}


def jackknife(values, full):
    """Delete-one-batch jackknife SE for a list of dicts of per-cell values."""
    n = len(values)
    out = {}
    for kind in full:
        out[kind] = {}
        for name, item in full[kind].items():
            means = np.array([v[kind][name]["mean"] for v in values])
            per_seed = {s: np.array([v[kind][name]["per_seed"][s] for v in values]) for s in item["per_seed"]}
            se = lambda x: float(math.sqrt((n - 1) / n * np.sum((x - x.mean()) ** 2)))  # noqa: E731
            out[kind][name] = dict(mean=se(means), per_seed={s: se(x) for s, x in per_seed.items()})
    return out


# --------------------------------------------------------- information/KL

def smooth(probabilities, lam):
    return (1 - lam) * probabilities + lam / probabilities.shape[-1]


def kl_terms(p1, p2, lam):
    """Expected per-record KL (nats) and its decisive component (P2 MC count zero)."""
    q1, q2 = smooth(p1, lam), smooth(p2, lam)
    terms = np.where(p1 > 0, q1 * np.log(q1 / q2), 0.)
    kl = terms.sum(axis=-1)
    decisive_mass = np.where((p1 > 0) & (p2 == 0), p1, 0.).sum(axis=-1)
    decisive_kl = np.where((p1 > 0) & (p2 == 0), terms, 0.).sum(axis=-1)
    return kl, decisive_mass, decisive_kl


def channel_distributions(panel, seed, reps):
    k = len(reps) * panel.batches * panel.kb
    counts = sum(panel.q[seed, r, b][:, 0] for r in reps for b in range(panel.batches))   # (5,512,5,3)
    pattern = pattern_probs(counts / k)                                                      # (5,512,5,4)
    lifetime = sum(panel.life[seed, r] for r in reps) / k                                    # (5,512,5,13)
    joint = sum(panel.joint[seed, r] for r in reps) / k                                      # (5,512,1024)
    return dict(pattern=pattern, lifetime=lifetime, joint=joint)


def information(panel, seeds, lam_values):
    """Per-record expected KL for every ordered law pair, three feedback channels."""
    reference_laws, order, _ = design()
    result = dict(per_seed={}, lambdas=list(lam_values))
    variants = {"pooled": panel.replicates, **{f"replicate_{r}": [r] for r in panel.replicates}}
    for seed in seeds:
        laws = reference_laws(seed)
        entry = dict(laws=[law_dict(law) for law in laws], variants={})
        for name, reps in variants.items():
            dist = channel_distributions(panel, seed, reps)
            block = {}
            for lam in lam_values:
                mats = {}
                for channel in ("pattern", "lifetime", "joint"):
                    kl = np.zeros((5, 5))
                    dmass = np.zeros((5, 5))
                    dkl = np.zeros((5, 5))
                    for i in range(5):
                        for j in range(5):
                            if i == j:
                                continue
                            a, b = dist[channel][i], dist[channel][j]
                            v, m, d = kl_terms(a, b, lam)
                            kl[i, j], dmass[i, j], dkl[i, j] = v.mean(), m.mean(), d.mean()
                    mats[channel] = dict(kl=kl.tolist(), decisive_mass=dmass.tolist(), decisive_kl=dkl.tolist())
                block[repr(lam)] = mats
            entry["variants"][name] = block
        result["per_seed"][str(seed)] = entry
    # adjacent single-cue pairs, the mode pair and the delay pair, by role
    rows = []
    for seed in seeds:
        o = order(seed)
        pairs = [("mode", 0, 1, None), (f"add_cue_{o[0]}", 2, 1, o[0]),
                 (f"add_cue_{o[1]}", 3, 2, o[1]), (f"add_cue_{o[2]}", 4, 3, o[2])]
        for label, i, j, cue in pairs:
            item = dict(seed=seed, pair=label, cue=cue, law_i=ROLES[i], law_j=ROLES[j],
                        delay_pair=cue == 1, values={})
            for name in variants:
                for lam in lam_values:
                    block = result["per_seed"][str(seed)]["variants"][name][repr(lam)]
                    for channel in ("pattern", "lifetime", "joint"):
                        kij = block[channel]["kl"][i][j]
                        kji = block[channel]["kl"][j][i]
                        item["values"][f"{name}|{lam!r}|{channel}"] = dict(
                            kl_ij=kij, kl_ji=kji,
                            decisive_mass_ij=block[channel]["decisive_mass"][i][j],
                            decisive_mass_ji=block[channel]["decisive_mass"][j][i],
                            records_20to1_ij=math.log(20) / kij if kij > 0 else None,
                            records_20to1_ji=math.log(20) / kji if kji > 0 else None)
            rows.append(item)
    result["pairs"] = rows
    return result


def pair_summary(info):
    """Cross-seed means by pair type; 'with' = law containing the added cue (or baseA for mode).

    KL(with||without) is the expected log-odds gain per record when the law
    WITH the dependency is true. Plug-in KL from MC histograms is biased
    upward by O(1/K); bias is estimated from the two half-sample replicates
    (bias(K) ~ mean_replicate - pooled) and a Richardson value 2*pooled - mean_replicate.
    """
    out = {}
    for label in ("mode", "add_cue_0", "add_cue_1", "add_cue_2"):
        rows = [r for r in info["pairs"] if r["pair"] == label]
        block = {}
        for lam in info["lambdas"]:
            for channel in ("pattern", "lifetime", "joint"):
                key = f"{lam!r}|{channel}"
                item = {}
                for direction, field in (("with_true", "kl_ij"), ("without_true", "kl_ji")):
                    pooled = np.array([r["values"][f"pooled|{key}"][field] for r in rows])
                    reps = np.array([[r["values"][f"replicate_{k}|{key}"][field] for k in (0, 1)] for r in rows])
                    mass_field = "decisive_mass_ij" if field == "kl_ij" else "decisive_mass_ji"
                    decisive = np.array([r["values"][f"pooled|{key}"][mass_field] for r in rows])
                    rich = 2 * pooled - reps.mean(axis=1)
                    item[direction] = dict(
                        kl_mean=float(pooled.mean()), kl_min=float(pooled.min()), kl_max=float(pooled.max()),
                        kl_per_seed={str(r["seed"]): float(v) for r, v in zip(rows, pooled)},
                        replicate_mean=float(reps.mean()), bias_estimate=float(reps.mean() - pooled.mean()),
                        richardson_mean=float(rich.mean()),
                        records_for_20to1=float(math.log(20) / pooled.mean()),
                        records_for_20to1_range=[float(math.log(20) / pooled.max()), float(math.log(20) / pooled.min())],
                        decisive_mass_mean=float(decisive.mean()))
                block[key] = item
        out[label] = dict(seeds=[r["seed"] for r in rows], roles=[[r["law_i"], r["law_j"]] for r in rows], values=block)
    return out


def zero_likelihood_counts(case_data, estimate):
    """Support records whose observed pattern has zero MC probability, by law (before smoothing)."""
    reference_laws, _, _ = design()
    true_zero, other_zero, records = 0, 0, 0
    per_law_zero = np.zeros(5, int)
    for seed in estimate["p"]:
        probs = pattern_probs(estimate["ps"][seed])
        for target in range(5):
            for rep in (0, 1):
                observed = case_data[seed]["support_outcomes"][target, rep].sum(axis=1)
                rows = np.arange(32) + 32 * rep
                zero = probs[:, rows, observed] == 0          # (5 laws, 32)
                true_zero += int(zero[target].sum())
                other_zero += int(np.delete(zero, target, axis=0).sum())
                per_law_zero += zero.sum(axis=1)
                records += 32
    return dict(records=records, zero_under_true_law=true_zero, zero_under_other_laws=other_zero,
                note="a zero under the true law can only be an MC miss; smoothing lambda protects against it")


# --------------------------------------------------- identification check

def identification_task(task):
    """Fresh 32-record support batches (analysis-only seeds) and MC pattern probabilities."""
    from acp_cl.acquisition.world import Law
    from acp_cl.persistence.study import trial_seed
    world = MC.world()
    laws = [Law(m, tuple(a)) for m, a in task["laws"]]
    require(laws[0].mode == laws[1].mode, "identification pairs share the hidden mode")
    rows_all, outcomes_all, feats = [], [], dict(factors=[], signals=[], reserves=[], actions=[])
    for b in range(task["batches"]):
        seed_b = trial_seed(task["root"], "ideal_observer_identification", [task["seed"], task["pair"], task["replicate"], b])
        per_law = []
        for law in laws:
            experience = world.experience(law, 32, seed_b)
            cases = world.dataset(law, 32, seed_b)
            require(np.array_equal(experience.observations, cases.observations), "support dataset path differs")
            per_law.append((experience, cases))
        (e0, c0), (e1, c1) = per_law
        require(np.array_equal(c0.base.supplies, c1.base.supplies) and np.array_equal(e0.actions, e1.actions)
                and np.array_equal(c0.signals, c1.signals), "same-mode pair supports differ")
        rows = np.zeros(32, bool)
        for cue in set(laws[0].active) ^ set(laws[1].active):
            rows |= MC.mask_affected_like(c0.factors, cue)
        require(np.array_equal(e0.survival[~rows], e1.survival[~rows]), "outcomes differ outside affected rows")
        rows_all.append(rows)
        outcomes_all.append(np.stack([e0.survival.sum(axis=1), e1.survival.sum(axis=1)]))
        feats["factors"].append(c0.factors)
        feats["signals"].append(c0.signals)
        feats["reserves"].append(c0.base.reserves)
        feats["actions"].append(e0.actions.astype(np.int64))
    rows = np.concatenate(rows_all)
    outcomes = np.concatenate(outcomes_all, axis=1)                  # (2 truths, n records)
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    sub = dict(kb=task["kb"], channel=task["channel"], seed=task["seed"], replicate=task["replicate"],
               batch=task["pair"], laws=task["laws"], key=task["key"],
               **{k: v[rows] for k, v in feats.items()})
    mc = MC.pair_task(sub)
    probabilities = np.full((2, len(rows), 4), np.nan)
    probabilities[:, rows] = mc["probabilities"]
    return dict(key=task["key"], rows=rows, outcomes=outcomes, probabilities=probabilities,
                case_simulations=mc["case_simulations"])


def identification_summary(results, lam_values, sizes=(32, 128, 512)):
    out = {}
    for lam in lam_values:
        block = {}
        for truth in (0, 1):
            per_size = {n: [] for n in sizes}
            llr_records = []
            for r in results:
                rows, probs, observed = r["rows"], r["probabilities"], r["outcomes"][truth]
                llr = np.zeros(len(rows))
                q = smooth(probs[:, rows], lam)
                idx = np.arange(rows.sum())
                obs = observed[rows]
                llr[rows] = np.log(q[truth, idx, obs]) - np.log(q[1 - truth, idx, obs])
                llr_records.append(llr)
                for n in sizes:
                    per_size[n].append(llr[:n].sum())
            llr_all = np.concatenate(llr_records)
            stats = {}
            for n in sizes:
                odds = np.array(per_size[n])
                post = 1 / (1 + np.exp(-odds))
                stats[str(n)] = dict(mean_posterior_true=float(post.mean()), median_posterior_true=float(np.median(post)),
                                     frac_odds_ge_20=float(np.mean(odds >= math.log(20))),
                                     frac_odds_le_1_20=float(np.mean(odds <= -math.log(20))),
                                     mean_log_odds=float(odds.mean()), sd_log_odds=float(odds.std()))
            block[f"truth_{truth}"] = dict(sizes=stats, mean_llr_per_record=float(llr_all.mean()),
                                           se_llr_per_record=float(llr_all.std() / math.sqrt(len(llr_all))),
                                           replicates=len(results))
        out[repr(lam)] = block
    return out


# ------------------------------------------------------ decision relevance

def decision_relevance(endpoints, case_data, maps, learner_scored_cells):
    L = loader()
    reference_laws, order, _ = design()
    predictor_maps = dict(maps)
    predictor_maps["learner"] = forecasts(endpoints, None, None, "learner")
    rows = []
    cells = [c for c in learner_scored_cells if c["kind"] == "cue_benefit"]
    for cell in cells:
        phase, cue = cell["phase"], cell["cue"]
        entry = dict(cell=cell["name"], cue=None, per_predictor={})
        for kind, fmap in predictor_maps.items():
            per_seed = {}
            for seed in cell["per_seed"]:
                seed = int(seed)
                if phase == "fresh":
                    endpoint = next(endpoints[seed, f"fresh_{s}"] for s in (1, 2, 3)
                                    if (cell["stage"] is None and endpoints[seed, f"fresh_{s}"].introduced_cue == cell["cue"])
                                    or (cell["stage"] == s))
                    target = endpoint.targets[0]
                    c = endpoint.introduced_cue
                else:
                    endpoint = endpoints[seed, "interleaved"]
                    target = endpoint.targets[cell["slot"]]
                    c = cue
                entry["cue"] = "introduced" if (phase == "fresh" and cell["stage"] is not None) else c
                values = []
                for rep in (0, 1):
                    correct, flip = target.correct[rep], target.flips[c][rep]
                    fields = (flip.truth, flip.affected, flip.valid)
                    a = L.locked_score(fmap[correct.key], *fields)
                    b = L.locked_score(fmap[flip.key], *fields)
                    values.append((a["focus_survival"], b["focus_survival"],
                                   float(flip.truth[flip.affected][:, :, -1].max(axis=1).mean()),
                                   float(flip.truth[flip.affected, 0, -1].mean())))
                v = np.mean(values, axis=0)
                per_seed[str(seed)] = dict(correct=float(v[0]), flipped=float(v[1]), loss=float(v[0] - v[1]),
                                           clairvoyant=float(v[2]), no_transfer=float(v[3]))
            means = {k: float(np.mean([x[k] for x in per_seed.values()]))
                     for k in ("correct", "flipped", "loss", "clairvoyant", "no_transfer")}
            entry["per_predictor"][kind] = dict(mean=means, per_seed=per_seed)
        rows.append(entry)
    # all-case H=12 survival of the first-argmax action per interleaved slot (correct probes)
    slots = []
    for slot in range(5):
        item = dict(slot=slot, role=ROLES[slot], per_predictor={})
        for kind, fmap in predictor_maps.items():
            per_seed = {}
            for seed in sorted({s for s, _ in endpoints}):
                target = endpoints[seed, "interleaved"].targets[slot]
                vals = []
                for rep in (0, 1):
                    probe = target.correct[rep]
                    m = L.locked_score(fmap[probe.key], probe.truth, probe.affected, probe.valid)
                    vals.append((m["survival"], m["clairvoyant_upper"], m["no_transfer"]))
                v = np.mean(vals, axis=0)
                per_seed[str(seed)] = dict(survival=float(v[0]), clairvoyant=float(v[1]), no_transfer=float(v[2]))
            item["per_predictor"][kind] = dict(
                mean={k: float(np.mean([x[k] for x in per_seed.values()])) for k in ("survival", "clairvoyant", "no_transfer")},
                per_seed=per_seed)
        slots.append(item)
    return dict(cue_cells=rows, interleaved_slots=slots)


# ------------------------------------------------------------------- main

def posterior_diagnostics(case_data, weights, seeds):
    reference_laws, _, _ = design()
    rows = []
    for seed in seeds:
        laws = reference_laws(seed)
        for target in range(5):
            for rep in (0, 1):
                w = weights[seed][target, rep]
                delay = np.array([1 in law.active for law in laws])
                same_mode = np.array([law.mode == laws[target].mode for law in laws])
                rows.append(dict(seed=seed, slot=target, role=ROLES[target], replicate=rep,
                                 law=law_dict(laws[target]), weights=w.tolist(),
                                 mass_true=float(w[target]), mass_true_mode=float(w[same_mode].sum()),
                                 target_delay_active=bool(delay[target]),
                                 mass_delay_active=float(w[delay].sum()),
                                 map_law=int(np.argmax(w))))
    summary = {}
    for target in range(5):
        items = [r for r in rows if r["slot"] == target]
        delay_items = [r for r in items if r["target_delay_active"]]
        summary[ROLES[target]] = dict(
            mean_mass_true=float(np.mean([r["mass_true"] for r in items])),
            min_mass_true=float(np.min([r["mass_true"] for r in items])),
            mean_mass_true_mode=float(np.mean([r["mass_true_mode"] for r in items])),
            map_correct_rate=float(np.mean([r["map_law"] == target for r in items])),
            delay_targets=len(delay_items),
            mean_mass_delay_active_if_delay_target=(float(np.mean([r["mass_delay_active"] for r in delay_items]))
                                                    if delay_items else None))
    return dict(rows=rows, summary=summary)


def delay_cell_posteriors(diag, learner_cells):
    out = {}
    for cell in learner_cells:
        if cell["kind"] != "cue_benefit" or cell["phase"] != "interleaved" or cell["cue"] != 1:
            continue
        items = [r for r in diag["rows"] if r["slot"] == cell["slot"] and str(r["seed"]) in map(str, cell["per_seed"])]
        out[cell["name"]] = dict(
            mean_mass_true=float(np.mean([r["mass_true"] for r in items])),
            mean_mass_delay_active=float(np.mean([r["mass_delay_active"] for r in items])),
            per_seed={str(s): dict(mass_true=float(np.mean([r["mass_true"] for r in items if r["seed"] == int(s)])),
                                   mass_delay_active=float(np.mean([r["mass_delay_active"] for r in items
                                                                    if r["seed"] == int(s)])))
                      for s in cell["per_seed"]})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="NEW directory for results.json")
    parser.add_argument("--kb", type=int, default=512, help="MC samples per batch")
    parser.add_argument("--batches", type=int, default=16, help="batches per MC replicate")
    parser.add_argument("--gauge-seeds", type=int, nargs="*", default=[18101, 18104])
    parser.add_argument("--gauge-batches", type=int, default=16)
    parser.add_argument("--pair-replicates", type=int, default=64)
    parser.add_argument("--pair-kb", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--allow-outside", action="store_true", help="pilot runs outside reports/ (tests only)")
    args = parser.parse_args()
    require(1 <= args.workers <= 6, "at most 6 worker processes")
    out = args.out.resolve()
    if not args.allow_outside:
        require(out.is_relative_to(ROOT / "reports" / "information_audit" / "ideal_observer"),
                "outputs belong under reports/information_audit/ideal_observer/")
    require(not (out / "results.json").exists(), "refusing to overwrite an existing results.json")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    L = loader()
    reference_laws, order, stage_law = design()
    log("loading endpoints and regenerating evaluator cases")
    endpoints = L.load_all()
    regenerated = {key: L.regenerate(endpoint) for key, endpoint in endpoints.items()}
    seeds = sorted({seed for seed, _ in endpoints})
    source_lock = L.source_lock_check()
    require(all(v["matches"] for v in source_lock.values()), "live sources differ from the run lock")
    learner = L.score_cells(endpoints)
    comparison = L.compare_with_summary(learner)
    require(comparison["max_abs_diff"] == 0.0, "learner rescoring differs from summary.json")
    case_data = {seed: seed_case_data(endpoints, regenerated, seed) for seed in seeds}

    log("verification (a): realised supplies -> saved truth")
    realised_check = verify_realised(regenerated, seeds)
    log("verification: deduplicated vs naive MC")
    dedup_check = verify_dedup(case_data, seeds)
    log("verification (b): supply statistics")
    supply_check = verify_supply_statistics(regenerated, seeds)

    log("primary MC (exact reserves, two independent replicates)")
    tasks = [make_task(case_data[s], s, 1, rep, b, args.kb, False)
             for s in seeds for rep in (0, 1) for b in range(args.batches)]
    results, sims_primary = run_pool(MC.panel_task, tasks, args.workers, "primary")
    panel = Panel(results, seeds, (0, 1), args.batches, args.kb)
    del results

    log("scoring pooled / replicate / jackknife estimates")
    pooled = panel.estimate()
    scored, maps, weights, loglik = score_all(endpoints, case_data, pooled)
    full = cell_values(scored)
    by_rep, maps_rep = {}, {}
    for rep in (0, 1):
        s_rep, maps_rep[rep], _, _ = score_all(endpoints, case_data, panel.estimate(keep=[rep]))
        by_rep[rep] = cell_values(s_rep)
    deletions_pooled, deletions_rep = [], {0: [], 1: []}
    for rep in (0, 1):
        for b in range(args.batches):
            deletions_pooled.append(cell_values(score_all(endpoints, case_data, panel.estimate(drop=(rep, b)))[0]))
            deletions_rep[rep].append(cell_values(score_all(endpoints, case_data,
                                                            panel.estimate(keep=[rep], drop=(rep, b)))[0]))
    se_pooled = jackknife(deletions_pooled, full)
    se_rep = {rep: jackknife(deletions_rep[rep], by_rep[rep]) for rep in (0, 1)}

    log("lambda sensitivity")
    lambda_cells = {}
    lambda_weights = {}
    for lam in LAMBDAS:
        s, _, w, _ = score_all(endpoints, case_data, pooled, lam)
        lambda_cells[repr(lam)] = {c["name"]: dict(mean=c["metric"]["mean"], per_seed=c["per_seed"], passed=c["passed"])
                                   for c in s["I"]["cells"]}
        lambda_weights[repr(lam)] = {str(seed): w[seed].tolist() for seed in seeds}

    log("gauge-consistent reserve sensitivity")
    gauge = None
    if args.gauge_seeds:
        gtasks = [make_task(case_data[s], s, 2, 0, b, args.kb, True)
                  for s in args.gauge_seeds for b in range(args.gauge_batches)]
        gresults, sims_gauge = run_pool(MC.panel_task, gtasks, args.workers, "gauge")
        gpanel = Panel(gresults, args.gauge_seeds, (0,), args.gauge_batches, args.kb)
        del gresults
        gest = gpanel.estimate()
        mixed = dict(p={s: gest["p"].get(s, pooled["p"][s]) for s in seeds},
                     ps={s: gest["ps"].get(s, pooled["ps"][s]) for s in seeds})
        gscored, _, gweights, _ = score_all(endpoints, case_data, mixed)
        gvals = cell_values(gscored)
        gdeletions = []
        for b in range(args.gauge_batches):
            gd = gpanel.estimate(drop=(0, b))
            gdeletions.append(cell_values(score_all(endpoints, case_data, dict(
                p={s: gd["p"].get(s, pooled["p"][s]) for s in seeds},
                ps={s: gd["ps"].get(s, pooled["ps"][s]) for s in seeds}))[0]))
        gse = jackknife(gdeletions, gvals)
        gauge = dict(seeds=args.gauge_seeds, samples_per_case=args.gauge_batches * args.kb,
                     case_simulations=sims_gauge, cells={})
        for kind in PREDICTORS:
            gauge["cells"][kind] = {}
            for name in full[kind]:
                per = {}
                for s in args.gauge_seeds:
                    if s in full[kind][name]["per_seed"]:
                        per[str(s)] = dict(exact=full[kind][name]["per_seed"][s],
                                           gauge=gvals[kind][name]["per_seed"][s],
                                           difference=gvals[kind][name]["per_seed"][s] - full[kind][name]["per_seed"][s],
                                           mc_se_gauge=gse[kind][name]["per_seed"][s],
                                           mc_se_exact=se_pooled[kind][name]["per_seed"][s])
                if per:
                    gauge["cells"][kind][name] = per
        gauge["posterior_mass_true"] = {
            str(s): dict(exact=[float(weights[s][t, r, t]) for t in range(5) for r in (0, 1)],
                         gauge=[float(gweights[s][t, r, t]) for t in range(5) for r in (0, 1)])
            for s in args.gauge_seeds}

    log("information per record (KL)")
    info = information(panel, seeds, (1e-6, 1e-4, 1e-3, 1e-2))

    # hardest pair per seed (channel i, pooled, primary lambda) and the delay pair
    hardest = []
    for seed in seeds:
        mat = np.array(info["per_seed"][str(seed)]["variants"]["pooled"][repr(PRIMARY_LAMBDA)]["pattern"]["kl"])
        sym = mat + mat.T
        np.fill_diagonal(sym, np.inf)
        i, j = np.unravel_index(np.argmin(sym), sym.shape)
        k = order(seed).index(1) + 1
        delay_pair = (k + 1, k)  # slots: stage_k law vs its predecessor (baseB = slot 1 when k = 1)
        hardest.append(dict(seed=seed, hardest_pair=sorted([int(i), int(j)]), hardest_roles=[ROLES[i], ROLES[j]],
                            hardest_sym_kl=float(sym[i, j]), delay_pair=sorted(delay_pair),
                            hardest_is_delay_pair=sorted([int(i), int(j)]) == sorted(delay_pair)))

    log("direct identification check with fresh supports")
    id_tasks, pair_meta = [], {}
    for h in hardest:
        seed = h["seed"]
        laws = reference_laws(seed)
        pairs = {tuple(h["delay_pair"])}
        if not h["hardest_is_delay_pair"] and laws[h["hardest_pair"][0]].mode == laws[h["hardest_pair"][1]].mode:
            pairs.add(tuple(h["hardest_pair"]))
        for pair in sorted(pairs):
            a, b = pair
            pair_code = a * 10 + b
            pair_meta[seed, pair_code] = dict(seed=seed, slots=[a, b], roles=[ROLES[a], ROLES[b]],
                                              laws=[law_dict(laws[a]), law_dict(laws[b])])
            for rep in range(args.pair_replicates):
                id_tasks.append(dict(key=(seed, pair_code, rep), root=MC.MC_ROOT, channel=3, seed=seed,
                                     pair=pair_code, replicate=rep, batches=16, kb=args.pair_kb,
                                     laws=[(laws[a].mode, list(laws[a].active)), (laws[b].mode, list(laws[b].active))]))
    id_results, sims_id = run_pool(identification_task, id_tasks, args.workers, "identification")
    identification = []
    for (seed, code), meta in sorted(pair_meta.items()):
        items = [id_results[k] for k in sorted(id_results) if k[0] == seed and k[1] == code]
        identification.append(dict(**meta, summary=identification_summary(items, (1e-6, 1e-4, 1e-2))))

    log("decision relevance")
    decision = decision_relevance(endpoints, case_data, maps, learner["cells"])
    decision_reps = [decision_relevance(endpoints, case_data, maps_rep[rep], learner["cells"]) for rep in (0, 1)]
    decision["mc_replicate_check"] = dict(
        max_abs_difference_cue_cells={k: max(abs(a["per_predictor"][k]["mean"][m] - b["per_predictor"][k]["mean"][m])
                                             for a, b in zip(*(d["cue_cells"] for d in decision_reps))
                                             for m in ("correct", "flipped", "loss")) for k in PREDICTORS},
        max_abs_difference_per_seed_cue_cells={k: max(abs(a["per_predictor"][k]["per_seed"][sd][m]
                                                         - b["per_predictor"][k]["per_seed"][sd][m])
                                                     for a, b in zip(*(d["cue_cells"] for d in decision_reps))
                                                     for sd in a["per_predictor"][k]["per_seed"]
                                                     for m in ("correct", "flipped", "loss")) for k in PREDICTORS},
        max_abs_difference_slots={k: max(abs(a["per_predictor"][k]["mean"]["survival"] - b["per_predictor"][k]["mean"]["survival"])
                                         for a, b in zip(*(d["interleaved_slots"] for d in decision_reps)))
                                  for k in PREDICTORS},
        note="each replicate uses half the pooled MC samples; first-argmax survival is not smooth in the forecast")
    zero_lik = zero_likelihood_counts(case_data, pooled)
    diag = posterior_diagnostics(case_data, weights, seeds)
    delay_post = delay_cell_posteriors(diag, learner["cells"])

    # ------------------------------------------------------------- assemble
    cells = []
    for c in learner["cells"]:
        name = c["name"]
        entry = dict(name=name, kind=c["kind"], phase=c["phase"], slot=c["slot"], stage=c["stage"], cue=c["cue"],
                     minimum=c["minimum"], maximum=c["maximum"],
                     threshold=(f">= {c['minimum']}" if c["minimum"] is not None else f"<= {c['maximum']}"),
                     learner=dict(mean=c["metric"]["mean"], per_seed={str(k): v for k, v in c["per_seed"].items()},
                                  passed=c["passed"]))
        for kind in PREDICTORS:
            cell = next(x for x in scored[kind]["cells"] if x["name"] == name)
            entry[kind] = dict(mean=cell["metric"]["mean"], per_seed={str(k): v for k, v in cell["per_seed"].items()},
                               passed=cell["passed"], mc_se_mean=se_pooled[kind][name]["mean"],
                               mc_se_per_seed={str(k): v for k, v in se_pooled[kind][name]["per_seed"].items()},
                               replicate_means=[by_rep[r][kind][name]["mean"] for r in (0, 1)],
                               replicate_mc_se_mean=[se_rep[r][kind][name]["mean"] for r in (0, 1)])
        cells.append(entry)

    cue_cells = [c for c in cells if c["kind"] == "cue_benefit"]
    se_values = [v for c in cue_cells for k in PREDICTORS
                 for v in [c[k]["mc_se_mean"], *c[k]["mc_se_per_seed"].values()]]
    rep_se_values = [v for c in cue_cells for k in PREDICTORS for v in c[k]["replicate_mc_se_mean"]]
    zs = []
    for c in cue_cells:
        for k in PREDICTORS:
            d = c[k]["replicate_means"][0] - c[k]["replicate_means"][1]
            s = math.sqrt(sum(x ** 2 for x in c[k]["replicate_mc_se_mean"]))
            zs.append(d / s if s > 0 else 0.)
    per_seed_rep_diffs = []
    for c in cue_cells:
        for k in PREDICTORS:
            for s in c[k]["per_seed"]:
                a = by_rep[0][k][c["name"]]["per_seed"][int(s)]
                b = by_rep[1][k][c["name"]]["per_seed"][int(s)]
                per_seed_rep_diffs.append(abs(a - b))
    mc_error = dict(
        samples_per_case_pooled=pooled["samples"], samples_per_replicate=args.batches * args.kb,
        batches_per_replicate=args.batches, batch_size=args.kb,
        max_se_cue_benefit_pooled=float(max(se_values)),
        max_se_cue_benefit_single_replicate_seed_mean=float(max(rep_se_values)),
        replicate_difference_z_cue_benefit_means=dict(max_abs=float(np.max(np.abs(zs))), rms=float(np.sqrt(np.mean(np.square(zs)))), n=len(zs)),
        max_abs_replicate_difference_cue_benefit_per_seed=float(max(per_seed_rep_diffs)),
        implied_pooled_se_from_replicates_rms=float(np.sqrt(np.mean(np.square(per_seed_rep_diffs))) / 2),
        criterion_se_below_0_0002=bool(max(se_values) < 2e-4),
        max_se_any_cell_pooled=float(max(v for c in cells for k in PREDICTORS
                                         for v in [c[k]["mc_se_mean"], *c[k]["mc_se_per_seed"].values()])))

    def mean(name, kind):
        return next(c for c in cells if c["name"] == name)[kind]["mean"]

    delay_names = [f"interleaved/stage_{s}/cue_1/cue_benefit" for s in (1, 2, 3)]
    answers = dict(
        I_passes_baseA_brier=next(c for c in cells if c["name"] == "interleaved/slot_0/absolute_brier")["I"]["passed"],
        I_baseA_brier=mean("interleaved/slot_0/absolute_brier", "I"),
        O_baseA_brier=mean("interleaved/slot_0/absolute_brier", "O"),
        learner_baseA_brier=mean("interleaved/slot_0/absolute_brier", "learner"),
        I_passes_delay_cells={n: next(c for c in cells if c["name"] == n)["I"]["passed"] for n in delay_names},
        I_delay_cue_benefit={n: mean(n, "I") for n in delay_names},
        O_delay_cue_benefit_fresh=mean("fresh/cue_1/cue_benefit", "O"),
        learner_delay_cue_benefit_fresh=mean("fresh/cue_1/cue_benefit", "learner"),
        learner_fraction_of_O_fresh=mean("fresh/cue_1/cue_benefit", "learner") / mean("fresh/cue_1/cue_benefit", "O"),
        O_delay_cue_benefit_interleaved={n: mean(n, "O") for n in delay_names},
        learner_delay_cue_benefit_interleaved={n: mean(n, "learner") for n in delay_names},
        learner_fraction_of_O_interleaved={n: mean(n, "learner") / mean(n, "O") for n in delay_names},
        I_fraction_of_O_interleaved={n: mean(n, "I") / mean(n, "O") for n in delay_names},
        P_delay_cue_benefit={n: mean(n, "P") for n in delay_names},
        excess_brier_learner_minus_I={
            ROLES[s]: dict(mean=mean(f"interleaved/slot_{s}/absolute_brier", "learner")
                           - mean(f"interleaved/slot_{s}/absolute_brier", "I"),
                           per_seed={k: next(c for c in cells if c["name"] == f"interleaved/slot_{s}/absolute_brier")["learner"]["per_seed"][k]
                                     - next(c for c in cells if c["name"] == f"interleaved/slot_{s}/absolute_brier")["I"]["per_seed"][k]
                                     for k in map(str, seeds)}) for s in range(5)},
        excess_brier_learner_minus_O={ROLES[s]: mean(f"interleaved/slot_{s}/absolute_brier", "learner")
                                      - mean(f"interleaved/slot_{s}/absolute_brier", "O") for s in range(5)},
        failed_cells={k: [c["name"] for c in cells if not c[k]["passed"]] for k in ("learner", *PREDICTORS)})

    runtime = time.time() - T0
    results_doc = dict(
        label=LABEL,
        description=("Bayes ideal-observer information audit of the locked representation-learning "
                     "qualification. No learner is fitted; O/I/P are analysis references built from "
                     "EVALUATOR-ONLY quantities (law identities, supply process). Cue benefit is a "
                     "sensitivity measure, not a proper score: I is a reference, not an upper bound."),
        locked_decision_unchanged=comparison["decision_locked"],
        environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                         workers=args.workers),
        runtime_minutes=runtime / 60,
        settings=dict(primary_lambda=PRIMARY_LAMBDA, lambdas=list(LAMBDAS), kb=args.kb, batches=args.batches,
                      mc_replicates=2, samples_per_case=pooled["samples"], mc_root_seed=MC.MC_ROOT,
                      gauge_seeds=args.gauge_seeds, gauge_batches=args.gauge_batches,
                      pair_replicates=args.pair_replicates, pair_kb=args.pair_kb,
                      case_simulations=dict(primary=sims_primary, gauge=gauge["case_simulations"] if gauge else 0,
                                            identification=sims_id)),
        verification=dict(source_lock_all_match=True, learner_rescoring=comparison,
                          realised_supplies_reproduce_truth=realised_check, dedup_exact=dedup_check,
                          supply_statistics=supply_check),
        mc_error=mc_error,
        answers=answers,
        cells=cells,
        posterior=dict(diagnostics=diag, delay_cells=delay_post, zero_mc_likelihood=zero_lik,
                       lambda_sensitivity=dict(cells=lambda_cells, weights=lambda_weights)),
        gauge_sensitivity=gauge,
        information=dict(kl=info, pair_summary=pair_summary(info), hardest_pairs=hardest,
                         identification=identification),
        decision_relevance=decision,
    )
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results_doc, indent=1, allow_nan=False) + "\n"
    with (out / "results.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    log(f"wrote {out / 'results.json'} ({len(text) / 1e6:.1f} MB); runtime {runtime / 60:.1f} min")
    print(json.dumps(dict(label=LABEL, mc_error=mc_error, answers=answers, runtime_minutes=runtime / 60,
                          hardest=hardest), indent=1))


if __name__ == "__main__":
    main()
