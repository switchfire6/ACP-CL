"""Step D0b: the bounded redesign round declared in reports/commons_d0/summary.md.

Targets (declared by the lead before any search; not changed here):
  relevance (pair-averaged cross-context policy regret): B >= .05 and C >= .05,
  no B or C pair below .02, and A <= 3 x min(B, C);
  separability: B high (<= 5 episodes to .95), C low (>= 40 episodes).
Allowed levers: transfer sizes / action set, which channels B and C touch,
seasonal amplitude (imbalance, A flip amplitude), entity-3 sign, probe regime.
The physics engine, references and verification code are D0's (extended with
options whose defaults reproduce D0 bit for bit).

Stages (outputs in reports/commons_d0b/, raw arrays in runs/commons_d0b/):
  screen       coarse relevance grid (cheap: value tables only)
  refine       finer relevance grid around the best coarse configurations
  separability pair pools (likelihood + identification) for the finalists; choose the world
  final        fresh-draw calibration of all 12 factor pairs at the chosen world
  verify       D0 verification suite on the chosen world (+ D0 edge sets)
  voi | schedule | stream (easiness + one-step lookahead) | timing
  all          everything in order

Usage: set PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src, then
  .venv/Scripts/python.exe -B scripts/commons_d0b.py <stage> [--workers 6]
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import commons_d0 as D0  # noqa: E402
from acp_cl.commons import calibration as C  # noqa: E402
from acp_cl.commons import references as R  # noqa: E402
from acp_cl.commons import verification as V  # noqa: E402
from acp_cl.commons.world import (N_CONTEXTS, SURVIVED, WorldParams, action_names,  # noqa: E402
                                  context_name, n_actions, observe, sample_episodes, seed_for,
                                  simulate, world_spec)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "commons_d0b"
RUNS = ROOT / "runs" / "commons_d0b"

TARGETS = {"rel_B_min": 0.05, "rel_C_min": 0.05, "pair_min": 0.02, "A_over_minBC_max": 3.0,
           "sep_B_max_episodes": 5.0, "sep_C_min_episodes": 40.0}
SCREEN_N, SCREEN_M = 384, 1024
POOL_N, POOL_M = 2048, 4096
LEVEL = "passive_steady"   # C must be invisible to probes to be low-separability (D0 result)
O_M = 2 ** 14
CHUNK = 500
STREAM_LENGTH = 20000
PAIRS = {f: C.factor_pairs(f) for f in "ABC"}

# Fixed physics shared by every candidate (D0 values).
BASE = dict(base_signs=[1.0, -1.0, -0.5], probe_start="steady", flip_amplitude=1.0, b_gap=0.6, c_delta=6.0)


def make_params(cfg):
    cfg = dict(cfg)
    for key in ("base_signs", "amounts", "c_channels"):
        if key in cfg:
            cfg[key] = tuple(cfg[key])
    return WorldParams(**cfg)


def cfg_key(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def with_modes(cfg):
    """Complete a candidate: mode-dependent channel settings (declared)."""
    cfg = dict(cfg)
    if cfg.get("c_mode", "ch23") in ("same", "route"):
        cfg["c_channels"] = [1, 2]
        cfg["d31"] = 0.0        # gated 3<->1 is fast (0) exactly like the gated 2<->3
    return cfg


# ---------------------------------------------------------------- relevance screen

def relevance_from_tables(v1, v2):
    """Pair relevance for all 12 factor pairs from 8-context tables (n, 8, A)."""
    n = v1.shape[0]
    rows = np.arange(n)
    best = v1.argmax(2)  # (n, 8)
    out = {}
    for f, pairs in PAIRS.items():
        vals = []
        for i, j in pairs:
            ai, aj = best[:, i], best[:, j]
            r_ij = (v2[rows, i, ai] - v2[rows, i, aj]).mean()
            r_ji = (v2[rows, j, aj] - v2[rows, j, ai]).mean()
            vals.append(0.5 * (r_ij + r_ji))
        out[f] = vals
    return out


def screen_job(cfg, n, m, tag):
    path = RUNS / tag / f"{cfg_key(cfg)}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        body = json.dumps(data["result"], sort_keys=True)
        if hashlib.sha256(body.encode()).hexdigest() == data["sha256"]:
            return data["result"]
    params = make_params(cfg)
    t0 = time.time()
    ep = sample_episodes(n, seed_for("d0b_screen_episodes"))
    v1 = R.value_tables(params, ep, list(range(N_CONTEXTS)), m, 11, "d0b_select").value
    v2 = R.value_tables(params, ep, list(range(N_CONTEXTS)), m, 11, "d0b_score").value
    rel = relevance_from_tables(v1, v2)
    rows = np.arange(n)
    result = {"cfg": cfg, "key": cfg_key(cfg),
              "rel_A": float(np.mean(rel["A"])), "rel_B": float(np.mean(rel["B"])), "rel_C": float(np.mean(rel["C"])),
              "pairs_A": rel["A"], "pairs_B": rel["B"], "pairs_C": rel["C"],
              "o_value": float(v2[rows[:, None], np.arange(8)[None], v1.argmax(2)].mean()),
              "none_value": float(v2[:, :, 0].mean()), "n_actions": int(v1.shape[2]),
              "seconds": time.time() - t0}
    result.update(score(result))
    D0.write_json(path, {"result": result, "sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()})
    return result


def score(r):
    mn = min(r["rel_B"], r["rel_C"])
    pair_min = min(min(r["pairs_B"]), min(r["pairs_C"]))
    ratio = r["rel_A"] / mn if mn > 0 else float("inf")
    margins = {"B": r["rel_B"] / TARGETS["rel_B_min"], "C": r["rel_C"] / TARGETS["rel_C_min"],
               "pair": pair_min / TARGETS["pair_min"], "A": TARGETS["A_over_minBC_max"] / ratio if ratio > 0 else 0.0}
    return {"min_BC": mn, "pair_min_BC": pair_min, "A_over_minBC": ratio,
            "relevance_targets_met": bool(all(v >= 1 for v in margins.values())),
            "relevance_margin": float(min(margins.values())), "margins": margins}


def coarse_grid():
    grid = []
    for amounts, imb, e3, bmode, cmode, flip in itertools.product(
            ([3.0], [6.0], [3.0, 6.0]), (0.8, 1.2), (-0.5, -1.0, 0.5), ("ch12", "both", "swap"),
            ("ch23", "same", "route"), (0.5, 1.0)):
        cfg = dict(BASE, amounts=amounts, imbalance=imb, base_signs=[1.0, -1.0, e3], b_mode=bmode,
                   c_mode=cmode, flip_amplitude=flip, b_gap=0.7, c_delta=8.0)
        grid.append(with_modes(cfg))
    return grid


def run_screen(grid, tag, workers):
    jobs = [(cfg_key(c), (screen_job, (c, SCREEN_N, SCREEN_M, tag))) for c in grid]
    res = D0.run_jobs(D0._call, jobs, workers)
    return [res[k] for k, _ in jobs]


def frontier(rows):
    """Pareto set over (rel_B, rel_C, pair_min_BC, -A_over_minBC)."""
    keys = lambda r: (r["rel_B"], r["rel_C"], r["pair_min_BC"], -r["A_over_minBC"])  # noqa: E731
    out = []
    for r in rows:
        kr = keys(r)
        dominated = any(all(a >= b for a, b in zip(keys(o), kr)) and keys(o) != kr for o in rows)
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: -r["relevance_margin"])


def flat(r):
    c = r["cfg"]
    return {"key": r["key"], "amounts": c.get("amounts"), "e31": c.get("e31", 0.8), "imbalance": c.get("imbalance"), "e3_sign": c["base_signs"][2],
            "b_mode": c.get("b_mode"), "c_mode": c.get("c_mode"), "flip": c.get("flip_amplitude"), "b_gap": c.get("b_gap"),
            "c_delta": c.get("c_delta"), "rel_A": round(r["rel_A"], 4), "rel_B": round(r["rel_B"], 4), "rel_C": round(r["rel_C"], 4),
            "pair_min_BC": round(r["pair_min_BC"], 4), "A_over_minBC": round(r["A_over_minBC"], 2),
            "relevance_margin": round(r["relevance_margin"], 3), "targets_met": r["relevance_targets_met"],
            "o_value": round(r["o_value"], 3), "none_value": round(r["none_value"], 3)}


def cmd_screen(args):
    rows = run_screen(coarse_grid(), "screen", args.workers)
    D0.write_csv(REPORT / "screen.csv", [flat(r) for r in rows])
    fr = frontier(rows)
    D0.write_json(REPORT / "screen.json", {"settings": {"episodes": SCREEN_N, "mc_draws": SCREEN_M, "fixed": BASE,
                                                        "coarse_axes": "amounts {3},{6},{3,6}; imbalance .8/1.2; entity-3 sign -.5/-1/+.5; "
                                                                       "b_mode ch12/both/swap; c_mode ch23/same/route; flip .5/1; b_gap .7; c_delta 8"},
                                           "targets": TARGETS, "n_configs": len(rows),
                                           "n_meeting_relevance_targets": sum(r["relevance_targets_met"] for r in rows),
                                           "frontier": [flat(r) for r in fr]})
    for r in sorted(rows, key=lambda r: -r["relevance_margin"])[:15]:
        print(flat(r), flush=True)


def refine_grid(best):
    """Declared refinement around each seed: b_gap {.5,.7,.8} (e31 raised to .9 when b_gap = .8 so the
    swapped route stays valid), c_delta {6, 12 (never arrives)}, A flip {.3,.5,.75}, imbalance {seed, seed+.2},
    amounts {seed's, [3, 8]}."""
    grid = []
    for r in best:
        c = r["cfg"]
        for b_gap, c_delta, flip, imb, amounts in itertools.product(
                (0.5, 0.7, 0.8), (6.0, 12.0), (0.3, 0.5, 0.75), (c["imbalance"], c["imbalance"] + 0.2),
                (tuple(c["amounts"]), (3.0, 8.0))):
            cfg = dict(c, b_gap=b_gap, c_delta=c_delta, flip_amplitude=flip, imbalance=round(imb, 2), amounts=list(amounts))
            if b_gap >= 0.8:
                cfg["e31"] = 0.9
            grid.append(with_modes(cfg))
    uniq = {cfg_key(g): g for g in grid}
    return list(uniq.values())


def margin_no_a(r):
    return min(r["rel_B"] / TARGETS["rel_B_min"], r["rel_C"] / TARGETS["rel_C_min"], r["pair_min_BC"] / TARGETS["pair_min"])


def cmd_refine(args):
    coarse = [json.loads(p.read_text(encoding="utf-8"))["result"] for p in (RUNS / "screen").glob("*.json")]
    # Declared: 8 seeds = best coarse configurations by min(B/.05, C/.05, pair_min/.02) (A is left to the flip
    # amplitude), distinct in (amounts, b_mode, c_mode, entity-3 sign, imbalance).
    seen, best = set(), []
    for r in sorted(coarse, key=lambda r: -margin_no_a(r)):
        sig = (tuple(r["cfg"]["amounts"]), r["cfg"]["b_mode"], r["cfg"]["c_mode"], r["cfg"]["base_signs"][2], r["cfg"]["imbalance"])
        if sig in seen:
            continue
        seen.add(sig)
        best.append(r)
        if len(best) == 8:
            break
    rows = run_screen(refine_grid(best), "refine", args.workers)
    allrows = coarse + rows
    D0.write_csv(REPORT / "refine.csv", [flat(r) for r in rows])
    fr = frontier(allrows)
    D0.write_json(REPORT / "refine.json", {"seeds_from_coarse": [flat(r) for r in best], "n_configs": len(rows),
                                           "n_meeting_relevance_targets": sum(r["relevance_targets_met"] for r in rows),
                                           "max_min_BC": max(r["min_BC"] for r in allrows),
                                           "frontier_all": [flat(r) for r in fr]})
    for r in sorted(rows, key=lambda r: -r["relevance_margin"])[:15]:
        print(flat(r), flush=True)


# ---------------------------------------------------------------- separability for finalists

def pool_path(tag, key, i, j):
    return RUNS / "pools" / tag / key / f"{i}{j}.npz"


def pool_job(tag, cfg, i, j, n, m, seed_key):
    key = cfg_key(cfg)
    path = pool_path(tag, key, i, j)
    loaded = D0.load_verified(path)
    if loaded is not None:
        return str(path)
    params = make_params(cfg)
    t0 = time.time()
    pool = C.pair_pool(params, i, j, n, seed_for(seed_key, i, j), m, m)
    arrays = {"v_select": pool.v_select, "v_score": pool.v_score, "se_score": pool.se_score}
    for level in C.LEVELS:
        arrays[f"llr_{level}"] = pool.llr[level]
    D0.save_arrays(path, meta={"cfg": cfg, "pair": [i, j], "n": n, "m": m, "stats": pool.stats,
                               "seconds": time.time() - t0}, **arrays)
    return str(path)


def pool_metrics(tag, cfg, factors="ABC", levels=C.LEVELS):
    key = cfg_key(cfg)
    rows = []
    for f in factors:
        for i, j in PAIRS[f]:
            arrays, info = D0.load_verified(pool_path(tag, key, i, j))
            pool = D0.pool_from_arrays(arrays, i, j)
            for level in levels:
                mt = C.pool_metrics(pool, level, n_boot=200, seed=seed_for("d0b_boot", key, i, j, level),
                                    id_sequences=D0.ID_SEQUENCES, id_cap=D0.ID_CAP)
                st = info["meta"]["stats"]
                rows.append({"key": key, "factor": f, "pair": f"{context_name(i)}-{context_name(j)}", "i": i, "j": j,
                             "level": level, **mt,
                             "pattern_entries_floored_frac": st.get("floored", 0) / max(st.get("entries", 1), 1)})
    return rows


def grade_rows(rows, level=LEVEL):
    out = {}
    for f in "ABC":
        rs = [r for r in rows if r["factor"] == f and r["level"] == level]
        if not rs:
            continue
        out[f] = {"id_mean": float(np.mean([r["id_mean"] for r in rs])), "id_p90_max": float(max(r["id_p90"] for r in rs)),
                  "id_censored_max": float(max(max(r["id_censored_truth_i"], r["id_censored_truth_j"]) for r in rs)),
                  "chernoff": float(np.mean([r["chernoff_behaviour"] for r in rs])),
                  "sym_kl": float(np.mean([r["sym_kl_behaviour"] for r in rs])),
                  "relevance": float(np.mean([r["relevance"] for r in rs])),
                  "relevance_se": float(np.sqrt(np.sum([r["relevance_se"] ** 2 for r in rs])) / len(rs)),
                  "relevance_pairs": [r["relevance"] for r in rs], "id_pairs": [r["id_mean"] for r in rs]}
    return out


def full_targets(g):
    t = TARGETS
    mn = min(g["B"]["relevance"], g["C"]["relevance"])
    checks = {"rel_B": g["B"]["relevance"] >= t["rel_B_min"], "rel_C": g["C"]["relevance"] >= t["rel_C_min"],
              "pair_min": min(g["B"]["relevance_pairs"] + g["C"]["relevance_pairs"]) >= t["pair_min"],
              "sep_B": g["B"]["id_mean"] <= t["sep_B_max_episodes"], "sep_C": g["C"]["id_mean"] >= t["sep_C_min_episodes"]}
    if "A" in g:
        checks["A_ratio"] = g["A"]["relevance"] <= t["A_over_minBC_max"] * mn
    margins = {"rel_B": g["B"]["relevance"] / t["rel_B_min"], "rel_C": g["C"]["relevance"] / t["rel_C_min"],
               "pair_min": min(g["B"]["relevance_pairs"] + g["C"]["relevance_pairs"]) / t["pair_min"],
               "sep_B": t["sep_B_max_episodes"] / max(g["B"]["id_mean"], 1e-9), "sep_C": g["C"]["id_mean"] / t["sep_C_min_episodes"]}
    if "A" in g:
        margins["A_ratio"] = t["A_over_minBC_max"] * mn / max(g["A"]["relevance"], 1e-9)
    return checks, margins


def all_screen_rows():
    rows = []
    for tag in ("screen", "refine"):
        for p in (RUNS / tag).glob("*.json"):
            rows.append(json.loads(p.read_text(encoding="utf-8"))["result"])
    return rows


def cmd_separability(args):
    rows = all_screen_rows()
    # Declared finalist rule: configurations meeting all relevance targets in the screen, best 6 by
    # relevance margin (distinct keys); if fewer than 6 meet them, fill with the best remaining margins.
    ranked = sorted(rows, key=lambda r: (not r["relevance_targets_met"], -r["relevance_margin"]))
    finalists = ranked[:6]
    jobs = []
    for r in finalists:
        for f in "ABC":
            for i, j in PAIRS[f]:
                jobs.append(((r["key"], i, j), (pool_job, ("finalists", r["cfg"], i, j, POOL_N, POOL_M, "d0b_pool"))))
    D0.run_jobs(D0._call, jobs, args.workers)
    results = []
    allrows = []
    for r in finalists:
        prow = pool_metrics("finalists", r["cfg"])
        allrows += prow
        g = grade_rows(prow)
        checks, margins = full_targets(g)
        results.append({"screen": flat(r), "cfg": r["cfg"], "grades_passive_steady": g,
                        "grades_active": grade_rows(prow, "active"), "grades_passive_episode": grade_rows(prow, "passive_episode"),
                        "checks": checks, "margins": margins, "all_targets_met": bool(all(checks.values())),
                        "min_margin": float(min(margins.values()))})
    D0.write_csv(REPORT / "separability_pairs.csv", [{k: v for k, v in r.items() if k != "behaviour_action_freq"} for r in allrows])
    met = [x for x in results if x["all_targets_met"]]
    # Declared choice: among finalists meeting every target, the largest minimum margin;
    # otherwise the finalist with the largest minimum margin (best achievable) and stop.
    chosen = max(met or results, key=lambda x: x["min_margin"])
    D0.write_json(REPORT / "separability.json", {"level": LEVEL, "targets": TARGETS, "finalists": results,
                                                 "chosen_key": cfg_key(chosen["cfg"]), "chosen_cfg": chosen["cfg"],
                                                 "targets_reached": bool(met),
                                                 "choice_rule": "all targets met -> max of min margin; else max of min margin (best achievable)"})
    for x in results:
        print(x["screen"]["key"], x["all_targets_met"], round(x["min_margin"], 3),
              {f: (round(x["grades_passive_steady"][f]["id_mean"], 1), round(x["grades_passive_steady"][f]["relevance"], 4)) for f in "ABC"},
              flush=True)


def chosen_cfg():
    return json.loads((REPORT / "separability.json").read_text(encoding="utf-8"))["chosen_cfg"]


def cmd_cfrontier(args):
    """C's separability-relevance trade-off at the chosen structure: c_delta {2, 3, 4} plus the chosen value."""
    base = chosen_cfg()
    cfgs = [dict(base, c_delta=d) for d in (2.0, 3.0, 4.0)]
    jobs = [((cfg_key(c), i, j), (pool_job, ("cfrontier", c, i, j, POOL_N, POOL_M, "d0b_cfrontier")))
            for c in cfgs for i, j in PAIRS["C"]]
    D0.run_jobs(D0._call, jobs, args.workers)
    out = []
    for c in cfgs:
        g = grade_rows(pool_metrics("cfrontier", c, factors="C"))
        out.append({"c_delta": c["c_delta"], **{lv: g2["C"] for lv, g2 in
                                                ((lv, grade_rows(pool_metrics("cfrontier", c, factors="C", levels=(lv,)), lv))
                                                 for lv in (LEVEL, "active"))}})
    sep = json.loads((REPORT / "separability.json").read_text(encoding="utf-8"))
    ch = [f for f in sep["finalists"] if cfg_key(f["cfg"]) == cfg_key(base)][0]
    out.append({"c_delta": base["c_delta"], LEVEL: ch["grades_passive_steady"]["C"], "active": ch["grades_active"]["C"]})
    D0.write_json(REPORT / "c_frontier.json", {"cfg": base, "rows": sorted(out, key=lambda r: r["c_delta"])})
    for r in sorted(out, key=lambda r: r["c_delta"]):
        print(r["c_delta"], round(r[LEVEL]["id_mean"], 1), round(r[LEVEL]["relevance"], 4), [round(x, 4) for x in r[LEVEL]["relevance_pairs"]], flush=True)


# ---------------------------------------------------------------- final calibration on fresh draws

def cmd_final(args):
    cfg = chosen_cfg()
    jobs = [((f, i, j), (pool_job, ("final", cfg, i, j, POOL_N, POOL_M, "d0b_final"))) for f in "ABC" for i, j in PAIRS[f]]
    D0.run_jobs(D0._call, jobs, args.workers)
    rows = pool_metrics("final", cfg)
    D0.write_csv(REPORT / "calibration_final_pairs.csv", [{k: v for k, v in r.items() if k != "behaviour_action_freq"} for r in rows])
    grades = {lv: grade_rows(rows, lv) for lv in C.LEVELS}
    checks, margins = full_targets(grades[LEVEL])
    params = make_params(cfg)
    spec = world_spec(params)
    spec["actions"] = {i: n for i, n in enumerate(action_names(params))}
    D0.write_json(REPORT / "world_spec.json", spec)
    D0.write_json(REPORT / "calibration.json", {"cfg": cfg, "level": LEVEL, "grades": grades, "targets": TARGETS,
                                                "checks": checks, "margins": margins, "all_targets_met": bool(all(checks.values())),
                                                "settings": {"pool_episodes": POOL_N, "mc_draws": POOL_M,
                                                             "identification_sequences": D0.ID_SEQUENCES, "identification_cap": D0.ID_CAP,
                                                             "draws": "fresh (seed key d0b_final), independent of the finalist pools"}})
    print(json.dumps({"checks": checks, "margins": margins}, indent=1), flush=True)
    for lv in C.LEVELS:
        print(lv, {f: (round(g["id_mean"], 1), round(g["relevance"], 4), [round(x, 4) for x in g["relevance_pairs"]]) for f, g in grades[lv].items()}, flush=True)


# ---------------------------------------------------------------- verify

def smc_case_job(name, e, truth, action, ctx, entity, cfg):
    params = make_params(cfg)
    ep, _ = V.find_edge_cases(params)
    r = V.likelihood_case_smc(params, ep, e, truth, action, ctx, entity)
    r["case"] = name
    return r


def cmd_verify(args):
    t0 = time.time()
    cfg = chosen_cfg()
    params = make_params(cfg)
    out = {"cfg": cfg}
    variants = [params, params.with_dials(probe_start="episode"), params.with_dials(c_delta=2.5, d12=0.4),
                params.with_dials(c_delta=14.0, flip_amplitude=0.3)]
    out["simulator_vs_slow_reference"] = V.check_simulator(n=1500, param_sets=V.edge_params() + variants)
    print("simulator", out["simulator_vs_slow_reference"]["pass"], flush=True)
    out["gaussian_sensor_marginal"] = V.gaussian_marginal_check(params)
    out["o_value_tables_crn"] = V.check_value_tables(params)
    print("value tables", out["o_value_tables_crn"]["pass"], flush=True)
    out["filter_vs_bruteforce"] = V.check_filter()
    out["negative_supply_audit"] = V.negative_supply_audit(params)
    ep, cases = V.find_edge_cases(params)
    jobs = []
    for name, (e, truth, action) in cases.items():
        for ctx in (truth, V.alternative_context(truth, action, params)):
            for entity in range(3):
                jobs.append(((name, ctx, entity), (smc_case_job, (name, e, truth, action, ctx, entity, cfg))))
    lik = D0.run_jobs(D0._call, jobs, args.workers)
    lik_rows = [lik[k] for k, _ in jobs]
    joint = {}
    names = action_names(params)
    covered = {}
    for name, (e, truth, action) in cases.items():
        for ctx in (truth, V.alternative_context(truth, action, params)):
            joint[f"{name}_ctx{ctx}"] = V.joint_factorisation_case(params, ep, e, truth, action, ctx)
        one = ep.take(np.array([e]))
        sup = one.supplies(params, np.array([truth]))
        r = simulate(params, np.array([truth]), one, np.array([action]), sup)
        covered[name] = {"episode": e, "truth": context_name(truth), "action": names[action],
                         "reserves": one.reserves[0].round(3).tolist(), "fail_steps": r.fail_step[0].tolist(),
                         "overflow": r.overflow[0].round(3).tolist(), "sent": float(r.sent[0]),
                         "first_arrival_step_index": int(r.arrival_step_first[0]), "min_supply": float(sup.min())}
    out["likelihood_bruteforce"] = {"method": "as D0 (sequential prior-sampling estimator vs conjugate estimator)", "cases": covered,
                                    "per_entity": lik_rows, "joint_full_simulator_vs_product_and_code": joint,
                                    "pass": bool(all(r["agree"] for r in lik_rows) and all(j["agree"] for j in joint.values())),
                                    "max_abs_z": float(max(abs(r.get("z", 0.0)) for r in lik_rows))}
    print("likelihood", out["likelihood_bruteforce"]["pass"], flush=True)
    p1 = C.pair_pool(params, 0, 1, 48, 5, 512, 512)
    p2 = C.pair_pool(params, 0, 1, 48, 5, 512, 512)
    d1 = V.digest_arrays(p1.v_select, p1.v_score, *[p1.llr[k] for k in C.LEVELS])
    d2 = V.digest_arrays(p2.v_select, p2.v_score, *[p2.llr[k] for k in C.LEVELS])
    c1 = compute_chunk(cfg, 1, 0, 8, None)
    c2 = compute_chunk(cfg, 1, 0, 8, None)
    out["determinism"] = {"pair_pool": [d1, d2], "stream_chunk": [c1, c2], "pass": bool(d1 == d2 and c1 == c2)}
    # D0 defaults still reproduce D0 (a stored D0 stream chunk recomputed bit for bit)
    d0cal = json.loads((ROOT / "reports/commons_d0/calibration.json").read_text(encoding="utf-8"))
    d0dials = {k: d0cal["proposed_dials"][k] for k in D0.ANCHOR}
    info = json.loads((ROOT / "runs/commons_d0/stream/seed_1/chunk_01500.sha256.json").read_text(encoding="utf-8"))
    out["d0_reproduction"] = {"pass": bool(D0.compute_chunk(1, d0dials, "steady", 1500, 2000) == info["digest"])}
    checks = ["simulator_vs_slow_reference", "gaussian_sensor_marginal", "o_value_tables_crn", "filter_vs_bruteforce",
              "negative_supply_audit", "likelihood_bruteforce", "determinism", "d0_reproduction"]
    out["all_pass"] = bool(all(out[k]["pass"] for k in checks))
    out["seconds"] = time.time() - t0
    D0.write_json(REPORT / "verification.json", out)
    print("ALL PASS" if out["all_pass"] else "FAILURES: " + ", ".join(k for k in checks if not out[k]["pass"]), flush=True)


# ---------------------------------------------------------------- VOI

def voi_job(cfg, truth, n):
    path = RUNS / "voi" / cfg_key(cfg) / f"truth_{truth}.npz"
    if D0.load_verified(path) is not None:
        return str(path)
    params = make_params(cfg)
    na = n_actions(params)
    episodes = sample_episodes(n, seed_for("d0b_voi_episodes"))
    truth_arr = np.full(n, truth)
    supplies = episodes.supplies(params, truth_arr)
    fs = C.fail_steps_all_actions(params, truth, episodes, supplies)
    y_supply, _ = observe(params, truth_arr, episodes, supplies, probes_on=False)
    stats = {}
    pattern = np.empty((n, na, N_CONTEXTS))
    for e in range(n):
        pattern[e] = R.pattern_loglik(params, episodes.reserves[e], int(episodes.cue[e, 0]), y_supply[e],
                                      {a: fs[e, a] for a in range(na)}, list(range(N_CONTEXTS)), list(range(na)),
                                      seed_for("d0b_voi_pattern", truth, e), R.PatternSettings(), stats)
    sup, _ = R.supply_loglik(params, y_supply)
    arrays = {"pattern": pattern, "supply": sup[:, [c >> 2 & 1 for c in range(N_CONTEXTS)]]}
    for level in C.LEVELS:
        lp, on = C.level_params(params, level)
        _, yp = observe(lp, truth_arr, episodes, supplies, probes_on=on)
        arrays[f"probe_{level}"] = R.probe_loglik(lp, yp, episodes.cue[:, 0], on)
    D0.save_arrays(path, meta={"truth": truth, "cfg": cfg, "stats": stats}, **arrays)
    return str(path)


def cmd_voi(args):
    cfg = chosen_cfg()
    params = make_params(cfg)
    names = action_names(params)
    D0.run_jobs(D0._call, [(t, (voi_job, (cfg, t, D0.VOI_N))) for t in range(N_CONTEXTS)], args.workers)
    data = [D0.load_verified(RUNS / "voi" / cfg_key(cfg) / f"truth_{t}.npz")[0] for t in range(N_CONTEXTS)]
    out = {"cfg": cfg, "episodes_per_context": D0.VOI_N}
    for level in C.LEVELS:
        lv = {}
        probe_only = [d[f"probe_{level}"] for d in data]
        sensors = [d["supply"] + d[f"probe_{level}"] for d in data]
        lv["mi_probe_only"] = C.mi_from_logliks(probe_only)
        lv["mi_sensors_only"] = C.mi_from_logliks(sensors)
        per_action = []
        per_factor = {f: [] for f in "ABC"}
        for a in range(len(names)):
            full = [s + d["pattern"][:, a] for s, d in zip(sensors, data)]
            per_action.append(C.mi_from_logliks(full))
            for f in "ABC":
                per_factor[f].append(C.binary_factor_mi(full, f))
        vals = np.array([v[0] for v in per_action])
        lv["mi_full_by_action"] = dict(zip(names, per_action))
        lv["voi_max_minus_min_nats"] = float(vals.max() - vals.min())
        for f in "ABC":
            fv = np.array([v[0] for v in per_factor[f]])
            lv[f"binary_mi_{f}_by_action"] = dict(zip(names, per_factor[f]))
            lv[f"binary_mi_{f}_sensors_only"] = C.binary_factor_mi(sensors, f)
            lv[f"binary_voi_{f}_max_minus_min"] = float(fv.max() - fv.min())
        out[level] = lv
    D0.write_json(REPORT / "voi.json", out)
    print(json.dumps({lv: {k: out[lv][k] for k in ("mi_probe_only", "mi_sensors_only", "voi_max_minus_min_nats",
                                                     "binary_voi_B_max_minus_min", "binary_voi_C_max_minus_min")} for lv in C.LEVELS}, indent=1))


# ---------------------------------------------------------------- schedule

def cmd_schedule(args):
    cal = json.loads((REPORT / "calibration.json").read_text(encoding="utf-8"))
    id_time = cal["grades"][LEVEL]["C"]["id_mean"]
    spec = C.ScheduleSpec()
    out = {"spec": spec.__dict__, "training": [context_name(c) for c in C.TRAINING], "held_out": [context_name(c) for c in C.HELD_OUT],
           "minority": context_name(C.MINORITY), "low_separability_contrast": {"factor": "C", "level": LEVEL, "id_mean_episodes": id_time,
                                                                             "id_p90_max": cal["grades"][LEVEL]["C"]["id_p90_max"]},
           "note": "the schedule generator is unchanged from D0 (it does not depend on the world)", "seeds": {}}
    for seed in (1, 2, 3):
        ctx, order, info = C.generate_schedule(seed, spec)
        st = C.schedule_statistics(ctx, order, info, spec, id_time)
        lengths = np.array([r for _, r in order])
        st["runs_between_0.5x_and_2x_low_sep_id_time"] = float(((lengths >= 0.5 * id_time) & (lengths <= 2 * id_time)).mean())
        # C-only switches (the low-separability contrast in the stream)
        c_only = [(a, b, r2) for (a, _), (b, r2) in zip(order, order[1:]) if (a ^ b) == 1]
        st["c_only_switches"] = len(c_only)
        st["c_only_switch_run_lengths"] = [r2 for _, _, r2 in c_only]
        st["sha256"] = hashlib.sha256(ctx.tobytes()).hexdigest()
        out["seeds"][str(seed)] = st
    D0.write_json(REPORT / "schedule.json", out)
    for seed, s in out["seeds"].items():
        print(seed, s["runs"], s["run_length"], s["runs_shorter_than_low_sep_id_time"], s["runs_shorter_than_2x_low_sep_id_time"],
              s["c_only_switches"], s["c_only_switch_run_lengths"], flush=True)


# ---------------------------------------------------------------- stream

def stream_data(seed):
    contexts, order, info = C.generate_schedule(seed)
    return contexts, sample_episodes(len(contexts), seed_for("d0b_stream_episodes", seed))


def chunk_job(cfg, seed, start, stop):
    path = RUNS / "stream" / cfg_key(cfg) / f"seed_{seed}" / f"chunk_{start:05d}.npz"
    loaded = D0.load_verified(path)
    if loaded is not None:
        return str(path), loaded[1]["digest"], True
    return str(path), compute_chunk(cfg, seed, start, stop, path), False


def compute_chunk(cfg, seed, start, stop, path=None):
    params = make_params(cfg)
    na = n_actions(params)
    m = O_M if path is not None else 512
    contexts, episodes = stream_data(seed)
    idx = np.arange(start, stop)
    ep = episodes.take(idx)
    ctx = contexts[idx].astype(np.int64)
    n = len(idx)
    t0 = time.time()
    v1 = np.empty((n, N_CONTEXTS, na))
    v1se = np.empty_like(v1)
    v2 = np.empty((n, na))
    v2se = np.empty_like(v2)
    for k, e in enumerate(idx):
        z = R.draws(seed_for(seed, "d0b_O_select", int(e)), m)
        v1[k], v1se[k] = R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), list(range(N_CONTEXTS)), z)
        z = R.draws(seed_for(seed, "d0b_O_score", int(e)), m)
        val, se = R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), [int(ctx[k])], z)
        v2[k], v2se[k] = val[0], se[0]
    t_values = time.time() - t0
    supplies = ep.supplies(params, ctx)
    fs = np.empty((n, na, 3), dtype=np.int8)
    for a in range(na):
        fs[:, a] = simulate(params, ctx, ep, np.full(n, a), supplies).fail_step
    y_supply, y_probe = observe(params, ctx, ep, supplies, probes_on=True)
    t1 = time.time()
    stats = {}
    sensor, pattern = R.episode_loglik(params, ep, y_supply, y_probe, fs, list(range(na)),
                                       seed_for(seed, "d0b_I_pattern", start), R.PatternSettings(), True, stats)
    arrays = {"v1": v1, "v1se": v1se, "v2": v2, "v2se": v2se, "fail_steps": fs, "y_supply": y_supply, "y_probe": y_probe,
              "sensor_ll": sensor, "pattern_ll": pattern, "contexts": ctx}
    if path is None:
        return D0.array_digest(arrays)
    return D0.save_arrays(path, meta={"cfg": cfg, "seed": seed, "start": start, "stop": stop, "pattern_stats": stats,
                                      "seconds_values": t_values, "seconds_likelihood": time.time() - t1}, **arrays)


def load_stream(cfg, seed):
    parts, metas = {}, []
    for start in range(0, STREAM_LENGTH, CHUNK):
        arrays, info = D0.load_verified(RUNS / "stream" / cfg_key(cfg) / f"seed_{seed}" / f"chunk_{start:05d}.npz")
        metas.append(info["meta"])
        for k, v in arrays.items():
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}, metas


class Lookahead:
    """One-step lookahead on I's belief (declared):

    Q(a) = sum_c w_c V_t[c, a] + F(a),  F(a) = E_{c ~ w, x ~ p(x | c, a)} [ U(w'(x)) ],
    U(w) = mean over a fixed pool of 256 next-episode situations of max_a' sum_c w_c V[e, c, a'],
    w'(x) = the filter's predictive belief for the next episode after updating on x.

    Exact pruning: by Jensen and the martingale property of the posterior, every F(a) lies in
    [U(wbar), sum_c wbar_c U(e_c)] where wbar is the two-step predictive belief without new data, so
    only actions with myopic(a) + EVPI(wbar) >= max myopic can win. If one action survives, it is the
    myopic action (no approximation). Otherwise F is estimated for the surviving actions with K = 8
    hypothetical episodes per context with w_c >= .005 (the same noise for every action), full
    likelihoods (exact sensors + pattern MC with k=512, refine 4096).
    """

    def __init__(self, params, pool_values, seed, k=8, min_weight=0.005):
        self.params, self.pool, self.k, self.min_weight = params, pool_values, k, min_weight
        self.rng_seed = seed
        self.settings = R.PatternSettings(k=512, k_refine=4096, min_count=8)
        self.u_perfect = pool_values.max(2).mean(0)  # (8,)
        self.evaluated = 0
        self.pruned_to_one = 0
        self.evpi_sum = 0.0

    def utility(self, w):
        return float((np.einsum("c,eca->ea", w, self.pool)).max(1).mean())

    def _clone(self, filt):
        trial = copy.copy(filt)
        trial.state = None if filt.state is None else filt.state.copy()
        return trial

    def choose(self, t, w, filt, v1_t, reserves, cue):
        myopic = w @ v1_t
        best = int(np.argmax(myopic))
        trial = self._clone(filt)
        trial.update(np.zeros(N_CONTEXTS))
        wbar, _ = trial.weights()
        evpi = float(wbar @ self.u_perfect - self.utility(wbar))
        self.evpi_sum += evpi
        alive = [a for a in range(len(myopic)) if myopic[a] + evpi >= myopic[best]]
        if len(alive) <= 1:
            self.pruned_to_one += 1
            return best, 0.0
        self.evaluated += 1
        params = self.params
        cands = [c for c in range(N_CONTEXTS) if w[c] >= self.min_weight]
        wc = np.array([w[c] for c in cands])
        wc = wc / wc.sum()
        ep = sample_episodes(self.k, seed_for(self.rng_seed, "lookahead", t))
        ep.reserves[:] = reserves
        ep.cue[:] = cue
        future = np.zeros(len(myopic))
        for ci, c in enumerate(cands):
            ctx = np.full(self.k, c)
            sup = ep.supplies(params, ctx)
            ys, yp = observe(params, ctx, ep, sup, probes_on=True)
            sensor_sup, _ = R.supply_loglik(params, ys)
            sensor = sensor_sup[:, [x >> 2 & 1 for x in range(N_CONTEXTS)]] + R.probe_loglik(params, yp, ep.cue[:, 0], True)
            for a in alive:
                fs = simulate(params, ctx, ep, np.full(self.k, a), sup).fail_step
                u = 0.0
                for j in range(self.k):
                    pl = R.pattern_loglik(params, ep.reserves[j], int(ep.cue[j, 0]), ys[j], {a: fs[j]},
                                          list(range(N_CONTEXTS)), [a], seed_for(self.rng_seed, "la_pl", t, c, j),
                                          self.settings)[0]
                    trial = self._clone(filt)
                    trial.update(sensor[j] + pl)
                    u += self.utility(trial.weights()[0])
                future[a] += wc[ci] * u / self.k
        q = np.full(len(myopic), -np.inf)
        q[alive] = myopic[alive] + future[alive]
        choice = int(np.argmax(q))
        return choice, float(q[choice] - q[best])


def lookahead_pool(params, n=256, m=4096):
    ep = sample_episodes(n, seed_for("d0b_lookahead_pool"))
    return R.value_tables(params, ep, list(range(N_CONTEXTS)), m, 3, "d0b_la_pool").value


def run_agents(cfg, seed, data, contexts, episodes, with_lookahead=True):
    params = make_params(cfg)
    na = n_actions(params)
    n = len(contexts)
    rows = np.arange(n)
    v1 = data["v1"]
    realised = data["fail_steps"].min(2) == SURVIVED
    a_o = v1[rows, contexts].argmax(1)
    model = C.schedule_model()
    out = {"O": a_o}
    extra = {}
    agents = [("I", None)]
    if with_lookahead:
        agents.append(("lookahead", Lookahead(params, lookahead_pool(params), seed_for("d0b_la", seed))))
    for name, la in agents:
        filt = R.RunLengthFilter(model)
        acts = np.empty(n, dtype=np.int64)
        w_true = np.empty(n)
        gains = []
        t0 = time.time()
        for t in range(n):
            w, pred = filt.weights()
            if la is None:
                acts[t] = R.myopic_action(w, v1[t])
            else:
                acts[t], g = la.choose(t, w, filt, v1[t], episodes.reserves[t], episodes.cue[t])
                gains.append(g)
            w_true[t] = w[contexts[t]]
            filt.update(data["sensor_ll"][t] + data["pattern_ll"][t, acts[t]], pred)
            if la is not None and t % 1000 == 0:
                print(f"lookahead t={t} evaluated={la.evaluated} {time.time() - t0:.0f}s", flush=True)
        out[name] = acts
        extra[name] = {"w_true": w_true, "seconds": time.time() - t0}
        if la is not None:
            extra[name].update({"decisions_with_lookahead": la.evaluated, "decisions_pruned_to_myopic": la.pruned_to_one,
                                "mean_evpi_bound": la.evpi_sum / n, "estimated_q_gain_mean": float(np.mean(gains))})
    # tabular (as D0, action set of this world)
    cents, scale = D0.tabular_centroids(params)
    feats = D0.features(data["y_supply"], data["y_probe"])
    rng = np.random.default_rng(seed_for("d0b_tabular", seed))
    bins = np.digitize(episodes.reserves, [5.0, 9.0])
    state = bins[:, 0] * 9 + bins[:, 1] * 3 + bins[:, 2]
    cue = episodes.cue[:, 0] * 4 + episodes.cue[:, 1] * 2 + episodes.cue[:, 2]
    succ = np.zeros((27, N_CONTEXTS, 8, na))
    fail = np.zeros_like(succ)
    a_t = np.empty(n, dtype=np.int64)
    chat = np.zeros(n, dtype=np.int64)
    ctx_hat = 0
    for t in range(n):
        s = (state[t], ctx_hat, cue[t])
        a_t[t] = int(np.argmax(rng.beta(1 + succ[s], 1 + fail[s])))
        chat[t] = ctx_hat
        (succ if realised[t, a_t[t]] else fail)[s + (a_t[t],)] += 1
        d = (((feats[t][None, None] - cents) / scale) ** 2).sum(-1)[:, episodes.cue[t, 0]]
        ctx_hat = int(np.argmin(d))
    out["tabular"] = a_t
    out["random"] = np.random.default_rng(seed_for("d0b_random", seed)).integers(0, na, n)
    out["none"] = np.zeros(n, dtype=np.int64)
    extra["tabular"] = {"context_accuracy": float((chat == contexts).mean())}
    return out, extra


def cmd_stream(args):
    cfg = chosen_cfg()
    params = make_params(cfg)
    seed = 1
    jobs = [((seed, s), (chunk_job, (cfg, seed, s, min(s + CHUNK, STREAM_LENGTH)))) for s in range(0, STREAM_LENGTH, CHUNK)]
    t0 = time.time()
    done = D0.run_jobs(D0._call, jobs, args.workers)
    rng = np.random.default_rng(int(time.time()))
    audit = []
    for key in rng.choice(len(jobs), 2, replace=False):
        (sd, s), (_, a) = jobs[key]
        path = RUNS / "stream" / cfg_key(cfg) / f"seed_{sd}" / f"chunk_{s:05d}.npz"
        digest = compute_chunk(a[0], a[1], a[2], a[3], path.with_name(path.stem + "_audit.npz"))
        audit.append({"start": int(s), "identical": digest == done[(sd, s)][1]})
        for extra_file in (path.with_name(path.stem + "_audit.npz"), path.with_name(path.stem + "_audit.sha256.json")):
            extra_file.unlink(missing_ok=True)
    data, metas = load_stream(cfg, seed)
    contexts = data["contexts"]
    _, episodes = stream_data(seed)
    actions, extra = run_agents(cfg, seed, data, contexts, episodes, with_lookahead=not args.no_lookahead)
    rows = np.arange(len(contexts))
    v2 = data["v2"]
    q = np.array_split(rows, 4)
    o_val = v2[rows, actions["O"]]
    realised = data["fail_steps"].min(2) == SURVIVED
    res = {"cfg": cfg, "seed": seed, "episodes": int(len(rows)), "chunk_audit": audit, "agents": {}}
    for name, a in actions.items():
        val = v2[rows, a]
        reg = o_val - val
        se = np.sqrt(data["v2se"][rows, actions["O"]] ** 2 + data["v2se"][rows, a] ** 2)
        res["agents"][name] = {"mean_expected_survival": float(val.mean()), "mean_realised_survival": float(realised[rows, a].mean()),
                               "cumulative_regret_vs_O": float(reg.sum()),
                               "cumulative_regret_vs_O_se_conservative": float(np.sqrt((se ** 2).sum())),
                               "per_quarter_mean_regret_vs_O": [float(reg[qq].mean()) for qq in q],
                               "action_freq": np.bincount(a, minlength=n_actions(params)).tolist()}
    i_reg = res["agents"]["I"]["cumulative_regret_vs_O"]
    t_reg = res["agents"]["tabular"]["cumulative_regret_vs_O"]
    res["tabular_regret_over_I_regret"] = float(t_reg / i_reg) if i_reg > 0 else None
    res["too_easy_flag"] = bool(i_reg > 0 and t_reg <= 1.2 * i_reg)
    res["tabular_context_accuracy"] = extra["tabular"]["context_accuracy"]
    if "lookahead" in actions:
        la_reg = res["agents"]["lookahead"]["cumulative_regret_vs_O"]
        diff = v2[rows, actions["lookahead"]] - v2[rows, actions["I"]]
        res["lookahead_vs_I"] = {"gain_in_cumulative_regret": float(i_reg - la_reg),
                                 "gain_fraction_of_I_regret": float((i_reg - la_reg) / i_reg) if i_reg > 0 else None,
                                 "paired_se_of_gain_conservative": float(np.sqrt(((data["v2se"][rows, actions["lookahead"]] ** 2)
                                                                                  + data["v2se"][rows, actions["I"]] ** 2)[diff != 0].sum())),
                                 "episodes_with_different_action": int((actions["lookahead"] != actions["I"]).sum()),
                                 "decisions_with_lookahead": extra["lookahead"]["decisions_with_lookahead"],
                                 "decisions_pruned_to_myopic": extra["lookahead"]["decisions_pruned_to_myopic"],
                                 "mean_evpi_bound": extra["lookahead"]["mean_evpi_bound"],
                                 "seconds": extra["lookahead"]["seconds"],
                                 "rule": "report I as a myopic reference and adopt lookahead if gain > 10% of I's regret vs O",
                                 "adopt_lookahead_as_reference": bool(i_reg > 0 and (i_reg - la_reg) > 0.1 * i_reg)}
    lags = D0.identification_lags(contexts, extra["I"]["w_true"])
    def lag_summary(sel):
        vals = [x["lag"] for x in lags if sel(x) and x["lag"] is not None]
        cens = sum(1 for x in lags if sel(x) and x["lag"] is None)
        return {"runs": len(vals) + cens, "median": float(np.median(vals)) if vals else None,
                "mean": float(np.mean(vals)) if vals else None, "max": int(max(vals)) if vals else None, "never_reached": cens}
    res["I_identification_lag_after_switch"] = {"all": lag_summary(lambda x: x["start"] > 0),
                                                "changed_only_C": lag_summary(lambda x: x["changed_factors"] == "C"),
                                                "changed_only_B": lag_summary(lambda x: x["changed_factors"] == "B"),
                                                "heldout_runs": lag_summary(lambda x: x["heldout"]),
                                                "minority_runs": lag_summary(lambda x: x["minority"])}
    stats = {}
    for mt in metas:
        for k, v in mt["pattern_stats"].items():
            stats[k] = stats.get(k, 0) + v
    res["likelihood_pattern_stats"] = stats
    res["o_mc_se_mean"] = float(data["v2se"][rows, actions["O"]].mean())
    res["seconds"] = time.time() - t0
    D0.save_arrays(RUNS / "stream" / cfg_key(cfg) / f"seed_{seed}_agents.npz", meta={"seed": seed},
                   **{f"action_{k}": v for k, v in actions.items()})
    D0.write_json(REPORT / "easiness.json", res)
    print(json.dumps({k: v["cumulative_regret_vs_O"] for k, v in res["agents"].items()}, indent=1), res.get("lookahead_vs_I"), flush=True)


# ---------------------------------------------------------------- timing

def cmd_timing(args):
    cfg = chosen_cfg()
    params = make_params(cfg)
    na = n_actions(params)
    ep = sample_episodes(40, seed_for("d0b_timing"))
    ctx = np.full(40, 5)
    out = {"machine": platform.processor(), "threads": 1, "n_actions": na}
    def per(fn, reps=40):
        t0 = time.perf_counter()
        for k in range(reps):
            fn(k % 40)
        return (time.perf_counter() - t0) / reps
    out["O_one_context_s"] = per(lambda k: R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), [5], R.draws(k, O_M)))
    out["I_values_all_8_contexts_s"] = per(lambda k: R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), list(range(8)), R.draws(k, O_M)))
    sup = ep.supplies(params, ctx)
    fs = np.stack([simulate(params, ctx, ep, np.full(40, a), sup).fail_step for a in range(na)], 1)
    ys, yp = observe(params, ctx, ep, sup)
    t0 = time.perf_counter()
    R.episode_loglik(params, ep, ys, yp, fs, list(range(na)), 3)
    out["likelihood_all_actions_s"] = (time.perf_counter() - t0) / 40
    t0 = time.perf_counter()
    R.episode_loglik(params, ep, ys, yp, fs, [1], 3)
    out["likelihood_one_action_s"] = (time.perf_counter() - t0) / 40
    ez = json.loads((REPORT / "easiness.json").read_text(encoding="utf-8")) if (REPORT / "easiness.json").exists() else {}
    la = ez.get("lookahead_vs_I", {})
    if la:
        out["lookahead_stream_seconds"] = la["seconds"]
        out["lookahead_decisions_with_lookahead"] = la["decisions_with_lookahead"]
    d0t = json.loads((ROOT / "reports/commons_d0/timing.json").read_text(encoding="utf-8"))
    arm = d0t["nn_model_based_arm_decision_plus_update_s"] * na / 7   # planning scales with the action count
    seeds, arms, workers = 12, 6, 6
    ref = out["I_values_all_8_contexts_s"] + out["O_one_context_s"] + out["likelihood_all_actions_s"]
    ref_h = ref * STREAM_LENGTH * seeds / 3600
    la_h = (la.get("seconds", 0.0) * seeds / 3600) if la else 0.0
    arm_h = arm * STREAM_LENGTH * seeds * arms / 3600
    out["projection_D2"] = {"assumptions": {"seeds": seeds, "arms": arms, "episodes": STREAM_LENGTH, "workers": workers,
                                            "arm_per_episode_s": arm, "arm_proxy": "D0 neural proxy scaled by n_actions/7",
                                            "lookahead_reference": "one lookahead pass per stream (sequential, 1 worker per stream)"},
                            "reference_cpu_hours": ref_h, "lookahead_cpu_hours": la_h, "arm_cpu_hours": arm_h,
                            "wall_hours_on_6_workers": (ref_h + la_h + arm_h) / workers,
                            "over_12h_flag": bool((ref_h + la_h + arm_h) / workers > 12)}
    D0.write_json(REPORT / "timing.json", out)
    print(json.dumps(out, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["screen", "refine", "separability", "cfrontier", "final", "verify", "voi", "schedule",
                                            "stream", "timing", "all"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-lookahead", action="store_true")
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, 6))
    REPORT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    cmds = {"screen": cmd_screen, "refine": cmd_refine, "separability": cmd_separability, "cfrontier": cmd_cfrontier, "final": cmd_final,
            "verify": cmd_verify, "voi": cmd_voi, "schedule": cmd_schedule, "stream": cmd_stream, "timing": cmd_timing}
    if args.command == "all":
        for name in ("screen", "refine", "separability", "cfrontier", "final", "verify", "voi", "schedule", "stream", "timing"):
            cmds[name](args)
    else:
        cmds[args.command](args)


if __name__ == "__main__":
    main()
