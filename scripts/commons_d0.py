"""Step D0 of the commons benchmark (docs/step_d_design.md): world, evaluator-only
references (O, I), verification, calibration of the factor dials, the
schedule generator, the tabular easiness check and CPU timing. No learners.

Usage (never writes bytecode):
  set PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src, then
  .venv/Scripts/python.exe -B scripts/commons_d0.py verify
  .venv/Scripts/python.exe -B scripts/commons_d0.py calibrate   [--workers 6]
  .venv/Scripts/python.exe -B scripts/commons_d0.py voi         [--workers 6]
  .venv/Scripts/python.exe -B scripts/commons_d0.py schedule
  .venv/Scripts/python.exe -B scripts/commons_d0.py easiness    [--workers 6]
  .venv/Scripts/python.exe -B scripts/commons_d0.py timing
  .venv/Scripts/python.exe -B scripts/commons_d0.py all

Outputs: reports/commons_d0/*.json|csv|png; raw arrays: runs/commons_d0/.
Every raw array file has a sha256 sidecar; existing files are re-verified on
load (resume) and a random subset is recomputed and compared bit-for-bit.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from acp_cl.commons import calibration as C  # noqa: E402
from acp_cl.commons import references as R  # noqa: E402
from acp_cl.commons import verification as V  # noqa: E402
from acp_cl.commons.world import (N_ACTIONS, N_CONTEXTS, PROBE, SURVIVED, WorldParams, context_name, observe, sample_episodes,
                                  seed_for, simulate, world_spec)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "commons_d0"
RUNS = ROOT / "runs" / "commons_d0"

# Fixed physics (dials are set separately). base_signs (1, -1, -.5): entity 3 has a
# mild first-half deficit so that the 2<->3 and 3<->1 channels are decision-relevant.
BASE = WorldParams(base_signs=(1.0, -1.0, -0.5))
ANCHOR = {"flip_amplitude": 1.0, "b_gap": 0.3, "c_delta": 6.0}
GRID = {"A": ("flip_amplitude", [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]),
        "B": ("b_gap", [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8]),
        "C": ("c_delta", [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0])}
POOL_N = 2048          # episodes per pair pool
POOL_M = 4096          # MC draws per value (selection and scoring sets, independent)
ID_SEQUENCES = 2000
ID_CAP = 4000
O_M = 2 ** 14          # O's draws per value in streams (selection and scoring)
STREAM_LENGTH = 20000
CHUNK = 500
VOI_N = 1024
TARGET = {"B": ("high", "high"), "C": ("low", "high")}   # lead's example target cells


# ---------------------------------------------------------------- io helpers

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def array_digest(arrays):
    return V.digest_arrays(*[arrays[k] for k in sorted(arrays)])


def save_arrays(path, meta=None, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    digest = array_digest(arrays)
    write_json(path.with_suffix(".sha256.json"), {"digest": digest, "keys": sorted(arrays), "meta": meta or {}})
    return digest


def load_verified(path):
    """Load arrays and verify them against the sidecar digest; None if absent or corrupt."""
    path = Path(path)
    side = path.with_suffix(".sha256.json")
    if not path.exists() or not side.exists():
        return None
    try:
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        info = json.loads(side.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - corrupt file: recompute
        return None
    if array_digest(arrays) != info["digest"]:
        print(f"digest mismatch, recomputing: {path}", flush=True)
        return None
    return arrays, info


def params_from(dials, probe_start="episode"):
    return BASE.with_dials(**dials, probe_start=probe_start)


def run_jobs(fn, jobs, workers):
    results = {}
    if workers <= 1:
        for key, args in jobs:
            results[key] = fn(*args)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): key for key, args in jobs}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


# ---------------------------------------------------------------- calibration pools

def pool_path(tag, factor, dial, i, j):
    return RUNS / "calibration" / tag / f"{factor}_{dial:g}_{i}{j}.npz"


def pool_job(tag, factor, dial, i, j, dials, n, m, seed_key):
    path = pool_path(tag, factor, dial, i, j)
    loaded = load_verified(path)
    if loaded is not None:
        return str(path), loaded[1]["digest"], True
    params = params_from(dials)
    t0 = time.time()
    pool = C.pair_pool(params, i, j, n, seed_for(seed_key, i, j), m, m)
    arrays = {"v_select": pool.v_select, "v_score": pool.v_score, "se_score": pool.se_score}
    for level in C.LEVELS:
        arrays[f"llr_{level}"] = pool.llr[level]
    digest = save_arrays(path, meta={"dials": dials, "pair": [i, j], "n": n, "m": m, "stats": pool.stats,
                                     "seconds": time.time() - t0}, **arrays)
    return str(path), digest, False


def pool_from_arrays(arrays, i, j):
    return C.PairPool((i, j), arrays["v_select"], arrays["v_score"], arrays["se_score"],
                      {level: arrays[f"llr_{level}"] for level in C.LEVELS}, {})


def metrics_rows(tag, factor, dial, dials, pairs):
    rows = []
    for i, j in pairs:
        arrays, info = load_verified(pool_path(tag, factor, dial, i, j))
        pool = pool_from_arrays(arrays, i, j)
        for level in C.LEVELS:
            m = C.pool_metrics(pool, level, n_boot=200, seed=seed_for("boot", factor, dial, i, j, level),
                               id_sequences=ID_SEQUENCES, id_cap=ID_CAP)
            rows.append({"factor": factor, "dial_name": GRID[factor][0], "dial": dial, "pair": f"{context_name(i)}-{context_name(j)}",
                         "i": i, "j": j, "level": level, **{k: v for k, v in m.items() if k != "behaviour_action_freq"},
                         "behaviour_action_freq": m["behaviour_action_freq"],
                         "pattern_entries_floored_frac": info["meta"]["stats"].get("floored", 0) / max(info["meta"]["stats"].get("entries", 1), 1)})
    return rows


def summarise(rows):
    out = {}
    for r in rows:
        key = (r["factor"], r["dial"], r["level"])
        out.setdefault(key, []).append(r)
    summary = []
    for (factor, dial, level), rs in sorted(out.items()):
        id_mean = float(np.mean([r["id_mean"] for r in rs]))
        rel = float(np.mean([r["relevance"] for r in rs]))
        sep, relg = C.grade(id_mean, rel)
        summary.append({"factor": factor, "dial": dial, "level": level, "id_mean_pairavg": id_mean,
                        "id_p90_max": float(max(r["id_p90"] for r in rs)),
                        "chernoff_pairavg": float(np.mean([r["chernoff_behaviour"] for r in rs])),
                        "sym_kl_pairavg": float(np.mean([r["sym_kl_behaviour"] for r in rs])),
                        "relevance_pairavg": rel,
                        "relevance_se_pairavg": float(np.sqrt(np.sum([r["relevance_se"] ** 2 for r in rs])) / len(rs)),
                        "relevance_min_pair": float(min(r["relevance"] for r in rs)),
                        "relevance_max_pair": float(max(r["relevance"] for r in rs)),
                        "separability": sep, "relevance_grade": relg,
                        "pair_grades": [list(C.grade(r["id_mean"], r["relevance"])) for r in rs]})
    return summary


def propose(summary):
    """Declared rule: for each passive probe-start variant, the smallest dial reaching the
    target cell (pair-averaged grades); prefer the variant reaching both targets."""
    proposals = {}
    for level in ("passive_episode", "passive_steady"):
        choice, reach = {}, {}
        for factor in ("B", "C"):
            rows = [s for s in summary if s["factor"] == factor and s["level"] == level]
            cells = sorted({(s["separability"], s["relevance_grade"]) for s in rows})
            reach[factor] = [list(c) for c in cells]
            hits = [s for s in rows if (s["separability"], s["relevance_grade"]) == TARGET[factor]]
            if hits:
                best = min(hits, key=lambda s: s["dial"])
                choice[factor] = {"dial": best["dial"], "achieved": True,
                                  "cell": [best["separability"], best["relevance_grade"]]}
            else:
                # nearest: the target separability with maximal relevance, else maximal relevance
                same_sep = [s for s in rows if s["separability"] == TARGET[factor][0]]
                pool = same_sep or rows
                best = max(pool, key=lambda s: s["relevance_pairavg"])
                choice[factor] = {"dial": best["dial"], "achieved": False,
                                  "cell": [best["separability"], best["relevance_grade"]]}
        proposals[level] = {"choice": choice, "reachable_cells": reach,
                            "both_achieved": all(c["achieved"] for c in choice.values())}
    preferred = next((lv for lv in ("passive_episode", "passive_steady") if proposals[lv]["both_achieved"]), None)
    return proposals, preferred


def cmd_calibrate(args):
    t0 = time.time()
    jobs = []
    for factor, (name, values) in GRID.items():
        for dial in values:
            dials = dict(ANCHOR, **{name: dial})
            for i, j in C.factor_pairs(factor):
                jobs.append(((factor, dial, i, j), (pool_job, ("grid", factor, dial, i, j, dials, POOL_N, POOL_M, "calib_grid"))))
    print(f"calibration grid: {len(jobs)} pools", flush=True)
    done = run_jobs(_call, [(k, a) for k, a in jobs], args.workers)
    reused = sum(1 for v in done.values() if v[2])
    rows = []
    for factor, (name, values) in GRID.items():
        for dial in values:
            rows += metrics_rows("grid", factor, dial, dict(ANCHOR, **{name: dial}), C.factor_pairs(factor))
    summary = summarise(rows)
    proposals, preferred = propose(summary)
    level = preferred or "passive_steady"
    chosen = dict(ANCHOR)
    chosen["b_gap"] = proposals[level]["choice"]["B"]["dial"]
    chosen["c_delta"] = proposals[level]["choice"]["C"]["dial"]
    probe_start = "steady" if level == "passive_steady" else "episode"
    # Final re-estimate on FRESH draws at the proposed dials, all 12 factor pairs.
    fjobs = []
    for factor in ("A", "B", "C"):
        dial = chosen[GRID[factor][0]]
        for i, j in C.factor_pairs(factor):
            fjobs.append(((factor, dial, i, j), (pool_job, ("final", factor, dial, i, j, chosen, POOL_N, POOL_M, "calib_final"))))
    run_jobs(_call, fjobs, args.workers)
    final_rows = []
    for factor in ("A", "B", "C"):
        final_rows += metrics_rows("final", factor, chosen[GRID[factor][0]], chosen, C.factor_pairs(factor))
    final_summary = summarise(final_rows)
    write_csv(REPORT / "calibration_grid.csv", rows)
    write_csv(REPORT / "calibration_final_pairs.csv", final_rows)
    result = {
        "world_spec": world_spec(params_from(chosen, probe_start)),
        "settings": {"pool_episodes": POOL_N, "mc_draws_select": POOL_M, "mc_draws_score": POOL_M,
                     "identification_sequences": ID_SEQUENCES, "identification_cap": ID_CAP,
                     "pattern_likelihood": R.PatternSettings().__dict__, "anchor_dials_during_scan": ANCHOR,
                     "grid": {k: v[1] for k, v in GRID.items()}, "levels": list(C.LEVELS),
                     "behaviour_policy": "myopic I with 50/50 belief over the pair (argmax of the mean of the two selection tables)",
                     "separability_metric_for_grading": "pair-averaged mean episodes for a pair-restricted myopic I to reach posterior .95 from 50/50",
                     "relevance_metric_for_grading": "pair-averaged cross-context policy regret (both directions averaged), selection/scoring on independent draws",
                     "thresholds": C.THRESHOLDS, "target_cells": {k: list(v) for k, v in TARGET.items()},
                     "proposal_rule": "per passive probe-start variant: smallest dial whose pair-averaged grade equals the target cell; prefer 'episode' if both targets reached there, else 'steady'"},
        "grid_summary": summary,
        "proposals": proposals,
        "preferred_level": level,
        "proposed_dials": dict(chosen, probe_start=probe_start),
        "final_summary_fresh_draws": final_summary,
        "pool_reuse": {"reused": reused, "computed": len(done) - reused},
        "seconds": time.time() - t0,
    }
    write_json(REPORT / "calibration.json", result)
    figure_calibration(summary, final_summary, chosen, level)
    print(json.dumps({"preferred_level": level, "proposed": result["proposed_dials"]}, indent=1), flush=True)


def _call(fn, args):
    return fn(*args)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def figure_calibration(summary, final_summary, chosen, level):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    names = {"A": "A flip amplitude", "B": "B efficiency gap", "C": "C delay difference"}
    th = C.THRESHOLDS
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, lv in zip(axes, C.LEVELS):
        ax.axvspan(0.8, th["separability_high_max_id_episodes"], color="#eeeeee", zorder=0)
        ax.axvspan(th["separability_low_min_id_episodes"], ID_CAP * 1.2, color="#eeeeee", zorder=0)
        ax.axhline(th["relevance_high_min"], color="#999999", lw=0.8, ls="--")
        ax.axhline(th["relevance_low_max"], color="#999999", lw=0.8, ls=":")
        for factor in ("A", "B", "C"):
            rows = sorted([s for s in summary if s["factor"] == factor and s["level"] == lv], key=lambda s: s["dial"])
            x = [s["id_mean_pairavg"] for s in rows]
            y = [max(s["relevance_pairavg"], 1e-4) for s in rows]
            ax.plot(x, y, "-o", color=colors[factor], lw=1.5, ms=5, label=names[factor])
            for s in rows[:: max(1, len(rows) // 4)] + rows[-1:]:
                ax.annotate(f"{s['dial']:g}", (s["id_mean_pairavg"], max(s["relevance_pairavg"], 1e-4)),
                            textcoords="offset points", xytext=(4, 3), fontsize=7, color="#52514e")
            fs = [s for s in final_summary if s["factor"] == factor and s["level"] == lv]
            for s in fs:
                ax.plot([s["id_mean_pairavg"]], [max(s["relevance_pairavg"], 1e-4)], marker="*", ms=13,
                        color=colors[factor], markeredgecolor="#0b0b0b", markeredgewidth=0.6, ls="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, ID_CAP * 1.2)
        ax.set_title(lv.replace("_", " "), fontsize=10)
        ax.grid(alpha=0.2)
    fig.supxlabel("episodes for pair-restricted I to reach posterior .95 (pair average; shaded bands: high / low separability)", fontsize=9)
    axes[0].set_ylabel("cross-context policy regret (pair average)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Commons D0 calibration: separability x relevance per factor dial (stars: proposed dials, fresh draws; "
                 f"preferred level {level}); dashed/dotted: high/low relevance thresholds", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORT / "calibration.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- VOI

def voi_job(truth, dials, n):
    path = RUNS / "voi" / f"truth_{truth}.npz"
    loaded = load_verified(path)
    if loaded is not None:
        return str(path)
    params = params_from(dials)
    episodes = sample_episodes(n, seed_for("voi_episodes"))
    truth_arr = np.full(n, truth)
    supplies = episodes.supplies(params, truth_arr)
    fs = C.fail_steps_all_actions(params, truth, episodes, supplies)
    y_supply, _ = observe(params, truth_arr, episodes, supplies, probes_on=False)
    stats = {}
    pattern = np.empty((n, N_ACTIONS, N_CONTEXTS))
    for e in range(n):
        pattern[e] = R.pattern_loglik(params, episodes.reserves[e], int(episodes.cue[e, 0]), y_supply[e],
                                      {a: fs[e, a] for a in range(N_ACTIONS)}, list(range(N_CONTEXTS)),
                                      list(range(N_ACTIONS)), seed_for("voi_pattern", truth, e), R.PatternSettings(), stats)
    sup, _ = R.supply_loglik(params, y_supply)
    a_idx = np.array([c >> 2 & 1 for c in range(N_CONTEXTS)])
    arrays = {"pattern": pattern, "supply": sup[:, a_idx]}
    for level in C.LEVELS:
        lp, on = C.level_params(params, level)
        _, yp = observe(lp, truth_arr, episodes, supplies, probes_on=on)
        arrays[f"probe_{level}"] = R.probe_loglik(lp, yp, episodes.cue[:, 0], on)
    save_arrays(path, meta={"truth": truth, "dials": dials, "stats": stats}, **arrays)
    return str(path)


def cmd_voi(args):
    cal = json.loads((REPORT / "calibration.json").read_text(encoding="utf-8"))
    dials = {k: cal["proposed_dials"][k] for k in ANCHOR}
    run_jobs(_call, [(t, (voi_job, (t, dials, VOI_N))) for t in range(N_CONTEXTS)], args.workers)
    data = [load_verified(RUNS / "voi" / f"truth_{t}.npz")[0] for t in range(N_CONTEXTS)]
    out = {"dials": dials, "episodes_per_context": VOI_N, "prior": "uniform over the 8 contexts (and uniform over each factor pair for binary MI)"}
    for level in C.LEVELS:
        lv = {}
        probe_only = [d[f"probe_{level}"] for d in data]
        sensors = [d["supply"] + d[f"probe_{level}"] for d in data]
        lv["mi_probe_only"] = C.mi_from_logliks(probe_only)
        lv["mi_sensors_only"] = C.mi_from_logliks(sensors)
        per_action, per_factor = [], {f: [] for f in "ABC"}
        for a in range(N_ACTIONS):
            full = [s + d["pattern"][:, a] for s, d in zip(sensors, data)]
            per_action.append(C.mi_from_logliks(full))
            for f in "ABC":
                per_factor[f].append(C.binary_factor_mi(full, f))
        lv["mi_full_by_action"] = {name: v for name, v in zip(ACTION_NAMES_LIST, per_action)}
        vals = np.array([v[0] for v in per_action])
        lv["voi_max_minus_min_nats"] = float(vals.max() - vals.min())
        lv["argmax_action"] = ACTION_NAMES_LIST[int(vals.argmax())]
        lv["argmin_action"] = ACTION_NAMES_LIST[int(vals.argmin())]
        for f in "ABC":
            fv = np.array([v[0] for v in per_factor[f]])
            lv[f"binary_mi_{f}_by_action"] = {name: v for name, v in zip(ACTION_NAMES_LIST, per_factor[f])}
            lv[f"binary_mi_{f}_sensors_only"] = C.binary_factor_mi(sensors, f)
            lv[f"binary_mi_{f}_probe_only"] = C.binary_factor_mi(probe_only, f)
            lv[f"binary_voi_{f}_max_minus_min"] = float(fv.max() - fv.min())
        out[level] = lv
    write_json(REPORT / "voi.json", out)
    print(json.dumps({lv: {k: out[lv][k] for k in ("mi_probe_only", "mi_sensors_only", "voi_max_minus_min_nats",
                                                     "binary_voi_B_max_minus_min", "binary_voi_C_max_minus_min")}
                      for lv in C.LEVELS}, indent=1), flush=True)


ACTION_NAMES_LIST = ["none", "1->2", "2->1", "2->3", "3->2", "3->1", "1->3"]


# ---------------------------------------------------------------- schedule

def cmd_schedule(args):
    cal = json.loads((REPORT / "calibration.json").read_text(encoding="utf-8"))
    level = cal["preferred_level"]
    low = [s for s in cal["final_summary_fresh_draws"] if s["factor"] == "C" and s["level"] == level][0]
    id_time = low["id_mean_pairavg"]
    spec = C.ScheduleSpec()
    out = {"spec": spec.__dict__, "training": [context_name(c) for c in C.TRAINING],
           "held_out": [context_name(c) for c in C.HELD_OUT], "minority": context_name(C.MINORITY),
           "run_length_distribution": f"P(L) proportional to 1/L on {{{spec.run_min}..{spec.run_max}}} (discrete log-uniform)",
           "run_length_mean": float((np.arange(1, spec.run_max + 1) * spec.run_pmf()).sum()),
           "low_separability_contrast": {"factor": "C", "level": level, "id_mean_episodes": id_time},
           "i_filter_model": ("semi-Markov over (context, run age); run pmf (1-eps) D + eps U{1..run_max}; switch matrix proportional "
                              "to each phase's quota shares with zero diagonal; forced switch at the phase boundary (episode 15000)"),
           "generator": ("per phase, per context: run lengths iid from D until the exact episode quota is filled (last run truncated); runs "
                         "ordered by sequential sampling weighted by remaining run counts with no immediate repeats; rejection until the "
                         "minority has >= 5 runs and max/min absence >= 4; held-out combos only in phase 2 (last quarter)"),
           "seeds": {}}
    for seed in (1, 2, 3):
        ctx, order, info = C.generate_schedule(seed, spec)
        out["seeds"][str(seed)] = C.schedule_statistics(ctx, order, info, spec, id_time)
        out["seeds"][str(seed)]["sha256"] = hashlib.sha256(ctx.tobytes()).hexdigest()
        out["seeds"][str(seed)]["run_sequence"] = [[context_name(c), r] for c, r in order]
        save_arrays(RUNS / "schedules" / f"seed_{seed}.npz", meta={"seed": seed}, contexts=ctx)
    write_json(REPORT / "schedule.json", out)
    for seed, s in out["seeds"].items():
        print(seed, {k: s[k] for k in ("runs", "run_length", "minority", "heldout_first_episode")}, flush=True)


# ---------------------------------------------------------------- stream (easiness)

def stream_data(seed, params):
    contexts, order, info = C.generate_schedule(seed)
    episodes = sample_episodes(len(contexts), seed_for("stream_episodes", seed))
    return contexts, episodes


def chunk_job(seed, dials, probe_start, start, stop):
    path = RUNS / "stream" / f"seed_{seed}" / f"chunk_{start:05d}.npz"
    loaded = load_verified(path)
    if loaded is not None:
        return str(path), loaded[1]["digest"], True
    return str(path), compute_chunk(seed, dials, probe_start, start, stop, path), False


def compute_chunk(seed, dials, probe_start, start, stop, path=None):
    params = params_from(dials, probe_start)
    contexts, episodes = stream_data(seed, params)
    idx = np.arange(start, stop)
    ep = episodes.take(idx)
    ctx = contexts[idx].astype(np.int64)
    n = len(idx)
    t0 = time.time()
    v1 = np.empty((n, N_CONTEXTS, N_ACTIONS))
    v1se = np.empty_like(v1)
    v2 = np.empty((n, N_ACTIONS))
    v2se = np.empty_like(v2)
    for k, e in enumerate(idx):
        z = R.draws(seed_for(seed, "O_select", int(e)), O_M)
        v1[k], v1se[k] = R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), list(range(N_CONTEXTS)), z)
        z = R.draws(seed_for(seed, "O_score", int(e)), O_M)
        val, se = R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), [int(ctx[k])], z)
        v2[k], v2se[k] = val[0], se[0]
    t_values = time.time() - t0
    supplies = ep.supplies(params, ctx)
    fs = np.empty((n, N_ACTIONS, 3), dtype=np.int8)
    for a in range(N_ACTIONS):
        fs[:, a] = simulate(params, ctx, ep, np.full(n, a), supplies).fail_step
    y_supply, y_probe = observe(params, ctx, ep, supplies, probes_on=True)
    t1 = time.time()
    stats = {}
    sensor, pattern = R.episode_loglik(params, ep, y_supply, y_probe, fs, list(range(N_ACTIONS)),
                                       seed_for(seed, "I_pattern", start), R.PatternSettings(), True, stats)
    t_lik = time.time() - t1
    arrays = {"v1": v1, "v1se": v1se, "v2": v2, "v2se": v2se, "fail_steps": fs, "y_supply": y_supply,
              "y_probe": y_probe, "sensor_ll": sensor, "pattern_ll": pattern, "contexts": ctx}
    meta = {"seed": seed, "start": start, "stop": stop, "dials": dials, "probe_start": probe_start, "pattern_stats": stats,
            "seconds_values": t_values, "seconds_likelihood": t_lik}
    if path is None:
        return array_digest(arrays)
    return save_arrays(path, meta=meta, **arrays)


def load_stream(seed, length):
    parts = {}
    metas = []
    for start in range(0, length, CHUNK):
        arrays, info = load_verified(RUNS / "stream" / f"seed_{seed}" / f"chunk_{start:05d}.npz")
        metas.append(info["meta"])
        for k, v in arrays.items():
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}, metas


def tabular_centroids(params, n=4000, seed=77):
    """Privileged nearest-centroid table: expected last-episode features per (context, that episode's cue1)."""
    feats, labels = [], []
    ep = sample_episodes(n, seed_for("tabular_centroids", seed))
    for c in range(N_CONTEXTS):
        ctx = np.full(n, c)
        sup = ep.supplies(params, ctx)
        ys, yp = observe(params, ctx, ep, sup, probes_on=True)
        f = features(ys, yp)
        feats.append(f)
        labels.append(np.stack([ctx, ep.cue[:, 0]], 1))
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    scale = np.concatenate([feats[(labels[:, 0] == c) & (labels[:, 1] == q)] -
                            feats[(labels[:, 0] == c) & (labels[:, 1] == q)].mean(0)
                            for c in range(N_CONTEXTS) for q in (0, 1)]).std(0) + 1e-9
    cents = np.zeros((N_CONTEXTS, 2, feats.shape[1]))
    for c in range(N_CONTEXTS):
        for q in (0, 1):
            cents[c, q] = feats[(labels[:, 0] == c) & (labels[:, 1] == q)].mean(0)
    return cents, scale


def features(y_supply, y_probe):
    return np.stack([y_supply[:, :6, 0].mean(1) - y_supply[:, 6:, 0].mean(1),
                     y_supply[:, :6, 1].mean(1) - y_supply[:, 6:, 1].mean(1),
                     y_probe[:, :, 0].mean(1) / PROBE,
                     y_probe[:, :3, 1].mean(1) / PROBE], 1)


def run_agents(seed, data, params, contexts, episodes):
    n = len(contexts)
    rows = np.arange(n)
    v1, v2 = data["v1"], data["v2"]
    realised = (data["fail_steps"].min(2) == SURVIVED)  # (n, 7)
    a_o = v1[rows, contexts].argmax(1)
    # I
    model = C.schedule_model()
    filt = R.RunLengthFilter(model)
    a_i = np.empty(n, dtype=np.int64)
    w_true = np.empty(n)
    w_max = np.empty(n)
    for t in range(n):
        w, pred = filt.weights()
        a_i[t] = R.myopic_action(w, v1[t])
        w_true[t] = w[contexts[t]]
        w_max[t] = w.max()
        filt.update(data["sensor_ll"][t] + data["pattern_ll"][t, a_i[t]], pred)
    # tabular: binned reserves x nearest-centroid context from last episode's sensors x cue; Thompson on Beta(1,1)
    cents, scale = tabular_centroids(params)
    feats = features(data["y_supply"], data["y_probe"])
    rng = np.random.default_rng(seed_for("tabular", seed))
    bins = np.digitize(episodes.reserves, [5.0, 9.0])
    state = bins[:, 0] * 9 + bins[:, 1] * 3 + bins[:, 2]
    cue = episodes.cue[:, 0] * 4 + episodes.cue[:, 1] * 2 + episodes.cue[:, 2]
    succ = np.zeros((27, N_CONTEXTS, 8, N_ACTIONS))
    fail = np.zeros_like(succ)
    a_t = np.empty(n, dtype=np.int64)
    chat = np.zeros(n, dtype=np.int64)
    ctx_hat = 0
    for t in range(n):
        s = (state[t], ctx_hat, cue[t])
        draws = rng.beta(1 + succ[s], 1 + fail[s])
        a_t[t] = int(np.argmax(draws))
        chat[t] = ctx_hat
        if realised[t, a_t[t]]:
            succ[s + (a_t[t],)] += 1
        else:
            fail[s + (a_t[t],)] += 1
        d = (((feats[t][None, None] - cents) / scale) ** 2).sum(-1)[:, episodes.cue[t, 0]]
        ctx_hat = int(np.argmin(d))
    a_r = np.random.default_rng(seed_for("random_policy", seed)).integers(0, N_ACTIONS, n)
    a_n = np.zeros(n, dtype=np.int64)
    return {"O": a_o, "I": a_i, "tabular": a_t, "random": a_r, "none": a_n}, w_true, w_max, chat


def identification_lags(contexts, w_true, threshold=0.95):
    starts = np.flatnonzero(np.r_[True, contexts[1:] != contexts[:-1]])
    ends = np.r_[starts[1:], len(contexts)]
    out = []
    for s, e in zip(starts, ends):
        ok = np.flatnonzero(w_true[s:e] >= threshold)
        prev = int(contexts[s - 1]) if s else -1
        changed = "".join(f for f, bit in (("A", 4), ("B", 2), ("C", 1)) if prev >= 0 and (prev ^ int(contexts[s])) & bit)
        out.append({"start": int(s), "length": int(e - s), "context": context_name(int(contexts[s])),
                    "changed_factors": changed, "lag": int(ok[0]) if len(ok) else None,
                    "heldout": int(contexts[s]) in C.HELD_OUT, "minority": int(contexts[s]) == C.MINORITY})
    return out


def cmd_easiness(args):
    cal = json.loads((REPORT / "calibration.json").read_text(encoding="utf-8"))
    dials = {k: cal["proposed_dials"][k] for k in ANCHOR}
    probe_start = cal["proposed_dials"]["probe_start"]
    params = params_from(dials, probe_start)
    seed = 1
    jobs = [((seed, s), (chunk_job, (seed, dials, probe_start, s, min(s + CHUNK, STREAM_LENGTH))))
            for s in range(0, STREAM_LENGTH, CHUNK)]
    t0 = time.time()
    done = run_jobs(_call, jobs, args.workers)
    # bit-flip audit: recompute 3 random chunks and compare digests
    rng = np.random.default_rng(int(time.time()))
    audit = []
    for key in rng.choice(len(jobs), 3, replace=False):
        (sd, s), (_, a) = jobs[key]
        digest = compute_chunk(*a[:5])
        audit.append({"start": int(s), "stored": done[(sd, s)][1], "recomputed": digest,
                      "identical": digest == done[(sd, s)][1]})
    data, metas = load_stream(seed, STREAM_LENGTH)
    contexts = data["contexts"]
    _, episodes = stream_data(seed, params)
    actions, w_true, w_max, chat = run_agents(seed, data, params, contexts, episodes)
    rows = np.arange(len(contexts))
    v2 = data["v2"]
    q = np.array_split(rows, 4)
    res = {"dials": dials, "probe_start": probe_start, "seed": seed, "episodes": int(len(rows)),
           "chunk_audit": audit, "chunks_reused": int(sum(1 for v in done.values() if v[2])),
           "agents": {}}
    o_val = v2[rows, actions["O"]]
    realised = data["fail_steps"].min(2) == SURVIVED
    for name, a in actions.items():
        val = v2[rows, a]
        reg = o_val - val
        se = np.sqrt(data["v2se"][rows, actions["O"]] ** 2 + data["v2se"][rows, a] ** 2)
        res["agents"][name] = {
            "mean_expected_survival": float(val.mean()),
            "mean_realised_survival": float(realised[rows, a].mean()),
            "cumulative_regret_vs_O": float(reg.sum()),
            "cumulative_regret_vs_O_se_conservative": float(np.sqrt((se ** 2).sum())),
            "per_quarter_mean_regret_vs_O": [float(reg[qq].mean()) for qq in q],
            "per_quarter_cumulative_regret_vs_O": [float(reg[qq].sum()) for qq in q],
            "action_freq": np.bincount(a, minlength=N_ACTIONS).tolist(),
        }
    i_reg = res["agents"]["I"]["cumulative_regret_vs_O"]
    t_reg = res["agents"]["tabular"]["cumulative_regret_vs_O"]
    res["I_minus_O_mean_expected_survival"] = float((v2[rows, actions["I"]] - o_val).mean())
    res["tabular_regret_over_I_regret"] = float(t_reg / i_reg) if i_reg > 0 else None
    res["too_easy_flag"] = bool(i_reg > 0 and t_reg <= 1.2 * i_reg)
    res["verdict_rule"] = "too easy if tabular cumulative regret vs O <= 1.2 x I's cumulative regret vs O"
    res["tabular_agent"] = ("state = 3 reserve bins per entity (<5, 5-9, >=9; 27 cells) x context estimate (nearest PRIVILEGED centroid "
                            "of 4 last-episode sensor features: entity-1/2 first-minus-second-half supply readings, mean 1->2 probe/.3, "
                            "mean of first three 2->3 probe readings/.3; centroids = expected features per (context, that episode's cue1), "
                            "distance standardised by within-context SD) x full 3-bit cue (8) = 1728 cells; Thompson sampling from "
                            "Beta(1+successes, 1+failures) per cell-action on realised joint survival")
    res["tabular_context_accuracy"] = float((chat == contexts).mean())
    res["tabular_context_accuracy_by_factor"] = {f: float((((chat ^ contexts) & bit) == 0).mean())
                                                 for f, bit in (("A", 4), ("B", 2), ("C", 1))}
    lags = identification_lags(contexts, w_true)
    def lag_summary(sel):
        vals = [x["lag"] for x in lags if sel(x) and x["lag"] is not None]
        cens = sum(1 for x in lags if sel(x) and x["lag"] is None)
        return {"runs": len(vals) + cens, "median": float(np.median(vals)) if vals else None,
                "mean": float(np.mean(vals)) if vals else None, "max": int(max(vals)) if vals else None,
                "never_reached": cens}
    res["I_identification_lag_after_switch"] = {
        "all": lag_summary(lambda x: x["start"] > 0),
        "changed_only_C": lag_summary(lambda x: x["changed_factors"] == "C"),
        "changed_only_B": lag_summary(lambda x: x["changed_factors"] == "B"),
        "changed_includes_A": lag_summary(lambda x: "A" in x["changed_factors"]),
        "heldout_runs": lag_summary(lambda x: x["heldout"]),
        "minority_runs": lag_summary(lambda x: x["minority"]),
    }
    res["I_weight_on_truth_mean"] = float(w_true.mean())
    res["I_weight_on_truth_per_quarter"] = [float(w_true[qq].mean()) for qq in q]
    stats = {}
    for m in metas:
        for k, v in m["pattern_stats"].items():
            stats[k] = stats.get(k, 0) + v
    res["likelihood_pattern_stats"] = stats
    res["o_mc_se_mean"] = float(data["v2se"][rows, actions["O"]].mean())
    res["seconds"] = time.time() - t0
    save_arrays(RUNS / "stream" / f"seed_{seed}_agents.npz", meta={"seed": seed},
                **{f"action_{k}": v for k, v in actions.items()}, w_true=w_true, tab_context=chat)
    write_json(REPORT / "easiness.json", res)
    figure_easiness(res, actions, o_val, v2, rows)
    print(json.dumps({k: res["agents"][k]["cumulative_regret_vs_O"] for k in res["agents"]}, indent=1),
          res["too_easy_flag"], flush=True)


def figure_easiness(res, actions, o_val, v2, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"I": "#2a78d6", "tabular": "#eb6834", "random": "#1baf7a", "none": "#eda100"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("I", "tabular", "random", "none"):
        cum = np.cumsum(o_val - v2[rows, actions[name]])
        ax.plot(rows + 1, cum, lw=2, color=colors[name], label=name)
        ax.annotate(name, (rows[-1] + 1, cum[-1]), textcoords="offset points", xytext=(4, 0), fontsize=8, color="#52514e")
    ax.axvline(15000, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("episode (held-out combinations after the dashed line)")
    ax.set_ylabel("cumulative expected joint-survival regret vs O")
    ax.set_title("Commons D0 easiness check (passive level, proposed dials, schedule seed 1)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT / "easiness.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- timing

def cmd_timing(args):
    cal = json.loads((REPORT / "calibration.json").read_text(encoding="utf-8"))
    dials = {k: cal["proposed_dials"][k] for k in ANCHOR}
    params = params_from(dials, cal["proposed_dials"]["probe_start"])
    ep = sample_episodes(40, seed_for("timing"))
    ctx = np.full(40, 5)
    out = {"machine": platform.processor(), "python": platform.python_version(), "threads": 1}
    def per(fn, reps=40):
        t0 = time.perf_counter()
        for k in range(reps):
            fn(k)
        return (time.perf_counter() - t0) / reps
    out["O_select_one_context_s"] = per(lambda k: R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), [5], R.draws(k, O_M)))
    out["O_score_one_context_s"] = out["O_select_one_context_s"]
    out["I_values_all_8_contexts_s"] = per(lambda k: R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), list(range(8)), R.draws(k, O_M)))
    sup = ep.supplies(params, ctx)
    fs = np.stack([simulate(params, ctx, ep, np.full(40, a), sup).fail_step for a in range(7)], 1)
    ys, yp = observe(params, ctx, ep, sup)
    t0 = time.perf_counter()
    R.episode_loglik(params, ep, ys, yp, fs, [0], 3)
    out["likelihood_one_action_s"] = (time.perf_counter() - t0) / 40
    t0 = time.perf_counter()
    R.episode_loglik(params, ep, ys, yp, fs, list(range(7)), 3)
    out["likelihood_all_actions_s"] = (time.perf_counter() - t0) / 40
    filt = R.RunLengthFilter(C.schedule_model())
    t0 = time.perf_counter()
    for k in range(200):
        w, pred = filt.weights()
        filt.update(np.zeros(8), pred)
    out["filter_step_s"] = (time.perf_counter() - t0) / 200
    out["world_step_s"] = per(lambda k: simulate(params, ctx[:1], ep.take(np.array([k % 40])), np.array([1])), 200)
    out["mc_planning_256x7_numpy_s"] = per(lambda k: R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), [5], R.draws(k, 256)))
    # neural proxy for a model-based arm: ensemble of 5 MLPs (in 16 -> 64 -> 64 -> 8) rolled out 12 steps on 256 x 7 samples,
    # plus 4 Adam updates of a 64-row minibatch per episode per member.
    try:
        import torch
        torch.set_num_threads(1)
        nets = [torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 8)) for _ in range(5)]
        opts = [torch.optim.Adam(n.parameters(), 1e-3) for n in nets]
        x = torch.randn(256 * 7, 16)
        xb = torch.randn(64, 16)
        yb = torch.randn(64, 8)
        def nn_decision(_):
            with torch.no_grad():
                for net in nets:
                    h = x
                    for _t in range(12):
                        h = torch.cat([net(h), h[:, 8:]], 1)
            for net, opt in zip(nets, opts):
                for _u in range(4):
                    opt.zero_grad()
                    loss = ((net(xb) - yb) ** 2).mean()
                    loss.backward()
                    opt.step()
        nn_decision(0)
        out["nn_model_based_arm_decision_plus_update_s"] = per(nn_decision, 20)
    except Exception as exc:  # noqa: BLE001
        out["nn_model_based_arm_decision_plus_update_s"] = None
        out["nn_error"] = repr(exc)
    seeds, arms, workers = 12, 6, 6
    ref_per_episode = out["I_values_all_8_contexts_s"] + out["O_score_one_context_s"] + out["likelihood_all_actions_s"] + out["filter_step_s"]
    arm_per_episode = out["nn_model_based_arm_decision_plus_update_s"] or out["mc_planning_256x7_numpy_s"]
    ref_hours = ref_per_episode * STREAM_LENGTH * seeds / 3600
    arm_hours = arm_per_episode * STREAM_LENGTH * seeds * arms / 3600
    out["projection_D2"] = {
        "assumptions": {"seeds": seeds, "arms": arms, "episodes": STREAM_LENGTH, "workers": workers,
                        "references_per_episode": "O selection+I values (8 contexts, shared draws) + O scoring (true context) + likelihood for all 7 actions + filter",
                        "arm_per_episode": "neural proxy (5-member MLP ensemble, 12-step rollouts of 256 x 7 samples, 4 Adam steps/member/episode); real arms may differ several-fold"},
        "reference_cpu_hours": ref_hours, "arm_cpu_hours": arm_hours,
        "wall_hours_on_6_workers": (ref_hours + arm_hours) / workers,
        "over_12h_flag": bool((ref_hours + arm_hours) / workers > 12)}
    write_json(REPORT / "timing.json", out)
    print(json.dumps(out, indent=1), flush=True)


# ---------------------------------------------------------------- verify

def smc_case_job(name, e, truth, action, ctx, entity, dials):
    params = params_from(dials)
    ep, _ = V.find_edge_cases(params)
    r = V.likelihood_case_smc(params, ep, e, truth, action, ctx, entity)
    r["case"] = name
    return r


def cmd_verify(args):
    t0 = time.time()
    dials = dict(ANCHOR)
    cal_path = REPORT / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        dials = {k: cal["proposed_dials"][k] for k in ANCHOR}
        spec = world_spec(params_from(dials, cal["proposed_dials"]["probe_start"]))
        spec["proposed_by"] = "reports/commons_d0/calibration.json (proposal rule declared there)"
        write_json(REPORT / "world_spec.json", spec)
    params = params_from(dials)
    out = {"dials": dials}
    out["simulator_vs_slow_reference"] = V.check_simulator(n=1500)
    print("simulator", out["simulator_vs_slow_reference"]["pass"], flush=True)
    out["gaussian_sensor_marginal"] = V.gaussian_marginal_check(params)
    out["o_value_tables_crn"] = V.check_value_tables(params)
    print("value tables", out["o_value_tables_crn"]["pass"], flush=True)
    out["filter_vs_bruteforce"] = V.check_filter()
    print("filter", out["filter_vs_bruteforce"]["pass"], flush=True)
    out["negative_supply_audit"] = V.negative_supply_audit(params)
    ep, cases = V.find_edge_cases(params)
    jobs = []
    for name, (e, truth, action) in cases.items():
        alt = V.alternative_context(truth, action)
        for ctx in (truth, alt):
            for entity in range(3):
                jobs.append(((name, ctx, entity), (smc_case_job, (name, e, truth, action, ctx, entity, dials))))
    lik = run_jobs(_call, jobs, args.workers)
    lik_rows = [lik[k] for k, _ in jobs]
    joint = {}
    for name, (e, truth, action) in cases.items():
        for ctx in (truth, V.alternative_context(truth, action)):
            joint[f"{name}_ctx{ctx}"] = V.joint_factorisation_case(params, ep, e, truth, action, ctx)
    covered = {}
    for name, (e, truth, action) in cases.items():
        one = ep.take(np.array([e]))
        sup = one.supplies(params, np.array([truth]))
        r = simulate(params, np.array([truth]), one, np.array([action]), sup)
        covered[name] = {"episode": e, "truth": context_name(truth), "action": ACTION_NAMES_LIST[action],
                         "reserves": one.reserves[0].round(3).tolist(), "fail_steps": r.fail_step[0].tolist(),
                         "overflow": r.overflow[0].round(3).tolist(), "sent": float(r.sent[0]),
                         "first_arrival_step_index": int(r.arrival_step_first[0]),
                         "min_supply": float(sup.min())}
    # naive prior importance sampling, for the record (why the sequential form is used)
    name = "typical_2"
    e, truth, action = cases[name]
    naive = V.likelihood_case(params, ep, e, truth, action, truth, 0, m_is=2 ** 21, k_conj=2 ** 16)
    out["likelihood_bruteforce"] = {
        "method": ("per entity: sequential prior-sampling estimator (bootstrap particle filter; supplies drawn from the prior, exact "
                   "Gaussian sensor density x failure-pattern indicator per step, multinomial resampling; 6 runs x 2^17 particles; "
                   "scalar reference physics) vs I's estimator (exact Gaussian marginal x conjugate posterior sampling through the "
                   "per-entity kernel, 2^18 draws); alternative contexts flip B or C so the supply prior is shared"),
        "cases": covered, "per_entity": lik_rows,
        "joint_full_simulator_vs_product_and_code": joint,
        "naive_prior_is_record": naive,
        "pass": bool(all(r["agree"] for r in lik_rows) and all(j["agree"] for j in joint.values())),
        "max_abs_z": float(max(abs(r.get("z", 0.0)) for r in lik_rows)),
    }
    print("likelihood", out["likelihood_bruteforce"]["pass"], flush=True)
    # determinism: pools, value tables and a stream chunk recomputed twice
    p1 = C.pair_pool(params, 0, 1, 48, 5, 512, 512)
    p2 = C.pair_pool(params, 0, 1, 48, 5, 512, 512)
    d1 = V.digest_arrays(p1.v_select, p1.v_score, *[p1.llr[k] for k in C.LEVELS])
    d2 = V.digest_arrays(p2.v_select, p2.v_score, *[p2.llr[k] for k in C.LEVELS])
    s1 = compute_small_chunk(dials)
    s2 = compute_small_chunk(dials)
    out["determinism"] = {"pair_pool_digest_1": d1, "pair_pool_digest_2": d2, "stream_chunk_1": s1, "stream_chunk_2": s2,
                          "pass": bool(d1 == d2 and s1 == s2)}
    checks = ["simulator_vs_slow_reference", "gaussian_sensor_marginal", "o_value_tables_crn", "filter_vs_bruteforce",
              "negative_supply_audit", "likelihood_bruteforce", "determinism"]
    out["all_pass"] = bool(all(out[k]["pass"] for k in checks))
    out["seconds"] = time.time() - t0
    write_json(REPORT / "verification.json", out)
    print("ALL PASS" if out["all_pass"] else "FAILURES: " + ", ".join(k for k in checks if not out[k]["pass"]), flush=True)


def compute_small_chunk(dials):
    params = params_from(dials)
    ep = sample_episodes(6, seed_for("det_chunk"))
    ctx = np.array([0, 3, 5, 7, 2, 4])
    v = [R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), list(range(8)), R.draws(k, 1024))[0] for k in range(6)]
    sup = ep.supplies(params, ctx)
    fs = np.stack([simulate(params, ctx, ep, np.full(6, a), sup).fail_step for a in range(7)], 1)
    ys, yp = observe(params, ctx, ep, sup)
    sensor, pattern = R.episode_loglik(params, ep, ys, yp, fs, list(range(7)), 9)
    return V.digest_arrays(np.array(v), fs, sensor, pattern)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["verify", "calibrate", "voi", "schedule", "easiness", "timing", "all"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, 6))
    REPORT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    commands = {"verify": cmd_verify, "calibrate": cmd_calibrate, "voi": cmd_voi, "schedule": cmd_schedule,
                "easiness": cmd_easiness, "timing": cmd_timing}
    if args.command == "all":
        for name in ("calibrate", "verify", "voi", "schedule", "easiness", "timing"):
            commands[name](args)
    else:
        commands[args.command](args)


if __name__ == "__main__":
    main()
