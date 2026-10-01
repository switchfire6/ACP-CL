"""Step D2w (docs/commons_d2w_protocol.md): STAGE 0 calibration only (no learner exists yet).

Stage 0, as declared: the A x B world with C fixed at C0 (contexts A0B0, A0B1, A1B0, A1B1 = world indices 0, 2, 4, 6);
the evidence dial is the sensor noise (supply sigma_s, probe sigma_p, probes on/off); for each of the four
one-factor pairs (A: 0-4, 2-6; B: 0-2, 4-6), D0b's pair-pool machinery is re-run under the new observation model
(src/acp_cl/commons/references_d2w.py):

* pool: 2,048 episodes per pair; O's tables for the pair's two contexts (selection and scoring, 4,096 independent
  draws each; they do not depend on the sensors);
* per-episode full log-likelihood ratio (exact sensor terms + failure-pattern term by latent-supply posterior
  sampling) for every one of the 13 actions, data generated under each member of the pair;
* episodes to a .95 posterior for the pair-restricted myopic ideal observer (calibration.identification_time:
  uniform prior over the pair, stationary truth, 2,000 sequences, cap 4,000), relevance (cross-context policy
  regret, calibration.pool_metrics), and the 99th percentile of |per-episode LLR| at the declared behaviour action
  (I's myopic action at a 50/50 belief, D0b's declaration), pooled over both truths.

Selection rule (declared): among grid points meeting, for all four pairs, 20 <= episodes-to-.95 <= 60 and
p99 |LLR| < 3 nats, the smallest noise, ordered by sigma_s first, then probes on with the smallest sigma_p, with
"probes off" last. If no grid point qualifies, stage 0 FAILS and is reported before any learner is built.

Usage (PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=src):  .venv/Scripts/python.exe -B scripts/commons_d2w.py stage0
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import commons_d1 as D1  # noqa: E402
import commons_d2 as S2  # noqa: E402
from acp_cl.commons import calibration as C  # noqa: E402
from acp_cl.commons import d1_env as E  # noqa: E402
from acp_cl.commons import references as R  # noqa: E402
from acp_cl.commons import references_d2w as W  # noqa: E402
from acp_cl.commons.world import context_name, sample_episodes, seed_for, simulate  # noqa: E402

RUNS = ROOT / "runs" / "commons_d2w_stage0"
REPORT = ROOT / "reports" / "commons_d2w" / "stage0"
CONTEXTS = (0, 2, 4, 6)
PAIRS = {"A": [(0, 4), (2, 6)], "B": [(0, 2), (4, 6)]}
POOL_N, POOL_M = 2048, 4096
SIGMA_S = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1000.0]
SIGMA_P = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 1000.0]
TARGET = dict(id_min=20.0, id_max=60.0, p99_llr_max=3.0)
NA = 13


def pool_job(i, j, sigma_s):
    """Pattern + supply LLR (log p_i - log p_j) for all 13 actions under each truth; O tables (sigma-free) saved once."""
    path = RUNS / f"pool_{i}{j}_s{sigma_s:g}.npz"
    if D1.load_arrays(path, strict=False) is not None:
        return str(path), "verified_existing"
    params, _ = E.load_world(ROOT)
    t0 = time.time()
    ep = sample_episodes(POOL_N, seed_for("d2w_stage0_pool", i, j))
    tab = RUNS / f"tables_{i}{j}.npz"
    if D1.load_arrays(tab, strict=False) is None:
        vs = R.value_tables(params, ep, [i, j], POOL_M, seed_for("d2w_stage0", i, j), "select")
        vc = R.value_tables(params, ep, [i, j], POOL_M, seed_for("d2w_stage0", i, j), "score")
        D1.save_arrays(tab, meta=dict(pair=[i, j]), v_select=vs.value, v_score=vc.value, se_score=vc.se)
    noise = W.Noise(sigma_s, 0.1, True)
    out = {}
    for row, truth in enumerate((i, j)):
        s = ep.supplies(params, np.full(POOL_N, truth))
        ys, _ = W.observe(params, np.full(POOL_N, truth), ep, s, noise)
        fs = np.stack([simulate(params, np.full(POOL_N, truth), ep, np.full(POOL_N, a), s).fail_step for a in range(NA)], 1)
        pat = np.empty((POOL_N, NA, 2))
        stats = {}
        for e in range(POOL_N):
            pat[e] = W.pattern_loglik(params, ep.reserves[e], int(ep.cue[e, 0]), ys[e], {a: fs[e, a] for a in range(NA)}, [i, j],
                                      list(range(NA)), seed_for("d2w_stage0_pattern", i, j, truth, e), noise, stats=stats)
        sup = W.supply_loglik(params, ys, noise)
        ai, aj = (i >> 2) & 1, (j >> 2) & 1
        out[f"llr_pattern_{row}"] = pat[:, :, 0] - pat[:, :, 1]
        out[f"llr_supply_{row}"] = sup[:, ai] - sup[:, aj]
        out[f"floored_{row}"] = np.array([stats.get("floored", 0), stats.get("entries", 0)])
    D1.save_arrays(path, meta=dict(pair=[i, j], sigma_s=sigma_s, seconds=time.time() - t0), **out)
    return str(path), "completed"


def probe_llr(params, i, j, truth, sigma_p, probes_on):
    """Probe-sensor LLR (log p_i - log p_j) of the pool's episodes generated under ``truth`` (sigma_s-free)."""
    ep = sample_episodes(POOL_N, seed_for("d2w_stage0_pool", i, j))
    noise = W.Noise(0.1, sigma_p, probes_on)
    s = ep.supplies(params, np.full(POOL_N, truth))
    _, yp = W.observe(params, np.full(POOL_N, truth), ep, s, noise)
    pl = W.probe_loglik(params, yp, ep.cue[:, 0], noise)
    return pl[:, i] - pl[:, j]


def run_jobs(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=D1.worker_init, initargs=(1,)) as pool:
        futs = {pool.submit(pool_job, *a): a for a in jobs}
        for k, f in enumerate(as_completed(futs)):
            path, status = f.result()
            print(json.dumps(dict(done=k + 1, total=len(jobs), job=list(futs[f]), status=status)), flush=True)


def metrics_for(params, i, j, sigma_s, sigma_p, probes_on, cache):
    tab, _ = D1.load_arrays(RUNS / f"tables_{i}{j}.npz")
    arr, _ = D1.load_arrays(RUNS / f"pool_{i}{j}_s{sigma_s:g}.npz")
    llr = np.empty((2, POOL_N, NA))
    for row, truth in enumerate((i, j)):
        key = (i, j, truth, sigma_p, probes_on)
        if key not in cache:
            cache[key] = probe_llr(params, i, j, truth, sigma_p, probes_on)
        llr[row] = arr[f"llr_pattern_{row}"] + arr[f"llr_supply_{row}"][:, None] + cache[key][:, None]
    pool = C.PairPool((i, j), tab["v_select"], tab["v_score"], tab["se_score"], {"d2w": llr}, {})
    m = C.pool_metrics(pool, "d2w", n_boot=20, seed=seed_for("d2w_stage0_boot", i, j), id_sequences=2000, id_cap=4000)
    rows = np.arange(POOL_N)
    b = C.behaviour_actions(tab["v_select"])
    beh = np.abs(np.concatenate([llr[0][rows, b], llr[1][rows, b]]))
    none = np.abs(np.concatenate([llr[0][:, 0], llr[1][:, 0]]))
    pat_beh = np.abs(np.concatenate([arr["llr_pattern_0"][rows, b], arr["llr_pattern_1"][rows, b]]))
    return dict(pair=f"{context_name(i)}-{context_name(j)}", id_mean=m["id_mean"], id_p90=m["id_p90"],
                id_censored=max(m["id_censored_truth_i"], m["id_censored_truth_j"]), relevance=m["relevance"],
                kl_behaviour=0.5 * (m["kl_i_j"] + m["kl_j_i"]), p99_abs_llr_behaviour=float(np.percentile(beh, 99)),
                p99_abs_llr_no_transfer=float(np.percentile(none, 99)), p99_abs_llr_pattern_only_behaviour=float(np.percentile(pat_beh, 99)),
                mean_abs_llr_pattern_only_behaviour=float(pat_beh.mean()),
                floored_fraction=float((arr["floored_0"][0] + arr["floored_1"][0]) / max(arr["floored_0"][1] + arr["floored_1"][1], 1)))


def cmd_stage0(args):
    RUNS.mkdir(parents=True, exist_ok=True)
    jobs = [(i, j, s) for f in PAIRS for i, j in PAIRS[f] for s in SIGMA_S]
    run_jobs(jobs, min(args.workers, 6))
    params, _ = E.load_world(ROOT)
    cache, grid = {}, []
    for ss in SIGMA_S:
        for probes_on in (True, False):
            for sp in (SIGMA_P if probes_on else [0.1]):
                rows = {f: [metrics_for(params, i, j, ss, sp, probes_on, cache) for i, j in PAIRS[f]] for f in PAIRS}
                ok = all(TARGET["id_min"] <= r["id_mean"] <= TARGET["id_max"] and r["p99_abs_llr_behaviour"] < TARGET["p99_llr_max"]
                         for f in rows for r in rows[f])
                grid.append(dict(sigma_s=ss, sigma_p=sp if probes_on else None, probes_on=probes_on, meets_all=bool(ok), pairs=rows))
                print(json.dumps(dict(sigma_s=ss, sigma_p=sp if probes_on else "off", ok=ok,
                                      A=[(round(r["id_mean"], 1), round(r["p99_abs_llr_behaviour"], 2)) for r in rows["A"]],
                                      B=[(round(r["id_mean"], 1), round(r["p99_abs_llr_behaviour"], 2)) for r in rows["B"]])), flush=True)
    order = sorted([g for g in grid if g["meets_all"]], key=lambda g: (g["sigma_s"], not g["probes_on"], g["sigma_p"] or 1e9))
    chosen = order[0] if order else None
    out = dict(label="STAGE 0 calibration (docs/commons_d2w_protocol.md). No learner involved.", targets=TARGET,
               grid_sigma_s=SIGMA_S, grid_sigma_p=SIGMA_P, pool_episodes=POOL_N, o_draws=POOL_M,
               behaviour="I's myopic action at a 50/50 pair belief (D0b's declaration); identification: pair-restricted myopic I",
               selection_rule="smallest noise meeting all conditions for all 4 pairs: sigma_s first, then probes on by sigma_p, probes off last",
               chosen=chosen, stage0_passed=chosen is not None, grid=grid, utc=D1.now())
    D1.write_json(REPORT / "stage0.json", S2.clean(out))
    print("STAGE 0", "PASSED" if chosen else "FAILED: no grid point meets all declared conditions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["stage0"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    {"stage0": cmd_stage0}[args.command](args)


if __name__ == "__main__":
    main()
