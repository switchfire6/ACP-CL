"""Companion to ideal_observer_check.py: exact-reserve variant and panel-conditional KL.

POST HOC / EXPLORATORY / NOT DECISION-BEARING. Never trains, updates, unpickles
or executes a learner or checkpoint; never writes into runs/; cannot alter the
locked STOP decision. Law identities, hidden modes, exact reserves and
counterfactual outcomes are EVALUATOR-ONLY; the predictors here are analysis
references, not learners.

Purpose (step 4 of the independent check). The first independent run
(ideal_observer_check.py) integrates each reserve over its gauge quantisation
bin, because the observer only sees the frames. The audited analysis used the
EXACT (evaluator-only) reserves in its primary run, with the gauge version as
a three-seed sensitivity. This script reuses the functions of
ideal_observer_check.py (my own code) to produce:
  (a) O, I and prior-only scores with exact reserves for queries AND supports
      (like-for-like with the audited primary), four fresh MC replicates;
  (b) the panel-conditional per-record KL estimand used by the audited
      'information' block: the mean over the seed's 512 query cases x 5 actions
      of KL(delay law || predecessor) on the 4-pattern feedback channel,
      with exact reserves and with gauge integration.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ideal_observer_check as C  # noqa: E402  (my own functions)

from acp_cl.acquisition.world import AcquisitionWorld  # noqa: E402

DEFAULT_OUTPUT = ROOT / "reports" / "information_audit" / "ideal_observer" / "independent_check_exact.json"


def support_worker_exact(seed, replicate, laws, supports, draws, chunk=8):
    """Pattern counts (support_rep, law, 32, 4) with exact support reserves."""
    started = time.perf_counter()
    world = AcquisitionWorld()
    rng = np.random.default_rng([C.ROOT_SEED, seed, replicate, 13])
    counts = []
    for observed, actions, reserves_exact in supports:
        n = len(actions)
        per_law = np.zeros((len(laws), n, 4), dtype=np.int64)
        for start in range(0, n, chunk):
            rows = slice(start, min(start + chunk, n))
            reserves, supplies = C.sample_hidden(rng, observed["levels"][rows], observed["factors"][rows, 0], draws,
                                                 reserves_exact[rows])
            for index, law in enumerate(laws):
                per_law[index, rows] = C.pattern_counts(world, law, reserves, supplies, observed["factors"][rows],
                                                        observed["signals"][rows], actions[rows])
        counts.append(per_law)
    return dict(kind="support", seed=seed, replicate=replicate, draws=draws, counts=np.stack(counts),
                seconds=time.perf_counter() - started)


def panel_kl_worker(seed, replicate, delay_law, predecessor, observed, reserves_exact, draws, exact, chunk=16):
    """Counts (512,5,4) of the 4-pattern channel under the delay law and its predecessor.

    Only delayed queries with a nonzero action are simulated; all other
    (query, action) pairs have identical physics under both laws (KL 0)."""
    started = time.perf_counter()
    world = AcquisitionWorld()
    rng = np.random.default_rng([C.ROOT_SEED, seed, replicate, 20 + int(exact)])
    factors, signals, levels = observed["factors"], observed["signals"], observed["levels"]
    n = len(factors)
    counts = np.zeros((2, n, 5, 4), dtype=np.int64)
    rows_all = np.flatnonzero(factors[:, 2] == 1)
    pairs = [(i, a) for i in rows_all for a in range(1, 5)]
    for start in range(0, len(pairs), chunk):
        block = pairs[start:start + chunk]
        idx = np.array([i for i, _ in block])
        act = np.array([a for _, a in block])
        reserves, supplies = C.sample_hidden(rng, levels[idx], factors[idx, 0], draws,
                                             reserves_exact[idx] if exact else None)
        for k, law in enumerate((delay_law, predecessor)):
            counts[k, idx, act] = C.pattern_counts(world, law, reserves, supplies, factors[idx], signals[idx], act)
    # uninformative pairs: fill both laws with an identical deterministic placeholder (KL 0)
    counts[:, :, 0, 3] = draws
    counts[:, factors[:, 2] == 0, :, 3] = draws
    return dict(kind="panel_kl", seed=seed, replicate=replicate, exact=exact, draws=draws, counts=counts,
                seconds=time.perf_counter() - started)


def panel_kl(counts_d, counts_p, draws, lam=None):
    """Mean over 512 x 5 (query, action) pairs; lam=None uses +.5 count smoothing, else
    mixture smoothing (1-lam) p + lam/4 restricted to p_D > 0 terms."""
    p_d, p_p = counts_d / draws, counts_p / draws
    if lam is None:
        q_d, q_p = (counts_d + .5) / (draws + 2.), (counts_p + .5) / (draws + 2.)
        terms = q_d * np.log(q_d / q_p)
    else:
        q_d, q_p = (1 - lam) * p_d + lam / 4, (1 - lam) * p_p + lam / 4
        terms = np.where(p_d > 0, q_d * np.log(q_d / q_p), 0.)
    return float(terms.sum(axis=-1).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[18101, 18102, 18103, 18104, 18105, 18106])
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--query-draws", type=int, default=4096)
    parser.add_argument("--support-draws", type=int, default=65536)
    parser.add_argument("--kl-draws", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    require = C.require
    require(args.workers <= 6, "at most 6 worker processes")
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "reports" / "information_audit"), "outputs belong under reports/information_audit/")
    require(not output.exists(), "refusing to overwrite an existing output")
    import ideal_observer_load as loader

    started = time.perf_counter()
    endpoints, observed, supports, exact_q = {}, {}, {}, {}
    for seed in args.seeds:
        endpoint = loader.load_endpoint(seed=seed, phase="interleaved")
        laws = C.my_laws(seed)
        require([t.law for t in endpoint.targets] == laws, "reference laws differ")
        endpoints[seed] = endpoint
        query = C.decode(endpoint.targets[0].correct[0].observations)
        observed[seed] = query
        regen = loader.regenerate(endpoint)
        exact_q[seed] = np.asarray(regen.queries[0].reserves, dtype=np.float64)
        require(np.array_equal(np.rint(exact_q[seed] / C.CAPACITY * C.GAUGE).astype(np.int64), query["levels"]),
                "exact reserves inconsistent with gauge")
        sets = []
        for rep in range(len(endpoint.targets[0].correct)):
            base = endpoint.targets[0].correct[rep]
            reserves = np.asarray(regen.supports[0, rep].cases.base.reserves, dtype=np.float64)
            for target in endpoint.targets:
                require(np.array_equal(np.asarray(regen.supports[target.slot, rep].cases.base.reserves), reserves),
                        "support reserves differ across targets")
            obs = C.decode(base.support_observations)
            require(np.array_equal(np.rint(reserves / C.CAPACITY * C.GAUGE).astype(np.int64), obs["levels"]),
                    "support reserves inconsistent with gauge")
            sets.append((obs, base.support_actions.astype(np.int64), reserves))
        supports[seed] = sets
    print(f"loaded {len(args.seeds)} seeds in {time.perf_counter() - started:.1f}s", flush=True)

    jobs, results = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for seed in args.seeds:
            laws = C.my_laws(seed)
            stage = C.delay_stage(seed)
            for exact in (True, False):
                jobs.append(pool.submit(panel_kl_worker, seed, 0, laws[stage + 1], laws[stage], observed[seed],
                                        exact_q[seed], args.kl_draws, exact))
            for rep in range(args.replicates):
                jobs.append(pool.submit(C.query_worker, seed, 10 + rep, laws, observed[seed], args.query_draws, 32,
                                        exact_q[seed]))
                jobs.append(pool.submit(support_worker_exact, seed, 10 + rep, laws, supports[seed], args.support_draws))
        for index, future in enumerate(as_completed(jobs), 1):
            item = future.result()
            results.append(item)
            print(f"{index}/{len(jobs)} {item['kind']} seed={item['seed']} rep={item['replicate']} "
                  f"{item['seconds']:.0f}s elapsed={time.perf_counter() - started:.0f}s", flush=True)

    report = dict(label=C.LABEL, settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  definitions=__doc__, seeds={})
    for seed in args.seeds:
        laws = C.my_laws(seed)
        stage = C.delay_stage(seed)
        endpoint = endpoints[seed]
        queries = sorted([r for r in results if r["kind"] == "query" and r["seed"] == seed], key=lambda r: r["replicate"])
        sups = sorted([r for r in results if r["kind"] == "support" and r["seed"] == seed], key=lambda r: r["replicate"])
        per_rep = [C.score_seed(endpoint, laws, q["forecasts"], s["counts"], s["draws"])[0] for q, s in zip(queries, sups)]
        pooled_f = {key: np.mean([q["forecasts"][key] for q in queries], axis=0) for key in queries[0]["forecasts"]}
        pooled_c = np.sum([s["counts"] for s in sups], axis=0)
        pooled, posts = C.score_seed(endpoint, laws, pooled_f, pooled_c, sum(s["draws"] for s in sups))

        def se(getter):
            values = np.array([getter(rows) for rows in per_rep], dtype=np.float64)
            return float(values.std(ddof=1) / np.sqrt(len(values)))

        slots = {}
        for slot, row in pooled.items():
            entry = dict(law=row["law"], posterior=row["posterior"])
            for name in ("oracle_brier", "ideal_brier", "prior_brier"):
                entry[name] = row[name]
                entry[name + "_se"] = se(lambda rows, n=name, k=slot: rows[k][n])
            for name in ("oracle", "ideal", "prior"):
                key = f"{name}_cue_benefit"
                entry[key] = row[key]
                entry[key + "_se"] = {c: se(lambda rows, k=slot, c=c, n=key: rows[k][n][c]) for c in row[key]}
            slots[str(slot)] = entry
        kl = {}
        for exact in (True, False):
            item = [r for r in results if r["kind"] == "panel_kl" and r["seed"] == seed and r["exact"] == exact][0]
            d, p = item["counts"]
            kl["exact" if exact else "gauge"] = {
                "count_smoothing": panel_kl(d, p, item["draws"]),
                "lam_1e-06": panel_kl(d, p, item["draws"], 1e-6),
                "lam_0.0001": panel_kl(d, p, item["draws"], 1e-4),
                "reverse_count_smoothing": panel_kl(p, d, item["draws"]),
                "reverse_lam_1e-06": panel_kl(p, d, item["draws"], 1e-6),
                "draws": item["draws"]}
        report["seeds"][str(seed)] = dict(delay_stage=stage, slots=slots, panel_kl=kl)
        print(json.dumps(dict(seed=seed, oracle=[round(slots[str(s)]["oracle_brier"], 6) for s in range(5)],
                              ideal=[round(slots[str(s)]["ideal_brier"], 6) for s in range(5)], panel_kl=kl)), flush=True)
    report["seconds"] = time.perf_counter() - started
    if args.no_write:
        print("--no-write: nothing written")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output.relative_to(ROOT).as_posix()} in {report['seconds']:.0f}s")


if __name__ == "__main__":
    main()
