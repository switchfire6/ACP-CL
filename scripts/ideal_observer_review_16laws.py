"""ADVERSARIAL REVIEW scratch check: ideal observer with a WEAKER hypothesis space.

POST HOC / EXPLORATORY / NOT DECISION-BEARING. Same independent physics/MC as
ideal_observer_review_independent.py, but the observer I16 does NOT know which 5 laws
exist: uniform prior over all 16 laws (mode 0/1 x every subset of the 3 cues). Scores the
31 cells with the locked scorer and reports whether I16 still passes, plus the O decision
value of each cue (flipped vs cue-marginalised forecasts, expected under O).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import ideal_observer_review_independent as R  # noqa: E402

ALL_LAWS = [(m, tuple(s)) for m in (0, 1) for k in range(4) for s in combinations(range(3), k)]


def worker(job):
    seed, pq, ps = R.mc_worker(job)
    return seed, pq.astype(np.float32), ps


def main():
    t0 = time.time()
    import ideal_observer_load as L
    from acp_cl.acquisition.world import AcquisitionWorld
    from acp_cl.persistence.study import trial_seed
    from acp_cl.representation_learning.design import reference_laws
    locked = R.locked_module()
    endpoints = L.load_all(verify=False)
    seeds = sorted({s for s, _ in endpoints})
    world = AcquisitionWorld()
    k = int(os.environ.get("REVIEW_K", 4096))
    jobs = []
    for seed in seeds:
        laws = reference_laws(seed)
        qc = world.dataset(laws[0], 512, trial_seed(seed, "acquisition_query", 0))
        sup = [world.dataset(laws[0], 32, trial_seed(seed, "acquisition_support", [r, False])) for r in (0, 1)]
        acts = [endpoints[seed, "interleaved"].targets[0].correct[r].support_actions.astype(int) for r in (0, 1)]
        q = dict(reserves=qc.base.reserves, lossy=qc.factors[:, 1] == 1, delayed=qc.factors[:, 2] == 1,
                 source=qc.factors[:, 0].astype(float), signals=qc.signals)
        s = dict(reserves=np.concatenate([x.base.reserves for x in sup]),
                 lossy=np.concatenate([x.factors[:, 1] == 1 for x in sup]),
                 delayed=np.concatenate([x.factors[:, 2] == 1 for x in sup]),
                 source=np.concatenate([x.factors[:, 0].astype(float) for x in sup]),
                 signals=np.concatenate([x.signals for x in sup]), actions=np.concatenate(acts))
        jobs.append(dict(seed=seed, laws=ALL_LAWS, query=q, support=s, k=k, kb=128))
    with ProcessPoolExecutor(max_workers=6) as pool:
        mc = {seed: (pq, ps) for seed, pq, ps in pool.map(worker, jobs)}
    print(f"[{time.time()-t0:.0f}s] MC done", flush=True)
    lam = 1e-4
    index = {law: i for i, law in enumerate(ALL_LAWS)}
    weights = {}
    for seed in seeds:
        ps = (1 - lam) * mc[seed][1] + lam / 4
        laws = reference_laws(seed)
        w = np.zeros((5, 2, len(ALL_LAWS)))
        for t in range(5):
            for rep in (0, 1):
                obs = endpoints[seed, "interleaved"].targets[t].correct[rep].support_outcomes.sum(axis=1)
                ll = np.log(ps[:, 32 * rep + np.arange(32), obs]).sum(axis=1)
                x = np.exp(ll - ll.max())
                w[t, rep] = x / x.sum()
        weights[seed] = w

    def forecast(kind, seed, slot, rep, variant):
        pq = mc[seed][0].astype(np.float64)
        law = reference_laws(seed)[slot]
        if kind == "O":
            return pq[index[(law.mode, law.active)], variant]
        return np.tensordot(weights[seed][slot, rep], pq[:, variant], axes=1)

    results = {}
    for kind in ("O", "I16"):
        rows = []
        for seed in seeds:
            laws = reference_laws(seed)
            for phase in ("fresh_1", "fresh_2", "fresh_3", "interleaved"):
                ep = endpoints[seed, phase]
                targets = []
                for target in ep.targets:
                    slot = laws.index(target.law)
                    preds, gains = [], {}
                    for rep, corr in enumerate(target.correct):
                        fc = forecast(kind, seed, slot, rep, 0)
                        preds.append(locked.score(fc, corr.truth, corr.affected, corr.valid, target.marginal))
                        for cue, flips in target.flips.items():
                            fl = flips[rep]
                            ff = forecast(kind, seed, slot, rep, 1 + cue)
                            gains.setdefault(str(cue), []).append(locked.cue_benefit(
                                dict(probabilities=fc, truth=corr.truth, valid=corr.valid),
                                dict(probabilities=ff, truth=fl.truth, affected=fl.affected, valid=fl.valid)))
                    avg = {m: float(np.mean([p[m] for p in preds])) for m in preds[0]}
                    targets.append(dict(metrics=avg, marginal_gain=avg["marginal_brier"] - avg["brier"],
                                        cue_benefits={c: float(np.mean(v)) for c, v in gains.items()}))
                rows.append(dict(seed=seed, phase=phase, arm="outcome", introduced_cue=ep.introduced_cue,
                                 end_targets=targets))
        res = locked.qualification(rows, dict(seeds=seeds, smoke=False))
        results[kind] = {g["name"]: dict(mean=g["metric"]["mean"], passed=g["passed"],
                                         per_seed=dict(zip(g["metric"]["seeds"], g["metric"]["values"])))
                         for g in res["groups"]}
    # posterior mass of I16 on the true law / true mode / true delay status
    mass = {}
    for t, role in enumerate(("baseA", "baseB", "stage1", "stage2", "stage3")):
        vals, modes, delay_ok = [], [], []
        for seed in seeds:
            law = reference_laws(seed)[t]
            for rep in (0, 1):
                w = weights[seed][t, rep]
                vals.append(w[index[(law.mode, law.active)]])
                modes.append(sum(w[i] for i, (m, a) in enumerate(ALL_LAWS) if m == law.mode))
                delay_ok.append(sum(w[i] for i, (m, a) in enumerate(ALL_LAWS) if (1 in a) == (1 in law.active)))
        mass[role] = dict(mass_true=float(np.mean(vals)), mass_true_mode=float(np.mean(modes)),
                          mass_true_delay_status=float(np.mean(delay_ok)))
    # O decision value per cue on its affected subset, expected under O (seed x slot mean), interleaved
    decision = {}
    for cue in (0, 1, 2):
        flips, margs = [], []
        for seed in seeds:
            laws = reference_laws(seed)
            for slot in range(2, 5):
                if cue not in laws[slot].active:
                    continue
                aff = endpoints[seed, "interleaved"].targets[slot].flips[cue][0].affected
                fc, ff = forecast("O", seed, slot, 0, 0), forecast("O", seed, slot, 0, 1 + cue)
                fm = .5 * (fc + ff)
                n = np.arange(512)
                best = fc[n, fc[:, :, -1].argmax(1), -1][aff].mean()
                flips.append(best - fc[n, ff[:, :, -1].argmax(1), -1][aff].mean())
                margs.append(best - fc[n, fm[:, :, -1].argmax(1), -1][aff].mean())
        decision[f"cue_{cue}"] = dict(expected_flip_loss_mean=float(np.mean(flips)), max=float(np.max(flips)),
                                      expected_cue_marginalised_loss_mean=float(np.mean(margs)),
                                      marg_max=float(np.max(margs)), n=len(flips))
    report = dict(label="POST HOC / EXPLORATORY / NOT DECISION-BEARING (adversarial review scratch)", k=k,
                  hypothesis_space=[dict(mode=m, active=list(a)) for m, a in ALL_LAWS],
                  cells=results, failed={k2: [n for n, v in r.items() if not v["passed"]] for k2, r in results.items()},
                  posterior_mass=mass, O_decision_value_interleaved=decision)
    out = ROOT / f"reports/information_audit/ideal_observer/review_16laws_K{k}.json"
    if k >= 4096 and not out.exists():
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(failed=report["failed"], posterior_mass=mass, decision=decision), indent=1))
    for name in results["I16"]:
        if "cue_1" in name or "absolute_brier" in name:
            print(name, "O", round(results["O"][name]["mean"], 5), "I16", round(results["I16"][name]["mean"], 5))
    print(f"[{time.time()-t0:.0f}s] done")


if __name__ == "__main__":
    main()
