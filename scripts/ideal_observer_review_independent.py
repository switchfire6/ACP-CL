"""ADVERSARIAL REVIEW scratch check of the ideal-observer audit.

POST HOC / EXPLORATORY / NOT DECISION-BEARING. Fits no learner, reads no checkpoint.

Independent re-implementation (does NOT call acp_cl physics, ideal_observer_mc or the
implementer's forecast/scoring code):
  * own supply sampler (means / season / source sign / N(0,.35) / clip / mode swap),
  * own transfer physics incl. the three cue dependencies,
  * own posterior over the 5 reference laws from the saved support outcomes,
  * own O / I / P forecasts and own cell aggregation using the LOCKED scorer's
    ``score`` / ``cue_benefit`` / ``estimate`` functions (imported from
    scripts/summarize_representation_learning.py),
  * features decoded from the saved observation IMAGES (gauge, glyphs, weather, pulses)
    and checked against evaluator features.
Results are compared with reports/information_audit/ideal_observer/results.json.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

ACTIONS = np.array([0., -2., 2., -4., 4.])
STEPS = 12
REVIEW_ROOT = 777000123


def locked_module():
    spec = importlib.util.spec_from_file_location("locked_scorer", HERE / "summarize_representation_learning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- own physics

def my_supplies(rng, source, k):
    season = np.where(np.arange(STEPS) < 6, 1., -1.)
    means = 1.9 + 0.8 * source[None, :, None, None] * season[None, None, :, None] * np.array([1., -1.])
    return np.clip(means + rng.normal(0., .35, (k, len(source), STEPS, 2)), 0., None)


def my_simulate(reserves, lossy, delayed, source, signals, supplies, mode, active, action):
    """reserves (N,2); supplies (N,12,2) ALREADY in the mode frame; scalar action."""
    n = len(reserves)
    req = ACTIONS[action]
    direction = 1 if req < 0 else 0
    donor, recipient = direction, 1 - direction
    eff = np.where(lossy, .35, 1.)
    delay = np.where(delayed, 2, 0)
    if 0 in active:
        eff = np.where(lossy, np.where(signals[:, 0] == direction, .95, .10), eff)
    if 1 in active:
        delay = np.where(delayed, np.where(signals[:, 1] == direction, 1, 4), delay)
    sup = supplies
    if 2 in active:
        aff = lossy | delayed
        s_eff = source * (1 - 2 * mode)
        cross = np.where(signals[:, 2] == 0, 3, 9)
        new = np.where(np.arange(STEPS)[None] < cross[:, None], 1., -1.)
        old = np.where(np.arange(STEPS) < 6, 1., -1.)
        change = .8 * s_eff[:, None, None] * (new - old[None])[..., None] * np.array([1., -1.])
        sup = np.where(aff[:, None, None], np.maximum(sup + change, 0.), sup)
    stock = np.array(reserves, dtype=np.float64, copy=True)
    sent = np.minimum(abs(req), stock[:, donor])
    stock[:, donor] -= sent
    transit = sent * eff
    alive = np.ones(n, bool)
    surv = np.zeros((n, 3), np.uint8)
    for step in range(STEPS):
        arriving = delay == step
        stock[arriving, recipient] += transit[arriving]
        stock = stock + sup[:, step]
        stock = np.minimum(stock, 18.)
        stock = stock - np.minimum(stock, 2.)
        alive &= ~np.any(stock <= 1e-10, axis=1)
        if (step + 1) % 4 == 0:
            surv[:, (step + 1) // 4 - 1] = alive
    return surv


def mc_worker(job):
    """Independent MC for one seed: query survival probs per (law, variant) and support pattern probs."""
    os.environ["OMP_NUM_THREADS"] = "1"
    seed, laws, q, s, k_total, kb = (job[x] for x in ("seed", "laws", "query", "support", "k", "kb"))
    rng = np.random.default_rng([REVIEW_ROOT, seed])
    nq, ns = len(q["source"]), len(s["source"])
    pq = np.zeros((len(laws), 4, nq, 5, 3))
    ps = np.zeros((len(laws), ns, 4))
    for _ in range(k_total // kb):
        base_q = my_supplies(rng, q["source"], kb)
        base_s = my_supplies(rng, s["source"], kb)
        tq = lambda a: np.broadcast_to(a[None], (kb, *a.shape)).reshape(kb * len(a), *a.shape[1:])  # noqa: E731
        R, LO, DE, SO = tq(q["reserves"]), tq(q["lossy"]), tq(q["delayed"]), tq(q["source"])
        for li, (mode, active) in enumerate(laws):
            sup = (base_q[..., ::-1] if mode == 1 else base_q).reshape(kb * nq, STEPS, 2)
            for variant in (None, 0, 1, 2):
                if variant is not None and variant not in active:
                    pq[li, 1 + variant] += 0  # filled from correct below
                    continue
                sig = q["signals"].copy()
                if variant is not None:
                    sig[:, variant] ^= 1
                SG = tq(sig)
                for a in range(5):
                    out = my_simulate(R, LO, DE, SO, SG, sup, mode, active, a).reshape(kb, nq, 3)
                    pq[li, 0 if variant is None else 1 + variant, :, a] += out.sum(axis=0)
            # support: performed actions only
            ts = lambda a: np.broadcast_to(a[None], (kb, *a.shape)).reshape(kb * len(a), *a.shape[1:])  # noqa: E731
            sup_s = (base_s[..., ::-1] if mode == 1 else base_s).reshape(kb * ns, STEPS, 2)
            RS, LS, DS, SS, GS = (ts(s[x]) for x in ("reserves", "lossy", "delayed", "source", "signals"))
            AS = ts(s["actions"])
            counts = np.zeros((kb * ns, 3), np.uint8)
            for a in range(5):
                m = AS == a
                if m.any():
                    counts[m] = my_simulate(RS[m], LS[m], DS[m], SS[m], GS[m], sup_s[m], mode, active, a)
            level = counts.sum(axis=1).reshape(kb, ns)
            ps[li] += (level[..., None] == np.arange(4)).sum(axis=0)
    pq /= k_total
    ps /= k_total
    for li, (mode, active) in enumerate(laws):
        for c in (0, 1, 2):
            if c not in active:
                pq[li, 1 + c] = pq[li, 0]
    return seed, pq, ps


# ------------------------------------------------------------- image decode

def decode(images, glyphs):
    n = len(images)
    gauge = np.stack([images[:, 0, 0, 1], images[:, 0, 0, 9]], axis=1).astype(float)
    fields = []
    for field, col in enumerate((1, 6, 11)):
        patch = images[:, 0, 5:8, col:col + 3]
        match = [np.all(patch == glyphs[field, v][None], axis=(1, 2)) for v in (0, 1)]
        assert np.all(match[0] ^ match[1]), "ambiguous glyph"
        fields.append(match[1].astype(int))
    col0 = np.argmax(images[:, 0, 10, :] == 255, axis=1)
    source = np.where(col0 == 5, 1, np.where(col0 == 11, -1, 0))
    assert np.all(source != 0)
    signals = np.stack([(images[:, 0, 8, c + 1] == 255).astype(np.uint8) for c in (1, 6, 11)], axis=1)
    return dict(gauge=gauge, lossy=fields[0], delayed=fields[1], gate=fields[2], source=source, signals=signals)


# ------------------------------------------------------------- main

def main():
    t0 = time.time()
    import ideal_observer_load as L
    from acp_cl.acquisition.world import AcquisitionWorld
    from acp_cl.persistence.study import trial_seed
    from acp_cl.representation_learning.design import reference_laws
    locked = locked_module()
    endpoints = L.load_all(verify=False)
    seeds = sorted({s for s, _ in endpoints})
    world = AcquisitionWorld()
    report = dict(label="POST HOC / EXPLORATORY / NOT DECISION-BEARING (adversarial review scratch)")
    jobs, feats = [], {}
    physics_checked = 0
    for seed in seeds:
        laws = reference_laws(seed)
        qcases = world.dataset(laws[0], 512, trial_seed(seed, "acquisition_query", 0))
        ep = endpoints[seed, "interleaved"]
        # decode query features from the SAVED images and compare with evaluator features
        dq = decode(ep.targets[0].correct[0].observations, world.glyphs)
        assert np.array_equal(dq["gauge"], np.rint(qcases.base.reserves / 18 * 240)), "gauge"
        assert np.array_equal(dq["lossy"], qcases.factors[:, 1]) and np.array_equal(dq["delayed"], qcases.factors[:, 2])
        assert np.array_equal(dq["source"], qcases.factors[:, 0]) and np.array_equal(dq["signals"], qcases.signals)
        assert np.all(dq["gate"] == 0)
        # own physics reproduces saved truth from the realised (evaluator) supplies, every law
        for slot, law in enumerate(laws):
            cases = world.dataset(law, 512, trial_seed(seed, "acquisition_query", 0))
            truth = ep.targets[slot].correct[0].truth
            for a in range(5):
                out = my_simulate(cases.base.reserves, cases.factors[:, 1] == 1, cases.factors[:, 2] == 1,
                                  cases.factors[:, 0].astype(float), cases.signals, cases.base.supplies,
                                  law.mode, law.active, a)
                assert np.array_equal(out, truth[:, a]), f"own physics misses truth {seed} {slot} {a}"
                physics_checked += 1
            # exact generative check: undo swap, recover noise, compare to replayed rng draws
        supports = []
        for rep in (0, 1):
            sc = world.dataset(laws[0], 32, trial_seed(seed, "acquisition_support", [rep, False]))
            probe = ep.targets[0].correct[rep]
            ds = decode(probe.support_observations, world.glyphs)
            assert np.array_equal(ds["gauge"], np.rint(sc.base.reserves / 18 * 240))
            assert np.array_equal(ds["source"], sc.factors[:, 0]) and np.array_equal(ds["signals"], sc.signals)
            assert np.array_equal(ds["lossy"], sc.factors[:, 1]) and np.array_equal(ds["delayed"], sc.factors[:, 2])
            supports.append((sc, probe.support_actions.astype(int)))
            for slot, law in enumerate(laws):
                c2 = world.dataset(law, 32, trial_seed(seed, "acquisition_support", [rep, False]))
                acts = ep.targets[slot].correct[rep].support_actions.astype(int)
                out = np.zeros((32, 3), np.uint8)
                for a in range(5):
                    m = acts == a
                    if m.any():
                        out[m] = my_simulate(c2.base.reserves[m], c2.factors[m, 1] == 1, c2.factors[m, 2] == 1,
                                             c2.factors[m, 0].astype(float), c2.signals[m], c2.base.supplies[m],
                                             law.mode, law.active, a)
                assert np.array_equal(out, ep.targets[slot].correct[rep].support_outcomes), "support outcome"
                physics_checked += 1
        q = dict(reserves=qcases.base.reserves, lossy=qcases.factors[:, 1] == 1, delayed=qcases.factors[:, 2] == 1,
                 source=qcases.factors[:, 0].astype(float), signals=qcases.signals)
        s = dict(reserves=np.concatenate([x[0].base.reserves for x in supports]),
                 lossy=np.concatenate([x[0].factors[:, 1] == 1 for x in supports]),
                 delayed=np.concatenate([x[0].factors[:, 2] == 1 for x in supports]),
                 source=np.concatenate([x[0].factors[:, 0].astype(float) for x in supports]),
                 signals=np.concatenate([x[0].signals for x in supports]),
                 actions=np.concatenate([x[1] for x in supports]))
        feats[seed] = (q, s)
        jobs.append(dict(seed=seed, laws=[(law.mode, tuple(law.active)) for law in laws], query=q, support=s,
                         k=int(os.environ.get("REVIEW_K", 8192)), kb=256))
    report["own_physics_arrays_matching_truth"] = physics_checked
    print(f"[{time.time()-t0:.0f}s] features decoded from images and own physics verified ({physics_checked} arrays)",
          flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        mc = {seed: (pq, ps) for seed, pq, ps in pool.map(mc_worker, jobs)}
    print(f"[{time.time()-t0:.0f}s] MC done", flush=True)

    lam = 1e-4
    weights = {}
    for seed in seeds:
        pq, ps = mc[seed]
        ps_s = (1 - lam) * ps + lam / 4
        w = np.zeros((5, 2, 5))
        for t in range(5):
            for rep in (0, 1):
                obs = endpoints[seed, "interleaved"].targets[t].correct[rep].support_outcomes.sum(axis=1)
                ll = np.log(ps_s[:, 32 * rep + np.arange(32), obs]).sum(axis=1)
                x = np.exp(ll - ll.max())
                w[t, rep] = x / x.sum()
        weights[seed] = w

    def forecast(kind, seed, slot, rep, variant):
        pq = mc[seed][0]
        if kind == "O":
            f = pq[slot, variant]
        elif kind == "P":
            f = pq[:, variant].mean(axis=0)
        else:
            f = np.tensordot(weights[seed][slot, rep], pq[:, variant], axes=1)
        return np.asarray(f, dtype=np.float64)

    # own cell aggregation with the locked score/cue_benefit/estimate
    def end_targets(seed, phase, kind):
        ep = endpoints[seed, phase]
        laws = reference_laws(seed)
        out = []
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
            avg = {k: float(np.mean([p[k] for p in preds])) for k in preds[0]}
            out.append(dict(metrics=avg, marginal_gain=avg["marginal_brier"] - avg["brier"],
                            cue_benefits={c: float(np.mean(v)) for c, v in gains.items()}))
        return out

    ours = {}
    for kind in ("O", "I", "P"):
        rows = [dict(seed=seed, phase=phase, arm="outcome", introduced_cue=endpoints[seed, phase].introduced_cue,
                     end_targets=end_targets(seed, phase, kind))
                for seed in seeds for phase in ("fresh_1", "fresh_2", "fresh_3", "interleaved")]
        config = dict(seeds=seeds, smoke=False)
        result = locked.qualification(rows, config)
        ours[kind] = {g["name"]: dict(mean=g["metric"]["mean"], values=g["metric"]["values"],
                                      seeds=g["metric"]["seeds"], passed=g["passed"]) for g in result["groups"]}
    theirs = json.loads((ROOT / "reports/information_audit/ideal_observer/results.json").read_text())
    comparison = []
    worst = {}
    for cell in theirs["cells"]:
        for kind in ("O", "I", "P"):
            mine = ours[kind][cell["name"]]
            se = cell[kind]["mc_se_mean"]
            diff = mine["mean"] - cell[kind]["mean"]
            per_seed_diff = max(abs(v - cell[kind]["per_seed"][str(s)]) for s, v in zip(mine["seeds"], mine["values"]))
            comparison.append(dict(cell=cell["name"], kind=kind, theirs=cell[kind]["mean"], ours=mine["mean"],
                                   diff=diff, their_se=se, z_vs_their_se_x_sqrt3=diff / (se * math.sqrt(3)) if se else None,
                                   max_per_seed_diff=per_seed_diff, passed_theirs=cell[kind]["passed"],
                                   passed_ours=mine["passed"]))
            worst[kind] = max(worst.get(kind, 0), abs(diff))
    report["cell_comparison"] = comparison
    report["max_abs_mean_diff"] = worst
    report["pass_flags_agree"] = all(c["passed_theirs"] == c["passed_ours"] for c in comparison)
    # posterior comparison
    diag = theirs["posterior"]["diagnostics"]["rows"]
    wd = max(abs(np.array(r["weights"]) - weights[r["seed"]][r["slot"], r["replicate"]]).max() for r in diag)
    report["max_abs_posterior_weight_diff"] = float(wd)
    report["own_posterior_mass_true_by_role"] = {
        role: float(np.mean([weights[s][t, r, t] for s in seeds for r in (0, 1)]))
        for t, role in enumerate(("baseA", "baseB", "stage1", "stage2", "stage3"))}

    # decision relevance of the delay cue for O (focus survival on delayed rows)
    dec = {}
    for seed in seeds:
        laws = reference_laws(seed)
        ep = endpoints[seed, "interleaved"]
        for slot in range(2, 5):
            if 1 not in laws[slot].active:
                continue
            tgt = ep.targets[slot]
            fl = tgt.flips[1][0]
            aff = fl.affected
            fc, ff = forecast("O", seed, slot, 0, 0), forecast("O", seed, slot, 0, 2)
            a1, a2 = fc[:, :, -1].argmax(axis=1), ff[:, :, -1].argmax(axis=1)
            s1 = fl.truth[np.arange(512), a1, -1][aff]
            s2 = fl.truth[np.arange(512), a2, -1][aff]
            # O with the delay cue marginalised (average of correct and flipped forecasts)
            fm = 0.5 * (fc + ff)
            a3 = fm[:, :, -1].argmax(axis=1)
            s3 = fl.truth[np.arange(512), a3, -1][aff]
            # expected (not realised) survival under O's own probabilities
            e1 = fc[np.arange(512), a1, -1][aff].mean()
            e2 = fc[np.arange(512), a2, -1][aff].mean()
            e3 = fc[np.arange(512), a3, -1][aff].mean()
            dec[f"{seed}/slot{slot}"] = dict(realised_correct=int(s1.sum()), realised_flipped=int(s2.sum()),
                                             realised_marginalised=int(s3.sum()), n_affected=int(aff.sum()),
                                             expected_correct=float(e1), expected_flipped=float(e2),
                                             expected_marginalised=float(e3),
                                             argmax_changes_flip=int((a1 != a2)[aff].sum()),
                                             argmax_changes_marg=int((a1 != a3)[aff].sum()))
    report["delay_decision_O"] = dec
    k_used = int(os.environ.get("REVIEW_K", 8192))
    out = ROOT / f"reports/information_audit/ideal_observer/review_independent_K{k_used}.json"
    if k_used >= 8192 and not out.exists():
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(max_abs_mean_diff=worst, pass_flags_agree=report["pass_flags_agree"],
                          max_abs_posterior_weight_diff=wd,
                          mass_true=report["own_posterior_mass_true_by_role"]), indent=1))
    for c in comparison:
        if "cue_1" in c["cell"] or "slot_0/absolute" in c["cell"]:
            print(c["cell"], c["kind"], round(c["theirs"], 6), round(c["ours"], 6), round(c["diff"], 6),
                  None if c["z_vs_their_se_x_sqrt3"] is None else round(c["z_vs_their_se_x_sqrt3"], 2))
    zs = [c["z_vs_their_se_x_sqrt3"] for c in comparison if c["z_vs_their_se_x_sqrt3"] is not None]
    print("rms z", float(np.sqrt(np.mean(np.square(zs)))), "max |z|", float(np.max(np.abs(zs))))
    for k, v in dec.items():
        print(k, v)
    print(f"[{time.time()-t0:.0f}s] done")


if __name__ == "__main__":
    main()
