"""Step D1b, Amendment 1 (docs/commons_d1b_protocol.md, "Amendment 1, before the lock").

Arms: MB (D1's MBLearner, D1's randomness keys), MB-prior (acp_cl.commons.learners_d1b.MBPrior) and MB-count
(acp_cl.commons.learners_count.MBCount). Development 2 ("trap rescue"): A1B1C1 on D1's known-trapped and known-clean
test seeds. Test: R2 on A1B1C1 x 24 fresh seeds; R1 on the other 4 contexts x the first 12 of them (+ their A1B1C1);
R3 on the R1 seeds. Reuses scripts/commons_d1.py and scripts/commons_d1b.py by import (neither is edited; D1b's
development-1 outputs stay verifiable against their own manifest).

Commands (PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=src):
  .venv/Scripts/python.exe -B scripts/commons_d1b2.py check   (configs/commons_d1b_smoke2.json -> runs/commons_d1b_smoke2)
  .venv/Scripts/python.exe -B scripts/commons_d1b2.py dev     (configs/commons_d1b_dev2.json   -> runs/commons_d1b_dev2, reports/commons_d1b/dev2)
  .venv/Scripts/python.exe -B scripts/commons_d1b2.py lock | run | replicate | analyze   (configs/commons_d1b.json -> runs/commons_d1b)
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
from math import comb  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import commons_d1 as D1  # noqa: E402
import commons_d1b as B1  # noqa: E402
from acp_cl.commons import d1_env as E  # noqa: E402
from acp_cl.commons import learners as L  # noqa: E402
from acp_cl.commons import learners_count as LC  # noqa: E402
from acp_cl.commons import learners_d1b as LB  # noqa: E402
from acp_cl.commons.world import SURVIVED, seed_for  # noqa: E402

sha, digest, read_json, write_json, now, require, log_event = D1.sha, D1.digest, D1.read_json, D1.write_json, D1.now, D1.require, D1.log_event
save_arrays, load_arrays, quarantine, fresh, chain = D1.save_arrays, D1.load_arrays, D1.quarantine, D1.fresh, D1.chain

LABEL = ("PROSPECTIVE STUDY under docs/commons_d1b_protocol.md, Amendment 1 (Step D1b). O is evaluator-only. "
         "All arms are bounded (1,024-episode reservoir).")
SCRIPT_FILES = ("scripts/commons_d1.py", "scripts/commons_d1b.py", "scripts/commons_d1b2.py")
SOURCE_FILES = tuple(f"src/acp_cl/commons/{n}.py" for n in
                     ("__init__", "world", "references", "calibration", "verification", "d1_env", "learners", "learners_d1b",
                      "learners_count"))
ARMS = ("MB", "MB-prior", "MB-count")
KNOB = {"MB-prior": "beta", "MB-count": "c"}
GRID = {"MB-prior": {"beta": [1, 3]}, "MB-count": {"c": [0.05, 0.15]}}
TRAP_CONTEXT = "A1B1C1"
DEV2_TRAPPED = [21103, 21110, 21111, 21112]
DEV2_CLEAN = [21104, 21105, 21106, 21107]
TEST_R2_SEEDS = list(range(21301, 21325))
TEST_R1_SEEDS = list(range(21301, 21313))
SMOKE_SEED = 21299
REPLICATE_SEEDS = (21301, 21306, 21311)
REPLICATE_OTHER_CONTEXT = "A0B0C1"
PROBES = [256, 512, 1024, 2048, 3072, 4096]
DISAGREEMENT = [1024, 4096]
TRAP = dict(context=TRAP_CONTEXT, regret_max=0.025, actions=[6, 9, 12], ratio_min=0.5, critical_loss=0.015,
            definition=("Ratified (Amendment 1). Critical actions: the non-null actions whose removal from O's action set costs "
                        ">= .015 expected survival on the A1B1C1 panel (O's choice by the selection table, values by the scoring "
                        "table) = {6: 1->3x3, 9: 2->3x6, 12: 1->3x6}. f_O(a) = fraction of the 512 A1B1C1 panel states in which a "
                        "is O's selected action. Usage condition (one-sided), per seed: for every critical a, training usage "
                        "u(a) = (A1B1C1 training episodes with action a) / 4,096 >= 0.5 f_O(a). A seed is clean if its A1B1C1 "
                        "greedy-probe regret at 4,096 is <= .025 AND the usage condition holds; otherwise it is trapped."))


# ------------------------------------------------------------------ config

def arm_values(arm):
    return list(GRID[arm][KNOB[arm]]) if arm in KNOB else [None]


def tag_of(arm, value):
    return "base" if arm == "MB" else f"{KNOB[arm]}{value:g}"


def value_of(arm, tag):
    for v in arm_values(arm):
        if tag_of(arm, v) == tag:
            return v
    raise ValueError(f"{arm} {tag}")


def cells(config):
    out = [(s, TRAP_CONTEXT) for s in config["trap_context_seeds"]]
    out += [(s, c) for s in config["other_context_seeds"] for c in config["contexts"] if c != TRAP_CONTEXT]
    return out


def fit_tasks(config, arms=None):
    tasks = []
    for arm in arms or config["arms"]:
        values = arm_values(arm) if config["phase"] == "dev2" or arm not in KNOB else [config["knobs"][arm]]
        for v in values:
            for seed, ctx in cells(config):
                tasks.append((arm, tag_of(arm, v), seed, ctx))
    return tasks


def oracle_tasks(config):
    out = [("panel", None, ctx, None) for ctx in sorted({c for _, c in cells(config)})]
    for seed, ctx in cells(config):
        for start in range(0, config["episodes"], config["oracle"]["stream_chunk"]):
            out.append(("stream", seed, ctx, start))
    return out


def validate_config(c):
    expected = {"study", "protocol", "phase", "world", "contexts", "trap_context_seeds", "other_context_seeds", "episodes",
                "checkpoint_every", "probe_episodes", "disagreement_episodes", "panel_size", "oracle", "learner", "base_knobs",
                "d1_selection", "d1b_dev1", "grid", "knobs", "arms", "disagreement_states", "analysis", "workers", "threads",
                "device", "stall_seconds", "stage_attempts", "notes"}
    if set(c) != expected:
        raise ValueError(f"invalid configuration fields: {sorted(set(c) ^ expected)}")
    require(c["protocol"] == "docs/commons_d1b_protocol.md" and c["phase"] in ("smoke2", "dev2", "test"), "protocol/phase")
    require(c["world"] == dict(separability="reports/commons_d0b/separability.json", spec="reports/commons_d0b/world_spec.json",
                               key=E.WORLD_KEY), "the D0b world")
    require(c["contexts"] == list(E.TRAINING_CONTEXTS), "the 5 training contexts")
    require(c["learner"] == D1.FIXED_LEARNER and c["base_knobs"] == B1.BASE_KNOBS, "D1's MB with its selected knobs")
    for key in ("d1_selection", "d1b_dev1"):
        require(sha(ROOT / c[key]["path"]) == c[key]["sha256"], f"{key}: file differs from the recorded hash")
    require(read_json(ROOT / c["d1_selection"]["path"])["selection"]["MB"]["selected"] == B1.BASE_KNOBS, "base = D1's MB selection")
    require(c["grid"] == GRID and c["arms"] == list(ARMS), "arms and grid fixed by Amendment 1")
    o = c["oracle"]
    require(set(o) == {"draws", "stream_chunk", "floor_draws"} and o["draws"] == 2 ** 14 and o["floor_draws"] == 256, "O as in D1")
    require(c["device"] == "cpu" and c["threads"] == 1 and 1 <= c["workers"] <= 6, "CPU, 1 thread, <= 6 workers")
    require(set(c["disagreement_episodes"]) <= set(c["probe_episodes"]), "disagreement probes are taken at greedy probes")
    a = dict(c["analysis"])
    if c["phase"] == "test":
        boot = {k: a.pop(k, None) for k in ("bootstrap_resamples", "analysis_seed")}
        require(boot["bootstrap_resamples"] == 20000 and type(boot["analysis_seed"]) is int, "declared bootstrap")
    dev2 = {k: a.pop(k, None) for k in ("dev2_trapped", "dev2_clean")}
    require(a == dict(competent_mean=0.015, competent_worst=0.025, r1_seeds_required=10, r2_seeds_required=20,
                      endpoint=c["episodes"], trap=TRAP, payback_reference="MB"), "pre-declared analysis constants")
    ph = c["phase"]
    if ph == "smoke2":
        require(c["trap_context_seeds"] == [SMOKE_SEED] and c["other_context_seeds"] == [SMOKE_SEED] and c["episodes"] == 256, "smoke")
    elif ph == "dev2":
        require(dev2 == dict(dev2_trapped=DEV2_TRAPPED, dev2_clean=DEV2_CLEAN) and c["trap_context_seeds"] == DEV2_TRAPPED + DEV2_CLEAN
                and c["other_context_seeds"] == [] and c["episodes"] == 4096 and c["probe_episodes"] == PROBES
                and c["disagreement_episodes"] == DISAGREEMENT and c["knobs"] is None, "development 2 design")
    else:
        require(c["trap_context_seeds"] == TEST_R2_SEEDS and c["other_context_seeds"] == TEST_R1_SEEDS and c["episodes"] == 4096
                and c["probe_episodes"] == PROBES and c["disagreement_episodes"] == DISAGREEMENT, "test design")
    if ph in ("smoke2", "test"):
        require(isinstance(c["knobs"], dict) and set(c["knobs"]) == set(KNOB), "one knob per remedy")
        for arm, v in c["knobs"].items():
            if v is None:
                raise ValueError(f"knob for {arm} is unfilled (fill from development 2 before locking)")
            require(v in arm_values(arm), f"{arm} knob must be a grid value")


def load_config(path):
    c = read_json(path)
    validate_config(c)
    return c


# ------------------------------------------------------------------ lock (explicit source list, see the D1 note)

def lock_payload(config, config_path):
    scripts = {n: sha(ROOT / n) for n in SCRIPT_FILES}
    source = {n: sha(ROOT / n) for n in SOURCE_FILES}
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                   device=config["device"], threads=config["threads"], workers=config["workers"], deterministic=True)
    identity = dict(config_sha256=digest(config), config_file_sha256=sha(config_path), protocol_sha256=sha(ROOT / config["protocol"]),
                    scripts_sha256=digest(scripts), source_sha256=digest(source), runtime_sha256=digest(runtime),
                    world_spec_sha256=sha(ROOT / config["world"]["spec"]))
    return dict(identity=identity, config=copy.deepcopy(config), config_path=Path(config_path).resolve().relative_to(ROOT).as_posix(),
                script_files=scripts, source_files=source, runtime=runtime)


def lock(config, config_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = lock_payload(config, config_path)
    path = output / "manifest.json"
    if path.exists():
        old = read_json(path)
        if old["identity"] != payload["identity"]:
            raise ValueError(f"an existing manifest locks different files: "
                             f"{[k for k in payload['identity'] if payload['identity'][k] != old['identity'].get(k)]}")
        return old
    manifest = dict(label=LABEL, phase=config["phase"], **payload, locked_utc=now(), fit_tasks=[list(t) for t in fit_tasks(config)])
    (output / "protocol_at_lock.md").write_bytes((ROOT / config["protocol"]).read_bytes())
    (output / "config_at_lock.json").write_bytes(Path(config_path).read_bytes())
    with zipfile.ZipFile(output / "source_at_lock.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(payload["script_files"]) | set(payload["source_files"])):
            archive.write(ROOT / name, name)
    manifest["source_zip_sha256"] = sha(output / "source_at_lock.zip")
    write_json(path, manifest)
    log_event(output, event="locked", identity=payload["identity"], phase=config["phase"])
    return manifest


def check_lock(config, config_path, output):
    manifest = read_json(Path(output) / "manifest.json")
    live = lock_payload(config, config_path)
    if manifest["identity"] != live["identity"]:
        raise ValueError(f"live files differ from the lock: {[k for k in live['identity'] if live['identity'][k] != manifest['identity'].get(k)]}")
    require(sha(Path(output) / "protocol_at_lock.md") == manifest["identity"]["protocol_sha256"], "archived protocol changed")
    return manifest


# ------------------------------------------------------------------ learners and fits

def make_arm_learner(config, arm, value, seed, c, track_ema=True, capacity=None):
    lr = config["learner"]
    hyper = L.Hyper(lr=B1.BASE_KNOBS["lr"], U=B1.BASE_KNOBS["U"], ema_decay=B1.BASE_KNOBS["ema_decay"], members=lr["members"],
                    hidden=lr["hidden"], minibatch=lr["minibatch"], grad_clip=lr["grad_clip"], bootstrap_p=lr["bootstrap_p"],
                    aux_weight=lr["aux_weight"], plan_samples=lr["plan_samples"], capacity=capacity or lr["capacity"], bounded=True,
                    total_episodes=config["episodes"])
    seeds = D1.learner_seeds(seed, c, "MB")
    if arm == "MB":
        return L.make_learner("MB", hyper, seeds, track_ema=track_ema), hyper
    rngs = {k: np.random.default_rng(seeds[k]) for k in L.RNG_STREAMS}
    if arm == "MB-prior":
        return LB.MBPrior(hyper, rngs, value, np.random.default_rng(seed_for("commons_d1b_prior", seed, c)), track_ema), hyper
    if arm == "MB-count":
        return LC.MBCount(hyper, rngs, value, track_ema), hyper
    raise ValueError(arm)


def fit_dir(output, arm, tag, seed, ctx, sub="fits"):
    return Path(output) / sub / f"{arm}__{tag}__s{seed}__{ctx}"


def run_fit(config, identity, output, arm, tag, seed, ctx, sub="fits", interrupt_after=None, capacity=None, track_ema=True,
            forced_actions=None):
    """D1's fit loop with the Amendment-1 arms; disagreement probes only at config['disagreement_episodes']."""
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = fit_dir(output, arm, tag, seed, ctx, sub)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, ckpt_path = directory / "result.json", directory / "checkpoint.pt"
    if result_path.exists():
        try:
            D1.verify_result(directory, identity)
            return dict(status="verified_existing", seconds=0.0)
        except Exception as error:  # noqa: BLE001
            quarantine(output, result_path, f"unverifiable result: {error!r}")
    value = value_of(arm, tag)
    params, _ = E.load_world(ROOT)
    c = E.context_of(ctx)
    n, every = config["episodes"], config["checkpoint_every"]
    probe_at = set(D1.probe_schedule(config))
    dis_at = set(config["disagreement_episodes"])
    stream = E.Stream(params, c, seed, n)
    learner, hyper = make_arm_learner(config, arm, value, seed, c, track_ema, capacity)
    panel = E.Panel(params, c, config["panel_size"], hyper.plan_samples, with_draws=True)
    nd = config["disagreement_states"]
    spec = dict(arm=arm, tag=tag, value=value, seed=seed, context=ctx, context_index=c, hyper=hyper.__dict__, episodes=n,
                checkpoint_every=every, probe_episodes=sorted(probe_at), disagreement_episodes=sorted(dis_at), track_ema=track_ema,
                forced_actions=forced_actions is not None, capacity_override=capacity,
                learner_class=f"{type(learner).__module__}.{type(learner).__qualname__}")
    saved = D1.checkpoint_load(output, ckpt_path, identity)
    resumed = False
    if saved is not None and saved["record"]["spec"] == spec:
        learner.load(saved["learner"])
        record, trace, probes, dis = saved["record"], saved["trace"], saved["probes"], saved["disagreement"]
        require(learner.hashes() == record["checkpoint_hashes"], "checkpoint learner state changed")
        record["resumes"].append(dict(utc=now(), episodes_done=record["episodes_done"]))
        resumed = True
    else:
        record = dict(identity=identity, label=LABEL, spec=spec, episodes_done=0, resumes=[], started_utc=now(), training_seconds=0.0,
                      probe_seconds=0.0, elapsed_seconds=0.0, stream_fingerprint=stream.fingerprint(), decisions_chain="",
                      sensors_chain="", outcomes_chain="", initial_hashes=learner.hashes(), max_replay=0)
        trace = dict(action=np.zeros(n, np.int8), member=np.zeros(n, np.int8), realised=np.zeros(n, bool),
                     replay_size=np.zeros(n, np.int16), base_action=np.zeros(n, np.int8), count_of_action=np.zeros(n, np.int16))
        probes, dis = [], []
    require(record["stream_fingerprint"] == stream.fingerprint(), "stream draws changed")

    def do_probe(episode):
        t0 = time.perf_counter()
        probes.append(dict(episode=episode, **D1.probe(learner, panel, "MB")))
        if episode in dis_at:
            before = learner.hashes()
            vals = LB.member_values(learner, panel.reserves[:nd], panel.cue[:nd], panel.eps[:nd])
            require(learner.hashes() == before, "the disagreement probe changed the learner")
            dis.append(dict(episode=episode, member_sd=vals.std(0).astype(np.float32)))
        record["probe_seconds"] += time.perf_counter() - t0

    started = time.perf_counter()
    if record["episodes_done"] == 0 and not probes:
        do_probe(0)
    for k in range(record["episodes_done"], n):
        t0 = time.perf_counter()
        decision = stream.decision(k)
        action = learner.act(decision)
        base = getattr(learner, "last_base_action", action)
        if forced_actions is not None:
            action = int(forced_actions[k])
        counts = getattr(learner, "last_counts", None)
        outcome = stream.outcome(k, action)
        learner.observe(decision, action, outcome)
        record["training_seconds"] += time.perf_counter() - t0
        record["decisions_chain"] = chain(record["decisions_chain"], decision.reserves, decision.cue)
        record["sensors_chain"] = chain(record["sensors_chain"], outcome.y_supply, outcome.y_probe)
        record["outcomes_chain"] = chain(record["outcomes_chain"], np.int64(action), outcome.fail_step)
        trace["action"][k] = action
        trace["member"][k] = learner.last_thompson
        trace["realised"][k] = bool(outcome.fail_step.min() == SURVIVED)
        trace["replay_size"][k] = learner.replay.size
        trace["base_action"][k] = base
        trace["count_of_action"][k] = -1 if counts is None else int(counts[action])
        require(learner.replay.size <= hyper.capacity, "bounded replay exceeded its budget")
        record["max_replay"] = max(record["max_replay"], learner.replay.size)
        record["episodes_done"] = k + 1
        if k + 1 in probe_at:
            do_probe(k + 1)
        if (k + 1) % every == 0:
            record["elapsed_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            record["checkpoint_hashes"] = learner.hashes()
            D1.checkpoint_save(ckpt_path, dict(identity=identity, learner=learner.state(), record=record, trace=trace, probes=probes,
                                               disagreement=dis))
        if interrupt_after is not None and k + 1 == interrupt_after:
            raise D1.Interrupted(f"deliberate interruption after episode {k + 1}")
    record["elapsed_seconds"] += time.perf_counter() - started
    arrays = dict(trace)
    arrays["probe_episode"] = np.array([p["episode"] for p in probes], np.int32)
    for key in probes[0]:
        if key != "episode":
            arrays[f"probe_{key}"] = np.stack([np.asarray(p[key]) for p in probes])
    arrays["disagreement_episode"] = np.array([x["episode"] for x in dis], np.int32)
    arrays["disagreement_member_sd"] = (np.stack([x["member_sd"] for x in dis]) if dis
                                        else np.zeros((0, nd, L.ACTIONS), np.float32))
    info = save_arrays(directory / "trace.npz", meta=dict(arm=arm, tag=tag, seed=seed, context=ctx), **arrays)
    record.update(final_hashes=learner.hashes(), max_replay=int(max(record["max_replay"], learner.replay.max_size)),
                  replay=dict(capacity=learner.replay.capacity, max_size=learner.replay.max_size, seen=learner.replay.seen,
                              evictions=learner.replay.evictions, record_bytes=learner.replay.record_bytes()),
                  optimizer_steps=learner.steps, macs=dict(learner.macs, total=sum(learner.macs.values())),
                  memory_bytes=learner.memory_bytes(), parameters=learner.n_params(), trace_digest=info["digest"], resumed=resumed,
                  completed_utc=now())
    record["artifact_hashes"] = {"trace.npz": sha(directory / "trace.npz"), "trace.sha256.json": sha(directory / "trace.sha256.json")}
    record["checkpoint_hashes"] = learner.hashes()
    D1.checkpoint_save(ckpt_path, dict(identity=identity, learner=learner.state(), record=record, trace=trace, probes=probes,
                                       disagreement=dis))
    record["artifact_hashes"]["checkpoint.pt"] = sha(ckpt_path)
    write_json(result_path, record)
    return dict(status="completed", seconds=record["elapsed_seconds"])


# ------------------------------------------------------------------ pool (B1's, with this module's dispatch)

def estimated_cost(config, kind, payload):
    if kind == "fit":
        probes = len(D1.probe_schedule(config))
        return {"MB": 1.0, "MB-prior": 1.8, "MB-count": 1.02}[payload["arm"]] * (config["episodes"] / 4096 * 70 + probes * 25)
    return 0.23 * config["panel_size"] if payload["kind"] == "panel" else 0.075 * config["oracle"]["stream_chunk"]


def dispatch(kind, payload):
    if kind == "fit":
        return run_fit(**payload)
    return D1.run_oracle(**payload)


def run_pool(config, output, jobs, stage):
    """B1.run_pool's logic (kill a broken/stalled pool, log every failure, retry; finished tasks only verify)."""
    saved = (B1.estimated_cost, B1.dispatch)
    B1.estimated_cost, B1.dispatch = estimated_cost, dispatch
    try:
        return B1.run_pool(config, output, jobs, stage)
    finally:
        B1.estimated_cost, B1.dispatch = saved


def fit_jobs(config, identity, output, tasks, sub="fits"):
    return [("fit", dict(config=config, identity=identity, output=str(output), arm=a, tag=t, seed=s, ctx=c, sub=sub))
            for a, t, s, c in tasks]


def oracle_jobs(config, identity, output):
    return [("oracle", dict(config=config, identity=identity, output=str(output), kind=k, seed=s, ctx=c, start=st))
            for k, s, c, st in oracle_tasks(config)]


def verify_all(config, output, identity, tasks, sub="fits"):
    fits = []
    for arm, tag, seed, ctx in tasks:
        d = fit_dir(output, arm, tag, seed, ctx, sub)
        rec = D1.verify_result(d, identity)
        load_arrays(d / "trace.npz")
        fits.append(dict(arm=arm, tag=tag, seed=seed, context=ctx, path=(d / "result.json").relative_to(output).as_posix(),
                         sha256=sha(d / "result.json"), elapsed_seconds=rec["elapsed_seconds"]))
    oracle = []
    for kind, seed, ctx, start in oracle_tasks(config):
        p = D1.oracle_path(output, kind, seed, ctx, start)
        _, info = load_arrays(p)
        require(info["meta"]["identity"] == identity, f"oracle identity mismatch {p}")
        oracle.append(dict(path=p.relative_to(output).as_posix(), sha256=sha(p)))
    return fits, oracle


# ------------------------------------------------------------------ metrics

def check_trap_definition(tables):
    losses = B1.critical_losses(tables, TRAP_CONTEXT)
    derived = [a for a in range(1, L.ACTIONS) if losses[a] >= TRAP["critical_loss"]]
    require(derived == TRAP["actions"], f"trap action set {derived} differs from the declared {TRAP['actions']}")
    return dict(losses=losses, derived=derived, o_best_frequency=B1.o_best_frequency(tables, TRAP_CONTEXT).tolist())


def fit_metrics(config, tables, output, arm, tag, seed, ctx, sub="fits"):
    d = fit_dir(output, arm, tag, seed, ctx, sub)
    rec = read_json(d / "result.json")
    tr, _ = load_arrays(d / "trace.npz")
    c = E.context_of(ctx)
    ptab, stab = tables.panel_tab(ctx), tables.stream_tab(seed, ctx)
    rows = np.arange(config["panel_size"])
    curve = [float(D1.panel_regret(ptab, c, a).mean()) for a in tr["probe_action"]]
    online = D1.stream_regret(stab, tr["action"])
    greedy = tr["probe_action"][-1].astype(np.int64)
    score = ptab["score"][:, c]
    pred = tr["probe_value"][-1].astype(np.float64)
    usage = np.bincount(tr["action"].astype(np.int64), minlength=L.ACTIONS) / len(tr["action"])
    m = dict(arm=arm, tag=tag, seed=seed, context=ctx, regret_endpoint=curve[-1], regret_curve=curve,
             curve_episodes=tr["probe_episode"].tolist(), cumulative_online_regret=float(online.sum()),
             online_regret_first_1024=float(online[:1024].sum()), online_regret_last_1024=float(online[-1024:].sum()),
             realised_survival=float(tr["realised"].mean()), training_usage=usage.tolist(),
             greedy_usage=(np.bincount(greedy, minlength=L.ACTIONS) / len(greedy)).tolist(),
             bonus_changed_fraction=float((tr["action"] != tr["base_action"]).mean()),
             bonus_changed_blocks=[float(x.mean()) for x in (tr["action"] != tr["base_action"]).reshape(-1, 512)]
             if len(tr["action"]) % 512 == 0 else [],
             online_regret_blocks=[float(x.mean()) for x in online.reshape(-1, 512)] if len(online) % 512 == 0 else [],
             disagreement={str(int(e)): float(x.mean()) for e, x in zip(tr["disagreement_episode"], tr["disagreement_member_sd"])},
             hazard_nll_endpoint=float(tr["probe_hazard_nll"][-1]),
             value_optimism_greedy=float((pred[rows, greedy] - score[rows, greedy]).mean()),
             greedy_equals_O_fraction=float((greedy == ptab["select"][:, c].argmax(1)).mean()),
             wall_seconds=rec["elapsed_seconds"], training_seconds=rec["training_seconds"], probe_seconds=rec["probe_seconds"],
             macs_train=rec["macs"]["train"], macs_act=rec["macs"]["act"], macs_probe=rec["macs"]["probe"],
             bytes_total=rec["memory_bytes"]["total"], max_replay=rec["max_replay"])
    if ctx == TRAP_CONTEXT:
        f = B1.o_best_frequency(tables, ctx)
        m["critical_usage"] = {str(a): float(usage[a]) for a in TRAP["actions"]}
        m["critical_floor"] = {str(a): float(TRAP["ratio_min"] * f[a]) for a in TRAP["actions"]}
        m["usage_ok"] = bool(all(usage[a] >= TRAP["ratio_min"] * f[a] for a in TRAP["actions"]))
        m["regret_ok"] = bool(curve[-1] <= TRAP["regret_max"])
        m["clean"] = bool(m["usage_ok"] and m["regret_ok"])
    return m


def summarize(values, idx):
    v = np.asarray(values, np.float64)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                ci95=D1.boot_mean_ci(idx, v) if idx is not None else None)


def mcnemar_exact(b, c_):
    """Two-sided exact McNemar p for discordant counts b (arm clean, MB trapped) and c_ (arm trapped, MB clean)."""
    n = b + c_
    if n == 0:
        return 1.0
    k = min(b, c_)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


# ------------------------------------------------------------------ development 2

def cmd_dev(args):
    config = load_config(args.config)
    require(config["phase"] == "dev2", "dev runs the development-2 configuration")
    output, report = Path(args.output), Path(args.report)
    identity = lock(config, args.config, output)["identity"]
    log_event(output, event="dev2_start", identity=identity)
    tasks = fit_tasks(config)
    t0 = time.perf_counter()
    run_pool(config, output, oracle_jobs(config, identity, output) + fit_jobs(config, identity, output, tasks), "dev2")
    check_lock(config, args.config, output)
    fits, oracle = verify_all(config, output, identity, tasks)
    crc = D1.crc_scan(output)
    write_json(output / "crc_scan.json", crc)
    require(crc["passed"], f"CRC scan failed: {crc['failures']}")
    write_json(output / "completion.json", dict(identity=identity, fits=fits, oracle=oracle, crc=crc, utc=now(),
                                                wall_seconds_this_invocation=time.perf_counter() - t0))
    dev2_report(config, output, report, identity, tasks, crc)


def d1_reproduction(config, output):
    """On D1's seeds, the MB arm must reproduce the stored D1 test fits (and O's stream tables) bit for bit."""
    rows = {}
    for seed in config["trap_context_seeds"]:
        ref = ROOT / f"runs/commons_d1/fits/MB__lr0.001_U2_d0.99__s{seed}__{TRAP_CONTEXT}"
        new = fit_dir(output, "MB", "base", seed, TRAP_CONTEXT)
        a, _ = load_arrays(ref / "trace.npz")
        b, _ = load_arrays(new / "trace.npz")
        ra, rb = read_json(ref / "result.json"), read_json(new / "result.json")
        arrays = all(a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes() for k in a)
        recs = all(ra[k] == rb[k] for k in ("stream_fingerprint", "decisions_chain", "sensors_chain", "outcomes_chain", "initial_hashes",
                                            "final_hashes", "checkpoint_hashes", "optimizer_steps"))
        oracle = all(load_arrays(ROOT / f"runs/commons_d1/oracle/stream_s{seed}_{TRAP_CONTEXT}_{s:05d}.npz")[1]["digest"]
                     == load_arrays(D1.oracle_path(output, "stream", seed, TRAP_CONTEXT, s))[1]["digest"]
                     for s in range(0, config["episodes"], config["oracle"]["stream_chunk"]))
        rows[str(seed)] = dict(trace_arrays_equal=bool(arrays), records_equal=bool(recs), stream_tables_equal=bool(oracle))
    panel = (load_arrays(ROOT / f"runs/commons_d1/oracle/panel_{TRAP_CONTEXT}.npz")[1]["digest"]
             == load_arrays(D1.oracle_path(output, "panel", None, TRAP_CONTEXT, None))[1]["digest"])
    return dict(seeds=rows, panel_tables_equal=bool(panel), passed=bool(panel and all(all(v.values()) for v in rows.values())))


def dev2_report(config, output, report, identity, tasks, crc):
    tables = D1.Tables(config, output)
    trap_def = check_trap_definition(tables)
    rows = [fit_metrics(config, tables, output, *t) for t in tasks]
    by = {(r["arm"], r["tag"], r["seed"]): r for r in rows}
    trapped, clean = config["analysis"]["dev2_trapped"], config["analysis"]["dev2_clean"]
    repro = d1_reproduction(config, output)
    per_setting = {}
    for arm in ARMS:
        for v in arm_values(arm):
            tag = tag_of(arm, v)
            seeds = {}
            for s in trapped + clean:
                r, mb = by[(arm, tag, s)], by[("MB", "base", s)]
                seeds[str(s)] = dict(group="trapped" if s in trapped else "clean", regret=r["regret_endpoint"],
                                     regret_ok=r["regret_ok"], usage_ok=r["usage_ok"], clean=r["clean"],
                                     critical_usage=r["critical_usage"], critical_floor=r["critical_floor"],
                                     action9_greedy_usage=r["greedy_usage"][9], cumulative_online_regret=r["cumulative_online_regret"],
                                     bonus_changed_fraction=r["bonus_changed_fraction"], MB_regret=mb["regret_endpoint"],
                                     rescued=bool(s in trapped and r["regret_ok"]), newly_trapped=bool(s in clean and not r["regret_ok"]))
            rescued = sum(x["rescued"] for x in seeds.values())
            new = sum(x["newly_trapped"] for x in seeds.values())
            per_setting[f"{arm}|{tag}"] = dict(
                arm=arm, tag=tag, value=v, rescued=rescued, newly_trapped=new, score=rescued - new,
                rescued_with_usage=sum(x["rescued"] and x["usage_ok"] for x in seeds.values() if x["group"] == "trapped"),
                clean_seeds_total=sum(x["clean"] for x in seeds.values()),
                cumulative_online_regret_mean=float(np.mean([x["cumulative_online_regret"] for x in seeds.values()])),
                regret_mean=float(np.mean([x["regret"] for x in seeds.values()])),
                disagreement={str(e): float(np.mean([by[(arm, tag, s)]["disagreement"][str(e)] for s in trapped + clean]))
                              for e in config["disagreement_episodes"]},
                hazard_nll_endpoint=float(np.mean([by[(arm, tag, s)]["hazard_nll_endpoint"] for s in trapped + clean])),
                value_optimism_greedy=float(np.mean([by[(arm, tag, s)]["value_optimism_greedy"] for s in trapped + clean])),
                wall_seconds=float(np.mean([by[(arm, tag, s)]["wall_seconds"] for s in trapped + clean])),
                per_seed=seeds)
    selection = {}
    for arm in KNOB:
        cands = [per_setting[f"{arm}|{tag_of(arm, v)}"] for v in arm_values(arm)]
        best = max(cands, key=lambda x: (x["score"], -x["cumulative_online_regret_mean"]))
        selection[arm] = dict(selected_tag=best["tag"], selected_value=best["value"],
                              table=[{k: x[k] for k in ("tag", "rescued", "newly_trapped", "score", "cumulative_online_regret_mean")}
                                     for x in cands],
                              rule="max (rescued trapped seeds with A1B1C1 regret <= .025) - (newly trapped clean seeds); "
                                   "tie -> lower mean cumulative online regret")
    out = dict(label="DEVELOPMENT 2 (trap rescue; D1 test seeds re-used as declared in Amendment 1). Not a test result. " + LABEL,
               identity=identity, trap_definition=dict(TRAP, **trap_def), d1_reproduction=repro, selection=selection,
               per_setting=per_setting, crc=crc, utc=now())
    write_json(report / "dev2_results.json", out)
    D1.write_csv(report / "dev2_seeds.csv", [dict(setting=k, seed=s, **{kk: (json.dumps(vv) if isinstance(vv, dict) else vv) for kk, vv in x.items()})
                                             for k, v in per_setting.items() for s, x in v["per_seed"].items()])
    D1.write_csv(report / "dev2_fits.csv", [{k: v for k, v in r.items() if k not in ("regret_curve", "curve_episodes")} for r in rows])
    print(json.dumps(dict(selection={a: s["selected_tag"] for a, s in selection.items()}, d1_reproduction=repro["passed"],
                          scores={k: (v["rescued"], v["newly_trapped"]) for k, v in per_setting.items()}), indent=1), flush=True)
    return out


# ------------------------------------------------------------------ test

def cmd_lock(args):
    config = load_config(args.config)
    require(config["phase"] in ("test", "smoke2"), "lock is for the smoke or test config")
    print(json.dumps(lock(config, args.config, args.output)["identity"], indent=1))


def cmd_run(args):
    config = load_config(args.config)
    output = Path(args.output)
    identity = check_lock(config, args.config, output)["identity"]
    log_event(output, event="run_start", identity=identity)
    t0 = time.perf_counter()
    tasks = fit_tasks(config)
    statuses = run_pool(config, output, oracle_jobs(config, identity, output) + fit_jobs(config, identity, output, tasks), "run")
    check_lock(config, args.config, output)
    fits, oracle = verify_all(config, output, identity, tasks)
    crc = D1.crc_scan(output)
    write_json(output / "crc_scan.json", crc)
    require(crc["passed"], f"CRC scan failed: {crc['failures']}")
    completion = dict(identity=identity, fits=fits, oracle=oracle, statuses=statuses, crc=crc, utc=now(),
                      wall_seconds_this_invocation=time.perf_counter() - t0)
    write_json(output / "completion.json", completion)
    log_event(output, event="run_complete", seconds=time.perf_counter() - t0)
    return completion


def replicate_tasks(config):
    """Test: 3 seeds x 3 arms for A1B1C1 and one other context (A0B0C1). Smoke: every fit."""
    if config["phase"] != "test":
        return fit_tasks(config)
    return [t for t in fit_tasks(config) if t[2] in REPLICATE_SEEDS and t[3] in (TRAP_CONTEXT, REPLICATE_OTHER_CONTEXT)]


def cmd_replicate(args):
    config = load_config(args.config)
    output = Path(args.output)
    identity = check_lock(config, args.config, output)["identity"]
    completion = read_json(output / "completion.json")
    for f in completion["fits"] + completion["oracle"]:
        require(sha(output / f["path"]) == f["sha256"], f"file changed since completion: {f['path']}")
    tasks = fit_tasks(config)
    verify_all(config, output, identity, tasks)
    rtasks = replicate_tasks(config)
    t0 = time.perf_counter()
    run_pool(config, output, fit_jobs(config, identity, output, rtasks, sub="replicate"), "replicate")
    comparisons, mismatches = {}, []
    for arm, tag, seed, ctx in rtasks:
        rows = D1.compare_fits(fit_dir(output, arm, tag, seed, ctx), fit_dir(output, arm, tag, seed, ctx, "replicate"))
        key = f"{arm}:{tag}:{seed}:{ctx}"
        comparisons[key] = rows
        mismatches += [f"{key}:{k}" for k, v in rows.items() if not v]
    crc = D1.crc_scan(output)
    rep = dict(identity=identity, utc=now(), fits_compared=len(rtasks), main_fits_reverified=len(tasks), comparisons=comparisons,
               mismatches=mismatches, crc=crc, passed=not mismatches and crc["passed"], seconds=time.perf_counter() - t0)
    write_json(output / "replicate.json", rep)
    log_event(output, event="replicate_complete", passed=rep["passed"], mismatches=mismatches)
    if not rep["passed"]:
        raise RuntimeError(f"REPLICATE MISMATCH OR CRC FAILURE, do not analyze: {mismatches} {crc['failures']}")
    return rep


def cmd_analyze(args, require_replicate=True):
    config = load_config(args.config)
    output, report = Path(args.output), Path(args.report)
    identity = check_lock(config, args.config, output)["identity"]
    if require_replicate:
        rep = read_json(output / "replicate.json")
        require(rep["passed"] and rep["identity"] == identity, "the replicate pass must pass before analysis")
    tasks = fit_tasks(config)
    verify_all(config, output, identity, tasks)
    crc = D1.crc_scan(output)
    require(crc["passed"], "CRC scan failed")
    tables = D1.Tables(config, output)
    trap_def = check_trap_definition(tables)
    rows = [fit_metrics(config, tables, output, *t) for t in tasks]
    a = config["analysis"]
    arms = {arm: tag_of(arm, config["knobs"].get(arm)) for arm in ARMS}
    by = {(r["arm"], r["seed"], r["context"]): r for r in rows}
    r2_seeds, r1_seeds, ctxs = config["trap_context_seeds"], config["other_context_seeds"], config["contexts"]
    B = a.get("bootstrap_resamples", 2000)
    rng = np.random.default_rng(a.get("analysis_seed", 0))
    idx2 = rng.integers(0, len(r2_seeds), (B, len(r2_seeds))) if len(r2_seeds) > 1 else None
    idx1 = rng.integers(0, len(r1_seeds), (B, len(r1_seeds))) if len(r1_seeds) > 1 else None
    need2 = a["r2_seeds_required"] if config["phase"] == "test" else len(r2_seeds)
    need1 = a["r1_seeds_required"] if config["phase"] == "test" else len(r1_seeds)
    # R2: the trap rate on A1B1C1 (primary)
    r2 = {}
    for arm in ARMS:
        cl = [by[(arm, s, TRAP_CONTEXT)]["clean"] for s in r2_seeds]
        r2[arm] = dict(clean_seeds=int(sum(cl)), required=need2, escapes=bool(sum(cl) >= need2),
                       regret_ok=int(sum(by[(arm, s, TRAP_CONTEXT)]["regret_ok"] for s in r2_seeds)),
                       usage_ok=int(sum(by[(arm, s, TRAP_CONTEXT)]["usage_ok"] for s in r2_seeds)),
                       trap_rate=summarize([0.0 if x else 1.0 for x in cl], idx2),
                       per_seed={str(s): dict(regret=by[(arm, s, TRAP_CONTEXT)]["regret_endpoint"], clean=by[(arm, s, TRAP_CONTEXT)]["clean"],
                                              usage=by[(arm, s, TRAP_CONTEXT)]["critical_usage"]) for s in r2_seeds})
    for arm in ARMS:
        if arm == "MB":
            continue
        x = np.array([by[(arm, s, TRAP_CONTEXT)]["clean"] for s in r2_seeds])
        m = np.array([by[("MB", s, TRAP_CONTEXT)]["clean"] for s in r2_seeds])
        b_, c_ = int((x & ~m).sum()), int((~x & m).sum())
        r2[arm]["paired_vs_MB"] = dict(both_clean=int((x & m).sum()), arm_clean_MB_trapped=b_, arm_trapped_MB_clean=c_,
                                       both_trapped=int((~x & ~m).sum()),
                                       trap_rate_difference=summarize((~x).astype(float) - (~m).astype(float), idx2),
                                       mcnemar_exact_p=mcnemar_exact(b_, c_))
    # R1: competence on the R1 seeds (5 contexts; A1B1C1 from the first 12 trap-context seeds)
    r1, per1 = {}, {}
    for arm in ARMS:
        ps = {}
        for s in r1_seeds:
            vals = [by[(arm, s, c)]["regret_endpoint"] for c in ctxs]
            ps[str(s)] = dict(context_average=float(np.mean(vals)), worst_context=float(max(vals)), worst_context_name=ctxs[int(np.argmax(vals))],
                              mean_ok=bool(np.mean(vals) <= a["competent_mean"]), worst_ok=bool(max(vals) <= a["competent_worst"]),
                              cumulative_online_regret=float(np.sum([by[(arm, s, c)]["cumulative_online_regret"] for c in ctxs])))
        both = sum(v["mean_ok"] and v["worst_ok"] for v in ps.values())
        per1[arm] = ps
        r1[arm] = dict(seeds_meeting_both=int(both), seeds_meeting_mean=int(sum(v["mean_ok"] for v in ps.values())),
                       seeds_meeting_worst=int(sum(v["worst_ok"] for v in ps.values())), required=need1, competent=bool(both >= need1),
                       context_average=summarize([v["context_average"] for v in ps.values()], idx1),
                       worst_context=summarize([v["worst_context"] for v in ps.values()], idx1),
                       by_context={c: summarize([by[(arm, s, c)]["regret_endpoint"] for s in r1_seeds], idx1) for c in ctxs},
                       per_seed=ps)
    r3 = {}
    for arm in ARMS:
        if arm == "MB":
            continue
        d = [per1[arm][str(s)]["cumulative_online_regret"] - per1["MB"][str(s)]["cumulative_online_regret"] for s in r1_seeds]
        k = sum(x <= 0 for x in d)
        per_ctx = {}
        for c in ctxs:
            dc = [by[(arm, s, c)]["cumulative_online_regret"] - by[("MB", s, c)]["cumulative_online_regret"] for s in r1_seeds]
            per_ctx[c] = dict(seeds_at_or_below_MB=int(sum(x <= 0 for x in dc)), net_cost_vs_MB=summarize(dc, idx1))
        r3[arm] = dict(seeds_at_or_below_MB=int(k), required=need1, pays_back=bool(k >= need1), applies=bool(r1[arm]["competent"]),
                       net_cost_vs_MB=summarize(d, idx1), by_context=per_ctx,
                       rule="overall (declared): 5-context cumulative online regret <= MB's in >= 10/12 R1 seeds; per context descriptive")
    competent = [arm for arm in ARMS if r1[arm]["competent"]]
    core = (min(competent, key=lambda x: np.mean([per1[x][str(s)]["cumulative_online_regret"] for s in r1_seeds]))
            if competent else None)
    remedies_competent = [arm for arm in competent if arm != "MB"]
    escaping = [arm for arm in ARMS if arm != "MB" and r2[arm]["escapes"]]
    if remedies_competent:
        implication = f"Competent remedy(ies) {remedies_competent}; D2 core = {core} (lowest mean cumulative online regret among competent arms)."
    elif escaping:
        implication = f"{escaping} escape the trap but miss R1: declared D1c round on replay regularisation."
    else:
        implication = "No remedy escapes the trap: report and decide with the user (VOI signal or count-based bonuses)."
    usage_profile = {arm: dict(training=np.mean([by[(arm, s, TRAP_CONTEXT)]["training_usage"] for s in r2_seeds], 0).tolist(),
                               greedy=np.mean([by[(arm, s, TRAP_CONTEXT)]["greedy_usage"] for s in r2_seeds], 0).tolist())
                     for arm in ARMS}
    descriptive = {arm: dict(curve=np.mean([by[(arm, s, c)]["regret_curve"] for s in r1_seeds for c in ctxs], 0).tolist(),
                             curve_A1B1C1_24=np.mean([by[(arm, s, TRAP_CONTEXT)]["regret_curve"] for s in r2_seeds], 0).tolist(),
                             disagreement={str(e): float(np.mean([r["disagreement"][str(e)] for r in rows if r["arm"] == arm]))
                                           for e in config["disagreement_episodes"]},
                             hazard_nll=float(np.mean([r["hazard_nll_endpoint"] for r in rows if r["arm"] == arm])),
                             value_optimism=float(np.mean([r["value_optimism_greedy"] for r in rows if r["arm"] == arm])),
                             bonus_changed_fraction=float(np.mean([r["bonus_changed_fraction"] for r in rows if r["arm"] == arm])),
                             bonus_changed_by_512_block=np.mean([r["bonus_changed_blocks"] for r in rows if r["arm"] == arm], 0).tolist(),
                             online_regret_by_512_block=np.mean([r["online_regret_blocks"] for r in rows if r["arm"] == arm], 0).tolist(),
                             wall_seconds=float(np.mean([r["wall_seconds"] for r in rows if r["arm"] == arm])),
                             macs_total=float(np.mean([r["macs_train"] + r["macs_act"] + r["macs_probe"] for r in rows if r["arm"] == arm])),
                             bytes_total=float(np.mean([r["bytes_total"] for r in rows if r["arm"] == arm])))
                   for arm in ARMS}
    out = dict(label=LABEL + (" SMOKE ONLY - NOT SCIENCE." if config["phase"] == "smoke2" else ""), identity=identity, knobs=config["knobs"],
               bootstrap=dict(resamples=B, analysis_seed=a.get("analysis_seed")), R2=r2, R1=r1, R3=r3, competent_arms=competent,
               d2_core=core, escaping_remedies=escaping, implication=implication, usage_profile_A1B1C1=usage_profile,
               o_best_frequency_A1B1C1=trap_def["o_best_frequency"], descriptive=descriptive, curve_episodes=rows[0]["curve_episodes"],
               trap_definition=dict(TRAP, **trap_def), crc=crc, fits=len(rows), utc=now(),
               design_note=("R1 and R2 share cells: R1's A1B1C1 values are the first 12 of R2's 24 A1B1C1 seeds (as intended by "
                            "Amendment 1), so R1 and R2 are not independent."))
    write_json(report / "results.json", out)
    plot_results(out, config, report / "d1b_results.png")
    D1.write_csv(report / "fits.csv", [{k: v for k, v in r.items() if k not in ("regret_curve", "curve_episodes")} for r in rows])
    print(json.dumps(dict(R2={k: v["clean_seeds"] for k, v in r2.items()}, R1={k: v["seeds_meeting_both"] for k, v in r1.items()},
                          implication=implication), indent=1), flush=True)
    return out


def plot_results(out, config, path):
    """Left: greedy-probe regret vs episodes (R1 cells), bar .015. Right: A1B1C1 regret per seed (24), bar .025, clean counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    color = {"MB": "#2a78d6", "MB-prior": "#eb6834", "MB-count": "#1baf7a"}   # categorical slots 1-3
    marker = {"MB": "o", "MB-prior": "s", "MB-count": "^"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    fig.patch.set_facecolor(surface)
    for a_ in (ax, bx):
        a_.set_facecolor(surface)
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a_.spines[side].set_color(muted)
        a_.tick_params(colors=muted, labelsize=9)
        a_.grid(True, axis="y", color=grid, linewidth=0.8)
        a_.set_axisbelow(True)
    eps = np.array(out["curve_episodes"][1:])
    ends = []
    for arm in ARMS:
        m = np.array(out["descriptive"][arm]["curve"][1:])
        ax.plot(eps, m, "-", color=color[arm], marker=marker[arm], markersize=5, linewidth=2, label=arm)
        ends.append([float(np.log10(max(m[-1], 1e-6))), arm, m[-1]])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06)
    for y, arm, v in ends:
        ax.annotate(f"{arm} {v:.4f}", (eps[-1], v), xytext=(eps[-1] * 1.02, 10 ** y), textcoords="data", va="center", fontsize=8, color=ink)
    bar = config["analysis"]["competent_mean"]
    ax.axhline(bar, color=muted, linestyle=":", linewidth=1.2)
    ax.set_yscale("log")
    ticks = [0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.5]
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ticks if lo <= t <= hi], [f"{t:g}" for t in ticks if lo <= t <= hi])
    ax.minorticks_off()
    ax.set_xlim(0, eps[-1] * 1.3)
    ax.set_xticks([0, 1024, 2048, 3072, 4096] if eps[-1] == 4096 else [0, eps[-1]])
    ax.annotate(f"competence bar {bar:g}", (0, bar), xytext=(4, 4), textcoords="offset points", fontsize=8, color=muted)
    ax.set_xlabel("episodes of experience (stationary context)", color=ink, fontsize=9)
    ax.set_ylabel("greedy-probe regret vs O (survival)", color=ink, fontsize=9)
    ax.set_title("R1 cells: mean regret over seeds x 5 contexts", fontsize=10, color=ink, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.06))
    trap = config["analysis"]["trap"]["regret_max"]
    for i, arm in enumerate(ARMS):
        ps = out["R2"][arm]["per_seed"]
        vals = np.array([ps[s]["regret"] for s in ps])
        clean = np.array([ps[s]["clean"] for s in ps])
        jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2) * (0.5 / max(len(vals), 1))
        bx.scatter(i + jitter[clean], vals[clean], s=26, facecolor=color[arm], edgecolor=color[arm], marker=marker[arm], zorder=3)
        bx.scatter(i + jitter[~clean], vals[~clean], s=26, facecolor=surface, edgecolor=color[arm], marker=marker[arm], linewidth=1.5, zorder=3)
        bx.annotate(f"{int(clean.sum())}/{len(vals)} clean", (i, vals.max()), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=8, color=ink)
    bx.axhline(trap, color=muted, linestyle=":", linewidth=1.2)
    bx.annotate(f"trap threshold {trap:g}", (-0.45, trap), xytext=(0, 4), textcoords="offset points", fontsize=8, color=muted)
    bx.set_xticks(range(len(ARMS)), list(ARMS), fontsize=8, color=ink)
    bx.set_ylabel("A1B1C1 greedy-probe regret at 4,096", color=ink, fontsize=9)
    bx.set_title("R2: A1B1C1 per seed (open = trapped:\nregret > threshold or usage condition failed)", fontsize=10, color=ink, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# ------------------------------------------------------------------ smoke checks

def import_check():
    tree = ast.parse((ROOT / "src/acp_cl/commons/learners_count.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {x.name for x in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add("." * node.level + (node.module or ""))
    prev = B1.import_check()
    return dict(learners_count_imports=sorted(mods), earlier=prev,
                passed=bool(mods <= {"__future__", "numpy", "torch", ".learners"} and prev["passed"]))


def count_unit_check(config):
    """MB-count: the bonus is exactly c/sqrt(1+n_a) with n_a counted over the CURRENT reservoir (after evictions)."""
    params, _ = E.load_world(ROOT)
    st = E.Stream(params, 7, SMOKE_SEED, 80)
    lrn, _ = make_arm_learner(config, "MB-count", 0.15, SMOKE_SEED, 7, capacity=16)
    track, manual_ok, differs_after_eviction = True, True, False
    cumulative = np.zeros(L.ACTIONS, np.int64)
    for k in range(80):
        dec = st.decision(k)
        plan_state = copy.deepcopy(lrn.rngs["plan"].bit_generator.state)
        th_state = copy.deepcopy(lrn.rngs["thompson"].bit_generator.state)
        a = lrn.act(dec)
        counts = lrn.reservoir_counts()
        track &= bool(np.array_equal(lrn.last_counts, counts))
        # manual recomputation of the bonus-adjusted argmax
        r_th = np.random.default_rng(0)
        r_th.bit_generator.state = th_state
        member = int(r_th.integers(0, lrn.h.members))
        r_pl = np.random.default_rng(0)
        r_pl.bit_generator.state = plan_state
        eps = torch.from_numpy(r_pl.standard_normal((lrn.h.plan_samples, 12, 3)).astype(np.float32))
        with torch.no_grad():
            y = lrn.supply_draws(eps, lrn.ema[1])[None]
            v = torch.exp(lrn._values(lrn.ema[0], member, dec.reserves[None], dec.cue[None], y).sum(-1)).mean(-1)[0].numpy()
        manual = int(np.argmax(v + (0.15 / np.sqrt(1.0 + counts)).astype(np.float32)))
        manual_ok &= manual == a and member == lrn.last_thompson and lrn.last_base_action == int(np.argmax(v))
        act = int(k % L.ACTIONS)                    # force diverse actions so evictions change the counts
        lrn.observe(dec, act, st.outcome(k, act))
        cumulative[act] += 1
        if lrn.replay.seen > 16:
            differs_after_eviction |= not np.array_equal(lrn.reservoir_counts(), cumulative)
        require(lrn.reservoir_counts().sum() == lrn.replay.size, "counts must sum to the reservoir size")
    return dict(counts_equal_reservoir_bincount=bool(track), bonus_argmax_matches_manual=bool(manual_ok),
                reservoir_counts_differ_from_cumulative_after_eviction=bool(differs_after_eviction),
                evictions=lrn.replay.evictions, passed=bool(track and manual_ok and differs_after_eviction and lrn.replay.evictions > 0))


def c0_identity_check(config, identity, output):
    """MB-count with c = 0 is bit-identical to MB (fit level, same seed and context)."""
    base = fresh(output / "checks" / "c0")
    cfg = dict(config, episodes=128, checkpoint_every=64, probe_episodes=[64, 128], disagreement_episodes=[128])
    run_fit(cfg, identity, base, "MB", "base", SMOKE_SEED, TRAP_CONTEXT)
    saved = GRID["MB-count"]["c"]
    GRID["MB-count"]["c"] = saved + [0.0]            # c = 0 is not a grid value; admitted for this check only
    try:
        run_fit(cfg, identity, base, "MB-count", "c0", SMOKE_SEED, TRAP_CONTEXT)
    finally:
        GRID["MB-count"]["c"] = saved
    a, _ = load_arrays(fit_dir(base, "MB", "base", SMOKE_SEED, TRAP_CONTEXT) / "trace.npz")
    b, _ = load_arrays(fit_dir(base, "MB-count", "c0", SMOKE_SEED, TRAP_CONTEXT) / "trace.npz")
    ra = read_json(fit_dir(base, "MB", "base", SMOKE_SEED, TRAP_CONTEXT) / "result.json")
    rb = read_json(fit_dir(base, "MB-count", "c0", SMOKE_SEED, TRAP_CONTEXT) / "result.json")
    keys = [k for k in a if k != "count_of_action"]
    same = {k: bool(a[k].tobytes() == b[k].tobytes()) for k in keys}
    return dict(arrays_equal=same, final_hashes_equal=ra["final_hashes"] == rb["final_hashes"],
                chains_equal=all(ra[k] == rb[k] for k in ("decisions_chain", "sensors_chain", "outcomes_chain")),
                passed=bool(all(same.values()) and ra["final_hashes"] == rb["final_hashes"]))


def trace_check(config, output, tasks):
    rows, ok = {}, True
    for arm, tag, seed, ctx in tasks:
        tr, _ = load_arrays(fit_dir(output, arm, tag, seed, ctx) / "trace.npz")
        changed = float((tr["action"] != tr["base_action"]).mean())
        good = (changed > 0) if arm == "MB-count" else (changed == 0)
        good &= bool((tr["count_of_action"] >= 0).all()) if arm == "MB-count" else bool((tr["count_of_action"] == -1).all())
        good &= list(tr["disagreement_episode"]) == config["disagreement_episodes"]
        rows[f"{arm}|{ctx}"] = dict(bonus_changed_fraction=changed, passed=bool(good))
        ok &= good
    return dict(fits=rows, passed=bool(ok))


def training_check(config, identity, output):
    """With actions forced to MB's, MB-count trains bit-identically to MB (the bonus acts only on acting)."""
    base = fresh(output / "checks" / "training")
    cfg = dict(config, episodes=128, checkpoint_every=64, probe_episodes=[64, 128], disagreement_episodes=[128])
    run_fit(cfg, identity, base, "MB", "base", SMOKE_SEED, "A0B0C1")
    tr, _ = load_arrays(fit_dir(base, "MB", "base", SMOKE_SEED, "A0B0C1") / "trace.npz")
    run_fit(cfg, identity, base, "MB-count", "c0.15", SMOKE_SEED, "A0B0C1", forced_actions=tr["action"].astype(np.int64))
    h1 = read_json(fit_dir(base, "MB", "base", SMOKE_SEED, "A0B0C1") / "result.json")["final_hashes"]
    h2 = read_json(fit_dir(base, "MB-count", "c0.15", SMOKE_SEED, "A0B0C1") / "result.json")["final_hashes"]
    return dict(online_identical=h1["online"] == h2["online"], optimizer_identical=h1["optimizer"] == h2["optimizer"],
                passed=bool(h1["online"] == h2["online"] and h1["optimizer"] == h2["optimizer"]))


def eviction_check(config, identity, output):
    base = fresh(output / "checks" / "eviction")
    cfg = dict(config, episodes=128, checkpoint_every=64, probe_episodes=[64, 128], disagreement_episodes=[128])
    run_fit(cfg, identity, base, "MB-count", "c0.15", SMOKE_SEED, "A0B1C0", capacity=64)
    rec = read_json(fit_dir(base, "MB-count", "c0.15", SMOKE_SEED, "A0B1C0") / "result.json")
    tr, _ = load_arrays(fit_dir(base, "MB-count", "c0.15", SMOKE_SEED, "A0B1C0") / "trace.npz")
    return dict(max_replay=rec["max_replay"], evictions=rec["replay"]["evictions"],
                max_count_seen=int(tr["count_of_action"].max()),
                passed=bool(rec["max_replay"] == 64 and rec["replay"]["evictions"] > 0 and tr["count_of_action"].max() <= 64))


def resume_check(config, identity, output, tasks):
    base = fresh(output / "checks" / "resume")
    out = {}
    for arm in ("MB-count", "MB-prior"):
        _, tag, seed, ctx = [t for t in tasks if t[0] == arm][0]
        try:
            run_fit(config, identity, base, arm, tag, seed, ctx, interrupt_after=100)
        except D1.Interrupted:
            pass
        side = read_json(fit_dir(base, arm, tag, seed, ctx) / "checkpoint.sha256.json")
        run_fit(config, identity, base, arm, tag, seed, ctx)
        rec = read_json(fit_dir(base, arm, tag, seed, ctx) / "result.json")
        rows = D1.compare_fits(fit_dir(output, arm, tag, seed, ctx), fit_dir(base, arm, tag, seed, ctx))
        out[arm] = dict(checkpoint_at_interrupt=side["episodes_done"], resumed=rec["resumed"], comparison=rows,
                        passed=bool(rec["resumed"] and side["episodes_done"] == 64 and all(rows.values())))
    out["passed"] = all(v["passed"] for v in out.values())
    return out


def d1_identity_check(identity, output):
    ref = ROOT / "runs/commons_d1/fits/MB__lr0.001_U2_d0.99__s21110__A1B1C1"
    cfg = dict(read_json(ROOT / "configs/commons_d1b_dev2.json"))
    base = fresh(output / "checks" / "d1_identity")
    run_fit(cfg, identity, base, "MB", "base", 21110, TRAP_CONTEXT)
    a, _ = load_arrays(ref / "trace.npz")
    b, _ = load_arrays(fit_dir(base, "MB", "base", 21110, TRAP_CONTEXT) / "trace.npz")
    ra, rb = read_json(ref / "result.json"), read_json(fit_dir(base, "MB", "base", 21110, TRAP_CONTEXT) / "result.json")
    arrays = {k: bool(a[k].tobytes() == b[k].tobytes()) for k in a}
    recs = all(ra[k] == rb[k] for k in ("stream_fingerprint", "decisions_chain", "sensors_chain", "outcomes_chain", "initial_hashes",
                                        "final_hashes", "checkpoint_hashes", "optimizer_steps"))
    return dict(reference=ref.relative_to(ROOT).as_posix(), arrays=arrays, records_equal=bool(recs),
                passed=bool(all(arrays.values()) and recs))


def cmd_check(args):
    config = load_config(args.config)
    require(config["phase"] == "smoke2", "check runs the smoke-2 configuration")
    output = Path(args.output)
    fresh(output)
    D1.worker_init(1)
    t0 = time.perf_counter()
    checks = dict(label="ENGINEERING CHECKS, D1b Amendment 1 smoke (seed 21299, 256 episodes). Not science.", utc=now())
    params, cfg = E.load_world(ROOT)
    checks["world"] = dict(passed=E.cfg_key(cfg) == E.WORLD_KEY)
    checks["imports"] = import_check()
    checks["legal_observations"] = dict(passed=D1.legal_observation_check(params)["passed"])
    bad = L.make_decision(np.ones(3), np.zeros(3))
    object.__setattr__(bad, "context", 7)
    lrn, _ = make_arm_learner(config, "MB-count", 0.15, SMOKE_SEED, 7)
    try:
        lrn.act(bad)
        checks["legal_observations"]["count_rejects_illegal"] = False
    except L.IllegalObservation:
        checks["legal_observations"]["count_rejects_illegal"] = True
    checks["legal_observations"]["passed"] &= checks["legal_observations"]["count_rejects_illegal"]
    checks["reservoir_uniform_and_bounded"] = D1.reservoir_check()
    prior = B1.remedy_unit_check(config)["prior"]
    checks["prior_active_only_where_declared"] = prior
    checks["count_bonus_and_reservoir_counts"] = count_unit_check(config)
    print("unit", {k: v.get("passed") for k, v in checks.items() if isinstance(v, dict)}, flush=True)
    identity = lock(config, args.config, output)["identity"]
    a = argparse.Namespace(config=args.config, output=str(output), report=str(output / "analysis"))
    cmd_run(a)
    tasks = fit_tasks(config)
    second = cmd_run(a)
    checks["rerun_only_verifies"] = dict(passed=all(s["status"] == "verified_existing" for s in second["statuses"]))
    tables = D1.Tables(config, output)
    flat = dict(config, seeds=[SMOKE_SEED])
    checks["o_table_lookups"] = D1.o_lookup_check(flat, output, tables, params)
    checks["common_random_numbers"] = D1.crn_check(flat, output, tasks)
    checks["bounded_replay_in_run"] = dict(passed=all(read_json(fit_dir(output, *t) / "result.json")["max_replay"] <= 1024 for t in tasks))
    checks["remedy_traces"] = trace_check(config, output, tasks)
    checks["count_c0_bit_identical_to_MB"] = c0_identity_check(config, identity, output)
    checks["count_training_unchanged"] = training_check(config, identity, output)
    checks["count_eviction_path"] = eviction_check(config, identity, output)
    checks["determinism_and_resumption"] = resume_check(config, identity, output, tasks)
    checks["MB_arm_bit_identical_to_D1"] = d1_identity_check(identity, output)
    print("fit", {k: v.get("passed") for k, v in checks.items() if isinstance(v, dict)}, flush=True)
    rep = cmd_replicate(a)
    checks["replicate_pass"] = dict(fits_compared=rep["fits_compared"], mismatches=rep["mismatches"], passed=rep["passed"])
    victim = fit_dir(output, *tasks[0]) / "trace.npz"
    backup = victim.read_bytes()
    raw = bytearray(backup)
    raw[len(raw) // 3] ^= 0xFF
    victim.write_bytes(bytes(raw))
    try:
        D1.verify_result(victim.parent, identity)
        detected = False
    except ValueError:
        detected = True
    crc_bad = D1.crc_scan(output)
    victim.write_bytes(backup)
    checks["corruption_detected"] = dict(passed=bool(detected and not crc_bad["passed"]))
    res = cmd_analyze(a)
    checks["analysis_pipeline"] = dict(implication_smoke_only=res["implication"], passed=True)
    checks["all_passed"] = bool(all(v["passed"] for v in checks.values() if isinstance(v, dict) and "passed" in v))
    checks["seconds"] = time.perf_counter() - t0
    write_json(output / "checks" / "checks.json", checks)
    print("ALL PASS" if checks["all_passed"] else "FAILURES: " + ", ".join(k for k, v in checks.items()
                                                                          if isinstance(v, dict) and v.get("passed") is False), flush=True)
    if not checks["all_passed"]:
        raise SystemExit(1)


DEFAULTS = {"check": ("configs/commons_d1b_smoke2.json", "runs/commons_d1b_smoke2", "runs/commons_d1b_smoke2/analysis"),
            "dev": ("configs/commons_d1b_dev2.json", "runs/commons_d1b_dev2", "reports/commons_d1b/dev2"),
            "lock": ("configs/commons_d1b.json", "runs/commons_d1b", "reports/commons_d1b"),
            "run": ("configs/commons_d1b.json", "runs/commons_d1b", "reports/commons_d1b"),
            "replicate": ("configs/commons_d1b.json", "runs/commons_d1b", "reports/commons_d1b"),
            "analyze": ("configs/commons_d1b.json", "runs/commons_d1b", "reports/commons_d1b")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(DEFAULTS))
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--report")
    args = parser.parse_args()
    d = DEFAULTS[args.command]
    args.config, args.output, args.report = (str(ROOT / (v or dflt)) for v, dflt in
                                             ((args.config, d[0]), (args.output, d[1]), (args.report, d[2])))
    {"check": cmd_check, "dev": cmd_dev, "lock": cmd_lock, "run": cmd_run, "replicate": cmd_replicate,
     "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
