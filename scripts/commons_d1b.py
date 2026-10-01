"""Step D1b: exploration remedies for the model-based learner (docs/commons_d1b_protocol.md).

Reuses scripts/commons_d1.py by import (oracle tasks, tables, hashing, CRC, checkpoints and several checks)
and D1's learners (``acp_cl.commons.learners``; the MB arm IS D1's MBLearner with D1's randomness keys).
Remedies live in ``acp_cl.commons.learners_d1b``. Nothing of D1 is edited.

Commands (set PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src):
  .venv/Scripts/python.exe -B scripts/commons_d1b.py check     (configs/commons_d1b_smoke.json -> runs/commons_d1b_smoke)
  .venv/Scripts/python.exe -B scripts/commons_d1b.py dev       (configs/commons_d1b_dev.json   -> runs/commons_d1b_dev, reports/commons_d1b/dev)
  .venv/Scripts/python.exe -B scripts/commons_d1b.py lock | run | replicate | analyze   (configs/commons_d1b.json -> runs/commons_d1b)
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zipfile  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import commons_d1 as D1  # noqa: E402
from acp_cl.commons import d1_env as E  # noqa: E402
from acp_cl.commons import learners as L  # noqa: E402
from acp_cl.commons import learners_d1b as LB  # noqa: E402
from acp_cl.commons.world import SURVIVED, seed_for  # noqa: E402

sha, digest, read_json, write_json, now, require, log_event = D1.sha, D1.digest, D1.read_json, D1.write_json, D1.now, D1.require, D1.log_event
save_arrays, load_arrays, quarantine, fresh, chain = D1.save_arrays, D1.load_arrays, D1.quarantine, D1.fresh, D1.chain

LABEL = ("PROSPECTIVE STUDY under docs/commons_d1b_protocol.md (Step D1b). O is evaluator-only. "
         "All arms are bounded (1,024-episode reservoir).")
SCRIPT_FILES = ("scripts/commons_d1.py", "scripts/commons_d1b.py")
ARMS = ("MB", "MB-prior", "MB-UCB", "MB-eps", "MB-wide")
KNOB = {"MB-prior": "beta", "MB-UCB": "kappa", "MB-eps": "epsilon"}
GRID = {"MB-prior": {"beta": [1, 3]}, "MB-UCB": {"kappa": [1, 2]}, "MB-eps": {"epsilon": [0.03, 0.1]}, "MB-wide": {"hidden": 256}}
BASE_KNOBS = {"lr": 0.001, "U": 2, "ema_decay": 0.99}
DEV_SEEDS = [21201, 21202, 21203]
TEST_SEEDS = list(range(21301, 21313))
SMOKE_SEED = 21299
REPLICATE_SEEDS = (21301, 21306, 21311)
PROBES = [256, 512, 1024, 2048, 3072, 4096]
TRAP = dict(context="A1B1C1", regret_max=0.025, actions=[6, 9, 12], ratio_min=0.5, ratio_max=2.0, critical_loss=0.015,
            definition=("Decision-critical non-null actions in the trap context A1B1C1: the non-null actions whose removal from "
                        "O's action set costs >= .015 expected survival on the A1B1C1 panel (O's choice by the selection table, "
                        "values by the scoring table): {6: 1->3x3, 9: 2->3x6, 12: 1->3x6}. f_O(a) = fraction of the 512 A1B1C1 panel "
                        "states in which a is O's selected action. A seed meets the usage condition if, for every a in the set, its "
                        "training usage u(a) = (episodes of the 4,096 in A1B1C1 in which it took a) / 4,096 satisfies "
                        "0.5 f_O(a) <= u(a) <= 2 f_O(a). A remedy escapes the trap if >= 10/12 seeds have A1B1C1 regret <= .025 AND "
                        "meet the usage condition (both in the same seed)."))


# ------------------------------------------------------------------ config

def arm_values(arm):
    if arm in KNOB:
        return list(GRID[arm][KNOB[arm]])
    return [None]


def tag_of(arm, value):
    if arm == "MB":
        return "base"
    if arm == "MB-wide":
        return "w256"
    return f"{KNOB[arm]}{value:g}"


def value_of(arm, tag):
    for v in arm_values(arm):
        if tag_of(arm, v) == tag:
            return v
    raise ValueError(f"{arm} {tag}")


def validate_config(c):
    expected = {"study", "protocol", "phase", "world", "contexts", "seeds", "episodes", "checkpoint_every", "probe_episodes",
                "panel_size", "oracle", "learner", "base_knobs", "d1_selection", "grid", "knobs", "arms", "disagreement_states",
                "analysis", "workers", "threads", "device", "stall_seconds", "stage_attempts", "notes"}
    if set(c) != expected:
        raise ValueError(f"invalid configuration fields: {sorted(set(c) ^ expected)}")
    require(c["protocol"] == "docs/commons_d1b_protocol.md" and c["phase"] in ("smoke", "dev", "test"), "protocol/phase")
    require(c["world"] == dict(separability="reports/commons_d0b/separability.json", spec="reports/commons_d0b/world_spec.json",
                               key=E.WORLD_KEY), "the D0b world")
    require(c["contexts"] == list(E.TRAINING_CONTEXTS), "the 5 training contexts")
    require(c["learner"] == D1.FIXED_LEARNER and c["base_knobs"] == BASE_KNOBS, "D1's MB with its selected knobs")
    dv = c["d1_selection"]
    require(dv["path"] == "reports/commons_d1/dev/dev_results.json" and sha(ROOT / dv["path"]) == dv["sha256"], "D1 selection file hash")
    require(read_json(ROOT / dv["path"])["selection"]["MB"]["selected"] == BASE_KNOBS, "base knobs = D1's MB selection")
    require(c["grid"] == GRID and c["arms"] == list(ARMS), "arms and development grid are fixed by the protocol")
    o = c["oracle"]
    require(set(o) == {"draws", "stream_chunk", "floor_draws"} and o["draws"] == 2 ** 14 and o["floor_draws"] == 256, "O as in D1")
    require(c["device"] == "cpu" and c["threads"] == 1 and 1 <= c["workers"] <= 6, "CPU, 1 thread, <= 6 workers")
    require(c["episodes"] % c["checkpoint_every"] == 0 and c["episodes"] % o["stream_chunk"] == 0, "tiling")
    a = dict(c["analysis"])
    if c["phase"] == "test":
        boot = {k: a.pop(k, None) for k in ("bootstrap_resamples", "analysis_seed")}
        require(boot["bootstrap_resamples"] == 20000 and type(boot["analysis_seed"]) is int, "declared bootstrap")
    require(a == dict(competent_mean=0.015, competent_worst=0.025, consistency_seeds=10, endpoint=c["episodes"], trap=TRAP,
                      payback_reference="MB"), "pre-declared analysis constants")
    ph = c["phase"]
    if ph == "smoke":
        require(c["seeds"] == [SMOKE_SEED] and c["episodes"] == 256, "smoke: seed 21299, 256 episodes")
    elif ph == "dev":
        require(c["seeds"] == DEV_SEEDS and c["episodes"] == 4096 and c["probe_episodes"] == PROBES and c["panel_size"] == 512
                and c["knobs"] is None, "dev: seeds 21201-21203, N 4096, amended probes, whole grid")
    else:
        require(c["seeds"] == TEST_SEEDS and c["episodes"] == 4096 and c["probe_episodes"] == PROBES and c["panel_size"] == 512,
                "test: seeds 21301-21312")
    if ph in ("smoke", "test"):
        require(isinstance(c["knobs"], dict) and set(c["knobs"]) == set(KNOB), "one knob per remedy")
        for arm, v in c["knobs"].items():
            if v is None:
                raise ValueError(f"knob for {arm} is unfilled (fill from the development selection before locking)")
            require(v in arm_values(arm), f"{arm} knob must be a grid value")


def load_config(path):
    c = read_json(path)
    validate_config(c)
    return c


def fit_tasks(config, arms=None):
    tasks = []
    for arm in arms or config["arms"]:
        values = arm_values(arm) if config["phase"] == "dev" or arm not in KNOB else [config["knobs"][arm]]
        for v in values:
            for seed in config["seeds"]:
                for ctx in config["contexts"]:
                    tasks.append((arm, tag_of(arm, v), seed, ctx))
    return tasks


def fit_dir(output, arm, tag, seed, ctx, sub="fits"):
    return Path(output) / sub / f"{arm}__{tag}__s{seed}__{ctx}"


# ------------------------------------------------------------------ lock

def source_manifest():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "src/acp_cl/commons").glob("*.py"))}


def lock_payload(config, config_path):
    scripts = {name: sha(ROOT / name) for name in SCRIPT_FILES}
    source = source_manifest()
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                   device=config["device"], threads=config["threads"], workers=config["workers"], deterministic=True)
    identity = dict(config_sha256=digest(config), config_file_sha256=sha(config_path), protocol_sha256=sha(ROOT / config["protocol"]),
                    scripts_sha256=digest(scripts), source_sha256=digest(source), runtime_sha256=digest(runtime),
                    world_spec_sha256=sha(ROOT / config["world"]["spec"]), d1_selection_sha256=config["d1_selection"]["sha256"])
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
    hyper = L.Hyper(lr=BASE_KNOBS["lr"], U=BASE_KNOBS["U"], ema_decay=BASE_KNOBS["ema_decay"], members=lr["members"],
                    hidden=GRID["MB-wide"]["hidden"] if arm == "MB-wide" else lr["hidden"], minibatch=lr["minibatch"],
                    grad_clip=lr["grad_clip"], bootstrap_p=lr["bootstrap_p"], aux_weight=lr["aux_weight"],
                    plan_samples=lr["plan_samples"], capacity=capacity or lr["capacity"], bounded=True,
                    total_episodes=config["episodes"])
    seeds = D1.learner_seeds(seed, c, "MB")          # D1's keys: every arm shares MB's init, masks, draws
    if arm in ("MB", "MB-wide"):
        return L.make_learner("MB", hyper, seeds, track_ema=track_ema), hyper
    rngs = {k: np.random.default_rng(seeds[k]) for k in L.RNG_STREAMS}
    if arm == "MB-prior":
        return LB.MBPrior(hyper, rngs, value, np.random.default_rng(seed_for("commons_d1b_prior", seed, c)), track_ema), hyper
    if arm == "MB-UCB":
        return LB.MBUCB(hyper, rngs, value, track_ema), hyper
    if arm == "MB-eps":
        return LB.MBEps(hyper, rngs, value, np.random.default_rng(seed_for("commons_d1b_eps", seed, c)), track_ema), hyper
    raise ValueError(arm)


def run_fit(config, identity, output, arm, tag, seed, ctx, sub="fits", interrupt_after=None, capacity=None, track_ema=True,
            forced_actions=None):
    """D1's fit loop (D1.run_fit) with the D1b arm factory and extra traces (exploration flags, ensemble disagreement)."""
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
    stream = E.Stream(params, c, seed, n)
    learner, hyper = make_arm_learner(config, arm, value, seed, c, track_ema, capacity)
    panel = E.Panel(params, c, config["panel_size"], hyper.plan_samples, with_draws=True)
    nd = config["disagreement_states"]
    spec = dict(arm=arm, tag=tag, value=value, seed=seed, context=ctx, context_index=c, hyper=hyper.__dict__, episodes=n,
                checkpoint_every=every, probe_episodes=sorted(probe_at), track_ema=track_ema,
                forced_actions=forced_actions is not None, capacity_override=capacity,
                learner_class=f"{type(learner).__module__}.{type(learner).__qualname__}")
    saved = D1.checkpoint_load(output, ckpt_path, identity)
    resumed = False
    if saved is not None and saved["record"]["spec"] == spec:
        learner.load(saved["learner"])
        record, trace, probes = saved["record"], saved["trace"], saved["probes"]
        require(learner.hashes() == record["checkpoint_hashes"], "checkpoint learner state changed")
        record["resumes"].append(dict(utc=now(), episodes_done=record["episodes_done"]))
        resumed = True
    else:
        record = dict(identity=identity, label=LABEL, spec=spec, episodes_done=0, resumes=[], started_utc=now(), training_seconds=0.0,
                      probe_seconds=0.0, elapsed_seconds=0.0, stream_fingerprint=stream.fingerprint(), decisions_chain="",
                      sensors_chain="", outcomes_chain="", initial_hashes=learner.hashes(), max_replay=0)
        trace = dict(action=np.zeros(n, np.int8), member=np.zeros(n, np.int8), realised=np.zeros(n, bool),
                     replay_size=np.zeros(n, np.int16), base_action=np.zeros(n, np.int8), explore=np.zeros(n, bool),
                     act_sd=np.full(n, np.nan, np.float32))
        probes = []
    require(record["stream_fingerprint"] == stream.fingerprint(), "stream draws changed")

    def do_probe(episode):
        t0 = time.perf_counter()
        out = D1.probe(learner, panel, "MB")
        before = learner.hashes()
        vals = LB.member_values(learner, panel.reserves[:nd], panel.cue[:nd], panel.eps[:nd])
        require(learner.hashes() == before, "the disagreement probe changed the learner")
        out["member_sd"] = vals.std(0).astype(np.float32)                     # (nd, 13) population SD over members
        probes.append(dict(episode=episode, **out))
        record["probe_seconds"] += time.perf_counter() - t0

    started = time.perf_counter()
    if record["episodes_done"] == 0 and not probes:
        do_probe(0)
    for k in range(record["episodes_done"], n):
        t0 = time.perf_counter()
        decision = stream.decision(k)
        action = learner.act(decision)
        base = getattr(learner, "last_base_action", action)
        explore = getattr(learner, "last_explore", False)
        if forced_actions is not None:
            action = int(forced_actions[k])
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
        trace["explore"][k] = explore
        trace["act_sd"][k] = getattr(learner, "last_sd", np.nan)
        require(learner.replay.size <= hyper.capacity, "bounded replay exceeded its budget")
        record["max_replay"] = max(record["max_replay"], learner.replay.size)
        record["episodes_done"] = k + 1
        if k + 1 in probe_at:
            do_probe(k + 1)
        if (k + 1) % every == 0:
            record["elapsed_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            record["checkpoint_hashes"] = learner.hashes()
            D1.checkpoint_save(ckpt_path, dict(identity=identity, learner=learner.state(), record=record, trace=trace, probes=probes))
        if interrupt_after is not None and k + 1 == interrupt_after:
            raise D1.Interrupted(f"deliberate interruption after episode {k + 1}")
    record["elapsed_seconds"] += time.perf_counter() - started
    arrays = dict(trace)
    arrays["probe_episode"] = np.array([p["episode"] for p in probes], np.int32)
    for key in probes[0]:
        if key != "episode":
            arrays[f"probe_{key}"] = np.stack([np.asarray(p[key]) for p in probes])
    info = save_arrays(directory / "trace.npz", meta=dict(arm=arm, tag=tag, seed=seed, context=ctx), **arrays)
    record.update(final_hashes=learner.hashes(), max_replay=int(max(record["max_replay"], learner.replay.max_size)),
                  replay=dict(capacity=learner.replay.capacity, max_size=learner.replay.max_size, seen=learner.replay.seen,
                              evictions=learner.replay.evictions, record_bytes=learner.replay.record_bytes()),
                  optimizer_steps=learner.steps, macs=dict(learner.macs, total=sum(learner.macs.values())),
                  memory_bytes=learner.memory_bytes(), parameters=learner.n_params(), trace_digest=info["digest"], resumed=resumed,
                  completed_utc=now())
    record["artifact_hashes"] = {"trace.npz": sha(directory / "trace.npz"), "trace.sha256.json": sha(directory / "trace.sha256.json")}
    record["checkpoint_hashes"] = learner.hashes()
    D1.checkpoint_save(ckpt_path, dict(identity=identity, learner=learner.state(), record=record, trace=trace, probes=probes))
    record["artifact_hashes"]["checkpoint.pt"] = sha(ckpt_path)
    write_json(result_path, record)
    return dict(status="completed", seconds=record["elapsed_seconds"])


# ------------------------------------------------------------------ pool

def estimated_cost(config, kind, payload):
    if kind == "fit":
        n = config["episodes"] / 4096
        probes = len(D1.probe_schedule(config))
        factor = dict(MB=1.0, **{"MB-prior": 2.0, "MB-UCB": 1.4, "MB-eps": 1.0, "MB-wide": 3.5})[payload["arm"]]
        return factor * (n * 70 + probes * 22 * 1.25)
    return 0.23 * config["panel_size"] if payload["kind"] == "panel" else 0.075 * config["oracle"]["stream_chunk"]


def dispatch(kind, payload):
    if kind == "fit":
        return run_fit(**payload)
    return D1.run_oracle(**payload)


def job_label(kind, payload):
    if kind == "fit":
        return dict(kind="fit", arm=payload["arm"], tag=payload["tag"], seed=payload["seed"], context=payload["ctx"])
    return dict(kind="oracle", part=payload["kind"], seed=payload["seed"], context=payload["ctx"], start=payload["start"])


def run_pool(config, output, jobs, stage):
    """As D1.run_pool (kill a broken or stalled pool, log every failure, retry the stage; finished tasks only verify)."""
    jobs = sorted(jobs, key=lambda job: -estimated_cost(config, *job))
    for attempt in range(1, config["stage_attempts"] + 1):
        log_event(output, event="stage_start", stage=stage, attempt=attempt, jobs=len(jobs))
        started, failures, statuses = time.perf_counter(), [], []
        pool = ProcessPoolExecutor(max_workers=config["workers"], initializer=D1.worker_init, initargs=(config["threads"],))
        futures = {pool.submit(dispatch, kind, payload): (kind, payload) for kind, payload in jobs}
        pending, stalled = set(futures), False
        while pending:
            done, pending = wait(pending, timeout=config["stall_seconds"], return_when=FIRST_COMPLETED)
            if not done:
                stalled = True
                killed = D1.kill_pool(pool)
                log_event(output, event="pool_stalled", stage=stage, attempt=attempt, pending=len(pending), killed_processes=killed)
                for future in pending:
                    failures.append(dict(**job_label(*futures[future]), error="pool stalled; killed"))
                break
            for future in done:
                kind, payload = futures[future]
                label = job_label(kind, payload)
                try:
                    result = future.result()
                    statuses.append(dict(label, status=result.get("status"), seconds=result.get("seconds")))
                    log_event(output, event="task_done", stage=stage, attempt=attempt, **label, status=result.get("status"),
                              seconds=result.get("seconds"))
                    print(json.dumps(dict(stage=stage, done=len(statuses), total=len(jobs), **label, status=result.get("status"),
                                          seconds=round(result.get("seconds") or 0, 1), elapsed=round(time.perf_counter() - started))),
                          flush=True)
                except Exception as error:  # noqa: BLE001 - every failed attempt is kept and reported
                    failures.append(dict(**label, error=repr(error)))
                    log_event(output, event="task_failed", stage=stage, attempt=attempt, **label, error=repr(error),
                              traceback=traceback.format_exc())
                    print(json.dumps(dict(failed=label, error=repr(error))), flush=True)
        if not stalled:
            pool.shutdown(wait=True)
        if not failures:
            log_event(output, event="stage_complete", stage=stage, attempt=attempt, seconds=time.perf_counter() - started)
            return statuses
        log_event(output, event="stage_attempt_failed", stage=stage, attempt=attempt, failures=len(failures))
    raise RuntimeError(f"stage {stage}: tasks failed after {config['stage_attempts']} attempts; rerun to resume")


def fit_jobs(config, identity, output, tasks, sub="fits"):
    return [("fit", dict(config=config, identity=identity, output=str(output), arm=a, tag=t, seed=s, ctx=c, sub=sub))
            for a, t, s, c in tasks]


def oracle_jobs(config, identity, output):
    return [("oracle", dict(config=config, identity=identity, output=str(output), kind=k, seed=s, ctx=c, start=st))
            for k, s, c, st in D1.oracle_tasks(config)]


def verify_all(config, output, identity, tasks, sub="fits"):
    fits = []
    for arm, tag, seed, ctx in tasks:
        d = fit_dir(output, arm, tag, seed, ctx, sub)
        rec = D1.verify_result(d, identity)
        load_arrays(d / "trace.npz")
        fits.append(dict(arm=arm, tag=tag, seed=seed, context=ctx, path=(d / "result.json").relative_to(output).as_posix(),
                         sha256=sha(d / "result.json"), elapsed_seconds=rec["elapsed_seconds"]))
    oracle = []
    for kind, seed, ctx, start in D1.oracle_tasks(config):
        p = D1.oracle_path(output, kind, seed, ctx, start)
        _, info = load_arrays(p)
        require(info["meta"]["identity"] == identity, f"oracle identity mismatch {p}")
        oracle.append(dict(path=p.relative_to(output).as_posix(), sha256=sha(p)))
    return fits, oracle


# ------------------------------------------------------------------ metrics

def o_best_frequency(tables, ctx):
    tab = tables.panel_tab(ctx)
    return np.bincount(tab["select"][:, E.context_of(ctx)].argmax(1), minlength=L.ACTIONS) / len(tab["select"])


def critical_losses(tables, ctx):
    """Expected survival O loses on the panel if each action were removed from its action set."""
    tab = tables.panel_tab(ctx)
    c = E.context_of(ctx)
    rows = np.arange(len(tab["select"]))
    sel, sc = tab["select"][:, c], tab["score"][:, c]
    a_o = sel.argmax(1)
    out = []
    for a in range(L.ACTIONS):
        s2 = sel.copy()
        s2[:, a] = -np.inf
        out.append(float((sc[rows, a_o] - sc[rows, s2.argmax(1)]).mean()))
    return out


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
    usage = (np.bincount(tr["action"].astype(np.int64), minlength=L.ACTIONS) / len(tr["action"])).tolist()
    greedy_usage = (np.bincount(greedy, minlength=L.ACTIONS) / len(greedy)).tolist()
    m = dict(arm=arm, tag=tag, seed=seed, context=ctx, regret_endpoint=curve[-1], regret_curve=curve,
             curve_episodes=tr["probe_episode"].tolist(),
             cumulative_online_regret=float(online.sum()), online_regret_per_episode=float(online.mean()),
             online_regret_first_1024=float(online[:1024].sum()), online_regret_last_1024=float(online[-1024:].sum()),
             realised_survival=float(tr["realised"].mean()),
             greedy_equals_O_fraction=float((greedy == ptab["select"][:, c].argmax(1)).mean()),
             training_usage=usage, greedy_usage=greedy_usage, explore_fraction=float(tr["explore"].mean()),
             explore_matches_base_elsewhere=bool(np.array_equal(tr["action"][~tr["explore"]], tr["base_action"][~tr["explore"]])),
             disagreement_curve=[float(x.mean()) for x in tr["probe_member_sd"]],
             act_sd_mean=float(np.nanmean(tr["act_sd"])) if np.isfinite(tr["act_sd"]).any() else None,
             hazard_nll_endpoint=float(tr["probe_hazard_nll"][-1]),
             value_optimism_greedy=float((pred[rows, greedy] - score[rows, greedy]).mean()),
             value_rmse_all_actions=float(np.sqrt(((pred - score) ** 2).mean())),
             wall_seconds=rec["elapsed_seconds"], training_seconds=rec["training_seconds"], probe_seconds=rec["probe_seconds"],
             macs_train=rec["macs"]["train"], macs_act=rec["macs"]["act"], macs_probe=rec["macs"]["probe"],
             bytes_total=rec["memory_bytes"]["total"], parameters=rec["parameters"], max_replay=rec["max_replay"])
    if ctx == TRAP["context"]:
        f = o_best_frequency(tables, ctx)
        ratios = {str(a): (usage[a] / f[a] if f[a] > 0 else None) for a in TRAP["actions"]}
        m["trap_usage_ratios"] = ratios
        m["trap_usage_ok"] = bool(all(TRAP["ratio_min"] <= r <= TRAP["ratio_max"] for r in ratios.values()))
        m["trap_regret_ok"] = bool(curve[-1] <= TRAP["regret_max"])
    return m


def check_trap_definition(tables):
    """The declared trap action set must be exactly the actions whose removal costs O >= critical_loss."""
    losses = critical_losses(tables, TRAP["context"])
    derived = [a for a in range(1, L.ACTIONS) if losses[a] >= TRAP["critical_loss"]]
    require(derived == TRAP["actions"], f"trap action set {derived} differs from the declared {TRAP['actions']}")
    return dict(losses=losses, derived=derived, o_best_frequency=o_best_frequency(tables, TRAP["context"]).tolist())


def summarize(values, idx):
    v = np.asarray(values, np.float64)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                ci95=D1.boot_mean_ci(idx, v) if idx is not None else None)


def per_seed_readings(config, by, arm, tag):
    ctxs = config["contexts"]
    out = {}
    for s in config["seeds"]:
        vals = [by[(arm, tag, s, c)]["regret_endpoint"] for c in ctxs]
        trap = by[(arm, tag, s, TRAP["context"])]
        out[str(s)] = dict(context_average=float(np.mean(vals)), worst_context=float(max(vals)),
                           worst_context_name=ctxs[int(np.argmax(vals))], mean_ok=bool(np.mean(vals) <= config["analysis"]["competent_mean"]),
                           worst_ok=bool(max(vals) <= config["analysis"]["competent_worst"]),
                           trap_regret=trap["regret_endpoint"], trap_regret_ok=trap["trap_regret_ok"], trap_usage_ok=trap["trap_usage_ok"],
                           trap_usage_ratios=trap["trap_usage_ratios"], action9_training_usage=trap["training_usage"][9],
                           action9_greedy_usage=trap["greedy_usage"][9],
                           cumulative_online_regret=float(np.sum([by[(arm, tag, s, c)]["cumulative_online_regret"] for c in ctxs])))
    return out


def arm_summary(config, by, rows, arm, tag, idx):
    ps = per_seed_readings(config, by, arm, tag)
    rs = [r for r in rows if r["arm"] == arm and r["tag"] == tag]
    ctxs = config["contexts"]
    seeds = config["seeds"]
    curve = np.mean([r["regret_curve"] for r in rs], 0)
    dis = np.mean([r["disagreement_curve"] for r in rs], 0)
    return dict(
        arm=arm, tag=tag, cells=len(rs),
        context_average=summarize([v["context_average"] for v in ps.values()], idx),
        worst_context=summarize([v["worst_context"] for v in ps.values()], idx),
        seeds_meeting_both=int(sum(v["mean_ok"] and v["worst_ok"] for v in ps.values())),
        seeds_meeting_mean=int(sum(v["mean_ok"] for v in ps.values())), seeds_meeting_worst=int(sum(v["worst_ok"] for v in ps.values())),
        trap_seeds_regret_ok=int(sum(v["trap_regret_ok"] for v in ps.values())),
        trap_seeds_usage_ok=int(sum(v["trap_usage_ok"] for v in ps.values())),
        trap_seeds_both=int(sum(v["trap_regret_ok"] and v["trap_usage_ok"] for v in ps.values())),
        by_context={c: summarize([by[(arm, tag, s, c)]["regret_endpoint"] for s in seeds], idx) for c in ctxs},
        per_seed=ps, curve_episodes=rs[0]["curve_episodes"], curve_mean=curve.tolist(), disagreement_curve=dis.tolist(),
        cumulative_online_regret=summarize([v["cumulative_online_regret"] / len(ctxs) for v in ps.values()], idx),
        online_regret_first_1024=float(np.mean([r["online_regret_first_1024"] for r in rs])),
        online_regret_last_1024=float(np.mean([r["online_regret_last_1024"] for r in rs])),
        realised_survival=float(np.mean([r["realised_survival"] for r in rs])),
        action9_training_usage_A1B1C1=[v["action9_training_usage"] for v in ps.values()],
        trap_training_usage_A1B1C1={str(a): float(np.mean([by[(arm, tag, s, TRAP["context"])]["training_usage"][a] for s in seeds]))
                                    for a in range(L.ACTIONS)},
        explore_fraction=float(np.mean([r["explore_fraction"] for r in rs])),
        hazard_nll_endpoint=float(np.mean([r["hazard_nll_endpoint"] for r in rs])),
        value_optimism_greedy=float(np.mean([r["value_optimism_greedy"] for r in rs])),
        value_rmse_all_actions=float(np.mean([r["value_rmse_all_actions"] for r in rs])),
        greedy_equals_O_fraction=float(np.mean([r["greedy_equals_O_fraction"] for r in rs])),
        wall_seconds=float(np.mean([r["wall_seconds"] for r in rs])), training_seconds=float(np.mean([r["training_seconds"] for r in rs])),
        probe_seconds=float(np.mean([r["probe_seconds"] for r in rs])),
        macs={k: float(np.mean([r[k] for r in rs])) for k in ("macs_train", "macs_act", "macs_probe")},
        bytes_total=float(np.mean([r["bytes_total"] for r in rs])), parameters=int(rs[0]["parameters"]),
        max_replay=int(max(r["max_replay"] for r in rs)))


def select_setting(rows, arm, cells):
    """Protocol rule: among the remedy's settings whose mean greedy regret at N is within one paired SE (over
    seed x context cells) of the remedy's best greedy regret, the lowest mean cumulative online regret
    (ties -> lower greedy regret)."""
    by = {}
    for r in rows:
        if r["arm"] == arm:
            by.setdefault(r["tag"], {})[(r["seed"], r["context"])] = r
    for tag in by:
        require(sorted(by[tag]) == cells, f"incomplete development cells for {arm} {tag}")
    g = {t: float(np.mean([by[t][c]["regret_endpoint"] for c in cells])) for t in by}
    cum = {t: float(np.mean([by[t][c]["cumulative_online_regret"] for c in cells])) for t in by}
    best = min(g, key=lambda t: (g[t], t))
    table = []
    for t in sorted(by):
        diff = np.array([by[t][c]["regret_endpoint"] - by[best][c]["regret_endpoint"] for c in cells])
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if t != best else 0.0
        table.append(dict(tag=t, value=value_of(arm, t), greedy_regret=g[t], diff_vs_best=float(diff.mean()), paired_se=se,
                          within=bool(t == best or diff.mean() <= se), cumulative_online_regret=cum[t]))
    within = [x for x in table if x["within"]]
    chosen = min(within, key=lambda x: (x["cumulative_online_regret"], x["greedy_regret"], x["tag"]))
    return dict(arm=arm, knob=KNOB[arm], best_greedy_tag=best, selected_tag=chosen["tag"], selected_value=chosen["value"],
                table=table, rule="within one paired SE of the best greedy regret -> lowest mean cumulative online regret")


# ------------------------------------------------------------------ dev

def cmd_dev(args):
    config = load_config(args.config)
    output, report = Path(args.output), Path(args.report)
    manifest = lock(config, args.config, output)
    identity = manifest["identity"]
    log_event(output, event="dev_start", identity=identity)
    tasks = fit_tasks(config)
    t0 = time.perf_counter()
    run_pool(config, output, oracle_jobs(config, identity, output) + fit_jobs(config, identity, output, tasks), "dev")
    check_lock(config, args.config, output)
    fits, oracle = verify_all(config, output, identity, tasks)
    crc = D1.crc_scan(output)
    write_json(output / "crc_scan.json", crc)
    require(crc["passed"], f"CRC scan failed: {crc['failures']}")
    write_json(output / "completion.json", dict(identity=identity, fits=fits, oracle=oracle, crc=crc, utc=now(),
                                                wall_seconds_this_invocation=time.perf_counter() - t0))
    dev_report(config, output, report, identity, tasks, crc)


def dev_report(config, output, report, identity, tasks, crc):
    tables = D1.Tables(config, output)
    trap_def = check_trap_definition(tables)
    rows = [fit_metrics(config, tables, output, *t) for t in tasks]
    by = {(r["arm"], r["tag"], r["seed"], r["context"]): r for r in rows}
    cells = sorted({(r["seed"], r["context"]) for r in rows})
    selection = {arm: select_setting(rows, arm, cells) for arm in KNOB}
    idx = None
    summary = {}
    for arm in ARMS:
        for v in arm_values(arm):
            tag = tag_of(arm, v)
            summary[f"{arm}|{tag}"] = arm_summary(config, by, rows, arm, tag, idx)
    mb = {c: by[("MB", "base") + c]["regret_endpoint"] for c in cells}
    pathology = {}
    for key, s in summary.items():
        arm, tag = key.split("|")
        if arm == "MB":
            continue
        diff = np.array([by[(arm, tag) + c]["regret_endpoint"] - mb[c] for c in cells])
        pathology[key] = dict(greedy_minus_MB=float(diff.mean()), paired_se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                              worse_than_MB=bool(diff.mean() > 0))
    scale = {ctx: D1.context_scale(tables.panel_tab(ctx), E.context_of(ctx)) for ctx in config["contexts"]}
    out = dict(label="DEVELOPMENT PHASE (seeds 21201-21203); not a test result. " + LABEL, identity=identity,
               selection=selection, summary=summary, pathology_vs_MB=pathology, trap_definition=dict(TRAP, **trap_def),
               context_scale=scale, crc=crc, utc=now())
    write_json(report / "dev_results.json", out)
    D1.write_csv(report / "dev_fits.csv", [{k: v for k, v in r.items() if k not in ("regret_curve", "curve_episodes", "disagreement_curve")}
                                           for r in rows])
    D1.write_csv(report / "dev_curves.csv", [dict(arm=r["arm"], tag=r["tag"], seed=r["seed"], context=r["context"], episode=e, regret=v,
                                                  disagreement=dd)
                                             for r in rows for e, v, dd in zip(r["curve_episodes"], r["regret_curve"], r["disagreement_curve"])])
    print(json.dumps({a: s["selected_tag"] for a, s in selection.items()}, indent=1), flush=True)
    return out


# ------------------------------------------------------------------ test

def cmd_lock(args):
    config = load_config(args.config)
    require(config["phase"] in ("test", "smoke"), "lock is for the smoke or test config")
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
    seeds = list(REPLICATE_SEEDS) if config["phase"] == "test" else list(config["seeds"])
    return [t for t in fit_tasks(config) if t[2] in seeds]


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
    rep = dict(identity=identity, utc=now(), replicate_seeds=sorted({t[2] for t in rtasks}), fits_compared=len(rtasks),
               main_fits_reverified=len(tasks), comparisons=comparisons, mismatches=mismatches, crc=crc,
               passed=not mismatches and crc["passed"], seconds=time.perf_counter() - t0)
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
    trap_def = check_trap_definition(tables) if config["phase"] == "test" else dict(skipped="smoke")
    rows = [fit_metrics(config, tables, output, *t) for t in tasks]
    by = {(r["arm"], r["tag"], r["seed"], r["context"]): r for r in rows}
    a = config["analysis"]
    seeds = config["seeds"]
    need = a["consistency_seeds"] if config["phase"] == "test" else len(seeds)
    B = a.get("bootstrap_resamples", 2000)
    idx = np.random.default_rng(a.get("analysis_seed", 0)).integers(0, len(seeds), (B, len(seeds))) if len(seeds) > 1 else None
    arms = {arm: tag_of(arm, config["knobs"].get(arm)) for arm in ARMS}
    summary = {arm: arm_summary(config, by, rows, arm, tag, idx) for arm, tag in arms.items()}
    r1 = {arm: dict(seeds_meeting_both=s["seeds_meeting_both"], seeds_meeting_mean=s["seeds_meeting_mean"],
                    seeds_meeting_worst=s["seeds_meeting_worst"], required=need, competent=bool(s["seeds_meeting_both"] >= need))
          for arm, s in summary.items()}
    r2 = {arm: dict(seeds_regret_ok=s["trap_seeds_regret_ok"], seeds_usage_ok=s["trap_seeds_usage_ok"], seeds_both=s["trap_seeds_both"],
                    required=need, escapes=bool(s["trap_seeds_both"] >= need)) for arm, s in summary.items()}
    r3 = {}
    for arm, tag in arms.items():
        if arm == "MB":
            continue
        d = [summary[arm]["per_seed"][str(s)]["cumulative_online_regret"] - summary["MB"]["per_seed"][str(s)]["cumulative_online_regret"]
             for s in seeds]
        k = sum(x <= 0 for x in d)
        r3[arm] = dict(seeds_at_or_below_MB=int(k), required=need, pays_back=bool(k >= need),
                       applies=bool(r1[arm]["competent"]), net_cost_vs_MB=summarize(d, idx))
    competent = [arm for arm in ARMS if r1[arm]["competent"] and arm != "MB"]
    core = min(competent, key=lambda arm: summary[arm]["cumulative_online_regret"]["mean"]) if competent else None
    escapes = [arm for arm in ARMS if r2[arm]["escapes"] and arm != "MB"]
    if competent:
        implication = f"Competent remedy(ies) {competent}; D2 core = {core} (lowest mean cumulative online regret)."
    elif escapes:
        implication = f"{escapes} escape the trap but miss R1: declared D1c round on replay regularisation."
    else:
        implication = "No remedy escapes the trap: report and decide with the user (VOI signal or count-based bonuses)."
    out = dict(label=LABEL + (" SMOKE ONLY - NOT SCIENCE." if config["phase"] == "smoke" else ""), identity=identity,
               knobs=config["knobs"], bootstrap=dict(resamples=B, analysis_seed=a.get("analysis_seed")), R1=r1, R2=r2, R3=r3,
               competent_remedies=competent, d2_core=core, escaping_remedies=escapes, implication=implication, summary=summary,
               trap_definition=dict(TRAP, **trap_def), crc=crc, fits=len(rows), utc=now())
    write_json(report / "results.json", out)
    D1.write_csv(report / "fits.csv", [{k: v for k, v in r.items() if k not in ("regret_curve", "curve_episodes", "disagreement_curve")}
                                       for r in rows])
    print(json.dumps(dict(R1={k: v["seeds_meeting_both"] for k, v in r1.items()}, R2={k: v["seeds_both"] for k, v in r2.items()},
                          implication=implication), indent=1), flush=True)
    return out


# ------------------------------------------------------------------ smoke checks

def import_check():
    tree = ast.parse((ROOT / "src/acp_cl/commons/learners_d1b.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {x.name for x in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add("." * node.level + (node.module or ""))
    allowed = {"__future__", "numpy", "torch", "torch.nn", ".learners"}
    d1 = D1.learner_import_check()
    return dict(d1b_imports=sorted(mods), d1_learners=d1, passed=bool(mods <= allowed and d1["passed"]))


def remedy_unit_check(config):
    """Each remedy is active only where declared (learner level)."""
    params, _ = E.load_world(ROOT)
    seed, c = SMOKE_SEED, 7
    st = E.Stream(params, c, seed, 40)
    out = {}
    base, _ = make_arm_learner(config, "MB", None, seed, c)
    p1, _ = make_arm_learner(config, "MB-prior", 1, seed, c)
    p3, _ = make_arm_learner(config, "MB-prior", 3, seed, c)
    same_init = all(torch.equal(a, b) and torch.equal(a, d) for a, b, d in zip(base.net.params, p1.net.params, p3.net.params))
    x = torch.randn(5, 7, MBLEN)
    f = L.Ensemble.forward(p1.net.params, x)
    pr = L.Ensemble.forward(p1.prior, x)
    y = torch.randn(1, 8, 36)
    r = np.array([[5.0, 6.0, 7.0]])
    q = np.array([[1, 0, 1]], np.uint8)
    lin = bool(p1.beta == 1.0 and p3.beta == 3.0 and all(torch.equal(a_, b_) for a_, b_ in zip(p1.prior, p3.prior)))
    base_logits = base._values(base.ema[0], 2, r, q, y)
    raw = p1._raw_logits(p1.ema[0], 2, r, q, y)
    prior_changes = not torch.allclose(p1._values(p1.ema[0], 2, r, q, y), base_logits)
    base_equals_raw = torch.allclose(torch.nn.functional.logsigmoid(-raw).sum(-1), base_logits, atol=1e-6)
    prior0 = LB.MBPrior(p1.h, {k: np.random.default_rng(v) for k, v in D1.learner_seeds(seed, c, "MB").items()}, 0.0,
                        np.random.default_rng(seed_for("commons_d1b_prior", seed, c)))
    beta0_equals_mb = torch.equal(prior0._values(prior0.ema[0], 2, r, q, y), base_logits)
    ph = p1.prior_hash
    for k in range(12):
        dd = st.decision(k)
        p1.observe(dd, p1.act(dd), st.outcome(k, 3))
    prior_fixed = LB.tensor_hash(p1.prior) == ph and not any(p.requires_grad for p in p1.prior)
    out["prior"] = dict(base_members_identical_to_MB_at_init=bool(same_init), prior_output_nonzero=bool(pr.abs().max() > 0),
                        forward_shape_ok=bool(f.shape == (5, 7, 36)), base_path_equals_raw_logits=bool(base_equals_raw),
                        prior_changes_values=bool(prior_changes), beta0_reproduces_MB_exactly=bool(beta0_equals_mb),
                        beta_linear=bool(lin), prior_never_trained=bool(prior_fixed),
                        other_arms_have_no_prior=not any(hasattr(make_arm_learner(config, a, v, seed, c)[0], "prior")
                                                         for a, v in (("MB", None), ("MB-UCB", 1), ("MB-eps", 0.1), ("MB-wide", None))))
    out["prior"]["passed"] = all(v for k, v in out["prior"].items())
    # UCB: the action is argmax(mean + kappa * SD) over members on the same draws; no Thompson draw
    u, _ = make_arm_learner(config, "MB-UCB", 2, seed, c)
    for k in range(12):
        dd = st.decision(k)
        u.observe(dd, 5, st.outcome(k, 5))
    dd = st.decision(20)
    plan_state = copy.deepcopy(u.rngs["plan"].bit_generator.state)
    th_state = copy.deepcopy(u.rngs["thompson"].bit_generator.state)
    a = u.act(dd)
    eps = np.random.default_rng(0)
    eps.bit_generator.state = plan_state
    e = eps.standard_normal((u.h.plan_samples, 12, 3)).astype(np.float32)
    vals = LB.member_values(u, dd.reserves[None], dd.cue[None], e[None])[:, 0]
    manual = int(np.argmax(vals.mean(0) + 2 * vals.std(0)))
    out["ucb"] = dict(action=a, manual=manual, equals=bool(a == manual),
                      thompson_stream_untouched=u.rngs["thompson"].bit_generator.state == th_state,
                      passed=bool(a == manual and u.rngs["thompson"].bit_generator.state == th_state))
    return out


MBLEN = L.MBLearner.n_in


def trace_remedy_check(config, output, tasks):
    """Fit-level: epsilon explores at about its rate and otherwise takes the Thompson action; other arms never explore;
    UCB never draws a Thompson member; MB-wide alone has width 256."""
    rows = {}
    ok = True
    for arm, tag, seed, ctx in tasks:
        tr, _ = load_arrays(fit_dir(output, arm, tag, seed, ctx) / "trace.npz")
        rec = read_json(fit_dir(output, arm, tag, seed, ctx) / "result.json")
        n = len(tr["action"])
        r = dict(explore_fraction=float(tr["explore"].mean()),
                 non_explore_equals_base=bool(np.array_equal(tr["action"][~tr["explore"]], tr["base_action"][~tr["explore"]])),
                 thompson_members=sorted(set(tr["member"].tolist())), hidden=rec["spec"]["hyper"]["hidden"])
        if arm == "MB-eps":
            eps = value_of(arm, tag)
            z = (r["explore_fraction"] - eps) / np.sqrt(eps * (1 - eps) / n)
            good = abs(z) < 4 and r["non_explore_equals_base"]
            r["z"] = float(z)
        else:
            good = r["explore_fraction"] == 0 and np.array_equal(tr["action"], tr["base_action"])
        if arm == "MB-UCB":
            good &= r["thompson_members"] == [-1]
        else:
            good &= -1 not in r["thompson_members"]
        good &= r["hidden"] == (256 if arm == "MB-wide" else 128)
        r["passed"] = bool(good)
        rows[f"{arm}|{ctx}"] = r
        ok &= good
    return dict(fits=rows, passed=bool(ok))


def training_unchanged_check(config, identity, output):
    """With the actions forced to MB's, MB-UCB and MB-eps train exactly like MB (online weights and optimiser bit-identical);
    MB-prior differs (its prior enters the loss, as declared); EMA does not interfere with MB-prior's training."""
    base = fresh(output / "checks" / "training")
    seed, ctx = SMOKE_SEED, "A1B1C1"
    cfg = dict(config, episodes=128, checkpoint_every=64, probe_episodes=[64, 128])
    run_fit(cfg, identity, base, "MB", "base", seed, ctx)
    tr, _ = load_arrays(fit_dir(base, "MB", "base", seed, ctx) / "trace.npz")
    forced = tr["action"].astype(np.int64)
    ref = read_json(fit_dir(base, "MB", "base", seed, ctx) / "result.json")["final_hashes"]
    out = {}
    for arm, v in (("MB-UCB", 2), ("MB-eps", 0.1), ("MB-prior", 3)):
        run_fit(cfg, identity, base, arm, tag_of(arm, v), seed, ctx, forced_actions=forced)
        h = read_json(fit_dir(base, arm, tag_of(arm, v), seed, ctx) / "result.json")["final_hashes"]
        same = h["online"] == ref["online"] and h["optimizer"] == ref["optimizer"]
        out[arm] = dict(online_and_optimizer_identical_to_MB=same, expected=arm != "MB-prior", passed=bool(same == (arm != "MB-prior")))
    run_fit(cfg, identity, base / "noema", "MB-prior", "beta3", seed, ctx, forced_actions=forced, track_ema=False)
    h1 = read_json(fit_dir(base, "MB-prior", "beta3", seed, ctx) / "result.json")["final_hashes"]
    h2 = read_json(fit_dir(base / "noema", "MB-prior", "beta3", seed, ctx) / "result.json")["final_hashes"]
    out["MB-prior_ema_does_not_interfere"] = dict(online_identical=h1["online"] == h2["online"], ema_differs=h1["ema"] != h2["ema"],
                                                  passed=bool(h1["online"] == h2["online"] and h1["ema"] != h2["ema"]))
    out["passed"] = all(v["passed"] for v in out.values())
    return out


def d1_identity_check(identity, output):
    """The D1b MB arm reproduces a stored D1 test fit (same seed and context) bit for bit through the D1b fit loop."""
    d1_cfg = read_json(ROOT / "configs/commons_d1.json")
    ref = ROOT / "runs/commons_d1/fits/MB__lr0.001_U2_d0.99__s21101__A1B1C1"
    if not (ref / "result.json").exists():
        return dict(skipped="no stored D1 fit", passed=False)
    cfg = dict(read_json(ROOT / "configs/commons_d1b_dev.json"), seeds=[21101], probe_episodes=d1_cfg["probe_episodes"])
    base = fresh(output / "checks" / "d1_identity")
    run_fit(cfg, identity, base, "MB", "base", 21101, "A1B1C1")
    a, _ = load_arrays(ref / "trace.npz")
    b, _ = load_arrays(fit_dir(base, "MB", "base", 21101, "A1B1C1") / "trace.npz")
    ra, rb = read_json(ref / "result.json"), read_json(fit_dir(base, "MB", "base", 21101, "A1B1C1") / "result.json")
    keys = sorted(a.files if hasattr(a, "files") else a)
    arrays_equal = {k: bool(a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes()) for k in keys}
    chains = all(ra[k] == rb[k] for k in ("stream_fingerprint", "decisions_chain", "sensors_chain", "outcomes_chain",
                                          "initial_hashes", "final_hashes", "checkpoint_hashes", "max_replay", "optimizer_steps"))
    chains = chains and ra["macs"]["train"] == rb["macs"]["train"] and ra["macs"]["act"] == rb["macs"]["act"]  # D1b adds probe MACs
    return dict(reference=ref.relative_to(ROOT).as_posix(), d1_arrays_equal=arrays_equal, records_equal=bool(chains),
                extra_d1b_arrays=sorted(set(b) - set(keys)), passed=bool(all(arrays_equal.values()) and chains))


def eviction_check(config, identity, output):
    base = fresh(output / "checks" / "eviction")
    cfg = dict(config, episodes=128, checkpoint_every=64, probe_episodes=[64, 128])
    run_fit(cfg, identity, base, "MB-eps", "epsilon0.1", SMOKE_SEED, "A0B1C0", capacity=64)
    rec = read_json(fit_dir(base, "MB-eps", "epsilon0.1", SMOKE_SEED, "A0B1C0") / "result.json")
    return dict(max_replay=rec["max_replay"], evictions=rec["replay"]["evictions"],
                passed=bool(rec["max_replay"] == 64 and rec["replay"]["evictions"] > 0))


def resume_check(config, identity, output, tasks):
    base = fresh(output / "checks" / "resume")
    out = {}
    for arm in ("MB-eps", "MB-prior", "MB-UCB"):
        _, tag, seed, ctx = [t for t in tasks if t[0] == arm][-1]
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


def legal_check(config):
    params, _ = E.load_world(ROOT)
    st = E.Stream(params, 7, SMOKE_SEED, 4)
    d, o = st.decision(0), st.outcome(0, 3)
    bad = L.make_decision(d.reserves, d.cue)
    object.__setattr__(bad, "context", 7)
    rows = {}
    for arm, v in (("MB-prior", 1), ("MB-UCB", 1), ("MB-eps", 0.1), ("MB-wide", None)):
        lrn, _ = make_arm_learner(config, arm, v, SMOKE_SEED, 7)
        try:
            lrn.act(bad)
            rows[arm] = False
        except L.IllegalObservation:
            rows[arm] = True
    base = D1.legal_observation_check(params)
    return dict(d1=base["passed"], remedies_reject_illegal=rows, passed=bool(base["passed"] and all(rows.values())))


def cmd_check(args):
    config = load_config(args.config)
    require(config["phase"] == "smoke", "check runs the smoke configuration")
    output = Path(args.output)
    fresh(output)
    D1.worker_init(1)
    t0 = time.perf_counter()
    checks = dict(label="ENGINEERING CHECKS on the D1b smoke run (seed 21299, 256 episodes). Not science.", utc=now())
    params, cfg = E.load_world(ROOT)
    checks["world"] = dict(key=E.cfg_key(cfg), passed=E.cfg_key(cfg) == E.WORLD_KEY)
    checks["imports"] = import_check()
    checks["legal_observations"] = legal_check(config)
    checks["reservoir_uniform_and_bounded"] = D1.reservoir_check()
    checks["remedy_active_only_where_declared_unit"] = remedy_unit_check(config)
    checks["remedy_active_only_where_declared_unit"]["passed"] = all(v["passed"] for v in checks["remedy_active_only_where_declared_unit"].values()
                                                                     if isinstance(v, dict))
    print("unit", {k: v.get("passed") for k, v in checks.items() if isinstance(v, dict)}, flush=True)
    identity = lock(config, args.config, output)["identity"]
    a = argparse.Namespace(config=args.config, output=str(output), report=str(output / "analysis"))
    cmd_run(a)
    tasks = fit_tasks(config)
    second = cmd_run(a)
    checks["rerun_only_verifies"] = dict(passed=all(s["status"] == "verified_existing" for s in second["statuses"]))
    tables = D1.Tables(config, output)
    checks["o_table_lookups"] = D1.o_lookup_check(config, output, tables, params)
    checks["common_random_numbers"] = D1.crn_check(config, output, tasks)
    checks["bounded_replay_in_run"] = dict(passed=all(read_json(fit_dir(output, *t) / "result.json")["max_replay"] <= 1024 for t in tasks))
    checks["remedy_active_only_where_declared_fits"] = trace_remedy_check(config, output, tasks)
    checks["training_unchanged_and_ema"] = training_unchanged_check(config, identity, output)
    checks["bounded_replay_eviction_path"] = eviction_check(config, identity, output)
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
    checks["timing_smoke"] = {t[0]: read_json(fit_dir(output, *t) / "result.json")["elapsed_seconds"] for t in tasks if t[3] == "A1B1C1"}
    checks["all_passed"] = bool(all(v["passed"] for v in checks.values() if isinstance(v, dict) and "passed" in v))
    checks["seconds"] = time.perf_counter() - t0
    write_json(output / "checks" / "checks.json", checks)
    print("ALL PASS" if checks["all_passed"] else "FAILURES: " + ", ".join(k for k, v in checks.items()
                                                                          if isinstance(v, dict) and v.get("passed") is False), flush=True)
    if not checks["all_passed"]:
        raise SystemExit(1)


DEFAULTS = {"check": ("configs/commons_d1b_smoke.json", "runs/commons_d1b_smoke", "runs/commons_d1b_smoke/analysis"),
            "dev": ("configs/commons_d1b_dev.json", "runs/commons_d1b_dev", "reports/commons_d1b/dev"),
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
