"""Step D1: learner competence in stationary commons contexts.

Implements docs/commons_d1_protocol.md (prospective, 2026-09-29). World: the accepted
D0b world (reports/commons_d0b/world_spec.json, key 1586f04b8d3cc8c2), loaded and
verified by ``acp_cl.commons.d1_env.load_world``. Learners: ``acp_cl.commons.learners``.

Commands (set PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src; never writes bytecode):
  .venv/Scripts/python.exe -B scripts/commons_d1.py check     [--config configs/commons_d1_smoke.json --output runs/commons_d1_smoke]
  .venv/Scripts/python.exe -B scripts/commons_d1.py dev       [--config configs/commons_d1_dev.json   --output runs/commons_d1_dev --report reports/commons_d1/dev]
  .venv/Scripts/python.exe -B scripts/commons_d1.py lock      [--config configs/commons_d1.json       --output runs/commons_d1]
  .venv/Scripts/python.exe -B scripts/commons_d1.py run       [...]
  .venv/Scripts/python.exe -B scripts/commons_d1.py replicate [...]
  .venv/Scripts/python.exe -B scripts/commons_d1.py analyze   [... --report reports/commons_d1]

Integrity: every task writes into its own directory; results carry the lock (or dev
manifest) identity and sha256 hashes of every artifact; arrays have sha256 sidecars;
fits checkpoint every ``checkpoint_every`` episodes (atomic torch.save + sha256 sidecar)
and resume bit-for-bit; every attempt is logged to attempts.jsonl; the process pool
(<= 6 single-thread workers) is killed and the stage retried if it breaks or stalls;
every zip archive (npz, pt, zip) is CRC-checked before analysis.
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
import csv  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zipfile  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from acp_cl.commons import d1_env as E  # noqa: E402
from acp_cl.commons import learners as L  # noqa: E402
from acp_cl.commons import references as R  # noqa: E402
from acp_cl.commons.world import (N_CONTEXTS, SURVIVED, context_name, sample_episodes, seed_for,  # noqa: E402
                                  simulate)

LABEL = ("PROSPECTIVE STUDY under docs/commons_d1_protocol.md (Step D1). O is evaluator-only; "
         "MF-unbounded and MB-unbounded are PRIVILEGED memory references.")
SCRIPT_FILES = ("scripts/commons_d1.py",)
BOUNDED = ("MF", "MB")
UNBOUNDED = {"MF": "MF-unbounded", "MB": "MB-unbounded"}
ARMS = ("MF", "MB", "MF-unbounded", "MB-unbounded")
TEST_SEEDS = list(range(21101, 21113))
DEV_SEEDS = [21001, 21002]
SMOKE_SEED = 21099
REPLICATE_SEEDS = (21101, 21106, 21111)
GRID = {"lr": [0.001, 0.003], "U": [2, 8], "ema_decay": [0.99, 0.998]}
AMENDED_PROBES = [256, 512, 1024, 2048, 3072, 4096]   # test-run probes (amendment before the lock); plus episode 0
VOLATILE = {"elapsed_seconds", "training_seconds", "probe_seconds", "started_utc", "completed_utc", "utc", "resumes",
            "resumed", "artifact_hashes", "seconds", "wall"}


class Interrupted(RuntimeError):
    """Deliberate interruption used only by the resumption checks."""


# ------------------------------------------------------------------ basics

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise AssertionError(message)


def log_event(output, **event):
    path = Path(output) / "attempts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(utc=now(), **event), default=_default) + "\n")


def array_digest(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(str((k, a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_arrays(path, meta=None, **arrays):
    """np.savez (atomic) + sha256 sidecar of the file and of the array contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    info = dict(digest=array_digest(arrays), file_sha256=sha(path), keys=sorted(arrays), meta=meta or {})
    write_json(path.with_suffix(".sha256.json"), info)
    return info


def load_arrays(path, strict=True):
    """Arrays verified against the sidecar (content digest and file hash); None if absent/corrupt and not strict."""
    path = Path(path)
    side = path.with_suffix(".sha256.json")
    try:
        info = read_json(side)
        if sha(path) != info["file_sha256"]:
            raise ValueError("file hash mismatch")
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        if array_digest(arrays) != info["digest"]:
            raise ValueError("content digest mismatch")
        return arrays, info
    except Exception as error:  # noqa: BLE001
        if strict:
            raise ValueError(f"unverifiable array file {path}: {error!r}") from error
        return None


def quarantine(output, path, reason):
    path = Path(path)
    if not path.exists():
        return
    target = Path(output) / "quarantine" / f"{int(time.time() * 1000)}_{path.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    log_event(output, event="quarantined", path=str(path), to=str(target), reason=reason)


def worker_init(threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def fresh(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    return path


# ------------------------------------------------------------------ config

def cfg_tag(knobs):
    return f"lr{knobs['lr']:g}_U{knobs['U']}_d{knobs['ema_decay']:g}"


def grid_configs():
    return [dict(lr=lr, U=u, ema_decay=d) for lr, u, d in itertools.product(GRID["lr"], GRID["U"], GRID["ema_decay"])]


FIXED_LEARNER = dict(members=5, hidden=128, minibatch=64, grad_clip=5.0, bootstrap_p=0.5, aux_weight=1.0,
                     plan_samples=256, capacity=1024)


def validate_config(c):
    expected = {"study", "protocol", "phase", "world", "contexts", "seeds", "episodes", "checkpoint_every", "panel_size",
                "oracle", "learner", "grid", "knobs", "arms", "analysis", "workers", "threads", "device",
                "stall_seconds", "stage_attempts", "notes"}
    if c.get("phase") == "test":
        expected |= {"probe_episodes", "dev_results"}   # amendment before the lock; selection provenance
    if set(c) != expected:
        raise ValueError(f"invalid configuration fields: {sorted(set(c) ^ expected)}")
    if c["protocol"] != "docs/commons_d1_protocol.md" or c["phase"] not in ("smoke", "dev", "test"):
        raise ValueError("protocol/phase")
    if c["world"] != dict(separability="reports/commons_d0b/separability.json",
                          spec="reports/commons_d0b/world_spec.json", key=E.WORLD_KEY):
        raise ValueError("the world is the accepted D0b world")
    if c["contexts"] != list(E.TRAINING_CONTEXTS):
        raise ValueError("the 5 D2 training combinations are fixed")
    if c["learner"] != FIXED_LEARNER:
        raise ValueError("learner architecture is fixed by the protocol")
    if c["grid"] != GRID:
        raise ValueError("the development grid is fixed by the protocol")
    o = c["oracle"]
    if set(o) != {"draws", "stream_chunk", "floor_draws"} or o["draws"] != 2 ** 14 or o["floor_draws"] != 256:
        raise ValueError("O uses 2^14 draws; the planning floor uses M = 256")
    if c["device"] != "cpu" or c["threads"] != 1 or not 1 <= c["workers"] <= 6:
        raise ValueError("CPU, one thread and at most six workers are required")
    a = dict(c["analysis"])
    if c["phase"] == "test":
        boot = {k: a.pop(k, None) for k in ("bootstrap_resamples", "analysis_seed")}
        if boot["bootstrap_resamples"] != 20000 or type(boot["analysis_seed"]) is not int:
            raise ValueError("test analysis: 20,000 paired seed resamples and a declared integer analysis seed")
    if a != dict(competent_mean=0.015, competent_worst=0.025, consistency_seeds=10, budget_cost=0.005,
                 endpoint=c["episodes"]):
        raise ValueError("pre-declared analysis constants are fixed")
    if c["episodes"] % c["checkpoint_every"] or c["episodes"] % o["stream_chunk"]:
        raise ValueError("checkpoints and oracle chunks must tile the stream")
    phase = c["phase"]
    if phase == "smoke":
        require(c["seeds"] == [SMOKE_SEED] and c["episodes"] == 256, "smoke: seed 21099, 256 episodes")
    elif phase == "dev":
        require(c["seeds"] == DEV_SEEDS and c["episodes"] == 4096 and c["checkpoint_every"] == 256
                and c["panel_size"] == 512, "dev: seeds 21001-21002, N = 4096, checkpoints every 256, panel 512")
        require(c["knobs"] is None and c["arms"] == list(BOUNDED), "dev: the grid is run for MF and MB")
    else:
        require(c["seeds"] == TEST_SEEDS and c["episodes"] == 4096 and c["checkpoint_every"] == 256
                and c["panel_size"] == 512, "test: seeds 21101-21112, N = 4096, checkpoints every 256, panel 512")
        require(c["arms"] == list(ARMS), "test: all four arms")
        require(c["probe_episodes"] == AMENDED_PROBES, "test: greedy probes at the amended episodes (protocol amendment)")
        dv = c["dev_results"]
        require(set(dv) == {"path", "sha256"} and dv["path"] == "reports/commons_d1/dev/dev_results.json",
                "test: dev_results = {path, sha256}")
        require(sha(ROOT / dv["path"]) == dv["sha256"], "dev_results.json differs from the hash recorded in the config")
        sel = read_json(ROOT / dv["path"])["selection"]
        for fam in BOUNDED:
            require(c["knobs"][fam] == sel[fam]["selected"], f"test knobs for {fam} differ from the development selection")
    if phase in ("smoke", "test"):
        if not isinstance(c["knobs"], dict) or set(c["knobs"]) != {"MF", "MB"}:
            raise ValueError("knobs per family are required")
        for fam, k in c["knobs"].items():
            if k is None or any(k.get(key) is None for key in ("lr", "U", "ema_decay")):
                raise ValueError(f"knobs for {fam} are unfilled (fill from the development selection before locking)")
            if set(k) != {"lr", "U", "ema_decay"} or k not in grid_configs():
                raise ValueError(f"knobs for {fam} must be one grid configuration")


def load_config(path):
    c = read_json(path)
    validate_config(c)
    return c


def hyper_for(config, arm, knobs, capacity=None):
    fam = arm.split("-")[0]
    lr = dict(config["learner"])
    return fam, L.Hyper(lr=knobs["lr"], U=knobs["U"], ema_decay=knobs["ema_decay"], members=lr["members"],
                        hidden=lr["hidden"], minibatch=lr["minibatch"], grad_clip=lr["grad_clip"],
                        bootstrap_p=lr["bootstrap_p"], aux_weight=lr["aux_weight"], plan_samples=lr["plan_samples"],
                        capacity=capacity or lr["capacity"], bounded=not arm.endswith("unbounded"),
                        total_episodes=config["episodes"])


def probe_schedule(config):
    """Episodes (after training on that many) at which the greedy probe runs; 0 is always probed.
    Smoke/dev: every checkpoint_every episodes. Test: the amended set."""
    eps = config.get("probe_episodes")
    if eps is None:
        eps = list(range(config["checkpoint_every"], config["episodes"] + 1, config["checkpoint_every"]))
    require(eps == sorted(set(eps)) and eps[-1] == config["episodes"] and eps[0] > 0, "probe schedule must end at N")
    return [0] + list(eps)


def learner_seeds(seed, context, family):
    """Shared by the bounded and unbounded arm of a family and by every grid configuration."""
    return {k: seed_for("commons_d1_learner", seed, int(context), family, k) for k in L.RNG_STREAMS}


# ------------------------------------------------------------------ identity / lock

def source_manifest():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "src/acp_cl/commons").glob("*.py"))}


def lock_payload(config, config_path):
    scripts = {name: sha(ROOT / name) for name in SCRIPT_FILES}
    source = source_manifest()
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   platform=platform.platform(), device=config["device"], threads=config["threads"],
                   workers=config["workers"], deterministic=True)
    identity = dict(config_sha256=digest(config), config_file_sha256=sha(config_path),
                    protocol_sha256=sha(ROOT / config["protocol"]), scripts_sha256=digest(scripts),
                    source_sha256=digest(source), runtime_sha256=digest(runtime),
                    world_spec_sha256=sha(ROOT / config["world"]["spec"]))
    return dict(identity=identity, config=copy.deepcopy(config),
                config_path=Path(config_path).resolve().relative_to(ROOT).as_posix(),
                script_files=scripts, source_files=source, runtime=runtime)


def lock(config, config_path, output):
    """Write (or re-check) the manifest that every result must match."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = lock_payload(config, config_path)
    path = output / "manifest.json"
    if path.exists():
        old = read_json(path)
        if old["identity"] != payload["identity"]:
            changed = [k for k in payload["identity"] if payload["identity"][k] != old["identity"].get(k)]
            raise ValueError(f"an existing manifest locks different files: {changed}")
        return old
    manifest = dict(label=LABEL, phase=config["phase"], **payload, locked_utc=now(),
                    fit_tasks=[list(t) for t in fit_tasks(config)])
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
        changed = [k for k in live["identity"] if live["identity"][k] != manifest["identity"].get(k)]
        raise ValueError(f"live files differ from the lock: {changed}")
    if sha(Path(output) / "protocol_at_lock.md") != manifest["identity"]["protocol_sha256"]:
        raise ValueError("archived protocol changed")
    return manifest


# ------------------------------------------------------------------ tasks

def fit_tasks(config, knobs_by_family=None, arms=None):
    """(arm, knobs, seed, context) for the phase; dev: the whole grid for the bounded arms."""
    tasks = []
    arms = arms or config["arms"]
    for arm in arms:
        fam = arm.split("-")[0]
        if knobs_by_family is not None:
            configs = [knobs_by_family[fam]]
        elif config["phase"] == "dev":
            configs = grid_configs()
        else:
            configs = [config["knobs"][fam]]
        for knobs in configs:
            for seed in config["seeds"]:
                for ctx in config["contexts"]:
                    tasks.append((arm, cfg_tag(knobs), seed, ctx))
    return tasks


def knobs_from_tag(tag):
    for k in grid_configs():
        if cfg_tag(k) == tag:
            return k
    raise ValueError(tag)


def fit_dir(output, arm, tag, seed, ctx, sub="fits"):
    return Path(output) / sub / f"{arm}__{tag}__s{seed}__{ctx}"


def oracle_tasks(config):
    tasks = [("panel", None, ctx, None) for ctx in config["contexts"]]
    step = config["oracle"]["stream_chunk"]
    for seed in config["seeds"]:
        for ctx in config["contexts"]:
            for start in range(0, config["episodes"], step):
                tasks.append(("stream", seed, ctx, start))
    return tasks


def oracle_path(output, kind, seed, ctx, start):
    if kind == "panel":
        return Path(output) / "oracle" / f"panel_{ctx}.npz"
    return Path(output) / "oracle" / f"stream_s{seed}_{ctx}_{start:05d}.npz"


def run_oracle(config, identity, output, kind, seed, ctx, start):
    path = oracle_path(output, kind, seed, ctx, start)
    if path.exists():
        loaded = load_arrays(path, strict=False)
        if loaded is not None and loaded[1]["meta"].get("identity") == identity:
            return dict(status="verified_existing", seconds=0.0)
        quarantine(output, path, "unverifiable or identity mismatch")
        quarantine(output, path.with_suffix(".sha256.json"), "sidecar of quarantined file")
    params, _ = E.load_world(ROOT)
    c = E.context_of(ctx)
    t0 = time.perf_counter()
    if kind == "panel":
        arrays = E.panel_tables(params, c, config["panel_size"], config["oracle"]["draws"], config["oracle"]["floor_draws"])
    else:
        stop = start + config["oracle"]["stream_chunk"]
        arrays = E.stream_tables(params, c, seed, config["episodes"], start, stop, config["oracle"]["draws"])
    seconds = time.perf_counter() - t0
    save_arrays(path, meta=dict(identity=identity, kind=kind, seed=seed, context=ctx, start=start, seconds=seconds), **arrays)
    return dict(status="completed", seconds=seconds)


# ------------------------------------------------------------------ fits

def checkpoint_save(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
    write_json(path.with_suffix(".sha256.json"), dict(file_sha256=sha(path), episodes_done=obj["record"]["episodes_done"]))


def checkpoint_load(output, path, identity):
    path = Path(path)
    side = path.with_suffix(".sha256.json")
    if not path.exists():
        return None
    try:
        info = read_json(side)
        if sha(path) != info["file_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj["identity"] != identity:
            raise ValueError("checkpoint identity mismatch")
        return obj
    except Exception as error:  # noqa: BLE001 - corrupt checkpoint: keep it, restart the fit
        quarantine(output, path, f"unloadable checkpoint: {error!r}")
        quarantine(output, side, "sidecar of quarantined checkpoint")
        return None


def chain(prev, *arrays):
    h = hashlib.sha256(prev.encode())
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(a.dtype.str.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def verify_result(directory, identity):
    record = read_json(Path(directory) / "result.json")
    if record["identity"] != identity:
        raise ValueError(f"result identity mismatch: {directory}")
    for name, value in record["artifact_hashes"].items():
        if sha(Path(directory) / name) != value:
            raise ValueError(f"artifact changed: {directory}/{name}")
    return record


def probe(learner, panel, fam):
    """Greedy (ensemble-mean, EMA) predictions on the panel; must not change the learner."""
    before = (learner.hashes(), json.dumps(learner.rng_state(), sort_keys=True, default=str))
    g = learner.greedy(panel.reserves, panel.cue, eps=panel.eps)
    out = dict(action=g["action"].astype(np.int8), value=g["value"].astype(np.float32),
               entity=g["entity"].astype(np.float32))
    if fam == "MF":
        out["aux"] = g["aux"].astype(np.float32)
    else:
        n = panel.size
        acts = np.repeat(np.arange(L.ACTIONS)[None], n, 0).reshape(-1)
        out["hazard_nll"] = np.float64(learner.hazard_nll(
            np.repeat(panel.reserves, L.ACTIONS, 0), np.repeat(panel.cue, L.ACTIONS, 0), acts,
            np.repeat(panel.y_supply, L.ACTIONS, 0), panel.fail.reshape(-1, 3)))
        out.update({k: v.astype(np.float32) for k, v in learner.tables().items()})
    after = (learner.hashes(), json.dumps(learner.rng_state(), sort_keys=True, default=str))
    require(before == after, "the greedy probe changed the learner")
    return out


def run_fit(config, identity, output, arm, tag, seed, ctx, sub="fits", interrupt_after=None, capacity=None,
            track_ema=True, forced_actions=None, keep_checkpoint=True):
    """One arm in one stationary context for N episodes; resumable bit-for-bit."""
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = fit_dir(output, arm, tag, seed, ctx, sub)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, ckpt_path = directory / "result.json", directory / "checkpoint.pt"
    if result_path.exists():
        try:
            verify_result(directory, identity)
            return dict(status="verified_existing", arm=arm, tag=tag, seed=seed, context=ctx, seconds=0.0)
        except Exception as error:  # noqa: BLE001
            quarantine(output, result_path, f"unverifiable result: {error!r}")
    knobs = knobs_from_tag(tag)
    fam, hyper = hyper_for(config, arm, knobs, capacity)
    params, _ = E.load_world(ROOT)
    c = E.context_of(ctx)
    n, every = config["episodes"], config["checkpoint_every"]
    probe_at = set(probe_schedule(config))
    stream = E.Stream(params, c, seed, n)
    panel = E.Panel(params, c, config["panel_size"], hyper.plan_samples, with_draws=(fam == "MB"))
    learner = L.make_learner(fam, hyper, learner_seeds(seed, c, fam), track_ema=track_ema)
    spec = dict(arm=arm, family=fam, tag=tag, knobs=knobs, seed=seed, context=ctx, context_index=c,
                hyper=hyper.__dict__, episodes=n, checkpoint_every=every, probe_episodes=sorted(probe_at), track_ema=track_ema,
                forced_actions=forced_actions is not None, capacity_override=capacity)
    saved = checkpoint_load(output, ckpt_path, identity)
    resumed = False
    if saved is not None and saved["record"]["spec"] == spec:
        learner.load(saved["learner"])
        record, trace, probes = saved["record"], saved["trace"], saved["probes"]
        require(learner.hashes() == record["checkpoint_hashes"], "checkpoint learner state changed")
        record["resumes"].append(dict(utc=now(), episodes_done=record["episodes_done"]))
        resumed = True
    else:
        record = dict(identity=identity, label=LABEL, spec=spec, episodes_done=0, resumes=[], started_utc=now(),
                      training_seconds=0.0, probe_seconds=0.0, elapsed_seconds=0.0,
                      stream_fingerprint=stream.fingerprint(), decisions_chain="", sensors_chain="", outcomes_chain="",
                      initial_hashes=learner.hashes(), max_replay=0)
        trace = dict(action=np.zeros(n, np.int8), member=np.zeros(n, np.int8), realised=np.zeros(n, bool),
                     replay_size=np.zeros(n, np.int16))
        probes = []
    require(record["stream_fingerprint"] == stream.fingerprint(), "stream draws changed")
    started = time.perf_counter()
    if record["episodes_done"] == 0 and not probes:
        t0 = time.perf_counter()
        probes.append(dict(episode=0, **probe(learner, panel, fam)))
        record["probe_seconds"] += time.perf_counter() - t0
    for k in range(record["episodes_done"], n):
        t0 = time.perf_counter()
        decision = stream.decision(k)
        action = learner.act(decision)
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
        require(learner.replay.size <= learner.replay.capacity, "replay exceeded its capacity")
        if hyper.bounded:
            require(learner.replay.size <= hyper.capacity, "bounded replay exceeded its budget")
        record["max_replay"] = max(record["max_replay"], learner.replay.size)
        record["episodes_done"] = k + 1
        if k + 1 in probe_at:
            t0 = time.perf_counter()
            probes.append(dict(episode=k + 1, **probe(learner, panel, fam)))
            record["probe_seconds"] += time.perf_counter() - t0
        if (k + 1) % every == 0:
            record["elapsed_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            record["checkpoint_hashes"] = learner.hashes()
            checkpoint_save(ckpt_path, dict(identity=identity, learner=learner.state(), record=record, trace=trace,
                                            probes=probes))
        if interrupt_after is not None and k + 1 == interrupt_after:
            raise Interrupted(f"deliberate interruption after episode {k + 1}")
    record["elapsed_seconds"] += time.perf_counter() - started
    arrays = {k: v for k, v in trace.items()}
    arrays["probe_episode"] = np.array([p["episode"] for p in probes], np.int32)
    for key in probes[0]:
        if key != "episode":
            arrays[f"probe_{key}"] = np.stack([np.asarray(p[key]) for p in probes])
    info = save_arrays(directory / "trace.npz", meta=dict(arm=arm, tag=tag, seed=seed, context=ctx), **arrays)
    record["final_hashes"] = learner.hashes()
    record["max_replay"] = int(max(record["max_replay"], learner.replay.max_size))
    record["replay"] = dict(capacity=learner.replay.capacity, bounded=hyper.bounded, max_size=learner.replay.max_size,
                            seen=learner.replay.seen, evictions=learner.replay.evictions,
                            record_bytes=learner.replay.record_bytes())
    record["optimizer_steps"] = learner.steps
    record["macs"] = dict(learner.macs, total=sum(learner.macs.values()))
    record["memory_bytes"] = learner.memory_bytes()
    record["parameters"] = learner.n_params()
    record["trace_digest"] = info["digest"]
    record["resumed"] = resumed
    record["completed_utc"] = now()
    record["artifact_hashes"] = {"trace.npz": sha(directory / "trace.npz"), "trace.sha256.json": sha(directory / "trace.sha256.json")}
    if keep_checkpoint:
        record["checkpoint_hashes"] = learner.hashes()
        checkpoint_save(ckpt_path, dict(identity=identity, learner=learner.state(), record=record, trace=trace, probes=probes))
        record["artifact_hashes"]["checkpoint.pt"] = sha(ckpt_path)
    write_json(result_path, record)
    return dict(status="completed", arm=arm, tag=tag, seed=seed, context=ctx, seconds=record["elapsed_seconds"])


# ------------------------------------------------------------------ pool

def estimated_cost(config, kind, payload):
    n = config["episodes"] / 4096
    if kind == "fit":
        knobs = knobs_from_tag(payload["tag"])
        fam = payload["arm"].split("-")[0]
        probes = len(probe_schedule(config))
        if fam == "MB":
            return n * 4096 * (0.0055 + 0.0027 * knobs["U"]) + probes * 11 * config["panel_size"] / 512
        return n * 4096 * (0.0013 + 0.0021 * knobs["U"])
    if payload["kind"] == "panel":
        return config["panel_size"] * 0.23
    return config["oracle"]["stream_chunk"] * 0.075


def dispatch(kind, payload):
    if kind == "fit":
        return run_fit(**payload)
    return run_oracle(**payload)


def job_label(kind, payload):
    if kind == "fit":
        return dict(kind="fit", arm=payload["arm"], tag=payload["tag"], seed=payload["seed"], context=payload["ctx"])
    return dict(kind="oracle", part=payload["kind"], seed=payload["seed"], context=payload["ctx"], start=payload["start"])


def kill_pool(pool):
    processes = list(getattr(pool, "_processes", {}).values())
    for p in processes:
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass
    return len(processes)


def run_pool(config, output, jobs, stage):
    """At most config['workers'] single-thread processes; a broken or stalled pool is killed, every
    failure is logged, and the stage retried with a fresh pool (finished tasks only verify)."""
    jobs = sorted(jobs, key=lambda job: -estimated_cost(config, *job))
    for attempt in range(1, config["stage_attempts"] + 1):
        log_event(output, event="stage_start", stage=stage, attempt=attempt, jobs=len(jobs))
        started, failures, statuses = time.perf_counter(), [], []
        if config["workers"] == 1:
            for kind, payload in jobs:
                try:
                    result = dispatch(kind, payload)
                    statuses.append(dict(job_label(kind, payload), status=result.get("status"), seconds=result.get("seconds")))
                except Exception as error:  # noqa: BLE001
                    failures.append(dict(**job_label(kind, payload), error=repr(error)))
                    log_event(output, event="task_failed", stage=stage, attempt=attempt, **job_label(kind, payload),
                              error=repr(error), traceback=traceback.format_exc())
        else:
            pool = ProcessPoolExecutor(max_workers=config["workers"], initializer=worker_init, initargs=(config["threads"],))
            futures = {pool.submit(dispatch, kind, payload): (kind, payload) for kind, payload in jobs}
            pending, stalled = set(futures), False
            while pending:
                done, pending = wait(pending, timeout=config["stall_seconds"], return_when=FIRST_COMPLETED)
                if not done:
                    stalled = True
                    killed = kill_pool(pool)
                    log_event(output, event="pool_stalled", stage=stage, attempt=attempt, pending=len(pending),
                              killed_processes=killed, stall_seconds=config["stall_seconds"])
                    for future in pending:
                        failures.append(dict(**job_label(*futures[future]), error="pool stalled; killed"))
                    break
                for future in done:
                    kind, payload = futures[future]
                    label = job_label(kind, payload)
                    try:
                        result = future.result()
                        statuses.append(dict(label, status=result.get("status"), seconds=result.get("seconds")))
                        log_event(output, event="task_done", stage=stage, attempt=attempt, **label,
                                  status=result.get("status"), seconds=result.get("seconds"))
                        print(json.dumps(dict(stage=stage, done=len(statuses), total=len(jobs), **label,
                                              status=result.get("status"), seconds=round(result.get("seconds") or 0, 1),
                                              elapsed=round(time.perf_counter() - started))), flush=True)
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
    return [("fit", dict(config=config, identity=identity, output=str(output), arm=arm, tag=tag, seed=seed, ctx=ctx, sub=sub))
            for arm, tag, seed, ctx in tasks]


def oracle_jobs(config, identity, output):
    return [("oracle", dict(config=config, identity=identity, output=str(output), kind=kind, seed=seed, ctx=ctx, start=start))
            for kind, seed, ctx, start in oracle_tasks(config)]


# ------------------------------------------------------------------ integrity

def crc_scan(output, skip=("quarantine",)):
    rows, bad = 0, []
    for path in sorted(Path(output).rglob("*")):
        if not path.is_file() or path.suffix not in (".npz", ".pt", ".zip"):
            continue
        if any(part in skip for part in path.relative_to(output).parts):
            continue
        rows += 1
        try:
            with zipfile.ZipFile(path) as archive:
                first = archive.testzip()
            if first is not None:
                bad.append(dict(path=path.relative_to(output).as_posix(), member=first))
        except Exception as error:  # noqa: BLE001
            bad.append(dict(path=path.relative_to(output).as_posix(), error=repr(error)))
    return dict(archives=rows, failures=bad, passed=not bad, utc=now())


def verify_all(config, output, identity, tasks, sub="fits"):
    fits = []
    for arm, tag, seed, ctx in tasks:
        d = fit_dir(output, arm, tag, seed, ctx, sub)
        rec = verify_result(d, identity)
        load_arrays(d / "trace.npz")
        fits.append(dict(arm=arm, tag=tag, seed=seed, context=ctx, path=(d / "result.json").relative_to(output).as_posix(),
                         sha256=sha(d / "result.json"), elapsed_seconds=rec["elapsed_seconds"]))
    oracle = []
    for kind, seed, ctx, start in oracle_tasks(config):
        p = oracle_path(output, kind, seed, ctx, start)
        arrays, info = load_arrays(p)
        require(info["meta"]["identity"] == identity, f"oracle identity mismatch {p}")
        oracle.append(dict(path=p.relative_to(output).as_posix(), sha256=sha(p)))
    return fits, oracle


# ------------------------------------------------------------------ analysis helpers

class Tables:
    """O's panel and stream tables, loaded and verified once."""

    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.panel, self.stream = {}, {}

    def panel_tab(self, ctx):
        if ctx not in self.panel:
            arrays, _ = load_arrays(oracle_path(self.output, "panel", None, ctx, None))
            self.panel[ctx] = arrays
        return self.panel[ctx]

    def stream_tab(self, seed, ctx):
        key = (seed, ctx)
        if key not in self.stream:
            parts = []
            for start in range(0, self.config["episodes"], self.config["oracle"]["stream_chunk"]):
                parts.append(load_arrays(oracle_path(self.output, "stream", seed, ctx, start))[0])
            self.stream[key] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            require(np.array_equal(self.stream[key]["index"], np.arange(self.config["episodes"])), "stream tables do not tile")
        return self.stream[key]


def panel_regret(tab, c, actions):
    """Regret on the panel: O's action by the selection table, values by the independent scoring table."""
    rows = np.arange(len(actions))
    score = tab["score"][:, c]
    a_o = tab["select"][:, c].argmax(1)
    return score[rows, a_o] - score[rows, np.asarray(actions, np.int64)]


def stream_regret(stab, actions):
    rows = np.arange(len(actions))
    a_o = stab["select"].argmax(1)
    return stab["score"][rows, a_o] - stab["score"][rows, np.asarray(actions, np.int64)]


def context_scale(tab, c):
    """O - none, O - random and the relevance of each factor (cross-context policy regret) on the panel."""
    rows = np.arange(len(tab["score"]))
    score = tab["score"][:, c]
    a_o = tab["select"][:, c].argmax(1)
    v_o = score[rows, a_o]
    out = dict(o_value=float(v_o.mean()), none_value=float(score[:, 0].mean()),
               o_minus_none=float((v_o - score[:, 0]).mean()), o_minus_random=float((v_o[:, None] - score).mean()),
               o_value_se=float(np.sqrt((tab["score_se"][rows, c, a_o] ** 2).mean() / len(rows))))
    for f, bit in E.FACTOR_BITS.items():
        other = tab["select"][:, c ^ bit].argmax(1)
        out[f"relevance_{f}"] = float((v_o - score[rows, other]).mean())
    floor = tab["floor_select"].argmax(1)
    out["planning_floor_M256"] = float((v_o - score[rows, floor]).mean())
    return out


def log_loss(p, y):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_metrics(config, tables, output, arm, tag, seed, ctx, sub="fits"):
    d = fit_dir(output, arm, tag, seed, ctx, sub)
    rec = read_json(d / "result.json")
    tr, _ = load_arrays(d / "trace.npz")
    c = E.context_of(ctx)
    ptab = tables.panel_tab(ctx)
    stab = tables.stream_tab(seed, ctx)
    rows = np.arange(config["panel_size"])
    curve = [float(panel_regret(ptab, c, a).mean()) for a in tr["probe_action"]]
    online = stream_regret(stab, tr["action"])
    block = config["checkpoint_every"]
    fam = arm.split("-")[0]
    alive = (ptab["fail"] == SURVIVED).astype(np.float64)                 # (512, 13, 3)
    ent = tr["probe_entity"][-1]
    greedy = tr["probe_action"][-1].astype(np.int64)
    score = ptab["score"][:, c]
    pred = tr["probe_value"][-1].astype(np.float64)
    m = dict(arm=arm, family=fam, tag=tag, seed=seed, context=ctx,
             regret_endpoint=curve[-1], regret_curve=curve, curve_episodes=tr["probe_episode"].tolist(),
             regret_endpoint_se_states=float(panel_regret(ptab, c, greedy).std(ddof=1) / np.sqrt(len(rows))),
             greedy_equals_O_fraction=float((greedy == ptab["select"][:, c].argmax(1)).mean()),
             cumulative_online_regret=float(online.sum()), online_regret_per_episode=float(online.mean()),
             online_regret_blocks=[float(x.mean()) for x in online.reshape(-1, block)],
             realised_survival=float(tr["realised"].mean()),
             entity_logloss_endpoint=log_loss(ent, alive),
             value_error_greedy_mean=float((pred[rows, greedy] - score[rows, greedy]).mean()),
             value_rmse_all_actions=float(np.sqrt(((pred - score) ** 2).mean())),
             value_mean_error_all_actions=float((pred - score).mean()),
             reliability=reliability(pred, score),
             wall_seconds=rec["elapsed_seconds"], training_seconds=rec["training_seconds"], probe_seconds=rec["probe_seconds"],
             macs_train=rec["macs"]["train"], macs_act=rec["macs"]["act"], macs_probe=rec["macs"]["probe"],
             bytes_total=rec["memory_bytes"]["total"], bytes_replay_peak=rec["memory_bytes"]["replay_peak"],
             max_replay=rec["max_replay"], optimizer_steps=rec["optimizer_steps"])
    if fam == "MB":
        mu, var = E.true_sensor_moments(E.load_world(ROOT)[0], c)
        sm, sl = tr["probe_supply_mean"][-1], tr["probe_supply_logvar"][-1]
        m.update(hazard_nll_endpoint=float(tr["probe_hazard_nll"][-1]),
                 supply_mean_rmse=float(np.sqrt(((sm - mu) ** 2).mean())),
                 supply_sd_ratio_mean=float(np.exp(0.5 * sl).mean() / np.sqrt(var)),
                 supply_kl_per_cell=float((0.5 * (np.exp(sl) / var + (sm - mu) ** 2 / var - 1 + np.log(var) - sl)).mean()),
                 planning_error_greedy_mean=m["value_error_greedy_mean"])
    else:
        aux = tr["probe_aux"][-1]                                          # (512, 13, 3, 3)
        m["aux_logloss_h4_h8_h12"] = [log_loss(aux[..., j], (ptab["fail"] > h).astype(np.float64))
                                      for j, h in enumerate((4, 8, 12))]
    return m


def reliability(pred, score, bins=10):
    """Per predicted-value bin [0, .1), ..., [.9, 1]: count, sum of predictions, sum of O's values."""
    b = np.clip((pred * bins).astype(np.int64), 0, bins - 1).ravel()
    return [np.bincount(b, minlength=bins).tolist(), np.bincount(b, pred.ravel(), bins).tolist(),
            np.bincount(b, score.ravel(), bins).tolist()]


def mean_se(x):
    x = np.asarray(x, np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def select_config(rows, family, endpoint_key="regret_endpoint"):
    """Protocol rule: lowest dev mean of greedy-probe regret at N; if several configurations are within
    one paired SE of the best (paired over seed x context cells), take the cheapest (fewest training
    multiply-adds, i.e. smallest U); ties in cost are broken by the lower mean."""
    by = {}
    for r in rows:
        if r["family"] == family and r["arm"] == family:
            by.setdefault(r["tag"], {})[(r["seed"], r["context"])] = r
    cells = sorted(next(iter(by.values())))
    for tag, cellmap in by.items():
        require(sorted(cellmap) == cells, f"incomplete dev cells for {family} {tag}")
    means = {tag: float(np.mean([by[tag][c][endpoint_key] for c in cells])) for tag in by}
    best = min(means, key=lambda t: (means[t], t))
    table = []
    for tag in sorted(by):
        diff = np.array([by[tag][c][endpoint_key] - by[best][c][endpoint_key] for c in cells])
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if tag != best else 0.0
        cost = float(np.mean([by[tag][c]["macs_train"] for c in cells]))
        table.append(dict(tag=tag, mean=means[tag], diff_vs_best=float(diff.mean()), paired_se=se,
                          within_one_paired_se=bool(tag == best or diff.mean() <= se), train_macs=cost,
                          wall_seconds=float(np.mean([by[tag][c]["wall_seconds"] for c in cells]))))
    within = [t for t in table if t["within_one_paired_se"]]
    chosen = min(within, key=lambda t: (t["train_macs"], t["mean"], t["tag"]))
    return dict(family=family, best_tag=best, selected_tag=chosen["tag"], selected=knobs_from_tag(chosen["tag"]),
                cells=len(cells), table=table, within=[t["tag"] for t in within],
                rule="lowest mean regret at N; configurations within one paired SE (seed x context cells) of the best "
                     "-> the cheapest (training multiply-adds; ties -> lower mean)")


# ------------------------------------------------------------------ dev

def cmd_dev(args):
    config = load_config(args.config)
    output, report = Path(args.output), Path(args.report)
    reanalysis = None
    if args.reanalyze:
        # Re-analysis only: every stored result is verified against the STORED dev identity; only the script
        # (analysis code) may differ from the manifest, and that difference is recorded in the report.
        manifest = read_json(output / "manifest.json")
        identity = manifest["identity"]
        live = lock_payload(config, args.config)["identity"]
        changed = [k for k in live if live[k] != identity.get(k)]
        require(set(changed) <= {"scripts_sha256"}, f"re-analysis refused: {changed} differ from the dev manifest")
        reanalysis = dict(changed_from_manifest=changed, live_scripts=lock_payload(config, args.config)["script_files"],
                          note="fits were NOT re-run; results verified against the stored dev identity")
        log_event(output, event="dev_reanalyze", changed=changed)
    else:
        manifest = lock(config, args.config, output)
        identity = manifest["identity"]
        log_event(output, event="dev_start", identity=identity)
    grid = fit_tasks(config)
    t0 = time.perf_counter()
    if not args.reanalyze:
        run_pool(config, output, oracle_jobs(config, identity, output) + fit_jobs(config, identity, output, grid), "dev_grid")
    verify_all(config, output, identity, grid)
    tables = Tables(config, output)
    rows = [fit_metrics(config, tables, output, *t) for t in grid]
    selection = {fam: select_config(rows, fam) for fam in BOUNDED}
    if not args.reanalyze:
        write_json(output / "selection.json", dict(identity=identity, selection=selection, utc=now()))
    else:
        stored = read_json(output / "selection.json")["selection"]
        require({f: stored[f]["selected_tag"] for f in BOUNDED} == {f: selection[f]["selected_tag"] for f in BOUNDED},
                "re-analysis changed the selection")
    unb = []
    if not args.skip_unbounded:
        unb = [(UNBOUNDED[fam], cfg_tag(selection[fam]["selected"]), seed, ctx)
               for fam in BOUNDED for seed in config["seeds"] for ctx in config["contexts"]]
        if not args.reanalyze:
            run_pool(config, output, fit_jobs(config, identity, output, unb), "dev_unbounded")
        rows += [fit_metrics(config, tables, output, *t) for t in unb]
    if not args.reanalyze:
        check_lock(config, args.config, output)
    fits, oracle = verify_all(config, output, identity, grid + unb)
    crc = crc_scan(output)
    write_json(output / "crc_scan.json", crc)
    require(crc["passed"], f"CRC scan failed: {crc['failures']}")
    if not args.reanalyze:
        write_json(output / "completion.json", dict(identity=identity, fits=fits, oracle=oracle, crc=crc, utc=now(),
                                                    wall_seconds_this_invocation=time.perf_counter() - t0))
    dev_report(config, tables, rows, selection, report, identity, crc, len(fits), reanalysis)


def dev_report(config, tables, rows, selection, report, identity, crc, n_fits, reanalysis=None):
    report = Path(report)
    scale = {}
    for ctx in config["contexts"]:
        scale[ctx] = context_scale(tables.panel_tab(ctx), E.context_of(ctx))
    curve_eps = rows[0]["curve_episodes"]
    summary = {}
    for fam in BOUNDED:
        for tag in sorted({r["tag"] for r in rows if r["family"] == fam}):
            for arm in (fam, UNBOUNDED[fam]):
                rs = [r for r in rows if r["arm"] == arm and r["tag"] == tag]
                if not rs:
                    continue
                key = f"{arm}|{tag}"
                ctx_means = {ctx: float(np.mean([r["regret_endpoint"] for r in rs if r["context"] == ctx]))
                             for ctx in config["contexts"]}
                seed_avg = {str(s): float(np.mean([r["regret_endpoint"] for r in rs if r["seed"] == s])) for s in config["seeds"]}
                seed_worst = {str(s): float(max(r["regret_endpoint"] for r in rs if r["seed"] == s)) for s in config["seeds"]}
                curve = np.mean([r["regret_curve"] for r in rs], 0)
                summary[key] = dict(
                    arm=arm, tag=tag, cells=len(rs), regret_endpoint_mean=mean_se([r["regret_endpoint"] for r in rs]),
                    regret_by_context=ctx_means, seed_context_average=seed_avg, seed_worst_context=seed_worst,
                    competent_like_seeds=int(sum(seed_avg[str(s)] <= config["analysis"]["competent_mean"]
                                                 and seed_worst[str(s)] <= config["analysis"]["competent_worst"]
                                                 for s in config["seeds"])),
                    curve_episodes=curve_eps, regret_curve_mean=curve.tolist(),
                    cumulative_online_regret_mean=mean_se([r["cumulative_online_regret"] for r in rs]),
                    online_regret_per_episode_mean=float(np.mean([r["online_regret_per_episode"] for r in rs])),
                    realised_survival_mean=float(np.mean([r["realised_survival"] for r in rs])),
                    entity_logloss_endpoint=float(np.mean([r["entity_logloss_endpoint"] for r in rs])),
                    value_error_greedy_mean=float(np.mean([r["value_error_greedy_mean"] for r in rs])),
                    value_rmse_all_actions=float(np.mean([r["value_rmse_all_actions"] for r in rs])),
                    greedy_equals_O_fraction=float(np.mean([r["greedy_equals_O_fraction"] for r in rs])),
                    wall_seconds_mean=float(np.mean([r["wall_seconds"] for r in rs])),
                    training_seconds_mean=float(np.mean([r["training_seconds"] for r in rs])),
                    probe_seconds_mean=float(np.mean([r["probe_seconds"] for r in rs])),
                    macs_train=float(np.mean([r["macs_train"] for r in rs])), macs_act=float(np.mean([r["macs_act"] for r in rs])),
                    macs_probe=float(np.mean([r["macs_probe"] for r in rs])),
                    bytes_total=float(np.mean([r["bytes_total"] for r in rs])),
                    bytes_replay_peak=float(np.mean([r["bytes_replay_peak"] for r in rs])),
                    max_replay=int(max(r["max_replay"] for r in rs)))
                if fam == "MB":
                    for k in ("hazard_nll_endpoint", "supply_mean_rmse", "supply_sd_ratio_mean", "supply_kl_per_cell"):
                        summary[key][k] = float(np.mean([r[k] for r in rs]))
    # bounded - unbounded at the selected configurations (paired by seed x context)
    budget = {}
    for fam in BOUNDED:
        tag = cfg_tag(selection[fam]["selected"])
        b = {(r["seed"], r["context"]): r["regret_endpoint"] for r in rows if r["arm"] == fam and r["tag"] == tag}
        u = {(r["seed"], r["context"]): r["regret_endpoint"] for r in rows if r["arm"] == UNBOUNDED[fam] and r["tag"] == tag}
        if u:
            diffs = [b[k] - u[k] for k in sorted(u)]
            budget[fam] = dict(tag=tag, bounded_minus_unbounded=mean_se(diffs),
                               per_seed={str(s): float(np.mean([b[(s, c)] - u[(s, c)] for c in config["contexts"]]))
                                         for s in config["seeds"]})
    timing = dict(
        fits=n_fits,
        wall_seconds_by_family_U={f"{fam}_U{u}": float(np.mean([r["wall_seconds"] for r in rows
                                                                 if r["family"] == fam and knobs_from_tag(r["tag"])["U"] == u]))
                                  for fam in BOUNDED for u in GRID["U"]})
    out = dict(label="DEVELOPMENT PHASE (seeds 21001-21002); not a test result. " + LABEL, identity=identity,
               endpoint=config["episodes"], bar=dict(mean=config["analysis"]["competent_mean"], worst=config["analysis"]["competent_worst"]),
               context_scale=scale, selection=selection, summary=summary, budget_cost_dev=budget, timing=timing,
               crc=crc, reanalysis=reanalysis, utc=now())
    write_json(report / "dev_results.json", out)
    write_csv(report / "dev_fits.csv", [{k: v for k, v in r.items() if k not in ("regret_curve", "curve_episodes", "online_regret_blocks")}
                                        for r in rows])
    long = []
    for r in rows:
        for ep, v in zip(r["curve_episodes"], r["regret_curve"]):
            long.append(dict(arm=r["arm"], family=r["family"], tag=r["tag"], seed=r["seed"], context=r["context"], episode=ep, regret=v))
    write_csv(report / "dev_curves.csv", long)
    write_csv(report / "dev_summary.csv", [dict(key=k, **{kk: vv for kk, vv in v.items() if kk not in ("curve_episodes",)})
                                           for k, v in summary.items()])
    print(json.dumps(dict(selection={f: s["selected_tag"] for f, s in selection.items()},
                          best={f: s["best_tag"] for f, s in selection.items()}), indent=1), flush=True)
    return out


# ------------------------------------------------------------------ test run / replicate / analyze

def cmd_lock(args):
    config = load_config(args.config)
    require(config["phase"] in ("test", "smoke"), "lock is for the smoke or test config")
    manifest = lock(config, args.config, args.output)
    print(json.dumps(manifest["identity"], indent=1))


def cmd_run(args):
    config = load_config(args.config)
    output = Path(args.output)
    manifest = check_lock(config, args.config, output)
    identity = manifest["identity"]
    log_event(output, event="run_start", identity=identity)
    t0 = time.perf_counter()
    tasks = fit_tasks(config)
    statuses = run_pool(config, output, oracle_jobs(config, identity, output) + fit_jobs(config, identity, output, tasks), "run")
    check_lock(config, args.config, output)
    fits, oracle = verify_all(config, output, identity, tasks)
    crc = crc_scan(output)
    write_json(output / "crc_scan.json", crc)
    require(crc["passed"], f"CRC scan failed: {crc['failures']}")
    completion = dict(identity=identity, fits=fits, oracle=oracle, statuses=statuses, crc=crc, utc=now(),
                      wall_seconds_this_invocation=time.perf_counter() - t0)
    write_json(output / "completion.json", completion)
    log_event(output, event="run_complete", seconds=time.perf_counter() - t0)
    return completion


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in VOLATILE and not str(k).endswith("_seconds")}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def npz_equal(a, b):
    with np.load(a) as x, np.load(b) as y:
        if set(x.files) != set(y.files):
            return False
        return all(x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and x[k].tobytes() == y[k].tobytes() for k in x.files)


def compare_fits(main_dir, copy_dir):
    a, b = read_json(main_dir / "result.json"), read_json(copy_dir / "result.json")
    rows = dict(result_json_normalized=normalize(a) == normalize(b), trace_npz=npz_equal(main_dir / "trace.npz", copy_dir / "trace.npz"),
                final_hashes=a["final_hashes"] == b["final_hashes"], trace_digest=a["trace_digest"] == b["trace_digest"])
    if (main_dir / "checkpoint.pt").exists() and (copy_dir / "checkpoint.pt").exists():
        ca = torch.load(main_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        cb = torch.load(copy_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        rows["checkpoint_hashes"] = ca["record"]["checkpoint_hashes"] == cb["record"]["checkpoint_hashes"] == a["final_hashes"]
    return rows


def replicate_tasks(config):
    seeds = list(REPLICATE_SEEDS) if config["phase"] == "test" else list(config["seeds"])
    return [t for t in fit_tasks(config) if t[2] in seeds]


def cmd_replicate(args):
    config = load_config(args.config)
    output = Path(args.output)
    manifest = check_lock(config, args.config, output)
    identity = manifest["identity"]
    completion = read_json(output / "completion.json")
    for fit in completion["fits"]:
        require(sha(output / fit["path"]) == fit["sha256"], f"result.json changed since completion: {fit['path']}")
    for o in completion["oracle"]:
        require(sha(output / o["path"]) == o["sha256"], f"oracle file changed since completion: {o['path']}")
    tasks = fit_tasks(config)
    verify_all(config, output, identity, tasks)
    rtasks = replicate_tasks(config)
    t0 = time.perf_counter()
    run_pool(config, output, fit_jobs(config, identity, output, rtasks, sub="replicate"), "replicate")
    comparisons, mismatches = {}, []
    for arm, tag, seed, ctx in rtasks:
        rows = compare_fits(fit_dir(output, arm, tag, seed, ctx), fit_dir(output, arm, tag, seed, ctx, "replicate"))
        key = f"{arm}:{tag}:{seed}:{ctx}"
        comparisons[key] = rows
        mismatches += [f"{key}:{k}" for k, v in rows.items() if not v]
    crc = crc_scan(output)
    rep = dict(identity=identity, utc=now(), replicate_seeds=sorted({t[2] for t in rtasks}), fits_compared=len(rtasks),
               main_fits_reverified=len(tasks), comparisons=comparisons, mismatches=mismatches, crc=crc,
               passed=not mismatches and crc["passed"], seconds=time.perf_counter() - t0,
               normalized_away=sorted(VOLATILE) + ["*_seconds"])
    write_json(output / "replicate.json", rep)
    log_event(output, event="replicate_complete", passed=rep["passed"], mismatches=mismatches)
    if not rep["passed"]:
        raise RuntimeError(f"REPLICATE MISMATCH OR CRC FAILURE, do not analyze: {mismatches} {crc['failures']}")
    return rep


def boot_mean_ci(matrix_idx, values):
    """Percentile 95% CI of the mean over seeds under the shared (paired) seed resamples."""
    v = np.asarray(values, np.float64)
    means = v[matrix_idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_ci_safe(idx, v):
    return boot_mean_ci(idx, v) if idx is not None else None


def summarize(values, idx):
    v = np.asarray(values, np.float64)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, ci95=boot_ci_safe(idx, v))


def cmd_analyze(args, require_replicate=True):
    config = load_config(args.config)
    output, report = Path(args.output), Path(args.report)
    manifest = check_lock(config, args.config, output)
    identity = manifest["identity"]
    if require_replicate:
        rep = read_json(output / "replicate.json")
        require(rep["passed"] and rep["identity"] == identity, "the replicate pass must pass before analysis")
    tasks = fit_tasks(config)
    verify_all(config, output, identity, tasks)
    crc = crc_scan(output)
    require(crc["passed"], f"CRC scan failed: {crc['failures']}")
    tables = Tables(config, output)
    rows = [fit_metrics(config, tables, output, *t) for t in tasks]
    a = config["analysis"]
    seeds, ctxs = config["seeds"], config["contexts"]
    need = a["consistency_seeds"] if config["phase"] == "test" else len(seeds)
    B = a.get("bootstrap_resamples", 2000)
    rng = np.random.default_rng(a.get("analysis_seed", 0))
    idx = rng.integers(0, len(seeds), (B, len(seeds))) if len(seeds) > 1 else None
    by = {(r["arm"], r["seed"], r["context"]): r for r in rows}

    def per_seed(arm, key, agg=np.mean):
        return [float(agg([by[(arm, s, c)][key] for c in ctxs])) for s in seeds]

    readings = {}
    for arm in ARMS:
        avg, worst = per_seed(arm, "regret_endpoint"), per_seed(arm, "regret_endpoint", max)
        ps = {str(s): dict(context_average=avg[i], worst_context=worst[i],
                           worst_context_name=max(ctxs, key=lambda c: by[(arm, s, c)]["regret_endpoint"]),
                           mean_ok=bool(avg[i] <= a["competent_mean"]), worst_ok=bool(worst[i] <= a["competent_worst"]))
              for i, s in enumerate(seeds)}
        ok = sum(v["mean_ok"] and v["worst_ok"] for v in ps.values())
        readings[arm] = dict(
            per_seed=ps, seeds_meeting_mean=int(sum(v["mean_ok"] for v in ps.values())),
            seeds_meeting_worst=int(sum(v["worst_ok"] for v in ps.values())), seeds_meeting_both=int(ok), required=need,
            competent=bool(ok >= need), privileged=arm.endswith("unbounded"),
            context_average=summarize(avg, idx), worst_context=summarize(worst, idx),
            by_context={c: dict(summarize([by[(arm, s, c)]["regret_endpoint"] for s in seeds], idx),
                                min=float(min(by[(arm, s, c)]["regret_endpoint"] for s in seeds)),
                                max=float(max(by[(arm, s, c)]["regret_endpoint"] for s in seeds))) for c in ctxs})
    r2 = {}
    for fam in BOUNDED:
        diff = [float(np.mean([by[(fam, s, c)]["regret_endpoint"] - by[(UNBOUNDED[fam], s, c)]["regret_endpoint"] for c in ctxs]))
                for s in seeds]
        k = sum(v >= a["budget_cost"] for v in diff)
        r2[fam] = dict(per_seed=dict(zip(map(str, seeds), diff)), seeds_at_or_above=int(k), required=need,
                       budget_costs=bool(k >= need), bounded_minus_unbounded=summarize(diff, idx),
                       by_context={c: summarize([by[(fam, s, c)]["regret_endpoint"] - by[(UNBOUNDED[fam], s, c)]["regret_endpoint"]
                                                 for s in seeds], idx) for c in ctxs})
    competent = [arm for arm in BOUNDED if readings[arm]["competent"]]
    unb_competent = [arm for arm in UNBOUNDED.values() if readings[arm]["competent"]]
    if competent:
        implication = f"At least one bounded arm is competent: D2 proceeds with {competent} as cores."
    elif unb_competent:
        implication = ("No bounded arm is competent but an unbounded arm is: the 1,024-episode budget is too tight at this N; "
                       "D2 raises the budget to the smallest competent size, measured in a declared D1b development round.")
    else:
        implication = "No arm is competent: the learner family is the bottleneck; declared D1b architecture round."
    # ---- descriptive
    episodes = rows[0]["curve_episodes"]
    curves = {}
    for arm in ARMS:
        per = np.array([[by[(arm, s, c)]["regret_curve"] for c in ctxs] for s in seeds])   # (S, C, P)
        seed_avg = per.mean(1)
        curves[arm] = dict(episodes=episodes, mean=seed_avg.mean(0).tolist(),
                           ci95=[boot_ci_safe(idx, seed_avg[:, j]) for j in range(len(episodes))],
                           seed_min=seed_avg.min(0).tolist(), seed_max=seed_avg.max(0).tolist(),
                           by_context={c: per[:, i].mean(0).tolist() for i, c in enumerate(ctxs)})

    def desc(arm, key):
        return summarize(per_seed(arm, key), idx)
    secondary = {}
    for arm in ARMS:
        blocks = np.mean([by[(arm, s, c)]["online_regret_blocks"] for s in seeds for c in ctxs], 0)
        secondary[arm] = dict(
            cumulative_online_regret=desc(arm, "cumulative_online_regret"),
            online_regret_per_episode=desc(arm, "online_regret_per_episode"),
            online_regret_by_256_block=blocks.tolist(),
            realised_survival=desc(arm, "realised_survival"),
            greedy_equals_O_fraction=desc(arm, "greedy_equals_O_fraction"),
            entity_logloss_endpoint=desc(arm, "entity_logloss_endpoint"),
            value_optimism_greedy=desc(arm, "value_error_greedy_mean"),
            value_mean_error_all_actions=desc(arm, "value_mean_error_all_actions"),
            value_rmse_all_actions=desc(arm, "value_rmse_all_actions"),
            reliability=np.sum([np.array(by[(arm, s, c)]["reliability"]) for s in seeds for c in ctxs], 0).tolist(),
            costs={k: float(np.mean([by[(arm, s, c)][k] for s in seeds for c in ctxs]))
                   for k in ("wall_seconds", "training_seconds", "probe_seconds", "macs_train", "macs_act", "macs_probe",
                             "bytes_total", "bytes_replay_peak", "optimizer_steps")},
            max_replay=int(max(by[(arm, s, c)]["max_replay"] for s in seeds for c in ctxs)))
        if arm.startswith("MB"):
            for k in ("hazard_nll_endpoint", "supply_mean_rmse", "supply_sd_ratio_mean", "supply_kl_per_cell"):
                secondary[arm][k] = desc(arm, k)
        else:
            secondary[arm]["aux_logloss_h4_h8_h12"] = np.mean([by[(arm, s, c)]["aux_logloss_h4_h8_h12"]
                                                               for s in seeds for c in ctxs], 0).tolist()
    contrasts = {}
    for label, x, y, key in (("MF_minus_MB", "MF", "MB", "regret_endpoint"),
                             ("MFunb_minus_MBunb", "MF-unbounded", "MB-unbounded", "regret_endpoint"),
                             ("MF_minus_MB_cumulative_online", "MF", "MB", "cumulative_online_regret"),
                             ("MB_hazard_nll_bounded_minus_unbounded", "MB", "MB-unbounded", "hazard_nll_endpoint"),
                             ("MB_optimism_bounded_minus_unbounded", "MB", "MB-unbounded", "value_error_greedy_mean")):
        d = [float(np.mean([by[(x, s, c)][key] - by[(y, s, c)][key] for c in ctxs])) for s in seeds]
        contrasts[label] = dict(summarize(d, idx), seeds_positive=int(sum(v > 0 for v in d)))
    scale = {ctx: context_scale(tables.panel_tab(ctx), E.context_of(ctx)) for ctx in ctxs}
    out = dict(label=LABEL + (" SMOKE ONLY - NOT SCIENCE." if config["phase"] == "smoke" else ""), identity=identity,
               endpoint=config["episodes"], bar=dict(mean=a["competent_mean"], worst=a["competent_worst"], seeds=need),
               bootstrap=dict(resamples=B, analysis_seed=a.get("analysis_seed"),
                              unit="paired seed resamples (all arms and contrasts share the resample)"),
               R1=readings, R2=r2, competent_bounded_arms=competent, competent_unbounded_arms=unb_competent,
               implication=implication, curves=curves, secondary=secondary, contrasts=contrasts, context_scale=scale,
               probe_episodes=episodes, crc=crc, fits=len(rows), utc=now())
    write_json(report / "results.json", out)
    write_csv(report / "fits.csv", [{k: v for k, v in r.items()
                                     if k not in ("regret_curve", "curve_episodes", "online_regret_blocks", "reliability")}
                                    for r in rows])
    write_csv(report / "curves.csv", [dict(arm=r["arm"], seed=r["seed"], context=r["context"], episode=ep, regret=v)
                                      for r in rows for ep, v in zip(r["curve_episodes"], r["regret_curve"])])
    write_csv(report / "per_context.csv", [dict(arm=arm, context=c, **readings[arm]["by_context"][c])
                                           for arm in ARMS for c in ctxs])
    write_csv(report / "per_seed.csv", [dict(arm=arm, seed=s, **readings[arm]["per_seed"][s])
                                        for arm in ARMS for s in readings[arm]["per_seed"]])
    plot_curves(out, report / "regret_curves.png")
    print(json.dumps(dict(competent=competent, unbounded_competent=unb_competent, implication=implication,
                          seeds_meeting_both={arm: readings[arm]["seeds_meeting_both"] for arm in ARMS}), indent=1), flush=True)
    return out


def plot_curves(out, path):
    """Greedy-probe regret vs episodes per arm (context average; band = seed range), plus each seed's endpoint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    color = {"MF": "#2a78d6", "MB": "#eb6834"}          # categorical slots 1 and 2 (family); line style = memory
    style = {"MF": "-", "MB": "-", "MF-unbounded": "--", "MB-unbounded": "--"}
    names = {"MF": "MF (bounded)", "MB": "MB (bounded)", "MF-unbounded": "MF-unbounded (privileged)",
             "MB-unbounded": "MB-unbounded (privileged)"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
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
    bar = out["bar"]["mean"]
    last = 0
    ends = []
    for arm in ARMS:
        cv = out["curves"][arm]
        e = np.array(cv["episodes"][1:])
        m = np.array(cv["mean"][1:])
        fam = arm.split("-")[0]
        ax.fill_between(e, cv["seed_min"][1:], cv["seed_max"][1:], color=color[fam],
                        alpha=0.12 if arm in BOUNDED else 0.06, linewidth=0)
        ax.plot(e, m, style[arm], color=color[fam], linewidth=2, marker="o", markersize=4, label=names[arm])
        ends.append([float(np.log10(max(m[-1], 1e-6))), arm, m[-1]])
        last = e[-1]
    ends.sort()
    for i in range(1, len(ends)):   # direct end labels, de-collided in log space
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06)
    for y, arm, v in ends:
        ax.annotate(f"{arm} {v:.3f}", (last, v), xytext=(last * 1.02, 10 ** y), textcoords="data", va="center",
                    fontsize=8, color=ink)
    ax.axhline(bar, color=muted, linestyle=":", linewidth=1.2)
    ax.set_xlim(0, last * 1.28)
    ax.set_xticks([0, 1024, 2048, 3072, 4096] if last == 4096 else [0, last])
    ax.set_yscale("log")
    ticks = [0.002, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.5]
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ticks if lo <= t <= hi], [f"{t:g}" for t in ticks if lo <= t <= hi])
    ax.minorticks_off()
    ax.annotate(f"competence bar {bar:g}", (0, bar), xytext=(4, 4), textcoords="offset points", fontsize=8, color=muted)
    ax.set_xlabel("episodes of experience (stationary context)", color=ink, fontsize=9)
    ax.set_ylabel("greedy-probe regret vs O (survival)", color=ink, fontsize=9)
    ax.set_title("Regret vs episodes: mean over seeds x contexts (band: range of seed means)", fontsize=10, color=ink, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.06))
    for i, arm in enumerate(ARMS):
        ps = out["R1"][arm]["per_seed"]
        vals = np.array([ps[s]["context_average"] for s in ps])
        jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.025
        fam = arm.split("-")[0]
        bx.scatter(i + jitter, vals, s=34, facecolor=color[fam] if arm in BOUNDED else surface, edgecolor=color[fam],
                   linewidth=1.5, zorder=3)
        bx.annotate(f"{out['R1'][arm]['seeds_meeting_both']}/{len(vals)}", (i, vals.max()), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color=ink)
    bx.axhline(bar, color=muted, linestyle=":", linewidth=1.2)
    bx.set_yscale("log")
    lo, hi = bx.get_ylim()
    bx.set_yticks([t for t in ticks if lo <= t <= hi], [f"{t:g}" for t in ticks if lo <= t <= hi])
    bx.minorticks_off()
    bx.set_xticks(range(len(ARMS)), [a_.replace("-unbounded", "\nunbounded") for a_ in ARMS], fontsize=8, color=ink)
    bx.set_title("Each seed's context average at the endpoint\n(label: seeds meeting both R1 conditions)", fontsize=10,
                 color=ink, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# ------------------------------------------------------------------ smoke checks

def learner_import_check():
    """The learner module imports nothing that could leak world state."""
    tree = ast.parse((ROOT / "src/acp_cl/commons/learners.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
    allowed = {"__future__", "dataclasses", "hashlib", "numpy", "torch", "torch.nn"}
    return dict(imports=sorted(mods), allowed=sorted(allowed), passed=mods <= allowed)


def legal_observation_check(params):
    rows = {}
    stream = E.Stream(params, 7, SMOKE_SEED, 16)
    d, o = stream.decision(0), stream.outcome(0, 3)
    rows["legal_accepted"] = True
    L.check_decision(d)
    L.check_outcome(o)
    bad = []

    def rejects(fn):
        try:
            fn()
        except L.IllegalObservation:
            return True
        return False
    bad.append(("dict_decision", rejects(lambda: L.check_decision(dict(reserves=d.reserves, cue=d.cue)))))
    extra = L.make_decision(d.reserves, d.cue)
    object.__setattr__(extra, "context", 7)
    bad.append(("decision_with_context_attribute", rejects(lambda: L.check_decision(extra))))
    extra_o = L.make_outcome(o.y_supply, o.y_probe, o.fail_step)
    object.__setattr__(extra_o, "supplies", stream.supplies[0])
    bad.append(("outcome_with_latent_supplies", rejects(lambda: L.check_outcome(extra_o))))
    bad.append(("counterfactual_fail_table", rejects(lambda: L.make_outcome(o.y_supply, o.y_probe, stream.fail[0]))))
    bad.append(("writeable_arrays", rejects(lambda: L.check_decision(L.Decision(d.reserves.copy(), d.cue.copy())))))
    bad.append(("wrong_dtype", rejects(lambda: L.check_decision(L.Decision(
        np.asarray(d.reserves, np.float32), d.cue)))))
    for fam in ("MF", "MB"):
        h = L.Hyper(lr=0.003, U=1, ema_decay=0.99, total_episodes=4)
        lrn = L.make_learner(fam, h, learner_seeds(1, 7, fam))
        bad.append((f"{fam}_act_rejects_extra", rejects(lambda: lrn.act(extra))))
        bad.append((f"{fam}_observe_rejects_extra", rejects(lambda: lrn.observe(d, 3, extra_o))))
        bad.append((f"{fam}_observe_rejects_bad_action", rejects(lambda: lrn.observe(d, 13, o))))
    rows["illegal_rejected"] = dict(bad)
    sensors_noisy = bool(np.abs(o.y_supply - stream.supplies[0]).max() > 0)
    rows["supply_sensors_are_noisy_not_latent"] = sensors_noisy
    rows["outcome_is_chosen_action_only"] = bool(np.array_equal(o.fail_step, stream.fail[0, 3]))
    rows["passed"] = bool(all(rows["illegal_rejected"].values()) and sensors_noisy and rows["outcome_is_chosen_action_only"])
    return rows


def reservoir_check():
    cap, n, trials = 64, 640, 400
    counts = np.zeros(n)
    maxsize = 0
    for t in range(trials):
        rp = L.Replay(cap, 5, np.random.default_rng(seed_for("d1_reservoir_check", t)), bounded=True)
        d = L.make_decision(np.ones(3), np.zeros(3))
        o = L.make_outcome(np.zeros((12, 3)), np.zeros((12, 3)), np.full(3, 13))
        for k in range(n):
            rp.add(d, 0, o, np.ones(5, bool))
            maxsize = max(maxsize, rp.size)
            require(rp.size <= cap, "reservoir exceeded capacity")
        counts[rp.data["episode"][: rp.size]] += 1
    expected = trials * cap / n
    chi2 = float(((counts - expected) ** 2 / (expected * (1 - cap / n))).sum())   # binomial variance per index
    # early vs late thirds must be included equally often (uniform reservoir)
    thirds = [float(counts[i * n // 3:(i + 1) * n // 3].mean() / expected) for i in range(3)]
    return dict(capacity=cap, inserted=n, trials=trials, max_size=maxsize, chi2=chi2, dof=n - 1,
                chi2_z=float((chi2 - (n - 1)) / np.sqrt(2 * (n - 1))), inclusion_ratio_by_third=thirds,
                passed=bool(maxsize == cap and abs((chi2 - (n - 1)) / np.sqrt(2 * (n - 1))) < 4
                            and all(abs(x - 1) < 0.05 for x in thirds)))


def o_lookup_check(config, output, tables, params):
    rng = np.random.default_rng(5)
    rows = dict(stream=[], panel=[])
    m = config["oracle"]["draws"]
    for ctx in config["contexts"][:5]:
        c = E.context_of(ctx)
        seed = config["seeds"][0]
        stab = tables.stream_tab(seed, ctx)
        ep = sample_episodes(config["episodes"], seed_for("commons_d1_stream", seed, c))
        for e in rng.choice(config["episodes"], 2, replace=False):
            z = R.draws(seed_for("commons_d1_O_score", seed, c, int(e)), m)
            v, _ = R.episode_values(params, ep.reserves[e], int(ep.cue[e, 0]), [c], z)
            rows["stream"].append(bool(np.array_equal(v[0], stab["score"][e])))
        ptab = tables.panel_tab(ctx)
        for e in rng.choice(config["panel_size"], 2, replace=False):
            z = R.draws(seed_for("commons_d1_panel_O_select", c, int(e)), m)
            v, _ = R.episode_values(params, ptab["reserves"][e], int(ptab["cue"][e, 0]), list(range(N_CONTEXTS)), z)
            rows["panel"].append(bool(np.array_equal(v, ptab["select"][e])))
    # independent brute force: the full simulator on M copies of the state with the same draws (joint indicator)
    ctx = config["contexts"][3]
    c = E.context_of(ctx)
    ptab = tables.panel_tab(ctx)
    brute = []
    for e in range(3):
        z = R.draws(seed_for("commons_d1_panel_O_score", c, e), m)
        ep = sample_episodes(m, 0)
        ep.reserves[:] = ptab["reserves"][e]
        ep.cue[:] = ptab["cue"][e]
        ep.z_supply[:] = z
        for a in (0, 3, 9):
            joint = simulate(params, np.full(m, c), ep, np.full(m, a)).joint_survival.mean()
            val, se = ptab["score"][e, c, a], ptab["score_se"][e, c, a]
            brute.append(dict(state=e, action=a, table=float(val), brute_joint=float(joint), z=float((joint - val) / max(se, 1e-9) / 1.5)))
    # regret lookups: vectorised helper vs an explicit loop, for a random action vector
    acts = rng.integers(0, 13, config["panel_size"])
    vec = panel_regret(ptab, c, acts)
    loop = np.array([ptab["score"][s, c, int(np.argmax(ptab["select"][s, c]))] - ptab["score"][s, c, acts[s]]
                     for s in range(config["panel_size"])])
    sel_vs_score_draws_differ = not np.array_equal(R.draws(seed_for("commons_d1_panel_O_select", c, 0), 8),
                                                   R.draws(seed_for("commons_d1_panel_O_score", c, 0), 8))
    return dict(stream_entries_recomputed_equal=rows["stream"], panel_entries_recomputed_equal=rows["panel"],
                brute_force_vs_table=brute, regret_vectorised_equals_loop=bool(np.array_equal(vec, loop)),
                selection_and_scoring_draws_independent=bool(sel_vs_score_draws_differ),
                note="brute-force z uses the product-estimator SE x 1.5 as a conservative scale (the joint-indicator "
                     "estimator on the same draws has a different variance)",
                passed=bool(all(rows["stream"]) and all(rows["panel"]) and all(abs(b["z"]) < 4 for b in brute)
                            and np.array_equal(vec, loop) and sel_vs_score_draws_differ))


def crn_check(config, output, tasks):
    """Across arms (and grid configurations) at a (seed, context): identical stream draws, decision states
    and sensor readings; outcomes differ only through the chosen actions."""
    by = {}
    for arm, tag, seed, ctx in tasks:
        rec = read_json(fit_dir(output, arm, tag, seed, ctx) / "result.json")
        by.setdefault((seed, ctx), []).append((arm, tag, rec))
    rows = {}
    ok = True
    params, _ = E.load_world(ROOT)
    for (seed, ctx), recs in by.items():
        fp = {r["stream_fingerprint"] for _, _, r in recs}
        dec = {r["decisions_chain"] for _, _, r in recs}
        sen = {r["sensors_chain"] for _, _, r in recs}
        # outcome of each arm equals the world simulated on the shared draws at its own actions
        stream = E.Stream(params, E.context_of(ctx), seed, config["episodes"])
        outcome_ok = True
        for arm, tag, _ in recs:
            tr, _ = load_arrays(fit_dir(output, arm, tag, seed, ctx) / "trace.npz")
            sim = simulate(params, np.full(config["episodes"], E.context_of(ctx)), stream.episodes,
                           tr["action"].astype(np.int64), stream.supplies).joint_survival
            outcome_ok &= bool(np.array_equal(sim, tr["realised"]))
        rows[f"{seed}_{ctx}"] = dict(arms=[f"{a}|{t}" for a, t, _ in recs], stream_fingerprints=len(fp), decision_chains=len(dec),
                                     sensor_chains=len(sen), outcomes_match_shared_world=outcome_ok)
        ok &= len(fp) == 1 and len(dec) == 1 and len(sen) == 1 and outcome_ok
    # streams of different contexts / seeds must differ; the panel must be disjoint from every stream
    s1 = sample_episodes(64, seed_for("commons_d1_stream", config["seeds"][0], 0)).reserves
    s2 = sample_episodes(64, seed_for("commons_d1_stream", config["seeds"][0], 4)).reserves
    disjoint = True
    for ctx in config["contexts"]:
        c = E.context_of(ctx)
        pan = sample_episodes(config["panel_size"], seed_for("commons_d1_panel", c)).reserves
        for seed in config["seeds"]:
            st = sample_episodes(config["episodes"], seed_for("commons_d1_stream", seed, c)).reserves
            disjoint &= not (set(map(tuple, pan.round(12))) & set(map(tuple, st.round(12))))
    return dict(cells=rows, contexts_have_distinct_streams=not np.array_equal(s1, s2), panel_disjoint_from_streams=bool(disjoint),
                passed=bool(ok and disjoint and not np.array_equal(s1, s2)))


def ema_check(config, identity, output):
    """The EMA is a read-only shadow: with the actions forced to a recorded sequence, the online weights and
    optimiser states are bit-identical with the EMA tracked (decay .99 or .998) or not tracked at all;
    and the EMA equals the declared recursion."""
    out = {}
    base = output / "checks" / "ema"
    for fam in BOUNDED:
        tag = cfg_tag(dict(lr=0.003, U=2, ema_decay=0.99))
        tag2 = cfg_tag(dict(lr=0.003, U=2, ema_decay=0.998))
        seed, ctx = config["seeds"][0], "A0B0C1"
        run_fit(config, identity, fresh(base / f"{fam}_a"), fam, tag, seed, ctx, keep_checkpoint=True)
        tr, _ = load_arrays(fit_dir(base / f"{fam}_a", fam, tag, seed, ctx) / "trace.npz")
        forced = tr["action"].astype(np.int64)
        run_fit(config, identity, fresh(base / f"{fam}_b"), fam, tag, seed, ctx, track_ema=False, forced_actions=forced)
        run_fit(config, identity, fresh(base / f"{fam}_c"), fam, tag2, seed, ctx, forced_actions=forced)
        ra = read_json(fit_dir(base / f"{fam}_a", fam, tag, seed, ctx) / "result.json")
        rb = read_json(fit_dir(base / f"{fam}_b", fam, tag, seed, ctx) / "result.json")
        rc = read_json(fit_dir(base / f"{fam}_c", fam, tag2, seed, ctx) / "result.json")
        same_online = ra["final_hashes"]["online"] == rb["final_hashes"]["online"] == rc["final_hashes"]["online"]
        same_opt = ra["final_hashes"]["optimizer"] == rb["final_hashes"]["optimizer"] == rc["final_hashes"]["optimizer"]
        ema_differs = ra["final_hashes"]["ema"] != rb["final_hashes"]["ema"] and ra["final_hashes"]["ema"] != rc["final_hashes"]["ema"]
        # recursion: 5 manual steps
        h = L.Hyper(lr=0.003, U=1, ema_decay=0.9, total_episodes=8)
        lrn = L.make_learner(fam, h, learner_seeds(seed, 1, fam))
        st = E.Stream(E.load_world(ROOT)[0], 1, seed, 8)
        manual = [p.detach().clone() for p in lrn.net.params]
        rec_ok = True
        for k in range(5):
            d = st.decision(k)
            lrn.observe(d, 1, st.outcome(k, 1))
            manual = [0.9 * m_ + 0.1 * p.detach() for m_, p in zip(manual, lrn.net.params)]
            rec_ok &= all(torch.allclose(m_, e, atol=1e-6, rtol=0) for m_, e in zip(manual, lrn.ema[0]))
        out[fam] = dict(online_weights_identical=same_online, optimizer_identical=same_opt, ema_tracks_differently=ema_differs,
                        ema_recursion_matches=bool(rec_ok), passed=bool(same_online and same_opt and ema_differs and rec_ok))
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def eviction_check(config, identity, output):
    """Bounded replay in the full learner path with a 64-episode budget (the smoke stream is shorter than 1,024):
    never exceeds its budget, evicts, and is identical to the unbounded arm until the first eviction."""
    out = {}
    base = fresh(output / "checks" / "eviction")
    tag = cfg_tag(dict(lr=0.003, U=2, ema_decay=0.99))
    seed, ctx = config["seeds"][0], "A0B1C0"
    sub = dict(config, episodes=128, checkpoint_every=64)
    for fam in BOUNDED:
        run_fit(sub, identity, base, fam, tag, seed, ctx, capacity=64)
        run_fit(sub, identity, base, UNBOUNDED[fam], tag, seed, ctx)
        rb = read_json(fit_dir(base, fam, tag, seed, ctx) / "result.json")
        ru = read_json(fit_dir(base, UNBOUNDED[fam], tag, seed, ctx) / "result.json")
        tb, _ = load_arrays(fit_dir(base, fam, tag, seed, ctx) / "trace.npz")
        tu, _ = load_arrays(fit_dir(base, UNBOUNDED[fam], tag, seed, ctx) / "trace.npz")
        # before the first eviction (episode 65) the arms are the same learner; the probe at 64 must agree
        same_first = bool(np.array_equal(tb["action"][:65], tu["action"][:65])
                          and np.array_equal(tb["probe_value"][:2], tu["probe_value"][:2]))
        out[fam] = dict(max_replay_bounded=rb["max_replay"], max_replay_unbounded=ru["max_replay"],
                        evictions=rb["replay"]["evictions"], identical_until_first_eviction=same_first,
                        diverge_after=bool(not np.array_equal(tb["probe_value"][2], tu["probe_value"][2])),
                        passed=bool(rb["max_replay"] == 64 and ru["max_replay"] == 128 and rb["replay"]["evictions"] > 0 and same_first))
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def resume_check(config, identity, output, main_output, tasks):
    """Determinism (re-fit from scratch equals the pool run) and resumption (interrupted mid-block, resumed)."""
    out = {}
    base = fresh(output / "checks" / "resume")
    for arm in ("MF", "MB-unbounded"):
        arm_tasks = [t for t in tasks if t[0] == arm]
        _, tag, seed, ctx = arm_tasks[-1]
        try:
            run_fit(config, identity, base, arm, tag, seed, ctx, interrupt_after=100)
        except Interrupted:
            pass
        side = read_json(fit_dir(base, arm, tag, seed, ctx) / "checkpoint.sha256.json")
        run_fit(config, identity, base, arm, tag, seed, ctx)
        rec = read_json(fit_dir(base, arm, tag, seed, ctx) / "result.json")
        rows = compare_fits(fit_dir(main_output, arm, tag, seed, ctx), fit_dir(base, arm, tag, seed, ctx))
        out[arm] = dict(checkpoint_episodes_at_interrupt=side["episodes_done"], resumed=rec["resumed"],
                        resumes=rec["resumes"], comparison=rows,
                        passed=bool(rec["resumed"] and side["episodes_done"] == 64 and all(rows.values())))
    # a corrupted checkpoint is quarantined and the fit restarts from scratch with the same result
    arm, tag, seed, ctx = [t for t in tasks if t[0] == "MF"][0]
    try:
        run_fit(config, identity, base, arm, tag, seed, ctx, interrupt_after=70)
    except Interrupted:
        pass
    ck = fit_dir(base, arm, tag, seed, ctx) / "checkpoint.pt"
    raw = bytearray(ck.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    ck.write_bytes(bytes(raw))
    run_fit(config, identity, base, arm, tag, seed, ctx)
    rows = compare_fits(fit_dir(main_output, arm, tag, seed, ctx), fit_dir(base, arm, tag, seed, ctx))
    q = list((base / "quarantine").glob("*checkpoint.pt"))
    out["corrupt_checkpoint"] = dict(quarantined=len(q), comparison=rows, passed=bool(q and all(rows.values())))
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def amendment_check(identity, output):
    """The amended test probe schedule changes only when probes are taken: one dev cell per family is re-fitted with
    the amended schedule and must reproduce the stored dev fit (actions, final weights and the probes at the shared
    episodes) bit for bit."""
    dev_cfg = read_json(ROOT / "configs/commons_d1_dev.json")
    dev_out = ROOT / "runs/commons_d1_dev"
    cfg = dict(dev_cfg, probe_episodes=AMENDED_PROBES)
    base = fresh(output / "checks" / "amendment")
    out = {}
    seed, ctx = DEV_SEEDS[0], "A1B1C1"
    sel = read_json(ROOT / "reports/commons_d1/dev/dev_results.json")["selection"]
    for fam in BOUNDED:
        tag = cfg_tag(sel[fam]["selected"])
        ref = fit_dir(dev_out, fam, tag, seed, ctx)
        if not (ref / "result.json").exists():
            out[fam] = dict(skipped="no development fit", passed=False)
            continue
        run_fit(cfg, identity, base, fam, tag, seed, ctx)
        a, _ = load_arrays(ref / "trace.npz")
        b, _ = load_arrays(fit_dir(base, fam, tag, seed, ctx) / "trace.npz")
        ra, rb = read_json(ref / "result.json"), read_json(fit_dir(base, fam, tag, seed, ctx) / "result.json")
        pos = [list(a["probe_episode"]).index(e) for e in b["probe_episode"]]
        probes_equal = all(np.array_equal(a[k][pos], b[k]) for k in b if k.startswith("probe_") and k != "probe_episode")
        out[fam] = dict(probe_episodes=b["probe_episode"].tolist(), actions_equal=bool(np.array_equal(a["action"], b["action"])),
                        final_weights_equal=ra["final_hashes"] == rb["final_hashes"], probes_equal_at_shared_episodes=bool(probes_equal),
                        probe_seconds_dev=ra["probe_seconds"], probe_seconds_amended=rb["probe_seconds"],
                        passed=bool(np.array_equal(a["action"], b["action"]) and ra["final_hashes"] == rb["final_hashes"] and probes_equal
                                    and b["probe_episode"].tolist() == [0] + AMENDED_PROBES))
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def cmd_check(args):
    config = load_config(args.config)
    require(config["phase"] == "smoke", "check runs the smoke configuration")
    output = Path(args.output)
    fresh(output)
    worker_init(1)
    t0 = time.perf_counter()
    checks = dict(label="ENGINEERING CHECKS on the smoke run (seed 21099, 256 episodes). Not science.", utc=now())
    params, cfg = E.load_world(ROOT)
    checks["world"] = dict(key=E.cfg_key(cfg), params_sha256=params.digest(), passed=E.cfg_key(cfg) == E.WORLD_KEY)
    checks["learner_module_imports"] = learner_import_check()
    checks["legal_observations"] = legal_observation_check(params)
    checks["reservoir_uniform_and_bounded"] = reservoir_check()
    print("unit checks", {k: v.get("passed") for k, v in checks.items() if isinstance(v, dict)}, flush=True)
    # the smoke study: lock, run, run again (verify only), replicate, analyze
    manifest = lock(config, args.config, output)
    identity = manifest["identity"]
    a = argparse.Namespace(config=args.config, output=str(output), report=str(output / "analysis"))
    first = cmd_run(a)
    tasks = fit_tasks(config)
    second = cmd_run(a)
    checks["rerun_only_verifies"] = dict(statuses=sorted({s["status"] for s in second["statuses"]}),
                                         passed=all(s["status"] == "verified_existing" for s in second["statuses"]))
    tables = Tables(config, output)
    checks["o_table_lookups"] = o_lookup_check(config, output, tables, params)
    checks["common_random_numbers"] = crn_check(config, output, tasks)
    checks["bounded_replay_in_run"] = dict(max_replay={f"{t[0]}_{t[3]}": read_json(fit_dir(output, *t) / "result.json")["max_replay"]
                                                      for t in tasks},
                                           passed=all(read_json(fit_dir(output, *t) / "result.json")["max_replay"]
                                                      <= config["learner"]["capacity"] for t in tasks))
    checks["bounded_replay_eviction_path"] = eviction_check(config, identity, output)
    checks["ema_does_not_interfere"] = ema_check(config, identity, output)
    checks["determinism_and_resumption"] = resume_check(config, identity, output, output, tasks)
    checks["amended_probe_schedule_equivalence"] = amendment_check(identity, output)
    print("fit checks", {k: v.get("passed") for k, v in checks.items() if isinstance(v, dict)}, flush=True)
    rep = cmd_replicate(a)
    checks["replicate_pass"] = dict(fits_compared=rep["fits_compared"], mismatches=rep["mismatches"], crc=rep["crc"]["passed"],
                                    passed=rep["passed"])
    # corruption is detected by the hash/CRC machinery
    victim = fit_dir(output, *tasks[0]) / "trace.npz"
    backup = victim.read_bytes()
    raw = bytearray(backup)
    raw[len(raw) // 3] ^= 0xFF
    victim.write_bytes(bytes(raw))
    try:
        verify_result(victim.parent, identity)
        detected = False
    except ValueError:
        detected = True
    crc_bad = crc_scan(output)
    victim.write_bytes(backup)
    checks["corruption_detected"] = dict(hash_detects=detected, crc_detects=not crc_bad["passed"], passed=bool(detected and not crc_bad["passed"]))
    res = cmd_analyze(a)
    checks["analysis_pipeline"] = dict(implication_smoke_only=res["implication"], passed=True)
    first_rec = read_json(fit_dir(output, "MB", *tasks[[t[0] for t in tasks].index("MB")][1:]) / "result.json")
    checks["timing_smoke"] = dict(MF_seconds=[read_json(fit_dir(output, *t) / "result.json")["elapsed_seconds"] for t in tasks if t[0] == "MF"],
                                  MB_seconds=[read_json(fit_dir(output, *t) / "result.json")["elapsed_seconds"] for t in tasks if t[0] == "MB"],
                                  MB_probe_seconds_per_checkpoint=first_rec["probe_seconds"] / (config["episodes"] / config["checkpoint_every"] + 1))
    checks["all_passed"] = bool(all(v["passed"] for k, v in checks.items() if isinstance(v, dict) and "passed" in v))
    checks["seconds"] = time.perf_counter() - t0
    write_json(output / "checks" / "checks.json", checks)
    print("ALL PASS" if checks["all_passed"] else "FAILURES: " + ", ".join(k for k, v in checks.items()
                                                                          if isinstance(v, dict) and v.get("passed") is False), flush=True)
    if not checks["all_passed"]:
        raise SystemExit(1)


# ------------------------------------------------------------------ main

DEFAULTS = {"check": ("configs/commons_d1_smoke.json", "runs/commons_d1_smoke", "runs/commons_d1_smoke/analysis"),
            "dev": ("configs/commons_d1_dev.json", "runs/commons_d1_dev", "reports/commons_d1/dev"),
            "lock": ("configs/commons_d1.json", "runs/commons_d1", "reports/commons_d1"),
            "run": ("configs/commons_d1.json", "runs/commons_d1", "reports/commons_d1"),
            "replicate": ("configs/commons_d1.json", "runs/commons_d1", "reports/commons_d1"),
            "analyze": ("configs/commons_d1.json", "runs/commons_d1", "reports/commons_d1")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(DEFAULTS))
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--report")
    parser.add_argument("--skip-unbounded", action="store_true", help="dev only: skip the unbounded arms")
    parser.add_argument("--reanalyze", action="store_true", help="dev only: re-run the analysis on verified stored fits")
    args = parser.parse_args()
    d = DEFAULTS[args.command]
    args.config = str(ROOT / (args.config or d[0])) if not (args.config and Path(args.config).is_absolute()) else args.config
    args.output = str(ROOT / (args.output or d[1])) if not (args.output and Path(args.output).is_absolute()) else args.output
    args.report = str(ROOT / (args.report or d[2])) if not (args.report and Path(args.report).is_absolute()) else args.report
    {"check": cmd_check, "dev": cmd_dev, "lock": cmd_lock, "run": cmd_run, "replicate": cmd_replicate,
     "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
