"""Step D2: recoverable context knowledge under recurrence (docs/commons_d2_protocol.md), PHASE 1.

Arms (src/acp_cl/commons/learners_d2.py): G, W, AM, AM-factored, FC, FC-single, FC-K1, OC on the shared MB-count core.
References on every stream (evaluator-only, reused from D0b by import): O (context-known MC tables, 2^14 draws,
independent selection / scoring draws), L (D0b's one-step lookahead on the run-length filter; the primary reference)
and I-myopic. Counterfactual probes: 32 fixed decision states per context (all 8, incl. held-out), at positions
{2, 5, 10, 20, 40} of every run (position j = the decision of the j-th episode of the run, i.e. after j - 1 episodes
of the new context), each arm's greedy action (EMA ensemble mean, no bonus) scored by O's scoring table of the run's
context over all 13 actions, reported as excess over L's action on the same states at the same time.

Integrity (as D1/D1b, by import of scripts/commons_d1.py): manifest/lock with an explicit source-file list,
attempts.jsonl, resumable checkpointed fits, sha256 sidecars, CRC scans, replicate pass, <= 6 single-thread workers.
References are cached per stream in runs/commons_d2_refs/ under their own identity (world key, reference code hashes,
REF_VERSION), so learner-code changes never invalidate them.

Commands (PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=src):
  .venv/Scripts/python.exe -B scripts/commons_d2.py check    (configs/commons_d2_smoke.json -> runs/commons_d2_smoke)
  .venv/Scripts/python.exe -B scripts/commons_d2.py dev      (configs/commons_d2_dev.json   -> runs/commons_d2_dev, reports/commons_d2/dev)
  .venv/Scripts/python.exe -B scripts/commons_d2.py dev-report  (re-analysis of finished dev fits against the stored dev manifest)
  .venv/Scripts/python.exe -B scripts/commons_d2.py lock | run | replicate | analyze   (configs/commons_d2.json -> runs/commons_d2)
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
import math  # noqa: E402
from math import comb  # noqa: E402
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

import commons_d0b as D0B  # noqa: E402
import commons_d1 as D1  # noqa: E402
from acp_cl.commons import calibration as C  # noqa: E402
from acp_cl.commons import d1_env as E  # noqa: E402
from acp_cl.commons import learners as L  # noqa: E402
from acp_cl.commons import learners_count as LC  # noqa: E402
from acp_cl.commons import learners_d2 as D  # noqa: E402
from acp_cl.commons import references as R  # noqa: E402
from acp_cl.commons.world import (N_CONTEXTS, SURVIVED, action_table, context_name, factors, n_actions, observe,  # noqa: E402
                                  sample_episodes, seed_for, simulate)

sha, digest, read_json, write_json, now, require, log_event = D1.sha, D1.digest, D1.read_json, D1.write_json, D1.now, D1.require, D1.log_event
save_arrays, load_arrays, quarantine, fresh, chain = D1.save_arrays, D1.load_arrays, D1.quarantine, D1.fresh, D1.chain

LABEL = ("PROSPECTIVE STUDY under docs/commons_d2_protocol.md (Step D2). O, L and I-myopic are evaluator-only; OC is "
         "privileged (lagged true factor levels); W is descriptive (privileged reset timing).")
SCRIPT_FILES = ("scripts/commons_d0.py", "scripts/commons_d0b.py", "scripts/commons_d1.py", "scripts/commons_d2.py")
SOURCE_FILES = tuple(f"src/acp_cl/commons/{n}.py" for n in
                     ("__init__", "world", "references", "calibration", "verification", "d1_env", "learners", "learners_count",
                      "learners_d2"))
REF_FILES = ("src/acp_cl/commons/world.py", "src/acp_cl/commons/references.py", "src/acp_cl/commons/calibration.py",
             "src/acp_cl/commons/d1_env.py", "scripts/commons_d0.py", "scripts/commons_d0b.py")
REF_VERSION = 1
REFS = ROOT / "runs" / "commons_d2_refs"
ARMS = ("G", "W", "AM", "AM-factored", "FC", "FC-single", "FC-K1", "OC")
KNOBS = {"AM": ("k", "enc_lr"), "AM-factored": ("k", "enc_lr"), "FC": ("tau", "code_lr"), "FC-single": ("tau", "code_lr")}
GRID = {"AM": {"k": [1, 8, 32], "enc_lr": [0.001, 0.003]}, "AM-factored": {"k": [1, 8, 32], "enc_lr": [0.001, 0.003]},
        "FC": {"tau": [5, 15], "code_lr": [0.01, 0.03]}, "FC-single": {"tau": [5, 15], "code_lr": [0.01, 0.03]}}
DEV_SEEDS = [22101, 22102, 22103]
TEST_SEEDS_12 = list(range(22201, 22213))
TEST_SEEDS_24 = list(range(22201, 22225))
SMOKE_SEED = 22199
POSITIONS = (2, 5, 10, 20, 40)
RMIN_POSITIONS = (5, 10, 20)
RHELD_POSITIONS = (2, 5, 10, 20, 40)
WINDOWS = {"identification_lag": (1, 2), "model_quality": (3, 20), "C_specific": (1, 40)}
TRAINING_NAMES = ("A0B0C0", "A1B0C0", "A0B1C0", "A0B0C1", "A1B1C1")
CRITICAL_ACTIONS = (6, 9, 12)          # D1b's A1B1C1 trap actions (descriptive trap flag)


# ------------------------------------------------------------------ config

def arm_tag(arm, knobs, config):
    if arm in ("G", "W"):
        return "base"
    if arm == "OC":
        return f"clr{knobs['code_lr']:g}"
    if arm == "FC-K1":
        return f"clr{knobs['code_lr']:g}"
    if arm in ("AM", "AM-factored"):
        return f"k{knobs['k']}_elr{knobs['enc_lr']:g}"
    return f"tau{knobs['tau']:g}_clr{knobs['code_lr']:g}_{config['fc']['spawn_rule']}"


def grid_knobs(arm):
    a, b = KNOBS[arm]
    return [{a: x, b: y} for x in GRID[arm][a] for y in GRID[arm][b]]


def knobs_for(config, arm, tag):
    if arm in ("G", "W"):
        return {}
    if arm == "OC":
        return {"code_lr": config["oc_code_lr"]}
    if arm == "FC-K1":
        return {"code_lr": float(tag[3:])}
    cands = grid_knobs(arm) if config["phase"] == "dev" else [config["knobs"][arm]]
    for kn in cands:
        if arm_tag(arm, kn, config) == tag:
            return dict(kn)
    raise ValueError(f"{arm} {tag}")


def validate_config(c):
    expected = {"study", "protocol", "phase", "world", "seeds", "episodes", "core", "fc", "oc_code_lr", "grid", "knobs", "arms",
                "probe_positions", "panel_size", "oracle", "checkpoint_every", "parity", "analysis", "workers", "threads", "device",
                "stall_seconds", "stage_attempts", "notes"}
    if set(c) != expected:
        raise ValueError(f"invalid configuration fields: {sorted(set(c) ^ expected)}")
    require(c["study"] == "commons_d2" and c["protocol"] == "docs/commons_d2_protocol.md", "study / protocol")
    require(c["phase"] in ("smoke", "dev", "test"), "phase")
    require(c["world"] == dict(separability="reports/commons_d0b/separability.json", spec="reports/commons_d0b/world_spec.json",
                               key=E.WORLD_KEY), "the D0b world")
    require(c["core"] == dict(lr=0.001, U=2, ema_decay=0.99, count_c=0.05, members=5, hidden=128, minibatch=64, grad_clip=5.0,
                              bootstrap_p=0.5, plan_samples=256, capacity=1024), "D1b's MB-count core, fixed by the protocol")
    fc = c["fc"]
    require(fc["max_codes"] == 6 and fc["h"] == 0.05 and fc["switch_posterior"] == 0.95 and fc["default_fraction"] == 0.2
            and fc["spawn_run"] == 2 and fc["spawn_rule"] in ("literal", "lrt") and fc["fit_steps"] == 20 and fc["fit_ridge"] == 1.0
            and fc.get("code_decay", 0.0) >= 0,
            "FC constants fixed by the protocol (fit_steps / fit_ridge: the proposed 'lrt' revision's declared constants)")
    require(c["grid"] == GRID and c["arms"] == list(ARMS), "arms and declared knob grids")
    require(c["probe_positions"] == list(POSITIONS) and c["panel_size"] == 32, "probe design")
    require(c["oracle"] == dict(draws=2 ** 14, stream_chunk=500), "O as in D0b")
    require(c["device"] == "cpu" and c["threads"] == 1 and 1 <= c["workers"] <= 6, "CPU, 1 thread, <= 6 workers")
    a = c["analysis"]
    require(a["v1_margin"] == 0.005 and a["v2_fraction"] == 0.5 and a["c1_margin"] == 0.002 and a["relative_margin"] == 0.25
            and a["oc_gate_max"] == 0.010 and a["oc_gate_positions"] == [5, 10, 20, 40] and a["power_target"] == 0.8,
            "pre-declared analysis constants")
    if c["phase"] == "smoke":
        require(c["seeds"] == [SMOKE_SEED] and c["episodes"] == 2000, "smoke: seed 22199, 2,000 episodes")
    elif c["phase"] == "dev":
        require(c["seeds"] == DEV_SEEDS and c["episodes"] == 20000 and c["knobs"] is None, "development design")
    else:
        require(c["seeds"] in (TEST_SEEDS_12, TEST_SEEDS_24) and c["episodes"] == 20000, "test design")
    if c["phase"] in ("smoke", "test"):
        kn = c["knobs"]
        require(isinstance(kn, dict) and set(kn) == set(KNOBS) | {"FC-K1", "AM*", "delta"}, "one knob set per tuned arm + AM* + delta")
        for arm in KNOBS:
            require(kn[arm] in grid_knobs(arm), f"{arm} knob must be a grid value")
        require(kn["FC-K1"] == {"code_lr": kn["FC"]["code_lr"]}, "FC-K1 uses FC's code lr")
        require(kn["AM*"] in ("AM", "AM-factored"), "AM* is AM or AM-factored")


def load_config(path):
    c = read_json(path)
    validate_config(c)
    return c


def fit_tasks(config, stage="all"):
    """(arm, tag, seed). Dev stages: 'gates' (G, OC), 'grid' (W, AM, AM-factored, FC, FC-single), 'k1' (FC-K1 at the
    selected FC code lr, filled by the caller)."""
    out = []
    ph = config["phase"]
    for arm in config["arms"]:
        if ph == "dev":
            if arm in KNOBS:
                tags = [arm_tag(arm, kn, config) for kn in grid_knobs(arm)]
            elif arm == "FC-K1":
                continue
            else:
                tags = [arm_tag(arm, {"code_lr": config["oc_code_lr"]} if arm == "OC" else {}, config)]
        else:
            kn = config["knobs"].get(arm, {"code_lr": config["oc_code_lr"]} if arm == "OC" else {})
            tags = [arm_tag(arm, kn, config)]
        for tag in tags:
            for seed in config["seeds"]:
                out.append((arm, tag, seed))
    if ph == "dev":
        if stage == "gates":
            out = [t for t in out if t[0] in ("G", "OC")]
        elif stage == "grid":
            out = [t for t in out if t[0] not in ("G", "OC")]
    return out


# ------------------------------------------------------------------ identity / lock (explicit source list)

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


# ------------------------------------------------------------------ streams and panels (evaluator side)

def world():
    params, _ = E.load_world(ROOT)
    return params


def schedule_spec(length):
    return C.ScheduleSpec(length=int(length))


class Stream:
    """The D2 stream of a seed: D0's schedule generator (declared spec, only the length changes for the smoke run) and
    common random numbers per (seed, episode) shared by every arm and reference; outcomes of all 13 actions are
    precomputed (a learner only ever receives the chosen one)."""

    def __init__(self, params, seed, length):
        self.seed, self.n = int(seed), int(length)
        self.spec = schedule_spec(length)
        contexts, order, info = C.generate_schedule(self.seed, self.spec)
        self.contexts = contexts.astype(np.int64)
        self.order, self.info = order, info
        self.episodes = sample_episodes(self.n, seed_for("commons_d2_stream", self.seed))
        self.supplies = self.episodes.supplies(params, self.contexts)
        self.y_supply, self.y_probe = observe(params, self.contexts, self.episodes, self.supplies, probes_on=True)
        self.fail = np.stack([simulate(params, self.contexts, self.episodes, np.full(self.n, a), self.supplies).fail_step
                              for a in range(n_actions(params))], 1)
        self.levels = np.stack(factors(self.contexts), 1).astype(np.uint8)
        self.starts = np.cumsum([0] + [r for _, r in order])[:-1].astype(np.int64)
        self.lengths = np.array([r for _, r in order], np.int64)
        self.run_context = np.array([c for c, _ in order], np.int64)

    def decision(self, k):
        return L.make_decision(self.episodes.reserves[k], self.episodes.cue[k])

    def outcome(self, k, action):
        return L.make_outcome(self.y_supply[k], self.y_probe[k], self.fail[k, int(action)])

    def label(self, k):
        return D.make_label(self.levels[k])

    def fingerprint(self):
        return D1.digest(dict(episodes=self.episodes.fingerprint(), contexts=D1.array_digest({"c": self.contexts})))


def probe_events(starts, lengths, run_context, positions=POSITIONS):
    """(t, run, position, context) for every run and every position j <= run length (t = start + j - 1)."""
    out = []
    for r, (s, n, c) in enumerate(zip(starts, lengths, run_context)):
        for j in positions:
            if j <= n:
                out.append((int(s + j - 1), r, int(j), int(c)))
    return np.array(out, np.int64).reshape(-1, 4)


def schedule_arrays(seed, length):
    contexts, order, _ = C.generate_schedule(int(seed), schedule_spec(length))
    starts = np.cumsum([0] + [r for _, r in order])[:-1].astype(np.int64)
    return contexts.astype(np.int64), starts, np.array([r for _, r in order], np.int64), np.array([c for c, _ in order], np.int64)


class Panel:
    """32 decision states of a context (seed- and arm-independent) and the fixed planning draws for greedy probes."""

    def __init__(self, context, size, plan_samples):
        self.context = int(context)
        ep = sample_episodes(size, seed_for("commons_d2_panel", self.context))
        self.reserves, self.cue = ep.reserves, ep.cue
        rng = np.random.default_rng(seed_for("commons_d2_panel_plan", self.context))
        self.eps = rng.standard_normal((size, plan_samples, 12, 3), dtype=np.float32)


# ------------------------------------------------------------------ references (cached per stream, own identity)

def ref_identity(config):
    files = {n: sha(ROOT / n) for n in REF_FILES}
    return digest(dict(version=REF_VERSION, world_key=E.WORLD_KEY, files=files, oracle=config["oracle"], panel_size=config["panel_size"],
                       positions=config["probe_positions"], spec="calibration.ScheduleSpec defaults, length = episodes"))


def ref_dir(seed, length):
    return REFS / f"stream_n{length}_s{seed}"


def panel_ref_path(ctx):
    return REFS / "panels" / f"panel_{context_name(ctx)}.npz"


def ref_ok(path, rid):
    loaded = load_arrays(path, strict=False)
    return loaded is not None and loaded[1]["meta"].get("ref_identity") == rid


def run_ref_panel(config, rid, ctx):
    path = panel_ref_path(ctx)
    if path.exists():
        if ref_ok(path, rid):
            return dict(status="verified_existing", seconds=0.0)
        quarantine(REFS, path, "unverifiable or reference identity mismatch")
        quarantine(REFS, path.with_suffix(".sha256.json"), "sidecar of quarantined file")
    params = world()
    t0 = time.perf_counter()
    panel = Panel(ctx, config["panel_size"], 1)
    contexts = list(range(N_CONTEXTS))
    m = config["oracle"]["draws"]
    sel, sel_se = E.o_values(params, panel.reserves, panel.cue, contexts, ("commons_d2_panel_O_select", ctx), m)
    sco, sco_se = E.o_values(params, panel.reserves, panel.cue, contexts, ("commons_d2_panel_O_score", ctx), m)
    save_arrays(path, meta=dict(ref_identity=rid, context=ctx, seconds=time.perf_counter() - t0), reserves=panel.reserves,
                cue=panel.cue, select=sel, select_se=sel_se, score=sco, score_se=sco_se)
    return dict(status="completed", seconds=time.perf_counter() - t0)


def run_ref_chunk(config, rid, seed, length, start):
    """O's tables (selection over all 8 contexts, scoring in the true context) and I's full likelihoods (D0b's
    compute_chunk logic with D2's draws) for stream episodes start..stop-1."""
    path = ref_dir(seed, length) / f"chunk_{start:05d}.npz"
    if path.exists():
        if ref_ok(path, rid):
            return dict(status="verified_existing", seconds=0.0)
        quarantine(REFS, path, "unverifiable or reference identity mismatch")
        quarantine(REFS, path.with_suffix(".sha256.json"), "sidecar of quarantined file")
    params = world()
    na = n_actions(params)
    m = config["oracle"]["draws"]
    stop = min(start + config["oracle"]["stream_chunk"], length)
    t0 = time.perf_counter()
    contexts, _, _, _ = schedule_arrays(seed, length)
    episodes = sample_episodes(length, seed_for("commons_d2_stream", seed))
    idx = np.arange(start, stop)
    ep = episodes.take(idx)
    ctx = contexts[idx]
    n = len(idx)
    v1 = np.empty((n, N_CONTEXTS, na))
    v1se = np.empty_like(v1)
    v2 = np.empty((n, na))
    v2se = np.empty_like(v2)
    for k, e in enumerate(idx):
        z = R.draws(seed_for(seed, "commons_d2_O_select", int(e)), m)
        v1[k], v1se[k] = R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), list(range(N_CONTEXTS)), z)
        z = R.draws(seed_for(seed, "commons_d2_O_score", int(e)), m)
        val, se = R.episode_values(params, ep.reserves[k], int(ep.cue[k, 0]), [int(ctx[k])], z)
        v2[k], v2se[k] = val[0], se[0]
    supplies = ep.supplies(params, ctx)
    fs = np.empty((n, na, 3), dtype=np.int8)
    for a in range(na):
        fs[:, a] = simulate(params, ctx, ep, np.full(n, a), supplies).fail_step
    y_supply, y_probe = observe(params, ctx, ep, supplies, probes_on=True)
    stats = {}
    sensor, pattern = R.episode_loglik(params, ep, y_supply, y_probe, fs, list(range(na)),
                                       seed_for(seed, "commons_d2_I_pattern", start), R.PatternSettings(), True, stats)
    save_arrays(path, meta=dict(ref_identity=rid, seed=seed, length=length, start=start, stop=stop, pattern_stats=stats,
                                seconds=time.perf_counter() - t0),
                v1=v1, v1se=v1se, v2=v2, v2se=v2se, fail_steps=fs, sensor_ll=sensor, pattern_ll=pattern, contexts=ctx, index=idx)
    return dict(status="completed", seconds=time.perf_counter() - t0)


def load_ref_stream(config, seed, length, rid=None):
    parts = {}
    for start in range(0, length, config["oracle"]["stream_chunk"]):
        arrays, info = load_arrays(ref_dir(seed, length) / f"chunk_{start:05d}.npz")
        if rid is not None:
            require(info["meta"]["ref_identity"] == rid, "reference identity mismatch")
        for k, v in arrays.items():
            parts.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    require(np.array_equal(out["index"], np.arange(length)), "reference chunks do not tile the stream")
    return out


def load_panel_tables(config, rid=None):
    out = {}
    for ctx in range(N_CONTEXTS):
        arrays, info = load_arrays(panel_ref_path(ctx))
        if rid is not None:
            require(info["meta"]["ref_identity"] == rid, "panel reference identity mismatch")
        out[ctx] = arrays
    return out


def run_ref_agents(config, rid, seed, length):
    """I-myopic and L (D0b's Lookahead, pool and filter model) on the stream, plus their actions on the probe panels
    at every probe event (the agent's belief at that decision; the lookahead for probe states uses its own noise key)."""
    path = ref_dir(seed, length) / "agents.npz"
    if path.exists():
        if ref_ok(path, rid):
            return dict(status="verified_existing", seconds=0.0)
        quarantine(REFS, path, "unverifiable or reference identity mismatch")
        quarantine(REFS, path.with_suffix(".sha256.json"), "sidecar of quarantined file")
    params = world()
    t0 = time.perf_counter()
    data = load_ref_stream(config, seed, length, rid)
    panels = load_panel_tables(config, rid)
    contexts, starts, lengths, run_ctx = schedule_arrays(seed, length)
    require(np.array_equal(data["contexts"], contexts), "reference contexts differ from the schedule")
    episodes = sample_episodes(length, seed_for("commons_d2_stream", seed))
    events = probe_events(starts, lengths, run_ctx, config["probe_positions"])
    at = {}
    for i, (t, _, _, _) in enumerate(events):
        at.setdefault(int(t), []).append(i)
    n = length
    rows = np.arange(n)
    v1 = data["v1"]
    model = C.schedule_model(schedule_spec(length))
    pool = D0B.lookahead_pool(params)
    npanel = config["panel_size"]
    out = dict(events=events, a_O=v1[rows, contexts].argmax(1))
    stats = {}
    for name in ("I", "L"):
        filt = R.RunLengthFilter(model)
        la = D0B.Lookahead(params, pool, seed_for("commons_d2_L", seed)) if name == "L" else None
        la_probe = D0B.Lookahead(params, pool, seed_for("commons_d2_L_probe", seed)) if name == "L" else None
        acts = np.empty(n, np.int64)
        probe_acts = np.empty((len(events), npanel), np.int64)
        w_true = np.empty(n)
        for t in range(n):
            w, pred = filt.weights()
            for i in at.get(t, []):
                ptab = panels[int(events[i, 3])]
                for s in range(npanel):
                    if la is None:
                        probe_acts[i, s] = R.myopic_action(w, ptab["select"][s])
                    else:
                        probe_acts[i, s] = la_probe.choose(t, w, filt, ptab["select"][s], ptab["reserves"][s], ptab["cue"][s])[0]
            if la is None:
                acts[t] = R.myopic_action(w, v1[t])
            else:
                acts[t] = la.choose(t, w, filt, v1[t], episodes.reserves[t], episodes.cue[t])[0]
            w_true[t] = w[contexts[t]]
            filt.update(data["sensor_ll"][t] + data["pattern_ll"][t, acts[t]], pred)
        out[f"a_{name}"] = acts
        out[f"probe_{name}"] = probe_acts
        out[f"w_true_{name}"] = w_true
        if la is not None:
            stats = dict(decisions_with_lookahead=la.evaluated, decisions_pruned_to_myopic=la.pruned_to_one,
                         probe_decisions_with_lookahead=la_probe.evaluated, probe_pruned=la_probe.pruned_to_one)
    save_arrays(path, meta=dict(ref_identity=rid, seed=seed, length=length, stats=stats, seconds=time.perf_counter() - t0), **out)
    return dict(status="completed", seconds=time.perf_counter() - t0)


def ref_jobs(config, rid, seeds=None, stage="chunks"):
    seeds = config["seeds"] if seeds is None else seeds
    n = config["episodes"]
    if stage == "chunks":
        jobs = [("ref_panel", dict(config=config, rid=rid, ctx=c)) for c in range(N_CONTEXTS)]
        jobs += [("ref_chunk", dict(config=config, rid=rid, seed=s, length=n, start=st))
                 for s in seeds for st in range(0, n, config["oracle"]["stream_chunk"])]
        return jobs
    return [("ref_agents", dict(config=config, rid=rid, seed=s, length=n)) for s in seeds]


class Refs:
    """Verified reference arrays for the analysis."""

    def __init__(self, config, rid):
        self.config, self.rid = config, rid
        self.panels = load_panel_tables(config, rid)
        self.streams = {}

    def stream(self, seed):
        if seed not in self.streams:
            n = self.config["episodes"]
            data = load_ref_stream(self.config, seed, n, self.rid)
            agents, info = load_arrays(ref_dir(seed, n) / "agents.npz")
            require(info["meta"]["ref_identity"] == self.rid, "agents reference identity mismatch")
            data.update({k: v for k, v in agents.items()})
            data["agent_stats"] = info["meta"]["stats"]
            self.streams[seed] = data
        return self.streams[seed]


# ------------------------------------------------------------------ learners

FC_TYPES = ("FC", "FC-single", "FC-K1", "OC")
AM_TYPES = ("AM", "AM-factored")


def make_hyper(config, capacity, episodes=None):
    c = config["core"]
    return L.Hyper(lr=c["lr"], U=c["U"], ema_decay=c["ema_decay"], members=c["members"], hidden=c["hidden"], minibatch=c["minibatch"],
                   grad_clip=c["grad_clip"], bootstrap_p=c["bootstrap_p"], plan_samples=c["plan_samples"], capacity=int(capacity),
                   bounded=True, total_episodes=int(episodes or config["episodes"]))


def learner_seeds(seed, key="commons_d2_learner"):
    """Shared by every arm and knob setting at a seed (common initial randomness)."""
    return {k: seed_for(key, int(seed), k) for k in D.RNG_STREAMS}


def fc_knobs(config, arm, knobs):
    kn = dict(knobs)
    if arm in ("FC", "FC-single"):
        kn["spawn_rule"] = config["fc"]["spawn_rule"]
    if arm in ("FC", "FC-single", "FC-K1", "OC") and "code_decay" not in kn:
        kn["code_decay"] = config["fc"].get("code_decay", 0.0)
    return kn


_BUDGET = {}


def fc_budget_bytes(config):
    """FC's replay budget: reservoir (1,024 entries incl. the 3 per-episode code assignments) + library code values."""
    if "fc" not in _BUDGET:
        lrn = D.make_d2_learner("FC", make_hyper(config, config["core"]["capacity"]), learner_seeds(0), dict(code_lr=0.01, tau=15.0))
        _BUDGET["fc"] = lrn.memory_report()
    return _BUDGET["fc"]["budget_bytes"]


def am_capacity(config, arm, k):
    """AM reservoir capacity (episodes) so that reservoir (with its k stored summaries per entry) + encoder bytes = FC's."""
    key = (arm, int(k))
    if key not in _BUDGET:
        probe = D.make_d2_learner(arm, make_hyper(config, 1), learner_seeds(0), dict(k=k, enc_lr=0.001))
        rec = probe.replay.record_bytes()
        _BUDGET[key] = int(round((fc_budget_bytes(config) - probe.context_bytes()) / rec))
    return _BUDGET[key]


def build_learner(config, arm, knobs, seed, capacity=None, episodes=None):
    cap = capacity or (am_capacity(config, arm, knobs["k"]) if arm in AM_TYPES else config["core"]["capacity"])
    kn = fc_knobs(config, arm, knobs)
    lrn = D.make_d2_learner(arm, make_hyper(config, cap, episodes), learner_seeds(seed), kn, config["core"]["count_c"])
    return lrn, cap


def fit_dir(output, arm, tag, seed, sub="fits"):
    return Path(output) / sub / f"{arm}__{tag}__s{seed}"


# ------------------------------------------------------------------ one fit (resumable bit for bit)

def run_fit(config, identity, output, arm, tag, seed, sub="fits", interrupt_after=None, episodes=None, knobs=None):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = fit_dir(output, arm, tag, seed, sub)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, ckpt_path = directory / "result.json", directory / "checkpoint.pt"
    if result_path.exists():
        try:
            D1.verify_result(directory, identity)
            return dict(status="verified_existing", seconds=0.0)
        except Exception as error:  # noqa: BLE001
            quarantine(output, result_path, f"unverifiable result: {error!r}")
    params = world()
    n = int(episodes or config["episodes"])
    every = config["checkpoint_every"]
    stream = Stream(params, seed, config["episodes"])
    knobs = dict(knobs) if knobs is not None else knobs_for(config, arm, tag)
    learner, cap = build_learner(config, arm, knobs, seed)
    panels = {c: Panel(c, config["panel_size"], config["core"]["plan_samples"]) for c in range(N_CONTEXTS)}
    events = probe_events(stream.starts, stream.lengths, stream.run_context, config["probe_positions"])
    events = events[events[:, 0] < n]
    ev_at = {int(t): i for i, t in enumerate(events[:, 0])}
    switches = set(int(s) for s in stream.starts[1:] if s < n)
    nl = 3
    spec = dict(arm=arm, tag=tag, knobs=fc_knobs(config, arm, knobs), seed=seed, episodes=n, capacity=cap,
                checkpoint_every=every, probe_events=len(events), learner_class=f"{type(learner).__module__}.{type(learner).__qualname__}")
    saved = D1.checkpoint_load(output, ckpt_path, identity)
    resumed = False
    if saved is not None and saved["record"]["spec"] == spec:
        learner.load(saved["learner"])
        record, trace = saved["record"], saved["trace"]
        require(learner.hashes() == record["checkpoint_hashes"], "checkpoint learner state changed")
        record["resumes"].append(dict(utc=now(), episodes_done=record["episodes_done"]))
        resumed = True
    else:
        record = dict(identity=identity, label=LABEL, spec=spec, episodes_done=0, resumes=[], started_utc=now(), training_seconds=0.0,
                      probe_seconds=0.0, elapsed_seconds=0.0, stream_fingerprint=stream.fingerprint(), decisions_chain="",
                      sensors_chain="", outcomes_chain="", initial_hashes=learner.hashes(), resets_at=[])
        trace = dict(action=np.zeros(n, np.int8), base_action=np.zeros(n, np.int8), member=np.zeros(n, np.int8),
                     realised=np.zeros(n, bool), replay_size=np.zeros(n, np.int16), count_of_action=np.zeros(n, np.int16),
                     ctx=np.zeros((n, D.CTX_ALL), np.float16), active=np.full((n, nl), -1, np.int16),
                     assign=np.full((n, nl), -1, np.int16), margin=np.full((n, nl), np.nan, np.float32),
                     post_max=np.full((n, nl), np.nan, np.float32), spawned=np.zeros((n, nl), bool),
                     changed=np.zeros((n, nl), bool), lib_size=np.zeros((n, nl), np.int8),
                     probe_events=events, probe_action=np.zeros((len(events), config["panel_size"]), np.int8),
                     probe_value=np.zeros((len(events), config["panel_size"], L.ACTIONS), np.float32))
    require(record["stream_fingerprint"] == stream.fingerprint(), "stream draws changed")
    code_arm = isinstance(learner, D.CodeLearner)

    def do_probe(i):
        t0 = time.perf_counter()
        panel = panels[int(events[i, 3])]
        before = (learner.hashes(), json.dumps(learner.rng_state(), sort_keys=True, default=str))
        g = learner.greedy(panel.reserves, panel.cue, panel.eps)
        after = (learner.hashes(), json.dumps(learner.rng_state(), sort_keys=True, default=str))
        require(before == after, "the greedy probe changed the learner")
        trace["probe_action"][i] = g["action"]
        trace["probe_value"][i] = g["value"]
        record["probe_seconds"] += time.perf_counter() - t0

    started = time.perf_counter()
    for t in range(record["episodes_done"], n):
        if arm == "W" and t in switches:
            learner.reset()
            record["resets_at"].append(t)
        if t in ev_at:
            do_probe(ev_at[t])
        t0 = time.perf_counter()
        if code_arm:
            if learner.mode == "oracle":
                trace["active"][t] = -1 if learner.last_levels is None else learner.last_levels
            elif learner.mode == "single":
                trace["active"][t, 0] = learner.active[0]
            else:
                trace["active"][t] = learner.active
        decision = stream.decision(t)
        action = learner.act(decision)
        outcome = stream.outcome(t, action)
        learner.observe(decision, action, outcome, stream.label(t) if arm == "OC" else None)
        record["training_seconds"] += time.perf_counter() - t0
        record["decisions_chain"] = chain(record["decisions_chain"], decision.reserves, decision.cue)
        record["sensors_chain"] = chain(record["sensors_chain"], outcome.y_supply, outcome.y_probe)
        record["outcomes_chain"] = chain(record["outcomes_chain"], np.int64(action), outcome.fail_step)
        trace["action"][t] = action
        trace["base_action"][t] = learner.last_base_action
        trace["member"][t] = learner.last_thompson
        trace["realised"][t] = bool(outcome.fail_step.min() == SURVIVED)
        trace["replay_size"][t] = learner.replay.size
        trace["count_of_action"][t] = int(learner.last_counts[action])
        trace["ctx"][t] = learner.last_ctx
        if code_arm:
            if learner.mode == "oracle":
                trace["assign"][t] = stream.levels[t]
            else:
                m = learner.n_lib
                trace["assign"][t, :m] = learner.active
                trace["lib_size"][t, :m] = learner.alive.sum(1)
                for s, info in learner.diag.items():
                    trace["margin"][t, s] = info["margin"]
                    trace["post_max"][t, s] = info["post_max"]
                    trace["spawned"][t, s] = info["spawned"]
                    trace["changed"][t, s] = info["changed"]
        require(learner.replay.size <= cap, "bounded replay exceeded its budget")
        record["episodes_done"] = t + 1
        if (t + 1) % every == 0 and t + 1 < n:
            record["elapsed_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            record["checkpoint_hashes"] = learner.hashes()
            D1.checkpoint_save(ckpt_path, dict(identity=identity, learner=learner.state(), record=record, trace=trace))
        if interrupt_after is not None and t + 1 == interrupt_after:
            raise D1.Interrupted(f"deliberate interruption after episode {t + 1}")
    record["elapsed_seconds"] += time.perf_counter() - started
    info = save_arrays(directory / "trace.npz", meta=dict(arm=arm, tag=tag, seed=seed), **trace)
    extra = {}
    if code_arm:
        extra = dict(final_library_sizes=learner.alive.sum(1).tolist(), spawns=learner.spawns.tolist(), changes=learner.changes.tolist(),
                     final_ids=learner.ids.tolist(), final_alive=learner.alive.tolist())
    record.update(final_hashes=learner.hashes(), replay=dict(capacity=learner.replay.capacity, max_size=learner.replay.max_size,
                                                             seen=learner.replay.seen, evictions=learner.replay.evictions,
                                                             record_bytes=learner.replay.record_bytes()),
                  memory=learner.memory_report(), fc_budget_bytes=fc_budget_bytes(config), optimizer_steps=learner.steps,
                  resets=learner.resets, default_flag_counts=learner.default_flag_counts.tolist(),
                  macs=dict(learner.macs, total=sum(learner.macs.values())), parameters=learner.n_params(), trace_digest=info["digest"],
                  resumed=resumed, completed_utc=now(), **extra)
    record["artifact_hashes"] = {"trace.npz": sha(directory / "trace.npz"), "trace.sha256.json": sha(directory / "trace.sha256.json")}
    record["checkpoint_hashes"] = learner.hashes()
    D1.checkpoint_save(ckpt_path, dict(identity=identity, learner=learner.state(), record=record, trace=trace))
    record["artifact_hashes"]["checkpoint.pt"] = sha(ckpt_path)
    write_json(result_path, record)
    return dict(status="completed", seconds=record["elapsed_seconds"])


# ------------------------------------------------------------------ core parity gate (stationary slice, D1b's protocol)

def parity_dir(output, arm, seed, ctx):
    return Path(output) / "parity" / f"{arm}__s{seed}__{ctx}"


def parity_tasks(config):
    p = config["parity"]
    return [(arm, s, ctx) for arm in ("G", "MB-count") for s in config["seeds"] for ctx in p["contexts"]]


def run_parity_fit(config, identity, output, arm, seed, ctx):
    """G (the D2 core, constant context) or D1b's MB-count on D1's stationary stream of (seed, context) for
    parity.episodes episodes; greedy probe at the end on D1's 512-state panel of that context."""
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = parity_dir(output, arm, seed, ctx)
    result_path = directory / "result.json"
    if result_path.exists():
        try:
            D1.verify_result(directory, identity)
            return dict(status="verified_existing", seconds=0.0)
        except Exception as error:  # noqa: BLE001
            quarantine(output, result_path, f"unverifiable result: {error!r}")
    directory.mkdir(parents=True, exist_ok=True)
    params = world()
    p = config["parity"]
    c = E.context_of(ctx)
    n = p["episodes"]
    t0 = time.perf_counter()
    stream = E.Stream(params, c, seed, n)
    hyper = make_hyper(config, config["core"]["capacity"], n)
    if arm == "G":
        learner = D.make_d2_learner("G", hyper, learner_seeds(seed, f"commons_d2_parity_{ctx}"), {}, config["core"]["count_c"])
    else:
        rngs = {k: np.random.default_rng(v) for k, v in D1.learner_seeds(seed, c, "MB").items()}
        learner = LC.MBCount(hyper, rngs, config["core"]["count_c"])
    actions = np.zeros(n, np.int8)
    for k in range(n):
        dec = stream.decision(k)
        a = learner.act(dec)
        learner.observe(dec, a, stream.outcome(k, a))
        actions[k] = a
    panel = E.Panel(params, c, p["panel_size"], config["core"]["plan_samples"], with_draws=True)
    g = learner.greedy(panel.reserves, panel.cue, eps=panel.eps)
    info = save_arrays(directory / "trace.npz", meta=dict(arm=arm, seed=seed, context=ctx), action=actions,
                       probe_action=g["action"].astype(np.int8), probe_value=g["value"].astype(np.float32),
                       panel_reserves=panel.reserves)
    rec = dict(identity=identity, arm=arm, seed=seed, context=ctx, episodes=n, stream_fingerprint=stream.fingerprint(),
               final_hashes=learner.hashes(), trace_digest=info["digest"], elapsed_seconds=time.perf_counter() - t0, completed_utc=now(),
               artifact_hashes={"trace.npz": sha(directory / "trace.npz"), "trace.sha256.json": sha(directory / "trace.sha256.json")})
    write_json(result_path, rec)
    return dict(status="completed", seconds=rec["elapsed_seconds"])


def d1b_panel(config, ctx):
    arrays, _ = load_arrays(ROOT / config["parity"]["d1b_panels"] / f"panel_{ctx}.npz")
    return arrays


def parity_gate(config, output):
    p = config["parity"]
    rows = []
    for arm, seed, ctx in parity_tasks(config):
        d = parity_dir(output, arm, seed, ctx)
        tr, _ = load_arrays(d / "trace.npz")
        tab = d1b_panel(config, ctx)
        require(np.array_equal(tab["reserves"], tr["panel_reserves"]), "D1b panel states differ from D1's panel")
        reg = D1.panel_regret(tab, E.context_of(ctx), tr["probe_action"].astype(np.int64))
        rows.append(dict(arm=arm, seed=seed, context=ctx, regret=float(reg.mean())))
    g = [r["regret"] for r in rows if r["arm"] == "G"]
    m = [r["regret"] for r in rows if r["arm"] == "MB-count"]
    require(sha(ROOT / p["reference"]["path"]) == p["reference"]["sha256"], "D1b results file changed")
    ref = read_json(ROOT / p["reference"]["path"])
    d1b = ref["R1"]["MB-count"]["context_average"]["mean"]
    require(abs(d1b - p["reference"]["value"]) < 1e-12, "D1b reference value")
    by_ctx = {ctx: dict(G=float(np.mean([r["regret"] for r in rows if r["arm"] == "G" and r["context"] == ctx])),
                        MB_count=float(np.mean([r["regret"] for r in rows if r["arm"] == "MB-count" and r["context"] == ctx])),
                        D1b_test=ref["R1"]["MB-count"]["by_context"][ctx]["mean"]) for ctx in p["contexts"]}
    gm = float(np.mean(g))
    return dict(rule=f"|G context-average greedy regret at {p['episodes']} (dev seeds x {len(p['contexts'])} contexts) - D1b MB-count "
                     f"test context average ({d1b:.5f})| <= {p['tolerance']}",
                G_context_average=gm, MB_count_same_seeds=float(np.mean(m)), D1b_reference=d1b, difference=gm - d1b,
                paired_G_minus_MBcount=float(np.mean(np.array(g) - np.array(m))), by_context=by_ctx, fits=rows,
                passed=bool(abs(gm - d1b) <= p["tolerance"]))


# ------------------------------------------------------------------ pool

DISPATCH = {"fit": run_fit, "parity": run_parity_fit, "ref_panel": run_ref_panel, "ref_chunk": run_ref_chunk,
            "ref_agents": run_ref_agents}
COST = {"G": 1.0, "W": 1.0, "OC": 1.05, "FC-K1": 1.05, "FC": 1.4, "FC-single": 1.5, "AM": 1.3, "AM-factored": 1.5}


def estimated_cost(config, kind, payload):
    n = config["episodes"]
    if kind == "fit":
        k = (payload.get("knobs") or knobs_for(config, payload["arm"], payload["tag"])).get("k", 0)
        return COST[payload["arm"]] * (n * 0.013 + 600) * (1 + k / 16)
    if kind == "parity":
        return config["parity"]["episodes"] * 0.02
    if kind == "ref_panel":
        return 80.0
    if kind == "ref_chunk":
        return 0.17 * config["oracle"]["stream_chunk"]
    return 0.1 * n


def dispatch(kind, payload):
    return DISPATCH[kind](**payload)


def job_label(kind, payload):
    return dict(kind=kind, **{k: payload[k] for k in ("arm", "tag", "seed", "ctx", "start") if k in payload})


def run_pool(config, output, jobs, stage):
    """D1's pool logic: at most config['workers'] single-thread processes; a broken or stalled pool is killed, every
    failure is logged, and the stage retried with a fresh pool (finished tasks only verify)."""
    jobs = sorted(jobs, key=lambda job: -estimated_cost(config, *job))
    for attempt in range(1, config["stage_attempts"] + 1):
        log_event(output, event="stage_start", stage=stage, attempt=attempt, jobs=len(jobs))
        started, failures, statuses = time.perf_counter(), [], []
        if config["workers"] == 1:
            for kind, payload in jobs:
                label = job_label(kind, payload)
                try:
                    result = dispatch(kind, payload)
                    statuses.append(dict(label, status=result.get("status"), seconds=result.get("seconds")))
                    log_event(output, event="task_done", stage=stage, attempt=attempt, **label, status=result.get("status"),
                              seconds=result.get("seconds"))
                except Exception as error:  # noqa: BLE001
                    failures.append(dict(**label, error=repr(error)))
                    log_event(output, event="task_failed", stage=stage, attempt=attempt, **label, error=repr(error),
                              traceback=traceback.format_exc())
                    print(json.dumps(dict(failed=label, error=repr(error))), flush=True)
        else:
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


def fit_jobs(config, identity, output, tasks, sub="fits", knobs=None):
    """knobs: optional {(arm, tag): knobs} for tasks whose knobs are not derivable from the config (FC-K1 in dev)."""
    return [("fit", dict(config=config, identity=identity, output=str(output), arm=a, tag=t, seed=s, sub=sub,
                         **({"knobs": knobs[(a, t)]} if knobs and (a, t) in knobs else {}))) for a, t, s in tasks]


def parity_jobs(config, identity, output):
    return [("parity", dict(config=config, identity=identity, output=str(output), arm=a, seed=s, ctx=c)) for a, s, c in parity_tasks(config)]


def verify_fits(output, identity, tasks, sub="fits"):
    fits = []
    for arm, tag, seed in tasks:
        d = fit_dir(output, arm, tag, seed, sub)
        rec = D1.verify_result(d, identity)
        load_arrays(d / "trace.npz")
        fits.append(dict(arm=arm, tag=tag, seed=seed, path=(d / "result.json").relative_to(output).as_posix(),
                         sha256=sha(d / "result.json"), elapsed_seconds=rec["elapsed_seconds"]))
    return fits


# ------------------------------------------------------------------ metrics of one fit

def visits_of_runs(run_ctx):
    """Visit index (1 = first encounter) of every run."""
    seen, out = {}, []
    for c in run_ctx:
        seen[int(c)] = seen.get(int(c), 0) + 1
        out.append(seen[int(c)])
    return np.array(out, np.int64)


def slope(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 2 or np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def nanmean(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def fit_metrics(config, refs, output, arm, tag, seed, sub="fits"):
    d = fit_dir(output, arm, tag, seed, sub)
    rec = read_json(d / "result.json")
    tr, _ = load_arrays(d / "trace.npz")
    st = refs.stream(seed)
    n = config["episodes"]
    contexts, starts, lengths, run_ctx = schedule_arrays(seed, n)
    events = tr["probe_events"]
    require(np.array_equal(events, st["events"]), "probe events differ between the fit and the references")
    rows = np.arange(n)
    a = tr["action"].astype(np.int64)
    v2 = st["v2"]
    a_L, a_O, a_I = st["a_L"], st["a_O"], st["a_I"]
    excess = v2[rows, a_L] - v2[rows, a]                          # online excess over L (O's scoring table)
    late = rows >= (3 * n) // 4
    # position of each episode in its run, run index
    run_of = np.repeat(np.arange(len(starts)), lengths)
    pos = rows - starts[run_of] + 1
    visits = visits_of_runs(run_ctx)
    # probes
    pex, preg = np.empty(len(events)), np.empty(len(events))
    for i, (t, r, j, c) in enumerate(events):
        tab = refs.panels[int(c)]
        score = tab["score"][:, int(c)]
        s = np.arange(score.shape[0])
        aa = tr["probe_action"][i].astype(np.int64)
        pex[i] = float((score[s, st["probe_L"][i]] - score[s, aa]).mean())
        preg[i] = float((score[s, tab["select"][:, int(c)].argmax(1)] - score[s, aa]).mean())
    ev_visit = visits[events[:, 1]]
    ev_pos = events[:, 2]
    ev_ctx = events[:, 3]
    rmin_sel = (ev_ctx == C.MINORITY) & (ev_visit >= 2) & np.isin(ev_pos, RMIN_POSITIONS)
    rheld_sel = np.isin(ev_ctx, C.HELD_OUT) & (ev_visit == 1) & np.isin(ev_pos, RHELD_POSITIONS)
    m = dict(arm=arm, tag=tag, seed=seed, S=float(excess.mean()), S_late=float(excess[late].mean()),
             S_first_three_quarters=float(excess[~late].mean()), R_min=nanmean(pex[rmin_sel]), R_held=nanmean(pex[rheld_sel]),
             R_min_probes=int(rmin_sel.sum()), R_held_probes=int(rheld_sel.sum()),
             R_min_regret_vs_O=nanmean(preg[rmin_sel]), R_held_regret_vs_O=nanmean(preg[rheld_sel]),
             cumulative_online_excess=float(excess.sum()), realised_survival=float(tr["realised"].mean()),
             L_regret_vs_O_mean=float((v2[rows, a_O] - v2[rows, a_L]).mean()),
             I_excess_over_L_mean=float((v2[rows, a_L] - v2[rows, a_I]).mean()))
    # probe curves: held-out first encounters and minority recurrences, by position
    m["R_held_by_position"] = {str(p): nanmean(pex[rheld_sel & (ev_pos == p)]) for p in RHELD_POSITIONS}
    m["R_min_by_position"] = {str(p): nanmean(pex[rmin_sel & (ev_pos == p)]) for p in RMIN_POSITIONS}
    m["heldout_first_by_context_position"] = {context_name(c): {str(p): nanmean(pex[(ev_ctx == c) & (ev_visit == 1) & (ev_pos == p)])
                                                                for p in RHELD_POSITIONS} for c in C.HELD_OUT}
    # online windows after switches (runs >= 1), truncated runs weighted by episodes
    after = run_of >= 1
    m["online_windows"] = {name: nanmean(excess[after & (pos >= lo) & (pos <= hi)]) for name, (lo, hi) in WINDOWS.items()}
    inf = np.isin(a, D.INFORMATIVE_ACTIONS)
    inf_o = np.isin(a_O, D.INFORMATIVE_ACTIONS)
    m["transfer_2_3_rate"] = {name: nanmean(inf[after & (pos >= lo) & (pos <= hi)]) for name, (lo, hi) in
                              dict(WINDOWS, after_40=(41, 10 ** 9)).items()}
    m["transfer_2_3_rate_O"] = {name: nanmean(inf_o[after & (pos >= lo) & (pos <= hi)]) for name, (lo, hi) in
                                dict(WINDOWS, after_40=(41, 10 ** 9)).items()}
    minority = contexts == C.MINORITY
    usage = np.bincount(a[minority], minlength=L.ACTIONS) / max(minority.sum(), 1)
    f_o = np.bincount(a_O[minority], minlength=L.ACTIONS) / max(minority.sum(), 1)
    m["trap_flag_A1B1C1"] = bool(any(usage[x] < 0.5 * f_o[x] for x in CRITICAL_ACTIONS))
    m["critical_usage_A1B1C1"] = {str(x): [float(usage[x]), float(f_o[x])] for x in CRITICAL_ACTIONS}
    m["bonus_changed_fraction"] = float((tr["action"] != tr["base_action"]).mean())
    # compounding (supporting reading) and taxonomy (descriptive)
    slopes, per_ctx = [], {}
    for c in range(N_CONTEXTS):
        vals = []
        for r in np.flatnonzero((run_ctx == c) & (visits >= 2)):
            sel = (events[:, 1] == r) & np.isin(ev_pos, RMIN_POSITIONS)
            if sel.any():
                vals.append((int(visits[r]), float(pex[sel].mean())))
        if len(vals) >= 2:
            s_ = slope([v for v, _ in vals], [x for _, x in vals])
            slopes.append(s_)
            per_ctx[context_name(c)] = dict(visits=[v for v, _ in vals], excess=[x for _, x in vals], slope=s_)
    m["visit_slope"] = nanmean(slopes)
    m["visit_slope_by_context"] = per_ctx
    first_train = np.isin(ev_ctx, [E.context_of(x) for x in TRAINING_NAMES]) & (ev_visit == 1)
    m["first_encounter_training_early"] = nanmean(pex[first_train])
    m["first_encounter_heldout_late"] = nanmean(pex[np.isin(ev_ctx, C.HELD_OUT) & (ev_visit == 1)])
    stab = []
    for r in np.flatnonzero(visits >= 2):
        prev = np.flatnonzero((run_ctx[:r] == run_ctx[r]))[-1]
        pe = np.flatnonzero(events[:, 1] == prev)
        cur = (events[:, 1] == r) & np.isin(ev_pos, (2, 5))
        if len(pe) and cur.any():
            stab.append(float(pex[cur].mean() - pex[pe[np.argmax(ev_pos[pe])]]))
    m["taxonomy"] = dict(stability_recurrence_minus_end_of_previous_visit=nanmean(stab),
                         plasticity_first_encounters=nanmean(pex[(ev_visit == 1) & (ev_pos >= 5)]),
                         self_selected_trap_flag=m["trap_flag_A1B1C1"])
    # FC diagnostics
    if arm in ("FC", "FC-single", "FC-K1", "OC"):
        levels = np.stack(factors(contexts), 1)
        true_switches = [int((np.diff(levels[:, f]) != 0).sum()) for f in range(3)]
        nl = 1 if arm == "FC-single" else 3
        purity, codes_total, changes = [], [], []
        for s in range(nl):
            fs = range(3) if arm == "FC-single" else [s]
            assign = tr["assign"][:, s]
            pur_f = []
            for f in fs:
                ok = 0
                for cid in np.unique(assign):
                    sel = assign == cid
                    frac = levels[sel, f].mean()
                    ok += sel.sum() * max(frac, 1 - frac)
                pur_f.append(float(ok / n))
            purity.append(float(np.mean(pur_f)))
            codes_total.append(int(rec.get("spawns", [0] * nl)[s] + (2 if arm == "OC" else 1)))
            changes.append(int(rec.get("changes", [0] * nl)[s]))
        m["fc"] = dict(true_switches_by_factor=true_switches, purity_by_slot=purity, mean_purity=float(np.mean(purity)),
                       codes_created_by_slot=codes_total, final_library_sizes=rec.get("final_library_sizes"),
                       active_changes_by_slot=changes, spawns=rec.get("spawns"),
                       changes_over_switches=[c_ / max(true_switches[s if arm != "FC-single" else 0], 1) for s, c_ in enumerate(changes)],
                       default_minibatch_fraction=[x / max(rec["optimizer_steps"], 1) for x in rec["default_flag_counts"]])
        if arm == "FC-single":
            m["fc"]["changes_over_switches"] = [changes[0] / max(sum(1 for i in range(1, n) if contexts[i] != contexts[i - 1]), 1)]
    m["cost"] = dict(wall_seconds=rec["elapsed_seconds"], training_seconds=rec["training_seconds"], probe_seconds=rec["probe_seconds"],
                     macs=rec["macs"]["total"], budget_bytes=rec["memory"]["budget_bytes"], replay_capacity=rec["memory"]["replay_capacity"],
                     replay_record_bytes=rec["memory"]["replay_record_bytes"], context_bytes=rec["memory"]["context_bytes"],
                     fc_budget_bytes=rec["fc_budget_bytes"], byte_ratio_to_fc=rec["memory"]["budget_bytes"] / rec["fc_budget_bytes"])
    m["_probe_excess"] = pex
    m["_events"] = events
    m["_visits"] = ev_visit
    return m


# ------------------------------------------------------------------ arm-level analysis

def seedvals(rows, arm, tag, key, seeds):
    by = {(r["arm"], r["tag"], r["seed"]): r for r in rows}
    return np.array([by[(arm, tag, s)][key] for s in seeds], np.float64)


def oc_gate(config, rows, oc_tag, g_tag, seeds):
    a = config["analysis"]
    by = {(r["arm"], r["tag"], r["seed"]): r for r in rows}
    per_pos = {}
    for p in a["oc_gate_positions"]:
        vals = []
        for s in seeds:
            r = by[("OC", oc_tag, s)]
            sel = np.isin(r["_events"][:, 3], C.HELD_OUT) & (r["_visits"] == 1) & (r["_events"][:, 2] == p)
            vals += list(r["_probe_excess"][sel])
        per_pos[str(p)] = dict(mean=nanmean(vals), probes=len(vals))
    comp = all(v["mean"] <= a["oc_gate_max"] for v in per_pos.values())
    beats = []
    for s in seeds:
        g, o = by[("G", g_tag, s)], by[("OC", oc_tag, s)]
        beats.append(dict(seed=s, G_minus_OC_R_held=g["R_held"] - o["R_held"], G_minus_OC_R_min=g["R_min"] - o["R_min"],
                          passes=bool(g["R_held"] - o["R_held"] >= a["v1_margin"] and g["R_min"] - o["R_min"] >= a["v1_margin"])))
    n_beat = sum(b["passes"] for b in beats)
    return dict(rule=("OC's held-out first-encounter probe excess over L <= .010 at every position 5, 10, 20, 40 (mean over dev seeds x "
                      "held-out combinations reaching that position) AND OC beats G V1-like (G - OC >= .005 on R-held and on R-min) "
                      "in >= 2 of 3 seeds"),
                competence_by_position=per_pos, competence_passed=bool(comp), beats_G=beats, seeds_beating_G=int(n_beat),
                passed=bool(comp and n_beat >= 2))


def pass_condition(fc, am, delta, rel):
    return fc - am <= -max(rel * am, delta)


def binom_tail(p, n, k):
    return float(sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def power_rule(diffs, am_vals, delta, rel, target):
    """P(seed passes) under d ~ N(dev mean, dev SD); threshold at the dev-mean AM* value; binomial tails."""
    mu, sd = float(np.mean(diffs)), float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    thr = -max(rel * float(np.mean(am_vals)), delta)
    if sd > 0:
        p = 0.5 * math.erfc(-(thr - mu) / (sd * math.sqrt(2)))
    else:
        p = float(mu <= thr)
    p12, p24 = binom_tail(p, 12, 10), binom_tail(p, 24, 20)
    return dict(mean_difference=mu, sd_difference=sd, threshold=thr, p_seed_pass=p, P_ge_10_of_12=p12, P_ge_20_of_24=p24,
                use_24=bool(p12 < target and p24 >= target))


def pathology(config, rows, tag, seeds):
    fc = config["fc"]
    per = []
    for s in seeds:
        r = [x for x in rows if x["arm"] == "FC" and x["tag"] == tag and x["seed"] == s][0]["fc"]
        churn = any(c > 3 * max(sw, 0) for c, sw in zip(r["active_changes_by_slot"], r["true_switches_by_factor"]))
        many = any(c > fc["max_codes"] for c in r["codes_created_by_slot"])
        impure = r["mean_purity"] < 0.6
        per.append(dict(seed=s, changes=r["active_changes_by_slot"], true_switches=r["true_switches_by_factor"],
                        codes_created=r["codes_created_by_slot"], mean_purity=r["mean_purity"], purity_by_slot=r["purity_by_slot"],
                        churn=bool(churn), too_many_codes=bool(many), impure=bool(impure), fails=bool(churn or many or impure)))
    return dict(rule=("FC fails if, in any development seed (per-seed reading, as Step C): active-code changes > 3 x true switches of "
                      "the slot's factor in any slot; > 6 codes created in any slot (the library itself is capped at 6); or mean "
                      "code-to-factor purity (mean over slots of the episode-weighted majority-level fraction) < .6"),
                per_seed=per, fails=bool(any(p["fails"] for p in per)))


def arm_table(rows, seeds):
    out = {}
    keys = sorted({(r["arm"], r["tag"]) for r in rows}, key=lambda x: (ARMS.index(x[0]), x[1]))
    for arm, tag in keys:
        rs = [r for r in rows if r["arm"] == arm and r["tag"] == tag and r["seed"] in seeds]
        if len(rs) != len(seeds):
            continue
        out[f"{arm}|{tag}"] = dict(arm=arm, tag=tag, **{k: float(np.mean([r[k] for r in rs])) for k in
                                                       ("S", "S_late", "R_min", "R_held", "visit_slope", "first_encounter_training_early",
                                                        "first_encounter_heldout_late", "realised_survival")},
                                   S_per_seed=[r["S"] for r in rs], R_min_per_seed=[r["R_min"] for r in rs],
                                   R_held_per_seed=[r["R_held"] for r in rs],
                                   bytes=float(np.mean([r["cost"]["budget_bytes"] for r in rs])),
                                   byte_ratio_to_fc=float(np.mean([r["cost"]["byte_ratio_to_fc"] for r in rs])),
                                   replay_capacity=int(rs[0]["cost"]["replay_capacity"]),
                                   wall_minutes=float(np.mean([r["cost"]["wall_seconds"] for r in rs]) / 60),
                                   online_windows={w: float(np.mean([r["online_windows"][w] for r in rs])) for w in WINDOWS},
                                   transfer_2_3_rate={w: float(np.mean([r["transfer_2_3_rate"][w] for r in rs])) for w in rs[0]["transfer_2_3_rate"]},
                                   trap_flags=int(sum(r["trap_flag_A1B1C1"] for r in rs)),
                                   taxonomy={k: float(np.mean([r["taxonomy"][k] for r in rs])) for k in rs[0]["taxonomy"]},
                                   fc={k: [r["fc"][k] for r in rs] for k in rs[0]["fc"]} if "fc" in rs[0] else None)
    return out


def select_by_S(table, arm):
    cands = [v for v in table.values() if v["arm"] == arm]
    best = min(cands, key=lambda v: v["S"])
    return best["tag"], [dict(tag=v["tag"], S=v["S"], R_min=v["R_min"], R_held=v["R_held"]) for v in cands]


def criteria(config, rows, seeds, tags, delta):
    """V1, V2, C1-C3, normalised gains and the compounding reading for FC vs AM* (per seed and seed mean)."""
    a = config["analysis"]
    val = {arm: {k: seedvals(rows, arm if arm != "AM*" else tags["AM*arm"], tags[arm], k, seeds) for k in ("S", "S_late", "R_min", "R_held", "visit_slope")}
           for arm in ("G", "OC", "FC", "AM*")}
    out = {}
    need = len(seeds) if len(seeds) < 12 else (10 if len(seeds) == 12 else 20)
    v1 = [(val["G"]["R_held"][i] - val["OC"]["R_held"][i] >= a["v1_margin"]) and (val["G"]["R_min"][i] - val["OC"]["R_min"][i] >= a["v1_margin"])
          for i in range(len(seeds))]
    out["V1"] = dict(per_seed=[bool(x) for x in v1], seeds_passing=int(sum(v1)), required=need, passes=bool(sum(v1) >= need),
                     G_minus_OC_R_held=(val["G"]["R_held"] - val["OC"]["R_held"]).tolist(),
                     G_minus_OC_R_min=(val["G"]["R_min"] - val["OC"]["R_min"]).tolist())
    gm = {k: float(val["G"][k].mean()) for k in ("R_held", "R_min", "S")}
    om = {k: float(val["OC"][k].mean()) for k in ("R_held", "R_min", "S")}
    amm = {k: float(val["AM*"][k].mean()) for k in ("R_held", "R_min", "S")}
    out["V2"] = dict(G_minus_AMstar_R_held=gm["R_held"] - amm["R_held"], half_headroom=a["v2_fraction"] * (gm["R_held"] - om["R_held"]),
                     passes=bool(gm["R_held"] - amm["R_held"] >= a["v2_fraction"] * (gm["R_held"] - om["R_held"])))
    out["C1"] = dict(S_FC=float(val["FC"]["S"].mean()), S_AMstar=amm["S"], passes=bool(val["FC"]["S"].mean() <= amm["S"] + a["c1_margin"]))
    for name, key in (("C2", "R_min"), ("C3", "R_held")):
        fcv, amv = val["FC"][key], val["AM*"][key]
        per = [bool(pass_condition(fcv[i], amv[i], delta[key], a["relative_margin"])) for i in range(len(seeds))]
        mean_ok = bool(pass_condition(float(fcv.mean()), float(amv.mean()), delta[key], a["relative_margin"]))
        out[name] = dict(outcome=key, FC=fcv.tolist(), AMstar=amv.tolist(), difference=(fcv - amv).tolist(), delta=delta[key],
                         per_seed_pass=per, seeds_passing=int(sum(per)), required=need, seed_mean_pass=mean_ok,
                         passes=bool(sum(per) >= need and mean_ok))
    gains = {}
    for arm in ("FC", "AM*", "OC", "G"):
        v = val[arm]
        gains[arm] = {k: ((gm[k] - float(v[k].mean())) / (gm[k] - om[k]) if abs(gm[k] - om[k]) > 1e-12 else float("nan"))
                      for k in ("R_held", "R_min", "S")}
    out["normalised_gain"] = gains
    out["headroom_gap_G_minus_OC"] = dict(R_held=gm["R_held"] - om["R_held"], R_min=gm["R_min"] - om["R_min"])
    fs, ams = val["FC"]["visit_slope"], val["AM*"]["visit_slope"]
    more_neg = [bool(fs[i] < ams[i]) for i in range(len(seeds))]
    out["compounding"] = dict(visit_slope_FC=fs.tolist(), visit_slope_AMstar=ams.tolist(), FC_more_negative=more_neg,
                              seeds_FC_more_negative=int(sum(more_neg)),
                              payback=dict(FC=dict(S=float(val["FC"]["S"].mean()), S_late=float(val["FC"]["S_late"].mean())),
                                           AMstar=dict(S=amm["S"], S_late=float(val["AM*"]["S_late"].mean()))))
    v1p, v2p = out["V1"]["passes"], out["V2"]["passes"]
    c1, c2, c3 = out["C1"]["passes"], out["C2"]["passes"], out["C3"]["passes"]
    if not v1p:
        verdict = "UNINFORMATIVE"
    elif not v2p:
        verdict = "COMPETITOR TOO WEAK"
    elif c1 and c2 and c3:
        verdict = "STRONG GO"
    elif c1 and (c2 != c3):
        verdict = "QUALIFIED GO"
    else:
        verdict = "MECHANISM NO-GO"
    out["verdict_rule_applied"] = verdict
    out["flag_compounding_vs_nogo"] = bool(verdict in ("MECHANISM NO-GO",) and sum(more_neg) >= (9 if len(seeds) == 12 else len(seeds)))
    return out


# ------------------------------------------------------------------ development

def progress(output, **state):
    write_json(Path(output) / "progress.json", dict(utc=now(), **state))
    log_event(output, event="progress", **state)


def strip(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def clean(value):
    """JSON-safe copy: non-finite floats (undefined readings, e.g. a visit slope with < 2 recurrences) become null."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def cmd_dev(args):
    config = load_config(args.config)
    require(config["phase"] == "dev", "dev runs the development configuration")
    output, report = Path(args.output), Path(args.report)
    identity = lock(config, args.config, output)["identity"]
    rid = ref_identity(config)
    log_event(output, event="dev_start", identity=identity, ref_identity=rid)
    seeds = config["seeds"]
    t0 = time.perf_counter()
    # 1. references (O tables, likelihoods) + the two gates' fits
    progress(output, stage="1 references + gate fits (G, OC) + core parity fits")
    run_pool(config, output, ref_jobs(config, rid) + fit_jobs(config, identity, output, fit_tasks(config, "gates"))
             + parity_jobs(config, identity, output), "dev_refs_gates")
    progress(output, stage="2 reference agents (I-myopic, L incl. probe panels)")
    run_pool(config, output, ref_jobs(config, rid, stage="agents"), "dev_ref_agents")
    refs = Refs(config, rid)
    gate_tasks = fit_tasks(config, "gates")
    verify_fits(output, identity, gate_tasks)
    rows = [fit_metrics(config, refs, output, *t) for t in gate_tasks]
    g_tag, oc_tag = "base", arm_tag("OC", {"code_lr": config["oc_code_lr"]}, config)
    gates = dict(oc=oc_gate(config, rows, oc_tag, g_tag, seeds), core_parity=parity_gate(config, output), utc=now(), identity=identity)
    gates["passed"] = bool(gates["oc"]["passed"] and gates["core_parity"]["passed"])
    write_json(report / "gates.json", clean(gates))
    print(json.dumps(dict(oc_gate=gates["oc"]["passed"], oc_competence=gates["oc"]["competence_by_position"],
                          oc_beats_G=gates["oc"]["seeds_beating_G"], parity=gates["core_parity"]["passed"],
                          parity_G=gates["core_parity"]["G_context_average"], parity_D1b=gates["core_parity"]["D1b_reference"]), indent=1),
          flush=True)
    if not gates["passed"] and not args.continue_after_gate_failure:
        progress(output, stage="STOPPED: a development gate failed", gates=dict(oc=gates["oc"]["passed"], parity=gates["core_parity"]["passed"]))
        raise SystemExit("A development gate failed: fix the implementation (allowed), then rerun dev. See gates.json.")
    # 3. declared knob grids (+ W)
    progress(output, stage="3 knob grids (AM, AM-factored, FC, FC-single) + W")
    grid = fit_tasks(config, "grid")
    run_pool(config, output, fit_jobs(config, identity, output, grid), "dev_grid")
    verify_fits(output, identity, grid)
    rows += [fit_metrics(config, refs, output, *t) for t in grid]
    table = arm_table(rows, seeds)
    fc_tag, _ = select_by_S(table, "FC")
    fc_knob = knobs_for(config, "FC", fc_tag)
    # 4. FC-K1 at FC's selected code lr (the library disabled; also = FC under the literal spawn rule, see check)
    progress(output, stage="4 FC-K1 at the selected FC code lr", fc_selected=fc_tag)
    k1_tasks = [("FC-K1", arm_tag("FC-K1", {"code_lr": fc_knob["code_lr"]}, config), s) for s in seeds]
    run_pool(config, output, fit_jobs(config, identity, output, k1_tasks), "dev_fc_k1")
    verify_fits(output, identity, k1_tasks)
    rows += [fit_metrics(config, refs, output, *t) for t in k1_tasks]
    check_lock(config, args.config, output)
    all_tasks = gate_tasks + grid + k1_tasks
    fits = verify_fits(output, identity, all_tasks)
    crc = D1.crc_scan(output)
    write_json(output / "crc_scan.json", crc)
    require(crc["passed"], f"CRC scan failed: {crc['failures']}")
    write_json(output / "completion.json", dict(identity=identity, ref_identity=rid, fits=fits, crc=crc, utc=now(),
                                                wall_seconds_this_invocation=time.perf_counter() - t0))
    dev_report(config, rows, gates, report, identity, rid, crc, args)
    progress(output, stage="done")


def test_estimate(config, table, sel, refs_seconds, seeds_n):
    per = {}
    for arm in ARMS:
        tag = sel.get(arm)
        key = f"{arm}|{tag}"
        if key in table:
            per[arm] = table[key]["wall_minutes"]
    fits_cpu_h = sum(per.values()) * seeds_n / 60
    refs_cpu_h = refs_seconds * seeds_n / 3600
    return dict(minutes_per_fit_by_arm=per, seeds=seeds_n, fits_cpu_hours=fits_cpu_h, references_cpu_hours=refs_cpu_h,
                replicate_cpu_hours=3 * (per.get("FC", 0) + per.get(sel.get("AM*arm"), 0)) / 60,
                wall_hours_on_6_workers=(fits_cpu_h + refs_cpu_h) / 6 + 3 * (per.get("FC", 0) + per.get(sel.get("AM*arm"), 0)) / 60 / 6,
                note="dev wall minutes were measured with 6 concurrent single-thread workers; the L pass is sequential per seed")


def dev_report(config, rows, gates, report, identity, rid, crc, args):
    seeds = config["seeds"]
    a = config["analysis"]
    table = arm_table(rows, seeds)
    sel, grids = {}, {}
    for arm in KNOBS:
        sel[arm], grids[arm] = select_by_S(table, arm)
    sel.update({"G": "base", "W": "base", "OC": arm_tag("OC", {"code_lr": config["oc_code_lr"]}, config),
                "FC-K1": arm_tag("FC-K1", {"code_lr": knobs_for(config, "FC", sel["FC"])["code_lr"]}, config)})
    am_star = min(("AM", "AM-factored"), key=lambda x: table[f"{x}|{sel[x]}"]["S"])
    sel["AM*arm"] = am_star
    tags = {"G": "base", "OC": sel["OC"], "FC": sel["FC"], "AM*": sel[am_star], "AM*arm": am_star}
    path_ = pathology(config, rows, sel["FC"], seeds)
    delta, power = {}, {}
    for key in ("R_min", "R_held"):
        fcv = seedvals(rows, "FC", sel["FC"], key, seeds)
        amv = seedvals(rows, am_star, sel[am_star], key, seeds)
        diffs = fcv - amv
        delta[key] = float(np.std(diffs, ddof=1))
        power[key] = power_rule(diffs, amv, delta[key], a["relative_margin"], a["power_target"])
    use_24 = bool(power["R_min"]["use_24"] or power["R_held"]["use_24"])
    crit = criteria(config, rows, seeds, tags, delta)
    refs_seconds = 0.0
    for s in seeds:
        for start in range(0, config["episodes"], config["oracle"]["stream_chunk"]):
            refs_seconds += load_arrays(ref_dir(s, config["episodes"]) / f"chunk_{start:05d}.npz")[1]["meta"]["seconds"]
        refs_seconds += load_arrays(ref_dir(s, config["episodes"]) / "agents.npz")[1]["meta"]["seconds"]
    refs_seconds /= len(seeds)
    est = test_estimate(config, table, sel, refs_seconds, 24 if use_24 else 12)
    literal_note = ("The FC spawn rule used here is '" + config["fc"]["spawn_rule"] + "'. Under the protocol's literal rule a "
                    "one-code library's default IS that code, so no spawn can ever happen and FC is bit-identical to FC-K1 at the "
                    "same code lr (verified in the smoke check); FC-K1 below therefore also reports literal FC at the selected code lr.")
    out = dict(label="DEVELOPMENT (seeds 22101-22103). Descriptive; not a test result. " + LABEL, identity=identity, ref_identity=rid,
               gates=gates, fc_spawn_rule=config["fc"]["spawn_rule"], fc_spawn_rule_note=literal_note,
               selection=dict(knobs={k: v for k, v in sel.items() if k != "AM*arm"}, AM_star=am_star, grids=grids,
                              rule="lowest development whole-stream excess S (seed mean); AM* = the tuned AM or AM-factored with the lower S"),
               pathology=path_, delta=delta, power=power, test_seeds=24 if use_24 else 12,
               power_rule="24 seeds iff, for C2 or C3, P(>= 10/12) < .8 and P(>= 20/24) >= .8 at the development effect size and SD",
               criteria_dev_level=crit, arms=table, test_estimate=est, crc=crc, utc=now(),
               per_fit=[strip(r) for r in rows])
    write_json(report / "dev_results.json", clean(out))
    D1.write_csv(report / "dev_fits.csv", [dict(arm=r["arm"], tag=r["tag"], seed=r["seed"], S=r["S"], S_late=r["S_late"], R_min=r["R_min"],
                                                R_held=r["R_held"], visit_slope=r["visit_slope"], bytes=r["cost"]["budget_bytes"],
                                                capacity=r["cost"]["replay_capacity"], wall_min=r["cost"]["wall_seconds"] / 60,
                                                trap=r["trap_flag_A1B1C1"]) for r in rows])
    draft = dict(read_json(args.config))
    draft.update(phase="test", seeds=TEST_SEEDS_24 if use_24 else TEST_SEEDS_12,
                 knobs={**{arm: knobs_for(config, arm, sel[arm]) for arm in KNOBS}, "FC-K1": {"code_lr": knobs_for(config, "FC", sel["FC"])["code_lr"]},
                        "AM*": am_star, "delta": delta},
                 notes=("DRAFT from development (reports/commons_d2/dev/dev_results.json), NOT LOCKED. Knobs by lowest dev S; AM* = "
                        f"{am_star}; delta = dev per-seed SD of FC - AM* (R-min {delta['R_min']:.5f}, R-held {delta['R_held']:.5f}); "
                        f"seeds by the power rule ({'24' if use_24 else '12'}). FC spawn rule '{config['fc']['spawn_rule']}'"
                        + (" is a PROPOSED REVISION of the protocol text and needs the lead's approval before any lock." if config["fc"]["spawn_rule"] != "literal" else ".")
                        + (" FC FAILED the pathology stop in development." if path_["fails"] else "")))
    write_json(ROOT / "configs" / "commons_d2.json", clean(draft))
    print(json.dumps(dict(selection=out["selection"]["knobs"], AM_star=am_star, pathology=path_["fails"], delta=delta,
                          test_seeds=out["test_seeds"], verdict_rule_on_dev=crit["verdict_rule_applied"],
                          arms={k: dict(S=round(v["S"], 5), R_min=round(v["R_min"], 5), R_held=round(v["R_held"], 5)) for k, v in table.items()}),
                     indent=1), flush=True)
    return out


def cmd_dev_report(args):
    """Re-analysis of finished development fits against the STORED dev manifest (not the live files), so an analysis-code fix
    does not invalidate the fits. The live script/source hashes are recorded next to the locked ones."""
    config = read_json(Path(args.output) / "config_at_lock.json")
    validate_config(config)
    output, report = Path(args.output), Path(args.report)
    manifest = read_json(output / "manifest.json")
    identity = manifest["identity"]
    live = lock_payload(config, Path(args.output) / "config_at_lock.json")
    rid = ref_identity(config)
    refs = Refs(config, rid)
    seeds = config["seeds"]
    gate_tasks, grid = fit_tasks(config, "gates"), fit_tasks(config, "grid")
    rows = [fit_metrics(config, refs, output, *t) for t in gate_tasks + grid]
    verify_fits(output, identity, gate_tasks + grid)
    table = arm_table(rows, seeds)
    fc_tag, _ = select_by_S(table, "FC")
    k1 = [("FC-K1", arm_tag("FC-K1", {"code_lr": knobs_for(config, "FC", fc_tag)["code_lr"]}, config), s) for s in seeds]
    verify_fits(output, identity, k1)
    rows += [fit_metrics(config, refs, output, *t) for t in k1]
    g_tag, oc_tag = "base", arm_tag("OC", {"code_lr": config["oc_code_lr"]}, config)
    gates = dict(oc=oc_gate(config, rows, oc_tag, g_tag, seeds), core_parity=parity_gate(config, output), utc=now(), identity=identity)
    gates["passed"] = bool(gates["oc"]["passed"] and gates["core_parity"]["passed"])
    crc = D1.crc_scan(output)
    require(crc["passed"], "CRC scan failed")
    args.config = str(Path(args.output) / "config_at_lock.json")
    out = dev_report(config, rows, gates, report, identity, rid, crc, args)
    out["reanalysis"] = dict(locked_scripts=manifest["script_files"], live_scripts=live["script_files"],
                             locked_source=manifest["source_files"], live_source=live["source_files"], utc=now())
    write_json(report / "dev_results.json", clean(out))
    return out


# ------------------------------------------------------------------ test (lock, run, replicate, analyze)

def cmd_lock(args):
    config = load_config(args.config)
    require(config["phase"] in ("test", "smoke"), "lock is for the smoke or test config")
    print(json.dumps(lock(config, args.config, args.output)["identity"], indent=1))


def cmd_run(args):
    config = load_config(args.config)
    output = Path(args.output)
    identity = check_lock(config, args.config, output)["identity"]
    rid = ref_identity(config)
    log_event(output, event="run_start", identity=identity, ref_identity=rid)
    t0 = time.perf_counter()
    tasks = fit_tasks(config)
    statuses = run_pool(config, output, ref_jobs(config, rid) + fit_jobs(config, identity, output, tasks), "run")
    statuses += run_pool(config, output, ref_jobs(config, rid, stage="agents"), "run_agents")
    check_lock(config, args.config, output)
    fits = verify_fits(output, identity, tasks)
    Refs(config, rid)
    crc = D1.crc_scan(output)
    write_json(output / "crc_scan.json", crc)
    require(crc["passed"], f"CRC scan failed: {crc['failures']}")
    completion = dict(identity=identity, ref_identity=rid, fits=fits, statuses=statuses, crc=crc, utc=now(),
                      wall_seconds_this_invocation=time.perf_counter() - t0)
    write_json(output / "completion.json", completion)
    log_event(output, event="run_complete", seconds=time.perf_counter() - t0)
    return completion


def replicate_tasks(config):
    """Test: 3 seeds x {FC, AM*}. Smoke: every fit."""
    tasks = fit_tasks(config)
    if config["phase"] != "test":
        return tasks
    seeds = config["seeds"][:: max(1, len(config["seeds"]) // 3)][:3]
    return [t for t in tasks if t[2] in seeds and t[0] in ("FC", config["knobs"]["AM*"])]


def cmd_replicate(args):
    config = load_config(args.config)
    output = Path(args.output)
    identity = check_lock(config, args.config, output)["identity"]
    completion = read_json(output / "completion.json")
    for f in completion["fits"]:
        require(sha(output / f["path"]) == f["sha256"], f"file changed since completion: {f['path']}")
    tasks = fit_tasks(config)
    verify_fits(output, identity, tasks)
    Refs(config, completion["ref_identity"])
    rtasks = replicate_tasks(config)
    t0 = time.perf_counter()
    run_pool(config, output, fit_jobs(config, identity, output, rtasks, sub="replicate"), "replicate")
    comparisons, mismatches = {}, []
    for arm, tag, seed in rtasks:
        rows = D1.compare_fits(fit_dir(output, arm, tag, seed), fit_dir(output, arm, tag, seed, "replicate"))
        key = f"{arm}:{tag}:{seed}"
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
    verify_fits(output, identity, tasks)
    crc = D1.crc_scan(output)
    require(crc["passed"], "CRC scan failed")
    refs = Refs(config, ref_identity(config))
    rows = [fit_metrics(config, refs, output, *t) for t in tasks]
    kn = config["knobs"]
    am_star = kn["AM*"]
    tags = {"G": "base", "OC": arm_tag("OC", {"code_lr": config["oc_code_lr"]}, config), "FC": arm_tag("FC", kn["FC"], config),
            "AM*": arm_tag(am_star, kn[am_star], config), "AM*arm": am_star}
    crit = criteria(config, rows, config["seeds"], tags, kn["delta"])
    fc_path = pathology(config, rows, tags["FC"], config["seeds"]) if config["phase"] == "test" else None
    out = dict(label=LABEL + (" SMOKE ONLY - NOT SCIENCE." if config["phase"] == "smoke" else ""), identity=identity, knobs=kn,
               criteria=crit, verdict=crit["verdict_rule_applied"], fc_pathology_on_test=fc_path, arms=arm_table(rows, config["seeds"]),
               crc=crc, fits=len(rows), utc=now(), per_fit=[strip(r) for r in rows])
    write_json(report / "results.json", clean(out))
    D1.write_csv(report / "fits.csv", [dict(arm=r["arm"], tag=r["tag"], seed=r["seed"], S=r["S"], S_late=r["S_late"], R_min=r["R_min"],
                                            R_held=r["R_held"], visit_slope=r["visit_slope"]) for r in rows])
    print(json.dumps(dict(verdict=out["verdict"], V1=crit["V1"]["passes"], V2=crit["V2"]["passes"], C1=crit["C1"]["passes"],
                          C2=crit["C2"]["passes"], C3=crit["C3"]["passes"]), indent=1), flush=True)
    return out


# ------------------------------------------------------------------ smoke checks

def import_check():
    tree = ast.parse((ROOT / "src/acp_cl/commons/learners_d2.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {x.name for x in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add("." * node.level + (node.module or ""))
    allowed = {"__future__", "dataclasses", "math", "numpy", "torch", "torch.nn", ".learners", "itertools", "copy", "hashlib"}
    return dict(imports=sorted(mods), allowed=sorted(allowed), passed=bool(mods <= allowed))


def informative_check(params):
    ch, donor, rec, amount = action_table(params)
    derived = [a for a in range(len(ch)) if ch[a] == 1]
    return dict(derived_2_3_actions=derived, declared=list(D.INFORMATIVE_ACTIONS), passed=derived == list(D.INFORMATIVE_ACTIONS))


def rejects(fn):
    try:
        fn()
    except L.IllegalObservation:
        return True
    return False


def legal_check(config, params):
    st = Stream(params, SMOKE_SEED, config["episodes"])
    d, o = st.decision(0), st.outcome(0, 3)
    extra = L.make_decision(d.reserves, d.cue)
    object.__setattr__(extra, "context", 7)
    extra_o = L.make_outcome(o.y_supply, o.y_probe, o.fail_step)
    object.__setattr__(extra_o, "supplies", st.supplies[0])
    lab = st.label(0)
    bad_lab = D.Label(np.array(st.levels[0], np.uint8))
    rows = {}
    for arm in ARMS:
        kn = {"k": 8, "enc_lr": 0.003, "code_lr": 0.03, "tau": 15.0}
        lrn, _ = build_learner(config, arm, kn if arm != "OC" else {"code_lr": 0.03}, SMOKE_SEED, capacity=64)
        r = dict(act_rejects_extra=rejects(lambda: lrn.act(extra)), observe_rejects_extra=rejects(lambda: lrn.observe(d, 3, extra_o, lab if arm == "OC" else None)),
                 observe_rejects_bad_action=rejects(lambda: lrn.observe(d, 13, o, lab if arm == "OC" else None)),
                 rejects_writeable_label=rejects(lambda: lrn.observe(d, 3, o, bad_lab)))
        if arm == "OC":
            r["oc_requires_label"] = rejects(lambda: lrn.observe(d, 3, o, None))
            lrn.act(d)
            lrn.observe(d, 3, o, lab)
            r["oc_accepts_label_after_outcome"] = True
        else:
            r["rejects_label"] = rejects(lambda: lrn.observe(d, 3, o, lab))
        rows[arm] = r
    ok = all(all(v.values()) for v in rows.values())
    return dict(per_arm=rows, passed=bool(ok))


def filter_check():
    rng = np.random.default_rng(seed_for("commons_d2_filter_check"))
    worst = 0.0
    cases = 0
    for k in (2, 3):
        for tt in range(1, 7):
            for _ in range(3):
                ll = rng.normal(0, 3, (tt, k))
                lp = np.full(k, -math.log(k))
                for t in range(tt):
                    lp = D.sticky_step(lp, ll[t], 0.05) if t > 0 else (lambda x: x - D.logsumexp_np(x))(lp + ll[t])
                bf = D.sticky_bruteforce(ll, 0.05)
                worst = max(worst, float(np.max(np.abs(np.exp(lp) - np.exp(bf)))))
                cases += 1
    return dict(cases=cases, max_abs_posterior_difference=worst,
                note="first step = prior x likelihood (uniform prior), later steps = D.sticky_step as used by FC",
                passed=bool(worst < 1e-10))


def byte_check(config):
    fc = fc_budget_bytes(config)
    rows = {}
    for arm in AM_TYPES:
        for k in GRID[arm]["k"]:
            cap = am_capacity(config, arm, k)
            lrn = D.make_d2_learner(arm, make_hyper(config, cap), learner_seeds(0), dict(k=k, enc_lr=0.001))
            b = lrn.memory_report()
            rows[f"{arm}|k{k}"] = dict(capacity=cap, record_bytes=b["replay_record_bytes"], encoder_bytes=b["context_bytes"],
                                       budget_bytes=b["budget_bytes"], ratio_to_fc=b["budget_bytes"] / fc)
    return dict(fc_budget_bytes=fc, am=rows, passed=bool(all(abs(r["ratio_to_fc"] - 1) <= 0.05 for r in rows.values())))


def literal_equivalence_check(config, identity, output):
    """Under the protocol's literal spawn rule FC never spawns (a one-code library's default is that code) and is
    bit-identical to FC-K1 at the same code lr."""
    base = fresh(output / "checks" / "literal")
    cfg = dict(config, fc=dict(config["fc"], spawn_rule="literal"))
    kn = {"tau": 5, "code_lr": 0.03}
    run_fit(cfg, identity, base, "FC", "literal", SMOKE_SEED, episodes=400, knobs=kn)
    run_fit(cfg, identity, base, "FC-K1", "clr0.03", SMOKE_SEED, episodes=400, knobs={"code_lr": 0.03})
    a = read_json(fit_dir(base, "FC", "literal", SMOKE_SEED) / "result.json")
    b = read_json(fit_dir(base, "FC-K1", "clr0.03", SMOKE_SEED) / "result.json")
    ta, _ = load_arrays(fit_dir(base, "FC", "literal", SMOKE_SEED) / "trace.npz")
    tb, _ = load_arrays(fit_dir(base, "FC-K1", "clr0.03", SMOKE_SEED) / "trace.npz")
    same = {k: bool(np.array_equal(ta[k], tb[k], equal_nan=True)) for k in ("action", "probe_action", "probe_value", "ctx", "assign")}
    return dict(online_hash_equal=a["final_hashes"]["online"] == b["final_hashes"]["online"],
                ema_hash_equal=a["final_hashes"]["ema"] == b["final_hashes"]["ema"], traces_equal=same, fc_spawns=a["spawns"],
                max_literal_margin=float(np.nanmax(np.abs(ta["margin"]))), episodes=400,
                passed=bool(a["final_hashes"]["online"] == b["final_hashes"]["online"] and all(same.values()) and sum(a["spawns"]) == 0))


def am_summary_check(config, output, params):
    """(a) every reservoir entry's stored history = the summaries of its preceding k episodes recomputed from the stream and
    the fit's own actions; (b) a training step does not read the live window (perturbing it leaves training bit-identical)."""
    tag = arm_tag("AM", config["knobs"]["AM"], config)
    d = fit_dir(output, "AM", tag, SMOKE_SEED)
    ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
    tr, _ = load_arrays(d / "trace.npz")
    st = Stream(params, SMOKE_SEED, config["episodes"])
    lrn, _ = build_learner(config, "AM", config["knobs"]["AM"], SMOKE_SEED)
    lrn.load(ck["learner"])
    k = lrn.k
    data = lrn.replay.data
    ok, checked = True, 0
    for slot in range(lrn.replay.size):
        e = int(data["episode"][slot])
        for j in range(k):
            src = e - k + j
            if src < 0:
                ok &= bool(data["hist_i"][slot, j, 0] == -1)
                continue
            a = int(tr["action"][src])
            f, i = D.summarize_episode(st.decision(src), a, st.outcome(src, a))
            ok &= bool(np.array_equal(f, data["hist_f"][slot, j]) and np.array_equal(i, data["hist_i"][slot, j]))
        checked += 1
    import copy as _copy
    s0 = _copy.deepcopy(lrn.state())
    lrn._train_step()
    h1 = lrn.hashes()
    lrn.load(_copy.deepcopy(s0))
    lrn.window_f = np.full_like(lrn.window_f, 7.0)
    lrn.window_i = np.full_like(lrn.window_i, 3)
    lrn._train_step()
    h2 = lrn.hashes()
    return dict(entries_checked=checked, stored_histories_match_stream=bool(ok), training_ignores_live_window=bool(h1 == h2),
                passed=bool(ok and checked > 0 and h1 == h2))


def default_check(config, output):
    tag = arm_tag("FC", config["knobs"]["FC"], config)
    d = fit_dir(output, "FC", tag, SMOKE_SEED)
    rec = read_json(d / "result.json")
    steps = rec["optimizer_steps"]
    frac = [c / steps for c in rec["default_flag_counts"]]
    se = math.sqrt(0.2 * 0.8 / steps)
    ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
    lrn, _ = build_learner(config, "FC", config["knobs"]["FC"], SMOKE_SEED)
    lrn.load(ck["learner"])
    data = lrn.replay.data
    rows = np.arange(lrn.replay.size)
    out = {}
    rng = np.random.default_rng(1)
    with torch.no_grad():
        for s, grp, key in ((0, lrn.supply, "y_supply"), (1, lrn.probe, "y_probe")):
            y = torch.from_numpy(data[key][rows])
            assigned = lrn.ctx_rows(rows, np.zeros(3, bool))[:, s * D.CTX:(s + 1) * D.CTX]
            default = lrn._default(lrn.codes, s).detach()[None].expand(len(rows), -1)
            rand = torch.from_numpy(rng.normal(0, 1, (len(rows), D.CTX)).astype(np.float32))
            out[f"slot{s}"] = dict(nll_assigned=float(D.gauss_nll_rows(grp, assigned, y)), nll_default=float(D.gauss_nll_rows(grp, default, y)),
                                   nll_random_code=float(D.gauss_nll_rows(grp, rand, y)))
    ok_frac = all(abs(f - 0.2) <= 4 * se for f in frac)
    ok_nll = all(v["nll_default"] < v["nll_random_code"] for v in out.values())
    return dict(default_minibatch_fraction=frac, binomial_se=se, reservoir_nll=out,
                rule="default used on 20% +- 4 SE of minibatches per slot, and the core's reservoir NLL under the default is below "
                     "its NLL under a random (untrained) code",
                passed=bool(ok_frac and ok_nll))


def informative_scoring_check(config, output):
    tag = arm_tag("FC", config["knobs"]["FC"], config)
    tr, _ = load_arrays(fit_dir(output, "FC", tag, SMOKE_SEED) / "trace.npz")
    inf = np.isin(tr["action"].astype(np.int64), D.INFORMATIVE_ACTIONS)
    f_scored = np.isfinite(tr["margin"][:, 2])
    s_scored = np.isfinite(tr["margin"][:, :2]).all(1)
    f_filter = np.isfinite(tr["post_max"][:, 2])
    changes_ok = bool(np.all(~tr["changed"][:, 2] | inf))
    return dict(informative_episodes=int(inf.sum()), failure_slot_scored_iff_informative=bool(np.array_equal(inf, f_scored)),
                failure_filter_advanced_iff_informative=bool(np.array_equal(inf, f_filter)), supply_probe_scored_every_episode=bool(s_scored.all()),
                failure_code_changes_only_on_informative=changes_ok,
                passed=bool(np.array_equal(inf, f_scored) and np.array_equal(inf, f_filter) and s_scored.all() and changes_ok))


def switch_rule_check(config, output):
    tag = arm_tag("FC", config["knobs"]["FC"], config)
    tr, _ = load_arrays(fit_dir(output, "FC", tag, SMOKE_SEED) / "trace.npz")
    ch = tr["changed"] & ~tr["spawned"]
    ok = bool(np.all(tr["post_max"][ch] >= 0.95))
    return dict(filter_changes=int(ch.sum()), spawns=int(tr["spawned"].sum()), all_filter_changes_at_posterior_ge_95=ok, passed=ok)


def w_check(config, output, st):
    rec = read_json(fit_dir(output, "W", "base", SMOKE_SEED) / "result.json")
    starts = [int(s) for s in st.starts[1:]]
    return dict(resets_at=rec["resets_at"], true_switches=starts, passed=bool(rec["resets_at"] == starts and rec["resets"] == len(starts)))


def oc_label_check(config, output, st):
    tag = arm_tag("OC", {"code_lr": config["oc_code_lr"]}, config)
    tr, _ = load_arrays(fit_dir(output, "OC", tag, SMOKE_SEED) / "trace.npz")
    lagged = bool(np.all(tr["active"][0] == -1) and np.array_equal(tr["active"][1:], st.levels[:-1].astype(np.int16)))
    assign = bool(np.array_equal(tr["assign"], st.levels.astype(np.int16)))
    return dict(oc_decision_index_is_previous_episode_level=lagged, oc_replay_assignment_is_own_level=assign,
                note="the harness hands a label only to OC and only after the episode's outcome; every other arm rejects labels "
                     "(legal_observations)", passed=bool(lagged and assign))


def resume_check(config, identity, output):
    base = fresh(output / "checks" / "resume")
    out = {}
    for arm in ("FC", "AM", "W"):
        t = [x for x in fit_tasks(config) if x[0] == arm][0]
        try:
            run_fit(config, identity, base, *t, interrupt_after=config["checkpoint_every"] + 100)
        except D1.Interrupted:
            pass
        side = read_json(fit_dir(base, *t) / "checkpoint.sha256.json")
        run_fit(config, identity, base, *t)
        rec = read_json(fit_dir(base, *t) / "result.json")
        rows = D1.compare_fits(fit_dir(output, *t), fit_dir(base, *t))
        out[arm] = dict(checkpoint_at_interrupt=side["episodes_done"], resumed=rec["resumed"], comparison=rows,
                        passed=bool(rec["resumed"] and side["episodes_done"] == config["checkpoint_every"] and all(rows.values())))
    out["passed"] = all(v["passed"] for v in out.values())
    return out


def lookup_check(config, rid, params):
    """O table entries recomputed directly; I-myopic re-run from the stored likelihoods; L decisions (stream and probe)
    recomputed by replaying L's filter on its own stored actions; the online-excess lookup against an explicit loop."""
    n = config["episodes"]
    data = load_ref_stream(config, SMOKE_SEED, n, rid)
    ag, _ = load_arrays(ref_dir(SMOKE_SEED, n) / "agents.npz")
    panels = load_panel_tables(config, rid)
    contexts, starts, lengths, run_ctx = schedule_arrays(SMOKE_SEED, n)
    ep = sample_episodes(n, seed_for("commons_d2_stream", SMOKE_SEED))
    m = config["oracle"]["draws"]
    rng = np.random.default_rng(3)
    o_ok = []
    for e in rng.choice(n, 3, replace=False):
        z = R.draws(seed_for(SMOKE_SEED, "commons_d2_O_select", int(e)), m)
        v, _ = R.episode_values(params, ep.reserves[e], int(ep.cue[e, 0]), list(range(N_CONTEXTS)), z)
        o_ok.append(bool(np.array_equal(v, data["v1"][e])))
        z = R.draws(seed_for(SMOKE_SEED, "commons_d2_O_score", int(e)), m)
        v, _ = R.episode_values(params, ep.reserves[e], int(ep.cue[e, 0]), [int(contexts[e])], z)
        o_ok.append(bool(np.array_equal(v[0], data["v2"][e])))
    c = 5
    z = R.draws(seed_for("commons_d2_panel_O_score", c, 2), m)
    v, _ = R.episode_values(params, panels[c]["reserves"][2], int(panels[c]["cue"][2, 0]), list(range(N_CONTEXTS)), z)
    o_ok.append(bool(np.array_equal(v, panels[c]["score"][2])))
    model = C.schedule_model(schedule_spec(n))
    filt = R.RunLengthFilter(model)
    acts = np.empty(n, np.int64)
    for t in range(n):
        w, pred = filt.weights()
        acts[t] = R.myopic_action(w, data["v1"][t])
        filt.update(data["sensor_ll"][t] + data["pattern_ll"][t, acts[t]], pred)
    i_ok = bool(np.array_equal(acts, ag["a_I"]))
    pool = D0B.lookahead_pool(params)
    la = D0B.Lookahead(params, pool, seed_for("commons_d2_L", SMOKE_SEED))
    lp = D0B.Lookahead(params, pool, seed_for("commons_d2_L_probe", SMOKE_SEED))
    events = ag["events"]
    check_t = set(int(x) for x in rng.choice(n, 6, replace=False)) | {int(events[3, 0])}
    filt = R.RunLengthFilter(model)
    l_ok = []
    for t in range(max(check_t) + 1):
        w, pred = filt.weights()
        if t in check_t:
            l_ok.append(bool(la.choose(t, w, filt, data["v1"][t], ep.reserves[t], ep.cue[t])[0] == ag["a_L"][t]))
            for i in np.flatnonzero(events[:, 0] == t):
                ptab = panels[int(events[i, 3])]
                l_ok.append(bool(all(lp.choose(t, w, filt, ptab["select"][s], ptab["reserves"][s], ptab["cue"][s])[0] == ag["probe_L"][i, s]
                                     for s in range(4))))
        filt.update(data["sensor_ll"][t] + data["pattern_ll"][t, ag["a_L"][t]], pred)
    rows = np.arange(n)
    acts_r = rng.integers(0, 13, n)
    vec = data["v2"][rows, ag["a_L"]] - data["v2"][rows, acts_r]
    loop = np.array([data["v2"][t, ag["a_L"][t]] - data["v2"][t, acts_r[t]] for t in range(n)])
    l_reg = float((data["v2"][rows, ag["a_O"]] - data["v2"][rows, ag["a_L"]]).mean())
    i_reg = float((data["v2"][rows, ag["a_O"]] - data["v2"][rows, ag["a_I"]]).mean())
    return dict(o_entries_recomputed_equal=o_ok, i_myopic_rerun_equal=i_ok, l_decisions_recomputed_equal=l_ok,
                excess_vectorised_equals_loop=bool(np.array_equal(vec, loop)), L_mean_regret_vs_O=l_reg, I_mean_regret_vs_O=i_reg,
                passed=bool(all(o_ok) and i_ok and all(l_ok) and np.array_equal(vec, loop)))


def crn_check(config, output, st):
    fps, dec = set(), set()
    for t in fit_tasks(config):
        rec = read_json(fit_dir(output, *t) / "result.json")
        fps.add(rec["stream_fingerprint"])
        dec.add(rec["decisions_chain"])
    return dict(stream_fingerprints=len(fps), decision_chains=len(dec), passed=bool(len(fps) == 1 and len(dec) == 1))


def cmd_check(args):
    config = load_config(args.config)
    require(config["phase"] == "smoke", "check runs the smoke configuration")
    output = Path(args.output)
    fresh(output)
    D1.worker_init(1)
    t0 = time.perf_counter()
    checks = dict(label="ENGINEERING CHECKS, D2 smoke (seed 22199, 2,000 episodes). Not science.", utc=now())
    params = world()
    checks["world"] = dict(key=E.WORLD_KEY, passed=True)
    checks["imports"] = import_check()
    checks["informative_actions"] = informative_check(params)
    checks["legal_observations"] = legal_check(config, params)
    checks["filter_matches_bruteforce"] = filter_check()
    checks["byte_parity"] = byte_check(config)
    print("unit", {k: v.get("passed") for k, v in checks.items() if isinstance(v, dict)}, flush=True)
    identity = lock(config, args.config, output)["identity"]
    rid = ref_identity(config)
    checks["fc_literal_rule_equals_fc_k1"] = literal_equivalence_check(config, identity, output)
    print("literal", checks["fc_literal_rule_equals_fc_k1"]["passed"], flush=True)
    a = argparse.Namespace(config=args.config, output=str(output), report=str(output / "analysis"))
    cmd_run(a)
    second = cmd_run(a)
    checks["rerun_only_verifies"] = dict(passed=all(s["status"] == "verified_existing" for s in second["statuses"]))
    st = Stream(params, SMOKE_SEED, config["episodes"])
    checks["w_resets_at_true_switches"] = w_check(config, output, st)
    checks["oc_label_lagged_only_oc"] = oc_label_check(config, output, st)
    checks["am_trains_on_stored_summaries"] = am_summary_check(config, output, params)
    checks["default_code_in_distribution"] = default_check(config, output)
    checks["informative_episode_scoring"] = informative_scoring_check(config, output)
    checks["fc_switch_rule"] = switch_rule_check(config, output)
    checks["common_random_numbers"] = crn_check(config, output, st)
    checks["bounded_replay"] = dict(passed=all(read_json(fit_dir(output, *t) / "result.json")["replay"]["max_size"]
                                               <= read_json(fit_dir(output, *t) / "result.json")["replay"]["capacity"] for t in fit_tasks(config)))
    checks["byte_parity_in_run"] = dict(ratios={f"{t[0]}": read_json(fit_dir(output, *t) / "result.json")["memory"]["budget_bytes"]
                                                / fc_budget_bytes(config) for t in fit_tasks(config) if t[0] in AM_TYPES},
                                        passed=all(abs(read_json(fit_dir(output, *t) / "result.json")["memory"]["budget_bytes"] / fc_budget_bytes(config) - 1) <= 0.05
                                                   for t in fit_tasks(config) if t[0] in AM_TYPES))
    checks["l_and_o_lookups"] = lookup_check(config, rid, params)
    checks["determinism_and_resumption"] = resume_check(config, identity, output)
    pc = dict(config, parity=dict(config["parity"]))
    for arm, seed, ctx in parity_tasks(pc):
        run_parity_fit(pc, identity, output, arm, seed, ctx)
    checks["parity_pipeline"] = dict(result=parity_gate(pc, output), passed=True,
                                     note=f"pipeline only ({config['parity']['episodes']} episodes; the gate's verdict is meaningless here)")
    print("fit", {k: v.get("passed") for k, v in checks.items() if isinstance(v, dict)}, flush=True)
    rep = cmd_replicate(a)
    checks["replicate_pass"] = dict(fits_compared=rep["fits_compared"], mismatches=rep["mismatches"], passed=rep["passed"])
    victim = fit_dir(output, *fit_tasks(config)[0]) / "trace.npz"
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
    checks["analysis_pipeline"] = dict(verdict_smoke_only=res["verdict"], passed=True)
    checks["all_passed"] = bool(all(v["passed"] for v in checks.values() if isinstance(v, dict) and "passed" in v))
    checks["seconds"] = time.perf_counter() - t0
    write_json(output / "checks" / "checks.json", clean(checks))
    print("ALL PASS" if checks["all_passed"] else "FAILURES: " + ", ".join(k for k, v in checks.items()
                                                                          if isinstance(v, dict) and v.get("passed") is False), flush=True)
    if not checks["all_passed"]:
        raise SystemExit(1)


DEFAULTS = {"check": ("configs/commons_d2_smoke.json", "runs/commons_d2_smoke", "runs/commons_d2_smoke/analysis"),
            "dev": ("configs/commons_d2_dev.json", "runs/commons_d2_dev", "reports/commons_d2/dev"),
            "dev-report": ("configs/commons_d2_dev.json", "runs/commons_d2_dev", "reports/commons_d2/dev"),
            "lock": ("configs/commons_d2.json", "runs/commons_d2", "reports/commons_d2"),
            "run": ("configs/commons_d2.json", "runs/commons_d2", "reports/commons_d2"),
            "replicate": ("configs/commons_d2.json", "runs/commons_d2", "reports/commons_d2"),
            "analyze": ("configs/commons_d2.json", "runs/commons_d2", "reports/commons_d2")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(DEFAULTS))
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--report")
    parser.add_argument("--continue-after-gate-failure", action="store_true",
                        help="dev only: continue to the knob grids even if a gate failed (diagnostic; the failure stays recorded)")
    args = parser.parse_args()
    d = DEFAULTS[args.command]
    args.config, args.output, args.report = (str(ROOT / (v or dflt)) for v, dflt in
                                             ((args.config, d[0]), (args.output, d[1]), (args.report, d[2])))
    {"check": cmd_check, "dev": cmd_dev, "dev-report": cmd_dev_report, "lock": cmd_lock, "run": cmd_run, "replicate": cmd_replicate,
     "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
