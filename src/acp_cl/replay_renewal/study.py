"""Prospective replay decomposition and boundary-independent memory policies."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch

from acp_cl.acquisition.study import (Evaluator, load, marginal, marginal_update,
    preceding_history, validate_config as validate_base)
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch, order, stage_law
from acp_cl.contextual.learner import ContextLearner
from acp_cl.persistence.study import digest
from acp_cl.training_state.study import BRANCHES, final_law, source_manifest as prior_sources
from .memory import POLICIES, RESET_ARMS, make_memory, memory_state, reset_memory


EXTRA_FIELDS = {"experiment", "recent_packets", "cpu_affinity"}


def validate_config(config):
    validate_base({k: v for k, v in config.items() if k not in EXTRA_FIELDS})
    if not EXTRA_FIELDS <= config.keys() or config["kind"] != "diagnostic":
        raise ValueError("replay study requires all extension fields and diagnostic kind")
    if config["experiment"] not in ("decomposition", "policy"):
        raise ValueError("unknown experiment")
    if type(config["recent_packets"]) is not int or not 0 < config["recent_packets"] < config["memory_packets"]:
        raise ValueError("split requires nonempty recent and historical stores")
    if type(config["cpu_affinity"]) is not int or config["cpu_affinity"] < 0:
        raise ValueError("invalid affinity mask")


def source_manifest():
    sources = prior_sources()
    base = Path(__file__).resolve().parent
    for path in sorted(base.glob("*.py")):
        sources[path.relative_to(base.parent.parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def affinity(mask=0):
    """Apply the requested mask and report the actual process affinity."""
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        process = kernel.GetCurrentProcess()
        if mask and not kernel.SetProcessAffinityMask(process, mask):
            raise ctypes.WinError(ctypes.get_last_error())
        actual, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(actual), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        return actual.value
    if hasattr(os, "sched_getaffinity"):
        if mask:
            os.sched_setaffinity(0, {i for i in range(mask.bit_length()) if mask & (1 << i)})
        return sum(1 << i for i in os.sched_getaffinity(0))
    if mask:
        raise ValueError("affinity is unsupported on this platform")
    return None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def new_learner(model, seed, config, device="cpu", policy="uniform"):
    learner = ContextLearner(model, seed, config, device)
    learner.memory = make_memory(policy, config, seed)
    return learner


def fork(learner, arm, seed, config):
    if arm == "fresh":
        result = new_learner(learner.method, seed, config, str(learner.device))
        result.history = copy.deepcopy(learner.history)
        return result
    result = copy.deepcopy(learner)
    result.memory = reset_memory(learner.memory, arm, seed)
    return result


def episode(config, identity, learner, seed, stage, arm, law, directory, cue=None, branch="novel"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, result_path = directory/"checkpoint.pt", directory/"result.json"
    device = str(learner.device)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        saved = load(checkpoint, identity, device)
        if result["identity"] != identity or digest(saved["record"]) != digest(result):
            raise ValueError("completed result/checkpoint mismatch")
        return saved["learner"], result
    evaluator, world = Evaluator(seed, config), AcquisitionWorld()
    channel = f"stage_{stage}" if branch == "novel" else "final"
    if checkpoint.exists():
        saved = load(checkpoint, identity, device)
        learner, record = saved["learner"], saved["record"]
    else:
        record = dict(identity=identity, seed=seed, model=learner.method, stage=stage, arm=arm,
            branch=branch, law=asdict(law), cue=cue, channel=channel, batches_done=0, batch_sha256=[],
            before=evaluator.probe(learner, law, cue, branch),
            valid_before=evaluator.valid_panel(learner, law, branch), start_diagnostics=learner.diagnostics(),
            start_memory=memory_state(learner.memory),
            curve=[dict(arrivals=0, metrics=evaluator.evaluate(learner, law, learner.history, cue, branch))],
            marginal_count=[0]*5, marginal_success=[[0]*3 for _ in range(5)], elapsed_seconds=0.)
        atomic_checkpoint(directory/"before.pt", dict(identity=identity, learner=learner, record=record))
    started = time.perf_counter()
    for index in range(record["batches_done"], config["episode_size"]//config["batch_size"]):
        data = batch(world, law, seed, channel, index, config["batch_size"])
        learner.train(data)
        marginal_update(record, data)
        record["batch_sha256"].append(data.fingerprint())
        record["batches_done"] = index+1
        arrivals = (index+1)*config["batch_size"]
        if arrivals % config["probe_every"] == 0:
            record["curve"].append(dict(arrivals=arrivals,
                metrics=evaluator.evaluate(learner, law, learner.history, cue, branch, marginal=marginal(record))))
            record["elapsed_seconds"] += time.perf_counter()-started
            atomic_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record))
            started = time.perf_counter()
    record["after"] = evaluator.probe(learner, law, cue, branch, marginal=marginal(record))
    record["valid_after"] = evaluator.valid_panel(learner, law, branch)
    record["diagnostics"] = learner.diagnostics()
    record["end_memory"] = memory_state(learner.memory)
    record["elapsed_seconds"] += time.perf_counter()-started
    # The final checkpoint also contains the completed record, so result publication
    # can be retried after an interruption without losing any training updates.
    atomic_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record))
    write_json(result_path, record)
    return learner, record


def run_prefix(config, identity, seed, model, directory, device, policy):
    checkpoint = directory/"prefix.pt"
    world, evaluator = AcquisitionWorld(), Evaluator(seed, config)
    if checkpoint.exists():
        saved = load(checkpoint, identity, device)
        learner, record = saved["learner"], saved["record"]
    else:
        learner = new_learner(model, seed, config, device, policy)
        record = dict(blocks=[], policy=policy)
    for block in range(len(record["blocks"]), config["prefix_blocks"]):
        law, hashes = Law((seed+block) % 2), []
        for index in range(config["prefix_size"]//config["batch_size"]):
            data = batch(world, law, seed, f"prefix_{block}", index, config["batch_size"])
            learner.train(data)
            hashes.append(data.fingerprint())
        record["blocks"].append(dict(law=asdict(law), batch_sha256=hashes,
            endpoint=evaluator.evaluate(learner, law, learner.history), diagnostics=learner.diagnostics()))
        record["memory"] = memory_state(learner.memory)
        atomic_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record))
    write_json(directory/"prefix.json", dict(identity=identity, seed=seed, model=model, **record))
    return learner


def job_policies(config):
    return ("base",) if config["experiment"] == "decomposition" else (*POLICIES, "fresh")


def counts(config):
    pairs = len(config["seeds"])*len(config["models"])
    return dict(trajectories=pairs*len(job_policies(config)),
        episodes=pairs*(7 if config["experiment"] == "decomposition" else 24),
        prefix_fits=pairs*(1 if config["experiment"] == "decomposition" else 3))


def run_job(config, identity, seed, model, policy, output, device):
    actual = affinity(config["cpu_affinity"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/f"{model}_{seed}_{policy}"
    directory.mkdir(exist_ok=True)
    if policy == "base":
        base = run_prefix(config, identity, seed, model, directory, device, "uniform")
        for arm in (*RESET_ARMS, "fresh"):
            episode(config, identity, fork(base, arm, seed, config), seed, 1, arm, stage_law(seed, 1),
                    directory/f"stage_1_{arm}", order(seed)[0])
    elif policy == "fresh":
        for stage in range(1, 4):
            fresh = new_learner(model, seed, config, device)
            fresh.history = preceding_history(config, seed, stage)
            episode(config, identity, fresh, seed, stage, "fresh", stage_law(seed, stage),
                    directory/f"stage_{stage}_fresh", order(seed)[stage-1])
    else:
        learner = run_prefix(config, identity, seed, model, directory, device, policy)
        for stage in range(1, 4):
            learner, _ = episode(config, identity, learner, seed, stage, policy, stage_law(seed, stage),
                directory/f"stage_{stage}_{policy}", order(seed)[stage-1])
        for branch in BRANCHES:
            episode(config, identity, copy.deepcopy(learner), seed, 4, policy,
                final_law(seed, branch, config["feedback_noise"]), directory/f"final_{branch}_{policy}",
                0 if branch == "revision" else None, branch)
    return dict(seed=seed, model=model, policy=policy, affinity=actual)


def run_suite(config, output, device="cpu", resume=False, protocol=None, lock_only=False):
    validate_config(config)
    actual = affinity(config["cpu_affinity"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    runtime = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, device=device, threads=config["threads"], workers=config["workers"],
        deterministic=True, affinity=actual)
    identity = dict(config_sha256=digest(config), source_sha256=digest(sources), runtime_sha256=digest(runtime))
    path = output/"manifest.json"
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("resume requires identical config/source/runtime")
        if protocol and (output/"protocol_at_lock.md").read_bytes() != Path(protocol).read_bytes():
            raise ValueError("protocol changed after locking")
    else:
        if any(output.iterdir()):
            raise ValueError("new output must be empty")
        now = datetime.now(timezone.utc).isoformat()
        write_json(path, dict(identity=identity, config=config, runtime=runtime, source_files=sources, created_utc=now))
        root = Path(__file__).resolve().parents[2]
        with zipfile.ZipFile(output/"training_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in sources:
                z.write(root/name, name)
        if protocol:
            content = Path(protocol).read_bytes()
            with (output/"protocol_at_lock.md").open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            write_json(output/"protocol_lock.json", dict(**identity, locked_utc=now,
                protocol_sha256=hashlib.sha256(content).hexdigest()))
    if lock_only:
        print(json.dumps(dict(locked=str(output), identity=identity, **counts(config))), flush=True)
        return
    jobs = [(config, identity, s, m, p, output, device)
            for s in config["seeds"] for m in config["models"] for p in job_policies(config)]
    finished = []
    if config["workers"] == 1:
        for index, job in enumerate(jobs, 1):
            finished.append(run_job(*job))
            print(f"{index}/{len(jobs)} trajectories: {finished[-1]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [pool.submit(run_job, *job) for job in jobs]
            try:
                for index, future in enumerate(as_completed(futures), 1):
                    finished.append(future.result())
                    print(f"{index}/{len(jobs)} trajectories: {finished[-1]}", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                print("A job failed; pending jobs cancelled, active jobs are finishing safely.", flush=True)
                raise
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("scientific source changed during run")
    if any(item["affinity"] != actual for item in finished):
        raise ValueError("worker affinity differs from locked runtime")
    write_json(output/"completion.json", dict(identity=identity, **counts(config), jobs=finished,
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--protocol")
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    run_suite(json.loads(Path(args.config).read_text(encoding="utf-8")), args.output,
              args.device, args.resume, args.protocol, args.lock_only)
