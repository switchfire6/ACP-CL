"""Locked optimizer-by-replay reset diagnostic using unchanged acquisition tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import torch

from acp_cl.acquisition.study import (episode, preceding_history, run_prefix,
    source_manifest as acquisition_sources, validate_config)
from acp_cl.acquisition.world import final_law as acquisition_final_law, order, stage_law
from acp_cl.conditional.learner import PacketMemory
from acp_cl.contextual.learner import ContextLearner
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest, write_json


FACTORIAL = ("continue", "optimizer_reset", "replay_reset", "state_reset")
ARMS = (*FACTORIAL, "fresh")
BRANCHES = ("return", "revision", "clean", "noise")


def source_manifest():
    sources = acquisition_sources()
    base = Path(__file__).resolve().parent
    for path in sorted(base.glob("*.py")):
        sources[path.relative_to(base.parent.parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def final_law(seed, branch, noise):
    if branch == "clean":
        return stage_law(seed, 3)
    return acquisition_final_law(seed, branch, noise)


def fork(learner, arm, seed, config):
    """Change exactly the declared factors; preserve learned weights and history."""
    if arm not in ARMS:
        raise ValueError("unknown factorial arm")
    if arm == "fresh":
        result = ContextLearner(learner.method, seed, config, str(learner.device))
        result.history = copy.deepcopy(learner.history)
        return result
    result = copy.deepcopy(learner)
    if arm in ("optimizer_reset", "state_reset"):
        result.optimizer = torch.optim.Adam(result.model.parameters(), lr=config["lr"])
    if arm in ("replay_reset", "state_reset"):
        result.memory = PacketMemory(config["memory_packets"], seed)
    return result


def run_job(config, identity, seed, model, output, device):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/f"{model}_{seed}"
    directory.mkdir(exist_ok=True)
    learner = None if config["kind"] == "development" else run_prefix(config, identity, seed, model, directory, device)
    for stage in range(1, 4):
        law, cue = stage_law(seed, stage), order(seed)[stage-1]
        if config["kind"] == "development":
            fresh = ContextLearner(model, seed, config, device)
            fresh.history = preceding_history(config, seed, stage)
            episode(config, identity, fresh, seed, stage, "fresh", law, directory/f"stage_{stage}_fresh", cue)
        else:
            before_hash = state_hash(learner.model.state_dict())
            continuing = None
            for arm in ARMS:
                end, _ = episode(config, identity, fork(learner, arm, seed, config), seed, stage, arm,
                                  law, directory/f"stage_{stage}_{arm}", cue)
                if arm == "continue":
                    continuing = end
            if state_hash(learner.model.state_dict()) != before_hash:
                raise AssertionError("a factorial arm changed its parent")
            learner = continuing
    if config["kind"] == "diagnostic":
        for branch in BRANCHES:
            law = final_law(seed, branch, config["feedback_noise"])
            for arm in FACTORIAL:
                episode(config, identity, fork(learner, arm, seed, config), seed, 4, arm, law,
                        directory/f"final_{branch}_{arm}", 0 if branch == "revision" else None, branch)
    return dict(seed=seed, model=model)


def run_suite(config, output, device="cpu", resume=False, protocol=None):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    runtime = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, device=device, threads=config["threads"], workers=config["workers"], deterministic=True)
    identity = dict(config_sha256=digest(config), source_sha256=digest(sources), runtime_sha256=digest(runtime))
    path = output/"manifest.json"
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("resume requires identical config/source/runtime")
    else:
        if any(output.iterdir()):
            raise ValueError("new output must be empty")
        now = datetime.now(timezone.utc).isoformat()
        write_json(path, dict(identity=identity, config=config, runtime=runtime, source_files=sources, created_utc=now))
        root = Path(__file__).resolve().parent.parent.parent
        with zipfile.ZipFile(output/"training_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in sources:
                z.write(root/name, name)
        if protocol:
            content = Path(protocol).read_bytes()
            (output/"protocol_at_lock.md").write_bytes(content)
            write_json(output/"protocol_lock.json", dict(**identity, locked_utc=now,
                protocol_sha256=hashlib.sha256(content).hexdigest()))
    jobs = [(config, identity, s, m, output, device) for s in config["seeds"] for m in config["models"]]
    if config["workers"] == 1:
        for index, job in enumerate(jobs, 1):
            print(f"{index}/{len(jobs)} trajectories: {run_job(*job)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [pool.submit(run_job, *job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                print(f"{index}/{len(jobs)} trajectories: {future.result()}", flush=True)
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("scientific source changed during run")
    count = len(jobs)*(3 if config["kind"] == "development" else 3*len(ARMS)+len(BRANCHES)*len(FACTORIAL))
    write_json(output/"completion.json", dict(identity=identity, trajectories=len(jobs), episodes=count,
        prefix_fits=0 if config["kind"] == "development" else len(jobs),
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--protocol")
    args = parser.parse_args()
    run_suite(json.loads(Path(args.config).read_text(encoding="utf-8")), args.output,
              args.device, args.resume, args.protocol)
