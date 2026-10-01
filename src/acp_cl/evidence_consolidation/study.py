"""Run source-locked continuous evidence-consolidation experiments with recovery."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch

from acp_cl.acquisition.study import Evaluator, load, validate_config as validate_base
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.persistence.study import digest
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import (affinity, atomic_checkpoint, episode, new_learner,
                                       source_manifest as previous_sources, write_json)
from .mechanism import Consolidation, POLICIES
from .schedule import experiences, schedule


EXTRAS = {"cpu_affinity", "evidence_window", "evidence_margin", "evidence_positive",
          "burst_period", "burst_length", "burst_noise", "blocks"}


def validate_config(config):
    validate_base({k: v for k, v in config.items() if k not in EXTRAS})
    if not EXTRAS <= config.keys():
        raise ValueError("missing evidence settings")
    for name in EXTRAS-{"evidence_margin", "burst_noise", "cpu_affinity"}:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("positive integer evidence/schedule settings required")
    if config["evidence_positive"] > config["evidence_window"]:
        raise ValueError("positive evidence count exceeds window")
    if not 0 <= config["evidence_margin"] < 1 or not 0 <= config["burst_noise"] <= 1:
        raise ValueError("invalid gate margin or corruption probability")
    if not 0 < config["burst_length"] <= config["burst_period"] or not 3 <= config["blocks"] <= 18:
        raise ValueError("invalid schedule length")
    if type(config["cpu_affinity"]) is not int or config["cpu_affinity"] < 0:
        raise ValueError("invalid CPU affinity")


def source_manifest():
    sources = previous_sources()
    base = Path(__file__).resolve().parent
    for path in sorted(base.glob("*.py")):
        sources[path.relative_to(base.parent.parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def clean_score(predictions, experience):
    rows = np.arange(len(experience))
    return {p: float(((v[rows, experience.actions]-experience.survival)**2).mean())
            for p, v in predictions.items()}


def evaluate(bundle, evaluator, block):
    return {p: evaluator.evaluate(learner, block.law, bundle.draft.history, block.cue, block.branch)
            for p, learner in bundle.predictors().items()}


def valid_panel(bundle, evaluator, block):
    return {p: evaluator.valid_panel(learner, block.law, block.branch)
            for p, learner in bundle.predictors().items()}


def knowledge_panel(bundle, evaluator, seed, index):
    """Evaluation only: all distinct laws already encountered, correct fresh support."""
    laws = [Law(0), Law(1)]
    for block in schedule(seed)[:index+1]:
        if block.law not in laws:
            laws.append(block.law)
    return [dict(law=asdict(law), policies={p: float(np.mean([
        evaluator.evaluate(learner, law, evaluator.support(law, r))["brier"]
        for r in range(bundle.settings["support_replicates"])]))
        for p, learner in bundle.predictors().items()}) for law in laws]


def prefix(config, identity, seed, model, directory, device):
    path = directory/"prefix.pt"
    if path.exists():
        saved = load(path, identity, device)
        bundle, record = saved["bundle"], saved["record"]
    else:
        bundle = Consolidation(new_learner(model, seed, config, device), config)
        record = dict(identity=identity, seed=seed, model=model, blocks=[], decisions=[])
    world = AcquisitionWorld()
    for k in range(len(record["blocks"]), config["prefix_blocks"]):
        law = Law((seed+k) % 2)
        hashes = []
        for i in range(config["prefix_size"]//config["batch_size"]):
            data = batch(world, law, seed, f"prefix_{k}", i, config["batch_size"])
            predictions, proposed = bundle.prepare(data.observations)
            decision = bundle.observe(data, predictions, proposed)
            if decision:
                record["decisions"].append(decision)
            hashes.append(data.fingerprint())
        record["blocks"].append(dict(law=asdict(law), batch_sha256=hashes))
        record["end_state"] = bundle.state()
        record["end_memory"] = memory_state(bundle.draft.memory)
        atomic_checkpoint(path, dict(identity=identity, bundle=bundle, record=record))
    write_json(directory/"prefix.json", record)
    return bundle


def run_block(config, identity, bundle, seed, model, block, directory):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, result_file = directory/"checkpoint.pt", directory/"result.json"
    if result_file.exists():
        saved = load(checkpoint, identity, str(bundle.draft.device))
        record = json.loads(result_file.read_text(encoding="utf-8"))
        if digest(saved["record"]) != digest(record):
            raise ValueError("completed block differs from checkpoint")
        return saved["bundle"], record
    evaluator = Evaluator(seed, config)
    if checkpoint.exists():
        saved = load(checkpoint, identity, str(bundle.draft.device))
        bundle, record = saved["bundle"], saved["record"]
    else:
        record = dict(identity=identity, seed=seed, model=model, block=asdict(block),
                      batches_done=0, batch_sha256=[], clean_sha256=[], prequential=[], decisions=[],
                      start_state=bundle.state(), start_memory=memory_state(bundle.draft.memory),
                      valid_before=valid_panel(bundle, evaluator, block),
                      curve=[dict(arrivals=0, policies=evaluate(bundle, evaluator, block))],
                      elapsed_seconds=0.)
        atomic_checkpoint(directory/"before.pt", dict(identity=identity, bundle=bundle, record=record))
    started = time.perf_counter()
    for i in range(record["batches_done"], config["episode_size"]//config["batch_size"]):
        reported, clean = experiences(block, seed, i, config)
        predictions, proposed = bundle.prepare(reported.observations)
        record["prequential"].append(dict(arrivals=(i+1)*config["batch_size"],
            truth_brier=clean_score(predictions, clean), reported_brier=clean_score(predictions, reported)))
        decision = bundle.observe(reported, predictions, proposed)
        if decision:
            record["decisions"].append(decision)
        record["batch_sha256"].append(reported.fingerprint())
        record["clean_sha256"].append(clean.fingerprint())
        record["batches_done"] = i+1
        arrivals = (i+1)*config["batch_size"]
        if arrivals % config["probe_every"] == 0:
            record["curve"].append(dict(arrivals=arrivals, policies=evaluate(bundle, evaluator, block)))
            record["elapsed_seconds"] += time.perf_counter()-started
            atomic_checkpoint(checkpoint, dict(identity=identity, bundle=bundle, record=record))
            started = time.perf_counter()
    record["valid_after"] = valid_panel(bundle, evaluator, block)
    record["knowledge_after"] = knowledge_panel(bundle, evaluator, seed, block.index)
    record["end_state"] = bundle.state()
    record["end_memory"] = memory_state(bundle.draft.memory)
    record["elapsed_seconds"] += time.perf_counter()-started
    atomic_checkpoint(checkpoint, dict(identity=identity, bundle=bundle, record=record))
    write_json(result_file, record)
    return bundle, record


def counts(config):
    trajectories = len(config["seeds"])*len(config["models"])
    return dict(trajectories=trajectories, blocks=trajectories*config["blocks"],
                fresh_episodes=3*trajectories, prefix_fits=trajectories,
                policy_evaluations=trajectories*config["blocks"]*len(POLICIES))


def run_job(config, identity, seed, model, output, device):
    actual = affinity(config["cpu_affinity"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/f"{model}_{seed}"
    directory.mkdir(exist_ok=True)
    bundle = prefix(config, identity, seed, model, directory, device)
    for block in schedule(seed)[:config["blocks"]]:
        if block.index < 3:
            fresh = new_learner(model, seed, config, device)
            fresh.history = copy.deepcopy(bundle.draft.history)
            episode(config, identity, fresh, seed, block.index+1, "fresh", block.law,
                    directory/f"fresh_{block.index+1}", block.cue)
        bundle, record = run_block(config, identity, bundle, seed, model, block,
                                   directory/f"block_{block.index:02}")
        scores = {p: round(float(np.mean([r["truth_brier"][p] for r in record["prequential"]])), 5)
                  for p in POLICIES}
        print(json.dumps(dict(seed=seed, model=model, block=block.index+1, label=block.label,
                              truth=scores, adoptions=bundle.adoptions)), flush=True)
    return dict(seed=seed, model=model, affinity=actual, final_state=bundle.state())


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
            raise ValueError("protocol changed after lock")
    else:
        if any(output.iterdir()):
            raise ValueError("new run output must be empty")
        now = datetime.now(timezone.utc).isoformat()
        write_json(path, dict(identity=identity, config=config, runtime=runtime,
                             source_files=sources, created_utc=now))
        root = Path(__file__).resolve().parents[2]
        with zipfile.ZipFile(output/"training_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(root/name, name)
        if protocol:
            content = Path(protocol).read_bytes()
            (output/"protocol_at_lock.md").write_bytes(content)
            write_json(output/"protocol_lock.json", dict(**identity, locked_utc=now,
                protocol_sha256=hashlib.sha256(content).hexdigest()))
    if lock_only:
        print(json.dumps(dict(locked=str(output), identity=identity, **counts(config))), flush=True)
        return
    jobs = [(config, identity, seed, model, output, device)
            for seed in config["seeds"] for model in config["models"]]
    if config["workers"] == 1:
        finished = [run_job(*job) for job in jobs]
    else:
        finished = []
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [pool.submit(run_job, *job) for job in jobs]
            try:
                for future in as_completed(futures):
                    finished.append(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                print("Failed; queued jobs cancelled, running jobs finishing safely.", flush=True)
                raise
    if digest(source_manifest()) != identity["source_sha256"] or any(j["affinity"] != actual for j in finished):
        raise ValueError("source or worker runtime changed")
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
