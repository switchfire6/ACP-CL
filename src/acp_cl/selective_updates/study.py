"""Locked first-introduction experiment for current emphasis and replay protection."""

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
import zipfile

import numpy as np
import torch

from acp_cl.acquisition.study import Evaluator, load, validate_config as validate_base
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch, order, stage_law
from acp_cl.evidence_consolidation.study import source_manifest as previous_sources
from acp_cl.persistence.study import digest
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import (affinity, atomic_checkpoint, episode,
                                      new_learner, write_json)
from .learner import ARMS, fork


ROOT = Path(__file__).resolve().parents[3]


def validate_config(config):
    validate_base({k: v for k, v in config.items() if k != "cpu_affinity"})
    if type(config.get("cpu_affinity")) is not int or config["cpu_affinity"] < 0:
        raise ValueError("nonnegative integer cpu_affinity required")
    if config["kind"] != "development":
        raise ValueError("this is a fixed first-introduction development study")


def source_manifest():
    sources = previous_sources()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        sources[path.relative_to(ROOT/"src").as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def counts(config):
    pairs = len(config["seeds"])*len(config["models"])
    return dict(trajectories=pairs, episodes=pairs*(len(ARMS)+1), prefix_fits=pairs)


def analysis_files(config_path=None, protocol=None):
    paths = list((ROOT/"scripts").glob("*.py"))
    paths += list((ROOT/"tests").glob("test_selective*.py"))
    if config_path is not None:
        paths.append(Path(config_path).resolve())
    if protocol is not None:
        paths.append(Path(protocol).resolve())
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def verify_analysis(directory):
    directory = Path(directory)
    lock = json.loads((directory/"analysis_lock.json").read_text(encoding="utf-8"))
    if digest(lock["config"]) != lock["config_sha256"]:
        raise ValueError("analysis configuration hash mismatch")
    with zipfile.ZipFile(directory/"analysis_at_lock.zip") as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(lock["files"]):
            raise ValueError("analysis archive file set or CRC mismatch")
        for name, expected in lock["files"].items():
            path = (ROOT/name).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("analysis path escaped repository")
            if (hashlib.sha256(path.read_bytes()).hexdigest() != expected
                    or hashlib.sha256(archive.read(name)).hexdigest() != expected):
                raise ValueError(f"analysis changed after locking: {name}")
    return lock


def run_prefix(config, identity, seed, model, directory, device):
    directory = Path(directory)
    checkpoint = directory/"prefix.pt"
    evaluator, world = Evaluator(seed, config), AcquisitionWorld()
    if checkpoint.exists():
        saved = load(checkpoint, identity, device)
        learner, record = saved["learner"], saved["record"]
        if (record["seed"], record["model"], record["identity"]) != (seed, model, identity):
            raise ValueError("prefix record identity mismatch")
    else:
        learner = new_learner(model, seed, config, device)
        record = dict(identity=identity, seed=seed, model=model, policy="uniform", blocks=[])
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
    write_json(directory/"prefix.json", record)
    return learner


def run_job(config, identity, seed, model, output, device):
    actual = affinity(config["cpu_affinity"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/f"{model}_{seed}"
    directory.mkdir(exist_ok=True)
    base = run_prefix(config, identity, seed, model, directory, device)
    parent_state = digest(dict(model=base.diagnostics()["final_hash"], memory=memory_state(base.memory)))
    for arm in (*ARMS, "fresh"):
        if arm == "fresh":
            learner = fork(new_learner(model, seed, config, device), "reference")
            learner.history = copy.deepcopy(base.history)
        else:
            learner = fork(base, arm)
        episode(config, identity, learner, seed, 1, arm, stage_law(seed, 1),
                directory/f"stage_1_{arm}", order(seed)[0])
        print(json.dumps(dict(seed=seed, model=model, arm=arm, completed=True)), flush=True)
    if digest(dict(model=base.diagnostics()["final_hash"], memory=memory_state(base.memory))) != parent_state:
        raise AssertionError("an intervention changed the common parent")
    return dict(seed=seed, model=model, affinity=actual)


def run_suite(config, output, device="cpu", resume=False, protocol=None, lock_only=False,
              config_path=None):
    validate_config(config)
    actual = affinity(config["cpu_affinity"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    runtime = dict(python=platform.python_version(), platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, device=device,
        threads=config["threads"], workers=config["workers"], deterministic=True, affinity=actual)
    identity = dict(config_sha256=digest(config), source_sha256=digest(sources), runtime_sha256=digest(runtime))
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or manifest["identity"] != identity:
            raise ValueError("resume requires identical config/source/runtime")
        if (manifest["config"] != config or manifest["runtime"] != runtime
                or manifest["source_files"] != sources):
            raise ValueError("manifest contents differ from the locked identity")
        with zipfile.ZipFile(output/"training_source.zip") as archive:
            if set(archive.namelist()) != set(sources):
                raise ValueError("source archive file set mismatch")
            if any(hashlib.sha256(archive.read(n)).hexdigest() != h for n, h in sources.items()):
                raise ValueError("source archive hash mismatch")
        analysis_lock = verify_analysis(output)
        if analysis_lock["config"] != config or analysis_lock["config_sha256"] != identity["config_sha256"]:
            raise ValueError("configuration changed after analysis lock")
        if (output/"protocol_at_lock.md").exists() != (output/"protocol_lock.json").exists():
            raise ValueError("incomplete protocol lock")
        if (output/"protocol_lock.json").exists():
            protocol_lock = json.loads((output/"protocol_lock.json").read_text(encoding="utf-8"))
            if (any(protocol_lock[k] != v for k, v in identity.items())
                    or hashlib.sha256((output/"protocol_at_lock.md").read_bytes()).hexdigest()
                    != protocol_lock["protocol_sha256"]):
                raise ValueError("protocol changed after locking")
        if protocol is not None and (output/"protocol_at_lock.md").read_bytes() != Path(protocol).read_bytes():
            raise ValueError("protocol changed after locking")
    else:
        if any(output.iterdir()):
            raise ValueError("new output directory must be empty")
        now = datetime.now(timezone.utc).isoformat()
        manifest = dict(identity=identity, config=config, runtime=runtime, source_files=sources,
                        created_utc=now, study_scope="local first-introduction development")
        with zipfile.ZipFile(output/"training_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(ROOT/"src"/name, name)
        locked_files = analysis_files(config_path, protocol)
        with zipfile.ZipFile(output/"analysis_at_lock.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in locked_files:
                archive.write(ROOT/name, name)
        write_json(output/"analysis_lock.json", dict(locked_utc=now, files=locked_files,
                                                    config=config, config_sha256=digest(config)))
        if protocol is not None:
            contents = Path(protocol).read_bytes()
            (output/"protocol_at_lock.md").write_bytes(contents)
            write_json(output/"protocol_lock.json", dict(**identity, locked_utc=now,
                protocol_sha256=hashlib.sha256(contents).hexdigest()))
        write_json(manifest_path, manifest)
    if lock_only:
        result = dict(locked=True, identity=identity, **counts(config), output=str(output))
        print(json.dumps(result, indent=2))
        return result
    if (output/"completion.json").exists():
        completion = json.loads((output/"completion.json").read_text(encoding="utf-8"))
        if completion["identity"] != identity:
            raise ValueError("completion identity mismatch")
        return completion
    jobs = [(seed, model) for model in config["models"] for seed in config["seeds"]]
    finished = []
    if config["workers"] == 1:
        for seed, model in jobs:
            finished.append(run_job(config, identity, seed, model, output, device))
    else:
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            pending = [pool.submit(run_job, config, identity, seed, model, output, device)
                       for seed, model in jobs]
            for future in as_completed(pending):
                finished.append(future.result())
    if source_manifest() != sources:
        raise ValueError("scientific source changed during fitting")
    verify_analysis(output)
    completion = dict(identity=identity, **counts(config), jobs=sorted(finished, key=lambda x: (x["model"], x["seed"])),
                      completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/"completion.json", completion)
    print(json.dumps(dict(completed=True, **counts(config)), indent=2))
    return completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    configuration = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_suite(configuration, args.output, args.device, args.resume, args.protocol,
              args.lock_only, args.config)
