"""Matched-history study with shared prefixes and independently learned branches."""

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

from acp_cl.conditional.study import Evaluator as BaseEvaluator
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import atomic_checkpoint, digest, trial_seed, write_json
from acp_cl.persistence.world import concatenate
from .learner import ContextLearner, METHODS
from .world import BRANCHES, Environment, ContextWorld, challenge, challenge_batch, prefix, prefix_batch


def source_manifest():
    base = Path(__file__).resolve().parent
    root = base.parent.parent
    paths = [base.parent / "__init__.py"]
    for directory in (base, base.parent / "conditional", base.parent / "persistence"):
        paths.extend(sorted(directory.glob("*.py")))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_config(config):
    fields = {"study", "seeds", "gaps", "branches", "methods", "block_size", "batch_size",
              "eval_size", "probe_every", "support_replicates", "width", "experts", "context_width",
              "decoder_width", "lr", "memory_packets", "updates_per_batch", "evidence_strength",
              "feedback_noise", "interaction_features", "threads", "workers"}
    if set(config) != fields:
        raise ValueError(f"configuration fields differ: {set(config) ^ fields}")
    ints = fields - {"study", "seeds", "gaps", "branches", "methods", "lr",
                     "evidence_strength", "feedback_noise", "interaction_features"}
    if type(config["interaction_features"]) is not bool:
        raise ValueError("interaction_features must be boolean")
    if any(type(config[k]) is not int or config[k] < 1 for k in ints):
        raise ValueError("positive integer settings required")
    if any(not np.isfinite(config[k]) or config[k] <= 0 for k in ("lr", "evidence_strength")):
        raise ValueError("invalid continuous setting")
    if not 0 <= config["feedback_noise"] <= 1:
        raise ValueError("invalid feedback noise")
    if config["batch_size"] % 8 or config["eval_size"] % 8 or config["experts"] < 2:
        raise ValueError("eight-cell batches and at least two diagnostic heads required")
    if config["block_size"] % 4 or (config["block_size"] // 4) % config["probe_every"]:
        raise ValueError("probe cadence must divide quarter-block length")
    if config["probe_every"] % config["batch_size"]:
        raise ValueError("probe cadence must be a batch multiple")
    for key in ("seeds", "gaps", "branches", "methods"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError("nonempty unique selections required")
    if any(type(s) is not int or s < 0 for s in config["seeds"]):
        raise ValueError("invalid seed")
    if any(m not in METHODS for m in config["methods"]):
        raise ValueError("unknown method")
    if any(g not in ("short", "long") for g in config["gaps"]) or any(
            b not in BRANCHES for b in config["branches"]):
        raise ValueError("unknown branch or gap")


class Evaluator(BaseEvaluator):
    def __init__(self, seed, settings):
        self.seed, self.settings = seed, settings
        self.world, self.cache = ContextWorld(), {}

    def support(self, condition, replicate, channel="switch"):
        return self.world.experience(condition, self.settings["batch_size"],
            trial_seed(self.seed, "context_" + channel, [condition.gated, replicate]))[0]

    def switch_probe(self, learner, condition):
        before = state_hash(learner.model.state_dict())
        size = self.settings["batch_size"]
        points, rows = sorted(set((0, size // 4, size // 2, size))), []
        for replicate in range(self.settings["support_replicates"]):
            fresh = self.support(condition, replicate)
            curve = []
            for count in points:
                support = learner.history
                if count:
                    added = fresh.take(slice(0, count))
                    support = (added if support is None else concatenate(support, added).take(slice(-size, None)))
                curve.append(dict(feedback=count, metrics=self.evaluate(learner, condition, support)))
            opposite = Environment(1 - condition.mode, condition.gated, condition.feedback_noise)
            rows.append(dict(curve=curve,
                opposite=self.evaluate(learner, condition, self.support(opposite, replicate)),
                erased=self.evaluate(learner, condition, None),
                broken_binding=self.evaluate(learner, condition, fresh,
                    override=None if learner.method == "pooled" else
                    "oracle" if learner.method == "oracle" else "shuffled")))
        if before != state_hash(learner.model.state_dict()):
            raise AssertionError("inference probe changed model")
        return dict(condition=asdict(condition), replicates=rows, model_sha256=before)


def checked_load(path, identity, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["identity"] != identity:
        raise ValueError("checkpoint identity mismatch")
    return payload


def run_branch(config, identity, seed, gap, method, branch, prefix_learner, prefix_record, output, device):
    directory = Path(output) / f"{gap}_{branch}_{method}_{seed}"
    directory.mkdir(exist_ok=True)
    result_path, checkpoint = directory / "result.json", directory / "checkpoint.pt"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError("result identity mismatch")
        return result
    started = time.perf_counter()
    evaluator, world = Evaluator(seed, config), ContextWorld()
    condition = challenge(seed, branch, config)
    if checkpoint.exists():
        payload = checked_load(checkpoint, identity, device)
        learner, record = payload["learner"], payload["record"]
    else:
        learner = copy.deepcopy(prefix_learner)
        record = dict(identity=identity, seed=seed, gap=gap, method=method, branch=branch,
                      prefix=copy.deepcopy(prefix_record), condition=asdict(condition),
                      before=evaluator.switch_probe(learner, condition), batches_done=0,
                      batch_sha256=[], elapsed_branch_seconds=0.,
                      curve=[dict(arrivals=0, metrics=evaluator.evaluate(learner, condition, learner.history))])
    for batch in range(record["batches_done"], config["block_size"] // config["batch_size"]):
        data, modes = challenge_batch(world, seed, branch, config, batch)
        learner.train(data, modes if method == "oracle" else None)
        record["batch_sha256"].append(data.fingerprint())
        record["batches_done"] = batch + 1
        arrivals = (batch + 1) * config["batch_size"]
        if arrivals % config["probe_every"] == 0:
            record["curve"].append(dict(arrivals=arrivals,
                metrics=evaluator.evaluate(learner, condition, learner.history)))
            record["elapsed_branch_seconds"] += time.perf_counter() - started
            atomic_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record))
            started = time.perf_counter()
    record["after"] = evaluator.switch_probe(learner, condition)
    record["known"] = {str(mode): evaluator.switch_probe(learner, Environment(mode)) for mode in (0, 1)}
    record["diagnostics"] = learner.diagnostics()
    record["elapsed_branch_seconds"] += time.perf_counter() - started
    write_json(result_path, record)
    return record


def run_job(config, identity, seed, gap, method, output, device):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(output) / f"prefix_{gap}_{method}_{seed}"
    directory.mkdir(exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    evaluator, world = Evaluator(seed, config), ContextWorld()
    started = time.perf_counter()
    if checkpoint.exists():
        payload = checked_load(checkpoint, identity, device)
        learner, record = payload["learner"], payload["record"]
    else:
        learner = ContextLearner(method, seed, config, device)
        record = dict(blocks=[], elapsed_seconds=0.)
    blocks = prefix(seed, gap, config)
    for index in range(len(record["blocks"]), len(blocks)):
        block = blocks[index]
        if index == 4 and method == "recurrent_frozen":
            learner.freeze_features()
        condition = Environment(block["mode"])
        curve = [dict(arrivals=0, metrics=evaluator.evaluate(learner, condition, learner.history))]
        hashes = []
        for batch in range(block["size"] // config["batch_size"]):
            data, modes = prefix_batch(world, seed, block, batch, config["batch_size"])
            hashes.append(data.fingerprint())
            learner.train(data, modes if method == "oracle" else None)
            arrivals = (batch + 1) * config["batch_size"]
            if arrivals % config["probe_every"] == 0:
                curve.append(dict(arrivals=arrivals, metrics=evaluator.evaluate(learner, condition, learner.history)))
        record["blocks"].append(dict(**block, batch_sha256=hashes, curve=curve,
                                     encoder_hash=learner.diagnostics()["encoder_hash"]))
        record["elapsed_seconds"] += time.perf_counter() - started
        atomic_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record))
        started = time.perf_counter()
    prefix_hash = state_hash(learner.model.state_dict())
    for branch in config["branches"]:
        run_branch(config, identity, seed, gap, method, branch, learner, record, output, device)
    if prefix_hash != state_hash(learner.model.state_dict()):
        raise AssertionError("a branch mutated the shared prefix learner")
    return dict(seed=seed, gap=gap, method=method)


def run_suite(config, output, device="cpu", resume=False, protocol=None):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    runtime = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                   numpy=np.__version__, device=device, threads=config["threads"], workers=config["workers"],
                   deterministic=True)
    identity = dict(config_sha256=digest(config), source_sha256=digest(sources), runtime_sha256=digest(runtime))
    path = output / "manifest.json"
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("resume requires identical config/source/runtime")
    else:
        if any(output.iterdir()):
            raise ValueError("new output directory must be empty")
        now = datetime.now(timezone.utc).isoformat()
        write_json(path, dict(identity=identity, config=config, runtime=runtime, source_files=sources, created_utc=now))
        root = Path(__file__).resolve().parent.parent.parent
        with zipfile.ZipFile(output / "training_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(root / name, name)
        if protocol:
            content = Path(protocol).read_bytes()
            (output / "protocol_at_lock.md").write_bytes(content)
            write_json(output / "protocol_lock.json", dict(**identity, locked_utc=now,
                       protocol_sha256=hashlib.sha256(content).hexdigest()))
    jobs = [(config, identity, seed, gap, method, output, device)
            for seed in config["seeds"] for gap in config["gaps"] for method in config["methods"]]
    if config["workers"] == 1:
        for i, job in enumerate(jobs, 1):
            result = run_job(*job)
            print(f"{i}/{len(jobs)} prefixes + branches: {result}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [pool.submit(run_job, *job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                print(f"{i}/{len(jobs)} prefixes + branches: {future.result()}", flush=True)
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("scientific source changed during suite")
    write_json(output / "completion.json", dict(identity=identity, prefixes=len(jobs),
        branches=len(jobs)*len(config["branches"]), completed_utc=datetime.now(timezone.utc).isoformat()))


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
