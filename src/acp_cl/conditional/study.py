"""Paired streams, inference-only probes, source locking, and crash recovery."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import atomic_checkpoint, digest, trial_seed, write_json
from acp_cl.persistence.world import concatenate
from .learner import ConditionalLearner, METHODS
from .world import Condition, ConditionalWorld, schedule, training_batch


def source_manifest():
    base = Path(__file__).resolve().parent
    root = base.parent.parent
    paths = [base.parent / "__init__.py", *sorted(base.glob("*.py")),
             *sorted((base.parent / "persistence").glob("*.py"))]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_config(config):
    fields = {"study", "seeds", "schedules", "methods", "block_size", "batch_size", "eval_size",
              "probe_every", "support_replicates", "width", "experts", "lr", "memory_packets",
              "updates_per_batch", "evidence_strength", "threads", "workers"}
    if set(config) != fields:
        raise ValueError(f"configuration fields differ: {set(config) ^ fields}")
    integer_fields = fields - {"study", "seeds", "schedules", "methods", "lr", "evidence_strength"}
    if any(type(config[k]) is not int or config[k] < 1 for k in integer_fields):
        raise ValueError("positive integer settings required")
    if config["experts"] < 2:
        raise ValueError("the privileged oracle diagnostic needs at least two heads")
    if config["batch_size"] % 8 or config["eval_size"] % 8:
        raise ValueError("batch and evaluation sizes must be multiples of eight")
    if config["block_size"] % config["probe_every"] or config["probe_every"] % config["batch_size"]:
        raise ValueError("block/probe/batch sizes must divide one another")
    if any(not np.isfinite(config[k]) or config[k] <= 0 for k in ("lr", "evidence_strength")):
        raise ValueError("positive finite floating point settings required")
    for key in ("seeds", "schedules", "methods"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must be a nonempty unique list")
    if any(type(s) is not int or s < 0 for s in config["seeds"]):
        raise ValueError("invalid seed")
    if any(method not in METHODS for method in config["methods"]):
        raise ValueError("unknown method")
    for stream in config["schedules"]:
        schedule(config["seeds"][0], stream)


class Evaluator:
    def __init__(self, seed, settings):
        self.seed, self.settings = seed, settings
        self.world = ConditionalWorld()
        self.cache = {}

    def dataset(self, condition):
        key = (condition.mode, condition.gated)
        if key not in self.cache:
            trials = self.world.dataset(condition, self.settings["eval_size"],
                                        trial_seed(self.seed, "evaluation", key))
            outcomes = self.world.counterfactuals(trials.cases)
            self.cache[key] = trials, outcomes
        return self.cache[key]

    def evaluate(self, learner, condition, support, override=None):
        trials, outcomes = self.dataset(condition)
        probabilities, weights, experts = learner.predict(trials.cases.observations, support,
            trials.modes if learner.method == "oracle" else None, override)
        actions = probabilities[:, :, -1].argmax(axis=1)
        selected = outcomes[np.arange(len(actions)), actions]
        return dict(survival=float(selected[:, -1].mean()),
                    horizons=selected.mean(axis=0).tolist(),
                    brier=float(((probabilities - outcomes) ** 2).mean()),
                    no_transfer=float(outcomes[:, 0, -1].mean()),
                    clairvoyant_upper=float(outcomes[:, :, -1].max(axis=1).mean()),
                    best_fixed=float(outcomes[:, :, -1].mean(axis=0).max()),
                    weights=weights.mean(axis=0).tolist(),
                    entropy=float(-(weights * np.log(weights.clip(1e-12))).sum(axis=1).mean()),
                    disagreement=float(experts[:, :, :, -1].std(axis=1).mean()),
                    action_counts=np.bincount(actions, minlength=5).tolist())

    def support(self, condition, replicate, channel="switch"):
        return self.world.experience(condition, self.settings["batch_size"],
            trial_seed(self.seed, channel, [condition.gated, replicate]))[0]

    def switch_probe(self, learner, condition):
        """No training, no state mutation; support feedback comes before queries.

        Query trials are distinct from support trials. Paired opposite-mode
        support has identical observations/actions and different consequences.
        """
        before = state_hash(learner.model.state_dict())
        points = sorted(set((0, self.settings["batch_size"] // 4,
                             self.settings["batch_size"] // 2, self.settings["batch_size"])))
        results = []
        for replicate in range(self.settings["support_replicates"]):
            fresh = self.support(condition, replicate)
            curve = []
            for n in points:
                if n == 0:
                    support = learner.history
                elif learner.history is None:
                    support = fresh.take(slice(0, n))
                else:
                    support = concatenate(learner.history, fresh.take(slice(0, n))).take(
                        slice(-self.settings["batch_size"], None))
                curve.append(dict(feedback=n, metrics=self.evaluate(learner, condition, support)))
            opposite = Condition(1 - condition.mode, condition.gated) if condition.mode != -1 else condition
            wrong = self.support(opposite, replicate)
            results.append(dict(curve=curve,
                opposite=self.evaluate(learner, condition, wrong),
                erased=self.evaluate(learner, condition, None),
                broken_binding=self.evaluate(learner, condition, fresh,
                    override="shuffled" if learner.method != "oracle" else "oracle")))
        if state_hash(learner.model.state_dict()) != before:
            raise AssertionError("inference probe changed parameters")
        return dict(condition=asdict(condition), replicates=results, model_sha256=before)

    def final_panel(self, learner):
        return {f"mode{mode}_gate{int(gated)}": self.switch_probe(learner, Condition(mode, gated))
                for gated in (False, True) for mode in (0, 1)}


def run_one(config, identity, method, seed, stream, output, device):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(output) / f"{stream}_{method}_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = directory / "result.json", directory / "checkpoint.pt"
    if result_path.exists():
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError("completed run identity mismatch")
        return record
    started = time.perf_counter()
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("checkpoint identity mismatch")
        learner, record = saved["learner"], saved["record"]
    else:
        learner = ConditionalLearner(method, seed, config, device)
        record = dict(identity=identity, method=method, seed=seed, schedule=stream,
                      blocks=[], elapsed_seconds=0.0)
    world, evaluator = ConditionalWorld(), Evaluator(seed, config)
    conditions = schedule(seed, stream)
    for block in range(len(record["blocks"]), len(conditions)):
        if block == 4 and method == "frozen_features":
            learner.freeze_features()
        if block == 8:
            record["return_probe"] = evaluator.switch_probe(learner, conditions[block])
            # Retain the exact frozen weights and stale history for an independent
            # recomputation of the main scientific endpoint, not just the final fit.
            atomic_checkpoint(directory / "return_checkpoint.pt", dict(identity=identity, learner=learner))
        condition = conditions[block]
        curve = [dict(arrivals=0, metrics=evaluator.evaluate(learner, condition, learner.history))]
        hashes = []
        for batch in range(config["block_size"] // config["batch_size"]):
            data, modes = training_batch(world, seed, stream, block, batch, config["batch_size"])
            hashes.append(data.fingerprint())
            learner.train(data, modes if method == "oracle" else None)
            arrivals = (batch + 1) * config["batch_size"]
            if arrivals % config["probe_every"] == 0:
                curve.append(dict(arrivals=arrivals,
                                  metrics=evaluator.evaluate(learner, condition, learner.history)))
        record["blocks"].append(dict(block=block, condition=asdict(condition),
                                     batch_sha256=hashes, curve=curve))
        record["elapsed_seconds"] += time.perf_counter() - started
        atomic_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record))
        started = time.perf_counter()
    record["final_panel"] = evaluator.final_panel(learner)
    record["diagnostics"] = learner.diagnostics()
    record["elapsed_seconds"] += time.perf_counter() - started
    write_json(result_path, record)
    return record


def run_suite(config, output, device="cpu", resume=False, protocol=None):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    runtime = dict(python=platform.python_version(), platform=platform.platform(),
                   torch=torch.__version__, numpy=np.__version__, device=device,
                   threads=config["threads"], workers=config["workers"], deterministic=True)
    identity = dict(config_sha256=digest(config), source_sha256=digest(sources),
                    runtime_sha256=digest(runtime))
    path = output / "manifest.json"
    if path.exists():
        if not resume:
            raise ValueError("existing study requires --resume or a new output path")
        if json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("resume config/source/runtime mismatch")
    else:
        if any(output.iterdir()):
            raise ValueError("new study directory must be empty")
        now = datetime.now(timezone.utc).isoformat()
        write_json(path, dict(identity=identity, config=config, source_files=sources,
                             runtime=runtime, created_utc=now))
        root = Path(__file__).resolve().parent.parent.parent
        with zipfile.ZipFile(output / "training_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(root / name, name)
        if protocol is not None:
            content = Path(protocol).read_bytes()
            (output / "protocol_at_lock.md").write_bytes(content)
            write_json(output / "protocol_lock.json", dict(**identity, locked_utc=now,
                       protocol_sha256=hashlib.sha256(content).hexdigest()))
    jobs = [(config, identity, method, seed, stream, output, device)
            for stream in config["schedules"] for seed in config["seeds"] for method in config["methods"]]
    completed = 0
    if config["workers"] == 1:
        for job in jobs:
            record = run_one(*job)
            completed += 1
            print(f"{completed}/{len(jobs)} {record['schedule']} {record['method']} seed={record['seed']}",
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [pool.submit(run_one, *job) for job in jobs]
            for future in as_completed(futures):
                record = future.result()
                completed += 1
                print(f"{completed}/{len(jobs)} {record['schedule']} {record['method']} seed={record['seed']}",
                      flush=True)
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("training source changed during suite")
    write_json(output / "completion.json", dict(identity=identity, runs=completed,
                                                completed_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--protocol")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_suite(config, args.output, args.device, args.resume, args.protocol)


if __name__ == "__main__":
    main()
