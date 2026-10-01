"""Reproducible study runner with paired data, isolated probes and crash recovery."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .learner import Learner
from .memory import METHODS
from .world import ACTIONS, COMPOSITIONS, HORIZONS, KNOWN, TransferWorld, schedule


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def trial_seed(seed, channel, identifier):
    return int(digest([seed, channel, identifier])[:15], 16)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_manifest():
    base = Path(__file__).resolve().parent
    root = base.parent.parent
    paths = [base.parent / "__init__.py", *sorted(base.glob("*.py"))]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_config(config):
    required = {"study", "seeds", "schedules", "methods", "block_size", "batch_size", "eval_size",
                "probe_every", "width", "lr", "memory_capacity", "updates_per_batch", "threads"}
    if set(config) != required:
        raise ValueError(f"configuration fields differ: {set(config) ^ required}")
    for key in ("block_size", "batch_size", "eval_size", "probe_every", "width", "memory_capacity",
                "updates_per_batch", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not np.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("lr must be finite and positive")
    if config["block_size"] % config["batch_size"] or config["probe_every"] % config["batch_size"]:
        raise ValueError("block size and probe cadence must be multiples of batch size")
    if config["block_size"] % config["probe_every"]:
        raise ValueError("probe cadence must divide block size")
    for key in ("methods", "seeds", "schedules"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must be a nonempty unique list")
    if any(type(seed) is not int or seed < 0 for seed in config["seeds"]):
        raise ValueError("seeds must be nonnegative integers")
    if any(method not in METHODS for method in config["methods"]):
        raise ValueError("unknown method")
    for stream in config["schedules"]:
        schedule(config["seeds"][0], stream)


class Evaluator:
    def __init__(self, world, seed, size):
        self.world, self.seed, self.size = world, seed, size
        self.cache = {}

    def dataset(self, regime):
        key = digest(asdict(regime))
        if key not in self.cache:
            cases = self.world.cases(regime, self.size, trial_seed(self.seed, "evaluation", key))
            outcomes = self.world.counterfactuals(cases)
            self.cache[key] = cases, outcomes
        return self.cache[key]

    def evaluate(self, learner, regime, *, temporal_control=False):
        cases, outcomes = self.dataset(regime)
        observations = cases.observations
        if temporal_control:
            observations = np.repeat(observations[:, -1:], observations.shape[1], axis=1)
        predictions = learner.predict(observations)
        action = predictions[:, :, -1].argmax(axis=1)
        selected = outcomes[np.arange(self.size), action]
        brier = (predictions - outcomes) ** 2
        consequential = np.ptp(outcomes[:, :, -1], axis=1) > 0
        lifetime = self.world.simulate(cases, action).lifetime
        return {
            "survival": {str(h): float(selected[:, j].mean()) for j, h in enumerate(HORIZONS)},
            "restricted_lifetime": float(np.minimum(lifetime, max(HORIZONS)).mean()),
            "brier": float(brier.mean()),
            "consequential_brier": (float(brier[consequential].mean())
                                     if consequential.any() else None),
            "no_transfer": float(outcomes[:, 0, -1].mean()),
            "best_fixed_action": float(outcomes[:, :, -1].mean(axis=0).max()),
            "clairvoyant_upper": float(outcomes[:, :, -1].max(axis=1).mean()),
            "consequential_fraction": float(consequential.mean()),
            "action_counts": np.bincount(action, minlength=len(ACTIONS)).tolist(),
        }


def panel(evaluator, learner, regimes):
    return {regime.name: evaluator.evaluate(learner, regime) for regime in regimes}


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_one(config, identity, method, seed, stream, output, device):
    directory = output / f"{stream}_{method}_{seed}"
    directory.mkdir(exist_ok=True)
    result_path, checkpoint_path = directory / "result.json", directory / "checkpoint.pt"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing["identity"] != identity:
            raise ValueError("completed result identity mismatch")
        return existing
    world = TransferWorld()
    regimes, schedule_info = schedule(seed, stream)
    evaluator = Evaluator(world, seed, config["eval_size"])
    start_time = time.perf_counter()
    if checkpoint_path.exists():
        # Only load this runner's trusted local checkpoint, after identity checks.
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload["identity"] != identity:
            raise ValueError("checkpoint identity mismatch")
        learner, record = payload["learner"], payload["record"]
        start_block = len(record["blocks"])
    else:
        learner = Learner(method, seed, config, device)
        record = {"identity": identity, "method": method, "seed": seed, "schedule": stream,
                  "schedule_info": schedule_info, "blocks": [], "panels": [], "elapsed_seconds": 0.0}
        record["panels"].append({"after_block": -1,
                                 "metrics": panel(evaluator, learner, (*KNOWN, *COMPOSITIONS))})
        start_block = 0
    occurrence = Counter(regime.name for regime in regimes[:start_block])
    for block, regime in enumerate(regimes[start_block:], start=start_block):
        if method == "frozen_late" and block == 4:
            learner.freeze_encoder()
        data = world.record(regime, config["block_size"], trial_seed(
            seed, "training", [asdict(regime), occurrence[regime.name]]))
        occurrence[regime.name] += 1
        curves = [{"arrivals": 0, "metrics": evaluator.evaluate(learner, regime)}]
        for start in range(0, len(data), config["batch_size"]):
            learner.train(data.take(slice(start, start + config["batch_size"])))
            arrivals = start + config["batch_size"]
            if arrivals % config["probe_every"] == 0:
                curves.append({"arrivals": arrivals, "metrics": evaluator.evaluate(learner, regime)})
        record["blocks"].append({"block": block, "regime": asdict(regime),
                                  "data_sha256": data.fingerprint(), "curve": curves})
        if block in (3, 9, 10, 11):
            record["panels"].append({"after_block": block,
                                     "metrics": panel(evaluator, learner, (*KNOWN, *COMPOSITIONS))})
        record["elapsed_seconds"] += time.perf_counter() - start_time
        atomic_checkpoint(checkpoint_path, {"identity": identity, "learner": learner, "record": record})
        start_time = time.perf_counter()
        first, last = (item["metrics"]["survival"]["12"] for item in (curves[0], curves[-1]))
        print(f"{stream} {method} seed={seed} block={block + 1}/12 "
              f"survival={first:.3f}->{last:.3f}", flush=True)
    # A changed rule invalidates the previous target's labels. Keep raw old-world
    # probes for transparency, but exclude that obsolete target from final retention.
    valid_known = [r for r in KNOWN if stream != "reversal" or r.name != schedule_info["target"]]
    record["final_temporal_control"] = {
        r.name: evaluator.evaluate(learner, r, temporal_control=True) for r in valid_known}
    record["diagnostics"] = learner.diagnostics()
    record["elapsed_seconds"] += time.perf_counter() - start_time
    write_json(result_path, record)
    return record


def run_scratch(config, identity, seed, stream, output, device):
    path = output / f"{stream}_scratch_{seed}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError("scratch identity mismatch")
        return record
    regime = schedule(seed, stream)[0][-1]
    world = TransferWorld()
    data = world.record(regime, config["block_size"], trial_seed(seed, "training", [asdict(regime), 0]))
    evaluator = Evaluator(world, seed, config["eval_size"])
    learner = Learner("uniform", seed, config, device)
    curve = [{"arrivals": 0, "metrics": evaluator.evaluate(learner, regime)}]
    for start in range(0, len(data), config["batch_size"]):
        learner.train(data.take(slice(start, start + config["batch_size"])))
        arrivals = start + config["batch_size"]
        if arrivals % config["probe_every"] == 0:
            curve.append({"arrivals": arrivals, "metrics": evaluator.evaluate(learner, regime)})
    record = {"identity": identity, "seed": seed, "schedule": stream,
              "data_sha256": data.fingerprint(), "curve": curve, "diagnostics": learner.diagnostics()}
    write_json(path, record)
    return record


def run_suite(config, output: Path, device="cpu", resume=False):
    validate_config(config)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    runtime = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
               "platform": platform.platform(), "device": device, "threads": torch.get_num_threads(),
               "deterministic": True}
    if device.startswith("cuda"):
        runtime.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(torch.device(device)))
    sources = source_manifest()
    identity = {"config_sha256": digest(config), "source_sha256": digest(sources),
                "runtime_sha256": digest(runtime)}
    if output.exists():
        if not resume:
            raise ValueError("output exists; use a new path, or --resume for identical source/config/runtime")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("suite source/config/runtime mismatch; use a new output path")
    else:
        output.mkdir(parents=True)
        manifest = {"identity": identity, "config": config, "runtime": runtime,
                    "source_files": sources, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": "fixed pilot configuration; no evaluation-driven tuning"}
        write_json(output / "manifest.json", manifest)
        with zipfile.ZipFile(output / "training_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(Path(__file__).resolve().parent.parent.parent / name, name)
            archive.writestr("config.json", json.dumps(config, indent=2))
    for seed in config["seeds"]:
        for stream in config["schedules"]:
            for method in config["methods"]:
                run_one(config, identity, method, seed, stream, output, device)
            run_scratch(config, identity, seed, stream, output, device)
    write_json(output / "completion.json", {"identity": identity,
               "runs": len(config["seeds"]) * len(config["schedules"]) * len(config["methods"]),
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_suite(config, args.output, args.device, args.resume)


if __name__ == "__main__":
    main()
