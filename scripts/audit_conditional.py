"""Recompute primary/final probes and verify causal memories from local checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from acp_cl.conditional.learner import PacketMemory
from acp_cl.conditional.study import Evaluator, source_manifest
from acp_cl.conditional.world import Condition, ConditionalWorld, schedule, training_batch
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest, write_json


def compare(actual, expected, path="root"):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"different fields at {path}")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"different lengths at {path}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=1e-6, atol=1e-8):
            raise AssertionError(f"numeric mismatch at {path}: {actual} versus {expected}")
    elif actual != expected:
        raise AssertionError(f"mismatch at {path}")


def audit(directory, destination):
    started = time.perf_counter()
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    config, identity = manifest["config"], manifest["identity"]
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("current code differs from the pilot's training source")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    world, runs, largest_residual = ConditionalWorld(), [], 0.
    batch_size = config["batch_size"]
    per_block = config["block_size"] // batch_size
    for stream in config["schedules"]:
        for seed in config["seeds"]:
            regenerated = [training_batch(world, seed, stream, block, batch, batch_size)
                           for block in range(12) for batch in range(per_block)]
            hashes = [data.fingerprint() for data, _ in regenerated]
            for method in config["methods"]:
                folder = directory / f"{stream}_{method}_{seed}"
                record = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                if [h for b in record["blocks"] for h in b["batch_sha256"]] != hashes:
                    raise AssertionError("training records do not reproduce")
                saved = torch.load(folder / "checkpoint.pt", map_location=manifest["runtime"]["device"],
                                   weights_only=False)
                if saved["identity"] != identity:
                    raise AssertionError("checkpoint identity mismatch")
                learner = saved["learner"]
                if state_hash(learner.model.state_dict()) != record["diagnostics"]["final_hash"]:
                    raise AssertionError("model hash mismatch")
                rebuilt = PacketMemory(0 if method == "current_only" else config["memory_packets"], seed)
                for index in range(len(regenerated)):
                    rebuilt.add(index)  # Membership RNG depends only on arrivals.
                if rebuilt.ids != learner.memory.ids:
                    raise AssertionError("reservoir membership does not reproduce")
                for index, packet in zip(learner.memory.ids, learner.memory.packets):
                    data, modes = regenerated[index]
                    if packet.query.fingerprint() != data.fingerprint():
                        raise AssertionError("stored query differs from its actual arrival")
                    if index == 0:
                        if packet.support is not None:
                            raise AssertionError("first query has future support")
                    elif packet.support.fingerprint() != regenerated[index - 1][0].fingerprint():
                        raise AssertionError("stored support is not the preceding observed batch")
                    if method == "oracle":
                        np.testing.assert_array_equal(packet.oracle_modes, modes)
                    elif packet.oracle_modes is not None:
                        raise AssertionError("hidden labels entered ordinary replay")
                if learner.history.fingerprint() != regenerated[-1][0].fingerprint():
                    raise AssertionError("live history mismatch")
                evaluator = Evaluator(seed, config)
                before_history = learner.history.fingerprint()
                compare(evaluator.final_panel(learner), record["final_panel"])
                if before_history != learner.history.fingerprint():
                    raise AssertionError("evaluation modified history")
                before_return = torch.load(folder / "return_checkpoint.pt",
                    map_location=manifest["runtime"]["device"], weights_only=False)
                if before_return["identity"] != identity:
                    raise AssertionError("return checkpoint identity mismatch")
                return_learner = before_return["learner"]
                if return_learner.cost["arrivals"] != 8 * config["block_size"]:
                    raise AssertionError("return checkpoint is not pre-return")
                if return_learner.history.fingerprint() != regenerated[8 * per_block - 1][0].fingerprint():
                    raise AssertionError("return checkpoint history not stale pre-return evidence")
                condition = schedule(seed, stream)[8]
                compare(evaluator.switch_probe(return_learner, condition), record["return_probe"])
                if record["return_probe"]["model_sha256"] != state_hash(return_learner.model.state_dict()):
                    raise AssertionError("primary endpoint weights mismatch")
                runs.append(dict(schedule=stream, method=method, seed=seed,
                                 primary_recomputed=True, final_panels_recomputed=True,
                                 causal_replay_verified=True))
            # Check conservation in every mode/gate condition independently of fits.
            for mode in (-1, 0, 1):
                for gated in (False, True):
                    trials = world.dataset(Condition(mode, gated), config["eval_size"], seed)
                    for action in range(5):
                        result = world.simulate(trials.cases, np.full(config["eval_size"], action))
                        largest_residual = max(largest_residual, result.conservation_error)
                        if result.min_stock < 0 or result.conservation_error > 1e-10:
                            raise AssertionError("physical accounting failed")
            print(f"audited {stream} seed={seed}; {len(runs)} checkpoints", flush=True)
    report = dict(passed=True, identity=identity, runs=len(runs),
                  primary_checkpoints=len(runs), final_checkpoints=len(runs),
                  largest_conservation_residual=largest_residual,
                  elapsed_seconds=time.perf_counter() - started, checks=runs,
                  scope="Local checkpoint/data/probe recomputation; not an independent replication.")
    write_json(destination, report)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.input, args.output)
