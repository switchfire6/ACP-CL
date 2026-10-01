"""Recompute final probes and verify actual training-memory contents of local runs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import (Evaluator, digest, source_manifest, trial_seed, write_json)
from acp_cl.persistence.world import COMPOSITIONS, KNOWN, TransferWorld, concatenate, schedule


def assert_metrics_equal(actual, expected):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("metric keys differ")
        for key in expected:
            assert_metrics_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        if actual != expected:
            raise AssertionError("metric counts differ")
    elif expected is None:
        if actual is not None:
            raise AssertionError("missing metric differs")
    elif not np.isclose(actual, expected, rtol=0, atol=1e-7):
        raise AssertionError(f"metric differs: {actual} versus {expected}")


def audit(directory, output):
    manifest = json.loads((directory / "manifest.json").read_text())
    config, identity = manifest["config"], manifest["identity"]
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("current source differs from training; audit using archived source")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    device = manifest["runtime"]["device"]
    world = TransferWorld()
    checked, largest_balance_error = [], 0.0
    started = time.perf_counter()
    for seed in config["seeds"]:
        for stream in config["schedules"]:
            regimes, _ = schedule(seed, stream)
            occurrences, all_data, data_hashes = Counter(), None, []
            for regime in regimes:
                seed_value = trial_seed(seed, "training", [asdict(regime), occurrences[regime.name]])
                data = world.record(regime, config["block_size"], seed_value)
                cases = world.cases(regime, config["block_size"], seed_value)
                outcome = world.simulate(cases, data.actions)
                largest_balance_error = max(largest_balance_error, outcome.conservation_error)
                if outcome.min_stock < 0 or outcome.conservation_error > 1e-9:
                    raise AssertionError("resource accounting failed")
                data_hashes.append(data.fingerprint())
                all_data = data if all_data is None else concatenate(all_data, data)
                occurrences[regime.name] += 1
            evaluator = Evaluator(world, seed, config["eval_size"])
            for method in config["methods"]:
                folder = directory / f"{stream}_{method}_{seed}"
                record = json.loads((folder / "result.json").read_text())
                if record["identity"] != identity:
                    raise ValueError("result identity differs")
                if [b["data_sha256"] for b in record["blocks"]] != data_hashes:
                    raise AssertionError("regenerated current arrivals differ")
                # These are checkpoints produced locally by this exact audited source.
                payload = torch.load(folder / "checkpoint.pt", map_location=device, weights_only=False)
                if payload["identity"] != identity or len(payload["record"]["blocks"]) != 12:
                    raise ValueError("incomplete/mismatched checkpoint")
                learner = payload["learner"]
                if state_hash(learner.model.state_dict()) != record["diagnostics"]["final_hash"]:
                    raise AssertionError("checkpoint/result model mismatch")
                if learner.memory.seen != len(all_data) or np.any(learner.memory.ids >= len(all_data)):
                    raise AssertionError("memory contains future or unobserved arrivals")
                expected_data = all_data.take(learner.memory.ids)
                for name in ("observations", "actions", "survival"):
                    np.testing.assert_array_equal(getattr(learner.memory.data, name), getattr(expected_data, name))
                final_panel = next(p["metrics"] for p in record["panels"] if p["after_block"] == 11)
                for regime in (*KNOWN, *COMPOSITIONS):
                    assert_metrics_equal(evaluator.evaluate(learner, regime), final_panel[regime.name])
                assert_metrics_equal(evaluator.evaluate(learner, regimes[-1]),
                                     record["blocks"][-1]["curve"][-1]["metrics"])
                if state_hash(learner.model.state_dict()) != record["diagnostics"]["final_hash"]:
                    raise AssertionError("evaluation mutated model")
                checked.append({"schedule": stream, "method": method, "seed": seed,
                                "checkpoint_sha256": hashlib.sha256((folder / "checkpoint.pt").read_bytes()).hexdigest()})
            print(f"audited seed={seed} schedule={stream}: data, conservation, memory and final probes", flush=True)
    result = {"identity": identity, "runs_checked": len(checked), "max_conservation_error": largest_balance_error,
              "final_probes_recomputed_per_run": 9, "actual_memory_contents_verified": True,
              "training_arrivals_regenerated": True, "elapsed_seconds": time.perf_counter() - started,
              "checkpoints": checked,
              "scope": "Local deterministic recomputation; not independent replication or complete trajectory replay",
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.input, args.output)


if __name__ == "__main__":
    main()
