"""Recompute contextual probes and causal memory from trusted local checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from audit_conditional import compare
from summarize_contextual import audit_records
from acp_cl.conditional.learner import PacketMemory
from acp_cl.contextual.study import Evaluator, source_manifest
from acp_cl.contextual.world import ContextWorld, Environment, challenge, challenge_batch, prefix, prefix_batch
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest, write_json


def audit_memory(learner, regenerated, config, seed, method):
    rebuilt = PacketMemory(config["memory_packets"], seed)
    for index in range(len(regenerated)):
        for _ in range(config["updates_per_batch"]):
            rebuilt.sample()
        rebuilt.add(index)
    if rebuilt.ids != learner.memory.ids or rebuilt.seen != learner.memory.seen:
        raise AssertionError("reservoir membership/count mismatch")
    for kind in ("membership_rng", "sampling_rng"):
        if getattr(rebuilt, kind).bit_generator.state != getattr(learner.memory, kind).bit_generator.state:
            raise AssertionError("memory RNG state mismatch")
    for index, packet in zip(learner.memory.ids, learner.memory.packets):
        data, modes = regenerated[index]
        if packet.query.fingerprint() != data.fingerprint():
            raise AssertionError("stored query differs from actual arrival")
        if index == 0:
            if packet.support is not None:
                raise AssertionError("first packet has future history")
        elif packet.support.fingerprint() != regenerated[index-1][0].fingerprint():
            raise AssertionError("support is not the preceding observed batch")
        if method == "oracle":
            np.testing.assert_array_equal(packet.oracle_modes, modes)
        elif packet.oracle_modes is not None:
            raise AssertionError("hidden labels entered ordinary replay")
    if learner.history.fingerprint() != regenerated[-1][0].fingerprint():
        raise AssertionError("live history mismatch")
    if learner.cost["arrivals"] != len(regenerated)*config["batch_size"]:
        raise AssertionError("arrival count mismatch")


def audit(directory, destination):
    started = time.perf_counter()
    directory = Path(directory)
    manifest, _, records, _ = audit_records(directory, require_lock=False)
    config, identity = manifest["config"], manifest["identity"]
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("audit must run with the exact archived training source")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    world, checks, prefix_count = ContextWorld(), [], 0
    device = manifest["runtime"]["device"]
    for seed in config["seeds"]:
        for gap in config["gaps"]:
            data = [prefix_batch(world, seed, block, batch, config["batch_size"])
                    for block in prefix(seed, gap, config)
                    for batch in range(block["size"]//config["batch_size"])]
            data_hashes = [d.fingerprint() for d, _ in data]
            challenges = {branch: [challenge_batch(world, seed, branch, config, batch)
                for batch in range(config["block_size"]//config["batch_size"])] for branch in config["branches"]}
            for method in config["methods"]:
                saved = torch.load(directory/f"prefix_{gap}_{method}_{seed}/checkpoint.pt",
                                   map_location=device, weights_only=False)
                if saved["identity"] != identity:
                    raise AssertionError("prefix identity mismatch")
                prefix_learner = saved["learner"]
                audit_memory(prefix_learner, data, config, seed, method)
                prefix_hash = state_hash(prefix_learner.model.state_dict())
                prefix_count += 1
                for branch in config["branches"]:
                    record = records[seed, gap, method, branch]
                    if [h for b in record["prefix"]["blocks"] for h in b["batch_sha256"]] != data_hashes:
                        raise AssertionError("prefix data do not reproduce")
                    if record["prefix"] != saved["record"] or record["before"]["model_sha256"] != prefix_hash:
                        raise AssertionError("branch did not start from recorded prefix")
                    fresh = challenges[branch]
                    if record["batch_sha256"] != [d.fingerprint() for d, _ in fresh]:
                        raise AssertionError("challenge data do not reproduce")
                    evaluator = Evaluator(seed, config)
                    condition = challenge(seed, branch, config)
                    compare(evaluator.switch_probe(prefix_learner, condition), record["before"])
                    final = torch.load(directory/f"{gap}_{branch}_{method}_{seed}/checkpoint.pt",
                                       map_location=device, weights_only=False)
                    if final["identity"] != identity:
                        raise AssertionError("branch identity mismatch")
                    learner = final["learner"]
                    if state_hash(learner.model.state_dict()) != record["diagnostics"]["final_hash"]:
                        raise AssertionError("final model hash mismatch")
                    audit_memory(learner, data+fresh, config, seed, method)
                    compare(learner.diagnostics(), record["diagnostics"])
                    compare(final["record"]["curve"], record["curve"])
                    compare(evaluator.switch_probe(learner, condition), record["after"])
                    for mode in (0, 1):
                        compare(evaluator.switch_probe(learner, Environment(mode)), record["known"][str(mode)])
                    checks.append(dict(seed=seed, gap=gap, method=method, branch=branch,
                        data_regenerated=True, replay_causality=True, initial_final_probes_recomputed=True))
            print(f"audited {gap} seed={seed}: {len(checks)} branches", flush=True)
    result = dict(passed=True, identity=identity, prefix_checkpoints=prefix_count,
        final_checkpoints=len(checks), elapsed_seconds=time.perf_counter()-started, checks=checks,
        scope="Local recomputation with exact archived training source; not independent replication.")
    write_json(destination, result)
    return result


def archived_audit(directory, destination, source_directory):
    directory, source_directory = Path(directory), Path(source_directory).resolve()
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    source_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(directory/"training_source.zip") as archive:
        for name, expected in manifest["source_files"].items():
            target = (source_directory/name).resolve()
            if not target.is_relative_to(source_directory):
                raise ValueError("source path escapes audit directory")
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("archived source hash mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != content:
                raise ValueError("audit source directory already contains a different version")
            target.write_bytes(content)
    environment = dict(os.environ, PYTHONPATH=str(source_directory))
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--input", str(directory.resolve()),
                    "--output", str(Path(destination).resolve())], env=environment, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archived-source", help="Extract verified source here and audit in a separate process")
    args = parser.parse_args()
    if args.archived_source:
        archived_audit(args.input, args.output, args.archived_source)
    else:
        audit(args.input, args.output)
