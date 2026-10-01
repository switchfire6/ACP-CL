"""Recompute acquisition probes and verify diagnostic forks and causal memories."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import torch

from audit_conditional import compare
from summarize_acquisition import audit_records
from acp_cl.acquisition.study import (Evaluator, fork, marginal, preceding_history, source_manifest)
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.conditional.learner import Packet, PacketMemory
from acp_cl.contextual.learner import ContextLearner
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest, write_json


def tensor_tree(value):
    if isinstance(value, torch.Tensor):
        v = value.detach().cpu().contiguous()
        return dict(dtype=str(v.dtype), shape=list(v.shape), sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): tensor_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree(v) for v in value]
    return value


def memory_signature(memory):
    return dict(ids=list(memory.ids), seen=memory.seen,
        membership=memory.membership_rng.bit_generator.state,
        sampling=memory.sampling_rng.bit_generator.state,
        packets=[dict(support=None if p.support is None else p.support.fingerprint(),
                      query=p.query.fingerprint(), hidden=p.oracle_modes is not None) for p in memory.packets])


def signature(learner):
    return digest(dict(model=state_hash(learner.model.state_dict()),
        optimizer=tensor_tree(learner.optimizer.state_dict()),
        trainable=[p.requires_grad for p in learner.model.parameters()],
        memory=memory_signature(learner.memory),
        history=None if learner.history is None else learner.history.fingerprint()))


def advance_memory(memory, history, records, config):
    memory = copy.deepcopy(memory)
    for data in records:
        for _ in range(config["updates_per_batch"]):
            memory.sample()
        memory.add(Packet(history, data))
        history = data
    return memory, history


def law_from_json(value):
    return Law(value["mode"], tuple(value["active"]), value["revised"], value["noise"])


def audit(directory, output):
    started = time.perf_counter()
    directory = Path(directory)
    manifest, _, records, _ = audit_records(directory)
    config, identity = manifest["config"], manifest["identity"]
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("audit requires the study's exact training source")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    world, checks, prefix_count = AcquisitionWorld(), [], 0

    def load(path):
        saved = torch.load(path, map_location=manifest["runtime"]["device"], weights_only=False)
        if saved["identity"] != identity:
            raise AssertionError("checkpoint identity mismatch")
        return saved["learner"], saved["record"]

    for seed in config["seeds"]:
        for model in config["models"]:
            folder = directory/f"{model}_{seed}"
            evaluator = Evaluator(seed, config)
            prefix = None
            if config["kind"] == "diagnostic":
                prefix, record = load(folder/"prefix.pt")
                data = []
                for block, saved in enumerate(record["blocks"]):
                    law = Law((seed+block) % 2)
                    observed = [batch(world, law, seed, f"prefix_{block}", i, config["batch_size"])
                                for i in range(config["prefix_size"]//config["batch_size"])]
                    if saved["batch_sha256"] != [d.fingerprint() for d in observed] or digest(saved["law"]) != digest(asdict(law)):
                        raise AssertionError("prefix data/condition mismatch")
                    data += observed
                memory, history = advance_memory(PacketMemory(config["memory_packets"], seed), None, data, config)
                if memory_signature(memory) != memory_signature(prefix.memory) or history.fingerprint() != prefix.history.fingerprint():
                    raise AssertionError("prefix replay is not causal")
                if state_hash(prefix.model.state_dict()) != record["blocks"][-1]["diagnostics"]["final_hash"]:
                    raise AssertionError("prefix model hash mismatch")
                prefix_count += 1
            for key in sorted(k for k in records if k[0] == seed and k[1] == model):
                r = records[key]
                stage, arm, branch = key[2:]
                tag = f"stage_{stage}_{arm}" if branch == "novel" else f"final_{branch}"
                before, _ = load(folder/tag/"before.pt")
                after, saved = load(folder/tag/"checkpoint.pt")
                if config["kind"] == "development":
                    expected = ContextLearner(model, seed, config, str(before.device))
                    expected.history = preceding_history(config, seed, stage)
                else:
                    base = prefix if stage == 1 else load(folder/f"stage_{stage-1 if stage < 4 else 3}_continue/checkpoint.pt")[0]
                    expected = fork(base, arm, seed, config) if branch == "novel" else copy.deepcopy(base)
                if signature(before) != signature(expected):
                    raise AssertionError("diagnostic fork differs in model, optimizer, replay, history, or trainability")
                law = law_from_json(r["law"])
                observed = [batch(world, law, seed, r["channel"], i, config["batch_size"])
                            for i in range(config["episode_size"]//config["batch_size"])]
                if r["batch_sha256"] != [d.fingerprint() for d in observed]:
                    raise AssertionError("episode data do not regenerate")
                memory, history = advance_memory(before.memory, before.history, observed, config)
                if memory_signature(memory) != memory_signature(after.memory) or history.fingerprint() != after.history.fingerprint():
                    raise AssertionError("episode replay is not causal")
                compare(after.diagnostics(), r["diagnostics"])
                compare(saved["curve"], r["curve"])
                initial_state, final_state = signature(before), signature(after)
                compare(evaluator.probe(before, law, r["cue"], branch), r["before"])
                compare(evaluator.valid_panel(before, law, branch), r["valid_before"])
                compare(evaluator.probe(after, law, r["cue"], branch, marginal=marginal(r)), r["after"])
                compare(evaluator.valid_panel(after, law, branch), r["valid_after"])
                compare(evaluator.evaluate(before, law, before.history, r["cue"], branch), r["curve"][0]["metrics"])
                compare(evaluator.evaluate(after, law, after.history, r["cue"], branch, marginal=marginal(r)), r["curve"][-1]["metrics"])
                if initial_state != signature(before) or final_state != signature(after):
                    raise AssertionError("evaluation changed learning state")
                checks.append(dict(seed=seed, model=model, stage=stage, arm=arm, branch=branch,
                    exact_fork_verified=True, causal_data_replay_verified=True, initial_final_probes_recomputed=True))
            print(f"audited {model} seed={seed}: {len(checks)} episodes", flush=True)
    result = dict(passed=True, identity=identity, prefix_checkpoints=prefix_count,
        before_checkpoints=len(checks), final_checkpoints=len(checks), checks=checks,
        elapsed_seconds=time.perf_counter()-started,
        scope="Local exact-source checkpoint, fork, causal-memory and probe audit; not independent replication.")
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.input, args.output)
