"""Recompute exact forks, causal memory states and endpoint probes."""

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch

from audit_conditional import compare
from audit_training_state import advance_memory, law_from_json, tensor_tree
from summarize_replay_renewal import audit_records, record_path
from acp_cl.acquisition.study import Evaluator, marginal, preceding_history
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest
from acp_cl.replay_renewal.memory import POLICIES, make_memory, memory_state
from acp_cl.replay_renewal.study import affinity, fork, new_learner, source_manifest, write_json


def signature(learner):
    return digest(dict(model=state_hash(learner.model.state_dict()),
        optimizer=tensor_tree(learner.optimizer.state_dict()),
        trainable=[p.requires_grad for p in learner.model.parameters()],
        memory=memory_state(learner.memory),
        history=None if learner.history is None else learner.history.fingerprint()))


def audit(directory, output):
    started = time.perf_counter()
    directory = Path(directory)
    manifest, records, _ = audit_records(directory)
    config, identity = manifest["config"], manifest["identity"]
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("audit requires the exact study source")
    affinity(config["cpu_affinity"])
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
            evaluator, prefixes = Evaluator(seed, config), {}
            tags = ("base",) if config["experiment"] == "decomposition" else POLICIES
            for policy in tags:
                prefix, record = load(directory/f"{model}_{seed}_{policy}"/"prefix.pt")
                if len(record["blocks"]) != config["prefix_blocks"]:
                    raise AssertionError("incomplete prefix")
                data = []
                for block, saved in enumerate(record["blocks"]):
                    law = Law((seed+block) % 2)
                    observed = [batch(world, law, seed, f"prefix_{block}", i, config["batch_size"])
                                for i in range(config["prefix_size"]//config["batch_size"])]
                    if saved["batch_sha256"] != [d.fingerprint() for d in observed] or digest(saved["law"]) != digest(asdict(law)):
                        raise AssertionError("prefix data/condition mismatch")
                    data += observed
                actual_policy = "uniform" if policy == "base" else policy
                memory, history = advance_memory(make_memory(actual_policy, config, seed), None, data, config)
                if memory_state(memory) != memory_state(prefix.memory) or history.fingerprint() != prefix.history.fingerprint():
                    raise AssertionError("prefix memory is not causal")
                compare(memory_state(prefix.memory), record["memory"])
                compare(prefix.diagnostics(), record["blocks"][-1]["diagnostics"])
                initial = signature(prefix)
                compare(evaluator.evaluate(prefix, law, prefix.history), record["blocks"][-1]["endpoint"])
                if signature(prefix) != initial:
                    raise AssertionError("prefix probe changed state")
                prefixes[policy] = prefix
                prefix_count += 1
            for key in sorted(k for k in records if k[0] == seed and k[1] == model):
                r = records[key]
                stage, arm, branch = key[2:]
                folder = record_path(directory, config, key)
                before, start_record = load(folder/"before.pt")
                after, saved = load(folder/"checkpoint.pt")
                if digest(saved) != digest(r):
                    raise AssertionError("completed checkpoint/result mismatch")
                if arm == "fresh":
                    expected = new_learner(model, seed, config, str(before.device))
                    expected.history = preceding_history(config, seed, stage)
                elif config["experiment"] == "decomposition":
                    expected = fork(prefixes["base"], arm, seed, config)
                elif stage == 1:
                    expected = prefixes[arm]
                else:
                    previous = seed, model, min(stage-1, 3), arm, "novel"
                    expected = load(record_path(directory, config, previous)/"checkpoint.pt")[0]
                if signature(before) != signature(expected):
                    raise AssertionError("wrong model/optimizer/memory/history at episode start")
                compare(before.diagnostics(), r["start_diagnostics"])
                compare(memory_state(before.memory), r["start_memory"])
                if digest(start_record["before"]) != digest(r["before"]):
                    raise AssertionError("before checkpoint record mismatch")
                law = law_from_json(r["law"])
                observed = [batch(world, law, seed, r["channel"], i, config["batch_size"])
                            for i in range(config["episode_size"]//config["batch_size"])]
                if r["batch_sha256"] != [d.fingerprint() for d in observed]:
                    raise AssertionError("episode data do not regenerate")
                memory, history = advance_memory(before.memory, before.history, observed, config)
                if memory_state(memory) != memory_state(after.memory) or history.fingerprint() != after.history.fingerprint():
                    raise AssertionError("episode memory is not causal")
                compare(memory_state(after.memory), r["end_memory"])
                compare(after.diagnostics(), r["diagnostics"])
                initial_state, final_state = signature(before), signature(after)
                compare(evaluator.probe(before, law, r["cue"], branch), r["before"])
                compare(evaluator.valid_panel(before, law, branch), r["valid_before"])
                compare(evaluator.probe(after, law, r["cue"], branch, marginal=marginal(r)), r["after"])
                compare(evaluator.valid_panel(after, law, branch), r["valid_after"])
                compare(evaluator.evaluate(before, law, before.history, r["cue"], branch), r["curve"][0]["metrics"])
                compare(evaluator.evaluate(after, law, after.history, r["cue"], branch, marginal=marginal(r)), r["curve"][-1]["metrics"])
                if initial_state != signature(before) or final_state != signature(after):
                    raise AssertionError("evaluation changed learning state")
                ids, seen = after.memory.ids, after.memory.seen
                checks.append(dict(seed=seed, model=model, stage=stage, arm=arm, branch=branch,
                    exact_initial_state_verified=True, causal_data_replay_verified=True, endpoint_probes_recomputed=True,
                    final_packet_ages=[seen-1-i for i in ids], mean_final_packet_age=sum(seen-1-i for i in ids)/len(ids)))
            print(f"audited {model} seed={seed}: {len(checks)} episodes", flush=True)
    paired_count = 0
    if config["experiment"] == "policy":
        for seed in config["seeds"]:
            for model in config["models"]:
                for arm in POLICIES:
                    clean, noisy = (records[seed, model, 4, arm, b] for b in ("clean", "noise"))
                    compare(clean["valid_before"], noisy["valid_before"])
                    compare(clean["curve"][0], noisy["curve"][0])
                    compare(clean["start_memory"], noisy["start_memory"])
                    for i in range(config["episode_size"]//config["batch_size"]):
                        a, b = (batch(world, law_from_json(r["law"]), seed, r["channel"], i, config["batch_size"])
                                for r in (clean, noisy))
                        if not torch.equal(torch.from_numpy(a.observations), torch.from_numpy(b.observations)) or not torch.equal(
                                torch.from_numpy(a.actions), torch.from_numpy(b.actions)):
                            raise AssertionError("clean/noise input/action mismatch")
                    paired_count += 1
    result = dict(passed=True, identity=identity, prefix_checkpoints=prefix_count, matched_clean_noise_pairs=paired_count,
        before_checkpoints=len(checks), final_checkpoints=len(checks), checks=checks, elapsed_seconds=time.perf_counter()-started,
        scope="Local exact-source state, data, memory and initial/final probe audit. Intermediate weights are not independently retrained.")
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.input, args.output)
