"""Audit exact forks, causal replay, work counters and saved endpoint predictions."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from audit_conditional import compare
from audit_training_state import advance_memory, tensor_tree
from acp_cl.acquisition.study import Evaluator, marginal, marginal_update
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch, order, stage_law
from acp_cl.conditional.learner import Packet
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest
from acp_cl.replay_renewal.memory import make_memory, memory_state
from acp_cl.replay_renewal.study import affinity, new_learner, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(learner):
    """Learning state, including experimental counters; excludes module eval flags."""
    return digest(dict(
        model=state_hash(learner.model.state_dict()),
        optimizer=tensor_tree(learner.optimizer.state_dict()),
        trainable=[p.requires_grad for p in learner.model.parameters()],
        method=learner.method,
        settings=learner.settings,
        memory=memory_state(learner.memory),
        history=None if learner.history is None else learner.history.fingerprint(),
        diagnostics=learner.diagnostics(),
    ))


def check_optimizer(learner, expected_steps, config):
    """Verify actual Adam state, not merely the copied cost counter."""
    require(isinstance(learner.optimizer, torch.optim.Adam), "optimizer is not Adam")
    parameters = list(learner.model.parameters())
    require(all(p.requires_grad for p in parameters), "unexpected frozen parameter")
    registered = [p for group in learner.optimizer.param_groups for p in group["params"]]
    require(len(registered) == len(parameters)
            and all(a is b for a, b in zip(registered, parameters)),
            "optimizer parameter ownership mismatch")
    require(all(torch.isfinite(p).all().item() for p in learner.model.state_dict().values()),
            "nonfinite model state")
    for group in learner.optimizer.param_groups:
        require(group["lr"] == config["lr"] and group["betas"] == (.9, .999)
                and group["eps"] == 1e-8 and group["weight_decay"] == 0
                and not group["amsgrad"] and not group["maximize"],
                "changed optimizer hyperparameters")
    if not expected_steps:
        require(not learner.optimizer.state, "fresh optimizer contains carried state")
        return
    require(set(learner.optimizer.state) == set(parameters), "incomplete Adam state")
    for parameter, state in learner.optimizer.state.items():
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "unexpected Adam state")
        require(float(state["step"]) == expected_steps, "Adam step counter mismatch")
        for name in ("exp_avg", "exp_avg_sq"):
            require(state[name].shape == parameter.shape
                    and torch.isfinite(state[name]).all().item(), "invalid Adam tensor")
        require((state["exp_avg_sq"] >= 0).all().item(), "negative Adam second moment")


def check_original_cost(learner, batches, config):
    diagnostics = learner.diagnostics()
    size, updates = config["batch_size"], config["updates_per_batch"]
    expected = dict(arrivals=batches*size, optimizer_steps=batches*updates,
        query_presentations=2*batches*size*updates,
        support_presentations=2*batches*size*updates,
        replay_presentations=max(0, batches-1)*size*updates,
        duplicate_presentations=min(1, batches)*size*updates)
    require(all(type(diagnostics["cost"][k]) is int and diagnostics["cost"][k] == v
                for k, v in expected.items()), "original training cost mismatch")
    require(math.isfinite(diagnostics["cost"]["training_seconds"])
            and diagnostics["cost"]["training_seconds"] >= 0, "invalid training duration")
    require(diagnostics["packets_seen"] == batches, "lifetime packet counter mismatch")
    require(all(packet.oracle_modes is None for packet in learner.memory.packets),
            "hidden labels supplied to replay")
    check_optimizer(learner, batches*updates, config)


def check_extra_work(learner, arm, batches, config):
    """Validate saved diagnostics; individual intermediate steps are not re-executed."""
    internal_arm = "reference" if arm == "fresh" else arm
    selective = learner.diagnostics()["selective_updates"]
    require(selective["arm"] == internal_arm, "wrong selective update arm")
    weight = .5 if internal_arm in ("reference", "protected") else .75
    require(selective["current_weight"] == weight, "wrong current loss weight")
    mode = ("none" if internal_arm in ("reference", "current") else
            "shrink" if internal_arm == "shrink" else "project")
    require(selective["mode"] == mode, "wrong displacement mode")
    stats = selective["stats"]
    size, updates = config["batch_size"], config["updates_per_batch"]
    steps = batches*updates
    expected = dict(steps=steps, packets=batches, training_probability_calls=2*steps,
        reference_gradient_calls=steps, objective_backward_calls=steps,
        reference_backward_query_presentations=steps*size,
        reference_backward_support_presentations=steps*size,
        diagnostic_probability_calls=batches, diagnostic_query_presentations=batches*size,
        diagnostic_support_presentations=batches*size)
    require(all(type(stats[k]) is int and stats[k] == value for k, value in expected.items()),
            "selective extra-work budget mismatch")
    for name in ("projection_eligible_steps", "projection_applied_steps", "shrink_applied_steps",
                 "rewrite_steps", "zero_reference_steps", "zero_displacement_steps"):
        require(type(stats[name]) is int and 0 <= stats[name] <= steps,
                f"invalid selective counter: {name}")
    require(type(stats["finite_step_increases"]) is int
            and 0 <= stats["finite_step_increases"] <= batches,
            "invalid finite-step increase counter")
    for name in ("max_positive_projected_residual", "max_positive_applied_residual"):
        require(math.isfinite(stats[name]) and stats[name] >= 0, "invalid residual diagnostic")
    if internal_arm in ("reference", "current"):
        require(stats["projection_applied_steps"] == stats["shrink_applied_steps"]
                == stats["rewrite_steps"] == 0, "unprojected arm rewrote Adam displacement")
    elif internal_arm in ("protected", "combined"):
        require(stats["shrink_applied_steps"] == 0
                and stats["rewrite_steps"] == stats["projection_eligible_steps"]
                and stats["projection_applied_steps"] <= stats["rewrite_steps"],
                "projected-arm intervention counter mismatch")
    else:
        require(stats["projection_applied_steps"] == 0
                and stats["rewrite_steps"] == stats["projection_eligible_steps"]
                and stats["shrink_applied_steps"] <= stats["rewrite_steps"],
                "shrink-arm intervention counter mismatch")
    require(stats["projection_applied_steps"] <= stats["projection_eligible_steps"],
            "projection applied without an eligible proposal")
    step_summaries = ("hd_before", "hd_projected", "hd_after", "removed_norm_ratio",
                      "retained_norm_ratio", "applied_removed_norm_ratio", "sampled_age_packets")
    packet_summaries = ("finite_replay_loss_before", "finite_replay_loss_after",
                        "finite_replay_loss_change")
    require(set(stats["summaries"]) == set(step_summaries+packet_summaries),
            "wrong selective summary set")
    for name, aggregate in stats["summaries"].items():
        count = aggregate["count"]
        require(type(count) is int and count == (steps if name in step_summaries else batches),
                f"invalid aggregate count: {name}")
        if count == 0:
            require(aggregate["sum"] == 0 and aggregate["min"] is None
                    and aggregate["max"] is None, "nonempty zero-count aggregate")
        else:
            total, low, high = (aggregate[key] for key in ("sum", "min", "max"))
            require(all(math.isfinite(value) for value in (total, low, high)) and low <= high,
                    "invalid selective aggregate")
            tolerance = 1e-9*max(1., abs(total), count*abs(low), count*abs(high))
            require(count*low-tolerance <= total <= count*high+tolerance,
                    "aggregate sum outside observed range")
    return expected


def replay_age_summary(memory, history, observed, config):
    """Regenerate the actual uniform draws without touching any learned weights."""
    memory = copy.deepcopy(memory)
    ages = []
    for current in observed:
        for _ in range(config["updates_per_batch"]):
            selected = memory.sample()
            if selected is None:
                ages.append(0)
            else:
                index = next(i for i, stored in enumerate(memory.packets) if stored is selected)
                ages.append(memory.seen-1-memory.ids[index])
        memory.add(Packet(history, current))
        history = current
    return dict(count=len(ages), sum=float(sum(ages)),
                min=float(min(ages)) if ages else None, max=float(max(ages)) if ages else None)


def audit(directory, output):
    # Lazy imports keep low-level integrity checks usable independently of the runner.
    from summarize_selective_updates import audit_records
    from acp_cl.selective_updates.learner import ARMS, fork
    from acp_cl.selective_updates.study import source_manifest

    started = time.perf_counter()
    directory = Path(directory)
    manifest, records, _ = audit_records(directory)
    config, identity = manifest["config"], manifest["identity"]
    require(digest(source_manifest()) == identity["source_sha256"],
            "audit requires the exact study source")
    affinity(config["cpu_affinity"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    world, checks, prefix_checks = AcquisitionWorld(), [], []
    prefix_batches = config["prefix_blocks"]*config["prefix_size"]//config["batch_size"]
    episode_batches = config["episode_size"]//config["batch_size"]

    def load(path):
        saved = torch.load(path, map_location=manifest["runtime"]["device"], weights_only=False)
        require(saved["identity"] == identity, "checkpoint identity mismatch")
        return saved["learner"], saved["record"]

    for seed in config["seeds"]:
        data_by_arm = {}
        for model in config["models"]:
            folder, evaluator = directory/f"{model}_{seed}", Evaluator(seed, config)
            prefix, prefix_record = load(folder/"prefix.pt")
            require(prefix.method == model, "prefix model type mismatch")
            require(prefix_record["identity"] == identity and prefix_record["seed"] == seed
                    and prefix_record["model"] == model and prefix_record["policy"] == "uniform",
                    "wrong prefix record identity")
            require(len(prefix_record["blocks"]) == config["prefix_blocks"], "incomplete prefix")
            require(digest(json.loads((folder/"prefix.json").read_text(encoding="utf-8")))
                    == digest(prefix_record), "prefix checkpoint/JSON mismatch")
            require("selective_updates" not in prefix.diagnostics(),
                    "shared prefix used selective update mechanism")
            observed_prefix = []
            for index, saved in enumerate(prefix_record["blocks"]):
                law = Law((seed+index) % 2)
                observed = [batch(world, law, seed, f"prefix_{index}", i, config["batch_size"])
                            for i in range(config["prefix_size"]//config["batch_size"])]
                require(saved["batch_sha256"] == [item.fingerprint() for item in observed]
                        and digest(saved["law"]) == digest(asdict(law)),
                        "prefix data/condition mismatch")
                observed_prefix.extend(observed)
            memory, history = advance_memory(make_memory("uniform", config, seed), None,
                                             observed_prefix, config)
            require(memory_state(memory) == memory_state(prefix.memory)
                    and history.fingerprint() == prefix.history.fingerprint(),
                    "prefix replay/history is not causal")
            compare(memory_state(prefix.memory), prefix_record["memory"])
            compare(prefix.diagnostics(), prefix_record["blocks"][-1]["diagnostics"])
            check_original_cost(prefix, prefix_batches, config)
            before_probe = signature(prefix)
            compare(evaluator.evaluate(prefix, law, prefix.history),
                    prefix_record["blocks"][-1]["endpoint"])
            require(signature(prefix) == before_probe, "prefix evaluation changed learning state")
            prefix_checks.append(dict(seed=seed, model=model, causal_data_replay_verified=True,
                final_prefix_probe_recomputed=True, optimizer_steps=prefix_batches*config["updates_per_batch"]))
            carried_end_memories = []
            for arm in (*ARMS, "fresh"):
                key, episode_folder = (seed, model, 1, arm, "novel"), folder/f"stage_1_{arm}"
                record = records[key]
                before, start_record = load(episode_folder/"before.pt")
                after, final_record = load(episode_folder/"checkpoint.pt")
                require(digest(final_record) == digest(record), "checkpoint/result mismatch")
                if arm == "fresh":
                    expected = fork(new_learner(model, seed, config, str(before.device)), "reference")
                    expected.history = copy.deepcopy(prefix.history)
                    base_batches = 0
                else:
                    expected, base_batches = fork(prefix, arm), prefix_batches
                require(signature(before) == signature(expected),
                        "wrong model/optimizer/memory/history/counters at episode start")
                require(before.method == after.method == model, "episode model type mismatch")
                compare(before.diagnostics(), record["start_diagnostics"])
                compare(memory_state(before.memory), record["start_memory"])
                check_original_cost(before, base_batches, config)
                check_extra_work(before, arm, 0, config)
                for field in ("identity", "seed", "model", "stage", "arm", "branch", "law",
                              "cue", "channel", "before", "valid_before", "start_diagnostics",
                              "start_memory"):
                    compare(start_record[field], record[field])
                require(start_record["batches_done"] == 0 and start_record["batch_sha256"] == []
                        and start_record["curve"] == record["curve"][:1],
                        "before checkpoint contains later training")
                law = stage_law(seed, 1)
                require(record["cue"] == order(seed)[0]
                        and digest(record["law"]) == digest(asdict(law))
                        and record["channel"] == "stage_1", "wrong first-introduction condition")
                observed = [batch(world, law, seed, "stage_1", i, config["batch_size"])
                            for i in range(episode_batches)]
                packet_hashes = [item.fingerprint() for item in observed]
                require(record["batch_sha256"] == packet_hashes, "episode data do not regenerate")
                data_by_arm[model, arm] = packet_hashes
                reconstructed = dict(marginal_count=[0]*5, marginal_success=[[0]*3 for _ in range(5)])
                for item in observed:
                    marginal_update(reconstructed, item)
                compare(reconstructed["marginal_count"], record["marginal_count"])
                compare(reconstructed["marginal_success"], record["marginal_success"])
                memory, history = advance_memory(before.memory, before.history, observed, config)
                require(memory_state(memory) == memory_state(after.memory)
                        and history.fingerprint() == after.history.fingerprint(),
                        "episode replay/history is not causal")
                compare(memory_state(after.memory), record["end_memory"])
                compare(after.diagnostics(), record["diagnostics"])
                check_original_cost(after, base_batches+episode_batches, config)
                work = check_extra_work(after, arm, episode_batches, config)
                compare(replay_age_summary(before.memory, before.history, observed, config),
                        after.diagnostics()["selective_updates"]["stats"]["summaries"]["sampled_age_packets"])
                if arm != "fresh":
                    carried_end_memories.append(memory_state(after.memory))
                initial_state, final_state = signature(before), signature(after)
                compare(evaluator.probe(before, law, record["cue"], "novel"), record["before"])
                compare(evaluator.valid_panel(before, law, "novel"), record["valid_before"])
                compare(evaluator.probe(after, law, record["cue"], "novel", marginal=marginal(record)),
                        record["after"])
                compare(evaluator.valid_panel(after, law, "novel"), record["valid_after"])
                compare(evaluator.evaluate(before, law, before.history, record["cue"], "novel"),
                        record["curve"][0]["metrics"])
                compare(evaluator.evaluate(after, law, after.history, record["cue"], "novel",
                                           marginal=marginal(record)), record["curve"][-1]["metrics"])
                require(signature(before) == initial_state and signature(after) == final_state,
                        "evaluation changed learning state")
                checks.append(dict(seed=seed, model=model, stage=1, arm=arm,
                    exact_initial_state_verified=True, causal_data_replay_verified=True,
                    optimizer_state_and_counters_verified=True, extra_work_budget_verified=work,
                    endpoint_probes_recomputed=True))
            require(len({digest(value) for value in carried_end_memories}) == 1,
                    "different replay membership or random state across carried arms")
            print(f"audited {model} seed={seed}: {len(checks)} episodes", flush=True)
        require(len({tuple(value) for value in data_by_arm.values()}) == 1,
                "different arrival data across architectures/arms")
    result = dict(passed=True, identity=identity, prefix_checkpoints=len(prefix_checks),
        before_checkpoints=len(checks), final_checkpoints=len(checks),
        matched_carried_arm_groups=len(prefix_checks), prefix_checks=prefix_checks, checks=checks,
        elapsed_seconds=time.perf_counter()-started,
        scope=("Local exact-source lock, checkpoint, fork, causal-data/replay, Adam/work-counter "
               "and initial/final query/valid-panel audit. Intermediate model weights, gradient "
               "geometry and individual update losses are not independently retrained or rescored. "
               "Saved diagnostic aggregates are checked for consistency, not independent replication."))
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    audit(arguments.input, arguments.output)
