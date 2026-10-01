"""Audit core/residual checkpoint chains, causal data and recoverable forecasts.

This checks complete data/replay provenance and all numerical probe records. It
recovers before/after and first-packet forecasts from retained weights, but does
not retrain the ordinary stream or recover discarded intermediate model states.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from acp_cl.acquisition.study import Evaluator
from acp_cl.acquisition.world import AcquisitionWorld, batch, mask_affected, mask_valid, order, signal_images, stage_law
from acp_cl.conditional.learner import Hypotheses, Packet
from acp_cl.contextual.learner import ContextLearner, HistoryNetwork
from acp_cl.core_residual.design import ARMS, counts, law_from_record, phases, prefix_phases, validate_config
from acp_cl.core_residual.evaluation import array_hash, recompute_metrics
from acp_cl.core_residual.learner import CoreResidualLearner, learner_signature
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest
from acp_cl.persistence.world import concatenate
from acp_cl.predictive_value.mechanism import _value_hash, predict_all
from acp_cl.predictive_value.study import load, read_json
from acp_cl.replay_renewal.memory import ReservoirMemory, memory_state
from acp_cl.replay_renewal.study import affinity, new_learner, write_json


ENCODERS = ("frame.", "temporal.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(left, right, message):
    """Exact tensor/value comparison, permitting JSON tuple/list normalization."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        require(isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.cpu(), right.cpu()), message)
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        require(isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.shape == right.shape and left.dtype == right.dtype
                and np.array_equal(left, right), message)
    elif isinstance(left, dict) and isinstance(right, dict):
        require(left.keys() == right.keys(), message)
        for name in left:
            same(left[name], right[name], f"{message}.{name}")
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        require(len(left) == len(right), message)
        for index, (a, b) in enumerate(zip(left, right)):
            same(a, b, f"{message}.{index}")
    else:
        require(left == right, message)


def history_hash(history):
    return None if history is None else history.fingerprint()


def named_optimizer(learner, prefix=""):
    names = {parameter: name.removeprefix(prefix) for name, parameter in learner.model.named_parameters()
             if name.startswith(prefix)}
    groups = []
    for group in learner.optimizer.param_groups:
        selected = [names[parameter] for parameter in group["params"] if parameter in names]
        if selected:
            groups.append(dict(parameters=selected,
                settings={key: value for key, value in group.items() if key not in ("params", "param_names")}))
    return dict(groups=groups, state={names[parameter]: value for parameter, value in learner.optimizer.state.items()
                                     if parameter in names})


def _finite_tensors(value):
    if isinstance(value, torch.Tensor):
        require(torch.isfinite(value).all().item(), "nonfinite saved tensor")
    elif isinstance(value, dict):
        for child in value.values():
            _finite_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_tensors(child)


def check_optimizer(learner, expected_steps, config):
    require(type(learner.optimizer) is torch.optim.Adam, "optimizer is not ordinary Adam")
    parameters = dict(learner.model.named_parameters())
    registered = [p for group in learner.optimizer.param_groups for p in group["params"]]
    require(len(learner.optimizer.param_groups) == 1 and len(registered) == len(parameters)
            and all(a is b for a, b in zip(registered, parameters.values())),
            "optimizer ownership or parameter order mismatch")
    require(set(learner.optimizer.state) <= set(parameters.values()), "foreign Adam parameter state")
    group = learner.optimizer.param_groups[0]
    require(group["lr"] == config["lr"] and group["betas"] == (.9, .999)
            and group["eps"] == 1e-8 and group["weight_decay"] == 0
            and not group["amsgrad"] and not group["maximize"], "changed Adam hyperparameters")
    if "param_names" in group:
        require(group["param_names"] == list(parameters), "Adam parameter names mismatch")
    _finite_tensors(learner.model.state_dict())
    for name, parameter in parameters.items():
        count = expected_steps[name]
        _finite_tensors(parameter.grad)
        if not count:
            require(parameter not in learner.optimizer.state, "untrained parameter acquired Adam state")
            continue
        require(parameter in learner.optimizer.state, "missing trained Adam parameter state")
        value = learner.optimizer.state[parameter]
        require(set(value) == {"step", "exp_avg", "exp_avg_sq"}, "unexpected Adam state fields")
        require(value["step"].numel() == 1 and float(value["step"]) == count, "Adam step count mismatch")
        for field in ("exp_avg", "exp_avg_sq"):
            require(value[field].shape == parameter.shape and value[field].dtype == parameter.dtype,
                    "Adam moment shape or dtype mismatch")
        _finite_tensors(value)
        require((value["exp_avg_sq"] >= 0).all().item(), "negative Adam second moment")


def check_learning_state(learner, config, ordinary_packets, post_packets=0, prefix_steps=0):
    """Check actual trainability and optimizer counters for each separate path."""
    require(learner.settings == config and type(learner.memory) is ReservoirMemory,
            "learner settings or uniform memory changed")
    require(all(packet.oracle_modes is None for packet in learner.memory.packets), "hidden labels in replay")
    size, updates = config["batch_size"], config["updates_per_batch"]
    expected_cost = dict(arrivals=ordinary_packets * size, optimizer_steps=ordinary_packets * updates,
        query_presentations=2 * ordinary_packets * size * updates,
        support_presentations=2 * ordinary_packets * size * updates,
        replay_presentations=max(0, ordinary_packets - 1) * size * updates,
        duplicate_presentations=min(1, ordinary_packets) * size * updates)
    require(set(learner.cost) == set(expected_cost) | {"training_seconds"}, "cost field coverage mismatch")
    for name, value in expected_cost.items():
        require(type(learner.cost[name]) is int and learner.cost[name] == value, "original learning cost mismatch")
    require(math.isfinite(learner.cost["training_seconds"]) and learner.cost["training_seconds"] >= 0,
            "invalid training duration")
    require(learner.memory.seen == learner.memory.age == ordinary_packets, "reservoir age or count mismatch")
    steps = {}
    if type(learner) is ContextLearner:
        require(post_packets == 0 and all(p.requires_grad for p in learner.model.parameters()),
                "ordinary prefix trainability mismatch")
        steps = {name: ordinary_packets * updates for name, _ in learner.model.named_parameters()}
    else:
        require(type(learner) is CoreResidualLearner and learner.arm in ARMS
                and learner.model.arm == learner.arm, "core/residual arm mismatch")
        for name, parameter in learner.model.named_parameters():
            is_core = name.startswith("core.")
            fixed_encoder = learner.arm == "fixed_features" and name.startswith(
                ("residual.frame.", "residual.temporal."))
            trainable = (not is_core or learner.arm == "joint") and not fixed_encoder
            require(parameter.requires_grad == trainable, "path trainability mismatch")
            steps[name] = (prefix_steps + post_packets * updates if learner.arm == "joint" else prefix_steps) if is_core else (
                0 if fixed_encoder else post_packets * updates)
            if post_packets and not trainable:
                require(parameter.grad is None, "frozen parameter has a gradient after updates")
        work = dict(packets=post_packets, optimizer_steps=post_packets * updates,
            backward_calls=post_packets * updates, logical_probability_calls=2 * post_packets * updates,
            core_forward_calls=2 * post_packets * updates, residual_forward_calls=2 * post_packets * updates,
            core_query_presentations=2 * post_packets * updates * size,
            residual_query_presentations=2 * post_packets * updates * size,
            core_support_presentations=2 * post_packets * updates * size,
            residual_support_presentations=2 * post_packets * updates * size,
            core_backward_paths=2 * post_packets * updates * (learner.arm == "joint"),
            residual_backward_paths=2 * post_packets * updates)
        same(learner.path_work, work, "path work mismatch")
        if learner.arm != "joint":
            same(learner.model.core.state_dict(), learner.core_at_fork, "frozen core weights changed")
            require(_value_hash(named_optimizer(learner, "core.")) == learner.core_optimizer_at_fork_hash,
                    "frozen core Adam changed")
            require(all(not module.training for module in learner.model.core.modules()), "frozen core mode changed")
        if learner.arm == "fixed_features":
            encoder = {name: value for name, value in learner.model.residual.state_dict().items()
                       if name.startswith(ENCODERS)}
            same(encoder, {name: value for name, value in learner.residual_at_fork.items()
                           if name.startswith(ENCODERS)}, "fixed random residual encoder changed")
            require(all(not module.training for parent in (learner.model.residual.frame, learner.model.residual.temporal)
                        for module in parent.modules()), "fixed residual encoder mode changed")
    check_optimizer(learner, steps, config)


def expected_residual(method, seed, config):
    """Rebuild only the declared random initialization, without using fork_core."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 167921)
        if method == "conditional":
            result = Hypotheses(config["residual_width"], config["experts"])
            layers = [head[-1] for head in result.heads]
        else:
            result = HistoryNetwork(config["residual_width"], config["residual_context_width"],
                config["residual_decoder_width"], config["interaction_features"])
            layers = [result.decoder[-1]]
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
    return result


def verify_fork(parent, fork, arm, seed, config):
    """Check copied tensors/moments and independent initialization, not fork code."""
    require(type(parent) is ContextLearner and type(fork) is CoreResidualLearner
            and fork.arm == arm and fork.method == parent.method and fork.device == parent.device,
            "fork architecture or arm mismatch")
    require(fork.residual_seed == seed + 167921 and fork.core_prefix_initial_hash == parent.initial_hash,
            "fork initialization identity mismatch")
    same(fork.model.core.state_dict(), parent.model.state_dict(), "fork core weights differ from learned prefix")
    same(fork.core_at_fork, parent.model.state_dict(), "core measurement snapshot differs from prefix")
    same(named_optimizer(fork, "core."), named_optimizer(parent), "fork did not preserve complete core Adam")
    same(fork.optimizer.defaults, parent.optimizer.defaults, "fork optimizer defaults changed")
    require(fork.core_optimizer_at_fork_hash == _value_hash(named_optimizer(parent)), "core optimizer snapshot hash mismatch")
    for old, new in zip(parent.model.parameters(), fork.model.core.parameters()):
        same(old.grad, new.grad, "fork dropped or changed a core gradient")
        require(old.data_ptr() != new.data_ptr(), "fork shares core parameter storage")
    residual = expected_residual(parent.method, seed, config)
    same(fork.model.residual.state_dict(), residual.state_dict(), "residual random initialization differs")
    same(fork.residual_at_fork, residual.state_dict(), "residual measurement snapshot differs")
    require(not named_optimizer(fork, "residual.")["state"], "residual starts with Adam state")
    require(all(p.grad is None for p in fork.model.residual.parameters()), "residual starts with gradients")
    initial_encoder = {"core." + name: value for name, value in parent.initial_encoder.items()}
    initial_encoder.update({"residual." + name: value for name, value in residual.state_dict().items()
                            if name.startswith(ENCODERS)})
    same(fork.initial_encoder, initial_encoder, "fork initial encoder measurement mismatch")
    require(fork.initial_encoder_hash == state_hash(initial_encoder)
            and fork.initial_hash == state_hash(fork.model.state_dict()), "fork model hashes mismatch")
    same(fork.cost, parent.cost, "fork changed ordinary costs")
    same(memory_state(fork.memory), memory_state(parent.memory), "fork changed replay or RNG state")
    require(history_hash(fork.history) == history_hash(parent.history)
            and fork.peak_memory_bytes == parent.peak_memory_bytes, "fork changed history or peak memory")
    for name, module in fork.model.core.named_modules():
        expected = dict(parent.model.named_modules())[name].training if arm == "joint" else False
        require(module.training == expected, "fork core module mode mismatch")
    check_learning_state(fork, config, parent.memory.seen, prefix_steps=parent.cost["optimizer_steps"])


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate numerical array names")
        return {key: archive[key] for key in archive.files}


def advance_reference(memory, history, data, updates):
    """Independent uniform replay draws and insertion; no learner update occurs."""
    replay = []
    for _ in range(updates):
        replay.append(memory.ids[int(memory.sampling_rng.integers(0, len(memory.ids)))]
                      if memory.ids else memory.seen)
    identifier = memory.seen
    memory.seen += 1
    memory.age += 1
    packet = Packet(history, data)
    if len(memory.ids) < memory.capacity:
        memory.ids.append(identifier)
        memory.packets.append(packet)
    elif memory.capacity:
        slot = int(memory.membership_rng.integers(0, memory.age))
        if slot < memory.capacity:
            memory.ids[slot], memory.packets[slot] = identifier, packet
    return replay


def regenerate_packets(before, after, record, config, seed, phase, arrays):
    world = AcquisitionWorld()
    law = law_from_record(phase["law"])
    size, updates = config["batch_size"], config["updates_per_batch"]
    count = phase["size"] // size
    require(set(arrays) == {"probabilities", "actions", "outcomes"}, "training arrays mismatch")
    require(arrays["probabilities"].shape == (count, size, 5, 3)
            and arrays["actions"].shape == (count, size)
            and arrays["outcomes"].shape == (count, size, 3), "training array dimensions mismatch")
    probability = arrays["probabilities"]
    epsilon = np.finfo(np.float32).eps
    require(probability.dtype == np.float32 and np.isfinite(probability).all()
            and (probability >= -epsilon).all() and (probability <= 1 + epsilon).all(),
            "invalid training predictions")
    require(len(record["packets"]) == len(record["batch_sha256"]) == count
            and record["batches_done"] == count, "packet record coverage mismatch")
    memory, history = copy.deepcopy(before.memory), copy.deepcopy(before.history)
    all_data, marginals = [], [None]
    action_count, success = np.zeros(5, dtype=np.int64), np.zeros((5, 3), dtype=np.int64)
    peak = before.peak_memory_bytes
    before_signature = learner_signature(before)
    for index in range(count):
        data = batch(world, law, seed, phase["channel"], index, size)
        same(arrays["actions"][index], data.actions, "performed training actions differ from regenerated stream")
        same(arrays["outcomes"][index], data.survival, "training feedback differs from physical stream")
        prediction = probability[index]
        brier = float(np.square(prediction[np.arange(size), data.actions].astype(np.float64)
                               - data.survival.astype(np.float64)).mean())
        row = dict(index=index, id=memory.seen, query_sha256=data.fingerprint(),
            support_sha256=history_hash(history), memory_ids_before=list(memory.ids),
            prequential_brier=brier)
        row["replay_ids"] = advance_reference(memory, history, data, updates)
        same(record["packets"][index], row, "packet provenance or error mismatch")
        require(record["batch_sha256"][index] == data.fingerprint(), "training batch fingerprint mismatch")
        if index == 0:
            same(predict_all(before, data.observations, history), prediction, "first prequential forecast mismatch")
        for action in range(5):
            selected = data.actions == action
            action_count[action] += int(selected.sum())
            success[action] += data.survival[selected].sum(axis=0, dtype=np.int64)
        marginals.append((success + .5) / (action_count[:, None] + 1.))
        all_data.append(data)
        history = data
        peak = max(peak, memory.nbytes())
    require(learner_signature(before) == before_signature, "prequential audit mutated before state")
    same(memory_state(memory), memory_state(after.memory), "regenerated ending replay/RNG mismatch")
    require(history_hash(history) == history_hash(after.history) and after.peak_memory_bytes == peak,
            "regenerated ending history or replay-byte peak mismatch")
    same(record["marginal_count"], action_count.tolist(), "marginal counts mismatch")
    same(record["marginal_success"], success.tolist(), "marginal successes mismatch")
    require(record["prequential_brier"] == float(np.mean([row["prequential_brier"] for row in record["packets"]])),
            "phase prequential error mismatch")
    return all_data, marginals


def evaluation_plan(seed, config, phase, before, after, data, marginals):
    """Recreate chronological query/support interventions without using traces."""
    evaluator = Evaluator(seed, config)
    law, cue, branch = law_from_record(phase["law"]), phase["cue"], phase["branch"]
    plan = []

    def add(target, support, learner, selected_cue=None, selected_branch=branch, flipped=False, marginal=None):
        plan.append(dict(law=target, support=support, learner=learner, cue=selected_cue,
                         branch=selected_branch, flipped=flipped, marginal=marginal))

    def probe(target, history, learner, selected_cue, selected_branch, marginal=None):
        for replicate in range(config["support_replicates"]):
            fresh = evaluator.support(target, replicate)
            for feedback in sorted(set((0, config["batch_size"] // 4, config["batch_size"] // 2,
                                        config["batch_size"]))):
                support = history
                if feedback:
                    added = fresh.take(slice(0, feedback))
                    support = added if support is None else concatenate(support, added).take(slice(-config["batch_size"], None))
                add(target, support, learner, selected_cue, selected_branch, marginal=marginal)
            if selected_cue is not None:
                add(target, fresh, learner, selected_cue, selected_branch, flipped=True)

    def valid_panel(target, learner):
        for mode in (0, 1):
            target_mode = replace(target, mode=mode, noise=0.)
            for replicate in range(config["support_replicates"]):
                add(target_mode, evaluator.support(target_mode, replicate, True, branch), learner)

    probe(law, before.history, before, cue, branch)
    valid_panel(law, before)
    add(law, before.history, before, cue)
    for index, experience in enumerate(data, 1):
        if index * config["batch_size"] % config["probe_every"] == 0:
            add(law, experience, after if index == len(data) else None, cue, marginal=marginals[index])
    probe(law, after.history, after, cue, branch, marginals[-1])
    valid_panel(law, after)
    if phase["name"] == "return":
        probe(stage_law(seed, 1), after.history, after, order(seed)[0], "novel")
    return evaluator, plan


def check_evaluations(directory, record, config, seed, phase, before, after, data, marginals):
    metadata = read_json(directory / "evaluations.json")
    arrays = read_arrays(directory / "evaluations.npz")
    evaluator, plan = evaluation_plan(seed, config, phase, before, after, data, marginals)
    require(metadata["seed"] == seed and len(metadata["records"]) == len(plan), "evaluation call coverage mismatch")
    known_keys, predictions = set(), 0
    queries, supports, observed_supports = 0, 0, 0
    before_signatures = (learner_signature(before), learner_signature(after))
    for index, (row, call) in enumerate(zip(metadata["records"], plan)):
        law = asdict(call["law"])
        same(row["law"], law, "evaluation law provenance mismatch")
        require(row["index"] == index and row["cue"] == call["cue"] and row["branch"] == call["branch"]
                and row["flipped"] == call["flipped"], "evaluation call ordering/intervention mismatch")
        support = call["support"]
        require(row["support_fingerprint"] == history_hash(support), "evaluation support is not the declared causal context")
        same(row["marginal"], None if call["marginal"] is None else call["marginal"].tolist(),
             "evaluation marginal includes wrong arrivals")
        cases, truth = evaluator.dataset(call["law"])
        observations = cases.observations
        if call["flipped"]:
            signals = cases.signals.copy()
            signals[:, call["cue"]] ^= 1
            observations = signal_images(observations, signals)
        expected = dict(observations=observations, truth=truth, affected=mask_affected(cases, call["cue"]),
                        valid=mask_valid(cases, call["branch"]))
        if support is not None:
            expected.update(support_observations=support.observations, support_actions=support.actions,
                            support_outcomes=support.survival)
        keys = {name: f"e{index}_{name}" for name in (*expected, "probabilities")}
        require(set(row["array_sha256"]) == set(keys) and set(keys.values()) <= set(arrays),
                "evaluation raw-array coverage mismatch")
        for name, key in keys.items():
            require(array_hash(arrays[key]) == row["array_sha256"][name], "evaluation raw-array hash mismatch")
            if name in expected:
                same(arrays[key], expected[name], "evaluation physics, sensors or support mismatch")
        probabilities = arrays[keys["probabilities"]]
        require(probabilities.shape == (config["eval_size"], 5, 3), "evaluation prediction shape mismatch")
        metrics = recompute_metrics(probabilities, truth, expected["affected"], expected["valid"], call["marginal"])
        same(row["metrics"], metrics, "evaluation arithmetic mismatch")
        require(row["query_sha256"] == array_hash(observations), "evaluation query hash mismatch")
        if call["learner"] is not None:
            require(row["model_sha256"] == state_hash(call["learner"].model.state_dict()),
                    "recoverable evaluation model identity mismatch")
            same(predict_all(call["learner"], observations, support), probabilities,
                 "recoverable evaluation forecast differs from retained model")
            predictions += 1
        query_count, observed_count = len(observations), 0 if support is None else len(support)
        support_count = config["batch_size"] if support is None else observed_count
        require(row["query_presentations"] == query_count and row["support_presentations"] == support_count
                and row["observed_support_presentations"] == observed_count, "evaluation presentation count mismatch")
        queries += query_count
        supports += support_count
        observed_supports += observed_count
        known_keys.update(keys.values())
    require(set(arrays) == known_keys, "unreferenced evaluation arrays")
    expected_counts = dict(calls=len(plan), query_presentations=queries, support_presentations=supports,
                           observed_support_presentations=observed_supports)
    for name, value in expected_counts.items():
        require(metadata[name] == record["evaluation_files"][name] == value, "evaluation total cost mismatch")
    require(before_signatures == (learner_signature(before), learner_signature(after)), "evaluation audit mutated model state")
    return dict(evaluation_calls=len(plan), evaluation_prediction_calls_recomputed=predictions,
                intermediate_evaluation_calls_arithmetic_only=len(plan) - predictions, **{
                    name: value for name, value in expected_counts.items() if name != "calls"})


def check_phase(directory, config, identity, seed, model, arm, phase, previous,
                ordinary_packets, post_packets=0, prefix_steps=0):
    record = read_json(directory / "result.json")
    before_payload, after_payload = (load(directory / name, identity) for name in ("before.pt", "checkpoint.pt"))
    before, after = before_payload["learner"], after_payload["learner"]
    require(record["identity"] == identity and record["seed"] == seed and record["model"] == model
            and record["arm"] == arm and record["phase"] == phase["name"], "phase identity mismatch")
    for name in ("law", "cue", "branch", "size", "channel"):
        same(record[name], phase[name], "phase schedule mismatch")
    require(learner_signature(before) == record["start_signature"] == learner_signature(previous),
            "phase start is not the preceding full learner state")
    require(learner_signature(after) == record["end_signature"], "phase final signature mismatch")
    same(after_payload["record"], record, "final checkpoint record mismatch")
    for name, value in before_payload["record"].items():
        if name not in ("batches_done", "batch_sha256", "packets", "curve", "marginal_count", "marginal_success", "elapsed_seconds"):
            same(value, record[name], "before checkpoint record differs")
    require(before_payload["record"]["batches_done"] == 0 and not before_payload["record"]["packets"],
            "before checkpoint is not before learning")
    for name, digest_value in record["artifact_hashes"].items():
        require(file_hash(directory / name) == digest_value, "phase artifact hash mismatch")
    same(record["start_memory"], memory_state(before.memory), "start replay record mismatch")
    same(record["end_memory"], memory_state(after.memory), "end replay record mismatch")
    same(record["start_diagnostics"], before.diagnostics(), "start diagnostics mismatch")
    same(record["diagnostics"], after.diagnostics(), "end diagnostics mismatch")
    check_learning_state(before, config, ordinary_packets, post_packets, prefix_steps)
    packet_count = phase["size"] // config["batch_size"]
    check_learning_state(after, config, ordinary_packets + packet_count,
                         post_packets + packet_count if type(after) is CoreResidualLearner else 0, prefix_steps)
    require(after.cost["training_seconds"] >= before.cost["training_seconds"], "training time moved backwards")
    same(before.initial_encoder, after.initial_encoder, "initial encoder measurement changed during learning")
    require(before.initial_hash == after.initial_hash and before.initial_encoder_hash == after.initial_encoder_hash,
            "ordinary initialization provenance changed")
    if isinstance(before, CoreResidualLearner):
        for name in ("core_at_fork", "residual_at_fork", "initial_encoder"):
            same(getattr(before, name), getattr(after, name), "measurement snapshot changed during learning")
        for name in ("initial_hash", "initial_encoder_hash", "core_prefix_initial_hash", "residual_seed", "core_optimizer_at_fork_hash"):
            require(getattr(before, name) == getattr(after, name), "initialization provenance changed")
    arrays = read_arrays(directory / "training.npz")
    data, marginals = regenerate_packets(before, after, record, config, seed, phase, arrays)
    evaluation = check_evaluations(directory, record, config, seed, phase, before, after, data, marginals)
    result = dict(phase=phase["name"], arm=arm, packets=packet_count,
        result_sha256=file_hash(directory / "result.json"), before_sha256=file_hash(directory / "before.pt"),
        checkpoint_sha256=file_hash(directory / "checkpoint.pt"), training_sha256=file_hash(directory / "training.npz"),
        evaluations_json_sha256=file_hash(directory / "evaluations.json"),
        evaluations_npz_sha256=file_hash(directory / "evaluations.npz"),
        causal_stream_and_replay_verified=True, exact_full_state_chain_verified=True,
        trainability_adam_and_work_verified=True, first_prequential_prediction_recomputed=True, **evaluation)
    return after, result


def audit_job(directory, config, identity, seed, model):
    """Tensor audit for one job; outer audit enforces cohort/source locks first."""
    directory = Path(directory)
    affinity(config["cpu_affinity"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    job = read_json(directory / "result.json")
    require(job["identity"] == identity and job["seed"] == seed and job["model"] == model, "job identity mismatch")
    learner = new_learner(model, seed, config)
    checked, relative_names, ordinary_packets = [], [], 0
    for phase in prefix_phases(seed, config):
        learner, row = check_phase(directory / phase["name"], config, identity, seed, model, "prefix",
                                   phase, learner, ordinary_packets)
        ordinary_packets += row["packets"]
        checked.append(row)
        relative_names.append(phase["name"])
    require(learner_signature(learner) == job["parent_signature"], "learned prefix signature mismatch")
    parent_signature = learner_signature(learner)
    prefix_steps = ordinary_packets * config["updates_per_batch"]
    insertions, endings = {}, {}
    initial_states = []
    for arm in ARMS:
        schedule = phases(seed, config, arm)
        arm_before = load(directory / arm / schedule[0]["name"] / "before.pt", identity)["learner"]
        verify_fork(learner, arm_before, arm, seed, config)
        require(job["insertion"][arm]["signature"] == learner_signature(arm_before), "insertion signature mismatch")
        same(job["insertion"][arm]["diagnostics"], arm_before.diagnostics(), "insertion diagnostics mismatch")
        initial_states.append(arm_before)
        current, post = arm_before, 0
        for phase in schedule:
            relative = f"{arm}/{phase['name']}"
            current, row = check_phase(directory / relative, config, identity, seed, model, arm, phase,
                                       current, ordinary_packets + post, post, prefix_steps)
            checked.append(row)
            relative_names.append(relative)
            post += row["packets"]
        insertions[arm] = learner_signature(arm_before)
        endings[arm] = learner_signature(current)
    # All learned arms insert precisely the same residual weights and preserve
    # the same copied core moments/replay. Only trainability/module modes vary.
    for candidate in initial_states[1:]:
        same(candidate.model.state_dict(), initial_states[0].model.state_dict(), "learned arms start from different weights")
        same(named_optimizer(candidate), named_optimizer(initial_states[0]), "learned arms start with different Adam state")
        same(memory_state(candidate.memory), memory_state(initial_states[0].memory), "learned arms start with different replay")
    fresh_base = new_learner(model, seed, config)
    maintenance = phases(seed, config, "joint")[0]
    fresh_base.history = batch(AcquisitionWorld(), law_from_record(maintenance["law"]), seed,
        maintenance["channel"], maintenance["size"] // config["batch_size"] - 1, config["batch_size"])
    fresh_phase = phases(seed, config, "fresh")[0]
    fresh = load(directory / "fresh" / fresh_phase["name"] / "before.pt", identity)["learner"]
    verify_fork(fresh_base, fresh, "joint", seed, config)
    fresh, row = check_phase(directory / "fresh" / fresh_phase["name"], config, identity, seed, model,
                             "fresh", fresh_phase, fresh, 0)
    checked.append(row)
    relative_names.append(f"fresh/{fresh_phase['name']}")
    endings["fresh"] = learner_signature(fresh)
    same(job["final_signatures"], endings, "job final learner signatures mismatch")
    require(learner_signature(learner) == parent_signature, "audit changed learned prefix")
    require(set(job["phase_hashes"]) == set(job["checkpoint_hashes"]) == set(relative_names), "job phase coverage mismatch")
    for relative in relative_names:
        require(job["phase_hashes"][relative] == file_hash(directory / relative / "result.json")
                and job["checkpoint_hashes"][relative] == file_hash(directory / relative / "checkpoint.pt"),
                "job artifact hashes mismatch")
    return dict(seed=seed, model=model, phases=checked, forks_verified=len(ARMS), fresh_initializations_verified=1,
        before_checkpoints=len(checked), final_checkpoints=len(checked),
        ordinary_packets=sum(row["packets"] for row in checked),
        first_prequential_predictions_recomputed=len(checked),
        evaluation_calls=sum(row["evaluation_calls"] for row in checked),
        evaluation_prediction_calls_recomputed=sum(row["evaluation_prediction_calls_recomputed"] for row in checked),
        intermediate_evaluation_calls_arithmetic_only=sum(row["intermediate_evaluation_calls_arithmetic_only"] for row in checked),
        query_presentations=sum(row["query_presentations"] for row in checked),
        support_presentations=sum(row["support_presentations"] for row in checked),
        observed_support_presentations=sum(row["observed_support_presentations"] for row in checked),
        result_sha256=file_hash(directory / "result.json"), parent_signature=parent_signature, insertion=insertions)


def audit(directory, output, workers=None):
    from summarize_core_residual import audit_records
    from acp_cl.core_residual.study import check_locks, source_manifest

    started = time.perf_counter()
    directory = Path(directory)
    manifest, records, hashes = audit_records(directory)
    config, identity = manifest["config"], manifest["identity"]
    validate_config(config)
    check_locks(directory, verify_live=True)
    require(digest(source_manifest()) == identity["source_sha256"], "audit source differs from locked source")
    analysis = read_json(directory / "analysis_lock.json")
    auditor_name = "scripts/audit_core_residual.py"
    require(analysis["files"].get(auditor_name) == file_hash(__file__), "tensor auditor is not the locked analysis")
    declared = counts(config)
    require(len(records) == declared["total_phases"], "incomplete phase cohort")
    jobs = [(directory / f"{model}_{seed}", config, identity, seed, model)
            for model in config["models"] for seed in config["seeds"]]
    actual_workers = config["workers"] if workers is None else workers
    require(type(actual_workers) is int and actual_workers > 0, "positive audit worker count required")
    if actual_workers == 1:
        checks = [audit_job(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = [pool.submit(audit_job, *job) for job in jobs]
            checks = [future.result() for future in as_completed(futures)]
    checks.sort(key=lambda row: (row["model"], row["seed"]))
    totals = {name: sum(row[name] for row in checks) for name in (
        "forks_verified", "fresh_initializations_verified", "before_checkpoints", "final_checkpoints",
        "ordinary_packets", "first_prequential_predictions_recomputed", "evaluation_calls",
        "evaluation_prediction_calls_recomputed", "intermediate_evaluation_calls_arithmetic_only",
        "query_presentations", "support_presentations", "observed_support_presentations")}
    require(totals["before_checkpoints"] == totals["final_checkpoints"] == declared["total_phases"]
            and totals["ordinary_packets"] == declared["training_packets"], "tensor audit cohort coverage mismatch")
    result = dict(passed=True, identity=identity, jobs=len(jobs), **totals,
        manifest_sha256=file_hash(directory / "manifest.json"), audit_source_sha256=file_hash(__file__),
        result_hashes=hashes, checks=checks, elapsed_seconds=time.perf_counter() - started,
        scope="Exact source/analysis/protocol/config locks and complete cohort. Regenerated ordinary physical streams, "
              "uniform replay draws, histories and marginal counts; checkpoint chains, full Adam ownership and counters, "
              "independently checked identical forks, frozen cores and fixed random residual encoders; all probe physics, "
              "causal supports and numerical metrics. Before/after, endpoint and first prequential forecasts are recomputed "
              "from retained weights. Other intermediate forecasts are checked against raw arrays and provenance only. "
              "No full ordinary retraining, discarded-weight reconstruction or historical wall-clock creation-order proof.")
    write_json(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    result = audit(args.input, args.output, args.workers)
    print({key: value for key, value in result.items() if key not in ("checks", "result_hashes")})
