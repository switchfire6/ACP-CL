"""Independently reconstruct rehearsal-state hybrids and paired predictions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_conditional import compare
from audit_predictive_value import (
    _anchor_hash, _history_hash, _owned_tensor_pointers, _tensor_bytes,
    check_bundle, checked_load, check_packet_row, check_shadows, file_hash,
    read_json, recompute_prediction, reproduce_shadow, require, verify_choice,
)
from audit_selective_updates import check_optimizer, check_original_cost
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch, stage_law
from acp_cl.conditional.learner import Packet
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest, trial_seed
from acp_cl.predictive_value.design import law_from_record
from acp_cl.predictive_value.mechanism import _value_hash, packet_fingerprint, predict_all, state_signature
from acp_cl.rehearsal_state.design import counts, schedule, validate_config
from acp_cl.replay_renewal.memory import make_memory, memory_state
from acp_cl.replay_renewal.study import affinity, new_learner, write_json


CELLS = tuple(f"{index:03b}" for index in range(8))
BRANCHES = ("current", "return")


def active_tree(learner):
    """Named active tensors, independent of serialized optimizer parameter IDs."""
    names = {parameter:name for name, parameter in learner.model.named_parameters()}
    return dict(model=learner.model.state_dict(),
        gradients={name:parameter.grad for name, parameter in learner.model.named_parameters()},
        trainable={name:parameter.requires_grad for name, parameter in learner.model.named_parameters()},
        module_modes={name:module.training for name, module in learner.model.named_modules()},
        optimizer_state={names[parameter]:state for parameter, state in learner.optimizer.state.items()},
        optimizer_groups=[{**{key:value for key, value in group.items() if key != "params"},
                           "parameter_names":[names[parameter] for parameter in group["params"]]}
                          for group in learner.optimizer.param_groups])


def active_hash(learner):
    return _value_hash(active_tree(learner))


def transplant_independent(old, new, cell):
    """Copy old inactive state, then assign weights and complete named Adam state.

Only the three declared factors may vary. Gradients are copied from the weight
donor; the first rehearsal update clears them before backward evaluation.
"""
    require(cell in CELLS, "unknown W/O/A cell")
    require(old.method == new.method and old.settings == new.settings
            and old.device == new.device, "incompatible snapshot learners")
    weight_donor, optimizer_donor = (old, new)[int(cell[0])], (old, new)[int(cell[1])]
    result = copy.deepcopy(old)
    target_parameters = dict(result.model.named_parameters())
    weight_parameters = dict(weight_donor.model.named_parameters())
    optimizer_parameters = dict(optimizer_donor.model.named_parameters())
    require(target_parameters.keys() == weight_parameters.keys() == optimizer_parameters.keys(),
            "parameter names differ across snapshots")
    result.model.load_state_dict(weight_donor.model.state_dict(), strict=True)
    donor_modes = dict(weight_donor.model.named_modules())
    for name, module in result.model.named_modules():
        module.training = donor_modes[name].training
    for name, parameter in target_parameters.items():
        source = weight_parameters[name]
        require(parameter.shape == source.shape and parameter.dtype == source.dtype,
                "weight shape/dtype mismatch")
        parameter.requires_grad_(source.requires_grad)
        parameter.grad = None if source.grad is None else source.grad.detach().clone()
    require(isinstance(result.optimizer, torch.optim.Adam)
            and isinstance(optimizer_donor.optimizer, torch.optim.Adam), "Adam required")
    require(len(result.optimizer.param_groups) == len(optimizer_donor.optimizer.param_groups),
            "optimizer group coverage mismatch")
    donor_names = {parameter:name for name, parameter in optimizer_parameters.items()}
    result.optimizer.state.clear()
    for group, donor_group in zip(result.optimizer.param_groups, optimizer_donor.optimizer.param_groups):
        names = [donor_names[parameter] for parameter in donor_group["params"]]
        group.clear()
        group.update(copy.deepcopy({key:value for key, value in donor_group.items() if key != "params"}))
        group["params"] = [target_parameters[name] for name in names]
    result.optimizer.defaults = copy.deepcopy(optimizer_donor.optimizer.defaults)
    for name, source in optimizer_parameters.items():
        if source in optimizer_donor.optimizer.state:
            result.optimizer.state[target_parameters[name]] = copy.deepcopy(optimizer_donor.optimizer.state[source])
    return result


def reproduce_factor(old, new, old_anchor, new_anchor, replay, cell, updates):
    """Apply the prescribed Adam loop without calling the experimental mechanism."""
    require(type(updates) is int and updates > 0, "positive rehearsal dose required")
    old_state, new_state = state_signature(old), state_signature(new)
    source_packets = [packet_fingerprint(packet) for packet in (old_anchor, new_anchor, replay)]
    learner = transplant_independent(old, new, cell)
    anchor = (old_anchor, new_anchor)[int(cell[2])]
    for _ in range(updates):
        learner.model.train()
        losses = []
        for packet in (anchor, replay):
            require(packet.oracle_modes is None, "hidden labels in rehearsal")
            probabilities = learner.probabilities(packet.query.observations, packet.support)[0]
            actions = torch.as_tensor(packet.query.actions.astype(np.int64), device=learner.device)
            outcomes = torch.as_tensor(packet.query.survival.astype(np.float32), device=learner.device)
            predicted = probabilities[torch.arange(len(actions), device=learner.device), actions]
            losses.append(F.binary_cross_entropy(predicted.clamp(1e-6, 1-1e-6), outcomes))
        learner.optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(learner.model.parameters(), 5.)
        learner.optimizer.step()
    require(state_signature(old) == old_state and state_signature(new) == new_state,
            "audit factor reconstruction mutated a donor")
    require([packet_fingerprint(packet) for packet in (old_anchor, new_anchor, replay)] == source_packets,
            "audit factor reconstruction mutated causal packets")
    return learner


def recorded_active_hash(learner):
    """Recreate the recorded serialization convention without its implementation."""
    return _value_hash(dict(model=learner.model.state_dict(), optimizer=learner.optimizer.state_dict(),
        gradients={name:parameter.grad for name, parameter in learner.model.named_parameters()},
        trainable={name:parameter.requires_grad for name, parameter in learner.model.named_parameters()},
        module_modes={name:module.training for name, module in learner.model.named_modules()}))


def inactive_hash(learner):
    return _value_hash(dict(method=learner.method, device=str(learner.device), settings=learner.settings,
        memory=memory_state(learner.memory), history=_history_hash(learner.history),
        cost=learner.cost, peak_memory_bytes=learner.peak_memory_bytes,
        initial_hash=learner.initial_hash, initial_encoder=learner.initial_encoder,
        initial_encoder_hash=learner.initial_encoder_hash))


def loss_pair(learner, anchor, replay):
    """Independently rescore clamped BCE with no mode or state mutation."""
    signature = state_signature(learner)
    modes = [(module, module.training) for module in learner.model.modules()]
    result = []
    try:
        learner.model.eval()
        with torch.no_grad():
            for packet in (anchor, replay):
                probabilities = learner.probabilities(packet.query.observations, packet.support)[0]
                actions = torch.as_tensor(packet.query.actions.astype(np.int64), device=learner.device)
                labels = torch.as_tensor(packet.query.survival.astype(np.float32), device=learner.device)
                performed = probabilities[torch.arange(len(actions), device=learner.device), actions]
                result.append(float(F.binary_cross_entropy(performed.clamp(1e-6, 1-1e-6), labels)))
    finally:
        for module, training in modes:
            module.training = training
    require(state_signature(learner) == signature, "audit loss instrumentation changed state")
    require(all(math.isfinite(value) for value in result), "nonfinite diagnostic BCE")
    return result


def expected_factor_work(old, new, initial, saved, anchor, packet, cell, config):
    updates, size = config["rehearsal_updates"], config["batch_size"]
    donors = (state_signature(old), state_signature(new))
    before, after = loss_pair(initial, anchor, packet), loss_pair(saved, anchor, packet)
    squared = 0.
    for left, right in zip(initial.model.parameters(), saved.model.parameters()):
        squared += float((right.detach().double()-left.detach().double()).square().sum())
    work = dict(steps=updates, forwards=2*updates, backward_calls=updates,
        query_presentations=2*size*updates, support_presentations=2*size*updates,
        model_bytes=_tensor_bytes(saved.model.state_dict()),
        optimizer_bytes=_tensor_bytes(saved.optimizer.state_dict()),
        gradient_bytes=sum(_tensor_bytes(p.grad) for p in saved.model.parameters()),
        initial_encoder_bytes=_tensor_bytes(saved.initial_encoder),
        replay_bytes=old.memory.nbytes(),
        history_bytes=0 if old.history is None else Packet(None, old.history).nbytes(),
        anchor_packet_sha256=packet_fingerprint(anchor), replay_packet_sha256=packet_fingerprint(packet),
        initial_model_sha256=state_hash(initial.model.state_dict()),
        final_model_sha256=state_hash(saved.model.state_dict()),
        initial_optimizer_sha256=_value_hash(initial.optimizer.state_dict()),
        final_optimizer_sha256=_value_hash(saved.optimizer.state_dict()),
        parent_state_sha256=state_signature(initial), shadow_state_sha256=state_signature(saved),
        parent_unchanged=True, memory_history_unchanged=True,
        cell=cell, weights_bit=int(cell[0]), optimizer_bit=int(cell[1]), anchor_bit=int(cell[2]),
        old_donor_sha256=donors[0], new_donor_sha256=donors[1],
        weights_donor_sha256=donors[int(cell[0])], optimizer_donor_sha256=donors[int(cell[1])],
        inactive_donor_sha256=donors[0], canonical_inactive_sha256=inactive_hash(old),
        initial_active_sha256=recorded_active_hash(initial), final_active_sha256=recorded_active_hash(saved),
        update_l2=math.sqrt(squared), update_norm_dtype="float64", diagnostic_forwards=4,
        diagnostic_query_presentations=4*size, diagnostic_support_presentations=4*size,
        losses=dict(anchor_before=before[0], replay_before=before[1], anchor_after=after[0], replay_after=after[1]),
        donors_unchanged=True, inactive_unchanged=True)
    work["explicit_tensor_bytes_subtotal"] = sum(work[name] for name in (
        "model_bytes", "optimizer_bytes", "gradient_bytes", "initial_encoder_bytes", "replay_bytes", "history_bytes"))
    work["explicit_tensor_bytes_scope"] = (
        "Sum of explicit model, optimizer, gradient, initial-encoder, replay and history storage. "
        "Replay/history aliases may be conservatively double counted. Excludes autograd graphs, "
        "operator workspaces, Python objects and allocator overhead; not measured peak RAM.")
    return work


def check_factor_cell(payload, old, new, old_anchor, new_anchor, bundle, cell, config):
    """Independently reproduce every saved factor fork, including all-new equality."""
    require(cell in CELLS[1:], "000 must reuse the independently audited score forks")
    parents = dict(old=state_signature(old), new=state_signature(new))
    require(payload["parents"] == parents and payload["cell"] == cell
            and payload["ids"] == bundle["ids"]
            and payload["anchor_old_sha256"] == packet_fingerprint(old_anchor)
            and payload["anchor_new_sha256"] == packet_fingerprint(new_anchor), "factor provenance mismatch")
    learners, work = payload["learners"], payload["work"]
    require(len(learners) == len(work) == len(bundle["packets"])
            == len(payload["signatures"]) == len(payload["active_signatures"]), "factor coverage mismatch")
    seen = _owned_tensor_pointers(old)|_owned_tensor_pointers(new)
    for learner, counters, replay, signature, active in zip(learners, work, bundle["packets"],
                                                           payload["signatures"], payload["active_signatures"]):
        pointers = _owned_tensor_pointers(learner)
        require(pointers.isdisjoint(seen), "factor tensor ownership overlaps")
        seen.update(pointers)
        donor = (old, new)[int(cell[1])]
        check_optimizer(learner, donor.cost["optimizer_steps"]+config["rehearsal_updates"], config)
        initial = transplant_independent(old, new, cell)
        expected = reproduce_factor(old, new, old_anchor, new_anchor, replay, cell, config["rehearsal_updates"])
        require(state_signature(learner) == signature == state_signature(expected),
                "factor state differs from independently reproduced rehearsal")
        require(active_hash(learner) == active_hash(expected) and active == recorded_active_hash(learner),
                "factor active state signature mismatch")
        require(inactive_hash(learner) == inactive_hash(old), "factor changed canonical inactive state")
        if cell == "111":
            endpoint = reproduce_shadow(new, new_anchor, replay, config["rehearsal_updates"])
            require(active_hash(learner) == active_hash(endpoint), "all-new active endpoint mismatch")
        anchor = (old_anchor, new_anchor)[int(cell[2])]
        expected_work = expected_factor_work(old, new, initial, learner, anchor, replay, cell, config)
        text_fields = {"interpretation", "factor_scope"}
        times = {"training_seconds", "transfer_seconds", "diagnostic_seconds"}
        require(set(counters) == set(expected_work)|times|text_fields, "factor work field coverage mismatch")
        for name, value in expected_work.items():
            compare(counters[name], value, "factor work."+name)
        for name in times:
            require(math.isfinite(counters[name]) and counters[name] >= 0, "invalid factor elapsed time")
        require(all(isinstance(counters[name], str) and counters[name] for name in text_fields),
                "missing factor interpretation")
    require(parents == dict(old=state_signature(old), new=state_signature(new)), "factor audit mutated donors")
    return len(learners)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def regenerate_streams(seed, index, config, history):
    """Couple exogenous draws; only the law and subsequent causal feedback differ."""
    require(history is not None, "missing actual boundary history")
    world = AcquisitionWorld()
    histories = [history, history]
    streams, rows, observations, actions, outcomes, truth = [], [], [], [], [], []
    for packet in range(config["validation_size"]//config["batch_size"]):
        draw = trial_seed(seed, f"acquisition_predictive_value_validation_{index}", packet)
        data, physical = [], []
        for law in (stage_law(seed, 1), Law(seed % 2)):
            experience = world.experience(law, config["batch_size"], draw)
            cases = world.dataset(law, config["batch_size"], draw)
            values = world.counterfactuals(cases)
            require(np.array_equal(experience.observations, cases.observations)
                    and np.array_equal(experience.survival, values[np.arange(len(experience)), experience.actions]),
                    "paired feedback differs from physical truth")
            data.append(experience)
            physical.append(values)
        require(np.array_equal(data[0].observations, data[1].observations)
                and np.array_equal(data[0].actions, data[1].actions), "uncoupled branch observations/actions")
        streams.append([(data[b], histories[b]) for b in range(2)])
        rows.append({branch:dict(query_sha256=data[b].fingerprint(), support_sha256=histories[b].fingerprint())
                     for b, branch in enumerate(BRANCHES)})
        observations.append(data[0].observations)
        actions.append(data[0].actions)
        outcomes.append([value.survival for value in data])
        truth.append(physical)
        histories = data
    arrays = dict(observations=np.asarray(observations), actions=np.asarray(actions),
        outcomes=np.swapaxes(np.asarray(outcomes), 0, 1), truth=np.swapaxes(np.asarray(truth), 0, 1),
        initial_observations=history.observations, initial_actions=history.actions,
        initial_outcomes=history.survival)
    return streams, rows, arrays


def check_prediction_array(probabilities, learners, streams, arrays):
    """Recover every all-action forecast and independently aggregate physical metrics."""
    packet_count, size = len(streams), len(streams[0][0][0])
    require(probabilities.dtype == np.float32
            and probabilities.shape == (2, packet_count, len(learners), size, 5, 3),
            "invalid saved prediction array shape/dtype")
    signatures = [state_signature(learner) for learner in learners]
    result = {}
    for branch, name in enumerate(BRANCHES):
        errors, survivals = [], []
        for packet, stream in enumerate(streams):
            data, history = stream[branch]
            expected = np.asarray([predict_all(learner, data.observations, history) for learner in learners])
            require(np.array_equal(probabilities[branch, packet], expected),
                    "saved diagnostic prediction differs from frozen-state reconstruction")
            require(np.isfinite(expected).all() and np.all(np.diff(expected, axis=-1) <= np.finfo(np.float32).eps),
                    "invalid horizon predictions")
            performed = expected[:, np.arange(size), data.actions].astype(np.float64)
            errors.append(np.square(performed-data.survival).mean(axis=(1, 2)))
            greedy = expected[..., -1].argmax(axis=-1)
            survivals.append(arrays["truth"][branch, packet, np.arange(size)[None, :], greedy, -1].mean(axis=1))
        errors, survivals = np.asarray(errors), np.asarray(survivals)
        result[name] = dict(brier=errors.mean(axis=0).tolist(), survival=survivals.mean(axis=0).tolist(),
            early_brier=errors[0].tolist(), late_brier=errors[1:].mean(axis=0).tolist())
    require(signatures == [state_signature(learner) for learner in learners], "frozen evaluation mutated state")
    return result


def audit_assessment(directory, config, identity, seed, model, index, old_payload, new,
                     metadata, anchor_new, bundle, score_record, score_payload, device="cpu"):
    """Check score-only selection before opening any held-out diagnostic files."""
    choice, choice_hash = verify_choice(directory/f"assessment_{index}"/"choice.json",
        bundle["metadata"][:-1], score_record, identity, seed, model, index)
    folder = directory/f"diagnostic_{index}"
    expected_files = {"before.pt", "data.npz", "stream.json", "result.json", "zero.pt"}
    expected_files |= {f"cell_{cell}.pt" for cell in CELLS[1:]}
    expected_files |= {f"predictions_{cell}.{suffix}" for cell in (*CELLS, "zero") for suffix in ("json", "npz")}
    require({path.name for path in folder.iterdir()} == expected_files, "diagnostic artifact coverage mismatch")
    old, anchor_old = old_payload["learner"], old_payload["anchor"]
    parents = dict(old=state_signature(old), new=state_signature(new))
    boundary = checked_load(folder/"before.pt", identity, device)
    require(state_signature(boundary["learner"]) == parents["new"]
            and boundary["record"] == dict(parents=parents) and boundary["metadata"] == metadata
            and _anchor_hash(boundary["anchor"]) == _anchor_hash(anchor_new),
            "diagnostic boundary differs from ordinary score endpoint")
    record = read_json(folder/"result.json")
    streams, rows, arrays = regenerate_streams(seed, index, config, new.history)
    saved_arrays = read_arrays(folder/"data.npz")
    require(saved_arrays.keys() == arrays.keys() and all(saved_arrays[key].dtype == arrays[key].dtype
            and np.array_equal(saved_arrays[key], arrays[key]) for key in arrays),
            "paired branch data or causal initial history differs")
    require(read_json(folder/"stream.json") == dict(identity=identity, seed=seed, model=model, index=index,
            branches=list(BRANCHES), rows=rows), "paired stream provenance differs")
    panels = {branch:{} for branch in BRANCHES}
    work, active, state_files, prediction_files, zero = {}, {}, {}, {}, None
    forks, prediction_calls = 0, 0
    for cell in (*CELLS, "zero"):
        if cell == "000":
            path, payload = directory/f"score_{index}"/"shadows.pt", score_payload
        elif cell == "zero":
            path = folder/"zero.pt"
            payload = checked_load(path, identity, device)
            require(payload["parents"] == parents and len(payload["learners"]) == 2,
                    "zero-update donor coverage mismatch")
            require(payload["signatures"] == [parents["old"], parents["new"]]
                    == [state_signature(learner) for learner in payload["learners"]],
                    "zero-update copies differ from exact donors")
            seen = _owned_tensor_pointers(old)|_owned_tensor_pointers(new)
            for learner in payload["learners"]:
                pointers = _owned_tensor_pointers(learner)
                require(pointers.isdisjoint(seen), "zero-update tensor ownership overlaps")
                seen.update(pointers)
        else:
            path = folder/f"cell_{cell}.pt"
            payload = checked_load(path, identity, device)
            forks += check_factor_cell(payload, old, new, anchor_old, anchor_new, bundle, cell, config)
        learners = payload["learners"]
        active[cell] = [recorded_active_hash(learner) for learner in learners]
        prediction_path = folder/f"predictions_{cell}.npz"
        saved_predictions = read_arrays(prediction_path)
        require(set(saved_predictions) == {"probabilities"}, "unexpected prediction array fields")
        probabilities = saved_predictions["probabilities"]
        require(read_json(folder/f"predictions_{cell}.json") == dict(identity=identity, parents=parents,
                active_signatures=active[cell], data_sha256=file_hash(folder/"data.npz"),
                predictions_sha256=file_hash(prediction_path)), "prediction seal differs from data/state")
        metrics = check_prediction_array(probabilities, learners, streams, arrays)
        prediction_calls += 2*len(streams)*len(learners)
        if cell == "zero":
            zero = metrics
            for weight in (0, 1):
                reference = probabilities[0, 0, weight]
                for suffix in ("00", "01", "10", "11"):
                    untrained = transplant_independent(old, new, str(weight)+suffix)
                    data, history = streams[0][0]
                    require(np.array_equal(predict_all(untrained, data.observations, history), reference),
                            "zero-dose prediction depends on optimizer/anchor")
        else:
            work[cell] = payload["work"]
            for branch in BRANCHES:
                panels[branch][cell] = metrics[branch]
        state_files[cell] = dict(path=path.relative_to(directory).as_posix(), sha256=file_hash(path))
        prediction_files[cell] = dict(path=prediction_path.relative_to(directory).as_posix(),
                                      sha256=file_hash(prediction_path))
    relevant = [folder/"before.pt", folder/"data.npz", folder/"stream.json",
        directory/f"assessment_{index}"/"candidates.pt", directory/f"assessment_{index}"/"choice.json",
        directory/f"score_{index}"/"result.json", directory/f"score_{index}"/"before.pt"]
    relevant += [directory/item["path"] for item in state_files.values()]+list(folder.glob("predictions_*"))
    expected = dict(identity=identity, seed=seed, model=model, index=index, cue=score_record["phase"]["cue"],
        parents=parents, old_steps=old.cost["optimizer_steps"], new_steps=new.cost["optimizer_steps"],
        old_anchor_sha256=packet_fingerprint(anchor_old), new_anchor_sha256=packet_fingerprint(anchor_new),
        candidates=bundle["metadata"][:-1], replacement=bundle["metadata"][-1],
        reservoir_ids=bundle["reservoir_ids"], anchor_id=bundle["anchor_id"],
        query_support_overlaps=bundle["query_support_overlaps"],
        candidate_storage_bytes=sum(packet.nbytes() for packet in bundle["packets"]),
        choice=choice, panels=panels, zero=zero, work=work, active_signatures=active,
        state_files=state_files, prediction_files=prediction_files,
        artifact_hashes={path.relative_to(directory).as_posix():file_hash(path) for path in set(relevant)},
        prediction_work=dict(score_forwards=(config["candidate_count"]+1)*config["score_size"]//config["batch_size"],
            factorial_forwards=2*len(streams)*8*(config["candidate_count"]+1), zero_forwards=4*len(streams)))
    compare(expected, record, f"assessment[{index}]")
    require(choice_hash == file_hash(directory/f"assessment_{index}"/"choice.json")
            and parents == dict(old=state_signature(old), new=state_signature(new)),
            "diagnostic audit changed selection or ordinary donors")
    return dict(index=index, result_sha256=file_hash(folder/"result.json"),
        before_sha256=file_hash(folder/"before.pt"), data_sha256=file_hash(folder/"data.npz"),
        stream_sha256=file_hash(folder/"stream.json"), state_files=state_files, prediction_files=prediction_files,
        choice_sha256=choice_hash, candidates_sha256=file_hash(directory/f"assessment_{index}"/"candidates.pt"),
        score_result_sha256=file_hash(directory/f"score_{index}"/"result.json"), parents=parents,
        newly_reproduced_forks=forks, reused_score_forks=len(score_payload["learners"]), zero_update_copies=2,
        diagnostic_prediction_calls_recomputed=prediction_calls,
        loss_instrumentation_forwards_recomputed=4*forks,
        all_new_active_endpoints_verified=len(bundle["ids"]),
        paired_causal_branches_verified=True, unchanged_donors_verified=True,
        score_only_selection_verified=True)


def audit_job(directory, config, identity, seed, model, device="cpu"):
    """Audit one completed trajectory; scientific lock checking is the caller's job."""
    validate_config(config)
    affinity(config["cpu_affinity"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(directory)
    phases = schedule(seed, config)
    expected_folders = {phase["name"] for phase in phases}|{
        "assessment_0", "assessment_1", "diagnostic_0", "diagnostic_1", "result.json"}
    require({path.name for path in directory.iterdir()} == expected_folders, "trajectory artifact coverage mismatch")
    world, memory = AcquisitionWorld(), make_memory("uniform", config, seed)
    expected_learner = new_learner(model, seed, config, device)
    metadata, history, anchor, seen, peak = {}, None, None, 0, 0
    bundles, starts, shadows, records, phase_checks, assessments = {}, {}, {}, {}, [], []
    total_forks = total_predictions = 0
    for phase in phases:
        name, kind, index = phase["name"], phase["kind"], phase["assessment"]
        folder = directory/name
        require({path.name for path in folder.iterdir()} == {"before.pt", "checkpoint.pt", "result.json"}
                |({"shadows.pt"} if kind == "score" else set()), "ordinary phase artifact coverage mismatch")
        if name.startswith("validation_"):
            # Score selection is checked before any future diagnostic state/data is opened.
            check = audit_assessment(directory, config, identity, seed, model, index, starts[index],
                expected_learner, metadata, anchor, bundles[index], records[f"score_{index}"], shadows[index], device)
            assessments.append(check)
            total_forks += check["newly_reproduced_forks"]
            total_predictions += check["diagnostic_prediction_calls_recomputed"]
        before, after = (checked_load(folder/file, identity, device) for file in ("before.pt", "checkpoint.pt"))
        initial, final = before["learner"], after["learner"]
        record = read_json(folder/"result.json")
        require(after["record"] == record and record["identity"] == identity
                and record["seed"] == seed and record["model"] == model and record["phase"] == phase,
                "ordinary record identity/schedule mismatch")
        require(initial.method == final.method == model and initial.settings == final.settings == config,
                "ordinary learner type/settings mismatch")
        require(state_signature(initial) == state_signature(expected_learner) == record["before_signature"],
                "ordinary starting state chain mismatch")
        require(before["metadata"] == metadata and _anchor_hash(before["anchor"]) == _anchor_hash(anchor),
                "ordinary starting metadata/anchor mismatch")
        require(memory_state(initial.memory) == memory_state(memory)
                and _history_hash(initial.history) == _history_hash(history), "ordinary starting replay/history mismatch")
        expected_before = {key:record[key] for key in ("identity", "seed", "model", "phase",
            "before_signature", "start_diagnostics", "start_memory")}
        expected_before.update(packets=[], evaluations=[], completed_packets=0, shadow_work=[], frozen_work=[])
        require(before["record"] == expected_before, "ordinary before checkpoint contains later evidence")
        compare(initial.diagnostics(), record["start_diagnostics"])
        compare(memory_state(initial.memory), record["start_memory"])
        check_original_cost(initial, seen, config)
        require(initial.peak_memory_bytes == peak, "ordinary starting peak storage mismatch")
        signature = state_signature(initial)
        shadow = None
        if kind == "score":
            bundle = checked_load(directory/f"assessment_{index}"/"candidates.pt", identity, device)
            check_bundle(bundle, before, metadata, anchor, seed, index, config)
            shadow = checked_load(folder/"shadows.pt", identity, device)
            total_forks += check_shadows(shadow, initial, anchor, bundle, config)
            require(record["shadow_work"] == shadow["work"] and record["frozen_work"] == [],
                    "score shadow work mismatch")
            bundles[index], starts[index], shadows[index] = bundle, before, shadow
        else:
            require(record["shadow_work"] == record["frozen_work"] == record["evaluations"] == [],
                    "ordinary train phase contains undeclared diagnostic work")
        count = phase["size"]//config["batch_size"]
        require(record["completed_packets"] == count == len(record["packets"])
                and len(record["evaluations"]) == (count if shadow else 0), "incomplete ordinary phase")
        evaluations = []
        for packet_index, row in enumerate(record["packets"]):
            data = batch(world, law_from_record(phase["law"]), seed, "predictive_value_"+name,
                         packet_index, config["batch_size"])
            info = check_packet_row(row, data, history, seen, phase, packet_index, config)
            if packet_index == 0:
                prediction = recompute_prediction(initial, data, history)
                compare(prediction["probabilities"], row["probabilities"])
                require(prediction["prediction_sha256"] == row["original_prediction_sha256"]
                        and state_hash(initial.model.state_dict()) == row["original_model_sha256"],
                        "ordinary phase-start prediction provenance mismatch")
            if shadow:
                evaluation = dict(packet_id=seen, query_sha256=data.fingerprint(), support_sha256=_history_hash(history),
                    reapplied=[recompute_prediction(learner, data, history) for learner in shadow["learners"]])
                compare(evaluation, record["evaluations"][packet_index])
                evaluations.append(evaluation)
                total_predictions += len(shadow["learners"])
            for _ in range(config["updates_per_batch"]):
                memory.sample()
            anchor = Packet(history, data)
            memory.add(anchor)
            history, seen = data, seen+1
            metadata[info["id"]] = info
            metadata = {identifier:metadata[identifier] for identifier in memory.ids}
            peak = max(peak, memory.nbytes())
        require(state_signature(initial) == signature, "ordinary audit mutated initial state")
        if shadow:
            require([state_signature(learner) for learner in shadow["learners"]] == shadow["signatures"],
                    "score forecasts mutated frozen rehearsal models")
            losses = np.asarray([[value["brier"] for value in row["reapplied"]] for row in evaluations]).mean(axis=0).tolist()
            compare(losses, record["score_losses"])
            verify_choice(directory/f"assessment_{index}"/"choice.json", bundles[index]["metadata"][:-1],
                          record, identity, seed, model, index)
        require(memory_state(final.memory) == memory_state(memory)
                and _history_hash(final.history) == _history_hash(history), "ordinary causal replay/history differs")
        require(after["metadata"] == metadata and record["metadata"] == {str(key):value for key,value in metadata.items()},
                "ordinary final memory metadata differs")
        require(_anchor_hash(after["anchor"]) == _anchor_hash(anchor) and final.peak_memory_bytes == peak,
                "ordinary final anchor or peak storage differs")
        require(state_signature(final) == record["final_signature"], "ordinary final signature mismatch")
        compare(final.diagnostics(), record["diagnostics"])
        compare(memory_state(final.memory), record["memory"])
        check_original_cost(final, seen, config)
        records[name], expected_learner = record, final
        phase_checks.append(dict(phase=name, packets=count, cumulative_packets=seen,
            result_sha256=file_hash(folder/"result.json"), before_sha256=file_hash(folder/"before.pt"),
            checkpoint_sha256=file_hash(folder/"checkpoint.pt"),
            shadows_sha256=None if shadow is None else file_hash(folder/"shadows.pt"),
            rehearsal_forks_reproduced=0 if shadow is None else len(shadow["learners"]),
            diagnostic_predictions_recomputed=len(evaluations),
            causal_data_replay_metadata_verified=True, exact_start_chain_verified=True,
            adam_counters_verified=True, start_prediction_recomputed=True))
    for index in (0, 1):
        require({path.name for path in (directory/f"assessment_{index}").iterdir()} == {"candidates.pt", "choice.json"},
                "candidate artifact coverage mismatch")
    job = read_json(directory/"result.json")
    require(job["identity"] == identity and job["seed"] == seed and job["model"] == model
            and job["phase_names"] == [phase["name"] for phase in phases], "trajectory identity mismatch")
    require(job["phase_hashes"] == {phase["name"]:file_hash(directory/phase["name"]/"result.json") for phase in phases}
            and job["assessment_hashes"] == {str(index):file_hash(directory/f"diagnostic_{index}"/"result.json")
                                            for index in (0, 1)}, "trajectory artifact hashes mismatch")
    compare(expected_learner.diagnostics(), job["diagnostics"])
    require(job["final_signature"] == state_signature(expected_learner), "trajectory final state mismatch")
    require(job["prequential_work"] == dict(forwards=seen, query_presentations=seen*config["batch_size"],
            support_presentations=seen*config["batch_size"]), "ordinary prequential work mismatch")
    result = dict(seed=seed, model=model, before_checkpoints=len(phases), final_checkpoints=len(phases),
        diagnostic_boundary_checkpoints=2, assessments=2, ordinary_packets=seen,
        original_packet_errors_recomputed=seen, original_predictions_recomputed=len(phases),
        diagnostic_forks_reproduced=total_forks, zero_update_copies=4,
        diagnostic_prediction_calls_recomputed=total_predictions,
        loss_instrumentation_forwards_recomputed=sum(item["loss_instrumentation_forwards_recomputed"] for item in assessments),
        phases=phase_checks, assessment_checks=assessments,
        input_stream_sha256=digest([row["query_sha256"] for phase in phases for row in records[phase["name"]]["packets"]]))
    print(f"audited {model} seed={seed}: {total_forks} factor forks, {total_predictions} diagnostic forecasts", flush=True)
    return result


def audit(directory, output, workers=None):
    """Scientific entry point: locked record reader and exact source are mandatory."""
    from acp_cl.rehearsal_state.study import source_manifest
    from summarize_rehearsal_state import audit_records

    started = time.perf_counter()
    directory, output = Path(directory), Path(output)
    require(not output.resolve().is_relative_to(directory.resolve()), "audit output must be outside the run directory")
    manifest, records, hashes = audit_records(directory)
    config, identity = manifest["config"], manifest["identity"]
    require(digest(source_manifest()) == identity["source_sha256"], "audit requires exact locked study source")
    count = config["workers"] if workers is None else workers
    require(type(count) is int and count >= 1, "positive audit worker count required")
    jobs = [(seed, model) for model in config["models"] for seed in config["seeds"]]
    checks = []
    if count == 1:
        for seed, model in jobs:
            checks.append(audit_job(directory/f"{model}_{seed}", config, identity, seed, model, manifest["runtime"]["device"]))
    else:
        with ProcessPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(audit_job, directory/f"{model}_{seed}", config, identity, seed, model,
                                   manifest["runtime"]["device"]) for seed, model in jobs]
            for future in as_completed(futures):
                checks.append(future.result())
    for seed in config["seeds"]:
        require(len({item["input_stream_sha256"] for item in checks if item["seed"] == seed}) == 1,
                "ordinary input streams differ across architectures")
    totals = {name:sum(item[name] for item in checks) for name in (
        "before_checkpoints", "final_checkpoints", "diagnostic_boundary_checkpoints", "assessments",
        "ordinary_packets", "original_packet_errors_recomputed", "original_predictions_recomputed",
        "diagnostic_forks_reproduced", "zero_update_copies", "diagnostic_prediction_calls_recomputed",
        "loss_instrumentation_forwards_recomputed")}
    expected = counts(config)
    require(len(checks) == expected["trajectories"] and len(records) == totals["assessments"] == expected["assessments"],
            "cohort trajectory/assessment coverage mismatch")
    for actual, declared in (("diagnostic_forks_reproduced", "diagnostic_forks"),
            ("zero_update_copies", "zero_update_copies"),
            ("diagnostic_prediction_calls_recomputed", "diagnostic_prediction_forwards"),
            ("loss_instrumentation_forwards_recomputed", "extra_loss_forwards")):
        require(totals[actual] == expected[declared], "cohort declared diagnostic work differs")
    result = dict(passed=True, identity=identity, trajectories=len(checks), **totals,
        result_hashes=hashes, manifest_sha256=file_hash(directory/"manifest.json"),
        audit_source_sha256=file_hash(__file__), checks=sorted(checks, key=lambda item:(item["model"], item["seed"])),
        elapsed_seconds=time.perf_counter()-started,
        scope=("Exact-source/record locks; regenerated ordinary data, uniform replay, causal history and metadata; "
               "checkpoint state chains, gradients, parameter ownership, Adam and work counters; independently "
               "transplanted full named Adam state and reproduced every short rehearsal intervention; complete "
               "diagnostic forecasts and paired causal branch physics. The reused 000 forks are independently "
               "reproduced at scoring. All 111 active states equal ordinary new-parent rehearsal; inactive state "
               "deliberately remains old. Both zero-update donor copies and their forecasts are verified. Original "
               "prequential errors are recomputed from stored arrays; only phase-start forecasts are recovered "
               "from historical weights. The full ordinary learning trajectory is not independently retrained. "
               "Score-only choice reconstruction verifies causal dependencies, not historical wall-clock file "
               "creation order. No new scientific seeds are fitted by this audit."))
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int)
    arguments = parser.parse_args()
    audit(arguments.input, arguments.output, arguments.workers)
