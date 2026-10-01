"""Regenerate causal records, reproduce rehearsal forks and rescore all panels.

The ordinary learning trajectory is not independently retrained. Historical
prequential arrays are checked arithmetically and against causal metadata;
their predictions are independently recoverable only at saved phase starts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_conditional import compare
from audit_selective_updates import check_optimizer, check_original_cost
from acp_cl.acquisition.world import AcquisitionWorld, batch
from acp_cl.conditional.learner import Packet
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest, trial_seed
from acp_cl.predictive_value.design import law_from_record, schedule, validate_config
from acp_cl.predictive_value.mechanism import (
    _value_hash, packet_fingerprint, predict_all, state_signature,
)
from acp_cl.predictive_value.study import load, source_manifest
from acp_cl.replay_renewal.memory import make_memory, memory_state
from acp_cl.replay_renewal.study import affinity, new_learner, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.asarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode()+value.tobytes()).hexdigest()


def _hash_string(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _history_hash(history):
    return None if history is None else history.fingerprint()


def _anchor_hash(anchor):
    return None if anchor is None else packet_fingerprint(anchor)


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def checked_load(path, identity, device="cpu"):
    """Use the gradient-restoring loader, then validate its explicit coverage."""
    saved = load(path, identity, device)
    learners = [saved["learner"]] if "learner" in saved else saved.get("learners", [])
    require(len(saved["saved_gradients"]) == len(learners), "gradient coverage mismatch")
    for learner, stored in zip(learners, saved["saved_gradients"]):
        parameters = dict(learner.model.named_parameters())
        require(set(stored) == set(parameters), "gradient parameter coverage mismatch")
        for name, parameter in parameters.items():
            gradient = parameter.grad
            if stored[name] is None:
                require(gradient is None, "restored unexpected gradient")
            else:
                require(isinstance(gradient, torch.Tensor)
                        and gradient.shape == parameter.shape
                        and gradient.dtype == parameter.dtype
                        and gradient.device == parameter.device
                        and torch.isfinite(gradient).all().item(), "invalid restored gradient")
                require(torch.equal(gradient.cpu(), stored[name].cpu()), "gradient changed in load")
                require(gradient.data_ptr() != parameter.data_ptr(), "gradient aliases parameter")
    return saved


def independent_choice(candidates, losses):
    """No validation argument exists: selections use score feedback and past error."""
    ids = [item["id"] for item in candidates]
    losses = np.asarray(losses, dtype=np.float64)
    errors = np.asarray([item["original_brier"] for item in candidates], dtype=np.float64)
    require(len(ids) > 0 and len(set(ids)) == len(ids)
            and losses.shape == (len(ids)+1,) and np.isfinite(losses).all()
            and np.isfinite(errors).all(), "invalid selection inputs")
    values = losses[-1]-losses[:-1]
    selected = min(zip(-values, ids))[1]
    accurate = min(zip(errors, ids))[1]
    require(selected == min(zip(losses[:-1], ids))[1],
            "common-baseline ranking identity failed")
    return dict(value_id=int(selected), accuracy_id=int(accurate),
                score_losses=losses.tolist(), values=values.tolist())


def verify_choice(path, candidates, score_record, identity, seed, model, index):
    expected = dict(identity=identity, seed=seed, model=model, index=index,
                    score_record_sha256=digest(score_record),
                    choice=independent_choice(candidates, score_record["score_losses"]))
    require(read_json(path) == expected, "sealed choice does not follow score-window evidence")
    return expected["choice"], file_hash(path)


def check_packet_row(row, data, history, packet_id, phase, local_index, config):
    expected = dict(id=packet_id, original_steps=packet_id*config["updates_per_batch"],
        query_sha256=data.fingerprint(), support_sha256=_history_hash(history),
        origin_law=phase["law"], origin_phase=phase["name"], origin_index=local_index)
    for name, value in expected.items():
        require(row[name] == value, f"original packet provenance mismatch: {name}")
    require(set(row) == set(expected)|{"original_brier", "original_model_sha256",
            "original_prediction_sha256", "probabilities", "actions", "outcomes"},
            "unexpected prequential row fields")
    require(row["actions"] == data.actions.tolist() and row["outcomes"] == data.survival.tolist(),
            "recorded actions/outcomes differ from regenerated experience")
    require(_hash_string(row["original_model_sha256"])
            and _hash_string(row["original_prediction_sha256"]), "invalid prediction/model fingerprint")
    probabilities = np.asarray(row["probabilities"], dtype=np.float64)
    epsilon = np.finfo(np.float32).eps
    require(probabilities.shape == (len(data), 3) and np.isfinite(probabilities).all()
            and np.all(probabilities >= -epsilon) and np.all(probabilities <= 1+epsilon)
            and np.all(np.diff(probabilities, axis=1) <= epsilon),
            "invalid original performed-action probabilities")
    brier = float(np.square(probabilities-data.survival).mean())
    compare(brier, row["original_brier"], "original_brier")
    return {key: value for key, value in row.items()
            if key not in ("probabilities", "actions", "outcomes")}


def check_bundle(bundle, before, metadata, anchor, seed, index, config):
    parent = before["learner"]
    require(bundle["parent_signature"] == state_signature(parent)
            and bundle["parent_steps"] == parent.cost["optimizer_steps"],
            "candidate parent state mismatch")
    latest = parent.memory.seen-1
    pool = sorted(set(parent.memory.ids)-{latest})
    generator = np.random.default_rng(trial_seed(seed, "predictive_value_candidates", index))
    chosen = generator.choice(pool, config["candidate_count"]+1, replace=False).tolist()
    ids = sorted(chosen[:-1])+[chosen[-1]]
    require(bundle["ids"] == ids and bundle["anchor_id"] == latest
            and bundle["reservoir_ids"] == list(parent.memory.ids),
            "candidate sampling or cutoff mismatch")
    require(_anchor_hash(bundle["anchor"]) == _anchor_hash(anchor), "candidate anchor mismatch")
    require(len(bundle["packets"]) == len(ids) == len(bundle["metadata"]),
            "candidate packet coverage mismatch")
    by_id = dict(zip(parent.memory.ids, parent.memory.packets))
    for packet_id, packet, info in zip(ids, bundle["packets"], bundle["metadata"]):
        require(packet_fingerprint(packet) == packet_fingerprint(by_id[packet_id]),
                "candidate packet differs from causal reservoir")
        require(info == metadata[packet_id], "candidate original metadata mismatch")
    overlaps = [[i, j] for i in ids for j in ids if i != j and by_id[j].support is not None
                and by_id[i].query.fingerprint() == by_id[j].support.fingerprint()]
    require(bundle["query_support_overlaps"] == overlaps, "candidate overlap record mismatch")


def reproduce_shadow(parent, anchor, replay, updates):
    """Independent implementation of the declared equal-BCE clipped-Adam recipe."""
    shadow = copy.deepcopy(parent)
    for old, new in zip(parent.model.parameters(), shadow.model.parameters()):
        new.grad = None if old.grad is None else old.grad.detach().clone()
    require(state_signature(shadow) == state_signature(parent), "inexact audit rehearsal fork")
    for _ in range(updates):
        shadow.model.train()
        losses = []
        for packet in (anchor, replay):
            require(packet.oracle_modes is None, "hidden labels in rehearsal packet")
            probabilities = shadow.probabilities(packet.query.observations, packet.support)[0]
            actions = torch.as_tensor(packet.query.actions.astype(np.int64), device=shadow.device)
            labels = torch.as_tensor(packet.query.survival.astype(np.float32), device=shadow.device)
            selected = probabilities[torch.arange(len(actions), device=shadow.device), actions]
            losses.append(F.binary_cross_entropy(selected.clamp(1e-6, 1-1e-6), labels))
        shadow.optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(shadow.model.parameters(), 5.)
        shadow.optimizer.step()
    return shadow


def _owned_tensor_pointers(learner):
    pointers = set()
    for parameter in learner.model.parameters():
        pointers.add(parameter.data_ptr())
        if parameter.grad is not None:
            pointers.add(parameter.grad.data_ptr())
    for state in learner.optimizer.state.values():
        pointers.update(value.data_ptr() for value in state.values()
                        if isinstance(value, torch.Tensor))
    return pointers


def check_shadows(payload, parent, anchor, bundle, config):
    start = state_signature(parent)
    learners, work = payload["learners"], payload["work"]
    require(payload["parent_signature"] == start
            and payload["anchor_sha256"] == packet_fingerprint(anchor)
            and payload["ids"] == bundle["ids"], "shadow parent/anchor/candidate mismatch")
    require(len(learners) == len(bundle["ids"]) == len(work) == len(payload["signatures"]),
            "shadow coverage mismatch")
    seen_pointers = _owned_tensor_pointers(parent)
    updates, size = config["rehearsal_updates"], config["batch_size"]
    for saved, counters, signature, packet in zip(learners, work, payload["signatures"], bundle["packets"]):
        pointers = _owned_tensor_pointers(saved)
        require(pointers.isdisjoint(seen_pointers), "shadow tensor ownership overlaps")
        seen_pointers.update(pointers)
        check_optimizer(saved, parent.cost["optimizer_steps"]+updates, config)
        expected = reproduce_shadow(parent, anchor, packet, updates)
        require(state_signature(saved) == signature == state_signature(expected),
                "saved shadow differs from independently reproduced rehearsal")
        expected_work = dict(steps=updates, forwards=2*updates, backward_calls=updates,
            query_presentations=2*size*updates, support_presentations=2*size*updates,
            model_bytes=_tensor_bytes(saved.model.state_dict()),
            optimizer_bytes=_tensor_bytes(saved.optimizer.state_dict()),
            gradient_bytes=sum(_tensor_bytes(p.grad) for p in saved.model.parameters()),
            initial_encoder_bytes=_tensor_bytes(saved.initial_encoder),
            replay_bytes=parent.memory.nbytes(),
            history_bytes=0 if parent.history is None else Packet(None, parent.history).nbytes(),
            anchor_packet_sha256=packet_fingerprint(anchor), replay_packet_sha256=packet_fingerprint(packet),
            initial_model_sha256=state_hash(parent.model.state_dict()),
            final_model_sha256=state_hash(saved.model.state_dict()),
            initial_optimizer_sha256=_value_hash(parent.optimizer.state_dict()),
            final_optimizer_sha256=_value_hash(saved.optimizer.state_dict()),
            parent_state_sha256=start, shadow_state_sha256=signature,
            parent_unchanged=True, memory_history_unchanged=True)
        expected_work["explicit_tensor_bytes_subtotal"] = sum(expected_work[name] for name in (
            "model_bytes", "optimizer_bytes", "gradient_bytes", "initial_encoder_bytes",
            "replay_bytes", "history_bytes"))
        expected_work["explicit_tensor_bytes_scope"] = (
            "Sum of explicit model, optimizer, gradient, initial-encoder, replay and history storage. "
            "Replay/history aliases may be conservatively double counted. Excludes autograd graphs, "
            "operator workspaces, Python objects and allocator overhead; not measured peak RAM.")
        require(set(counters) == set(expected_work)|{"training_seconds", "interpretation"},
                "unexpected rehearsal work fields")
        for name, value in expected_work.items():
            require(counters[name] == value, f"rehearsal work mismatch: {name}")
        require(math.isfinite(counters["training_seconds"]) and counters["training_seconds"] >= 0,
                "invalid rehearsal time")
        require(memory_state(saved.memory) == memory_state(parent.memory)
                and _history_hash(saved.history) == _history_hash(parent.history)
                and saved.cost == parent.cost, "rehearsal changed ordinary state or counters")
    require(state_signature(parent) == start, "audit rehearsal changed parent")
    return len(learners)


def recompute_prediction(learner, data, support, truth=None):
    probabilities = predict_all(learner, data.observations, support)
    performed = probabilities[np.arange(len(data)), data.actions].astype(np.float64)
    result = dict(probabilities=performed.tolist(), prediction_sha256=array_hash(probabilities),
                  brier=float(np.square(performed-data.survival).mean()))
    if truth is not None:
        actions = np.argmax(probabilities[:, :, -1], axis=1)
        result.update(actions=actions.tolist(),
            survival=float(truth[np.arange(len(data)), actions, -1].mean()))
    return result


def panel_aggregates(rows, key):
    brier = np.asarray([[entry["brier"] for entry in row[key]] for row in rows])
    survival = np.asarray([[entry["survival"] for entry in row[key]] for row in rows])
    return dict(brier=brier.mean(axis=0).tolist(), survival=survival.mean(axis=0).tolist(),
                early_brier=brier[0].tolist(), late_brier=brier[1:].mean(axis=0).tolist())


def audit_job(directory, config, identity, seed, model, device="cpu"):
    """Audit one completed trajectory; caller handles scientific manifest locks."""
    validate_config(config)
    affinity(config["cpu_affinity"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(directory)
    phases = schedule(seed, config)
    require({path.name for path in directory.iterdir()} ==
            {phase["name"] for phase in phases}|{"assessment_0", "assessment_1", "result.json"},
            "unexpected or missing trajectory artifacts")
    world, memory = AcquisitionWorld(), make_memory("uniform", config, seed)
    expected_learner = new_learner(model, seed, config, device)
    metadata, history, anchor, seen = {}, None, None, 0
    peak_memory = 0
    bundles, frozen_states, seals, records, checks = {}, {}, {}, {}, []
    total_shadows = total_predictions = 0
    for phase in phases:
        name, kind, assessment = phase["name"], phase["kind"], phase["assessment"]
        folder = directory/name
        expected_files = {"before.pt", "checkpoint.pt", "result.json"}
        if kind in ("score", "near", "return"):
            expected_files.add("shadows.pt")
        require({path.name for path in folder.iterdir()} == expected_files,
                "unexpected or missing phase artifacts")
        # This verifies the score seal before reading any later outcome/checkpoint.
        if kind in ("near", "return"):
            _, seal = verify_choice(directory/f"assessment_{assessment}"/"choice.json",
                bundles[assessment]["metadata"][:-1], records[f"score_{assessment}"],
                identity, seed, model, assessment)
            require(seal == seals[assessment], "selection changed after scoring")
        before = checked_load(folder/"before.pt", identity, device)
        after = checked_load(folder/"checkpoint.pt", identity, device)
        initial, final = before["learner"], after["learner"]
        record = read_json(folder/"result.json")
        require(after["record"] == record, "phase checkpoint/JSON mismatch")
        require(record["identity"] == identity and record["seed"] == seed
                and record["model"] == model and record["phase"] == phase,
                "phase identity or schedule mismatch")
        require(initial.method == final.method == model
                and initial.settings == final.settings == config, "learner type/settings mismatch")
        require(state_signature(initial) == state_signature(expected_learner)
                == record["before_signature"], "phase starting state chain mismatch")
        require(before["metadata"] == metadata and _anchor_hash(before["anchor"]) == _anchor_hash(anchor),
                "phase starting metadata or anchor mismatch")
        require(memory_state(initial.memory) == memory_state(memory)
                and _history_hash(initial.history) == _history_hash(history),
                "phase starting replay/history mismatch")
        expected_before = {key: record[key] for key in ("identity", "seed", "model", "phase",
            "before_signature", "start_diagnostics", "start_memory")}
        expected_before.update(packets=[], evaluations=[], completed_packets=0,
                               shadow_work=[], frozen_work=[])
        require(before["record"] == expected_before, "before checkpoint contains later evidence")
        compare(initial.diagnostics(), record["start_diagnostics"])
        compare(memory_state(initial.memory), record["start_memory"])
        check_original_cost(initial, seen, config)
        require(initial.peak_memory_bytes == peak_memory, "starting peak memory mismatch")
        initial_signature = state_signature(initial)
        shadow_payload = None
        if kind == "score":
            bundle = checked_load(directory/f"assessment_{assessment}"/"candidates.pt", identity, device)
            check_bundle(bundle, before, metadata, anchor, seed, assessment, config)
            bundles[assessment] = bundle
        if kind in ("score", "near", "return"):
            shadow_payload = checked_load(folder/"shadows.pt", identity, device)
            total_shadows += check_shadows(shadow_payload, initial, anchor, bundles[assessment], config)
            require(record["shadow_work"] == shadow_payload["work"] and record["frozen_work"] == [],
                    "phase shadow work mismatch")
            if kind == "score":
                frozen_states[assessment] = shadow_payload
        else:
            require(record["shadow_work"] == record["frozen_work"] == []
                    and record["evaluations"] == [], "ordinary phase has unexpected diagnostic work")
        count = phase["size"]//config["batch_size"]
        require(record["completed_packets"] == count and len(record["packets"]) == count,
                "incomplete ordinary phase")
        require(len(record["evaluations"]) == (count if shadow_payload else 0),
                "incomplete diagnostic phase")
        regenerated_evaluations = []
        for index, row in enumerate(record["packets"]):
            law = law_from_record(phase["law"])
            channel = "predictive_value_"+name
            data = batch(world, law, seed, channel, index, config["batch_size"])
            info = check_packet_row(row, data, history, seen, phase, index, config)
            if index == 0:
                prediction = recompute_prediction(initial, data, history)
                compare(prediction["probabilities"], row["probabilities"])
                require(prediction["prediction_sha256"] == row["original_prediction_sha256"]
                        and state_hash(initial.model.state_dict()) == row["original_model_sha256"],
                        "phase-start original prediction provenance mismatch")
            if shadow_payload is not None:
                truth = None
                if kind != "score":
                    cases = world.dataset(law, len(data), trial_seed(seed, "acquisition_"+channel, index))
                    require(np.array_equal(cases.observations, data.observations),
                            "truth/query observation mismatch")
                    truth = world.counterfactuals(cases)
                evaluation = dict(packet_id=seen, query_sha256=data.fingerprint(),
                                  support_sha256=_history_hash(history))
                if truth is not None:
                    evaluation["truth"] = truth.tolist()
                evaluation["reapplied"] = [recompute_prediction(item, data, history, truth)
                    for item in shadow_payload["learners"]]
                total_predictions += len(shadow_payload["learners"])
                if kind != "score":
                    evaluation["frozen"] = [recompute_prediction(item, data, history, truth)
                        for item in frozen_states[assessment]["learners"]]
                    total_predictions += len(frozen_states[assessment]["learners"])
                compare(evaluation, record["evaluations"][index], f"{name}.evaluation[{index}]")
                regenerated_evaluations.append(evaluation)
            for _ in range(config["updates_per_batch"]):
                memory.sample()
            anchor = Packet(history, data)
            memory.add(anchor)
            history, seen = data, seen+1
            metadata[info["id"]] = info
            metadata = {packet_id: metadata[packet_id] for packet_id in memory.ids}
            peak_memory = max(peak_memory, memory.nbytes())
        require(state_signature(initial) == initial_signature, "audit changed starting learner")
        if shadow_payload is not None:
            for payload in (shadow_payload, frozen_states[assessment]):
                require([state_signature(item) for item in payload["learners"]] == payload["signatures"],
                        "panel prediction changed frozen shadow state")
        require(memory_state(final.memory) == memory_state(memory)
                and _history_hash(final.history) == _history_hash(history),
                "ordinary replay/history does not regenerate")
        require(after["metadata"] == metadata and record["metadata"] ==
                {str(key): value for key, value in metadata.items()}, "final memory metadata mismatch")
        require(_anchor_hash(after["anchor"]) == _anchor_hash(anchor), "final causal anchor mismatch")
        require(final.peak_memory_bytes == peak_memory, "peak replay storage mismatch")
        require(state_signature(final) == record["final_signature"], "final learner signature mismatch")
        compare(final.diagnostics(), record["diagnostics"])
        compare(memory_state(final.memory), record["memory"])
        check_original_cost(final, seen, config)
        if kind == "score":
            losses = np.asarray([[item["brier"] for item in row["reapplied"]]
                                 for row in regenerated_evaluations]).mean(axis=0).tolist()
            compare(losses, record["score_losses"])
            _, seals[assessment] = verify_choice(directory/f"assessment_{assessment}"/"choice.json",
                bundles[assessment]["metadata"][:-1], record, identity, seed, model, assessment)
        elif kind in ("near", "return"):
            compare({key: panel_aggregates(regenerated_evaluations, key) for key in ("reapplied", "frozen")},
                    record["panels"])
        records[name], expected_learner = record, final
        checks.append(dict(phase=name, packets=count, cumulative_packets=seen,
            result_sha256=file_hash(folder/"result.json"),
            before_sha256=file_hash(folder/"before.pt"),
            checkpoint_sha256=file_hash(folder/"checkpoint.pt"),
            shadows_sha256=None if shadow_payload is None else file_hash(folder/"shadows.pt"),
            causal_data_replay_metadata_verified=True, exact_start_chain_verified=True,
            adam_counters_verified=True, start_prediction_recomputed=True,
            diagnostic_predictions_recomputed=len(regenerated_evaluations),
            rehearsal_forks_reproduced=0 if shadow_payload is None else len(shadow_payload["learners"])))
    for index in (0, 1):
        folder, bundle = directory/f"assessment_{index}", bundles[index]
        require({path.name for path in folder.iterdir()} == {"candidates.pt", "choice.json", "result.json"},
                "unexpected or missing assessment artifacts")
        assessment = read_json(folder/"result.json")
        score = records[f"score_{index}"]
        expected = dict(identity=identity, seed=seed, model=model, index=index,
            cue=score["phase"]["cue"], gap=score["phase"]["gap"],
            candidates=bundle["metadata"][:-1], replacement=bundle["metadata"][-1],
            reservoir_ids=bundle["reservoir_ids"], anchor_id=bundle["anchor_id"],
            parent_steps=bundle["parent_steps"], query_support_overlaps=bundle["query_support_overlaps"],
            candidate_storage_bytes=sum(packet.nbytes() for packet in bundle["packets"]),
            metadata_storage_bytes=len(json.dumps(bundle["metadata"]).encode()),
            choice=independent_choice(bundle["metadata"][:-1], score["score_losses"]),
            panels={f"{name}_{kind}":records[f"{name}_{index}"]["panels"][kind]
                    for name in ("near", "return") for kind in ("reapplied", "frozen")},
            phase_hashes={f"{name}_{index}":file_hash(directory/f"{name}_{index}"/"result.json")
                          for name in ("score", "near", "gap", "return")},
            work={name:records[f"{name}_{index}"]["shadow_work"] for name in ("score", "near", "return")},
            prediction_work=dict(score_shadow_forwards=(config["candidate_count"]+1)*config["score_size"]//config["batch_size"],
                validation_shadow_forwards=4*(config["candidate_count"]+1)*config["validation_size"]//config["batch_size"],
                query_records_per_forward=config["batch_size"], support_records_per_forward=config["batch_size"]))
        compare(expected, assessment)
    job = read_json(directory/"result.json")
    require(job["identity"] == identity and job["seed"] == seed and job["model"] == model
            and job["phase_names"] == [phase["name"] for phase in phases], "trajectory identity mismatch")
    require(job["phase_hashes"] == {phase["name"]:file_hash(directory/phase["name"]/"result.json") for phase in phases}
            and job["assessment_hashes"] == {str(index):file_hash(directory/f"assessment_{index}"/"result.json") for index in (0, 1)},
            "trajectory artifact hashes mismatch")
    compare(expected_learner.diagnostics(), job["diagnostics"])
    require(job["final_signature"] == state_signature(expected_learner), "trajectory final signature mismatch")
    result = dict(seed=seed, model=model, before_checkpoints=len(phases), final_checkpoints=len(phases),
        assessments=2, ordinary_packets=seen, original_packet_errors_recomputed=seen,
        original_predictions_recomputed=len(phases), diagnostic_forks_reproduced=total_shadows,
        diagnostic_prediction_calls_recomputed=total_predictions, phases=checks,
        input_stream_sha256=digest([row["query_sha256"] for phase in phases for row in records[phase["name"]]["packets"]]))
    print(f"audited {model} seed={seed}: {total_shadows} rehearsal forks, {total_predictions} diagnostic forwards", flush=True)
    return result


def audit(directory, output, workers=None):
    """Scientific entry point: locked reader and exact source are mandatory."""
    from summarize_predictive_value import audit_records

    started = time.perf_counter()
    directory = Path(directory)
    manifest, records, _ = audit_records(directory)
    config, identity = manifest["config"], manifest["identity"]
    require(digest(source_manifest()) == identity["source_sha256"], "audit requires the exact study source")
    count = config["workers"] if workers is None else workers
    require(type(count) is int and count >= 1, "positive audit worker count required")
    jobs = [(seed, model) for model in config["models"] for seed in config["seeds"]]
    checks = []
    if count == 1:
        for seed, model in jobs:
            checks.append(audit_job(directory/f"{model}_{seed}", config, identity, seed, model, manifest["runtime"]["device"]))
    else:
        with ProcessPoolExecutor(max_workers=count) as pool:
            pending = [pool.submit(audit_job, directory/f"{model}_{seed}", config, identity, seed, model,
                                   manifest["runtime"]["device"]) for seed, model in jobs]
            for future in as_completed(pending):
                checks.append(future.result())
    for seed in config["seeds"]:
        require(len({item["input_stream_sha256"] for item in checks if item["seed"] == seed}) == 1,
                "ordinary input streams differ across architectures")
    require(len(records) == sum(item["assessments"] for item in checks), "assessment coverage mismatch")
    result = dict(passed=True, identity=identity, trajectories=len(checks),
        before_checkpoints=sum(item["before_checkpoints"] for item in checks),
        final_checkpoints=sum(item["final_checkpoints"] for item in checks),
        assessments=sum(item["assessments"] for item in checks),
        diagnostic_forks_reproduced=sum(item["diagnostic_forks_reproduced"] for item in checks),
        diagnostic_prediction_calls_recomputed=sum(item["diagnostic_prediction_calls_recomputed"] for item in checks),
        checks=sorted(checks, key=lambda item:(item["model"], item["seed"])),
        elapsed_seconds=time.perf_counter()-started,
        scope=("Exact-source/record locks, causal input/replay/history/metadata reconstruction, phase-state chains, "
               "Adam and work counters, independently recreated short rehearsal forks and all diagnostic predictions. "
               "Original prequential errors are recomputed from stored arrays and all phase-start predictions are "
               "recomputed from saved weights. The full ordinary learning trajectory is not independently retrained; "
               "other historical prediction/model fingerprints have provenance and format checks, not independent "
               "prediction recovery. Score-only choice reconstruction verifies causal dependencies, not historical "
               "wall-clock file-creation order. This is not a new-seed scientific replication."))
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int)
    arguments = parser.parse_args()
    audit(arguments.input, arguments.output, arguments.workers)
