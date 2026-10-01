"""Audit trusted local consolidation checkpoints, causal replay and saved probes."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from audit_conditional import compare
from audit_replay_renewal import signature as learner_signature
from audit_training_state import advance_memory
from acp_cl.acquisition.study import Evaluator, marginal
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.evidence_consolidation.mechanism import DELAYED, POLICIES
from acp_cl.evidence_consolidation.schedule import experiences, schedule
from acp_cl.evidence_consolidation.study import (counts, evaluate, knowledge_panel,
    source_manifest, valid_panel, validate_config)
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import affinity, new_learner, write_json


def semantic(value):
    """Exclude measured wall-clock fields, retaining every scientific value."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items()
                if k not in ("elapsed_seconds", "training_seconds", "completed_utc", "created_utc")}
    if isinstance(value, (list, tuple)):
        return [semantic(v) for v in value]
    return value


def signature(bundle):
    snapshots = {**bundle.committed, "proposal": bundle.proposal}
    return digest(dict(draft=learner_signature(bundle.draft), state=semantic(bundle.state()),
        snapshots={name: None if s is None else dict(
            actual_hash=state_hash(s.model.state_dict()), advertised_hash=s.sha256,
            trainable=[p.requires_grad for p in s.model.parameters()],
            history=None if s.history is None else s.history.fingerprint())
            for name, s in snapshots.items()}, last_decision=bundle.last_decision))


def checked_manifest(directory):
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    config, identity = manifest["config"], manifest["identity"]
    validate_config(config)
    for key, value in (("config_sha256", config), ("source_sha256", manifest["source_files"]),
                       ("runtime_sha256", manifest["runtime"])):
        if digest(value) != identity[key]:
            raise AssertionError("manifest identity mismatch")
    if source_manifest() != manifest["source_files"]:
        raise ValueError("audit requires the exact study source")
    with zipfile.ZipFile(directory/"training_source.zip") as archive:
        if set(archive.namelist()) != set(manifest["source_files"]):
            raise AssertionError("training source archive file set mismatch")
        for name, expected in manifest["source_files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise AssertionError("training source archive hash mismatch")
    lock_path, protocol_path = directory/"protocol_lock.json", directory/"protocol_at_lock.md"
    if lock_path.exists() != protocol_path.exists():
        raise AssertionError("incomplete protocol lock")
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if any(lock[k] != v for k, v in identity.items()) or hashlib.sha256(
                protocol_path.read_bytes()).hexdigest() != lock["protocol_sha256"]:
            raise AssertionError("protocol lock mismatch")
    completion = json.loads((directory/"completion.json").read_text(encoding="utf-8"))
    if completion["identity"] != identity or any(completion[k] != v for k, v in counts(config).items()):
        raise AssertionError("completion/count mismatch")
    expected = {(s, m) for s in config["seeds"] for m in config["models"]}
    if len(completion["jobs"]) != len(expected) or {
            (j["seed"], j["model"]) for j in completion["jobs"]} != expected:
        raise AssertionError("completion job set mismatch")
    if any(j["affinity"] != manifest["runtime"]["affinity"] for j in completion["jobs"]):
        raise AssertionError("worker affinity mismatch")
    return manifest, completion


def check_budget(learner, batches, config):
    n, updates = batches*config["batch_size"], batches*config["updates_per_batch"]
    expected = dict(arrivals=n, optimizer_steps=updates,
                    query_presentations=2*n*config["updates_per_batch"],
                    support_presentations=2*n*config["updates_per_batch"])
    if any(learner.cost[k] != v for k, v in expected.items()):
        raise AssertionError("draft arrival/gradient/presentation budget mismatch")
    steps = [int(s["step"].item()) for s in learner.optimizer.state.values()]
    if (updates and (not steps or any(v != updates for v in steps))) or (not updates and steps):
        raise AssertionError("optimizer step state differs from declared work")
    if learner.memory.seen != batches or len(learner.memory.ids) > config["memory_packets"]:
        raise AssertionError("replay lifetime/capacity mismatch")
    if any(p.oracle_modes is not None for p in learner.memory.packets):
        raise AssertionError("privileged labels present in replay")


def check_snapshots(bundle, config):
    for snapshot in [*bundle.committed.values(), bundle.proposal]:
        if snapshot is None:
            continue
        if state_hash(snapshot.model.state_dict()) != snapshot.sha256:
            raise AssertionError("snapshot hash differs from actual parameters")
        if any(p.requires_grad for p in snapshot.model.parameters()):
            raise AssertionError("snapshot is trainable")
        if snapshot.history is not None or hasattr(snapshot, "optimizer") or hasattr(snapshot, "memory"):
            raise AssertionError("snapshot owns extra learning/history state")
    remaining = bundle.batches % config["evidence_window"]
    if any(len(bundle.gains[p]) != remaining for p in DELAYED):
        raise AssertionError("evidence window counter mismatch")
    if (bundle.proposal is None) != (remaining == 0):
        raise AssertionError("proposal lifetime mismatch")
    if remaining and bundle.proposal_batch != bundle.batches-remaining:
        raise AssertionError("proposal birth counter mismatch")
    if bundle.proposal_count != (bundle.batches+config["evidence_window"]-1)//config["evidence_window"]:
        raise AssertionError("proposal allocation counter mismatch")
    check_budget(bundle.draft, bundle.batches, config)


def advance_lineage(chain, decisions, config):
    """Independent decision arithmetic; no claim to rescore missing window weights."""
    width, margin = config["evidence_window"], config["evidence_margin"]
    for decision in decisions:
        if decision["start_batch"] != chain["end"] or decision["end_batch"] != chain["end"]+width:
            raise AssertionError("gate window missing, overlapping or reset at a boundary")
        if decision["committed_before"] != chain["committed"]:
            raise AssertionError("committed snapshot lineage mismatch")
        candidate = decision["proposal_hash"]
        if not isinstance(candidate, str) or len(candidate) != 64:
            raise AssertionError("invalid proposal hash")
        if chain["end"] == 0 and candidate != chain["initial"]:
            raise AssertionError("first proposal was not original draft")
        for policy in DELAYED:
            gains = np.asarray(decision["gains"][policy], dtype=float)
            if gains.shape != (width,) or not np.isfinite(gains).all() or np.any(abs(gains) > 1):
                raise AssertionError("invalid held-out gain window")
            accepted = (True if policy == "periodic" else bool(gains[0] > margin)
                        if policy == "single" else bool(gains.mean() > margin and
                            (gains > 0).sum() >= config["evidence_positive"]))
            if decision["accepted"][policy] != accepted:
                raise AssertionError("gate decision does not follow saved gains")
            if accepted:
                chain["committed"][policy] = candidate
                chain["adoptions"][policy] += 1
        chain["end"] += width


def check_lineage(bundle, chain, config):
    if bundle.batches//config["evidence_window"]*config["evidence_window"] != chain["end"]:
        raise AssertionError("decisions missing from final state")
    if {p: v.sha256 for p, v in bundle.committed.items()} != chain["committed"]:
        raise AssertionError("final committed model was not accepted proposal")
    if bundle.adoptions != chain["adoptions"]:
        raise AssertionError("adoption count mismatch")
    check_snapshots(bundle, config)


def audit(directory, output):
    started = time.perf_counter()
    directory = Path(directory)
    manifest, completion = checked_manifest(directory)
    config, identity = manifest["config"], manifest["identity"]
    affinity(config["cpu_affinity"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    block_checks, fresh_checks, prefix_count, windows = [], [], 0, 0

    def load(path, kind="bundle"):
        saved = torch.load(path, map_location=manifest["runtime"]["device"], weights_only=False)
        if saved["identity"] != identity or saved["record"]["identity"] != identity:
            raise AssertionError("checkpoint identity mismatch")
        return saved[kind], saved["record"]

    for seed in config["seeds"]:
        for model in config["models"]:
            folder = directory/f"{model}_{seed}"
            evaluator, world = Evaluator(seed, config), AcquisitionWorld()
            original = new_learner(model, seed, config, manifest["runtime"]["device"])
            initial_hash = state_hash(original.model.state_dict())
            chain = dict(initial=initial_hash, end=0, committed={p: initial_hash for p in DELAYED},
                         adoptions={p: 0 for p in DELAYED})
            previous, record = load(folder/"prefix.pt")
            compare(record, json.loads((folder/"prefix.json").read_text(encoding="utf-8")))
            if len(record["blocks"]) != config["prefix_blocks"]:
                raise AssertionError("incomplete prefix")
            data = []
            for k, saved in enumerate(record["blocks"]):
                law = Law((seed+k) % 2)
                observed = [batch(world, law, seed, f"prefix_{k}", i, config["batch_size"])
                            for i in range(config["prefix_size"]//config["batch_size"])]
                if saved["batch_sha256"] != [d.fingerprint() for d in observed] or digest(saved["law"]) != digest(asdict(law)):
                    raise AssertionError("prefix stream does not regenerate")
                data.extend(observed)
            memory, history = advance_memory(original.memory, None, data, config)
            compare(memory_state(memory), memory_state(previous.draft.memory))
            if history.fingerprint() != previous.draft.history.fingerprint():
                raise AssertionError("prefix history mismatch")
            advance_lineage(chain, record["decisions"], config)
            check_lineage(previous, chain, config)
            compare(previous.state(), record["end_state"])
            compare(memory_state(memory), record["end_memory"])
            prefix_count += 1
            expected_folders = {f"block_{i:02}" for i in range(config["blocks"])}
            if {p.parent.name for p in folder.glob("block_*/result.json")} != expected_folders:
                raise AssertionError("block record set mismatch")
            if {p.parent.name for p in folder.glob("fresh_*/result.json")} != {"fresh_1", "fresh_2", "fresh_3"}:
                raise AssertionError("fresh record set mismatch")
            for block in schedule(seed)[:config["blocks"]]:
                target = folder/f"block_{block.index:02}"
                before, start_record = load(target/"before.pt")
                after, record = load(target/"checkpoint.pt")
                compare(record, json.loads((target/"result.json").read_text(encoding="utf-8")))
                if signature(before) != signature(previous):
                    raise AssertionError("continuous trajectory reset or start-state mismatch")
                if digest(record["block"]) != digest(asdict(block)) or record["seed"] != seed or record["model"] != model:
                    raise AssertionError("wrong block metadata")
                count = config["episode_size"]//config["batch_size"]
                pairs = [experiences(block, seed, i, config) for i in range(count)]
                data, clean = [p[0] for p in pairs], [p[1] for p in pairs]
                for reported, truth in pairs:
                    if not np.array_equal(reported.observations, truth.observations) or not np.array_equal(reported.actions, truth.actions):
                        raise AssertionError("clean/noisy paired observations/actions differ")
                if record["batches_done"] != count or record["batch_sha256"] != [d.fingerprint() for d in data] or record["clean_sha256"] != [d.fingerprint() for d in clean]:
                    raise AssertionError("block stream does not regenerate")
                arrivals = list(range(config["batch_size"], config["episode_size"]+1, config["batch_size"]))
                if [r["arrivals"] for r in record["prequential"]] != arrivals:
                    raise AssertionError("prequential record missing a packet")
                for row in record["prequential"]:
                    for label in ("truth_brier", "reported_brier"):
                        if set(row[label]) != set(POLICIES) or any(not np.isfinite(v) or not 0 <= v <= 1 for v in row[label].values()):
                            raise AssertionError("invalid prequential scores")
                first_signature = signature(before)
                first_predictions = {p: learner.predict(data[0].observations, before.draft.history)[0]
                                     for p, learner in before.predictors().items()}
                first_proposal = before.draft if before.proposal is None else before.proposal
                proposed = first_proposal.predict(data[0].observations, before.draft.history)[0]
                rows, actions = np.arange(len(data[0])), data[0].actions
                feedback = data[0].survival[:, -1].astype(float)
                proposed_loss = (proposed[rows, actions, -1]-feedback)**2
                saved_gains = record["decisions"][0]["gains"] if record["decisions"] else after.gains
                offset = before.batches % config["evidence_window"]
                for policy in DELAYED:
                    gained = float(((first_predictions[policy][rows, actions, -1]-feedback)**2-proposed_loss).mean())
                    compare(gained, saved_gains[policy][offset])
                for label, outcome in (("truth_brier", clean[0]), ("reported_brier", data[0])):
                    expected_scores = {p: float(np.square(v[np.arange(len(outcome)), outcome.actions]
                                                         -outcome.survival).mean())
                                       for p, v in first_predictions.items()}
                    compare(expected_scores, record["prequential"][0][label])
                if signature(before) != first_signature:
                    raise AssertionError("pre-feedback scoring mutated learning state")
                if record["decisions"]:
                    expected_proposal = (before.proposal.sha256 if before.proposal is not None else
                                         state_hash(before.draft.model.state_dict()))
                    if record["decisions"][0]["proposal_hash"] != expected_proposal:
                        raise AssertionError("first window did not validate the saved pre-feedback proposal")
                if [r["arrivals"] for r in record["curve"]] != list(range(0, config["episode_size"]+1, config["probe_every"])):
                    raise AssertionError("incomplete probe curve")
                memory, history = advance_memory(before.draft.memory, before.draft.history, data, config)
                compare(memory_state(memory), memory_state(after.draft.memory))
                if history.fingerprint() != after.draft.history.fingerprint():
                    raise AssertionError("noncausal end history")
                compare(before.state(), record["start_state"])
                compare(after.state(), record["end_state"])
                compare(memory_state(before.draft.memory), record["start_memory"])
                compare(memory_state(after.draft.memory), record["end_memory"])
                compare(start_record["curve"][0], record["curve"][0])
                initial_state, final_state = signature(before), signature(after)
                compare(evaluate(before, evaluator, block), record["curve"][0]["policies"])
                compare(evaluate(after, evaluator, block), record["curve"][-1]["policies"])
                compare(valid_panel(before, evaluator, block), record["valid_before"])
                compare(valid_panel(after, evaluator, block), record["valid_after"])
                compare(knowledge_panel(after, evaluator, seed, block.index), record["knowledge_after"])
                if signature(before) != initial_state or signature(after) != final_state:
                    raise AssertionError("evaluation mutated learning state")
                advance_lineage(chain, record["decisions"], config)
                check_lineage(after, chain, config)
                if block.index < 3:
                    fresh_folder = folder/f"fresh_{block.index+1}"
                    fresh_before, fresh_start = load(fresh_folder/"before.pt", "learner")
                    fresh_after, fresh = load(fresh_folder/"checkpoint.pt", "learner")
                    compare(fresh, json.loads((fresh_folder/"result.json").read_text(encoding="utf-8")))
                    if any(fresh[k] != v for k, v in dict(seed=seed, model=model, stage=block.index+1,
                            arm="fresh", branch="novel", cue=block.cue, batches_done=count).items()) or digest(
                            fresh["law"]) != digest(asdict(block.law)):
                        raise AssertionError("fresh reference metadata mismatch")
                    expected = new_learner(model, seed, config, str(fresh_before.device))
                    expected.history = copy.deepcopy(before.draft.history)
                    if learner_signature(expected) != learner_signature(fresh_before):
                        raise AssertionError("fresh reference is not original weights with matched history")
                    if fresh["batch_sha256"] != record["batch_sha256"]:
                        raise AssertionError("fresh/current observations do not match")
                    memory, history = advance_memory(fresh_before.memory, fresh_before.history, data, config)
                    compare(memory_state(memory), memory_state(fresh_after.memory))
                    if history.fingerprint() != fresh_after.history.fingerprint():
                        raise AssertionError("fresh reference history mismatch")
                    check_budget(fresh_before, 0, config)
                    check_budget(fresh_after, count, config)
                    compare(fresh_before.diagnostics(), fresh["start_diagnostics"])
                    compare(fresh_after.diagnostics(), fresh["diagnostics"])
                    compare(memory_state(fresh_before.memory), fresh["start_memory"])
                    compare(memory_state(fresh_after.memory), fresh["end_memory"])
                    sig_before, sig_after = learner_signature(fresh_before), learner_signature(fresh_after)
                    compare(evaluator.probe(fresh_before, block.law, block.cue), fresh["before"])
                    compare(evaluator.probe(fresh_after, block.law, block.cue, marginal=marginal(fresh)), fresh["after"])
                    compare(evaluator.valid_panel(fresh_before, block.law), fresh["valid_before"])
                    compare(evaluator.valid_panel(fresh_after, block.law), fresh["valid_after"])
                    compare(evaluator.evaluate(fresh_before, block.law, fresh_before.history, block.cue),
                            fresh["curve"][0]["metrics"])
                    compare(evaluator.evaluate(fresh_after, block.law, fresh_after.history, block.cue,
                                               marginal=marginal(fresh)), fresh["curve"][-1]["metrics"])
                    compare(fresh_start["before"], fresh["before"])
                    if learner_signature(fresh_before) != sig_before or learner_signature(fresh_after) != sig_after:
                        raise AssertionError("fresh probes mutated learning state")
                    fresh_checks.append(dict(seed=seed, model=model, stage=block.index+1))
                block_checks.append(dict(seed=seed, model=model, block=block.index,
                    raw_and_clean_regenerated=True, causal_memory_verified=True,
                    endpoint_probes_recomputed=True, first_packet_prequential_recomputed=True,
                    first_packet_gate_gain_recomputed=True, gate_lineage_verified=True))
                previous = after
            job = next(j for j in completion["jobs"] if (j["seed"], j["model"]) == (seed, model))
            compare(previous.state(), job["final_state"])
            windows += chain["end"]//config["evidence_window"]
            print(f"audited {model} seed={seed}: {config['blocks']} blocks", flush=True)
    result = dict(passed=True, identity=identity, prefix_checkpoints=prefix_count,
        block_before_checkpoints=len(block_checks), block_final_checkpoints=len(block_checks),
        fresh_before_checkpoints=len(fresh_checks), fresh_final_checkpoints=len(fresh_checks),
        decision_windows=windows, block_checks=block_checks, fresh_checks=fresh_checks,
        elapsed_seconds=time.perf_counter()-started,
        scope="Exact-source local checkpoint, stream, replay, budget, first-packet prequential, endpoint-probe and gate-arithmetic/lineage audit. "
              "Intermediate window weights are not saved: individual gate gains and every prequential loss are not independently rescored or retrained.")
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.input, args.output)
