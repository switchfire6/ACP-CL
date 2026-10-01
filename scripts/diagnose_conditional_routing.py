"""Post-hoc routing diagnosis on archived models with new, split probe data.

The gate-aware selector is privileged: it knows the gate's sensor location and
is calibrated separately for each true condition. It is a diagnostic only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from acp_cl.conditional.study import source_manifest
from acp_cl.conditional.world import Condition, ConditionalWorld
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest, trial_seed, write_json


def gate_values(world, observations):
    matches = (observations[:, 0, 5:8, 11:14, None] ==
               world.glyphs[2].transpose(1, 2, 0)[None]).all(axis=(1, 2))
    if not np.all(matches.sum(axis=1) == 1):
        raise ValueError("gate alphabet is ambiguous")
    return matches.argmax(axis=1)


def diagnose(directory, output, calibration_size=384, query_size=1024):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if digest(source_manifest()) != manifest["identity"]["source_sha256"]:
        raise ValueError("archived model code differs from current imported code")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    world, rows = ConditionalWorld(), []
    for seed in manifest["config"]["seeds"]:
        checkpoint = torch.load(directory / f"recurring_conditional_{seed}" / "checkpoint.pt",
                                map_location="cpu", weights_only=False)
        if checkpoint["identity"] != manifest["identity"]:
            raise ValueError("checkpoint identity mismatch")
        learner = checkpoint["learner"]
        before = state_hash(learner.model.state_dict())
        for mode in (0, 1):
            for gated in (False, True):
                condition = Condition(mode, gated)
                tag = asdict(condition)
                calibration, _ = world.experience(condition, calibration_size,
                    trial_seed(seed, "routing_calibration", tag))
                trials = world.dataset(condition, query_size,
                    trial_seed(seed, "routing_query", tag))
                truth = world.counterfactuals(trials.cases)
                _, _, calibrated = learner.predict(calibration.observations)
                _, _, predictions = learner.predict(trials.cases.observations)
                predicted = calibrated[np.arange(calibration_size), :, calibration.actions, -1]
                p = predicted.clip(1e-6, 1 - 1e-6)
                y = calibration.survival[:, -1, None]
                losses = -(y*np.log(p) + (1-y)*np.log1p(-p))
                global_head = int(losses.mean(axis=0).argmin())
                calibration_gate = gate_values(world, calibration.observations)
                query_gate = gate_values(world, trials.cases.observations)
                head_for_gate = {int(g): int(losses[calibration_gate == g].mean(axis=0).argmin())
                                 for g in np.unique(calibration_gate)}
                selected = np.asarray([head_for_gate[int(g)] for g in query_gate])

                def survival(probs):
                    actions = probs[:, :, -1].argmax(axis=1)
                    return float(truth[np.arange(query_size), actions, -1].mean())

                ordinary = []
                for replicate in range(4):
                    support, _ = world.experience(condition, 32,
                        trial_seed(seed, "routing_support", [tag, replicate]))
                    ordinary.append(survival(learner.predict(trials.cases.observations, support)[0]))
                rows.append(dict(seed=seed, condition=tag, ordinary=float(np.mean(ordinary)),
                    global_calibrated=survival(predictions[:, global_head]),
                    gate_calibrated=survival(predictions[np.arange(query_size), selected]),
                    global_head=global_head, gate_heads=head_for_gate,
                    calibration_sha256=calibration.fingerprint(), model_sha256=before))
        if state_hash(learner.model.state_dict()) != before:
            raise AssertionError("diagnostic changed archived weights")
    aggregate = {}
    for gated in (False, True):
        subset = [r for r in rows if r["condition"]["gated"] == gated]
        aggregate[str(gated)] = {k: float(np.mean([r[k] for r in subset]))
                                 for k in ("ordinary", "global_calibrated", "gate_calibrated")}
    result = dict(source_identity=manifest["identity"], calibration_size=calibration_size,
                  query_size=query_size, rows=rows, aggregate=aggregate,
                  scope="Exploratory diagnosis, privileged gate/condition grouping; no weight updates; new calibration and query seeds; not a deployable routing method or a causal attribution of all acquisition loss.")
    write_json(output, result)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/conditional_pilot")
    parser.add_argument("--output", default="reports/contextual/routing_diagnosis.json")
    args = parser.parse_args()
    diagnose(args.input, args.output)
