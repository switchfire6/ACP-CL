"""Independently verify exported predictive-value evidence using NumPy only.

This reads no live source files, imports no training/analysis modules and loads
no pickles. It checks portable arithmetic and the archived tensor-audit coverage;
it neither retrains the ordinary trajectory nor independently recreates physics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np


MODELS = ("conditional", "recurrent")
PANELS = ("near_reapplied", "near_frozen", "return_reapplied", "return_frozen")
METRICS = ("brier", "survival", "early_brier", "late_brier")
SELECTORS = ("value", "accuracy", "uniform", "replacement", "oracle")
CONTRASTS = {
    "value_minus_uniform": ("value", "uniform"),
    "value_minus_accuracy": ("value", "accuracy"),
    "accuracy_minus_uniform": ("accuracy", "uniform"),
    "value_minus_replacement": ("value", "replacement"),
    "accuracy_minus_replacement": ("accuracy", "replacement"),
    "uniform_minus_replacement": ("uniform", "replacement"),
    "oracle_minus_uniform": ("oracle", "uniform"),
    "value_minus_oracle": ("value", "oracle"),
}
SCIENTIFIC = dict(kind="diagnostic", prefix_blocks=4, prefix_size=1024,
    acquisition_size=8192, recovery_size=1024, score_size=512, validation_size=512,
    gaps=[512, 4096], batch_size=32, candidate_count=8, rehearsal_updates=12,
    width=64, experts=4, context_width=12, decoder_width=64, interaction_features=True,
    lr=.002, memory_packets=16, updates_per_batch=12, evidence_strength=1.)
EPSILON = 8*np.finfo(np.float32).eps


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite portable value")


def decode(contents):
    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, "duplicate JSON object key")
            result[name] = value
        return result

    value = json.loads(contents, object_pairs_hook=pairs)
    finite_tree(value)
    return value


def read(path):
    return decode(Path(path).read_text(encoding="utf-8"))


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def original_json(value):
    return (json.dumps(value, indent=2, allow_nan=False)+"\n").encode("utf-8")


def valid_hash(value):
    require(isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value), "invalid SHA256")


def relative_path(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value,
            "invalid archive path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts
            and value == path.as_posix() and value != ".", "archive path escapes or aliases archive")
    return value


def compare(actual, expected, location="value"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"different keys: {location}")
        for name in expected:
            compare(actual[name], expected[name], f"{location}.{name}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"different list length: {location}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{location}[{index}]")
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), f"numeric mismatch: {location}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"value mismatch: {location}")


def close(actual, expected, label):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=0, atol=1e-14), f"arithmetic mismatch: {label}")


def array(value, shape, label, probabilities=False):
    result = np.asarray(value, dtype=float)
    lower, upper = (-EPSILON, 1+EPSILON) if probabilities else (0, 1)
    require(result.shape == shape and np.isfinite(result).all()
            and np.all(result >= lower) and np.all(result <= upper), f"invalid {label} array")
    return result


def actions(value, size):
    require(isinstance(value, list) and len(value) == size
            and all(type(item) is int and 0 <= item < 5 for item in value), "invalid action array")
    return np.asarray(value, dtype=int)


def verify_artifacts(directory):
    directory = Path(directory).resolve()
    expected = read(directory/"artifact_manifest.json")
    present = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
               if path.is_file() and path != directory/"artifact_manifest.json"}
    require(isinstance(expected, dict) and set(expected) == present, "portable artifact file set changed")
    for name, entry in expected.items():
        path = (directory/relative_path(name)).resolve()
        require(path.is_relative_to(directory), "artifact path escapes archive")
        content = path.read_bytes()
        compare(entry, dict(bytes=len(content), sha256=sha(content)), f"artifact {name}")
    return len(expected)


def verify_zip(path, expected):
    for name, value in expected.items():
        relative_path(name)
        valid_hash(value)
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected)
                and archive.testzip() is None, "locked ZIP file set or CRC mismatch")
        for name, value in expected.items():
            require(sha(archive.read(name)) == value, f"locked ZIP hash mismatch: {name}")


def schedule(seed, config):
    cue = tuple(permutations(range(3)))[seed % 6][0]
    phases = []

    def append(name, mode, active, size, kind="train", assessment=None, gap=None):
        phases.append(dict(name=name, law=dict(mode=mode, active=active, revised=False, noise=0.),
            size=size, kind=kind, assessment=assessment, gap=gap, cue=cue))

    for index in range(config["prefix_blocks"]):
        append(f"prefix_{index}", (seed+index) % 2, [], config["prefix_size"])
    append("acquisition", 1-seed % 2, [cue], config["acquisition_size"])
    gaps = config["gaps"][::-1] if (seed//6) % 2 else config["gaps"]
    for index, gap in enumerate(gaps):
        append(f"score_{index}", 1-seed % 2, [cue], config["score_size"], "score", index, gap)
        append(f"near_{index}", 1-seed % 2, [cue], config["validation_size"], "near", index, gap)
        append(f"gap_{index}", 1-seed % 2, [cue], gap, "train", index, gap)
        append(f"return_{index}", seed % 2, [], config["validation_size"], "return", index, gap)
        if index == 0:
            append("recovery", 1-seed % 2, [cue], config["recovery_size"])
    return phases


def expected_counts(config):
    seeds, models = config["seeds"], config["models"]
    require(seeds and models and len(set(seeds)) == len(seeds) and len(set(models)) == len(models)
            and all(type(seed) is int and seed >= 0 for seed in seeds)
            and set(models) <= set(MODELS), "invalid configured jobs")
    require(type(config["candidate_count"]) is int and config["candidate_count"] >= 2
            and config["batch_size"] > 0 and config["validation_size"] >= 2*config["batch_size"],
            "invalid candidate or validation dimensions")
    require(len(config["gaps"]) == 2 and 0 < config["gaps"][0] < config["gaps"][1], "invalid gaps")
    for phase in schedule(seeds[0], config):
        require(phase["size"] > 0 and phase["size"] % config["batch_size"] == 0, "invalid phase exposure")
    trajectories = len(seeds)*len(models)
    return dict(trajectories=trajectories, assessments=2*trajectories,
        ordinary_arrivals_per_trajectory=sum(phase["size"] for phase in schedule(seeds[0], config)),
        diagnostic_forks=6*(config["candidate_count"]+1)*trajectories)


def expected_paths(config):
    result = {}
    for model in config["models"]:
        for seed in config["seeds"]:
            prefix = f"{model}_{seed}"
            result[f"{prefix}/result.json"] = "trajectory"
            for phase in schedule(seed, config):
                result[f"{prefix}/{phase['name']}/result.json"] = "phase"
            for index in (0, 1):
                result[f"{prefix}/assessment_{index}/result.json"] = "assessment"
                result[f"{prefix}/assessment_{index}/choice.json"] = "choice"
    return result


def read_records(folder, config, identity, summary):
    expected, records = expected_paths(config), {}
    listed = summary["result_hashes"]
    require(len(listed) == len(expected) and len({item["path"] for item in listed}) == len(listed)
            and {item["path"]: item["kind"] for item in listed} == expected, "original record hash coverage mismatch")

    def put(path, value, kind):
        relative_path(path)
        require(path in expected and expected[path] == kind and path not in records
                and value["identity"] == identity, "unexpected, duplicated or misidentified portable record")
        records[path] = value

    with gzip.open(folder/"raw_results.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            value = decode(line)
            put(f"{value['model']}_{value['seed']}/assessment_{value['index']}/result.json", value, "assessment")
    with gzip.open(folder/"phase_results.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            item = decode(line)
            require(set(item) == {"path", "record"}, "invalid phase record wrapper")
            put(item["path"], item["record"], "phase")
    for kind in ("trajectory", "choice"):
        for path, value in read(folder/f"{kind}_records.json").items():
            put(path, value, kind)
    reconstructed = {path: sha(original_json(value)) for path, value in records.items()}
    require(reconstructed == {item["path"]: item["sha256"] for item in listed},
            "portable records differ from original JSON hashes")
    return records, reconstructed


def verify_phase(record, expected, config, seed, model, start_id, previous_query):
    size, count = config["batch_size"], config["candidate_count"]+1
    require(record["seed"] == seed and record["model"] == model and record["phase"] == expected,
            "phase identity or schedule mismatch")
    packets = record["packets"]
    require(record["completed_packets"] == len(packets) == expected["size"]//size, "incomplete phase packets")
    valid_hash(record["before_signature"])
    valid_hash(record["final_signature"])
    for offset, packet in enumerate(packets):
        require(packet["id"] == start_id+offset
                and packet["original_steps"] == (start_id+offset)*config["updates_per_batch"]
                and packet["origin_phase"] == expected["name"] and packet["origin_index"] == offset
                and packet["origin_law"] == expected["law"]
                and packet["support_sha256"] == previous_query, "noncausal packet timeline or support")
        for field in ("query_sha256", "original_prediction_sha256", "original_model_sha256"):
            valid_hash(packet[field])
        probabilities = array(packet["probabilities"], (size, 3), "prequential probabilities", True)
        outcomes = array(packet["outcomes"], (size, 3), "performed outcomes")
        actions(packet["actions"], size)
        close(packet["original_brier"], np.square(probabilities-outcomes).mean(), "original prequential Brier")
        previous_query = packet["query_sha256"]
    for diagnostics, memory, seen in ((record["start_diagnostics"], record["start_memory"], start_id),
                                    (record["diagnostics"], record["memory"], start_id+len(packets))):
        require(diagnostics["cost"]["arrivals"] == seen*size
                and diagnostics["cost"]["optimizer_steps"] == seen*config["updates_per_batch"], "ordinary cost mismatch")
        require(memory["type"] == "ReservoirMemory" and memory["seen"] == memory["age"] == seen
                and memory["capacity"] == config["memory_packets"]
                and len(memory["ids"]) == len(set(memory["ids"])) == min(seen, config["memory_packets"])
                and all(type(i) is int and 0 <= i < seen for i in memory["ids"]), "ordinary reservoir mismatch")
    require(set(record["metadata"]) == {str(i) for i in record["memory"]["ids"]}, "unbounded or missing metadata")
    kind = expected["kind"]
    scoring = kind in ("score", "near", "return")
    require(len(record["evaluations"]) == (len(packets) if scoring else 0)
            and len(record["shadow_work"]) == (count if scoring else 0)
            and record["frozen_work"] == [], "diagnostic coverage mismatch")
    verify_work(record["shadow_work"], config, record["before_signature"])
    for packet, evaluation in zip(packets, record["evaluations"]):
        require(evaluation["packet_id"] == packet["id"]
                and evaluation["query_sha256"] == packet["query_sha256"]
                and evaluation["support_sha256"] == packet["support_sha256"], "unmatched diagnostic query or support")
        outcomes = np.asarray(packet["outcomes"], dtype=float)
        groups = ("reapplied",) if kind == "score" else ("reapplied", "frozen")
        permitted = {"packet_id", "query_sha256", "support_sha256", *groups}
        if kind != "score":
            permitted.add("truth")
            truth = array(evaluation["truth"], (size, 5, 3), "counterfactual truth")
            close(outcomes, truth[np.arange(size), actions(packet["actions"], size)], "reported outcomes versus physical truth")
        require(set(evaluation) == permitted, "diagnostic prediction group mismatch")
        for group in groups:
            require(len(evaluation[group]) == count, "missing diagnostic predictor")
            for prediction in evaluation[group]:
                valid_hash(prediction["prediction_sha256"])
                probabilities = array(prediction["probabilities"], (size, 3), "shadow probabilities", True)
                close(prediction["brier"], np.square(probabilities-outcomes).mean(), "shadow Brier")
                if kind != "score":
                    chosen = actions(prediction["actions"], size)
                    close(prediction["survival"], truth[np.arange(size), chosen, -1].mean(), "reported greedy-action survival")
    if kind == "score":
        close(record["score_losses"], np.mean([[p["brier"] for p in row["reapplied"]]
              for row in record["evaluations"]], axis=0), "score mean")
    elif scoring:
        require(set(record["panels"]) == {"reapplied", "frozen"}, "validation panel coverage mismatch")
        for group in ("reapplied", "frozen"):
            brier = np.asarray([[p["brier"] for p in row[group]] for row in record["evaluations"]])
            survival = np.asarray([[p["survival"] for p in row[group]] for row in record["evaluations"]])
            for field, values in dict(brier=brier.mean(axis=0), survival=survival.mean(axis=0),
                                     early_brier=brier[0], late_brier=brier[1:].mean(axis=0)).items():
                close(record["panels"][group][field], values, f"{group}/{field}")
    return previous_query


def verify_work(work, config, parent_signature):
    size, updates = config["batch_size"], config["rehearsal_updates"]
    for item in work:
        for name, value in dict(steps=updates, forwards=2*updates, backward_calls=updates,
                query_presentations=2*updates*size, support_presentations=2*updates*size).items():
            require(type(item[name]) is int and item[name] == value, "unmatched rehearsal work")
        require(item["parent_unchanged"] is True and item["memory_history_unchanged"] is True
                and item["parent_state_sha256"] == parent_signature, "rehearsal parent mutation or mismatch")
        require(type(item["training_seconds"]) in (int, float) and item["training_seconds"] >= 0,
                "invalid rehearsal wall time")
        fields = ("model_bytes", "optimizer_bytes", "gradient_bytes", "initial_encoder_bytes", "replay_bytes", "history_bytes")
        require(all(type(item[name]) is int and item[name] >= 0 for name in fields), "invalid explicit storage")
        require(item["explicit_tensor_bytes_subtotal"] == sum(item[name] for name in fields)
                and "not measured peak RAM" in item["explicit_tensor_bytes_scope"], "invalid storage subtotal or scope")
        for name in ("anchor_packet_sha256", "replay_packet_sha256", "initial_model_sha256", "final_model_sha256",
                     "initial_optimizer_sha256", "final_optimizer_sha256", "shadow_state_sha256"):
            valid_hash(item[name])
    for name in ("anchor_packet_sha256", "initial_model_sha256", "initial_optimizer_sha256", "parent_state_sha256"):
        require(len({item[name] for item in work}) <= 1, "diagnostic forks started from different states")


def candidate_sample(reservoir_ids, anchor_id, count, seed, index):
    pool = sorted(set(reservoir_ids)-{anchor_id})
    require(len(pool) >= count+1, "insufficient candidate capacity")
    key = int(digest([seed, "predictive_value_candidates", index])[:15], 16)
    selected = np.random.default_rng(key).choice(pool, count+1, replace=False).tolist()
    return sorted(selected[:-1]), int(selected[-1])


def selected_indices(record):
    candidates, choice = record["candidates"], record["choice"]
    ids = [candidate["id"] for candidate in candidates]
    require(len(ids) >= 2 and len(ids) == len(set(ids))
            and record["replacement"]["id"] not in ids, "duplicate or insufficient candidate IDs")
    losses = array(choice["score_losses"], (len(ids)+1,), "score losses", True)
    values = losses[-1]-losses[:-1]
    close(choice["values"], values, "common replacement values")
    value = min(range(len(ids)), key=lambda i: (-values[i], ids[i]))
    accuracy = min(range(len(ids)), key=lambda i: (candidates[i]["original_brier"], ids[i]))
    require(choice["value_id"] == ids[value] and choice["accuracy_id"] == ids[accuracy], "sealed choice differs from causal rule")
    return value, accuracy


def verify_timelines(records, hashes, config, identity, affinity):
    assessments = []
    for model in config["models"]:
        for seed in config["seeds"]:
            prefix, phases, packets = f"{model}_{seed}", {}, {}
            trajectory = records[f"{prefix}/result.json"]
            planned = schedule(seed, config)
            names = [phase["name"] for phase in planned]
            require((trajectory["seed"], trajectory["model"]) == (seed, model)
                    and trajectory["affinity"] == affinity and trajectory["phase_names"] == names
                    and trajectory["phase_hashes"] == {name: hashes[f"{prefix}/{name}/result.json"] for name in names}
                    and trajectory["assessment_hashes"] == {str(i): hashes[f"{prefix}/assessment_{i}/result.json"] for i in (0, 1)},
                    "trajectory coverage or linked hashes mismatch")
            previous_query, previous_signature, start_id = None, None, 0
            for phase in planned:
                record = records[f"{prefix}/{phase['name']}/result.json"]
                if previous_signature is not None:
                    require(record["before_signature"] == previous_signature, "broken ordinary state chain")
                previous_query = verify_phase(record, phase, config, seed, model, start_id, previous_query)
                packets.update({packet["id"]: packet for packet in record["packets"]})
                for packet_id, metadata in record["metadata"].items():
                    compare(metadata, {key: value for key, value in packets[int(packet_id)].items()
                            if key not in {"probabilities", "actions", "outcomes"}}, "bounded original metadata")
                previous_signature = record["final_signature"]
                start_id += len(record["packets"])
                phases[phase["name"]] = record
            require(trajectory["final_signature"] == previous_signature
                    and trajectory["diagnostics"] == record["diagnostics"], "trajectory final state mismatch")
            compare(trajectory["prequential_work"], dict(forwards=start_id,
                query_presentations=start_id*config["batch_size"], support_presentations=start_id*config["batch_size"]),
                "ordinary prequential cost")
            for index in (0, 1):
                record = records[f"{prefix}/assessment_{index}/result.json"]
                seal = records[f"{prefix}/assessment_{index}/choice.json"]
                score = phases[f"score_{index}"]
                require((record["seed"], record["model"], record["index"]) == (seed, model, index)
                        and record["gap"] == score["phase"]["gap"] and record["cue"] == score["phase"]["cue"]
                        and record["anchor_id"] == score["start_memory"]["seen"]-1
                        and record["reservoir_ids"] == score["start_memory"]["ids"]
                        and record["parent_steps"] == score["start_diagnostics"]["cost"]["optimizer_steps"],
                        "assessment state or schedule mismatch")
                candidates, replacement = candidate_sample(record["reservoir_ids"], record["anchor_id"],
                    config["candidate_count"], seed, index)
                require([item["id"] for item in record["candidates"]] == candidates
                        and record["replacement"]["id"] == replacement, "candidate randomization mismatch")
                for candidate in [*record["candidates"], record["replacement"]]:
                    require(candidate["id"] < record["anchor_id"], "candidate was not previously observed")
                    compare(candidate, {key: value for key, value in packets[candidate["id"]].items()
                            if key not in {"probabilities", "actions", "outcomes"}}, "original candidate provenance")
                selected_indices(record)
                close(record["choice"]["score_losses"], score["score_losses"], "selection score window")
                require(seal == dict(identity=identity, seed=seed, model=model, index=index,
                        score_record_sha256=digest(score), choice=record["choice"]), "choice seal mismatch")
                require(record["phase_hashes"] == {f"{name}_{index}": hashes[f"{prefix}/{name}_{index}/result.json"]
                        for name in ("score", "near", "gap", "return")}, "assessment phase hash mismatch")
                require(set(record["panels"]) == set(PANELS), "missing assessment panels")
                for window in ("near", "return"):
                    for group in ("reapplied", "frozen"):
                        compare(record["panels"][f"{window}_{group}"], phases[f"{window}_{index}"]["panels"][group],
                                "assessment panel provenance")
                for window in ("score", "near", "return"):
                    compare(record["work"][window], phases[f"{window}_{index}"]["shadow_work"], "assessment work provenance")
                for position in range(config["candidate_count"]+1):
                    require(len({record["work"][window][position]["replay_packet_sha256"]
                                 for window in ("score", "near", "return")}) == 1, "reapplied packet changed")
                compare(record["prediction_work"], dict(
                    score_shadow_forwards=(config["candidate_count"]+1)*config["score_size"]//config["batch_size"],
                    validation_shadow_forwards=4*(config["candidate_count"]+1)*config["validation_size"]//config["batch_size"],
                    query_records_per_forward=config["batch_size"], support_records_per_forward=config["batch_size"]),
                    "diagnostic prediction cost")
                assessments.append(record)
    return assessments


def bootstrap(values, seeds):
    values = np.asarray(values, dtype=float)
    require(values.shape == (len(seeds),) and len(seeds) > 0 and len(set(seeds)) == len(seeds)
            and np.isfinite(values).all(), "invalid independent seed values")
    values = np.where(np.abs(values) < 1e-12, 0., values)
    draws = np.random.default_rng(27192026).integers(0, len(values), (20000, len(values)))
    samples = values[draws].mean(axis=1)
    return dict(mean=float(values.mean()), lower=float(np.quantile(samples, .025)), upper=float(np.quantile(samples, .975)),
        positive=int((values > 0).sum()), negative=int((values < 0).sum()), n=len(values),
        differences=values.tolist(), seeds=list(seeds))


def selectors(record):
    value, accuracy = selected_indices(record)
    count = len(record["candidates"])
    ids = [item["id"] for item in record["candidates"]]
    result = {}
    for panel, data in {"score": dict(brier=record["choice"]["score_losses"]), **record["panels"]}.items():
        oracle = min(range(count), key=lambda i: (data["brier"][i], ids[i]))
        metrics = ("brier",) if panel == "score" else METRICS
        chosen = {"value": value, "accuracy": accuracy, "replacement": count, "oracle": oracle}
        result[panel] = {name: {metric: float(data[metric][position]) for metric in metrics}
                         for name, position in chosen.items()}
        result[panel]["uniform"] = {metric: float(np.mean(data[metric][:count])) for metric in metrics}
    return result


def scientific(config):
    return (config["seeds"] == list(range(14001, 14013)) and set(config["models"]) == set(MODELS)
            and all(config.get(name) == value for name, value in SCIENTIFIC.items()))


def screen(contrasts, eligible):
    brier = contrasts["value_minus_uniform"]["brier"]
    survival = contrasts["value_minus_uniform"]["survival"]
    accuracy = contrasts["value_minus_accuracy"]["brier"]
    improved = sum(value < 0 for value in brier["differences"])
    gates = dict(complete_scientific_cohort=dict(passed=bool(eligible)),
        brier_gain=dict(passed=brier["mean"] <= -.0005, value=brier["mean"], maximum=-.0005),
        seed_consistency=dict(passed=improved >= 9, improved_seeds=improved, minimum=9, n=brier["n"]),
        accuracy_control=dict(passed=accuracy["mean"] <= 0, value=accuracy["mean"], maximum=0.),
        survival=dict(passed=survival["mean"] >= -.005, value=survival["mean"], minimum=-.005))
    return dict(passed=all(row["passed"] for row in gates.values()), gates=gates,
                failed=[name for name, row in gates.items() if not row["passed"]])


def recompute_statistics(records, config):
    groups = defaultdict(list)
    keys = {(seed, model, index) for seed in config["seeds"] for model in config["models"] for index in (0, 1)}
    observed = [(row["seed"], row["model"], row["index"]) for row in records]
    require(len(observed) == len(set(observed)) and set(observed) == keys, "independent seed/assessment coverage mismatch")
    for record in records:
        groups[record["model"], record["seed"]].append(selectors(record))
    contrasts, means, screens = {}, {}, {}
    for model in config["models"]:
        contrasts[model], means[model] = {}, {}
        for panel in ("score", *PANELS):
            metrics = ("brier",) if panel == "score" else METRICS
            means[model][panel] = {selector: {metric: float(np.mean([
                np.mean([row[panel][selector][metric] for row in groups[model, seed]])
                for seed in config["seeds"]])) for metric in metrics} for selector in SELECTORS}
            contrasts[model][panel] = {}
            for name, (left, right) in CONTRASTS.items():
                contrasts[model][panel][name] = {metric: bootstrap([
                    float(np.mean([row[panel][left][metric]-row[panel][right][metric]
                                   for row in groups[model, seed]])) for seed in config["seeds"]], config["seeds"])
                    for metric in metrics}
        screens[model] = {window: screen(contrasts[model][f"{window}_reapplied"], scientific(config))
                          for window in ("near", "return")}
    return contrasts, means, screens


def verify_statistics(summary, assessments, config):
    contrasts, means, screens = recompute_statistics(assessments, config)
    compare(summary["contrasts"], contrasts, "paired seed statistics")
    compare(summary["selector_means"], means, "selector means")
    compare(summary["eligible_scientific_cohort"], scientific(config), "scientific cohort eligibility")
    for model in config["models"]:
        for window in ("near", "return"):
            stored = summary["screen"][model][window]
            compare({name: stored[name] for name in ("passed", "gates", "failed")}, screens[model][window], "allocation screen")
        near, returned = (screens[model][window]["passed"] for window in ("near", "return"))
        recommendation = ("eligible_for_separately_specified_recurrence_policy" if near and returned
            else "eligible_for_separately_specified_local_policy_only" if near
            else "return_only_signal_does_not_pass_local_tier" if returned else "stop_expansion_of_fixed_score_recipe")
        compare(summary["screen"][model]["recommendation"], recommendation, "research allocation recommendation")
    inference = summary["inference"]
    require(inference["bootstrap_seed"] == 27192026 and inference["bootstrap_resamples"] == 20000
            and inference["count_per_seed"] == 2, "inference specification changed")
    return screens


def verify_audit(audit, config, identity, hashes):
    require(audit["passed"] is True and audit["identity"] == identity, "failed or unrelated tensor audit")
    count = expected_counts(config)
    phase_count = config["prefix_blocks"]+10
    packets = count["ordinary_arrivals_per_trajectory"]//config["batch_size"]
    forks = 6*(config["candidate_count"]+1)
    predictions = 2*(config["candidate_count"]+1)*(config["score_size"]+4*config["validation_size"])//config["batch_size"]
    expected = dict(trajectories=count["trajectories"], before_checkpoints=phase_count*count["trajectories"],
        final_checkpoints=phase_count*count["trajectories"], assessments=count["assessments"],
        diagnostic_forks_reproduced=forks*count["trajectories"],
        diagnostic_prediction_calls_recomputed=predictions*count["trajectories"])
    for name, value in expected.items():
        compare(audit[name], value, f"audit {name}")
    checks = audit["checks"]
    require(len(checks) == count["trajectories"] and {(r["seed"], r["model"]) for r in checks}
            == {(seed, model) for seed in config["seeds"] for model in config["models"]}, "audit job coverage mismatch")
    for check in checks:
        for name, value in dict(before_checkpoints=phase_count, final_checkpoints=phase_count, assessments=2,
                ordinary_packets=packets, original_packet_errors_recomputed=packets,
                original_predictions_recomputed=phase_count, diagnostic_forks_reproduced=forks,
                diagnostic_prediction_calls_recomputed=predictions).items():
            compare(check[name], value, f"audit job {name}")
        planned = schedule(check["seed"], config)
        require([row["phase"] for row in check["phases"]] == [p["name"] for p in planned], "audit phase coverage mismatch")
        cumulative = 0
        for row, phase in zip(check["phases"], planned):
            size = phase["size"]//config["batch_size"]
            cumulative += size
            scoring = phase["kind"] in ("score", "near", "return")
            require(row["packets"] == size and row["cumulative_packets"] == cumulative
                    and row["diagnostic_predictions_recomputed"] == (size if scoring else 0)
                    and row["rehearsal_forks_reproduced"] == (config["candidate_count"]+1 if scoring else 0), "audit phase work mismatch")
            require(all(row[name] is True for name in ("causal_data_replay_metadata_verified", "exact_start_chain_verified",
                    "adam_counters_verified", "start_prediction_recomputed")), "missing tensor audit assertions")
            path = f"{check['model']}_{check['seed']}/{phase['name']}/result.json"
            require(row["result_sha256"] == hashes[path], "tensor audit record hash mismatch")
            for name in ("before_sha256", "checkpoint_sha256"):
                valid_hash(row[name])
        valid_hash(check["input_stream_sha256"])
    return expected


def verify(directory):
    folder = Path(directory)
    artifact_count = verify_artifacts(folder)
    manifest, config, summary = read(folder/"manifest.json"), read(folder/"config.json"), read(folder/"summary.json")
    identity = manifest["identity"]
    compare(manifest["config"], config, "manifest configuration")
    compare(summary["config"], config, "summary configuration")
    compare(summary["identity"], identity, "summary identity")
    for name, value in (("config_sha256", config), ("source_sha256", manifest["source_files"]), ("runtime_sha256", manifest["runtime"])):
        require(identity[name] == digest(value), "identity digest mismatch")
    verify_zip(folder/"training_source.zip", manifest["source_files"])
    protocol, analysis = read(folder/"protocol_lock.json"), read(folder/"analysis_lock.json")
    require(all(protocol[name] == value for name, value in identity.items())
            and protocol["protocol_sha256"] == sha((folder/"protocol_at_lock.md").read_bytes()), "protocol lock mismatch")
    require(analysis["config"] == config and analysis["config_sha256"] == identity["config_sha256"], "analysis configuration mismatch")
    required = {"scripts/summarize_predictive_value.py", "tests/test_predictive_value_summary.py"}
    require(required <= analysis["files"].keys()
            and any(name.startswith("docs/") and value == protocol["protocol_sha256"] for name, value in analysis["files"].items()),
            "prospective analysis or protocol absent from lock")
    verify_zip(folder/"analysis_at_lock.zip", analysis["files"])
    require(manifest["created_utc"] == protocol["locked_utc"] == analysis["locked_utc"], "lock creation times differ")
    completion = read(folder/"completion.json")
    compare(completion["identity"], identity, "completion identity")
    counts = expected_counts(config)
    compare(summary["counts"], counts, "summary counts")
    for name, value in counts.items():
        compare(completion[name], value, f"completion {name}")
    jobs = completion["jobs"]
    require(len(jobs) == counts["trajectories"] and {(j["seed"], j["model"]) for j in jobs}
            == {(seed, model) for seed in config["seeds"] for model in config["models"]}
            and all(j["affinity"] == manifest["runtime"]["affinity"] for j in jobs), "completion job/affinity coverage mismatch")
    records, hashes = read_records(folder, config, identity, summary)
    assessments = verify_timelines(records, hashes, config, identity, manifest["runtime"]["affinity"])
    screens = verify_statistics(summary, assessments, config)
    audit_counts = verify_audit(read(folder/"audit.json"), config, identity, hashes)
    compare(read(folder/"record_integrity.json"), summary["record_integrity"], "record integrity copy")
    require(summary["record_integrity"]["passed"] is True
            and summary["record_integrity"]["all_expected_assessments"] == counts["assessments"], "record integrity coverage mismatch")
    with (folder/"rows.jsonl").open(encoding="utf-8") as stream:
        compare([decode(line) for line in stream], summary["rows"], "portable descriptive row copy")
    return dict(passed=True, identity=identity, artifact_files=artifact_count, original_records=len(records),
        **counts, tensor_audit_coverage=audit_counts,
        allocation_screens={model: {window: result["passed"] for window, result in tiers.items()} for model, tiers in screens.items()},
        scope="Portable file/lock/identity/coverage checks; original JSON hashes; causal record chains; "
              "performed-action Brier and reported greedy-action survival arithmetic; exact selector means, "
              "paired seed bootstrap and allocation criteria. Full all-action predictions and greedy argmax "
              "are covered by the separately archived tensor audit, whose hashes and coverage are checked. "
              "No model pickle, live source, ordinary retraining or independent physics replication.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.input), indent=2, allow_nan=False))
