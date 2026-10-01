"""Verify portable rehearsal-state results using only stdlib and NumPy.

No live repository imports, model pickles, retraining or physics replication.
All-action forecasts, paired-context provenance, metric arithmetic and every
prespecified factorial contrast are independently recomputed from exports.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

EPSILON = 8*np.finfo(np.float32).eps
MODELS = ("conditional", "recurrent")
BRANCHES = ("current", "return")
CELLS = tuple(f"{value:03b}" for value in range(8))
FACTORS = ("weights", "optimizer", "anchor")
METRICS = ("brier", "survival", "early_brier", "late_brier")
SELECTORS = ("value", "accuracy", "uniform", "replacement", "oracle")
RESPONSES = ("selection", "chosen", "uniform")
SCIENTIFIC = dict(kind="diagnostic", prefix_blocks=4, prefix_size=1024,
    acquisition_size=8192, inter_assessment_size=4096, score_size=512, validation_size=512,
    batch_size=32, candidate_count=8, rehearsal_updates=12, width=64, experts=4,
    context_width=12, decoder_width=64, interaction_features=True, lr=.002,
    memory_packets=16, updates_per_batch=12, evidence_strength=1.)
CELL_CONTRASTS = {
    "value_minus_uniform": ("value", "uniform"),
    "value_minus_accuracy": ("value", "accuracy"),
    "accuracy_minus_uniform": ("accuracy", "uniform"),
    "value_minus_replacement": ("value", "replacement"),
    "uniform_minus_replacement": ("uniform", "replacement"),
    "oracle_minus_uniform": ("oracle", "uniform"),
    "value_minus_oracle": ("value", "oracle"),
}


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


def candidate_sample(reservoir_ids, anchor_id, count, seed, index):
    pool = sorted(set(reservoir_ids)-{anchor_id})
    require(len(pool) >= count+1, "insufficient candidate capacity")
    key = int(digest([seed, "predictive_value_candidates", index])[:15], 16)
    selected = np.random.default_rng(key).choice(pool, count+1, replace=False).tolist()
    return sorted(selected[:-1]), int(selected[-1])


def schedule(seed, config):
    cue = tuple(permutations(range(3)))[seed % 6][0]
    result = []

    def add(name, mode, active, size, kind="train", assessment=None):
        result.append(dict(name=name, law=dict(mode=mode, active=active, revised=False, noise=0.),
            size=size, kind=kind, assessment=assessment, gap=None, cue=cue))

    for index in range(config["prefix_blocks"]):
        add(f"prefix_{index}", (seed+index) % 2, [], config["prefix_size"])
    add("acquisition", 1-seed % 2, [cue], config["acquisition_size"])
    for index in (0, 1):
        add(f"score_{index}", 1-seed % 2, [cue], config["score_size"], "score", index)
        add(f"validation_{index}", 1-seed % 2, [cue], config["validation_size"], "train", index)
        if index == 0:
            add("inter_assessment", 1-seed % 2, [cue], config["inter_assessment_size"])
    return result


def expected_counts(config):
    seeds, models = config["seeds"], config["models"]
    require(seeds and models and len(set(seeds)) == len(seeds) and len(set(models)) == len(models)
            and all(type(seed) is int and seed >= 0 for seed in seeds)
            and set(models) <= set(MODELS), "invalid configured jobs")
    require(type(config["candidate_count"]) is int and config["candidate_count"] >= 2
            and type(config["batch_size"]) is int and config["batch_size"] > 0
            and config["validation_size"] >= 2*config["batch_size"], "invalid study dimensions")
    for phase in schedule(seeds[0], config):
        require(type(phase["size"]) is int and phase["size"] > 0
                and phase["size"] % config["batch_size"] == 0, "invalid phase exposure")
    trajectories = len(seeds)*len(models)
    assessments, candidates = 2*trajectories, config["candidate_count"]+1
    score, validation = (config[name]//config["batch_size"] for name in ("score_size", "validation_size"))
    return dict(trajectories=trajectories, assessments=assessments,
        ordinary_arrivals_per_trajectory=sum(phase["size"] for phase in schedule(seeds[0], config)),
        diagnostic_forks=assessments*8*candidates, instrumented_forks=assessments*7*candidates,
        zero_update_copies=2*assessments,
        diagnostic_prediction_forwards=assessments*(score*candidates+2*validation*(8*candidates+2)),
        extra_loss_forwards=4*assessments*7*candidates)


def experience_hash(observations, performed_actions, outcomes):
    value = hashlib.sha256()
    for array in (observations, performed_actions, outcomes):
        value.update(str((array.shape, array.dtype.str)).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        require(len(set(archive.files)) == len(archive.files), "duplicate NPZ array names")
        return {name: archive[name] for name in archive.files}


def validate_data(data, config):
    packets, size = config["validation_size"]//config["batch_size"], config["batch_size"]
    shapes = dict(observations=(packets, size, 4, 12, 16), actions=(packets, size),
        outcomes=(2, packets, size, 3), truth=(2, packets, size, 5, 3),
        initial_observations=(size, 4, 12, 16), initial_actions=(size,), initial_outcomes=(size, 3))
    require(set(data) == set(shapes), "paired data array set mismatch")
    for name, shape in shapes.items():
        require(data[name].shape == shape and data[name].dtype == np.uint8, f"paired data shape/dtype mismatch: {name}")
        if "actions" in name:
            require(np.all(data[name] < 5), "invalid performed action")
        elif name in ("outcomes", "truth", "initial_outcomes"):
            require(np.all(data[name] <= 1), "invalid physical outcome")
    for branch in range(2):
        truth = data["truth"][branch]
        actual = truth[np.arange(packets)[:, None], np.arange(size)[None, :], data["actions"]]
        require(np.array_equal(actual, data["outcomes"][branch]), "performed outcomes differ from paired truth")


def metric_panels(probabilities, data, config, count):
    validate_data(data, config)
    packets, size = config["validation_size"]//config["batch_size"], config["batch_size"]
    require(probabilities.shape == (2, packets, count, size, 5, 3)
            and probabilities.dtype == np.float32 and np.isfinite(probabilities).all()
            and np.all(probabilities >= -np.finfo(np.float32).eps)
            and np.all(probabilities <= 1+np.finfo(np.float32).eps), "invalid all-action prediction array")
    result = {}
    for branch_index, branch in enumerate(BRANCHES):
        values = probabilities[branch_index]
        chosen = np.take_along_axis(values, data["actions"][:, None, :, None, None], axis=3).squeeze(3)
        errors = np.square(chosen.astype(np.float64)-data["outcomes"][branch_index, :, None]).mean(axis=(2, 3))
        greedy = values[..., -1].argmax(axis=-1)
        truth = data["truth"][branch_index]
        survival = truth[np.arange(packets)[:, None, None], np.arange(size)[None, None, :], greedy, -1].mean(axis=-1)
        result[branch] = dict(brier=errors.mean(axis=0).tolist(), survival=survival.mean(axis=0).tolist(),
            early_brier=errors[0].tolist(), late_brier=errors[1:].mean(axis=0).tolist())
    return result


def verify_contexts(stream, data, validation, score, config, identity, seed, model, index):
    require(stream["identity"] == identity and (stream["seed"], stream["model"], stream["index"]) == (seed, model, index)
            and stream["branches"] == list(BRANCHES), "paired stream identity or branch order mismatch")
    packets = config["validation_size"]//config["batch_size"]
    require(len(stream["rows"]) == packets == len(validation["packets"]), "paired stream coverage mismatch")
    initial = experience_hash(data["initial_observations"], data["initial_actions"], data["initial_outcomes"])
    require(initial == score["packets"][-1]["query_sha256"], "initial context is not actual preceding history")
    previous = [initial, initial]
    for packet, row in enumerate(stream["rows"]):
        require(set(row) == set(BRANCHES), "stream branch coverage mismatch")
        for branch_index, branch in enumerate(BRANCHES):
            query = experience_hash(data["observations"][packet], data["actions"][packet], data["outcomes"][branch_index, packet])
            require(row[branch] == dict(query_sha256=query, support_sha256=previous[branch_index]),
                    "noncausal, cross-branch or changed evaluation context")
            previous[branch_index] = query
        original = validation["packets"][packet]
        require(original["query_sha256"] == row["current"]["query_sha256"]
                and original["support_sha256"] == row["current"]["support_sha256"], "ordinary current stream differs from diagnostic branch")
        require(np.array_equal(original["actions"], data["actions"][packet])
                and np.array_equal(original["outcomes"], data["outcomes"][0, packet]), "ordinary current feedback differs")


def selected_indices(record):
    ids = [row["id"] for row in record["candidates"]]
    require(len(ids) >= 2 and len(set(ids)) == len(ids) and record["replacement"]["id"] not in ids,
            "candidate pool duplicated or incomplete")
    losses = array(record["choice"]["score_losses"], (len(ids)+1,), "score losses", True)
    values = losses[-1]-losses[:-1]
    close(record["choice"]["values"], values, "common replacement values")
    selected = min(range(len(ids)), key=lambda i: (-values[i], ids[i]))
    accurate = min(range(len(ids)), key=lambda i: (record["candidates"][i]["original_brier"], ids[i]))
    require(record["choice"]["value_id"] == ids[selected] and record["choice"]["accuracy_id"] == ids[accurate],
            "choice differs from score-only or original-error rule")
    return selected, accurate


def coefficients():
    result = {"total_111_minus_000": {"111": 1., "000": -1.}}
    for axis, name in enumerate(FACTORS):
        result[f"{name}_main"] = {cell: (2*int(cell[axis])-1)/4 for cell in CELLS}
        others = [position for position in range(3) if position != axis]
        for bits in product("01", repeat=2):
            pair = [cell for cell in CELLS if all(cell[position] == bit for position, bit in zip(others, bits))]
            label = f"{name}_at_"+"_".join(f"{FACTORS[position]}{bit}" for position, bit in zip(others, bits))
            result[label] = {pair[1]: 1., pair[0]: -1.}
        restored = "111"[:axis]+"0"+"111"[axis+1:]
        result[f"repair_{name}"] = {"111": 1., restored: -1.}
    for left, right in combinations(range(3), 2):
        result[f"{FACTORS[left]}_{FACTORS[right]}_interaction"] = {
            cell: .5*(2*int(cell[left])-1)*(2*int(cell[right])-1) for cell in CELLS}
    result["three_way_interaction"] = {cell: float(np.prod([2*int(bit)-1 for bit in cell])) for cell in CELLS}
    return result


def assessment_metrics(record):
    chosen, accurate = selected_indices(record)
    ids, count = [r["id"] for r in record["candidates"]], len(record["candidates"])
    panels, responses, effects, zeros = {}, {}, {}, {}
    for branch in BRANCHES:
        require(set(record["panels"][branch]) == set(CELLS), "factorial cell set changed")
        panels[branch], responses[branch] = {}, {target: {} for target in RESPONSES}
        zeros[branch] = {str(weight): {metric: float(record["zero"][branch][metric][weight]) for metric in METRICS}
                         for weight in (0, 1)}
        for cell in CELLS:
            source = record["panels"][branch][cell]
            for metric in METRICS:
                array(source[metric], (count+1,), "cell metric", True)
                array(record["zero"][branch][metric], (2,), "zero-reference metric", True)
            oracle = min(range(count), key=lambda i: (source["brier"][i], ids[i]))
            selectors = {name: {metric: float(source[metric][index]) for metric in METRICS}
                         for name, index in (("value", chosen), ("accuracy", accurate), ("replacement", count), ("oracle", oracle))}
            selectors["uniform"] = {metric: float(np.mean(source[metric][:count])) for metric in METRICS}
            contrasts = {name: {metric: selectors[a][metric]-selectors[b][metric] for metric in METRICS}
                         for name, (a, b) in CELL_CONTRASTS.items()}
            for name in ("value", "accuracy", "uniform", "replacement"):
                contrasts[f"{name}_minus_matching_zero"] = {metric: selectors[name][metric]-zeros[branch][cell[0]][metric]
                                                           for metric in METRICS}
            panels[branch][cell] = dict(selectors=selectors, contrasts=contrasts,
                brier_spread=float(max(source["brier"][:count])-min(source["brier"][:count])))
            for target, item in (("selection", contrasts["value_minus_uniform"]), ("chosen", selectors["value"]), ("uniform", selectors["uniform"])):
                responses[branch][target][cell] = item
        effects[branch] = {target: {name: {metric: float(sum(weight*responses[branch][target][cell][metric]
            for cell, weight in weights.items())) for metric in METRICS} for name, weights in coefficients().items()}
            for target in RESPONSES}
    environment = {target: {cell: {metric: responses["return"][target][cell][metric]-responses["current"][target][cell][metric]
                                   for metric in METRICS} for cell in CELLS} for target in RESPONSES}
    return dict(panels=panels, effects=effects, zero=zeros, environment=environment)


def scientific(config):
    return (config["seeds"] == list(range(15001, 15013)) and set(config["models"]) == set(MODELS)
            and all(config.get(name) == value for name, value in SCIENTIFIC.items()))


def screen(total, eligible):
    positive = sum(value > 0 for value in total["differences"])
    gates = dict(complete_scientific_cohort=dict(passed=bool(eligible)),
        deterioration=dict(passed=total["mean"] >= .0005, value=total["mean"], minimum=.0005),
        seed_consistency=dict(passed=positive >= 9, positive_seeds=positive, minimum=9, n=total["n"]))
    return dict(passed=all(row["passed"] for row in gates.values()), gates=gates,
                failed=[name for name, row in gates.items() if not row["passed"]])


def recompute_statistics(records, config):
    expected = {(seed, model, index) for seed in config["seeds"] for model in config["models"] for index in (0, 1)}
    keys = [(record["seed"], record["model"], record["index"]) for record in records]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "seed/assessment coverage mismatch")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["model"], record["seed"]].append(assessment_metrics(record))

    def aggregate(values, seeds):
        first = values[0][0]
        if isinstance(first, dict):
            return {name: aggregate([[row[name] for row in seed] for seed in values], seeds) for name in first}
        return bootstrap([float(np.mean(seed)) for seed in values], seeds)

    result = {name: {} for name in ("panels", "effects", "zero", "environment")}
    for model in config["models"]:
        values = [grouped[model, seed] for seed in config["seeds"]]
        for name in result:
            result[name][model] = aggregate([[row[name] for row in seed] for seed in values], config["seeds"])
    result["screen"] = {model: screen(result["effects"][model]["current"]["selection"]["total_111_minus_000"]["brier"], scientific(config))
                        for model in config["models"]}
    return result


def verify_statistics(summary, records, config):
    expected = recompute_statistics(records, config)
    for section in ("panels", "effects", "zero", "environment"):
        compare(summary[section], expected[section], section)
    compare(summary["coefficients"], coefficients(), "factor coefficients")
    compare(summary["eligible_scientific_cohort"], scientific(config), "scientific eligibility")
    for model in config["models"]:
        compare({key: summary["screen"][model][key] for key in ("passed", "gates", "failed")}, expected["screen"][model], "replication screen")
    require(summary["inference"]["bootstrap_seed"] == 27192026
            and summary["inference"]["bootstrap_resamples"] == 20000
            and summary["inference"]["bit_order"] == list(FACTORS), "inference specification differs")
    return expected["screen"]


@lru_cache(maxsize=4)
def _bootstrap_draws(size):
    result = np.random.default_rng(27192026).integers(0, size, (20000, size))
    result.flags.writeable = False
    return result


def bootstrap(values, seeds):
    values = np.asarray(values, dtype=float)
    require(values.shape == (len(seeds),) and len(seeds) > 0 and len(set(seeds)) == len(seeds)
            and np.isfinite(values).all(), "invalid independent seed values")
    values = np.where(np.abs(values) < 1e-12, 0., values)
    draws = _bootstrap_draws(len(values))
    samples = values[draws].mean(axis=1)
    return dict(mean=float(values.mean()), lower=float(np.quantile(samples, .025)), upper=float(np.quantile(samples, .975)),
        positive=int((values > 0).sum()), negative=int((values < 0).sum()), n=len(values),
        differences=values.tolist(), seeds=list(seeds))


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


def diagnostic_artifacts(index):
    result = {f"diagnostic_{index}/before.pt": "checkpoint", f"diagnostic_{index}/data.npz": "array",
        f"diagnostic_{index}/stream.json": "auxiliary", f"assessment_{index}/candidates.pt": "checkpoint",
        f"assessment_{index}/choice.json": "auxiliary", f"score_{index}/result.json": "phase",
        f"score_{index}/before.pt": "checkpoint", f"score_{index}/shadows.pt": "checkpoint",
        f"diagnostic_{index}/zero.pt": "checkpoint"}
    result.update({f"diagnostic_{index}/cell_{cell}.pt": "checkpoint" for cell in CELLS if cell != "000"})
    for cell in (*CELLS, "zero"):
        result[f"diagnostic_{index}/predictions_{cell}.npz"] = "array"
        result[f"diagnostic_{index}/predictions_{cell}.json"] = "auxiliary"
    return result


def expected_paths(config):
    result = {}
    for seed in config["seeds"]:
        for model in config["models"]:
            prefix = f"{model}_{seed}"
            result[f"{prefix}/result.json"] = "trajectory"
            for phase in schedule(seed, config):
                result[f"{prefix}/{phase['name']}/result.json"] = "phase"
            for index in (0, 1):
                result[f"{prefix}/diagnostic_{index}/result.json"] = "assessment"
                result.update({f"{prefix}/{path}": kind for path, kind in diagnostic_artifacts(index).items()})
    return result


def read_records(folder, config, identity, summary):
    expected, records = expected_paths(config), {}
    require(summary["array_archive_prefix"] == "arrays", "array archive prefix changed")
    listed = summary["result_hashes"]
    require(len(listed) == len(expected) and len({r["path"] for r in listed}) == len(listed)
            and {r["path"]: r["kind"] for r in listed} == expected, "original record/array/checkpoint coverage mismatch")
    hashes = {row["path"]: row["sha256"] for row in listed}
    for path, value in hashes.items():
        relative_path(path)
        valid_hash(value)

    def put(path, value, kind):
        require(path in expected and expected[path] == kind and path not in records
                and value["identity"] == identity, "unexpected, duplicate or misidentified portable record")
        require(sha(original_json(value)) == hashes[path], "original JSON record hash mismatch")
        records[path] = value

    with gzip.open(folder/"raw_results.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            value = decode(line)
            put(f"{value['model']}_{value['seed']}/diagnostic_{value['index']}/result.json", value, "assessment")
    with gzip.open(folder/"phase_results.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            item = decode(line)
            require(set(item) == {"path", "record"}, "invalid phase wrapper")
            put(item["path"], item["record"], "phase")
    for kind in ("trajectory", "auxiliary"):
        for path, value in read(folder/f"{kind}_records.json").items():
            put(path, value, kind)
    require(set(records) == {path for path, kind in expected.items() if kind in ("assessment", "phase", "trajectory", "auxiliary")},
            "missing portable JSON records")
    arrays = {path for path, kind in expected.items() if kind == "array"}
    actual = {path.relative_to(folder/"arrays").as_posix() for path in (folder/"arrays").rglob("*") if path.is_file()}
    require(actual == arrays, "portable numerical array file set changed")
    for path in arrays:
        require(sha((folder/"arrays"/path).read_bytes()) == hashes[path], "numerical array original hash mismatch")
    return records, hashes


def verify_file_integrity(folder, records, hashes, config):
    value = read(folder/"file_integrity.json")
    require(value["passed"] is True, "failed original-file integrity report")
    rows = value["files"]
    listed = {row["path"]: row for row in rows}
    # The preflight inventories structured/binary run artifacts. Protocol Markdown
    # is checked directly against both its protocol lock and the analysis ZIP.
    copied = {"manifest.json", "completion.json", "training_source.zip", "analysis_lock.json",
              "analysis_at_lock.zip", "protocol_lock.json"}
    expected = set(expected_paths(config))|copied
    for seed in config["seeds"]:
        for model in config["models"]:
            for phase in schedule(seed, config):
                for name in ("before.pt", "checkpoint.pt"):
                    expected.add(f"{model}_{seed}/{phase['name']}/{name}")
    require(len(listed) == len(rows) and set(listed) == expected, "original-file integrity coverage mismatch")
    for path, row in listed.items():
        relative_path(path)
        valid_hash(row["sha256"])
        require(row["valid"] is True and type(row["bytes"]) is int and row["bytes"] > 0,
                "invalid original-file integrity entry")
        if path in hashes:
            require(row["sha256"] == hashes[path], "original-file hash differs from portable reference")
        content = (folder/path).read_bytes() if path in copied else original_json(records[path]) if path in records else None
        if content is not None:
            require(row["sha256"] == sha(content) and row["bytes"] == len(content), "original bytes differ from portable copy")
    return listed


def verify_factor_work(record, config, score):
    count, size, updates = config["candidate_count"]+1, config["batch_size"], config["rehearsal_updates"]
    require(set(record["work"]) == set(CELLS), "factor work cell coverage mismatch")
    canonical = set()
    for cell in CELLS:
        work = record["work"][cell]
        require(len(work) == count, "factor work candidate coverage mismatch")
        if cell == "000":
            compare(work, score["shadow_work"], "reused all-old score family")
        for index, item in enumerate(work):
            for name, value in dict(steps=updates, forwards=2*updates, backward_calls=updates,
                    query_presentations=2*updates*size, support_presentations=2*updates*size).items():
                compare(item[name], value, "matched rehearsal budget")
            require(item["parent_unchanged"] is True and item["memory_history_unchanged"] is True, "rehearsal mutation flag")
            anchor = record["old_anchor_sha256"] if cell[2] == "0" else record["new_anchor_sha256"]
            require(item["anchor_packet_sha256"] == anchor, "wrong factor anchor")
            require(item["replay_packet_sha256"] == record["work"]["000"][index]["replay_packet_sha256"], "factor replay changed")
            fields = ("model_bytes", "optimizer_bytes", "gradient_bytes", "initial_encoder_bytes", "replay_bytes", "history_bytes")
            require(all(type(item[name]) is int and item[name] >= 0 for name in fields)
                    and item["explicit_tensor_bytes_subtotal"] == sum(item[name] for name in fields), "factor storage subtotal mismatch")
            if cell != "000":
                require(item["cell"] == cell and [item[name+"_bit"] for name in FACTORS] == list(map(int, cell)), "wrong factor cell bits")
                require(item["old_donor_sha256"] == record["parents"]["old"]
                        and item["new_donor_sha256"] == record["parents"]["new"]
                        and item["inactive_donor_sha256"] == record["parents"]["old"]
                        and item["weights_donor_sha256"] == record["parents"]["new" if cell[0] == "1" else "old"]
                        and item["optimizer_donor_sha256"] == record["parents"]["new" if cell[1] == "1" else "old"], "factor donor mismatch")
                require(item["final_active_sha256"] == record["active_signatures"][cell][index]
                        and item["donors_unchanged"] is True and item["inactive_unchanged"] is True, "factor active state mismatch")
                canonical.add(item["canonical_inactive_sha256"])
                require(item["diagnostic_forwards"] == 4 and item["diagnostic_query_presentations"] == 4*size
                        and item["diagnostic_support_presentations"] == 4*size
                        and item["update_l2"] >= 0 and item["update_norm_dtype"] == "float64", "factor instrumentation mismatch")
                require(set(item["losses"]) == {"anchor_before", "anchor_after", "replay_before", "replay_after"}
                        and all(type(loss) in (int, float) and loss >= 0 for loss in item["losses"].values()), "invalid diagnostic BCE")
    require(len(canonical) == 1, "inactive state differs across factor cells")


def verify_timelines(folder, records, hashes, config, identity, affinity):
    assessments, input_hashes = [], defaultdict(set)
    for model in config["models"]:
        for seed in config["seeds"]:
            prefix = f"{model}_{seed}"
            trajectory = records[f"{prefix}/result.json"]
            planned = schedule(seed, config)
            names = [phase["name"] for phase in planned]
            require((trajectory["seed"], trajectory["model"]) == (seed, model) and trajectory["affinity"] == affinity
                    and trajectory["phase_names"] == names
                    and trajectory["phase_hashes"] == {name: hashes[f"{prefix}/{name}/result.json"] for name in names}
                    and trajectory["assessment_hashes"] == {str(i): hashes[f"{prefix}/diagnostic_{i}/result.json"] for i in (0, 1)}, "trajectory coverage mismatch")
            phases, packets, seen, previous_query, previous_state = {}, {}, 0, None, None
            for phase in planned:
                record = records[f"{prefix}/{phase['name']}/result.json"]
                require(previous_state is None or previous_state == record["before_signature"], "ordinary state chain mismatch")
                previous_query = verify_phase(record, phase, config, seed, model, seen, previous_query)
                packets.update({packet["id"]: packet for packet in record["packets"]})
                for packet_id, metadata in record["metadata"].items():
                    compare(metadata, {name: value for name, value in packets[int(packet_id)].items()
                                       if name not in {"probabilities", "actions", "outcomes"}}, "original metadata provenance")
                previous_state, seen = record["final_signature"], seen+len(record["packets"])
                phases[phase["name"]] = record
            input_hashes[seed].add(digest([packet["query_sha256"] for packet in packets.values()]))
            require(trajectory["final_signature"] == previous_state and trajectory["diagnostics"] == record["diagnostics"], "trajectory final state mismatch")
            compare(trajectory["prequential_work"], dict(forwards=seen, query_presentations=seen*config["batch_size"],
                    support_presentations=seen*config["batch_size"]), "ordinary inference budget")
            for index in (0, 1):
                base = f"{prefix}/diagnostic_{index}"
                record, score = records[f"{base}/result.json"], phases[f"score_{index}"]
                validation = phases[f"validation_{index}"]
                require((record["seed"], record["model"], record["index"]) == (seed, model, index)
                        and record["cue"] == score["phase"]["cue"]
                        and record["parents"] == dict(old=score["before_signature"], new=score["final_signature"])
                        and validation["before_signature"] == record["parents"]["new"], "diagnostic cutoff mismatch")
                require(record["old_steps"] == score["start_diagnostics"]["cost"]["optimizer_steps"]
                        and record["new_steps"] == score["diagnostics"]["cost"]["optimizer_steps"]
                        and record["anchor_id"] == score["start_memory"]["seen"]-1
                        and record["reservoir_ids"] == score["start_memory"]["ids"], "diagnostic donor time mismatch")
                ids, replacement = candidate_sample(record["reservoir_ids"], record["anchor_id"], config["candidate_count"], seed, index)
                require([row["id"] for row in record["candidates"]] == ids and record["replacement"]["id"] == replacement, "candidate draw mismatch")
                for candidate in [*record["candidates"], record["replacement"]]:
                    require(candidate["id"] < record["anchor_id"], "candidate was not previously observed")
                    compare(candidate, {name: value for name, value in packets[candidate["id"]].items()
                            if name not in {"probabilities", "actions", "outcomes"}}, "candidate prequential provenance")
                selected_indices(record)
                close(record["choice"]["score_losses"], score["score_losses"], "choice score loss")
                compare(records[f"{prefix}/assessment_{index}/choice.json"], dict(identity=identity, seed=seed, model=model,
                    index=index, score_record_sha256=digest(score), choice=record["choice"]), "sealed score choice")
                require(set(record["artifact_hashes"]) == set(diagnostic_artifacts(index)), "diagnostic artifact coverage mismatch")
                for path, expected in record["artifact_hashes"].items():
                    require(hashes[f"{prefix}/{path}"] == expected, "diagnostic artifact reference mismatch")
                data = read_arrays(folder/"arrays"/f"{base}/data.npz")
                validate_data(data, config)
                verify_contexts(records[f"{base}/stream.json"], data, validation, score, config, identity, seed, model, index)
                require(set(record["state_files"]) == set(record["prediction_files"]) == set(record["active_signatures"]) == {*CELLS, "zero"},
                        "factor state/prediction coverage mismatch")
                for cell in (*CELLS, "zero"):
                    relative = f"diagnostic_{index}/predictions_{cell}.npz"
                    prediction = read_arrays(folder/"arrays"/f"{prefix}/{relative}")
                    require(set(prediction) == {"probabilities"}, "prediction NPZ fields mismatch")
                    count = 2 if cell == "zero" else config["candidate_count"]+1
                    actual = metric_panels(prediction["probabilities"], data, config, count)
                    require(len(record["active_signatures"][cell]) == count, "active state family size mismatch")
                    for value in record["active_signatures"][cell]:
                        valid_hash(value)
                    for branch in BRANCHES:
                        compare(record["zero"][branch] if cell == "zero" else record["panels"][branch][cell], actual[branch], "all-action metric recomputation")
                    seal = records[f"{base}/predictions_{cell}.json"]
                    compare(seal, dict(identity=identity, parents=record["parents"], active_signatures=record["active_signatures"][cell],
                        data_sha256=hashes[f"{base}/data.npz"], predictions_sha256=hashes[f"{prefix}/{relative}"]), "prediction seal")
                    state_path = f"score_{index}/shadows.pt" if cell == "000" else f"diagnostic_{index}/zero.pt" if cell == "zero" else f"diagnostic_{index}/cell_{cell}.pt"
                    compare(record["state_files"][cell], dict(path=state_path, sha256=hashes[f"{prefix}/{state_path}"]), "state file mapping")
                    compare(record["prediction_files"][cell], dict(path=relative, sha256=hashes[f"{prefix}/{relative}"]), "prediction file mapping")
                verify_factor_work(record, config, score)
                size, count = config["batch_size"], config["candidate_count"]+1
                compare(record["prediction_work"], dict(score_forwards=count*config["score_size"]//size,
                    factorial_forwards=2*config["validation_size"]//size*8*count,
                    zero_forwards=4*config["validation_size"]//size), "diagnostic prediction work")
                assessments.append(record)
    require(all(len(values) == 1 for values in input_hashes.values()), "ordinary inputs differ across architectures")
    return assessments


def verify_audit(audit, config, identity, records, hashes, integrity, manifest_sha, analysis_files):
    require(audit["passed"] is True and audit["identity"] == identity
            and audit["manifest_sha256"] == manifest_sha
            and audit["audit_source_sha256"] == analysis_files["scripts/audit_rehearsal_state.py"], "unrelated or unsealed tensor audit")
    declared = expected_counts(config)
    jobs, phases = declared["trajectories"], config["prefix_blocks"]+6
    ordinary = declared["ordinary_arrivals_per_trajectory"]//config["batch_size"]
    totals = dict(trajectories=jobs, before_checkpoints=phases*jobs, final_checkpoints=phases*jobs,
        diagnostic_boundary_checkpoints=2*jobs, assessments=2*jobs, ordinary_packets=ordinary*jobs,
        original_packet_errors_recomputed=ordinary*jobs, original_predictions_recomputed=phases*jobs,
        diagnostic_forks_reproduced=declared["diagnostic_forks"], zero_update_copies=declared["zero_update_copies"],
        diagnostic_prediction_calls_recomputed=declared["diagnostic_prediction_forwards"],
        loss_instrumentation_forwards_recomputed=declared["extra_loss_forwards"])
    for name, value in totals.items():
        compare(audit[name], value, f"tensor audit {name}")
    require({row["path"]: row["sha256"] for row in audit["result_hashes"]} == hashes
            and len(audit["result_hashes"]) == len(hashes)
            and {row["path"]: row["kind"] for row in audit["result_hashes"]} == expected_paths(config), "tensor audit artifact reference mismatch")
    checks = audit["checks"]
    require(len(checks) == jobs and {(row["seed"], row["model"]) for row in checks}
            == {(seed, model) for seed in config["seeds"] for model in config["models"]}, "tensor audit job coverage mismatch")
    count, packets = config["candidate_count"]+1, config["validation_size"]//config["batch_size"]

    def check_hash(actual, path):
        require(actual == integrity[path]["sha256"], f"tensor audit file reference mismatch: {path}")

    for check in checks:
        prefix = f"{check['model']}_{check['seed']}"
        for name, value in totals.items():
            if name != "trajectories":
                compare(check[name], value//jobs, f"tensor job {name}")
        planned, seen, queries = schedule(check["seed"], config), 0, []
        require([row["phase"] for row in check["phases"]] == [phase["name"] for phase in planned], "tensor audit phase coverage mismatch")
        for row, phase in zip(check["phases"], planned):
            amount = phase["size"]//config["batch_size"]
            seen += amount
            scoring = phase["kind"] == "score"
            require(row["packets"] == amount and row["cumulative_packets"] == seen
                    and row["rehearsal_forks_reproduced"] == (count if scoring else 0)
                    and row["diagnostic_predictions_recomputed"] == (amount if scoring else 0), "tensor audit phase work mismatch")
            require(all(row[name] is True for name in ("causal_data_replay_metadata_verified", "exact_start_chain_verified",
                    "adam_counters_verified", "start_prediction_recomputed")), "tensor audit phase assertion missing")
            for key, filename in (("result_sha256", "result.json"), ("before_sha256", "before.pt"), ("checkpoint_sha256", "checkpoint.pt")):
                check_hash(row[key], f"{prefix}/{phase['name']}/{filename}")
            if scoring:
                check_hash(row["shadows_sha256"], f"{prefix}/{phase['name']}/shadows.pt")
            else:
                require(row["shadows_sha256"] is None, "unexpected ordinary shadow audit")
            queries.extend(packet["query_sha256"] for packet in records[f"{prefix}/{phase['name']}/result.json"]["packets"])
        require(check["input_stream_sha256"] == digest(queries), "tensor audit input stream mismatch")
        rows = check["assessment_checks"]
        require(len(rows) == 2 and {row["index"] for row in rows} == {0, 1}, "tensor assessment coverage mismatch")
        for row in rows:
            index = row["index"]
            base = f"{prefix}/diagnostic_{index}"
            record = records[f"{base}/result.json"]
            for key, relative in (("result_sha256", f"diagnostic_{index}/result.json"),
                    ("before_sha256", f"diagnostic_{index}/before.pt"), ("data_sha256", f"diagnostic_{index}/data.npz"),
                    ("stream_sha256", f"diagnostic_{index}/stream.json"), ("choice_sha256", f"assessment_{index}/choice.json"),
                    ("candidates_sha256", f"assessment_{index}/candidates.pt"), ("score_result_sha256", f"score_{index}/result.json")):
                check_hash(row[key], f"{prefix}/{relative}")
            for field in ("parents", "state_files", "prediction_files"):
                compare(row[field], record[field], f"tensor assessment {field}")
            for name, value in dict(newly_reproduced_forks=7*count, reused_score_forks=count, zero_update_copies=2,
                    diagnostic_prediction_calls_recomputed=2*packets*(8*count+2),
                    loss_instrumentation_forwards_recomputed=4*7*count, all_new_active_endpoints_verified=count).items():
                compare(row[name], value, f"tensor assessment {name}")
            require(all(row[name] is True for name in ("paired_causal_branches_verified", "unchanged_donors_verified",
                    "score_only_selection_verified")), "tensor assessment assertion missing")
    return totals


def verify_packaging(folder):
    names = ("verifier_lock.json", "verifier_source.zip", "verify_archive.py")
    present = [bool((folder/name).is_file()) for name in names]
    require(all(present) or not any(present), "partial standalone verifier seal")
    if not any(present):
        return False
    lock = read(folder/"verifier_lock.json")
    source, tests = "scripts/verify_rehearsal_state_archive.py", "tests/test_rehearsal_state_archive.py"
    require(set(lock["files"]) == {source, tests}, "standalone verifier source coverage mismatch")
    verify_zip(folder/"verifier_source.zip", lock["files"])
    require(sha((folder/"verify_archive.py").read_bytes()) == lock["files"][source], "copied verifier differs from sealed source")
    require(sha(Path(__file__).read_bytes()) == lock["files"][source], "executing verifier differs from sealed source")
    return True


def verify(directory):
    folder = Path(directory)
    artifact_count = verify_artifacts(folder)
    verifier_sealed = verify_packaging(folder)
    manifest, config, summary = read(folder/"manifest.json"), read(folder/"config.json"), read(folder/"summary.json")
    identity = manifest["identity"]
    compare(manifest["config"], config, "manifest config")
    compare(summary["config"], config, "summary config")
    compare(summary["identity"], identity, "summary identity")
    for name, value in (("config_sha256", config), ("source_sha256", manifest["source_files"]), ("runtime_sha256", manifest["runtime"])):
        require(identity[name] == digest(value), "manifest identity mismatch")
    verify_zip(folder/"training_source.zip", manifest["source_files"])
    protocol, analysis = read(folder/"protocol_lock.json"), read(folder/"analysis_lock.json")
    require(all(protocol[name] == value for name, value in identity.items())
            and protocol["protocol_sha256"] == sha((folder/"protocol_at_lock.md").read_bytes()), "protocol lock mismatch")
    require(analysis["config"] == config and analysis["config_sha256"] == identity["config_sha256"], "analysis config mismatch")
    required = {"scripts/summarize_rehearsal_state.py", "tests/test_rehearsal_state_summary.py", "scripts/audit_rehearsal_state.py",
                "scripts/summarize_predictive_value.py"}
    require(required <= analysis["files"].keys() and any(name.startswith("docs/") and value == protocol["protocol_sha256"]
            for name, value in analysis["files"].items()), "prospective analysis/audit/protocol missing from lock")
    verify_zip(folder/"analysis_at_lock.zip", analysis["files"])
    require(manifest["created_utc"] == analysis["locked_utc"] == protocol["locked_utc"], "lock creation times differ")
    completion = read(folder/"completion.json")
    compare(completion["identity"], identity, "completion identity")
    counts = expected_counts(config)
    compare(summary["counts"], counts, "summary counts")
    for name, value in counts.items():
        compare(completion[name], value, f"completion {name}")
    jobs = completion["jobs"]
    require(len(jobs) == counts["trajectories"] and {(job["seed"], job["model"]) for job in jobs}
            == {(seed, model) for seed in config["seeds"] for model in config["models"]}
            and all(job["affinity"] == manifest["runtime"]["affinity"] for job in jobs), "completion job/affinity mismatch")
    records, hashes = read_records(folder, config, identity, summary)
    integrity = verify_file_integrity(folder, records, hashes, config)
    assessments = verify_timelines(folder, records, hashes, config, identity, manifest["runtime"]["affinity"])
    screens = verify_statistics(summary, assessments, config)
    audit = verify_audit(read(folder/"audit.json"), config, identity, records, hashes, integrity,
                         sha((folder/"manifest.json").read_bytes()), analysis["files"])
    compare(read(folder/"record_integrity.json"), summary["record_integrity"], "record integrity copy")
    require(summary["record_integrity"]["passed"] is True
            and summary["record_integrity"]["assessments"] == counts["assessments"], "record integrity coverage mismatch")
    with (folder/"rows.jsonl").open(encoding="utf-8") as stream:
        compare([decode(line) for line in stream], summary["rows"], "descriptive row copy")
    return dict(passed=True, identity=identity, artifact_files=artifact_count, standalone_verifier_sealed=verifier_sealed,
        original_json_records=len(records), original_run_files=len(integrity), numerical_arrays=sum(name.endswith(".npz") for name in hashes),
        **counts, tensor_audit_coverage=audit, replication_screen={model: value["passed"] for model, value in screens.items()},
        scope="Independent portable file/lock/identity/coverage checks, original JSON and NPZ hashes, causal branch contexts, "
              "all-action Brier and greedy survival, early/late metrics, exact candidate selectors and matching zero baselines, "
              "all factorial/repair/environment effects, seed bootstrap and primary replication screen. "
              "Checkpoint hashes and complete reconstruction coverage are bound to the archived tensor audit. "
              "No live source imports, pickle loading, ordinary retraining or independent physics replication. "
              "Descriptive rank/cue/coverage summaries and plots are checksummed but not independently recalculated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.input), indent=2, allow_nan=False))
