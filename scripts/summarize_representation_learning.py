"""Independent NumPy scoring for the locked temporal-representation study."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from itertools import permutations
import json
from pathlib import Path
import zipfile

import numpy as np


ARMS = ("outcome", "final_frame", "sequence", "compute")
METRICS = ("brier", "focus_brier", "survival", "focus_survival", "valid_brier",
           "valid_survival", "no_transfer", "clairvoyant_upper")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON: " + value)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=nonfinite)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def fingerprint(*values):
    digest = hashlib.sha256()
    for value in values:
        require(not value.dtype.hasobject, "object arrays are not valid scientific records")
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def law(mode, active=(), revised=False):
    return dict(mode=mode, active=list(active), revised=revised, noise=0.)


def laws(seed):
    order = tuple(permutations(range(3)))[seed % 6]
    return [law(seed % 2), law(1 - seed % 2),
            *[law(1 - seed % 2, sorted(order[:stage])) for stage in (1, 2, 3)]]


def law_key(value):
    return f"m{value['mode']}_a{''.join(map(str, value['active']))}_r{int(value['revised'])}"


def expected_phases(seed, config):
    reference, phases = laws(seed), []
    order = tuple(permutations(range(3)))[seed % 6]

    def phase(name, target, size, cue=None, branch="novel", targets=None):
        return dict(name=name, cue=cue, branch=branch, blocks=[dict(law=target, size=size)],
                    targets=[target] if targets is None else targets)

    if config["kind"] == "qualification":
        phases = [phase(f"fresh_{stage}", reference[stage + 1], config["episode_size"],
                        order[stage - 1]) for stage in (1, 2, 3)]
        phases.append(dict(name="interleaved", cue=None, branch="novel", targets=reference,
            blocks=[dict(law=target, size=config["interleave_block"])
                    for _ in range(config["interleave_cycles"]) for target in reference]))
    else:
        phases = [phase(f"prefix_{i}", law((seed + i) % 2), config["prefix_size"], targets=[])
                  for i in range(4)]
        phases += [phase(f"novel_{stage}", reference[stage + 1], config["episode_size"],
                         order[stage - 1]) for stage in (1, 2, 3)]
        for name, target, cue, branch in (
                ("return_old", reference[0], None, "return"),
                ("return_new", reference[-1], None, "novel"),
                ("revision", law(reference[-1]["mode"], (0, 1, 2), True), 0, "revision"),
                ("return_unrevised", reference[-1], 0, "novel")):
            phases.append(phase(name, target, config["episode_size"], cue, branch))
    return phases


def score(prediction, truth, affected, valid, marginal=None):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    n, eps = len(truth), np.finfo(np.float32).eps
    require(n > 0 and prediction.shape == truth.shape == (n, 5, 3)
            and prediction.dtype.kind == "f" and np.isfinite(prediction).all()
            and np.all((prediction >= -eps) & (prediction <= 1 + eps))
            and np.all(np.diff(prediction, axis=-1) <= eps), "invalid survival forecast")
    require(np.isin(truth, (0, 1)).all()
            and np.all(np.diff(truth.astype(np.int16), axis=-1) <= 0), "invalid survival truth")
    for mask in (affected, valid):
        require(mask.dtype == np.bool_ and mask.shape == (n,) and mask.any(), "invalid scoring mask")
    error = np.square(prediction.astype(np.float64) - truth.astype(np.float64)).mean(axis=(1, 2))
    survived = truth[np.arange(n), prediction[:, :, -1].argmax(axis=1), -1]
    result = dict(brier=float(error.mean()), focus_brier=float(error[affected].mean()),
        survival=float(survived.mean()), focus_survival=float(survived[affected].mean()),
        valid_brier=float(error[valid].mean()), valid_survival=float(survived[valid].mean()),
        no_transfer=float(truth[:, 0, -1].mean()),
        clairvoyant_upper=float(truth[:, :, -1].max(axis=1).mean()))
    if marginal is not None:
        marginal = np.asarray(marginal, dtype=np.float64)
        require(marginal.shape == (5, 3) and np.isfinite(marginal).all()
                and np.all((marginal >= 0) & (marginal <= 1)), "invalid marginal forecast")
        result["marginal_brier"] = float(np.square(marginal[None] - truth).mean())
    return result


def marginal(count, success):
    count, success = np.asarray(count), np.asarray(success)
    require(count.shape == (5,) and count.dtype.kind in "iu" and np.all(count >= 0)
            and success.shape == (5, 3) and success.dtype.kind in "iuf"
            and np.isfinite(success).all() and np.all(success == np.floor(success))
            and np.all((success >= 0) & (success <= count[:, None])), "invalid training totals")
    return (success.astype(np.float64) + .5) / (count[:, None] + 1.)


def cue_benefit(correct, flipped):
    require(np.array_equal(correct["truth"], flipped["truth"])
            and np.array_equal(correct["valid"], flipped["valid"]), "cue probe changed physical target")
    fields = (flipped["truth"], flipped["affected"], flipped["valid"])
    baseline = score(correct["probabilities"], *fields)["focus_brier"]
    return score(flipped["probabilities"], *fields)["focus_brier"] - baseline


@lru_cache(maxsize=None)
def draws(n):
    return np.random.default_rng(28192026).integers(0, n, (20000, n))


def estimate(values, seeds):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (len(seeds),) and len(seeds) > 0 and len(set(seeds)) == len(seeds)
            and np.isfinite(values).all(), "invalid seed-balanced summary")
    lower, upper = np.quantile(values[draws(len(seeds))].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), lower=float(lower), upper=float(upper), seeds=list(seeds),
                values=values.tolist(), n=len(seeds), negative=int((values < 0).sum()))


def qualification(rows, config):
    seeds = config["seeds"]
    require({(row["seed"], row["phase"]) for row in rows}
            == {(seed, phase) for seed in seeds for phase in ("fresh_1", "fresh_2", "fresh_3", "interleaved")}
            and len(rows) == 4 * len(seeds) and all(row["arm"] == "outcome" for row in rows),
            "qualification cohort is incomplete or duplicated")
    groups = []

    def group(name, selected, field, minimum=None, maximum=None):
        require(selected, "empty required qualification group: " + name)
        values = [field(row) for row in selected]
        stat = estimate(values, [row["seed"] for row in selected])
        passed = ((minimum is None or stat["mean"] >= minimum)
                  and (maximum is None or stat["mean"] <= maximum))
        groups.append(dict(name=name, metric=stat, minimum=minimum, maximum=maximum, passed=passed))

    fresh = [row for row in rows if row["phase"].startswith("fresh_")]
    fresh_groups = [(f"fresh/stage_{stage}", [row for row in fresh if row["phase"] == f"fresh_{stage}"])
                    for stage in (1, 2, 3)]
    fresh_groups += [(f"fresh/cue_{cue}", [row for row in fresh if row["introduced_cue"] == cue])
                     for cue in sorted({row["introduced_cue"] for row in fresh})]
    for name, selected in fresh_groups:
        group(name + "/marginal_gain", selected,
              lambda row: row["end_targets"][0]["marginal_gain"], minimum=.02)
        group(name + "/cue_benefit", selected,
              lambda row: row["end_targets"][0]["cue_benefits"][str(row["introduced_cue"])], minimum=.002)
    selected = [row for row in rows if row["phase"] == "interleaved"]
    for slot in range(5):
        group(f"interleaved/slot_{slot}/marginal_gain", selected,
              lambda row: row["end_targets"][slot]["marginal_gain"], minimum=.02)
        group(f"interleaved/slot_{slot}/absolute_brier", selected,
              lambda row: row["end_targets"][slot]["metrics"]["brier"], maximum=.12)
        if slot >= 2:
            for cue in (0, 1, 2):
                subset = [row for row in selected if str(cue) in row["end_targets"][slot]["cue_benefits"]]
                if subset:
                    group(f"interleaved/stage_{slot - 1}/cue_{cue}/cue_benefit", subset,
                          lambda row: row["end_targets"][slot]["cue_benefits"][str(cue)], minimum=.002)
    raw_pass = all(item["passed"] for item in groups)
    eligible = not config["smoke"] and seeds == list(range(18101, 18107))
    return dict(eligible=eligible, raw_passed=raw_pass,
                decision="INELIGIBLE" if not eligible else "PASS" if raw_pass else "STOP",
                groups=groups, per_phase=rows)


def read_evaluations(folder, config, seed):
    payload = read(folder / "evaluations.json")
    require(payload["seed"] == seed, "evaluation seed differs")
    with np.load(folder / "evaluations.npz", allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate evaluation arrays")
        arrays = {name: archive[name] for name in archive.files}
    records, values, used = payload["records"], [], set()
    counters = dict(query_presentations=0, support_presentations=0, observed_support_presentations=0)
    for index, record in enumerate(records):
        require(record["index"] == index and record["branch"] in ("novel", "return", "revision", "noise")
                and record["cue"] in (None, 0, 1, 2) and type(record["flipped"]) is bool
                and (not record["flipped"] or record["cue"] is not None), "invalid evaluation trace identity")
        present = record["support_fingerprint"] is not None
        names = {"probabilities", "truth", "observations", "affected", "valid"}
        if present:
            names |= {"support_observations", "support_actions", "support_outcomes"}
        require(set(record["array_sha256"]) == names, "evaluation field coverage differs")
        item = {}
        for name in names:
            key = f"e{index}_{name}"
            require(key in arrays and fingerprint(arrays[key]) == record["array_sha256"][name],
                    "missing or changed evaluation array: " + key)
            item[name] = arrays[key]
            used.add(key)
        n = len(item["truth"])
        require(n == config["eval_size"] and item["observations"].shape == (n, 4, 12, 16)
                and item["observations"].dtype == np.uint8
                and record["query_sha256"] == fingerprint(item["observations"]), "invalid query images")
        support_count = 0
        if present:
            support = tuple(item["support_" + name] for name in ("observations", "actions", "outcomes"))
            support_count = len(support[1])
            require(support_count > 0 and support[0].shape == (support_count, 4, 12, 16)
                    and support[0].dtype == np.uint8 and support[1].shape == (support_count,)
                    and support[1].dtype.kind in "iu" and np.all((support[1] >= 0) & (support[1] < 5))
                    and support[2].shape == (support_count, 3) and np.isin(support[2], (0, 1)).all()
                    and fingerprint(*support) == record["support_fingerprint"], "invalid evaluation support")
        expected = dict(query_presentations=n, observed_support_presentations=support_count,
                        support_presentations=support_count if present else config["batch_size"])
        for name, value in expected.items():
            require(record[name] == value, "evaluation presentation count differs")
            counters[name] += value
        metrics = score(item["probabilities"], item["truth"], item["affected"], item["valid"], record["marginal"])
        require(set(metrics) == set(record["metrics"])
                and all(abs(metrics[key] - record["metrics"][key]) < 1e-12 for key in metrics),
                "recorded evaluation metric differs from raw arrays")
        values.append(item)
    require(used == set(arrays) and payload["calls"] == len(records)
            and all(payload[name] == value for name, value in counters.items()), "incomplete evaluation ledger")
    return records, values


def read_training(folder, record, config):
    with np.load(folder / "training.npz", allow_pickle=False) as archive:
        required = {"probabilities", "truth", "actions", "outcomes", "affected", "valid", "law_index", "observations"}
        require(set(archive.files) == required, "training array coverage differs")
        values = {name: archive[name] for name in required}
    batch = config["batch_size"]
    n_packets = sum(block["size"] for block in record["phase_spec"]["blocks"]) // batch
    require(record["size"] == n_packets * batch and record["batches_done"] == n_packets
            and values["actions"].shape == (n_packets, batch)
            and values["outcomes"].shape == (n_packets, batch, 3)
            and values["observations"].shape == (n_packets, batch, 4, 12, 16)
            and values["observations"].dtype == np.uint8, "incomplete training stream")
    keys, index = [], []
    for block in record["phase_spec"]["blocks"]:
        key = law_key(block["law"])
        if key not in keys:
            keys.append(key)
        index += [keys.index(key)] * (block["size"] // batch)
    require(values["law_index"].dtype.kind in "iu" and np.array_equal(values["law_index"], index),
            "training law sequence differs from protocol")
    flat = {name: value.reshape((-1, *value.shape[2:])) for name, value in values.items() if name != "law_index"}
    actions, truth = flat["actions"], flat["truth"]
    require(actions.dtype.kind in "iu" and np.all((actions >= 0) & (actions < 5))
            and np.array_equal(flat["outcomes"], truth[np.arange(len(actions)), actions]),
            "performed outcomes differ from physical truth")
    require(set(record["marginals"]) == set(keys), "training marginal law coverage differs")
    for law_index, key in enumerate(keys):
        mask = np.repeat(values["law_index"] == law_index, batch)
        counts = np.bincount(actions[mask], minlength=5)
        successes = np.array([flat["outcomes"][mask & (actions == action)].sum(axis=0) for action in range(5)])
        totals = record["marginals"][key]
        require(np.array_equal(counts, totals["count"]) and np.array_equal(successes, totals["success"]),
                "training-only marginal totals do not match recorded observations")
        marginal(totals["count"], totals["success"])
    metrics = score(flat["probabilities"], truth, flat["affected"], flat["valid"])
    return metrics, flat


def endpoint_targets(record, traces, arrays, config):
    require([item["law"] for item in record["end_probes"]] == record["phase_spec"]["targets"],
            "endpoint target law coverage differs")
    result, used, model_hashes = [], set(), set()
    for item in record["end_probes"]:
        target, correct_ids = item["law"], item["correct"]
        require(len(correct_ids) == config["support_replicates"]
                and set(item["flips"]) == set(map(str, target["active"])), "endpoint cue/support coverage differs")
        predictions, gains = [], {}
        for replicate, trace_id in enumerate(correct_ids):
            require(type(trace_id) is int and 0 <= trace_id < len(traces) and trace_id not in used,
                    "invalid or reused correct endpoint trace")
            used.add(trace_id)
            trace, correct = traces[trace_id], arrays[trace_id]
            require(trace["law"] == target and trace["cue"] is None and not trace["flipped"]
                    and trace["observed_support_presentations"] == config["batch_size"],
                    "correct endpoint trace is not the declared full-support probe")
            totals = record["marginals"][law_key(target)]
            marginal_value = marginal(totals["count"], totals["success"])
            require(np.array_equal(np.asarray(trace["marginal"]), marginal_value), "endpoint marginal is not training-only")
            predictions.append(score(correct["probabilities"], correct["truth"], correct["affected"],
                                     correct["valid"], marginal_value))
            model_hashes.add(trace["model_sha256"])
            for cue, flipped_ids in item["flips"].items():
                require(len(flipped_ids) == config["support_replicates"], "cue support replicates differ")
                index = flipped_ids[replicate]
                require(type(index) is int and 0 <= index < len(traces) and index not in used,
                        "invalid or reused flipped endpoint trace")
                used.add(index)
                flip = traces[index]
                require(flip["law"] == target and flip["cue"] == int(cue) and flip["flipped"]
                        and flip["support_fingerprint"] == trace["support_fingerprint"]
                        and flip["model_sha256"] == trace["model_sha256"], "unpaired endpoint cue probe")
                gains.setdefault(cue, []).append(cue_benefit(correct, arrays[index]))
        averaged = {key: float(np.mean([value[key] for value in predictions])) for key in predictions[0]}
        result.append(dict(law=target, metrics=averaged,
                           marginal_gain=averaged["marginal_brier"] - averaged["brier"],
                           cue_benefits={cue: float(np.mean(value)) for cue, value in gains.items()}))
    require(len(model_hashes) <= 1, "endpoint traces use different fitted model weights")
    require(not model_hashes or model_hashes == {record["final_model_sha256"]},
            "endpoint probe does not use the terminal fitted model")
    return result


def trace_schedule(record, traces, arrays, training, config, initial_support):
    phase, batch = record["phase_spec"], config["batch_size"]
    packet_laws = [block["law"] for block in phase["blocks"] for _ in range(block["size"] // batch)]
    require(len(record["packets"]) == len(packet_laws), "training packet ledger is incomplete")
    history, fingerprints = initial_support, []
    for index, packet in enumerate(record["packets"]):
        start, end = index * batch, (index + 1) * batch
        current = fingerprint(*(training[name][start:end] for name in ("observations", "actions", "outcomes")))
        require(packet["index"] == index and packet["query_sha256"] == current
                and packet["support_sha256"] == history, "training packet uses the wrong original support")
        history = current
        fingerprints.append(current)
    require(record["start_eval"]["trace"] == 0 and traces[0]["law"] == packet_laws[0]
            and traces[0]["model_sha256"] == record["initial_model_sha256"]
            and traces[0]["support_fingerprint"] == initial_support, "invalid entry evaluation")
    index = 1
    points = [end for end in range(1, len(packet_laws) + 1)
              if end * batch % config["probe_every"] == 0 or end == len(packet_laws)]
    require(len(record["curve"]) == len(points), "online probe coverage differs")
    for point, end in zip(record["curve"], points):
        trace = traces[index]
        require(point["arrivals"] == end * batch and point["law"] == trace["law"] == packet_laws[end - 1]
                and point["metrics"] == dict(trace["metrics"], trace=index)
                and trace["support_fingerprint"] == fingerprints[end - 1]
                and trace["cue"] == phase["cue"] and trace["branch"] == phase["branch"]
                and not trace["flipped"], "online probe chronology differs")
        index += 1
    require(traces[index - 1]["model_sha256"] == record["final_model_sha256"], "last probe is not terminal")
    for item in record["end_probes"]:
        for replicate, correct in enumerate(item["correct"]):
            require(correct == index, "endpoint correct trace is out of order")
            index += 1
            for cue in item["law"]["active"]:
                require(item["flips"][str(cue)][replicate] == index, "endpoint cue trace is out of order")
                index += 1
    valid = None
    if config["kind"] == "main" and not phase["name"].startswith("prefix_"):
        valid = {}
        for mode in (0, 1):
            scores = []
            target = dict(packet_laws[-1], mode=mode, noise=0.)
            for _ in range(config["support_replicates"]):
                trace = traces[index]
                require(trace["law"] == target and trace["cue"] is None and not trace["flipped"]
                        and trace["branch"] == phase["branch"]
                        and trace["model_sha256"] == record["final_model_sha256"]
                        and trace["observed_support_presentations"] == batch, "invalid valid-old panel")
                scores.append(score(*(arrays[index][key] for key in ("probabilities", "truth", "affected", "valid"))))
                index += 1
            valid[str(mode)] = {key: float(np.mean([value[key] for value in scores]))
                                for key in ("valid_brier", "valid_survival")}
        require(valid == record["valid_after"], "valid-old panel summary differs from raw traces")
    else:
        require(record["valid_after"] is None, "unexpected valid-old panel")
    require(index == len(traces), "unaccounted evaluation calls")
    return valid, history


def read_run(directory):
    directory = Path(directory).resolve()
    manifest, rows, inventory, paired = read(directory / "manifest.json"), [], [], {}
    config, identity = manifest["config"], manifest["identity"]
    for name, field in (("config", "config"), ("source", "source_files"),
                        ("analysis", "analysis_files"), ("runtime", "runtime")):
        require(json_hash(manifest[field]) == identity[name + "_sha256"], "manifest identity differs")
    require(sha(directory / "protocol_at_lock.md") == identity["protocol_sha256"], "locked protocol changed")
    prerequisite = manifest["prerequisite"]
    require(config["kind"] != "main" or config["smoke"] or prerequisite is not None,
            "scientific main lacks the declared qualification/engineering prerequisite")
    if prerequisite is not None:
        allocation_path = directory / "engineering_at_lock.json"
        require(sha(allocation_path) == prerequisite["engineering_sha256"]
                and read(allocation_path)["compute_updates_per_batch"] == config["compute_updates_per_batch"],
                "archived engineering allocation differs from its locked prerequisite")
    require(manifest["analysis_files"]["scripts/summarize_representation_learning.py"] == sha(__file__),
            "scorer differs from the pre-run analysis lock")
    locked = {**manifest["source_files"], **manifest["analysis_files"]}
    with zipfile.ZipFile(directory / "source_at_lock.zip") as archive:
        require(len(archive.namelist()) == len(locked) and set(archive.namelist()) == set(locked), "source archive coverage differs")
        require(all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in locked.items()),
                "locked source archive changed")
    completion = read(directory / "completion.json")
    jobs = {f"jobs/{arm}_{seed}/result.json" for arm in config["arms"] for seed in config["seeds"]}
    require(completion["identity"] == identity and len(completion["jobs"]) == len(jobs)
            and {item["path"] for item in completion["jobs"]} == jobs, "completed job coverage differs")

    def checked(base, relative, expected):
        path = (base / relative).resolve()
        require(path.is_relative_to(base) and sha(path) == expected, "missing/changed artifact: " + relative)
        inventory.append(dict(path=path.relative_to(directory).as_posix(), sha256=expected))
        return path

    for job_file in completion["jobs"]:
        path = checked(directory, job_file["path"], job_file["sha256"])
        job = read(path)
        seed, arm = job["seed"], job["arm"]
        require(job["identity"] == identity and job_file["path"] == f"jobs/{arm}_{seed}/result.json", "job identity differs")
        phases = expected_phases(seed, config)
        require(set(job["phase_hashes"]) == {p["name"] + "/result.json" for p in phases}
                and set(job["checkpoint_hashes"]) == {p["name"] + "/checkpoint.pt" for p in phases},
                "phase/checkpoint coverage differs")
        history, parent = None, None
        for phase in phases:
            name = phase["name"]
            folder = path.parent / name
            record = read(checked(path.parent, name + "/result.json", job["phase_hashes"][name + "/result.json"]))
            checked(path.parent, name + "/checkpoint.pt", job["checkpoint_hashes"][name + "/checkpoint.pt"])
            require(record["identity"] == identity and record["seed"] == seed and record["arm"] == arm
                    and record["phase"] == name and record["phase_spec"] == phase, "phase differs from declared schedule")
            if config["kind"] == "qualification":
                history, parent = None, None
            require(parent is None or record["start_signature"] == parent, "continuous learner was reset between phases")
            start, end = record["start_diagnostics"], record["diagnostics"]
            updates = config["compute_updates_per_batch"] if arm == "compute" else config["updates_per_batch"]
            require(end["cost"]["arrivals"] - start["cost"]["arrivals"] == record["size"]
                    and end["cost"]["optimizer_steps"] - start["cost"]["optimizer_steps"]
                    == record["size"] // config["batch_size"] * updates, "fitting exposure/update budget differs")
            if parent is None:
                require(start["cost"]["arrivals"] == start["cost"]["optimizer_steps"] == 0
                        and start["optimizer_bytes"] == 0 and start["memory_ids"] == []
                        and record["initial_model_sha256"] == start["initial_hash"], "fresh learner is not fresh")
            init_key = (seed, "initial_predictor_hash")
            require(paired.setdefault(init_key, start["initial_predictor_hash"]) == start["initial_predictor_hash"],
                    "predictive initializations differ between phases or comparators")
            require(set(record["artifact_hashes"]) == {"before.pt", "evaluations.json", "evaluations.npz", "training.npz"},
                    "phase raw artifact coverage differs")
            for relative, expected in record["artifact_hashes"].items():
                checked(folder, relative, expected)
            require(record["training_file"] == dict(path="training.npz", sha256=record["artifact_hashes"]["training.npz"]),
                    "training descriptor differs")
            traces, arrays = read_evaluations(folder, config, seed)
            descriptor = dict(json=dict(path="evaluations.json", sha256=record["artifact_hashes"]["evaluations.json"]),
                arrays=dict(path="evaluations.npz", sha256=record["artifact_hashes"]["evaluations.npz"]), calls=len(traces),
                **{field: sum(trace[field] for trace in traces) for field in
                   ("query_presentations", "support_presentations", "observed_support_presentations")})
            require(record["evaluation_files"] == descriptor, "evaluation descriptor differs from trace ledger")
            stream, training = read_training(folder, record, config)
            targets = endpoint_targets(record, traces, arrays, config)
            valid, history = trace_schedule(record, traces, arrays, training, config, history)
            parent = record["end_signature"]
            for field in ("observations", "truth", "actions", "outcomes", "affected", "valid"):
                key, value = (seed, name, field), fingerprint(training[field])
                require(paired.setdefault(key, value) == value, "comparators received different training experience")
            for index, item in enumerate(arrays):
                for field in item.keys() - {"probabilities"}:
                    key, value = (seed, name, index, field), fingerprint(item[field])
                    require(paired.setdefault(key, value) == value, "comparators received different evaluation inputs")
            rows.append(dict(seed=seed, arm=arm, phase=name, introduced_cue=phase["cue"],
                size=record["size"], stream=stream, end_targets=targets, valid_after=valid,
                diagnostics=record["diagnostics"], start_diagnostics=record["start_diagnostics"],
                elapsed_seconds=record["elapsed_seconds"]))
    counts = dict(jobs=len(jobs), phases=len(rows), training_arrivals=sum(row["size"] for row in rows),
        optimizer_steps=sum(row["diagnostics"]["cost"]["optimizer_steps"]
                            - row["start_diagnostics"]["cost"]["optimizer_steps"] for row in rows))
    require(counts == manifest["expected_counts"], "total fitting budget differs from manifest")
    return rows, config, dict(counts=counts, manifest_sha256=sha(directory / "manifest.json"),
                              completion_sha256=sha(directory / "completion.json"), files=inventory)


def main_comparison(rows, config):
    seeds, checks = config["seeds"], []
    keyed = {(row["seed"], row["arm"], row["phase"]): row for row in rows}
    names = [phase["name"] for phase in expected_phases(seeds[0], config)]
    require(len(rows) == len(keyed) and set(keyed) == {
        (seed, arm, phase) for seed in seeds for arm in ARMS for phase in names}, "main cohort is incomplete")
    novel = [f"novel_{stage}" for stage in (1, 2, 3)]
    returns = ["return_old", "return_new", "revision", "return_unrevised"]

    def phase(seed, arm, name):
        return keyed[seed, arm, name]

    def overall(seed, arm, metric):
        chosen = [phase(seed, arm, name) for name in names]
        return float(np.average([row["stream"][metric] for row in chosen],
                                weights=[row["size"] for row in chosen]))

    def acquisition(seed, arm):
        return float(np.mean([phase(seed, arm, name)["stream"]["focus_brier"] for name in novel]))

    def check(name, values, selected=seeds, minimum=None, maximum=None, consistent=False):
        stat = estimate(values, selected)
        passed = ((minimum is None or stat["mean"] >= minimum)
                  and (maximum is None or stat["mean"] <= maximum)
                  and (not consistent or stat["negative"] >= 5))
        checks.append(dict(name=name, metric=stat, minimum=minimum, maximum=maximum,
                           requires_five_negative=consistent, passed=passed))

    for baseline in ("outcome", "final_frame", "compute"):
        check("acquisition_minus_" + baseline, [acquisition(seed, "sequence") - acquisition(seed, baseline)
              for seed in seeds], maximum=-.001 if baseline == "final_frame" else -.002, consistent=True)
        for cue in range(3):
            matching = [(seed, next(name for name in novel
                         if phase(seed, "sequence", name)["introduced_cue"] == cue)) for seed in seeds]
            check(f"introduced_cue_{cue}_minus_{baseline}", [phase(seed, "sequence", name)["stream"]["focus_brier"]
                - phase(seed, baseline, name)["stream"]["focus_brier"] for seed, name in matching], maximum=.005)
        for metric, minimum, maximum in (("brier", None, .005), ("survival", -.01, None)):
            check(f"whole_{metric}_minus_{baseline}", [overall(seed, "sequence", metric) - overall(seed, baseline, metric)
                  for seed in seeds], minimum=minimum, maximum=maximum)
            for name in returns:
                check(f"{name}_{metric}_minus_{baseline}", [phase(seed, "sequence", name)["stream"][metric]
                    - phase(seed, baseline, name)["stream"][metric] for seed in seeds], minimum=minimum, maximum=maximum)
        for name in novel + returns:
            for mode in ("0", "1"):
                check(f"{name}/valid_mode_{mode}_minus_{baseline}", [phase(seed, "sequence", name)["valid_after"][mode]["valid_brier"]
                    - phase(seed, baseline, name)["valid_after"][mode]["valid_brier"] for seed in seeds], maximum=.005)
    for name in novel + returns:
        for mode in ("0", "1"):
            check(f"{name}/valid_mode_{mode}_absolute", [phase(seed, "sequence", name)["valid_after"][mode]["valid_brier"]
                  for seed in seeds], maximum=.12)
        if name == "return_old":
            check("return_old/endpoint_absolute", [phase(seed, "sequence", name)["end_targets"][0]["metrics"]["brier"]
                  for seed in seeds], maximum=.12)
        else:
            for cue in range(3):
                selected = [seed for seed in seeds if str(cue) in phase(seed, "sequence", name)["end_targets"][0]["cue_benefits"]]
                if selected:
                    check(f"{name}/active_cue_{cue}", [phase(seed, "sequence", name)["end_targets"][0]["cue_benefits"][str(cue)]
                          for seed in selected], selected=selected, minimum=.002)
    levels = {arm: dict(acquisition=estimate([acquisition(seed, arm) for seed in seeds], seeds),
                        whole={metric: estimate([overall(seed, arm, metric) for seed in seeds], seeds)
                               for metric in ("brier", "survival")}) for arm in ARMS}
    eligible = not config["smoke"] and seeds == list(range(18201, 18207))
    raw_pass = all(item["passed"] for item in checks)
    return dict(eligible=eligible, raw_passed=raw_pass, decision="INELIGIBLE" if not eligible else "PASS" if raw_pass else "STOP",
                groups=checks, levels=levels, per_phase=rows)


def markdown(result):
    lines = ["# Temporal-representation learning: " + result["kind"], "",
             f"Decision: **{result['decision']}**. Raw gate conjunction: {result['raw_passed']}. "
             f"Eligible scientific cohort: {result['eligible']}.", "",
             "Brier values are raw units. Intervals are descriptive paired-seed bootstrap intervals. "
             "Support replicates are averaged within seed; every required cell must pass.", "",
             "| Required cell | Mean [95% interval] | Minimum | Maximum | Pass |",
             "|---|---:|---:|---:|---|"]
    for item in result["groups"]:
        value = item["metric"]
        lines.append(f"| {item['name']} | {value['mean']:.6f} [{value['lower']:.6f}, {value['upper']:.6f}] "
                     f"| {item['minimum']} | {item['maximum']} | {item['passed']} |")
    lines += ["", "A qualification failure stops the candidate comparison. A main failure stops this "
              "fixed reconstruction recipe. No subgroup promotion, exposure increase, or auxiliary-weight rescue.", ""]
    return "\n".join(lines)


def summarize(directory, output):
    rows, config, integrity = read_run(directory)
    result = (qualification if config["kind"] == "qualification" else main_comparison)(rows, config)
    result.update(kind=config["kind"], qualified=result["eligible"] and result["raw_passed"],
                  integrity=integrity, bootstrap=dict(seed=28192026, resamples=20000,
                  unit="independent seed", interval="descriptive percentile 95%"))
    output = Path(output)
    write(output / "summary.json", result)
    (output / "summary.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run, args.output)
    print(json.dumps(dict(decision=summary["decision"], qualified=summary["qualified"]), indent=2))
