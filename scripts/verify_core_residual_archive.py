"""Verify a sealed core/residual archive with Python stdlib and NumPy only.

No repository imports, model unpickling, retraining or simulator execution.
The archived tensor audit supplies the separately stated checkpoint evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from functools import lru_cache
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np


ARMS = ("joint", "separate", "fixed_features")
PHASES = ("maintenance", "novel", "return")
MODELS = ("conditional", "recurrent")
METRICS = ("brier", "focus_brier", "survival", "focus_survival", "valid_brier",
           "valid_survival", "no_transfer", "clairvoyant_upper")
COMPARISONS = {"separate_minus_joint":("separate", "joint"),
    "separate_minus_fixed_features":("separate", "fixed_features"),
    "fixed_features_minus_joint":("fixed_features", "joint")}
SCIENTIFIC = dict(kind="development", seeds=list(range(16001, 16007)), models=list(MODELS),
    prefix_blocks=4, prefix_size=1024, episode_size=8192, batch_size=32, eval_size=512,
    probe_every=256, support_replicates=2, width=64, experts=4, context_width=12,
    decoder_width=64, interaction_features=True, lr=.002, memory_packets=16,
    updates_per_batch=12, evidence_strength=1., feedback_noise=0., maintenance_size=1024,
    return_size=2048, residual_width=32, residual_context_width=8, residual_decoder_width=32)
EXECUTION_FIELDS = {"study", "threads", "workers", "cpu_affinity"}
EPSILON = np.finfo(np.float32).eps


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite JSON value")


def decode(contents):
    def object_pairs(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, "duplicate JSON key")
            result[name] = value
        return result
    result = json.loads(contents, object_pairs_hook=object_pairs)
    finite_tree(result)
    return result


def read(path):
    return decode(Path(path).read_text(encoding="utf-8"))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def original_json(value):
    return (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()


def valid_hash(value):
    require(isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value), "invalid SHA256")


def relative_path(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value, "invalid archive path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and value == path.as_posix()
            and value != ".", "archive path escapes or aliases its root")
    return value


def compare(actual, expected, label="value"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "different keys: "+label)
        for key in expected:
            compare(actual[key], expected[key], label+"."+str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), "different sequence: "+label)
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, label+f"[{index}]")
    elif isinstance(expected, (float, np.floating)):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, float(expected), rel_tol=1e-11, abs_tol=1e-12), "numeric mismatch: "+label)
    else:
        require(type(actual) is type(expected) and actual == expected, "value mismatch: "+label)


def verify_artifacts(folder):
    folder = Path(folder).resolve()
    manifest = read(folder/"artifact_manifest.json")
    present = {path.relative_to(folder).as_posix() for path in folder.rglob("*")
               if path.is_file() and path != folder/"artifact_manifest.json"}
    require(isinstance(manifest, dict) and set(manifest) == present, "portable artifact coverage changed")
    for name, item in manifest.items():
        path = (folder/relative_path(name)).resolve()
        require(path.is_relative_to(folder), "artifact resolves outside archive")
        data = path.read_bytes()
        compare(item, dict(bytes=len(data), sha256=sha(data)), "artifact "+name)
    return len(manifest)


def verify_zip(path, expected):
    for name, value in expected.items():
        relative_path(name)
        valid_hash(value)
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected)
                and archive.testzip() is None, "locked ZIP coverage/CRC mismatch")
        for name, value in expected.items():
            require(sha(archive.read(name)) == value, "locked ZIP hash mismatch")


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate NPZ key")
        return {name:archive[name] for name in archive.files}


def array_hash(value):
    require(isinstance(value, np.ndarray) and not value.dtype.hasobject, "invalid raw array")
    return sha(str((value.shape, value.dtype.str)).encode()+value.tobytes(order="C"))


def experience_hash(observations, actions, outcomes):
    value = hashlib.sha256()
    for array in (observations, actions, outcomes):
        value.update(str((array.shape, array.dtype.str)).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def probabilities(value, shape):
    require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
            and np.all(value >= -EPSILON) and np.all(value <= 1+EPSILON)
            and np.all(np.diff(value, axis=-1) <= EPSILON), "invalid raw survival probabilities")


def raw_metrics(prediction, truth, affected, valid, marginal=None):
    require(prediction.ndim == 3 and len(prediction) > 0, "nonempty evaluation predictions required")
    size = len(prediction)
    probabilities(prediction, (size, 5, 3))
    require(truth.dtype == np.uint8 and truth.shape == prediction.shape
            and np.isin(truth, (0, 1)).all() and np.all(np.diff(truth.astype(np.int16), axis=-1) <= 0),
            "invalid physical truth")
    for mask in (affected, valid):
        require(mask.dtype == np.bool_ and mask.shape == (size,) and mask.any(), "invalid affected/valid mask")
    errors = np.square(prediction.astype(np.float64)-truth.astype(np.float64)).mean(axis=(1, 2))
    chosen = prediction[:, :, -1].argmax(axis=1)
    survival = truth[np.arange(size), chosen, -1].astype(np.float64)
    result = dict(brier=float(errors.mean()), focus_brier=float(errors[affected].mean()),
        survival=float(survival.mean()), focus_survival=float(survival[affected].mean()),
        valid_brier=float(errors[valid].mean()), valid_survival=float(survival[valid].mean()),
        no_transfer=float(truth[:, 0, -1].mean()), clairvoyant_upper=float(truth[:, :, -1].max(axis=1).mean()))
    if marginal is not None:
        marginal = np.asarray(marginal, dtype=float)
        require(marginal.shape == (5, 3) and np.isfinite(marginal).all()
                and np.all(marginal >= 0) and np.all(marginal <= 1), "invalid action/horizon marginal")
        result["marginal_brier"] = float(np.square(marginal[None]-truth).mean())
    return result


def auc(curve, field):
    x = np.asarray([point["arrivals"] for point in curve], dtype=float)
    y = np.asarray([point["metrics"][field] for point in curve], dtype=float)
    require(len(x) > 1 and x[0] == 0 and np.all(np.diff(x) > 0) and np.isfinite(y).all(), "invalid learning curve")
    return float(np.dot(np.diff(x), .5*(y[1:]+y[:-1]))/x[-1])


def probe_mean(probe, field, flipped=False):
    rows = probe["replicates"]
    require(bool(rows), "empty diagnostic support replicates")
    return float(np.mean([row["flipped"][field] if flipped else row["curve"][-1]["metrics"][field] for row in rows]))


def phase_metrics(record):
    def get(when, field):
        return probe_mean(record[when], field)
    result = dict(brier_auc=auc(record["curve"], "focus_brier"), all_brier_auc=auc(record["curve"], "brier"),
        survival_auc=auc(record["curve"], "survival"), before=get("before", "focus_brier"), after=get("after", "focus_brier"),
        endpoint=record["curve"][-1]["metrics"]["focus_brier"], before_survival=get("before", "survival"),
        after_survival=get("after", "survival"), marginal_gain=get("after", "marginal_brier")-get("after", "brier"),
        prequential_brier=float(np.mean([packet["prequential_brier"] for packet in record["packets"]])))
    for mode in ("0", "1"):
        for when in ("before", "after"):
            result[f"valid_{when}_mode_{mode}"] = record[f"valid_{when}"][mode]["valid_brier"]
            result[f"valid_survival_{when}_mode_{mode}"] = record[f"valid_{when}"][mode]["valid_survival"]
        result["valid_damage_mode_"+mode] = result["valid_after_mode_"+mode]-result["valid_before_mode_"+mode]
    for when in ("before", "after"):
        result["valid_"+when] = .5*sum(result[f"valid_{when}_mode_{mode}"] for mode in ("0", "1"))
    result["valid_damage"] = result["valid_after"]-result["valid_before"]
    if record["cue"] is not None:
        result["cue_effect"] = probe_mean(record["after"], "focus_brier", True)-result["after"]
        result["cue_effect_before"] = probe_mean(record["before"], "focus_brier", True)-result["before"]
        result["cue_learning_gain"] = result["cue_effect"]-result["cue_effect_before"]
    if record["novel_after_return"] is not None:
        value = record["novel_after_return"]
        result.update(novel_after_return_brier=probe_mean(value, "focus_brier"),
            novel_after_return_survival=probe_mean(value, "survival"),
            novel_after_return_cue_effect=probe_mean(value, "focus_brier", True)-probe_mean(value, "focus_brier"))
    return result


@lru_cache(maxsize=None)
def bootstrap_draws(size):
    return np.random.default_rng(27192026).integers(0, size, (20000, size))


def bootstrap(values, seeds):
    values = np.asarray(values, dtype=np.float64).copy()
    require(values.shape == (len(seeds),) and len(seeds) and len(set(seeds)) == len(seeds)
            and np.isfinite(values).all(), "invalid paired seed sample")
    values[np.abs(values) < 1e-12] = 0
    draws = values[bootstrap_draws(len(values))].mean(axis=1)
    return dict(mean=float(values.mean()), lower=float(np.percentile(draws, 2.5)), upper=float(np.percentile(draws, 97.5)),
        positive=int((values > 0).sum()), negative=int((values < 0).sum()), n=len(values), differences=values.tolist(), seeds=list(seeds))


def eligible(config):
    return {key:value for key,value in config.items() if key not in EXECUTION_FIELDS} == SCIENTIFIC


def upper(metric, maximum):
    return dict(passed=metric["mean"] <= maximum, value=metric["mean"], maximum=maximum,
        lower=metric["lower"], upper=metric["upper"], uncertainty_crosses_limit=metric["upper"] > maximum)


def lower(metric, minimum):
    return dict(passed=metric["mean"] >= minimum, value=metric["mean"], minimum=minimum,
        lower=metric["lower"], upper=metric["upper"], uncertainty_crosses_limit=metric["lower"] < minimum)


def gates_result(gates):
    failed = [key for key, value in gates.items() if not value["passed"]]
    return dict(passed=not failed, gates=gates, failed=failed)


def primary_screen(comparison, qualified, complete):
    novel, returned = comparison["novel"], comparison["return"]
    improved = sum(value < 0 for value in novel["brier_auc"]["differences"])
    gates = dict(complete_development_cohort=dict(passed=bool(complete)), fresh_qualification=dict(passed=bool(qualified)),
        acquisition_gain=upper(novel["brier_auc"], -.002),
        acquisition_consistency=dict(passed=improved >= 5, improved_seeds=improved, minimum=5, n=novel["brier_auc"]["n"]))
    for mode in ("0", "1"):
        gates["valid_old_mode_"+mode] = upper(novel["valid_after_mode_"+mode], .005)
    gates["return_prediction"] = upper(returned["all_brier_auc"], .005)
    gates["post_return_novel_retention"] = upper(returned["novel_after_return_brier"], .005)
    for phase in ("novel", "return"):
        gates["survival_"+phase] = lower(comparison[phase]["survival_auc"], -.01)
    return gates_result(gates)


def attribution_screen(comparison, complete):
    metric = comparison["novel"]["brier_auc"]
    improved = sum(value < 0 for value in metric["differences"])
    return gates_result(dict(complete_development_cohort=dict(passed=bool(complete)),
        feature_adaptation_gain=upper(metric, -.001),
        feature_adaptation_consistency=dict(passed=improved >= 5, improved_seeds=improved, minimum=5, n=metric["n"])))


def fresh_qualification(rows, models):
    groups, by_model = [], {}
    for model in models:
        fresh = [row for row in rows if row["model"] == model and row["arm"] == "fresh"]
        require(fresh and all(row["phase"] == "novel" for row in fresh), "fresh qualification coverage mismatch")
        partitions = [("overall", fresh)]+[("cue_"+str(cue), [row for row in fresh if row["cue"] == cue])
                                            for cue in sorted({row["cue"] for row in fresh})]
        for label, selected in partitions:
            gain, use = (float(np.mean([row[field] for row in selected])) for field in ("marginal_gain", "cue_effect"))
            groups.append(dict(model=model, group=label, n=len(selected), seeds=[row["seed"] for row in selected],
                marginal_gain=gain, cue_effect=use, passed=gain >= .02 and use >= .002))
        by_model[model] = all(row["passed"] for row in groups if row["model"] == model)
    return dict(groups=groups, by_model=by_model)


def recompute_statistics(records, config):
    rows = []
    for record in records:
        if record["arm"] != "prefix":
            rows.append({**{key:record[key] for key in ("seed", "model", "arm", "phase", "cue")}, **phase_metrics(record)})
    rows.sort(key=lambda row:(row["seed"], row["model"], row["arm"], row["phase"]))
    keyed = {(row["seed"], row["model"], row["arm"], row["phase"]):row for row in rows}
    expected = {(seed, model, arm, phase) for seed in config["seeds"] for model in config["models"]
                for arm in (*ARMS, "fresh") for phase in (("novel",) if arm == "fresh" else PHASES)}
    require(len(rows) == len(keyed) and set(keyed) == expected, "incomplete/duplicate learning records")
    seeds = config["seeds"]
    levels, comparisons, fresh, slices = {}, {}, {}, {}
    for model in config["models"]:
        levels[model], comparisons[model], fresh[model], slices[model] = {}, {}, {}, {}
        for phase in PHASES:
            levels[model][phase] = {}
            for arm in ((*ARMS, "fresh") if phase == "novel" else ARMS):
                fields = set(keyed[seeds[0], model, arm, phase])-{ "seed", "model", "arm", "phase", "cue"}
                levels[model][phase][arm] = {field:bootstrap([keyed[seed, model, arm, phase][field] for seed in seeds], seeds)
                                            for field in sorted(fields)}
        for label, (first, second) in COMPARISONS.items():
            comparisons[model][label] = {phase:{field:bootstrap([
                keyed[seed, model, first, phase][field]-keyed[seed, model, second, phase][field] for seed in seeds], seeds)
                for field in levels[model][phase][first]} for phase in PHASES}
        for arm in ARMS:
            fresh[model][arm+"_minus_fresh"] = {field:bootstrap([
                keyed[seed, model, arm, "novel"][field]-keyed[seed, model, "fresh", "novel"][field] for seed in seeds], seeds)
                for field in levels[model]["novel"][arm]}
        for cue in sorted({keyed[seed, model, "joint", "novel"]["cue"] for seed in seeds}):
            subset = [seed for seed in seeds if keyed[seed, model, "joint", "novel"]["cue"] == cue]
            slices[model][str(cue)] = dict(seeds=subset, n=len(subset), separate_minus_joint={phase:{field:
                float(np.mean([keyed[seed, model, "separate", phase][field]-keyed[seed, model, "joint", phase][field] for seed in subset]))
                for field in levels[model][phase]["joint"]} for phase in PHASES})
    qualified, complete = fresh_qualification(rows, config["models"]), eligible(config)
    return dict(rows=rows, levels=levels, comparisons=comparisons, fresh_comparisons=fresh,
        descriptive_cue_slices=slices, qualification=qualified, eligible_development_cohort=complete,
        screen={model:primary_screen(comparisons[model]["separate_minus_joint"], qualified["by_model"][model], complete)
                for model in config["models"]},
        attribution={model:attribution_screen(comparisons[model]["separate_minus_fixed_features"], complete)
                     for model in config["models"]})


def plan(seed, config):
    cue = tuple(permutations(range(3)))[seed % 6][0]
    def entry(arm, name, mode, active, size, branch, channel):
        return dict(arm=arm, phase=name, relative=name if arm == "prefix" else arm+"/"+name,
            law=dict(mode=mode, active=active, revised=False, noise=0.), size=size,
            cue=cue if active else None, branch=branch, channel=channel)
    result = [entry("prefix", f"prefix_{index}", (seed+index) % 2, [], config["prefix_size"], "novel", f"prefix_{index}")
              for index in range(config["prefix_blocks"])]
    for arm in (*ARMS, "fresh"):
        if arm != "fresh":
            result.append(entry(arm, "maintenance", 1-seed % 2, [], config["maintenance_size"], "novel", "core_residual_maintenance"))
        result.append(entry(arm, "novel", 1-seed % 2, [cue], config["episode_size"], "novel", "core_residual_novel"))
        if arm != "fresh":
            result.append(entry(arm, "return", seed % 2, [], config["return_size"], "return", "core_residual_return"))
    return result


def expected_counts(config):
    require(set(config) == set(SCIENTIFIC)|EXECUTION_FIELDS, "unknown/missing scientific configuration fields")
    for field in ("prefix_blocks", "prefix_size", "episode_size", "batch_size", "eval_size", "probe_every",
            "support_replicates", "width", "experts", "context_width", "decoder_width", "memory_packets",
            "updates_per_batch", "threads", "workers", "maintenance_size", "return_size", "residual_width",
            "residual_context_width", "residual_decoder_width"):
        require(type(config[field]) is int and config[field] > 0, "positive integer setting required")
    require(config["prefix_blocks"] >= 2 and config["prefix_blocks"] % 2 == 0 and config["experts"] == 4
            and config["batch_size"] % 8 == 0 and config["eval_size"] % 64 == 0
            and config["probe_every"] % config["batch_size"] == 0, "invalid paired experimental dimensions")
    require(config["kind"] == "development" and config["interaction_features"] is True
            and config["feedback_noise"] == 0 and config["residual_width"] <= config["width"], "unsupported scientific recipe")
    require(type(config["cpu_affinity"]) is int and config["cpu_affinity"] >= 0
            and math.isfinite(config["lr"]) and config["lr"] > 0
            and math.isfinite(config["evidence_strength"]) and config["evidence_strength"] > 0, "invalid numeric setting")
    require(config["seeds"] and config["models"] and len(set(config["seeds"])) == len(config["seeds"])
            and len(set(config["models"])) == len(config["models"])
            and all(type(seed) is int and seed >= 0 for seed in config["seeds"])
            and set(config["models"]) <= set(MODELS), "invalid configured job coverage")
    for field in ("prefix_size", "episode_size", "maintenance_size", "return_size"):
        require(config[field] % config["probe_every"] == 0, "phase exposure is not probe aligned")
    jobs = len(config["seeds"])*len(config["models"])
    arrivals = jobs*(config["prefix_blocks"]*config["prefix_size"]
        +3*(config["maintenance_size"]+config["episode_size"]+config["return_size"])+config["episode_size"])
    return dict(jobs=jobs, prefix_fits=jobs, prefix_phases=jobs*config["prefix_blocks"],
        learning_trajectories=3*jobs, fresh_references=jobs, post_prefix_episodes=10*jobs,
        total_phases=jobs*(config["prefix_blocks"]+10), training_arrivals=arrivals,
        training_packets=arrivals//config["batch_size"], optimizer_steps=arrivals//config["batch_size"]*config["updates_per_batch"])


def expected_paths(config):
    result = {}
    for seed in config["seeds"]:
        for model in config["models"]:
            prefix = f"{model}_{seed}"
            result[prefix+"/result.json"] = "job"
            for phase in plan(seed, config):
                base = prefix+"/"+phase["relative"]+"/"
                result[base+"result.json"] = "phase"
                result[base+"evaluations.json"] = "evaluation"
                result[base+"evaluations.npz"] = "evaluation_arrays"
                result[base+"training.npz"] = "training_arrays"
                result[base+"before.pt"] = "before"
                result[base+"checkpoint.pt"] = "checkpoint"
    return result


def phase_path(record):
    base = f"{record['model']}_{record['seed']}/"
    return base+(record["phase"] if record["arm"] == "prefix" else record["arm"]+"/"+record["phase"])+"/result.json"


class Memory:
    """Independent uniform-reservoir membership and replay RNG reconstruction."""
    def __init__(self, seed, capacity):
        self.capacity, self.seen = capacity, 0
        self.ids, self.packets = [], []
        self.membership = np.random.default_rng(seed+47201)
        self.sampling = np.random.default_rng(seed+62071)

    def snapshot(self):
        return dict(type="ReservoirMemory", capacity=self.capacity, seen=self.seen, ids=list(self.ids),
            membership=self.membership.bit_generator.state, sampling=self.sampling.bit_generator.state,
            age=self.seen, packets=list(self.packets))

    def step(self, query, support, updates):
        replay = [self.ids[int(self.sampling.integers(0, len(self.ids)))] if self.ids else self.seen for _ in range(updates)]
        identifier = self.seen
        self.seen += 1
        packet = dict(support=support, query=query, hidden=False)
        if len(self.ids) < self.capacity:
            self.ids.append(identifier)
            self.packets.append(packet)
        else:
            slot = int(self.membership.integers(0, self.seen))
            if slot < self.capacity:
                self.ids[slot], self.packets[slot] = identifier, packet
        return replay


def trace_arrays(trace, arrays, config):
    require(set(trace) == {"seed", "records", "calls", "query_presentations", "support_presentations", "observed_support_presentations"},
            "evaluation trace schema differs")
    names, totals, material = set(), dict(calls=0, query_presentations=0, support_presentations=0, observed_support_presentations=0), []
    for index, row in enumerate(trace["records"]):
        expected = {"index", "law", "cue", "branch", "flipped", "support_fingerprint", "query_sha256", "model_sha256",
            "metrics", "marginal", "query_presentations", "support_presentations", "observed_support_presentations", "array_sha256"}
        require(set(row) == expected and row["index"] == index and type(row["index"]) is int, "trace row schema/order differs")
        fields = {"probabilities", "truth", "observations", "affected", "valid"}
        if row["support_fingerprint"] is not None:
            fields |= {"support_observations", "support_actions", "support_outcomes"}
        require(set(row["array_sha256"]) == fields, "trace array field coverage differs")
        values = {}
        for field in fields:
            key = f"e{index}_{field}"
            require(key in arrays and array_hash(arrays[key]) == row["array_sha256"][field], "trace array digest mismatch")
            names.add(key)
            values[field] = arrays[key]
        require(values["observations"].dtype == np.uint8
                and values["observations"].shape == (config["eval_size"], 4, 12, 16)
                and row["query_sha256"] == array_hash(values["observations"]), "trace raw query differs")
        valid_hash(row["model_sha256"])
        observed = 0
        if row["support_fingerprint"] is not None:
            sx, sa, sy = (values[field] for field in ("support_observations", "support_actions", "support_outcomes"))
            observed = len(sa)
            require(observed > 0 and observed <= config["batch_size"] and sx.dtype == sa.dtype == sy.dtype == np.uint8
                    and sx.shape == (observed, 4, 12, 16) and sa.shape == (observed,) and np.all(sa < 5)
                    and sy.shape == (observed, 3) and np.isin(sy, (0, 1)).all(), "invalid causal support arrays")
            require(experience_hash(sx, sa, sy) == row["support_fingerprint"], "support fingerprint differs from stored arrays")
        expected_metrics = raw_metrics(values["probabilities"], values["truth"], values["affected"], values["valid"], row["marginal"])
        compare(row["metrics"], expected_metrics, "raw evaluation metrics")
        require(row["query_presentations"] == config["eval_size"] and row["observed_support_presentations"] == observed
                and row["support_presentations"] == (observed or config["batch_size"]), "trace presentation counts differ")
        totals["calls"] += 1
        for name in ("query_presentations", "support_presentations", "observed_support_presentations"):
            totals[name] += row[name]
        material.append(values)
    require(names == set(arrays), "unreferenced or missing raw evaluation arrays")
    for name, value in totals.items():
        compare(trace[name], value, "evaluation work."+name)
    return material, totals


def verify_trace_program(record, trace, material, config, history, last_history, marginals):
    """Check every probe/curve call in execution order and its past-only context."""
    rows, position = trace["records"], 0
    supports = (0, config["batch_size"]//4, config["batch_size"]//2, config["batch_size"])
    start_hash, final_hash = (record[name]["final_hash"] for name in ("start_diagnostics", "diagnostics"))

    def consume(metrics, law, cue, branch, flipped, support_hash=None, check_support=False, model_hash=None, marginal=None):
        nonlocal position
        require(position < len(rows), "missing evaluation call")
        row = rows[position]
        compare(metrics, {**row["metrics"], "trace":position}, "evaluation trace reference")
        require(row["law"] == law and row["cue"] == cue and row["branch"] == branch and row["flipped"] is flipped,
                "evaluation law/cue/branch/order mismatch")
        compare(row["marginal"], marginal, "evaluation marginal source")
        if check_support:
            require(row["support_fingerprint"] == support_hash, "evaluation is not using declared preceding context")
        if model_hash is not None:
            require(row["model_sha256"] == model_hash, "boundary evaluation uses wrong saved model")
        current = position
        position += 1
        return current

    def probe(value, law, cue, branch, initial_history, model_hash, marginal=None):
        require(value["law"] == law and value["model_sha256"] == model_hash
                and len(value["replicates"]) == config["support_replicates"], "probe metadata/replicates differ")
        for replicate in value["replicates"]:
            require(set(replicate) == ({"curve", "flipped"} if cue is not None else {"curve"})
                    and [point["feedback"] for point in replicate["curve"]] == list(supports), "probe feedback budget differs")
            indices = []
            for point in replicate["curve"]:
                indices.append(consume(point["metrics"], law, cue, branch, False, initial_history,
                    point["feedback"] == 0, model_hash, marginal))
            full = material[indices[-1]]
            base = material[indices[0]]
            for feedback, index in zip(supports[1:], indices[1:]):
                current = material[index]
                for suffix in ("observations", "actions", "outcomes"):
                    field = "support_"+suffix
                    fresh = full[field][:feedback]
                    expected = fresh if initial_history is None else np.concatenate((base[field], fresh))[-config["batch_size"]:]
                    require(np.array_equal(current[field], expected), "probe support is not the declared causal replacement")
            if cue is not None:
                consume(replicate["flipped"], law, cue, branch, True, rows[indices[-1]]["support_fingerprint"], True, model_hash)

    def valid_panel(panel, law, branch, model_hash):
        require(set(panel) == {"0", "1"}, "valid-old mode coverage differs")
        for mode in (0, 1):
            selected = []
            target = {**law, "mode":mode, "noise":0.}
            for _ in range(config["support_replicates"]):
                require(position < len(rows), "missing valid-panel evaluation call")
                row = rows[position]
                index = consume({**row["metrics"], "trace":position}, target, None, branch, False, model_hash=model_hash)
                selected.append(rows[index]["metrics"])
            expected = {name:float(np.mean([row[name] for row in selected])) for name in ("valid_brier", "valid_survival")}
            compare(panel[str(mode)], expected, "valid-old aggregation")

    law, cue, branch = record["law"], record["cue"], record["branch"]
    probe(record["before"], law, cue, branch, history, start_hash)
    valid_panel(record["valid_before"], law, branch, start_hash)
    expected_arrivals = list(range(0, record["size"]+1, config["probe_every"]))
    require([point["arrivals"] for point in record["curve"]] == expected_arrivals, "curve exposure coverage differs")
    for point in record["curve"]:
        arrival = point["arrivals"]
        packet = arrival//config["batch_size"]-1
        support = history if not arrival else record["packets"][packet]["query_sha256"]
        model_hash = start_hash if not arrival else final_hash if arrival == record["size"] else None
        consume(point["metrics"], law, cue, branch, False, support, True, model_hash, None if not arrival else marginals[packet])
    probe(record["after"], law, cue, branch, last_history, final_hash, marginals[-1])
    valid_panel(record["valid_after"], law, branch, final_hash)
    if record["phase"] == "return":
        target_cue = tuple(permutations(range(3)))[record["seed"] % 6][0]
        novel = dict(mode=1-record["seed"] % 2, active=[target_cue], revised=False, noise=0.)
        probe(record["novel_after_return"], novel, target_cue, "novel", last_history, final_hash)
    else:
        require(record["novel_after_return"] is None, "unexpected post-return probe")
    require(position == len(rows), "unreferenced evaluation calls")
    return position


def check_work(diagnostic, seen, post_packets, internal, config):
    size, steps = config["batch_size"], config["updates_per_batch"]
    expected = dict(arrivals=seen*size, optimizer_steps=seen*steps,
        query_presentations=2*seen*size*steps, support_presentations=2*seen*size*steps,
        replay_presentations=max(0, seen-1)*size*steps, duplicate_presentations=min(1, seen)*size*steps)
    require(set(diagnostic["cost"]) == set(expected)|{"training_seconds"}, "ordinary cost schema differs")
    for name, value in expected.items():
        compare(diagnostic["cost"][name], value, "ordinary cumulative work."+name)
    require(diagnostic["cost"]["training_seconds"] >= 0 and diagnostic["packets_seen"] == seen, "invalid cumulative work")
    for name in ("initial_hash", "final_hash", "initial_encoder_hash", "encoder_hash"):
        valid_hash(diagnostic[name])
    if internal is None:
        require("core_residual" not in diagnostic, "residual inserted into the ordinary prefix")
        return
    item = diagnostic["core_residual"]
    require(item["arm"] == internal, "incorrect internal trainability arm")
    amount = post_packets*steps
    work = dict(packets=post_packets, optimizer_steps=amount, backward_calls=amount,
        logical_probability_calls=2*amount, core_forward_calls=2*amount, residual_forward_calls=2*amount,
        core_query_presentations=2*amount*size, residual_query_presentations=2*amount*size,
        core_support_presentations=2*amount*size, residual_support_presentations=2*amount*size,
        core_backward_paths=2*amount*(internal == "joint"), residual_backward_paths=2*amount)
    compare(item["work"], work, "composite cumulative work")
    p = item["parameters"]
    require(all(type(value) is int and value >= 0 for value in p.values()) and p["total"] == diagnostic["parameters"]
            and p["total"] == p["core"]+p["residual"]
            and p["core_trainable"] == (p["core"] if internal == "joint" else 0)
            and p["core_encoder_trainable"] == (p["core_encoder"] if internal == "joint" else 0)
            and p["residual_encoder_trainable"] == (0 if internal == "fixed_features" else p["residual_encoder"])
            and p["residual_trainable"] == p["residual"]-(p["residual_encoder"] if internal == "fixed_features" else 0)
            and p["total_trainable"] == p["core_trainable"]+p["residual_trainable"], "declared parameter/trainability counts differ")
    require(diagnostic["optimizer_bytes"] == item["core_optimizer_bytes"]+item["residual_optimizer_bytes"], "component Adam storage differs")
    for name, value in item.items():
        if name.endswith("_hash"):
            valid_hash(value)
    if internal != "joint":
        require(item["core_hash"] == item["core_at_fork_hash"]
                and item["core_optimizer_hash"] == item["core_optimizer_at_fork_hash"]
                and item["core_displacement_from_fork"] == item["core_encoder_displacement_from_fork"] == 0.,
                "frozen core or retained Adam state changed")
    if internal == "fixed_features":
        require(item["residual_encoder_hash"] == item["residual_initial_encoder_hash"]
                and item["residual_encoder_displacement_from_initial"] == 0., "fixed random encoder changed")


def verify_phase(record, expected, training, trace, evaluation, memory, history, post_packets, config, identity):
    require(record["identity"] == identity, "phase identity differs")
    for name in ("arm", "phase", "law", "size", "cue", "branch", "channel"):
        compare(record[name], expected[name], "declared phase."+name)
    for name in ("start_signature", "end_signature"):
        valid_hash(record[name])
    size, count = config["batch_size"], record["size"]//config["batch_size"]
    require(record["batches_done"] == count == len(record["packets"]) == len(record["batch_sha256"]), "incomplete phase packets")
    require(set(training) == {"probabilities", "actions", "outcomes"}, "training numerical fields differ")
    probabilities(training["probabilities"], (count, size, 5, 3))
    actions, outcomes = training["actions"], training["outcomes"]
    require(actions.dtype == outcomes.dtype == np.uint8 and actions.shape == (count, size)
            and outcomes.shape == (count, size, 3) and np.all(actions < 5) and np.isin(outcomes, (0, 1)).all(), "invalid training feedback")
    compare(record["start_memory"], memory.snapshot(), "initial uniform memory")
    internal = None if record["arm"] == "prefix" else "joint" if record["arm"] == "fresh" else record["arm"]
    check_work(record["start_diagnostics"], memory.seen, post_packets, internal, config)
    initial_history = history
    marginal_count, marginal_success = np.zeros(5, dtype=np.int64), np.zeros((5, 3), dtype=np.float64)
    marginals = []
    for index, packet in enumerate(record["packets"]):
        require(set(packet) == {"index", "id", "query_sha256", "support_sha256", "memory_ids_before", "replay_ids", "prequential_brier"},
                "unexpected ordinary packet fields")
        valid_hash(packet["query_sha256"])
        require(packet["index"] == index and packet["id"] == memory.seen and packet["support_sha256"] == history
                and packet["query_sha256"] == record["batch_sha256"][index] and packet["memory_ids_before"] == memory.ids,
                "ordinary packet identity/context/replay differs")
        compare(packet["replay_ids"], memory.step(packet["query_sha256"], history, config["updates_per_batch"]), "exact uniform replay draws")
        history = packet["query_sha256"]
        performed = training["probabilities"][index, np.arange(size), actions[index]].astype(np.float64)
        compare(packet["prequential_brier"], float(np.square(performed-outcomes[index]).mean()), "pre-update performed Brier")
        for action in range(5):
            mask = actions[index] == action
            marginal_count[action] += int(mask.sum())
            marginal_success[action] += outcomes[index][mask].sum(axis=0, dtype=np.int64)
        marginals.append(((marginal_success+.5)/(marginal_count[:, None]+1.)).tolist())
    compare(record["end_memory"], memory.snapshot(), "final uniform memory")
    check_work(record["diagnostics"], memory.seen, post_packets+count if internal else 0, internal, config)
    for label, snapshot in (("start_diagnostics", record["start_memory"]), ("diagnostics", record["end_memory"])):
        require(record[label]["memory_ids"] == snapshot["ids"], "diagnostic replay membership differs")
    compare(record["marginal_count"], marginal_count.tolist(), "training-only marginal counts")
    compare(record["marginal_success"], marginal_success.tolist(), "training-only marginal outcomes")
    compare(record["prequential_brier"], float(np.mean([packet["prequential_brier"] for packet in record["packets"]])), "phase prequential mean")
    require(record["diagnostics"]["cost"]["training_seconds"] >= record["start_diagnostics"]["cost"]["training_seconds"]
            and record["elapsed_seconds"] >= 0, "negative elapsed training work")
    require(trace["seed"] == record["seed"], "evaluator seed differs from phase")
    material, totals = trace_arrays(trace, evaluation, config)
    verify_trace_program(record, trace, material, config, initial_history, history, marginals)
    for name, value in totals.items():
        compare(record["evaluation_files"][name], value, "phase evaluation work")
    return history, totals


def read_records(folder, config, identity, summary):
    expected = expected_paths(config)
    require(summary["array_archive_prefix"] == "arrays", "portable numerical prefix differs")
    listed = summary["result_hashes"]
    require(len(listed) == len(expected) and len({row["path"] for row in listed}) == len(listed)
            and {row["path"]:row["kind"] for row in listed} == expected, "complete original artifact coverage differs")
    hashes = {row["path"]:row["sha256"] for row in listed}
    for name, value in hashes.items():
        relative_path(name)
        valid_hash(value)
    records = {}
    def put(path, record, kind):
        require(path in expected and expected[path] == kind and path not in records, "unexpected or duplicate portable record")
        if kind != "evaluation":
            require(record["identity"] == identity, "original JSON identity differs")
        require(sha(original_json(record)) == hashes[path], "original JSON bytes differ from hash")
        records[path] = record
    with gzip.open(folder/"raw_results.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            record = decode(line)
            put(phase_path(record), record, "phase")
    with gzip.open(folder/"evaluation_records.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            wrapper = decode(line)
            require(set(wrapper) == {"path", "record"}, "unexpected evaluator wrapper")
            put(wrapper["path"], wrapper["record"], "evaluation")
    for path, record in read(folder/"job_records.json").items():
        put(path, record, "job")
    require(set(records) == {path for path,kind in expected.items() if kind in ("phase", "evaluation", "job")}, "missing JSON records")
    arrays = {path for path,kind in expected.items() if kind in ("evaluation_arrays", "training_arrays")}
    present = {path.relative_to(folder/"arrays").as_posix() for path in (folder/"arrays").rglob("*") if path.is_file()}
    require(present == arrays, "numerical archive file set differs")
    for path in arrays:
        require(sha((folder/"arrays"/path).read_bytes()) == hashes[path], "original numerical bytes changed")
    return records, hashes


def verify_timelines(folder, records, hashes, config, identity, affinity):
    # Imports remain stdlib-only; deepcopy preserves independent NumPy generators.
    import copy

    phase_rows, phase_counts, streams = [], {}, {}
    for seed in config["seeds"]:
        for model in config["models"]:
            root = f"{model}_{seed}"
            job = records[root+"/result.json"]
            planned = plan(seed, config)
            names = {phase["relative"] for phase in planned}
            require(job["seed"] == seed and job["model"] == model and job["affinity"] == affinity
                    and set(job["phase_hashes"]) == set(job["checkpoint_hashes"]) == names
                    and set(job["insertion"]) == set(ARMS) and set(job["final_signatures"]) == {*ARMS, "fresh"}, "job provenance/coverage differs")
            state = Memory(seed, config["memory_packets"])
            history, previous, prefix_memory, prefix_history, prefix_end = None, None, None, None, None
            prior_arm, post, maintenance_history = "prefix", 0, None
            inserted = []
            for phase in planned:
                arm = phase["arm"]
                path = root+"/"+phase["relative"]+"/"
                record = records[path+"result.json"]
                require(record["seed"] == seed and record["model"] == model, "phase job key differs")
                require(job["phase_hashes"][phase["relative"]] == hashes[path+"result.json"]
                        and job["checkpoint_hashes"][phase["relative"]] == hashes[path+"checkpoint.pt"], "job-bound artifact hash differs")
                require(set(record["artifact_hashes"]) == {"before.pt", "evaluations.json", "evaluations.npz", "training.npz"}, "phase artifact binding coverage differs")
                for filename, value in record["artifact_hashes"].items():
                    require(value == hashes[path+filename], "phase artifact hash differs")
                compare(record["training_file"], dict(path="training.npz", sha256=hashes[path+"training.npz"]), "training file binding")
                for field, filename in (("json", "evaluations.json"), ("arrays", "evaluations.npz")):
                    compare(record["evaluation_files"][field], dict(path=filename, sha256=hashes[path+filename]), "evaluator file binding")
                if arm != prior_arm:
                    if prior_arm == "prefix":
                        prefix_memory, prefix_history, prefix_end = copy.deepcopy(state), history, previous
                        require(job["parent_signature"] == prefix_end["end_signature"], "shared prefix parent changed")
                    else:
                        require(job["final_signatures"][prior_arm] == previous["end_signature"], "continuous trajectory endpoint changed")
                    post = 0
                    if arm == "fresh":
                        state, history = Memory(seed, config["memory_packets"]), maintenance_history
                        require(record["start_diagnostics"]["cost"]["arrivals"] == 0
                                and record["start_diagnostics"]["optimizer_bytes"] == 0, "fresh qualification inherits training state")
                    else:
                        state, history = copy.deepcopy(prefix_memory), prefix_history
                        insertion = job["insertion"][arm]
                        compare(insertion, dict(signature=record["start_signature"], diagnostics=record["start_diagnostics"]), "shared insertion record")
                        compare(record["start_diagnostics"]["cost"], prefix_end["diagnostics"]["cost"], "insertion preserves cumulative work")
                        ext = record["start_diagnostics"]["core_residual"]
                        require(ext["core_hash"] == ext["core_at_fork_hash"] == prefix_end["diagnostics"]["final_hash"]
                                and ext["residual_hash"] == ext["residual_initial_hash"]
                                and ext["residual_displacement_from_initial"] == 0., "insertion changes core or nonzero residual")
                        inserted.append((ext["core_hash"], ext["residual_initial_hash"], ext["core_optimizer_at_fork_hash"], ext["parameters"]["total"]))
                    previous = None
                if previous is not None:
                    require(record["start_signature"] == previous["end_signature"], "continuous full-state chain differs")
                    compare(record["start_diagnostics"], previous["diagnostics"], "continuous diagnostics")
                elif arm == "prefix":
                    require(record["start_diagnostics"]["final_hash"] == record["start_diagnostics"]["initial_hash"]
                            and record["start_diagnostics"]["optimizer_bytes"] == 0, "ordinary prefix was pretrained")
                training = read_arrays(folder/"arrays"/(path+"training.npz"))
                trace = records[path+"evaluations.json"]
                evaluation = read_arrays(folder/"arrays"/(path+"evaluations.npz"))
                history, totals = verify_phase(record, phase, training, trace, evaluation, state, history, post, config, identity)
                phase_rows.append(record)
                phase_counts[path+"result.json"] = totals
                if arm != "prefix":
                    post += record["batches_done"]
                if arm == "joint" and phase["phase"] == "maintenance":
                    maintenance_history = history
                key = seed, phase["phase"]
                stream = digest(dict(queries=record["batch_sha256"], actions=training["actions"].tolist(), outcomes=training["outcomes"].tolist()))
                require(key not in streams or streams[key] == stream, "training inputs differ across paired arms/architectures")
                streams[key] = stream
                previous, prior_arm = record, arm
            require(len(set(inserted)) == 1, "carried arms differ in inserted state or total capacity")
            require(job["final_signatures"][prior_arm] == previous["end_signature"], "fresh endpoint differs")
    return phase_rows, phase_counts


def resource_row(record, config):
    before, after = record["start_diagnostics"], record["diagnostics"]
    delta = {name:after["cost"][name]-before["cost"][name] for name in after["cost"]}
    extension = after.get("core_residual")
    multiplier = 1 if extension is None else 2
    prequential = dict(calls=record["batches_done"], query_presentations=record["size"], support_presentations=record["size"],
        observed_support_presentations=config["batch_size"]*sum(packet["support_sha256"] is not None for packet in record["packets"]))
    evaluation = {name:record["evaluation_files"][name] for name in ("calls", "query_presentations", "support_presentations", "observed_support_presentations")}
    parameters = dict(total=after["parameters"], total_trainable=after["parameters"], core=after["parameters"],
                      core_trainable=after["parameters"], residual=0, residual_trainable=0) if extension is None else extension["parameters"]
    calls = 2*record["batches_done"]*config["updates_per_batch"]
    result = {name:record[name] for name in ("seed", "model", "arm", "phase")}
    result.update(parameters=parameters, model_bytes=after["model_bytes"], optimizer_bytes=after["optimizer_bytes"],
        core_optimizer_bytes=after["optimizer_bytes"] if extension is None else extension["core_optimizer_bytes"],
        residual_optimizer_bytes=0 if extension is None else extension["residual_optimizer_bytes"],
        measurement_snapshot_bytes=None if extension is None else extension["measurement_snapshot_bytes"],
        peak_replay_bytes=after["peak_replay_bytes"], history_bytes=after["history_bytes"], training_cost=delta,
        training_logical_probability_calls=calls, prequential=prequential, evaluation=evaluation,
        raw_pathways_per_logical_call=multiplier,
        total_raw_path_forward_calls=multiplier*(calls+prequential["calls"]+evaluation["calls"]),
        total_raw_path_query_presentations=multiplier*(delta["query_presentations"]+prequential["query_presentations"]+evaluation["query_presentations"]),
        total_raw_path_support_presentations=multiplier*(delta["support_presentations"]+prequential["support_presentations"]+evaluation["support_presentations"]))
    if extension is not None:
        result["path_work"] = {name:extension["work"][name]-before["core_residual"]["work"][name] for name in extension["work"]}
        result["cumulative_path_work"] = extension["work"]
        result["component_displacement"] = {name:value for name,value in extension.items() if "displacement" in name}
    return result


def verify_statistics(summary, rows, config):
    expected = recompute_statistics(rows, config)
    for key in ("rows", "levels", "comparisons", "fresh_comparisons", "descriptive_cue_slices", "eligible_development_cohort"):
        compare(summary[key], expected[key], key)
    for key in ("screen", "attribution"):
        require(set(summary[key]) == set(config["models"]), "decision architecture coverage differs")
        for model, decision in expected[key].items():
            require(set(summary[key][model]) == {*decision, "interpretation"}, "unexpected decision fields")
            compare({name:summary[key][model][name] for name in decision}, decision, key+"/"+model)
    require(set(summary["qualification"]) == {"groups", "by_model", "interpretation"}, "qualification fields differ")
    compare({name:summary["qualification"][name] for name in expected["qualification"]}, expected["qualification"], "fresh qualification")
    def order(row):
        return row["seed"], row["model"], row["arm"], row["phase"]
    compare(summary["resource_rows"], [resource_row(row, config) for row in sorted(rows, key=order)], "resource accounting")
    return expected


def verify_file_integrity(folder, records, hashes, config):
    result = read(folder/"file_integrity.json")
    require(result["passed"] is True, "original run file-integrity check failed")
    entries = {row["path"]:row for row in result["files"]}
    copied = {"manifest.json", "completion.json", "training_source.zip", "analysis_lock.json", "analysis_at_lock.zip", "protocol_lock.json"}
    require(len(entries) == len(result["files"]) and set(entries) == set(expected_paths(config))|copied,
            "original run integrity coverage differs")
    for name, row in entries.items():
        relative_path(name)
        valid_hash(row["sha256"])
        require(row["valid"] is True and type(row["bytes"]) is int and row["bytes"] > 0, "invalid original integrity entry")
        if name in hashes:
            require(row["sha256"] == hashes[name], "original-file digest differs from exported reference")
        data = (folder/name).read_bytes() if name in copied else original_json(records[name]) if name in records else (
            (folder/"arrays"/name).read_bytes() if name.endswith(".npz") else None)
        if data is not None:
            require(row["sha256"] == sha(data) and row["bytes"] == len(data), "original JSON/metadata bytes differ")
    return entries


def verify_audit(audit, records, hashes, config, identity, phase_counts, integrity, manifest_hash, analysis):
    require(audit["passed"] is True and audit["identity"] == identity and audit["manifest_sha256"] == manifest_hash
            and audit["audit_source_sha256"] == analysis["files"]["scripts/audit_core_residual.py"], "unrelated or unsealed tensor audit")
    require(len(audit["result_hashes"]) == len(hashes) and {row["path"]:row["sha256"] for row in audit["result_hashes"]} == hashes
            and {row["path"]:row["kind"] for row in audit["result_hashes"]} == expected_paths(config), "tensor audit artifact coverage differs")
    counts = expected_counts(config)
    checks = audit["checks"]
    jobs = {(seed, model) for seed in config["seeds"] for model in config["models"]}
    require(len(checks) == len(jobs) and {(row["seed"], row["model"]) for row in checks} == jobs, "tensor audit job coverage differs")
    totals = {}
    for check in checks:
        seed, model = check["seed"], check["model"]
        root = f"{model}_{seed}"
        job = records[root+"/result.json"]
        require(check["result_sha256"] == hashes[root+"/result.json"] and check["parent_signature"] == job["parent_signature"]
                and check["insertion"] == {arm:job["insertion"][arm]["signature"] for arm in ARMS}, "tensor audit donor/fork binding differs")
        planned = plan(seed, config)
        require(len(check["phases"]) == len(planned), "tensor audit phase coverage differs")
        expected = dict(forks_verified=3, fresh_initializations_verified=1, before_checkpoints=len(planned),
            final_checkpoints=len(planned), ordinary_packets=0, first_prequential_predictions_recomputed=len(planned),
            evaluation_calls=0, evaluation_prediction_calls_recomputed=0, intermediate_evaluation_calls_arithmetic_only=0,
            query_presentations=0, support_presentations=0, observed_support_presentations=0)
        for row, phase in zip(check["phases"], planned):
            prefix = root+"/"+phase["relative"]+"/"
            require(row["arm"] == phase["arm"] and row["phase"] == phase["phase"], "tensor audit phase order differs")
            expected["ordinary_packets"] += phase["size"]//config["batch_size"]
            require(row["packets"] == phase["size"]//config["batch_size"], "tensor audit packet count differs")
            for label, filename in (("result_sha256", "result.json"), ("before_sha256", "before.pt"),
                    ("checkpoint_sha256", "checkpoint.pt"), ("training_sha256", "training.npz"),
                    ("evaluations_json_sha256", "evaluations.json"), ("evaluations_npz_sha256", "evaluations.npz")):
                require(row[label] == integrity[prefix+filename]["sha256"], "tensor audit phase file binding differs")
            for label in ("causal_stream_and_replay_verified", "exact_full_state_chain_verified",
                    "trainability_adam_and_work_verified", "first_prequential_prediction_recomputed"):
                require(row[label] is True, "tensor audit verification assertion missing")
            work = phase_counts[prefix+"result.json"]
            intermediate = phase["size"]//config["probe_every"]-1
            coverage = dict(evaluation_calls=work["calls"], evaluation_prediction_calls_recomputed=work["calls"]-intermediate,
                intermediate_evaluation_calls_arithmetic_only=intermediate,
                **{name:work[name] for name in ("query_presentations", "support_presentations", "observed_support_presentations")})
            for label, value in coverage.items():
                compare(row[label], value, "tensor phase coverage."+label)
                expected[label] += value
        for label, value in expected.items():
            compare(check[label], value, "tensor job coverage."+label)
            totals[label] = totals.get(label, 0)+value
    for label, value in totals.items():
        compare(audit[label], value, "tensor cohort coverage."+label)
    require(audit["jobs"] == counts["jobs"] and totals["ordinary_packets"] == counts["training_packets"]
            and totals["before_checkpoints"] == counts["total_phases"], "tensor audit cohort count differs")
    return totals


def verify_packaging(folder):
    names = ("verifier_lock.json", "verifier_source.zip", "verify_archive.py")
    present = [(folder/name).is_file() for name in names]
    require(all(present) or not any(present), "partial standalone verifier seal")
    if not any(present):
        return False
    lock = read(folder/"verifier_lock.json")
    source, tests = "scripts/verify_core_residual_archive.py", "tests/test_core_residual_archive.py"
    require(set(lock["files"]) == {source, tests}, "standalone verifier source coverage differs")
    verify_zip(folder/"verifier_source.zip", lock["files"])
    require(sha((folder/"verify_archive.py").read_bytes()) == sha(Path(__file__).read_bytes()) == lock["files"][source],
            "copied/executing verifier differs from separate source seal")
    return True


def verify(folder):
    folder = Path(folder).resolve()
    artifacts, packaging = verify_artifacts(folder), verify_packaging(folder)
    manifest, config, summary = read(folder/"manifest.json"), read(folder/"config.json"), read(folder/"summary.json")
    identity = manifest["identity"]
    compare(manifest["config"], config, "manifest config")
    compare(summary["config"], config, "summary config")
    compare(summary["identity"], identity, "summary identity")
    require(set(identity) == {"config_sha256", "source_sha256", "runtime_sha256"}, "scientific identity fields differ")
    for field, value in (("config_sha256", config), ("source_sha256", manifest["source_files"]), ("runtime_sha256", manifest["runtime"])):
        require(identity[field] == digest(value), "manifest identity digest mismatch")
    require({"acp_cl/core_residual/learner.py", "acp_cl/core_residual/study.py", "acp_cl/core_residual/evaluation.py"}
            <= manifest["source_files"].keys(), "core/residual training source missing")
    verify_zip(folder/"training_source.zip", manifest["source_files"])
    protocol, analysis = read(folder/"protocol_lock.json"), read(folder/"analysis_lock.json")
    require(all(protocol[key] == value for key,value in identity.items())
            and protocol["protocol_sha256"] == sha((folder/"protocol_at_lock.md").read_bytes()), "protocol lock mismatch")
    require(analysis["config"] == config and analysis["config_sha256"] == identity["config_sha256"], "analysis configuration mismatch")
    required = {"scripts/summarize_core_residual.py", "tests/test_core_residual_summary.py", "scripts/audit_core_residual.py",
                "tests/test_core_residual_learner.py", "tests/test_core_residual_evaluation.py",
                "tests/test_core_residual_study.py", "tests/test_core_residual_audit.py"}
    require(required <= analysis["files"].keys() and any(name.startswith("docs/") and value == protocol["protocol_sha256"]
            for name,value in analysis["files"].items()), "prospective analysis/audit/protocol coverage missing")
    require("scripts/verify_core_residual_archive.py" not in analysis["files"]
            and "tests/test_core_residual_archive.py" not in analysis["files"], "packaging code included in scientific lock")
    verify_zip(folder/"analysis_at_lock.zip", analysis["files"])
    require(manifest["created_utc"] == analysis["locked_utc"] == protocol["locked_utc"], "lock creation times differ")
    complete, counts = read(folder/"completion.json"), expected_counts(config)
    locked, completed = (datetime.fromisoformat(value) for value in (manifest["created_utc"], complete["completed_utc"]))
    require(locked.utcoffset() is not None and completed.utcoffset() is not None and completed >= locked,
            "completion precedes scientific lock or timestamp lacks timezone")
    compare(complete["identity"], identity, "completion identity")
    compare(summary["counts"], counts, "scientific counts")
    for name, value in counts.items():
        compare(complete[name], value, "completion."+name)
    jobs = complete["job_records"]
    require(len(jobs) == counts["jobs"] and {(row["seed"], row["model"]) for row in jobs}
            == {(seed,model) for seed in config["seeds"] for model in config["models"]}
            and all(row["affinity"] == manifest["runtime"]["affinity"] for row in jobs), "completion job/runtime coverage differs")
    records, hashes = read_records(folder, config, identity, summary)
    integrity = verify_file_integrity(folder, records, hashes, config)
    rows, phase_counts = verify_timelines(folder, records, hashes, config, identity, manifest["runtime"]["affinity"])
    statistics = verify_statistics(summary, rows, config)
    audit = verify_audit(read(folder/"audit.json"), records, hashes, config, identity, phase_counts, integrity,
                         sha((folder/"manifest.json").read_bytes()), analysis)
    recorded = summary["record_integrity"]
    expected_integrity = dict(phases=len(rows), training_packets=sum(row["batches_done"] for row in rows),
        evaluation_calls=sum(value["calls"] for value in phase_counts.values()),
        evaluation_query_presentations=sum(value["query_presentations"] for value in phase_counts.values()),
        evaluation_support_presentations=sum(value["support_presentations"] for value in phase_counts.values()),
        raw_numerical_reconstruction_passed=True)
    require(set(recorded) == {*expected_integrity, "scope"}, "record-integrity schema differs")
    compare({key:recorded[key] for key in expected_integrity}, expected_integrity, "record integrity")
    return dict(passed=True, identity=identity, standalone_verifier_sealed=packaging, artifact_files=artifacts,
        original_run_files=len(integrity), numerical_arrays=2*counts["total_phases"], **counts,
        tensor_audit_coverage=audit,
        primary_screen={model:value["passed"] for model,value in statistics["screen"].items()},
        attribution_screen={model:value["passed"] for model,value in statistics["attribution"].items()},
        scope="Portable file/source/protocol/analysis identity and coverage checks; original JSON/NPZ digests; "
              "uniform replay membership/draws and causal context chains; every raw all-action, performed-action, "
              "survival, mask and marginal score; held-out AUC, support-probe aggregation, seed bootstrap, fresh "
              "qualification, fixed primary/attribution gates and declared resource arithmetic. Saved model/optimizer "
              "and simulator checks are bound to the separately archived tensor audit and its exact coverage. "
              "No repository imports, pickle loading, ordinary retraining, independently reconstructed physical "
              "world or overwritten intermediate model states. File seals establish internal consistency, not an external signature.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.input), indent=2, allow_nan=False))
