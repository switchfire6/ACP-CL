"""Locked continuous core/residual development analysis and numerical archive checks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(16001, 16007))
MODELS = ("conditional", "recurrent")
ARMS = ("joint", "separate", "fixed_features")
PHASES = ("maintenance", "novel", "return")
COMPARISONS = {"separate_minus_joint": ("separate", "joint"),
               "separate_minus_fixed_features": ("separate", "fixed_features"),
               "fixed_features_minus_joint": ("fixed_features", "joint")}
PROBE_FIELDS = ("brier", "focus_brier", "survival", "focus_survival", "valid_brier",
                "valid_survival", "no_transfer", "clairvoyant_upper")
REQUIRED_ANALYSIS = ("scripts/summarize_core_residual.py", "tests/test_core_residual_summary.py")
PROBABILITY_TOLERANCE = np.finfo(np.float32).eps


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(np.isfinite(value), "nonfinite JSON value")


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    finite_tree(value)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    finite_tree(value)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def mean(values):
    values = list(values)
    require(bool(values), "cannot average an empty group")
    return float(np.mean(values))


def paired(values, seeds=None):
    """Inherited paired-seed bootstrap arithmetic, with no interval-based gate."""
    values = np.asarray(values, dtype=float).copy()
    require(values.ndim == 1 and len(values) and np.isfinite(values).all(), "invalid seed values")
    values[np.abs(values) < 1e-12] = 0.
    rng = np.random.default_rng(27192026)
    boot = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    result = dict(mean=mean(values), lower=float(np.quantile(boot, .025)),
        upper=float(np.quantile(boot, .975)), positive=int((values > 0).sum()),
        negative=int((values < 0).sum()), n=len(values), differences=values.tolist())
    if seeds is not None:
        require(len(seeds) == len(values), "seed labels do not match values")
        result["seeds"] = list(seeds)
    return result


def probe(value, field, flipped=False, entry=False):
    return mean(row["flipped"][field] if flipped else
                row["curve"][0 if entry else -1]["metrics"][field] for row in value["replicates"])


def area(curve, field):
    x = np.asarray([row["arrivals"] for row in curve], dtype=float)
    y = np.asarray([row["metrics"][field] for row in curve], dtype=float)
    require(len(x) >= 2 and x[0] == 0 and (np.diff(x) > 0).all() and np.isfinite(y).all(),
            "invalid held-out learning curve")
    return float(np.sum(np.diff(x)*(y[:-1]+y[1:])/2)/x[-1])


def phase_metrics(record):
    """Held-out post-update AUC remains distinct from pre-update training-stream error."""
    result = dict(brier_auc=area(record["curve"], "focus_brier"),
        all_brier_auc=area(record["curve"], "brier"), survival_auc=area(record["curve"], "survival"),
        before=probe(record["before"], "focus_brier"), after=probe(record["after"], "focus_brier"),
        endpoint=record["curve"][-1]["metrics"]["focus_brier"],
        before_survival=probe(record["before"], "survival"),
        after_survival=probe(record["after"], "survival"),
        marginal_gain=probe(record["after"], "marginal_brier")-probe(record["after"], "brier"),
        prequential_brier=mean(row["prequential_brier"] for row in record["packets"]))
    for mode in ("0", "1"):
        for when in ("before", "after"):
            result[f"valid_{when}_mode_{mode}"] = record[f"valid_{when}"][mode]["valid_brier"]
            result[f"valid_survival_{when}_mode_{mode}"] = record[f"valid_{when}"][mode]["valid_survival"]
        result[f"valid_damage_mode_{mode}"] = result[f"valid_after_mode_{mode}"]-result[f"valid_before_mode_{mode}"]
    for when in ("before", "after"):
        result[f"valid_{when}"] = mean(result[f"valid_{when}_mode_{mode}"] for mode in ("0", "1"))
    result["valid_damage"] = result["valid_after"]-result["valid_before"]
    if record["cue"] is not None:
        result["cue_effect"] = probe(record["after"], "focus_brier", flipped=True)-result["after"]
        result["cue_effect_before"] = probe(record["before"], "focus_brier", flipped=True)-result["before"]
        result["cue_learning_gain"] = result["cue_effect"]-result["cue_effect_before"]
    if record.get("novel_after_return") is not None:
        value = record["novel_after_return"]
        result["novel_after_return_brier"] = probe(value, "focus_brier")
        result["novel_after_return_survival"] = probe(value, "survival")
        result["novel_after_return_cue_effect"] = probe(value, "focus_brier", flipped=True)-result["novel_after_return_brier"]
    return result


def upper_gate(metric, maximum):
    return dict(passed=metric["mean"] <= maximum, value=metric["mean"], maximum=maximum,
        lower=metric["lower"], upper=metric["upper"], uncertainty_crosses_limit=metric["upper"] > maximum)


def lower_gate(metric, minimum):
    return dict(passed=metric["mean"] >= minimum, value=metric["mean"], minimum=minimum,
        lower=metric["lower"], upper=metric["upper"], uncertainty_crosses_limit=metric["lower"] < minimum)


def finish_screen(gates, interpretation):
    failed = [name for name, value in gates.items() if not value["passed"]]
    return dict(passed=not failed, gates=gates, failed=failed, interpretation=interpretation)


def primary_screen(comparison, qualified, eligible=True):
    novel, returned = comparison["novel"], comparison["return"]
    improved = sum(value < 0 for value in novel["brier_auc"]["differences"])
    gates = dict(complete_development_cohort=dict(passed=bool(eligible)),
        fresh_qualification=dict(passed=bool(qualified)),
        acquisition_gain=upper_gate(novel["brier_auc"], -.002),
        acquisition_consistency=dict(passed=improved >= 5, improved_seeds=improved, minimum=5,
                                    n=novel["brier_auc"]["n"]))
    for mode in ("0", "1"):
        gates[f"valid_old_mode_{mode}"] = upper_gate(novel[f"valid_after_mode_{mode}"], .005)
    gates["return_prediction"] = upper_gate(returned["all_brier_auc"], .005)
    gates["post_return_novel_retention"] = upper_gate(returned["novel_after_return_brier"], .005)
    for phase in ("novel", "return"):
        gates[f"survival_{phase}"] = lower_gate(comparison[phase]["survival_auc"], -.01)
    return finish_screen(gates, "Fixed separate-versus-joint development allocation screen; "
        "not a significance or noninferiority test. No ablation may replace the primary candidate.")


def attribution_screen(comparison, eligible=True):
    metric = comparison["novel"]["brier_auc"]
    improved = sum(value < 0 for value in metric["differences"])
    gates = dict(complete_development_cohort=dict(passed=bool(eligible)),
        feature_adaptation_gain=upper_gate(metric, -.001),
        feature_adaptation_consistency=dict(passed=improved >= 5, improved_seeds=improved, minimum=5, n=metric["n"]))
    return finish_screen(gates, "Separate-versus-fixed-random-visual-features attribution screen; "
        "passing cannot rescue a failed primary outcome screen.")


def qualification(rows, models):
    groups, by_model = [], {}
    for model in models:
        fresh = [row for row in rows if row["model"] == model and row["arm"] == "fresh" and row["phase"] == "novel"]
        require(bool(fresh), "fresh qualification records are missing")
        model_groups = [("overall", fresh)]
        model_groups += [(f"cue_{cue}", [row for row in fresh if row["cue"] == cue]) for cue in sorted({row["cue"] for row in fresh})]
        for name, group in model_groups:
            gain, cue = mean(row["marginal_gain"] for row in group), mean(row["cue_effect"] for row in group)
            groups.append(dict(model=model, group=name, n=len(group), seeds=[row["seed"] for row in group],
                marginal_gain=gain, cue_effect=cue, passed=gain >= .02 and cue >= .002))
        by_model[model] = all(row["passed"] for row in groups if row["model"] == model)
    return dict(groups=groups, by_model=by_model,
                interpretation="Fresh joint architecture; learnability only, not a retention control. Two seeds per scientific cue.")


def scientific_eligibility(config):
    """Compare the locked scientific configuration, permitting execution-only portability."""
    expected_path = ROOT/"configs/core_residual_development.json"
    if not expected_path.is_file():
        return False
    expected = read(expected_path)
    execution = {"study", "threads", "workers", "cpu_affinity"}
    def scientific(c):
        return {key: value for key, value in c.items() if key not in execution}
    return (config.get("kind") == "development" and config.get("seeds") == SEEDS
            and config.get("models") == list(MODELS) and scientific(config) == scientific(expected))


def expected_learning_keys(config):
    return {(seed, model, arm, phase) for seed in config["seeds"] for model in config["models"]
            for arm in (*ARMS, "fresh") for phase in (("novel",) if arm == "fresh" else PHASES)}


def analyze_records(records, config):
    records = list(records.values()) if isinstance(records, dict) else list(records)
    rows, keys = [], []
    for record in records:
        finite_tree(record)
        if record["phase"].startswith("prefix_"):
            continue
        key = tuple(record[name] for name in ("seed", "model", "arm", "phase"))
        keys.append(key)
        row = {name: record[name] for name in ("seed", "model", "arm", "phase", "cue")}
        row.update(phase_metrics(record))
        rows.append(row)
    require(len(keys) == len(set(keys)) and set(keys) == expected_learning_keys(config),
            "every learned arm/phase and fresh novel reference must occur exactly once")
    rows.sort(key=lambda row: tuple(row[name] for name in ("seed", "model", "arm", "phase")))
    by_key = {tuple(row[name] for name in ("seed", "model", "arm", "phase")): row for row in rows}
    seeds, models = config["seeds"], config["models"]
    qualified = qualification(rows, models)
    levels, comparisons, fresh_comparisons, slices = {}, {}, {}, {}
    for model in models:
        levels[model], comparisons[model], fresh_comparisons[model] = {}, {}, {}
        for phase in PHASES:
            arms = (*ARMS, "fresh") if phase == "novel" else ARMS
            levels[model][phase] = {}
            for arm in arms:
                sample = by_key[seeds[0], model, arm, phase]
                fields = [key for key in sample if key not in ("seed", "model", "arm", "phase", "cue")]
                levels[model][phase][arm] = {field: paired([by_key[seed, model, arm, phase][field] for seed in seeds], seeds)
                                            for field in fields}
        for name, (first, second) in COMPARISONS.items():
            comparisons[model][name] = {phase: {field: paired([
                by_key[seed, model, first, phase][field]-by_key[seed, model, second, phase][field]
                for seed in seeds], seeds) for field in levels[model][phase][first]} for phase in PHASES}
        for arm in ARMS:
            fresh_comparisons[model][f"{arm}_minus_fresh"] = {field: paired([
                by_key[seed, model, arm, "novel"][field]-by_key[seed, model, "fresh", "novel"][field]
                for seed in seeds], seeds) for field in levels[model]["novel"][arm]}
        slices[model] = {}
        for cue in sorted({row["cue"] for row in rows if row["phase"] == "novel"}):
            cue_seeds = [seed for seed in seeds if by_key[seed, model, "joint", "novel"]["cue"] == cue]
            slices[model][str(cue)] = dict(seeds=cue_seeds, n=len(cue_seeds),
                separate_minus_joint={phase: {field: mean(by_key[seed, model, "separate", phase][field]
                    -by_key[seed, model, "joint", phase][field] for seed in cue_seeds)
                    for field in levels[model][phase]["joint"]} for phase in PHASES})
    eligible = scientific_eligibility(config)
    screen = {model: primary_screen(comparisons[model]["separate_minus_joint"], qualified["by_model"][model], eligible)
              for model in models}
    attribution = {model: attribution_screen(comparisons[model]["separate_minus_fixed_features"], eligible) for model in models}
    return dict(config=config, rows=rows, levels=levels, comparisons=comparisons,
        fresh_comparisons=fresh_comparisons, qualification=qualified, screen=screen,
        attribution=attribution, descriptive_cue_slices=slices, eligible_development_cohort=eligible,
        inference="Six independent seeds per architecture. Curves/supports/actions/horizons are not replicates. "
            "20,000 paired seed bootstrap resamples, RNG 27192026; intervals are descriptive. "
            "Held-out post-update AUC and performed-action prequential error are distinct outcomes.")


def study_api():
    from acp_cl.core_residual import study
    return study


def valid_hash(value, optional=False):
    require((optional and value is None) or (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)), "invalid SHA256 fingerprint")


def safe_path(directory, relative):
    path = (Path(directory)/relative).resolve()
    require(isinstance(relative, str) and not Path(relative).is_absolute()
            and path.is_relative_to(Path(directory).resolve()), "artifact path escapes its directory")
    return path


def checked_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected)
                and archive.testzip() is None, "archive file set or CRC mismatch")
        require(all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in expected.items()),
                "archive content hash mismatch")


def verify_analysis_lock(directory, config, protocol_hash):
    lock = read(Path(directory)/"analysis_lock.json")
    require(lock["config"] == config and lock["config_sha256"] == digest(config), "analysis configuration mismatch")
    files = lock["files"]
    require(set(REQUIRED_ANALYSIS) <= set(files), "prospective analysis files missing from lock")
    checked_zip(Path(directory)/"analysis_at_lock.zip", files)
    for name, value in files.items():
        path = safe_path(ROOT, name)
        require(path.is_file() and sha(path) == value, f"live analysis/config/protocol differs from lock: {name}")
    require(any(name.startswith("docs/") and value == protocol_hash for name, value in files.items()),
            "protocol missing from analysis lock")
    return lock


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        require(len(data.files) == len(set(data.files)), "duplicate numerical array name")
        return {name: np.array(data[name], copy=True) for name in data.files}


def array_hash(value):
    value = np.asarray(value)
    require(not value.dtype.hasobject, "object arrays are forbidden")
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode()+value.tobytes(order="C")).hexdigest()


def experience_hash(observations, actions, outcomes):
    value = hashlib.sha256()
    for array in (observations, actions, outcomes):
        value.update(str((array.shape, array.dtype.str)).encode())
        value.update(array.tobytes(order="C"))
    return value.hexdigest()


def check_probabilities(values, shape):
    values = np.asarray(values)
    require(values.shape == shape and values.dtype.kind == "f" and np.isfinite(values).all(),
            "invalid prediction array shape/type/values")
    require(values.min() >= -PROBABILITY_TOLERANCE and values.max() <= 1+PROBABILITY_TOLERANCE,
            f"prediction outside float32-roundoff bounds: min={values.min()}, max={values.max()}")
    require((np.diff(values, axis=-1) <= PROBABILITY_TOLERANCE).all(), "survival predictions are not nested")


def check_truth(values, shape):
    require(values.shape == shape and values.dtype.kind in "buif" and np.isin(values, (0, 1)).all()
            and (np.diff(values.astype(float), axis=-1) <= 0).all(), "invalid nested binary outcomes")


def close(actual, expected, message):
    require(np.shape(actual) == np.shape(expected) and np.isfinite(np.asarray(actual, dtype=float)).all()
            and np.allclose(actual, expected, rtol=0., atol=1e-12), message)


def recompute_metrics(probabilities, truth, affected, valid, marginal=None):
    check_probabilities(probabilities, (len(probabilities), 5, 3))
    check_truth(truth, probabilities.shape)
    for mask in (affected, valid):
        require(mask.dtype == np.bool_ and mask.shape == (len(probabilities),) and mask.any(), "invalid evaluator subset")
    errors = np.square(probabilities.astype(np.float64)-truth.astype(np.float64)).mean(axis=(1, 2))
    actions = probabilities[:, :, -1].argmax(axis=1)
    survival = truth[np.arange(len(actions)), actions, -1].astype(np.float64)
    result = dict(brier=mean(errors), focus_brier=mean(errors[affected]), survival=mean(survival),
        focus_survival=mean(survival[affected]), valid_brier=mean(errors[valid]),
        valid_survival=mean(survival[valid]), no_transfer=mean(truth[:, 0, -1]),
        clairvoyant_upper=mean(truth[:, :, -1].max(axis=1)))
    if marginal is not None:
        marginal = np.asarray(marginal, dtype=np.float64)
        require(marginal.shape == (5, 3) and np.isfinite(marginal).all() and (marginal >= 0).all()
                and (marginal <= 1).all(), "invalid training-only marginal probabilities")
        result["marginal_brier"] = float(np.square(marginal[None]-truth.astype(np.float64)).mean())
    return result


def verify_training(record, directory, config):
    descriptor = record["training_file"]
    require(descriptor["path"] == "training.npz" and sha(directory/descriptor["path"]) == descriptor["sha256"],
            "training numerical file hash mismatch")
    data = read_arrays(directory/descriptor["path"])
    require(set(data) == {"probabilities", "actions", "outcomes"}, "training numerical array coverage mismatch")
    packets, size = record["size"]//config["batch_size"], config["batch_size"]
    p, a, y = (data[key] for key in ("probabilities", "actions", "outcomes"))
    check_probabilities(p, (packets, size, 5, 3))
    check_truth(y, (packets, size, 3))
    require(a.shape == (packets, size) and a.dtype.kind in "iu" and (a >= 0).all() and (a < 5).all(),
            "invalid performed actions")
    selected = p[np.arange(packets)[:, None], np.arange(size)[None], a].astype(np.float64)
    errors = np.square(selected-y.astype(np.float64)).mean(axis=(1, 2))
    close([row["prequential_brier"] for row in record["packets"]], errors,
          "prequential packet error differs from raw performed predictions")
    close(record["prequential_brier"], errors.mean(), "phase prequential mean differs from packet errors")
    counts = np.array([(a == action).sum() for action in range(5)])
    success = np.array([y[a == action].sum(axis=0) for action in range(5)])
    require(np.array_equal(counts, record["marginal_count"]) and np.array_equal(success, record["marginal_success"]),
            "training-only marginal totals differ from arrivals")
    return data


def verify_evaluations(record, directory, config, training):
    descriptor = record["evaluation_files"]
    for key, name in (("json", "evaluations.json"), ("arrays", "evaluations.npz")):
        require(descriptor[key]["path"] == name and sha(directory/name) == descriptor[key]["sha256"],
                "evaluation file hash mismatch")
    payload, arrays = read(directory/"evaluations.json"), read_arrays(directory/"evaluations.npz")
    require(payload["seed"] == record["seed"], "evaluation seed mismatch")
    rows, expected_arrays = payload["records"], set()
    for index, row in enumerate(rows):
        require(row["index"] == index, "evaluation trace order changed")
        names = set(row["array_sha256"])
        required = {"probabilities", "truth", "observations", "affected", "valid"}
        if row["support_fingerprint"] is not None:
            required |= {"support_observations", "support_actions", "support_outcomes"}
        require(names == required, "evaluation per-call array coverage mismatch")
        keys = {name: f"e{index}_{name}" for name in names}
        require(set(keys.values()) <= set(arrays), "evaluation numerical array missing")
        values = {name: arrays[key] for name, key in keys.items()}
        expected_arrays.update(keys.values())
        require(all(array_hash(values[name]) == value for name, value in row["array_sha256"].items()),
                "evaluation numerical array hash mismatch")
        size = config["eval_size"]
        require(values["observations"].shape == (size, 4, 12, 16) and values["observations"].dtype == np.uint8
                and row["query_sha256"] == array_hash(values["observations"]), "evaluation query fingerprint mismatch")
        valid_hash(row["model_sha256"])
        require(values["probabilities"].shape == (size, 5, 3), "evaluation query count mismatch")
        count = 0
        if row["support_fingerprint"] is not None:
            x, a, y = (values[name] for name in ("support_observations", "support_actions", "support_outcomes"))
            count = len(a)
            require(count > 0 and x.shape == (count, 4, 12, 16) and x.dtype == np.uint8
                    and a.shape == (count,) and a.dtype.kind in "iu" and (a >= 0).all() and (a < 5).all(),
                    "invalid evaluation causal support")
            check_truth(y, (count, 3))
            require(experience_hash(x, a, y) == row["support_fingerprint"], "evaluation support fingerprint mismatch")
        require(row["query_presentations"] == size and row["observed_support_presentations"] == count
                and row["support_presentations"] == (count or config["batch_size"]), "evaluation presentation count mismatch")
        calculated = recompute_metrics(values["probabilities"], values["truth"], values["affected"], values["valid"], row["marginal"])
        require(set(calculated) == set(row["metrics"]), "evaluation metric fields changed")
        for field, value in calculated.items():
            close(row["metrics"][field], value, f"evaluation {field} differs from raw predictions")
    require(set(arrays) == expected_arrays, "unreferenced evaluation numerical arrays")
    counts = dict(calls=len(rows), **{name: sum(row[name] for row in rows) for name in
        ("query_presentations", "support_presentations", "observed_support_presentations")})
    require(all(payload[name] == value == descriptor[name] for name, value in counts.items()),
            "evaluation aggregate work mismatch")
    cursor = 0

    def consume(metrics, law, model_hash, cue, branch, flipped=False, marginal=None, support_hash=False):
        nonlocal cursor
        require(cursor < len(rows), "missing referenced evaluation trace")
        row = rows[cursor]
        require(metrics["trace"] == cursor and {key: value for key, value in metrics.items() if key != "trace"} == row["metrics"],
                "phase metric/trace reference differs from raw evaluation")
        require(digest(row["law"]) == digest(law) and row["cue"] == cue and row["branch"] == branch
                and row["flipped"] is flipped, "evaluation law/cue/branch changed")
        if model_hash is not None:
            require(row["model_sha256"] == model_hash, "evaluation endpoint model hash mismatch")
        require((marginal is None) == (row["marginal"] is None), "evaluation marginal presence mismatch")
        if marginal is not None:
            close(row["marginal"], marginal, "evaluation marginal used wrong training prefix")
        if support_hash is not False:
            require(row["support_fingerprint"] == support_hash, "live-history evaluation used different support")
        cursor += 1

    def consume_probe(value, law, model_hash, cue, branch, marginal=None, live_support=False):
        require(digest(value["law"]) == digest(law) and value["model_sha256"] == model_hash
                and len(value["replicates"]) == config["support_replicates"], "probe endpoint or replicate mismatch")
        for replicate in value["replicates"]:
            require([point["feedback"] for point in replicate["curve"]] == sorted(set((0, config["batch_size"]//4,
                    config["batch_size"]//2, config["batch_size"]))), "support refresh doses changed")
            for point in replicate["curve"]:
                consume(point["metrics"], law, model_hash, cue, branch, marginal=marginal,
                        support_hash=live_support if point["feedback"] == 0 else False)
            require(("flipped" in replicate) == (cue is not None), "missing or unexpected cue-flip probe")
            if cue is not None:
                consume(replicate["flipped"], law, model_hash, cue, branch, flipped=True)

    def consume_valid(value, law, model_hash, branch):
        require(set(value) == {"0", "1"}, "valid-old mode missing")
        for mode in (0, 1):
            target, collected = dict(law, mode=mode, noise=0.), []
            for _ in range(config["support_replicates"]):
                require(cursor < len(rows), "missing valid-old trace")
                metric = dict(rows[cursor]["metrics"], trace=cursor)
                collected.append(metric)
                consume(metric, target, model_hash, None, branch)
            require(set(value[str(mode)]) == {"valid_brier", "valid_survival"}, "valid-old metric fields changed")
            for field in ("valid_brier", "valid_survival"):
                close(value[str(mode)][field], mean(row[field] for row in collected), "valid-old support mean mismatch")

    law, cue, branch = (record[name] for name in ("law", "cue", "branch"))
    before_hash, after_hash = (record[name]["final_hash"] for name in ("start_diagnostics", "diagnostics"))
    first_support, last_support = record["packets"][0]["support_sha256"], record["packets"][-1]["query_sha256"]
    consume_probe(record["before"], law, before_hash, cue, branch, live_support=first_support)
    consume_valid(record["valid_before"], law, before_hash, branch)
    consume(record["curve"][0]["metrics"], law, before_hash, cue, branch, support_hash=first_support)
    for point in record["curve"][1:]:
        end = point["arrivals"]//config["batch_size"]
        a, y = training["actions"][:end], training["outcomes"][:end]
        count = np.array([(a == action).sum() for action in range(5)])
        success = np.array([y[a == action].sum(axis=0) for action in range(5)])
        consume(point["metrics"], law, after_hash if point["arrivals"] == record["size"] else None, cue, branch,
                marginal=(success+.5)/(count[:, None]+1.), support_hash=record["packets"][end-1]["query_sha256"])
    marginal = (np.asarray(record["marginal_success"])+.5)/(np.asarray(record["marginal_count"])[:, None]+1.)
    consume_probe(record["after"], law, after_hash, cue, branch, marginal, last_support)
    consume_valid(record["valid_after"], law, after_hash, branch)
    if record["phase"] == "return":
        novel = next(item for item in study_api().phases(record["seed"], config, "joint") if item["name"] == "novel")
        consume_probe(record["novel_after_return"], novel["law"], after_hash, novel["cue"], "novel", live_support=last_support)
    else:
        require(record["novel_after_return"] is None, "novel return probe outside return phase")
    require(cursor == len(rows), "unreferenced evaluator calls")
    return counts


def check_memory(memory, config):
    ids, seen = memory["ids"], memory["seen"]
    require(memory["type"] == "ReservoirMemory" and memory["capacity"] == config["memory_packets"]
            and memory["age"] == seen, "uniform reservoir policy/capacity/age changed")
    require(type(seen) is int and seen >= 0 and len(ids) == min(seen, config["memory_packets"])
            and len(ids) == len(set(ids)) and all(type(index) is int and 0 <= index < seen for index in ids),
            "invalid uniform reservoir membership")
    require(len(memory["packets"]) == len(ids), "replay packet coverage mismatch")
    for item in memory["packets"]:
        require(item["hidden"] is False, "latent labels entered ordinary memory")
        valid_hash(item["query"])
        valid_hash(item["support"], optional=True)


def verify_work(record, config):
    before, after = (record[name] for name in ("start_diagnostics", "diagnostics"))
    size, updates = record["size"], config["updates_per_batch"]
    packets = size//config["batch_size"]
    expected = dict(arrivals=size, optimizer_steps=packets*updates, query_presentations=2*size*updates,
                    support_presentations=2*size*updates)
    require(all(after["cost"][key]-before["cost"][key] == value for key, value in expected.items()),
            "arrival/update/presentation budget changed")
    require(sum(after["cost"][key]-before["cost"][key] for key in ("replay_presentations", "duplicate_presentations"))
            == size*updates and after["cost"]["training_seconds"] >= before["cost"]["training_seconds"],
            "replay budget or training time mismatch")
    require(all(before[key] == after[key] for key in ("parameters", "model_bytes", "initial_hash", "initial_encoder_hash")),
            "architecture or initial-state identity changed within a phase")
    for diagnostic, memory in ((before, record["start_memory"]), (after, record["end_memory"])):
        check_memory(memory, config)
        require(diagnostic["memory_ids"] == memory["ids"] and diagnostic["packets_seen"] == memory["seen"],
                "memory and diagnostic mismatch")
        for key in ("initial_hash", "final_hash", "initial_encoder_hash", "encoder_hash"):
            valid_hash(diagnostic[key])
    require(record["end_memory"]["seen"]-record["start_memory"]["seen"] == packets, "reservoir lifetime reset")
    if record["arm"] == "prefix":
        require("core_residual" not in before and "core_residual" not in after, "residual inserted during ordinary prefix")
        return
    internal = "joint" if record["arm"] == "fresh" else record["arm"]
    first, last = before["core_residual"], after["core_residual"]
    require(first["arm"] == last["arm"] == internal, "wrong residual trainability rule")
    require(first["parameters"] == last["parameters"] and first["parameters"]["total"] == before["parameters"],
            "residual capacity or trainability changed within a phase")
    parameters = first["parameters"]
    require(parameters["total"] == parameters["core"]+parameters["residual"]
            and parameters["core_trainable"] == (parameters["core"] if internal == "joint" else 0)
            and parameters["residual_encoder_trainable"] == (0 if internal == "fixed_features" else parameters["residual_encoder"])
            and parameters["residual_trainable"] == parameters["residual"]-(parameters["residual_encoder"] if internal == "fixed_features" else 0)
            and parameters["total_trainable"] == parameters["core_trainable"]+parameters["residual_trainable"],
            "declared parameter trainability mismatch")
    for key in ("residual_seed", "core_at_fork_hash", "core_prefix_initial_hash", "residual_initial_hash",
                "residual_initial_encoder_hash", "core_optimizer_at_fork_hash"):
        require(first[key] == last[key], "core/residual initialization provenance changed")
    if internal != "joint":
        require(first["core_hash"] == last["core_hash"] == last["core_at_fork_hash"]
                and first["core_optimizer_hash"] == last["core_optimizer_hash"] == last["core_optimizer_at_fork_hash"]
                and last["core_displacement_from_fork"] == 0., "frozen core weights or Adam state changed")
    if internal == "fixed_features":
        require(first["residual_encoder_hash"] == last["residual_encoder_hash"] == last["residual_initial_encoder_hash"]
                and last["residual_encoder_displacement_from_initial"] == 0., "fixed random visual features changed")
    work = dict(packets=packets, optimizer_steps=packets*updates, backward_calls=packets*updates,
        logical_probability_calls=2*packets*updates, core_forward_calls=2*packets*updates,
        residual_forward_calls=2*packets*updates, core_query_presentations=2*size*updates,
        residual_query_presentations=2*size*updates, core_support_presentations=2*size*updates,
        residual_support_presentations=2*size*updates, core_backward_paths=2*packets*updates*(internal == "joint"),
        residual_backward_paths=2*packets*updates)
    require(set(first["work"]) == set(last["work"]) == set(work)
            and all(last["work"][key]-first["work"][key] == value for key, value in work.items()),
            "core/residual forward/backward work mismatch")


def verify_phase(record, directory, config, identity, seed, model, arm, phase):
    require(record["identity"] == identity and (record["seed"], record["model"], record["arm"], record["phase"])
            == (seed, model, arm, phase["name"]), "phase identity/key mismatch")
    require(all(digest(record[key]) == digest(phase[key]) for key in ("law", "size", "cue", "branch", "channel")),
            "phase schedule/law/cue mismatch")
    for name in ("start_signature", "end_signature"):
        valid_hash(record[name])
    packets, batch_size = phase["size"]//config["batch_size"], config["batch_size"]
    require(record["batches_done"] == packets and len(record["batch_sha256"]) == len(record["packets"]) == packets,
            "incomplete phase training packets")
    require([row["arrivals"] for row in record["curve"]] == list(range(0, phase["size"]+1, config["probe_every"])),
            "incomplete held-out learning curve")
    for index, row in enumerate(record["packets"]):
        valid_hash(row["query_sha256"])
        valid_hash(row["support_sha256"], optional=True)
        require(row["index"] == index and row["id"] == record["start_memory"]["seen"]+index
                and row["query_sha256"] == record["batch_sha256"][index], "packet identity or query provenance mismatch")
        if index:
            require(row["support_sha256"] == record["batch_sha256"][index-1], "noncausal ordinary training support")
        ids = row["memory_ids_before"]
        require(len(ids) == min(row["id"], config["memory_packets"]) and len(ids) == len(set(ids))
                and all(type(i) is int and 0 <= i < row["id"] for i in ids), "invalid pre-update replay membership")
        require(len(row["replay_ids"]) == config["updates_per_batch"] and
                all(i in (ids or [row["id"]]) for i in row["replay_ids"]), "noncausal replay sampling metadata")
    require(record["elapsed_seconds"] >= 0 and record["size"] == packets*batch_size, "invalid phase work duration")
    require(set(record["artifact_hashes"]) == {"before.pt", "evaluations.json", "evaluations.npz", "training.npz"},
            "phase artifact hash coverage mismatch")
    require(set(path.name for path in directory.iterdir()) ==
            {"before.pt", "checkpoint.pt", "result.json", "evaluations.json", "evaluations.npz", "training.npz"},
            "phase artifact file set mismatch")
    for name, value in record["artifact_hashes"].items():
        require(sha(safe_path(directory, name)) == value, "phase artifact hash mismatch")
    verify_work(record, config)
    training = verify_training(record, directory, config)
    return verify_evaluations(record, directory, config, training)


def audit_records(directory):
    """Strict source/JSON/numerical checks; no model pickles or ordinary retraining."""
    directory = Path(directory).resolve()
    study = study_api()
    original = read(directory/"manifest.json")
    manifest = study.check_locks(directory, device=original["runtime"]["device"])
    config, identity = manifest["config"], manifest["identity"]
    study.validate_config(config)
    protocol = read(directory/"protocol_lock.json")
    verify_analysis_lock(directory, config, protocol["protocol_sha256"])
    records, hashes, expected_results = {}, [], set()
    coverage = defaultdict(int)
    jobs = {(seed, model) for seed in config["seeds"] for model in config["models"]}
    for seed, model in sorted(jobs):
        job_dir = directory/f"{model}_{seed}"
        job = read(job_dir/"result.json")
        require(job["identity"] == identity and (job["seed"], job["model"]) == (seed, model)
                and job["affinity"] == manifest["runtime"]["affinity"], "job identity/runtime mismatch")
        scheduled = [("prefix", item, item["name"]) for item in study.prefix_phases(seed, config)]
        scheduled += [(arm, item, f"{arm}/{item['name']}") for arm in (*ARMS, "fresh") for item in study.phases(seed, config, arm)]
        paths = {relative for _, _, relative in scheduled}
        require(set(job["phase_hashes"]) == set(job["checkpoint_hashes"]) == paths
                and set(job["insertion"]) == set(ARMS) and set(job["final_signatures"]) == {*ARMS, "fresh"},
                "job phase/checkpoint/final-state coverage mismatch")
        expected_results.add(job_dir/"result.json")
        hashes.append(dict(path=(job_dir/"result.json").relative_to(directory).as_posix(), sha256=sha(job_dir/"result.json"), kind="job"))
        for arm, phase, relative in scheduled:
            folder = job_dir/relative
            result_path = folder/"result.json"
            require(sha(result_path) == job["phase_hashes"][relative]
                    and sha(folder/"checkpoint.pt") == job["checkpoint_hashes"][relative], "job-bound phase/checkpoint hash mismatch")
            record = read(result_path)
            key = seed, model, arm, phase["name"]
            require(key not in records, "duplicate phase key")
            counts = verify_phase(record, folder, config, identity, seed, model, arm, phase)
            for name, value in counts.items():
                coverage[name] += value
            coverage["training_packets"] += record["batches_done"]
            coverage["training_arrivals"] += record["size"]
            records[key] = record
            expected_results.add(result_path)
            for name in ("result.json", "before.pt", "checkpoint.pt", "evaluations.json", "evaluations.npz", "training.npz"):
                path = folder/name
                hashes.append(dict(path=path.relative_to(directory).as_posix(), sha256=sha(path), kind={
                    "result.json": "phase", "before.pt": "before", "checkpoint.pt": "checkpoint",
                    "evaluations.json": "evaluation", "evaluations.npz": "evaluation_arrays", "training.npz": "training_arrays"}[name]))
        prefix = [records[seed, model, "prefix", phase["name"]] for phase in study.prefix_phases(seed, config)]
        first = prefix[0]
        require(first["start_memory"]["seen"] == 0 and first["start_diagnostics"]["optimizer_bytes"] == 0
                and first["start_diagnostics"]["cost"]["arrivals"] == 0
                and first["start_diagnostics"]["final_hash"] == first["start_diagnostics"]["initial_hash"]
                and first["packets"][0]["support_sha256"] is None, "prefix did not start from scratch")
        for previous, current in zip(prefix, prefix[1:]):
            check_chain(previous, current)
        parent = prefix[-1]
        require(job["parent_signature"] == parent["end_signature"], "shared prefix parent signature mismatch")
        for arm in ARMS:
            chain = [records[seed, model, arm, name] for name in PHASES]
            initial, insertion = chain[0], job["insertion"][arm]
            require(insertion["signature"] == initial["start_signature"] and insertion["diagnostics"] == initial["start_diagnostics"],
                    "insertion record differs from continuous arm start")
            require(initial["start_memory"] == parent["end_memory"]
                    and initial["packets"][0]["support_sha256"] == parent["packets"][-1]["query_sha256"]
                    and initial["start_diagnostics"]["cost"] == parent["diagnostics"]["cost"], "residual insertion reset ordinary state")
            require(initial["start_diagnostics"]["core_residual"]["core_hash"] == parent["diagnostics"]["final_hash"],
                    "residual insertion changed learned core weights")
            for previous, current in zip(chain, chain[1:]):
                check_chain(previous, current)
            require(job["final_signatures"][arm] == chain[-1]["end_signature"], "final continuous learner signature mismatch")
        fresh = records[seed, model, "fresh", "novel"]
        fresh_before = fresh["start_diagnostics"]
        require(fresh_before["optimizer_bytes"] == 0 and fresh_before["cost"]["arrivals"] == 0
                and fresh_before["initial_hash"] == fresh_before["final_hash"] and fresh["start_memory"]["seen"] == 0
                and fresh["packets"][0]["support_sha256"] == records[seed, model, "joint", "novel"]["packets"][0]["support_sha256"]
                and job["final_signatures"]["fresh"] == fresh["end_signature"], "fresh reference is not correctly initialized")
        for phase in PHASES:
            group = [records[seed, model, arm, phase] for arm in ARMS]
            for item in group[1:]:
                require(item["start_memory"] == group[0]["start_memory"] and item["end_memory"] == group[0]["end_memory"],
                        "continuous arm changed uniform memory stream")
                require([packet["replay_ids"] for packet in item["packets"]] ==
                        [packet["replay_ids"] for packet in group[0]["packets"]], "continuous arms used different replay draws")
                require(item["start_diagnostics"]["parameters"] == group[0]["start_diagnostics"]["parameters"],
                        "joint comparator has different total parameter capacity")
        initial = [records[seed, model, arm, "maintenance"] for arm in ARMS]
        require(all(item["before"] == initial[0]["before"] and item["valid_before"] == initial[0]["valid_before"]
                    for item in initial), "initial arm predictions are not matched")
    require(set(directory.rglob("result.json")) == expected_results, "unexpected or missing result records")
    for seed in config["seeds"]:
        grouped = defaultdict(list)
        for key, value in records.items():
            if key[0] == seed:
                grouped[value["phase"]].append(value)
        for group in grouped.values():
            require(all(item["batch_sha256"] == group[0]["batch_sha256"] for item in group),
                    "models/arms received different ordinary observations")
    complete = read(directory/"completion.json")
    expected_counts = study.counts(config)
    require(complete["identity"] == identity and all(complete[name] == value for name, value in expected_counts.items()),
            "completion identity or cohort count mismatch")
    require(len(complete["job_records"]) == len(jobs)
            and {(row["seed"], row["model"]) for row in complete["job_records"]} == jobs
            and all(row["affinity"] == manifest["runtime"]["affinity"] for row in complete["job_records"]),
            "completion worker coverage/runtime mismatch")
    require(len(records) == expected_counts["total_phases"] and coverage["training_packets"] == expected_counts["training_packets"]
            and coverage["training_arrivals"] == expected_counts["training_arrivals"], "audited cohort totals differ")
    return manifest, records, sorted(hashes, key=lambda row: row["path"])


def check_chain(previous, current):
    require(current["start_signature"] == previous["end_signature"]
            and current["start_memory"] == previous["end_memory"]
            and current["start_diagnostics"] == previous["diagnostics"]
            and current["packets"][0]["support_sha256"] == previous["packets"][-1]["query_sha256"],
            "phase boundary reset or replaced a continuing learner")


def resource_row(record, config):
    before, after = record["start_diagnostics"], record["diagnostics"]
    cost = {name: after["cost"][name]-before["cost"][name] for name in after["cost"]}
    residual = after.get("core_residual")
    paths = 2 if residual is not None else 1
    training_calls = 2*record["batches_done"]*config["updates_per_batch"]
    prequential = dict(calls=record["batches_done"], query_presentations=record["size"],
        support_presentations=record["size"], observed_support_presentations=config["batch_size"]*
        sum(row["support_sha256"] is not None for row in record["packets"]))
    evaluation = {key: record["evaluation_files"][key] for key in
                  ("calls", "query_presentations", "support_presentations", "observed_support_presentations")}
    parameters = (dict(residual["parameters"]) if residual is not None else
                  dict(total=after["parameters"], total_trainable=after["parameters"],
                       core=after["parameters"], core_trainable=after["parameters"], residual=0, residual_trainable=0))
    row = {key: record[key] for key in ("seed", "model", "arm", "phase")}
    row.update(parameters=parameters, model_bytes=after["model_bytes"], optimizer_bytes=after["optimizer_bytes"],
        core_optimizer_bytes=after["optimizer_bytes"] if residual is None else residual["core_optimizer_bytes"],
        residual_optimizer_bytes=0 if residual is None else residual["residual_optimizer_bytes"],
        measurement_snapshot_bytes=None if residual is None else residual["measurement_snapshot_bytes"],
        peak_replay_bytes=after["peak_replay_bytes"], history_bytes=after["history_bytes"],
        training_cost=cost, training_logical_probability_calls=training_calls,
        prequential=prequential, evaluation=evaluation, raw_pathways_per_logical_call=paths,
        total_raw_path_forward_calls=paths*(training_calls+prequential["calls"]+evaluation["calls"]),
        total_raw_path_query_presentations=paths*(cost["query_presentations"]+prequential["query_presentations"]+evaluation["query_presentations"]),
        total_raw_path_support_presentations=paths*(cost["support_presentations"]+prequential["support_presentations"]+evaluation["support_presentations"]))
    if residual is not None:
        first = before["core_residual"]
        row["path_work"] = {key: residual["work"][key]-first["work"][key] for key in residual["work"]}
        row["cumulative_path_work"] = dict(residual["work"])
        row["component_displacement"] = {key: value for key, value in residual.items() if "displacement" in key}
    return row


def interval(metric, scale=100.):
    return f"{scale*metric['mean']:+.4f} [{scale*metric['lower']:+.4f}, {scale*metric['upper']:+.4f}]"


def markdown(result):
    lines = ["# Learned core and adaptive raw-input residual", "",
        "Prospective fixed development comparison. Brier differences below are raw Brier units times 100; "
        "survival differences are percentage points. Negative Brier favors the first arm.", "",
        result["inference"], "", "## Fixed primary outcome and feature attribution", "",
        "| Architecture | Novel separate−joint [95% interval] | Improved seeds | Primary | Feature attribution |",
        "|---|---:|---:|---|---|"]
    for model in result["config"]["models"]:
        value = result["comparisons"][model]["separate_minus_joint"]["novel"]["brier_auc"]
        lines.append(f"| {model} | {interval(value)} | {value['negative']}/{value['n']} | "
                     f"{'Pass' if result['screen'][model]['passed'] else 'Fail'} | "
                     f"{'Pass' if result['attribution'][model]['passed'] else 'Fail'} |")
    lines += ["", "Only separate versus joint is the primary outcome comparison. The attribution comparison "
        "uses fixed random residual visual features; it cannot rescue failure or promote an ablation.", ""]
    for model in result["config"]["models"]:
        lines += [f"### {model}: declared gates", "", "| Gate | Value | Decision |", "|---|---:|---|"]
        for name, gate in result["screen"][model]["gates"].items():
            value = gate.get("value", gate.get("improved_seeds", "—"))
            lines.append(f"| {name} | {value} | {'Pass' if gate['passed'] else 'Fail'} |")
        lines += ["", "| Comparison / phase | Affected Brier AUC | All-case Brier AUC | Survival AUC |",
                  "|---|---:|---:|---:|"]
        for name, phases in result["comparisons"][model].items():
            for phase, values in phases.items():
                lines.append(f"| {name} / {phase} | {interval(values['brier_auc'])} | "
                             f"{interval(values['all_brier_auc'])} | {interval(values['survival_auc'])} |")
        values = result["comparisons"][model]["separate_minus_joint"]
        lines += ["", "| Retention contrast: separate−joint | Brier x100 [95% interval] |", "|---|---:|"]
        for label, phase, field in (("Novel end: valid old mode 0", "novel", "valid_after_mode_0"),
            ("Novel end: valid old mode 1", "novel", "valid_after_mode_1"),
            ("Return end: novel law, correct support", "return", "novel_after_return_brier")):
            lines.append(f"| {label} | {interval(values[phase][field])} |")
        lines += ["", "| Phase / arm | Held-out affected AUC x100 | Pre-update performed Brier x100 | Cue benefit x100 |",
                  "|---|---:|---:|---:|"]
        for phase, arms in result["levels"][model].items():
            for arm, values in arms.items():
                cue = f"{100*values['cue_effect']['mean']:.4f}" if "cue_effect" in values else "—"
                lines.append(f"| {phase} / {arm} | {100*values['brier_auc']['mean']:.4f} | "
                             f"{100*values['prequential_brier']['mean']:.4f} | {cue} |")
        lines += ["", "| Seed | Novel separate−joint Brier AUC x100 |", "|---|---:|"]
        metric = result["comparisons"][model]["separate_minus_joint"]["novel"]["brier_auc"]
        for seed, value in zip(metric["seeds"], metric["differences"]):
            lines.append(f"| {seed} | {100*value:+.4f} |")
        lines.append("")
    lines += ["## Fresh qualification", "", "| Architecture / group | Seeds | Marginal gain | Correct-cue benefit | Pass |",
              "|---|---:|---:|---:|---|"]
    for row in result["qualification"]["groups"]:
        lines.append(f"| {row['model']} / {row['group']} | {row['n']} | {row['marginal_gain']:.6f} | "
                     f"{row['cue_effect']:.6f} | {row['passed']} |")
    lines += ["", "## Resource and verification scope", "",
        "resource_rows includes total/trainable parameters, active and retained Adam storage, measurement "
        "snapshots, replay/history, training work, prequential predictions, and every evaluator call. "
        "A composite logical call executes two raw pathways; prefix calls execute one. Freezing changes "
        "backward work. Equal capacity and update counts are not equal FLOPs or measured peak memory.", "",
        "The reader checks locks, cohort and state-chain metadata, raw prediction arithmetic, original "
        "record/artifact hashes and causal-support references. It does not load model pickles or independently "
        "retrain ordinary updates. A separate checkpoint/physics audit has a separately stated scope.", "",
        "Every arm, all six seeds, cue slices, absolute outcomes and paired intervals are preserved in summary.json. "
        "A failed qualification or guard stays failed; there is no outcome-based exposure change or subgroup promotion.", ""]
    return "\n".join(lines)


def summarize(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(directory != output and not output.is_relative_to(directory), "summary export must be outside input runs")
    manifest, records, hashes = audit_records(directory)
    output.mkdir(parents=True, exist_ok=True)
    result = analyze_records(records, manifest["config"])
    result.update(identity=manifest["identity"], counts=study_api().counts(manifest["config"]), result_hashes=hashes,
                  array_archive_prefix="arrays")
    result["resource_rows"] = [resource_row(record, manifest["config"]) for _, record in sorted(records.items())]
    result["record_integrity"] = dict(phases=len(records), training_packets=sum(row["batches_done"] for row in records.values()),
        evaluation_calls=sum(row["evaluation_files"]["calls"] for row in records.values()),
        evaluation_query_presentations=sum(row["evaluation_files"]["query_presentations"] for row in records.values()),
        evaluation_support_presentations=sum(row["evaluation_files"]["support_presentations"] for row in records.values()),
        raw_numerical_reconstruction_passed=True,
        scope="Source/analysis/config/protocol locks, complete phase/hash/state-chain metadata and raw numerical arithmetic; no pickle loading or ordinary retraining.")
    for name in ("manifest.json", "completion.json", "training_source.zip", "analysis_lock.json", "analysis_at_lock.zip",
                 "protocol_lock.json", "protocol_at_lock.md"):
        shutil.copyfile(directory/name, output/name)
    write_json(output/"config.json", manifest["config"])
    with gzip.open(output/"raw_results.jsonl.gz", "wt", encoding="utf-8") as handle:
        for _, record in sorted(records.items()):
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False)+"\n")
    jobs = {item["path"]: read(directory/item["path"]) for item in hashes if item["kind"] == "job"}
    write_json(output/"job_records.json", jobs)
    with gzip.open(output/"evaluation_records.jsonl.gz", "wt", encoding="utf-8") as handle:
        for item in hashes:
            if item["kind"] == "evaluation":
                handle.write(json.dumps(dict(path=item["path"], record=read(directory/item["path"])),
                                       separators=(",", ":"), allow_nan=False)+"\n")
            elif item["kind"] in ("evaluation_arrays", "training_arrays"):
                target = output/"arrays"/item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(directory/item["path"], target)
                require(sha(target) == item["sha256"], "exported numerical bytes changed")
    write_json(output/"summary.json", result)
    (output/"summary.md").write_text(markdown(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = summarize(args.input, args.output)
    print(json.dumps(dict(counts=summary["counts"], primary={model: value["passed"] for model, value in summary["screen"].items()},
                          attribution={model: value["passed"] for model, value in summary["attribution"].items()}), indent=2))
