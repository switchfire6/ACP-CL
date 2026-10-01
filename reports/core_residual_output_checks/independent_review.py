"""Independent NumPy review of saved-model component outputs; no model imports.

This reads immutable JSON/NPZ/checkpoint bytes, never unpickles a checkpoint,
reruns a predictor, trains a model, or imports the experiment's analysis code.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np


SEEDS = tuple(range(16001, 16007))
MODELS = ("conditional", "recurrent")
ARMS = ("joint", "separate")
TARGETS = ("novel_end", "return_entry", "return_end", "novel_after_return")
METRICS = ("brier", "focus_brier", "survival", "focus_survival", "valid_brier",
           "valid_survival", "no_transfer", "clairvoyant_upper")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, "duplicate JSON key")
            result[name] = value
        return result

    def reject(value):
        raise ValueError("nonfinite JSON " + value)

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=reject)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def array_hash(array):
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode()
                          + array.tobytes(order="C")).hexdigest()


def support_hash(values):
    result = hashlib.sha256()
    for name in ("support_observations", "support_actions", "support_outcomes"):
        array = values[name]
        result.update(str((array.shape, array.dtype.str)).encode())
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def score(prediction, truth, affected, valid):
    require(prediction.dtype == np.float32 and truth.dtype == np.uint8
            and prediction.shape == truth.shape == (512, 5, 3), "raw forecast dimensions/dtypes")
    epsilon = np.finfo(np.float32).eps
    require(np.isfinite(prediction).all() and prediction.min() >= -epsilon
            and prediction.max() <= 1 + epsilon and np.all(np.diff(prediction, axis=2) <= epsilon),
            "invalid forecast probability")
    require(np.isin(truth, (0, 1)).all() and np.all(np.diff(truth.astype(int), axis=2) <= 0),
            "invalid physical survival outcomes")
    for mask in (affected, valid):
        require(mask.shape == (512,) and mask.dtype == np.bool_ and mask.any(), "invalid mask")
    errors = np.square(prediction.astype(np.float64) - truth).reshape(512, 15).sum(axis=1) / 15
    choices = np.argmax(prediction[:, :, 2], axis=1)
    terminal = truth[:, :, 2]
    survival = np.take_along_axis(terminal, choices[:, None], axis=1)[:, 0]
    result = {}
    for suffix, mask in (("", np.ones(512, dtype=bool)), ("focus_", affected), ("valid_", valid)):
        result[suffix + "brier"] = float(np.sum(errors[mask]) / np.count_nonzero(mask))
        result[suffix + "survival"] = float(np.sum(survival[mask]) / np.count_nonzero(mask))
    result["no_transfer"] = float(np.sum(terminal[:, 0]) / 512)
    result["clairvoyant_upper"] = float(np.sum(np.max(terminal, axis=1)) / 512)
    return result


class Comparison:
    def __init__(self):
        self.scalars = 0
        self.maximum_error = 0.

    def equal(self, actual, expected, label):
        if isinstance(actual, dict):
            require(set(actual) == set(expected), label + " keys")
            for name in actual:
                self.equal(actual[name], expected[name], label + "." + name)
        elif isinstance(actual, (tuple, list)):
            require(len(actual) == len(expected), label + " length")
            for index, (a, b) in enumerate(zip(actual, expected)):
                self.equal(a, b, f"{label}[{index}]")
        elif type(actual) is float:
            delta = abs(actual - expected)
            self.maximum_error = max(self.maximum_error, delta)
            self.scalars += 1
            require(delta <= 1e-12, label + " numerical mismatch")
        else:
            require(actual == expected, label + " mismatch")


def stats(values, draws):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (6,) and np.isfinite(values).all(), "six finite independent seed values required")
    resampled = values[draws].sum(axis=1) / 6
    lower, upper = np.percentile(resampled, [2.5, 97.5])
    return dict(mean=float(np.sum(values) / 6), lower=float(lower), upper=float(upper),
                n=6, seeds=list(SEEDS), values=values.tolist(),
                positive=int(np.count_nonzero(values > 0)), negative=int(np.count_nonzero(values < 0)))


def run(directory):
    started = time.perf_counter()
    directory = Path(directory).resolve()
    summary = read(directory / "summary.json")
    manifest = read(directory / "manifest.json")
    completion = read(directory / "completion.json")
    specification = manifest["specification"]
    require(specification["seeds"] == list(SEEDS) and specification["models"] == list(MODELS)
            and specification["arms"] == list(ARMS) and specification["support_replicates"] == 2,
            "development specification coverage")
    source = directory / "source_at_lock.zip"
    require(sha(source) == manifest["source_archive_sha256"], "locked source ZIP digest")
    with zipfile.ZipFile(source) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist()))
                and set(archive.namelist()) == set(specification["source_files"]), "source ZIP coverage")
        for name, expected in specification["source_files"].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, "locked source member digest")
    base = Path(specification["base"])
    historical_hashes, historical_traces = {}, {}

    def historical(name, expected):
        if name not in historical_hashes:
            historical_hashes[name] = sha(base / name)
        require(historical_hashes[name] == expected, "historical file changed: " + name)

    for name, expected in specification["base_files"].items():
        historical(name, expected)
    require(read(base / "manifest.json")["identity"] == specification["identity"], "historical study identity")
    completed = {(row["model"], row["seed"]): row for row in completion["jobs"]}
    expected_jobs = {(model, seed) for model in MODELS for seed in SEEDS}
    require(len(completion["jobs"]) == 12 and set(completed) == expected_jobs
            and completion["rows"] == 384 and completion["checkpoints"] == 48
            and completion["optimizer_updates"] == 0, "completion coverage")
    expected_rows = {(arm, target, context, replicate, flipped) for arm in ARMS for target in TARGETS
        for context in ("actual", "refreshed") for replicate in ((0,) if context == "actual" else (0, 1))
        for flipped in ((False, True) if context == "refreshed" and target in ("novel_end", "novel_after_return") else (False,))}
    input_hashes = {name: sha(directory / name) for name in
                    ("summary.json", "manifest.json", "completion.json", "source_at_lock.zip")}
    comparison, recomputed, raw_metric_count = Comparison(), [], 0
    frozen_pairs, cross_checkpoint_frozen_pairs, full_matches = 0, 0, 0
    paired_inputs, grouped_queries = {}, {}
    state_count = 0
    per_job = []
    for model, seed in sorted(expected_jobs):
        filename = f"jobs/{model}_{seed}.json"
        job = read(directory / filename)
        job_hash = sha(directory / filename)
        input_hashes[filename] = job_hash
        require(job_hash == completed[model, seed]["json_sha256"], "completed job digest")
        require(job["model"] == model and job["seed"] == seed and job["support_replicates"] == 2,
                "job metadata")
        row_keys = [(r["arm"], r["target"], r["context"], r["replicate"], r["flipped"]) for r in job["rows"]]
        require(len(row_keys) == 32 and set(row_keys) == expected_rows, "intervention coverage")
        checkpoint_map = {entry["path"]: entry for entry in job["checkpoints"]}
        require(len(job["checkpoints"]) == 4 and set(checkpoint_map) == {
            f"{model}_{seed}/{arm}/{phase}/checkpoint.pt" for arm in ARMS for phase in ("novel", "return")},
            "saved-state coverage")
        for entry in checkpoint_map.values():
            historical(entry["path"], entry["sha256"])
            require(entry["signature_before"] == entry["signature_after"]
                    and entry["rng_before"] == entry["rng_after"], "recorded state/RNG preservation")
            state_count += 1
        arrays_name = f"jobs/{model}_{seed}.npz"
        input_hashes[arrays_name] = sha(directory / arrays_name)
        require(job["arrays"] == {"file": f"{model}_{seed}.npz", "sha256": input_hashes[arrays_name]},
                "raw archive digest")
        frozen, job_pairs, job_cross_pairs = {}, 0, 0
        cells = ("full", "core", "residual", "full_core_route", "core_full_route") if model == "conditional" else (
            "full", "core", "residual")
        with np.load(directory / arrays_name, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "duplicate array name")
            used = set()
            for index, row in enumerate(job["rows"]):
                require(row["index"] == index and row["model"] == model and row["seed"] == seed,
                        "row identity")
                fields = {"observations", "truth", "affected", "valid", "archived_full", "support_observations",
                          "support_actions", "support_outcomes"} | {"probabilities_" + cell for cell in cells}
                require(set(row["array_sha256"]) == fields and set(row["metrics"]) == set(cells), "raw field coverage")
                values = {}
                for field in fields:
                    name = f"c{index}_{field}"
                    require(name in archive, "missing raw array")
                    values[field] = archive[name]
                    require(array_hash(values[field]) == row["array_sha256"][field], "raw array digest")
                    used.add(name)
                original, full = values["archived_full"], values["probabilities_full"]
                require(original.dtype == full.dtype and original.shape == full.shape
                        and original.tobytes() == full.tobytes(), "historical full output byte mismatch")
                full_matches += 1
                source_row = row["source"]
                for key, hash_key in (("checkpoint", "checkpoint_sha256"), ("trace_json", "trace_json_sha256"),
                                      ("trace_arrays", "trace_arrays_sha256")):
                    historical(source_row[key], source_row[hash_key])
                if source_row["trace_json"] not in historical_traces:
                    historical_traces[source_row["trace_json"]] = read(base / source_row["trace_json"])
                trace = historical_traces[source_row["trace_json"]]["records"][source_row["trace"]]
                require(trace["index"] == source_row["trace"] and trace["flipped"] == row["flipped"],
                        "historical trace identity")
                for key in ("query_sha256", "support_fingerprint", "model_sha256"):
                    require(trace[key] == source_row[key], "historical trace provenance")
                with np.load(base / source_row["trace_arrays"], allow_pickle=False) as historical_arrays:
                    for field in fields - {"probabilities_" + cell for cell in cells}:
                        old_field = "probabilities" if field == "archived_full" else field
                        old_array = historical_arrays[f'e{source_row["trace"]}_{old_field}']
                        require(array_hash(old_array) == trace["array_sha256"][old_field],
                                "historical trace array digest")
                        require(old_array.dtype == values[field].dtype and old_array.shape == values[field].shape
                                and old_array.tobytes() == values[field].tobytes(),
                                "diagnostic array does not reproduce the historical trace")
                state = checkpoint_map[source_row["checkpoint"]]
                require(source_row["checkpoint_sha256"] == state["sha256"]
                        and row["signature"] == state["signature_before"], "row/checkpoint binding")
                query_hash, context_hash = array_hash(values["observations"]), support_hash(values)
                require(source_row["query_sha256"] == query_hash
                        and source_row["support_fingerprint"] == context_hash, "query/support binding")
                for field in ("observations", "truth", "affected", "valid", "support_observations", "support_actions", "support_outcomes"):
                    flipped = row["flipped"] if field == "observations" else False
                    key = seed, row["target"], row["context"], row["replicate"], flipped, field
                    current_hash = array_hash(values[field])
                    require(key not in paired_inputs or paired_inputs[key] == current_hash, "unpaired component/context data")
                    paired_inputs[key] = current_hash
                    if not field.startswith("support_"):
                        key = seed, row["target"], flipped, field
                        require(key not in grouped_queries or grouped_queries[key] == current_hash,
                                "context intervention changed physical query")
                        grouped_queries[key] = current_hash
                if row["arm"] == "separate":
                    key = query_hash, context_hash
                    forecast_hash = array_hash(values["probabilities_core"])
                    if key in frozen:
                        require(frozen[key][0] == forecast_hash, "frozen core output changed at fixed inputs")
                        job_pairs += 1
                        job_cross_pairs += frozen[key][1] != source_row["checkpoint"]
                    frozen[key] = forecast_hash, source_row["checkpoint"]
                metrics = {}
                for cell in cells:
                    metrics[cell] = score(values["probabilities_" + cell], values["truth"], values["affected"], values["valid"])
                    comparison.equal(metrics[cell], row["metrics"][cell], "raw component metrics")
                    raw_metric_count += 8
                recomputed.append({**{key: row[key] for key in ("model", "seed", "arm", "target", "context", "replicate", "flipped")},
                                   "metrics": metrics})
            require(used == set(archive.files), "unused numerical arrays")
        require(job_pairs == job["frozen_core_forecast_pairs_checked"] == completed[model, seed]["frozen_pairs"] == 8,
                "frozen-input check coverage")
        frozen_pairs += job_pairs
        cross_checkpoint_frozen_pairs += job_cross_pairs
        per_job.append(dict(model=model, seed=seed, rows=32, checkpoint_states=4,
                            frozen_pairs=job_pairs, cross_checkpoint_frozen_pairs=job_cross_pairs))
    require(raw_metric_count == 12288 and frozen_pairs == completion["frozen_core_forecast_pairs_checked"] == 96,
            "complete numerical coverage")

    groups = defaultdict(list)
    for row in recomputed:
        key = tuple(row[k] for k in ("model", "seed", "arm", "target", "context", "flipped"))
        groups[key].append(row)
    per_seed = {}
    for key, rows in groups.items():
        require(sorted(row["replicate"] for row in rows) == ([0] if key[4] == "actual" else [0, 1]),
                "within-seed replicate coverage")
        per_seed[key] = {cell: {metric: float(sum(row["metrics"][cell][metric] for row in rows) / len(rows))
                               for metric in METRICS} for cell in rows[0]["metrics"]}
    require(len(per_seed) == len(summary["per_seed"]) == 240, "per-seed coverage")
    for row in summary["per_seed"]:
        key = tuple(row[k] for k in ("model", "seed", "arm", "target", "context", "flipped"))
        comparison.equal(per_seed[key], row["metrics"], "within-seed averaging")
    draws = np.random.default_rng(17291).integers(0, 6, size=(20000, 6))
    require(summary["bootstrap"]["seed"] == 17291 and summary["bootstrap"]["resamples"] == 20000
            and summary["development_allocation_eligible"], "bootstrap or eligibility metadata")
    decisions = {}
    for model in MODELS:
        def vector(arm, target, context, cell, metric="brier"):
            return np.array([per_seed[model, seed, arm, target, context, False][cell][metric] for seed in SEEDS])

        old_harm = vector("separate", "return_entry", "refreshed", "full") - vector("separate", "return_entry", "refreshed", "core")
        new_gain = vector("separate", "novel_end", "refreshed", "core", "focus_brier") - vector("separate", "novel_end", "refreshed", "full", "focus_brier")
        actual_gap = vector("separate", "return_entry", "actual", "full") - vector("joint", "return_entry", "actual", "full")
        context_gain = vector("separate", "return_entry", "actual", "full") - vector("separate", "return_entry", "refreshed", "full")
        refreshed_gap = vector("separate", "return_entry", "refreshed", "full") - vector("joint", "return_entry", "refreshed", "full")
        metrics = {name: stats(values, draws) for name, values in (
            ("old_harm", old_harm), ("novel_benefit", new_gain), ("actual_separate_minus_joint", actual_gap),
            ("refreshed_support_gain", context_gain), ("refreshed_separate_minus_joint", refreshed_gap))}
        split = dict(old_harm_mean=bool(old_harm.mean() >= .005), old_harm_consistency=bool((old_harm > 0).sum() >= 5),
                     novel_benefit_mean=bool(new_gain.mean() >= .002), novel_benefit_consistency=bool((new_gain > 0).sum() >= 5))
        context = dict(actual_return_disadvantage=bool(actual_gap.mean() > 0), refresh_gain_mean=bool(context_gain.mean() >= .005),
                       refresh_gain_consistency=bool((context_gain >= .005).sum() >= 5), refreshed_gap_tolerance=bool(refreshed_gap.mean() <= .005))
        allocation = "GO" if all(split.values()) else "CONTEXT_PIVOT" if all(context.values()) else "STOP"
        result = dict(metrics=metrics, split_gates=split, context_gates=context, allocation=allocation,
                      context_evaluated_for_allocation=not all(split.values()),
                      seeds_with_refresh_gain_at_least_005=int((context_gain >= .005).sum()))
        recorded = summary["decisions"][model]
        for name, value in result.items():
            comparison.equal(value, recorded[name], "allocation." + name)
        decisions[model] = result

    decomposition, largest_closure = [], 0.
    coefficients = {"content_at_core_route": {"full_core_route": 1, "core": -1},
                    "route_on_core_content": {"core_full_route": 1, "core": -1},
                    "interaction": {"full": 1, "full_core_route": -1, "core_full_route": -1, "core": 1},
                    "total_full_minus_core": {"full": 1, "core": -1}}
    for recorded in summary["conditional_decomposition"]:
        terms = {}
        for term, weights in coefficients.items():
            terms[term] = {}
            for metric in METRICS:
                values = [sum(weight * per_seed["conditional", seed, recorded["arm"], recorded["target"],
                    recorded["context"], recorded["flipped"]][cell][metric] for cell, weight in weights.items()) for seed in SEEDS]
                terms[term][metric] = stats(values, draws)
                comparison.equal(terms[term][metric], recorded["terms"][term][metric], "conditional decomposition")
        for metric in METRICS:
            difference = sum(np.array(terms[name][metric]["values"]) for name in
                             ("content_at_core_route", "route_on_core_content", "interaction")) - terms["total_full_minus_core"][metric]["values"]
            largest_closure = max(largest_closure, float(np.abs(difference).max()))
        decomposition.append({**{key: recorded[key] for key in ("arm", "target", "context", "flipped")}, "terms": terms})
    require(len(decomposition) == 20 and largest_closure < 1e-12, "conditional decomposition coverage/closure")
    # These are useful claim checks, not additional decision rules.
    require(all(decisions[m]["metrics"]["actual_separate_minus_joint"]["negative"] == 6 for m in MODELS),
            "claimed six-seed favorable actual return-entry comparison differs")
    return dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        input_directory=directory.as_posix(), reviewer_source_sha256=sha(__file__), input_hashes=input_hashes,
        historical_files_checked=len(historical_hashes), historical_hashes=historical_hashes,
        coverage=dict(jobs=12, saved_states=state_count, rows=len(recomputed), raw_metric_scalars=raw_metric_count,
            per_seed_groups=len(per_seed), original_full_byte_matches=full_matches,
            historical_trace_fields_byte_matched=8 * len(recomputed), frozen_input_pairs=frozen_pairs,
            cross_checkpoint_frozen_input_pairs=cross_checkpoint_frozen_pairs, allocation_statistics=10,
            conditional_decomposition_statistics=20 * 4 * 8),
        numeric_comparisons=comparison.scalars, maximum_absolute_discrepancy=comparison.maximum_error,
        decomposition_maximum_closure_error=largest_closure, decisions=decisions,
        conditional_return_entry_separate=[row for row in decomposition if row["arm"] == "separate"
                                          and row["target"] == "return_entry" and not row["flipped"]],
        job_checks=per_job, elapsed_seconds=time.perf_counter() - started,
        claim_limits="GO is a retrospective effort-allocation heuristic, not a new acquisition/retention pass or deployable policy. "
            "Actual return-entry separate-minus-joint Brier is favorable in all six seeds of both architectures; "
            "this does not reverse the earlier return-trajectory AUC failure. Refreshed contexts are evaluator supplied. "
            "Crossover decomposition includes interaction and does not identify a unique learning cause. "
            "Frozen-input invariance here checks raw core forecast hashes; unchanged checkpoint bytes and reported "
            "state/RNG signatures are bound, but this NumPy script does not inspect tensor contents, reconstruct forecasts, "
            "rerun physics, unpickle models, retrain, or prove historical wall-clock file creation order.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("passed", "coverage", "maximum_absolute_discrepancy", "elapsed_seconds")}, indent=2))
