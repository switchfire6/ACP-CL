"""Verify portable selective-update archives and recompute critical results.

Only standard Python and NumPy are required. This never imports training or
analysis modules, loads model pickles, or consults the live source repository.
It checks archived evidence; it is not an independent scientific replication.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import zipfile

import numpy as np


ARMS = ("reference", "current", "protected", "combined", "shrink", "fresh")
FIELDS = ("brier_auc", "valid_after", "valid_after_mode_0", "valid_after_mode_1", "survival_auc")
ORDERS = tuple(permutations(range(3)))


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
        require(math.isfinite(value), "nonfinite portable record")


def decode(contents):
    result = json.loads(contents)
    finite_tree(result)
    return result


def read(path):
    return decode(Path(path).read_text(encoding="utf-8"))


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def original_json(record):
    return (json.dumps(record, indent=2, allow_nan=False)+"\n").encode("utf-8")


def compare(actual, expected, location="value"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"different keys: {location}")
        for key, value in expected.items():
            compare(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"different list length: {location}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{location}[{index}]")
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), f"numeric mismatch: {location}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"value mismatch: {location}")


def verify_artifacts(directory):
    directory = Path(directory).resolve()
    expected = read(directory/"artifact_manifest.json")
    present = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
               if path.is_file() and path != directory/"artifact_manifest.json"}
    require(set(expected) == present, "portable artifact file set changed")
    for name, entry in expected.items():
        path = (directory/name).resolve()
        require(path.is_relative_to(directory), "artifact path escapes archive")
        contents = path.read_bytes()
        require(entry == dict(bytes=len(contents), sha256=sha(contents)), f"artifact bytes/hash changed: {name}")
    return len(expected)


def verify_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected)
                and archive.testzip() is None, "archive ZIP file set/CRC mismatch")
        for name, expected_hash in expected.items():
            require(sha(archive.read(name)) == expected_hash, f"archive ZIP hash mismatch: {name}")


def record_paths(config):
    episodes = {f"{model}_{seed}/stage_1_{arm}/result.json"
                for seed in config["seeds"] for model in config["models"] for arm in ARMS}
    prefixes = {f"{model}_{seed}/prefix.json" for seed in config["seeds"] for model in config["models"]}
    return episodes, prefixes


def read_records(folder, config, identity, summary):
    episode_paths, prefix_paths = record_paths(config)
    expected = episode_paths | prefix_paths
    listed = {entry["path"]: entry["sha256"] for entry in summary["result_hashes"]}
    require(len(listed) == len(summary["result_hashes"]) and set(listed) == expected,
            "incomplete or duplicated original record hashes")
    records, reconstructed = {}, {}
    with gzip.open(folder/"raw_results.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            record = decode(line)
            seed, model, arm = record["seed"], record["model"], record["arm"]
            path = f"{model}_{seed}/stage_1_{arm}/result.json"
            require(path in episode_paths and path not in reconstructed and record["identity"] == identity
                    and record["stage"] == 1 and record["branch"] == "novel", "unexpected/duplicate raw episode")
            cue = ORDERS[seed % 6][0]
            require(record["cue"] == cue and record["channel"] == "stage_1"
                    and record["law"] == dict(mode=1-seed % 2, active=[cue], revised=False, noise=0.),
                    "raw episode law/cue/channel mismatch")
            require(record["batches_done"] == config["episode_size"]//config["batch_size"]
                    and len(record["batch_sha256"]) == record["batches_done"], "incomplete raw episode")
            require([point["arrivals"] for point in record["curve"]]
                    == list(range(0, config["episode_size"]+1, config["probe_every"])), "incomplete raw acquisition curve")
            records[seed, model, arm] = record
            reconstructed[path] = sha(original_json(record))
    prefixes = read(folder/"prefix_records.json")
    for tag, record in prefixes.items():
        path = f"{tag}/prefix.json"
        require(path in prefix_paths and path not in reconstructed and record["identity"] == identity
                and tag == f"{record['model']}_{record['seed']}" and record["policy"] == "uniform"
                and len(record["blocks"]) == config["prefix_blocks"], "prefix record coverage/identity mismatch")
        reconstructed[path] = sha(original_json(record))
    require(reconstructed == listed, "portable raw records differ from original JSON hashes")
    return records, prefixes, reconstructed


def mean(values):
    return float(np.mean(list(values)))


def metrics(record):
    """Independent computation from recorded curves and correctly supported probes."""
    x = np.array([point["arrivals"] for point in record["curve"]], dtype=float)
    require(x[0] == 0 and x[-1] > 0 and np.all(np.diff(x) > 0), "invalid integration grid")

    def area(field):
        y = np.array([point["metrics"][field] for point in record["curve"]], dtype=float)
        return float(np.sum(np.diff(x)*(y[:-1]+y[1:])/2)/x[-1])

    def support(field, flipped=False):
        return mean(row["flipped"][field] if flipped else row["curve"][-1]["metrics"][field]
                    for row in record["after"]["replicates"])

    valid = [record["valid_after"][str(mode)]["valid_brier"] for mode in (0, 1)]
    return dict(brier_auc=area("focus_brier"), survival_auc=area("survival"),
        valid_after=mean(valid), valid_after_mode_0=valid[0], valid_after_mode_1=valid[1],
        marginal_gain=support("marginal_brier")-support("brier"),
        cue_effect=support("focus_brier", True)-support("focus_brier"))


def bootstrap(values, seeds):
    values = np.asarray(values, dtype=float)
    require(len(values) == len(seeds) > 0 and len(set(seeds)) == len(seeds), "invalid independent seed values")
    values = np.where(np.abs(values) < 1e-12, 0., values)
    rng = np.random.default_rng(27192026)
    boot = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), lower=float(np.quantile(boot, .025)), upper=float(np.quantile(boot, .975)),
                positive=int(np.count_nonzero(values > 0)), n=len(values), differences=values.tolist(), seeds=list(seeds))


def qualification(raw, records, config):
    groups, by_model = {}, {}
    for model in config["models"]:
        for dimension in ("stage", "cue"):
            labels = {seed: records[seed, model, "fresh"][dimension] for seed in config["seeds"]}
            for label in sorted(set(labels.values())):
                seeds = [seed for seed, value in labels.items() if value == label]
                marginal = mean(raw[seed, model, "fresh"]["marginal_gain"] for seed in seeds)
                cue = mean(raw[seed, model, "fresh"]["cue_effect"] for seed in seeds)
                groups[f"{model}_{dimension}_{label}"] = dict(marginal_gain=marginal, cue_effect=cue,
                    acquisition_ok=marginal >= .02, cue_use_ok=cue >= .002, n=len(seeds))
        by_model[model] = all(row["acquisition_ok"] and row["cue_use_ok"] for name, row in groups.items()
                             if name.startswith(model+"_"))
    return dict(passed=all(by_model.values()), by_model=by_model, marginal_threshold=.02, cue_threshold=.002, groups=groups)


def verify_screen(stored, comparisons, qualified, eligible, attribution=False):
    """Check every declared mean-based gate and its descriptive interval warning."""
    expected = {"complete_development_cohort": dict(passed=bool(eligible)),
                "fresh_qualification": dict(passed=bool(qualified))}

    def maximum(name, metric, limit):
        expected[name] = dict(passed=metric["mean"] <= limit, value=metric["mean"], maximum=limit,
            lower=metric["lower"], upper=metric["upper"], uncertainty_crosses_limit=metric["upper"] > limit)

    if attribution:
        for control in ("current", "shrink"):
            maximum(f"retention_vs_{control}", comparisons[control]["valid_after"], -.002)
            maximum(f"acquisition_cost_vs_{control}", comparisons[control]["brier_auc"], .001)
    else:
        candidate = comparisons["reference"]
        maximum("acquisition_gain", candidate["brier_auc"], -.002)
        improved = sum(value < 0 for value in candidate["brier_auc"]["differences"])
        expected["acquisition_consistency"] = dict(passed=improved >= 5, improved_seeds=improved,
                                                   minimum=5, n=candidate["brier_auc"]["n"])
        for mode in (0, 1):
            maximum(f"valid_old_mode_{mode}", candidate[f"valid_after_mode_{mode}"], .005)
        survival = candidate["survival_auc"]
        expected["survival"] = dict(passed=survival["mean"] >= -.01, value=survival["mean"], minimum=-.01,
            lower=survival["lower"], upper=survival["upper"], uncertainty_crosses_limit=survival["lower"] < -.01)
    compare(stored["gates"], expected, "allocation gates")
    compare(stored["passed"], all(row["passed"] for row in expected.values()), "allocation decision")
    compare(stored["failed"], [name for name, row in expected.items() if not row["passed"]], "failed criteria")


def verify_results(records, config, summary):
    raw = {key: metrics(record) for key, record in records.items()}
    rows = {(row["seed"], row["model"], row["arm"]): row for row in summary["rows"]}
    require(len(rows) == len(summary["rows"]) and set(rows) == set(raw), "summary metric-row coverage mismatch")
    for key, values in raw.items():
        for field, value in values.items():
            compare(rows[key][field], value, f"raw metric {key}:{field}")
    qualified = qualification(raw, records, config)
    compare(summary["qualification"], qualified, "fresh qualification")
    eligible = (config["seeds"] == list(range(13001, 13007)) and set(config["models"]) == {"conditional", "recurrent"}
                and config["prefix_blocks"] == 4 and config["prefix_size"] == 1024 and config["episode_size"] == 8192)
    compare(summary["eligible_development_cohort"], eligible, "cohort eligibility")
    for model in config["models"]:
        contrasts = {}
        for control in ("reference", "current", "shrink"):
            contrasts[control] = {field: bootstrap([raw[seed, model, "combined"][field]-raw[seed, model, control][field]
                for seed in config["seeds"]], config["seeds"]) for field in FIELDS}
            for field, value in contrasts[control].items():
                compare(summary["contrasts"][model][f"combined_minus_{control}"][field], value,
                        f"paired contrast {model}/combined_minus_{control}/{field}")
        verify_screen(summary["screen"][model], contrasts, qualified["by_model"][model], eligible)
        verify_screen(summary["protection_attribution"][model], contrasts, qualified["by_model"][model], eligible, attribution=True)
    return dict(independent_metric_rows=len(raw), critical_paired_contrasts=len(config["models"])*3*len(FIELDS),
                fresh_qualification_recomputed=True, outcome_and_attribution_screens_recomputed=True)


def verify_cohort(folder, name):
    manifest, summary, config = read(folder/"manifest.json"), read(folder/"summary.json"), read(folder/"config.json")
    identity = manifest["identity"]
    require(manifest["config"] == config == summary["config"] and summary["identity"] == identity, "cohort copies disagree")
    for key, value in (("config_sha256", config), ("source_sha256", manifest["source_files"]), ("runtime_sha256", manifest["runtime"])):
        require(digest(value) == identity[key], "manifest identity mismatch")
    verify_zip(folder/"training_source.zip", manifest["source_files"])
    protocol = read(folder/"protocol_lock.json")
    require(all(protocol[key] == value for key, value in identity.items())
            and sha((folder/"protocol_at_lock.md").read_bytes()) == protocol["protocol_sha256"], "protocol lock mismatch")
    analysis = read(folder/"analysis_lock.json")
    require(analysis["config"] == config and analysis["config_sha256"] == identity["config_sha256"], "analysis configuration mismatch")
    verify_zip(folder/"analysis_at_lock.zip", analysis["files"])
    required = {"scripts/summarize_selective_updates.py", "scripts/summarize_training_state.py",
                "scripts/summarize_replay_renewal.py", "tests/test_selective_update_summary.py"}
    require(required <= analysis["files"].keys() and any(path.startswith("docs/") and expected == protocol["protocol_sha256"]
            for path, expected in analysis["files"].items()), "analysis/protocol coverage missing from lock")
    require(config["models"] == ["conditional", "recurrent"], "architecture coverage changed")
    require(config["seeds"] == ([13991] if name == "smoke" else list(range(13001, 13007))), "seed coverage changed")
    trajectories = len(config["seeds"])*len(config["models"])
    counts = dict(trajectories=trajectories, episodes=6*trajectories, prefix_fits=trajectories)
    require(summary["counts"] == counts, "summary counts mismatch")
    completion = read(folder/"completion.json")
    require(completion["identity"] == identity and all(completion[key] == value for key, value in counts.items()), "completion mismatch")
    pairs = {(seed, model) for seed in config["seeds"] for model in config["models"]}
    require(len(completion["jobs"]) == trajectories and {(row["seed"], row["model"]) for row in completion["jobs"]} == pairs
            and all(row["affinity"] == manifest["runtime"]["affinity"] for row in completion["jobs"]), "worker coverage/runtime mismatch")
    records, prefixes, original_hashes = read_records(folder, config, identity, summary)
    with (folder/"rows.jsonl").open(encoding="utf-8") as stream:
        compare([decode(line) for line in stream], summary["rows"], "portable row copy")
    recomputed = verify_results(records, config, summary)
    audit = read(folder/"audit.json")
    require(audit["passed"] is True and audit["identity"] == identity
            and audit["prefix_checkpoints"] == trajectories and audit["before_checkpoints"] == counts["episodes"]
            and audit["final_checkpoints"] == counts["episodes"] and audit["matched_carried_arm_groups"] == trajectories,
            "full checkpoint audit count/identity mismatch")
    require(len(audit["prefix_checks"]) == trajectories
            and {(row["seed"], row["model"]) for row in audit["prefix_checks"]} == pairs
            and len(audit["checks"]) == len(records)
            and {(row["seed"], row["model"], row["arm"]) for row in audit["checks"]} == set(records), "audit coverage mismatch")
    for row in audit["checks"]:
        require(all(row[key] is True for key in ("exact_initial_state_verified", "causal_data_replay_verified",
            "optimizer_state_and_counters_verified", "endpoint_probes_recomputed")), "audit contains an unpassed episode")
    integrity = read(folder/"file_integrity.json")
    files = {row["path"]: row for row in integrity["files"]}
    require(integrity["passed"] is True and len(files) == len(integrity["files"])
            and all(row["valid"] is True for row in files.values()), "stable-file check failed or duplicated")
    expected_files = set(original_hashes)
    for path in original_hashes:
        if path.endswith("prefix.json"):
            expected_files.add(path[:-len("prefix.json")]+"prefix.pt")
        else:
            expected_files.update(path[:-len("result.json")]+ending for ending in ("before.pt", "checkpoint.pt"))
    metadata = ("manifest.json", "completion.json", "protocol_lock.json", "training_source.zip", "analysis_lock.json", "analysis_at_lock.zip")
    expected_files.update(metadata)
    require(expected_files <= files.keys(), "stable-file check omits required checkpoint/record/lock files")
    for path, expected in original_hashes.items():
        require(files[path]["sha256"] == expected, "stable-file hash differs from original portable JSON")
    for path in metadata:
        data = (folder/path).read_bytes()
        require(files[path]["sha256"] == sha(data) and files[path]["bytes"] == len(data), "stable metadata file differs from portable copy")
    record_integrity = read(folder/"record_integrity.json")
    require(record_integrity == summary["record_integrity"] and record_integrity["passed"] is True
            and record_integrity["identity"] == identity, "record-integrity copy mismatch")
    return dict(**counts, raw_prefix_records=len(prefixes), stable_files=len(files), source_sha256=identity["source_sha256"],
                protocol_sha256=protocol["protocol_sha256"], **recomputed)


def verify(directory):
    directory = Path(directory).resolve()
    artifacts = verify_artifacts(directory)
    cohorts = {name: verify_cohort(directory/name, name) for name in ("smoke", "development")}
    require(cohorts["smoke"]["source_sha256"] == cohorts["development"]["source_sha256"], "cohorts used different scientific source")
    require(cohorts["smoke"]["protocol_sha256"] == cohorts["development"]["protocol_sha256"], "cohorts used different protocols")
    result = dict(passed=True, artifacts=artifacts, cohorts=cohorts,
        scope="Portable byte/lock/coverage checks and independent recomputation of critical recorded metrics, bootstrap intervals and screens; no model retraining or scientific replication.")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    verify(parser.parse_args().input)
