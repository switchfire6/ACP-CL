"""Audit all sixteen frozen CIFAR-10 transfer runs and apply the declared rule.

This is a study-specific descriptive report, with individual paired outcomes
and no bootstrap. Nothing is written until all runs and pairing checks pass.
Only trusted local checkpoints produced by this project may be supplied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

if __package__:
    from . import prepare_v3_study as validation
    from .archive_results import reviewable_json
    from .audit_v2_pairing import buffer_hash
else:
    import prepare_v3_study as validation
    from archive_results import reviewable_json
    from audit_v2_pairing import buffer_hash


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-12
METHODS = ["er_v3", "recycle_v3", "newborn_v3", "newborn_matched_v3"]
CONDITIONS = ["recurring", "stationary"]
SEEDS = [1063, 1174]
IDENTITY_KEYS = ("config_sha256", "source_sha256", "runtime_sha256", "runtime_fingerprint", "execution_device")
CONDITION_METADATA = {
    "regime", "domain_order", "boundary_change_flags", "signal_change_flags", "signal_domain_order",
    "validation_domain_ids_by_experience", "validation_pool_ids_by_experience",
}


def read(path):
    return validation._load(path)[0]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path, root):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("report input/output paths must stay inside the workspace") from error


def finite(value, name, low=None, high=None):
    return validation._finite(value, name, low, high)


def close(observed, expected, name, tolerance=1e-10):
    if not math.isclose(finite(observed, name), expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{name} differs from independently recomputed value")


def same_json_values(left, right):
    """Compare checkpoint tuples and JSON arrays without weakening value checks."""
    def normalize(value):
        return json.loads(json.dumps(value, allow_nan=False))

    return normalize(left) == normalize(right)


def id_hash(ids):
    if not isinstance(ids, list) or any(type(value) is not int or not 0 <= value < 50000 for value in ids):
        raise ValueError("base-image IDs must be official training positions 0..49999")
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def validate_lock(path, root):
    lock = read(path)
    if lock.get("schema_version") != 1 or lock.get("status") != "frozen before first comparative run":
        raise ValueError("a frozen pre-outcome version-one lock is required")
    validation._identity(lock)
    for key, expected in (("methods", METHODS), ("conditions", CONDITIONS), ("seeds", SEEDS),
                          ("expected_runs", 16), ("expected_updates_per_run", 1410),
                          ("expected_current_examples_per_run", 45000)):
        if lock.get(key) != expected:
            raise ValueError(f"lock {key} differs from the declared pilot")
    commit = lock.get("source_freeze_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("missing full source-freeze commit")
    if lock.get("execution_device") != "cuda":
        raise ValueError("the locked comparative pilot requires CUDA")
    for kind in ("study", "protocol"):
        file = root / lock[kind]["path"]
        relative(file, root)
        if sha(file) != lock[kind]["sha256"]:
            raise ValueError(f"frozen {kind} byte hash mismatch")
    study = read(root / lock["study"]["path"])
    for key in ("methods", "conditions", "seeds", "expected_runs", "expected_updates_per_run", "expected_current_examples_per_run"):
        if study.get(key) != lock[key]:
            raise ValueError(f"study and lock disagree on {key}")
    if study.get("protocol_sha256") != lock["protocol"]["sha256"] or study.get("expected_reset_units_per_recycling_run") != 204:
        raise ValueError("study protocol/reset-count identity mismatch")
    configs = {}
    if set(lock.get("configurations", {})) != set(CONDITIONS):
        raise ValueError("lock must cover both configurations exactly")
    for condition in CONDITIONS:
        entry = lock["configurations"][condition]
        file = root / entry["path"]
        relative(file, root)
        config = read(file)
        if sha(file) != entry["sha256"] or validation.config_hash(config) != entry["config_sha256"]:
            raise ValueError("frozen configuration identity mismatch")
        if config.get("study_spec_sha256") != lock["study"]["sha256"] or config.get("methods") != METHODS or config.get("seeds") != SEEDS:
            raise ValueError("configuration differs from the frozen study")
        if config["data"].get("regime") != condition or config.get("single_pass") is not True \
                or config.get("epochs") != 1 or "steps_per_experience" in config:
            raise ValueError("configuration violates condition/single-pass protocol")
        configs[condition] = config
    archive = root / lock["dataset_archive"]["path"]
    relative(archive, root)
    if archive.stat().st_size != lock["dataset_archive"]["bytes"] or sha(archive) != lock["dataset_archive"]["sha256"]:
        raise ValueError("local dataset archive differs from the frozen archive identity")
    return lock, study, configs


def endpoints(result, condition):
    """Independent fixed-point AUC, final-domain and stationary drawdown calculation."""
    if condition not in CONDITIONS:
        raise ValueError("unknown pilot condition")
    matrix, curves = result.get("accuracy_matrix"), result.get("learning_curves")
    if not isinstance(matrix, list) or len(matrix) != 30 or any(not isinstance(row, list) or len(row) != 30 for row in matrix):
        raise ValueError("require the complete 30-by-30 accuracy matrix")
    if not isinstance(curves, list) or len(curves) != 30:
        raise ValueError("require all 30 acquisition curves")
    positions = [0] + list(range(128, 1409, 128)) + [1500]
    aucs = []
    for index, curve in enumerate(curves):
        if not isinstance(curve, list) or any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in curve) \
                or [point[0] for point in curve] != positions:
            raise ValueError("acquisition evaluations must cover 0/128/256/.../1408/1500 presentations")
        for _, value in curve:
            finite(value, "curve accuracy", 0, 1)
        # Two trapezoids of width 128, divided by the fixed 256-image horizon.
        aucs.append((curve[0][1] + 2*curve[1][1] + curve[2][1])/4)
        for j, value in enumerate(matrix[index]):
            observed = j == index or ((index+1) % 10 == 0 and j <= index)
            if observed:
                finite(value, "observed matrix accuracy", 0, 1)
            elif value is not None:
                raise ValueError("matrix contains measurements outside the declared evaluation schedule")
        close(matrix[index][index], curve[-1][1], "matrix diagonal")
    order = [i % 3 for i in range(30)] if condition == "recurring" else [0]*30
    if result.get("stream_metadata", {}).get("domain_order") != order:
        raise ValueError("domain order differs from the declared condition")
    domains = {}
    for domain in sorted(set(order)):
        values = [matrix[-1][i] for i, value in enumerate(order) if value == domain]
        if max(values)-min(values) > TOLERANCE:
            raise ValueError("same-domain final measurements disagree within one snapshot")
        domains[("original", "grayscale", "blur")[domain]] = statistics.fmean(values)
    final, late = statistics.fmean(matrix[-1]), statistics.fmean(aucs[15:])
    close(statistics.fmean(domains.values()), final, "equal-domain final mean")
    if not isinstance(result.get("early_auc"), list) or len(result["early_auc"]) != 30:
        raise ValueError("missing reported per-experience AUCs")
    for actual, expected in zip(result["early_auc"], aucs):
        close(actual, expected, "reported acquisition AUC")
    for key, expected in (("final_accuracy", final), ("late_early_auc", late)):
        close(result.get("metrics", {}).get(key), expected, key)
    for key, value in result["metrics"].items():
        finite(value, f"reported metric {key}")
    diagonals = [matrix[i][i] for i in range(30)]
    peak, drawdown = diagonals[0], 0.0
    for value in diagonals:
        peak, drawdown = max(peak, value), max(drawdown, peak-value)
    return {"early_auc_256_by_experience": aucs, "late_early_auc": late,
            "final_accuracy": final, "final_accuracy_by_domain": domains,
            "untrained_original_accuracy": curves[0][0][1],
            "stationary_end_of_experience_accuracy": diagonals if condition == "stationary" else None,
            "stationary_maximum_drawdown": drawdown if condition == "stationary" else None}


def effort_decision(rows, rule):
    """Serialize every comparison, including inclusive-threshold rounding tolerance."""
    if rule.get("newborn_late_auc_each_gain_positive") is not True \
            or rule.get("matched_late_auc_each_gain_nonnegative") is not True:
        raise ValueError("decision rule differs from the predeclared per-seed comparisons")
    expected = {(c, m, s) for c in CONDITIONS for m in METHODS for s in SEEDS}
    indexed = {(r["condition"], r["method"], r["seed"]): r["recomputed"] for r in rows}
    if len(rows) != 16 or set(indexed) != expected:
        raise ValueError("decision requires all sixteen unique predeclared runs")
    checks = {"adequacy": [], "signal": [], "retention": []}

    def record(group, value, operator, threshold, **context):
        value = finite(value, "decision operand")
        passed = {">=": value >= threshold-TOLERANCE, "<=": value <= threshold+TOLERANCE,
                  ">": value > threshold+TOLERANCE, "<": value < threshold-TOLERANCE}[operator]
        checks[group].append({**context, "value": value, "operator": operator, "threshold": threshold,
                              "passed": passed, "reason": f"{value:.12g} {operator} {threshold:.12g}: {'pass' if passed else 'fail'}"})

    paired = []
    for seed in SEEDS:
        for method in ("er_v3", "recycle_v3"):
            row = indexed["stationary", method, seed]
            context = {"condition": "stationary", "method": method, "seed": seed}
            record("adequacy", row["final_accuracy"], ">=", rule["stationary_baseline_final_min"], metric="final_accuracy", **context)
            record("adequacy", row["final_accuracy"]-row["untrained_original_accuracy"], ">=",
                   rule["stationary_baseline_improvement_min"], metric="improvement_over_untrained", **context)
        baseline = indexed["recurring", "recycle_v3", seed]
        record("adequacy", baseline["final_accuracy"], ">=", rule["recurring_recycling_final_min"],
               condition="recurring", method="recycle_v3", seed=seed, metric="final_accuracy")
        record("adequacy", baseline["late_early_auc"], "<", rule["recurring_recycling_late_auc_max_exclusive"],
               condition="recurring", method="recycle_v3", seed=seed, metric="late_early_auc")
        for method in ("newborn_v3", "newborn_matched_v3"):
            difference = indexed["recurring", method, seed]["late_early_auc"]-baseline["late_early_auc"]
            paired.append({"method": method, "seed": seed, "newborn_late_auc": indexed["recurring", method, seed]["late_early_auc"],
                           "recycling_late_auc": baseline["late_early_auc"], "difference": difference})
            record("signal", difference, ">" if method == "newborn_v3" else ">=", 0,
                   condition="recurring", method=method, seed=seed, metric="paired_late_auc_gain")
            for condition in CONDITIONS:
                for comparator in ("er_v3", "recycle_v3"):
                    loss = indexed[condition, comparator, seed]["final_accuracy"]-indexed[condition, method, seed]["final_accuracy"]
                    record("retention", loss, "<=", rule["maximum_final_loss_against_er_or_recycling"],
                           condition=condition, method=method, comparator=comparator, seed=seed, metric="final_accuracy_loss")
    for method, key in (("newborn_v3", "newborn_late_auc_mean_gain_min"),
                        ("newborn_matched_v3", "matched_late_auc_mean_gain_min")):
        record("signal", statistics.fmean(r["difference"] for r in paired if r["method"] == method), ">=", rule[key],
               condition="recurring", method=method, seeds=SEEDS, metric="mean_paired_late_auc_gain")
    passed = {name: all(item["passed"] for item in items) for name, items in checks.items()}
    advance = all(passed.values())
    return {"advance_frozen_recipe": advance, "decision": "advance to a separately planned larger study" if advance else "stop scaling this frozen recipe",
            "passed": passed, "checks": checks, "paired_recurring_late_auc": paired,
            "absolute_comparison_tolerance": TOLERANCE,
            "comparison_semantics": "Inclusive thresholds permit 1e-12 rounding error; strict positive/below thresholds require separation beyond 1e-12.",
            "scope": rule["scope"], "uncertainty": "Two paired seeds; no bootstrap or significance claim."}


def audit_arrivals(artifact, metadata, seed):
    groups = metadata.get("current_arrival_ids_by_experience")
    if not isinstance(groups, list) or len(groups) != 30 or any(len(ids) != 1500 for ids in groups):
        raise ValueError("source arrival IDs must cover 30 x 1500 images")
    source_ids = [value for ids in groups for value in ids]
    if len(set(source_ids)) != 45000 or artifact.get("source_ordered_ids_sha256") != id_hash(source_ids):
        raise ValueError("source arrival uniqueness/hash mismatch")
    if metadata.get("current_arrival_id_hashes_by_experience") != [id_hash(ids) for ids in groups]:
        raise ValueError("source per-experience ID hash mismatch")
    records = artifact.get("experiences")
    if not isinstance(records, list) or len(records) != 30:
        raise ValueError("actual-arrival artifact is incomplete")
    actual = []
    for index, (ids, row) in enumerate(zip(groups, records)):
        generator = torch.Generator().manual_seed(seed+12907+997*index)
        expected = [ids[position] for position in torch.randperm(1500, generator=generator).tolist()]
        if row.get("experience") != index or row.get("ordered_base_image_ids") != expected \
                or row.get("batch_sizes") != [32]*46+[28] or row.get("current_examples") != 1500 \
                or row.get("optimizer_steps") != 47 or row.get("ordered_base_image_ids_sha256") != id_hash(expected):
            raise ValueError("actual arrival sequence/batch exposure differs from the private seeded single pass")
        actual.extend(expected)
    for key, value in (("current_examples", 45000), ("unique_current_base_images", 45000),
                       ("optimizer_steps", 1410), ("completed_experiences", 30),
                       ("ordered_arrival_ids_sha256", id_hash(actual))):
        if artifact.get(key) != value:
            raise ValueError(f"actual-arrival artifact {key} mismatch")
    splits = metadata["split_indices"]
    for name, ids in splits.items():
        if metadata["split_index_hashes"].get(name) != id_hash(ids) or len(set(ids)) != len(ids):
            raise ValueError("source split identity/uniqueness mismatch")
    if set(splits) != {"train", "validation", "validation_reserved", "validation_unused", "unused_train", "test"} \
            or splits["train"] != source_ids or len(splits["validation"]) != 500 or len(splits["validation_reserved"]) != 5000 \
            or set(source_ids) & set(splits["validation_reserved"]) or not set(splits["validation"]) <= set(splits["validation_reserved"]) \
            or set(splits["validation_unused"]) != set(splits["validation_reserved"])-set(splits["validation"]) \
            or splits["test"] or splits["unused_train"]:
        raise ValueError("source splits violate the training/validation reservation")
    description = metadata["source_description"]
    fingerprint = hashlib.sha256(json.dumps(description, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if fingerprint != metadata.get("source_fingerprint") or metadata.get("official_test_images_materialized") is not False:
        raise ValueError("source fingerprint or official-test isolation mismatch")
    return id_hash(actual)


def audit_allocation(trace, method, result):
    if method not in METHODS:
        raise ValueError("unknown pilot method")
    if not isinstance(trace, list) or len(trace) != 1410:
        raise ValueError("allocation trace must contain 1410 updates")
    resets, clipping = 0, 0
    for index, event in enumerate(trace, 1):
        count = 3 if method != "er_v3" and index > 40 and (index-40) % 20 == 0 else 0
        if event.get("step") != index or event.get("recycled_by_population") != {"adapter": count}:
            raise ValueError("allocation step/reset schedule mismatch")
        resets += count
        proposed = finite(event.get("proposed_update_norm"), "proposed norm", 0)
        applied = finite(event.get("update_norm"), "applied norm", 0, 0.100001)
        scale = finite(event.get("clip_scale"), "clip scale", 0, 1)
        if scale <= 0:
            raise ValueError("clip scale must be positive")
        close(scale, min(1, 0.1/proposed) if proposed else 1, "clip scale", 2e-6)
        close(applied, min(proposed, 0.1), "complete update norm", 2e-6)
        nominal = finite(event.get("nominal_feature_gain"), "nominal feature gain", 0, 2.000001)
        target = 1 if index <= 40 else 0.5
        if method != "newborn_v3" or index <= 40:
            close(nominal, target, "nominal feature budget", 2e-6)
        elif nominal < target-2e-6:
            raise ValueError("newborn feature mean falls below its mature gain")
        close(event.get("effective_feature_gain"), nominal*scale, "achieved feature gain", 2e-6)
        for key in ("feature_data_displacement", "head_data_displacement"):
            finite(event.get(key), key, 0)
        clipping += int(scale < 1)
    if resets != (0 if method == "er_v3" else 204) or result["diagnostics"].get("total_recycled") != resets \
            or result["allocation_summary"].get("steps") != 1410:
        raise ValueError("cumulative update/reset counts disagree")
    return {"updates": 1410, "unit_resets": resets, "clipped_updates": clipping,
            "post_warmup_mean_nominal_feature_gain": statistics.fmean(row["nominal_feature_gain"] for row in trace[40:]),
            "post_warmup_maximum_nominal_feature_gain": max(row["nominal_feature_gain"] for row in trace[40:]),
            "maximum_optimizer_update_norm": max(row["update_norm"] for row in trace),
            "scope": "The complete optimizer cap excludes structural reset mutations."}


def replay_nonpixel_hash(buffer):
    # Across curricula pixels intentionally differ; reservoir and sampling RNG
    # states, labels and unique arrival order can still be checked exactly.
    nonpixel = dict(buffer)
    nonpixel["x"] = torch.empty(0, dtype=torch.uint8)
    return buffer_hash(nonpixel)


def summarize(runs: Path, output: Path, lock_path: Path, *, root: Path = ROOT):
    root, runs, output, lock_path = root.resolve(), runs.resolve(), output.resolve(), lock_path.resolve()
    for path in (runs, output, lock_path):
        relative(path, root)
    lock, study, configs = validate_lock(lock_path, root)
    expected = {runs / c / f"{m}_seed{s}" / "result.json" for c in CONDITIONS for m in METHODS for s in SEEDS}
    if set(runs.rglob("result.json")) != expected or {p.name for p in runs.iterdir() if p.is_dir()} != set(CONDITIONS):
        raise ValueError("missing/unexpected/duplicate comparative result directories; all sixteen are required")
    rows, checks, internal, seed_provenance = [], [], {}, {}
    for condition in CONDITIONS:
        directory = runs / condition
        config = configs[condition]
        manifest = read(directory / "manifest.json")
        expected_directories = {f"{m}_seed{s}" for m in METHODS for s in SEEDS}
        if {p.name for p in directory.iterdir() if p.is_dir()} != expected_directories:
            raise ValueError("unexpected/incomplete attempt directory")
        expected_identity = {key: lock[key] for key in IDENTITY_KEYS if key != "config_sha256"}
        expected_identity["config_sha256"] = lock["configurations"][condition]["config_sha256"]
        if manifest.get("config") != config or manifest.get("methods") != METHODS or manifest.get("seeds") != SEEDS \
                or any(manifest.get(k) != v for k, v in expected_identity.items()):
            raise ValueError("suite manifest differs from the frozen lock")
        for method in METHODS:
            for seed in SEEDS:
                location = directory / f"{method}_seed{seed}"
                result = read(location / "result.json")
                if any(result.get(k) != v for k, v in expected_identity.items()) or result.get("config") != config \
                        or result.get("method") != method or result.get("seed") != seed:
                    raise ValueError("result identity differs from locked condition/method/seed")
                if result.get("eval_split") != "validation" or result.get("class_order") != list(range(10)) \
                        or result.get("information_access") != "training stream only":
                    raise ValueError("fixed labels/evaluation/information access mismatch")
                if result.get("evaluation_schedule", {}).get("early_examples") != 256 \
                        or result["evaluation_schedule"].get("retention_every") != 10:
                    raise ValueError("evaluation schedule differs from the frozen protocol")
                recomputed = endpoints(result, condition)
                arrival_path = location / "current_arrivals.json"
                arrivals = read(arrival_path)
                if result["current_arrival_audit"].get("artifact_sha256") != sha(arrival_path):
                    raise ValueError("actual-arrival artifact byte hash mismatch")
                arrival_digest = audit_arrivals(arrivals, result["stream_metadata"], seed)
                if result["cost"].get("current_examples") != 45000 or result["current_arrival_audit"].get("reservoir_current_arrivals") != 45000:
                    raise ValueError("actual current exposure/replay arrivals mismatch")
                allocation_path = location / "allocation.json"
                if result.get("allocation_trace_sha256") != sha(allocation_path):
                    raise ValueError("allocation trace byte hash mismatch")
                trace = read(allocation_path)["trace"]
                allocation_check = audit_allocation(trace, method, result)
                saved = torch.load(location / "checkpoint.pt", map_location="cpu", weights_only=False)
                for key in (*IDENTITY_KEYS, "method", "seed"):
                    if saved.get(key) != result[key]:
                        raise ValueError("checkpoint/result identity mismatch")
                learner = saved["learner"]
                if saved.get("completed") != 30 or saved.get("current_arrivals") != arrivals \
                        or learner.get("step_number") != 1410 or learner["buffer"].get("num_seen") != 45000 \
                        or learner.get("allocation_trace") != trace or learner.get("cost") != result["cost"] \
                        or not same_json_values(saved.get("curves"), result["learning_curves"]):
                    raise ValueError("checkpoint completion/exposure/trace/curve mismatch")
                saved_matrix = [[None if math.isnan(value) else value for value in row] for row in saved["matrix"]]
                if saved_matrix != result["accuracy_matrix"] or learner["model"]["head.weight"].shape[0] != 10 \
                        or any(not bool(torch.isfinite(value).all()) for value in learner["model"].values()):
                    raise ValueError("checkpoint matrix/fixed classifier/finite parameter mismatch")
                metadata = result["stream_metadata"]
                common = {k: v for k, v in metadata.items() if k not in CONDITION_METADATA}
                domain_metadata = {k: metadata[k] for k in CONDITION_METADATA}
                if seed not in seed_provenance:
                    seed_provenance[seed] = {"common_metadata": common, "conditions": {}, "actual_current_arrivals": arrivals}
                provenance = seed_provenance[seed]
                if provenance["common_metadata"] != common or provenance["actual_current_arrivals"] != arrivals:
                    raise ValueError("paired source/split/current-arrival provenance differs between methods or conditions")
                if condition in provenance["conditions"] and provenance["conditions"][condition] != domain_metadata:
                    raise ValueError("paired condition metadata differs between methods")
                provenance["conditions"][condition] = domain_metadata
                compact_fields = ("method", "seed", *IDENTITY_KEYS, "environment", "class_order", "eval_split", "metrics",
                                  "accuracy_matrix", "learning_curves", "early_auc", "cost", "wall_seconds", "experience_seconds",
                                  "diagnostics", "allocation_summary", "current_arrival_audit", "stream_fingerprints",
                                  "model_parameters", "replay_bytes", "peak_cuda_bytes", "evaluation_schedule", "information_access")
                row = {key: result[key] for key in compact_fields if key in result}
                row.update(condition=condition, recomputed=recomputed, run_path=relative(location, root),
                           source_provenance=f"provenance/seed{seed}.json")
                row["artifact_sha256"] = {name: sha(location / name) for name in
                                           ("result.json", "allocation.json", "current_arrivals.json", "events.json", "checkpoint.pt")}
                rows.append(row)
                checks.append({"condition": condition, "method": method, "seed": seed, **allocation_check,
                               "arrival_ids_sha256": arrival_digest, "result_sha256": row["artifact_sha256"]["result.json"]})
                internal[condition, method, seed] = {"stream": result["stream_fingerprints"], "warmup": trace[:40],
                    "replay": buffer_hash(learner["buffer"]), "replay_nonpixel": replay_nonpixel_hash(learner["buffer"]),
                    "arrival": arrival_digest, "cost": {k: result["cost"][k] for k in ("current_examples", "replay_examples",
                        "probe_forward_examples", "train_forward_calls", "backward_calls")}}
    pairing = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            values = [internal[condition, method, seed] for method in METHODS]
            if any(value != values[0] for value in values[1:]):
                raise ValueError("within-condition actual stream/warmup/replay/exposure pairing failed")
            pairing.append({"condition": condition, "seed": seed, "methods": METHODS,
                            "stream_fingerprints": values[0]["stream"], "arrival_ids_sha256": values[0]["arrival"],
                            "replay_and_sampling_rng_sha256": values[0]["replay"], "common_training_cost": values[0]["cost"],
                            "warmup_updates_exact": 40})
        a, b = (internal[condition, "er_v3", seed] for condition in CONDITIONS)
        for key in ("warmup", "replay_nonpixel", "arrival", "cost"):
            if a[key] != b[key]:
                raise ValueError(f"cross-condition {key} pairing failed")
    decision = effort_decision(rows, study["decision_rule"])
    summary = {"schema_version": 1, "status": "all sixteen frozen comparative runs audited",
               "study_scope": "Sixteen fixed development runs in a two-seed exploratory transfer pilot; effort-allocation rule, not efficacy or statistical significance.",
               "lock": {"path": relative(lock_path, root), "sha256": sha(lock_path), "source_freeze_commit": lock["source_freeze_commit"],
                        **{key: lock[key] for key in IDENTITY_KEYS if key != "config_sha256"}},
               "comparative_runs": 16, "optimizer_updates": 22560, "engineering_runs_included": 0,
               "decision": decision, "runs": [{"condition": row["condition"], "method": row["method"], "seed": row["seed"],
                    **row["recomputed"]} for row in rows],
               "timing_caution": "Logged wall time includes concurrent workloads; it is not GPU time or a method-speed comparison.",
               "retention_caution": "Repeated domain panels are correlated; generic task-column forgetting/BWT is archived but not treated as independent-task retention."}
    pairing_report = {"checks_passed": True, "runs": checks, "within_condition": pairing,
                      "cross_condition": {"paired_arrivals_and_split_ids": True, "paired_replay_labels_and_rngs": True,
                                          "common_original_first_40_updates": True,
                                          "pixels": "Transformed replay pixels intentionally differ between curricula."}}
    payloads = {
        output / "summary.json": (reviewable_json(summary)+"\n").encode(),
        output / "individual_results.json": (reviewable_json(rows)+"\n").encode(),
        output / "pairing_audit.json": (reviewable_json(pairing_report)+"\n").encode(),
        output / "summary.md": markdown_summary(summary).encode(),
    }
    for seed, provenance in seed_provenance.items():
        # Source ID groups duplicate split_indices.train exactly. Retain one
        # source sequence and the distinct actually shuffled arrival sequence.
        compact = copy.deepcopy(provenance)
        compact["common_metadata"].pop("current_arrival_ids_by_experience")
        compact["source_experience_id_layout"] = "Consecutive 1500-item blocks in common_metadata.split_indices.train"
        payloads[output / f"provenance/seed{seed}.json"] = (reviewable_json(compact)+"\n").encode()
    validation._write_without_conflicts(payloads)
    return summary


def markdown_summary(summary):
    decision = summary["decision"]
    lines = ["# CIFAR-10 transfer pilot", "", summary["study_scope"], "",
             "All 16 predeclared development runs completed; 22,560 optimizer updates. Engineering smoke runs are excluded. No cross-seed confidence interval is estimated.", "",
             f"Frozen source SHA-256: `{summary['lock']['source_sha256']}`. Source-freeze commit: `{summary['lock']['source_freeze_commit']}`.", "",
             "| Condition | Method | Seed | Final % | Late first-256 AUC % | Stationary max drawdown, pp |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary["runs"]:
        drawdown = row["stationary_maximum_drawdown"]
        lines.append(f"| {row['condition']} | {row['method']} | {row['seed']} | {100*row['final_accuracy']:.2f} | "
                     f"{100*row['late_early_auc']:.2f} | {'n/a' if drawdown is None else f'{100*drawdown:.2f}'} |")
    lines += ["", f"Predeclared effort decision: **{decision['decision']}**.", "",
              "| Check | Pass |", "|---|---|"]
    for name, passed in decision["passed"].items():
        lines.append(f"| {name} | {'yes' if passed else 'no'} |")
    lines += ["", "Each paired recurring late-AUC difference (percentage points):", "",
              "| Method minus recycling | Seed | Difference, pp |", "|---|---:|---:|"]
    for row in decision["paired_recurring_late_auc"]:
        lines.append(f"| {row['method']} | {row['seed']} | {100*row['difference']:+.3f} |")
    lines += ["", "Every threshold operand, comparison and pass/fail reason is retained in summary.json. Per-domain final accuracies, all 30 AUCs, and all stationary diagonals are retained there too.", "",
              summary["retention_caution"], "", summary["timing_caution"], "",
              "The optimizer cap excludes reset mutations. Nominal gain matching precedes clipping. Raw checkpoints are locally retained; compact artifacts reference their hashes without distributing them.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/cifar10_transfer")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cifar10_transfer")
    parser.add_argument("--lock", type=Path, default=ROOT / "configs/cifar10_transfer/lock.json")
    args = parser.parse_args(argv)
    try:
        result = summarize(args.runs, args.output, args.lock)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"CIFAR transfer report refused: {error}\n")
    print(f"Audited {result['comparative_runs']} comparative runs; {result['decision']['decision']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
