"""Independent arithmetic/checkpoint audit; no training or project metric imports.

Run only after the complete locked cohort exists. Checkpoints must be trusted
local files. This is an implementation/artifact check, not a training replication.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(first, second, message, tolerance=1e-10):
    require(math.isfinite(float(first)) and math.isfinite(float(second))
            and abs(first-second) <= tolerance, message)


def tensor_tree_hash(value):
    """Also reject nonfinite learner state while hashing exact tensor bytes."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            require(not item.is_floating_point() or bool(torch.isfinite(item).all()),
                    "nonfinite checkpoint tensor")
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for member in item:
                visit(member)
        else:
            require(not isinstance(item, float) or math.isfinite(item), "nonfinite checkpoint scalar")
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def area(curve, horizon):
    points = np.asarray(curve, dtype=float)
    require(np.isfinite(points).all() and points[0, 0] == 0
            and np.all(np.diff(points[:, 0]) > 0) and points[-1, 0] >= horizon,
            "invalid acquisition curve")
    require(np.all((points[:, 1] >= 0) & (points[:, 1] <= 1)), "invalid accuracy")
    total = 0.0
    for (left, low), (right, high) in zip(points, points[1:]):
        if left >= horizon:
            break
        end = min(right, horizon)
        interpolated = low + (high-low)*(end-left)/(right-left)
        total += (end-left)*(low+interpolated)/2
    return total/horizon


def endpoints(result, config):
    matrix = np.asarray(result["accuracy_matrix"], dtype=float)
    count = config["data"]["n_experiences"]
    require(matrix.shape == (count, count), "matrix shape")
    curves = result["learning_curves"]
    require(len(curves) == count, "incomplete curves")
    values = [area(curve, config["early_examples"]) for curve in curves]
    for index, (curve, value) in enumerate(zip(curves, values)):
        require([p[0] for p in curve] == [0, *range(128, 1500, 128), 1500], "curve schedule")
        close(value, result["early_auc"][index], "stored AUC")
        close(curve[-1][1], matrix[index, index], "curve/diagonal")
    order = result["stream_metadata"]["domain_order"]
    domains = {}
    for domain in sorted(set(order)):
        entries = [matrix[-1, i] for i, value in enumerate(order) if value == domain]
        require(all(value == entries[0] for value in entries), "shared final panel differs")
        domains[str(domain)] = entries[0]
    final, late = float(np.mean(list(domains.values()))), float(np.mean(values[count//2:]))
    close(final, result["metrics"]["final_accuracy"], "final endpoint")
    close(late, result["metrics"]["late_early_auc"], "late AUC endpoint")
    diagonal = np.diag(matrix)
    drawdown = float(max(np.maximum.accumulate(diagonal)-diagonal)) if len(domains) == 1 else None
    return {"final_accuracy": final, "late_early_auc": late, "early_auc": values,
            "untrained_original": curves[0][0][1], "per_domain_final": domains,
            "stationary_drawdown": drawdown}


def decision(rows, spec):
    rule, seeds = spec["decision_rule"], spec["seeds"]
    def at(condition, method, seed):
        return rows[f"{condition}/{method}_seed{seed}"]
    adequacy, guards, contrasts = [], [], {}
    for seed in seeds:
        for method in ("er_v3", "recycle_v3"):
            row = at("stationary", method, seed)
            improvement = row["final_accuracy"]-row["untrained_original"]
            adequacy.append({"seed": seed, "condition": "stationary", "method": method,
                             "final": row["final_accuracy"], "improvement": improvement,
                             "passed": row["final_accuracy"] >= rule["stationary_baseline_final_min"]
                             and improvement >= rule["stationary_baseline_improvement_min"]-1e-12})
        row = at("recurring", "recycle_v3", seed)
        adequacy.append({"seed": seed, "condition": "recurring", "method": "recycle_v3",
                         "final": row["final_accuracy"], "late_auc": row["late_early_auc"],
                         "passed": row["final_accuracy"] >= rule["recurring_recycling_final_min"]
                         and row["late_early_auc"] < rule["recurring_recycling_late_auc_max_exclusive"]})
    for condition in spec["conditions"]:
        for method in ("newborn_v3", "newborn_matched_v3"):
            for baseline in ("recycle_v3", "er_v3"):
                key = f"{condition}/{method}-minus-{baseline}"
                contrasts[key] = {}
                for metric in ("final_accuracy", "late_early_auc"):
                    pairs = [at(condition, method, seed)[metric]-at(condition, baseline, seed)[metric]
                             for seed in seeds]
                    contrasts[key][metric] = {"per_seed": dict(zip(map(str, seeds), pairs)),
                                              "mean": float(np.mean(pairs))}
                for seed in seeds:
                    delta = contrasts[key]["final_accuracy"]["per_seed"][str(seed)]
                    guards.append({"contrast": key, "seed": seed, "difference": delta,
                                   "passed": delta >= -rule["maximum_final_loss_against_er_or_recycling"]-1e-12})
    gain = contrasts["recurring/newborn_v3-minus-recycle_v3"]["late_early_auc"]
    matched = contrasts["recurring/newborn_matched_v3-minus-recycle_v3"]["late_early_auc"]
    signal = (all(v > 0 for v in gain["per_seed"].values())
              and gain["mean"] >= rule["newborn_late_auc_mean_gain_min"]-1e-12
              and all(v >= -1e-12 for v in matched["per_seed"].values())
              and matched["mean"] >= rule["matched_late_auc_mean_gain_min"]-1e-12)
    checks = {"adequacy_passed": all(x["passed"] for x in adequacy), "signal_passed": signal,
              "retention_guard_passed": all(x["passed"] for x in guards)}
    return {**checks, "advance": all(checks.values()), "adequacy": adequacy,
            "retention_guards": guards, "contrasts": contrasts}


def audit(root, locked):
    lock, spec = read(locked/"lock.json"), read(locked/"study.json")
    configs = {condition: read(locked/f"{condition}.json") for condition in spec["conditions"]}
    # Fail before reading/comparing any outcomes if even one planned result is missing.
    expected = [root/c/f"{m}_seed{s}" for c in spec["conditions"]
                for m in spec["methods"] for s in spec["seeds"]]
    require(len(expected) == spec["expected_runs"] == 16, "cohort size")
    require(all((p/"result.json").is_file() for p in expected), "complete 16-run cohort required")
    for condition in spec["conditions"]:
        require({p.parent for p in (root/condition).glob("*/result.json")}
                == {p for p in expected if p.parent.name == condition}, "unexpected cohort member")
    def git(*args):
        return subprocess.check_output(["git", *args])
    manifests = {c: read(root/c/"manifest.json") for c in spec["conditions"]}
    commits = {manifest["environment"]["git_commit"] for manifest in manifests.values()}
    require(len(commits) == 1, "different launch commits")
    commit = commits.pop()
    require(git("show", f"{commit}:configs/cifar10_transfer/lock.json") == (locked/"lock.json").read_bytes(),
            "launch commit does not contain current lock")
    git("merge-base", "--is-ancestor", lock["source_freeze_commit"], commit)
    source = hashlib.sha256()
    for name in sorted(git("ls-tree", "--name-only", lock["source_freeze_commit"], "src/acp_cl/")
                       .decode().splitlines()):
        if name.endswith(".py"):
            source.update(Path(name).name.encode())
            source.update(git("show", f"{lock['source_freeze_commit']}:{name}"))
    require(source.hexdigest() == lock["source_sha256"], "source-freeze hash")
    artifacts = [lock["study"], lock["protocol"], *lock["configurations"].values()]
    for artifact in artifacts:
        path = Path(artifact["path"])
        require(sha(path) == artifact["sha256"], f"locked bytes changed: {path}")
        require(git("show", f"{lock['source_freeze_commit']}:{path.as_posix()}") == path.read_bytes(),
                f"artifact absent from source-freeze commit: {path}")
    require(canonical(lock["runtime_fingerprint"]) == lock["runtime_sha256"], "runtime hash")
    identity = {k: lock[k] for k in ("source_sha256", "runtime_sha256", "runtime_fingerprint", "execution_device")}
    rows, evidence, pairings, cross_condition, source_fingerprints = {}, [], {}, {}, set()
    for directory in expected:
        condition = directory.parent.name
        config, manifest = configs[condition], manifests[condition]
        result, arrivals = read(directory/"result.json"), read(directory/"current_arrivals.json")
        checkpoint = torch.load(directory/"checkpoint.pt", map_location="cpu", weights_only=False)
        trace, events = read(directory/"allocation.json")["trace"], read(directory/"events.json")["events"]
        method, seed = result["method"], result["seed"]
        require(directory.name == f"{method}_seed{seed}", "run path identity")
        for saved in (manifest, result, checkpoint):
            require(all(saved[k] == v for k, v in identity.items()), "source/runtime/device drift")
            require(saved["config_sha256"] == canonical(config), "configuration hash")
        require(manifest["config"] == result["config"] == config, "configuration drift")
        require(manifest["seeds"] == spec["seeds"] and manifest["methods"] == spec["methods"], "manifest cohort")
        require(checkpoint["method"] == method and checkpoint["seed"] == seed and checkpoint["completed"] == 30,
                "checkpoint run identity/completeness")
        require(result["information_access"] == "training stream only" and result["eval_split"] == "validation",
                "information access")
        learner = checkpoint["learner"]
        finite_hash = tensor_tree_hash(learner)
        require(learner["controller"] is None and learner["sensor"] is None
                and learner["yoked_schedule"] is None, "unexpected controller/sensor/offline trace")
        require(json.loads(json.dumps(checkpoint["curves"])) == result["learning_curves"] and learner["log"] == events
                and learner["allocation_trace"] == trace and learner["cost"] == result["cost"], "checkpoint/artifact mismatch")
        require(checkpoint["current_arrivals"] == arrivals and sha(directory/"current_arrivals.json")
                == result["current_arrival_audit"]["artifact_sha256"], "arrival artifact identity")
        metadata = result["stream_metadata"]
        source_fingerprints.add(metadata["source_fingerprint"])
        train = metadata["split_indices"]["train"]
        reserved = metadata["split_indices"]["validation_reserved"]
        require(len(train) == len(set(train)) == 45000 and len(reserved) == len(set(reserved)) == 5000
                and set(train).isdisjoint(reserved) and set(train)|set(reserved) == set(range(50000)), "split leakage/counts")
        require(set(metadata["split_indices"]["validation"]) <= set(reserved)
                and len(metadata["split_indices"]["validation"]) == 500
                and metadata["official_test_images_materialized"] is False, "validation/test partition")
        observed = []
        for index, record in enumerate(arrivals["experiences"]):
            source_ids = metadata["current_arrival_ids_by_experience"][index]
            generator = torch.Generator().manual_seed(seed+12907+997*index)
            ids = [source_ids[i] for i in torch.randperm(1500, generator=generator).tolist()]
            require(record["ordered_base_image_ids"] == ids and record["batch_sizes"] == [32]*46+[28], "actual shuffled arrivals")
            observed.extend(ids)
        require(len(observed) == len(set(observed)) == 45000 and set(observed) == set(train), "one-pass exposure")
        require(arrivals["ordered_arrival_ids_sha256"] == hashlib.sha256(np.asarray(observed, dtype="<i8").tobytes()).hexdigest(), "arrival digest")
        costs = {"current_examples": 45000, "replay_examples": 45088, "probe_forward_examples": 18048,
                 "train_forward_calls": 2819, "backward_calls": 2819, "sensor_forward_examples": 0}
        require(all(result["cost"][k] == v for k, v in costs.items())
                and learner["buffer"]["num_seen"] == 45000 and learner["step_number"] == len(trace) == 1410, "exposure/update counts")
        require(sha(directory/"allocation.json") == result["allocation_trace_sha256"], "allocation identity")
        reset_steps = list(range(60, 1411, 20)) if method != "er_v3" else []
        reset_events = {e["step"]: e for e in events if e["recycled"]}
        last_reset = np.full(64, -1)
        feature_count = sum(p.numel() for name, p in learner["model"].items() if not name.startswith("head."))
        for step, item in enumerate(trace, 1):
            require(item["step"] == step and item["recycled_by_population"] == {"adapter": 3 if step in reset_steps else 0}, "reset schedule")
            require(math.isfinite(item["update_norm"]) and 0 <= item["update_norm"] <= .1+1e-6, "optimizer cap")
            expected_gain = 1 if step <= 40 else .5
            if step <= 40 or method != "newborn_v3":
                close(item["nominal_feature_gain"], expected_gain, "nominal budget", 2e-6)
            else:
                ages = step-last_reset-1
                active = (last_reset >= 0) & (ages < 30)
                reconstructed = .5 + float((1.5*(1-ages[active]/30)).sum())*65/feature_count
                close(item["nominal_feature_gain"], reconstructed, "reset-age reconstructed newborn gain", 2e-6)
            close(item["effective_feature_gain"], item["nominal_feature_gain"]*item["clip_scale"], "achieved gain", 2e-6)
            close(item["clip_scale"], min(1, .1/item["proposed_update_norm"]), "clip scale", 2e-6)
            if step in reset_steps:
                indices = reset_events[step]["reset_indices"]["adapter"]
                require(len(set(indices)) == 3 and all(0 <= i < 64 and
                        (last_reset[i] < 0 or step-last_reset[i] >= 30) for i in indices), "reset eligibility")
                last_reset[indices] = step
        require([e["step"] for e in events if e["recycled"]] == reset_steps, "event reset schedule")
        require(result["diagnostics"]["total_recycled"] == learner["engine"]["total_recycled"] == 3*len(reset_steps), "reset total")
        require(all(e["total_maturations"] == e["total_local_consolidations"] == e["protected_units"] == 0 for e in events), "disabled intervention activated")
        newborn_peak = max(e["newborn_units"] for e in events)
        require((newborn_peak > 0) == (method in ("newborn_v3", "newborn_matched_v3")), "gain intervention inactive/unexpected")
        key = f"{condition}/{directory.name}"
        rows[key] = {"condition": condition, "method": method, "seed": seed, **endpoints(result, config)}
        signature = {"buffer": tensor_tree_hash(learner["buffer"]), "rng": tensor_tree_hash([
            learner["augmentation_rng"], learner["replay_augmentation_rng"]]),
            "stream": result["stream_fingerprints"], "arrivals": arrivals,
            "warmup": trace[:40], "untrained": rows[key]["untrained_original"]}
        group = f"{condition}/seed{seed}"
        require(group not in pairings or pairings[group] == signature, "within-condition pairing/warmup")
        pairings[group] = signature
        cross_signature = {"ids": observed, "splits": metadata["split_indices"], "warmup": trace[:40],
                           "replay_labels_and_rng": tensor_tree_hash({k: v for k, v in learner["buffer"].items() if k != "x"})}
        require(seed not in cross_condition or cross_condition[seed] == cross_signature, "cross-condition arrival/replay/RNG/warmup pairing")
        cross_condition[seed] = cross_signature
        evidence.append({"run": key, "input_hashes": {name: sha(directory/name) for name in
            ("result.json", "events.json", "allocation.json", "current_arrivals.json", "checkpoint.pt")},
            "learner_state_hash_finite": finite_hash, "updates": len(trace), "reset_units": 3*len(reset_steps),
            "maximum_update_norm": max(t["update_norm"] for t in trace), "clipped_updates": sum(t["clip_scale"] < 1 for t in trace),
            "maximum_logged_newborn_units": newborn_peak, "wall_seconds": result["wall_seconds"],
            "model_parameters": result["model_parameters"], "common_training_cost": costs})
    require(len(source_fingerprints) == 1, "source dataset fingerprint differs")
    manifest_created = {condition: datetime.fromtimestamp((root/condition/"manifest.json").stat().st_ctime,
                        timezone.utc).isoformat() for condition in spec["conditions"]}
    return {"status": "passed", "scope": "local independent arithmetic and artifact checks; no training rerun or external peer review",
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": lock["source_sha256"],
            "runtime_sha256": lock["runtime_sha256"], "source_freeze_commit": lock["source_freeze_commit"],
            "launch_lock_commit": commit, "lock_sha256": sha(locked/"lock.json"),
            "recorded_timing": {"lock_frozen_utc": lock["frozen_at_utc"], "launch_commit_utc": git("show", "-s", "--format=%cI", commit).decode().strip(),
                                "manifest_filesystem_created_utc": manifest_created, "interpretation": "recorded local provenance, not trusted-clock proof"},
            "runs": rows, "decision": decision(rows, spec), "evidence": evidence,
            "total_updates": sum(r["updates"] for r in evidence),
            "summed_run_wall_seconds": sum(r["wall_seconds"] for r in evidence),
            "limitations": ["Two development seeds; no confidence interval or efficacy claim.",
                "Concurrent condition suites share one GPU; summed run wall time is not GPU compute time.",
                "No training rerun or full transformed-stream regeneration; split/arrival metadata and recorded stream hashes checked.",
                "Warmup traces match; historical warmup weights are not separately checkpointed.",
                "Nominal gain equality is not equality of achieved displacement or compute; reset mutations are outside the cap.",
                "Repeated validation domains share base images; 30 experiences are not 30 independent samples."]}


def compare_summary(value, path):
    primary = read(path)
    require(len(primary["runs"]) == 16, "primary summary cohort")
    require({f"{r['condition']}/{r['method']}_seed{r['seed']}" for r in primary["runs"]} == set(value["runs"]), "primary cohort identity")
    domain_names = {"0": "original", "1": "grayscale", "2": "blur"}
    for row in primary["runs"]:
        other = value["runs"][f"{row['condition']}/{row['method']}_seed{row['seed']}"]
        for key in ("final_accuracy", "late_early_auc"):
            close(row[key], other[key], f"primary summary {key}")
        require(len(row["early_auc_256_by_experience"]) == 30, "primary AUC count")
        for first, second in zip(row["early_auc_256_by_experience"], other["early_auc"]):
            close(first, second, "primary per-experience AUC")
        for key, endpoint in other["per_domain_final"].items():
            close(row["final_accuracy_by_domain"][domain_names[key]], endpoint, "primary per-domain final")
        close(row["untrained_original_accuracy"], other["untrained_original"], "primary untrained baseline")
        if row["condition"] == "stationary":
            close(row["stationary_maximum_drawdown"], other["stationary_drawdown"], "primary drawdown")
    decision_ = value["decision"]
    require(primary["decision"]["advance_frozen_recipe"] == decision_["advance"], "primary effort decision")
    for short, full in (("adequacy", "adequacy_passed"), ("signal", "signal_passed"), ("retention", "retention_guard_passed")):
        require(primary["decision"]["passed"][short] == decision_[full], "primary decision subgroup")
    for row in primary["decision"]["paired_recurring_late_auc"]:
        own = decision_["contrasts"][f"recurring/{row['method']}-minus-recycle_v3"]["late_early_auc"]
        close(row["difference"], own["per_seed"][str(row["seed"])], "primary paired contrast")
    value["primary_summary_comparison"] = {"passed": True, "sha256": sha(path),
        "scope": "all 480 AUCs, 16 final/late endpoints, domain endpoints, stationary drawdowns, contrasts and effort decision"}


def write_report(output, value):
    output.mkdir(parents=True, exist_ok=True)
    (output/"independent_audit.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    decision_ = value["decision"]
    lines = ["# Independent CIFAR-10 transfer audit", "", "All 16 raw runs passed the independent artifact and arithmetic checks.", "",
             "This is a local independent agent recomputation, not a training rerun, external peer review, or scientific replication.", "",
             f"Frozen effort rule: **{'advance' if decision_['advance'] else 'do not advance this recipe'}**. "
             f"Adequacy: {decision_['adequacy_passed']}; signal: {decision_['signal_passed']}; retention guard: {decision_['retention_guard_passed']}.", "",
             "| Condition | Method | Seed | Final % | Late AUC % | Stationary drawdown, pp |",
             "|---|---|---:|---:|---:|---:|"]
    for row in value["runs"].values():
        draw = "—" if row["stationary_drawdown"] is None else f"{100*row['stationary_drawdown']:.4f}"
        lines.append(f"| {row['condition']} | {row['method']} | {row['seed']} | {100*row['final_accuracy']:.4f} | {100*row['late_early_auc']:.4f} | {draw} |")
    lines += ["", "Recomputed every first-256-presentation AUC, equal-domain final accuracy, all 30 stationary diagonal drawdowns, and the locked effort criteria. Verified identities, finite learner checkpoints, single-pass arrivals and split separation, replay/RNG and warmup pairing, reset counts, nominal budgets, and all 22,560 optimizer-update caps.", "",
              "All 480 independently integrated AUCs, endpoint/domain values, stationary drawdowns, primary paired contrasts, and the effort decision agree with the primary summary.", "",
              "Limits: " + " ".join(value["limitations"]), "", "[Detailed values, decision checks, and raw artifact hashes](independent_audit.json).", ""]
    (output/"independent_audit.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("runs/cifar10_transfer"))
    parser.add_argument("--locked", type=Path, default=Path("configs/cifar10_transfer"))
    parser.add_argument("--output", type=Path, default=Path("reports/cifar10_transfer"))
    parser.add_argument("--summary", type=Path, default=Path("reports/cifar10_transfer/summary.json"))
    args = parser.parse_args()
    result = audit(args.results, args.locked)
    compare_summary(result, args.summary)
    write_report(args.output, result)
    print(f"Verified {len(result['runs'])} runs; advance={result['decision']['advance']}")
