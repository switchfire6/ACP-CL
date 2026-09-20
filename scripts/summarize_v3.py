"""Audit and archive the complete locked v3 cohort without selecting outcomes."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acp_cl.analysis import analyze
from acp_cl.metrics import normalized_auc, paired_bootstrap

if __package__:
    from .archive_results import archive, reviewable_json
    from .audit_v2_pairing import audit as pairing_audit
    from . import prepare_v3_study as study
else:
    from archive_results import archive, reviewable_json
    from audit_v2_pairing import audit as pairing_audit
    import prepare_v3_study as study


LABELS = {
    "er_v3": "Replay",
    "recycle_v3": "Replay + recycling",
    "newborn_v3": "Newborn gain",
    "newborn_matched_v3": "Newborn, nominal budget matched",
    "protection_v3": "Longer protection",
    "consolidation_v3": "Local consolidation",
    "full_v3": "All local mechanisms",
}


def read(path):
    return study._load(path)[0]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reviewable_json(value) + "\n", encoding="utf-8", newline="\n")


def validate_locked_configs(locked: Path) -> tuple[dict, dict[str, dict]]:
    """Reconstruct every condition from the selection anchor, without writing.

    The supplied selection is the provenance anchor. This validates its internal
    identities and the entire derived cohort/configuration; a git commit or an
    external timestamp is still needed to establish historical locking time.
    """
    selection, selection_hash = study._load(locked / "selection.json")
    if selection.get("status") != "selected" or selection.get("evaluation_outcomes_consumed") is not False:
        raise ValueError("selection was not locked from development data alone")
    spec = selection.get("specification", {})
    development = study._seed_list(spec.get("development_seeds"), "development seeds")
    evaluation = study._seed_list(spec.get("evaluation_seeds"), "evaluation seeds")
    if set(development) & set(evaluation) or selection.get("development_seeds") != development \
            or selection.get("evaluation_seeds") != evaluation:
        raise ValueError("selection seed cohorts differ or overlap")
    digest = selection.get("study_spec_sha256")
    if not isinstance(digest, str) or not study._HASH.fullmatch(digest):
        raise ValueError("invalid study specification hash")
    selected = study._finite(selection.get("selected_feature_gain"), "selected feature gain", 0, 1)
    if selected <= 0 or selected not in spec.get("feature_gains", []):
        raise ValueError("selected gain is not a planned development candidate")
    factor = study._finite(spec.get("newborn_peak_factor"), "newborn peak factor", 1)
    if factor <= 1:
        raise ValueError("newborn peak factor must preserve the gain intervention")
    study._identity(selection.get("common_identity", {}))
    conditions = spec.get("evaluation_conditions", {})
    if set(conditions) != {"recurring", "stationary", "early_biased"}:
        raise ValueError("selection does not contain all three evaluation conditions")
    count = study._integer(spec.get("evaluation_experiences"), "evaluation experiences")
    configurations = {}
    for name in ("recurring", "stationary", "early_biased"):
        condition = conditions[name]
        methods = condition.get("methods", [])
        required = {"newborn_v3", "recycle_v3", "full_v3"}
        if len(methods) != len(set(methods)) or not required <= set(methods) or set(methods) - LABELS.keys():
            raise ValueError(f"invalid planned method cohort: {name}")
        expected_rendering = ("stationary" if name == "stationary" else "recurring",
                              "early_biased" if name == "early_biased" else "independent")
        if (condition.get("regime"), condition.get("color_policy")) != expected_rendering:
            raise ValueError(f"invalid planned rendering condition: {name}")
        config = copy.deepcopy(spec["base_config"])
        config.update(stage=f"v3 locked evaluation: {name}", study_phase="locked_evaluation",
                      study_spec_sha256=digest, methods=list(methods), seeds=list(evaluation),
                      primary_method="newborn_v3")
        config["data"].update(n_experiences=count, regime=condition["regime"],
                              color_policy=condition["color_policy"])
        config["data"].pop("bias_experiences", None)
        if "bias_experiences" in condition:
            config["data"]["bias_experiences"] = condition["bias_experiences"]
        if config.get("eval_split") != "validation" or config["data"].get("test_per_class") != 0 \
                or config.get("scratch_reference") is not False:
            raise ValueError("locked study must use validation only, without scratch fits")
        config["allocation_v3"].update(feature_gain=selected, newborn_peak=factor * selected)
        config["selection_provenance"] = {"artifact": "selection.json", "sha256": selection_hash,
                                           "study_spec_sha256": digest}
        if read(locked / f"{name}.json") != config:
            raise ValueError(f"locked configuration differs from the complete selection specification: {name}")
        configurations[name] = config
    return selection, configurations


def _checkpoint_identity(path, result, count):
    import torch

    # Only local trusted checkpoints produced by this project are supported.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("config_sha256", "source_sha256", "runtime_sha256", "runtime_fingerprint",
                "execution_device", "method", "seed"):
        if saved.get(key) != result[key]:
            raise ValueError(f"checkpoint/result identity mismatch for {key}: {path}")
    if saved.get("completed") != count:
        raise ValueError(f"incomplete checkpoint: {path}")


def _validate_outcomes(result, config):
    count = config["data"]["n_experiences"]
    cadence = config["retention_eval_every_experiences"]
    checkpoints = sorted(set(range(cadence, count + 1, cadence)) | {count})
    study._validate_metrics(result, config, checkpoints)
    curves = result.get("learning_curves")
    if not isinstance(curves, list) or len(curves) != count:
        raise ValueError("missing complete acquisition curves")
    horizon = config["early_examples"]
    for i, curve in enumerate(curves):
        if not curve or curve[-1][0] < horizon:
            raise ValueError("acquisition curve does not reach the prespecified horizon")
        value = normalized_auc(curve, horizon / curve[-1][0])
        if not math.isclose(value, result["early_auc"][i], rel_tol=0, abs_tol=1e-10):
            raise ValueError("acquisition AUC does not match its learning curve")
        if curve[-1][1] != result["accuracy_matrix"][i][i]:
            raise ValueError("acquisition curve and matrix diagonal differ")
    matrix = result["accuracy_matrix"]
    forgetting = np.mean([max(matrix[i][j] for i in range(j, count) if matrix[i][j] is not None)
                          - matrix[-1][j] for j in range(count - 1)]) if count > 1 else 0.0
    if not math.isclose(forgetting, result["metrics"]["forgetting"], rel_tol=0, abs_tol=1e-10):
        raise ValueError("forgetting does not match observed matrix checkpoints")


def _activation_diagnostics(result, events):
    diagnostics = result["diagnostics"]
    counts = {key: study._integer(diagnostics.get(key), key, minimum=0)
              for key in ("total_maturations", "total_local_consolidations")}
    if counts["total_local_consolidations"] > counts["total_maturations"]:
        raise ValueError("local consolidations exceed maturations")
    activation = {**counts, "final_mean_consolidation": study._finite(
        diagnostics.get("mean_consolidation"), "mean consolidation", 0, 1)}
    for key in ("newborn_units", "protected_units", "consolidation_gain_reduction"):
        values = [study._finite(event.get(key), key, 0) for event in events]
        activation[f"maximum_logged_{key}"] = max(values)
    activation["maximum_logged_mean_gain_increment_before_consolidation"] = max(0.0, max(
        study._finite(event["pre_consolidation_feature_gain"], "pre-consolidation gain", 0)
        - study._finite(event["scheduled_feature_gain"], "scheduled gain", 0) for event in events))
    activation["event_sampling_scope"] = "monitor/reset events; maxima need not cover unlogged updates"
    return activation


def fixed_set_drawdown(result):
    """Largest observed decline on exactly the first experience's held-out set."""
    interval = result["config"].get("retention_eval_every_experiences", 1)
    values = [row[0] for i, row in enumerate(result["accuracy_matrix"])
              if (i + 1) % interval == 0 or i == len(result["accuracy_matrix"]) - 1]
    if any(v is None or not np.isfinite(v) for v in values):
        raise ValueError("missing/nonfinite fixed-set retention measurement")
    return max(float(previous - current)
               for previous, current in zip(np.maximum.accumulate(values), values))


def audit_suite(directory, config):
    manifest = read(directory / "manifest.json")
    if manifest["config"] != config or manifest["methods"] != config["methods"] \
            or manifest["seeds"] != config["seeds"]:
        raise ValueError(f"locked configuration/method/seed mismatch: {directory}")
    if manifest.get("config_sha256") != study.config_hash(config):
        raise ValueError(f"manifest configuration/hash mismatch: {directory}")
    identity = study._identity(manifest)
    expected = {(m, s) for m in config["methods"] for s in config["seeds"]}
    actual, rows, checks = set(), {}, []
    total_steps = config["steps_per_experience"] * config["data"]["n_experiences"]
    allocation = config["allocation_v3"]
    warmup, interval = allocation["warmup_steps"], allocation["reset_interval"]
    for path in sorted(directory.glob("*/result.json")):
        result = read(path)
        method, seed = result["method"], result["seed"]
        if (method, seed) not in expected or (method, seed) in actual:
            raise ValueError(f"unexpected/duplicate result: {path}")
        actual.add((method, seed))
        if result["config"] != config or result["information_access"] != "training stream only":
            raise ValueError(f"configuration/information-access mismatch: {path}")
        if study._identity(result) != identity:
            raise ValueError(f"source/runtime/device identity mismatch: {path}")
        for key in ("config_sha256", "source_sha256", "runtime_sha256", "runtime_fingerprint", "execution_device"):
            if result[key] != manifest[key]:
                raise ValueError(f"identity mismatch for {key}: {path}")
        if result.get("eval_split") != "validation" or result.get("class_order") != [0, 1, 2, 3]:
            raise ValueError(f"evaluation split or shared classifier labels differ: {path}")
        _validate_outcomes(result, config)
        _checkpoint_identity(path.parent / "checkpoint.pt", result, config["data"]["n_experiences"])
        file = path.parent / "allocation.json"
        if hashlib.sha256(file.read_bytes()).hexdigest() != result["allocation_trace_sha256"]:
            raise ValueError(f"allocation hash mismatch: {path}")
        trace = read(file)["trace"]
        if len(trace) != total_steps:
            raise ValueError(f"incomplete trace: {path}")
        scheduled_units = 0
        for i, event in enumerate(trace, start=1):
            if event["step"] != i:
                raise ValueError(f"nonsequential trace: {path}")
            expected_count = allocation["reset_count"] if method != "er_v3" and i > warmup \
                and (i - warmup) % interval == 0 else 0
            if event["recycled_by_population"] != {"adapter": expected_count}:
                raise ValueError(f"scheduled reset count differs at {path}, update {i}")
            scheduled_units += sum(event["recycled_by_population"].values())
            for key in ("nominal_feature_gain", "effective_feature_gain", "clip_scale", "update_norm",
                        "proposed_update_norm", "feature_data_displacement", "head_data_displacement"):
                study._finite(event.get(key), f"allocation {key}", 0)
            if not 0 <= event["update_norm"] <= allocation["max_update_norm"] + 1e-6:
                raise ValueError(f"optimizer update cap exceeded: {path}, update {i}")
            if not 0 < event["clip_scale"] <= 1:
                raise ValueError(f"invalid clip scale: {path}, update {i}")
            expected_gain = 1.0 if i <= warmup else allocation["feature_gain"]
            if i <= warmup or method in ("er_v3", "recycle_v3", "newborn_matched_v3", "protection_v3"):
                if abs(event["nominal_feature_gain"] - expected_gain) > 2e-6:
                    raise ValueError(f"nominal feature budget mismatch: {path}, update {i}")
            if abs(event["effective_feature_gain"] -
                   event["nominal_feature_gain"] * event["clip_scale"]) > 2e-6:
                raise ValueError(f"post-clip gain mismatch: {path}, update {i}")
        if result["diagnostics"]["total_recycled"] != scheduled_units:
            raise ValueError(f"cumulative reset count differs: {path}")
        post_warmup = trace[warmup:]
        if not post_warmup:
            raise ValueError(f"no post-warmup allocation in complete study: {path}")
        computed_allocation = {
            "steps": total_steps,
            "mean_feature_gain": float(np.mean([e["effective_feature_gain"] for e in trace])),
            "mean_nominal_feature_gain": float(np.mean([e["nominal_feature_gain"] for e in trace])),
            "post_warmup_nominal_feature_gain": float(np.mean([e["nominal_feature_gain"] for e in post_warmup])),
            "post_warmup_effective_feature_gain": float(np.mean([e["effective_feature_gain"] for e in post_warmup])),
            "clipped_updates": sum(e["clip_scale"] < 1 for e in trace),
            "maximum_proposed_update_norm": max(e["proposed_update_norm"] for e in trace),
            "maximum_applied_update_norm": max(e["update_norm"] for e in trace),
            "summed_head_data_displacement": sum(e["head_data_displacement"] for e in trace),
            "summed_feature_data_displacement": sum(e["feature_data_displacement"] for e in trace),
        }
        for key, value in computed_allocation.items():
            stored = study._finite(result["allocation_summary"].get(key), f"allocation summary {key}", 0)
            if not math.isclose(stored, value, rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError(f"allocation summary differs from trace for {key}: {path}")
        events_file = path.parent / "events.json"
        events = read(events_file).get("events", [])
        if not events or events[-1].get("step") != total_steps or any(
            e.get("phase") != "scheduled" or e.get("controller") for e in events
        ):
            raise ValueError(f"incomplete events or unexpected controller access: {path}")
        for key in ("total_maturations", "total_local_consolidations"):
            if events[-1].get(key) != result["diagnostics"].get(key):
                raise ValueError(f"final maturation diagnostic differs from events: {path}")
        result["mechanism_activation"] = _activation_diagnostics(result, events)
        result["fixed_first_set_drawdown"] = fixed_set_drawdown(result)
        rows.setdefault(method, {})[seed] = result
        checks.append({"method": method, "seed": seed, "steps": total_steps,
                       "unit_resets": scheduled_units, "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "events_sha256": hashlib.sha256(events_file.read_bytes()).hexdigest(),
                       "mechanism_activation": result["mechanism_activation"],
                       "fixed_first_set_drawdown": result["fixed_first_set_drawdown"]})
    if actual != expected:
        raise ValueError(f"incomplete suite: {directory}; missing {sorted(expected - actual)}")
    for seed in config["seeds"]:
        if len({json.dumps(rows[m][seed]["stream_fingerprints"], sort_keys=True)
                for m in config["methods"]}) != 1:
            raise ValueError(f"actual stream tensors differ between methods: seed {seed}")
        warmup_traces = [read(directory / f"{m}_seed{seed}" / "allocation.json")["trace"][:warmup]
                         for m in config["methods"]]
        if any(t != warmup_traces[0] for t in warmup_traces[1:]):
            raise ValueError(f"warmup allocation/displacement differs between methods: seed {seed}")
    return rows, {"checks_passed": True, "runs": checks,
                  "scope": "complete outcomes, runtime/checkpoint identities, actual tensor hashes, warmup traces, causal reset schedule, nominal budgets and optimizer cap",
                  "pairing": pairing_audit(directory)}


def summarize(results_root: Path, locked: Path, output: Path):
    cohorts, audits = {}, {}
    selection, configurations = validate_locked_configs(locked)
    # Validate the entire cohort before any analysis/report writer is invoked.
    for condition, config in configurations.items():
        directory = results_root / condition
        manifest = read(directory / "manifest.json")
        if any(manifest[key] != value for key, value in selection["common_identity"].items()):
            raise ValueError("source/runtime/device changed between development and evaluation")
        cohorts[condition], audits[condition] = audit_suite(directory, config)
    seeds = sorted(cohorts["recurring"]["recycle_v3"])
    evaluation_pairing = []
    for seed in seeds:
        independent = cohorts["recurring"]["recycle_v3"][seed]
        biased = cohorts["early_biased"]["recycle_v3"][seed]
        if independent["stream_fingerprints"]["validation"] != biased["stream_fingerprints"]["validation"]:
            raise ValueError("early-bias evaluation tensors are not exactly paired")
        evaluation_pairing.append({"seed": seed, "validation_sha256": independent["stream_fingerprints"]["validation"]})
    contrasts, rows, per_seed, activations = [], [], [], []
    for condition, methods in cohorts.items():
        for method, runs in methods.items():
            items = list(runs.values())
            row = {"condition": condition, "method": method, "n_seeds": len(items)}
            row.update({key: float(np.mean([r["metrics"][key] for r in items]))
                        for key in ("final_accuracy", "late_early_auc", "forgetting")})
            row.update({key: float(np.mean([r["allocation_summary"][key] for r in items]))
                        for key in ("post_warmup_nominal_feature_gain", "post_warmup_effective_feature_gain",
                                    "clipped_updates",
                                    "summed_head_data_displacement", "summed_feature_data_displacement")})
            row["maximum_applied_update_norm"] = max(r["allocation_summary"]["maximum_applied_update_norm"]
                                                     for r in items)
            row["maximum_fixed_set_drawdown"] = max(r["fixed_first_set_drawdown"] for r in items)
            row["mean_unit_resets"] = float(np.mean([r["diagnostics"]["total_recycled"] for r in items]))
            for key in ("total_maturations", "total_local_consolidations", "final_mean_consolidation"):
                row[f"mean_{key}"] = float(np.mean([r["mechanism_activation"][key] for r in items]))
            rows.append(row)
            for seed in seeds:
                result = runs[seed]
                per_seed.append({"condition": condition, "method": method, "seed": seed,
                                 "final_accuracy": result["metrics"]["final_accuracy"],
                                 "late_early_auc": result["metrics"]["late_early_auc"],
                                 "forgetting": result["metrics"]["forgetting"],
                                 "fixed_first_set_drawdown": result["fixed_first_set_drawdown"]})
                activations.append({"condition": condition, "method": method, "seed": seed,
                                    **result["mechanism_activation"]})
            if method != "recycle_v3":
                for endpoint in ("final_accuracy", "late_early_auc"):
                    comparator = methods["recycle_v3"]
                    values = [runs[s]["metrics"][endpoint] for s in seeds]
                    reference = [comparator[s]["metrics"][endpoint] for s in seeds]
                    contrasts.append({"condition": condition, "method": method,
                                      "comparator": "recycle_v3", "endpoint": endpoint,
                                      "seeds": seeds, "method_values": values, "comparator_values": reference,
                                      "per_seed_differences": [a-b for a, b in zip(values, reference)],
                                      **paired_bootstrap(values, reference, seed=37119)})
    interactions = []
    for endpoint in ("final_accuracy", "late_early_auc"):
        differences = []
        for seed in seeds:
            effect = {}
            for condition in ("recurring", "early_biased"):
                methods = cohorts[condition]
                effect[condition] = methods["full_v3"][seed]["metrics"][endpoint] - \
                    methods["recycle_v3"][seed]["metrics"][endpoint]
            differences.append(effect["early_biased"] - effect["recurring"])
        interactions.append({"endpoint": endpoint, "per_seed_differences": differences,
                             "seeds": seeds, **paired_bootstrap(differences, [0] * len(seeds), seed=37119)})
    primary = next(c for c in contrasts if c["condition"] == "recurring" and c["method"] == "newborn_v3"
                   and c["endpoint"] == "late_early_auc")
    summary = {"status": "locked new-seed exploratory evaluation; not confirmatory",
               "runs": sum(len(a["runs"]) for a in audits.values()), "methods": rows,
               "primary_contrast": primary, "per_seed_endpoints": per_seed,
               "mechanism_activation": activations,
               "interval_convention": {"method": "unadjusted paired percentile bootstrap",
                                       "resampling_unit": "whole seed pair, or whole seed interaction",
                                       "confidence": 0.95, "resamples": 10000,
                                       "limitation": "three-seed exploratory intervals are unstable; no multiplicity adjustment"},
               "paired_against_recycling": contrasts, "early_bias_full_interaction": interactions,
               "exact_evaluation_pairing": evaluation_pairing,
               "selection": selection}
    for condition in configurations:
        directory = results_root / condition
        analyze(directory)
        archive(directory, output / condition)
        write(output / condition / "allocation_checks.json", audits[condition])
    write(output / "diagnostics.json", summary)
    lines = ["# V3 locked evaluation", "", "Three new seeds; no parameter selection on this cohort.", "",
             "Primary contrast: newborn gain minus replay + recycling on recurring late acquisition AUC.",
             f"Mean difference: {100*primary['mean_difference']:+.2f} pp "
             f"[95% unadjusted paired percentile bootstrap: {100*primary['ci_low']:+.2f}, "
             f"{100*primary['ci_high']:+.2f}]. Entire seed pairs are resampled; these three-seed intervals are unstable.",
             "Per-seed primary differences: " + ", ".join(
                 f"{seed}: {100*value:+.2f} pp" for seed, value in zip(seeds, primary["per_seed_differences"])) + ".", "",
             "| Condition | Method | Final % | Late AUC % | Max observed fixed-set drawdown, pp |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['condition']} | {LABELS[row['method']]} | {100*row['final_accuracy']:.2f} | "
                     f"{100*row['late_early_auc']:.2f} | {100*row['maximum_fixed_set_drawdown']:.2f} |")
    lines += ["", "Maturation and positive local consolidation counts (mean per run):", "",
              "| Condition | Method | Maturations | Local consolidations |",
              "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['condition']} | {LABELS[row['method']]} | "
                     f"{row['mean_total_maturations']:.1f} | {row['mean_total_local_consolidations']:.1f} |")
    lines += ["", "Complete per-seed endpoints, gain/protection activation diagnostics, and the paired early-bias interaction "
              "are in [diagnostics.json](diagnostics.json). Activation does not establish a beneficial causal effect.", "",
              "The nominal budget match precedes clipping. Every recycling condition has the same scheduled counts/times,",
              "but selects its own units. The optimizer cap excludes structural resets and is not a prediction-stability guarantee.",
              "Fixed-set drawdown is measured sparsely and can miss intermediate failures.", "",
              "![Acquisition trajectories](trajectories.png)", ""]
    (output / "diagnostics.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    plot(cohorts, output)
    return summary


def plot(cohorts, output):
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained", sharey=True)
    colors = dict(zip(LABELS, plt.get_cmap("tab10").colors))
    for axis, (condition, methods) in zip(axes, cohorts.items()):
        for method, runs in methods.items():
            values = np.array([r["early_auc"] for r in runs.values()]) * 100
            # Disjoint 5-experience means preserve all observations, without a
            # centered moving average using future values in the plotted mean.
            values = values.reshape(len(runs), -1, 5).mean(-1)
            x = np.arange(5, values.shape[1] * 5 + 1, 5)
            axis.plot(x, values.mean(0), color=colors[method], label=LABELS[method], linewidth=1.6)
            axis.fill_between(x, values.min(0), values.max(0), color=colors[method], alpha=0.07)
        axis.set(title=condition.replace("_", " ").capitalize(), xlabel="Experience (5-experience bins)", ylim=(15, 102))
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Early acquisition AUC (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    figure.suptitle("V3: locked new-seed trajectories; shading is the range across three seeds")
    figure.savefig(output / "trajectories.png", dpi=170)
    figure.savefig(output / "trajectories.svg")
    path = output / "trajectories.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("runs/v3_evaluation"))
    parser.add_argument("--locked", type=Path, default=Path("configs/v3/locked"))
    parser.add_argument("--output", type=Path, default=Path("reports/v3"))
    args = parser.parse_args()
    result = summarize(args.results, args.locked, args.output)
    print(f"Audited and archived {result['runs']} locked evaluation runs")
