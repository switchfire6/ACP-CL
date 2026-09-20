"""Audit and archive the complete locked v3 cohort without selecting outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acp_cl.analysis import analyze
from acp_cl.metrics import paired_bootstrap
from archive_results import archive, reviewable_json
from audit_v2_pairing import audit as pairing_audit


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
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reviewable_json(value) + "\n", encoding="utf-8", newline="\n")


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
        for key in ("config_sha256", "source_sha256", "runtime_sha256", "runtime_fingerprint"):
            if result[key] != manifest[key]:
                raise ValueError(f"identity mismatch for {key}: {path}")
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
            if not event["recycled_by_population"] or \
                    any(n != expected_count for n in event["recycled_by_population"].values()):
                raise ValueError(f"scheduled reset count differs at {path}, update {i}")
            scheduled_units += sum(event["recycled_by_population"].values())
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
        result["fixed_first_set_drawdown"] = fixed_set_drawdown(result)
        rows.setdefault(method, {})[seed] = result
        checks.append({"method": method, "seed": seed, "steps": total_steps,
                       "unit_resets": scheduled_units, "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
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
                  "scope": "actual tensor hashes, warmup traces, causal reset schedule, nominal budgets and optimizer cap",
                  "pairing": pairing_audit(directory)}


def summarize(results_root: Path, locked: Path, output: Path):
    cohorts, audits = {}, {}
    selection = read(locked / "selection.json")
    selection_hash = hashlib.sha256((locked / "selection.json").read_bytes()).hexdigest()
    if selection["status"] != "selected" or selection["evaluation_outcomes_consumed"]:
        raise ValueError("selection was not locked from development data alone")
    for condition in ("recurring", "stationary", "early_biased"):
        config = read(locked / f"{condition}.json")
        if config["selection_provenance"]["sha256"] != selection_hash \
                or config["allocation_v3"]["feature_gain"] != selection["selected_feature_gain"]:
            raise ValueError("locked selection artifact/gain mismatch")
        directory = results_root / condition
        manifest = read(directory / "manifest.json")
        if any(manifest[key] != value for key, value in selection["common_identity"].items()):
            raise ValueError("source/runtime/device changed between development and evaluation")
        cohorts[condition], audits[condition] = audit_suite(directory, config)
        analyze(directory)
        archive(directory, output / condition)
        write(output / condition / "allocation_checks.json", audits[condition])
    seeds = sorted(cohorts["recurring"]["recycle_v3"])
    evaluation_pairing = []
    for seed in seeds:
        independent = cohorts["recurring"]["recycle_v3"][seed]
        biased = cohorts["early_biased"]["recycle_v3"][seed]
        if independent["stream_fingerprints"]["validation"] != biased["stream_fingerprints"]["validation"]:
            raise ValueError("early-bias evaluation tensors are not exactly paired")
        evaluation_pairing.append({"seed": seed, "validation_sha256": independent["stream_fingerprints"]["validation"]})
    contrasts, rows = [], []
    for condition, methods in cohorts.items():
        for method, runs in methods.items():
            items = list(runs.values())
            row = {"condition": condition, "method": method, "n_seeds": len(items)}
            row.update({key: float(np.mean([r["metrics"][key] for r in items]))
                        for key in ("final_accuracy", "late_early_auc", "forgetting")})
            row.update({key: float(np.mean([r["allocation_summary"][key] for r in items]))
                        for key in ("post_warmup_nominal_feature_gain", "post_warmup_effective_feature_gain",
                                    "clipped_updates", "maximum_applied_update_norm",
                                    "summed_head_data_displacement", "summed_feature_data_displacement")})
            row["maximum_fixed_set_drawdown"] = max(r["fixed_first_set_drawdown"] for r in items)
            row["mean_unit_resets"] = float(np.mean([r["diagnostics"]["total_recycled"] for r in items]))
            rows.append(row)
            if method != "recycle_v3":
                for endpoint in ("final_accuracy", "late_early_auc"):
                    comparator = methods["recycle_v3"]
                    contrasts.append({"condition": condition, "method": method,
                                      "comparator": "recycle_v3", "endpoint": endpoint,
                                      **paired_bootstrap([runs[s]["metrics"][endpoint] for s in seeds],
                                                         [comparator[s]["metrics"][endpoint] for s in seeds], seed=37119)})
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
    summary = {"status": "locked new-seed exploratory evaluation; not confirmatory",
               "runs": sum(len(a["runs"]) for a in audits.values()), "methods": rows,
               "paired_against_recycling": contrasts, "early_bias_full_interaction": interactions,
               "exact_evaluation_pairing": evaluation_pairing,
               "selection": selection}
    write(output / "diagnostics.json", summary)
    lines = ["# V3 locked evaluation", "", "Three new seeds; no parameter selection on this cohort.", "",
             "| Condition | Method | Final % | Late AUC % | Max observed fixed-set drawdown, pp |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['condition']} | {LABELS[row['method']]} | {100*row['final_accuracy']:.2f} | "
                     f"{100*row['late_early_auc']:.2f} | {100*row['maximum_fixed_set_drawdown']:.2f} |")
    lines += ["", "The nominal budget match precedes clipping. Every recycling condition has the same scheduled counts/times,",
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
