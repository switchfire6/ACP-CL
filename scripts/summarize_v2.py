"""Report completed V2 suites without importing or changing training source.

Usage:
    python scripts/summarize_v2.py runs/v2_gaussian runs/v2_shapes_long --output reports/v2

Each input must contain every method/seed declared in its manifest. Outputs go
under OUTPUT/SUITE_NAME/. The script never polls, resumes, or modifies a run.
Statistics are descriptive; methods retain manifest order rather than ranking.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


MAIN_METHODS = (
    "er_recycle", "acp_v2", "acp_v2_no_newborn", "acp_v2_no_reopening", "acp_v2_oracle",
    "er_recycle_yoked_gain",
)
LABELS = {
    "er_recycle": "ER + recycle",
    "acp_v2": "ACP-v2",
    "acp_v2_no_newborn": "No newborn window",
    "acp_v2_no_reopening": "No reopening",
    "acp_v2_oracle": "Oracle \u2020",
    "er_recycle_yoked_gain": "Matched reset/gain \u2020",
}
METRICS = (
    "final_accuracy", "late_early_auc", "late_plasticity_gap", "resets",
    "mean_feature_gain", "summed_feature_data_displacement", "local_maturations",
    "local_consolidations", "mean_consolidation", "reopening_events",
    "sensor_state_bytes", "input_centroid_accuracy", "detector_change_window_fraction",
    "detector_outside_window_fraction", "detector_mean_delay_steps",
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require_number(value, label: str) -> float:
    if not finite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def mean(values):
    return statistics.fmean(values) if values else None


def time_series(trace: list[dict], max_bins: int = 100) -> list[dict]:
    """Plot bins preserve all steps; cumulative reset endpoints are exact."""
    count = len(trace)
    bins = min(max_bins, count)
    output, cumulative = [], 0
    for index in range(bins):
        start, end = index * count // bins, (index + 1) * count // bins
        records = trace[start:end]
        resets = sum(sum(row["recycled_by_population"].values()) for row in records)
        cumulative += resets
        output.append({
            "first_step": records[0]["step"], "last_step": records[-1]["step"],
            "mean_feature_gain": mean([row["effective_feature_gain"] for row in records]),
            "resets": resets, "cumulative_resets": cumulative,
            "summed_feature_data_displacement": math.fsum(row["feature_data_displacement"] for row in records),
        })
    return output


def summarize_run(directory: Path, manifest: dict, method: str, seed: int) -> dict:
    result_path = directory / "result.json"
    events_path = directory / "events.json"
    allocation_path = directory / "allocation.json"
    result = load_json(result_path)
    if result.get("method") != method or result.get("seed") != seed:
        raise ValueError(f"run identity disagrees with directory/manifest: {directory}")
    for key in ("config_sha256", "source_sha256", "execution_device"):
        if result.get(key) != manifest.get(key):
            raise ValueError(f"{key} differs from suite manifest: {directory}")
    actual_hash = sha256(allocation_path)
    if actual_hash != result.get("allocation_trace_sha256"):
        raise ValueError(f"allocation trace hash mismatch: {directory}")
    events = load_json(events_path)["events"]
    trace = load_json(allocation_path)["trace"]
    if not trace or any(row.get("step") != index for index, row in enumerate(trace, start=1)):
        raise ValueError(f"allocation must contain every optimizer step exactly once: {directory}")
    allocation = result["allocation_summary"]
    if allocation.get("steps") != len(trace):
        raise ValueError(f"allocation step count disagrees with completed result: {directory}")
    n_experiences = manifest["config"]["data"]["n_experiences"]
    early_auc = result["early_auc"]
    if len(early_auc) != n_experiences:
        raise ValueError(f"result does not cover all {n_experiences} experiences: {directory}")
    configured_steps = manifest["config"].get("steps_per_experience")
    if configured_steps is not None and len(trace) != configured_steps * n_experiences:
        raise ValueError(f"trace does not cover the full configured training horizon: {directory}")
    if any(not finite(value) for value in early_auc):
        raise ValueError(f"invalid experience AUC: {directory}")
    populations = Counter()
    for row in trace:
        gain = require_number(row.get("effective_feature_gain"), f"{directory}: feature gain")
        displacement = require_number(row.get("feature_data_displacement"), f"{directory}: displacement")
        if gain < -1e-6 or gain > 1 + 1e-6 or displacement < 0:
            raise ValueError(f"invalid allocation value: {directory}, step {row['step']}")
        counts = row.get("recycled_by_population")
        if not isinstance(counts, dict) or any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError(f"invalid replacement counts: {directory}, step {row['step']}")
        populations.update(counts)
    gains = [row["effective_feature_gain"] for row in trace]
    displacement = math.fsum(row["feature_data_displacement"] for row in trace)
    mean_gain = statistics.fmean(gains)
    for name, actual in (("mean_feature_gain", mean_gain), ("summed_feature_data_displacement", displacement)):
        if not math.isclose(actual, allocation[name], rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError(f"{name} disagrees between allocation and result: {directory}")
    diagnostics = result["diagnostics"]
    resets = sum(populations.values())
    if resets != diagnostics["total_recycled"]:
        raise ValueError(f"replacement totals disagree between allocation and result: {directory}")
    reopenings = [
        row["step"] for row in events
        if row.get("controller", {}).get("previous_phase") == "adult"
        and row.get("controller", {}).get("phase") == "reopened"
    ]
    audit = result["detector_audit"]
    if reopenings != audit["reopening_steps"] or len(reopenings) != result["reopening_events"]:
        raise ValueError(f"reopening counts disagree between events and result: {directory}")
    event_resets = sum(row.get("recycled", 0) for row in events)
    if event_resets != resets:
        raise ValueError(f"replacement totals disagree between events and allocation: {directory}")
    access = result.get("information_access", "training stream only")
    extra_information = access != "training stream only" or "oracle" in method or "yoked" in method
    metrics = result["metrics"]
    delays = audit["delays_within_window"]
    output = {
        "method": method, "seed": seed, "information_access": access,
        "extra_information_diagnostic": extra_information,
        "final_accuracy": metrics["final_accuracy"],
        "late_early_auc": metrics["late_early_auc"],
        "late_plasticity_gap": metrics.get("late_plasticity_gap"),
        "resets": resets, "resets_by_population": dict(populations),
        "mean_feature_gain": mean_gain, "summed_feature_data_displacement": displacement,
        "local_maturations": diagnostics["total_maturations"],
        "local_consolidations": diagnostics["total_local_consolidations"],
        "mean_consolidation": diagnostics["mean_consolidation"],
        "reopening_events": len(reopenings),
        "sensor_state_bytes": result["sensor_state_bytes"],
        "input_centroid_accuracy": result.get("input_centroid_accuracy"),
        "detector_change_window_fraction": ratio(audit["changes_with_reopening_in_window"], audit["post_warmup_changes"]),
        "detector_outside_window_fraction": ratio(audit["reopenings_outside_change_windows"], len(reopenings)),
        "detector_mean_delay_steps": mean(delays),
        "detector_audit": audit,
        "optimizer_steps": len(trace), "event_records": len(events),
        "experience_early_auc": early_auc,
        "experience_scratch_gap": result.get("plasticity_gap"),
        "scratch_early_auc": result.get("scratch_early_auc"),
        "allocation_series": time_series(trace),
        "final_diagnostics": diagnostics,
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in (("result", result_path), ("events", events_path), ("allocation", allocation_path))
        },
    }
    for metric in METRICS:
        if output[metric] is not None:
            require_number(output[metric], f"{directory}: {metric}")
    return output


def summarize_suite(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"suite manifest does not exist: {manifest_path}")
    manifest = load_json(manifest_path)
    methods, seeds = manifest["methods"], manifest["seeds"]
    if not methods or not seeds or len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError(f"manifest requires nonempty unique methods and seeds: {directory}")
    expected = [(method, seed, directory / f"{method}_seed{seed}") for method in methods for seed in seeds]
    missing = []
    for method, seed, run_dir in expected:
        absent = [name for name in ("result.json", "events.json", "allocation.json") if not (run_dir / name).is_file()]
        if absent:
            missing.append({"method": method, "seed": seed, "files": absent})
    if missing:
        preview = "; ".join(f"{row['method']} seed={row['seed']} ({', '.join(row['files'])})" for row in missing[:8])
        raise ValueError(
            f"suite is incomplete: {directory}; {len(expected)-len(missing)}/{len(expected)} completed runs; "
            f"{len(missing)} missing runs. {preview}. Run this utility after the suite completes.")
    expected_paths = {run_dir.resolve() for _, _, run_dir in expected}
    unexpected = [path.parent for path in directory.glob("*/result.json") if path.parent.resolve() not in expected_paths]
    if unexpected:
        raise ValueError(f"completed runs outside manifest would be omitted: {unexpected}")
    runs = [summarize_run(run_dir, manifest, method, seed) for method, seed, run_dir in expected]
    summaries = []
    for method in methods:
        subset = [run for run in runs if run["method"] == method]
        if len({run["information_access"] for run in subset}) != 1:
            raise ValueError(f"information access varies across seeds for {method}")
        row = {
            "method": method, "n_seeds": len(subset), "seeds": [run["seed"] for run in subset],
            "information_access": subset[0]["information_access"],
            "extra_information_diagnostic": subset[0]["extra_information_diagnostic"],
        }
        for metric in METRICS:
            values = [run[metric] for run in subset if run[metric] is not None]
            row[metric] = mean(values)
            row[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
            row[metric + "_n"] = len(values)
        pooled_delays = [delay for run in subset for delay in run["detector_audit"]["delays_within_window"]]
        row["detector_pooled_counts"] = {
            "post_warmup_changes": sum(run["detector_audit"]["post_warmup_changes"] for run in subset),
            "changes_with_reopening_in_window": sum(run["detector_audit"]["changes_with_reopening_in_window"] for run in subset),
            "reopening_events": sum(run["reopening_events"] for run in subset),
            "reopenings_outside_change_windows": sum(run["detector_audit"]["reopenings_outside_change_windows"] for run in subset),
            "delays_within_window": pooled_delays,
            "mean_delay_steps": mean(pooled_delays),
        }
        summaries.append(row)
    return {
        "schema_version": 1, "suite": directory.name, "source_directory": str(directory.resolve()),
        "status": "complete; descriptive post-run diagnostics; no inferential conclusions",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "summary_script_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(manifest_path),
        "config_sha256": manifest["config_sha256"], "source_sha256": manifest["source_sha256"],
        "execution_device": manifest.get("execution_device"), "config": manifest["config"],
        "eval_split": manifest["config"].get("eval_split", "validation"),
        "completion": {"expected_runs": len(expected), "completed_runs": len(runs),
                       "method_count": len(methods), "seed_count": len(seeds), "seeds": seeds},
        "definitions": {
            "aggregation": "Equal-weight mean over completed seeds; sample SD when n>=2; each metric records its own n.",
            "late_early_auc": "Mean early acquisition AUC over experience indices n//2 through n-1.",
            "late_plasticity_gap": "Same late AUC minus same-experience scratch AUC; a diagnostic, not pure forward transfer.",
            "mean_feature_gain": "Mean over every optimizer step of the parameter-weighted non-head final data-update multiplier.",
            "summed_feature_data_displacement": "Sum of per-step non-head data-update L2 norms; path length, not net parameter displacement. Excludes anchors and decay.",
            "local_maturations": "Post-reset units that completed their local age requirement, including units with zero SI usefulness.",
            "local_consolidations": "Matured reset units with positive SI usefulness acquired locally; counted separately from all maturations.",
            "mean_consolidation": "Final within-run row-level mean consolidation, then mean across seeds; not a time average.",
            "detector_proximity": "Descriptive boundary proximity using recorded tolerance and eligible change counts. Not calibrated precision/recall.",
            "sensor_state_bytes": "Reported frozen-sensor tensor state payload bytes; zero where no separate frozen sensor is allocated.",
            "input_centroid_accuracy": "Evaluator-only raw-input centroid classifier fitted to the final replay reservoir; never fed to learner/controller.",
            "allocation_series": "At most 100 consecutive optimizer-step bins; gain is a bin mean and cumulative reset endpoints are exact.",
            "extra_information": "Oracle methods receive true changes; yoked methods receive offline ACP-v2 traces. These are diagnostics with extra information.",
        },
        "methods": summaries, "runs": runs,
    }


def fmt(value, digits=3, percent=False, signed=False):
    if value is None:
        return "n/a"
    value = value * 100 if percent else value
    return format(value, ("+" if signed else "") + f".{digits}f")


def markdown(report: dict, plotted: bool) -> str:
    completion = report["completion"]
    lines = [
        f"# {report['suite']}: completed-run diagnostics", "",
        f"Completed **{completion['completed_runs']}/{completion['expected_runs']}** runs: "
        f"{completion['method_count']} methods x {completion['seed_count']} seeds "
        f"({', '.join(map(str, completion['seeds']))}). Evaluation split: **{report['eval_split']}**.", "",
        "All table summaries are equal-weight seed means. Method order follows the manifest. "
        "Per-seed values, sample SDs, metric counts, source hashes, and full detector audits are in `diagnostics.json`.", "",
        "A dagger (\u2020) marks an **extra-information diagnostic**: oracle change times or offline allocation traces.", "",
        "| Method | Seeds | Final accuracy (%) | Late early AUC (%) | Scratch gap (pp) | Input-centroid accuracy (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["methods"]:
        label = row["method"] + (" \u2020" if row["extra_information_diagnostic"] else "")
        lines.append(f"| {label} | {row['n_seeds']} | {fmt(row['final_accuracy'], 2, True)} | "
                     f"{fmt(row['late_early_auc'], 2, True)} | {fmt(row['late_plasticity_gap'], 2, True, True)} | "
                     f"{fmt(row['input_centroid_accuracy'], 2, True)} |")
    lines.extend([
        "", "The input-centroid column is an evaluator-only classifier using the final replay reservoir. "
        "Scratch gap subtracts a fresh model's acquisition AUC on the identical experience.", "",
        "| Method | Unit resets | Mean gain | Sum data-update L2 | Local maturations | Useful local consolidations | Final mean c | Reopenings | Sensor bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["methods"]:
        label = row["method"] + (" \u2020" if row["extra_information_diagnostic"] else "")
        values = [fmt(row[key], digits) for key, digits in (
            ("resets", 1), ("mean_feature_gain", 4), ("summed_feature_data_displacement", 3),
            ("local_maturations", 1), ("local_consolidations", 1), ("mean_consolidation", 4),
            ("reopening_events", 1), ("sensor_state_bytes", 0),
        )]
        lines.append("| " + " | ".join([label, *values]) + " |")
    lines.extend([
        "", "Mean gain covers every optimizer step and weights feature parameters within a step. "
        "Summed data-update L2 is path length, not net displacement; it excludes classifier, decay, and anchor forces. "
        "Final mean c summarizes final consolidation rather than averaging over time.", "",
        "Detector proximity below pools counts across seeds. The recorded tolerance is in optimizer updates; "
        "these are descriptive proximity counts rather than calibrated detector scores.", "",
        "| Method | Reopenings | Changes with nearby reopening / post-warmup changes | Reopenings outside change windows | Mean nearby delay (steps) |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in report["methods"]:
        audit = row["detector_pooled_counts"]
        label = row["method"] + (" \u2020" if row["extra_information_diagnostic"] else "")
        lines.append(f"| {label} | {audit['reopening_events']} | "
                     f"{audit['changes_with_reopening_in_window']} / {audit['post_warmup_changes']} | "
                     f"{audit['reopenings_outside_change_windows']} | {fmt(audit['mean_delay_steps'], 1)} |")
    lines.extend(["", "Per-seed results:", "",
                  "| Method | Seed | Final accuracy (%) | Late AUC (%) | Scratch gap (pp) | Unit resets | Mean gain | Local maturations | Reopenings |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["runs"]:
        label = row["method"] + (" \u2020" if row["extra_information_diagnostic"] else "")
        lines.append(f"| {label} | {row['seed']} | {fmt(row['final_accuracy'], 2, True)} | "
                     f"{fmt(row['late_early_auc'], 2, True)} | {fmt(row['late_plasticity_gap'], 2, True, True)} | "
                     f"{row['resets']} | {fmt(row['mean_feature_gain'], 4)} | {row['local_maturations']} | {row['reopening_events']} |")
    extras = [row for row in report["methods"] if row["extra_information_diagnostic"]]
    if extras:
        lines.extend(["", "Extra-information access:", ""])
        lines.extend(f"- `{row['method']}`: {row['information_access']}." for row in extras)
    if plotted:
        lines.extend(["", "![Acquisition and allocation diagnostics](main_conditions.png)", "",
                      "The figure uses a fixed condition order. Curves show seed means; AUC bands show one sample SD when multiple seeds exist. "
                      "Bars show late AUC means and dots show individual seeds. No inferential comparison is computed."])
    lines.extend(["", f"Training source SHA-256: `{report['source_sha256']}`.", ""])
    return "\n".join(lines)


def plot_main_conditions(report: dict, output: Path) -> bool:
    methods = [method for method in MAIN_METHODS if any(run["method"] == method for run in report["runs"])]
    if not methods:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = dict(zip(MAIN_METHODS, ("#444444", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")))
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(12.5, 8), layout="constrained")
        handles = []
        for position, method in enumerate(methods):
            runs = [run for run in report["runs"] if run["method"] == method]
            color, label = colors[method], LABELS[method]
            style = "--" if method == "acp_v2_oracle" else ":" if method == "er_recycle_yoked_gain" else "-"
            auc = np.asarray([run["experience_early_auc"] for run in runs]) * 100
            experience = np.arange(1, auc.shape[1] + 1)
            line, = axes[0, 0].plot(experience, auc.mean(0), color=color, linestyle=style, label=label)
            handles.append(line)
            if len(runs) > 1:
                sd = auc.std(0, ddof=1)
                axes[0, 0].fill_between(experience, auc.mean(0)-sd, auc.mean(0)+sd, color=color, alpha=0.10)
            late = np.asarray([run["late_early_auc"] for run in runs]) * 100
            axes[0, 1].bar(position, late.mean(), color=color, alpha=0.7,
                           hatch="//" if method == "acp_v2_oracle" else "xx" if method == "er_recycle_yoked_gain" else None)
            offsets = np.linspace(-0.12, 0.12, len(runs)) if len(runs) > 1 else np.zeros(1)
            axes[0, 1].scatter(position+offsets, late, color=color, edgecolor="black", linewidth=0.6, s=22, zorder=3)
            axes[0, 1].text(position, min(101, float(late.max())+1.7), f"{late.mean():.1f}", ha="center", fontsize=8)
            step_grids = [[point["last_step"] for point in run["allocation_series"]] for run in runs]
            if any(grid != step_grids[0] for grid in step_grids[1:]):
                raise ValueError(f"cannot average allocation curves with different training horizons: {method}")
            steps = np.asarray(step_grids[0])
            resets = np.asarray([[point["cumulative_resets"] for point in run["allocation_series"]] for run in runs])
            gains = np.asarray([[point["mean_feature_gain"] for point in run["allocation_series"]] for run in runs])
            axes[1, 0].plot(np.r_[0, steps], np.r_[0, resets.mean(0)], color=color, linestyle=style)
            axes[1, 1].plot(steps, gains.mean(0), color=color, linestyle=style)
        axes[0, 0].set(title="Acquisition by experience (mean; bands: 1 SD)",
                       xlabel="Experience", ylabel="Early acquisition AUC (%)", ylim=(0, 100))
        axes[0, 1].set(title="Late acquisition AUC (dots: individual seeds)", ylabel="AUC (%)",
                       xticks=range(len(methods)), xticklabels=[LABELS[method] for method in methods], ylim=(0, 106))
        axes[0, 1].tick_params(axis="x", labelrotation=22, labelsize=8)
        axes[1, 0].set(title="Cumulative replacement counts", xlabel="Optimizer update",
                       ylabel="Units replaced (seed mean)", ylim=(0, None))
        axes[1, 1].set(title="Effective feature gain (step-bin mean)", xlabel="Optimizer update",
                       ylabel="Parameter-weighted gain", ylim=(0, 1.04))
        axes[0, 0].legend(handles=handles, loc="lower left", fontsize=7)
        figure.suptitle(f"{report['suite']} | descriptive acquisition and allocation\n"
                       "\u2020 Extra information: oracle boundaries or offline matched reset/gain trace", fontsize=12)
        for extension in ("png", "svg"):
            path = output / f"main_conditions.{extension}"
            figure.savefig(path, dpi=180)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n",
                                encoding="utf-8", newline="\n")
        plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="+", type=Path, help="Completed run-suite directories containing manifest.json")
    parser.add_argument("--output", required=True, type=Path, help="Report root; creates one subdirectory per suite name")
    args = parser.parse_args()
    try:
        if len({path.name for path in args.suites}) != len(args.suites):
            raise ValueError("suite directory names must be unique under a shared output root")
        destinations = [args.output.resolve() / path.name for path in args.suites]
        if any(destination == source.resolve() for source, destination in zip(args.suites, destinations)):
            raise ValueError("report destination must differ from each source run-suite directory")
        # Validate every requested suite before producing any report artifacts.
        reports = [summarize_suite(path.resolve()) for path in args.suites]
        for report, destination in zip(reports, destinations):
            destination.mkdir(parents=True, exist_ok=True)
            plotted = plot_main_conditions(report, destination)
            report["artifacts"] = ["diagnostics.json", "diagnostics.md"] + (
                ["main_conditions.png", "main_conditions.svg"] if plotted else [])
            (destination / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            (destination / "diagnostics.md").write_text(markdown(report, plotted), encoding="utf-8")
            completion = report["completion"]
            print(f"{report['suite']}: {completion['completed_runs']}/{completion['expected_runs']} runs -> {destination}")
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Cannot summarize: {error}\n")


if __name__ == "__main__":
    main()
