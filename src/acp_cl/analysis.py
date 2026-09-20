"""Exploratory paired summaries and standalone research plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import write_json
from .metrics import paired_bootstrap


ENDPOINTS = ("final_accuracy", "forgetting", "late_early_auc", "late_plasticity_gap")


def analyze(directory: Path) -> dict:
    files = sorted(directory.glob("*/result.json"))
    if not files:
        raise ValueError(f"no completed results in {directory}")
    results = [json.loads(file.read_text(encoding="utf-8")) for file in files]
    identities = {(r["config_sha256"], r["source_sha256"], r.get("execution_device"),
                   r.get("environment", {}).get("torch"),
                   r.get("environment", {}).get("numpy"),
                   r.get("environment", {}).get("gpu") if r.get("execution_device", "").startswith("cuda") else None)
                  for r in results}
    if len(identities) != 1:
        raise ValueError("refusing to pool different configurations, source versions, or execution environments")
    by_method = {}
    for result in results:
        method = result["method"]
        by_method.setdefault(method, {})
        if result["seed"] in by_method[method]:
            raise ValueError(f"duplicate seed for {method}")
        by_method[method][result["seed"]] = result
    rows = []
    for method, seed_results in by_method.items():
        runs = list(seed_results.values())
        row = {"method": method, "n_seeds": len(runs),
               "information_access": runs[0].get("information_access", "training stream only")}
        for endpoint in ENDPOINTS:
            values = [r["metrics"][endpoint] for r in runs if endpoint in r["metrics"]]
            row[endpoint] = float(np.mean(values)) if values else None
            row[endpoint+"_sd"] = float(np.std(values, ddof=1)) if len(values)>1 else None
        for name in ("wall_seconds", "replay_bytes", "peak_cuda_bytes", "reopening_events"):
            values = [r[name] for r in runs if r.get(name) is not None]
            row[name] = float(np.mean(values)) if values else None
        rows.append(row)
    paired = []
    focal = results[0]["config"].get("primary_method", "acp_v2" if "acp_v2" in by_method else "acp")
    if focal in by_method:
        for comparator, runs in by_method.items():
            if comparator == focal:
                continue
            common = sorted(set(runs) & set(by_method[focal]))
            for endpoint in ENDPOINTS:
                if not common or any(endpoint not in runs[s]["metrics"] or
                                     endpoint not in by_method[focal][s]["metrics"] for s in common):
                    continue
                # Pairing is valid only with exactly the same stream and config.
                for seed in common:
                    if runs[seed]["class_order"] != by_method[focal][seed]["class_order"]:
                        raise ValueError("paired class orders differ")
                    if runs[seed].get("stream_metadata") != by_method[focal][seed].get("stream_metadata"):
                        raise ValueError("paired stream metadata differ")
                interval = paired_bootstrap(
                    [by_method[focal][s]["metrics"][endpoint] for s in common],
                    [runs[s]["metrics"][endpoint] for s in common], seed=27183)
                paired.append({"contrast": f"{focal} - {comparator}", "endpoint": endpoint,
                               "seeds": common, "all_seeds_matched": set(runs)==set(by_method[focal]),
                               **interval})
    summary = {"status": "exploratory; unadjusted paired percentile bootstrap intervals",
               "eval_split": results[0]["eval_split"], "config": results[0]["config"],
               "config_sha256": results[0]["config_sha256"],
               "source_sha256": results[0]["source_sha256"], "methods": rows, "paired": paired}
    write_json(directory / "summary.json", summary)
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Exploratory experiment results", "",
             f"Evaluation split: **{summary['eval_split']}**. Values below are percentages or percentage points.",
             "These runs test implementation and generate hypotheses; they are not confirmatory evidence.", "",
             "| Method | Seeds | Final accuracy | Forgetting | Late early AUC | Scratch-adjusted AUC | Seconds/run |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "—" if value is None else f"{100*value:.2f}"
    for row in rows:
        lines.append(f"| {row['method']} | {row['n_seeds']} | {percent(row['final_accuracy'])} | "
                     f"{percent(row['forgetting'])} | {percent(row['late_early_auc'])} | "
                     f"{percent(row['late_plasticity_gap'])} | {row['wall_seconds']:.1f} |")
    if paired:
        lines.extend(["", f"Paired {focal} differences (95% unadjusted percentile bootstrap intervals; "
                      "small-seed intervals are unstable):", "",
                      "| Comparator | Endpoint | Pairs | Difference [95% CI], pp |",
                      "|---|---|---:|---:|"])
        for pair in paired:
            lines.append(f"| {pair['contrast']} | {pair['endpoint']} | {pair['n_pairs']} | "
                         f"{100*pair['mean_difference']:+.2f} "
                         f"[{100*pair['ci_low']:+.2f}, {100*pair['ci_high']:+.2f}] |")
    lines.extend(["", "Lower forgetting is better; higher accuracy and acquisition AUC are better.",
                  "Methods with oracle boundaries or offline yoked traces are extra-information diagnostics, not task-free competitors.",
                  "Scratch-adjusted AUC subtracts a fresh model's AUC on the identical experience; it is a diagnostic, not pure transfer.",
                  "Timing includes evaluation, diagnostics, and checkpoint writes. It is not a matched-compute comparison.",
                  "The recycling and DER++ comparators are explicitly documented implementation variants.",
                  "", "![Exploratory results](overview.png)", ""])
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plot_overview(directory, by_method)
    return summary


def plot_overview(directory: Path, by_method: dict) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    palette = plt.get_cmap("tab10")
    methods = list(by_method)
    for i, method in enumerate(methods):
        runs = list(by_method[method].values())
        color = palette(i % 10)
        accuracy = [100*r["metrics"]["final_accuracy"] for r in runs]
        axes[0, 0].bar(i, np.mean(accuracy), color=color, alpha=0.75)
        axes[0, 0].scatter(np.full(len(runs), i), accuracy, color=color, edgecolor="black", s=20)
        forgetting = [100*r["metrics"]["forgetting"] for r in runs]
        late_auc = [100*r["metrics"]["late_early_auc"] for r in runs]
        axes[0, 1].scatter(np.mean(forgetting), np.mean(late_auc), color=color, label=method, s=50)
        early = np.array([r["early_auc"] for r in runs])*100
        xs = np.arange(1, early.shape[1]+1)
        axes[1, 0].plot(xs, early.mean(0), marker="o", markersize=3, color=color, label=method)
        if len(runs)>1:
            axes[1, 0].fill_between(xs, early.mean(0)-early.std(0, ddof=1),
                                    early.mean(0)+early.std(0, ddof=1), color=color, alpha=0.08)
        gaps = [r["plasticity_gap"] for r in runs if r.get("plasticity_gap") is not None]
        if gaps:
            axes[1, 1].plot(xs, 100*np.mean(gaps, axis=0), marker="o", markersize=3,
                            color=color, label=method)
    axes[0, 0].set(xticks=np.arange(len(methods)), xticklabels=methods, ylabel="Accuracy (%)",
                   title="Final retained performance (dots: individual seeds)", ylim=(0, 100))
    axes[0, 0].tick_params(axis="x", labelrotation=35, labelsize=8)
    axes[0, 1].set(xlabel="Forgetting (percentage points; lower is better)",
                   ylabel="Late early-learning AUC (%)", title="Retention and acquisition")
    axes[0, 1].legend(fontsize=7, loc="best")
    axes[1, 0].set(xlabel="Experience", ylabel="Early-learning AUC (%)",
                   title="Acquisition across the stream (bands: ±1 SD)")
    axes[1, 1].axhline(0, color="gray", linewidth=1, linestyle="--")
    axes[1, 1].set(xlabel="Experience", ylabel="AUC difference (percentage points)",
                   title="Acquisition relative to same-experience scratch")
    figure.suptitle("ACP-CL development pilot — not a confirmatory study", fontsize=15)
    for suffix in ("png", "svg"):
        figure.savefig(directory / f"overview.{suffix}", dpi=160)
    plt.close(figure)
