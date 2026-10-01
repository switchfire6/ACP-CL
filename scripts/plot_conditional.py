"""Standalone scientific figure from the portable conditional-reuse archive."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = dict(conditional="#087f8c", pooled="#657284", shuffled="#be623a", oracle="#a88720")
LABELS = dict(conditional="Evidence routing", pooled="Pooled predictions",
              shuffled="Broken history binding", oracle="True mode supplied (diagnostic)")


def plot(archive, output):
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    summary = data["summary"]
    records = {(r["schedule"], r["method"], r["seed"]): r for r in data["records"]}
    seeds = summary["seeds"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.subplots_adjust(left=.075, right=.975, bottom=.09, top=.86, hspace=.42, wspace=.34)
    fig.suptitle("Can recent evidence reactivate a learned relationship?", x=.075, ha="left",
                 y=.975, fontsize=19, fontweight="bold", color="#172938")
    fig.text(.075, .93, "144 runs · 8 paired seeds · random initial features · 32 past outcomes · fixed model capacity",
             fontsize=11, color="#526372")
    fig.text(.075, .901, "Main test: a familiar condition returns after three blocks away. Model weights stay fixed during the probe.",
             fontsize=10, color="#526372")

    ax = axes[0, 0]
    for method in COLORS:
        curves = []
        for seed in seeds:
            reps = records["recurring", method, seed]["return_probe"]["replicates"]
            curves.append(np.mean([[p["metrics"]["survival"] for p in rep["curve"]] for rep in reps], axis=0))
        x = [p["feedback"] for p in reps[0]["curve"]]
        ax.plot(x, 100*np.mean(curves, axis=0), marker="o", color=COLORS[method],
                label=LABELS[method], linewidth=2)
    ax.set(title="A  Reuse without weight updates", xlabel="Feedback records from the returning condition",
           ylabel="Joint survival (%)", xticks=x)
    ax.grid(axis="y", alpha=.15)
    ax.legend(fontsize=8, frameon=False, loc="best")

    ax = axes[0, 1]
    primary = summary["primary"]
    differences = 100*np.asarray(primary["differences"])
    ax.axvline(0, color="#9aa5ae", linewidth=1)
    ax.axvline(2, color="#b9c4c6", linewidth=1, linestyle=":")
    ax.scatter(differences, range(len(seeds)), color=COLORS["conditional"], s=36)
    mid, low, high = (100*primary[k] for k in ("mean", "lower", "upper"))
    ax.errorbar(mid, len(seeds)+.5, xerr=[[mid-low], [high-mid]], fmt="D",
                color="#172938", capsize=4, markersize=6)
    ax.set(yticks=[*range(len(seeds)), len(seeds)+.5],
           yticklabels=[*map(str, seeds), "Mean + 95% interval"],
           title="B  Primary result, every seed", xlabel="Evidence routing minus pooled (percentage points)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.12)

    ax = axes[1, 0]
    kinds, labels = ("correct", "opposite", "erased", "broken_binding"), (
        "Correct\nhistory", "Opposite\nconsequences", "History\nerased", "Binding\nbroken")
    values = []
    for kind in kinds:
        group = []
        for seed in seeds:
            reps = records["recurring", "conditional", seed]["return_probe"]["replicates"]
            group.append(100*np.mean([rep["curve"][-1]["metrics"]["survival"] if kind == "correct"
                                     else rep[kind]["survival"] for rep in reps]))
        values.append(group)
    for index in range(len(seeds)):
        ax.plot(range(4), np.asarray(values)[:, index], color="#b9c7ca", alpha=.55, linewidth=.8)
    ax.plot(range(4), np.mean(values, axis=1), color=COLORS["conditional"], marker="o", linewidth=2.5)
    ax.set(title="C  Same weights and queries, different evidence", ylabel="Joint survival (%)",
           xticks=range(4), xticklabels=labels)
    ax.grid(axis="y", alpha=.15)
    ax.text(.02, .04, "Thin lines: individual seeds; thick line: mean", transform=ax.transAxes,
            fontsize=8, color="#526372")

    ax = axes[1, 1]
    contrasts = summary["contrasts"]
    groups = [("Recurring: primary", "recurring", "return_survival"),
              ("Stable condition", "stable", "return_survival"),
              ("Unpredictable condition", "unpredictable", "return_survival"),
              ("Final late-gate learning", "recurring", "final_gate")]
    ax.axvline(0, color="#9aa5ae", linewidth=1)
    ax.axvline(-2, color="#be623a", linewidth=1, linestyle=":", alpha=.55)
    for i, (_, stream, metric) in enumerate(groups):
        contrast = contrasts[stream]["conditional_minus_pooled"][metric]
        mid, low, high = (100*contrast[k] for k in ("mean", "lower", "upper"))
        ax.errorbar(mid, i, xerr=[[mid-low], [high-mid]], fmt="o", capsize=4,
                    color=COLORS["conditional"] if mid >= 0 else COLORS["shuffled"])
    ax.set(yticks=range(4), yticklabels=[g[0] for g in groups], title="D  Controls and late learning",
           xlabel="Evidence routing minus pooled (percentage points)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.12)
    fig.text(.075, .025, "Intervals: descriptive paired seed bootstrap; no multiple-comparison correction. "
             "Oracle uses hidden labels. No claim of general continual-learning performance.", fontsize=8,
             color="#526372")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix("."+suffix), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.archive, args.output)
