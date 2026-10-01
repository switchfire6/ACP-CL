"""Render a standalone scientific figure from the portable pilot archive."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"none": "Current only", "uniform": "Uniform replay", "recent": "Recent replay",
          "coverage": "Coverage", "recurrence": "Frequency", "relevance": "Survival relevance",
          "joint": "Combined rule", "frozen_encoder": "Random frozen features", "frozen_late": "Features frozen later"}
COLORS = {"none": "#9A647E", "uniform": "#2866A4", "recent": "#5D8475", "coverage": "#8093A7",
          "joint": "#CF653B", "frozen_late": "#7A6AAA"}


def paired_plot(ax, archive, endpoint, title):
    methods = [m for m in archive["manifest"]["config"]["methods"] if m != "uniform"]
    for index, method in enumerate(methods):
        row = next(r for r in archive["contrasts"] if r["schedule"] == "long"
                   and r["method"] == method and r["endpoint"] == endpoint)
        values = np.array(row["differences"]) * 100
        jitter = np.linspace(-.17, .17, len(values))
        color = COLORS.get(method, "#81929C")
        ax.scatter(values, index + jitter, s=23, alpha=.8, color=color, zorder=3)
        ax.plot([100 * row["lower"], 100 * row["upper"]], [index, index], color="#263744", lw=1.7)
        ax.scatter([100 * row["mean"]], [index], s=34, marker="D", color="#263744", zorder=4)
    ax.axvline(0, color="#8E9DA5", lw=1, linestyle="--")
    ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=13)
    ax.set_xlabel("Difference versus uniform replay (percentage points)")
    ax.grid(axis="x", color="#E3E8EC", linewidth=.6)


def curve_plot(ax, archive, stream, methods, title):
    for method in methods:
        records = [r for r in archive["runs"] if r["schedule"] == stream and r["method"] == method]
        x = [p["arrivals"] for p in records[0]["blocks"][-1]["curve"]]
        values = np.array([[p["metrics"]["survival"]["12"] for p in r["blocks"][-1]["curve"]]
                           for r in records]) * 100
        ax.plot(x, values.mean(axis=0), marker="o", markersize=4, linewidth=2,
                color=COLORS.get(method, "#8093A7"), label=LABELS[method])
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=13)
    ax.set_xlabel("New observed interactions")
    ax.set_ylabel("Joint survival through 12 steps (%)")
    ax.grid(color="#E3E8EC", linewidth=.6)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot(archive_path, output):
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        archive = json.load(handle)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#263744", "text.color": "#263744",
                         "axes.edgecolor": "#BCC6CB", "figure.facecolor": "#FAFCFD"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    paired_plot(axes[0, 0], archive, "entry_survival", "A   Remembering after a long absence")
    paired_plot(axes[0, 1], archive, "composition_final", "B   Transferring to withheld combinations")
    curve_plot(axes[1, 0], archive, "long", ["uniform", "joint", "coverage", "frozen_late"],
               "C   Learning a new dependency late")
    curve_plot(axes[1, 1], archive, "reversal", ["uniform", "joint", "recent", "none"],
               "D   Revising a familiar but changed rule")
    fig.suptitle("Can future usefulness guide what a learner keeps?", fontsize=20,
                 fontweight="bold", x=.05, ha="left", y=.98)
    fig.text(.05, .945, "Joint-persistence pilot  |  288 runs  |  8 paired seeds  |  fixed model and replay capacity",
             fontsize=11, color="#637782")
    primary = archive["primary"]
    fig.text(.05, .02,
             f"Primary combined-rule effect: {100 * primary['mean']:+.2f} pp. "
             f"Continuation screen: {'passed' if archive['continue'] else 'not met'}. "
             "Dots: paired seeds. Diamonds and bars: mean and descriptive 95% bootstrap interval.\n"
             "Learning curves show seed means. This small finite-horizon study does not establish a general continual-learning mechanism.",
             fontsize=9, color="#637782")
    fig.subplots_adjust(left=.17, right=.97, top=.885, bottom=.12, hspace=.37, wspace=.52)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix("." + extension), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.archive, args.output)


if __name__ == "__main__":
    main()
