"""Plot the completed matched-context diagnosis without rescoring checkpoints."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"start_draft": "#2767a5", "start_sustained": "#b76c22",
          "prefix_draft": "#707782", "end_draft": "#22846f"}
LABELS = {"start_draft": "Current draft", "start_sustained": "Sustained snapshot",
          "prefix_draft": "Original prefix", "end_draft": "After return training"}


def plot(source, output):
    source, output = Path(source), Path(output)
    data = json.loads(source.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, model in enumerate(("conditional", "recurrent")):
        ax, bars = axes[:, column]
        rows = [r for r in data["seed_means"] if r["model"] == model]
        for weight, color in COLORS.items():
            means = []
            for support in ("actual_start", "observed_8", "observed_16", "observed_32"):
                values = [100*r["brier"] for r in rows if r["weight"] == weight
                          and r["support"] == support]
                means.append(np.mean(values))
            ax.plot([0, 8, 16, 32], means, "o-", color=color, lw=2, ms=5, label=LABELS[weight])
        ax.set(title=f"{model.capitalize()}: refresh context, hold weights fixed",
               xlabel="Observed return records in the 32-record history",
               ylabel="All-action Brier ×100 (lower is better)", xticks=[0, 8, 16, 32])
        ax.grid(axis="y", alpha=.18)
        if column == 0:
            ax.legend(fontsize=9)
        names = list(COLORS)
        for index, weight in enumerate(names):
            values = np.array([100*r["brier"] for r in rows if r["weight"] == weight
                               and r["support"] == "fresh_target"])
            bars.bar(index, values.mean(), color=COLORS[weight], alpha=.8, width=.64)
            bars.scatter(index+np.linspace(-.16, .16, len(values)), values,
                         s=19, color="#27313d", edgecolors="white", linewidth=.5, zorder=3)
        bars.set(title="Same independent target-law support for every model",
                 ylabel="All-action Brier ×100 (lower is better)",
                 xticks=range(len(names)), xticklabels=[LABELS[name].replace(" ", "\n") for name in names])
        bars.grid(axis="y", alpha=.18)
        bars.set_axisbelow(True)
    fig.suptitle("Adam · context recovery and retained predictions", fontsize=17, fontweight="bold")
    fig.supxlabel("Six seeds per architecture; returns and support replicates averaged within seed.\n"
                  "Dots show individual seed means. Retrospective diagnosis; target support is a privileged reference.",
                  fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output/f"return_context.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    plot(arguments.input, arguments.output)
