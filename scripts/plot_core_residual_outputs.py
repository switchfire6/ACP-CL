"""Plot the two predeclared output-allocation contrasts, retaining every seed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    jobs = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((args.input / "jobs").glob("*.json"))]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.2), layout="constrained")
    colors = {"conditional": "#366bb0", "recurrent": "#168078"}
    specs = (("novel_end", "focus_brier", "core", "full", .002,
              "Does the correction add new knowledge?", "Core error − combined error"),
             ("return_entry", "brier", "full", "core", .005,
              "Does the correction disrupt old knowledge?", "Combined error − core error"))
    for row_index, model in enumerate(("conditional", "recurrent")):
        selected = sorted((job for job in jobs if job["model"] == model), key=lambda x: x["seed"])
        for col, (target, metric, left, right, margin, title, label) in enumerate(specs):
            values = []
            for job in selected:
                panel = [row for row in job["rows"] if row["arm"] == "separate"
                         and row["target"] == target and row["context"] == "refreshed"
                         and not row["flipped"]]
                assert len(panel) == 2
                values.append(np.mean([row["metrics"][left][metric] - row["metrics"][right][metric]
                                       for row in panel]))
            ax = axes[row_index, col]
            positions = np.arange(len(values))
            ax.bar(positions, values, width=.58, color=colors[model], alpha=.86)
            ax.axhline(0, color="#333333", linewidth=.8)
            ax.axhline(margin, color="#a86912", linestyle="--", linewidth=1,
                       label=f"Mean criterion: {margin:.3f}")
            ax.set_xticks(positions, [str(job["seed"]) for job in selected], fontsize=8)
            ax.set_ylabel(label + "\n(raw Brier units)", fontsize=9)
            ax.set_title(f"{model.capitalize()} · {title}", fontsize=10, loc="left", pad=12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.16)
            ax.set_axisbelow(True)
            ax.legend(loc="lower left", frameon=False, fontsize=8)
            ax.text(.98, .97, f"Mean {np.mean(values):+.4f} | {np.count_nonzero(np.array(values) > 0)}/{len(values)} positive",
                    ha="right", va="top", transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    fig.suptitle("Saved models: can useful old and new predictions coexist?\n"
                 "Frozen-core arm · two refreshed support samples averaged per seed · no training",
                 fontsize=14, fontweight="bold")
    fig.supxlabel("Each architecture needs both mean margins and ≥5/6 positive seeds in both panels.\n"
                  "This is an exploratory allocation signature, not a successful switching policy.", fontsize=10)
    fig.savefig(args.input / "allocation.png", dpi=180)
    fig.savefig(args.input / "allocation.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
