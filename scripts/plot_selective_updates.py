"""Present complete paired-seed development results without changing decisions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_training_state import paired


ARMS = ("current", "protected", "combined", "shrink")
LABELS = ("Current\nemphasis", "Protection\nonly", "Combined", "Shrink\ncontrol")
COLORS = ("#326aa0", "#80569b", "#23836a", "#848c95")


def plot(source, output):
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = {(r["model"], r["seed"], r["arm"]): r for r in data["rows"]}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, model in enumerate(("conditional", "recurrent")):
        for row_index, (field, title) in enumerate((("brier_auc", "Acquiring the new dependency"),
                                                   ("valid_after", "Retaining valid older predictions"))):
            ax = axes[row_index, column]
            ax.axhline(0, color="#53606d", linewidth=1, linestyle="--")
            for index, (arm, color) in enumerate(zip(ARMS, COLORS)):
                values = np.array([rows[model, seed, arm][field]-rows[model, seed, "reference"][field]
                                   for seed in data["config"]["seeds"]])*100
                summary = paired(values)
                ax.scatter(index+np.linspace(-.14, .14, len(values)), values,
                           color=color, alpha=.62, s=26, zorder=3)
                ax.errorbar(index, summary["mean"],
                            yerr=[[summary["mean"]-summary["lower"]], [summary["upper"]-summary["mean"]]],
                            fmt="D", ms=7, color=color, markeredgecolor="white", capsize=5, lw=2, zorder=4)
            ax.set(title=f"{model.capitalize()} · {title}", xticks=range(len(ARMS)),
                   xticklabels=LABELS, ylabel="Brier difference ×100 vs reference\n(lower is better)")
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("Adam · current emphasis and selective update protection", fontsize=16, fontweight="bold")
    fig.supxlabel("Dots: paired seed differences. Diamonds: means with descriptive 95% seed-bootstrap intervals.\n"
                  "Older-prediction plot averages modes; the decision rule checks each mode separately. Development only.",
                  fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output/f"selective_updates.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.input, args.output)
