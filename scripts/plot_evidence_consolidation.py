"""Plot lifetime cost, matched cycles, and acceptance under fixed proposals."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"immediate": "#405266", "periodic": "#b07720", "single": "#a66591", "sustained": "#147f83"}
LABELS = {"immediate": "Immediate", "periodic": "Fixed delay", "single": "One packet", "sustained": "Repeated evidence"}


def plot(directory):
    directory = Path(directory)
    s = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    rows = s["block_rows"]
    models, seeds = config["models"], config["seeds"]
    blocks = sorted({r["block"] for r in rows})
    # Seed units are preserved; a line never treats windows/blocks as new seeds.
    def values(model, policy, field):
        return np.array([[next(r[field] for r in rows if r["seed"] == seed and r["model"] == model
            and r["policy"] == policy and r["block"] == block) for block in blocks] for seed in seeds])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "#fafbf9", "axes.facecolor": "#fafbf9"})
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.3*len(models)), squeeze=False, layout="constrained")
    x = np.array(blocks)+1
    for row, model in enumerate(models):
        base = values(model, "immediate", "truth_brier")
        for policy in ("periodic", "single", "sustained"):
            difference = values(model, policy, "truth_brier")-base
            cumulative = difference.cumsum(axis=1)*3*config["episode_size"]
            axes[row, 0].plot(x, cumulative.mean(axis=0), color=COLORS[policy], label=LABELS[policy], linewidth=2)
            if policy == "sustained":
                for seed_line in cumulative:
                    axes[row, 0].plot(x, seed_line, color=COLORS[policy], alpha=.17, linewidth=.8)
            axes[row, 1].plot(x, 100*difference.mean(axis=0), color=COLORS[policy], label=LABELS[policy], linewidth=1.8, marker="o", markersize=3)
            adopts = values(model, policy, "adoptions")
            windows = config["episode_size"]//config["batch_size"]//config["evidence_window"]
            axes[row, 2].plot(x, 100*adopts.mean(axis=0)/windows, color=COLORS[policy], label=LABELS[policy], marker="o", markersize=3)
        for j, title in enumerate(("Cumulative physical prediction cost", "Local performance gap", "Proposals adopted")):
            ax = axes[row, j]
            ax.set_title(f"{model.capitalize()} · {title}", loc="left", fontsize=11)
            ax.set_xlabel("Continuous block")
            ax.set_xticks(x[::2])
            ax.grid(axis="y", alpha=.15)
            if j < 2:
                ax.axhline(0, color="#66717a", linewidth=.8)
            for block in (5, 10, 15):
                if block <= max(x):
                    ax.axvspan(block-.45, block+.45, color="#b07720", alpha=.09)
        axes[row, 0].set_ylabel("Summed Brier excess (3 horizons)\nLower is better; faint lines: individual seeds")
        axes[row, 1].set_ylabel("Block Brier difference ×100\nLower is better")
        axes[row, 2].set_ylabel("Accepted windows (%)")
        axes[row, 2].set_ylim(-3, 105)
    axes[0, 0].legend(frameon=False, loc="best", fontsize=9)
    tag = "Engineering check" if config["episode_size"] < 8192 else "Development" if config["kind"] == "development" else "Fixed pilot"
    unit = "seed" if len(seeds) == 1 else "seeds"
    noise_note = " · shaded blocks: report noise" if max(x) >= 5 else ""
    fig.suptitle(f"Adam: evidence before consolidation — {tag}\n{len(seeds)} {unit} per architecture · shared draft learning · gaps relative to immediate{noise_note}", fontsize=13)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(directory/f"lifetime_consolidation.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(models), figsize=(12, 5.2), squeeze=False, layout="constrained")
    endpoints = [("overall_truth", "Whole stream"), ("late_truth", "Final matched cycle"),
                 ("intro_focus_auc", "New-dependency learning"), ("return_truth", "Returning environment"),
                 ("noise_truth", "Noisy reports"), ("valid_after", "Still-valid knowledge")]
    for col, model in enumerate(models):
        ax = axes[0, col]
        contrast = s["contrasts"][model]["sustained_minus_immediate"]
        available = [(k, label) for k, label in endpoints if contrast[k] is not None]
        for index, (key, _) in enumerate(available):
            item = contrast[key]
            ax.plot([100*item["lower"], 100*item["upper"]], [index, index], color=COLORS["sustained"], linewidth=2)
            ax.scatter([100*item["mean"]], [index], color=COLORS["sustained"], s=55, zorder=3)
            ax.scatter(100*np.asarray(item["differences"]), index+np.linspace(-.12, .12, len(seeds)),
                       color=COLORS["sustained"], alpha=.35, s=16)
        ax.set_yticks(range(len(available)), [label for _, label in available])
        ax.invert_yaxis()
        ax.axvline(0, color="#66717a", linewidth=.8)
        ax.set_xlabel("Repeated evidence minus immediate · Brier ×100\nNegative favors repeated evidence")
        ax.set_title(model.capitalize(), loc="left")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle(f"Consolidation benefits and costs — {tag}\nMeans, paired-seed 95% bootstrap intervals, and individual seeds", fontsize=13)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(directory/f"consolidation_effects.{ext}", dpi=170)
    plt.close(fig)
    print(f"Saved lifetime and effect figures in {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    plot(parser.parse_args().input)
