"""Standalone figures for the locked acquisition diagnostic."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = dict(continue_="#2464A2", frozen="#C68B24", state_reset="#087F7A", fresh="#9B53AE")
LABELS = {"continue": "Continue", "frozen": "Freeze visual features",
          "state_reset": "Retain weights; reset state", "fresh": "Fresh weights"}


def color(arm):
    return COLORS["continue_" if arm == "continue" else arm]


def plot(directory):
    directory = Path(directory)
    result = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    rows, seeds = result["rows"], result["seeds"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "text.color": "#1F2D3C", "axes.labelcolor": "#405064"})
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.subplots_adjust(left=.17, right=.97, top=.85, bottom=.13, wspace=.55, hspace=.72)
    fig.suptitle("Can existing learners keep acquiring new dependencies?", x=.06, y=.985,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(.06, .953, f"{len(seeds)} independent seeds | three physical additions | fixed data and update budgets",
             color="#627085", fontsize=11)
    handles = []
    for column, model in enumerate(("conditional", "recurrent")):
        selected = [r for r in rows if r["model"] == model and r["branch"] == "novel"]
        for arm in LABELS:
            means = [np.mean([r["brier_auc"] for r in selected if r["arm"] == arm and r["stage"] == stage]) for stage in (1, 2, 3)]
            line, = axes[0, column].plot([1, 2, 3], np.array(means)*100, color=color(arm), marker="o", label=LABELS[arm])
            if column == 0:
                handles.append(line)
        axes[0, column].set(title=f"{'A' if column == 0 else 'B'}  {model.title()}: acquisition",
            xlabel="Successive dependency introduction", ylabel="Affected-subset Brier AUC x100\n(lower is better)", xticks=[1, 2, 3])
        axes[0, column].grid(axis="y", alpha=.2)
        contrasts = result["contrasts"][model]
        labels = ["Continue - fresh", "Continue - frozen", "Reset state - continue", "Reset state - fresh"]
        names = ["continue_minus_fresh", "continue_minus_frozen", "state_reset_minus_continue", "state_reset_minus_fresh"]
        for index, name in enumerate(names):
            d = contrasts[name]["brier_auc"]
            axes[1, column].plot(np.array([d["lower"], d["upper"]])*100, [index, index], color="#24364A", linewidth=2)
            axes[1, column].scatter(np.array(d["differences"])*100, index+np.linspace(-.12, .12, len(seeds)),
                                   facecolors="white", edgecolors="#697C92", s=25, zorder=3)
            axes[1, column].scatter([100*d["mean"]], [index], marker="D", color="#24364A", s=35, zorder=4)
        axes[1, column].axvline(0, color="#8D98A6", linewidth=1)
        axes[1, column].set(title=f"{'C' if column == 0 else 'D'}  Paired diagnostic differences",
            xlabel="Brier AUC difference x100\n(negative favors first arm)", yticks=range(4), yticklabels=labels)
        axes[1, column].invert_yaxis()
        for arm in ("continue", "frozen", "state_reset"):
            means = [np.mean([r["valid_damage"] for r in selected if r["arm"] == arm and r["stage"] == stage]) for stage in (1, 2, 3)]
            axes[2, column].plot([1, 2, 3], np.array(means)*100, color=color(arm), marker="o")
        axes[2, column].axhline(0, color="#8D98A6", linewidth=1)
        axes[2, column].set(title=f"{'E' if column == 0 else 'F'}  Change in valid old predictions",
            xlabel="Successive dependency introduction", ylabel="Valid-old Brier change x100\n(positive is damage)", xticks=[1, 2, 3])
        axes[2, column].grid(axis="y", alpha=.2)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, .929), ncol=2, frameon=False)
    fig.text(.06, .025, "Acquisition includes initial competence and adaptation; later episodes also change world complexity.\n"
        "Paired panels: circles are seeds, diamonds means, lines descriptive 95% intervals. Episodes averaged within seed.\n"
        f"Fresh-model qualification: {'PASS' if result['qualification']['passed'] else 'INCOMPLETE'}. "
        "Forecast improvement need not change the best one-time action. See the report for cue use and final branches.", fontsize=9, color="#627085")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory/f"overview.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    plot(args.input)
