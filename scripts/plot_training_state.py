"""Development qualification and optional completed factorial-study figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"continue": "#2864A2", "optimizer_reset": "#C68B24", "replay_reset": "#A14E70",
          "state_reset": "#087F7A", "fresh": "#8055B0"}
LABELS = {"continue": "Continue", "optimizer_reset": "Reset Adam", "replay_reset": "Reset replay",
          "state_reset": "Reset both", "fresh": "Fresh weights"}


def save(fig, path):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180, facecolor="white")
    plt.close(fig)


def plot_development(directory):
    attempts = []
    for p in sorted(directory.glob("development_*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        c = json.loads((p.parent/"config.json").read_text(encoding="utf-8"))
        attempts.append((c["episode_size"], s))
    attempts.sort(key=lambda a: a[0])
    if not attempts:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.subplots_adjust(left=.09, right=.97, top=.80, bottom=.13, wspace=.34, hspace=.51)
    fig.suptitle("Can fresh models learn all three dependencies?", x=.05, y=.97,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(.05, .918, "New development seeds; no optimizer/replay treatment outcomes select the exposure budget.",
             fontsize=10, color="#556579")
    fig.text(.05, .875, "Dots: seeds   |   Line: mean cue benefit   |   Red: development requirement   |   Gray: comparison requirement",
             fontsize=9, color="#556579")
    for i, model in enumerate(("conditional", "recurrent")):
        for cue, name in enumerate(("Transport efficiency", "Arrival delay", "Supply timing")):
            ax = axes[i, cue]
            means = []
            for j, (size, result) in enumerate(attempts):
                values = [100*r["cue_effect"] for r in result["rows"] if r["model"] == model and r["cue"] == cue]
                means.append(np.mean(values))
                ax.scatter(j+np.linspace(-.09, .09, len(values)), values, s=24, color="#7197B5", alpha=.75)
            ax.plot(range(len(attempts)), means, color="#173E5D", marker="D", linewidth=2)
            ax.axhline(.25, color="#B13C42", linestyle="--", linewidth=1.4)
            ax.axhline(.20, color="#959DA8", linestyle=":", linewidth=1.2)
            ax.axhline(0, color="#CDD4DC", linewidth=.8)
            ax.set(title=f"{model.title()}: {name}", xlabel="Training arrivals per fresh fit",
                   xticks=range(len(attempts)), xticklabels=[f"{size:,}" for size, _ in attempts])
            ax.set_xlim(-.35, len(attempts)-.65 if len(attempts)>1 else .35)
            if cue == 0:
                ax.set_ylabel("Correct-cue Brier benefit x100\n(higher means more predictive cue use)")
            ax.grid(axis="y", alpha=.15)
    fig.text(.05, .035, "Cue benefit: error with the query cue flipped minus error with the correct cue; clean physical outcomes fixed.\n"
             "Each panel uses its own scale. Qualification also requires marginal-predictor improvement and every stage group to pass.",
             fontsize=9, color="#556579")
    save(fig, directory/"development_qualification")


def plot_comparison(directory):
    p = directory/"summary.json"
    if not p.exists():
        return
    result = json.loads(p.read_text(encoding="utf-8"))
    if result["kind"] != "diagnostic":
        return
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.subplots_adjust(left=.17, right=.97, top=.85, bottom=.12, hspace=.70, wspace=.53)
    fig.suptitle("Which carried training state changes the learning tradeoff?", x=.055, y=.98,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(.055, .947, f"{len(result['seeds'])} paired seeds | identical initial weights and recent history in the four factorial arms",
             fontsize=10, color="#556579")
    handles = []
    effect_names = ("optimizer_with_replay_kept", "replay_with_optimizer_kept", "both_minus_continue", "interaction")
    effect_labels = ("Reset Adam / keep replay", "Reset replay / keep Adam", "Reset both", "Interaction")
    for col, model in enumerate(("conditional", "recurrent")):
        rows = [r for r in result["rows"] if r["model"] == model and r["branch"] == "novel"]
        for arm, color in COLORS.items():
            values = [np.mean([r["brier_auc"] for r in rows if r["arm"] == arm and r["stage"] == k])*100 for k in (1, 2, 3)]
            line, = axes[0, col].plot([1, 2, 3], values, color=color, marker="o", label=LABELS[arm])
            if col == 0:
                handles.append(line)
        axes[0, col].set(title=f"{model.title()}: new-dependency acquisition", xlabel="Successive dependency introduction",
                         ylabel="Affected-subset Brier AUC x100\n(lower is better)", xticks=[1, 2, 3])
        for i, name in enumerate(effect_names):
            d = result["contrasts"][model][name]["brier_auc"]
            axes[1, col].plot(100*np.array([d["lower"], d["upper"]]), [i, i], color="#273D54", linewidth=2)
            axes[1, col].scatter(100*np.array(d["differences"]), i+np.linspace(-.1, .1, d["n"]),
                                 facecolors="white", edgecolors="#8C9DAE", s=23, zorder=3)
            axes[1, col].scatter([100*d["mean"]], [i], marker="D", color="#273D54", zorder=4)
        axes[1, col].axvline(0, color="#939EAB", linewidth=1)
        axes[1, col].set(title="Paired acquisition effects", xlabel="Brier AUC difference x100\n(negative is improvement)",
                         yticks=range(4), yticklabels=effect_labels)
        axes[1, col].invert_yaxis()
        for arm in list(COLORS)[:4]:
            values = [np.mean([r["valid_damage"] for r in rows if r["arm"] == arm and r["stage"] == k])*100 for k in (1, 2, 3)]
            axes[2, col].plot([1, 2, 3], values, color=COLORS[arm], marker="o")
        axes[2, col].axhline(0, color="#939EAB", linewidth=1)
        axes[2, col].set(title="Change in still-valid old predictions", xlabel="Successive dependency introduction",
                         ylabel="Valid-old Brier change x100\n(positive is damage)", xticks=[1, 2, 3])
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .925), ncol=3, frameon=False)
    fig.text(.055, .03, "Seed differences average the three introductions before bootstrap; intervals are descriptive.\n"
        "Later episodes also change world complexity. Timing and reset information are privileged diagnostic interventions.\n"
        f"Fresh-model qualification: {'PASS' if result['qualification']['passed'] else 'INCOMPLETE'}. "
        "See the report for final branches and matched clean/noisy feedback.", fontsize=9, color="#556579")
    save(fig, directory/"overview")


def plot_final_branches(directory):
    path = directory/"summary.json"
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["kind"] != "diagnostic":
        return
    arms = tuple(COLORS)[:4]
    branches = ("return", "revision", "clean", "noise")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.subplots_adjust(left=.12, right=.97, top=.85, bottom=.13, hspace=.78, wspace=.38)
    fig.suptitle("Returning rules, revised rules, and misleading feedback", x=.06, y=.98,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(.06, .948, "Every branch starts from the same continuing learner; only the declared training state is reset.",
             fontsize=10, color="#556579")
    handles = []
    for col, model in enumerate(("conditional", "recurrent")):
        for j, arm in enumerate(arms):
            x = np.arange(len(branches))+(j-1.5)*.18
            for row, field in enumerate(("brier_auc", "valid_damage")):
                values = [100*result["arm_means"][model][branch][arm][field] for branch in branches]
                bars = axes[row, col].bar(x, values, width=.16, color=COLORS[arm], label=LABELS[arm])
                if row == 0 and col == 0:
                    handles.append(bars)
            effect = result["noise_minus_clean"][model][arm]["brier_auc"]
            ax = axes[2, col]
            ax.plot([j, j], 100*np.array([effect["lower"], effect["upper"]]), color=COLORS[arm], linewidth=2)
            ax.scatter(j+np.linspace(-.06, .06, effect["n"]), 100*np.array(effect["differences"]),
                       facecolors="white", edgecolors=COLORS[arm], s=25, zorder=3)
            ax.scatter([j], [100*effect["mean"]], marker="D", color=COLORS[arm], zorder=4)
        for row in (0, 1):
            axes[row, col].set(xticks=range(len(branches)),
                               xticklabels=["Return", "Revision", "Clean", "Noisy"])
            axes[row, col].grid(axis="y", alpha=.15)
            axes[row, col].set_axisbelow(True)
        axes[0, col].set(title=f"{model.title()}: final-branch prediction", ylabel="Brier AUC x100\n(lower is better)")
        axes[1, col].set(title="Change in still-valid old predictions",
                         ylabel="Valid-old Brier change x100\n(positive is damage)")
        axes[1, col].axhline(0, color="#939EAB", linewidth=.8)
        axes[2, col].set(title="Paired effect of corrupted feedback", xticks=range(len(arms)),
                         xticklabels=["Continue", "Reset Adam", "Reset replay", "Reset both"],
                         ylabel="Noisy minus clean Brier AUC x100\n(positive is harm from noise)")
        axes[2, col].axhline(0, color="#939EAB", linewidth=.8)
    fig.legend(handles=handles, labels=[LABELS[a] for a in arms], loc="upper left",
               bbox_to_anchor=(.055, .925), ncol=4, frameon=False)
    fig.text(.06, .035, f"Top and middle: means over {len(result['seeds'])} seeds. Compare interventions within each branch; evaluation subsets differ between branches.\n"
        "Bottom: each paired seed, mean diamond, and descriptive 95% bootstrap interval. Clean/noisy observations and actions match.\n"
        "Retention probes use clean support. Noisy prediction curves include corrupted feedback in the learner's recent history.",
        fontsize=9, color="#556579")
    save(fig, directory/"final_branches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
    directory = Path(args.input)
    plot_development(directory)
    plot_comparison(directory)
    plot_final_branches(directory)
