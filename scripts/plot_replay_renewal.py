"""Portable figures from the sealed replay-renewal summaries."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


COLORS = {"uniform": "#315F89", "recent": "#C0723C", "split": "#148477"}


def save(fig, path):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix("."+extension), dpi=170, facecolor="white")
    plt.close(fig)


def forest(ax, entries, scale=100):
    for i, (_, metric) in enumerate(entries):
        ax.plot(scale*np.array([metric["lower"], metric["upper"]]), [i, i], color="#334A60", lw=2)
        ax.scatter(scale*np.array(metric["differences"]), i+np.linspace(-.10, .10, metric["n"]),
                   s=23, facecolors="white", edgecolors="#879EAE", zorder=3)
        ax.scatter([scale*metric["mean"]], [i], color="#123B59", marker="D", s=32, zorder=4)
    ax.axvline(0, color="#A5AFB9", lw=1)
    ax.set_yticks(range(len(entries)), [name for name, _ in entries])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.spines[["top", "right"]].set_visible(False)


def decomposition(directory):
    path = directory/"decomposition/summary.json"
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads((path.parent/"config.json").read_text(encoding="utf-8"))
    title = "Engineering check" if "smoke" in config["study"] else "Adam"
    names = [
        ("clear_minus_keep", "Delete packets / keep age"),
        ("clear_rebase_minus_rebase", "Delete packets / rebase age"),
        ("rebase_minus_keep", "Rebase age / keep packets"),
        ("clear_rebase_minus_clear", "Rebase age / delete packets"),
        ("rng_reset_minus_keep", "Reset RNG / keep packets + age"),
        ("full_reset_minus_clear_rebase", "Reset RNG / clear + rebase"),
        ("full_reset_minus_keep", "Reset all replay components"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(left=.25, right=.97, top=.82, bottom=.10, wspace=.85, hspace=.48)
    fig.suptitle(f"{title} | What does a replay reset actually change?", x=.04, y=.98,
                 ha="left", fontsize=20, weight="bold")
    fig.text(.04, .927, f"First new dependency only · {len(result['seeds'])} paired seeds per architecture · identical starting weights and optimizer",
             fontsize=11, color="#536477")
    fig.text(.04, .883, "Circles: individual seeds   Diamonds: paired means   Lines: descriptive 95% bootstrap intervals",
             fontsize=10, color="#536477")
    for row, field in enumerate(("brier_auc", "valid_after")):
        for col, model in enumerate(("conditional", "recurrent")):
            ax = axes[row, col]
            forest(ax, [(label, result["contrasts"][model][name][field]) for name, label in names])
            ax.set_title(f"{model.title()}: {'new learning' if row == 0 else 'still-valid old knowledge'}", pad=12)
            ax.set_xlabel("Brier AUC difference ×100" if row == 0 else "Final Brier difference ×100")
    fig.text(.04, .027, "Negative values mean lower error. These local reset contrasts do not test an autonomous renewal policy or repeated resets.",
             fontsize=10, color="#536477")
    save(fig, directory/"decomposition_effects")


def policies(directory):
    path = directory/"policy/summary.json"
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads((path.parent/"config.json").read_text(encoding="utf-8"))
    capacity, recent = config["memory_packets"], config["recent_packets"]
    title = "Engineering check" if "smoke" in config["study"] else "Adam"
    labels = {"uniform": f"Uniform history ({capacity})", "recent": f"Recent only ({capacity})",
              "split": f"Recent {recent} + historical {capacity-recent}"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(left=.075, right=.97, top=.79, bottom=.12, wspace=.34, hspace=.48)
    fig.suptitle(f"{title} | Learning with two memory timescales", x=.04, y=.98,
                 ha="left", fontsize=21, weight="bold")
    fig.text(.04, .928, f"Policies run from initialization; no change-triggered reset. Same {capacity}-packet budget, data, and update count.",
             fontsize=11, color="#536477")
    handles = []
    for row, model in enumerate(("conditional", "recurrent")):
        for col, field in enumerate(("brier_auc", "valid_after")):
            ax = axes[row, col]
            for arm, color in COLORS.items():
                values = [result["stage_means"][model][str(k)][arm][field]*100 for k in (1, 2, 3)]
                line, = ax.plot([1, 2, 3], values, marker="o", color=color, label=labels[arm], lw=2)
                if row == col == 0:
                    handles.append(line)
                for stage in (1, 2, 3):
                    seeds = [r[field]*100 for r in result["rows"] if r["model"] == model and r["arm"] == arm
                             and r["branch"] == "novel" and r["stage"] == stage]
                    ax.scatter(stage+np.linspace(-.065, .065, len(seeds)), seeds, s=11, color=color, alpha=.35)
            ax.set(title=f"{model.title()}: {'acquisition' if col == 0 else 'retention'}",
                xticks=[1, 2, 3], xlabel="Successive new dependency",
                ylabel="Prediction Brier AUC ×100" if col == 0 else "Final valid-old Brier ×100")
        ax = axes[row, 2]
        for j, (arm, color) in enumerate(COLORS.items()):
            d = result["noise_minus_clean"][model][arm]["brier_auc"]
            ax.bar(j, d["mean"]*100, color=color, alpha=.9, width=.65)
            ax.plot([j, j], [d["lower"]*100, d["upper"]*100], color="#273D54", lw=2)
            ax.scatter(j+np.linspace(-.1, .1, d["n"]), np.array(d["differences"])*100,
                s=18, facecolors="white", edgecolors="#273D54", zorder=4)
        ax.axhline(0, color="#A5AFB9", lw=1)
        ax.set(title=f"{model.title()}: noise penalty", xticks=[0, 1, 2], xticklabels=["Uniform", "Recent", "Split"],
            ylabel="Noisy − clean Brier AUC ×100", xlabel="Matched input and action streams")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, .89), ncol=3, frameon=False, fontsize=11)
    fig.text(.04, .035, f"Lower error and smaller noise penalties are better. Dots show all {len(result['seeds'])} seeds; later policies have different learned weights.\n"
             "The fixed split must satisfy acquisition, retention, return, revision, noise, and survival requirements separately for each architecture.",
             fontsize=10, color="#536477")
    save(fig, directory/"continuous_memory")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=.16, right=.97, top=.80, bottom=.13, wspace=.47, hspace=.44)
    fig.suptitle(f"{title} | Does the fixed {recent} + {capacity-recent} memory pass?", x=.045, y=.97,
                 ha="left", fontsize=19, weight="bold")
    outcomes = "   |   ".join(f"{m.title()}: {'PASS' if s['passed'] else 'FAIL'}" for m, s in result["screen"].items())
    fig.text(.045, .912, outcomes, fontsize=14, weight="bold", color="#334A60")
    fig.text(.045, .867, "Split minus uniform · negative means lower error · red ticks mark the maximum allowed mean", fontsize=10, color="#536477")
    for row, field in enumerate(("brier_auc", "valid_after")):
        for col, model in enumerate(("conditional", "recurrent")):
            entries = [("New dependencies", result["contrasts"][model]["split_minus_uniform"][field])]
            entries += [(b.title(), result["final_contrasts"][b][model]["split_minus_uniform"][field])
                        for b in ("return", "revision", "clean", "noise")]
            forest(axes[row, col], entries)
            for i in range(len(entries)):
                limit = -.2 if i == 0 and row == 0 else .5
                axes[row, col].plot([limit, limit], [i-.24, i+.24], color="#B24747", lw=2)
            axes[row, col].set(title=f"{model.title()}: {'prediction' if row == 0 else 'valid-old retention'}",
                xlabel="Brier difference ×100")
    fig.text(.045, .035, f"{len(result['seeds'])} paired seeds, not independent episodes. Acquisition averages three introductions within seed.\n"
             "The full screen also requires qualification, ≥5/6 acquisition improvements, matched noise and survival limits. Passing is only a pilot result.",
             fontsize=10, color="#536477")
    save(fig, directory/"policy_screen")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    root = Path(args.input)
    decomposition(root)
    policies(root)
