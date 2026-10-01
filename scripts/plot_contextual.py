"""Export the locked comparison as standalone scientific figures."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


COLORS = dict(pooled="#89929C", conditional="#2366AB", recurrent="#00897E",
              query_routed="#9746B5", recurrent_frozen="#C38820", oracle="#243044")
LABELS = dict(pooled="Pooled", conditional="Original conditional", recurrent="Recurrent",
              query_routed="Query routing", recurrent_frozen="Frozen features", oracle="Oracle (privileged)")


def plot(directory):
    directory = Path(directory)
    with gzip.open(directory/"archive.json.gz", "rt", encoding="utf-8") as handle:
        archive = json.load(handle)
    summary, config = archive["summary"], archive["manifest"]["config"]
    records = {(r["seed"], r["gap"], r["method"], r["branch"]): r for r in archive["records"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.labelcolor": "#344254", "text.color": "#172536", "axes.edgecolor": "#CAD0D7"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(top=.83, bottom=.15, hspace=.50, wspace=.35)
    fig.suptitle("Learning what changes, retaining what still applies", x=.065, y=.985,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(.065, .942, f"Locked matched-history pilot  |  {len(config['seeds'])} seeds  |  "
             "two gap orders per seed  |  equal arrivals and update counts", color="#586577", fontsize=11)
    handles = []
    for axis, branch, title in ((axes[0, 0], "novel", "A  Learning a new visible dependency"),
                                (axes[1, 1], "noise", "D  Noisy feedback; unchanged physics")):
        for method in config["methods"]:
            x = np.array([p["arrivals"] for p in records[config["seeds"][0], config["gaps"][0], method, branch]["curve"]])
            y = np.array([[np.mean([records[s, g, method, branch]["curve"][i]["metrics"]["survival"]
                for g in config["gaps"]]) for i in range(len(x))] for s in config["seeds"]])
            line, = axis.plot(x, y.mean(axis=0), color=COLORS[method], label=LABELS[method],
                linewidth=2.1 if method in ("query_routed", "recurrent", "conditional") else 1.3,
                linestyle="--" if method in ("oracle", "recurrent_frozen") else "-")
            if branch == "novel":
                handles.append(line)
        axis.set(title=title, xlabel="New observed outcomes", ylabel="Joint survival through H=12")
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.set_xticks(x)
        axis.grid(axis="y", alpha=.2)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.058, .916), ncol=3, frameon=False)

    axis = axes[0, 1]
    contrasts = summary["contrasts"]["novel"]
    for i, comparator in enumerate(("recurrent", "conditional")):
        values = contrasts[f"query_routed_minus_{comparator}"]["acquisition_auc"]
        points = np.array(values["differences"])*100
        axis.scatter(points, i+np.linspace(-.10, .10, len(points)), s=32,
                     facecolors="white", edgecolors=COLORS[comparator], zorder=3)
        axis.plot(np.array([values["lower"], values["upper"]])*100, [i, i], color="#25384D", linewidth=2)
        axis.scatter([100*values["mean"]], [i], color="#25384D", marker="D", s=40, zorder=4)
    axis.axvline(0, color="#687789", linewidth=1)
    axis.axvline(2, color="#9746B5", linestyle=":", linewidth=1)
    axis.set(title="B  Query-routing gain in new-learning AUC",
             xlabel="Paired difference (percentage points)", yticks=[0, 1],
             yticklabels=["vs recurrent", "vs original conditional"], ylim=(-.45, 1.45))
    axis.text(.02, .04, "Circles: seeds; diamond: mean; line: 95% paired interval\n"
              "Dotted line: declared +2 pp practical threshold", transform=axis.transAxes,
              fontsize=8, color="#586577")

    axis = axes[1, 0]
    for i, method in enumerate(config["methods"]):
        x = summary["aggregate"]["return"][method]["before"]
        y = summary["aggregate"]["novel"][method]["known_after"]
        axis.plot([x, y], [i, i], color=COLORS[method], alpha=.5)
        axis.scatter(x, i, facecolors="white", edgecolors=COLORS[method], s=65, zorder=3)
        axis.scatter(y, i, color=COLORS[method], marker="D", s=30, zorder=4)
    axis.margins(x=.1, y=.12)
    axis.set(title="C  Reuse and retention of valid old knowledge",
        xlabel="Joint survival (open: return; filled: after new learning)",
        yticks=range(len(config["methods"])), yticklabels=[LABELS[m] for m in config["methods"]])
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.grid(axis="x", alpha=.2)
    fig.text(.065, .032, "Means average gap orders within seed. Query outcomes are disjoint from feedback. "
        "Intervals are descriptive; this is one synthetic world.\n"
        f"Prespecified comparison decision: {summary['decision'].upper()}. "
        "See the full report for adequacy checks, all controls, and limitations.", fontsize=9, color="#586577")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(directory/f"overview.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    plot(args.input)
