"""Render the recorded controller trace without rerunning or modifying training."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot(run: Path, output: Path):
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    records = json.loads((run / "events.json").read_text(encoding="utf-8"))["events"]
    events = [e for e in records if e.get("controller")]
    if not events:
        raise ValueError("this run has no controller trace")
    colors = {"scaffold": "#c3c8d2", "open": "#c4e6d6", "closing": "#ffe0a3",
              "adult": "#d1dcf0", "reopened": "#ecc5c5"}
    steps = [e["step"] for e in events]
    figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for start, end, event in zip(steps, steps[1:]+[result["cost"]["current_examples"] //
                                                    result["config"]["batch_size"]], events):
        for axis in axes:
            axis.axvspan(start, max(start, end), color=colors[event["phase"]], alpha=0.35, linewidth=0)
    axes[0].plot(steps, [e["controller"]["novelty"] for e in events], color="#253f62", label="Novelty command")
    threshold = result["config"].get("controller", {}).get("open_threshold", 0.7)
    axes[0].axhline(threshold, linestyle="--", color="#a75529", label="Reopening threshold")
    axes[0].set(ylabel="Controller command", ylim=(0, 1))
    axes[0].legend(loc="upper right", fontsize=8)
    for block in events[0]["gates"]:
        axes[1].plot(steps, [e["gates"][block] for e in events], label=block, linewidth=1.3)
    axes[1].set(ylabel="Block update gate", ylim=(0, 1.05))
    axes[1].legend(loc="upper right", fontsize=8, ncol=3)
    axes[2].plot(steps, [e["mean_consolidation"] for e in events], color="#6e4892", label="Mean consolidation")
    axes[2].set(xlabel="Optimizer update", ylabel="Mean consolidation", ylim=(0, 1))
    recycle_steps = [e["step"] for e in records if e.get("recycled", 0)]
    axes[2].plot(recycle_steps, [0.025]*len(recycle_steps), "|", color="#1c6850", label="Recycling event")
    axes[2].legend(loc="upper right", fontsize=8)
    figure.legend(handles=[Patch(facecolor=color, label=phase) for phase, color in colors.items()],
                  loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle(f"{result['method']} · seed {result['seed']} · recorded development controller trace")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    plot(options.run, options.output)
