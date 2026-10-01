"""Plot the portable dual-path diagnostic archive without loading checkpoints."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(archive_path: Path, output: Path):
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    methods = archive["methods"]
    by_method = {m: [r for r in archive["runs"] if r["method"] == m] for m in methods}
    labels = [m.replace("dual_", "two-path\n").replace("_", " ") for m in methods]
    colors = plt.get_cmap("tab10")(np.arange(len(methods)))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for i, method in enumerate(methods):
        runs = by_method[method]
        final = [r["metrics"]["final_accuracy"] * 100 for r in runs]
        acquired = [np.mean([c[-1][1] for c in r["learning_curves"]]) * 100 for r in runs]
        for axis, values in ((axes[0, 0], final), (axes[0, 1], acquired)):
            axis.bar(i, np.mean(values), color=colors[i], alpha=.6)
            axis.scatter(i + np.linspace(-.07, .07, len(values)), values,
                         color=colors[i], edgecolor="black", s=32, zorder=3)
    for axis in axes[0]:
        axis.set(xticks=range(len(methods)), xticklabels=labels, ylabel="Accuracy (%)")
        axis.tick_params(axis="x", labelrotation=20, labelsize=8)
        axis.set_ylim(bottom=0)
    axes[0, 0].set_title("Final retention across all 100 classes")
    axes[0, 0].axhline(1, color="gray", linestyle=":", linewidth=1, label="Uniform chance: 1%")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Current-group accuracy after learning\nMean over ten experiences")
    for method in ("dual_joint", "dual_derpp", "dual_consolidate"):
        if method not in by_method:
            continue
        color = colors[methods.index(method)]
        late, retained = [], []
        for r in by_method[method]:
            curves = r["learning_curves"][len(r["learning_curves"]) // 2:]
            xs = np.asarray(curves[0])[:, 0]
            values = np.mean([np.asarray(c)[:, 1] for c in curves], axis=0) * 100
            late.append(values)
            axes[1, 0].plot(xs, values, color=color, alpha=.25, linewidth=1)
            matrix = np.asarray(r["accuracy_matrix"], dtype=float)
            ys = np.array([np.mean(matrix[j, :j + 1]) for j in range(len(matrix))]) * 100
            retained.append(ys)
            axes[1, 1].plot(np.arange(1, len(ys) + 1), ys, color=color, alpha=.25, linewidth=1)
        axes[1, 0].plot(xs, np.mean(late, axis=0), color=color, label=method, linewidth=2)
        axes[1, 1].plot(np.arange(1, len(retained[0]) + 1), np.mean(retained, axis=0),
                        color=color, label=method, linewidth=2)
    horizon = archive["config"].get("early_examples", 512)
    axes[1, 0].axvspan(0, horizon, color="gray", alpha=.12)
    axes[1, 0].set(title=f"Late acquisition curves (experiences 6-10)\nShaded: first {horizon} examples",
                   xlabel="Current examples presented", ylabel="Current-group accuracy (%)", ylim=(0, None))
    axes[1, 1].set(title="Retention as the evaluated class set grows\nNot a causal plasticity measure",
                   xlabel="Experience", ylabel="Seen-class mean accuracy (%)", ylim=(0, None))
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("From-scratch dual-path pilot: two seeds, descriptive results\nDots/thin curves: seeds; bars/thick curves: means; compute is not matched", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.archive, args.output)
