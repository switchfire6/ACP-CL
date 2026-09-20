"""Plot every recurring v3 contrast and seed from the audited compact archive."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "er_v3": "Replay (no recycling)",
    "newborn_v3": "Newborn gain (primary AUC contrast)",
    "newborn_matched_v3": "Newborn gain, nominal budget matched",
    "protection_v3": "Longer protection",
    "consolidation_v3": "Local consolidation",
    "full_v3": "All local mechanisms",
}


def plot(archive: Path, output: Path) -> None:
    diagnostics = json.loads((archive / "diagnostics.json").read_text(encoding="utf-8"))
    if diagnostics["runs"] != 51:
        raise ValueError("the complete audited 51-run cohort is required")
    results = json.loads((archive / "recurring" / "individual_results.json").read_text(encoding="utf-8"))
    seeds = diagnostics["selection"]["evaluation_seeds"]
    expected = {(method, seed) for method in ["recycle_v3", *METHODS] for seed in seeds}
    observed = {(r["method"], r["seed"]): r for r in results}
    if len(results) != len(observed) or set(observed) != expected or len(seeds) != 3:
        raise ValueError("missing, duplicate, or unexpected method/seed results")
    source = diagnostics["selection"]["common_identity"]["source_sha256"]
    if any(r["source_sha256"] != source for r in results):
        raise ValueError("archive source differs from the frozen study")

    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), sharey=True, layout="constrained")
    offsets = np.linspace(-0.15, 0.15, len(seeds))
    colors = ("#0072B2", "#E69F00", "#009E73")
    rows = []
    for axis, endpoint, title in zip(
        axes,
        ("late_early_auc", "final_accuracy"),
        ("Late acquisition AUC", "Final accuracy"),
    ):
        for index, method in enumerate(METHODS):
            values = []
            for seed, offset, color in zip(seeds, offsets, colors):
                candidate = float(observed[method, seed]["metrics"][endpoint])
                baseline = float(observed["recycle_v3", seed]["metrics"][endpoint])
                if not np.isfinite([candidate, baseline]).all():
                    raise ValueError("nonfinite endpoint")
                difference = 100 * (candidate - baseline)
                values.append(difference)
                axis.scatter(difference, index + offset, color=color, s=38,
                             label=f"Seed {seed}" if index == 0 else None, zorder=3)
                rows.append({"endpoint": endpoint, "method": method, "seed": seed,
                             "candidate": candidate, "recycling": baseline,
                             "difference_pp": difference})
            axis.scatter(np.mean(values), index, color="black", marker="|", s=180,
                         linewidth=2.2, label="Three-seed mean" if index == 0 else None, zorder=4)
        axis.axvline(0, color="#777777", linewidth=1, linestyle="--", zorder=1)
        axis.axhspan(0.65, 1.35, color="#f6f2d8", zorder=0)
        axis.set(title=title, xlabel="Difference from replay + recycling (percentage points)")
        axis.grid(axis="x", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(METHODS)), list(METHODS.values()))
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle("V3 recurring domains: paired differences on all three new seeds\n"
                    "Primary endpoint: newborn gain minus recycling on late AUC; other contrasts are exploratory",
                    fontsize=12)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "paired_contrasts.png", dpi=170)
    figure.savefig(output / "paired_contrasts.svg")
    plt.close(figure)
    svg = output / "paired_contrasts.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")
    with (output / "paired_contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("reports/v3"))
    parser.add_argument("--output", type=Path, default=Path("reports/v3"))
    args = parser.parse_args()
    plot(args.archive, args.output)
