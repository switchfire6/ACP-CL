"""Export separate paired contrasts for recurring shapes and CIFAR-10 studies.

Every seed in each archived cohort is displayed. The studies have different
tasks, designs, and seed cohorts; no pooled estimate, interval, or significance
test is computed. Only compact public archives are needed to regenerate this
figure, never checkpoints, downloaded images, or ignored training runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

if __package__:
    from .plot_cifar10_transfer import validate_records as validate_cifar
else:
    from plot_cifar10_transfer import validate_records as validate_cifar


ROOT = Path(__file__).resolve().parents[1]
SHAPE_SEEDS = (731, 842, 953)
CIFAR_SEEDS = (1063, 1174)
SHAPE_METHODS = ("er_v3", "recycle_v3", "newborn_v3", "newborn_matched_v3",
                 "protection_v3", "consolidation_v3", "full_v3")
VARIANTS = ("newborn_v3", "newborn_matched_v3")
ENDPOINTS = ("late_early_auc", "final_accuracy")
COLORS = {"newborn_v3": "#D55E00", "newborn_matched_v3": "#009E73"}
ROW_LABELS = ("Newborn gain", "Newborn gain\n(nominal mean matched)")
MARKERS = ("o", "s", "^")


def _probability(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return float(value)


def _shapes(records: list) -> dict:
    expected = {(method, seed) for method in SHAPE_METHODS for seed in SHAPE_SEEDS}
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("the full recurring-shapes archive of 21 results is required")
    observed = {}
    for record in records:
        key = (record["method"], record["seed"])
        if key not in expected or key in observed:
            raise ValueError(f"unexpected or duplicate shapes method/seed: {key}")
        if record.get("eval_split") != "validation" \
                or record.get("evaluation_schedule", {}).get("early_examples") != 256:
            raise ValueError("shapes must use the archived validation/256-presentation protocol")
        auc = record.get("early_auc")
        matrix = record.get("accuracy_matrix")
        if not isinstance(auc, list) or len(auc) != 100 or not isinstance(matrix, list) \
                or len(matrix) != 100 or any(not isinstance(row, list) or len(row) != 100 for row in matrix):
            raise ValueError("shapes require complete 100-experience outcomes")
        values = [_probability(value, "shapes AUC") for value in auc]
        final = [_probability(value, "shapes final row") for value in matrix[-1]]
        for endpoint, recomputed in (("late_early_auc", statistics.fmean(values[50:])),
                                     ("final_accuracy", statistics.fmean(final))):
            recorded = _probability(record["metrics"].get(endpoint), endpoint)
            if not math.isclose(recorded, recomputed, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"shapes {endpoint} disagrees with its stored series")
        observed[key] = record
    for field in ("source_sha256", "runtime_sha256", "execution_device", "config_sha256"):
        if any(not isinstance(record.get(field), str) or not record[field] for record in records) \
                or len({record[field] for record in records}) != 1:
            raise ValueError(f"shapes archive has missing or mixed {field}")
    return observed


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def prepare(shapes_path: Path, cifar_path: Path) -> dict:
    raw = {"shapes": shapes_path.read_bytes(), "cifar10": cifar_path.read_bytes()}
    shapes = _shapes(json.loads(raw["shapes"]))
    cifar_all = validate_cifar(json.loads(raw["cifar10"]))
    cifar = {(method, seed): result for (condition, method, seed), result in cifar_all.items()
             if condition == "recurring"}
    studies = {}
    for name, records, seeds, experiences, labels, domains in (
        ("shapes", shapes, SHAPE_SEEDS, 100, 4, 8),
        ("cifar10", cifar, CIFAR_SEEDS, 30, 10, 3),
    ):
        comparisons = []
        for endpoint in ENDPOINTS:
            for method in VARIANTS:
                differences = []
                for seed in seeds:
                    candidate = records[method, seed]["metrics"][endpoint]
                    reference = records["recycle_v3", seed]["metrics"][endpoint]
                    differences.append({"seed": seed, "candidate": candidate,
                                        "recycling": reference,
                                        "difference_pp": 100 * (candidate - reference)})
                comparisons.append({"endpoint": endpoint, "method": method,
                                    "comparator": "recycle_v3", "paired_seeds": differences,
                                    "mean_difference_pp": statistics.fmean(
                                        row["difference_pp"] for row in differences)})
        first = next(iter(records.values()))
        studies[name] = {
            "condition": "recurring", "seeds": list(seeds), "experiences": experiences,
            "labels": labels, "domains": domains,
            "late_auc_experiences_inclusive": [experiences // 2 + 1, experiences],
            "source_sha256": first["source_sha256"], "runtime_sha256": first["runtime_sha256"],
            "config_sha256": first["config_sha256"], "comparisons": comparisons,
        }
    return {
        "schema_version": 1, "purpose": "descriptive comparison of two distinct exploratory studies",
        "source_archives": {
            name: {"path": _path_label(path), "sha256": hashlib.sha256(raw[name]).hexdigest()}
            for name, path in (("shapes", shapes_path), ("cifar10", cifar_path))
        },
        "early_auc_horizon_current_presentations": 256,
        "contrast_units": "percentage points", "pooled_estimate": None,
        "uncertainty_intervals": None, "significance_tests": None,
        "studies": studies,
    }


def _comparison(study: dict, endpoint: str, method: str) -> dict:
    return next(row for row in study["comparisons"]
                if row["endpoint"] == endpoint and row["method"] == method)


def _figure(data: dict):
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))
    figure.subplots_adjust(left=0.19, right=0.97, bottom=0.27, top=0.755,
                           wspace=0.30, hspace=0.58)
    figure.suptitle("Temporary newborn gain: two exploratory studies", y=0.965, fontsize=17)
    figure.text(0.5, 0.917, "Paired differences from replay + recycling; each study shown separately",
                ha="center", fontsize=11, color="#444444")
    for column, (name, title, subtitle) in enumerate((
        ("shapes", "Procedural shapes", "4 labels | 8 domains | 100 experiences | 3 seeds"),
        ("cifar10", "CIFAR-10 transfer", "10 labels | 3 image views | 30 experiences | 2 seeds"),
    )):
        study = data["studies"][name]
        center = sum((axes[0, column].get_position().x0, axes[0, column].get_position().x1)) / 2
        figure.text(center, 0.851, title, ha="center", fontsize=13, weight="bold")
        figure.text(center, 0.819, subtitle, ha="center", fontsize=9.5, color="#444444")
        for row, endpoint in enumerate(ENDPOINTS):
            axis = axes[row, column]
            all_values = [item["difference_pp"] for other in data["studies"].values()
                          for method in VARIANTS
                          for item in _comparison(other, endpoint, method)["paired_seeds"]]
            extent = max(0.5, max(abs(value) for value in all_values) * 1.17)
            offsets = np.linspace(-0.17, 0.17, len(study["seeds"]))
            for method_index, method in enumerate(VARIANTS):
                comparison = _comparison(study, endpoint, method)
                for point, offset, marker in zip(comparison["paired_seeds"], offsets, MARKERS):
                    axis.scatter(point["difference_pp"], method_index + offset, color=COLORS[method],
                                 marker=marker, s=57, edgecolors="white", linewidths=0.75, zorder=4)
                axis.scatter(comparison["mean_difference_pp"], method_index, color="#222222",
                             marker="|", s=205, linewidths=2.4, zorder=5)
            axis.axvline(0, color="#666666", linestyle="--", linewidth=1, zorder=1)
            axis.set(yticks=(0, 1), yticklabels=ROW_LABELS, ylim=(1.48, -0.48),
                     xlim=(-extent, extent), xlabel="Difference from recycling (pp)")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.grid(axis="x", color="#dddddd", linewidth=0.6)
            axis.tick_params(axis="y", length=0, pad=8)
            axis.spines[["top", "right", "left"]].set_visible(False)
            title = "Late adaptation AUC" if endpoint == "late_early_auc" else "Final validation accuracy"
            axis.set_title(title, fontsize=11, loc="left", pad=9)
        handles = [Line2D([], [], color="#444444", linestyle="none", marker=marker,
                          markerfacecolor="none", markersize=7, label=str(seed))
                   for seed, marker in zip(study["seeds"], MARKERS)]
        handles.append(Line2D([], [], color="#222222", linestyle="none", marker="|",
                              markersize=12, markeredgewidth=2.4, label="Mean"))
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(center, 0.191),
                      ncol=len(handles), title="Paired seed points", title_fontsize=9,
                      fontsize=9, frameon=False, columnspacing=1.2, handletextpad=0.4)
    notes = (
        "AUC covers the first 256 current presentations; late AUC averages the latter half of each study's experiences.",
        "Different tasks, designs, and seed cohorts; exploratory validation results. No pooled estimate, significance test, or confidence interval.",
        "Matched means nominal feature gain is matched, not realized updates. All paired seeds are shown; axes are shared within each metric.",
    )
    for position, note in zip((0.105, 0.072, 0.039), notes, strict=True):
        figure.text(0.5, position, note, ha="center", fontsize=8.8, color="#444444")
    return figure


def plot(shapes_path: Path, cifar_path: Path, output: Path) -> dict:
    data = prepare(shapes_path, cifar_path)
    data["plot_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    salt = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9.5, "svg.hashsalt": salt, "svg.fonttype": "none"}):
        figure = _figure(data)
        try:
            output.mkdir(parents=True, exist_ok=True)
            metadata = {"Title": "Temporary newborn gain: two exploratory studies",
                        "Creator": "ACP-CL plot_research_overview.py",
                        "Description": "Separate paired contrasts versus recycling; all seeds; percentage points; no pooled estimate or confidence intervals."}
            figure.savefig(output / "research_overview.png", dpi=200, metadata=metadata)
            figure.savefig(output / "research_overview.svg", metadata={**metadata, "Date": None})
        finally:
            plt.close(figure)
    svg = output / "research_overview.svg"
    svg.write_bytes(("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n").encode())
    data["figure_files"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                            for name in ("research_overview.png", "research_overview.svg")}
    (output / "research_overview.json").write_bytes((json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes-archive", type=Path, default=ROOT / "reports/v3/recurring/individual_results.json")
    parser.add_argument("--cifar-archive", type=Path, default=ROOT / "reports/cifar10_transfer/individual_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    try:
        plot(args.shapes_archive, args.cifar_archive, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Exported separate exploratory study contrasts: {args.output / 'research_overview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
