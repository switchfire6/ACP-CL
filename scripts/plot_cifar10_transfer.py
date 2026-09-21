"""Plot the complete 16-run CIFAR-10 transfer pilot from its compact archive.

The companion analyzer owns the detailed source, checkpoint, allocation, and
arrival audit. This exporter requires the exact complete cohort, coherent
recorded identities, and valid plotted series. It never reads ignored raw runs,
loads checkpoints, computes confidence intervals, or interprets pilot success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("recurring", "stationary")
SEEDS = (1063, 1174)
METHODS = ("er_v3", "recycle_v3", "newborn_v3", "newborn_matched_v3")
GAIN_METHODS = METHODS[2:]
LABELS = {
    "er_v3": "Replay",
    "recycle_v3": "Replay + recycling",
    "newborn_v3": "Newborn gain",
    "newborn_matched_v3": "Newborn gain (nominal mean matched)",
}
COLORS = {
    "er_v3": "#777777",
    "recycle_v3": "#0072B2",
    "newborn_v3": "#D55E00",
    "newborn_matched_v3": "#009E73",
}
MARKERS = ("o", "s")
N_EXPERIENCES = 30
EARLY_EXAMPLES = 256
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _unit_value(value, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{description} must be a finite number in [0,1]")
    return float(value)


def validate_records(records: list) -> dict[tuple[str, str, int], dict]:
    """Validate only the complete archive and the quantities this plot uses."""
    if not isinstance(records, list) or len(records) != 16:
        raise ValueError("the complete 16-run archive is required; partial cohorts are not plotted")
    expected = {(condition, method, seed) for condition in CONDITIONS
                for method in METHODS for seed in SEEDS}
    observed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every archived result must be an object")
        if type(record.get("seed")) is not int:
            raise ValueError("archive seed identities must be integers")
        key = (record.get("condition"), record.get("method"), record["seed"])
        if key not in expected or key in observed:
            raise ValueError(f"duplicate or unexpected condition/method/seed: {key}")
        if record.get("eval_split") != "validation":
            raise ValueError(f"the transfer pilot uses validation outcomes: {key}")
        if record.get("evaluation_schedule", {}).get("early_examples") != EARLY_EXAMPLES:
            raise ValueError(f"early AUC must use the fixed 256-example horizon: {key}")
        for field in ("source_sha256", "runtime_sha256", "config_sha256"):
            value = record.get(field)
            if not isinstance(value, str) or _HASH.fullmatch(value) is None:
                raise ValueError(f"missing or malformed {field}: {key}")
        if not isinstance(record.get("execution_device"), str) or not record["execution_device"]:
            raise ValueError(f"missing execution_device: {key}")
        auc = record.get("early_auc")
        if not isinstance(auc, list) or len(auc) != N_EXPERIENCES:
            raise ValueError(f"all 30 early AUC observations are required: {key}")
        values = [_unit_value(value, f"early_auc[{i}] for {key}") for i, value in enumerate(auc)]
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"missing endpoint metrics: {key}")
        late = _unit_value(metrics.get("late_early_auc"), f"late AUC for {key}")
        final = _unit_value(metrics.get("final_accuracy"), f"final accuracy for {key}")
        if not math.isclose(late, statistics.fmean(values[N_EXPERIENCES // 2:]),
                            rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"late AUC differs from the stored experience-16-to-30 series: {key}")
        matrix = record.get("accuracy_matrix")
        if not isinstance(matrix, list) or len(matrix) != N_EXPERIENCES \
                or any(not isinstance(row, list) or len(row) != N_EXPERIENCES for row in matrix):
            raise ValueError(f"a complete 30-by-30 accuracy matrix is required: {key}")
        final_values = [_unit_value(value, f"final matrix row for {key}") for value in matrix[-1]]
        if not math.isclose(final, statistics.fmean(final_values), rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"final accuracy differs from the complete final matrix row: {key}")
        observed[key] = record
    if set(observed) != expected:
        raise ValueError(f"missing pilot results: {sorted(expected - set(observed))}")
    for field in ("source_sha256", "runtime_sha256", "execution_device"):
        if len({record[field] for record in records}) != 1:
            raise ValueError(f"the archive mixes {field} identities")
    for condition in CONDITIONS:
        if len({record["config_sha256"] for key, record in observed.items() if key[0] == condition}) != 1:
            raise ValueError(f"the archive mixes configuration identities within {condition}")
    return observed


def paired_differences(records: dict, condition: str, method: str, endpoint: str) -> np.ndarray:
    """Return candidate-minus-recycling differences, paired by seed, in pp."""
    return 100 * np.asarray([
        records[condition, method, seed]["metrics"][endpoint]
        - records[condition, "recycle_v3", seed]["metrics"][endpoint]
        for seed in SEEDS
    ], dtype=float)


def _paired_panel(axis, records: dict, groups: list, endpoint: str, extent: float) -> None:
    """Render observed seed points and their mean; no uncertainty interval."""
    for group_index, (_, condition, methods) in enumerate(groups):
        offsets = (0.0,) if len(methods) == 1 else (-0.18, 0.18)
        for method, method_offset in zip(methods, offsets, strict=True):
            values = paired_differences(records, condition, method, endpoint)
            center = group_index + method_offset
            for value, seed_offset, marker in zip(values, (-0.065, 0.065), MARKERS, strict=True):
                axis.scatter(value, center + seed_offset, color=COLORS[method], marker=marker,
                             s=54, linewidths=0.7, edgecolors="white", zorder=4)
            axis.scatter(values.mean(), center, color="#222222", marker="|", s=125,
                         linewidths=1.8, zorder=5)
    axis.axvline(0, color="#666666", linestyle="--", linewidth=1, zorder=1)
    axis.set(yticks=range(len(groups)), yticklabels=[group[0] for group in groups],
             ylim=(len(groups) - 0.5, -0.5), xlim=(-extent, extent),
             xlabel="Difference from replay + recycling (pp)")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    axis.tick_params(axis="y", length=0, pad=9)
    axis.spines[["top", "right", "left"]].set_visible(False)


def _figure(records: dict):
    figure, axes = plt.subplots(2, 2, figsize=(14, 9.4))
    figure.subplots_adjust(left=0.15, right=0.975, bottom=0.22, top=0.815,
                           hspace=0.53, wspace=0.33)
    experiences = np.arange(1, N_EXPERIENCES + 1)
    for column, condition in enumerate(CONDITIONS):
        axis = axes[0, column]
        for method in METHODS:
            values = np.asarray([records[condition, method, seed]["early_auc"] for seed in SEEDS]) * 100
            for row, linestyle in zip(values, ("-", "--"), strict=True):
                axis.plot(experiences, row, color=COLORS[method], alpha=0.22,
                          linewidth=0.9, linestyle=linestyle, zorder=2)
            axis.plot(experiences, values.mean(axis=0), color=COLORS[method], linewidth=2.0,
                      label=LABELS[method], zorder=3)
        axis.set(xlim=(1, N_EXPERIENCES), ylim=(0, 100), xlabel="Experience",
                 xticks=[1, 5, 10, 15, 20, 25, 30], yticks=np.arange(0, 101, 20))
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.set_ylabel("Early adaptation AUC (%)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        title = "(a) Recurring domains" if condition == "recurring" else "(b) Stationary original images"
        axis.set_title(title, loc="left", fontsize=12, pad=12)

    all_differences = np.concatenate([
        paired_differences(records, condition, method, endpoint)
        for condition, endpoint in (("recurring", "late_early_auc"),
                                    ("recurring", "final_accuracy"),
                                    ("stationary", "final_accuracy"))
        for method in GAIN_METHODS
    ])
    extent = max(0.25, 1.28 * float(np.abs(all_differences).max()))
    _paired_panel(axes[1, 0], records, [
        ("Newborn gain", "recurring", ("newborn_v3",)),
        ("Newborn gain\n(nominal mean matched)", "recurring", ("newborn_matched_v3",)),
    ], "late_early_auc", extent)
    _paired_panel(axes[1, 1], records, [
        ("Recurring", "recurring", GAIN_METHODS),
        ("Stationary", "stationary", GAIN_METHODS),
    ], "final_accuracy", extent)
    axes[1, 0].set_title("(c) Recurring: late AUC differences", loc="left", fontsize=12, pad=12)
    axes[1, 1].set_title("(d) Final validation accuracy differences", loc="left", fontsize=12, pad=12)
    figure.suptitle("CIFAR-10 transfer pilot", fontsize=17, y=0.97)
    figure.text(0.5, 0.925, "16 complete runs | 2 paired seeds | descriptive development results",
                ha="center", fontsize=11, color="#444444")
    method_handles = [Line2D([], [], color=COLORS[method], linewidth=2, label=LABELS[method])
                      for method in METHODS]
    figure.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 0.901),
                  ncol=4, frameon=False, fontsize=9.5, handlelength=2.4, columnspacing=1.7)
    seed_handles = [Line2D([], [], linestyle="none", marker=marker, color="#333333",
                          markerfacecolor="none", markersize=7, label=f"Seed {seed}")
                    for seed, marker in zip(SEEDS, MARKERS, strict=True)]
    seed_handles.append(Line2D([], [], linestyle="none", marker="|", color="#222222",
                               markersize=12, markeredgewidth=1.8, label="Two-seed mean"))
    figure.legend(handles=seed_handles, loc="lower center", bbox_to_anchor=(0.5, 0.118),
                  ncol=3, frameon=False, fontsize=10, columnspacing=2.5)
    notes = (
        "Faint curves: individual seeds; thick curves: their mean. Paired points are observed seed differences; no intervals.",
        "AUC integrates accuracy over the first 256 current examples; late AUC averages experiences 16-30. Differences are percentage points (pp).",
        "Validation pools are reused and correlated. Nominal mean gain matching does not imply equal realized optimizer updates.",
    )
    for y, note in zip((0.095, 0.066, 0.037), notes, strict=True):
        figure.text(0.5, y, note, ha="center", va="center", fontsize=9, color="#444444")
    return figure


def plot(archive: Path, output: Path) -> dict:
    if archive.is_dir():
        archive = archive / "individual_results.json"
    raw = archive.read_bytes()
    records = validate_records(json.loads(raw))
    archive_hash = hashlib.sha256(raw).hexdigest()
    style = {"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
             "ytick.labelsize": 10, "svg.hashsalt": archive_hash, "svg.fonttype": "none"}
    with plt.rc_context(style):
        figure = _figure(records)
        try:
            output.mkdir(parents=True, exist_ok=True)
            description = "Complete 16-run, two-seed descriptive CIFAR-10 transfer pilot; validation only; no confidence intervals."
            figure.savefig(output / "overview.png", dpi=200,
                           metadata={"Title": "CIFAR-10 transfer pilot", "Description": description,
                                     "Creator": "ACP-CL plot_cifar10_transfer.py"})
            figure.savefig(output / "overview.svg",
                           metadata={"Title": "CIFAR-10 transfer pilot", "Description": description,
                                     "Creator": "ACP-CL plot_cifar10_transfer.py", "Date": None})
        finally:
            plt.close(figure)
    svg = output / "overview.svg"
    svg.write_bytes(("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n").encode("utf-8"))
    first = next(iter(records.values()))
    manifest = {
        "schema_version": 1, "input_archive": archive.name,
        "input_archive_sha256": archive_hash,
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": first["source_sha256"], "runtime_sha256": first["runtime_sha256"],
        "execution_device": first["execution_device"], "runs": 16,
        "conditions": list(CONDITIONS), "methods": list(METHODS), "seeds": list(SEEDS),
        "early_auc_horizon_current_examples": EARLY_EXAMPLES,
        "late_auc_experiences_inclusive": [16, 30],
        "curve_units": "percent", "contrast_units": "percentage points",
        "contrast_definition": "candidate minus recycle_v3 within each condition and seed",
        "uncertainty_intervals": "none; two-seed descriptive development pilot",
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in ("overview.png", "overview.svg")},
    }
    (output / "overview_plot.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "reports" / "cifar10_transfer" / "individual_results.json",
                        help="complete compact individual_results.json, or its parent directory")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "cifar10_transfer",
                        help="directory for overview.png, overview.svg, and overview_plot.json")
    args = parser.parse_args()
    try:
        manifest = plot(args.archive, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Exported complete {manifest['runs']}-run descriptive figure: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
