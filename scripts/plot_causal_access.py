"""Render already scored causal-access results; never edit evidence or decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


MODELS = ("conditional", "recurrent")
POLICIES = ("core", "full", "half", "causal", "joint")
COLORS = {"core": "#687e95", "full": "#62846b", "half": "#ac9567",
          "causal": "#216ba5", "joint": "#9676af", "reference": "#797979"}
CUES = ("Efficiency", "Delay", "Supply timing")


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#dce3e9", linewidth=.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#aab7c2")
    axis.tick_params(colors="#344454", length=3)
    axis.yaxis.set_major_locator(MaxNLocator(5))


def seed_points(axis, x, values, mean, color, spread=.13, bar=.21):
    offsets = np.linspace(-spread, spread, len(values)) if len(values) > 1 else [0.]
    axis.scatter(x + np.asarray(offsets), values, s=35, color=color, alpha=.80,
                 edgecolor="white", linewidth=.7, zorder=3)
    axis.plot([x-bar, x+bar], [mean, mean], color=color, linewidth=3,
              solid_capstyle="round", zorder=4)


def save_figure(figure, directory, stem):
    paths = []
    for extension in ("png", "svg"):
        path = directory / (stem + "." + extension)
        figure.savefig(path, dpi=170, facecolor="white")
        paths.append(path)
    plt.close(figure)
    return paths


def phase_figure(summary, directory):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")
    figure.subplots_adjust(left=.08, right=.98, top=.84, bottom=.13,
                           hspace=.48, wspace=.17)
    figure.suptitle("How the five fixed prediction policies performed", x=.08,
                   y=.97, ha="left", fontsize=18, weight="bold", color="#203143")
    subtitle = "Frozen switching stream · all transitions included · lower error is better"
    if not summary["development"]:
        subtitle = "ENGINEERING SMOKE — not scientific results · " + subtitle
    figure.text(.08, .925, subtitle, fontsize=10.5, color="#526270")
    labels = {"whole": "Whole stream", "old": "Old conditions", "novel": "Novel conditions"}
    for row, model in enumerate(MODELS):
        records = sorted((item for item in summary["per_seed"] if item["model"] == model),
                         key=lambda item: item["seed"])
        item = summary["models"][model]
        maximum = max(record["stream"][part][policy]["brier"]
                      for record in records for part in labels for policy in POLICIES)
        for column, part in enumerate(labels):
            axis = axes[row, column]
            style_axis(axis)
            axis.axvspan(2.6, 3.4, color=COLORS["causal"], alpha=.055, zorder=0)
            for index, policy in enumerate(POLICIES):
                values = [record["stream"][part][policy]["brier"] for record in records]
                mean = item["levels"][part][policy]["brier"]["mean"]
                seed_points(axis, index, values, mean, COLORS[policy])
            axis.set_xticks(range(5), [policy.title() for policy in POLICIES])
            axis.set_xlim(-.5, 4.5)
            axis.set_ylim(0, max(.10, maximum * 1.13))
            axis.set_title(labels[part], fontsize=12, loc="left", pad=10)
            if column == 0:
                axis.set_ylabel(model.title() + "\nAll-case Brier", fontsize=11)
                status = ("Qualified" if item["qualified"] else "Qualification failed")
                online = "pass" if item["raw_primary_passed"] else "fail"
                axis.text(0, 1.17, f"{model.title()}  |  {status}; raw online screen: {online}",
                          transform=axis.transAxes, fontsize=10, color="#526270")
    figure.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color="#687e95", markersize=5,
               label="One fresh seed"),
        Line2D([], [], color="#687e95", linewidth=3, label="Seed mean")],
        loc="lower left", bbox_to_anchor=(.075, .043), frameon=False, ncol=2, fontsize=10)
    figure.text(.08, .023,
                "All-case means shown here. The scientific decision also checks affected-case "
                "error, cue use, absolute competence and survival.", fontsize=9, color="#526270")
    return save_figure(figure, directory, "policy_brier")


def grouped_points(axis, records, groups, series):
    offsets = np.linspace(-.22, .22, len(series))
    for cue in range(3):
        selected = [record for record in records if record["cue"] == cue]
        if not selected:
            continue
        for offset, (key, label, color, reader, mean_reader) in zip(offsets, series, strict=True):
            values = [reader(record) for record in selected]
            mean = mean_reader(groups, cue)
            seed_points(axis, cue+offset, values, mean, color, spread=.045, bar=.075)
    axis.set_xticks(range(3), CUES)
    axis.set_xlim(-.55, 2.55)
    return [Line2D([], [], color=color, linewidth=3, label=label)
            for key, label, color, reader, mean_reader in series]


def qualification_figure(summary, directory):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    figure.subplots_adjust(left=.085, right=.98, top=.79, bottom=.15,
                           hspace=.52, wspace=.24)
    figure.suptitle("Are useful predictive functions available?", x=.085, y=.97,
                   ha="left", fontsize=18, weight="bold", color="#203143")
    subtitle = "Selected competence and cue checks · dots show seeds; bars show cue-group means"
    if not summary["development"]:
        subtitle = "ENGINEERING SMOKE — not scientific results · " + subtitle
    figure.text(.085, .925, subtitle, fontsize=10.3, color="#526270")
    excess_series = [
        ("core_old_excess", "Old core: all cases", COLORS["core"]),
        ("full_novel_focus_excess", "Novel full: affected cases", COLORS["full"])]
    cue_series = [
        ("specialist_novel_cue_gain", "Specialist · fresh support", COLORS["reference"]),
        ("full_novel_cue_gain", "Full · fresh support", COLORS["full"]),
        ("causal", "Causal · actual stream", COLORS["causal"])]
    handles_top = handles_bottom = None
    for column, model in enumerate(MODELS):
        records = sorted((item for item in summary["per_seed"] if item["model"] == model),
                         key=lambda item: item["seed"])
        groups = summary["models"][model]
        for axis in axes[:, column]:
            style_axis(axis)
            axis.axhline(0, color="#9eaab5", linewidth=.7, zorder=1)
        axis = axes[0, column]
        series = [(key, label, color,
                   lambda record, key=key: record["qualification"][key],
                   lambda group, cue, key=key: group["qualification"][f"cue_{cue}"]["metrics"][key])
                  for key, label, color in excess_series]
        handles_top = grouped_points(axis, records, groups, series)
        axis.axhline(.02, color="#a45148", linestyle="--", linewidth=1.4)
        axis.text(.02, 1.02, "Allowed excess: at most +0.020", transform=axis.transAxes,
                  fontsize=9.5, color="#90483f")
        axis.set_title(model.title(), loc="left", fontsize=13, pad=28)
        axis.set_ylabel("Brier above the relevant specialist\nLower is better", fontsize=10)
        low, high = axis.get_ylim()
        axis.set_ylim(min(low, -.008), max(high, .035))

        axis = axes[1, column]
        series = []
        for key, label, color in cue_series:
            def reader(record, key=key):
                return (record["cue_gain"]["causal"] if key == "causal"
                        else record["qualification"][key])

            def mean_reader(group, cue, key=key):
                return (group["cue_guard"][f"cue_{cue}"]["mean"] if key == "causal"
                        else group["qualification"][f"cue_{cue}"]["metrics"][key])

            series.append((key, label, color, reader, mean_reader))
        handles_bottom = grouped_points(axis, records, groups, series)
        axis.axhline(.002, color="#a45148", linestyle="--", linewidth=1.4)
        axis.text(.02, 1.02, "Required benefit: at least +0.002", transform=axis.transAxes,
                  fontsize=9.5, color="#90483f")
        axis.set_ylabel("Benefit from the correct novel cue\nHigher is better", fontsize=10)
        low, high = axis.get_ylim()
        axis.set_ylim(min(low, -.004), max(high, .012))
    figure.legend(handles=handles_top, loc="upper left", bbox_to_anchor=(.08, .895),
                  frameon=False, ncol=2, fontsize=10)
    figure.legend(handles=handles_bottom, loc="lower left", bbox_to_anchor=(.08, .064),
                  frameon=False, ncol=3, fontsize=10)
    figure.text(.085, .044,
                "Fresh support is a privileged diagnostic and never enters the online mixer. "
                "Cue benefit is flipped minus correct affected-case Brier.", fontsize=9, color="#526270")
    figure.text(.085, .022,
                "Qualification additionally requires adequate specialist error and gains over "
                "a training-only marginal predictor; these checks are in the saved summary.",
                fontsize=9, color="#526270")
    return save_figure(figure, directory, "availability_cues")


def render(summary_path, output):
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if set(summary["models"]) != set(MODELS) or not summary["per_seed"]:
        raise ValueError("Both architectures and recorded per-seed metrics are required")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#344454", "svg.fonttype": "none"}):
        return phase_figure(summary, directory) + qualification_figure(summary, directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.summary, args.output):
        print(path)
