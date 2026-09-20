"""Verify and archive the complete v3 development grid, including a failed screen.

Only development outcomes bound by selection.json are read. The existing
selection validator checks all planned runs before any report is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

if __package__:
    from . import prepare_v3_study as study
    from .archive_results import archive
else:
    import prepare_v3_study as study
    from archive_results import archive


ROOT = Path(__file__).resolve().parents[1]


def validate_selection(selection_path: Path, results_root: Path) -> tuple[dict, bytes, list[dict]]:
    """Recompute the selection from complete inputs and verify their byte hashes.

    Paths are derived from the specification and actual suite layout rather than
    followed from untrusted artifact entries. Event hashes are additional archive
    provenance: the original selection binds results, manifests, and allocations.
    """
    raw = selection_path.read_bytes()
    artifact, selection_hash = study._load(selection_path)
    if hashlib.sha256(raw).hexdigest() != selection_hash:
        raise ValueError("selection artifact changed while being read")
    if artifact.get("schema_version") != 1 or artifact.get("status") not in (
        "selected", "no_candidate_passed"
    ):
        raise ValueError("a completed version-one selection artifact is required")
    if artifact.get("evaluation_outcomes_consumed") is not False:
        raise ValueError("selection must contain development outcomes only")
    spec, digest = artifact.get("specification"), artifact.get("study_spec_sha256")
    if not isinstance(spec, dict) or not isinstance(digest, str) or not study._HASH.fullmatch(digest):
        raise ValueError("selection is missing its embedded specification or specification hash")
    if artifact.get("development_seeds") != spec.get("development_seeds") or \
            artifact.get("evaluation_seeds") != spec.get("evaluation_seeds"):
        raise ValueError("selection seed identities disagree with the embedded specification")
    development = study._seed_list(spec.get("development_seeds"), "development_seeds")
    evaluation = study._seed_list(spec.get("evaluation_seeds"), "evaluation_seeds")
    if set(development) & set(evaluation):
        raise ValueError("development and evaluation seed identities overlap")

    records, identity = study._read_development(spec, digest, results_root)
    stored = artifact.get("input_results")
    if not isinstance(stored, list) or len(stored) != len(records) or \
            any(not isinstance(record, dict) for record in stored):
        raise ValueError("selection does not bind every complete development input")
    by_path = {record.get("result_path"): record for record in stored}
    if len(by_path) != len(stored) or set(by_path) != {r["result_path"] for r in records}:
        raise ValueError("selection has missing, duplicate, or unexpected development inputs")
    for record in records:
        expected = by_path[record["result_path"]]
        for field in ("manifest_sha256", "result_sha256", "allocation_sha256"):
            if expected.get(field) != record[field]:
                raise ValueError(f"selection input {field} mismatch: {record['result_path']}")
        if expected != record:
            raise ValueError(f"selection input metadata/metrics mismatch: {record['result_path']}")
    if artifact.get("common_identity") != identity:
        raise ValueError("selection source/runtime/device identity mismatch")

    candidates, selected = study._choose(spec, records)
    expected_status = "selected" if selected is not None else "no_candidate_passed"
    if artifact.get("candidates") != candidates or artifact.get("selected_feature_gain") != selected or \
            artifact["status"] != expected_status:
        raise ValueError("selection outcome does not match its complete development inputs")
    cadence = spec["base_config"]["retention_eval_every_experiences"]
    rule = artifact.get("selection_rule", {})
    expected_rule = {
        "stationary_checkpoint_experiences": list(range(cadence, spec["development_experiences"] + 1, cadence)),
        "stationary_pool": "first-experience validation column at full-retention checkpoints only; acquisition diagonal at experience 1 excluded",
        "maximum_stationary_drawdown": spec["maximum_stationary_drawdown"],
        "stationary_screen": "every development seed must pass independently",
        "stationary_final_tolerance": spec["stationary_final_tolerance"],
        "score": "0.5 * (mean recurring final accuracy + mean recurring late acquisition AUC)",
        "score_tie_tolerance": spec["selection_score_tie_tolerance"],
        "tie_break": "smallest feature gain within tolerance of the maximum eligible score",
    }
    if rule != expected_rule:
        raise ValueError("selection rule disagrees with the embedded specification")

    archive_inputs = []
    for record in records:
        path = results_root / record["result_path"]
        result, result_hash = study._load(path)
        if result_hash != record["result_sha256"]:
            raise ValueError(f"result changed while being validated: {record['result_path']}")
        # The shared analysis helper plots forgetting and formats elapsed time.
        study._finite(result["metrics"].get("forgetting"), "forgetting")
        study._finite(result.get("wall_seconds"), "wall_seconds", minimum=0)
        events_path = path.parent / "events.json"
        events, events_hash = study._load(events_path)
        events = events.get("events")
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise ValueError(f"missing or malformed archived event list: {events_path}")
        for event in events:
            controller = event.get("controller", {})
            if not isinstance(controller, dict) or controller.get("transition") and \
                    ("step" not in event or "phase" not in event):
                raise ValueError(f"malformed archived controller event: {events_path}")
        archive_inputs.append({**record,
                               "events_path": events_path.relative_to(results_root).as_posix(),
                               "events_sha256": events_hash})
    return artifact, raw, archive_inputs


def _decision(candidate: dict) -> str:
    if candidate["selected"]:
        return "Selected"
    if not candidate["passes_stationary_screen"]:
        return "Screen failed"
    if not candidate["within_stationary_final_tolerance"]:
        return "Stationary final below cutoff"
    if candidate["in_score_tie_band"]:
        return "Score tie; larger gain"
    return "Eligible; lower score"


def _table_rows(artifact: dict) -> list[dict]:
    rows = []
    for candidate in artifact["candidates"]:
        rows.append({
            "feature_gain": candidate["feature_gain"],
            "stationary_screen": "pass" if candidate["passes_stationary_screen"] else "fail",
            "mean_recurring_final_accuracy": candidate["mean_recurring_final_accuracy"],
            "mean_recurring_late_auc": candidate["mean_recurring_late_auc"],
            "mean_stationary_final_accuracy": candidate["mean_stationary_final_accuracy"],
            "maximum_stationary_drawdown": max(candidate["stationary_drawdowns"].values()),
            **{f"stationary_drawdown_seed_{seed}": candidate["stationary_drawdowns"][str(seed)]
               for seed in artifact["development_seeds"]},
            "selection_score": candidate["selection_score"],
            "within_stationary_final_tolerance": candidate["within_stationary_final_tolerance"],
            "in_score_tie_band": candidate["in_score_tie_band"],
            "selected": candidate["selected"],
            "decision": _decision(candidate),
        })
    return rows


def _plot(artifact: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    candidates, seeds = artifact["candidates"], artifact["development_seeds"]
    xs = np.arange(len(candidates))
    labels = [f"{row['feature_gain']:g}" for row in candidates]
    figure = plt.figure(figsize=(13, 9), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(2.0, 1.35))
    performance = figure.add_subplot(grid[0, 0])
    retention = figure.add_subplot(grid[0, 1])
    table_ax = figure.add_subplot(grid[1, :])
    series = (
        ("mean_recurring_final_accuracy", "Recurring final", "#2369a2", "o"),
        ("mean_recurring_late_auc", "Recurring late AUC", "#be6512", "s"),
        ("mean_stationary_final_accuracy", "Stationary final", "#357c55", "^"),
    )
    for key, label, color, marker in series:
        performance.plot(xs, [candidate[key] for candidate in candidates],
                         color=color, marker=marker, linewidth=1.5, markersize=7, label=label)
    performance.set(ylabel="Accuracy or normalized AUC", ylim=(0, 1),
                    title=f"Mean development outcomes ({len(seeds)} seeds)")
    performance.yaxis.set_major_formatter(PercentFormatter(1))
    performance.legend(loc="best", fontsize=9)

    width = min(0.32, 0.7 / len(seeds))
    for i, seed in enumerate(seeds):
        offsets = xs + (i - (len(seeds) - 1) / 2) * width
        values = [row["stationary_drawdowns"][str(seed)] for row in candidates]
        retention.bar(offsets, values, width, label=f"Seed {seed}", alpha=0.85)
    threshold = artifact["specification"]["maximum_stationary_drawdown"]
    retention.axhline(threshold, color="#9b302a", linestyle="--", linewidth=1.5,
                     label=f"Screen threshold ({100 * threshold:g} pp)")
    ymax = max(threshold, *(max(row["stationary_drawdowns"].values()) for row in candidates))
    retention.set(ylabel="Largest earlier-to-later decrease (pp)",
                  ylim=(0, min(1.05, max(0.15, 1.25 * ymax))),
                  title="Stationary retention on the same first-experience pool")
    retention.yaxis.set_major_formatter(PercentFormatter(1, symbol=""))
    retention.legend(loc="best", fontsize=9)
    for axes in (performance, retention):
        axes.set(xticks=xs, xticklabels=labels, xlabel="Feature gain (categorical grid)",
                 xlim=(-0.5, len(candidates) - 0.5))
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(axis="y", alpha=0.18)
        for i, row in enumerate(candidates):
            if row["selected"]:
                axes.axvspan(i - 0.43, i + 0.43, color="#2c8550", alpha=0.09, zorder=0)
            elif not row["passes_stationary_screen"]:
                axes.axvspan(i - 0.43, i + 0.43, color="#bd3c32", alpha=0.07, zorder=0)

    table_ax.axis("off")
    columns = ["Gain", "Stationary\nscreen", "Recurring\nfinal (%)", "Recurring\nlate AUC (%)",
               "Stationary\nfinal (%)", "Max drawdown\n(pp; all seeds)", "Score\n(%)", "Decision"]
    cells = [[f"{row['feature_gain']:g}", "PASS" if row["passes_stationary_screen"] else "FAIL",
              f"{100 * row['mean_recurring_final_accuracy']:.2f}",
              f"{100 * row['mean_recurring_late_auc']:.2f}",
              f"{100 * row['mean_stationary_final_accuracy']:.2f}",
              f"{100 * max(row['stationary_drawdowns'].values()):.2f}",
              f"{100 * row['selection_score']:.2f}", _decision(row)] for row in candidates]
    table = table_ax.table(cellText=cells, colLabels=columns, cellLoc="center", loc="center",
                           colWidths=[0.06, 0.10, 0.13, 0.13, 0.13, 0.13, 0.07, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5d9")
        if row == 0:
            cell.set_facecolor("#e9edf1")
            cell.set_text_props(weight="bold")
        elif candidates[row - 1]["selected"]:
            cell.set_facecolor("#dcefe2")
            cell.set_text_props(weight="bold")
        elif not candidates[row - 1]["passes_stationary_screen"]:
            cell.set_facecolor("#fae9e6")
    selected = artifact["selected_feature_gain"]
    verdict = f"selected gain {selected:g}" if selected is not None else "no candidate passed"
    figure.suptitle(f"V3 development gain grid: {verdict}\n"
                   "All candidates and development seeds; no evaluation outcomes", fontsize=15)
    checkpoints = artifact["selection_rule"]["stationary_checkpoint_experiences"]
    table_ax.set_title("Prespecified screen and selection (all outcomes retained)", fontsize=11)
    table_ax.text(0, 0.01,
                  f"Drawdown: first-pool accuracy at experiences {', '.join(map(str, checkpoints))}; "
                  "the experience-1 diagonal is excluded.\n"
                  "Score = 0.5 × (recurring final + recurring late AUC). "
                  "These development outcomes selected the gain; they are not confirmation.",
                  transform=table_ax.transAxes, fontsize=9, va="bottom")
    for suffix in ("png", "svg"):
        figure.savefig(output / f"gain_grid.{suffix}", dpi=170,
                       metadata={"Creator": "ACP-CL v3 development archive"})
    plt.close(figure)
    svg = output / "gain_grid.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")


def _markdown(artifact: dict, selection_hash: str) -> str:
    spec, rows = artifact["specification"], _table_rows(artifact)
    selected = artifact["selected_feature_gain"]
    verdict = (f"The prespecified rule selected feature gain **{selected:g}**." if selected is not None
               else "**No candidate passed the prespecified stationary screen. No gain was locked.**")
    lines = ["# V3 development gain selection", "", verdict, "",
             f"All {len(artifact['input_results'])} completed runs are included: "
             f"{len(rows)} gains × 2 regimes × {len(artifact['development_seeds'])} seeds "
             f"({', '.join(map(str, artifact['development_seeds']))}). "
             "Every run uses `recycle_v3`, independent colors, and the validation split. "
             "These outcomes were used for selection; they are not confirmatory evaluation.", "",
             "Values are percentages, except drawdown in percentage points. CSV values retain the original 0–1 scale.", "",
             "| Gain | Stationary screen | Recurring final | Recurring late AUC | Stationary final | Max drawdown | Score | Decision |",
             "|---:|:---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['feature_gain']:g} | {row['stationary_screen'].upper()} | "
                     f"{100 * row['mean_recurring_final_accuracy']:.2f} | "
                     f"{100 * row['mean_recurring_late_auc']:.2f} | "
                     f"{100 * row['mean_stationary_final_accuracy']:.2f} | "
                     f"{100 * row['maximum_stationary_drawdown']:.2f} | "
                     f"{100 * row['selection_score']:.2f} | {row['decision']} |")
    checkpoints = artifact["selection_rule"]["stationary_checkpoint_experiences"]
    lines.extend(["", "![All v3 development candidates](gain_grid.png)", "",
                  "Drawdown is the maximum earlier-to-later loss on the same first-experience validation pool "
                  f"at experiences {', '.join(map(str, checkpoints))}. The acquisition diagonal at experience 1 "
                  f"is excluded. Every seed must have drawdown ≤ {100 * spec['maximum_stationary_drawdown']:g} pp. "
                  f"Passing candidates must then be within {100 * spec['stationary_final_tolerance']:g} pp "
                  "of the best passing stationary final mean. Among those candidates, the rule maximizes "
                  "0.5 × (mean recurring final accuracy + mean recurring late AUC); "
                  f"scores within {100 * spec['selection_score_tie_tolerance']:g} pp of the eligible maximum "
                  "tie, and the smallest gain wins.", "",
                  "The original selection bytes are preserved in [selection.json](selection.json). "
                  "Result, allocation, and manifest byte hashes were checked against that artifact before archiving. "
                  "Additional event-file hashes are recorded in [archive_manifest.json](archive_manifest.json); "
                  "events were not inputs to the selection score. "
                  "No held-out evaluation result was read.", "",
                  f"Selection SHA-256: `{selection_hash}`.", "",
                  f"Training source SHA-256: `{artifact['common_identity']['source_sha256']}`.", "",
                  f"Training runtime SHA-256: `{artifact['common_identity']['runtime_sha256']}`.", "",
                  "Complete suite archives:", ""])
    for stem in study.development_configs(spec, artifact["study_spec_sha256"]):
        lines.append(f"- [{stem}]({stem}/summary.md)")
    lines.extend(["", "[Numeric candidate table](gain_grid.csv) · [Vector figure](gain_grid.svg)", ""])
    return "\n".join(lines)


def summarize(selection_path: Path, results_root: Path, output: Path) -> dict:
    artifact, raw, records = validate_selection(selection_path, results_root)
    selection_hash = hashlib.sha256(raw).hexdigest()
    copied_selection = output / "selection.json"
    if copied_selection.exists() and copied_selection.read_bytes() != raw:
        raise FileExistsError(f"preserving archive bound to a different selection: {copied_selection}")
    result_directory, output_directory = results_root.resolve(), output.resolve()
    if result_directory == output_directory or result_directory in output_directory.parents or \
            output_directory in result_directory.parents:
        raise ValueError("the development archive and input result tree must not overlap")

    # Import the shared plotting helper only after every selection input passes.
    # Limit CPU work while other experiment processes may be using the GPU.
    import torch
    torch.set_num_threads(2)
    from acp_cl.analysis import analyze

    stems = list(study.development_configs(artifact["specification"], artifact["study_spec_sha256"]))
    for stem in stems:
        analyze(results_root / stem)
        archive(results_root / stem, output / stem)
    copied_selection.write_bytes(raw)
    rows = _table_rows(artifact)
    with (output / "gain_grid.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(artifact, output)
    (output / "summary.md").write_text(_markdown(artifact, selection_hash), encoding="utf-8", newline="\n")
    names = ["selection.json", "gain_grid.csv", "gain_grid.png", "gain_grid.svg", "summary.md"]
    names.extend(f"{stem}/{name}" for stem in stems for name in (
        "manifest.json", "summary.json", "summary.csv", "summary.md", "overview.png",
        "overview.svg", "individual_results.json"
    ))
    archived_files = [{"path": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
                      for name in sorted(names)]
    manifest = {
        "schema_version": 1, "status": artifact["status"], "selection_sha256": selection_hash,
        "selected_feature_gain": artifact["selected_feature_gain"],
        "evaluation_outcomes_consumed": False, "input_hashes_verified": True,
        "input_paths_relative_to": "the development results root passed to this script",
        "suite_archives": stems, "input_results": records,
        "archive_files": archived_files,
    }
    (output / "archive_manifest.json").write_bytes(study._bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "configs" / "v3" / "locked" / "selection.json")
    parser.add_argument("--results", type=Path, default=ROOT / "runs" / "v3_development")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "v3" / "development")
    args = parser.parse_args()
    try:
        report = summarize(args.selection, args.results, args.output)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Archived {len(report['input_results'])} development runs: {report['status']}; {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
