"""Audit and summarize the locked joint-persistence pilot without tuning it."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile

import numpy as np

from acp_cl.persistence.study import digest, write_json
from acp_cl.persistence.world import COMPOSITIONS, KNOWN, schedule


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(curve):
    x = np.asarray([point["arrivals"] for point in curve], dtype=float)
    y = np.asarray([point["metrics"]["survival"]["12"] for point in curve])
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2) / x[-1])


def mean(values):
    value = float(statistics.mean(values))
    # A numerical residual must not turn an exact rate tie into a screen pass.
    # This tolerance is far below one outcome in the entire pilot cohort.
    return 0.0 if abs(value) < 1e-12 else value


def metrics(record, scratch):
    panels = {p["after_block"]: p["metrics"] for p in record["panels"]}
    blocks = record["blocks"]
    target = record["schedule_info"]["target"]
    valid_known = [r.name for r in KNOWN if record["schedule"] != "reversal" or r.name != target]
    composition_names = {r.name for r in COMPOSITIONS}
    seen = {b["regime"]["name"] for b in blocks}
    held_out = composition_names - seen
    entry = blocks[10]["curve"][0]["metrics"]["survival"]["12"]
    previous_target = [b for b in blocks[:10] if b["regime"]["name"] == target][-1]
    before_gap = previous_target["curve"][-1]["metrics"]["survival"]["12"]
    return_curve = blocks[10]["curve"]
    crossing = [p["arrivals"] for p in return_curve
                if p["metrics"]["survival"]["12"] >= before_gap]
    values = {
        "entry_survival": entry,
        "pre_gap_survival": before_gap,
        "forgetting": before_gap - entry if record["schedule_info"]["target_returns"] else None,
        "recovered": bool(crossing) if record["schedule_info"]["target_returns"] else None,
        "recovery_arrivals_capped": (crossing[0] if crossing else return_curve[-1]["arrivals"])
                                     if record["schedule_info"]["target_returns"] else None,
        "return_auc": auc(return_curve),
        "composition_after_warmup": mean(panels[3][r.name]["survival"]["12"] for r in COMPOSITIONS),
        "composition_final": mean(panels[11][name]["survival"]["12"] for name in sorted(held_out)),
        "known_final": mean(panels[11][name]["survival"]["12"] for name in valid_known),
        "late_auc": auc(blocks[-1]["curve"]),
        "late_initial": blocks[-1]["curve"][0]["metrics"]["survival"]["12"],
        "late_final": blocks[-1]["curve"][-1]["metrics"]["survival"]["12"],
        "late_vs_scratch_auc": auc(blocks[-1]["curve"]) - auc(scratch["curve"]),
        "initial_acquisition_gain": mean(b["curve"][-1]["metrics"]["survival"]["12"]
                                         - b["curve"][-1]["metrics"]["no_transfer"] for b in blocks[:4]),
        "history_removal_drop": mean(panels[11][name]["survival"]["12"]
                                     - record["final_temporal_control"][name]["survival"]["12"]
                                     for name in valid_known),
        "elapsed_seconds": record["elapsed_seconds"],
        "selection_seconds": record["diagnostics"]["cost"]["selection_seconds"],
        "learning_seconds": record["diagnostics"]["cost"]["training_seconds"]
                            + record["diagnostics"]["cost"]["selection_seconds"],
    }
    return values


def paired_interval(values, repetitions=20000):
    values = np.asarray(values, dtype=float)
    values = np.where(np.abs(values) < 1e-12, 0.0, values)
    rng = np.random.default_rng(90871)
    bootstrap = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), lower=float(np.quantile(bootstrap, .025)),
                upper=float(np.quantile(bootstrap, .975)), positive=int((values > 0).sum()),
                n=len(values), differences=values.tolist())


def audit_inputs(directory, protocol_lock):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    config, identity = manifest["config"], manifest["identity"]
    if digest(config) != identity["config_sha256"] or digest(manifest["runtime"]) != identity["runtime_sha256"]:
        raise ValueError("manifest config/runtime digest mismatch")
    if digest(manifest["source_files"]) != identity["source_sha256"]:
        raise ValueError("manifest source digest mismatch")
    if any(protocol_lock[key] != identity[key] for key in ("config_sha256", "source_sha256")):
        raise ValueError("locked protocol and actual suite differ")
    with zipfile.ZipFile(directory / "training_source.zip") as archive:
        for name, expected in manifest["source_files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"archived source mismatch: {name}")
    expected_runs = {(stream, method, seed) for stream in config["schedules"]
                     for method in config["methods"] for seed in config["seeds"]}
    records, artifacts = {}, []
    for path in sorted(directory.glob("*/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = record["schedule"], record["method"], record["seed"]
        if key not in expected_runs or key in records or record["identity"] != identity:
            raise ValueError("unexpected, duplicate, or mismatched result")
        regimes, info = schedule(record["seed"], record["schedule"])
        if info != record["schedule_info"] or [asdict(r) for r in regimes] != [
                b["regime"] for b in record["blocks"]]:
            raise ValueError("result schedule mismatch")
        expected_x = list(range(0, config["block_size"] + 1, config["probe_every"]))
        if any([p["arrivals"] for p in b["curve"]] != expected_x for b in record["blocks"]):
            raise ValueError("incomplete learning curve")
        for block in record["blocks"]:
            for point in block["curve"]:
                result = point["metrics"]
                survival = result["survival"]["12"]
                if not 0 <= survival <= result["clairvoyant_upper"] <= 1:
                    raise ValueError("survival outside attainable bounds")
        costs = record["diagnostics"]["cost"]
        arrivals = 12 * config["block_size"]
        if costs["arrivals"] != arrivals or costs["optimizer_steps"] != (
                arrivals // config["batch_size"] * config["updates_per_batch"]):
            raise ValueError("arrival/update count mismatch")
        if costs["training_forwards"] != 2 * arrivals * config["updates_per_batch"]:
            raise ValueError("training forward count mismatch")
        if len(record["diagnostics"]["final_memory_ids"]) != (
                0 if record["method"] == "none" else config["memory_capacity"]):
            raise ValueError("memory capacity mismatch")
        if record["method"] == "frozen_encoder" and record["diagnostics"]["encoder_displacement_l2"] != 0:
            raise ValueError("frozen encoder changed")
        records[key] = record
        artifacts.append({"path": path.relative_to(directory).as_posix(), "sha256": file_hash(path)})
    if set(records) != expected_runs:
        raise ValueError(f"incomplete suite: {len(expected_runs - set(records))} missing runs")
    scratch = {}
    for seed in config["seeds"]:
        for stream in config["schedules"]:
            path = directory / f"{stream}_scratch_{seed}.json"
            reference = json.loads(path.read_text(encoding="utf-8"))
            if reference["identity"] != identity:
                raise ValueError("scratch identity mismatch")
            scratch[stream, seed] = reference
            group = [records[stream, method, seed] for method in config["methods"]]
            first = group[0]
            for record in group:
                if [b["data_sha256"] for b in record["blocks"]] != [
                        b["data_sha256"] for b in first["blocks"]]:
                    raise ValueError("unpaired current data")
                if record["diagnostics"]["initial_hash"] != first["diagnostics"]["initial_hash"]:
                    raise ValueError("unpaired initialization")
                if record["diagnostics"]["history_counts"] != first["diagnostics"]["history_counts"]:
                    raise ValueError("frequency estimator received different observations")
                if record["blocks"][-1]["data_sha256"] != reference["data_sha256"]:
                    raise ValueError("fresh reference received different data")
            replay = [r for r in group if r["method"] != "none"]
            for cost in ("replay_presentations", "current_presentations", "scoring_forwards"):
                if len({r["diagnostics"]["cost"][cost] for r in replay}) != 1:
                    raise ValueError(f"replay methods have unequal {cost}")
            if len({r["diagnostics"]["peak_memory_payload_bytes"] for r in replay}) != 1:
                raise ValueError("replay payload budgets differ")
            uniform = records.get((stream, "uniform", seed))
            frozen = records.get((stream, "frozen_late", seed))
            if uniform and frozen and uniform["blocks"][:4] != frozen["blocks"][:4]:
                raise ValueError("late-frozen control differs before intervention")
    for seed in config["seeds"]:
        if "short" in config["schedules"] and "long" in config["schedules"]:
            method = config["methods"][0]
            fingerprints = [[b["data_sha256"] for b in records[s, method, seed]["blocks"]]
                            for s in ("short", "long")]
            if sorted(fingerprints[0]) != sorted(fingerprints[1]):
                raise ValueError("short/long schedules do not share the same data multiset")
    completion = json.loads((directory / "completion.json").read_text())
    if completion["identity"] != identity or completion["runs"] != len(records):
        raise ValueError("completion record mismatch")
    return manifest, records, scratch, artifacts


def summarize(directory, output, lock_path):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest, records, scratch, artifacts = audit_inputs(directory, lock)
    config = manifest["config"]
    rows = [{"schedule": stream, "method": method, "seed": seed,
             **metrics(record, scratch[stream, seed])}
            for (stream, method, seed), record in records.items()]
    lookup = {(r["schedule"], r["method"], r["seed"]): r for r in rows}
    summaries, contrasts = [], []
    endpoint_names = ("entry_survival", "composition_after_warmup", "composition_final", "known_final",
                      "late_auc", "late_final", "late_vs_scratch_auc", "initial_acquisition_gain",
                      "history_removal_drop", "learning_seconds", "selection_seconds", "elapsed_seconds")
    for stream in config["schedules"]:
        for method in config["methods"]:
            group = [lookup[stream, method, seed] for seed in config["seeds"]]
            summaries.append({"schedule": stream, "method": method,
                              **{metric: mean(r[metric] for r in group) for metric in endpoint_names}})
            if method != "uniform":
                for metric in endpoint_names:
                    values = [lookup[stream, method, seed][metric]
                              - lookup[stream, "uniform", seed][metric] for seed in config["seeds"]]
                    contrasts.append({"schedule": stream, "method": method, "endpoint": metric,
                                      **paired_interval(values)})
    primary = next(r for r in contrasts if r["schedule"] == "long" and r["method"] == "joint"
                   and r["endpoint"] == "entry_survival")
    coverage_gain = mean(lookup["long", "joint", s]["entry_survival"]
                         - lookup["long", "coverage", s]["entry_survival"] for s in config["seeds"])
    baseline_gain = mean(r["initial_acquisition_gain"] for r in rows if r["method"] == "uniform")
    secondary = {metric: mean(lookup["long", "joint", s][metric] - lookup["long", "uniform", s][metric]
                              for s in config["seeds"]) for metric in ("composition_final", "late_auc")}
    criteria = lock["continuation"]
    checks = {
        "baseline_adequate": baseline_gain * 100 >= criteria["uniform_acquisition_gain_pp_at_least"],
        "primary_mean": primary["mean"] * 100 >= criteria["primary_mean_pp_at_least"],
        "positive_seed_count": primary["positive"] >= criteria["positive_seeds_at_least"],
        "beats_coverage": coverage_gain > 0,
        "secondary_noninferiority_screen": min(secondary.values()) * 100 >= -criteria["maximum_secondary_loss_pp"],
    }
    archive = {"manifest": manifest, "protocol_lock": lock, "rows": rows, "summaries": summaries,
               "contrasts": contrasts, "primary": primary, "screen": checks,
               "continue": all(checks.values()), "baseline_acquisition_gain": baseline_gain,
               "coverage_difference": coverage_gain, "secondary_differences": secondary,
               "runs": list(records.values()), "scratch": list(scratch.values()), "source_artifacts": artifacts,
               "analysis_sha256": file_hash(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(output / "archive.json.gz"), mode="wb", mtime=0) as handle:
        handle.write(json.dumps(archive, separators=(",", ":"), allow_nan=False).encode())
    write_json(output / "summary.json", {k: v for k, v in archive.items()
                                         if k not in ("runs", "scratch", "source_artifacts")})
    for name in ("manifest.json", "training_source.zip", "completion.json"):
        if (directory / name).resolve() != (output / name).resolve():
            shutil.copyfile(directory / name, output / name)
    lines = ["# Joint-persistence pilot: complete descriptive results", "",
             f"{len(records)} comparative runs and {len(scratch)} fresh late-block references. "
             "All planned results included. Intervals resample paired seeds and are exploratory, "
             "not adjusted for secondary comparisons.", "",
             f"Primary long-gap return difference, joint minus uniform: **{100 * primary['mean']:+.2f} pp** "
             f"(paired bootstrap 95% interval {100 * primary['lower']:+.2f} to {100 * primary['upper']:+.2f}); "
             f"positive in {primary['positive']}/{primary['n']} seeds.", "",
             f"Predeclared continuation screen: **{'PASS' if archive['continue'] else 'FAIL'}**. "
             f"Baseline acquisition gain over no transfer: {100 * baseline_gain:.2f} pp.", "",
             "| Criterion | Pass |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in checks.items()]
    for stream in config["schedules"]:
        lines += ["", f"## {stream}", "", "Percentages; late AUC integrates survival over current arrivals.", "",
                  "| Method | Entry, block 11 | Final held-out combinations | Final valid known | Late AUC | Late endpoint | Learning + selection, seconds |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in [r for r in summaries if r["schedule"] == stream]:
            lines.append(f"| {row['method']} | {100 * row['entry_survival']:.2f} | "
                         f"{100 * row['composition_final']:.2f} | {100 * row['known_final']:.2f} | "
                         f"{100 * row['late_auc']:.2f} | {100 * row['late_final']:.2f} | {row['learning_seconds']:.2f} |")
    lines += ["", "## Primary contrast by seed", "", "| Seed | Joint | Uniform | Difference, pp |",
              "|---|---:|---:|---:|"]
    for seed in config["seeds"]:
        joint, uniform = (lookup["long", method, seed]["entry_survival"] for method in ("joint", "uniform"))
        lines.append(f"| {seed} | {100 * joint:.2f} | {100 * uniform:.2f} | {100 * (joint - uniform):+.2f} |")
    lines += ["", "## Scope and accounting", "",
              "The implemented recurrence score is a coarse historical-frequency proxy. The relevance score is "
              "predicted action sensitivity, not a causal estimate of memory value. Parameter counts, memory "
              "payload and optimization work are matched among full-learning replay methods; selection overhead "
              "is additional. Frozen controls have less backward work. The no-memory control repeats current "
              "examples and scores fewer candidates.", "",
              "The four initial contexts correlate source direction with the loss/delay glyph combination. "
              "The withheld combinations break that correlation. Therefore known-context performance alone "
              "does not establish use of temporal or causal structure. History-removal probes are sensitivity "
              "checks and may create distribution shift. Final no-return composition scores exclude the "
              "combination subsequently trained; reversal retention excludes the now-invalid old target.", "",
              "The fresh late-block fits differ in replay contents and history. Their differences from continual "
              "learners are descriptive and cannot isolate plasticity. No claim of general continual representation "
              "learning, inherited learning, indefinite survival, or a novel evolutionary algorithm follows.", "",
              "Full raw curves, seeds, hashes, resource counters and fresh fits are in `archive.json.gz`. "
              "The source archive and manifest support regeneration. Checkpoints remain in the local run directory."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(records), "primary": primary, "screen": checks}, indent=2))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("reports/persistence/protocol_lock.json"))
    args = parser.parse_args()
    summarize(args.input, args.output, args.lock)


if __name__ == "__main__":
    main()
