"""Audit and report the locked conditional-reuse pilot, retaining every run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from acp_cl.conditional.world import schedule
from acp_cl.persistence.study import digest, write_json


def mean(values):
    values = list(values)
    value = float(np.mean(values))
    return 0.0 if abs(value) < 1e-12 else value


def paired(values):
    values = np.asarray(values, dtype=float)
    values[np.abs(values) < 1e-12] = 0
    rng = np.random.default_rng(3172026)
    estimates = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return dict(mean=mean(values), lower=float(np.quantile(estimates, .025)),
                upper=float(np.quantile(estimates, .975)), positive=int((values > 0).sum()),
                n=len(values), differences=values.tolist())


def probe_score(probe, kind="correct", field="survival"):
    def extract(rep):
        if kind == "correct":
            return rep["curve"][-1]["metrics"][field]
        if kind == "entry":
            return rep["curve"][0]["metrics"][field]
        return rep[kind][field]
    return mean(extract(rep) for rep in probe["replicates"])


def auc(curve):
    x = np.asarray([p["arrivals"] for p in curve])
    y = np.asarray([p["metrics"]["survival"] for p in curve])
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2) / x[-1])


def metrics(record):
    probe, panel = record["return_probe"], record["final_panel"]
    modes = [record["seed"] % 2] if record["schedule"] == "stable" else [0, 1]
    unpredictable = record["schedule"] == "unpredictable"
    return dict(
        return_survival=probe_score(probe),
        return_entry=probe_score(probe, "entry"),
        correct_minus_opposite=probe_score(probe) - probe_score(probe, "opposite"),
        correct_minus_erased=probe_score(probe) - probe_score(probe, "erased"),
        correct_minus_broken=probe_score(probe) - probe_score(probe, "broken_binding"),
        return_no_transfer=probe_score(probe, field="no_transfer"),
        return_upper=probe_score(probe, field="clairvoyant_upper"),
        return_brier=probe_score(probe, field="brier"),
        return_entropy=probe_score(probe, field="entropy"),
        return_disagreement=probe_score(probe, field="disagreement"),
        final_known=(None if unpredictable else
                     mean(probe_score(panel[f"mode{m}_gate0"]) for m in modes)),
        final_gate=(None if unpredictable else
                    mean(probe_score(panel[f"mode{m}_gate1"]) for m in modes)),
        final_online=record["blocks"][-1]["curve"][-1]["metrics"]["survival"],
        late_auc=auc(record["blocks"][9]["curve"]),
        late_endpoint=record["blocks"][9]["curve"][-1]["metrics"]["survival"],
        initial_gain=mean(b["curve"][-1]["metrics"]["survival"]
                          - b["curve"][-1]["metrics"]["no_transfer"] for b in record["blocks"][:4]),
        training_seconds=record["diagnostics"]["cost"]["training_seconds"],
        elapsed_seconds=record["elapsed_seconds"])


def read_and_audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((directory / "protocol_lock.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    identity, config = manifest["identity"], manifest["config"]
    for key, content in (("source_sha256", manifest["source_files"]),
                         ("config_sha256", config), ("runtime_sha256", manifest["runtime"])):
        if digest(content) != identity[key] or lock[key] != identity[key]:
            raise ValueError("manifest/lock mismatch")
    if hashlib.sha256((directory / "protocol_at_lock.md").read_bytes()).hexdigest() != lock["protocol_sha256"]:
        raise ValueError("protocol archive hash mismatch")
    with zipfile.ZipFile(directory / "training_source.zip") as archive:
        for name, expected in manifest["source_files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("archived training source hash mismatch")
    expected = {(stream, method, seed) for stream in config["schedules"]
                for method in config["methods"] for seed in config["seeds"]}
    if completion["identity"] != identity or completion["runs"] != len(expected):
        raise ValueError("incomplete or mismatched completion marker")
    records, artifacts = {}, []
    for path in directory.glob("*/result.json"):
        r = json.loads(path.read_text(encoding="utf-8"))
        key = r["schedule"], r["method"], r["seed"]
        if key not in expected or key in records or r["identity"] != identity:
            raise ValueError("unexpected/duplicate/mismatched result")
        conditions = schedule(r["seed"], r["schedule"])
        if [b["condition"] for b in r["blocks"]] != [asdict(c) for c in conditions]:
            raise ValueError("schedule mismatch")
        xs = list(range(0, config["block_size"] + 1, config["probe_every"]))
        if any([p["arrivals"] for p in b["curve"]] != xs for b in r["blocks"]):
            raise ValueError("incomplete online curve")
        for block in r["blocks"]:
            if len(block["batch_sha256"]) != config["block_size"] // config["batch_size"]:
                raise ValueError("incomplete training records")
            for point in block["curve"]:
                m = point["metrics"]
                if not 0 <= m["survival"] <= m["clairvoyant_upper"] <= 1:
                    raise ValueError("invalid survival bounds")
        d, arrivals = r["diagnostics"], 12 * config["block_size"]
        cost = d["cost"]
        checks = dict(arrivals=arrivals, optimizer_steps=arrivals // config["batch_size"]
                      * config["updates_per_batch"], query_presentations=2 * arrivals * config["updates_per_batch"],
                      support_presentations=2 * arrivals * config["updates_per_batch"])
        if any(cost[k] != v for k, v in checks.items()):
            raise ValueError("training budget mismatch")
        if len(d["memory_ids"]) != (0 if r["method"] == "current_only" else config["memory_packets"]):
            raise ValueError("memory capacity mismatch")
        if d["packets_seen"] != arrivals // config["batch_size"]:
            raise ValueError("arrival packet mismatch")
        probes = [r["return_probe"], *r["final_panel"].values()]
        feedback = [0, config["batch_size"] // 4, config["batch_size"] // 2, config["batch_size"]]
        for probe in probes:
            if len(probe["replicates"]) != config["support_replicates"]:
                raise ValueError("incomplete support replicates")
            for rep in probe["replicates"]:
                if [p["feedback"] for p in rep["curve"]] != feedback:
                    raise ValueError("incomplete switch curve")
        if any(p["model_sha256"] != d["final_hash"] for p in r["final_panel"].values()):
            raise ValueError("final probe altered weights")
        records[key] = r
        artifacts.append(dict(path=path.relative_to(directory).as_posix(),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if set(records) != expected:
        raise ValueError(f"incomplete suite: {len(expected - set(records))} missing")
    for stream in config["schedules"]:
        for seed in config["seeds"]:
            group = [records[stream, m, seed] for m in config["methods"]]
            first = group[0]
            for record in group:
                if [b["batch_sha256"] for b in record["blocks"]] != [b["batch_sha256"] for b in first["blocks"]]:
                    raise ValueError("different data across methods")
                if record["diagnostics"]["initial_hash"] != first["diagnostics"]["initial_hash"]:
                    raise ValueError("different initialization across methods")
            replay = [r for r in group if r["method"] != "current_only"]
            if any(r["diagnostics"]["memory_ids"] != replay[0]["diagnostics"]["memory_ids"] for r in replay):
                raise ValueError("different reservoir memberships")
            if "conditional" in config["methods"] and "frozen_features" in config["methods"]:
                if records[stream, "conditional", seed]["blocks"][:4] != records[stream, "frozen_features", seed]["blocks"][:4]:
                    raise ValueError("frozen-feature diagnostic differs before intervention")
    return manifest, lock, records, artifacts


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, lock, records, artifacts = read_and_audit(directory)
    config = manifest["config"]
    values = {key: metrics(record) for key, record in records.items()}
    aggregate, contrasts = {}, {}
    for stream in config["schedules"]:
        aggregate[stream], contrasts[stream] = {}, {}
        for method in config["methods"]:
            rows = [values[stream, method, seed] for seed in config["seeds"]]
            aggregate[stream][method] = {k: None if rows[0][k] is None else mean(row[k] for row in rows)
                                        for k in rows[0]}
            if method != "pooled":
                contrasts[stream][method + "_minus_pooled"] = {
                    k: paired([values[stream, method, s][k] - values[stream, "pooled", s][k]
                               for s in config["seeds"]]) for k in rows[0] if rows[0][k] is not None}
        for comparison in ("shuffled", "current_only", "frozen_features"):
            contrasts[stream]["conditional_minus_" + comparison] = {
                k: paired([values[stream, "conditional", s][k] - values[stream, comparison, s][k]
                           for s in config["seeds"]]) for k in rows[0] if rows[0][k] is not None}
        contrasts[stream]["conditional_correct_minus_opposite"] = paired([
            values[stream, "conditional", s]["correct_minus_opposite"] for s in config["seeds"]])
    recurring = aggregate["recurring"]
    primary = contrasts["recurring"]["conditional_minus_pooled"]["return_survival"]
    binding = contrasts["recurring"]["conditional_correct_minus_opposite"]
    adequacy = dict(oracle_gain_over_no_transfer=recurring["oracle"]["return_survival"]
                    - recurring["oracle"]["return_no_transfer"],
                    oracle_gain_over_pooled=recurring["oracle"]["return_survival"]
                    - recurring["pooled"]["return_survival"])
    adequacy["passed"] = (adequacy["oracle_gain_over_no_transfer"] >= .05
                          and adequacy["oracle_gain_over_pooled"] >= .02)
    criteria = dict(primary_gain_at_least_2pp=primary["mean"] >= .02,
                    primary_positive_in_6_seeds=primary["positive"] >= 6,
                    gain_over_shuffled_at_least_2pp=contrasts["recurring"][
                        "conditional_minus_shuffled"]["return_survival"]["mean"] >= .02,
                    opposite_history_drop_at_least_2pp=binding["mean"] >= .02,
                    opposite_history_drop_positive_in_6_seeds=binding["positive"] >= 6,
                    unpredictable_loss_no_worse_than_2pp=contrasts["unpredictable"][
                        "conditional_minus_pooled"]["return_survival"]["mean"] >= -.02,
                    late_gate_loss_no_worse_than_2pp=contrasts["recurring"][
                        "conditional_minus_pooled"]["final_gate"]["mean"] >= -.02)
    summary = dict(aggregate=aggregate, contrasts=contrasts, primary=primary,
                   adequacy=adequacy, criteria=criteria,
                   decision=("inconclusive" if not adequacy["passed"] else
                             "pass" if all(criteria.values()) else "fail"),
                   seeds=config["seeds"], runs=len(records), audit=dict(passed=True, result_hashes=artifacts))
    write_json(output / "summary.json", summary)
    for name in ("manifest.json", "protocol_lock.json", "protocol_at_lock.md", "training_source.zip", "completion.json"):
        shutil.copy2(directory / name, output / name)
    archive = dict(manifest=manifest, protocol_lock=lock, summary=summary,
                   records=[records[k] for k in sorted(records)])
    with gzip.GzipFile(filename=str(output / "archive.json.gz"), mode="wb", mtime=0) as handle:
        handle.write(json.dumps(archive, separators=(",", ":"), allow_nan=False).encode())
    lines = ["# Conditional relationship reuse: complete results", "",
             f"{len(records)} runs; adequacy **{adequacy['passed']}**; continuation screen **{summary['decision']}**.", "",
             "Values below are percentages. Intervals are paired seed bootstrap descriptions,",
             "not multiplicity-adjusted significance statements. All eight seeds are retained.", ""]
    for stream in config["schedules"]:
        lines += [f"## {stream}", "", "| Method | Inference-only probe | Valid old modes | Late gate | Late AUC | Actual final stream |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method in config["methods"]:
            row = aggregate[stream][method]
            scores = [row[k] for k in ("return_survival", "final_known", "final_gate", "late_auc", "final_online")]
            lines.append(f"| {method} | " + " | ".join("n/a" if v is None else f"{100*v:.2f}" for v in scores) + " |")
        lines += [""]
    lines += ["## Primary seed differences", "", "| Seed | Conditional | Pooled | Difference (pp) | Correct minus opposite (pp) |",
              "|---|---:|---:|---:|---:|"]
    for seed, difference in zip(config["seeds"], primary["differences"]):
        conditional, pooled = values["recurring", "conditional", seed], values["recurring", "pooled", seed]
        lines.append(f"| {seed} | {100*conditional['return_survival']:.2f} | {100*pooled['return_survival']:.2f} | "
                     f"{100*difference:+.2f} | {100*conditional['correct_minus_opposite']:+.2f} |")
    lines += ["", f"Mean paired difference: {100*primary['mean']:+.2f} pp; 95% interval "
              f"[{100*primary['lower']:+.2f}, {100*primary['upper']:+.2f}]; {primary['positive']}/8 positive.", "",
              "## Predeclared criteria", ""]
    lines += [f"- {name}: **{passed}**" for name, passed in criteria.items()]
    lines += ["", "The stable stream's known-mode scores exclude its never-trained mode. The",
              "unpredictable stream is scored on independent mixed modes; its final persistent-",
              "mode panels are OOD diagnostics and are retained only in the raw archive.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(runs=len(records), primary=primary, adequacy=adequacy,
                          criteria=criteria, decision=summary["decision"]), indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
