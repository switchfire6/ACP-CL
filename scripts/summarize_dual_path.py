"""Portable, descriptive dual-path pilot report; no small-cohort inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from acp_cl.experiment import config_hash, write_json


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(directory: Path, output: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = {k: manifest[k] for k in
                ("config_sha256", "source_sha256", "execution_device", "runtime_sha256")}
    config, methods, seeds = manifest["config"], manifest["methods"], manifest["seeds"]
    if identity["config_sha256"] != config_hash(config):
        raise ValueError("manifest configuration hash mismatch")
    expected = {(m, s) for m in methods for s in seeds}
    found, records, arrival_hashes = set(), [], {}
    for path in sorted(directory.glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        pair = result["method"], result["seed"]
        if pair not in expected or pair in found:
            raise ValueError("unexpected or duplicate method/seed result")
        found.add(pair)
        if any(result.get(k) != v for k, v in identity.items()):
            raise ValueError("result configuration/source/runtime mismatch")
        events_path = path.parent / "events.json"
        events = json.loads(events_path.read_text(encoding="utf-8"))["events"]
        consolidation = [e for e in events if e.get("consolidation")]
        if result["method"] == "dual_consolidate":
            settings = config["dual_path"]
            expected_steps = list(range(settings["consolidation_interval"],
                                        result["cost"]["online_optimizer_steps"] + 1,
                                        settings["consolidation_interval"]))
            if [e["step"] for e in consolidation] != expected_steps:
                raise ValueError("consolidation cadence mismatch")
            if result["cost"]["consolidation_optimizer_steps"] != \
                    len(expected_steps) * settings["consolidation_steps"]:
                raise ValueError("consolidation update accounting mismatch")
        if config.get("single_pass"):
            arrival_path = path.parent / "current_arrivals.json"
            audit = result["current_arrival_audit"]
            if audit["artifact_sha256"] != _digest(arrival_path):
                raise ValueError("arrival artifact hash mismatch")
            artifact = json.loads(arrival_path.read_text(encoding="utf-8"))
            ids = [i for e in artifact["experiences"] for i in e["ordered_base_image_ids"]]
            if len(ids) != len(set(ids)) or len(ids) != result["cost"]["current_examples"]:
                raise ValueError("current examples are not unique or accounting differs")
            seed = result["seed"]
            if seed in arrival_hashes and arrival_hashes[seed] != artifact["ordered_arrival_ids_sha256"]:
                raise ValueError("methods saw different current arrivals")
            arrival_hashes[seed] = artifact["ordered_arrival_ids_sha256"]
        probes = []
        for event in consolidation:
            stages = event.get("probe", {})
            a, b, c = (stages.get(k) for k in
                       ("after_online", "after_consolidation_before_reset", "after_reset"))
            if all(stage is not None for stage in (a, b, c)):
                probes.append({"step": event["step"],
                               "net_transfer_accuracy_pp": 100 * (c["accuracy"] - a["accuracy"]),
                               "reset_accuracy_pp": 100 * (c["accuracy"] - b["accuracy"]),
                               "net_transfer_loss": c["loss"] - a["loss"], "stages": stages})
        keys = ("method", "seed", "metrics", "learning_curves", "early_auc", "scratch_early_auc",
                "scratch_reference_method", "plasticity_gap", "accuracy_matrix", "cost",
                "diagnostics", "model_parameters", "replay_bytes", "peak_cuda_bytes", "wall_seconds",
                "current_arrival_audit", "stream_fingerprints", "class_order")
        records.append({**{k: result[k] for k in keys if k in result},
                        "consolidation_probes": probes,
                        "source_artifacts": {"result": str(path), "result_sha256": _digest(path),
                                             "events_sha256": _digest(events_path)}})
    if found != expected:
        raise ValueError(f"incomplete suite: missing {sorted(expected - found)}")
    rows = []
    for method in methods:
        runs = [r for r in records if r["method"] == method]
        rows.append({"method": method, "n_seeds": len(runs), **{
            name: statistics.mean(r["metrics"][name] for r in runs)
            for name in ("final_accuracy", "forgetting", "late_early_auc", "late_plasticity_gap")
            if all(name in r["metrics"] for r in runs)}})
    archive = {"status": "initial feasibility diagnostic; two-seed descriptions are not significance tests",
               **identity, "config": config, "methods": methods, "seeds": seeds,
               "summary": rows, "runs": records, "manifest_sha256": _digest(directory / "manifest.json"),
               "analysis_script_sha256": _digest(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "archive.json", archive)
    lines = ["# Dual-path representation-learning pilot", "",
             "Initial feasibility diagnostic. All planned runs are included; no significance claim.", "",
             f"Source SHA-256: `{identity['source_sha256']}`.",
             f"Configuration SHA-256: `{identity['config_sha256']}`.", "",
             "Means below are percentages; forgetting and scratch differences are percentage points.", "",
             "| Method | Seeds | Final accuracy | Forgetting | Late acquisition AUC | Scratch difference |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [f"{100 * row[k]:.2f}" if k in row else "n/a" for k in
                  ("final_accuracy", "forgetting", "late_early_auc", "late_plasticity_gap")]
        lines.append(f"| {row['method']} | {row['n_seeds']} | " + " | ".join(values) + " |")
    lines += ["", "## Every seed", "",
              "| Method | Seed | Final accuracy | Late AUC | Scratch difference |",
              "|---|---:|---:|---:|---:|"]
    for r in records:
        m = r["metrics"]
        gap = f"{100*m['late_plasticity_gap']:.2f}" if "late_plasticity_gap" in m else "n/a"
        lines.append(f"| {r['method']} | {r['seed']} | {100*m['final_accuracy']:.2f} | "
                     f"{100*m['late_early_auc']:.2f} | {gap} |")
    lines += ["", "## Resource accounting", "",
              "Equal arrivals do not imply equal memory or compute. Branch-example counts are not FLOPs.", "",
              "| Method | Seed | Parameters | Replay bytes | Current examples | Online replay examples | Consolidation examples | Extra updates |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        c = r["cost"]
        lines.append(f"| {r['method']} | {r['seed']} | {r['model_parameters']} | {r['replay_bytes']} | "
                     f"{c['current_examples']} | {c['replay_examples']} | "
                     f"{c.get('consolidation_examples', 0)} | {c.get('consolidation_optimizer_steps', 0)} |")
    lines += ["", "## Consolidation and renewal", "",
              "Mean changes on a paired training-memory probe, equally weighted over consolidation events.",
              "These are overlapping, potentially in-pool examples, not independent validation observations.", "",
              "| Seed | Events with probes | Net transfer accuracy change, pp | Reset-only change, pp | Net loss change |",
              "|---|---:|---:|---:|---:|"]
    for r in records:
        p = r["consolidation_probes"]
        if p:
            lines.append(f"| {r['seed']} | {len(p)} | " + " | ".join(
                f"{statistics.mean(e[k] for e in p):+.4f}" for k in
                ("net_transfer_accuracy_pp", "reset_accuracy_pp", "net_transfer_loss")) + " |")
    lines += ["", "The two-pathway scratch reference trains both networks jointly on each new experience without replay.",
              "It matches architecture, not consolidation policy or total compute. Scratch differences are diagnostic, not pure plasticity.",
              "AUC integrates current presentations and includes any intervening consolidation work.",
              "Low forgetting can reflect weak initial learning. Inspect acquisition together with retention.",
              "The frozen-path ablation keeps the stable network at random initialization; no method uses pretrained weights.",
              "Ten class groups and two seeds cannot establish sustained plasticity or a general architectural advantage.", "",
              "See [the protocol](../../docs/dual_path.md) and [the portable archive](archive.json).", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
    print(args.output / "summary.md")
