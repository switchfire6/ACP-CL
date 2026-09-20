"""Keep compact, reviewable pilot artifacts in git; leave weights/data in runs/."""

import argparse
import json
from pathlib import Path
import shutil


def reviewable_json(value, depth: int = 0) -> str:
    """Keep numeric matrix rows compact while retaining object indentation."""
    indent, inner = "  " * depth, "  " * (depth+1)
    if isinstance(value, dict) and value:
        return "{\n" + ",\n".join(inner + json.dumps(key) + ": " + reviewable_json(item, depth+1)
                                  for key, item in value.items()) + "\n" + indent + "}"
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        return "[\n" + ",\n".join(inner + reviewable_json(item, depth+1) for item in value) + "\n" + indent + "]"
    return json.dumps(value, allow_nan=False)


def archive(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "summary.json", "summary.csv", "summary.md", "overview.png", "overview.svg"):
        shutil.copy2(source / name, destination / name)
        if name.endswith(".svg"):
            # Matplotlib's multiline path attributes include trailing spaces.
            # Newlines preserve the separators while keeping git diffs clean.
            path = destination / name
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines)+"\n", encoding="utf-8", newline="\n")
    fields = ("method", "seed", "config_sha256", "source_sha256", "execution_device", "eval_split",
              "class_order", "metrics", "accuracy_matrix", "early_auc", "scratch_early_auc",
              "plasticity_gap", "phase_monitor_counts", "reopening_events", "diagnostics",
              "cost", "wall_seconds", "replay_bytes", "model_parameters", "peak_cuda_bytes",
              "information_access", "evaluation_schedule", "allocation_summary", "allocation_source_sha256",
              "input_centroid_accuracy", "detector_audit", "sensor_state_bytes", "allocation_trace_sha256",
              "runtime_fingerprint", "runtime_sha256")
    results = []
    for file in sorted(source.glob("*/result.json")):
        result = json.loads(file.read_text(encoding="utf-8"))
        item = {key: result[key] for key in fields if key in result}
        events = json.loads((file.parent / "events.json").read_text(encoding="utf-8"))["events"]
        item["transitions"] = [{"step": e["step"], "phase": e["phase"],
                                "reason": e["controller"].get("reason")}
                               for e in events if e.get("controller", {}).get("transition")]
        item["risk_contractions"] = sum(bool(e.get("controller", {}).get("safety_contracted")) for e in events)
        results.append(item)
    (destination / "individual_results.json").write_text(reviewable_json(results)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    archive(args.source, args.destination)
