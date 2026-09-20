"""Post-hoc color-shortcut diagnostic for trusted local shape checkpoints.

Example:
    python scripts/evaluate_shape_shortcuts.py runs/v2_shapes_long \
        --output runs/shape_shortcut_audit --device cpu

Only completed recurring-shape runs are eligible. Images are freshly generated
from the recorded domain parameters under their original color association and
under color_correlation=0. No learning, controller feedback, model selection, or
hyperparameter tuning occurs. This is not confirmatory held-out performance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from acp_cl.data import preprocess
from acp_cl.models import make_model
from acp_cl.shapes import ShapeDomain, _render_sample


ROOT = Path(__file__).resolve().parents[1]
AUDIT_NAMESPACE = 0x41554449  # ASCII "AUDI"; generator training/split namespace is 403.
DEFAULT_METHODS = ("er_recycle", "acp_v2", "acp_v2_oracle")
IDENTITY_FIELDS = ("config_sha256", "source_sha256", "method", "seed", "execution_device")


@dataclass
class AuditDomain:
    domain_id: int
    original: torch.Tensor
    uncorrelated: torch.Tensor
    labels: torch.Tensor
    original_color_correlation: float


def _config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "acp_cl").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sample_rng(seed: int, audit_seed: int, domain_id: int, label: int, index: int):
    seed, audit_seed = int(seed) % (2**64), int(audit_seed) % (2**64)
    sequence = np.random.SeedSequence([
        seed & 0xFFFFFFFF, seed >> 32, AUDIT_NAMESPACE,
        audit_seed & 0xFFFFFFFF, audit_seed >> 32, domain_id, label, index,
    ])
    return np.random.default_rng(sequence)


def make_audit_images(
    metadata: dict, seed: int, samples_per_class: int = 64, audit_seed: int = 20260920
) -> tuple[list[AuditDomain], str]:
    """Create paired audit images once per stream seed and reuse across methods.

    Geometry and texture draws precede the palette branch in _render_sample, so
    they match exactly. A successful original correlation branch skips a palette
    draw; the counterfactual always draws a random palette entry. Consequently
    subsequent color jitter and sensor-noise realizations can differ, although
    their distributions and parameters stay fixed. Independent per-sample RNGs
    prevent that difference from shifting any later sample's geometry.
    """
    if type(samples_per_class) is not int or samples_per_class < 1:
        raise ValueError("samples_per_class must be a positive integer")
    if metadata.get("regime") != "recurring":
        raise ValueError("this initial audit supports recurring domains only")
    domain_count = int(metadata["n_domains"])
    image_size = int(metadata["image_size"])
    if domain_count < 1 or image_size < 16:
        raise ValueError("invalid recorded domain count or image size")
    domains, digest = [], hashlib.sha256()
    for domain_id in range(domain_count):
        parameters = metadata["domain_parameters"][str(domain_id)]
        original_domain = ShapeDomain(**parameters)
        counterfactual_domain = replace(original_domain, color_correlation=0.0)
        original = np.empty((4 * samples_per_class, 3, image_size, image_size), dtype=np.uint8)
        uncorrelated = np.empty_like(original)
        labels = np.repeat(np.arange(4, dtype=np.int64), samples_per_class)
        for label in range(4):
            for index in range(samples_per_class):
                a = _render_sample(
                    _sample_rng(seed, audit_seed, domain_id, label, index),
                    label, original_domain, image_size,
                )
                b = _render_sample(
                    _sample_rng(seed, audit_seed, domain_id, label, index),
                    label, counterfactual_domain, image_size,
                )
                if (a.center, a.area_side, a.angle) != (b.center, b.area_side, b.angle) or \
                        not np.array_equal(a.alpha, b.alpha):
                    raise RuntimeError("counterfactual rendering changed paired shape geometry")
                position = label * samples_per_class + index
                original[position], uncorrelated[position] = a.image, b.image
        digest.update(str(domain_id).encode())
        digest.update(original.tobytes())
        digest.update(uncorrelated.tobytes())
        digest.update(labels.tobytes())
        domains.append(AuditDomain(
            domain_id, torch.from_numpy(original), torch.from_numpy(uncorrelated),
            torch.from_numpy(labels), float(original_domain.color_correlation),
        ))
    return domains, digest.hexdigest()


def load_final_model(result: dict, checkpoint_path: Path, device: torch.device):
    """Validate the local final checkpoint against its completion record."""
    if result["config_sha256"] != _config_hash(result["config"]):
        raise ValueError("result config payload does not match its recorded hash")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    for field in IDENTITY_FIELDS:
        if field not in result or checkpoint.get(field) != result[field]:
            raise ValueError(f"checkpoint/result identity mismatch: {field}")
    expected = result["config"]["data"]["n_experiences"]
    if checkpoint.get("completed") != expected:
        raise ValueError("the checkpoint does not contain all configured experiences")
    if checkpoint["learner"].get("method") != result["method"]:
        raise ValueError("checkpoint learner method differs from result method")
    if result.get("class_order") != [0, 1, 2, 3]:
        raise ValueError("shape audit requires the fixed four-label output space")
    config = result["config"]
    image_size = int(config["data"].get("image_size", 32))
    # Initialization is overwritten by the checkpoint; avoid perturbing ambient
    # CPU RNGs even though audit-image generation already uses private generators.
    with torch.random.fork_rng(devices=[]):
        model = make_model(config["model"], (3, image_size, image_size), 4, int(config.get("width", 128)))
    model.load_state_dict(checkpoint["learner"]["model"], strict=True)
    model.to(device).eval()
    return model, hashlib.sha256(checkpoint_bytes).hexdigest()


@torch.inference_mode()
def evaluate_model(model, domains: list[AuditDomain], device: torch.device, batch_size: int = 256) -> dict:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    model.eval()
    totals = {key: 0 for key in ("examples", "original_correct", "uncorrelated_correct", "both_correct",
                                "original_only_correct", "uncorrelated_only_correct", "both_wrong",
                                "same_prediction")}
    by_domain = []
    for domain in domains:
        predictions = []
        for raw in (domain.original, domain.uncorrelated):
            parts = []
            for start in range(0, len(raw), batch_size):
                x = preprocess(raw[start:start + batch_size].to(device), "procedural_shapes")
                parts.append(model(x).argmax(1).cpu())
            predictions.append(torch.cat(parts))
        original_correct = predictions[0] == domain.labels
        uncorrelated_correct = predictions[1] == domain.labels
        counts = {
            "examples": len(domain.labels),
            "original_correct": int(original_correct.sum()),
            "uncorrelated_correct": int(uncorrelated_correct.sum()),
            "both_correct": int((original_correct & uncorrelated_correct).sum()),
            "original_only_correct": int((original_correct & ~uncorrelated_correct).sum()),
            "uncorrelated_only_correct": int((~original_correct & uncorrelated_correct).sum()),
            "both_wrong": int((~original_correct & ~uncorrelated_correct).sum()),
            "same_prediction": int((predictions[0] == predictions[1]).sum()),
        }
        for key in totals:
            totals[key] += counts[key]
        by_domain.append({
            "domain_id": domain.domain_id,
            "original_color_correlation": domain.original_color_correlation,
            **counts,
            "original_accuracy": counts["original_correct"] / counts["examples"],
            "uncorrelated_accuracy": counts["uncorrelated_correct"] / counts["examples"],
            "accuracy_drop": (counts["original_correct"] - counts["uncorrelated_correct"]) / counts["examples"],
        })
    if not totals["examples"]:
        raise ValueError("at least one nonempty audit domain is required")
    total = totals["examples"]
    return {
        **totals,
        "original_accuracy": totals["original_correct"] / total,
        "uncorrelated_accuracy": totals["uncorrelated_correct"] / total,
        "accuracy_drop": (totals["original_correct"] - totals["uncorrelated_correct"]) / total,
        "prediction_agreement": totals["same_prediction"] / total,
        "by_domain": by_domain,
    }


def run_audit(
    directory: Path,
    output: Path,
    methods: list[str] | tuple[str, ...] = DEFAULT_METHODS,
    seeds: list[int] | None = None,
    samples_per_class: int = 64,
    audit_seed: int = 20260920,
    device_name: str = "cpu",
    batch_size: int = 256,
) -> dict:
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty and unique")
    if seeds is not None and (not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError("explicit seeds must be nonempty and unique")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    selected, skipped = [], []
    for result_path in sorted(directory.glob("*/result.json")):
        payload = result_path.read_bytes()
        result = json.loads(payload)
        if result.get("method") not in methods or (seeds is not None and result.get("seed") not in seeds):
            continue
        data = result.get("config", {}).get("data", {})
        if data.get("dataset") != "procedural_shapes" or data.get("regime", "recurring") != "recurring":
            skipped.append({"path": str(result_path), "reason": "only recurring procedural shapes are supported"})
            continue
        checkpoint = result_path.parent / "checkpoint.pt"
        if not checkpoint.exists():
            skipped.append({"path": str(result_path), "reason": "completed result has no checkpoint"})
            continue
        selected.append((result_path, result, checkpoint, hashlib.sha256(payload).hexdigest()))
    if not selected:
        raise ValueError("no completed selected shape checkpoints are available")
    identities = {(r["config_sha256"], r["source_sha256"]) for _, r, _, _ in selected}
    if len(identities) != 1:
        raise ValueError("refusing to pool different training configurations or source versions")
    pairs = [(result["method"], result["seed"]) for _, result, _, _ in selected]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate method/seed completion records")
    audit_source = _source_hash()
    by_seed: dict[int, tuple[dict, list[AuditDomain], str]] = {}
    runs = []
    for result_path, result, checkpoint, result_hash in selected:
        seed = result["seed"]
        metadata = result["stream_metadata"]
        if seed not in by_seed:
            domains, image_hash = make_audit_images(metadata, seed, samples_per_class, audit_seed)
            by_seed[seed] = (metadata, domains, image_hash)
        else:
            prior_metadata, domains, image_hash = by_seed[seed]
            if prior_metadata != metadata:
                raise ValueError("paired methods have different recorded stream metadata")
        model, checkpoint_hash = load_final_model(result, checkpoint, device)
        metrics = evaluate_model(model, domains, device, batch_size)
        runs.append({
            "method": result["method"], "seed": seed,
            "information_access_during_training": result.get("information_access", "training stream only"),
            "training_identity": {field: result[field] for field in IDENTITY_FIELDS},
            "training_environment": result.get("environment"),
            "audit_source_matches_training": result["source_sha256"] == audit_source,
            "result_path": str(result_path.resolve()), "result_sha256": result_hash,
            "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
            "audit_images_sha256": image_hash,
            "trained_domain_ids": sorted(set(metadata["domain_order"])),
            **metrics,
        })
        print(f"{result['method']} seed={seed}: original={metrics['original_accuracy']:.2%}, "
              f"uncorrelated={metrics['uncorrelated_accuracy']:.2%}, "
              f"drop={100 * metrics['accuracy_drop']:+.2f} pp", flush=True)
        del model
    method_summaries = []
    for method in methods:
        available = [run for run in runs if run["method"] == method]
        if not available:
            continue
        method_summaries.append({
            "method": method, "n_seeds": len(available), "seeds": sorted(run["seed"] for run in available),
            **{name: float(np.mean([run[name] for run in available]))
               for name in ("original_accuracy", "uncorrelated_accuracy", "accuracy_drop")},
        })
    available_seeds = sorted({run["seed"] for run in runs})
    missing = [{"method": method, "seed": seed} for method in methods for seed in available_seeds
               if (method, seed) not in set(pairs)]
    config = selected[0][1]["config"]
    summary: dict[str, Any] = {
        "status": "post-hoc shortcut diagnostic; not confirmatory held-out performance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(directory.resolve()),
        "audit_source_sha256": audit_source,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audit_runtime": {"python_device": str(device), "torch": torch.__version__, "numpy": np.__version__,
                          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "training_config": config,
        "protocol": {
            "audit_seed": audit_seed, "rng_namespace": AUDIT_NAMESPACE,
            "samples_per_class_per_domain": samples_per_class, "labels": [0, 1, 2, 3],
            "domain_ids": list(range(config["data"].get("n_domains", 8))),
            "domain_parameters_source": "recorded training stream metadata; no regenerated or fitted nuisance parameters",
            "conditions": ["original recorded color_correlation", "color_correlation=0 (uniform palette independent of label)"],
            "paired_latents": "same shape, center, scale, rotation, silhouette mask, and pre-palette texture draws",
            "pairing_limit": "palette-branch RNG consumption can change subsequent background/foreground color jitter and sensor-noise realizations; their distributions remain unchanged",
            "freshness": "independent per-sample audit namespace, disjoint from training/validation/test seed construction",
            "model_updates_controller_feedback_or_tuning": "none",
            "aggregation": "equal examples per label/domain; method summaries average available seeds and report their identities",
            "interpretation": "sensitivity to removing a rendering shortcut; not proof of shape understanding or a causal effect isolated from all pixel changes",
        },
        "methods": method_summaries,
        "runs": runs,
        "missing_method_seed_pairs_among_available_seeds": missing,
        "skipped": skipped,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "shortcut_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_markdown(output / "shortcut_audit.md", summary)
    return summary


def _write_markdown(path: Path, summary: dict) -> None:
    protocol = summary["protocol"]
    lines = [
        "# Post-hoc shape shortcut diagnostic", "",
        "This exploratory audit is **not confirmatory held-out performance**. No model updates, controller feedback, or tuning were performed.", "",
        f"Each model receives {protocol['samples_per_class_per_domain']} fresh images per label in each of "
        f"{len(protocol['domain_ids'])} configured domains, paired with renders whose foreground palette is independent of the label.", "",
        "Geometry and pre-palette texture draws match. Palette-branch RNG consumption can change color jitter and sensor-noise realizations; their distributions remain fixed. Positive drops indicate sensitivity to this rendering change.", "",
        "| Method | Available seeds | Original accuracy | Uncorrelated colors | Drop, pp |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["methods"]:
        lines.append(f"| {row['method']} | {', '.join(map(str, row['seeds']))} | "
                     f"{100 * row['original_accuracy']:.2f}% | {100 * row['uncorrelated_accuracy']:.2f}% | "
                     f"{100 * row['accuracy_drop']:+.2f} |")
    lines.extend(["", "| Method | Seed | Original accuracy | Uncorrelated colors | Drop, pp |",
                  "|---|---:|---:|---:|---:|"])
    for row in summary["runs"]:
        lines.append(f"| {row['method']} | {row['seed']} | {100 * row['original_accuracy']:.2f}% | "
                     f"{100 * row['uncorrelated_accuracy']:.2f}% | {100 * row['accuracy_drop']:+.2f} |")
    lines.extend(["", "Oracle methods had access to true domain-change times during training. Yoked methods, when present, consume completed ACP-v2 allocation traces; the gain variant also uses the future run-average gain. They are extra-information diagnostics, not task-free competitors. Per-domain results, source/config/checkpoint hashes, and paired correctness counts are recorded in shortcut_audit.json."])
    if summary["missing_method_seed_pairs_among_available_seeds"]:
        lines.append("Some requested method/seed runs were incomplete or unavailable at discovery; method means use the explicitly listed available seeds.")
    if any(not row["audit_source_matches_training"] for row in summary["runs"]):
        lines.append("Audit source differs from training source; both hashes are recorded. Interpret these results as an additional implementation-dependent diagnostic.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=ROOT / "runs" / "v2_shapes_long")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--audit-seed", type=int, default=20260920)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    run_audit(args.directory, args.output, args.methods, args.seeds, args.samples_per_class,
              args.audit_seed, args.device, args.batch_size)


if __name__ == "__main__":
    main()
