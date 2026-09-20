"""Untuned raw-pixel centroid diagnostic for the procedural shape generator.

From the repository root, run:
    python scripts/check_shapes_centroids.py

No model is trained and no validation observation changes sampling or settings.
The default protocol uses the first eight development experiences, seed 101,
256 training examples per domain, and independent validation examples.  It
also measures a capacity-256 reservoir after one shuffled pass through all
training examples in these eight experiences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from acp_cl.replay import ReservoirBuffer
from acp_cl.shapes import build_shapes_stream


ROOT = Path(__file__).resolve().parents[1]


def run(config_file: Path, seed: int, experiences: int, buffer_size: int) -> dict:
    config_bytes = config_file.read_bytes()
    source_config = json.loads(config_bytes.decode("utf-8-sig"))
    config = dict(source_config["data"])
    if config["dataset"] != "procedural_shapes":
        raise ValueError("the source config must describe procedural_shapes")
    if buffer_size <= 0 or buffer_size % 4:
        raise ValueError("buffer_size must be positive and divisible by four")
    if buffer_size // 4 > config["train_per_class"]:
        raise ValueError("the balanced sample requires more training images per class")
    config["n_experiences"] = experiences
    torch.set_num_threads(2)
    stream = build_shapes_stream(config, seed)
    views = [
        (experience.validation.tensors[0].float().flatten(1) / 255,
         experience.validation.tensors[1])
        for experience in stream.experiences
    ]
    if any(len(labels) == 0 for _, labels in views):
        raise ValueError("the diagnostic requires nonempty validation splits")

    def accuracies(raw: torch.Tensor, labels: torch.Tensor) -> list[float]:
        values = raw.float().flatten(1) / 255
        classes = labels.unique(sorted=True)
        centers = torch.stack([values[labels == label].mean(0) for label in classes])
        center_norm = centers.square().sum(1)
        return [
            float((classes[(center_norm - 2 * x @ centers.T).argmin(1)] == y).float().mean())
            for x, y in views
        ]

    balanced_count = buffer_size // 4
    balanced_name = f"balanced_first_{balanced_count}_per_label"
    random_name = f"first_{buffer_size}_shuffled_arrivals"
    matrices: dict[str, list[list[float]]] = {balanced_name: [], random_name: []}
    random_label_counts = []
    reservoir = ReservoirBuffer(buffer_size, seed=seed + 6151)
    for index, experience in enumerate(stream.experiences):
        x, y = experience.train.tensors
        balanced = torch.cat([torch.where(y == label)[0][:balanced_count] for label in range(4)])
        # Same first-epoch permutation as experiment.batches. A single complete
        # pass is used here, irrespective of the model-training step budget.
        generator = torch.Generator().manual_seed(seed + 12907 + 997 * index)
        order = torch.randperm(len(y), generator=generator)
        random = order[:buffer_size]
        random_label_counts.append(torch.bincount(y[random], minlength=4).tolist())
        matrices[balanced_name].append(accuracies(x[balanced], y[balanced]))
        matrices[random_name].append(accuracies(x[random], y[random]))
        reservoir.add(x[order], y[order])

    source_digest = hashlib.sha256()
    for name in ("shapes.py", "data.py", "replay.py"):
        source_digest.update(name.encode())
        source_digest.update((ROOT / "src" / "acp_cl" / name).read_bytes())
    try:
        source_name = str(config_file.resolve().relative_to(ROOT))
    except ValueError:
        source_name = str(config_file.resolve())
    result = {
        "seed": seed,
        "source_config": source_name,
        "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "generator_and_replay_source_sha256": source_digest.hexdigest(),
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
        "data": config,
        "protocol": {
            "classifier": "Euclidean nearest class centroid in flattened raw RGB / 255",
            "balanced_sampling": f"first {balanced_count} training images from each fixed label",
            "random_sampling": f"first {buffer_size} arrivals in the experiment's first-epoch permutation",
            "matrix_rows": "training experience in domain_order",
            "matrix_columns": "independent validation experience in domain_order",
            "reservoir": "one complete shuffled training pass per experience, independent membership RNG",
            "model_training_or_tuning": "none",
            "chance_accuracy": 0.25,
        },
        "domain_order": stream.metadata["domain_order"],
        "validation_examples_per_domain": [len(y) for _, y in views],
        "random_label_counts": random_label_counts,
    }
    for scheme, matrix in matrices.items():
        values = np.asarray(matrix)
        result[scheme] = {
            "within_domain": np.diag(values).tolist(),
            "mean_within_domain": float(np.trace(values) / experiences),
            "mean_other_domain": float((values.sum() - np.trace(values)) / (experiences * (experiences - 1)))
                if experiences > 1 else None,
            "all_train_eval_domain_matrix": matrix,
        }
    saved = reservoir.state_dict()
    final_accuracies = accuracies(saved["x"], saved["y"])
    result["pooled_reservoir"] = {
        "capacity": buffer_size,
        "num_seen": reservoir.num_seen,
        "label_counts": torch.bincount(saved["y"], minlength=4).tolist(),
        "per_domain_validation_accuracy": final_accuracies,
        "mean_accuracy": float(np.mean(final_accuracies)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v2_shapes_development.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "v2" / "generator_centroid_check.json")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--experiences", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=256)
    args = parser.parse_args()
    result = run(args.config, args.seed, args.experiences, args.buffer_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)
    for name, metrics in result.items():
        if isinstance(metrics, dict) and "mean_within_domain" in metrics:
            other = metrics["mean_other_domain"]
            other_text = "n/a" if other is None else f"{other:.2%}"
            print(f"{name}: same domain {metrics['mean_within_domain']:.2%}; "
                  f"other domains {other_text}")
    print(f"Pooled reservoir: {result['pooled_reservoir']['mean_accuracy']:.2%}")


if __name__ == "__main__":
    main()
