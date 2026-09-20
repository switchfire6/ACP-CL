"""Verify actual completed-run replay pairing and optional stream prefixes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def buffer_hash(state):
    digest = hashlib.sha256()
    tensors = [("x", state["x"]), ("y", state["y"]),
               ("membership_rng", state["reservoir_rng_state"])]
    tensors += sorted(state["sample_rng_states"].items())
    for key, tensor in tensors:
        digest.update(key.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    digest.update(str(state["num_seen"]).encode())
    return digest.hexdigest()


def audit(directory: Path, prefix: Path | None = None):
    manifest = read(directory / "manifest.json")
    findings = []
    for seed in manifest["seeds"]:
        signatures, exposure, prefixes = {}, {}, {}
        for method in manifest["methods"]:
            location = directory / f"{method}_seed{seed}"
            result = read(location / "result.json")
            # These are trusted local checkpoints produced by this project.
            saved = torch.load(location / "checkpoint.pt", map_location="cpu", weights_only=False)
            if saved["completed"] != manifest["config"]["data"]["n_experiences"]:
                raise ValueError(f"incomplete checkpoint: {location}")
            for key in ("config_sha256", "source_sha256", "method", "seed", "execution_device"):
                if saved[key] != result[key]:
                    raise ValueError(f"checkpoint/result identity mismatch: {location}")
            signatures[method] = buffer_hash(saved["learner"]["buffer"])
            exposure[method] = {k: result["cost"][k] for k in (
                "current_examples", "replay_examples", "probe_forward_examples",
                "train_forward_calls", "backward_calls")}
            earlier = prefix / f"{method}_seed{seed}" if prefix else None
            # Constant gain uses the entire source-run budget, so its prefix
            # intentionally differs when the total horizon changes.
            if earlier and earlier.exists() and method != "er_recycle_yoked_gain":
                prior = read(earlier / "result.json")
                current_trace = read(location / "allocation.json")["trace"]
                previous_trace = read(earlier / "allocation.json")["trace"]
                equal = (result["learning_curves"][:len(prior["learning_curves"])] == prior["learning_curves"]
                         and current_trace[:len(previous_trace)] == previous_trace)
                if not equal:
                    raise ValueError(f"training prefix differs: {location}")
                prefixes[method] = len(previous_trace)
        if len(set(signatures.values())) != 1:
            raise ValueError(f"paired replay contents/RNGs differ for seed {seed}")
        if len({json.dumps(v, sort_keys=True) for v in exposure.values()}) != 1:
            raise ValueError(f"paired training exposure differs for seed {seed}")
        findings.append({"seed": seed, "methods": list(signatures),
                         "buffer_and_sampling_rng_sha256": next(iter(signatures.values())),
                         "common_training_exposure": next(iter(exposure.values())),
                         "exact_prefix_updates_verified": prefixes})
    return {"suite": directory.name, "config_sha256": manifest["config_sha256"],
            "source_sha256": manifest["source_sha256"], "checks_passed": True, "seeds": findings,
            "scope": "actual replay tensors, membership/train/probe RNGs, exposure counts; optional exact curves/allocation prefixes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.directory, args.prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    print(f"Verified {len(value['seeds'])} seed groups in {args.directory}")
