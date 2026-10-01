"""Paired, reproducible experiment orchestration; boundaries live only here."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch

from .data import Stream, build_stream
from .learner import Learner, METHODS, YOKED_METHODS
from .metrics import continual_metrics, normalized_auc
from .models import make_model
from .audit import controller_timing_audit, replay_centroid_accuracy
from .dual_path import DUAL_METHODS, DualPathLearner


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(data), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def stream_fingerprints(stream: Stream) -> dict[str, str]:
    """Hash actual ordered examples, independently of descriptive metadata."""
    signatures = {}
    for split in ("train", "validation", "test"):
        digest = hashlib.sha256()
        for experience in stream.experiences:
            digest.update(str(experience.index).encode())
            for tensor in getattr(experience, split).tensors:
                digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        signatures[split] = digest.hexdigest()
    return signatures


def runtime_identity(device: torch.device) -> dict:
    """Canonical execution runtime, independent of optional environment logging."""
    fingerprint = {"python": platform.python_version(),
                   "torch": str(torch.__version__), "numpy": str(np.__version__)}
    if device.type == "cuda":
        fingerprint.update(cuda_runtime=torch.version.cuda,
                           gpu=torch.cuda.get_device_name(device))
    return {"runtime_fingerprint": fingerprint, "runtime_sha256": config_hash(fingerprint)}


def environment() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_sha256": source_hash()}


def seed_everything(seed: int, threads: int = 4) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    return torch.device(name)


def validate_config(config: dict) -> None:
    for required in ("data", "model", "batch_size"):
        if required not in config:
            raise ValueError(f"missing config key: {required}")
    if config.get("eval_split", "validation") not in ("validation", "test"):
        raise ValueError("eval_split must be validation or test")
    for field, default in (("batch_size", 32), ("epochs", 1), ("eval_every_steps", 10)):
        if type(config.get(field, default)) is not int or config.get(field, default) < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config.get("steps_per_experience") is not None:
        if type(config["steps_per_experience"]) is not int or config["steps_per_experience"] < 1:
            raise ValueError("steps_per_experience must be a positive integer or omitted")
    if not 0 < config.get("early_fraction", 0.2) <= 1:
        raise ValueError("early_fraction must be in (0,1]")
    for field in ("early_examples", "retention_eval_every_experiences", "progress_every_experiences"):
        if field in config and (type(config[field]) is not int or config[field] < 1):
            raise ValueError(f"{field} must be a positive integer")
    for method in config.get("methods", ["er", "acp"]):
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")
    _validate_runner_options(config)


def _validate_runner_options(config: dict) -> None:
    for field in ("single_pass", "evaluation_cache_shared_pools"):
        if field in config and type(config[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")
    if config.get("single_pass", False):
        if type(config.get("epochs", 1)) is not int or config.get("epochs", 1) != 1:
            raise ValueError("single_pass requires exactly one epoch")
        if "steps_per_experience" in config:
            raise ValueError("single_pass requires steps_per_experience to be omitted")


def _batch_indices(experience, config: dict, seed: int):
    """The sole index schedule, shared by training and the evaluator's ID audit."""
    _validate_runner_options(config)
    size = len(experience.train)
    generator = torch.Generator().manual_seed(seed + 12907 + 997*experience.index)
    batch_size = config["batch_size"]
    limit = config.get("steps_per_experience")
    epochs = config.get("epochs", 1) if limit is None else int(np.ceil(limit*batch_size/size))+1
    count = 0
    for _ in range(epochs):
        for indices in torch.randperm(size, generator=generator).split(batch_size):
            if limit is not None and count >= limit:
                return
            count += 1
            yield indices


def batches(experience, config: dict, seed: int, *, include_indices: bool = False):
    """Repeated exposure is unchanged unless the explicit single_pass flag is set.

    Optional indices are for the runner's provenance audit, never the learner.
    """
    x, y = experience.train.tensors
    for indices in _batch_indices(experience, config, seed):
        batch = (x[indices], y[indices])
        yield (*batch, indices) if include_indices else batch


def _arrival_digest(ids: list[int]) -> str:
    """Method-independent SHA-256 of ordered little-endian signed int64 IDs."""
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _arrival_record(index: int, ids: list[int], batch_sizes: list[int]) -> dict:
    return {"experience": index, "current_examples": len(ids), "optimizer_steps": len(batch_sizes),
            "batch_sizes": batch_sizes, "ordered_base_image_ids": ids,
            "ordered_base_image_ids_sha256": _arrival_digest(ids)}


def _single_pass_plan(stream: Stream, config: dict, seed: int) -> tuple[list[dict], str]:
    """Validate unique base-image identities before constructing a learner."""
    _validate_runner_options(config)
    groups = stream.metadata.get("current_arrival_ids_by_experience")
    if not isinstance(groups, list) or len(groups) != len(stream.experiences):
        raise ValueError("single_pass requires current_arrival_ids_by_experience metadata")
    recorded_hashes = stream.metadata.get("current_arrival_id_hashes_by_experience")
    if recorded_hashes is not None and (not isinstance(recorded_hashes, list) or len(recorded_hashes) != len(groups)):
        raise ValueError("single_pass source ID hashes must cover every experience")
    original, planned = [], []
    for index, (experience, ids) in enumerate(zip(stream.experiences, groups)):
        if experience.index != index or not isinstance(ids, list) or len(ids) != len(experience.train) or not ids:
            raise ValueError("single_pass IDs must match each nonempty train dataset in experience order")
        if any(type(value) is not int or not 0 <= value < 2**63 for value in ids):
            raise ValueError("single_pass base-image IDs must be nonnegative signed-int64 integers")
        if recorded_hashes is not None and recorded_hashes[index] != _arrival_digest(ids):
            raise ValueError("single_pass source ID metadata hash mismatch")
        original.extend(ids)
        batches_ = [indices.tolist() for indices in _batch_indices(experience, config, seed)]
        ordered = [ids[position] for batch in batches_ for position in batch]
        planned.append(_arrival_record(index, ordered, [len(batch) for batch in batches_]))
    if len(set(original)) != len(original):
        raise ValueError("single_pass requires globally unique current base-image IDs")
    return planned, _arrival_digest(original)


def _arrival_artifact(records: list[dict], source_ids_sha256: str) -> dict:
    ordered = [value for record in records for value in record["ordered_base_image_ids"]]
    return {
        "schema_version": 1, "id_namespace": "source training-set positional base-image IDs",
        "hash_encoding": "SHA-256 of ordered little-endian signed int64 IDs; no method or domain labels",
        "source_ordered_ids_sha256": source_ids_sha256,
        "ordered_arrival_ids_sha256": _arrival_digest(ordered),
        "current_examples": len(ordered), "unique_current_base_images": len(set(ordered)),
        "optimizer_steps": sum(record["optimizer_steps"] for record in records),
        "completed_experiences": len(records), "experiences": records,
    }


def _verify_arrival_exposure(learner: Learner, artifact: dict) -> None:
    expected = artifact["current_examples"]
    if learner.cost["current_examples"] != expected or learner.buffer.num_seen != expected \
            or learner.step_number != artifact["optimizer_steps"]:
        raise ValueError("single_pass actual current exposure/replay arrivals/updates disagree with the ID audit")


def new_learner(config: dict, stream: Stream, method: str, seed: int,
                device: torch.device) -> Learner | DualPathLearner:
    seed_everything(seed + 16127, int(config.get("threads", 4)))
    if method in (*DUAL_METHODS, "dual_scratch"):
        return DualPathLearner(config, stream.input_shape, stream.num_classes, method, seed, device)
    model = make_model(config["model"], stream.input_shape, stream.num_classes,
                       width=int(config.get("width", 128))).to(device)
    return Learner(model, method, config, seed)


def train_experience(learner: Learner, experience, config: dict, seed: int,
                     *, arrival_callback=None) -> list[tuple[int, float]]:
    evaluation = getattr(experience, config.get("eval_split", "validation"))
    if len(evaluation) == 0:
        raise ValueError("selected evaluation split is empty")
    curve = [(0, learner.accuracy(evaluation))]
    examples, last_eval = 0, 0
    iterator = (batches(experience, config, seed) if arrival_callback is None else
                batches(experience, config, seed, include_indices=True))
    for step, batch in enumerate(iterator, start=1):
        x, y = batch[:2]
        learner.train_batch(x, y)
        if arrival_callback is not None:
            arrival_callback(batch[2])
        examples += len(y)
        if step % config.get("eval_every_steps", 10) == 0:
            curve.append((examples, learner.accuracy(evaluation)))
            last_eval = examples
    if examples != last_eval:
        curve.append((examples, learner.accuracy(evaluation)))
    return curve


def acquisition_auc(curve: list, config: dict) -> float:
    """Fixed presentation horizon when configured; never silently shorten it."""
    horizon = config.get("early_examples")
    if horizon is None:
        return normalized_auc(curve, config.get("early_fraction", 0.2))
    if curve[-1][0] < horizon:
        raise ValueError("early_examples exceeds an experience's actual presentation horizon")
    return normalized_auc(curve, horizon / curve[-1][0])


def scratch_reference(config: dict, stream: Stream, seed: int, device: torch.device,
                      output: Path, resume: bool = False,
                      reference_method: str = "finetune") -> dict | None:
    if not config.get("scratch_reference", True):
        return None
    if reference_method not in ("finetune", "dual_scratch"):
        raise ValueError("scratch reference must be replay-free")
    prefix = "scratch_dual" if reference_method == "dual_scratch" else "scratch"
    file = output / f"{prefix}_seed{seed}.json"
    identity = {"config_sha256": config_hash(config), "source_sha256": source_hash(),
                "seed": seed, "reference_method": reference_method,
                "execution_device": str(device), **runtime_identity(device)}
    if resume and file.exists():
        result = json.loads(file.read_text(encoding="utf-8"))
        if any(result.get(k) != v for k, v in identity.items()):
            raise ValueError(f"scratch cache configuration/source/runtime mismatch: {file}")
        return result
    start = time.perf_counter()
    curves, aucs, costs = [], [], []
    for experience in stream.experiences:
        learner = new_learner(config, stream, reference_method, seed, device)
        curve = train_experience(learner, experience, config, seed)
        curves.append(curve)
        aucs.append(acquisition_auc(curve, config))
        costs.append(learner.cost)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {**identity, "kind": "per-experience scratch diagnostic",
              "reference_method": reference_method, "early_auc": aucs,
              "curves": curves, "costs": costs, "wall_seconds": time.perf_counter()-start}
    write_json(file, result)
    return result


def run_one(config: dict, stream: Stream, method: str, seed: int,
            device: torch.device, output: Path, scratch: dict | None = None,
            resume: bool = False) -> dict:
    _validate_runner_options(config)
    expected_reference = "dual_scratch" if method in DUAL_METHODS else "finetune"
    if scratch is not None and scratch.get("reference_method", "finetune") != expected_reference:
        raise ValueError("scratch reference architecture does not match learner")
    single_pass = config.get("single_pass", False)
    planned_arrivals, source_ids_sha256 = _single_pass_plan(stream, config, seed) if single_pass else ([], "")
    cache_shared_pools = config.get("evaluation_cache_shared_pools", False)
    run_dir = output / f"{method}_seed{seed}"
    result_file = run_dir / "result.json"
    checkpoint_file = run_dir / "checkpoint.pt"
    identity = {"config_sha256": config_hash(config), "source_sha256": source_hash(),
                "method": method, "seed": seed, "execution_device": str(device),
                **runtime_identity(device)}
    allocation = None
    if method in YOKED_METHODS:
        source_dir = output / f"acp_v2_seed{seed}"
        if not (source_dir / "allocation.json").exists():
            raise ValueError("run acp_v2 for this seed before yoked diagnostics")
        source_result = json.loads((source_dir / "result.json").read_text(encoding="utf-8"))
        if source_result.get("method") != "acp_v2" or \
                any(source_result.get(k) != identity[k] for k in identity if k != "method"):
            raise ValueError("yoked source configuration/source/seed/device/runtime mismatch")
        source_bytes = (source_dir / "allocation.json").read_bytes()
        identity["allocation_source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        if identity["allocation_source_sha256"] != source_result.get("allocation_trace_sha256"):
            raise ValueError("yoked source allocation trace hash mismatch")
        allocation = json.loads(source_bytes)["trace"]
    if result_file.exists():
        if not resume:
            raise FileExistsError(f"run already exists: {run_dir}; choose a new --output or --resume")
        result = json.loads(result_file.read_text(encoding="utf-8"))
        if any(result.get(k) != v for k, v in identity.items()):
            raise ValueError(f"configuration/source/runtime mismatch: {result_file}")
        if single_pass:
            raw_arrivals = (run_dir / "current_arrivals.json").read_bytes()
            expected = _arrival_artifact(planned_arrivals, source_ids_sha256)
            if json.loads(raw_arrivals) != expected \
                    or result.get("current_arrival_audit", {}).get("artifact_sha256") != hashlib.sha256(raw_arrivals).hexdigest() \
                    or result.get("cost", {}).get("current_examples") != expected["current_examples"]:
                raise ValueError("single_pass completed result/current-arrival audit mismatch")
        return result
    checkpoint = None
    if resume and checkpoint_file.exists():
        # Only deserialize local, trusted checkpoints produced by this program.
        # Check identity before constructing a learner or loading model state.
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if any(checkpoint.get(k) != v for k, v in identity.items()):
            raise ValueError(f"checkpoint configuration/source/runtime mismatch: {checkpoint_file}")
        if single_pass:
            completed = checkpoint.get("completed")
            if type(completed) is not int or not 0 <= completed <= len(planned_arrivals) \
                    or checkpoint.get("current_arrivals") != _arrival_artifact(planned_arrivals[:completed], source_ids_sha256):
                raise ValueError("single_pass checkpoint/current-arrival audit mismatch")
    run_dir.mkdir(parents=True, exist_ok=True)
    learner = new_learner(config, stream, method, seed, device)
    if allocation is not None:
        learner.set_yoked_schedule(allocation)
    count = len(stream.experiences)
    matrix = np.full((count, count), np.nan)
    curves, experience_times = [], []
    actual_arrivals = []
    cache_counts = {"retention_cache_hits": 0, "retention_accuracy_calls": 0}
    completed, previous_seconds = 0, 0.0
    if checkpoint is not None:
        learner.load_state_dict(checkpoint["learner"])
        matrix = np.asarray(checkpoint["matrix"], dtype=float)
        curves, experience_times = checkpoint["curves"], checkpoint["experience_times"]
        completed, previous_seconds = checkpoint["completed"], checkpoint["wall_seconds"]
        if single_pass:
            actual_arrivals = copy.deepcopy(checkpoint["current_arrivals"]["experiences"])
            _verify_arrival_exposure(learner, checkpoint["current_arrivals"])
        if cache_shared_pools:
            saved_cache = checkpoint.get("evaluation_cache_counts")
            if not isinstance(saved_cache, dict) or set(saved_cache) != set(cache_counts) \
                    or any(type(value) is not int or value < 0 for value in saved_cache.values()):
                raise ValueError("invalid checkpoint evaluation-cache counters")
            cache_counts = dict(saved_cache)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(completed, count):
        experience_start = time.perf_counter()
        # This is the sole boundary-information path. Ordinary learners never
        # receive this callback, a domain ID, or an evaluation measurement.
        changes = stream.metadata.get("signal_change_flags", [False] + [True]*(count-1))
        if learner.oracle and i > 0 and changes[i]:
            learner.notify_oracle_boundary()
        if single_pass:
            observed_ids, batch_sizes = [], []
            ids = stream.metadata["current_arrival_ids_by_experience"][i]

            def observe_arrivals(indices):
                positions = indices.tolist()
                observed_ids.extend(ids[position] for position in positions)
                batch_sizes.append(len(positions))

            curves.append(train_experience(learner, stream.experiences[i], config, seed,
                                           arrival_callback=observe_arrivals))
            observed = _arrival_record(i, observed_ids, batch_sizes)
            if observed != planned_arrivals[i]:
                raise ValueError("single_pass actual arrival order differs from its planned ID sequence")
            actual_arrivals.append(observed)
            _verify_arrival_exposure(learner, _arrival_artifact(actual_arrivals, source_ids_sha256))
        else:
            curves.append(train_experience(learner, stream.experiences[i], config, seed))
        matrix[i, i] = curves[-1][-1][1]
        full_evaluation = (i+1) % config.get("retention_eval_every_experiences", 1) == 0 or i == count-1
        # This dictionary belongs only to the current post-training snapshot.
        # Holding each object prevents identity reuse; no entry crosses an update.
        snapshot_cache = {}
        if cache_shared_pools:
            current = getattr(stream.experiences[i], config.get("eval_split", "validation"))
            snapshot_cache[id(current)] = (current, matrix[i, i])
        for j in range(i if full_evaluation else 0):
            evaluation = getattr(stream.experiences[j], config.get("eval_split", "validation"))
            if cache_shared_pools and id(evaluation) in snapshot_cache:
                matrix[i, j] = snapshot_cache[id(evaluation)][1]
                cache_counts["retention_cache_hits"] += 1
            else:
                matrix[i, j] = learner.accuracy(evaluation)
                if cache_shared_pools:
                    snapshot_cache[id(evaluation)] = (evaluation, matrix[i, j])
                    cache_counts["retention_accuracy_calls"] += 1
        del snapshot_cache
        if device.type == "cuda":
            torch.cuda.synchronize()
        experience_times.append(time.perf_counter()-experience_start)
        if (i+1) % config.get("progress_every_experiences", 1) == 0 or i == count-1:
            print(f"{method} seed={seed} experience={i+1}/{count} "
                  f"observed_accuracy={np.nanmean(matrix[i]):.3f} "
                  f"phase={learner.controller.phase.value if learner.controller else 'uncontrolled'}",
                  flush=True)
        if config.get("checkpoint", True):
            checkpoint = {**identity, "learner": learner.state_dict(), "matrix": matrix.tolist(),
                          "curves": curves, "experience_times": experience_times, "completed": i+1,
                          "wall_seconds": previous_seconds + time.perf_counter()-start}
            if single_pass:
                checkpoint["current_arrivals"] = _arrival_artifact(actual_arrivals, source_ids_sha256)
            if cache_shared_pools:
                checkpoint["evaluation_cache_counts"] = dict(cache_counts)
            temporary = checkpoint_file.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_file)
    if device.type == "cuda":
        torch.cuda.synchronize()
    early_auc = [acquisition_auc(c, config) for c in curves]
    late_start = count // 2
    stats = continual_metrics(matrix)
    # Sparse retention evaluation must not average single-domain diagonals as
    # if they were complete seen-domain evaluations.
    full_rows = [float(matrix[i, :i+1].mean()) for i in range(count)
                 if np.isfinite(matrix[i, :i+1]).all()]
    stats["average_incremental_accuracy"] = float(np.mean(full_rows))
    stats["late_early_auc"] = float(np.mean(early_auc[late_start:]))
    stats["mean_early_auc"] = float(np.mean(early_auc))
    if method in DUAL_METHODS or config.get("report_acquisition_change", False):
        # Retained accuracy at arrival and subsequent improvement are distinct.
        # Neither alone is a causal measure of plasticity.
        stats["mean_arrival_accuracy"] = float(np.mean([c[0][1] for c in curves]))
        stats["mean_within_experience_gain"] = float(np.mean([c[-1][1] - c[0][1] for c in curves]))
    plasticity_gap = None
    if scratch is not None:
        plasticity_gap = [a-b for a, b in zip(early_auc, scratch["early_auc"])]
        stats["late_plasticity_gap"] = float(np.mean(plasticity_gap[late_start:]))
    phases = [event["phase"] for event in learner.log if event.get("monitor")]
    phase_counts = {phase: phases.count(phase) for phase in sorted(set(phases))}
    reopenings = sum(event.get("controller", {}).get("previous_phase") == "adult" and
                     event.get("controller", {}).get("phase") == "reopened"
                     for event in learner.log)
    result = {**identity, "config": copy.deepcopy(config), "environment": environment(),
              "eval_split": config.get("eval_split", "validation"), "class_order": stream.class_order,
              "stream_metadata": stream.metadata, "metrics": stats, "accuracy_matrix": matrix,
              "learning_curves": curves, "early_auc": early_auc,
              "scratch_early_auc": scratch["early_auc"] if scratch else None,
              "plasticity_gap": plasticity_gap, "cost": learner.cost,
              "wall_seconds": previous_seconds+time.perf_counter()-start,
              "experience_seconds": experience_times, "phase_monitor_counts": phase_counts,
              "reopening_events": reopenings, "diagnostics": learner.diagnostics()
                  if method in DUAL_METHODS else learner.engine.diagnostics(),
              "sensor_state_bytes": learner.sensor.nbytes() if learner.sensor else 0,
              "information_access": "true signal-domain change times; diagnostic only" if learner.oracle else
                  "offline ACP-v2 allocation trace; diagnostic only" if learner.yoked else "training stream only",
              "evaluation_schedule": {"retention_every": config.get("retention_eval_every_experiences", 1),
                                      "early_examples": config.get("early_examples"),
                                      "forgetting": "maximum over observed checkpoints; sparse estimates are lower bounds"},
              "allocation_summary": {
                  "mean_feature_gain": float(np.mean([e["effective_feature_gain"] for e in learner.allocation_trace]))
                      if learner.allocation_trace else None,
                  "summed_feature_data_displacement": sum(e["feature_data_displacement"] for e in learner.allocation_trace),
                  "steps": len(learner.allocation_trace)},
              "input_centroid_accuracy": replay_centroid_accuracy(learner, stream, config.get("eval_split", "validation")),
              "detector_audit": controller_timing_audit(learner.log, stream, curves, config),
              "model_parameters": sum(p.numel() for p in learner.model.parameters()),
              "replay_bytes": learner.buffer.nbytes(), "replay_examples": len(learner.buffer),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    if method in DUAL_METHODS:
        result["stream_fingerprints"] = stream_fingerprints(stream)
        result["allocation_summary"] = None
        result["scratch_reference_method"] = scratch.get("reference_method") if scratch else None
        result["architecture"] = "independent raw-input stable and fast predictors; summed logits"
        result["resource_comparison"] = "equal arrivals; report consolidation compute separately; not matched FLOPs"
        result["consolidation_events"] = learner.consolidations
    if learner.v3:
        result["stream_fingerprints"] = stream_fingerprints(stream)
        trace = learner.allocation_trace
        warmup = config.get("allocation_v3", {}).get("warmup_steps", 40)
        post_warmup = trace[warmup:]
        result["allocation_summary"].update({
            "mean_nominal_feature_gain": float(np.mean([e["nominal_feature_gain"] for e in trace])),
            "post_warmup_nominal_feature_gain": float(np.mean([
                e["nominal_feature_gain"] for e in post_warmup])) if post_warmup else None,
            "post_warmup_effective_feature_gain": float(np.mean([
                e["effective_feature_gain"] for e in post_warmup])) if post_warmup else None,
            "clipped_updates": sum(e["clip_scale"] < 1 for e in trace),
            "maximum_proposed_update_norm": max(e["proposed_update_norm"] for e in trace),
            "maximum_applied_update_norm": max(e["update_norm"] for e in trace),
            "summed_head_data_displacement": sum(e["head_data_displacement"] for e in trace),
            "gain_interpretation": "nominal before global step clipping; effective after clipping; feature parameters only",
            "norm_bound_scope": "complete optimizer displacement, including head, decay and anchors; excludes recycling resets",
        })
    if cache_shared_pools:
        result["evaluation_schedule"]["shared_pool_cache"] = {
            "enabled": True, "scope": "TensorDataset object identity within one post-training snapshot only",
            "reuse_current_diagonal": True, **cache_counts,
        }
    if single_pass:
        arrival_artifact = _arrival_artifact(actual_arrivals, source_ids_sha256)
        _verify_arrival_exposure(learner, arrival_artifact)
        arrival_path = run_dir / "current_arrivals.json"
        write_json(arrival_path, arrival_artifact)
        result["current_arrival_audit"] = {
            key: value for key, value in arrival_artifact.items() if key != "experiences"
        }
        result["current_arrival_audit"].update({
            "artifact": "current_arrivals.json", "artifact_sha256": hashlib.sha256(arrival_path.read_bytes()).hexdigest(),
            "reservoir_current_arrivals": learner.buffer.num_seen,
            "learner_receives_ids": False,
        })
    write_json(run_dir / "allocation.json", {"trace": learner.allocation_trace})
    result["allocation_trace_sha256"] = hashlib.sha256((run_dir / "allocation.json").read_bytes()).hexdigest()
    write_json(run_dir / "events.json", {"events": learner.log})
    write_json(result_file, result)
    return clean_json(result)


def _preflight_resume(output: Path, seeds: list[int], methods: list[str],
                      identity: dict, scratch_required: bool) -> None:
    """Validate consumed artifacts before overwriting provenance or loading learners."""
    required_methods = list(methods)
    if any(method in YOKED_METHODS for method in methods) and "acp_v2" not in required_methods:
        required_methods.append("acp_v2")
    for seed in seeds:
        candidates = []
        if scratch_required:
            candidates.append((output / f"scratch_seed{seed}.json",
                               {**identity, "seed": seed, "reference_method": "finetune"}))
            if any(method in DUAL_METHODS for method in methods):
                candidates.append((output / f"scratch_dual_seed{seed}.json",
                                   {**identity, "seed": seed, "reference_method": "dual_scratch"}))
        for method in required_methods:
            run_dir = output / f"{method}_seed{seed}"
            result_file = run_dir / "result.json"
            # A completed result is the consumed cache; its old checkpoint is unused.
            file = result_file if result_file.exists() else run_dir / "checkpoint.pt"
            candidates.append((file, {**identity, "seed": seed, "method": method}))
        for file, expected in candidates:
            if not file.exists():
                continue
            saved = (torch.load(file, map_location="cpu", weights_only=False)
                     if file.suffix == ".pt" else json.loads(file.read_text(encoding="utf-8")))
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError(f"resume configuration/source/runtime mismatch: {file}")


def run_suite(config: dict, *, output: Path, seeds: list[int], methods: list[str],
              device_name: str = "auto", resume: bool = False) -> list[dict]:
    validate_config(config)
    if not seeds or len(seeds) != len(set(seeds)) or len(methods) != len(set(methods)):
        raise ValueError("seeds/methods must be nonempty and unique")
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError("choose at least one recognized method")
    device = resolve_device(device_name)
    output.mkdir(parents=True, exist_ok=True)
    suite_identity = {"config_sha256": config_hash(config), "source_sha256": source_hash(),
                      "execution_device": str(device), **runtime_identity(device)}
    manifest_file = output / "manifest.json"
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"output exists: {output}; use --resume or a fresh directory")
        if any(old.get(k) != v for k, v in suite_identity.items()):
            raise ValueError("suite config/source changed or runtime/device mismatch; "
                             "use a fresh output directory")
    if resume:
        _preflight_resume(output, seeds, methods, suite_identity,
                          scratch_required=config.get("scratch_reference", True))
    write_json(manifest_file, {**suite_identity, "config": config, "seeds": seeds,
                               "methods": methods, "environment": environment(), "device": str(device)})
    results = []
    for seed in seeds:
        seed_everything(seed, int(config.get("threads", 4)))
        stream = build_stream(config["data"], seed)
        print(f"Prepared {config['data']['dataset']} seed={seed}, "
              f"{stream.num_classes} classes, device={device}", flush=True)
        scratch = scratch_reference(config, stream, seed, device, output, resume=resume) \
            if any(m not in DUAL_METHODS for m in methods) else None
        dual_scratch = scratch_reference(config, stream, seed, device, output, resume=resume,
                                         reference_method="dual_scratch") \
            if any(m in DUAL_METHODS for m in methods) else None
        ordered_methods = [m for m in methods if m not in YOKED_METHODS] + [m for m in methods if m in YOKED_METHODS]
        for method in ordered_methods:
            reference = dual_scratch if method in DUAL_METHODS else scratch
            results.append(run_one(config, stream, method, seed, device, output, reference, resume))
    return results
