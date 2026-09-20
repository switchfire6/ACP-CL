"""Evaluator-only task geometry and controller timing diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from .data import preprocess


@torch.no_grad()
def replay_centroid_accuracy(learner, stream, split: str) -> float | None:
    """Nearest raw-input centroid using only the final replay buffer.

    This diagnostic never trains the model or feeds its result to the learner.
    No reservoir sampling RNG is advanced. Missing classes cannot be predicted.
    """
    if not len(learner.buffer):
        return None
    saved = learner.buffer.state_dict()
    x = preprocess(saved["x"], learner.dataset).flatten(1)
    labels = saved["y"].unique(sorted=True)
    centers = torch.stack([x[saved["y"] == label].mean(0) for label in labels])
    center_norm = centers.square().sum(1)
    correct, total = 0, 0
    for experience in stream.experiences:
        evaluation = getattr(experience, split)
        raw, truth = evaluation.tensors
        for start in range(0, len(truth), 256):
            values = preprocess(raw[start:start+256], learner.dataset).flatten(1)
            # The common ||x||^2 term does not affect the argmin.
            prediction = labels[(center_norm - 2*values @ centers.T).argmin(1)]
            correct += int((prediction == truth[start:start+256]).sum())
            total += len(prediction)
    return correct / total


def controller_timing_audit(events: list[dict], stream, curves: list, config: dict) -> dict:
    """Descriptive boundary proximity; not a calibrated change-detector score.

    Acquisition horizons count examples, while boundaries count optimizer
    updates. Shape configs use full equal batches; arbitrary partial batches
    are handled through the known per-experience loader length.
    """
    starts = [0]
    for experience in stream.experiences[:-1]:
        updates = config.get("steps_per_experience")
        if updates is None:
            updates = int(np.ceil(len(experience.train) / config["batch_size"])) * config.get("epochs", 1)
        starts.append(starts[-1] + updates)
    flags = stream.metadata.get("signal_change_flags", [False]+[True]*(len(starts)-1))
    boundaries = [step for step, changed in zip(starts, flags) if changed and step > 0]
    cc = config.get("controller", {})
    warmup = cc.get("scaffold_steps", 300) + cc.get("min_open_steps", 300) + \
        cc.get("closing_windows", 6) * cc.get("monitor_interval", 25)
    eligible = [step for step in boundaries if step >= warmup]
    tolerance = config.get("detection_tolerance_steps", 2*cc.get("monitor_interval", 25))
    reopening_steps = [e["step"] for e in events
                       if e.get("controller", {}).get("previous_phase") == "adult"
                       and e.get("controller", {}).get("phase") == "reopened"]
    delays = [min([r-b for r in reopening_steps if 0 < r-b <= tolerance], default=None)
              for b in eligible]
    outside = sum(not any(0 < r-b <= tolerance for b in boundaries) for r in reopening_steps)
    return {"signal_change_steps": boundaries, "post_warmup_changes": len(eligible),
            "tolerance_steps": tolerance, "reopening_steps": reopening_steps,
            "changes_with_reopening_in_window": sum(d is not None for d in delays),
            "delays_within_window": [d for d in delays if d is not None],
            "reopenings_outside_change_windows": outside,
            "interpretation": "descriptive proximity only; endogenous health/loss can also justify reopening"}
