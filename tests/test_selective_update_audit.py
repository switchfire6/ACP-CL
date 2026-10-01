"""Saved-state and independently recomputed probe checks for selective learning."""

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_selective_updates import audit, signature
from acp_cl.selective_updates import study


@pytest.fixture(scope="module")
def completed_cohort(tmp_path_factory):
    config = json.loads(Path("configs/selective_updates_smoke.json").read_text())
    config.update(prefix_size=16, episode_size=16, probe_every=8, updates_per_batch=2)
    directory = tmp_path_factory.mktemp("selective_audit")/"run"
    study.run_suite(config, directory, protocol=Path("docs/selective_updates_protocol.md"))
    return directory, config


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_complete_checkpoint_audit_keeps_all_learning_states_unchanged(completed_cohort, tmp_path):
    directory, config = completed_cohort
    files = sorted(directory.glob("*/*.pt"))+sorted(directory.glob("*/*/*.pt"))
    before = {path: signature(load(path)["learner"]) for path in files}
    result = audit(directory, tmp_path/"audit.json")
    assert result["passed"]
    assert result["prefix_checkpoints"] == 2
    assert result["before_checkpoints"] == result["final_checkpoints"] == 12
    assert result["matched_carried_arm_groups"] == 2
    assert "not independently retrained" in result["scope"]
    assert {path: signature(load(path)["learner"]) for path in files} == before
    expected_steps = config["episode_size"]//config["batch_size"]*config["updates_per_batch"]
    for check in result["checks"]:
        assert check["extra_work_budget_verified"]["steps"] == expected_steps
        assert check["endpoint_probes_recomputed"]


@pytest.mark.parametrize("fault", (
    "optimizer_before", "memory_rng_before", "hidden_before", "counters_before",
    "fresh_history_before", "adam_steps_after", "negative_adam_after", "frozen_after",
    "memory_rng_after", "endpoint_after", "marginal_after", "sampled_age_after",
))
def test_auditor_rejects_checkpoint_corruption_even_with_matching_json(
        fault, completed_cohort, tmp_path):
    source, config = completed_cohort
    directory = tmp_path/"tampered"
    shutil.copytree(source, directory)
    arm = "fresh" if fault == "fresh_history_before" else "reference"
    episode = directory/f"conditional_{config['seeds'][0]}"/f"stage_1_{arm}"
    checkpoint = episode/("before.pt" if fault.endswith("before") else "checkpoint.pt")
    saved = load(checkpoint)
    learner, record = saved["learner"], saved["record"]
    if fault == "optimizer_before":
        next(iter(learner.optimizer.state.values()))["exp_avg"].add_(.01)
    elif fault in ("memory_rng_before", "memory_rng_after"):
        learner.memory.sample()
    elif fault == "hidden_before":
        packet = learner.memory.packets[0]
        packet.oracle_modes = np.zeros(len(packet.query), dtype=np.int64)
    elif fault == "counters_before":
        learner.selective_stats["steps"] = 1
    elif fault == "fresh_history_before":
        learner.history.survival[0, 0] ^= 1
    elif fault == "adam_steps_after":
        next(iter(learner.optimizer.state.values()))["step"].add_(1)
    elif fault == "negative_adam_after":
        next(iter(learner.optimizer.state.values()))["exp_avg_sq"].fill_(-1)
    elif fault == "frozen_after":
        next(learner.model.parameters()).requires_grad_(False)
    elif fault == "endpoint_after":
        record["after"]["replicates"][0]["curve"][0]["metrics"]["brier"] += .01
    elif fault == "marginal_after":
        action = next(i for i, count in enumerate(record["marginal_count"]) if count)
        previous = record["marginal_success"][action][0]
        record["marginal_success"][action][0] = previous-1 if previous else 1
    else:
        aggregate = learner.selective_stats["summaries"]["sampled_age_packets"]
        for field in ("min", "max"):
            aggregate[field] += 1
        aggregate["sum"] += aggregate["count"]
        record["diagnostics"] = learner.diagnostics()
    torch.save(saved, checkpoint)
    if fault.endswith("after"):
        (episode/"result.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    with pytest.raises((ValueError, AssertionError)):
        audit(directory, tmp_path/"rejected.json")
    assert not (tmp_path/"rejected.json").exists()


def test_auditor_rejects_identity_disagreement_in_a_trusted_checkpoint(
        completed_cohort, tmp_path):
    source, config = completed_cohort
    directory = tmp_path/"identity"
    shutil.copytree(source, directory)
    path = directory/f"conditional_{config['seeds'][0]}"/"stage_1_reference"/"before.pt"
    saved = load(path)
    saved["identity"] = dict(saved["identity"], source_sha256="0"*64)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        audit(directory, tmp_path/"rejected.json")
