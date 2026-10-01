"""Rehearsal/prediction reproduction and rejection of coherent record corruption."""

import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_predictive_value import audit_job, checked_load, file_hash, independent_choice
from acp_cl.persistence.learner import state_hash
from acp_cl.predictive_value import study
from acp_cl.predictive_value.design import schedule
from acp_cl.predictive_value.mechanism import _value_hash, state_signature


@pytest.fixture(scope="module")
def completed_jobs(tmp_path_factory):
    config = study.read_json("configs/predictive_value_smoke.json")
    config.update(prefix_size=16, acquisition_size=32, recovery_size=8,
                  score_size=16, validation_size=16, gaps=[8, 16],
                  checkpoint_every=8, workers=1, cpu_affinity=0)
    root = tmp_path_factory.mktemp("predictive_value_audit")
    identity = {"engineering": "unlocked-audit-fixture"}
    for model in config["models"]:
        study.run_job(config, identity, config["seeds"][0], model, root)
    return root, config, identity


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_full_local_audit_recreates_forks_and_predictions_without_modifying_files(completed_jobs, model):
    root, config, identity = completed_jobs
    seed = config["seeds"][0]
    directory = root/f"{model}_{seed}"
    before = {path.relative_to(directory).as_posix(): file_hash(path)
              for path in directory.rglob("*") if path.is_file()}
    result = audit_job(directory, config, identity, seed, model)
    phases = schedule(seed, config)
    packets = sum(phase["size"]//config["batch_size"] for phase in phases)
    assert result["before_checkpoints"] == result["final_checkpoints"] == len(phases)
    assert result["assessments"] == 2
    assert result["ordinary_packets"] == result["original_packet_errors_recomputed"] == packets
    assert result["original_predictions_recomputed"] == len(phases)
    assert result["diagnostic_forks_reproduced"] == 6*(config["candidate_count"]+1)
    assert result["diagnostic_prediction_calls_recomputed"] == 2*(config["candidate_count"]+1)*(
        config["score_size"]+4*config["validation_size"])//config["batch_size"]
    assert {path.relative_to(directory).as_posix(): file_hash(path)
            for path in directory.rglob("*") if path.is_file()} == before
    for phase in result["phases"]:
        assert phase["result_sha256"] == before[phase["phase"]+"/result.json"]
        assert phase["before_sha256"] == before[phase["phase"]+"/before.pt"]
        assert phase["checkpoint_sha256"] == before[phase["phase"]+"/checkpoint.pt"]


def copy_job(completed_jobs, tmp_path):
    root, config, identity = completed_jobs
    seed, model = config["seeds"][0], "conditional"
    target = tmp_path/f"{model}_{seed}"
    shutil.copytree(root/target.name, target)
    return target, config, identity, seed, model


@pytest.mark.parametrize("fault", (
    "before_optimizer", "after_memory_rng", "missing_gradient", "nonfinite_gradient",
    "candidate_target", "candidate_hidden_label", "candidate_metadata",
    "shadow_moment_resealed", "prequential_actions", "prequential_support",
    "prequential_steps", "validation_prediction", "validation_truth",
))
def test_checkpoint_audit_rejects_corruption_even_when_json_matches(completed_jobs, tmp_path, fault):
    directory, config, identity, seed, model = copy_job(completed_jobs, tmp_path)
    phase = "score_0"
    if fault == "before_optimizer":
        path = directory/phase/"before.pt"
    elif fault.startswith("candidate_"):
        path = directory/"assessment_0"/"candidates.pt"
    elif fault == "shadow_moment_resealed":
        path = directory/phase/"shadows.pt"
    elif fault.startswith("validation_"):
        phase = "near_0"
        path = directory/phase/"checkpoint.pt"
    else:
        phase = "prefix_0"
        path = directory/phase/"checkpoint.pt"
    saved = study.load(path, identity)
    if fault == "before_optimizer":
        next(iter(saved["learner"].optimizer.state.values()))["exp_avg"].add_(.01)
    elif fault == "after_memory_rng":
        saved["learner"].memory.sample()
    elif fault == "missing_gradient":
        saved["saved_gradients"][0].pop(next(iter(saved["saved_gradients"][0])))
        torch.save(saved, path)
    elif fault == "nonfinite_gradient":
        next(saved["learner"].model.parameters()).grad.fill_(float("nan"))
    elif fault == "candidate_target":
        saved["packets"][0].query.survival[0, 0] ^= 1
    elif fault == "candidate_hidden_label":
        saved["packets"][0].oracle_modes = np.zeros(config["batch_size"], dtype=np.int64)
    elif fault == "candidate_metadata":
        saved["metadata"][0]["original_brier"] += .01
    elif fault == "shadow_moment_resealed":
        learner = saved["learners"][0]
        next(iter(learner.optimizer.state.values()))["exp_avg"].add_(.01)
        saved["signatures"][0] = state_signature(learner)
        saved["work"][0].update(shadow_state_sha256=state_signature(learner),
            final_optimizer_sha256=_value_hash(learner.optimizer.state_dict()),
            final_model_sha256=state_hash(learner.model.state_dict()))
    elif fault == "prequential_actions":
        values = saved["record"]["packets"][0]["actions"]
        values[0] = (values[0]+1) % 5
    elif fault == "prequential_support":
        saved["record"]["packets"][0]["support_sha256"] = "0"*64
    elif fault == "prequential_steps":
        saved["record"]["packets"][0]["original_steps"] += 1
    elif fault == "validation_prediction":
        saved["record"]["evaluations"][0]["reapplied"][0]["probabilities"][0][0] += .01
    else:
        values = saved["record"]["evaluations"][0]["truth"]
        values[0][0][0] = 1-values[0][0][0]
    if fault != "missing_gradient":
        study.save(path, saved)
    if path.name == "checkpoint.pt":
        (directory/phase/"result.json").write_text(json.dumps(saved["record"]), encoding="utf-8")
    with pytest.raises((ValueError, AssertionError)):
        audit_job(directory, config, identity, seed, model)


def test_wrong_selection_is_rejected_before_validation_checkpoint_is_opened(completed_jobs, tmp_path):
    directory, config, identity, seed, model = copy_job(completed_jobs, tmp_path)
    path = directory/"assessment_0"/"choice.json"
    choice = study.read_json(path)
    bundle = study.load(directory/"assessment_0"/"candidates.pt", identity)
    other = next(packet_id for packet_id in bundle["ids"][:-1]
                 if packet_id != choice["choice"]["value_id"])
    choice["choice"]["value_id"] = other
    path.write_text(json.dumps(choice), encoding="utf-8")
    # If the auditor opens future state before checking the choice, unpickling fails.
    (directory/"near_0"/"checkpoint.pt").write_bytes(b"validation must not be opened yet")
    with pytest.raises(ValueError, match="sealed choice does not follow score-window"):
        audit_job(directory, config, identity, seed, model)


def test_original_accuracy_and_value_controls_ignore_future_labels_and_latent_metadata():
    candidates = [dict(id=9, original_brier=.04), dict(id=3, original_brier=.2)]
    losses = [.12, .08, .1]
    expected = independent_choice(candidates, losses)
    assert expected["value_id"] == 3 and expected["accuracy_id"] == 9
    contaminated = copy.deepcopy(candidates)
    for candidate in contaminated:
        candidate.update(validation_loss=-candidate["id"], origin_law={"mode": candidate["id"]})
    assert independent_choice(contaminated, losses) == expected
    assert independent_choice(candidates, [.1, .1, .2])["value_id"] == 3


def test_gradient_loader_rejects_identity_mismatch(completed_jobs):
    root, config, _ = completed_jobs
    path = root/f"conditional_{config['seeds'][0]}"/"score_0"/"before.pt"
    with pytest.raises(ValueError, match="checkpoint identity"):
        checked_load(path, {"engineering": "wrong"})
