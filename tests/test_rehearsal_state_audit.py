"""Independent named-Adam transplant, hybrid reproduction and panel audit checks."""

import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_rehearsal_state import (
    CELLS, active_hash, audit_job, file_hash, recorded_active_hash,
    reproduce_factor, transplant_independent,
)
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.conditional.learner import Packet
from acp_cl.predictive_value.mechanism import _value_hash, clone_exact, make_shadow, predict_all, state_signature
from acp_cl.predictive_value.study import read_json
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import new_learner
from acp_cl.rehearsal_state import study
from acp_cl.rehearsal_state.design import counts, schedule


@pytest.fixture(params=("conditional", "recurrent"))
def donors(request):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = read_json("configs/predictive_value_smoke.json")
    config.update(cpu_affinity=0, workers=1)
    learner = new_learner(request.param, 15991, config)
    world = AcquisitionWorld()
    packets = []
    for index in range(3):
        data = batch(world, Law(index % 2), 15991, "factor-audit-donors", index, config["batch_size"])
        packet = Packet(learner.history, data)
        learner.train(data)
        packets.append(packet)
        if index == 0:
            old = clone_exact(learner)
    return old, learner, packets[0], packets[-1], packets[1], config


def test_named_transplant_copies_complete_adam_state_and_preserves_inactive_state(donors):
    old, new, old_anchor, new_anchor, replay, config = donors
    old_signature, new_signature = state_signature(old), state_signature(new)
    for cell in CELLS:
        result = transplant_independent(old, new, cell)
        weight_donor, moment_donor = (old, new)[int(cell[0])], (old, new)[int(cell[1])]
        assert all(torch.equal(result.model.state_dict()[name], value)
                   for name, value in weight_donor.model.state_dict().items())
        destination = dict(result.model.named_parameters())
        source = dict(moment_donor.model.named_parameters())
        assert len(result.optimizer.state) == len(source)
        for name in destination:
            actual = result.optimizer.state[destination[name]]
            expected = moment_donor.optimizer.state[source[name]]
            assert actual.keys() == expected.keys()
            for key in expected:
                assert torch.equal(actual[key], expected[key])
                assert actual[key].data_ptr() != expected[key].data_ptr()
        assert result.cost == old.cost and memory_state(result.memory) == memory_state(old.memory)
        assert result.history.fingerprint() == old.history.fingerprint()
        registered = [p for group in result.optimizer.param_groups for p in group["params"]]
        assert all(a is b for a, b in zip(registered, result.model.parameters()))
    assert state_signature(old) == old_signature and state_signature(new) == new_signature


def test_allnew_reconstruction_matches_ordinary_new_rehearsal_active_state(donors):
    old, new, old_anchor, new_anchor, replay, config = donors
    expected, _ = make_shadow(new, new_anchor, replay, config["rehearsal_updates"])
    actual = reproduce_factor(old, new, old_anchor, new_anchor, replay, "111", config["rehearsal_updates"])
    assert active_hash(actual) == active_hash(expected)
    assert actual.cost == old.cost and expected.cost == new.cost
    assert state_signature(actual) != state_signature(expected)
    assert all(float(state["step"]) == new.cost["optimizer_steps"]+config["rehearsal_updates"]
               for state in actual.optimizer.state.values())


def test_zero_update_predictions_do_not_depend_on_optimizer_or_anchor(donors):
    old, new, _, _, replay, _ = donors
    for weight in ("0", "1"):
        values = [predict_all(transplant_independent(old, new, weight+suffix),
                              replay.query.observations, new.history)
                  for suffix in ("00", "01", "10", "11")]
        assert all((value == values[0]).all() for value in values[1:])


def test_independent_transplant_rejects_incompatible_snapshot(donors):
    old, new, *_ = donors
    bad = copy.deepcopy(new)
    bad.settings["lr"] *= 2
    with pytest.raises(ValueError, match="incompatible"):
        transplant_independent(old, bad, "010")


@pytest.fixture(scope="module")
def completed_jobs(tmp_path_factory):
    config = read_json("configs/rehearsal_state_smoke.json")
    config.update(prefix_size=16, acquisition_size=32, score_size=16,
                  validation_size=16, inter_assessment_size=16,
                  checkpoint_every=8, workers=1, cpu_affinity=0)
    root = tmp_path_factory.mktemp("rehearsal_state_audit")
    identity = {"engineering":"unlocked-factor-audit-fixture"}
    for model in config["models"]:
        study.run_job(config, identity, config["seeds"][0], model, root)
    return root, config, identity


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_complete_audit_recovers_all_forks_and_forecasts_without_mutating_files(completed_jobs, model):
    root, config, identity = completed_jobs
    seed = config["seeds"][0]
    directory = root/f"{model}_{seed}"
    before = {path.relative_to(directory).as_posix():file_hash(path)
              for path in directory.rglob("*") if path.is_file()}
    result = audit_job(directory, config, identity, seed, model)
    declared = counts({**config, "models":[model]})
    assert result["diagnostic_forks_reproduced"] == declared["diagnostic_forks"]
    assert result["diagnostic_prediction_calls_recomputed"] == declared["diagnostic_prediction_forwards"]
    assert result["loss_instrumentation_forwards_recomputed"] == declared["extra_loss_forwards"]
    assert result["zero_update_copies"] == declared["zero_update_copies"]
    assert result["before_checkpoints"] == result["final_checkpoints"] == len(schedule(seed, config))
    assert result["diagnostic_boundary_checkpoints"] == 2
    assert result["ordinary_packets"]*config["batch_size"] == declared["ordinary_arrivals_per_trajectory"]
    assert result["original_packet_errors_recomputed"] == result["ordinary_packets"]
    for check in result["assessment_checks"]:
        assert check["all_new_active_endpoints_verified"] == config["candidate_count"]+1
        assert check["paired_causal_branches_verified"] and check["unchanged_donors_verified"]
        assert check["score_only_selection_verified"]
        assert check["result_sha256"] == before[f"diagnostic_{check['index']}/result.json"]
    after = {path.relative_to(directory).as_posix():file_hash(path)
             for path in directory.rglob("*") if path.is_file()}
    assert after == before


def copy_job(completed_jobs, tmp_path):
    root, config, identity = completed_jobs
    seed, model = config["seeds"][0], "conditional"
    directory = tmp_path/f"{model}_{seed}"
    shutil.copytree(root/directory.name, directory)
    return directory, config, identity, seed, model


@pytest.mark.parametrize("fault", (
    "factor_moment_resealed", "factor_work_anchor", "zero_weight_resealed", "missing_gradient",
    "paired_outcomes", "paired_support", "diagnostic_prediction", "ordinary_return_feedback",
    "prequential_steps",
))
def test_tensor_causal_and_arithmetic_audit_rejects_corruption(completed_jobs, tmp_path, fault):
    directory, config, identity, seed, model = copy_job(completed_jobs, tmp_path)
    folder = directory/"diagnostic_0"
    if fault in ("factor_moment_resealed", "factor_work_anchor", "missing_gradient"):
        path = folder/"cell_010.pt"
        payload = study.load(path, identity)
        if fault == "factor_moment_resealed":
            learner = payload["learners"][0]
            next(iter(learner.optimizer.state.values()))["exp_avg"].add_(.01)
            payload["signatures"][0] = state_signature(learner)
            payload["active_signatures"][0] = recorded_active_hash(learner)
            payload["work"][0].update(shadow_state_sha256=state_signature(learner),
                final_active_sha256=recorded_active_hash(learner),
                final_optimizer_sha256=_value_hash(learner.optimizer.state_dict()))
        elif fault == "factor_work_anchor":
            payload["work"][0]["anchor_packet_sha256"] = payload["anchor_new_sha256"]
        else:
            payload["saved_gradients"][0].pop(next(iter(payload["saved_gradients"][0])))
            torch.save(payload, path)
        if fault != "missing_gradient":
            study.save(path, payload)
    elif fault == "zero_weight_resealed":
        path = folder/"zero.pt"
        payload = study.load(path, identity)
        with torch.no_grad():
            next(payload["learners"][0].model.parameters()).add_(.01)
        payload["signatures"][0] = state_signature(payload["learners"][0])
        study.save(path, payload)
    elif fault == "paired_outcomes":
        path = folder/"data.npz"
        arrays = study.read_arrays(path)
        arrays["outcomes"][1, 0, 0, 0] ^= 1
        study.save_arrays(path, **arrays)
    elif fault == "paired_support":
        path = folder/"stream.json"
        stream = read_json(path)
        stream["rows"][1]["return"]["support_sha256"] = stream["rows"][1]["current"]["support_sha256"]
        # These clean laws can coincidentally share feedback; force a distinct bad support if needed.
        if stream == read_json(path):
            stream["rows"][1]["return"]["support_sha256"] = "0"*64
        path.write_text(json.dumps(stream), encoding="utf-8")
    elif fault == "diagnostic_prediction":
        path = folder/"predictions_000.npz"
        arrays = study.read_arrays(path)
        arrays["probabilities"][0, 0, 0, 0, 0, 0] += np.float32(.001)
        study.save_arrays(path, **arrays)
        seal_path = folder/"predictions_000.json"
        seal = read_json(seal_path)
        seal["predictions_sha256"] = file_hash(path)
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
    else:
        phase = "validation_0" if fault == "ordinary_return_feedback" else "prefix_0"
        path = directory/phase/"checkpoint.pt"
        payload = study.load(path, identity)
        if fault == "ordinary_return_feedback":
            payload["learner"].history.survival[0, 0] ^= 1
            payload["record"]["final_signature"] = state_signature(payload["learner"])
        else:
            payload["record"]["packets"][0]["original_steps"] += 1
        study.save(path, payload)
        (directory/phase/"result.json").write_text(json.dumps(payload["record"]), encoding="utf-8")
    with pytest.raises((ValueError, AssertionError)):
        audit_job(directory, config, identity, seed, model)


def test_choice_is_rejected_before_any_heldout_boundary_is_opened(completed_jobs, tmp_path):
    directory, config, identity, seed, model = copy_job(completed_jobs, tmp_path)
    path = directory/"assessment_0"/"choice.json"
    choice = read_json(path)
    candidates = study.load(directory/"assessment_0"/"candidates.pt", identity)
    choice["choice"]["value_id"] = next(identifier for identifier in candidates["ids"][:-1]
                                        if identifier != choice["choice"]["value_id"])
    path.write_text(json.dumps(choice), encoding="utf-8")
    (directory/"diagnostic_0"/"before.pt").write_bytes(b"future diagnostic state must not be opened yet")
    with pytest.raises(ValueError, match="sealed choice does not follow score-window evidence"):
        audit_job(directory, config, identity, seed, model)
