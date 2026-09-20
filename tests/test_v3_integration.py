"""Causal v3 allocation through the shared experiment/replay interface."""

import copy
import json

import pytest
import torch

from acp_cl import experiment
from acp_cl.data import build_stream
from acp_cl.plasticity_v3 import V3_METHODS
from test_experiment import (
    assert_tree_equal, lightweight_environment, saved_learner_state, tiny_config,
)

__all__ = ["tiny_config", "lightweight_environment"]


@pytest.fixture
def config_v3(tiny_config):
    tiny_config["steps_per_experience"] = 12
    tiny_config["plasticity"].update(maturity_steps=2, recycle_cutoff=0.9)
    tiny_config["allocation_v3"] = {
        "feature_gain": 0.5, "warmup_steps": 2, "newborn_steps": 2,
        "newborn_peak": 1.0, "protection_steps": 4,
        "reset_interval": 3, "reset_count": 1, "max_update_norm": 0.1,
    }
    return tiny_config


@pytest.mark.parametrize("method", V3_METHODS)
def test_v3_exact_checkpoint_continuation_after_actual_recycling(config_v3, method):
    stream = build_stream(config_v3["data"], 731)
    first = experiment.new_learner(config_v3, stream, method, 731, torch.device("cpu"))
    second = experiment.new_learner(config_v3, stream, method, 731, torch.device("cpu"))
    sequence = list(experiment.batches(stream.experiences[0], config_v3, 731))
    for x, y in sequence[:6]:
        first.train_batch(x, y)
    if method != "er_v3":
        assert first.engine.total_recycled > 0
    second.load_state_dict(saved_learner_state(first))
    for x, y in sequence[6:]:
        assert_tree_equal(first.train_batch(x, y), second.train_batch(x, y))
    assert_tree_equal(saved_learner_state(first), saved_learner_state(second))


def test_v3_common_warmup_exact_and_evaluation_isolated(config_v3):
    stream = build_stream(config_v3["data"], 731)
    learners = [experiment.new_learner(config_v3, stream, m, 731, torch.device("cpu"))
                for m in V3_METHODS]
    sequence = list(experiment.batches(stream.experiences[0], config_v3, 731))
    for x, y in sequence[:config_v3["allocation_v3"]["warmup_steps"]]:
        for learner in learners:
            learner.train_batch(x, y)
    for learner in learners:
        assert learner.controller is None and learner.sensor is None
        assert not learner.oracle and not learner.yoked
        assert learner.engine.total_recycled == 0
        assert_tree_equal(learners[0].model.state_dict(), learner.model.state_dict())
        for name in learner.engine.state:
            assert_tree_equal(learners[0].engine.state[name]["momentum"],
                              learner.engine.state[name]["momentum"])
        assert_tree_equal(learners[0].buffer.state_dict(), learner.buffer.state_dict())
        assert_tree_equal(learners[0].augmentation_rng.get_state(),
                          learner.augmentation_rng.get_state())
        assert_tree_equal(learners[0].engine.generator.get_state(),
                          learner.engine.generator.get_state())
        with pytest.raises(ValueError, match="oracle"):
            learner.notify_oracle_boundary()
        before = copy.deepcopy(learner.state_dict())
        learner.accuracy(stream.experiences[0].validation)
        after = copy.deepcopy(learner.state_dict())
        after["cost"]["evaluation_examples"] = before["cost"]["evaluation_examples"]
        assert_tree_equal(before, after)


def test_v3_suite_counts_nominal_budgets_and_data_pairing(config_v3, tmp_path):
    results = experiment.run_suite(config_v3, output=tmp_path, seeds=[731],
                                   methods=list(V3_METHODS), device_name="cpu")
    resets, buffers = [], []
    warmup = config_v3["allocation_v3"]["warmup_steps"]
    for result in results:
        method = result["method"]
        location = tmp_path / f"{method}_seed731"
        trace = json.loads((location / "allocation.json").read_text())["trace"]
        state = torch.load(location / "checkpoint.pt", weights_only=False, map_location="cpu")
        buffers.append(state["learner"]["buffer"])
        if method != "er_v3":
            resets.append([row["recycled_by_population"] for row in trace])
        assert result["information_access"] == "training stream only"
        assert result["stream_fingerprints"] == results[0]["stream_fingerprints"]
        summary = result["allocation_summary"]
        assert summary["maximum_applied_update_norm"] <= 0.100001
        if method in ("recycle_v3", "newborn_matched_v3"):
            for row in trace[warmup:]:
                assert row["nominal_feature_gain"] == pytest.approx(0.5, abs=1e-6)
        assert all(row["effective_feature_gain"] <= row["nominal_feature_gain"] + 1e-6
                   for row in trace)
    assert all(schedule == resets[0] for schedule in resets)
    for buffer in buffers[1:]:
        assert_tree_equal(buffers[0], buffer)


def test_v3_changed_allocation_rejected_before_model_mutation(config_v3):
    stream = build_stream(config_v3["data"], 731)
    original = experiment.new_learner(config_v3, stream, "newborn_v3", 731, torch.device("cpu"))
    for x, y in experiment.batches(stream.experiences[0], config_v3, 731):
        original.train_batch(x, y)
    changed = copy.deepcopy(config_v3)
    changed["allocation_v3"]["feature_gain"] = 0.4
    target = experiment.new_learner(changed, stream, "newborn_v3", 731, torch.device("cpu"))
    before = saved_learner_state(target)
    with pytest.raises(ValueError, match="v3 allocation configuration"):
        target.load_state_dict(saved_learner_state(original))
    assert_tree_equal(before, saved_learner_state(target))
