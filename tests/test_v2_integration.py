"""Mechanism and information-isolation checks across v2 components."""

import copy
import json

import pytest
import torch

from acp_cl import experiment
from acp_cl.data import build_stream
from test_experiment import tiny_config, lightweight_environment, assert_tree_equal, saved_learner_state

# Imported fixtures deliberately reuse the exact paired orchestration fixture.
__all__ = ["tiny_config", "lightweight_environment"]


def test_fixed_acquisition_horizon_and_short_stream_rejection():
    assert experiment.acquisition_auc([(0, 0), (100, 1), (200, 1)], {"early_examples": 100}) == 0.5
    with pytest.raises(ValueError, match="horizon"):
        experiment.acquisition_auc([(0, 0), (100, 1)], {"early_examples": 200})


def test_yoked_reset_counts_and_gain_budget_match_exactly(tiny_config, tmp_path):
    tiny_config["plasticity"]["newborn_steps"] = 2
    tiny_config["steps_per_experience"] = 8
    methods = ["er_recycle_yoked_gain", "acp_v2", "er_recycle_yoked"]
    results = experiment.run_suite(tiny_config, output=tmp_path, seeds=[19], methods=methods, device_name="cpu")
    by_method = {r["method"]: r for r in results}
    source = json.loads((tmp_path / "acp_v2_seed19" / "allocation.json").read_text())["trace"]
    assert by_method["acp_v2"]["diagnostics"]["total_recycled"] > 0
    for method in methods:
        trace = json.loads((tmp_path / f"{method}_seed19" / "allocation.json").read_text())["trace"]
        assert [r["recycled_by_population"] for r in trace] == [r["recycled_by_population"] for r in source]
        assert by_method[method]["cost"]["current_examples"] == by_method["acp_v2"]["cost"]["current_examples"]
    assert by_method["er_recycle_yoked_gain"]["allocation_summary"]["mean_feature_gain"] == pytest.approx(
        by_method["acp_v2"]["allocation_summary"]["mean_feature_gain"], abs=1e-6)
    assert "diagnostic" in by_method["er_recycle_yoked"]["information_access"]
    assert by_method["er_recycle_yoked"]["allocation_source_sha256"]


def test_yoked_missing_source_rejected(tiny_config, tmp_path):
    stream = build_stream(tiny_config["data"], 19)
    with pytest.raises(ValueError, match="before yoked"):
        experiment.run_one(tiny_config, stream, "er_recycle_yoked", 19, torch.device("cpu"), tmp_path)


@pytest.mark.parametrize("method", ["acp_v2", "acp_v2_no_newborn", "acp_v2_oracle", "acp_v2_learned_sensor"])
def test_v2_checkpoint_continuation_exact(tiny_config, method):
    tiny_config["plasticity"]["newborn_steps"] = 2
    stream = build_stream(tiny_config["data"], 19)
    first = experiment.new_learner(tiny_config, stream, method, 19, torch.device("cpu"))
    second = experiment.new_learner(tiny_config, stream, method, 19, torch.device("cpu"))
    sequence = list(experiment.batches(stream.experiences[0], tiny_config, 19))
    first.train_batch(*sequence[0])
    if first.oracle:
        first.notify_oracle_boundary()
    second.load_state_dict(saved_learner_state(first))
    for x, y in sequence[1:]:
        assert_tree_equal(first.train_batch(x, y), second.train_batch(x, y))
    assert_tree_equal(saved_learner_state(first), saved_learner_state(second))


def test_only_oracle_receives_evaluator_boundary(tiny_config, tmp_path, monkeypatch):
    from acp_cl.learner import Learner
    calls = []
    original = Learner.notify_oracle_boundary

    def record(self):
        calls.append(self.method)
        return original(self)

    monkeypatch.setattr(Learner, "notify_oracle_boundary", record)
    experiment.run_suite(tiny_config, output=tmp_path, seeds=[19], methods=["er", "acp_v2", "acp_v2_oracle"], device_name="cpu")
    assert calls == ["acp_v2_oracle"]


def test_sparse_retention_final_row_complete_and_evaluation_state_independent(tiny_config, tmp_path):
    tiny_config["data"]["n_experiences"] = 4
    tiny_config["retention_eval_every_experiences"] = 3
    stream = build_stream(tiny_config["data"], 19)
    result = experiment.run_one(tiny_config, stream, "acp_v2", 19, torch.device("cpu"), tmp_path)
    assert result["accuracy_matrix"][1][0] is None
    assert all(value is not None for value in result["accuracy_matrix"][-1])
    learner = experiment.new_learner(tiny_config, stream, "acp_v2", 19, torch.device("cpu"))
    learner.train_batch(*next(experiment.batches(stream.experiences[0], tiny_config, 19)))
    before = copy.deepcopy(learner.state_dict())
    learner.accuracy(stream.experiences[0].validation)
    after = copy.deepcopy(learner.state_dict())
    after["cost"]["evaluation_examples"] = before["cost"]["evaluation_examples"]
    assert_tree_equal(before, after)
