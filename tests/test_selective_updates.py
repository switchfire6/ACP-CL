"""Actual-displacement geometry, original-reference equivalence, and bounded state."""

import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_training_state import signature, tensor_tree
from acp_cl.acquisition.world import AcquisitionWorld, Law
from acp_cl.persistence.learner import state_hash
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import new_learner
from acp_cl.selective_updates.learner import (
    ARMS, ARM_SETTINGS, COUNTERS, SUMMARY_NAMES, SelectiveLearner, choose_displacement,
    fork, new_stats, project_displacement,
)


@pytest.fixture
def settings():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return dict(width=8, experts=4, context_width=5, decoder_width=8,
        interaction_features=True, lr=.002, memory_packets=4, batch_size=8,
        evidence_strength=1., updates_per_batch=3)


def experience(index, mode=None):
    return AcquisitionWorld().experience(
        Law(index % 2 if mode is None else mode, (0, 1) if index >= 3 else ()), 8, 5100+index)


def trained(model, settings):
    learner = new_learner(model, 51, settings)
    for index in range(3):
        learner.train(experience(index))
    return learner


def cost_without_clock(learner):
    return learner.cost | {"training_seconds": 0.}


def selective_without_clock(learner):
    result = learner.diagnostics()
    result["cost"]["training_seconds"] = 0.
    return result


def test_fixed_arms_keep_weighting_and_protection_factors_separate():
    assert ARMS == ("reference", "current", "protected", "combined", "shrink")
    assert ARM_SETTINGS == dict(reference=(.5, "none"), current=(.75, "none"),
        protected=(.5, "project"), combined=(.75, "project"), shrink=(.75, "shrink"))


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
@pytest.mark.parametrize("prefix", (False, True))
def test_reference_is_bitwise_original_through_replay_replacement(model, prefix, settings):
    ordinary = trained(model, settings) if prefix else new_learner(model, 51, settings)
    selective = fork(ordinary, "reference")
    for index in range(3, 13):
        packet = experience(index)
        ordinary.train(packet)
        selective.train(packet)
        assert signature(ordinary) == signature(selective)
        assert tensor_tree(ordinary.optimizer.state_dict()) == tensor_tree(selective.optimizer.state_dict())
        assert memory_state(ordinary.memory) == memory_state(selective.memory)
        assert cost_without_clock(ordinary) == cost_without_clock(selective)
        assert [module.training for module in ordinary.model.modules()] == [
            module.training for module in selective.model.modules()]
        assert all(torch.equal(a.grad, b.grad) for a, b in zip(
            ordinary.model.parameters(), selective.model.parameters()))
    assert selective.selective_stats["packets"] == 10
    assert selective.selective_stats["projection_applied_steps"] == 0
    assert selective.selective_stats["rewrite_steps"] == 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("d,h,expected,eligible", [
    ([3., 4.], [1., 0.], [0., 4.], True),
    ([-3., 4.], [1., 0.], [-3., 4.], False),
    ([0., 4.], [1., 0.], [0., 4.], False),
    ([3., 4.], [0., 0.], [3., 4.], False),
    ([0., 0.], [1., 3.], [0., 0.], False),
    ([2., 4.], [1., 2.], [0., 0.], True),
])
def test_projection_has_correct_sign_zero_handling_and_no_input_mutation(dtype, d, h, expected, eligible):
    displacement, reference = torch.tensor(d, dtype=dtype), torch.tensor(h, dtype=dtype)
    old_d, old_h = displacement.clone(), reference.clone()
    actual, geometry = project_displacement(displacement, reference)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    assert geometry["projection_eligible"] is eligible
    assert geometry["hd_projected"] <= 0
    assert torch.equal(displacement, old_d) and torch.equal(reference, old_h)


def test_projection_is_scale_invariant_in_reference_and_reports_double_residual():
    displacement = torch.tensor([.17, -.13, .21], dtype=torch.float64)
    reference = torch.tensor([.8, -.5, .6], dtype=torch.float64)
    result, geometry = project_displacement(displacement, reference)
    tiny, tiny_geometry = project_displacement(displacement, reference*1e-100)
    huge, _ = project_displacement(displacement, reference*1e100)
    torch.testing.assert_close(tiny, result, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(huge, result, rtol=1e-14, atol=1e-15)
    assert geometry["projection_eligible"] and tiny_geometry["projection_eligible"]
    assert abs(geometry["hd_projected"]) <= 1e-14*float(reference.norm()*displacement.norm())
    assert geometry["retained_norm_ratio"] <= 1


def test_shrink_retains_own_direction_and_matches_own_projected_norm():
    displacement, reference = torch.tensor([3., 4.]), torch.tensor([1., 0.])
    projected, geometry = choose_displacement(displacement, reference, "project")
    shrink, shrink_geometry = choose_displacement(displacement, reference, "shrink")
    unchanged, none_geometry = choose_displacement(displacement, reference, "none")
    torch.testing.assert_close(shrink, torch.tensor([2.4, 3.2], dtype=torch.float64))
    assert float(shrink.norm()) == pytest.approx(float(projected.norm()))
    assert geometry == shrink_geometry == none_geometry
    assert torch.equal(unchanged, displacement.double())
    assert float(torch.dot(reference.double(), shrink)) > 0  # It does not enforce the halfspace.
    aligned, _ = choose_displacement(-displacement, reference, "shrink")
    assert torch.equal(aligned, -displacement.double())
    zero, _ = choose_displacement(torch.zeros(2), reference, "shrink")
    assert torch.equal(zero, torch.zeros(2, dtype=torch.float64))


def test_first_order_constraint_does_not_guarantee_finite_step_loss_decrease():
    # At theta=(0,0), L(theta)=theta[0]+theta[1]**2 has gradient (1,0).
    # Removing the adverse linear component leaves positive quadratic loss.
    projected, geometry = project_displacement(torch.tensor([1., 2.]), torch.tensor([1., 0.]))
    assert geometry["hd_projected"] == 0
    assert float(projected[0]+projected[1]**2) > 0


@pytest.mark.parametrize("d,h", [
    (torch.zeros(0), torch.zeros(0)),
    (torch.zeros(2, 2), torch.zeros(2, 2)),
    (torch.zeros(2), torch.zeros(3)),
    (torch.tensor([float("nan")]), torch.ones(1)),
    (torch.zeros(1), torch.tensor([float("inf")])),
])
def test_invalid_geometry_is_rejected(d, h):
    with pytest.raises(ValueError):
        project_displacement(d, h)


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_projection_and_shrink_keep_their_ordinary_single_step_adam_moments(model, settings):
    settings = settings | dict(updates_per_batch=1)
    original = trained(model, settings)
    learners = {arm: fork(original, arm) for arm in ARMS}
    for learner in learners.values():
        learner.train(experience(30))
    for ordinary, candidate in (("reference", "protected"), ("current", "combined"), ("current", "shrink")):
        assert tensor_tree(learners[ordinary].optimizer.state_dict()) == tensor_tree(
            learners[candidate].optimizer.state_dict())
        assert all(torch.equal(a.grad, b.grad) for a, b in zip(
            learners[ordinary].model.parameters(), learners[candidate].model.parameters()))
    assert len({json.dumps(memory_state(x.memory), sort_keys=True) for x in learners.values()}) == 1


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
@pytest.mark.parametrize("arm", ARMS)
def test_all_arms_match_work_and_keep_bounded_diagnostics(model, arm, settings, monkeypatch):
    learner = fork(trained(model, settings), arm)
    expected_fields = set(learner.selective_stats)
    before_cost = dict(learner.cost)
    probability_calls = []
    original_probabilities = learner.probabilities

    def counted(observations, support, oracle_modes=None, override=None):
        assert oracle_modes is None
        probability_calls.append((len(observations), torch.is_grad_enabled()))
        return original_probabilities(observations, support, oracle_modes, override)

    monkeypatch.setattr(learner, "probabilities", counted)
    for index in range(3, 7):
        learner.train(experience(index))
    stats = learner.selective_stats
    steps, packets, size = 4*settings["updates_per_batch"], 4, settings["batch_size"]
    assert set(stats) == expected_fields
    assert set(stats["summaries"]) == set(SUMMARY_NAMES)
    assert stats["steps"] == steps and stats["packets"] == packets
    assert stats["training_probability_calls"] == 2*steps
    assert stats["reference_gradient_calls"] == stats["objective_backward_calls"] == steps
    assert stats["reference_backward_query_presentations"] == steps*size
    assert stats["reference_backward_support_presentations"] == steps*size
    assert stats["diagnostic_probability_calls"] == packets
    assert stats["diagnostic_query_presentations"] == packets*size
    assert stats["diagnostic_support_presentations"] == packets*size
    assert len(probability_calls) == 2*steps+packets
    assert sum(enabled for _, enabled in probability_calls) == 2*steps
    assert learner.cost["query_presentations"]-before_cost["query_presentations"] == 2*steps*size
    assert learner.cost["support_presentations"]-before_cost["support_presentations"] == 2*steps*size
    assert learner.cost["optimizer_steps"]-before_cost["optimizer_steps"] == steps
    assert 0 <= stats["projection_eligible_steps"] <= steps
    assert stats["projection_applied_steps"] <= stats["rewrite_steps"]
    assert stats["shrink_applied_steps"] <= stats["rewrite_steps"]
    if ARM_SETTINGS[arm][1] == "none":
        assert stats["rewrite_steps"] == stats["projection_applied_steps"] == stats["shrink_applied_steps"] == 0
    else:
        assert stats["rewrite_steps"] == stats["projection_eligible_steps"]
    assert 0 <= stats["finite_step_increases"] <= packets
    for name, aggregate in stats["summaries"].items():
        expected_count = packets if name.startswith("finite_replay_") else steps
        assert aggregate["count"] == expected_count
        assert aggregate["min"] <= aggregate["sum"]/expected_count <= aggregate["max"]+1e-14
        assert all(np.isfinite(aggregate[k]) for k in ("sum", "min", "max"))
    report = learner.diagnostics()["selective_updates"]
    assert report["temporary_parameter_gradient_buffer_bytes_estimate"] > 0
    assert "not measured peak RAM" in report["temporary_buffer_estimate_scope"]
    report["stats"]["steps"] = -1
    assert learner.selective_stats["steps"] == steps


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_fork_preserves_full_training_state_without_sharing_mutable_state(model, settings):
    original = trained(model, settings)
    before = signature(original)
    left, right = fork(original, "reference"), fork(original, "combined")
    assert signature(left) == signature(right) == before
    assert tensor_tree(left.optimizer.state_dict()) == tensor_tree(original.optimizer.state_dict())
    assert left.selective_stats == right.selective_stats == new_stats()
    assert left.settings == original.settings and left.settings is not original.settings
    assert left.memory.membership_rng is not original.memory.membership_rng
    assert left.memory.sampling_rng is not original.memory.sampling_rng
    assert not np.shares_memory(left.history.observations, original.history.observations)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(left.model.parameters(), original.model.parameters()))
    assert all(a.grad is not None and b.grad is not None and torch.equal(a.grad, b.grad)
               and a.grad.data_ptr() != b.grad.data_ptr()
               for a, b in zip(left.model.parameters(), original.model.parameters()))
    model_ids = {id(p) for p in left.model.parameters()}
    assert {id(p) for group in left.optimizer.param_groups for p in group["params"]} == model_ids
    left.train(experience(3))
    assert signature(original) == signature(right) == before
    assert right.selective_stats == new_stats()


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
@pytest.mark.parametrize("arm", ("reference", "combined", "shrink"))
def test_serialized_resume_matches_uninterrupted_learning(model, arm, settings):
    full = fork(trained(model, settings), arm)
    resumed = copy.deepcopy(full)
    for index in range(3, 7):
        full.train(experience(index))
    resumed.train(experience(3))
    saved = io.BytesIO()
    torch.save(resumed, saved)
    saved.seek(0)
    resumed = torch.load(saved, weights_only=False, map_location="cpu")
    for index in range(4, 7):
        resumed.train(experience(index))
    assert signature(full) == signature(resumed)
    assert memory_state(full.memory) == memory_state(resumed.memory)
    assert selective_without_clock(full) == selective_without_clock(resumed)


def test_latent_labels_and_unsupported_forks_are_rejected_before_learning(settings):
    learner = fork(trained("conditional", settings), "combined")
    before = signature(learner)
    with pytest.raises(ValueError, match="latent modes"):
        learner.train(experience(3), np.zeros(8, dtype=np.uint8))
    assert signature(learner) == before and learner.selective_stats == new_stats()
    with pytest.raises(ValueError, match="unknown selective"):
        fork(learner, "unknown")
    with pytest.raises(ValueError, match="contextual learner"):
        fork(object(), "reference")
    learner.freeze_features()
    with pytest.raises(ValueError, match="all model parameters"):
        fork(learner, "reference")


def test_empty_diagnostics_are_json_finite_and_direct_constructor_is_uniform(settings):
    learner = SelectiveLearner("conditional", 51, settings, arm="protected")
    stats = learner.diagnostics()["selective_updates"]["stats"]
    assert all(stats[name] == 0 for name in COUNTERS)
    assert all(part == dict(count=0, sum=0., min=None, max=None) for part in stats["summaries"].values())
    json.dumps(learner.diagnostics(), allow_nan=False)
    expected = new_learner("conditional", 51, settings)
    assert state_hash(learner.model.state_dict()) == state_hash(expected.model.state_dict())
    assert memory_state(learner.memory) == memory_state(expected.memory)
