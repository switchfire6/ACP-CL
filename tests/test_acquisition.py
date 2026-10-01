"""Causal data, genuine prediction changes, diagnostic isolation and recovery."""

import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from acp_cl.acquisition.study import Evaluator, fork, run_suite, validate_config
from acp_cl.acquisition.world import (AcquisitionWorld, Law, mask_valid, order,
                                     signal_images, stage_law)
from acp_cl.contextual.learner import ContextLearner
from acp_cl.persistence.learner import state_hash


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return dict(study="test", kind="diagnostic", seeds=[19], models=["conditional"],
        prefix_blocks=2, prefix_size=16, episode_size=16, batch_size=8, eval_size=64,
        probe_every=8, support_replicates=1, width=12, experts=4, context_width=8,
        decoder_width=12, interaction_features=True, lr=.002, memory_packets=3,
        updates_per_batch=1, evidence_strength=1., feedback_noise=.2, threads=1, workers=1)


def test_full_evaluation_factorial_and_last_frames_do_not_reveal_signals():
    world = AcquisitionWorld()
    cases = world.dataset(Law(0), 512, 91)
    assert len(np.unique(np.c_[cases.factors, cases.signals], axis=0)) == 64
    flipped = signal_images(cases.observations, 1-cases.signals)
    np.testing.assert_array_equal(flipped[:, -1], cases.observations[:, -1])
    assert not np.array_equal(flipped, cases.observations)


@pytest.mark.parametrize("cue", range(3))
def test_new_cue_is_irrelevant_before_activation_but_changes_physics_after(cue):
    world = AcquisitionWorld()
    cases = world.dataset(Law(0), 1024, 3)
    flipped = cases.signals.copy()
    flipped[:, cue] ^= 1
    np.testing.assert_array_equal(world.counterfactuals(cases),
                                  world.counterfactuals(replace(cases, signals=flipped)))
    active = replace(cases, law=Law(0, (cue,)))
    a, b = world.counterfactuals(active), world.counterfactuals(replace(active, signals=flipped))
    assert np.any(a != b)
    valid = mask_valid(active)
    np.testing.assert_array_equal(a[valid], world.counterfactuals(cases)[valid])
    outcome = world.simulate(active, np.arange(1024) % 5)
    assert outcome.conservation_error < 1e-10 and outcome.min_stock >= 0


def test_revision_invalidates_only_the_declared_old_relationship_subset():
    world = AcquisitionWorld()
    a = world.dataset(Law(1, (0, 1, 2)), 1024, 9)
    b = replace(a, law=replace(a.law, revised=True))
    pa, pb = world.counterfactuals(a), world.counterfactuals(b)
    valid = mask_valid(a, "revision")
    np.testing.assert_array_equal(pa[valid], pb[valid])
    assert np.any(pa[~valid] != pb[~valid])


def test_noise_changes_feedback_only_and_experience_has_no_hidden_fields():
    world, law = AcquisitionWorld(), Law(0, (0, 1, 2))
    a, b = (world.experience(c, 512, 9) for c in (law, replace(law, noise=.2)))
    assert set(vars(a)) == {"observations", "actions", "survival"}
    np.testing.assert_array_equal(a.observations, b.observations)
    np.testing.assert_array_equal(a.actions, b.actions)
    assert np.any(a.survival != b.survival)
    assert np.all(np.diff(b.survival.astype(int), axis=1) <= 0)


@pytest.mark.parametrize("method", ("conditional", "recurrent"))
def test_diagnostic_forks_match_context_and_reset_state_without_changing_parent(method, config):
    learner = ContextLearner(method, 19, config)
    world = AcquisitionWorld()
    for i in range(3):
        learner.train(world.experience(Law(i % 2), 8, i+4))
    before = copy.deepcopy(learner.model.state_dict())
    fresh, reset, frozen = (fork(learner, arm, 19, config) for arm in ("fresh", "state_reset", "frozen"))
    assert len(fresh.optimizer.state) == len(reset.optimizer.state) == 0
    assert len(fresh.memory.packets) == len(reset.memory.packets) == 0
    assert fresh.memory.sampling_rng.bit_generator.state == reset.memory.sampling_rng.bit_generator.state
    assert all(x.history.fingerprint() == learner.history.fingerprint() for x in (fresh, reset, frozen))
    assert state_hash(reset.model.state_dict()) == state_hash(before)
    assert state_hash(fresh.model.state_dict()) == fresh.initial_hash
    frozen.train(world.experience(Law(0, (0,)), 8, 17))
    for key, value in frozen.model.state_dict().items():
        if key.startswith(("frame.", "temporal.")):
            assert torch.equal(value, before[key])
    assert all(p.requires_grad for p in learner.model.frame.parameters())
    assert state_hash(learner.model.state_dict()) == state_hash(before)


def test_probes_do_not_change_learning_or_rng(config):
    a, b = (ContextLearner("recurrent", 19, config) for _ in range(2))
    world, evaluator = AcquisitionWorld(), Evaluator(19, config)
    for i in range(3):
        data = world.experience(Law(0, (0,)), 8, i+10)
        a.train(data)
        b.train(data)
        evaluator.probe(a, Law(0, (0,)), cue=0)
        evaluator.valid_panel(a, Law(0, (0,)))
    assert state_hash(a.model.state_dict()) == state_hash(b.model.state_dict())
    assert a.history.fingerprint() == b.history.fingerprint()
    assert a.memory.sampling_rng.bit_generator.state == b.memory.sampling_rng.bit_generator.state


def test_six_fresh_seeds_counterbalance_every_dependency_order():
    assert len({order(s) for s in range(7001, 7007)}) == 6
    for seed in range(7001, 7007):
        assert stage_law(seed, 3).active == (0, 1, 2)


def test_resume_of_partial_episode_is_exact(config, tmp_path, monkeypatch):
    from acp_cl.acquisition import study
    complete, interrupted = tmp_path/"complete", tmp_path/"interrupted"
    run_suite(config, complete)
    original = study.atomic_checkpoint

    def crash(path, payload):
        original(path, payload)
        r = payload["record"]
        if r.get("stage") == 2 and r.get("arm") == "state_reset" and r.get("batches_done") == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(study, "atomic_checkpoint", crash)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_suite(config, interrupted)
    monkeypatch.setattr(study, "atomic_checkpoint", original)
    run_suite(config, interrupted, resume=True)
    for p in complete.glob("*/*/result.json"):
        a, b = (json.loads(x.read_text()) for x in (p, interrupted/p.relative_to(complete)))
        for k in ("curve", "before", "after", "valid_before", "valid_after", "batch_sha256"):
            assert a[k] == b[k]
        assert a["diagnostics"]["final_hash"] == b["diagnostics"]["final_hash"]
    with pytest.raises(ValueError, match="identical"):
        run_suite(dict(config, lr=.003), interrupted, resume=True)


@pytest.mark.parametrize("field,value", (("eval_size", 32), ("episode_size", 17),
                                         ("prefix_blocks", 3), ("interaction_features", False)))
def test_invalid_configs(config, field, value):
    with pytest.raises(ValueError):
        validate_config(dict(config, **{field: value}))
