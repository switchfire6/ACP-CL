"""Factor isolation, matched noise controls, and recovery for the next study."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_acquisition import memory_signature, signature, tensor_tree
from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.contextual.learner import ContextLearner
from acp_cl.training_state.study import FACTORIAL, final_law, fork, run_suite


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return dict(study="test", kind="diagnostic", seeds=[19], models=["conditional"],
        prefix_blocks=2, prefix_size=16, episode_size=16, batch_size=8, eval_size=64,
        probe_every=8, support_replicates=1, width=12, experts=4, context_width=8,
        decoder_width=12, interaction_features=True, lr=.002, memory_packets=3,
        updates_per_batch=1, evidence_strength=1., feedback_noise=.2, threads=1, workers=1)


@pytest.mark.parametrize("method", ("conditional", "recurrent"))
@pytest.mark.parametrize("arm", FACTORIAL)
def test_only_declared_state_changes_and_parent_is_isolated(method, arm, config):
    base = ContextLearner(method, 19, config)
    for i in range(5):
        base.train(AcquisitionWorld().experience(Law(i % 2), 8, i+10))
    before = signature(base)
    changed = fork(base, arm, 19, config)
    assert changed.diagnostics()["final_hash"] == base.diagnostics()["final_hash"]
    assert changed.history.fingerprint() == base.history.fingerprint()
    assert all(p.requires_grad for p in changed.model.parameters())
    if arm in ("optimizer_reset", "state_reset"):
        assert len(changed.optimizer.state) == 0
        assert changed.optimizer.param_groups[0]["lr"] == base.optimizer.param_groups[0]["lr"]
    else:
        assert tensor_tree(changed.optimizer.state_dict()) == tensor_tree(base.optimizer.state_dict())
    if arm in ("replay_reset", "state_reset"):
        assert not changed.memory.packets and changed.memory.seen == 0
        expected = ContextLearner(method, 19, config)
        assert memory_signature(changed.memory) == memory_signature(expected.memory)
    else:
        assert memory_signature(changed.memory) == memory_signature(base.memory)
    assert signature(base) == before
    changed.train(AcquisitionWorld().experience(Law(0, (0,)), 8, 51))
    assert signature(base) == before
    assert changed.cost["optimizer_steps"] == base.cost["optimizer_steps"]+1
    assert changed.cost["query_presentations"] == base.cost["query_presentations"]+16


@pytest.mark.parametrize("method", ("conditional", "recurrent"))
def test_optimizer_only_replays_exact_same_packets_as_continue(method, config):
    base = ContextLearner(method, 19, config)
    for i in range(7):
        base.train(AcquisitionWorld().experience(Law(i % 2), 8, i+100))
    a, b = (fork(base, arm, 19, config) for arm in ("continue", "optimizer_reset"))
    for i in range(6):
        data = AcquisitionWorld().experience(Law(1, (0,)), 8, i+300)
        a.train(data)
        b.train(data)
        assert memory_signature(a.memory) == memory_signature(b.memory)
    fresh = fork(base, "fresh", 19, config)
    assert fresh.diagnostics()["final_hash"] == fresh.initial_hash
    assert fresh.history.fingerprint() == base.history.fingerprint()


def test_final_clean_noise_pair_has_same_actions_images_and_physics():
    world = AcquisitionWorld()
    laws = [final_law(8001, branch, .2) for branch in ("clean", "noise")]
    a, b = (batch(world, law, 8001, "final", 3, 512) for law in laws)
    np.testing.assert_array_equal(a.observations, b.observations)
    np.testing.assert_array_equal(a.actions, b.actions)
    assert np.any(a.survival != b.survival)
    cases = [world.dataset(law, 512, 9) for law in laws]
    np.testing.assert_array_equal(*[world.counterfactuals(c) for c in cases])


def test_resume_preserves_factorial_forks_and_all_final_branches(config, tmp_path, monkeypatch):
    from acp_cl.acquisition import study as shared
    complete, interrupted = tmp_path/"complete", tmp_path/"interrupted"
    run_suite(config, complete)
    original = shared.atomic_checkpoint

    def crash(path, payload):
        original(path, payload)
        r = payload["record"]
        if r.get("branch") == "noise" and r.get("arm") == "replay_reset" and r.get("batches_done") == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(shared, "atomic_checkpoint", crash)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_suite(config, interrupted)
    monkeypatch.setattr(shared, "atomic_checkpoint", original)
    run_suite(config, interrupted, resume=True)
    paths = list(complete.glob("*/*/result.json"))
    assert len(paths) == 31
    for p in paths:
        a, b = (json.loads(x.read_text()) for x in (p, interrupted/p.relative_to(complete)))
        for key in ("curve", "before", "after", "valid_before", "valid_after", "batch_sha256"):
            assert a[key] == b[key]
        assert a["diagnostics"]["final_hash"] == b["diagnostics"]["final_hash"]
    with pytest.raises(ValueError, match="identical"):
        run_suite(dict(config, lr=.003), interrupted, resume=True)
    c = copy.deepcopy(config)
    c["kind"] = "development"
    run_suite(c, tmp_path/"development")
    completion = json.loads((tmp_path/"development/completion.json").read_text())
    assert completion["episodes"] == 3 and completion["prefix_fits"] == 0
