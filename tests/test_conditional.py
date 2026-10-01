"""Causal and scientific contracts for the conditional-reuse follow-up."""

import copy
import json

import numpy as np
import pytest
import torch

from acp_cl.conditional.learner import ConditionalLearner, Packet, PacketMemory, evidence_weights
from acp_cl.conditional.study import Evaluator, run_suite, validate_config
from acp_cl.conditional.world import Condition, ConditionalWorld, GRID, schedule, training_batch
from acp_cl.persistence.learner import state_hash


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return dict(study="test", seeds=[19], schedules=["recurring"], methods=["conditional"],
                block_size=16, batch_size=8, eval_size=8, probe_every=16, support_replicates=1,
                width=12, experts=4, lr=.002, memory_packets=3, updates_per_batch=1,
                evidence_strength=1., threads=1, workers=1)


@pytest.mark.parametrize("mode", (0, 1, -1))
@pytest.mark.parametrize("gated", (False, True))
def test_factorial_balance_physics_and_performed_action_only(mode, gated):
    world, condition = ConditionalWorld(), Condition(mode, gated)
    trials = world.dataset(condition, 64, 31)
    cells, counts = np.unique(trials.factors, axis=0, return_counts=True)
    np.testing.assert_array_equal(cells, np.asarray(GRID))
    np.testing.assert_array_equal(counts, np.full(8, 8))
    for action in range(5):
        outcomes = world.simulate(trials.cases, np.full(64, action))
        assert outcomes.conservation_error < 1e-10
        assert outcomes.min_stock >= 0
        assert np.all(np.diff(outcomes.survival.astype(int), axis=1) <= 0)
    data, _ = world.experience(condition, 64, 31)
    assert set(vars(data)) == {"observations", "actions", "survival"}
    counterfactuals = world.counterfactuals(trials.cases)
    np.testing.assert_array_equal(data.survival, counterfactuals[np.arange(64), data.actions])


@pytest.mark.parametrize("gated", (False, True))
def test_mode_changes_consequences_without_changing_images_or_actions(gated):
    world = ConditionalWorld()
    a, _ = world.experience(Condition(0, gated), 1024, 31)
    b, _ = world.experience(Condition(1, gated), 1024, 31)
    np.testing.assert_array_equal(a.observations, b.observations)
    np.testing.assert_array_equal(a.actions, b.actions)
    assert (a.survival != b.survival).mean() > .1


def test_likelihood_selects_explanation_and_uses_only_terminal_labels():
    logits = torch.zeros(8, 2, 5, 3)
    logits[:, 0, :, -1] = 2
    logits[:, 1, :, -1] = -2
    actions = torch.zeros(8, dtype=torch.long)
    y = torch.ones(8, 3)
    positive = evidence_weights(logits, actions, y, 1.)
    assert positive[0] > .99
    y[:, :2] = 0
    assert torch.equal(positive, evidence_weights(logits, actions, y, 1.))
    y[:, -1] = 0
    assert evidence_weights(logits, actions, y, 1.)[1] > .99
    assert torch.isfinite(evidence_weights(logits * 1000, actions, y, 1.)).all()


def test_support_is_strictly_past_and_replay_keeps_original_binding(config):
    learner = ConditionalLearner("conditional", 9, config)
    world = ConditionalWorld()
    first, _ = world.experience(Condition(0), 8, 1)
    second, _ = world.experience(Condition(1), 8, 2)
    learner.train(first)
    assert learner.memory.packets[0].support is None
    learner.train(second)
    packet = learner.memory.packets[1]
    assert packet.support.fingerprint() == first.fingerprint()
    assert packet.query.fingerprint() == second.fingerprint()
    assert packet.oracle_modes is None
    assert learner.history.fingerprint() == second.fingerprint()


def test_ordinary_interface_rejects_latent_modes(config):
    data, modes = ConditionalWorld().experience(Condition(0), 8, 1)
    learner = ConditionalLearner("conditional", 9, config)
    with pytest.raises(ValueError, match="latent"):
        learner.train(data, modes)
    oracle = ConditionalLearner("oracle", 9, config)
    with pytest.raises(ValueError, match="oracle mode"):
        oracle.train(data)


def test_probes_cannot_update_model_history_replay_or_rng(config):
    learners = [ConditionalLearner("conditional", 9, config) for _ in range(2)]
    world, evaluator = ConditionalWorld(), Evaluator(9, config)
    for step in range(3):
        data, _ = world.experience(Condition(step % 2), 8, step + 2)
        for learner in learners:
            learner.train(data)
        evaluator.switch_probe(learners[0], Condition(1))
    a, b = learners
    assert state_hash(a.model.state_dict()) == state_hash(b.model.state_dict())
    assert a.history.fingerprint() == b.history.fingerprint()
    assert a.memory.ids == b.memory.ids
    assert a.memory.sampling_rng.bit_generator.state == b.memory.sampling_rng.bit_generator.state
    assert a.cost["optimizer_steps"] == b.cost["optimizer_steps"]


def test_pooled_ignores_history_and_routing_is_permutation_invariant(config):
    world = ConditionalWorld()
    a, _ = world.experience(Condition(0), 8, 1)
    b, _ = world.experience(Condition(1), 8, 1)
    pooled, conditional = (ConditionalLearner(m, 9, config) for m in ("pooled", "conditional"))
    np.testing.assert_array_equal(pooled.predict(a.observations, a)[0],
                                  pooled.predict(a.observations, b)[0])
    permuted = a.take(np.arange(7, -1, -1))
    np.testing.assert_allclose(conditional.predict(b.observations, a)[0],
                               conditional.predict(b.observations, permuted)[0], atol=1e-7)


def test_replay_bounded_and_independent_of_sampling():
    a, b = (PacketMemory(4, 9) for _ in range(2))
    world = ConditionalWorld()
    for index in range(30):
        data, _ = world.experience(Condition(index % 2), 8, index)
        packet = Packet(None, data)
        a.add(packet)
        b.add(packet)
        for _ in range(3):
            a.sample()
    assert a.ids == b.ids
    assert len(a.ids) == len(set(a.ids)) == 4
    assert a.seen == 30


def test_freezing_preserves_learned_features_while_heads_remain_plastic(config):
    learner = ConditionalLearner("frozen_features", 9, config)
    data, _ = ConditionalWorld().experience(Condition(0), 8, 1)
    learner.train(data)
    before = copy.deepcopy(learner.model.state_dict())
    learner.freeze_features()
    learner.train(data)
    after = learner.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if not k.startswith("heads."))
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("heads."))


def test_checkpoint_resume_is_exact_and_rejects_changed_identity(config, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run_suite(config, full)
    import shutil
    shutil.copytree(full, resumed)
    result = resumed / "recurring_conditional_19" / "result.json"
    result.unlink()  # Re-evaluate the saved complete learner; do not retrain.
    run_suite(config, resumed, resume=True)
    a = json.loads((full / "recurring_conditional_19" / "result.json").read_text())
    b = json.loads(result.read_text())
    assert a["diagnostics"] == b["diagnostics"]
    assert a["return_probe"] == b["return_probe"]
    assert a["final_panel"] == b["final_panel"]
    altered = dict(config, lr=.001)
    with pytest.raises(ValueError, match="mismatch"):
        run_suite(altered, resumed, resume=True)


def test_interrupted_block_resume_matches_uninterrupted_learning(config, tmp_path, monkeypatch):
    from acp_cl.conditional import study

    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    run_suite(config, full)
    original = study.atomic_checkpoint

    def crash_after_four_blocks(path, payload):
        original(path, payload)
        if "record" in payload and len(payload["record"]["blocks"]) == 4:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(study, "atomic_checkpoint", crash_after_four_blocks)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_suite(config, interrupted)
    monkeypatch.setattr(study, "atomic_checkpoint", original)
    run_suite(config, interrupted, resume=True)
    a = json.loads((full / "recurring_conditional_19" / "result.json").read_text())
    b = json.loads((interrupted / "recurring_conditional_19" / "result.json").read_text())
    assert a["blocks"] == b["blocks"]
    assert a["return_probe"] == b["return_probe"]
    assert a["final_panel"] == b["final_panel"]
    for field in ("final_hash", "memory_ids", "packets_seen", "encoder_displacement"):
        assert a["diagnostics"][field] == b["diagnostics"][field]
    for field in ("arrivals", "optimizer_steps", "query_presentations", "replay_presentations"):
        assert a["diagnostics"]["cost"][field] == b["diagnostics"]["cost"][field]


def test_schedule_and_data_are_deterministic_without_mode_metadata_in_experience():
    conditions = schedule(19, "recurring")
    assert conditions[4] == conditions[8]
    assert all(c.mode != conditions[8].mode for c in conditions[5:8])
    world = ConditionalWorld()
    a, _ = training_batch(world, 19, "recurring", 4, 2, 32)
    b, _ = training_batch(world, 19, "recurring", 4, 2, 32)
    assert a.fingerprint() == b.fingerprint()
    assert all(c.mode == -1 for c in schedule(19, "unpredictable"))


@pytest.mark.parametrize("field,value", (("batch_size", 7), ("experts", 1), ("evidence_strength", 0),
                                         ("eval_size", 9), ("methods", ["unknown"])))
def test_invalid_scientific_config_rejected(config, field, value):
    with pytest.raises(ValueError):
        validate_config(dict(config, **{field: value}))
