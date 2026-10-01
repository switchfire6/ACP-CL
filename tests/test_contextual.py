"""Causal interfaces, matched data/capacity, and shared-prefix isolation."""

import copy
import json

import numpy as np
import pytest
import torch

from acp_cl.conditional.learner import ConditionalLearner
from acp_cl.contextual.learner import ContextLearner, METHODS
from acp_cl.contextual.study import Evaluator, run_suite, validate_config
from acp_cl.contextual.world import (ContextWorld, Environment, challenge, challenge_batch,
                                    prefix, prefix_batch)
from acp_cl.persistence.learner import state_hash


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return dict(study="test", seeds=[19], gaps=["short"], branches=["return", "novel", "noise"],
                methods=["recurrent"], block_size=32, batch_size=8, eval_size=8, probe_every=8,
                support_replicates=1, width=12, experts=4, context_width=8, decoder_width=12,
                lr=.002, memory_packets=3, updates_per_batch=1, evidence_strength=1.,
                feedback_noise=.2, interaction_features=True, threads=1, workers=1)


def test_near_equal_capacity_and_identical_initial_visual_encoder(config):
    config.update(width=64, context_width=12, decoder_width=64)
    learners = [ContextLearner(m, 117, config) for m in METHODS]
    counts = [sum(p.numel() for p in learner.model.parameters()) for learner in learners]
    assert min(counts) == 57831
    assert max(counts) == 58112
    assert max(counts)/min(counts) < 1.005
    assert len({learner.initial_encoder_hash for learner in learners}) == 1


def test_original_reference_remains_exact(config):
    reference = ConditionalLearner("conditional", 9, config)
    current = ContextLearner("conditional", 9, config)
    world = ContextWorld()
    for i in range(3):
        data, _ = world.experience(Environment(i % 2), 8, i + 3)
        reference.train(data)
        current.train(data)
    assert state_hash(reference.model.state_dict()) == state_hash(current.model.state_dict())
    assert reference.memory.ids == current.memory.ids


def test_query_routing_starts_at_reference_function(config):
    reference = ContextLearner("conditional", 9, config)
    modified = ContextLearner("query_routed", 9, config)
    support, _ = ContextWorld().experience(Environment(0), 8, 1)
    query, _ = ContextWorld().experience(Environment(1, True), 8, 2)
    np.testing.assert_allclose(reference.predict(query.observations, support)[0],
                               modified.predict(query.observations, support)[0], atol=1e-7)
    modified.train(support)
    assert torch.count_nonzero(modified.model.query_route.weight) > 0


@pytest.mark.parametrize("method", ("recurrent", "query_routed"))
def test_past_only_support_and_no_persistent_hidden_state(method, config):
    learner = ContextLearner(method, 9, config)
    world = ContextWorld()
    first, _ = world.experience(Environment(0), 8, 1)
    second, _ = world.experience(Environment(1), 8, 2)
    before = learner.predict(second.observations, first)[0]
    learner.predict(first.observations, second)  # Must not influence a later query.
    np.testing.assert_array_equal(before, learner.predict(second.observations, first)[0])
    learner.train(first)
    learner.train(second)
    assert learner.memory.packets[0].support is None
    assert learner.memory.packets[1].support.fingerprint() == first.fingerprint()
    assert learner.memory.packets[1].query.fingerprint() == second.fingerprint()
    with pytest.raises(ValueError, match="latent"):
        learner.train(first, np.zeros(8, dtype=np.uint8))


@pytest.mark.parametrize("method", ("recurrent", "query_routed"))
def test_probes_do_not_change_learning_history_or_replay_rng(method, config):
    a, b = (ContextLearner(method, 9, config) for _ in range(2))
    world, evaluator = ContextWorld(), Evaluator(9, config)
    for i in range(3):
        data, _ = world.experience(Environment(i % 2), 8, i + 10)
        a.train(data)
        b.train(data)
        evaluator.switch_probe(a, Environment(0, True))
    assert state_hash(a.model.state_dict()) == state_hash(b.model.state_dict())
    assert a.history.fingerprint() == b.history.fingerprint()
    assert a.memory.sampling_rng.bit_generator.state == b.memory.sampling_rng.bit_generator.state


def test_gap_orders_have_exactly_same_records_counts_and_total_duration(config):
    world = ContextWorld()
    records = {}
    for gap in ("short", "long"):
        blocks = prefix(117, gap, config)
        records[gap] = [prefix_batch(world, 117, block, i, 8)[0].fingerprint()
                        for block in blocks for i in range(block["size"] // 8)]
        assert blocks[-1]["mode"] == 0  # B for seed 117.
        latest_a = max(i for i, b in enumerate(blocks) if b["mode"] == 1)
        assert 7 - latest_a == (1 if gap == "short" else 3)
    assert sorted(records["short"]) == sorted(records["long"])
    assert records["short"] != records["long"]


def test_noise_changes_measurement_not_physics_or_actions():
    world = ContextWorld()
    clean, _ = world.experience(Environment(0), 4096, 9)
    noisy, _ = world.experience(Environment(0, feedback_noise=.2), 4096, 9)
    np.testing.assert_array_equal(clean.observations, noisy.observations)
    np.testing.assert_array_equal(clean.actions, noisy.actions)
    assert .05 < (clean.survival != noisy.survival).any(axis=1).mean() < .25
    assert np.all(np.diff(noisy.survival.astype(int), axis=1) <= 0)
    a, b = (world.dataset(c, 32, 9) for c in (Environment(0), Environment(0, feedback_noise=.2)))
    np.testing.assert_array_equal(a.cases.supplies, b.cases.supplies)
    np.testing.assert_array_equal(world.counterfactuals(a.cases), world.counterfactuals(b.cases))


def test_pooled_diagnostic_never_switches_to_evidence_routing(config):
    learner = ContextLearner("pooled", 9, config)
    report = Evaluator(9, config).switch_probe(learner, Environment(0))
    for replicate in report["replicates"]:
        assert replicate["curve"][-1]["metrics"] == replicate["broken_binding"]


def test_branch_clones_share_initial_state_without_cross_branch_learning(config, tmp_path):
    run_suite(config, tmp_path)
    records = [json.loads((tmp_path / f"short_{branch}_recurrent_19/result.json").read_text())
               for branch in config["branches"]]
    assert len({r["before"]["model_sha256"] for r in records}) == 1
    assert all(r["prefix"] == records[0]["prefix"] for r in records)
    checkpoint = torch.load(tmp_path / "prefix_short_recurrent_19/checkpoint.pt", weights_only=False)
    assert state_hash(checkpoint["learner"].model.state_dict()) == records[0]["before"]["model_sha256"]
    assert checkpoint["learner"].cost["arrivals"] == sum(b["size"] for b in prefix(19, "short", config))


def test_feature_freeze_preserves_encoder_but_trains_context(config):
    learner = ContextLearner("recurrent_frozen", 9, config)
    data, _ = ContextWorld().experience(Environment(0), 8, 9)
    learner.train(data)
    before = copy.deepcopy(learner.model.state_dict())
    learner.freeze_features()
    learner.train(data)
    for key, value in learner.model.state_dict().items():
        if key.startswith(("frame.", "temporal.")):
            assert torch.equal(value, before[key])
    assert any(not torch.equal(v, before[k]) for k, v in learner.model.state_dict().items()
               if k.startswith("context."))


def test_resume_after_partial_branch_is_exact(config, tmp_path, monkeypatch):
    from acp_cl.contextual import study
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    run_suite(config, full)
    original = study.atomic_checkpoint

    def crash(path, payload):
        original(path, payload)
        if payload["record"].get("branch") == "novel" and payload["record"].get("batches_done") == 2:
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(study, "atomic_checkpoint", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_suite(config, interrupted)
    monkeypatch.setattr(study, "atomic_checkpoint", original)
    run_suite(config, interrupted, resume=True)
    for branch in config["branches"]:
        relative = f"short_{branch}_recurrent_19/result.json"
        a, b = (json.loads((p / relative).read_text()) for p in (full, interrupted))
        for key in ("curve", "before", "after", "known", "batch_sha256"):
            assert a[key] == b[key]
        assert a["diagnostics"]["final_hash"] == b["diagnostics"]["final_hash"]
    with pytest.raises(ValueError, match="identical"):
        run_suite(dict(config, lr=.001), interrupted, resume=True)


def test_challenge_semantics_and_actual_records(config):
    assert challenge(19, "return", config) == Environment(1)
    assert challenge(19, "novel", config) == Environment(0, gated=True)
    assert challenge(19, "noise", config) == Environment(0, feedback_noise=.2)
    a, _ = challenge_batch(ContextWorld(), 19, "novel", config, 0)
    b, _ = challenge_batch(ContextWorld(), 19, "novel", config, 0)
    assert a.fingerprint() == b.fingerprint()
    assert set(vars(a)) == {"observations", "actions", "survival"}


@pytest.mark.parametrize("field,value", (("block_size", 31), ("feedback_noise", -1),
                                         ("context_width", 0), ("branches", ["unknown"])))
def test_invalid_configs(config, field, value):
    with pytest.raises(ValueError):
        validate_config(dict(config, **{field: value}))
