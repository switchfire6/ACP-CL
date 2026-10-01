"""Memory equivalence, causal partitions, factor isolation and exact recovery."""

import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_replay_renewal import audit, signature, tensor_tree
from summarize_replay_renewal import audit_records, policy_screen, summarize
from acp_cl.acquisition.world import AcquisitionWorld, Law
from acp_cl.conditional.learner import PacketMemory
from acp_cl.contextual.learner import ContextLearner
from acp_cl.replay_renewal.memory import (POLICIES, RESET_ARMS, RecentHistoricalMemory,
    ReservoirMemory, memory_state, reset_memory)
from acp_cl.replay_renewal import study


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return json.loads(Path("configs/replay_renewal_policy_smoke.json").read_text()) | {"models": ["conditional"]}


@pytest.mark.parametrize("capacity", (0, 1, 4, 16))
def test_reservoir_matches_original_and_zero_recent_boundary(capacity):
    old = PacketMemory(capacity, 71)
    new = ReservoirMemory(capacity, 71)
    split = RecentHistoricalMemory(capacity, 0, 71)
    for i in range(400):
        for _ in range(12):
            assert old.sample() == new.sample() == split.sample()
        for memory in (old, new, split):
            memory.add(i)
        assert old.ids == new.ids == split.ids
        assert old.membership_rng.bit_generator.state == new.membership_rng.bit_generator.state == split.membership_rng.bit_generator.state
        assert new.age == old.seen == split.seen


@pytest.mark.parametrize("recent", (1, 8, 16))
def test_recent_and_historical_are_bounded_disjoint_past_only(recent):
    memory = RecentHistoricalMemory(16, recent, 7)
    for i in range(1000):
        sampled = memory.sample()
        assert sampled is None or sampled < i
        memory.add(i)
        assert memory.recent_ids == list(range(max(0, i+1-recent), i+1))
        assert set(memory.recent_ids).isdisjoint(memory.historical_ids)
        assert len(memory.ids) == min(16, i+1)
        assert all(j < i+1-recent for j in memory.historical_ids)
    draws = [memory.sample() for _ in range(10000)]
    proportion = sum(i in memory.recent_ids for i in draws)/len(draws)
    assert abs(proportion-recent/16) < .02


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_original_uniform_training_is_bitwise_unchanged(model, config):
    old = ContextLearner(model, 19, config)
    new = study.new_learner(model, 19, config)
    for i in range(20):
        data = AcquisitionWorld().experience(Law(i % 2), 8, 100+i)
        old.train(data)
        new.train(data)
        assert old.diagnostics()["final_hash"] == new.diagnostics()["final_hash"]
        assert tensor_tree(old.optimizer.state_dict()) == tensor_tree(new.optimizer.state_dict())
        assert old.memory.ids == new.memory.ids


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
@pytest.mark.parametrize("arm", RESET_ARMS)
def test_reset_isolates_declared_factors(model, arm, config):
    base = study.new_learner(model, 19, config)
    for i in range(10):
        base.train(AcquisitionWorld().experience(Law(i % 2), 8, 300+i))
    before = signature(base)
    changed = study.fork(base, arm, 19, config)
    assert base.diagnostics()["final_hash"] == changed.diagnostics()["final_hash"]
    assert tensor_tree(base.optimizer.state_dict()) == tensor_tree(changed.optimizer.state_dict())
    assert changed.history.fingerprint() == base.history.fingerprint()
    clear = arm in ("clear", "clear_rebase", "full_reset")
    rebase = arm in ("rebase", "clear_rebase", "full_reset")
    assert changed.memory.ids == ([] if clear else base.memory.ids)
    assert changed.memory.age == (len(changed.memory.ids) if rebase else base.memory.age)
    assert changed.memory.seen == base.memory.seen
    rng_source = ReservoirMemory(4, 19) if arm in ("rng_reset", "full_reset") else base.memory
    for field in ("membership_rng", "sampling_rng"):
        assert getattr(changed.memory, field).bit_generator.state == getattr(rng_source, field).bit_generator.state
    changed.train(AcquisitionWorld().experience(Law(0, (0,)), 8, 520))
    assert signature(base) == before


def test_full_reset_matches_old_reset_learning_but_preserves_lifetime_ids(config):
    base = study.new_learner("conditional", 19, config)
    for i in range(12):
        base.train(AcquisitionWorld().experience(Law(i % 2), 8, 600+i))
    a, b = study.fork(base, "full_reset", 19, config), copy.deepcopy(base)
    b.memory = PacketMemory(4, 19)
    for i in range(20):
        data = AcquisitionWorld().experience(Law(1, (1,)), 8, 700+i)
        a.train(data)
        b.train(data)
        assert a.diagnostics()["final_hash"] == b.diagnostics()["final_hash"]
        assert a.memory.ids == [j+12 for j in b.memory.ids]


@pytest.mark.parametrize("experiment", ("decomposition", "policy"))
def test_full_cohort_resume_exact_forks_audit_and_budget(experiment, config, tmp_path, monkeypatch):
    config = dict(config, experiment=experiment)
    complete, interrupted = tmp_path/"complete", tmp_path/"interrupted"
    study.run_suite(config, complete)
    original = study.atomic_checkpoint

    def crash(path, payload):
        original(path, payload)
        r = payload["record"]
        target = r.get("arm") == "clear_rebase" if experiment == "decomposition" else r.get("branch") == "noise" and r.get("arm") == "split"
        if target and r.get("batches_done") == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(study, "atomic_checkpoint", crash)
    with pytest.raises(RuntimeError, match="simulated"):
        study.run_suite(config, interrupted)
    monkeypatch.setattr(study, "atomic_checkpoint", original)
    study.run_suite(config, interrupted, resume=True)
    _, a, _ = audit_records(complete)
    _, b, _ = audit_records(interrupted)
    assert len(a) == (7 if experiment == "decomposition" else 24)
    for key, record in a.items():
        for field in ("curve", "before", "after", "valid_before", "valid_after", "batch_sha256", "start_memory", "end_memory"):
            assert record[field] == b[key][field]
        assert record["diagnostics"]["final_hash"] == b[key]["diagnostics"]["final_hash"]
    result = audit(interrupted, tmp_path/"audit.json")
    assert result["passed"] and result["prefix_checkpoints"] == (1 if experiment == "decomposition" else 3)
    assert result["matched_clean_noise_pairs"] == (0 if experiment == "decomposition" else 3)
    summary = summarize(interrupted, tmp_path/"summary")
    assert len(summary["rows"]) == len(a)
    with pytest.raises(ValueError, match="identical"):
        study.run_suite(dict(config, lr=.003), interrupted, resume=True)


def metric(value):
    return dict(mean=value, lower=value-.0001, upper=value+.0001, differences=[value]*6, n=6)


def test_screen_uses_terminal_retention_and_every_guardrail():
    novel = {"brier_auc": metric(-.003), "valid_after": metric(0), "valid_damage": metric(.9), "survival_auc": metric(0)}
    final = {b: copy.deepcopy(novel) for b in study.BRANCHES}
    assert policy_screen(novel, final, metric(0), True)["passed"]
    for field, value in (("brier_auc", -.001), ("valid_after", .006), ("survival_auc", -.011)):
        changed = copy.deepcopy(novel)
        changed[field] = metric(value)
        assert not policy_screen(changed, final, metric(0), True)["passed"]
    for branch in study.BRANCHES:
        for field, value in (("brier_auc", .006), ("valid_after", .006), ("survival_auc", -.011)):
            changed = copy.deepcopy(final)
            changed[branch][field] = metric(value)
            assert not policy_screen(novel, changed, metric(0), True)["passed"]
    assert not policy_screen(novel, final, metric(.006), True)["passed"]
    assert not policy_screen(novel, final, metric(0), False)["passed"]
    changed = copy.deepcopy(novel)
    changed["brier_auc"]["differences"] = [-.004]*4+[.001]*2
    assert not policy_screen(changed, final, metric(0), True)["passed"]


def test_invalid_memory_and_config_are_rejected(config):
    with pytest.raises(ValueError):
        RecentHistoricalMemory(4, 5, 19)
    with pytest.raises(ValueError):
        reset_memory(ReservoirMemory(4, 19), "unknown", 19)
    for change in ({"recent_packets": 0}, {"cpu_affinity": -1}, {"experiment": "other"}):
        with pytest.raises(ValueError):
            study.validate_config(dict(config, **change))
    learners = [study.new_learner("conditional", 19, config, policy=p) for p in POLICIES]
    assert len({p.initial_hash for p in learners}) == 1
    assert len({memory_state(p.memory)["type"] for p in learners}) == 2
