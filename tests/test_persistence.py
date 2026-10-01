"""Scientific contracts for the bounded joint-persistence study."""

import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from acp_cl.persistence.learner import Learner, state_hash
from acp_cl.persistence.memory import METHODS, Memory
from acp_cl.persistence.study import Evaluator, run_suite, trial_seed, validate_config
from acp_cl.persistence.world import (ACTIONS, COMPOSITIONS, HORIZONS, IMAGE_SHAPE, KNOWN,
                                      Cases, TransferWorld, schedule)


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return dict(study="test", seeds=[19], schedules=["short"], methods=["joint"], block_size=16,
                batch_size=8, eval_size=8, probe_every=16, width=12, lr=.003,
                memory_capacity=12, updates_per_batch=1, threads=1)


@pytest.mark.parametrize("regime", (*KNOWN, *COMPOSITIONS))
def test_resource_conservation_and_absorbing_failure(regime):
    world = TransferWorld()
    cases = world.cases(regime, 47, 12)
    for action in range(len(ACTIONS)):
        outcome = world.simulate(cases, np.full(47, action))
        assert outcome.conservation_error < 1e-10
        assert outcome.min_stock >= 0
        assert np.all(np.diff(outcome.survival.astype(int), axis=1) <= 0)
        for index, horizon in enumerate(HORIZONS):
            np.testing.assert_array_equal(outcome.survival[:, index], outcome.lifetime > horizon)


def deterministic_cases(reserves=(4, 10), efficiency=1.0, delay=0):
    return Cases(np.zeros((1, *IMAGE_SHAPE), dtype=np.uint8), np.array([reserves], dtype=float),
                 np.tile(np.array([3.0, 1.0]), (1, 12, 1)),
                 np.array([efficiency]), np.array([delay]))


def test_exchange_can_help_or_harm_and_does_not_create_supply():
    world = TransferWorld()
    outcomes = [world.simulate(deterministic_cases(), np.array([a])) for a in range(5)]
    assert outcomes[0].survival[0, -1] == 0  # B exhausts reserve without help.
    assert outcomes[4].survival[0, -1] == 1  # A sends four; both survive.
    assert outcomes[3].lifetime[0] < outcomes[0].lifetime[0]  # Wrong direction harms B.
    lossy = world.simulate(deterministic_cases(efficiency=.35), np.array([4]))
    assert lossy.survival[0, -1] == 0  # Losses cannot be wished away by pooling.


def test_transport_can_arrive_too_late_and_cannot_revive():
    world = TransferWorld()
    immediate = world.simulate(deterministic_cases((4, .5)), np.array([4]))
    delayed = world.simulate(deterministic_cases((4, .5), delay=2), np.array([4]))
    assert immediate.survival[0, 0] == 1
    assert delayed.lifetime[0] == 1
    assert not delayed.survival.any()


def test_history_is_required_when_current_observations_match():
    world = TransferWorld()
    a = world.cases(KNOWN[0], 20, 88)
    b = world.cases(replace(KNOWN[0], source=-1), 20, 88)
    np.testing.assert_array_equal(a.observations[:, -1], b.observations[:, -1])
    assert not np.array_equal(a.observations, b.observations)
    assert not np.array_equal(world.counterfactuals(a), world.counterfactuals(b))


def test_record_exposes_only_actual_randomized_action_and_its_outcome():
    world = TransferWorld()
    data = world.record(KNOWN[2], 100, 771)
    assert set(vars(data)) == {"observations", "actions", "survival"}
    cases = world.cases(KNOWN[2], 100, 771)
    counterfactuals = world.counterfactuals(cases)
    np.testing.assert_array_equal(data.survival, counterfactuals[np.arange(100), data.actions])
    assert data.fingerprint() == world.record(KNOWN[2], 100, 771).fingerprint()


def test_short_and_long_gaps_are_count_matched_and_do_not_expose_schedule():
    short, short_info = schedule(21, "short")
    long, long_info = schedule(21, "long")
    assert sorted(r.name for r in short) == sorted(r.name for r in long)
    assert short[10] == long[10]
    assert short_info["gap_blocks"] == 1
    assert long_info["gap_blocks"] == 5
    assert short[8] == long[4] == short[10]


def test_no_return_does_not_reintroduce_retired_context_through_late_gate():
    regimes, _ = schedule(21, "no_return")
    retired, replacement, late = regimes[0], regimes[10], regimes[11]
    assert (retired.lossy, retired.delayed) != (late.lossy, late.delayed)
    assert (replacement.source, replacement.lossy, replacement.delayed) == (
        late.source, late.lossy, late.delayed)


@pytest.mark.parametrize("method", METHODS)
def test_memory_is_bounded_and_sampling_cannot_change_membership(method):
    memory = Memory(method, 12, 44)
    world = TransferWorld()
    for step in range(6):
        current = world.record(KNOWN[step % 4], 8, step)
        pool = memory.candidates(current)
        memory.add(current, np.linspace(.05, .8, len(pool)))
        assert len(memory.data) <= 12
        assert memory.history.counts.sum() == (step + 1) * 8
    before = memory.ids.copy()
    for _ in range(10):
        memory.sample(8)
    np.testing.assert_array_equal(before, memory.ids)
    assert len(set(memory.ids)) == len(memory.ids)
    if method == "recent":
        np.testing.assert_array_equal(memory.ids, np.arange(36, 48))


def test_frequency_estimator_cannot_see_actions_or_labels():
    data = TransferWorld().record(KNOWN[0], 40, 3)
    modified = copy.deepcopy(data)
    modified.actions[:] = 4
    modified.survival[:] = 0
    first, second = Memory("recurrence", 8, 9), Memory("recurrence", 8, 9)
    for memory, values in ((first, data), (second, modified)):
        memory.add(values, np.ones(40) * .5)
    np.testing.assert_array_equal(first.history.counts, second.history.counts)
    np.testing.assert_array_equal(first.ids, second.ids)


def test_probes_do_not_change_learning_or_replay_rng(config):
    learners = [Learner("joint", 17, config) for _ in range(2)]
    evaluator = Evaluator(TransferWorld(), 17, 8)
    for step in range(4):
        batch = TransferWorld().record(KNOWN[step % 4], 8, step + 9)
        for learner in learners:
            learner.train(batch)
        evaluator.evaluate(learners[0], KNOWN[0])
    assert state_hash(learners[0].model.state_dict()) == state_hash(learners[1].model.state_dict())
    np.testing.assert_array_equal(learners[0].memory.ids, learners[1].memory.ids)
    assert learners[0].memory.sampling_rng.bit_generator.state == \
        learners[1].memory.sampling_rng.bit_generator.state


def test_encoder_can_learn_and_freezing_is_real(config):
    learned, frozen = (Learner(m, 17, config) for m in ("uniform", "frozen_encoder"))
    assert learned.initial_hash == frozen.initial_hash
    batch = TransferWorld().record(KNOWN[0], 8, 18)
    for learner in (learned, frozen):
        learner.train(batch)
    assert learned.diagnostics()["encoder_displacement_l2"] > 0
    assert frozen.diagnostics()["encoder_displacement_l2"] == 0
    assert frozen.diagnostics()["final_hash"] != frozen.initial_hash
    learned.freeze_encoder()
    encoder_hash = state_hash({k: v for k, v in learned.model.state_dict().items()
                               if not k.startswith("head.")})
    learned.train(batch)
    assert state_hash({k: v for k, v in learned.model.state_dict().items()
                       if not k.startswith("head.")}) == encoder_hash


def test_checkpoint_continuation_is_exact(config, tmp_path):
    learner = Learner("joint", 8, config)
    batch = TransferWorld().record(KNOWN[0], 8, 87)
    learner.train(batch)
    path = tmp_path / "trusted_local.pt"
    torch.save(learner, path)
    resumed = torch.load(path, weights_only=False)
    following = TransferWorld().record(KNOWN[2], 8, 88)
    for instance in (learner, resumed):
        instance.train(following)
    assert state_hash(learner.model.state_dict()) == state_hash(resumed.model.state_dict())
    np.testing.assert_array_equal(learner.memory.ids, resumed.memory.ids)
    assert learner.memory.sampling_rng.bit_generator.state == resumed.memory.sampling_rng.bit_generator.state


def test_suite_resume_identity_and_paired_training(config, tmp_path):
    config = dict(config, methods=["uniform", "joint"])
    output = tmp_path / "study"
    run_suite(config, output)
    results = [json.loads(p.read_text()) for p in output.glob("*/result.json")]
    assert len(results) == 2
    assert [b["data_sha256"] for b in results[0]["blocks"]] == [
        b["data_sha256"] for b in results[1]["blocks"]]
    for result in results:
        assert result["diagnostics"]["cost"]["arrivals"] == 12 * config["block_size"]
    completed_before = [p.read_bytes() for p in sorted(output.glob("*/result.json"))]
    run_suite(config, output, resume=True)
    assert completed_before == [p.read_bytes() for p in sorted(output.glob("*/result.json"))]
    changed = dict(config, lr=.02)
    with pytest.raises(ValueError, match="mismatch"):
        run_suite(changed, output, resume=True)
    with pytest.raises(ValueError, match="output exists"):
        run_suite(config, output)


def test_separate_random_streams_and_config_validation(config):
    assert trial_seed(1, "training", 3) != trial_seed(1, "evaluation", 3)
    for update in ({"block_size": 17}, {"methods": ["unknown"]}, {"lr": float("nan")},
                   {"seeds": [1, 1]}, {"memory_capacity": 0}):
        with pytest.raises(ValueError):
            validate_config(dict(config, **update))
