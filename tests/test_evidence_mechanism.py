"""Future-only evidence, exact snapshot promotion and matched observation streams."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_training_state import memory_signature, signature, tensor_tree
from acp_cl.acquisition.world import AcquisitionWorld, Law, order, stage_law
from acp_cl.contextual.learner import ContextLearner
from acp_cl.evidence_consolidation.mechanism import (
    DELAYED, POLICIES, Consolidation, Snapshot, accept,
)
from acp_cl.evidence_consolidation.schedule import experiences, schedule, training_law
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.world import Experience


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return json.loads(Path("configs/replay_renewal_policy_smoke.json").read_text()) | dict(
        evidence_window=8, evidence_margin=.002, evidence_positive=6,
        burst_period=64, burst_length=16, burst_noise=.8,
    )


def synthetic_scores(current, gain):
    """Known terminal Brier improvement without coupling acceptance to learning."""
    assert not current.survival.any()
    proposed = np.full((len(current), 5, 3), .5)
    predictions = {policy: np.full_like(proposed, np.sqrt(.25+gain))
                   for policy in POLICIES}
    return predictions, proposed


@pytest.mark.parametrize("policy,gains,expected", [
    ("periodic", [-.5]*8, True),
    ("single", [.003]+[-.5]*7, True),
    ("single", [.002]+[.5]*7, False),
    ("single", [-.001]+[.5]*7, False),
    ("sustained", [.01]*6+[-.001]*2, True),
    ("sustained", [.01]*5+[-.001]*3, False),
    ("sustained", [.001]*6+[-.01]*2, False),
    ("sustained", [.002]*8, False),
    ("sustained", [.01]*5+[0.]*3, False),
])
def test_acceptance_requires_declared_future_evidence(policy, gains, expected):
    assert accept(policy, gains, .002, 6) is expected


@pytest.mark.parametrize("policy", DELAYED)
@pytest.mark.parametrize("gains", ([], [np.nan], [np.inf], [-np.inf]))
def test_all_delayed_rules_reject_empty_or_nonfinite_evidence(policy, gains):
    with pytest.raises(ValueError, match="finite nonempty"):
        accept(policy, gains, .002, 6)


def test_unknown_acceptance_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        accept("unknown", [.01]*8, .002, 6)


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_frozen_snapshot_has_no_optimizer_replay_or_private_history(model, config):
    learner = ContextLearner(model, 19, config)
    data = AcquisitionWorld().experience(Law(0), 8, 600)
    learner.train(data)
    snapshot = Snapshot(learner)
    expected = state_hash(learner.model.state_dict())
    assert snapshot.sha256 == expected
    assert snapshot.history is None
    assert not hasattr(snapshot, "optimizer")
    assert not hasattr(snapshot, "memory")
    assert all(not p.requires_grad for p in snapshot.model.parameters())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(
        learner.model.parameters(), snapshot.model.parameters()))
    learner.train(AcquisitionWorld().experience(Law(1), 8, 601))
    assert state_hash(learner.model.state_dict()) != expected
    assert state_hash(snapshot.model.state_dict()) == snapshot.sha256 == expected


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_draft_is_bitwise_standalone_uniform_training_through_decisions(model, config):
    standalone = ContextLearner(model, 19, config)
    draft = ContextLearner(model, 19, config)
    mechanism = Consolidation(draft, config)
    assert mechanism.predictors()["immediate"] is draft
    observed_decisions = []
    for index in range(24):
        current = AcquisitionWorld().experience(Law(index % 2, (0, 1)), 8, 700+index)
        before = signature(draft)
        predictions, proposed = mechanism.prepare(current.observations)
        assert signature(draft) == before
        repeated, repeated_proposal = mechanism.prepare(current.observations)
        assert mechanism.proposal_count == index//8+1
        assert signature(draft) == before
        for policy in POLICIES:
            np.testing.assert_array_equal(predictions[policy], repeated[policy])
        np.testing.assert_array_equal(proposed, repeated_proposal)
        for values in (*predictions.values(), proposed):
            assert np.isfinite(values).all()
            assert ((0 <= values) & (values <= 1)).all()
            assert (np.diff(values, axis=-1) <= 0).all()
        decision = mechanism.observe(current, predictions, proposed)
        standalone.train(current)
        assert signature(draft) == signature(standalone)
        assert tensor_tree(draft.optimizer.state_dict()) == tensor_tree(
            standalone.optimizer.state_dict())
        assert memory_signature(draft.memory) == memory_signature(standalone.memory)
        assert len(draft.memory.packets) == min(index+1, config["memory_packets"])
        assert draft.cost | {"training_seconds": 0} == (
            standalone.cost | {"training_seconds": 0})
        if decision is not None:
            observed_decisions.append(decision["end_batch"])
            assert set(decision["accepted"]) == set(DELAYED)
    assert observed_decisions == [8, 16, 24]
    assert mechanism.adoptions["periodic"] == 3
    assert mechanism.state()["batches"] == 24


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_promotion_copies_the_tested_version_and_all_rules_wait_same_window(model, config):
    draft = ContextLearner(model, 19, config)
    mechanism = Consolidation(draft, config)
    world = AcquisitionWorld()
    decisions = []
    for window in range(2):
        expected_hash = state_hash(draft.model.state_dict())
        before_hashes = dict(mechanism.state()["committed"])
        proposal = None
        for offset in range(8):
            current = world.experience(Law(window), 8, 800+8*window+offset)
            current = replace(current, survival=np.zeros_like(current.survival))
            mechanism.prepare(current.observations)
            if proposal is None:
                proposal = mechanism.proposal
            assert mechanism.proposal is proposal
            assert state_hash(proposal.model.state_dict()) == expected_hash
            assert mechanism.state()["committed"] == before_hashes
            # In window two the first observation disagrees, while the other
            # seven agree. Single and sustained must make different decisions.
            gain = -.004 if window == 1 and offset == 0 else .02
            predictions, proposed = synthetic_scores(current, gain)
            decision = mechanism.observe(current, predictions, proposed)
            if offset < 7:
                assert decision is None
                assert mechanism.proposal is proposal
                assert mechanism.state()["committed"] == before_hashes
                assert len(mechanism.state()["gains"]["single"]) == offset+1
            else:
                decisions.append(decision)
                assert decision["start_batch"] == 8*window
                assert decision["end_batch"] == 8*(window+1)
                assert decision["committed_before"] == before_hashes
                assert decision["proposal_hash"] == expected_hash
                assert decision["accepted"] == dict(
                    periodic=True, single=window == 0, sustained=True)
                assert mechanism.committed["periodic"] is proposal
                assert mechanism.committed["sustained"] is proposal
                assert mechanism.state()["gains"] == {policy: [] for policy in DELAYED}
                assert mechanism.proposal is None
                assert mechanism.proposal_batch is None
                assert state_hash(draft.model.state_dict()) != expected_hash
                for policy in DELAYED:
                    target = expected_hash if decision["accepted"][policy] else before_hashes[policy]
                    assert mechanism.committed[policy].sha256 == target
                    assert state_hash(mechanism.committed[policy].model.state_dict()) == target
        assert state_hash(proposal.model.state_dict()) == expected_hash
    assert mechanism.adoptions == dict(periodic=2, single=1, sustained=2)
    # A later decision cannot erase the recorded evidence for the first window.
    assert all(len(values) == 8 for decision in decisions for values in decision["gains"].values())


def test_every_predictor_uses_only_preceding_support(config, monkeypatch):
    draft = ContextLearner("conditional", 19, config)
    mechanism = Consolidation(draft, config)
    original_predict = ContextLearner.predict
    calls = []

    def spy(self, observations, support=None, *args, **kwargs):
        calls.append((observations, support))
        return original_predict(self, observations, support, *args, **kwargs)

    monkeypatch.setattr(ContextLearner, "predict", spy)
    previous = None
    for index in range(3):
        current = AcquisitionWorld().experience(Law(0), 8, 900+index)
        calls.clear()
        predictions, proposed = mechanism.prepare(current.observations)
        assert len(calls) == 5
        assert all(observations is current.observations and support is previous
                   for observations, support in calls)
        mechanism.observe(current, predictions, proposed)
        assert draft.history is current
        # Before capacity is reached, the last packet is always the new packet.
        assert draft.memory.packets[-1].query is current
        assert draft.memory.packets[-1].support is previous
        previous = current


def test_feedback_scores_only_performed_action_and_terminal_outcome(config, monkeypatch):
    draft = ContextLearner("conditional", 19, config)
    mechanism = Consolidation(draft, config)
    monkeypatch.setattr(draft, "train", lambda current: None)
    current = AcquisitionWorld().experience(Law(0), 8, 920)
    current = replace(current, survival=np.zeros_like(current.survival))
    mechanism.prepare(current.observations)
    predictions, proposed = synthetic_scores(current, .03)
    # Make every unobserved action and both earlier horizons disagree wildly.
    # None is legitimate evidence for the terminal action actually performed.
    for row, action in enumerate(current.actions):
        for values in predictions.values():
            values[row, :, :2] = 1.
            values[row, np.arange(5) != action, -1] = 1.
        proposed[row, :, :2] = 0.
        proposed[row, np.arange(5) != action, -1] = 0.
    mechanism.observe(current, predictions, proposed)
    for gains in mechanism.gains.values():
        assert gains == pytest.approx([.03])


def test_state_reports_are_detached_from_future_decisions(config):
    draft = ContextLearner("conditional", 19, config)
    mechanism = Consolidation(draft, config)
    current = AcquisitionWorld().experience(Law(0), 8, 930)
    predictions, proposed = mechanism.prepare(current.observations)
    mechanism.observe(current, predictions, proposed)
    saved = mechanism.state()
    actual = copy.deepcopy(saved)
    saved["adoptions"]["periodic"] = 999
    saved["gains"]["sustained"].append(999)
    saved["committed"]["single"] = "modified"
    assert mechanism.state() == actual


@pytest.mark.parametrize("seed", range(19, 25))
def test_schedule_introduces_all_cues_and_matches_early_late_conditions(seed):
    blocks = schedule(seed)
    assert len(blocks) == 18
    assert [block.index for block in blocks] == list(range(18))
    assert tuple(block.cue for block in blocks[:3]) == order(seed)
    assert [block.law for block in blocks[:3]] == [stage_law(seed, n) for n in (1, 2, 3)]
    for cycle in range(3):
        group = blocks[3+5*cycle:8+5*cycle]
        assert [block.label for block in group] == ["clean", "noise", "recovery", "return", "revision"]
        assert all(block.cycle == cycle for block in group)
        assert group[0].law == group[1].law == group[2].law == stage_law(seed, 3)
        assert group[3].law == Law(seed % 2)
        assert group[4].law == replace(stage_law(seed, 3), revised=True)
        assert group[1].noise_kind == ("burst" if cycle == 1 else "persistent")
        assert all(block.law.noise == 0 for block in group)
    # This is matched experimental difficulty, not reuse of the same samples.
    # Noise remains a separate endpoint rather than masquerading as real change.
    for early, late in zip(blocks[3:8], blocks[13:18]):
        assert replace(early, index=late.index, cycle=late.cycle) == late
    assert [block.label for block in blocks[3:8] if block.noise_kind == "none"] == [
        "clean", "recovery", "return", "revision"]


def test_burst_noise_boundaries_do_not_change_latent_laws(config):
    blocks = schedule(19)
    burst = blocks[9]
    for index, expected_noise in ((0, .8), (15, .8), (16, 0.), (63, 0.),
                                  (64, .8), (79, .8), (80, 0.)):
        observed = training_law(burst, index, config)
        assert observed.noise == expected_noise
        assert replace(observed, noise=0.) == burst.law
    for index in (0, 15, 16, 63, 64, 80):
        assert training_law(blocks[4], index, config).noise == config["feedback_noise"]
        assert training_law(blocks[14], index, config).noise == config["feedback_noise"]
        assert training_law(blocks[3], index, config).noise == 0.


@pytest.mark.parametrize("block_index,arrival", ((3, 0), (4, 0), (9, 0), (9, 16), (14, 0)))
def test_report_noise_changes_only_labels_with_reproducible_inputs_actions(block_index, arrival, config):
    config = dict(config, batch_size=64)
    block = schedule(19)[block_index]
    reported, clean = experiences(block, 19, arrival, config)
    repeat_reported, repeat_clean = experiences(block, 19, arrival, config)
    assert repeat_reported.fingerprint() == reported.fingerprint()
    assert repeat_clean.fingerprint() == clean.fingerprint()
    np.testing.assert_array_equal(reported.observations, clean.observations)
    np.testing.assert_array_equal(reported.actions, clean.actions)
    for data in (reported, clean):
        assert set(np.unique(data.survival)) <= {0, 1}
        assert (np.diff(data.survival.astype(int), axis=-1) <= 0).all()
    if training_law(block, arrival, config).noise:
        assert not np.array_equal(reported.survival, clean.survival)
    else:
        assert reported is clean


def test_structured_reporting_error_can_be_indistinguishable_from_a_true_change():
    """A limitation of evidence alone; the benchmark's random noise is easier."""
    world, law = AcquisitionWorld(), Law(0, (0, 1, 2))
    unchanged = world.experience(law, 256, 970)
    changed = world.experience(replace(law, revised=True), 256, 970)
    np.testing.assert_array_equal(unchanged.observations, changed.observations)
    np.testing.assert_array_equal(unchanged.actions, changed.actions)
    assert not np.array_equal(unchanged.survival, changed.survival)
    # If the unchanged world reports exactly the changed world's outcomes,
    # every permitted input byte is identical. No learner can identify which
    # cause generated that record without another assumption or observation.
    structured_error = Experience(unchanged.observations, unchanged.actions,
                                  changed.survival.copy())
    assert structured_error.fingerprint() == changed.fingerprint()
