"""Causal feedback selection and arithmetic for the frozen output mixture."""

import inspect
import json
import math

import numpy as np
import pytest

from acp_cl.causal_access.mechanism import LaggedMixer, mix_predictions


def forecasts(count=1, core=.2, full=.8):
    return (np.full((count, 5, 3), core, dtype=np.float32),
            np.full((count, 5, 3), full, dtype=np.float32))


def labels(count, terminal=1):
    return np.full((count, 3), terminal, dtype=np.uint8)


def test_feedback_changes_only_subsequent_packet_and_prediction_is_read_only():
    mixer = LaggedMixer()
    core, full = forecasts()
    before = mixer.signature()
    prediction = mixer.predict(core, full)
    assert mixer.weight() == .5 and mixer.signature() == before
    np.testing.assert_array_equal(prediction, np.full(core.shape, .5, dtype=np.float32))
    mixer.observe(core, full, np.array([2]), labels(1))
    expected = float(full[0, 2, -1]) / (float(core[0, 2, -1]) + float(full[0, 2, -1]))
    assert mixer.weight() == pytest.approx(expected)
    np.testing.assert_array_equal(prediction, np.full(core.shape, .5, dtype=np.float32))
    assert (mixer.predict(core, full) > prediction).all()
    assert list(inspect.signature(mixer.predict).parameters) == ["core", "full"]


def test_exact_last_32_scores_and_cumulative_feedback_cost():
    core, full = forecasts(45)
    full[:, :, :] = np.linspace(.1, .9, 45)[:, None, None]
    actions = np.arange(45) % 5
    observed = labels(45)
    observed[::2] = 0
    mixer = LaggedMixer()
    for start, end in ((0, 17), (17, 45)):
        mixer.observe(core[start:end], full[start:end], actions[start:end], observed[start:end])
    expected = []
    for index in range(13, 45):
        p_core, p_full = float(core[index, actions[index], -1]), float(full[index, actions[index], -1])
        expected.append(math.log(p_full / p_core) if observed[index, -1]
                        else math.log((1 - p_full) / (1 - p_core)))
    np.testing.assert_allclose(mixer.state()["scores"], expected, atol=1e-15, rtol=1e-15)
    assert mixer.weight() == pytest.approx(1 / (1 + math.exp(-math.fsum(expected))))
    assert mixer.samples_seen == 45 and mixer.packets_seen == 2
    detached = mixer.state()
    assert json.loads(json.dumps(detached)) == detached
    detached["scores"].clear()
    assert len(mixer.state()["scores"]) == 32


def test_only_performed_action_terminal_feedback_is_used():
    core, full = forecasts(2)
    actions = np.array([1, 4])
    observed = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.uint8)
    altered_core, altered_full = core.copy(), full.copy()
    altered_core[:, :, :] = .999
    altered_full[:, :, :] = .001
    altered_core[np.arange(2), actions, -1] = core[np.arange(2), actions, -1]
    altered_full[np.arange(2), actions, -1] = full[np.arange(2), actions, -1]
    altered_labels = observed.copy()
    altered_labels[0, :2] = 0
    one, two = LaggedMixer(), LaggedMixer()
    one.observe(core, full, actions, observed)
    two.observe(altered_core, altered_full, actions, altered_labels)
    assert one.signature() == two.signature()


def test_batch_partition_preserves_final_evidence_but_counts_packets():
    rng = np.random.default_rng(728)
    core = rng.uniform(.01, .99, (43, 5, 3)).astype(np.float32)
    full = rng.uniform(.01, .99, (43, 5, 3)).astype(np.float32)
    actions = rng.integers(0, 5, 43)
    observed = np.repeat(rng.integers(0, 2, 43)[:, None], 3, axis=1)
    whole, split = LaggedMixer(), LaggedMixer()
    whole.observe(core, full, actions, observed)
    for start, end in ((0, 15), (15, 33), (33, 43)):
        split.observe(core[start:end], full[start:end], actions[start:end], observed[start:end])
    assert whole.state()["scores"] == split.state()["scores"]
    assert whole.weight() == split.weight() and whole.samples_seen == split.samples_seen
    assert whole.packets_seen == 1 and split.packets_seen == 3


@pytest.mark.parametrize("terminal", (0, 1))
@pytest.mark.parametrize("clip", (1e-6, 1e-300))
def test_clipped_extreme_forecasts_have_finite_scores_and_stable_sigmoid(terminal, clip):
    core, full = forecasts(100, core=0, full=1)
    mixer = LaggedMixer(capacity=100, clip=clip)
    mixer.observe(core, full, np.zeros(100, dtype=np.int64), labels(100, terminal))
    assert np.isfinite(mixer.state()["scores"]).all()
    assert mixer.weight() == float(terminal)
    np.testing.assert_array_equal(mixer.predict(core, full), full if terminal else core)


@pytest.mark.parametrize("weight", (0., .37, .5, 1.))
def test_convex_mixture_rounds_once_and_does_not_mutate_inputs(weight):
    rng = np.random.default_rng(82)
    core, full = (np.minimum.accumulate(rng.random((8, 5, 3)).astype(np.float32), axis=-1)
                  for _ in range(2))
    originals = core.copy(), full.copy()
    actual = mix_predictions(core, full, weight)
    expected = ((1 - weight) * core.astype(np.float64) + weight * full.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and (np.diff(actual, axis=-1) <= 0).all()
    assert (actual >= np.minimum(core, full)).all() and (actual <= np.maximum(core, full)).all()
    for value, original in zip((core, full), originals, strict=True):
        np.testing.assert_array_equal(value, original)
        assert not np.shares_memory(actual, value)


def test_observe_does_not_mutate_forecasts_actions_or_labels():
    core, full = forecasts(5)
    actions, outcomes = np.arange(5), labels(5)
    values = core, full, actions, outcomes
    before = [value.copy() for value in values]
    LaggedMixer().observe(*values)
    for value, original in zip(values, before, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("capacity,clip", [(0, 1e-6), (1.5, 1e-6), (True, 1e-6),
                                          (32, 0), (32, .5), (32, float("nan"))])
def test_invalid_constructor(capacity, clip):
    with pytest.raises(ValueError):
        LaggedMixer(capacity, clip)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.01, 1.01])
def test_invalid_probabilities_and_weights_leave_state_unchanged(bad):
    core, full = forecasts()
    mixer = LaggedMixer()
    before = mixer.signature()
    full[0, 0, 0] = bad
    with pytest.raises(ValueError):
        mixer.observe(core, full, np.array([2]), labels(1))
    assert mixer.signature() == before
    core, full = forecasts()
    with pytest.raises(ValueError):
        mix_predictions(core, full, bad)


@pytest.mark.parametrize("actions,outcomes", [([5], [[1, 1, 1]]), ([-1], [[1, 1, 1]]),
        ([1.0], [[1, 1, 1]]), ([1], [[0, 1, 0]]), ([1], [[1, .5, 0]]),
        ([1], [[1, 1, float("nan")]]), ([1, 2], [[1, 1, 1]]), ([1], [1])])
def test_invalid_feedback_does_not_partially_update(actions, outcomes):
    core, full = forecasts()
    mixer = LaggedMixer()
    before = mixer.signature()
    with pytest.raises(ValueError):
        mixer.observe(core, full, np.asarray(actions), np.asarray(outcomes))
    assert mixer.signature() == before


def test_float32_boundary_tolerance_clips_only_evidence_not_predictions():
    tolerance = np.finfo(np.float32).eps
    core, full = forecasts(core=-tolerance, full=1 + tolerance)
    mixer = LaggedMixer()
    mixer.observe(core, full, np.array([0]), labels(1))
    assert np.isfinite(mixer.state()["scores"]).all()
    np.testing.assert_array_equal(mix_predictions(core, full, 1), full)
