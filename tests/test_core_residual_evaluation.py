"""Complete trace arithmetic, causal forecast inputs and resumable evaluation."""

import copy
from dataclasses import replace
import hashlib
import json
import pickle
import random

import numpy as np
import pytest
import torch

from acp_cl.acquisition.study import Evaluator
from acp_cl.acquisition.world import AcquisitionWorld, Law, signal_images
from acp_cl.core_residual import evaluation
from acp_cl.core_residual.evaluation import TraceEvaluator, array_hash, recompute_metrics
from acp_cl.persistence.world import concatenate
from acp_cl.predictive_value.mechanism import clone_exact, state_signature
from acp_cl.replay_renewal.study import new_learner


@pytest.fixture
def settings():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return dict(width=8, experts=4, context_width=5, decoder_width=8,
                interaction_features=True, lr=.002, memory_packets=4, batch_size=8,
                evidence_strength=1., updates_per_batch=2, eval_size=64, support_replicates=2)


def trained(method, settings):
    learner = new_learner(method, 61, settings)
    world = AcquisitionWorld()
    for index in range(3):
        learner.train(world.experience(Law(index % 2), settings["batch_size"], 540 + index))
    return learner


def rng_signature():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
                                       torch.get_rng_state().numpy()), protocol=5)).hexdigest()


def arrays_for_metrics():
    rng = np.random.default_rng(5)
    probabilities = rng.random((4, 5, 3)).astype(np.float32)
    probabilities[:, :, -1] = .5  # Deliberate greedy ties must choose action zero.
    truth = rng.integers(0, 2, size=(4, 5, 3), dtype=np.uint8)
    affected = np.array([True, False, True, False])
    valid = np.array([False, True, True, False])
    return probabilities, truth, affected, valid


def test_numpy_metrics_use_all_actions_float64_and_first_greedy_tie():
    probabilities, truth, affected, valid = arrays_for_metrics()
    marginal = np.linspace(.1, .9, 15).reshape(5, 3)
    metrics = recompute_metrics(probabilities, truth, affected, valid, marginal)
    errors = [sum((float(probabilities[i, a, h]) - int(truth[i, a, h])) ** 2
                  for a in range(5) for h in range(3)) / 15 for i in range(4)]
    assert metrics["brier"] == pytest.approx(sum(errors) / 4, abs=1e-16)
    assert metrics["focus_brier"] == pytest.approx((errors[0] + errors[2]) / 2, abs=1e-16)
    assert metrics["valid_brier"] == pytest.approx((errors[1] + errors[2]) / 2, abs=1e-16)
    assert metrics["survival"] == float(truth[:, 0, -1].mean())
    assert metrics["focus_survival"] == float(truth[affected, 0, -1].mean())
    assert metrics["valid_survival"] == float(truth[valid, 0, -1].mean())
    assert metrics["no_transfer"] == metrics["survival"]
    assert metrics["clairvoyant_upper"] == float(truth[:, :, -1].max(axis=1).mean())
    reference = sum((float(marginal[a, h]) - int(truth[i, a, h])) ** 2
                    for i in range(4) for a in range(5) for h in range(3)) / 60
    assert metrics["marginal_brier"] == pytest.approx(reference, abs=2e-16)
    assert metrics["brier"] != float(((probabilities - truth) ** 2).mean())


def test_float32_epsilon_bounds_are_tolerated_without_clamping():
    probabilities = np.full((1, 5, 3), 1 + np.finfo(np.float32).eps, dtype=np.float32)
    truth = np.ones_like(probabilities, dtype=np.uint8)
    selected = np.ones(1, dtype=bool)
    result = recompute_metrics(probabilities, truth, selected, selected)
    assert result["brier"] == float(np.finfo(np.float32).eps ** 2)
    probabilities[:] = -np.finfo(np.float32).eps
    truth[:] = 0
    assert recompute_metrics(probabilities, truth, selected, selected)["brier"] == result["brier"]


@pytest.mark.parametrize("field,value", [
    (0, np.zeros((4, 5))), (0, np.full((4, 5, 3), np.nan)),
    (0, np.full((4, 5, 3), 1.01)), (0, np.full((4, 5, 3), -.01)),
    (0, np.ones((4, 5, 3), dtype=complex)), (0, np.ones((4, 5, 3), dtype=object)),
    (1, np.full((4, 5, 3), .5)), (1, np.zeros((3, 5, 3))),
    (2, np.ones(4, dtype=np.uint8)), (2, np.zeros(4, dtype=bool)),
    (3, np.ones(3, dtype=bool)), (3, np.zeros(4, dtype=bool)),
])
def test_malformed_metric_arrays_are_rejected(field, value):
    values = list(arrays_for_metrics())
    values[field] = value
    with pytest.raises(ValueError):
        recompute_metrics(*values)


@pytest.mark.parametrize("marginal", [np.ones(3), np.full((5, 3), np.inf),
                                    np.full((5, 3), -1), np.full((5, 3), 1.1)])
def test_malformed_marginals_are_rejected(marginal):
    with pytest.raises(ValueError):
        recompute_metrics(*arrays_for_metrics(), marginal=marginal)


@pytest.mark.parametrize("method", ("conditional", "recurrent"))
@pytest.mark.parametrize("law,cue,branch,flipped", [
    (Law(0), None, "return", False), (Law(1, (0,)), 0, "novel", False),
    (Law(1, (0, 1)), 1, "novel", True),
    (Law(0, (0, 1, 2), revised=True), 0, "revision", True),
    (Law(1, (0, 1, 2), noise=.3), 2, "noise", False),
])
def test_trace_preserves_old_metrics_up_to_float64_accumulation(method, law, cue, branch, flipped, settings):
    learner = trained(method, settings)
    reference = clone_exact(learner)
    ordinary, traced = Evaluator(61, settings), TraceEvaluator(61, settings)
    marginal = np.linspace(.2, .8, 15).reshape(5, 3)
    expected = ordinary.evaluate(reference, law, reference.history, cue, branch, flipped, marginal)
    actual = traced.evaluate(learner, law, learner.history, cue, branch, flipped, marginal)
    assert actual.pop("trace") == 0
    assert set(actual) == set(expected)
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], abs=4e-8, rel=0)
    assert traced.records[0]["metrics"] == actual
    assert traced.records[0]["law"]["noise"] == law.noise
    assert all(not name.startswith("oracle") for name in vars(learner.history))


@pytest.mark.parametrize("method", ("conditional", "recurrent"))
def test_probe_and_valid_panel_cover_every_forecast_and_preserve_full_state(method, settings):
    learner = trained(method, settings)
    learner.model.train()
    learner.model.frame.eval()
    state, rng = state_signature(learner), rng_signature()
    evaluator = TraceEvaluator(61, settings)
    law = Law(1, (0, 1))
    probe = evaluator.probe(learner, law, cue=1)
    panel = evaluator.valid_panel(learner, law)
    assert state_signature(learner) == state
    assert rng_signature() == rng
    assert set(panel) == {"0", "1"}
    assert all(set(row) == {"valid_brier", "valid_survival"} for row in panel.values())
    assert len(evaluator.records) == settings["support_replicates"] * 7
    assert [row["index"] for row in evaluator.records] == list(range(14))
    for rep, row in enumerate(probe["replicates"]):
        fresh = evaluator.support(law, rep)
        for position, point in enumerate(row["curve"]):
            trace = point["metrics"]["trace"]
            assert trace == rep * 5 + position
            feedback = point["feedback"]
            support = learner.history if not feedback else concatenate(
                learner.history, fresh.take(slice(0, feedback))).take(slice(-settings["batch_size"], None))
            assert evaluator.records[trace]["support_fingerprint"] == support.fingerprint()
            for suffix, array in (("observations", support.observations), ("actions", support.actions),
                                  ("outcomes", support.survival)):
                np.testing.assert_array_equal(evaluator.arrays[f"e{trace}_support_{suffix}"], array)
        assert row["flipped"]["trace"] == rep * 5 + 4
    assert all(not np.shares_memory(value, learner.history.observations)
               for value in evaluator.arrays.values())


def test_flipped_probe_changes_only_query_sensor_signal_and_keeps_physical_truth(settings, monkeypatch):
    learner = trained("conditional", settings)
    evaluator = TraceEvaluator(61, settings)
    law, support = Law(1, (0,)), learner.history
    calls = []
    original = evaluation.predict_all

    def observe_arguments(candidate, observations, past_support):
        assert candidate is learner and past_support is support
        assert isinstance(observations, np.ndarray)
        assert set(vars(past_support)) == {"observations", "actions", "survival"}
        calls.append(observations.copy())
        return original(candidate, observations, past_support)

    monkeypatch.setattr(evaluation, "predict_all", observe_arguments)
    evaluator.evaluate(learner, law, support, cue=0)
    evaluator.evaluate(learner, law, support, cue=0, flipped=True)
    cases, _ = evaluator.dataset(law)
    signals = cases.signals.copy()
    signals[:, 0] ^= 1
    np.testing.assert_array_equal(calls[0], cases.observations)
    np.testing.assert_array_equal(calls[1], signal_images(cases.observations, signals))
    for name in ("truth", "affected", "valid", "support_observations", "support_actions", "support_outcomes"):
        np.testing.assert_array_equal(evaluator.arrays[f"e0_{name}"], evaluator.arrays[f"e1_{name}"])
    assert evaluator.records[0]["query_sha256"] != evaluator.records[1]["query_sha256"]
    assert evaluator.records[0]["support_fingerprint"] == evaluator.records[1]["support_fingerprint"]


def test_no_support_counts_dummy_encoder_work_without_inventing_feedback(settings):
    learner = new_learner("conditional", 61, settings)
    evaluator = TraceEvaluator(61, settings)
    result = evaluator.evaluate(learner, Law(0), None)
    row = evaluator.records[result["trace"]]
    assert row["support_fingerprint"] is None
    assert row["support_presentations"] == settings["batch_size"]
    assert row["observed_support_presentations"] == 0
    assert len(evaluator.arrays) == 5


def test_forecast_exception_restores_mixed_modes_and_does_not_append_a_trace(settings, monkeypatch):
    learner = trained("conditional", settings)
    learner.model.train()
    learner.model.frame.eval()
    state, rng = state_signature(learner), rng_signature()
    evaluator = TraceEvaluator(61, settings)

    def fail(observations, support):
        learner.model.eval()
        raise RuntimeError("inference interrupted")

    monkeypatch.setattr(learner, "predict", fail)
    with pytest.raises(RuntimeError, match="inference interrupted"):
        evaluator.evaluate(learner, Law(0), learner.history)
    assert state_signature(learner) == state and rng_signature() == rng
    assert evaluator.snapshot() == {"records": [], "arrays": {}}


def assert_snapshots_equal(left, right):
    assert left["records"] == right["records"]
    assert left["arrays"].keys() == right["arrays"].keys()
    for name, values in left["arrays"].items():
        np.testing.assert_array_equal(values, right["arrays"][name])


@pytest.mark.parametrize("method", ("conditional", "recurrent"))
def test_snapshot_resume_and_export_roundtrip_are_exact(method, settings, tmp_path):
    learner = trained(method, settings)
    evaluator = TraceEvaluator(61, settings)
    law = Law(1, (0, 1, 2))
    evaluator.evaluate(learner, law, None)
    evaluator.evaluate(learner, law, learner.history, cue=2)
    payload = pickle.loads(pickle.dumps(evaluator.snapshot(), protocol=5))
    resumed = TraceEvaluator(61, settings, snapshot=payload)
    marginal = np.full((5, 3), .5)
    assert evaluator.probe(learner, law, cue=2, marginal=marginal) == resumed.probe(
        learner, law, cue=2, marginal=marginal)
    assert evaluator.valid_panel(learner, law) == resumed.valid_panel(learner, law)
    assert_snapshots_equal(evaluator.snapshot(), resumed.snapshot())
    assert evaluator.export(tmp_path / "complete") == resumed.export(tmp_path / "resumed")
    report = evaluator.export(tmp_path / "complete")
    assert report["calls"] == 16
    assert report["query_presentations"] == 16 * settings["eval_size"]
    assert report["support_presentations"] == 16 * settings["batch_size"]
    assert report["observed_support_presentations"] == 15 * settings["batch_size"]
    for key in ("json", "arrays"):
        path = tmp_path / "complete" / report[key]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report[key]["sha256"]
    record = json.loads((tmp_path / "complete" / "evaluations.json").read_text())
    with np.load(tmp_path / "complete" / "evaluations.npz", allow_pickle=False) as arrays:
        restored = TraceEvaluator(61, settings, dict(records=record["records"], arrays=dict(arrays)))
    assert_snapshots_equal(evaluator.snapshot(), restored.snapshot())
    assert not list(tmp_path.rglob("*.tmp"))


def test_snapshots_and_restored_traces_have_independent_storage(settings):
    learner = trained("conditional", settings)
    evaluator = TraceEvaluator(61, settings)
    evaluator.evaluate(learner, Law(0), learner.history)
    snapshot = evaluator.snapshot()
    restored = TraceEvaluator(61, settings, snapshot)
    snapshot["records"][0]["cue"] = 2
    snapshot["arrays"]["e0_observations"][:] = 0
    assert evaluator.records[0]["cue"] is None and restored.records[0]["cue"] is None
    assert np.any(restored.arrays["e0_observations"])
    assert not np.shares_memory(restored.arrays["e0_observations"], evaluator.arrays["e0_observations"])


@pytest.mark.parametrize("fault", ["missing", "extra", "reindex", "metric", "queryhash",
                                   "supporthash", "supportcount", "querycount", "arrayhash",
                                   "probability", "predictioncount", "mask", "actions", "marginal"])
def test_malformed_checkpoint_traces_are_rejected_even_when_modified_arrays_are_rehashed(fault, settings):
    learner = trained("conditional", settings)
    evaluator = TraceEvaluator(61, settings)
    evaluator.evaluate(learner, Law(0), learner.history)
    value = evaluator.snapshot()
    row, arrays = value["records"][0], value["arrays"]
    changed = None
    if fault == "missing":
        del arrays["e0_truth"]
    elif fault == "extra":
        arrays["unreferenced"] = np.zeros(2)
    elif fault == "reindex":
        row["index"] = 1
    elif fault == "metric":
        row["metrics"]["brier"] += .1
    elif fault == "queryhash":
        row["query_sha256"] = "0" * 64
    elif fault == "supporthash":
        row["support_fingerprint"] = "0" * 64
    elif fault == "supportcount":
        row["support_presentations"] += 1
    elif fault == "querycount":
        row["query_presentations"] -= 1
    elif fault == "arrayhash":
        arrays["e0_probabilities"][:] = 0
    elif fault == "probability":
        changed = "probabilities"
        arrays["e0_probabilities"][:] = 1.5
    elif fault == "predictioncount":
        changed = "probabilities"
        arrays["e0_probabilities"] = arrays["e0_probabilities"][:-1]
    elif fault == "mask":
        changed = "valid"
        arrays["e0_valid"][:] = False
    elif fault == "actions":
        changed = "support_actions"
        arrays["e0_support_actions"][:] = 5
    elif fault == "marginal":
        row["marginal"] = [0, 0, 0]
    if changed is not None:
        row["array_sha256"][changed] = array_hash(arrays[f"e0_{changed}"])
    with pytest.raises(ValueError):
        TraceEvaluator(61, settings, snapshot=value)


def test_invalid_probe_arguments_cannot_leave_partial_records(settings):
    learner = trained("conditional", settings)
    evaluator = TraceEvaluator(61, settings)
    for kwargs in (dict(flipped=True), dict(cue=3), dict(cue=True), dict(branch="unknown")):
        with pytest.raises(ValueError):
            evaluator.evaluate(learner, Law(0), learner.history, **kwargs)
    assert not evaluator.records and not evaluator.arrays
    with pytest.raises(ValueError, match="support budgets"):
        TraceEvaluator(61, dict(settings, batch_size=16)).evaluate(learner, Law(0), learner.history)


def test_noise_metadata_does_not_change_clean_query_truth(settings):
    learner = trained("conditional", settings)
    evaluator = TraceEvaluator(61, settings)
    law = Law(1, (0, 1))
    evaluator.evaluate(learner, law, learner.history, branch="noise")
    evaluator.evaluate(learner, replace(law, noise=.8), learner.history, branch="noise")
    assert evaluator.records[0]["law"] != evaluator.records[1]["law"]
    assert evaluator.records[0]["metrics"] == evaluator.records[1]["metrics"]
    for suffix in ("observations", "truth", "probabilities"):
        np.testing.assert_array_equal(evaluator.arrays[f"e0_{suffix}"], evaluator.arrays[f"e1_{suffix}"])


def test_export_failure_never_replaces_an_existing_npz_with_a_partial_file(settings, tmp_path, monkeypatch):
    learner = trained("conditional", settings)
    evaluator = TraceEvaluator(61, settings)
    evaluator.evaluate(learner, Law(0), None)
    first = evaluator.export(tmp_path)
    before = (tmp_path / "evaluations.npz").read_bytes()

    def fail(handle, **arrays):
        handle.write(b"partial archive")
        raise OSError("interrupted disk write")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="interrupted disk write"):
        evaluator.export(tmp_path)
    assert (tmp_path / "evaluations.npz").read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == first["arrays"]["sha256"]


def test_empty_snapshot_export_and_invalid_snapshot_shapes(settings, tmp_path):
    evaluator = TraceEvaluator(61, settings, {"records": [], "arrays": {}})
    assert evaluator.export(tmp_path)["calls"] == 0
    for invalid in ([], {}, {"records": [], "arrays": {}, "extra": 0},
                    {"records": {}, "arrays": {}}, {"records": [], "arrays": []}):
        with pytest.raises(ValueError):
            TraceEvaluator(61, settings, snapshot=copy.deepcopy(invalid))
