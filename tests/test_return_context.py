"""Matched support replacement, exact contrasts, read-only probes and provenance."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import diagnose_return_context as diagnostic
from diagnose_return_context import (array_hash, best_single_head, build_contrasts,
    bundle_signature, check_files, file_entry, paired_summary, probe, replace_history,
    score_predictions, snapshot_code, PREDICTION_ROUNDOFF)
from acp_cl.acquisition.world import AcquisitionWorld, Law
from acp_cl.evidence_consolidation.mechanism import Consolidation
from acp_cl.persistence.world import Experience
from acp_cl.replay_renewal.study import new_learner


def records(first=0):
    values = np.arange(first, first+32, dtype=np.uint8)
    return Experience(values[:, None, None, None], values % 5, np.tile(values[:, None], (1, 3)))


@pytest.mark.parametrize("count", (0, 8, 16, 32))
def test_replacement_uses_latest_old_and_first_new_without_mutation(count):
    old, new = records(), records(32)
    original = old.fingerprint(), new.fingerprint()
    result = replace_history(old, new, count)
    expected = np.concatenate((np.arange(count, 32), np.arange(32, 32+count)))
    np.testing.assert_array_equal(result.observations[:, 0, 0, 0], expected)
    assert len(result) == 32
    assert (old.fingerprint(), new.fingerprint()) == original


@pytest.mark.parametrize("count", (-1, 33, 1.5, True))
def test_invalid_replacement_is_rejected(count):
    with pytest.raises(ValueError, match="replacement count"):
        replace_history(records(), records(32), count)


def test_all_action_brier_and_chosen_action_survival_are_distinct():
    truth = np.zeros((2, 5, 3), dtype=float)
    truth[:, 2] = 1
    predictions = np.full_like(truth, .25)
    predictions[:, 2] = .75
    score = score_predictions(predictions, truth)
    assert score == dict(brier=.0625, survival=1.)
    predictions[:, 1] = .9
    assert score_predictions(predictions, truth)["survival"] == 0.
    with pytest.raises(ValueError, match="matched query"):
        score_predictions(predictions[:, 0], truth[:, 0])


@pytest.mark.parametrize("boundary,delta", ((1., 1.), (0., -1.)))
def test_float32_endpoint_roundoff_is_scored_without_clipping(boundary, delta):
    epsilon = np.finfo(np.float32).eps
    predictions = np.full((2, 5, 3), boundary+delta*epsilon, dtype=np.float32)
    truth = np.full_like(predictions, boundary)
    unchanged = predictions.copy()
    result = score_predictions(predictions, truth)
    assert result["brier"] == float(np.mean((unchanged-truth)**2))
    assert result["brier"] == float(epsilon**2) and result["brier"] > 0
    np.testing.assert_array_equal(predictions, unchanged)


@pytest.mark.parametrize("invalid", (-2*PREDICTION_ROUNDOFF, 1+2*PREDICTION_ROUNDOFF))
def test_larger_probability_range_violations_are_rejected_with_actual_extrema(invalid):
    predictions = np.full((2, 5, 3), invalid)
    with pytest.raises(ValueError, match="prediction exceeds.*min=.*max=.*tolerance="):
        score_predictions(predictions, np.zeros_like(predictions))


@pytest.mark.parametrize("invalid", (-np.finfo(np.float32).eps, 1+np.finfo(np.float32).eps))
def test_outcome_range_remains_strict_even_within_prediction_roundoff_tolerance(invalid):
    truth = np.full((2, 5, 3), invalid)
    with pytest.raises(ValueError, match=r"outcome is outside \[0, 1\]: min=.*max="):
        score_predictions(np.zeros_like(truth), truth)


def test_best_single_head_is_not_a_lower_bound_on_mixture_error():
    truth = np.zeros((2, 5, 3))
    truth[1] = 1
    heads = np.stack((np.zeros_like(truth), np.ones_like(truth)), axis=1)
    selected = best_single_head(heads, truth)
    mixture = score_predictions(heads.mean(axis=1), truth)
    assert selected["brier"] == .5 and mixture["brier"] == .25
    assert selected["query_selected"] and not selected["deployable"]


@pytest.mark.parametrize("model", ("conditional", "recurrent"))
def test_real_predictor_probe_preserves_complete_bundle_state(model):
    torch.set_num_threads(1)
    settings = dict(width=4, experts=4, context_width=3, decoder_width=4,
        interaction_features=True, lr=.002, memory_packets=2, batch_size=32,
        evidence_strength=1., updates_per_batch=1, evidence_window=2,
        evidence_margin=.002, evidence_positive=1)
    learner = new_learner(model, 37, settings)
    world = AcquisitionWorld()
    learner.history = world.experience(Law(0), 32, 31)
    bundle = Consolidation(learner, settings)
    packet = world.experience(Law(0), 32, 39)
    predictions, proposed = bundle.prepare(packet.observations)
    bundle.observe(packet, predictions, proposed)
    assert learner.optimizer.state and learner.memory.ids == [0]
    assert any(parameter.grad is not None for parameter in learner.model.parameters())
    assert bundle.committed["periodic"] is bundle.committed["sustained"]
    cases = world.dataset(Law(0), 64, 45)
    truth = world.counterfactuals(cases)
    learner.model.train()
    learner.model.temporal.eval()  # Mixed child flags must also survive predict().
    before = bundle_signature(bundle)
    for predictor in (learner, bundle.committed["sustained"]):
        score, heads = probe(predictor, cases.observations, truth, learner.history,
                             include_heads=model == "conditional")
        assert bundle_signature(bundle) == before
        assert 0 <= score["brier"] <= 1 and 0 <= score["survival"] <= 1
        assert -PREDICTION_ROUNDOFF <= score["prediction_min"] <= score["prediction_max"] <= 1+PREDICTION_ROUNDOFF
        assert score["prediction_range_tolerance"] == PREDICTION_ROUNDOFF
        assert (heads is not None) == (model == "conditional")


def test_probe_rejects_parameter_mutation_and_restores_training_flag():
    class Mutating:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)

        def predict(self, observations, support):
            self.model.eval()
            with torch.no_grad():
                self.model.weight.add_(1.)
            values = np.zeros((len(observations), 5, 3))
            return values, None, None

    learner = Mutating()
    with pytest.raises(AssertionError, match="changed weights"):
        probe(learner, np.zeros((2, 1)), np.zeros((2, 5, 3)), records())
    assert learner.model.training


def synthetic_rows():
    rows = []
    for seed, offset in ((1, 0.), (2, .01)):
        for block, cycle_offset in ((6, 0.), (11, .03), (16, .09)):
            for weight, base in (("start_draft", .1), ("start_sustained", .15),
                                 ("prefix_draft", .08), ("start_periodic", .12),
                                 ("end_draft", .07), ("previous_return_draft", .06)):
                if weight == "previous_return_draft" and block == 6:
                    continue
                for support in ("actual_start", "observed_8", "observed_16", "observed_32", "fresh_target"):
                    penalty = (.08 if weight == "start_sustained" else .04) if support == "actual_start" else 0.
                    replicates = [(0, -.01), (1, .01)] if support == "fresh_target" else [(None, 0.)]
                    for replicate, variation in replicates:
                        rows.append(dict(model="conditional", seed=seed, block=block,
                            weight=weight, support=support, replicate=replicate,
                            brier=base+offset+cycle_offset+penalty+variation, survival=.8-base-penalty))
    return rows


def test_matched_decomposition_and_seed_aggregation_do_not_pseudoreplicate():
    result = build_contrasts(synthetic_rows())
    actual = result["contrasts"]["conditional"]["sustained_minus_start_draft.actual_start"]["brier"]
    assert actual["n"] == 2
    assert actual["mean"] == pytest.approx(.09)
    decomposition = result["decomposition"]["conditional"]["fresh_target"]["brier"]
    assert decomposition["target_gap"]["mean"] == pytest.approx(.05)
    assert decomposition["context_interaction"]["mean"] == pytest.approx(.04)
    assert abs(decomposition["identity_residual"]["mean"]) < 1e-12
    means = [row for row in result["seed_means"] if row["weight"] == "start_draft"
             and row["support"] == "fresh_target" and row["seed"] == 1]
    assert means[0]["returns"] == 3 and means[0]["brier"] == pytest.approx(.14)
    prior = [row for row in result["seed_contrasts"] if row["contrast"] == "start_draft_minus_previous_return.fresh_target"]
    assert all(row["returns"] == 2 and row["brier"] == pytest.approx(.04) for row in prior)
    end_prior = result["contrasts"]["conditional"]["end_draft_minus_previous_return.fresh_target"]["brier"]
    end_prefix = result["contrasts"]["conditional"]["end_draft_minus_prefix.fresh_target"]["brier"]
    assert end_prior["mean"] == pytest.approx(.01) and end_prefix["mean"] == pytest.approx(-.01)
    partial = result["contrasts"]["conditional"]["actual_start_minus_observed_8.start_draft"]["brier"]
    assert partial["n"] == 2 and partial["mean"] == pytest.approx(.04)
    duplicate = synthetic_rows()
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate"):
        build_contrasts(duplicate)


def test_paired_bootstrap_reproducible_seed_unit_and_invalid_values():
    result = paired_summary([-.2, .1, .4], [11, 12, 13])
    assert result == paired_summary([-.2, .1, .4], [11, 12, 13])
    assert result["n"] == 3 and result["positive"] == 2 and result["negative"] == 1
    with pytest.raises(ValueError, match="one value per independent seed"):
        paired_summary([1., 2.], [1, 1])
    with pytest.raises(ValueError, match="finite nonempty"):
        paired_summary([np.nan])


def test_input_integrity_rejects_changed_file_and_escape(tmp_path):
    source = tmp_path/"checkpoint.pt"
    source.write_bytes(b"original checkpoint")
    expected = {source.name: file_entry(source)}
    check_files(tmp_path, expected)
    source.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="input file changed"):
        check_files(tmp_path, expected)
    with pytest.raises(ValueError, match="escaped"):
        check_files(tmp_path, {"../outside": dict(bytes=0, sha256="none")})


def test_code_and_protocol_snapshot_rejects_postlock_changes(tmp_path):
    script, protocol = tmp_path/"script.py", tmp_path/"protocol.md"
    script.write_text("pass\n")
    protocol.write_text("Retrospective plan.\n")
    files = {"script.py": script, "protocol.md": protocol}
    output = tmp_path/"locked"
    first = snapshot_code(output, files, dict(identity="test"))
    assert first["retrospective"]
    assert snapshot_code(output, files, dict(identity="test")) == first
    protocol.write_text("Changed after analysis lock.\n")
    with pytest.raises(ValueError, match="changed after lock"):
        snapshot_code(output, files, dict(identity="test"))


def test_prepare_verifies_protocol_outside_older_pt_json_zip_integrity_scan(tmp_path, monkeypatch):
    """Exercise preflight/lock only with dummy checkpoints; never load or score them."""
    run, archive, output = tmp_path/"run", tmp_path/"archive", tmp_path/"diagnostic"
    run.mkdir()
    (archive/"pilot").mkdir(parents=True)
    config = json.loads(Path("configs/evidence_consolidation_pilot.json").read_text())
    sources = diagnostic.source_manifest()
    manifest = dict(config=config, source_files=sources,
                    identity=dict(source_sha256=diagnostic.digest(sources)))
    for path in (run/"manifest.json", archive/"pilot/manifest.json"):
        path.write_text(json.dumps(manifest))
    for model in config["models"]:
        for seed in config["seeds"]:
            folder = run/f"{model}_{seed}"
            folder.mkdir()
            for name in ("prefix.pt", "prefix.json"):
                (folder/name).write_bytes(b"synthetic input; not a pickle")
            for block in diagnostic.RETURNS:
                target = folder/f"block_{block:02}"
                target.mkdir()
                for name in ("before.pt", "checkpoint.pt", "result.json"):
                    (target/name).write_bytes(b"synthetic input; not a pickle")
    for name in ("completion.json", "training_source.zip", "protocol_lock.json"):
        (run/name).write_bytes(b"synthetic metadata")
    protocol_text = "Original sealed protocol.\n"
    (run/"protocol_at_lock.md").write_text(protocol_text)
    (archive/"pilot/protocol_at_lock.md").write_text(protocol_text)
    # The original integrity utility deliberately enumerated PT/JSON/ZIP only.
    files = [dict(path=path.relative_to(run).as_posix(), **file_entry(path), valid=True)
             for path in run.rglob("*") if path.is_file() and path.suffix != ".md"]
    (archive/"pilot/file_integrity.json").write_text(json.dumps(dict(passed=True, files=files)))
    (archive/"artifact_manifest.json").write_text(json.dumps({
        "pilot/protocol_at_lock.md": file_entry(archive/"pilot/protocol_at_lock.md")}))
    protocol = tmp_path/"new_protocol.md"
    protocol.write_text("Retrospective diagnostic protocol.\n")
    monkeypatch.setattr(diagnostic, "verify_archive", lambda path: dict(passed=True, artifacts=1))
    prepared = diagnostic.prepare(run, archive, output, protocol)
    assert prepared[4]["protocol_at_lock.md"] == file_entry(run/"protocol_at_lock.md")
    assert (output/"analysis_lock.json").exists() and not (output/"summary.json").exists()
    (run/"protocol_at_lock.md").write_text("Tampered after the archived run.\n")
    with pytest.raises(ValueError, match="input file changed"):
        diagnostic.prepare(run, archive, output, protocol)


def test_query_hash_includes_shape_dtype_and_truth():
    observations = np.arange(12, dtype=np.uint8).reshape(2, 6)
    assert array_hash(observations) != array_hash(observations.reshape(3, 4))
    assert array_hash(observations) != array_hash(observations.astype(np.int64))
    assert array_hash(observations, np.zeros(2)) != array_hash(observations, np.ones(2))
