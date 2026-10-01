"""Scientific analysis contracts: endpoints, paired units, completeness, and gates."""

import importlib.util
import json
from pathlib import Path

import pytest

from acp_cl.contextual.learner import METHODS
from acp_cl.contextual.study import run_suite


path = Path(__file__).resolve().parents[1]/"scripts/summarize_contextual.py"
spec = importlib.util.spec_from_file_location("contextual_summary", path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_acquisition_integrates_actual_arrival_spacing_including_entry():
    curve = [dict(arrivals=x, metrics=dict(survival=y)) for x, y in ((0, .2), (2, .8), (10, .6))]
    assert summary.auc(curve) == pytest.approx(.66)


def test_two_gap_orders_are_averaged_within_seed_before_bootstrap():
    config = dict(seeds=[1, 2], gaps=["short", "long"], methods=["a", "b"], branches=["novel"])
    raw = {(s, g, m, "novel"): {"auc": value}
           for s in config["seeds"] for g in config["gaps"]
           for m, value in (("a", float(s)+(0 if g == "short" else 2)), ("b", 0.))}
    values, aggregate = summary.aggregate_seeds(raw, config)
    estimate = summary.paired([values[s, "a", "novel"]["auc"]-values[s, "b", "novel"]["auc"] for s in config["seeds"]])
    assert estimate["n"] == 2
    assert estimate["differences"] == [2., 3.]
    assert aggregate["novel"]["a"]["auc"] == 2.5


def test_recurrent_policy_acquisition_does_not_satisfy_history_gate():
    r = dict(initial_gain=.1, before=.8, before_no_transfer=.6, before_history_effect=0.)
    result = summary.adequacy_checks(r, dict(initial_gain=.1))
    assert not result["recurrent_uses_history"]
    assert sum(result.values()) == 3


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("contextual_analysis")
    config = dict(study="analysis_smoke", seeds=[19], gaps=["short", "long"],
        branches=["return", "novel", "noise"], methods=list(METHODS),
        block_size=32, batch_size=8, eval_size=8, probe_every=8, support_replicates=1,
        width=12, experts=4, context_width=8, decoder_width=12, lr=.002,
        memory_packets=3, updates_per_batch=1, evidence_strength=1., feedback_noise=.2,
        interaction_features=True, threads=1, workers=1)
    protocol = folder/"engineering.md"
    protocol.write_text("Engineering fixture, not a scientific result.\n")
    run_suite(config, folder/"runs", protocol=protocol)
    return folder


def test_complete_smoke_can_be_audited_but_cannot_establish_superiority(completed):
    result = summary.summarize(completed/"runs", completed/"summary")
    assert result["prefixes"] == 12
    assert result["branches"] == 36
    assert result["primary"]["n"] == 1
    assert result["decision"] == "inconclusive"


def test_missing_branch_rejected(completed):
    path = next((completed/"runs").glob("*/result.json"))
    hidden = path.with_suffix(".hidden")
    path.rename(hidden)
    try:
        with pytest.raises(ValueError, match="missing"):
            summary.audit_records(completed/"runs")
    finally:
        hidden.rename(path)


def test_identity_tampering_rejected(completed):
    path = completed/"runs/protocol_lock.json"
    saved = path.read_bytes()
    changed = json.loads(saved)
    changed["config_sha256"] = "invalid"
    path.write_text(json.dumps(changed))
    try:
        with pytest.raises(ValueError, match="mismatch"):
            summary.audit_records(completed/"runs")
    finally:
        path.write_bytes(saved)


def test_numerical_ties_are_not_counted_as_positive():
    estimate = summary.paired([1e-17, -1e-17])
    assert estimate["positive"] == 0
    assert estimate["mean"] == estimate["lower"] == estimate["upper"] == 0.
