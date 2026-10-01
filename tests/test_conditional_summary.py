"""Analysis checks for the predeclared endpoint and valid retention masks."""

import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_conditional.py"
spec = importlib.util.spec_from_file_location("conditional_summary", path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def probe(value):
    fields = dict(survival=value, no_transfer=.4, clairvoyant_upper=.9,
                  brier=.1, entropy=.5, disagreement=.2)
    return dict(replicates=[dict(curve=[dict(feedback=0, metrics=dict(fields, survival=.1)),
                                          dict(feedback=32, metrics=fields)],
                                  opposite=dict(fields, survival=value - .2),
                                  erased=dict(fields, survival=.3),
                                  broken_binding=dict(fields, survival=.35))])


def record(stream):
    curve = [dict(arrivals=0, metrics=dict(survival=.4, no_transfer=.3)),
             dict(arrivals=1024, metrics=dict(survival=.6, no_transfer=.3))]
    return dict(seed=3001, schedule=stream, return_probe=probe(.7),
                final_panel={f"mode{m}_gate{g}": probe(.5 + .2*m) for m in (0, 1) for g in (0, 1)},
                blocks=[dict(curve=curve) for _ in range(12)],
                diagnostics=dict(cost=dict(training_seconds=1.)), elapsed_seconds=2.)


def test_primary_uses_frozen_probe_after_feedback_not_online_learning():
    values = summary.metrics(record("recurring"))
    assert values["return_survival"] == .7
    assert values["return_entry"] == .1
    assert values["correct_minus_opposite"] == pytest.approx(.2)
    assert values["late_auc"] == pytest.approx(.5)
    assert values["final_known"] == pytest.approx(.6)


def test_stable_retention_excludes_never_seen_mode():
    assert summary.metrics(record("stable"))["final_known"] == pytest.approx(.7)


def test_unpredictable_does_not_report_persistent_mode_panels_as_retention():
    values = summary.metrics(record("unpredictable"))
    assert values["final_known"] is None
    assert values["final_gate"] is None
    assert values["final_online"] == .6


def test_ties_and_pairing_do_not_become_positive_from_rounding():
    result = summary.paired([1e-17, -1e-17, 0., 0.])
    assert result["positive"] == 0
    assert result["mean"] == result["lower"] == result["upper"] == 0
    assert summary.paired([.1, .1, .1])["lower"] == pytest.approx(.1)
