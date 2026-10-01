"""Endpoint and qualification tests independent of model implementation."""

import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1]/"scripts/summarize_acquisition.py"
spec = importlib.util.spec_from_file_location("acquisition_summary", path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_brier_area_weights_arrivals_and_includes_entry():
    curve = [dict(arrivals=x, metrics=dict(focus_brier=y)) for x, y in ((0, .3), (2, .1), (10, .05))]
    assert summary.area(curve, "focus_brier") == pytest.approx(.1)


def test_qualification_requires_each_cue_not_only_stage_average():
    rows = [dict(model="conditional", arm="fresh", branch="novel", stage=stage,
                 cue=(stage+seed)%3, marginal_gain=.1, cue_effect=.01)
            for seed in range(3) for stage in (1, 2, 3)]
    for row in rows:
        if row["cue"] == 0:
            row["cue_effect"] = 0.
    result = summary.qualification(rows, ["conditional"])
    assert all(result["groups"][f"conditional_stage_{i}"]["cue_use_ok"] for i in (1, 2, 3))
    assert not result["groups"]["conditional_cue_0"]["cue_use_ok"]
    assert not result["passed"]


def test_qualification_requires_acquisition_as_well_as_cue_sensitivity():
    rows = [dict(model="conditional", arm="fresh", branch="novel", stage=k+1,
                 cue=k, marginal_gain=.019, cue_effect=.1) for k in range(3)]
    assert not summary.qualification(rows, ["conditional"])["passed"]
    for r in rows:
        r["marginal_gain"] = .02
    assert summary.qualification(rows, ["conditional"])["passed"]


def test_rounding_noise_is_not_counted_as_a_seed_effect():
    result = summary.paired([1e-17, -1e-17, 0.])
    assert result["positive"] == 0
    assert result["mean"] == result["lower"] == result["upper"] == 0.
    assert result["n"] == 3
