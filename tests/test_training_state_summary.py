"""Factorial signs, interactions, replication units and stricter development gate."""

import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1]/"scripts/summarize_training_state.py"
spec = importlib.util.spec_from_file_location("training_state_summary", path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_known_factorial_effects_and_interaction():
    values = dict(continue_=10, optimizer_reset=8, replay_reset=7, state_reset=4)
    values["continue"] = values.pop("continue_")
    expected = dict(optimizer_with_replay_kept=-2, optimizer_with_replay_reset=-3,
        replay_with_optimizer_kept=-3, replay_with_optimizer_reset=-4,
        optimizer_main=-2.5, replay_main=-3.5, interaction=-1, both_minus_continue=-6)
    for name, value in expected.items():
        assert summary.contrast(values, summary.CONTRASTS[name]) == value


def test_additive_effects_have_zero_interaction():
    values = {"continue": .20, "optimizer_reset": .18, "replay_reset": .16, "state_reset": .14}
    assert summary.contrast(values, summary.CONTRASTS["interaction"]) == pytest.approx(0.)


def test_three_episodes_do_not_triple_the_number_of_replicates():
    raw = {}
    for seed in (1, 2):
        for stage in (1, 2, 3):
            for arm, scale in zip(summary.ARMS, (10, 8, 7, 4, 15)):
                raw[seed, "conditional", stage, arm, "novel"] = dict.fromkeys(
                    ("brier_auc", "valid_damage", "after", "survival_auc", "valid_after", "cue_effect"), seed*stage*scale)
    result = summary.factorial(raw, dict(models=["conditional"], seeds=[1, 2]), "novel")
    r = result["conditional"]["optimizer_main"]["brier_auc"]
    assert r["n"] == 2 and r["differences"] == [-5., -10.]


def test_development_margin_does_not_change_comparison_threshold():
    rows = [dict(model="conditional", arm="fresh", branch="novel", stage=k+1,
                 cue=k, marginal_gain=.1, cue_effect=.0024) for k in range(3)]
    assert not summary.qualification(rows, ["conditional"], "development")["passed"]
    assert summary.qualification(rows, ["conditional"], "diagnostic")["passed"]
    rows[1]["cue_effect"] = .0019
    assert not summary.qualification(rows, ["conditional"], "diagnostic")["passed"]
