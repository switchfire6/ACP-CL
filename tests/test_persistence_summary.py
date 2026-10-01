"""Check study endpoint semantics independently of any favorable outcome."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from acp_cl.persistence.world import COMPOSITIONS, KNOWN, schedule


spec = importlib.util.spec_from_file_location(
    "persistence_summary", Path(__file__).resolve().parents[1] / "scripts" / "summarize_persistence.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def score(value):
    return {"survival": {"12": value}, "no_transfer": .2}


def example(stream):
    regimes, info = schedule(19, stream)
    blocks = [{"regime": {"name": regime.name}, "curve": [
        {"arrivals": 0, "metrics": score(.3)}, {"arrivals": 100, "metrics": score(.7)}]}
              for regime in regimes]
    panel = {r.name: score(.4) for r in (*KNOWN, *COMPOSITIONS)}
    record = {"schedule": stream, "schedule_info": info, "blocks": blocks,
              "panels": [{"after_block": 3, "metrics": copy.deepcopy(panel)},
                         {"after_block": 11, "metrics": copy.deepcopy(panel)}],
              "final_temporal_control": {r.name: score(.1) for r in KNOWN},
              "elapsed_seconds": 2,
              "diagnostics": {"cost": {"training_seconds": 1, "selection_seconds": .1}}}
    scratch = {"curve": copy.deepcopy(blocks[-1]["curve"])}
    return record, scratch


def test_entry_and_auc_do_not_substitute_end_of_learning():
    record, scratch = example("long")
    metrics = summary.metrics(record, scratch)
    assert metrics["entry_survival"] == .3
    assert metrics["late_auc"] == .5
    assert metrics["late_final"] == .7
    assert metrics["late_vs_scratch_auc"] == 0
    assert metrics["forgetting"] == pytest.approx(.4)


def test_no_return_composition_excludes_the_now_trained_combination():
    record, scratch = example("no_return")
    trained_name = record["blocks"][10]["regime"]["name"]
    record["panels"][-1]["metrics"][trained_name] = score(1)
    metrics = summary.metrics(record, scratch)
    assert metrics["composition_final"] == .4
    assert metrics["forgetting"] is None
    assert metrics["recovered"] is None


def test_reversal_does_not_count_an_obsolete_rule_as_required_retention():
    record, scratch = example("reversal")
    invalid = record["schedule_info"]["target"]
    record["panels"][-1]["metrics"][invalid] = score(0)
    assert summary.metrics(record, scratch)["known_final"] == .4


def test_nonrecovery_is_censored_not_dropped():
    record, scratch = example("long")
    record["blocks"][10]["curve"][-1]["metrics"] = score(.5)
    metrics = summary.metrics(record, scratch)
    assert not metrics["recovered"]
    assert metrics["recovery_arrivals_capped"] == 100


def test_paired_bootstrap_keeps_the_seed_as_the_unit():
    result = summary.paired_interval([.02] * 8, repetitions=100)
    assert result["mean"] == pytest.approx(.02)
    assert result["lower"] == pytest.approx(.02)
    assert result["upper"] == pytest.approx(.02)
    assert result["positive"] == 8
    mixed = summary.paired_interval([-.1, .1], repetitions=1000)
    assert mixed["mean"] == 0
    assert mixed["positive"] == 1
    assert np.isfinite([mixed["lower"], mixed["upper"]]).all()


def test_floating_point_tie_cannot_count_as_an_improvement():
    assert summary.mean([.1 + .2 - .3]) == 0
    result = summary.paired_interval([0, 1e-17, -1e-17], repetitions=20)
    assert result["positive"] == 0
    assert result["mean"] == 0
