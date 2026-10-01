"""Prospective scoring, every screen gate, and portable-record tamper rejection."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from summarize_evidence_consolidation import (
    CONTRAST_FIELDS, audit_records, block_metrics, comparisons, pilot_screen, seed_metrics, summarize,
)
from summarize_training_state import paired
from acp_cl.evidence_consolidation import study
from acp_cl.evidence_consolidation.mechanism import POLICIES
from acp_cl.evidence_consolidation.schedule import schedule


@pytest.fixture(scope="module")
def complete_run(tmp_path_factory):
    """Tiny full schedule, two architectures, and windows crossing block boundaries."""
    torch.set_num_threads(1)
    config = json.loads(Path("configs/evidence_consolidation_smoke.json").read_text())
    config.update(blocks=18, evidence_window=3, evidence_positive=2)
    directory = tmp_path_factory.mktemp("evidence_summary")/"run"
    study.run_suite(config, directory, protocol="docs/evidence_consolidation_protocol.md")
    return directory, config


def test_full_small_cohort_audit_and_portable_summary(complete_run, tmp_path):
    directory, config = complete_run
    manifest, blocks, fresh, prefixes, hashes = audit_records(directory)
    assert manifest["config"] == config
    assert len(blocks) == 36 and len(fresh) == 6 and len(prefixes) == 2
    assert len(hashes) == 44
    summary = summarize(directory, tmp_path/"portable")
    assert len(summary["block_rows"]) == 144
    assert len(summary["seed_rows"]) == 8
    assert len(summary["fresh_rows"]) == 6
    assert all(s["status"] == "INCONCLUSIVE" and not s["eligible"]
               for s in summary["screen"].values())
    assert all(s["failed"] == [] for s in summary["screen"].values())
    assert summary["window_rows"]
    assert {r["proposal_age_packets"] for r in summary["window_rows"]} == {3}
    for resource in summary["resource_rows"]:
        assert resource["shared_runner_maximum_model_copies"] == 5
        assert resource["shared_runner_scoring_prediction_passes"] == 5*resource["draft_packets"]
        assert resource["training_probability_passes"] == 2*resource["training_cost"]["optimizer_steps"]
        assert resource["standalone"]["immediate"]["maximum_model_copies"] == 1
        assert resource["standalone"]["sustained"]["maximum_model_bytes"] == 3*resource["model_bytes"]
        assert resource["standalone"]["sustained"]["maximum_evidence_gain_values"] == 3
    assert {r["label"] for r in summary["block_rows"]} >= {"noise", "recovery", "return", "revision"}
    for row in summary["seed_rows"]:
        chosen = [r for r in summary["block_rows"] if (r["seed"], r["model"], r["policy"])
                  == (row["seed"], row["model"], row["policy"])]
        assert row["overall_truth"] == pytest.approx(sum(r["truth_brier"] for r in chosen)/18)
        assert chosen[-1]["cumulative_truth_total"] == pytest.approx(row["truth_total"])
        assert chosen[-1]["cumulative_truth_mean"] == pytest.approx(row["overall_truth"])
    with gzip.open(tmp_path/"portable"/"raw_results.jsonl.gz", "rt", encoding="utf-8") as stream:
        raw = [json.loads(line) for line in stream]
    assert len(raw) == 42 and sum("arm" in r for r in raw) == 6
    archived_prefixes = json.loads((tmp_path/"portable"/"prefix_records.json").read_text())
    originals = {f"{r['model']}_{r['seed']}/block_{r['block']['index']:02}/result.json": r
                 for r in raw if "block" in r}
    originals.update({f"{r['model']}_{r['seed']}/fresh_{r['stage']}/result.json": r
                      for r in raw if "arm" in r})
    originals.update({f"{key}/prefix.json": value for key, value in archived_prefixes.items()})
    for row in hashes:
        content = (json.dumps(originals[row["path"]], indent=2, allow_nan=False)+"\n").encode()
        assert hashlib.sha256(content).hexdigest() == row["sha256"]
    for name in ("manifest.json", "completion.json", "training_source.zip", "protocol_lock.json", "protocol_at_lock.md"):
        assert (tmp_path/"portable"/name).read_bytes() == (directory/name).read_bytes()


def test_three_block_smoke_has_unavailable_late_endpoints(tmp_path):
    torch.set_num_threads(1)
    config = json.loads(Path("configs/evidence_consolidation_smoke.json").read_text())
    config["models"] = ["conditional"]
    directory = tmp_path/"smoke"
    study.run_suite(config, directory)
    result = summarize(directory, tmp_path/"summary")
    assert len(result["block_rows"]) == 12 and len(result["fresh_rows"]) == 3
    assert all(r["late_truth"] is None and r["early_truth"] is None for r in result["seed_rows"])
    screen = result["screen"]["conditional"]
    assert screen["status"] == "INCONCLUSIVE" and not screen["passed"]
    assert screen["gates"]["immediate_late_truth_gain"]["passed"] is None
    assert result["contrasts"]["conditional"]["sustained_minus_immediate"]["late_minus_early_advantage"] is None
    assert "unavailable" in (tmp_path/"summary"/"summary.md").read_text()


def test_score_definitions_keep_packet_truth_separate_from_query_auc():
    policy = "sustained"
    record = dict(
        prequential=[dict(truth_brier={policy: .1}, reported_brier={policy: .3}),
                     dict(truth_brier={policy: .3}, reported_brier={policy: .7})],
        curve=[dict(arrivals=x, policies={policy: dict(focus_brier=f, brier=b, survival=s)})
               for x, f, b, s in ((0, .9, .6, .2), (8, .5, .4, .6), (16, .1, .2, 1.))],
        valid_before={policy: {"0": dict(valid_brier=.4), "1": dict(valid_brier=.6)}},
        valid_after={policy: {"0": dict(valid_brier=.2), "1": dict(valid_brier=.4)}},
        knowledge_after=[dict(policies={policy: .3}), dict(policies={policy: .9})],
        start_state=dict(adoptions={policy: 1}), end_state=dict(adoptions={policy: 3}), decisions=[{}, {}])
    row = block_metrics(record, policy)
    assert row["truth_brier"] == pytest.approx(.2)
    assert row["reported_brier"] == pytest.approx(.5)
    assert row["focus_auc"] == pytest.approx(.5)
    assert row["brier_auc"] == pytest.approx(.4)
    assert row["survival_auc"] == pytest.approx(.6)
    assert row["valid_after"] == pytest.approx(.3)
    assert row["valid_damage"] == pytest.approx(-.2)
    assert row["knowledge_after"] == pytest.approx(.6)
    assert row["adoptions"] == 2


def test_phase_matched_cycles_and_seed_aggregation():
    rows = [dict(block=b.index, label=b.label, truth_brier=b.index/100,
                 reported_brier=.5, focus_auc=.1+b.index/100, survival_auc=.8,
                 valid_after=.2+b.index/100, valid_damage=.01, knowledge_after=b.index/100,
                 adoptions=1) for b in schedule(29)]
    result = seed_metrics(rows, dict(episode_size=8192))
    for field, expected in (("overall_truth", .085), ("early_truth", .05), ("late_truth", .15),
                            ("intro_focus_auc", .11), ("return_truth", .11), ("noise_truth", .09),
                            ("revision_truth", .12), ("recovery_truth", .10),
                            ("valid_after", .285), ("knowledge_after", .17), ("knowledge_mean", .085)):
        assert result[field] == pytest.approx(expected)
    assert result["truth_total"] == pytest.approx(.085*18*8192*3)
    assert result["adoptions"] == 18
    config = dict(seeds=list(range(6)), models=["conditional"])
    seed_rows = []
    for seed in config["seeds"]:
        for policy in POLICIES:
            values = {f: .2 for f in CONTRAST_FIELDS}
            values.update(early_truth=.2, late_truth=.2)
            if policy == "sustained":
                values.update(early_truth=.197, late_truth=.194)
            seed_rows.append(dict(seed=seed, model="conditional", policy=policy, **values))
    compared = comparisons(seed_rows, config)["conditional"]["sustained_minus_immediate"]
    assert compared["early_truth"]["mean"] == pytest.approx(-.003)
    assert compared["late_truth"]["mean"] == pytest.approx(-.006)
    assert compared["late_minus_early_advantage"]["mean"] == pytest.approx(.003)
    assert compared["late_truth"]["n"] == 6


def gate_inputs(n=6):
    values = {f: paired([0.]*n) for f in CONTRAST_FIELDS}
    for field in ("overall_truth", "late_truth"):
        values[field] = paired([-.003]*n)
    contrasts = {f"sustained_minus_{control}": copy.deepcopy(values)
                 for control in ("immediate", "periodic", "single")}
    return contrasts, dict(kind="diagnostic", seeds=list(range(n)), blocks=18)


@pytest.mark.parametrize("control,field,bad", [
    ("immediate", "overall_truth", -.001), ("periodic", "overall_truth", -.001),
    ("immediate", "late_truth", -.001), ("periodic", "late_truth", -.001),
    ("immediate", "intro_focus_auc", .006), ("immediate", "return_truth", .006),
    ("immediate", "noise_truth", .006), ("immediate", "valid_after", .006),
    ("immediate", "survival_auc", -.011),
])
def test_every_prospective_mean_gate(control, field, bad):
    contrasts, config = gate_inputs()
    assert pilot_screen(contrasts, True, config)["passed"]
    contrasts[f"sustained_minus_{control}"][field] = paired([bad]*6)
    result = pilot_screen(contrasts, True, config)
    assert result["eligible"] and result["status"] == "FAIL" and len(result["failed"]) == 1
    assert not result["passed"]


@pytest.mark.parametrize("control", ("immediate", "periodic"))
def test_overall_consistency_is_five_of_six_and_fresh_qualification_required(control):
    contrasts, config = gate_inputs()
    contrasts[f"sustained_minus_{control}"]["overall_truth"] = paired([-.01]*4+[.001]*2)
    result = pilot_screen(contrasts, True, config)
    assert result["failed"] == [f"{control}_overall_consistency"]
    contrasts, config = gate_inputs()
    assert pilot_screen(contrasts, False, config)["failed"] == ["fresh_qualification"]


def test_screen_boundaries_warning_and_single_is_a_comparison_not_a_gate():
    contrasts, config = gate_inputs()
    for control in ("immediate", "periodic"):
        for field in ("overall_truth", "late_truth"):
            contrasts[f"sustained_minus_{control}"][field].update(mean=-.002, upper=-.001)
    immediate = contrasts["sustained_minus_immediate"]
    for field in ("intro_focus_auc", "return_truth", "noise_truth", "valid_after"):
        immediate[field].update(mean=.005, lower=-.01, upper=.006)
    immediate["survival_auc"].update(mean=-.01, lower=-.02, upper=0.)
    contrasts["sustained_minus_single"] = {f: paired([.5]*6) for f in CONTRAST_FIELDS}
    result = pilot_screen(contrasts, True, config)
    assert result["passed"]
    assert result["gates"]["immediate_overall_truth_gain"]["uncertainty_crosses_limit"]
    assert result["gates"]["immediate_valid_after_guardrail"]["uncertainty_crosses_limit"]
    assert result["gates"]["immediate_survival_guardrail"]["uncertainty_crosses_limit"]
    contrasts, config = gate_inputs(2)
    config["kind"] = "development"
    result = pilot_screen(contrasts, True, config)
    assert result["status"] == "INCONCLUSIVE" and not result["passed"] and not result["eligible"]
    assert result["gates"]["immediate_overall_consistency"]["passed"] is None


def copy_records(source, output):
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".zip", ".md"):
            target = output/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


@pytest.mark.parametrize("tamper", (
    "missing", "extra", "prefix_missing", "identity", "law", "cost", "counter", "policy",
    "nan", "pending", "acceptance", "fresh_state", "fresh_data", "architecture_data",
    "completion_jobs", "completion_count", "source", "protocol", "protocol_missing", "config",
))
def test_corrupt_or_incomplete_portable_records_are_rejected(complete_run, tmp_path, tamper):
    source, _ = complete_run
    directory = tmp_path/"tampered"
    copy_records(source, directory)
    path = directory/"conditional_29"/"block_00"/"result.json"
    record = json.loads(path.read_text())
    if tamper == "missing":
        path.unlink()
    elif tamper == "extra":
        extra = directory/"unplanned"/"result.json"
        extra.parent.mkdir()
        shutil.copyfile(path, extra)
    elif tamper == "prefix_missing":
        (directory/"conditional_29"/"prefix.json").unlink()
    elif tamper == "source":
        with zipfile.ZipFile(directory/"training_source.zip", "w") as archive:
            archive.writestr("unexpected.py", "pass\n")
    elif tamper == "protocol":
        (directory/"protocol_at_lock.md").write_text("revised after results\n")
    elif tamper == "protocol_missing":
        (directory/"protocol_lock.json").unlink()
    else:
        if tamper == "identity":
            record["identity"]["config_sha256"] = "a"*64
        elif tamper == "law":
            record["block"]["law"]["mode"] ^= 1
        elif tamper == "cost":
            record["end_state"]["draft"]["cost"]["optimizer_steps"] += 1
        elif tamper == "counter":
            record["end_state"]["batches"] += 1
        elif tamper == "policy":
            del record["prequential"][0]["truth_brier"]["single"]
        elif tamper == "nan":
            record["prequential"][0]["truth_brier"]["single"] = float("nan")
        elif tamper == "pending":
            path = directory/"conditional_29"/"block_01"/"result.json"
            record = json.loads(path.read_text())
            record["end_state"]["gains"]["sustained"].append(.01)
        elif tamper == "acceptance":
            record["decisions"][0]["accepted"]["sustained"] ^= True
        elif tamper in ("fresh_state", "fresh_data"):
            path = directory/"conditional_29"/"fresh_1"/"result.json"
            record = json.loads(path.read_text())
            if tamper == "fresh_state":
                record["start_diagnostics"]["optimizer_bytes"] = 100
            else:
                record["batch_sha256"][0] = "a"*64
        elif tamper == "architecture_data":
            path = directory/"recurrent_29"/"block_04"/"result.json"
            record = json.loads(path.read_text())
            record["batch_sha256"][0] = "a"*64
        elif tamper in ("completion_jobs", "completion_count"):
            path = directory/"completion.json"
            record = json.loads(path.read_text())
            if tamper == "completion_jobs":
                record["jobs"][1] = record["jobs"][0]
            else:
                record["blocks"] += 1
        elif tamper == "config":
            path = directory/"manifest.json"
            record = json.loads(path.read_text())
            record["config"]["evidence_positive"] = 100
        path.write_text(json.dumps(record, indent=2)+"\n")
    with pytest.raises(ValueError):
        audit_records(directory)
