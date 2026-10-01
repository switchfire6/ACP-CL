"""Independent portable arithmetic, causal provenance and archive tampering."""

import copy
import gzip
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import verify_predictive_value_archive as verifier


def config():
    return dict(seeds=[14991], models=["conditional"], prefix_blocks=2, prefix_size=16,
        acquisition_size=32, recovery_size=16, score_size=16, validation_size=16,
        gaps=[8, 16], batch_size=8, candidate_count=2, rehearsal_updates=2,
        updates_per_batch=1, memory_packets=4)


def seal(directory):
    entries = {path.relative_to(directory).as_posix(): dict(bytes=len(path.read_bytes()), sha256=verifier.sha(path.read_bytes()))
               for path in directory.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"}
    (directory/"artifact_manifest.json").write_text(json.dumps(entries))


@pytest.mark.parametrize("fault", ("bytes", "extra", "missing"))
def test_artifact_hash_and_exact_file_set_reject_changes(tmp_path, fault):
    path = tmp_path/"records.json"
    path.write_text("{}")
    seal(tmp_path)
    assert verifier.verify_artifacts(tmp_path) == 1
    if fault == "bytes":
        path.write_text("[]")
    elif fault == "extra":
        (tmp_path/"unlisted.txt").write_text("extra")
    else:
        path.unlink()
    with pytest.raises(ValueError):
        verifier.verify_artifacts(tmp_path)


@pytest.mark.parametrize("name", ("../outside", "a/../outside", "/absolute", "C:/absolute", "a\\b", "a//b", "./a"))
def test_archive_paths_cannot_escape_or_alias(name):
    with pytest.raises(ValueError):
        verifier.relative_path(name)


@pytest.mark.parametrize("contents", ('{"x": NaN}', '{"x": Infinity}', '{"x": 1,"x": 2}'))
def test_json_rejects_nonfinite_and_duplicate_fields(contents):
    with pytest.raises(ValueError):
        verifier.decode(contents)


def test_source_zip_rejects_replacement_and_unlisted_entries(tmp_path):
    path = tmp_path/"source.zip"
    expected = {"source.py": verifier.sha(b"locked")}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"locked")
    verifier.verify_zip(path, expected)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verifier.verify_zip(path, expected)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"locked")
        archive.writestr("extra.py", b"extra")
    with pytest.raises(ValueError, match="file set"):
        verifier.verify_zip(path, expected)


def portable_records(folder):
    c, identity = config(), {"test": "synthetic"}
    records = {}
    for path, kind in verifier.expected_paths(c).items():
        record = dict(identity=identity, seed=c["seeds"][0], model="conditional")
        if kind in ("assessment", "choice"):
            record["index"] = int(path.split("assessment_")[1].split("/")[0])
        records[path] = record

    def write():
        with gzip.open(folder/"raw_results.jsonl.gz", "wt", encoding="utf-8") as stream:
            for path, record in records.items():
                if path.endswith("/result.json") and "assessment_" in path:
                    stream.write(json.dumps(record)+"\n")
        with gzip.open(folder/"phase_results.jsonl.gz", "wt", encoding="utf-8") as stream:
            for path, record in records.items():
                if path.count("/") == 2 and "assessment_" not in path:
                    stream.write(json.dumps(dict(path=path, record=record))+"\n")
        for kind in ("trajectory", "choice"):
            subset = {path: record for path, record in records.items()
                      if (path.count("/") == 1 if kind == "trajectory" else path.endswith("/choice.json"))}
            (folder/f"{kind}_records.json").write_text(json.dumps(subset))

    write()
    summary = dict(result_hashes=[dict(path=path, kind=kind, sha256=verifier.sha(verifier.original_json(records[path])))
                                 for path, kind in verifier.expected_paths(c).items()])
    return c, identity, records, summary, write


@pytest.mark.parametrize("fault", ("changed", "missing", "duplicate", "identity", "hash_listing"))
def test_original_json_reconstruction_rejects_tampering_without_outer_manifest(tmp_path, fault):
    c, identity, records, summary, write = portable_records(tmp_path)
    assert verifier.read_records(tmp_path, c, identity, summary)[0] == records
    path = next(path for path in records if "/prefix_0/" in path)
    if fault == "changed":
        records[path]["surprise"] = 7
    elif fault == "missing":
        records.pop(path)
    elif fault == "identity":
        records[path]["identity"] = {"test": "other"}
    elif fault == "hash_listing":
        summary["result_hashes"].append(copy.deepcopy(summary["result_hashes"][0]))
    write()
    if fault == "duplicate":
        with gzip.open(tmp_path/"phase_results.jsonl.gz", "at", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(path=path, record=records[path]))+"\n")
    with pytest.raises(ValueError):
        verifier.read_records(tmp_path, c, identity, summary)


def phase_fixture():
    c = dict(config(), batch_size=2, validation_size=4, updates_per_batch=3)
    phase = dict(name="near_0", law=dict(mode=0, active=[0], revised=False, noise=0.),
                 size=4, kind="near", assessment=0, gap=8, cue=0)
    size, start = 2, 3
    packet_rows, evaluations = [], []
    previous = "a"*64
    for offset in range(2):
        truth = np.zeros((size, 5, 3))
        truth[0, 1] = 1
        truth[1, 2] = 1
        performed = [1, 0]
        outcomes = truth[np.arange(size), performed]
        probabilities = np.full((size, 3), .25+.1*offset)
        query_hash = str(offset+1)*64
        row = dict(id=start+offset, original_steps=(start+offset)*c["updates_per_batch"],
            origin_phase=phase["name"], origin_index=offset, origin_law=phase["law"],
            query_sha256=query_hash, support_sha256=previous, original_prediction_sha256="b"*64,
            original_model_sha256="c"*64, probabilities=probabilities.tolist(), actions=performed,
            outcomes=outcomes.tolist(), original_brier=float(np.square(probabilities-outcomes).mean()))
        packet_rows.append(row)
        evaluation = dict(packet_id=row["id"], query_sha256=query_hash, support_sha256=previous, truth=truth.tolist())
        for group in ("reapplied", "frozen"):
            evaluation[group] = []
            for position, probability in enumerate((.1, .4, .8)):
                predictions = np.full((size, 3), probability+.02*offset)
                chosen = ([1, 2], [0, 2], [0, 0])[position]
                evaluation[group].append(dict(probabilities=predictions.tolist(), prediction_sha256="d"*64,
                    brier=float(np.square(predictions-outcomes).mean()), actions=chosen,
                    survival=float(truth[np.arange(size), chosen, -1].mean())))
        evaluations.append(evaluation)
        previous = query_hash
    work = dict(steps=2, forwards=4, backward_calls=2, query_presentations=8, support_presentations=8,
        parent_unchanged=True, memory_history_unchanged=True, parent_state_sha256="e"*64, training_seconds=.01,
        model_bytes=100, optimizer_bytes=220, gradient_bytes=100, initial_encoder_bytes=50,
        replay_bytes=60, history_bytes=20, explicit_tensor_bytes_subtotal=550,
        explicit_tensor_bytes_scope="Explicit buffer sum, not measured peak RAM.")
    for name in ("anchor_packet_sha256", "replay_packet_sha256", "initial_model_sha256", "final_model_sha256",
                 "initial_optimizer_sha256", "final_optimizer_sha256", "shadow_state_sha256"):
        work[name] = "f"*64
    record = dict(seed=14991, model="conditional", phase=phase, packets=packet_rows, completed_packets=2,
        before_signature="e"*64, final_signature="f"*64, evaluations=evaluations,
        shadow_work=[copy.deepcopy(work) for _ in range(3)], frozen_work=[], panels={})
    for label, memory_label, seen in (("start_diagnostics", "start_memory", start), ("diagnostics", "memory", start+2)):
        record[label] = dict(cost=dict(arrivals=seen*size, optimizer_steps=seen*c["updates_per_batch"]))
        record[memory_label] = dict(type="ReservoirMemory", seen=seen, age=seen, capacity=4,
                                   ids=list(range(max(0, seen-4), seen)))
    record["metadata"] = {str(i): {} for i in record["memory"]["ids"]}
    for group in ("reapplied", "frozen"):
        brier = np.asarray([[p["brier"] for p in row[group]] for row in evaluations])
        survival = np.asarray([[p["survival"] for p in row[group]] for row in evaluations])
        record["panels"][group] = dict(brier=brier.mean(axis=0).tolist(), survival=survival.mean(axis=0).tolist(),
                                     early_brier=brier[0].tolist(), late_brier=brier[1:].mean(axis=0).tolist())
    return c, phase, record


@pytest.mark.parametrize("fault", ("prequential_brier", "shadow_brier", "survival", "truth", "query", "support", "action", "panel", "work", "storage"))
def test_phase_arithmetic_and_causal_matching_detect_local_tampering(fault):
    c, phase, record = phase_fixture()
    assert verifier.verify_phase(record, phase, c, 14991, "conditional", 3, "a"*64) == "2"*64
    if fault == "prequential_brier":
        record["packets"][0]["original_brier"] += .01
    elif fault == "shadow_brier":
        record["evaluations"][0]["reapplied"][0]["brier"] += .01
    elif fault == "survival":
        record["evaluations"][0]["frozen"][0]["survival"] = 0.
    elif fault == "truth":
        record["evaluations"][0]["truth"][0][1][0] = 0.
    elif fault == "query":
        record["evaluations"][0]["query_sha256"] = "9"*64
    elif fault == "support":
        record["packets"][1]["support_sha256"] = "a"*64
    elif fault == "action":
        record["evaluations"][0]["reapplied"][0]["actions"][0] = 5
    elif fault == "panel":
        record["panels"]["reapplied"]["late_brier"][0] += .01
    elif fault == "work":
        record["shadow_work"][1]["steps"] += 1
    else:
        record["shadow_work"][0]["explicit_tensor_bytes_subtotal"] += 1
    with pytest.raises(ValueError):
        verifier.verify_phase(record, phase, c, 14991, "conditional", 3, "a"*64)


def assessment(seed=14991, index=0, delta=0.):
    candidates = [dict(id=7, original_brier=.2), dict(id=3, original_brier=.1)]
    score = [.2, .3, .4]
    record = dict(seed=seed, model="conditional", index=index, candidates=candidates,
        replacement=dict(id=9, original_brier=.15), choice=dict(value_id=7, accuracy_id=3,
            score_losses=score, values=[score[-1]-loss for loss in score[:-1]]), panels={})
    for panel in verifier.PANELS:
        record["panels"][panel] = dict(brier=[.2+delta, .4, .3], survival=[.1, .9, .5],
                                       early_brier=[.25+delta, .45, .3], late_brier=[.15+delta, .35, .3])
    return record


def test_uniform_is_mean_candidate_loss_and_oracle_uses_brier_for_every_metric():
    record = assessment()
    record["panels"]["near_reapplied"]["brier"] = [0., 1., .3]
    calculated = verifier.selectors(record)["near_reapplied"]
    assert calculated["uniform"]["brier"] == .5  # An averaged-probability ensemble would give .25.
    assert calculated["oracle"]["brier"] == 0.
    assert calculated["oracle"]["survival"] == .1  # It must not switch to the survival winner.


def test_causal_selection_respects_exact_id_ties_and_rejects_validation_reselection():
    record = assessment()
    record["choice"].update(score_losses=[.2, .2, .4], values=[.2, .2], value_id=3)
    assert verifier.selected_indices(record) == (1, 1)
    record["choice"]["value_id"] = 7
    with pytest.raises(ValueError, match="sealed choice"):
        verifier.selected_indices(record)


def test_independent_seed_means_average_two_assessments_before_bootstrap():
    c = dict(config(), seeds=[14991, 14992])
    records = [assessment(seed, index, delta) for seed, index, delta in
               ((14991, 0, -.1), (14991, 1, .1), (14992, 0, .1), (14992, 1, .3))]
    contrasts, means, screens = verifier.recompute_statistics(records, c)
    result = contrasts["conditional"]["near_reapplied"]["value_minus_uniform"]["brier"]
    assert result["differences"] == pytest.approx([-.1, 0.])
    assert result["n"] == 2 and result["seeds"] == c["seeds"] and result["negative"] == 1
    expected = np.array([-.1, 0.])
    indices = np.random.default_rng(27192026).integers(0, 2, (20000, 2))
    assert [result["lower"], result["upper"]] == pytest.approx(np.quantile(expected[indices].mean(axis=1), [.025, .975]))
    assert means["conditional"]["near_reapplied"]["value"]["brier"] == pytest.approx(.3)
    assert not screens["conditional"]["near"]["passed"]
    with pytest.raises(ValueError, match="coverage"):
        verifier.recompute_statistics(records[:-1], c)


def test_screens_use_mean_gates_and_seed_consistency_not_confidence_bounds():
    seeds = list(range(14001, 14013))
    brier = verifier.bootstrap([-.001]*9+[.001]*3, seeds)
    # Mean exactly -.0005 may lie a floating-point ulp away; declare the boundary exactly.
    brier["mean"] = -.0005
    brier["upper"] = 99.
    row = dict(value_minus_uniform=dict(brier=brier, survival=verifier.bootstrap([-.005]*12, seeds)),
               value_minus_accuracy=dict(brier=verifier.bootstrap([0.]*12, seeds)))
    result = verifier.screen(row, True)
    assert result["passed"]
    row["value_minus_uniform"]["survival"]["mean"] = -.0051
    assert verifier.screen(row, True)["failed"] == ["survival"]
    assert "complete_scientific_cohort" in verifier.screen(row, False)["failed"]


def audit_fixture():
    c, identity = config(), {"test": "audit"}
    phases, hashes, cumulative = [], {}, 0
    for phase in verifier.schedule(14991, c):
        count = phase["size"]//c["batch_size"]
        cumulative += count
        scoring = phase["kind"] in ("score", "near", "return")
        path = f"conditional_14991/{phase['name']}/result.json"
        hashes[path] = "a"*64
        phases.append(dict(phase=phase["name"], packets=count, cumulative_packets=cumulative,
            result_sha256=hashes[path], before_sha256="b"*64, checkpoint_sha256="c"*64,
            shadows_sha256="d"*64 if scoring else None,
            causal_data_replay_metadata_verified=True, exact_start_chain_verified=True,
            adam_counters_verified=True, start_prediction_recomputed=True,
            diagnostic_predictions_recomputed=count if scoring else 0,
            rehearsal_forks_reproduced=3 if scoring else 0))
    predictions = 2*3*(c["score_size"]+4*c["validation_size"])//c["batch_size"]
    check = dict(seed=14991, model="conditional", before_checkpoints=len(phases), final_checkpoints=len(phases),
        assessments=2, ordinary_packets=cumulative, original_packet_errors_recomputed=cumulative,
        original_predictions_recomputed=len(phases), diagnostic_forks_reproduced=18,
        diagnostic_prediction_calls_recomputed=predictions, phases=phases, input_stream_sha256="e"*64)
    audit = dict(passed=True, identity=identity, trajectories=1, before_checkpoints=len(phases), final_checkpoints=len(phases),
        assessments=2, diagnostic_forks_reproduced=18, diagnostic_prediction_calls_recomputed=predictions, checks=[check])
    return c, identity, hashes, audit


@pytest.mark.parametrize("fault", ("count", "job", "phase", "hash", "flag", "scope_count"))
def test_archived_tensor_audit_requires_exact_job_phase_and_rehearsal_coverage(fault):
    c, identity, hashes, audit = audit_fixture()
    assert verifier.verify_audit(audit, c, identity, hashes)["trajectories"] == 1
    if fault == "count":
        audit["diagnostic_forks_reproduced"] -= 1
    elif fault == "job":
        audit["checks"][0]["seed"] = 14992
    elif fault == "phase":
        audit["checks"][0]["phases"].pop()
    elif fault == "hash":
        audit["checks"][0]["phases"][0]["result_sha256"] = "f"*64
    elif fault == "flag":
        audit["checks"][0]["phases"][0]["exact_start_chain_verified"] = False
    else:
        audit["checks"][0]["original_predictions_recomputed"] = audit["checks"][0]["ordinary_packets"]
    with pytest.raises(ValueError):
        verifier.verify_audit(audit, c, identity, hashes)


def test_schedule_counts_match_declared_main_cohort_without_importing_runner():
    c = dict(SCIENTIFIC=1, **verifier.SCIENTIFIC, seeds=list(range(14001, 14013)), models=list(verifier.MODELS))
    assert verifier.expected_counts(c) == dict(trajectories=24, assessments=48,
        ordinary_arrivals_per_trajectory=20992, diagnostic_forks=1296)
    assert len(verifier.schedule(14001, c)) == 14


def test_verifier_imports_no_live_training_or_analysis_code():
    import ast
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and (name.startswith("acp_cl") or "summarize" in name or "audit_predictive" in name
                             or "torch" in name or "pickle" in name) for name in imports)
