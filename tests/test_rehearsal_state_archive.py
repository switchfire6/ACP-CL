"""Independent all-action metrics, factorial arithmetic and portable tampering."""

import ast
import gzip
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import verify_rehearsal_state_archive as verifier


def config():
    return dict(seeds=[15991], models=["conditional"], prefix_blocks=2, prefix_size=16,
        acquisition_size=32, inter_assessment_size=16, score_size=16, validation_size=16,
        batch_size=8, candidate_count=2, rehearsal_updates=2, updates_per_batch=1, memory_packets=4)


def seal(folder):
    entries = {path.relative_to(folder).as_posix(): dict(bytes=len(path.read_bytes()), sha256=verifier.sha(path.read_bytes()))
               for path in folder.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"}
    (folder/"artifact_manifest.json").write_text(json.dumps(entries))


@pytest.mark.parametrize("fault", ("changed", "extra", "missing", "partial_manifest"))
def test_strict_artifact_manifest_and_bytes(fault, tmp_path):
    (tmp_path/"data.json").write_text("{}")
    (tmp_path/"other.json").write_text("[]")
    seal(tmp_path)
    assert verifier.verify_artifacts(tmp_path) == 2
    if fault == "changed":
        (tmp_path/"data.json").write_text("[]")
    elif fault == "extra":
        (tmp_path/"surprise.json").write_text("{}")
    elif fault == "missing":
        (tmp_path/"data.json").unlink()
    else:
        manifest = json.loads((tmp_path/"artifact_manifest.json").read_text())
        manifest.pop("other.json")
        (tmp_path/"artifact_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verifier.verify_artifacts(tmp_path)


@pytest.mark.parametrize("name", ("../escape", "a/../escape", "/absolute", "C:/absolute", "a\\b", "./a"))
def test_paths_cannot_escape_or_alias(name):
    with pytest.raises(ValueError):
        verifier.relative_path(name)


def test_duplicate_json_keys_nonfinite_and_locked_zip_changes_rejected(tmp_path):
    for text in ('{"x":NaN}', '{"x":1,"x":2}'):
        with pytest.raises(ValueError):
            verifier.decode(text)
    path = tmp_path/"source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"locked")
    verifier.verify_zip(path, {"source.py": verifier.sha(b"locked")})
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"altered")
    with pytest.raises(ValueError):
        verifier.verify_zip(path, {"source.py": verifier.sha(b"locked")})


def raw_arrays():
    c = dict(config(), batch_size=2, validation_size=4)
    observations = np.zeros((2, 2, 4, 12, 16), dtype=np.uint8)
    observations[1, :, 0, 0, 0] = 1
    actions = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    truth = np.zeros((2, 2, 2, 5, 3), dtype=np.uint8)
    truth[0, :, :, 0] = 1
    truth[1, :, :, 1] = 1
    outcomes = np.stack([branch[np.arange(2)[:, None], np.arange(2)[None], actions] for branch in truth])
    data = dict(observations=observations, actions=actions, outcomes=outcomes, truth=truth,
        initial_observations=np.full((2, 4, 12, 16), 2, dtype=np.uint8),
        initial_actions=np.zeros(2, dtype=np.uint8), initial_outcomes=np.zeros((2, 3), dtype=np.uint8))
    probabilities = np.zeros((2, 2, 3, 2, 5, 3), dtype=np.float32)
    probabilities[:, :, 0, :, 0] = 1
    probabilities[:, :, 1, :, 1] = 1
    probabilities[:, :, 2] = .5
    return c, data, probabilities


def test_all_action_brier_greedy_survival_and_early_late_are_independent():
    c, data, probabilities = raw_arrays()
    panels = verifier.metric_panels(probabilities, data, c, 3)
    assert panels["current"] == dict(brier=[0., 1., .25], survival=[1., 0., 1.], early_brier=[0., 1., .25], late_brier=[0., 1., .25])
    assert panels["return"] == dict(brier=[1., 0., .25], survival=[0., 1., 0.], early_brier=[1., 0., .25], late_brier=[1., 0., .25])
    # Argmax tie-breaking is the first action; survival is not inferred from Brier.
    probabilities[:, 1, 0] = .5
    panels = verifier.metric_panels(probabilities, data, c, 3)
    assert panels["current"]["brier"][0] == .125
    assert panels["current"]["early_brier"][0] == 0. and panels["current"]["late_brier"][0] == .25


@pytest.mark.parametrize("fault", ("dtype", "nonfinite", "probability", "candidate_axis", "zero_axis", "truth", "action", "missing_data"))
def test_corrupt_or_wrong_cell_array_schema_rejected(fault):
    c, data, probabilities = raw_arrays()
    count = 3
    if fault == "dtype":
        probabilities = probabilities.astype(np.float64)
    elif fault == "nonfinite":
        probabilities[0, 0, 0, 0, 0, 0] = np.nan
    elif fault == "probability":
        probabilities[0, 0, 0, 0, 0, 0] = 1.1
    elif fault == "candidate_axis":
        probabilities = probabilities[:, :, :2]
    elif fault == "zero_axis":
        count = 2
    elif fault == "truth":
        data["truth"][0, 0, 0, 0, 0] = 0
    elif fault == "action":
        data["actions"][0, 0] = 5
    else:
        data.pop("initial_outcomes")
    with pytest.raises(ValueError):
        verifier.metric_panels(probabilities, data, c, count)


def stream_fixture():
    c, data, _ = raw_arrays()
    identity = {"test": "contexts"}
    initial = verifier.experience_hash(data["initial_observations"], data["initial_actions"], data["initial_outcomes"])
    score, validation = dict(packets=[dict(query_sha256=initial)]), dict(packets=[])
    previous, rows = [initial, initial], []
    for packet in range(2):
        row = {}
        for branch_index, branch in enumerate(verifier.BRANCHES):
            query = verifier.experience_hash(data["observations"][packet], data["actions"][packet], data["outcomes"][branch_index, packet])
            row[branch] = dict(query_sha256=query, support_sha256=previous[branch_index])
            previous[branch_index] = query
        rows.append(row)
        validation["packets"].append(dict(**row["current"], actions=data["actions"][packet].tolist(), outcomes=data["outcomes"][0, packet].tolist()))
    stream = dict(identity=identity, seed=15991, model="conditional", index=0, branches=list(verifier.BRANCHES), rows=rows)
    return c, data, score, validation, identity, stream


@pytest.mark.parametrize("fault", ("cross_branch", "future_support", "initial", "ordinary_feedback", "branch_order"))
def test_matched_initial_context_and_branch_specific_feedback_are_enforced(fault):
    c, data, score, validation, identity, stream = stream_fixture()
    verifier.verify_contexts(stream, data, validation, score, c, identity, 15991, "conditional", 0)
    if fault == "cross_branch":
        stream["rows"][1]["return"]["support_sha256"] = stream["rows"][1]["current"]["support_sha256"]
    elif fault == "future_support":
        stream["rows"][0]["return"]["support_sha256"] = stream["rows"][0]["return"]["query_sha256"]
    elif fault == "initial":
        data["initial_observations"][0, 0, 0, 0] = 3
    elif fault == "ordinary_feedback":
        validation["packets"][0]["outcomes"][0][0] = 0
    else:
        stream["branches"].reverse()
    with pytest.raises(ValueError):
        verifier.verify_contexts(stream, data, validation, score, c, identity, 15991, "conditional", 0)


def assessment(seed=15991, index=0, multiplier=1.):
    scores = [.1, .2, .3]
    record = dict(seed=seed, model="conditional", index=index,
        candidates=[dict(id=7, original_brier=.2), dict(id=3, original_brier=.1)],
        replacement=dict(id=9, original_brier=.4), choice=dict(value_id=7, accuracy_id=3,
            score_losses=scores, values=[scores[-1]-value for value in scores[:-1]]), panels={}, zero={})
    for branch in verifier.BRANCHES:
        record["panels"][branch] = {}
        record["zero"][branch] = {metric: [.4, .2] for metric in verifier.METRICS}
        for cell in verifier.CELLS:
            w, o, a = map(int, cell)
            delta = multiplier*.001*(1+2*w+3*o+5*a+7*w*o+11*w*a+13*o*a+17*w*o*a)
            if branch == "return":
                delta += .01
            record["panels"][branch][cell] = {metric: [.3+delta, .3-delta, .9] for metric in verifier.METRICS}
    return record


def test_all_factorial_coefficients_conditional_main_pair_triple_and_repairs():
    result = verifier.assessment_metrics(assessment())
    effects = result["effects"]["current"]["selection"]
    assert effects["total_111_minus_000"]["brier"] == pytest.approx(.058)
    assert effects["weights_at_optimizer0_anchor0"]["brier"] == pytest.approx(.002)
    assert effects["weights_main"]["brier"] == pytest.approx(.002+.007/2+.011/2+.017/4)
    assert effects["weights_optimizer_interaction"]["brier"] == pytest.approx(.007+.017/2)
    assert effects["three_way_interaction"]["brier"] == pytest.approx(.017)
    assert effects["repair_weights"]["brier"] == pytest.approx(.037)
    assert effects["repair_optimizer"]["brier"] == pytest.approx(.040)
    assert effects["repair_anchor"]["brier"] == pytest.approx(.046)
    assert result["effects"]["current"]["uniform"]["three_way_interaction"]["brier"] == pytest.approx(0)
    assert result["environment"]["selection"]["111"]["brier"] == pytest.approx(.01)
    assert len(verifier.coefficients()) == 23


def test_uniform_excludes_replacement_and_zero_reference_follows_only_weight_bit():
    result = verifier.assessment_metrics(assessment())
    for cell in verifier.CELLS:
        row = result["panels"]["current"][cell]
        assert row["selectors"]["uniform"]["brier"] == pytest.approx(.3)
        zero = .4 if cell[0] == "0" else .2
        assert row["contrasts"]["value_minus_matching_zero"]["brier"] == pytest.approx(row["selectors"]["value"]["brier"]-zero)


def test_choice_and_validation_oracle_use_declared_ties_and_one_candidate_for_all_metrics():
    record = assessment()
    record["choice"].update(score_losses=[.1, .1, .3], values=[.3-.1, .3-.1], value_id=3)
    assert verifier.selected_indices(record) == (1, 1)
    record["panels"]["current"]["000"].update(brier=[0., 1., .5], survival=[.1, .9, .5])
    oracle = verifier.assessment_metrics(record)["panels"]["current"]["000"]["selectors"]["oracle"]
    assert oracle["brier"] == 0. and oracle["survival"] == .1
    record["choice"]["value_id"] = 7
    with pytest.raises(ValueError, match="choice"):
        verifier.selected_indices(record)


def summary_fixture():
    c = dict(config(), seeds=[15991, 15992])
    records = [assessment(seed, index, multiplier) for seed, index, multiplier in
               ((15991, 0, 1.), (15991, 1, 2.), (15992, 0, 1.), (15992, 1, .5))]
    summary = verifier.recompute_statistics(records, c)
    summary.update(coefficients=verifier.coefficients(), eligible_scientific_cohort=False,
        inference=dict(bootstrap_seed=27192026, bootstrap_resamples=20000, bit_order=list(verifier.FACTORS)))
    return c, records, summary


def test_two_assessments_are_averaged_before_seed_bootstrap():
    c, _, summary = summary_fixture()
    primary = summary["effects"]["conditional"]["current"]["selection"]["total_111_minus_000"]["brier"]
    expected = np.array([.058*1.5, .058*.75])
    assert primary["differences"] == pytest.approx(expected.tolist())
    assert primary["n"] == 2 and primary["seeds"] == c["seeds"]
    draws = np.random.default_rng(27192026).integers(0, 2, (20000, 2))
    assert [primary["lower"], primary["upper"]] == pytest.approx(np.quantile(expected[draws].mean(axis=1), [.025, .975]))


def test_resealed_wrong_factorial_metric_is_rejected_independently(tmp_path):
    c, records, summary = summary_fixture()
    verifier.verify_statistics(summary, records, c)
    summary["effects"]["conditional"]["current"]["selection"]["three_way_interaction"]["brier"]["mean"] += .01
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    seal(tmp_path)
    assert verifier.verify_artifacts(tmp_path) == 1
    with pytest.raises(ValueError, match="numeric mismatch"):
        verifier.verify_statistics(verifier.read(tmp_path/"summary.json"), records, c)


def test_primary_screen_uses_declared_means_and_positive_seed_count():
    primary = verifier.bootstrap([.001]*9+[-.001]*3, list(range(15001, 15013)))
    primary["mean"], primary["lower"] = .0005, -99.
    assert verifier.screen(primary, True)["passed"]
    primary["differences"][0] = 0.
    assert verifier.screen(primary, True)["failed"] == ["seed_consistency"]
    assert "complete_scientific_cohort" in verifier.screen(primary, False)["failed"]


def record_fixture(folder):
    c, identity = config(), {"test": "portable"}
    paths = verifier.expected_paths(c)
    records, hashes = {}, {}
    for path, kind in paths.items():
        if kind == "checkpoint":
            hashes[path] = "a"*64
        elif kind == "array":
            destination = folder/"arrays"/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, probabilities=np.zeros(1, dtype=np.float32))
            hashes[path] = verifier.sha(destination.read_bytes())
        else:
            record = dict(identity=identity, seed=15991, model="conditional")
            if kind == "assessment":
                record["index"] = int(path.split("diagnostic_")[1].split("/")[0])
            records[path] = record
            hashes[path] = verifier.sha(verifier.original_json(record))

    def write():
        with gzip.open(folder/"raw_results.jsonl.gz", "wt", encoding="utf-8") as stream:
            for path, record in records.items():
                if paths[path] == "assessment":
                    stream.write(json.dumps(record)+"\n")
        with gzip.open(folder/"phase_results.jsonl.gz", "wt", encoding="utf-8") as stream:
            for path, record in records.items():
                if paths[path] == "phase":
                    stream.write(json.dumps(dict(path=path, record=record))+"\n")
        for kind in ("trajectory", "auxiliary"):
            (folder/f"{kind}_records.json").write_text(json.dumps({path: value for path, value in records.items() if paths[path] == kind}))

    write()
    summary = dict(array_archive_prefix="arrays", result_hashes=[dict(path=path, kind=kind, sha256=hashes[path]) for path, kind in paths.items()])
    return c, identity, records, summary, write


@pytest.mark.parametrize("fault", ("record", "missing", "duplicate", "array", "extra_array", "checkpoint_reference"))
def test_original_json_npz_and_checkpoint_reference_coverage_survive_resealing(tmp_path, fault):
    c, identity, records, summary, write = record_fixture(tmp_path)
    verifier.read_records(tmp_path, c, identity, summary)
    phase = next(path for path in records if "/prefix_0/" in path)
    if fault == "record":
        records[phase]["tampered"] = True
    elif fault == "missing":
        records.pop(phase)
    elif fault == "array":
        path = next((tmp_path/"arrays").rglob("*.npz"))
        np.savez_compressed(path, probabilities=np.ones(1, dtype=np.float32))
    elif fault == "extra_array":
        np.savez_compressed(tmp_path/"arrays"/"extra.npz", x=np.zeros(1))
    elif fault == "checkpoint_reference":
        summary["result_hashes"] = [row for row in summary["result_hashes"] if row["kind"] != "checkpoint"]
    write()
    if fault == "duplicate":
        with gzip.open(tmp_path/"phase_results.jsonl.gz", "at", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(path=phase, record=records[phase]))+"\n")
    seal(tmp_path)
    verifier.verify_artifacts(tmp_path)
    with pytest.raises(ValueError):
        verifier.read_records(tmp_path, c, identity, summary)


def test_no_pickle_npz_or_live_training_analysis_imports(tmp_path):
    path = tmp_path/"unsafe.npz"
    np.savez_compressed(path, value=np.array([{"hidden": 1}], dtype=object))
    with pytest.raises(ValueError):
        verifier.read_arrays(path)
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and (name.startswith("acp_cl") or "summarize" in name or "audit_rehearsal" in name
                             or name in ("torch", "pickle")) for name in imports)


def test_main_counts_include_reused_cells_zero_copies_and_instrumentation():
    c = dict(verifier.SCIENTIFIC, seeds=list(range(15001, 15013)), models=list(verifier.MODELS))
    assert verifier.expected_counts(c) == dict(trajectories=24, assessments=48,
        ordinary_arrivals_per_trajectory=18432, diagnostic_forks=3456, instrumented_forks=3024,
        zero_update_copies=96, diagnostic_prediction_forwards=120576, extra_loss_forwards=12096)


def audit_fixture():
    c, identity = config(), {"test": "tensor-audit"}
    paths = verifier.expected_paths(c)
    hashes = {path: "a"*64 for path in paths}
    integrity = {path: dict(sha256=value) for path, value in hashes.items()}
    records, phase_rows, queries, cumulative = {}, [], [], 0
    prefix = "conditional_15991"
    for phase in verifier.schedule(15991, c):
        count = phase["size"]//c["batch_size"]
        cumulative += count
        scoring = phase["kind"] == "score"
        base = f"{prefix}/{phase['name']}"
        records[f"{base}/result.json"] = dict(packets=[dict(query_sha256="b"*64) for _ in range(count)])
        queries += ["b"*64]*count
        for name in ("before.pt", "checkpoint.pt"):
            integrity[f"{base}/{name}"] = dict(sha256="a"*64)
        phase_rows.append(dict(phase=phase["name"], packets=count, cumulative_packets=cumulative,
            result_sha256="a"*64, before_sha256="a"*64, checkpoint_sha256="a"*64,
            shadows_sha256="a"*64 if scoring else None, rehearsal_forks_reproduced=3 if scoring else 0,
            diagnostic_predictions_recomputed=count if scoring else 0,
            causal_data_replay_metadata_verified=True, exact_start_chain_verified=True,
            adam_counters_verified=True, start_prediction_recomputed=True))
    assessment_rows = []
    for index in (0, 1):
        base = f"{prefix}/diagnostic_{index}"
        parents = dict(old="c"*64, new="d"*64)
        state_files, prediction_files = {}, {}
        for cell in (*verifier.CELLS, "zero"):
            state = f"score_{index}/shadows.pt" if cell == "000" else f"diagnostic_{index}/zero.pt" if cell == "zero" else f"diagnostic_{index}/cell_{cell}.pt"
            state_files[cell] = dict(path=state, sha256="a"*64)
            prediction_files[cell] = dict(path=f"diagnostic_{index}/predictions_{cell}.npz", sha256="a"*64)
        records[f"{base}/result.json"] = dict(parents=parents, state_files=state_files, prediction_files=prediction_files)
        assessment_rows.append(dict(index=index, result_sha256="a"*64, before_sha256="a"*64, data_sha256="a"*64,
            stream_sha256="a"*64, choice_sha256="a"*64, candidates_sha256="a"*64, score_result_sha256="a"*64,
            parents=parents, state_files=state_files, prediction_files=prediction_files,
            newly_reproduced_forks=21, reused_score_forks=3, zero_update_copies=2,
            diagnostic_prediction_calls_recomputed=2*2*(24+2), loss_instrumentation_forwards_recomputed=84,
            all_new_active_endpoints_verified=3, paired_causal_branches_verified=True,
            unchanged_donors_verified=True, score_only_selection_verified=True))
    declared = verifier.expected_counts(c)
    totals = dict(before_checkpoints=8, final_checkpoints=8, diagnostic_boundary_checkpoints=2, assessments=2,
        ordinary_packets=cumulative, original_packet_errors_recomputed=cumulative, original_predictions_recomputed=8,
        diagnostic_forks_reproduced=declared["diagnostic_forks"], zero_update_copies=4,
        diagnostic_prediction_calls_recomputed=declared["diagnostic_prediction_forwards"],
        loss_instrumentation_forwards_recomputed=declared["extra_loss_forwards"])
    check = dict(seed=15991, model="conditional", **totals, phases=phase_rows, assessment_checks=assessment_rows,
                 input_stream_sha256=verifier.digest(queries))
    audit = dict(passed=True, identity=identity, trajectories=1, **totals, checks=[check], manifest_sha256="e"*64,
        audit_source_sha256="f"*64, result_hashes=[dict(path=path, kind=kind, sha256=hashes[path]) for path, kind in paths.items()])
    return (audit, c, identity, records, hashes, integrity, "e"*64, {"scripts/audit_rehearsal_state.py": "f"*64})


@pytest.mark.parametrize("fault", ("missing_job", "count", "phase_hash", "missing_assessment", "endpoints", "zero", "cell_mapping", "input", "source"))
def test_full_auditor_counts_donor_endpoints_and_file_references_cannot_be_changed(fault):
    arguments = audit_fixture()
    audit = arguments[0]
    assert verifier.verify_audit(*arguments)["trajectories"] == 1
    if fault == "missing_job":
        audit["checks"] = []
    elif fault == "count":
        audit["loss_instrumentation_forwards_recomputed"] -= 4
    elif fault == "phase_hash":
        audit["checks"][0]["phases"][0]["checkpoint_sha256"] = "9"*64
    elif fault == "missing_assessment":
        audit["checks"][0]["assessment_checks"].pop()
    elif fault == "endpoints":
        audit["checks"][0]["assessment_checks"][0]["all_new_active_endpoints_verified"] = 2
    elif fault == "zero":
        audit["checks"][0]["assessment_checks"][0]["zero_update_copies"] = 3
    elif fault == "cell_mapping":
        # Replace the map rather than changing the fixture's intentionally shared dictionaries.
        item = audit["checks"][0]["assessment_checks"][0]
        item["state_files"] = {**item["state_files"], "111": dict(path="diagnostic_0/cell_110.pt", sha256="a"*64)}
    elif fault == "input":
        audit["checks"][0]["input_stream_sha256"] = "8"*64
    else:
        audit["audit_source_sha256"] = "7"*64
    with pytest.raises(ValueError):
        verifier.verify_audit(*arguments)


@pytest.mark.parametrize("fault", ("partial", "copied_script", "zip", "executed_source"))
def test_standalone_verifier_source_seal_is_complete_and_corresponds_to_executed_code(tmp_path, fault):
    assert verifier.verify_packaging(tmp_path) is False
    source_name = "scripts/verify_rehearsal_state_archive.py"
    test_name = "tests/test_rehearsal_state_archive.py"
    source = Path(verifier.__file__).read_bytes()
    tests = b"test contents"

    def package(contents):
        (tmp_path/"verifier_lock.json").write_text(json.dumps(dict(files={source_name: verifier.sha(contents), test_name: verifier.sha(tests)})))
        (tmp_path/"verify_archive.py").write_bytes(contents)
        with zipfile.ZipFile(tmp_path/"verifier_source.zip", "w") as archive:
            archive.writestr(source_name, contents)
            archive.writestr(test_name, tests)

    package(source)
    assert verifier.verify_packaging(tmp_path) is True
    if fault == "partial":
        (tmp_path/"verifier_source.zip").unlink()
    elif fault == "copied_script":
        (tmp_path/"verify_archive.py").write_bytes(b"altered")
    elif fault == "zip":
        with zipfile.ZipFile(tmp_path/"verifier_source.zip", "w") as archive:
            archive.writestr(source_name, source+b"# change\n")
            archive.writestr(test_name, tests)
    else:
        package(b"foreign but consistently sealed source")
    with pytest.raises(ValueError):
        verifier.verify_packaging(tmp_path)
