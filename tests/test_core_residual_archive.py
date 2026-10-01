"""Independent portable arithmetic, causal provenance and corruption checks."""

import ast
import copy
import gzip
from itertools import permutations
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import verify_core_residual_archive as verifier


def config(full=False):
    value = dict(copy.deepcopy(verifier.SCIENTIFIC), study="portable_fixture", threads=1, workers=1, cpu_affinity=0)
    if not full:
        value.update(seeds=[16991], models=["conditional"], prefix_blocks=2, prefix_size=64,
            episode_size=128, maintenance_size=32, return_size=64, batch_size=8, eval_size=64,
            probe_every=32, support_replicates=1, width=8, context_width=4, decoder_width=8,
            residual_width=4, residual_context_width=2, residual_decoder_width=4,
            updates_per_batch=1, memory_packets=4)
    return value


def seal(folder):
    entries = {path.relative_to(folder).as_posix(): dict(bytes=path.stat().st_size, sha256=verifier.sha(path.read_bytes()))
               for path in folder.rglob("*") if path.is_file() and path != folder/"artifact_manifest.json"}
    (folder/"artifact_manifest.json").write_bytes(verifier.original_json(entries))


@pytest.mark.parametrize("fault", ("changed", "extra", "missing", "partial_manifest"))
def test_strict_artifact_coverage_and_bytes(fault, tmp_path):
    (tmp_path/"first.json").write_text("{}")
    (tmp_path/"second.json").write_text("[]")
    seal(tmp_path)
    assert verifier.verify_artifacts(tmp_path) == 2
    if fault == "changed":
        (tmp_path/"first.json").write_text("[]")
    elif fault == "extra":
        (tmp_path/"unexpected.json").write_text("{}")
    elif fault == "missing":
        (tmp_path/"first.json").unlink()
    else:
        entries = verifier.read(tmp_path/"artifact_manifest.json")
        entries.pop("first.json")
        (tmp_path/"artifact_manifest.json").write_text(json.dumps(entries))
    with pytest.raises(ValueError):
        verifier.verify_artifacts(tmp_path)


@pytest.mark.parametrize("name", ("../escape", "a/../escape", "/absolute", "C:/absolute", "a\\b", "./a", "a//b", ""))
def test_archive_paths_cannot_escape_or_alias(name):
    with pytest.raises(ValueError):
        verifier.relative_path(name)


def test_duplicate_json_nonfinite_and_locked_zip_changes_fail(tmp_path):
    for contents in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'):
        with pytest.raises(ValueError):
            verifier.decode(contents)
    path = tmp_path/"locked.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"original")
    verifier.verify_zip(path, {"source.py":verifier.sha(b"original")})
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"changed")
    with pytest.raises(ValueError):
        verifier.verify_zip(path, {"source.py":verifier.sha(b"original")})


def numerical_fixture():
    predictions = np.full((2, 5, 3), .5, dtype=np.float32)
    predictions[0, 0] = [1., .5, 0.]
    predictions[1, 1] = [1., 1., 1.]
    truth = np.zeros((2, 5, 3), dtype=np.uint8)
    truth[0, 0] = [1, 1, 0]
    truth[1, 1] = 1
    return predictions, truth, np.array([True, False]), np.array([False, True])


def test_all_action_masked_brier_argmax_survival_and_marginal_are_distinct():
    arrays = numerical_fixture()
    result = verifier.raw_metrics(*arrays, marginal=np.full((5, 3), .5))
    assert result["brier"] == pytest.approx((3.25+3.)/30)
    assert result["focus_brier"] == pytest.approx(3.25/15)
    assert result["valid_brier"] == pytest.approx(3./15)
    assert result["survival"] == .5 and result["focus_survival"] == 0. and result["valid_survival"] == 1.
    assert result["no_transfer"] == 0. and result["clairvoyant_upper"] == .5
    assert result["marginal_brier"] == .25
    # Every forecast tied at H12 chooses the first action, not the best truth.
    arrays[0][:] = .5
    assert verifier.raw_metrics(*arrays)["survival"] == 0.


@pytest.mark.parametrize("fault", ("dtype", "nan", "range", "horizon", "shape", "truth", "mask", "empty_mask"))
def test_invalid_raw_prediction_or_mask_rejected(fault):
    prediction, truth, affected, valid = numerical_fixture()
    if fault == "dtype":
        prediction = prediction.astype(np.float64)
    elif fault == "nan":
        prediction[0, 0, 0] = np.nan
    elif fault == "range":
        prediction[0, 0, 0] = 1.1
    elif fault == "horizon":
        prediction[0, 0] = [0., .5, 1.]
    elif fault == "shape":
        prediction = prediction[:, :4]
    elif fault == "truth":
        truth[0, 0, 0] = 2
    elif fault == "mask":
        affected = affected.astype(np.uint8)
    else:
        valid[:] = False
    with pytest.raises(ValueError):
        verifier.raw_metrics(prediction, truth, affected, valid)


def test_auc_includes_zero_and_normalizes_uneven_intervals():
    curve = [dict(arrivals=x, metrics=dict(error=y)) for x,y in ((0, .4), (2, .2), (8, .1))]
    assert verifier.auc(curve, "error") == pytest.approx((2*.3+6*.15)/8)
    curve[0]["arrivals"] = 1
    with pytest.raises(ValueError):
        verifier.auc(curve, "error")


def test_bootstrap_resamples_only_independent_seeds_and_preserves_direction():
    values, seeds = [-.003, .001, -.005, -.002, .004, 1e-14], list(range(6))
    result = verifier.bootstrap(values, seeds)
    sample = np.asarray(values)
    sample[-1] = 0
    draws = np.random.default_rng(27192026).integers(0, 6, (20000, 6))
    assert result["mean"] == pytest.approx(sample.mean())
    assert [result["lower"], result["upper"]] == pytest.approx(np.quantile(sample[draws].mean(axis=1), [.025, .975]))
    assert result["n"] == 6 and result["negative"] == 3 and result["positive"] == 2
    assert result["differences"] == sample.tolist() and result["seeds"] == seeds
    with pytest.raises(ValueError):
        verifier.bootstrap(values, [0]*6)


def metric_probe(error, cue=True):
    metrics = {name:(.7 if "survival" in name else error) for name in verifier.METRICS}
    metrics["marginal_brier"] = error+.04
    replicate = dict(curve=[dict(feedback=n, metrics=dict(metrics)) for n in (0, 8, 16, 32)])
    if cue:
        replicate["flipped"] = dict(metrics, focus_brier=error+.004)
    return dict(replicates=[copy.deepcopy(replicate), copy.deepcopy(replicate)])


def phase_record(seed, model, arm, phase):
    cue = tuple(permutations(range(3)))[seed % 6][0] if phase == "novel" else None
    error = .2 + ({"joint":0., "separate":-.003, "fixed_features":-.001, "fresh":-.01}[arm] if cue is not None else 0.)
    before, after = metric_probe(error, cue is not None), metric_probe(error, cue is not None)
    metrics = after["replicates"][0]["curve"][-1]["metrics"]
    valid = {str(mode):dict(valid_brier=.1, valid_survival=.7) for mode in (0, 1)}
    return dict(seed=seed, model=model, arm=arm, phase=phase, cue=cue, before=before, after=after,
        valid_before=copy.deepcopy(valid), valid_after=valid,
        curve=[dict(arrivals=x, metrics=dict(metrics)) for x in (0, 64, 256)],
        packets=[dict(prequential_brier=.4), dict(prequential_brier=.5)],
        novel_after_return=metric_probe(.25+(.001 if arm == "separate" else 0.)) if phase == "return" else None)


@pytest.fixture(scope="module")
def statistics_fixture():
    cfg = config(full=True)
    records = [phase_record(seed, model, arm, phase) for seed in cfg["seeds"] for model in cfg["models"]
               for arm in (*verifier.ARMS, "fresh") for phase in (("novel",) if arm == "fresh" else verifier.PHASES)]
    return cfg, records, verifier.recompute_statistics(records, cfg)


def test_three_comparisons_qualification_and_attribution_use_paired_seed_units(statistics_fixture):
    cfg, records, result = statistics_fixture
    assert len(records) == 120 and result["eligible_development_cohort"]
    assert all(row["prequential_brier"] == .45 for row in result["rows"])
    for model in cfg["models"]:
        assert result["screen"][model]["passed"] and result["attribution"][model]["passed"]
        comparison = result["comparisons"][model]["separate_minus_joint"]
        assert comparison["novel"]["brier_auc"]["mean"] == pytest.approx(-.003)
        assert comparison["novel"]["brier_auc"]["n"] == 6
        assert comparison["return"]["novel_after_return_brier"]["mean"] == pytest.approx(.001)
        assert result["comparisons"][model]["separate_minus_fixed_features"]["novel"]["brier_auc"]["mean"] == pytest.approx(-.002)
    assert all(row["n"] == 2 for row in result["qualification"]["groups"] if row["group"].startswith("cue_"))


@pytest.mark.parametrize("phase,metric,value,gate", (
    ("novel", "brier_auc", -.0019, "acquisition_gain"),
    ("novel", "valid_after_mode_0", .0051, "valid_old_mode_0"),
    ("novel", "valid_after_mode_1", .0051, "valid_old_mode_1"),
    ("return", "all_brier_auc", .0051, "return_prediction"),
    ("return", "novel_after_return_brier", .0051, "post_return_novel_retention"),
    ("novel", "survival_auc", -.0101, "survival_novel"),
    ("return", "survival_auc", -.0101, "survival_return")))
def test_each_predeclared_primary_guard_is_separate(statistics_fixture, phase, metric, value, gate):
    comparison = copy.deepcopy(statistics_fixture[2]["comparisons"]["conditional"]["separate_minus_joint"])
    comparison[phase][metric]["mean"] = value
    assert verifier.primary_screen(comparison, True, True)["failed"] == [gate]


def test_five_of_six_threshold_and_mean_gate_are_not_replaced_by_interval(statistics_fixture):
    comparison = copy.deepcopy(statistics_fixture[2]["comparisons"]["conditional"]["separate_minus_joint"])
    comparison["novel"]["brier_auc"].update(mean=-.002, lower=-.006, upper=.001, differences=[-.003]*5+[0.])
    result = verifier.primary_screen(comparison, True, True)
    assert result["passed"] and result["gates"]["acquisition_gain"]["uncertainty_crosses_limit"]
    comparison["novel"]["brier_auc"]["differences"] = [-.004]*4+[.001]*2
    assert verifier.primary_screen(comparison, True, True)["failed"] == ["acquisition_consistency"]


def test_one_weak_fresh_cue_blocks_only_its_model_and_attribution_cannot_rescue_it(statistics_fixture):
    result = statistics_fixture[2]
    rows = copy.deepcopy(result["rows"])
    for row in rows:
        if row["model"] == "conditional" and row["arm"] == "fresh" and row["cue"] == 1:
            row["cue_effect"] = .0019
    qualification = verifier.fresh_qualification(rows, verifier.MODELS)
    assert qualification["by_model"] == dict(conditional=False, recurrent=True)
    primary = result["comparisons"]["conditional"]["separate_minus_joint"]
    assert verifier.primary_screen(primary, False, True)["failed"] == ["fresh_qualification"]
    assert verifier.attribution_screen(result["comparisons"]["conditional"]["separate_minus_fixed_features"], True)["passed"]


def test_modified_recipe_and_smoke_are_ineligible_and_full_counts_exact():
    cfg = config(full=True)
    assert verifier.eligible(cfg) and verifier.eligible(dict(cfg, workers=9, cpu_affinity=4))
    assert not verifier.eligible(config()) and not verifier.eligible(dict(cfg, episode_size=4096))
    assert verifier.expected_counts(cfg) == dict(jobs=12, prefix_fits=12, prefix_phases=48,
        learning_trajectories=36, fresh_references=12, post_prefix_episodes=120, total_phases=168,
        training_arrivals=552960, training_packets=17280, optimizer_steps=207360)
    cfg["unexpected"] = 1
    with pytest.raises(ValueError):
        verifier.expected_counts(cfg)


@pytest.mark.parametrize("fault", ("missing", "duplicate"))
def test_all_arm_phase_seed_records_are_required(statistics_fixture, fault):
    cfg, records, _ = statistics_fixture
    values = list(records)
    values.pop() if fault == "missing" else values.append(records[0])
    with pytest.raises(ValueError):
        verifier.recompute_statistics(values, cfg)


def trace_fixture():
    prediction, truth, affected, valid = numerical_fixture()
    arrays = dict(e0_probabilities=prediction, e0_truth=truth, e0_affected=affected, e0_valid=valid,
        e0_observations=np.zeros((2, 4, 12, 16), dtype=np.uint8),
        e0_support_observations=np.zeros((1, 4, 12, 16), dtype=np.uint8),
        e0_support_actions=np.zeros(1, dtype=np.uint8), e0_support_outcomes=np.zeros((1, 3), dtype=np.uint8))
    row = dict(index=0, law=dict(mode=0, active=[], revised=False, noise=0.), cue=None, branch="novel", flipped=False,
        support_fingerprint=verifier.experience_hash(*(arrays["e0_support_"+name] for name in ("observations", "actions", "outcomes"))),
        query_sha256=verifier.array_hash(arrays["e0_observations"]), model_sha256="a"*64,
        metrics=verifier.raw_metrics(prediction, truth, affected, valid), marginal=None,
        query_presentations=2, support_presentations=1, observed_support_presentations=1,
        array_sha256={key[3:]:verifier.array_hash(value) for key,value in arrays.items()})
    trace = dict(seed=1, records=[row], calls=1, query_presentations=2, support_presentations=1, observed_support_presentations=1)
    return trace, arrays, dict(eval_size=2, batch_size=8)


@pytest.mark.parametrize("fault", ("metric", "array", "context", "extra", "count", "query"))
def test_trace_scores_arrays_support_and_presentations_are_cross_bound(fault):
    trace, arrays, cfg = trace_fixture()
    verifier.trace_arrays(trace, arrays, cfg)
    if fault == "metric":
        trace["records"][0]["metrics"]["brier"] += .01
    elif fault == "array":
        arrays["e0_probabilities"][0, 0, 0] = .75
    elif fault == "context":
        trace["records"][0]["support_fingerprint"] = "b"*64
    elif fault == "extra":
        arrays["unreferenced"] = np.zeros(1)
    elif fault == "count":
        trace["support_presentations"] += 1
    else:
        trace["records"][0]["query_sha256"] = "b"*64
    with pytest.raises(ValueError):
        verifier.trace_arrays(trace, arrays, cfg)


def test_uniform_membership_is_independent_of_replay_draw_count_and_forks_keep_rng():
    first, second = verifier.Memory(17, 4), verifier.Memory(17, 4)
    previous = None
    for packet in range(30):
        query = verifier.sha(str(packet).encode())
        a, b = first.step(query, previous, 1), second.step(query, previous, 12)
        assert first.ids == second.ids and first.packets == second.packets
        if packet == 0:
            assert a == [0] and b == [0]*12
        previous = query
    fork = copy.deepcopy(second)
    assert fork.step("a"*64, previous, 12) == second.step("a"*64, previous, 12)
    assert fork.snapshot() == second.snapshot()
    fork.step("b"*64, "a"*64, 1)
    assert second.seen == 31 and fork.seen == 32


@pytest.fixture(scope="module")
def smoke_phase_fixture():
    source = Path(__file__).resolve().parents[1]/"reports/core_residual/smoke"
    if not (source/"audit.json").is_file():
        pytest.skip("completed locked smoke export is not present")
    cfg, summary, manifest = (verifier.read(source/name) for name in ("config.json", "summary.json", "manifest.json"))
    records, _ = verifier.read_records(source, cfg, manifest["identity"], summary)
    key = "conditional_16991/prefix_0/"
    return (records[key+"result.json"], verifier.plan(16991, cfg)[0],
        verifier.read_arrays(source/"arrays"/(key+"training.npz")), records[key+"evaluations.json"],
        verifier.read_arrays(source/"arrays"/(key+"evaluations.npz")), cfg, manifest["identity"])


@pytest.mark.parametrize("fault", ("future_support", "replay", "prequential", "marginal", "memory", "boundary_model", "curve_context"))
def test_causal_replay_prequential_and_evaluation_context_checks_on_locked_smoke(smoke_phase_fixture, fault):
    row, expected, training, trace, evaluation, cfg, identity = copy.deepcopy(smoke_phase_fixture)
    def check():
        return verifier.verify_phase(row, expected, training, trace, evaluation,
            verifier.Memory(row["seed"], cfg["memory_packets"]), None, 0, cfg, identity)
    check()
    if fault == "future_support":
        row["packets"][0]["support_sha256"] = row["packets"][0]["query_sha256"]
    elif fault == "replay":
        row["packets"][2]["replay_ids"][0] = 999
    elif fault == "prequential":
        row["packets"][0]["prequential_brier"] += .01
    elif fault == "marginal":
        row["marginal_count"][0] += 1
    elif fault == "memory":
        row["end_memory"]["seen"] += 1
    elif fault == "boundary_model":
        trace["records"][0]["model_sha256"] = "9"*64
    else:
        # Change the trace's support reference and update its array fingerprint:
        # independent support arithmetic alone would accept it, but the causal chain must not.
        index = row["curve"][-1]["metrics"]["trace"]
        name = f"e{index}_support_observations"
        evaluation[name][0, 0, 0, 0] ^= np.uint8(1)
        trace["records"][index]["array_sha256"]["support_observations"] = verifier.array_hash(evaluation[name])
        trace["records"][index]["support_fingerprint"] = verifier.experience_hash(*(
            evaluation[f"e{index}_support_"+field] for field in ("observations", "actions", "outcomes")))
    with pytest.raises(ValueError):
        check()


def record_fixture(folder):
    cfg, identity = config(), {"test":"raw-export"}
    paths, records, hashes = verifier.expected_paths(cfg), {}, {}
    for path, kind in paths.items():
        if kind == "phase":
            pieces = path.split("/")
            records[path] = dict(identity=identity, seed=16991, model="conditional",
                arm="prefix" if pieces[1].startswith("prefix_") else pieces[1], phase=pieces[-2])
        elif kind == "job":
            records[path] = dict(identity=identity)
        elif kind == "evaluation":
            records[path] = dict(seed=16991, records=[])
        if path in records:
            hashes[path] = verifier.sha(verifier.original_json(records[path]))
        elif kind.endswith("arrays"):
            target = folder/"arrays"/path
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, sample=np.zeros(1, dtype=np.float32))
            hashes[path] = verifier.sha(target.read_bytes())
        else:
            hashes[path] = "a"*64
    def write():
        with gzip.open(folder/"raw_results.jsonl.gz", "wt", encoding="utf-8") as handle:
            for path,row in records.items():
                if paths[path] == "phase":
                    handle.write(json.dumps(row)+"\n")
        with gzip.open(folder/"evaluation_records.jsonl.gz", "wt", encoding="utf-8") as handle:
            for path,row in records.items():
                if paths[path] == "evaluation":
                    handle.write(json.dumps(dict(path=path, record=row))+"\n")
        (folder/"job_records.json").write_text(json.dumps({path:row for path,row in records.items() if paths[path] == "job"}))
    write()
    summary = dict(array_archive_prefix="arrays", result_hashes=[dict(path=path, kind=kind, sha256=hashes[path]) for path,kind in paths.items()])
    return cfg, identity, records, summary, write


@pytest.mark.parametrize("fault", ("record", "missing", "duplicate", "array", "extra_array", "checkpoint_reference", "kind"))
def test_original_json_npz_and_checkpoint_provenance_survive_resealing(tmp_path, fault):
    cfg, identity, records, summary, write = record_fixture(tmp_path)
    verifier.read_records(tmp_path, cfg, identity, summary)
    phase = next(path for path in records if path.endswith("prefix_0/result.json"))
    if fault == "record":
        records[phase]["tampered"] = True
    elif fault == "missing":
        records.pop(phase)
    elif fault == "array":
        np.savez_compressed(next((tmp_path/"arrays").rglob("*.npz")), changed=np.zeros(1))
    elif fault == "extra_array":
        np.savez_compressed(tmp_path/"arrays"/"unexpected.npz", changed=np.zeros(1))
    elif fault == "checkpoint_reference":
        summary["result_hashes"] = [row for row in summary["result_hashes"] if row["kind"] != "checkpoint"]
    elif fault == "kind":
        summary["result_hashes"][0]["kind"] = "phase"
    write()
    if fault == "duplicate":
        with gzip.open(tmp_path/"raw_results.jsonl.gz", "at", encoding="utf-8") as handle:
            handle.write(json.dumps(records[phase])+"\n")
    seal(tmp_path)
    verifier.verify_artifacts(tmp_path)
    with pytest.raises(ValueError):
        verifier.read_records(tmp_path, cfg, identity, summary)


def audit_fixture():
    cfg, identity = config(), {"test":"tensor-audit"}
    paths = verifier.expected_paths(cfg)
    hashes = {path:"a"*64 for path in paths}
    integrity = {path:dict(sha256=value) for path,value in hashes.items()}
    records = {"conditional_16991/result.json":dict(parent_signature="b"*64,
        insertion={arm:dict(signature="c"*64) for arm in verifier.ARMS})}
    phases, phase_counts = [], {}
    totals = dict(forks_verified=3, fresh_initializations_verified=1,
        before_checkpoints=12, final_checkpoints=12, ordinary_packets=116,
        first_prequential_predictions_recomputed=12, evaluation_calls=0,
        evaluation_prediction_calls_recomputed=0, intermediate_evaluation_calls_arithmetic_only=0,
        query_presentations=0, support_presentations=0, observed_support_presentations=0)
    for phase in verifier.plan(16991, cfg):
        intermediate = phase["size"]//cfg["probe_every"]-1
        calls = 2*(7 if phase["cue"] is not None else 6)+intermediate+2+(5 if phase["phase"] == "return" else 0)
        work = dict(calls=calls, query_presentations=calls*64, support_presentations=calls*8, observed_support_presentations=calls*8)
        phase_counts["conditional_16991/"+phase["relative"]+"/result.json"] = work
        coverage = dict(evaluation_calls=calls, evaluation_prediction_calls_recomputed=calls-intermediate,
            intermediate_evaluation_calls_arithmetic_only=intermediate,
            **{key:work[key] for key in ("query_presentations", "support_presentations", "observed_support_presentations")})
        for key,value in coverage.items():
            totals[key] += value
        phases.append(dict(arm=phase["arm"], phase=phase["phase"], packets=phase["size"]//cfg["batch_size"],
            result_sha256="a"*64, before_sha256="a"*64, checkpoint_sha256="a"*64, training_sha256="a"*64,
            evaluations_json_sha256="a"*64, evaluations_npz_sha256="a"*64,
            causal_stream_and_replay_verified=True, exact_full_state_chain_verified=True,
            trainability_adam_and_work_verified=True, first_prequential_prediction_recomputed=True, **coverage))
    check = dict(seed=16991, model="conditional", result_sha256="a"*64, parent_signature="b"*64,
        insertion={arm:"c"*64 for arm in verifier.ARMS}, phases=phases, **totals)
    audit = dict(passed=True, identity=identity, jobs=1, manifest_sha256="e"*64, audit_source_sha256="f"*64,
        checks=[check], result_hashes=[dict(path=path, kind=kind, sha256=hashes[path]) for path,kind in paths.items()], **totals)
    return audit, records, hashes, cfg, identity, phase_counts, integrity, "e"*64, dict(files={"scripts/audit_core_residual.py":"f"*64})


@pytest.mark.parametrize("fault", ("job", "fork", "checkpoint", "phase", "intermediate", "assertion", "manifest", "source", "donor"))
def test_tensor_audit_coverage_and_exact_checkpoint_binding_cannot_be_relabelled(fault):
    arguments = audit_fixture()
    audit = arguments[0]
    assert verifier.verify_audit(*arguments)["ordinary_packets"] == 116
    if fault == "job":
        audit["checks"] = []
    elif fault == "fork":
        audit["forks_verified"] -= 1
    elif fault == "checkpoint":
        audit["checks"][0]["phases"][0]["checkpoint_sha256"] = "9"*64
    elif fault == "phase":
        audit["checks"][0]["phases"].pop()
    elif fault == "intermediate":
        audit["checks"][0]["phases"][0]["evaluation_prediction_calls_recomputed"] += 1
    elif fault == "assertion":
        audit["checks"][0]["phases"][0]["trainability_adam_and_work_verified"] = False
    elif fault == "manifest":
        audit["manifest_sha256"] = "8"*64
    elif fault == "source":
        audit["audit_source_sha256"] = "7"*64
    else:
        audit["checks"][0]["insertion"]["separate"] = "6"*64
    with pytest.raises(ValueError):
        verifier.verify_audit(*arguments)


@pytest.mark.parametrize("fault", ("partial", "copied_script", "zip", "executed_source"))
def test_separate_verifier_seal_binds_executed_code_and_tests(tmp_path, fault):
    assert verifier.verify_packaging(tmp_path) is False
    source_name, test_name = "scripts/verify_core_residual_archive.py", "tests/test_core_residual_archive.py"
    source, tests = Path(verifier.__file__).read_bytes(), b"test contents"
    def package(contents):
        (tmp_path/"verifier_lock.json").write_text(json.dumps(dict(files={source_name:verifier.sha(contents), test_name:verifier.sha(tests)})))
        (tmp_path/"verify_archive.py").write_bytes(contents)
        with zipfile.ZipFile(tmp_path/"verifier_source.zip", "w") as archive:
            archive.writestr(source_name, contents)
            archive.writestr(test_name, tests)
    package(source)
    assert verifier.verify_packaging(tmp_path)
    if fault == "partial":
        (tmp_path/"verifier_source.zip").unlink()
    elif fault == "copied_script":
        (tmp_path/"verify_archive.py").write_bytes(b"altered")
    elif fault == "zip":
        with zipfile.ZipFile(tmp_path/"verifier_source.zip", "w") as archive:
            archive.writestr(source_name, source+b"# changed\n")
            archive.writestr(test_name, tests)
    else:
        package(source+b"# altered everywhere except executing verifier\n")
    with pytest.raises(ValueError):
        verifier.verify_packaging(tmp_path)


def test_no_pickle_or_live_scientific_imports(tmp_path):
    np.savez_compressed(tmp_path/"unsafe.npz", value=np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        verifier.read_arrays(tmp_path/"unsafe.npz")
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert all(name.split(".")[0] in {*sys.stdlib_module_names, "numpy"} for name in imports if name)
    assert not {"torch", "pickle"}.intersection(imports)


def test_complete_locked_smoke_portably_and_resealed_wrong_decision(tmp_path):
    # Optional integration data; the independent synthetic tests above do not need a fitted run.
    source = Path(__file__).resolve().parents[1]/"reports/core_residual/smoke"
    if not (source/"audit.json").is_file():
        pytest.skip("completed locked smoke export is not present")
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    seal(tmp_path)
    result = verifier.verify(tmp_path)
    assert result["total_phases"] == 24 and result["tensor_audit_coverage"]["evaluation_calls"] == 416
    assert result["tensor_audit_coverage"]["intermediate_evaluation_calls_arithmetic_only"] == 34
    summary = verifier.read(tmp_path/"summary.json")
    summary["screen"]["conditional"]["passed"] = True
    (tmp_path/"summary.json").write_bytes(verifier.original_json(summary))
    seal(tmp_path)
    verifier.verify_artifacts(tmp_path)
    with pytest.raises(ValueError, match="screen/conditional.passed"):
        verifier.verify(tmp_path)
