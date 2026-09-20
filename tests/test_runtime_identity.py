"""Runtime provenance must survive caches, checkpoints, and paired analysis."""

import copy
import json
import platform

import numpy as np
import pytest
import torch

from acp_cl import analysis, experiment
from acp_cl.data import build_stream
from test_experiment import tiny_config, lightweight_environment

# The shared fixture deliberately replaces environment(), not runtime_identity().
__all__ = ["tiny_config", "lightweight_environment"]


@pytest.fixture
def recorded_suite(tiny_config, tmp_path, monkeypatch):
    config = copy.deepcopy(tiny_config)
    config["scratch_reference"] = True
    monkeypatch.setattr(analysis, "plot_overview", lambda *args: None)
    experiment.run_suite(config, output=tmp_path, seeds=[19],
                         methods=["er", "acp"], device_name="cpu")
    return config, tmp_path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reject_side_effects(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("resume drift reached training, learner construction, or a manifest write")
    monkeypatch.setattr(experiment, "new_learner", unexpected)
    monkeypatch.setattr(experiment, "build_stream", unexpected)
    monkeypatch.setattr(experiment, "write_json", unexpected)


def alter_runtime(saved, change="different"):
    if change == "missing":
        saved.pop("runtime_fingerprint")
        saved.pop("runtime_sha256")
    else:
        saved["runtime_fingerprint"]["torch"] = "different-torch"
        saved["runtime_sha256"] = experiment.config_hash(saved["runtime_fingerprint"])


def alter_artifact(directory, location, change="different"):
    run_directory = directory / "er_seed19"
    if location == "checkpoint":
        file = run_directory / "checkpoint.pt"
        saved = torch.load(file, map_location="cpu", weights_only=False)
        alter_runtime(saved, change)
        torch.save(saved, file)
        (run_directory / "result.json").unlink()
    else:
        file = {"manifest": directory / "manifest.json",
                "result": run_directory / "result.json",
                "scratch": directory / "scratch_seed19.json"}[location]
        saved = read_json(file)
        alter_runtime(saved, change)
        experiment.write_json(file, saved)


def test_runtime_reads_real_versions_independently_of_environment(monkeypatch):
    def unexpected_gpu_query(*args, **kwargs):
        pytest.fail("CPU identity queried the CUDA device")
    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_gpu_query)
    identity = experiment.runtime_identity(torch.device("cpu"))
    expected = {"python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": str(np.__version__)}
    assert experiment.environment() == {"test_environment": True}
    assert identity["runtime_fingerprint"] == expected
    assert identity["runtime_sha256"] == experiment.config_hash(expected)
    assert identity == experiment.runtime_identity(torch.device("cpu"))
    assert identity["runtime_sha256"] == experiment.config_hash(dict(reversed(list(expected.items()))))


def test_cuda_identity_uses_requested_device_and_cpu_ignores_cuda_changes(monkeypatch):
    selected_devices = []
    def device_name(device):
        selected_devices.append(device)
        return "selected GPU"
    monkeypatch.setattr(torch.cuda, "get_device_name", device_name)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    cpu_identity = experiment.runtime_identity(torch.device("cpu"))
    cuda_identity = experiment.runtime_identity(torch.device("cuda:3"))
    assert selected_devices == [torch.device("cuda:3")]
    assert cuda_identity["runtime_fingerprint"] == {
        **cpu_identity["runtime_fingerprint"], "cuda_runtime": "12.8", "gpu": "selected GPU"}
    monkeypatch.setattr(torch.version, "cuda", "99.9")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "replacement GPU")
    assert experiment.runtime_identity(torch.device("cpu")) == cpu_identity
    assert experiment.runtime_identity(torch.device("cuda:3")) != cuda_identity


def test_suite_result_checkpoint_and_scratch_record_same_runtime(recorded_suite):
    _, directory = recorded_suite
    saved = [read_json(directory / "manifest.json"),
             read_json(directory / "scratch_seed19.json"),
             read_json(directory / "er_seed19/result.json"),
             torch.load(directory / "er_seed19/checkpoint.pt", map_location="cpu", weights_only=False)]
    expected = experiment.runtime_identity(torch.device("cpu"))
    for artifact in saved:
        assert {key: artifact[key] for key in expected} == expected


@pytest.mark.parametrize("location", ["manifest", "result", "checkpoint", "scratch"])
@pytest.mark.parametrize("change", ["different", "missing"])
def test_resume_preflight_refuses_artifact_drift_before_any_write(
        recorded_suite, monkeypatch, location, change):
    config, directory = recorded_suite
    alter_artifact(directory, location, change)
    manifest_before = (directory / "manifest.json").read_bytes()
    reject_side_effects(monkeypatch)
    with pytest.raises(ValueError, match="runtime"):
        experiment.run_suite(config, output=directory, seeds=[19], methods=["er"],
                             device_name="cpu", resume=True)
    assert (directory / "manifest.json").read_bytes() == manifest_before


@pytest.mark.parametrize("location", ["result", "checkpoint", "scratch"])
def test_direct_resume_checks_runtime_before_learner_construction(recorded_suite, monkeypatch, location):
    config, directory = recorded_suite
    stream = build_stream(config["data"], 19)
    alter_artifact(directory, location)
    reject_side_effects(monkeypatch)
    with pytest.raises(ValueError, match="runtime"):
        if location == "scratch":
            experiment.scratch_reference(config, stream, 19, torch.device("cpu"), directory, resume=True)
        else:
            experiment.run_one(config, stream, "er", 19, torch.device("cpu"), directory, resume=True)


@pytest.mark.parametrize("library", ["python", "torch", "numpy"])
def test_resume_refuses_actual_library_drift(recorded_suite, monkeypatch, library):
    config, directory = recorded_suite
    manifest_before = (directory / "manifest.json").read_bytes()
    if library == "python":
        monkeypatch.setattr(experiment.platform, "python_version", lambda: "99.0.0")
    elif library == "torch":
        monkeypatch.setattr(torch, "__version__", "99.0.0")
    else:
        monkeypatch.setattr(np, "__version__", "99.0.0")
    reject_side_effects(monkeypatch)
    with pytest.raises(ValueError, match="runtime"):
        experiment.run_suite(config, output=directory, seeds=[19], methods=["er"],
                             device_name="cpu", resume=True)
    assert (directory / "manifest.json").read_bytes() == manifest_before


@pytest.mark.parametrize("field", ["cuda_runtime", "gpu"])
def test_resume_refuses_cuda_runtime_or_gpu_drift_without_gpu_training(
        tiny_config, tmp_path, monkeypatch, field):
    device = torch.device("cuda:2")
    monkeypatch.setattr(experiment, "resolve_device", lambda name: device)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda selected: "GPU A")
    manifest = {"config_sha256": experiment.config_hash(tiny_config),
                "source_sha256": experiment.source_hash(), "execution_device": str(device),
                **experiment.runtime_identity(device)}
    experiment.write_json(tmp_path / "manifest.json", manifest)
    manifest_before = (tmp_path / "manifest.json").read_bytes()
    if field == "cuda_runtime":
        monkeypatch.setattr(torch.version, "cuda", "12.9")
    else:
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda selected: "GPU B")
    reject_side_effects(monkeypatch)
    with pytest.raises(ValueError, match="runtime"):
        experiment.run_suite(tiny_config, output=tmp_path, seeds=[19], methods=["er"],
                             device_name="cuda:2", resume=True)
    assert (tmp_path / "manifest.json").read_bytes() == manifest_before


@pytest.mark.parametrize("field", ["python", "torch", "numpy", "cuda_runtime", "gpu"])
def test_analysis_refuses_mixed_runtime_fingerprints(recorded_suite, field):
    _, directory = recorded_suite
    for method in ("er", "acp"):
        file = directory / f"{method}_seed19/result.json"
        result = read_json(file)
        if field in ("cuda_runtime", "gpu"):
            result["execution_device"] = "cuda"
            result["runtime_fingerprint"].update(cuda_runtime="12.8", gpu="GPU A")
        if method == "er":
            result["runtime_fingerprint"][field] = "different"
        result["runtime_sha256"] = experiment.config_hash(result["runtime_fingerprint"])
        experiment.write_json(file, result)
    with pytest.raises(ValueError, match="pool"):
        analysis.analyze(directory)


@pytest.mark.parametrize("change", ["mixed_legacy", "invalid_hash", "missing_hash", "missing_fingerprint"])
def test_analysis_refuses_partial_or_corrupt_runtime_identity(recorded_suite, change):
    _, directory = recorded_suite
    file = directory / "er_seed19/result.json"
    result = read_json(file)
    if change == "mixed_legacy":
        alter_runtime(result, "missing")
    elif change == "invalid_hash":
        result["runtime_sha256"] = "incorrect"
    elif change == "missing_hash":
        result.pop("runtime_sha256")
    else:
        result.pop("runtime_fingerprint")
    experiment.write_json(file, result)
    with pytest.raises(ValueError, match="pool"):
        analysis.analyze(directory)


def test_analysis_accepts_uniform_legacy_artifacts(recorded_suite):
    _, directory = recorded_suite
    for file in directory.glob("*/result.json"):
        result = read_json(file)
        alter_runtime(result, "missing")
        experiment.write_json(file, result)
    summary = analysis.analyze(directory)
    assert len(summary["methods"]) == 2
    assert summary["runtime_sha256"] is None


@pytest.mark.parametrize("field", ["python", "cuda_runtime"])
def test_analysis_legacy_fallback_rejects_python_and_cuda_drift(recorded_suite, field):
    _, directory = recorded_suite
    for method in ("er", "acp"):
        file = directory / f"{method}_seed19/result.json"
        result = read_json(file)
        alter_runtime(result, "missing")
        if field == "cuda_runtime":
            result["execution_device"] = "cuda"
        result["environment"][field] = "different" if method == "er" else "same"
        experiment.write_json(file, result)
    with pytest.raises(ValueError, match="pool"):
        analysis.analyze(directory)


def test_analysis_cpu_pooling_ignores_irrelevant_gpu_metadata(recorded_suite):
    _, directory = recorded_suite
    file = directory / "er_seed19/result.json"
    result = read_json(file)
    result["environment"].update(cuda_runtime="different CUDA installation", gpu="different GPU")
    experiment.write_json(file, result)
    summary = analysis.analyze(directory)
    assert summary["runtime_sha256"] == result["runtime_sha256"]


def test_analysis_does_not_require_the_training_runtime_on_analysis_host(recorded_suite, monkeypatch):
    _, directory = recorded_suite
    expected = read_json(directory / "er_seed19/result.json")["runtime_sha256"]
    def unexpected(*args, **kwargs):
        pytest.fail("analysis tried to compare artifacts to its own execution runtime")
    monkeypatch.setattr(experiment, "runtime_identity", unexpected)
    assert analysis.analyze(directory)["runtime_sha256"] == expected
