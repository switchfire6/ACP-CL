"""Audit pairing, isolation, and final-checkpoint provenance invariants."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from acp_cl.models import make_model
from acp_cl.shapes import build_shapes_stream
from scripts import evaluate_shape_shortcuts as audit


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def stream():
    return build_shapes_stream({
        "dataset": "procedural_shapes", "regime": "recurring", "n_domains": 2,
        "n_experiences": 2, "image_size": 16, "train_per_class": 2,
        "validation_per_class": 1, "test_per_class": 0,
    }, 101)


def test_audit_images_are_private_reproducible_fresh_and_label_balanced(stream):
    torch_before = torch.get_rng_state().clone()
    numpy_before = copy.deepcopy(np.random.get_state())
    first, digest = audit.make_audit_images(stream.metadata, 101, samples_per_class=2)
    second, other_digest = audit.make_audit_images(stream.metadata, 101, samples_per_class=2)
    assert digest == other_digest
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    prior = {image.numpy().tobytes() for experience in stream.experiences
             for split in (experience.train, experience.validation) for image in split.tensors[0]}
    for a, b in zip(first, second):
        assert a.original.dtype == a.uncorrelated.dtype == torch.uint8
        assert torch.bincount(a.labels, minlength=4).tolist() == [2] * 4
        torch.testing.assert_close(a.original, b.original)
        torch.testing.assert_close(a.uncorrelated, b.uncorrelated)
        assert all(image.numpy().tobytes() not in prior for image in a.original)
    _, changed_digest = audit.make_audit_images(stream.metadata, 101, samples_per_class=2, audit_seed=1)
    assert digest != changed_digest


def test_palette_counterfactual_exactly_preserves_each_samples_geometry(stream, monkeypatch):
    renderer = audit._render_sample
    rendered = []

    def record(*args):
        result = renderer(*args)
        rendered.append(result)
        return result

    monkeypatch.setattr(audit, "_render_sample", record)
    audit.make_audit_images(stream.metadata, 101, samples_per_class=2)
    for a, b in zip(rendered[::2], rendered[1::2]):
        assert a.center == b.center and a.angle == b.angle and a.area_side == b.area_side
        np.testing.assert_array_equal(a.alpha, b.alpha)


def test_already_uncorrelated_domain_yields_identical_control_images(stream):
    metadata = copy.deepcopy(stream.metadata)
    for domain in metadata["domain_parameters"].values():
        domain["color_correlation"] = 0
    domains, _ = audit.make_audit_images(metadata, 101, samples_per_class=2)
    for domain in domains:
        torch.testing.assert_close(domain.original, domain.uncorrelated, rtol=0, atol=0)
    model = make_model("mlp", (3, 16, 16), 4, width=8)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    state = torch.get_rng_state().clone()
    metrics = audit.evaluate_model(model, domains, torch.device("cpu"), batch_size=3)
    assert metrics["accuracy_drop"] == 0
    assert metrics["prediction_agreement"] == 1
    assert metrics["original_only_correct"] == metrics["uncorrelated_only_correct"] == 0
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def _completed_run(directory, stream, method="er_recycle"):
    config = {
        "data": {"dataset": "procedural_shapes", "regime": "recurring", "n_domains": 2,
                 "n_experiences": 2, "image_size": 16},
        "model": "cnn", "width": 8,
    }
    result = {
        "config": config, "config_sha256": audit._config_hash(config),
        "source_sha256": audit._source_hash(), "execution_device": "cpu",
        "method": method, "seed": 101, "class_order": [0, 1, 2, 3],
        "stream_metadata": stream.metadata,
    }
    model = make_model("cnn", (3, 16, 16), 4, width=8)
    checkpoint = {field: result[field] for field in audit.IDENTITY_FIELDS}
    checkpoint.update({"completed": 2, "learner": {"method": method, "model": model.state_dict()}})
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "checkpoint.pt"
    torch.save(checkpoint, path)
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result, path


@pytest.mark.parametrize("mismatch", ["seed", "source_sha256", "completed", "config_payload"])
def test_only_matching_final_checkpoint_is_evaluated(stream, tmp_path, mismatch):
    result, path = _completed_run(tmp_path, stream)
    if mismatch == "config_payload":
        result["config"]["width"] = 16
    else:
        checkpoint = torch.load(path, weights_only=False)
        checkpoint[mismatch] = {"seed": 202, "source_sha256": "different-source", "completed": 1}[mismatch]
        torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="mismatch|hash|configured experiences"):
        audit.load_final_model(result, path, torch.device("cpu"))


def test_complete_posthoc_audit_reuses_images_and_preserves_checkpoints(stream, tmp_path):
    directory, output = tmp_path / "runs", tmp_path / "audit"
    methods = ("er_recycle", "acp_v2")
    hashes = {}
    for method in methods:
        _, path = _completed_run(directory / f"{method}_seed101", stream, method)
        hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = audit.run_audit(directory, output, methods, samples_per_class=2, batch_size=3)
    assert len(result["runs"]) == 2
    assert len({run["audit_images_sha256"] for run in result["runs"]}) == 1
    assert all(run["audit_source_matches_training"] for run in result["runs"])
    for run in result["runs"]:
        assert run["examples"] == 16
        assert run["examples"] == sum(run[key] for key in (
            "both_correct", "original_only_correct", "uncorrelated_only_correct", "both_wrong"))
        assert run["accuracy_drop"] == pytest.approx(run["original_accuracy"] - run["uncorrelated_accuracy"])
    assert (output / "shortcut_audit.json").exists()
    assert "not confirmatory" in (output / "shortcut_audit.md").read_text()
    for path, expected in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
