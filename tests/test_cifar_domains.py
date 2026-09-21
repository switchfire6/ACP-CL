"""Leakage, pairing, provenance, and deterministic transform checks, offline only."""

import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acp_cl.cifar_domains import _transform_domain, build_cifar10_domains


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _source(per_class=18):
    # Independently sampled pixels include spatial structure for blur checks.
    rng = np.random.default_rng(9041)
    data = rng.integers(0, 256, (10 * per_class, 32, 32, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(10), per_class)
    # Interleave classes to catch assumptions that official indices are grouped.
    order = rng.permutation(len(labels))
    # torchvision exposes a noncontiguous NHWC view of underlying NCHW storage.
    images = np.ascontiguousarray(data[order].transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
    assert not images.flags.c_contiguous
    return SimpleNamespace(data=images, targets=labels[order].tolist(), train=True)


def _config(**updates):
    result = {
        "dataset": "cifar10_domains", "n_experiences": 6, "train_per_class": 2,
        "validation_reserved_per_class": 4, "validation_per_class": 2,
        "test_per_class": 0, "regime": "recurring", "download": False,
    }
    result.update(updates)
    return result


def _index_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _assert_same_stream(first, second):
    assert first.metadata == second.metadata
    assert first.class_order == second.class_order
    for left, right in zip(first.experiences, second.experiences, strict=True):
        for split in ("train", "validation", "test"):
            for x, y in zip(getattr(left, split).tensors, getattr(right, split).tensors, strict=True):
                assert torch.equal(x, y)


def test_single_arrivals_cover_unique_source_ids_and_never_reserved_validation():
    source = _source()
    stream = build_cifar10_domains(_config(), 17, source=source)
    metadata = stream.metadata
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    indices = metadata["split_indices"]
    arrivals = metadata["current_arrival_ids_by_experience"]
    flattened = [value for ids in arrivals for value in ids]
    assert flattened == indices["train"]
    assert len(flattened) == len(set(flattened)) == 120
    assert len(indices["validation"]) == 20
    assert len(indices["validation_reserved"]) == 40
    assert len(indices["validation_unused"]) == 20
    assert len(indices["unused_train"]) == 20
    train, reserved, unused = (set(indices[key]) for key in ("train", "validation_reserved", "unused_train"))
    active, inactive = (set(indices[key]) for key in ("validation", "validation_unused"))
    assert train.isdisjoint(reserved) and train.isdisjoint(unused) and reserved.isdisjoint(unused)
    assert train | reserved | unused == set(range(len(source.targets)))
    assert active.isdisjoint(inactive) and active | inactive == reserved
    assert stream.num_classes == 10 and stream.class_order == list(range(10))
    assert stream.input_shape == (3, 32, 32)
    assert metadata["single_current_arrival_per_source_image"] is True
    for experience, ids in zip(stream.experiences, arrivals, strict=True):
        x, y = experience.train.tensors
        assert experience.classes == tuple(range(10))
        assert x.shape == (20, 3, 32, 32) and x.dtype == torch.uint8
        assert y.dtype == torch.long
        assert y.tolist() == np.asarray(source.targets)[ids].tolist()
        assert torch.bincount(y, minlength=10).tolist() == [2] * 10
        assert len(experience.test) == 0
        assert experience.test.tensors[0].shape == (0, 3, 32, 32)
    assert metadata["official_test_images_materialized"] is False


def test_recurring_and_stationary_pair_arrival_ids_labels_and_validation_base_ids():
    source = _source()
    recurring = build_cifar10_domains(_config(), 17, source=source)
    stationary = build_cifar10_domains(_config(regime="stationary"), 17, source=source)
    for key in ("current_arrival_ids_by_experience", "current_arrival_id_hashes_by_experience",
                "split_indices", "split_index_hashes", "source_fingerprint"):
        assert recurring.metadata[key] == stationary.metadata[key]
    assert recurring.metadata["domain_order"] == [0, 1, 2, 0, 1, 2]
    assert stationary.metadata["domain_order"] == [0] * 6
    assert recurring.metadata["signal_change_flags"] == [False, True, True, True, True, True]
    assert stationary.metadata["signal_change_flags"] == [False] * 6
    assert stationary.metadata["boundary_change_flags"] == [False] * 6
    for index, (left, right) in enumerate(zip(recurring.experiences, stationary.experiences, strict=True)):
        assert torch.equal(left.train.tensors[1], right.train.tensors[1])
        base = source.data[stationary.metadata["current_arrival_ids_by_experience"][index]]
        assert np.array_equal(right.train.tensors[0].permute(0, 2, 3, 1).numpy(), base)
        assert torch.equal(left.train.tensors[0], right.train.tensors[0]) == (index % 3 == 0)


def test_validation_reuses_same_dataset_per_domain_and_is_not_in_training():
    source = _source()
    recurring = build_cifar10_domains(_config(), 42, source=source)
    stationary = build_cifar10_domains(_config(regime="stationary"), 42, source=source)
    experiences = recurring.experiences
    for domain in range(3):
        assert experiences[domain].validation is experiences[domain + 3].validation
    assert len({id(exp.validation) for exp in experiences}) == 3
    assert len({id(exp.validation) for exp in stationary.experiences}) == 1
    assert len({id(exp.test) for exp in experiences}) == 1
    active = recurring.metadata["split_indices"]["validation"]
    raw = source.data[active]
    assert np.array_equal(experiences[0].validation.tensors[0].permute(0, 2, 3, 1).numpy(), raw)
    labels = np.asarray(source.targets)[active].tolist()
    for experience in experiences:
        assert experience.validation.tensors[1].tolist() == labels
        assert torch.bincount(experience.validation.tensors[1], minlength=10).tolist() == [2] * 10
    # A validation tensor's private storage cannot mutate its source or arrivals.
    original_source = source.data.copy()
    original_train = experiences[0].train.tensors[0].clone()
    experiences[0].validation.tensors[0].zero_()
    assert np.array_equal(source.data, original_source)
    assert torch.equal(experiences[0].train.tensors[0], original_train)


def test_validation_panel_size_and_horizon_do_not_change_existing_arrivals():
    source = _source()
    first = build_cifar10_domains(_config(validation_per_class=1), 17, source=source)
    expanded_panel = build_cifar10_domains(_config(validation_per_class=3), 17, source=source)
    short = build_cifar10_domains(_config(n_experiences=2, validation_per_class=1), 17, source=source)
    assert first.metadata["current_arrival_ids_by_experience"] == expanded_panel.metadata["current_arrival_ids_by_experience"]
    assert first.metadata["split_indices"]["validation_reserved"] == expanded_panel.metadata["split_indices"]["validation_reserved"]
    assert short.metadata["current_arrival_ids_by_experience"] == first.metadata["current_arrival_ids_by_experience"][:2]
    for label in range(10):
        small_ids = first.metadata["split_indices"]["validation"][label : label + 1]
        larger_ids = expanded_panel.metadata["split_indices"]["validation"][label * 3 : label * 3 + 3]
        assert set(small_ids).issubset(larger_ids)


def test_private_rngs_are_reproducible_and_do_not_touch_global_rngs():
    source = _source()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    first = build_cifar10_domains(_config(), 91, source=source)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    assert np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    try:
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        second = build_cifar10_domains(_config(), 91, source=source)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
    _assert_same_stream(first, second)
    other_seed = build_cifar10_domains(_config(), 92, source=source)
    assert first.metadata["source_fingerprint"] == other_seed.metadata["source_fingerprint"]
    assert first.metadata["current_arrival_ids_by_experience"] != other_seed.metadata["current_arrival_ids_by_experience"]


def test_source_and_index_fingerprints_match_documented_encoding_and_detect_changes():
    source = _source()
    stream = build_cifar10_domains(_config(), 17, source=source)
    metadata = stream.metadata
    description = metadata["source_description"]
    assert description["images_sha256"] == hashlib.sha256(source.data.tobytes()).hexdigest()
    assert description["labels_sha256"] == _index_hash(source.targets)
    canonical = json.dumps(description, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert metadata["source_fingerprint"] == hashlib.sha256(canonical.encode()).hexdigest()
    for split, ids in metadata["split_indices"].items():
        assert metadata["split_index_hashes"][split] == _index_hash(ids)
    for ids, digest in zip(metadata["current_arrival_ids_by_experience"], metadata["current_arrival_id_hashes_by_experience"], strict=True):
        assert digest == _index_hash(ids)
    changed_images = _source()
    changed_images.data[0, 0, 0, 0] ^= np.uint8(1)
    image_stream = build_cifar10_domains(_config(), 17, source=changed_images)
    assert image_stream.metadata["source_fingerprint"] != metadata["source_fingerprint"]
    assert image_stream.metadata["split_indices"] == metadata["split_indices"]
    changed_labels = _source()
    other_class = next(index for index, label in enumerate(changed_labels.targets) if label != changed_labels.targets[0])
    changed_labels.targets[0], changed_labels.targets[other_class] = changed_labels.targets[other_class], changed_labels.targets[0]
    label_stream = build_cifar10_domains(_config(), 17, source=changed_labels)
    assert label_stream.metadata["source_fingerprint"] != metadata["source_fingerprint"]


def test_loader_constructs_only_training_set_and_no_test_transform(monkeypatch, tmp_path):
    from torchvision import datasets

    calls = []

    class FakeCifar10:
        def __init__(self, root, train, download):
            calls.append((root, train, download))
            assert train is True, "Official test images/labels must never be requested"
            fixture = _source()
            self.data, self.targets, self.train = fixture.data, fixture.targets, train

    monkeypatch.setattr(datasets, "CIFAR10", FakeCifar10)
    stream = build_cifar10_domains(_config(root=tmp_path, download=True), 11)
    assert calls == [(str(tmp_path), True, True)]
    assert stream.metadata["source_injected"] is False
    assert all(len(experience.test) == 0 for experience in stream.experiences)


def test_data_dispatch_and_normalization_never_apply_stochastic_augmentation(monkeypatch):
    from acp_cl.data import build_stream, preprocess

    source = _source()
    monkeypatch.setattr("acp_cl.cifar_domains._load_training_source", lambda root, download: source)
    stream = build_stream(_config(), seed=29)
    raw = stream.experiences[0].train.tensors[0][:4]
    generator = torch.Generator().manual_seed(317)
    state = generator.get_state().clone()
    expected = (raw.float() / 255.0 - 0.5) / 0.5
    actual = preprocess(raw, "cifar10_domains", training=True, generator=generator)
    assert torch.equal(actual, expected)
    assert torch.equal(preprocess(raw, "cifar10_domains", training=False), expected)
    assert torch.equal(generator.get_state(), state)


def test_raw_training_file_hashes_are_recorded_without_consulting_test_list(tmp_path):
    source = _source()
    folder = tmp_path / "fixture"
    folder.mkdir()
    (folder / "train_batch").write_bytes(b"local training fixture bytes")
    (folder / "batches.meta").write_bytes(b"label names")

    class Fixture:
        data = source.data
        targets = source.targets
        train = True
        root = tmp_path
        base_folder = "fixture"
        train_list = [("train_batch", "declared-training-md5")]
        meta = {"filename": "batches.meta", "md5": "declared-metadata-md5"}

        @property
        def test_list(self):
            raise AssertionError("Builder must not inspect the test file list")

    stream = build_cifar10_domains(_config(), 1, source=Fixture())
    identities = stream.metadata["raw_source_identities"]["training_and_label_metadata_files"]
    assert [item["name"] for item in identities] == ["train_batch", "batches.meta"]
    for item in identities:
        raw = (folder / item["name"]).read_bytes()
        assert item["bytes"] == len(raw)
        assert item["sha256"] == hashlib.sha256(raw).hexdigest()
        assert item["declared_md5"].startswith("declared-")


def test_grayscale_has_exact_prespecified_rgb_mix_and_no_mutation():
    colors = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255], [50, 50, 50]], dtype=torch.uint8)
    raw = colors[:, :, None, None].expand(-1, -1, 3, 3).clone()
    original = raw.clone()
    transformed = _transform_domain(raw, "grayscale")
    expected = torch.tensor([[112, 61, 61], [120, 171, 120], [23, 23, 74], [50, 50, 50]], dtype=torch.uint8)
    assert torch.equal(transformed, expected[:, :, None, None].expand_as(raw))
    assert torch.equal(raw, original)
    assert _transform_domain(raw, "original") is raw


def test_gaussian_blur_reflects_edges_and_does_not_mix_color_channels():
    raw = torch.zeros(1, 3, 3, 3, dtype=torch.uint8)
    raw[0, 0, 1, 1] = 255
    original = raw.clone()
    transformed = _transform_domain(raw, "blur")
    # Reflection duplicates the center impulse at boundaries. Zero-padding or
    # replication would produce different edges and corners.
    expected_red = torch.tensor([[77, 63, 77], [63, 52, 63], [77, 63, 77]], dtype=torch.uint8)
    assert torch.equal(transformed[0, 0], expected_red)
    assert transformed[0, 1:].count_nonzero() == 0
    assert torch.equal(raw, original)
    assert torch.equal(_transform_domain(torch.full_like(raw, 127), "blur"), torch.full_like(raw, 127))


def test_chunked_transforms_preserve_order_and_match_per_image_results():
    raw = torch.from_numpy(_source(per_class=14).data).permute(0, 3, 1, 2).contiguous()
    for domain in ("grayscale", "blur"):
        all_images = _transform_domain(raw, domain)
        for index in (0, 127, 128, 139):
            assert torch.equal(all_images[index : index + 1], _transform_domain(raw[index : index + 1], domain))
        assert all_images.dtype == torch.uint8 and all_images.shape == raw.shape


@pytest.mark.parametrize("updates,match", [
    ({"dataset": "cifar100"}, "dataset"),
    ({"regime": "noise_only"}, "regime"),
    ({"n_experiences": 0}, "n_experiences"),
    ({"n_experiences": True}, "n_experiences"),
    ({"train_per_class": 1.5}, "train_per_class"),
    ({"train_per_class": 0}, "train_per_class"),
    ({"validation_reserved_per_class": -1}, "validation_reserved_per_class"),
    ({"validation_per_class": 0}, "validation_per_class"),
    ({"validation_per_class": 5}, "cannot exceed"),
    ({"test_per_class": 1}, "test_per_class"),
    ({"classes_per_experience": 2}, "classes_per_experience"),
    ({"n_domains": 2}, "n_domains"),
    ({"image_size": 16}, "image_size"),
    ({"download": "false"}, "download"),
    ({"root": None}, "root"),
    ({"colour_policy": "independent"}, "Unknown"),
])
def test_invalid_configs_fail_before_loading(monkeypatch, updates, match):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("Invalid config must fail before accessing data")

    monkeypatch.setattr("acp_cl.cifar_domains._load_training_source", forbidden_load)
    with pytest.raises(ValueError, match=match):
        build_cifar10_domains(_config(**updates), 1)


def test_infeasible_single_pass_and_default_production_counts_fail_explicitly():
    with pytest.raises(ValueError, match="current images cannot be repeated"):
        build_cifar10_domains(_config(n_experiences=8), 1, source=_source())
    with pytest.raises(ValueError, match="30 experiences x 150 train.*500.*5000"):
        build_cifar10_domains({}, 1, source=_source())


@pytest.mark.parametrize("fault", ["test_partition", "float_images", "wrong_shape", "missing_class", "float_labels", "wrong_length"])
def test_source_validation_rejects_wrong_partitions_shapes_or_labels(fault):
    source = _source()
    if fault == "test_partition":
        source.train = False
    elif fault == "float_images":
        source.data = source.data.astype(np.float32)
    elif fault == "wrong_shape":
        source.data = source.data[:, :16]
    elif fault == "missing_class":
        source.targets = [0 if label == 9 else label for label in source.targets]
    elif fault == "float_labels":
        source.targets = [float(label) for label in source.targets]
    else:
        source.targets.pop()
    with pytest.raises(ValueError):
        build_cifar10_domains(_config(), 1, source=source)


@pytest.mark.parametrize("seed", [True, 1.5, "2"])
def test_seed_must_be_an_integer(seed):
    with pytest.raises(ValueError, match="seed"):
        build_cifar10_domains(_config(), seed, source=_source())
