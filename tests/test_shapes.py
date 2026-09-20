"""Distribution and rendering constraints for the procedural domain stream."""

from dataclasses import asdict, replace
import copy
import math

import numpy as np
import pytest
import torch

from acp_cl.shapes import SHAPE_NAMES, _make_domains, _render_sample, build_shapes_stream


@pytest.fixture
def config():
    return {
        "dataset": "procedural_shapes", "n_experiences": 5, "n_domains": 2,
        "regime": "recurring", "train_per_class": 3, "validation_per_class": 2,
        "test_per_class": 1, "image_size": 24,
    }


def test_fixed_four_labels_recurrence_and_tensor_contract(config):
    # Class-incremental settings must not silently change this target function.
    config["classes_per_experience"] = 17
    stream = build_shapes_stream(config, 5)
    assert stream.num_classes == 4
    assert stream.input_shape == (3, 24, 24)
    assert stream.class_order == [0, 1, 2, 3]
    assert stream.metadata["shape_names"] == list(SHAPE_NAMES)
    order = stream.metadata["domain_order"]
    assert order[0] != order[1]
    assert order[0] == order[2] == order[4]
    assert order[1] == order[3]
    assert stream.metadata["boundary_change_flags"] == [False, True, True, True, True]
    for experience in stream.experiences:
        assert experience.classes == (0, 1, 2, 3)
        for split, count in (("train", 3), ("validation", 2), ("test", 1)):
            dataset = getattr(experience, split)
            assert len(dataset.tensors) == 2  # No task/domain-ID input tensor.
            x, y = dataset.tensors
            assert x.dtype == torch.uint8 and x.shape == (4 * count, 3, 24, 24)
            assert y.dtype == torch.long
            assert torch.bincount(y, minlength=4).tolist() == [count] * 4


def test_reproducibility_and_global_rng_independence(config):
    np.random.seed(27)
    torch.manual_seed(31)
    numpy_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()
    a = build_shapes_stream(config, 9)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    np.random.seed(100)
    torch.manual_seed(200)
    b = build_shapes_stream(config, 9)
    assert a.metadata == b.metadata
    for ea, eb in zip(a.experiences, b.experiences):
        for name in ("train", "validation", "test"):
            for ta, tb in zip(getattr(ea, name).tensors, getattr(eb, name).tensors):
                torch.testing.assert_close(ta, tb, rtol=0, atol=0)


def test_all_splits_and_recurring_experiences_have_fresh_pixels(config):
    stream = build_shapes_stream(config, 11)
    seen = set()
    for experience in stream.experiences:
        for split in ("train", "validation", "test"):
            for image in getattr(experience, split).tensors[0]:
                pixels = image.numpy().tobytes()
                assert pixels not in seen
                seen.add(pixels)


def test_training_sample_count_does_not_change_validation_or_test(config):
    first = build_shapes_stream(config, 6)
    config["train_per_class"] = 7
    second = build_shapes_stream(config, 6)
    for a, b in zip(first.experiences, second.experiences):
        for split in ("validation", "test"):
            torch.testing.assert_close(getattr(a, split).tensors[0], getattr(b, split).tensors[0])


def test_stationary_control_has_no_changes_but_new_samples(config):
    config["regime"] = "stationary"
    stream = build_shapes_stream(config, 7)
    assert len(set(stream.metadata["domain_order"])) == 1
    assert not any(stream.metadata["boundary_change_flags"])
    assert not any(stream.metadata["signal_change_flags"])
    assert not torch.equal(stream.experiences[0].train.tensors[0], stream.experiences[1].train.tensors[0])


def test_noise_only_keeps_all_signal_parameters_and_labels_fixed(config):
    config["regime"] = "noise_only"
    stream = build_shapes_stream(config, 7)
    assert len(set(stream.metadata["signal_domain_order"])) == 1
    assert not any(stream.metadata["signal_change_flags"])
    assert any(stream.metadata["boundary_change_flags"])
    parameters = list(stream.metadata["domain_parameters"].values())
    without_noise = [dict(domain) for domain in parameters]
    amplitudes = [domain.pop("sensor_noise_amplitude") for domain in without_noise]
    assert without_noise[0] == without_noise[1]
    assert len(set(amplitudes)) == 2
    assert min(amplitudes) >= 0 and max(amplitudes) <= 0.12
    for experience in stream.experiences:
        assert experience.classes == (0, 1, 2, 3)


def test_single_domain_controls_are_identical_distributions_and_draws(config):
    config["n_domains"] = 1
    streams = []
    for regime in ("recurring", "stationary", "noise_only"):
        config["regime"] = regime
        streams.append(build_shapes_stream(config, 17))
    for a, b, c in zip(*(stream.experiences for stream in streams)):
        for split in ("train", "validation", "test"):
            torch.testing.assert_close(getattr(a, split).tensors[0], getattr(b, split).tensors[0])
            torch.testing.assert_close(getattr(a, split).tensors[0], getattr(c, split).tensors[0])


def test_noise_control_changes_only_bounded_sensor_perturbation():
    domain = _make_domains(3, 1)[0]
    clean = replace(domain, sensor_noise_amplitude=0)
    noisy = replace(domain, sensor_noise_amplitude=0.12)
    for label in range(4):
        a = _render_sample(np.random.default_rng(22), label, clean, 32)
        b = _render_sample(np.random.default_rng(22), label, noisy, 32)
        np.testing.assert_array_equal(a.alpha, b.alpha)
        assert a.center == b.center and a.angle == b.angle and a.area_side == b.area_side
        delta = np.abs(a.image.astype(np.int16) - b.image.astype(np.int16))
        assert delta.max() <= math.ceil(255 * 0.12)
        assert delta.mean() > 5


def test_domain_nuisances_and_label_color_associations_change():
    domains = _make_domains(32, 8)
    assert len({domain.background_texture for domain in domains}) == 4
    assert len({domain.label_palette for domain in domains}) == 4
    assert len({domain.background_color for domain in domains}) == 8
    assert all(0 < domain.color_correlation < 1 for domain in domains)
    # Palette colors stay separated from the background even when their exact
    # correlation with labels changes between domains.
    for domain in domains:
        foreground = np.asarray(domain.foreground_palette)
        background = np.asarray(domain.background_color)
        assert np.min(np.abs(foreground.mean(axis=1) - background.mean())) > 0.3
        assert len(set(domain.label_palette)) == 4


@pytest.mark.parametrize("image_size", [16, 32, 48])
def test_geometry_stays_in_frame_has_area_matching_and_varied_positions(image_size):
    domain = _make_domains(41, 1)[0]
    centers, sizes, angles = [], [], []
    for draw in range(20):
        expected_geometry = None
        for label in range(4):
            sample = _render_sample(np.random.default_rng(draw), label, domain, image_size)
            alpha = sample.alpha
            assert alpha.max() == 1 and alpha.min() == 0
            assert not alpha[0].any() and not alpha[-1].any()
            assert not alpha[:, 0].any() and not alpha[:, -1].any()
            # Discrete coverage follows the deliberately matched continuous area.
            assert abs(float(alpha.sum()) - sample.area_side**2) / sample.area_side**2 < 0.12
            geometry = (sample.center, sample.area_side, sample.angle, sample.placement_radius)
            if expected_geometry is None:
                expected_geometry = geometry
            else:
                assert geometry == expected_geometry  # Placement distribution is label independent.
        centers.append(sample.center)
        sizes.append(sample.area_side)
        angles.append(sample.angle)
    assert np.ptp(np.array(centers)[:, 0]) > image_size * 0.25
    assert np.ptp(np.array(centers)[:, 1]) > image_size * 0.25
    assert max(sizes) / min(sizes) > 1.5
    assert np.ptp(angles) > math.pi


def test_zero_test_split_and_invalid_configuration(config):
    config["test_per_class"] = 0
    stream = build_shapes_stream(config, 4)
    assert all(experience.test.tensors[0].shape == (0, 3, 24, 24) for experience in stream.experiences)
    assert all(len(experience.test) == 0 for experience in stream.experiences)
    for key, value in (("image_size", 8), ("n_domains", 0), ("regime", "label_permutation")):
        broken = dict(config, **{key: value})
        with pytest.raises(ValueError):
            build_shapes_stream(broken, 4)


def test_shape_domain_metadata_is_plain_json_data(config):
    import json

    stream = build_shapes_stream(config, 8)
    json.dumps(stream.metadata, allow_nan=False)
    assert asdict(_make_domains(8, 1)[0])["sensor_noise_amplitude"] == 0.01
