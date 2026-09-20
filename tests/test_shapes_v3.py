"""Color-curriculum controls and exact pairing of non-palette latent draws."""

import copy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

import acp_cl.shapes as shapes


@pytest.fixture
def config():
    return {
        "dataset": "procedural_shapes", "regime": "recurring", "n_experiences": 4,
        "n_domains": 2, "image_size": 16, "train_per_class": 3,
        "validation_per_class": 2, "test_per_class": 1,
    }


def _tensor_digest(stream):
    digest = hashlib.sha256()
    for experience in stream.experiences:
        for split in ("train", "validation", "test"):
            for tensor in getattr(experience, split).tensors:
                digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("regime,image_hash,metadata_hash", [
    ("recurring", "717cfed50d6399b69c88bb5a51b3e76659f093bcaa6260ea144b716ad5a460b4",
     "8afdc68fd309753015bd28d90541d3573ef2d99d6b4404f430a98e463e414ad2"),
    ("stationary", "5788c294eb370c18d73fe2832d104864a47119f597c32ba11cb607ff43fe71b0",
     "c3e38f04a88ce26691e25f53ae93d2751d9bfe7c21940932b1d736164b8a5b01"),
    ("noise_only", "db9ad968735a5c094ebdf76a0998e59e0bf3a46c888451f106e33bb1da1c1935",
     "9dc273844f8ad218ca9f90e064822718342459b325d62c49b4a7c9aafac216bb"),
])
def test_original_policy_preserves_pre_v3_image_and_metadata_goldens(regime, image_hash, metadata_hash):
    # Captured from the frozen version-1 generator before adding color policies.
    config = {
        "dataset": "procedural_shapes", "regime": regime, "n_experiences": 3,
        "n_domains": 2, "image_size": 16, "train_per_class": 2,
        "validation_per_class": 1, "test_per_class": 1,
    }
    for explicit in (False, True):
        if explicit:
            config["color_policy"] = "original"
        stream = shapes.build_shapes_stream(config, 123)
        assert _tensor_digest(stream) == image_hash
        metadata = json.dumps(stream.metadata, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(metadata).hexdigest() == metadata_hash
        assert stream.metadata["generator_version"] == 1
        assert "color_policy" not in stream.metadata


def test_fixed_palette_draws_preserve_geometry_noise_and_subsequent_samples():
    base = replace(
        shapes._make_domains(7, 1)[0],
        background_color=(0.10, 0.11, 0.12),
        foreground_palette=((0.25, 0.30, 0.35), (0.40, 0.45, 0.50),
                            (0.55, 0.60, 0.65), (0.65, 0.45, 0.30)),
        sensor_noise_amplitude=0.01,
    )
    biased = replace(base, color_correlation=0.95)
    independent = replace(base, color_correlation=0)
    a_rng, b_rng = np.random.default_rng(19), np.random.default_rng(19)
    changed = 0
    for index in range(20):
        label = index % 4
        a = shapes._render_sample(a_rng, label, biased, 24, fixed_color_draws=True)
        b = shapes._render_sample(b_rng, label, independent, 24, fixed_color_draws=True)
        assert (a.center, a.angle, a.area_side) == (b.center, b.angle, b.area_side)
        np.testing.assert_array_equal(a.alpha, b.alpha)
        assert a_rng.bit_generator.state == b_rng.bit_generator.state
        # All background pixels are bit identical. On the foreground, the entire
        # change is explained by alpha times the chosen palette difference, up
        # to uint8 rounding. This also checks shared texture/jitter/noise draws.
        np.testing.assert_array_equal(a.image[:, a.alpha == 0], b.image[:, b.alpha == 0])
        palette_delta = np.asarray(base.foreground_palette[a.palette_index]) - np.asarray(
            base.foreground_palette[b.palette_index])
        expected = 255 * palette_delta[:, None, None] * a.alpha[None]
        observed = a.image.astype(np.float64) - b.image.astype(np.float64)
        assert np.abs(observed - expected).max() <= 1.000001
        changed += a.palette_index != b.palette_index
    assert changed > 0


def test_independent_palette_sequence_does_not_depend_on_shape_label():
    domain = replace(shapes._make_domains(31, 1)[0], color_correlation=0)
    sequences = []
    for label in range(4):
        rng = np.random.default_rng(45)
        sequences.append([
            shapes._render_sample(rng, label, domain, 16, fixed_color_draws=True).palette_index
            for _ in range(64)
        ])
    assert all(sequence == sequences[0] for sequence in sequences[1:])
    assert set(sequences[0]) == {0, 1, 2, 3}
    # The deliberately broad bound checks that all four palette values are used;
    # independence itself is established by the exact equal-label sequences.
    assert min(np.bincount(sequences[0], minlength=4)) >= 5


def test_early_bias_schedule_and_every_held_out_render_are_independent(config, monkeypatch):
    actual_correlations = []
    renderer = shapes._render_sample

    def record(rng, label, domain, image_size, **kwargs):
        actual_correlations.append((domain.color_correlation, kwargs.get("fixed_color_draws")))
        return renderer(rng, label, domain, image_size, **kwargs)

    monkeypatch.setattr(shapes, "_render_sample", record)
    stream = shapes.build_shapes_stream(dict(config, color_policy="early_biased", bias_experiences=2), 101)
    expected = []
    for experience in range(4):
        expected.extend([(0.95 if experience < 2 else 0.0, True)] * 12)
        expected.extend([(0.0, True)] * 8)
        expected.extend([(0.0, True)] * 4)
    assert actual_correlations == expected
    metadata = stream.metadata
    assert metadata["generator_version"] == 2
    assert metadata["color_policy"] == "early_biased"
    assert metadata["train_color_correlation_by_experience"] == [0.95, 0.95, 0, 0]
    assert metadata["evaluation_color_correlation_by_experience"] == [0, 0, 0, 0]
    assert metadata["evaluation_color_splits"] == ["validation", "test"]
    assert metadata["train_color_change_flags"] == [False, False, True, False]
    assert all(domain["color_correlation"] == 0 for domain in metadata["domain_parameters"].values())
    assert all(len(experience.train.tensors) == 2 for experience in stream.experiences)


@pytest.mark.parametrize("regime", ["recurring", "stationary", "noise_only"])
def test_curricula_have_identical_evaluation_and_identical_late_training(config, regime):
    config["regime"] = regime
    independent = shapes.build_shapes_stream(dict(config, color_policy="independent"), 11)
    early = shapes.build_shapes_stream(dict(config, color_policy="early_biased", bias_experiences=2), 11)
    assert independent.metadata["split_seeds"] == early.metadata["split_seeds"]
    assert independent.metadata["domain_order"] == early.metadata["domain_order"]
    assert independent.metadata["signal_change_flags"] == early.metadata["signal_change_flags"]
    for index, (a, b) in enumerate(zip(independent.experiences, early.experiences)):
        for split in ("validation", "test"):
            for ta, tb in zip(getattr(a, split).tensors, getattr(b, split).tensors):
                torch.testing.assert_close(ta, tb, rtol=0, atol=0)
        torch.testing.assert_close(a.train.tensors[1], b.train.tensors[1], rtol=0, atol=0)
        if index >= 2:
            torch.testing.assert_close(a.train.tensors[0], b.train.tensors[0], rtol=0, atol=0)
        else:
            assert not torch.equal(a.train.tensors[0], b.train.tensors[0])


def test_zero_bias_equals_independent_and_default_bias_is_eight(config):
    independent = shapes.build_shapes_stream(dict(config, color_policy="independent"), 21)
    zero_bias = shapes.build_shapes_stream(dict(config, color_policy="early_biased", bias_experiences=0), 21)
    assert _tensor_digest(independent) == _tensor_digest(zero_bias)
    default_bias = shapes.build_shapes_stream(dict(config, color_policy="early_biased"), 21)
    assert default_bias.metadata["bias_experiences"] == 8
    assert default_bias.metadata["observed_biased_training_experiences"] == 4
    assert default_bias.metadata["train_color_correlation_by_experience"] == [0.95] * 4


def test_split_independence_and_global_rng_isolation_for_new_policy(config):
    config.update(color_policy="early_biased", bias_experiences=2)
    torch_before = torch.get_rng_state().clone()
    numpy_before = copy.deepcopy(np.random.get_state())
    first = shapes.build_shapes_stream(config, 43)
    second = shapes.build_shapes_stream(dict(config, train_per_class=7), 43)
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    after = np.random.get_state()
    assert numpy_before[0] == after[0]
    np.testing.assert_array_equal(numpy_before[1], after[1])
    assert numpy_before[2:] == after[2:]
    for a, b in zip(first.experiences, second.experiences):
        for split in ("validation", "test"):
            torch.testing.assert_close(getattr(a, split).tensors[0], getattr(b, split).tensors[0], rtol=0, atol=0)
    seen = set()
    for experience in first.experiences:
        for split in (experience.train, experience.validation, experience.test):
            for image in split.tensors[0]:
                pixels = image.numpy().tobytes()
                assert pixels not in seen
                seen.add(pixels)


@pytest.mark.parametrize("policy", [None, True, "", "Independent", "biased", 0])
def test_invalid_policy_is_rejected(config, policy):
    with pytest.raises(ValueError, match="color_policy"):
        shapes.build_shapes_stream(dict(config, color_policy=policy), 1)


@pytest.mark.parametrize("bias", [True, False, -1, 1.0, 1.5, "2", None])
def test_bias_horizon_requires_nonnegative_integer(config, bias):
    with pytest.raises(ValueError, match="bias_experiences"):
        shapes.build_shapes_stream(dict(config, color_policy="early_biased", bias_experiences=bias), 1)


@pytest.mark.parametrize("policy", ["original", "independent"])
def test_unused_bias_configuration_is_rejected(config, policy):
    with pytest.raises(ValueError, match="only used"):
        shapes.build_shapes_stream(dict(config, color_policy=policy, bias_experiences=2), 1)
