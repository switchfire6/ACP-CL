"""Invariants that guard the experiment's fairness and measurement contracts."""

import math

import numpy as np
import pytest
import torch

from acp_cl.data import CIFAR100_MEAN, CIFAR100_STD, build_stream, preprocess
from acp_cl.metrics import (
    continual_metrics,
    normalized_auc,
    paired_bootstrap,
    representation_metrics,
)
from acp_cl.models import make_model
from acp_cl.replay import ReservoirBuffer


def _assert_equal_batches(a, b):
    assert a is not None and b is not None
    torch.testing.assert_close(a.x, b.x)
    torch.testing.assert_close(a.y, b.y)
    if a.logits is None:
        assert b.logits is None
    else:
        torch.testing.assert_close(a.logits, b.logits)


def test_reservoir_membership_independent_of_training_and_monitor_sampling():
    first = ReservoirBuffer(17, seed=12)
    second = ReservoirBuffer(17, seed=12)
    for offset in range(0, 100, 5):
        values = torch.arange(offset, offset + 5)
        first.add(values[:, None].to(torch.uint8), values)
        second.add(values[:, None].to(torch.uint8), values)
        first.sample(7)
        first.sample(4, stream="monitor")
        first.sample(6, stream="monitor")
    assert len(first) == len(second) == 17
    assert first.num_seen == second.num_seen == 100
    torch.testing.assert_close(first.state_dict()["x"], second.state_dict()["x"])
    torch.testing.assert_close(first.state_dict()["y"], second.state_dict()["y"])


def test_monitor_sampling_does_not_change_training_samples():
    first, second = ReservoirBuffer(20, 4), ReservoirBuffer(20, 4)
    x, y = torch.arange(40).reshape(20, 2), torch.arange(20)
    first.add(x, y)
    second.add(x, y)
    for _ in range(5):
        first.sample(7, stream="monitor")
        _assert_equal_batches(first.sample(6), second.sample(6))


def test_reservoir_checkpoint_restores_all_rngs_and_saved_logits():
    buffer = ReservoirBuffer(11, seed=37)
    x = torch.arange(60, dtype=torch.uint8).reshape(20, 3)
    y = torch.arange(20)
    logits = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    buffer.add(x, y, logits)
    buffer.sample(3)
    buffer.sample(5, stream="monitor")
    snapshot = buffer.state_dict()
    restored = ReservoirBuffer(0)
    restored.load_state_dict(snapshot)
    for stream in ("monitor", "train", "monitor", "train"):
        _assert_equal_batches(buffer.sample(7, stream=stream), restored.sample(7, stream=stream))
    buffer.add(x + 1, y + 20, logits + 1)
    restored.add(x + 1, y + 20, logits + 1)
    _assert_equal_batches(buffer.sample(11), restored.sample(11))
    assert restored.num_seen == 40
    assert restored.nbytes() == 11 * (3 + 8 + 4 * 4)
    # Snapshots and sample results cannot mutate the stored examples.
    before = restored.state_dict()["x"].clone()
    restored.sample(11).x.zero_()
    snapshot["x"].zero_()
    torch.testing.assert_close(restored.state_dict()["x"], before)


def test_reservoir_stores_clones_and_samples_without_replacement():
    buffer = ReservoirBuffer(10)
    x = torch.arange(20, dtype=torch.uint8).reshape(10, 2)
    y = torch.arange(10)
    buffer.add(x, y)
    x.zero_()
    y.zero_()
    batch = buffer.sample(100)
    assert len(batch.y.unique()) == 10
    assert batch.x.dtype == torch.uint8
    assert batch.logits is None
    assert buffer.nbytes() == 10 * (2 + 8)


def test_zero_capacity_and_empty_reservoir_are_valid():
    buffer = ReservoirBuffer(0)
    buffer.add(torch.ones(3, 2), torch.zeros(3, dtype=torch.long))
    assert buffer.sample(2) is None
    assert buffer.num_seen == 3
    assert len(buffer) == buffer.nbytes() == 0
    other = ReservoirBuffer(5)
    other.load_state_dict(buffer.state_dict())
    assert other.capacity == 0 and other.num_seen == 3
    assert ReservoirBuffer(2).sample(2) is None


def test_reservoir_rejects_mixed_logit_storage_and_invalid_requests():
    buffer = ReservoirBuffer(2)
    buffer.add(torch.ones(1, 2), torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="consistently"):
        buffer.add(torch.ones(1, 2), torch.zeros(1, dtype=torch.long), torch.ones(1, 4))
    with pytest.raises(ValueError):
        buffer.sample(-1)
    with pytest.raises(ValueError):
        buffer.sample(1, stream="evaluation")
    with pytest.raises(ValueError):
        ReservoirBuffer(-1)


def test_continual_metrics_hand_computed_and_future_entries_ignored():
    matrix = np.array([[0.9, 42, 42], [0.7, 0.8, 42], [0.6, 0.75, 0.85]])
    metrics = continual_metrics(matrix)
    assert metrics["final_accuracy"] == pytest.approx(2.2 / 3)
    assert metrics["average_incremental_accuracy"] == pytest.approx((0.9 + 0.75 + 2.2 / 3) / 3)
    assert metrics["backward_transfer"] == pytest.approx(-0.175)
    assert metrics["forgetting"] == pytest.approx(0.175)


def test_forgetting_uses_intermediate_peak_and_includes_final_observation():
    matrix = [[0.5, math.nan, math.nan], [0.8, 0.6, math.nan], [0.7, 0.7, 0.9]]
    metrics = continual_metrics(matrix)
    assert metrics["backward_transfer"] == pytest.approx(0.15)
    assert metrics["forgetting"] == pytest.approx(0.05)
    assert continual_metrics([[0.75]]) == {
        "final_accuracy": 0.75,
        "average_incremental_accuracy": 0.75,
        "backward_transfer": 0.0,
        "forgetting": 0.0,
    }


def test_continual_metrics_skips_unavailable_lower_triangle_entries():
    metrics = continual_metrics([[0.8, math.nan], [math.nan, 0.6]])
    assert metrics["final_accuracy"] == 0.6
    assert metrics["average_incremental_accuracy"] == pytest.approx(0.7)
    assert math.isnan(metrics["backward_transfer"])


def test_auc_interpolates_exact_requested_horizon():
    points = [(0, 0), (2, 1), (4, 0)]
    assert normalized_auc(points) == pytest.approx(0.5)
    assert normalized_auc(points, fraction=0.25) == pytest.approx(0.25)
    assert normalized_auc(points, fraction=0.75) == pytest.approx(1.75 / 3)
    assert normalized_auc([(0, 0.7)]) == pytest.approx(0.7)
    assert normalized_auc(points, fraction=0) == 0
    with pytest.raises(ValueError, match="baseline"):
        normalized_auc([(1, 0), (2, 1)])
    with pytest.raises(ValueError, match="increasing"):
        normalized_auc([(0, 0), (0, 1)])


def test_representation_metrics_known_ranks_and_uncentered_dormancy():
    # Centered orthogonal columns have equal singular values and rank two.
    features = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])
    metrics = representation_metrics(features)
    assert metrics["effective_rank"] == pytest.approx(2)
    assert metrics["stable_rank"] == pytest.approx(2)
    assert metrics["dormant_fraction"] == pytest.approx(1 / 3)
    constant = representation_metrics(torch.ones(4, 5))
    assert constant == {"effective_rank": 0, "stable_rank": 0, "dormant_fraction": 0}
    zero = representation_metrics(torch.zeros(4, 5))
    assert zero == {"effective_rank": 0, "stable_rank": 0, "dormant_fraction": 1}
    single = representation_metrics(torch.tensor([[1.0, 0]]))
    assert single["effective_rank"] == single["stable_rank"] == 0
    assert single["dormant_fraction"] == 0.5


def test_representation_rank_does_not_exceed_centered_sample_bound():
    generator = torch.Generator().manual_seed(1)
    metrics = representation_metrics(torch.randn(3, 20, generator=generator))
    assert 1 <= metrics["effective_rank"] <= 2 + 1e-10
    assert 1 <= metrics["stable_rank"] <= 2 + 1e-10


def test_bootstrap_resamples_pairs_and_is_reproducible():
    # Huge between-seed variance disappears under correct paired resampling.
    a = [1001, 2, -997, 504]
    b = [1000, 1, -998, 503]
    result = paired_bootstrap(a, b, seed=29, n_resamples=500)
    assert result == {"mean_difference": 1, "ci_low": 1, "ci_high": 1, "n_pairs": 4}
    a = [1, 4, 7]
    b = [0, 2, 4]
    assert paired_bootstrap(a, b, 3, 100) == paired_bootstrap(a, b, 3, 100)
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])


def test_synthetic_stream_is_reproducible_disjoint_and_rng_isolated():
    config = {
        "dataset": "synthetic", "n_experiences": 3, "classes_per_experience": 2,
        "input_dim": 8, "train_per_class": 6, "validation_per_class": 2,
        "test_per_class": 3,
    }
    torch.manual_seed(97)
    global_state = torch.get_rng_state().clone()
    first = build_stream(config, seed=42)
    torch.testing.assert_close(torch.get_rng_state(), global_state)
    torch.manual_seed(12)
    second = build_stream(config, seed=42)
    assert first.class_order == second.class_order
    assert first.num_classes == 6 and first.input_shape == (8,)
    assert sorted(label for experience in first.experiences for label in experience.classes) == list(range(6))
    for a, b in zip(first.experiences, second.experiences):
        assert a.classes == b.classes
        sample_sets = []
        for name, count in (("train", 12), ("validation", 4), ("test", 6)):
            x, y = getattr(a, name).tensors
            torch.testing.assert_close(x, getattr(b, name).tensors[0])
            torch.testing.assert_close(y, getattr(b, name).tensors[1])
            assert len(x) == count
            assert x.dtype == torch.float32
            assert set(y.tolist()) == set(a.classes)
            sample_sets.append({row.numpy().tobytes() for row in x})
        assert sample_sets[0].isdisjoint(sample_sets[1])
        assert sample_sets[0].isdisjoint(sample_sets[2])
        assert sample_sets[1].isdisjoint(sample_sets[2])


def test_cifar_subset_has_disjoint_train_validation_and_dense_labels(monkeypatch):
    from torchvision import datasets

    class FakeCifar100:
        def __init__(self, root, train, download):
            count = 6 if train else 3
            self.targets = np.repeat(np.arange(100), count).tolist()
            self.data = np.zeros((100 * count, 8, 8, 3), dtype=np.uint8)
            self.data[..., 0] = np.asarray(self.targets)[:, None, None]
            self.data[..., 1] = np.tile(np.arange(count), 100)[:, None, None]
            self.data[..., 2] = 0 if train else 255

    monkeypatch.setattr(datasets, "CIFAR100", FakeCifar100)
    stream = build_stream({
        "dataset": "cifar100", "n_experiences": 3, "classes_per_experience": 2,
        "train_per_class": 3, "validation_per_class": 2, "test_per_class": 2,
    }, seed=23)
    assert stream.num_classes == 6
    assert len(set(stream.class_order)) == 6
    indices = stream.metadata["split_indices"]
    assert set(indices["train"]).isdisjoint(indices["validation"])
    assert sorted(label for experience in stream.experiences for label in experience.classes) == list(range(6))
    for experience in stream.experiences:
        assert experience.train.tensors[0].dtype == torch.uint8
        assert experience.train.tensors[0].shape == (6, 3, 8, 8)
        assert (experience.test.tensors[0][:, 2] == 255).all()
        assert (experience.train.tensors[0][:, 2] == 0).all()
        for name in ("train", "validation", "test"):
            x, y = getattr(experience, name).tensors
            expected = [stream.metadata["label_mapping"][str(int(label))] for label in x[:, 0, 0, 0]]
            assert y.tolist() == expected


def test_preprocess_is_reproducible_and_does_not_mutate_raw_data_or_global_rng():
    raw = torch.arange(2 * 3 * 32 * 32).remainder(256).to(torch.uint8).reshape(2, 3, 32, 32)
    before = raw.clone()
    global_state = torch.get_rng_state().clone()
    a = preprocess(raw, "cifar100", training=True, generator=torch.Generator().manual_seed(7))
    b = preprocess(raw, "cifar100", training=True, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(raw, before)
    torch.testing.assert_close(torch.get_rng_state(), global_state)
    assert a.shape == raw.shape and a.dtype == torch.float32
    evaluation = preprocess(raw, "cifar100")
    mean = torch.tensor(CIFAR100_MEAN)[None, :, None, None]
    std = torch.tensor(CIFAR100_STD)[None, :, None, None]
    torch.testing.assert_close(evaluation, (raw.float() / 255 - mean) / std)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_preprocess_accepts_cpu_generator_for_cuda_images():
    raw = torch.arange(2 * 3 * 32 * 32).remainder(256).to(torch.uint8).reshape(2, 3, 32, 32)
    a = preprocess(raw, "cifar100", True, torch.Generator().manual_seed(7))
    b = preprocess(raw.cuda(), "cifar100", True, torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b.cpu())


@pytest.mark.parametrize("name,shape", [("mlp", (9,)), ("cnn", (3, 16, 16)), ("resnet18", (3, 16, 16))])
def test_model_plastic_blocks_partition_parameters_and_units_are_safe(name, shape):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = make_model(name, shape, num_classes=5, width=16)
        x = torch.randn(2, *shape)
        logits, features = model(x, return_features=True)
        assert logits.shape == (2, 5)
        assert all(not value.requires_grad for value in features.values())
        blocks = model.plastic_blocks()
        grouped = [id(p) for block in blocks.values() for p in block.parameters() if p.requires_grad]
        expected = [id(p) for p in model.parameters() if p.requires_grad]
        assert len(grouped) == len(set(grouped))
        assert set(grouped) == set(expected)
        assert list(blocks)[-1] == "head"
        logits.square().mean().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        units = model.recyclable_units()
        assert len(units) == (2 if name == "mlp" else 1)
        for unit in units:
            assert unit.block in blocks
            assert unit.name in features
            assert unit.block != "head"
            assert unit.incoming.weight.shape[0] == unit.outgoing.weight.shape[1]
            assert features[unit.name].shape[1] == unit.incoming.weight.shape[0]
            # Zeroing its sole consumer decouples the incoming unit completely.
            # This would fail for unaccounted residual paths or norm couplings.
            with torch.no_grad():
                unit.outgoing.weight[:, 0].zero_()
                before = model(x).clone()
                unit.incoming.weight[0].fill_(100)
                unit.incoming.bias[0].fill_(100)
                after = model(x)
            torch.testing.assert_close(before, after)
        assert all(torch.count_nonzero(m.bias) == 0 for m in model.modules()
                   if isinstance(m, torch.nn.Linear) and m.bias is not None
                   and m not in [unit.incoming for unit in units])
    finally:
        torch.set_num_threads(previous_threads)
