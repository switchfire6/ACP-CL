import copy

import pytest
import torch

from acp_cl.sensor import FrozenInputSensor


def test_sensor_is_private_rng_reproducible_and_fixed_across_learner_changes():
    inputs = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 10
    model = torch.nn.Linear(6, 3)
    before_rng = torch.random.get_rng_state().clone()
    sensor = FrozenInputSensor(seed=17, width=13)
    before = sensor.transform(inputs)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert torch.equal(before, FrozenInputSensor(17, 13).transform(inputs))
    assert not torch.equal(before, FrozenInputSensor(18, 13).transform(inputs))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(inputs).square().mean().backward()
    optimizer.step()
    torch.nn.init.normal_(model.weight)
    torch.nn.init.zeros_(model.bias)
    assert torch.equal(before, sensor.transform(inputs))
    assert before.shape == (8, 13)
    assert not before.requires_grad
    assert sensor.nbytes() == (6 * 13 + 13) * 4


def test_image_pooling_preserves_fixed_descriptor_for_constant_resizing():
    sensor = FrozenInputSensor(3)
    images = torch.full((2, 3, 16, 16), 0.25, requires_grad=True)
    small = sensor.transform(images)
    large = sensor.transform(torch.full((2, 3, 32, 32), 0.25))
    assert torch.equal(small, large)
    assert small.shape == (2, 64)
    assert small.dtype == torch.float32
    assert not small.requires_grad
    assert not torch.equal(small, sensor.transform(torch.zeros_like(images)))
    with pytest.raises(ValueError, match="kind/dimension"):
        sensor.transform(torch.zeros(2, 1, 16, 16))
    with pytest.raises(ValueError, match="kind/dimension"):
        sensor.transform(torch.zeros(2, 48))


@pytest.mark.parametrize("initialized", [False, True])
def test_sensor_checkpoint_restores_cold_and_bound_state(initialized):
    inputs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    sensor = FrozenInputSensor(12, width=9)
    if initialized:
        sensor.transform(inputs)
    saved = sensor.state_dict()
    restored = FrozenInputSensor(12, width=9)
    restored.load_state_dict(saved)
    assert torch.equal(sensor.transform(inputs), restored.transform(inputs))
    if initialized:
        saved["projection"].zero_()
        saved["bias"].zero_()
        assert torch.count_nonzero(sensor.state_dict()["projection"])


@pytest.mark.parametrize("change", [
    {"version": True}, {"seed": 99}, {"width": 7}, {"pool_size": 8},
    {"input_dimension": -2}, {"input_kind": "learner"},
    {"projection": torch.ones(1)}, {"bias": torch.tensor([float("nan")])},
])
def test_sensor_invalid_checkpoint_does_not_mutate_sensor(change):
    inputs = torch.ones(3, 5)
    sensor = FrozenInputSensor(8, width=6)
    expected = sensor.transform(inputs)
    saved = copy.deepcopy(sensor.state_dict())
    with pytest.raises(ValueError):
        sensor.load_state_dict({**saved, **change})
    assert torch.equal(sensor.transform(inputs), expected)


@pytest.mark.parametrize("bad", [
    torch.ones(3, 4, 5), torch.ones(0, 3), torch.ones(2, 0),
    torch.ones(2, 3, dtype=torch.int64), torch.full((2, 3), float("inf")),
])
def test_sensor_rejects_bad_inputs_without_binding(bad):
    sensor = FrozenInputSensor(1)
    with pytest.raises(ValueError):
        sensor.transform(bad)
    assert sensor.state_dict()["input_kind"] is None


@pytest.mark.parametrize("seed,width", [(True, 64), (-1, 64), (2**63, 64), (1, 0), (1, True)])
def test_sensor_rejects_invalid_configuration(seed, width):
    with pytest.raises(ValueError):
        FrozenInputSensor(seed, width)
