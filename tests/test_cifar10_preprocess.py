"""The transfer pilot must not inherit legacy CIFAR-100 augmentation/statistics."""

import pytest
import torch

from acp_cl.data import preprocess


def test_cifar10_domain_preprocessing_is_fixed_and_does_not_draw_rng():
    x = torch.tensor([0, 127, 255], dtype=torch.uint8).view(1, 3, 1, 1).expand(2, 3, 8, 8)
    generator = torch.Generator().manual_seed(87)
    before = generator.get_state().clone()
    expected = (x.float() / 255 - 0.5) / 0.5
    assert torch.equal(preprocess(x, "cifar10_domains", True, generator), expected)
    assert torch.equal(preprocess(x.float()/255, "cifar10_domains", False), expected)
    assert torch.equal(generator.get_state(), before)


@pytest.mark.parametrize("shape", [(2, 3, 8), (2, 1, 8, 8), (2, 8, 8, 3)])
def test_cifar10_domain_preprocessing_rejects_wrong_channel_layout(shape):
    with pytest.raises(ValueError, match="NCHW"):
        preprocess(torch.zeros(shape), "cifar10_domains")
