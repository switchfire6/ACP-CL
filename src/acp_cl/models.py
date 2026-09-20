"""Models with explicit plasticity groups and conservatively recyclable units.

The CNN and ResNet-18 have a Linear/ReLU adapter before their classifier.  Only
this adapter is recycled: recycling a channel within GroupNorm or a residual
branch would not be an independent unit replacement.  Consequently ``resnet18``
means a CIFAR-stem, GroupNorm ResNet-18 *with an adapter*, not the unmodified
ImageNet architecture.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class UnitSpec:
    """A ReLU population with one incoming layer and one outgoing consumer.

    ``name`` indexes the returned activation dictionary, and ``block`` indexes
    ``plastic_blocks()``.  All populations returned here use Linear layers;
    the wider annotation permits future, explicitly audited convolutional ones.
    """

    name: str
    block: str
    incoming: nn.Linear | nn.Conv2d
    outgoing: nn.Linear | nn.Conv2d


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _initialize(module: nn.Module) -> None:
    # Recycling must use the same incoming-weight distribution.
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int, width: int):
        super().__init__()
        self.hidden1 = nn.Linear(math.prod(input_shape), width)
        self.hidden2 = nn.Linear(width, width)
        self.head = nn.Linear(width, num_classes)
        self.apply(_initialize)

    def forward(self, x: Tensor, return_features: bool = False):
        h1 = torch.relu(self.hidden1(x.flatten(1)))
        h2 = torch.relu(self.hidden2(h1))
        logits = self.head(h2)
        if not return_features:
            return logits
        h2_detached = h2.detach()
        return logits, {
            "hidden1": h1.detach(),
            "hidden2": h2_detached,
            "representation": h2_detached,
        }

    def plastic_blocks(self) -> OrderedDict[str, nn.Module]:
        return OrderedDict(hidden1=self.hidden1, hidden2=self.hidden2, head=self.head)

    def recyclable_units(self) -> list[UnitSpec]:
        return [
            UnitSpec("hidden1", "hidden1", self.hidden1, self.hidden2),
            UnitSpec("hidden2", "hidden2", self.hidden2, self.head),
        ]


class SmallCNN(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int, width: int):
        super().__init__()
        channels = [max(8, width // 4), max(8, width // 2), max(8, width)]
        self.conv1 = self._block(input_shape[0], channels[0], stride=1)
        self.conv2 = self._block(channels[0], channels[1], stride=2)
        self.conv3 = self._block(channels[1], channels[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.adapter = nn.Linear(channels[-1], width)
        self.head = nn.Linear(width, num_classes)
        self.apply(_initialize)

    @staticmethod
    def _block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, return_features: bool = False):
        features: dict[str, Tensor] = {}
        for name in ("conv1", "conv2", "conv3"):
            x = getattr(self, name)(x)
            if return_features:
                features[name] = x.detach()
        x = torch.relu(self.adapter(self.pool(x).flatten(1)))
        logits = self.head(x)
        if not return_features:
            return logits
        features["adapter"] = x.detach()
        features["representation"] = features["adapter"]
        return logits, features

    def plastic_blocks(self) -> OrderedDict[str, nn.Module]:
        return OrderedDict(
            (name, getattr(self, name))
            for name in ("conv1", "conv2", "conv3", "adapter", "head")
        )

    def recyclable_units(self) -> list[UnitSpec]:
        return [UnitSpec("adapter", "adapter", self.adapter, self.head)]


class ResNet18(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int, width: int):
        super().__init__()
        # Import lazily so MLP-only experiments do not require torchvision.
        from torchvision.models import resnet18

        backbone = resnet18(weights=None, norm_layer=_group_norm)
        self.stem = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, stride=1, padding=1, bias=False),
            backbone.bn1,
            backbone.relu,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.adapter = nn.Linear(512, width)
        self.head = nn.Linear(width, num_classes)
        self.apply(_initialize)

    def forward(self, x: Tensor, return_features: bool = False):
        features: dict[str, Tensor] = {}
        for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
            x = getattr(self, name)(x)
            if return_features:
                features[name] = x.detach()
        x = torch.relu(self.adapter(self.pool(x).flatten(1)))
        logits = self.head(x)
        if not return_features:
            return logits
        features["adapter"] = x.detach()
        features["representation"] = features["adapter"]
        return logits, features

    def plastic_blocks(self) -> OrderedDict[str, nn.Module]:
        return OrderedDict(
            (name, getattr(self, name))
            for name in ("stem", "layer1", "layer2", "layer3", "layer4", "adapter", "head")
        )

    def recyclable_units(self) -> list[UnitSpec]:
        return [UnitSpec("adapter", "adapter", self.adapter, self.head)]


def make_model(
    name: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    width: int = 128,
) -> nn.Module:
    """Create a model; every trainable parameter occurs in exactly one block."""
    if not input_shape or any(int(size) <= 0 for size in input_shape):
        raise ValueError("input_shape must contain positive dimensions")
    if num_classes <= 0 or width <= 0:
        raise ValueError("num_classes and width must be positive")
    name = name.lower()
    if name == "mlp":
        return MLP(input_shape, num_classes, width)
    if name not in {"cnn", "resnet18"}:
        raise ValueError(f"Unknown model {name!r}; choose mlp, cnn, or resnet18")
    if len(input_shape) != 3:
        raise ValueError("CNN models require (channels, height, width) input_shape")
    cls = SmallCNN if name == "cnn" else ResNet18
    return cls(input_shape, num_classes, width)
