"""Reproducible streams with evaluator-only experience and class metadata.

CIFAR examples stay raw uint8 tensors until ``preprocess`` is called.  This
keeps replay's image storage cost independent of float training precision.
Every source of data randomness is private and separate from model RNGs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset


@dataclass
class Experience:
    index: int
    classes: tuple[int, ...]
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset


@dataclass
class Stream:
    experiences: list[Experience]
    input_shape: tuple[int, ...]
    num_classes: int
    class_order: list[int]
    metadata: dict[str, Any]


def _rng(seed: int, namespace: int) -> np.random.Generator:
    value = int(seed) % (2**64)
    return np.random.default_rng(
        np.random.SeedSequence([value & 0xFFFFFFFF, value >> 32, namespace])
    )


def _generator(seed: int, namespace: int) -> torch.Generator:
    # SeedSequence avoids accidental adjacent-stream overlap from seed+offset.
    state = _rng(seed, namespace).integers(0, 2**63 - 1, dtype=np.int64)
    return torch.Generator().manual_seed(int(state))


def _integer(config: dict, key: str, default: int, minimum: int = 0) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return int(value)


def build_stream(config: dict, seed: int) -> Stream:
    """Build disjoint development/test streams without touching global RNGs.

    For class-incremental synthetic/CIFAR-100 streams,
    ``classes_per_experience * n_experiences`` defines the selected class count.
    Fixed-label domain streams are dispatched to their own builders.
    CIFAR selection is a seeded permutation of the original 100 class IDs.  Its
    selected labels are remapped densely in sorted original-ID order; the
    ``class_order`` retains original IDs and ``Experience.classes`` uses remapped
    labels.  The model receives only samples, never experience identifiers.
    """
    dataset = str(config.get("dataset", "synthetic")).lower()
    if dataset == "procedural_shapes":
        from .shapes import build_shapes_stream
        return build_shapes_stream(config, seed)
    if dataset == "cifar10_domains":
        from .cifar_domains import build_cifar10_domains
        return build_cifar10_domains(config, seed)
    n_experiences = _integer(config, "n_experiences", 5, minimum=1)
    classes_per_experience = _integer(config, "classes_per_experience", 2, minimum=1)
    n_classes = n_experiences * classes_per_experience
    if dataset == "synthetic":
        return _synthetic(config, seed, n_experiences, classes_per_experience, n_classes)
    if dataset == "cifar100":
        if n_classes > 100:
            raise ValueError("CIFAR-100 has only 100 classes")
        return _cifar100(config, seed, n_experiences, classes_per_experience, n_classes)
    raise ValueError(f"Unknown dataset {dataset!r}; choose synthetic, cifar100, procedural_shapes, or cifar10_domains")


def _synthetic(
    config: dict, seed: int, n_experiences: int, classes_per_experience: int, n_classes: int
) -> Stream:
    input_dim = _integer(config, "input_dim", 32, minimum=1)
    counts = {
        "train": _integer(config, "train_per_class", 128, minimum=1),
        "validation": _integer(config, "validation_per_class", 32),
        "test": _integer(config, "test_per_class", 64),
    }
    noise = float(config.get("noise", 0.35))
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    class_order = _rng(seed, 101).permutation(n_classes).tolist()
    prototypes = torch.randn(n_classes, input_dim, generator=_generator(seed, 102))
    splits: dict[str, dict[int, tuple[Tensor, Tensor]]] = {}
    split_indices: dict[str, list[int]] = {}
    offset = 0
    for namespace, (split_name, count) in enumerate(counts.items(), start=103):
        generator = _generator(seed, namespace)
        splits[split_name] = {}
        for label in range(n_classes):
            x = prototypes[label] + noise * torch.randn(count, input_dim, generator=generator)
            y = torch.full((count,), label, dtype=torch.long)
            splits[split_name][label] = (x, y)
        size = count * n_classes
        split_indices[split_name] = list(range(offset, offset + size))
        offset += size
    experiences = []
    for i in range(n_experiences):
        classes = tuple(class_order[i * classes_per_experience : (i + 1) * classes_per_experience])
        datasets = {}
        for split_name in counts:
            xs, ys = zip(*(splits[split_name][label] for label in classes))
            datasets[split_name] = TensorDataset(torch.cat(xs), torch.cat(ys))
        experiences.append(Experience(i, classes, **datasets))
    return Stream(
        experiences=experiences,
        input_shape=(input_dim,),
        num_classes=n_classes,
        class_order=class_order,
        metadata={
            "dataset": "synthetic",
            "seed": int(seed),
            "noise": noise,
            "split_counts_per_class": counts,
            "split_indices": split_indices,
            "label_mapping": {str(label): label for label in range(n_classes)},
            "raw_dtype": "float32",
            "normalization": "identity",
            "purpose": "implementation smoke test; not evidence on visual continual learning",
        },
    )


def _cifar100(
    config: dict, seed: int, n_experiences: int, classes_per_experience: int, n_classes: int
) -> Stream:
    from torchvision.datasets import CIFAR100

    root = str(Path(config.get("root", "data")).expanduser())
    download = bool(config.get("download", False))
    training_source = CIFAR100(root=root, train=True, download=download)
    test_source = CIFAR100(root=root, train=False, download=download)
    train_targets = np.asarray(training_source.targets, dtype=np.int64)
    test_targets = np.asarray(test_source.targets, dtype=np.int64)
    class_order = _rng(seed, 201).permutation(100)[:n_classes].tolist()
    label_mapping = {label: mapped for mapped, label in enumerate(sorted(class_order))}
    validation_count = _integer(config, "validation_per_class", 50)
    train_count = config.get("train_per_class")
    if train_count is not None:
        train_count = _integer(config, "train_per_class", 0, minimum=1)
    test_count = config.get("test_per_class")
    if test_count is not None:
        test_count = _integer(config, "test_per_class", 0)
    split_rng = _rng(seed, 202)
    test_rng = _rng(seed, 203)
    indices: dict[str, dict[int, np.ndarray]] = {name: {} for name in ("train", "validation", "test")}
    for label in class_order:
        source = split_rng.permutation(np.flatnonzero(train_targets == label))
        remaining = len(source) - validation_count
        chosen_train = remaining if train_count is None else train_count
        if remaining < chosen_train or chosen_train <= 0:
            raise ValueError(
                f"Class {label} has {len(source)} training examples; requested "
                f"{chosen_train} train and {validation_count} validation"
            )
        indices["validation"][label] = source[:validation_count]
        indices["train"][label] = source[validation_count : validation_count + chosen_train]
        source_test = test_rng.permutation(np.flatnonzero(test_targets == label))
        chosen_test = len(source_test) if test_count is None else test_count
        if chosen_test > len(source_test):
            raise ValueError(f"Class {label} has only {len(source_test)} test examples")
        indices["test"][label] = source_test[:chosen_test]

    def dataset_for(split_name: str, original_classes: list[int]) -> TensorDataset:
        selected = np.concatenate([indices[split_name][label] for label in original_classes])
        source = test_source if split_name == "test" else training_source
        source_targets = test_targets if split_name == "test" else train_targets
        # Advanced numpy indexing materializes a private copy before conversion.
        x = torch.from_numpy(np.asarray(source.data)[selected]).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor([label_mapping[int(label)] for label in source_targets[selected]], dtype=torch.long)
        return TensorDataset(x, y)

    experiences = []
    for i in range(n_experiences):
        original_classes = class_order[i * classes_per_experience : (i + 1) * classes_per_experience]
        classes = tuple(label_mapping[label] for label in original_classes)
        experiences.append(
            Experience(
                i,
                classes,
                train=dataset_for("train", original_classes),
                validation=dataset_for("validation", original_classes),
                test=dataset_for("test", original_classes),
            )
        )
    return Stream(
        experiences=experiences,
        input_shape=tuple(experiences[0].train.tensors[0].shape[1:]),
        num_classes=n_classes,
        class_order=class_order,
        metadata={
            "dataset": "cifar100",
            "seed": int(seed),
            "root": root,
            "split_indices": {
                split: np.concatenate([by_class[label] for label in class_order]).tolist()
                for split, by_class in indices.items()
            },
            "split_index_sources": {
                "train": "official training set",
                "validation": "official training set",
                "test": "official test set",
            },
            "label_mapping": {str(label): mapped for label, mapped in label_mapping.items()},
            "raw_dtype": "uint8",
            "normalization": {"mean": list(CIFAR100_MEAN), "std": list(CIFAR100_STD)},
        },
    )


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def preprocess(
    x: Tensor,
    dataset: str,
    training: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Convert a batch, with optional crop/flip for legacy CIFAR-100 only.

    Float CIFAR inputs are assumed to be in [0,1]; uint8 inputs are divided by
    255.  Augmentation draws use the provided generator's device and are moved
    to the image device, so CPU generators work with CUDA images.  With no
    generator, each call uses a private, deterministic seed-0 generator; pass a
    persistent explicit generator to advance a reproducible augmentation stream.
    The CIFAR-10 domain pilot instead uses fixed .5 mean/std and no augmentation.
    """
    dataset = dataset.lower()
    if dataset == "synthetic":
        return x.to(dtype=torch.float32)
    if dataset in ("procedural_shapes", "cifar10_domains"):
        if dataset == "cifar10_domains" and (x.ndim != 4 or x.shape[1] != 3):
            raise ValueError("CIFAR-10 domain preprocessing requires an NCHW batch with 3 channels")
        # The natural-image transfer pilot carries the shape study's fixed
        # normalization and uses no stochastic crop/flip augmentation.
        images = x.to(dtype=torch.float32)
        if x.dtype == torch.uint8:
            images = images / 255.0
        return (images - 0.5) / 0.5
    if dataset != "cifar100":
        raise ValueError(f"Unknown dataset {dataset!r}")
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("CIFAR preprocessing requires an NCHW batch with 3 channels")
    images = x.to(dtype=torch.float32)
    if x.dtype == torch.uint8:
        images = images / 255.0
    if training and len(images):
        if generator is None:
            generator = torch.Generator().manual_seed(0)
        count, _, height, width = images.shape
        padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
        offsets = torch.randint(0, 9, (count, 2), generator=generator, device=generator.device)
        offsets = offsets.to(images.device)
        batch = torch.arange(count, device=images.device)[:, None, None]
        rows = offsets[:, 0, None, None] + torch.arange(height, device=images.device)[None, :, None]
        cols = offsets[:, 1, None, None] + torch.arange(width, device=images.device)[None, None, :]
        images = padded.permute(0, 2, 3, 1)[batch, rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(count, generator=generator, device=generator.device).to(images.device) < 0.5
        images = torch.where(flip[:, None, None, None], images.flip(-1), images)
    mean = images.new_tensor(CIFAR100_MEAN)[None, :, None, None]
    std = images.new_tensor(CIFAR100_STD)[None, :, None, None]
    return (images - mean) / std
