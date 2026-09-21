"""Single-arrival, fixed-label domains from the official CIFAR-10 training set.

Each class is independently shuffled using a private RNG. A validation reserve
is removed first, and its leading examples form one fixed active panel. The
remaining training examples are partitioned into disjoint, balanced arrivals.
Domain transforms are deterministic and are applied only after this split.

The default stream uses 30 experiences of 150 training examples per class,
reserves 500 per class, and evaluates an active panel of 50 per class. Recurring
domains cycle through original, partly grayscale, and Gaussian-blurred images;
stationary streams use original images with exactly the same ordered base IDs.
Validation reuses one TensorDataset object per domain. No official test images
or labels are loaded by this builder, and every experience's test set is empty.
The torchvision constructor may still verify archive/test-file checksums as part
of its dataset integrity check; ``train=False`` is never constructed.

This is a finite domain-shift bridge, not a standard class-incremental CIFAR
protocol. Its three simple transformations do not model general natural drift.
Repeated validation measurements reuse the same base images and are correlated.
Source IDs and domain IDs are evaluator metadata, never model inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset

if TYPE_CHECKING:
    from .data import Stream


DOMAIN_NAMES = ("original", "grayscale", "blur")
_GENERATOR_VERSION = 1
_TRANSFORM_BATCH_SIZE = 128
_CONFIG_KEYS = {
    "dataset", "n_experiences", "train_per_class", "validation_reserved_per_class",
    "validation_per_class", "test_per_class", "regime", "root", "download",
    "classes_per_experience", "n_domains", "image_size",
}


def _integer(config: dict, key: str, default: int, minimum: int = 0) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return int(value)


def _private_rng(seed: int, label: int) -> np.random.Generator:
    unsigned = int(seed) % (2**64)
    return np.random.default_rng(np.random.SeedSequence([
        unsigned & 0xFFFFFFFF, unsigned >> 32, 601, label,
    ]))


def _load_training_source(root: str, download: bool) -> Any:
    """Monkeypatchable loader; no official test-set instance is constructed."""
    from torchvision.datasets import CIFAR10

    return CIFAR10(root=root, train=True, download=download)


def _blur_kernel() -> Tensor:
    coordinate = torch.arange(-1, 2, dtype=torch.float32)
    weights = torch.exp(-0.5 * coordinate.square())
    weights = weights / weights.sum()
    return weights[:, None] * weights[None, :]


def _transform_domain(images: Tensor, domain: str) -> Tensor:
    """Transform CPU uint8 NCHW images without mutation or random draws.

    Original images are returned as-is. Other domains receive new storage.
    Float intermediates are bounded to 128 examples, independent of stream size.
    ``torch.round`` uses round-to-nearest with ties to even.
    """
    if domain not in DOMAIN_NAMES:
        raise ValueError(f"Unknown CIFAR-10 domain {domain!r}")
    if (images.device.type != "cpu" or images.dtype != torch.uint8
            or images.ndim != 4 or images.shape[1] != 3
            or min(images.shape[-2:]) < 2):
        raise ValueError("Domain transforms require CPU uint8 NCHW images with 3 channels and H,W >= 2")
    if domain == "original":
        return images
    output = torch.empty_like(images)
    kernel = _blur_kernel()[None, None].expand(3, 1, 3, 3) if domain == "blur" else None
    for offset in range(0, len(images), _TRANSFORM_BATCH_SIZE):
        chunk = images[offset : offset + _TRANSFORM_BATCH_SIZE].to(torch.float32)
        if domain == "grayscale":
            gray = 0.299 * chunk[:, 0:1] + 0.587 * chunk[:, 1:2] + 0.114 * chunk[:, 2:3]
            transformed = 0.2 * chunk + 0.8 * gray
        else:
            transformed = F.conv2d(F.pad(chunk, (1, 1, 1, 1), mode="reflect"), kernel, groups=3)
        output[offset : offset + len(chunk)] = transformed.round().clamp(0, 255).to(torch.uint8)
    return output


def _array_hash(array: np.ndarray) -> str:
    """Hash logical C-order bytes, bounding copies for torchvision's NHWC view."""
    digest = hashlib.sha256()
    if array.flags.c_contiguous:
        digest.update(memoryview(array).cast("B"))
    else:
        bytes_per_item = int(np.prod(array.shape[1:])) * array.dtype.itemsize
        chunk_size = max(1, (1024 * 1024) // max(1, bytes_per_item))
        for offset in range(0, len(array), chunk_size):
            chunk = np.ascontiguousarray(array[offset : offset + chunk_size])
            digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _index_hash(indices: list[int]) -> str:
    return _array_hash(np.asarray(indices, dtype="<i8"))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_source_identities(source: Any, root: str) -> dict[str, Any]:
    """Record declared training-file identities and actual hashes if available.

    No test-file attribute is consulted. This optional provenance supplements
    the mandatory hashes of every materialized source image and label.
    """
    identity: dict[str, Any] = {
        "loader": f"{type(source).__module__}.{type(source).__qualname__}",
        "partition": "train",
    }
    base_folder = getattr(source, "base_folder", None)
    if isinstance(base_folder, str):
        identity["base_folder"] = base_folder
    files = []
    entries = list(getattr(source, "train_list", ()))
    meta = getattr(source, "meta", None)
    if isinstance(meta, dict) and isinstance(meta.get("filename"), str):
        entries.append((meta["filename"], meta.get("md5")))
    source_root = Path(getattr(source, "root", root)).expanduser().resolve()
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            continue
        filename, expected_md5 = entry
        if not isinstance(filename, str):
            continue
        item: dict[str, Any] = {"name": filename, "declared_md5": expected_md5}
        if isinstance(base_folder, str):
            path = (source_root / base_folder / filename).resolve()
            # Fixture-provided paths must not escape their declared source root.
            if path.is_relative_to(source_root) and path.is_file():
                item.update({"bytes": path.stat().st_size, "sha256": _file_hash(path)})
        files.append(item)
    identity["training_and_label_metadata_files"] = files
    return identity


def _source_arrays(source: Any) -> tuple[np.ndarray, np.ndarray]:
    if getattr(source, "train", True) is not True:
        raise ValueError("CIFAR-10 domain source must be the official training partition")
    images = np.asarray(source.data)
    raw_labels = np.asarray(source.targets)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("CIFAR-10 source images must be uint8 NHWC with shape (N, 32, 32, 3)")
    if (raw_labels.ndim != 1 or len(raw_labels) != len(images)
            or raw_labels.dtype.kind not in "iu"):
        raise ValueError("CIFAR-10 source targets must be a matching one-dimensional integer array")
    labels = raw_labels.astype("<i8", copy=False)
    if set(np.unique(labels).tolist()) != set(range(10)):
        raise ValueError("CIFAR-10 source must contain exactly labels 0 through 9")
    return images, labels


def build_cifar10_domains(config: dict, seed: int, *, source: Any = None) -> Stream:
    """Build the fixed-label stream; optional source injection is for offline tests.

    All counts are per class. ``validation_reserved_per_class`` includes the
    active ``validation_per_class`` panel, with the remainder held aside unused.
    ``train_per_class`` is the number of distinct current images per experience,
    not the number reused each epoch. The runner must use one pass per experience
    to preserve a single current arrival per image; replay may revisit arrivals.
    """
    from .data import Experience, Stream

    if not isinstance(config, dict):
        raise ValueError("CIFAR-10 domain config must be a dictionary")
    if not all(isinstance(key, str) for key in config):
        raise ValueError("CIFAR-10 domain config keys must be strings")
    unknown = set(config).difference(_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown CIFAR-10 domain config keys: {sorted(unknown)}")
    if config.get("dataset", "cifar10_domains") != "cifar10_domains":
        raise ValueError("dataset must be 'cifar10_domains'")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    regime = config.get("regime", "recurring")
    if regime not in ("recurring", "stationary"):
        raise ValueError("regime must be recurring or stationary")
    n_experiences = _integer(config, "n_experiences", 30, minimum=1)
    train_count = _integer(config, "train_per_class", 150, minimum=1)
    reserve_count = _integer(config, "validation_reserved_per_class", 500, minimum=1)
    active_count = _integer(config, "validation_per_class", 50, minimum=1)
    if active_count > reserve_count:
        raise ValueError("validation_per_class cannot exceed validation_reserved_per_class")
    for key, expected in (("classes_per_experience", 10), ("n_domains", 3),
                          ("image_size", 32), ("test_per_class", 0)):
        if _integer(config, key, expected) != expected:
            raise ValueError(f"{key} must be {expected} for cifar10_domains")
    download = config.get("download", False)
    if not isinstance(download, bool):
        raise ValueError("download must be a boolean")
    try:
        root = str(Path(config.get("root", "data")).expanduser())
    except TypeError as error:
        raise ValueError("root must be a filesystem path") from error
    injected = source is not None
    if source is None:
        source = _load_training_source(root, download)
    images, labels = _source_arrays(source)

    total_train_count = n_experiences * train_count
    indices = {name: [] for name in (
        "train", "validation", "validation_reserved", "validation_unused", "unused_train", "test",
    )}
    train_by_class = []
    for label in range(10):
        available = np.flatnonzero(labels == label)
        required = reserve_count + total_train_count
        if required > len(available):
            raise ValueError(
                f"Class {label} has {len(available)} source images, but {n_experiences} experiences "
                f"x {train_count} train + {reserve_count} validation reserve require {required}; "
                "current images cannot be repeated"
            )
        ordered = _private_rng(int(seed), label).permutation(available)
        reserved = ordered[:reserve_count]
        train_by_class.append(ordered[reserve_count : required])
        indices["validation"].extend(reserved[:active_count].tolist())
        indices["validation_reserved"].extend(reserved.tolist())
        indices["validation_unused"].extend(reserved[active_count:].tolist())
        indices["unused_train"].extend(ordered[required:].tolist())

    arrivals = [
        np.concatenate([per_class[i * train_count : (i + 1) * train_count]
                        for per_class in train_by_class]).tolist()
        for i in range(n_experiences)
    ]
    indices["train"] = [value for experience_ids in arrivals for value in experience_ids]
    domain_order = [i % 3 if regime == "recurring" else 0 for i in range(n_experiences)]
    changes = [False] + [left != right for left, right in zip(domain_order, domain_order[1:])]

    def materialize(selected: list[int], domain_id: int) -> TensorDataset:
        ids = np.asarray(selected, dtype=np.int64)
        # Advanced indexing copies before conversion: source arrays stay private.
        raw = torch.from_numpy(images[ids]).permute(0, 3, 1, 2).contiguous()
        x = _transform_domain(raw, DOMAIN_NAMES[domain_id])
        y = torch.from_numpy(labels[ids].astype(np.int64, copy=True))
        return TensorDataset(x, y)

    validation = {domain_id: materialize(indices["validation"], domain_id)
                  for domain_id in sorted(set(domain_order))}
    empty_test = TensorDataset(torch.empty(0, 3, 32, 32, dtype=torch.uint8),
                               torch.empty(0, dtype=torch.long))
    experiences = [
        Experience(i, tuple(range(10)), materialize(selected, domain_order[i]),
                   validation[domain_order[i]], empty_test)
        for i, selected in enumerate(arrivals)
    ]

    images_hash = _array_hash(images)
    labels_hash = _array_hash(labels)
    source_description = {
        "dataset": "CIFAR10", "partition": "train", "image_layout": "NHWC",
        "image_shape": list(images.shape), "image_dtype": "uint8",
        "label_shape": list(labels.shape), "label_dtype": "little-endian int64",
        "images_sha256": images_hash, "labels_sha256": labels_hash,
    }
    source_fingerprint = hashlib.sha256(json.dumps(
        source_description, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    rounding = "float32; round to nearest with ties to even; clamp [0,255]; cast uint8"
    metadata = {
        "dataset": "cifar10_domains", "generator_version": _GENERATOR_VERSION,
        "seed": int(seed), "regime": regime, "root": root,
        "num_classes": 10, "label_mapping": {str(i): i for i in range(10)},
        "source_fingerprint": source_fingerprint,
        "source_fingerprint_encoding": "SHA256 of canonical sorted compact JSON of source_description",
        "source_description": source_description,
        "source_injected": injected,
        "raw_source_identities": _raw_source_identities(source, root),
        "current_arrival_ids_by_experience": arrivals,
        "current_arrival_id_hashes_by_experience": [_index_hash(values) for values in arrivals],
        "current_arrival_identity": "zero-based positional index in official CIFAR-10 training data",
        "single_current_arrival_per_source_image": True,
        "split_indices": indices,
        "split_index_hashes": {name: _index_hash(values) for name, values in indices.items()},
        "split_index_hash_encoding": "SHA256 of ordered little-endian int64 index bytes",
        "split_index_sources": {name: "official training set" for name in indices if name != "test"},
        "split_counts_per_class": {
            "train_per_experience": train_count, "train_total": total_train_count,
            "validation": active_count, "validation_reserved": reserve_count,
            "validation_unused": reserve_count - active_count, "test": 0,
        },
        "split_policy": "per-class private permutation; reserve validation first; active panel is reserve prefix; consume training once",
        "split_rng": {"algorithm": "numpy default_rng PCG64", "namespace": 601,
                      "per_class_seed_components": "seed low32, seed high32, namespace, label"},
        "domain_order": domain_order, "domain_names": {str(i): name for i, name in enumerate(DOMAIN_NAMES)},
        "boundary_change_flags": changes, "signal_change_flags": changes.copy(),
        "signal_domain_order": domain_order.copy(),
        "validation_domain_ids_by_experience": domain_order.copy(),
        "validation_pool_ids_by_experience": [f"cifar10_validation_domain_{d}" for d in domain_order],
        "validation_pool_reuse": "one shared TensorDataset object per domain; same active base IDs in every domain",
        "domain_transforms": {
            "original": {"operation": "identity"},
            "grayscale": {"operation": "0.2*RGB + 0.8*gray",
                          "gray": "0.299*R + 0.587*G + 0.114*B", "quantization": rounding},
            "blur": {"operation": "depthwise 3x3 Gaussian convolution", "sigma": 1.0,
                     "kernel_size": 3, "kernel": _blur_kernel().tolist(),
                     "kernel_definition": "float32 exp(-0.5*x*x), x=[-1,0,1]; normalize 1D then outer product",
                     "padding": "reflection, one pixel per side", "quantization": rounding},
        },
        "raw_dtype": "uint8", "image_size": 32,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                          "uint8_scale": 255.0},
        "stochastic_augmentation": False,
        "official_test_images_materialized": False,
        "metadata_visibility": "source IDs, domain IDs, and change flags are evaluator-only",
        "limitations": [
            "Three fixed deterministic image transforms, not arbitrary natural distribution drift.",
            "Validation domains share base images; repeated/domain measurements are correlated.",
            "Validation reserve outside the active panel remains unused; no official test evaluation.",
            "Single current arrivals require one runner pass per experience; replay intentionally revisits images.",
        ],
    }
    return Stream(experiences, (3, 32, 32), 10, list(range(10)), metadata)
