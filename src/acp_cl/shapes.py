"""A procedural, fixed-label continual domain-learning benchmark.

The labels always mean disk, square, equilateral triangle, and cross.  Domains
change rendering nuisances and an imperfect label/color association; they never
change the target function.  Shapes have approximately equal filled area at a
given sampled scale, uniformly random rotations, and label-independent random
positions.  A shared conservative placement radius keeps every shape in frame.

This is a controlled mechanism benchmark, not a substitute for natural images.
Silhouettes are simple, their edges are deliberately high contrast, and their
synthetic textures do not reproduce natural-image statistics.  The imperfect
color association is an intentional shortcut whose permutation changes across
domains.  Area matching and position randomization remove obvious deterministic
size/location cues; they do not establish that the benchmark is shortcut-free.
At very low resolution, rasterization can still change apparent shape/area.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset


SHAPE_NAMES = ("disk", "square", "triangle", "cross")
_TRIANGLE_RADIUS_PER_AREA_SIDE = math.sqrt(4 / (3 * math.sqrt(3)))
_TEXTURES = ("stripes", "checker", "waves", "grain")


@dataclass(frozen=True)
class ShapeDomain:
    background_color: tuple[float, float, float]
    foreground_palette: tuple[tuple[float, float, float], ...]
    label_palette: tuple[int, int, int, int]
    color_correlation: float
    background_texture: str
    foreground_texture: str
    background_angle: float
    foreground_angle: float
    background_frequency: float
    foreground_frequency: float
    background_amplitude: float
    foreground_amplitude: float
    sensor_noise_amplitude: float = 0.01


@dataclass(frozen=True)
class _RenderedSample:
    image: np.ndarray
    alpha: np.ndarray
    center: tuple[float, float]
    area_side: float
    angle: float
    placement_radius: float


def _private_rng(seed: int, *namespace: int) -> np.random.Generator:
    unsigned = int(seed) % (2**64)
    sequence = np.random.SeedSequence([unsigned & 0xFFFFFFFF, unsigned >> 32, *namespace])
    return np.random.default_rng(sequence)


def _integer(config: dict, key: str, default: int, minimum: int = 0) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return int(value)


def _make_domains(seed: int, count: int) -> list[ShapeDomain]:
    rng = _private_rng(seed, 401)
    canonical_palette = np.array(
        [[1.0, 0.25, 0.25], [0.25, 1.0, 0.25], [0.25, 0.25, 1.0], [1.0, 1.0, 0.25]]
    )
    initial_label_palette = rng.permutation(4)
    texture_order = rng.permutation(len(_TEXTURES))
    first_light = int(rng.integers(2))
    domains = []
    for domain_id in range(count):
        light_background = (domain_id + first_light) % 2 == 1
        if light_background:
            background = rng.uniform(0.78, 0.90, size=3)
            palette = 0.42 - 0.34 * canonical_palette
        else:
            background = rng.uniform(0.10, 0.23, size=3)
            palette = 0.55 + 0.40 * canonical_palette
        palette += rng.uniform(-0.02, 0.02, size=(4, 3))
        domains.append(
            ShapeDomain(
                background_color=tuple(float(value) for value in background),
                foreground_palette=tuple(tuple(float(value) for value in row) for row in palette),
                label_palette=tuple(int(value) for value in np.roll(initial_label_palette, domain_id % 4)),
                color_correlation=(0.40, 0.65, 0.85)[domain_id % 3],
                background_texture=_TEXTURES[int(texture_order[domain_id % len(_TEXTURES)])],
                foreground_texture=_TEXTURES[int(texture_order[(domain_id + 1) % len(_TEXTURES)])],
                background_angle=float(rng.uniform(0, 2 * math.pi)),
                foreground_angle=float(rng.uniform(0, 2 * math.pi)),
                background_frequency=float(rng.uniform(1.5, 5.5)),
                foreground_frequency=float(rng.uniform(2.0, 6.5)),
                background_amplitude=float(rng.uniform(0.025, 0.060)),
                foreground_amplitude=float(rng.uniform(0.025, 0.060)),
            )
        )
    return domains


def _texture(
    name: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    angle: float,
    frequency: float,
    rng: np.random.Generator,
) -> np.ndarray:
    angle += float(rng.uniform(-0.3, 0.3))
    frequency *= float(rng.uniform(0.8, 1.2))
    phase_x, phase_y = rng.uniform(0, 2 * math.pi, size=2)
    x = math.cos(angle) * grid_x + math.sin(angle) * grid_y
    y = -math.sin(angle) * grid_x + math.cos(angle) * grid_y
    wave_x = np.sin(2 * math.pi * frequency * x + phase_x)
    if name == "stripes":
        return wave_x
    wave_y = np.sin(2 * math.pi * frequency * y + phase_y)
    if name == "checker":
        return wave_x * wave_y
    if name == "waves":
        return 0.5 * (wave_x + wave_y)
    if name == "grain":
        return rng.uniform(-1, 1, size=grid_x.shape)
    raise ValueError(f"Unknown texture {name!r}")


def _render_sample(
    rng: np.random.Generator, label: int, domain: ShapeDomain, image_size: int
) -> _RenderedSample:
    """Render an RGB sample; masks/geometry are returned only for generator QA."""
    if label not in range(4):
        raise ValueError("shape label must be one of 0, 1, 2, 3")
    # The filled area of every continuous silhouette is area_side**2.
    area_side = float(rng.uniform(0.20, 0.36) * image_size)
    placement_radius = _TRIANGLE_RADIUS_PER_AREA_SIDE * area_side
    margin = max(1.0, image_size / 32)
    low, high = placement_radius + margin, image_size - placement_radius - margin
    center_x, center_y = (float(value) for value in rng.uniform(low, high, size=2))
    angle = float(rng.uniform(0, 2 * math.pi))

    # Two-by-two coverage antialiasing and subpixel centers suppress a fixed
    # pixel-grid edge cue without requiring an imaging-library dependency.
    coordinates = (np.arange(image_size * 2, dtype=np.float64) + 0.5) / 2
    grid_x, grid_y = np.meshgrid(coordinates - center_x, coordinates - center_y)
    x = math.cos(angle) * grid_x + math.sin(angle) * grid_y
    y = -math.sin(angle) * grid_x + math.cos(angle) * grid_y
    if label == 0:
        mask = x * x + y * y <= area_side * area_side / math.pi
    elif label == 1:
        mask = (np.abs(x) <= area_side / 2) & (np.abs(y) <= area_side / 2)
    elif label == 2:
        radius = _TRIANGLE_RADIUS_PER_AREA_SIDE * area_side
        mask = (y <= radius / 2) & (y >= math.sqrt(3) * x - radius) & (y >= -math.sqrt(3) * x - radius)
    else:
        half_extent, half_arm = 0.625 * area_side, 0.25 * area_side
        mask = ((np.abs(x) <= half_extent) & (np.abs(y) <= half_arm)) | (
            (np.abs(y) <= half_extent) & (np.abs(x) <= half_arm)
        )
    alpha = mask.reshape(image_size, 2, image_size, 2).mean(axis=(1, 3)).astype(np.float32)

    coordinates = (np.arange(image_size, dtype=np.float64) + 0.5) / image_size
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    background_texture = _texture(
        domain.background_texture, grid_x, grid_y,
        domain.background_angle, domain.background_frequency, rng,
    )
    foreground_texture = _texture(
        domain.foreground_texture, grid_x, grid_y,
        domain.foreground_angle, domain.foreground_frequency, rng,
    )
    if rng.random() < domain.color_correlation:
        palette_index = domain.label_palette[label]
    else:
        palette_index = int(rng.integers(4))
    background = np.array(domain.background_color) + rng.uniform(-0.025, 0.025, size=3)
    foreground = np.array(domain.foreground_palette[palette_index]) + rng.uniform(-0.03, 0.03, size=3)
    background = background + domain.background_amplitude * background_texture[:, :, None]
    foreground = foreground + domain.foreground_amplitude * foreground_texture[:, :, None]
    image = (1 - alpha[:, :, None]) * background + alpha[:, :, None] * foreground
    # Noise draws occur even at amplitude zero, preserving counterfactual draws
    # when only the sensor amplitude is changed in a noise-only control.
    image += domain.sensor_noise_amplitude * rng.uniform(-1, 1, size=image.shape)
    image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8).transpose(2, 0, 1).copy()
    return _RenderedSample(image, alpha, (center_x, center_y), area_side, angle, placement_radius)


def build_shapes_stream(config: dict, seed: int):
    """Build independently sampled splits for a fixed four-label domain stream.

    ``classes_per_experience`` is deliberately irrelevant: all experiences have
    all four labels.  A domain recurrence repeats its *distribution*, never its
    exact examples.  No task/domain identifier is included in training tensors.
    """
    # Imported here because data.build_stream dispatches back to this module.
    from .data import Experience, Stream

    n_experiences = _integer(config, "n_experiences", 8, minimum=1)
    n_domains = _integer(config, "n_domains", 8, minimum=1)
    image_size = _integer(config, "image_size", 32, minimum=16)
    counts = {
        "train": _integer(config, "train_per_class", 128, minimum=1),
        "validation": _integer(config, "validation_per_class", 32),
        "test": _integer(config, "test_per_class", 64),
    }
    regime = str(config.get("regime", "recurring")).lower()
    if regime not in {"recurring", "stationary", "noise_only"}:
        raise ValueError("regime must be recurring, stationary, or noise_only")
    domains = _make_domains(seed, n_domains)
    domain_cycle = [int(value) for value in _private_rng(seed, 402).permutation(n_domains)]
    base_domain_id = domain_cycle[0]
    if regime == "stationary":
        domain_order = [base_domain_id] * n_experiences
    else:
        domain_order = [domain_cycle[index % n_domains] for index in range(n_experiences)]
    if regime == "noise_only":
        # Keep *all* signal-generation parameters fixed; only bounded iid
        # additive sensor noise changes. Its maximum amplitude is 0.12 in [0,1].
        base_domain = domains[base_domain_id]
        levels = np.linspace(0.01, 0.12, n_domains) if n_domains > 1 else np.array([0.01])
        domains = [base_domain for _ in range(n_domains)]
        for domain_id, level in zip(domain_cycle, levels):
            domains[domain_id] = replace(base_domain, sensor_noise_amplitude=float(level))
    signal_domain_order = domain_order if regime == "recurring" else [base_domain_id] * n_experiences

    experiences = []
    split_seeds: list[dict[str, list[int]]] = []
    for experience_index, domain_id in enumerate(domain_order):
        datasets = {}
        experience_seeds = {}
        for split_id, (split_name, count) in enumerate(counts.items()):
            images = np.empty((4 * count, 3, image_size, image_size), dtype=np.uint8)
            labels = np.repeat(np.arange(4, dtype=np.int64), count)
            experience_seeds[split_name] = []
            for label in range(4):
                # One private generator per experience/split/class ensures that
                # changing train counts cannot perturb validation/test samples.
                seed_rng = _private_rng(seed, 403, experience_index, split_id, label)
                draw_seed = int(seed_rng.integers(0, 2**63 - 1, dtype=np.int64))
                experience_seeds[split_name].append(draw_seed)
                rng = np.random.default_rng(draw_seed)
                for sample_index in range(count):
                    rendered = _render_sample(rng, label, domains[domain_id], image_size)
                    images[label * count + sample_index] = rendered.image
            datasets[split_name] = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
        split_seeds.append(experience_seeds)
        experiences.append(Experience(experience_index, (0, 1, 2, 3), **datasets))
    boundary_change_flags = [False] + [a != b for a, b in zip(domain_order, domain_order[1:])]
    signal_change_flags = [False] + [a != b for a, b in zip(signal_domain_order, signal_domain_order[1:])]
    metadata: dict[str, Any] = {
        "dataset": "procedural_shapes",
        "generator_version": 1,
        "seed": int(seed),
        "regime": regime,
        "n_domains": n_domains,
        "shape_names": list(SHAPE_NAMES),
        "label_mapping": {name: index for index, name in enumerate(SHAPE_NAMES)},
        "domain_cycle": domain_cycle,
        "domain_order": domain_order,
        "boundary_change_flags": boundary_change_flags,
        "signal_domain_order": signal_domain_order,
        "signal_change_flags": signal_change_flags,
        "domain_parameters": {str(index): asdict(domain) for index, domain in enumerate(domains)},
        "sensor_noise_amplitude_by_experience": [domains[index].sensor_noise_amplitude for index in domain_order],
        "split_counts_per_class": counts,
        "split_seeds": split_seeds,
        "raw_dtype": "uint8",
        "image_size": image_size,
        "geometry": {
            "rotation_radians": [0.0, 2 * math.pi],
            "area_side_image_fraction": [0.20, 0.36],
            "continuous_filled_area": "area_side squared, equal across labels",
            "centers": "uniform independent x/y within shared label-independent bounding radius",
            "margin_pixels": max(1.0, image_size / 32),
            "antialiasing": "2 by 2 subpixel coverage",
        },
        "metadata_visibility": "domain identities and change flags are evaluator-only",
        "limitations": [
            "Controlled synthetic mechanism benchmark; not confirmatory evidence about natural images.",
            "Four simple high-contrast silhouette labels recur; this is domain-incremental, not class-incremental.",
            "Imperfect label/color correlations intentionally permit changing shortcuts.",
            "Area/position matching is not a guarantee against all rendering shortcuts.",
            "Low-resolution rasterization can alter apparent silhouette area and small cross arms.",
            "Noise-only varies bounded independent sensor noise, not label noise or the signal domain.",
            "Recurrence reuses nuisance parameters while generating fresh independent examples.",
        ],
    }
    return Stream(experiences, (3, image_size, image_size), 4, [0, 1, 2, 3], metadata)
