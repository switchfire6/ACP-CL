"""Monte Carlo worker for the Bayes ideal-observer information audit.

POST HOC / EXPLORATORY / NOT DECISION-BEARING. Evaluator-only analysis aid: it
marginalises the hidden per-step supply noise of the KNOWN simulator for given
observable case features (reserves, source/lossy/delayed factors, three pulse
signals) and a given hidden law. It never fits, trains or updates any learner
and never reads or executes checkpoints.

Dynamics are NOT re-implemented. Sampled base supplies are passed through the
project's own ``AcquisitionWorld.simulate`` (i.e. ``AcquisitionWorld.physics``
followed by ``TransferWorld.simulate``). Only the supply draw of
``TransferWorld.cases`` / ``ConditionalWorld.dataset`` is replicated here:

    means    = mean_supply + imbalance * source * season(step) * [+1, -1]
    supplies = clip(means + Normal(0, supply_noise), 0, None)   # mode-0 frame
    mode 1   : supplies[..., ::-1]                              # column swap after clipping

The cue-2 supply-timing shift and its re-clip happen inside ``physics``.
Common random numbers: one noise tensor per task is shared by every law,
signal (cue-flip) variant and action.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
try:  # pragma: no cover - environment dependent
    import acp_cl  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(ROOT / "src"))

from types import SimpleNamespace  # noqa: E402

from acp_cl.acquisition.world import AcquisitionWorld, Law, LawCases, mask_affected  # noqa: E402
from acp_cl.persistence.world import HORIZONS, Cases, TransferWorld  # noqa: E402

LABEL = "POST HOC / EXPLORATORY / NOT DECISION-BEARING"
STEPS = max(HORIZONS)
MC_ROOT = 2026092801          # disjoint from every project seed / trial_seed channel
GAUGE_STEP = TransferWorld.capacity / 240.   # reserve gauge: level = rint(r / 18 * 240)
# Which cues can change the physics of a row: cue 0 -> lossy rows, cue 1 ->
# delayed rows, cue 2 -> lossy or delayed rows (acquisition.world.physics).
ROW_TYPE_CUES = {0: (), 1: (0, 2), 2: (1, 2), 3: (0, 1, 2)}   # type = lossy + 2*delayed


@lru_cache(maxsize=1)
def world():
    return AcquisitionWorld()


def mask_affected_like(factors, cue):
    """acquisition.world.mask_affected applied to bare factor columns (project code reused)."""
    return mask_affected(SimpleNamespace(factors=factors, observations=factors), cue)


def row_types(factors):
    return (factors[:, 1].astype(int) + 2 * factors[:, 2].astype(int))


def supply_means(factors):
    """(n,12,2) mode-0 frame supply means, exactly as TransferWorld.cases (gate 0)."""
    w = TransferWorld
    season = np.ones(STEPS)
    season[STEPS // 2:] = -1
    source = factors[:, 0].astype(np.float64)
    return (w.mean_supply + w.imbalance * source[:, None, None]
            * season[None, :, None] * np.array([1, -1]))


def base_efficiency(factors):
    return np.where(factors[:, 1] == 1, 0.35, 1.0)


def base_delay(factors):
    return np.where(factors[:, 2] == 1, 2, 0).astype(int)


def mode_frame(supplies0, mode):
    """Apply the hidden-mode column swap (ConditionalWorld.dataset) to mode-0 supplies."""
    return supplies0[..., ::-1] if mode == 1 else supplies0


def law_cases(reserves, factors, signals, supplies, law):
    """Evaluator LawCases with a zero-stride dummy observation (unused by the physics)."""
    n = len(reserves)
    dummy = np.broadcast_to(np.zeros((1, 1), np.uint8), (n, 1))
    base = Cases(dummy, np.ascontiguousarray(reserves, dtype=np.float64),
                 np.ascontiguousarray(supplies, dtype=np.float64),
                 base_efficiency(factors), base_delay(factors))
    return LawCases(dummy, base, factors, signals, law)


def simulate(reserves, factors, signals, supplies, law, actions):
    """Project dynamics: AcquisitionWorld.simulate(LawCases, actions) -> Outcomes."""
    cases = law_cases(reserves, factors, signals, supplies, law)
    return world().simulate(cases, np.asarray(actions, dtype=np.int64))


def effective(active, row_type, action):
    """Cues of ``active`` that can affect this row type and action.

    Action 0 moves nothing, so efficiency (cue 0) and delay (cue 1) cannot
    matter for it; only the supply-timing shift (cue 2) can.
    """
    cues = tuple(c for c in active if c in ROW_TYPE_CUES[row_type])
    if action == 0:
        cues = tuple(c for c in cues if c == 2)
    return cues


def variant_keys(laws, row_type, action):
    """Unique physics configurations for all (law, cue-flip variant) of one row type/action."""
    mapping, keys = {}, []
    for index, (mode, active) in enumerate(laws):
        for variant in (None, 0, 1, 2):
            cues = effective(active, row_type, action)
            flip = variant if (variant is not None and variant in cues) else None
            key = (mode, cues, flip)
            if key not in keys:
                keys.append(key)
            mapping[index, variant] = key
    return keys, mapping


def sample_reserves(rng, levels, kb):
    """Gauge-consistent reserves: uniform within the gauge bin intersected with U(1,13)."""
    low = np.maximum(1.0, (levels - .5) * GAUGE_STEP)
    high = np.minimum(13.0, (levels + .5) * GAUGE_STEP)
    if np.any(high <= low):
        raise AssertionError("empty gauge bin")
    return low[None] + (high - low)[None] * rng.random((kb, *levels.shape))


def tile_rows(values, kb):
    return np.broadcast_to(values[None], (kb, *values.shape)).reshape(kb * len(values), *values.shape[1:])


def panel_task(task):
    """One MC batch for one seed: query-panel survival counts under every law/variant.

    Returns int32 counts of survival at horizons 4/8/12 (``query_counts``
    [law, variant(0=correct, 1+c=cue c flipped), query, action, horizon]),
    correct-variant lifetime histograms, sparse joint five-action pattern
    counts, and performed-action survival counts for the support records.
    """
    kb, laws = task["kb"], [tuple((m, tuple(a))) for m, a in task["laws"]]
    rng = np.random.default_rng([MC_ROOT, task["channel"], task["seed"], task["replicate"], task["batch"]])
    query, support = task["query"], task["support"]
    nq, ns = len(query["factors"]), len(support["factors"])
    noise_q = rng.normal(0., TransferWorld.supply_noise, (kb, nq, STEPS, 2))
    noise_s = rng.normal(0., TransferWorld.supply_noise, (kb, ns, STEPS, 2))
    if task["gauge"]:
        res_q = sample_reserves(rng, query["levels"], kb)
        res_s = sample_reserves(rng, support["levels"], kb)
    else:
        res_q = np.broadcast_to(query["reserves"][None], (kb, nq, 2))
        res_s = np.broadcast_to(support["reserves"][None], (kb, ns, 2))
    sup_q = np.clip(supply_means(query["factors"])[None] + noise_q, 0, None)
    sup_s = np.clip(supply_means(support["factors"])[None] + noise_s, 0, None)

    nl = len(laws)
    counts = np.zeros((nl, 4, nq, 5, 3), np.int32)
    lifetime = np.zeros((nl, nq, 5, STEPS + 1), np.int32)
    patterns = np.zeros((nl, kb, nq), np.int64)      # joint five-action code (correct variant)
    types = row_types(query["factors"])
    sims = 0
    for row_type in range(4):
        rows = np.flatnonzero(types == row_type)
        if not len(rows):
            continue
        factors = tile_rows(query["factors"][rows], kb)
        reserves = res_q[:, rows].reshape(kb * len(rows), 2)
        for action in range(5):
            keys, mapping = variant_keys(laws, row_type, action)
            results = {}
            for key in keys:
                mode, cues, flip = key
                signals = query["signals"][rows].copy()
                if flip is not None:
                    signals[:, flip] ^= 1
                supplies = mode_frame(sup_q[:, rows], mode).reshape(kb * len(rows), STEPS, 2)
                out = simulate(reserves, factors, tile_rows(signals, kb), supplies, Law(mode, cues),
                               np.full(kb * len(rows), action))
                sims += kb * len(rows)
                results[key] = (out.survival.reshape(kb, len(rows), 3), out.lifetime.reshape(kb, len(rows)))
            for (index, variant), key in mapping.items():
                survival, life = results[key]
                slot = 0 if variant is None else 1 + variant
                counts[index, slot, rows, action] = survival.sum(axis=0, dtype=np.int64)
                if variant is None:
                    onehot = life[..., None] == np.arange(1, STEPS + 2)[None, None]
                    lifetime[index, rows, action] = onehot.sum(axis=0)
                    patterns[index][:, rows] += survival.sum(axis=2).astype(np.int64) * 4 ** action
    joint = []
    for index in range(nl):
        flat = np.arange(nq)[None, :] * 1024 + patterns[index]
        values, freq = np.unique(flat.ravel(), return_counts=True)
        joint.append((values.astype(np.int32), freq.astype(np.int32)))

    support_counts = np.zeros((nl, ns, 3), np.int32)
    factors_s = tile_rows(support["factors"], kb)
    signals_s = tile_rows(support["signals"], kb)
    reserves_s = res_s.reshape(kb * ns, 2)
    actions_s = tile_rows(support["actions"].astype(np.int64), kb)
    for index, (mode, active) in enumerate(laws):
        supplies = mode_frame(sup_s, mode).reshape(kb * ns, STEPS, 2)
        out = simulate(reserves_s, factors_s, signals_s, supplies, Law(mode, active), actions_s)
        sims += kb * ns
        support_counts[index] = out.survival.reshape(kb, ns, 3).sum(axis=0)
    return dict(key=task["key"], kb=kb, query_counts=counts, lifetime=lifetime, joint=joint,
                support_counts=support_counts, case_simulations=sims)


def naive_panel_counts(task):
    """Reference without row-type/action deduplication (for exactness checks)."""
    kb, laws = task["kb"], [tuple((m, tuple(a))) for m, a in task["laws"]]
    rng = np.random.default_rng([MC_ROOT, task["channel"], task["seed"], task["replicate"], task["batch"]])
    query, support = task["query"], task["support"]
    nq, ns = len(query["factors"]), len(support["factors"])
    noise_q = rng.normal(0., TransferWorld.supply_noise, (kb, nq, STEPS, 2))
    rng.normal(0., TransferWorld.supply_noise, (kb, ns, STEPS, 2))
    if task["gauge"]:
        res_q = sample_reserves(rng, query["levels"], kb)
    else:
        res_q = np.broadcast_to(query["reserves"][None], (kb, nq, 2))
    sup_q = np.clip(supply_means(query["factors"])[None] + noise_q, 0, None)
    counts = np.zeros((len(laws), 4, nq, 5, 3), np.int32)
    factors = tile_rows(query["factors"], kb)
    reserves = res_q.reshape(kb * nq, 2)
    for index, (mode, active) in enumerate(laws):
        supplies = mode_frame(sup_q, mode).reshape(kb * nq, STEPS, 2)
        for variant in (None, 0, 1, 2):
            signals = query["signals"].copy()
            if variant is not None and variant in active:
                signals[:, variant] ^= 1
            for action in range(5):
                out = simulate(reserves, factors, tile_rows(signals, kb), supplies, Law(mode, active),
                               np.full(kb * nq, action))
                counts[index, 0 if variant is None else 1 + variant, :, action] = \
                    out.survival.reshape(kb, nq, 3).sum(axis=0)
    return counts


def pair_task(task):
    """Performed-action pattern probabilities under two laws for fresh support records.

    Used by the direct posterior-identification check. Records are simulated by
    the project generator ``AcquisitionWorld.experience`` in 32-record batches
    with analysis-only seeds; the MC marginalises supply noise with common
    random numbers across the two laws.
    """
    kb = task["kb"]
    rng = np.random.default_rng([MC_ROOT, task["channel"], task["seed"], task["replicate"], task["batch"]])
    factors, signals, reserves, actions = (task[k] for k in ("factors", "signals", "reserves", "actions"))
    n = len(factors)
    probabilities = np.zeros((len(task["laws"]), n, 4))
    if n:
        noise = rng.normal(0., TransferWorld.supply_noise, (kb, n, STEPS, 2))
        sup0 = np.clip(supply_means(factors)[None] + noise, 0, None)
        for index, (mode, active) in enumerate(task["laws"]):
            supplies = mode_frame(sup0, mode).reshape(kb * n, STEPS, 2)
            out = simulate(tile_rows(reserves, kb), tile_rows(factors, kb), tile_rows(signals, kb),
                           supplies, Law(mode, tuple(active)), tile_rows(actions.astype(np.int64), kb))
            level = out.survival.sum(axis=1).reshape(kb, n)
            probabilities[index] = (level[..., None] == np.arange(4)).mean(axis=0)
    return dict(key=task["key"], probabilities=probabilities, case_simulations=kb * n * len(task["laws"]))
