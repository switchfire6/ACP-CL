"""A conserving, finite-horizon two-participant transfer world.

Each trial permits one transfer, followed by exogenous supplies and maintenance.
Training records contain the outcome of ONE randomized action. Counterfactual
actions and latent regimes exist only in the evaluator, never in learner input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


ACTIONS = np.array([0.0, -2.0, 2.0, -4.0, 4.0], dtype=np.float64)
HORIZONS = (4, 8, 12)
IMAGE_SHAPE = (4, 12, 16)


@dataclass(frozen=True)
class Regime:
    name: str
    source: int
    lossy: int
    delayed: int
    gated: bool = False
    reversed: bool = False

    def __post_init__(self):
        if self.source not in (-1, 1) or self.lossy not in (0, 1) or self.delayed not in (0, 1):
            raise ValueError("invalid regime factors")


KNOWN = (
    Regime("a_clean_now", 1, 0, 0),
    Regime("b_clean_later", -1, 0, 1),
    Regime("a_lossy_later", 1, 1, 1),
    Regime("b_lossy_now", -1, 1, 0),
)
COMPOSITIONS = (
    Regime("a_clean_later", 1, 0, 1),
    Regime("b_clean_now", -1, 0, 0),
    Regime("a_lossy_now", 1, 1, 0),
    Regime("b_lossy_later", -1, 1, 1),
)


@dataclass
class Cases:
    observations: np.ndarray
    reserves: np.ndarray
    supplies: np.ndarray
    efficiency: np.ndarray
    delay: np.ndarray


@dataclass
class Outcomes:
    survival: np.ndarray
    lifetime: np.ndarray
    conservation_error: float
    min_stock: float


@dataclass
class Experience:
    observations: np.ndarray
    actions: np.ndarray
    survival: np.ndarray

    def __len__(self):
        return len(self.actions)

    def take(self, indices):
        return Experience(self.observations[indices], self.actions[indices], self.survival[indices])

    def fingerprint(self):
        digest = hashlib.sha256()
        for value in (self.observations, self.actions, self.survival):
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()


def concatenate(first: Experience, second: Experience) -> Experience:
    return Experience(*(np.concatenate((a, b)) for a, b in zip(
        (first.observations, first.actions, first.survival),
        (second.observations, second.actions, second.survival))))


class TransferWorld:
    capacity = 18.0
    maintenance = 2.0
    mean_supply = 1.9
    imbalance = 0.8
    supply_noise = 0.35

    def __init__(self):
        # Fixed sensor alphabet shared across cohorts. No pretrained features.
        rng = np.random.default_rng(16031991)
        self.glyphs = (rng.random((3, 2, 3, 3)) > 0.5).astype(np.uint8) * 180 + 35

    def cases(self, regime: Regime, size: int, seed: int) -> Cases:
        if size < 1:
            raise ValueError("size must be positive")
        rng = np.random.default_rng(seed)
        reserves = rng.uniform(1.0, 13.0, (size, 2))
        gate = rng.integers(0, 2, size) if regime.gated else np.zeros(size, dtype=int)
        source = np.full(size, regime.source) * (1 - 2 * gate)
        if regime.reversed:
            source *= -1
        # Supply advantage changes hands halfway through the finite trial.
        # Sending everything early can now leave the donor unable to bridge
        # its later shortage; reserves, loss, and delay all matter for action.
        season = np.ones(max(HORIZONS))
        season[max(HORIZONS) // 2:] = -1
        means = (self.mean_supply + self.imbalance * source[:, None, None]
                 * season[None, :, None] * np.array([1, -1]))
        supplies = np.clip(means + rng.normal(
            0, self.supply_noise, (size, max(HORIZONS), 2)), 0, None)
        observations = self.render(reserves, regime, gate, rng)
        return Cases(observations, reserves, supplies,
                     np.full(size, 0.35 if regime.lossy else 1.0),
                     np.full(size, 2 if regime.delayed else 0, dtype=int))

    def render(self, reserves, regime, gate, rng):
        size = len(reserves)
        images = rng.integers(0, 12, (size, *IMAGE_SHAPE), dtype=np.uint8)
        # Calibrated resource gauges are observations, not computed policy features.
        for participant, start in ((0, 1), (1, 9)):
            level = np.rint(reserves[:, participant] / self.capacity * 240).astype(np.uint8)
            images[:, :, 0:4, start:start + 6] = level[:, None, None, None]
        for field, col in enumerate((1, 6, 11)):
            values = (np.full(size, regime.lossy), np.full(size, regime.delayed), gate)[field]
            images[:, :, 5:8, col:col + 3] = self.glyphs[field, values][:, None]
        # The final weather position is identical for both source directions.
        # Earlier frames reveal motion; final-frame appearance cannot reveal it.
        for frame in range(4):
            col = 8 - (3 - frame) * regime.source
            images[:, frame, 10, col] = 255
        return images

    def simulate(self, cases: Cases, actions: np.ndarray) -> Outcomes:
        """Evaluate chosen actions with explicit transport, loss, overflow and use.

        Positive actions move A -> B. Departures occur before step-one supplies;
        delay zero arrives immediately, delay two before step-three maintenance.
        Failure is checked after each maintenance operation and is irreversible.
        Reservoir physics continues after failure only for accounting; no revival
        is credited. Consumed resources are capped at the resources available.
        """
        actions = np.asarray(actions)
        size = len(cases.reserves)
        if actions.shape != (size,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("one integer action index is required per case")
        if np.any((actions < 0) | (actions >= len(ACTIONS))):
            raise ValueError("action out of range")
        stock = cases.reserves.copy()
        initial_total = stock.sum(axis=1)
        request = ACTIONS[actions]
        donor = (request < 0).astype(int)
        recipient = 1 - donor
        rows = np.arange(size)
        sent = np.minimum(np.abs(request), stock[rows, donor])
        stock[rows, donor] -= sent
        in_transit = sent * cases.efficiency
        losses = sent - in_transit
        total_supply = np.zeros(size)
        total_consumed = np.zeros(size)
        total_overflow = np.zeros(size)
        alive = np.ones(size, dtype=bool)
        lifetime = np.full(size, max(HORIZONS) + 1, dtype=np.int16)
        survival = np.zeros((size, len(HORIZONS)), dtype=np.uint8)
        max_error, min_stock = 0.0, float(stock.min())
        for step in range(max(HORIZONS)):
            arriving = cases.delay == step
            stock[rows[arriving], recipient[arriving]] += in_transit[arriving]
            in_transit[arriving] = 0
            supply = cases.supplies[:, step]
            total_supply += supply.sum(axis=1)
            stock += supply
            overflow = np.maximum(stock - self.capacity, 0)
            total_overflow += overflow.sum(axis=1)
            stock = np.minimum(stock, self.capacity)
            consumed = np.minimum(stock, self.maintenance)
            total_consumed += consumed.sum(axis=1)
            stock -= consumed
            failed = np.any(stock <= 1e-10, axis=1)
            lifetime[alive & failed] = step + 1
            alive &= ~failed
            if step + 1 in HORIZONS:
                survival[:, HORIZONS.index(step + 1)] = alive
            balance = (stock.sum(axis=1) + in_transit + losses + total_overflow + total_consumed
                       - initial_total - total_supply)
            max_error = max(max_error, float(np.max(np.abs(balance))))
            min_stock = min(min_stock, float(stock.min()))
        return Outcomes(survival, lifetime, max_error, min_stock)

    def record(self, regime: Regime, size: int, seed: int) -> Experience:
        cases = self.cases(regime, size, seed)
        rng = np.random.default_rng(seed ^ 0x6A09E667)
        actions = rng.integers(0, len(ACTIONS), size, dtype=np.uint8)
        outcomes = self.simulate(cases, actions)
        return Experience(cases.observations, actions, outcomes.survival)

    def counterfactuals(self, cases: Cases):
        return np.stack([self.simulate(cases, np.full(len(cases.reserves), action)).survival
                         for action in range(len(ACTIONS))], axis=1)


def schedule(seed: int, name: str):
    """Schedules are evaluator-only. Learner receives one uninterrupted stream.

    Short/long have identical multisets, counts, and terminal-return position;
    only placement of the penultimate target exposure changes (gaps 1 vs 5).
    """
    if name not in ("short", "long", "no_return", "reversal"):
        raise ValueError("unknown schedule")
    order = np.random.default_rng(seed ^ 41041).permutation(4).tolist()
    base = [KNOWN[i] for i in order]
    middle = [0, 1, 2, 3, 1, 2] if name != "short" else [1, 2, 3, 1, 0, 2]
    target = base[0]
    return_regime = COMPOSITIONS[order[0]] if name == "no_return" else target
    late_basis = return_regime if name == "no_return" else target
    late = Regime("late_gate", late_basis.source, late_basis.lossy, late_basis.delayed, gated=True)
    if name == "reversal":
        late = Regime("changed_rule", target.source, target.lossy, target.delayed, reversed=True)
    regimes = base + [base[i] for i in middle] + [return_regime, late]
    return regimes, {"order": order, "return_block": 10, "late_block": 11,
                     "target": target.name, "gap_blocks": 1 if name == "short" else 5,
                     "target_returns": name != "no_return"}
