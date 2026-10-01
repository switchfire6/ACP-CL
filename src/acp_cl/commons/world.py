"""Step D0: a three-entity resource commons with agent-chosen transfers.

Physics follows ``acp_cl.persistence.world.TransferWorld.simulate``:

* the transfer departs before step-1 supplies; the loss (1 - e) of the sent
  amount is booked at departure; a delay-0 transfer arrives immediately before
  the first supply, a delay-d transfer before the supply of step d+1;
* each step: arrivals, then supply, then clipping at capacity (overflow booked),
  then maintenance (consumption capped at the non-negative stock), then the
  irreversible failure check (reserve <= 1e-10 after maintenance);
* physics continues after failure for accounting only.

Extensions for Step D (every choice is recorded in ``world_spec``):

* three entities, three directed channels 1->2, 2->3, 3->1 and seven actions;
* supplies are NOT clipped at 0 (negative supply is allowed and audited);
* real-valued delays: a delay d = k + f (k integer, 0 <= f < 1) delivers the
  fraction (1 - f) before the supply of step k+1 and f one step later, so the
  delay dial is continuous; integer delays reproduce the persistence convention;
* an external probe flow of .3 units per channel per step that never enters a
  reserve (it is measured at the channel outlet and discharged to an external
  sink), so its likelihood is exact and its level (passive on / active off)
  never changes survival.

Everything here is evaluator-side. Contexts are integers 0..7 with bits
A = (c >> 2) & 1, B = (c >> 1) & 1, C = c & 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json

import numpy as np


T = 12
N_ENTITIES = 3
N_CONTEXTS = 8
HORIZONS = (4, 8, 12)
CAPACITY = 18.0
MAINTENANCE = 2.0
SUPPLY_NOISE = 0.35
OBS_NOISE = 0.1
PROBE = 0.3
AMOUNT = 3.0
FAIL_TOL = 1e-10
RESERVE_LOW, RESERVE_HIGH = 1.0, 13.0
SURVIVED = T + 1  # failure-step code for "alive after step 12"

CHANNELS = ((0, 1), (1, 2), (2, 0))  # forward direction entry -> exit (0-based)
CHANNEL_NAMES = ("1->2", "2->3", "3->1")
ACTION_NAMES = ("none", "1->2", "2->1", "2->3", "3->2", "3->1", "1->3")
N_ACTIONS = 7
ACTION_CHANNEL = np.array([-1, 0, 0, 1, 1, 2, 2])
ACTION_DONOR = np.array([-1, 0, 1, 1, 2, 2, 0])
ACTION_RECIPIENT = np.array([-1, 1, 0, 2, 1, 0, 2])


def action_table(params=None):
    """Per action: channel, donor, recipient, amount (index 0 = no transfer).

    Actions 1..6 repeat the D0 order (1->2, 2->1, 2->3, 3->2, 3->1, 1->3) for
    each amount in ``params.amounts``; the default reproduces D0's 7 actions.
    """
    amounts = (AMOUNT,) if params is None else tuple(params.amounts)
    channel, donor, recipient, amount = [-1], [-1], [-1], [0.0]
    for size in amounts:
        channel += list(ACTION_CHANNEL[1:])
        donor += list(ACTION_DONOR[1:])
        recipient += list(ACTION_RECIPIENT[1:])
        amount += [float(size)] * 6
    return np.array(channel), np.array(donor), np.array(recipient), np.array(amount)


def n_actions(params=None):
    return 1 + 6 * (1 if params is None else len(params.amounts))


def action_names(params=None):
    amounts = (AMOUNT,) if params is None else tuple(params.amounts)
    names = ["none"]
    for size in amounts:
        names += [f"{n}x{size:g}" if len(amounts) > 1 else n for n in ACTION_NAMES[1:]]
    return names


def factors(context):
    context = np.asarray(context)
    return (context >> 2) & 1, (context >> 1) & 1, context & 1


def context_name(context):
    a, b, c = (int(v) for v in factors(int(context)))
    return f"A{a}B{b}C{c}"


def context_index(a, b, c):
    return 4 * a + 2 * b + c


@dataclass(frozen=True)
class WorldParams:
    """Fixed physics plus the three graded dials (flip_amplitude, b_gap, c_delta)."""

    mean_supply: float = 1.9
    imbalance: float = 0.8
    season_flip_step: int = 6
    # Seasonal sign of each entity at A = 0 (+1: surplus in the first half).
    base_signs: tuple = (1.0, -1.0, 0.0)
    entity3_mean_offset: float = 0.0
    e12: float = 0.9
    d12: float = 1.0
    e23: float = 0.9
    d23_fast: float = 0.0
    e31: float = 0.8
    d31: float = 2.0
    flip_amplitude: float = 1.0   # dial A: A=1 multiplies signs of entities 1,2 by (1 - 2*amp)
    b_gap: float = 0.3            # dial B: e12 at B=1 is e12 - b_gap
    c_delta: float = 2.0          # dial C: slow delay on 2<->3 is d23_fast + c_delta
    probe_start: str = "episode"  # "episode": channel empty at t=0; "steady": flow already running
    # ---- D0b extensions (defaults reproduce D0 exactly and are omitted from to_json) ----
    amounts: tuple = (3.0,)       # transfer sizes; actions = none + 6 directed channel moves per amount
    b_mode: str = "ch12"          # "ch12": B=1 lowers e12; "both": B=1 lowers e12 and e31;
                                  # "swap": B=1 lowers e12, B=0 lowers e31 (the efficient route swaps)
    c_mode: str = "ch23"          # "ch23": D0; "same": every channel in c_channels fast iff C=1 and cue1=1;
                                  # "route": C=1 -> cue1=1 makes 2<->3 fast, cue1=0 makes 3<->1 fast; C=0 both slow
    c_channels: tuple = (1,)      # channels gated by C (1 = 2<->3, 2 = 3<->1); fast delay = d23_fast / d31 / d12

    def __post_init__(self):
        if self.probe_start not in ("episode", "steady"):
            raise ValueError("probe_start must be 'episode' or 'steady'")
        if not 0 < self.e12 - self.b_gap <= 1 or not 0 <= self.c_delta:
            raise ValueError("invalid dials")
        if self.b_mode not in ("ch12", "both", "swap") or self.c_mode not in ("ch23", "same", "route"):
            raise ValueError("invalid factor mode")
        if self.b_mode != "ch12" and not 0 < self.e31 - self.b_gap <= 1:
            raise ValueError("b_gap too large for channel 3->1")
        if self.c_mode == "route" and tuple(sorted(self.c_channels)) != (1, 2):
            raise ValueError("route mode gates channels 2<->3 and 3<->1")
        if self.c_mode == "ch23" and tuple(self.c_channels) != (1,):
            raise ValueError("ch23 mode gates only channel 2<->3")
        if not self.amounts or any(a <= 0 for a in self.amounts):
            raise ValueError("amounts must be positive")

    def with_dials(self, **dials):
        return replace(self, **dials)

    def to_json(self):
        out = asdict(self)
        defaults = {"amounts": (3.0,), "b_mode": "ch12", "c_mode": "ch23", "c_channels": (1,)}
        for key, value in defaults.items():
            current = tuple(out[key]) if isinstance(out[key], (tuple, list)) else out[key]
            if current == value:
                del out[key]
            elif isinstance(current, tuple):
                out[key] = list(current)
        return out

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()


def supply_means(params: WorldParams) -> np.ndarray:
    """Mean supply mu[A, t, i] for A in {0, 1}; shape (2, T, 3)."""
    season = np.where(np.arange(T) < params.season_flip_step, 1.0, -1.0)
    out = np.empty((2, T, N_ENTITIES))
    for a in (0, 1):
        signs = np.array(params.base_signs, dtype=float)
        if a:
            signs[:2] *= 1.0 - 2.0 * params.flip_amplitude
        out[a] = params.mean_supply + params.imbalance * season[:, None] * signs[None, :]
        out[a, :, 2] += params.entity3_mean_offset
    return out


def channel_params(params: WorldParams, context, cue1):
    """Efficiency and delay of each channel, shape (n, 3) each.

    Reverse transfers use the same (efficiency, delay) as the forward channel:
    a channel is one physical pipe per entity pair.
    """
    context = np.atleast_1d(np.asarray(context))
    cue1 = np.broadcast_to(np.asarray(cue1), context.shape)
    _, b, c = factors(context)
    n = context.shape[0]
    eff = np.empty((n, 3))
    delay = np.empty((n, 3))
    eff[:, 0] = np.where(b == 1, params.e12 - params.b_gap, params.e12)
    eff[:, 1] = params.e23
    eff[:, 2] = params.e31
    if params.b_mode == "both":
        eff[:, 2] = np.where(b == 1, params.e31 - params.b_gap, params.e31)
    elif params.b_mode == "swap":
        eff[:, 2] = np.where(b == 0, params.e31 - params.b_gap, params.e31)
    fast_delay = (params.d12, params.d23_fast, params.d31)
    delay[:, 0] = params.d12
    delay[:, 2] = params.d31
    if params.c_mode == "ch23":
        fast = (c == 1) & (cue1 == 1)  # C=1: fast when cue 1 is aligned (= 1)
        delay[:, 1] = np.where(fast, params.d23_fast, params.d23_fast + params.c_delta)
    else:
        for k in params.c_channels:
            if params.c_mode == "same":
                fast = (c == 1) & (cue1 == 1)
            else:  # route: the cue picks which gated channel is fast under C=1
                fast = (c == 1) & (cue1 == (1 if k == 1 else 0))
            delay[:, k] = np.where(fast, fast_delay[k], fast_delay[k] + params.c_delta)
    return eff, delay


def split_delay(delay):
    """Integer part k and fraction f: (1-f) arrives at step k, f at step k+1."""
    delay = np.asarray(delay, dtype=float)
    k = np.floor(delay)
    return k.astype(np.int64), delay - k


def arrival_profile(amount, delay):
    """Arrival amount per step (…, T) for an amount departing before step 0."""
    amount = np.asarray(amount, dtype=float)
    k, f = split_delay(delay)
    k, f = np.broadcast_arrays(k, f)
    amount = np.broadcast_to(amount, k.shape)
    steps = np.arange(T)
    out = (np.where(steps == k[..., None], amount[..., None] * (1.0 - f[..., None]), 0.0)
           + np.where(steps == k[..., None] + 1, amount[..., None] * f[..., None], 0.0))
    return out


def probe_arrivals(params: WorldParams, eff, delay, probes_on=True):
    """Probe arrivals per step per channel, shape (n, T, 3). Deterministic given context and cue."""
    eff = np.asarray(eff)
    n = eff.shape[0]
    if not probes_on:
        return np.zeros((n, T, 3))
    if params.probe_start == "steady":
        return np.broadcast_to(PROBE * eff[:, None, :], (n, T, 3)).copy()
    k, f = split_delay(delay)
    steps = np.arange(T)[None, :, None]
    # Injection at every step 0..T-1; arrivals at t come from injections t-k (1-f) and t-k-1 (f).
    first = (steps - k[:, None, :] >= 0).astype(float) * (1.0 - f[:, None, :])
    second = (steps - k[:, None, :] - 1 >= 0).astype(float) * f[:, None, :]
    return PROBE * eff[:, None, :] * (first + second)


@dataclass
class Episodes:
    """Evaluator-owned episode draws. Supplies are mu(A) + SUPPLY_NOISE * z_supply.

    The standard-normal draws are context-free, so the same Episodes object can
    be replayed under any context with common random numbers.
    """

    reserves: np.ndarray       # (n, 3)
    cue: np.ndarray            # (n, 3) uint8; cue[:, 0] is "cue 1"
    z_supply: np.ndarray       # (n, T, 3)
    z_obs_supply: np.ndarray   # (n, T, 3)
    z_obs_probe: np.ndarray    # (n, T, 3)

    def __len__(self):
        return len(self.reserves)

    def take(self, index):
        return Episodes(*(getattr(self, k)[index] for k in
                          ("reserves", "cue", "z_supply", "z_obs_supply", "z_obs_probe")))

    def supplies(self, params: WorldParams, context):
        a, _, _ = factors(np.broadcast_to(np.asarray(context), (len(self),)))
        return supply_means(params)[a] + SUPPLY_NOISE * self.z_supply

    def fingerprint(self):
        digest = hashlib.sha256()
        for key in ("reserves", "cue", "z_supply", "z_obs_supply", "z_obs_probe"):
            value = np.ascontiguousarray(getattr(self, key))
            digest.update(str((key, value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()


def seed_for(*keys) -> int:
    text = json.dumps(list(keys), sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def sample_episodes(n, seed) -> Episodes:
    rng = np.random.default_rng(seed)
    reserves = rng.uniform(RESERVE_LOW, RESERVE_HIGH, (n, N_ENTITIES))
    cue = rng.integers(0, 2, (n, 3), dtype=np.uint8)
    z_supply = rng.standard_normal((n, T, N_ENTITIES))
    z_obs_supply = rng.standard_normal((n, T, N_ENTITIES))
    z_obs_probe = rng.standard_normal((n, T, 3))
    return Episodes(reserves, cue, z_supply, z_obs_supply, z_obs_probe)


@dataclass
class SimResult:
    fail_step: np.ndarray        # (n, 3) int8; 1..12, or SURVIVED
    joint_survival: np.ndarray   # (n,) bool, all alive after step 12
    horizon_survival: np.ndarray # (n, 3) bool, joint survival after steps 4, 8, 12
    sent: np.ndarray
    loss: np.ndarray
    overflow: np.ndarray         # (n, 3) total overflow per entity
    consumed: np.ndarray         # (n, 3)
    final_stock: np.ndarray      # (n, 3)
    in_transit_final: np.ndarray # (n,)
    min_stock_seen: np.ndarray   # (n, 3) min stock after maintenance
    arrival_step_first: np.ndarray  # (n,) first step with a positive transfer arrival (T if none)
    conservation_error: float
    probe_conservation_error: float


def transfer_setup(params, context, reserves, cue, action):
    """Departure bookkeeping shared by the full simulator and the entity kernels."""
    action = np.asarray(action)
    n = len(reserves)
    if action.shape != (n,) or not np.issubdtype(action.dtype, np.integer):
        raise ValueError("one integer action per episode is required")
    table_ch, table_donor, table_rec, table_amount = action_table(params)
    if np.any((action < 0) | (action >= len(table_ch))):
        raise ValueError("action out of range")
    eff, delay = channel_params(params, context, cue[:, 0])
    rows = np.arange(n)
    ch = table_ch[action]
    donor = table_donor[action]
    recipient = table_rec[action]
    moving = action > 0
    stock = np.array(reserves, dtype=float, copy=True)
    sent = np.zeros(n)
    sent[moving] = np.minimum(table_amount[action][moving], stock[rows[moving], donor[moving]])
    stock[rows[moving], donor[moving]] -= sent[moving]
    e = np.where(moving, eff[rows, np.maximum(ch, 0)], 1.0)
    d = np.where(moving, delay[rows, np.maximum(ch, 0)], 0.0)
    delivered = sent * e
    loss = sent - delivered
    arrivals = np.zeros((n, T, N_ENTITIES))
    profile = arrival_profile(delivered, d)  # (n, T)
    arrivals[rows[moving], :, recipient[moving]] = profile[moving]
    return stock, sent, delivered, loss, arrivals, eff, delay


def simulate(params: WorldParams, context, episodes: Episodes, action, supplies=None,
             probes_on=True) -> SimResult:
    """Vectorised full simulation of n episodes (context and action per episode)."""
    n = len(episodes)
    context = np.broadcast_to(np.asarray(context), (n,)).astype(np.int64)
    action = np.broadcast_to(np.asarray(action), (n,)).astype(np.int64)
    if supplies is None:
        supplies = episodes.supplies(params, context)
    stock, sent, delivered, loss, arrivals, eff, delay = transfer_setup(
        params, context, episodes.reserves, episodes.cue, action)
    initial_total = np.asarray(episodes.reserves, dtype=float).sum(axis=1)
    in_transit = delivered.copy()
    total_supply = np.zeros(n)
    overflow_total = np.zeros((n, N_ENTITIES))
    consumed_total = np.zeros((n, N_ENTITIES))
    fail_step = np.full((n, N_ENTITIES), SURVIVED, dtype=np.int8)
    horizon = np.zeros((n, len(HORIZONS)), dtype=bool)
    min_stock = np.full((n, N_ENTITIES), np.inf)
    arrived_any = arrivals.sum(axis=2) > 0
    first_arrival = np.where(arrived_any.any(axis=1), arrived_any.argmax(axis=1), T)
    max_error = 0.0
    for t in range(T):
        stock = stock + arrivals[:, t]
        in_transit = in_transit - arrivals[:, t].sum(axis=1)
        s = supplies[:, t]
        total_supply += s.sum(axis=1)
        stock = stock + s
        over = np.maximum(stock - CAPACITY, 0.0)
        overflow_total += over
        stock = np.minimum(stock, CAPACITY)
        consumed = np.clip(stock, 0.0, MAINTENANCE)
        consumed_total += consumed
        stock = stock - consumed
        min_stock = np.minimum(min_stock, stock)
        newly = (fail_step == SURVIVED) & (stock <= FAIL_TOL)
        fail_step[newly] = t + 1
        if t + 1 in HORIZONS:
            horizon[:, HORIZONS.index(t + 1)] = (fail_step == SURVIVED).all(axis=1)
        balance = (stock.sum(axis=1) + in_transit + loss + overflow_total.sum(axis=1)
                   + consumed_total.sum(axis=1) - initial_total - total_supply)
        max_error = max(max_error, float(np.max(np.abs(balance))) if n else 0.0)
    # Probe ledger (external flow, never enters reserves).
    probe_error = 0.0
    if probes_on and params.probe_start == "episode":
        arrived = probe_arrivals(params, eff, delay, True).sum(axis=1)  # (n, 3)
        injected = PROBE * T
        lost = injected * (1 - eff)
        k, f = split_delay(delay)
        # Delivered mass still in transit at the end: injections whose arrival is >= T.
        pending = PROBE * eff * (np.clip(k, 0, T) * (1 - f) + np.clip(k + 1, 0, T) * f)
        probe_error = float(np.max(np.abs(injected - lost - arrived - pending))) if n else 0.0
    return SimResult(fail_step, fail_step.min(axis=1) == SURVIVED, horizon, sent, loss,
                     overflow_total, consumed_total, stock, in_transit, min_stock,
                     first_arrival, max_error, probe_error)


def observe(params: WorldParams, context, episodes: Episodes, supplies, probes_on=True):
    """Post-episode sensors: noisy supply and probe-arrival readings (sigma = OBS_NOISE)."""
    n = len(episodes)
    context = np.broadcast_to(np.asarray(context), (n,))
    eff, delay = channel_params(params, context, episodes.cue[:, 0])
    y_supply = supplies + OBS_NOISE * episodes.z_obs_supply
    y_probe = probe_arrivals(params, eff, delay, probes_on) + OBS_NOISE * episodes.z_obs_probe
    return y_supply, y_probe


def entity_kernel(x0, arrivals, supplies, want_step=False):
    """Single-entity replay, bit-identical to ``simulate`` for that entity.

    x0: (...,) reserve after departure; arrivals: (..., T) or None; supplies: (..., T).
    Returns alive-after-T (bool) or the failure step (int8, SURVIVED if alive).
    """
    x = np.array(x0, dtype=float, copy=True)
    shape = np.broadcast_shapes(x.shape, supplies.shape[:-1])
    x = np.broadcast_to(x, shape).copy()
    if want_step:
        step = np.full(shape, SURVIVED, dtype=np.int8)
    else:
        alive = np.ones(shape, dtype=bool)
    for t in range(T):
        if arrivals is not None:
            x = x + arrivals[..., t]
        x = x + supplies[..., t]
        x = np.minimum(x, CAPACITY)
        x = x - np.clip(x, 0.0, MAINTENANCE)
        if want_step:
            step[(step == SURVIVED) & (x <= FAIL_TOL)] = t + 1
        else:
            alive &= x > FAIL_TOL
    return step if want_step else alive


def world_spec(params: WorldParams):
    mu = supply_means(params)
    return {
        "entities": N_ENTITIES, "T": T, "horizons_reported": list(HORIZONS),
        "objective": "joint survival of all three entities after step 12",
        "capacity": CAPACITY, "maintenance_per_step": MAINTENANCE,
        "initial_reserves": f"iid U({RESERVE_LOW}, {RESERVE_HIGH}) per entity per episode; episodes independent",
        "supply": f"s_i(t) ~ N(mu_i(A, t), {SUPPLY_NOISE}^2), NOT clipped; negative supply allowed and audited",
        "supply_mean": ("mu_i(A,t) = mean_supply + imbalance * season(t) * sign_i(A) (+ entity3_mean_offset for i=3); "
                        "season(t) = +1 for steps 1..6, -1 for steps 7..12; sign(A=0) = base_signs; "
                        "A=1 multiplies the signs of entities 1 and 2 by (1 - 2*flip_amplitude) (amplitude 1 = full flip)"),
        "supply_mean_table_A0": mu[0].round(4).tolist(),
        "supply_mean_table_A1": mu[1].round(4).tolist(),
        "channels": {"1->2": {"efficiency": f"{params.e12} (B=0) / {params.e12 - params.b_gap:.4f} (B=1)", "delay": params.d12},
                     "2->3": {"efficiency": params.e23,
                              "delay": (f"C=0: {params.d23_fast + params.c_delta} always; C=1: {params.d23_fast} if cue1 = 1 "
                                        f"(aligned, fast) else {params.d23_fast + params.c_delta}")},
                     "3->1": {"efficiency": params.e31, "delay": params.d31}},
        "reverse_transfers": "use the same channel (efficiency, delay) as the forward direction: one physical pipe per pair",
        "actions": {i: name for i, name in enumerate(ACTION_NAMES)},
        "transfer_amount": f"sent = min({AMOUNT}, donor reserve); loss (1-e)*sent booked at departure",
        "timing": ("transfer departs before step 1; a delay d = k + f delivers (1-f) of the delivered amount before the supply of step k+1 "
                   "and f one step later (integer delays reproduce the persistence convention; d >= 12 never arrives)"),
        "step_order": "arrivals -> supply -> clip at capacity (overflow booked) -> maintenance min(max(stock,0), 2) -> failure check (reserve <= 1e-10)",
        "failure": "irreversible; physics continues after failure for accounting only (a failed entity still receives supply/arrivals)",
        "probe": (f"external {PROBE} units per channel per step injected at the channel entry, loss (1-e) at injection, travels with the "
                  "channel's delay (split rule as transfers), measured at the outlet and discharged to an external sink: it NEVER enters a "
                  "reserve, so probe level does not change survival or optimal actions"),
        "probe_start": {"value": params.probe_start,
                        "episode": "channel empty at t=0: probe arrivals start after the delay (onset reveals delay)",
                        "steady": "probe flow already running before the episode: arrivals are .3e every step (delay invisible to the probe)"},
        "probe_levels": {"passive": "probes on", "active": "probes off (channel sensors read pure noise)"},
        "observations_decision_time": "exact reserves (3) + 3-bit cue (iid uniform); cue bit 1 matters only when C=1; bits 2,3 are distractors",
        "observations_post_episode": (f"supply sensors y = s + N(0,{OBS_NOISE}^2) per entity per step; probe sensors y = probe arrivals + "
                                      f"N(0,{OBS_NOISE}^2) per channel per step (the agent's own transfer is NOT sensed); failure step "
                                      "(1..12) or survival per entity; mid-episode reserves are not observed"),
        "contexts": "c = 4A + 2B + C; A: seasonality of entities 1,2 flipped; B: channel 1->2 efficiency lowered by b_gap; "
                    "C: channel 2<->3 delay fast (d23_fast) only when C=1 and cue1 = 1, otherwise d23_fast + c_delta",
        **({} if (tuple(params.amounts), params.b_mode, params.c_mode) == ((3.0,), "ch12", "ch23") else {"d0b_options": {
            "amounts": list(params.amounts),
            "actions": action_names(params),
            "b_mode": {"ch12": "B=1 lowers channel 1->2 efficiency by b_gap",
                       "both": "B=1 lowers channels 1->2 and 3->1 by b_gap",
                       "swap": (f"B=0: 1->2 at {params.e12}, 3->1 at {params.e31 - params.b_gap:.4f}; "
                                f"B=1: 1->2 at {params.e12 - params.b_gap:.4f}, 3->1 at {params.e31} (the efficient route swaps)")}[params.b_mode],
            "c_mode": {"ch23": "C gates the 2<->3 delay only (fast iff C=1 and cue1=1)",
                       "same": f"channels {list(params.c_channels)} are fast iff C=1 and cue1=1, else fast + c_delta",
                       "route": "C=1: cue1=1 makes 2<->3 fast, cue1=0 makes 3<->1 fast; C=0: both slow (fast + c_delta)"}[params.c_mode],
            "note": "the 'channels' entry above describes the D0 wiring; d0b_options override it"}}),
        "params": params.to_json(),
        "params_sha256": params.digest(),
    }
