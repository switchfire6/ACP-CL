"""Independent reimplementation check of the Bayes ideal-observer audit.

POST HOC / EXPLORATORY / NOT DECISION-BEARING. This script is an analysis
reference for the locked representation-learning qualification run. It never
trains, updates, unpickles or executes a learner or checkpoint, never writes
into runs/, and cannot alter the locked STOP decision.

Law identities, hidden modes, clean physical outcomes and counterfactual
outcomes are EVALUATOR-ONLY quantities. The ideal observer below is an
analysis reference, not a learner.

Independence: this file was written without reading
scripts/ideal_observer_representation.py, scripts/ideal_observer_mc.py or
reports/information_audit/ideal_observer/results.json. It uses
scripts/ideal_observer_load.py only to read saved endpoint arrays (query
images, supports, truth, masks, learner forecasts) and to regenerate evaluator
cases for a pipeline check. Its own code does the following:
  * decodes every observable (reserve gauges, lossy/delayed glyphs, weather
    motion, pulse-order signals) from the uint8 frames;
  * samples hidden supply noise N(0, .35) clipped at 0 and the reserve
    within its gauge quantisation bin;
  * builds the law-specific physics by calling AcquisitionWorld.physics
    (through AcquisitionWorld.simulate) on those sampled cases;
  * computes per-law survival predictives, the support likelihoods, the law
    posterior, the ideal-observer mixture and the locked-style scores.

Definitions
  oracle O   : predictive P(y | query frames, true law), integrating hidden noise
               and gauge quantisation.
  ideal I    : sum_L w_L(support) P(y | query frames, L) over the 5 reference
               laws, uniform prior, w from the 32-record support likelihood.
  prior-only : uniform mixture over the 5 laws (no support).
Scores follow reports/information_audit/scoring_spec.md: all-case Brier, and
cue benefit = focus Brier(flipped forecast) - focus Brier(correct forecast) on
the cue's affected subset, per support replicate, then replicate-averaged.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from acp_cl.acquisition.world import AcquisitionWorld, Law, LawCases, order  # noqa: E402
from acp_cl.persistence.world import Cases  # noqa: E402

LABEL = "POST HOC / EXPLORATORY / NOT DECISION-BEARING"
CAPACITY, GAUGE = 18.0, 240.0
RESERVE_LOW, RESERVE_HIGH = 1.0, 13.0
MEAN, IMBALANCE, NOISE_SD = 1.9, 0.8, 0.35
STEPS = 12
GLYPH_COLUMNS = (1, 6, 11)
ROOT_SEED = 20260928
DEFAULT_OUTPUT = ROOT / "reports" / "information_audit" / "ideal_observer" / "independent_check.json"


def require(value, message):
    if not value:
        raise AssertionError(message)


# ----------------------------------------------------------------- the observer's world model

def my_laws(seed):
    """The five reference laws, built here from the documented rule (not imported)."""
    first = seed % 2
    cues = order(seed)
    return [Law(first), Law(1 - first)] + [Law(1 - first, tuple(sorted(cues[:s]))) for s in (1, 2, 3)]


def decode(images):
    """Recover every observable from (n,4,12,16) uint8 frames.

    Returns gauge levels (n,2), source (n,) in {-1,1}, factors (n,3) =
    (source, lossy, delayed) and signals (n,3). Asserts internal consistency.
    """
    images = np.asarray(images)
    require(images.dtype == np.uint8 and images.shape[1:] == (4, 12, 16), "bad image array")
    n = len(images)
    levels = np.stack([images[:, 0, 0, 1], images[:, 0, 0, 9]], axis=1).astype(np.int64)
    for participant, start in enumerate((1, 9)):
        block = images[:, :, 0:4, start:start + 6].reshape(n, -1)
        require((block == levels[:, participant, None]).all(), "gauge block is not uniform")
    glyphs = AcquisitionWorld().glyphs  # fixed sensor alphabet: part of the known simulator
    values = np.zeros((n, 3), dtype=np.int64)
    for field, col in enumerate(GLYPH_COLUMNS):
        patch = images[:, 0, 5:8, col:col + 3]
        match = [(patch == glyphs[field, v][None]).all(axis=(1, 2)) for v in (0, 1)]
        require(not (glyphs[field, 0] == glyphs[field, 1]).all(), "indistinguishable glyphs")
        require((match[0] ^ match[1]).all(), "glyph not decodable")
        values[:, field] = match[1]
        for frame in range(1, 4):
            require((images[:, frame, 5:8, col:col + 3] == patch).all(), "glyph differs across frames")
    require((values[:, 2] == 0).all(), "unexpected gated glyph")
    bright = images[:, 0, 10, :] == 255
    require((bright.sum(axis=1) == 1).all(), "weather marker not unique")
    column = bright.argmax(axis=1)
    require(np.isin(column, (5, 11)).all(), "weather marker in unexpected column")
    source = np.where(column == 5, 1, -1)
    for frame in range(4):
        expected = 8 - (3 - frame) * source
        require((images[np.arange(n), frame, 10, expected] == 255).all(), "weather motion inconsistent")
    signals = np.zeros((n, 3), dtype=np.uint8)
    rows = np.arange(n)
    for cue, col in enumerate(GLYPH_COLUMNS):
        bit = (images[:, 0, 8, col + 1] == 255).astype(np.uint8)
        for row in (8, 9):
            require((images[rows, 0, row, col + bit] == 255).all(), "pulse frame 0 inconsistent")
            require((images[rows, 2, row, col + 1 - bit] == 255).all(), "pulse frame 2 inconsistent")
        signals[:, cue] = bit
    factors = np.stack([source, values[:, 0], values[:, 1]], axis=1).astype(np.int64)
    return dict(levels=levels, factors=factors, signals=signals)


def affected_mask(factors, cue):
    if cue == 0:
        return factors[:, 1] == 1
    if cue == 1:
        return factors[:, 2] == 1
    if cue == 2:
        return (factors[:, 1] == 1) | (factors[:, 2] == 1)
    return np.ones(len(factors), dtype=bool)


def sample_hidden(rng, levels, source, draws, exact=None):
    """Own sampler: reserves uniform in the gauge bin (prior U(1,13)) and clipped supplies.

    Returns reserves (n,K,2) and PRE-MODE supplies (n,K,12,2) (column 0 = A)."""
    n = len(levels)
    if exact is None:
        low = np.maximum((levels - .5) * CAPACITY / GAUGE, RESERVE_LOW)
        high = np.minimum((levels + .5) * CAPACITY / GAUGE, RESERVE_HIGH)
        require((high > low).all(), "empty gauge bin")
        reserves = low[:, None, :] + rng.random((n, draws, 2)) * (high - low)[:, None, :]
    else:
        reserves = np.repeat(np.asarray(exact, dtype=np.float64)[:, None, :], draws, axis=1)
    season = np.where(np.arange(STEPS) < STEPS // 2, 1.0, -1.0)
    means = MEAN + IMBALANCE * source[:, None, None, None] * season[None, None, :, None] * np.array([1.0, -1.0])
    supplies = np.maximum(means + rng.normal(0.0, NOISE_SD, (n, draws, STEPS, 2)), 0.0)
    return reserves, supplies


def law_cases(law, reserves, supplies, factors, signals):
    """Flatten sampled hidden states into an AcquisitionWorld LawCases for one law."""
    m, draws = reserves.shape[:2]
    total = m * draws
    swapped = supplies[..., ::-1] if law.mode == 1 else supplies  # hidden mode swaps the columns
    base = Cases(np.zeros(total, dtype=np.uint8), reserves.reshape(total, 2),
                 np.ascontiguousarray(swapped).reshape(total, STEPS, 2),
                 np.repeat(np.where(factors[:, 1] == 1, .35, 1.0), draws),
                 np.repeat(np.where(factors[:, 2] == 1, 2, 0), draws).astype(np.int64))
    return LawCases(base.observations, base, np.repeat(factors, draws, axis=0),
                    np.repeat(signals, draws, axis=0), law)


def survival_probabilities(world, law, reserves, supplies, factors, signals):
    """Mean survival (m,5,3) for every action under one law, over the K draws."""
    m, draws = reserves.shape[:2]
    cases = law_cases(law, reserves, supplies, factors, signals)
    out = np.empty((m, 5, 3))
    for action in range(5):
        survival = world.simulate(cases, np.full(m * draws, action, dtype=np.int64)).survival
        out[:, action] = survival.reshape(m, draws, 3).mean(axis=1)
    return out


def pattern_counts(world, law, reserves, supplies, factors, signals, actions):
    """Counts (m,4) of the performed-action feedback pattern (# horizons survived)."""
    m, draws = reserves.shape[:2]
    cases = law_cases(law, reserves, supplies, factors, signals)
    survival = world.simulate(cases, np.repeat(np.asarray(actions, dtype=np.int64), draws)).survival
    survival = survival.reshape(m, draws, 3)
    require(np.all(np.diff(survival.astype(np.int16), axis=-1) <= 0), "non-nested survival")
    pattern = survival.sum(axis=-1)
    return np.stack([(pattern == p).sum(axis=1) for p in range(4)], axis=1)


# ----------------------------------------------------------------------- MC workers

def query_worker(seed, replicate, laws, observed, draws, chunk=32, exact=None):
    """Per-law query predictives for one independent MC replicate.

    Returns {(law_index, cue or -1): (512,5,3)}; flipped entries are computed on
    the cue's affected rows with common random numbers and equal the correct
    forecast elsewhere (physics there is independent of that cue).
    """
    started = time.perf_counter()
    world = AcquisitionWorld()
    rng = np.random.default_rng([ROOT_SEED, seed, replicate, 1 if exact is None else 2])
    levels, factors, signals = observed["levels"], observed["factors"], observed["signals"]
    n = len(levels)
    result = {}
    for index, law in enumerate(laws):
        result[index, -1] = np.empty((n, 5, 3))
        for cue in law.active:
            result[index, cue] = np.empty((n, 5, 3))
    for start in range(0, n, chunk):
        rows = slice(start, min(start + chunk, n))
        reserves, supplies = sample_hidden(rng, levels[rows], factors[rows, 0], draws,
                                           None if exact is None else exact[rows])
        for index, law in enumerate(laws):
            correct = survival_probabilities(world, law, reserves, supplies, factors[rows], signals[rows])
            result[index, -1][rows] = correct
            for cue in law.active:
                mask = affected_mask(factors[rows], cue)
                forecast = correct.copy()
                if mask.any():
                    flipped = signals[rows][mask].copy()
                    flipped[:, cue] ^= 1
                    forecast[mask] = survival_probabilities(world, law, reserves[mask], supplies[mask],
                                                            factors[rows][mask], flipped)
                result[index, cue][rows] = forecast
    return dict(kind="query", seed=seed, replicate=replicate, exact=exact is not None, draws=draws,
                forecasts=result, seconds=time.perf_counter() - started)


def support_worker(seed, replicate, laws, supports, draws, chunk=8):
    """Pattern counts (support_rep, law, 32, 4) for the saved supports."""
    started = time.perf_counter()
    world = AcquisitionWorld()
    rng = np.random.default_rng([ROOT_SEED, seed, replicate, 3])
    counts = []
    for observed, actions in supports:
        n = len(actions)
        per_law = np.zeros((len(laws), n, 4), dtype=np.int64)
        for start in range(0, n, chunk):
            rows = slice(start, min(start + chunk, n))
            reserves, supplies = sample_hidden(rng, observed["levels"][rows], observed["factors"][rows, 0], draws)
            for index, law in enumerate(laws):
                per_law[index, rows] = pattern_counts(world, law, reserves, supplies, observed["factors"][rows],
                                                      observed["signals"][rows], actions[rows])
        counts.append(per_law)
    return dict(kind="support", seed=seed, replicate=replicate, draws=draws, counts=np.stack(counts),
                seconds=time.perf_counter() - started)


def kl_worker(seed, replicate, delay_law, predecessor, records, draws, chunk=16):
    """Per-record KL(delay law || predecessor) under the performed-action feedback channel.

    Records follow the support/training record distribution: exactly half are
    delayed (4 per grid cell in a 32-record batch), reserves U(1,13) seen through
    the gauge, i.i.d. uniform pulse signals and uniform actions. Non-delayed
    records and action 0 have identical physics under both laws (asserted on a
    sample) and contribute exactly 0. Only delayed records are simulated here;
    the per-record KL over all records is 0.5 x the delayed mean.
    """
    started = time.perf_counter()
    world = AcquisitionWorld()
    rng = np.random.default_rng([ROOT_SEED, seed, replicate, 4])
    cells = [(s, lossy) for s in (-1, 1) for lossy in (0, 1)]
    cell = np.repeat(np.arange(4), records // 4)
    source = np.array([cells[c][0] for c in cell])
    lossy = np.array([cells[c][1] for c in cell])
    factors = np.stack([source, lossy, np.ones_like(source)], axis=1).astype(np.int64)
    true_reserves = rng.uniform(RESERVE_LOW, RESERVE_HIGH, (records, 2))
    levels = np.rint(true_reserves / CAPACITY * GAUGE).astype(np.int64)
    signals = rng.integers(0, 2, (records, 3)).astype(np.uint8)
    actions = rng.integers(0, 5, records)
    kl = np.zeros(records)
    kl_half = np.zeros(records)
    reverse = np.zeros(records)
    llr = np.zeros(records)
    zero_support = 0
    informative = np.flatnonzero(actions != 0)
    for start in range(0, len(informative), chunk):
        rows = informative[start:start + chunk]
        halves = []
        for _ in range(2):  # two independent halves: full-K and half-K estimates (bias check)
            reserves, supplies = sample_hidden(rng, levels[rows], source[rows], draws // 2)
            halves.append((pattern_counts(world, delay_law, reserves, supplies, factors[rows], signals[rows], actions[rows]),
                           pattern_counts(world, predecessor, reserves, supplies, factors[rows], signals[rows], actions[rows])))
        count_d = halves[0][0] + halves[1][0]
        count_p = halves[0][1] + halves[1][1]
        zero_support += int(((count_d > 0) & (count_p == 0)).sum() + ((count_p > 0) & (count_d == 0)).sum())
        p_d = (count_d + .5) / (draws + 2.0)
        p_p = (count_p + .5) / (draws + 2.0)
        kl[rows] = (p_d * np.log(p_d / p_p)).sum(axis=1)
        h_d = (halves[0][0] + .5) / (draws // 2 + 2.0)
        h_p = (halves[0][1] + .5) / (draws // 2 + 2.0)
        kl_half[rows] = (h_d * np.log(h_d / h_p)).sum(axis=1)
        reverse[rows] = (p_p * np.log(p_p / p_d)).sum(axis=1)
        outcome = np.array([rng.choice(4, p=row / row.sum()) for row in p_d])
        llr[rows] = np.log(p_d[np.arange(len(rows)), outcome] / p_p[np.arange(len(rows)), outcome])
    # Sanity check: action 0 and non-delayed records are physically identical under both laws.
    check = slice(0, 8)
    reserves, supplies = sample_hidden(rng, levels[check], source[check], 64)
    for fac, act in ((factors[check] * np.array([1, 1, 0]), np.full(8, 3)), (factors[check], np.zeros(8, dtype=int))):
        a = world.simulate(law_cases(delay_law, reserves, supplies, fac, signals[check]), np.repeat(act, 64)).survival
        b = world.simulate(law_cases(predecessor, reserves, supplies, fac, signals[check]), np.repeat(act, 64)).survival
        require(np.array_equal(a, b), "uninformative record differs between delay law and predecessor")
    return dict(kind="kl", seed=seed, replicate=replicate, draws=draws, cell=cell, kl=kl, kl_half=kl_half, reverse=reverse,
                llr=llr, actions=actions, zero_support=zero_support, seconds=time.perf_counter() - started)


# ----------------------------------------------------------------------- scoring

def brier(p, y, mask=None):
    error = np.square(np.asarray(p, dtype=np.float64) - y.astype(np.float64)).mean(axis=(1, 2))
    return float(error.mean() if mask is None else error[mask].mean())


def check_forecast(p):
    eps = np.finfo(np.float32).eps
    require(np.isfinite(p).all() and (p >= -eps).all() and (p <= 1 + eps).all()
            and (np.diff(p, axis=-1) <= eps).all(), "invalid forecast")


def posterior(counts, outcomes, draws, floor=.5):
    """Posterior over laws from (law, 32, 4) counts; uniform prior."""
    pattern = outcomes.astype(np.int64).sum(axis=1)
    require(np.all(np.diff(outcomes.astype(np.int16), axis=-1) <= 0), "support outcome not nested")
    observed = counts[:, np.arange(len(pattern)), pattern].astype(np.float64)
    loglik = np.log(np.maximum(observed, floor) / draws).sum(axis=1)
    weights = np.exp(loglik - loglik.max())
    return weights / weights.sum(), loglik, (observed == 0).sum(axis=1)


def score_seed(endpoint, laws, forecasts, counts, draws_support):
    """All per-seed quantities for one MC estimate (per-replicate or pooled)."""
    rows = {}
    posts = {}
    for target in endpoint.targets:
        slot = target.slot
        truth = target.correct[0].truth
        oracle = forecasts[slot, -1]
        prior = sum(forecasts[i, -1] for i in range(len(laws))) / len(laws)
        for p in (oracle, prior):
            check_forecast(p)
        row = dict(law=[target.law.mode, list(target.law.active)],
                   oracle_brier=brier(oracle, truth), prior_brier=brier(prior, truth),
                   ideal_brier=[], learner_brier=[], posterior=[],
                   oracle_cue_benefit={}, ideal_cue_benefit={}, prior_cue_benefit={}, learner_cue_benefit={})
        for rep, correct in enumerate(target.correct):
            weights, _, zeros = posterior(counts[rep], correct.support_outcomes, draws_support)
            posts[slot, rep] = weights
            ideal = sum(weights[i] * forecasts[i, -1] for i in range(len(laws)))
            check_forecast(ideal)
            row["ideal_brier"].append(brier(ideal, truth))
            row["learner_brier"].append(brier(correct.probabilities, truth))
            row["posterior"].append(weights.tolist())
            for cue, flips in target.flips.items():
                flip = flips[rep]
                mask = flip.affected
                ideal_flip = sum(weights[i] * forecasts[i, cue if cue in laws[i].active else -1]
                                 for i in range(len(laws)))
                prior_flip = sum(forecasts[i, cue if cue in laws[i].active else -1]
                                 for i in range(len(laws))) / len(laws)
                pairs = dict(oracle=(forecasts[slot, cue], oracle), ideal=(ideal_flip, ideal),
                             prior=(prior_flip, prior), learner=(flip.probabilities, correct.probabilities))
                for name, (flipped, base) in pairs.items():
                    row[f"{name}_cue_benefit"].setdefault(str(cue), []).append(
                        brier(flipped, truth, mask) - brier(base, truth, mask))
        for name in ("ideal_brier", "learner_brier"):
            row[name] = float(np.mean(row[name]))
        for name in ("oracle", "ideal", "prior", "learner"):
            key = f"{name}_cue_benefit"
            row[key] = {cue: float(np.mean(v)) for cue, v in row[key].items()}
        rows[slot] = row
    return rows, posts


def delay_stage(seed):
    return order(seed).index(1) + 1


# ----------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[18104, 18105, 18101])
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--query-draws", type=int, default=4096)
    parser.add_argument("--support-draws", type=int, default=65536)
    parser.add_argument("--kl-records", type=int, default=2048)
    parser.add_argument("--kl-draws", type=int, default=16384)
    parser.add_argument("--exact-seeds", type=int, nargs="*", default=[18104, 18105, 18101])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true", help="smoke test: print only")
    args = parser.parse_args()
    require(args.workers <= 6, "at most 6 worker processes")
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "reports" / "information_audit"), "outputs belong under reports/information_audit/")
    require(not output.exists(), "refusing to overwrite an existing output")

    sys.path.insert(0, str(ROOT / "scripts"))
    import ideal_observer_load as loader  # read-only loader (saved arrays + regeneration)

    started = time.perf_counter()
    endpoints, observed, supports, checks = {}, {}, {}, {}
    for seed in args.seeds:
        endpoint = loader.load_endpoint(seed=seed, phase="interleaved")
        laws = my_laws(seed)
        require([t.law for t in endpoint.targets] == laws, "my reference laws differ from the saved targets")
        endpoints[seed] = endpoint
        first = endpoint.targets[0].correct[0]
        query = decode(first.observations)
        # every correct probe shows the same panel; flipped probes show exactly that cue flipped
        for target in endpoint.targets:
            for rep, correct in enumerate(target.correct):
                require(np.array_equal(correct.observations, first.observations), "query panel differs across targets")
                require(np.array_equal(correct.affected, np.ones(512, bool)), "correct mask not all-case")
                for cue, flips in target.flips.items():
                    flipped = decode(flips[rep].observations)
                    expected = query["signals"].copy()
                    expected[:, cue] ^= 1
                    require(np.array_equal(flipped["signals"], expected)
                            and np.array_equal(flipped["factors"], query["factors"])
                            and np.array_equal(flipped["levels"], query["levels"]), "flip decode mismatch")
                    require(np.array_equal(flips[rep].affected, affected_mask(query["factors"], cue)),
                            "my affected mask differs")
        observed[seed] = query
        support_sets = []
        for rep in range(len(endpoint.targets[0].correct)):
            base = endpoint.targets[0].correct[rep]
            for target in endpoint.targets:
                require(np.array_equal(target.correct[rep].support_observations, base.support_observations)
                        and np.array_equal(target.correct[rep].support_actions, base.support_actions),
                        "support observations/actions differ across targets")
            support_sets.append((decode(base.support_observations), base.support_actions.astype(np.int64)))
        supports[seed] = support_sets
        # Pipeline check: my decoded observables + my case construction + the evaluator's
        # realised hidden state reproduce saved truth and saved support outcomes exactly.
        regen = loader.regenerate(endpoint)
        world = AcquisitionWorld()
        exact = {}
        for target in endpoint.targets:
            qc = regen.queries[target.slot]
            require(np.array_equal(qc.factors, query["factors"]) and np.array_equal(qc.signals, query["signals"]),
                    "decoded factors/signals differ from evaluator cases")
            require(np.array_equal(np.rint(qc.reserves / CAPACITY * GAUGE).astype(np.int64), query["levels"]),
                    "decoded gauge differs")
            pre = qc.realised["base_supplies"][..., ::-1] if target.law.mode == 1 else qc.realised["base_supplies"]
            mine = survival_probabilities(world, target.law, np.asarray(qc.reserves)[:, None, :],
                                          np.asarray(pre)[:, None], query["factors"], query["signals"])
            require(np.array_equal(mine.astype(np.uint8), target.correct[0].truth), "pipeline truth mismatch")
            for rep in range(len(target.correct)):
                sc = regen.supports[target.slot, rep]
                obs, actions = support_sets[rep]
                pre = sc.realised["base_supplies"][..., ::-1] if target.law.mode == 1 else sc.realised["base_supplies"]
                counts = pattern_counts(world, target.law, np.asarray(sc.cases.base.reserves)[:, None, :],
                                        np.asarray(pre)[:, None], obs["factors"], obs["signals"], actions)
                require(np.array_equal(counts.argmax(axis=1), target.correct[rep].support_outcomes.sum(axis=1)),
                        "pipeline support outcome mismatch")
            exact[target.slot] = np.asarray(qc.reserves)
        require(all(np.array_equal(exact[0], v) for v in exact.values()), "reserves differ across target panels")
        # Noise sampler check against the evaluator's realised base supplies (pre-mode).
        qc = regen.queries[0]
        pre = qc.realised["base_supplies"][..., ::-1] if endpoint.targets[0].law.mode == 1 else qc.realised["base_supplies"]
        season = np.where(np.arange(STEPS) < 6, 1.0, -1.0)
        means = MEAN + IMBALANCE * query["factors"][:, 0, None, None] * season[None, :, None] * np.array([1.0, -1.0])
        residual = (np.asarray(pre) - means)[np.asarray(pre) > 0]
        mine_r, mine_s = sample_hidden(np.random.default_rng(1), query["levels"], query["factors"][:, 0], 8)
        mine_res = (mine_s - means[:, None])[mine_s > 0]
        checks[seed] = dict(pipeline_truth_exact=True, pipeline_support_outcomes_exact=True,
                            decoded_observables_match_evaluator=True,
                            realised_noise_mean=float(residual.mean()), realised_noise_sd=float(residual.std()),
                            sampler_noise_mean=float(mine_res.mean()), sampler_noise_sd=float(mine_res.std()),
                            realised_zero_fraction=float((np.asarray(pre) == 0).mean()),
                            sampler_reserve_within_bin=bool((np.rint(mine_r / CAPACITY * GAUGE).astype(np.int64)
                                                             == query["levels"][:, None]).all()))
        checks[seed]["exact_reserves"] = exact[0]
        # fresh probe of the delay-introducing law shares this panel and truth
        stage = delay_stage(seed)
        fresh = loader.load_endpoint(seed=seed, phase=f"fresh_{stage}")
        require(fresh.targets[0].law == laws[stage + 1]
                and np.array_equal(fresh.targets[0].correct[0].truth, endpoint.targets[stage + 1].correct[0].truth)
                and np.array_equal(fresh.targets[0].correct[0].observations, first.observations),
                "fresh delay panel differs from interleaved panel")
        checks[seed]["fresh_delay_panel_identical"] = True
        checks[seed]["fresh_learner_delay_benefit"] = float(np.mean([
            brier(fresh.targets[0].flips[1][r].probabilities, fresh.targets[0].correct[r].truth, fresh.targets[0].flips[1][r].affected)
            - brier(fresh.targets[0].correct[r].probabilities, fresh.targets[0].correct[r].truth, fresh.targets[0].flips[1][r].affected)
            for r in range(2)]))
    print(f"loaded and checked {len(args.seeds)} seeds in {time.perf_counter() - started:.1f}s", flush=True)

    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for seed in args.seeds:
            laws = my_laws(seed)
            stage = delay_stage(seed)
            for rep in range(args.replicates):
                jobs.append(pool.submit(query_worker, seed, rep, laws, observed[seed], args.query_draws))
                jobs.append(pool.submit(support_worker, seed, rep, laws, supports[seed], args.support_draws))
                jobs.append(pool.submit(kl_worker, seed, rep, laws[stage + 1], laws[stage], args.kl_records,
                                        args.kl_draws))
            if seed in args.exact_seeds:
                jobs.append(pool.submit(query_worker, seed, 0, laws, observed[seed], args.query_draws, 32,
                                        checks[seed]["exact_reserves"]))
        results = []
        for index, future in enumerate(as_completed(jobs), 1):
            item = future.result()
            results.append(item)
            print(f"{index}/{len(jobs)} {item['kind']} seed={item['seed']} rep={item['replicate']} "
                  f"{item['seconds']:.0f}s elapsed={time.perf_counter() - started:.0f}s", flush=True)

    report = dict(label=LABEL, settings=vars(args) | dict(output=str(output.relative_to(ROOT).as_posix())),
                  definitions=__doc__, seeds={})
    for seed in args.seeds:
        laws = my_laws(seed)
        stage = delay_stage(seed)
        endpoint = endpoints[seed]
        queries = sorted([r for r in results if r["kind"] == "query" and r["seed"] == seed and not r["exact"]],
                         key=lambda r: r["replicate"])
        sups = sorted([r for r in results if r["kind"] == "support" and r["seed"] == seed], key=lambda r: r["replicate"])
        kls = sorted([r for r in results if r["kind"] == "kl" and r["seed"] == seed], key=lambda r: r["replicate"])
        per_rep = []
        for q, s in zip(queries, sups):
            rows, _ = score_seed(endpoint, laws, q["forecasts"], s["counts"], s["draws"])
            per_rep.append(rows)
        pooled_forecasts = {key: np.mean([q["forecasts"][key] for q in queries], axis=0) for key in queries[0]["forecasts"]}
        pooled_counts = np.sum([s["counts"] for s in sups], axis=0)
        pooled_draws = sum(s["draws"] for s in sups)
        pooled, posts = score_seed(endpoint, laws, pooled_forecasts, pooled_counts, pooled_draws)

        def se(getter):
            values = np.array([getter(rows) for rows in per_rep], dtype=np.float64)
            return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None

        slots = {}
        for slot, row in pooled.items():
            entry = {}
            for name in ("oracle_brier", "ideal_brier", "prior_brier", "learner_brier"):
                entry[name] = row[name]
                entry[name + "_se"] = se(lambda rows, n=name, k=slot: rows[k][n])
            for name in ("oracle", "ideal", "prior", "learner"):
                key = f"{name}_cue_benefit"
                entry[key] = row[key]
                entry[key + "_se"] = {c: se(lambda rows, k=slot, c=c, n=key: rows[k][n][c]) for c in row[key]}
            entry["law"] = row["law"]
            entry["posterior"] = row["posterior"]
            slots[str(slot)] = entry
        # posterior on the delay-introducing target's saved supports
        target_slot = stage + 1
        delay_active = [i for i, law in enumerate(laws) if 1 in law.active]
        posterior_rows = []
        for rep in range(2):
            w = posts[target_slot, rep]
            rep_values = [np.asarray(rows[target_slot]["posterior"][rep]) for rows in per_rep]
            posterior_rows.append(dict(
                replicate=rep, weights=w.tolist(), true_law=float(w[target_slot]),
                delay_active=float(w[delay_active].sum()), predecessor=float(w[target_slot - 1]),
                true_law_se=float(np.std([v[target_slot] for v in rep_values], ddof=1) / np.sqrt(len(rep_values))),
                delay_active_se=float(np.std([v[delay_active].sum() for v in rep_values], ddof=1) / np.sqrt(len(rep_values)))))
        # zero-count diagnostics for the pooled likelihoods of each target's true law
        zero_true = {}
        for target in endpoint.targets:
            for rep, correct in enumerate(target.correct):
                _, _, zeros = posterior(pooled_counts[rep], correct.support_outcomes, pooled_draws)
                zero_true[f"{target.slot}/{rep}"] = int(zeros[target.slot])
        # KL
        kl_rep = [0.5 * float(k["kl"].mean()) for k in kls]
        reverse_rep = [0.5 * float(k["reverse"].mean()) for k in kls]
        kl_all = 0.5 * float(np.concatenate([k["kl"] for k in kls]).mean())
        kl_half_all = 0.5 * float(np.concatenate([k["kl_half"] for k in kls]).mean())
        reverse_all = 0.5 * float(np.concatenate([k["reverse"] for k in kls]).mean())
        llr = np.concatenate([k["llr"] for k in kls])
        cell = np.concatenate([k["cell"] for k in kls])
        rng = np.random.default_rng([ROOT_SEED, seed, 99])
        totals = np.zeros(20000)
        for c in range(4):
            pool_c = llr[cell == c]
            totals += pool_c[rng.integers(0, len(pool_c), (20000, 4))].sum(axis=1)
        exact_run = [r for r in results if r["kind"] == "query" and r["seed"] == seed and r["exact"]]
        exact_rows = None
        if exact_run:
            f = exact_run[0]["forecasts"]
            exact_rows = {}
            for target in endpoint.targets:
                truth = target.correct[0].truth
                exact_rows[str(target.slot)] = dict(
                    oracle_brier=brier(f[target.slot, -1], truth),
                    oracle_cue_benefit={str(c): brier(f[target.slot, c], truth, affected_mask(observed[seed]["factors"], c))
                                        - brier(f[target.slot, -1], truth, affected_mask(observed[seed]["factors"], c))
                                        for c in target.law.active})
        report["seeds"][str(seed)] = dict(
            order=list(order(seed)), delay_stage=stage, laws=[[l.mode, list(l.active)] for l in laws],
            checks={k: v for k, v in checks[seed].items() if k != "exact_reserves"},
            slots=slots,
            delay_target_posterior=posterior_rows,
            zero_count_true_law_records=zero_true,
            kl=dict(delay_law=[laws[stage + 1].mode, list(laws[stage + 1].active)],
                    predecessor=[laws[stage].mode, list(laws[stage].active)],
                    per_record_nats=kl_all, per_record_nats_se=float(np.std(kl_rep, ddof=1) / np.sqrt(len(kl_rep))),
                    per_record_nats_by_replicate=kl_rep,
                    per_record_nats_half_draws=kl_half_all,
                    reverse_per_record_nats=reverse_all,
                    reverse_se=float(np.std(reverse_rep, ddof=1) / np.sqrt(len(reverse_rep))),
                    per_informative_record_nats=kl_all / 0.4,
                    records_for_20_to_1=float(np.log(20) / kl_all),
                    records_for_20_to_1_by_replicate=[float(np.log(20) / v) for v in kl_rep],
                    p_support32_lr_ge_20=float((totals >= np.log(20)).mean()),
                    mean_support32_log_lr=float(totals.mean()),
                    zero_support_pattern_events=int(sum(k["zero_support"] for k in kls)),
                    records_simulated=int(sum(len(k["kl"]) for k in kls)), draws=args.kl_draws),
            exact_reserve_sensitivity=exact_rows,
            fresh_learner_delay_benefit=checks[seed]["fresh_learner_delay_benefit"])
        print(json.dumps(dict(seed=seed, oracle=[slots[str(s)]["oracle_brier"] for s in range(5)],
                              ideal=[slots[str(s)]["ideal_brier"] for s in range(5)],
                              delay=[(s, slots[str(s)]["oracle_cue_benefit"].get("1"),
                                      slots[str(s)]["ideal_cue_benefit"].get("1")) for s in range(5)],
                              posterior=posterior_rows, kl=kl_all), indent=None), flush=True)
    report["seconds"] = time.perf_counter() - started

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    if args.no_write:
        print("--no-write: nothing written")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    print(f"wrote {output.relative_to(ROOT).as_posix()} in {report['seconds']:.0f}s")


if __name__ == "__main__":
    main()
