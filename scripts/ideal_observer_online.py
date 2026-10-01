"""POST HOC / EXPLORATORY / NOT DECISION-BEARING online ideal-observer reanalysis.

Reads the saved prequential (online) arrays of the locked representation-learning
qualification fits (runs/representation_learning_qualification) and compares the
ordinary recurrent reference with evaluator-side ideal-observer references.

Nothing here executes, unpickles or updates a learner: no ``*.pt`` file is
opened and no study runner is called. Training cases are regenerated with the
project's own seeds (``study.training_seed`` and ``AcquisitionWorld.dataset``);
every regenerated observation array must byte-match the saved one, and the
regenerated actions, performed outcomes, counterfactual truth, masks and packet
fingerprints must match the saved arrays/records.

Ideal observers are ANALYSIS REFERENCES, not learners. They use evaluator-only
knowledge (the physical law set and simulator) and are never fed back into any
locked decision:

* law-known observer: knows the batch's physical law; decodes reserves (gauge
  bins), source direction, lossy/delayed glyphs and the three pulse-order cues
  from the query pixels, and integrates the hidden supply noise and gauge
  quantisation by Monte Carlo (common random numbers across laws/actions).
* support-posterior observer: knows the five interleaved reference laws, starts
  from a uniform prior and conditions ONLY on the same 32-record support the
  learner received (previous batch: observations, performed action, 3-horizon
  outcome), then forecasts with the posterior mixture.
* no-context observer: uniform mixture over the five reference laws.

Law identities, clean physical outcomes and counterfactual outcomes remain
evaluator-only. All outputs are labelled POST HOC / EXPLORATORY /
NOT DECISION-BEARING and do not alter any locked decision.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from acp_cl.acquisition.world import (AcquisitionWorld, LawCases, mask_affected,  # noqa: E402
                                      mask_valid, order, stage_law)
from acp_cl.persistence.study import trial_seed  # noqa: E402
from acp_cl.persistence.world import Cases  # noqa: E402
from acp_cl.representation_learning.design import (law_from_record, phase_laws,  # noqa: E402
                                                   qualification_phases, reference_laws)
from acp_cl.representation_learning.study import training_seed  # noqa: E402


RUN = ROOT / 'runs' / 'representation_learning_qualification'
DEFAULT_OUTPUT = ROOT / 'reports' / 'information_audit' / 'online'
LABEL = 'POST HOC / EXPLORATORY / NOT DECISION-BEARING'
SALT = 20260928
CHUNK = 16            # batches per Monte Carlo chunk; divides the 32-batch block
LEVELS = 240          # gauge quantisation used by TransferWorld.render
PHASES = ('fresh_1', 'fresh_2', 'fresh_3', 'interleaved')
SLOT_NAMES = ('baseA', 'baseB', 'stage1', 'stage2', 'stage3')
CUE_NAMES = ('efficiency', 'delay', 'supply_timing')


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def config():
    manifest = read_json(RUN / 'manifest.json')
    return manifest['config']


# --------------------------------------------------------------------------- #
# Observation decoding (pixels only) and Monte Carlo ideal observer            #
# --------------------------------------------------------------------------- #

def decode(world, observations):
    """Recover every observable case variable from the query pixels alone."""
    obs = np.asarray(observations)
    require(obs.dtype == np.uint8 and obs.ndim == 4 and obs.shape[1:] == (4, 12, 16),
            'uint8 [N,4,12,16] observations required')
    levels = []
    for start in (1, 9):
        patch = obs[:, :, 0:4, start:start + 6]
        level = patch[:, 0, 0, 0]
        require((patch == level[:, None, None, None]).all(), 'non-uniform reserve gauge')
        levels.append(level.astype(np.int64))
    glyphs = []
    for field, column in enumerate((1, 6, 11)):
        patch = obs[:, :, 5:8, column:column + 3]
        match = [(patch == world.glyphs[field, value][None, None]).all(axis=(1, 2, 3))
                 for value in (0, 1)]
        require((match[0] ^ match[1]).all(), 'undecodable glyph')
        glyphs.append(match[1].astype(np.int64))
    require(not glyphs[2].any(), 'gated cases are outside this study')
    weather = obs[:, :, 10, :]
    source = np.zeros(len(obs), dtype=np.int64)
    for value in (1, -1):
        hit = np.all([weather[:, frame, 8 - (3 - frame) * value] == 255 for frame in range(4)], axis=0)
        source[hit] = value
    require((source != 0).all(), 'undecodable weather motion')
    signals = []
    for column in (1, 6, 11):
        first = obs[:, 0, 8, column:column + 2] == 255
        second = obs[:, 2, 8, column:column + 2] == 255
        require((first.sum(1) == 1).all() and (second.sum(1) == 1).all()
                and np.array_equal(first[:, 1], second[:, 0]), 'undecodable pulse order')
        signals.append(first[:, 1].astype(np.uint8))
    return (np.stack(levels, 1), np.stack([source, glyphs[0], glyphs[1]], 1),
            np.stack(signals, 1))


def draw_states(world, levels, factors, draws, rng):
    """Posterior draws of the hidden state given decoded pixels (pre-mode supplies)."""
    levels = np.asarray(levels, dtype=np.float64)
    low = np.maximum((levels - .5) * world.capacity / LEVELS, 1.)
    high = np.minimum((levels + .5) * world.capacity / LEVELS, 13.)
    require((high > low).all(), 'empty reserve bin')
    count = len(levels)
    reserves = low[:, None, :] + rng.random((count, draws, 2)) * (high - low)[:, None, :]
    season = np.where(np.arange(12) < 6, 1., -1.)
    means = (world.mean_supply + world.imbalance * factors[:, 0, None, None]
             * season[None, :, None] * np.array([1., -1.]))
    supplies = np.clip(means[:, None] + rng.normal(0., world.supply_noise, (count, draws, 12, 2)),
                       0., None)
    return reserves, supplies


def outcome_counts(world, reserves, supplies, factors, signals, law):
    """Counts of survival patterns (0..3 horizons survived) per case and action."""
    count, draws = reserves.shape[:2]
    total = count * draws
    flat = supplies.reshape(total, 12, 2)
    if law.mode == 1:  # ConditionalWorld swaps the supply columns in mode 1
        flat = flat[:, :, ::-1]
    repeated = np.repeat(np.asarray(factors, dtype=np.int64), draws, axis=0)
    dummy = np.zeros(total, dtype=np.uint8)
    base = Cases(dummy, reserves.reshape(total, 2).copy(), np.ascontiguousarray(flat),
                 np.where(repeated[:, 1] == 1, .35, 1.), np.where(repeated[:, 2] == 1, 2, 0))
    cases = LawCases(dummy, base, repeated, np.repeat(np.asarray(signals, dtype=np.uint8), draws, 0), law)
    result = np.empty((count, 5, 4), dtype=np.uint16)
    for action in range(5):
        survival = world.simulate(cases, np.full(total, action, dtype=np.int64)).survival
        require(np.all(np.diff(survival.astype(np.int16), axis=1) <= 0), 'non-monotone survival')
        survived = survival.astype(np.int64).sum(1).reshape(count, draws)
        for pattern in range(4):
            result[:, action, pattern] = (survived == pattern).sum(1)
    return result


def forecast(counts, draws):
    counts = np.asarray(counts, dtype=np.float64)
    return np.stack([counts[..., h + 1:].sum(-1) for h in range(3)], -1) / draws


def pre_mode_supplies(cases):
    supplies = cases.base.supplies.copy()
    if cases.law.mode == 1:
        supplies = supplies[:, :, ::-1]
    return supplies


def exact_replay(world, cases, truth, levels, factors, signals):
    """The Monte Carlo pipeline with the TRUE latent state reproduces the truth exactly."""
    reserves = cases.base.reserves[:, None, :]
    supplies = pre_mode_supplies(cases)[:, None]
    counts = outcome_counts(world, reserves, supplies, factors, signals, cases.law)
    pattern = truth.astype(np.int64).sum(-1)
    require(np.array_equal(np.take_along_axis(counts, pattern[..., None], -1)[..., 0],
                           np.ones_like(pattern)), 'ideal-observer physics differs from truth')
    require(np.array_equal(np.rint(cases.base.reserves / world.capacity * LEVELS).astype(np.int64),
                           levels), 'gauge decoding mismatch')


# --------------------------------------------------------------------------- #
# Stage 1: regenerate, verify, and compute ideal-observer counts (workers)     #
# --------------------------------------------------------------------------- #

def phase_task(seed, phase_index, draws):
    started = time.perf_counter()
    cfg = config()
    world = AcquisitionWorld()
    phase = qualification_phases(seed, cfg)[phase_index]
    name = phase['name']
    folder = RUN / 'jobs' / f'outcome_{seed}' / name
    record = read_json(folder / 'result.json')
    require(record['seed'] == seed and record['phase'] == name and record['arm'] == 'outcome',
            'record identity')
    require(json.loads(json.dumps(phase)) == record['phase_spec'], 'phase specification differs')
    require(sha(folder / 'training.npz') == record['training_file']['sha256']
            == record['artifact_hashes']['training.npz'], 'training.npz hash')
    require(sha(folder / 'evaluations.npz') == record['artifact_hashes']['evaluations.npz'],
            'evaluations.npz hash')
    with np.load(folder / 'training.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    size = cfg['batch_size']
    laws = phase_laws(phase, size)
    distinct = list(dict.fromkeys(laws))
    require(arrays['law_index'].tolist() == [distinct.index(law) for law in laws], 'law index')
    if name == 'interleaved':
        candidates = reference_laws(seed)
        require(distinct == candidates, 'interleaved law order')
    else:
        stage = int(name.split('_')[1])
        candidates = [stage_law(seed, stage), stage_law(seed, stage - 1)]
        require(distinct == candidates[:1], 'fresh law')
    batches = len(laws)
    require(arrays['observations'].shape == (batches, size, 4, 12, 16), 'observation shape')
    require(record['batches_done'] == batches and len(record['packets']) == batches, 'packet count')
    levels = np.empty((batches, size, 2), dtype=np.int64)
    factors = np.empty((batches, size, 3), dtype=np.int64)
    signals = np.empty((batches, size, 3), dtype=np.uint8)
    previous = None
    for index, law in enumerate(laws):
        data_seed = training_seed(seed, name, index)
        cases = world.dataset(law, size, data_seed)
        data = world.experience(law, size, data_seed)
        saved = arrays['observations'][index]
        require(saved.dtype == cases.observations.dtype == np.uint8
                and saved.shape == cases.observations.shape
                and saved.tobytes() == cases.observations.tobytes(), 'observation bytes differ')
        require(np.array_equal(data.observations, saved)
                and np.array_equal(data.actions, arrays['actions'][index])
                and np.array_equal(data.survival, arrays['outcomes'][index]), 'performed data differ')
        truth = world.counterfactuals(cases)
        require(truth.dtype == arrays['truth'].dtype and np.array_equal(truth, arrays['truth'][index]),
                'counterfactual truth differs')
        require(np.array_equal(truth[np.arange(size), data.actions], data.survival), 'performed truth')
        require(np.array_equal(mask_affected(cases, phase['cue']), arrays['affected'][index])
                and np.array_equal(mask_valid(cases, phase['branch']), arrays['valid'][index]), 'masks')
        packet = record['packets'][index]
        require(packet['index'] == index and packet['id'] == index
                and packet['query_sha256'] == data.fingerprint()
                and packet['support_sha256'] == previous, 'packet fingerprint')
        previous = data.fingerprint()
        lv, fac, sig = decode(world, saved)
        require(np.array_equal(fac, cases.factors) and np.array_equal(sig, cases.signals),
                'pixel decoding differs from evaluator factors')
        exact_replay(world, cases, truth, lv, fac, sig)
        levels[index], factors[index], signals[index] = lv, fac, sig
    count = batches * size
    levels, factors, signals = (x.reshape(count, *x.shape[2:]) for x in (levels, factors, signals))
    counts = np.empty((count, len(candidates), 5, 4), dtype=np.uint16)
    flip_rows = np.zeros(count, dtype=bool)
    flip_counts = np.zeros((count, 5, 4), dtype=np.uint16)
    for start in range(0, batches, CHUNK):
        require(len(set(laws[start:start + CHUNK])) == 1, 'chunk spans laws')
        law = laws[start]
        rows = np.arange(start * size, min(start + CHUNK, batches) * size)
        rng = np.random.default_rng([SALT, seed, phase_index, start])
        reserves, supplies = draw_states(world, levels[rows], factors[rows], draws, rng)
        for position, candidate in enumerate(candidates):
            counts[rows, position] = outcome_counts(world, reserves, supplies, factors[rows],
                                                    signals[rows], candidate)
        if 1 in law.active:
            sub = np.flatnonzero(factors[rows, 2] == 1)
            flipped = signals[rows][sub].copy()
            flipped[:, 1] ^= 1
            flip_counts[rows[sub]] = outcome_counts(world, reserves[sub], supplies[sub],
                                                    factors[rows][sub], flipped, law)
            flip_rows[rows[sub]] = True
    memory = np.full((batches, cfg['memory_packets']), -1, dtype=np.int32)
    for index, packet in enumerate(record['packets']):
        memory[index, :len(packet['memory_ids_before'])] = packet['memory_ids_before']
    replay = np.asarray([packet['replay_ids'] for packet in record['packets']], dtype=np.int32)
    endpoint = endpoint_task(world, cfg, seed, phase_index, record, folder, candidates, draws)
    return dict(
        seed=seed, phase=name, cue=phase['cue'], draws=draws,
        candidates=[dict(mode=c.mode, active=list(c.active)) for c in candidates],
        law_index=arrays['law_index'].astype(np.int16),
        probabilities=arrays['probabilities'].reshape(count, 5, 3),
        truth=arrays['truth'].reshape(count, 5, 3), actions=arrays['actions'].reshape(count),
        outcomes=arrays['outcomes'].reshape(count, 3), levels=levels.astype(np.int16),
        factors=factors.astype(np.int8), signals=signals, counts=counts,
        flip_rows=flip_rows, flip_counts=flip_counts, replay=replay, memory=memory,
        curve=[dict(arrivals=c['arrivals'], law=c['law'], brier=c['metrics']['brier'])
               for c in record['curve']],
        endpoint=endpoint, batches_verified=batches, seconds=time.perf_counter() - started)


def endpoint_task(world, cfg, seed, phase_index, record, folder, candidates, draws):
    """Ideal-observer values on the saved endpoint probe panels (same cases/supports)."""
    evaluations = read_json(folder / 'evaluations.json')
    records = evaluations['records']
    panel_seed = trial_seed(seed, 'acquisition_query', 0)
    reference = world.dataset(candidates[0], cfg['eval_size'], panel_seed)
    levels, factors, signals = decode(world, reference.observations)
    require(np.array_equal(factors, reference.factors) and np.array_equal(signals, reference.signals),
            'panel decoding')
    rng = np.random.default_rng([SALT, seed, phase_index, 10 ** 6])
    reserves, supplies = draw_states(world, levels, factors, draws, rng)
    panel = np.stack([outcome_counts(world, reserves, supplies, factors, signals, c)
                      for c in candidates], 1)
    result = []
    with np.load(folder / 'evaluations.npz', allow_pickle=False) as archive:
        for probe in record['end_probes']:
            law = law_from_record(probe['law'])
            target = candidates.index(law)
            cases = world.dataset(law, cfg['eval_size'], panel_seed)
            require(np.array_equal(cases.observations, reference.observations), 'panel observations')
            truth = world.counterfactuals(cases)
            exact_replay(world, cases, truth, levels, factors, signals)
            correct, supports = [], []
            for trace in probe['correct']:
                row = records[trace]
                require(row['law'] == dict(mode=law.mode, active=list(law.active), revised=False,
                                           noise=0.) and not row['flipped'], 'probe record')
                require(np.array_equal(archive[f'e{trace}_observations'], cases.observations)
                        and np.array_equal(archive[f'e{trace}_truth'], truth), 'probe arrays')
                correct.append(archive[f'e{trace}_probabilities'])
                support_obs = archive[f'e{trace}_support_observations']
                s_levels, s_factors, s_signals = decode(world, support_obs)
                s_rng = np.random.default_rng([SALT, seed, phase_index, 10 ** 6 + 1 + trace])
                s_res, s_sup = draw_states(world, s_levels, s_factors, draws, s_rng)
                supports.append(dict(
                    counts=np.stack([outcome_counts(world, s_res, s_sup, s_factors, s_signals, c)
                                     for c in candidates], 1),
                    actions=archive[f'e{trace}_support_actions'].astype(np.int64),
                    outcomes=archive[f'e{trace}_support_outcomes'].astype(np.int64)))
            cues = {}
            for cue, traces in probe['flips'].items():
                cue = int(cue)
                flipped = signals.copy()
                flipped[:, cue] ^= 1
                flip = outcome_counts(world, reserves, supplies, factors, flipped, law)
                model = [archive[f'e{t}_probabilities'] for t in traces]
                for t in traces:
                    require(records[t]['flipped'] and records[t]['cue'] == cue, 'flip record')
                cues[cue] = dict(flip_counts=flip, model_flipped=model,
                                 affected=mask_affected(cases, cue))
            result.append(dict(law=dict(mode=law.mode, active=list(law.active)), target=target,
                               truth=truth, model=correct, supports=supports, cues=cues,
                               model_brier=[records[t]['metrics']['brier'] for t in probe['correct']]))
    return dict(panel_counts=panel, results=result)


def collect(seeds, draws, workers, cache):
    tasks = [(seed, index) for index in (3, 0, 1, 2) for seed in seeds]
    results = {}
    pending = []
    for seed, index in tasks:
        path = None if cache is None else Path(cache) / f'{seed}_{PHASES[index]}_{draws}.npy'
        if path is not None and path.exists():
            results[(seed, PHASES[index])] = np.load(path, allow_pickle=True).item()
        else:
            pending.append((seed, index, path))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(phase_task, seed, index, draws): (seed, index, path)
                       for seed, index, path in pending}
            for future in as_completed(futures):
                seed, index, path = futures[future]
                value = future.result()
                print(json.dumps(dict(seed=seed, phase=PHASES[index],
                                      seconds=round(value['seconds'], 1))), flush=True)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, np.asarray(value, dtype=object), allow_pickle=True)
                results[(seed, PHASES[index])] = value
    return results


# --------------------------------------------------------------------------- #
# Stage 2: analysis helpers                                                    #
# --------------------------------------------------------------------------- #

def softmax(values):
    values = values - values.max(-1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(-1, keepdims=True)


def case_brier(p, y):
    return np.square(p - y).mean(axis=(-2, -1))


def seed_meta(seed):
    sequence = order(seed)
    delay_stage = sequence.index(1) + 1
    return dict(seed=seed, parity=seed % 2, baseA_mode=seed % 2, stage_mode=1 - seed % 2,
                order=[CUE_NAMES[c] for c in sequence], delay_stage=delay_stage,
                delay_slot=delay_stage + 1, predecessor_slot=delay_stage,
                efficiency_slot=sequence.index(0) + 2, supply_slot=sequence.index(2) + 2)


def covariates(factors, signals, levels, cue):
    """Pre-treatment case covariates (all decodable from pixels) for adjustment."""
    source, lossy, delayed = (factors[:, i].astype(np.float64) for i in range(3))
    a, b = levels[:, 0] / LEVELS, levels[:, 1] / LEVELS
    z = 1. - 2. * signals.astype(np.float64)
    columns = [source, lossy, delayed, a, b, source * a, source * b, lossy * a, lossy * b,
               delayed * a, delayed * b, source * lossy, source * delayed, lossy * delayed]
    for other in range(3):
        if other != cue:
            columns += [z[:, other], z[:, other] * lossy, z[:, other] * delayed, z[:, other] * source]
    matrix = np.column_stack(columns)
    matrix = matrix[:, matrix.std(0) > 1e-12]
    chosen = []
    for j in range(matrix.shape[1]):
        trial = np.column_stack([np.ones(len(matrix))] + [matrix[:, k] for k in chosen + [j]])
        if np.linalg.matrix_rank(trial) == trial.shape[1]:
            chosen.append(j)
    return matrix[:, chosen]


def contrast(values, factors, signals, levels, rows, cue, clusters):
    """Aligned-minus-misaligned forecast/outcome difference for transfer requests.

    For each case, D = mean over sizes (2,4) of [A->B request] - [B->A request].
    A->B (direction 0) is aligned with / favoured by the cue when its signal is 0,
    so the coefficient on z = 1-2*signal is the aligned-minus-misaligned effect.
    Signals are randomised, so the estimate is unbiased; pre-treatment covariates
    only reduce variance. Standard errors are cluster-robust over 32-case batches.
    """
    rows = np.flatnonzero(rows)
    if len(rows) < 64:
        return None
    v = values[rows].astype(np.float64)
    difference = ((v[:, 2] - v[:, 1]) + (v[:, 4] - v[:, 3])) / 2.
    target = np.column_stack([difference, difference.mean(1)])
    z = 1. - 2. * signals[rows, cue].astype(np.float64)
    design = np.column_stack([np.ones(len(rows)), covariates(factors[rows], signals[rows],
                                                             levels[rows], cue), z])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    inverse = np.linalg.inv(design.T @ design)
    _, groups = np.unique(clusters[rows], return_inverse=True)
    se = []
    for j in range(target.shape[1]):
        sums = np.zeros((groups.max() + 1, design.shape[1]))
        np.add.at(sums, groups, design * residual[:, j:j + 1])
        se.append(float(np.sqrt((inverse @ (sums.T @ sums) @ inverse)[-1, -1])))
    return dict(effect=beta[-1].tolist(), se=se, raw=(target * z[:, None]).mean(0).tolist(),
                n=int(len(rows)))


def summarise(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if not len(values):
        return None
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return dict(mean=float(values.mean()), se=se, min=float(values.min()), max=float(values.max()),
                n=int(len(values)), values=values.tolist())


# --------------------------------------------------------------------------- #
# Stage 2: per-seed derived arrays                                             #
# --------------------------------------------------------------------------- #

def running_marginal(laws, actions, outcomes, size, count):
    """Law-identity cumulative performed-action marginal (evaluator reference)."""
    batches = len(laws)
    success = np.zeros((count, 5, 3))
    seen = np.zeros((count, 5))
    result = np.empty((batches, 5, 3))
    actions = actions.reshape(batches, size)
    outcomes = outcomes.reshape(batches, size, 3)
    for b, law in enumerate(laws):
        result[b] = (success[law] + .5) / (seen[law][:, None] + 1.)
        np.add.at(seen[law], actions[b], 1)
        np.add.at(success[law], actions[b], outcomes[b])
    return result


def interleaved_frame(r, draws, size):
    law = r['law_index'].astype(np.int64)
    batches, count = len(law), len(law) * size
    rows = np.arange(count)
    case_law = np.repeat(law, size)
    p = r['probabilities'].astype(np.float64)
    y = r['truth'].astype(np.float64)
    P = forecast(r['counts'], draws)
    laws = P.shape[1]
    variance = P * (1. - P) / (draws - 1)
    io = P[rows, case_law]
    frame = dict(law=law, batch=np.arange(batches), block=np.arange(batches) // size,
                 cycle=np.arange(batches) // size // 5, position=np.arange(batches) % size)
    per_case = dict(model=case_brier(p, y),
                    io=case_brier(io, y) - variance[rows, case_law].mean((1, 2)))
    q = (r['counts'].astype(np.float64) + .5) / (draws + 2)
    pattern = r['outcomes'].astype(np.int64).sum(1)
    actions = r['actions'].astype(np.int64)
    like = np.log(q[rows[:, None], np.arange(laws)[None], actions[:, None], pattern[:, None]])
    loglik = like.reshape(batches, size, laws).sum(1)
    posterior = np.full((batches, laws), 1. / laws)
    posterior[1:] = softmax(loglik[:-1])
    weights = np.repeat(posterior, size, 0)
    p_post = np.einsum('nl,nlah->nah', weights, P)
    per_case['post'] = case_brier(p_post, y) - np.einsum('nl,nlah->n', weights, variance) / 15
    per_case['mix'] = case_brier(P.mean(1), y) - variance.mean(1).mean((1, 2))
    marginal = running_marginal(law, r['actions'], r['outcomes'], size, laws)
    per_case['marginal'] = case_brier(np.repeat(marginal, size, 0), y)
    for l in range(laws):
        per_case[f'as_law{l}'] = case_brier(P[:, l], y) - variance[:, l].mean((1, 2))
    a, b = P[:, 0], P[:, 1]
    difference = a - b
    per_case['lam_den'] = np.square(difference).sum((1, 2))
    per_case['lam_model'] = ((p - b) * difference).sum((1, 2))
    per_case['lam_post'] = ((p_post - b) * difference).sum((1, 2))
    per_case['dist_A'] = np.square(p - a).mean((1, 2))
    per_case['dist_B'] = np.square(p - b).mean((1, 2))
    per_case['dist_true'] = np.square(p - io).mean((1, 2))
    for key, value in per_case.items():
        frame[key] = value.reshape(batches, size).mean(1)
    frame['posterior'] = posterior
    frame['post_true'] = posterior[np.arange(batches), law]
    modes = np.asarray([c['mode'] for c in r['candidates']])
    frame['post_mode'] = (posterior * (modes[None] == modes[law][:, None])).sum(1)
    frame['support_loglik'] = loglik
    flip = forecast(r['flip_counts'], draws)
    flip_rows = r['flip_rows']
    cases = dict(p=p, y=y, io=io, P=P, post=p_post, mix=P.mean(1), flip=flip, flip_rows=flip_rows,
                 flip_var=(flip * (1. - flip) / (draws - 1)).mean((1, 2)),
                 io_var=variance[rows, case_law].mean((1, 2)),
                 factors=r['factors'].astype(np.int64), signals=r['signals'],
                 levels=r['levels'].astype(np.int64), clusters=np.repeat(np.arange(batches), size),
                 case_law=case_law, position=np.repeat(frame['position'], size),
                 cycle=np.repeat(frame['cycle'], size))
    return frame, cases


def fresh_frame(r, draws, size):
    batches = len(r['law_index'])
    count = batches * size
    p = r['probabilities'].astype(np.float64)
    y = r['truth'].astype(np.float64)
    P = forecast(r['counts'], draws)
    variance = P * (1. - P) / (draws - 1)
    marginal = running_marginal(np.zeros(batches, dtype=np.int64), r['actions'], r['outcomes'], size, 1)
    per_case = dict(model=case_brier(p, y), io=case_brier(P[:, 0], y) - variance[:, 0].mean((1, 2)),
                    io_predecessor=case_brier(P[:, 1], y) - variance[:, 1].mean((1, 2)),
                    marginal=case_brier(np.repeat(marginal, size, 0), y))
    frame = {key: value.reshape(batches, size).mean(1) for key, value in per_case.items()}
    frame['chunk'] = np.arange(batches) // size
    flip = forecast(r['flip_counts'], draws)
    cases = dict(p=p, y=y, io=P[:, 0], io_predecessor=P[:, 1], flip=flip, flip_rows=r['flip_rows'],
                 flip_var=(flip * (1. - flip) / (draws - 1)).mean((1, 2)),
                 io_var=variance[:, 0].mean((1, 2)), factors=r['factors'].astype(np.int64),
                 signals=r['signals'], levels=r['levels'].astype(np.int64),
                 clusters=np.repeat(np.arange(batches), size),
                 chunk=np.repeat(frame['chunk'], size))
    return frame, cases


def ideal_flip_benefit(cases, rows):
    rows = rows & cases['flip_rows']
    if not rows.any():
        return None
    y, io, flip = cases['y'][rows], cases['io'][rows], cases['flip'][rows]
    correct = case_brier(io, y) - cases['io_var'][rows]
    flipped = case_brier(flip, y) - cases['flip_var'][rows]
    blind = case_brier((io + flip) / 2., y) - (cases['io_var'][rows] + cases['flip_var'][rows]) / 4.
    return dict(flip=float((flipped - correct).mean()), blind=float((blind - correct).mean()),
                n=int(rows.sum()))


def endpoint_summary(value, draws, interleaved):
    panel = forecast(value['panel_counts'], draws)
    variance = panel * (1. - panel) / (draws - 1)
    out = []
    for item in value['results']:
        truth = item['truth'].astype(np.float64)
        target = item['target']
        io = panel[:, target]
        io_brier = case_brier(io, truth) - variance[:, target].mean((1, 2))
        model = np.mean([case_brier(p.astype(np.float64), truth).mean() for p in item['model']])
        require(np.isclose(model, np.mean(item['model_brier']), atol=1e-12), 'endpoint Brier recomputation')
        row = dict(law=item['law'], target=target, model_brier=float(model),
                   io_brier=float(io_brier.mean()),
                   as_law={str(l): float((case_brier(panel[:, l], truth)
                                          - variance[:, l].mean((1, 2))).mean())
                           for l in range(panel.shape[1])}, cues={})
        if interleaved and target in (0, 1):
            a, b = panel[:, 0], panel[:, 1]
            row['lambda_modeA'] = float(np.mean([((p.astype(np.float64) - b) * (a - b)).sum()
                                                 / np.square(a - b).sum() for p in item['model']]))
        if interleaved:
            posts, briers = [], []
            for support in item['supports']:
                q = (support['counts'].astype(np.float64) + .5) / (draws + 2)
                pattern = support['outcomes'].sum(1)
                index = np.arange(len(pattern))
                loglik = np.log(q[index[:, None], np.arange(q.shape[1])[None],
                                  support['actions'][:, None], pattern[:, None]]).sum(0)
                weights = softmax(loglik)
                posts.append(float(weights[target]))
                mixed = np.einsum('l,nlah->nah', weights, panel)
                briers.append(float((case_brier(mixed, truth)
                                     - np.einsum('l,nlah->n', weights, variance) / 15).mean()))
            row['post_true'] = float(np.mean(posts))
            row['io_post_brier'] = float(np.mean(briers))
        for cue, data in item['cues'].items():
            flip = forecast(data['flip_counts'], draws)
            affected = data['affected']
            flip_var = (flip * (1. - flip) / (draws - 1)).mean((1, 2))
            correct = io_brier[affected].mean()
            flipped = (case_brier(flip, truth) - flip_var)[affected].mean()
            blind = (case_brier((io + flip) / 2., truth)
                     - (variance[:, target].mean((1, 2)) + flip_var) / 4.)[affected].mean()
            model_benefit = np.mean([
                case_brier(f.astype(np.float64), truth)[affected].mean()
                - case_brier(c.astype(np.float64), truth)[affected].mean()
                for f, c in zip(data['model_flipped'], item['model'])])
            row['cues'][str(cue)] = dict(ideal_flip=float(flipped - correct),
                                         ideal_blind=float(blind - correct),
                                         model_flip=float(model_benefit))
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Stage 2: the four questions                                                  #
# --------------------------------------------------------------------------- #

REFERENCES = ('model', 'io', 'post', 'mix', 'marginal')


def mean_where(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def question_1(inter, fresh, seeds):
    rows = []
    for seed in seeds:
        frame = inter[seed][0]
        for slot in range(5):
            for cycle in range(8):
                sel = (frame['law'] == slot) & (frame['cycle'] == cycle)
                row = dict(seed=seed, slot=slot, law=SLOT_NAMES[slot], cycle=cycle + 1)
                for key in REFERENCES:
                    row[key] = mean_where(frame[key], sel)
                row['model_pos2plus'] = mean_where(frame['model'], sel & (frame['position'] > 0))
                row['io_as_baseB'] = mean_where(frame['as_law1'], sel)
                rows.append(row)
    table = {}
    for slot in range(5):
        for cycle in range(1, 9):
            cell = [r for r in rows if r['slot'] == slot and r['cycle'] == cycle]
            table[f'{slot}_{cycle}'] = {key: summarise([r[key] for r in cell])
                                        for key in REFERENCES + ('model_pos2plus', 'io_as_baseB')}
    per_seed = {}
    for seed in seeds:
        mine = [r for r in rows if r['seed'] == seed]
        get = {(r['slot'], r['cycle']): r for r in mine}
        others8 = np.mean([get[(s, 8)]['model'] for s in range(1, 5)])
        per_seed[seed] = dict(
            baseA_cycle1=get[(0, 1)]['model'], baseA_cycle8=get[(0, 8)]['model'],
            baseA_best_cycle=int(np.argmin([get[(0, c)]['model'] for c in range(1, 9)]) + 1),
            baseA_best=float(min(get[(0, c)]['model'] for c in range(1, 9))),
            others_cycle1=float(np.mean([get[(s, 1)]['model'] for s in range(1, 5)])),
            others_cycle8=float(others8),
            gap_cycle8=float(get[(0, 8)]['model'] - others8),
            gap_cycle8_pos2plus=float(get[(0, 8)]['model_pos2plus']
                                      - np.mean([get[(s, 8)]['model_pos2plus'] for s in range(1, 5)])),
            baseA_excess_io_cycle8=float(get[(0, 8)]['model'] - get[(0, 8)]['io']),
            others_excess_io_cycle8=float(np.mean([get[(s, 8)]['model'] - get[(s, 8)]['io']
                                                   for s in range(1, 5)])),
            baseA_minus_marginal_cycle8=float(get[(0, 8)]['model'] - get[(0, 8)]['marginal']),
            baseA_as_baseB_io_cycle8=float(get[(0, 8)]['io_as_baseB']),
            improvement={SLOT_NAMES[s]: float(get[(s, 1)]['model'] - get[(s, 8)]['model'])
                         for s in range(5)},
            improvement_late={SLOT_NAMES[s]: float(np.mean([get[(s, c)]['model'] for c in (2, 3)])
                                                   - np.mean([get[(s, c)]['model'] for c in (7, 8)]))
                              for s in range(5)})
    # Cross-law ideal-observer matrix: forecast with law L on cases of law M.
    cross = {}
    for true in range(5):
        for used in range(5):
            cross[f'{true}_{used}'] = summarise([
                float(inter[seed][0][f'as_law{used}'][inter[seed][0]['law'] == true].mean())
                for seed in seeds])
    fresh_rows = []
    for seed in seeds:
        for stage in (1, 2, 3):
            frame = fresh[(seed, stage)][0]
            for chunk in range(8):
                sel = frame['chunk'] == chunk
                fresh_rows.append(dict(seed=seed, stage=stage, chunk=chunk + 1,
                                       model=mean_where(frame['model'], sel),
                                       io=mean_where(frame['io'], sel),
                                       io_predecessor=mean_where(frame['io_predecessor'], sel),
                                       marginal=mean_where(frame['marginal'], sel)))
    return dict(rows=rows, table=table, per_seed=per_seed, cross=cross, fresh_rows=fresh_rows)


def profile_stats(profile):
    profile = np.asarray(profile, dtype=np.float64)
    plateau = float(profile[16:].mean())
    excess = profile - plateau
    smooth = np.convolve(excess, np.ones(3) / 3., mode='valid')
    half = None
    if excess[0] > 0:
        below = np.flatnonzero(smooth <= .5 * excess[0])
        half = int(below[0] + 1) if len(below) else None
    slope = float(np.polyfit(np.arange(2, 33), profile[1:], 1)[0])
    return dict(position1=float(profile[0]), position2=float(profile[1]),
                plateau_17_32=plateau, switch_cost=float(excess[0]),
                drop_1_to_2=float(profile[0] - profile[1]),
                excess_after_2=float(profile[1] - plateau),
                share_resolved_at_2=(float(1. - excess[1] / excess[0]) if excess[0] > 0 else None),
                half_life_position=half, slope_2_32_per_batch=slope)


def question_2(inter, seeds, results_by_seed):
    profiles, stats, lam = {}, {}, {}
    for cycles_name, low in (('cycles_2_8', 1), ('cycles_5_8', 4)):
        for slot in range(5):
            for key in ('model', 'post', 'io'):
                per_seed = []
                for seed in seeds:
                    frame = inter[seed][0]
                    base = (frame['law'] == slot) & (frame['cycle'] >= low)
                    per_seed.append([float(frame[key][base & (frame['position'] == pos)].mean())
                                     for pos in range(32)])
                per_seed = np.asarray(per_seed)
                profiles[f'{cycles_name}/{SLOT_NAMES[slot]}/{key}'] = dict(
                    mean=per_seed.mean(0).tolist(),
                    se=(per_seed.std(0, ddof=1) / np.sqrt(len(seeds))).tolist(),
                    per_seed={str(s): v.tolist() for s, v in zip(seeds, per_seed)})
                stats[f'{cycles_name}/{SLOT_NAMES[slot]}/{key}'] = dict(
                    seed_mean_profile=profile_stats(per_seed.mean(0)),
                    per_seed={str(s): profile_stats(v) for s, v in zip(seeds, per_seed)})
        for slot in (0, 1):
            for key in ('lam_model', 'lam_post'):
                per_seed = []
                for seed in seeds:
                    frame = inter[seed][0]
                    base = (frame['law'] == slot) & (frame['cycle'] >= low)
                    per_seed.append([float(frame[key][base & (frame['position'] == pos)].sum()
                                           / frame['lam_den'][base & (frame['position'] == pos)].sum())
                                     for pos in range(32)])
                per_seed = np.asarray(per_seed)
                lam[f'{cycles_name}/{SLOT_NAMES[slot]}/{key}'] = dict(
                    mean=per_seed.mean(0).tolist(),
                    per_seed={str(s): v.tolist() for s, v in zip(seeds, per_seed)},
                    position1=summarise(per_seed[:, 0]), position2=summarise(per_seed[:, 1]),
                    positions_2_32=summarise(per_seed[:, 1:].mean(1)),
                    positions_17_32=summarise(per_seed[:, 16:].mean(1)))
    # Mode projection of baseA blocks by cycle (positions 2..32) and distances.
    by_cycle = {}
    for seed in seeds:
        frame = inter[seed][0]
        values = []
        for cycle in range(8):
            sel = (frame['law'] == 0) & (frame['cycle'] == cycle) & (frame['position'] > 0)
            values.append(dict(
                lam_model=float(frame['lam_model'][sel].sum() / frame['lam_den'][sel].sum()),
                dist_A=float(frame['dist_A'][sel].mean()), dist_B=float(frame['dist_B'][sel].mean())))
        by_cycle[str(seed)] = values
    # Forgetting between visits versus context switching.
    forgetting = {}
    for slot in range(5):
        rows = []
        for seed in seeds:
            frame = inter[seed][0]
            model = frame['model']
            values = []
            for cycle in range(1, 8):
                now = (frame['law'] == slot) & (frame['cycle'] == cycle)
                before = (frame['law'] == slot) & (frame['cycle'] == cycle - 1)
                values.append(dict(
                    end_previous=float(model[before & (frame['position'] >= 28)].mean()),
                    position1=float(model[now & (frame['position'] == 0)].mean()),
                    position2=float(model[now & (frame['position'] == 1)].mean()),
                    positions_2_4=float(model[now & (frame['position'] >= 1)
                                              & (frame['position'] <= 3)].mean())))
            rows.append(dict(seed=seed,
                             end_previous=float(np.mean([v['end_previous'] for v in values])),
                             position1=float(np.mean([v['position1'] for v in values])),
                             position2=float(np.mean([v['position2'] for v in values])),
                             positions_2_4=float(np.mean([v['positions_2_4'] for v in values]))))
        forgetting[SLOT_NAMES[slot]] = dict(
            per_seed=rows,
            weight_loss=summarise([r['positions_2_4'] - r['end_previous'] for r in rows]),
            context_step=summarise([r['position1'] - r['position2'] for r in rows]))
    # Block-end evaluator curve (same-law support) from result.json.
    curve = {}
    for seed in seeds:
        values = [c['brier'] for c in results_by_seed[seed]['curve']]
        require(len(values) == 40, 'curve length')
        curve[str(seed)] = {SLOT_NAMES[s]: values[s::5] for s in range(5)}
    # Ideal-observer posterior on the true law / mode given the learner's own support.
    posterior = {}
    for slot in range(5):
        for name, position in (('position1', 0), ('positions_2_32', None)):
            true_law, true_mode = [], []
            for seed in seeds:
                frame = inter[seed][0]
                sel = (frame['law'] == slot) & (frame['cycle'] >= 1)
                sel = sel & ((frame['position'] == 0) if position == 0 else (frame['position'] > 0))
                true_law.append(float(frame['post_true'][sel].mean()))
                true_mode.append(float(frame['post_mode'][sel].mean()))
            posterior[f'{SLOT_NAMES[slot]}/{name}'] = dict(true_law=summarise(true_law),
                                                           true_mode=summarise(true_mode))
    starts = {}
    for seed in seeds:
        frame = inter[seed][0]
        starts[str(seed)] = {f'position{k + 1}': [
            float(frame['model'][(frame['law'] == 0) & (frame['cycle'] == c)
                                 & (frame['position'] == k)].mean()) for c in range(8)]
            for k in range(4)}
        starts[str(seed)]['lambda_position2'] = [
            float(frame['lam_model'][(frame['law'] == 0) & (frame['cycle'] == c) & (frame['position'] == 1)].sum()
                  / frame['lam_den'][(frame['law'] == 0) & (frame['cycle'] == c) & (frame['position'] == 1)].sum())
            for c in range(8)]
    return dict(profiles=profiles, stats=stats, lam=lam, lam_by_cycle=by_cycle,
                forgetting=forgetting, curve=curve, posterior=posterior, baseA_block_starts=starts)


def question_3(inter, fresh, seeds, meta):
    out = dict(per_seed={}, pooled={})
    collect_keys = {}

    def put(key, seed, value):
        collect_keys.setdefault(key, {})[seed] = value

    for seed in seeds:
        m = meta[seed]
        frame, cases = inter[seed]
        delayed = cases['factors'][:, 2] == 1
        lossy = cases['factors'][:, 1] == 1
        mine = {}
        roles = dict(delay_first=[m['delay_slot']], predecessor=[m['predecessor_slot']],
                     delay_later=list(range(m['delay_slot'] + 1, 5)), baseA=[0])
        for role, slots in roles.items():
            if not slots:
                continue
            law_rows = np.isin(cases['case_law'], slots)
            for source, values in (('model', cases['p']), ('io', cases['io']), ('truth', cases['y']),
                                   ('post', cases['post']), ('mix', cases['mix'])):
                result = contrast(values, cases['factors'], cases['signals'], cases['levels'],
                                  delayed & law_rows, 1, cases['clusters'])
                mine[f'interleaved/{role}/{source}'] = result
                put(f'interleaved/{role}/{source}', seed, result)
            for label, sel in (('position1', cases['position'] == 0),
                               ('positions_2_32', cases['position'] > 0),
                               ('cycles_5_8', cases['cycle'] >= 4),
                               ('cycles_1_4', cases['cycle'] < 4)):
                result = contrast(cases['p'], cases['factors'], cases['signals'], cases['levels'],
                                  delayed & law_rows & sel, 1, cases['clusters'])
                mine[f'interleaved/{role}/model/{label}'] = result
                put(f'interleaved/{role}/model/{label}', seed, result)
            if role != 'predecessor' and role != 'baseA':
                benefit = ideal_flip_benefit(cases, delayed & law_rows)
                mine[f'interleaved/{role}/ideal_benefit'] = benefit
                put(f'interleaved/{role}/ideal_benefit', seed, benefit)
            # Non-delayed placebo rows (the delay law cannot act there).
            for source, values in (('model', cases['p']), ('io', cases['io']), ('truth', cases['y'])):
                result = contrast(values, cases['factors'], cases['signals'], cases['levels'],
                                  ~delayed & law_rows, 1, cases['clusters'])
                put(f'interleaved/{role}/{source}/nondelayed_placebo', seed, result)
            result = contrast(cases['p'], cases['factors'], cases['signals'], cases['levels'],
                              ~delayed & law_rows & (cases['cycle'] >= 4), 1, cases['clusters'])
            put(f'interleaved/{role}/model/nondelayed_placebo/cycles_5_8', seed, result)
        for slot in (m['delay_slot'], m['predecessor_slot']):
            role = 'delay_first' if slot == m['delay_slot'] else 'predecessor'
            for cycle in range(8):
                sel = (cases['case_law'] == slot) & (cases['cycle'] == cycle)
                result = contrast(cases['p'], cases['factors'], cases['signals'], cases['levels'],
                                  delayed & sel, 1, cases['clusters'])
                put(f'interleaved/{role}/model/cycle_{cycle + 1}', seed, result)
        # Ideal-observer discriminability of a 32-record support.
        loglik = frame['support_loglik']
        d, before = m['delay_slot'], m['predecessor_slot']
        for role, slot in (('delay_first', d), ('predecessor', before)):
            sel = frame['law'] == slot
            llr = loglik[sel, d] - loglik[sel, before]
            put(f'discriminability/{role}_support/llr_delay_vs_predecessor', seed,
                dict(mean=float(llr.mean()), share_favouring_delay=float((llr > 0).mean()),
                     posterior_delay_vs_pair=float((1. / (1. + np.exp(-llr))).mean())))
        sel = frame['law'] == 0
        llr = loglik[sel, 0] - loglik[sel, 1]
        put('discriminability/baseA_support/llr_baseA_vs_baseB', seed,
            dict(mean=float(llr.mean()), share_favouring_delay=float((llr > 0).mean()),
                 posterior_delay_vs_pair=float((1. / (1. + np.exp(-llr))).mean())))
        # Positive control: the efficiency cue (lossy cases) in its first active law.
        e = m['efficiency_slot']
        for role, slot in (('efficiency_first', e), ('efficiency_predecessor', e - 1)):
            law_rows = cases['case_law'] == slot
            for source, values in (('model', cases['p']), ('io', cases['io']), ('post', cases['post'])):
                result = contrast(values, cases['factors'], cases['signals'], cases['levels'],
                                  lossy & law_rows, 0, cases['clusters'])
                put(f'control/{role}/{source}', seed, result)
            for source, values in (('model', cases['p']), ('io', cases['io'])):
                result = contrast(values, cases['factors'], cases['signals'], cases['levels'],
                                  ~lossy & law_rows, 0, cases['clusters'])
                put(f'control/{role}/{source}/nonlossy_placebo', seed, result)
        # Fresh fits.
        k = m['delay_stage']
        frame_f, cases_f = fresh[(seed, k)]
        delayed_f = cases_f['factors'][:, 2] == 1
        for source, values in (('model', cases_f['p']), ('io', cases_f['io']), ('truth', cases_f['y']),
                               ('io_predecessor', cases_f['io_predecessor'])):
            result = contrast(values, cases_f['factors'], cases_f['signals'], cases_f['levels'],
                              delayed_f, 1, cases_f['clusters'])
            put(f'fresh/delay_fit/{source}', seed, result)
        for source, values in (('model', cases_f['p']), ('io', cases_f['io'])):
            result = contrast(values, cases_f['factors'], cases_f['signals'], cases_f['levels'],
                              ~delayed_f, 1, cases_f['clusters'])
            put(f'fresh/delay_fit/{source}/nondelayed_placebo', seed, result)
            result = contrast(values, cases_f['factors'], cases_f['signals'], cases_f['levels'],
                              ~delayed_f & (cases_f['chunk'] >= 4), 1, cases_f['clusters'])
            put(f'fresh/delay_fit/{source}/nondelayed_placebo/chunks_5_8', seed, result)
        for chunk in range(8):
            result = contrast(cases_f['p'], cases_f['factors'], cases_f['signals'], cases_f['levels'],
                              delayed_f & (cases_f['chunk'] == chunk), 1, cases_f['clusters'])
            put(f'fresh/delay_fit/model/chunk_{chunk + 1}', seed, result)
        result = contrast(cases_f['p'], cases_f['factors'], cases_f['signals'], cases_f['levels'],
                          delayed_f & (cases_f['chunk'] >= 4), 1, cases_f['clusters'])
        put('fresh/delay_fit/model/chunks_5_8', seed, result)
        put('fresh/delay_fit/ideal_benefit', seed, ideal_flip_benefit(cases_f, delayed_f))
        for stage in (1, 2, 3):
            if stage == k:
                continue
            role = 'fresh/delay_inactive_fit' if stage < k else 'fresh/delay_later_fit'
            frame_s, cases_s = fresh[(seed, stage)]
            delayed_s = cases_s['factors'][:, 2] == 1
            result = contrast(cases_s['p'], cases_s['factors'], cases_s['signals'], cases_s['levels'],
                              delayed_s, 1, cases_s['clusters'])
            put(f'{role}/model/stage_{stage}', seed, result)
        out['per_seed'][str(seed)] = {key: value for key, value in mine.items()}
    pooled = {}
    for key, by_seed in collect_keys.items():
        values = [v for v in by_seed.values() if v is not None]
        if not values:
            continue
        if 'effect' in values[0]:
            pooled[key] = dict(
                horizon_mean=summarise([v['effect'][3] for v in values]),
                horizons=[summarise([v['effect'][h] for v in values]) for h in range(3)],
                per_seed={str(s): (None if v is None else dict(effect=v['effect'], se=v['se'], n=v['n']))
                          for s, v in by_seed.items()})
        else:
            pooled[key] = {name: summarise([v[name] for v in values]) for name in values[0]
                           if name != 'n'}
            pooled[key]['per_seed'] = {str(s): v for s, v in by_seed.items()}
    out['pooled'] = pooled
    # Seed-level difference: does the model separate delay-active from its predecessor?
    diffs = []
    for seed in seeds:
        a = collect_keys['interleaved/delay_first/model'][seed]
        b = collect_keys['interleaved/predecessor/model'][seed]
        diffs.append(a['effect'][3] - b['effect'][3])
    out['delay_minus_predecessor_model'] = summarise(diffs)
    for key in ('model/cycles_5_8', 'model/positions_2_32', 'io', 'post', 'mix', 'truth',
                'model/nondelayed_placebo'):
        out[f'delay_minus_predecessor/{key}'] = summarise([
            collect_keys[f'interleaved/delay_first/{key}'][s]['effect'][3]
            - collect_keys[f'interleaved/predecessor/{key}'][s]['effect'][3] for s in seeds])
    out['delayed_minus_nondelayed/delay_first/model'] = summarise([
        collect_keys['interleaved/delay_first/model'][s]['effect'][3]
        - collect_keys['interleaved/delay_first/model/nondelayed_placebo'][s]['effect'][3]
        for s in seeds])
    out['delayed_minus_nondelayed/fresh_delay_fit/model'] = summarise([
        collect_keys['fresh/delay_fit/model'][s]['effect'][3]
        - collect_keys['fresh/delay_fit/model/nondelayed_placebo'][s]['effect'][3] for s in seeds])
    out['efficiency_first_minus_predecessor/model'] = summarise([
        collect_keys['control/efficiency_first/model'][s]['effect'][3]
        - collect_keys['control/efficiency_predecessor/model'][s]['effect'][3] for s in seeds])
    ratios = []
    for seed in seeds:
        a = collect_keys['interleaved/delay_first/model'][seed]['effect'][3]
        b = collect_keys['interleaved/delay_first/io'][seed]['effect'][3]
        ratios.append(a / b)
    out['model_over_io_interleaved'] = summarise(ratios)
    ratios = []
    for seed in seeds:
        a = collect_keys['fresh/delay_fit/model'][seed]['effect'][3]
        b = collect_keys['fresh/delay_fit/io'][seed]['effect'][3]
        ratios.append(a / b)
    out['model_over_io_fresh'] = summarise(ratios)
    return out


def question_4(inter, results, seeds, meta, q1, q2, endpoint):
    rows = []
    for seed in seeds:
        frame = inter[seed][0]
        r = results[(seed, 'interleaved')]
        law = frame['law']
        replay, memory = r['replay'], r['memory']
        replay_law = law[replay]
        in_a = law == 0
        start_a = np.flatnonzero(in_a & (frame['position'] == 0))
        valid_memory = memory >= 0
        memory_law = np.where(valid_memory, law[np.clip(memory, 0, None)], -1)
        stats = dict(
            seed=seed, parity=meta[seed]['parity'], baseA_mode=meta[seed]['baseA_mode'],
            order=meta[seed]['order'], delay_stage=meta[seed]['delay_stage'],
            replay_share={SLOT_NAMES[s]: float((replay_law == s).mean()) for s in range(5)},
            replay_baseA_share_outside_baseA=float((replay_law[~in_a] == 0).mean()),
            replay_baseA_share_inside_baseA=float((replay_law[in_a] == 0).mean()),
            replay_baseA_share_cycles_5_8_outside=float(
                (replay_law[~in_a & (frame['cycle'] >= 4)] == 0).mean()),
            memory_baseA_at_block_start=[int((memory_law[b] == 0).sum()) for b in start_a],
            memory_baseA_mean=float(((memory_law == 0).sum(1) / np.maximum(valid_memory.sum(1), 1)).mean()),
            replay_cross_law_support_share=float(((replay % 32 == 0) & (replay > 0)).mean()),
            replay_mean_age_batches=float((frame['batch'][:, None] - replay).mean()),
            baseA_online_cycle8=q1['per_seed'][seed]['baseA_cycle8'],
            baseA_online_cycles_5_8=float(frame['model'][in_a & (frame['cycle'] >= 4)].mean()),
            baseA_block_end_curve=q2['curve'][str(seed)]['baseA'],
            baseA_endpoint=endpoint[str(seed)]['interleaved'][0]['model_brier'],
            gap_cycle8=q1['per_seed'][seed]['gap_cycle8'],
            baseA_lambda_positions_2_32=float(np.mean(
                q2['lam']['cycles_2_8/baseA/lam_model']['per_seed'][str(seed)][1:])))
        rows.append(stats)
    # Visit-level relation (cycles 2..8): replay composition before a baseA visit
    # versus the context-driven switch at its second batch (first same-law support).
    visits = []
    for seed in seeds:
        frame = inter[seed][0]
        r = results[(seed, 'interleaved')]
        law, replay, memory = frame['law'], r['replay'], r['memory']
        switch = (np.arange(len(law)) % 32 == 0) & (np.arange(len(law)) > 0) & np.isin(law, (0, 1))
        for cycle in range(1, 8):
            start = cycle * 5 * 32
            previous = np.arange(start - 4 * 32, start)
            drawn = replay[previous].ravel()
            held = memory[start][memory[start] >= 0]
            visits.append(dict(
                seed=seed, cycle=cycle + 1,
                replay_baseA_share_prior_4_blocks=float((law[drawn] == 0).mean()),
                replay_mode_switch_share_prior_4_blocks=float(switch[drawn].mean()),
                memory_baseA_at_start=int((law[held] == 0).sum()),
                memory_mode_switch_at_start=int(switch[held].sum()),
                position1=float(frame['model'][start]), position2=float(frame['model'][start + 1]),
                lambda_position2=float(frame['lam_model'][start + 1] / frame['lam_den'][start + 1]),
                visit_mean=float(frame['model'][start:start + 32].mean())))
    visit_corr = {}
    for x_name in ('replay_baseA_share_prior_4_blocks', 'replay_mode_switch_share_prior_4_blocks',
                   'memory_baseA_at_start', 'memory_mode_switch_at_start'):
        for y_name in ('position2', 'lambda_position2', 'visit_mean'):
            x = np.asarray([v[x_name] for v in visits], dtype=np.float64)
            y = np.asarray([v[y_name] for v in visits], dtype=np.float64)
            # Within-seed centring removes seed-level offsets.
            seeds_arr = np.asarray([v['seed'] for v in visits])
            xc, yc = x.copy(), y.copy()
            for s in seeds:
                xc[seeds_arr == s] -= x[seeds_arr == s].mean()
                yc[seeds_arr == s] -= y[seeds_arr == s].mean()
            visit_corr[f'{x_name}~{y_name}'] = dict(
                pooled=float(np.corrcoef(x, y)[0, 1]), within_seed=float(np.corrcoef(xc, yc)[0, 1]),
                n=len(visits))
    correlations = {}
    x = np.asarray([r['replay_baseA_share_outside_baseA'] for r in rows])
    for name in ('baseA_online_cycles_5_8', 'baseA_endpoint', 'gap_cycle8'):
        y = np.asarray([r[name] for r in rows])
        correlations[f'replay_baseA_share_outside_baseA~{name}'] = float(np.corrcoef(x, y)[0, 1])
    parity = {}
    for value in (0, 1):
        sel = [r for r in rows if r['baseA_mode'] == value]
        parity[f'baseA_mode_{value}'] = dict(
            seeds=[r['seed'] for r in sel],
            baseA_endpoint=summarise([r['baseA_endpoint'] for r in sel]),
            baseA_online_cycles_5_8=summarise([r['baseA_online_cycles_5_8'] for r in sel]))
    endpoint_lambda = [endpoint[str(s)]['interleaved'][0]['lambda_modeA'] for s in seeds]
    endpoint_brier = [endpoint[str(s)]['interleaved'][0]['model_brier'] for s in seeds]
    correlations['endpoint_lambda~endpoint_brier'] = float(np.corrcoef(endpoint_lambda, endpoint_brier)[0, 1])
    x = np.asarray([r['replay_cross_law_support_share'] for r in rows])
    for name in ('baseA_online_cycles_5_8', 'baseA_endpoint'):
        y = np.asarray([r[name] for r in rows])
        correlations[f'replay_cross_law_support_share~{name}'] = float(np.corrcoef(x, y)[0, 1])
    for row, lam_value in zip(rows, endpoint_lambda):
        row['baseA_endpoint_lambda'] = lam_value
    return dict(rows=rows, correlations=correlations, parity=parity, visits=visits,
                visit_correlations=visit_corr)


def structure_checks(seeds):
    out = {}
    for seed in seeds:
        laws = reference_laws(seed)
        modes = [law.mode for law in laws]
        d = seed_meta(seed)['delay_slot']
        first, previous = laws[d], laws[d - 1]
        out[str(seed)] = dict(
            modes=modes, baseA_unique_mode=modes.count(modes[0]) == 1,
            baseA_stream_share=.2,
            delay_first=dict(mode=first.mode, active=list(first.active)),
            predecessor=dict(mode=previous.mode, active=list(previous.active)),
            differs_only_by_delay=(first.mode == previous.mode
                                   and set(first.active) ^ set(previous.active) == {1}))
    return out


# --------------------------------------------------------------------------- #
# Stage 3: outputs                                                             #
# --------------------------------------------------------------------------- #

def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path, rows, fields):
    lines = [','.join(fields)]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                cells.append(f'{value:.6f}')
            elif isinstance(value, (list, tuple)):
                cells.append('"' + ' '.join(map(str, value)) + '"')
            else:
                cells.append('' if value is None else str(value))
        lines.append(','.join(cells))
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def analyse(results, seeds, draws, output):
    cfg = config()
    size = cfg['batch_size']
    output.mkdir(parents=True, exist_ok=True)
    meta = {seed: seed_meta(seed) for seed in seeds}
    inter, fresh = {}, {}
    for seed in seeds:
        inter[seed] = interleaved_frame(results[(seed, 'interleaved')], draws, size)
        for stage in (1, 2, 3):
            fresh[(seed, stage)] = fresh_frame(results[(seed, f'fresh_{stage}')], draws, size)
    endpoint = {str(seed): {phase: endpoint_summary(results[(seed, phase)]['endpoint'], draws,
                                                    phase == 'interleaved')
                            for phase in PHASES} for seed in seeds}
    by_seed = {seed: results[(seed, 'interleaved')] for seed in seeds}
    verification = dict(
        batches=int(sum(results[key]['batches_verified'] for key in results)),
        cases=int(sum(results[key]['batches_verified'] for key in results) * size),
        phases=len(results), draws=draws,
        checks=['saved observations byte-match AcquisitionWorld.dataset(law, 32, training_seed)',
                'actions, performed outcomes, counterfactual truth, affected/valid masks match',
                'packet query/support fingerprints match result.json',
                'pixel-decoded source/lossy/delayed/cues equal evaluator factors/signals; '
                'gauge levels equal rint(reserve/18*240)',
                'ideal-observer physics with the TRUE latent state reproduces every truth pattern',
                'endpoint panels/supports regenerate and saved endpoint Brier recomputes exactly'])
    verification['mc_correction'] = float(np.mean([inter[s][1]['io_var'].mean() for s in seeds]))
    q1 = question_1(inter, fresh, seeds)
    q2 = question_2(inter, seeds, by_seed)
    q3 = question_3(inter, fresh, seeds, meta)
    q4 = question_4(inter, results, seeds, meta, q1, q2, endpoint)
    report = dict(label=LABEL, draws=draws, seeds=seeds, meta={str(k): v for k, v in meta.items()},
                  structure=structure_checks(seeds), verification=verification,
                  endpoint=endpoint, q1=q1, q2=q2, q3=q3, q4=q4)
    report = jsonable(report)
    (output / 'results.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n',
                                         encoding='utf-8', newline='\n')
    write_csv(output / 'cycle_brier.csv', q1['rows'],
              ['seed', 'law', 'cycle', 'model', 'model_pos2plus', 'io', 'post', 'mix', 'marginal',
               'io_as_baseB'])
    write_csv(output / 'fresh_chunk_brier.csv', q1['fresh_rows'],
              ['seed', 'stage', 'chunk', 'model', 'io', 'io_predecessor', 'marginal'])
    profile_rows = []
    for slot in range(5):
        for pos in range(32):
            row = dict(law=SLOT_NAMES[slot], position=pos + 1)
            for key in ('model', 'post', 'io'):
                row[key] = report['q2']['profiles'][f'cycles_2_8/{SLOT_NAMES[slot]}/{key}']['mean'][pos]
                row[key + '_late'] = report['q2']['profiles'][f'cycles_5_8/{SLOT_NAMES[slot]}/{key}']['mean'][pos]
            if slot < 2:
                row['lambda_model'] = report['q2']['lam'][f'cycles_2_8/{SLOT_NAMES[slot]}/lam_model']['mean'][pos]
                row['lambda_post'] = report['q2']['lam'][f'cycles_2_8/{SLOT_NAMES[slot]}/lam_post']['mean'][pos]
            profile_rows.append(row)
    write_csv(output / 'position_profile.csv', profile_rows,
              ['law', 'position', 'model', 'post', 'io', 'model_late', 'post_late', 'io_late',
               'lambda_model', 'lambda_post'])
    contrast_rows = []
    for key, value in report['q3']['pooled'].items():
        if 'horizon_mean' in value:
            for seed, item in value['per_seed'].items():
                if item is None:
                    continue
                contrast_rows.append(dict(cell=key, seed=seed, effect_mean=item['effect'][3],
                                          se_mean=item['se'][3], effect_h4=item['effect'][0],
                                          effect_h8=item['effect'][1], effect_h12=item['effect'][2],
                                          n=item['n']))
    write_csv(output / 'cue_contrasts.csv', contrast_rows,
              ['cell', 'seed', 'effect_mean', 'se_mean', 'effect_h4', 'effect_h8', 'effect_h12', 'n'])
    write_csv(output / 'replay_and_seed_table.csv', report['q4']['rows'],
              ['seed', 'baseA_mode', 'order', 'delay_stage', 'baseA_endpoint', 'baseA_online_cycle8',
               'baseA_online_cycles_5_8', 'gap_cycle8', 'baseA_lambda_positions_2_32',
               'replay_baseA_share_outside_baseA', 'replay_baseA_share_inside_baseA',
               'replay_cross_law_support_share', 'memory_baseA_at_block_start'])
    figures(report, output)
    (output / 'summary.md').write_text(render_summary(report), encoding='utf-8', newline='\n')
    print(json.dumps(dict(written=str(output)), indent=None), flush=True)
    return report


LAW_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')  # fixed categorical order
INK, MUTED, GRID, SURFACE, REFERENCE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb', '#8a8984'
LAW_LABELS = ('Base A (unique mode)', 'Base B', 'Stage 1', 'Stage 2', 'Stage 3')


def _axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=11, color=INK, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def figures(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    q1, q2, q3 = report['q1'], report['q2'], report['q3']
    stamp = LABEL
    cycles = np.arange(1, 9)
    # Figure 1: per-law online Brier by cycle, and baseA by seed.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE)
    ax = axes[0]
    for slot in range(5):
        values = [q1['table'][f'{slot}_{c}']['model']['mean'] for c in cycles]
        ax.plot(cycles, values, color=LAW_COLORS[slot], lw=2, marker='o', ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=LAW_LABELS[slot], zorder=3)
    ideal = [np.mean([q1['table'][f'{s}_{c}']['io']['mean'] for s in range(5)]) for c in cycles]
    ax.plot(cycles, ideal, color=REFERENCE, lw=1.4, label='Ideal observer, law known', zorder=2)
    ax.text(8.1, ideal[-1], 'ideal', color=MUTED, fontsize=8, va='center')
    _axes(ax, 'Online all-case Brier per law and cycle (seed mean)', 'Interleaving cycle',
          'Brier (lower is better)')
    ax.set_ylim(0, None)
    ax.set_xlim(.7, 8.6)
    ax.legend(frameon=False, fontsize=8, loc='lower left', ncol=2, labelcolor=INK)
    ax = axes[1]
    for seed in report['seeds']:
        values = [r['model'] for r in q1['rows'] if r['seed'] == seed and r['slot'] == 0]
        ax.plot(cycles, values, color=REFERENCE, lw=1, alpha=.8, zorder=2,
                label='Base A, individual seeds' if seed == report['seeds'][0] else None)
    mean_a = [q1['table'][f'0_{c}']['model']['mean'] for c in cycles]
    others = [np.mean([q1['table'][f'{s}_{c}']['model']['mean'] for s in range(1, 5)]) for c in cycles]
    ax.plot(cycles, mean_a, color=LAW_COLORS[0], lw=2.2, marker='o', ms=5, markeredgecolor=SURFACE,
            label='Base A, seed mean', zorder=3)
    ax.plot(cycles, others, color=INK, lw=1.6, label='Other four laws, mean', zorder=3)
    _axes(ax, 'Base A stays flat across cycles (seeds and mean)', 'Interleaving cycle',
          'Brier (lower is better)')
    ax.set_ylim(0, None)
    ax.set_xlim(.7, 8.9)
    ax.legend(frameon=False, fontsize=8, loc='lower left', labelcolor=INK)
    fig.text(.01, .005, stamp, color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(output / 'online_brier_by_cycle.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: within-block position profile and mode-A projection.
    positions = np.arange(1, 33)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE)
    ax = axes[0]
    for slot in range(5):
        values = q2['profiles'][f'cycles_2_8/{SLOT_NAMES[slot]}/model']['mean']
        ax.plot(positions, values, color=LAW_COLORS[slot], lw=2, label=LAW_LABELS[slot], zorder=3)
    post = q2['profiles']['cycles_2_8/baseA/post']['mean']
    ax.plot(positions, post, color=REFERENCE, lw=1.4, zorder=2,
            label='Ideal observer, same 32-record support (Base A)')
    _axes(ax, 'Brier by batch position within a 1024-arrival block (cycles 2-8)',
          'Batch position in block (1 = support from previous law)', 'Brier (lower is better)')
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc='upper right', labelcolor=INK)
    ax = axes[1]
    for slot, name in ((0, 'Base A blocks'), (1, 'Base B blocks')):
        ax.plot(positions, q2['lam'][f'cycles_2_8/{SLOT_NAMES[slot]}/lam_model']['mean'],
                color=LAW_COLORS[slot], lw=2, label=f'Model, {name}', zorder=3)
        ax.plot(positions, q2['lam'][f'cycles_2_8/{SLOT_NAMES[slot]}/lam_post']['mean'],
                color=REFERENCE, lw=1.4, zorder=2,
                label='Ideal observer, same support' if slot == 0 else None)
    lam_end = np.mean([report['endpoint'][str(s)]['interleaved'][0]['lambda_modeA'] for s in report['seeds']])
    ax.scatter([2], [lam_end], s=60, color=INK, zorder=4, edgecolor=SURFACE, linewidth=2,
               label='Endpoint probe, Base A support, no Base A update')
    _axes(ax, 'Mode-A projection (0 = like Base B, 1 = like Base A)',
          'Batch position in block', 'Projection coefficient')
    ax.set_ylim(-.05, 1.08)
    ax.legend(frameon=False, fontsize=8, loc='center right', labelcolor=INK)
    fig.text(.01, .005, stamp, color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(output / 'within_block_profile.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Figure 3: aligned-minus-misaligned contrasts for the arrival-delay cue.
    pooled = q3['pooled']
    rows = [
        ('Ideal observer, law known (delay-first law)', 'interleaved/delay_first/io'),
        ('Evaluator truth (delay-first law)', 'interleaved/delay_first/truth'),
        ('Ideal observer, same one-batch support', 'interleaved/delay_first/post'),
        ('Model, fresh delay fit, chunks 5-8', 'fresh/delay_fit/model/chunks_5_8'),
        ('Model, fresh delay fit, non-delayed placebo', 'fresh/delay_fit/model/nondelayed_placebo'),
        ('Model, interleaved delay-first law', 'interleaved/delay_first/model'),
        ('Model, interleaved delay-first, cycles 5-8', 'interleaved/delay_first/model/cycles_5_8'),
        ('Model, interleaved delay-first, non-delayed placebo',
         'interleaved/delay_first/model/nondelayed_placebo'),
        ('Model, interleaved predecessor (delay inactive)', 'interleaved/predecessor/model'),
        ('Model, interleaved Base A (delay inactive)', 'interleaved/baseA/model'),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor=SURFACE)
    for i, (label, key) in enumerate(rows):
        stat = pooled[key]['horizon_mean']
        ax.scatter(stat['values'], np.full(len(stat['values']), i), s=16, color=REFERENCE,
                   zorder=2, alpha=.9)
        color = REFERENCE if key.endswith(('io', 'truth', 'post')) else LAW_COLORS[0]
        ax.errorbar(stat['mean'], i, xerr=2 * stat['se'], fmt='o', ms=7, color=color if i > 2 else INK,
                    mec=SURFACE, mew=1.5, capsize=3, lw=1.6, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    _axes(ax, 'Arrival-delay cue: aligned minus misaligned transfer forecasts (delayed cases unless placebo)',
          'Survival-probability difference, mean of horizons 4/8/12 (dots: seeds; bar: mean +/- 2 SE)', '')
    fig.text(.01, .005, stamp, color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(output / 'delay_cue_contrast.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _f(value, digits=4):
    return '--' if value is None else f'{value:.{digits}f}'


def _ms(stat, digits=4):
    if stat is None:
        return '--'
    return f"{stat['mean']:.{digits}f} +/- {stat['se']:.{digits}f}"


def render_summary(report):
    q1, q2, q3, q4 = report['q1'], report['q2'], report['q3'], report['q4']
    seeds = report['seeds']
    pooled = q3['pooled']
    ver = report['verification']
    L = []
    add = L.append
    add('# Online ideal-observer reanalysis of the representation-learning qualification fits')
    add('')
    add(f'**{LABEL}.** This is a post hoc reanalysis of saved online arrays. It does not change the '
        'locked qualification result (STOP before the candidate comparison), any threshold, or any '
        'decision. Ideal observers below are evaluator-side analysis references that use law '
        'identities and the simulator; they are not learners and were never trained or fed back.')
    add('')
    add('Script: `scripts/ideal_observer_online.py`. Machine-readable results: `results.json`; '
        'tables: `cycle_brier.csv`, `fresh_chunk_brier.csv`, `position_profile.csv`, '
        '`cue_contrasts.csv`, `replay_and_seed_table.csv`; figures: `online_brier_by_cycle.png`, '
        '`within_block_profile.png`, `delay_cue_contrast.png`.')
    add('')
    add('## Data, verification and references')
    add('')
    add(f"* Saved online arrays of all {ver['phases']} qualification phases (6 seeds x fresh_1..3 + "
        f"interleaved): {ver['batches']:,} training batches, {ver['cases']:,} online cases. Each "
        'forecast (5 actions x 3 horizons) was made BEFORE that batch\'s feedback, with the previous '
        'batch as the 32-record support. No model, checkpoint (`*.pt`) or study runner was executed.')
    for check in ver['checks']:
        add(f'* Verified: {check}.')
    add('* Structure verified for every seed: stage laws use mode 1-seed%2, so Base B and the three '
        'novelty laws share one hidden mode and Base A is the only law with its mode (20% of the '
        'interleaved stream); the law that first adds the delay cue differs from the block '
        'immediately before it only by the delay dependency (same mode, active sets differ by {1}): '
        + ', '.join(f"{s}: {report['structure'][str(s)]['baseA_unique_mode'] and report['structure'][str(s)]['differs_only_by_delay']}"
                    for s in seeds) + '.')
    add(f"* References (all scored with the same all-case Brier as the model: 5 actions x 3 horizons "
        f"against evaluator counterfactual truth). **Law-known ideal observer**: decodes reserves "
        f"(gauge bins), weather direction, lossy/delayed glyphs and the three pulse cues from pixels, "
        f"knows the batch's law, integrates hidden supply noise and gauge quantisation by Monte Carlo "
        f"({report['draws']} draws/case, common random numbers; Monte Carlo variance removed from Brier, "
        f"mean correction {_f(ver.get('mc_correction'), 5)}). **Support-posterior observer**: uniform "
        f"prior over the 5 interleaved laws, conditioned only on the learner's own 32-record support. "
        f"**No-context observer**: uniform mixture of the 5 laws. **Law marginal**: running "
        f"performed-action marginal of the batch's own law (the qualification's marginal reference).")
    add('')
    # ---------------------------------------------------------------- Q1
    add('## Q1. Per-law online Brier by cycle (interleaved fit)')
    add('')
    add('Seed-mean online all-case Brier per 1024-arrival block (32 batches); each law gets one block per cycle.')
    add('')
    add('| Law | ' + ' | '.join(f'c{c}' for c in range(1, 9)) + ' | c1-c8 | law-known ideal | same-support ideal | no-context ideal | law marginal (c2-c8) |')
    add('|---|' + '---:|' * 13)
    for slot in range(5):
        cells = [q1['table'][f'{slot}_{c}']['model']['mean'] for c in range(1, 9)]
        io = np.mean([q1['table'][f'{slot}_{c}']['io']['mean'] for c in range(1, 9)])
        post = np.mean([q1['table'][f'{slot}_{c}']['post']['mean'] for c in range(1, 9)])
        mix = np.mean([q1['table'][f'{slot}_{c}']['mix']['mean'] for c in range(1, 9)])
        marg = np.mean([q1['table'][f'{slot}_{c}']['marginal']['mean'] for c in range(2, 9)])
        add(f'| {LAW_LABELS[slot]} | ' + ' | '.join(_f(v) for v in cells)
            + f' | {_f(cells[0] - cells[-1])} | {_f(io)} | {_f(post)} | {_f(mix)} | {_f(marg)} |')
    add('')
    ps = q1['per_seed']
    gaps = [ps[str(s)]['gap_cycle8'] for s in seeds]
    gaps2 = [ps[str(s)]['gap_cycle8_pos2plus'] for s in seeds]
    imp_a = [ps[str(s)]['improvement']['baseA'] for s in seeds]
    add('| Seed | Base A c1 | Base A c8 | best cycle | others c1 | others c8 | c8 gap (A - others) | c8 gap, positions 2-32 | A excess over law-known ideal, c8 | others excess, c8 | A with Base-B-law ideal |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for s in seeds:
        v = ps[str(s)]
        add(f"| {s} | {_f(v['baseA_cycle1'])} | {_f(v['baseA_cycle8'])} | {v['baseA_best_cycle']} | "
            f"{_f(v['others_cycle1'])} | {_f(v['others_cycle8'])} | {_f(v['gap_cycle8'])} | "
            f"{_f(v['gap_cycle8_pos2plus'])} | {_f(v['baseA_excess_io_cycle8'])} | "
            f"{_f(v['others_excess_io_cycle8'])} | {_f(v['baseA_as_baseB_io_cycle8'])} |")
    add('')
    fresh = q1['fresh_rows']
    fresh8 = {st: np.mean([r['model'] for r in fresh if r['stage'] == st and r['chunk'] == 8]) for st in (1, 2, 3)}
    inter8 = {st: q1['table'][f'{st + 1}_8']['model']['mean'] for st in (1, 2, 3)}
    starts = q2['baseA_block_starts']
    p1_by_cycle = [np.mean([starts[str(s)]['position1'][c] for s in seeds]) for c in range(8)]
    p2_by_cycle = [np.mean([starts[str(s)]['position2'][c] for s in seeds]) for c in range(8)]
    add(f"**Answer.** Base A does **not** improve over cycles: its seed-mean online Brier is "
        f"{_f(q1['table']['0_1']['model']['mean'])} in cycle 1 and {_f(q1['table']['0_8']['model']['mean'])} "
        f"in cycle 8 (per-seed cycle-1 minus cycle-8 change {_ms(summarise(imp_a))}; "
        f"{sum(v < 0 for v in imp_a)} of {len(imp_a)} seeds get worse), "
        f"and every cycle lies in {_f(min(q1['table'][f'0_{c}']['model']['mean'] for c in range(1, 9)), 3)}-"
        f"{_f(max(q1['table'][f'0_{c}']['model']['mean'] for c in range(1, 9)), 3)}. The other laws "
        f"improve, but almost entirely between cycle 1 and 2 (Base B {_f(q1['table']['1_1']['model']['mean'], 3)} -> "
        f"{_f(q1['table']['1_2']['model']['mean'], 3)}, stage 1 {_f(q1['table']['2_1']['model']['mean'], 3)} -> "
        f"{_f(q1['table']['2_2']['model']['mean'], 3)}) and are flat or drift upward afterwards (Base B "
        f"{_f(q1['table']['1_2']['model']['mean'], 3)} in cycle 2 -> {_f(q1['table']['1_8']['model']['mean'], 3)} "
        f"in cycle 8); Base A's cycle-1 block is "
        f"the model's very first block, which is why it has no early drop. At cycle 8 Base A ends "
        f"{_ms(summarise(gaps))} Brier above the mean of the other four laws (seed range "
        f"{_f(min(gaps))} to {_f(max(gaps))}; {_ms(summarise(gaps2))} excluding the first, cross-law-support "
        f"batch). All laws are far from the law-known ideal (about .044 for every law); Base A's excess "
        f"is {_ms(summarise([ps[str(s)]['baseA_excess_io_cycle8'] for s in seeds]))} versus "
        f"{_ms(summarise([ps[str(s)]['others_excess_io_cycle8'] for s in seeds]))} for the others. "
        f"Online Base A is nevertheless much better than predicting it with the majority mode "
        f"(Base-B-law ideal on Base A cases: {_f(np.mean([ps[str(s)]['baseA_as_baseB_io_cycle8'] for s in seeds]), 3)}) "
        f"or with its own law marginal ({_f(np.mean([q1['table'][f'0_{c}']['marginal']['mean'] for c in range(2, 9)]), 3)} in cycles 2-8).")
    add('')
    add(f"The locked endpoint value (Base A {_f(np.mean([report['endpoint'][str(s)]['interleaved'][0]['model_brier'] for s in seeds]))}) "
        f"is much worse than the block-averaged online value because the endpoint is taken at the worst "
        f"point of the cycle: after 4,096 arrivals of mode-B laws, with Base A support but before any "
        f"Base A update. Its online analogue is the start of a Base A visit (cycle-mean position 1 = "
        f"{_f(np.mean(p1_by_cycle[1:]), 3)} with the previous law's support; position 2 = "
        f"{_f(np.mean(p2_by_cycle[1:]), 3)} with Base A support), not the block average. The block-end "
        f"evaluator curve in `result.json` (same-law support after 32 Base A batches) is "
        f"{_f(np.mean([np.mean(q2['curve'][str(s)]['baseA']) for s in seeds]), 3)} on average.")
    add('')
    add(f"Interleaving also costs the stage laws relative to their fresh fits at matched exposure (chunk/cycle 8): "
        + '; '.join(f'stage {st} fresh {_f(fresh8[st], 3)} vs interleaved {_f(inter8[st], 3)}' for st in (1, 2, 3))
        + '.')
    add('')
    add('Cross-law ideal-observer Brier (rows: true law of the online cases; columns: law used by the ideal forecast). '
        'The hidden mode dominates; the cue laws are close to each other.')
    add('')
    add('| True \\ used | ' + ' | '.join(LAW_LABELS) + ' |')
    add('|---|' + '---:|' * 5)
    for t in range(5):
        add(f'| {LAW_LABELS[t]} | ' + ' | '.join(_f(q1['cross'][f'{t}_{u}']['mean'], 3) for u in range(5)) + ' |')
    add('')
    # ---------------------------------------------------------------- Q2
    add('## Q2. Within-block dynamics (switch cost and recognition)')
    add('')
    add('Seed-mean Brier by batch position inside each 32-batch block, cycles 2-8 pooled (position 1 '
        'uses the previous law\'s last batch as support; later positions use same-law support). '
        'Plateau = positions 17-32. Half-life = first position whose 3-batch running excess is at most '
        'half the position-1 excess.')
    add('')
    add('| Law | pos 1 | pos 2 | pos 3 | pos 4 | pos 8 | pos 16 | plateau | switch cost | share resolved at pos 2 | half-life | same-support ideal pos 1 -> pos 2 |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for slot in range(5):
        prof = q2['profiles'][f'cycles_2_8/{SLOT_NAMES[slot]}/model']['mean']
        st = q2['stats'][f'cycles_2_8/{SLOT_NAMES[slot]}/model']['seed_mean_profile']
        post = q2['profiles'][f'cycles_2_8/{SLOT_NAMES[slot]}/post']['mean']
        add(f"| {LAW_LABELS[slot]} | {_f(prof[0], 3)} | {_f(prof[1], 3)} | {_f(prof[2], 3)} | {_f(prof[3], 3)} | "
            f"{_f(prof[7], 3)} | {_f(prof[15], 3)} | {_f(st['plateau_17_32'], 3)} | {_f(st['switch_cost'], 3)} | "
            f"{_f(st['share_resolved_at_2'], 2)} | {st['half_life_position']} | {_f(post[0], 3)} -> {_f(post[1], 3)} |")
    add('')
    lamA = q2['lam']['cycles_2_8/baseA/lam_model']
    lamB = q2['lam']['cycles_2_8/baseB/lam_model']
    add('Mode-A projection coefficient of the forecasts, lambda = <p - p_B, p_A - p_B> / |p_A - p_B|^2, where p_A and p_B '
        'are the law-known ideal forecasts under Base A and Base B (0 = forecasts vary like Base B, 1 = like Base A):')
    add('')
    add('| Blocks | pos 1 | pos 2 | pos 3 | pos 4 | pos 8 | pos 17-32 | same-support ideal pos 1 / pos 2+ |')
    add('|---|---:|---:|---:|---:|---:|---:|---|')
    for name, lam, post in (('Base A', lamA, q2['lam']['cycles_2_8/baseA/lam_post']),
                            ('Base B', lamB, q2['lam']['cycles_2_8/baseB/lam_post'])):
        m = lam['mean']
        add(f"| {name} | {_f(m[0], 2)} | {_f(m[1], 2)} | {_f(m[2], 2)} | {_f(m[3], 2)} | {_f(m[7], 2)} | "
            f"{_f(lam['positions_17_32']['mean'], 2)} | {_f(post['mean'][0], 2)} / {_f(np.mean(post['mean'][1:]), 2)} |")
    add('')
    fg = q2['forgetting']
    add('| Law | end of previous visit (pos 29-32) | next visit pos 1 | next visit pos 2 | weight loss between visits (pos 2-4 minus previous end) | context step (pos 1 minus pos 2) |')
    add('|---|---:|---:|---:|---:|---:|')
    for slot in range(5):
        v = fg[SLOT_NAMES[slot]]
        rows = v['per_seed']
        add(f"| {LAW_LABELS[slot]} | {_f(np.mean([r['end_previous'] for r in rows]), 3)} | "
            f"{_f(np.mean([r['position1'] for r in rows]), 3)} | {_f(np.mean([r['position2'] for r in rows]), 3)} | "
            f"{_ms(v['weight_loss'], 3)} | {_ms(v['context_step'], 3)} |")
    add('')
    lam_end = [report['endpoint'][str(s)]['interleaved'][0]['lambda_modeA'] for s in seeds]
    stA = q2['stats']['cycles_2_8/baseA/model']['seed_mean_profile']
    stB = q2['stats']['cycles_2_8/baseB/model']['seed_mean_profile']
    disc = pooled['discriminability/baseA_support/llr_baseA_vs_baseB']
    add(f"**Answer.** Base A error **drops sharply after batch 1** (evidence of context use) but only "
        f"part-way: {_f(stA['position1'], 3)} at position 1 -> {_f(stA['position2'], 3)} at position 2 "
        f"({_f(stA['share_resolved_at_2'] * 100, 0)}% of the switch cost of {_f(stA['switch_cost'], 3)} "
        f"resolved), then a slow decline through the rest of the block (position 4 "
        f"{_f(q2['profiles']['cycles_2_8/baseA/model']['mean'][3], 3)}, position 8 "
        f"{_f(q2['profiles']['cycles_2_8/baseA/model']['mean'][7], 3)}, position 16 "
        f"{_f(q2['profiles']['cycles_2_8/baseA/model']['mean'][15], 3)}) to "
        f"{_f(stA['plateau_17_32'], 3)} in positions 17-32. The mode-A projection moves "
        f"{_f(lamA['mean'][0], 2)} -> {_f(lamA['mean'][1], 2)} at the first same-law support and only "
        f"reaches {_f(lamA['positions_17_32']['mean'], 2)} late in the block. The ideal observer given the "
        f"same single support recognises Base A completely at position 2 (Brier "
        f"{_f(q2['profiles']['cycles_2_8/baseA/post']['mean'][1], 3)}, lambda 1.00, posterior on Base A "
        f"{_f(q2['posterior']['baseA/positions_2_32']['true_law']['mean'], 2)}; one 32-record Base A support "
        f"favours Base A over Base B by {_f(disc['mean']['mean'], 1)} nats on average), so the residual is "
        f"not an information limit of the 32-record window. Base B, which follows Base A, pays the "
        f"largest position-1 cost ({_f(stB['position1'], 3)}; its forecasts are Base-A-like, lambda "
        f"{_f(lamB['mean'][0], 2)}) and recovers faster ({_f(stB['share_resolved_at_2'] * 100, 0)}% at "
        f"position 2). Stage laws have small switch costs ({', '.join(_f(q2['stats'][f'cycles_2_8/stage{k}/model']['seed_mean_profile']['switch_cost'], 3) for k in (1, 2, 3))}) "
        f"because their predecessor shares their mode. Recognition delay: the ideal observer needs one "
        f"batch; the model's half-life is {stA['half_life_position']} batches for Base A, but the remaining "
        f"{_f(stA['excess_after_2'], 3)} excess after position 2 decays only over the whole block, i.e. "
        f"the rest of mode A is re-learned in the weights on every visit.")
    add('')
    add(f"Two further observations separate context from weights. (i) Forgetting between visits: with the "
        f"correct same-law support (positions 2-4), Base A starts each visit "
        f"{_ms(fg['baseA']['weight_loss'], 3)} Brier worse than it ended its previous visit (Base B "
        f"{_ms(fg['baseB']['weight_loss'], 3)}; stage laws below .01), so what the model knows about mode A "
        f"is held mostly in weights that four mode-B blocks overwrite. (ii) A natural experiment at the "
        f"locked endpoint: Base A support with NO Base A weight update gives Brier "
        f"{_f(np.mean([report['endpoint'][str(s)]['interleaved'][0]['model_brier'] for s in seeds]), 3)} and "
        f"lambda {_f(np.mean(lam_end), 2)} (seed values {', '.join(_f(x, 2) for x in lam_end)}), essentially "
        f"the online position-2 level ({_f(stA['position2'], 3)}, lambda {_f(lamA['mean'][1], 2)}); so the "
        f"position 1 -> 2 drop is attributable mostly to the support, not to the 12 updates on the "
        f"first Base A batch. Context therefore carries roughly half of the mode switch; the rest is "
        f"in-weight relearning.")
    add('')
    # ---------------------------------------------------------------- Q3
    add('## Q3. Arrival-delay dependency online')
    add('')
    add('Estimator: on delayed cases (glyph delayed = 1), D = mean over request sizes 2 and 4 of '
        '[A->B forecast] - [B->A forecast]; its coefficient on z = 1 - 2 x (cue-1 signal) is the '
        'aligned-minus-misaligned difference (aligned = transfer direction equals the cue-1 signal, '
        'which the delay law maps to delay 1 instead of 4). Signals are randomised, so the estimate is '
        'unbiased; pixel-decodable covariates only reduce variance; SEs are over the 6 seeds. Units: '
        'survival probability, mean of horizons 4/8/12.')
    add('')
    add('| Cell | mean +/- SE (6 seeds) | h4 | h8 | h12 | per-seed values |')
    add('|---|---:|---:|---:|---:|---|')
    cells = [
        ('True effect: law-known ideal, delay-first law', 'interleaved/delay_first/io'),
        ('True effect: evaluator truth, delay-first law', 'interleaved/delay_first/truth'),
        ('Law-known ideal, later delay-active laws', 'interleaved/delay_later/io'),
        ('Ideal with the same one-batch support, delay-first law', 'interleaved/delay_first/post'),
        ('Ideal with the same one-batch support, predecessor law', 'interleaved/predecessor/post'),
        ('No-context ideal (5-law mixture), delay-first law', 'interleaved/delay_first/mix'),
        ('Model, fresh delay fit (all chunks)', 'fresh/delay_fit/model'),
        ('Model, fresh delay fit, chunks 5-8', 'fresh/delay_fit/model/chunks_5_8'),
        ('Model, fresh delay fit, chunk 8', 'fresh/delay_fit/model/chunk_8'),
        ('Model, fresh delay fit, NON-delayed placebo', 'fresh/delay_fit/model/nondelayed_placebo'),
        ('Model, interleaved delay-first law', 'interleaved/delay_first/model'),
        ('Model, interleaved delay-first law, cycles 5-8', 'interleaved/delay_first/model/cycles_5_8'),
        ('Model, interleaved delay-first law, position 1', 'interleaved/delay_first/model/position1'),
        ('Model, interleaved delay-first law, NON-delayed placebo', 'interleaved/delay_first/model/nondelayed_placebo'),
        ('Model, interleaved later delay-active laws', 'interleaved/delay_later/model'),
        ('Model, interleaved predecessor law (delay inactive)', 'interleaved/predecessor/model'),
        ('Model, interleaved predecessor law, cycles 5-8', 'interleaved/predecessor/model/cycles_5_8'),
        ('Model, interleaved Base A (delay inactive)', 'interleaved/baseA/model'),
        ('Law-known ideal, predecessor law (should be 0)', 'interleaved/predecessor/io'),
        ('Law-known ideal, NON-delayed placebo (should be 0)', 'interleaved/delay_first/io/nondelayed_placebo'),
    ]
    for label, key in cells:
        v = pooled[key]
        add(f"| {label} | {_ms(v['horizon_mean'])} | " + ' | '.join(_f(h['mean']) for h in v['horizons'])
            + ' | ' + ', '.join(_f(x) for x in v['horizon_mean']['values']) + ' |')
    add('')
    ib_i = pooled['interleaved/delay_first/ideal_benefit']
    ib_f = pooled['fresh/delay_fit/ideal_benefit']
    end_ideal, end_model_f, end_model_i, eff_ideal, sup_ideal = [], [], [], [], []
    for s in seeds:
        e = report['endpoint'][str(s)]
        k = report['meta'][str(s)]['delay_stage']
        end_ideal.append(e[f'fresh_{k}'][0]['cues']['1']['ideal_flip'])
        end_model_f.append(e[f'fresh_{k}'][0]['cues']['1']['model_flip'])
        end_model_i.append(np.mean([t['cues']['1']['model_flip'] for t in e['interleaved'] if '1' in t['cues']]))
        eff_ideal.append(np.mean([t['cues']['0']['ideal_flip'] for t in e['interleaved'] if '0' in t['cues']]))
        sup_ideal.append(np.mean([t['cues']['2']['ideal_flip'] for t in e['interleaved'] if '2' in t['cues']]))
    add('Brier value of the delay cue (flip test = Brier with cue-1 pulse order flipped minus correct, '
        'on delayed cases; cue-blind = averaging the two forecasts):')
    add('')
    add('| Quantity | value |')
    add('|---|---:|')
    add(f"| Ideal flip benefit, online delayed cases, interleaved delay-first law | {_ms(ib_i['flip'])} |")
    add(f"| Ideal flip benefit, online delayed cases, fresh delay fit | {_ms(ib_f['flip'])} |")
    add(f"| Ideal cue-blind value (information value of cue 1) | {_ms(ib_i['blind'])} |")
    add(f"| Ideal flip benefit on the locked endpoint panel, fresh delay fit | {_ms(summarise(end_ideal))} |")
    add(f"| Model flip benefit on the same panel, fresh delay fit (locked scorer value .003365) | {_ms(summarise(end_model_f))} |")
    add(f"| Model flip benefit on the endpoint panels, interleaved delay-active laws | {_ms(summarise(end_model_i))} |")
    add(f"| For scale: ideal flip benefit, efficiency cue / supply-timing cue (interleaved endpoints) | {_f(np.mean(eff_ideal))} / {_f(np.mean(sup_ideal))} |")
    add('')
    dp = pooled['discriminability/delay_first_support/llr_delay_vs_predecessor']
    pp = pooled['discriminability/predecessor_support/llr_delay_vs_predecessor']
    add(f"**True size.** Where the delay law acts (delayed cases, transfer requests), aligned transfers "
        f"survive more often by {_f(pooled['interleaved/delay_first/io']['horizon_mean']['mean'], 3)} "
        f"(ideal) / {_f(pooled['interleaved/delay_first/truth']['horizon_mean']['mean'], 3)} (truth) in "
        f"probability, concentrated at horizon 4 ({_f(pooled['interleaved/delay_first/io']['horizons'][0]['mean'], 3)}) "
        f"and small at 8/12 ({_f(pooled['interleaved/delay_first/io']['horizons'][1]['mean'], 3)}, "
        f"{_f(pooled['interleaved/delay_first/io']['horizons'][2]['mean'], 3)}). In Brier units the best "
        f"possible flip benefit is about {_f(ib_i['flip']['mean'], 3)} (endpoint panels "
        f"{_f(np.mean(end_ideal), 3)}), roughly half the efficiency cue ({_f(np.mean(eff_ideal), 3)}) and "
        f"one-fourteenth of supply timing ({_f(np.mean(sup_ideal), 3)}). The locked .002 threshold is "
        f"{_f(.002 / np.mean(end_ideal) * 100, 0)}% of the ideal delay flip benefit; the fresh model reached "
        f"about {_f(np.mean(end_model_f) / np.mean(end_ideal) * 100, 0)}% of it.")
    add('')
    add(f"**Model.** Online, the fresh delay fit's aligned-minus-misaligned forecast difference grows to "
        f"{_f(pooled['fresh/delay_fit/model/chunks_5_8']['horizon_mean']['mean'], 4)} in chunks 5-8 "
        f"(model/ideal ratio over the whole fit {_ms(q3['model_over_io_fresh'], 2)}). In the interleaved "
        f"fit it is {_ms(pooled['interleaved/delay_first/model']['horizon_mean'])} in the delay-first law "
        f"(ratio {_ms(q3['model_over_io_interleaved'], 2)}; "
        f"{_f(pooled['interleaved/delay_first/model/cycles_5_8']['horizon_mean']['mean'], 4)} in cycles 5-8) "
        f"versus {_ms(pooled['interleaved/predecessor/model']['horizon_mean'])} in the delay-inactive "
        f"predecessor law. The delay-first minus predecessor difference is "
        f"{_ms(q3['delay_minus_predecessor_model'])} overall and "
        f"{_ms(q3['delay_minus_predecessor/model/cycles_5_8'])} in cycles 5-8, against a law-known "
        f"ideal difference of {_ms(q3['delay_minus_predecessor/io'])} and a same-support ideal difference "
        f"of {_ms(q3['delay_minus_predecessor/post'])}.")
    add('')
    add(f"**Specificity.** The model's cue-1 sensitivity is not specific to the cases the law affects: "
        f"on NON-delayed cases of the same delay-first blocks (where cue 1 is physically irrelevant; ideal "
        f"{_f(pooled['interleaved/delay_first/io/nondelayed_placebo']['horizon_mean']['mean'], 4)}, truth "
        f"{_f(pooled['interleaved/delay_first/truth/nondelayed_placebo']['horizon_mean']['mean'], 4)}) the model "
        f"shows {_ms(pooled['interleaved/delay_first/model/nondelayed_placebo']['horizon_mean'])}, so the "
        f"delayed-minus-non-delayed difference is {_ms(q3['delayed_minus_nondelayed/delay_first/model'])}. "
        f"Even the fresh delay fit has a large non-delayed placebo response "
        f"({_ms(pooled['fresh/delay_fit/model/nondelayed_placebo']['horizon_mean'])}; delayed minus "
        f"non-delayed {_ms(q3['delayed_minus_nondelayed/fresh_delay_fit/model'])}). The efficiency cue "
        f"(which passed its interleaved gate) shows the same pattern: model contrast on lossy cases "
        f"{_ms(pooled['control/efficiency_first/model']['horizon_mean'])} in its first active law vs "
        f"{_ms(pooled['control/efficiency_predecessor/model']['horizon_mean'])} in the inactive predecessor "
        f"(ideal {_f(pooled['control/efficiency_first/io']['horizon_mean']['mean'], 3)} vs "
        f"{_f(pooled['control/efficiency_predecessor/io']['horizon_mean']['mean'], 3)}).")
    add('')
    add(f"**Can the law be told apart?** For the ideal observer, one 32-record support distinguishes the "
        f"delay-first law from its predecessor only weakly: mean log-likelihood ratio "
        f"{_f(dp['mean']['mean'], 2)} nats for supports drawn from the delay law "
        f"({_f(dp['share_favouring_delay']['mean'] * 100, 0)}% favour it) and {_f(pp['mean']['mean'], 2)} "
        f"for supports from the predecessor ({_f((1 - pp['share_favouring_delay']['mean']) * 100, 0)}% "
        f"favour it), versus {_f(disc['mean']['mean'], 0)} nats for Base A vs Base B. These two laws are the "
        f"least separable pair in the stream, so a learner whose context is one batch can only "
        f"represent delay activity weakly through context and must otherwise carry it in weights that "
        f"alternate between delay-active and delay-inactive blocks.")
    add('')
    add(f"**Answer.** The delay dependency is real but small (about {_f(pooled['interleaved/delay_first/io']['horizon_mean']['mean'], 3)} "
        f"survival probability; best flip benefit about {_f(ib_i['flip']['mean'], 3)} Brier). Online, the "
        f"interleaved model's forecasts carry about {_f(q3['model_over_io_interleaved']['mean'] * 100, 0)}% of the true "
        f"aligned-minus-misaligned difference (fresh fit about {_f(q3['model_over_io_fresh']['mean'] * 100, 0)}%), and "
        f"almost all of that response is a generic pulse-1 direction bias that is also present in "
        f"delay-inactive laws and on non-delayed cases. There is at most a weak sign of law "
        f"discrimination (delay-first minus predecessor {_ms(q3['delay_minus_predecessor_model'])}, "
        f"{_ms(q3['delay_minus_predecessor/model/cycles_5_8'])} late), of the same order as the "
        f"non-delayed placebo difference ({_ms(q3['delay_minus_predecessor/model/nondelayed_placebo'])}); "
        f"nothing in these arrays shows the model conditioning its delay response on the law.")
    add('')
    # ---------------------------------------------------------------- Q4
    add('## Q4. Other observations: seeds, mode parity, cue order, replay')
    add('')
    add('| Seed | Base A mode | cue order | delay stage | endpoint Base A Brier | endpoint lambda | online Base A c5-8 | mean pos-2 lambda | Base A replay share outside Base A blocks | Base A packets in memory at visit starts (c1..c8) |')
    add('|---|---:|---|---:|---:|---:|---:|---:|---:|---|')
    for row in q4['rows']:
        add(f"| {row['seed']} | {row['baseA_mode']} | {'/'.join(row['order'])} | {row['delay_stage']} | "
            f"{_f(row['baseA_endpoint'])} | {_f(row['baseA_endpoint_lambda'], 2)} | {_f(row['baseA_online_cycles_5_8'])} | "
            f"{_f(np.mean(starts[str(row['seed'])]['lambda_position2'][1:]), 2)} | "
            f"{_f(row['replay_baseA_share_outside_baseA'], 3)} | {' '.join(map(str, row['memory_baseA_at_block_start']))} |")
    add('')
    vc = q4['visit_correlations']
    by_count = {}
    for v in q4['visits']:
        by_count.setdefault(min(v['memory_baseA_at_start'], 5) // 3, []).append(v)
    low, high = by_count.get(0, []), by_count.get(1, [])
    cc = q4['correlations']
    par = q4['parity']
    add(f"* **Pathological seeds.** 18104 and 18101 have the worst endpoint Base A Brier "
        f"({_f(q4['rows'][3]['baseA_endpoint'], 3)}, {_f(q4['rows'][0]['baseA_endpoint'], 3)}; locked "
        f"marginal gains were negative). Endpoint Brier is almost entirely explained by how far the "
        f"forecasts move toward mode A given the Base A support (endpoint lambda vs Brier across seeds "
        f"r = {_f(cc['endpoint_lambda~endpoint_brier'], 3)}). This switch quality is volatile visit to visit "
        f"within every seed (position-2 lambda ranges, cycles 2-8: "
        + '; '.join(f"{s} {_f(min(starts[str(s)]['lambda_position2'][1:]), 2)}-{_f(max(starts[str(s)]['lambda_position2'][1:]), 2)}" for s in seeds)
        + '), so the single endpoint draw exaggerates seed differences; online Base A cycles 5-8 Brier '
        f"spans only {_f(min(r['baseA_online_cycles_5_8'] for r in q4['rows']), 3)}-{_f(max(r['baseA_online_cycles_5_8'] for r in q4['rows']), 3)}.")
    add(f"* **Replay composition.** Across the 42 Base A visits of cycles 2-8, the number of Base A packets "
        f"in the 16-packet reservoir at visit start predicts the context switch: 0-2 packets "
        f"(n={len(low)}) give position-2 Brier {_f(np.mean([v['position2'] for v in low]), 3)} and lambda "
        f"{_f(np.mean([v['lambda_position2'] for v in low]), 2)}; 3 or more (n={len(high)}) give "
        f"{_f(np.mean([v['position2'] for v in high]), 3)} and {_f(np.mean([v['lambda_position2'] for v in high]), 2)} "
        f"(pooled r with position-2 Brier {_f(vc['memory_baseA_at_start~position2']['pooled'], 2)}, within-seed "
        f"{_f(vc['memory_baseA_at_start~position2']['within_seed'], 2)}; with lambda "
        f"{_f(vc['memory_baseA_at_start~lambda_position2']['pooled'], 2)} / "
        f"{_f(vc['memory_baseA_at_start~lambda_position2']['within_seed'], 2)}). Seed level (n=6, descriptive): "
        f"Base A share of replay draws outside Base A blocks ({_f(min(r['replay_baseA_share_outside_baseA'] for r in q4['rows']), 2)}-"
        f"{_f(max(r['replay_baseA_share_outside_baseA'] for r in q4['rows']), 2)}; 0.20 expected) correlates "
        f"r = {_f(cc['replay_baseA_share_outside_baseA~baseA_online_cycles_5_8'], 2)} with online Base A "
        f"Brier (cycles 5-8) and r = {_f(cc['replay_baseA_share_outside_baseA~baseA_endpoint'], 2)} with the "
        f"endpoint. Mode-switch packets (block starts whose support comes from the other mode) show no "
        f"visit-level relation (within-seed r {_f(vc['memory_mode_switch_at_start~position2']['within_seed'], 2)} "
        f"with position-2 Brier), although the seed-level share of all cross-law-support packets correlates "
        f"r = {_f(cc['replay_cross_law_support_share~baseA_online_cycles_5_8'], 2)} with online Base A (n=6, "
        f"driven by long-lived reservoir entries; not interpretable).")
    add(f"* **Mode parity.** Base A mode 0 (seeds {par['baseA_mode_0']['seeds']}) endpoint "
        f"{_ms(par['baseA_mode_0']['baseA_endpoint'], 3)}, online c5-8 {_ms(par['baseA_mode_0']['baseA_online_cycles_5_8'], 3)}; "
        f"mode 1 (seeds {par['baseA_mode_1']['seeds']}) endpoint {_ms(par['baseA_mode_1']['baseA_endpoint'], 3)}, "
        f"online {_ms(par['baseA_mode_1']['baseA_online_cycles_5_8'], 3)}. Three seeds per parity with "
        f"overlapping ranges: no evidence for a parity effect.")
    add('* **Cue order.** Each order occurs in exactly one seed, and the stage-3 law that immediately '
        'precedes every Base A block has all three cues regardless of order, so order can only act '
        'through the stage-1/2 laws; nothing in the per-seed table tracks order.')
    add(f"* **Stage laws are mutually confusable, even ideally.** With the same one-batch support the ideal "
        f"observer's posterior on the true stage law is only "
        + ', '.join(_f(q2['posterior'][f'stage{k}/positions_2_32']['true_law']['mean'], 2) for k in (1, 2, 3))
        + f" (mode posterior 1.00), yet its Brier stays near .045 because confusable laws forecast alike. "
        f"The no-context mixture costs mode-B laws only .07-.09 but Base A "
        f"{_f(np.mean([q1['table'][f'0_{c}']['mix']['mean'] for c in range(1, 9)]), 3)}: context is "
        f"indispensable only for the unique-mode law.")
    add('')
    add('## Caveats')
    add('')
    add('* POST HOC and EXPLORATORY; six seeds; correlations are descriptive; nothing here re-opens or '
        'relaxes the locked qualification, which stands as STOP.')
    add('* Ideal observers use evaluator-only knowledge (law set, simulator, noise model). They bound '
        'what the observations and the 32-record support contain; they are not candidate learners.')
    add('* No model was executed, so support and weight effects at position 2 are separated only '
        'indirectly (endpoint natural experiment; forgetting between visits). The position-2 forecast '
        'follows 12 optimizer steps on the first batch of the block.')
    add('* Monte Carlo error of the ideal observers is small (256 common-random-number draws per case; '
        'variance-corrected Brier) relative to every reported gap. Two independent full collections '
        'gave a byte-identical summary; one Monte Carlo chunk (seed 18102) differed at the level of a '
        'few draws, changing stored values by less than 1e-5 Brier.')
    add('')
    return '\n'.join(L) + '\n'


# --------------------------------------------------------------------------- #
# Command line                                                                 #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--draws', type=int, default=256)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seeds', type=int, nargs='*', default=None)
    parser.add_argument('--cache', default=None,
                        help='optional scratch directory OUTSIDE the repository for stage-1 arrays')
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    require(1 <= args.workers <= 6, 'use at most six worker processes')
    require(args.draws >= 32, 'at least 32 Monte Carlo draws required')
    cfg = config()
    seeds = cfg['seeds'] if args.seeds is None else args.seeds
    require(set(seeds) <= set(cfg['seeds']), 'seeds must belong to the locked qualification cohort')
    if args.cache is not None:
        require(ROOT not in Path(args.cache).resolve().parents, 'cache must be outside the repository')
    started = time.perf_counter()
    results = collect(seeds, args.draws, args.workers, args.cache)
    print(json.dumps(dict(collected=len(results), seconds=round(time.perf_counter() - started, 1))),
          flush=True)
    if args.collect_only:
        return
    analyse(results, seeds, args.draws, Path(args.output))


if __name__ == '__main__':
    main()
