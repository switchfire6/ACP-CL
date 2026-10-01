"""Step C: consolidated recall under recurring, unannounced laws.

Implements docs/consolidated_recall_protocol.md (prospective, written and revised
2026-09-29 before any fitting). Earlier studies' machinery is imported, never
edited: Step B's learner settings, oracle realised-supply verification and
save helpers (scripts/competence_gap.py), Step B2's integrity helpers
(scripts/competence_ceiling.py), the MC primitives (scripts/ideal_observer_mc.py),
the unchanged recurrent reference learner (RepresentationLearner 'outcome'),
the project's conditional learner (ContextLearner('conditional'), COND-4) and
the qualification's interleaved laws (reference_laws).

Stream: 8 cycles x (Base A, Base B, stage 1, stage 2, stage 3) x 1,024 arrivals
(1,280 batches of 32) from AcquisitionWorld.experience with seeds
trial_seed(seed, 'consolidated_recall_stream', batch). Every arm of a seed sees
the identical stream and (except COND-4, a different architecture) the same
initialisation (seed + 73019). Every forecast is made before the labels of its
batch are used; l(model, batch) is computed after the forecast and before
training (protocol steps 1-7; scripts/consolidated_recall_learners.py).

References (evaluator-only): O (law-known) and I (same-information sticky
filter over the five laws) from a 512-sample supply-noise Monte Carlo with
common random numbers across laws and actions, exact reserves, joint pattern
counts for all 5 actions. Development seeds compute only the true-law column
(= O = I's true-law column, identical noise); test seeds compute all 5 laws.

Usage (never writes bytecode):
  set PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src, then
  .venv/Scripts/python.exe -B scripts/consolidated_recall.py check     --config configs/consolidated_recall_smoke.json --output runs/consolidated_recall_smoke
  .venv/Scripts/python.exe -B scripts/consolidated_recall.py dev       --config configs/consolidated_recall_dev.json   --output runs/consolidated_recall_dev --report reports/consolidated_recall/dev
  .venv/Scripts/python.exe -B scripts/consolidated_recall.py lock      --config configs/consolidated_recall.json       --output runs/consolidated_recall
  .venv/Scripts/python.exe -B scripts/consolidated_recall.py run       --config configs/consolidated_recall.json       --output runs/consolidated_recall
  .venv/Scripts/python.exe -B scripts/consolidated_recall.py replicate --config configs/consolidated_recall.json       --output runs/consolidated_recall
  .venv/Scripts/python.exe -B scripts/consolidated_recall.py analyze   --config configs/consolidated_recall.json       --output runs/consolidated_recall --report reports/consolidated_recall
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import argparse  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
from dataclasses import asdict  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zipfile  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
try:  # pragma: no cover - environment dependent
    import acp_cl  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import competence_gap as G  # noqa: E402
import competence_ceiling as B2  # noqa: E402
import consolidated_recall_learners as L  # noqa: E402
import ideal_observer_mc as MC  # noqa: E402
from acp_cl.acquisition.world import AcquisitionWorld, Law  # noqa: E402
from acp_cl.contextual.learner import ContextLearner  # noqa: E402
from acp_cl.persistence.learner import state_hash  # noqa: E402
from acp_cl.persistence.study import digest, trial_seed, write_json as _write_json  # noqa: E402
from acp_cl.predictive_value.mechanism import _value_hash, predict_all, state_signature  # noqa: E402
from acp_cl.predictive_value.study import load as load_checkpoint, save as save_checkpoint  # noqa: E402
from acp_cl.representation_learning.design import reference_laws  # noqa: E402
from acp_cl.representation_learning.learner import RepresentationLearner, learner_signature  # noqa: E402

LABEL = ('PROSPECTIVE STUDY under docs/consolidated_recall_protocol.md (Step C). O and I are evaluator-only '
         'references; CR-oracle-mode and CR-oracle-law are PRIVILEGED arms (lagged true index).')
SCRIPT_FILES = ('scripts/consolidated_recall.py', 'scripts/consolidated_recall_learners.py',
                'scripts/competence_gap.py', 'scripts/competence_gap_learner.py',
                'scripts/competence_ceiling.py', 'scripts/competence_ceiling_learners.py',
                'scripts/ideal_observer_mc.py')
DEV_SEEDS = [20101, 20102, 20103]
TEST_SEEDS = list(range(20201, 20213))
SMOKE_SEEDS = [20199]
REPLICATE_SEEDS = (20201, 20206, 20211)
MC_CHANNEL, MC_CHECK_CHANNEL = 41, 42
CR_ARMS = ('CR', 'CR-K1', 'CR-no-recall', 'CR-oracle-mode', 'CR-oracle-law')
TEST_ARMS = ('ON-R256', 'ON-ref', 'EMA-global', 'CR-K1', 'CR', 'CR-no-recall', 'CR-oracle-mode', 'CR-oracle-law',
             'COND-4')
MECHANISM = dict(h=.05, K_max=8, pattern_floor=.01, recall_posterior=.95, spawn_run=2,
                 adam_first_moment_at_recall='reset', adam_second_moment_at_recall='keep')
LAW_NAMES = ('baseA', 'baseB', 'stage1', 'stage2', 'stage3')


class Interrupted(RuntimeError):
    """Deliberate interruption used only by the resumption checks."""


sha, read_json, now, require, save_npz, log_event = G.sha, G.read_json, G.now, G.require, G.save_npz, G.log_event


def jsonable(value):
    """Plain JSON types; non-finite floats become null (write_json forbids NaN)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_json(path, value):
    _write_json(path, jsonable(value))


# ------------------------------------------------------------------ config

def validate_config(c):
    expected = {'study', 'protocol', 'phase', 'seeds', 'stream', 'batch_size', 'eval_size', 'support_replicates',
                'context_width', 'experts', 'interaction_features', 'lr', 'evidence_strength', 'auxiliary_weight',
                'reference', 'cond4', 'mechanism', 'decay_grid', 'calibration', 'selected', 'development', 'oracle',
                'analysis', 'pathology', 'checkpoint_every_batches', 'stall_seconds', 'stage_attempts', 'device',
                'threads', 'workers', 'cpu_affinity'}
    if set(c) != expected:
        raise ValueError(f'invalid configuration fields: {set(c) ^ expected}')
    if c['phase'] not in ('smoke', 'dev', 'test'):
        raise ValueError('phase must be smoke, dev or test')
    fixed = dict(batch_size=32, eval_size=512, support_replicates=2, context_width=12, experts=4,
                 interaction_features=True, lr=.002, evidence_strength=1., auxiliary_weight=1.)
    if any(c[k] != v for k, v in fixed.items()):
        raise ValueError('reference learner settings are fixed by the protocol')
    if c['reference'] != dict(W=64, U=12, R=256, R_ref=16):
        raise ValueError('reference network settings are fixed by the protocol')
    if c['cond4'] != dict(method='conditional', width=64, experts=4, U=12, R=256, lr=.002, evidence_strength=1.):
        raise ValueError('COND-4 is the native conditional learner at U12/R256')
    if c['mechanism'] != MECHANISM:
        raise ValueError('fixed CR hyperparameters differ from the protocol')
    if c['decay_grid'] != [.99, .995, .998]:
        raise ValueError('the decay grid is fixed by the protocol')
    if c['device'] != 'cpu' or c['threads'] != 1 or not 1 <= c['workers'] <= 6:
        raise ValueError('CPU, one thread and at most six workers are required')
    s = c['stream']
    if set(s) != {'cycles', 'block_arrivals', 'channel'} or s['channel'] != 'consolidated_recall_stream':
        raise ValueError('invalid stream fields')
    if s['block_arrivals'] % c['batch_size']:
        raise ValueError('blocks must be whole batches')
    o = c['oracle']
    if set(o) != {'samples', 'sub', 'batches_per_task', 'laws', 'channel', 'io_h', 'io_floor', 'reserves'}:
        raise ValueError('invalid oracle fields')
    if (o['samples'] != 512 or o['io_h'] != .05 or o['io_floor'] != 1e-3 or o['channel'] != MC_CHANNEL
            or o['reserves'] != 'exact' or o['samples'] % o['sub']):
        raise ValueError('oracle settings are fixed by the protocol')
    if (s['block_arrivals'] // c['batch_size']) % o['batches_per_task']:
        raise ValueError('oracle tasks must tile each block (one law per task)')
    cal = c['calibration']
    if set(cal) != {'cycles', 'positions', 'percentile', 'method', 'statistic'} or cal['percentile'] != 99:
        raise ValueError('invalid calibration fields')
    a = c['analysis']
    if set(a) != {'excess_cycles', 'recovery_positions', 'plateau_positions', 'thresholds', 'consistency_seeds',
                  'bootstrap_resamples', 'analysis_seed'}:
        raise ValueError('invalid analysis fields')
    if (a['thresholds'] != dict(C1=.75, C2=.85, C3=.60) or a['consistency_seeds'] != 10
            or a['bootstrap_resamples'] != 20000):
        raise ValueError('pre-declared criteria are fixed')
    if c['pathology'] != dict(pending_fraction=.30, kstar_changes_per_mode_switch=3, max_snapshots_end=5):
        raise ValueError('the pathology stop rule is fixed')
    sel = c['selected']
    if set(sel) != {'CR', 'CR-K1', 'EMA-global'} or set(sel['EMA-global']) != {'d'} \
            or any(set(sel[k]) != {'d', 'tau'} for k in ('CR', 'CR-K1')):
        raise ValueError('invalid selected fields')
    if c['phase'] == 'smoke':
        if c['seeds'] != SMOKE_SEEDS or s != dict(cycles=2, block_arrivals=128, channel=s['channel']):
            raise ValueError('smoke: seed 20199 and 2 cycles x 5 laws x 128 arrivals')
        if any(v is None for arm in sel.values() for v in arm.values()):
            raise ValueError('smoke declares smoke-only d/tau values')
    else:
        if s['cycles'] != 8 or s['block_arrivals'] != 1024:
            raise ValueError('scientific stream is 8 cycles x 5 laws x 1024 arrivals')
        if cal['cycles'] != [1, 2] or cal['positions'] != [5, 32]:
            raise ValueError('tau calibration uses positions 5-32 of cycles 1-2')
        if a['excess_cycles'] != [3, 8] or a['recovery_positions'] != [2, 5] or a['plateau_positions'] != [17, 32]:
            raise ValueError('outcome windows are fixed by the protocol')
        if c['phase'] == 'dev':
            if c['seeds'] != DEV_SEEDS or o['laws'] != 'true':
                raise ValueError('development: seeds 20101-20103 and O (true-law column) only')
            if any(v is not None for arm in sel.values() for v in arm.values()):
                raise ValueError('development selects d and tau; they are not given')
        else:
            if c['seeds'] != TEST_SEEDS or o['laws'] != 'all':
                raise ValueError('test: seeds 20201-20212 with the full five-law I')


def require_selected(config):
    sel = config['selected']
    if any(v is None for arm in sel.values() for v in arm.values()):
        raise ValueError('d and tau placeholders are not filled; the test config cannot be locked or run')
    if config['phase'] == 'test':
        dev = config['development']
        require(dev.get('results_sha256') and sha(ROOT / dev['results']) == dev['results_sha256'],
                'development results missing or changed')
        results = read_json(ROOT / dev['results'])
        chosen = results['selection']
        require(chosen['CR']['d'] == sel['CR']['d'] and chosen['CR-K1']['d'] == sel['CR-K1']['d']
                and chosen['EMA-global']['d'] == sel['EMA-global']['d'], 'selected d differ from development')
        tau = results['tau']['values']
        require(tau[str(sel['CR']['d'])] == sel['CR']['tau'] and tau[str(sel['CR-K1']['d'])] == sel['CR-K1']['tau'],
                'tau differ from the development calibration')
        require(not results['pathology']['fired'], 'the development pathology stop rule fired')


# ----------------------------------------------------------------- stream

def block_batches(config):
    return config['stream']['block_arrivals'] // config['batch_size']


def schedule(config):
    B = block_batches(config)
    T = config['stream']['cycles'] * 5 * B
    t = np.arange(T)
    block = t // B
    return dict(batches=T, block_batches=B, law_index=block % 5, cycle=block // 5 + 1, position=t % B + 1,
                block=block)


def stream_batch(config, world, laws, seed, index):
    li = int((index // block_batches(config)) % 5)
    law = laws[li]
    data_seed = trial_seed(seed, config['stream']['channel'], index)
    data = world.experience(law, config['batch_size'], data_seed)
    cases = world.dataset(law, config['batch_size'], data_seed)
    truth = world.counterfactuals(cases)
    require(np.array_equal(cases.observations, data.observations), 'stream cases differ from experience')
    require(np.array_equal(truth[np.arange(len(data)), data.actions], data.survival),
            'performed outcome differs from physical outcome')
    return li, law, data, cases, truth


def mode_switches(config, seed):
    laws = reference_laws(seed)
    modes = np.array([laws[i].mode for i in schedule(config)['law_index']])
    return int((modes[1:] != modes[:-1]).sum()), modes


# ---------------------------------------------------------------- learners

def settings(config, R, U=None):
    return G.learner_settings(config, dict(W=config['reference']['W'], R=R, U=U or config['reference']['U']))


def cond4_settings(config):
    c = config['cond4']
    s = G.learner_settings(config, dict(W=c['width'], R=c['R'], U=c['U']))
    s.update(width=c['width'], experts=c['experts'], lr=c['lr'], evidence_strength=c['evidence_strength'])
    return s


def make_learner(config, seed, arm):
    if arm == 'COND-4':
        return ContextLearner(config['cond4']['method'], seed, cond4_settings(config))
    R = config['reference']['R_ref'] if arm == 'ON-ref' else config['reference']['R']
    return RepresentationLearner('outcome', seed, settings(config, R))


def signature(learner):
    if isinstance(learner, RepresentationLearner):
        return learner_signature(learner, ignore_walltime=True)
    return state_signature(learner, ignore_walltime=True)


def fit_spec(config, kind, arm, d=None, tau=None, decays=None):
    if kind == 'global':
        return dict(kind='global', arm='ON-R256', decays=[float(x) for x in decays])
    if kind == 'plain':
        require(arm in ('ON-ref', 'COND-4', 'ON-R256'), 'unknown plain arm')
        return dict(kind='plain', arm=arm)
    require(arm in L.VARIANTS, 'unknown CR arm')
    return dict(kind='cr', arm=arm, d=float(d), tau=float(tau), variant=dict(L.VARIANTS[arm]))


def fit_dir(output, seed, spec):
    base = Path(output) / 'fits'
    if spec['kind'] == 'global':
        return base / f'global_{seed}'
    if spec['kind'] == 'plain':
        return base / f'{spec["arm"]}_{seed}'
    return base / f'{spec["arm"]}_d{spec["d"]}_{seed}'


def verify_result(directory, identity):
    record = read_json(Path(directory) / 'result.json')
    if record['identity'] != identity:
        raise ValueError(f'result identity mismatch: {directory}')
    for name, value in record['artifact_hashes'].items():
        if sha(Path(directory) / name) != value:
            raise ValueError(f'artifact changed: {directory}/{name}')
    return record


def build_mechanism(config, spec, learner, state):
    if spec['kind'] == 'global':
        require(not learner.optimizer._optimizer_step_post_hooks, 'optimizer already carries hooks')
        return L.GlobalEMAs(learner, spec['decays'], state)
    if spec['kind'] == 'cr':
        require(not learner.optimizer._optimizer_step_post_hooks, 'optimizer already carries hooks')
        return L.ConsolidatedRecall(learner, spec['arm'], spec['d'], spec['tau'], config['mechanism'], state)
    return None


def mechanism_hash(mech):
    return None if mech is None else _value_hash(mech.state())


ARRAY_KEYS = dict(
    common=('probabilities', 'truth', 'actions', 'outcomes', 'law_index', 'ell_theta'),
    global_=('ema_probabilities', 'ell_ema'),
    cr=('theta_probabilities', 'ell_snap', 'snap_ids', 'posterior', 'source_id', 'kstar_forecast_id',
        'kstar_forecast_index', 'kstar_after', 'pending_after', 'recalled', 'copied', 'spawned', 'created',
        'replaced_id', 'K_after', 'exceed', 'oracle_index', 'consolidations'))


def run_fit(config, identity, seed, spec, output, interrupt_after=None, index_override=None, probe=None):
    """One prequential fit over the whole stream (resumable). ``index_override`` and ``probe`` are check-only."""
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = fit_dir(output, seed, spec)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = directory / 'result.json', directory / 'checkpoint.pt'
    if result_path.exists():
        verify_result(directory, identity)
        return dict(kind='fit', seed=seed, arm=spec['arm'], d=spec.get('d'), status='verified_existing')
    sched = schedule(config)
    T, size = sched['batches'], config['batch_size']
    laws, world = reference_laws(seed), AcquisitionWorld()
    floor = config['mechanism']['pattern_floor']
    kind, arm = spec['kind'], spec['arm']
    shadow = make_learner(config, seed, 'ON-R256') if kind in ('global', 'cr') else None
    resumed = False
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, identity)
        learner, record, arrays = saved['learner'], saved['record'], saved['arrays']
        require(record['seed'] == seed and record['spec'] == spec, 'checkpoint spec changed')
        require(signature(learner) == record['checkpoint_signature'], 'checkpoint learner changed')
        mech = build_mechanism(config, spec, learner, saved['mechanism'])
        require(mechanism_hash(mech) == record['checkpoint_mechanism_sha256'], 'checkpoint mechanism changed')
        record['resumes'].append(dict(utc=now(), batches_done=record['batches_done']))
        resumed = True
    else:
        learner = make_learner(config, seed, 'COND-4' if arm == 'COND-4' else ('ON-ref' if arm == 'ON-ref' else 'ON-R256'))
        if shadow is not None:
            require(state_hash(shadow.model.state_dict()) == state_hash(learner.model.state_dict()),
                    'shadow does not share the initialisation')
        mech = build_mechanism(config, spec, learner, None)
        record = dict(identity=identity, label=LABEL, seed=seed, spec=spec, settings=learner.settings,
                      privileged=arm in L.PRIVILEGED, privileged_label=L.PRIVILEGED.get(arm),
                      laws=[asdict(x) for x in laws], law_names=list(LAW_NAMES),
                      learner_class=f'{type(learner).__module__}.{type(learner).__qualname__}',
                      initial_model_sha256=state_hash(learner.model.state_dict()),
                      initial_signature=signature(learner), batches_done=0, fingerprints=[], resumes=[],
                      elapsed_seconds=0., started_utc=now(), forecast_forwards=0,
                      index_override=None if index_override is None else {str(k): v for k, v in index_override.items()})
        keys = ARRAY_KEYS['common'] + (ARRAY_KEYS['global_'] if kind == 'global' else ()) + \
            (ARRAY_KEYS['cr'] if kind == 'cr' else ())
        arrays = {k: [] for k in keys}
    K_max = spec['variant']['K_max'] if kind == 'cr' else 0
    started = time.perf_counter()
    for t in range(record['batches_done'], T):
        forwards = 0
        li, law, data, cases, truth = stream_batch(config, world, laws, seed, t)
        support = learner.history
        # ---- step 1: forecasts (observations and past-only support only)
        p_theta = predict_all(learner, data.observations, support)
        forwards += 1
        forecast = p_theta
        if kind == 'global':
            p_ema = []
            for e in mech.emas:
                L.load_weights(shadow.model, e.weights())
                p_ema.append(predict_all(shadow, data.observations, support))
                forwards += 1
        elif kind == 'cr':
            source = mech.forecast_source()
            kstar_forecast_id = mech.snapshots[mech.kstar]['id']
            kstar_forecast_index = mech.snapshots[mech.kstar]['index']
            ids_now = mech.ids()
            p_snap = []
            for w in mech.snapshot_weights():
                L.load_weights(shadow.model, w)
                p_snap.append(predict_all(shadow, data.observations, support))
                forwards += 1
            forecast = p_theta if source == -1 else p_snap[source]
        # ---- step 2: labels are used only from here on
        ell_theta = L.pattern_loglik(p_theta, data.actions, data.survival, floor)
        if kind == 'global':
            ell_ema = [L.pattern_loglik(p, data.actions, data.survival, floor) for p in p_ema]
            learner.train(data)
        elif kind == 'cr':
            ell_snap = [L.pattern_loglik(p, data.actions, data.survival, floor) for p in p_snap]
            index = None
            if spec['variant']['index'] != 'inferred':
                index = int(law.mode) if spec['variant']['index'] == 'mode' else li
                if index_override is not None and t in index_override:
                    index = int(index_override[t])
            if probe is not None:
                probe('before_observe', t, learner, mech)
            steps_before = mech.counters['consolidations']
            event = mech.observe(t, ell_theta, ell_snap, index)
            if probe is not None:
                probe('after_observe', t, learner, mech, event)
            mech.begin_training(law.mode, li)
            learner.train(data)
            mech.end_training()
        else:
            learner.train(data)
        # ---- record
        record['forecast_forwards'] += forwards
        record['fingerprints'].append(data.fingerprint())
        arrays['probabilities'].append(np.array(forecast, dtype=np.float32, copy=True))
        arrays['truth'].append(truth.astype(np.uint8))
        arrays['actions'].append(data.actions.copy())
        arrays['outcomes'].append(data.survival.copy())
        arrays['law_index'].append(li)
        arrays['ell_theta'].append(ell_theta)
        if kind == 'global':
            arrays['ema_probabilities'].append(np.stack(p_ema).astype(np.float32))
            arrays['ell_ema'].append(ell_ema)
        elif kind == 'cr':
            pad = lambda values, fill: list(values) + [fill] * (K_max - len(values))  # noqa: E731
            arrays['theta_probabilities'].append(np.array(p_theta, dtype=np.float32, copy=True))
            arrays['ell_snap'].append(pad(ell_snap, np.nan))
            arrays['snap_ids'].append(pad(ids_now, -1))
            arrays['posterior'].append(pad(mech.posterior.tolist() if spec['variant']['index'] == 'inferred'
                                           else [], np.nan))
            arrays['source_id'].append(-1 if source == -1 else ids_now[source])
            arrays['kstar_forecast_id'].append(kstar_forecast_id)
            arrays['kstar_forecast_index'].append(-1 if kstar_forecast_index is None else kstar_forecast_index)
            arrays['kstar_after'].append(event['kstar_after'])
            arrays['pending_after'].append(event['pending_after'])
            arrays['recalled'].append(event['recalled'])
            arrays['copied'].append(event['copied'])
            arrays['spawned'].append(event['spawned'])
            arrays['created'].append(event['created'])
            arrays['replaced_id'].append(-1 if event['replaced_id'] is None else event['replaced_id'])
            arrays['K_after'].append(event['K_after'])
            arrays['exceed'].append(event['exceed'])
            arrays['oracle_index'].append(-1 if index is None else index)
            arrays['consolidations'].append(mech.counters['consolidations'] - steps_before)
        record['batches_done'] = t + 1
        if (t + 1) % config['checkpoint_every_batches'] == 0 or t + 1 == T:
            record['elapsed_seconds'] += time.perf_counter() - started
            started = time.perf_counter()
            record['checkpoint_signature'] = signature(learner)
            record['checkpoint_mechanism_sha256'] = mechanism_hash(mech)
            save_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record, arrays=arrays,
                                             mechanism=None if mech is None else mech.state()))
        if interrupt_after is not None and t + 1 == interrupt_after:
            raise Interrupted(f'deliberate interruption after batch {t + 1}')
    if mech is not None:
        mech.remove()
    require(learner.memory.seen == T and len(learner.memory.packets) == min(learner.memory.capacity, T),
            'replay capacity not honoured')
    diagnostics = learner.diagnostics()
    record['diagnostics'] = diagnostics
    record['final_model_sha256'] = state_hash(learner.model.state_dict())
    record['end_signature'] = signature(learner)
    record['optimizer_sha256'] = _value_hash(learner.optimizer.state_dict())
    out = {}
    n = T * size
    out['probabilities'] = np.concatenate(arrays['probabilities'])
    out['truth'] = np.concatenate(arrays['truth'])
    out['actions'] = np.concatenate(arrays['actions'])
    out['outcomes'] = np.concatenate(arrays['outcomes'])
    out['law_index'] = np.asarray(arrays['law_index'], dtype=np.int8)
    out['ell_theta'] = np.asarray(arrays['ell_theta'], dtype=np.float64)
    require(out['probabilities'].shape == (n, 5, 3), 'forecast array shape')
    if kind == 'global':
        out['ema_probabilities'] = np.stack(arrays['ema_probabilities'], axis=1).reshape(len(spec['decays']), n, 5, 3)
        out['ell_ema'] = np.asarray(arrays['ell_ema'], dtype=np.float64)
        out['decays'] = np.asarray(spec['decays'])
        record['ema'] = [dict(decay=e.decay, count=e.count, steps=e.steps,
                              model_sha256=state_hash(dict(zip([k for k, _ in learner.model.named_parameters()],
                                                                e.weights()))))
                         for e in mech.emas]
        require(all(e.steps == diagnostics['cost']['optimizer_steps'] for e in mech.emas), 'EMA missed a step')
    elif kind == 'cr':
        out['theta_probabilities'] = np.concatenate(arrays['theta_probabilities'])
        for key in ('ell_snap', 'posterior'):
            out[key] = np.asarray(arrays[key], dtype=np.float64)
        for key in ('snap_ids', 'source_id', 'kstar_forecast_id', 'kstar_forecast_index', 'kstar_after',
                    'replaced_id', 'K_after', 'oracle_index', 'consolidations'):
            out[key] = np.asarray(arrays[key], dtype=np.int64)
        for key in ('pending_after', 'recalled', 'copied', 'spawned', 'created', 'exceed'):
            out[key] = np.asarray(arrays[key], dtype=bool)
        record['cr'] = dict(counters=mech.counters, snapshots=mech.summary(T), final_K=len(mech.snapshots),
                            snapshot_bytes=mech.nbytes(), final_posterior=mech.posterior.tolist(),
                            final_kstar_id=mech.snapshots[mech.kstar]['id'],
                            snapshot_model_sha256={str(s['id']): state_hash(dict(zip(
                                [k for k, _ in learner.model.named_parameters()], s['ema'].weights())))
                                for s in mech.snapshots})
        require(mech.counters['consolidations'] == int(out['consolidations'].sum()), 'consolidation count')
    save_npz(directory / 'training.npz', out)
    packet = G.forward_macs(config['reference']['W'], config['context_width'], size)
    steps = diagnostics['cost']['optimizer_steps']
    params = diagnostics['parameters']
    record['work'] = dict(packet_forward_macs=packet, training_macs=3 * 2 * packet * learner.settings['updates_per_batch'] * T,
                          forecast_and_likelihood_macs=packet * record['forecast_forwards'],
                          average_update_flops=3 * params * (mech.counters['consolidations'] if kind == 'cr' else
                                                             steps * len(spec['decays']) if kind == 'global' else 0),
                          note='COND-4 uses the same per-packet proxy (Hypotheses encoder + 4 heads, approx.)')
    record['work']['total_macs'] = (record['work']['training_macs'] + record['work']['forecast_and_likelihood_macs']
                                    + record['work']['average_update_flops'] // 2)
    extra = 0
    if kind == 'global':
        extra = sum(e.nbytes() for e in mech.emas)
    elif kind == 'cr':
        extra = mech.nbytes()
    record['memory_bytes'] = dict(model=diagnostics['model_bytes'], optimizer=diagnostics['optimizer_bytes'],
                                  peak_replay=diagnostics['peak_replay_bytes'], history=diagnostics['history_bytes'],
                                  averages=extra, total=diagnostics['model_bytes'] + diagnostics['optimizer_bytes']
                                  + diagnostics['peak_replay_bytes'] + diagnostics['history_bytes'] + extra)
    record['parameters'] = params
    record['optimizer_steps'] = steps
    record['training_seconds'] = diagnostics['cost']['training_seconds']
    record['resumed'] = resumed
    record['elapsed_seconds'] += time.perf_counter() - started
    record['completed_utc'] = now()
    record['checkpoint_signature'] = signature(learner)
    record['checkpoint_mechanism_sha256'] = mechanism_hash(mech)
    save_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record, arrays=arrays,
                                     mechanism=None if mech is None else mech.state()))
    record['artifact_hashes'] = {'training.npz': sha(directory / 'training.npz'), 'checkpoint.pt': sha(checkpoint)}
    write_json(result_path, record)
    return dict(kind='fit', seed=seed, arm=arm, d=spec.get('d'), status='completed', seconds=record['elapsed_seconds'])


# ------------------------------------------------------------ the oracle

def pattern_counts(reserves, factors, signals, sup0, laws, dedup=True):
    """Joint pattern counts (n, len(laws), 5 actions, 4 patterns) from mode-0 supplies sup0 (S, n, 12, 2).
    The same sup0 (common random numbers) is used for every law and action; laws that share the mode and
    the cues effective for a row type and action have identical physics and are simulated once (dedup)."""
    S, n = sup0.shape[:2]
    counts = np.zeros((n, len(laws), 5, 4), np.int64)
    types = MC.row_types(factors)
    for row_type in range(4):
        rows = np.flatnonzero(types == row_type)
        if not len(rows):
            continue
        tiled = [MC.tile_rows(v[rows], S) for v in (reserves, factors, signals)]
        for action in range(5):
            cache = {}
            for slot, law in enumerate(laws):
                cues = MC.effective(law.active, row_type, action) if dedup else tuple(law.active)
                key = (law.mode, cues)
                if key not in cache:
                    supplies = MC.mode_frame(sup0[:, rows], law.mode).reshape(S * len(rows), MC.STEPS, 2)
                    out = MC.simulate(*tiled, supplies, Law(law.mode, cues), np.full(S * len(rows), action))
                    pattern = observed_pattern_array(out.survival).reshape(S, len(rows))
                    cache[key] = np.stack([(pattern == k).sum(axis=0) for k in range(4)], axis=1)
                counts[rows, slot, action] += cache[key]
    return counts


def observed_pattern_array(survival):
    s = np.asarray(survival).astype(np.int64)
    require(not np.any(np.diff(s, axis=1) > 0), 'non-monotone simulated survival')
    return s.sum(axis=1)


def oracle_task(task):
    """MC over the hidden supply noise for one block-aligned run of batches (exact reserves)."""
    path = Path(task['path'])
    if path.exists():
        return dict(kind='oracle', seed=task['seed'], status='existing')
    config, seed, first, count = task['config'], task['seed'], task['first_batch'], task['batches']
    o = config['oracle']
    laws_all, world = reference_laws(seed), AcquisitionWorld()
    started = time.perf_counter()
    parts = [stream_batch(config, world, laws_all, seed, i) for i in range(first, first + count)]
    law_ids = {li for li, _, _, _, _ in parts}
    require(len(law_ids) == 1, 'oracle task spans laws')
    true_li = law_ids.pop()
    verified = sum(G.verify_realised(c, law, t, d.actions, d.survival) for _, law, d, c, t in parts)
    reserves = np.concatenate([c.base.reserves for _, _, _, c, _ in parts])
    factors = np.concatenate([c.factors for _, _, _, c, _ in parts])
    signals = np.concatenate([c.signals for _, _, _, c, _ in parts])
    truth = np.concatenate([t for _, _, _, _, t in parts])
    columns = list(range(5)) if task['laws'] == 'all' else [true_li]
    laws = [laws_all[i] for i in columns]
    n, samples, sub = len(reserves), o['samples'], o['sub']
    counts = np.zeros((n, len(laws), 5, 4), np.int64)
    means = MC.supply_means(factors)
    for part in range(samples // sub):
        rng = np.random.default_rng([MC.MC_ROOT, task.get('channel', o['channel']), seed, first, part])
        noise = rng.normal(0., MC.TransferWorld.supply_noise, (sub, n, MC.STEPS, 2))
        sup0 = np.clip(means[None] + noise, 0, None)
        counts += pattern_counts(reserves, factors, signals, sup0, laws, dedup=task.get('dedup', True))
    require(np.all(counts.sum(axis=-1) == samples), 'pattern counts do not sum to the sample count')
    save_npz(path, dict(counts=counts.astype(np.uint16), columns=np.asarray(columns, np.int8),
                        true_law=np.array(true_li), samples=np.array(samples), truth=truth.astype(np.uint8),
                        actions=np.concatenate([d.actions for _, _, d, _, _ in parts]),
                        outcomes=np.concatenate([d.survival for _, _, d, _, _ in parts]),
                        factors=factors, signals=signals, reserves=reserves,
                        fingerprints=np.array([d.fingerprint() for _, _, d, _, _ in parts]),
                        verified_cells=np.array(verified), seconds=np.array(time.perf_counter() - started)))
    return dict(kind='oracle', seed=seed, status='completed', seconds=time.perf_counter() - started)


def oracle_jobs(config, output, seeds=None):
    jobs, o = [], config['oracle']
    for seed in seeds or config['seeds']:
        for first in range(0, schedule(config)['batches'], o['batches_per_task']):
            jobs.append(('oracle', dict(config=config, seed=seed, first_batch=first, batches=o['batches_per_task'],
                                        laws=o['laws'],
                                        path=str(Path(output) / 'oracle' / 'tasks' / f'{seed}_{first:05d}.npz'))))
    return jobs


def assemble_oracle(config, output, seeds=None):
    o, summary = config['oracle'], {}
    T = schedule(config)['batches']
    for seed in seeds or config['seeds']:
        path = Path(output) / 'oracle' / f'seed_{seed}.npz'
        pieces = []
        seconds = 0.
        for first in range(0, T, o['batches_per_task']):
            with np.load(Path(output) / 'oracle' / 'tasks' / f'{seed}_{first:05d}.npz') as z:
                pieces.append({k: z[k] for k in z.files})
                require(int(z['samples']) == o['samples'], 'oracle sample count changed')
                seconds += float(z['seconds'])
        law_index = np.concatenate([np.full(len(p['truth']), int(p['true_law'])) for p in pieces])
        arrays = dict(counts=np.concatenate([p['counts'] for p in pieces]),
                      columns=pieces[0]['columns'] if o['laws'] == 'all' else np.array([-1], np.int8),
                      law_index=law_index.astype(np.int8),
                      **{k: np.concatenate([p[k] for p in pieces]) for k in
                         ('truth', 'actions', 'outcomes', 'factors', 'signals', 'reserves', 'fingerprints')})
        if o['laws'] == 'all':
            require(all(np.array_equal(p['columns'], np.arange(5)) for p in pieces), 'all-law columns')
        digest_ = save_npz(path, arrays)
        summary[str(seed)] = dict(path=path.relative_to(Path(output)).as_posix(), sha256=digest_,
                                  task_seconds=seconds, laws=o['laws'])
    return summary


def survival_from_counts(counts, samples):
    c = np.asarray(counts, dtype=np.float64)
    return np.stack([c[..., 1:].sum(-1), c[..., 2:].sum(-1), c[..., 3]], axis=-1) / samples


def oracle_probabilities(orc, config):
    """O: the true-law column."""
    counts = orc['counts']
    if config['oracle']['laws'] == 'all':
        counts = counts[np.arange(len(counts)), orc['law_index'].astype(np.int64)]
    else:
        counts = counts[:, 0]
    return survival_from_counts(counts, config['oracle']['samples'])


def ideal_observer(orc, config):
    """I: sticky filter over the five laws (h, floored pattern likelihood); forecast for batch t is the
    posterior-weighted mixture after batch t-1 (uniform before batch 1)."""
    o = config['oracle']
    require(o['laws'] == 'all', 'I needs all five laws')
    size = config['batch_size']
    counts = orc['counts'].astype(np.float64)                     # (N, 5, 5, 4)
    N = len(counts)
    T = N // size
    q = np.maximum(counts / o['samples'], o['io_floor'])
    q = q / q.sum(axis=-1, keepdims=True)
    pattern = orc['outcomes'].astype(np.int64).sum(axis=1)
    actions = orc['actions'].astype(np.int64)
    rows = np.arange(N)
    like = np.log(q[rows[:, None], np.arange(5)[None], actions[:, None], pattern[:, None]])  # (N, 5)
    ell = like.reshape(T, size, 5).sum(axis=1)
    P = survival_from_counts(counts, o['samples'])                 # (N, 5 laws, 5, 3)
    posterior = np.full(5, .2)
    weights = np.zeros((T, 5))
    after = np.zeros((T, 5))
    for t in range(T):
        weights[t] = posterior
        posterior = L.filter_step(posterior, ell[t], o['io_h'])
        after[t] = posterior
    forecast = np.einsum('nl,nlah->nah', np.repeat(weights, size, axis=0), P)
    return forecast, dict(ell=ell, weights=weights, posterior=after)


# --------------------------------------------------------- lock/manifest

def lock_payload(config, config_path):
    protocol = ROOT / config['protocol']
    scripts = {name: sha(ROOT / name) for name in SCRIPT_FILES}
    source = G.source_manifest()
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   platform=platform.platform(), device=config['device'], threads=config['threads'],
                   workers=config['workers'], deterministic=True, cpu_affinity=config['cpu_affinity'])
    identity = dict(config_sha256=digest(config), config_file_sha256=sha(config_path),
                    protocol_sha256=sha(protocol), scripts_sha256=digest(scripts),
                    source_sha256=digest(source), runtime_sha256=digest(runtime))
    return dict(identity=identity, config=copy.deepcopy(config),
                config_path=Path(config_path).resolve().relative_to(ROOT).as_posix(),
                protocol_path=config['protocol'], script_files=scripts, source_files=source, runtime=runtime)


def lock(config, config_path, output, kind='test'):
    validate_config(config)
    if kind == 'test':
        require_selected(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = lock_payload(config, config_path)
    path = output / 'manifest.json'
    if path.exists():
        old = read_json(path)
        if old['identity'] != payload['identity']:
            changed = [k for k in payload['identity'] if payload['identity'][k] != old['identity'][k]]
            raise ValueError(f'an existing manifest locks different files: {changed}')
        return old
    manifest = dict(label=LABEL, kind=kind, **payload, locked_utc=now())
    if kind == 'test' and config['phase'] == 'test':
        manifest['development'] = dict(config['development'])
    (output / 'protocol_at_lock.md').write_bytes((ROOT / config['protocol']).read_bytes())
    (output / 'config_at_lock.json').write_bytes(Path(config_path).read_bytes())
    with zipfile.ZipFile(output / 'source_at_lock.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(payload['script_files']) | set(payload['source_files'])):
            archive.write(ROOT / name, name)
    manifest['source_zip_sha256'] = sha(output / 'source_at_lock.zip')
    write_json(path, manifest)
    log_event(output, event='locked', kind=kind, identity=payload['identity'])
    return manifest


def check_lock(config, config_path, output):
    manifest = read_json(Path(output) / 'manifest.json')
    live = lock_payload(config, config_path)
    if manifest['identity'] != live['identity']:
        changed = [k for k in live['identity'] if live['identity'][k] != manifest['identity'][k]]
        raise ValueError(f'live files differ from the lock: {changed}')
    if sha(Path(output) / 'protocol_at_lock.md') != manifest['identity']['protocol_sha256']:
        raise ValueError('archived protocol changed')
    if sha(Path(output) / 'source_at_lock.zip') != manifest['source_zip_sha256']:
        raise ValueError('archived source zip changed')
    return manifest


# ------------------------------------------------------------------ pools

def estimated_cost(config, kind, payload):
    T = schedule(config)['batches']
    if kind == 'fit':
        spec = payload['spec']
        base = T * .22
        if spec['kind'] == 'cr':
            base *= 1.15
        if spec['kind'] == 'global':
            base *= 1. + .03 * len(spec['decays'])
        return base
    return payload['batches'] * config['batch_size'] * config['oracle']['samples'] * 5 * \
        (3 if payload['laws'] == 'all' else 1) / 6e5


def dispatch(kind, payload):
    if kind == 'fit':
        return run_fit(**payload)
    return oracle_task(payload)


def job_label(kind, payload):
    if kind == 'fit':
        s = payload['spec']
        return dict(kind='fit', seed=payload['seed'], arm=s['arm'], fit_kind=s['kind'], d=s.get('d'))
    return dict(kind='oracle', seed=payload['seed'], first_batch=payload['first_batch'], laws=payload['laws'])


def kill_pool(pool):
    processes = list(getattr(pool, '_processes', {}).values())
    for p in processes:
        try:
            p.kill()
        except Exception:  # pragma: no cover
            pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:  # pragma: no cover
        pass
    return len(processes)


def run_pool(config, output, jobs, stage):
    """Run jobs on at most config['workers'] single-thread processes. A broken or stalled pool is killed,
    every failure is logged, and the stage is retried with a fresh pool (finished tasks only verify)."""
    jobs = sorted(jobs, key=lambda job: -estimated_cost(config, *job))
    statuses = []
    for attempt in range(1, config['stage_attempts'] + 1):
        log_event(output, event='stage_start', stage=stage, attempt=attempt, jobs=len(jobs))
        started, failures, statuses = time.perf_counter(), [], []
        pool = ProcessPoolExecutor(max_workers=config['workers'], initializer=G.worker_init,
                                   initargs=(config['cpu_affinity'], config['threads']))
        futures = {pool.submit(dispatch, kind, payload): (kind, payload) for kind, payload in jobs}
        pending = set(futures)
        stalled = False
        while pending:
            done, pending = wait(pending, timeout=config['stall_seconds'], return_when=FIRST_COMPLETED)
            if not done:
                stalled = True
                killed = kill_pool(pool)
                log_event(output, event='pool_stalled', stage=stage, attempt=attempt, pending=len(pending),
                          killed_processes=killed, stall_seconds=config['stall_seconds'])
                for future in pending:
                    kind, payload = futures[future]
                    failures.append(dict(**job_label(kind, payload), error='pool stalled; killed'))
                    log_event(output, event='task_failed', stage=stage, attempt=attempt, **job_label(kind, payload),
                              error='pool stalled; killed')
                break
            for future in done:
                kind, payload = futures[future]
                label = job_label(kind, payload)
                try:
                    result = future.result()
                    statuses.append(dict(label, status=result.get('status')))
                    log_event(output, event='task_done', stage=stage, attempt=attempt, **label,
                              status=result.get('status'), seconds=result.get('seconds'))
                    if kind == 'fit':
                        print(json.dumps(dict(stage=stage, **label, status=result.get('status'),
                                              seconds=round(result.get('seconds') or 0, 1),
                                              elapsed=round(time.perf_counter() - started))), flush=True)
                except Exception as error:  # every failed attempt is kept and reported
                    failures.append(dict(**label, error=repr(error)))
                    log_event(output, event='task_failed', stage=stage, attempt=attempt, **label, error=repr(error),
                              traceback=traceback.format_exc())
                    print(json.dumps(dict(failed=label, error=repr(error))), flush=True)
        if not stalled:
            pool.shutdown(wait=True)
        if not failures:
            log_event(output, event='stage_complete', stage=stage, attempt=attempt,
                      seconds=time.perf_counter() - started)
            return statuses
        log_event(output, event='stage_attempt_failed', stage=stage, attempt=attempt, failures=len(failures))
    raise RuntimeError(f'stage {stage}: tasks failed after {config["stage_attempts"]} attempts; rerun to resume')


def fit_job(config, identity, output, seed, spec):
    return ('fit', dict(config=config, identity=identity, seed=seed, spec=spec, output=str(output)))


# ------------------------------------------------------------- integrity

def crc_scan(output, skip=('superseded', 'quarantine')):
    """CRC-check every zip archive (npz, torch .pt, zip) under output."""
    rows, bad = 0, []
    for path in sorted(Path(output).rglob('*')):
        if not path.is_file() or path.suffix not in ('.npz', '.pt', '.zip'):
            continue
        if any(part in skip for part in path.relative_to(output).parts):
            continue
        rows += 1
        try:
            with zipfile.ZipFile(path) as archive:
                first = archive.testzip()
            if first is not None:
                bad.append(dict(path=path.relative_to(output).as_posix(), member=first))
        except Exception as error:
            bad.append(dict(path=path.relative_to(output).as_posix(), error=repr(error)))
    return dict(archives=rows, failures=bad, passed=not bad, utc=now())


def npz_equal(a, b):
    """Array-by-array equality; float arrays compared bitwise (NaN padding compares equal to itself)."""
    with np.load(a) as x, np.load(b) as y:
        if set(x.files) != set(y.files):
            return False
        for k in x.files:
            u, v = x[k], y[k]
            if u.dtype != v.dtype or u.shape != v.shape:
                return False
            if u.dtype.kind == 'f':
                if u.tobytes() != v.tobytes():
                    return False
            elif not np.array_equal(u, v):
                return False
        return True


def verify_all_results(output, identity):
    count = 0
    for result in sorted((Path(output) / 'fits').glob('*/result.json')):
        verify_result(result.parent, identity)
        count += 1
    return count


# ------------------------------------------------------------- analysis

def seed_frames(config, output, seed, fits):
    """Case-Brier per batch for O (and I when available) and every requested forecast stream."""
    with np.load(Path(output) / 'oracle' / f'seed_{seed}.npz') as z:
        orc = {k: z[k] for k in z.files}
    sched = schedule(config)
    size, T = config['batch_size'], sched['batches']
    truth = orc['truth']
    P_O = oracle_probabilities(orc, config)
    o_case = G.case_brier(P_O, truth)
    frame = dict(O=o_case.reshape(T, size).mean(axis=1), O_case=o_case, P_O=P_O, truth=truth,
                 mc_variance_bias=float((P_O * (1 - P_O)).mean() / config['oracle']['samples']))
    if config['oracle']['laws'] == 'all':
        P_I, info = ideal_observer(orc, config)
        i_case = G.case_brier(P_I, truth)
        frame.update(I=i_case.reshape(T, size).mean(axis=1), I_case=i_case, P_I=P_I, I_info=info)
    streams = {}
    for name, (directory, key, index) in fits.items():
        with np.load(Path(directory) / 'training.npz') as z:
            require(np.array_equal(z['truth'], truth), f'truth differs between the oracle and {name}')
            p = z[key] if index is None else z[key][index]
        case = G.case_brier(p, truth)
        streams[name] = dict(batch=case.reshape(T, size).mean(axis=1),
                             survival=G.first_argmax_survival(p, truth).reshape(T, size).mean(axis=1))
    frame['streams'] = streams
    frame['O_survival'] = G.first_argmax_survival(P_O, truth).reshape(T, size).mean(axis=1)
    if 'P_I' in frame:
        frame['I_survival'] = G.first_argmax_survival(frame['P_I'], truth).reshape(T, size).mean(axis=1)
    return frame


def windows(config):
    s, a = schedule(config), config['analysis']
    c0, c1 = a['excess_cycles']
    in_cycles = (s['cycle'] >= c0) & (s['cycle'] <= c1)
    r0, r1 = a['recovery_positions']
    p0, p1 = a['plateau_positions']
    return dict(stream=in_cycles,
                recovery=in_cycles & (s['law_index'] == 0) & (s['position'] >= r0) & (s['position'] <= r1),
                plateau=in_cycles & (s['position'] >= p0) & (s['position'] <= p1),
                base_a=in_cycles & (s['law_index'] == 0))


def stream_metrics(config, frame, name):
    w, s = windows(config), schedule(config)
    x = frame['streams'][name]['batch']
    ex = x - frame['O']
    result = dict(stream_excess=float(ex[w['stream']].mean()), recovery_excess_vs_O=float(ex[w['recovery']].mean()),
                  plateau_excess=float(ex[w['plateau']].mean()), brier=float(x[w['stream']].mean()),
                  survival_regret=float((frame['O_survival'] - frame['streams'][name]['survival'])[w['stream']].mean()),
                  recovery_curve_vs_O=[float(ex[w['base_a'] & (s['position'] == p)].mean())
                                       for p in range(1, s['block_batches'] + 1)],
                  excess_by_cycle=[float(ex[s['cycle'] == c].mean()) for c in range(1, config['stream']['cycles'] + 1)],
                  excess_by_law=[float(ex[w['stream'] & (s['law_index'] == k)].mean()) for k in range(5)])
    if 'I' in frame:
        ei = x - frame['I']
        result.update(recovery_excess_vs_I=float(ei[w['recovery']].mean()),
                      stream_excess_vs_I=float(ei[w['stream']].mean()),
                      recovery_curve_vs_I=[float(ei[w['base_a'] & (s['position'] == p)].mean())
                                           for p in range(1, s['block_batches'] + 1)])
    return result


def cr_diagnostics(config, directory, seed):
    record = read_json(Path(directory) / 'result.json')
    switches, modes = mode_switches(config, seed)
    s = schedule(config)
    law_index = s['law_index']
    w = windows(config)
    with np.load(Path(directory) / 'training.npz') as z:
        kstar, pending, recalled, spawned = z['kstar_after'], z['pending_after'], z['recalled'], z['spawned']
        K_after = z['K_after']
    cr = record['cr']
    snaps = cr['snapshots']
    total = sum(sum(x['by_mode'].values()) for x in snaps)
    mode_pure = sum(max(x['by_mode'].values()) for x in snaps if x['by_mode'])
    law_pure = sum(max(x['by_law'].values()) for x in snaps if x['by_law'])
    assign_mode = assign_law = 0
    for sid in np.unique(kstar):
        sel = kstar == sid
        assign_mode += np.bincount(modes[sel]).max()
        assign_law += np.bincount(law_index[sel]).max()
    changes = cr['counters']['kstar_changes']
    return dict(final_snapshots=int(K_after[-1]), snapshots_created=1 + cr['counters']['spawns'],
                recalls=cr['counters']['recalls'], spawns=cr['counters']['spawns'],
                replacements=cr['counters']['replacements'], kstar_changes=changes,
                true_mode_switches=switches, kstar_changes_per_mode_switch=changes / switches,
                pending_fraction=float(pending.mean()), pending_fraction_cycles_3_8=float(pending[w['stream']].mean()),
                consolidation_mode_purity=mode_pure / total if total else None,
                consolidation_law_purity=law_pure / total if total else None,
                assignment_mode_purity=float(assign_mode / len(kstar)),
                assignment_law_purity=float(assign_law / len(kstar)),
                recall_and_spawn_same_batch=cr['counters']['recall_and_spawn_same_batch'],
                recall_batches=np.flatnonzero(recalled).tolist(), spawn_batches=np.flatnonzero(spawned).tolist(),
                snapshots=snaps)


def paired_se(values_a, values_b):
    diff = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    if len(diff) < 2:
        return 0.
    return float(diff.std(ddof=1) / math.sqrt(len(diff)))


def select_decay(per_d, seeds):
    """Lowest mean excess; any d within one paired seed-SE of the best -> take the largest."""
    means = {d: float(np.mean([per_d[d][s] for s in seeds])) for d in per_d}
    best = min(means, key=means.get)
    rows = {}
    for d in per_d:
        se = paired_se([per_d[d][s] for s in seeds], [per_d[best][s] for s in seeds])
        rows[str(d)] = dict(mean=means[d], difference_from_best=means[d] - means[best], paired_se=se,
                            within_one_se=bool(means[d] - means[best] <= se))
    within = [d for d in per_d if rows[str(d)]['within_one_se']]
    return dict(d=max(within), best_mean_d=best, candidates=rows, within=sorted(within))


def normal_cdf(x):
    return .5 * (1. + math.erf(x / math.sqrt(2.)))


def binomial_tail(p, n=12, k=10):
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def power_note(numerator, denominator, threshold, seeds, n_test=12, needed=10):
    num = np.array([numerator[s] for s in seeds])
    den = np.array([denominator[s] for s in seeds])
    ratios = num / den
    diffs = num - den
    mean_r, sd_r = float(ratios.mean()), float(ratios.std(ddof=1)) if len(ratios) > 1 else float('nan')
    mean_d, sd_d = float(diffs.mean()), float(diffs.std(ddof=1)) if len(diffs) > 1 else float('nan')
    p_seed_r = normal_cdf((1. - mean_r) / sd_r) if sd_r and sd_r > 0 else float(mean_r < 1)
    p_seed_d = normal_cdf((0. - mean_d) / sd_d) if sd_d and sd_d > 0 else float(mean_d < 0)
    ratio_of_means = float(num.mean() / den.mean())
    p_mean = normal_cdf((threshold - mean_r) / (sd_r / math.sqrt(n_test))) if sd_r and sd_r > 0 else float(mean_r <= threshold)
    p10 = binomial_tail(p_seed_r, n_test, needed)
    return dict(per_seed_ratio={str(s): float(r) for s, r in zip(seeds, ratios)}, mean_of_ratios=mean_r,
                ratio_of_means=ratio_of_means, sd_ratio=sd_r, per_seed_difference={str(s): float(x) for s, x in zip(seeds, diffs)},
                mean_difference=mean_d, sd_difference=sd_d, threshold=threshold,
                p_seed_improves_from_ratio_normal=p_seed_r, p_seed_improves_from_difference_normal=p_seed_d,
                p_at_least_10_of_12=p10, p_at_least_10_of_12_from_difference=binomial_tail(p_seed_d, n_test, needed),
                p_seed_mean_ratio_at_or_below_threshold=p_mean,
                below_half=bool(p10 < .5),
                method=('normal model of the per-seed ratio (mean, SD with ddof=1 from the development seeds); '
                        'P(a seed improves) = Phi((1-mean)/SD); P(>=10/12) is the binomial tail; the threshold '
                        'part uses the mean of 12 ratios ~ N(mean, SD/sqrt(12)). Informal, not a gate.'))


# -------------------------------------------------------------- dev phase

def dev_tau(config, output, identity, seeds):
    s = schedule(config)
    c0, c1 = config['calibration']['cycles']
    p0, p1 = config['calibration']['positions']
    mask = (s['cycle'] >= c0) & (s['cycle'] <= c1) & (s['position'] >= p0) & (s['position'] <= p1)
    values, samples, per_seed, inputs = {}, {}, {}, {}
    for i, d in enumerate(config['decay_grid']):
        pooled = []
        for seed in seeds:
            directory = fit_dir(output, seed, fit_spec(config, 'global', None, decays=config['decay_grid']))
            record = verify_result(directory, identity)
            inputs[str(seed)] = record['artifact_hashes']['training.npz']
            with np.load(directory / 'training.npz') as z:
                require(float(z['decays'][i]) == d, 'decay order')
                stat = z['ell_theta'] - z['ell_ema'][:, i]
            pooled.append(stat[mask])
            per_seed.setdefault(str(d), {})[str(seed)] = dict(p99=float(np.percentile(stat[mask], 99)),
                                                              median=float(np.median(stat[mask])))
        pooled = np.concatenate(pooled)
        values[str(d)] = float(np.percentile(pooled, config['calibration']['percentile']))
        samples[str(d)] = dict(n=int(len(pooled)), per_seed=int(mask.sum()), mean=float(pooled.mean()),
                               sd=float(pooled.std(ddof=1)), quantiles={q: float(np.percentile(pooled, float(q)))
                                                                        for q in ('50', '90', '95', '99', '99.5')},
                               max=float(pooled.max()), fraction_negative=float((pooled < 0).mean()))
    return dict(values=values, samples=samples, per_seed=per_seed, statistic='l_theta - l_EMA(d) (nats per batch)',
                window=dict(cycles=[c0, c1], positions=[p0, p1]), percentile=config['calibration']['percentile'],
                method='numpy.percentile, linear interpolation, pooled over the development seeds',
                input_training_npz_sha256=inputs)


def run_dev(config, config_path, output, report, include_optional=True):
    validate_config(config)
    require(config['phase'] in ('dev', 'smoke'), 'dev runs on the development (or smoke) config')
    output, report = Path(output), Path(report)
    manifest = lock(config, config_path, output, kind='dev')
    identity = manifest['identity']
    seeds = config['seeds']
    grid = config['decay_grid']
    log_event(output, event='dev_start', identity=identity)
    timings = {}
    t0 = time.perf_counter()
    # ---- stage 1: ON-R256 + EMA-global at every d, ON-ref, COND-4, O
    g_spec = fit_spec(config, 'global', None, decays=grid)
    jobs = [fit_job(config, identity, output, s, g_spec) for s in seeds]
    jobs += [fit_job(config, identity, output, s, fit_spec(config, 'plain', a)) for s in seeds for a in ('ON-ref', 'COND-4')]
    jobs += oracle_jobs(config, output)
    run_pool(config, output, jobs, 'dev_stage1')
    oracle = assemble_oracle(config, output)
    timings['stage1'] = time.perf_counter() - t0
    tau = dev_tau(config, output, identity, seeds)
    tau_path = output / 'tau.json'
    if tau_path.exists():
        require(read_json(tau_path)['values'] == tau['values'], 'tau recomputation differs from the stored tau')
    else:
        write_json(tau_path, dict(tau, utc=now(), identity=identity))
    log_event(output, event='tau_calibrated', values=tau['values'])
    # ---- stage 2: CR and CR-K1 at each d with tau(d)
    t1 = time.perf_counter()
    jobs = [fit_job(config, identity, output, s, fit_spec(config, 'cr', arm, d, tau['values'][str(d)]))
            for s in seeds for arm in ('CR', 'CR-K1') for d in grid]
    run_pool(config, output, jobs, 'dev_stage2')
    timings['stage2'] = time.perf_counter() - t1
    # ---- selection (uses O and the stage 1-2 fits)
    frames = dev_frames(config, output, seeds, tau, None)
    selection = dev_selection(config, frames, seeds)
    write_json(output / 'selection.json', dict(selection=selection, utc=now(), identity=identity))
    d_cr = selection['CR']['d']
    # ---- stage 3 (optional, descriptive): the other CR arms at CR's selected d
    if include_optional:
        t2 = time.perf_counter()
        jobs = [fit_job(config, identity, output, s, fit_spec(config, 'cr', arm, d_cr, tau['values'][str(d_cr)]))
                for s in seeds for arm in ('CR-no-recall', 'CR-oracle-mode', 'CR-oracle-law')]
        run_pool(config, output, jobs, 'dev_stage3')
        timings['stage3'] = time.perf_counter() - t2
    check_lock(config, config_path, output)
    verified = verify_all_results(output, identity)
    crc = crc_scan(output)
    write_json(output / 'crc_scan.json', crc)
    require(crc['passed'], f'CRC scan failed: {crc["failures"]}')
    timings['total'] = time.perf_counter() - t0
    results = dev_analysis(config, output, report, seeds, tau, selection, d_cr if include_optional else None,
                           identity, oracle, verified, crc, timings)
    completion = dict(identity=identity, utc=now(), verified_fits=verified, crc=crc, timings=timings,
                      dev_results=(report / 'dev_results.json').as_posix(),
                      dev_results_sha256=sha(report / 'dev_results.json'))
    write_json(output / 'completion.json', completion)
    log_event(output, event='dev_complete', seconds=timings['total'])
    return results


def dev_frames(config, output, seeds, tau, d_cr):
    grid = config['decay_grid']
    frames = {}
    for seed in seeds:
        g = fit_dir(output, seed, fit_spec(config, 'global', None, decays=grid))
        fits = {'ON-R256': (g, 'probabilities', None)}
        for i, d in enumerate(grid):
            fits[f'EMA-global@{d}'] = (g, 'ema_probabilities', i)
        for arm in ('CR', 'CR-K1'):
            for d in grid:
                fits[f'{arm}@{d}'] = (fit_dir(output, seed, fit_spec(config, 'cr', arm, d, tau['values'][str(d)])),
                                      'probabilities', None)
        for arm in ('ON-ref', 'COND-4'):
            directory = fit_dir(output, seed, fit_spec(config, 'plain', arm))
            if (directory / 'result.json').exists():
                fits[arm] = (directory, 'probabilities', None)
        if d_cr is not None:
            for arm in ('CR-no-recall', 'CR-oracle-mode', 'CR-oracle-law'):
                fits[f'{arm}@{d_cr}'] = (fit_dir(output, seed, fit_spec(config, 'cr', arm, d_cr, tau['values'][str(d_cr)])),
                                         'probabilities', None)
        frames[seed] = seed_frames(config, output, seed, fits)
    return frames


def dev_selection(config, frames, seeds):
    grid = config['decay_grid']
    out = {}
    for arm in ('CR', 'CR-K1', 'EMA-global'):
        per_d = {d: {s: stream_metrics(config, frames[s], f'{arm}@{d}')['stream_excess'] for s in seeds} for d in grid}
        out[arm] = select_decay(per_d, seeds)
    out['rule'] = ('lowest mean development stream excess over O (cycles 3-8); if several d are within one paired '
                   'seed-SE of the best (SE of the per-seed difference to the best d, ddof=1, /sqrt(3)), take the '
                   'largest of them')
    return out


def dev_analysis(config, output, report, seeds, tau, selection, d_cr, identity, oracle, verified, crc, timings):
    frames = dev_frames(config, output, seeds, tau, d_cr)
    grid = config['decay_grid']
    names = sorted(frames[seeds[0]]['streams'])
    table = {}
    for name in names:
        per = {str(s): stream_metrics(config, frames[s], name) for s in seeds}
        table[name] = dict(
            stream_excess=dict(mean=float(np.mean([per[str(s)]['stream_excess'] for s in seeds])),
                               per_seed={str(s): per[str(s)]['stream_excess'] for s in seeds}),
            recovery_excess_vs_O=dict(mean=float(np.mean([per[str(s)]['recovery_excess_vs_O'] for s in seeds])),
                                      per_seed={str(s): per[str(s)]['recovery_excess_vs_O'] for s in seeds}),
            plateau_excess=dict(mean=float(np.mean([per[str(s)]['plateau_excess'] for s in seeds])),
                                per_seed={str(s): per[str(s)]['plateau_excess'] for s in seeds}),
            survival_regret=dict(mean=float(np.mean([per[str(s)]['survival_regret'] for s in seeds])),
                                 per_seed={str(s): per[str(s)]['survival_regret'] for s in seeds}),
            recovery_curve_vs_O=np.mean([per[str(s)]['recovery_curve_vs_O'] for s in seeds], axis=0).tolist(),
            excess_by_cycle=np.mean([per[str(s)]['excess_by_cycle'] for s in seeds], axis=0).tolist(),
            excess_by_law_cycles_3_8=np.mean([per[str(s)]['excess_by_law'] for s in seeds], axis=0).tolist(),
            brier=float(np.mean([per[str(s)]['brier'] for s in seeds])))
    # CR diagnostics for every CR-type fit
    diagnostics = {}
    for seed in seeds:
        for arm in ('CR', 'CR-K1'):
            for d in grid:
                spec = fit_spec(config, 'cr', arm, d, tau['values'][str(d)])
                diagnostics.setdefault(f'{arm}@{d}', {})[str(seed)] = cr_diagnostics(config, fit_dir(output, seed, spec), seed)
        if d_cr is not None:
            for arm in ('CR-no-recall', 'CR-oracle-mode', 'CR-oracle-law'):
                spec = fit_spec(config, 'cr', arm, d_cr, tau['values'][str(d_cr)])
                diagnostics.setdefault(f'{arm}@{d_cr}', {})[str(seed)] = cr_diagnostics(config, fit_dir(output, seed, spec), seed)
    compact = {k: {s: {kk: vv for kk, vv in v.items() if kk not in ('snapshots', 'recall_batches', 'spawn_batches')}
                   for s, v in rows.items()} for k, rows in diagnostics.items()}
    # pathology rule on the selected CR
    p = config['pathology']
    sel_cr = f'CR@{selection["CR"]["d"]}'
    per_seed_flags = {}
    for s, row in diagnostics[sel_cr].items():
        per_seed_flags[s] = dict(pending_fraction=row['pending_fraction'],
                                 pending_over=row['pending_fraction'] > p['pending_fraction'],
                                 kstar_changes=row['kstar_changes'], true_mode_switches=row['true_mode_switches'],
                                 kstar_over=row['kstar_changes'] > p['kstar_changes_per_mode_switch'] * row['true_mode_switches'],
                                 final_snapshots=row['final_snapshots'],
                                 snapshots_over=row['final_snapshots'] > p['max_snapshots_end'])
    fired_any_seed = any(v['pending_over'] or v['kstar_over'] or v['snapshots_over'] for v in per_seed_flags.values())
    mean_pending = float(np.mean([v['pending_fraction'] for v in per_seed_flags.values()]))
    mean_changes = float(np.mean([v['kstar_changes'] for v in per_seed_flags.values()]))
    mean_switches = float(np.mean([v['true_mode_switches'] for v in per_seed_flags.values()]))
    mean_snaps = float(np.mean([v['final_snapshots'] for v in per_seed_flags.values()]))
    fired_mean = (mean_pending > p['pending_fraction'] or mean_changes > p['kstar_changes_per_mode_switch'] * mean_switches
                  or mean_snaps > p['max_snapshots_end'])
    pathology = dict(rule=p, selected=sel_cr, per_seed=per_seed_flags, fired_any_seed=bool(fired_any_seed),
                     fired_on_seed_means=bool(fired_mean), fired=bool(fired_any_seed),
                     reading=('fires if ANY development seed shows any condition (the literal, conservative '
                              'reading); the seed-mean reading is reported alongside'),
                     implication=('CR is NOT taken to test; revise the protocol and use a new, disjoint '
                                  'development cohort' if fired_any_seed else 'CR may proceed to lock'))
    # power note at the selected d values
    def ex(name, key='stream_excess'):
        return {s: stream_metrics(config, frames[s], name)[key] for s in seeds}
    cr_name, k1_name = sel_cr, f'CR-K1@{selection["CR-K1"]["d"]}'
    thr = config['analysis']['thresholds']
    power = dict(C1=power_note(ex(cr_name), ex('ON-R256'), thr['C1'], seeds),
                 C2=power_note(ex(cr_name), ex(k1_name), thr['C2'], seeds),
                 C3_proxy_vs_O=power_note(ex(cr_name, 'recovery_excess_vs_O'), ex('ON-R256', 'recovery_excess_vs_O'),
                                          thr['C3'], seeds),
                 note=('C3 is defined relative to I, which is not computed for development seeds; the C3 row uses '
                       'recovery excess over O as a proxy and is descriptive only'))
    results = dict(
        label=LABEL + ' DEVELOPMENT PHASE (seeds 20101-20103); results may change only d and tau.',
        protocol=config['protocol'], identity=identity, config=config, analysed_utc=now(), seeds=seeds,
        oracle=dict(kind='O only (true-law column of the 512-sample MC with common random numbers); I is NOT '
                         'computed for development seeds, so recovery is reported relative to O',
                    summary=oracle,
                    O_stream_brier={str(s): float(frames[s]['O'][windows(config)['stream']].mean()) for s in seeds},
                    O_mc_variance_bias={str(s): frames[s]['mc_variance_bias'] for s in seeds}),
        windows=dict(stream_cycles=config['analysis']['excess_cycles'],
                     recovery_positions=config['analysis']['recovery_positions'],
                     plateau_positions=config['analysis']['plateau_positions']),
        tau=tau, selection=selection, arms=table, cr_diagnostics=compact,
        cr_snapshot_details={k: {s: dict(snapshots=v['snapshots'], recall_batches=v['recall_batches'],
                                         spawn_batches=v['spawn_batches']) for s, v in rows.items()}
                             for k, rows in diagnostics.items()},
        pathology=pathology, power=power, fits_verified=verified, crc_scan=crc, timings_seconds=timings,
        costs=dev_costs(config, output, seeds, tau, d_cr))
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / 'dev_results.json', results)
    with (report / 'dev_per_seed.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['arm', 'seed', 'stream_excess', 'recovery_excess_vs_O', 'plateau_excess', 'survival_regret'])
        for name, row in table.items():
            for s in seeds:
                writer.writerow([name, s, row['stream_excess']['per_seed'][str(s)], row['recovery_excess_vs_O']['per_seed'][str(s)],
                                 row['plateau_excess']['per_seed'][str(s)], row['survival_regret']['per_seed'][str(s)]])
    return results


def dev_costs(config, output, seeds, tau, d_cr):
    rows = {}
    for directory in sorted((Path(output) / 'fits').glob('*/result.json')):
        r = read_json(directory)
        key = r['spec']['arm'] + (f'@{r["spec"]["d"]}' if r['spec']['kind'] == 'cr' else
                                  '+EMA' if r['spec']['kind'] == 'global' else '')
        rows.setdefault(key, []).append(dict(elapsed_seconds=r['elapsed_seconds'], total_macs=r['work']['total_macs'],
                                             memory_bytes=r['memory_bytes']['total'],
                                             averages_bytes=r['memory_bytes']['averages']))
    return {k: {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]} for k, v in rows.items()}


# ------------------------------------------------------------- test run

def test_specs(config):
    sel = config['selected']
    specs = [fit_spec(config, 'global', None, decays=[sel['EMA-global']['d']]),
             fit_spec(config, 'plain', 'ON-ref'), fit_spec(config, 'plain', 'COND-4'),
             fit_spec(config, 'cr', 'CR', sel['CR']['d'], sel['CR']['tau']),
             fit_spec(config, 'cr', 'CR-K1', sel['CR-K1']['d'], sel['CR-K1']['tau'])]
    specs += [fit_spec(config, 'cr', arm, sel['CR']['d'], sel['CR']['tau'])
              for arm in ('CR-no-recall', 'CR-oracle-mode', 'CR-oracle-law')]
    return specs


def run_study(config, config_path, output):
    validate_config(config)
    require_selected(config)
    output = Path(output)
    manifest = check_lock(config, config_path, output)
    identity = manifest['identity']
    log_event(output, event='run_start', identity=identity)
    started = time.perf_counter()
    jobs = [fit_job(config, identity, output, s, spec) for s in config['seeds'] for spec in test_specs(config)]
    jobs += oracle_jobs(config, output)
    statuses = run_pool(config, output, jobs, 'run')
    oracle = assemble_oracle(config, output)
    check_lock(config, config_path, output)
    fits = []
    for seed in config['seeds']:
        for spec in test_specs(config):
            directory = fit_dir(output, seed, spec)
            record = verify_result(directory, identity)
            fits.append(dict(seed=seed, arm=spec['arm'], kind=spec['kind'], d=spec.get('d'),
                             path=(directory / 'result.json').relative_to(output).as_posix(),
                             sha256=sha(directory / 'result.json'), elapsed_seconds=record['elapsed_seconds']))
    crc = crc_scan(output)
    write_json(output / 'crc_scan.json', crc)
    require(crc['passed'], f'CRC scan failed: {crc["failures"]}')
    completion = dict(identity=identity, fits=fits, oracle=oracle, statuses=statuses, crc=crc,
                      wall_seconds_this_invocation=time.perf_counter() - started, completed_utc=now())
    write_json(output / 'completion.json', completion)
    log_event(output, event='run_complete', seconds=time.perf_counter() - started)
    return completion


VOLATILE = {'elapsed_seconds', 'training_seconds', 'started_utc', 'completed_utc', 'utc', 'resumes', 'resumed',
            'artifact_hashes', 'checkpoint_signature'}


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in VOLATILE and not str(k).endswith('_seconds')}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def replicate(config, config_path, output):
    """Protocol replicate pass: re-fit CR and ON-R256 (the global fit) from scratch for three test seeds and
    compare bit-for-bit; re-verify every stored hash; CRC-check every archive."""
    validate_config(config)
    output = Path(output)
    manifest = check_lock(config, config_path, output)
    identity = manifest['identity']
    log_event(output, event='replicate_start')
    completion = read_json(output / 'completion.json')
    for fit in completion['fits']:
        require(sha(output / fit['path']) == fit['sha256'], f'result.json changed since completion: {fit["path"]}')
        verify_result((output / fit['path']).parent, identity)
    for seed, row in completion['oracle'].items():
        require(sha(output / row['path']) == row['sha256'], f'oracle file changed: {row["path"]}')
    seeds = list(REPLICATE_SEEDS) if config['phase'] == 'test' else list(config['seeds'])
    sel = config['selected']
    specs = [fit_spec(config, 'global', None, decays=[sel['EMA-global']['d']]),
             fit_spec(config, 'cr', 'CR', sel['CR']['d'], sel['CR']['tau'])]
    target = output / 'replicate'
    jobs = [fit_job(config, identity, target, s, spec) for s in seeds for spec in specs]
    run_pool(config, output, jobs, 'replicate')
    comparisons, mismatches = {}, []
    for s in seeds:
        for spec in specs:
            a_dir, b_dir = fit_dir(output, s, spec), fit_dir(target, s, spec)
            a, b = read_json(a_dir / 'result.json'), read_json(b_dir / 'result.json')
            rows = dict(result_json_normalized=normalize(a) == normalize(b),
                        training_npz=npz_equal(a_dir / 'training.npz', b_dir / 'training.npz'),
                        final_model_sha256=a['final_model_sha256'] == b['final_model_sha256'],
                        end_signature=a['end_signature'] == b['end_signature'],
                        optimizer_sha256=a['optimizer_sha256'] == b['optimizer_sha256'])
            ca, cb = load_checkpoint(a_dir / 'checkpoint.pt', identity), load_checkpoint(b_dir / 'checkpoint.pt', identity)
            rows['checkpoint_mechanism'] = _value_hash(ca['mechanism']) == _value_hash(cb['mechanism'])
            rows['checkpoint_learner'] = signature(ca['learner']) == signature(cb['learner']) == a['end_signature']
            key = f'{spec["arm"]}:{spec["kind"]}:{s}'
            comparisons[key] = rows
            mismatches += [f'{key}:{k}' for k, v in rows.items() if not v]
    crc = crc_scan(output)
    report = dict(identity=identity, utc=now(), seeds=seeds, comparisons=comparisons, mismatches=mismatches,
                  main_fits_reverified=len(completion['fits']), crc=crc, passed=not mismatches and crc['passed'],
                  note='npz archives are compared array-by-array (zip entries carry write timestamps)')
    write_json(output / 'replicate.json', report)
    log_event(output, event='replicate_complete', passed=report['passed'], mismatches=mismatches)
    if not report['passed']:
        raise RuntimeError(f'REPLICATE MISMATCH OR CRC FAILURE, do not analyze: {mismatches} {crc["failures"]}')
    return report


def analyze(config, config_path, output, report, require_lock=True):
    validate_config(config)
    output, report = Path(output), Path(report)
    manifest = check_lock(config, config_path, output) if require_lock else read_json(output / 'manifest.json')
    identity = manifest['identity']
    if config['phase'] == 'test':
        rep = read_json(output / 'replicate.json')
        require(rep['passed'], 'the replicate pass did not pass; analysis is stopped')
    seeds, a = config['seeds'], config['analysis']
    n = len(seeds)
    draws = np.random.default_rng(a['analysis_seed']).integers(0, n, (a['bootstrap_resamples'], n))
    sel = config['selected']
    specs = {s['arm']: s for s in test_specs(config)}
    frames, diag = {}, {}
    for seed in seeds:
        g = fit_dir(output, seed, specs['ON-R256'])
        fits = {'ON-R256': (g, 'probabilities', None), 'EMA-global': (g, 'ema_probabilities', 0)}
        for arm in ('ON-ref', 'COND-4', *CR_ARMS):
            fits[arm] = (fit_dir(output, seed, specs[arm]), 'probabilities', None)
        for arm, (directory, _, _) in fits.items():
            verify_result(directory, identity)
        frames[seed] = seed_frames(config, output, seed, fits)
        for arm in CR_ARMS:
            diag.setdefault(arm, {})[str(seed)] = cr_diagnostics(config, fit_dir(output, seed, specs[arm]), seed)
    metrics = {arm: {s: stream_metrics(config, frames[s], arm) for s in seeds} for arm in TEST_ARMS}

    def vec(arm, key):
        return np.array([metrics[arm][s][key] for s in seeds])

    def criterion(name, arm, comparator, key, threshold):
        num, den = vec(arm, key), vec(comparator, key)
        diff = num - den
        ratio = float(num.mean() / den.mean())
        improved = int((diff < 0).sum())
        passed = ratio <= threshold and improved >= a['consistency_seeds']
        return dict(question=name, learner=arm, comparator=comparator, outcome=key, threshold=threshold,
                    seed_mean_ratio=ratio, ratio_bootstrap95=G.ratio_bootstrap(num, den, draws),
                    mean_of_per_seed_ratios=float((num / den).mean()),
                    per_seed_ratio={str(s): float(x) for s, x in zip(seeds, num / den)},
                    per_seed_difference={str(s): float(x) for s, x in zip(seeds, diff)},
                    seeds_improved=improved, passes=bool(passed))
    C1 = criterion('Does consolidated recall beat plain online learning?', 'CR', 'ON-R256', 'stream_excess',
                   a['thresholds']['C1'])
    C2 = criterion('Does indexing beat one average with switch detection?', 'CR', 'CR-K1', 'stream_excess',
                   a['thresholds']['C2'])
    C3 = criterion('Does recall speed recovery?', 'CR', 'ON-R256', 'recovery_excess_vs_I', a['thresholds']['C3'])
    verdict = ('SUPPORTED' if C1['passes'] and C2['passes'] and C3['passes'] else
               'PARTIAL' if C1['passes'] else 'NOT SUPPORTED')
    arms = {}
    for arm in TEST_ARMS:
        row = {}
        for key in ('stream_excess', 'recovery_excess_vs_I', 'stream_excess_vs_I', 'plateau_excess', 'survival_regret',
                    'recovery_excess_vs_O', 'brier'):
            v = vec(arm, key)
            row[key] = dict(mean=float(v.mean()), bootstrap95=G.bootstrap(v, draws),
                            per_seed={str(s): float(x) for s, x in zip(seeds, v)})
        row['recovery_curve_vs_I'] = np.mean([metrics[arm][s]['recovery_curve_vs_I'] for s in seeds], axis=0).tolist()
        row['recovery_curve_vs_O'] = np.mean([metrics[arm][s]['recovery_curve_vs_O'] for s in seeds], axis=0).tolist()
        row['excess_by_cycle'] = np.mean([metrics[arm][s]['excess_by_cycle'] for s in seeds], axis=0).tolist()
        row['privileged'] = arm in L.PRIVILEGED
        arms[arm] = row
    closed = (vec('ON-R256', 'stream_excess_vs_I') - vec('CR', 'stream_excess_vs_I'))
    descriptive = dict(
        ema_global=dict(stream_excess_ratio_vs_ON_R256=float(vec('EMA-global', 'stream_excess').mean() / vec('ON-R256', 'stream_excess').mean())),
        recall_effect=dict(cr_minus_no_recall_stream=float((vec('CR', 'stream_excess') - vec('CR-no-recall', 'stream_excess')).mean()),
                           cr_minus_no_recall_recovery_vs_I=float((vec('CR', 'recovery_excess_vs_I') - vec('CR-no-recall', 'recovery_excess_vs_I')).mean())),
        inference_gap=dict(cr_minus_oracle_mode_stream=float((vec('CR', 'stream_excess') - vec('CR-oracle-mode', 'stream_excess')).mean()),
                           oracle_mode_vs_CR_K1_ratio=float(vec('CR-oracle-mode', 'stream_excess').mean() / vec('CR-K1', 'stream_excess').mean())),
        fraction_of_ON_R256_excess_over_I_closed=float(closed.mean() / vec('ON-R256', 'stream_excess_vs_I').mean()))
    compact = {k: {s: {kk: vv for kk, vv in v.items() if kk not in ('snapshots', 'recall_batches', 'spawn_batches')}
                   for s, v in rows.items()} for k, rows in diag.items()}
    costs = {}
    for arm, spec in specs.items():
        rows = [read_json(fit_dir(output, s, spec) / 'result.json') for s in seeds]
        costs[arm] = dict(elapsed_seconds=float(np.mean([r['elapsed_seconds'] for r in rows])),
                          total_macs=float(np.mean([r['work']['total_macs'] for r in rows])),
                          memory_bytes=float(np.mean([r['memory_bytes']['total'] for r in rows])),
                          averages_bytes=float(np.mean([r['memory_bytes']['averages'] for r in rows])))
    results = dict(label=LABEL, protocol=config['protocol'], identity=identity, config=config,
                   manifest_locked_utc=manifest.get('locked_utc'), analysed_utc=now(), selected=sel,
                   criteria=dict(C1=C1, C2=C2, C3=C3), verdict=verdict, arms=arms, descriptive=descriptive,
                   cr_diagnostics=compact,
                   oracle=dict(O_stream_brier={str(s): float(frames[s]['O'][windows(config)['stream']].mean()) for s in seeds},
                               I_stream_brier={str(s): float(frames[s]['I'][windows(config)['stream']].mean()) for s in seeds},
                               O_mc_variance_bias={str(s): frames[s]['mc_variance_bias'] for s in seeds}),
                   costs=costs, bootstrap=dict(resamples=a['bootstrap_resamples'], seed=a['analysis_seed'],
                                               note='paired seed-resample percentile intervals; descriptive only'))
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / 'results.json', results)
    with (report / 'per_seed.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        keys = ('stream_excess', 'recovery_excess_vs_I', 'stream_excess_vs_I', 'plateau_excess', 'survival_regret')
        writer.writerow(['seed', 'arm', *keys])
        for s in seeds:
            for arm in TEST_ARMS:
                writer.writerow([s, arm, *[metrics[arm][s][k] for k in keys]])
    return results


# ------------------------------------------------------------------ checks

def fresh(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    return path


def run_checks(config, config_path, output):
    validate_config(config)
    require(config['phase'] == 'smoke', 'check runs on the smoke config')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = Path(output)
    out = base / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    seed = config['seeds'][0]
    identity = dict(check='consolidated_recall')
    report = dict(label=LABEL, seed=seed, utc=now(), checks={}, timings={})
    sched = schedule(config)
    T = sched['batches']
    laws, world = reference_laws(seed), AcquisitionWorld()
    stream = [stream_batch(config, world, laws, seed, i) for i in range(T)]
    floor = config['mechanism']['pattern_floor']
    tau = config['selected']['CR']['tau']
    d = config['selected']['CR']['d']

    def save():
        write_json(out / 'checks.json', report)

    def timed(name, fn):
        started = time.perf_counter()
        value = fn()
        report['timings'][name] = time.perf_counter() - started
        return value

    def npz(directory):
        with np.load(Path(directory) / 'training.npz') as z:
            return {k: z[k] for k in z.files}

    # 1. settings and shared initialisation
    rows, hashes = {}, set()
    for arm in ('ON-R256', 'ON-ref'):
        learner = make_learner(config, seed, arm)
        m = learner.model
        require(m.frame[0].out_features == 64 and m.temporal.hidden_size == 64 and m.decoder[0].out_features == 64
                and m.context.hidden_size == 12 and m.context.input_size == 64 + 8 + 5 * 64, 'architecture')
        require(learner.memory.capacity == (256 if arm == 'ON-R256' else 16) and learner.updates_per_batch == 12, 'R/U')
        require(learner.settings['lr'] == .002, 'lr')
        hashes.add(learner.initial_hash)
        rows[arm] = dict(R=learner.memory.capacity, U=learner.updates_per_batch, initial_hash=learner.initial_hash,
                         parameters=sum(p.numel() for p in m.parameters()))
    c4 = make_learner(config, seed, 'COND-4')
    require(type(c4.model).__name__ == 'Hypotheses' and len(c4.model.heads) == 4 and c4.method == 'conditional'
            and c4.memory.capacity == 256 and c4.settings['updates_per_batch'] == 12
            and c4.settings['evidence_strength'] == 1., 'COND-4 settings')
    rows['COND-4'] = dict(R=c4.memory.capacity, U=c4.settings['updates_per_batch'], heads=len(c4.model.heads),
                          parameters=sum(p.numel() for p in c4.model.parameters()), initial_hash=c4.initial_hash)
    require(len(hashes) == 1, 'initialisation not shared')
    B = block_batches(config)
    require(T == 2 * 5 * 4 and B == 4 and [s[0] for s in stream[:8]] == [0] * 4 + [1] * 4, 'smoke schedule')
    modes = [laws[i].mode for i in sched['law_index']]
    require(laws[0].mode != laws[1].mode and all(laws[i].mode == laws[1].mode for i in (2, 3, 4)),
            'Base A is the only law in its mode')
    report['checks']['settings_and_shared_initialisation'] = dict(
        passed=True, arms=rows, laws=[asdict(x) for x in laws], batches=T, block_batches=B,
        mode_switches=mode_switches(config, seed)[0], stream_channel=config['stream']['channel'])
    save()

    # 2. pattern likelihood equals a direct computation
    learner = make_learner(config, seed, 'ON-R256')
    diffs = []
    for i in range(6):
        li, law, data, cases, truth = stream[i]
        p = predict_all(learner, data.observations, learner.history)
        diffs.append(abs(L.pattern_loglik(p, data.actions, data.survival, floor)
                         - L.pattern_loglik_direct(p, data.actions, data.survival, floor)))
        learner.train(data)
    toy = np.array([[[.999, .5, .2]] * 5, [[.3, .3, .004]] * 5], dtype=np.float32)
    y = np.array([[1, 0, 0], [1, 1, 1]], np.uint8)
    act = np.array([0, 3], np.uint8)
    q0 = np.array([max(1 - .999, .01), .499, .3, .2]); q0 /= q0.sum()
    q1 = np.array([.7, .01, max(.3 - .004, .01), .01]); q1 /= q1.sum()
    manual = math.log(q0[1]) + math.log(q1[3])
    toy_value = L.pattern_loglik(toy, act, y, floor)
    require(max(diffs) < 1e-9 and abs(toy_value - manual) < 1e-6, 'pattern likelihood differs from direct computation')
    report['checks']['pattern_likelihood'] = dict(passed=True, max_abs_difference_stream=max(diffs),
                                                  toy_value=toy_value, toy_manual=manual,
                                                  floor=floor, categories='(1-p4, p4-p8, p8-p12, p12), floored, renormalised')
    save()

    # 3. bias-corrected EMA matches a float64 closed form (count 0: start/EMA-global; count 1: spawn)
    learner = make_learner(config, seed, 'ON-R256')
    names = [k for k, _ in learner.model.named_parameters()]
    snaps = [{k: v.detach().clone() for k, v in learner.model.named_parameters()}]
    start = L.BiasCorrectedEMA(L.parameters(learner), d, 0)
    holder = {}
    learner.optimizer.register_step_post_hook(lambda *_: snaps.append(
        {k: v.detach().clone() for k, v in learner.model.named_parameters()}))
    learner.optimizer.register_step_post_hook(lambda *_: start.update(L.parameters(learner)))
    learner.optimizer.register_step_post_hook(lambda *_: holder['spawn'].update(L.parameters(learner))
                                              if 'spawn' in holder else None)
    require(all(torch.equal(a, b) for a, b in zip(start.weights(), L.parameters(learner))), 'count-0 phi != theta_0')
    for i in range(4):
        if i == 2:
            holder['spawn'] = L.BiasCorrectedEMA(L.parameters(learner), d, 1)
            spawn_index = len(snaps) - 1
            first = max(float((a - b.detach()).abs().max()) for a, b in zip(holder['spawn'].weights(), L.parameters(learner)))
        learner.train(stream[i][2])
    ref0 = L.ema_closed_form(snaps, d, 0)
    ref1 = L.ema_closed_form(snaps[spawn_index:], d, 1)
    err0 = max(float((w.double() - ref0[k]).abs().max()) for k, w in zip(names, start.weights()))
    err1 = max(float((w.double() - ref1[k]).abs().max()) for k, w in zip(names, holder['spawn'].weights()))
    require(start.steps == 48 and holder['spawn'].steps == 24 and err0 < 1e-5 and err1 < 1e-5 and first < 1e-6,
            f'bias-corrected EMA differs from its closed form: {err0}, {err1}, {first}')
    report['checks']['bias_corrected_ema_closed_form'] = dict(
        passed=True, decay=d, count0_steps=start.steps, count0_max_abs_error=err0, count1_steps=holder['spawn'].steps,
        count1_max_abs_error=err1, spawn_phi_minus_theta=first, tolerance=1e-5,
        closed_form='phi_n = sum_j (1-d) d^(n-j) theta_j / (1 - d^n); spawn counts theta_spawn as one sample')
    save()

    # 4. EMA-global fast weights bit-identical to ON-R256 (plain fit, no averaging hooks)
    plain_dir, global_dir = fresh(out / 'plain'), fresh(out / 'global')
    grid = config['decay_grid']
    timed('fit_plain_ON-R256', lambda: run_fit(config, identity, seed, fit_spec(config, 'plain', 'ON-R256'), plain_dir))
    timed('fit_global', lambda: run_fit(config, identity, seed, fit_spec(config, 'global', None, decays=grid), global_dir))
    pr = read_json(fit_dir(plain_dir, seed, fit_spec(config, 'plain', 'ON-R256')) / 'result.json')
    gr = read_json(fit_dir(global_dir, seed, fit_spec(config, 'global', None, decays=grid)) / 'result.json')
    pz = npz(fit_dir(plain_dir, seed, fit_spec(config, 'plain', 'ON-R256')))
    gz = npz(fit_dir(global_dir, seed, fit_spec(config, 'global', None, decays=grid)))
    gck = load_checkpoint(fit_dir(global_dir, seed, fit_spec(config, 'global', None, decays=grid)) / 'checkpoint.pt', identity)
    same = dict(end_signature=pr['end_signature'] == gr['end_signature'],
                final_model=pr['final_model_sha256'] == gr['final_model_sha256'],
                optimizer=pr['optimizer_sha256'] == gr['optimizer_sha256'],
                fast_forecasts=np.array_equal(pz['probabilities'], gz['probabilities']),
                ell_theta=np.array_equal(pz['ell_theta'], gz['ell_theta']),
                fingerprints=pr['fingerprints'] == gr['fingerprints'],
                checkpoint_optimizer_has_no_hooks=not gck['learner'].optimizer._optimizer_step_post_hooks,
                ema_differs_from_fast=not np.array_equal(gz['ema_probabilities'][0], gz['probabilities']))
    require(all(same.values()), f'EMA-global changed the fast weights: {same}')
    report['checks']['ema_global_fast_weights_bit_identical'] = dict(passed=True, identical=same, decays=grid,
                                                                     ema=gr['ema'])
    save()

    # 5. CR with recall, pending and spawn disabled and one snapshot reproduces EMA-global at the same d
    single_rows = {}
    for i, dd in enumerate(grid):
        s_spec = fit_spec(config, 'cr', 'CR-single', dd, tau)
        s_dir = fresh(out / f'single_{i}')
        timed(f'fit_cr_single_{dd}', lambda: run_fit(config, identity, seed, s_spec, s_dir))
        sz = npz(fit_dir(s_dir, seed, s_spec))
        sr = read_json(fit_dir(s_dir, seed, s_spec) / 'result.json')
        ok = dict(predictions=np.array_equal(sz['probabilities'], gz['ema_probabilities'][i]),
                  fast_forecasts=np.array_equal(sz['theta_probabilities'], gz['probabilities']),
                  final_model=sr['final_model_sha256'] == gr['final_model_sha256'],
                  snapshot_equals_ema=sr['cr']['snapshot_model_sha256']['0'] == gr['ema'][i]['model_sha256'],
                  never_pending=not sz['pending_after'].any(), no_spawn=not sz['spawned'].any(),
                  no_recall=not sz['recalled'].any(), consolidations=sr['cr']['counters']['consolidations'] == gr['optimizer_steps'])
        require(all(ok.values()), f'CR-single differs from EMA-global at d={dd}: {ok}')
        single_rows[str(dd)] = ok
    report['checks']['cr_single_reproduces_ema_global'] = dict(passed=True, per_decay=single_rows)
    save()

    # 6. determinism and resumption (CR, CR-K1, oracle-mode, global, COND-4)
    det = {}
    for arm_spec in (fit_spec(config, 'cr', 'CR', d, tau), fit_spec(config, 'cr', 'CR-K1', d, tau),
                     fit_spec(config, 'cr', 'CR-oracle-mode', d, tau), fit_spec(config, 'plain', 'COND-4')):
        dirs = [fresh(out / f'det_{arm_spec["arm"]}_{k}') for k in ('a', 'b', 'resume')]
        timed(f'det_{arm_spec["arm"]}_a', lambda: run_fit(config, identity, seed, arm_spec, dirs[0]))
        timed(f'det_{arm_spec["arm"]}_b', lambda: run_fit(config, identity, seed, arm_spec, dirs[1]))
        try:
            run_fit(config, identity, seed, arm_spec, dirs[2], interrupt_after=13)
            raise AssertionError('interruption did not happen')
        except Interrupted:
            pass
        run_fit(config, identity, seed, arm_spec, dirs[2])
        again = run_fit(config, identity, seed, arm_spec, dirs[2])['status']
        ra, rb, rr = (read_json(fit_dir(x, seed, arm_spec) / 'result.json') for x in dirs)
        ok = dict(deterministic_npz=npz_equal(fit_dir(dirs[0], seed, arm_spec) / 'training.npz',
                                                 fit_dir(dirs[1], seed, arm_spec) / 'training.npz'),
                  deterministic_state=ra['end_signature'] == rb['end_signature'] and normalize(ra) == normalize(rb),
                  resumed_npz=npz_equal(fit_dir(dirs[0], seed, arm_spec) / 'training.npz',
                                           fit_dir(dirs[2], seed, arm_spec) / 'training.npz'),
                  resumed_state=ra['end_signature'] == rr['end_signature'] and rr['resumed'] and len(rr['resumes']) == 1,
                  resumed_record=normalize(ra) == normalize(rr),
                  completed_task_verified_not_rerun=again == 'verified_existing')
        require(all(ok.values()), f'determinism/resumption failed for {arm_spec["arm"]}: {ok}')
        det[arm_spec['arm']] = dict(ok, interrupted_after_batch=13, resumed_from=rr['resumes'][0]['batches_done'])
    g_resume = fresh(out / 'global_resume')
    g_spec = fit_spec(config, 'global', None, decays=grid)
    try:
        run_fit(config, identity, seed, g_spec, g_resume, interrupt_after=21)
        raise AssertionError('interruption did not happen')
    except Interrupted:
        pass
    run_fit(config, identity, seed, g_spec, g_resume)
    grr = read_json(fit_dir(g_resume, seed, g_spec) / 'result.json')
    ok = grr['end_signature'] == gr['end_signature'] and npz_equal(fit_dir(g_resume, seed, g_spec) / 'training.npz',
                                                                     fit_dir(global_dir, seed, g_spec) / 'training.npz') \
        and grr['ema'] == gr['ema']
    require(ok, 'global fit resumption differs')
    det['global'] = dict(resumed_identical=True, interrupted_after_batch=21)
    report['checks']['determinism_and_resumption'] = dict(passed=True, arms=det)
    save()

    # 7. filter equals a brute-force forward pass
    rng = np.random.default_rng(7)
    ells = rng.normal(0, 3, (7, 3))
    init = np.array([1., 0., 0.])
    rec, post = [], init
    for e in ells:
        post = L.filter_step(post, e, config['mechanism']['h'])
        rec.append(post)
    brute = L.forward_bruteforce(init, ells, config['mechanism']['h'])
    matrix = L.forward_matrix(init, ells, config['mechanism']['h'])
    synthetic = max(float(np.abs(np.array(rec) - brute).max()), float(np.abs(matrix - brute).max()))
    cz = npz(fit_dir(out / 'det_CR_a', seed, fit_spec(config, 'cr', 'CR', d, tau)))
    replay_err, segments = 0., 0
    post = np.ones(1)
    for t in range(T):
        K = int((cz['snap_ids'][t] >= 0).sum())
        require(len(post) == K, 'posterior/snapshot count mismatch in replay')
        alpha = L.forward_matrix(post, [cz['ell_snap'][t][:K]], config['mechanism']['h'])[-1]
        logged = cz['posterior'][t][:int(cz['K_after'][t])]
        if cz['spawned'][t]:
            segments += 1
            slot = int(np.argmax(logged))
            point = logged.max() == 1. and logged.sum() == 1.
            if t + 1 < T:
                point = point and list(cz['snap_ids'][t + 1]).index(int(cz['kstar_after'][t])) == slot
            require(point, 'posterior after a spawn is not a point mass on the new snapshot')
            post = logged
        else:
            replay_err = max(replay_err, float(np.abs(logged - alpha).max()))
            post = alpha
    require(synthetic < 1e-12 and replay_err < 1e-10, f'filter differs from the forward pass: {synthetic}, {replay_err}')
    report['checks']['filter_equals_bruteforce_forward'] = dict(
        passed=True, synthetic_K3_T7_max_abs=synthetic, paths=3 ** 8,
        cr_smoke_replay_max_abs=replay_err, cr_spawn_resets=segments,
        note='log-space filter vs path enumeration and a probability-space matrix forward pass; the smoke CR run '
             'is replayed from its logged likelihoods (point-mass resets at spawns)')
    save()

    # 8. oracle arms receive only the lagged index
    lag = {}
    for arm in ('CR-oracle-mode', 'CR-oracle-law'):
        o_spec = fit_spec(config, 'cr', arm, d, tau)
        base_dir = out / f'det_{arm}_a' if arm == 'CR-oracle-mode' else fresh(out / f'lag_{arm}_base')
        if arm == 'CR-oracle-law':
            run_fit(config, identity, seed, o_spec, base_dir)
        z0 = npz(fit_dir(base_dir, seed, o_spec))
        true_index = np.array(modes if arm == 'CR-oracle-mode' else sched['law_index'])
        require(np.array_equal(z0['oracle_index'], true_index), 'oracle index is not the true index of the batch')
        # at the forecast of batch t, k*'s index is the true index of batch t-1
        lagged = all(int(z0['kstar_forecast_index'][t]) == int(true_index[t - 1]) for t in range(1, T))
        t0 = 22
        flipped = int(1 - true_index[t0]) if arm == 'CR-oracle-mode' else int((true_index[t0] + 1) % 5)
        alt_dir = fresh(out / f'lag_{arm}_alt')
        run_fit(config, identity, seed, o_spec, alt_dir, index_override={t0: flipped})
        z1 = npz(fit_dir(alt_dir, seed, o_spec))
        rows_before = (t0 + 1) * config['batch_size']
        before = np.array_equal(z0['probabilities'][:rows_before], z1['probabilities'][:rows_before])
        after = not np.array_equal(z0['probabilities'][rows_before:], z1['probabilities'][rows_before:])
        require(lagged and before and after, f'{arm}: index not lagged ({lagged}, {before}, {after})')
        lag[arm] = dict(kstar_index_at_forecast_equals_previous_batch_index=lagged,
                        altered_index_at_batch=t0, forecasts_up_to_and_including_that_batch_unchanged=before,
                        later_forecasts_change=after, recalls=int(z0['recalled'].sum()), created=int(z0['created'].sum()))
    report['checks']['oracle_arms_receive_only_lagged_index'] = dict(passed=True, arms=lag)
    save()

    # 9. Adam first-moment reset happens only at recall
    adam = {}
    for arm in ('CR', 'CR-oracle-mode', 'CR-no-recall'):
        state = {}
        events = dict(copied=0, not_copied=0)

        def probe(stage, t, learner, mech, event=None):
            opt = learner.optimizer
            snap = dict(m=[opt.state[p]['exp_avg'].clone() if p in opt.state else None for p in L.parameters(learner)],
                        v=[opt.state[p]['exp_avg_sq'].clone() if p in opt.state else None for p in L.parameters(learner)],
                        step=[opt.state[p]['step'].clone() if p in opt.state else None for p in L.parameters(learner)],
                        theta=[p.detach().clone() for p in L.parameters(learner)])
            if stage == 'before_observe':
                state['before'] = snap
                return
            b = state['before']
            eq = lambda xs, ys: all((x is None and y is None) or torch.equal(x, y) for x, y in zip(xs, ys))  # noqa: E731
            if event['copied']:
                events['copied'] += 1
                phi = mech.snapshots[mech.kstar]['ema'].weights()
                require(all(x is None or not x.any() for x in snap['m']), 'first moment not zero after recall')
                require(eq(b['v'], snap['v']) and eq(b['step'], snap['step']), 'second moment/step changed at recall')
                require(all(torch.equal(p, q) for p, q in zip(snap['theta'], phi)), 'theta != phi after recall')
            else:
                events['not_copied'] += 1
                require(eq(b['m'], snap['m']) and eq(b['v'], snap['v']) and eq(b['step'], snap['step'])
                        and eq(b['theta'], snap['theta']), 'optimizer or theta changed without a recall')
        a_spec = fit_spec(config, 'cr', arm, d, tau)
        run_fit(config, identity, seed, a_spec, fresh(out / f'adam_{arm}'), probe=probe)
        az = npz(fit_dir(out / f'adam_{arm}', seed, a_spec))
        require(np.array_equal(az['probabilities'], npz(fit_dir(out / ('det_CR_a' if arm == 'CR' else
                                                                          'det_CR-oracle-mode_a' if arm == 'CR-oracle-mode'
                                                                          else f'adam_{arm}'), seed, a_spec))['probabilities']),
                'probe changed the fit')
        adam[arm] = dict(events, recall_events=int(az['recalled'].sum()))
    require(adam['CR-oracle-mode']['copied'] >= 1 and adam['CR-no-recall']['copied'] == 0, 'no recall exercised')
    report['checks']['adam_first_moment_reset_only_at_recall'] = dict(passed=True, arms=adam)
    save()

    # 10. O and I reproduce truth with realised supplies; common random numbers shared across laws
    cells = sum(G.verify_realised(c, law, t, dd_.actions, dd_.survival) for _, law, dd_, c, t in stream)
    pipeline_truth = 0
    for li, law, data, cases, truth in stream:
        sup0 = MC.mode_frame(cases.base.supplies, law.mode)[None]      # realised supplies in the mode-0 frame
        counts = pattern_counts(cases.base.reserves, cases.factors, cases.signals, sup0, laws)
        onehot = counts[:, li]
        pattern = truth.astype(np.int64).sum(axis=2)
        require(np.array_equal(np.take_along_axis(onehot, pattern[..., None], -1)[..., 0], np.ones_like(pattern)),
                'MC pipeline with realised supplies misses truth')
        pipeline_truth += int(pattern.size)
    small = dict(config, oracle=dict(config['oracle'], samples=64, sub=32))
    paths = {}
    for mode_name, laws_mode, dedup in (('all', 'all', True), ('true', 'true', True), ('naive', 'all', False)):
        p = out / f'oracle_{mode_name}.npz'
        if p.exists():
            p.unlink()
        timed(f'oracle_task_{mode_name}', lambda: oracle_task(dict(config=small, seed=seed, first_batch=8, batches=4,
                                                                      laws=laws_mode, path=str(p), dedup=dedup,
                                                                      channel=MC_CHECK_CHANNEL)))
        with np.load(p) as z:
            paths[mode_name] = {k: z[k] for k in z.files}
    true_li = int(paths['all']['true_law'])
    crn_column = np.array_equal(paths['all']['counts'][:, true_li], paths['true']['counts'][:, 0])
    dedup_equal = np.array_equal(paths['all']['counts'], paths['naive']['counts'])
    clean = MC.row_types(paths['all']['factors']) == 0
    same_mode = [i for i in range(5) if laws[i].mode == laws[1].mode]
    clean_equal = all(np.array_equal(paths['all']['counts'][clean, i], paths['all']['counts'][clean, same_mode[0]])
                      for i in same_mode)
    differs = not np.array_equal(paths['all']['counts'][:, 0], paths['all']['counts'][:, 1])
    require(crn_column and dedup_equal and clean_equal and differs, 'CRN/dedup checks failed')
    report['checks']['oracle_O_I_truth_and_common_random_numbers'] = dict(
        passed=True, realised_supply_cells_verified=cells, mc_pipeline_truth_cells=pipeline_truth,
        O_equals_I_true_law_column=crn_column, dedup_equals_naive=dedup_equal,
        clean_immediate_rows_identical_across_same_mode_laws=clean_equal, modes_differ=differs)
    save()

    # 11. the smoke study end to end: dev pipeline, lock, run, rerun = verify only, replicate, analyze
    pipeline = dict(archived_previous_smoke_outputs=archive_smoke_outputs(base))
    dev_config = dict(config, oracle=dict(config['oracle'], laws='true'))
    dev_out = base / 'dev'
    dev_results = timed('smoke_dev', lambda: run_dev(dev_config, config_path, dev_out, base / 'dev_analysis'))
    pipeline['dev'] = dict(tau=dev_results['tau']['values'], selection={k: v['d'] for k, v in dev_results['selection'].items()
                                                                        if isinstance(v, dict)},
                           pathology_fired=dev_results['pathology']['fired'])
    manifest = timed('smoke_lock', lambda: lock(config, config_path, base))
    attempts = []
    for attempt in range(3):
        try:
            first = timed(f'smoke_run_attempt_{attempt}', lambda: run_study(config, config_path, base))
            attempts.append(dict(attempt=attempt, status='completed'))
            break
        except RuntimeError as error:
            attempts.append(dict(attempt=attempt, status='failed', error=repr(error)))
    else:
        raise RuntimeError(f'smoke run failed three times: {attempts}')
    pipeline['run_attempts'] = attempts
    second = timed('smoke_rerun', lambda: run_study(config, config_path, base))
    statuses = {s['status'] for s in second['statuses']}
    require(statuses <= {'verified_existing', 'existing'}, f'rerun did not only verify: {statuses}')
    strip = lambda summary: {k: {kk: vv for kk, vv in v.items() if kk != 'sha256'} for k, v in summary.items()}  # noqa: E731
    require(strip(first['oracle']) == strip(second['oracle']), 'oracle assembly not reproducible')
    rep = timed('smoke_replicate', lambda: replicate(config, config_path, base))
    results = timed('smoke_analyze', lambda: analyze(config, config_path, base, base / 'analysis'))
    pipeline.update(manifest_identity=manifest['identity'], fits=len(first['fits']), rerun_statuses=sorted(statuses),
                    replicate=dict(passed=rep['passed'], comparisons=len(rep['comparisons'])),
                    verdict_smoke_only=results['verdict'], crc=first['crc'],
                    task_seconds={f"{f['arm']}:{f['kind']}": f['elapsed_seconds'] for f in first['fits']},
                    oracle_seconds_per_seed={k: v['task_seconds'] for k, v in first['oracle'].items()},
                    note='smoke outputs cannot inform scientific choices')
    report['checks']['smoke_pipeline'] = dict(passed=True, **pipeline)
    report['all_passed'] = all(v['passed'] for v in report['checks'].values())
    save()
    return report


def archive_smoke_outputs(base):
    names = ('manifest.json', 'protocol_at_lock.md', 'config_at_lock.json', 'source_at_lock.zip', 'completion.json',
             'fits', 'oracle', 'analysis', 'replicate', 'replicate.json', 'crc_scan.json', 'dev', 'dev_analysis')
    present = [n for n in names if (base / n).exists()]
    if not present:
        return None
    if (base / 'manifest.json').exists():
        require(read_json(base / 'manifest.json')['config']['phase'] == 'smoke', 'refusing to move a scientific run')
    target = base / 'superseded' / now().replace(':', '').replace('+', '_')
    target.mkdir(parents=True, exist_ok=True)
    for n in present:
        shutil.move(str(base / n), str(target / n))
    return target.relative_to(base).as_posix()


# -------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('check', 'dev', 'lock', 'run', 'replicate', 'analyze'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report')
    parser.add_argument('--no-optional', action='store_true', help='dev: skip the optional stage-3 arms')
    args = parser.parse_args()
    config = read_json(args.config)
    output = Path(args.output)
    try:
        if args.command == 'check':
            report = run_checks(config, args.config, output)
            print(json.dumps({k: v['passed'] for k, v in report['checks'].items()}, indent=1))
        elif args.command == 'dev':
            if not args.report:
                raise SystemExit('--report is required for dev')
            results = run_dev(config, args.config, output, args.report, include_optional=not args.no_optional)
            print(json.dumps(dict(tau=results['tau']['values'],
                                  selection={k: v['d'] for k, v in results['selection'].items() if isinstance(v, dict)},
                                  pathology=results['pathology']['fired']), indent=1))
        elif args.command == 'lock':
            print(json.dumps(lock(config, args.config, output)['identity'], indent=1))
        elif args.command == 'run':
            completion = run_study(config, args.config, output)
            print(json.dumps(dict(fits=len(completion['fits']), seconds=completion['wall_seconds_this_invocation'])))
        elif args.command == 'replicate':
            rep = replicate(config, args.config, output)
            print(json.dumps(dict(passed=rep['passed'], mismatches=rep['mismatches'])))
        else:
            if not args.report:
                raise SystemExit('--report is required for analyze')
            results = analyze(config, args.config, output, args.report)
            print(json.dumps(dict(verdict=results['verdict'],
                                  criteria={k: v['passes'] for k, v in results['criteria'].items()}), indent=1))
    except Exception as error:
        log_event(output, event=f'{args.command}_failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
