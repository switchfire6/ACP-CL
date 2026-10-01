"""Step B: the stationary competence gap -- runner and pre-declared analysis.

Implements docs/competence_gap_protocol.md (prospective, written 2026-09-28
before any fitting). One stationary law ``stage_law(seed, 3)`` per seed, seeds
19101-19106, the unchanged recurrent reference learner
(``RepresentationLearner`` arm ``outcome``) and ten arms:

* 8 factorial arms: updates per current batch U in {12, 3}, replay capacity
  R in {16, 256} packets, width W in {64, 256} (frame/temporal width and decoder
  width; context width stays 12). Reference = U12_R16_W64.
* ``exposure``: the reference trained for 32,768 arrivals, scored at 8,192 (must
  reproduce the reference bit-for-bit) and 32,768.
* ``all_action``: PRIVILEGED DIAGNOSTIC -- the reference with a loss over all five
  actions' 3-horizon counterfactual labels (scripts/competence_gap_learner.py).

Every arm of a seed sees the same arrival stream
``AcquisitionWorld.experience(law, 32, trial_seed(seed, 'competence_gap_training', i))``
and arms with the same architecture share the initialisation (the project
learner seeds its network with ``seed + 73019``). The endpoint panel is the
project ``TraceEvaluator(seed, settings)``: 512 factorial queries from
``trial_seed(seed, 'acquisition_query', 0)`` and two 32-record supports from
``trial_seed(seed, 'acquisition_support', [r, False])``; these channels differ
from the training channel, so panel and training draws are disjoint.

The law-known oracle O (evaluator-only; the only component that uses the law)
comes from ``scripts/ideal_observer_mc.py``: ``panel_task`` on the endpoint panel
(K = endpoint_kb * endpoint_batches >= 4096, common random numbers across
actions and cue-flip variants, exact reserves; gauge-integrated reserves as a
sensitivity) and the same MC primitives (supply draw + the project's
``AcquisitionWorld.simulate``) on every online training case (K >= 1024).

Usage (never writes bytecode):
  set PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src, then
  .venv/Scripts/python.exe -B scripts/competence_gap.py check   --config C --output O
  .venv/Scripts/python.exe -B scripts/competence_gap.py lock    --config C --output O
  .venv/Scripts/python.exe -B scripts/competence_gap.py run     --config C --output O
  .venv/Scripts/python.exe -B scripts/competence_gap.py analyze --config C --output O --report R
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
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

import ideal_observer_mc as MC  # noqa: E402
from competence_gap_learner import AllActionLearner, PRIVILEGED_LABEL  # noqa: E402
from acp_cl.acquisition.study import Evaluator  # noqa: E402
from acp_cl.acquisition.world import AcquisitionWorld, Law, mask_affected, stage_law  # noqa: E402
from acp_cl.core_residual.evaluation import TraceEvaluator, recompute_metrics  # noqa: E402
from acp_cl.persistence.learner import state_hash  # noqa: E402
from acp_cl.persistence.study import digest, trial_seed, write_json  # noqa: E402
from acp_cl.predictive_value.mechanism import _value_hash, predict_all  # noqa: E402
from acp_cl.predictive_value.study import load as load_checkpoint, save as save_checkpoint  # noqa: E402
from acp_cl.replay_renewal.study import affinity  # noqa: E402
from acp_cl.representation_learning.learner import RepresentationLearner, learner_signature  # noqa: E402
from acp_cl.representation_learning.study import endpoint_probes, update_marginal  # noqa: E402

LABEL = ('PROSPECTIVE STUDY under docs/competence_gap_protocol.md (Step B). '
         'Oracle O is evaluator-only; the all_action arm is a PRIVILEGED DIAGNOSTIC.')
TRAINING_CHANNEL = 'competence_gap_training'
MC_ENDPOINT_EXACT, MC_ENDPOINT_GAUGE, MC_ONLINE, MC_CHECK = 31, 32, 33, 34
REFERENCE = 'U12_R16_W64'
EXTRA_ARMS = ('exposure', 'all_action')
SCRIPT_FILES = ('scripts/competence_gap.py', 'scripts/competence_gap_learner.py',
                'scripts/ideal_observer_mc.py')


class Interrupted(RuntimeError):
    """Deliberate interruption used only by the resumption check."""


# ------------------------------------------------------------------ basics

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise AssertionError(message)


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return sha(path)


def factorial_arms(config):
    f = config['factors']
    return [f'U{u}_R{r}_W{w}' for u in f['U'] for r in f['R'] for w in f['W']]


def all_arms(config):
    return factorial_arms(config) + list(EXTRA_ARMS)


def arm_spec(config, arm):
    ref = config['reference']
    if arm == 'exposure':
        return dict(arm=arm, kind='exposure', U=ref['U'], R=ref['R'], W=ref['W'],
                    arrivals=config['exposure_arrivals'],
                    score_at=[config['arrivals'], config['exposure_arrivals']], all_actions=False)
    if arm == 'all_action':
        return dict(arm=arm, kind='privileged_diagnostic', U=ref['U'], R=ref['R'], W=ref['W'],
                    arrivals=config['arrivals'], score_at=[config['arrivals']], all_actions=True)
    if arm not in factorial_arms(config):
        raise ValueError(f'unknown arm {arm}')
    u, r, w = (int(part[1:]) for part in arm.split('_'))
    return dict(arm=arm, kind='factorial', U=u, R=r, W=w, arrivals=config['arrivals'],
                score_at=[config['arrivals']], all_actions=False)


def learner_settings(config, spec):
    """Settings for the unchanged RepresentationLearner ('outcome' arm)."""
    return dict(batch_size=config['batch_size'], eval_size=config['eval_size'],
                support_replicates=config['support_replicates'], width=spec['W'],
                experts=config['experts'], context_width=config['context_width'],
                decoder_width=spec['W'], interaction_features=config['interaction_features'],
                lr=config['lr'], memory_packets=spec['R'], updates_per_batch=spec['U'],
                compute_updates_per_batch=spec['U'], auxiliary_weight=config['auxiliary_weight'],
                evidence_strength=config['evidence_strength'])


def make_learner(config, spec, seed):
    settings = learner_settings(config, spec)
    if spec['all_actions']:
        return AllActionLearner(seed, settings, all_actions=True)
    return RepresentationLearner('outcome', seed, settings)


def study_law(config, seed):
    return stage_law(seed, config['stage'])


def validate_config(c):
    expected = {'study', 'protocol', 'smoke', 'seeds', 'stage', 'arrivals', 'exposure_arrivals', 'chunk',
                'checkpoint_every', 'batch_size', 'eval_size', 'support_replicates', 'reference',
                'factors', 'context_width', 'experts', 'interaction_features', 'lr',
                'evidence_strength', 'auxiliary_weight', 'oracle', 'analysis', 'device', 'threads',
                'workers', 'cpu_affinity'}
    if set(c) != expected:
        raise ValueError(f'invalid configuration fields: {set(c) ^ expected}')
    b = c['batch_size']
    for key in ('arrivals', 'exposure_arrivals', 'chunk', 'checkpoint_every'):
        if type(c[key]) is not int or c[key] < 1 or c[key] % b:
            raise ValueError(f'{key} must be a positive multiple of the batch size')
    if c['arrivals'] % c['chunk'] or c['exposure_arrivals'] % c['chunk'] or c['exposure_arrivals'] <= c['arrivals']:
        raise ValueError('chunks must tile both streams and exposure must be longer')
    o = c['oracle']
    per_task = o['online_batches_per_task']
    if (c['exposure_arrivals'] // b) % per_task or (c['arrivals'] // b) % per_task:
        raise ValueError('online oracle tasks must tile the stream')
    if o['online_samples'] % o['online_sub']:
        raise ValueError('online oracle sub-batches must tile the sample count')
    if c['stage'] != 3 or c['device'] != 'cpu' or c['threads'] != 1 or not 1 <= c['workers'] <= 6:
        raise ValueError('stage-3 law, CPU, one thread and at most six workers are required')
    if c['reference'] != dict(U=12, R=16, W=64) or c['factors'] != dict(U=[12, 3], R=[16, 256], W=[64, 256]):
        raise ValueError('declared factor levels are fixed by the protocol')
    fixed = dict(batch_size=32, eval_size=512, support_replicates=2, context_width=12, experts=4,
                 interaction_features=True, lr=.002, evidence_strength=1., auxiliary_weight=1.)
    if any(c[k] != v for k, v in fixed.items()):
        raise ValueError('reference learner settings are fixed by the protocol')
    a = c['analysis']
    if a != dict(matter_fraction=.25, consistency_seeds=5, competent_excess=.010,
                 bootstrap_resamples=20000, analysis_seed=19282026):
        raise ValueError('pre-declared analysis constants are fixed')
    if not c['smoke']:
        if (c['seeds'] != list(range(19101, 19107)) or c['arrivals'] != 8192
                or c['exposure_arrivals'] != 32768 or c['chunk'] != 1024):
            raise ValueError('scientific run must use the declared seeds, stream lengths and chunk')
        if o['endpoint_kb'] * o['endpoint_batches'] < 4096 or o['online_samples'] < 1024:
            raise ValueError('oracle sample counts below the declared minimum')


# ------------------------------------------------------------ lock/manifest

def source_manifest():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'src/acp_cl').rglob('*.py'))}


def lock_payload(config, config_path):
    protocol = ROOT / config['protocol']
    scripts = {name: sha(ROOT / name) for name in SCRIPT_FILES}
    source = source_manifest()
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   platform=platform.platform(), device=config['device'], threads=config['threads'],
                   workers=config['workers'], deterministic=True, cpu_affinity=config['cpu_affinity'])
    identity = dict(config_sha256=digest(config), config_file_sha256=sha(config_path),
                    protocol_sha256=sha(protocol), scripts_sha256=digest(scripts),
                    source_sha256=digest(source), runtime_sha256=digest(runtime))
    return dict(identity=identity, config=copy.deepcopy(config),
                config_path=Path(config_path).resolve().relative_to(ROOT).as_posix(),
                protocol_path=config['protocol'], script_files=scripts, source_files=source,
                runtime=runtime)


def lock(config, config_path, output):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = lock_payload(config, config_path)
    path = output / 'manifest.json'
    if path.exists():
        old = read_json(path)
        if old['identity'] != payload['identity']:
            raise ValueError('an existing manifest locks a different config/protocol/source/runtime')
        return old
    manifest = dict(label=LABEL, **payload, locked_utc=now(),
                    arms=all_arms(config), fits=len(config['seeds']) * len(all_arms(config)))
    (output / 'protocol_at_lock.md').write_bytes((ROOT / config['protocol']).read_bytes())
    (output / 'config_at_lock.json').write_bytes(Path(config_path).read_bytes())
    with zipfile.ZipFile(output / 'source_at_lock.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(payload['script_files']) | set(payload['source_files'])):
            archive.write(ROOT / name, name)
    manifest['source_zip_sha256'] = sha(output / 'source_at_lock.zip')
    write_json(path, manifest)
    return manifest


def check_lock(config, config_path, output):
    manifest = read_json(Path(output) / 'manifest.json')
    live = lock_payload(config, config_path)
    if manifest['identity'] != live['identity']:
        changed = [k for k in live['identity'] if live['identity'][k] != manifest['identity'][k]]
        raise ValueError(f'live files differ from the lock: {changed}')
    if sha(Path(output) / 'protocol_at_lock.md') != manifest['identity']['protocol_sha256']:
        raise ValueError('archived protocol changed')
    return manifest


def log_event(output, **event):
    path = Path(output) / 'attempts.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(utc=now(), **event)) + '\n')


# ------------------------------------------------------------- work proxy

def forward_macs(width, context, queries, supports=32, frames=4, interaction=True):
    """Multiply-adds of one HistoryNetwork packet forward (support + query rows)."""
    rows = supports + queries
    frame = rows * frames * 192 * width
    temporal = rows * frames * 3 * 2 * width * width
    events = width + 5 + 3 + (5 * width if interaction else 0)
    mixing = supports * 5 * width * 2 if interaction else 0
    context_gru = supports * 3 * (events * context + context * context)
    decoder = queries * ((width + context) * width + width * 15)
    return frame + temporal + mixing + context_gru + decoder


def work_proxy(config, spec, batches, evaluate_calls):
    w, c, b = spec['W'], config['context_width'], config['batch_size']
    packet = forward_macs(w, c, b)
    training = 3 * 2 * packet * spec['U'] * batches          # 2 packets/update, backward ~ 2x forward
    online = packet * batches                                 # one causal forecast per arrival batch
    evaluation = forward_macs(w, c, config['eval_size']) * evaluate_calls
    return dict(packet_forward_macs=packet, training_macs=training, online_forecast_macs=online,
                evaluation_macs=evaluation, total_macs=training + online + evaluation,
                formula='training = 3 x 2 packets x U x batches x packet_forward (backward ~ 2x forward)')


# ---------------------------------------------------------------- the fit

def stream_batch(world, law, seed, index, size):
    data_seed = trial_seed(seed, TRAINING_CHANNEL, index)
    data = world.experience(law, size, data_seed)
    cases = world.dataset(law, size, data_seed)
    truth = world.counterfactuals(cases)
    require(np.array_equal(cases.observations, data.observations), 'stream cases differ from experience')
    require(np.array_equal(truth[np.arange(len(data)), data.actions], data.survival),
            'performed outcome differs from physical outcome')
    return data, cases, truth


def worker_init(mask, threads):
    if mask:
        affinity(mask)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def fit_dir(output, seed, arm):
    return Path(output) / 'fits' / f'{arm}_{seed}'


def verify_fit(directory, identity):
    record = read_json(Path(directory) / 'result.json')
    if record['identity'] != identity:
        raise ValueError('fit identity mismatch')
    for name, value in record['artifact_hashes'].items():
        if sha(Path(directory) / name) != value:
            raise ValueError(f'fit artifact changed: {directory}/{name}')
    return record


def run_fit(config, identity, seed, arm, output, interrupt_after=None):
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = fit_dir(output, seed, arm)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = directory / 'result.json', directory / 'checkpoint.pt'
    if result_path.exists():
        verify_fit(directory, identity)
        return dict(seed=seed, arm=arm, status='verified_existing')
    spec = arm_spec(config, arm)
    law, world, size = study_law(config, seed), AcquisitionWorld(), config['batch_size']
    batches = spec['arrivals'] // size
    resumed = False
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, identity)
        learner, record, online = saved['learner'], saved['record'], saved['online']
        require(record['seed'] == seed and record['arm'] == arm and record['spec'] == spec, 'checkpoint spec changed')
        evaluator = TraceEvaluator(seed, learner.settings, saved['evaluation'])
        require(learner_signature(learner) == record['checkpoint_signature'], 'checkpoint learner changed')
        resumed = True
        record['resumes'].append(dict(utc=now(), batches_done=record['batches_done']))
    else:
        learner = make_learner(config, spec, seed)
        evaluator = TraceEvaluator(seed, learner.settings)
        record = dict(identity=identity, label=LABEL, seed=seed, arm=arm, spec=spec,
                      settings=learner.settings, law=dict(mode=law.mode, active=list(law.active)),
                      privileged=spec['all_actions'], privileged_label=PRIVILEGED_LABEL if spec['all_actions'] else None,
                      learner_class=f'{type(learner).__module__}.{type(learner).__qualname__}',
                      initial_model_sha256=state_hash(learner.model.state_dict()),
                      initial_signature=learner_signature(learner, ignore_walltime=True),
                      batches_done=0, fingerprints=[], marginals={}, scores=[], resumes=[],
                      elapsed_seconds=0., started_utc=now())
        online = dict(probabilities=[], truth=[], actions=[], outcomes=[])
    started = time.perf_counter()
    for index in range(record['batches_done'], batches):
        data, cases, truth = stream_batch(world, law, seed, index, size)
        probabilities = predict_all(learner, data.observations, learner.history)
        if spec['all_actions']:
            learner.train(data, truth)
        else:
            learner.train(data)
        record['fingerprints'].append(data.fingerprint())
        online['probabilities'].append(np.array(probabilities, dtype=np.float32, copy=True))
        online['truth'].append(truth.astype(np.uint8))
        online['actions'].append(data.actions.copy())
        online['outcomes'].append(data.survival.copy())
        update_marginal(record['marginals'], law, data)
        record['batches_done'] = index + 1
        arrivals = (index + 1) * size
        if arrivals in spec['score_at']:
            before = learner_signature(learner)
            probes = endpoint_probes(evaluator, learner, [asdict(law)], record['marginals'])
            require(learner_signature(learner) == before, 'endpoint scoring changed the learner')
            record['scores'].append(dict(arrivals=arrivals, probes=probes,
                                         model_sha256=state_hash(learner.model.state_dict()),
                                         signature=learner_signature(learner, ignore_walltime=True),
                                         memory_packets=len(learner.memory.packets),
                                         memory_capacity=learner.memory.capacity,
                                         diagnostics=learner.diagnostics()))
        if arrivals % config['checkpoint_every'] == 0 or index + 1 == batches:
            record['elapsed_seconds'] += time.perf_counter() - started
            started = time.perf_counter()
            record['checkpoint_signature'] = learner_signature(learner)
            save_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record,
                                             evaluation=evaluator.snapshot(), online=online))
        if interrupt_after is not None and index + 1 == interrupt_after:
            raise Interrupted(f'deliberate interruption after batch {index + 1}')
    require(len(learner.memory.packets) == min(spec['R'], batches), 'replay capacity not honoured')
    diagnostics = learner.diagnostics()
    record['diagnostics'] = diagnostics
    record['final_model_sha256'] = state_hash(learner.model.state_dict())
    record['end_signature'] = learner_signature(learner, ignore_walltime=True)
    record['evaluation_files'] = evaluator.export(directory)
    arrays = {k: np.concatenate(v) for k, v in online.items()}
    save_npz(directory / 'training.npz', arrays)
    calls = len(evaluator.records)
    record['work'] = work_proxy(config, spec, batches, calls)
    record['memory_bytes'] = dict(model=diagnostics['model_bytes'], optimizer=diagnostics['optimizer_bytes'],
                                  peak_replay=diagnostics['peak_replay_bytes'], history=diagnostics['history_bytes'],
                                  total=diagnostics['model_bytes'] + diagnostics['optimizer_bytes']
                                  + diagnostics['peak_replay_bytes'] + diagnostics['history_bytes'])
    record['parameters'] = diagnostics['parameters']
    record['optimizer_steps'] = diagnostics['cost']['optimizer_steps']
    record['training_seconds'] = diagnostics['cost']['training_seconds']
    record['resumed'] = resumed
    record['elapsed_seconds'] += time.perf_counter() - started
    record['completed_utc'] = now()
    record['artifact_hashes'] = {name: sha(directory / name) for name in
                                 ('evaluations.json', 'evaluations.npz', 'training.npz')}
    record['checkpoint_signature'] = learner_signature(learner)
    save_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record,
                                     evaluation=evaluator.snapshot(), online=online))
    record['artifact_hashes']['checkpoint.pt'] = sha(checkpoint)
    write_json(result_path, record)
    return dict(seed=seed, arm=arm, status='completed', seconds=record['elapsed_seconds'])


# ------------------------------------------------------------- the oracle

def gauge_levels(observations):
    levels = []
    for start in (1, 9):
        block = observations[:, :, 0:4, start:start + 6].reshape(len(observations), -1)
        require(np.all(block == block[:, :1]), 'gauge region is not constant')
        levels.append(block[:, 0].astype(np.float64))
    return np.stack(levels, axis=1)


def verify_realised(cases, law, truth, actions=None, outcomes=None):
    """O's physics (MC.simulate with exact reserves/factors/signals) given the realised
    supplies reproduces evaluator truth exactly, for every action."""
    base = cases.base
    require(np.array_equal(MC.base_efficiency(cases.factors), base.efficiency)
            and np.array_equal(MC.base_delay(cases.factors), base.delay), 'base efficiency/delay differ')
    for action in range(5):
        out = MC.simulate(base.reserves, cases.factors, cases.signals, base.supplies, law,
                          np.full(len(base.reserves), action))
        require(np.array_equal(out.survival, truth[:, action]), 'realised supplies miss truth')
    if actions is not None:
        out = MC.simulate(base.reserves, cases.factors, cases.signals, base.supplies, law,
                          np.asarray(actions, dtype=np.int64))
        require(np.array_equal(out.survival, outcomes), 'realised performed outcomes differ')
    levels = gauge_levels(cases.observations)
    require(np.array_equal(levels, np.rint(base.reserves / 18 * 240)), 'gauge levels differ from reserves')
    return int(truth.size)


def endpoint_case_data(config, seed):
    law = study_law(config, seed)
    settings = dict(eval_size=config['eval_size'], batch_size=config['batch_size'],
                    support_replicates=config['support_replicates'])
    evaluator, world = Evaluator(seed, settings), AcquisitionWorld()
    cases, truth = evaluator.dataset(law)
    cells = verify_realised(cases, law, truth)
    supports = []
    for rep in range(config['support_replicates']):
        support_seed = trial_seed(seed, 'acquisition_support', [rep, False])
        data = evaluator.support(law, rep)
        scases = world.dataset(law, config['batch_size'], support_seed)
        require(np.array_equal(scases.observations, data.observations), 'support cases differ')
        cells += verify_realised(scases, law, world.counterfactuals(scases), data.actions, data.survival)
        supports.append((scases, data))
    query = dict(reserves=np.ascontiguousarray(cases.base.reserves), levels=gauge_levels(cases.observations),
                 factors=np.ascontiguousarray(cases.factors), signals=np.ascontiguousarray(cases.signals))
    support = dict(reserves=np.concatenate([s.base.reserves for s, _ in supports]),
                   levels=np.concatenate([gauge_levels(s.observations) for s, _ in supports]),
                   factors=np.concatenate([s.factors for s, _ in supports]),
                   signals=np.concatenate([s.signals for s, _ in supports]),
                   actions=np.concatenate([d.actions for _, d in supports]).astype(np.int64))
    return dict(law=law, cases=cases, truth=truth, query=query, support=support, verified_cells=cells,
                support_seeds=[trial_seed(seed, 'acquisition_support', [r, False])
                               for r in range(config['support_replicates'])],
                query_seed=trial_seed(seed, 'acquisition_query', 0))


def endpoint_oracle_task(task):
    """One MC batch (kb samples) of MC.panel_task on the endpoint panel; saved to disk."""
    path = Path(task['path'])
    if path.exists():
        return dict(path=str(path), status='existing')
    config, seed = task['config'], task['seed']
    data = endpoint_case_data(config, seed)
    law = data['law']
    mc_task = dict(key=(task['channel'], seed, 0, task['batch']), channel=task['channel'], seed=seed,
                   replicate=0, batch=task['batch'], kb=config['oracle']['endpoint_kb'],
                   gauge=task['gauge'], laws=[(law.mode, list(law.active))],
                   query=data['query'], support=data['support'])
    started = time.perf_counter()
    result = MC.panel_task(mc_task)
    save_npz(path, dict(query_counts=result['query_counts'][0], support_counts=result['support_counts'][0],
                        kb=np.array(result['kb']), case_simulations=np.array(result['case_simulations']),
                        seconds=np.array(time.perf_counter() - started)))
    return dict(path=str(path), status='completed', simulations=int(result['case_simulations']))


def online_oracle_task(task):
    """Law-known O on one block of training cases: MC over the hidden supply noise
    (exact reserves; supply draw of ideal_observer_mc; project simulate)."""
    path = Path(task['path'])
    if path.exists():
        return dict(path=str(path), status='existing')
    config, seed, first, count = task['config'], task['seed'], task['first_batch'], task['batches']
    law, world, size = study_law(config, seed), AcquisitionWorld(), config['batch_size']
    started = time.perf_counter()
    parts = [stream_batch(world, law, seed, i, size) for i in range(first, first + count)]
    reserves = np.concatenate([c.base.reserves for _, c, _ in parts])
    factors = np.concatenate([c.factors for _, c, _ in parts])
    signals = np.concatenate([c.signals for _, c, _ in parts])
    truth = np.concatenate([t for _, _, t in parts])
    verified = sum(verify_realised(c, law, t, d.actions, d.survival) for d, c, t in parts)
    n, samples, sub = len(reserves), config['oracle']['online_samples'], config['oracle']['online_sub']
    counts = np.zeros((n, 5, 3), np.int64)
    means = MC.supply_means(factors)
    for part in range(samples // sub):
        rng = np.random.default_rng([MC.MC_ROOT, MC_ONLINE, seed, first, part])
        noise = rng.normal(0., MC.TransferWorld.supply_noise, (sub, n, MC.STEPS, 2))
        supplies = MC.mode_frame(np.clip(means[None] + noise, 0, None), law.mode).reshape(sub * n, MC.STEPS, 2)
        tiled = [MC.tile_rows(v, sub) for v in (reserves, factors, signals)]
        for action in range(5):
            out = MC.simulate(*tiled, supplies, law, np.full(sub * n, action))
            counts[:, action] += np.rint(out.survival.reshape(sub, n, 3).sum(axis=0)).astype(np.int64)
    save_npz(path, dict(counts=counts, samples=np.array(samples), truth=truth.astype(np.uint8),
                        factors=factors, signals=signals, reserves=reserves,
                        fingerprints=np.array([d.fingerprint() for d, _, _ in parts]),
                        verified_cells=np.array(verified), seconds=np.array(time.perf_counter() - started)))
    return dict(path=str(path), status='completed', simulations=n * samples * 5)


def oracle_tasks(config, output):
    tasks = []
    o, size = config['oracle'], config['batch_size']
    for seed in config['seeds']:
        kinds = [(False, MC_ENDPOINT_EXACT, 'exact')] + ([(True, MC_ENDPOINT_GAUGE, 'gauge')]
                                                         if o['gauge_sensitivity'] else [])
        for gauge, channel, name in kinds:
            for b in range(o['endpoint_batches']):
                tasks.append(('endpoint', dict(config=config, seed=seed, gauge=gauge, channel=channel, batch=b,
                              path=str(Path(output) / 'oracle' / 'endpoint' / f'{name}_{seed}_{b:02d}.npz'))))
        total = config['exposure_arrivals'] // size
        for first in range(0, total, o['online_batches_per_task']):
            tasks.append(('online', dict(config=config, seed=seed, first_batch=first,
                          batches=o['online_batches_per_task'],
                          path=str(Path(output) / 'oracle' / 'online' / f'{seed}_{first:05d}.npz'))))
    return tasks


def assemble_oracle(config, output):
    """Collect per-task MC files into one verified file per seed."""
    o, size = config['oracle'], config['batch_size']
    summary = {}
    for seed in config['seeds']:
        path = Path(output) / 'oracle' / f'seed_{seed}.npz'
        data = endpoint_case_data(config, seed)
        arrays = dict(panel_truth=data['truth'].astype(np.uint8), panel_factors=data['cases'].factors,
                      panel_signals=data['cases'].signals,
                      panel_observations_sha256=np.array(hashlib.sha256(data['cases'].observations.tobytes()).hexdigest()))
        simulations = 0
        for name in ('exact', 'gauge') if o['gauge_sensitivity'] else ('exact',):
            counts = []
            for b in range(o['endpoint_batches']):
                with np.load(Path(output) / 'oracle' / 'endpoint' / f'{name}_{seed}_{b:02d}.npz') as z:
                    counts.append(z['query_counts'].astype(np.int64))
                    simulations += int(z['case_simulations'])
            counts = np.stack(counts)                                  # (batches, 4 variants, 512, 5, 3)
            arrays[f'endpoint_{name}_batch_counts'] = counts
            arrays[f'endpoint_{name}'] = counts.sum(axis=0) / (len(counts) * o['endpoint_kb'])
        total = config['exposure_arrivals'] // size
        pieces = []
        for first in range(0, total, o['online_batches_per_task']):
            with np.load(Path(output) / 'oracle' / 'online' / f'{seed}_{first:05d}.npz') as z:
                pieces.append({k: z[k] for k in ('counts', 'truth', 'factors', 'signals', 'reserves', 'fingerprints')})
                require(int(z['samples']) == o['online_samples'], 'online sample count changed')
                simulations += int(z['counts'].shape[0]) * o['online_samples'] * 5
        for key in ('counts', 'truth', 'factors', 'signals', 'reserves', 'fingerprints'):
            arrays[f'online_{key}'] = np.concatenate([p[key] for p in pieces])
        arrays['online'] = arrays['online_counts'] / o['online_samples']
        digest_ = save_npz(path, arrays)
        summary[str(seed)] = dict(path=path.relative_to(Path(output)).as_posix(), sha256=digest_,
                                  case_simulations=simulations, endpoint_verified_cells=data['verified_cells'],
                                  query_seed=data['query_seed'], support_seeds=data['support_seeds'])
    return summary


# --------------------------------------------------------------------- run

def estimated_cost(config, kind, payload):
    if kind == 'fit':
        spec = arm_spec(config, payload['arm'])
        return spec['arrivals'] / 1024 * spec['U'] / 12 * (1.9 if spec['W'] == 256 else 1.) * 5.4
    if kind == 'online':
        return payload['batches'] * config['batch_size'] * config['oracle']['online_samples'] * 5 / 6.8e5
    return 5.


def dispatch(kind, payload):
    if kind == 'fit':
        return run_fit(**payload)
    if kind == 'online':
        return online_oracle_task(payload)
    return endpoint_oracle_task(payload)


def run_study(config, config_path, output):
    validate_config(config)
    output = Path(output)
    manifest = check_lock(config, config_path, output)
    identity = manifest['identity']
    log_event(output, event='run_start', identity=identity)
    jobs = [('fit', dict(config=config, identity=identity, seed=s, arm=a, output=str(output)))
            for s in config['seeds'] for a in all_arms(config)]
    jobs += oracle_tasks(config, output)
    jobs.sort(key=lambda job: -estimated_cost(config, *job))
    started, failures = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=config['workers'], initializer=worker_init,
                             initargs=(config['cpu_affinity'], config['threads'])) as pool:
        futures = {pool.submit(dispatch, kind, payload): (kind, payload) for kind, payload in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            kind, payload = futures[future]
            label = dict(kind=kind, seed=payload['seed'], arm=payload.get('arm'),
                         part=payload.get('first_batch', payload.get('batch')))
            try:
                result = future.result()
                if kind == 'fit':
                    print(json.dumps(dict(done=done, total=len(jobs), **result,
                                          elapsed=round(time.perf_counter() - started))), flush=True)
            except Exception as error:  # every failed attempt is kept and reported
                failures.append(dict(**label, error=repr(error), traceback=traceback.format_exc()))
                log_event(output, event='task_failed', **label, error=repr(error), traceback=traceback.format_exc())
                print(json.dumps(dict(failed=label, error=repr(error))), flush=True)
    if failures:
        log_event(output, event='run_incomplete', failures=len(failures))
        raise RuntimeError(f'{len(failures)} tasks failed; rerun `run` to resume')
    oracle = assemble_oracle(config, output)
    check_lock(config, config_path, output)
    fits = []
    for seed in config['seeds']:
        for arm in all_arms(config):
            directory = fit_dir(output, seed, arm)
            record = verify_fit(directory, identity)
            fits.append(dict(seed=seed, arm=arm, path=(directory / 'result.json').relative_to(output).as_posix(),
                             sha256=sha(directory / 'result.json'), elapsed_seconds=record['elapsed_seconds']))
    completion = dict(identity=identity, fits=fits, oracle=oracle, wall_seconds_this_invocation=time.perf_counter() - started,
                      completed_utc=now())
    write_json(output / 'completion.json', completion)
    log_event(output, event='run_complete', seconds=time.perf_counter() - started)
    return completion


# ---------------------------------------------------------------- checks

def _optimizer_hash(learner):
    return _value_hash(learner.optimizer.state_dict())


def run_checks(config, output):
    """Engineering checks (determinism, settings, replay, all-action loss, oracle, resume)."""
    validate_config(config)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(output) / 'checks'
    output.mkdir(parents=True, exist_ok=True)
    seed = config['seeds'][0]
    law, world, size = study_law(config, seed), AcquisitionWorld(), config['batch_size']
    stream = [stream_batch(world, law, seed, i, size) for i in range(48)]
    report = dict(label=LABEL, seed=seed, utc=now(), checks={})

    # 1. arm settings applied, initialisation sharing
    rows, init = {}, {}
    for arm in all_arms(config):
        spec = arm_spec(config, arm)
        learner = make_learner(config, spec, seed)
        m = learner.model
        require(m.frame[0].out_features == spec['W'] and m.temporal.hidden_size == spec['W'], 'width not applied')
        require(m.decoder[0].out_features == spec['W'] and m.context.hidden_size == config['context_width'],
                'decoder/context width not applied')
        require(m.context.input_size == spec['W'] + 8 + 5 * spec['W'], 'interaction features missing')
        require(learner.memory.capacity == spec['R'] and learner.updates_per_batch == spec['U'], 'R/U not applied')
        require(isinstance(learner, AllActionLearner) == spec['all_actions'], 'privileged flag misapplied')
        for data, _, truth in stream[:2]:
            learner.train(data, truth) if spec['all_actions'] else learner.train(data)
        require(learner.cost['optimizer_steps'] == 2 * spec['U'], 'optimizer steps differ from U')
        init.setdefault(spec['W'], set()).add(learner.initial_hash)
        rows[arm] = dict(W=spec['W'], R=spec['R'], U=spec['U'], parameters=learner.diagnostics()['parameters'],
                         initial_hash=learner.initial_hash, optimizer_steps_after_2_batches=learner.cost['optimizer_steps'],
                         forward_macs_per_packet=forward_macs(spec['W'], config['context_width'], size))
    require(all(len(v) == 1 for v in init.values()) and len(init) == 2, 'initialisation not shared within architecture')
    report['checks']['arm_settings'] = dict(passed=True, arms=rows,
                                            shared_initial_hash_by_width={str(k): sorted(v) for k, v in init.items()})

    # 2. replay capacity honoured (reservoir) with the real learner memory
    capacity = {}
    for R in config['factors']['R']:
        spec = dict(arm_spec(config, REFERENCE), R=R, U=1)
        learner = make_learner(config, spec, seed)
        sizes = []
        for data, _, _ in stream[:40]:
            learner.train(data)
            sizes.append(len(learner.memory.packets))
        ids = learner.memory.ids
        require(max(sizes) == min(R, 40) and len(set(ids)) == len(ids) and all(0 <= i < 40 for i in ids),
                'replay capacity violated')
        capacity[str(R)] = dict(max_packets=max(sizes), seen=learner.memory.seen, ids=list(ids),
                                peak_replay_bytes=learner.peak_memory_bytes)
    report['checks']['replay_capacity'] = dict(passed=True, **capacity)

    # 3. all-action learner: performed-only mode reproduces the project loop exactly;
    #    all-action loss equals an independent float64 BCE over 5 x 3 labels.
    spec = arm_spec(config, REFERENCE)
    ref = RepresentationLearner('outcome', seed, learner_settings(config, spec))
    alt = AllActionLearner(seed, learner_settings(config, spec), all_actions=False)
    for data, _, truth in stream[:4]:
        ref.train(data)
        alt.train(data, truth)
    same = (state_hash(ref.model.state_dict()) == state_hash(alt.model.state_dict())
            and _optimizer_hash(ref) == _optimizer_hash(alt) and ref.memory.ids == alt.memory.ids
            and {k: v for k, v in ref.cost.items() if not k.endswith('seconds')}
            == {k: v for k, v in alt.cost.items() if not k.endswith('seconds')})
    require(same, 'performed-only mode of the subclass differs from RepresentationLearner.train')
    full = AllActionLearner(seed, learner_settings(config, arm_spec(config, 'all_action')), all_actions=True)
    for data, _, truth in stream[:3]:
        full.train(data, truth)
    item = full.memory.packets[-1]
    data, _, truth = stream[2]
    require(np.array_equal(item.truth, truth), 'replayed packet does not carry its own counterfactual labels')
    loss = float(full.packet_loss(item).detach())
    with torch.no_grad():
        probs = full.forward_with_features(item.query.observations, item.support)[0].double().clamp(1e-6, 1 - 1e-6).numpy()
    y = item.truth.astype(np.float64)
    manual = float(-(y * np.log(probs) + (1 - y) * np.log(1 - probs)).mean())
    full.all_actions = False
    performed = float(full.packet_loss(item).detach())
    full.all_actions = True
    p_sel = probs[np.arange(len(data)), data.actions]
    manual_performed = float(-(data.survival * np.log(p_sel) + (1 - data.survival) * np.log(1 - p_sel)).mean())
    require(abs(loss - manual) < 1e-5 and abs(performed - manual_performed) < 1e-5, 'all-action loss incorrect')
    report['checks']['all_action_loss'] = dict(
        passed=True, performed_only_mode_bit_identical_to_reference=True, all_action_loss=loss,
        independent_float64=manual, performed_loss_same_state=performed, independent_performed=manual_performed,
        label_cells_per_packet=int(item.truth.size), privileged_label_presentations=full.privileged_label_presentations)

    # 4. determinism: two independent short fits of the reference give identical state and forecasts
    def short_fit(batches):
        learner = make_learner(config, spec, seed)
        forecasts = []
        for data, _, _ in stream[:batches]:
            forecasts.append(predict_all(learner, data.observations, learner.history))
            learner.train(data)
        return learner_signature(learner, ignore_walltime=True), np.stack(forecasts)
    a, b = short_fit(16), short_fit(16)
    require(a[0] == b[0] and np.array_equal(a[1], b[1]), 'fit is not deterministic')
    report['checks']['determinism'] = dict(passed=True, batches=16, signature=a[0])

    # 5. oracle physics reproduces truth; dedup MC equals naive MC; lean online MC agrees with panel_task
    data = endpoint_case_data(config, seed)
    online_cells = sum(verify_realised(c, law, t, d.actions, d.survival) for d, c, t in stream)
    mc = dict(key=(MC_CHECK, seed, 0, 0), channel=MC_CHECK, seed=seed, replicate=0, batch=0, kb=4, gauge=False,
              laws=[(law.mode, list(law.active))], query=data['query'], support=data['support'])
    require(np.array_equal(MC.panel_task(mc)['query_counts'], MC.naive_panel_counts(mc)), 'dedup MC differs from naive')
    kb, reps = 512, 2
    panel = sum(MC.panel_task(dict(mc, kb=kb, batch=b, key=(MC_CHECK, seed, 0, b)))['query_counts'][0, 0]
                for b in range(reps)) / (kb * reps)
    task = dict(config=dict(config, oracle=dict(config['oracle'], online_samples=kb * reps, online_sub=kb)),
                seed=seed, first_batch=0, batches=16, path=str(output / 'lean_check.npz'))
    if Path(task['path']).exists():
        Path(task['path']).unlink()
    # lean MC on the endpoint panel itself (same cases as panel_task) via the same primitives
    cases = data['cases']
    means = MC.supply_means(cases.factors)
    counts = np.zeros((len(cases.factors), 5, 3), np.int64)
    for part in range(reps):
        rng = np.random.default_rng([MC.MC_ROOT, MC_CHECK, seed, 999, part])
        noise = rng.normal(0., MC.TransferWorld.supply_noise, (kb, len(cases.factors), MC.STEPS, 2))
        supplies = MC.mode_frame(np.clip(means[None] + noise, 0, None), law.mode).reshape(-1, MC.STEPS, 2)
        tiled = [MC.tile_rows(v, kb) for v in (cases.base.reserves, cases.factors, cases.signals)]
        for action in range(5):
            out = MC.simulate(*tiled, supplies, law, np.full(len(supplies), action))
            counts[:, action] += np.rint(out.survival.reshape(kb, -1, 3).sum(axis=0)).astype(np.int64)
    lean = counts / (kb * reps)
    brier = lambda p: float(((p - data['truth']) ** 2).mean())  # noqa: E731
    se = np.sqrt((panel * (1 - panel) + lean * (1 - lean)) / (kb * reps) + 1e-12)
    z = (panel - lean) / se
    require(abs(brier(panel) - brier(lean)) < 2e-3 and float(np.mean(z ** 2)) < 1.3, 'lean MC disagrees with panel_task')
    online_oracle_task(task)
    with np.load(task['path']) as zfile:
        require(np.array_equal(zfile['truth'], np.concatenate([t for _, _, t in stream[:16]])), 'online oracle truth misaligned')
        lean_online = zfile['counts'] / (kb * reps)
    report['checks']['oracle'] = dict(
        passed=True, realised_supply_cells_verified=dict(endpoint_and_supports=data['verified_cells'],
                                                         online_first_48_batches=online_cells),
        dedup_equals_naive=True, panel_task_brier=brier(panel), lean_brier=brier(lean),
        mean_z_squared=float(np.mean(z ** 2)), max_abs_z=float(np.abs(z).max()),
        online_task_brier_first_512=float(((lean_online - np.concatenate([t for _, _, t in stream[:16]])) ** 2).mean()))

    # 6. evaluator/training disjointness
    training_seeds = {trial_seed(seed, TRAINING_CHANNEL, i) for i in range(config['exposure_arrivals'] // size)}
    evaluator_seeds = {data['query_seed'], *data['support_seeds']}
    panel_hashes = {hashlib.sha256(o.tobytes()).hexdigest() for o in data['cases'].observations}
    train_hashes = {hashlib.sha256(o.tobytes()).hexdigest() for d, _, _ in stream for o in d.observations}
    require(not training_seeds & evaluator_seeds and not panel_hashes & train_hashes, 'panel not disjoint')
    report['checks']['disjointness'] = dict(passed=True, training_seed_count=len(training_seeds),
                                            evaluator_seeds=sorted(evaluator_seeds), shared_seeds=0,
                                            shared_images_first_48_batches=0)

    # 7. resumption: interrupted + resumed fit equals an uninterrupted fit
    small = dict(config, arrivals=512, exposure_arrivals=1024, chunk=256, checkpoint_every=256)
    identity = dict(check='resume')
    for name in ('uninterrupted', 'interrupted'):
        target = output / f'resume_{name}'
        if target.exists():
            import shutil
            shutil.rmtree(target)
    run_fit(small, identity, seed, 'exposure', str(output / 'resume_uninterrupted'))
    try:
        run_fit(small, identity, seed, 'exposure', str(output / 'resume_interrupted'), interrupt_after=13)
        raise AssertionError('interruption did not happen')
    except Interrupted:
        pass
    run_fit(small, identity, seed, 'exposure', str(output / 'resume_interrupted'))
    a = read_json(fit_dir(output / 'resume_uninterrupted', seed, 'exposure') / 'result.json')
    b = read_json(fit_dir(output / 'resume_interrupted', seed, 'exposure') / 'result.json')
    with np.load(fit_dir(output / 'resume_uninterrupted', seed, 'exposure') / 'training.npz') as za, \
            np.load(fit_dir(output / 'resume_interrupted', seed, 'exposure') / 'training.npz') as zb:
        same_online = all(np.array_equal(za[k], zb[k]) for k in za.files)
    same_scores = [x['probes'] == y['probes'] and x['signature'] == y['signature'] for x, y in zip(a['scores'], b['scores'])]
    require(a['end_signature'] == b['end_signature'] and same_online and all(same_scores)
            and a['artifact_hashes']['evaluations.npz'] == b['artifact_hashes']['evaluations.npz']
            and b['resumed'] and len(b['resumes']) == 1, 'resumed fit differs')
    report['checks']['resumption'] = dict(passed=True, interrupted_after_batch=13, resumed_from_batch=b['resumes'][0]['batches_done'],
                                          end_signature=a['end_signature'], online_identical=True, endpoint_identical=True)
    write_json(output / 'checks.json', report)
    return report


# --------------------------------------------------------------- analysis

def monotone(p):
    return np.minimum.accumulate(np.asarray(p, dtype=np.float64), axis=-1)


def case_brier(p, truth):
    return ((np.asarray(p, dtype=np.float64) - truth.astype(np.float64)) ** 2).mean(axis=(1, 2))


def first_argmax_survival(p, truth):
    actions = np.asarray(p)[:, :, -1].argmax(axis=1)
    return truth[np.arange(len(actions)), actions, -1].astype(np.float64)


def load_trace(directory):
    records = read_json(Path(directory) / 'evaluations.json')['records']
    arrays = np.load(Path(directory) / 'evaluations.npz')
    return records, arrays


def endpoint_metrics(records, arrays, probes, oracle, cues):
    """Learner metrics for one score point (replicate-averaged per the locked rules)."""
    row = probes[0]
    truth = None
    per_rep = []
    for r, trace in enumerate(row['correct']):
        p = arrays[f'e{trace}_probabilities']
        t = arrays[f'e{trace}_truth']
        truth = t if truth is None else truth
        require(np.array_equal(t, truth) and np.array_equal(t, oracle['panel_truth']), 'panel truth mismatch')
        m = recompute_metrics(p, t, arrays[f'e{trace}_affected'], arrays[f'e{trace}_valid'],
                              records[trace]['marginal'])
        require(m == records[trace]['metrics'], 'recorded metrics differ from raw arrays')
        benefits = {}
        for c in cues:
            f = row['flips'][str(c)][r]
            affected = arrays[f'e{f}_affected']
            flipped = recompute_metrics(arrays[f'e{f}_probabilities'], t, affected, arrays[f'e{f}_valid'])
            correct = recompute_metrics(p, t, affected, arrays[f'e{f}_valid'])
            benefits[c] = flipped['focus_brier'] - correct['focus_brier']
        per_rep.append(dict(brier=m['brier'], survival=m['survival'], marginal_brier=m['marginal_brier'],
                            benefits=benefits))
    return dict(brier=float(np.mean([x['brier'] for x in per_rep])),
                survival=float(np.mean([x['survival'] for x in per_rep])),
                marginal_brier=float(np.mean([x['marginal_brier'] for x in per_rep])),
                cue_benefit={str(c): float(np.mean([x['benefits'][c] for x in per_rep])) for c in cues},
                replicate_brier=[x['brier'] for x in per_rep])


def oracle_endpoint(oracle, cues, kb, name='exact'):
    p = oracle[f'endpoint_{name}']                                 # (4 variants, 512, 5, 3)
    truth = oracle['panel_truth']
    correct = monotone(p[0])
    everything = np.ones(len(truth), bool)
    m = recompute_metrics(correct, truth, everything, everything)
    benefits = {}
    factors = oracle['panel_factors']
    for c in cues:
        affected = MC.mask_affected_like(factors, c)
        flipped = monotone(p[1 + c])
        benefits[str(c)] = float(case_brier(flipped, truth)[affected].mean() - case_brier(correct, truth)[affected].mean())
    # delete-one-batch jackknife MC standard error of O's all-case Brier
    counts = oracle[f'endpoint_{name}_batch_counts'][:, 0]      # (batches, 512, 5, 3)
    total, n = counts.sum(axis=0), len(counts)
    samples_per_batch = int(kb)
    require(np.allclose(total / (n * samples_per_batch), p[0]), 'endpoint MC sample count mismatch')
    jack = np.array([float(((monotone((total - counts[i]) / ((n - 1) * samples_per_batch)) - truth) ** 2).mean())
                     for i in range(n)])
    se = float(np.sqrt((n - 1) / n * ((jack - jack.mean()) ** 2).sum()))
    variance_bias = float((p[0] * (1 - p[0])).mean() / (n * samples_per_batch))
    return dict(brier=m['brier'], survival=m['survival'], cue_benefit=benefits, mc_se_brier=se,
                samples=n * samples_per_batch, mc_variance_bias=variance_bias)


def bootstrap(values, draws):
    values = np.asarray(values, dtype=np.float64)
    means = values[draws].mean(axis=1)
    return dict(lower=float(np.percentile(means, 2.5)), upper=float(np.percentile(means, 97.5)))


def ratio_bootstrap(numerator, denominator, draws):
    num, den = np.asarray(numerator)[draws].mean(axis=1), np.asarray(denominator)[draws].mean(axis=1)
    ratio = num / den
    return dict(lower=float(np.percentile(ratio, 2.5)), upper=float(np.percentile(ratio, 97.5)))


def analyze(config, config_path, output, report, require_lock=True):
    validate_config(config)
    output, report = Path(output), Path(report)
    manifest = check_lock(config, config_path, output) if require_lock else read_json(output / 'manifest.json')
    identity = manifest['identity']
    seeds, arms, factorial = config['seeds'], all_arms(config), factorial_arms(config)
    a = config['analysis']
    cues = [0, 1, 2]
    n = len(seeds)
    draws = np.random.default_rng(a['analysis_seed']).integers(0, n, (a['bootstrap_resamples'], n))
    chunk_batches = config['chunk'] // config['batch_size']
    verification = dict(stream_identical_across_arms=True, exposure_prefix_bit_identical={},
                        replay_capacity_honoured=True, initialisation_shared=True, oracle_online_calibration={})
    oracle, O = {}, {}
    for seed in seeds:
        with np.load(output / 'oracle' / f'seed_{seed}.npz') as z:
            oracle[seed] = {k: z[k] for k in z.files}
        kb = config['oracle']['endpoint_kb']
        O[seed] = dict(exact=oracle_endpoint(oracle[seed], cues, kb, 'exact'))
        if config['oracle']['gauge_sensitivity']:
            O[seed]['gauge'] = oracle_endpoint(oracle[seed], cues, kb, 'gauge')
        o_online = monotone(oracle[seed]['online'])
        truth = oracle[seed]['online_truth'].astype(np.float64)
        p = oracle[seed]['online']
        z = (truth.sum(axis=0) - p.sum(axis=0)) / np.sqrt((p * (1 - p)).sum(axis=0) + 1e-12)
        verification['oracle_online_calibration'][str(seed)] = dict(max_abs_z=float(np.abs(z).max()),
                                                                     z_by_action_horizon=z.round(3).tolist())
        oracle[seed]['online_case_brier'] = case_brier(o_online, oracle[seed]['online_truth'])
        O[seed]['online_brier_all'] = float(oracle[seed]['online_case_brier'].mean())
        O[seed]['online_mc_variance_bias'] = float((p * (1 - p)).mean() / config['oracle']['online_samples'])

    # ---- per-fit metrics
    fits, online, costs = {}, {}, {}
    for seed in seeds:
        stream = [str(x) for x in oracle[seed]['online_fingerprints']]
        for arm in arms:
            directory = fit_dir(output, seed, arm)
            record = verify_fit(directory, identity)
            spec = record['spec']
            require(record['fingerprints'] == stream[:len(record['fingerprints'])], 'arrival stream differs across arms')
            require(record['scores'][-1]['memory_packets'] <= spec['R'] and record['diagnostics']['packets_seen']
                    == spec['arrivals'] // config['batch_size'], 'replay capacity or stream length wrong')
            records, arrays = load_trace(directory)
            entry = {}
            for score in record['scores']:
                metrics = endpoint_metrics(records, arrays, score['probes'], oracle[seed], cues)
                o = O[seed]['exact']
                metrics.update(O_brier=o['brier'], excess=metrics['brier'] - o['brier'],
                               O_survival=o['survival'], survival_regret=o['survival'] - metrics['survival'],
                               O_cue_benefit=o['cue_benefit'],
                               cue_benefit_minus_O={c: metrics['cue_benefit'][c] - o['cue_benefit'][c] for c in o['cue_benefit']},
                               marginal_gain=metrics['marginal_brier'] - metrics['brier'])
                if 'gauge' in O[seed]:
                    metrics['excess_vs_gauge_O'] = metrics['brier'] - O[seed]['gauge']['brier']
                entry[score['arrivals']] = metrics
            arrays.close()
            with np.load(directory / 'training.npz') as t:
                probabilities, truth = t['probabilities'], t['truth']
            require(np.array_equal(truth, oracle[seed]['online_truth'][:len(truth)]), 'online truth mismatch')
            learner_case = case_brier(probabilities, truth)
            o_case = oracle[seed]['online_case_brier'][:len(truth)]
            per = len(truth) // (chunk_batches * config['batch_size'])
            learner_chunks = learner_case.reshape(per, -1).mean(axis=1)
            o_chunks = o_case.reshape(per, -1).mean(axis=1)
            online[seed, arm] = dict(learner=learner_chunks.tolist(), O=o_chunks.tolist(),
                                     excess=(learner_chunks - o_chunks).tolist())
            fits[seed, arm] = dict(record=record, scores=entry)
            costs[seed, arm] = dict(elapsed_seconds=record['elapsed_seconds'], training_seconds=record['training_seconds'],
                                    total_macs=record['work']['total_macs'], training_macs=record['work']['training_macs'],
                                    parameters=record['parameters'], memory_bytes=record['memory_bytes']['total'],
                                    peak_replay_bytes=record['memory_bytes']['peak_replay'],
                                    optimizer_steps=record['optimizer_steps'])
        # exposure prefix identity and shared initialisation
        ref, exp = fits[seed, REFERENCE]['record'], fits[seed, 'exposure']['record']
        with np.load(fit_dir(output, seed, REFERENCE) / 'training.npz') as r_, \
                np.load(fit_dir(output, seed, 'exposure') / 'training.npz') as e_:
            prefix = all(np.array_equal(r_[k], e_[k][:len(r_[k])]) for k in r_.files)
        same = (ref['scores'][0]['signature'] == exp['scores'][0]['signature']
                and ref['scores'][0]['model_sha256'] == exp['scores'][0]['model_sha256']
                and fits[seed, REFERENCE]['scores'][config['arrivals']] == fits[seed, 'exposure']['scores'][config['arrivals']]
                and prefix)
        verification['exposure_prefix_bit_identical'][str(seed)] = bool(same)
        require(same, f'exposure arm does not reproduce the reference at {config["arrivals"]} (seed {seed})')
        for width in config['factors']['W']:
            hashes = {fits[seed, arm]['record']['initial_model_sha256'] for arm in arms
                      if fits[seed, arm]['record']['spec']['W'] == width}
            require(len(hashes) == 1, 'initialisation not shared')

    def excess(arm, arrivals=None):
        arrivals = arrivals or config['arrivals']
        return np.array([fits[s, arm]['scores'][arrivals]['excess'] for s in seeds])

    def metric(arm, key, arrivals=None):
        arrivals = arrivals or config['arrivals']
        return np.array([fits[s, arm]['scores'][arrivals][key] for s in seeds])

    ref_excess = excess(REFERENCE)
    ref_mean = float(ref_excess.mean())
    # ---- arm table
    arm_rows = {}
    points = [(arm, config['arrivals']) for arm in arms] + [('exposure', config['exposure_arrivals'])]
    for arm, arrivals in points:
        key = arm if not (arm == 'exposure' and arrivals == config['exposure_arrivals']) else 'exposure@final'
        if arm == 'exposure' and arrivals == config['arrivals']:
            key = 'exposure@arrivals'
        values = excess(arm, arrivals)
        spec = fits[seeds[0], arm]['record']['spec']
        arm_rows[key] = dict(
            arm=arm, arrivals=arrivals, U=spec['U'], R=spec['R'], W=spec['W'], privileged=spec['all_actions'],
            excess=dict(mean=float(values.mean()), per_seed=dict(zip(map(str, seeds), values.tolist())),
                        bootstrap95=bootstrap(values, draws)),
            brier=float(metric(arm, 'brier', arrivals).mean()), O_brier=float(metric(arm, 'O_brier', arrivals).mean()),
            fraction_of_reference_excess_closed=float((ref_excess - values).mean() / ref_mean),
            survival=float(metric(arm, 'survival', arrivals).mean()),
            survival_regret=dict(mean=float(metric(arm, 'survival_regret', arrivals).mean()),
                                 per_seed=dict(zip(map(str, seeds), metric(arm, 'survival_regret', arrivals).tolist()))),
            marginal_gain=float(metric(arm, 'marginal_gain', arrivals).mean()),
            cue_benefit={str(c): float(np.mean([fits[s, arm]['scores'][arrivals]['cue_benefit'][str(c)] for s in seeds])) for c in cues},
            O_cue_benefit={str(c): float(np.mean([O[s]['exact']['cue_benefit'][str(c)] for s in seeds])) for c in cues},
            cue_benefit_per_seed={str(c): [fits[s, arm]['scores'][arrivals]['cue_benefit'][str(c)] for s in seeds] for c in cues},
        )
        if config['oracle']['gauge_sensitivity']:
            arm_rows[key]['excess_vs_gauge_O'] = float(np.mean([fits[s, arm]['scores'][arrivals]['excess_vs_gauge_O'] for s in seeds]))
    # ---- factor main effects (non-reference level minus reference level)
    def cell(u, r, w):
        return excess(f'U{u}_R{r}_W{w}')
    F = config['factors']
    effects = {}
    for factor, levels in F.items():
        low, high = levels                                   # levels[0] is the reference level
        others = [k for k in F if k != factor]
        diffs = []
        for x in F[others[0]]:
            for y in F[others[1]]:
                values = {factor: None, others[0]: x, others[1]: y}
                values[factor] = high
                hi = cell(values['U'], values['R'], values['W'])
                values[factor] = low
                lo = cell(values['U'], values['R'], values['W'])
                diffs.append(hi - lo)
        effect = np.mean(diffs, axis=0)                      # per seed
        reduces = int((effect < 0).sum())
        increases = int((effect > 0).sum())
        mean = float(effect.mean())
        closes = (-mean >= a['matter_fraction'] * ref_mean) and reduces >= a['consistency_seeds']
        widens = (mean >= a['matter_fraction'] * ref_mean) and increases >= a['consistency_seeds']
        protocol_sign = -1. if factor == 'U' else 1.         # high-minus-low with high = larger level
        effects[factor] = dict(
            reference_level=low, other_level=high,
            effect_other_minus_reference=dict(mean=mean, per_seed=dict(zip(map(str, seeds), effect.tolist())),
                                              bootstrap95=bootstrap(effect, draws)),
            protocol_high_minus_low=dict(high=max(levels), low=min(levels), mean=protocol_sign * mean,
                                         per_seed=dict(zip(map(str, seeds), (protocol_sign * effect).tolist()))),
            fraction_of_reference_excess_removed=-mean / ref_mean,
            fraction_bootstrap95=ratio_bootstrap(-effect, ref_excess, draws),
            seeds_reducing_excess=reduces, seeds_increasing_excess=increases,
            matters=bool(closes),
            verdict=('MATTERS: moving from the reference level removes >= 25% of the reference excess in >= 5/6 seeds'
                     if closes else
                     'does not close the gap; the reference level is better by >= 25% of the reference excess in >= 5/6 seeds (descriptive)'
                     if widens else 'does not matter by the pre-declared rule'))
    # ---- descriptive interactions
    interactions = {}
    names = list(F)
    for i in range(3):
        for j in range(i + 1, 3):
            fi, fj = names[i], names[j]
            fk = [k for k in names if k not in (fi, fj)][0]
            values = []
            for z in F[fk]:
                def c(vi, vj):
                    d = {fi: vi, fj: vj, fk: z}
                    return cell(d['U'], d['R'], d['W'])
                (i0, i1), (j0, j1) = F[fi], F[fj]
                values.append((c(i1, j1) - c(i0, j1)) - (c(i1, j0) - c(i0, j0)))
            v = np.mean(values, axis=0)
            interactions[f'{fi}x{fj}'] = dict(mean=float(v.mean()), per_seed=dict(zip(map(str, seeds), v.tolist())),
                                              bootstrap95=bootstrap(v, draws))
    # ---- exposure reading
    exp_final = excess('exposure', config['exposure_arrivals'])
    gain = ref_excess - exp_final
    exposure = dict(
        excess_at_arrivals=dict(mean=ref_mean, per_seed=dict(zip(map(str, seeds), ref_excess.tolist()))),
        excess_at_final=dict(mean=float(exp_final.mean()), per_seed=dict(zip(map(str, seeds), exp_final.tolist())),
                             bootstrap95=bootstrap(exp_final, draws)),
        reduction=dict(mean=float(gain.mean()), per_seed=dict(zip(map(str, seeds), gain.tolist())),
                       bootstrap95=bootstrap(gain, draws)),
        fraction_closed=float(gain.mean() / ref_mean), fraction_bootstrap95=ratio_bootstrap(gain, ref_excess, draws),
        seeds_reduced=int((gain > 0).sum()),
        sample_shortage_contributes=bool(gain.mean() >= a['matter_fraction'] * ref_mean),
        prefix_bit_identical_all_seeds=all(verification['exposure_prefix_bit_identical'].values()))
    exposure['verdict'] = ('a shortage of samples contributes to the gap (>= 25% closed by the final arrival count)'
                           if exposure['sample_shortage_contributes'] else
                           'the gap is a steady state (< 25% of the reference excess closed by the final arrival count)')
    # ---- all-action reading
    all_excess = excess('all_action')
    closed = ref_excess - all_excess
    all_action = dict(label=PRIVILEGED_LABEL,
                      excess=dict(mean=float(all_excess.mean()), per_seed=dict(zip(map(str, seeds), all_excess.tolist())),
                                  bootstrap95=bootstrap(all_excess, draws)),
                      closed=dict(mean=float(closed.mean()), per_seed=dict(zip(map(str, seeds), closed.tolist()))),
                      fraction_closed=float(closed.mean() / ref_mean),
                      fraction_closed_per_seed=dict(zip(map(str, seeds), (closed / ref_excess).tolist())),
                      fraction_bootstrap95=ratio_bootstrap(closed, ref_excess, draws),
                      seeds_improved=int((closed > 0).sum()))
    # ---- competent learner (declared candidate arms; the privileged arm is reported separately)
    candidates = {k: v['excess']['mean'] for k, v in arm_rows.items() if not v['privileged']}
    competent = sorted(k for k, v in candidates.items() if v <= a['competent_excess'])
    competence = dict(threshold=a['competent_excess'], candidate_arm_excess=candidates, competent_arms=competent,
                      best_candidate=min(candidates, key=candidates.get),
                      privileged_all_action_excess=arm_rows['all_action']['excess']['mean'],
                      privileged_all_action_within_threshold=bool(arm_rows['all_action']['excess']['mean'] <= a['competent_excess']),
                      verdict=('competent base learner found: ' + ', '.join(competent)) if competent else
                      'no declared arm reaches excess <= .010: the gap is characterised, not solved')
    ranking = sorted(((k, v['excess']['mean']) for k, v in arm_rows.items()), key=lambda kv: kv[1])
    # ---- online curves
    online_summary = {}
    for arm in arms:
        curves = np.array([online[s, arm]['excess'] for s in seeds])
        online_summary[arm] = dict(chunk_arrivals=[(i + 1) * config['chunk'] for i in range(curves.shape[1])],
                                   mean_excess=curves.mean(axis=0).tolist(),
                                   mean_learner=np.mean([online[s, arm]['learner'] for s in seeds], axis=0).tolist(),
                                   mean_O=np.mean([online[s, arm]['O'] for s in seeds], axis=0).tolist(),
                                   per_seed={str(s): online[s, arm]['excess'] for s in seeds})
    # ---- costs
    cost_summary = {}
    for arm in arms:
        rows = [costs[s, arm] for s in seeds]
        cost_summary[arm] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    oracle_summary = {str(s): dict(exact=O[s]['exact'], gauge=O[s].get('gauge'), online_brier_all=O[s]['online_brier_all'],
                                   online_mc_variance_bias=O[s]['online_mc_variance_bias']) for s in seeds}
    oracle_summary['mean_brier'] = float(np.mean([O[s]['exact']['brier'] for s in seeds]))
    oracle_summary['mean_survival'] = float(np.mean([O[s]['exact']['survival'] for s in seeds]))
    results = dict(label=LABEL, protocol=config['protocol'], identity=identity, config=config,
                   manifest_locked_utc=manifest.get('locked_utc'), analysed_utc=now(),
                   sign_convention=('main effects are reported as other level minus reference level '
                                    '(U3-U12, R256-R16, W256-W64); negative = reduces excess. protocol_high_minus_low '
                                    'gives the literal high-minus-low (U12-U3, R256-R16, W256-W64).'),
                   verification=verification, oracle=oracle_summary, reference_excess_mean=ref_mean,
                   arms=arm_rows, main_effects=effects, interactions_descriptive=interactions,
                   exposure=exposure, all_action=all_action, competence=competence,
                   ranking_descriptive=ranking, online=online_summary, costs=cost_summary,
                   bootstrap=dict(resamples=a['bootstrap_resamples'], seed=a['analysis_seed'],
                                  note='paired seed resamples; descriptive only'))
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / 'results.json', results)
    with (report / 'per_seed_arm.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'arm', 'arrivals', 'U', 'R', 'W', 'privileged', 'learner_brier', 'O_brier', 'excess',
                         'survival', 'O_survival', 'survival_regret', 'marginal_gain', 'cue0_benefit', 'cue1_benefit',
                         'cue2_benefit', 'O_cue0_benefit', 'O_cue1_benefit', 'O_cue2_benefit', 'elapsed_seconds',
                         'training_seconds', 'total_macs', 'parameters', 'memory_bytes', 'optimizer_steps'])
        for seed in seeds:
            for arm in arms:
                for arrivals, m in fits[seed, arm]['scores'].items():
                    spec = fits[seed, arm]['record']['spec']
                    c = costs[seed, arm]
                    writer.writerow([seed, arm, arrivals, spec['U'], spec['R'], spec['W'], spec['all_actions'],
                                     m['brier'], m['O_brier'], m['excess'], m['survival'], m['O_survival'],
                                     m['survival_regret'], m['marginal_gain'],
                                     *[m['cue_benefit'][str(k)] for k in cues], *[m['O_cue_benefit'][str(k)] for k in cues],
                                     c['elapsed_seconds'], c['training_seconds'], c['total_macs'], c['parameters'],
                                     c['memory_bytes'], c['optimizer_steps']])
    with (report / 'online_excess.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'arm', 'chunk', 'chunk_end_arrivals', 'learner_brier', 'O_brier', 'excess'])
        for seed in seeds:
            for arm in arms:
                row = online[seed, arm]
                for i, (lb, ob, ex) in enumerate(zip(row['learner'], row['O'], row['excess'])):
                    writer.writerow([seed, arm, i, (i + 1) * config['chunk'], lb, ob, ex])
    plot_online(online_summary, config, report / 'online_excess.png')
    return results


def plot_online(summary, config, path):
    """Small multiples, one shared y-axis: hue = U (blue U12, orange U3), dash = R
    (solid R16, dashed R256), one panel per width; a third panel shows the
    exposure arm (the reference continued) and the privileged all-action arm."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None
    blue, orange, aqua, ink, muted = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e'
    threshold = config['analysis']['competent_excess']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.6]))
    for ax, width in zip(axes[:2], config['factors']['W']):
        for u in config['factors']['U']:
            for r in config['factors']['R']:
                arm = f'U{u}_R{r}_W{width}'
                row = summary[arm]
                ax.plot(row['chunk_arrivals'], row['mean_excess'], color=blue if u == 12 else orange,
                        ls='-' if r == 16 else '--', lw=2, marker='o', ms=4,
                        label=f'U{u} R{r}' + (' (reference)' if arm == REFERENCE else ''))
        ax.set_title(f'Factorial arms, W{width}', fontsize=10, color=ink)
        ax.set_xlim(0, config['arrivals'] * 1.02)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax = axes[2]
    row = summary['exposure']
    ax.plot(row['chunk_arrivals'], row['mean_excess'], color=blue, lw=2, marker='o', ms=3,
            label='exposure (reference continued to %d)' % config['exposure_arrivals'])
    row = summary['all_action']
    ax.plot(row['chunk_arrivals'], row['mean_excess'], color=aqua, lw=2, marker='s', ms=4,
            label='all-action feedback (PRIVILEGED diagnostic)')
    ax.axvline(config['arrivals'], color=muted, lw=.8, ls=':')
    ax.set_title('Exposure and all-action arms', fontsize=10, color=ink)
    ax.set_xlim(0, config['exposure_arrivals'] * 1.01)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    for ax in axes:
        ax.axhline(threshold, color=muted, lw=1, ls='-.')
        ax.text(ax.get_xlim()[1] * .99, threshold, f'competence threshold {threshold:.3f}', ha='right',
                va='bottom', fontsize=7, color=muted)
        ax.axhline(0, color=muted, lw=.6)
        ax.set_xlabel('arrivals at chunk end', color=muted)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel('online Brier minus oracle O (mean of seeds)', color=muted)
    fig.suptitle(f'Online prequential excess Brier over the law-known oracle, per {config["chunk"]}-arrival chunk',
                 fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


# -------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('check', 'lock', 'run', 'analyze'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.workers is not None:
        raise SystemExit('workers are fixed by the locked config')
    if args.command == 'check':
        print(json.dumps(run_checks(config, args.output)['checks'], indent=1)[:4000])
    elif args.command == 'lock':
        print(json.dumps(lock(config, args.config, args.output)['identity'], indent=1))
    elif args.command == 'run':
        completion = run_study(config, args.config, args.output)
        print(json.dumps(dict(fits=len(completion['fits']), seconds=completion['wall_seconds_this_invocation'])))
    else:
        if not args.report:
            raise SystemExit('--report is required for analyze')
        results = analyze(config, args.config, args.output, args.report)
        print(json.dumps(dict(reference_excess=results['reference_excess_mean'],
                              effects={k: v['verdict'] for k, v in results['main_effects'].items()},
                              exposure=results['exposure']['verdict'], competence=results['competence']['verdict']),
                         indent=1))


if __name__ == '__main__':
    main()
