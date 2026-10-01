"""Step B2: the achievable competence ceiling and a decomposition of the gap.

Implements docs/competence_ceiling_protocol.md (prospective, written 2026-09-29
before any fitting). One stationary law ``stage_law(seed, 3)`` per seed, seeds
19301-19312. Step B's implementation (scripts/competence_gap.py) is imported,
never edited: the arrival stream ``stream_batch`` (channel
``competence_gap_training``), the learner settings, the endpoint panel and
probes (``TraceEvaluator`` + ``endpoint_probes``), the oracle tasks and the
analysis helpers.

Arms
* Online (single pass over ``online.arrivals`` arrivals, exactly Step B's
  ``run_fit``): ON-ref (U12 R16 W64) and ON-R256 (U12 R256 W64). Each fit also
  tracks a weight EMA (decay per optimiser step, initialised at the initial
  weights, never used in training; scripts/competence_ceiling_learners.py) and
  is scored with both the online weights (ON-ref, ON-R256) and the EMA weights
  (ON-ref-EMA, ON-R256-EMA), at the endpoint and prequentially.
* OFF-pixel, for each N: the same HistoryNetwork and the same initialisation
  as ON-ref, trained offline on the first N records of the seed's stream
  (80/20 split by packet, one causal packet per minibatch, Adam .002 with
  per-step cosine decay to 0 over 40 epochs, gradient clip 5, performed-action
  BCE, best-validation weights).
* OFF-feature / OFF-feature-all, for each N: PRIVILEGED ceilings. Ensembles of
  5 MLPs on exact latent features with performed-action (OFF-feature) or all
  five actions' counterfactual labels (OFF-feature-all).
* O: the law-known MC oracle on the endpoint panel (exact reserves only) and on
  every online training case.

Usage (never writes bytecode):
  set PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src, then
  .venv/Scripts/python.exe -B scripts/competence_ceiling.py check   --config C --output O
  .venv/Scripts/python.exe -B scripts/competence_ceiling.py lock    --config C --output O
  .venv/Scripts/python.exe -B scripts/competence_ceiling.py run     --config C --output O
  .venv/Scripts/python.exe -B scripts/competence_ceiling.py replicate --config C --output O
  .venv/Scripts/python.exe -B scripts/competence_ceiling.py analyze --config C --output O --report R
``replicate`` is an INTEGRITY ADDITION, NOT IN THE PROTOCOL (added before the main lock): it re-verifies
every main-run fit against its stored hashes and re-fits ON-ref, OFF-pixel and OFF-feature at N = readings_at
for seeds 19301, 19306, 19311 into O/replicate, comparing everything bit-for-bit; a mismatch stops the study.
With a smoke config, ``check`` also locks, runs (twice: the second invocation
must only verify) and analyses the smoke study into ``O`` and ``O/analysis``.
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
import hashlib  # noqa: E402
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
from torch import nn  # noqa: E402

import competence_gap as G  # noqa: E402
import competence_ceiling_learners as L  # noqa: E402
import ideal_observer_mc as MC  # noqa: E402
from acp_cl.acquisition.world import AcquisitionWorld  # noqa: E402
from acp_cl.core_residual.evaluation import TraceEvaluator, recompute_metrics  # noqa: E402
from acp_cl.persistence.learner import state_hash  # noqa: E402
from acp_cl.persistence.study import digest, trial_seed, write_json  # noqa: E402
from acp_cl.predictive_value.mechanism import _value_hash, predict_all  # noqa: E402
from acp_cl.predictive_value.study import load as load_checkpoint, save as save_checkpoint  # noqa: E402
from acp_cl.replay_renewal.study import atomic_checkpoint  # noqa: E402
from acp_cl.representation_learning.learner import RepresentationLearner, learner_signature  # noqa: E402
from acp_cl.representation_learning.study import endpoint_probes, marginal_probabilities, update_marginal  # noqa: E402

LABEL = ('PROSPECTIVE STUDY under docs/competence_ceiling_protocol.md (Step B2). Oracle O is evaluator-only; '
         'OFF-feature and OFF-feature-all are PRIVILEGED CEILINGS, never candidate learners.')
ONLINE_ARMS = ('ON-ref', 'ON-R256')
EMA_ARMS = {'ON-ref': 'ON-ref-EMA', 'ON-R256': 'ON-R256-EMA'}
OFFLINE_ARMS = ('OFF-pixel', 'OFF-feature', 'OFF-feature-all')
FEATURE_VARIANTS = {'OFF-feature': 'performed', 'OFF-feature-all': 'all'}
PRIVILEGED = {'OFF-feature': L.FEATURE_LABEL, 'OFF-feature-all': L.FEATURE_ALL_LABEL}
FEATURE_INIT_CHANNEL = 'competence_ceiling_feature_init'
SCRIPT_FILES = ('scripts/competence_ceiling.py', 'scripts/competence_ceiling_learners.py',
                'scripts/competence_gap.py', 'scripts/competence_gap_learner.py', 'scripts/ideal_observer_mc.py')
MAIN_SEEDS = list(range(19301, 19313))
CUES = [0, 1, 2]


class Interrupted(RuntimeError):
    """Deliberate interruption used only by the resumption checks."""


sha, read_json, now, require, save_npz, log_event = G.sha, G.read_json, G.now, G.require, G.save_npz, G.log_event


# ------------------------------------------------------------------ config

def validate_config(c):
    expected = {'study', 'protocol', 'smoke', 'seeds', 'stage', 'batch_size', 'eval_size', 'support_replicates',
                'context_width', 'experts', 'interaction_features', 'lr', 'evidence_strength', 'auxiliary_weight',
                'online', 'offline', 'oracle', 'analysis', 'device', 'threads', 'workers', 'cpu_affinity'}
    if set(c) != expected:
        raise ValueError(f'invalid configuration fields: {set(c) ^ expected}')
    b = c['batch_size']
    fixed = dict(stage=3, batch_size=32, eval_size=512, support_replicates=2, context_width=12, experts=4,
                 interaction_features=True, lr=.002, evidence_strength=1., auxiliary_weight=1.)
    if any(c[k] != v for k, v in fixed.items()):
        raise ValueError('reference learner settings and the law are fixed by the protocol')
    if c['device'] != 'cpu' or c['threads'] != 1 or not 1 <= c['workers'] <= 6:
        raise ValueError('CPU, one thread and at most six workers are required')
    on = c['online']
    if set(on) != {'arrivals', 'chunk', 'checkpoint_every', 'ema_decay', 'arms'}:
        raise ValueError('invalid online fields')
    if on['arms'] != {'ON-ref': dict(U=12, R=16, W=64), 'ON-R256': dict(U=12, R=256, W=64)} or on['ema_decay'] != .998:
        raise ValueError('online arms and EMA decay are fixed by the protocol')
    for key in ('arrivals', 'chunk', 'checkpoint_every'):
        if type(on[key]) is not int or on[key] < 1 or on[key] % b:
            raise ValueError(f'online {key} must be a positive multiple of the batch size')
    if on['arrivals'] % on['chunk'] or on['arrivals'] % on['checkpoint_every']:
        raise ValueError('chunks and checkpoints must tile the online stream')
    off = c['offline']
    if set(off) != {'N', 'validation_fraction', 'pixel', 'feature'} or off['validation_fraction'] != .2:
        raise ValueError('invalid offline fields or split')
    if not off['N'] or any(type(n) is not int or n % b or n < 8 * b for n in off['N']) or off['N'] != sorted(set(off['N'])):
        raise ValueError('offline N must be increasing multiples of the batch size')
    p, f = off['pixel'], off['feature']
    if set(p) != {'max_epochs', 'lr', 'schedule', 'grad_clip', 'shuffle_seed'}:
        raise ValueError('invalid OFF-pixel fields')
    if p['lr'] != .002 or p['grad_clip'] != 5. or p['schedule'] != 'cosine_per_step_to_zero':
        raise ValueError('OFF-pixel optimiser is fixed by the protocol')
    if set(f) != {'members', 'hidden', 'layers', 'lr', 'minibatch', 'max_epochs', 'patience', 'weight_decays',
                  'shuffle_seed', 'weight_decay_form'}:
        raise ValueError('invalid OFF-feature fields')
    if (f['members'] != 5 or f['hidden'] != 128 or f['layers'] != 2 or f['lr'] != 1e-3 or f['minibatch'] != 256
            or f['weight_decays'] != [0., 1e-4, 1e-3] or f['weight_decay_form'] != 'adam_l2'):
        raise ValueError('OFF-feature design is fixed by the protocol')
    o = c['oracle']
    if set(o) != {'endpoint_kb', 'endpoint_batches', 'endpoint_channel', 'online_samples', 'online_sub',
                  'online_batches_per_task'}:
        raise ValueError('invalid oracle fields')
    if (on['arrivals'] // b) % o['online_batches_per_task'] or o['online_samples'] % o['online_sub']:
        raise ValueError('online oracle tasks must tile the stream and samples')
    if o['endpoint_channel'] != G.MC_ENDPOINT_EXACT:
        raise ValueError('the endpoint oracle uses the Step B exact channel')
    a = c['analysis']
    if set(a) != {'competent_excess', 'dominant_fraction', 'matter_fraction', 'consistency_seeds', 'readings_at',
                  'bootstrap_resamples', 'analysis_seed'}:
        raise ValueError('invalid analysis fields')
    if (a['competent_excess'] != .010 or a['dominant_fraction'] != .5 or a['matter_fraction'] != .25
            or a['consistency_seeds'] != 10 or a['bootstrap_resamples'] != 20000 or a['analysis_seed'] != 19392026):
        raise ValueError('pre-declared analysis constants are fixed')
    if a['readings_at'] != on['arrivals'] or a['readings_at'] not in off['N']:
        raise ValueError('readings are taken where online arrivals equal an offline N')
    if c['smoke']:
        if set(c['seeds']) & set(MAIN_SEEDS) or any(s < 19300 or s > 19399 for s in c['seeds']):
            raise ValueError('smoke seeds must be fresh and disjoint from the scientific seeds')
    else:
        if (c['seeds'] != MAIN_SEEDS or on['arrivals'] != 8192 or on['chunk'] != 1024
                or off['N'] != [2048, 8192, 32768] or p['max_epochs'] != 40
                or f['max_epochs'] != 400 or f['patience'] != 30):
            raise ValueError('scientific run must use the declared seeds, stream, N values and epochs')
        if o['endpoint_kb'] * o['endpoint_batches'] < 4096 or o['online_samples'] < 1024:
            raise ValueError('oracle sample counts below the declared minimum')


def online_spec(config, arm):
    a = config['online']['arms'][arm]
    return dict(arm=arm, kind='online', U=a['U'], R=a['R'], W=a['W'], arrivals=config['online']['arrivals'],
                score_at=[config['online']['arrivals']], all_actions=False)


def settings_for(config, arm='ON-ref'):
    return G.learner_settings(config, online_spec(config, arm))


def reference_learner(config, seed, arm='ON-ref'):
    """The unchanged project learner; seeds its network with seed + 73019 (shared initialisation)."""
    return RepresentationLearner('outcome', seed, settings_for(config, arm))


def all_tasks(config):
    tasks = [('online', s, a, None) for s in config['seeds'] for a in ONLINE_ARMS]
    tasks += [('pixel', s, 'OFF-pixel', n) for s in config['seeds'] for n in config['offline']['N']]
    tasks += [('feature', s, a, n) for s in config['seeds'] for a in FEATURE_VARIANTS for n in config['offline']['N']]
    return tasks


# ------------------------------------------------------------- lock/manifest

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
    tasks = all_tasks(config)
    manifest = dict(label=LABEL, **payload, locked_utc=now(), online_arms=list(ONLINE_ARMS),
                    ema_arms=list(EMA_ARMS.values()), offline_arms=list(OFFLINE_ARMS),
                    fits=len(tasks), fit_tasks=[list(t) for t in tasks])
    (output / 'protocol_at_lock.md').write_bytes((ROOT / config['protocol']).read_bytes())
    (output / 'config_at_lock.json').write_bytes(Path(config_path).read_bytes())
    with zipfile.ZipFile(output / 'source_at_lock.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(payload['script_files']) | set(payload['source_files'])):
            archive.write(ROOT / name, name)
    manifest['source_zip_sha256'] = sha(output / 'source_at_lock.zip')
    write_json(path, manifest)
    log_event(output, event='locked', identity=payload['identity'])
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


def verify_result(directory, identity):
    record = read_json(Path(directory) / 'result.json')
    if record['identity'] != identity:
        raise ValueError(f'result identity mismatch: {directory}')
    for name, value in record['artifact_hashes'].items():
        if sha(Path(directory) / name) != value:
            raise ValueError(f'artifact changed: {directory}/{name}')
    return record


def task_dir(output, kind, seed, arm, n=None):
    if kind == 'online':
        return Path(output) / 'online' / f'{arm}_{seed}'
    if kind == 'pixel':
        return Path(output) / 'pixel' / f'N{n}_{seed}'
    return Path(output) / 'feature' / f'{FEATURE_VARIANTS[arm]}_N{n}_{seed}'


def model_bytes(model):
    return sum(v.numel() * v.element_size() for v in model.state_dict().values())


# ------------------------------------------------------------- online arms

def run_online(config, identity, seed, arm, output, interrupt_after=None):
    """Step B's run_fit for one online arm, plus a non-interfering weight EMA."""
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = task_dir(output, 'online', seed, arm)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = directory / 'result.json', directory / 'checkpoint.pt'
    if result_path.exists():
        verify_result(directory, identity)
        return dict(kind='online', seed=seed, arm=arm, status='verified_existing')
    spec = online_spec(config, arm)
    law, world, size = G.study_law(config, seed), AcquisitionWorld(), config['batch_size']
    batches, decay = spec['arrivals'] // size, config['online']['ema_decay']
    shadow = reference_learner(config, seed, arm)          # EMA holder, identical initial weights
    resumed = False
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, identity)
        learner, record, online = saved['learner'], saved['record'], saved['online']
        require(record['seed'] == seed and record['arm'] == arm and record['spec'] == spec, 'checkpoint spec changed')
        require(learner_signature(learner) == record['checkpoint_signature'], 'checkpoint learner changed')
        require(state_hash(shadow.model.state_dict()) == record['initial_model_sha256'], 'EMA initialisation changed')
        shadow.model.load_state_dict(saved['ema']['state'])
        require(state_hash(shadow.model.state_dict()) == record['checkpoint_ema_sha256'], 'EMA checkpoint changed')
        ema = L.WeightEMA(learner, shadow, decay, saved['ema']['steps'])
        evaluator = TraceEvaluator(seed, learner.settings, saved['evaluation'])
        ema_evaluator = TraceEvaluator(seed, learner.settings, saved['ema_evaluation'])
        resumed = True
        record['resumes'].append(dict(utc=now(), batches_done=record['batches_done']))
    else:
        learner = reference_learner(config, seed, arm)
        require(state_hash(shadow.model.state_dict()) == state_hash(learner.model.state_dict()),
                'EMA not initialised at the initial weights')
        ema = L.WeightEMA(learner, shadow, decay)
        evaluator, ema_evaluator = TraceEvaluator(seed, learner.settings), TraceEvaluator(seed, learner.settings)
        record = dict(identity=identity, label=LABEL, seed=seed, arm=arm, ema_arm=EMA_ARMS[arm], spec=spec,
                      settings=learner.settings, law=dict(mode=law.mode, active=list(law.active)),
                      ema_decay=decay, learner_class=f'{type(learner).__module__}.{type(learner).__qualname__}',
                      initial_model_sha256=state_hash(learner.model.state_dict()),
                      initial_signature=learner_signature(learner, ignore_walltime=True),
                      batches_done=0, fingerprints=[], marginals={}, scores=[], ema_scores=[], resumes=[],
                      elapsed_seconds=0., started_utc=now())
        online = dict(probabilities=[], ema_probabilities=[], truth=[], actions=[], outcomes=[])
    started = time.perf_counter()
    for index in range(record['batches_done'], batches):
        data, cases, truth = G.stream_batch(world, law, seed, index, size)
        probabilities = predict_all(learner, data.observations, learner.history)
        ema_probabilities = predict_all(shadow, data.observations, learner.history)
        learner.train(data)
        require(ema.steps == learner.cost['optimizer_steps'], 'EMA missed an optimiser step')
        record['fingerprints'].append(data.fingerprint())
        online['probabilities'].append(np.array(probabilities, dtype=np.float32, copy=True))
        online['ema_probabilities'].append(np.array(ema_probabilities, dtype=np.float32, copy=True))
        online['truth'].append(truth.astype(np.uint8))
        online['actions'].append(data.actions.copy())
        online['outcomes'].append(data.survival.copy())
        update_marginal(record['marginals'], law, data)
        record['batches_done'] = index + 1
        arrivals = (index + 1) * size
        if arrivals in spec['score_at']:
            before = learner_signature(learner)
            probes = endpoint_probes(evaluator, learner, [asdict(law)], record['marginals'])
            ema_probes = endpoint_probes(ema_evaluator, shadow, [asdict(law)], record['marginals'])
            require(learner_signature(learner) == before, 'endpoint scoring changed the learner')
            record['scores'].append(dict(arrivals=arrivals, probes=probes,
                                         model_sha256=state_hash(learner.model.state_dict()),
                                         signature=learner_signature(learner, ignore_walltime=True),
                                         memory_packets=len(learner.memory.packets),
                                         memory_capacity=learner.memory.capacity,
                                         diagnostics=learner.diagnostics()))
            record['ema_scores'].append(dict(arrivals=arrivals, probes=ema_probes, ema_steps=ema.steps,
                                             model_sha256=state_hash(shadow.model.state_dict())))
        if arrivals % config['online']['checkpoint_every'] == 0 or index + 1 == batches:
            record['elapsed_seconds'] += time.perf_counter() - started
            started = time.perf_counter()
            record['checkpoint_signature'] = learner_signature(learner)
            record['checkpoint_ema_sha256'] = state_hash(shadow.model.state_dict())
            save_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record,
                                             evaluation=evaluator.snapshot(), ema_evaluation=ema_evaluator.snapshot(),
                                             ema=dict(state=ema.state(), steps=ema.steps, decay=decay), online=online))
        if interrupt_after is not None and index + 1 == interrupt_after:
            raise Interrupted(f'deliberate interruption after batch {index + 1}')
    ema.remove()
    require(len(learner.memory.packets) == min(spec['R'], batches), 'replay capacity not honoured')
    diagnostics = learner.diagnostics()
    record['diagnostics'] = diagnostics
    record['final_model_sha256'] = state_hash(learner.model.state_dict())
    record['ema_model_sha256'] = state_hash(shadow.model.state_dict())
    record['ema_steps'] = ema.steps
    record['end_signature'] = learner_signature(learner, ignore_walltime=True)
    record['evaluation_files'] = evaluator.export(directory)
    record['ema_evaluation_files'] = ema_evaluator.export(directory / 'ema')
    save_npz(directory / 'training.npz', {k: np.concatenate(v) for k, v in online.items()})
    work = G.work_proxy(config, spec, batches, len(evaluator.records))
    packet = work['packet_forward_macs']
    params = diagnostics['parameters']
    work.update(ema_forecast_macs=packet * batches,
                ema_evaluation_macs=G.forward_macs(spec['W'], config['context_width'], config['eval_size'])
                * len(ema_evaluator.records),
                ema_update_flops=3 * params * ema.steps)
    work['total_macs_including_ema'] = work['total_macs'] + work['ema_forecast_macs'] + work['ema_evaluation_macs']
    record['work'] = work
    record['memory_bytes'] = dict(model=diagnostics['model_bytes'], optimizer=diagnostics['optimizer_bytes'],
                                  peak_replay=diagnostics['peak_replay_bytes'], history=diagnostics['history_bytes'],
                                  ema=model_bytes(shadow.model),
                                  total=diagnostics['model_bytes'] + diagnostics['optimizer_bytes']
                                  + diagnostics['peak_replay_bytes'] + diagnostics['history_bytes']
                                  + model_bytes(shadow.model))
    record['parameters'] = params
    record['optimizer_steps'] = diagnostics['cost']['optimizer_steps']
    record['training_seconds'] = diagnostics['cost']['training_seconds']
    record['resumed'] = resumed
    record['elapsed_seconds'] += time.perf_counter() - started
    record['completed_utc'] = now()
    record['artifact_hashes'] = {name: sha(directory / name) for name in
                                 ('evaluations.json', 'evaluations.npz', 'ema/evaluations.json',
                                  'ema/evaluations.npz', 'training.npz')}
    record['checkpoint_signature'] = learner_signature(learner)
    record['checkpoint_ema_sha256'] = state_hash(shadow.model.state_dict())
    save_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record,
                                     evaluation=evaluator.snapshot(), ema_evaluation=ema_evaluator.snapshot(),
                                     ema=dict(state=ema.state(), steps=ema.steps, decay=decay), online=online))
    record['artifact_hashes']['checkpoint.pt'] = sha(checkpoint)
    write_json(result_path, record)
    return dict(kind='online', seed=seed, arm=arm, status='completed', seconds=record['elapsed_seconds'])


# ------------------------------------------------------------ offline data

def offline_stream(config, seed, n):
    """The first n records of the seed's arrival stream: stream_batch(world, law, seed, i, 32), i < n/32."""
    law, world, size = G.study_law(config, seed), AcquisitionWorld(), config['batch_size']
    return law, [G.stream_batch(world, law, seed, i, size) for i in range(n // size)]


def panel_image_hashes(config, seed, law):
    evaluator = TraceEvaluator(seed, settings_for(config))
    cases, _ = evaluator.dataset(law)
    hashes = {hashlib.sha256(o.tobytes()).hexdigest() for o in cases.observations}
    support = {hashlib.sha256(o.tobytes()).hexdigest() for r in range(config['support_replicates'])
               for o in evaluator.support(law, r).observations}
    return hashes, support


def disjointness(config, seed, law, stream_data):
    panel, support = panel_image_hashes(config, seed, law)
    train = {hashlib.sha256(o.tobytes()).hexdigest() for d in stream_data for o in d.observations}
    shared_panel, shared_support = len(train & panel), len(train & support)
    require(not shared_panel and not shared_support, 'training records overlap the endpoint panel or supports')
    return dict(training_images=len(train), panel_images=len(panel), support_images=len(support),
                shared_with_panel=shared_panel, shared_with_supports=shared_support)


# ------------------------------------------------------------- OFF-pixel

def run_pixel(config, identity, seed, n, output, interrupt_after_epoch=None):
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = task_dir(output, 'pixel', seed, 'OFF-pixel', n)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = directory / 'result.json', directory / 'checkpoint.pt'
    if result_path.exists():
        verify_result(directory, identity)
        return dict(kind='pixel', seed=seed, arm='OFF-pixel', N=n, status='verified_existing')
    p = config['offline']['pixel']
    settings = settings_for(config)
    law, parts = offline_stream(config, seed, n)
    stream = [d for d, _, _ in parts]
    fingerprints = [d.fingerprint() for d in stream]
    packets = L.offline_packets(stream)
    n_train, n_val = L.split_packets(packets, config['offline']['validation_fraction'])
    learner = reference_learner(config, seed)
    initial = state_hash(learner.model.state_dict())
    optimizer = torch.optim.Adam(learner.model.parameters(), lr=p['lr'])
    total_steps = p['max_epochs'] * n_train
    started = time.perf_counter()
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, identity)
        state = saved['state']
        require(state['fingerprints'] == fingerprints and state['initial_model_sha256'] == initial,
                'OFF-pixel checkpoint data or initialisation changed')
        learner.model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        require(_value_hash(optimizer.state_dict()) == state['optimizer_hash'], 'optimizer checkpoint changed')
        best_model = saved['best_model']
        state['resumes'].append(dict(utc=now(), epochs_done=state['epochs_done']))
    else:
        best_model = None
        state = dict(epochs_done=0, best_validation=None, best_epoch=0, log=[], resumes=[], elapsed_seconds=0.,
                     fingerprints=fingerprints, initial_model_sha256=initial, optimizer_steps=0)
    for epoch in range(state['epochs_done'] + 1, p['max_epochs'] + 1):
        order = np.random.default_rng([p['shuffle_seed'], seed, n, epoch]).permutation(n_train)
        learner.model.train()
        losses, lr = [], None
        for j, k in enumerate(order):
            lr = L.cosine_lr(p['lr'], (epoch - 1) * n_train + j, total_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            loss = L.packet_loss(learner, packets[int(k)])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(learner.model.parameters(), p['grad_clip'])
            optimizer.step()
            losses.append(float(loss.detach()))
        state['optimizer_steps'] += len(order)
        val_loss, val_brier = L.packet_validation(learner, packets[n_train:])
        improved = state['best_validation'] is None or val_loss < state['best_validation']
        if improved:
            state['best_validation'], state['best_epoch'] = val_loss, epoch
            best_model = {k: v.detach().clone() for k, v in learner.model.state_dict().items()}
        state['log'].append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_loss=val_loss,
                                 validation_performed_brier=val_brier, last_lr=lr, improved=bool(improved),
                                 model_sha256=state_hash(learner.model.state_dict())))
        state['epochs_done'] = epoch
        state['elapsed_seconds'] += time.perf_counter() - started
        started = time.perf_counter()
        state['optimizer_hash'] = _value_hash(optimizer.state_dict())
        atomic_checkpoint(checkpoint, dict(identity=identity, state=state, model=learner.model.state_dict(),
                                           optimizer=optimizer.state_dict(), best_model=best_model,
                                           saved_gradients=[]))
        if interrupt_after_epoch is not None and epoch == interrupt_after_epoch:
            raise Interrupted(f'deliberate interruption after epoch {epoch}')
    scorer = reference_learner(config, seed)
    require(state_hash(scorer.model.state_dict()) == initial, 'OFF-pixel initialisation not reproducible')
    scorer.model.load_state_dict(best_model)
    marginals = {}
    for d in stream:
        update_marginal(marginals, law, d)
    evaluator = TraceEvaluator(seed, scorer.settings)
    probes = endpoint_probes(evaluator, scorer, [asdict(law)], marginals)
    record = dict(identity=identity, label=LABEL, seed=seed, arm='OFF-pixel', N=n, settings=settings,
                  law=dict(mode=law.mode, active=list(law.active)), training=dict(p, validation_fraction=
                  config['offline']['validation_fraction'], train_packets=n_train, validation_packets=n_val,
                  train_records=n_train * config['batch_size'], validation_records=n_val * config['batch_size'],
                  first_packet_support=None, minibatch='one causal 32-record packet with its preceding batch as support'),
                  initial_model_sha256=initial, best_model_sha256=state_hash(scorer.model.state_dict()),
                  final_model_sha256=state_hash(learner.model.state_dict()), best_epoch=state['best_epoch'],
                  best_validation_loss=state['best_validation'], epochs=state['log'], resumes=state['resumes'],
                  fingerprints=fingerprints, marginals=marginals, probes=probes,
                  disjointness=disjointness(config, seed, law, stream), optimizer_steps=state['optimizer_steps'])
    record['evaluation_files'] = evaluator.export(directory)
    packet_macs = G.forward_macs(64, config['context_width'], config['batch_size'])
    record['work'] = dict(packet_forward_macs=packet_macs, training_macs=3 * packet_macs * state['optimizer_steps'],
                          validation_macs=packet_macs * n_val * state['epochs_done'],
                          evaluation_macs=G.forward_macs(64, config['context_width'], config['eval_size'])
                          * len(evaluator.records))
    record['work']['total_macs'] = sum(v for k, v in record['work'].items() if k != 'packet_forward_macs')
    data_bytes = sum(d.observations.nbytes + d.actions.nbytes + d.survival.nbytes for d in stream)
    mb = model_bytes(learner.model)
    record['memory_bytes'] = dict(model=mb, optimizer=2 * mb, best_copy=mb, data=data_bytes, total=4 * mb + data_bytes)
    record['parameters'] = sum(q.numel() for q in learner.model.parameters())
    record['resumed'] = bool(state['resumes'])
    record['elapsed_seconds'] = state['elapsed_seconds'] + time.perf_counter() - started
    record['completed_utc'] = now()
    record['artifact_hashes'] = {name: sha(directory / name) for name in
                                 ('evaluations.json', 'evaluations.npz', 'checkpoint.pt')}
    write_json(result_path, record)
    return dict(kind='pixel', seed=seed, arm='OFF-pixel', N=n, status='completed', seconds=record['elapsed_seconds'])


# ------------------------------------------------------------- OFF-feature

def feature_panel(config, seed):
    data = G.endpoint_case_data(config, seed)
    cases = data['cases']
    x = L.latent_features(cases.base.reserves, cases.factors, cases.signals)
    flips = []
    for c in CUES:
        signals = cases.signals.copy()
        signals[:, c] ^= 1
        flips.append(L.latent_features(cases.base.reserves, cases.factors, signals))
    return data, x, flips


def feature_metrics(probabilities, flips, truth, factors, marginal=None):
    everything = np.ones(len(truth), bool)
    m = recompute_metrics(probabilities, truth, everything, everything, marginal)
    benefits = {}
    for c in CUES:
        affected = MC.mask_affected_like(factors, c)
        benefits[str(c)] = (recompute_metrics(flips[c], truth, affected, everything)['focus_brier']
                            - recompute_metrics(probabilities, truth, affected, everything)['focus_brier'])
    return dict(brier=m['brier'], survival=m['survival'], marginal_brier=m.get('marginal_brier'), cue_benefit=benefits)


def run_feature(config, identity, seed, arm, n, output, interrupt_after_member=None):
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = task_dir(output, 'feature', seed, arm, n)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = directory / 'result.json', directory / 'checkpoint.pt'
    if result_path.exists():
        verify_result(directory, identity)
        return dict(kind='feature', seed=seed, arm=arm, N=n, status='verified_existing')
    all_actions = FEATURE_VARIANTS[arm] == 'all'
    f, size = config['offline']['feature'], config['batch_size']
    law, parts = offline_stream(config, seed, n)
    fingerprints = [d.fingerprint() for d, _, _ in parts]
    x = L.latent_features(np.concatenate([c.base.reserves for _, c, _ in parts]),
                          np.concatenate([c.factors for _, c, _ in parts]),
                          np.concatenate([c.signals for _, c, _ in parts]))
    actions = np.concatenate([d.actions for d, _, _ in parts]).astype(np.int64)
    performed = np.concatenate([d.survival for d, _, _ in parts]).astype(np.float32)
    truth = np.concatenate([t for _, _, t in parts]).astype(np.float32)
    require(np.array_equal(truth[np.arange(len(actions)), actions], performed), 'performed labels differ from truth')
    n_train, n_val = L.split_packets(list(range(len(parts))), config['offline']['validation_fraction'])
    train, validation = np.arange(n_train * size), np.arange(n_train * size, n)
    tensors = (torch.as_tensor(x), torch.as_tensor(actions), torch.as_tensor(performed), torch.as_tensor(truth))
    panel, panel_x, panel_flips = feature_panel(config, seed)
    started = time.perf_counter()
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, identity)
        state = saved['state']
        require(state['fingerprints'] == fingerprints, 'OFF-feature checkpoint data changed')
        members = saved['members']
        state['resumes'].append(dict(utc=now(), completed=sorted(members)))
    else:
        members, state = {}, dict(fingerprints=fingerprints, resumes=[], elapsed_seconds=0.)
    shuffle_key = [f['shuffle_seed'], seed, n]
    for wi, wd in enumerate(f['weight_decays']):
        for m in range(f['members']):
            key = f'{wi}_{m}'
            if key in members:
                continue
            init_seed = trial_seed(seed, FEATURE_INIT_CHANNEL, m)
            model = L.make_member(init_seed, f)
            info = L.train_member(model, tensors, train, validation, f, wd, all_actions, shuffle_key)
            members[key] = dict(state={k: v.detach().clone() for k, v in model.state_dict().items()}, info=info,
                                init_seed=init_seed, initial_sha256=state_hash(L.make_member(init_seed, f).state_dict()),
                                weight_decay=wd)
            state['elapsed_seconds'] += time.perf_counter() - started
            started = time.perf_counter()
            atomic_checkpoint(checkpoint, dict(identity=identity, state=state, members=members, saved_gradients=[]))
            if interrupt_after_member is not None and len(members) == interrupt_after_member:
                raise Interrupted(f'deliberate interruption after {len(members)} member fits')
    # ensemble per weight decay: validation loss of the ensemble mean (selection), panel forecasts
    candidates, arrays = [], {}
    xv, av, yv, tv = (t[validation] for t in tensors)
    for wi, wd in enumerate(f['weight_decays']):
        val, pan, flp = [], [], [[] for _ in CUES]
        for m in range(f['members']):
            model = L.make_member(members[f'{wi}_{m}']['init_seed'], f)
            model.load_state_dict(members[f'{wi}_{m}']['state'])
            val.append(L.member_probabilities(model, xv.numpy()))
            pan.append(L.member_probabilities(model, panel_x))
            for c in CUES:
                flp[c].append(L.member_probabilities(model, panel_flips[c]))
        mean_val = torch.as_tensor(np.mean(val, axis=0))
        loss = float(L.feature_objective(mean_val, av, yv.double(), tv.double(), all_actions))
        candidates.append(dict(weight_decay=wd, ensemble_validation_loss=loss,
                               member_validation_loss=[members[f'{wi}_{m}']['info']['best_validation'] for m in range(f['members'])],
                               member_best_epoch=[members[f'{wi}_{m}']['info']['best_epoch'] for m in range(f['members'])],
                               member_epochs_run=[members[f'{wi}_{m}']['info']['epochs_run'] for m in range(f['members'])]))
        arrays[f'wd{wi}_probabilities'] = G.monotone(np.mean(pan, axis=0))
        for c in CUES:
            arrays[f'wd{wi}_flip_{c}'] = G.monotone(np.mean(flp[c], axis=0))
        arrays[f'wd{wi}_member_probabilities'] = np.stack(pan)
    chosen = int(np.argmin([c['ensemble_validation_loss'] for c in candidates]))   # first minimum on ties
    probabilities = arrays[f'wd{chosen}_probabilities']
    flips = [arrays[f'wd{chosen}_flip_{c}'] for c in CUES]
    panel_truth = panel['truth']
    marginal_counts = {}
    for d, _, _ in parts:
        update_marginal(marginal_counts, law, d)
    marginal = marginal_probabilities(next(iter(marginal_counts.values())))
    metrics = feature_metrics(probabilities, flips, panel_truth, panel['cases'].factors, marginal)
    arrays.update(probabilities=probabilities, **{f'flip_{c}': flips[c] for c in CUES},
                  truth=panel_truth.astype(np.uint8), factors=panel['cases'].factors, signals=panel['cases'].signals,
                  panel_features=panel_x, marginal=marginal)
    save_npz(directory / 'panel.npz', arrays)
    steps = sum(mm['info']['optimizer_steps'] for mm in members.values())
    epochs = sum(mm['info']['epochs_run'] for mm in members.values())
    per_record = L.FEATURE_WIDTH * f['hidden'] + f['hidden'] * f['hidden'] * (f['layers'] - 1) + f['hidden'] * 15
    record = dict(identity=identity, label=LABEL, privileged=True, privileged_label=PRIVILEGED[arm], seed=seed, arm=arm,
                  N=n, variant=FEATURE_VARIANTS[arm], law=dict(mode=law.mode, active=list(law.active)),
                  design=dict(f, inputs=['reserve_a/18', 'reserve_b/18', 'source>0', 'lossy', 'delayed',
                                         'signal0', 'signal1', 'signal2'],
                              validation_fraction=config['offline']['validation_fraction'],
                              train_records=len(train), validation_records=len(validation),
                              selection='weight decay with the lowest ensemble-mean validation loss (panel never used)',
                              forecast='mean of member probabilities, then cummin over horizons'),
                  candidates=candidates, chosen_weight_decay=f['weight_decays'][chosen], chosen_index=chosen,
                  members={k: dict(info=v['info'], init_seed=v['init_seed'], initial_sha256=v['initial_sha256'],
                                   weight_decay=v['weight_decay']) for k, v in sorted(members.items())},
                  fingerprints=fingerprints, marginal=marginal.tolist(), metrics=metrics,
                  feature_matrix_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                  label_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
                  panel_query_seed=panel['query_seed'], resumes=state['resumes'],
                  privileged_label_cells=int(len(train) * 15 if all_actions else 0))
    record['work'] = dict(forward_macs_per_record=per_record,
                          training_macs=3 * per_record * sum(mm['info']['epochs_run'] * len(train) for mm in members.values()),
                          validation_macs=per_record * len(validation) * epochs,
                          evaluation_macs=per_record * len(panel_x) * 4 * len(members))
    record['work']['total_macs'] = record['work']['training_macs'] + record['work']['validation_macs'] + record['work']['evaluation_macs']
    params = sum(q.numel() for q in L.FeatureMLP(L.FEATURE_WIDTH, f['hidden'], f['layers']).parameters())
    record['parameters'] = params * f['members']
    record['optimizer_steps'] = steps
    record['memory_bytes'] = dict(models=4 * params * f['members'], optimizer=8 * params, data=int(x.nbytes + truth.nbytes),
                                  total=4 * params * f['members'] + 8 * params + int(x.nbytes + truth.nbytes))
    record['resumed'] = bool(state['resumes'])
    record['elapsed_seconds'] = state['elapsed_seconds'] + time.perf_counter() - started
    record['completed_utc'] = now()
    record['artifact_hashes'] = {name: sha(directory / name) for name in ('panel.npz', 'checkpoint.pt')}
    write_json(result_path, record)
    return dict(kind='feature', seed=seed, arm=arm, N=n, status='completed', seconds=record['elapsed_seconds'])


# ------------------------------------------------------------- the oracle

def oracle_tasks(config, output):
    tasks, o, size = [], config['oracle'], config['batch_size']
    for seed in config['seeds']:
        for b in range(o['endpoint_batches']):
            tasks.append(('endpoint', dict(config=config, seed=seed, gauge=False, channel=o['endpoint_channel'], batch=b,
                                           path=str(Path(output) / 'oracle' / 'endpoint' / f'exact_{seed}_{b:02d}.npz'))))
        for first in range(0, config['online']['arrivals'] // size, o['online_batches_per_task']):
            tasks.append(('online_oracle', dict(config=config, seed=seed, first_batch=first,
                                                batches=o['online_batches_per_task'],
                                                path=str(Path(output) / 'oracle' / 'online' / f'{seed}_{first:05d}.npz'))))
    return tasks


def assemble_oracle(config, output):
    o, size = config['oracle'], config['batch_size']
    summary = {}
    for seed in config['seeds']:
        path = Path(output) / 'oracle' / f'seed_{seed}.npz'
        data = G.endpoint_case_data(config, seed)
        arrays = dict(panel_truth=data['truth'].astype(np.uint8), panel_factors=data['cases'].factors,
                      panel_signals=data['cases'].signals,
                      panel_observations_sha256=np.array(hashlib.sha256(data['cases'].observations.tobytes()).hexdigest()))
        simulations, seconds, counts = 0, 0., []
        for b in range(o['endpoint_batches']):
            with np.load(Path(output) / 'oracle' / 'endpoint' / f'exact_{seed}_{b:02d}.npz') as z:
                counts.append(z['query_counts'].astype(np.int64))
                simulations += int(z['case_simulations'])
                seconds += float(z['seconds'])
                require(int(z['kb']) == o['endpoint_kb'], 'endpoint sample count changed')
        counts = np.stack(counts)
        arrays['endpoint_exact_batch_counts'] = counts
        arrays['endpoint_exact'] = counts.sum(axis=0) / (len(counts) * o['endpoint_kb'])
        pieces = []
        for first in range(0, config['online']['arrivals'] // size, o['online_batches_per_task']):
            with np.load(Path(output) / 'oracle' / 'online' / f'{seed}_{first:05d}.npz') as z:
                pieces.append({k: z[k] for k in ('counts', 'truth', 'factors', 'signals', 'reserves', 'fingerprints')})
                require(int(z['samples']) == o['online_samples'], 'online sample count changed')
                simulations += int(z['counts'].shape[0]) * o['online_samples'] * 5
                seconds += float(z['seconds'])
        for key in ('counts', 'truth', 'factors', 'signals', 'reserves', 'fingerprints'):
            arrays[f'online_{key}'] = np.concatenate([p[key] for p in pieces])
        arrays['online'] = arrays['online_counts'] / o['online_samples']
        digest_ = save_npz(path, arrays)
        summary[str(seed)] = dict(path=path.relative_to(Path(output)).as_posix(), sha256=digest_,
                                  case_simulations=simulations, task_seconds=seconds,
                                  endpoint_verified_cells=data['verified_cells'],
                                  query_seed=data['query_seed'], support_seeds=data['support_seeds'])
    return summary


# --------------------------------------------------------------------- run

def estimated_cost(config, kind, payload):
    """Scheduling weights (seconds) from a timing probe of the main settings; longest tasks start first."""
    if kind == 'online':
        return config['online']['arrivals'] / 8192 * 52
    if kind == 'pixel':
        return payload['n'] / 8192 * 70 * config['offline']['pixel']['max_epochs'] / 40
    if kind == 'feature':
        return (30 + payload['n'] / 32768 * 450) * config['offline']['feature']['max_epochs'] / 400
    if kind == 'online_oracle':
        return payload['batches'] * config['batch_size'] * config['oracle']['online_samples'] * 5 / 6.8e5
    return 5.


def dispatch(kind, payload):
    if kind == 'online':
        return run_online(**payload)
    if kind == 'pixel':
        return run_pixel(**payload)
    if kind == 'feature':
        return run_feature(**payload)
    if kind == 'online_oracle':
        return G.online_oracle_task(payload)
    return G.endpoint_oracle_task(payload)


def run_jobs(config, identity, output):
    jobs = []
    for kind, seed, arm, n in all_tasks(config):
        payload = dict(config=config, identity=identity, seed=seed, output=str(output))
        if kind == 'online':
            payload['arm'] = arm
        elif kind == 'pixel':
            payload['n'] = n
        else:
            payload.update(arm=arm, n=n)
        jobs.append((kind, payload))
    jobs += oracle_tasks(config, output)
    jobs.sort(key=lambda job: -estimated_cost(config, *job))
    return jobs


def run_study(config, config_path, output):
    validate_config(config)
    output = Path(output)
    manifest = check_lock(config, config_path, output)
    identity = manifest['identity']
    log_event(output, event='run_start', identity=identity)
    jobs = run_jobs(config, identity, output)
    started, failures, statuses = time.perf_counter(), [], []
    with ProcessPoolExecutor(max_workers=config['workers'], initializer=G.worker_init,
                             initargs=(config['cpu_affinity'], config['threads'])) as pool:
        futures = {pool.submit(dispatch, kind, payload): (kind, payload) for kind, payload in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            kind, payload = futures[future]
            label = dict(kind=kind, seed=payload['seed'], arm=payload.get('arm'), N=payload.get('n'),
                         part=payload.get('first_batch', payload.get('batch')))
            try:
                result = future.result()
                statuses.append(dict(label, status=result.get('status')))
                log_event(output, event='task_done', **label, status=result.get('status'), seconds=result.get('seconds'))
                if kind in ('online', 'pixel', 'feature'):
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
    for kind, seed, arm, n in all_tasks(config):
        directory = task_dir(output, kind, seed, arm, n)
        record = verify_result(directory, identity)
        fits.append(dict(kind=kind, seed=seed, arm=arm, N=n,
                         path=(directory / 'result.json').relative_to(output).as_posix(),
                         sha256=sha(directory / 'result.json'), elapsed_seconds=record['elapsed_seconds']))
    completion = dict(identity=identity, fits=fits, oracle=oracle, statuses=statuses,
                      wall_seconds_this_invocation=time.perf_counter() - started, completed_utc=now())
    write_json(output / 'completion.json', completion)
    log_event(output, event='run_complete', seconds=time.perf_counter() - started)
    return completion


# --------------------------------------------------------------- analysis

def summary_stats(values, draws):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), bootstrap95=G.bootstrap(values, draws))


def analyze(config, config_path, output, report, require_lock=True):
    validate_config(config)
    output, report = Path(output), Path(report)
    manifest = check_lock(config, config_path, output) if require_lock else read_json(output / 'manifest.json')
    identity = manifest['identity']
    seeds, a = config['seeds'], config['analysis']
    n_seeds, at = len(seeds), a['readings_at']
    draws = np.random.default_rng(a['analysis_seed']).integers(0, n_seeds, (a['bootstrap_resamples'], n_seeds))
    chunk_rows = config['online']['chunk']
    kb = config['oracle']['endpoint_kb']
    oracle, O = {}, {}
    verification = dict(online_stream_equals_oracle_stream=True, offline_data_is_stream_prefix=True,
                        shared_initialisation_on_ref_on_r256_off_pixel=True, ema_initialised_at_initial_weights=True,
                        ema_steps_equal_optimizer_steps=True, disjoint_from_panel=True, oracle_online_calibration={})
    for seed in seeds:
        with np.load(output / 'oracle' / f'seed_{seed}.npz') as z:
            oracle[seed] = {k: z[k] for k in z.files}
        O[seed] = G.oracle_endpoint(oracle[seed], CUES, kb, 'exact')
        p = oracle[seed]['online']
        truth = oracle[seed]['online_truth'].astype(np.float64)
        zval = (truth.sum(axis=0) - p.sum(axis=0)) / np.sqrt((p * (1 - p)).sum(axis=0) + 1e-12)
        verification['oracle_online_calibration'][str(seed)] = dict(max_abs_z=float(np.abs(zval).max()))
        oracle[seed]['online_case_brier'] = G.case_brier(G.monotone(p), oracle[seed]['online_truth'])
        O[seed]['online_brier_all'] = float(oracle[seed]['online_case_brier'].mean())
        O[seed]['online_mc_variance_bias'] = float((p * (1 - p)).mean() / config['oracle']['online_samples'])

    entries, online_curves, costs, records = {}, {}, {}, {}

    def finish(seed, metrics):
        o = O[seed]
        metrics.update(O_brier=o['brier'], excess=metrics['brier'] - o['brier'], O_survival=o['survival'],
                       survival_regret=o['survival'] - metrics['survival'], O_cue_benefit=o['cue_benefit'],
                       cue_benefit_minus_O={c: metrics['cue_benefit'][c] - o['cue_benefit'][c] for c in o['cue_benefit']},
                       marginal_gain=(metrics['marginal_brier'] - metrics['brier'])
                       if metrics.get('marginal_brier') is not None else None)
        return metrics

    for seed in seeds:
        stream = [str(x) for x in oracle[seed]['online_fingerprints']]
        inits = set()
        for arm in ONLINE_ARMS:
            directory = task_dir(output, 'online', seed, arm)
            record = verify_result(directory, identity)
            records[seed, arm] = record
            require(record['fingerprints'] == stream, 'online stream differs from the oracle stream')
            require(record['ema_steps'] == record['optimizer_steps'], 'EMA step count differs')
            require(record['diagnostics']['packets_seen'] == config['online']['arrivals'] // config['batch_size'],
                    'online stream length wrong')
            inits.add(record['initial_model_sha256'])
            for weights, sub, scores, name in (('online', directory, record['scores'], arm),
                                              ('ema', directory / 'ema', record['ema_scores'], EMA_ARMS[arm])):
                recs, arrays = G.load_trace(sub)
                entries[seed, name, at] = finish(seed, G.endpoint_metrics(recs, arrays, scores[-1]['probes'],
                                                                           oracle[seed], CUES))
                arrays.close()
            with np.load(directory / 'training.npz') as t:
                require(np.array_equal(t['truth'], oracle[seed]['online_truth']), 'online truth mismatch')
                for key, name in (('probabilities', arm), ('ema_probabilities', EMA_ARMS[arm])):
                    case = G.case_brier(t[key], t['truth'])
                    o_case = oracle[seed]['online_case_brier']
                    lc = case.reshape(-1, chunk_rows).mean(axis=1)
                    oc = o_case.reshape(-1, chunk_rows).mean(axis=1)
                    online_curves[seed, name] = dict(learner=lc.tolist(), O=oc.tolist(), excess=(lc - oc).tolist())
            c = dict(elapsed_seconds=record['elapsed_seconds'], total_macs=record['work']['total_macs_including_ema'],
                     memory_bytes=record['memory_bytes']['total'], parameters=record['parameters'],
                     optimizer_steps=record['optimizer_steps'])
            costs[seed, arm] = costs[seed, EMA_ARMS[arm]] = c
        for n in config['offline']['N']:
            directory = task_dir(output, 'pixel', seed, 'OFF-pixel', n)
            record = verify_result(directory, identity)
            records[seed, 'OFF-pixel', n] = record
            k = min(len(record['fingerprints']), len(stream))
            require(record['fingerprints'][:k] == stream[:k], 'OFF-pixel data are not the stream prefix')
            require(len(record['fingerprints']) * config['batch_size'] == n, 'OFF-pixel record count wrong')
            inits.add(record['initial_model_sha256'])
            require(record['disjointness']['shared_with_panel'] == 0 and record['disjointness']['shared_with_supports'] == 0,
                    'OFF-pixel training overlaps the panel')
            recs, arrays = G.load_trace(directory)
            entries[seed, 'OFF-pixel', n] = finish(seed, G.endpoint_metrics(recs, arrays, record['probes'], oracle[seed], CUES))
            arrays.close()
            costs[seed, 'OFF-pixel', n] = dict(elapsed_seconds=record['elapsed_seconds'], total_macs=record['work']['total_macs'],
                                               memory_bytes=record['memory_bytes']['total'], parameters=record['parameters'],
                                               optimizer_steps=record['optimizer_steps'], best_epoch=record['best_epoch'])
            for arm in FEATURE_VARIANTS:
                directory = task_dir(output, 'feature', seed, arm, n)
                frec = verify_result(directory, identity)
                records[seed, arm, n] = frec
                require(frec['fingerprints'] == record['fingerprints'], 'OFF-feature data differ from OFF-pixel data')
                with np.load(directory / 'panel.npz') as z:
                    require(np.array_equal(z['truth'], oracle[seed]['panel_truth']), 'feature panel truth mismatch')
                    require(np.array_equal(z['factors'], oracle[seed]['panel_factors']), 'feature panel factors mismatch')
                    m = feature_metrics(z['probabilities'], [z[f'flip_{c}'] for c in CUES], z['truth'], z['factors'],
                                        z['marginal'])
                require(m == frec['metrics'], 'recorded OFF-feature metrics differ from the raw arrays')
                entries[seed, arm, n] = finish(seed, dict(m))
                costs[seed, arm, n] = dict(elapsed_seconds=frec['elapsed_seconds'], total_macs=frec['work']['total_macs'],
                                           memory_bytes=frec['memory_bytes']['total'], parameters=frec['parameters'],
                                           optimizer_steps=frec['optimizer_steps'],
                                           chosen_weight_decay=frec['chosen_weight_decay'])
        require(len(inits) == 1, f'initialisation not shared between ON-ref, ON-R256 and OFF-pixel (seed {seed})')

    def ex(arm, n=at):
        return np.array([entries[s, arm, n]['excess'] for s in seeds])

    def per_seed(values):
        return dict(zip(map(str, seeds), np.asarray(values, dtype=float).tolist()))

    ref, r256 = ex('ON-ref'), ex('ON-R256')
    ref_ema, r256_ema = ex('ON-ref-EMA'), ex('ON-R256-EMA')
    pix, feat, feat_all = ex('OFF-pixel'), ex('OFF-feature'), ex('OFF-feature-all')
    ref_mean = float(ref.mean())
    consistency, bar = a['consistency_seeds'], a['competent_excess']

    # ---- R1 calibration
    ceiling = {str(n): dict(**summary_stats(ex('OFF-feature', n), draws), per_seed=per_seed(ex('OFF-feature', n)))
               for n in config['offline']['N']}
    c_at = float(feat.mean())
    infeasible = c_at > bar
    arms_at = ['ON-ref', 'ON-R256', 'ON-ref-EMA', 'ON-R256-EMA', 'OFF-pixel', 'OFF-feature', 'OFF-feature-all']
    r1 = dict(rule=f'OFF-feature seed-mean excess at N={at} > {bar} => the "{bar} over O" bar is infeasible at this N',
              ceiling_C_at_readings=c_at, ceiling_bootstrap95=G.bootstrap(feat, draws), ceiling_by_N=ceiling,
              bar_infeasible=bool(infeasible),
              verdict=(f'INFEASIBLE at N={at}: C(N)={c_at:.4f} > {bar}; future bars are excess over C(N) <= {bar}'
                       if infeasible else f'the bar stands: C(N)={c_at:.4f} <= {bar}'),
              excess_over_ceiling_descriptive={arm: float((ex(arm) - feat).mean()) for arm in arms_at},
              within_bar_over_ceiling_descriptive={arm: bool((ex(arm) - feat).mean() <= bar) for arm in arms_at})

    # ---- R2 decomposition
    shares = dict(estimation=feat, perception_architecture=pix - feat, online_optimisation=ref - pix)
    definitions = dict(estimation='E = OFF-feature excess', perception_architecture='P = OFF-pixel - OFF-feature',
                       online_optimisation='Q = ON-ref - OFF-pixel')
    r2_shares = {}
    for name, values in shares.items():
        mean = float(values.mean())
        positive = int((values > 0).sum())
        dominant = mean >= a['dominant_fraction'] * ref_mean and positive >= consistency
        r2_shares[name] = dict(definition=definitions[name], mean=mean, per_seed=per_seed(values),
                               bootstrap95=G.bootstrap(values, draws), fraction_of_on_ref_excess=mean / ref_mean,
                               fraction_bootstrap95=G.ratio_bootstrap(values, ref, draws),
                               seeds_positive=positive, seeds_negative=int((values < 0).sum()),
                               negative_mean=bool(mean < 0), dominant=bool(dominant))
    dominant = [k for k, v in r2_shares.items() if v['dominant']]
    largest = max(r2_shares, key=lambda k: r2_shares[k]['mean'])
    implications = dict(
        online_optimisation='Q dominant: the single-pass online update rule limits competence; next is online '
                            'consolidation (weight averaging, slow/fast weights), then tested against plasticity under law switches.',
        perception_architecture='P dominant: the pixel encoder or this network is the limit; fix perception and '
                                'architecture offline first; no continual-learning claim until fixed.',
        estimation='E dominant: the data are the limit; O is the wrong yardstick at this N; score future studies as '
                   'regret against C(N) and move towards denser consequences (Step D).')
    r2 = dict(on_ref_excess=dict(mean=ref_mean, per_seed=per_seed(ref), bootstrap95=G.bootstrap(ref, draws)),
              identity_check=float(np.abs(sum(shares.values()) - ref).max()),
              rule=f'dominant if >= {a["dominant_fraction"]:.0%} of ON-ref seed-mean excess and positive in >= '
                   f'{consistency}/{n_seeds} seeds; otherwise mixed', shares=r2_shares, dominant=dominant,
              largest_share=largest,
              verdict=('dominant: ' + ', '.join(dominant)) if dominant else
              f'MIXED (largest share: {largest}, {r2_shares[largest]["fraction_of_on_ref_excess"]:.0%} of ON-ref excess; not dominant)',
              implication=(' | '.join(implications[k] for k in dominant) if dominant else
                           f'mixed: choose the next step by the largest share ({largest}), stating that it is not dominant'))

    # ---- R3 replay re-test and R4 weight averaging
    def removal(base, other, base_name, other_name):
        reduction = base - other
        base_mean = float(base.mean())
        passed = float(reduction.mean()) >= a['matter_fraction'] * base_mean and int((reduction > 0).sum()) >= consistency
        return dict(base=base_name, other=other_name, base_mean=base_mean, other_mean=float(other.mean()),
                    reduction=dict(mean=float(reduction.mean()), per_seed=per_seed(reduction),
                                   bootstrap95=G.bootstrap(reduction, draws)),
                    fraction_removed=float(reduction.mean()) / base_mean,
                    fraction_bootstrap95=G.ratio_bootstrap(reduction, base, draws),
                    seeds_reduced=int((reduction > 0).sum()), passes=bool(passed))
    r3 = removal(ref, r256, 'ON-ref', 'ON-R256')
    r3['rule'] = f'confirmed if ON-R256 removes >= {a["matter_fraction"]:.0%} of ON-ref seed-mean excess and reduces it in >= {consistency}/{n_seeds} seeds'
    r3['verdict'] = 'R256 CONFIRMED' if r3['passes'] else 'R256 not confirmed'
    r4 = removal(r256, r256_ema, 'ON-R256', 'ON-R256-EMA')
    r4['rule'] = f'EMA helps if ON-R256-EMA removes >= {a["matter_fraction"]:.0%} of ON-R256 seed-mean excess and reduces it in >= {consistency}/{n_seeds} seeds'
    r4['verdict'] = 'EMA HELPS' if r4['passes'] else 'EMA does not help by the pre-declared rule'
    r4['on_ref_ema_vs_on_ref_descriptive'] = removal(ref, ref_ema, 'ON-ref', 'ON-ref-EMA')

    # ---- descriptive
    arm_rows = {}
    points = [(arm, at) for arm in ('ON-ref', 'ON-R256', 'ON-ref-EMA', 'ON-R256-EMA')]
    points += [(arm, n) for arm in OFFLINE_ARMS for n in config['offline']['N']]
    for arm, n in points:
        values = ex(arm, n)
        row = dict(arm=arm, N=n, privileged=arm in PRIVILEGED, privileged_label=PRIVILEGED.get(arm),
                   excess=dict(mean=float(values.mean()), per_seed=per_seed(values), bootstrap95=G.bootstrap(values, draws)),
                   brier=float(np.mean([entries[s, arm, n]['brier'] for s in seeds])),
                   O_brier=float(np.mean([O[s]['brier'] for s in seeds])),
                   survival=float(np.mean([entries[s, arm, n]['survival'] for s in seeds])),
                   survival_regret=dict(**summary_stats([entries[s, arm, n]['survival_regret'] for s in seeds], draws),
                                        per_seed=per_seed([entries[s, arm, n]['survival_regret'] for s in seeds])),
                   cue_benefit={str(c): float(np.mean([entries[s, arm, n]['cue_benefit'][str(c)] for s in seeds])) for c in CUES},
                   O_cue_benefit={str(c): float(np.mean([O[s]['cue_benefit'][str(c)] for s in seeds])) for c in CUES},
                   cue_benefit_minus_O={str(c): summary_stats([entries[s, arm, n]['cue_benefit_minus_O'][str(c)] for s in seeds], draws)
                                        for c in CUES})
        arm_rows[f'{arm}@{n}'] = row
    learning_curves = {arm: {str(n): arm_rows[f'{arm}@{n}']['excess'] for n in config['offline']['N']} for arm in OFFLINE_ARMS}
    sparse = {}
    for n in config['offline']['N']:
        values = ex('OFF-feature', n) - ex('OFF-feature-all', n)
        sparse[str(n)] = dict(mean=float(values.mean()), per_seed=per_seed(values), bootstrap95=G.bootstrap(values, draws),
                              seeds_positive=int((values > 0).sum()))
    online_summary = {}
    for name in ('ON-ref', 'ON-ref-EMA', 'ON-R256', 'ON-R256-EMA'):
        curves = np.array([online_curves[s, name]['excess'] for s in seeds])
        online_summary[name] = dict(chunk_end_arrivals=[(i + 1) * chunk_rows for i in range(curves.shape[1])],
                                    mean_excess=curves.mean(axis=0).tolist(),
                                    mean_learner=np.mean([online_curves[s, name]['learner'] for s in seeds], axis=0).tolist(),
                                    mean_O=np.mean([online_curves[s, name]['O'] for s in seeds], axis=0).tolist(),
                                    per_seed={str(s): online_curves[s, name]['excess'] for s in seeds})
    cost_summary = {}
    for key in sorted({k[1:] for k in costs}, key=str):
        rows = [costs[(s, *key)] for s in seeds]
        cost_summary['@'.join(map(str, key))] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]
                                                 if not isinstance(rows[0][k], str)}
    oracle_summary = {str(s): O[s] for s in seeds}
    oracle_summary['mean_brier'] = float(np.mean([O[s]['brier'] for s in seeds]))
    oracle_summary['mean_survival'] = float(np.mean([O[s]['survival'] for s in seeds]))
    oracle_summary['max_mc_se_brier'] = float(max(O[s]['mc_se_brier'] for s in seeds))
    results = dict(label=LABEL, protocol=config['protocol'], identity=identity, config=config,
                   manifest_locked_utc=manifest.get('locked_utc'), analysed_utc=now(), readings_at=at,
                   privileged_ceilings=PRIVILEGED, verification=verification, oracle=oracle_summary,
                   readings=dict(R1_calibration=r1, R2_decomposition=r2, R3_replay_retest=r3, R4_weight_averaging=r4),
                   arms=arm_rows, learning_curves=learning_curves, sparse_label_cost_in_ceiling=sparse,
                   online=online_summary, costs=cost_summary,
                   bootstrap=dict(resamples=a['bootstrap_resamples'], seed=a['analysis_seed'],
                                  note='95% paired seed-resample percentile intervals; descriptive only'))
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / 'results.json', results)
    with (report / 'per_seed.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'arm', 'N', 'privileged', 'learner_brier', 'O_brier', 'excess', 'survival', 'O_survival',
                         'survival_regret', 'marginal_gain', 'cue0_benefit', 'cue1_benefit', 'cue2_benefit',
                         'O_cue0_benefit', 'O_cue1_benefit', 'O_cue2_benefit', 'elapsed_seconds', 'total_macs',
                         'memory_bytes', 'optimizer_steps'])
        for seed in seeds:
            for arm, n in points:
                m = entries[seed, arm, n]
                c = costs[(seed, arm) if arm.startswith('ON') else (seed, arm, n)]
                writer.writerow([seed, arm, n, arm in PRIVILEGED, m['brier'], m['O_brier'], m['excess'], m['survival'],
                                 m['O_survival'], m['survival_regret'], m['marginal_gain'],
                                 *[m['cue_benefit'][str(k)] for k in CUES], *[m['O_cue_benefit'][str(k)] for k in CUES],
                                 c['elapsed_seconds'], c['total_macs'], c['memory_bytes'], c['optimizer_steps']])
    with (report / 'online_excess.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'arm', 'chunk', 'chunk_end_arrivals', 'learner_brier', 'O_brier', 'excess'])
        for seed in seeds:
            for name in online_summary:
                row = online_curves[seed, name]
                for i, (lb, ob, e) in enumerate(zip(row['learner'], row['O'], row['excess'])):
                    writer.writerow([seed, name, i, (i + 1) * chunk_rows, lb, ob, e])
    plot_learning_curves(results, config, report / 'learning_curves.png')
    return results


def plot_learning_curves(results, config, path):
    """Excess over O against N (log2 axis) for the offline arms; online arms as markers at readings_at."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None
    blue, orange, aqua, yellow, magenta = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
    ink, muted = '#0b0b0b', '#52514e'
    ns = config['offline']['N']
    at = config['analysis']['readings_at']
    bar = config['analysis']['competent_excess']
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    styles = {'OFF-pixel': (blue, '-', 'o', 'OFF-pixel (same network and init as ON-ref)'),
              'OFF-feature': (orange, '-', 's', 'OFF-feature (PRIVILEGED perception)'),
              'OFF-feature-all': (aqua, '--', 'D', 'OFF-feature-all (PRIVILEGED perception + labels)')}
    for arm, (color, ls, marker, label) in styles.items():
        rows = [results['arms'][f'{arm}@{n}']['excess'] for n in ns]
        mean = [r['mean'] for r in rows]
        low = [r['mean'] - r['bootstrap95']['lower'] for r in rows]
        high = [r['bootstrap95']['upper'] - r['mean'] for r in rows]
        ax.errorbar(ns, mean, yerr=[low, high], color=color, ls=ls, lw=2, marker=marker, ms=8, capsize=3,
                    label=label)
    online = {'ON-ref': (yellow, 'o', True), 'ON-ref-EMA': (yellow, 'o', False),
              'ON-R256': (magenta, 's', True), 'ON-R256-EMA': (magenta, 's', False)}
    offsets = {'ON-ref': -.09, 'ON-ref-EMA': -.03, 'ON-R256': .03, 'ON-R256-EMA': .09}
    for name, (color, marker, filled) in online.items():
        row = results['arms'][f'{name}@{at}']['excess']
        x = at * 2 ** offsets[name]
        ax.hlines(row['mean'], at * 2 ** (offsets[name] - .025), at * 2 ** (offsets[name] + .025), color=color, lw=2)
        ax.plot([x], [row['mean']], marker=marker, ms=9, color=color, mfc=color if filled else 'white', mew=2, ls='none',
                label=f'{name} (online, {at} arrivals)' + ('' if filled else ', EMA weights'))
    ax.axhline(bar, color=muted, lw=1.2, ls='--')
    ax.text(ns[0], bar, f' competence bar: excess {bar:.3f} over O', va='bottom', ha='left', fontsize=8, color=muted)
    ax.axhline(0, color=muted, lw=.6)
    ax.set_xscale('log', base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel('training records N (log2 scale)', color=muted)
    ax.set_ylabel('panel Brier minus oracle O (mean of seeds, 95% bootstrap)', color=muted)
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.))
    count = len(config['seeds'])
    ax.set_title(f'Achievable competence ceiling: excess over the law-known oracle '
                 f'({count} seed{"s" if count > 1 else ""})', fontsize=10.5, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------- replicate pass

REPLICATE_LABEL = ('INTEGRITY ADDITION, NOT IN THE PROTOCOL (requested before the main lock): selected fits are '
                   're-run from scratch and compared bit-for-bit with the main run; every main-run fit is re-verified '
                   'against its stored hashes. Any mismatch stops the study before analysis.')
REPLICATE_SEEDS = (19301, 19306, 19311)
VOLATILE = {'elapsed_seconds', 'training_seconds', 'started_utc', 'completed_utc', 'utc', 'resumes', 'resumed',
            'artifact_hashes', 'checkpoint_signature', 'evaluation_files', 'ema_evaluation_files'}


def normalize(value):
    """Drop wall-clock, resume and zip-timestamp-dependent fields; everything else must match exactly."""
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in VOLATILE and not str(k).endswith('_seconds')}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def replicate_tasks(config):
    seeds = list(REPLICATE_SEEDS) if not config['smoke'] else list(config['seeds'])
    n = config['analysis']['readings_at']
    return [('online', s, 'ON-ref', None) for s in seeds] + [('pixel', s, 'OFF-pixel', n) for s in seeds] + \
        [('feature', s, 'OFF-feature', n) for s in seeds]


def compare_fit(kind, main_dir, copy_dir, identity):
    rows = {}
    a, b = read_json(main_dir / 'result.json'), read_json(copy_dir / 'result.json')
    rows['result_json_normalized'] = normalize(a) == normalize(b)
    ca, cb = load_checkpoint(main_dir / 'checkpoint.pt', identity), load_checkpoint(copy_dir / 'checkpoint.pt', identity)
    if kind == 'online':
        for name in ('training.npz', 'evaluations.npz', 'ema/evaluations.npz'):
            rows[name] = npz_equal(main_dir / name, copy_dir / name)
        for name in ('evaluations.json', 'ema/evaluations.json'):
            rows[name] = sha(main_dir / name) == sha(copy_dir / name)
        rows['end_signature'] = a['end_signature'] == b['end_signature']
        rows['final_model_sha256'] = a['final_model_sha256'] == b['final_model_sha256']
        rows['ema_model_sha256'] = a['ema_model_sha256'] == b['ema_model_sha256']
        rows['checkpoint_learner_signature'] = (learner_signature(ca['learner'], True) == learner_signature(cb['learner'], True)
                                                == a['end_signature'])
        rows['checkpoint_optimizer'] = _value_hash(ca['learner'].optimizer.state_dict()) == _value_hash(cb['learner'].optimizer.state_dict())
        rows['checkpoint_ema_state'] = (state_hash(ca['ema']['state']) == state_hash(cb['ema']['state']) == a['ema_model_sha256']
                                        and ca['ema']['steps'] == cb['ema']['steps'])
        rows['checkpoint_online_arrays'] = _value_hash(ca['online']) == _value_hash(cb['online'])
        rows['checkpoint_evaluation_snapshots'] = (_value_hash(ca['evaluation']) == _value_hash(cb['evaluation'])
                                                   and _value_hash(ca['ema_evaluation']) == _value_hash(cb['ema_evaluation']))
    elif kind == 'pixel':
        rows['evaluations.npz'] = npz_equal(main_dir / 'evaluations.npz', copy_dir / 'evaluations.npz')
        rows['evaluations.json'] = sha(main_dir / 'evaluations.json') == sha(copy_dir / 'evaluations.json')
        rows['best_model_sha256'] = a['best_model_sha256'] == b['best_model_sha256']
        rows['final_model_sha256'] = a['final_model_sha256'] == b['final_model_sha256']
        rows['checkpoint_model'] = state_hash(ca['model']) == state_hash(cb['model']) == a['final_model_sha256']
        rows['checkpoint_best_model'] = state_hash(ca['best_model']) == state_hash(cb['best_model']) == a['best_model_sha256']
        rows['checkpoint_optimizer'] = _value_hash(ca['optimizer']) == _value_hash(cb['optimizer'])
        rows['checkpoint_state'] = normalize(ca['state']) == normalize(cb['state'])
    else:
        rows['panel.npz'] = npz_equal(main_dir / 'panel.npz', copy_dir / 'panel.npz')
        rows['member_state_hashes'] = ({k: v['info']['state_sha256'] for k, v in a['members'].items()}
                                       == {k: v['info']['state_sha256'] for k, v in b['members'].items()})
        rows['checkpoint_member_states'] = (
            {k: state_hash(v['state']) for k, v in ca['members'].items()}
            == {k: state_hash(v['state']) for k, v in cb['members'].items()}
            == {k: v['info']['state_sha256'] for k, v in a['members'].items()})
        rows['metrics'] = a['metrics'] == b['metrics']
    return rows


def replicate(config, config_path, output):
    """Integrity addition: re-verify every main-run fit, then re-fit selected tasks and compare bit-for-bit."""
    validate_config(config)
    output = Path(output)
    manifest = check_lock(config, config_path, output)
    identity = manifest['identity']
    log_event(output, event='replicate_start', label=REPLICATE_LABEL)
    completion = read_json(output / 'completion.json')
    stored = {}
    for kind, seed, arm, n in all_tasks(config):
        directory = task_dir(output, kind, seed, arm, n)
        verify_result(directory, identity)
        stored[f'{kind}:{arm}:{seed}:{n}'] = True
    for fit in completion['fits']:
        require(sha(output / fit['path']) == fit['sha256'], f'result.json changed since completion: {fit["path"]}')
    for seed, row in completion['oracle'].items():
        require(sha(output / row['path']) == row['sha256'], f'oracle file changed: {row["path"]}')
    target = output / 'replicate'
    jobs = []
    for kind, seed, arm, n in replicate_tasks(config):
        payload = dict(config=config, identity=identity, seed=seed, output=str(target))
        if kind == 'online':
            payload['arm'] = arm
        elif kind == 'pixel':
            payload['n'] = n
        else:
            payload.update(arm=arm, n=n)
        jobs.append((kind, payload))
    jobs.sort(key=lambda job: -estimated_cost(config, *job))
    failures, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=config['workers'], initializer=G.worker_init,
                             initargs=(config['cpu_affinity'], config['threads'])) as pool:
        futures = {pool.submit(dispatch, kind, payload): (kind, payload) for kind, payload in jobs}
        for future in as_completed(futures):
            kind, payload = futures[future]
            label = dict(kind=kind, seed=payload['seed'], arm=payload.get('arm'), N=payload.get('n'))
            try:
                result = future.result()
                log_event(output, event='replicate_task_done', **label, status=result.get('status'))
            except Exception as error:
                failures.append(dict(**label, error=repr(error)))
                log_event(output, event='replicate_task_failed', **label, error=repr(error),
                          traceback=traceback.format_exc())
    if failures:
        raise RuntimeError(f'{len(failures)} replicate tasks failed; rerun `replicate` to resume: {failures}')
    comparisons, mismatches = {}, []
    for kind, seed, arm, n in replicate_tasks(config):
        rows = compare_fit(kind, task_dir(output, kind, seed, arm, n), task_dir(target, kind, seed, arm, n), identity)
        key = f'{arm}:{seed}' + (f':N{n}' if n else '')
        comparisons[key] = rows
        mismatches += [f'{key}:{k}' for k, v in rows.items() if not v]
    report = dict(label=REPLICATE_LABEL, identity=identity, utc=now(), replicate_seeds=sorted({t[1] for t in replicate_tasks(config)}),
                  replicate_N=config['analysis']['readings_at'], main_fits_reverified=len(stored),
                  main_result_hashes_match_completion=True, oracle_files_match_completion=True,
                  comparisons=comparisons, mismatches=mismatches, passed=not mismatches,
                  seconds=time.perf_counter() - started,
                  normalized_away=sorted(VOLATILE) + ['*_seconds'],
                  note='npz archives are compared array-by-array (zip entries carry write timestamps)')
    write_json(output / 'replicate.json', report)
    log_event(output, event='replicate_complete', passed=report['passed'], mismatches=mismatches)
    if mismatches:
        raise RuntimeError(f'REPLICATE MISMATCH, do not analyze: {mismatches}')
    return report


# ---------------------------------------------------------------- checks

def check_config(config):
    """A tiny variant of the config for engineering checks (never used for science)."""
    c = copy.deepcopy(config)
    c['online'].update(arrivals=640, chunk=320, checkpoint_every=320)
    c['offline']['N'] = [256, 512]
    c['offline']['pixel']['max_epochs'] = 3
    c['offline']['feature'].update(max_epochs=12, patience=4)
    return c


def step_b_config(config, arrivals, checkpoint_every):
    """A Step-B-shaped config so Step B's own run_fit can serve as the fit WITHOUT EMA."""
    keys = ('stage', 'batch_size', 'eval_size', 'support_replicates', 'context_width', 'experts',
            'interaction_features', 'lr', 'evidence_strength', 'auxiliary_weight', 'threads')
    return dict({k: config[k] for k in keys}, arrivals=arrivals, exposure_arrivals=2 * arrivals, chunk=checkpoint_every,
                checkpoint_every=checkpoint_every, reference=dict(U=12, R=16, W=64),
                factors=dict(U=[12, 3], R=[16, 256], W=[64, 256]))


def npz_equal(a, b):
    with np.load(a) as x, np.load(b) as y:
        return set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)


def trace_equal(a, b):
    ja, jb = read_json(Path(a) / 'evaluations.json'), read_json(Path(b) / 'evaluations.json')
    return ja == jb and npz_equal(Path(a) / 'evaluations.npz', Path(b) / 'evaluations.npz')


def fresh(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    return path


def run_checks(config, config_path, output):
    validate_config(config)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = Path(output)
    out = base / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    seed = config['seeds'][0]
    small = check_config(config)
    law, world, size = G.study_law(config, seed), AcquisitionWorld(), config['batch_size']
    max_n = max(config['offline']['N'])
    ref_stream = [G.stream_batch(world, law, seed, i, size) for i in range(max_n // size)]
    ref_fps = [d.fingerprint() for d, _, _ in ref_stream]
    report = dict(label=LABEL, seed=seed, utc=now(), checks={}, timings={})
    identity = dict(check='competence_ceiling')

    def save():
        write_json(out / 'checks.json', report)

    def timed(name, fn):
        started = time.perf_counter()
        value = fn()
        report['timings'][name] = time.perf_counter() - started
        return value

    # 1. settings and shared initialisation (ON-ref, ON-R256, OFF-pixel)
    hashes, rows = set(), {}
    for arm in ONLINE_ARMS:
        spec = online_spec(config, arm)
        learner = reference_learner(config, seed, arm)
        m = learner.model
        require(m.frame[0].out_features == spec['W'] and m.temporal.hidden_size == spec['W']
                and m.decoder[0].out_features == spec['W'] and m.context.hidden_size == 12
                and m.context.input_size == spec['W'] + 8 + 5 * spec['W'], 'architecture not applied')
        require(learner.memory.capacity == spec['R'] and learner.updates_per_batch == spec['U'], 'R/U not applied')
        require(not any(True for _ in m.buffers()), 'unexpected buffers')
        sb = G.learner_settings(step_b_config(config, 640, 320), G.arm_spec(step_b_config(config, 640, 320),
                                                                             f'U{spec["U"]}_R{spec["R"]}_W{spec["W"]}'))
        require(learner.settings == sb, 'online settings differ from Step B')
        hashes.add(state_hash(m.state_dict()))
        rows[arm] = dict(U=spec['U'], R=spec['R'], W=spec['W'], initial_model_sha256=state_hash(m.state_dict()),
                         parameters=sum(q.numel() for q in m.parameters()))
    pixel_initial = state_hash(reference_learner(config, seed).model.state_dict())
    hashes.add(pixel_initial)
    require(len(hashes) == 1, 'initialisation not shared between ON-ref, ON-R256 and OFF-pixel')
    first = reference_learner(config, seed)
    first.train(ref_stream[0][0])
    require(first.memory.packets[0].support is None, 'online first packet support is not None')
    require(L.offline_packets([d for d, _, _ in ref_stream[:2]])[0].support is None
            and L.offline_packets([d for d, _, _ in ref_stream[:2]])[1].support is ref_stream[0][0],
            'offline packets do not mirror online packets')
    report['checks']['settings_and_shared_initialisation'] = dict(
        passed=True, arms=rows, off_pixel_initial_model_sha256=pixel_initial,
        first_online_packet_support='None (dummy zero support, context zeroed)', offline_first_packet_support='None')
    save()

    # 2. EMA non-interference against Step B's own run_fit (the fit without EMA)
    sbc = step_b_config(config, small['online']['arrivals'], small['online']['checkpoint_every'])
    ema_rows = {}
    for arm in ONLINE_ARMS:
        spec = online_spec(config, arm)
        name = f'U{spec["U"]}_R{spec["R"]}_W{spec["W"]}'
        dir_b, dir_e = fresh(out / 'stepb_no_ema'), fresh(out / 'with_ema')
        timed(f'step_b_fit_{arm}', lambda: G.run_fit(sbc, identity, seed, name, str(dir_b)))
        timed(f'ema_fit_{arm}', lambda: run_online(small, identity, seed, arm, str(dir_e)))
        rb = read_json(G.fit_dir(dir_b, seed, name) / 'result.json')
        re_ = read_json(task_dir(dir_e, 'online', seed, arm) / 'result.json')
        cb = load_checkpoint(G.fit_dir(dir_b, seed, name) / 'checkpoint.pt', identity)['learner']
        ce = load_checkpoint(task_dir(dir_e, 'online', seed, arm) / 'checkpoint.pt', identity)['learner']
        with np.load(G.fit_dir(dir_b, seed, name) / 'training.npz') as x, \
                np.load(task_dir(dir_e, 'online', seed, arm) / 'training.npz') as y:
            same_online = all(np.array_equal(x[k], y[k]) for k in x.files)
            ema_differs = not np.array_equal(y['probabilities'], y['ema_probabilities'])
        same = dict(
            end_signature=rb['end_signature'] == re_['end_signature'],
            final_model=rb['final_model_sha256'] == re_['final_model_sha256'],
            checkpoint_learner_signature=learner_signature(cb, True) == learner_signature(ce, True),
            checkpoint_optimizer=_value_hash(cb.optimizer.state_dict()) == _value_hash(ce.optimizer.state_dict()),
            checkpoint_model=state_hash(cb.model.state_dict()) == state_hash(ce.model.state_dict()),
            checkpoint_memory_ids=cb.memory.ids == ce.memory.ids,
            checkpoint_optimizer_has_no_hooks=not ce.optimizer._optimizer_step_post_hooks,
            fingerprints=rb['fingerprints'] == re_['fingerprints'],
            online_forecasts=same_online,
            endpoint_traces=trace_equal(G.fit_dir(dir_b, seed, name), task_dir(dir_e, 'online', seed, arm)),
            score_signature=rb['scores'][-1]['signature'] == re_['scores'][-1]['signature'])
        require(all(same.values()), f'EMA tracking changed the online fit ({arm}): {same}')
        require(re_['ema_steps'] == re_['optimizer_steps'] and ema_differs
                and re_['ema_model_sha256'] != re_['final_model_sha256'], 'EMA not tracked')
        ema_rows[arm] = dict(step_b_arm=name, arrivals=small['online']['arrivals'], identical=same,
                             end_signature=re_['end_signature'], ema_steps=re_['ema_steps'],
                             ema_model_sha256=re_['ema_model_sha256'])
    # closed-form EMA check with per-step snapshots (a second, read-only post-hook)
    learner, shadow = reference_learner(config, seed), reference_learner(config, seed)
    ema = L.WeightEMA(learner, shadow, config['online']['ema_decay'])
    snapshots = [{k: v.detach().clone() for k, v in learner.model.named_parameters()}]
    learner.optimizer.register_step_post_hook(
        lambda *_: snapshots.append({k: v.detach().clone() for k, v in learner.model.named_parameters()}))
    for d, _, _ in ref_stream[:3]:
        learner.train(d)
    reference = L.ema_reference(snapshots, config['online']['ema_decay'])
    error = max(float((dict(shadow.model.named_parameters())[k].detach().double() - v).abs().max())
                for k, v in reference.items())
    require(len(snapshots) == 37 and ema.steps == 36 and error < 1e-6, 'EMA does not match its closed form')
    report['checks']['ema_non_interference'] = dict(
        passed=True, against='competence_gap.run_fit (Step B, no EMA) on the same stream', arms=ema_rows,
        closed_form=dict(optimizer_steps=36, max_abs_error_vs_float64=error, initialised_at_initial_weights=True))
    save()

    # 3. determinism: two identical short fits, for an online arm, OFF-pixel and OFF-feature
    det = {}
    a_dir, b_dir = fresh(out / 'det_a'), fresh(out / 'det_b')
    for d in (a_dir, b_dir):
        timed(f'det_online_{d.name}', lambda: run_online(small, identity, seed, 'ON-ref', str(d)))
        timed(f'det_pixel_{d.name}', lambda: run_pixel(small, identity, seed, 512, str(d)))
        timed(f'det_feature_{d.name}', lambda: run_feature(small, identity, seed, 'OFF-feature', 512, str(d)))
    ra, rb = (read_json(task_dir(d, 'online', seed, 'ON-ref') / 'result.json') for d in (a_dir, b_dir))
    det['online'] = (ra['end_signature'] == rb['end_signature'] and ra['ema_model_sha256'] == rb['ema_model_sha256']
                     and npz_equal(*(task_dir(d, 'online', seed, 'ON-ref') / 'training.npz' for d in (a_dir, b_dir)))
                     and trace_equal(*(task_dir(d, 'online', seed, 'ON-ref') for d in (a_dir, b_dir)))
                     and trace_equal(*(task_dir(d, 'online', seed, 'ON-ref') / 'ema' for d in (a_dir, b_dir))))
    pa, pb = (read_json(task_dir(d, 'pixel', seed, 'OFF-pixel', 512) / 'result.json') for d in (a_dir, b_dir))
    det['pixel'] = (pa['best_model_sha256'] == pb['best_model_sha256'] and pa['epochs'] == pb['epochs']
                    and trace_equal(*(task_dir(d, 'pixel', seed, 'OFF-pixel', 512) for d in (a_dir, b_dir))))
    fa, fb = (read_json(task_dir(d, 'feature', seed, 'OFF-feature', 512) / 'result.json') for d in (a_dir, b_dir))
    det['feature'] = (fa['members'] == fb['members'] and fa['metrics'] == fb['metrics']
                      and npz_equal(*(task_dir(d, 'feature', seed, 'OFF-feature', 512) / 'panel.npz' for d in (a_dir, b_dir))))
    require(all(det.values()), f'fits are not deterministic: {det}')
    report['checks']['determinism'] = dict(passed=True, identical=det, online_arrivals=small['online']['arrivals'],
                                           offline_N=512, pixel_epochs=small['offline']['pixel']['max_epochs'],
                                           pixel_best_epoch=pa['best_epoch'], feature_chosen_weight_decay=fa['chosen_weight_decay'])
    save()

    # 4. offline data = exact online stream prefix (fingerprints)
    k = small['online']['arrivals'] // size
    prefix = dict(online_fit_equals_stream=ra['fingerprints'] == ref_fps[:k],
                  step_b_fit_equals_stream=read_json(G.fit_dir(out / 'stepb_no_ema', seed, 'U12_R256_W64') / 'result.json')['fingerprints'] == ref_fps[:k],
                  pixel_equals_stream=pa['fingerprints'] == ref_fps[:512 // size],
                  feature_equals_stream=fa['fingerprints'] == ref_fps[:512 // size],
                  offline_stream_function=[d.fingerprint() for d, _, _ in offline_stream(config, seed, max_n)[1]] == ref_fps,
                  feature_inputs_are_stream_cases=fa['feature_matrix_sha256'] == hashlib.sha256(L.latent_features(
                      np.concatenate([c.base.reserves for _, c, _ in ref_stream[:512 // size]]),
                      np.concatenate([c.factors for _, c, _ in ref_stream[:512 // size]]),
                      np.concatenate([c.signals for _, c, _ in ref_stream[:512 // size]])).tobytes()).hexdigest(),
                  feature_labels_are_stream_labels=fa['label_sha256'] == hashlib.sha256(
                      np.concatenate([t for _, _, t in ref_stream[:512 // size]]).astype(np.float32).tobytes()).hexdigest())
    require(all(prefix.values()), f'offline data are not the stream prefix: {prefix}')
    report['checks']['offline_data_is_stream_prefix'] = dict(passed=True, identical=prefix, reference_batches=len(ref_fps),
                                                             first_fingerprint=ref_fps[0], last_fingerprint=ref_fps[-1])
    save()

    # 5. oracle reproduces evaluator truth with realised supplies; MC primitives agree
    data = G.endpoint_case_data(config, seed)
    stream_cells = sum(G.verify_realised(c, law, t, d.actions, d.survival) for d, c, t in ref_stream)
    mc = dict(key=(G.MC_CHECK, seed, 0, 0), channel=G.MC_CHECK, seed=seed, replicate=0, batch=0, kb=4, gauge=False,
              laws=[(law.mode, list(law.active))], query=data['query'], support=data['support'])
    require(np.array_equal(MC.panel_task(mc)['query_counts'], MC.naive_panel_counts(mc)), 'dedup MC differs from naive')
    task_path = out / 'online_oracle_check.npz'
    if task_path.exists():
        task_path.unlink()
    oconf = dict(config, oracle=dict(config['oracle'], online_samples=256, online_sub=256))
    timed('online_oracle_16_batches_256_samples', lambda: G.online_oracle_task(dict(config=oconf, seed=seed, first_batch=0,
                                                                                  batches=16, path=str(task_path))))
    with np.load(task_path) as z:
        truth16 = np.concatenate([t for _, _, t in ref_stream[:16]])
        require(np.array_equal(z['truth'], truth16) and list(map(str, z['fingerprints'])) == ref_fps[:16],
                'online oracle misaligned with the stream')
        online_brier = float(((G.monotone(z['counts'] / 256) - truth16) ** 2).mean())
    endpoint_path = out / 'endpoint_oracle_check.npz'
    if endpoint_path.exists():
        endpoint_path.unlink()
    econf = dict(config, oracle=dict(config['oracle'], endpoint_kb=256))
    timed('endpoint_oracle_batch_256', lambda: G.endpoint_oracle_task(dict(config=econf, seed=seed, gauge=False,
                                                                         channel=G.MC_CHECK, batch=0, path=str(endpoint_path))))
    with np.load(endpoint_path) as z:
        p = z['query_counts'][0] / 256
        endpoint_brier = float(((G.monotone(p) - data['truth']) ** 2).mean())
    report['checks']['oracle'] = dict(passed=True, realised_supply_cells_verified=dict(
        endpoint_panel_and_supports=data['verified_cells'], stream_first_batches=stream_cells),
        dedup_equals_naive=True, online_task_aligned=True, online_brier_first_512_K256=online_brier,
        endpoint_brier_K256=endpoint_brier, exact_channel=G.MC_ENDPOINT_EXACT)
    save()

    # 6. disjointness of training records from the panel and supports
    main_max = 32768
    training_seeds = {trial_seed(seed, G.TRAINING_CHANNEL, i) for i in range(max(main_max, max_n) // size)}
    evaluator_seeds = {data['query_seed'], *data['support_seeds']}
    dis = disjointness(config, seed, law, [d for d, _, _ in ref_stream])
    require(not training_seeds & evaluator_seeds, 'training and evaluator seeds overlap')
    report['checks']['disjointness'] = dict(passed=True, training_seed_count=len(training_seeds),
                                            evaluator_seeds=sorted(evaluator_seeds), shared_seeds=0, images=dis)
    save()

    # 7. losses, schedule and features
    learner = reference_learner(config, seed)
    packets = L.offline_packets([d for d, _, _ in ref_stream[:3]])
    item = packets[2]
    loss = float(L.packet_loss(learner, item).detach())
    with torch.no_grad():
        probs = learner.forward_with_features(item.query.observations, item.support)[0].double().clamp(1e-6, 1 - 1e-6).numpy()
    sel = probs[np.arange(len(item.query)), item.query.actions]
    y = item.query.survival.astype(np.float64)
    manual = float(-(y * np.log(sel) + (1 - y) * np.log(1 - sel)).mean())
    require(abs(loss - manual) < 1e-5, 'OFF-pixel loss is not the performed-action BCE')
    total = 40 * 100
    lrs = [L.cosine_lr(.002, s, total) for s in (0, total // 2, total - 1)]
    require(lrs[0] == .002 and abs(lrs[1] - .001) < 1e-12 and 0 < lrs[2] < 1e-8, 'cosine schedule wrong')
    q = data['query']
    x = L.latent_features(q['reserves'], q['factors'], q['signals'])
    require(np.allclose(x[:, :2] * 240, q['levels'], atol=.5 + 1e-5), 'feature reserves disagree with the gauges')
    require(np.array_equal(x[:, 2], (q['factors'][:, 0] == 1).astype(np.float32))
            and np.array_equal(x[:, 5:], q['signals'].astype(np.float32)), 'feature coding wrong')
    report['checks']['losses_schedule_features'] = dict(passed=True, pixel_packet_loss=loss, independent_float64=manual,
                                                        cosine_lr_samples=lrs, feature_width=int(x.shape[1]))
    save()

    # 8. resumption after interruption (online, OFF-pixel, OFF-feature)
    r_dir = fresh(out / 'resume')
    try:
        run_online(small, identity, seed, 'ON-ref', str(r_dir), interrupt_after=13)
        raise AssertionError('interruption did not happen')
    except Interrupted:
        pass
    run_online(small, identity, seed, 'ON-ref', str(r_dir))
    rr = read_json(task_dir(r_dir, 'online', seed, 'ON-ref') / 'result.json')
    online_ok = (rr['end_signature'] == ra['end_signature'] and rr['ema_model_sha256'] == ra['ema_model_sha256']
                 and rr['resumed'] and len(rr['resumes']) == 1
                 and npz_equal(task_dir(r_dir, 'online', seed, 'ON-ref') / 'training.npz',
                               task_dir(a_dir, 'online', seed, 'ON-ref') / 'training.npz')
                 and trace_equal(task_dir(r_dir, 'online', seed, 'ON-ref'), task_dir(a_dir, 'online', seed, 'ON-ref'))
                 and trace_equal(task_dir(r_dir, 'online', seed, 'ON-ref') / 'ema', task_dir(a_dir, 'online', seed, 'ON-ref') / 'ema'))
    try:
        run_pixel(small, identity, seed, 512, str(r_dir), interrupt_after_epoch=2)
        raise AssertionError('interruption did not happen')
    except Interrupted:
        pass
    run_pixel(small, identity, seed, 512, str(r_dir))
    pr = read_json(task_dir(r_dir, 'pixel', seed, 'OFF-pixel', 512) / 'result.json')
    pixel_ok = (pr['best_model_sha256'] == pa['best_model_sha256'] and pr['final_model_sha256'] == pa['final_model_sha256']
                and pr['epochs'] == pa['epochs'] and len(pr['resumes']) == 1
                and trace_equal(task_dir(r_dir, 'pixel', seed, 'OFF-pixel', 512), task_dir(a_dir, 'pixel', seed, 'OFF-pixel', 512)))
    try:
        run_feature(small, identity, seed, 'OFF-feature', 512, str(r_dir), interrupt_after_member=4)
        raise AssertionError('interruption did not happen')
    except Interrupted:
        pass
    run_feature(small, identity, seed, 'OFF-feature', 512, str(r_dir))
    fr = read_json(task_dir(r_dir, 'feature', seed, 'OFF-feature', 512) / 'result.json')
    feature_ok = (fr['members'] == fa['members'] and fr['metrics'] == fa['metrics'] and len(fr['resumes']) == 1
                  and npz_equal(task_dir(r_dir, 'feature', seed, 'OFF-feature', 512) / 'panel.npz',
                                task_dir(a_dir, 'feature', seed, 'OFF-feature', 512) / 'panel.npz'))
    existing = run_online(small, identity, seed, 'ON-ref', str(r_dir))['status'] == 'verified_existing'
    require(online_ok and pixel_ok and feature_ok and existing, 'resumed fits differ')
    report['checks']['resumption'] = dict(passed=True, online=dict(interrupted_after_batch=13,
                                                                  resumed_from_batch=rr['resumes'][0]['batches_done']),
                                          pixel=dict(interrupted_after_epoch=2, resumed_from_epoch=pr['resumes'][0]['epochs_done']),
                                          feature=dict(interrupted_after_member_fits=4,
                                                       resumed_with=len(fr['resumes'][0]['completed'])),
                                          completed_task_is_verified_not_rerun=existing)
    save()

    # 9. smoke study end to end (lock, run, rerun = verify only, analyze)
    if config['smoke']:
        pipeline = dict(archived_previous_smoke_outputs=archive_smoke_outputs(base))
        manifest = timed('smoke_lock', lambda: lock(config, config_path, base))
        attempts = []
        for attempt in range(3):          # a failed invocation is kept in attempts.jsonl and resumed
            try:
                first = timed(f'smoke_run_attempt_{attempt}', lambda: run_study(config, config_path, base))
                attempts.append(dict(attempt=attempt, status='completed'))
                break
            except RuntimeError as error:
                attempts.append(dict(attempt=attempt, status='failed', error=repr(error)))
        else:
            raise RuntimeError(f'smoke run failed three times: {attempts}')
        pipeline['run_attempts'] = attempts
        pipeline['failed_task_events'] = [json.loads(line) for line in
                                          (base / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()
                                          if '"task_failed"' in line and json.loads(line)['utc'] >= manifest['locked_utc']]
        for event in pipeline['failed_task_events']:
            event.pop('traceback', None)
        second = timed('smoke_rerun', lambda: run_study(config, config_path, base))
        statuses = {s['status'] for s in second['statuses'] if s['kind'] in ('online', 'pixel', 'feature')}
        oracle_statuses = {s['status'] for s in second['statuses'] if s['kind'] not in ('online', 'pixel', 'feature')}
        require(statuses == {'verified_existing'} and oracle_statuses == {'existing'}, 'rerun did not only verify')
        strip = lambda summary: {k: {kk: vv for kk, vv in v.items() if kk != 'sha256'} for k, v in summary.items()}  # noqa: E731
        require(strip(first['oracle']) == strip(second['oracle']), 'oracle assembly not reproducible')
        rep = timed('smoke_replicate', lambda: replicate(config, config_path, base))
        require(rep['passed'], 'smoke replicate pass mismatched')
        pipeline['replicate'] = dict(passed=rep['passed'], comparisons=len(rep['comparisons']),
                                     main_fits_reverified=rep['main_fits_reverified'])
        results = timed('smoke_analyze', lambda: analyze(config, config_path, base, base / 'analysis'))
        pipeline.update(manifest_identity=manifest['identity'], fits=len(first['fits']),
                        rerun_statuses=sorted(statuses | oracle_statuses), report=(base / 'analysis').resolve().relative_to(ROOT).as_posix(),
                        readings={k: v.get('verdict') for k, v in results['readings'].items()},
                        note='smoke outputs cannot inform scientific choices')
        pipeline['task_seconds'] = {f"{f['kind']}:{f['arm']}:{f['N']}": f['elapsed_seconds'] for f in first['fits']}
        pipeline['oracle_task_seconds_per_seed'] = {k: v['task_seconds'] for k, v in first['oracle'].items()}
        pipeline['estimate'] = estimate_main(config, first, base)
        report['checks']['smoke_pipeline'] = dict(passed=True, **pipeline)
    report['all_passed'] = all(v['passed'] for v in report['checks'].values())
    save()
    return report


def archive_smoke_outputs(base):
    """Move an earlier smoke study (never a scientific run) aside so the smoke lock matches the live files.
    Nothing is deleted: failed attempts stay in attempts.jsonl and in the archived folder."""
    names = ('manifest.json', 'protocol_at_lock.md', 'config_at_lock.json', 'source_at_lock.zip', 'completion.json',
             'online', 'pixel', 'feature', 'oracle', 'analysis', 'replicate', 'replicate.json')
    present = [n for n in names if (base / n).exists()]
    if not present:
        return None
    if (base / 'manifest.json').exists():
        require(read_json(base / 'manifest.json')['config']['smoke'], 'refusing to move a scientific run')
    target = base / 'superseded' / now().replace(':', '').replace('+', '_')
    target.mkdir(parents=True, exist_ok=True)
    for n in present:
        shutil.move(str(base / n), str(target / n))
    return target.relative_to(base).as_posix()


def estimate_main(config, completion, base):
    """Scale smoke timings to the main configuration (one worker-second each; 6 workers)."""
    main = read_json(ROOT / 'configs' / 'competence_ceiling.json')
    seconds = {}
    online = [f for f in completion['fits'] if f['kind'] == 'online']
    per_arrival = np.mean([f['elapsed_seconds'] for f in online]) / config['online']['arrivals']
    seconds['online'] = per_arrival * main['online']['arrivals'] * len(ONLINE_ARMS) * len(main['seeds'])
    step_times, feature_rate = [], []
    for f in completion['fits']:
        record = read_json(base / f['path'])
        if f['kind'] == 'pixel':
            step_times.append(record['elapsed_seconds'] / max(1, record['optimizer_steps']))
        if f['kind'] == 'feature':
            epochs = sum(m['info']['epochs_run'] for m in record['members'].values())
            feature_rate.append(record['elapsed_seconds'] / max(1, epochs * record['design']['train_records']))
    step = float(np.mean(step_times))
    pixel = sum(main['offline']['pixel']['max_epochs'] * (n // 32 - int(round(.2 * n / 32))) for n in main['offline']['N'])
    seconds['pixel'] = step * pixel * len(main['seeds'])
    rate = float(np.mean(feature_rate))
    f = main['offline']['feature']
    per_member_records = sum(int(n * .8) for n in main['offline']['N'])
    fits = f['members'] * len(f['weight_decays']) * len(FEATURE_VARIANTS) * len(main['seeds'])
    seconds['feature_upper_400_epochs'] = rate * per_member_records * f['max_epochs'] * fits
    seconds['feature_typical_100_epochs'] = rate * per_member_records * 100 * fits
    per_seed_oracle = np.mean([v['task_seconds'] for v in completion['oracle'].values()])
    o, mo = config['oracle'], main['oracle']
    scale = (mo['endpoint_batches'] * mo['endpoint_kb'] + main['online']['arrivals'] * mo['online_samples'] / 32 / 16) / \
            (o['endpoint_batches'] * o['endpoint_kb'] + config['online']['arrivals'] * o['online_samples'] / 32 / 16)
    seconds['oracle_rough'] = per_seed_oracle * scale * len(main['seeds'])
    typical = seconds['online'] + seconds['pixel'] + seconds['feature_typical_100_epochs'] + seconds['oracle_rough']
    upper = seconds['online'] + seconds['pixel'] + seconds['feature_upper_400_epochs'] + seconds['oracle_rough']
    return dict(cpu_seconds=seconds, pixel_seconds_per_optimizer_step=step, feature_seconds_per_record_epoch=rate,
                online_seconds_per_arrival=per_arrival, workers=main['workers'],
                wall_hours_typical=typical / main['workers'] / 3600, wall_hours_upper=upper / main['workers'] / 3600,
                note=('smoke-scaled estimate; smoke fits run concurrently on 6 workers, OFF-feature epochs depend on early '
                      'stopping, the oracle scaling is rough (endpoint and online MC mixed)'))


# -------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('check', 'lock', 'run', 'replicate', 'analyze'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.workers is not None:
        raise SystemExit('workers are fixed by the locked config')
    output = Path(args.output)
    try:
        if args.command == 'check':
            report = run_checks(config, args.config, output)
            print(json.dumps({k: v['passed'] for k, v in report['checks'].items()}, indent=1))
        elif args.command == 'lock':
            print(json.dumps(lock(config, args.config, output)['identity'], indent=1))
        elif args.command == 'run':
            completion = run_study(config, args.config, output)
            print(json.dumps(dict(fits=len(completion['fits']), seconds=completion['wall_seconds_this_invocation'])))
        elif args.command == 'replicate':
            rep = replicate(config, args.config, output)
            print(json.dumps(dict(passed=rep['passed'], comparisons=len(rep['comparisons']),
                                  main_fits_reverified=rep['main_fits_reverified'], mismatches=rep['mismatches'])))
        else:
            if not args.report:
                raise SystemExit('--report is required for analyze')
            results = analyze(config, args.config, output, args.report)
            print(json.dumps({k: v['verdict'] for k, v in results['readings'].items()}, indent=1))
    except Exception as error:
        log_event(output, event=f'{args.command}_failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
