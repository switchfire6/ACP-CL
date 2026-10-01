"""Locked, resumable representation learning with complete forecast artifacts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import torch

from acp_cl.acquisition.world import AcquisitionWorld, mask_affected, mask_valid
from acp_cl.core_residual.evaluation import TraceEvaluator
from acp_cl.core_residual.study import expected_replay_ids
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest, trial_seed, write_json
from acp_cl.predictive_value.mechanism import predict_all
from acp_cl.predictive_value.study import load, save
from acp_cl.replay_renewal.study import affinity
from .design import (counts, law_from_record, law_key, main_phases, phase_laws,
                     qualification_phases, validate_config)
from .learner import RepresentationLearner, learner_signature


ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_manifest():
    return {p.relative_to(ROOT).as_posix(): sha(p)
            for p in sorted((ROOT/'src/acp_cl').rglob('*.py'))}


def save_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def training_seed(seed, phase, index):
    return trial_seed(seed, 'representation_training', [phase, index])


def update_marginal(marginals, law, data):
    entry = marginals.setdefault(law_key(law), dict(count=[0]*5, success=[[0]*3 for _ in range(5)]))
    for action in range(5):
        selected = data.actions == action
        entry['count'][action] += int(selected.sum())
        entry['success'][action] = (np.asarray(entry['success'][action])
                                    + data.survival[selected].sum(axis=0)).tolist()


def marginal_probabilities(entry):
    return (np.asarray(entry['success'])+.5)/(np.asarray(entry['count'])[:, None]+1.)


def endpoint_probes(evaluator, learner, targets, marginals):
    before = learner_signature(learner)
    result = []
    for target in targets:
        law = law_from_record(target)
        marginal = marginal_probabilities(marginals[law_key(law)])
        row = dict(law=target, correct=[], flips={str(c): [] for c in law.active})
        for rep in range(evaluator.settings['support_replicates']):
            support = evaluator.support(law, rep)
            row['correct'].append(evaluator.evaluate(learner, law, support,
                                                    marginal=marginal)['trace'])
            for cue in law.active:
                row['flips'][str(cue)].append(evaluator.evaluate(
                    learner, law, support, cue=cue, flipped=True)['trace'])
        result.append(row)
    if learner_signature(learner) != before:
        raise AssertionError('endpoint evaluation changed learner state')
    return result


def verify_phase(directory, identity):
    directory = Path(directory)
    record = read_json(directory/'result.json')
    if record['identity'] != identity:
        raise ValueError('phase identity mismatch')
    if any(sha(directory/name) != value for name, value in record['artifact_hashes'].items()):
        raise ValueError('phase artifact changed')
    return record


def require_phase(record, config, seed, arm, phase):
    if (record['seed'] != seed or record['arm'] != arm or record['kind'] != config['kind']
            or digest(record['phase_spec']) != digest(phase)):
        raise ValueError('requested phase specification changed')


def fit_phase(config, identity, learner, seed, arm, phase, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, result_path = directory/'checkpoint.pt', directory/'result.json'
    parent = learner_signature(learner)
    if result_path.exists():
        record = verify_phase(directory, identity)
        require_phase(record, config, seed, arm, phase)
        saved = load(checkpoint, identity)
        if (record['start_signature'] != parent or digest(saved['record']) != digest(record)
                or learner_signature(saved['learner']) != record['end_signature']):
            raise ValueError('phase checkpoint or parent changed')
        return saved['learner'], record
    laws = phase_laws(phase, config['batch_size'])
    distinct = list(dict.fromkeys(laws))
    if checkpoint.exists():
        saved = load(checkpoint, identity)
        learner, record, training = saved['learner'], saved['record'], saved.get('training')
        require_phase(record, config, seed, arm, phase)
        if record['start_signature'] != parent:
            raise ValueError('partial phase parent changed')
        if 'end_signature' in record:
            if learner_signature(learner) != record['end_signature']:
                raise ValueError('completed phase checkpoint changed')
            if any(sha(directory/n) != v for n, v in record['artifact_hashes'].items()):
                raise ValueError('completed phase arrays changed')
            write_json(result_path, record)
            return learner, record
        evaluator = TraceEvaluator(seed, config, saved['evaluation'])
    else:
        evaluator = TraceEvaluator(seed, config)
        signature = learner_signature(learner)
        start = evaluator.evaluate(learner, laws[0], learner.history, phase['cue'], phase['branch'])
        if learner_signature(learner) != signature:
            raise AssertionError('entry evaluation changed learner')
        record = dict(identity=identity, seed=seed, arm=arm, phase=phase['name'],
                      phase_spec=phase, kind=config['kind'], size=len(laws)*config['batch_size'],
                      batches_done=0, packets=[], marginals={}, start_signature=parent,
                      initial_model_sha256=state_hash(learner.model.state_dict()),
                      start_diagnostics=learner.diagnostics(), start_eval=start,
                      curve=[], elapsed_seconds=0.)
        training = {k: [] for k in ('observations', 'probabilities', 'truth', 'actions',
                                    'outcomes', 'affected', 'valid', 'law_index')}
        save(directory/'before.pt', dict(identity=identity, learner=learner, record=record))
    world = AcquisitionWorld()
    updates = config['compute_updates_per_batch'] if arm == 'compute' else config['updates_per_batch']
    started = time.perf_counter()
    for index in range(record['batches_done'], len(laws)):
        law = laws[index]
        data_seed = training_seed(seed, phase['name'], index)
        data = world.experience(law, config['batch_size'], data_seed)
        support = None if learner.history is None else learner.history.fingerprint()
        probabilities = predict_all(learner, data.observations, learner.history)
        # Evaluator-only full outcomes are produced after the causal forecast.
        cases = world.dataset(law, config['batch_size'], data_seed)
        truth = world.counterfactuals(cases)
        if not np.array_equal(truth[np.arange(len(data)), data.actions], data.survival):
            raise AssertionError('performed outcome differs from physical outcome')
        record['packets'].append(dict(index=index, id=learner.memory.seen,
            query_sha256=data.fingerprint(), support_sha256=support,
            memory_ids_before=list(learner.memory.ids), replay_ids=expected_replay_ids(learner, updates)))
        learner.train(data)
        values = dict(observations=data.observations, probabilities=probabilities,
                      truth=truth, actions=data.actions, outcomes=data.survival,
                      affected=mask_affected(cases, phase['cue']),
                      valid=mask_valid(cases, phase['branch']), law_index=distinct.index(law))
        for key, value in values.items():
            training[key].append(np.array(value, copy=True))
        update_marginal(record['marginals'], law, data)
        record['batches_done'] = index+1
        arrivals = (index+1)*config['batch_size']
        if arrivals % config['probe_every'] == 0 or index+1 == len(laws):
            record['curve'].append(dict(arrivals=arrivals, law=asdict(law),
                metrics=evaluator.evaluate(learner, law, learner.history,
                                           phase['cue'], phase['branch'])))
            record['elapsed_seconds'] += time.perf_counter()-started
            save(checkpoint, dict(identity=identity, learner=learner, record=record,
                                 evaluation=evaluator.snapshot(), training=training))
            started = time.perf_counter()
    record['end_probes'] = endpoint_probes(evaluator, learner, phase['targets'], record['marginals'])
    signature = learner_signature(learner)
    record['valid_after'] = (evaluator.valid_panel(learner, laws[-1], phase['branch'])
        if config['kind'] == 'main' and not phase['name'].startswith('prefix_') else None)
    if learner_signature(learner) != signature:
        raise AssertionError('retention evaluation changed learner state')
    record['final_model_sha256'] = state_hash(learner.model.state_dict())
    record['diagnostics'] = learner.diagnostics()
    record['end_signature'] = learner_signature(learner)
    record['evaluation_files'] = evaluator.export(directory)
    save_arrays(directory/'training.npz', {key: np.asarray(value) for key, value in training.items()})
    record['training_file'] = dict(path='training.npz', sha256=sha(directory/'training.npz'))
    record['artifact_hashes'] = {name: sha(directory/name) for name in
                                ('before.pt', 'evaluations.json', 'evaluations.npz', 'training.npz')}
    record['elapsed_seconds'] += time.perf_counter()-started
    save(checkpoint, dict(identity=identity, learner=learner, record=record))
    write_json(result_path, record)
    print(json.dumps(dict(seed=seed, arm=arm, phase=phase['name'], completed=True)), flush=True)
    return learner, record


def verify_job(directory, identity):
    directory = Path(directory)
    record = read_json(directory/'result.json')
    if record['identity'] != identity:
        raise ValueError('job identity changed')
    for name, value in record['phase_hashes'].items():
        if sha(directory/name) != value:
            raise ValueError('job phase hash changed')
        verify_phase((directory/name).parent, identity)
    if any(sha(directory/name) != value for name, value in record['checkpoint_hashes'].items()):
        raise ValueError('job checkpoint changed')
    return record


def run_job(config, identity, seed, arm, output):
    affinity(config['cpu_affinity'])
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/'jobs'/f'{arm}_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'result.json').exists():
        verify_job(directory, identity)
        return dict(seed=seed, arm=arm)
    phases = (qualification_phases if config['kind'] == 'qualification' else main_phases)(seed, config)
    learner, phase_paths = None, []
    for phase in phases:
        if learner is None or config['kind'] == 'qualification':
            learner = RepresentationLearner(arm, seed, config)
        path = directory/phase['name']
        learner, _ = fit_phase(config, identity, learner, seed, arm, phase, path)
        phase_paths.append(path)
    result = dict(identity=identity, seed=seed, arm=arm,
                  phase_hashes={f'{p.name}/result.json': sha(p/'result.json') for p in phase_paths},
                  checkpoint_hashes={f'{p.name}/checkpoint.pt': sha(p/'checkpoint.pt') for p in phase_paths})
    write_json(directory/'result.json', result)
    return dict(seed=seed, arm=arm)


def analysis_files(config_path, protocol):
    paths = [Path(config_path), Path(protocol), ROOT/'scripts/summarize_representation_learning.py']
    calibration = ROOT/'scripts/calibrate_representation_compute.py'
    if calibration.exists():
        paths.append(calibration)
    paths.extend(sorted((ROOT/'tests').glob('test_representation*.py')))
    return {p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in paths}


def check_locks(output, verify_live=True):
    output = Path(output)
    manifest = read_json(output/'manifest.json')
    identity = manifest['identity']
    if (digest(manifest['config']) != identity['config_sha256']
            or digest(manifest['source_files']) != identity['source_sha256']
            or digest(manifest['analysis_files']) != identity['analysis_sha256']
            or digest(manifest['runtime']) != identity['runtime_sha256']
            or sha(output/'protocol_at_lock.md') != identity['protocol_sha256']):
        raise ValueError('manifest lock mismatch')
    if verify_live and source_manifest() != manifest['source_files']:
        raise ValueError('live source file coverage changed')
    prerequisite = manifest.get('prerequisite')
    if (prerequisite is not None
            and sha(output/'engineering_at_lock.json') != prerequisite['engineering_sha256']):
        raise ValueError('archived engineering allocation changed')
    with zipfile.ZipFile(output/'source_at_lock.zip') as archive:
        for name, expected in {**manifest['source_files'], **manifest['analysis_files']}.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('archived source changed')
            if verify_live and sha(ROOT/name) != expected:
                raise ValueError(f'live locked source changed: {name}')
    return manifest


def run_suite(config, output, protocol, config_path, resume=False, lock_only=False,
              qualification=None, engineering=None):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source, analysis = source_manifest(), analysis_files(config_path, protocol)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   platform=platform.platform(), threads=config['threads'], workers=config['workers'])
    identity = dict(config_sha256=digest(config), source_sha256=digest(source),
                    analysis_sha256=digest(analysis), protocol_sha256=sha(protocol),
                    runtime_sha256=digest(runtime))
    prerequisite = None
    if config['kind'] == 'main' and not config['smoke']:
        if qualification is None or engineering is None:
            raise ValueError('main requires qualified reference and fixed engineering allocation')
        qualification = Path(qualification)
        check_locks(qualification)
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location('representation_summary_gate',
                                      ROOT/'scripts/summarize_representation_learning.py')
        module = module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        checked = module.summarize(qualification, output/'qualification_recheck')
        if not checked['qualified']:
            raise ValueError('reference qualification failed; main comparison is stopped')
        allocation = read_json(engineering)
        if allocation['compute_updates_per_batch'] != config['compute_updates_per_batch']:
            raise ValueError('compute allocation differs from engineering lock')
        settings = ('batch_size', 'width', 'context_width', 'decoder_width',
                    'interaction_features', 'memory_packets', 'lr',
                    'updates_per_batch', 'auxiliary_weight')
        payload = {key: value for key, value in allocation.items() if key != 'allocation_sha256'}
        if (allocation['status'] != 'complete' or digest(payload) != allocation['allocation_sha256']
                or allocation['protocol_sha256'] != sha(protocol)
                or any(allocation['config'][key] != config[key] for key in settings)
                or digest(allocation['source_files']) != allocation['source_files_sha256']
                or any(sha(ROOT/name) != value for name, value in allocation['source_files'].items())):
            raise ValueError('engineering provenance or model settings changed')
        prerequisite = dict(qualification=str(qualification.resolve()),
                            qualification_manifest_sha256=sha(qualification/'manifest.json'),
                            engineering=str(Path(engineering).resolve()),
                            engineering_sha256=sha(engineering))
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        old = check_locks(output)
        if not resume or old['identity'] != identity or old['prerequisite'] != prerequisite:
            raise ValueError('resume requires identical config/source/runtime/prerequisites')
    else:
        if any(p.name != 'qualification_recheck' for p in output.iterdir()):
            raise ValueError('new output directory must be empty')
        manifest = dict(identity=identity, config=copy.deepcopy(config), runtime=runtime,
                        source_files=source, analysis_files=analysis, prerequisite=prerequisite,
                        expected_counts=counts(config), created_utc=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, manifest)
        (output/'protocol_at_lock.md').write_bytes(Path(protocol).read_bytes())
        if prerequisite is not None:
            (output/'engineering_at_lock.json').write_bytes(Path(engineering).read_bytes())
        with zipfile.ZipFile(output/'source_at_lock.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(set(source) | set(analysis)):
                archive.write(ROOT/name, name)
    if lock_only:
        return identity
    jobs = [(seed, arm) for seed in config['seeds'] for arm in config['arms']]
    started = time.perf_counter()
    if config['workers'] == 1:
        for seed, arm in jobs:
            run_job(config, identity, seed, arm, output)
    else:
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            futures = [pool.submit(run_job, config, identity, seed, arm, str(output)) for seed, arm in jobs]
            for future in as_completed(futures):
                future.result()
    check_locks(output)
    records = []
    for seed, arm in jobs:
        relative = f'jobs/{arm}_{seed}/result.json'
        verify_job(output/Path(relative).parent, identity)
        records.append(dict(path=relative, sha256=sha(output/relative)))
    completion = output/'completion.json'
    if not completion.exists():
        write_json(completion, dict(identity=identity, jobs=records,
            elapsed_seconds=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat()))
    elif read_json(completion)['jobs'] != records:
        raise ValueError('completed job set changed')
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', default='docs/representation_learning_protocol.md')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--qualification')
    parser.add_argument('--engineering')
    args = parser.parse_args()
    run_suite(read_json(args.config), args.output, args.protocol, args.config, args.resume,
              args.lock_only, args.qualification, args.engineering)


if __name__ == '__main__':
    main()
