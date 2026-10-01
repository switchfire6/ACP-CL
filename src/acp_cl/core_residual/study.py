"""Locked, resumable continuous core/residual learning experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch

from acp_cl.acquisition.study import marginal, marginal_update
from acp_cl.acquisition.world import AcquisitionWorld, batch, order, stage_law
from acp_cl.persistence.study import digest
from acp_cl.predictive_value.mechanism import predict_all
from acp_cl.predictive_value.study import load, read_json, save
from acp_cl.rehearsal_state import study as previous
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import affinity, new_learner, write_json
from acp_cl.selective_updates.study import verify_analysis
from .design import ARMS, counts, law_from_record, phases, prefix_phases, validate_config
from .evaluation import TraceEvaluator
from .learner import fork_core, learner_signature, new_from_scratch


ROOT = Path(__file__).resolve().parents[3]
save_arrays, read_arrays = previous.save_arrays, previous.read_arrays


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    files = previous.source_manifest()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        files[path.relative_to(ROOT/'src').as_posix()] = sha(path)
    return files


def expected_replay_ids(learner, updates):
    generator = copy.deepcopy(learner.memory.sampling_rng)
    ids = list(learner.memory.ids)
    return [ids[int(generator.integers(0, len(ids)))] if ids else learner.memory.seen
            for _ in range(updates)]


def training_metrics(probabilities, actions, outcomes):
    selected = np.asarray(probabilities)[np.arange(len(actions)), actions].astype(np.float64)
    return float(np.square(selected-np.asarray(outcomes, dtype=np.float64)).mean())


def boundary(evaluator, learner, law, cue, branch, marginal_values=None):
    signature = learner_signature(learner)
    probe = evaluator.probe(learner, law, cue, branch, marginal=marginal_values)
    valid = evaluator.valid_panel(learner, law, branch)
    if learner_signature(learner) != signature:
        raise AssertionError('evaluation changed complete learner state')
    return probe, valid


def episode(config, identity, learner, seed, model, arm, phase, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    before_path, checkpoint = directory/'before.pt', directory/'checkpoint.pt'
    result_path = directory/'result.json'
    parent = learner_signature(learner)
    if result_path.exists():
        result = read_json(result_path)
        saved = load(checkpoint, identity, str(learner.device))
        if (result['identity'] != identity or result['start_signature'] != parent
                or digest(saved['record']) != digest(result)
                or learner_signature(saved['learner']) != result['end_signature']):
            raise ValueError('completed phase state or record changed')
        if any(sha(directory/name) != value for name,value in result['artifact_hashes'].items()):
            raise ValueError('completed phase artifact changed')
        return saved['learner'], result
    law, cue, branch = law_from_record(phase['law']), phase['cue'], phase['branch']
    if checkpoint.exists():
        saved = load(checkpoint, identity, str(learner.device))
        learner, record = saved['learner'], saved['record']
        if record['start_signature'] != parent:
            raise ValueError('partial phase parent changed')
        if 'end_signature' in record:
            if (learner_signature(learner) != record['end_signature']
                    or any(sha(directory/name) != value for name,value in record['artifact_hashes'].items())):
                raise ValueError('completed checkpoint artifact changed')
            write_json(result_path, record)
            return learner, record
        evaluator = TraceEvaluator(seed, config, saved['evaluation'])
        predictions, actions, outcomes = (saved['training'][key] for key in
                                          ('probabilities', 'actions', 'outcomes'))
    else:
        evaluator = TraceEvaluator(seed, config)
        before, valid_before = boundary(evaluator, learner, law, cue, branch)
        record = dict(identity=identity, seed=seed, model=model, arm=arm, phase=phase['name'],
            law=phase['law'], cue=cue, branch=branch, size=phase['size'], channel=phase['channel'],
            batches_done=0, batch_sha256=[], packets=[], before=before, valid_before=valid_before,
            start_diagnostics=learner.diagnostics(), start_signature=parent,
            start_memory=memory_state(learner.memory),
            curve=[dict(arrivals=0, metrics=evaluator.evaluate(learner, law, learner.history, cue, branch))],
            marginal_count=[0]*5, marginal_success=[[0]*3 for _ in range(5)], elapsed_seconds=0.)
        predictions, actions, outcomes = [], [], []
        save(before_path, dict(identity=identity, learner=learner, record=copy.deepcopy(record)))
    if not before_path.exists():
        raise ValueError('missing initial phase checkpoint')
    world = AcquisitionWorld()
    started = time.perf_counter()
    for index in range(record['batches_done'], phase['size']//config['batch_size']):
        data = batch(world, law, seed, phase['channel'], index, config['batch_size'])
        support = None if learner.history is None else learner.history.fingerprint()
        probabilities = predict_all(learner, data.observations, learner.history)
        packet = dict(index=index, id=learner.memory.seen, query_sha256=data.fingerprint(),
            support_sha256=support, memory_ids_before=list(learner.memory.ids),
            replay_ids=expected_replay_ids(learner, config['updates_per_batch']),
            prequential_brier=training_metrics(probabilities, data.actions, data.survival))
        learner.train(data)
        record['packets'].append(packet)
        record['batch_sha256'].append(data.fingerprint())
        predictions.append(probabilities)
        actions.append(data.actions.copy())
        outcomes.append(data.survival.copy())
        marginal_update(record, data)
        record['batches_done'] = index+1
        arrivals = (index+1)*config['batch_size']
        if arrivals % config['probe_every'] == 0:
            record['curve'].append(dict(arrivals=arrivals,
                metrics=evaluator.evaluate(learner, law, learner.history, cue, branch,
                                           marginal=marginal(record))))
            record['elapsed_seconds'] += time.perf_counter()-started
            save(checkpoint, dict(identity=identity, learner=learner, record=record,
                evaluation=evaluator.snapshot(), training=dict(probabilities=predictions,
                                                               actions=actions, outcomes=outcomes)))
            started = time.perf_counter()
    record['after'], record['valid_after'] = boundary(evaluator, learner, law, cue, branch,
                                                     marginal(record))
    record['novel_after_return'] = None
    if phase['name'] == 'return':
        signature = learner_signature(learner)
        record['novel_after_return'] = evaluator.probe(learner, stage_law(seed, 1), order(seed)[0])
        if learner_signature(learner) != signature:
            raise AssertionError('post-return novel probe changed learner state')
    record['diagnostics'] = learner.diagnostics()
    record['end_signature'] = learner_signature(learner)
    record['end_memory'] = memory_state(learner.memory)
    record['prequential_brier'] = float(np.mean([row['prequential_brier'] for row in record['packets']]))
    record['evaluation_files'] = evaluator.export(directory)
    save_arrays(directory/'training.npz', probabilities=np.asarray(predictions),
                actions=np.asarray(actions), outcomes=np.asarray(outcomes))
    record['training_file'] = dict(path='training.npz', sha256=sha(directory/'training.npz'))
    record['artifact_hashes'] = {name:sha(directory/name) for name in
                                ('before.pt', 'evaluations.json', 'evaluations.npz', 'training.npz')}
    record['elapsed_seconds'] += time.perf_counter()-started
    # A completed checkpoint contains exactly the final record and explicit gradients.
    save(checkpoint, dict(identity=identity, learner=learner, record=record))
    write_json(result_path, record)
    print(json.dumps(dict(seed=seed, model=model, arm=arm, phase=phase['name'], completed=True)), flush=True)
    return learner, record


def run_job(config, identity, seed, model, output, device='cpu'):
    actual = affinity(config['cpu_affinity'])
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/f'{model}_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/'result.json'
    if result_path.exists():
        record = read_json(result_path)
        if record['identity'] != identity:
            raise ValueError('completed job identity changed')
        for field, name in (('phase_hashes', 'result.json'), ('checkpoint_hashes', 'checkpoint.pt')):
            if any(sha(directory/phase/name) != value for phase,value in record[field].items()):
                raise ValueError('completed job artifact changed')
        for phase in record['phase_hashes']:
            phase_record = read_json(directory/phase/'result.json')
            if any(sha(directory/phase/name) != value
                   for name,value in phase_record['artifact_hashes'].items()):
                raise ValueError('completed job raw artifact changed')
        return dict(seed=seed, model=model, affinity=actual)
    learner = new_learner(model, seed, config, device)
    names = []
    for phase in prefix_phases(seed, config):
        learner, _ = episode(config, identity, learner, seed, model, 'prefix', phase, directory/phase['name'])
        names.append(phase['name'])
    parent_signature = learner_signature(learner)
    final_signatures, insertion = {}, {}
    for arm in ARMS:
        fork = fork_core(learner, arm, seed, config)
        insertion[arm] = dict(signature=learner_signature(fork), diagnostics=fork.diagnostics())
        for phase in phases(seed, config, arm):
            relative = f"{arm}/{phase['name']}"
            fork, _ = episode(config, identity, fork, seed, model, arm, phase, directory/relative)
            names.append(relative)
        final_signatures[arm] = learner_signature(fork)
    if learner_signature(learner) != parent_signature:
        raise AssertionError('continuous arms changed their shared parent')
    fresh = new_from_scratch(model, seed, config, device)
    maintenance = phases(seed, config, 'joint')[0]
    fresh.history = batch(AcquisitionWorld(), law_from_record(maintenance['law']), seed,
        maintenance['channel'], maintenance['size']//config['batch_size']-1, config['batch_size'])
    for phase in phases(seed, config, 'fresh'):
        relative = f"fresh/{phase['name']}"
        fresh, _ = episode(config, identity, fresh, seed, model, 'fresh', phase, directory/relative)
        names.append(relative)
    final_signatures['fresh'] = learner_signature(fresh)
    result = dict(identity=identity, seed=seed, model=model, affinity=actual,
        parent_signature=parent_signature, insertion=insertion, final_signatures=final_signatures,
        phase_hashes={name:sha(directory/name/'result.json') for name in names},
        checkpoint_hashes={name:sha(directory/name/'checkpoint.pt') for name in names})
    write_json(result_path, result)
    print(json.dumps(dict(seed=seed, model=model, job_completed=True)), flush=True)
    return dict(seed=seed, model=model, affinity=actual)


def analysis_files(config_path, protocol):
    paths = [p for p in (ROOT/'scripts').glob('*.py') if p.name != 'verify_core_residual_archive.py']
    paths += [p for p in (ROOT/'tests').glob('test_core_residual*.py')
              if p.name != 'test_core_residual_archive.py']
    if config_path:
        paths.append(Path(config_path).resolve())
    if protocol:
        paths.append(Path(protocol).resolve())
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}


def check_locks(output, config=None, protocol=None, device='cpu', verify_live=True):
    manifest = previous.check_locks(output, config, protocol, device, verify_live=False)
    if verify_live:
        if manifest['source_files'] != source_manifest() or manifest['runtime']['device'] != device:
            raise ValueError('scientific source or device changed after locking')
        verify_analysis(output)
    return manifest


def run_suite(config, output, protocol, resume=False, lock_only=False, device='cpu', config_path=None):
    validate_config(config)
    actual = affinity(config['cpu_affinity'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    runtime = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, device=device, threads=config['threads'], workers=config['workers'],
        deterministic=True, affinity=actual)
    identity = dict(config_sha256=digest(config), source_sha256=digest(sources), runtime_sha256=digest(runtime))
    if (output/'manifest.json').exists():
        if not resume:
            raise ValueError('existing study requires explicit resume')
        manifest = check_locks(output, config, protocol, device)
        if manifest['identity'] != identity or manifest['runtime'] != runtime:
            raise ValueError('resume requires identical source, config and runtime')
    else:
        if any(output.iterdir()):
            raise ValueError('new study requires an empty output directory')
        now = datetime.now(timezone.utc).isoformat()
        manifest = dict(identity=identity, config=config, runtime=runtime, source_files=sources, created_utc=now)
        with zipfile.ZipFile(output/'training_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(ROOT/'src'/name, name)
        files = analysis_files(config_path, protocol)
        with zipfile.ZipFile(output/'analysis_at_lock.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                archive.write(ROOT/name, name)
        write_json(output/'analysis_lock.json', dict(locked_utc=now, files=files, config=config,
                                                    config_sha256=identity['config_sha256']))
        contents = Path(protocol).read_bytes()
        (output/'protocol_at_lock.md').write_bytes(contents)
        write_json(output/'protocol_lock.json', dict(**identity, locked_utc=now,
                    protocol_sha256=hashlib.sha256(contents).hexdigest()))
        write_json(output/'manifest.json', manifest)
    if lock_only:
        result = dict(locked=True, identity=identity, **counts(config))
        print(json.dumps(result, indent=2))
        return result
    if (output/'completion.json').exists():
        completed = read_json(output/'completion.json')
        if completed['identity'] != identity:
            raise ValueError('completion identity changed')
        for model in config['models']:
            for seed in config['seeds']:
                run_job(config, identity, seed, model, output, device)
        return completed
    jobs = [(seed, model) for model in config['models'] for seed in config['seeds']]
    if config['workers'] == 1:
        finished = [run_job(config, identity, seed, model, output, device) for seed,model in jobs]
    else:
        finished = []
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            futures = [pool.submit(run_job, config, identity, seed, model, output, device) for seed,model in jobs]
            for future in as_completed(futures):
                finished.append(future.result())
    check_locks(output, config, protocol, device)
    completed = dict(identity=identity, **counts(config),
        job_records=sorted(finished, key=lambda row:(row['model'],row['seed'])),
        completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'completion.json', completed)
    print(json.dumps(dict(completed=True, **counts(config)), indent=2))
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lock-only', action='store_true')
    args = parser.parse_args()
    run_suite(read_json(args.config), args.output, args.protocol, args.resume, args.lock_only,
              args.device, args.config)
