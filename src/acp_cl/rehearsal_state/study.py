"""Locked, resumable state-factor diagnosis with paired environmental branches."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import zipfile

import numpy as np
import torch

from acp_cl.acquisition.world import order
from acp_cl.persistence.study import digest
from acp_cl.predictive_value import study as previous
from acp_cl.predictive_value.design import select
from acp_cl.predictive_value.mechanism import clone_exact, packet_fingerprint, predict_all, state_signature
from acp_cl.replay_renewal.study import affinity, new_learner, write_json
from acp_cl.selective_updates.study import verify_analysis
from .design import BRANCHES, CELLS, counts, paired_batch, schedule, validate_config
from .mechanism import active_signature, make_factor_shadow


ROOT = Path(__file__).resolve().parents[3]
read_json, save, load = previous.read_json, previous.save, previous.load


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    sources = previous.source_manifest()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        sources[path.relative_to(ROOT/'src').as_posix()] = sha(path)
    return sources


def save_arrays(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def metric_panels(probabilities, arrays):
    """Full all-action arrays -> one metric vector per branch and candidate."""
    probabilities = np.asarray(probabilities)
    outcomes, truth, actions = (arrays[key] for key in ('outcomes', 'truth', 'actions'))
    expected = (2, len(actions), probabilities.shape[2], actions.shape[1], 5, 3)
    epsilon = np.finfo(np.float32).eps
    if (probabilities.shape != expected or not np.isfinite(probabilities).all()
            or np.any(probabilities < -epsilon) or np.any(probabilities > 1+epsilon)):
        raise ValueError('invalid complete diagnostic probabilities')
    result = {}
    for branch_index, branch in enumerate(BRANCHES):
        errors, survivals = [], []
        for packet_index, packet_actions in enumerate(actions):
            rows = probabilities[branch_index, packet_index]
            performed = rows[:, np.arange(len(packet_actions)), packet_actions].astype(np.float64)
            errors.append(np.square(performed-outcomes[branch_index, packet_index]).mean(axis=(1, 2)))
            greedy = rows[..., -1].argmax(axis=-1)
            physical = truth[branch_index, packet_index]
            survivals.append(physical[np.arange(len(packet_actions))[None, :], greedy, -1].mean(axis=1))
        errors, survivals = np.asarray(errors), np.asarray(survivals)
        result[branch] = dict(brier=errors.mean(axis=0).tolist(), survival=survivals.mean(axis=0).tolist(),
            early_brier=errors[0].tolist(), late_brier=errors[1:].mean(axis=0).tolist())
    return result


def build_streams(seed, index, config, initial_history):
    if initial_history is None:
        raise ValueError('assessment requires actual preceding history')
    histories = [initial_history, initial_history]
    streams, rows, observations, actions, outcomes, truths = [], [], [], [], [], []
    for packet in range(config['validation_size']//config['batch_size']):
        data, truth = paired_batch(seed, index, packet, config)
        streams.append([(data[b], histories[b]) for b in range(2)])
        rows.append({branch:dict(query_sha256=data[b].fingerprint(), support_sha256=histories[b].fingerprint())
                     for b, branch in enumerate(BRANCHES)})
        observations.append(data[0].observations)
        actions.append(data[0].actions)
        outcomes.append([d.survival for d in data])
        truths.append(truth)
        histories = data
    arrays = dict(observations=np.asarray(observations), actions=np.asarray(actions),
        outcomes=np.swapaxes(np.asarray(outcomes), 0, 1), truth=np.swapaxes(np.asarray(truths), 0, 1),
        initial_observations=initial_history.observations, initial_actions=initial_history.actions,
        initial_outcomes=initial_history.survival)
    return streams, rows, arrays


def verify_choice(directory, index, bundle, identity):
    score = read_json(directory/f'score_{index}'/'result.json')
    choice = read_json(directory/f'assessment_{index}'/'choice.json')
    expected = select(previous.observable_candidates(bundle['metadata'][:-1]), score['score_losses'])
    if (choice['identity'] != identity or choice['score_record_sha256'] != digest(score)
            or choice['choice'] != expected):
        raise ValueError('score-only choice seal mismatch')
    return expected


def prepare_cells(directory, index, config, identity, old, new, anchor_old, anchor_new, bundle):
    """Construct all states before generating either validation branch."""
    folder = directory/f'diagnostic_{index}'
    signatures = dict(old=state_signature(old), new=state_signature(new))
    paths = {'000':directory/f'score_{index}'/'shadows.pt'}
    for cell in CELLS[1:]:
        path = folder/f'cell_{cell}.pt'
        paths[cell] = path
        if path.exists():
            value = load(path, identity, str(old.device))
            if (value['parents'] != signatures or value['cell'] != cell or value['ids'] != bundle['ids']
                    or value['anchor_old_sha256'] != packet_fingerprint(anchor_old)
                    or value['anchor_new_sha256'] != packet_fingerprint(anchor_new)
                    or value['signatures'] != [state_signature(m) for m in value['learners']]
                    or value['active_signatures'] != [active_signature(m) for m in value['learners']]):
                raise ValueError('saved factorial cell differs from its donors or state')
            continue
        learners, work = [], []
        for packet in bundle['packets']:
            learner, cost = make_factor_shadow(old, new, anchor_old, anchor_new, packet, cell,
                                               config['rehearsal_updates'])
            learners.append(learner)
            work.append(cost)
        save(path, dict(identity=identity, parents=signatures, cell=cell, ids=bundle['ids'],
            anchor_old_sha256=packet_fingerprint(anchor_old), anchor_new_sha256=packet_fingerprint(anchor_new),
            learners=learners, work=work, signatures=[state_signature(m) for m in learners],
            active_signatures=[active_signature(m) for m in learners]))
    zero_path = folder/'zero.pt'
    if zero_path.exists():
        value = load(zero_path, identity, str(old.device))
        if value['parents'] != signatures or value['signatures'] != [state_signature(m) for m in value['learners']]:
            raise ValueError('zero-update references changed')
    else:
        learners = [clone_exact(old), clone_exact(new)]
        save(zero_path, dict(identity=identity, parents=signatures, learners=learners,
                            signatures=[state_signature(m) for m in learners]))
    if signatures != dict(old=state_signature(old), new=state_signature(new)):
        raise AssertionError('factor construction changed ordinary donors')
    paths['zero'] = zero_path
    return paths


def predict_family(learners, streams):
    signatures = [state_signature(m) for m in learners]
    results = []
    for branch in range(2):
        packets = []
        for stream in streams:
            data, history = stream[branch]
            packets.append([predict_all(m, data.observations, history) for m in learners])
        results.append(packets)
    if signatures != [state_signature(m) for m in learners]:
        raise AssertionError('branch predictions changed frozen state')
    return np.asarray(results)


def assessment(directory, config, identity, seed, model, index, new, metadata, anchor_new):
    folder = directory/f'diagnostic_{index}'
    folder.mkdir(exist_ok=True)
    old_payload = load(directory/f'score_{index}'/'before.pt', identity, str(new.device))
    old, anchor_old = old_payload['learner'], old_payload['anchor']
    bundle = load(directory/f'assessment_{index}'/'candidates.pt', identity, str(new.device))
    choice = verify_choice(directory, index, bundle, identity)
    parents = dict(old=state_signature(old), new=state_signature(new))
    before_path = folder/'before.pt'
    if not before_path.exists():
        save(before_path, previous.state_payload(identity, new, metadata, anchor_new,
                                                 dict(parents=parents)))
    before = load(before_path, identity, str(new.device))
    if (before['record']['parents'] != parents or state_signature(before['learner']) != parents['new']
            or packet_fingerprint(before['anchor']) != packet_fingerprint(anchor_new)):
        raise ValueError('assessment boundary changed')
    if (folder/'result.json').exists():
        result = read_json(folder/'result.json')
        if result['identity'] != identity or result['parents'] != parents or result['choice'] != choice:
            raise ValueError('completed assessment provenance changed')
        if any(sha(directory/name) != expected for name,expected in result['artifact_hashes'].items()):
            raise ValueError('completed assessment artifact changed')
        return result
    paths = prepare_cells(directory, index, config, identity, old, new, anchor_old, anchor_new, bundle)
    streams, rows, arrays = build_streams(seed, index, config, new.history)
    data_path = folder/'data.npz'
    if data_path.exists():
        saved = read_arrays(data_path)
        if set(saved) != set(arrays) or any(not np.array_equal(saved[k], arrays[k]) for k in arrays):
            raise ValueError('paired branch data changed')
    else:
        save_arrays(data_path, **arrays)
    write_json(folder/'stream.json', dict(identity=identity, seed=seed, model=model, index=index,
                                         branches=list(BRANCHES), rows=rows))
    panels = {branch:{} for branch in BRANCHES}
    work, active, state_hashes, prediction_hashes, zero = {}, {}, {}, {}, None
    for cell, path in paths.items():
        payload = load(path, identity, str(new.device))
        learners = payload['learners']
        if payload['signatures'] != [state_signature(m) for m in learners]:
            raise ValueError('diagnostic model state changed')
        prediction_path, seal_path = folder/f'predictions_{cell}.npz', folder/f'predictions_{cell}.json'
        active[cell] = [active_signature(m) for m in learners]
        if seal_path.exists():
            seal = read_json(seal_path)
            if (seal['identity'] != identity or seal['parents'] != parents
                    or seal['active_signatures'] != active[cell]
                    or seal['data_sha256'] != sha(data_path) or seal['predictions_sha256'] != sha(prediction_path)):
                raise ValueError('prediction checkpoint seal changed')
            probabilities = read_arrays(prediction_path)['probabilities']
        else:
            probabilities = predict_family(learners, streams)
            save_arrays(prediction_path, probabilities=probabilities)
            write_json(seal_path, dict(identity=identity, parents=parents, active_signatures=active[cell],
                data_sha256=sha(data_path), predictions_sha256=sha(prediction_path)))
        metrics = metric_panels(probabilities, arrays)
        if cell == 'zero':
            zero = metrics
        else:
            work[cell] = payload['work']
            for branch in BRANCHES:
                panels[branch][cell] = metrics[branch]
        state_hashes[cell] = dict(path=path.relative_to(directory).as_posix(), sha256=sha(path))
        prediction_hashes[cell] = dict(path=prediction_path.relative_to(directory).as_posix(), sha256=sha(prediction_path))
    if parents != dict(old=state_signature(old), new=state_signature(new)):
        raise AssertionError('assessment changed donors')
    relevant = [before_path, data_path, folder/'stream.json',
        directory/f'assessment_{index}'/'candidates.pt', directory/f'assessment_{index}'/'choice.json',
        directory/f'score_{index}'/'result.json', directory/f'score_{index}'/'before.pt']
    relevant += list(paths.values())+[p for p in folder.glob('predictions_*') if p.suffix in ('.npz', '.json')]
    result = dict(identity=identity, seed=seed, model=model, index=index, cue=order(seed)[0],
        parents=parents, old_steps=old.cost['optimizer_steps'], new_steps=new.cost['optimizer_steps'],
        old_anchor_sha256=packet_fingerprint(anchor_old), new_anchor_sha256=packet_fingerprint(anchor_new),
        candidates=bundle['metadata'][:-1], replacement=bundle['metadata'][-1],
        reservoir_ids=bundle['reservoir_ids'], anchor_id=bundle['anchor_id'],
        query_support_overlaps=bundle['query_support_overlaps'],
        candidate_storage_bytes=sum(p.nbytes() for p in bundle['packets']),
        choice=choice, panels=panels, zero=zero, work=work, active_signatures=active,
        state_files=state_hashes, prediction_files=prediction_hashes,
        artifact_hashes={p.relative_to(directory).as_posix():sha(p) for p in sorted(set(relevant))},
        prediction_work=dict(score_forwards=(config['candidate_count']+1)*config['score_size']//config['batch_size'],
            factorial_forwards=2*config['validation_size']//config['batch_size']*8*(config['candidate_count']+1),
            zero_forwards=4*config['validation_size']//config['batch_size']))
    write_json(folder/'result.json', result)
    print(json.dumps(dict(seed=seed, model=model, assessment=index, factorial_completed=True)), flush=True)
    return result


def run_job(config, identity, seed, model, output, device='cpu'):
    actual = affinity(config['cpu_affinity'])
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/f'{model}_{seed}'
    directory.mkdir(exist_ok=True)
    learner = new_learner(model, seed, config, device)
    metadata, anchor = {}, None
    phases = schedule(seed, config)
    for phase in phases:
        if phase['name'].startswith('validation_'):
            assessment(directory, config, identity, seed, model, phase['assessment'], learner, metadata, anchor)
        learner, metadata, anchor, _ = previous.run_phase(config, identity, seed, model,
            phase, learner, metadata, anchor, directory)
    result = dict(identity=identity, seed=seed, model=model, affinity=actual,
        phase_names=[p['name'] for p in phases],
        phase_hashes={p['name']:sha(directory/p['name']/'result.json') for p in phases},
        assessment_hashes={str(i):sha(directory/f'diagnostic_{i}'/'result.json') for i in (0, 1)},
        diagnostics=learner.diagnostics(), final_signature=state_signature(learner),
        prequential_work=dict(forwards=learner.cost['arrivals']//config['batch_size'],
            query_presentations=learner.cost['arrivals'], support_presentations=learner.cost['arrivals']))
    write_json(directory/'result.json', result)
    print(json.dumps(dict(seed=seed, model=model, trajectory_completed=True)), flush=True)
    return dict(seed=seed, model=model, affinity=actual)


def locked_analysis_files(config_path, protocol):
    # The independent portable verifier is sealed separately as packaging work.
    paths = [p for p in (ROOT/'scripts').glob('*.py') if p.name != 'verify_rehearsal_state_archive.py']
    paths += [p for p in (ROOT/'tests').glob('test_rehearsal_state*.py')
              if p.name != 'test_rehearsal_state_archive.py']
    if config_path:
        paths.append(Path(config_path).resolve())
    if protocol:
        paths.append(Path(protocol).resolve())
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}


def check_locks(output, config=None, protocol=None, device='cpu', verify_live=True):
    output = Path(output)
    manifest = read_json(output/'manifest.json')
    identity = manifest['identity']
    if config is not None and manifest['config'] != config:
        raise ValueError('configuration changed after locking')
    for key, value in (('config_sha256', manifest['config']), ('source_sha256', manifest['source_files']),
                       ('runtime_sha256', manifest['runtime'])):
        if identity[key] != digest(value):
            raise ValueError('manifest identity mismatch')
    if verify_live and (manifest['source_files'] != source_manifest() or manifest['runtime']['device'] != device):
        raise ValueError('source or device changed after locking')
    with zipfile.ZipFile(output/'training_source.zip') as archive:
        if (len(archive.namelist()) != len(manifest['source_files'])
                or set(archive.namelist()) != set(manifest['source_files']) or archive.testzip() is not None
                or any(hashlib.sha256(archive.read(n)).hexdigest() != h for n,h in manifest['source_files'].items())):
            raise ValueError('training source archive mismatch')
    lock = verify_analysis(output) if verify_live else read_json(output/'analysis_lock.json')
    if lock['config'] != manifest['config'] or lock['config_sha256'] != identity['config_sha256']:
        raise ValueError('analysis configuration mismatch')
    protocol_lock = read_json(output/'protocol_lock.json')
    if (any(protocol_lock[k] != v for k,v in identity.items())
            or sha(output/'protocol_at_lock.md') != protocol_lock['protocol_sha256']
            or (protocol is not None and Path(protocol).read_bytes() != (output/'protocol_at_lock.md').read_bytes())):
        raise ValueError('protocol lock mismatch')
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
            raise ValueError('a new study requires an empty output directory')
        now = datetime.now(timezone.utc).isoformat()
        manifest = dict(identity=identity, config=config, runtime=runtime, source_files=sources, created_utc=now)
        with zipfile.ZipFile(output/'training_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(ROOT/'src'/name, name)
        files = locked_analysis_files(config_path, protocol)
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
            raise ValueError('completion identity mismatch')
        return completed
    jobs = [(seed, model) for model in config['models'] for seed in config['seeds']]
    finished = []
    if config['workers'] == 1:
        for seed, model in jobs:
            finished.append(run_job(config, identity, seed, model, output, device))
    else:
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            futures = [pool.submit(run_job, config, identity, seed, model, output, device) for seed,model in jobs]
            for future in as_completed(futures):
                finished.append(future.result())
    check_locks(output, config, protocol, device)
    completed = dict(identity=identity, **counts(config), jobs=sorted(finished, key=lambda r:(r['model'], r['seed'])),
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
