"""Verify physical records, replay evolution, and saved-boundary forecasts.

No optimizer updates are rerun. Entry/final checkpoint forecasts are exact
reconstructions; intermediate forecasts retain arithmetic/provenance coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from acp_cl.acquisition.world import AcquisitionWorld, mask_affected, mask_valid, signal_images
from acp_cl.causal_access.study import rng_signature
from acp_cl.conditional.learner import Packet
from acp_cl.core_residual.evaluation import TraceEvaluator
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest
from acp_cl.predictive_value.mechanism import predict_all
from acp_cl.predictive_value.study import load
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import affinity, write_json
from acp_cl.representation_learning.design import (counts, law_from_record, law_key, main_phases,
                                                   phase_laws, qualification_phases, validate_config)
from acp_cl.representation_learning.learner import RepresentationLearner, learner_signature
from acp_cl.representation_learning.study import (check_locks, read_json, require_phase, sha,
                                                  training_seed, verify_job, verify_phase)


def equal(actual, expected, description, checked):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not np.array_equal(actual, expected)):
        raise AssertionError(f'Exact array mismatch: {description}')
    checked['array_checks'] += 1
    checked['scalar_checks'] += actual.size


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify_training(config, seed, arm, phase, record, arrays, before, after, checked):
    laws = phase_laws(phase, config['batch_size'])
    distinct = list(dict.fromkeys(laws))
    if len(record['packets']) != len(laws) or record['batches_done'] != len(laws):
        raise AssertionError('Training packet coverage mismatch')
    if any(len(value) != len(laws) for value in arrays.values()):
        raise AssertionError('Training arrays do not cover every packet')
    memory, history = copy.deepcopy(before.memory), copy.deepcopy(before.history)
    world, updates = AcquisitionWorld(), after.updates_per_batch
    work, marginal = Counter(), {}
    peak = before.peak_memory_bytes
    curve_arrivals = {row['arrivals'] for row in record['curve']}
    curve_supports = {}
    for index, law in enumerate(laws):
        data_seed = training_seed(seed, phase['name'], index)
        data = world.experience(law, config['batch_size'], data_seed)
        cases = world.dataset(law, config['batch_size'], data_seed)
        truth = world.counterfactuals(cases)
        expected = dict(observations=data.observations, truth=truth, actions=data.actions,
            outcomes=data.survival, affected=mask_affected(cases, phase['cue']),
            valid=mask_valid(cases, phase['branch']), law_index=np.asarray(distinct.index(law)))
        for name, value in expected.items():
            equal(arrays[name][index], value, f'{phase["name"]}/{index}/{name}', checked)
        equal(truth[np.arange(len(data)), data.actions], data.survival, 'performed physical truth', checked)
        probabilities = arrays['probabilities'][index]
        if (probabilities.shape != (len(data), 5, 3) or probabilities.dtype != np.float32
                or not np.isfinite(probabilities).all() or probabilities.min() < -np.finfo(np.float32).eps
                or probabilities.max() > 1 + np.finfo(np.float32).eps
                or (np.diff(probabilities, axis=-1) > 0).any()):
            raise AssertionError('Invalid saved ordinary forecast')
        if index == 0:
            equal(probabilities, predict_all(before, data.observations, before.history),
                  'first pre-feedback ordinary forecast', checked)
            checked['ordinary_forecasts_reconstructed'] += 1
        else:
            checked['ordinary_forecasts_not_reconstructed'] += 1
        ids = list(memory.ids)
        replay_ids = []
        for _ in range(updates):
            if ids:
                replay_ids.append(ids[int(memory.sampling_rng.integers(0, len(ids)))])
                work['replay_presentations'] += len(data)
            else:
                replay_ids.append(memory.seen)
                work['duplicate_presentations'] += len(data)
        packet_record = dict(index=index, id=memory.seen, query_sha256=data.fingerprint(),
            support_sha256=None if history is None else history.fingerprint(),
            memory_ids_before=ids, replay_ids=replay_ids)
        if record['packets'][index] != packet_record:
            raise AssertionError(f'Wrong causal context or replay draws at packet {index}')
        memory.add(Packet(history, data))
        peak = max(peak, memory.nbytes())
        history = data
        entry = marginal.setdefault(law_key(law), dict(count=np.zeros(5, dtype=np.int64),
                                                      success=np.zeros((5, 3), dtype=np.int64)))
        for action in range(5):
            selected = data.actions == action
            entry['count'][action] += selected.sum()
            entry['success'][action] += data.survival[selected].astype(np.int64).sum(axis=0)
        arrivals = (index + 1) * len(data)
        if arrivals in curve_arrivals:
            curve_supports[arrivals] = data
        work['arrivals'] += len(data)
        checked['training_packets'] += 1
        checked['training_arrivals'] += len(data)
        checked['replay_choices'] += updates
    normalized = {key: {field: value.tolist() for field, value in entry.items()}
                  for key, entry in marginal.items()}
    if record['marginals'] != normalized:
        raise AssertionError('Training-only per-law marginals changed')
    if memory_state(memory) != memory_state(after.memory) or history.fingerprint() != after.history.fingerprint():
        raise AssertionError('Final history/replay packets or private RNG states do not reconstruct')
    if peak != after.peak_memory_bytes:
        raise AssertionError('Peak replay byte accounting differs')
    steps, batch = len(laws) * updates, config['batch_size']
    auxiliary = arm in ('sequence', 'final_frame')
    diagnostics = after.diagnostics()
    work.update(optimizer_steps=steps, query_presentations=2 * batch * steps,
        support_presentations=2 * batch * steps, encoder_forward_calls=2 * steps,
        encoder_presentations=4 * batch * steps, auxiliary_forward_calls=2 * steps if auxiliary else 0,
        auxiliary_reconstructions=2 * batch * steps if auxiliary else 0,
        reconstruction_pixels=2 * batch * steps * 768 if auxiliary else 0,
        auxiliary_backward_steps=steps if auxiliary else 0,
        predictive_parameter_updates=diagnostics['predictive_parameters'] * steps,
        auxiliary_parameter_updates=diagnostics['auxiliary_parameters'] * steps)
    for name, expected in work.items():
        if after.cost[name] - before.cost[name] != expected:
            raise AssertionError(f'Incorrect work increment: {name}')
    checked['optimizer_steps_counted'] += steps
    return curve_supports


def verify_evaluations(config, seed, phase, record, directory, before, after, supports, checked):
    metadata = read_json(directory / 'evaluations.json')
    arrays = read_arrays(directory / 'evaluations.npz')
    evaluator = TraceEvaluator(seed, config, dict(records=metadata['records'], arrays=arrays))
    plan = {}

    def register(index, law, support, cue=None, branch='novel', flipped=False):
        if index in plan:
            raise AssertionError('Evaluation index used twice')
        plan[index] = (law, support, cue, branch, flipped)

    laws = phase_laws(phase, config['batch_size'])
    register(record['start_eval']['trace'], laws[0], before.history, phase['cue'], phase['branch'])
    expected_arrivals = [index * config['batch_size'] for index in range(1, len(laws) + 1)
                         if index * config['batch_size'] % config['probe_every'] == 0 or index == len(laws)]
    if [row['arrivals'] for row in record['curve']] != expected_arrivals:
        raise AssertionError('Incorrect online probe schedule')
    for row in record['curve']:
        law = laws[row['arrivals'] // config['batch_size'] - 1]
        if law != law_from_record(row['law']):
            raise AssertionError('Probe law does not match the arrival schedule')
        register(row['metrics']['trace'], law, supports[row['arrivals']], phase['cue'], phase['branch'])
    if len(record['end_probes']) != len(phase['targets']):
        raise AssertionError('Endpoint target coverage mismatch')
    for target, row in zip(phase['targets'], record['end_probes'], strict=True):
        law = law_from_record(target)
        if law_from_record(row['law']) != law or set(row['flips']) != {str(cue) for cue in law.active}:
            raise AssertionError('Endpoint law/cue coverage mismatch')
        if len(row['correct']) != config['support_replicates']:
            raise AssertionError('Endpoint replicate coverage mismatch')
        for rep, index in enumerate(row['correct']):
            support = evaluator.support(law, rep)
            register(index, law, support)
            for cue in law.active:
                if len(row['flips'][str(cue)]) != config['support_replicates']:
                    raise AssertionError('Cue replicate coverage mismatch')
                register(row['flips'][str(cue)][rep], law, support, cue=cue, flipped=True)
    valid_expected = config['kind'] == 'main' and not phase['name'].startswith('prefix_')
    if (record['valid_after'] is not None) != valid_expected:
        raise AssertionError('Retention panel coverage mismatch')
    if valid_expected:
        next_index = len(plan)
        for mode in (0, 1):
            law = replace(laws[-1], mode=mode, noise=0.)
            for rep in range(config['support_replicates']):
                register(next_index, law, evaluator.support(law, rep, True, phase['branch']), branch=phase['branch'])
                next_index += 1
    if set(plan) != set(range(len(evaluator.records))):
        raise AssertionError('Incomplete evaluation call plan')
    models = {state_hash(before.model.state_dict()): before, state_hash(after.model.state_dict()): after}
    for index, row in enumerate(evaluator.records):
        law, support, cue, branch, flipped = plan[index]
        if (law_from_record(row['law']) != law or row['cue'] != cue
                or row['branch'] != branch or row['flipped'] != flipped):
            raise AssertionError(f'Wrong evaluation condition at call {index}')
        cases, truth = evaluator.dataset(law)
        observations = cases.observations
        if flipped:
            signals = cases.signals.copy()
            signals[:, cue] ^= 1
            observations = signal_images(observations, signals)
        for name, value in (('observations', observations), ('truth', truth),
                             ('affected', mask_affected(cases, cue)), ('valid', mask_valid(cases, branch))):
            equal(arrays[f'e{index}_{name}'], value, f'evaluation {index} {name}', checked)
        fingerprint = None if support is None else support.fingerprint()
        if row['support_fingerprint'] != fingerprint:
            raise AssertionError('Evaluation used incorrect preceding/fresh support')
        if support is not None:
            for name, value in (('observations', support.observations), ('actions', support.actions),
                                 ('outcomes', support.survival)):
                equal(arrays[f'e{index}_support_{name}'], value, f'evaluation {index} support {name}', checked)
        if row['model_sha256'] in models:
            equal(arrays[f'e{index}_probabilities'], predict_all(models[row['model_sha256']], observations, support),
                  f'evaluation {index} saved-model forecast', checked)
            checked['evaluation_forecasts_reconstructed'] += 1
        else:
            checked['evaluation_forecasts_not_reconstructed'] += 1
        checked['evaluation_calls'] += 1
    for name in ('query_presentations', 'support_presentations', 'observed_support_presentations'):
        if metadata[name] != sum(row[name] for row in evaluator.records):
            raise AssertionError('Evaluation presentation accounting differs')
    if metadata['calls'] != len(evaluator.records):
        raise AssertionError('Evaluation call accounting differs')


def audit_job(run, config, identity, seed, arm):
    affinity(config['cpu_affinity'])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng_before = rng_signature()
    directory = Path(run) / 'jobs' / f'{arm}_{seed}'
    job = verify_job(directory, identity)
    phases = (qualification_phases if config['kind'] == 'qualification' else main_phases)(seed, config)
    if set(job['phase_hashes']) != {f'{phase["name"]}/result.json' for phase in phases}:
        raise AssertionError('Job phase coverage differs from the protocol')
    initial = learner_signature(RepresentationLearner(arm, seed, config))
    prior, total, rows = initial, Counter(), []
    for phase in phases:
        folder = directory / phase['name']
        record = verify_phase(folder, identity)
        require_phase(record, config, seed, arm, phase)
        start, final = load(folder / 'before.pt', identity), load(folder / 'checkpoint.pt', identity)
        before, after = start['learner'], final['learner']
        signatures = [learner_signature(model) for model in (before, after)]
        if (signatures != [record['start_signature'], record['end_signature']]
                or signatures[0] != (initial if config['kind'] == 'qualification' else prior)
                or digest(final['record']) != digest(record)):
            raise AssertionError('Saved boundary state, fresh initialization, or phase continuation mismatch')
        if digest(before.diagnostics()) != digest(record['start_diagnostics']) or digest(after.diagnostics()) != digest(record['diagnostics']):
            raise AssertionError('Saved diagnostics differ from checkpoint states')
        checked = Counter(restored_checkpoints=2, phases=1)
        training = read_arrays(folder / 'training.npz')
        supports = verify_training(config, seed, arm, phase, record, training, before, after, checked)
        verify_evaluations(config, seed, phase, record, folder, before, after, supports, checked)
        if [learner_signature(model) for model in (before, after)] != signatures:
            raise AssertionError('Auditing changed saved learner state')
        prior = signatures[1]
        total.update(checked)
        rows.append(dict(phase=phase['name'], result_sha256=sha(folder / 'result.json'),
                         before_sha256=sha(folder / 'before.pt'), checkpoint_sha256=sha(folder / 'checkpoint.pt'),
                         checks=dict(checked)))
        print(f'Audited {arm} seed {seed}: {phase["name"]}', flush=True)
    if rng_signature() != rng_before:
        raise AssertionError('Audit changed global RNG state')
    verify_job(directory, identity)
    return dict(seed=seed, arm=arm, result_sha256=sha(directory / 'result.json'),
                learner_states_unchanged=True, global_rng_unchanged=True, checks=dict(total), phases=rows)


def audit(run, output, workers=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError('Keep verification output outside the completed run')
    manifest = check_locks(run)
    config, identity = manifest['config'], manifest['identity']
    validate_config(config)
    completion = read_json(run / 'completion.json')
    if completion['identity'] != identity or manifest['expected_counts'] != counts(config):
        raise AssertionError('Completion identity or declared counts differ')
    tasks = [(seed, arm) for seed in config['seeds'] for arm in config['arms']]
    expected_paths = {f'jobs/{arm}_{seed}/result.json' for seed, arm in tasks}
    if len(completion['jobs']) != len(tasks) or {row['path'] for row in completion['jobs']} != expected_paths:
        raise AssertionError('Completed job coverage differs')
    if any(sha(run / row['path']) != row['sha256'] for row in completion['jobs']):
        raise AssertionError('Completed job result changed')
    workers = config['workers'] if workers is None else workers
    if type(workers) is not int or not 1 <= workers <= config['workers']:
        raise ValueError('Audit workers must stay within the declared resource budget')
    rows = []
    if workers == 1:
        rows = [audit_job(run, config, identity, seed, arm) for seed, arm in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(audit_job, str(run), config, identity, seed, arm) for seed, arm in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    total = Counter()
    for row in rows:
        total.update(row['checks'])
    expected = counts(config)
    for key in ('phases', 'training_arrivals'):
        if total[key] != expected[key]:
            raise AssertionError('Audit work differs from declared totals')
    if total['optimizer_steps_counted'] != expected['optimizer_steps']:
        raise AssertionError('Audited optimizer count differs from the declared total')
    check_locks(run)
    result = dict(status='passed', run=str(run), study=config['study'], identity=identity,
        audit_script_sha256=sha(__file__), completed_utc=datetime.now(timezone.utc).isoformat(),
        workers=workers, threads=1, jobs=len(rows), checks=dict(total),
        learner_states_unchanged=True, global_rng_unchanged=True,
        limitations=['No optimizer updates were rerun.',
                     'Ordinary interior and intermediate-model evaluation forecasts were not reconstructed.',
                     'The audit adds no outcome selection or promotion criteria.'],
        job_records=sorted(rows, key=lambda row: (row['seed'], row['arm'])))
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    result = audit(args.run, args.output, args.workers)
    print({key: result[key] for key in ('status', 'jobs', 'checks')})
