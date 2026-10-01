"""Reconstruct sampled frozen forecasts and physical data without refitting.

The sample is fixed: first and last packet of every stream block, plus every
qualification replicate and its declared cue intervention. No decision metrics
are calculated here. The separate summary verifies arithmetic over all rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from acp_cl.acquisition.world import AcquisitionWorld, Law, mask_affected, order, signal_images, stage_law
from acp_cl.causal_access.design import counts, validate_config
from acp_cl.causal_access.study import check_locks, rng_signature, verify_job
from acp_cl.core_residual.evaluation import array_hash
from acp_cl.core_residual.learner import learner_signature
from acp_cl.core_residual.study import sha
from acp_cl.core_residual_outputs import component_predictions
from acp_cl.persistence.study import trial_seed
from acp_cl.predictive_value.study import load, read_json
from acp_cl.replay_renewal.study import affinity, write_json


def equal(actual, expected, description, checked):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise AssertionError(f'Exact array reconstruction failed: {description}')
    checked['exact_array_checks'] += 1
    checked['exact_scalar_checks'] += actual.size


def flip(cases, cue):
    signals = cases.signals.copy()
    signals[:, cue] ^= 1
    return signal_images(cases.observations, signals)


def native(learners, observations, support, checked, reference=None):
    separate = component_predictions(learners['separate'], observations, support)
    joint = component_predictions(learners['joint'], observations, support)
    values = dict(core=separate['core'], full=separate['full'], joint=joint['full'])
    checked['component_calls'] += 2
    if reference is not None:
        values['reference'] = component_predictions(learners[reference], observations, support)['full']
        checked['component_calls'] += 1
    return values


def check_work(record, config):
    batch, query = config['batch_size'], config['eval_size']
    reps = config['support_replicates']
    packets = config['stream_blocks'] * config['stream_block_size'] // batch
    expected = {}
    for name in ('separate', 'joint', 'reference_old', 'reference_novel'):
        qualification_calls = reps * (1 if name == 'reference_old' else 2 if name == 'reference_novel' else 3)
        stream_calls = 3 * packets // 2 if name in ('separate', 'joint') else 0
        calls = qualification_calls + stream_calls
        query_presentations = qualification_calls * query + stream_calls * batch
        support_presentations = calls * batch
        expected[name] = dict(component_calls=calls, query_presentations=query_presentations,
            support_presentations=support_presentations,
            path_presentations=2 * (query_presentations + support_presentations))
    if record['evaluation_work'] != expected:
        raise AssertionError('Recorded physical evaluation work differs from the declared schedule')
    return expected


def audit_job(run, config, identity, model, seed):
    directory = Path(run) / 'jobs' / f'{model}_{seed}'
    record = verify_job(directory, identity)
    checked = Counter()
    with np.load(directory / record['artifact']['file'], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if {key: array_hash(value) for key, value in arrays.items()} != record['array_sha256']:
        raise AssertionError('Forecast array inventory or hashes changed')
    rng_before = rng_signature()
    learners = {}
    checkpoint_hashes = {}
    for name in ('separate', 'joint', 'reference_old', 'reference_novel'):
        relative = f'fit/{name}/novel/checkpoint.pt' if name in ('separate', 'joint') else f'fit/{name}/checkpoint.pt'
        checkpoint = directory / relative
        if relative not in record['checkpoint_hashes']:
            raise AssertionError('Final checkpoint is not bound by the recorded checkpoint inventory')
        learners[name] = load(checkpoint, identity)['learner']
        checkpoint_hashes[relative] = sha(checkpoint)
    before = {name: learner_signature(learner) for name, learner in learners.items()}
    if before != record['frozen_before']:
        raise AssertionError('Gradient-restored final checkpoints do not match evaluation states')
    work = check_work(record, config)
    world, cue = AcquisitionWorld(), order(seed)[0]
    batch, reps = config['batch_size'], config['support_replicates']
    initial = learners['separate'].history
    if initial.fingerprint() != learners['joint'].history.fingerprint():
        raise AssertionError('Final learner histories do not match')
    for field, value in (('observations', initial.observations), ('actions', initial.actions),
                         ('outcomes', initial.survival)):
        equal(arrays['initial_support_' + field], value, 'initial support ' + field, checked)
    if initial.fingerprint() != record['stream']['initial_support']:
        raise AssertionError('Initial support fingerprint mismatch')

    for condition, law in (('old', Law(seed % 2)), ('novel', stage_law(seed, 1))):
        prefix = 'q_' + condition
        cases = world.dataset(law, config['eval_size'], trial_seed(seed, 'causal_access_qual_query', 0))
        equal(arrays[prefix + '_observations'], cases.observations, prefix + ' images', checked)
        equal(arrays[prefix + '_truth'], world.counterfactuals(cases), prefix + ' truth', checked)
        equal(arrays[prefix + '_affected'], mask_affected(cases, None if condition == 'old' else cue),
              prefix + ' affected subset', checked)
        flipped = flip(cases, cue) if condition == 'novel' else None
        if flipped is not None:
            equal(arrays[prefix + '_flipped_observations'], flipped, prefix + ' cue images', checked)
        for rep in range(reps):
            support = world.experience(law, batch, trial_seed(seed, 'causal_access_qual_support', rep))
            for field, value in (('observations', support.observations), ('actions', support.actions),
                                 ('outcomes', support.survival)):
                equal(arrays[prefix + '_support_' + field][rep], value, prefix + ' support ' + field, checked)
            for suffix, observations in (('', cases.observations), ('_flipped', flipped)):
                if observations is None:
                    continue
                values = native(learners, observations, support, checked, 'reference_' + condition)
                for name, value in values.items():
                    equal(arrays[prefix + '_' + name + suffix][rep], value,
                          f'{prefix}/{rep}/{name}{suffix}', checked)
                    checked['native_forecast_arrays'] += 1
            checked['qualification_replicates'] += 1

    packets_per_block = config['stream_block_size'] // batch
    packets = config['stream_blocks'] * packets_per_block
    sampled = sorted({block * packets_per_block + position for block in range(config['stream_blocks'])
                      for position in (0, packets_per_block - 1)})
    if record['stream']['packet_count'] != packets or len(record['stream']['support_fingerprints']) != packets:
        raise AssertionError('Recorded stream packet counts differ from design')
    if arrays['stream_weights'].shape != (packets,) or arrays['stream_actions'].shape != (packets * batch,):
        raise AssertionError('Saved stream array sizes differ from design')
    if (record['stream']['mixer_final']['samples_seen'] != packets * batch
            or record['stream']['mixer_final']['packets_seen'] != packets):
        raise AssertionError('Mixer feedback counts differ from ordinary stream counts')

    def ordinary(index):
        block = index // packets_per_block
        law = stage_law(seed, 1) if block % 2 else Law(seed % 2)
        data_seed = trial_seed(seed, 'causal_access_stream', index)
        cases = world.dataset(law, batch, data_seed)
        experience = world.experience(law, batch, data_seed)
        interval = slice(index * batch, (index + 1) * batch)
        for field, value in (('observations', experience.observations), ('actions', experience.actions),
                             ('outcomes', experience.survival)):
            equal(arrays['stream_' + field][interval], value, f'packet {index} {field}', checked)
        return block, cases, experience, interval

    for index in sampled:
        block, cases, _, interval = ordinary(index)
        support = initial if index == 0 else ordinary(index - 1)[2]
        if support.fingerprint() != record['stream']['support_fingerprints'][index]:
            raise AssertionError(f'Wrong previous-packet support at stream packet {index}')
        equal(arrays['stream_truth'][interval], world.counterfactuals(cases), f'packet {index} truth', checked)
        equal(arrays['stream_affected'][interval], mask_affected(cases, cue if block % 2 else None),
              f'packet {index} affected subset', checked)
        equal(arrays['stream_block'][interval], np.full(batch, block, dtype=np.int64),
              f'packet {index} block', checked)
        equal(arrays['stream_is_novel'][interval], np.full(batch, bool(block % 2), dtype=bool),
              f'packet {index} condition', checked)
        flipped = flip(cases, cue) if block % 2 else cases.observations.copy()
        equal(arrays['stream_flipped_observations'][interval], flipped, f'packet {index} cue images', checked)
        original_values = None
        for suffix, observations in (('', cases.observations), ('_flipped', flipped)):
            values = (original_values if suffix and not block % 2
                      else native(learners, observations, support, checked))
            if not suffix:
                original_values = values
            for name, value in values.items():
                equal(arrays['stream_' + name + suffix][interval], value,
                      f'packet {index}/{name}{suffix}', checked)
                checked['native_forecast_arrays'] += 1
        checked['sampled_stream_packets'] += 1

    after = {name: learner_signature(learner) for name, learner in learners.items()}
    if before != after or rng_signature() != rng_before:
        raise AssertionError('Forecast reconstruction changed a learner or global random state')
    if any(sha(directory / relative) != value for relative, value in checkpoint_hashes.items()):
        raise AssertionError('A source checkpoint changed during auditing')
    return dict(model=model, seed=seed, result_sha256=sha(directory / 'result.json'),
        forecast_sha256=sha(directory / record['artifact']['file']), checkpoint_sha256=checkpoint_hashes,
        sampled_packet_indices=sampled, checks=dict(checked), learner_states_unchanged=True,
        global_rng_unchanged=True, evaluation_work=work)


def audit(run, report):
    run = Path(run).resolve()
    report = Path(report).resolve()
    if report == run or run in report.parents:
        raise ValueError('Write the verification report outside the completed run directory')
    manifest = check_locks(run)
    config, identity = manifest['config'], manifest['identity']
    validate_config(config)
    completion = read_json(run / 'completion.json')
    if completion['identity'] != identity or any(completion[key] != value for key, value in counts(config).items()):
        raise AssertionError('Completion identity or aggregate counts differ from the locked design')
    actual_affinity = affinity(config['cpu_affinity'])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    jobs, total = [], Counter()
    for model in config['models']:
        for seed in config['seeds']:
            row = audit_job(run, config, identity, model, seed)
            expected = next(item for item in completion['job_records']
                            if item['model'] == model and item['seed'] == seed)
            if expected['result_sha256'] != row['result_sha256']:
                raise AssertionError('Job result changed since completion')
            jobs.append(row)
            total.update(row['checks'])
            print(f'Audited {model} seed {seed}: {row["checks"]}', flush=True)
    check_locks(run)
    result = dict(status='passed', study=config['study'], identity=identity, run=str(run),
        audit_script_sha256=sha(Path(__file__)), completed_utc=datetime.now(timezone.utc).isoformat(),
        sampling_rule='first and last packet of every block; every qualification replicate and cue flip',
        limitations=['ordinary training was not rerun', 'interior stream forecasts were not reconstructed',
                     'this audit adds no outcome metrics or promotion criteria'],
        jobs=len(jobs), restored_checkpoints=4 * len(jobs), checks=dict(total), affinity=actual_affinity,
        threads=1, learner_states_unchanged=True, global_rng_unchanged=True, job_records=jobs)
    write_json(report, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--report', required=True)
    options = parser.parse_args()
    summary = audit(options.run, options.report)
    print({key: summary[key] for key in ('status', 'jobs', 'restored_checkpoints', 'checks')})
