"""Locked fresh fitting followed by causally scored frozen switching streams."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import platform
import random
import zipfile

import numpy as np
import torch

from acp_cl.acquisition.world import (AcquisitionWorld, Law, mask_affected, order,
                                      signal_images, stage_law)
from acp_cl.core_residual.design import prefix_phases
from acp_cl.core_residual.evaluation import array_hash
from acp_cl.core_residual.learner import fork_core, learner_signature, new_from_scratch
from acp_cl.core_residual.study import episode, save_arrays, sha, source_manifest as prior_sources
from acp_cl.core_residual_outputs import component_predictions
from acp_cl.persistence.study import digest, trial_seed
from acp_cl.persistence.world import Experience
from acp_cl.predictive_value.study import read_json
from acp_cl.replay_renewal.study import affinity, new_learner, write_json
from .design import POLICIES, counts, phases, reference_phase, validate_config
from .mechanism import LaggedMixer, mix_predictions


ROOT = Path(__file__).resolve().parents[3]


def source_manifest():
    files = prior_sources()
    paths = [ROOT/'src/acp_cl/core_residual_outputs.py', *Path(__file__).parent.glob('*.py')]
    files.update({p.relative_to(ROOT/'src').as_posix(): sha(p) for p in sorted(paths)})
    return files


def analysis_files(config_path, protocol):
    paths = [ROOT/'scripts/summarize_causal_access.py', *ROOT.glob('tests/test_causal_access*.py'),
             Path(config_path).resolve(), Path(protocol).resolve()]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}


def rng_signature():
    state = np.random.get_state()
    return digest(dict(python=repr(random.getstate()), numpy=[state[0], state[1].tolist(),
        state[2], state[3], state[4]], torch=array_hash(torch.get_rng_state().numpy())))


def flipped_images(cases, cue):
    signals = cases.signals.copy()
    signals[:, cue] ^= 1
    return signal_images(cases.observations, signals)


class Forecaster:
    """Count executed shared forwards; never update a learner or its history."""

    def __init__(self, learners):
        self.learners = learners
        self.work = {key: dict(component_calls=0, query_presentations=0,
                              support_presentations=0, path_presentations=0) for key in learners}

    def predict(self, name, observations, support):
        record = self.work[name]
        record['component_calls'] += 1
        record['query_presentations'] += len(observations)
        record['support_presentations'] += len(support)
        # component_predictions executes the core and residual once each.
        record['path_presentations'] += 2 * (len(observations) + len(support))
        return component_predictions(self.learners[name], observations, support)


def qualification(config, seed, forecaster, arrays):
    world, cue = AcquisitionWorld(), order(seed)[0]
    for condition, law in (('old', Law(seed % 2)), ('novel', stage_law(seed, 1))):
        prefix = f'q_{condition}'
        cases = world.dataset(law, config['eval_size'], trial_seed(seed, 'causal_access_qual_query', 0))
        arrays[prefix+'_observations'] = cases.observations.copy()
        arrays[prefix+'_truth'] = world.counterfactuals(cases)
        arrays[prefix+'_affected'] = mask_affected(cases, None if condition == 'old' else cue)
        flipped = flipped_images(cases, cue) if condition == 'novel' else None
        if flipped is not None:
            arrays[prefix+'_flipped_observations'] = flipped
        collected = defaultdict(list)
        for rep in range(config['support_replicates']):
            support = world.experience(law, config['batch_size'],
                trial_seed(seed, 'causal_access_qual_support', rep))
            for field, value in (('observations', support.observations), ('actions', support.actions),
                                 ('outcomes', support.survival)):
                collected['support_'+field].append(value.copy())
            for suffix, observations in (('', cases.observations), ('_flipped', flipped)):
                if observations is None:
                    continue
                separate = forecaster.predict('separate', observations, support)
                values = dict(core=separate['core'], full=separate['full'],
                    joint=forecaster.predict('joint', observations, support)['full'],
                    reference=forecaster.predict('reference_'+condition, observations, support)['full'])
                for name, value in values.items():
                    collected[name+suffix].append(value)
        arrays.update({prefix+'_'+key: np.asarray(value) for key, value in collected.items()})


def frozen_stream(config, seed, forecaster, arrays):
    world, cue = AcquisitionWorld(), order(seed)[0]
    history = forecaster.learners['separate'].history
    if history is None or history.fingerprint() != forecaster.learners['joint'].history.fingerprint():
        raise AssertionError('matched learners must start with the same actual novel training history')
    for field, value in (('observations', history.observations), ('actions', history.actions),
                         ('outcomes', history.survival)):
        arrays['initial_support_'+field] = value.copy()
    mixer = LaggedMixer(config['mixer_capacity'], config['mixer_clip'])
    initial, supports, weights, collected = history.fingerprint(), [], [], defaultdict(list)
    batch_size = config['batch_size']
    packets = config['stream_blocks'] * config['stream_block_size'] // batch_size
    for index in range(packets):
        block = index * batch_size // config['stream_block_size']
        is_novel = bool(block % 2)
        law = stage_law(seed, 1) if is_novel else Law(seed % 2)
        data_seed = trial_seed(seed, 'causal_access_stream', index)
        cases = world.dataset(law, batch_size, data_seed)
        actions = np.random.default_rng(trial_seed(data_seed, 'performed_action', 0)).integers(
            0, 5, batch_size, dtype=np.uint8)
        supports.append(history.fingerprint())
        weight = mixer.weight()
        weights.append(weight)
        separate = forecaster.predict('separate', cases.observations, history)
        core, full = separate['core'], separate['full']
        values = dict(core=core, full=full, half=mix_predictions(core, full, .5),
            causal=mixer.predict(core, full),
            joint=forecaster.predict('joint', cases.observations, history)['full'])
        flipped = flipped_images(cases, cue) if is_novel else cases.observations.copy()
        if is_novel:
            probe = forecaster.predict('separate', flipped, history)
            flipped_values = dict(core=probe['core'], full=probe['full'],
                half=mix_predictions(probe['core'], probe['full'], .5),
                causal=mix_predictions(probe['core'], probe['full'], weight),
                joint=forecaster.predict('joint', flipped, history)['full'])
        else:
            flipped_values = values
        # The only release of current outcomes is after every ordinary/probe forecast.
        truth = world.counterfactuals(cases)
        outcomes = truth[np.arange(batch_size), actions].copy()
        record = dict(observations=cases.observations, flipped_observations=flipped,
            actions=actions, outcomes=outcomes, truth=truth,
            affected=mask_affected(cases, cue if is_novel else None),
            is_novel=np.full(batch_size, is_novel, dtype=bool),
            block=np.full(batch_size, block, dtype=np.int64))
        record.update(values)
        record.update({name+'_flipped': value for name, value in flipped_values.items()})
        if set(values) != set(POLICIES):
            raise AssertionError('missing policy')
        for name, value in record.items():
            collected[name].append(value.copy())
        mixer.observe(core, full, actions, outcomes)
        history = Experience(cases.observations.copy(), actions.copy(), outcomes)
        # learner.history, replay, gradients, parameters and optimizer are not assigned.
    arrays.update({'stream_'+key: np.concatenate(value) for key, value in collected.items()})
    arrays['stream_weights'] = np.asarray(weights, dtype=np.float64)
    return dict(packet_count=packets, initial_support=initial, support_fingerprints=supports,
                mixer_final=mixer.state(), mixer_signature=mixer.signature())


def verify_job(directory, identity):
    directory = Path(directory)
    result = read_json(directory/'result.json')
    if result['identity'] != identity or sha(directory/result['artifact']['file']) != result['artifact']['sha256']:
        raise ValueError('completed forecast artifact or identity changed')
    for field in ('phase_hashes', 'checkpoint_hashes'):
        if any(sha(directory/name) != value for name, value in result[field].items()):
            raise ValueError('completed fit artifact changed')
    for name in result['phase_hashes']:
        path = directory/name
        record = read_json(path)
        if any(sha(path.parent/artifact) != value for artifact, value in record['artifact_hashes'].items()):
            raise ValueError('completed fit raw data changed')
    if result['frozen_before'] != result['frozen_after'] or result['rng_before'] != result['rng_after']:
        raise ValueError('evaluation changed frozen state')
    return result


def run_job(config, identity, seed, model, output):
    actual = affinity(config['cpu_affinity'])
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/'jobs'/f'{model}_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'result.json').exists():
        verify_job(directory, identity)
    else:
        base = new_learner(model, seed, config)
        phase_paths, learners, marginals, insertion = [], {}, {}, {}

        def fit(learner, arm, phase, relative):
            path = directory/'fit'/relative
            fitted, record = episode(config, identity, learner, seed, model, arm, phase, path)
            phase_paths.append(path)
            return fitted, record

        for phase in prefix_phases(seed, config):
            base, _ = fit(base, 'prefix', phase, phase['name'])
        parent_signature = learner_signature(base)
        for arm in ('joint', 'separate'):
            learner = fork_core(base, arm, seed, config)
            insertion[arm] = dict(signature=learner_signature(learner), diagnostics=learner.diagnostics())
            for phase in phases(seed, config):
                learner, _ = fit(learner, arm, phase, f"{arm}/{phase['name']}")
            learners[arm] = learner
        if learner_signature(base) != parent_signature:
            raise AssertionError('shared prefix changed')
        for condition, offset in (('old', 191311), ('novel', 193321)):
            reference = new_from_scratch(model, seed+offset, config)
            phase = reference_phase(seed, config, condition)
            reference, record = fit(reference, 'reference_'+condition, phase, phase['name'])
            learners['reference_'+condition] = reference
            marginals[condition] = dict(counts=record['marginal_count'], success=record['marginal_success'])
        frozen_before = {name: learner_signature(learner) for name, learner in learners.items()}
        rng_before = rng_signature()
        forecaster, arrays = Forecaster(learners), {}
        qualification(config, seed, forecaster, arrays)
        stream = frozen_stream(config, seed, forecaster, arrays)
        frozen_after = {name: learner_signature(learner) for name, learner in learners.items()}
        rng_after = rng_signature()
        if frozen_before != frozen_after or rng_before != rng_after:
            raise AssertionError('frozen evaluation changed learner or global RNG state')
        save_arrays(directory/'forecasts.npz', **arrays)
        result = dict(identity=identity, seed=seed, model=model, cue=order(seed)[0], affinity=actual,
            artifact=dict(file='forecasts.npz', sha256=sha(directory/'forecasts.npz')),
            array_sha256={key: array_hash(value) for key, value in arrays.items()},
            phase_hashes={(path/'result.json').relative_to(directory).as_posix(): sha(path/'result.json')
                          for path in phase_paths},
            checkpoint_hashes={(path/'checkpoint.pt').relative_to(directory).as_posix(): sha(path/'checkpoint.pt')
                               for path in phase_paths},
            parent_signature=parent_signature, insertion=insertion,
            frozen_before=frozen_before, frozen_after=frozen_after, rng_before=rng_before, rng_after=rng_after,
            stream=stream, reference_marginals=marginals, evaluation_work=forecaster.work,
            final_training_diagnostics={name: learner.diagnostics() for name, learner in learners.items()})
        write_json(directory/'result.json', result)
        print(f'Frozen stream completed: {model} seed {seed}', flush=True)
    return dict(seed=seed, model=model, affinity=actual, result_sha256=sha(directory/'result.json'))


def check_locks(output, config=None, protocol=None, verify_live=True):
    output = Path(output)
    manifest = read_json(output/'manifest.json')
    for name, value in manifest['archives'].items():
        if sha(output/name) != value:
            raise ValueError('locked archive changed')
    if sha(output/'protocol_at_lock.md') != manifest['protocol_sha256']:
        raise ValueError('locked protocol changed')
    if config is not None and config != manifest['config']:
        raise ValueError('configuration changed')
    if protocol is not None and sha(protocol) != manifest['protocol_sha256']:
        raise ValueError('live protocol changed')
    if verify_live:
        if source_manifest() != manifest['source_files']:
            raise ValueError('scientific source changed after lock')
        if any(sha(ROOT/name) != value for name, value in manifest['analysis_files'].items()):
            raise ValueError('scientific analysis changed after lock')
    expected = dict(config_sha256=digest(manifest['config']), source_sha256=digest(manifest['source_files']),
        analysis_sha256=digest(manifest['analysis_files']), runtime_sha256=digest(manifest['runtime']))
    if expected != manifest['identity']:
        raise ValueError('manifest identity is inconsistent')
    return manifest


def run_suite(config_path, output, protocol, resume=False, lock_only=False):
    config = read_json(config_path)
    validate_config(config)
    actual = affinity(config['cpu_affinity'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources, analysis = source_manifest(), analysis_files(config_path, protocol)
    runtime = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, device='cpu', threads=config['threads'], workers=config['workers'],
        deterministic=True, affinity=actual)
    identity = dict(config_sha256=digest(config), source_sha256=digest(sources),
        analysis_sha256=digest(analysis), runtime_sha256=digest(runtime))
    if (output/'manifest.json').exists():
        if not resume:
            raise ValueError('existing study requires explicit --resume')
        manifest = check_locks(output, config, protocol)
        if identity != manifest['identity']:
            raise ValueError('source, analysis or runtime differs from locked study')
    else:
        if any(output.iterdir()):
            raise ValueError('new study requires an empty output directory')
        archives = {}
        for name, files, base in (('training_source.zip', sources, ROOT/'src'),
                                   ('analysis_at_lock.zip', analysis, ROOT)):
            with zipfile.ZipFile(output/name, 'w', zipfile.ZIP_DEFLATED) as archive:
                for relative in files:
                    archive.write(base/relative, relative)
            archives[name] = sha(output/name)
        (output/'protocol_at_lock.md').write_bytes(Path(protocol).read_bytes())
        manifest = dict(identity=identity, config=config, runtime=runtime, source_files=sources,
            analysis_files=analysis, archives=archives, protocol_sha256=sha(protocol),
            created_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'manifest.json', manifest)
    if lock_only:
        print(dict(locked=True, identity=identity, **counts(config)), flush=True)
        return manifest
    jobs = [(seed, model) for model in config['models'] for seed in config['seeds']]
    finished = []
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures = [pool.submit(run_job, config, identity, seed, model, output) for seed, model in jobs]
        for future in as_completed(futures):
            finished.append(future.result())
    check_locks(output, config, protocol)
    completed = dict(identity=identity, **counts(config),
        job_records=sorted(finished, key=lambda row: (row['model'], row['seed'])),
        completed_utc=datetime.now(timezone.utc).isoformat())
    if (output/'completion.json').exists():
        old = read_json(output/'completion.json')
        if any(old[key] != completed[key] for key in completed if key != 'completed_utc'):
            raise ValueError('completion changed during verification')
        return old
    write_json(output/'completion.json', completed)
    print(dict(completed=True, **counts(config)), flush=True)
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lock-only', action='store_true')
    args = parser.parse_args()
    run_suite(args.config, args.output, args.protocol, args.resume, args.lock_only)
