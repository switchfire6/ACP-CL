"""Bounded runner checks for causal data flow, crash recovery, and stop gates."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from acp_cl.acquisition.world import AcquisitionWorld
from acp_cl.causal_access.study import rng_signature
from acp_cl.representation_learning import study
from acp_cl.representation_learning.design import (ARMS, counts, law_key, main_phases,
                                                   phase_laws, qualification_phases, validate_config)
from acp_cl.representation_learning.learner import RepresentationLearner, learner_signature


@pytest.fixture
def config():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return dict(study='representation_test', kind='qualification', seeds=[18099], arms=['outcome'],
        batch_size=8, eval_size=64, probe_every=32, support_replicates=1, width=8,
        experts=4, context_width=4, decoder_width=8, interaction_features=True,
        lr=.002, memory_packets=4, updates_per_batch=1, compute_updates_per_batch=2,
        auxiliary_weight=1., evidence_strength=1., threads=1, workers=1, cpu_affinity=1365,
        episode_size=64, prefix_size=32, interleave_block=32, interleave_cycles=2, smoke=True)


def fit(config, directory, arm='outcome', phase=None):
    seed = config['seeds'][0]
    phase = qualification_phases(seed, config)[0] if phase is None else phase
    return study.fit_phase(config, {'test': 1}, RepresentationLearner(arm, seed, config),
                           seed, arm, phase, directory)


def inventory(directory):
    return {path.relative_to(directory).as_posix(): study.sha(path)
            for path in Path(directory).rglob('*') if path.is_file()}


def sandbox(tmp_path, monkeypatch, config, qualified=False):
    """A tiny temporary archive root isolates lock mechanics from repo edits."""
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    source = tmp_path / 'src/acp_cl/placeholder.py'
    source.parent.mkdir(parents=True)
    source.write_text('value = 1\n', encoding='utf-8')
    script = tmp_path / 'scripts/summarize_representation_learning.py'
    script.parent.mkdir()
    script.write_text(f'def summarize(*args):\n    return {{"qualified": {qualified!r}}}\n', encoding='utf-8')
    config_path, protocol = tmp_path / 'config.json', tmp_path / 'protocol.md'
    study.write_json(config_path, config)
    protocol.write_text('Bounded test protocol.\n', encoding='utf-8')
    return config_path, protocol


def test_schedules_are_counterbalanced_cumulative_and_exactly_counted(config):
    validate_config(config)
    assert counts(config) == dict(jobs=1, phases=4, training_arrivals=512, optimizer_steps=64)
    qualification = qualification_phases(config['seeds'][0], config)
    assert [len(phase_laws(item, 8)[0].active) for item in qualification[:3]] == [1, 2, 3]
    interleaved = phase_laws(qualification[-1], 8)
    assert len(interleaved) == 40
    assert all(interleaved.count(law) == 8 for law in set(interleaved))
    main = dict(config, kind='main', arms=list(ARMS))
    validate_config(main)
    assert counts(main) == dict(jobs=4, phases=44, training_arrivals=2304, optimizer_steps=360)
    phases = main_phases(main['seeds'][0], main)
    laws = [phase_laws(item, 8)[0] for item in phases]
    assert laws[0] == laws[2] == laws[7] and laws[1] == laws[3]
    assert laws[6] == laws[8] == laws[10] and laws[9].revised and not laws[10].revised


def test_complete_phase_reload_is_read_only_and_guards_requested_phase(config, tmp_path):
    directory = tmp_path / 'phase'
    learner, record = fit(config, directory)
    before = inventory(directory)
    reloaded, repeated = fit(config, directory)
    assert learner_signature(reloaded) == learner_signature(learner)
    assert study.digest(repeated) == study.digest(record) and inventory(directory) == before
    changed = deepcopy(qualification_phases(config['seeds'][0], config)[0])
    changed['cue'] = (changed['cue'] + 1) % 3
    with pytest.raises(ValueError, match='phase specification'):
        fit(config, directory, phase=changed)
    changed_config = dict(config, kind='main')
    with pytest.raises(ValueError, match='phase specification'):
        fit(changed_config, directory)
    with pytest.raises(ValueError, match='phase specification'):
        fit(config, directory, arm='sequence')


@pytest.mark.parametrize('crash_at', ('partial', 'final'))
def test_crash_resume_preserves_predictions_gradients_optimizer_and_replay(config, tmp_path, monkeypatch, crash_at):
    config['probe_every'] = 16
    uninterrupted, _ = fit(config, tmp_path / 'uninterrupted', arm='sequence')
    original_save = study.save
    directory = tmp_path / 'resumed'

    def save_then_crash(path, payload):
        original_save(path, payload)
        record = payload['record']
        if Path(path).name == 'checkpoint.pt' and (
                crash_at == 'partial' and record['batches_done'] == 2
                or crash_at == 'final' and 'end_signature' in record):
            raise RuntimeError('simulated interruption after atomic save')

    monkeypatch.setattr(study, 'save', save_then_crash)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        fit(config, directory, arm='sequence')
    assert (directory / 'checkpoint.pt').exists() and not (directory / 'result.json').exists()
    partial = study.load(directory / 'checkpoint.pt', {'test': 1})
    if crash_at == 'partial':
        assert partial['record']['batches_done'] == 2
        changed = deepcopy(qualification_phases(config['seeds'][0], config)[0])
        changed['cue'] = (changed['cue'] + 1) % 3
        with pytest.raises(ValueError, match='phase specification'):
            fit(config, directory, arm='sequence', phase=changed)
    monkeypatch.setattr(study, 'save', original_save)
    resumed, _ = fit(config, directory, arm='sequence')
    assert learner_signature(resumed, ignore_walltime=True) == learner_signature(uninterrupted, ignore_walltime=True)
    for name in ('training.npz', 'evaluations.npz', 'evaluations.json'):
        assert (directory / name).read_bytes() == (tmp_path / 'uninterrupted' / name).read_bytes()


def test_online_forecasts_receive_only_previous_support_before_current_training(config, tmp_path, monkeypatch):
    seed, world = config['seeds'][0], AcquisitionWorld()
    phase = qualification_phases(seed, config)[0]
    laws = phase_laws(phase, config['batch_size'])
    previous, captured, predicted = None, [], study.predict_all
    original_train = RepresentationLearner.train
    events = []

    def predict(learner, observations, support=None):
        assert isinstance(observations, np.ndarray)
        assert (None if support is None else support.fingerprint()) == previous
        assert learner.cost['arrivals'] == len(captured) * config['batch_size']
        before = learner_signature(learner)
        probabilities = predicted(learner, observations, support)
        assert learner_signature(learner) == before
        captured.append(probabilities.copy())
        events.append('predict')
        return probabilities

    def train(learner, current, oracle_modes=None):
        nonlocal previous
        assert events[-1] == 'predict' and oracle_modes is None
        expected = world.experience(laws[len(captured) - 1], config['batch_size'],
            study.training_seed(seed, phase['name'], len(captured) - 1))
        assert current.fingerprint() == expected.fingerprint()
        events.append('train')
        result = original_train(learner, current)
        previous = current.fingerprint()
        return result

    monkeypatch.setattr(study, 'predict_all', predict)
    monkeypatch.setattr(RepresentationLearner, 'train', train)
    rng = rng_signature()
    _, record = fit(config, tmp_path / 'phase')
    assert rng_signature() == rng and events == ['predict', 'train'] * len(laws)
    assert record['packets'][0]['support_sha256'] is None
    for before, after in zip(record['packets'], record['packets'][1:], strict=False):
        assert after['support_sha256'] == before['query_sha256']
    with np.load(tmp_path / 'phase/training.npz', allow_pickle=False) as training:
        np.testing.assert_array_equal(training['probabilities'], np.asarray(captured))


def test_endpoint_probe_guard_detects_training_state_mutation(config):
    seed = config['seeds'][0]
    learner = RepresentationLearner('outcome', seed, config)
    law = phase_laws(qualification_phases(seed, config)[0], config['batch_size'])[0]
    data = AcquisitionWorld().experience(law, config['batch_size'], 67219)
    marginals = {}
    study.update_marginal(marginals, law, data)

    class BadEvaluator:
        settings = config

        def support(self, law, rep):
            return data

        def evaluate(self, actual, *args, **kwargs):
            actual.cost['optimizer_steps'] += 1
            return {'trace': 0}

    with pytest.raises(AssertionError, match='endpoint evaluation changed'):
        study.endpoint_probes(BadEvaluator(), learner, [asdict(law)], marginals)


def test_marginals_count_only_performed_training_outcomes_and_remain_law_specific(config):
    seed, world, marginals = config['seeds'][0], AcquisitionWorld(), {}
    phases = qualification_phases(seed, config)
    laws = [phase_laws(item, config['batch_size'])[0] for item in phases[:2]]
    for index, law in enumerate(laws):
        data = world.experience(law, config['batch_size'], 91321 + index)
        study.update_marginal(marginals, law, data)
        entry = marginals[law_key(law)]
        for action in range(5):
            mask = data.actions == action
            assert entry['count'][action] == mask.sum()
            np.testing.assert_array_equal(entry['success'][action], data.survival[mask].sum(axis=0))
        expected = (np.asarray(entry['success'], dtype=np.float64) + .5) / (np.asarray(entry['count'])[:, None] + 1)
        np.testing.assert_array_equal(study.marginal_probabilities(entry), expected)
    assert len(marginals) == 2


def test_corrupted_completed_raw_artifact_stops_resume(config, tmp_path):
    directory = tmp_path / 'phase'
    fit(config, directory)
    path = directory / 'training.npz'
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='artifact changed'):
        fit(config, directory)


def test_four_arm_smoke_and_complete_suite_resume_preserve_every_artifact(config, tmp_path, monkeypatch):
    config.update(kind='main', arms=list(ARMS))
    config_path, protocol = sandbox(tmp_path, monkeypatch, config)
    output = tmp_path / 'run'
    identity = study.run_suite(config, output, protocol, config_path)
    before = inventory(output)
    assert len(study.read_json(output / 'completion.json')['jobs']) == 4
    repeated = study.run_suite(config, output, protocol, config_path, resume=True)
    assert repeated == identity and inventory(output) == before
    with pytest.raises(ValueError, match='resume requires'):
        study.run_suite(config, output, protocol, config_path)


def test_lock_verifies_runtime_and_complete_live_source_inventory(config, tmp_path, monkeypatch):
    config_path, protocol = sandbox(tmp_path, monkeypatch, config)
    output = tmp_path / 'run'
    study.run_suite(config, output, protocol, config_path, lock_only=True)
    manifest = study.check_locks(output)
    changed = deepcopy(manifest)
    changed['runtime']['threads'] += 1
    study.write_json(output / 'manifest.json', changed)
    with pytest.raises(ValueError, match='lock mismatch'):
        study.check_locks(output)
    study.write_json(output / 'manifest.json', manifest)
    added = tmp_path / 'src/acp_cl/added.py'
    added.write_text('unexpected = True\n', encoding='utf-8')
    with pytest.raises(ValueError, match='source'):
        study.check_locks(output)


def scientific_main(config):
    result = dict(config, smoke=False, kind='main', seeds=list(range(18201, 18207)), arms=list(ARMS),
        batch_size=32, eval_size=512, support_replicates=2, width=64, context_width=12,
        decoder_width=64, memory_packets=16, updates_per_batch=12, compute_updates_per_batch=14,
        episode_size=8192, prefix_size=1024, interleave_block=1024, interleave_cycles=8,
        probe_every=1024, workers=6)
    validate_config(result)
    return result


def test_missing_or_failed_qualification_stops_main_before_any_fit(config, tmp_path, monkeypatch):
    config = scientific_main(config)
    config_path, protocol = sandbox(tmp_path, monkeypatch, config)
    calls = []
    monkeypatch.setattr(study, 'run_job', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='requires qualified reference'):
        study.run_suite(config, tmp_path / 'missing', protocol, config_path)
    monkeypatch.setattr(study, 'check_locks', lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match='qualification failed'):
        study.run_suite(config, tmp_path / 'failed', protocol, config_path,
                        qualification=tmp_path / 'qualification', engineering=tmp_path / 'engineering.json')
    assert not calls and not (tmp_path / 'failed/manifest.json').exists()


def test_mismatched_engineering_compute_allocation_stops_before_fit(config, tmp_path, monkeypatch):
    config = scientific_main(config)
    config_path, protocol = sandbox(tmp_path, monkeypatch, config, qualified=True)
    monkeypatch.setattr(study, 'check_locks', lambda *args, **kwargs: {})
    engineering = tmp_path / 'engineering.json'
    study.write_json(engineering, {'compute_updates_per_batch': config['compute_updates_per_batch'] + 1})
    with pytest.raises(ValueError, match='allocation differs'):
        study.run_suite(config, tmp_path / 'run', protocol, config_path,
                        qualification=tmp_path / 'qualification', engineering=engineering)
    assert not (tmp_path / 'run/manifest.json').exists()
