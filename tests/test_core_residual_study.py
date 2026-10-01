"""Continuous causal trajectories, fixed resources and crash-safe exact resumption."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from acp_cl.acquisition.world import AcquisitionWorld, batch, order
from acp_cl.core_residual import study
from acp_cl.core_residual.design import ARMS, counts, law_from_record, phases, prefix_phases, validate_config
from acp_cl.core_residual.learner import learner_signature


def configuration():
    return dict(study.read_json('configs/core_residual_smoke.json'), models=['conditional'],
                workers=1, cpu_affinity=0)


def test_prospective_cohort_counts_and_cue_balance():
    config = study.read_json('configs/core_residual_development.json')
    validate_config(config)
    assert counts(config) == dict(jobs=12, prefix_fits=12, prefix_phases=48,
        learning_trajectories=36, fresh_references=12, post_prefix_episodes=120,
        total_phases=168, training_arrivals=552960, training_packets=17280,
        optimizer_steps=207360)
    balance = Counter((order(seed)[0], seed % 2) for seed in config['seeds'])
    assert len(balance) == 6 and set(balance.values()) == {1}
    for seed in config['seeds']:
        prefix = prefix_phases(seed, config)
        schedule = phases(seed, config, 'joint')
        assert prefix[-1]['law'] == schedule[0]['law']
        assert prefix[0]['law'] == schedule[-1]['law']
        assert schedule[0]['law']['mode'] == schedule[1]['law']['mode']
        assert len(schedule[1]['law']['active']) == 1
        assert phases(seed, config, 'fresh') == [schedule[1]]
        assert all(phases(seed, config, arm) == schedule for arm in ARMS)


@pytest.mark.parametrize('key,value', [('maintenance_size',0), ('return_size',33),
    ('residual_width',65), ('residual_context_width',False), ('cpu_affinity',-1),
    ('feedback_noise',.1), ('prefix_blocks',3), ('undeclared_switch',True)])
def test_invalid_or_undeclared_interventions_fail(key, value):
    config = configuration()
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize('model', ['conditional', 'recurrent'])
def test_continuous_state_stream_pairing_and_empty_fresh_reference(tmp_path, model):
    config, seed, identity = configuration(), 16991, {'engineering':'continuous'}
    study.run_job(config, identity, seed, model, tmp_path)
    folder = tmp_path/f'{model}_{seed}'
    records = {arm:{phase['name']:study.read_json(folder/arm/phase['name']/'result.json')
        for phase in phases(seed, config, arm)} for arm in (*ARMS,'fresh')}
    for arm in ARMS:
        for previous, following in (('maintenance','novel'), ('novel','return')):
            assert records[arm][previous]['end_signature'] == records[arm][following]['start_signature']
            assert records[arm][previous]['end_memory'] == records[arm][following]['start_memory']
            assert records[arm][following]['packets'][0]['support_sha256'] == records[arm][previous]['batch_sha256'][-1]
        for name in ('maintenance','novel','return'):
            reference = records['joint'][name]
            actual = records[arm][name]
            assert actual['batch_sha256'] == reference['batch_sha256']
            assert actual['start_memory'] == reference['start_memory']
            assert actual['end_memory'] == reference['end_memory']
            for left,right in zip(actual['packets'],reference['packets']):
                assert {k:v for k,v in left.items() if k != 'prequential_brier'} == {
                    k:v for k,v in right.items() if k != 'prequential_brier'}
    fresh = study.load(folder/'fresh/novel/before.pt', identity)['learner']
    assert fresh.memory.seen == 0 and not fresh.memory.packets
    assert not fresh.optimizer.state
    assert fresh.cost['optimizer_steps'] == 0
    assert fresh.history.fingerprint() == records['joint']['maintenance']['batch_sha256'][-1]
    assert records['fresh']['novel']['batch_sha256'] == records['joint']['novel']['batch_sha256']
    for arm in (*ARMS,'fresh'):
        for record in records[arm].values():
            assert record['batches_done'] == record['size']//config['batch_size']
            assert record['curve'][0]['arrivals'] == 0 and record['curve'][-1]['arrivals'] == record['size']


@pytest.mark.parametrize('model', ['conditional','recurrent'])
@pytest.mark.parametrize('arm', ['prefix','separate'])
def test_interrupted_phase_resumes_identical_predictions_replay_and_state(tmp_path, monkeypatch, model, arm):
    config, seed, identity = configuration(), 16991, {'engineering':'crash-resume'}
    phase = prefix_phases(seed, config)[0] if arm == 'prefix' else phases(seed,config,arm)[1]

    def initial():
        base = study.new_learner(model,seed,config)
        return base if arm == 'prefix' else study.fork_core(base,arm,seed,config)

    control, expected = study.episode(config, identity, initial(),
        seed, model, arm, phase, tmp_path/'control')
    real = study.predict_all
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == config['probe_every']//config['batch_size']+2:
            raise RuntimeError('deliberate interrupted packet')
        return real(*args, **kwargs)

    monkeypatch.setattr(study, 'predict_all', interrupt)
    with pytest.raises(RuntimeError, match='deliberate interrupted'):
        study.episode(config, identity, initial(),
            seed, model, arm, phase, tmp_path/'resumed')
    checkpoint = study.load(tmp_path/'resumed/checkpoint.pt', identity)
    assert checkpoint['record']['batches_done'] == config['probe_every']//config['batch_size']
    monkeypatch.setattr(study, 'predict_all', real)
    actual, record = study.episode(config, identity, initial(),
        seed, model, arm, phase, tmp_path/'resumed')
    assert learner_signature(actual, ignore_walltime=True) == learner_signature(control, ignore_walltime=True)
    for key in ('curve','packets','batch_sha256','before','after','valid_after','start_memory','end_memory'):
        assert record[key] == expected[key]
    for name in ('training.npz','evaluations.npz'):
        first, second = (study.read_arrays(tmp_path/folder/name) for folder in ('control','resumed'))
        assert first.keys() == second.keys()
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


def test_completed_checkpoint_recovers_missing_result_and_rejects_raw_tampering(tmp_path, monkeypatch):
    config, seed, identity = configuration(), 16991, {'engineering':'durable-checkpoint'}
    phase = prefix_phases(seed, config)[0]
    real = study.write_json

    def fail_result(path, value):
        if Path(path).name == 'result.json':
            raise RuntimeError('deliberate final JSON interruption')
        return real(path, value)

    monkeypatch.setattr(study, 'write_json', fail_result)
    with pytest.raises(RuntimeError, match='deliberate final'):
        study.episode(config, identity, study.new_learner('conditional', seed, config),
            seed, 'conditional', 'prefix', phase, tmp_path)
    assert not (tmp_path/'result.json').exists()
    checkpoint = study.load(tmp_path/'checkpoint.pt', identity)
    assert 'end_signature' in checkpoint['record']
    monkeypatch.setattr(study, 'write_json', real)
    learner, record = study.episode(config, identity, study.new_learner('conditional', seed, config),
        seed, 'conditional', 'prefix', phase, tmp_path)
    assert record == checkpoint['record']
    assert learner_signature(learner) == record['end_signature']
    saved = {p.name:study.sha(p) for p in tmp_path.iterdir() if p.is_file()}
    study.episode(config, identity, study.new_learner('conditional', seed, config),
        seed, 'conditional', 'prefix', phase, tmp_path)
    assert saved == {p.name:study.sha(p) for p in tmp_path.iterdir() if p.is_file()}
    path = tmp_path/'training.npz'
    path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(ValueError, match='artifact changed'):
        study.episode(config, identity, study.new_learner('conditional', seed, config),
            seed, 'conditional', 'prefix', phase, tmp_path)


def test_completed_job_is_read_only_and_checks_raw_artifacts(tmp_path):
    config, seed, identity = configuration(), 16991, {'engineering':'whole-job'}
    study.run_job(config, identity, seed, 'conditional', tmp_path)
    saved = {p.relative_to(tmp_path):study.sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    study.run_job(config, identity, seed, 'conditional', tmp_path)
    assert saved == {p.relative_to(tmp_path):study.sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    path = tmp_path/'conditional_16991/separate/novel/evaluations.npz'
    path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(ValueError, match='raw artifact changed'):
        study.run_job(config, identity, seed, 'conditional', tmp_path)


def test_prequential_prediction_is_before_first_training_update(tmp_path):
    config, seed, identity = configuration(), 16991, {'engineering':'causal'}
    model, phase = 'conditional', prefix_phases(seed, config)[0]
    learner = study.new_learner(model, seed, config)
    data = batch(AcquisitionWorld(), law_from_record(phase['law']), seed, phase['channel'], 0, config['batch_size'])
    before = study.predict_all(learner, data.observations, None)
    study.episode(config, identity, learner, seed, model, 'prefix', phase, tmp_path)
    raw = study.read_arrays(tmp_path/'training.npz')
    np.testing.assert_array_equal(raw['probabilities'][0], before)
    np.testing.assert_array_equal(raw['outcomes'][0], data.survival)


def test_changed_config_and_source_are_rejected_after_lock(tmp_path, monkeypatch):
    config = configuration()
    protocol = Path('docs/core_residual_protocol.md')
    study.run_suite(config, tmp_path, protocol, lock_only=True)
    with pytest.raises(ValueError, match='configuration changed'):
        study.run_suite(dict(config, lr=config['lr']*2), tmp_path, protocol, resume=True, lock_only=True)
    original = study.source_manifest()
    monkeypatch.setattr(study, 'source_manifest', lambda:dict(original, **{'changed.py':'0'*64}))
    with pytest.raises(ValueError, match='source or device changed'):
        study.run_suite(config, tmp_path, protocol, resume=True, lock_only=True)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
