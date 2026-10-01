"""Causal branch pairing, exact resumability and locked study configuration."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.predictive_value.mechanism import state_signature
from acp_cl.rehearsal_state import study
from acp_cl.rehearsal_state.design import BRANCHES, CELLS, counts, paired_batch, schedule, validate_config


def configuration():
    return dict(study.read_json('configs/rehearsal_state_smoke.json'), models=['conditional'],
                cpu_affinity=0, workers=1)


def test_fixed_scientific_counts_and_balance():
    c = study.read_json('configs/rehearsal_state_diagnostic.json')
    validate_config(c)
    assert counts(c) == dict(trajectories=24, assessments=48, ordinary_arrivals_per_trajectory=18432,
        diagnostic_forks=3456, instrumented_forks=3024, zero_update_copies=96,
        diagnostic_prediction_forwards=120576, extra_loss_forwards=12096)
    combinations = Counter()
    for seed in c['seeds']:
        phases = schedule(seed, c)
        assert len(phases) == 10
        assert [p['name'] for p in phases if p['kind'] == 'score'] == ['score_0', 'score_1']
        assert all(p['kind'] == 'train' for p in phases if p['name'].startswith('validation_'))
        combinations[phases[0]['cue'], seed % 2] += 1
    assert len(combinations) == 6 and set(combinations.values()) == {2}


@pytest.mark.parametrize('key,value', [('inter_assessment_size', 0), ('inter_assessment_size', 33),
    ('gaps', [32, 64]), ('recovery_size', 32), ('validation_size', 8), ('candidate_count', 15)])
def test_invalid_configuration_fails(key, value):
    c = configuration()
    c[key] = value
    with pytest.raises(ValueError):
        validate_config(c)


@pytest.mark.parametrize('seed', range(15001, 15007))
def test_paired_branches_share_observations_actions_and_initial_history_only(seed):
    c = configuration()
    initial = batch(AcquisitionWorld(), Law(0), 3, 'initial', 0, c['batch_size'])
    streams, rows, arrays = study.build_streams(seed, 0, c, initial)
    assert rows[0]['current']['support_sha256'] == rows[0]['return']['support_sha256'] == initial.fingerprint()
    for index, stream in enumerate(streams):
        expected, truth = paired_batch(seed, 0, index, c)
        for branch, ((data, support), name) in enumerate(zip(stream, BRANCHES)):
            assert data.fingerprint() == expected[branch].fingerprint()
            assert rows[index][name]['query_sha256'] == data.fingerprint()
            assert rows[index][name]['support_sha256'] == support.fingerprint()
            assert support.fingerprint() == (initial if index == 0 else streams[index-1][branch][0]).fingerprint()
            np.testing.assert_array_equal(arrays['truth'][branch, index], truth[branch])
        np.testing.assert_array_equal(stream[0][0].observations, stream[1][0].observations)
        np.testing.assert_array_equal(stream[0][0].actions, stream[1][0].actions)
        ordinary = batch(AcquisitionWorld(), study.previous.law_from_record(schedule(seed, c)[-1]['law']),
                         seed, 'predictive_value_validation_0', index, c['batch_size'])
        assert ordinary.fingerprint() == stream[0][0].fingerprint()
    assert np.any(arrays['outcomes'][0] != arrays['outcomes'][1])


def test_full_probability_metrics_use_performed_and_greedy_actions_separately():
    probabilities = np.full((2, 2, 3, 2, 5, 3), .25, dtype=np.float32)
    probabilities[..., 4, -1] = .9
    arrays = dict(actions=np.asarray([[0, 1], [2, 3]]), outcomes=np.zeros((2, 2, 2, 3)),
                  truth=np.zeros((2, 2, 2, 5, 3)))
    arrays['truth'][..., 4, -1] = 1
    result = study.metric_panels(probabilities, arrays)
    for branch in BRANCHES:
        assert result[branch] == dict(brier=[.0625]*3, survival=[1.]*3,
                                      early_brier=[.0625]*3, late_brier=[.0625]*3)
    probabilities[0, 0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        study.metric_panels(probabilities, arrays)


def test_interrupted_factor_predictions_resume_without_changing_learning_or_choice(tmp_path, monkeypatch):
    c, identity = configuration(), {'engineering':'interruption'}
    control, interrupted = tmp_path/'control', tmp_path/'interrupted'
    control.mkdir()
    interrupted.mkdir()
    study.run_job(c, identity, 15991, 'conditional', control)
    real = study.predict_family
    calls = 0

    def fail(learners, streams):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('deliberate interruption after three durable families')
        return real(learners, streams)

    monkeypatch.setattr(study, 'predict_family', fail)
    with pytest.raises(RuntimeError, match='deliberate interruption'):
        study.run_job(c, identity, 15991, 'conditional', interrupted)
    folder = interrupted/'conditional_15991'
    assert len(list((folder/'diagnostic_0').glob('predictions_*.json'))) == 3
    assert not (folder/'validation_0').exists()
    monkeypatch.setattr(study, 'predict_family', real)
    study.run_job(c, identity, 15991, 'conditional', interrupted)
    for index in (0, 1):
        a = study.read_json(control/f'conditional_15991/diagnostic_{index}/result.json')
        b = study.read_json(folder/f'diagnostic_{index}/result.json')
        assert a['choice'] == b['choice'] and a['panels'] == b['panels'] and a['zero'] == b['zero']
        for cell in (*CELLS, 'zero'):
            first = study.read_arrays(control/f'conditional_15991/diagnostic_{index}/predictions_{cell}.npz')
            second = study.read_arrays(folder/f'diagnostic_{index}/predictions_{cell}.npz')
            np.testing.assert_array_equal(first['probabilities'], second['probabilities'])
    a = study.load(control/'conditional_15991/validation_1/checkpoint.pt', identity)['learner']
    b = study.load(folder/'validation_1/checkpoint.pt', identity)['learner']
    assert state_signature(a, ignore_walltime=True) == state_signature(b, ignore_walltime=True)
    before = {p.relative_to(folder):study.sha(p) for p in folder.rglob('*') if p.is_file()}
    study.run_job(c, identity, 15991, 'conditional', interrupted)
    assert before == {p.relative_to(folder):study.sha(p) for p in folder.rglob('*') if p.is_file()}
    path = folder/'diagnostic_0/predictions_010.npz'
    path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(ValueError, match='artifact changed'):
        study.run_job(c, identity, 15991, 'conditional', interrupted)


def test_no_return_feedback_reaches_ordinary_learning(tmp_path):
    c, identity = configuration(), {'engineering':'ordinary-only'}
    full, baseline = tmp_path/'full', tmp_path/'baseline'
    full.mkdir()
    baseline.mkdir()
    study.run_job(c, identity, 15991, 'conditional', full)
    ordinary = study.new_learner('conditional', 15991, c)
    metadata, anchor = {}, None
    for phase in schedule(15991, c):
        ordinary, metadata, anchor, _ = study.previous.run_phase(c, identity, 15991, 'conditional', phase,
            ordinary, metadata, anchor, baseline)
    actual = study.load(full/'conditional_15991/validation_1/checkpoint.pt', identity)['learner']
    assert state_signature(ordinary, ignore_walltime=True) == state_signature(actual, ignore_walltime=True)


def test_config_change_after_lock_rejected(tmp_path):
    c = configuration()
    protocol = Path('docs/rehearsal_state_protocol.md')
    study.run_suite(c, tmp_path, protocol, lock_only=True)
    changed = dict(c, lr=c['lr']*2)
    with pytest.raises(ValueError, match='configuration changed'):
        study.run_suite(changed, tmp_path, protocol, resume=True, lock_only=True)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
