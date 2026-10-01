"""Cheap integration checks for the frozen stream's information boundary."""

from copy import deepcopy
import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from acp_cl.acquisition.world import AcquisitionWorld, Law, stage_law
from acp_cl.causal_access import study
from acp_cl.causal_access.design import counts, validate_config
from acp_cl.persistence.world import Experience
from acp_cl.predictive_value.study import read_json


class FakeForecaster:
    """Forecasts depend on images and supplied history, with no task argument."""

    def __init__(self, history, events=None):
        self.learners = {name: SimpleNamespace(history=deepcopy(history)) for name in
                         ('separate', 'joint', 'reference_old', 'reference_novel')}
        self.calls = []
        self.events = events

    def predict(self, name, observations, support):
        assert isinstance(observations, np.ndarray) and isinstance(support, Experience)
        if self.events is not None:
            self.events.append('predict')
        self.calls.append((name, observations.copy(), support.fingerprint()))
        signal = observations[:, 0, 8][:, [1, 6, 11]].astype(np.float64).mean(axis=1) / 255
        previous = .02 * support.survival[:, -1].mean()
        core = np.broadcast_to((.15 + .1 * signal + previous)[:, None, None],
                               (len(observations), 5, 3)).astype(np.float32).copy()
        full = np.broadcast_to((.8 - .1 * signal + previous)[:, None, None],
                               core.shape).astype(np.float32).copy()
        return dict(core=core, full=full)


@pytest.fixture
def scenario():
    config = read_json('configs/causal_access_smoke.json')
    config.update(stream_block_size=16)
    seed = config['seeds'][0]
    history = AcquisitionWorld().experience(stage_law(seed, 1), config['batch_size'], 47381)
    return config, seed, history


def test_all_packet_predictions_precede_truth_release_and_feedback(monkeypatch, scenario):
    config, seed, history = scenario
    events = []
    original_mixer = study.LaggedMixer

    class OrderedWorld(AcquisitionWorld):
        def counterfactuals(self, cases):
            events.append('truth')
            return super().counterfactuals(cases)

    class OrderedMixer(original_mixer):
        def observe(self, core, full, actions, outcomes):
            events.append('feedback')
            return super().observe(core, full, actions, outcomes)

    monkeypatch.setattr(study, 'AcquisitionWorld', OrderedWorld)
    monkeypatch.setattr(study, 'LaggedMixer', OrderedMixer)
    result = study.frozen_stream(config, seed, FakeForecaster(history, events), {})
    expected = []
    for index in range(result['packet_count']):
        novel = (index * config['batch_size'] // config['stream_block_size']) % 2
        expected += ['predict'] * (4 if novel else 2) + ['truth', 'feedback']
    assert events == expected


def test_only_previous_packet_is_history_and_learners_and_rng_are_unchanged(scenario):
    config, seed, history = scenario
    forecaster, arrays = FakeForecaster(history), {}
    saved, rng = pickle.dumps(forecaster.learners), study.rng_signature()
    result = study.frozen_stream(config, seed, forecaster, arrays)
    assert pickle.dumps(forecaster.learners) == saved and study.rng_signature() == rng
    assert arrays['stream_weights'][0] == .5
    assert result['initial_support'] == history.fingerprint()
    expected = [history.fingerprint()]
    batch = config['batch_size']
    for start in range(0, len(arrays['stream_actions']) - batch, batch):
        interval = slice(start, start + batch)
        expected.append(Experience(arrays['stream_observations'][interval],
                                   arrays['stream_actions'][interval],
                                   arrays['stream_outcomes'][interval]).fingerprint())
    assert result['support_fingerprints'] == expected
    call = 0
    for index, fingerprint in enumerate(expected):
        novel = (index * batch // config['stream_block_size']) % 2
        count = 4 if novel else 2
        assert all(record[2] == fingerprint for record in forecaster.calls[call:call + count])
        call += count
    assert call == len(forecaster.calls)


def test_exogenous_actions_and_feedback_exactly_match_ordinary_world_experience(scenario):
    config, seed, history = scenario
    arrays = {}
    result = study.frozen_stream(config, seed, FakeForecaster(history), arrays)
    world, batch = AcquisitionWorld(), config['batch_size']
    for index in range(result['packet_count']):
        block = index * batch // config['stream_block_size']
        law = stage_law(seed, 1) if block % 2 else Law(seed % 2)
        expected = world.experience(law, batch, study.trial_seed(seed, 'causal_access_stream', index))
        interval = slice(index * batch, (index + 1) * batch)
        np.testing.assert_array_equal(arrays['stream_observations'][interval], expected.observations)
        np.testing.assert_array_equal(arrays['stream_actions'][interval], expected.actions)
        np.testing.assert_array_equal(arrays['stream_outcomes'][interval], expected.survival)


def test_weights_and_probe_mixtures_reconstruct_from_original_past_terminal_forecasts(scenario):
    config, seed, history = scenario
    arrays = {}
    result = study.frozen_stream(config, seed, FakeForecaster(history), arrays)
    scores, batch = [], config['batch_size']
    for index in range(result['packet_count']):
        log_odds = math.fsum(scores[-config['mixer_capacity']:])
        weight = 1 / (1 + math.exp(-log_odds))
        assert arrays['stream_weights'][index] == pytest.approx(weight, abs=1e-15)
        interval = slice(index * batch, (index + 1) * batch)
        for suffix in ('', '_flipped'):
            core = arrays['stream_core' + suffix][interval].astype(np.float64)
            full = arrays['stream_full' + suffix][interval].astype(np.float64)
            expected = ((1 - weight) * core + weight * full).astype(np.float32)
            np.testing.assert_allclose(arrays['stream_causal' + suffix][interval], expected,
                                       rtol=0, atol=np.finfo(np.float32).eps)
        for row in range(index * batch, (index + 1) * batch):
            action = arrays['stream_actions'][row]
            terminal = arrays['stream_outcomes'][row, -1]
            likelihoods = [float(arrays['stream_' + name][row, action, -1])
                           for name in ('core', 'full')]
            if not terminal:
                likelihoods = [1 - value for value in likelihoods]
            scores.append(math.log(likelihoods[1]) - math.log(likelihoods[0]))
    np.testing.assert_allclose(result['mixer_final']['scores'], scores[-32:], atol=1e-15, rtol=0)
    assert result['mixer_final']['samples_seen'] == len(arrays['stream_actions'])
    assert result['mixer_final']['packets_seen'] == result['packet_count']


def test_cue_probes_use_only_novel_blocks_and_leave_old_predictions_identical(scenario):
    config, seed, history = scenario
    forecaster, arrays = FakeForecaster(history), {}
    result = study.frozen_stream(config, seed, forecaster, arrays)
    assert len(forecaster.calls) == 3 * result['packet_count']
    old = ~arrays['stream_is_novel']
    for name in ('core', 'full', 'half', 'causal', 'joint', 'observations'):
        flipped = 'stream_flipped_observations' if name == 'observations' else 'stream_' + name + '_flipped'
        np.testing.assert_array_equal(arrays['stream_' + name][old], arrays[flipped][old])
    assert np.any(arrays['stream_observations'][~old] != arrays['stream_flipped_observations'][~old])


def test_qualification_does_not_supply_extra_stream_history_or_feedback(scenario):
    config, seed, history = scenario
    forecaster, arrays = FakeForecaster(history), {}
    saved = pickle.dumps(forecaster.learners)
    study.qualification(config, seed, forecaster, arrays)
    assert pickle.dumps(forecaster.learners) == saved
    result = study.frozen_stream(config, seed, forecaster, arrays)
    assert arrays['stream_weights'][0] == .5 and result['initial_support'] == history.fingerprint()
    assert result['mixer_final']['samples_seen'] == len(arrays['stream_actions'])
    assert arrays['q_novel_full_flipped'].shape == arrays['q_novel_full'].shape


def test_mismatched_initial_training_histories_are_rejected_before_prediction(scenario):
    config, seed, history = scenario
    forecaster = FakeForecaster(history)
    forecaster.learners['joint'].history.actions[0] = (history.actions[0] + 1) % 5
    with pytest.raises(AssertionError, match='same actual novel training history'):
        study.frozen_stream(config, seed, forecaster, {})
    assert not forecaster.calls


def test_forecaster_counts_actual_two_path_calls_without_changing_support(monkeypatch, scenario):
    _, _, history = scenario
    learner, called = object(), []
    original = history.fingerprint()

    def component(model, observations, support):
        assert model is learner and support is history
        called.append(len(observations))
        return {'core': np.zeros((len(observations), 5, 3), dtype=np.float32)}

    monkeypatch.setattr(study, 'component_predictions', component)
    forecaster = study.Forecaster({'separate': learner})
    forecaster.predict('separate', history.observations[:3], history)
    forecaster.predict('separate', history.observations[:2], history)
    assert called == [3, 2] and history.fingerprint() == original
    assert forecaster.work['separate'] == dict(component_calls=2, query_presentations=5,
        support_presentations=2 * len(history), path_presentations=2 * (5 + 2 * len(history)))


def test_locked_configuration_and_declared_work_counts():
    development = read_json('configs/causal_access_development.json')
    smoke = read_json('configs/causal_access_smoke.json')
    validate_config(development)
    validate_config(smoke)
    assert counts(development) == dict(jobs=12, total_phases=120, training_arrivals=466944,
        training_packets=14592, optimizer_steps=175104, stream_arrivals=196608)
    assert counts(smoke) == dict(jobs=2, total_phases=16, training_arrivals=1408,
        training_packets=176, optimizer_steps=176, stream_arrivals=256)
    for key, value in (('mixer_capacity', 64), ('mixer_clip', .001), ('feedback_noise', .1),
                       ('lr', .003), ('stream_blocks', 15), ('models', ['conditional'])):
        changed = deepcopy(development)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_config(changed)


def test_lock_verification_detects_changed_source_analysis_archive_and_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    live_sources = {'mechanism.py': 'original-source-hash'}
    monkeypatch.setattr(study, 'source_manifest', lambda: live_sources.copy())
    output = tmp_path / 'run'
    output.mkdir()
    protocol, analysis = tmp_path / 'protocol.md', tmp_path / 'analysis.py'
    protocol.write_text('fixed protocol', encoding='utf-8')
    analysis.write_text('fixed analysis', encoding='utf-8')
    (output / 'protocol_at_lock.md').write_bytes(protocol.read_bytes())
    archive = output / 'training_source.zip'
    archive.write_bytes(b'fixed archive bytes')
    config, runtime = {'setting': 1}, {'python': 'fixed'}
    analysis_files = {'analysis.py': study.sha(analysis)}
    identity = dict(config_sha256=study.digest(config), source_sha256=study.digest(live_sources),
                    analysis_sha256=study.digest(analysis_files), runtime_sha256=study.digest(runtime))
    manifest = dict(config=config, runtime=runtime, identity=identity, source_files=live_sources.copy(),
                    analysis_files=analysis_files, archives={'training_source.zip': study.sha(archive)},
                    protocol_sha256=study.sha(protocol))
    study.write_json(output / 'manifest.json', manifest)
    assert study.check_locks(output, config, protocol) == manifest
    live_sources['mechanism.py'] = 'changed-source-hash'
    with pytest.raises(ValueError, match='source changed'):
        study.check_locks(output)
    live_sources['mechanism.py'] = 'original-source-hash'
    for path in (analysis, archive, output / 'protocol_at_lock.md'):
        original = path.read_bytes()
        path.write_bytes(b'changed')
        with pytest.raises(ValueError, match='changed'):
            study.check_locks(output)
        path.write_bytes(original)
    assert study.check_locks(output, config, protocol) == manifest
