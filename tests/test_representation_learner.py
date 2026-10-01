"""Shared representation gradients, causal replay, and exact outcome control."""

import inspect
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from acp_cl.acquisition.world import AcquisitionWorld, Law, signal_images
from acp_cl.causal_access.study import rng_signature
from acp_cl.contextual.learner import ContextLearner
from acp_cl.persistence.world import Experience, IMAGE_SHAPE
from acp_cl.predictive_value.mechanism import _value_hash
from acp_cl.predictive_value.study import load, read_json, save
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.representation_learning import learner as module
from acp_cl.representation_learning.learner import (ARMS, RepresentationLearner, learner_signature,
                                                  reconstruction_target)


@pytest.fixture
def settings():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = read_json('configs/core_residual_smoke.json')
    config.update(batch_size=8, width=8, context_width=4, decoder_width=8,
                  updates_per_batch=2, compute_updates_per_batch=3, auxiliary_weight=1., memory_packets=2)
    return config


def examples(config, count=1):
    world = AcquisitionWorld()
    return [world.experience(Law(index % 2, (0, 2)), config['batch_size'], 67100 + index)
            for index in range(count)]


def predictor_state(learner):
    return {name: value for name, value in learner.model.state_dict().items()
            if not name.startswith('auxiliary.')}


def test_all_arms_start_with_identical_reference_predictor_and_unchanged_global_rng(settings):
    before = rng_signature()
    reference = ContextLearner('recurrent', 74, settings)
    learners = [RepresentationLearner(arm, 74, settings) for arm in ARMS]
    assert rng_signature() == before
    for learner in learners:
        assert learner.method == 'recurrent' and isinstance(learner, ContextLearner)
        assert _value_hash(predictor_state(learner)) == _value_hash(reference.model.state_dict())
        assert _value_hash(learner.initial_encoder) == _value_hash(reference.initial_encoder)
        assert learner.initial_predictor_hash == reference.initial_hash
        assert memory_state(learner.memory) == memory_state(reference.memory)
    assert _value_hash(learners[1].model.auxiliary.state_dict()) == _value_hash(learners[2].model.auxiliary.state_dict())
    assert not hasattr(learners[0].model, 'auxiliary') and not hasattr(learners[3].model, 'auxiliary')


@pytest.mark.parametrize('interaction_features', (False, True))
def test_outcome_control_exactly_matches_original_multi_packet_training(settings, interaction_features):
    settings['interaction_features'] = interaction_features
    reference = ContextLearner('recurrent', 82, settings)
    actual = RepresentationLearner('outcome', 82, settings)
    for data in examples(settings, 7):
        before = rng_signature()
        reference.train(data)
        actual.train(data)
        assert rng_signature() == before
        assert _value_hash(actual.model.state_dict()) == _value_hash(reference.model.state_dict())
        assert _value_hash(actual.optimizer.state_dict()) == _value_hash(reference.optimizer.state_dict())
        assert _value_hash({name: value.grad for name, value in actual.model.named_parameters()}) == _value_hash(
            {name: value.grad for name, value in reference.model.named_parameters()})
        assert memory_state(actual.memory) == memory_state(reference.memory)
        assert actual.history.fingerprint() == reference.history.fingerprint()
        for name in reference.cost.keys() - {'training_seconds'}:
            assert actual.cost[name] == reference.cost[name]
        for support in (None, data):
            for expected, observed in zip(reference.predict(data.observations, support),
                                          actual.predict(data.observations, support), strict=True):
                np.testing.assert_array_equal(observed, expected)


def test_training_reuses_exact_query_latent_and_never_reconstructs_support(settings, monkeypatch):
    learner = RepresentationLearner('sequence', 85, settings)
    previous, current = examples(settings, 2)
    learner.train(previous)
    calls, latent_objects, targets = [], [], []
    original_forward, original_target = learner.forward_with_features, module.reconstruction_target

    def forward(observations, support=None, oracle_modes=None, override=None):
        result = original_forward(observations, support, oracle_modes, override)
        latent_objects.append(result[-1])
        calls.append((observations.copy(), None if support is None else support.fingerprint()))
        return result

    def target(observations, arm):
        targets.append(observations.detach().cpu().numpy().copy())
        return original_target(observations, arm)

    encoder_calls, auxiliary_calls = [], []
    original_encode = learner.model.encode

    def encode(observations):
        encoder_calls.append(len(observations))
        return original_encode(observations)

    def check_shared_latent(_, arguments):
        assert arguments[0] is latent_objects[-1]
        assert len(arguments[0]) == settings['batch_size']
        auxiliary_calls.append(True)

    monkeypatch.setattr(learner, 'forward_with_features', forward)
    monkeypatch.setattr(learner.model, 'encode', encode)
    monkeypatch.setattr(module, 'reconstruction_target', target)
    handle = learner.model.auxiliary.register_forward_pre_hook(check_shared_latent)
    learner.train(current)
    handle.remove()
    assert encoder_calls == [2 * settings['batch_size']] * (2 * settings['updates_per_batch'])
    assert len(auxiliary_calls) == len(calls) == len(targets) == 2 * settings['updates_per_batch']
    for index, ((query, support), target_query) in enumerate(zip(calls, targets, strict=True)):
        expected = current if index % 2 == 0 else previous
        np.testing.assert_array_equal(query, expected.observations)
        np.testing.assert_array_equal(target_query, expected.observations)
        assert support == (previous.fingerprint() if index % 2 == 0 else None)


def test_auxiliary_loss_reaches_visual_encoder_without_context_or_outcome_head(settings):
    learner = RepresentationLearner('sequence', 88, settings)
    data = examples(settings)[0]
    probs, _, _, features = learner.forward_with_features(data.observations)
    assert features.requires_grad and probs.requires_grad
    reconstruction = learner.model.auxiliary(features).sigmoid().reshape(-1, *IMAGE_SHAPE)
    loss = F.mse_loss(reconstruction, reconstruction_target(data.observations, 'sequence'))
    loss.backward()
    for part in (learner.model.frame, learner.model.temporal, learner.model.auxiliary):
        assert sum(float(parameter.grad.abs().sum()) for parameter in part.parameters()) > 0
    assert all(parameter.grad is None for parameter in learner.model.context.parameters())
    assert all(parameter.grad is None for parameter in learner.model.decoder.parameters())


def test_temporal_target_distinguishes_pulse_order_while_final_frame_control_does_not():
    blank = np.zeros((2, *IMAGE_SHAPE), dtype=np.uint8)
    observations = signal_images(blank, np.array([[0, 0, 0], [1, 0, 0]], dtype=np.uint8))
    sequence = reconstruction_target(observations, 'sequence')
    spatial = reconstruction_target(observations, 'final_frame')
    assert not torch.equal(sequence[0], sequence[1])
    assert torch.equal(spatial[0], spatial[1])
    assert sequence.shape == spatial.shape == (2, *IMAGE_SHAPE)
    assert torch.equal(spatial, torch.as_tensor(observations[:, -1:]).expand_as(spatial).float() / 255)
    expected_mse = np.square(observations.astype(np.float64) / 255).mean()
    assert float(F.mse_loss(torch.zeros_like(sequence), sequence)) == pytest.approx(expected_mse)
    np.testing.assert_array_equal(observations[:, -1], blank[:, -1])


def test_prediction_and_reconstruction_preserve_state_and_aux_decoder_is_unused_for_prediction(settings):
    learner = RepresentationLearner('sequence', 90, settings)
    data = examples(settings)[0]
    learner.train(data)
    for index, part in enumerate(learner.model.modules()):
        part.training = bool(index % 2)
    before, rng = learner_signature(learner), rng_signature()
    calls = []
    handle = learner.model.auxiliary.register_forward_hook(lambda *args: calls.append(True))
    prediction = learner.predict(data.observations, data)
    assert not calls
    reconstruction = learner.reconstruction(data.observations)
    handle.remove()
    assert calls == [True]
    assert learner_signature(learner) == before and rng_signature() == rng
    assert reconstruction.dtype == np.float32 and reconstruction.shape == data.observations.shape
    assert np.isfinite(reconstruction).all() and (reconstruction >= 0).all() and (reconstruction <= 1).all()
    assert prediction[0].shape == (len(data), 5, 3)
    assert prediction[1].shape == (len(data), 1) and prediction[2].shape == (len(data), 1, 5, 3)


def test_forward_uses_only_images_and_past_support_and_rejects_privileged_modes(settings):
    learner = RepresentationLearner('sequence', 93, settings)
    previous, current = examples(settings, 2)
    learner.train(previous)
    prediction = learner.predict(current.observations, previous)[0]
    alternate = Experience(current.observations.copy(), (current.actions + 1) % 5,
                           1 - current.survival[:, ::-1].copy())
    np.testing.assert_array_equal(prediction, learner.predict(alternate.observations, previous)[0])
    assert 'outcomes' not in inspect.signature(learner.forward_with_features).parameters
    before = learner_signature(learner)
    with pytest.raises(ValueError, match='latent modes'):
        learner.train(current, np.zeros(len(current)))
    with pytest.raises(ValueError, match='latent modes'):
        learner.predict(current.observations, previous, np.zeros(len(current)))
    assert learner_signature(learner) == before


def test_replay_membership_matches_across_arms_and_original_contexts_are_retained(settings):
    learners = [RepresentationLearner(arm, 95, settings) for arm in ARMS]
    previous, originals = None, {}
    for data in examples(settings, 8):
        originals[data.fingerprint()] = None if previous is None else previous.fingerprint()
        for learner in learners:
            learner.train(data)
            for packet in learner.memory.packets:
                assert packet.oracle_modes is None
                assert (None if packet.support is None else packet.support.fingerprint()) == originals[packet.query.fingerprint()]
            assert learner.history.fingerprint() == data.fingerprint()
        baseline = memory_state(learners[0].memory)
        for learner in learners[1:]:
            state = memory_state(learner.memory)
            assert state['ids'] == baseline['ids'] and state['membership'] == baseline['membership']
            assert state['packets'] == baseline['packets']
        previous = data
    assert learners[-1].cost['optimizer_steps'] == 8 * settings['compute_updates_per_batch']
    assert all(item.cost['optimizer_steps'] == 8 * settings['updates_per_batch'] for item in learners[:-1])


@pytest.mark.parametrize('arm', ARMS)
def test_loss_work_and_memory_accounting_match_executed_updates(settings, arm):
    learner = RepresentationLearner(arm, 97, settings)
    for data in examples(settings, 3):
        learner.train(data)
    report, cost = learner.diagnostics(), learner.cost
    updates = 3 * learner.updates_per_batch
    query_presentations = 2 * updates * settings['batch_size']
    auxiliary = arm in ('final_frame', 'sequence')
    assert cost['optimizer_steps'] == updates and cost['query_presentations'] == query_presentations
    assert cost['encoder_forward_calls'] == 2 * updates
    assert cost['encoder_presentations'] == 2 * query_presentations
    assert cost['auxiliary_forward_calls'] == (2 * updates if auxiliary else 0)
    assert cost['auxiliary_reconstructions'] == (query_presentations if auxiliary else 0)
    assert cost['reconstruction_pixels'] == (query_presentations * math.prod(IMAGE_SHAPE) if auxiliary else 0)
    assert cost['auxiliary_backward_steps'] == (updates if auxiliary else 0)
    assert cost['predictive_parameter_updates'] == updates * report['predictive_parameters']
    assert cost['auxiliary_parameter_updates'] == updates * report['auxiliary_parameters']
    assert report['parameters'] == sum(value.numel() for value in learner.model.parameters())
    assert report['replay_capacity'] == settings['memory_packets'] and report['packets_seen'] == 3
    assert report['optimizer_bytes'] > 0 and report['peak_replay_bytes'] >= learner.memory.nbytes()
    assert report['mean_objective_loss'] == pytest.approx(report['mean_outcome_loss'] + (
        settings['auxiliary_weight'] * report['mean_reconstruction_loss'] if auxiliary else 0), abs=2e-7)
    if not auxiliary:
        with pytest.raises(ValueError, match='no reconstruction decoder'):
            learner.reconstruction(data.observations)


@pytest.mark.parametrize('arm', ('outcome', 'sequence'))
def test_checkpoint_restores_gradients_rng_replay_and_exact_continuation(settings, tmp_path, arm):
    learner = RepresentationLearner(arm, 101, settings)
    first, second = examples(settings, 2)
    learner.train(first)
    before, rng = learner_signature(learner), rng_signature()
    path, identity = tmp_path / 'checkpoint.pt', {'test_identity': 1}
    save(path, dict(identity=identity, learner=learner, record={}))
    restored = load(path, identity)['learner']
    assert learner_signature(restored) == before and rng_signature() == rng
    learner.train(second)
    restored.train(second)
    assert learner_signature(restored, ignore_walltime=True) == learner_signature(learner, ignore_walltime=True)


def test_invalid_arm_and_auxiliary_settings_are_rejected(settings):
    with pytest.raises(ValueError, match='Unknown'):
        RepresentationLearner('oracle', 1, settings)
    for name, value in (('compute_updates_per_batch', 0), ('compute_updates_per_batch', True),
                        ('auxiliary_weight', 0.), ('auxiliary_weight', float('nan'))):
        invalid = dict(settings, **{name: value})
        with pytest.raises(ValueError):
            RepresentationLearner('sequence', 1, invalid)
