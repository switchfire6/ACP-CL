"""Independent matrix-work counts and fixed engineering allocation safeguards."""

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from acp_cl.acquisition.world import AcquisitionWorld, Law


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'representation_compute_calibration', ROOT / 'scripts/calibrate_representation_compute.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture
def config():
    return json.loads((ROOT / 'configs/representation_learning_qualification.json').read_text())


def test_dense_proxy_counts_both_packets_both_encoders_and_query_only_decoders(config):
    proxy = module.dense_mac_proxy(config)
    assert proxy['packet_forward_components'] == dict(
        frame_linear=3145728, temporal_gru=6291456, context_gru=465408,
        outcome_decoder=186368)
    assert proxy['auxiliary_query_forward'] == 1703936
    assert proxy['outcome_training_per_update'] == 60533760
    assert proxy['auxiliary_training_per_update'] == 70757376
    assert proxy['auxiliary_base_ratio'] == 70757376 / 60533760
    config['interaction_features'] = False
    assert module.dense_mac_proxy(config)['packet_forward_components']['context_gru'] == 96768


def test_allocation_uses_larger_auxiliary_median_time_and_rounds_up(config):
    proxy = module.dense_mac_proxy(config)
    allocation = module.allocate_updates(12, dict(outcome=10, final_frame=10, sequence=10),
                                         proxy['auxiliary_base_ratio'])
    assert allocation['compute_updates_per_batch'] == 15
    assert module.allocate_updates(12, dict(outcome=10, final_frame=14, sequence=11), 1.1)[
        'compute_updates_per_batch'] == 17
    assert module.allocate_updates(12, dict(outcome=10, final_frame=9, sequence=8), .9)[
        'compute_updates_per_batch'] == 12


@pytest.mark.parametrize('seconds,ratio', [
    (dict(outcome=0, final_frame=1, sequence=1), 1),
    (dict(outcome=1, final_frame=math.nan, sequence=1), 1),
    (dict(outcome=1, sequence=1), 1),
    (dict(outcome=1, final_frame=1, sequence=1), math.inf),
])
def test_allocation_rejects_missing_or_nonfinite_measurements(seconds, ratio):
    with pytest.raises(ValueError):
        module.allocate_updates(12, seconds, ratio)


def test_fixed_engineering_schedule_reverses_and_fills_the_reservoir(config):
    assert module.ENGINEERING_SEED == 18091
    assert module.WARMUP_BATCHES >= config['memory_packets']
    assert module.MEASURED_BATCHES == 8 and module.ROUNDS == 5
    assert [module.round_order(i) for i in range(5)] == [
        ('outcome', 'final_frame', 'sequence'), ('sequence', 'final_frame', 'outcome'),
        ('outcome', 'final_frame', 'sequence'), ('sequence', 'final_frame', 'outcome'),
        ('outcome', 'final_frame', 'sequence')]


def test_small_actual_round_has_paired_original_contexts_and_exports_work_only(config, monkeypatch):
    torch.set_num_threads(1)
    config.update(batch_size=8, width=8, context_width=4, decoder_width=8,
                  memory_packets=1, updates_per_batch=1, compute_updates_per_batch=1)
    monkeypatch.setattr(module, 'WARMUP_BATCHES', 1)
    monkeypatch.setattr(module, 'MEASURED_BATCHES', 1)
    world = AcquisitionWorld()
    packets = [world.experience(Law(i % 2), 8, 92000 + i) for i in range(2)]
    fingerprints = [packet.fingerprint() for packet in packets]
    result = module.measure_round(config, packets, 1)
    assert result['arm_order'] == ['sequence', 'final_frame', 'outcome']
    assert [packet.fingerprint() for packet in packets] == fingerprints
    assert len({a['warmup_replay_sha256'] for a in result['arms'].values()}) == 1
    assert len({a['final_replay_sha256'] for a in result['arms'].values()}) == 1
    for arm, row in result['arms'].items():
        assert row['training_seconds'] > 0
        assert row['measured_work']['optimizer_steps'] == 1
        assert row['measured_work']['encoder_presentations'] == 32
        assert row['measured_work']['auxiliary_reconstructions'] == (0 if arm == 'outcome' else 16)
    encoded = json.dumps(result)
    assert all(key not in encoded for key in ('loss', 'brier', 'accuracy', 'probabilities'))


@pytest.mark.parametrize('existing', ('allocation.json', 'allocation.attempt.json'))
def test_existing_attempt_or_result_cannot_be_overwritten(tmp_path, existing):
    path = tmp_path / existing
    path.write_text('preserved')
    with pytest.raises(FileExistsError, match='preserved'):
        module.run(tmp_path / 'missing-config.json', tmp_path / 'allocation.json')
    assert path.read_text() == 'preserved'
