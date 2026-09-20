import copy

import pytest
import torch

from acp_cl.monitor import OnlineMonitor


def observe_window(monitor, generator, shift=0.0):
    for _ in range(25):
        features = torch.randn(96, 32, generator=generator) + shift
        monitor.add_loss(1.0, features)
    return monitor.observe(features, has_replay=True)


def test_window_drift_accounts_for_stationary_sampling_noise_and_detects_shift():
    monitor = OnlineMonitor(0.9)
    generator = torch.Generator().manual_seed(64)
    drifts = [observe_window(monitor, generator)[0].drift for _ in range(12)]
    assert sum(drifts[1:])/11 < 0.025
    assert observe_window(monitor, generator, shift=1.0)[0].drift > 0.8


def test_surprise_compares_to_previous_window_and_damage_is_paired():
    monitor = OnlineMonitor()
    generator = torch.Generator().manual_seed(11)
    observe_window(monitor, generator)
    monitor.record_damage(2.0, 2.5)
    features = torch.randn(32, 32, generator=generator)
    monitor.add_loss(4.0, features)
    signals, _ = monitor.observe(features, True)
    assert signals.surprise > 2
    assert signals.replay_damage == pytest.approx(0.25)
    assert monitor.record_damage(2.0, 1.0) == 0


def test_monitor_checkpoint_restores_pending_window_features():
    a = OnlineMonitor()
    generator = torch.Generator().manual_seed(27)
    observe_window(a, generator)
    features = torch.randn(32, 32, generator=generator)
    a.add_loss(0.7, features)
    b = OnlineMonitor()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    assert a.observe(features, True) == b.observe(features, True)
