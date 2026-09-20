"""Training-only signals; no experience identities or evaluation data enter here."""

from __future__ import annotations

import math

import torch

from .controller import MonitorSignals
from .metrics import representation_metrics


class OnlineMonitor:
    """EMA innovation signals, evaluated before incorporating the current window.

    Loss surprise uses an exponentially weighted absolute deviation (a robust
    scale proxy, not a calibrated z-score). Drift is a diagonal standardized
    distance between current and reference feature means. Replay damage must be
    supplied from paired before/after losses on the SAME replay probe.
    """

    def __init__(self, decay: float = 0.9):
        if not 0 <= decay < 1:
            raise ValueError("monitor decay must be in [0, 1)")
        self.decay = decay
        self.loss_mean: float | None = None
        self.loss_deviation = 0.0
        self.previous_loss: float | None = None
        self.mean: torch.Tensor | None = None
        self.variance: torch.Tensor | None = None
        self.mean_uncertainty: torch.Tensor | None = None
        self.feature_sum: torch.Tensor | None = None
        self.feature_squares: torch.Tensor | None = None
        self.feature_count = 0
        self.rank_peak = 0.0
        self.replay_damage = 0.0
        self.losses: list[float] = []

    def add_loss(self, loss: float, features: torch.Tensor | None = None) -> None:
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        self.losses.append(loss)
        if features is not None:
            features = features.detach().float().flatten(1)
            total = features.sum(0).cpu()
            squares = features.square().sum(0).cpu()
            if self.feature_sum is None:
                self.feature_sum, self.feature_squares = total, squares
            else:
                self.feature_sum.add_(total)
                self.feature_squares.add_(squares)
            self.feature_count += len(features)

    @torch.no_grad()
    def observe(self, features: torch.Tensor, has_replay: bool) -> tuple[MonitorSignals, dict]:
        if not self.losses:
            raise ValueError("a monitoring window needs training losses")
        loss = sum(self.losses) / len(self.losses)
        self.losses.clear()
        features = features.detach().float().flatten(1)
        if self.feature_count:
            current_mean = self.feature_sum / self.feature_count
            current_variance = (self.feature_squares / self.feature_count-current_mean.square()).clamp_min(0)
            current_uncertainty = current_variance / self.feature_count
        else:
            current_mean = features.mean(0).cpu()
            current_variance = features.var(0, unbiased=False).cpu()
            current_uncertainty = current_variance / len(features)
        self.feature_sum, self.feature_squares, self.feature_count = None, None, 0
        health = representation_metrics(features)
        rank = float(health["effective_rank"])
        rank_fraction = rank / max(1, min(features.shape[0] - 1, features.shape[1]))
        deficit = float(health["dormant_fraction"])
        if self.rank_peak > 0:
            deficit = 0.5 * (deficit + max(0.0, 1 - rank_fraction / self.rank_peak))
        self.rank_peak = max(self.rank_peak, rank_fraction)
        if self.loss_mean is None:
            surprise, drift, change = 0.0, 0.0, 1.0
            self.loss_mean = loss
            self.loss_deviation = max(abs(loss) * 0.1, 1e-3)
            self.mean = current_mean.clone()
            self.variance = current_variance.clone()
            self.mean_uncertainty = current_uncertainty.clone()
        else:
            surprise = max(0.0, (loss - self.loss_mean) / max(self.loss_deviation, 0.05))
            # Positive loss surprise and drift are capped; one noisy batch cannot
            # create an arbitrarily large controller command.
            surprise = min(6.0, surprise)
            # Correct the squared distance for the independent-sampling floor.
            # It is a heuristic when temporal batches are correlated, not a
            # statistical distribution-shift test. Full-window means reduce noise.
            squared_shift = ((current_mean-self.mean).square()-
                             current_uncertainty-self.mean_uncertainty)
            drift = min(6.0, math.sqrt(max(0.0, float((squared_shift /
                                                       (self.variance+1e-3)).mean()))))
            change = (loss - self.previous_loss) / max(abs(self.previous_loss), 0.1)
            innovation = abs(loss - self.loss_mean)
            self.loss_deviation = self.decay * self.loss_deviation + (1-self.decay) * innovation
            self.loss_mean = self.decay * self.loss_mean + (1-self.decay) * loss
            self.mean.lerp_(current_mean, 1-self.decay)
            self.variance.lerp_(current_variance, 1-self.decay)
            self.mean_uncertainty.mul_(self.decay**2).add_((1-self.decay)**2 * current_uncertainty)
        self.previous_loss = loss
        signals = MonitorSignals(
            surprise=surprise, drift=drift, plasticity_deficit=deficit,
            replay_damage=max(0.0, self.replay_damage), loss_change=change,
            has_replay=has_replay,
        )
        return signals, {**health, "normalized_effective_rank": rank_fraction,
                         "window_loss": loss, "loss_change": change,
                         "surprise": surprise, "drift": drift,
                         "plasticity_deficit": deficit,
                         "replay_damage": self.replay_damage}

    def record_damage(self, before: float, after: float) -> float:
        self.replay_damage = max(0.0, (after - before) / max(abs(before), 0.1))
        return self.replay_damage

    def state_dict(self) -> dict:
        return dict(self.__dict__)

    def load_state_dict(self, state: dict) -> None:
        self.__dict__.update(state)
