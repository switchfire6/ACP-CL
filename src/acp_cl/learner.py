"""Online learners sharing data, optimizer, replay, and accounting machinery."""

from __future__ import annotations

from dataclasses import asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from .controller import ControllerConfig, CriticalPeriodController, Phase
from .controller_v2 import ReopeningController
from .data import preprocess
from .monitor import OnlineMonitor
from .plasticity import PlasticityConfig, PlasticityEngine
from .replay import ReservoirBuffer
from .sensor import FrozenInputSensor


YOKED_METHODS = ("er_recycle_yoked", "er_recycle_yoked_gain")
METHODS = (
    "finetune", "er", "er_recycle", "derpp", "fixed", "acp",
    "acp_no_consolidation", "acp_no_replay", "acp_no_recycling",
    "acp_global", "acp_no_relaxation", "acp_no_damage", "acp_no_health",
    "acp_random_recycling", "acp_no_reopening",
    "acp_v2", "acp_v2_no_newborn", "acp_v2_no_reopening",
    "acp_v2_learned_sensor", "acp_v2_oracle",
) + YOKED_METHODS


class Learner:
    def __init__(self, model: nn.Module, method: str, config: dict, seed: int = 0):
        if method not in METHODS:
            raise ValueError(f"unknown method {method}; choose from {METHODS}")
        self.model, self.method, self.config = model, method, config
        self.device = next(model.parameters()).device
        self.dataset = config["data"]["dataset"]
        acp = method.startswith("acp") or method == "fixed"
        self.v2 = method.startswith("acp_v2")
        self.oracle = method == "acp_v2_oracle"
        self.yoked = method in YOKED_METHODS
        self.use_replay = method not in ("finetune", "acp_no_replay")
        self.replay_weight = float(config.get("replay_weight", 1.0))
        self.der_alpha = float(config.get("der_alpha", 0.5))
        self.replay_batch_size = int(config.get("replay_batch_size", 32))
        self.probe_batch_size = int(config.get("probe_batch_size", 32))
        if min(self.replay_weight, self.der_alpha, self.replay_batch_size, self.probe_batch_size) < 0:
            raise ValueError("replay coefficients and batch sizes must be nonnegative")
        cc = dict(config.get("controller", {}))
        cc.update({"adaptive": method != "fixed",
                   "allow_reopening": method not in ("acp_no_reopening", "acp_v2_no_reopening"),
                   "local_gating": method != "acp_global",
                   "use_replay_damage": method != "acp_no_damage",
                   "use_plasticity_signal": method != "acp_no_health"})
        self.controller_config = ControllerConfig(**cc)
        pc = dict(config.get("plasticity", {}))
        pc["newborn_steps"] = pc.get("newborn_steps", 50) if self.v2 and \
            method != "acp_v2_no_newborn" else 0
        pc["g_min"] = self.controller_config.g_min if acp else 0.0
        self.engine = PlasticityEngine(
            model, PlasticityConfig(**pc), consolidation=acp and method != "acp_no_consolidation",
            recycling=(acp and method != "acp_no_recycling") or method == "er_recycle" or self.yoked,
            relaxation=method != "acp_no_relaxation", random_recycling=method == "acp_random_recycling",
            seed=seed + 4109,
        )
        blocks = [name for name in model.plastic_blocks() if name != "head"]
        self.controller = CriticalPeriodController(blocks, self.controller_config) if acp else None
        if self.v2:
            self.controller = ReopeningController(blocks, self.controller_config, oracle=self.oracle)
        self.sensor = FrozenInputSensor(seed=seed+18223) if self.v2 and \
            method != "acp_v2_learned_sensor" else None
        self.monitor = OnlineMonitor(float(config.get("monitor_decay", 0.9)))
        self.buffer = ReservoirBuffer(int(config.get("replay_capacity", 2000)) if self.use_replay else 0,
                                      seed=seed+6151)
        self.augmentation_rng = torch.Generator().manual_seed(seed+8171)
        self.replay_augmentation_rng = torch.Generator().manual_seed(seed+10103)
        self.step_number = 0
        self.log: list[dict] = []
        self.allocation_trace: list[dict] = []
        self.yoked_schedule: list[dict] | None = None
        self.yoked_gain = 1.0
        self.cost = {"current_examples": 0, "replay_examples": 0,
                     "probe_forward_examples": 0, "train_forward_calls": 0,
                     "backward_calls": 0, "evaluation_examples": 0,
                     "sensor_forward_examples": 0}

    def set_yoked_schedule(self, trace: list[dict]) -> None:
        if not self.yoked or not trace:
            raise ValueError("a nonempty allocation trace is required for a yoked diagnostic")
        if any(row["step"] != i+1 or not 0 <= row["effective_feature_gain"] <= 1.000001
               for i, row in enumerate(trace)):
            raise ValueError("invalid allocation trace")
        self.yoked_schedule = trace
        self.yoked_gain = sum(row["effective_feature_gain"] for row in trace) / len(trace) \
            if self.method == "er_recycle_yoked_gain" else 1.0

    def notify_oracle_boundary(self) -> None:
        if not self.oracle:
            raise ValueError("only the explicitly labeled oracle may receive boundaries")
        self.controller.notify_boundary()

    def _scores(self, current: dict[str, torch.Tensor], old: dict[str, torch.Tensor]) -> dict:
        scores = {}
        mean_c = self.engine.block_consolidation()
        for block in self.controller.block_names:
            names = [n for n in current if self.engine.block_for[n] == block]
            sum_new = sum(float(current[n].square().sum()) for n in names)
            sum_old = sum(float(old[n].square().sum()) for n in names) if old else 0
            dot = sum(float((current[n]*old[n]).sum()) for n in names) if old else 0
            elements = sum(current[n].numel() for n in names)
            conflict = max(0.0, -dot / max(math.sqrt(sum_new*sum_old), 1e-12))
            # RMS gradients avoid selecting a large block merely for having more
            # parameters. Standardization across blocks follows below.
            scores[block] = [math.sqrt(sum_new/max(elements, 1)), conflict, mean_c[block]]
        norms = [v[0] for v in scores.values()]
        mean = sum(norms)/len(norms)
        std = math.sqrt(sum((v-mean)**2 for v in norms)/len(norms))
        return {block: (values[0]-mean)/max(std, 1e-12)-values[1]-values[2]
                for block, values in scores.items()}

    def train_batch(self, raw_x: torch.Tensor, y: torch.Tensor) -> dict:
        if self.yoked and (self.yoked_schedule is None or self.step_number >= len(self.yoked_schedule)):
            raise ValueError("yoked diagnostic requires a complete source allocation trace")
        self.model.train()
        self.step_number += 1
        step = self.step_number
        monitoring = step % self.controller_config.monitor_interval == 0
        replay = self.buffer.sample(self.replay_batch_size, device=self.device)
        probe = self.buffer.sample(self.probe_batch_size, device=self.device, stream="monitor") \
            if monitoring else None
        x = preprocess(raw_x.to(self.device), self.dataset, training=True,
                       generator=self.augmentation_rng)
        y = y.to(self.device)
        probe_x = preprocess(probe.x, self.dataset) if probe is not None else None
        with torch.no_grad():
            probe_before = float(F.cross_entropy(self.model(probe_x), probe.y)) \
                if probe is not None else None
        logits, features = self.model(x, return_features=True)
        loss = F.cross_entropy(logits, y)
        self.cost["current_examples"] += len(y)
        self.cost["train_forward_calls"] += 1
        params = tuple(self.engine.params.values())
        names = tuple(self.engine.params)
        new_grad = dict(zip(names, torch.autograd.grad(loss, params)))
        self.cost["backward_calls"] += 1
        old_grad = {}
        replay_loss = 0.0
        if replay is not None:
            old_x = preprocess(replay.x, self.dataset, training=True,
                               generator=self.replay_augmentation_rng)
            old_logits = self.model(old_x)
            old_loss = F.cross_entropy(old_logits, replay.y)
            if self.method == "derpp":
                # Stored predictions were measured at arrival BEFORE its update.
                # This variant uses the same sampled examples for CE and MSE.
                old_loss = self.replay_weight*old_loss + self.der_alpha*F.mse_loss(
                    old_logits, replay.logits)
            else:
                old_loss = self.replay_weight*old_loss
            old_grad = dict(zip(names, torch.autograd.grad(old_loss, params)))
            replay_loss = float(old_loss.detach())
            self.cost["replay_examples"] += len(replay.y)
            self.cost["train_forward_calls"] += 1
            self.cost["backward_calls"] += 1
        gradients = {n: g + old_grad[n] if old_grad else g for n, g in new_grad.items()}
        novelty_features = features["representation"]
        if self.sensor is not None:
            novelty_features = self.sensor.transform(preprocess(raw_x.to(self.device), self.dataset))
            self.cost["sensor_forward_examples"] += len(y)
        self.monitor.add_loss(float(loss.detach()), novelty_features)
        monitor_log = {}
        controller_log = {}
        if monitoring:
            signals, monitor_log = self.monitor.observe(features["representation"], probe is not None)
            if self.controller:
                controller_log = self.controller.observe(step, signals, self._scores(new_grad, old_grad))
                if self.controller.should_consolidate:
                    self.engine.consolidate(self.controller.consolidation_blocks)
        gates = self.controller.gates() if self.controller else {b: 1.0 for b in self.model.plastic_blocks()}
        if self.yoked:
            gates = {b: self.yoked_gain for b in self.model.plastic_blocks() if b != "head"}
        reopened = self.controller.selected_blocks if self.controller and \
            self.controller.phase == Phase.REOPENED else ()
        update_log = self.engine.step(gradients, gates, reopened)
        self.engine.update_utilities(features)
        adult = self.controller is not None and self.controller.phase in (Phase.ADULT, Phase.REOPENED)
        forced = self.yoked_schedule[step-1]["recycled_by_population"] if self.yoked else None
        recycle_log = self.engine.recycle(adult=adult, forced_counts=forced)
        self.allocation_trace.append({"step": step,
                                      "effective_feature_gain": update_log["effective_feature_gain"],
                                      "feature_data_displacement": update_log["feature_data_displacement"],
                                      "recycled_by_population": recycle_log["recycled_by_population"]})
        # Probe uses the same unaugmented images and labels on both sides. It is
        # sampled separately from the update, but chance overlap is possible.
        if probe is not None:
            with torch.no_grad():
                probe_after = float(F.cross_entropy(self.model(probe_x), probe.y))
            monitor_log["paired_replay_damage"] = self.monitor.record_damage(probe_before, probe_after)
            monitor_log["probe_loss_before"] = probe_before
            monitor_log["probe_loss_after"] = probe_after
            self.cost["probe_forward_examples"] += 2*len(probe.y)
        self.buffer.add(raw_x, y.cpu(), logits.detach().cpu() if self.method == "derpp" else None)
        event = {"step": step, "loss": float(loss.detach()), "replay_loss": replay_loss,
                 "phase": self.controller.phase.value if self.controller else "uncontrolled",
                 "gates": gates, **update_log, **recycle_log}
        if monitoring or recycle_log["recycled"]:
            event.update({"monitor": monitor_log, "controller": controller_log,
                          **self.engine.diagnostics(), "replay_bytes": self.buffer.nbytes()})
            self.log.append(event)
        return event

    @torch.no_grad()
    def accuracy(self, dataset, batch_size: int = 256) -> float:
        self.model.eval()
        correct = 0
        x, y = dataset.tensors
        for start in range(0, len(y), batch_size):
            xb = preprocess(x[start:start+batch_size].to(self.device), self.dataset)
            logits = self.model(xb)
            correct += int((logits.argmax(1).cpu() == y[start:start+batch_size]).sum())
        self.cost["evaluation_examples"] += len(y)
        return correct/max(1, len(y))

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "engine": self.engine.state_dict(),
                "controller": self.controller.state_dict() if self.controller else None,
                "monitor": self.monitor.state_dict(), "buffer": self.buffer.state_dict(),
                "sensor": self.sensor.state_dict() if self.sensor else None,
                "allocation_trace": self.allocation_trace, "yoked_schedule": self.yoked_schedule,
                "yoked_gain": self.yoked_gain,
                "step_number": self.step_number, "log": self.log, "cost": self.cost,
                "augmentation_rng": self.augmentation_rng.get_state(),
                "replay_augmentation_rng": self.replay_augmentation_rng.get_state(),
                "method": self.method, "plasticity_config": asdict(self.engine.config)}

    def load_state_dict(self, state: dict) -> None:
        if state["method"] != self.method:
            raise ValueError("checkpoint method does not match learner")
        if state["plasticity_config"] != asdict(self.engine.config):
            raise ValueError("checkpoint plasticity configuration does not match learner")
        # Controller performs its own config/schema validation before tensors
        # are changed; the experiment runner additionally hashes the full config.
        if self.controller:
            self.controller.load_state_dict(state["controller"])
        self.model.load_state_dict(state["model"])
        self.engine.load_state_dict(state["engine"])
        self.monitor.load_state_dict(state["monitor"])
        if self.sensor:
            self.sensor.load_state_dict(state["sensor"])
        self.allocation_trace = state.get("allocation_trace", [])
        self.yoked_schedule, self.yoked_gain = state.get("yoked_schedule"), state.get("yoked_gain", 1.0)
        self.buffer.load_state_dict(state["buffer"])
        self.step_number, self.log, self.cost = state["step_number"], state["log"], state["cost"]
        self.augmentation_rng.set_state(state["augmentation_rng"].cpu())
        self.replay_augmentation_rng.set_state(state["replay_augmentation_rng"].cpu())
