"""From-scratch predictors with separate acquisition and consolidation.

The stable and fast networks both see raw preprocessed input. Consolidation
uses a bounded training-only pool and cached, immutable combined predictions;
no task identifier, validation score, or historical checkpoint is a teacher.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import preprocess
from .models import make_model
from .replay import ReservoirBuffer


DUAL_METHODS = ("dual_joint", "dual_derpp", "dual_frozen", "dual_consolidate")


@dataclass(frozen=True)
class DualPathConfig:
    fast_width: int = 32
    consolidation_interval: int = 73
    consolidation_steps: int = 24
    consolidation_batch_size: int = 64
    consolidation_pool_size: int = 256
    consolidation_lr: float = 0.03
    distillation_weight: float = 1.0
    temperature: float = 2.0
    max_grad_norm: float = 10.0
    probe_interval: int = 10

    def __post_init__(self):
        for name in ("fast_width", "consolidation_interval", "consolidation_steps",
                     "consolidation_batch_size", "consolidation_pool_size", "probe_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("consolidation_lr", "temperature", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.distillation_weight) or self.distillation_weight < 0:
            raise ValueError("distillation_weight must be finite and nonnegative")


def tensor_bytes(value) -> int:
    """Tensor payload only; does not estimate Python or allocator overhead."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def _seeded_model(name, input_shape, num_classes, width, seed):
    # Initialization stays on CPU and cannot advance any caller's CPU/CUDA RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed) % (2**63))
        return make_model(name, input_shape, num_classes, width)


class DualPredictor(nn.Module):
    def __init__(self, stable, fast):
        super().__init__()
        self.stable, self.fast = stable, fast

    def forward(self, x):
        return self.stable(x) + self.fast(x)


class DualPathLearner:
    """Fixed two-network capacity; extra consolidation compute is explicit.

    ``dual_scratch`` is an internal fresh, jointly trained, replay-free reference.
    ``dual_frozen`` leaves the initially random stable path frozen indefinitely;
    it is an ablation, not a pretrained-encoder method.
    """

    controller = sensor = None
    oracle = yoked = v3 = False

    def __init__(self, config, input_shape, num_classes, method, seed=0, device="cpu"):
        if method not in (*DUAL_METHODS, "dual_scratch"):
            raise ValueError(f"unknown dual-path method: {method}")
        self.config = copy.deepcopy(config)
        self.method, self.dataset = method, config["data"]["dataset"]
        self.device = torch.device(device)
        options = dict(config.get("dual_path", {}))
        options.setdefault("fast_width", max(8, int(config.get("width", 128)) // 2))
        self.settings = DualPathConfig(**options)
        self.model_name, self.input_shape = config["model"], tuple(input_shape)
        self.num_classes = num_classes
        self.optimizer_config = {k: config.get("plasticity", {}).get(k, default)
                                 for k, default in (("lr", .03), ("momentum", .9),
                                                    ("weight_decay", .0001))}
        if any(not math.isfinite(v) or v < 0 for v in self.optimizer_config.values()):
            raise ValueError("optimizer settings must be finite and nonnegative")
        self.replay_weight = float(config.get("replay_weight", 1.0))
        self.der_alpha = float(config.get("der_alpha", .5))
        for name in ("replay_weight", "der_alpha"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.replay_batch_size = config.get("replay_batch_size", 32)
        self.probe_batch_size = config.get("probe_batch_size", 32)
        for name in ("replay_batch_size", "probe_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.separated = method in ("dual_frozen", "dual_consolidate")
        stable = _seeded_model(self.model_name, input_shape, num_classes,
                               int(config.get("width", 128)), seed + 16127)
        fast = _seeded_model(self.model_name, input_shape, num_classes,
                             self.settings.fast_width, seed + 46183)
        nn.init.zeros_(fast.head.weight)
        nn.init.zeros_(fast.head.bias)
        self.model = DualPredictor(stable, fast).to(self.device)
        self.model.stable.requires_grad_(not self.separated)
        self.online_optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad], **self.optimizer_config)
        self.stable_optimizer = torch.optim.SGD(
            self.model.stable.parameters(), **{**self.optimizer_config,
                                               "lr": self.settings.consolidation_lr})
        self.buffer = ReservoirBuffer(0 if method == "dual_scratch" else
                                      config.get("replay_capacity", 256), seed=seed + 6151)
        self.augmentation_rng = torch.Generator().manual_seed(seed + 8171)
        self.replay_augmentation_rng = torch.Generator().manual_seed(seed + 10103)
        self.consolidation_rng = torch.Generator().manual_seed(seed + 55109)
        self.reset_rng = torch.Generator().manual_seed(seed + 66109)
        self.step_number = self.consolidations = self.fast_resets = 0
        self.log = []
        self.allocation_trace = []  # Not a neuron-gain learner; no allocation claims.
        self.cost = dict.fromkeys((
            "current_examples", "replay_examples", "probe_forward_examples",
            "train_forward_calls", "backward_calls", "evaluation_examples",
            "sensor_forward_examples", "online_branch_forward_examples",
            "online_branch_backward_examples", "online_optimizer_steps",
            "consolidation_examples", "consolidation_backward_calls",
            "consolidation_optimizer_steps", "teacher_forward_examples",
            "consolidation_branch_forward_examples", "probe_branch_forward_examples",
            "evaluation_branch_forward_examples"), 0)
        self.peak_auxiliary_payload_bytes = 0
        self.peak_persistent_payload_bytes = 0
        self._record_memory()

    def _record_memory(self, auxiliary_bytes=0):
        self.peak_auxiliary_payload_bytes = max(self.peak_auxiliary_payload_bytes, auxiliary_bytes)
        persistent = (tensor_bytes(self.model.state_dict()) + self.buffer.nbytes() +
                      tensor_bytes(self.online_optimizer.state) +
                      tensor_bytes(self.stable_optimizer.state))
        self.peak_persistent_payload_bytes = max(self.peak_persistent_payload_bytes, persistent)

    def _online_mode(self):
        self.model.train()
        if self.separated:
            self.model.stable.eval()
            self.model.stable.requires_grad_(False)

    @torch.no_grad()
    def _probe(self, x, y):
        if x is None:
            return None
        self.model.eval()
        logits = self.model(x)
        self.cost["probe_forward_examples"] += len(y)
        self.cost["probe_branch_forward_examples"] += 2 * len(y)
        return {"loss": float(F.cross_entropy(logits, y)),
                "accuracy": float((logits.argmax(1) == y).float().mean())}

    def _step(self, loss, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite dual-path loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for group in optimizer.param_groups for p in group["params"]]
        norm = nn.utils.clip_grad_norm_(params, self.settings.max_grad_norm,
                                       error_if_nonfinite=True)
        before = [p.detach().clone() for p in params]
        optimizer.step()
        displacement = math.sqrt(sum(float((p.detach() - old).square().sum())
                                     for p, old in zip(params, before)))
        # Snapshot payload is transient diagnostic memory, not a teacher model.
        return displacement, float(norm), tensor_bytes(before)

    def train_batch(self, raw_x, y):
        self.step_number += 1
        due = self.method == "dual_consolidate" and \
            self.step_number % self.settings.consolidation_interval == 0
        monitor = due or self.step_number % self.settings.probe_interval == 0
        replay = self.buffer.sample(self.replay_batch_size, device=self.device)
        probe = self.buffer.sample(self.probe_batch_size, device=self.device, stream="monitor") \
            if monitor else None
        probe_x = preprocess(probe.x, self.dataset) if probe is not None else None
        probe_y = probe.y if probe is not None else None
        before = self._probe(probe_x, probe_y)
        self._online_mode()
        x = preprocess(raw_x.to(self.device), self.dataset, training=True,
                       generator=self.augmentation_rng)
        y = y.to(self.device)
        logits = self.model(x)
        current_loss = F.cross_entropy(logits, y)
        loss, replay_loss = current_loss, 0.0
        self.cost["current_examples"] += len(y)
        self.cost["train_forward_calls"] += 1
        online_examples = len(y)
        if replay is not None:
            replay_x = preprocess(replay.x, self.dataset, training=True,
                                  generator=self.replay_augmentation_rng)
            old_logits = self.model(replay_x)
            old_loss = self.replay_weight * F.cross_entropy(old_logits, replay.y)
            if self.method == "dual_derpp":
                old_loss = old_loss + self.der_alpha * F.mse_loss(old_logits, replay.logits)
            replay_loss = float(old_loss.detach())
            loss = loss + old_loss
            online_examples += len(replay.y)
            self.cost["replay_examples"] += len(replay.y)
            self.cost["train_forward_calls"] += 1
        update_norm, grad_norm, snapshot_bytes = self._step(loss, self.online_optimizer)
        self.cost["backward_calls"] += 1
        self.cost["online_optimizer_steps"] += 1
        self.cost["online_branch_forward_examples"] += 2 * online_examples
        self.cost["online_branch_backward_examples"] += (1 if self.separated else 2) * online_examples
        self._record_memory(snapshot_bytes)
        after_online = self._probe(probe_x, probe_y)
        event = {"step": self.step_number, "phase": "separated" if self.separated else "joint",
                 "loss": float(current_loss.detach()), "replay_loss": replay_loss,
                 "update_norm": update_norm, "gradient_norm_before_clipping": grad_norm,
                 "consolidation": None}
        if monitor:
            event["probe"] = {"before_online": before, "after_online": after_online}
        if due:
            event["consolidation"] = self._consolidate(raw_x, y)
            event["probe"]["after_consolidation_before_reset"] = self._probe(probe_x, probe_y)
            self._reset_fast()
            event["probe"]["after_reset"] = self._probe(probe_x, probe_y)
        self.buffer.add(raw_x, y.cpu(), logits.detach().cpu()
                        if self.method == "dual_derpp" else None)
        self._record_memory()
        if monitor:
            self.log.append(event)
        self._online_mode()
        return event

    def _consolidate(self, raw_x, y):
        # Draw from the reservoir using a private RNG. Snapshotting its state
        # would needlessly copy the entire replay and its historical logits.
        count = min(len(self.buffer), self.settings.consolidation_pool_size)
        xs, ys = [raw_x.to(self.device)], [y]
        if count:
            replay = self.buffer.sample(count, device=self.device,
                                        generator=self.consolidation_rng)
            xs.append(replay.x)
            ys.append(replay.y)
        pool_x = preprocess(torch.cat(xs), self.dataset)
        pool_y = torch.cat(ys)
        self.model.eval()
        with torch.no_grad():
            target = self.model(pool_x).detach()
        self.cost["teacher_forward_examples"] += len(pool_y)
        self.cost["consolidation_branch_forward_examples"] += 2 * len(pool_y)
        self.model.stable.requires_grad_(True)
        self.model.stable.train()
        t = self.settings.temperature
        last_loss = 0.0
        pool_bytes = tensor_bytes((xs, ys, pool_x, pool_y, target))
        for _ in range(self.settings.consolidation_steps):
            ids = torch.randperm(len(pool_y), generator=self.consolidation_rng)
            ids = ids[:self.settings.consolidation_batch_size].to(self.device)
            student = self.model.stable(pool_x[ids])
            kd = F.kl_div(F.log_softmax(student / t, dim=1),
                          F.softmax(target[ids] / t, dim=1), reduction="batchmean") * t * t
            loss = F.cross_entropy(student, pool_y[ids]) + self.settings.distillation_weight * kd
            _, _, snapshot_bytes = self._step(loss, self.stable_optimizer)
            last_loss = float(loss.detach())
            self.cost["consolidation_examples"] += len(ids)
            self.cost["consolidation_backward_calls"] += 1
            self.cost["consolidation_optimizer_steps"] += 1
            self.cost["consolidation_branch_forward_examples"] += len(ids)
            self._record_memory(pool_bytes + snapshot_bytes)
        self.model.stable.requires_grad_(False)
        self.model.stable.eval()
        self.consolidations += 1
        return {"pool_examples": len(pool_y), "replay_pool_examples": count,
                "optimizer_steps": self.settings.consolidation_steps, "last_loss": last_loss,
                "teacher": "cached pre-consolidation combined logits on training pool"}

    def _reset_fast(self):
        reset_seed = int(torch.randint(2**63 - 1, (), generator=self.reset_rng))
        replacement = _seeded_model(self.model_name, self.input_shape, self.num_classes,
                                    self.settings.fast_width, reset_seed)
        nn.init.zeros_(replacement.head.weight)
        nn.init.zeros_(replacement.head.bias)
        self.model.fast.load_state_dict(replacement.state_dict())
        self.online_optimizer.state.clear()
        self.online_optimizer.zero_grad(set_to_none=True)
        self.fast_resets += 1
        self._record_memory(tensor_bytes(replacement.state_dict()))

    @torch.no_grad()
    def accuracy(self, dataset, batch_size=256):
        self.model.eval()
        raw_x, y = dataset.tensors
        correct = 0
        for start in range(0, len(y), batch_size):
            x = preprocess(raw_x[start:start + batch_size].to(self.device), self.dataset)
            correct += int((self.model(x).argmax(1).cpu() == y[start:start + batch_size]).sum())
        self.cost["evaluation_examples"] += len(y)
        self.cost["evaluation_branch_forward_examples"] += 2 * len(y)
        return correct / max(1, len(y))

    def diagnostics(self):
        return {"consolidations": self.consolidations, "fast_resets": self.fast_resets,
                "stable_parameters": sum(p.numel() for p in self.model.stable.parameters()),
                "fast_parameters": sum(p.numel() for p in self.model.fast.parameters()),
                "model_payload_bytes": tensor_bytes(self.model.state_dict()),
                "optimizer_payload_bytes": tensor_bytes(self.online_optimizer.state) +
                    tensor_bytes(self.stable_optimizer.state),
                "peak_persistent_payload_bytes": self.peak_persistent_payload_bytes,
                "peak_auxiliary_payload_bytes": self.peak_auxiliary_payload_bytes,
                "memory_scope": "tracked tensor payload; excludes activations, gradients, allocator, Python and logs",
                "inference": "stable(raw input) + fast(raw input); fixed output support; no task IDs"}

    def _identity(self):
        return {"method": self.method, "model_name": self.model_name,
                "input_shape": self.input_shape, "num_classes": self.num_classes,
                "width": self.config.get("width", 128), "dataset": self.dataset,
                "settings": asdict(self.settings), "optimizer_config": self.optimizer_config,
                "replay_capacity": self.buffer.capacity, "replay_batch_size": self.replay_batch_size,
                "probe_batch_size": self.probe_batch_size, "replay_weight": self.replay_weight,
                "der_alpha": self.der_alpha}

    def state_dict(self):
        return copy.deepcopy({
            "version": 1, "identity": self._identity(), "model": self.model.state_dict(),
            "online_optimizer": self.online_optimizer.state_dict(),
            "stable_optimizer": self.stable_optimizer.state_dict(), "buffer": self.buffer.state_dict(),
            "step_number": self.step_number, "consolidations": self.consolidations,
            "fast_resets": self.fast_resets, "cost": self.cost, "log": self.log,
            "peak_auxiliary_payload_bytes": self.peak_auxiliary_payload_bytes,
            "peak_persistent_payload_bytes": self.peak_persistent_payload_bytes,
            "rngs": {name: getattr(self, name).get_state() for name in (
                "augmentation_rng", "replay_augmentation_rng", "consolidation_rng", "reset_rng")}})

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("identity") != self._identity():
            raise ValueError("dual-path checkpoint configuration mismatch")
        self.model.load_state_dict(state["model"])
        self.online_optimizer.load_state_dict(state["online_optimizer"])
        self.stable_optimizer.load_state_dict(state["stable_optimizer"])
        self.buffer.load_state_dict(state["buffer"])
        for name in ("step_number", "consolidations", "fast_resets", "cost", "log",
                     "peak_auxiliary_payload_bytes", "peak_persistent_payload_bytes"):
            setattr(self, name, copy.deepcopy(state[name]))
        for name, rng in state["rngs"].items():
            getattr(self, name).set_state(rng.cpu())
        self._online_mode()
