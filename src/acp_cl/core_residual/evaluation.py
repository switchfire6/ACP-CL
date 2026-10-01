"""Acquisition probes with complete, independently scoreable forecast traces.

The learner receives only query observations and the supplied preceding support.
Physical laws, counterfactual outcomes, masks and cue interventions belong to the
evaluator. Tracing changes metric accumulation to float64, not prediction.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from acp_cl.acquisition.study import Evaluator
from acp_cl.acquisition.world import Law, mask_affected, mask_valid, signal_images
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.world import Experience, IMAGE_SHAPE
from acp_cl.predictive_value.mechanism import predict_all


_BASE_ARRAYS = ("probabilities", "truth", "observations", "affected", "valid")
_SUPPORT_ARRAYS = ("support_observations", "support_actions", "support_outcomes")
_RECORD_FIELDS = {
    "index", "law", "cue", "branch", "flipped", "support_fingerprint",
    "query_sha256", "model_sha256", "metrics", "marginal", "query_presentations",
    "support_presentations", "observed_support_presentations", "array_sha256",
}


def array_hash(values):
    """Hash shape, dtype and C-order bytes using the archived array convention."""
    values = np.asarray(values)
    if values.dtype.hasobject:
        raise ValueError("object arrays cannot be traced")
    return hashlib.sha256(
        str((values.shape, values.dtype.str)).encode() + values.tobytes(order="C")
    ).hexdigest()


def _numeric(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in "buif" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return values


def recompute_metrics(probabilities, truth, affected, valid, marginal=None):
    """Reconstruct the old all-action metrics using only NumPy arrays.

    Brier uses every action and all three horizons, with float64 subtraction and
    averaging. Survival uses the first argmax action at the last horizon. The
    probabilities are never clamped; float32 epsilon excursions are accepted.
    Masks must select at least one case so that every reported mean is defined.
    """
    probabilities = _numeric(probabilities, "probabilities")
    truth = _numeric(truth, "truth")
    if (probabilities.ndim != 3 or probabilities.shape[1:] != (5, 3)
            or len(probabilities) == 0 or truth.shape != probabilities.shape):
        raise ValueError("nonempty matching query-by-five-action-by-three-horizon arrays required")
    epsilon = np.finfo(np.float32).eps
    if np.any(probabilities < -epsilon) or np.any(probabilities > 1 + epsilon):
        raise ValueError("probabilities exceed float32 epsilon bounds")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("physical truth must be binary")
    masks = []
    for name, values in (("affected", affected), ("valid", valid)):
        values = np.asarray(values)
        if (values.dtype != np.bool_ or values.shape != (len(probabilities),)
                or not values.any()):
            raise ValueError(f"{name} must be a nonempty boolean mask over queries")
        masks.append(values)
    affected, valid = masks
    truth64 = truth.astype(np.float64)
    error = ((probabilities.astype(np.float64) - truth64) ** 2).mean(axis=(1, 2))
    actions = probabilities[:, :, -1].argmax(axis=1)
    survival = truth64[np.arange(len(actions)), actions, -1]
    result = dict(
        brier=float(error.mean()), focus_brier=float(error[affected].mean()),
        survival=float(survival.mean()), focus_survival=float(survival[affected].mean()),
        valid_brier=float(error[valid].mean()), valid_survival=float(survival[valid].mean()),
        no_transfer=float(truth64[:, 0, -1].mean()),
        clairvoyant_upper=float(truth64[:, :, -1].max(axis=1).mean()),
    )
    if marginal is not None:
        marginal = _numeric(marginal, "marginal")
        if (marginal.shape != (5, 3) or np.any(marginal < 0)
                or np.any(marginal > 1)):
            raise ValueError("marginal must be a five-action-by-three-horizon probability array")
        result["marginal_brier"] = float(
            ((marginal.astype(np.float64)[None] - truth64) ** 2).mean()
        )
    return result


def _images(values, size, name):
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint8
            or values.shape != (size, *IMAGE_SHAPE)):
        raise ValueError(f"{name} must contain query-sized uint8 sensor images")


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class TraceEvaluator(Evaluator):
    """The existing acquisition evaluator with a checkpointable per-call trace.

    Inherited ``probe`` and ``valid_panel`` keep their schemas. Every evaluate
    result adds its integer ``trace`` index; even calls whose results are later
    averaged by valid_panel remain in the complete trace. None support executes
    batch_size dummy encoder rows, counted separately from observed support rows.
    """

    def __init__(self, seed, settings, snapshot=None):
        if type(seed) is not int or seed < 0:
            raise ValueError("a nonnegative integer evaluator seed is required")
        for key in ("eval_size", "batch_size", "support_replicates"):
            if type(settings.get(key)) is not int or settings[key] < 1:
                raise ValueError(f"positive integer {key} required")
        if settings["eval_size"] % 64:
            raise ValueError("evaluation size must contain complete 64-case factorial grids")
        super().__init__(seed, copy.deepcopy(settings))
        self.records, self.arrays = [], {}
        if snapshot is not None:
            if not isinstance(snapshot, dict) or set(snapshot) != {"records", "arrays"}:
                raise ValueError("snapshot must contain exactly records and arrays")
            self.records = copy.deepcopy(snapshot["records"])
            self.arrays = copy.deepcopy(snapshot["arrays"])
            self._validate_trace()

    def evaluate(self, learner, law, support, cue=None, branch="novel", flipped=False, marginal=None):
        if not isinstance(law, Law):
            raise ValueError("an evaluator-owned physical Law is required")
        if cue is not None and (type(cue) is not int or cue not in range(3)):
            raise ValueError("cue must be None or an integer from zero to two")
        if branch not in ("novel", "return", "revision", "noise") or type(flipped) is not bool:
            raise ValueError("invalid evaluation branch or flipped flag")
        if flipped and cue is None:
            raise ValueError("flipped-cue probe requires a cue")
        if learner.settings["batch_size"] != self.settings["batch_size"]:
            raise ValueError("learner and evaluator support budgets differ")
        cases, truth = self.dataset(law)
        observations = cases.observations
        if flipped:
            signals = cases.signals.copy()
            signals[:, cue] ^= 1
            observations = signal_images(observations, signals)
        model_hash = state_hash(learner.model.state_dict())
        # Only these two observable arguments reach the unchanged predictor.
        probabilities = predict_all(learner, observations, support)
        affected, valid = mask_affected(cases, cue), mask_valid(cases, branch)
        metrics = recompute_metrics(probabilities, truth, affected, valid, marginal)
        values = dict(probabilities=probabilities, truth=truth, observations=observations,
                      affected=affected, valid=valid)
        if support is not None:
            values.update(support_observations=support.observations,
                          support_actions=support.actions, support_outcomes=support.survival)
        values = {key: np.array(value, copy=True, order="C") for key, value in values.items()}
        index = len(self.records)
        law_record = asdict(law)
        law_record["active"] = list(law.active)
        row = dict(
            index=index, law=law_record, cue=cue, branch=branch, flipped=flipped,
            support_fingerprint=None if support is None else support.fingerprint(),
            query_sha256=array_hash(observations), model_sha256=model_hash,
            metrics=metrics, marginal=None if marginal is None else np.asarray(marginal).tolist(),
            query_presentations=len(observations),
            support_presentations=self.settings["batch_size"] if support is None else len(support),
            observed_support_presentations=0 if support is None else len(support),
            array_sha256={key: array_hash(value) for key, value in values.items()},
        )
        self._validate_row(row, values, index)
        self.arrays.update({f"e{index}_{key}": value for key, value in values.items()})
        self.records.append(row)
        return dict(metrics, trace=index)

    def _validate_row(self, row, values, index):
        if (not isinstance(row, dict) or set(row) != _RECORD_FIELDS
                or type(row["index"]) is not int or row["index"] != index):
            raise ValueError("invalid or out-of-order evaluation record")
        law = row["law"]
        if (not isinstance(law, dict) or set(law) != {"mode", "active", "revised", "noise"}
                or type(law["mode"]) is not int or type(law["active"]) is not list
                or type(law["revised"]) is not bool):
            raise ValueError("invalid recorded physical law")
        Law(law["mode"], tuple(law["active"]), law["revised"], law["noise"])
        cue = row["cue"]
        if (cue is not None and (type(cue) is not int or cue not in range(3))
                or row["branch"] not in ("novel", "return", "revision", "noise")
                or type(row["flipped"]) is not bool or (row["flipped"] and cue is None)):
            raise ValueError("invalid recorded probe metadata")
        if not _sha256(row["model_sha256"]) or not _sha256(row["query_sha256"]):
            raise ValueError("invalid model or query fingerprint")
        present = row["support_fingerprint"] is not None
        expected = set(_BASE_ARRAYS + (_SUPPORT_ARRAYS if present else ()))
        if set(values) != expected or set(row["array_sha256"]) != expected:
            raise ValueError("evaluation array coverage mismatch")
        for name, value in values.items():
            if not isinstance(value, np.ndarray) or array_hash(value) != row["array_sha256"][name]:
                raise ValueError("evaluation array hash mismatch")
        size = self.settings["eval_size"]
        _images(values["observations"], size, "query observations")
        if values["probabilities"].shape != (size, 5, 3):
            raise ValueError("prediction count differs from query count")
        if row["query_sha256"] != array_hash(values["observations"]):
            raise ValueError("query fingerprint mismatch")
        if type(row["query_presentations"]) is not int or row["query_presentations"] != size:
            raise ValueError("query count mismatch")
        if present:
            actions = values["support_actions"]
            outcomes = _numeric(values["support_outcomes"], "support outcomes")
            if actions.ndim != 1:
                raise ValueError("support actions must be a one-dimensional array")
            count = len(actions)
            _images(values["support_observations"], count, "support observations")
            if (not count or actions.shape != (count,) or actions.dtype.kind not in "iu"
                    or np.any(actions < 0) or np.any(actions >= 5)
                    or outcomes.shape != (count, 3) or np.any(outcomes < 0) or np.any(outcomes > 1)):
                raise ValueError("invalid recorded support")
            support = Experience(values["support_observations"], actions, outcomes)
            if support.fingerprint() != row["support_fingerprint"]:
                raise ValueError("support fingerprint mismatch")
        else:
            count = 0
        if (type(row["observed_support_presentations"]) is not int
                or row["observed_support_presentations"] != count
                or type(row["support_presentations"]) is not int
                or row["support_presentations"] != (count if present else self.settings["batch_size"])):
            raise ValueError("support count mismatch")
        metrics = recompute_metrics(values["probabilities"], values["truth"],
                                    values["affected"], values["valid"], row["marginal"])
        if row["metrics"] != metrics:
            raise ValueError("recorded evaluation metrics differ from raw arrays")

    def _validate_trace(self):
        if not isinstance(self.records, list) or not isinstance(self.arrays, dict):
            raise ValueError("records must be a list and arrays a dictionary")
        expected = set()
        for index, row in enumerate(self.records):
            if not isinstance(row, dict) or not isinstance(row.get("array_sha256"), dict):
                raise ValueError("invalid evaluation record")
            names = {name: f"e{index}_{name}" for name in row["array_sha256"]}
            if not set(names.values()) <= set(self.arrays):
                raise ValueError("missing evaluation arrays")
            self._validate_row(row, {name: self.arrays[key] for name, key in names.items()}, index)
            expected.update(names.values())
        if set(self.arrays) != expected:
            raise ValueError("unreferenced evaluation arrays")

    def snapshot(self):
        """Return an independent, directly pickleable record/array checkpoint."""
        self._validate_trace()
        return dict(records=copy.deepcopy(self.records), arrays=copy.deepcopy(self.arrays))

    def export(self, folder):
        """Atomically replace each JSON/NPZ artifact and return relative hashes."""
        self._validate_trace()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        counts = dict(calls=len(self.records), **{
            name: sum(row[name] for row in self.records) for name in (
                "query_presentations", "support_presentations", "observed_support_presentations")
        })
        arrays_path = folder / "evaluations.npz"
        arrays_temporary = folder / "evaluations.npz.tmp"
        with arrays_temporary.open("wb") as handle:
            np.savez_compressed(handle, **self.arrays)
        arrays_temporary.replace(arrays_path)
        json_path = folder / "evaluations.json"
        json_temporary = folder / "evaluations.json.tmp"
        payload = dict(seed=self.seed, records=self.records, **counts)
        json_temporary.write_bytes((json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())
        json_temporary.replace(json_path)
        return dict(json=dict(path=json_path.name, sha256=hashlib.sha256(json_path.read_bytes()).hexdigest()),
                    arrays=dict(path=arrays_path.name, sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest()),
                    **counts)
