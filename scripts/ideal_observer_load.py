"""Read-only endpoint loader, case regeneration and locked-cell rescoring.

POST HOC / EXPLORATORY / NOT DECISION-BEARING. This module is an analysis aid
for the locked representation-learning qualification run
(runs/representation_learning_qualification). It never trains, updates,
unpickles or executes a learner or checkpoint, never writes into runs/, and
cannot alter the locked STOP decision recorded in
reports/representation_learning/qualification/summary.json.

Law identities, clean physical outcomes, counterfactual outcomes and the
regenerated case objects exposed here are EVALUATOR-ONLY quantities. An ideal
observer built from them is an analysis reference, not a learner.

Main entry points
-----------------
load_endpoint(run_dir, seed, phase)  -> Endpoint   (saved endpoint probe traces)
load_all(run_dir)                    -> {(seed, phase): Endpoint}
regenerate(endpoint)                 -> Regenerated (evaluator-owned cases, asserted
                                        byte-identical to the saved arrays)
score_cells(endpoints, probabilities=None)
                                     -> the 31 qualification cells, computed like
                                        scripts/summarize_representation_learning.py,
                                        optionally with substituted forecasts
compare_with_summary(result, summary_path)
structural_evidence(endpoints, regenerated)

Run ``python -B scripts/ideal_observer_load.py --verify`` for the full check.
"""

from __future__ import annotations

import sys

# The repository keeps sealed file inventories: never create __pycache__ files.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from functools import lru_cache  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
try:  # an editable install normally provides acp_cl; fall back to the source tree
    import acp_cl  # noqa: F401,E402
except ImportError:  # pragma: no cover - environment dependent
    sys.path.insert(0, str(ROOT / "src"))

from acp_cl.acquisition.world import (AcquisitionWorld, Law, LawCases, mask_affected,  # noqa: E402
                                      mask_valid, order, signal_images, stage_law)
from acp_cl.core_residual.evaluation import TraceEvaluator  # noqa: E402
from acp_cl.persistence.study import trial_seed  # noqa: E402
from acp_cl.persistence.world import Experience, IMAGE_SHAPE  # noqa: E402
from acp_cl.representation_learning.design import (law_from_record, law_key,  # noqa: E402
                                                   qualification_phases, reference_laws)


LABEL = "POST HOC / EXPLORATORY / NOT DECISION-BEARING"
DEFAULT_RUN = ROOT / "runs" / "representation_learning_qualification"
DEFAULT_SUMMARY = ROOT / "reports" / "representation_learning" / "qualification" / "summary.json"
PHASES = ("fresh_1", "fresh_2", "fresh_3", "interleaved")
ARM = "outcome"
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 28192026, 20000
# Live modules whose code paths regenerate evaluator cases; checked against the run lock.
REGENERATION_SOURCES = (
    "src/acp_cl/acquisition/world.py", "src/acp_cl/acquisition/study.py",
    "src/acp_cl/conditional/world.py", "src/acp_cl/persistence/world.py",
    "src/acp_cl/persistence/study.py", "src/acp_cl/core_residual/evaluation.py",
    "src/acp_cl/representation_learning/design.py",
    "src/acp_cl/predictive_value/mechanism.py", "src/acp_cl/conditional/learner.py",
    "src/acp_cl/contextual/learner.py", "src/acp_cl/representation_learning/learner.py",
    "src/acp_cl/representation_learning/study.py",
)
LOCKED_SCORER = "scripts/summarize_representation_learning.py"


def require(value, message):
    if not value:
        raise AssertionError(message)


# --------------------------------------------------------------------------- I/O

def read_json(path):
    """Strict JSON read (duplicate keys and non-finite constants rejected)."""

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key in {path}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON constant {value} in {path}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=nonfinite)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(*values):
    """Archived array convention: sha256 over str((shape, dtype.str)) + C bytes."""
    digest = hashlib.sha256()
    for value in values:
        require(not value.dtype.hasobject, "object arrays are not valid records")
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def law_record(law):
    value = asdict(law)
    value["active"] = list(law.active)
    return value


def _frozen(array):
    array = np.array(array, copy=True, order="C")
    array.setflags(write=False)
    return array


@lru_cache(maxsize=None)
def _manifest(run_dir):
    return read_json(Path(run_dir) / "manifest.json")


def run_config(run_dir=DEFAULT_RUN):
    return _manifest(str(Path(run_dir).resolve()))["config"]


def source_lock_check(run_dir=DEFAULT_RUN):
    """Compare live regeneration/scoring sources with the run's locked hashes."""
    manifest = _manifest(str(Path(run_dir).resolve()))
    locked = {**manifest["source_files"], **manifest["analysis_files"]}
    report = {}
    for name in (*REGENERATION_SOURCES, LOCKED_SCORER):
        live = sha(ROOT / name)
        report[name] = dict(locked=locked.get(name), live=live, matches=locked.get(name) == live)
    return report


# ------------------------------------------------------------------ endpoint load

@dataclass(frozen=True, eq=False)
class EndpointEval:
    """One saved endpoint evaluation call (evaluations.json record + npz arrays)."""

    seed: int
    phase: str
    trace: int                 # index in evaluations.json / key prefix e<trace>_ in the npz
    slot: int                  # index into phase_spec["targets"] / end_probes
    replicate: int             # support replicate (0 or 1)
    law: Law                   # evaluator-only target law
    cue: int | None            # None for the correct probe, flipped cue otherwise
    flipped: bool
    branch: str
    marginal: np.ndarray | None  # (5,3) training-only marginal (correct probes only)
    observations: np.ndarray   # (512,4,12,16) uint8 query images as presented
    probabilities: np.ndarray  # (512,5,3) float32 learner forecasts (predict_all)
    truth: np.ndarray          # (512,5,3) uint8 evaluator counterfactual survival
    affected: np.ndarray       # (512,) bool = mask_affected(cases, cue)
    valid: np.ndarray          # (512,) bool = mask_valid(cases, branch)
    support_observations: np.ndarray  # (32,4,12,16) uint8
    support_actions: np.ndarray       # (32,) uint8 performed actions
    support_outcomes: np.ndarray      # (32,3) uint8 performed-action survival
    support_fingerprint: str
    model_sha256: str
    recorded_metrics: dict

    @property
    def key(self):
        return (self.seed, self.phase, self.trace)

    def support(self):
        return Experience(self.support_observations, self.support_actions, self.support_outcomes)


@dataclass(frozen=True, eq=False)
class Target:
    slot: int
    law: Law
    marginal: np.ndarray                         # (5,3) float64 (success+.5)/(count+1)
    correct: tuple[EndpointEval, ...]            # indexed by support replicate
    flips: dict[int, tuple[EndpointEval, ...]]   # cue -> per-replicate flipped probes

    @property
    def stage(self):
        return None if self.slot < 2 else self.slot - 1


@dataclass(frozen=True, eq=False)
class Endpoint:
    run_dir: Path
    seed: int
    phase: str
    introduced_cue: int | None
    phase_spec: dict
    config: dict
    targets: tuple[Target, ...]
    final_model_sha256: str
    marginals: dict = field(repr=False)

    @property
    def evaluations(self):
        items = [e for t in self.targets for e in (*t.correct, *(x for v in t.flips.values() for x in v))]
        return sorted(items, key=lambda item: item.trace)


def _expected_phase(seed, config, phase):
    specs = {spec["name"]: spec for spec in qualification_phases(seed, config)}
    return json.loads(json.dumps(specs[phase]))  # tuples -> lists, as in result.json


def load_endpoint(run_dir=DEFAULT_RUN, seed=18101, phase="interleaved", verify=True):
    """Load the endpoint probe traces of one qualification phase.

    Uses result.json ``end_probes`` trace indices and the evaluations.json /
    evaluations.npz ledger. With ``verify`` the phase/artefact file hashes,
    per-array fingerprints, support fingerprints, training-only marginals,
    terminal-model identity and recorded per-call metrics are all checked.
    """
    run_dir = Path(run_dir).resolve()
    config = run_config(run_dir)
    require(config["kind"] == "qualification", "loader covers the qualification run only")
    require(phase in PHASES and seed in config["seeds"], "unknown seed or phase")
    job_dir = run_dir / "jobs" / f"{ARM}_{seed}"
    folder = job_dir / phase
    record = read_json(folder / "result.json")
    if verify:
        job = read_json(job_dir / "result.json")
        require(sha(folder / "result.json") == job["phase_hashes"][f"{phase}/result.json"],
                "phase result.json differs from the job ledger")
        for name in ("evaluations.json", "evaluations.npz"):
            require(sha(folder / name) == record["artifact_hashes"][name], f"{name} changed")
        require(record["seed"] == seed and record["phase"] == phase and record["arm"] == ARM,
                "phase identity differs")
        require(record["phase_spec"] == _expected_phase(seed, config, phase),
                "phase specification differs from design.qualification_phases")
    payload = read_json(folder / "evaluations.json")
    traces = payload["records"]
    require(payload["seed"] == seed and all(t["index"] == i for i, t in enumerate(traces)),
            "evaluation ledger identity differs")
    require([item["law"] for item in record["end_probes"]] == record["phase_spec"]["targets"],
            "end-probe target coverage differs")
    targets = []
    with np.load(folder / "evaluations.npz", allow_pickle=False) as archive:

        def evaluation(trace_id, slot, replicate):
            trace = traces[trace_id]
            arrays = {}
            for name, expected in trace["array_sha256"].items():
                value = archive[f"e{trace_id}_{name}"]
                if verify:
                    require(fingerprint(value) == expected, f"array e{trace_id}_{name} changed")
                arrays[name] = value
            require(trace["support_fingerprint"] is not None, "endpoint probe lacks a support set")
            if verify:
                require(fingerprint(arrays["support_observations"], arrays["support_actions"],
                                    arrays["support_outcomes"]) == trace["support_fingerprint"],
                        "support fingerprint differs")
            marginal = None if trace["marginal"] is None else _frozen(np.asarray(trace["marginal"], dtype=np.float64))
            return EndpointEval(
                seed=seed, phase=phase, trace=trace_id, slot=slot, replicate=replicate,
                law=law_from_record(trace["law"]), cue=trace["cue"], flipped=trace["flipped"],
                branch=trace["branch"], marginal=marginal,
                **{name: _frozen(arrays[name]) for name in (
                    "observations", "probabilities", "truth", "affected", "valid",
                    "support_observations", "support_actions", "support_outcomes")},
                support_fingerprint=trace["support_fingerprint"], model_sha256=trace["model_sha256"],
                recorded_metrics=trace["metrics"])

        for slot, item in enumerate(record["end_probes"]):
            law = law_from_record(item["law"])
            totals = record["marginals"][law_key(law)]
            marginal = _frozen(locked_marginal(totals["count"], totals["success"]))
            correct = tuple(evaluation(t, slot, rep) for rep, t in enumerate(item["correct"]))
            flips = {int(cue): tuple(evaluation(t, slot, rep) for rep, t in enumerate(ids))
                     for cue, ids in item["flips"].items()}
            require(len(correct) == config["support_replicates"]
                    and set(flips) == set(law.active)
                    and all(len(v) == config["support_replicates"] for v in flips.values()),
                    "endpoint replicate/cue coverage differs")
            for rep, probe in enumerate(correct):
                require(probe.law == law and probe.cue is None and not probe.flipped
                        and np.array_equal(probe.marginal, marginal),
                        "correct probe is not the declared full-support probe with training-only marginal")
                for cue, values in flips.items():
                    flip = values[rep]
                    require(flip.law == law and flip.cue == cue and flip.flipped and flip.marginal is None
                            and flip.support_fingerprint == probe.support_fingerprint
                            and flip.model_sha256 == probe.model_sha256,
                            "flipped probe is not paired with its correct probe")
            targets.append(Target(slot, law, marginal, correct, flips))
    endpoint = Endpoint(run_dir, seed, phase, record["phase_spec"]["cue"], record["phase_spec"], config,
                        tuple(targets), record["final_model_sha256"], record["marginals"])
    if verify:
        require({e.model_sha256 for e in endpoint.evaluations} == {record["final_model_sha256"]},
                "endpoint probes do not all use the terminal model")
        for item in endpoint.evaluations:
            metrics = locked_score(item.probabilities, item.truth, item.affected, item.valid, item.marginal)
            require(set(metrics) == set(item.recorded_metrics)
                    and all(abs(metrics[k] - item.recorded_metrics[k]) < 1e-12 for k in metrics),
                    f"recorded metrics differ from raw arrays at trace {item.trace}")
    return endpoint


def load_all(run_dir=DEFAULT_RUN, seeds=None, phases=PHASES, verify=True):
    config = run_config(run_dir)
    seeds = config["seeds"] if seeds is None else seeds
    return {(seed, phase): load_endpoint(run_dir, seed, phase, verify)
            for seed in seeds for phase in phases}


# --------------------------------------------------------------- regeneration

@dataclass(frozen=True, eq=False)
class QueryCases:
    """Evaluator-owned 512-query panel for one target law (EVALUATOR-ONLY)."""

    seed: int
    law: Law
    cases: LawCases            # observations, base Cases, factors, signals, law
    truth: np.ndarray          # (512,5,3) AcquisitionWorld.counterfactuals(cases)
    realised: dict             # see realised_physics

    @property
    def factors(self):       # (n,3): source in {-1,1}, lossy, delayed
        return self.cases.factors

    @property
    def signals(self):       # (n,3) pulse-order bits of cues 0/1/2
        return self.cases.signals

    @property
    def reserves(self):
        return self.cases.base.reserves

    def observations(self, cue=None):
        """Presented query images; with ``cue`` the pulse order of that cue is flipped."""
        if cue is None:
            return self.cases.observations
        signals = self.cases.signals.copy()
        signals[:, cue] ^= 1
        return signal_images(self.cases.observations, signals)

    def affected(self, cue=None):
        return mask_affected(self.cases, cue)

    def valid(self, branch="novel"):
        return mask_valid(self.cases, branch)


@dataclass(frozen=True, eq=False)
class SupportCases:
    """Evaluator-owned 32-record support set (EVALUATOR-ONLY beyond obs/action/outcome)."""

    seed: int
    law: Law
    replicate: int
    data_seed: int             # trial_seed(seed, "acquisition_support", [replicate, False])
    cases: LawCases
    actions: np.ndarray        # performed actions (uint8)
    outcomes: np.ndarray       # physical survival of the performed actions (uint8)
    truth: np.ndarray          # (32,5,3) counterfactual survival of every action
    realised: dict

    def experience(self):
        return Experience(self.cases.observations, self.actions, self.outcomes)


@dataclass(frozen=True, eq=False)
class Regenerated:
    endpoint: Endpoint
    queries: dict              # slot -> QueryCases
    supports: dict             # (slot, replicate) -> SupportCases

    def cases_for(self, evaluation):
        """(QueryCases, SupportCases) behind one EndpointEval."""
        return self.queries[evaluation.slot], self.supports[evaluation.slot, evaluation.replicate]


def realised_physics(world, cases):
    """Law-effective per-case physics (EVALUATOR-ONLY; for verification).

    base_supplies: (n,12,2) the clipped noisy supply draw after the hidden-mode
    column swap (ConditionalWorld); supplies: after the law's supply-timing
    dependency (identical for every action, asserted); efficiency/delay:
    (n,5) per performed action index.
    """
    per_action = [world.physics(cases, np.full(len(cases.observations), action)) for action in range(5)]
    for item in per_action[1:]:
        require(np.array_equal(item.supplies, per_action[0].supplies),
                "law-effective supplies unexpectedly depend on the action")
    return dict(base_supplies=_frozen(cases.base.supplies), supplies=_frozen(per_action[0].supplies),
                efficiency=_frozen(np.stack([p.efficiency for p in per_action], axis=1)),
                delay=_frozen(np.stack([p.delay for p in per_action], axis=1)))


def _same(first, second, message):
    require(first.dtype == second.dtype and first.shape == second.shape
            and first.tobytes() == second.tobytes(), message)


def regenerate(endpoint, check=True):
    """Rebuild every evaluator-owned case object behind an endpoint's probes.

    Queries come from ``TraceEvaluator(seed, config).dataset(law)`` (the same
    cached Evaluator.dataset path, seed trial_seed(seed, "acquisition_query", 0))
    and supports from ``Evaluator.support(law, replicate)`` plus
    ``AcquisitionWorld.dataset(law, batch_size, trial_seed(seed,
    "acquisition_support", [replicate, False]))`` which is the dataset call made
    inside AcquisitionWorld.experience. With ``check`` it asserts byte identity
    of all saved observations (correct and flipped), exact counterfactual truth
    from a fresh AcquisitionWorld, identical masks, and that saved support
    outcomes equal the physical outcomes of the performed actions.
    """
    evaluator = TraceEvaluator(endpoint.seed, endpoint.config)
    world, independent = evaluator.world, AcquisitionWorld()
    size = endpoint.config["batch_size"]
    queries, supports = {}, {}
    for target in endpoint.targets:
        cases, truth = evaluator.dataset(target.law)
        query = QueryCases(endpoint.seed, target.law, cases, _frozen(truth), realised_physics(world, cases))
        queries[target.slot] = query
        if check:
            _same(independent.counterfactuals(cases), truth, "fresh counterfactuals differ from Evaluator truth")
        for rep, correct in enumerate(target.correct):
            data_seed = trial_seed(endpoint.seed, "acquisition_support", [rep, False])
            experience = evaluator.support(target.law, rep)
            support_cases = world.dataset(target.law, size, data_seed)
            physical = world.simulate(support_cases, experience.actions).survival
            support_truth = world.counterfactuals(support_cases)
            support = SupportCases(endpoint.seed, target.law, rep, data_seed, support_cases,
                                   _frozen(experience.actions), _frozen(physical), _frozen(support_truth),
                                   realised_physics(world, support_cases))
            supports[target.slot, rep] = support
            if not check:
                continue
            _same(correct.observations, cases.observations, "query observations differ")
            _same(correct.truth, truth, "saved truth differs from regenerated counterfactuals")
            _same(correct.affected, mask_affected(cases, None), "correct-probe affected mask differs")
            _same(correct.valid, mask_valid(cases, correct.branch), "valid mask differs")
            _same(experience.observations, support_cases.observations, "support dataset path differs")
            _same(correct.support_observations, support_cases.observations, "support observations differ")
            _same(correct.support_actions, experience.actions, "support actions differ")
            _same(correct.support_outcomes, experience.survival, "support outcomes differ from Evaluator.support")
            _same(correct.support_outcomes, physical, "support outcomes differ from physical outcomes")
            _same(support_truth[np.arange(size), experience.actions], physical,
                  "performed outcome differs from the counterfactual row")
            require(experience.fingerprint() == correct.support_fingerprint, "support fingerprint differs")
            for cue, flips in target.flips.items():
                flip = flips[rep]
                _same(flip.observations, query.observations(cue), f"flipped cue-{cue} observations differ")
                _same(flip.truth, truth, "flipped probe changed physical truth")
                _same(flip.affected, mask_affected(cases, cue), f"cue-{cue} affected mask differs")
                _same(flip.valid, mask_valid(cases, flip.branch), "flipped valid mask differs")
                for name in ("support_observations", "support_actions", "support_outcomes"):
                    _same(getattr(flip, name), getattr(correct, name), "flipped probe changed its support")
    return Regenerated(endpoint, queries, supports)


# ------------------------------------------------------------------- scoring

def locked_marginal(count, success):
    count, success = np.asarray(count), np.asarray(success)
    require(count.shape == (5,) and success.shape == (5, 3), "invalid training totals")
    return (success.astype(np.float64) + .5) / (count[:, None] + 1.)


def monotone(probabilities):
    """Horizon-wise running minimum, as ConditionalLearner.predict's cummin(dim=-1)."""
    return np.minimum.accumulate(np.asarray(probabilities), axis=-1)


def locked_score(prediction, truth, affected, valid, marginal=None):
    """Mirror of ``score`` in scripts/summarize_representation_learning.py.

    All-case Brier averages the squared float64 error over all 5 actions and
    3 horizons per query, then over queries. Forecasts must be float, within
    float32 eps of [0,1] and non-increasing over horizons (within eps).
    """
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    n, eps = len(truth), np.finfo(np.float32).eps
    require(n > 0 and prediction.shape == truth.shape == (n, 5, 3)
            and prediction.dtype.kind == "f" and np.isfinite(prediction).all()
            and np.all((prediction >= -eps) & (prediction <= 1 + eps))
            and np.all(np.diff(prediction, axis=-1) <= eps), "invalid survival forecast")
    require(np.isin(truth, (0, 1)).all()
            and np.all(np.diff(truth.astype(np.int16), axis=-1) <= 0), "invalid survival truth")
    for mask in (affected, valid):
        require(mask.dtype == np.bool_ and mask.shape == (n,) and mask.any(), "invalid scoring mask")
    error = np.square(prediction.astype(np.float64) - truth.astype(np.float64)).mean(axis=(1, 2))
    survived = truth[np.arange(n), prediction[:, :, -1].argmax(axis=1), -1]
    result = dict(brier=float(error.mean()), focus_brier=float(error[affected].mean()),
                  survival=float(survived.mean()), focus_survival=float(survived[affected].mean()),
                  valid_brier=float(error[valid].mean()), valid_survival=float(survived[valid].mean()),
                  no_transfer=float(truth[:, 0, -1].mean()),
                  clairvoyant_upper=float(truth[:, :, -1].max(axis=1).mean()))
    if marginal is not None:
        marginal = np.asarray(marginal, dtype=np.float64)
        require(marginal.shape == (5, 3) and np.isfinite(marginal).all()
                and np.all((marginal >= 0) & (marginal <= 1)), "invalid marginal forecast")
        result["marginal_brier"] = float(np.square(marginal[None] - truth).mean())
    return result


@lru_cache(maxsize=None)
def _draws(n):
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, (BOOTSTRAP_RESAMPLES, n))


def seed_estimate(values, seeds):
    """Mirror of the locked ``estimate``: seed mean and percentile bootstrap."""
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (len(seeds),) and len(seeds) > 0 and len(set(seeds)) == len(seeds)
            and np.isfinite(values).all(), "invalid seed-balanced summary")
    lower, upper = np.quantile(values[_draws(len(seeds))].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), lower=float(lower), upper=float(upper), seeds=list(seeds),
                values=values.tolist(), n=len(seeds), negative=int((values < 0).sum()))


def _source(probabilities):
    if probabilities is None:
        return lambda endpoint, evaluation: evaluation.probabilities
    if callable(probabilities):
        return probabilities
    return lambda endpoint, evaluation: probabilities[evaluation.key]


def endpoint_targets(endpoint, probabilities=None):
    """Per-target endpoint summary exactly like the locked ``endpoint_targets``.

    ``probabilities``: None (saved learner forecasts), a callable
    ``f(endpoint, evaluation) -> (512,5,3) float array`` or a mapping keyed by
    ``(seed, phase, trace)``. Truth, masks, marginals, support pairing and
    replicate averaging are always taken from the saved evaluator record.
    """
    source = _source(probabilities)
    result = []
    for target in endpoint.targets:
        predictions, gains, replicates = [], {}, []
        for rep, correct in enumerate(target.correct):
            forecast = source(endpoint, correct)
            metrics = locked_score(forecast, correct.truth, correct.affected, correct.valid, target.marginal)
            predictions.append(metrics)
            row = dict(replicate=rep, trace=correct.trace, metrics=metrics, cue_benefits={})
            for cue, flips in target.flips.items():
                flip = flips[rep]
                require(np.array_equal(correct.truth, flip.truth) and np.array_equal(correct.valid, flip.valid),
                        "cue probe changed physical target")
                fields = (flip.truth, flip.affected, flip.valid)
                baseline = locked_score(forecast, *fields)["focus_brier"]
                flipped = locked_score(source(endpoint, flip), *fields)["focus_brier"]
                gains.setdefault(str(cue), []).append(flipped - baseline)
                row["cue_benefits"][str(cue)] = dict(trace=flip.trace, correct_focus_brier=baseline,
                                                     flipped_focus_brier=flipped, benefit=flipped - baseline)
            replicates.append(row)
        averaged = {key: float(np.mean([value[key] for value in predictions])) for key in predictions[0]}
        result.append(dict(slot=target.slot, law=law_record(target.law), metrics=averaged,
                           marginal_gain=averaged["marginal_brier"] - averaged["brier"],
                           cue_benefits={cue: float(np.mean(value)) for cue, value in gains.items()},
                           replicates=replicates))
    return result


def score_cells(endpoints, probabilities=None, seeds=None):
    """Compute the 31 qualification cells like the locked ``qualification``.

    ``endpoints`` maps (seed, phase) -> Endpoint for all four phases of every
    seed. Returns per-seed values, seed means, descriptive bootstrap intervals
    (same RNG as the locked scorer) and per-phase endpoint summaries.
    """
    if seeds is None:
        seeds = sorted({seed for seed, _ in endpoints})
    rows = []
    for seed in seeds:
        for phase in PHASES:
            endpoint = endpoints[seed, phase]
            rows.append(dict(seed=seed, phase=phase, introduced_cue=endpoint.introduced_cue,
                             end_targets=endpoint_targets(endpoint, probabilities)))
    cells = []

    def cell(name, selected, value, minimum=None, maximum=None, **meta):
        require(selected, "empty required qualification group: " + name)
        stat = seed_estimate([value(row) for row in selected], [row["seed"] for row in selected])
        passed = ((minimum is None or stat["mean"] >= minimum)
                  and (maximum is None or stat["mean"] <= maximum))
        cells.append(dict(name=name, **meta, per_seed=dict(zip(stat["seeds"], stat["values"])),
                          metric=stat, minimum=minimum, maximum=maximum, passed=passed))

    fresh = [row for row in rows if row["phase"].startswith("fresh_")]
    groups = [(f"fresh/stage_{s}", [r for r in fresh if r["phase"] == f"fresh_{s}"], dict(stage=s, cue=None))
              for s in (1, 2, 3)]
    groups += [(f"fresh/cue_{c}", [r for r in fresh if r["introduced_cue"] == c], dict(stage=None, cue=c))
               for c in sorted({r["introduced_cue"] for r in fresh})]
    for name, selected, meta in groups:
        cell(name + "/marginal_gain", selected, lambda r: r["end_targets"][0]["marginal_gain"],
             minimum=.02, kind="marginal_gain", phase="fresh", slot=0, **meta)
        cell(name + "/cue_benefit", selected,
             lambda r: r["end_targets"][0]["cue_benefits"][str(r["introduced_cue"])],
             minimum=.002, kind="cue_benefit", phase="fresh", slot=0, **meta)
    selected = [row for row in rows if row["phase"] == "interleaved"]
    for slot in range(5):
        stage = None if slot < 2 else slot - 1
        cell(f"interleaved/slot_{slot}/marginal_gain", selected,
             lambda r: r["end_targets"][slot]["marginal_gain"], minimum=.02,
             kind="marginal_gain", phase="interleaved", slot=slot, stage=stage, cue=None)
        cell(f"interleaved/slot_{slot}/absolute_brier", selected,
             lambda r: r["end_targets"][slot]["metrics"]["brier"], maximum=.12,
             kind="absolute_brier", phase="interleaved", slot=slot, stage=stage, cue=None)
        if slot >= 2:
            for cue in (0, 1, 2):
                subset = [r for r in selected if str(cue) in r["end_targets"][slot]["cue_benefits"]]
                if subset:
                    cell(f"interleaved/stage_{slot - 1}/cue_{cue}/cue_benefit", subset,
                         lambda r: r["end_targets"][slot]["cue_benefits"][str(cue)], minimum=.002,
                         kind="cue_benefit", phase="interleaved", slot=slot, stage=stage, cue=cue)
    require(len(cells) == 31, f"expected 31 qualification cells, got {len(cells)}")
    return dict(label=LABEL, seeds=list(seeds), cells=cells,
                seed_means={c["name"]: c["metric"]["mean"] for c in cells},
                raw_passed_recomputed=all(c["passed"] for c in cells),
                per_phase={f"{r['seed']}/{r['phase']}": r["end_targets"] for r in rows})


def compare_with_summary(result, summary_path=DEFAULT_SUMMARY):
    """Max absolute differences between a learner rescoring and summary.json."""
    summary = read_json(summary_path)
    locked = {g["name"]: g for g in summary["groups"]}
    require([c["name"] for c in result["cells"]] == [g["name"] for g in summary["groups"]],
            "cell names/order differ from summary.json")
    diffs = dict(per_seed_values=0., seed_means=0., bootstrap_bounds=0., end_target_metrics=0.)
    flags = []
    for item in result["cells"]:
        old, new = locked[item["name"]], item["metric"]
        require(old["metric"]["seeds"] == new["seeds"] and old["metric"]["n"] == new["n"]
                and old["metric"]["negative"] == new["negative"], "cell seed coverage differs: " + item["name"])
        diffs["per_seed_values"] = max(diffs["per_seed_values"], float(np.max(np.abs(
            np.asarray(old["metric"]["values"]) - np.asarray(new["values"])))))
        diffs["seed_means"] = max(diffs["seed_means"], abs(old["metric"]["mean"] - new["mean"]))
        diffs["bootstrap_bounds"] = max(diffs["bootstrap_bounds"], abs(old["metric"]["lower"] - new["lower"]),
                                        abs(old["metric"]["upper"] - new["upper"]))
        flags.append(old["passed"] == item["passed"] and old["minimum"] == item["minimum"]
                     and old["maximum"] == item["maximum"])
    for row in summary["per_phase"]:
        mine = result["per_phase"][f"{row['seed']}/{row['phase']}"]
        require(len(mine) == len(row["end_targets"]), "per-phase target coverage differs")
        for new, old in zip(mine, row["end_targets"]):
            require(new["law"] == old["law"] and set(new["cue_benefits"]) == set(old["cue_benefits"])
                    and set(new["metrics"]) == set(old["metrics"]), "per-phase target identity differs")
            values = [abs(new["metrics"][k] - old["metrics"][k]) for k in old["metrics"]]
            values += [abs(new["marginal_gain"] - old["marginal_gain"])]
            values += [abs(new["cue_benefits"][k] - old["cue_benefits"][k]) for k in old["cue_benefits"]]
            diffs["end_target_metrics"] = max(diffs["end_target_metrics"], *values)
    return dict(max_abs_diff=max(diffs.values()), components=diffs, pass_flags_match=all(flags),
                decision_locked=summary["decision"], raw_passed_locked=summary["raw_passed"],
                raw_passed_recomputed=result["raw_passed_recomputed"],
                failed_cells=[c["name"] for c in result["cells"] if not c["passed"]])


# --------------------------------------------------------- structural evidence

def structural_evidence(endpoints, regenerated, run_dir=DEFAULT_RUN):
    """Evidence for the two structural observations (per seed).

    (1) Hidden-mode imbalance: baseA = Law(seed%2) is the only reference law
        with its mode; baseB and stage_law(seed, 1..3) share mode 1-seed%2.
    (2) The novelty law that first adds cue 1 (arrival delay) differs from its
        predecessor (previous stage law, or baseB at stage 1; also the
        immediately preceding interleaved block) only by that dependency.
    """
    run_dir = Path(run_dir).resolve()
    config = run_config(run_dir)
    mode_rows, delay_rows = [], []
    for seed in config["seeds"]:
        laws = reference_laws(seed)
        spec = endpoints[seed, "interleaved"].phase_spec
        blocks = [law_from_record(b["law"]) for b in spec["blocks"]]
        sizes = [b["size"] for b in spec["blocks"]]
        with np.load(run_dir / "jobs" / f"{ARM}_{seed}" / "interleaved" / "training.npz",
                     allow_pickle=False) as archive:
            law_index = archive["law_index"]
        base_a_mode = laws[0].mode
        share = sum(s for law, s in zip(blocks, sizes) if law.mode == base_a_mode) / sum(sizes)
        fresh_modes = [law_from_record(endpoints[seed, f"fresh_{s}"].phase_spec["targets"][0]).mode
                       for s in (1, 2, 3)]
        regen = regenerated[seed, "interleaved"]
        qa, qb = regen.queries[0], regen.queries[1]
        mode_rows.append(dict(
            seed=seed, reference_modes=[law.mode for law in laws],
            base_a_mode=base_a_mode, others_mode=1 - seed % 2,
            base_a_unique_mode=all(law.mode != base_a_mode for law in laws[1:]),
            stage_law_modes=[stage_law(seed, s).mode for s in (1, 2, 3)],
            fresh_target_modes=fresh_modes,
            interleaved_arrival_share_of_base_a_mode=share,
            training_packet_share_law_index_0=float(np.mean(law_index == 0)),
            query_observations_identical_across_modes=bool(np.array_equal(
                qa.cases.observations, qb.cases.observations)),
            base_supplies_are_column_swaps=bool(np.array_equal(
                qa.realised["base_supplies"], qb.realised["base_supplies"][..., ::-1])),
            truth_differs_base_a_vs_base_b=int((qa.truth != qb.truth).any(axis=(1, 2)).sum())))
        stage = order(seed).index(1) + 1
        law = stage_law(seed, stage)
        previous = stage_law(seed, stage - 1) if stage > 1 else laws[1]
        slot, prev_slot = stage + 1, stage
        cycle_preceding = [blocks[i - 1] for i in range(1, len(blocks)) if blocks[i] == law]
        query, prior = regen.queries[slot], regen.queries[prev_slot]
        delayed = query.factors[:, 2] == 1
        differs = (query.truth != prior.truth).any(axis=(1, 2))
        delay_changed = (query.realised["delay"] != prior.realised["delay"]).any(axis=1)
        delay_rows.append(dict(
            seed=seed, order=list(order(seed)), delay_stage=stage, law=law_record(law),
            predecessor=law_record(previous),
            same_mode=law.mode == previous.mode,
            same_revised_noise=(law.revised, law.noise) == (previous.revised, previous.noise),
            active_difference=sorted(set(law.active) ^ set(previous.active)),
            predecessor_is_immediately_preceding_block=bool(cycle_preceding)
            and all(item == previous for item in cycle_preceding),
            observations_identical=bool(np.array_equal(query.cases.observations, prior.cases.observations)),
            supplies_identical=bool(np.array_equal(query.realised["supplies"], prior.realised["supplies"])),
            efficiency_identical=bool(np.array_equal(query.realised["efficiency"], prior.realised["efficiency"])),
            delay_changes_only_on_delayed=bool(not delay_changed[~delayed].any()),
            truth_differs_only_on_delayed=bool(not differs[~delayed].any()),
            delayed_queries=int(delayed.sum()), truth_differs_queries=int(differs.sum()),
            truth_differs_cells=int((query.truth != prior.truth).sum())))
    first = all(r["base_a_unique_mode"] and r["stage_law_modes"] == [r["others_mode"]] * 3
                and r["fresh_target_modes"] == [r["others_mode"]] * 3
                and r["interleaved_arrival_share_of_base_a_mode"] == .2
                and r["training_packet_share_law_index_0"] == .2 for r in mode_rows)
    second = all(r["same_mode"] and r["same_revised_noise"] and r["active_difference"] == [1]
                 and r["predecessor_is_immediately_preceding_block"] and r["supplies_identical"]
                 and r["efficiency_identical"] and r["delay_changes_only_on_delayed"]
                 and r["truth_differs_only_on_delayed"] for r in delay_rows)
    return dict(label=LABEL, mode_claim_confirmed=first, delay_claim_confirmed=second,
                mode_rows=mode_rows, delay_rows=delay_rows)


# ---------------------------------------------------------------------- CLI

def verify(run_dir=DEFAULT_RUN, summary_path=DEFAULT_SUMMARY):
    endpoints = load_all(run_dir)
    regenerated = {key: regenerate(endpoint) for key, endpoint in endpoints.items()}
    result = score_cells(endpoints)
    comparison = compare_with_summary(result, summary_path)
    structure = structural_evidence(endpoints, regenerated, run_dir)
    counts = dict(phases=len(endpoints), endpoint_evaluations=sum(len(e.evaluations) for e in endpoints.values()),
                  regenerated_query_panels=sum(len(r.queries) for r in regenerated.values()),
                  regenerated_supports=sum(len(r.supports) for r in regenerated.values()))
    return dict(label=LABEL, run=str(Path(run_dir).resolve().relative_to(ROOT).as_posix()),
                summary=str(Path(summary_path).resolve().relative_to(ROOT).as_posix()),
                source_lock=source_lock_check(run_dir), counts=counts,
                regeneration="all byte/exactness assertions passed", comparison=comparison,
                cells=[dict(name=c["name"], kind=c["kind"], phase=c["phase"], slot=c["slot"],
                            stage=c["stage"], cue=c["cue"], per_seed=c["per_seed"],
                            mean=c["metric"]["mean"], lower=c["metric"]["lower"], upper=c["metric"]["upper"],
                            minimum=c["minimum"], maximum=c["maximum"], passed=c["passed"])
                       for c in result["cells"]],
                structure=structure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--verify", action="store_true", help="load, regenerate, rescore and compare")
    parser.add_argument("--output", type=Path, help="NEW json file under reports/information_audit/")
    args = parser.parse_args()
    if not args.verify:
        parser.error("nothing to do; pass --verify")
    report = verify(args.run, args.summary)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        require(output.is_relative_to(ROOT / "reports" / "information_audit"),
                "outputs belong under reports/information_audit/")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:  # never overwrite
            handle.write(text)
    print(json.dumps(dict(label=LABEL, max_abs_diff=report["comparison"]["max_abs_diff"],
                          components=report["comparison"]["components"],
                          pass_flags_match=report["comparison"]["pass_flags_match"],
                          failed_cells=report["comparison"]["failed_cells"],
                          source_lock_all_match=all(v["matches"] for v in report["source_lock"].values()),
                          mode_claim_confirmed=report["structure"]["mode_claim_confirmed"],
                          delay_claim_confirmed=report["structure"]["delay_claim_confirmed"],
                          counts=report["counts"]), indent=2))


if __name__ == "__main__":
    main()
