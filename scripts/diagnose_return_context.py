"""Retrospective, read-only matched-query diagnosis of the completed Adam pilot.

This is an explanatory analysis of previously inspected results, not a new
prospective method comparison. No checkpoint is trained or written back.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
import torch

from verify_evidence_consolidation_archive import verify as verify_archive
from acp_cl.acquisition.study import Evaluator, load
from acp_cl.evidence_consolidation.schedule import experiences, schedule
from acp_cl.evidence_consolidation.study import source_manifest, validate_config
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest
from acp_cl.persistence.world import concatenate
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import write_json


ROOT = Path(__file__).resolve().parents[1]
RETURNS = (6, 11, 16)
MODELS = ("conditional", "recurrent")
SEEDS = tuple(range(12001, 12007))
BOOTSTRAP_SEED = 27192026
BOOTSTRAP_SAMPLES = 20000
PREDICTION_ROUNDOFF = 8*np.finfo(np.float32).eps
METRICS = ("brier", "survival")
SUPPORTS = ("actual_start", "observed_8", "observed_16", "observed_32",
            "actual_end", "fresh_target")
WEIGHTS = ("start_draft", "start_periodic", "start_sustained", "prefix_draft",
           "previous_return_draft", "end_draft")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_entry(path):
    content = Path(path).read_bytes()
    return dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def check_files(directory, expected):
    directory = Path(directory).resolve()
    for name, entry in expected.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or file_entry(path) != entry:
            raise ValueError(f"input file changed or escaped its directory: {name}")


def array_hash(*arrays):
    result = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        result.update(str(array.dtype).encode())
        result.update(json.dumps(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def value_hash(value):
    """Hash tensor optimizer state without pickle storage-address dependence."""
    result = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            result.update(b"tensor")
            result.update(array_hash(item.detach().cpu().numpy()).encode())
        elif isinstance(item, np.ndarray):
            result.update(b"array")
            result.update(array_hash(item).encode())
        elif isinstance(item, dict):
            result.update(b"dict")
            for key in sorted(item, key=repr):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for part in item:
                update(part)
        else:
            result.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        result.update(b";")

    update(value)
    return result.hexdigest()


def bundle_signature(bundle):
    predictors = dict(bundle.predictors())
    if bundle.proposal is not None:
        predictors["proposal"] = bundle.proposal
    models = {}
    for name, learner in predictors.items():
        models[name] = dict(
            weights=state_hash(learner.model.state_dict()),
            training=[module.training for module in learner.model.modules()],
            requires_grad=[parameter.requires_grad for parameter in learner.model.parameters()],
            gradients=[None if parameter.grad is None else value_hash(parameter.grad)
                       for parameter in learner.model.parameters()],
            history=None if learner.history is None else learner.history.fingerprint(),
            settings=learner.settings,
        )
    return value_hash(dict(state=bundle.state(), models=models,
        optimizer=bundle.draft.optimizer.state_dict(), memory=memory_state(bundle.draft.memory)))


def replace_history(history, observed, records):
    """Keep the newest old records and append the first new observed records."""
    if history is None or len(history) != len(observed):
        raise ValueError("matched nonempty history and observed packet required")
    if type(records) is not int or not 0 <= records <= len(history):
        raise ValueError("replacement count is outside the history window")
    if not records:
        return history
    return concatenate(history, observed.take(slice(0, records))).take(slice(-len(history), None))


def score_predictions(probabilities, truth):
    probabilities, truth = np.asarray(probabilities), np.asarray(truth)
    if probabilities.shape != truth.shape or probabilities.ndim != 3 or probabilities.shape[1:] != (5, 3):
        raise ValueError("expected matched query x five actions x three horizons")
    if not np.isfinite(probabilities).all() or not np.isfinite(truth).all():
        raise ValueError("nonfinite prediction or outcome")
    prediction_min, prediction_max = float(probabilities.min()), float(probabilities.max())
    truth_min, truth_max = float(truth.min()), float(truth.max())
    if truth_min < 0 or truth_max > 1:
        raise ValueError(f"outcome is outside [0, 1]: min={truth_min:.17g}, max={truth_max:.17g}")
    if prediction_min < -PREDICTION_ROUNDOFF or prediction_max > 1+PREDICTION_ROUNDOFF:
        raise ValueError("prediction exceeds float32 roundoff tolerance: "
                         f"min={prediction_min:.17g}, max={prediction_max:.17g}, "
                         f"tolerance={PREDICTION_ROUNDOFF:.17g}")
    # A float32 weighted mixture can overshoot an endpoint by a few ULPs.
    # Validate that narrow numerical range without clipping or changing scores.
    actions = probabilities[:, :, -1].argmax(axis=1)
    return dict(brier=float(np.mean((probabilities-truth)**2)),
                survival=float(np.mean(truth[np.arange(len(actions)), actions, -1])))


def best_single_head(heads, truth):
    """Query-selected explanatory reference; mixtures can outperform every head."""
    if heads.ndim != 4 or heads.shape[0] != truth.shape[0] or heads.shape[2:] != truth.shape[1:]:
        raise ValueError("head predictions do not match query outcomes")
    scores = [score_predictions(heads[:, index], truth) for index in range(heads.shape[1])]
    selected = int(np.argmin([score["brier"] for score in scores]))
    return dict(selected_head=selected, **scores[selected], per_head=scores,
                query_selected=True, deployable=False,
                interpretation="Best single head on these evaluation outcomes; not a lower bound on mixture error.")


def probe(learner, observations, truth, support, include_heads=False):
    """Restore train/eval flags and reject parameter mutation during prediction."""
    before = state_hash(learner.model.state_dict())
    modes = [(module, module.training) for module in learner.model.modules()]
    support_before = support.fingerprint()
    try:
        probabilities, _, heads = learner.predict(observations, support)
        scores = score_predictions(probabilities, truth)
        scores.update(prediction_min=float(probabilities.min()), prediction_max=float(probabilities.max()),
                      prediction_range_tolerance=float(PREDICTION_ROUNDOFF))
        reference = best_single_head(heads, truth) if include_heads else None
    finally:
        for module, training in modes:
            module.training = training
    if state_hash(learner.model.state_dict()) != before or support.fingerprint() != support_before:
        raise AssertionError("a diagnostic prediction changed weights or support")
    return scores, reference


def paired_summary(values, seeds=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty seed values required")
    if seeds is not None and (len(seeds) != len(values) or len(set(seeds)) != len(seeds)):
        raise ValueError("one value per independent seed required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    low, high = np.quantile(values[sample].mean(axis=1), [.025, .975])
    return dict(n=len(values), mean=float(values.mean()), lower=float(low), upper=float(high),
                positive=int((values > 0).sum()), negative=int((values < 0).sum()),
                zero=int((values == 0).sum()), differences=values.tolist(),
                seeds=None if seeds is None else list(seeds))


def build_contrasts(rows):
    """Average support replicates, then returns within seed, before bootstrapping."""
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["model"], row["seed"], row["block"], row["weight"], row["support"])
        unique = (*key, row["replicate"])
        if unique in seen:
            raise ValueError("duplicate diagnostic row")
        seen.add(unique)
        groups[key].append(row)
    block_means = []
    lookup = {}
    for key, items in sorted(groups.items()):
        values = {metric: float(np.mean([row[metric] for row in items])) for metric in METRICS}
        lookup[key] = values
        block_means.append(dict(zip(("model", "seed", "block", "weight", "support"), key),
                                support_replicates=len(items), **values))
    points = sorted({key[:3] for key in lookup})
    block_contrasts, block_decomposition = [], []
    definitions = (
        ("sustained_minus_start_draft", "start_sustained", "start_draft"),
        ("periodic_minus_start_draft", "start_periodic", "start_draft"),
        ("start_draft_minus_prefix", "start_draft", "prefix_draft"),
        ("start_draft_minus_previous_return", "start_draft", "previous_return_draft"),
        ("sustained_minus_prefix", "start_sustained", "prefix_draft"),
        ("sustained_minus_previous_return", "start_sustained", "previous_return_draft"),
        ("end_draft_minus_start_draft", "end_draft", "start_draft"),
        ("end_draft_minus_prefix", "end_draft", "prefix_draft"),
        ("end_draft_minus_previous_return", "end_draft", "previous_return_draft"),
    )
    for model, seed, block in points:
        def get(weight, support):
            return lookup.get((model, seed, block, weight, support))

        def emit(name, left, right):
            if left is not None and right is not None:
                block_contrasts.append(dict(model=model, seed=seed, block=block, contrast=name,
                    **{metric: left[metric]-right[metric] for metric in METRICS}))

        for support in SUPPORTS:
            for name, left, right in definitions:
                emit(f"{name}.{support}", get(left, support), get(right, support))
        for target in ("observed_8", "observed_16", "observed_32", "fresh_target"):
            for weight in WEIGHTS:
                emit(f"actual_start_minus_{target}.{weight}", get(weight, "actual_start"), get(weight, target))
            if target not in ("observed_32", "fresh_target"):
                continue
            actual_s, actual_d = get("start_sustained", "actual_start"), get("start_draft", "actual_start")
            target_s, target_d = get("start_sustained", target), get("start_draft", target)
            if all(value is not None for value in (actual_s, actual_d, target_s, target_d)):
                for metric in METRICS:
                    actual_gap = actual_s[metric]-actual_d[metric]
                    target_gap = target_s[metric]-target_d[metric]
                    sustained_context = actual_s[metric]-target_s[metric]
                    draft_context = actual_d[metric]-target_d[metric]
                    interaction = sustained_context-draft_context
                    residual = actual_gap-target_gap-interaction
                    if abs(residual) > 1e-12:
                        raise AssertionError("matched decomposition failed its exact identity")
                    block_decomposition.append(dict(model=model, seed=seed, block=block, target=target,
                        metric=metric, actual_gap=actual_gap, target_gap=target_gap,
                        sustained_context_penalty=sustained_context, draft_context_penalty=draft_context,
                        context_interaction=interaction, identity_residual=residual))

    def collapse(items, categories, values):
        grouped = defaultdict(list)
        for item in items:
            grouped[tuple(item[k] for k in categories)].append(item)
        return [dict(zip(categories, key), returns=len(part),
                     **{field: float(np.mean([item[field] for item in part])) for field in values})
                for key, part in sorted(grouped.items())]

    seed_means = collapse(block_means, ("model", "seed", "weight", "support"), METRICS)
    seed_contrasts = collapse(block_contrasts, ("model", "seed", "contrast"), METRICS)
    components = ("actual_gap", "target_gap", "sustained_context_penalty", "draft_context_penalty",
                  "context_interaction", "identity_residual")
    seed_decomposition = collapse(block_decomposition, ("model", "seed", "target", "metric"), components)
    contrasts, decomposition = {}, {}
    for model in sorted({row["model"] for row in rows}):
        contrasts[model], decomposition[model] = {}, {}
        for name in sorted({row["contrast"] for row in seed_contrasts if row["model"] == model}):
            selected = sorted([row for row in seed_contrasts if row["model"] == model and row["contrast"] == name],
                              key=lambda row: row["seed"])
            contrasts[model][name] = {metric: paired_summary([row[metric] for row in selected],
                                                           [row["seed"] for row in selected]) for metric in METRICS}
        for target in ("observed_32", "fresh_target"):
            decomposition[model][target] = {}
            for metric in METRICS:
                selected = sorted([row for row in seed_decomposition if row["model"] == model
                                   and row["target"] == target and row["metric"] == metric], key=lambda row: row["seed"])
                if selected:
                    decomposition[model][target][metric] = {
                        component: paired_summary([row[component] for row in selected], [row["seed"] for row in selected])
                        for component in components}
    return dict(block_means=block_means, seed_means=seed_means, block_contrasts=block_contrasts,
                seed_contrasts=seed_contrasts, contrasts=contrasts, block_decomposition=block_decomposition,
                seed_decomposition=seed_decomposition, decomposition=decomposition)


def snapshot_code(output, files, metadata):
    """Write or validate an immutable lock before any scientific query scoring."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = {name: file_entry(path) for name, path in sorted(files.items())}
    expected = dict(retrospective=True, files=entries, metadata=metadata)
    path = output/"analysis_lock.json"
    if path.exists():
        saved = read_json(path)
        if {key: saved[key] for key in expected} != expected:
            raise ValueError("analysis, protocol, runtime or input changed after lock")
        with zipfile.ZipFile(output/"analysis_at_lock.zip") as archive:
            if archive.testzip() is not None or set(archive.namelist()) != set(entries):
                raise ValueError("analysis snapshot file set or CRC changed")
            for name, entry in entries.items():
                value = archive.read(name)
                if len(value) != entry["bytes"] or hashlib.sha256(value).hexdigest() != entry["sha256"]:
                    raise ValueError("analysis snapshot changed after lock")
        return saved
    if any(output.iterdir()):
        raise ValueError("new diagnostic output directory must be empty")
    locked = dict(**expected, locked_utc=datetime.now(timezone.utc).isoformat())
    with zipfile.ZipFile(output/"analysis_at_lock.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, name)
    write_json(path, locked)
    return locked


def prepare(input_directory, archive_directory, output, protocol):
    directory, archive_directory, output = (Path(path).resolve() for path in (input_directory, archive_directory, output))
    if output.is_relative_to(directory) or output.is_relative_to(archive_directory):
        raise ValueError("diagnostic output must be outside both source runs and sealed archive")
    with contextlib.redirect_stdout(io.StringIO()):
        verified = verify_archive(archive_directory)
    manifest = read_json(directory/"manifest.json")
    if manifest != read_json(archive_directory/"pilot/manifest.json"):
        raise ValueError("input run differs from the archived pilot")
    config, identity = manifest["config"], manifest["identity"]
    validate_config(config)
    if tuple(config["seeds"]) != SEEDS or tuple(config["models"]) != MODELS or config["blocks"] != 18:
        raise ValueError("this diagnostic requires all declared pilot seeds, models and blocks")
    if config["batch_size"] != 32 or config["eval_size"] != 512 or config["support_replicates"] != 2:
        raise ValueError("unexpected pilot query or support size")
    if source_manifest() != manifest["source_files"] or digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("scientific source differs from the original pilot")
    expected_names = set()
    for model in MODELS:
        for seed in SEEDS:
            folder = f"{model}_{seed}"
            expected_names.add(f"{folder}/prefix.pt")
            expected_names.add(f"{folder}/prefix.json")
            for block in RETURNS:
                expected_names.update(f"{folder}/block_{block:02}/{name}"
                                      for name in ("before.pt", "checkpoint.pt", "result.json"))
    expected_names.update(("manifest.json", "completion.json", "training_source.zip",
                           "protocol_at_lock.md", "protocol_lock.json"))
    integrity = read_json(archive_directory/"pilot/file_integrity.json")
    archived = {row["path"]: dict(bytes=row["bytes"], sha256=row["sha256"]) for row in integrity["files"]}
    # The older PT/JSON/ZIP integrity scan omitted Markdown. Its protocol is
    # independently covered by the sealed archive manifest verified above.
    archived["protocol_at_lock.md"] = file_entry(archive_directory/"pilot/protocol_at_lock.md")
    if not integrity["passed"] or not expected_names <= archived.keys():
        raise ValueError("archived integrity coverage is incomplete")
    inputs = {name: archived[name] for name in sorted(expected_names)}
    check_files(directory, inputs)
    files = {"scripts/diagnose_return_context.py": Path(__file__).resolve(),
             "tests/test_return_context.py": ROOT/"tests/test_return_context.py",
             "scripts/verify_evidence_consolidation_archive.py": ROOT/"scripts/verify_evidence_consolidation_archive.py",
             "docs/return_context_protocol.md": Path(protocol).resolve()}
    files.update({f"src/{name}": ROOT/"src"/name for name in manifest["source_files"]})
    metadata = dict(input_directory=str(directory), archive_directory=str(archive_directory),
        input_identity=identity, input_files=inputs,
        archive_manifest=file_entry(archive_directory/"artifact_manifest.json"), archive_verification=verified,
        config=config, returns=list(RETURNS), models=list(MODELS), seeds=list(SEEDS),
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_samples=BOOTSTRAP_SAMPLES,
        prediction_range_tolerance=float(PREDICTION_ROUNDOFF),
        runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                     device="cpu", threads=1, deterministic=True))
    lock = snapshot_code(output, files, metadata)
    return directory, archive_directory, output, manifest, inputs, files, lock


def support_sets(before, after, block, seed, config, evaluator):
    observed, clean = experiences(block, seed, 0, config)
    if observed.fingerprint() != clean.fingerprint():
        raise AssertionError("return feedback unexpectedly contains reporting corruption")
    count = config["episode_size"]//config["batch_size"]
    previous = experiences(schedule(seed)[block.index-1], seed, count-1, config)[0]
    terminal = experiences(block, seed, count-1, config)[0]
    if before.draft.history.fingerprint() != previous.fingerprint() or after.draft.history.fingerprint() != terminal.fingerprint():
        raise AssertionError("saved causal histories differ from the regenerated stream")
    result = [("actual_start", None, before.draft.history)]
    result.extend((f"observed_{n}", None, replace_history(before.draft.history, observed, n)) for n in (8, 16, 32))
    result.append(("actual_end", None, after.draft.history))
    result.extend(("fresh_target", rep, evaluator.support(block.law, rep)) for rep in range(config["support_replicates"]))
    return result, observed


def markdown(summary):
    lines = ["# Adam: matched return-context diagnostic", "",
        "Retrospective descriptive analysis of the completed evidence-consolidation pilot. "
        "The analysis was locked before these additional query evaluations, after the original outcomes were known. "
        "There is no new method-selection screen or independent confirmation.", "",
        f"All {len(SEEDS)} seeds per architecture and all three returns were retained. "
        f"{len(summary['rows'])} weight/support evaluations use 512 matched queries each. "
        "Support replicates are averaged within a return, then returns within a seed. "
        "Intervals are descriptive 95% paired-seed bootstrap intervals (20,000 resamples).", "",
        "Brier differences below are multiplied by 100; negative favors the first named predictor. "
        "For context penalties, positive means lower error with target-law support.", "",
        "| Matched comparison | Conditional | Recurrent |", "|---|---:|---:|"]
    selected = (
        ("Sustained minus draft, actual preceding history", "sustained_minus_start_draft.actual_start"),
        ("Sustained minus draft, 32 observed return records", "sustained_minus_start_draft.observed_32"),
        ("Sustained minus draft, fresh target support", "sustained_minus_start_draft.fresh_target"),
        ("Draft context penalty, 32 observed return records", "actual_start_minus_observed_32.start_draft"),
        ("Sustained context penalty, 32 observed return records", "actual_start_minus_observed_32.start_sustained"),
        ("Current draft minus prefix, fresh target support", "start_draft_minus_prefix.fresh_target"),
        ("Current draft minus previous return, fresh target support (cycles 2/3)", "start_draft_minus_previous_return.fresh_target"),
        ("Return-end minus return-start draft, fresh target support", "end_draft_minus_start_draft.fresh_target"),
        ("Return-end draft minus prefix, fresh target support", "end_draft_minus_prefix.fresh_target"),
        ("Return-end draft minus previous return, fresh target support (cycles 2/3)", "end_draft_minus_previous_return.fresh_target"),
    )
    for label, key in selected:
        cells = []
        for model in MODELS:
            item = summary["contrasts"][model][key]["brier"]
            cells.append(f"{100*item['mean']:+.3f} [{100*item['lower']:+.3f}, {100*item['upper']:+.3f}]")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines.extend(["", "The exact prediction-level identity is actual-history gap = target-support gap "
        "+ (sustained context penalty - draft context penalty). It is not a decomposition of total "
        "online return loss into independently identified causes.", "",
        "| Context interaction, Brier x100 | Conditional | Recurrent |", "|---|---:|---:|"])
    for target in ("observed_32", "fresh_target"):
        cells = []
        for model in MODELS:
            item = summary["decomposition"][model][target]["brier"]["context_interaction"]
            cells.append(f"{100*item['mean']:+.3f} [{100*item['lower']:+.3f}, {100*item['upper']:+.3f}]")
        lines.append(f"| {target} | {' | '.join(cells)} |")
    lines.extend(["", "## Interpretation limits", "",
        "The preceding history is clean but belongs to the previous law. The return changes both "
        "base mode and active dependencies without announcing either in the images. Correct-support "
        "probes therefore concern contextual applicability, not specifically noise removal.", "",
        "The first return packet is ordinary observed feedback, used here with frozen weights on an "
        "independent query panel. Eight/sixteen-record replacement is a diagnostic subsampling probe. "
        "Fresh target support is evaluator-selected; actual-end support and end weights come from the "
        "future relative to the start. None is silently supplied to an online learner.", "",
        "Prefix and previous-return comparisons concern accessible predictions, not proof of neuron-level "
        "erasure. Conditional best-head references use these query outcomes for selection and are "
        "nondeployable; a mixture can outperform every single head. Recurrent models have no corresponding "
        "explicit-head reference. No representation was changed or trained.", "",
        "All rows, matched contrasts, six seed values, weight/support/query fingerprints, provenance, "
        "and the code/protocol lock are retained in summary.json, rows.jsonl, provenance.json, and "
        "analysis_lock.json. No input checkpoints or sealed archive artifacts were modified.", ""])
    return "\n".join(lines)


def diagnose(input_directory, archive_directory, output, protocol, lock_only=False):
    directory, archive_directory, output, manifest, inputs, files, lock = prepare(
        input_directory, archive_directory, output, protocol)
    if lock_only:
        result = dict(locked=True, retrospective=True, output=str(output), locked_utc=lock["locked_utc"],
                      input_files=len(inputs), code_files=len(files))
        print(json.dumps(result, indent=2))
        return result
    if (output/"completion.json").exists():
        raise ValueError("completed diagnostic output already exists; preserve it and use a new directory")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    identity, config = manifest["identity"], manifest["config"]
    rows, head_rows, provenance = [], [], []
    started = datetime.now(timezone.utc).isoformat()

    def get_checkpoint(relative):
        check_files(directory, {relative: inputs[relative]})
        saved = load(directory/relative, identity, "cpu")
        return saved["bundle"], saved["record"]

    for model in MODELS:
        for seed in SEEDS:
            folder = f"{model}_{seed}"
            prefix_path = f"{folder}/prefix.pt"
            prefix, prefix_record = get_checkpoint(prefix_path)
            if digest(prefix_record) != digest(read_json(directory/folder/"prefix.json")):
                raise ValueError("prefix checkpoint record differs from its archived JSON")
            prefix_signature = bundle_signature(prefix)
            previous = None
            previous_path = None
            previous_signature = None
            evaluator = Evaluator(seed, config)
            for index in RETURNS:
                block = schedule(seed)[index]
                if block.label != "return" or block.law.noise:
                    raise ValueError("declared return block is not an exact clean base return")
                start_path = f"{folder}/block_{index:02}/before.pt"
                end_path = f"{folder}/block_{index:02}/checkpoint.pt"
                before, start_record = get_checkpoint(start_path)
                after, end_record = get_checkpoint(end_path)
                for record in (start_record, end_record):
                    if (record["seed"], record["model"]) != (seed, model) or digest(record["block"]) != digest(asdict(block)):
                        raise ValueError("checkpoint return identity mismatch")
                if digest(end_record) != digest(read_json(directory/folder/f"block_{index:02}/result.json")):
                    raise ValueError("return checkpoint record differs from its archived JSON")
                signatures = {"before": bundle_signature(before), "after": bundle_signature(after)}
                candidates = [("start_draft", before.draft, start_path),
                              ("start_periodic", before.committed["periodic"], start_path),
                              ("start_sustained", before.committed["sustained"], start_path),
                              ("prefix_draft", prefix.draft, prefix_path),
                              ("end_draft", after.draft, end_path)]
                if previous is not None:
                    candidates.append(("previous_return_draft", previous.draft, previous_path))
                supports, observed = support_sets(before, after, block, seed, config, evaluator)
                cases, truth = evaluator.dataset(block.law)
                query_sha, truth_sha = array_hash(cases.observations), array_hash(truth)
                query_truth_sha = array_hash(cases.observations, truth)
                for weight, learner, path in candidates:
                    weight_sha = state_hash(learner.model.state_dict())
                    for support, replicate, history in supports:
                        metrics, heads = probe(learner, cases.observations, truth, history,
                                               include_heads=model == "conditional" and support == "actual_start")
                        rows.append(dict(model=model, seed=seed, block=index, cycle=block.cycle+1,
                            weight=weight, source_checkpoint=path, source_checkpoint_sha256=inputs[path]["sha256"],
                            weight_sha256=weight_sha, parameters=sum(p.numel() for p in learner.model.parameters()),
                            support=support, replicate=replicate, support_records=len(history),
                            support_sha256=history.fingerprint(), query_sha256=query_sha, truth_sha256=truth_sha,
                            query_truth_sha256=query_truth_sha, **metrics))
                        if heads is not None:
                            head_rows.append(dict(model=model, seed=seed, block=index, cycle=block.cycle+1,
                                weight=weight, weight_sha256=weight_sha, query_truth_sha256=query_truth_sha, **heads))
                if array_hash(cases.observations, truth) != query_truth_sha:
                    raise AssertionError("diagnostic changed the query panel")
                if (bundle_signature(before) != signatures["before"] or bundle_signature(after) != signatures["after"]
                        or bundle_signature(prefix) != prefix_signature
                        or (previous is not None and bundle_signature(previous) != previous_signature)):
                    raise AssertionError("diagnostic changed loaded learning, memory or model state")
                provenance.append(dict(model=model, seed=seed, block=index, law=asdict(block.law),
                    query_sha256=query_sha, truth_sha256=truth_sha, query_truth_sha256=query_truth_sha,
                    observed_first_packet_sha256=observed.fingerprint(),
                    before_state_sha256=signatures["before"], after_state_sha256=signatures["after"],
                    prefix_state_sha256=prefix_signature, previous_return_state_sha256=previous_signature,
                    source_global_batches=dict(return_start=before.batches, return_end=after.batches,
                        prefix=prefix.batches, previous_return=None if previous is None else previous.batches),
                    deployed_start_hashes={name: snapshot.sha256 for name, snapshot in before.committed.items()},
                    query_count=len(cases.observations), supports=len(supports), weights=len(candidates),
                    prediction_passes=len(supports)*len(candidates), optimizer_updates=0,
                    loaded_states_unchanged=True))
                previous, previous_path, previous_signature = after, end_path, signatures["after"]
            print(json.dumps(dict(model=model, seed=seed, returns=len(RETURNS), completed=True)), flush=True)

    check_files(directory, inputs)
    if source_manifest() != manifest["source_files"]:
        raise ValueError("scientific source changed during diagnosis")
    snapshot_code(output, files, lock["metadata"])
    with contextlib.redirect_stdout(io.StringIO()):
        verified_after = verify_archive(archive_directory)
    if verified_after != lock["metadata"]["archive_verification"]:
        raise ValueError("sealed archive changed during diagnosis")
    if len(rows) != len(SEEDS)*len(MODELS)*(5+6+6)*(5+config["support_replicates"]):
        raise AssertionError("incomplete crossed weight/support evaluations")
    result = dict(retrospective=True, input_identity=identity, config=config, started_utc=started,
        completed_utc=datetime.now(timezone.utc).isoformat(), locked_utc=lock["locked_utc"],
        command=dict(executable=sys.executable, argv=list(sys.argv)),
        bootstrap=dict(unit="independent seed; support replicates then returns averaged within seed",
                       samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, confidence=.95,
                       interpretation="descriptive, unadjusted; no prospective pass/fail rule"),
        rows=rows, conditional_query_selected_head_references=head_rows, provenance=provenance,
        verification=dict(input_files_unchanged=len(inputs), loaded_states_unchanged=True,
                          source_unchanged=True, code_protocol_lock_unchanged=True, archive=verified_after),
        **build_contrasts(rows))
    write_json(output/"summary.json", result)
    write_json(output/"provenance.json", dict(retrospective=True, input_identity=identity,
        locked_utc=lock["locked_utc"], command=result["command"], inputs=inputs,
        query_evaluations=provenance, verification=result["verification"]))
    with (output/"rows.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    (output/"summary.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    artifacts = {path.name: file_entry(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output/"completion.json", dict(retrospective=True, input_identity=identity,
        completed_utc=result["completed_utc"], rows=len(rows), return_states=len(provenance),
        command=result["command"],
        prediction_passes=sum(item["prediction_passes"] for item in provenance), optimizer_updates=0,
        artifacts=artifacts, verification=result["verification"]))
    print(json.dumps(dict(completed=True, output=str(output), rows=len(rows), returns=len(provenance)), indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    diagnose(args.input, args.archive, args.output, args.protocol, args.lock_only)
