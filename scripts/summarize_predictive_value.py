"""Locked predictive-rehearsal score validation, with independent seed inference.

The pure analysis functions consume JSON records and require only NumPy. No
candidate choice is recomputed using validation labels. The validation oracle
is explicitly retrospective and never participates in an allocation decision.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import shutil
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(14001, 14013))
MODELS = ("conditional", "recurrent")
PANELS = ("near_reapplied", "near_frozen", "return_reapplied", "return_frozen")
METRICS = ("brier", "survival", "early_brier", "late_brier")
SELECTORS = ("value", "accuracy", "uniform", "replacement", "oracle")
CONTRASTS = {
    "value_minus_uniform": ("value", "uniform"),
    "value_minus_accuracy": ("value", "accuracy"),
    "accuracy_minus_uniform": ("accuracy", "uniform"),
    "value_minus_replacement": ("value", "replacement"),
    "accuracy_minus_replacement": ("accuracy", "replacement"),
    "uniform_minus_replacement": ("uniform", "replacement"),
    "oracle_minus_uniform": ("oracle", "uniform"),
    "value_minus_oracle": ("value", "oracle"),
}
PROBABILITY_METRIC_TOLERANCE = 8*np.finfo(np.float32).eps
REQUIRED_ANALYSIS = ("scripts/summarize_predictive_value.py", "tests/test_predictive_value_summary.py")
SCIENTIFIC_SETTINGS = dict(kind="diagnostic", prefix_blocks=4, prefix_size=1024,
    acquisition_size=8192, recovery_size=1024, score_size=512, validation_size=512,
    gaps=[512, 4096], batch_size=32, candidate_count=8, rehearsal_updates=12,
    width=64, experts=4, context_width=12, decoder_width=64, interaction_features=True,
    lr=.002, memory_packets=16, updates_per_batch=12, evidence_strength=1.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean(values):
    values = list(values)
    require(bool(values), "cannot average empty observations")
    return float(np.mean(values))


def paired(values, seeds=None):
    """Fixed paired seed bootstrap; the independent unit is always a seed."""
    values = np.asarray(list(values), dtype=float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
            "finite nonempty one-dimensional seed differences required")
    values[np.abs(values) < 1e-12] = 0.
    rng = np.random.default_rng(27192026)
    boot = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    result = dict(mean=float(values.mean()), lower=float(np.quantile(boot, .025)),
        upper=float(np.quantile(boot, .975)), positive=int((values > 0).sum()),
        negative=int((values < 0).sum()), n=len(values), differences=values.tolist())
    if seeds is not None:
        require(len(seeds) == len(values) and len(set(seeds)) == len(seeds), "seed labels must be unique and aligned")
        result["seeds"] = list(seeds)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite record value")


def read(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    finite_tree(result)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def valid_hash(value, optional=False):
    require((optional and value is None) or (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)), "invalid SHA256 record field")


def metric_array(values, length, label):
    require(isinstance(values, list) and len(values) == length
            and all(type(v) in (int, float) for v in values), f"invalid {label} array")
    result = np.asarray(values, dtype=float)
    require(np.isfinite(result).all() and np.all(result >= 0)
            and np.all(result <= 1+PROBABILITY_METRIC_TOLERANCE), f"invalid {label} metric range")
    return result


def return_origin(law, seed):
    return (law.get("mode") == seed % 2 and list(law.get("active", [])) == []
            and law.get("revised") is False and law.get("noise") == 0)


def validate_assessment(record, candidate_count=None):
    """Validate metric arrays and past-only choices, without recomputing models."""
    finite_tree(record)
    seed, model, index = (record[name] for name in ("seed", "model", "index"))
    require(type(seed) is int and seed >= 0 and model in MODELS and type(index) is int and index in (0, 1),
            "invalid assessment key")
    require(record["cue"] == tuple(permutations(range(3)))[seed % 6][0], "wrong first novel cue")
    require(type(record["gap"]) is int and record["gap"] > 0, "invalid gap")
    candidates, replacement, choice = (record[name] for name in ("candidates", "replacement", "choice"))
    require(isinstance(candidates, list) and len(candidates) >= 2, "at least two candidates required")
    if candidate_count is not None:
        require(len(candidates) == candidate_count, "candidate count differs from configuration")
    ids = []
    for candidate in [*candidates, replacement]:
        require(type(candidate["id"]) is int and candidate["id"] >= 0, "invalid candidate ID")
        ids.append(candidate["id"])
        metric_array([candidate["original_brier"]], 1, "original Brier")
        require(type(candidate["original_steps"]) is int and candidate["original_steps"] >= 0,
                "invalid original optimizer step")
        valid_hash(candidate["query_sha256"])
        valid_hash(candidate["support_sha256"], optional=True)
        law = candidate["origin_law"]
        require(isinstance(law, dict) and set(law) == {"mode", "active", "revised", "noise"}
                and law["mode"] in (0, 1) and isinstance(law["active"], (tuple, list))
                and all(type(cue) is int and cue in (0, 1, 2) for cue in law["active"])
                and type(law["revised"]) is bool and law["noise"] == 0,
                "invalid clean source-law provenance")
    require(len(ids) == len(set(ids)), "duplicate candidate/replacement IDs")
    if "anchor_id" in record:
        require(record["anchor_id"] not in ids, "latest anchor is a candidate or replacement")
    count = len(candidates)
    losses = metric_array(choice["score_losses"], count+1, "score loss")
    values = np.asarray(choice["values"], dtype=float)
    require(values.shape == (count,) and np.isfinite(values).all()
            and np.allclose(values, losses[-1]-losses[:-1], rtol=0, atol=1e-14), "incorrect common-replacement values")
    # Match the declared largest-value rule, including exact floating-point ties.
    expected_values = losses[-1]-losses[:-1]
    value_index = min(range(count), key=lambda j: (-float(expected_values[j]), ids[j]))
    accuracy_index = min(range(count), key=lambda j: (candidates[j]["original_brier"], ids[j]))
    require(choice["value_id"] == ids[value_index], "value choice differs from score-window rule/tie break")
    require(choice["accuracy_id"] == ids[accuracy_index], "accuracy choice differs from original error/tie break")
    require(set(record["panels"]) == set(PANELS), "incomplete validation panels")
    for panel, data in record["panels"].items():
        for metric in METRICS:
            metric_array(data[metric], count+1, f"{panel}/{metric}")
    return seed, model, index


def assessment_metrics(record):
    """Derive selectors without using validation outcomes to alter fixed choices."""
    validate_assessment(record)
    candidates, choice = record["candidates"], record["choice"]
    ids = [item["id"] for item in candidates]
    count = len(ids)
    value_index, accuracy_index = (ids.index(choice[name]) for name in ("value_id", "accuracy_id"))
    panel_data = {"score": dict(brier=choice["score_losses"]), **record["panels"]}
    panels = {}
    for panel, data in panel_data.items():
        # This oracle is query-selected separately within each validation panel.
        # Its other outcomes use the SAME full-Brier-selected candidate.
        oracle_index = min(range(count), key=lambda j: (data["brier"][j], ids[j]))
        selectors = {selector: {} for selector in SELECTORS}
        for metric in (("brier",) if panel == "score" else METRICS):
            values = data[metric]
            for selector, index in (("value", value_index), ("accuracy", accuracy_index),
                                    ("replacement", count), ("oracle", oracle_index)):
                selectors[selector][metric] = float(values[index])
            selectors["uniform"][metric] = mean(values[:count])
        panels[panel] = dict(selectors=selectors,
            contrasts={name: {metric: selectors[first][metric]-selectors[second][metric]
                             for metric in selectors[first]} for name, (first, second) in CONTRASTS.items()},
            brier_spread=float(max(data["brier"][:count])-min(data["brier"][:count])),
            oracle_id=ids[oracle_index])
    coverage = [return_origin(item["origin_law"], record["seed"]) for item in candidates]
    query_support_overlaps = [(a["id"], b["id"]) for a in candidates for b in candidates
                              if a["id"] != b["id"] and a["query_sha256"] == b["support_sha256"]]
    row = {name: record[name] for name in ("seed", "model", "index", "cue", "gap")}
    row.update(candidate_ids=ids, replacement_id=record["replacement"]["id"],
        value_id=choice["value_id"], accuracy_id=choice["accuracy_id"], panels=panels,
        return_candidates=int(sum(coverage)), return_candidate_fraction=mean(coverage),
        value_return_origin=bool(coverage[value_index]), accuracy_return_origin=bool(coverage[accuracy_index]),
        selectors_agree=choice["value_id"] == choice["accuracy_id"],
        query_support_overlaps=[list(pair) for pair in query_support_overlaps],
        identical_query_pairs=sum(a["query_sha256"] == b["query_sha256"]
                                  for j, a in enumerate(candidates) for b in candidates[j+1:]),
        original_brier={"value": candidates[value_index]["original_brier"],
                        "accuracy": candidates[accuracy_index]["original_brier"],
                        "uniform": mean(item["original_brier"] for item in candidates)},
        original_steps={"value": candidates[value_index]["original_steps"],
                        "accuracy": candidates[accuracy_index]["original_steps"],
                        "uniform": mean(item["original_steps"] for item in candidates)})
    if "anchor_id" in record:
        row["age_packets"] = {"value": record["anchor_id"]-ids[value_index],
            "accuracy": record["anchor_id"]-ids[accuracy_index],
            "uniform": mean(record["anchor_id"]-index for index in ids)}
    return row


def tier_screen(panel_contrasts, eligible):
    uniform = panel_contrasts["value_minus_uniform"]
    accuracy = panel_contrasts["value_minus_accuracy"]
    brier, survival = uniform["brier"], uniform["survival"]
    improved = sum(value < 0 for value in brier["differences"])
    gates = dict(complete_scientific_cohort=dict(passed=bool(eligible)),
        brier_gain=dict(passed=brier["mean"] <= -.0005, value=brier["mean"], maximum=-.0005),
        seed_consistency=dict(passed=improved >= 9, improved_seeds=improved, minimum=9, n=brier["n"]),
        accuracy_control=dict(passed=accuracy["brier"]["mean"] <= 0, value=accuracy["brier"]["mean"], maximum=0.),
        survival=dict(passed=survival["mean"] >= -.005, value=survival["mean"], minimum=-.005))
    return dict(passed=all(gate["passed"] for gate in gates.values()), gates=gates,
        failed=[name for name, gate in gates.items() if not gate["passed"]],
        interpretation="Fixed mean-based research allocation criteria, not superiority/noninferiority tests. "
                       "A tie with original accuracy is not evidence of additional benefit over that cheaper signal.")


def eligible_config(config):
    # The full reader additionally applies the source-locked runner validator.
    return (config.get("seeds") == SEEDS and set(config.get("models", [])) == set(MODELS)
            and all(config.get(name) == value for name, value in SCIENTIFIC_SETTINGS.items()))


def analyze_records(records, config):
    """All assessments enter every primary estimate, including absent coverage."""
    if isinstance(records, dict):
        records = list(records.values())
    records = list(records)
    require(len(set(config["seeds"])) == len(config["seeds"])
            and len(set(config["models"])) == len(config["models"]), "duplicate configured seeds/models")
    keys = [validate_assessment(record, config.get("candidate_count")) for record in records]
    expected = {(seed, model, index) for seed in config["seeds"] for model in config["models"] for index in (0, 1)}
    require(len(keys) == len(set(keys)) and set(keys) == expected, "every configured seed/model needs both assessments exactly once")
    if "gaps" in config:
        for record in records:
            gaps = config["gaps"][::-1] if (record["seed"]//6) % 2 else config["gaps"]
            require(record["gap"] == gaps[record["index"]], "assessment gap/order differs from schedule")
    rows = sorted((assessment_metrics(record) for record in records), key=lambda r: (r["model"], r["seed"], r["index"]))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["seed"]].append(row)
    contrasts, means, spread, coverage, slices, transports = {}, {}, {}, {}, {}, {}
    for model in config["models"]:
        seeds = config["seeds"]
        contrasts[model], means[model], spread[model] = {}, {}, {}
        for panel in ("score", *PANELS):
            fields = ("brier",) if panel == "score" else METRICS
            contrasts[model][panel] = {contrast: {metric: paired([
                mean(row["panels"][panel]["contrasts"][contrast][metric] for row in grouped[model, seed])
                for seed in seeds], seeds) for metric in fields} for contrast in CONTRASTS}
            means[model][panel] = {selector: {metric: mean(
                mean(row["panels"][panel]["selectors"][selector][metric] for row in grouped[model, seed])
                for seed in seeds) for metric in fields} for selector in SELECTORS}
            spread[model][panel] = paired([mean(row["panels"][panel]["brier_spread"]
                                                      for row in grouped[model, seed]) for seed in seeds], seeds)
        model_rows = [row for row in rows if row["model"] == model]
        coverage[model] = dict(assessments=len(model_rows),
            with_return_candidate=sum(row["return_candidates"] > 0 for row in model_rows),
            without_return_candidate=sum(row["return_candidates"] == 0 for row in model_rows),
            mean_return_candidates=mean(row["return_candidates"] for row in model_rows),
            value_return_fraction=mean(row["value_return_origin"] for row in model_rows),
            accuracy_return_fraction=mean(row["accuracy_return_origin"] for row in model_rows),
            exact_uniform_return_fraction=mean(row["return_candidate_fraction"] for row in model_rows),
            selector_agreement_fraction=mean(row["selectors_agree"] for row in model_rows),
            assessments_with_query_support_overlap=sum(bool(row["query_support_overlaps"]) for row in model_rows),
            query_support_overlap_pairs=sum(len(row["query_support_overlaps"]) for row in model_rows),
            identical_query_pairs=sum(row["identical_query_pairs"] for row in model_rows))
        slices[model] = {}
        for grouping in ("cue", "gap", "index"):
            slices[model][grouping] = {}
            for label in sorted({row[grouping] for row in model_rows}):
                selected = [row for row in model_rows if row[grouping] == label]
                subset = [seed for seed in seeds if any(row["seed"] == seed for row in selected)]
                slices[model][grouping][str(label)] = dict(seeds=subset, n=len(subset), assessments=len(selected),
                    contrasts={panel: {contrast: {metric: mean(mean(
                        row["panels"][panel]["contrasts"][contrast][metric]
                        for row in selected if row["seed"] == seed) for seed in subset)
                        for metric in METRICS} for contrast in CONTRASTS} for panel in PANELS},
                    mean_return_candidates=mean(row["return_candidates"] for row in selected),
                    without_return_candidate=sum(row["return_candidates"] == 0 for row in selected))
        transports[model] = {window: {name: {metric: paired([mean(
            row["panels"][f"{window}_reapplied"]["contrasts"][name][metric]
            - row["panels"][f"{window}_frozen"]["contrasts"][name][metric]
            for row in grouped[model, seed]) for seed in seeds], seeds) for metric in METRICS}
            for name in ("value_minus_uniform", "value_minus_accuracy")} for window in ("near", "return")}
    eligible = eligible_config(config)
    screen = {model: {window: tier_screen(contrasts[model][f"{window}_reapplied"], eligible)
                     for window in ("near", "return")} for model in config["models"]}
    for tiers in screen.values():
        local, recurring = tiers["near"]["passed"], tiers["return"]["passed"]
        tiers["recommendation"] = ("eligible_for_separately_specified_recurrence_policy" if local and recurring
            else "eligible_for_separately_specified_local_policy_only" if local
            else "return_only_signal_does_not_pass_local_tier" if recurring else "stop_expansion_of_fixed_score_recipe")
    return dict(rows=rows, contrasts=contrasts, selector_means=means, candidate_spread=spread,
        coverage=coverage, descriptive_slices=slices, reapplied_minus_frozen_contrasts=transports,
        eligible_scientific_cohort=eligible, screen=screen,
        inference=dict(unit="independent seed; two assessments averaged first", bootstrap_seed=27192026,
            bootstrap_resamples=20000, interval="descriptive unadjusted percentile 95%", count_per_seed=2,
            oracle="Validation-Brier-selected single candidate; conditional, not deployable and not a survival oracle.",
            score_window="Used for selection; apparent performance here is optimistic and not validation."))


def checked_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(expected) and set(names) == set(expected) and archive.testzip() is None,
                "locked archive file set/CRC mismatch")
        require(all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in expected.items()),
                "locked archive content hash mismatch")


def verify_analysis_lock(directory, config, protocol_sha):
    lock = read(Path(directory)/"analysis_lock.json")
    require(lock["config"] == config and lock["config_sha256"] == digest(config), "analysis-locked configuration changed")
    require(set(REQUIRED_ANALYSIS) <= lock["files"].keys(), "prospective analysis files absent from lock")
    checked_zip(Path(directory)/"analysis_at_lock.zip", lock["files"])
    for name, expected in lock["files"].items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file() and sha(path) == expected,
                f"current locked analysis/configuration/protocol changed: {name}")
    require(any(name.startswith("docs/") and expected == protocol_sha for name, expected in lock["files"].items()),
            "protocol absent from analysis lock")
    return lock


def study_api():
    from acp_cl.predictive_value import study
    return study


def close_array(actual, expected, message):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=0, atol=1e-14), message)


def verify_phase(record, phase, seed, model, config, identity, start_id):
    """Check JSON timelines, equal work, and scalar panel aggregation."""
    size, count = config["batch_size"], config["candidate_count"]+1
    require(record["identity"] == identity and record["seed"] == seed and record["model"] == model
            and digest(record["phase"]) == digest(phase), "phase identity/schedule mismatch")
    packets = record["packets"]
    require(record["completed_packets"] == len(packets) == phase["size"]//size, "incomplete ordinary phase")
    for offset, packet in enumerate(packets):
        require(packet["id"] == start_id+offset and packet["original_steps"] == (start_id+offset)*config["updates_per_batch"]
                and packet["origin_phase"] == phase["name"] and packet["origin_index"] == offset
                and digest(packet["origin_law"]) == digest(phase["law"]), "original prequential timeline mismatch")
        for name in ("query_sha256", "original_prediction_sha256", "original_model_sha256"):
            valid_hash(packet[name])
        valid_hash(packet["support_sha256"], optional=start_id+offset == 0)
        if offset:
            require(packet["support_sha256"] == packets[offset-1]["query_sha256"], "noncausal within-phase support")
        predictions = np.asarray(packet["probabilities"], dtype=float)
        require(predictions.shape == (size, 3) and np.isfinite(predictions).all()
                and np.all(predictions >= -PROBABILITY_METRIC_TOLERANCE)
                and np.all(predictions <= 1+PROBABILITY_METRIC_TOLERANCE), "invalid prequential probabilities")
        require(len(packet["actions"]) == size and all(type(a) is int and 0 <= a < 5 for a in packet["actions"]),
                "invalid ordinary performed actions")
        outcomes = np.asarray(packet["outcomes"], dtype=float)
        require(outcomes.shape == (size, 3) and np.isfinite(outcomes).all()
                and np.all(outcomes >= 0) and np.all(outcomes <= 1), "invalid reported outcomes")
        metric_array([packet["original_brier"]], 1, "original prequential Brier")
        close_array(packet["original_brier"], np.square(predictions-outcomes).mean(), "original Brier arithmetic mismatch")
    before, after = record["start_diagnostics"]["cost"], record["diagnostics"]["cost"]
    require(before["arrivals"] == start_id*size and after["arrivals"] == (start_id+len(packets))*size
            and before["optimizer_steps"] == start_id*config["updates_per_batch"]
            and after["optimizer_steps"] == (start_id+len(packets))*config["updates_per_batch"],
            "ordinary trajectory received incorrect arrivals or optimizer steps")
    for memory, expected_seen in ((record["start_memory"], start_id), (record["memory"], start_id+len(packets))):
        require(memory["type"] == "ReservoirMemory" and memory["seen"] == memory["age"] == expected_seen
                and memory["capacity"] == config["memory_packets"]
                and len(memory["ids"]) == len(set(memory["ids"])) == min(expected_seen, config["memory_packets"]),
                "ordinary reservoir policy/lifetime mismatch")
    require(set(record["metadata"]) == {str(i) for i in record["memory"]["ids"]}, "bounded original-error metadata mismatch")
    evaluations = record["evaluations"]
    scoring = phase["kind"] in ("score", "near", "return")
    require(len(evaluations) == (len(packets) if scoring else 0), "evaluation window coverage mismatch")
    require(len(record["shadow_work"]) == (count if scoring else 0), "rehearsal work coverage mismatch")
    for work in record["shadow_work"]:
        updates = config["rehearsal_updates"]
        require(work["steps"] == updates and work["forwards"] == 2*updates
                and work["backward_calls"] == updates
                and work["query_presentations"] == work["support_presentations"] == 2*updates*size
                and work["parent_unchanged"] is True and work["memory_history_unchanged"] is True,
                "unmatched diagnostic work or mutated ordinary parent")
    for packet, evaluation in zip(packets, evaluations):
        require(evaluation["packet_id"] == packet["id"] and evaluation["query_sha256"] == packet["query_sha256"]
                and evaluation["support_sha256"] == packet["support_sha256"], "evaluation query/context mismatch")
        expected_groups = ("reapplied",) if phase["kind"] == "score" else ("reapplied", "frozen")
        extra = set() if phase["kind"] == "score" else {"truth"}
        require(set(evaluation)-{"packet_id", "query_sha256", "support_sha256"} == set(expected_groups)|extra,
                "evaluation shadow group mismatch")
        outcomes = np.asarray(packet["outcomes"], dtype=float)
        if phase["kind"] != "score":
            truth = np.asarray(evaluation["truth"], dtype=float)
            require(truth.shape == (size, 5, 3) and np.isfinite(truth).all()
                    and np.all(truth >= 0) and np.all(truth <= 1), "invalid evaluator counterfactual truth")
            close_array(outcomes, truth[np.arange(size), packet["actions"]], "clean reported/physical outcomes differ")
        for group in expected_groups:
            require(len(evaluation[group]) == count, "incomplete candidate/replacement predictions")
            for prediction in evaluation[group]:
                metric_array([prediction["brier"]], 1, "packet Brier")
                valid_hash(prediction["prediction_sha256"])
                probabilities = np.asarray(prediction["probabilities"], dtype=float)
                require(probabilities.shape == (size, 3) and np.isfinite(probabilities).all()
                        and np.all(probabilities >= -PROBABILITY_METRIC_TOLERANCE)
                        and np.all(probabilities <= 1+PROBABILITY_METRIC_TOLERANCE), "invalid shadow probability array")
                close_array(prediction["brier"], np.square(probabilities-outcomes).mean(), "shadow Brier arithmetic mismatch")
                if phase["kind"] != "score":
                    metric_array([prediction["survival"]], 1, "packet survival")
                    require(len(prediction["actions"]) == size
                            and all(type(a) is int and 0 <= a < 5 for a in prediction["actions"]), "invalid greedy actions")
                    close_array(prediction["survival"], truth[np.arange(size), prediction["actions"], -1].mean(),
                                "greedy-action survival arithmetic mismatch")
    if phase["kind"] == "score":
        close_array(record["score_losses"], np.mean([[item["brier"] for item in row["reapplied"]]
                                                    for row in evaluations], axis=0), "score scalar aggregation mismatch")
    elif scoring:
        require(set(record["panels"]) == {"reapplied", "frozen"} and len(evaluations) >= 2,
                "early/late panel needs at least two packets")
        for group in ("reapplied", "frozen"):
            for metric in ("brier", "survival"):
                values = [[item[metric] for item in row[group]] for row in evaluations]
                close_array(record["panels"][group][metric], np.mean(values, axis=0), "validation scalar aggregation mismatch")
                if metric == "brier":
                    close_array(record["panels"][group]["early_brier"], values[0], "initial-packet aggregation mismatch")
                    close_array(record["panels"][group]["late_brier"], np.mean(values[1:], axis=0), "later-packet aggregation mismatch")


def audit_records(directory):
    """Read-only structural/source audit; independent tensor reconstruction is separate."""
    directory = Path(directory)
    manifest = read(directory/"manifest.json")
    config, identity = manifest["config"], manifest["identity"]
    study = study_api()
    study.validate_config(config)
    for key, item in (("config_sha256", config), ("source_sha256", manifest["source_files"]),
                      ("runtime_sha256", manifest["runtime"])):
        require(digest(item) == identity[key], "manifest identity mismatch")
    require(study.source_manifest() == manifest["source_files"], "current scientific source changed after lock")
    checked_zip(directory/"training_source.zip", manifest["source_files"])
    protocol = read(directory/"protocol_lock.json")
    require(all(protocol[name] == value for name, value in identity.items())
            and sha(directory/"protocol_at_lock.md") == protocol["protocol_sha256"], "protocol identity/hash mismatch")
    verify_analysis_lock(directory, config, protocol["protocol_sha256"])
    completion = read(directory/"completion.json")
    require(completion["identity"] == identity and all(completion[name] == value for name, value in study.counts(config).items()),
            "incomplete cohort or completion identity mismatch")
    pairs = {(seed, model) for seed in config["seeds"] for model in config["models"]}
    jobs = completion["jobs"]
    require(len(jobs) == len(pairs) and {(job["seed"], job["model"]) for job in jobs} == pairs
            and all(job["affinity"] == manifest["runtime"]["affinity"] for job in jobs), "worker coverage or affinity mismatch")
    records, hashes, expected_paths = {}, [], set()
    def hashed(path, kind):
        hashes.append(dict(path=path.relative_to(directory).as_posix(), sha256=sha(path), kind=kind))
        expected_paths.add(path)
        return hashes[-1]["sha256"]
    for seed, model in sorted(pairs):
        folder = directory/f"{model}_{seed}"
        trajectory_path = folder/"result.json"
        trajectory = read(trajectory_path)
        require(trajectory["identity"] == identity and trajectory["seed"] == seed and trajectory["model"] == model
                and trajectory["affinity"] == manifest["runtime"]["affinity"], "trajectory identity/runtime mismatch")
        schedule = study.schedule(seed, config)
        require(trajectory["phase_names"] == [phase["name"] for phase in schedule]
                and set(trajectory["phase_hashes"]) == {phase["name"] for phase in schedule}
                and set(trajectory["assessment_hashes"]) == {"0", "1"}, "trajectory phase/assessment coverage mismatch")
        hashed(trajectory_path, "trajectory")
        phases, all_packets, start_id, previous_signature, previous_query = {}, {}, 0, None, None
        for phase in schedule:
            path = folder/phase["name"]/"result.json"
            require(path.is_file(), "missing phase result")
            value = read(path)
            require(hashed(path, "phase") == trajectory["phase_hashes"][phase["name"]], "phase result hash mismatch")
            verify_phase(value, phase, seed, model, config, identity, start_id)
            if previous_signature is not None:
                require(value["before_signature"] == previous_signature, "ordinary phase state chain mismatch")
            require(value["packets"][0]["support_sha256"] == previous_query, "noncausal cross-phase support")
            for packet in value["packets"]:
                all_packets[packet["id"]] = packet
            previous_signature, previous_query = value["final_signature"], value["packets"][-1]["query_sha256"]
            start_id += len(value["packets"])
            phases[phase["name"]] = value
            for name in ("before.pt", "checkpoint.pt"):
                require((path.parent/name).is_file(), "missing ordinary checkpoint")
            if phase["kind"] in ("score", "near", "return"):
                require((path.parent/"shadows.pt").is_file(), "missing diagnostic checkpoint")
        require(trajectory["final_signature"] == previous_signature
                and trajectory["diagnostics"] == phases[schedule[-1]["name"]]["diagnostics"], "trajectory terminal state mismatch")
        require(trajectory["prequential_work"] == dict(forwards=start_id, query_presentations=start_id*config["batch_size"],
                    support_presentations=start_id*config["batch_size"]), "ordinary prequential work mismatch")
        for index in (0, 1):
            path = folder/f"assessment_{index}"/"result.json"
            record = read(path)
            key = validate_assessment(record, config["candidate_count"])
            require(key == (seed, model, index) and record["identity"] == identity and key not in records,
                    "assessment identity/path mismatch")
            require(hashed(path, "assessment") == trajectory["assessment_hashes"][str(index)], "assessment result hash mismatch")
            score = phases[f"score_{index}"]
            require(record["gap"] == score["phase"]["gap"] and record["parent_steps"] == score["start_diagnostics"]["cost"]["optimizer_steps"]
                    and record["anchor_id"] == score["start_memory"]["seen"]-1
                    and record["reservoir_ids"] == score["start_memory"]["ids"], "candidate acquisition time/state mismatch")
            selected_ids, replacement_id = study.candidate_ids(record["reservoir_ids"], record["anchor_id"],
                                                               config["candidate_count"], seed, index)
            require([item["id"] for item in record["candidates"]] == selected_ids
                    and record["replacement"]["id"] == replacement_id, "candidate sampling mismatch")
            for candidate in [*record["candidates"], record["replacement"]]:
                original = all_packets[candidate["id"]]
                require(candidate == {name: value for name, value in original.items()
                                      if name not in {"probabilities", "actions", "outcomes"}},
                        "candidate original prequential provenance mismatch")
            close_array(record["choice"]["score_losses"], score["score_losses"], "choice uses wrong score-window loss")
            choice_path = path.parent/"choice.json"
            choice = read(choice_path)
            require(choice["identity"] == identity and (choice["seed"], choice["model"], choice["index"]) == key
                    and choice["score_record_sha256"] == digest(score) and choice["choice"] == record["choice"],
                    "sealed choice/score provenance mismatch")
            hashes.append(dict(path=choice_path.relative_to(directory).as_posix(), sha256=sha(choice_path), kind="choice"))
            require((path.parent/"candidates.pt").is_file(), "missing retained candidate checkpoint")
            require(set(record["phase_hashes"]) == {f"{name}_{index}" for name in ("score", "near", "gap", "return")},
                    "assessment phase coverage mismatch")
            for name, expected in record["phase_hashes"].items():
                require(expected == trajectory["phase_hashes"][name], "assessment phase hash mismatch")
            for name in ("score", "near", "return"):
                require(record["work"][name] == phases[f"{name}_{index}"]["shadow_work"], "assessment diagnostic work mismatch")
            for name in ("near", "return"):
                for kind in ("reapplied", "frozen"):
                    require(record["panels"][f"{name}_{kind}"] == phases[f"{name}_{index}"]["panels"][kind],
                            "assessment validation panel differs from underlying phase")
            records[key] = record
    require(set(directory.glob("**/result.json")) == expected_paths, "unexpected result records outside complete cohort")
    return manifest, records, hashes


def markdown(result):
    lines = ["# Predictive rehearsal value: score validation", "",
        "This is a fixed diagnostic of extra rehearsal value. The ordinary uniform-replay trajectory was unchanged. "
        "It does not evaluate an adaptive retention policy.", "",
        "The independent unit is a seed after averaging its two assessments. Intervals are descriptive 95% "
        "paired seed bootstrap intervals (20,000 resamples). Allocation screens are not significance or "
        "noninferiority tests. Original-accuracy ties do not establish extra benefit over the cheaper signal.", "",
        "| Model | Local tier | Return tier | Allocation |", "|---|---|---|---|"]
    for model, tiers in result["screen"].items():
        lines.append(f"| {model} | {'PASS' if tiers['near']['passed'] else 'FAIL'} | "
                     f"{'PASS' if tiers['return']['passed'] else 'FAIL'} | {tiers['recommendation']} |")
    if not result["eligible_scientific_cohort"]:
        lines += ["", "This engineering/subset cohort is ineligible for scientific continuation criteria."]
    lines += ["", "## Primary matched reapplication", "", "Brier is shown ×100; survival differences are percentage points. "
        "Negative Brier and positive survival favor value selection.", "",
        "| Model | Window | Value−uniform Brier [95% interval] | Improved seeds | Value−accuracy Brier [95% interval] | Survival−uniform [95% interval] |",
        "|---|---|---:|---:|---:|---:|"]
    def interval(item):
        return f"{100*item['mean']:+.4f} [{100*item['lower']:+.4f}, {100*item['upper']:+.4f}]"
    failures = []
    for model, panels in result["contrasts"].items():
        for window in ("near", "return"):
            item = panels[f"{window}_reapplied"]
            brier = item["value_minus_uniform"]["brier"]
            lines.append(f"| {model} | {window} | {interval(brier)} | {brier['negative']}/{brier['n']} | "
                f"{interval(item['value_minus_accuracy']['brier'])} | {interval(item['value_minus_uniform']['survival'])} |")
            failed = result["screen"][model][window]["failed"]
            failures.append(f"- {model}, {window}: failed criteria = {', '.join(failed) or 'none'}.")
    lines += ["", *failures, "", "## Every independent seed", "", "Value−uniform Brier ×100, after averaging both assessments.", "",
              "| Model | Seed | Local | Return |", "|---|---:|---:|---:|"]
    for model, panels in result["contrasts"].items():
        near, returning = (panels[f"{name}_reapplied"]["value_minus_uniform"]["brier"] for name in ("near", "return"))
        for seed, first, second in zip(near["seeds"], near["differences"], returning["differences"]):
            lines.append(f"| {model} | {seed} | {100*first:+.4f} | {100*second:+.4f} |")
    lines += ["", "## Descriptive controls and references", "",
        "The score-window result selects the winner and is not validation. The oracle uses validation Brier "
        "to choose one candidate; it is not deployable and does not optimize survival. The replacement is common "
        "to all candidates. Uniform averages candidate losses, not their predictions.", "",
        "| Model | Panel | Accuracy−uniform Brier | Value−replacement Brier | Oracle−uniform Brier | Value−oracle regret | Candidate spread |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for model, panels in result["contrasts"].items():
        for panel, comparisons in panels.items():
            values = [comparisons[name]["brier"]["mean"] for name in
                      ("accuracy_minus_uniform", "value_minus_replacement", "oracle_minus_uniform", "value_minus_oracle")]
            values.append(result["candidate_spread"][model][panel]["mean"])
            lines.append(f"| {model} | {panel} | "+" | ".join(f"{100*value:+.4f}" for value in values)+" |")
    lines += ["", "## Coverage and interpretation", ""]
    for model, coverage in result["coverage"].items():
        lines.append(f"- {model}: {coverage['with_return_candidate']}/{coverage['assessments']} assessments included "
                     f"a candidate from the exact returning law; all assessments remain in the primary estimate. "
                     f"{coverage['assessments_with_query_support_overlap']} assessments had candidate query/support overlap.")
    lines += ["", "Complete assessment rows, cue/gap/order slices, early/late losses, original prequential errors "
        "and steps, frozen-shadow comparisons, exact selector means and all seed differences are preserved in summary.json. "
        "Frozen versus reapplied contrasts combine changed parent weights, optimizer state and anchor; "
        "short versus long gaps combine learning exposure, arrival position and candidate age. Neither isolates elapsed time. "
        "This clean, known-schedule experiment does not establish noise discrimination, memory deletion value, "
        "longer physical planning or an improved deployed continual learner.", ""]
    return "\n".join(lines)


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    require(not output.resolve().is_relative_to(directory.resolve()), "summary output must be outside input runs")
    manifest, records, hashes = audit_records(directory)
    result = dict(identity=manifest["identity"], config=manifest["config"], counts=study_api().counts(manifest["config"]),
                  result_hashes=hashes, **analyze_records(records, manifest["config"]))
    result["resource_rows"] = []
    for key, record in sorted(records.items()):
        work = [item for phase in ("score", "near", "return") for item in record["work"][phase]]
        result["resource_rows"].append(dict(seed=key[0], model=key[1], index=key[2],
            diagnostic_rehearsals=len(work),
            rehearsal_totals={name: sum(item[name] for item in work) for name in
                ("steps", "forwards", "backward_calls", "query_presentations", "support_presentations", "training_seconds")},
            candidate_storage_bytes=record["candidate_storage_bytes"], metadata_storage_bytes=record["metadata_storage_bytes"],
            prediction_work=record["prediction_work"], copied_state_work=record["work"],
            scope="State tensor/storage sizes are explicit records, not measured peak RAM. "
                  "Uniform and original-error selectors would not need all this discarded work when deployed."))
    result["record_integrity"] = dict(passed=True, source_unchanged=True, analysis_lock=True,
        protocol_lock=True, configuration_identity=True, all_expected_assessments=len(records),
        scope="JSON/source/lock/coverage/selection validation; independent tensor audit is separate.")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", result), ("record_integrity.json", result["record_integrity"]),
                        ("config.json", manifest["config"])):
        write_json(output/name, value)
    for name in ("manifest.json", "completion.json", "training_source.zip", "protocol_lock.json", "protocol_at_lock.md",
                 "analysis_lock.json", "analysis_at_lock.zip"):
        shutil.copyfile(directory/name, output/name)
    with gzip.GzipFile(filename=str(output/"raw_results.jsonl.gz"), mode="wb", mtime=0) as stream:
        for key in sorted(records):
            stream.write((json.dumps(records[key], allow_nan=False)+"\n").encode("utf-8"))
    with gzip.GzipFile(filename=str(output/"phase_results.jsonl.gz"), mode="wb", mtime=0) as stream:
        for item in hashes:
            if item["kind"] == "phase":
                stream.write((json.dumps(dict(path=item["path"], record=read(directory/item["path"])), allow_nan=False)+"\n").encode("utf-8"))
    for kind in ("trajectory", "choice"):
        write_json(output/f"{kind}_records.json", {item["path"]: read(directory/item["path"])
                   for item in hashes if item["kind"] == kind})
    with (output/"rows.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in result["rows"]:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    (output/"summary.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    print(json.dumps(result["screen"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    summarize(arguments.input, arguments.output)
