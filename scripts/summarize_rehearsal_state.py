"""Prospective matched weight/Adam/anchor interventions on rehearsal usefulness."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import shutil

import numpy as np

from summarize_predictive_value import (
    PROBABILITY_METRIC_TOLERANCE, checked_zip, close_array, digest, finite_tree,
    mean, metric_array, paired, read, require, return_origin, sha, valid_hash, write_json,
    verify_phase,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(15001, 15013))
MODELS = ("conditional", "recurrent")
BRANCHES = ("current", "return")
CELLS = tuple("".join(bits) for bits in product("01", repeat=3))
FACTORS = ("weights", "optimizer", "anchor")
METRICS = ("brier", "survival", "early_brier", "late_brier")
SELECTORS = ("value", "accuracy", "uniform", "replacement", "oracle")
RESPONSES = ("selection", "chosen", "uniform")
CELL_CONTRASTS = {
    "value_minus_uniform": ("value", "uniform"),
    "value_minus_accuracy": ("value", "accuracy"),
    "accuracy_minus_uniform": ("accuracy", "uniform"),
    "value_minus_replacement": ("value", "replacement"),
    "uniform_minus_replacement": ("uniform", "replacement"),
    "oracle_minus_uniform": ("oracle", "uniform"),
    "value_minus_oracle": ("value", "oracle"),
}
SCIENTIFIC_SETTINGS = dict(kind="diagnostic", prefix_blocks=4, prefix_size=1024,
    acquisition_size=8192, inter_assessment_size=4096, score_size=512, validation_size=512,
    batch_size=32, candidate_count=8, rehearsal_updates=12, width=64, experts=4,
    context_width=12, decoder_width=64, interaction_features=True, lr=.002,
    memory_packets=16, updates_per_batch=12, evidence_strength=1.)
REQUIRED_ANALYSIS = ("scripts/summarize_rehearsal_state.py", "tests/test_rehearsal_state_summary.py",
                     "scripts/summarize_predictive_value.py")


def coefficients():
    """Full 2x2x2 contrasts; bit order is weights, complete Adam, causal anchor."""
    result = {"total_111_minus_000": {"111": 1., "000": -1.}}
    for axis, factor in enumerate(FACTORS):
        others = [i for i in range(3) if i != axis]
        result[f"{factor}_main"] = {cell: (.25 if cell[axis] == "1" else -.25) for cell in CELLS}
        for bits in product("01", repeat=2):
            old = ["0"]*3
            for other, bit in zip(others, bits):
                old[other] = bit
            new = old.copy()
            new[axis] = "1"
            name = f"{factor}_at_"+"_".join(f"{FACTORS[i]}{bit}" for i, bit in zip(others, bits))
            result[name] = {"".join(new): 1., "".join(old): -1.}
        repaired = list("111")
        repaired[axis] = "0"
        result[f"repair_{factor}"] = {"111": 1., "".join(repaired): -1.}
    for first, second in combinations(range(3), 2):
        result[f"{FACTORS[first]}_{FACTORS[second]}_interaction"] = {
            cell: .5*(1 if cell[first] == cell[second] else -1) for cell in CELLS}
    result["three_way_interaction"] = {cell: float((-1)**(3-sum(map(int, cell)))) for cell in CELLS}
    return result


EFFECTS = coefficients()


def rank_agreement(first, second):
    """Exact tied-average Spearman; constants remain explicitly undefined."""
    def ranks(values):
        values = np.asarray(values, dtype=float)
        require(values.ndim == 1 and len(values) > 1 and np.isfinite(values).all(), "invalid rank vector")
        result = np.empty(len(values), dtype=float)
        for value in np.unique(values):
            mask = values == value
            result[mask] = float((values < value).sum())+(int(mask.sum())+1)/2
        return result
    left, right = ranks(first), ranks(second)
    require(len(left) == len(right), "rank vectors have different candidates")
    left, right = left-left.mean(), right-right.mean()
    denominator = np.linalg.norm(left)*np.linalg.norm(right)
    return None if denominator == 0 else float(np.dot(left, right)/denominator)


def validate_assessment(record, count=None):
    finite_tree(record)
    seed, model, index = (record[name] for name in ("seed", "model", "index"))
    require(type(seed) is int and seed >= 0 and model in MODELS and type(index) is int and index in (0, 1),
            "invalid assessment key")
    require(record["cue"] == tuple(permutations(range(3)))[seed % 6][0], "incorrect acquisition cue")
    candidates = record["candidates"]
    require(isinstance(candidates, list) and len(candidates) >= 2 and (count is None or len(candidates) == count),
            "incorrect bounded candidate count")
    ids = []
    for candidate in [*candidates, record["replacement"]]:
        require(type(candidate["id"]) is int and candidate["id"] >= 0, "invalid candidate ID")
        ids.append(candidate["id"])
        metric_array([candidate["original_brier"]], 1, "original prequential Brier")
        require(type(candidate["original_steps"]) is int and candidate["original_steps"] >= 0,
                "invalid original optimizer step")
        valid_hash(candidate["query_sha256"])
        valid_hash(candidate["support_sha256"], optional=True)
        law = candidate["origin_law"]
        require(set(law) == {"mode", "active", "revised", "noise"}
                and law["mode"] in (0, 1) and isinstance(law["active"], (list, tuple))
                and all(type(cue) is int and cue in (0, 1, 2) for cue in law["active"])
                and type(law["revised"]) is bool and law["noise"] == 0., "invalid original source law")
    require(len(ids) == len(set(ids)), "duplicate candidate/replacement IDs")
    if "anchor_id" in record:
        require(record["anchor_id"] not in ids, "anchor included in candidate pool")
    count = len(candidates)
    choice = record["choice"]
    losses = metric_array(choice["score_losses"], count+1, "score-window Brier")
    values = losses[-1]-losses[:-1]
    close_array(choice["values"], values, "common-replacement values mismatch")
    selected = min(range(count), key=lambda i: (-values[i], ids[i]))
    accurate = min(range(count), key=lambda i: (candidates[i]["original_brier"], ids[i]))
    require(choice["value_id"] == ids[selected] and choice["accuracy_id"] == ids[accurate],
            "choice changed from the original score/original-error rule")
    require(set(record["panels"]) == set(record["zero"]) == set(BRANCHES), "incomplete paired branches")
    for branch in BRANCHES:
        require(set(record["panels"][branch]) == set(CELLS), "incomplete factorial cells")
        for cell in CELLS:
            for field in METRICS:
                metric_array(record["panels"][branch][cell][field], count+1, f"{branch}/{cell}/{field}")
        for field in METRICS:
            metric_array(record["zero"][branch][field], 2, f"{branch}/zero/{field}")
    return seed, model, index


def assessment_metrics(record):
    validate_assessment(record)
    candidates, choice = record["candidates"], record["choice"]
    ids = [item["id"] for item in candidates]
    count = len(candidates)
    chosen, accurate = ids.index(choice["value_id"]), ids.index(choice["accuracy_id"])
    panels, responses, effects = {}, {}, {}
    for branch in BRANCHES:
        panels[branch] = {}
        responses[branch] = {target: {} for target in RESPONSES}
        for cell in CELLS:
            raw = record["panels"][branch][cell]
            best = min(range(count), key=lambda i: (raw["brier"][i], ids[i]))
            selectors = {name: {} for name in SELECTORS}
            for field in METRICS:
                for selector, at in (("value", chosen), ("accuracy", accurate), ("replacement", count), ("oracle", best)):
                    selectors[selector][field] = float(raw[field][at])
                selectors["uniform"][field] = mean(raw[field][:-1])
            contrasts = {name: {field: selectors[first][field]-selectors[second][field] for field in METRICS}
                         for name, (first, second) in CELL_CONTRASTS.items()}
            zero_reference = int(cell[0])
            for selector in ("value", "accuracy", "uniform", "replacement"):
                contrasts[f"{selector}_minus_matching_zero"] = {
                    field: selectors[selector][field]-record["zero"][branch][field][zero_reference] for field in METRICS}
            panels[branch][cell] = dict(selectors=selectors, contrasts=contrasts, oracle_id=ids[best],
                brier_spread=float(max(raw["brier"][:-1])-min(raw["brier"][:-1])))
            responses[branch]["selection"][cell] = contrasts["value_minus_uniform"]
            responses[branch]["chosen"][cell] = selectors["value"]
            responses[branch]["uniform"][cell] = selectors["uniform"]
        effects[branch] = {target: {name: {field: float(sum(weight*responses[branch][target][cell][field]
            for cell, weight in weights.items())) for field in METRICS} for name, weights in EFFECTS.items()}
            for target in RESPONSES}
    zero = {branch: {str(weight): {field: record["zero"][branch][field][weight] for field in METRICS}
                    for weight in (0, 1)} for branch in BRANCHES}
    environment = {target: {cell: {field: responses["return"][target][cell][field]-responses["current"][target][cell][field]
                    for field in METRICS} for cell in CELLS} for target in RESPONSES}
    coverage = [return_origin(item["origin_law"], record["seed"]) for item in candidates]
    row = {name: record[name] for name in ("seed", "model", "index", "cue")}
    row.update(candidate_ids=ids, replacement_id=record["replacement"]["id"],
        value_id=choice["value_id"], accuracy_id=choice["accuracy_id"],
        panels=panels, effects=effects, zero=zero, environment=environment,
        return_candidates=sum(coverage), value_return_origin=bool(coverage[chosen]),
        accuracy_return_origin=bool(coverage[accurate]), selectors_agree=chosen == accurate,
        original_brier={"value": candidates[chosen]["original_brier"], "accuracy": candidates[accurate]["original_brier"],
                        "uniform": mean(item["original_brier"] for item in candidates)},
        original_steps={"value": candidates[chosen]["original_steps"], "accuracy": candidates[accurate]["original_steps"],
                        "uniform": mean(item["original_steps"] for item in candidates)},
        query_support_overlaps=[[a["id"], b["id"]] for a in candidates for b in candidates
            if a["id"] != b["id"] and a["query_sha256"] == b["support_sha256"]])
    row["rank_agreement"] = dict(
        score_to_validation={branch: {cell: rank_agreement(choice["score_losses"][:-1],
            record["panels"][branch][cell]["brier"][:-1]) for cell in CELLS} for branch in BRANCHES},
        all_old_to_cell={branch: {cell: rank_agreement(record["panels"][branch]["000"]["brier"][:-1],
            record["panels"][branch][cell]["brier"][:-1]) for cell in CELLS} for branch in BRANCHES},
        current_to_return={cell: rank_agreement(record["panels"]["current"][cell]["brier"][:-1],
            record["panels"]["return"][cell]["brier"][:-1]) for cell in CELLS})
    if "anchor_id" in record:
        row["age_packets"] = {"value": record["anchor_id"]-ids[chosen], "accuracy": record["anchor_id"]-ids[accurate],
                              "uniform": mean(record["anchor_id"]-item["id"] for item in candidates)}
    return row


def eligible_config(config):
    return (config.get("seeds") == SEEDS and set(config.get("models", [])) == set(MODELS)
            and all(config.get(name) == value for name, value in SCIENTIFIC_SETTINGS.items()))


def replication_screen(total, eligible=True):
    positive = sum(value > 0 for value in total["differences"])
    gates = dict(complete_scientific_cohort=dict(passed=bool(eligible)),
        deterioration=dict(passed=total["mean"] >= .0005, value=total["mean"], minimum=.0005),
        seed_consistency=dict(passed=positive >= 9, positive_seeds=positive, minimum=9, n=total["n"]))
    return dict(passed=all(gate["passed"] for gate in gates.values()), gates=gates,
        failed=[name for name, gate in gates.items() if not gate["passed"]],
        interpretation="Practical replication screen for current-branch D111−D000; not a significance test, "
                       "policy gate or permission to promote a counterfactual hybrid.")


def analyze_records(records, config):
    if isinstance(records, dict):
        records = list(records.values())
    records = list(records)
    require(len(set(config["seeds"])) == len(config["seeds"])
            and len(set(config["models"])) == len(config["models"]), "duplicate configured seed/model")
    keys = [validate_assessment(record, config.get("candidate_count")) for record in records]
    expected = {(seed, model, index) for seed in config["seeds"] for model in config["models"] for index in (0, 1)}
    require(len(keys) == len(set(keys)) and set(keys) == expected, "both assessments of every seed/model are required exactly once")
    rows = sorted([assessment_metrics(record) for record in records], key=lambda row: (row["model"], row["seed"], row["index"]))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["seed"]].append(row)
    panels, effects, zeros, environment, coverage, slices, ranks = {}, {}, {}, {}, {}, {}, {}
    for model in config["models"]:
        def aggregate(getter):
            return paired([mean(getter(row) for row in grouped[model, seed]) for seed in config["seeds"]], config["seeds"])
        panels[model] = {}
        for branch in BRANCHES:
            panels[model][branch] = {}
            for cell in CELLS:
                example = grouped[model, config["seeds"][0]][0]["panels"][branch][cell]
                panels[model][branch][cell] = {
                    section: {name: {field: aggregate(lambda row, field=field, name=name, section=section:
                        row["panels"][branch][cell][section][name][field]) for field in METRICS}
                        for name in example[section]} for section in ("selectors", "contrasts")}
                panels[model][branch][cell]["brier_spread"] = aggregate(lambda row: row["panels"][branch][cell]["brier_spread"])
        effects[model] = {branch: {target: {name: {field: aggregate(lambda row, branch=branch, target=target,
            name=name, field=field: row["effects"][branch][target][name][field]) for field in METRICS}
            for name in EFFECTS} for target in RESPONSES} for branch in BRANCHES}
        zeros[model] = {branch: {weight: {field: aggregate(lambda row, branch=branch, weight=weight,
            field=field: row["zero"][branch][weight][field]) for field in METRICS} for weight in ("0", "1")}
            for branch in BRANCHES}
        environment[model] = {target: {cell: {field: aggregate(lambda row, target=target, cell=cell,
            field=field: row["environment"][target][cell][field]) for field in METRICS} for cell in CELLS}
            for target in RESPONSES}
        model_rows = [row for row in rows if row["model"] == model]
        def rank_summary(getter):
            values = [[getter(row) for row in grouped[model, seed]] for seed in config["seeds"]]
            seed_values = [None if any(value is None for value in pair) else mean(pair) for pair in values]
            return dict(mean=None if any(value is None for value in seed_values) else mean(seed_values),
                seeds=config["seeds"], seed_values=seed_values, n=len(seed_values),
                undefined_assessments=sum(value is None for pair in values for value in pair),
                positive_seeds=sum(value is not None and value > 0 for value in seed_values))
        ranks[model] = {name: {branch: {cell: rank_summary(lambda row, name=name, branch=branch, cell=cell:
            row["rank_agreement"][name][branch][cell]) for cell in CELLS} for branch in BRANCHES}
            for name in ("score_to_validation", "all_old_to_cell")}
        ranks[model]["current_to_return"] = {cell: rank_summary(lambda row, cell=cell:
            row["rank_agreement"]["current_to_return"][cell]) for cell in CELLS}
        coverage[model] = dict(assessments=len(model_rows), with_return_candidate=sum(row["return_candidates"] > 0 for row in model_rows),
            without_return_candidate=sum(row["return_candidates"] == 0 for row in model_rows),
            mean_return_candidates=mean(row["return_candidates"] for row in model_rows),
            value_return_fraction=mean(row["value_return_origin"] for row in model_rows),
            accuracy_return_fraction=mean(row["accuracy_return_origin"] for row in model_rows),
            selector_agreement_fraction=mean(row["selectors_agree"] for row in model_rows),
            assessments_with_query_support_overlap=sum(bool(row["query_support_overlaps"]) for row in model_rows))
        slices[model] = {}
        for group in ("cue", "index"):
            slices[model][group] = {}
            for label in sorted({row[group] for row in model_rows}):
                selected = [row for row in model_rows if row[group] == label]
                seeds = [seed for seed in config["seeds"] if any(row["seed"] == seed for row in selected)]
                slices[model][group][str(label)] = dict(n=len(seeds), seeds=seeds, assessments=len(selected),
                    effects={branch: {name: {field: mean(mean(row["effects"][branch]["selection"][name][field]
                        for row in selected if row["seed"] == seed) for seed in seeds) for field in METRICS}
                        for name in EFFECTS} for branch in BRANCHES})
    eligible = eligible_config(config)
    return dict(rows=rows, panels=panels, effects=effects, zero=zeros, environment=environment, rank_agreement=ranks,
        coverage=coverage, descriptive_slices=slices, eligible_scientific_cohort=eligible,
        screen={model: replication_screen(effects[model]["current"]["selection"]["total_111_minus_000"]["brier"], eligible)
                for model in config["models"]}, coefficients=EFFECTS,
        inference=dict(unit="Independent seed; two assessments averaged before bootstrap", bootstrap_seed=27192026,
            bootstrap_resamples=20000, intervals="Descriptive unadjusted percentile 95%",
            bit_order=list(FACTORS), primary="Current branch, (chosen−uniform)111−(chosen−uniform)000",
            repair_sign="111 minus the cell that restores one component to old; positive Brier means restoration lowers error.",
            environment_sign="Return minus current law with branch-specific evolving feedback, initially identical history.",
            absolute_stats="The differences array for selector and zero metrics contains absolute seed values.",
            oracle="Validation-full-Brier-selected single candidate; all other oracle outcomes use that same candidate.",
            rank_agreement="Tied-average Spearman; average assessments then seeds. A constant comparison makes its seed "
                "and cohort rank mean undefined; no assessment is dropped and primary metrics are unaffected.",
            limits="Counterfactual state substitutions, not natural on-policy contributions, unique blame shares or a retention policy."))


def study_api():
    from acp_cl.rehearsal_state import study
    return study


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate NPZ keys")
        return {name: archive[name] for name in archive.files}


def experience_hash(observations, actions, outcomes):
    result = hashlib.sha256()
    for array in (observations, actions, outcomes):
        result.update(str((array.shape, array.dtype.str)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def validate_arrays(arrays, config):
    size, packets = config["batch_size"], config["validation_size"]//config["batch_size"]
    require(set(arrays) == {"observations", "actions", "outcomes", "truth", "initial_observations",
                           "initial_actions", "initial_outcomes"}, "diagnostic data NPZ schema mismatch")
    require(packets >= 2 and arrays["observations"].shape[:2] == (packets, size)
            and arrays["initial_observations"].shape == arrays["observations"].shape[1:]
            and arrays["actions"].shape == (packets, size) and arrays["initial_actions"].shape == (size,)
            and arrays["outcomes"].shape == (2, packets, size, 3)
            and arrays["truth"].shape == (2, packets, size, 5, 3)
            and arrays["initial_outcomes"].shape == (size, 3), "diagnostic paired data shapes mismatch")
    require(all(np.issubdtype(value.dtype, np.number) and np.isfinite(value).all() for value in arrays.values()),
            "invalid/nonfinite diagnostic data array")
    for field in ("actions", "initial_actions"):
        require(np.issubdtype(arrays[field].dtype, np.integer)
                and np.all(arrays[field] >= 0) and np.all(arrays[field] < 5), "invalid performed actions")
    for field in ("outcomes", "truth", "initial_outcomes"):
        require(np.all(arrays[field] >= 0) and np.all(arrays[field] <= 1), "invalid outcome/truth range")
    for branch in range(2):
        for packet in range(packets):
            require(np.array_equal(arrays["outcomes"][branch, packet],
                arrays["truth"][branch, packet, np.arange(size), arrays["actions"][packet]]),
                "clean reported feedback differs from counterfactual truth")


def metric_panels(probabilities, arrays, candidates):
    """Independently recompute panel arithmetic from complete stored probabilities."""
    packets, size = arrays["actions"].shape
    require(probabilities.shape == (2, packets, candidates, size, 5, 3)
            and np.issubdtype(probabilities.dtype, np.floating) and np.isfinite(probabilities).all()
            and np.all(probabilities >= -PROBABILITY_METRIC_TOLERANCE)
            and np.all(probabilities <= 1+PROBABILITY_METRIC_TOLERANCE), "invalid diagnostic probability arrays")
    result = {}
    for branch_index, branch in enumerate(BRANCHES):
        losses, survivals = [], []
        for packet in range(packets):
            predicted = probabilities[branch_index, packet]
            performed = predicted[:, np.arange(size), arrays["actions"][packet]].astype(float)
            losses.append(np.square(performed-arrays["outcomes"][branch_index, packet]).mean(axis=(1, 2)))
            chosen = predicted[..., -1].argmax(axis=-1)
            survivals.append(arrays["truth"][branch_index, packet, np.arange(size)[None], chosen, -1].mean(axis=1))
        losses, survivals = np.asarray(losses), np.asarray(survivals)
        result[branch] = dict(brier=losses.mean(axis=0).tolist(), survival=survivals.mean(axis=0).tolist(),
            early_brier=losses[0].tolist(), late_brier=losses[1:].mean(axis=0).tolist())
    return result


def verify_analysis_lock(directory, config, protocol_sha):
    lock = read(directory/"analysis_lock.json")
    require(lock["config"] == config and lock["config_sha256"] == digest(config), "analysis-locked config mismatch")
    require(set(REQUIRED_ANALYSIS) <= lock["files"].keys(), "required analysis absent from prospective lock")
    checked_zip(directory/"analysis_at_lock.zip", lock["files"])
    for name, expected in lock["files"].items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file() and sha(path) == expected,
                f"current analysis differs from lock: {name}")
    require(any(name.startswith("docs/") and value == protocol_sha for name, value in lock["files"].items()),
            "protocol absent from analysis lock")


def audit_records(directory):
    directory = Path(directory)
    study = study_api()
    manifest = study.check_locks(directory)
    config, identity = manifest["config"], manifest["identity"]
    study.validate_config(config)
    verify_analysis_lock(directory, config, read(directory/"protocol_lock.json")["protocol_sha256"])
    completion = read(directory/"completion.json")
    require(completion["identity"] == identity and all(completion[key] == value for key, value in study.counts(config).items()),
            "incomplete scientific cohort")
    pairs = {(seed, model) for seed in config["seeds"] for model in config["models"]}
    jobs = completion["jobs"]
    require(len(jobs) == len(pairs) and {(job["seed"], job["model"]) for job in jobs} == pairs
            and all(job["affinity"] == manifest["runtime"]["affinity"] for job in jobs), "worker coverage/runtime mismatch")
    records, hashed, result_paths, auxiliary = {}, {}, set(), set()
    def add_hash(path, kind, expected=None):
        value = sha(path)
        require(expected is None or value == expected, f"artifact hash mismatch: {path}")
        relative = path.relative_to(directory).as_posix()
        if relative not in hashed:
            hashed[relative] = dict(path=relative, sha256=value, kind=kind)
        require(hashed[relative]["sha256"] == value, "input changed during audit")
        if path.name == "result.json":
            result_paths.add(path)
        return value
    for seed, model in sorted(pairs):
        folder = directory/f"{model}_{seed}"
        trajectory_path = folder/"result.json"
        trajectory = read(trajectory_path)
        require(trajectory["identity"] == identity and (trajectory["seed"], trajectory["model"]) == (seed, model)
                and trajectory["affinity"] == manifest["runtime"]["affinity"], "trajectory identity/runtime mismatch")
        schedule = study.schedule(seed, config)
        names = [phase["name"] for phase in schedule]
        require(trajectory["phase_names"] == names and set(trajectory["phase_hashes"]) == set(names)
                and set(trajectory["assessment_hashes"]) == {"0", "1"}, "trajectory schedule/assessment coverage mismatch")
        add_hash(trajectory_path, "trajectory")
        phases, all_packets, seen, previous_signature, previous_query = {}, {}, 0, None, None
        for phase in schedule:
            path = folder/phase["name"]/"result.json"
            phase_record = read(path)
            add_hash(path, "phase", trajectory["phase_hashes"][phase["name"]])
            verify_phase(phase_record, phase, seed, model, config, identity, seen)
            require(previous_signature is None or phase_record["before_signature"] == previous_signature,
                    "ordinary state chain mismatch")
            require(phase_record["packets"][0]["support_sha256"] == previous_query, "ordinary causal history mismatch")
            for packet in phase_record["packets"]:
                all_packets[packet["id"]] = packet
            seen += len(phase_record["packets"])
            previous_signature, previous_query = phase_record["final_signature"], phase_record["packets"][-1]["query_sha256"]
            phases[phase["name"]] = phase_record
            for name in ("before.pt", "checkpoint.pt"):
                require((path.parent/name).is_file(), "missing ordinary checkpoint")
        require(trajectory["final_signature"] == previous_signature
                and trajectory["diagnostics"] == phases[names[-1]]["diagnostics"], "trajectory endpoint mismatch")
        require(trajectory["prequential_work"] == dict(forwards=seen, query_presentations=seen*config["batch_size"],
                support_presentations=seen*config["batch_size"]), "prequential forward budget mismatch")
        for index in (0, 1):
            path = folder/f"diagnostic_{index}"/"result.json"
            record = read(path)
            key = validate_assessment(record, config["candidate_count"])
            require(key == (seed, model, index) and record["identity"] == identity and key not in records,
                    "diagnostic identity/key mismatch")
            add_hash(path, "assessment", trajectory["assessment_hashes"][str(index)])
            score = phases[f"score_{index}"]
            require(record["parents"] == dict(old=score["before_signature"], new=score["final_signature"])
                    and record["old_steps"] == score["start_diagnostics"]["cost"]["optimizer_steps"]
                    and record["new_steps"] == score["diagnostics"]["cost"]["optimizer_steps"]
                    and record["anchor_id"] == score["start_memory"]["seen"]-1
                    and record["reservoir_ids"] == score["start_memory"]["ids"], "old/new cutoff or candidate provenance mismatch")
            ids, replacement = study.previous.candidate_ids(record["reservoir_ids"], record["anchor_id"],
                                                            config["candidate_count"], seed, index)
            require([item["id"] for item in record["candidates"]] == ids and record["replacement"]["id"] == replacement,
                    "candidate draw differs from fixed uniform sampling")
            for candidate in [*record["candidates"], record["replacement"]]:
                original = all_packets[candidate["id"]]
                require(candidate == {name: value for name, value in original.items()
                        if name not in {"probabilities", "actions", "outcomes"}}, "candidate original prediction provenance mismatch")
            choice_path = folder/f"assessment_{index}"/"choice.json"
            choice = read(choice_path)
            require(choice["identity"] == identity and (choice["seed"], choice["model"], choice["index"]) == key
                    and choice["score_record_sha256"] == digest(score) and choice["choice"] == record["choice"],
                    "score-only selection seal mismatch")
            close_array(record["choice"]["score_losses"], score["score_losses"], "selection score losses mismatch")
            required_artifacts = {f"diagnostic_{index}/before.pt", f"diagnostic_{index}/data.npz", f"diagnostic_{index}/stream.json",
                f"assessment_{index}/candidates.pt", f"assessment_{index}/choice.json", f"score_{index}/result.json",
                f"score_{index}/before.pt", f"score_{index}/shadows.pt", f"diagnostic_{index}/zero.pt"}
            required_artifacts |= {f"diagnostic_{index}/cell_{cell}.pt" for cell in CELLS if cell != "000"}
            required_artifacts |= {f"diagnostic_{index}/predictions_{cell}.{suffix}" for cell in (*CELLS, "zero") for suffix in ("npz", "json")}
            require(set(record["artifact_hashes"]) == required_artifacts, "diagnostic artifact set mismatch")
            for relative, expected in record["artifact_hashes"].items():
                artifact = (folder/relative).resolve()
                require(artifact.is_relative_to(folder.resolve()), "artifact escapes its trajectory")
                # Keep paths relative to the original input spelling for exports.
                artifact = folder/relative
                kind = "array" if artifact.suffix == ".npz" else "checkpoint" if artifact.suffix == ".pt" else "auxiliary"
                add_hash(artifact, kind, expected)
                if kind == "auxiliary" and artifact.name != "result.json":
                    auxiliary.add(artifact)
            data = read_arrays(path.parent/"data.npz")
            validate_arrays(data, config)
            stream = read(path.parent/"stream.json")
            require(stream["identity"] == identity and (stream["seed"], stream["model"], stream["index"]) == key
                    and stream["branches"] == list(BRANCHES) and len(stream["rows"]) == len(data["actions"]),
                    "paired branch stream provenance mismatch")
            initial_hash = experience_hash(data["initial_observations"], data["initial_actions"], data["initial_outcomes"])
            require(initial_hash == score["packets"][-1]["query_sha256"], "branch initial history is not actual cutoff history")
            branch_previous = [initial_hash, initial_hash]
            for packet, stream_row in enumerate(stream["rows"]):
                require(set(stream_row) == set(BRANCHES), "stream branch coverage mismatch")
                for b, branch in enumerate(BRANCHES):
                    query = experience_hash(data["observations"][packet], data["actions"][packet], data["outcomes"][b, packet])
                    require(stream_row[branch] == dict(query_sha256=query, support_sha256=branch_previous[b]),
                            "branch-specific causal support chain mismatch")
                    branch_previous[b] = query
                    if branch == "current":
                        ordinary = phases[f"validation_{index}"]["packets"][packet]
                        require(ordinary["query_sha256"] == query and ordinary["support_sha256"] == stream_row[branch]["support_sha256"],
                                "ordinary validation differs from the current diagnostic branch")
            require(set(record["state_files"]) == set(record["prediction_files"]) == set(record["active_signatures"]) == {*CELLS, "zero"}
                    and set(record["work"]) == set(CELLS), "state/work/prediction family coverage mismatch")
            for cell in (*CELLS, "zero"):
                count = 2 if cell == "zero" else config["candidate_count"]+1
                prediction_path = path.parent/f"predictions_{cell}.npz"
                prediction = read_arrays(prediction_path)
                require(set(prediction) == {"probabilities"}, "prediction NPZ schema mismatch")
                actual = metric_panels(prediction["probabilities"], data, count)
                for branch in BRANCHES:
                    expected = record["zero"][branch] if cell == "zero" else record["panels"][branch][cell]
                    for field in METRICS:
                        close_array(expected[field], actual[branch][field], "recorded factorial/zero metrics differ from raw arrays")
                seal = read(path.parent/f"predictions_{cell}.json")
                require(seal["identity"] == identity and seal["parents"] == record["parents"]
                        and seal["active_signatures"] == record["active_signatures"][cell]
                        and seal["data_sha256"] == sha(path.parent/"data.npz")
                        and seal["predictions_sha256"] == sha(prediction_path), "prediction state/data seal mismatch")
                require(len(record["active_signatures"][cell]) == count, "active state count mismatch")
                for value in record["active_signatures"][cell]:
                    valid_hash(value)
                expected_state = f"score_{index}/shadows.pt" if cell == "000" else f"diagnostic_{index}/zero.pt" if cell == "zero" else f"diagnostic_{index}/cell_{cell}.pt"
                require(record["state_files"][cell] == dict(path=expected_state, sha256=record["artifact_hashes"][expected_state])
                        and record["prediction_files"][cell] == dict(path=prediction_path.relative_to(folder).as_posix(), sha256=sha(prediction_path)),
                        "state/prediction file mapping mismatch")
                if cell != "zero":
                    work = record["work"][cell]
                    require(len(work) == count, "rehearsal work count mismatch")
                    if cell == "000":
                        require(work == score["shadow_work"], "all-old family was not reused from scoring")
                    for item in work:
                        updates, size = config["rehearsal_updates"], config["batch_size"]
                        require(item["steps"] == item["backward_calls"] == updates and item["forwards"] == 2*updates
                                and item["query_presentations"] == item["support_presentations"] == 2*updates*size
                                and item["parent_unchanged"] and item["memory_history_unchanged"], "rehearsal budget/state mismatch")
                        if cell != "000":
                            require(item["cell"] == cell and [item[name+"_bit"] for name in FACTORS] == list(map(int, cell))
                                    and item["old_donor_sha256"] == record["parents"]["old"]
                                    and item["new_donor_sha256"] == record["parents"]["new"]
                                    and item["diagnostic_forwards"] == 4
                                    and item["diagnostic_query_presentations"] == item["diagnostic_support_presentations"] == 4*size
                                    and item["donors_unchanged"] and item["inactive_unchanged"] and item["update_l2"] >= 0,
                                    "factorial donor/instrumentation mismatch")
            size, count = config["batch_size"], config["candidate_count"]+1
            require(record["prediction_work"] == dict(score_forwards=count*config["score_size"]//size,
                factorial_forwards=2*config["validation_size"]//size*8*count,
                zero_forwards=4*config["validation_size"]//size), "diagnostic prediction work mismatch")
            records[key] = record
    require(set(directory.glob("**/result.json")) == result_paths, "unexpected result records in cohort")
    # Every portable auxiliary record must be finite and JSON-readable.
    for path in auxiliary:
        read(path)
    return manifest, records, sorted(hashed.values(), key=lambda item: item["path"])


def markdown(result):
    lines = ["# Rehearsal state: controlled weight, Adam and anchor substitutions", "",
        "Cell bits are weights, complete Adam state and the complete causal anchor packet (0 old, 1 new). "
        "All cells use the same retained candidates and selected IDs. D is selected-candidate loss minus the exact "
        "mean candidate loss. The primary response is current-branch T=D111−D000, with two assessments averaged per seed.", "",
        "The replication screen requires mean T≥0.0005 and positive T in at least 9/12 seeds. It is a research "
        "allocation screen, not a significance test or policy promotion. All intervals are descriptive seed-bootstrap "
        "95% intervals using 20,000 resamples. Zero-update references are outside the candidate pool.", "",
        "| Model | T Brier ×100 [95% interval] | Positive seeds | Replication screen |", "|---|---:|---:|---|"]
    def interval(item):
        return f"{100*item['mean']:+.4f} [{100*item['lower']:+.4f}, {100*item['upper']:+.4f}]"
    for model, screen in result["screen"].items():
        value = result["effects"][model]["current"]["selection"]["total_111_minus_000"]["brier"]
        lines.append(f"| {model} | {interval(value)} | {value['positive']}/{value['n']} | {'PASS' if screen['passed'] else 'FAIL'} |")
    lines += [""]
    if not result["eligible_scientific_cohort"]:
        lines += ["Engineering/subset cohort: ineligible for the scientific replication screen.", ""]
    for model, screen in result["screen"].items():
        lines.append(f"- {model}: failed criteria = {', '.join(screen['failed']) or 'none'}.")
    lines += ["", "## Every cell", "", "Brier ×100 and survival %. Oracle is conditional on validation Brier, "
        "not deployable and not independently optimized for survival.", "",
        "| Model | Branch | Cell | Chosen Brier | Original-error Brier | Uniform Brier | Replacement Brier | Oracle Brier | D [interval] | Chosen survival | Uniform survival |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, branches in result["panels"].items():
        for branch, cells in branches.items():
            for cell, item in cells.items():
                values = [100*item["selectors"][name]["brier"]["mean"] for name in SELECTORS]
                survival = [100*item["selectors"][name]["survival"]["mean"] for name in ("value", "uniform")]
                lines.append(f"| {model} | {branch} | {cell} | "+" | ".join(f"{x:.4f}" for x in values)
                    +f" | {interval(item['contrasts']['value_minus_uniform']['brier'])} | {survival[0]:.3f} | {survival[1]:.3f} |")
    lines += ["", "## Factorial effects and prespecified repairs", "",
        "Positive D effects mean worse selected-versus-uniform usefulness. Positive chosen-loss repair effects mean "
        "the old-component repair reduces chosen loss; positive survival effects instead favor cell 111. "
        "Main effects average over the other two factors. Interactions prevent unique component blame.", "",
        "| Model | Branch | Effect | D Brier ×100 [interval] | Chosen Brier change ×100 [interval] | Chosen survival change, pp [interval] |",
        "|---|---|---|---:|---:|---:|"]
    for model, branches in result["effects"].items():
        for branch, targets in branches.items():
            for name in EFFECTS:
                lines.append(f"| {model} | {branch} | {name} | {interval(targets['selection'][name]['brier'])} "
                    f"| {interval(targets['chosen'][name]['brier'])} | {interval(targets['chosen'][name]['survival'])} |")
    lines += ["", "## Every independent seed", "", "Current-branch T, Brier ×100.", "",
              "| Model | Seed | T |", "|---|---:|---:|"]
    for model, branches in result["effects"].items():
        stats = branches["current"]["selection"]["total_111_minus_000"]["brier"]
        for seed, value in zip(stats["seeds"], stats["differences"]):
            lines.append(f"| {model} | {seed} | {100*value:+.4f} |")
    lines += ["", "All absolute/early/late/survival metrics, exact uniform expectation, zero-update references, "
        "environment contrasts, candidate coverage and cue/assessment slices are preserved in summary.json. "
        "The two branches begin with the same actual history and then receive their own causal feedback. "
        "Branch contrasts therefore include changed outcomes and evolving context. Every hybrid is a declared "
        "counterfactual substitution; these results do not establish an on-policy natural component effect, "
        "adaptive memory policy, elapsed-time mechanism or general continual-learning improvement. "
        "No largest-factor winner or policy gate is selected.", ""]
    return "\n".join(lines)


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    require(not output.resolve().is_relative_to(directory.resolve()), "output must be outside the run directory")
    manifest, records, hashes = audit_records(directory)
    result = dict(identity=manifest["identity"], config=manifest["config"], counts=study_api().counts(manifest["config"]),
                  result_hashes=hashes, **analyze_records(records, manifest["config"]))
    result["array_archive_prefix"] = "arrays"
    result["resource_rows"] = [dict(seed=key[0], model=key[1], index=key[2],
        candidate_storage_bytes=record["candidate_storage_bytes"], prediction_work=record["prediction_work"],
        cell_work=record["work"],
        scope="Complete explicit rehearsal and four-forward instrumentation counters. Copied-state bytes are not peak RAM.")
        for key, record in sorted(records.items())]
    result["record_integrity"] = dict(passed=True, assessments=len(records), source_lock=True, analysis_lock=True,
        protocol_lock=True, scope="Structural/array/metric/provenance audit; independent tensor reconstruction is separate.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"summary.json", result)
    write_json(output/"config.json", manifest["config"])
    write_json(output/"record_integrity.json", result["record_integrity"])
    for name in ("manifest.json", "completion.json", "training_source.zip", "protocol_lock.json", "protocol_at_lock.md",
                 "analysis_lock.json", "analysis_at_lock.zip"):
        shutil.copyfile(directory/name, output/name)
    with gzip.GzipFile(filename=str(output/"raw_results.jsonl.gz"), mode="wb", mtime=0) as stream:
        for key in sorted(records):
            stream.write((json.dumps(records[key], allow_nan=False)+"\n").encode())
    with gzip.GzipFile(filename=str(output/"phase_results.jsonl.gz"), mode="wb", mtime=0) as stream:
        for item in hashes:
            if item["kind"] == "phase":
                stream.write((json.dumps(dict(path=item["path"], record=read(directory/item["path"])), allow_nan=False)+"\n").encode())
    for kind in ("trajectory", "auxiliary"):
        write_json(output/f"{kind}_records.json", {item["path"]: read(directory/item["path"])
                   for item in hashes if item["kind"] == kind})
    with (output/"rows.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in result["rows"]:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    for item in hashes:
        if item["kind"] == "array":
            destination = output/result["array_archive_prefix"]/item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory/item["path"], destination)
            require(sha(destination) == item["sha256"], "exported prediction/data array hash mismatch")
    (output/"summary.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    print(json.dumps(result["screen"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
