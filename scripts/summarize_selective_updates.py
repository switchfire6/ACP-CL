"""Fixed first-introduction development contrasts, qualification and allocation gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

from summarize_replay_renewal import qualification
from summarize_training_state import mean, metrics, paired
from acp_cl.acquisition.world import Law, order, stage_law
from acp_cl.persistence.study import digest
from acp_cl.replay_renewal.study import write_json
from acp_cl.selective_updates.learner import ARMS, ARM_SETTINGS, COUNTERS, SUMMARY_NAMES, new_stats


ROOT = Path(__file__).resolve().parents[1]
ALL_ARMS = (*ARMS, "fresh")
CONTRASTS = {
    "current_minus_reference": {"current": 1, "reference": -1},
    "combined_minus_protected": {"combined": 1, "protected": -1},
    "protected_minus_reference": {"protected": 1, "reference": -1},
    "combined_minus_current": {"combined": 1, "current": -1},
    "current_main": {"current": .5, "reference": -.5, "combined": .5, "protected": -.5},
    "protection_main": {"protected": .5, "reference": -.5, "combined": .5, "current": -.5},
    "interaction": {"combined": 1, "current": -1, "protected": -1, "reference": 1},
    "combined_minus_reference": {"combined": 1, "reference": -1},
    "combined_minus_shrink": {"combined": 1, "shrink": -1},
    "shrink_minus_current": {"shrink": 1, "current": -1},
    "reference_minus_fresh": {"reference": 1, "fresh": -1},
    "combined_minus_fresh": {"combined": 1, "fresh": -1},
}
FIELDS = ("brier_auc", "all_brier_auc", "valid_after", "valid_after_mode_0", "valid_after_mode_1",
          "valid_damage", "valid_damage_mode_0", "valid_damage_mode_1", "after", "endpoint",
          "survival_auc", "cue_effect", "cue_learning_gain", "marginal_gain")
PROBE_FIELDS = ("brier", "focus_brier", "survival", "focus_survival", "valid_brier", "valid_survival",
                "no_transfer", "clairvoyant_upper")
REQUIRED_ANALYSIS = ("scripts/summarize_selective_updates.py", "tests/test_selective_update_summary.py",
                     "scripts/summarize_training_state.py", "scripts/summarize_replay_renewal.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite value in record")


def read(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    finite_tree(result)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def study_api():
    from acp_cl.selective_updates import study
    return study


def checked_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected) and archive.testzip() is None,
                "source/analysis archive file set or CRC mismatch")
        for name, expected_hash in expected.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected_hash, "source/analysis archive hash mismatch")


def verify_analysis_lock(directory, config, protocol_sha):
    lock = read(directory/"analysis_lock.json")
    files = lock["files"]
    require(lock["config"] == config and lock["config_sha256"] == digest(config),
            "configuration differs from analysis lock")
    require(set(REQUIRED_ANALYSIS) <= files.keys(), "required prospective analysis files missing from lock")
    checked_zip(directory/"analysis_at_lock.zip", files)
    for name, expected in files.items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file() and sha(path) == expected,
                f"current analysis/configuration/protocol differs from lock: {name}")
    require(any(name.startswith("docs/") and expected == protocol_sha for name, expected in files.items()),
            "protocol missing from analysis lock")
    return lock


def valid_hashes(values):
    require(all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                for value in values), "invalid SHA256 in record")


def check_probe_metrics(value, marginal=False):
    for field in (*PROBE_FIELDS, *(("marginal_brier",) if marginal else ())):
        score = value[field]
        require(type(score) in (int, float) and 0 <= score <= 1.000001, "invalid probe metric")


def check_probe(probe, law, model_hash, config, marginal=False):
    require(digest(probe["law"]) == digest(asdict(law)) and probe["model_sha256"] == model_hash,
            "probe/model/law mismatch")
    require(len(probe["replicates"]) == config["support_replicates"], "incomplete support replicates")
    size = config["batch_size"]
    for row in probe["replicates"]:
        require([point["feedback"] for point in row["curve"]] == sorted(set((0, size//4, size//2, size))),
                "incomplete support-refresh probe")
        for point in row["curve"]:
            check_probe_metrics(point["metrics"], marginal)
        # Evaluator.probe deliberately omits marginal predictions from its
        # flipped-cue diagnostic, even when ordinary support probes include them.
        check_probe_metrics(row["flipped"], marginal=False)


def check_memory(memory, config):
    ids = memory["ids"]
    require(memory["type"] == "ReservoirMemory" and memory["capacity"] == config["memory_packets"],
            "intervention changed the uniform memory policy or capacity")
    require(memory["seen"] == memory["age"] and len(ids) == min(memory["seen"], config["memory_packets"])
            and len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < memory["seen"] for i in ids),
            "invalid reservoir age or membership")
    require(len(memory["packets"]) == len(ids), "memory packet/ID mismatch")
    for packet in memory["packets"]:
        require(packet["hidden"] is False, "hidden labels in replay")
        valid_hashes([packet["query"], *([] if packet["support"] is None else [packet["support"]])])


def check_selective(before, after, arm, config):
    internal_arm = "reference" if arm == "fresh" else arm
    weight, mode = ARM_SETTINGS[internal_arm]
    for item in (before, after):
        require(item["arm"] == internal_arm and item["current_weight"] == weight and item["mode"] == mode,
                "wrong selective-update rule")
    require(before["stats"] == new_stats(), "selective counters were not reset at the experimental fork")
    stats = after["stats"]
    packets = config["episode_size"]//config["batch_size"]
    steps, size = packets*config["updates_per_batch"], config["batch_size"]
    expected = dict(steps=steps, packets=packets, training_probability_calls=2*steps,
        reference_gradient_calls=steps, objective_backward_calls=steps,
        reference_backward_query_presentations=steps*size, reference_backward_support_presentations=steps*size,
        diagnostic_probability_calls=packets, diagnostic_query_presentations=packets*size,
        diagnostic_support_presentations=packets*size)
    require(all(type(stats[name]) is int and stats[name] >= 0 for name in COUNTERS), "invalid selective counter")
    require(all(stats[name] == count for name, count in expected.items()), "unmatched selective work budget")
    require(stats["projection_applied_steps"]+stats["shrink_applied_steps"] <= stats["rewrite_steps"]
            <= stats["projection_eligible_steps"] <= steps, "invalid projection/rewrite counts")
    require(stats["finite_step_increases"] <= packets and stats["zero_reference_steps"] <= steps
            and stats["zero_displacement_steps"] <= steps, "invalid finite-step or zero-vector counts")
    require((mode != "none" or stats["rewrite_steps"] == 0)
            and (mode == "project" or stats["projection_applied_steps"] == 0)
            and (mode == "shrink" or stats["shrink_applied_steps"] == 0), "inactive protection rule changed parameters")
    require(set(stats["summaries"]) == set(SUMMARY_NAMES), "bounded selective summary names changed")
    for name, item in stats["summaries"].items():
        expected_count = packets if name.startswith("finite_replay_") else steps
        require(set(item) == {"count", "sum", "min", "max"} and item["count"] == expected_count,
                "incomplete bounded selective summaries")
        require(item["min"] <= item["sum"]/item["count"]+1e-12 and item["sum"]/item["count"] <= item["max"]+1e-12,
                "selective summary mean is outside its extrema")


def audit_episode(record, config, identity):
    seed, model, stage, arm, branch = (record[name] for name in ("seed", "model", "stage", "arm", "branch"))
    require(record["identity"] == identity and stage == 1 and branch == "novel" and arm in ALL_ARMS,
            "unexpected/identity-mismatched episode")
    law = stage_law(seed, 1)
    require(digest(record["law"]) == digest(asdict(law)) and record["cue"] == order(seed)[0]
            and record["channel"] == "stage_1", "wrong law, cue or training channel")
    packets = config["episode_size"]//config["batch_size"]
    require(record["batches_done"] == packets and len(record["batch_sha256"]) == packets, "incomplete episode")
    valid_hashes(record["batch_sha256"])
    require([point["arrivals"] for point in record["curve"]] == list(range(0, config["episode_size"]+1, config["probe_every"])),
            "incomplete acquisition curve")
    for point in record["curve"]:
        check_probe_metrics(point["metrics"], marginal=point["arrivals"] > 0)
    before, after = record["start_diagnostics"], record["diagnostics"]
    expected = dict(arrivals=config["episode_size"], optimizer_steps=packets*config["updates_per_batch"],
        query_presentations=2*config["episode_size"]*config["updates_per_batch"],
        support_presentations=2*config["episode_size"]*config["updates_per_batch"])
    require(all(after["cost"][name]-before["cost"][name] == value for name, value in expected.items()),
            "unmatched arrival/update/presentation budget")
    require(sum(after["cost"][name]-before["cost"][name] for name in ("replay_presentations", "duplicate_presentations"))
            == config["episode_size"]*config["updates_per_batch"], "replay presentation budget mismatch")
    require(after["cost"]["training_seconds"] >= before["cost"]["training_seconds"], "negative episode training time")
    for field in ("parameters", "model_bytes", "initial_hash", "initial_encoder_hash"):
        require(before[field] == after[field], "episode changed architecture or initial-state identity")
    for diagnostic in (before, after):
        valid_hashes([diagnostic[name] for name in ("initial_hash", "final_hash", "initial_encoder_hash", "encoder_hash")])
    if arm == "fresh":
        require(before["final_hash"] == before["initial_hash"] and before["optimizer_bytes"] == 0
                and before["memory_ids"] == [] and before["cost"]["arrivals"] == 0, "fresh reference is not fresh")
    else:
        require(before["optimizer_bytes"] > 0 and before["cost"]["arrivals"] == config["prefix_blocks"]*config["prefix_size"],
                "learned arm reset optimizer or lost prefix exposure")
    check_selective(before["selective_updates"], after["selective_updates"], arm, config)
    for label, diagnostic, marginal in (("before", before, False), ("after", after, True)):
        check_probe(record[label], law, diagnostic["final_hash"], config, marginal)
    for name in ("valid_before", "valid_after"):
        require(set(record[name]) == {"0", "1"}, "valid-old modes missing")
        for values in record[name].values():
            require(set(values) == {"valid_brier", "valid_survival"}
                    and all(type(value) in (int, float) and 0 <= value <= 1.000001 for value in values.values()),
                    "invalid per-mode valid-old probe")
    for name, diagnostic in (("start_memory", before), ("end_memory", after)):
        check_memory(record[name], config)
        require(diagnostic["memory_ids"] == record[name]["ids"] and diagnostic["packets_seen"] == record[name]["seen"],
                "memory/diagnostic mismatch")
    require(record["end_memory"]["seen"]-record["start_memory"]["seen"] == packets, "reservoir lifetime reset")
    return seed, model, stage, arm, branch


def audit_records(directory):
    """Portable JSON/source validation; full tensor checkpoint audit is separate."""
    directory = Path(directory)
    manifest = read(directory/"manifest.json")
    config, identity = manifest["config"], manifest["identity"]
    study = study_api()
    study.validate_config(config)
    for name, value in (("config_sha256", config), ("source_sha256", manifest["source_files"]),
                        ("runtime_sha256", manifest["runtime"])):
        require(digest(value) == identity[name], "manifest identity mismatch")
    require(study.source_manifest() == manifest["source_files"], "current scientific source differs from lock")
    checked_zip(directory/"training_source.zip", manifest["source_files"])
    lock = read(directory/"protocol_lock.json")
    require(all(lock[name] == value for name, value in identity.items())
            and sha(directory/"protocol_at_lock.md") == lock["protocol_sha256"], "protocol lock mismatch")
    verify_analysis_lock(directory, config, lock["protocol_sha256"])
    expected = {(seed, model, 1, arm, "novel") for seed in config["seeds"] for model in config["models"] for arm in ALL_ARMS}
    records, hashes = {}, []
    for path in sorted(directory.glob("*/*/result.json")):
        record = read(path)
        key = audit_episode(record, config, identity)
        require(key in expected and key not in records, "unexpected or duplicate episode")
        require(path == directory/f"{key[1]}_{key[0]}"/f"stage_1_{key[3]}"/"result.json", "record in wrong folder")
        records[key] = record
        hashes.append(dict(path=path.relative_to(directory).as_posix(), sha256=sha(path)))
    require(set(records) == expected, f"incomplete cohort: missing {len(expected-set(records))} episodes")
    completion = read(directory/"completion.json")
    require(completion["identity"] == identity and all(completion[name] == value for name, value in study.counts(config).items()),
            "completion count/identity mismatch")
    jobs = completion["jobs"]
    pairs = {(seed, model) for seed in config["seeds"] for model in config["models"]}
    require(len(jobs) == len(pairs) and {(job["seed"], job["model"]) for job in jobs} == pairs
            and all(job["affinity"] == manifest["runtime"]["affinity"] for job in jobs), "worker coverage or runtime mismatch")
    prefix_paths = {directory/f"{model}_{seed}"/"prefix.json" for seed, model in pairs}
    require(set(directory.glob("*/prefix.json")) == prefix_paths, "prefix coverage mismatch")
    for seed, model in sorted(pairs):
        path = directory/f"{model}_{seed}"/"prefix.json"
        prefix = read(path)
        require(prefix["identity"] == identity and (prefix["seed"], prefix["model"], prefix["policy"]) == (seed, model, "uniform")
                and len(prefix["blocks"]) == config["prefix_blocks"], "prefix identity/completion mismatch")
        for index, block in enumerate(prefix["blocks"]):
            require(digest(block["law"]) == digest(asdict(Law((seed+index) % 2)))
                    and len(block["batch_sha256"]) == config["prefix_size"]//config["batch_size"], "prefix law or data length mismatch")
            valid_hashes(block["batch_sha256"])
        check_memory(prefix["memory"], config)
        prefix_diagnostic = prefix["blocks"][-1]["diagnostics"]
        group = [records[seed, model, 1, arm, "novel"] for arm in ARMS]
        for record in group:
            before = {key: value for key, value in record["start_diagnostics"].items() if key != "selective_updates"}
            require(before == prefix_diagnostic, "learned arms did not share the unchanged prefix diagnostics")
            require(record["start_memory"] == prefix["memory"], "learned fork changed replay/RNG state")
        require(all(record["end_memory"] == group[0]["end_memory"] for record in group), "intervention changed replay membership or sampling")
        require(all(record["valid_before"] == group[0]["valid_before"] and record["before"] == group[0]["before"] for record in group),
                "learned arms do not share identical starting probes")
        fresh = records[seed, model, 1, "fresh", "novel"]
        require(fresh["start_diagnostics"]["initial_hash"] == prefix_diagnostic["initial_hash"], "fresh reference initialization differs")
        hashes.append(dict(path=path.relative_to(directory).as_posix(), sha256=sha(path)))
    for seed in config["seeds"]:
        group = [record for key, record in records.items() if key[0] == seed]
        require(all(record["batch_sha256"] == group[0]["batch_sha256"] for record in group), "arms/models received different observations")
        require(len({record["start_diagnostics"]["initial_encoder_hash"] for record in group}) == 1, "different initial visual encoders")
    return manifest, records, sorted(hashes, key=lambda row: row["path"])


def episode_metrics(record):
    result = metrics(record)
    for mode in ("0", "1"):
        before, after = (record[name][mode]["valid_brier"] for name in ("valid_before", "valid_after"))
        result[f"valid_before_mode_{mode}"] = before
        result[f"valid_after_mode_{mode}"] = after
        result[f"valid_damage_mode_{mode}"] = after-before
    result["encoder_displacement"] = record["diagnostics"]["encoder_displacement"]
    return result


def upper_gate(metric, maximum):
    return dict(passed=metric["mean"] <= maximum, value=metric["mean"], maximum=maximum,
                lower=metric["lower"], upper=metric["upper"], uncertainty_crosses_limit=metric["upper"] > maximum)


def primary_screen(comparison, qualified, eligible=True):
    gates = dict(complete_development_cohort=dict(passed=bool(eligible)), fresh_qualification=dict(passed=bool(qualified)))
    gates["acquisition_gain"] = upper_gate(comparison["brier_auc"], -.002)
    improved = sum(value < 0 for value in comparison["brier_auc"]["differences"])
    gates["acquisition_consistency"] = dict(passed=improved >= 5, improved_seeds=improved, minimum=5,
                                             n=comparison["brier_auc"]["n"])
    for mode in (0, 1):
        gates[f"valid_old_mode_{mode}"] = upper_gate(comparison[f"valid_after_mode_{mode}"], .005)
    survival = comparison["survival_auc"]
    gates["survival"] = dict(passed=survival["mean"] >= -.01, value=survival["mean"], minimum=-.01,
        lower=survival["lower"], upper=survival["upper"], uncertainty_crosses_limit=survival["lower"] < -.01)
    return dict(passed=all(gate["passed"] for gate in gates.values()), gates=gates,
        failed=[name for name, gate in gates.items() if not gate["passed"]],
        interpretation="Fixed development allocation criteria on means; not a superiority/noninferiority test or confirmation.")


def attribution_screen(comparisons, qualified, eligible=True):
    gates = dict(complete_development_cohort=dict(passed=bool(eligible)), fresh_qualification=dict(passed=bool(qualified)))
    for control in ("current", "shrink"):
        comparison = comparisons[f"combined_minus_{control}"]
        gates[f"retention_vs_{control}"] = upper_gate(comparison["valid_after"], -.002)
        gates[f"acquisition_cost_vs_{control}"] = upper_gate(comparison["brier_auc"], .001)
    return dict(passed=all(gate["passed"] for gate in gates.values()), gates=gates,
        failed=[name for name, gate in gates.items() if not gate["passed"]],
        interpretation="Separate practical attribution screen; passing does not prove a selective-protection explanation.")


def analyze_rows(rows, config):
    keys = [(row["seed"], row["model"], row["arm"]) for row in rows]
    expected = {(seed, model, arm) for seed in config["seeds"] for model in config["models"] for arm in ALL_ARMS}
    require(len(keys) == len(set(keys)) and set(keys) == expected, "metrics must contain every arm exactly once per seed/model")
    raw = {key: row for key, row in zip(keys, rows)}
    qualified = qualification(rows, config["models"])
    comparisons = {model: {name: {field: paired([
        sum(weight*raw[seed, model, arm][field] for arm, weight in coefficients.items())
        for seed in config["seeds"]]) for field in FIELDS} for name, coefficients in CONTRASTS.items()}
        for model in config["models"]}
    for model in comparisons:
        for contrast in comparisons[model].values():
            for field in contrast.values():
                field["seeds"] = list(config["seeds"])
    eligible = (config["seeds"] == list(range(13001, 13007))
                and set(config["models"]) == {"conditional", "recurrent"}
                and config["prefix_blocks"] == 4 and config["prefix_size"] == 1024 and config["episode_size"] == 8192)
    means = {model: {arm: {field: mean(raw[seed, model, arm][field] for seed in config["seeds"])
                           for field in (*FIELDS, "valid_before", "before", "training_seconds", "encoder_displacement")}
                     for arm in ALL_ARMS} for model in config["models"]}
    cues = {}
    for model in config["models"]:
        cues[model] = {}
        for cue in sorted({row["cue"] for row in rows if row["model"] == model}):
            seeds = [seed for seed in config["seeds"] if raw[seed, model, "fresh"]["cue"] == cue]
            cues[model][str(cue)] = dict(seeds=seeds, n=len(seeds), arms={arm: {
                field: mean(raw[seed, model, arm][field] for seed in seeds) for field in FIELDS} for arm in ALL_ARMS})
    return dict(qualification=qualified, contrasts=comparisons, arm_means=means, cue_means=cues,
        eligible_development_cohort=eligible,
        screen={model: primary_screen(comparisons[model]["combined_minus_reference"], qualified["by_model"][model], eligible)
                for model in config["models"]},
        protection_attribution={model: attribution_screen(comparisons[model], qualified["by_model"][model], eligible)
                                for model in config["models"]})


def resource_rows(records):
    rows = []
    for key, record in sorted(records.items()):
        before, after = record["start_diagnostics"], record["diagnostics"]
        selective = after["selective_updates"]
        stats = selective["stats"]
        rows.append(dict(seed=key[0], model=key[1], arm=key[3],
            parameters=after["parameters"], model_bytes=after["model_bytes"], optimizer_bytes=after["optimizer_bytes"],
            peak_replay_bytes=after["peak_replay_bytes"], history_bytes=after["history_bytes"],
            training_seconds=after["cost"]["training_seconds"]-before["cost"]["training_seconds"],
            current_weight=selective["current_weight"], mode=selective["mode"],
            projection_eligible_fraction=stats["projection_eligible_steps"]/stats["steps"],
            projection_applied_fraction=stats["projection_applied_steps"]/stats["steps"],
            shrink_applied_fraction=stats["shrink_applied_steps"]/stats["steps"],
            finite_step_increase_fraction=stats["finite_step_increases"]/stats["packets"],
            counters={name: stats[name] for name in COUNTERS},
            scalar_means={name: item["sum"]/item["count"] for name, item in stats["summaries"].items()},
            selective_diagnostics=selective))
        for field in ("temporary_parameter_gradient_buffer_bytes_estimate", "temporary_buffer_estimate_scope", "temporary_vector_dtype"):
            if field in selective:
                rows[-1][field] = selective[field]
    return rows


def markdown(result):
    lines = ["# Selective updates: first-introduction development", "",
        f"{result['counts']['episodes']} challenge episodes and {result['counts']['prefix_fits']} shared prefixes completed. "
        f"Paired seeds: {result['config']['seeds']}.", "",
        "Only combined is the primary candidate. These fixed development thresholds allocate further research; "
        "they are not superiority/noninferiority tests or independent confirmation. All intervals are descriptive "
        "95% seed bootstrap intervals with 20,000 resamples. No ablation is silently promoted.", "",
        "## Per-architecture decisions", "",
        "| Model | Fresh qualification | Combined outcome screen | Protection attribution |", "|---|---|---|---|"]
    for model in result["config"]["models"]:
        values = (result["qualification"]["by_model"][model], result["screen"][model]["passed"],
                  result["protection_attribution"][model]["passed"])
        lines.append(f"| {model} | {' | '.join('PASS' if value else 'FAIL' for value in values)} |")
    lines.append("")
    if not result["eligible_development_cohort"]:
        lines += ["This is an engineering/subset configuration and cannot pass the declared development allocation screen.", ""]
    for label in ("screen", "protection_attribution"):
        for model, screen in result[label].items():
            lines.append(f"- {model}, {label}: failed criteria = {', '.join(screen['failed']) or 'none'}.")
    lines += ["", "## Paired contrasts", "", "Brier values are multiplied by 100; negative is better. "
              "Survival differences use percentage points. Each independent seed contributes one episode per arm.", "",
              "| Model | Contrast | Acquisition AUC [95% interval] | Valid-old mean [95% interval] | Survival AUC [95% interval] |",
              "|---|---|---:|---:|---:|"]
    for model, comparisons in result["contrasts"].items():
        for name, fields in comparisons.items():
            values = [f"{100*fields[field]['mean']:+.3f} [{100*fields[field]['lower']:+.3f}, {100*fields[field]['upper']:+.3f}]"
                      for field in ("brier_auc", "valid_after", "survival_auc")]
            lines.append(f"| {model} | {name} | {' | '.join(values)} |")
    lines += ["", "## Retention guardrails by old mode", "", "Combined minus reference, Brier x100. "
              "The +.005 raw-Brier tolerance is checked in each mode, not only their average.", "",
              "| Model | Mode | Difference [95% interval] | Gate | Interval crosses limit |", "|---|---:|---:|---|---|"]
    for model in result["config"]["models"]:
        for mode in (0, 1):
            item = result["contrasts"][model]["combined_minus_reference"][f"valid_after_mode_{mode}"]
            gate = result["screen"][model]["gates"][f"valid_old_mode_{mode}"]
            lines.append(f"| {model} | {mode} | {100*item['mean']:+.3f} [{100*item['lower']:+.3f}, {100*item['upper']:+.3f}] "
                         f"| {gate['passed']} | {gate['uncertainty_crosses_limit']} |")
    lines += ["", "## Fresh-reference qualification", "",
              "| Group | N | Marginal gain x100 | Cue benefit x100 | Pass |", "|---|---:|---:|---:|---|"]
    for name, item in result["qualification"]["groups"].items():
        lines.append(f"| {name} | {item['n']} | {100*item['marginal_gain']:.3f} | {100*item['cue_effect']:.3f} "
                     f"| {item['acquisition_ok'] and item['cue_use_ok']} |")
    lines += ["", "## Every episode", "", "Brier and cue-benefit values x100; all raw metric values remain in summary.json.", "",
              "| Seed | Model | Cue | Arm | Acquisition AUC | Valid-old mode 0 | Valid-old mode 1 | Cue benefit |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        lines.append(f"| {row['seed']} | {row['model']} | {row['cue']} | {row['arm']} | {100*row['brier_auc']:.3f} "
                     f"| {100*row['valid_after_mode_0']:.3f} | {100*row['valid_after_mode_1']:.3f} | {100*row['cue_effect']:+.3f} |")
    lines += ["", "## Resource and interpretation limits", "",
        "All arms compute the replay reference gradient, hypothetical projection and final-update replay-loss diagnostic. "
        "These extra operations are recorded separately from the original presentation budget. "
        "The deployable ordinary reference does not require discarded diagnostic work. Full counters, scalar summaries, "
        "parameter/optimizer/replay bytes and episode wall time are preserved in resource_rows.", "",
        "The constraint is first-order reported BCE on one sampled historical packet; finite-step replay loss can still "
        "increase, and this is not protection of every old relationship or clean physical outcome. "
        "These known-boundary first-introduction forks do not demonstrate autonomous lifelong learning, recurrence, "
        "noise robustness, late acquisition or compounding. Fresh-model damage is not a forgetting control because its "
        "starting errors differ; interpret terminal levels separately.", "",
        "The portable record check validates source/configuration/protocol/analysis locks, complete paired inputs, "
        "recorded forks and resource counts. A separate full checkpoint audit verifies actual tensor states and probes. "
        "Neither check is independent scientific replication.", ""]
    return "\n".join(lines)


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    require(not output.resolve().is_relative_to(directory.resolve()), "summary output must be outside input runs")
    manifest, records, hashes = audit_records(directory)
    config = manifest["config"]
    rows = [dict(seed=key[0], model=key[1], stage=1, arm=key[3], branch="novel", cue=record["cue"],
                 **episode_metrics(record)) for key, record in sorted(records.items())]
    result = dict(identity=manifest["identity"], config=config, counts=study_api().counts(config),
        rows=rows, result_hashes=hashes, resource_rows=resource_rows(records), **analyze_rows(rows, config))
    result["record_integrity"] = dict(passed=True, identity=manifest["identity"],
        source_unchanged=True, configuration_identity=True, protocol_lock=True, analysis_lock=True,
        analysis_files=len(read(directory/"analysis_lock.json")["files"]), all_expected_episodes=len(records),
        identical_recorded_forks=True, matched_data_and_work=True,
        scope="Portable record/source/lock/budget validation; no tensor checkpoint re-evaluation in this summary.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"summary.json", result)
    write_json(output/"record_integrity.json", result["record_integrity"])
    write_json(output/"config.json", config)
    for name in ("manifest.json", "completion.json", "training_source.zip", "protocol_lock.json", "protocol_at_lock.md",
                 "analysis_lock.json", "analysis_at_lock.zip"):
        shutil.copyfile(directory/name, output/name)
    with gzip.GzipFile(filename=str(output/"raw_results.jsonl.gz"), mode="wb", mtime=0) as stream:
        for key in sorted(records):
            stream.write((json.dumps(records[key], allow_nan=False)+"\n").encode("utf-8"))
    prefixes = {path.parent.name: read(path) for path in sorted(directory.glob("*/prefix.json"))}
    write_json(output/"prefix_records.json", prefixes)
    with (output/"rows.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    (output/"summary.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    print(json.dumps(dict(episodes=len(records), qualification=result["qualification"],
        screen={model: dict(passed=screen["passed"], failed=screen["failed"]) for model, screen in result["screen"].items()},
        protection_attribution={model: dict(passed=screen["passed"], failed=screen["failed"])
                                for model, screen in result["protection_attribution"].items()}), indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
