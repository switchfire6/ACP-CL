"""Audit the fixed evidence-consolidation cohort and apply its prospective screen."""

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
from acp_cl.acquisition.world import Law
from acp_cl.evidence_consolidation.mechanism import DELAYED, POLICIES, accept
from acp_cl.evidence_consolidation.schedule import schedule, training_law
from acp_cl.evidence_consolidation.study import counts, validate_config, write_json
from acp_cl.persistence.study import digest


CONTRAST_FIELDS = (
    "overall_truth", "overall_reported", "truth_total", "early_truth", "late_truth",
    "intro_focus_auc", "return_truth", "noise_truth", "revision_truth", "recovery_truth",
    "valid_after", "valid_damage", "survival_auc", "knowledge_after", "knowledge_mean",
    "adoptions",
)
PROBE_FIELDS = ("brier", "focus_brier", "survival", "focus_survival", "valid_brier",
                "valid_survival", "no_transfer", "clairvoyant_upper")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite value in record")


def read_json(path):
    path = Path(path)
    require(path.is_file(), f"missing required file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    finite_tree(value)
    return value


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), f"wrong {label} keys")


def bounded(value, low=0., high=1.):
    require(type(value) in (float, int) and math.isfinite(value) and low <= value <= high,
            "invalid bounded score")


def hashes(values, number):
    require(len(values) == number, "incomplete packet hashes")
    for value in values:
        require(isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value), "invalid SHA256")


def cost(diag, batches, config):
    arrivals = batches*config["batch_size"]
    updates = config["updates_per_batch"]
    expected = dict(arrivals=arrivals, optimizer_steps=batches*updates,
                    query_presentations=2*arrivals*updates,
                    support_presentations=2*arrivals*updates,
                    replay_presentations=max(0, batches-1)*config["batch_size"]*updates,
                    duplicate_presentations=(config["batch_size"]*updates if batches else 0))
    require(all(type(diag["cost"][k]) is int and diag["cost"][k] == v
                for k, v in expected.items()), "unequal arrival/update/presentation budget")
    require(diag["cost"]["training_seconds"] >= 0, "negative training time")
    require(diag["packets_seen"] == batches, "draft packet counter mismatch")
    require((diag["optimizer_bytes"] > 0) == (batches > 0), "optimizer state mismatch")
    hashes([diag[k] for k in ("initial_hash", "final_hash", "initial_encoder_hash", "encoder_hash")], 4)


def memory(state, batches, config):
    ids = state["ids"]
    require(state["type"] == "ReservoirMemory" and state["capacity"] == config["memory_packets"],
            "memory policy/capacity mismatch")
    require(state["seen"] == batches and state["age"] == batches, "memory/global age mismatch")
    require(len(ids) == min(batches, config["memory_packets"]) and len(ids) == len(set(ids))
            and all(type(i) is int and 0 <= i < batches for i in ids), "invalid replay IDs")
    require(len(state["packets"]) == len(ids), "packet/ID mismatch")
    require(all(p["hidden"] is False for p in state["packets"]), "hidden labels in replay")
    for packet in state["packets"]:
        hashes([packet["query"]], 1)
        if packet["support"] is not None:
            hashes([packet["support"]], 1)


def bundle_state(state, saved_memory, batches, config):
    window = config["evidence_window"]
    offset = batches % window
    require(state["batches"] == batches, "global batch counter mismatch")
    require(state["proposal_count"] == (batches+window-1)//window, "proposal counter mismatch")
    keys(state["committed"], DELAYED, "committed policy")
    keys(state["gains"], DELAYED, "pending evidence")
    keys(state["adoptions"], DELAYED, "adoption counter")
    hashes(list(state["committed"].values()), len(DELAYED))
    for policy in DELAYED:
        n = state["adoptions"][policy]
        require(type(n) is int and 0 <= n <= batches//window, "invalid adoption count")
        require(len(state["gains"][policy]) == offset, "pending evidence/window mismatch")
        for gain in state["gains"][policy]:
            bounded(gain, -1., 1.)
    require(state["adoptions"]["periodic"] == batches//window, "periodic acceptance skipped")
    if offset:
        hashes([state["proposal"]], 1)
        require(state["proposal_batch"] == batches-offset, "pending proposal age mismatch")
    else:
        require(state["proposal"] is None and state["proposal_batch"] is None,
                "completed evidence window retains proposal")
    cost(state["draft"], batches, config)
    memory(saved_memory, batches, config)
    require(state["draft"]["memory_ids"] == saved_memory["ids"], "draft/replay IDs mismatch")


def decision_chain(decisions, initial, end, config):
    """Recompute acceptance from logged evidence, including a window across a block."""
    first = initial["batches"] if initial is not None else 0
    window = config["evidence_window"]
    endpoints = list(range((first//window+1)*window, end["batches"]+1, window))
    require([d["end_batch"] for d in decisions] == endpoints, "decision schedule mismatch")
    committed = (dict(initial["committed"]) if initial is not None else
                 {p: end["draft"]["initial_hash"] for p in DELAYED})
    adopted = dict(initial["adoptions"]) if initial is not None else {p: 0 for p in DELAYED}
    for index, decision in enumerate(decisions):
        require(decision["start_batch"] == decision["end_batch"]-window,
                "proposal validation length mismatch")
        keys(decision["gains"], DELAYED, "decision evidence")
        keys(decision["accepted"], DELAYED, "acceptance")
        require(decision["committed_before"] == committed, "committed snapshot chain broken")
        hashes([decision["proposal_hash"]], 1)
        if decision["start_batch"] == 0:
            require(decision["proposal_hash"] == end["draft"]["initial_hash"],
                    "first proposal is not original initialization")
        for policy in DELAYED:
            gains = decision["gains"][policy]
            require(len(gains) == window, "incomplete decision evidence")
            for value in gains:
                bounded(value, -1., 1.)
            if index == 0 and initial is not None and first % window:
                require(gains[:first % window] == initial["gains"][policy],
                        "evidence discarded at block boundary")
                require(decision["proposal_hash"] == initial["proposal"],
                        "proposal replaced at block boundary")
            expected = accept(policy, gains, config["evidence_margin"], config["evidence_positive"])
            require(type(decision["accepted"][policy]) is bool
                    and decision["accepted"][policy] == expected, "acceptance rule mismatch")
            if expected:
                committed[policy] = decision["proposal_hash"]
                adopted[policy] += 1
    require(end["committed"] == committed and end["adoptions"] == adopted,
            "acceptance/snapshot endpoint mismatch")
    if not decisions and initial is not None and first % window:
        require(end["proposal"] == initial["proposal"], "unfinished proposal replaced")
        require(all(end["gains"][p][:first % window] == initial["gains"][p] for p in DELAYED),
                "unfinished evidence replaced")


def probes(value):
    for field in PROBE_FIELDS:
        bounded(value[field])
    if "marginal_brier" in value:
        bounded(value["marginal_brier"])


def valid_panel(value):
    keys(value, ("0", "1"), "valid-old mode")
    for row in value.values():
        keys(row, ("valid_brier", "valid_survival"), "valid-old metric")
        for score in row.values():
            bounded(score)


def expected_laws(seed, index):
    laws = [Law(0), Law(1)]
    for block in schedule(seed)[:index+1]:
        if block.law not in laws:
            laws.append(block.law)
    return json.loads(json.dumps([asdict(law) for law in laws]))


def audit_fresh(record, config, identity, seed, model, stage):
    block = schedule(seed)[stage-1]
    require(record["identity"] == identity and (record["seed"], record["model"], record["stage"])
            == (seed, model, stage), "fresh identity mismatch")
    require(record["arm"] == "fresh" and record["branch"] == "novel"
            and digest(record["law"]) == digest(asdict(block.law)) and record["cue"] == block.cue
            and record["channel"] == f"stage_{stage}", "wrong fresh law/cue/channel")
    batches = config["episode_size"]//config["batch_size"]
    require(record["batches_done"] == batches, "incomplete fresh reference")
    hashes(record["batch_sha256"], batches)
    require([p["arrivals"] for p in record["curve"]]
            == list(range(0, config["episode_size"]+1, config["probe_every"])), "incomplete fresh curve")
    for point in record["curve"]:
        probes(point["metrics"])
    for name, number in (("start_diagnostics", 0), ("diagnostics", batches)):
        cost(record[name], number, config)
    for name, number in (("start_memory", 0), ("end_memory", batches)):
        memory(record[name], number, config)
    before, after = record["start_diagnostics"], record["diagnostics"]
    require(before["final_hash"] == before["initial_hash"] and not before["memory_ids"],
            "fresh reference starts with learned state")
    require(after["initial_hash"] == before["initial_hash"], "fresh initialization identity changed")
    for name, diag in (("before", before), ("after", after)):
        panel = record[name]
        require(panel["model_sha256"] == diag["final_hash"] and panel["law"] == record["law"],
                "fresh probe/model mismatch")
        require(len(panel["replicates"]) == config["support_replicates"], "fresh probe replicate mismatch")
        expected = sorted(set((0, config["batch_size"]//4, config["batch_size"]//2, config["batch_size"])))
        for replicate in panel["replicates"]:
            require([p["feedback"] for p in replicate["curve"]] == expected, "fresh support curve mismatch")
            for point in replicate["curve"]:
                probes(point["metrics"])
            probes(replicate["flipped"])
        valid_panel(record[f"valid_{name}"])
    require(after["memory_ids"] == record["end_memory"]["ids"], "fresh final memory mismatch")


def audit_records(directory):
    """Validate portable records; checkpoint/prediction regeneration is a separate audit."""
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json")
    config, identity = manifest["config"], manifest["identity"]
    validate_config(config)
    keys(identity, ("config_sha256", "source_sha256", "runtime_sha256"), "identity")
    for name, value in (("config_sha256", config), ("source_sha256", manifest["source_files"]),
                        ("runtime_sha256", manifest["runtime"])):
        require(digest(value) == identity[name], "manifest identity mismatch")
    runtime = manifest["runtime"]
    require(runtime["threads"] == config["threads"] and runtime["workers"] == config["workers"]
            and runtime["deterministic"] is True, "runtime recipe mismatch")
    if config["cpu_affinity"]:
        require(runtime["affinity"] == config["cpu_affinity"], "runtime affinity mismatch")
    protocol_files = [directory/n for n in ("protocol_lock.json", "protocol_at_lock.md")]
    require(all(p.is_file() for p in protocol_files) or not any(p.exists() for p in protocol_files),
            "incomplete protocol lock")
    if all(p.is_file() for p in protocol_files):
        lock = read_json(protocol_files[0])
        require(all(lock[k] == value for k, value in identity.items())
                and hashlib.sha256(protocol_files[1].read_bytes()).hexdigest() == lock["protocol_sha256"],
                "protocol lock mismatch")
    else:
        require(config["blocks"] < 18, "complete scientific cohort requires protocol lock")
    with zipfile.ZipFile(directory/"training_source.zip") as archive:
        require(len(archive.namelist()) == len(set(archive.namelist()))
                and set(archive.namelist()) == set(manifest["source_files"]), "source archive file set mismatch")
        for name, expected in manifest["source_files"].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, "source archive mismatch")
    pairs = [(s, m) for s in config["seeds"] for m in config["models"]]
    expected = {directory/f"{m}_{s}"/f"block_{i:02}"/"result.json": (s, m, i)
                for s, m in pairs for i in range(config["blocks"])}
    expected_fresh = {directory/f"{m}_{s}"/f"fresh_{stage}"/"result.json": (s, m, stage)
                      for s, m in pairs for stage in (1, 2, 3)}
    expected_prefix = {directory/f"{m}_{s}"/"prefix.json": (s, m) for s, m in pairs}
    require(set(directory.rglob("result.json")) == set(expected) | set(expected_fresh),
            "missing or extra block/fresh result files")
    require(set(directory.rglob("prefix.json")) == set(expected_prefix), "missing or extra prefix records")
    blocks, fresh, prefixes, result_hashes = {}, {}, {}, []
    for paths, target in ((expected_prefix, prefixes), (expected, blocks), (expected_fresh, fresh)):
        for path, key in sorted(paths.items()):
            target[key] = read_json(path)
            result_hashes.append(dict(path=path.relative_to(directory).as_posix(),
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    prefix_batches = config["prefix_blocks"]*config["prefix_size"]//config["batch_size"]
    episode_batches = config["episode_size"]//config["batch_size"]
    for seed, model in pairs:
        prefix = prefixes[seed, model]
        require(prefix["identity"] == identity and (prefix["seed"], prefix["model"]) == (seed, model),
                "prefix identity mismatch")
        require(len(prefix["blocks"]) == config["prefix_blocks"], "incomplete prefix")
        for i, row in enumerate(prefix["blocks"]):
            require(digest(row["law"]) == digest(asdict(Law((seed+i) % 2))), "wrong prefix law")
            hashes(row["batch_sha256"], config["prefix_size"]//config["batch_size"])
        bundle_state(prefix["end_state"], prefix["end_memory"], prefix_batches, config)
        decision_chain(prefix["decisions"], None, prefix["end_state"], config)
        previous, previous_memory = prefix["end_state"], prefix["end_memory"]
        for index, block in enumerate(schedule(seed)[:config["blocks"]]):
            record = blocks[seed, model, index]
            require(record["identity"] == identity and (record["seed"], record["model"]) == (seed, model)
                    and digest(record["block"]) == digest(asdict(block)), "block identity/law/schedule mismatch")
            require(record["batches_done"] == episode_batches, "incomplete block")
            hashes(record["batch_sha256"], episode_batches)
            hashes(record["clean_sha256"], episode_batches)
            require(all(record["batch_sha256"][i] == record["clean_sha256"][i]
                        for i in range(episode_batches) if training_law(block, i, config).noise == 0),
                    "clean packets differ from reported packets without corruption")
            start = prefix_batches+index*episode_batches
            for name, number in (("start", start), ("end", start+episode_batches)):
                bundle_state(record[f"{name}_state"], record[f"{name}_memory"], number, config)
            require(record["start_state"] == previous and record["start_memory"] == previous_memory,
                    "continuous state/memory chain broken")
            require(record["end_state"]["draft"]["initial_hash"] == previous["draft"]["initial_hash"],
                    "draft initialization identity changed")
            decision_chain(record["decisions"], previous, record["end_state"], config)
            require([p["arrivals"] for p in record["prequential"]]
                    == list(range(config["batch_size"], config["episode_size"]+1, config["batch_size"])),
                    "incomplete every-packet prequential scores")
            for point in record["prequential"]:
                for field in ("truth_brier", "reported_brier"):
                    keys(point[field], POLICIES, "prequential policy")
                    for score in point[field].values():
                        bounded(score)
            require([p["arrivals"] for p in record["curve"]]
                    == list(range(0, config["episode_size"]+1, config["probe_every"])), "incomplete block curve")
            for point in record["curve"]:
                keys(point["policies"], POLICIES, "probe policy")
                for value in point["policies"].values():
                    probes(value)
            for name in ("valid_before", "valid_after"):
                keys(record[name], POLICIES, "retention policy")
                for value in record[name].values():
                    valid_panel(value)
            require([v["law"] for v in record["knowledge_after"]] == expected_laws(seed, index),
                    "knowledge panel does not contain exactly the encountered laws")
            for item in record["knowledge_after"]:
                keys(item["policies"], POLICIES, "knowledge policy")
                for score in item["policies"].values():
                    bounded(score)
            if index < 3:
                reference = fresh[seed, model, index+1]
                audit_fresh(reference, config, identity, seed, model, index+1)
                require(reference["batch_sha256"] == record["batch_sha256"], "fresh/draft data mismatch")
                require(reference["start_diagnostics"]["initial_hash"] == previous["draft"]["initial_hash"]
                        and reference["start_diagnostics"]["initial_encoder_hash"]
                        == previous["draft"]["initial_encoder_hash"], "fresh/draft initialization mismatch")
            previous, previous_memory = record["end_state"], record["end_memory"]
    for seed in config["seeds"]:
        require(len({digest(prefixes[seed, m]["blocks"]) for m in config["models"]}) == 1,
                "different prefix data across architectures")
        require(len({prefixes[seed, m]["end_state"]["draft"]["initial_encoder_hash"]
                     for m in config["models"]}) == 1, "different initial encoders across architectures")
        for index in range(config["blocks"]):
            for field in ("batch_sha256", "clean_sha256"):
                require(len({tuple(blocks[seed, m, index][field]) for m in config["models"]}) == 1,
                        "different raw data across architectures")
    completion = read_json(directory/"completion.json")
    require(completion["identity"] == identity and all(completion[k] == v for k, v in counts(config).items()),
            "completion counts/identity mismatch")
    jobs = completion["jobs"]
    require(len(jobs) == len(pairs) and {(j["seed"], j["model"]) for j in jobs} == set(pairs),
            "missing/extra/duplicate completed job")
    for job in jobs:
        require(job["affinity"] == runtime["affinity"], "worker affinity mismatch")
        require(job["final_state"] == blocks[job["seed"], job["model"], config["blocks"]-1]["end_state"],
                "completion final state mismatch")
    return manifest, blocks, fresh, prefixes, result_hashes


def policy_area(curve, policy, field):
    return float(sum((b["arrivals"]-a["arrivals"])*
                     (a["policies"][policy][field]+b["policies"][policy][field])/2
                     for a, b in zip(curve[:-1], curve[1:]))/curve[-1]["arrivals"])


def block_metrics(record, policy):
    before = mean(v["valid_brier"] for v in record["valid_before"][policy].values())
    after = mean(v["valid_brier"] for v in record["valid_after"][policy].values())
    return dict(truth_brier=mean(p["truth_brier"][policy] for p in record["prequential"]),
                reported_brier=mean(p["reported_brier"][policy] for p in record["prequential"]),
                focus_auc=policy_area(record["curve"], policy, "focus_brier"),
                brier_auc=policy_area(record["curve"], policy, "brier"),
                survival_auc=policy_area(record["curve"], policy, "survival"),
                valid_before=before, valid_after=after, valid_damage=after-before,
                valid_before_modes=record["valid_before"][policy],
                valid_after_modes=record["valid_after"][policy],
                knowledge_after=mean(v["policies"][policy] for v in record["knowledge_after"]),
                adoptions=(record["end_state"]["adoptions"][policy]
                           -record["start_state"]["adoptions"][policy] if policy in DELAYED else 0),
                decision_windows=len(record["decisions"]))


def selected_mean(rows, field, select=lambda row: True):
    values = [r[field] for r in rows if select(r)]
    return mean(values) if values else None


def seed_metrics(rows, config):
    result = dict(overall_truth=selected_mean(rows, "truth_brier"),
                  overall_reported=selected_mean(rows, "reported_brier"),
                  early_truth=selected_mean(rows, "truth_brier", lambda r: 3 <= r["block"] <= 7),
                  late_truth=selected_mean(rows, "truth_brier", lambda r: 13 <= r["block"] <= 17),
                  intro_focus_auc=selected_mean(rows, "focus_auc", lambda r: r["block"] < 3),
                  valid_after=selected_mean(rows, "valid_after"),
                  valid_damage=selected_mean(rows, "valid_damage"),
                  survival_auc=selected_mean(rows, "survival_auc"),
                  knowledge_after=rows[-1]["knowledge_after"],
                  knowledge_mean=selected_mean(rows, "knowledge_after"),
                  adoptions=sum(r["adoptions"] for r in rows),
                  truth_total=sum(r["truth_brier"]*3*config["episode_size"] for r in rows))
    for label in ("return", "noise", "revision", "recovery"):
        result[f"{label}_truth"] = selected_mean(rows, "truth_brier", lambda r: r["label"] == label)
    return result


def comparisons(seed_rows, config):
    raw = {(r["seed"], r["model"], r["policy"]): r for r in seed_rows}
    result = {}
    for model in config["models"]:
        result[model] = {}
        for control in ("immediate", "periodic", "single"):
            values = {}
            for field in CONTRAST_FIELDS:
                pairs = [(raw[s, model, "sustained"][field], raw[s, model, control][field])
                         for s in config["seeds"]]
                values[field] = (paired([a-b for a, b in pairs])
                                 if all(a is not None and b is not None for a, b in pairs) else None)
            if values["early_truth"] is not None and values["late_truth"] is not None:
                values["late_minus_early_advantage"] = paired([
                    (raw[s, model, control]["late_truth"]-raw[s, model, "sustained"]["late_truth"])
                    -(raw[s, model, control]["early_truth"]-raw[s, model, "sustained"]["early_truth"])
                    for s in config["seeds"]])
            else:
                values["late_minus_early_advantage"] = None
            result[model][f"sustained_minus_{control}"] = values
    return result


def pilot_screen(contrasts, qualified, config):
    reasons = []
    if len(config["seeds"]) != 6:
        reasons.append("requires six independent seeds")
    if config["blocks"] != 18:
        reasons.append("requires all eighteen continuous blocks")
    if config["kind"] != "diagnostic":
        reasons.append("development cohort is descriptive")
    gates = dict(fresh_qualification=dict(passed=bool(qualified)))

    def upper(name, metric, limit):
        if metric is None:
            gates[name] = dict(passed=None, reason="endpoint unavailable", maximum=limit)
        else:
            gates[name] = dict(passed=metric["mean"] <= limit, value=metric["mean"], maximum=limit,
                               lower=metric["lower"], upper=metric["upper"],
                               uncertainty_crosses_limit=metric["upper"] > limit)

    for control in ("immediate", "periodic"):
        comparison = contrasts[f"sustained_minus_{control}"]
        for field in ("overall_truth", "late_truth"):
            upper(f"{control}_{field}_gain", comparison[field], -.002)
        metric = comparison["overall_truth"]
        improved = sum(v < 0 for v in metric["differences"])
        gates[f"{control}_overall_consistency"] = dict(
            passed=improved >= 5 if metric["n"] == 6 else None,
            improved_seeds=improved, minimum=5, required_n=6, n=metric["n"])
    immediate = contrasts["sustained_minus_immediate"]
    for field in ("intro_focus_auc", "return_truth", "noise_truth", "valid_after"):
        upper(f"immediate_{field}_guardrail", immediate[field], .005)
    metric = immediate["survival_auc"]
    gates["immediate_survival_guardrail"] = dict(
        passed=metric["mean"] >= -.01, value=metric["mean"], minimum=-.01,
        lower=metric["lower"], upper=metric["upper"], uncertainty_crosses_limit=metric["lower"] < -.01)
    failed = [name for name, gate in gates.items() if gate["passed"] is False]
    eligible = not reasons
    passed = eligible and all(g["passed"] is True for g in gates.values())
    return dict(eligible=eligible, eligibility_reasons=reasons, passed=passed,
                status="PASS" if passed else "FAIL" if eligible else "INCONCLUSIVE",
                failed=failed if eligible else [], descriptive_failed_gates=failed, gates=gates,
                interpretation="Fixed mean-based pilot screen; descriptive paired-seed intervals, not a noninferiority test.")


def window_rows(blocks, prefixes, config):
    rows = []
    for seed in config["seeds"]:
        for model in config["models"]:
            births = {p: 0 for p in DELAYED}
            prefix = prefixes[seed, model]
            records = [(None, prefix)] + [(i, blocks[seed, model, i]) for i in range(config["blocks"])]
            for index, record in records:
                for decision in record["decisions"]:
                    for policy in DELAYED:
                        old_birth = births[policy]
                        if decision["accepted"][policy]:
                            births[policy] = decision["start_batch"]
                        if index is not None:
                            gains = decision["gains"][policy]
                            rows.append(dict(seed=seed, model=model, block=index,
                                label=record["block"]["label"], policy=policy,
                                cycle=record["block"]["cycle"], noise_kind=record["block"]["noise_kind"],
                                start_batch=decision["start_batch"], end_batch=decision["end_batch"],
                                accepted=decision["accepted"][policy], mean_gain=mean(gains),
                                first_gain=gains[0], positive_packets=sum(g > 0 for g in gains),
                                minimum_gain=min(gains), maximum_gain=max(gains),
                                proposal_age_packets=decision["end_batch"]-decision["start_batch"],
                                committed_age_before=decision["end_batch"]-old_birth,
                                committed_age_after=decision["end_batch"]-births[policy]))
    return rows


def resource_rows(blocks, config):
    """Count implemented operations and logical standalone storage, not benchmark speed."""
    rows = []
    for seed in config["seeds"]:
        for model in config["models"]:
            final = blocks[seed, model, config["blocks"]-1]["end_state"]
            diag, packets = final["draft"], final["batches"]
            rows.append(dict(seed=seed, model=model, draft_packets=packets,
                parameters=diag["parameters"], model_bytes=diag["model_bytes"],
                optimizer_bytes=diag["optimizer_bytes"], peak_replay_bytes=diag["peak_replay_bytes"],
                history_bytes=diag["history_bytes"], memory_capacity_packets=config["memory_packets"],
                training_cost=diag["cost"], training_probability_passes=2*diag["cost"]["optimizer_steps"],
                shared_runner_scoring_prediction_passes=5*packets,
                shared_runner_maximum_model_copies=5,
                shared_runner_maximum_model_bytes=5*diag["model_bytes"],
                shared_runner_maximum_evidence_gain_values=len(DELAYED)*config["evidence_window"],
                standalone={policy: dict(maximum_model_copies=1 if policy == "immediate" else 3,
                    maximum_copied_parameters=diag["parameters"]*(1 if policy == "immediate" else 3),
                    maximum_model_bytes=diag["model_bytes"]*(1 if policy == "immediate" else 3),
                    policy_and_evidence_prediction_passes=packets*(1 if policy == "immediate" else 2),
                    maximum_evidence_gain_values=0 if policy == "immediate" else config["evidence_window"])
                    for policy in POLICIES}))
    return rows


def markdown(result):
    config = result["config"]
    lines = ["# Adam: evidence-consolidation diagnostic", "",
        f"{result['counts']['trajectories']} shared draft trajectories, "
        f"{result['counts']['blocks']} blocks, {result['counts']['fresh_episodes']} fresh references; "
        f"{len(config['seeds'])} independent seeds per architecture.", "",
        "All four policies deploy predictors from the same learned draft trajectory. "
        "This tests acceptance of updates, not different representation-learning trajectories.", "",
        "Primary truth error scores the performed action before training on every incoming packet, "
        "averaging all three outcome horizons with the actual preceding reported history. "
        "Focus and survival AUC instead integrate periodic independent all-action query probes; "
        "clean-support valid-old and knowledge panels are separate retention measurements.", "",
        "Brier numbers below are multiplied by 100. Error contrasts are sustained minus control; "
        "negative is better. Intervals use 20,000 paired-seed bootstrap resamples, seed 27192026. "
        "Packets and blocks are not independent seed replicates.", "",
        "| Model | Control | Overall truth difference [95% interval] | Late difference | Growing advantage, late minus early |",
        "|---|---|---:|---:|---:|"]

    def fmt(metric):
        return ("unavailable" if metric is None else
                f"{100*metric['mean']:+.3f} [{100*metric['lower']:+.3f}, {100*metric['upper']:+.3f}]")

    for model, comparisons_by_model in result["contrasts"].items():
        for control, values in comparisons_by_model.items():
            lines.append(f"| {model} | {control.removeprefix('sustained_minus_')} | "
                f"{fmt(values['overall_truth'])} | {fmt(values['late_truth'])} | "
                f"{fmt(values['late_minus_early_advantage'])} |")
    lines += ["", "Positive growing advantage means the candidate's relative error improved "
              "more in cycle 3 than cycle 1; it does not establish compounding representation gains.", ""]
    for model, screen in result["screen"].items():
        details = ("; ".join(screen["eligibility_reasons"]) if not screen["eligible"] else
                   "failed gates: "+(", ".join(screen["failed"]) or "none"))
        lines.append(f"* {model}: **{screen['status']}**; {details}.")
    lines += ["", "| Model | Gate | Pass | Value | Tolerance | Interval permits violation |",
              "|---|---|---|---:|---:|---|"]
    for model, screen in result["screen"].items():
        for name, gate in screen["gates"].items():
            value = (f"{100*gate['value']:+.3f}" if "value" in gate else
                     str(gate.get("improved_seeds", "-")))
            bound = (f"<= {100*gate['maximum']:+.3f}" if "maximum" in gate else
                     f">= {100*gate['minimum']:+.3f}" if "value" in gate else
                     "5 of 6" if "improved_seeds" in gate else "all fresh groups")
            lines.append(f"| {model} | {name} | {gate['passed']} | {value} | {bound} | "
                         f"{gate.get('uncertainty_crosses_limit', '-')} |")
    lines += ["", "| Fresh qualification group | Episodes | Marginal gain x100 | Cue benefit x100 | Pass |",
              "|---|---:|---:|---:|---|"]
    for name, group in result["qualification"]["groups"].items():
        lines.append(f"| {name} | {group['n']} | {100*group['marginal_gain']:.3f} | "
                     f"{100*group['cue_effect']:.3f} | {group['acquisition_ok'] and group['cue_use_ok']} |")
    lines += ["", "| Seed | Model | Policy | Overall truth | Early truth | Late truth | Valid-old level | Survival % | Adoptions |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    def scalar(value):
        return "unavailable" if value is None else f"{100*value:.3f}"
    for row in result["seed_rows"]:
        values = " | ".join(scalar(row[f]) for f in
                            ("overall_truth", "early_truth", "late_truth", "valid_after", "survival_auc"))
        lines.append(f"| {row['seed']} | {row['model']} | {row['policy']} | {values} | {row['adoptions']} |")
    lines += ["", "| Seed | Model | Block (1-based) | Condition | Policy | Truth | Reported | Focus AUC | Valid-old level | Survival % |",
              "|---:|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in result["block_rows"]:
        values = " | ".join(scalar(row[f]) for f in
                            ("truth_brier", "reported_brier", "focus_auc", "valid_after", "survival_auc"))
        lines.append(f"| {row['seed']} | {row['model']} | {row['block']+1} | {row['label']} | {row['policy']} | {values} |")
    lines += ["", "summary.json preserves both valid-old modes, every paired seed difference, "
              "per-block cumulative loss, knowledge endpoints, and window diagnostics. "
              "raw_results.jsonl.gz contains every complete block and fresh-reference record. "
              "Immediate has no adoption gate; its reported zero adoptions is not zero learning. "
              "Noise-block policy differences are not causal noise penalties without matched clean twin trajectories.", ""]
    lines += ["The immediate standalone learner needs one model. The delayed scaffold can retain "
              "three models (draft, proposal, committed) and uses two policy/evidence prediction "
              "passes per packet versus one for immediate, in addition to the shared training budget. "
              "resource_rows records parameter bytes, replay/optimizer sizes, and operation counts. "
              "These are logical standalone counts for the implemented scaffold, not measured runtime "
              "or total process RAM; evaluator probes are excluded from those online inference counts.", ""]
    return "\n".join(lines)


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    require(directory.resolve() != output.resolve(), "summary output must differ from run directory")
    manifest, blocks, fresh, prefixes, result_hashes = audit_records(directory)
    config = manifest["config"]
    fresh_rows = [dict(seed=s, model=m, stage=k, arm="fresh", branch="novel", cue=r["cue"], **metrics(r))
                  for (s, m, k), r in sorted(fresh.items())]
    qualified = qualification(fresh_rows, config["models"])
    block_rows, seed_rows = [], []
    for seed in config["seeds"]:
        for model in config["models"]:
            for policy in POLICIES:
                selected, cumulative = [], 0.
                for index in range(config["blocks"]):
                    record = blocks[seed, model, index]
                    row = dict(seed=seed, model=model, block=index, label=record["block"]["label"],
                               cycle=record["block"]["cycle"], cue=record["block"]["cue"],
                               noise_kind=record["block"]["noise_kind"], policy=policy,
                               **block_metrics(record, policy))
                    cumulative += row["truth_brier"]*config["episode_size"]*3
                    row["cumulative_truth_total"] = cumulative
                    row["cumulative_arrivals"] = (index+1)*config["episode_size"]
                    row["cumulative_truth_mean"] = cumulative/(3*row["cumulative_arrivals"])
                    selected.append(row)
                block_rows.extend(selected)
                seed_rows.append(dict(seed=seed, model=model, policy=policy, **seed_metrics(selected, config)))
    contrasts = comparisons(seed_rows, config)
    result = dict(experiment="evidence_consolidation", identity=manifest["identity"], config=config,
                  seeds=config["seeds"], counts=counts(config), qualification=qualified,
                  block_rows=block_rows, seed_rows=seed_rows, fresh_rows=fresh_rows,
                  window_rows=window_rows(blocks, prefixes, config), contrasts=contrasts,
                  resource_rows=resource_rows(blocks, config),
                  result_hashes=result_hashes, bootstrap=dict(resamples=20000, seed=27192026),
                  screen={m: pilot_screen(contrasts[m], qualified["by_model"][m], config)
                          for m in config["models"]},
                  definitions=dict(truth_total="Sum of squared error over arrivals and three horizons, excluding prefix.",
                      truth_brier="Every-packet performed-action pre-update error over three horizons with actual preceding reported history; clean outcomes are evaluator-only.",
                      query_auc="Periodic independent all-action query panels, integrated by trapezoids over arrivals; survival uses the predicted best action at terminal horizon.",
                      block_index="Zero-based; cycles 1 and 3 use blocks 3..7 and 13..17.",
                      knowledge_after="Final endpoint: equal mean over distinct encountered laws with clean support.",
                      valid_after="Equal mean of both valid modes, then all post-prefix block endpoints.",
                      late_minus_early_advantage="(control - sustained) late minus (control - sustained) early; positive favors growing advantage.",
                      immediate_adoptions="Zero denotes absence of an adoption gate, not absence of training updates.",
                      resources="Standalone counts describe the implemented scaffold; delayed maximum storage is draft+proposal+committed. Online prediction counts exclude training forwards and evaluator probes. Shared runner uses at most five model copies, five scoring passes per packet and 2*updates probability passes for training. Evidence scalar counts exclude counters, hashes and Python object overhead; byte counts are not total process RAM. Fresh-reference training is separate. No efficiency superiority is inferred.",
                      limitation="One shared draft per seed/model; acceptance policy effects do not establish representation-learning gains."))
    result["policy_means"] = {m: {p: {f: selected_mean([r for r in seed_rows
        if r["model"] == m and r["policy"] == p], f) if all(r[f] is not None for r in seed_rows
        if r["model"] == m and r["policy"] == p) else None for f in CONTRAST_FIELDS}
        for p in POLICIES} for m in config["models"]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"summary.json", result)
    (output/"summary.md").write_text(markdown(result), encoding="utf-8")
    for name in ("manifest.json", "completion.json", "training_source.zip", "protocol_lock.json", "protocol_at_lock.md"):
        if (directory/name).is_file():
            shutil.copyfile(directory/name, output/name)
    write_json(output/"config.json", config)
    write_json(output/"prefix_records.json", {f"{m}_{s}": r for (s, m), r in sorted(prefixes.items())})
    with gzip.GzipFile(filename=str(output/"raw_results.jsonl.gz"), mode="wb", mtime=0) as stream:
        for records in (blocks, fresh):
            for key in sorted(records):
                stream.write((json.dumps(records[key], allow_nan=False)+"\n").encode("utf-8"))
    print(json.dumps(dict(counts=result["counts"], qualification=qualified,
                         screen={m: {k: s[k] for k in ("status", "eligible", "failed")}
                                 for m, s in result["screen"].items()}), indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
