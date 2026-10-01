"""Audit complete records and apply the prospective replay-renewal comparisons."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from summarize_training_state import mean, metrics, paired
from acp_cl.acquisition.world import order, stage_law
from acp_cl.persistence.study import digest
from acp_cl.replay_renewal.memory import POLICIES, RESET_ARMS
from acp_cl.replay_renewal.study import counts, validate_config, write_json
from acp_cl.training_state.study import BRANCHES, final_law


DECOMPOSITION = {
    "clear_minus_keep": {"clear": 1, "keep": -1},
    "clear_rebase_minus_rebase": {"clear_rebase": 1, "rebase": -1},
    "rebase_minus_keep": {"rebase": 1, "keep": -1},
    "clear_rebase_minus_clear": {"clear_rebase": 1, "clear": -1},
    "clear_main": {"clear": .5, "keep": -.5, "clear_rebase": .5, "rebase": -.5},
    "rebase_main": {"rebase": .5, "keep": -.5, "clear_rebase": .5, "clear": -.5},
    "interaction": {"clear_rebase": 1, "clear": -1, "rebase": -1, "keep": 1},
    "rng_reset_minus_keep": {"rng_reset": 1, "keep": -1},
    "full_reset_minus_clear_rebase": {"full_reset": 1, "clear_rebase": -1},
    "full_reset_minus_keep": {"full_reset": 1, "keep": -1},
    "keep_minus_fresh": {"keep": 1, "fresh": -1},
}
POLICY_CONTRASTS = {"split_minus_uniform": {"split": 1, "uniform": -1},
    "recent_minus_uniform": {"recent": 1, "uniform": -1},
    "split_minus_recent": {"split": 1, "recent": -1}}
FIELDS = ("brier_auc", "valid_after", "valid_damage", "after", "survival_auc")


def expected_keys(config):
    pairs = [(s, m) for s in config["seeds"] for m in config["models"]]
    if config["experiment"] == "decomposition":
        return {(s, m, 1, a, "novel") for s, m in pairs for a in (*RESET_ARMS, "fresh")}
    return ({(s, m, k, a, "novel") for s, m in pairs for k in (1, 2, 3) for a in (*POLICIES, "fresh")}
            | {(s, m, 4, a, b) for s, m in pairs for a in POLICIES for b in BRANCHES})


def record_path(directory, config, key):
    seed, model, stage, arm, branch = key
    policy = "base" if config["experiment"] == "decomposition" else arm
    tag = f"stage_{stage}_{arm}" if branch == "novel" else f"final_{branch}_{arm}"
    return Path(directory)/f"{model}_{seed}_{policy}"/tag


def audit_records(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    c, identity = manifest["config"], manifest["identity"]
    validate_config(c)
    for name, value in (("config_sha256", c), ("source_sha256", manifest["source_files"]),
                        ("runtime_sha256", manifest["runtime"])):
        if digest(value) != identity[name]:
            raise ValueError("manifest mismatch")
    if (directory/"protocol_lock.json").exists():
        lock = json.loads((directory/"protocol_lock.json").read_text(encoding="utf-8"))
        if any(lock[k] != v for k, v in identity.items()) or hashlib.sha256(
                (directory/"protocol_at_lock.md").read_bytes()).hexdigest() != lock["protocol_sha256"]:
            raise ValueError("protocol mismatch")
    with zipfile.ZipFile(directory/"training_source.zip") as z:
        if set(z.namelist()) != set(manifest["source_files"]):
            raise ValueError("source archive file set mismatch")
        for name, expected in manifest["source_files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("source archive mismatch")
    expected, records, hashes = expected_keys(c), {}, []
    for path in sorted(directory.glob("*/*/result.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        key = r["seed"], r["model"], r["stage"], r["arm"], r["branch"]
        if key in records or key not in expected or r["identity"] != identity:
            raise ValueError("unexpected/duplicate/mismatched record")
        if path.parent != record_path(directory, c, key):
            raise ValueError("record in wrong folder")
        law = stage_law(key[0], key[2]) if key[4] == "novel" else final_law(key[0], key[4], c["feedback_noise"])
        cue = order(key[0])[key[2]-1] if key[4] == "novel" else 0 if key[4] == "revision" else None
        channel = f"stage_{key[2]}" if key[4] == "novel" else "final"
        if digest(asdict(law)) != digest(r["law"]) or r["cue"] != cue or r["channel"] != channel:
            raise ValueError("wrong law, cue or data channel")
        if r["batches_done"] != c["episode_size"]//c["batch_size"] or len(r["batch_sha256"]) != r["batches_done"]:
            raise ValueError("incomplete episode")
        if [p["arrivals"] for p in r["curve"]] != list(range(0, c["episode_size"]+1, c["probe_every"])):
            raise ValueError("incomplete curve")
        before, after = r["start_diagnostics"], r["diagnostics"]
        budget = dict(arrivals=c["episode_size"], optimizer_steps=r["batches_done"]*c["updates_per_batch"],
            query_presentations=2*c["episode_size"]*c["updates_per_batch"],
            support_presentations=2*c["episode_size"]*c["updates_per_batch"])
        if any(after["cost"][k]-before["cost"][k] != v for k, v in budget.items()):
            raise ValueError("unequal arrival/update/presentation budget")
        if r["before"]["model_sha256"] != before["final_hash"] or r["after"]["model_sha256"] != after["final_hash"]:
            raise ValueError("probe/model hash mismatch")
        if key[3] == "fresh" and (before["optimizer_bytes"] or before["memory_ids"] or before["final_hash"] != before["initial_hash"]):
            raise ValueError("fresh state is not fresh")
        if key[3] != "fresh" and not before["optimizer_bytes"]:
            raise ValueError("carried optimizer was reset")
        for memory in (r["start_memory"], r["end_memory"]):
            ids = memory["ids"]
            if len(ids) > c["memory_packets"] or len(ids) != len(set(ids)) or any(i >= memory["seen"] or i < 0 for i in ids):
                raise ValueError("invalid memory capacity or IDs")
            if any(p["hidden"] for p in memory["packets"]):
                raise ValueError("hidden labels supplied to replay")
        if r["end_memory"]["seen"]-r["start_memory"]["seen"] != r["batches_done"]:
            raise ValueError("lifetime packet counter reset")
        records[key] = r
        hashes.append(dict(path=path.relative_to(directory).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if set(records) != expected:
        raise ValueError(f"missing {len(expected-set(records))} episodes")
    completion = json.loads((directory/"completion.json").read_text(encoding="utf-8"))
    if completion["identity"] != identity or any(completion[k] != v for k, v in counts(c).items()):
        raise ValueError("completion mismatch")
    if len(completion["jobs"]) != counts(c)["trajectories"] or any(j["affinity"] != manifest["runtime"]["affinity"] for j in completion["jobs"]):
        raise ValueError("runtime worker mismatch")
    for seed in c["seeds"]:
        for stage, branch in sorted({(k[2], k[4]) for k in records}):
            group = [r for k, r in records.items() if (k[0], k[2], k[4]) == (seed, stage, branch)]
            if len({tuple(r["batch_sha256"]) for r in group}) != 1:
                raise ValueError("different raw data across methods/arms")
            if len({r["start_diagnostics"]["initial_encoder_hash"] for r in group}) != 1:
                raise ValueError("different initial encoders")
        for model in c["models"]:
            if c["experiment"] == "decomposition":
                if len({records[seed, model, 1, a, "novel"]["before"]["model_sha256"] for a in RESET_ARMS}) != 1:
                    raise ValueError("decomposition did not use common learned weights")
            else:
                for arm in POLICIES:
                    for stage in (2, 3):
                        if records[seed, model, stage, arm, "novel"]["before"]["model_sha256"] != records[seed, model, stage-1, arm, "novel"]["after"]["model_sha256"]:
                            raise ValueError("policy trajectory broken")
                    end = records[seed, model, 3, arm, "novel"]["after"]["model_sha256"]
                    if any(records[seed, model, 4, arm, b]["before"]["model_sha256"] != end for b in BRANCHES):
                        raise ValueError("final branches not forked from own policy")
    return manifest, records, hashes


def qualification(rows, models):
    groups, by_model = {}, {}
    for model in models:
        fresh = [r for r in rows if r["model"] == model and r["arm"] == "fresh"]
        for dimension in ("stage", "cue"):
            for value in sorted({r[dimension] for r in fresh}):
                selected = [r for r in fresh if r[dimension] == value]
                gain, use = (mean(r[f] for r in selected) for f in ("marginal_gain", "cue_effect"))
                groups[f"{model}_{dimension}_{value}"] = dict(marginal_gain=gain, cue_effect=use,
                    acquisition_ok=gain >= .02, cue_use_ok=use >= .002, n=len(selected))
        by_model[model] = all(g["acquisition_ok"] and g["cue_use_ok"] for k, g in groups.items() if k.startswith(model+"_"))
    return dict(passed=all(by_model.values()), by_model=by_model, marginal_threshold=.02, cue_threshold=.002, groups=groups)


def contrasts(raw, config, branch, coefficients):
    stages = (1,) if config["experiment"] == "decomposition" else (1, 2, 3) if branch == "novel" else (4,)
    fields = (*FIELDS, "cue_effect") if branch in ("novel", "revision") else FIELDS
    return {m: {name: {f: paired([mean(sum(weight*raw[s, m, stage, arm, branch][f]
        for arm, weight in coeff.items()) for stage in stages) for s in config["seeds"]])
        for f in fields} for name, coeff in coefficients.items()} for m in config["models"]}


def policy_screen(novel, final, extra_noise, qualified):
    """Fixed mean-based pilot gates; interval warnings are descriptive, not tests."""
    gates = dict(fresh_qualification=dict(passed=bool(qualified)))

    def upper(name, metric, threshold):
        gates[name] = dict(passed=metric["mean"] <= threshold, value=metric["mean"], maximum=threshold,
            lower=metric["lower"], upper=metric["upper"], uncertainty_crosses_limit=metric["upper"] > threshold)

    upper("acquisition_gain", novel["brier_auc"], -.002)
    improved = sum(v < 0 for v in novel["brier_auc"]["differences"])
    gates["acquisition_consistency"] = dict(passed=improved >= 5, improved_seeds=improved, minimum=5,
        n=novel["brier_auc"]["n"])
    upper("introduction_retention", novel["valid_after"], .005)
    upper("extra_noise_penalty", extra_noise, .005)

    def survival(name, metric):
        gates[name] = dict(passed=metric["mean"] >= -.01, value=metric["mean"], minimum=-.01,
            lower=metric["lower"], upper=metric["upper"], uncertainty_crosses_limit=metric["lower"] < -.01)

    survival("introduction_survival", novel["survival_auc"])
    for branch in BRANCHES:
        upper(f"{branch}_prediction", final[branch]["brier_auc"], .005)
        upper(f"{branch}_retention", final[branch]["valid_after"], .005)
        survival(f"{branch}_survival", final[branch]["survival_auc"])
    return dict(passed=all(g["passed"] for g in gates.values()),
        failed=[k for k, g in gates.items() if not g["passed"]], gates=gates,
        interpretation="Prospective pilot screen on means; not a noninferiority test or general-CL claim.")


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    manifest, records, hashes = audit_records(directory)
    c = manifest["config"]
    raw = {k: metrics(r) for k, r in records.items()}
    rows = [dict(seed=k[0], model=k[1], stage=k[2], arm=k[3], branch=k[4], cue=records[k]["cue"], **v)
            for k, v in raw.items()]
    qualified = qualification(rows, c["models"])
    decomposition = c["experiment"] == "decomposition"
    result = dict(experiment=c["experiment"], identity=manifest["identity"], seeds=c["seeds"],
        episodes=len(records), qualification=qualified, rows=rows, result_hashes=hashes,
        contrasts=contrasts(raw, c, "novel", DECOMPOSITION if decomposition else POLICY_CONTRASTS))
    result["arm_means"] = {}
    result["cue_means"] = {}
    result["stage_means"] = {}
    for model in c["models"]:
        selected = [r for r in rows if r["model"] == model]
        result["arm_means"][model] = {branch: {arm: {f: mean(r[f] for r in selected if r["branch"] == branch and r["arm"] == arm)
            for f in (*FIELDS, "valid_before", "before", "training_seconds")}
            for arm in sorted({r["arm"] for r in selected if r["branch"] == branch})}
            for branch in sorted({r["branch"] for r in selected})}
        for dimension in ("cue", "stage"):
            result[f"{dimension}_means"][model] = {str(value): {arm: {f: mean(r[f] for r in selected
                if r["branch"] == "novel" and r[dimension] == value and r["arm"] == arm)
                for f in (*FIELDS, "cue_effect")}
                for arm in sorted({r["arm"] for r in selected if r["branch"] == "novel"})}
                for value in sorted({r[dimension] for r in selected if r["branch"] == "novel"})}
    if not decomposition:
        result["final_contrasts"] = {b: contrasts(raw, c, b, POLICY_CONTRASTS) for b in BRANCHES}
        result["noise_minus_clean"] = {m: {a: {f: paired([raw[s, m, 4, a, "noise"][f]-raw[s, m, 4, a, "clean"][f]
            for s in c["seeds"]]) for f in FIELDS} for a in POLICIES} for m in c["models"]}
        result["extra_noise"] = {m: {name: paired([sum(w*(raw[s, m, 4, a, "noise"]["brier_auc"]-
            raw[s, m, 4, a, "clean"]["brier_auc"]) for a, w in coeff.items()) for s in c["seeds"]])
            for name, coeff in POLICY_CONTRASTS.items()} for m in c["models"]}
        result["screen"] = {m: policy_screen(result["contrasts"][m]["split_minus_uniform"],
            {b: result["final_contrasts"][b][m]["split_minus_uniform"] for b in BRANCHES},
            result["extra_noise"][m]["split_minus_uniform"], qualified["by_model"][m]) for m in c["models"]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"summary.json", result)
    for name in ("manifest.json", "completion.json", "training_source.zip", "protocol_lock.json", "protocol_at_lock.md"):
        if (directory/name).exists():
            shutil.copyfile(directory/name, output/name)
    write_json(output/"config.json", c)
    with gzip.open(output/"raw_results.jsonl.gz", "wt", encoding="utf-8") as stream:
        for r in records.values():
            stream.write(json.dumps(r, allow_nan=False)+"\n")
    prefixes = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*/prefix.json"))}
    write_json(output/"prefix_records.json", prefixes)
    lines = [f"# Replay renewal: {c['experiment']}", "", f"{len(records)} complete episodes; paired seeds {c['seeds']}.", "",
        f"Fresh qualification: **{'PASS' if qualified['passed'] else 'FAIL'}**. Thresholds: marginal gain .02, cue benefit .002.", "",
        "Error values below are Brier x100. Negative contrast is improvement; intervals are paired seed bootstrap, descriptive.", "",
        "| Model | Contrast | Acquisition AUC [95% interval] | Terminal valid-old difference |", "|---|---|---:|---:|"]
    for model, comparisons in result["contrasts"].items():
        for name, values in comparisons.items():
            d, v = values["brier_auc"], values["valid_after"]
            lines.append(f"| {model} | {name} | {100*d['mean']:+.3f} [{100*d['lower']:+.3f}, {100*d['upper']:+.3f}] | {100*v['mean']:+.3f} |")
    if not decomposition:
        lines += ["", "## Prospective policy screen", ""]
        for model, screen in result["screen"].items():
            lines.append(f"* {model}: **{'PASS' if screen['passed'] else 'FAIL'}**. Failed gates: {', '.join(screen['failed']) or 'none'}.")
        lines += ["", "Full final-branch contrasts, every gate and interval warning are in summary.json."]
    lines += ["", "## Qualification groups", "", "| Group | Episodes | Marginal gain | Cue benefit | Pass |", "|---|---:|---:|---:|---|"]
    for key, value in qualified["groups"].items():
        lines.append(f"| {key} | {value['n']} | {100*value['marginal_gain']:.3f} | {100*value['cue_effect']:.3f} | {value['acquisition_ok'] and value['cue_use_ok']} |")
    lines += ["", "## Every episode", "", "| Seed | Model | Stage | Arm / branch | AUC | Valid-old error | Damage | Cue benefit |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        use = f"{100*r['cue_effect']:+.3f}" if "cue_effect" in r else "n/a"
        lines.append(f"| {r['seed']} | {r['model']} | {r['stage']} | {r['arm']} / {r['branch']} | {100*r['brier_auc']:.3f} | {100*r['valid_after']:.3f} | {100*r['valid_damage']:+.3f} | {use} |")
    (output/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(experiment=c["experiment"], episodes=len(records), qualification=qualified,
        screen={m: dict(passed=s["passed"], failed=s["failed"]) for m, s in result.get("screen", {}).items()}), indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
