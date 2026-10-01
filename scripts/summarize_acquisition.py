"""Complete seed-paired acquisition diagnostics and fresh-model qualification."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from acp_cl.acquisition.study import ARMS, BRANCHES
from acp_cl.acquisition.world import final_law, order, stage_law
from acp_cl.persistence.study import digest, write_json


def mean(values):
    return float(np.mean(list(values)))


def paired(values):
    values = np.array(values, dtype=float)
    values[np.abs(values) < 1e-12] = 0
    rng = np.random.default_rng(27192026)
    boot = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    return dict(mean=mean(values), lower=float(np.quantile(boot, .025)),
        upper=float(np.quantile(boot, .975)), positive=int((values > 0).sum()),
        n=len(values), differences=values.tolist())


def probe(p, field, flip=False, entry=False):
    return mean(r["flipped"][field] if flip else r["curve"][0 if entry else -1]["metrics"][field]
                for r in p["replicates"])


def area(curve, field):
    x = np.array([p["arrivals"] for p in curve])
    y = np.array([p["metrics"][field] for p in curve])
    return float(np.sum(np.diff(x)*(y[:-1]+y[1:])/2)/x[-1])


def metrics(r):
    result = dict(brier_auc=area(r["curve"], "focus_brier"),
        all_brier_auc=area(r["curve"], "brier"), survival_auc=area(r["curve"], "survival"),
        before=probe(r["before"], "focus_brier"), after=probe(r["after"], "focus_brier"),
        endpoint=r["curve"][-1]["metrics"]["focus_brier"],
        after_survival=probe(r["after"], "survival"),
        before_survival=probe(r["before"], "survival"),
        marginal_gain=probe(r["after"], "marginal_brier")-probe(r["after"], "brier"),
        valid_before=mean(v["valid_brier"] for v in r["valid_before"].values()),
        valid_after=mean(v["valid_brier"] for v in r["valid_after"].values()),
        valid_survival_before=mean(v["valid_survival"] for v in r["valid_before"].values()),
        valid_survival_after=mean(v["valid_survival"] for v in r["valid_after"].values()),
        training_seconds=r["diagnostics"]["cost"]["training_seconds"]-r["start_diagnostics"]["cost"]["training_seconds"])
    result["valid_damage"] = result["valid_after"]-result["valid_before"]
    if r["cue"] is not None:
        result["cue_effect"] = probe(r["after"], "focus_brier", flip=True)-result["after"]
        result["cue_effect_before"] = probe(r["before"], "focus_brier", flip=True)-result["before"]
        result["cue_learning_gain"] = result["cue_effect"]-result["cue_effect_before"]
    return result


def audit_records(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    config, identity = manifest["config"], manifest["identity"]
    for name, content in (("config_sha256", config), ("source_sha256", manifest["source_files"]),
                          ("runtime_sha256", manifest["runtime"])):
        if digest(content) != identity[name]:
            raise ValueError("manifest identity mismatch")
    lock = None
    if (directory/"protocol_lock.json").exists():
        lock = json.loads((directory/"protocol_lock.json").read_text(encoding="utf-8"))
        if any(lock[k] != v for k, v in identity.items()) or hashlib.sha256(
                (directory/"protocol_at_lock.md").read_bytes()).hexdigest() != lock["protocol_sha256"]:
            raise ValueError("protocol lock mismatch")
    with zipfile.ZipFile(directory/"training_source.zip") as archive:
        for name, expected in manifest["source_files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("source archive mismatch")
    arms = ("fresh",) if config["kind"] == "development" else ARMS
    expected = {(s, m, k, a, "novel") for s in config["seeds"] for m in config["models"]
                for k in range(1, 4) for a in arms}
    if config["kind"] == "diagnostic":
        expected |= {(s, m, 4, "continue", b) for s in config["seeds"] for m in config["models"] for b in BRANCHES}
    records, hashes = {}, []
    for path in sorted(directory.glob("*/*/result.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        key = r["seed"], r["model"], r["stage"], r["arm"], r["branch"]
        if key in records or key not in expected or r["identity"] != identity:
            raise ValueError("unexpected/duplicate/identity-mismatched result")
        law = stage_law(r["seed"], r["stage"]) if r["branch"] == "novel" else final_law(r["seed"], r["branch"], config["feedback_noise"])
        if digest(asdict(law)) != digest(r["law"]):
            raise ValueError("wrong physical condition")
        cue = order(r["seed"])[r["stage"]-1] if r["branch"] == "novel" else 0 if r["branch"] == "revision" else None
        if r["cue"] != cue:
            raise ValueError("wrong new dependency")
        if r["batches_done"] != config["episode_size"]//config["batch_size"] or len(r["batch_sha256"]) != r["batches_done"]:
            raise ValueError("incomplete episode")
        if [p["arrivals"] for p in r["curve"]] != list(range(0, config["episode_size"]+1, config["probe_every"])):
            raise ValueError("incomplete acquisition curve")
        before, after = r["start_diagnostics"], r["diagnostics"]
        expected_cost = dict(arrivals=config["episode_size"],
            optimizer_steps=config["episode_size"]//config["batch_size"]*config["updates_per_batch"],
            query_presentations=2*config["episode_size"]*config["updates_per_batch"],
            support_presentations=2*config["episode_size"]*config["updates_per_batch"])
        if any(after["cost"][k]-before["cost"][k] != v for k, v in expected_cost.items()):
            raise ValueError("different arrival/update/exposure budget")
        if r["before"]["model_sha256"] != before["final_hash"] or r["after"]["model_sha256"] != after["final_hash"]:
            raise ValueError("probe/model mismatch")
        if r["arm"] == "frozen" and before["encoder_hash"] != after["encoder_hash"]:
            raise ValueError("frozen encoder changed")
        if r["arm"] == "fresh" and before["final_hash"] != before["initial_hash"]:
            raise ValueError("fresh model did not start randomly initialized")
        if r["arm"] in ("fresh", "state_reset") and before["memory_ids"]:
            raise ValueError("reset memory is not empty")
        records[key] = r
        hashes.append(dict(path=path.relative_to(directory).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if set(records) != expected:
        raise ValueError(f"missing {len(expected-set(records))} episodes")
    completion = json.loads((directory/"completion.json").read_text(encoding="utf-8"))
    if completion["identity"] != identity or completion["episodes"] != len(expected):
        raise ValueError("completion mismatch")
    trajectories = len(config["seeds"])*len(config["models"])
    if completion["trajectories"] != trajectories or completion["prefix_fits"] != (
            trajectories if config["kind"] == "diagnostic" else 0):
        raise ValueError("trajectory/prefix completion mismatch")
    for seed in config["seeds"]:
        for stage in range(1, 4):
            group = [records[seed, m, stage, a, "novel"] for m in config["models"] for a in arms]
            if any(r["batch_sha256"] != group[0]["batch_sha256"] for r in group):
                raise ValueError("methods/arms received different training data")
            if len({r["start_diagnostics"]["initial_encoder_hash"] for r in group}) != 1:
                raise ValueError("different initial visual encoders")
            if config["kind"] == "diagnostic":
                for model in config["models"]:
                    g = [records[seed, model, stage, a, "novel"] for a in ("continue", "frozen", "state_reset")]
                    if len({r["before"]["model_sha256"] for r in g}) != 1:
                        raise ValueError("counterfactual arms did not start at identical learned weights")
                    if stage > 1 and g[0]["before"]["model_sha256"] != records[seed, model, stage-1, "continue", "novel"]["after"]["model_sha256"]:
                        raise ValueError("trajectory did not carry the continuing arm")
        if config["kind"] == "diagnostic":
            for model in config["models"]:
                final_hash = records[seed, model, 3, "continue", "novel"]["after"]["model_sha256"]
                for branch in BRANCHES:
                    if records[seed, model, 4, "continue", branch]["before"]["model_sha256"] != final_hash:
                        raise ValueError("final branches do not share a starting model")
    return manifest, lock, records, hashes


def qualification(rows, models):
    groups = {}
    for model in models:
        for dimension, values in (("stage", (1, 2, 3)), ("cue", (0, 1, 2))):
            for value in values:
                selected = [r for r in rows if r["model"] == model and r["arm"] == "fresh"
                            and r["branch"] == "novel" and r[dimension] == value]
                gain, use = (mean(r[k] for r in selected) for k in ("marginal_gain", "cue_effect"))
                groups[f"{model}_{dimension}_{value}"] = dict(marginal_gain=gain, cue_effect=use,
                    acquisition_ok=gain >= .02, cue_use_ok=use >= .002, n=len(selected))
    return dict(passed=all(r["acquisition_ok"] and r["cue_use_ok"] for r in groups.values()), groups=groups)


def diagnostic_groups(rows, models):
    """Descriptive summaries of the already-declared cue and final-branch probes."""
    cues, final = {}, {}
    for model in models:
        cues[model], final[model] = {}, {}
        for cue in range(3):
            cues[model][str(cue)] = {}
            for arm in ARMS:
                group = sorted((r for r in rows if r["model"] == model and r["cue"] == cue
                    and r["arm"] == arm and r["branch"] == "novel"), key=lambda r: r["seed"])
                cues[model][str(cue)][arm] = dict(n=len(group), **{k: mean(r[k] for r in group)
                    for k in ("brier_auc", "before", "after", "cue_effect_before", "cue_effect",
                              "cue_learning_gain", "valid_damage")})
        for branch in BRANCHES:
            group = sorted((r for r in rows if r["model"] == model and r["branch"] == branch),
                           key=lambda r: r["seed"])
            final[model][branch] = dict(n=len(group), means={k: mean(r[k] for r in group)
                for k in ("brier_auc", "before", "after", "before_survival", "after_survival", "valid_damage")},
                brier_change=paired([r["after"]-r["before"] for r in group]),
                survival_change=paired([r["after_survival"]-r["before_survival"] for r in group]),
                valid_damage=paired([r["valid_damage"] for r in group]))
            if branch == "revision":
                final[model][branch]["cue_effect_before"] = mean(r["cue_effect_before"] for r in group)
                final[model][branch]["cue_effect_after"] = mean(r["cue_effect"] for r in group)
    return cues, final


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    manifest, lock, records, hashes = audit_records(directory)
    c = manifest["config"]
    raw = {key: metrics(r) for key, r in records.items()}
    rows = [dict(seed=k[0], model=k[1], stage=k[2], arm=k[3], branch=k[4], cue=records[k]["cue"], **v) for k, v in raw.items()]
    qualified = qualification(rows, c["models"])
    result = dict(kind=c["kind"], identity=manifest["identity"], qualification=qualified, rows=rows,
                  seeds=c["seeds"], episodes=len(records), result_hashes=hashes)
    if c["kind"] == "diagnostic":
        comparisons = (("continue", "fresh"), ("continue", "frozen"),
                       ("state_reset", "continue"), ("state_reset", "fresh"))
        contrasts = {}
        for model in c["models"]:
            contrasts[model] = {}
            for a, b in comparisons:
                # A minus B in raw metric units. Brier/valid-damage: lower is better.
                contrasts[model][f"{a}_minus_{b}"] = {field: paired([
                    mean(raw[s, model, k, a, "novel"][field]-raw[s, model, k, b, "novel"][field] for k in range(1, 4))
                    for s in c["seeds"]]) for field in ("brier_auc", "valid_damage", "survival_auc", "cue_effect")}
        result["contrasts"] = contrasts
        result["relative_late_minus_early"] = {m: paired([
            (raw[s, m, 3, "continue", "novel"]["brier_auc"]-raw[s, m, 3, "fresh", "novel"]["brier_auc"])
            -(raw[s, m, 1, "continue", "novel"]["brier_auc"]-raw[s, m, 1, "fresh", "novel"]["brier_auc"])
            for s in c["seeds"]]) for m in c["models"]}
        result["cue_groups"], result["final_branches"] = diagnostic_groups(rows, c["models"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"summary.json", result)
    for name in ("manifest.json", "training_source.zip", "completion.json", "protocol_lock.json", "protocol_at_lock.md"):
        if (directory/name).exists():
            shutil.copy2(directory/name, output/name)
    write_json(output/"config.json", c)
    with gzip.GzipFile(filename=str(output/"records.json.gz"), mode="wb", mtime=0) as f:
        f.write(json.dumps([records[k] for k in sorted(records)], separators=(",", ":"), allow_nan=False).encode())
    if c["kind"] == "diagnostic":
        prefixes = []
        for seed in c["seeds"]:
            for model in c["models"]:
                path = directory/f"{model}_{seed}"/"prefix.json"
                prefix = json.loads(path.read_text(encoding="utf-8"))
                if (prefix["identity"] != manifest["identity"] or prefix["seed"] != seed
                        or prefix["model"] != model or len(prefix["blocks"]) != c["prefix_blocks"]):
                    raise ValueError("prefix result identity/completion mismatch")
                prefixes.append(dict(path=path.relative_to(directory).as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), record=prefix))
        with gzip.GzipFile(filename=str(output/"prefix_records.json.gz"), mode="wb", mtime=0) as f:
            f.write(json.dumps(prefixes, separators=(",", ":"), allow_nan=False).encode())
    lines = ["# Acquisition diagnostic results", "", f"Kind: {c['kind']}; {len(c['seeds'])} independent seeds; {len(records)} episodes.",
        f"Fresh-model qualification: **{qualified['passed']}**.", "", "## Qualification", "",
        "| Group | Marginal Brier improvement | Correct-cue Brier benefit | Pass |", "|---|---:|---:|---|"]
    for name, row in qualified["groups"].items():
        lines.append(f"| {name} | {row['marginal_gain']:.5f} | {row['cue_effect']:.5f} | {row['acquisition_ok'] and row['cue_use_ok']} |")
    lines += ["", "## Every episode", "", "Brier errors below are multiplied by 100; lower is better. Positive valid damage is forgetting.", "",
        "| Seed | Model | Stage | Arm/branch | Cue | Brier AUC | Valid damage | Cue benefit |", "|---|---|---:|---|---|---:|---:|---:|"]
    for r in rows:
        cue_effect = f"{100*r['cue_effect']:+.3f}" if "cue_effect" in r else "n/a"
        lines.append(f"| {r['seed']} | {r['model']} | {r['stage']} | {r['arm']}/{r['branch']} | {r['cue']} | "
                     f"{100*r['brier_auc']:.3f} | {100*r['valid_damage']:+.3f} | {cue_effect} |")
    if "contrasts" in result:
        lines += ["", "## Paired acquisition contrasts", "", "Episodes averaged within seed before bootstrap; lower favors the first arm.", "",
            "| Model | A minus B | Brier AUC difference x100 | 95% descriptive interval |", "|---|---|---:|---|"]
        for model, contrasts in result["contrasts"].items():
            for name, row in contrasts.items():
                r = row["brier_auc"]
                lines.append(f"| {model} | {name} | {100*r['mean']:+.3f} | [{100*r['lower']:+.3f}, {100*r['upper']:+.3f}] |")
        lines += ["", "## Final branches", "", "Before/after probes use the same clean, fresh support protocol. "
            "Brier refers to the affected subset for revision and all queries for return/noise. "
            "All values below are multiplied by 100. These are secondary descriptive outcomes.", "",
            "| Model | Branch | Brier before | Brier after | Brier change | Valid-old damage | Survival before | Survival after |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
        for model, branches in result["final_branches"].items():
            for branch, group in branches.items():
                m = group["means"]
                lines.append(f"| {model} | {branch} | {100*m['before']:.3f} | {100*m['after']:.3f} | "
                    f"{100*group['brier_change']['mean']:+.3f} | {100*m['valid_damage']:+.3f} | "
                    f"{100*m['before_survival']:.3f} | {100*m['after_survival']:.3f} |")
    (output/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(kind=result["kind"], episodes=len(records), qualification=qualified), indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
