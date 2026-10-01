"""Complete paired analysis of the matched-history comparison."""

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

from acp_cl.contextual.world import challenge, prefix
from acp_cl.persistence.study import digest, write_json


def mean(values):
    value = float(np.mean(list(values)))
    return 0. if abs(value) < 1e-12 else value


def paired(values):
    values = np.asarray(values, dtype=float)
    values[np.abs(values) < 1e-12] = 0
    rng = np.random.default_rng(4172026)
    boot = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    return dict(mean=mean(values), lower=float(np.quantile(boot, .025)),
                upper=float(np.quantile(boot, .975)), positive=int((values > 0).sum()),
                n=len(values), differences=values.tolist())


def score(probe, kind="correct", field="survival"):
    def extract(rep):
        if kind in ("correct", "entry"):
            return rep["curve"][-1 if kind == "correct" else 0]["metrics"][field]
        return rep[kind][field]
    return mean(extract(rep) for rep in probe["replicates"])


def auc(curve):
    x = np.asarray([p["arrivals"] for p in curve])
    y = np.asarray([p["metrics"]["survival"] for p in curve])
    return float(np.sum(np.diff(x)*(y[:-1]+y[1:])/2)/x[-1])


def metrics(r):
    return dict(before=score(r["before"]), entry=score(r["before"], "entry"),
        before_no_transfer=score(r["before"], field="no_transfer"),
        before_history_effect=score(r["before"])-score(r["before"], "opposite"),
        acquisition_auc=auc(r["curve"]), endpoint=r["curve"][-1]["metrics"]["survival"],
        after=score(r["after"]), known_after=mean(score(p) for p in r["known"].values()),
        after_brier=score(r["after"], field="brier"),
        initial_gain=mean(b["curve"][-1]["metrics"]["survival"]
            - b["curve"][-1]["metrics"]["no_transfer"] for b in r["prefix"]["blocks"][:4]),
        training_seconds=r["diagnostics"]["cost"]["training_seconds"])


def adequacy_checks(recurrent, oracle):
    return dict(oracle_acquisition=oracle["initial_gain"] >= .05,
        recurrent_acquisition=recurrent["initial_gain"] >= .05,
        recurrent_return=recurrent["before"]-recurrent["before_no_transfer"] >= .05,
        recurrent_uses_history=recurrent["before_history_effect"] >= .02)


def aggregate_seeds(raw, config):
    # Repeated gap orders are paired measurements, not independent samples.
    values = {(s, m, b): {k: mean(raw[s, g, m, b][k] for g in config["gaps"])
                          for k in next(iter(raw.values()))}
              for s in config["seeds"] for m in config["methods"] for b in config["branches"]}
    aggregate = {b: {m: {k: mean(values[s, m, b][k] for s in config["seeds"])
                         for k in next(iter(raw.values()))} for m in config["methods"]} for b in config["branches"]}
    return values, aggregate


def audit_records(directory, require_lock=True):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    config, identity = manifest["config"], manifest["identity"]
    lock_path = directory/"protocol_lock.json"
    if require_lock or lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    else:
        lock = None
    for key, item in (("config_sha256", config), ("source_sha256", manifest["source_files"]),
                      ("runtime_sha256", manifest["runtime"])):
        if digest(item) != identity[key] or (lock is not None and identity[key] != lock[key]):
            raise ValueError("protocol/source/config/runtime mismatch")
    if lock is not None and hashlib.sha256((directory/"protocol_at_lock.md").read_bytes()).hexdigest() != lock["protocol_sha256"]:
        raise ValueError("protocol text mismatch")
    with zipfile.ZipFile(directory/"training_source.zip") as z:
        for name, value in manifest["source_files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != value:
                raise ValueError("archived source mismatch")
    expected = {(s, g, m, b) for s in config["seeds"] for g in config["gaps"]
                for m in config["methods"] for b in config["branches"]}
    records, artifacts = {}, []
    for path in sorted(directory.glob("*/result.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        key = r["seed"], r["gap"], r["method"], r["branch"]
        if key not in expected or key in records or r["identity"] != identity:
            raise ValueError("unexpected/duplicate/mismatched run")
        plan = prefix(r["seed"], r["gap"], config)
        if len(r["prefix"]["blocks"]) != 8 or r["condition"] != asdict(challenge(r["seed"], r["branch"], config)):
            raise ValueError("phase plan mismatch")
        for actual, block in zip(r["prefix"]["blocks"], plan):
            if any(actual[k] != v for k, v in block.items()):
                raise ValueError("prefix mismatch")
            if len(actual["batch_sha256"]) != block["size"]//config["batch_size"]:
                raise ValueError("missing prefix arrivals")
        if r["batches_done"] != config["block_size"]//config["batch_size"]:
            raise ValueError("incomplete challenge")
        if [p["arrivals"] for p in r["curve"]] != list(range(0, config["block_size"]+1, config["probe_every"])):
            raise ValueError("incomplete acquisition curve")
        cost = r["diagnostics"]["cost"]
        arrivals = sum(b["size"] for b in plan)+config["block_size"]
        if cost["arrivals"] != arrivals or cost["optimizer_steps"] != arrivals//config["batch_size"]*config["updates_per_batch"]:
            raise ValueError("different arrival/update budget")
        if any(cost[k] != 2*arrivals*config["updates_per_batch"] for k in ("query_presentations", "support_presentations")):
            raise ValueError("different observation exposures")
        if len(r["diagnostics"]["memory_ids"]) != config["memory_packets"]:
            raise ValueError("memory capacity mismatch")
        if any(p["model_sha256"] != r["diagnostics"]["final_hash"] for p in [r["after"], *r["known"].values()]):
            raise ValueError("probe weight mismatch")
        if r["method"] == "recurrent_frozen":
            frozen_hash = r["prefix"]["blocks"][3]["encoder_hash"]
            if any(b["encoder_hash"] != frozen_hash for b in r["prefix"]["blocks"][4:]) or r["diagnostics"]["encoder_hash"] != frozen_hash:
                raise ValueError("frozen encoder changed")
        records[key] = r
        artifacts.append(dict(path=path.relative_to(directory).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if set(records) != expected:
        raise ValueError(f"missing {len(expected-set(records))} branches")
    completion = json.loads((directory/"completion.json").read_text(encoding="utf-8"))
    if (completion["identity"] != identity or completion["branches"] != len(records)
            or completion["prefixes"] != len(config["seeds"])*len(config["gaps"])*len(config["methods"])):
        raise ValueError("completion mismatch")
    for seed in config["seeds"]:
        for gap in config["gaps"]:
            for method in config["methods"]:
                group = [records[seed, gap, method, b] for b in config["branches"]]
                if len({r["before"]["model_sha256"] for r in group}) != 1 or any(r["prefix"] != group[0]["prefix"] for r in group):
                    raise ValueError("branches do not share identical prefix learning")
            for branch in config["branches"]:
                group = [records[seed, gap, m, branch] for m in config["methods"]]
                for r in group:
                    for key in ("batch_sha256",):
                        if r[key] != group[0][key]:
                            raise ValueError("different challenge data")
                    if [b["batch_sha256"] for b in r["prefix"]["blocks"]] != [b["batch_sha256"] for b in group[0]["prefix"]["blocks"]]:
                        raise ValueError("different prefix data")
                    if r["diagnostics"]["initial_encoder_hash"] != group[0]["diagnostics"]["initial_encoder_hash"]:
                        raise ValueError("different visual initialization")
                    if r["diagnostics"]["memory_ids"] != group[0]["diagnostics"]["memory_ids"]:
                        raise ValueError("different reservoir selections")
            if {"recurrent", "recurrent_frozen"} <= set(config["methods"]):
                a, b = (records[seed, gap, m, config["branches"][0]] for m in ("recurrent", "recurrent_frozen"))
                if a["prefix"]["blocks"][:4] != b["prefix"]["blocks"][:4]:
                    raise ValueError("freeze control differs before intervention")
        if {"short", "long"} <= set(config["gaps"]):
            samples = [sorted(h for b in records[seed, g, config["methods"][0], config["branches"][0]]["prefix"]["blocks"]
                              for h in b["batch_sha256"]) for g in ("short", "long")]
            if samples[0] != samples[1]:
                raise ValueError("short/long prefix data multisets differ")
    return manifest, lock, records, artifacts


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, lock, records, artifacts = audit_records(directory)
    config = manifest["config"]
    raw = {k: metrics(r) for k, r in records.items()}
    # The independent unit is the seed. Average the two gap orders within seed.
    values, aggregate = aggregate_seeds(raw, config)
    gap_tables = {g: {b: {m: {k: mean(raw[s, g, m, b][k] for s in config["seeds"])
                              for k in next(iter(raw.values()))} for m in config["methods"]}
                      for b in config["branches"]} for g in config["gaps"]}
    comparisons = [("query_routed", "recurrent"), ("query_routed", "conditional"),
                   ("recurrent", "conditional"), ("recurrent", "pooled"),
                   ("recurrent", "recurrent_frozen")]
    contrasts = {b: {f"{a}_minus_{c}": {k: paired([values[s, a, b][k]-values[s, c, b][k] for s in config["seeds"]])
                                       for k in next(iter(raw.values()))}
                      for a, c in comparisons} for b in config["branches"]}
    primary = contrasts["novel"]["query_routed_minus_recurrent"]["acquisition_auc"]
    old_comparison = contrasts["novel"]["query_routed_minus_conditional"]["acquisition_auc"]
    r, o = aggregate["return"]["recurrent"], aggregate["return"]["oracle"]
    adequacy = adequacy_checks(r, o)
    criteria = dict(new_learning_gain_over_recurrent_2pp=primary["mean"] >= .02,
        new_learning_positive_6_seeds=primary["positive"] >= 6,
        new_learning_gain_over_original_2pp=old_comparison["mean"] >= .02,
        new_learning_over_original_positive_6_seeds=old_comparison["positive"] >= 6,
        return_loss_at_most_2pp=contrasts["return"]["query_routed_minus_conditional"]["before"]["mean"] >= -.02,
        valid_retention_loss_at_most_2pp=contrasts["novel"]["query_routed_minus_conditional"]["known_after"]["mean"] >= -.02,
        noise_loss_vs_recurrent_at_most_2pp=contrasts["noise"]["query_routed_minus_recurrent"]["acquisition_auc"]["mean"] >= -.02,
        noise_loss_vs_original_at_most_2pp=contrasts["noise"]["query_routed_minus_conditional"]["acquisition_auc"]["mean"] >= -.02)
    result = dict(aggregate=aggregate, gaps=gap_tables, contrasts=contrasts, primary=primary,
                  adequacy=adequacy, criteria=criteria,
                  decision="inconclusive" if not all(adequacy.values()) else "pass" if all(criteria.values()) else "fail",
                  seeds=config["seeds"], prefixes=len(config["seeds"])*len(config["gaps"])*len(config["methods"]),
                  branches=len(records), result_hashes=artifacts)
    write_json(output/"summary.json", result)
    for name in ("manifest.json", "protocol_lock.json", "protocol_at_lock.md", "training_source.zip", "completion.json"):
        shutil.copy2(directory/name, output/name)
    with gzip.GzipFile(filename=str(output/"archive.json.gz"), mode="wb", mtime=0) as f:
        f.write(json.dumps(dict(manifest=manifest, lock=lock, summary=result,
                               records=[records[k] for k in sorted(records)]), separators=(",", ":"), allow_nan=False).encode())
    lines = ["# Matched-history comparison: complete results", "",
             f"{result['prefixes']} shared prefixes; {result['branches']} branches; decision **{result['decision']}**.", "",
             f"Percentages; gap orders averaged within seed before paired analysis. {len(config['seeds'])} independent seeds.", ""]
    for branch, methods in aggregate.items():
        lines += [f"## {branch}", "", "| Method | Before, 32 feedback | Learning AUC | Endpoint | After, 32 feedback | Valid old modes |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method, row in methods.items():
            lines.append(f"| {method} | "+" | ".join(f"{100*row[k]:.2f}" for k in
                         ("before", "acquisition_auc", "endpoint", "after", "known_after"))+" |")
        lines += [""]
    lines += ["## Primary difference by seed", "", "| Seed | Query routing minus recurrent, new-learning AUC (pp) |", "|---|---:|"]
    lines += [f"| {s} | {100*d:+.2f} |" for s, d in zip(config["seeds"], primary["differences"])]
    lines += ["", f"Mean {100*primary['mean']:+.2f} pp; descriptive 95% paired interval "
              f"[{100*primary['lower']:+.2f}, {100*primary['upper']:+.2f}].", "", "## Adequacy and decision", ""]
    lines += [f"- {k}: **{v}**" for k, v in {**adequacy, **criteria}.items()]
    (output/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("primary", "adequacy", "criteria", "decision")}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
