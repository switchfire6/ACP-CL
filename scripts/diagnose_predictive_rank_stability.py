"""Retrospective rank stability from completed candidate losses; no model fitting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("conditional", "recurrent")
SEEDS = tuple(range(14001, 14013))
COMPARISONS = (
    ("score_to_near_frozen", "score", "near_frozen"),
    ("score_to_near_reapplied", "score", "near_reapplied"),
    ("score_to_return_frozen", "score", "return_frozen"),
    ("score_to_return_reapplied", "score", "return_reapplied"),
    ("near_frozen_to_reapplied", "near_frozen", "near_reapplied"),
    ("return_frozen_to_reapplied", "return_frozen", "return_reapplied"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def average_ranks(values):
    values = list(values)
    require(len(values) >= 2 and all(math.isfinite(value) for value in values),
            "at least two finite values are required")
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.]*len(values)
    start, tied_groups = 0, 0
    while start < len(order):
        stop = start+1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        tied_groups += int(stop-start > 1)
        rank = ((start+1)+stop)/2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks, tied_groups


def spearman(first, second):
    require(len(first) == len(second), "rank vectors differ in length")
    a, a_ties = average_ranks(first)
    b, b_ties = average_ranks(second)
    a_mean, b_mean = statistics.mean(a), statistics.mean(b)
    a = [value-a_mean for value in a]
    b = [value-b_mean for value in b]
    a_ss, b_ss = sum(value*value for value in a), sum(value*value for value in b)
    constant = ("both" if not a_ss and not b_ss else
                "left" if not a_ss else "right" if not b_ss else None)
    rho = None if constant else sum(x*y for x, y in zip(a, b))/math.sqrt(a_ss*b_ss)
    return dict(rho=rho, constant=constant, left_tied_groups=a_ties, right_tied_groups=b_ties)


def verify_known_examples():
    examples = (
        ("perfect", [1, 2, 3], [5, 7, 9], 1., None),
        ("opposite", [1, 2, 3], [9, 7, 5], -1., None),
        ("tied", [1, 1, 2, 3], [4, 5, 5, 6], 5/6, None),
        ("constant_left", [1, 1, 1], [1, 2, 3], None, "left"),
        ("constant_right", [1, 2, 3], [2, 2, 2], None, "right"),
        ("constant_both", [1, 1, 1], [2, 2, 2], None, "both"),
    )
    results = []
    for name, a, b, expected, constant in examples:
        actual = spearman(a, b)
        require(actual["constant"] == constant, f"constant handling failed: {name}")
        require(actual["rho"] is None if expected is None else
                math.isclose(actual["rho"], expected, rel_tol=0., abs_tol=1e-12),
                f"known rank example failed: {name}")
        results.append(dict(name=name, expected=expected, actual=actual, passed=True))
    require(average_ranks([1, 1, 2, 3])[0] == [1.5, 1.5, 3., 4.], "wrong average tie ranks")
    try:
        spearman([float("nan"), 1], [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite rank vector accepted")
    return results


def lock_sources(input_path, output):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    script, design = Path(__file__).resolve(), output/"design.md"
    require(design.is_file(), "write the retrospective design before running the analysis")
    files = {path.relative_to(ROOT).as_posix():sha(path) for path in (script, design)}
    input_record = dict(path=input_path.relative_to(ROOT).as_posix(), sha256=sha(input_path),
                        bytes=input_path.stat().st_size)
    lock_path, zip_path = output/"source_lock.json", output/"source.zip"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        require(lock["files"] == files and lock["input"] == input_record
                and lock["source_zip_sha256"] == sha(zip_path), "retrospective source/input lock changed")
    else:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                archive.write(ROOT/name, name)
        lock = dict(locked_utc=datetime.now(timezone.utc).isoformat(),
            designation="retrospective exploratory design after reviewing completed primary failures and secondary summaries",
            files=files, input=input_record, source_zip_sha256=sha(zip_path),
            python=platform.python_version(), before_correlations=True,
            no_fitting=True, no_new_gates=True, no_retuning=True)
        write_json(lock_path, lock)
    with zipfile.ZipFile(zip_path) as archive:
        require(len(archive.namelist()) == len(files) and set(archive.namelist()) == set(files)
                and archive.testzip() is None, "invalid retrospective source ZIP")
        require(all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in files.items()),
                "retrospective source ZIP contents changed")
    return lock


def collect(input_path):
    with gzip.open(input_path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    expected = {(seed, model, index) for seed in SEEDS for model in MODELS for index in (0, 1)}
    keys = [(row["seed"], row["model"], row["index"]) for row in records]
    require(len(keys) == len(expected) and set(keys) == expected, "incomplete or duplicate scientific cohort")
    identities = {json.dumps(row["identity"], sort_keys=True) for row in records}
    require(len(identities) == 1, "mixed scientific identities")
    return records


def compute(records):
    assessments = []
    for record in sorted(records, key=lambda row:(row["model"], row["seed"], row["index"])):
        ids = [row["id"] for row in record["candidates"]]
        require(len(ids) == len(set(ids)) == 8 and record["replacement"]["id"] not in ids,
                "wrong candidate/replacement coverage")
        vectors = {"score":record["choice"]["score_losses"]}
        vectors.update({name:record["panels"][name]["brier"]
                        for name in ("near_frozen", "near_reapplied", "return_frozen", "return_reapplied")})
        require(all(len(values) == 9 and all(math.isfinite(v) for v in values)
                    for values in vectors.values()), "invalid candidate-loss vector")
        correlations = {name:spearman(vectors[first][:-1], vectors[second][:-1])
                        for name, first, second in COMPARISONS}
        assessments.append(dict(seed=record["seed"], model=record["model"], index=record["index"],
            gap=record["gap"], candidate_ids=ids, replacement_id=record["replacement"]["id"],
            correlations=correlations))
    by_model = {}
    for model in MODELS:
        summaries = {}
        for name, _, _ in COMPARISONS:
            seeds = []
            undefined_assessments = []
            for seed in SEEDS:
                rows = [row for row in assessments if row["seed"] == seed and row["model"] == model]
                values = [row["correlations"][name]["rho"] for row in rows]
                missing = [row["index"] for row in rows if row["correlations"][name]["rho"] is None]
                undefined_assessments.extend(dict(seed=seed, index=row["index"],
                    constant=row["correlations"][name]["constant"]) for row in rows
                    if row["correlations"][name]["rho"] is None)
                seeds.append(dict(seed=seed, assessments=values,
                                  mean=None if missing else statistics.mean(values),
                                  undefined_indices=missing))
            valid = [row["mean"] for row in seeds if row["mean"] is not None]
            summaries[name] = dict(mean=statistics.mean(valid) if len(valid) == len(SEEDS) else None,
                positive=sum(value > 0 for value in valid), zero=sum(value == 0 for value in valid),
                negative=sum(value < 0 for value in valid), undefined_seeds=len(SEEDS)-len(valid),
                total_seeds=len(SEEDS), undefined_assessments=undefined_assessments, seeds=seeds)
        by_model[model] = summaries
    return assessments, by_model


def render(result):
    text = ["# Predictive-value rank stability: retrospective diagnosis", "",
        "Specified after reviewing the completed primary failures and secondary summaries. "
        "Exploratory only: no fitting, new gates, retuning or alteration of the failed decisions.", "",
        "Spearman correlations use exact tied-average ranks of the eight candidate Brier losses, "
        "excluding the common replacement. Two assessments are averaged within each seed before "
        "the twelve seed means are averaged. Positive values indicate agreement in loss ordering. "
        "No inferential intervals or p-values are computed.", "",
        "| Comparison | Conditional mean | Positive / 12 | Recurrent mean | Positive / 12 |",
        "|---|---:|---:|---:|---:|"]
    def display(value):
        return "undefined" if value is None else f"{value:+.4f}"
    for name, first, second in COMPARISONS:
        a, b = (result["models"][model][name] for model in MODELS)
        text.append(f"| {first} → {second} | {display(a['mean'])} | {a['positive']} | {display(b['mean'])} | {b['positive']} |")
    undefined = [dict(model=model, comparison=name, **row)
                 for model in MODELS for name, summary in result["models"][model].items()
                 for row in summary["undefined_assessments"]]
    text += ["", f"Undefined constant-vector assessment comparisons: {len(undefined)}. "
        "None are silently dropped; any undefined assessment makes its seed mean undefined, "
        "and any undefined seed prevents the overall mean.", ""]
    for model in MODELS:
        text += [f"## {model.capitalize()}: all seed means", "",
            "| Seed | Score→near frozen | Score→near reapplied | Score→return frozen | Score→return reapplied | Near frozen→reapplied | Return frozen→reapplied |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for index, seed in enumerate(SEEDS):
            values = [result["models"][model][name]["seeds"][index]["mean"] for name, _, _ in COMPARISONS]
            text.append(f"| {seed} | "+" | ".join(display(value) for value in values)+" |")
        text.append("")
    text += ["## Interpretation limits", "",
        "Frozen/reapplied comparisons share later queries and context but change the parent weights, "
        "Adam state and latest anchor. These correlations do not isolate why ordering changed. "
        "Rank agreement is separate from gain size, calibration, survival and retention. "
        "The original local and return criteria remain failed; a favorable secondary ordering "
        "does not promote a replacement policy.", "",
        "See [design.md](design.md) for exact formulas and constant handling, "
        "[results.json](results.json) for every assessment and verification example, and "
        "[source_lock.json](source_lock.json) / [source.zip](source.zip) for the retrospective snapshot.", "",
        f"Input compressed-file SHA256: `{result['input']['sha256']}`.",
        f"Analysis script SHA256: `{result['code_sha256']}`.", ""]
    return "\n".join(text)


def run(input_path, output):
    output = Path(output)
    require(not (output/"results.json").exists() and not (output/"results.md").exists(),
            "refusing to replace existing retrospective results")
    examples = verify_known_examples()
    lock = lock_sources(input_path, output)
    records = collect(input_path)
    assessments, models = compute(records)
    require(sha(input_path) == lock["input"]["sha256"], "raw results changed during reading")
    result = dict(designation=lock["designation"], input=lock["input"],
        code_sha256=sha(Path(__file__)), design_sha256=sha(output/"design.md"),
        source_lock_sha256=sha(output/"source_lock.json"), source_zip_sha256=lock["source_zip_sha256"],
        scientific_identity=records[0]["identity"], completed_utc=datetime.now(timezone.utc).isoformat(),
        counts=dict(models=2, seeds_per_model=12, assessments=48, comparisons_per_assessment=6,
                    assessment_correlations=288),
        formulas=dict(ranks="Ascending 1-based ranks; exact ties receive mean occupied rank.",
            rho="sum((r-rbar)*(q-qbar))/sqrt(sum((r-rbar)^2)*sum((q-qbar)^2))",
            seed_mean="Arithmetic mean of both assessment correlations; undefined if either is constant.",
            model_mean="Arithmetic mean of all12 seed means; undefined if any seed is undefined."),
        verification_examples=examples, assessments=assessments, models=models,
        no_fitting=True, no_new_gates=True, no_retuning=True,
        limitations="Retrospective, descriptive rank transport; no causal age attribution or policy validation.")
    write_json(output/"results.json", result)
    (output/"results.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({model:{name:dict(mean=row["mean"], positive=row["positive"],
        undefined_seeds=row["undefined_seeds"]) for name, row in models[model].items()}
        for model in MODELS}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="reports/predictive_value/diagnostic/raw_results.jsonl.gz")
    parser.add_argument("--output", default="reports/predictive_value_checks/rank_stability")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(verify_known_examples(), indent=2))
    else:
        run(args.input, args.output)
