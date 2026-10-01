"""Independent NumPy scoring of the retrospective saved-model output diagnostic.

No training/source-module imports. Replicate supports are averaged within each
seed before descriptive paired-seed intervals and fixed allocation heuristics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
from itertools import permutations
import json
from pathlib import Path

import numpy as np


MODELS = ("conditional", "recurrent")
ARMS = ("joint", "separate")
TARGETS = ("novel_end", "return_entry", "return_end", "novel_after_return")
NOVEL = ("novel_end", "novel_after_return")
METRICS = ("brier", "focus_brier", "survival", "focus_survival", "valid_brier",
           "valid_survival", "no_transfer", "clairvoyant_upper")
CELLS = dict(conditional=("full", "core", "residual", "full_core_route", "core_full_route"),
             recurrent=("full", "core", "residual"))
SEEDS = tuple(range(16001, 16007))
GROUP = ("model", "arm", "target", "context", "flipped")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(value):
    return sha(str((value.shape, value.dtype.str)).encode()+value.tobytes(order="C"))


def support_hash(values):
    digest = hashlib.sha256()
    for name in ("support_observations", "support_actions", "support_outcomes"):
        value = values[name]
        digest.update(str((value.shape,value.dtype.str)).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON: "+value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def score(prediction, truth, affected, valid):
    size = len(truth)
    tolerance = np.finfo(np.float32).eps
    require(size > 0 and prediction.dtype == np.float32 and prediction.shape == (size, 5, 3)
            and np.isfinite(prediction).all() and np.all(prediction >= -tolerance)
            and np.all(prediction <= 1+tolerance) and np.all(np.diff(prediction, axis=-1) <= tolerance),
            "invalid raw survival predictions")
    require(truth.dtype == np.uint8 and truth.shape == prediction.shape and np.isin(truth, (0, 1)).all()
            and np.all(np.diff(truth.astype(np.int16), axis=-1) <= 0), "invalid physical truth")
    for mask in (affected, valid):
        require(mask.dtype == np.bool_ and mask.shape == (size,) and mask.any(), "invalid scoring mask")
    errors = np.square(prediction.astype(np.float64)-truth.astype(np.float64)).mean(axis=(1, 2))
    chosen = prediction[:, :, -1].argmax(axis=1)
    survived = truth[np.arange(size), chosen, -1].astype(np.float64)
    return dict(brier=float(errors.mean()), focus_brier=float(errors[affected].mean()),
        survival=float(survived.mean()), focus_survival=float(survived[affected].mean()),
        valid_brier=float(errors[valid].mean()), valid_survival=float(survived[valid].mean()),
        no_transfer=float(truth[:, 0, -1].mean()), clairvoyant_upper=float(truth[:, :, -1].max(axis=1).mean()))


def row_key(row):
    return tuple(row[name] for name in ("arm", "target", "context", "replicate", "flipped"))


def expected_rows(replicates=2):
    return {(arm,target,context,replicate,flipped) for arm in ARMS for target in TARGETS
            for context in ("actual", "refreshed") for replicate in ((0,) if context == "actual" else range(replicates))
            for flipped in ((False, True) if context == "refreshed" and target in NOVEL else (False,))}


def read_jobs(directory, development=False):
    directory = Path(directory)
    files = sorted((directory/"jobs").glob("*.json"))
    require(files, "no completed output jobs")
    rows, jobs, hashes, arrays_used, paired, queries = [], set(), [], set(), {}, {}
    scalar_count = 0
    for file in files:
        job = read(file)
        seed, model = job["seed"], job["model"]
        require(type(seed) is int and seed >= 0 and model in MODELS and (seed,model) not in jobs,
                "invalid or duplicate output job")
        require(file.name == f"{model}_{seed}.json", "job path differs from job identity")
        replicates = job["support_replicates"]
        require(type(replicates) is int and replicates == (1 if seed == 16991 else 2)
                and (not development or replicates == 2), "unexpected support replicate budget")
        jobs.add((seed,model))
        hashes.append(dict(path=file.relative_to(directory).as_posix(), sha256=sha(file.read_bytes())))
        npz_name = f"{model}_{seed}.npz"
        require(set(job["arrays"]) == {"file", "sha256"} and job["arrays"]["file"] == npz_name,
                "unexpected numerical archive path")
        path = file.parent/npz_name
        require(sha(path.read_bytes()) == job["arrays"]["sha256"], "output NPZ digest differs")
        hashes.append(dict(path=path.relative_to(directory).as_posix(), sha256=job["arrays"]["sha256"]))
        arrays_used.add(npz_name)
        checkpoints = {row["path"]:row for row in job["checkpoints"]}
        expected_checkpoints = {f"{model}_{seed}/{arm}/{phase}/checkpoint.pt" for arm in ARMS for phase in ("novel", "return")}
        require(len(job["checkpoints"]) == 4 and set(checkpoints) == expected_checkpoints,
                "expected two saved model states per arm")
        require(all(row["signature_before"] == row["signature_after"] and row["rng_before"] == row["rng_after"]
                    for row in checkpoints.values()), "reported predictor or RNG state changed during interventions")
        checkpoint_models, frozen, frozen_pairs = {}, {}, 0
        require(len(job["rows"]) == 8+12*replicates and {row_key(row) for row in job["rows"]} == expected_rows(replicates),
                "incomplete or duplicate intervention/context coverage")
        with np.load(path, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "duplicate NPZ array name")
            used = set()
            for index,row in enumerate(job["rows"]):
                require(row["index"] == index and row["seed"] == seed and row["model"] == model,
                        "output row identity/order differs")
                cue = tuple(permutations(range(3)))[seed % 6][0]
                require(row["novel_cue"] == cue and row["cue"] == (cue if row["target"] in NOVEL else None)
                        and type(row["flipped"]) is bool and type(row["replicate"]) is int,
                        "cue or context intervention metadata differs")
                fields = {"observations", "truth", "affected", "valid", "archived_full",
                    "support_observations", "support_actions", "support_outcomes"}
                fields |= {"probabilities_"+cell for cell in CELLS[model]}
                require(set(row["array_sha256"]) == fields and set(row["metrics"]) == set(CELLS[model]),
                        "raw component field coverage differs")
                values = {}
                for field in fields:
                    name = f"c{index}_{field}"
                    require(name in archive, "missing component array")
                    values[field] = archive[name]
                    require(not values[field].dtype.hasobject and array_hash(values[field]) == row["array_sha256"][field],
                            "component array digest differs")
                    used.add(name)
                n, size = len(values["truth"]), len(values["support_actions"])
                require(values["observations"].dtype == np.uint8 and values["observations"].shape == (n, 4, 12, 16)
                        and size > 0 and values["support_observations"].dtype == values["support_actions"].dtype
                        == values["support_outcomes"].dtype == np.uint8
                        and values["support_observations"].shape == (size, 4, 12, 16)
                        and values["support_actions"].shape == (size,) and np.all(values["support_actions"] < 5)
                        and values["support_outcomes"].shape == (size, 3)
                        and np.isin(values["support_outcomes"], (0, 1)).all(), "invalid raw query/support context")
                if development:
                    require(n == 512 and size == 32, "development evaluator/support budget differs")
                checkpoint_phase = "novel" if row["target"] in ("novel_end", "return_entry") else "return"
                checkpoint_path = f"{model}_{seed}/{row['arm']}/{checkpoint_phase}/checkpoint.pt"
                checkpoint, source = checkpoints[checkpoint_path], row["source"]
                trace_phase = "novel" if row["target"] == "novel_end" else "return"
                trace_root = f"{model}_{seed}/{row['arm']}/{trace_phase}/"
                require(source["checkpoint"] == checkpoint_path and source["checkpoint_sha256"] == checkpoint["sha256"]
                        and row["signature"] == checkpoint["signature_before"] and source["phase"] == trace_phase
                        and source["trace_json"] == trace_root+"evaluations.json"
                        and source["trace_arrays"] == trace_root+"evaluations.npz"
                        and type(source["trace"]) is int and source["trace"] >= 0,
                        "row is not bound to its prescribed saved state and trace")
                require(source["query_sha256"] == array_hash(values["observations"])
                        and source["support_fingerprint"] == support_hash(values), "source input provenance differs")
                require(checkpoint_path not in checkpoint_models or checkpoint_models[checkpoint_path] == source["model_sha256"],
                        "one saved predictor has inconsistent model hashes")
                checkpoint_models[checkpoint_path] = source["model_sha256"]
                if row["arm"] == "separate":
                    frozen_key = source["query_sha256"], source["support_fingerprint"]
                    core_hash = array_hash(values["probabilities_core"])
                    if frozen_key in frozen:
                        require(frozen[frozen_key] == core_hash, "frozen core forecast changed under identical inputs")
                        frozen_pairs += 1
                    frozen[frozen_key] = core_hash
                original, full = values["archived_full"], values["probabilities_full"]
                require(original.shape == full.shape and original.dtype == full.dtype
                        and original.tobytes() == full.tobytes(), "full output differs from archived forecast bytes")
                for cell in CELLS[model]:
                    measured = score(values["probabilities_"+cell], values["truth"], values["affected"], values["valid"])
                    require(set(row["metrics"][cell]) == set(METRICS) and all(
                        np.isclose(row["metrics"][cell][metric], measured[metric], rtol=1e-11, atol=1e-12)
                        for metric in METRICS), "recorded component score differs from raw arrays")
                    scalar_count += len(METRICS)
                # Same seed/target/context uses exactly paired physical cases and feedback
                # across arms and architectures. Supports/truth also match cue flips.
                for field in ("observations", "truth", "affected", "valid", "support_observations", "support_actions", "support_outcomes"):
                    key = (seed,row["target"],row["context"],row["replicate"],row["flipped"] if field == "observations" else False,field)
                    value = row["array_sha256"][field]
                    require(key not in paired or paired[key] == value, "paired inputs or flip truth/support differ")
                    paired[key] = value
                    if not field.startswith("support_"):
                        query_key = (seed,row["target"],row["flipped"] if field == "observations" else False,field)
                        require(query_key not in queries or queries[query_key] == value,
                                "context contrast changes its physical query panel")
                        queries[query_key] = value
                rows.append(row)
            require(used == set(archive.files), "unreferenced raw component arrays")
            require(frozen_pairs >= 3*replicates and frozen_pairs == job["frozen_core_forecast_pairs_checked"],
                    "frozen-core repeated-input coverage differs")
    seeds = sorted({seed for seed,_ in jobs})
    require(jobs == {(seed,model) for seed in seeds for model in MODELS}, "both architectures required for every seed")
    require({path.name for path in (directory/"jobs").glob("*.npz")} == arrays_used, "unreferenced numerical job archive")
    if development:
        require(seeds == list(SEEDS), "allocation requires all six original development seeds")
    return rows, dict(jobs=len(jobs), checkpoints=4*len(jobs), intervention_rows=len(rows),
        independently_recomputed_scalars=scalar_count, original_full_byte_equal=len(rows),
        seeds=seeds, input_hashes=hashes, scope="Independent NumPy arithmetic, complete paired coverage, raw array/file hashes and "
        "equality with the copied historical full forecasts. Runner preservation and historical checkpoint binding are separate checks.")


@lru_cache(maxsize=None)
def draws(n):
    return np.random.default_rng(17291).integers(0, n, size=(20000, n))


def paired_stat(values, seeds):
    values = np.asarray(values, dtype=np.float64)
    require(len(seeds) > 0 and len(set(seeds)) == len(seeds) and values.shape == (len(seeds),)
            and np.isfinite(values).all(), "invalid independent seed sample")
    sampled = values[draws(len(seeds))].mean(axis=1)
    lo, hi = np.quantile(sampled, [.025, .975])
    return dict(mean=float(values.mean()), lower=float(lo), upper=float(hi), n=len(seeds), seeds=list(seeds),
                values=values.tolist(), positive=int((values > 0).sum()), negative=int((values < 0).sum()))


def seed_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"],*(row[name] for name in GROUP))].append(row)
    result = []
    for key,selected in sorted(groups.items()):
        seed, model, arm, target, context, flipped = key
        require(len(selected) in (1, 2), "unsupported support replicate count")
        expected = [0] if context == "actual" else list(range(len(selected)))
        require(sorted(row["replicate"] for row in selected) == expected, "support replicates are incomplete/duplicated")
        require(len({row["novel_cue"] for row in selected}) == 1, "within-seed cue changed")
        metrics = {cell:{metric:float(np.mean([row["metrics"][cell][metric] for row in selected]))
            for metric in METRICS} for cell in CELLS[model]}
        result.append(dict(seed=seed, model=model, arm=arm, target=target, context=context, flipped=flipped,
            novel_cue=selected[0]["novel_cue"], support_replicates=len(selected), metrics=metrics))
    return result


def summarize_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in GROUP)].append(row)
    levels, components, cue_slices, decomposition = [], [], [], []
    for key,selected in sorted(groups.items()):
        selected.sort(key=lambda row:row["seed"])
        label, seeds = dict(zip(GROUP,key)), [row["seed"] for row in selected]
        cells = CELLS[label["model"]]
        levels.append(dict(**label, metrics={cell:{metric:paired_stat([row["metrics"][cell][metric] for row in selected], seeds)
            for metric in METRICS} for cell in cells}))
        components.append(dict(**label, contrasts={"full_minus_"+cell:{metric:paired_stat([
            row["metrics"]["full"][metric]-row["metrics"][cell][metric] for row in selected], seeds)
            for metric in METRICS} for cell in cells if cell != "full"}))
        for cue in sorted({row["novel_cue"] for row in selected}):
            subset = [row for row in selected if row["novel_cue"] == cue]
            cue_slices.append(dict(**label, cue=cue, n=len(subset), seeds=[row["seed"] for row in subset],
                metrics={cell:{metric:float(np.mean([row["metrics"][cell][metric] for row in subset]))
                    for metric in METRICS} for cell in cells}))
        if label["model"] == "conditional":
            terms = {"content_at_core_route":{"full_core_route":1, "core":-1},
                "route_on_core_content":{"core_full_route":1, "core":-1},
                "interaction":{"full":1, "full_core_route":-1, "core_full_route":-1, "core":1},
                "total_full_minus_core":{"full":1, "core":-1}}
            values = {term:{metric:paired_stat([sum(weight*row["metrics"][cell][metric] for cell,weight in weights.items())
                for row in selected], seeds) for metric in METRICS} for term,weights in terms.items()}
            for metric in METRICS:
                total = sum(np.asarray(values[term][metric]["values"]) for term in terms if term != "total_full_minus_core")
                require(np.allclose(total, values["total_full_minus_core"][metric]["values"], atol=1e-12, rtol=1e-11),
                        "conditional crossover decomposition does not close")
            decomposition.append(dict(**label, terms=values))
    return levels, components, cue_slices, decomposition


def differences(rows, axis, first, second, name=None):
    positions = {name:index for index,name in enumerate(GROUP)}
    keep = [name for name in GROUP if name != axis]
    keyed = {(row["seed"],*(row[name] for name in GROUP)):row for row in rows}
    groups = defaultdict(list)
    for row in rows:
        if row[axis] == first and not (axis == "context" and row["flipped"]):
            key = [row["seed"],*(row[name] for name in GROUP)]
            key[1+positions[axis]] = second
            require(tuple(key) in keyed, "missing paired arm/context")
            other = keyed[tuple(key)]
            groups[tuple(row[name] for name in keep)].append((row,other))
    result = []
    for key,pairs in sorted(groups.items()):
        pairs.sort(key=lambda pair:pair[0]["seed"])
        label, seeds = dict(zip(keep,key)), [row["seed"] for row,_ in pairs]
        result.append(dict(**label, contrast=name or f"{first}_minus_{second}",
            metrics={cell:{metric:paired_stat([row["metrics"][cell][metric]-other["metrics"][cell][metric] for row,other in pairs], seeds)
                for metric in METRICS} for cell in CELLS[label["model"]]}))
    return result


def allocation(rows, model, eligible):
    selected = [row for row in rows if row["model"] == model and not row["flipped"]]
    seeds = sorted({row["seed"] for row in selected})
    keyed = {(row["seed"],row["arm"],row["target"],row["context"]):row["metrics"] for row in selected}
    def values(arm,target,context,cell,metric="brier"):
        return np.array([keyed[seed,arm,target,context][cell][metric] for seed in seeds])
    old_harm = values("separate","return_entry","refreshed","full")-values("separate","return_entry","refreshed","core")
    novel_benefit = values("separate","novel_end","refreshed","core","focus_brier")-values("separate","novel_end","refreshed","full","focus_brier")
    actual_gap = values("separate","return_entry","actual","full")-values("joint","return_entry","actual","full")
    refresh_gain = values("separate","return_entry","actual","full")-values("separate","return_entry","refreshed","full")
    refreshed_gap = values("separate","return_entry","refreshed","full")-values("joint","return_entry","refreshed","full")
    metrics = {name:paired_stat(value,seeds) for name,value in dict(old_harm=old_harm, novel_benefit=novel_benefit,
        actual_separate_minus_joint=actual_gap, refreshed_support_gain=refresh_gain, refreshed_separate_minus_joint=refreshed_gap).items()}
    split = dict(old_harm_mean=metrics["old_harm"]["mean"] >= .005, old_harm_consistency=int((old_harm > 0).sum()) >= 5,
        novel_benefit_mean=metrics["novel_benefit"]["mean"] >= .002, novel_benefit_consistency=int((novel_benefit > 0).sum()) >= 5)
    context = dict(actual_return_disadvantage=metrics["actual_separate_minus_joint"]["mean"] > 0,
        refresh_gain_mean=metrics["refreshed_support_gain"]["mean"] >= .005,
        refresh_gain_consistency=int((refresh_gain >= .005).sum()) >= 5,
        refreshed_gap_tolerance=metrics["refreshed_separate_minus_joint"]["mean"] <= .005)
    choice = "GO" if all(split.values()) else "CONTEXT_PIVOT" if all(context.values()) else "STOP"
    return dict(model=model, eligible=eligible, allocation=choice if eligible else "INELIGIBLE",
        metrics=metrics, split_gates=split, context_gates=context,
        context_evaluated_for_allocation=eligible and not all(split.values()),
        seeds_with_refresh_gain_at_least_005=int((refresh_gain >= .005).sum()),
        interpretation="Post hoc effort allocation on completed-cohort checkpoints; no promotion, proof of a deployable router, "
            "equivalence claim, or cue-subgroup rescue. A new policy requires separately locked fresh-data testing.")


def analyze(rows, development=False):
    per_seed = seed_rows(rows)
    seeds = sorted({row["seed"] for row in rows})
    require(not development or seeds == list(SEEDS), "development seed coverage differs")
    require(not development or all(row["support_replicates"] == (1 if row["context"] == "actual" else 2)
                                  for row in per_seed), "development refreshed replicate coverage differs")
    levels, components, slices, decomposition = summarize_groups(per_seed)
    return dict(development_allocation_eligible=development, bootstrap=dict(seed=17291, resamples=20000,
        unit="independent seed; refreshed support replicates averaged before contrasts", interval="descriptive percentile 95%"),
        per_seed=per_seed, levels=levels, component_contrasts=components,
        arm_contrasts=differences(per_seed,"arm","separate","joint"),
        context_contrasts=differences(per_seed,"context","actual","refreshed"),
        cue_contrasts=differences(per_seed,"flipped",True,False,"flipped_minus_correct"),
        conditional_decomposition=decomposition, descriptive_cue_slices=slices,
        decisions={model:allocation(per_seed,model,development) for model in MODELS},
        limits="Saved-model intervention, not new training or prospective confirmation. Component-only predictions are "
            "ablations of a combined predictor; the residual is trained as a correction, not a standalone competitor. "
            "Refreshed support uses evaluator-supplied context, not autonomous context detection. Conditional crossed "
            "cells give an exact loss decomposition with interaction, not a unique causal cause of learning failure.")


def markdown(result):
    def estimate(item):
        return f"{100*item['mean']:+.4f} [{100*item['lower']:+.4f}, {100*item['upper']:+.4f}]"
    lines = ["# Saved core/residual output diagnostic", "", result["limits"], "",
        "All Brier values below are multiplied by 100. Intervals are descriptive paired-seed bootstrap intervals; "
        "two refreshed supports are averaged within each seed. Full raw levels for every component and mask are in summary.json.", "",
        "| Architecture | Effort allocation | Old harm: full − core | Novel benefit: core − full |", "|---|---|---:|---:|"]
    for model,decision in result["decisions"].items():
        lines.append(f"| {model} | {decision['allocation']} | {estimate(decision['metrics']['old_harm'])} | {estimate(decision['metrics']['novel_benefit'])} |")
    lines += ["", "Old harm uses return-entry all-case Brier; novel benefit uses novel-end affected-subset Brier. "
        "Both use separate-arm weights and refreshed support. GO requires respective means ≥0.5000 and ≥0.2000 table units, "
        "each positive in at least five of six seeds. Smoke is ineligible.", "",
        "| Architecture | Actual separate − joint | Separate actual − refreshed | Refreshed separate − joint | Refresh gain ≥0.5000 |", "|---|---:|---:|---:|---:|"]
    for model,decision in result["decisions"].items():
        metrics = decision["metrics"]
        lines.append(f"| {model} | {estimate(metrics['actual_separate_minus_joint'])} | {estimate(metrics['refreshed_support_gain'])} | "
            f"{estimate(metrics['refreshed_separate_minus_joint'])} | {decision['seeds_with_refresh_gain_at_least_005']}/{metrics['old_harm']['n']} |")
    lines += ["", "Only when GO fails, CONTEXT_PIVOT requires positive mean actual disadvantage, mean refresh gain ≥0.5000 "
        "and that gain in ≥5/6 seeds, and refreshed gap ≤0.5000. Otherwise STOP. No alternate endpoints or subgroup decisions.", "",
        "| Conditional separate, all-case Brier | Context | Content at core route | Route on core content | Interaction | Total full − core |", "|---|---|---:|---:|---:|---:|"]
    for row in result["conditional_decomposition"]:
        if row["arm"] == "separate" and not row["flipped"]:
            terms = row["terms"]
            numbers = [f"{100*terms[name]['brier']['mean']:+.4f}" for name in ("content_at_core_route","route_on_core_content","interaction","total_full_minus_core")]
            lines.append(f"| {row['target']} | {row['context']} | "+" | ".join(numbers)+" |")
    lines += ["", "For each seed: content = loss(full content, core route) − loss(core); route = loss(core content, full route) − loss(core); "
        "interaction = loss(full) − loss(full content, core route) − loss(core content, full route) + loss(core). "
        "The three terms sum to full − core. Negative error differences improve prediction; survival signs have the opposite preference.", ""]
    return "\n".join(lines)


def summarize(directory, output, development=False):
    rows, integrity = read_jobs(directory, development)
    result = analyze(rows, development)
    result["record_integrity"] = integrity
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    value = summarize(args.input, args.output, args.development)
    print(json.dumps(dict(record_integrity=value["record_integrity"],
        decisions={model:decision["allocation"] for model,decision in value["decisions"].items()}), indent=2))
