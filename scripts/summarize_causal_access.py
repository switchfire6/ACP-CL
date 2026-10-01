"""Independent NumPy audit and seed-balanced scoring of fixed causal access.

No learner or training-code imports. The online reconstruction uses original
performed-action forecasts, and releases a packet's feedback only afterwards.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from itertools import permutations
import json
from pathlib import Path

import numpy as np


MODELS = ("conditional", "recurrent")
POLICIES = ("core", "full", "half", "causal", "joint")
SEEDS = tuple(range(17001, 17007))
METRICS = ("brier", "focus_brier", "survival", "focus_survival")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def array_hash(value):
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=invalid)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def score(prediction, truth, affected, mask=None):
    prediction, truth, affected = map(np.asarray, (prediction, truth, affected))
    count, eps = len(truth), 2 * np.finfo(np.float32).eps
    require(count > 0 and prediction.shape == truth.shape == (count, 5, 3),
            "invalid prediction/truth dimensions")
    require(prediction.dtype.kind == "f" and np.isfinite(prediction).all()
            and np.all(prediction >= -eps) and np.all(prediction <= 1 + eps)
            and np.all(np.diff(prediction, axis=-1) <= eps),
            "invalid survival predictions")
    require(np.isin(truth, (0, 1)).all()
            and np.all(np.diff(truth.astype(np.int16), axis=-1) <= 0),
            "invalid survival truth")
    require(affected.dtype == np.bool_ and affected.shape == (count,),
            "invalid affected mask")
    mask = np.ones(count, dtype=bool) if mask is None else np.asarray(mask)
    require(mask.dtype == np.bool_ and mask.shape == (count,) and mask.any(),
            "empty or invalid scoring slice")
    errors = np.square(prediction.astype(np.float64) - truth).mean(axis=(1, 2))
    survived = truth[np.arange(count), prediction[:, :, -1].argmax(axis=1), -1]
    focus = mask & affected
    return dict(brier=float(errors[mask].mean()), survival=float(survived[mask].mean()),
                focus_brier=float(errors[focus].mean()) if focus.any() else None,
                focus_survival=float(survived[focus].mean()) if focus.any() else None)


def reconstruct_weights(core, full, actions, outcomes, batch_size, window=32):
    """Return one pre-feedback weight per packet and each original log-score delta."""
    core, full = np.asarray(core), np.asarray(full)
    actions, outcomes = np.asarray(actions), np.asarray(outcomes)
    count = len(actions)
    require(count > 0 and batch_size > 0 and count % batch_size == 0 and window > 0,
            "invalid causal packet/window dimensions")
    require(core.shape == full.shape == (count, 5, 3) and actions.shape == (count,)
            and outcomes.shape == (count, 3) and actions.dtype.kind in "iu"
            and np.all((actions >= 0) & (actions < 5)) and np.isin(outcomes, (0, 1)).all(),
            "invalid causal feedback arrays")
    terminal = outcomes[:, -1].astype(np.float64)
    log_scores = []
    for prediction in (core, full):
        probability = prediction[np.arange(count), actions, -1].astype(np.float64)
        log_scores.append(terminal * np.log(np.clip(probability, 1e-6, 1 - 1e-6))
                          + (1 - terminal) * np.log(np.clip(1 - probability, 1e-6, 1 - 1e-6)))
    delta = log_scores[1] - log_scores[0]
    sums = np.array([delta[max(0, end - window):end].sum(dtype=np.float64)
                     for end in range(0, count, batch_size)])
    weights = np.exp(-np.logaddexp(0, -sums))
    require(weights[0] == .5 and np.isfinite(weights).all(), "invalid causal weights")
    return weights, delta


@lru_cache(maxsize=None)
def draws(count):
    return np.random.default_rng(17291).integers(0, count, size=(20000, count))


def estimate(values, seeds):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (len(seeds),) and len(seeds) > 0
            and len(set(seeds)) == len(seeds) and np.isfinite(values).all(),
            "invalid independent-seed contrast")
    lower, upper = np.quantile(values[draws(len(seeds))].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), lower=float(lower), upper=float(upper),
                n=len(seeds), seeds=list(seeds), values=values.tolist(),
                improves=int((values < 0).sum()))


def analyze(rows, development=False):
    """Reduce already audited raw arrays; never discard qualification failures."""
    models = {}
    for model in MODELS:
        selected = sorted((row for row in rows if row["model"] == model),
                          key=lambda row: row["seed"])
        seeds = [row["seed"] for row in selected]
        require(seeds and (not development or seeds == list(SEEDS)),
                "incomplete architecture/seed coverage")
        groups = {"overall": selected}
        groups.update({f"cue_{cue}": [row for row in selected if row["cue"] == cue]
                       for cue in sorted({row["cue"] for row in selected})})
        require(not development or len(groups) == 4, "incomplete cue coverage")
        qualification, cue_guard = {}, {}
        upper = {"specialist_old_brier": .12, "specialist_novel_brier": .12,
                 "core_old_excess": .02, "full_novel_focus_excess": .02}
        lower = {"old_marginal_gain": .02, "novel_marginal_gain": .02,
                 "specialist_novel_cue_gain": .002, "full_novel_cue_gain": .002}
        for name, group in groups.items():
            means = {key: float(np.mean([row["qualification"][key] for row in group]))
                     for key in (*upper, *lower)}
            gates = {key: means[key] <= margin for key, margin in upper.items()}
            gates.update({key: means[key] >= margin for key, margin in lower.items()})
            qualification[name] = dict(seeds=[row["seed"] for row in group],
                                       metrics=means, gates=gates, passed=all(gates.values()))
            value = float(np.mean([row["cue_gain"]["causal"] for row in group]))
            cue_guard[name] = dict(mean=value, passed=value >= .002)
        levels = {part: {policy: {metric: estimate(
                    [row["stream"][part][policy][metric] for row in selected], seeds)
                    for metric in METRICS if all(row["stream"][part][policy][metric]
                                                 is not None for row in selected)}
                    for policy in POLICIES} for part in ("whole", "old", "novel")}

        def contrast(part, metric, baseline):
            return estimate([row["stream"][part]["causal"][metric]
                             - row["stream"][part][baseline][metric] for row in selected], seeds)

        comparisons = {f"{part}_{metric}_minus_{baseline}": contrast(part, metric, baseline)
                       for part in ("whole", "old", "novel")
                       for metric in ("brier", "survival") for baseline in ("full", "half", "joint")}
        comparisons["novel_focus_brier_minus_full"] = contrast("novel", "focus_brier", "full")
        gates = {"whole_vs_full": comparisons["whole_brier_minus_full"]["mean"] <= -.002,
                 "whole_vs_full_consistency": comparisons["whole_brier_minus_full"]["improves"] >= 5,
                 "whole_vs_half": comparisons["whole_brier_minus_half"]["mean"] <= -.001,
                 "whole_vs_half_consistency": comparisons["whole_brier_minus_half"]["improves"] >= 5,
                 "whole_vs_joint": comparisons["whole_brier_minus_joint"]["mean"] <= .005,
                 "old_vs_full": comparisons["old_brier_minus_full"]["mean"] <= .005,
                 "novel_focus_vs_full": comparisons["novel_focus_brier_minus_full"]["mean"] <= .005}
        gates.update({f"{part}_survival_vs_{baseline}":
                      comparisons[f"{part}_survival_minus_{baseline}"]["mean"] >= -.01
                      for part in ("old", "novel") for baseline in ("full", "joint")})
        gates["novel_cue_all_groups"] = all(group["passed"] for group in cue_guard.values())
        qualified, passed = all(group["passed"] for group in qualification.values()), all(gates.values())
        cue_slices = {name: dict(seeds=[row["seed"] for row in group],
            contrasts={f"{part}_causal_minus_{baseline}": float(np.mean([
                row["stream"][part]["causal"]["brier"] - row["stream"][part][baseline]["brier"]
                for row in group])) for part in ("whole", "old", "novel")
                for baseline in ("full", "half", "joint")})
            for name, group in groups.items() if name != "overall"}
        models[model] = dict(qualification=qualification, qualified=qualified,
                            levels=levels, contrasts=comparisons, cue_guard=cue_guard,
                            descriptive_cue_slices=cue_slices,
                            raw_primary_gates=gates, raw_primary_passed=passed,
                            decision=("INELIGIBLE" if not development or not qualified
                                      else "PASS" if passed else "FAIL"))
    return dict(development=development, per_seed=rows, models=models,
                bootstrap=dict(seed=17291, resamples=20000, unit="independent seed",
                               interval="descriptive percentile 95%"),
                interpretation="Frozen prediction access only. Qualification and every primary gate "
                "must pass per architecture; intervals do not establish statistical noninferiority. "
                "No cue exclusion, optimizer update, recognition-delay exclusion, or test-score tuning.")


def fingerprint(*values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def training_marginal(counts, success, total):
    counts, success = np.asarray(counts), np.asarray(success)
    require(counts.shape == (5,) and success.shape == (5, 3)
            and counts.dtype.kind in "iu" and success.dtype.kind in "iuf"
            and np.isfinite(success).all() and np.all(success == np.floor(success))
            and np.all(counts >= 0) and np.all((success >= 0) & (success <= counts[:, None]))
            and counts.sum() == total, "invalid specialist training marginal")
    return (success.astype(np.float64) + .5) / (counts[:, None] + 1.)


def read_jobs(directory, development=False):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    config, rows, inventory, paired_inputs = manifest["config"], [], [], {}
    identity = {name + "_sha256": json_hash(manifest[field]) for name, field in
                (("config", "config"), ("source", "source_files"),
                 ("analysis", "analysis_files"), ("runtime", "runtime"))}
    require(identity == manifest["identity"], "manifest identity does not match its contents")
    require(set(manifest["archives"]) == {"training_source.zip", "analysis_at_lock.zip"},
            "locked source/analysis archive coverage differs")
    for name, expected_sha in {**manifest["archives"],
                               "protocol_at_lock.md": manifest["protocol_sha256"]}.items():
        require(sha(directory / name) == expected_sha, "locked archive/protocol changed: " + name)
        inventory.append(dict(path=name, sha256=expected_sha))
    require(manifest["analysis_files"]["scripts/summarize_causal_access.py"] == sha(__file__),
            "independent scorer differs from its pre-run lock")
    jobs = sorted((directory / "jobs").glob("*/result.json"))
    expected = {(model, seed) for model in config["models"] for seed in config["seeds"]}
    require(expected and len(jobs) == len(expected), "incomplete completed job coverage")
    require(not development or expected == {(model, seed) for model in MODELS for seed in SEEDS},
            "development cohort differs from locked cohort")
    completion = read(directory / "completion.json")
    finished = {(item["model"], item["seed"]): item for item in completion["job_records"]}
    require(completion["identity"] == identity and len(completion["job_records"]) == len(expected)
            and set(finished) == expected, "completion identity or cohort coverage differs")
    inventory.append(dict(path="completion.json", sha256=sha(directory / "completion.json")))
    observed = set()
    for path in jobs:
        job = read(path)
        model, seed, cue = job["model"], job["seed"], job["cue"]
        require((model, seed) in expected - observed and path.parent.name == f"{model}_{seed}",
                "unexpected or duplicate job identity")
        observed.add((model, seed))
        require(job["identity"] == manifest["identity"]
                and cue == tuple(permutations(range(3)))[seed % 6][0], "identity/cue mismatch")
        require(sha(path) == finished[model, seed]["result_sha256"], "completion result hash differs")
        require(set(job["frozen_before"]) == {"separate", "joint", "reference_old", "reference_novel"}
                and job["frozen_before"] == job["frozen_after"]
                and job["rng_before"] == job["rng_after"], "frozen predictor or global RNG state changed")
        files = {**job["checkpoint_hashes"], **job["phase_hashes"],
                 job["artifact"]["file"]: job["artifact"]["sha256"]}
        require(job["artifact"]["file"] == "forecasts.npz", "unexpected forecast archive path")
        for relative, digest in files.items():
            target = (path.parent / relative).resolve()
            require(target.is_relative_to(path.parent) and target.is_file()
                    and sha(target) == digest, "missing or changed input artifact: " + relative)
            inventory.append(dict(path=target.relative_to(directory).as_posix(), sha256=digest))
        for relative in job["phase_hashes"]:
            phase_path = path.parent / relative
            phase = read(phase_path)
            require(phase["identity"] == identity and phase["seed"] == seed and phase["model"] == model,
                    "fit phase identity differs")
            for name, expected_sha in phase["artifact_hashes"].items():
                raw = (phase_path.parent / name).resolve()
                require(raw.is_relative_to(phase_path.parent) and raw.is_file()
                        and sha(raw) == expected_sha, "fit phase raw artifact changed: " + name)
                inventory.append(dict(path=raw.relative_to(directory).as_posix(), sha256=expected_sha))
        inventory.append(dict(path=path.relative_to(directory).as_posix(), sha256=sha(path)))
        with np.load(path.parent / "forecasts.npz", allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files))
                    and set(archive.files) == set(job["array_sha256"]), "array coverage differs")
            values = {key: archive[key] for key in archive.files}
        for key, value in values.items():
            require(not value.dtype.hasobject and array_hash(value) == job["array_sha256"][key],
                    "raw array digest differs: " + key)
            if ("observations" in key or "support_" in key
                    or key in ("stream_truth", "stream_actions", "stream_outcomes", "stream_affected",
                               "stream_is_novel", "stream_block") or key.endswith(("_truth", "_affected"))):
                pair_key = (seed, key)
                digest = array_hash(value)
                require(paired_inputs.setdefault(pair_key, digest) == digest,
                        "architectures received different physical inputs or histories")
        batch_size, n = config["batch_size"], len(values["stream_actions"])
        blocks = config["stream_blocks"]
        block_size = config["stream_block_size"]
        require(n == blocks * block_size and n % batch_size == 0
                and job["stream"]["packet_count"] == n // batch_size, "stream truncated or extended")
        block = np.repeat(np.arange(blocks), block_size)
        novel = block % 2 == 1
        require(np.array_equal(values["stream_block"], block)
                and values["stream_is_novel"].dtype == np.bool_
                and np.array_equal(values["stream_is_novel"], novel), "switch schedule differs")
        support = tuple(values["initial_support_" + name]
                        for name in ("observations", "actions", "outcomes"))
        require([value.shape for value in support] == [(batch_size, 4, 12, 16),
                    (batch_size,), (batch_size, 3)] and all(value.dtype == np.uint8 for value in support),
                "invalid initial causal support")
        require(values["stream_observations"].shape == (n, 4, 12, 16)
                and values["stream_flipped_observations"].shape == (n, 4, 12, 16)
                and np.array_equal(values["stream_observations"][~novel],
                                   values["stream_flipped_observations"][~novel]),
                "invalid stream query shape or old cue flip")
        require(fingerprint(*support) == job["stream"]["initial_support"], "initial support differs")
        history = [fingerprint(*support)]
        for start in range(0, n - batch_size, batch_size):
            history.append(fingerprint(*(values["stream_" + name][start:start + batch_size]
                                         for name in ("observations", "actions", "outcomes"))))
        require(history == job["stream"]["support_fingerprints"], "support is not prior packet")
        truth, affected = values["stream_truth"], values["stream_affected"]
        require(np.array_equal(values["stream_outcomes"],
                               truth[np.arange(n), values["stream_actions"]]),
                "performed feedback differs from physical truth")
        predictions = {policy: values["stream_" + policy] for policy in POLICIES}
        weights, delta = reconstruct_weights(predictions["core"], predictions["full"],
                    values["stream_actions"], values["stream_outcomes"], batch_size)
        require(values["stream_weights"].shape == weights.shape
                and np.allclose(values["stream_weights"], weights, atol=2e-12, rtol=0),
                "causal weights differ from original past forecast scores")
        final = job["stream"]["mixer_final"]
        require(final["capacity"] == 32 and final["clip"] == 1e-6
                and final["samples_seen"] == n and final["packets_seen"] == n // batch_size
                and np.allclose(final["scores"], delta[-32:], atol=2e-12, rtol=0),
                "final score ledger differs from completed feedback")
        expanded = np.repeat(weights, batch_size)[:, None, None]
        for suffix in ("", "_flipped"):
            core, full = (values["stream_" + name + suffix] for name in ("core", "full"))
            for policy, weight in (("half", .5), ("causal", expanded)):
                mixed = (1 - weight) * core.astype(np.float64) + weight * full.astype(np.float64)
                require(np.allclose(values["stream_" + policy + suffix], mixed, atol=1e-7, rtol=0),
                        "mixture uses wrong probabilities or feedback weights")
        stream = {part: {policy: score(prediction, truth, affected, mask)
                         for policy, prediction in predictions.items()}
                  for part, mask in (("whole", np.ones(n, bool)), ("old", ~novel), ("novel", novel))}
        cue_gain = {}
        for policy in POLICIES:
            flipped = values["stream_" + policy + "_flipped"]
            require(np.array_equal(flipped[~novel], predictions[policy][~novel]),
                    "old-condition cue diagnostic unexpectedly changed inputs")
            cue_gain[policy] = score(flipped, truth, affected, novel)["focus_brier"] - stream["novel"][policy]["focus_brier"]
        qualification, qualification_levels = {}, {}
        for law in ("old", "novel"):
            prefix = "q_" + law + "_"
            target, focus = values[prefix + "truth"], values[prefix + "affected"]
            require(len(target) == config["eval_size"], "qualification query budget differs")
            metric = {}
            for policy in ("core", "full", "joint", "reference"):
                array = values[prefix + policy]
                require(len(array) == config["support_replicates"], "qualification support budget differs")
                metric[policy] = [score(item, target, focus) for item in array]
            qualification_levels[law] = {policy: {
                name: float(np.mean([item[name] for item in results]))
                for name in METRICS if all(item[name] is not None for item in results)}
                for policy, results in metric.items()}
            def mean(policy, field):
                return float(np.mean([item[field] for item in metric[policy]]))
            marginal = job["reference_marginals"][law]
            marginal_prediction = training_marginal(marginal["counts"], marginal["success"],
                                                    config["reference_size"])
            marginal_brier = float(np.square(marginal_prediction - target).mean())
            qualification["specialist_" + law + "_brier"] = mean("reference", "brier")
            qualification[law + "_marginal_gain"] = marginal_brier - mean("reference", "brier")
            if law == "old":
                qualification["core_old_excess"] = mean("core", "brier") - mean("reference", "brier")
            else:
                qualification["full_novel_focus_excess"] = mean("full", "focus_brier") - mean("reference", "focus_brier")
                for policy, label in (("reference", "specialist"), ("full", "full")):
                    flipped = [score(item, target, focus)["focus_brier"] for item in values[prefix + policy + "_flipped"]]
                    qualification[label + "_novel_cue_gain"] = float(np.mean(flipped)) - mean(policy, "focus_brier")
        rows.append(dict(model=model, seed=seed, cue=cue, stream=stream, cue_gain=cue_gain,
                         qualification=qualification, qualification_levels=qualification_levels,
                         packet_weights=weights.tolist(),
                         last_score_deltas=delta[-32:].tolist()))
    require(observed == expected, "missing cohort jobs")
    return rows, dict(manifest_sha256=sha(directory / "manifest.json"),
                      scorer_sha256=sha(__file__), files=inventory, jobs=len(rows),
                      array_provenance_checked=True, causal_weights_independently_reconstructed=True)


def self_test():
    counts, success = np.full(5, 2), np.tile([1., 1., 0.], (5, 1))
    require(np.allclose(training_marginal(counts, success, 10), [[.5, .5, 1 / 6]] * 5),
            "integral floating success totals were rejected or mis-scored")
    for invalid in (.5, np.nan, np.inf):
        altered = success.copy()
        altered[0, 0] = invalid
        try:
            training_marginal(counts, altered, 10)
        except ValueError:
            pass
        else:
            raise AssertionError("fractional/nonfinite marginal totals were accepted")
    core = np.full((8, 5, 3), .5)
    full = np.full((8, 5, 3), .75)
    actions = np.zeros(8, dtype=np.uint8)
    outcomes = np.repeat(np.array([1, 1, 0, 0, 1, 1, 0, 0])[:, None], 3, axis=1)
    weights, delta = reconstruct_weights(core, full, actions, outcomes, 2, 3)
    require(np.allclose(weights[:3], [.5, 9 / 13, 3 / 11], rtol=0, atol=1e-15),
            "causal rolling-window arithmetic failed")
    changed = outcomes.copy()
    changed[4:] = 1 - changed[4:]
    alternate, _ = reconstruct_weights(core, full, actions, changed, 2, 3)
    require(np.array_equal(weights[:3], alternate[:3]), "future feedback leaked into weights")
    require(np.allclose(delta[:4], [np.log(1.5)] * 2 + [np.log(.5)] * 2),
            "terminal performed-action log scores differ")
    truth = np.repeat(outcomes[:, None, :], 5, axis=1).astype(np.uint8)
    affected = np.array([True, False] * 4)
    metrics = score(core, truth, affected)
    require(metrics == dict(brier=.25, focus_brier=.25, survival=.5, focus_survival=.5),
            "independent Brier/argmax scoring failed")
    constant = estimate([.2] * 6, list(SEEDS))
    require(abs(constant["lower"] - .2) < 1e-15
            and abs(constant["upper"] - .2) < 1e-15, "bootstrap arithmetic failed")
    rows = []
    for model in MODELS:
        for index, seed in enumerate(SEEDS):
            stream = {part: {policy: dict(brier=.09 if policy in ("causal", "joint") else .1,
                         focus_brier=.09 if policy in ("causal", "joint") else .1,
                         survival=.8, focus_survival=.8) for policy in POLICIES}
                      for part in ("whole", "old", "novel")}
            rows.append(dict(model=model, seed=seed, cue=index % 3, stream=stream,
                cue_gain=dict(causal=.01), qualification=dict(specialist_old_brier=.08,
                specialist_novel_brier=.08, core_old_excess=.01, full_novel_focus_excess=.01,
                old_marginal_gain=.04, novel_marginal_gain=.04,
                specialist_novel_cue_gain=.01, full_novel_cue_gain=.01)))
    require(all(item["decision"] == "PASS" for item in analyze(rows, True)["models"].values()),
            "synthetic primary/qualification gate pass failed")
    rows[0]["qualification"]["specialist_old_brier"] = 1.
    decision = analyze(rows, True)["models"]["conditional"]
    require(decision["decision"] == "INELIGIBLE" and decision["raw_primary_passed"],
            "qualification failure was hidden by favorable primary outcomes")
    return dict(passed=True, checks=8)


def markdown(result):
    lines = ["# Causal access to fixed predictions", "", result["interpretation"], "",
             "Brier differences below use raw units; negative improves. Intervals are descriptive "
             "paired-seed bootstrap 95% intervals. All transitions and recognition delays are scored.", "",
             "| Architecture | Qualified | Raw primary pass | Decision |",
             "|---|---|---|---|"]
    for model, item in result["models"].items():
        lines.append(f"| {model} | {item['qualified']} | {item['raw_primary_passed']} | {item['decision']} |")
    lines += ["", "| Architecture | Causal minus comparator | Mean [95% interval] | Lower error seeds |",
              "|---|---|---:|---:|"]
    for model, item in result["models"].items():
        for name in ("whole_brier_minus_full", "whole_brier_minus_half", "whole_brier_minus_joint",
                     "old_brier_minus_full", "novel_focus_brier_minus_full"):
            value = item["contrasts"][name]
            lines.append(f"| {model} | {name} | {value['mean']:+.6f} "
                         f"[{value['lower']:+.6f}, {value['upper']:+.6f}] | {value['improves']}/{value['n']} |")
    lines += ["", "Qualification failures and raw gate failures:", ""]
    for model, item in result["models"].items():
        failed = [f"{group}/{name}" for group, detail in item["qualification"].items()
                  for name, passed in detail["gates"].items() if not passed]
        failed += [name for name, passed in item["raw_primary_gates"].items() if not passed]
        lines.append(f"- {model}: " + (", ".join(failed) or "none") + ".")
    return "\n".join(lines) + "\n"


def summarize(directory, output, development=False):
    rows, integrity = read_jobs(directory, development)
    result = analyze(rows, development)
    result["integrity"] = integrity
    write(output, result)
    Path(output).with_suffix(".md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        if args.input is None or args.output is None:
            parser.error("--input and --output are required unless --self-test is specified")
        result = summarize(args.input, args.output, args.development)
        print(json.dumps({model: item["decision"] for model, item in result["models"].items()}, indent=2))
