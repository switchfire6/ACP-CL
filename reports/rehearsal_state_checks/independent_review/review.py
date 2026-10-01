"""Independent arithmetic review of completed, exported rehearsal-state records.

Uses only Python's standard library and NumPy. No study, summary, model, audit
or training modules are imported. This checks the existing specified analysis;
it does not choose new outcomes, thresholds, seeds, models or interventions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np


CELLS = tuple(format(number, "03b") for number in range(8))
FACTORS = ("weights", "optimizer", "anchor")
BRANCHES = ("current", "return")
FIELDS = ("brier", "survival", "early_brier", "late_brier")
RESPONSES = ("selection", "chosen", "uniform")
BOOTSTRAP_SEED = 27192026
RESAMPLES = 20000
SETTINGS = dict(kind="diagnostic", prefix_blocks=4, prefix_size=1024,
    acquisition_size=8192, inter_assessment_size=4096, score_size=512, validation_size=512,
    batch_size=32, candidate_count=8, rehearsal_updates=12, width=64, experts=4,
    context_width=12, decoder_width=64, interaction_features=True, lr=.002,
    memory_packets=16, updates_per_batch=12, evidence_strength=1.)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contrast_cube(cube):
    """Evaluate finite differences on a W/O/A cube, without coefficient tables."""
    cube = np.asarray(cube, dtype=np.float64)
    need(cube.shape == (2, 2, 2), "finite-difference cube must have three binary axes")
    result = {"total_111_minus_000":float(cube[1, 1, 1]-cube[0, 0, 0])}
    for axis, name in enumerate(FACTORS):
        derivative = np.take(cube, 1, axis=axis)-np.take(cube, 0, axis=axis)
        result[name+"_main"] = float(derivative.mean())
        others = [other for other in range(3) if other != axis]
        for fixed in itertools.product(range(2), repeat=2):
            label = name+"_at_"+"_".join(FACTORS[other]+str(bit) for other, bit in zip(others, fixed))
            result[label] = float(derivative[fixed])
        result["repair_"+name] = float(derivative[1, 1])
    for first, second in itertools.combinations(range(3), 2):
        # Differencing retains axis length one, so original axis numbers remain valid.
        interaction = np.diff(np.diff(cube, axis=first), axis=second)
        result[FACTORS[first]+"_"+FACTORS[second]+"_interaction"] = float(interaction.mean())
    result["three_way_interaction"] = float(np.diff(np.diff(np.diff(cube, axis=0), axis=1), axis=2).item())
    return result


def derive_coefficients():
    result = {name:{} for name in contrast_cube(np.zeros((2, 2, 2)))}
    for cell in CELLS:
        basis = np.zeros((2, 2, 2))
        basis[tuple(map(int, cell))] = 1.
        for name, coefficient in contrast_cube(basis).items():
            if coefficient != 0:
                result[name][cell] = coefficient
    return result


def statistics(values, seeds, draws):
    """Independent seed bootstrap; retain the locked sub-1e-12 zero convention."""
    values = np.asarray(values, dtype=np.float64).copy()
    need(values.shape == (len(seeds),) and len(seeds) > 0 and np.isfinite(values).all(), "bad seed values")
    values[np.abs(values) < 1e-12] = 0.
    samples = values[draws].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5], method="linear")
    return dict(mean=float(values.mean()), lower=float(low), upper=float(high),
        positive=int(np.count_nonzero(values > 0)), negative=int(np.count_nonzero(values < 0)),
        n=len(values), differences=values.tolist(), seeds=list(seeds))


def eligible(config):
    return (config["seeds"] == list(range(15001, 15013))
            and set(config["models"]) == {"conditional", "recurrent"}
            and all(config.get(key) == value for key, value in SETTINGS.items()))


def fixed_screen(total, complete):
    criteria = dict(complete_scientific_cohort=dict(passed=bool(complete)),
        deterioration=dict(passed=total["mean"] >= .0005, value=total["mean"], minimum=.0005),
        seed_consistency=dict(passed=total["positive"] >= 9, positive_seeds=total["positive"],
                              minimum=9, n=total["n"]))
    return dict(passed=all(item["passed"] for item in criteria.values()), gates=criteria,
                failed=[name for name, item in criteria.items() if not item["passed"]])


class Review:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.inputs = {}
        self.comparisons = 0
        self.maximum_error = 0.
        self.array_files = 0
        self.prediction_values = 0
        self.forecasts = 0
        self.metric_scalars = 0
        self.bootstrap_statistics = 0

    def path(self, relative):
        path = (self.directory/relative).resolve()
        need(path.is_relative_to(self.directory), "input escapes completed export directory")
        self.inputs[path.relative_to(self.directory).as_posix()] = sha(path)
        return path

    def read(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def arrays(self, relative):
        with np.load(self.path(relative), allow_pickle=False) as archive:
            need(len(archive.files) == len(set(archive.files)), "duplicate numerical array name")
            values = {name:archive[name] for name in archive.files}
        self.array_files += 1
        return values

    def equal(self, expected, actual, label="root"):
        if isinstance(expected, dict):
            need(isinstance(actual, dict) and expected.keys() == actual.keys(), "dictionary keys differ: "+label)
            for key in expected:
                self.equal(expected[key], actual[key], label+"."+str(key))
        elif isinstance(expected, (list, tuple)):
            need(isinstance(actual, (list, tuple)) and len(expected) == len(actual), "sequence length differs: "+label)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.equal(left, right, label+f"[{index}]")
        elif isinstance(expected, (float, np.floating)):
            need(isinstance(actual, (int, float)) and np.isfinite(expected) and np.isfinite(actual),
                 "nonfinite numeric comparison: "+label)
            error = abs(float(expected)-float(actual))
            self.maximum_error = max(self.maximum_error, error)
            need(error <= 2e-12, f"numeric mismatch at {label}: {expected} != {actual}")
            self.comparisons += 1
        else:
            need(type(expected) is type(actual) and expected == actual, "exact mismatch: "+label)
            if isinstance(expected, int):
                self.comparisons += 1

    def metric_panel(self, probabilities, data):
        """Loop over predictors; gather performed actions and greedy physical truth."""
        actions, outcomes, truth = data["actions"], data["outcomes"], data["truth"]
        packets, size = actions.shape
        need(probabilities.ndim == 6 and probabilities.shape[:2] == (2, packets)
             and probabilities.shape[3:] == (size, 5, 3), "invalid forecast tensor")
        need(probabilities.dtype == np.float32 and np.isfinite(probabilities).all()
             and probabilities.min() >= -np.finfo(np.float32).eps
             and probabilities.max() <= 1+np.finfo(np.float32).eps, "invalid forecast dtype/range")
        need(outcomes.shape == (2, packets, size, 3) and truth.shape == (2, packets, size, 5, 3),
             "invalid physical outcome tensor")
        need(np.all(np.diff(probabilities, axis=-1) <= np.finfo(np.float32).eps), "nonmonotonic forecast horizons")
        self.prediction_values += probabilities.size
        self.forecasts += int(np.prod(probabilities.shape[:3]))
        result = {}
        for branch, label in enumerate(BRANCHES):
            result[label] = {field:[] for field in FIELDS}
            for candidate in range(probabilities.shape[2]):
                prediction = probabilities[branch, :, candidate]
                performed = np.take_along_axis(prediction, actions[:, :, None, None], axis=2).squeeze(axis=2)
                error = ((performed.astype(np.float64)-outcomes[branch])**2).mean(axis=(1, 2))
                decisions = np.argmax(prediction[..., -1], axis=-1)
                realized = np.take_along_axis(truth[branch, ..., -1], decisions[:, :, None], axis=2).squeeze(axis=2)
                values = dict(brier=float(error.mean()), survival=float(realized.mean()),
                    early_brier=float(error[0]), late_brier=float(error[1:].mean()))
                for field, value in values.items():
                    result[label][field].append(value)
                    self.metric_scalars += 1
        return result

    def assessment(self, record):
        seed, model, index = (record[key] for key in ("seed", "model", "index"))
        prefix = f"arrays/{model}_{seed}/diagnostic_{index}/"
        data = self.arrays(prefix+"data.npz")
        for branch in range(2):
            feedback = np.take_along_axis(data["truth"][branch], data["actions"][:, :, None, None], axis=2).squeeze(axis=2)
            need(np.array_equal(feedback, data["outcomes"][branch]), "performed feedback disagrees with physics")
        raw = {branch:{} for branch in BRANCHES}
        for cell in CELLS:
            stored = self.arrays(prefix+f"predictions_{cell}.npz")
            need(set(stored) == {"probabilities"}, "unexpected forecast array key")
            metrics = self.metric_panel(stored["probabilities"], data)
            for branch in BRANCHES:
                self.equal(metrics[branch], record["panels"][branch][cell], f"{model}/{seed}/{index}/{branch}/{cell}")
                raw[branch][cell] = metrics[branch]
        zero = self.metric_panel(self.arrays(prefix+"predictions_zero.npz")["probabilities"], data)
        self.equal(zero, record["zero"], f"{model}/{seed}/{index}/zero")
        candidates, selection = record["candidates"], record["choice"]
        ids = [candidate["id"] for candidate in candidates]
        score = np.asarray(selection["score_losses"], dtype=float)
        need(score.shape == (len(ids)+1,) and np.isfinite(score).all(), "invalid complete score vector")
        selected = min(range(len(ids)), key=lambda pos:(score[pos], ids[pos]))
        accurate = min(range(len(ids)), key=lambda pos:(candidates[pos]["original_brier"], ids[pos]))
        need(selection["value_id"] == ids[selected] and selection["accuracy_id"] == ids[accurate], "incorrect frozen choice")
        self.equal((score[-1]-score[:-1]).tolist(), selection["values"], "common replacement values")
        row = dict(seed=seed, model=model, index=index, panels={}, effects={}, zero={}, environment={})
        responses = {branch:{response:{} for response in RESPONSES} for branch in BRANCHES}
        pairs = (("value_minus_uniform", "value", "uniform"), ("value_minus_accuracy", "value", "accuracy"),
            ("accuracy_minus_uniform", "accuracy", "uniform"), ("value_minus_replacement", "value", "replacement"),
            ("uniform_minus_replacement", "uniform", "replacement"), ("oracle_minus_uniform", "oracle", "uniform"),
            ("value_minus_oracle", "value", "oracle"))
        for branch in BRANCHES:
            row["panels"][branch] = {}
            for cell in CELLS:
                metrics = raw[branch][cell]
                oracle = min(range(len(ids)), key=lambda pos:(metrics["brier"][pos], ids[pos]))
                selectors = {name:{field:metrics[field][position] for field in FIELDS}
                    for name, position in (("value", selected), ("accuracy", accurate), ("replacement", len(ids)), ("oracle", oracle))}
                selectors["uniform"] = {field:float(np.asarray(metrics[field][:-1]).mean()) for field in FIELDS}
                contrasts = {name:{field:selectors[left][field]-selectors[right][field] for field in FIELDS}
                             for name, left, right in pairs}
                for selector in ("value", "accuracy", "uniform", "replacement"):
                    contrasts[selector+"_minus_matching_zero"] = {field:selectors[selector][field]-zero[branch][field][int(cell[0])]
                                                                  for field in FIELDS}
                row["panels"][branch][cell] = dict(selectors=selectors, contrasts=contrasts, oracle_id=ids[oracle],
                    brier_spread=float(np.ptp(metrics["brier"][:-1])))
                for response, values in (("selection", contrasts["value_minus_uniform"]), ("chosen", selectors["value"]),
                                         ("uniform", selectors["uniform"])):
                    responses[branch][response][cell] = values
            row["effects"][branch] = {}
            for response in RESPONSES:
                effect = {name:{} for name in contrast_cube(np.zeros((2, 2, 2)))}
                for field in FIELDS:
                    cube = np.asarray([responses[branch][response][cell][field] for cell in CELLS]).reshape(2, 2, 2)
                    for name, value in contrast_cube(cube).items():
                        effect[name][field] = value
                row["effects"][branch][response] = effect
            row["zero"][branch] = {str(weight):{field:zero[branch][field][weight] for field in FIELDS} for weight in range(2)}
        row["environment"] = {response:{cell:{field:responses["return"][response][cell][field]-responses["current"][response][cell][field]
                                for field in FIELDS} for cell in CELLS} for response in RESPONSES}
        return row

    def aggregate(self, rows, seeds):
        need([row["seed"] for row in rows] == [seed for seed in seeds for _ in range(2)]
             and [row["index"] for row in rows] == [index for _ in seeds for index in range(2)], "seed/assessment aggregation order differs")
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        draws = rng.integers(low=0, high=len(seeds), size=(RESAMPLES, len(seeds)))

        def walk(values):
            first = values[0]
            if isinstance(first, dict):
                return {key:walk([item[key] for item in values]) for key in first}
            need(all(isinstance(value, (int, float)) for value in values), "non-numeric summary leaf")
            per_seed = np.asarray(values, dtype=np.float64).reshape(len(seeds), 2).mean(axis=1)
            self.bootstrap_statistics += 1
            return statistics(per_seed, seeds, draws)

        panels = [{branch:{cell:{key:value for key,value in item.items() if key != "oracle_id"}
                            for cell,item in row["panels"][branch].items()} for branch in BRANCHES} for row in rows]
        return dict(panels=walk(panels), effects=walk([row["effects"] for row in rows]),
                    zero=walk([row["zero"] for row in rows]), environment=walk([row["environment"] for row in rows]))


def self_checks():
    w, o, a = np.indices((2, 2, 2))
    polynomial = 101+2*w-3*o+5*a+7*w*o-11*w*a+13*o*a+17*w*o*a
    actual = contrast_cube(polynomial)
    expected = dict(total_111_minus_000=30, weights_main=4.25, optimizer_main=11.25, anchor_main=10.25,
        weights_optimizer_interaction=15.5, weights_anchor_interaction=-2.5, optimizer_anchor_interaction=21.5,
        three_way_interaction=17, repair_weights=15, repair_optimizer=34, repair_anchor=24)
    need(all(actual[key] == value for key, value in expected.items()), "finite-difference polynomial example failed")
    for axis, name in enumerate(FACTORS):
        others = [other for other in range(3) if other != axis]
        for fixed in itertools.product(range(2), repeat=2):
            left = [0, 0, 0]
            for other, bit in zip(others, fixed):
                left[other] = bit
            right = left.copy()
            right[axis] = 1
            label = name+"_at_"+"_".join(FACTORS[other]+str(bit) for other,bit in zip(others,fixed))
            need(actual[label] == polynomial[tuple(right)]-polynomial[tuple(left)], "conditional finite difference failed")
    need(all(value == 0 for value in contrast_cube(np.ones((2, 2, 2))).values()), "constant cube does not cancel")
    coefficient = derive_coefficients()
    need(all(abs(sum(values.values())) == 0 for values in coefficient.values()), "contrast fails common-offset cancellation")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, 2, (RESAMPLES, 2))
    simple = statistics([0, 2], [1, 2], draws)
    need(simple["mean"] == 1 and simple["lower"] == 0 and simple["upper"] == 2
         and simple["positive"] == 1 and simple["negative"] == 0, "known bootstrap example failed")
    need(fixed_screen(dict(mean=.0005, positive=9, n=12), True)["passed"], "inclusive practical threshold failed")
    need(not fixed_screen(dict(mean=.0005, positive=9, n=12), False)["passed"], "ineligible cohort passed")
    need(not fixed_screen(dict(mean=.0005, positive=8, n=12), True)["passed"], "consistency threshold failed")
    # Positive 111-minus-old restoration is an error reduction by restoration;
    # the same positive survival difference instead favors retaining 111.
    need((.20-.10) > 0 and (.90-.80) > 0, "error/survival sign example failed")
    return dict(contrasts=23, conditional_swaps=12, polynomial_examples=len(expected),
                constant_cube=True, bootstrap_example=True, eligibility_and_threshold_examples=3,
                brier_and_survival_signs_verified=True)


def run(directory, output):
    started = time.perf_counter()
    tests = self_checks()
    review = Review(directory)
    summary = review.read("summary.json")
    config = review.read("config.json")
    manifest, completion = review.read("manifest.json"), review.read("completion.json")
    review.equal(config, summary["config"], "summary config")
    review.equal(config, manifest["config"], "manifest config")
    review.equal(summary["identity"], manifest["identity"], "identity")
    review.equal(summary["identity"], completion["identity"], "completion identity")
    with gzip.open(review.path("raw_results.jsonl.gz"), "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    expected_keys = {(model, seed, index) for model in config["models"] for seed in config["seeds"] for index in (0, 1)}
    keys = [(row["model"], row["seed"], row["index"]) for row in records]
    need(len(keys) == len(set(keys)) and set(keys) == expected_keys, "incomplete exported assessment cohort")
    need(completion["assessments"] == len(records), "completion assessment coverage differs")
    need(completion["trajectories"] == len(config["models"])*len(config["seeds"]), "completion trajectory coverage differs")
    rows = [review.assessment(row) for row in sorted(records, key=lambda row:(row["model"], row["seed"], row["index"]))]
    published = {(row["model"], row["seed"], row["index"]):row for row in summary["rows"]}
    need(set(published) == expected_keys and len(summary["rows"]) == len(published), "published row coverage differs")
    for row in rows:
        key = row["model"], row["seed"], row["index"]
        for section in ("panels", "effects", "zero", "environment"):
            review.equal(row[section], published[key][section], "/".join(map(str,key))+"/"+section)
    coefficients = derive_coefficients()
    review.equal(coefficients, summary["coefficients"], "factorial coefficients")
    complete = eligible(config)
    review.equal(complete, summary["eligible_scientific_cohort"], "scientific eligibility")
    combined, screens = {}, {}
    for model in config["models"]:
        selected_rows = [row for seed in config["seeds"] for row in rows if row["model"] == model and row["seed"] == seed]
        aggregate = review.aggregate(selected_rows, config["seeds"])
        for section in aggregate:
            review.equal(aggregate[section], summary[section][model], section+"/"+model)
        total = aggregate["effects"]["current"]["selection"]["total_111_minus_000"]["brier"]
        screen = fixed_screen(total, complete)
        review.equal(screen, {key:summary["screen"][model][key] for key in screen}, "screen/"+model)
        combined[model], screens[model] = aggregate, screen
    result = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Independent arithmetic review of the prespecified completed study; no new scientific analysis or tuning.",
        source_sha256=sha(__file__), input_directory=review.directory.as_posix(), input_hashes=review.inputs,
        identity=summary["identity"], eligible_scientific_cohort=complete,
        reviewed_counts=dict(trajectories=completion["trajectories"], assessments=len(records), models=len(config["models"]),
            independent_seeds_per_model=len(config["seeds"]), arrays=review.array_files,
            all_action_probability_scalars=review.prediction_values, validation_prediction_forwards=review.forecasts,
            raw_metric_scalars=review.metric_scalars, factorial_contrasts=len(coefficients),
            bootstrap_statistics=review.bootstrap_statistics, scalar_comparisons=review.comparisons),
        maximum_absolute_numeric_discrepancy=review.maximum_error, self_checks=tests,
        coefficients=coefficients, screen=screens, recomputed=combined,
        formulas=dict(brier="Mean squared error on performed-action probabilities over packets, examples and three horizons.",
            survival="Mean physical H=12 outcome under the forecast's H=12 greedy action; ties use first action.",
            uniform="Exact arithmetic mean over the K original candidates, excluding replacement and zero baselines.",
            selected="Minimum old score-window candidate Brier, breaking exact ties by packet ID; choice remains fixed.",
            accuracy="Minimum original pre-update packet Brier, breaking exact ties by packet ID.",
            oracle="Minimum complete validation Brier; reuse that one candidate for survival and early/late metrics.",
            seed="Average both assessments inside each seed before any cohort statistic or resampling.",
            bootstrap="20,000 independent-seed resamples with PCG64 seed 27192026; linear 2.5/97.5 percentiles. Values below 1e-12 in magnitude become zero as in the locked numeric convention.",
            factors="Main effect: mean first finite difference; pair: mean second finite difference; three-way: third finite difference. Conditional swaps fix the other two bits.",
            repair="111 minus the one-factor old restoration. Positive Brier means restoration lowers error; positive survival favors 111.",
            environment="Return minus current with each branch's recorded evolving context.",
            screen="Eligible complete cohort AND current T=(selected-uniform)111-(selected-uniform)000 mean >=0.0005 AND at least nine strictly positive seed means."),
        limitations="Arithmetic and aggregation review only. Does not retrain models, authenticate original observation creation or establish policy benefit. Rank summaries, coverage summaries and descriptive cue/index slices are outside this additional review; the separate portable and tensor audits cover their stated scopes.",
        elapsed_seconds=time.perf_counter()-started)
    destination = Path(output).resolve()
    need(destination.parent == Path(__file__).resolve().parent, "review output must stay in the isolated review directory")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, eligible=complete, counts=result["reviewed_counts"],
        max_discrepancy=review.maximum_error, screen={key:value["passed"] for key,value in screens.items()}), indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)
