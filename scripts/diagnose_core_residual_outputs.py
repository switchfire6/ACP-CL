"""Bounded saved-checkpoint output diagnostic. Never calls a training method."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import random
import zipfile

import numpy as np
import torch

from acp_cl.core_residual.evaluation import array_hash, recompute_metrics
from acp_cl.core_residual.learner import learner_signature
from acp_cl.core_residual.study import check_locks
from acp_cl.core_residual_outputs import component_predictions
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.world import Experience
from acp_cl.predictive_value.study import load
from acp_cl.replay_renewal.study import affinity


ROOT = Path(__file__).resolve().parents[1]
OWN_FILES = (
    "src/acp_cl/core_residual_outputs.py", "scripts/diagnose_core_residual_outputs.py",
    "scripts/summarize_core_residual_outputs.py", "tests/test_core_residual_outputs.py",
    "docs/core_residual_output_review.md", "scripts/plot_core_residual_outputs.py",
)
ARMS = ("joint", "separate")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rng_hash():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
        torch.get_rng_state().numpy()), protocol=5)).hexdigest()


def selections(probe, target):
    """The duplicate zero-feedback probe replicates count once."""
    yield "actual", 0, False, probe["replicates"][0]["curve"][0]["metrics"]["trace"]
    for index, row in enumerate(probe["replicates"]):
        yield "refreshed", index, False, row["curve"][-1]["metrics"]["trace"]
        if target in ("novel_end", "novel_after_return"):
            yield "refreshed", index, True, row["flipped"]["trace"]


def job(base, output, seed, model, identity, expected_job_sha, support_replicates):
    affinity(1365)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    directory = Path(base) / f"{model}_{seed}"
    require(sha(directory / "result.json") == expected_job_sha, "base job ledger changed")
    job_result = read(directory / "result.json")
    rows, arrays, checkpoints = [], {}, []
    frozen_predictions, frozen_hash = {}, None
    frozen_checks = 0
    for arm in ARMS:
        phase_data = {}
        for phase in ("novel", "return"):
            folder = directory / arm / phase
            result_path = folder / "result.json"
            require(sha(result_path) == job_result["phase_hashes"][f"{arm}/{phase}"],
                    "base phase record changed")
            record = read(result_path)
            for key, filename in (("json", "evaluations.json"), ("arrays", "evaluations.npz")):
                require(sha(folder / filename) == record["evaluation_files"][key]["sha256"],
                        "base trace changed")
            phase_data[phase] = (record, read(folder / "evaluations.json"), folder)
        for checkpoint_phase in ("novel", "return"):
            record, _, folder = phase_data[checkpoint_phase]
            cp = folder / "checkpoint.pt"
            cp_hash = sha(cp)
            require(cp_hash == job_result["checkpoint_hashes"][f"{arm}/{checkpoint_phase}"],
                    "base checkpoint changed")
            saved = load(cp, identity, "cpu")
            learner = saved["learner"]
            signature = learner_signature(learner)
            require(signature == record["end_signature"], "checkpoint state differs from record")
            require(json.loads(json.dumps(saved["record"], allow_nan=False)) == record,
                    "checkpoint metadata differs from phase record after JSON tuple/list normalization")
            global_rng = rng_hash()
            weights_hash = state_hash(learner.model.state_dict())
            if arm == "separate":
                core_hash = state_hash(learner.model.core.state_dict())
                if frozen_hash is not None:
                    require(core_hash == frozen_hash, "frozen core changed between endpoints")
                frozen_hash = core_hash
            if checkpoint_phase == "novel":
                require(signature == phase_data["return"][0]["start_signature"],
                        "novel end is not exact return-entry state")
                targets = (("novel_end", "novel", "after"),
                           ("return_entry", "return", "before"))
            else:
                targets = (("return_end", "return", "after"),
                           ("novel_after_return", "return", "novel_after_return"))
            for target, trace_phase, probe_name in targets:
                trace_record, trace, trace_folder = phase_data[trace_phase]
                probe = trace_record[probe_name]
                require(len(probe["replicates"]) == support_replicates, "support replication changed")
                require(probe["model_sha256"] == weights_hash, "probe is not from saved weights")
                with np.load(trace_folder / "evaluations.npz", allow_pickle=False) as source:
                    for context, replicate, flipped, trace_index in selections(probe, target):
                        original = trace["records"][trace_index]
                        require(original["index"] == trace_index and original["flipped"] == flipped,
                                "trace selection mismatch")
                        require(original["model_sha256"] == weights_hash, "trace weights mismatch")
                        values = {key: source[f"e{trace_index}_{key}"].copy()
                                  for key in original["array_sha256"]}
                        require(all(array_hash(value) == original["array_sha256"][key]
                                    for key, value in values.items()), "base array hash mismatch")
                        support = Experience(*(values[key] for key in
                            ("support_observations", "support_actions", "support_outcomes")))
                        require(support.fingerprint() == original["support_fingerprint"],
                                "support identity mismatch")
                        require(len(support) == learner.settings["batch_size"], "support budget changed")
                        if context == "actual":
                            require(support.fingerprint() == learner.history.fingerprint(),
                                    "actual context differs from causal history")
                        predictions = component_predictions(learner, values["observations"], support)
                        require(np.array_equal(predictions["full"], values["probabilities"]),
                                "decomposition does not reconstruct archived original forecast exactly")
                        if arm == "separate":
                            key = (original["query_sha256"], support.fingerprint())
                            prediction_hash = array_hash(predictions["core"])
                            if key in frozen_predictions:
                                require(frozen_predictions[key] == prediction_hash,
                                        "frozen core forecast changed for identical observable input")
                                frozen_checks += 1
                            frozen_predictions[key] = prediction_hash
                        metrics = {cell: recompute_metrics(p, values["truth"], values["affected"],
                                                         values["valid"])
                                   for cell, p in predictions.items()}
                        require(all(metrics["full"][key] == original["metrics"][key]
                                    for key in metrics["full"]), "original metric mismatch")
                        values["archived_full"] = values.pop("probabilities")
                        values.update({"probabilities_" + key: value for key, value in predictions.items()})
                        index = len(rows)
                        arrays.update({f"c{index}_{key}": value for key, value in values.items()})
                        provenance = dict(
                            phase=trace_phase, trace=trace_index,
                            checkpoint=cp.relative_to(Path(base)).as_posix(), checkpoint_sha256=cp_hash,
                            trace_json=(trace_folder / "evaluations.json").relative_to(Path(base)).as_posix(),
                            trace_json_sha256=trace_record["evaluation_files"]["json"]["sha256"],
                            trace_arrays=(trace_folder / "evaluations.npz").relative_to(Path(base)).as_posix(),
                            trace_arrays_sha256=trace_record["evaluation_files"]["arrays"]["sha256"],
                            query_sha256=original["query_sha256"],
                            support_fingerprint=original["support_fingerprint"], model_sha256=weights_hash)
                        rows.append(dict(index=index, model=model, seed=seed, arm=arm, target=target,
                            context=context, replicate=replicate, flipped=flipped, cue=original["cue"],
                            novel_cue=phase_data["novel"][0]["cue"], source=provenance,
                            signature=signature, metrics=metrics,
                            array_sha256={key: array_hash(value) for key, value in values.items()}))
            require(learner_signature(learner) == signature, "ablation changed learner state")
            require(rng_hash() == global_rng, "ablation changed global random state")
            require(sha(cp) == cp_hash, "saved checkpoint file changed")
            checkpoints.append(dict(path=cp.relative_to(Path(base)).as_posix(), sha256=cp_hash,
                signature_before=signature, signature_after=learner_signature(learner),
                rng_before=global_rng, rng_after=rng_hash()))
    require(len(rows) == 8 + 12 * support_replicates, "incomplete endpoint panel")
    require(frozen_checks >= 3 * support_replicates, "missing repeated-input frozen-core checks")
    stem = Path(output) / "jobs" / f"{model}_{seed}"
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    result = dict(model=model, seed=seed, support_replicates=support_replicates,
        rows=rows, checkpoints=checkpoints,
        frozen_core_forecast_pairs_checked=frozen_checks,
        arrays=dict(file=stem.with_suffix(".npz").name, sha256=sha(stem.with_suffix(".npz"))))
    write(stem.with_suffix(".json"), result)
    return dict(model=model, seed=seed, rows=len(rows), checkpoints=len(checkpoints),
                frozen_pairs=frozen_checks, json_sha256=sha(stem.with_suffix(".json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    require(1 <= args.workers <= 6, "one to six workers required")
    base = args.base.resolve()
    output = args.output.resolve()
    require(base != output and base not in output.parents, "diagnostic must have a separate output directory")
    base_manifest = check_locks(base)
    config = base_manifest["config"]
    require(config["seeds"] in (list(range(16001, 16007)), [16991]), "unsupported saved cohort")
    require(config["models"] == ["conditional", "recurrent"]
            and config["support_replicates"] == (1 if config["seeds"] == [16991] else 2),
            "expected complete two-model panel and original support replication")
    affinity(1365)
    files = {name: sha(ROOT / name) for name in OWN_FILES}
    base_files = {name: sha(base / name) for name in
                  ("manifest.json", "analysis_lock.json", "protocol_lock.json", "completion.json",
                   "training_source.zip", "analysis_at_lock.zip")}
    base_files.update({f"{model}_{seed}/result.json": sha(base / f"{model}_{seed}" / "result.json")
                       for model in config["models"] for seed in config["seeds"]})
    specification = dict(base=str(base), identity=base_manifest["identity"], base_files=base_files,
        seeds=config["seeds"], models=config["models"], arms=list(ARMS),
        support_replicates=config["support_replicates"], source_files=files,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                     workers=args.workers, threads=1, affinity=1365),
        interpretation="exploratory saved-endpoint diagnostic; no fitting or deployable policy")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        require(manifest["specification"] == specification, "diagnostic lock changed")
        require(sha(output / "source_at_lock.zip") == manifest["source_archive_sha256"],
                "diagnostic source archive changed")
    else:
        require(not list(output.iterdir()), "new diagnostic requires empty output directory")
        with zipfile.ZipFile(output / "source_at_lock.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                archive.write(ROOT / name, name)
        manifest = dict(locked_utc=datetime.now(timezone.utc).isoformat(), specification=specification,
                        source_archive_sha256=sha(output / "source_at_lock.zip"))
        write(manifest_path, manifest)
    if args.lock_only:
        print(json.dumps(dict(locked=True, utc=manifest["locked_utc"], jobs=len(config["seeds"]) * 2)))
        return
    (output / "jobs").mkdir(exist_ok=True)
    require(not list((output / "jobs").iterdir()), "refuse to overwrite existing diagnostic forecasts")
    finished = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(job, base, output, seed, model, base_manifest["identity"],
                               base_files[f"{model}_{seed}/result.json"], config["support_replicates"])
                   for model in config["models"] for seed in config["seeds"]]
        for future in as_completed(pending):
            result = future.result()
            finished.append(result)
            print(json.dumps(result), flush=True)
    require(all(sha(ROOT / name) == value for name, value in files.items()), "source changed during diagnostic")
    require(all(sha(base / name) == value for name, value in base_files.items()), "base ledger changed")
    write(output / "completion.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        jobs=sorted(finished, key=lambda row: (row["model"], row["seed"])),
        rows=sum(row["rows"] for row in finished), checkpoints=sum(row["checkpoints"] for row in finished),
        frozen_core_forecast_pairs_checked=sum(row["frozen_pairs"] for row in finished),
        optimizer_updates=0))


if __name__ == "__main__":
    main()
