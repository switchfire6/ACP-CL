"""Source-locked acquisition diagnostics with causal replay and crash recovery."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch

from acp_cl.conditional.learner import PacketMemory
from acp_cl.contextual.learner import ContextLearner
from acp_cl.contextual.study import source_manifest as prior_sources
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import atomic_checkpoint, digest, trial_seed, write_json
from acp_cl.persistence.world import concatenate
from .world import (AcquisitionWorld, Law, batch, final_law, mask_affected, mask_valid,
                    order, signal_images, stage_law)


ARMS = ("continue", "frozen", "state_reset", "fresh")
BRANCHES = ("return", "revision", "noise")


def source_manifest():
    sources = prior_sources()
    base = Path(__file__).resolve().parent
    for path in sorted(base.glob("*.py")):
        sources[path.relative_to(base.parent.parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def validate_config(c):
    fields = {"study", "kind", "seeds", "models", "prefix_blocks", "prefix_size", "episode_size",
        "batch_size", "eval_size", "probe_every", "support_replicates", "width", "experts",
        "context_width", "decoder_width", "interaction_features", "lr", "memory_packets",
        "updates_per_batch", "evidence_strength", "feedback_noise", "threads", "workers"}
    if set(c) != fields or c["kind"] not in ("development", "diagnostic"):
        raise ValueError("invalid configuration fields or kind")
    ints = fields-{"study", "kind", "seeds", "models", "interaction_features", "lr", "evidence_strength", "feedback_noise"}
    if any(type(c[k]) is not int or c[k] < 1 for k in ints):
        raise ValueError("positive integer settings required")
    if c["prefix_blocks"] < 2 or c["prefix_blocks"] % 2 or c["experts"] != 4:
        raise ValueError("even prefix block count and four reference heads required")
    if c["batch_size"] % 8 or c["eval_size"] % 64:
        raise ValueError("batch/evaluation factorial size mismatch")
    if c["prefix_size"] % c["probe_every"] or c["episode_size"] % c["probe_every"] or c["probe_every"] % c["batch_size"]:
        raise ValueError("incompatible exposure/probe/batch sizes")
    if type(c["interaction_features"]) is not bool or not c["interaction_features"]:
        raise ValueError("use the qualified recurrent recipe")
    if any(not np.isfinite(c[k]) or c[k] <= 0 for k in ("lr", "evidence_strength")) or not 0 <= c["feedback_noise"] <= 1:
        raise ValueError("invalid continuous setting")
    for key in ("models", "seeds"):
        if not c[key] or len(c[key]) != len(set(c[key])):
            raise ValueError("nonempty unique selections required")
    if any(m not in ("conditional", "recurrent") for m in c["models"]) or any(type(s) is not int or s < 0 for s in c["seeds"]):
        raise ValueError("invalid model or seed")


class Evaluator:
    def __init__(self, seed, settings):
        self.seed, self.settings = seed, settings
        self.world, self.cache = AcquisitionWorld(), {}

    def dataset(self, law):
        law = replace(law, noise=0.)
        if law not in self.cache:
            cases = self.world.dataset(law, self.settings["eval_size"], trial_seed(self.seed, "acquisition_query", 0))
            self.cache[law] = cases, self.world.counterfactuals(cases)
        return self.cache[law]

    def evaluate(self, learner, law, support, cue=None, branch="novel", flipped=False, marginal=None):
        cases, truth = self.dataset(law)
        observations = cases.observations
        if flipped:
            if cue is None:
                raise ValueError("flipped-cue probe requires a cue")
            signals = cases.signals.copy()
            signals[:, cue] ^= 1
            observations = signal_images(observations, signals)
        probabilities = learner.predict(observations, support)[0]
        actions = probabilities[:, :, -1].argmax(axis=1)
        survival = truth[np.arange(len(actions)), actions, -1]
        brier = ((probabilities-truth)**2).mean(axis=(1, 2))
        affected, valid = mask_affected(cases, cue), mask_valid(cases, branch)
        result = dict(brier=float(brier.mean()), focus_brier=float(brier[affected].mean()),
            survival=float(survival.mean()), focus_survival=float(survival[affected].mean()),
            valid_brier=float(brier[valid].mean()), valid_survival=float(survival[valid].mean()),
            no_transfer=float(truth[:, 0, -1].mean()),
            clairvoyant_upper=float(truth[:, :, -1].max(axis=1).mean()))
        if marginal is not None:
            result["marginal_brier"] = float(((marginal[None]-truth)**2).mean())
        return result

    def support(self, law, replicate, valid_only=False, branch="novel"):
        size = self.settings["batch_size"]
        seed = trial_seed(self.seed, "acquisition_support", [replicate, valid_only])
        if not valid_only:
            return self.world.experience(law, size, seed)
        multiplier = 1 if branch == "return" else 2 if branch == "revision" else 4
        clean = replace(law, noise=0.)
        cases = self.world.dataset(clean, size*multiplier, seed)
        data = self.world.experience(clean, size*multiplier, seed)
        return data.take(mask_valid(cases, branch))

    def probe(self, learner, law, cue=None, branch="novel", marginal=None):
        before = state_hash(learner.model.state_dict())
        size, rows = self.settings["batch_size"], []
        for rep in range(self.settings["support_replicates"]):
            fresh = self.support(law, rep)
            curve = []
            for n in sorted(set((0, size//4, size//2, size))):
                support = learner.history
                if n:
                    added = fresh.take(slice(0, n))
                    support = added if support is None else concatenate(support, added).take(slice(-size, None))
                curve.append(dict(feedback=n, metrics=self.evaluate(learner, law, support, cue, branch,
                                                                    marginal=marginal)))
            row = dict(curve=curve)
            if cue is not None:
                row["flipped"] = self.evaluate(learner, law, fresh, cue, branch, flipped=True)
            rows.append(row)
        if before != state_hash(learner.model.state_dict()):
            raise AssertionError("probe changed parameters")
        return dict(law=asdict(law), model_sha256=before, replicates=rows)

    def valid_panel(self, learner, law, branch="novel"):
        result = {}
        for mode in (0, 1):
            target = replace(law, mode=mode, noise=0.)
            rows = [self.evaluate(learner, target, self.support(target, r, True, branch), branch=branch)
                    for r in range(self.settings["support_replicates"])]
            result[str(mode)] = {k: float(np.mean([r[k] for r in rows])) for k in ("valid_brier", "valid_survival")}
        return result


def fork(learner, arm, seed, config):
    if arm not in ARMS:
        raise ValueError("unknown diagnostic arm")
    if arm == "fresh":
        result = ContextLearner(learner.method, seed, config, str(learner.device))
        result.history = copy.deepcopy(learner.history)
    else:
        result = copy.deepcopy(learner)
        if arm == "frozen":
            result.freeze_features()
        if arm == "state_reset":
            result.optimizer = torch.optim.Adam(result.model.parameters(), lr=config["lr"])
            result.memory = PacketMemory(config["memory_packets"], seed)
    return result


def load(path, identity, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["identity"] != identity:
        raise ValueError("checkpoint identity mismatch")
    return payload


def marginal_update(record, data):
    for action in range(5):
        mask = data.actions == action
        record["marginal_count"][action] += int(mask.sum())
        record["marginal_success"][action] = (np.asarray(record["marginal_success"][action])
                                             + data.survival[mask].sum(axis=0)).tolist()


def marginal(record):
    return (np.asarray(record["marginal_success"])+.5)/(np.asarray(record["marginal_count"])[:, None]+1.)


def episode(config, identity, learner, seed, stage, arm, law, directory, cue=None, branch="novel"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, result_path = directory/"checkpoint.pt", directory/"result.json"
    device = str(learner.device)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError("result identity mismatch")
        return load(checkpoint, identity, device)["learner"], result
    evaluator, world = Evaluator(seed, config), AcquisitionWorld()
    channel = f"stage_{stage}" if branch == "novel" else "final"
    if checkpoint.exists():
        saved = load(checkpoint, identity, device)
        learner, record = saved["learner"], saved["record"]
    else:
        record = dict(identity=identity, seed=seed, model=learner.method, stage=stage, arm=arm,
            branch=branch, law=asdict(law), cue=cue, channel=channel, batches_done=0, batch_sha256=[],
            before=evaluator.probe(learner, law, cue, branch),
            valid_before=evaluator.valid_panel(learner, law, branch), start_diagnostics=learner.diagnostics(),
            curve=[dict(arrivals=0, metrics=evaluator.evaluate(learner, law, learner.history, cue, branch))],
            marginal_count=[0]*5, marginal_success=[[0]*3 for _ in range(5)], elapsed_seconds=0.)
        atomic_checkpoint(directory/"before.pt", dict(identity=identity, learner=learner, record=record))
    started = time.perf_counter()
    for index in range(record["batches_done"], config["episode_size"]//config["batch_size"]):
        data = batch(world, law, seed, channel, index, config["batch_size"])
        learner.train(data)
        marginal_update(record, data)
        record["batch_sha256"].append(data.fingerprint())
        record["batches_done"] = index+1
        arrivals = (index+1)*config["batch_size"]
        if arrivals % config["probe_every"] == 0:
            record["curve"].append(dict(arrivals=arrivals,
                metrics=evaluator.evaluate(learner, law, learner.history, cue, branch, marginal=marginal(record))))
            record["elapsed_seconds"] += time.perf_counter()-started
            atomic_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record))
            started = time.perf_counter()
    record["after"] = evaluator.probe(learner, law, cue, branch, marginal=marginal(record))
    record["valid_after"] = evaluator.valid_panel(learner, law, branch)
    record["diagnostics"] = learner.diagnostics()
    record["elapsed_seconds"] += time.perf_counter()-started
    write_json(result_path, record)
    return learner, record


def run_prefix(config, identity, seed, model, directory, device):
    checkpoint = directory/"prefix.pt"
    world, evaluator = AcquisitionWorld(), Evaluator(seed, config)
    if checkpoint.exists():
        saved = load(checkpoint, identity, device)
        learner, record = saved["learner"], saved["record"]
    else:
        learner = ContextLearner(model, seed, config, device)
        record = dict(blocks=[])
    for block in range(len(record["blocks"]), config["prefix_blocks"]):
        law = Law((seed+block) % 2)
        hashes = []
        for index in range(config["prefix_size"]//config["batch_size"]):
            data = batch(world, law, seed, f"prefix_{block}", index, config["batch_size"])
            learner.train(data)
            hashes.append(data.fingerprint())
        record["blocks"].append(dict(law=asdict(law), batch_sha256=hashes,
            endpoint=evaluator.evaluate(learner, law, learner.history), diagnostics=learner.diagnostics()))
        atomic_checkpoint(checkpoint, dict(identity=identity, learner=learner, record=record))
    write_json(directory/"prefix.json", dict(identity=identity, seed=seed, model=model, **record))
    return learner


def preceding_history(config, seed, stage):
    if stage == 1:
        channel, size = f"prefix_{config['prefix_blocks']-1}", config["prefix_size"]
    else:
        channel, size = f"stage_{stage-1}", config["episode_size"]
    return batch(AcquisitionWorld(), stage_law(seed, stage-1), seed, channel,
                 size//config["batch_size"]-1, config["batch_size"])


def run_job(config, identity, seed, model, output, device):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/f"{model}_{seed}"
    directory.mkdir(exist_ok=True)
    learner = None if config["kind"] == "development" else run_prefix(config, identity, seed, model, directory, device)
    for stage in range(1, 4):
        law, cue = stage_law(seed, stage), order(seed)[stage-1]
        if config["kind"] == "development":
            fresh = ContextLearner(model, seed, config, device)
            fresh.history = preceding_history(config, seed, stage)
            episode(config, identity, fresh, seed, stage, "fresh", law, directory/f"stage_{stage}_fresh", cue)
        else:
            before_hash = state_hash(learner.model.state_dict())
            continuing = None
            for arm in ARMS:
                end, _ = episode(config, identity, fork(learner, arm, seed, config), seed, stage, arm,
                                  law, directory/f"stage_{stage}_{arm}", cue)
                if arm == "continue":
                    continuing = end
            if state_hash(learner.model.state_dict()) != before_hash:
                raise AssertionError("a diagnostic arm changed its parent")
            learner = continuing
    if config["kind"] == "diagnostic":
        for branch in BRANCHES:
            law = final_law(seed, branch, config["feedback_noise"])
            episode(config, identity, copy.deepcopy(learner), seed, 4, "continue", law,
                    directory/f"final_{branch}", 0 if branch == "revision" else None, branch)
    return dict(seed=seed, model=model)


def run_suite(config, output, device="cpu", resume=False, protocol=None):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    runtime = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, device=device, threads=config["threads"], workers=config["workers"], deterministic=True)
    identity = dict(config_sha256=digest(config), source_sha256=digest(sources), runtime_sha256=digest(runtime))
    path = output/"manifest.json"
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("resume requires identical config/source/runtime")
    else:
        if any(output.iterdir()):
            raise ValueError("new output must be empty")
        now = datetime.now(timezone.utc).isoformat()
        write_json(path, dict(identity=identity, config=config, runtime=runtime, source_files=sources, created_utc=now))
        root = Path(__file__).resolve().parent.parent.parent
        with zipfile.ZipFile(output/"training_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in sources:
                z.write(root/name, name)
        if protocol:
            content = Path(protocol).read_bytes()
            (output/"protocol_at_lock.md").write_bytes(content)
            write_json(output/"protocol_lock.json", dict(**identity, locked_utc=now,
                protocol_sha256=hashlib.sha256(content).hexdigest()))
    jobs = [(config, identity, s, m, output, device) for s in config["seeds"] for m in config["models"]]
    if config["workers"] == 1:
        for index, job in enumerate(jobs, 1):
            print(f"{index}/{len(jobs)} trajectories: {run_job(*job)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [pool.submit(run_job, *job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                print(f"{index}/{len(jobs)} trajectories: {future.result()}", flush=True)
    if digest(source_manifest()) != identity["source_sha256"]:
        raise ValueError("scientific source changed during run")
    count = len(jobs)*(3 if config["kind"] == "development" else 15)
    write_json(output/"completion.json", dict(identity=identity, trajectories=len(jobs), episodes=count,
        prefix_fits=0 if config["kind"] == "development" else len(jobs),
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--protocol")
    args = parser.parse_args()
    run_suite(json.loads(Path(args.config).read_text(encoding="utf-8")), args.output,
              args.device, args.resume, args.protocol)
