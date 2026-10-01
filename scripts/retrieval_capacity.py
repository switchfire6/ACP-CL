"""Step A1: retrieval capacity of the saved interleaved learners.

PROSPECTIVE DIAGNOSTIC under docs/retrieval_capacity_protocol.md (written
2026-09-28 before any computation). It is a diagnostic on saved models: it does
not reopen the stopped representation recipe and changes no locked record.

Every probe operates on a deep copy of the loaded endpoint learner. The original
learner signature (acp_cl.representation_learning.learner.learner_signature) and
the global torch/numpy/python RNG states are checked before and after. Fitting
data are fresh simulator records whose seeds come from
trial_seed(seed, "retrieval_capacity_<probe>", ...), disjoint from project seeds.
Fits use only ordinary learner data (images, uniform random performed action and
its three survival labels), except P4 whose linear probes are by design fitted
to evaluator-only factor/signal labels (diagnostic only).

Probes (per seed 18101-18106):
  P1 support resampling, 64 fresh Base A (and Base B, descriptive) supports.
  P2 context-only fit of a free 12-d context c (primary), with Base B control
     and a Base A -> Base B specificity check.
  P3 decoder refit (encoder and context network frozen) for Base A and the first
     delay-active stage law.
  P4 logistic probes on frozen query latents.

Usage:  python -B scripts/retrieval_capacity.py   (writes reports/retrieval_capacity/)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # sealed file inventories: never create __pycache__

import copy  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import ideal_observer_load as iol  # noqa: E402
from acp_cl.acquisition.world import AcquisitionWorld, Law, order, stage_law  # noqa: E402
from acp_cl.persistence.study import trial_seed  # noqa: E402
from acp_cl.predictive_value.mechanism import predict_all  # noqa: E402
from acp_cl.predictive_value.study import load  # noqa: E402
from acp_cl.representation_learning.design import law_key, qualification_phases, reference_laws  # noqa: E402
from acp_cl.representation_learning.learner import learner_signature  # noqa: E402

LABEL = "PROSPECTIVE DIAGNOSTIC under docs/retrieval_capacity_protocol.md (Step A1)"
RUN = ROOT / "runs" / "representation_learning_qualification"
IDEAL = ROOT / "reports" / "information_audit" / "ideal_observer" / "results.json"
OUT = ROOT / "reports" / "retrieval_capacity"
SEEDS = tuple(range(18101, 18107))
BATCH = 32
# Protocol constants.
P1_SUPPORTS = 64
P2_RECORDS, P2_STEPS, P2_LR = 8192, 1000, .01
P3_RECORDS, P3_EPOCHS, P3_LR, P3_MINIBATCH = 8192, 20, .002, 32
P4_FIT, P4_EVAL = 4096, 4096
# Implementation choices not fixed by the protocol (recorded in results.json).
P4_L2, P4_MAX_ITER = 1e-3, 500
WORKERS = 6


# ----------------------------------------------------------------- utilities

def rng_state_hash():
    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().numpy().tobytes())
    state = np.random.get_state()
    digest.update(repr((state[0], state[2], state[3], state[4])).encode())
    digest.update(np.asarray(state[1]).tobytes())
    digest.update(repr(random.getstate()).encode())
    return digest.hexdigest()


def experience_fingerprints(batches):
    digest = hashlib.sha256()
    for item in batches:
        digest.update(item.fingerprint().encode())
    return digest.hexdigest()


def fresh(world, law, seed, probe, tag, count):
    """``count`` fresh ordinary 32-record batches; seeds disjoint from project seeds."""
    seeds = [trial_seed(seed, f"retrieval_capacity_{probe}", [tag, i]) for i in range(count)]
    return [world.experience(law, BATCH, s) for s in seeds], seeds


def tensors(data):
    return (torch.as_tensor(data.observations), torch.as_tensor(data.actions.astype(np.int64)),
            torch.as_tensor(data.survival.astype(np.float32)))


def context_of(model, support_features, actions, outcomes):
    """Mean-pooled context GRU over interaction events, exactly as the learner."""
    events = torch.cat((support_features, F.one_hot(actions, 5).float(), outcomes), dim=-1)
    interactions = (support_features[:, :, None] * F.one_hot(actions, 5)[:, None, :]
                    * (2 * outcomes[:, -1] - 1)[:, None, None]).flatten(start_dim=1)
    events = torch.cat((events, interactions), dim=-1)
    return model.context(events[None])[0].mean(dim=1)  # (1, 12)


def decode(decoder, query_features, context):
    logits = decoder(torch.cat((query_features, context.expand(len(query_features), -1)), dim=-1))
    return logits.reshape(-1, 5, 3).sigmoid()


def monotone(probabilities):
    """predict_all / ConditionalLearner.predict convention: cummin over horizons."""
    return probabilities.cummin(dim=-1).values


def forward(model, observations, support, decoder=None, joint=True):
    """Re-implemented learner forward (support-derived context), returns numpy forecasts."""
    decoder = model.decoder if decoder is None else decoder
    sx, sa, sy = tensors(support)
    query = torch.as_tensor(observations)
    if joint:  # the learner encodes cat(support, query) in one call
        features = model.encode(torch.cat((sx, query)))
        fs, fq = features[:len(sx)], features[len(sx):]
    else:
        fs, fq = model.encode(sx), model.encode(query)
    return monotone(decode(decoder, fq, context_of(model, fs, sa, sy))).numpy()


def brier(forecast, evaluation):
    return iol.locked_score(np.asarray(forecast), evaluation.truth, evaluation.affected,
                            evaluation.valid)["brier"]


def slot_summary(targets):
    return [dict(slot=t["slot"], law=t["law"], brier=t["metrics"]["brier"],
                 focus_brier=t["metrics"]["focus_brier"], marginal_gain=t["marginal_gain"],
                 cue_benefits=t["cue_benefits"]) for t in targets]


def block_end_reference(record, law):
    """Evaluator curve entries after each block of ``law`` (same-law support = learner.history)."""
    entries = [(i, c) for i, c in enumerate(record["curve"]) if c["law"] == iol.law_record(law)]
    iol.require(len(entries) == 8 and [i % 5 for i, _ in entries] == [entries[0][0]] * 8,
                "expected one block-end entry per cycle")
    values = [c["metrics"]["brier"] for _, c in entries]
    return dict(curve_indices=[i for i, _ in entries], arrivals=[c["arrivals"] for _, c in entries],
                brier_by_cycle=values, cycles_2_8=float(np.mean(values[1:])),
                all_cycles=float(np.mean(values)), cycle_8=float(values[-1]))


def logistic_probe(x_fit, y_fit, x_eval, y_eval):
    """L2-regularised logistic regression (torch LBFGS, float64) on standardised latents."""
    mean, std = x_fit.mean(0, keepdim=True), x_fit.std(0, keepdim=True).clamp_min(1e-8)
    a, b = (x_fit - mean) / std, (x_eval - mean) / std
    weight = torch.zeros(a.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1., max_iter=P4_MAX_ITER,
                                  tolerance_grad=1e-10, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(a @ weight + bias, y_fit)
                + .5 * P4_L2 * weight.square().sum())
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final = float(F.binary_cross_entropy_with_logits(a @ weight + bias, y_fit)
                      + .5 * P4_L2 * weight.square().sum())
        fit_acc = float(((a @ weight + bias > 0).double() == y_fit).double().mean())
        eval_acc = float(((b @ weight + bias > 0).double() == y_eval).double().mean())
    majority = float(y_fit.mean() >= .5)
    return dict(heldout_accuracy=eval_acc, fit_accuracy=fit_acc, final_loss=final,
                heldout_majority_baseline=float((y_eval == majority).double().mean()),
                heldout_positive_rate=float(y_eval.mean()))


# --------------------------------------------------------------- per seed

def run_seed(seed):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng_before = rng_state_hash()
    manifest = iol.read_json(RUN / "manifest.json")
    identity = manifest["identity"]
    folder = RUN / "jobs" / f"outcome_{seed}" / "interleaved"
    record = iol.read_json(folder / "result.json")
    endpoint = iol.load_endpoint(RUN, seed, "interleaved", verify=True)
    saved = load(folder / "checkpoint.pt", identity)
    original = saved["learner"]
    signature_before = learner_signature(original)
    iol.require(signature_before == record["end_signature"], "checkpoint is not the recorded endpoint")
    learner = copy.deepcopy(original)
    # Parameter.__deepcopy__ drops .grad; mirror it so the copy's signature is comparable.
    for (name, source_p), (other, target_p) in zip(original.model.named_parameters(),
                                                   learner.model.named_parameters()):
        iol.require(name == other, "parameter order differs")
        target_p.grad = None if source_p.grad is None else source_p.grad.detach().clone()
    iol.require(learner_signature(learner) == signature_before, "deep copy differs")
    model = learner.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    world = AcquisitionWorld()
    laws = reference_laws(seed)
    base_a, base_b = laws[0], laws[1]
    delay_stage = order(seed).index(1) + 1
    delay_law = stage_law(seed, delay_stage)
    delay_slot = delay_stage + 1
    iol.require([t.law for t in endpoint.targets] == laws and endpoint.targets[delay_slot].law == delay_law
                and 1 in delay_law.active, "endpoint target layout differs")
    data_seeds = {}
    out = dict(seed=seed, order=list(order(seed)), delay_stage=delay_stage, delay_slot=delay_slot,
               laws=dict(base_a=iol.law_record(base_a), base_b=iol.law_record(base_b),
                         delay=iol.law_record(delay_law)))

    # ---- saved learner endpoint (S) and references (E)
    saved_targets = iol.endpoint_targets(endpoint)
    out["saved"] = slot_summary(saved_targets)
    out["S_base_a"] = saved_targets[0]["metrics"]["brier"]
    out["S_base_b"] = saved_targets[1]["metrics"]["brier"]
    out["E_base_a"] = block_end_reference(record, base_a)
    out["E_delay_law"] = block_end_reference(record, delay_law)

    # ---- verification: forward reproduction of every saved endpoint evaluation
    verification = dict(evaluations=0, predict_all_exact=True, joint_max_abs=0., split_max_abs=0.,
                        joint_exact_all=True, base_a_joint_max_abs=0., base_a_joint_exact=True)
    with torch.no_grad():
        for evaluation in endpoint.evaluations:
            support = evaluation.support()
            p_predict = predict_all(learner, np.asarray(evaluation.observations), support)
            p_joint = forward(model, evaluation.observations, support, joint=True)
            p_split = forward(model, evaluation.observations, support, joint=False)
            target = evaluation.probabilities
            verification["evaluations"] += 1
            verification["predict_all_exact"] &= bool(np.array_equal(p_predict, target))
            verification["joint_exact_all"] &= bool(np.array_equal(p_joint, target))
            verification["joint_max_abs"] = max(verification["joint_max_abs"], float(np.abs(p_joint - target).max()))
            verification["split_max_abs"] = max(verification["split_max_abs"], float(np.abs(p_split - target).max()))
            if evaluation.slot == 0:
                verification["base_a_joint_exact"] &= bool(np.array_equal(p_joint, target))
                verification["base_a_joint_max_abs"] = max(verification["base_a_joint_max_abs"],
                                                           float(np.abs(p_joint - target).max()))
    iol.require(verification["joint_max_abs"] < 1e-5 and verification["split_max_abs"] < 1e-5,
                "re-implemented forward does not reproduce the saved learner")
    out["verification"] = verification

    panels = {slot: endpoint.targets[slot].correct for slot in range(5)}
    for slot, (first, second) in panels.items():
        iol.require(np.array_equal(first.observations, second.observations)
                    and np.array_equal(first.truth, second.truth), "replicate query panels differ")

    # ---- P1 support resampling
    p1 = {}
    mean_context = {}
    with torch.no_grad():
        for name, law, slot in (("base_a", base_a, 0), ("base_b", base_b, 1)):
            supports, seeds = fresh(world, law, seed, "P1", law_key(law), P1_SUPPORTS)
            data_seeds[f"P1/{name}"] = seeds
            panel = panels[slot][0]
            fq = model.encode(torch.as_tensor(panel.observations))
            contexts, forecasts, single = [], [], []
            for support in supports:
                sx, sa, sy = tensors(support)
                context = context_of(model, model.encode(sx), sa, sy)
                contexts.append(context)
                forecast = monotone(decode(model.decoder, fq, context)).numpy()
                forecasts.append(forecast)
                single.append(brier(forecast, panel))
            contexts = torch.cat(contexts)
            mean_context[name] = contexts.mean(0, keepdim=True)
            averaged = np.mean(np.asarray(forecasts, dtype=np.float64), axis=0)
            saved_contexts = []
            for evaluation in panels[slot]:
                sx, sa, sy = tensors(evaluation.support())
                saved_contexts.append(context_of(model, model.encode(sx), sa, sy))
            saved_contexts = torch.cat(saved_contexts)
            p1[name] = dict(
                slot=slot, supports=P1_SUPPORTS, data_fingerprint=experience_fingerprints(supports),
                single_support_brier_mean=float(np.mean(single)),
                single_support_brier_sd=float(np.std(single, ddof=1)),
                single_support_brier_min=float(np.min(single)), single_support_brier_max=float(np.max(single)),
                averaged_forecast_brier=brier(averaged, panel),
                averaged_context_brier=brier(monotone(decode(model.decoder, fq, mean_context[name])).numpy(), panel),
                context_mean=mean_context[name][0].tolist(),
                context_sd_per_dim=contexts.std(0).tolist(),
                context_mean_sd=float(contexts.std(0).mean()),
                saved_support_contexts=saved_contexts.tolist(),
                saved_support_distance_to_fresh_mean=[float(v) for v in
                                                     (saved_contexts - mean_context[name]).norm(dim=1)],
                fresh_distance_to_fresh_mean_mean=float((contexts - mean_context[name]).norm(dim=1).mean()))
    out["P1"] = p1

    # ---- P2 context-only fit
    p2 = {}
    fitted = {}
    for name, law in (("base_a", base_a), ("base_b", base_b)):
        batches, seeds = fresh(world, law, seed, "P2", law_key(law), P2_RECORDS // BATCH)
        data_seeds[f"P2/{name}"] = seeds
        observations = torch.as_tensor(np.concatenate([b.observations for b in batches]))
        actions = torch.as_tensor(np.concatenate([b.actions for b in batches]).astype(np.int64))
        labels = torch.as_tensor(np.concatenate([b.survival for b in batches]).astype(np.float32))
        with torch.no_grad():
            latents = model.encode(observations)
        rows = torch.arange(len(actions))
        c = torch.nn.Parameter(mean_context[name].clone())
        optimizer = torch.optim.Adam([c], lr=P2_LR)
        losses = []
        for step in range(P2_STEPS + 1):
            probabilities = decode(model.decoder, latents, c)
            loss = F.binary_cross_entropy(probabilities[rows, actions].clamp(1e-6, 1 - 1e-6), labels)
            losses.append(float(loss.detach()))
            if step == P2_STEPS:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        fitted[name] = c.detach().clone()
        cstar = fitted[name]

        def source(_endpoint, evaluation, cstar=cstar):
            with torch.no_grad():
                fq = model.encode(torch.as_tensor(evaluation.observations))
                return monotone(decode(model.decoder, fq, cstar)).numpy()

        targets = iol.endpoint_targets(endpoint, source)
        p2[name] = dict(records=len(actions), steps=P2_STEPS, lr=P2_LR,
                        data_fingerprint=experience_fingerprints(batches),
                        loss_initial=losses[0], loss_final=losses[-1],
                        loss_trace_every_50=losses[::50],
                        c_init=mean_context[name][0].tolist(), c_star=cstar[0].tolist(),
                        c_shift_norm=float((cstar - mean_context[name]).norm()),
                        scored=slot_summary(targets))
    out["P2"] = p2
    out["P2"]["primary_brier_base_a"] = p2["base_a"]["scored"][0]["brier"]
    out["P2"]["control_brier_base_b"] = p2["base_b"]["scored"][1]["brier"]
    out["P2"]["specificity_base_a_c_on_base_b"] = p2["base_a"]["scored"][1]["brier"]

    # ---- P3 decoder refit
    p3 = {}
    for name, law in (("base_a", base_a), ("delay", delay_law)):
        count = P3_RECORDS // BATCH
        batches, seeds = fresh(world, law, seed, "P3", law_key(law), count + 1)  # batch 0 = history only
        data_seeds[f"P3/{name}"] = seeds
        query_features, contexts, actions, labels = [], [], [], []
        with torch.no_grad():
            for previous, current in zip(batches[:-1], batches[1:]):
                sx, sa, sy = tensors(previous)
                qx, qa, qy = tensors(current)
                features = model.encode(torch.cat((sx, qx)))
                contexts.append(context_of(model, features[:BATCH], sa, sy))
                query_features.append(features[BATCH:])
                actions.append(qa)
                labels.append(qy)
        decoder = copy.deepcopy(model.decoder)
        for parameter in decoder.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=P3_LR)
        rows = torch.arange(P3_MINIBATCH)
        order_rng = np.random.default_rng(trial_seed(seed, "retrieval_capacity_P3", [law_key(law), "order"]))

        def batch_loss(i):
            probabilities = decode(decoder, query_features[i], contexts[i])
            return F.binary_cross_entropy(probabilities[rows, actions[i]].clamp(1e-6, 1 - 1e-6), labels[i])

        with torch.no_grad():
            initial = float(np.mean([float(batch_loss(i)) for i in range(count)]))
        epoch_losses = []
        for _ in range(P3_EPOCHS):
            values = []
            for i in order_rng.permutation(count):
                loss = batch_loss(int(i))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                values.append(float(loss.detach()))
            epoch_losses.append(float(np.mean(values)))
        with torch.no_grad():
            final = float(np.mean([float(batch_loss(i)) for i in range(count)]))
        decoder.requires_grad_(False)

        def source(_endpoint, evaluation, decoder=decoder):
            with torch.no_grad():
                return forward(model, evaluation.observations, evaluation.support(), decoder=decoder)

        targets = iol.endpoint_targets(endpoint, source)
        p3[name] = dict(records=count * BATCH, epochs=P3_EPOCHS, lr=P3_LR, minibatch=P3_MINIBATCH,
                        data_fingerprint=experience_fingerprints(batches),
                        loss_before_refit=initial, loss_after_refit=final, epoch_mean_losses=epoch_losses,
                        decoder_weight_change_norm=float(torch.sqrt(sum(
                            (a - b).square().sum() for a, b in zip(decoder.parameters(),
                                                                   model.decoder.parameters())))),
                        scored=slot_summary(targets))
    p3["base_a_brier"] = p3["base_a"]["scored"][0]["brier"]
    p3["delay_law_brier"] = p3["delay"]["scored"][delay_slot]["brier"]
    p3["delay_law_cue_benefits"] = p3["delay"]["scored"][delay_slot]["cue_benefits"]
    out["P3"] = p3

    # ---- P4 linear decodability of frozen query latents
    splits = {}
    for split, count in (("fit", P4_FIT // BATCH), ("eval", P4_EVAL // BATCH)):
        seeds = [trial_seed(seed, "retrieval_capacity_P4", [split, i]) for i in range(count)]
        data_seeds[f"P4/{split}"] = seeds
        observations, labels = [], []
        for s in seeds:
            experience = world.experience(base_a, BATCH, s)
            cases = world.dataset(base_a, BATCH, s)  # evaluator-only labels for the probe
            iol.require(np.array_equal(experience.observations, cases.observations), "P4 case mismatch")
            observations.append(experience.observations)
            labels.append(np.column_stack((cases.signals[:, 0], cases.signals[:, 1], cases.signals[:, 2],
                                           cases.factors[:, 2], cases.factors[:, 1],
                                           (cases.factors[:, 0] > 0).astype(np.int64))))
        with torch.no_grad():
            latents = model.encode(torch.as_tensor(np.concatenate(observations))).double()
        splits[split] = (latents, torch.as_tensor(np.concatenate(labels)).double())
    other = world.experience(delay_law, BATCH, data_seeds["P4/fit"][0]).observations
    names = ("cue_0_bit", "cue_1_bit", "cue_2_bit", "delayed", "lossy", "source")
    out["P4"] = dict(records_fit=P4_FIT, records_eval=P4_EVAL, law="base_a", l2=P4_L2,
                     images_identical_under_delay_law=bool(np.array_equal(
                         other, world.experience(base_a, BATCH, data_seeds["P4/fit"][0]).observations)),
                     probes={name: logistic_probe(splits["fit"][0], splits["fit"][1][:, j],
                                                  splits["eval"][0], splits["eval"][1][:, j])
                             for j, name in enumerate(names)})

    # ---- seed disjointness from project seeds
    config = manifest["config"]
    project = {trial_seed(seed, "acquisition_query", 0)}
    project |= {trial_seed(seed, "acquisition_support", [rep, flag]) for rep in range(4) for flag in (False, True)}
    for phase in qualification_phases(seed, config):
        n = sum(b["size"] for b in phase["blocks"]) // BATCH
        project |= {trial_seed(seed, "representation_training", [phase["name"], i]) for i in range(n)}
    mine = [s for values in data_seeds.values() for s in values]
    out["data_seeds"] = dict(
        channels=sorted({k.split("/")[0] for k in data_seeds}), counts={k: len(v) for k, v in data_seeds.items()},
        all_unique=len(set(mine)) == len(mine), project_seeds_checked=len(project),
        overlap_with_project=len(set(mine) & project),
        sha256=hashlib.sha256(json.dumps(data_seeds, sort_keys=True).encode()).hexdigest())
    iol.require(out["data_seeds"]["all_unique"] and not out["data_seeds"]["overlap_with_project"],
                "fitting seeds are not disjoint")

    # ---- invariants after
    signature_after = learner_signature(original)
    rng_after = rng_state_hash()
    out["invariants"] = dict(signature_before=signature_before, signature_after=signature_after,
                             signature_matches_record_end=signature_before == record["end_signature"],
                             signature_unchanged=signature_before == signature_after,
                             rng_before=rng_before, rng_after=rng_after, rng_unchanged=rng_before == rng_after,
                             checkpoint_sha256=iol.sha(folder / "checkpoint.pt"))
    iol.require(signature_before == signature_after and rng_before == rng_after, "state changed")
    out["seconds"] = time.perf_counter() - started
    return out


# ---------------------------------------------------------------- summary

def closed(S, value, E):
    return (S - value) / (S - E)


def summarize(rows, ideal):
    cells = {c["name"]: c for c in ideal["cells"]}
    table = []
    for r in rows:
        seed, key = r["seed"], str(r["seed"])
        stage = r["delay_stage"]
        o_cue1 = cells[f"interleaved/stage_{stage}/cue_1/cue_benefit"]["O"]["per_seed"][key]
        i_cue1 = cells[f"interleaved/stage_{stage}/cue_1/cue_benefit"]["I"]["per_seed"][key]
        saved_delay = r["saved"][r["delay_slot"]]
        E = r["E_base_a"]["cycles_2_8"]
        row = dict(
            seed=seed, S=r["S_base_a"], E=E, E_cycle_8=r["E_base_a"]["cycle_8"],
            E_all_cycles=r["E_base_a"]["all_cycles"],
            O=cells["interleaved/slot_0/absolute_brier"]["O"]["per_seed"][key],
            I=cells["interleaved/slot_0/absolute_brier"]["I"]["per_seed"][key],
            P1_single_mean=r["P1"]["base_a"]["single_support_brier_mean"],
            P1_avg_forecast=r["P1"]["base_a"]["averaged_forecast_brier"],
            P1_avg_context=r["P1"]["base_a"]["averaged_context_brier"],
            P2_c_star=r["P2"]["primary_brier_base_a"],
            P2_closed=closed(r["S_base_a"], r["P2"]["primary_brier_base_a"], E),
            P2_closed_E_cycle_8=closed(r["S_base_a"], r["P2"]["primary_brier_base_a"], r["E_base_a"]["cycle_8"]),
            P2_closed_E_all_cycles=closed(r["S_base_a"], r["P2"]["primary_brier_base_a"],
                                          r["E_base_a"]["all_cycles"]),
            S_B=r["S_base_b"], P2_control_B=r["P2"]["control_brier_base_b"],
            P2_control_diff=r["P2"]["control_brier_base_b"] - r["S_base_b"],
            P2_specificity_A_on_B=r["P2"]["specificity_base_a_c_on_base_b"],
            P3_base_a=r["P3"]["base_a_brier"],
            P3_closed=closed(r["S_base_a"], r["P3"]["base_a_brier"], E),
            delay_stage=stage, delay_S=saved_delay["brier"], delay_E=r["E_delay_law"]["cycles_2_8"],
            delay_P3=r["P3"]["delay_law_brier"],
            delay_cue1_saved=saved_delay["cue_benefits"]["1"],
            delay_cue1_P3=r["P3"]["delay_law_cue_benefits"]["1"], delay_cue1_O=o_cue1, delay_cue1_I=i_cue1,
            delay_cue1_P3_fraction_of_O=r["P3"]["delay_law_cue_benefits"]["1"] / o_cue1,
            delay_other_cues_saved={k: v for k, v in saved_delay["cue_benefits"].items() if k != "1"},
            delay_other_cues_P3={k: v for k, v in r["P3"]["delay_law_cue_benefits"].items() if k != "1"},
            P4_cue1_accuracy=r["P4"]["probes"]["cue_1_bit"]["heldout_accuracy"],
            P4={k: v["heldout_accuracy"] for k, v in r["P4"]["probes"].items()})
        table.append(row)
    fractions = np.asarray([t["P2_closed"] for t in table])
    control_ok = [abs(t["P2_control_diff"]) <= .02 for t in table]
    criteria = dict(seed_mean_closed_fraction=float(fractions.mean()),
                    mean_at_least_0_75=bool(fractions.mean() >= .75),
                    seeds_closing_at_least_0_5=int((fractions >= .5).sum()),
                    at_least_5_of_6=bool((fractions >= .5).sum() >= 5),
                    base_b_control_within_0_02_each_seed=control_ok,
                    base_b_control_within_0_02_seed_mean=bool(abs(np.mean([t["P2_control_diff"] for t in table])) <= .02),
                    base_b_control_all_seeds=bool(all(control_ok)))
    present = criteria["mean_at_least_0_75"] and criteria["at_least_5_of_6"] and criteria["base_b_control_all_seeds"]
    p3_fraction = np.asarray([t["P3_closed"] for t in table])
    sensitivity = {name: dict(seed_mean=float(np.mean([t[key] for t in table])),
                              seeds_at_least_0_5=int(sum(t[key] >= .5 for t in table)))
                   for name, key in (("E_cycle_8", "P2_closed_E_cycle_8"), ("E_all_cycles", "P2_closed_E_all_cycles"))}
    for t in table:  # post hoc sensitivity: remove the generic c-fit gain seen on Base B
        t["P2_closed_control_adjusted"] = (t["S"] - t["P2_c_star"] - (t["S_B"] - t["P2_control_B"])) / (t["S"] - t["E"])
    adjusted = np.asarray([t["P2_closed_control_adjusted"] for t in table])
    diffs = np.asarray([t["P2_control_diff"] for t in table])
    base = criteria["mean_at_least_0_75"] and criteria["at_least_5_of_6"]
    readings = dict(
        every_seed_two_sided=bool(base and np.all(np.abs(diffs) <= .02)),
        seed_mean_two_sided=bool(base and abs(diffs.mean()) <= .02),
        every_seed_one_sided_not_worse=bool(base and np.all(diffs <= .02)))
    means = {k: float(np.mean([t[k] for t in table])) for k in table[0]
             if isinstance(table[0][k], float)}
    return dict(per_seed=table, seed_means=means, criteria=criteria,
                verdict="PRESENT" if present else "NOT PRESENT",
                verdict_rule_reading="control checked per seed, two-sided |diff| <= .02 (coded before the full run)",
                verdict_under_alternative_readings={k: "PRESENT" if v else "NOT PRESENT" for k, v in readings.items()},
                control_adjusted_closed_fraction_post_hoc=dict(
                    seed_mean=float(adjusted.mean()), seeds_at_least_0_5=int((adjusted >= .5).sum()),
                    per_seed=adjusted.tolist()),
                P3_closed_fraction=dict(seed_mean=float(p3_fraction.mean()),
                                        seeds_at_least_0_5=int((p3_fraction >= .5).sum()),
                                        per_seed=p3_fraction.tolist()),
                E_sensitivity_for_P2=sensitivity)


def main():
    started = time.perf_counter()
    rng_before = rng_state_hash()
    OUT.mkdir(parents=True, exist_ok=True)
    source_lock = iol.source_lock_check(RUN)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(run_seed, SEEDS))
    ideal = iol.read_json(IDEAL)
    summary = summarize(rows, ideal)
    result = dict(
        label=LABEL, protocol="docs/retrieval_capacity_protocol.md",
        protocol_sha256=iol.sha(ROOT / "docs" / "retrieval_capacity_protocol.md"),
        script_sha256=iol.sha(Path(__file__)), run="runs/representation_learning_qualification",
        seeds=list(SEEDS),
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         platform=platform.platform(), workers=WORKERS, torch_threads_per_worker=1,
                         deterministic_algorithms=True, device="cpu"),
        settings=dict(batch=BATCH, P1_supports=P1_SUPPORTS, P2_records=P2_RECORDS, P2_steps=P2_STEPS,
                      P2_lr=P2_LR, P2_init="mean context of the 64 fresh P1 supports of the fitted law",
                      P2_loss="BCE (clamped 1e-6) on the performed action's 3 horizons, no cummin in training",
                      P3_records=P3_RECORDS, P3_epochs=P3_EPOCHS, P3_lr=P3_LR, P3_minibatch=P3_MINIBATCH,
                      P3_details="fresh Adam on decoder only (from current weights), no gradient clipping; "
                                 "minibatch = one fresh 32-record batch with context from its preceding fresh "
                                 "batch; batch order reshuffled each epoch with a trial_seed-derived RNG",
                      P4_fit=P4_FIT, P4_eval=P4_EVAL, P4_l2=P4_L2, P4_optimizer="LBFGS strong-Wolfe, float64",
                      P4_law="base_a (images are law-independent; checked)",
                      scoring="ideal_observer_load.endpoint_targets / locked_score (cummin forecasts, "
                              "saved panels, saved supports, replicate-averaged)",
                      E_definition="mean all-case brier of the result.json curve entries for Base A "
                                   "(block-end, support = learner.history, i.e. the last Base A batch) in "
                                   "cycles 2-8; cycle 8 alone and all 8 cycles reported as sensitivity",
                      closed_fraction="(S - Brier(c*)) / (S - E) per seed",
                      control_rule_reading="|Brier(c*_B on Base B) - S_B| <= .02 for every seed"),
        source_lock_all_match=all(v["matches"] for v in source_lock.values()),
        summary=summary, per_seed=rows,
        invariants=dict(all_signatures_unchanged=all(r["invariants"]["signature_unchanged"] for r in rows),
                        all_worker_rng_unchanged=all(r["invariants"]["rng_unchanged"] for r in rows),
                        main_rng_unchanged=rng_before == rng_state_hash(),
                        forward_reproduction=dict(
                            predict_all_exact_all_seeds=all(r["verification"]["predict_all_exact"] for r in rows),
                            joint_exact_all_seeds=all(r["verification"]["joint_exact_all"] for r in rows),
                            base_a_joint_exact_all_seeds=all(r["verification"]["base_a_joint_exact"] for r in rows),
                            joint_max_abs=max(r["verification"]["joint_max_abs"] for r in rows),
                            split_max_abs=max(r["verification"]["split_max_abs"] for r in rows))),
        runtime_seconds=time.perf_counter() - started)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = [k for k, v in summary["per_seed"][0].items() if not isinstance(v, dict)]
    with (OUT / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in summary["per_seed"]:
            writer.writerow(row)
    print(json.dumps(dict(verdict=summary["verdict"], criteria=summary["criteria"],
                          P3=summary["P3_closed_fraction"], invariants=result["invariants"],
                          runtime_seconds=result["runtime_seconds"]), indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
