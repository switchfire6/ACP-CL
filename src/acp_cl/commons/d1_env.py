"""Step D1 evaluator-side environment: the accepted D0b world, stationary training
streams with common random numbers, the greedy-probe panels and O's tables.

Nothing here is visible to a learner except through ``learners.make_decision``
and ``learners.make_outcome`` (the legal observations).

Randomness (all keys through ``world.seed_for``):
* training stream of (seed, context): ``sample_episodes(N, seed_for("commons_d1_stream", seed, context))``;
  the same draws serve every arm (common random numbers across arms);
* panel of a context (seed-independent, shared by every seed and arm):
  ``sample_episodes(512, seed_for("commons_d1_panel", context))``; MB's fixed
  planning draws for the panel: ``seed_for("commons_d1_panel_plan", context)``;
* O's tables: 2^14 draws per decision state, independent draws for action
  selection (``..._O_select``) and scoring (``..._O_score``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import references as R
from .learners import make_decision, make_outcome
from .world import (N_CONTEXTS, T, WorldParams, context_index, n_actions, observe, sample_episodes, seed_for,
                    simulate, supply_means, SUPPLY_NOISE, OBS_NOISE)

WORLD_KEY = "1586f04b8d3cc8c2"
TRAINING_CONTEXTS = ("A0B0C0", "A1B0C0", "A0B1C0", "A0B0C1", "A1B1C1")
FACTOR_BITS = {"A": 4, "B": 2, "C": 1}


def context_of(name):
    if len(name) != 6 or name[0] != "A" or name[2] != "B" or name[4] != "C":
        raise ValueError(f"bad context name {name}")
    return context_index(int(name[1]), int(name[3]), int(name[5]))


def cfg_key(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def make_params(cfg):
    cfg = dict(cfg)
    for key in ("base_signs", "amounts", "c_channels"):
        if key in cfg:
            cfg[key] = tuple(cfg[key])
    return WorldParams(**cfg)


def load_world(root, separability="reports/commons_d0b/separability.json",
               spec="reports/commons_d0b/world_spec.json", key=WORLD_KEY):
    """The accepted D0b world; verified against the stored key and the world spec's params digest."""
    root = Path(root)
    cfg = json.loads((root / separability).read_text(encoding="utf-8"))["chosen_cfg"]
    if cfg_key(cfg) != key:
        raise ValueError(f"world key mismatch: {cfg_key(cfg)} != {key}")
    params = make_params(cfg)
    stored = json.loads((root / spec).read_text(encoding="utf-8"))
    if stored["params_sha256"] != params.digest():
        raise ValueError("world_spec params digest mismatch")
    if n_actions(params) != 13:
        raise ValueError("the D0b world has 13 actions")
    return params, cfg


class Stream:
    """A stationary stream of N episodes in one context; outcomes for every action are
    precomputed from the same draws (the learner only ever receives the chosen one)."""

    def __init__(self, params, context, seed, n, channel="commons_d1_stream"):
        self.context, self.seed, self.n = int(context), int(seed), int(n)
        self.episodes = sample_episodes(n, seed_for(channel, seed, self.context))
        ctx = np.full(n, self.context)
        self.supplies = self.episodes.supplies(params, ctx)
        self.y_supply, self.y_probe = observe(params, ctx, self.episodes, self.supplies, probes_on=True)
        self.fail = np.stack([simulate(params, ctx, self.episodes, np.full(n, a), self.supplies).fail_step
                              for a in range(n_actions(params))], 1)                       # (n, 13, 3)

    def decision(self, k):
        return make_decision(self.episodes.reserves[k], self.episodes.cue[k])

    def outcome(self, k, action):
        return make_outcome(self.y_supply[k], self.y_probe[k], self.fail[k, int(action)])

    def fingerprint(self):
        return self.episodes.fingerprint()


class Panel:
    """512 decision states of a context (seed-independent), their outcomes under all 13 actions
    (for evaluation only) and MB's fixed planning draws."""

    def __init__(self, params, context, size, plan_samples, with_draws=True):
        self.context, self.size = int(context), int(size)
        self.episodes = sample_episodes(size, seed_for("commons_d1_panel", self.context))
        ctx = np.full(size, self.context)
        supplies = self.episodes.supplies(params, ctx)
        self.y_supply, _ = observe(params, ctx, self.episodes, supplies, probes_on=True)
        self.fail = np.stack([simulate(params, ctx, self.episodes, np.full(size, a), supplies).fail_step
                              for a in range(n_actions(params))], 1)
        self.reserves = self.episodes.reserves
        self.cue = self.episodes.cue
        self.eps = None
        if with_draws:
            rng = np.random.default_rng(seed_for("commons_d1_panel_plan", self.context))
            self.eps = rng.standard_normal((size, plan_samples, T, 3), dtype=np.float32)


def o_values(params, reserves, cue, contexts, keys, m):
    """O's MC table per decision state: (n, len(contexts), 13) values and SEs; draws seeded by keys + (e,)."""
    n = len(reserves)
    value = np.empty((n, len(contexts), n_actions(params)))
    se = np.empty_like(value)
    for e in range(n):
        z = R.draws(seed_for(*keys, e), m)
        value[e], se[e] = R.episode_values(params, reserves[e], int(cue[e, 0]), list(contexts), z)
    return value, se


def panel_tables(params, context, size, m, floor_m):
    """O's selection and scoring tables on the panel for all 8 contexts, and the M=floor_m
    true-model planner's selection values in the panel's own context."""
    panel = Panel(params, context, size, 0, with_draws=False)
    contexts = list(range(N_CONTEXTS))
    select, select_se = o_values(params, panel.reserves, panel.cue, contexts, ("commons_d1_panel_O_select", context), m)
    score, score_se = o_values(params, panel.reserves, panel.cue, contexts, ("commons_d1_panel_O_score", context), m)
    floor, _ = o_values(params, panel.reserves, panel.cue, [context], ("commons_d1_panel_floor", context), floor_m)
    return dict(reserves=panel.reserves, cue=panel.cue, select=select, select_se=select_se, score=score,
                score_se=score_se, floor_select=floor[:, 0], fail=panel.fail)


def stream_tables(params, context, seed, n, start, stop, m):
    """O's selection and scoring tables (true context only) for stream episodes start..stop-1."""
    ep = sample_episodes(n, seed_for("commons_d1_stream", seed, int(context)))
    idx = np.arange(start, stop)
    sel = np.empty((len(idx), n_actions(params)))
    sel_se, sco, sco_se = np.empty_like(sel), np.empty_like(sel), np.empty_like(sel)
    for k, e in enumerate(idx):
        z = R.draws(seed_for("commons_d1_O_select", seed, int(context), int(e)), m)
        v, s = R.episode_values(params, ep.reserves[e], int(ep.cue[e, 0]), [int(context)], z)
        sel[k], sel_se[k] = v[0], s[0]
        z = R.draws(seed_for("commons_d1_O_score", seed, int(context), int(e)), m)
        v, s = R.episode_values(params, ep.reserves[e], int(ep.cue[e, 0]), [int(context)], z)
        sco[k], sco_se[k] = v[0], s[0]
    return dict(select=sel, select_se=sel_se, score=sco, score_se=sco_se, index=idx,
                reserves=ep.reserves[idx], cue=ep.cue[idx])


def true_sensor_moments(params, context):
    """True mean (12, 3) and variance of the supply sensors in a context (for MB calibration)."""
    a = (int(context) >> 2) & 1
    return supply_means(params)[a], SUPPLY_NOISE ** 2 + OBS_NOISE ** 2
