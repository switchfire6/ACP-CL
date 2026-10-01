"""Step D2w evaluator-side observation model with adjustable sensor noise (docs/commons_d2w_protocol.md).

The physics, contexts, O's value tables and the run-length filter are D0b's, imported unchanged. Only what depends on
the SENSOR noise is re-implemented here with the noise as a parameter (D0b's references hard-code sigma_obs = .1):

* supply sensors   y_s = s + sigma_s * z_obs_supply   (latent supply s = mu(A, t) + .35 z_supply, unchanged);
* probe sensors    y_p = probe arrivals + sigma_p * z_obs_probe, or pure noise when probes are off;
* the exact Gaussian sensor log-likelihoods, and the failure-pattern log-likelihood E[1[pattern] | sensors, c] by
  conjugate-Gaussian posterior sampling of the latent supplies (posterior variance 1 / (1/.35^2 + 1/sigma_s^2)),
  a line-for-line copy of references.pattern_histograms / pattern_loglik with the variance as a parameter.

With sigma_s = sigma_p = .1 and probes on, every function reproduces D0b's references bit for bit (checked in the
D2w smoke check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import references as R
from .world import (N_CONTEXTS, N_ENTITIES, SUPPLY_NOISE, SURVIVED, T, channel_params, entity_kernel, factors,
                    probe_arrivals, supply_means)

LOG_2PI = float(np.log(2 * np.pi))


@dataclass(frozen=True)
class Noise:
    sigma_s: float = 0.1
    sigma_p: float = 0.1
    probes_on: bool = True

    @property
    def marg_var(self):
        return SUPPLY_NOISE ** 2 + self.sigma_s ** 2

    @property
    def post_var(self):
        return 1.0 / (1.0 / SUPPLY_NOISE ** 2 + 1.0 / self.sigma_s ** 2)

    def to_json(self):
        return dict(sigma_s=float(self.sigma_s), sigma_p=float(self.sigma_p), probes_on=bool(self.probes_on))


def observe(params, context, episodes, supplies, noise: Noise):
    n = len(episodes)
    context = np.broadcast_to(np.asarray(context), (n,))
    eff, delay = channel_params(params, context, episodes.cue[:, 0])
    y_supply = supplies + noise.sigma_s * episodes.z_obs_supply
    y_probe = probe_arrivals(params, eff, delay, noise.probes_on) + noise.sigma_p * episodes.z_obs_probe
    return y_supply, y_probe


def supply_loglik(params, y_supply, noise: Noise):
    """log p(y_supply | A) for A in {0, 1}: (n, 2)."""
    mu = supply_means(params)
    resid = y_supply[:, None] - mu[None]
    per = -0.5 * (resid ** 2 / noise.marg_var + np.log(noise.marg_var) + LOG_2PI)
    return per.sum(axis=(2, 3))


def probe_loglik(params, y_probe, cue1, noise: Noise):
    """log p(y_probe | c) for all 8 contexts: (n, 8)."""
    n = len(y_probe)
    out = np.empty((n, N_CONTEXTS))
    var = noise.sigma_p ** 2
    for c in range(N_CONTEXTS):
        eff, delay = channel_params(params, np.full(n, c), cue1)
        mean = probe_arrivals(params, eff, delay, noise.probes_on)
        out[:, c] = (-0.5 * ((y_probe - mean) ** 2 / var + np.log(var) + LOG_2PI)).sum(axis=(1, 2))
    return out


def posterior_mean(params, y_entity, a_level, entity, noise: Noise):
    mu = supply_means(params)[a_level, :, entity]
    return noise.post_var * (mu / SUPPLY_NOISE ** 2 + y_entity / noise.sigma_s ** 2)


def pattern_loglik(params, reserves, cue1, y_supply, fail_steps, contexts, actions, seed, noise: Noise,
                   settings=R.PatternSettings(), stats=None):
    """log P(failure pattern | sensors, c, a) for one episode, (len(actions), len(contexts)); references.pattern_loglik
    with the posterior variance of the latent supplies taken from ``noise``."""
    variants, index = R.entity_variants(params, reserves, cue1, contexts, actions)
    rng = np.random.default_rng(seed)
    sd = np.sqrt(noise.post_var)
    hists, ks = [], []
    for i in range(N_ENTITIES):
        a_levels, x0, arr, has_arr = R._variant_arrays(variants[i])
        u = rng.standard_normal((settings.k, T))
        means = np.stack([posterior_mean(params, y_supply[:, i], a, i, noise) for a in (0, 1)])
        supplies = means[a_levels][:, None, :] + sd * u[None]
        step = entity_kernel(x0[:, None], arr[:, None, :] if has_arr else None, supplies, want_step=True)
        hist = np.zeros((len(variants[i]), SURVIVED + 1), dtype=np.int64)
        for j in range(len(variants[i])):
            hist[j] = np.bincount(step[j], minlength=SURVIVED + 1)
        hists.append(hist)
        ks.append(np.full(len(variants[i]), settings.k))
    for i in range(N_ENTITIES):
        need = set()
        for a in actions:
            obs = int(fail_steps[a][i])
            for c in contexts:
                j = index[(c, a)][i]
                if hists[i][j, obs] < settings.min_count and ks[i][j] == settings.k:
                    need.add(j)
        if need:
            need = sorted(need)
            sub = [variants[i][j] for j in need]
            a_levels, x0, arr, has_arr = R._variant_arrays(sub)
            u = rng.standard_normal((settings.k_refine, T))
            means = np.stack([posterior_mean(params, y_supply[:, i], a, i, noise) for a in (0, 1)])
            supplies = means[a_levels][:, None, :] + sd * u[None]
            step = entity_kernel(x0[:, None], arr[:, None, :] if has_arr else None, supplies, want_step=True)
            for r, j in enumerate(need):
                hists[i][j] = np.bincount(step[r], minlength=SURVIVED + 1)
                ks[i][j] = settings.k_refine
            if stats is not None:
                stats["refined"] = stats.get("refined", 0) + len(need)
    out = np.zeros((len(actions), len(contexts)))
    for ai, a in enumerate(actions):
        for ci, c in enumerate(contexts):
            ids = index[(c, a)]
            for i in range(N_ENTITIES):
                obs = int(fail_steps[a][i])
                hits = hists[i][ids[i], obs]
                k = ks[i][ids[i]]
                if hits == 0:
                    if stats is not None:
                        stats["floored"] = stats.get("floored", 0) + 1
                    p = settings.floor_hits / k
                else:
                    p = hits / k
                out[ai, ci] += np.log(p)
    if stats is not None:
        stats["entries"] = stats.get("entries", 0) + len(actions) * len(contexts) * N_ENTITIES
    return out


def sensor_loglik(params, y_supply, y_probe, cue1, noise: Noise):
    """Exact sensor log-likelihood for all 8 contexts: (n, 8)."""
    sup = supply_loglik(params, y_supply, noise)
    a_of_c = np.array([int(factors(c)[0]) for c in range(N_CONTEXTS)])
    return sup[:, a_of_c] + probe_loglik(params, y_probe, cue1, noise)
