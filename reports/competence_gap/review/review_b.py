"""Reviewer check B: independent Monte Carlo of the law-known oracle O on the endpoint panel.

Own supply-noise sampler (TransferWorld.cases means + N(0, .35), clip at 0, then the
ConditionalWorld mode column swap), then the project physics AcquisitionWorld.simulate
(which applies the cue-2 timing shift and re-clip inside physics). Does not import
scripts/ideal_observer_mc.py.
"""
import sys
import json
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from acp_cl.acquisition.world import AcquisitionWorld, LawCases, stage_law
from acp_cl.conditional.world import GRID
from acp_cl.persistence.study import trial_seed
from acp_cl.persistence.world import Cases, TransferWorld

RUN = ROOT / 'runs' / 'competence_gap'
world = AcquisitionWorld()
K, CHUNK = 8192, 256
SEEDS = [int(s) for s in sys.argv[1:]] or [19101, 19102]


def my_means(source):
    season = np.where(np.arange(12) < 6, 1., -1.)
    return 1.9 + 0.8 * source[:, None, None] * season[None, :, None] * np.array([1., -1.])[None, None, :]


def reconstruct_supplies(law, seedq, size=512):
    """Replay the evaluator's RNG to rebuild realised supplies with MY mean formula."""
    parts, facts = [], []
    for index, (source, lossy, delayed) in enumerate(GRID):
        rng = np.random.default_rng(trial_seed(seedq, 'grid', index))
        n = size // 8
        rng.uniform(1.0, 13.0, (n, 2))
        noise = rng.normal(0, 0.35, (n, 12, 2))
        parts.append(np.clip(my_means(np.full(n, float(source))) + noise, 0, None))
    sup = np.concatenate(parts)
    if law.mode == 1:
        sup = sup[:, :, ::-1]
    order = np.random.default_rng(trial_seed(seedq, 'order', 0)).permutation(size)
    return sup[order]


def simulate(reserves, factors, signals, supplies, law, action):
    n = len(reserves)
    eff = np.where(factors[:, 1] == 1, 0.35, 1.0)
    dly = np.where(factors[:, 2] == 1, 2, 0).astype(int)
    dummy = np.zeros((n, 1), np.uint8)
    lc = LawCases(dummy, Cases(dummy, reserves, supplies, eff, dly), factors, signals, law)
    return world.simulate(lc, np.full(n, action, dtype=np.int64)).survival


report = {}
for seed in SEEDS:
    law = stage_law(seed, 3)
    seedq = trial_seed(seed, 'acquisition_query', 0)
    cases = world.dataset(law, 512, seedq)
    truth = world.counterfactuals(cases)
    recon = reconstruct_supplies(law, seedq)
    recon_exact = bool(np.array_equal(recon, cases.base.supplies))
    # realised-supply check: my simulate() wrapper with realised supplies reproduces truth
    realised_ok = all(np.array_equal(simulate(cases.base.reserves, cases.factors, cases.signals,
                                              cases.base.supplies, law, a), truth[:, a]) for a in range(5))
    n = len(cases.factors)
    counts = np.zeros((n, 5, 3), np.int64)
    rng = np.random.default_rng([777, seed, 20260929])
    means = my_means(cases.factors[:, 0].astype(np.float64))
    reserves = np.tile(cases.base.reserves, (CHUNK, 1))
    factors = np.tile(cases.factors, (CHUNK, 1))
    signals = np.tile(cases.signals, (CHUNK, 1))
    for _ in range(K // CHUNK):
        sup = np.clip(means[None] + rng.normal(0, TransferWorld.supply_noise, (CHUNK, n, 12, 2)), 0, None)
        if law.mode == 1:
            sup = sup[..., ::-1]
        sup = np.ascontiguousarray(sup.reshape(CHUNK * n, 12, 2))
        for a in range(5):
            counts[:, a] += simulate(reserves, factors, signals, sup, law, a).reshape(CHUNK, n, 3).sum(axis=0, dtype=np.int64)
    mine = counts / K
    with np.load(RUN / 'oracle' / f'seed_{seed}.npz') as z:
        saved = z['endpoint_exact'][0]
        ksaved = z['endpoint_exact_batch_counts'].shape[0] * 512
    se = np.sqrt(saved * (1 - saved) / ksaved + mine * (1 - mine) / K)
    live = se > 0
    zs = (saved - mine)[live] / se[live]
    b = lambda p: float(((np.minimum.accumulate(p, axis=-1) - truth) ** 2).mean())  # noqa: E731
    report[str(seed)] = dict(mode=law.mode, queries=n, K_mine=K, K_saved=int(ksaved),
                             reconstructed_realised_supplies_exact=recon_exact,
                             realised_supplies_reproduce_truth=bool(realised_ok),
                             brier_saved=b(saved), brier_mine=b(mine), brier_diff=b(saved) - b(mine),
                             mean_z2=float((zs ** 2).mean()), max_abs_z=float(np.abs(zs).max()),
                             cells_abs_z_gt4=int((np.abs(zs) > 4).sum()), live_cells=int(live.sum()),
                             deterministic_cells_disagree=int((np.abs(saved - mine)[~live] > 0).sum()),
                             max_abs_prob_diff=float(np.abs(saved - mine).max()))
    print(json.dumps({seed: report[str(seed)]}), flush=True)
(ROOT / 'reports/competence_gap/review' / f"review_b_{'_'.join(map(str, SEEDS))}.json").write_text(json.dumps(report, indent=1))
