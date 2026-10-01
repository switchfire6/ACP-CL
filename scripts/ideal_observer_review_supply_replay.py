"""ADVERSARIAL REVIEW scratch check: exact RNG replay of the base supply draw.

POST HOC / EXPLORATORY / NOT DECISION-BEARING. Rebuilds every query/support base supply
array of the qualification from raw PCG64 draws with the formula used by the MC
(ideal_observer_mc.supply_means + N(0,.35), clip at 0, then mode column swap, then the
ConditionalWorld row permutation) and checks byte equality with the evaluator cases.
Also checks the saved learner forecasts are already horizon-monotone (so the cummin in
the audit's forecasts() is a no-op for the learner).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import ideal_observer_mc as MC  # noqa: E402
from acp_cl.acquisition.world import AcquisitionWorld  # noqa: E402
from acp_cl.conditional.world import GRID  # noqa: E402
from acp_cl.persistence.study import trial_seed  # noqa: E402
from acp_cl.representation_learning.design import reference_laws  # noqa: E402


def replay(data_seed, size, mode):
    parts_sup, parts_res = [], []
    for index, (source, lossy, delayed) in enumerate(GRID):
        rng = np.random.default_rng(trial_seed(data_seed, "grid", index))
        reserves = rng.uniform(1.0, 13.0, (size // 8, 2))
        noise = rng.normal(0, .35, (size // 8, 12, 2))
        factors = np.tile(np.array([[source, lossy, delayed]]), (size // 8, 1))
        parts_sup.append(np.clip(MC.supply_means(factors) + noise, 0, None))
        parts_res.append(reserves)
    sup, res = np.concatenate(parts_sup), np.concatenate(parts_res)
    sup = MC.mode_frame(sup, mode)
    order = np.random.default_rng(trial_seed(data_seed, "order", 0)).permutation(size)
    return sup[order], res[order]


def main():
    world = AcquisitionWorld()
    checked = 0
    for seed in range(18101, 18107):
        for law in reference_laws(seed):
            for data_seed, size in [(trial_seed(seed, "acquisition_query", 0), 512)] + [
                    (trial_seed(seed, "acquisition_support", [rep, False]), 32) for rep in (0, 1)]:
                cases = world.dataset(law, size, data_seed)
                sup, res = replay(data_seed, size, law.mode)
                assert np.array_equal(sup, cases.base.supplies), (seed, law, size)
                assert np.array_equal(res, cases.base.reserves), (seed, law, size)
                checked += 1
    print("supply/reserve arrays reproduced exactly from raw draws with the MC formula:", checked)
    import ideal_observer_load as L
    endpoints = L.load_all(verify=False)
    worst = 0.
    for ep in endpoints.values():
        for ev in ep.evaluations:
            p = ev.probabilities.astype(np.float64)
            worst = max(worst, float(np.abs(np.minimum.accumulate(p, axis=-1) - p).max()))
    print("max |cummin(learner) - learner| over all saved endpoint forecasts:", worst)


if __name__ == "__main__":
    main()
