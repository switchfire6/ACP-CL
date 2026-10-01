"""Fast checks of the Step D0 commons world, references and schedule (the full
verification lives in scripts/commons_d0.py verify)."""

import numpy as np

from acp_cl.commons import calibration as C
from acp_cl.commons import references as R
from acp_cl.commons import verification as V
from acp_cl.commons.world import (N_ACTIONS, SURVIVED, Episodes, WorldParams, observe,
                                  sample_episodes, seed_for, simulate)

PARAMS = WorldParams(base_signs=(1.0, -1.0, -0.5), b_gap=0.6, c_delta=6.0)


def test_simulator_matches_slow_reference_and_conserves():
    result = V.check_simulator(n=150, seed=21)
    assert result["pass"], result


def test_joint_survival_is_all_entities_alive():
    ep = sample_episodes(500, 3)
    ep.reserves[:100] = 0.5
    res = simulate(PARAMS, np.zeros(500, dtype=int), ep, np.zeros(500, dtype=int))
    assert np.array_equal(res.joint_survival, (res.fail_step == SURVIVED).all(axis=1))
    assert np.array_equal(res.horizon_survival[:, -1], res.joint_survival)
    assert not res.joint_survival[:100].any()


def test_probe_level_never_changes_physics():
    ep = sample_episodes(200, 4)
    ctx = np.arange(200) % 8
    act = np.arange(200) % N_ACTIONS
    on = simulate(PARAMS, ctx, ep, act, probes_on=True)
    off = simulate(PARAMS, ctx, ep, act, probes_on=False)
    assert np.array_equal(on.fail_step, off.fail_step)
    assert np.array_equal(on.final_stock, off.final_stock)


def test_value_table_joint_indicator_equals_full_simulator():
    ep = sample_episodes(2, 5)
    z = R.draws(8, 256)
    _, _, joint = R.episode_values(PARAMS, ep.reserves[0], int(ep.cue[0, 0]), list(range(8)), z, joint=True)
    rep = Episodes(np.repeat(ep.reserves[:1], 256, 0), np.repeat(ep.cue[:1], 256, 0), z, np.zeros_like(z),
                   np.zeros_like(z))
    for c in (0, 3, 6):
        for a in range(N_ACTIONS):
            frac = simulate(PARAMS, np.full(256, c), rep, np.full(256, a)).joint_survival.mean()
            assert frac == joint[c, a]


def test_value_tables_deterministic_and_draw_sets_independent():
    ep = sample_episodes(3, 6)
    a = R.value_tables(PARAMS, ep, [0, 5], 512, 1, "select").value
    b = R.value_tables(PARAMS, ep, [0, 5], 512, 1, "select").value
    c = R.value_tables(PARAMS, ep, [0, 5], 512, 1, "score").value
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_filter_matches_bruteforce_enumeration():
    assert V.check_filter(seed=3)["pass"]


def test_likelihood_true_context_is_typical_and_sensor_term_exact():
    ep = sample_episodes(4, 7)
    ctx = np.full(4, 6)
    sup = ep.supplies(PARAMS, ctx)
    fs = np.stack([simulate(PARAMS, ctx, ep, np.full(4, a), sup).fail_step for a in range(N_ACTIONS)], 1)
    ys, yp = observe(PARAMS, ctx, ep, sup)
    sensor, pattern = R.episode_loglik(PARAMS, ep, ys, yp, fs, [0, 1], 2)
    total = sensor[:, None, :] + pattern
    # The true context is never excluded, and A/B (probe + supply sensors) are decisive.
    assert np.all(np.isfinite(total))
    assert np.all(total[:, :, 6] >= total[:, :, [0, 1, 2, 3]].max(-1))
    assert np.all(total[:, :, 6] >= total[:, :, [4, 5]].max(-1))


def test_schedule_constraints():
    ctx, order, info = C.generate_schedule(2)
    stats = C.schedule_statistics(ctx, order, info)
    assert stats["episodes"] == 20000
    assert stats["heldout_only_in_last_quarter"]
    assert stats["minority"]["recurrences"] >= 4
    assert all(c1 != c2 for (c1, _), (c2, _) in zip(order, order[1:]))
    assert abs(stats["per_context"]["7"]["share"] - 0.06) < 0.005
    again, _, _ = C.generate_schedule(2)
    assert np.array_equal(ctx, again)


def test_seed_for_is_stable():
    assert seed_for(1, "a", 2) == seed_for(1, "a", 2) != seed_for(1, "a", 3)
