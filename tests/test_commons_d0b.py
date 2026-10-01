"""Step D0b world options: defaults reproduce D0; new modes match the slow reference."""

import numpy as np

from acp_cl.commons import references as R
from acp_cl.commons import verification as V
from acp_cl.commons.world import (ACTION_CHANNEL, ACTION_DONOR, ACTION_RECIPIENT, WorldParams,
                                  action_table, channel_params, n_actions)

D0B = WorldParams(base_signs=(1.0, -1.0, -1.0), probe_start="steady", flip_amplitude=0.5, b_gap=0.8,
                  c_delta=6.0, amounts=(3.0, 6.0), imbalance=0.8, b_mode="swap", c_mode="ch23", e31=0.9)


def test_defaults_reproduce_d0_action_table_and_digest_keys():
    ch, donor, rec, amount = action_table(WorldParams())
    assert np.array_equal(ch, ACTION_CHANNEL) and np.array_equal(donor, ACTION_DONOR)
    assert np.array_equal(rec, ACTION_RECIPIENT) and np.all(amount[1:] == 3.0)
    assert n_actions(WorldParams()) == 7
    assert "amounts" not in WorldParams().to_json() and "b_mode" not in WorldParams().to_json()
    assert n_actions(D0B) == 13 and D0B.to_json()["amounts"] == [3.0, 6.0]


def test_new_modes_match_slow_reference():
    sets = [D0B,
            D0B.with_dials(b_mode="both", c_mode="same", c_channels=(1, 2), d31=0.0),
            D0B.with_dials(c_mode="route", c_channels=(1, 2), d31=0.0, amounts=(2.0, 5.0, 8.0))]
    result = V.check_simulator(n=120, seed=4, param_sets=sets)
    assert result["pass"], result


def test_swap_and_route_semantics():
    ctx = np.arange(8)
    eff, _ = channel_params(D0B, ctx, np.zeros(8, dtype=int))
    b = (ctx >> 1) & 1
    assert np.allclose(eff[b == 0, 2], 0.1) and np.allclose(eff[b == 1, 2], 0.9)
    assert np.allclose(eff[b == 1, 0], 0.1) and np.allclose(eff[b == 0, 0], 0.9)
    route = D0B.with_dials(c_mode="route", c_channels=(1, 2), d31=0.0)
    for cue in (0, 1):
        _, delay = channel_params(route, np.array([1, 0]), np.array([cue, cue]))
        fast = 1 if cue else 2
        assert delay[0, fast] == 0.0 and delay[0, 3 - fast] == 6.0
        assert delay[1, 1] == 6.0 and delay[1, 2] == 6.0


def test_value_tables_cover_all_actions():
    from acp_cl.commons.world import sample_episodes
    ep = sample_episodes(2, 3)
    vt = R.value_tables(D0B, ep, [0, 7], 256, 1, "t")
    assert vt.value.shape == (2, 2, 13)
