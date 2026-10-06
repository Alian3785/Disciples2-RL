"""The faster fixture must preserve behavior and fully isolate mutable state."""
from copy import deepcopy

import numpy as np

from campaign_env import CampaignEnv


def test_snapshot_matches_fresh_reset_and_public_action(campaign_factory):
    clone = campaign_factory(map_name="trade_train")
    fresh = CampaignEnv(map_name="trade_train", observation_version="local5",
                        scripted_capital_bot_enabled=False,
                        use_boss_starting_roster=False, log_enabled=False)
    try:
        fresh.reset(seed=42)
        assert clone.blue_team_state == fresh.blue_team_state
        assert clone.enemy_team_states == fresh.enemy_team_states
        np.testing.assert_array_equal(clone._build_obs(), fresh._build_obs())
        np.testing.assert_array_equal(clone.compute_action_mask(), fresh.compute_action_mask())
        action = next(i for i in range(8) if clone.compute_action_mask()[i])
        a, b = clone.step(action), fresh.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:4] == b[1:4]
        assert clone.grid_env.agent_pos == fresh.grid_env.agent_pos
    finally:
        fresh.close()


def test_mutations_and_position_callbacks_cannot_leak_between_copies(campaign_factory):
    first = campaign_factory(map_name="trade_train")
    second = campaign_factory(map_name="trade_train")
    initial_army = deepcopy(second.blue_team_state)
    initial_positions = dict(second.grid_env.enemy_positions)
    first.blue_team_state[0]["health"] = 0
    first.heroitems.append("fixture-only-item")
    first.gold = -123
    first.grid_env._enemy_cache_dirty = False
    second.grid_env._enemy_cache_dirty = False
    first.grid_env.enemy_positions[1] = (0, 0)
    assert first.grid_env._enemy_cache_dirty
    assert not second.grid_env._enemy_cache_dirty
    assert second.blue_team_state == initial_army
    assert dict(second.grid_env.enemy_positions) == initial_positions
    assert "fixture-only-item" not in second.heroitems
    assert second.gold != -123
    third = campaign_factory(map_name="trade_train")
    assert third.blue_team_state == initial_army
    assert dict(third.grid_env.enemy_positions) == initial_positions
    assert third.grid_env.enemy_positions._on_change.__self__ is third.grid_env
    # The random stream is cloned, not shared between environments.
    assert first.np_random.random() == second.np_random.random() == third.np_random.random()


def test_different_configuration_has_its_own_snapshot(campaign_factory):
    empire = campaign_factory(map_name="default", Realcapital=1)
    legions = campaign_factory(map_name="default", Realcapital=2)
    assert empire.Realcapital == 1 and legions.Realcapital == 2
    assert empire.blue_team_state != legions.blue_team_state
