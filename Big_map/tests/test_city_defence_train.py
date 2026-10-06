"""Scenario integration: ownership, faction settings, timing and both assaults."""
from copy import deepcopy

import pytest

from campaign_env import CampaignEnv
from maps.city_defence_train import CITY_NAME, CITY_TILE, CAPITAL_TILE, BOT_SPAWN_TILE


def make(capital=2, enabled=True):
    # Scripted attackers are explicitly requested for this scenario's tests.
    env = CampaignEnv(map_name="city_defence_train", Realcapital=capital,
                      observation_version="local5", scripted_capital_bot_enabled=enabled,
                      use_boss_starting_roster=False, log_enabled=False)
    env.reset(seed=42)
    return env


def living(team):
    return {u["position"]: u["name"] for u in team if u.get("health", 0) > 0}


def hire(env, option=0):
    action = env.GRID_GARRISON_HIRE_ACTION_START + option
    assert env.compute_action_mask()[action]
    result = env.step(action)
    assert result[4]["garrison_hired"]
    assert not (result[2] or result[3])
    return result


@pytest.mark.parametrize("capital", range(1, 6))
def test_initial_city_and_faction_specific_remote_hire(capital):
    e = make(capital)
    try:
        assert e.grid_env.grid_size == 48
        assert e.Realcapital == capital
        assert e.grid_env.agent_pos == e.CASTLE_POS == CAPITAL_TILE
        assert e._garrison_player_owns(CITY_NAME)
        assert e.legions_settlement_source_tile_by_name[CITY_NAME] == CITY_TILE
        assert e._garrison_capacity(CITY_NAME) == 5
        assert e.city_garrisons[CITY_NAME] == []
        assert CITY_TILE in e.legions_territory_tile_set
        assert e.scripted_capital_bot_position == BOT_SPAWN_TILE
        assert living(e.scripted_capital_bot_team_state) == {7: "Воин", 9: "Воин", 11: "Колдун"}
        hero = deepcopy(e.blue_team_state)
        gold = e.gold
        obs, _, _, _, info = hire(e)
        assert e.turns == 0
        assert e.blue_team_state == hero
        assert info["hired_unit_name"] == e.active_hire_options[0]["name"]
        assert e.gold == gold - e.active_hire_options[0]["gold"]
        assert e.observation_space.contains(obs)
        e.step(8)
        assert e.turns == 1 and e.mode == e.MODE_GRID
        assert e.scripted_capital_bot_position not in (BOT_SPAWN_TILE, CITY_TILE)
        obs, _, terminated, truncated, info = e.step(8)
        assert e.turns == 2 and not (terminated or truncated)
        assert e.scripted_capital_bot_position == CITY_TILE
        assert e.mode == e.MODE_BATTLE
        assert info["battle_context_kind"] == "city_garrison"
        assert e.blue_team_state == hero
        assert e.observation_space.contains(obs)
    finally:
        e.close()


def test_empty_city_is_lost_on_second_turn_not_at_reset_or_first_action():
    e = make()
    try:
        # Moving on an empty map must not produce an all-enemies victory.
        action = next(a for a in range(8) if e.compute_action_mask()[a])
        assert not any(e.step(action)[2:4])
        assert not any(e.step(8)[2:4])
        obs, reward, terminated, truncated, info = e.step(8)
        assert terminated and not truncated and info["campaign_result"] == "defeat"
        assert info["campaign_defeat_reason"] == "defended_city_captured"
        assert info["city_owners"][CITY_NAME] == "scripted_bot"
        assert not e._garrison_player_owns(CITY_NAME)
        assert reward < 0 and e.observation_space.contains(obs)
        assert e.city_defence_wins == 0 and e.city_defence_wave == 0
    finally:
        e.close()


def test_second_wave_respawns_after_two_turns_then_marches_and_second_win_finishes():
    e = make()
    try:
        hire(e)
        hero = deepcopy(e.blue_team_state)
        e.step(8)
        e.step(8)
        e.battle_env.winner = "blue"
        obs, _, terminated, truncated, info = e.step(0)
        assert not (terminated or truncated)
        assert e.city_defence_wins == 1 and e.city_defence_wave == 1
        assert e.scripted_capital_bot_state == "defeated"
        assert e.scripted_capital_bot_respawn_turns_left == 2
        assert e.grid_env.scripted_bot_position is None
        assert info["city_defence_attack_turn"] == 6
        assert e.observation_space.contains(obs)
        e.step(8)
        assert e.turns == 3 and e.mode == e.MODE_GRID
        assert e.city_defence_wins == 1
        assert e.scripted_capital_bot_state == "defeated"
        assert e.grid_env.scripted_bot_position is None
        e.step(8)
        assert e.turns == 4 and e.mode == e.MODE_GRID
        assert e.scripted_capital_bot_state == "hunting"
        assert e.scripted_capital_bot_position == BOT_SPAWN_TILE
        assert living(e.scripted_capital_bot_team_state) == {8: "Гном"}
        e.step(8)
        assert e.turns == 5 and e.mode == e.MODE_GRID
        e.step(8)
        assert e.turns == 6 and e.mode == e.MODE_BATTLE
        assert living([u for u in e.battle_env.combined if u["team"] == "red"]) == {2: "Гном"}
        e.battle_env.winner = "blue"
        obs, _, terminated, truncated, info = e.step(0)
        assert terminated and not truncated
        assert info["campaign_result"] == "victory"
        assert info["campaign_victory_reason"] == "city_assaults_defeated"
        assert e.city_defence_wins == 2
        assert e._garrison_player_owns(CITY_NAME)
        assert e.blue_team_state == hero
        assert e.grid_env.scripted_bot_position is None
        assert e.observation_space.contains(obs)
        shape = obs.shape
        obs, _ = e.reset(seed=42)
        assert obs.shape == shape
        assert e.turns == 0 and e.city_defence_wins == 0 and e.city_defence_wave == 0
        assert e.city_garrisons[CITY_NAME] == []
        assert living(e.scripted_capital_bot_team_state) == {7: "Воин", 9: "Воин", 11: "Колдун"}
    finally:
        e.close()


@pytest.mark.parametrize("wave", [0, 1])
def test_lost_garrison_ends_episode_in_either_wave(wave):
    e = make()
    try:
        hire(e)
        hero = deepcopy(e.blue_team_state)
        e.step(8)
        e.step(8)
        if wave:
            e.battle_env.winner = "blue"
            e.step(0)
            for _ in range(4):
                e.step(8)
        e.battle_env.winner = "red"
        obs, _, terminated, truncated, info = e.step(0)
        assert terminated and not truncated and info["campaign_result"] == "defeat"
        assert e.city_defence_wins == wave
        assert e.blue_team_state == hero
        assert e.observation_space.contains(obs)
    finally:
        e.close()


@pytest.mark.parametrize("capital", range(1, 6))
def test_real_combat_resolves_campaign_for_every_faction(capital):
    e = make(capital)
    try:
        for option in (0, 0, 0, 1, 1):
            hire(e, option)
        hero = deepcopy(e.blue_team_state)
        for day in range(1, 7):
            result = e.step(8)
            if e.mode == e.MODE_BATTLE:
                e.battle_env.seed(42)
                for _ in range(500):
                    result = e.step(e.battle_env.scripted_action_for_current_blue())
                    assert e.observation_space.contains(result[0])
                    if e.mode == e.MODE_GRID:
                        break
                assert e.mode == e.MODE_GRID
            assert e.blue_team_state == hero
            if result[2] or result[3]:
                break
        assert result[2] and not result[3]
        if result[4]["campaign_result"] == "victory":
            assert e.city_defence_wins == 2 and day == 6
            assert e._garrison_player_owns(CITY_NAME)
        else:
            assert result[4]["campaign_result"] == "defeat"
            assert not e._garrison_player_owns(CITY_NAME)
            assert e.city_defence_wins < 2
    finally:
        e.close()


def test_bot_disable_is_respected_on_reset():
    e = make(enabled=False)
    try:
        for _ in range(5):
            assert not any(e.step(8)[2:4])
        assert e.city_defence_wins == 0 and e._garrison_player_owns(CITY_NAME)
        assert not e.scripted_capital_bot_team_state
        e.reset(seed=42, options={"Realcapital": 2})
        assert e.Realcapital == 2
        assert e._map.starting_roster is None
        assert e._map.starting_capital_id is None
        assert e.active_hire_options == e._get_hire_options_for_capital(2)
    finally:
        e.close()


def test_scenario_objective_is_not_available_on_other_maps():
    with pytest.raises(ValueError, match="city_defence"):
        CampaignEnv(map_name="default", campaign_objective="city_defence",
                    observation_version="local5", scripted_capital_bot_enabled=False,
                    use_boss_starting_roster=False)
