"""Only a travelling hero army can complete Mirrow match's bot objective."""
from copy import deepcopy

import pytest

from campaign_env import CampaignEnv
from maps import available_maps


@pytest.fixture
def env():
    instance = CampaignEnv(map_name="mirrow_match", observation_version="local5",
                           scripted_capital_bot_enabled=True,
                           use_boss_starting_roster=False, log_enabled=False)
    instance.reset(seed=7)
    yield instance
    instance.close()


def place_bot(env, position=(7, 5)):
    env.scripted_capital_bot_position = position
    env._sync_scripted_capital_bot_grid_state()


def start_bot_battle(env, kind="hero", attacker="blue"):
    place_bot(env)
    env.current_enemy_id = -75
    env.battle_origin_pos = tuple(env.grid_env.agent_pos)
    env.current_battle_context = {"kind": kind}
    env.mode = env.MODE_BATTLE
    env._init_battle(-75, attacker_team=attacker)


def force_blue_win(env):
    for unit in env.battle_env.combined:
        if unit["team"] == "red":
            unit["health"] = unit["hp"] = 0
    env.battle_env.winner = "blue"
    return env.step(0)


@pytest.mark.parametrize("attacker", ["blue", "red"])
def test_one_real_hero_army_victory_is_terminal(env, attacker):
    start_bot_battle(env, attacker=attacker)
    obs, reward, done, timeout, info = force_blue_win(env)
    assert done and not timeout
    assert info["campaign_result"] == "victory"
    assert info["campaign_victory_reason"] == "scripted_bot_defeated_by_hero"
    assert info["objective_defeat_source"] == "hero_battle"
    assert info["final_objective_reward"] == env.reward_all_enemies
    assert env.scripted_bot_hero_victories == 1
    assert env.scripted_capital_bot_respawn_turns_left == 2
    assert env.scripted_capital_bot_state == "defeated"
    assert env.observation_space.contains(obs)


def test_movement_initiates_actual_bot_battle(env):
    place_bot(env)
    _, _, done, timeout, info = env.step(env.grid_env.ACTION_RIGHT)
    assert not done and not timeout
    assert env.mode == env.MODE_BATTLE and env.current_enemy_id == -75
    assert env.current_battle_context["scripted_bot"]
    assert info["enemy_id"] == -75
    assert sorted(u["name"] for u in env.battle_env.combined
                  if u["team"] == "red" and not env._is_empty_enemy_unit(u)) == [
                      "Рыцарь на пегасе", "Скваер", "Скваер", "Служка"]


def test_neutral_victory_even_last_neutral_does_not_complete(env):
    enemy_id = next(iter(env._enemy_configs))
    for other in env.grid_env.enemies_alive:
        if other != -75:
            env.grid_env.enemies_alive[other] = other == enemy_id
    env.current_enemy_id = enemy_id
    env.battle_origin_pos = env.grid_env.agent_pos
    env.current_battle_context = {"kind": "hero"}
    env.mode = env.MODE_BATTLE
    env._init_battle(enemy_id)
    _, _, done, timeout, info = force_blue_win(env)
    assert not done and not timeout
    assert info.get("campaign_result") != "victory"
    assert env.scripted_bot_hero_victories == 0


def test_last_neutral_disappearing_does_not_complete_grid_step(env):
    for enemy_id in env.grid_env.enemies_alive:
        if enemy_id != -75:
            env.grid_env.enemies_alive[enemy_id] = False
    _, _, done, timeout, info = env.step(env.grid_env.ACTION_REST)
    assert not done and not timeout
    assert info.get("campaign_result") != "victory"


def test_lethal_actual_map_spell_does_not_win_and_respawns(env):
    place_bot(env)
    result = env._cast_from_spell_spec(
        source="learned_offensive_spell", spell_key="lod_d2_s003", spell_description="test",
        spell_spec={"spell_kind": "damage", "damage": 10000, "damage_type": "Fire"})
    assert result["target_enemy_id"] == -75
    assert result["spell_enemy_defeated"]
    assert not result["objective_reward_eligible"]
    _, _, done, timeout, info = env._finalize_map_spell_step(
        reward=result["reward"], info={}, cast_result=result)
    assert not done and not timeout and info.get("campaign_result") != "victory"
    assert env.scripted_bot_hero_victories == 0
    assert env.scripted_capital_bot_state == "defeated"
    env._advance_turns(1)
    assert env.scripted_capital_bot_state == "defeated"
    env._advance_turns(1)
    assert env.scripted_capital_bot_state == "hunting"
    assert env.scripted_capital_bot_position == env.scripted_capital_bot_home
    assert all(env._unit_current_hp(u) == env._unit_max_hp(u)
               for u in env.scripted_capital_bot_team_state)
    assert env._campaign_objective_completion_reason(-75) is None


def test_summon_battle_win_does_not_win_campaign(env):
    start_bot_battle(env, kind="summon_spell")
    _, _, done, timeout, info = force_blue_win(env)
    assert not done and not timeout
    assert info["battle_result"] == "victory"
    assert info.get("campaign_result") != "victory"
    assert env.scripted_bot_hero_victories == 0
    assert env.scripted_capital_bot_state == "defeated"
    assert env.scripted_capital_bot_respawn_turns_left == 2


def test_reset_clears_completed_objective_and_wounds(env):
    start_bot_battle(env)
    force_blue_win(env)
    obs, _ = env.reset(seed=8)
    assert env.scripted_bot_hero_victories == 0
    assert not env.scripted_bot_objective_reward_granted
    assert env._campaign_objective_completion_reason(-75) is None
    assert env.grid_env.enemies_alive[-75]
    assert env.observation_space.contains(obs)


def test_nonfatal_magic_damage_survives_sync_and_movement(env):
    place_bot(env)
    before = sum(env._unit_current_hp(u) for u in env.scripted_capital_bot_team_state)
    result = env._cast_from_spell_spec(
        source="learned_offensive_spell", spell_key="lod_d2_s003", spell_description="test",
        spell_spec={"spell_kind": "damage", "damage": 1, "damage_type": "Fire"})
    assert result["spell_damage_total"] > 0
    expected = before - result["spell_damage_total"]
    env._sync_scripted_capital_bot_grid_state()
    assert sum(env._unit_current_hp(u) for u in env.scripted_capital_bot_team_state) == expected
    assert sum(env._unit_current_hp(u) for u in env._get_enemy_team_state(-75)) == expected


def test_field_bot_uses_own_recovery_not_neutral_healing(env):
    place_bot(env)
    for unit in env.scripted_capital_bot_team_state:
        if not env._is_empty_blue_unit(unit):
            unit["health"] = unit["hp"] = max(1, env._unit_current_hp(unit) - 10)
    env._sync_scripted_capital_bot_grid_state()
    before = deepcopy(env._get_enemy_team_state(-75))
    env._heal_wounded_enemy_teams_for_turn()
    assert env._get_enemy_team_state(-75) == before


@pytest.mark.parametrize("map_name", available_maps())
@pytest.mark.parametrize("version", ["local5", "baseline"])
def test_dynamic_bot_does_not_change_observation_or_action_shape(map_name, version):
    env = CampaignEnv(map_name=map_name, observation_version=version,
                      scripted_capital_bot_enabled=True, use_boss_starting_roster=False)
    try:
        obs, _ = env.reset(seed=42)
        shape, actions = env.observation_space.shape, env.action_space.n
        assert env.observation_space.contains(obs)
        assert env.grid_env._get_obs().shape == env.grid_env.observation_space.shape
        if env.scripted_capital_bot_enabled:
            env._mark_scripted_capital_bot_defeated()
            assert env.observation_space.contains(env._build_obs())
            env._advance_turns(2)
            assert env.observation_space.contains(env._build_obs())
        obs, _ = env.reset(seed=43)
        assert env.observation_space.shape == shape and env.action_space.n == actions
        assert env.observation_space.contains(obs)
        assert env.compute_action_mask().shape == (actions,)
    finally:
        env.close()
