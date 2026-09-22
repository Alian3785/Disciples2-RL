import pytest

from campaign_env import CampaignEnv
from test_campaign_green_dragon_victory import _DummyBattleEnv


def _city_env(enemy_id):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=False, Realcapital=2)
    env.reset(seed=123)
    tile = tuple(env.grid_env.enemy_positions[enemy_id])
    env.grid_env.agent_pos = (tile[0] - 1, tile[1])
    env.grid_env.obstacle_positions = set()
    env.grid_env.enemies_alive = dict.fromkeys(env.grid_env.enemies_alive, False)
    env.moves = 99
    return env, tile


def _enter(env, tile):
    action = next(a for a in range(8) if env.grid_env._target_pos_for_action(a) == tile)
    assert env.compute_action_mask()[action]
    return env.step(action)


def test_hero_city_battle_returns_to_origin_then_entry_captures_last_city():
    env, tile = _city_env(68)
    city = env._objective_city_for_enemy(68)
    env.captured_objective_cities = set(env.FINAL_OBJECTIVE_CITIES) - {city}
    env.grid_env.enemies_alive[68] = True
    origin = tuple(env.grid_env.agent_pos)
    _, _, terminated, _, info = _enter(env, tile)
    assert not terminated and info['battle_triggered']
    env.battle_env = _DummyBattleEnv()
    _, _, terminated, _, info = env.step(0)
    assert not terminated
    assert tuple(env.grid_env.agent_pos) == origin
    assert city not in env.captured_objective_cities
    assert not info.get('final_objective_reward')
    _, _, terminated, _, info = _enter(env, tile)
    assert terminated
    assert info['captured_objective_cities'] == [city]
    assert info['campaign_victory_reason'] == 'objective_cities_cleared'


@pytest.mark.parametrize('enemy_id', [67, 68])
def test_cleared_settlement_requires_entry_and_never_rewards_reentry(enemy_id):
    env, tile = _city_env(enemy_id)
    name = env.legions_settlement_territory_source_name_by_enemy_id[enemy_id]
    assert env._activate_legions_settlement_territory_if_cleared(enemy_id) == []
    _, _, _, _, info = _enter(env, tile)
    assert info['legions_settlement_territories_activated'] == [name]
    capture_turn = env.legions_active_settlement_territory_capture_turn_by_name[name]
    _enter(env, (tile[0] - 1, tile[1]))
    _, _, _, _, info = _enter(env, tile)
    assert not info.get('captured_objective_cities')
    assert not info.get('final_objective_reward')
    assert not info.get('legions_settlement_territories_activated')
    assert env.legions_active_settlement_territory_capture_turn_by_name[name] == capture_turn


def test_blocked_entry_and_surviving_defender_do_not_capture_city():
    env, tile = _city_env(68)
    env.grid_env.obstacle_positions.add(tile)
    action = next(a for a in range(8) if env.grid_env._target_pos_for_action(a) == tile)
    env.step(action)
    assert not env.captured_objective_cities
    assert not env.legions_active_settlement_territory_capture_turn_by_name
    env.grid_env.obstacle_positions.clear()
    env.grid_env.enemies_alive[34] = True
    env.grid_env.agent_pos = tile
    assert env._capture_objective_city_if_cleared(68) == []
    assert env._activate_legions_settlement_territory_if_cleared(68) == []
