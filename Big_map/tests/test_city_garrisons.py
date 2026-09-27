"""Regression tests for city recruitment and the interactive defence lifecycle."""
from copy import deepcopy

import numpy as np
import pytest

from campaign_env import CampaignEnv
from maps import available_maps


@pytest.fixture
def env():
    e = CampaignEnv(Realcapital=2, observation_version='local5',
                    scripted_capital_bot_enabled=False,
                    use_boss_starting_roster=False, log_enabled=False)
    e.reset(seed=42)
    e.gold = 10000
    yield e
    e.close()


def capture(e, index=0, level=5):
    city = e.garrison_city_names[index]
    for enemy_id in e.legions_settlement_territory_source_by_name[city]['required_enemy_ids']:
        e.grid_env.mark_enemy_defeated(enemy_id)
    e.grid_env.agent_pos = e.legions_settlement_source_tile_by_name[city]
    e._capture_cities_on_hero_entry(0, {})
    e.legions_settlement_level_by_name[city] = level
    return city


def hire(e, city, option=0):
    action = e.GRID_GARRISON_HIRE_ACTION_START + e.garrison_city_names.index(city) * 5 + option
    return e.step(action)


def bot(e, city, weak=False):
    # Explicitly construct the attacker only in focused bot integration tests.
    e.scripted_capital_bot_enabled = True
    e.scripted_capital_bot_team_state = e._create_scripted_capital_bot_team()
    if weak:
        for unit in e.scripted_capital_bot_team_state:
            unit.update(initiative=1, initiative_base=1, damage=1, original_damage=1)
    e.scripted_capital_bot_state = 'hunting'
    tile = e.legions_settlement_source_tile_by_name[city]
    e.scripted_capital_bot_position = (tile[0] - 1, tile[1])
    e.grid_env.agent_pos = tuple(e.CASTLE_POS)
    return tile


def test_five_stable_actions_per_city_and_remote_recruitment(env):
    start, size = env.GRID_GARRISON_HIRE_ACTION_START, env.action_space.n
    assert not env.compute_action_mask()[start:].any()
    for entry in env.active_buildings.values():
        if isinstance(entry, dict):
            entry['built'] = 1
    # Use the same set of required building names as the production gate.
    original = env._built_building_names_set
    env._built_building_names_set = lambda: original() | {
        o.get('required_building', '') for o in env.active_hire_options}
    for index in range(3):
        capture(env, index)
        assert env.compute_action_mask()[start:].sum() == (index + 1) * 5
        assert env.action_space.n == size
    env.grid_env.agent_pos = tuple(env.CASTLE_POS)
    hero, gold, moves = deepcopy(env.blue_team_state), env.gold, env.moves
    obs, reward, terminated, truncated, info = hire(env, env.garrison_city_names[1])
    assert info['garrison_hired']
    assert env.gold == gold - 50
    assert env.moves == moves and env.blue_team_state == hero
    assert env.observation_space.contains(obs)
    assert not (terminated or truncated)
    assert reward <= 0  # No repeatable recruitment reward farm.


@pytest.mark.parametrize('level', range(1, 6))
def test_level_capacity_and_gold(env, level):
    city = capture(env, level=level)
    for index in range(level):
        assert hire(env, city, 0 if index < 3 else 1)[4]['garrison_hired']
    assert len(env._garrison_occupied_positions(city)) == level
    assert not env.compute_action_mask()[env.GRID_GARRISON_HIRE_ACTION_START:][:5].any()
    gold = env.gold
    assert not hire(env, city)[4]['garrison_hired']
    assert env.gold == gold
    env.city_garrisons[city] = []
    env.gold = 49
    assert not hire(env, city)[4]['garrison_hired']
    env.gold = 50
    assert hire(env, city)[4]['garrison_hired']


def test_big_units_need_capacity_and_a_free_column(env):
    city = capture(env, level=1)
    assert not hire(env, city, 2)[4]['garrison_hired']
    env.legions_settlement_level_by_name[city] = 5
    assert hire(env, city, 2)[4]['garrison_hired']
    big = env.city_garrisons[city][0]
    assert env._garrison_occupied_positions(city) == {big['position'], env._blue_partner_position(big['position'])}
    assert hire(env, city, 2)[4]['garrison_hired']
    assert not hire(env, city, 2)[4]['garrison_hired']
    assert hire(env, city, 1)[4]['garrison_hired']
    assert len(env._garrison_occupied_positions(city)) == 5
    env.city_garrisons[city] = [env._build_unit_from_data(env._find_unit_data_by_name('Сектант'), 'blue', p)
                               for p in (10, 11, 12)]
    assert not hire(env, city, 2)[4]['garrison_hired']  # Two capacity points but no column.
    assert not hire(env, city, 1)[4]['garrison_hired']  # Back rank is full.
    assert hire(env, city, 0)[4]['garrison_hired']


def test_required_building_gate(env):
    city = capture(env)
    assert not hire(env, city, 3)[4]['garrison_hired']
    original = env._built_building_names_set
    env._built_building_names_set = lambda: original() | {'Храм Скорби'}
    assert hire(env, city, 3)[4]['garrison_hired']


def test_seeded_random_positions_and_all_front_slots(env):
    positions = set()
    for seed in range(12):
        env.reset(seed=seed)
        city = capture(env)
        env.gold = 1000
        first = hire(env, city)[4]['hired_position']
        positions.add(first)
        assert first in (7, 8, 9)
        assert hire(env, city, 1)[4]['hired_position'] in (10, 11, 12)
        env.reset(seed=seed)
        city = capture(env)
        env.gold = 1000
        assert hire(env, city)[4]['hired_position'] == first
    assert positions == {7, 8, 9}


@pytest.mark.parametrize('capital', range(1, 6))
def test_five_faction_rosters(capital):
    env = CampaignEnv(Realcapital=capital, observation_version='local5',
                      scripted_capital_bot_enabled=False, use_boss_starting_roster=False, log_enabled=False)
    env.reset(seed=42)
    city = capture(env)
    env.gold = 10000
    assert len(env.active_hire_options) == 5
    original = env._built_building_names_set
    env._built_building_names_set = lambda: original() | {o.get('required_building', '') for o in env.active_hire_options}
    for index, option in enumerate(env.active_hire_options):
        env.city_garrisons[city] = []
        info = hire(env, city, index)[4]
        assert info['garrison_hired']
        assert info['hired_unit_name'] == option['name']
    env.close()


def test_bot_enters_city_and_starts_agent_controlled_battle(env):
    city = capture(env)
    hire(env, city)
    tile = bot(env, city, weak=True)
    hero = deepcopy(env.blue_team_state)
    path = [env.scripted_capital_bot_position, tile, (tile[0] + 1, tile[1])]
    assert env._move_scripted_bot_along_path(path)[-1] == tile
    assert env.pending_garrison_city == city
    result = env._finish_pending_city_defence((None, 0, False, False, {}))
    assert result[4]['battle_context_kind'] == 'city_garrison'
    assert env.mode == env.MODE_BATTLE
    assert env.blue_team_state == hero
    assert not env.compute_action_mask()[env.GRID_GARRISON_HIRE_ACTION_START:].any()
    assert env.observation_space.contains(result[0])
    assert {u['name'] for u in env.battle_env.combined if u['team'] == 'blue' and u['health'] > 0} == {'Одержимый'}


@pytest.mark.parametrize('winner', ('red', 'blue', None))
def test_defence_result_preserves_hero_and_campaign(env, winner):
    city = capture(env)
    hire(env, city)
    bot(env, city, weak=True)
    hero = deepcopy(env.blue_team_state)
    hero_position = env.grid_env.agent_pos
    env._start_city_defence(city)
    for unit in env.battle_env.combined:
        if unit['team'] == 'blue' and unit['health'] > 0:
            unit['health'] -= 1
    env.battle_env.winner = winner
    if winner is None:
        env.battle_env.step = lambda action: (env.battle_env._obs(), 0, False, True, {})
    obs, _, terminated, truncated, info = env.step(0)
    assert not (terminated or truncated)
    assert env.mode == env.MODE_GRID
    assert env.blue_team_state == hero
    assert env.grid_env.agent_pos == hero_position
    assert env.observation_space.contains(obs)
    if winner == 'red':
        assert not env._garrison_player_owns(city)
        assert city in env.bot_city_capture_turns
        assert not hire(env, city)[4]['garrison_hired']
        tile = env.legions_settlement_source_tile_by_name[city]
        assert tile in env.empire_territory_tile_set
        assert tile not in env.legions_territory_tile_set
        before = len(env.empire_territory_tile_set)
        env.turns += 2
        env._refresh_faction_territories()
        assert len(env.empire_territory_tile_set) > before
        assert capture(env) == city
        assert city not in env.bot_city_capture_turns
        assert hire(env, city)[4]['garrison_hired']
    else:
        assert env._garrison_player_owns(city)
        assert env.city_garrisons[city][0]['health'] < env.city_garrisons[city][0]['max_health']
        assert env.city_garrisons[city][0]['armor'] == 0  # No accumulated city armour.


def test_unguarded_capture_and_bot_targeting(env):
    city = capture(env)
    tile = bot(env, city)
    tiles = env._scripted_bot_alive_enemy_tiles()
    assert tiles[tile] <= -1000
    env._move_scripted_bot_along_path([env.scripted_capital_bot_position, tile])
    assert city in env.bot_city_capture_turns
    assert env.pending_garrison_city is None
    assert tile not in env._scripted_bot_alive_enemy_tiles()


def test_end_turn_routes_city_attack_to_agent(env):
    city = capture(env)
    hire(env, city)
    bot(env, city, weak=True)
    obs, _, terminated, truncated, info = env.step(8)
    assert not (terminated or truncated)
    assert env.mode == env.MODE_BATTLE
    assert env.current_battle_context == {'kind': 'city_garrison', 'city': city}
    assert info['battle_triggered']
    assert env.observation_space.contains(obs)


def test_city_battle_waits_for_an_existing_battle(env):
    city = capture(env)
    hire(env, city)
    bot(env, city, weak=True)
    env.pending_garrison_city = city
    env.mode = env.MODE_BATTLE
    result = (None, 0, False, False, {})
    assert env._finish_pending_city_defence(result) == result
    assert env.pending_garrison_city == city
    env.mode = env.MODE_GRID
    assert env._finish_pending_city_defence(result)[4]['garrison_city'] == city


def test_real_combat_finishes_without_touching_hero(env):
    city = capture(env)
    hire(env, city)
    bot(env, city)
    hero = deepcopy(env.blue_team_state)
    env._start_city_defence(city)
    for _ in range(500):
        _, _, terminated, truncated, info = env.step(env.battle_env.scripted_action_for_current_blue())
        assert not (terminated or truncated)
        if env.mode == env.MODE_GRID:
            break
    assert env.mode == env.MODE_GRID
    assert info['battle_result'] in ('victory', 'defeat')
    assert env.blue_team_state == hero


def test_reset_clears_reserves_ownership_and_pending_defence(env):
    city = capture(env)
    hire(env, city)
    env._transfer_city_to_bot(city)
    env.pending_garrison_city = city
    expected_shape = env.observation_space.shape
    obs, _ = env.reset(seed=5)
    assert not env.city_garrisons and not env.bot_city_capture_turns
    assert env.pending_garrison_city is None
    assert obs.shape == expected_shape
    assert not env.compute_action_mask()[env.GRID_GARRISON_HIRE_ACTION_START:].any()


@pytest.mark.parametrize('map_name', available_maps())
def test_map_layout_and_reset(map_name):
    e = CampaignEnv(map_name=map_name, observation_version='local5',
                    scripted_capital_bot_enabled=False, use_boss_starting_roster=False, log_enabled=False)
    try:
        shape, actions = e.observation_space.shape, e.action_space.n
        obs, _ = e.reset(seed=42)
        assert obs.shape == shape
        assert e.observation_space.contains(obs)
        assert e.action_space.n == actions
        assert e.compute_action_mask().shape == (actions,)
        assert e.GRID_GARRISON_HIRE_ACTION_COUNT == 5 * len(e.garrison_city_names)
        assert tuple(e.CASTLE_POS) not in [e.legions_settlement_source_tile_by_name[c] for c in e.garrison_city_names]
        e.step(int(np.flatnonzero(e.compute_action_mask())[0]))
    finally:
        e.close()
