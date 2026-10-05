"""EM4: city guards apply the earned building-backed evolution before save."""
from copy import deepcopy
import json

import pytest

from campaign_env import CampaignEnv
from campaign_env_data import SETTLEMENT_ARMOR_BONUS_BY_LEVEL


def make_env():
    env = CampaignEnv(
        map_name='city_defence_train', Realcapital=2, observation_version='local5',
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    env.reset(seed=42)
    return env


def build_portal(env):
    key = next(key for key, data in env.active_buildings.items()
               if isinstance(data, dict) and data.get('unit') == 'Берсерк')
    result = env.step(env.GRID_BUILD_ACTION_START + env.building_keys.index(key))
    assert result[4]['built']


def attacker(env, xp=25):
    unit = env._build_unit_from_data(env._find_unit_data_by_name('Гном'), 'blue', 8)
    unit.update(hp=1, health=1, damage=0, original_damage=0,
                initiative=0, initiative_base=0, exp_kill=xp)
    env.scripted_capital_bot_team_state = env._build_battle_team_with_placeholders('blue', [unit])


def finish_battle(env):
    battle = env.battle_env
    battle.seed(42)
    total_reward = 0
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(battle.scripted_action_for_current_blue())
        total_reward += reward
        assert not truncated
        if env.mode != env.MODE_GRID:
            assert not terminated
        assert env.observation_space.contains(obs)
        if env.mode == env.MODE_GRID:
            assert info['battle_result'] == 'victory'
            assert battle._battle_exp_event_count > 0
            return battle, total_reward, info
    pytest.fail('Controlled garrison battle did not finish')


def canonical_stats(unit):
    return {key: unit.get(key) for key in (
        'name', 'Level', 'exp_current', 'exp_required', 'exp_kill', 'next_level_exp',
        'health', 'hp', 'max_health', 'maxhp', 'damage', 'armor', 'accuracy',
        'initiative_base', 'unit_type', 'turns_into', 'needaunit', 'is_big',
    )}


@pytest.mark.parametrize('built', (False, True))
def test_possessed_94_of_95_real_city_victory_and_next_battle(built):
    env = make_env()
    try:
        city = env.garrison_city_names[0]
        if built:
            build_portal(env)
        assert env.step(env.GRID_GARRISON_HIRE_ACTION_START)[4]['garrison_hired']
        guard = env.city_garrisons[city][0]
        assert guard['name'] == 'Одержимый'
        assert guard['exp_required'] == 95
        guard['exp_current'] = 94
        hero = deepcopy(env.blue_team_state)
        items = deepcopy(env.heroitems)
        gold, moves, position = env.gold, env.moves, env.grid_env.agent_pos
        attacker(env)
        env._start_city_defence(city)
        battle, _, info = finish_battle(env)
        assert battle.last_levelups == ['Одержимый']
        saved = env.city_garrisons[city][0]
        expected_name = 'Берсерк' if built else 'Одержимый'
        assert saved['name'] == expected_name
        assert info['garrison_unit_upgrades'] == int(built)
        expected = env._build_unit_from_data(
            env._find_unit_data_by_name(expected_name), 'blue', saved['position'])
        expected['exp_current'] = 0 if built else 94
        assert canonical_stats(saved) == canonical_stats(expected)
        assert 'garrison_base_armor' not in saved
        assert env.blue_team_state == hero and env.heroitems == items
        assert (env.gold, env.moves, env.grid_env.agent_pos) == (gold, moves, position)

        # Public snapshot data can round-trip as JSON and become the next
        # defence's roster. No standalone save-game API is assumed here.
        env.city_garrisons = json.loads(json.dumps(info['city_garrisons']))
        attacker(env, xp=0)
        env._start_city_defence(city)
        restored = next(unit for unit in env.battle_env.combined
                        if unit.get('team') == 'blue' and unit.get('name') == expected_name)
        assert restored['Level'] == saved['Level']
        assert restored['exp_current'] == saved['exp_current']
        assert restored['armor'] == min(90, saved['armor'] + SETTLEMENT_ARMOR_BONUS_BY_LEVEL[5])
        _, _, info2 = finish_battle(env)
        assert info2['garrison_unit_upgrades'] == 0
        assert canonical_stats(env.city_garrisons[city][0]) == canonical_stats(saved)
        assert env.blue_team_state == hero and env.heroitems == items
        assert (env.gold, env.moves, env.grid_env.agent_pos) == (gold, moves, position)
    finally:
        env.close()


def test_no_building_cap_then_build_and_promote_on_later_victory():
    env = make_env()
    try:
        city = env.garrison_city_names[0]
        assert env.step(env.GRID_GARRISON_HIRE_ACTION_START)[4]['garrison_hired']
        env.city_garrisons[city][0]['exp_current'] = 94
        for built in (False, True):
            if built:
                build_portal(env)
            attacker(env)
            env._start_city_defence(city)
            _, _, info = finish_battle(env)
            guard = env.city_garrisons[city][0]
            assert guard['name'] == ('Берсерк' if built else 'Одержимый')
            assert guard['exp_current'] == (0 if built else 94)
            assert info['garrison_unit_upgrades'] == int(built)
    finally:
        env.close()


def test_scripted_bot_city_entry_routes_to_garrison_growth():
    env = CampaignEnv(
        Realcapital=2, observation_version='local5', scripted_capital_bot_enabled=False,
        use_boss_starting_roster=False, log_enabled=False)
    env.reset(seed=42)
    try:
        env.gold = 10000
        city = env.garrison_city_names[0]
        for enemy_id in env.legions_settlement_territory_source_by_name[city]['required_enemy_ids']:
            env.grid_env.mark_enemy_defeated(enemy_id)
        city_tile = env.legions_settlement_source_tile_by_name[city]
        env.grid_env.agent_pos = city_tile
        env._capture_cities_on_hero_entry(0, {})
        env.legions_settlement_level_by_name[city] = 5
        env.grid_env.agent_pos = tuple(env.CASTLE_POS)
        build_portal(env)
        assert env.step(env.GRID_GARRISON_HIRE_ACTION_START)[4]['garrison_hired']
        env.city_garrisons[city][0]['exp_current'] = 94
        hero = deepcopy(env.blue_team_state)
        attacker(env)
        # Enable only this focused scripted-attack integration, no training.
        env.scripted_capital_bot_enabled = True
        env.scripted_capital_bot_state = 'hunting'
        env.scripted_capital_bot_position = (city_tile[0] - 1, city_tile[1])
        _, _, terminated, truncated, info = env.step(8)
        assert not (terminated or truncated)
        assert info['battle_triggered']
        assert env.current_battle_context == {'kind': 'city_garrison', 'city': city}
        _, _, info = finish_battle(env)
        assert info['garrison_unit_upgrades'] == 1
        assert env.city_garrisons[city][0]['name'] == 'Берсерк'
        assert env.blue_team_state == hero
        assert env._garrison_player_owns(city)
    finally:
        env.close()


@pytest.mark.parametrize('winner', ('red', None))
def test_other_outcomes_do_not_apply_stale_garrison_levelup_events(winner):
    env = make_env()
    try:
        city = env.garrison_city_names[0]
        build_portal(env)
        assert env.step(env.GRID_GARRISON_HIRE_ACTION_START)[4]['garrison_hired']
        attacker(env)
        env._start_city_defence(city)
        battle = env.battle_env
        guard = next(unit for unit in battle.combined if unit['name'] == 'Одержимый')
        guard['exp_current'] = 94
        battle._apply_exp_award_to_unit(guard, 1)
        assert battle.last_levelups
        battle.winner = winner
        if winner is None:
            battle.step = lambda action: (battle._obs(), 0, False, True, {})
        hero = deepcopy(env.blue_team_state)
        _, _, terminated, truncated, info = env.step(0)
        assert not truncated
        assert terminated == (winner == 'red')  # This map ends when its city is lost.
        assert info['garrison_unit_upgrades'] == 0
        assert guard['name'] == 'Одержимый'
        assert env.blue_team_state == hero
        assert info['battle_result'] == ('defeat' if winner == 'red' else 'timeout')
    finally:
        env.close()
