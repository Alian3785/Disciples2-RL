"""Defender-only fortification integration coverage.

Incoming waves advance through natural public REST steps. Capital attacks use
positioned hero fixtures with public movement steps; city ownership and reserve
battles use prepared fixtures. These check battle entry, not full-map navigation.
"""
from copy import deepcopy
import pytest
from campaign_env import CampaignEnv
from campaign_env_data import (
    SETTLEMENT_ARMOR_BONUS_BY_LEVEL,
    SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID,
)


def make(name):
    e = CampaignEnv(map_name=name, observation_version='local5',
                    scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
                    log_enabled=False)
    e.reset(seed=42)
    # Make blue move first so the armor check precedes enemy spell/status actions.
    for u in e.blue_team_state:
        if e._unit_max_hp(u) > 0:
            u.update(initiative=10000, initiative_base=10000)
    return e


def fighters(e, team):
    return [u for u in e.battle_env.combined if u['team'] == team and u['name'] != 'пусто']


def expected_armors(units):
    return {u['position']: int(u.get('armor', 0)) for u in units if u['name'] != 'пусто'}


def assert_team(e, team, originals, bonus):
    actual = fighters(e, team)
    assert actual
    for u in actual:
        assert u.get('settlement_armor_bonus', 0) == bonus, (team, u['name'], u)
        assert u['armor'] == originals[u['position']] + bonus, (team, u['name'], originals, u)


@pytest.mark.parametrize('name', ['siege_train', 'super_last_stand'])
def test_natural_wave_attacks_capital(name):
    e = make(name)
    try:
        blue = expected_armors(e.blue_team_state)
        for turn in range(100):
            result = e.step(8)
            assert not any(result[2:4]), result[4]
            if e.mode == e.MODE_BATTLE:
                break
        else:
            pytest.fail('No wave reached capital within 100 natural rests')
        assert tuple(e.grid_env.agent_pos) == tuple(e.CASTLE_POS)
        assert_team(e, 'blue', blue, 40)
        assert_team(e, 'red', expected_armors(e.enemy_team_states[e.current_enemy_id]), 0)
    finally:
        e.close()


@pytest.mark.parametrize('enemy_id', [5, 10])
@pytest.mark.parametrize('entry', ['same_tile', 'adjacent'])
def test_attack_wotan_enemy_capital_from_public_step(enemy_id, entry):
    e = make('wotans_retribution')
    try:
        target = tuple(e.empire_territory_source_tile)
        assert target == (13, 33)
        # Isolate each legitimate co-located defender without changing its location.
        for eid in e.grid_env.enemies_alive:
            e.grid_env.enemies_alive[eid] = eid == enemy_id
        e.grid_env.agent_pos = (target[0] + (1 if entry == 'same_tile' else 2), target[1])
        e.moves = 100
        blue = expected_armors(e.blue_team_state)
        red = expected_armors(e.enemy_team_states[enemy_id])
        result = e.step(2)
        assert e.mode == e.MODE_BATTLE, result[4]
        assert e.current_enemy_id == enemy_id
        assert not any(result[2:4])
        assert_team(e, 'blue', blue, 0)
        assert_team(e, 'red', red, 40)
    finally:
        e.close()


def test_player_leaves_capital_to_attack_field_stack():
    e = make('siege_train')
    try:
        target = (e.CASTLE_POS[0] + 1, e.CASTLE_POS[1])
        enemy_id = 1
        e.grid_env.enemy_positions[enemy_id] = target
        blue = expected_armors(e.blue_team_state)
        red = expected_armors(e.enemy_team_states[enemy_id])
        result = e.step(3)
        assert e.mode == e.MODE_BATTLE, result[4]
        assert e.battle_origin_pos == e.CASTLE_POS
        assert_team(e, 'blue', blue, 0)
        assert_team(e, 'red', red, 0)
    finally:
        e.close()


def test_natural_wave_defends_captured_city():
    e = make('super_last_stand')
    try:
        city = next(iter(e.legions_settlement_source_tile_by_name))
        tile = e.legions_settlement_source_tile_by_name[city]
        for eid in e.legions_settlement_territory_source_by_name[city]['required_enemy_ids']:
            e.grid_env.mark_enemy_defeated(eid)
        e.grid_env.agent_pos = tuple(tile)
        e._capture_cities_on_hero_entry(0, {})
        assert city in e.legions_active_settlement_territory_capture_turn_by_name
        blue = expected_armors(e.blue_team_state)
        for _ in range(100):
            result = e.step(8)
            assert not any(result[2:4]), result[4]
            if e.mode == e.MODE_BATTLE:
                break
        else:
            pytest.fail('No wave reached captured city within 100 natural rests')
        bonus = SETTLEMENT_ARMOR_BONUS_BY_LEVEL[e.legions_settlement_level_by_name[city]]
        assert_team(e, 'blue', blue, bonus)
        assert_team(e, 'red', expected_armors(e.enemy_team_states[e.current_enemy_id]), 0)
    finally:
        e.close()


def test_captured_city_reserve_defense_bonus_is_temporary():
    e = make('city_defence_train')
    try:
        city = next(iter(e.legions_settlement_source_tile_by_name))
        e.gold = 10000
        result = e.step(e.GRID_GARRISON_HIRE_ACTION_START)
        assert result[4]['garrison_hired']
        # Construct a focused attacker fixture without enabling the scripted bot.
        e.scripted_capital_bot_team_state = e._build_battle_team_with_placeholders(
            'blue', [e._build_unit_from_data(e._find_unit_data_by_name('Гном'), 'blue', 8)])
        for u in e.city_garrisons[city]:
            u.update(initiative=10000, initiative_base=10000)
        blue = expected_armors(e.city_garrisons[city])
        red = {p-6: a for p,a in expected_armors(e.scripted_capital_bot_team_state).items()}
        hero_before = deepcopy(e.blue_team_state)
        e._start_city_defence(city)
        bonus = SETTLEMENT_ARMOR_BONUS_BY_LEVEL[e._garrison_capacity(city)]
        for u in fighters(e, 'blue'):
            assert u['armor'] == min(90, blue[u['position']] + bonus)
        for u in fighters(e, 'red'):
            assert u['armor'] == red[u['position']]
        e.battle_env.winner = 'blue'
        result = e.step(0)
        assert e.blue_team_state == hero_before
        assert expected_armors(e.city_garrisons[city]) == blue
        assert e.mode == e.MODE_GRID
    finally:
        e.close()

@pytest.mark.parametrize('name', [
    'default', 'green_dragon_minimal', 'hire_train', 'small',
    'super_last_stand', 'wotans_retribution',
])
def test_all_static_city_defenders_keep_scaled_source_bonus(name):
    e = make(name)
    try:
        levels = getattr(e, 'SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID',
                         SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID)
        for enemy_id, level in levels.items():
            e.battle_origin_pos = tuple(e.CASTLE_POS)
            e.current_battle_context = {'kind': 'hero'}
            e._init_battle(enemy_id)
            assert_team(e, 'blue', expected_armors(e.blue_team_state), 0)
            assert_team(e, 'red', expected_armors(e.enemy_team_states[enemy_id]),
                        SETTLEMENT_ARMOR_BONUS_BY_LEVEL[level])
    finally:
        e.close()


def test_city_defender_has_no_portable_bonus_or_bonus_in_owned_city():
    e = make('wotans_retribution')
    try:
        eid = 6
        city = e.legions_settlement_territory_source_name_by_enemy_id[eid]
        source = e.grid_env.enemy_positions[eid]
        e.grid_env.enemy_positions[eid] = (source[0] + 1, source[1])
        e._init_battle(eid)
        assert_team(e, 'red', expected_armors(e.enemy_team_states[eid]), 0)
        e.grid_env.enemy_positions[eid] = source
        e.legions_active_settlement_territory_capture_turn_by_name[city] = e.turns
        e._init_battle(eid)
        assert_team(e, 'red', expected_armors(e.enemy_team_states[eid]), 0)
    finally:
        e.close()
