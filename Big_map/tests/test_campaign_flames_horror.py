import pytest
from battle_env import FIRST_HERO_ITEM_ENEMY_ACTION_START
from test_campaign_original_item_aliases import AliasEnv
from test_campaign_battle_equipment import _set_battle_hero_turn
from data_dicts_compact_lines import DATA, map_unit_to_battle


def _equipped_campaign(original):
    env = AliasEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2, use_boss_starting_roster=False)
    env.reset(seed=123)
    canonical = env._canonical_item_name(original)
    assert env.BATTLE_EQUIPPABLE_ITEM_NAMES.count(canonical) == 1
    env._grant_hero_item_reward(original)
    assert env._count_hero_item(canonical) == 1
    env._resolve_travel_hero()['hero_abilities'] = [env.HERO_SORCERY_LORE_ABILITY_KEY]
    action = env.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + env.BATTLE_EQUIPPABLE_ITEM_NAMES.index(canonical)
    assert env.compute_action_mask()[action]
    assert env.step(action)[4]['equip_battle_item_applied']
    _start_battle(env)
    return env


def _start_battle(env):
    env.current_enemy_id = 1
    source = next(u for u in DATA if u['кто'] == 'Гоблин')
    env.enemy_team_states[1] = [map_unit_to_battle(source, 'red', pos) for pos in (1, 2, 3)]
    env._init_battle(enemy_id=1)
    env.mode = env.MODE_BATTLE
    env.battle_env._advance_until_blue_turn = lambda: None
    _set_battle_hero_turn(env)


@pytest.mark.parametrize('original', ['Orb of Flames', 'g000ig9041'])
def test_flames_orb_pickup_equip_mass_burn_tick_and_consumption(original):
    env = _equipped_campaign(original)
    battle = env.battle_env
    red = [u for u in battle.combined if u['team'] == 'red' and battle._alive(u)]
    assert len(red) > 1
    for unit in red:
        unit['health'] = unit['hp'] = 200
        unit['immunity'] = unit['resistance'] = []
    red[-1]['immunity'] = ['Fire']
    assert env._hero_item_gold_value_by_name(original) == 1000
    assert battle.compute_action_mask()[FIRST_HERO_ITEM_ENEMY_ACTION_START]
    info = env.step(FIRST_HERO_ITEM_ENEMY_ACTION_START)[4]
    assert info['battle_hero_item_consumed']
    assert env._count_hero_item(original) == 0
    assert env.equipped_hero_items[0] is None
    assert all(u.get('burn_damage_per_tick') == 15 for u in red[:-1])
    assert all(u['health'] == 200 for u in red)
    assert not red[-1].get('burn_turns_left', 0)
    battle._apply_start_of_turn_effects(red[0])
    assert red[0]['health'] == 185


def test_horror_talisman_fear_five_charges_and_persistence_between_battles():
    env = _equipped_campaign('G000IG9141')
    assert env._hero_item_gold_value_by_name('Talisman of Horror') == 600
    assert env.equipped_hero_item_uses_left[0] == 5
    target = next(u for u in env.battle_env.combined if u['team'] == 'red' and u['position'] == 1)
    target['immunity'] = target['resistance'] = []
    info = env.step(FIRST_HERO_ITEM_ENEMY_ACTION_START)[4]
    assert info['battle_hero_item_effect_kind'] == 'fear'
    assert target['running_away'] == 1 and not target.get('paralyzed', 0)
    assert env.equipped_hero_item_uses_left[0] == 4
    for remaining in (3, 2, 1, 0):
        _start_battle(env)
        assert env.battle_env.equipped_hero_item_uses_left[0] == remaining + 1
        target = next(u for u in env.battle_env.combined if u['team'] == 'red' and u['position'] == 1)
        target['immunity'] = ['Mind']
        _set_battle_hero_turn(env)
        info = env.step(FIRST_HERO_ITEM_ENEMY_ACTION_START)[4]
        assert info['battle_hero_item_charge_spent']
        assert info['battle_hero_item_effect_value'] == 0
        assert info['battle_hero_item_uses_left'] == remaining
        assert env._count_hero_item('Talisman of Horror') == int(remaining > 0)
        _set_battle_hero_turn(env)
        assert not env.battle_env.compute_action_mask()[FIRST_HERO_ITEM_ENEMY_ACTION_START]
    assert env.equipped_hero_items[0] is None
    _start_battle(env)
    assert not env.battle_env.compute_action_mask()[FIRST_HERO_ITEM_ENEMY_ACTION_START]
