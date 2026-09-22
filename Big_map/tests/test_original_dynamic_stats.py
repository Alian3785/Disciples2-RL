from pathlib import Path

import pytest

from battle_env import BattleEnv
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from unit_dynamic_stats import DYNAMIC_STAT_PROFILES, apply_dynamic_stat_growth
from test_battle_dynamic_levelups import _unit, _reach_level


def test_dragon_follows_original_early_and_late_growth():
    dragon = _unit('Зелёный дракон')
    battle = BattleEnv(log_enabled=False)
    for level in range(2, 16):
        _reach_level(battle, dragon)
        early, late = min(level - 1, 9), max(0, level - 10)
        assert dragon['max_health'] == 600 + 60 * early + 30 * late
        assert dragon['damage'] == 60 + 6 * early + 3 * late
        assert dragon['accuracy'] == 75 + early
    assert (dragon['max_health'], dragon['damage']) == (1290, 129)


def test_healer_and_gargoyle_use_heal_and_armor_fields():
    battle = BattleEnv(log_enabled=False)
    healer = _unit('Патриарх')
    _reach_level(battle, healer)
    _reach_level(battle, healer)
    assert (healer['max_health'], healer['damage']) == (155, 144)
    gargoyle = _unit('Ониксовая гаргулья')
    before = gargoyle['armor']
    _reach_level(battle, gargoyle)
    assert gargoyle['armor'] == before + 1


def test_levelup_during_buffed_battle_survives_save_without_accumulating_buffs():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2, use_boss_starting_roster=False)
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    dragon = _unit('Зелёный дракон')
    env.blue_team_state[0] = dragon
    env._active_potion_positions(env.BARK_POTION_ITEM_NAME).add(7)
    env._active_potion_positions(env.STRENGTH_POTION_CANONICAL_ITEM_NAME).add(7)
    env._apply_support_spell_to_blue_stack('elf_d2_s020')  # temporary +50 HP
    env._init_battle(1)
    grown = next(u for u in env.battle_env.combined if u.get('team') == 'blue' and u['position'] == 7)
    env.battle_env._apply_exp_award_to_unit(grown, grown['exp_required'])
    env._save_blue_state()
    saved = env.blue_team_state[0]
    assert (saved['Level'], saved['max_health'], saved['damage'], saved['armor'], saved['accuracy']) == (2, 660, 66, 0, 76)
    env._advance_turns(1)
    env._init_battle(1)
    saved = next(u for u in env.battle_env.combined if u.get('team') == 'blue' and u['position'] == 7)
    assert (saved['max_health'], saved['damage'], saved['armor']) == (660, 66, 0)


def test_every_named_environment_template_has_a_profile():
    assert len(DYNAMIC_STAT_PROFILES) == 356
    for row in DATA:
        unit = map_unit_to_battle(row, 'blue', 7)
        assert unit['unit_id'] in DYNAMIC_STAT_PROFILES, unit['name']


def test_export_matches_all_original_dbf_profiles():
    from tools.inspect_sg_map import parse_dbf_rows
    root = Path(r'C:\Program Files (x86)\Steam\steamapps\common\Disciples II Rise of the Elves\Globals')
    if not root.exists():
        pytest.skip('Original game is not installed')
    upgrades = {r['UPGRADE_ID'].lower(): r for r in parse_dbf_rows(root / 'GDynUpgr.DBF')}
    fields = ('HIT_POINT', 'DAMAGE', 'HEAL', 'ARMOR', 'POWER', 'INITIATIVE')
    for row in parse_dbf_rows(root / 'Gunits.dbf'):
        profile = DYNAMIC_STAT_PROFILES[row['UNIT_ID'].lower()]
        assert profile[0] == int(row['DYN_UPG_LV'])
        for index, key in enumerate(('DYN_UPG1', 'DYN_UPG2'), 1):
            assert profile[index] == tuple(int(upgrades[row[key].lower()][f] or 0) for f in fields)


def test_unknown_id_is_not_silently_assigned_another_units_profile():
    unit = {'unit_id': 'custom', 'name': 'Зелёный дракон', 'max_health': 100}
    assert not apply_dynamic_stat_growth(unit, 2)
    assert unit['max_health'] == 100
