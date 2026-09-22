"""Regression cases verified against the installed Rise of the Elves DBFs."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from enemy_configs import ENEMY_CONFIGS


@pytest.mark.parametrize('name,field,expected', [
    ('Дева рощи', 'max_health', 85),  # Gunits g000uu8033, not summoned g000uu8133.
    ('Сильфида', 'exp_required', 2470),  # Gunits g000uu8035.
    ('Мизраэль', 'initiative', 90),  # Gattacks g000aa3001.
])
def test_corrected_unit_values_reach_battle_templates(name, field, expected):
    raw = next(u for u in DATA if u['кто'] == name)
    assert map_unit_to_battle(raw, 'blue', 7)[field] == expected


@pytest.mark.parametrize('spell_id,name', [
    ('g000ss0066', 'Темный энт'),
    ('g000ss0098', 'Малый энт'),
    ('g000ss0108', 'Великий энт'),
    ('g000ss0117', 'Буйный энт'),
])
def test_ent_scroll_consumes_item_starts_battle_and_preserves_hero(spell_id, name):
    env = CampaignEnv(map_name='scroll_train', log_enabled=False, persist_blue_hp=False)
    env.reset(seed=123)
    before_team = deepcopy(env.blue_team_state)
    before_mana = env._current_mana_totals()
    index, entry = next((i, e) for i, e in enumerate(env.scroll_cast_slot_entries())
                        if e['spell_id'] == spell_id)
    action = env.grid_scroll_cast_action_start + index
    assert env.compute_action_mask()[action]
    _, _, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert info['spell_cast_executed'] and info['scroll_item_consumed']
    assert env.count_scroll_item(entry['item_name']) == 0
    assert env._current_mana_totals() == before_mana
    assert env.mode == env.MODE_BATTLE
    assert env.current_battle_context['kind'] == 'summon_spell'
    summoned = [u for u in env.battle_env.combined
                if u.get('team') == 'blue' and u.get('max_health', 0) > 0]
    assert len(summoned) == 1
    assert summoned[0]['name'] == name
    assert env.blue_team_state == before_team


@pytest.mark.parametrize('item_name', ['Potion of Might', 'Might Potion', 'Зелье мощи'])
def test_might_alias_consumes_inventory_and_buffs_battle_damage_by_fifty_percent(item_name):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    unit = next(u for u in env.blue_team_state if u['position'] == 8)
    base_damage = unit['damage']
    assert base_damage > 0
    action_count = env.action_space.n
    env.heroitems = [item_name]
    action = env.GRID_ENERGY_ACTION_START + env.ENERGY_ELIXIR_POSITIONS.index(8)
    assert env.compute_action_mask()[action]
    _, _, _, _, info = env.step(action)
    assert info['combat_potion_applied'] and info['combat_potion_consumed']
    assert env._count_hero_item(env.ENERGY_ELIXIR_ITEM_NAME) == 0
    assert env.action_space.n == action_count
    assert env._potion_item_definition('Potion of Strength')['effect']['multiplier'] == 1.30
    env._init_battle(enemy_id=1)
    battle_unit = next(u for u in env.battle_env.combined
                       if u.get('team') == 'blue' and u['position'] == 8)
    assert battle_unit['damage'] == round(base_damage * 1.50)
    env._save_blue_state()
    assert next(u for u in env.blue_team_state if u['position'] == 8)['damage'] == base_damage


# Gspells.CASTING_C for g000ss0085/0087/0055/0056/0091.
@pytest.mark.parametrize('capital,spell_key,expected', [
    (3, 'mcl_d2_s015', {'hell': 0, 'life': 225, 'death': 0, 'rune': 225, 'nature': 0}),
    (3, 'mcl_d2_s020', {'hell': 0, 'life': 200, 'death': 200, 'rune': 400, 'nature': 0}),
    (2, 'lod_d2_s016', {'hell': 200, 'life': 100, 'death': 100, 'rune': 0, 'nature': 0}),
    (2, 'lod_d2_s017', {'hell': 200, 'life': 100, 'death': 100, 'rune': 0, 'nature': 0}),
    (2, 'lod_d2_s020', {'hell': 300, 'life': 0, 'death': 150, 'rune': 150, 'nature': 0}),
])
def test_corrected_cast_cost_controls_mask_and_spends_exact_mana(capital, spell_key, expected):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=capital,
                      use_boss_starting_roster=False)
    env.reset(seed=123)
    env.typeoflord = 2
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {10: (4, 3), 20: (10, 10)}
    env.grid_env.enemies_alive = {10: True, 20: True}
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[20] = deepcopy(ENEMY_CONFIGS[20])
    env.active_spells[spell_key]['learned'] = 1
    env.moves = 1
    costs = {env.SPELL_COST_MANA_KIND_BY_SUFFIX[k]: v for k, v in expected.items()}
    assert env._get_spell_use_costs(env.active_spells[spell_key]) == costs
    for kind, value in costs.items():
        setattr(env, env.MANA_ATTR_BY_KIND[kind], float(value))
    specs_and_starts = [
        (env._current_map_offensive_spell_action_specs(), env.grid_legion_damage_spell_action_start),
        (env._current_map_support_spell_action_specs(), env.grid_map_support_spell_action_start),
    ]
    action = next(start+i for specs, start in specs_and_starts
                  for i, spec in enumerate(specs) if spec['id'] == spell_key)
    for kind, value in costs.items():
        if value:
            setattr(env, env.MANA_ATTR_BY_KIND[kind], float(value-1))
            assert not env.compute_action_mask()[action]
            setattr(env, env.MANA_ATTR_BY_KIND[kind], float(value))
    assert env.compute_action_mask()[action]
    _, _, _, _, info = env.step(action)
    assert info['spell_cast_executed']
    for kind in costs:
        assert getattr(env, env.MANA_ATTR_BY_KIND[kind]) == pytest.approx(0)
