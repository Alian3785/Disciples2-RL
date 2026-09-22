from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from campaign_env import CampaignEnv
from scroll_spell_data import SCROLL_ITEM_DEFINITIONS
from test_campaign_terrain_movement import _entry_move_for_terrain, _set_travel_hero


CASES = [('forest', 'mcl_d2_s002', 'g000ss0022', 25, 4),
         ('water', 'mcl_d2_s007', 'g000ss0027', 50, 6)]


def _setup(terrain, book_id, scroll_id):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=3,
                      use_boss_starting_roster=False)
    env.reset(seed=123)
    _set_travel_hero(env, 'Герцог', flying=False)
    env.active_spells[book_id]['learned'] = 1
    env.scroll_magic_unlocked = True
    scroll = next(s for s in SCROLL_ITEM_DEFINITIONS if s['spell_id'] == scroll_id)
    env._static_chests = dict(env._static_chests)
    env._static_chests[(1, 1)] = (scroll['item_name'],)
    env._refresh_dynamic_action_layout()
    env.action_space = env.action_space.__class__(env.GRID_SWAP_UNIT_ACTION_START + env.GRID_SWAP_UNIT_ACTION_COUNT)
    env.heroitems = [scroll['item_name'], scroll['item_name']]
    start, move_action, tile = _entry_move_for_terrain(env, terrain)
    env.grid_env.agent_pos = start
    env.moves = 20
    book_action = next(env.grid_map_support_spell_action_start+i
                       for i,s in enumerate(env._current_map_support_spell_action_specs()) if s['id']==book_id)
    scroll_action = next(env.grid_scroll_cast_action_start+i
                         for i,s in enumerate(env.scroll_cast_slot_entries()) if s['spell_id']==scroll_id)
    return env, scroll, book_action, scroll_action, move_action, tile


@pytest.mark.parametrize('terrain,book_id,scroll_id,mana,normal_cost', CASES)
@pytest.mark.parametrize('source', ['book', 'scroll'])
def test_terrain_spell_moves_at_plain_cost_and_expires_next_day(terrain, book_id, scroll_id, mana, normal_cost, source):
    env, scroll, book_action, scroll_action, move_action, tile = _setup(terrain, book_id, scroll_id)
    assert env._campaign_tile_move_cost(tile) == normal_cost
    for kind in env.MANA_KIND_ORDER:
        setattr(env, env.MANA_ATTR_BY_KIND[kind], 0)
    rune_attr = env.MANA_ATTR_BY_KIND[env.SPELL_COST_MANA_KIND_BY_SUFFIX['rune']]
    if source == 'book':
        setattr(env, rune_attr, mana-1)
        assert not env.compute_action_mask()[book_action]
        setattr(env, rune_attr, mana)
    action = book_action if source == 'book' else scroll_action
    assert env.compute_action_mask()[action]
    _, _, _, _, info = env.step(action)
    assert info['spell_cast_executed']
    assert getattr(env, rune_attr) == 0
    assert env.count_scroll_item(scroll['item_name']) == (2 if source == 'book' else 1)
    assert env._blue_stack_spell_effect_summary()['ignored_terrain_penalties'] == [terrain]
    assert env._campaign_tile_move_cost(tile) == 2
    # The same effect cannot be charged again through either source.
    setattr(env, rune_attr, mana)
    assert not env.compute_action_mask()[book_action]
    assert not env.compute_action_mask()[scroll_action]
    _, _, _, _, repeated = env.step(scroll_action)
    assert not repeated['spell_cast_executed']
    assert env.count_scroll_item(scroll['item_name']) == (2 if source == 'book' else 1)
    assert getattr(env, rune_attr) == mana
    # Real grid movement consumes the reduced cost.
    moves_before = env.moves
    _, _, _, _, move_info = env.step(move_action)
    assert env.grid_env.agent_pos == tile
    assert move_info['move_points_spent'] == 2
    assert env.moves == moves_before-2
    env._advance_turns(1)
    assert not env._blue_stack_ignores_terrain_penalty(terrain)
    assert env._campaign_tile_move_cost(tile) == normal_cost
    assert env.compute_action_mask()[book_action]
    assert env.compute_action_mask()[scroll_action]


def test_terrain_spells_combine_without_affecting_other_stacks_or_surviving_reset():
    env, _, book_action, _, _, forest = _setup(*CASES[0][:3])
    _, _, water = _entry_move_for_terrain(env, 'water')
    env._apply_support_spell_to_blue_stack('mcl_d2_s002')
    assert env._campaign_tile_move_cost(water) == 6
    env._apply_support_spell_to_blue_stack('g000ss0027')
    assert env._campaign_tile_move_cost(forest) == 2
    assert env._campaign_tile_move_cost(water) == 2
    other_stack = deepcopy(env.blue_team_state)
    assert env._campaign_tile_move_cost(forest, units=other_stack, allow_boots=False, allow_spell_effects=False) == 4
    assert env._campaign_tile_move_cost(water, units=other_stack, allow_boots=False, allow_spell_effects=False) == 6
    assert env._blue_stack_spell_effect_summary()['ignored_terrain_penalties'] == ['forest', 'water']
    env.reset(seed=123)
    assert env._blue_stack_spell_effect_summary()['ignored_terrain_penalties'] == []


@pytest.mark.parametrize('terrain,book_id,scroll_id,mana,normal_cost', CASES)
def test_terrain_spell_preserves_dead_hero_penalty_and_does_not_stack_with_boots(terrain, book_id, scroll_id, mana, normal_cost):
    env, _, _, _, _, tile = _setup(terrain, book_id, scroll_id)
    boot = env.ELVEN_BOOTS_ITEM_NAME if terrain == 'forest' else env.BOOTS_OF_THE_ELEMENTS_ITEM_NAME
    env.heroitems.append(boot)
    env.equipped_boot_items = [boot]
    env._apply_support_spell_to_blue_stack(book_id)
    assert env._campaign_tile_move_cost(tile) == 2
    env._clear_all_blue_map_spell_effects()
    assert env._campaign_tile_move_cost(tile) == 2
    env._apply_support_spell_to_blue_stack(book_id)
    hero = env._resolve_travel_hero()
    hero['health'] = hero['hp'] = 0
    assert env._campaign_tile_move_cost(tile) == 4
