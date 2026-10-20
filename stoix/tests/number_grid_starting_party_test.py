"""Reference starting hero and party by faction and ruler, and the hero's own allowance; CUDA only."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, NumberGrid, REST, action_names, load_map
from stoix.envs.number_grid_combat import UNITS
from stoix.envs.number_grid_lords import STARTING_PARTIES, with_starting_party
from stoix.envs.number_grid_unit_swaps import swap_pairs, unit_swap_action_names
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state

ROOT = Path(__file__).resolve().parents[2]
SOURCES = json.loads((ROOT/'docs/validation/starting-party-sources.json').read_text(encoding='utf-8'))
RANGER = 'g000uu0020'  # Рейнджер людей, 35 movement points


@pytest.mark.core
def test_parties_match_reference_table_and_hero_allowances():
    assert set(STARTING_PARTIES) == set(SOURCES['parties'])
    for faction, parties in STARTING_PARTIES.items():
        assert set(parties) == {'warrior', 'mage', 'guildmaster'}
        for lord, roster in parties.items():
            names = [UNITS[key]['name'] if key else None for key in roster]
            assert names == SOURCES['parties'][faction][lord], (faction, lord)
            assert all(UNITS[key]['faction'] == faction for key in roster if key)
            heroes = [slot for slot, key in enumerate(roster) if key and UNITS[key].get('hero')]
            assert heroes == [1 if lord == 'warrior' else 3]
            hero = UNITS[roster[heroes[0]]]
            assert hero['movement'] == SOURCES['hero_movement'][hero['name']]


def test_maps_take_the_rulers_party_unless_the_roster_is_fixed():
    for game_map in (MAP, load_map('default')):
        assert 'hero_roster' not in game_map and 'hero_units' not in game_map
        for faction, parties in STARTING_PARTIES.items():
            for lord, roster in parties.items():
                resolved = with_starting_party({**game_map, 'faction': faction, 'lord_type': lord})
                assert resolved['hero_roster'] == roster and resolved['hero_units'] == 4
                names = action_names({**game_map, 'faction': faction, 'lord_type': lord})
                assert names[-15:] == unit_swap_action_names(resolved)
    fixed = ['possessed', 'duke', 'possessed', 'cultist', 'cultist', None]
    resolved = with_starting_party({**MAP, 'lord_type': 'mage', 'hero_roster': fixed})
    assert resolved['hero_roster'] == fixed and resolved['hero_units'] == 5
    legacy = load_map(str(ROOT/'maps/number_grid-24x24-v9-warrior-24-squads.json'))
    assert with_starting_party(legacy) is legacy
    assert swap_pairs(with_starting_party({**MAP, 'lord_type': 'warrior'})['hero_roster'])[:6] == (
        (0, 1), (0, 2), (0, 4), (1, 2), (1, 4), (2, 4))
    with pytest.raises(ValueError, match='faction'):
        with_starting_party({**MAP, 'faction': 'unknown'})


def test_new_ruler_starts_with_its_own_hero_and_allowance():
    env = NumberGrid(map_config={**MAP, 'faction': 'undead_hordes', 'lord_type': 'guildmaster'})
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(0))
    rows = env.progression.rows
    assert [rows[i]['key'] if i else None for i in state.unit_ids[:6].tolist()] == \
        ['g000uu0086', None, 'g000uu0086', 'nosferatu', None, 'reference_g000uu0080']
    assert state.hp[:6].tolist() == [120, 0, 120, 90, 0, 45]
    assert state.movement_points == 35 and env.map_config['hero_units'] == 4
    turns = env.turn_metadata()
    assert turns['movement_points'] == 35 and turns['battle_entry_cost'] == 18
    assert turns['dead_hero_movement_points'] == 20
    # Two heroes in a fixed party: the larger allowance, as movement_cap takes it.
    two = NumberGrid(map_config={**MAP, 'hero_roster': ['possessed', 'duke', 'possessed', RANGER, None, None]})
    assert two.turn_metadata()['movement_points'] == 35 and two.turn_metadata()['battle_entry_cost'] == 18


@pytest.mark.core
def test_hero_allowance_sets_rest_refill_battle_entry_and_dead_fallback(current_game):
    env, initial, advance, _ = current_game
    cap = compiled_method(env, 'movement_cap')
    assert cap(initial) == 20 and env.turn_metadata()['battle_entry_cost'] == 10
    ranger = hero_roster_state(env, initial, ['possessed', None, 'possessed', RANGER, None, 'cultist'])
    assert cap(ranger) == 35
    assert cap(ranger.replace(hp=ranger.hp.at[3].set(0))) == 20  # reference MOVES_PER_TURN
    assert cap(ranger.replace(unit_levels=ranger.unit_levels.at[3].set(4))) == 38  # +1 per level 2-10
    rested, _ = advance(ranger.replace(movement_points=jnp.int32(0)), jnp.int32(REST))
    assert rested.day == 2 and rested.movement_points == 35
    near = ranger.replace(position=env.opponent_positions[0]-env.directions[2], movement_points=jnp.int32(35))
    attacked, _ = advance(near, jnp.int32(2))
    assert attacked.in_battle and attacked.movement_points == 35-18


@pytest.mark.core
def test_pathfinding_is_a_fifth_of_the_heros_allowance(default_game):
    env, initial, _, _ = default_game
    cap = compiled_method(env, 'movement_cap')
    assert cap(initial) == 20
    levelled = initial.replace(unit_levels=initial.unit_levels.at[1].set(2))
    assert cap(levelled) == 20+1+4
    ranger = hero_roster_state(env, initial, ['squire', None, 'squire', RANGER, None, 'acolyte'])
    assert cap(ranger) == 35
    assert cap(ranger.replace(unit_levels=ranger.unit_levels.at[3].set(2))) == 35+1+7
