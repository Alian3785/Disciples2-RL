"""CUDA contracts: the player rearranges the hero's party on the map (docs/UNITS.md).

Python reference campaign_env_economy.py: _build_blue_unit_swap_pairs,
_can_swap_blue_unit_positions, _step_swap_blue_unit_positions,
_step_swap_big_blue_unit_positions; three swaps per turn, -0.004 per swap.
"""
import dataclasses
from itertools import combinations

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, REST, NumberGrid, action_names, load_map
from stoix.envs.number_grid_lords import with_starting_party
from stoix.envs.number_grid_unit_swaps import (
    MAX_PER_TURN, PENALTY, SLOT_FIELDS, UNITS_SWAPPED, swap_pairs, unit_swap_action_names,
)
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state

DEFAULT = load_map('default')
# Default starts with the reference party of a warrior ruler (docs/UNITS.md); since
# map v88 the current map starts with a mage, whose occupied slots come first in its own order.
CURRENT_PARTY = with_starting_party(MAP)['hero_roster']
WARRIOR_PARTY = with_starting_party({**MAP, 'lord_type': 'warrior'})['hero_roster']
DEFAULT_PARTY = with_starting_party(DEFAULT)['hero_roster']


def test_reference_layout_and_the_disabled_flag():
    # Occupied starting positions 7, 8, 9, 11 first, then 10 and 12 with every included position.
    assert swap_pairs(DEFAULT_PARTY) == swap_pairs(WARRIOR_PARTY) == (
        (0, 1), (0, 2), (0, 4), (1, 2), (1, 4), (2, 4), (0, 3), (1, 3), (2, 3), (3, 4),
        (0, 5), (1, 5), (2, 5), (3, 5), (4, 5))
    assert swap_pairs(CURRENT_PARTY)[:6] == ((0, 2), (0, 3), (0, 5), (2, 3), (2, 5), (3, 5))
    for roster in (CURRENT_PARTY, WARRIOR_PARTY, DEFAULT_PARTY, ['titan', None, None, None, None, None], [None]*6):
        assert sorted(swap_pairs(roster)) == list(combinations(range(6), 2))
    names = action_names(MAP)
    assert names[-15:] == unit_swap_action_names(with_starting_party(MAP)) == tuple(
        f'swap_{a}_{b}' for a, b in swap_pairs(CURRENT_PARTY))
    # Without the flag the map has no swap actions or features at all, not even masked ones.
    off = {**MAP, 'unit_swaps': False}
    assert action_names(off) == names[:-15] and not any(n.startswith('swap_') for n in action_names(off))
    env = NumberGrid(map_config=off)
    assert (env.num_actions, env.observation_size) == (229, 1827) and not hasattr(env, 'unit_swaps')
    with pytest.raises(ValueError, match='unit_swaps'):
        NumberGrid(map_config={**MAP, 'unit_swaps': 1})
    with pytest.raises(ValueError, match='version 42'):
        NumberGrid(map_config={**MAP, 'battle_observation_version': 41})


@pytest.mark.core
def test_every_slot_field_moves_with_its_unit(current_game, default_game):
    for env, state in (current_game[:2], default_game[:2]):
        slots = {f.name for f in dataclasses.fields(state) if getattr(state, f.name).shape[:1] == (12,)}
        # last_item_loot counts item kinds: twelve on Default by chance.
        assert set(SLOT_FIELDS) <= slots and slots - set(SLOT_FIELDS) <= {'last_item_loot'}


@pytest.mark.core
def test_swaps_through_step_limit_and_rest(current_game):
    env, initial, step, _ = current_game
    rules = env.unit_swaps
    mask, observe = compiled_method(env, 'action_mask'), compiled_method(env, 'observation')
    valid = jax.jit(rules.valid)
    action = {pair: rules.start+i for i, pair in enumerate(rules.pairs)}
    ids = initial.unit_ids[:6].tolist()  # possessed, duke, possessed, cultist, empty, empty

    def check(state):
        """The mask and the step's single-action check agree on every swap action."""
        m = mask(state)
        for a in range(rules.start, rules.end):
            assert bool(valid(state, jnp.int32(a))) == bool(m[a])
        return {pair: bool(m[a]) for pair, a in action.items()}

    # A permanent potion, a ward spell and wounds follow their unit.
    state = initial.replace(potion_bonus=initial.potion_bonus.at[0, 1].set(7.),
                            spell_wards=initial.spell_wards.at[3].set(5), hp=initial.hp.at[0].set(1))
    assert check(state) == {pair: pair != (4, 5) for pair in rules.pairs}  # two empty slots never swap
    swapped, ts = step(state, jnp.int32(action[0, 3]))
    index = rules.pairs.index((0, 3))
    assert swapped.unit_ids[:6].tolist() == [ids[3], ids[1], ids[2], ids[0], 0, 0]
    assert swapped.hp[3] == 1 and swapped.hp[0] == state.hp[3] and swapped.potion_bonus[3, 1] == 7
    assert swapped.spell_wards[0] == 5 and swapped.spell_wards[3] == 0
    assert float(ts.reward) == pytest.approx(-PENALTY-env.step_cost)
    assert swapped.last_event == UNITS_SWAPPED and swapped.last_swap == index
    assert swapped.unit_swaps_today == 1 and swapped.unit_swaps_used == 1 << index
    assert swapped.movement_points == state.movement_points and swapped.position.tolist() == state.position.tolist()
    usable = check(swapped)
    assert not usable[0, 3] and usable[1, 3]  # each pair once per turn
    obs = observe(swapped)
    assert obs.shape == (env.observation_size,) and obs[-16] == pytest.approx(1/3) and obs[-15+index] == 1

    # A unit moves into an empty slot; then the formerly empty pair is open.
    moved, _ = step(swapped, jnp.int32(action[1, 4]))
    assert moved.unit_ids[1] == 0 and moved.unit_ids[4] == ids[1] and moved.hp[1] == 0
    assert check(moved)[4, 5]
    third, _ = step(moved, jnp.int32(action[4, 5]))
    assert third.unit_ids[5] == ids[1] and third.unit_swaps_today == MAX_PER_TURN
    # Three swaps per turn: everything is masked and the step rejects a fourth.
    assert not any(check(third).values())
    rejected, ts = step(third, jnp.int32(action[0, 1]))
    assert (rejected.unit_ids == third.unit_ids).all() and float(ts.reward) == pytest.approx(-env.step_cost)
    # Rest starts the next turn: the count and the used pairs reset.
    rested, _ = step(third, jnp.int32(REST))
    assert rested.unit_swaps_today == 0 and rested.unit_swaps_used == 0 and check(rested)[0, 3]
    # Not in battle.
    assert not jax.jit(rules.available)(initial.replace(in_battle=jnp.bool_(True))).any()


def test_a_big_unit_moves_its_whole_column(current_game):
    env, initial, _, _ = current_game
    rules = env.unit_swaps
    available, apply = jax.jit(rules.available), jax.jit(rules.apply)
    ids = env.progression.ids
    # The titan stands in front and covers the back cell of column 0.
    state = hero_roster_state(env, initial, ['titan', 'possessed', None, None, 'cultist', None])
    usable = dict(zip(rules.pairs, available(state).tolist()))
    assert not usable[0, 3]  # one unit
    assert usable[0, 1] and usable[3, 4] and usable[0, 2] and usable[3, 5]  # whole columns
    assert usable[1, 4] and not usable[2, 5]  # ordinary units; two empty cells

    def swap(pair):
        after, reward = apply(state, jnp.int32(rules.start+rules.pairs.index(pair)), jnp.bool_(False))
        assert float(reward) == pytest.approx(-PENALTY)
        return after.unit_ids[:6].tolist(), after.hp[:6].tolist()

    titan_hp, possessed_hp, cultist_hp = (int(state.hp[i]) for i in (0, 1, 4))
    # From the back cell too: the units of column 1 keep their rows, the titan stays in front.
    for pair in ((0, 1), (3, 4), (0, 4)):
        assert swap(pair) == ([ids['possessed'], ids['titan'], 0, ids['cultist'], 0, 0],
                              [possessed_hp, titan_hp, 0, cultist_hp, 0, 0])
    assert swap((3, 5))[0] == [0, ids['possessed'], ids['titan'], 0, ids['cultist'], 0]


def test_default_map_and_manual_game(default_game, human_service):
    """Default's swap actions move the hero's equipment role; the manual game uses the same pairs."""
    env, initial, step, mask = default_game
    rules = env.unit_swaps
    assert rules.start == 291 and rules.pairs == swap_pairs(DEFAULT_PARTY)
    hero, cultist = int(initial.unit_ids[1]), int(initial.unit_ids[4])  # the hero has its own skill row
    assert env.equipment_heroes[hero] and env._equipment_hero(initial) == 1
    swapped, ts = step(initial, jnp.int32(rules.start+rules.pairs.index((1, 4))))
    assert swapped.unit_ids[4] == hero and swapped.unit_ids[1] == cultist and env._equipment_hero(swapped) == 4
    assert float(ts.reward) == pytest.approx(-PENALTY-env.step_cost)
    assert env.movement_cap(swapped) == env.movement_cap(initial) and not mask(swapped)[rules.start+rules.pairs.index((1, 4))]
    # The manual game shows the same pairs and applies them through the same JAX step.
    game = human_service.create(42, 'legions')
    meta = game['unit_swaps']
    assert meta['limit'] == MAX_PER_TURN and meta['pairs'][0] == [0, 1] and len(meta['pairs']) == 15
    result = human_service.act(game['session'], meta['start']+meta['pairs'].index([0, 3]))
    state = result['snapshot']['state']
    assert state['last_event'] == UNITS_SWAPPED and state['unit_swaps_today'] == 1
