"""Fog geometry and information boundaries in the NumberGrid policy input."""

import jax
import jax.numpy as jnp
import numpy as np

from stoix.envs.number_grid import NumberGrid, load_map
from stoix.envs.number_grid_fog import FogRules, revealed_at


def test_fog_geometry_at_hero_and_capital():
    capital = [[r, c] for r in range(11, 16) for c in range(9, 14)]
    rules = FogRules(dict(size=50, agent_position=[13, 13], capital_footprint=capital))
    assert int(rules.capital.sum()) == 285
    position = jnp.array([30, 30], jnp.int32)
    hero = rules.area(position, position, 5)
    assert int(hero.sum()) == 117
    assert not hero[25, 25] and hero[25, 26] and hero[35, 34]
    assert int(rules.area(position, position, 6).sum()) == 165
    edge = rules.area(jnp.array([1, 1]), jnp.array([1, 1]), 5)
    assert int(edge.sum()) == 35 and not edge[0].any() and not edge[:, 0].any()


def test_hidden_armies_do_not_change_observations_or_spell_targets():
    env = NumberGrid(map_config=load_map('default'))
    rules = env.casting
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert ts.observation.shape == env.observation_space().shape
    known = np.asarray(revealed_at(state, state.enemy_positions))
    assert np.any(known & np.asarray(state.alive))
    hidden = int(np.flatnonzero(~known & np.asarray(state.alive) & np.asarray(rules.field))[0])
    spell_ids = ('g000ss0041', 'g000ss0043')  # Hellhound summon and Ignis Mare damage.
    summon_row, damage_row = (next(i for i, row in enumerate(rules.rows) if row['id'] == spell)
                            for spell in spell_ids)
    state = state.replace(learned_spells=rules.book_bits[summon_row] | rules.book_bits[damage_row],
                          mana=jnp.full_like(state.mana, 5000))
    alive = jnp.zeros_like(state.alive).at[hidden].set(True)
    hidden_only = state.replace(alive=alive)
    hidden_gone = state.replace(alive=jnp.zeros_like(alive))
    action_mask = jax.jit(env.action_mask)
    observe = jax.jit(env.observation)
    step = jax.jit(env.step)
    np.testing.assert_array_equal(observe(hidden_only), observe(hidden_gone))
    np.testing.assert_array_equal(action_mask(hidden_only), action_mask(hidden_gone))
    any_target, field_target, distance = rules.targets(hidden_only)
    assert int(any_target) == int(field_target) == -1
    tile = state.enemy_positions[hidden]
    explored = env._reveal_hero(hidden_only.replace(position=tile)).replace(position=state.position)
    assert bool(jnp.all(explored.revealed | ~state.revealed))
    assert bool(revealed_at(explored, tile))
    assert int(rules.targets(explored)[0]) == hidden
    forced = (jnp.int32(hidden), jnp.int32(hidden), distance)
    for row in (summon_row, damage_row):
        action = jnp.int32(rules.start + row)
        assert action_mask(explored)[action] and not action_mask(hidden_only)[action]
        assert not rules.valid(hidden_only, action, env.max_hp(state), env.movement_cap(state), forced)
        after, _ = step(hidden_only, action)
        assert not after.in_battle
        np.testing.assert_array_equal(after.mana, state.mana)
        np.testing.assert_array_equal(after.spell_casts, state.spell_casts)
        np.testing.assert_array_equal(after.enemy_wounds, state.enemy_wounds)


def test_hidden_owned_city_can_upgrade_and_recruit_without_revealing_it():
    env = NumberGrid(map_config=load_map('city_defence_train'))
    state, _ = env.reset(jax.random.PRNGKey(0))
    position = env.territory.positions[0]
    assert state.city_owned[0] and not revealed_at(state, position)
    state = state.replace(city_levels=state.city_levels.at[0].set(2))
    garrison = env.garrisons
    upgrade, select, hire = map(jnp.int32, (env.territory.start, garrison.select_start, garrison.start))
    mask = jax.jit(env.action_mask)
    step = jax.jit(env.step)
    assert mask(state)[upgrade] and mask(state)[select]
    upgraded, _ = step(state, upgrade)
    selected, _ = step(upgraded, select)
    assert upgraded.city_levels[0] == 3 and selected.selected_garrison == 0 and mask(selected)[hire]
    hired, _ = step(selected, hire)
    assert hired.garrison_ids[0, 0] == garrison.offer_ids[0]
    assert hired.gold == state.gold - env.territory.costs[2] - garrison.offer_costs[0]
    assert not revealed_at(hired, position)
    assert not mask(hired)[garrison.swap_start:garrison.select_start].any()
    np.testing.assert_array_equal(env.territory.observation(hired), env.territory.observation(state))
    np.testing.assert_array_equal(env.garrisons.observation(hired)[:-garrison.count],
                                  env.garrisons.observation(state)[:-garrison.count])


def test_step_keeps_old_sight_and_opens_new_cells():
    env = NumberGrid(map_config=dict(size=24, agent_position=[12, 12], agent_number=1,
                                    opponent_positions=[[22, 22]], opponent_numbers=[1], max_steps=100))
    state, first = env.reset(jax.random.PRNGKey(0))
    assert state.revealed[12, 7] and not state.revealed[12, 18]
    moved, ts = jax.jit(env.step)(state, jnp.int32(2))
    assert moved.position.tolist() == [12, 13]
    assert moved.revealed[12, 7] and moved.revealed[12, 18]
    assert first.observation[-env.fog.observation_size:].reshape(24, 24)[12, 18] == 0
    assert ts.observation[-env.fog.observation_size:].reshape(24, 24)[12, 18] > 0
