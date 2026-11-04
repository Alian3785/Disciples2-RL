"""Original zone of control on the strategic map (docs/COMBAT.md#zones); CUDA only."""
import chex
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import CONTINUE, ENGAGE, WITHDRAW
from stoix.envs.number_grid_chests import CHEST_LOOT
from stoix.envs.number_grid_zones import NEIGHBOURS
from stoix.tests.number_grid_fixtures import compiled_method


def placed(state, armies):
    """Only the given armies live, on the given tiles; zones read live positions."""
    alive, positions = jnp.zeros_like(state.alive), state.enemy_positions
    for index, tile in armies.items():
        alive = alive.at[index].set(True)
        positions = positions.at[index].set(jnp.array(tile, jnp.int32))
    return state.replace(alive=alive, enemy_positions=positions)


def test_first_holder_follows_the_original_neighbour_order(current_game):
    env, initial, _, _ = current_game
    resolve = compiled_method(env.zones, 'resolve')
    assert NEIGHBOURS == ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    hero = initial.replace(position=jnp.array([20, 20]))
    destination = jnp.array([20, 21])
    # Discipl2.exe 0x7b2990 lists (-1,1) before (1,1), whatever the army indices.
    state = placed(hero, {3: (21, 22), 9: (19, 22)})
    entered, enemy = resolve(state, destination, jnp.bool_(True))
    assert entered and enemy == 9
    entered, enemy = resolve(placed(hero, {3: (19, 22), 9: (19, 22)}), destination, jnp.bool_(True))
    assert entered and enemy == 3  # one tile: the lower index, as for a direct attack
    assert resolve(state, destination, jnp.bool_(False))[1] == -1
    assert resolve(state, jnp.array([20, 19]), jnp.bool_(True))[1] == -1


def test_hero_beside_holders_fights_unless_it_steps_out_of_their_zones(current_game):
    env, initial, _, _ = current_game
    resolve = compiled_method(env.zones, 'resolve')
    hero = initial.replace(position=jnp.array([20, 20]))
    # 4 and 6 stand beside the hero; [20,21] is beside 4 only, yet 6 comes first around the hero.
    state = placed(hero, {4: (21, 21), 6: (19, 19), 9: (18, 22)})
    entered, enemy = resolve(state, jnp.array([20, 21]), jnp.bool_(True))
    assert not entered and enemy == 6
    entered, enemy = resolve(state, jnp.array([20, 19]), jnp.bool_(True))
    assert not entered and enemy == 6
    # [19,21] leaves both zones but enters 9's: the hero moves and fights 9.
    entered, enemy = resolve(state, jnp.array([19, 21]), jnp.bool_(True))
    assert entered and enemy == 9
    entered, enemy = resolve(placed(hero, {4: (21, 21), 6: (19, 19)}), jnp.array([19, 21]), jnp.bool_(True))
    assert entered and enemy == -1


@pytest.mark.core
def test_zone_step_pays_step_and_entry_and_a_retreat_keeps_the_new_tile(current_game):
    env, initial, advance, battle_step = current_game
    assert compiled_method(env, 'map_commands')(initial.replace(position=jnp.array([2, 6])))[2].tolist() == [0, 12]
    entered, _ = advance(initial.replace(position=jnp.array([2, 6])), jnp.int32(2))
    assert entered.in_battle and entered.enemy == 0 and entered.last_event == ENGAGE
    assert entered.movement_points == 8 and entered.map_steps == 1
    escaping = entered.replace(actor=jnp.int32(0), hp=entered.hp.at[1:6].set(0),
                               retreating=entered.retreating.at[0].set(True))
    escaped, _ = battle_step(escaping, jnp.int32(CONTINUE), escaping.battle_key, jnp.zeros(env.random_size))
    assert escaped.last_event == WITHDRAW and not escaped.in_battle and escaped.alive[0]
    chex.assert_trees_all_equal(escaped.position, jnp.array([2, 7]))


@pytest.mark.core
def test_settlements_garrisons_ruin_keepers_and_kin_hold_no_zone(current_game):
    env, initial, advance, _ = current_game
    holders = compiled_method(env.zones, 'holders')(initial)
    inside = jnp.any(jnp.all(initial.enemy_positions[:, None, :] == env.zones.inside[None], axis=-1), axis=-1)
    assert jnp.flatnonzero(inside).tolist() == [41, 42, 43, 44, 45]  # two garrisons per city, ruin keeper
    assert jnp.flatnonzero(env.zones.kin).tolist() == [14, 47]  # Legions-only stacks, Legions hero
    chex.assert_trees_all_equal(holders, initial.alive & ~inside & ~env.zones.kin)
    # Kin: stepping beside army 14 is free, attacking its tile is not forbidden.
    beside, _ = advance(initial.replace(position=jnp.array([12, 40])), jnp.int32(2))
    assert not beside.in_battle and beside.position.tolist() == [12, 41] and beside.movement_points == 18
    assert compiled_method(env, 'map_commands')(beside)[2].tolist() == [14, 10]
    # Entering the capital resolves the fort, not a stack beside it.
    guarded = initial.replace(position=jnp.array([2, 3]),
                              enemy_positions=initial.enemy_positions.at[1].set(jnp.array([1, 1])))
    home, _ = advance(guarded, jnp.int32(6))
    assert not home.in_battle and home.position.tolist() == [2, 2]
    resolve = compiled_method(env.zones, 'resolve')
    assert resolve(guarded, jnp.array([1, 2]), jnp.bool_(True))[1] == 1


@pytest.mark.core
def test_chest_beside_a_holder_opens_only_from_outside_its_zone(current_game):
    env, initial, advance, _ = current_game
    chest = 2
    assert env.chest_rules.positions[chest].tolist() == [25, 12]
    assert initial.enemy_positions[25].tolist() == [24, 12]
    fought, _ = advance(initial.replace(position=jnp.array([25, 10])), jnp.int32(2))
    assert fought.in_battle and fought.enemy == 25 and fought.position.tolist() == [25, 11]
    assert fought.chest_alive[chest] and not jnp.any(fought.last_loot)
    opened, _ = advance(initial.replace(position=jnp.array([27, 12])), jnp.int32(0))
    assert not opened.in_battle and opened.position.tolist() == [26, 12]
    assert not opened.chest_alive[chest] and opened.last_event == CHEST_LOOT
    chex.assert_trees_all_equal(opened.last_loot, env.chest_rules.loot[chest])
