"""Persistent wounds, persistent casualties and strategic rest on JAX/CUDA."""
from stoix.tests.number_grid_fixtures import compiled_method, basic_environment
import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.tests.number_grid_fixtures import MAP
from stoix.envs.number_grid import (
    NumberGrid, SHOOT, DEFEND, CONTINUE, REST, VICTORY, WITHDRAW, DEFEAT, ROUND_LIMIT_RETREAT,
)


@pytest.fixture(scope='module')
def env():
    return basic_environment()


def test_victory_and_withdrawal_preserve_dead_and_wounded_units(env):
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    state = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(0))
    wounded = jnp.array([12, 0, 44, 0, 1, 20], jnp.int32)
    victory = state.replace(hp=state.hp.at[:6].set(wounded).at[6:].set(0).at[6].set(1))
    fleeing = state.replace(hp=state.hp.at[:6].set(wounded),
                            escaped=state.escaped.at[2:6].set(True),
                            retreating=state.retreating.at[0].set(True))
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), victory, fleeing)
    result, _ = compiled_method(env, '_battle_step', batched=True)(
        batch, jnp.array([SHOOT, CONTINUE], jnp.int32), batch.battle_key, jnp.zeros((2, 36)))
    expected = jnp.array([12, 0, 44, 0, 1, 20, 0, 0, 0, 0, 0, 0], jnp.int32)
    chex.assert_trees_all_equal(result.hp, jnp.stack((expected, expected)))
    chex.assert_trees_all_equal(result.last_event, jnp.array([VICTORY, WITHDRAW], jnp.int32))
    assert not jnp.any(result.in_battle | result.done | result.lost)
    assert not jnp.any(result.escaped | result.retreating | result.defended)
    begun = compiled_method(env,'_begin_battle',batched=True)(result)
    chex.assert_trees_all_equal(begun.hp[:, :6], result.hp[:, :6])
    chex.assert_trees_all_equal(begun.hp[:, 6:], jnp.tile(jnp.array([20, 0, 0, 0, 0, 0]), (2, 1)))
    # The dead warrior remains at 0 HP when the next battle starts.
    units = compiled_method(env,'observation')(jax.tree.map(lambda x: x[0], begun))[124:232].reshape(12, 9)
    assert float(units[1, 0]) == pytest.approx(0.)


def test_rest_heals_living_units_once_caps_hp_and_never_revives(env):
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    state = state.replace(hp=state.hp.at[:6].set(jnp.array([1, 91, 44, 0, 20, 45])))
    points = jnp.array([0, 8, 20], jnp.int32)
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (3,)+x.shape), state).replace(movement_points=points)
    following, ts = compiled_method(env,'step',batched=True)(states, jnp.full(3, REST, jnp.int32))
    expected = jnp.array([6, 100, 45, 0, 25, 45, 0, 0, 0, 0, 0, 0], jnp.int32)
    chex.assert_trees_all_equal(following.hp, jnp.tile(expected, (3, 1)))
    chex.assert_trees_all_close(ts.reward, -.001*points)  # no separate healing reward
    chex.assert_trees_all_equal(following.day, jnp.full(3, 2, jnp.int32))
    chex.assert_trees_all_equal(following.gold, jnp.full(3, 100, jnp.int32))
    chex.assert_trees_all_equal(following.battle_key, states.battle_key)
    # Re-reading an observation cannot apply regeneration a second time.
    compiled_method(env,'observation',batched=True)(following).block_until_ready()
    again, _ = compiled_method(env,'step',batched=True)(following, jnp.full(3, REST, jnp.int32))
    chex.assert_trees_all_equal(again.hp[:, :6], jnp.tile(jnp.array([11, 100, 45, 0, 30, 45]), (3, 1)))
    assert jnp.all(again.day == 3)


def test_regeneration_uses_individual_max_hp_and_leaves_empty_slots_empty():
    env = NumberGrid(map_config={**MAP, 'hero_units': 5, 'hero_mage_slot': -1,
        'hero_combat_stats': [dict(max_hp=1), dict(max_hp=9), dict(max_hp=11),
                              dict(max_hp=101), dict(max_hp=120)],
        'enemy_units': [1]*24, 'enemy_warrior_slots': [0]*24})
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    state = state.replace(hp=state.hp.at[:5].set(1))
    rested, _ = compiled_method(env,'step')(state, jnp.int32(REST))
    chex.assert_trees_all_equal(rested.hp[:6], jnp.array([1, 2, 3, 12, 13, 0]))
    battle = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(4))
    battle = battle.replace(hp=battle.hp.at[:4].set(0).at[6:].set(0).at[6].set(1))
    won, _ = compiled_method(env,'_battle_step')(battle, jnp.int32(SHOOT), battle.battle_key, jnp.zeros(36))
    chex.assert_trees_all_equal(won.hp[:6], jnp.array([0, 0, 0, 0, 1, 0]))


def test_movement_building_invalid_and_battle_rest_do_not_heal(env):
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    state = state.replace(hp=state.hp.at[:6].set(1), gold=jnp.int32(1000))
    battle = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(11)))
    states = jax.tree.map(lambda *xs: jnp.stack(xs), state, state, state, battle, state.replace(done=jnp.bool_(True)))
    following, _ = compiled_method(env,'step',batched=True)(states, jnp.array([2, 18, -1, REST, REST], jnp.int32))
    chex.assert_trees_all_equal(following.hp, states.hp)
    assert following.buildings[1] != 0  # a real successful construction was tested
    chex.assert_trees_all_equal(following.day, states.day)


def test_total_defeat_and_forced_retreat_do_not_resurrect(env):
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    state = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(0)))
    lost = state.replace(hp=state.hp.at[:6].set(0).at[0].set(1), actor=jnp.int32(6))
    limited = state.replace(hp=state.hp.at[:6].set(0).at[0].set(12), actor=jnp.int32(0),
                            turn_phase=jnp.full(12, 2).at[0].set(0), round=jnp.int32(env.max_rounds))
    states = jax.tree.map(lambda *xs: jnp.stack(xs), lost, limited)
    following, _ = compiled_method(env, '_battle_step', batched=True)(
        states, jnp.array([CONTINUE, DEFEND], jnp.int32), states.battle_key, jnp.zeros((2, 36)))
    chex.assert_trees_all_equal(following.last_event, jnp.array([DEFEAT, ROUND_LIMIT_RETREAT], jnp.int32))
    assert following.done[0] and following.lost[0]
    assert not following.done[1] and not following.lost[1] and following.in_battle[1]
    chex.assert_trees_all_equal(following.hp[0, :6], jnp.zeros(6, jnp.int32))
    chex.assert_trees_all_equal(following.hp[1, :6], limited.hp[:6])
    final, _ = compiled_method(env,'step',batched=True)(following, jnp.full(2, REST, jnp.int32))
    chex.assert_trees_all_equal(final.hp, following.hp)


def test_human_service_uses_the_same_regeneration_and_reports_current_health(human_service):
    service = human_service
    created = service.create(42)
    session, env = created['session'], service.env
    assert created['turn_rules']['regeneration_percent'] is None
    assert created['turn_rules']['regeneration_rounding'] == 'ceil'
    assert created['turn_rules']['automatic_revive'] is False
    faction, state, reward = service.sessions[session]
    state = state.replace(hp=state.hp.at[:6].set(jnp.array([1, 50, 44, 0, 0, 0])))
    service.sessions[session] = (faction, state, reward)
    expected, ts = compiled_method(env,'step')(state, jnp.int32(REST))
    actual = service.act(session, REST)
    assert actual['snapshot'] == service.snapshot(env, expected, float(ts.reward))
    assert actual['snapshot']['state']['hp'][:6] == [67, 133, 110, 0, 0, 0]
