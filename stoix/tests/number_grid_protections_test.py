"""Source/ward regressions and named level-one armies; JAX/CUDA only."""
from stoix.tests.number_grid_fixtures import compiled_method
import copy
from functools import lru_cache

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, MAP, SHOOT, CONTINUE, IMMUNE, WARD, MISS
from stoix.envs.number_grid_combat import ATTACK_TYPES, UNITS, MELEE, AREA


def current(**changes):
    return NumberGrid(map_config={**copy.deepcopy(MAP), **changes})


def test_named_armies_exactly_match_python_reference_level_one_profiles(current_game):
    env = current_game[0]
    state, ts = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    chex.assert_trees_all_equal(state.hp[:6], jnp.array([120, 150, 120, 45, 0, 0]))
    assert ts.observation.shape == (1843,) and env.observation_size == 1843
    assert env.action_space().num_values == 244 and env.hero_count == 4
    expected = dict(duke=(150, 50, 80, 0, 50, 'weapon'), possessed=(120, 25, 80, 0, 50, 'weapon'), cultist=(45, 15, 80, 0, 40, 'fire'),
                    squire=(100, 25, 80, 0, 50, 'weapon'), archer=(45, 25, 80, 0, 60, 'weapon'))
    for name, values in expected.items():
        unit = UNITS[name]
        assert tuple(unit[k] for k in ('max_hp', 'damage', 'accuracy', 'armor', 'initiative', 'attack_type')) == values
        assert unit['level'] == 1 and not unit['immunities'] and not unit['protections']
    for index, (row, count) in enumerate(zip(env.combat_info['enemies'], MAP['enemy_units'])):
        assert sum(u is not None for u in row) == count
        assert sum(u['size'] for u in row if u) <= 6
    chex.assert_trees_all_equal(env.combat_traits[0, :6, 0], jnp.array([MELEE]*3+[AREA]+[0]*2, jnp.uint32))


@lru_cache(maxsize=1)
def source_environment():
    # Three immutable target profiles let all nine-source cases reuse one graph.
    hero = [dict(protections=list(ATTACK_TYPES),immunities=list(ATTACK_TYPES)),
            dict(protections=list(ATTACK_TYPES)),
            dict(immunities=['fire'],protections=['water']),{},{}]
    enemies = [[dict(attack_type=ATTACK_TYPES[i % 9], role='ranged')]+[{}]*(n-1)
               for i,n in enumerate(MAP['enemy_units'])]
    env = current(hero_units=5, hero_roster=['possessed']*3+['cultist']*2+[None],
        hero_combat_stats=hero,enemy_combat_stats=enemies,
        enemy_rosters=[['squire']+['archer']*(n-1)+[None]*(6-n) for n in MAP['enemy_units']])
    world,_ = compiled_method(env,'reset')(jax.random.PRNGKey(7))
    return env,world


def source_cases(target=1):
    env,world = source_environment()
    def battle(index):
        state = compiled_method(env,'_begin_battle')(world.replace(enemy=index))
        return state.replace(actor=jnp.int32(6),hp=state.hp.at[:6].set(0).at[0].set(100),
            unit_ids=state.unit_ids.at[0].set(env.progression.initial_ids[target]))
    return env,jax.vmap(battle)(jnp.arange(9,dtype=jnp.int32))


def attack_batch(env, states, values=None, action=CONTINUE):
    if values is None:
        values = jnp.zeros(env.random_size)
    attack = compiled_method(env,'_battle_step')
    results = [attack(jax.tree.map(lambda x,i=i:x[i],states),jnp.int32(action),states.battle_key[i],values)[0]
               for i in range(states.hp.shape[0])]
    return jax.tree.map(lambda *xs:jnp.stack(xs),*results)


def test_all_nine_immunities_block_repeated_hits_before_consuming_wards():
    env, states = source_cases(target=0)
    # A human/agent may deliberately attack immunity. Scripted enemies now
    # correctly defend when every available opponent is immune to their source.
    ids = states.unit_ids.at[:, 0].set(states.unit_ids[:, 6]).at[:, 6].set(states.unit_ids[:, 0])
    states = states.replace(actor=jnp.zeros(9, jnp.int32), unit_ids=ids)
    first = attack_batch(env, states, action=SHOOT)
    second = attack_batch(env, first.replace(actor=states.actor), action=SHOOT)
    chex.assert_trees_all_equal(first.hp, states.hp)
    chex.assert_trees_all_equal(second.hp, states.hp)
    chex.assert_trees_all_equal(second.wards_used, jnp.zeros((9, 12), jnp.uint32))
    chex.assert_trees_all_equal(first.last_immune, jnp.full(9, 1 << 6, jnp.uint32))
    chex.assert_trees_all_equal(first.last_event, jnp.full(9, IMMUNE, jnp.int32))
    assert jnp.all(compiled_method(env,'action_mask',batched=True)(states)[:, SHOOT])


def test_all_nine_wards_absorb_one_hit_misses_do_not_consume_and_rounds_do_not_restore():
    env, states = source_cases()
    missed = attack_batch(env, states, jnp.full(env.random_size, .999))
    chex.assert_trees_all_equal(missed.hp, states.hp)
    chex.assert_trees_all_equal(missed.wards_used, states.wards_used)
    chex.assert_trees_all_equal(missed.last_event, jnp.full(9, MISS, jnp.int32))
    states = states.replace(turn_phase=jnp.full((9, 12), 2, jnp.int32))
    first = attack_batch(env, states)
    assert jnp.all(first.round == 2)
    chex.assert_trees_all_equal(first.hp, states.hp)
    chex.assert_trees_all_equal(first.last_event, jnp.full(9, WARD, jnp.int32))
    chex.assert_trees_all_equal(first.wards_used[:, 0], jnp.left_shift(jnp.uint32(1), jnp.arange(9, dtype=jnp.uint32)))
    second = attack_batch(env, first.replace(actor=states.actor))
    chex.assert_trees_all_equal(second.hp[:, 0], jnp.full(9, 75, jnp.int32))
    chex.assert_trees_all_equal(second.wards_used, first.wards_used)
    restored = compiled_method(env,'_begin_battle',batched=True)(second)
    chex.assert_trees_all_equal(restored.wards_used, jnp.zeros((9, 12), jnp.uint32))
    chex.assert_trees_all_equal(restored.hp[:, :6], second.hp[:, :6])


def test_area_attack_checks_each_target_and_both_cultists_use_fire(elemental_attack_game):
    from stoix.tests.number_grid_fixtures import hero_roster_state
    env, world, _ = elemental_attack_game
    world = hero_roster_state(env,world,['possessed','duke','possessed','cultist','cultist',None])
    state = compiled_method(env,'_begin_battle')(world.replace(enemy=jnp.int32(11)))
    state = state.replace(unit_ids=state.unit_ids.at[6:8].set(env.progression.enemy_ids[2,:2]))
    state = state.replace(hp=state.hp.at[6:].set(jnp.array([45, 100, 0, 45, 45, 45])),
                          escaped=state.escaped.at[9].set(True))
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (2,)+x.shape), state)
    states = states.replace(actor=jnp.array([3, 4], jnp.int32))
    rolls = jnp.zeros(env.random_size).at[16].set(.999).at[22].set(.999)  # target slot 4 misses
    result = attack_batch(env,states,rolls,SHOOT)
    chex.assert_trees_all_equal(result.hp[:, 6:], jnp.tile(jnp.array([45, 100, 0, 45, 45, 30]), (2, 1)))
    chex.assert_trees_all_equal(result.last_damage, jnp.full(2, 15, jnp.int32))
    chex.assert_trees_all_equal(result.last_immune, jnp.full(2, 1 << 6, jnp.uint32))
    chex.assert_trees_all_equal(result.last_ward, jnp.full(2, 1 << 7, jnp.uint32))
    assert jnp.all(result.wards_used[:, 6] == 0) and jnp.all(result.wards_used[:, 7] == 4)
    chex.assert_trees_all_equal(result.last_target, jnp.full(2, -1, jnp.int32))
    traits = compiled_method(env,'unit_traits',batched=True)(result)
    assert jnp.all(traits[:, 7, 3] == 0) and jnp.all(traits[:, 6, 3] == 4)
    # Compact observation includes remaining, not just innate, protection bits.
    obs = compiled_method(env,'observation',batched=True)(result)
    encoded = obs[:, 112+5*env.num_opponents:160+5*env.num_opponents].reshape(2, 12, 4)
    chex.assert_trees_all_close(encoded[:, :, 3], traits[:, :, 3] / 511.)


def test_each_front_melee_unit_uses_its_own_reach_and_empty_slot_never_acts(current_game):
    env = current_game[0]
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    state = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(11)))
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,)+x.shape), state)
    states = states.replace(actor=jnp.arange(4, dtype=jnp.int32))
    masks = compiled_method(env,'action_mask',batched=True)(states)[:, SHOOT:SHOOT+6]
    chex.assert_trees_all_equal(masks, jnp.array([[1,1,0,0,0,0],[1,1,1,0,0,0],[0,1,1,0,0,0],
                                                [1,1,1,1,1,1]], jnp.bool_))
    invalid, _ = compiled_method(env,'step')(jax.tree.map(lambda x: x[0], states), jnp.int32(SHOOT+2))
    chex.assert_trees_all_equal(invalid.hp, state.hp)
    chex.assert_trees_all_equal(invalid.battle_key, state.battle_key)
    rear = states.replace(hp=states.hp.at[:, 6:9].set(0))
    reach = compiled_method(env,'action_mask',batched=True)(rear)[:, SHOOT:SHOOT+6]
    chex.assert_trees_all_equal(reach[:3], jnp.array([[0,0,0,1,1,0],[0,0,0,1,1,1],[0,0,0,0,1,1]], jnp.bool_))
    assert not jnp.any(states.hp[:, 4:6])


@pytest.mark.parametrize('change', [dict(attack_type='unknown'), dict(attack_type=None),
    dict(immunities='fire'), dict(protections=['unknown']), dict(role='unknown')])
def test_invalid_profiles_fail_before_training(change):
    with pytest.raises(ValueError):
        current(hero_combat_stats=[change]+[{}]*3)


def test_rosters_validate_count_slots_and_unknown_units():
    for roster in (['possessed']*6, ['possessed']*4, ['unknown']*5+[None], [True]*5+[None]):
        with pytest.raises(ValueError, match='Roster'):
            current(hero_units=4, hero_roster=roster)
    with pytest.raises(ValueError, match='enemy_rosters'):
        current(enemy_rosters=[])


def test_human_api_reports_named_units_sources_and_current_ward_state(human_service):
    game = human_service
    created = game.create(42)
    assert [u['name'] if u else None for u in created['combat']['heroes']] == ['Одержимый','Герцог','Одержимый']+['Сектант',None,None]
    assert created['combat']['heroes'][3]['attack_type'] == 'fire'
    assert len(created['combat']['attack_types']) == 9
    assert created['snapshot']['state']['wards_used'] == [0]*12
    assert created['snapshot']['state']['hp'][:6] == [120, 150, 120, 45, 0, 0]


def test_unrelated_sources_bypass_protection_and_immunity():
    env,states = source_cases(target=2)
    result = attack_batch(env, states)
    expected = jnp.full(9, 75, jnp.int32).at[2:4].set(100)
    chex.assert_trees_all_equal(result.hp[:, 0], expected)
    assert result.wards_used[2, 0] == 0 and result.wards_used[3, 0] == 8
