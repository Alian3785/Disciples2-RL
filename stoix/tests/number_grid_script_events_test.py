"""S101 event source/regression checks; run only on explicit request, on CUDA.

Source: docs/BETHREZENS_RETREAT.md. Not in the core budget.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import REST, NumberGrid, load_map
from stoix.tests.number_grid_fixtures import compiled_method


@pytest.fixture(scope='module')
def campaign():
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh with CUDA'
    env = NumberGrid(map_config=load_map('bethrezens_retreat'))
    initial, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    process = jax.jit(lambda s, new_day: env.script_events.process(env, s, new_day))
    return env, initial, process


def test_template_repeated_small_types_and_large_pairs():
    from scripts.build_bethrezens_retreat import UNIT_KEYS, UNITS
    from scripts.build_bethrezens_retreat_events import template_formation
    root = Path(__file__).resolve().parents[2]
    raw = json.loads((root/'docs/maps/bethrezens_retreat/layout.json').read_text(encoding='utf-8'))['raw_tables']
    rows = {r['ID']: r for r in raw['SStkTemp.dbf']}
    roster, _ = template_formation(rows['S101TM0000'], UNIT_KEYS, UNITS)
    assert roster.count(UNIT_KEYS['G000UU0092']) == 2
    roster, _ = template_formation(rows['S101TM0006'], UNIT_KEYS, UNITS)
    assert roster.count(UNIT_KEYS['G000UU0090']) == 2
    roster, _ = template_formation(rows['S101TM0001'], UNIT_KEYS, UNITS)
    assert roster.count(UNIT_KEYS['G000UU0094']) == 1 and roster[4] is None


def test_intro_runs_once_on_day_one_without_spending_movement(campaign):
    env, state, process = campaign
    assert state.position.tolist() == [23, 37]
    assert int(state.day) == 1 and int(state.map_steps) == 0
    assert int(state.movement_points) == int(env.movement_cap(state))
    fired = {e['id'] for e, flag in zip(env.script_events.events, state.script_events_fired) if bool(flag)}
    assert fired == {'S101EV0009', 'S101EV0013', 'S101EV0014'}
    again = process(state, jnp.bool_(True))
    assert jnp.array_equal(again.enemy_wounds, state.enemy_wounds)
    assert jnp.array_equal(again.enemy_positions, state.enemy_positions)
    assert int(jnp.sum(state.alive)) == 105


@pytest.mark.parametrize('day,number', [(10, 0), (20, 2), (30, 0), (50, 3)])
def test_calendar_day_and_once_only(campaign, day, number):
    env, initial, process = campaign
    before = initial.replace(day=jnp.int32(day))
    after = process(before, jnp.bool_(True))
    assert int(jnp.sum(after.alive)-jnp.sum(initial.alive)) == number
    repeat = process(after.replace(day=jnp.int32(day*2)), jnp.bool_(True))
    # An event already consumed must not spawn another copy (other scheduled
    # events at day*2 can still fire independently).
    assert jnp.all(repeat.script_events_fired | ~after.script_events_fired)
    if day == 10:
        enemy = env.map_config['reference_enemy_ids'].index('S101KC000c')
        assert int(after.script_orders[env.script_events.order_slot[enemy]]) == 1
    if day == 30:
        enemy = env.map_config['reference_enemy_ids'].index('S101KC0012')
        assert int(after.script_orders[env.script_events.order_slot[enemy]]) == 1
    no_day = process(before, jnp.bool_(False))
    assert jnp.array_equal(no_day.alive, initial.alive)


def test_ambush_uses_zone_and_spawns_three_distinct_parties(campaign):
    env, initial, process = campaign
    outside = initial.replace(position=jnp.array([3, 23], jnp.int32))
    assert not bool(process(outside, jnp.bool_(False)).script_events_fired[0])
    trigger = initial.replace(position=jnp.array([3, 22], jnp.int32))
    after = process(trigger, jnp.bool_(False))
    assert bool(after.script_events_fired[0])
    assert int(jnp.sum(after.alive)-jnp.sum(initial.alive)) == 3
    assert len({tuple(p) for p in after.enemy_positions[105:108].tolist()}) == 3
    assert jnp.all(after.hp[:6] <= initial.hp[:6])
    repeat = process(after, jnp.bool_(False))
    assert jnp.array_equal(repeat.hp, after.hp)
    assert jnp.array_equal(repeat.enemy_positions, after.enemy_positions)


def test_totem_exchange_requires_item_and_living_stack_and_is_once(campaign):
    env, initial, process = campaign
    totem, rune = [env.item_rules.keys.index(k) for k in ('agshlisga_totem', 'runestone')]
    before = initial.replace(position=jnp.array([66, 17], jnp.int32))
    no_item = process(before, jnp.bool_(False))
    assert not bool(jnp.any(no_item.item_inventory == rune))
    before = before.replace(item_inventory=before.item_inventory.at[0].set(totem))
    gone = process(before.replace(alive=before.alive.at[26].set(False)), jnp.bool_(False))
    assert bool(jnp.any(gone.item_inventory == totem))
    after = process(before, jnp.bool_(False))
    assert not bool(jnp.any(after.item_inventory == totem))
    assert int(jnp.sum(after.item_inventory == rune)) == 1
    repeat = process(after, jnp.bool_(False))
    assert jnp.array_equal(repeat.item_inventory, after.item_inventory)


def test_disabled_events_keep_shapes_and_static_start(campaign):
    env, _, _ = campaign
    baseline = NumberGrid(map_config={**load_map('bethrezens_retreat'), 'scenario_events_enabled': False})
    state, _ = compiled_method(baseline, 'reset')(jax.random.PRNGKey(42))
    assert baseline.observation_size == env.observation_size
    assert baseline.num_actions == env.num_actions
    assert state.position.tolist() == [22, 36]
    assert not bool(jnp.any(state.script_events_fired))
    assert int(jnp.sum(state.alive)) == 105
    assert baseline.reference_enemy_count == 105 and env.reference_enemy_count == 113


def test_initial_native_attack_orders_include_intro_and_seventh_party(campaign):
    """Installed Sstack ORDER=4/ORDER_TARG=KC0013; LOrder and TAppEdit X009TA2007."""
    env, state, _ = campaign
    root = Path(__file__).resolve().parents[2]
    raw = json.loads((root/'docs/maps/bethrezens_retreat/layout.json').read_text(encoding='utf-8'))['raw_tables']
    native = [r for r in raw['Sstack.dbf'] if r['ORDER'] == '4']
    assert {r['STACK_ID'] for r in native} == {
        'S101KC000f', 'S101KC0011', 'S101KC0072', 'S101KC0078',
        'S101KC0079', 'S101KC007b', 'S101KC007c'}
    imported = {r['enemy']: r for r in env.map_config['scenario_order_armies']}
    for row in native:
        enemy = env.map_config['reference_enemy_ids'].index(row['STACK_ID'])
        assert row['ORDER_TARG'] == imported[enemy]['target'] == 'S101KC0013'
        assert imported[enemy]['moves'] == int(row['MOVE']) == 20
        assert int(state.script_orders[env.script_events.order_slot[enemy]]) == 4


def test_intro_attackers_pursue_and_queue_battles_on_enemy_turn(campaign):
    env, initial, _ = campaign
    ids = ('S101KC000f', 'S101KC0011', 'S101KC0078', 'S101KC0079', 'S101KC007b', 'S101KC007c')
    enemies = jnp.array([env.map_config['reference_enemy_ids'].index(gid) for gid in ids])
    after = jax.jit(env.script_events.advance)(initial)
    assert jnp.all(after.enemy_positions[enemies] == initial.position)
    assert jnp.all(after.pending_encounters[enemies])
    assert not jnp.any(initial.pending_encounters)
    # Normal/stand orders remain stationary even with the added native attackers.
    fixed = env.map_config['reference_enemy_ids'].index('S101KC000c')
    assert jnp.array_equal(after.enemy_positions[fixed], initial.enemy_positions[fixed])
    assert bool(compiled_method(env, 'action_mask')(initial)[REST])
    attacked, _ = compiled_method(env, 'step')(initial, jnp.int32(REST))
    assert int(attacked.day) == 2
    assert int(attacked.enemy) in enemies.tolist() and bool(attacked.in_battle)


def test_exit_requires_ambush_response_and_drained_battle_queue(campaign):
    """User-approved adaptation, 2026-10-16; not a claimed native S101 condition."""
    env, initial, process = campaign
    won = compiled_method(env, '_campaign_won')
    at_goal = initial.replace(position=env.objective_position)
    assert not bool(won(at_goal))
    assert not bool(won(at_goal.replace(alive=jnp.zeros_like(initial.alive))))
    # Earlier enemy turns cannot unlock a future ambush.
    advanced = jax.jit(env.script_events.advance)(initial)
    assert not bool(advanced.script_objective_ready)
    trigger = initial.replace(position=jnp.array([5, 20], jnp.int32),
                              alive=jnp.zeros_like(initial.alive))
    spawned = process(trigger, jnp.bool_(False))
    at_goal = spawned.replace(position=env.objective_position)
    assert not bool(won(at_goal))  # movement/magic alone cannot open the gate
    responded = jax.jit(env.script_events.advance)(at_goal)
    assert bool(responded.script_objective_ready)
    assert int(jnp.sum(responded.pending_encounters & responded.alive)) == 3
    assert not bool(won(responded))
    remaining = responded.pending_encounters.at[105].set(False)
    assert not bool(won(responded.replace(pending_encounters=remaining)))
    cleared = responded.replace(pending_encounters=jnp.zeros_like(remaining))
    assert not bool(won(cleared.replace(in_battle=jnp.bool_(True))))
    assert bool(won(cleared))
    assert not bool(won(cleared.replace(position=jnp.array([3, 27], jnp.int32))))
    assert float(env.script_events.observation(at_goal)[-1]) == 0.
    assert float(env.script_events.observation(cleared)[-1]) == 1.
    fresh, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(43))
    assert not bool(fresh.script_objective_ready)


def test_one_day_walk_past_ambush_cannot_finish_even_at_centre(campaign):
    env, initial, _ = campaign
    step = compiled_method(env, 'step')
    mask = compiled_method(env, 'action_mask')
    # Isolate the late ambush; use the replay's bypass followed by the extra
    # three cells to the centre. No altered movement budget or terrain costs.
    state = initial.replace(position=jnp.array([6, 20], jnp.int32),
                            alive=jnp.zeros_like(initial.alive))
    path = ([5, 20], [4, 21], [3, 22], [2, 23], [3, 24], [2, 25],
            [3, 26], [3, 27], [3, 28], [3, 29], [2, 30])
    directions = env.directions.tolist()
    for point in path:
        delta = [p-int(q) for p, q in zip(point, state.position)]
        action = directions.index(delta)
        assert bool(mask(state)[action])
        state, _ = step(state, jnp.int32(action))
        assert state.position.tolist() == point and not bool(state.won)
        assert not bool(state.script_objective_ready)
    assert int(state.day) == int(initial.day)
    assert bool(state.script_events_fired[env.script_events.objective_event])
    attacked, _ = step(state, jnp.int32(REST))
    assert bool(attacked.script_objective_ready) and not bool(attacked.won)
    assert bool(attacked.in_battle | attacked.lost)
