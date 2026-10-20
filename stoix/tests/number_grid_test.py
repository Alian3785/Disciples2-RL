from functools import lru_cache
import dataclasses
from stoix.tests.number_grid_fixtures import compiled_method
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import numbergrid_reference as reference
from stoix.envs.number_grid_legacy import DIRECTIONS, MAP, NumberGrid as CurrentNumberGrid
from stoix.utils.make_env import make
from numbergrid_config import make_config
from numbergrid_progression import generate_opponent_numbers, validate_progression

LEGACY_MAP = json.loads((Path(__file__).resolve().parents[2] / 'maps/number_grid-16x16.json').read_text())


class NumberGrid(CurrentNumberGrid):
    """Keep original small-map rule regression cases independent of the current map."""
    def __init__(self, **kwargs):
        super().__init__(map_config=LEGACY_MAP, **kwargs)


@lru_cache(maxsize=1)
def legacy_grid():
    return NumberGrid()


@lru_cache(maxsize=1)
def legacy_current_grid():
    return CurrentNumberGrid()


def initial_state():
    return reference.initial_state(LEGACY_MAP)


def reference_step(state, action):
    return reference.step(state, action, LEGACY_MAP)


def shortest_path():
    return reference.shortest_path(LEGACY_MAP)


def plain(state):
    return {f.name: np.asarray(getattr(state, f.name)).tolist() for f in dataclasses.fields(state)}


def reset(env):
    return env.reset(jax.random.PRNGKey(0))[0]


def test_reset_and_observation():
    env = legacy_grid()
    state, ts = env.reset(jax.random.PRNGKey(1))
    assert plain(state) == initial_state()
    assert ts.first() and ts.discount == 1 and ts.reward == 0
    assert ts.observation.shape == (16,)
    assert ts.observation.dtype == jnp.float32
    assert env.action_space().num_values == 8
    assert np.isfinite(ts.observation).all()


def test_fixed_map_ignores_seed():
    env = legacy_grid()
    a, _ = env.reset(jax.random.PRNGKey(0))
    b, _ = env.reset(jax.random.PRNGKey(999))
    assert plain(a) == plain(b)


@pytest.mark.parametrize('action', range(8))
def test_eight_directions(action):
    env = legacy_grid()
    state = reset(env).replace(position=jnp.asarray([6, 10], jnp.int32))
    nxt, ts = compiled_method(env,'step')(state, jnp.int32(action))
    np.testing.assert_array_equal(nxt.position, np.asarray([6, 10]) + DIRECTIONS[action])
    assert nxt.step_count == 1 and ts.reward == 0


@pytest.mark.parametrize('position,action', [([1, 1], 7), ([1, 14], 1), ([14, 14], 3), ([14, 1], 5)])
def test_perimeter_wall(position, action):
    env = legacy_grid()
    state = reset(env).replace(position=jnp.asarray(position, jnp.int32), alive=jnp.asarray([True, False, False]))
    nxt, _ = compiled_method(env,'step')(state, jnp.int32(action))
    np.testing.assert_array_equal(nxt.position, position)
    assert nxt.step_count == 1


def test_weak_contact_and_no_double_reward():
    env = legacy_grid()
    state = reset(env).replace(position=jnp.asarray([3, 6], jnp.int32))
    nxt, ts = env.step(state, jnp.int32(2))
    assert ts.reward == 1 and nxt.number == 2 and not nxt.alive[0] and not nxt.done
    _, again = env.step(nxt, jnp.int32(2))
    assert again.reward == 0


@pytest.mark.parametrize('number', [1, 0])
def test_equal_or_stronger_loses(number):
    env = legacy_grid()
    state = reset(env).replace(position=jnp.asarray([8, 4], jnp.int32), number=jnp.int32(number))
    nxt, ts = env.step(state, jnp.int32(4))
    assert nxt.lost and nxt.done and not nxt.won
    assert ts.reward == -1 and ts.discount == 0 and ts.terminated()


def test_loss_has_priority_over_simultaneous_capture():
    env = NumberGrid()
    env.opponent_positions = jnp.asarray([[5, 5], [5, 7], [13, 13]], jnp.int32)
    state = reset(env).replace(position=jnp.asarray([3, 6], jnp.int32))
    nxt, ts = compiled_method(env,'step')(state, jnp.int32(4))
    assert nxt.lost and ts.reward == -1 and nxt.number == 1 and np.all(nxt.alive)


def test_multiple_weak_contacts_and_victory_bonus():
    env = NumberGrid()
    env.opponent_positions = jnp.asarray([[5, 5], [5, 7], [13, 13]], jnp.int32)
    state = reset(env).replace(position=jnp.asarray([3, 6], jnp.int32),
                               number=jnp.int32(2), alive=jnp.asarray([True, True, False]))
    nxt, ts = env.step(state, jnp.int32(4))
    assert nxt.won and nxt.number == 4 and ts.reward == 5 and ts.discount == 0


def test_timeout_is_truncation_and_bootstrappable():
    env = NumberGrid(max_steps=1)
    nxt, ts = env.step(reset(env), jnp.int32(0))
    assert nxt.done and not nxt.won and not nxt.lost
    assert ts.truncated() and ts.discount == 1 and ts.reward == 0
    after, again = env.step(nxt, jnp.int32(4))
    assert plain(after) == plain(nxt) and again.reward == 0


def test_win_on_last_step_beats_timeout():
    env = NumberGrid(max_steps=1)
    state = reset(env).replace(position=jnp.asarray([11, 13], jnp.int32),
                               number=jnp.int32(3), alive=jnp.asarray([False, False, True]))
    nxt, ts = env.step(state, jnp.int32(4))
    assert nxt.won and ts.terminated() and not ts.truncated() and ts.reward == 4
    after, again = env.step(nxt, jnp.int32(4))
    assert plain(after) == plain(nxt) and again.reward == 0


def test_opponent_cell_is_occupied():
    env = legacy_grid()
    state = reset(env).replace(position=jnp.asarray([3, 7], jnp.int32))
    nxt, ts = env.step(state, jnp.int32(2))
    np.testing.assert_array_equal(nxt.position, [3, 7])
    assert ts.reward == 1


def test_shortest_path_matches_jax_and_reward_six():
    env = legacy_grid()
    state = reset(env)
    compiled = compiled_method(env,'step')
    reference = initial_state()
    total = 0
    actions = shortest_path()
    for action in actions:
        state, ts = compiled(state, jnp.int32(action))
        reference, reward = reference_step(reference, action)
        assert plain(state) == reference
        assert float(ts.reward) == reward
        total += reward
    assert state.won and state.number == 4 and total == 6
    assert len(actions) == 17


def gpu_reference_sequence(env, actions):
    """Keep every transition, but transfer the GPU trace once rather than per step."""
    initial = reset(env)
    def advance(state, action):
        following, ts = env.step(state, action)
        carry = jax.tree.map(lambda start, value: jnp.where(following.done, start, value),
                             initial, following)
        return carry, (following, ts.reward)
    _, trace = jax.jit(lambda start, moves: jax.lax.scan(advance, start, moves))(
        initial, jnp.asarray(actions, jnp.int32))
    return jax.device_get(trace)


def test_random_transitions_against_independent_reference():
    actions = np.random.default_rng(17).integers(-1, 9, size=2048)
    states, rewards = gpu_reference_sequence(NumberGrid(), actions)
    ref = initial_state()
    for index, action in enumerate(actions):
        ref, reward = reference_step(ref, int(action))
        state = jax.tree.map(lambda value, index=index: value[index], states)
        assert plain(state) == ref
        assert float(rewards[index]) == reward
        if ref['done']:
            ref = initial_state()


def test_vmap_and_jit_run_on_selected_device():
    env = legacy_grid()
    states, _ = compiled_method(env,'reset',batched=True)(jax.random.split(jax.random.PRNGKey(1), 32))
    states, ts = compiled_method(env,'step',batched=True)(states, jnp.zeros(32, jnp.int32))
    assert states.position.shape == (32, 2)
    assert ts.observation.shape == (32, 16)
    assert np.all(np.asarray(states.step_count) == 1)
    assert states.position.devices() == {jax.devices()[0]}


def test_training_autoreset_preserves_terminal_observation():
    config = make_config(map_config=LEGACY_MAP)
    config.env.kwargs.max_steps = 1
    env, _ = make(config)
    states, ts = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    states, ts = compiled_method(env,'step')(states, jnp.zeros(2, jnp.int32))
    assert np.all(ts.truncated()) and np.all(ts.discount == 1)
    assert np.all(np.asarray(ts.observation[:, -1]) == 0)
    assert np.all(np.asarray(ts.extras['next_obs'][:, -1]) == 1)
    assert np.all(np.asarray(ts.extras['episode_metrics']['episode_length']) == 1)


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        make_config(1_000_001)


def test_current_map_counts_and_required_twelve():
    assert MAP['size'] == 24
    assert len(MAP['opponent_positions']) == len(MAP['opponent_numbers']) == 12
    assert all(1 <= n <= 12 for n in MAP['opponent_numbers'])
    assert 12 in MAP['opponent_numbers']
    assert 1 in MAP['opponent_numbers'] and MAP['agent_number'] == 2
    positions = [tuple(p) for p in MAP['opponent_positions']]
    assert len(set(positions)) == 12
    assert all(0 < r < 23 and 0 < c < 23 for r, c in positions)
    assert tuple(MAP['agent_position']) not in positions
    assert all(max(abs(r-MAP['agent_position'][0]), abs(c-MAP['agent_position'][1])) > 1
               for r, c in positions)


def test_current_full_observation_and_reset():
    env = legacy_current_grid()
    state, ts = env.reset(jax.random.PRNGKey(43))
    assert plain(state) == reference.initial_state(MAP)
    assert state.alive.shape == (12,)
    assert ts.observation.shape == (52,)
    assert env.observation_space().shape == (52,)
    assert env.max_return == pytest.approx(15 + 483 * env.exploration_bonus)
    np.testing.assert_allclose(ts.observation[3:51].reshape(12,4)[:,2],
                               np.asarray(MAP['opponent_numbers']) / env.number_scale)
    other, _ = env.reset(jax.random.PRNGKey(999))
    assert plain(other) == plain(state)


def test_current_grid_vmap_scan_on_gpu():
    env = legacy_current_grid()
    @jax.jit
    def rollout(keys):
        states, _ = jax.vmap(env.reset)(keys)
        def advance(states, actions):
            states, ts = jax.vmap(env.step)(states, actions)
            return states, ts
        return jax.lax.scan(advance, states, jnp.zeros((32, 64), jnp.int32))
    states, ts = rollout(jax.random.split(jax.random.PRNGKey(55),64))
    assert states.alive.shape == (64,12)
    assert ts.observation.shape == (32,64,52)
    assert all(x.devices() == {jax.devices()[0]} for x in jax.tree.leaves((states,ts)))


def test_current_grid_against_reference():
    actions = np.random.default_rng(23).integers(-1,9,size=1024)
    states, rewards = gpu_reference_sequence(CurrentNumberGrid(), actions)
    ref = reference.initial_state(MAP)
    for index, action in enumerate(actions):
        ref, reward = reference.step(ref, int(action), MAP)
        state = jax.tree.map(lambda value, index=index: value[index], states)
        assert plain(state) == ref
        assert float(rewards[index]) == pytest.approx(reward, abs=1e-6)
        if ref['done']:
            ref = reference.initial_state(MAP)


@pytest.mark.parametrize('enemy', range(12))
def test_current_grid_each_enemy_can_be_captured_when_weaker(enemy):
    env = legacy_current_grid()
    r,c = MAP['opponent_positions'][enemy]
    state = reset(env).replace(position=jnp.asarray([r-2,c], jnp.int32),
        number=jnp.int32(MAP['opponent_numbers'][enemy]+1), alive=jnp.arange(12)==enemy)
    nxt, ts = compiled_method(env,'step')(state,jnp.int32(4))
    assert nxt.won and float(ts.reward) == pytest.approx(4 + env.exploration_bonus - env.step_cost) and not np.any(nxt.alive)


def test_current_grid_bottom_right_boundary():
    env = legacy_current_grid()
    state = reset(env).replace(position=jnp.asarray([22,22],jnp.int32))
    nxt, ts = env.step(state,jnp.int32(3))
    np.testing.assert_array_equal(nxt.position,[22,22])
    assert nxt.step_count == 1 and float(ts.reward) == pytest.approx(-env.step_cost)


def test_current_grid_has_a_legal_winning_route():
    env = legacy_current_grid()
    state = reset(env)
    advance = compiled_method(env,'step')
    reward = 0.0
    for action in reference.winning_path(MAP):
        state, ts = advance(state, jnp.int32(action))
        reward += float(ts.reward)
    assert state.won and not state.lost and not np.any(state.alive)
    new_cells = sum(int(word).bit_count() for word in np.asarray(state.visited)) - 1
    assert state.number == 14
    assert reward == pytest.approx(15 + new_cells*env.exploration_bonus - int(state.step_count)*env.step_cost)


@pytest.mark.parametrize('action', [0, -1, 8])
def test_step_cost_applies_to_movement_and_invalid_actions(action):
    env = CurrentNumberGrid(map_config={**MAP, 'exploration_bonus': 0})
    state, initial = env.reset(jax.random.PRNGKey(0))
    assert initial.reward == 0
    nxt, ts = compiled_method(env,'step')(state, jnp.int32(action))
    assert nxt.step_count == 1 and not nxt.done
    assert float(ts.reward) == pytest.approx(-0.001)


@pytest.mark.parametrize('event', ['occupied', 'capture', 'loss', 'timeout'])
def test_step_cost_on_contacts_and_terminal_steps(event):
    env = CurrentNumberGrid(max_steps=1 if event == 'timeout' else None,
                           map_config={**MAP, 'exploration_bonus': 0})
    state = reset(env)
    action, base_reward = 0, 0
    if event in ('occupied', 'capture'):
        # Enemy 0 has value 1 at [3,8]. Moving onto it is blocked, but contact still captures it.
        state = state.replace(position=jnp.asarray([3,7] if event == 'occupied' else [3,6]))
        action, base_reward = 2, 1
    elif event == 'loss':
        state = state.replace(position=jnp.asarray([18,20]))
        action, base_reward = 4, -1
    nxt, ts = compiled_method(env,'step')(state, jnp.int32(action))
    assert float(ts.reward) == pytest.approx(base_reward - 0.001)
    if event in ('loss', 'timeout'):
        assert nxt.done
        after, again = env.step(nxt, jnp.int32(action))
        assert plain(after) == plain(nxt) and again.reward == 0
        assert float(ts.discount) == (0 if event == 'loss' else 1)


@pytest.mark.parametrize('numbers', [[1, 3] + [12]*10, [1]*10 + [12, 12]])
def test_progression_rejects_a_gap_even_with_one_and_twelve(numbers):
    with pytest.raises(ValueError, match='block progression'):
        CurrentNumberGrid(map_config={**MAP, 'opponent_numbers': numbers})


def test_current_map_guarantees_reaching_thirteen():
    assert MAP['minimum_reachable_number'] == 13
    assert validate_progression(MAP['opponent_numbers'], 2, 13) == 14
    assert generate_opponent_numbers(MAP['map_seed']) == MAP['opponent_numbers']


@pytest.mark.parametrize('seed', [0, 1, 42, 43, 44, 20261004])
def test_generated_numbers_have_a_legal_spatial_winning_route(seed):
    numbers = generate_opponent_numbers(seed)
    assert len(numbers) == 12 and 1 in numbers and 12 in numbers
    assert all(1 <= n <= 12 for n in numbers)
    assert generate_opponent_numbers(seed) == numbers
    config = {**MAP, 'opponent_numbers': numbers}
    state = reference.initial_state(config)
    for action in reference.winning_path(config):
        state, _ = reference.step(state, action, config)
    assert state['won'] and state['number'] == 14


def test_exploration_once_per_cell_and_reset():
    env = legacy_current_grid()
    state = reset(env)
    advance = compiled_method(env,'step')
    rewards = []
    # New [2,3], back to spawn, revisit [2,3], new [2,4], invalid action.
    for action in [2, 6, 2, 2, -1]:
        state, ts = advance(state, jnp.int32(action))
        rewards.append(float(ts.reward))
    np.testing.assert_allclose(rewards, [.009, -.001, -.001, .009, -.001], atol=1e-7)
    assert sum(int(w).bit_count() for w in np.asarray(state.visited)) == 3
    state, ts = advance(reset(env), jnp.int32(2))
    assert float(ts.reward) == pytest.approx(.009)


def test_exploration_blocked_loss_timeout_and_absorbing():
    env = legacy_current_grid()
    advance = compiled_method(env,'step')
    state, _ = advance(reset(env), jnp.int32(7))  # [1,1]
    state, ts = advance(state, jnp.int32(7))  # wall
    assert float(ts.reward) == pytest.approx(-.001)
    state = reset(env).replace(position=jnp.asarray([3,7]))
    _, ts = advance(state, jnp.int32(2))  # occupied, captures but doesn't move
    assert float(ts.reward) == pytest.approx(.999)
    state = reset(env).replace(position=jnp.asarray([18,20]))
    after, ts = advance(state, jnp.int32(4))
    assert after.lost and float(ts.reward) == pytest.approx(-1.001)
    short = CurrentNumberGrid(max_steps=1)
    after, ts = short.step(reset(short), jnp.int32(2))
    assert ts.truncated() and float(ts.reward) == pytest.approx(.009)
    again, ts = short.step(after, jnp.int32(4))
    assert plain(again) == plain(after) and ts.reward == 0


def test_exploration_bit31_and_independent_batched_memory():
    env = legacy_current_grid()
    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    states, _ = jax.vmap(env.reset)(keys)
    advance = compiled_method(env,'step',batched=True)
    states, _ = advance(states, jnp.asarray([2, 4]))
    states, ts = advance(states, jnp.asarray([6, 1]))  # spawn vs [2,3]
    np.testing.assert_allclose(ts.reward, [-.001, .009], atol=1e-7)
    state = reset(env).replace(position=jnp.asarray([6,14]))
    state, ts = env.step(state, jnp.int32(2))  # flat cell 159 -> bit 31
    assert int(state.visited[4]) == 2**31
    assert float(ts.reward) == pytest.approx(.009)


def test_exploration_autoreset_and_success_is_not_inferred_from_return():
    config = make_config(map_config={**MAP, 'exploration_bonus': 100})
    config.env.kwargs.max_steps = 1
    env, _ = make(config)
    state, _ = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    advance = compiled_method(env,'step')
    for _ in range(2):
        state, ts = advance(state, jnp.asarray([2, 4]))
        np.testing.assert_allclose(ts.reward, [99.999, 99.999], atol=1e-5)
        assert np.all(ts.extras['episode_metrics']['is_terminal_step'])
        assert not np.any(ts.extras['episode_metrics']['episode_success'])


@pytest.mark.parametrize('bonus', [-.01, float('nan'), float('inf')])
def test_invalid_exploration_bonus(bonus):
    with pytest.raises(ValueError, match='exploration_bonus'):
        CurrentNumberGrid(map_config={**MAP, 'exploration_bonus': bonus})
