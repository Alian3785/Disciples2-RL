"""Run the actual batched evaluator over deterministic objective life cycles."""
from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import pytest

import benchmark_number_grid as benchmark

pytestmark = pytest.mark.core


class EvaluationState(NamedTuple):
    step_count: object
    done: object
    won: object
    lost: object
    enemy: object
    in_battle: object
    hp: object
    enemy_wounds: object
    alive: object
    scenario_spawned: object


class TimeStep(NamedTuple):
    observation: object
    reward: object


class ObjectiveEpisode:
    """An absorbing scripted episode; the evaluator still runs JIT/vmap/while_loop."""
    objective_enemy = 0
    spell_casting_enabled = True

    def __init__(self, frames):
        self.frames = jnp.asarray(frames)
        self.casting = SimpleNamespace(enemy_max_hp=lambda state: jnp.full((1, 6), 100/6))

    def state(self, step):
        spawned, alive, wounds, battle_hp = self.frames[step]
        return EvaluationState(step, step == len(self.frames)-1, (spawned > 0) & (alive == 0),
            jnp.bool_(False), jnp.int32(0), battle_hp >= 0,
            jnp.concatenate((jnp.zeros(6), jnp.full(6, jnp.maximum(battle_hp, 0)/6))),
            jnp.full((1, 6), wounds/6), jnp.array([alive > 0]), jnp.array([spawned > 0]))

    def reset(self, key):
        return self.state(jnp.int32(0)), TimeStep(jnp.zeros(1), jnp.float32(0))

    def step(self, state, action):
        step = jnp.minimum(state.step_count+1, len(self.frames)-1)
        return self.state(step), TimeStep(jnp.zeros(1), jnp.float32(0))


@pytest.mark.parametrize('greedy', [True, False])
@pytest.mark.parametrize('frames,expected', [
    # Columns: spawned, alive, persistent wounds, battlefield HP (-1 outside combat).
    ([(0, 0, 0, -1), (0, 0, 0, -1)], 0.),
    ([(0, 0, 0, -1), (0, 0, 0, -1), (1, 1, 0, -1)], 0.),
    ([(0, 0, 0, -1), (0, 0, 0, -1), (1, 1, 40, -1), (1, 1, 10, -1)], .4),
    ([(0, 0, 0, -1), (0, 0, 0, -1), (1, 0, 0, -1)], 1.),
    ([(1, 1, 0, -1), (1, 1, 40, -1)], .4),
    ([(1, 1, 0, -1), (1, 1, 0, 75)], .25),
    ([(1, 1, 0, -1), (1, 0, 0, -1)], 1.),
])
def test_evaluation_damage_distinguishes_unspawned_alive_and_defeated(monkeypatch, frames, expected, greedy):
    episode = ObjectiveEpisode(frames)
    monkeypatch.setattr(benchmark, 'NumberGrid', lambda **kwargs: episode)
    monkeypatch.setattr(benchmark, 'wrap_wall_action_mask', lambda env: env)

    def distribution(params, observation):
        actions = jnp.zeros(observation.shape[0], jnp.int32)
        return SimpleNamespace(mode=lambda: actions, sample=lambda seed: actions)

    evaluate = benchmark.make_evaluator(SimpleNamespace(apply=distribution), episodes=3, greedy=greedy)
    result = evaluate(None)
    assert result['mean_target_damage_fraction'] == pytest.approx(expected)
    assert result['maximum_target_damage_fraction'] == pytest.approx(expected)
    assert result['successes'] == (3 if frames[-1][0] and not frames[-1][1] else 0)
    assert result['max_defeated_armies'] == int(bool(frames[-1][0] and not frames[-1][1]))
    assert result['action_selection'] == ('argmax' if greedy else 'sample')
