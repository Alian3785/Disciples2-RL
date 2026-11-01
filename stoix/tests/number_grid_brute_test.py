"""GPU checks for hidden game features, preserved masks, and reward learning."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from omegaconf import OmegaConf
from stoa import AddActionMaskWrapper, ArraySpace, DiscreteSpace, StepType, TimeStep

from stoix.envs.number_grid import wrap_wall_action_mask
from stoix.networks.base import FeedForwardActor, FeedForwardCritic
from stoix.networks.heads import CategoricalHead, ScalarCriticHead
from stoix.networks.torso import MLPTorso
from stoix.utils.make_env import apply_core_wrappers
from stoix.wrappers.number_grid_brute import BruteObservation


class VisibleGame:
    """Independent toy game with changing features, masks, and terminal rewards."""
    battle_mode = False
    mask_walls = True

    def observation_space(self, env_params=None):
        return ArraySpace((3,), jnp.float32)

    def action_space(self, env_params=None):
        return DiscreteSpace(3, dtype=jnp.int32)

    def reset(self, key, env_params=None):
        return jnp.int32(1), self.timestep(jnp.int32(1), jnp.float32(0), first=True)

    def step(self, state, action, env_params=None):
        following = state + 1
        reward = jnp.where(action == state % 2, 3., -2.)
        return following, self.timestep(following, reward)

    @staticmethod
    def timestep(state, reward, first=False):
        done = state >= 3
        kind = StepType.FIRST if first else jnp.where(done, StepType.TERMINATED, StepType.MID)
        return TimeStep(step_type=jnp.asarray(kind), reward=reward,
                        discount=jnp.where(done, 0., 1.),
                        observation=jnp.asarray([state, state + 100, -state], jnp.float32),
                        extras={'action_mask': jnp.asarray([state % 2 == 0, state == 1, True]),
                                'solved_episode': done})


@pytest.fixture(autouse=True, scope='module')
def cuda_only():
    assert jax.default_backend() == 'gpu'


def assert_hidden(timestep, expected_mask, extras_mask=None):
    assert set(timestep.observation) == {'observation', 'action_mask'}
    np.testing.assert_array_equal(timestep.observation['observation'], 0)
    np.testing.assert_array_equal(timestep.observation['action_mask'], expected_mask)
    np.testing.assert_array_equal(timestep.extras['action_mask'],
                                  expected_mask if extras_mask is None else extras_mask)


def test_changing_features_are_hidden_and_original_masks_and_feedback_preserved():
    game = VisibleGame()
    hidden = wrap_wall_action_mask(game)
    state, first = hidden.reset(jax.random.PRNGKey(1))
    assert_hidden(first, [False, True, True])
    assert hidden.observation_space().spaces['observation'].shape == (3,)
    assert hidden.observation_space().spaces['action_mask'].shape == (3,)
    step = jax.jit(hidden.step)
    for action in [jnp.int32(1), jnp.int32(0)]:
        expected_state, expected = game.step(state, action)
        state, timestep = step(state, action)
        assert int(state) == int(expected_state)
        assert float(timestep.reward) == float(expected.reward) == 3.
        assert bool(timestep.done()) == bool(expected.done())
        assert float(timestep.discount) == float(expected.discount)
        assert bool(timestep.extras['solved_episode']) == bool(expected.extras['solved_episode'])
        assert_hidden(timestep, expected.extras['action_mask'])
    assert bool(timestep.done())


def test_unmasked_environment_keeps_original_spaces_and_extras():
    game = VisibleGame()
    game.mask_walls = False
    hidden = wrap_wall_action_mask(game)
    state, timestep = hidden.reset(jax.random.PRNGKey(1))
    np.testing.assert_array_equal(timestep.observation, 0)
    np.testing.assert_array_equal(timestep.extras['action_mask'], [False, True, True])
    assert hidden.action_space().num_values == game.action_space().num_values
    assert hidden.observation_space().shape == game.observation_space().shape
    expected_state, expected = game.step(state, jnp.int32(1))
    state, timestep = jax.jit(hidden.step)(state, jnp.int32(1))
    assert int(state) == int(expected_state)
    np.testing.assert_array_equal(timestep.extras['action_mask'], expected.extras['action_mask'])
    np.testing.assert_array_equal(timestep.observation, 0)


def test_batched_autoreset_preserves_distinct_terminal_and_reset_masks():
    config = OmegaConf.create({'arch': {'num_envs': 2},
                              'env': {'env_name': 'number_grid', 'use_cached_auto_reset': True}})
    env = apply_core_wrappers(wrap_wall_action_mask(VisibleGame()), config)
    baseline = apply_core_wrappers(AddActionMaskWrapper(VisibleGame()), config)
    state, timestep = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(2), 2))
    original_state, _ = jax.jit(baseline.reset)(jax.random.split(jax.random.PRNGKey(2), 2))
    step = jax.jit(env.step)
    original_step = jax.jit(baseline.step)
    for actions in [jnp.ones(2, jnp.int32), jnp.zeros(2, jnp.int32), jnp.ones(2, jnp.int32)]:
        original_state, expected = original_step(original_state, actions)
        state, timestep = step(state, actions)
        assert_hidden(timestep, expected.observation['action_mask'], expected.extras['action_mask'])
        np.testing.assert_array_equal(timestep.extras['next_obs']['observation'], 0)
        np.testing.assert_array_equal(timestep.extras['next_obs']['action_mask'],
                                      expected.extras['next_obs']['action_mask'])
        expected_extras = {**expected.extras, 'next_obs': {
            **expected.extras['next_obs'], 'observation': jnp.zeros((2, 3), jnp.float32)}}
        for actual, original in zip(jax.tree.leaves(timestep.extras), jax.tree.leaves(expected_extras)):
            np.testing.assert_array_equal(actual, original)
        if bool(jnp.all(timestep.done())):
            np.testing.assert_array_equal(timestep.extras['episode_metrics']['episode_return'], 6.)
            np.testing.assert_array_equal(timestep.extras['episode_metrics']['episode_length'], 2)
            np.testing.assert_array_equal(timestep.observation['action_mask'], [[False, True, True]] * 2)
            np.testing.assert_array_equal(timestep.extras['next_obs']['action_mask'],
                                          [[False, False, True]] * 2)
    np.testing.assert_array_equal(timestep.done(), False)


def test_actor_keeps_forbidden_actions_at_zero_and_learns_from_reward():
    env = BruteObservation(AddActionMaskWrapper(VisibleGame()))
    _, first = env.reset(jax.random.PRNGKey(3))
    other = first.replace(observation={**first.observation, 'observation': jnp.full(3, 50.)})
    other = env._hide(other)
    actor = FeedForwardActor(CategoricalHead(3), MLPTorso([8], activation='tanh'))
    critic = FeedForwardCritic(ScalarCriticHead(), MLPTorso([8], activation='tanh'))
    params = actor.init(jax.random.PRNGKey(3), first.observation)
    critic_params = critic.init(jax.random.PRNGKey(4), first.observation)
    np.testing.assert_array_equal(actor.apply(params, first.observation).probs_parameter(),
                                  actor.apply(params, other.observation).probs_parameter())
    np.testing.assert_array_equal(critic.apply(critic_params, first.observation),
                                  critic.apply(critic_params, other.observation))
    previous = actor.apply(params, first.observation)
    np.testing.assert_array_equal(previous.probs_parameter() > 0, [False, True, True])
    changed_mask = {**first.observation, 'action_mask': jnp.array([True, False, True])}
    np.testing.assert_array_equal(actor.apply(params, changed_mask).probs_parameter() > 0,
                                  [True, False, True])
    samples = previous.sample(128, seed=jax.random.PRNGKey(5))
    assert bool(jnp.all(first.observation['action_mask'][samples]))
    old_log_prob = jax.lax.stop_gradient(previous.log_prob(jnp.int32(1)))

    def ppo_loss(weights):
        log_prob = actor.apply(weights, first.observation).log_prob(jnp.int32(1))
        ratio = jnp.exp(log_prob - old_log_prob)
        return -jnp.minimum(ratio * 3., jnp.clip(ratio, .8, 1.2) * 3.)

    gradient = jax.grad(ppo_loss)(params)
    updated = optax.apply_updates(params, jax.tree.map(lambda x: -.05 * x, gradient))
    assert float(actor.apply(updated, first.observation).prob(jnp.int32(1))) > float(previous.prob(jnp.int32(1)))
