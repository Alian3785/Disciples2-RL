"""GPU checks for hidden policy inputs, episode reset, and reward learning."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from omegaconf import OmegaConf
from stoa import ArraySpace, DiscreteSpace, StepType, TimeStep

from stoix.networks.base import FeedForwardActor, FeedForwardCritic
from stoix.networks.heads import CategoricalHead, ScalarCriticHead
from stoix.networks.torso import MLPTorso
from stoix.utils.make_env import apply_core_wrappers
from stoix.wrappers.number_grid_brute import BruteObservation


class VisibleGame:
    """Independent toy game with changing features, masks, and terminal rewards."""
    battle_mode = False

    def observation_space(self, env_params=None):
        return ArraySpace((3,), jnp.float32)

    def action_space(self, env_params=None):
        return DiscreteSpace(2, dtype=jnp.int32)

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
                        extras={'action_mask': jnp.asarray([state % 2 == 0, state % 2 == 1]),
                                'solved_episode': done})


@pytest.fixture(autouse=True, scope='module')
def cuda_only():
    assert jax.default_backend() == 'gpu'


def assert_hidden(timestep):
    assert set(timestep.observation) == {'observation', 'action_mask'}
    np.testing.assert_array_equal(timestep.observation['observation'], 0)
    np.testing.assert_array_equal(timestep.observation['action_mask'], True)
    np.testing.assert_array_equal(timestep.extras['action_mask'], True)


def test_changing_game_features_and_masks_are_hidden_without_changing_feedback():
    game = VisibleGame()
    hidden = BruteObservation(game)
    state, first = hidden.reset(jax.random.PRNGKey(1))
    assert_hidden(first)
    assert hidden.observation_space().spaces['observation'].shape == (3,)
    assert hidden.observation_space().spaces['action_mask'].shape == (2,)
    step = jax.jit(hidden.step)
    for action in [jnp.int32(1), jnp.int32(0)]:
        expected_state, expected = game.step(state, action)
        state, timestep = step(state, action)
        assert int(state) == int(expected_state)
        assert float(timestep.reward) == float(expected.reward) == 3.
        assert bool(timestep.done()) == bool(expected.done())
        assert float(timestep.discount) == float(expected.discount)
        assert bool(timestep.extras['solved_episode']) == bool(expected.extras['solved_episode'])
        assert_hidden(timestep)
    assert bool(timestep.done())


def test_batched_autoreset_also_hides_terminal_next_observation():
    config = OmegaConf.create({'arch': {'num_envs': 2},
                              'env': {'env_name': 'number_grid', 'use_cached_auto_reset': True}})
    env = apply_core_wrappers(BruteObservation(VisibleGame()), config)
    state, timestep = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(2), 2))
    step = jax.jit(env.step)
    for actions in [jnp.ones(2, jnp.int32), jnp.zeros(2, jnp.int32)]:
        state, timestep = step(state, actions)
        assert_hidden(timestep)
        np.testing.assert_array_equal(timestep.extras['next_obs']['observation'], 0)
        np.testing.assert_array_equal(timestep.extras['next_obs']['action_mask'], True)
    np.testing.assert_array_equal(timestep.done(), True)
    np.testing.assert_array_equal(timestep.extras['episode_metrics']['episode_return'], 6.)
    np.testing.assert_array_equal(timestep.extras['episode_metrics']['episode_length'], 2)
    state, timestep = step(state, jnp.ones(2, jnp.int32))
    assert_hidden(timestep)
    np.testing.assert_array_equal(timestep.done(), False)


def test_actor_and_critic_cannot_distinguish_states_and_policy_learns_reward():
    env = BruteObservation(VisibleGame())
    first = env._hide(VisibleGame.timestep(jnp.int32(1), jnp.float32(0)))
    other = env._hide(VisibleGame.timestep(jnp.int32(20), jnp.float32(50)))
    actor = FeedForwardActor(CategoricalHead(2), MLPTorso([8], activation='tanh'))
    critic = FeedForwardCritic(ScalarCriticHead(), MLPTorso([8], activation='tanh'))
    params = actor.init(jax.random.PRNGKey(3), first.observation)
    critic_params = critic.init(jax.random.PRNGKey(4), first.observation)
    np.testing.assert_array_equal(actor.apply(params, first.observation).probs_parameter(),
                                  actor.apply(params, other.observation).probs_parameter())
    np.testing.assert_array_equal(critic.apply(critic_params, first.observation),
                                  critic.apply(critic_params, other.observation))
    previous = actor.apply(params, first.observation)
    np.testing.assert_array_equal(previous.probs_parameter() > 0, True)
    old_log_prob = jax.lax.stop_gradient(previous.log_prob(jnp.int32(1)))

    def ppo_loss(weights):
        log_prob = actor.apply(weights, first.observation).log_prob(jnp.int32(1))
        ratio = jnp.exp(log_prob - old_log_prob)
        return -jnp.minimum(ratio * 3., jnp.clip(ratio, .8, 1.2) * 3.)

    gradient = jax.grad(ppo_loss)(params)
    updated = optax.apply_updates(params, jax.tree.map(lambda x: -.05 * x, gradient))
    assert float(actor.apply(updated, first.observation).prob(jnp.int32(1))) > float(previous.prob(jnp.int32(1)))
