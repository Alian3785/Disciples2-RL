"""Brute experiment: hide game observations and state-dependent action masks."""
import jax.numpy as jnp
from stoa import ArraySpace, DictSpace
from stoa.core_wrappers.wrapper import Wrapper


class BruteObservation(Wrapper):
    """Keep the original input shape, but give actor and critic constants only.

    Rewards, episode boundaries, transitions and reporting stay with the base
    environment. Legal-action checks still run inside the game; their results
    are not exposed to the policy, including through timestep extras.
    """

    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space()
        self._hidden_observation = jnp.zeros(space.shape, dtype=space.dtype)
        self._all_actions = jnp.ones((env.action_space().num_values,), dtype=jnp.bool_)

    def _hide(self, timestep):
        observation = {'observation': self._hidden_observation,
                       'action_mask': self._all_actions}
        extras = {**timestep.extras, 'action_mask': self._all_actions}
        return timestep.replace(observation=observation, extras=extras)

    def reset(self, rng_key, env_params=None):
        state, timestep = self._env.reset(rng_key, env_params)
        return state, self._hide(timestep)

    def step(self, state, action, env_params=None):
        state, timestep = self._env.step(state, action, env_params)
        return state, self._hide(timestep)

    def observation_space(self, env_params=None):
        return DictSpace({
            'observation': self._env.observation_space(env_params),
            'action_mask': ArraySpace(self._all_actions.shape, self._all_actions.dtype),
        })
