"""Brute experiment: hide game features while preserving action masks."""
import jax.numpy as jnp
from stoa import DictSpace
from stoa.core_wrappers.wrapper import Wrapper


class BruteObservation(Wrapper):
    """Keep feature shapes and action availability from the wrapped environment.

    Rewards, episode boundaries, transitions and reporting stay with the base
    environment. Only the game feature vector becomes constant zero; masks in
    observations and timestep extras pass through unchanged.
    """

    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space()
        if isinstance(space, DictSpace):
            space = space.spaces['observation']
        self._hidden_observation = jnp.zeros(space.shape, dtype=space.dtype)

    def _hide(self, timestep):
        observation = self._hidden_observation
        if isinstance(timestep.observation, dict):
            observation = {**timestep.observation, 'observation': observation}
        return timestep.replace(observation=observation)

    def reset(self, rng_key, env_params=None):
        state, timestep = self._env.reset(rng_key, env_params)
        return state, self._hide(timestep)

    def step(self, state, action, env_params=None):
        state, timestep = self._env.step(state, action, env_params)
        return state, self._hide(timestep)
