from typing import Optional, Tuple, Union

import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp
from chex import PRNGKey
from kinetix.environment.env import KinetixObservation
from kinetix.environment.spaces import MultiDiscrete
from kinetix.render.renderer_pixels import PixelsObservation

from stoa.env_adapters.base import AdapterStateWithKey
from stoa.env_adapters.gymnax import GymnaxToStoa, gymnax_space_to_stoa_space
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.spaces import MultiDiscreteSpace, Space


def kinetix_space_to_stoa_space(
    space: Union[gymnax_spaces.Discrete, gymnax_spaces.Box, gymnax_spaces.Dict, MultiDiscrete]
) -> Space:
    if isinstance(space, MultiDiscrete):
        return MultiDiscreteSpace(num_values=jnp.array(space.n))
    else:
        return gymnax_space_to_stoa_space(space)


class KinetixToStoa(GymnaxToStoa):
    """Kinetix environments in Stoa interface.

    Inherits from GymnaxToStoa and only overrides Kinetix-specific behavior
    for observation handling and metrics extraction.
    """

    def _fix_obs(self, obs: KinetixObservation) -> jnp.ndarray:
        """Fix observation to handle PixelsObservation objects.

        Args:
            obs: Raw observation from Kinetix environment.
        """
        if isinstance(obs, PixelsObservation):
            return obs.image
        return obs

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[AdapterStateWithKey, TimeStep]:
        """Reset the environment with Kinetix-specific observation handling."""
        reset_key, state_key = jax.random.split(rng_key)

        # If env_params are not provided use the default
        if env_params is None:
            env_params = self._env_params

        # Reset the environment (same as parent but we need access to raw obs)
        obs, kinetix_state = self._env.reset(reset_key, env_params)

        # Wrap the state with an rng key
        state = AdapterStateWithKey(
            base_env_state=kinetix_state,
            rng_key=state_key,
        )

        # Fix observation for Kinetix
        fixed_obs = self._fix_obs(obs)

        extras = {}
        extras["solve_rate"] = jnp.asarray(0.0, dtype=jnp.float32)
        extras["distance"] = jnp.asarray(0.0, dtype=jnp.float32)

        # Create the timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=fixed_obs,
            extras=extras,
        )

        return state, timestep

    def step(
        self,
        state: AdapterStateWithKey,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[AdapterStateWithKey, TimeStep]:
        """Step the environment with Kinetix-specific observation and metrics handling."""
        step_key, next_key = jax.random.split(state.rng_key)

        # If env_params are not provided use the default
        if env_params is None:
            env_params = self._env_params

        # Take a step
        obs, kinetix_state, reward, done, info = self._env.step(
            step_key, state.base_env_state, action, env_params
        )

        # Wrap the state with a new key
        new_state = AdapterStateWithKey(
            base_env_state=kinetix_state,
            rng_key=next_key,
        )

        # Fix observation for Kinetix
        fixed_obs = self._fix_obs(obs)

        # Extract Kinetix-specific metrics from info
        extras = {}
        extras["solve_rate"] = jnp.asarray(info["GoalR"], dtype=jnp.float32)
        extras["distance"] = jnp.asarray(info["distance"], dtype=jnp.float32)

        # Add any other info
        extras.update({k: v for k, v in info.items() if k not in ["GoalR", "distance"]})

        # Kinetix has no truncation, only termination (same as Gymnax)
        step_type = jax.lax.select(done, StepType.TERMINATED, StepType.MID)

        # Create the timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(1.0 - done, dtype=jnp.float32),
            observation=fixed_obs,
            extras=extras,
        )

        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space."""
        if env_params is None:
            env_params = self._env_params
        gymnax_obs_space = self._env.observation_space(env_params)
        return kinetix_space_to_stoa_space(gymnax_obs_space)

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the action space."""
        if env_params is None:
            env_params = self._env_params
        gymnax_action_space = self._env.action_space(env_params)
        return kinetix_space_to_stoa_space(gymnax_action_space)
