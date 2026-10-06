from typing import Any, Optional, Tuple, Union

import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from gymnax import EnvParams as GymnaxEnvParams
from gymnax.environments.environment import Environment as GymnaxEnvironment

from stoa.env_adapters.base import AdapterStateWithKey
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, BoundedArraySpace, DictSpace, DiscreteSpace, Space


def gymnax_space_to_stoa_space(
    space: Union[gymnax_spaces.Discrete, gymnax_spaces.Box, gymnax_spaces.Dict]
) -> Space:
    """Converts Gymnax spaces to stoa spaces."""
    if isinstance(space, gymnax_spaces.Discrete):
        return DiscreteSpace(num_values=space.n, dtype=jnp.int32)
    elif isinstance(space, gymnax_spaces.Box):
        # Check if the space is bounded
        bounded_below = np.all(np.isfinite(space.low))
        bounded_above = np.all(np.isfinite(space.high))

        if bounded_below and bounded_above:
            return BoundedArraySpace(
                shape=space.shape,
                dtype=space.dtype,
                minimum=space.low,
                maximum=space.high,
            )
        elif bounded_below:
            return BoundedArraySpace(
                shape=space.shape,
                dtype=space.dtype,
                minimum=space.low,
                maximum=jnp.inf,
            )
        elif bounded_above:
            return BoundedArraySpace(
                shape=space.shape,
                dtype=space.dtype,
                minimum=-jnp.inf,
                maximum=space.high,
            )
        else:
            # Unbounded space
            return ArraySpace(shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gymnax_spaces.Dict):
        # Convert nested dict spaces
        stoa_spaces = {
            key: gymnax_space_to_stoa_space(value) for key, value in space.spaces.items()
        }
        return DictSpace(spaces=stoa_spaces)
    else:
        raise TypeError(f"Unsupported Gymnax space type: {type(space)}")


class GymnaxToStoa(Environment):
    """Gymnax environments in stoa interface."""

    def __init__(
        self,
        env: GymnaxEnvironment,
        env_params: Optional[GymnaxEnvParams] = None,
    ):
        """Initialize the Gymnax adapter.

        Args:
            env: The Gymnax environment to adapt.
            env_params: Optional environment parameters.
        """
        self._env = env
        self._env_params: GymnaxEnvParams = env_params or env.default_params

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[AdapterStateWithKey, TimeStep]:
        """Reset the environment."""
        reset_key, state_key = jax.random.split(rng_key)

        # If env_params are not provided use the default
        if env_params is None:
            env_params = self._env_params

        # Reset the gymnax environment
        obs, gymnax_state = self._env.reset(reset_key, env_params)

        # Wrap the state with an rng key
        state = AdapterStateWithKey(
            base_env_state=gymnax_state,
            rng_key=state_key,
        )

        # Create the timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=obs,
            extras={},
        )

        return state, timestep

    def step(
        self,
        state: AdapterStateWithKey,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[AdapterStateWithKey, TimeStep]:
        """Step the environment."""
        step_key, next_key = jax.random.split(state.rng_key)

        # If env_params are not provided use the default
        if env_params is None:
            env_params = self._env_params

        # Take a gymnax step
        obs, gymnax_state, reward, done, info = self._env.step(
            step_key, state.base_env_state, action, env_params
        )

        # Wrap the state with a new key
        new_state = AdapterStateWithKey(
            base_env_state=gymnax_state,
            rng_key=next_key,
        )

        # Gymnax has no truncation, only termination
        step_type = jax.lax.select(done, StepType.TERMINATED, StepType.MID)

        # Create the timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(1.0 - done, dtype=jnp.float32),
            observation=obs,
            extras={**info},
        )

        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space."""
        if env_params is None:
            env_params = self._env_params
        gymnax_obs_space = self._env.observation_space(env_params)
        return gymnax_space_to_stoa_space(gymnax_obs_space)

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the action space."""
        if env_params is None:
            env_params = self._env_params
        gymnax_action_space = self._env.action_space(env_params)
        return gymnax_space_to_stoa_space(gymnax_action_space)

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the state space."""
        raise NotImplementedError(
            "Gymnax does not expose a state space. Use observation_space instead."
        )

    def render(self, state: AdapterStateWithKey, env_params: Optional[EnvParams] = None) -> Any:
        """Render the environment."""
        if env_params is None:
            env_params = self._env_params
        if hasattr(self._env, "render"):
            return self._env.render(state.base_env_state, env_params)
        else:
            raise NotImplementedError(f"Rendering not supported for {self._env.__class__.__name__}")
