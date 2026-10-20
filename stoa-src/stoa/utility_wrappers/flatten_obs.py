from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, Observation, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, BoundedArraySpace, Space


class FlattenObservationWrapper(Wrapper[State]):
    """Simple wrapper that flattens observations to 1D arrays.

    This wrapper flattens multi-dimensional observations into 1D vectors,
    which is useful for algorithms that expect flat observation spaces
    (e.g., some policy gradient methods or value function approximators).
    """

    def __init__(self, env: Environment):
        """Initialize the flatten observation wrapper.

        Args:
            env: Environment to wrap.
        """
        super().__init__(env)

        # Get observation space to determine flattened shape
        obs_space = self._env.observation_space()
        if obs_space.shape is None:
            raise ValueError(
                "FlattenObservationWrapper requires an observation space with a defined shape. "
                f"Got observation space: {obs_space}"
            )

        self._original_shape = obs_space.shape
        self._flattened_shape = (int(np.prod(obs_space.shape)),)

    def _flatten_observation(self, observation: Observation) -> jnp.ndarray:
        """Flatten an observation to 1D.

        Args:
            observation: The observation to flatten.

        Returns:
            Flattened observation as a 1D array.
        """
        # Ensure observation is float32 and flatten
        flat_obs = jnp.asarray(observation, dtype=jnp.float32)
        return flat_obs.reshape(self._flattened_shape)

    def _process_timestep(self, timestep: TimeStep) -> TimeStep:
        """Process a timestep by flattening observations.

        Args:
            timestep: The timestep to process.

        Returns:
            Timestep with flattened observation.
        """
        # Flatten main observation
        flattened_obs = self._flatten_observation(timestep.observation)
        new_timestep: TimeStep = timestep.replace(observation=flattened_obs)  # type: ignore
        return new_timestep

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment and flatten the observation.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (state, timestep) with flattened observation.
        """
        state, timestep = self._env.reset(rng_key, env_params)
        flattened_timestep = self._process_timestep(timestep)
        return state, flattened_timestep

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment and flatten the observation.

        Args:
            state: Current environment state.
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, timestep) with flattened observation.
        """
        new_state, timestep = self._env.step(state, action, env_params)
        flattened_timestep = self._process_timestep(timestep)
        return new_state, flattened_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the flattened observation space.

        Args:
            env_params: Optional environment parameters.

        Returns:
            Flattened observation space (1D).
        """
        original_space = self._env.observation_space(env_params)

        if isinstance(original_space, BoundedArraySpace):
            # For bounded spaces, we need to flatten the bounds too
            if original_space.minimum.shape == ():
                # Scalar bounds remain scalar
                flattened_minimum = original_space.minimum
                flattened_maximum = original_space.maximum
            else:
                # Flatten bounds
                flattened_minimum = original_space.minimum.reshape(self._flattened_shape)
                flattened_maximum = original_space.maximum.reshape(self._flattened_shape)

            return BoundedArraySpace(
                shape=self._flattened_shape,
                dtype=jnp.float32,
                minimum=flattened_minimum,
                maximum=flattened_maximum,
                name=f"flattened_{original_space.name}"
                if original_space.name
                else "flattened_observation",
            )

        elif isinstance(original_space, ArraySpace):
            return ArraySpace(
                shape=self._flattened_shape,
                dtype=jnp.float32,
                name=f"flattened_{original_space.name}"
                if original_space.name
                else "flattened_observation",
            )

        else:
            raise ValueError(f"Unsupported space type for flattening: {type(original_space)}")
