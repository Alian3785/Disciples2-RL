from typing import Optional, Tuple

import jax
from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment


class VmapWrapper(Wrapper[State]):
    """
    Wrapper that vectorizes environment operations using JAX's vmap.
    """

    def __init__(
        self,
        env: Environment,
        num_envs: Optional[int] = None,
        vectorize_over_env_params: bool = False,
    ):
        """
        Initialize the vmap wrapper.

        Args:
            env: The environment to vectorize.
            num_envs: Number of parallel environments to create.
                This allows for using a single JAX PRNG key to reset multiple environments.
                If not set, then the wrapper expects keys in the shape of (num_envs, ...).
            vectorize_over_env_params: If True, the reset and step methods will be vectorized over the
                environment parameters. This is useful for environments that support domain randomization.
        """
        super().__init__(env)

        if num_envs is not None and num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")

        self._num_envs = num_envs
        self._vectorize_over_env_params = vectorize_over_env_params

        env_params_in_axes = 0 if vectorize_over_env_params else None

        # Create vectorized versions of reset and step
        # vmap over the first axis (batch dimension)
        self._vmap_reset = jax.vmap(
            self._env.reset,
            in_axes=(0, env_params_in_axes),  # vmap over rng_keys, and maybe env_params
            out_axes=(0, 0),
        )

        self._vmap_step = jax.vmap(
            self._env.step,
            in_axes=(
                0,
                0,
                env_params_in_axes,
            ),  # vmap over states and actions, and maybe env_params
            out_axes=(0, 0),
        )

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """
        Reset all parallel environments.
        """
        if self._num_envs is not None:
            # Split the rng_key into num_envs keys
            rng_key = jax.random.split(rng_key, self._num_envs)

        states, timesteps = self._vmap_reset(rng_key, env_params)
        return states, timesteps

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """
        Step all parallel environments.
        """
        new_states, timesteps = self._vmap_step(state, action, env_params)
        return new_states, timesteps
