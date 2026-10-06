from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jax import Array

from stoa.core_wrappers.auto_reset import add_obs_to_extras
from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, Observation, State, TimeStep
from stoa.environment import Environment


class OptimisticResetVmapWrapper(Wrapper[State]):
    """
    Efficient vectorized environment wrapper with optimistic resets.

    This wrapper combines environment vectorization (vmap) with automatic resets,
    using an "optimistic" strategy that pre-generates reset states for efficiency.
    Instead of resetting environments one-by-one as they terminate, it generates
    a smaller number of reset states and distributes them to terminated environments.

    Note: This wrapper requires the environment state to have an 'rng_key' attribute.
    Use the AddRNGKey wrapper before this one if your environment doesn't have it.

    Args:
        env: The base environment to vectorize.
        num_envs: Number of parallel environments to run.
        reset_ratio: Number of environments per reset state generated.
                    Higher values are more efficient but may cause duplicate resets.
                    Must divide num_envs evenly.
    """

    def __init__(
        self, env: Environment, num_envs: int, reset_ratio: int, next_obs_in_extras: bool = False
    ):
        super().__init__(env)

        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if num_envs % reset_ratio != 0:
            raise ValueError(
                f"reset_ratio ({reset_ratio}) must evenly divide num_envs ({num_envs})"
            )

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        self.num_resets = num_envs // reset_ratio

        self._vmap_reset = jax.vmap(self._env.reset, in_axes=(0, None), out_axes=(0, 0))
        self._vmap_step = jax.vmap(self._env.step, in_axes=(0, 0, None), out_axes=(0, 0))

        self.next_obs_in_extras = next_obs_in_extras
        if next_obs_in_extras:
            self._maybe_add_obs_to_extras = add_obs_to_extras
        else:
            self._maybe_add_obs_to_extras = lambda timestep: timestep

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Resets all vectorized environments."""

        # If rng_key is a single key, split it into num_envs keys
        # else assume it has the shape (num_envs, ...)
        if jnp.logical_or(rng_key.ndim == 1, rng_key.ndim == 0):
            reset_keys = jax.random.split(rng_key, self.num_envs)
        else:
            reset_keys = rng_key

        # Reset all environments
        states, timesteps = self._vmap_reset(reset_keys, env_params)

        # For consistency, add the observation to extras if needed
        timesteps = self._maybe_add_obs_to_extras(timesteps)

        return states, timesteps

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Steps all vectorized environments with optimistic auto-resets."""
        # Step all environments
        new_states, timesteps = self._vmap_step(state, action, env_params)

        # Add the observation to extras if needed
        timesteps = self._maybe_add_obs_to_extras(timesteps)

        # Extract done flags and handle RNG keys for resets
        dones = timesteps.done()

        # Get RNG keys for reset generation and selection
        reset_keys_parent = new_states.rng_key

        # Get new RNG keys for resets
        reset_keys_base, reset_keys_select = jax.vmap(jax.random.split, out_axes=1)(
            reset_keys_parent
        )
        new_states = new_states.replace(rng_key=reset_keys_base)

        # Get num_resets reset keys
        reset_keys = reset_keys_select[: self.num_resets]
        # Perform env reset to get reset states and timesteps
        reset_states, reset_timesteps = self._vmap_reset(reset_keys, env_params)

        # Assign reset states to environments
        # First, create default assignments (each reset state maps to reset_ratio environments)
        reset_indices = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        # Use next available key
        rng_key_choice = reset_keys_select[self.num_resets]
        # Use weighted random selection to choose which environments get reset
        being_reset = jax.random.choice(
            rng_key_choice,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=dones.astype(jnp.float32),
            replace=False,
        )

        # Update reset_indices so selected environments map to unique reset states
        reset_indices = reset_indices.at[being_reset].set(jnp.arange(self.num_resets))

        # Gather the reset states and timesteps according to the mapping
        reset_states = jax.tree_map(lambda x: x[reset_indices], reset_states)
        reset_timesteps = jax.tree_map(lambda x: x[reset_indices], reset_timesteps)

        # Auto-reset: select between stepped state and reset state based on done flag
        def auto_reset(
            done: Array,
            state_reset: State,
            state_step: State,
            obs_reset: Observation,
            obs_step: Observation,
        ) -> Tuple[State, Observation]:
            state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_reset, state_step)
            obs = jax.lax.select(done, obs_reset, obs_step)
            return state, obs

        # Extract observations from reset and stepped timesteps
        reset_observations = reset_timesteps.observation
        stepped_observations = timesteps.observation

        # Vectorized auto-reset across all environments
        final_states, final_observations = jax.vmap(auto_reset)(
            dones, reset_states, new_states, reset_observations, stepped_observations
        )

        # Create new timestep with final observations
        final_timesteps = timesteps.replace(observation=final_observations)  # type: ignore

        return final_states, final_timesteps
