from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from navix.environments import Environment as NavixEnvironment
from navix.environments import Timestep as NavixTimestep

from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, DiscreteSpace, Space


class NavixToStoa(Environment):
    """Navix environments in Stoa interface."""

    def __init__(self, env: NavixEnvironment):
        """Initialize the Navix adapter.

        Args:
            env: The Navix environment to adapt.
        """
        self._env = env
        self._n_actions = len(env.action_set)

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[NavixTimestep, TimeStep]:
        """Reset the environment.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters (unused for Navix).

        Returns:
            Tuple of (navix_timestep, stoa_timestep).
        """
        # Reset the Navix environment
        navix_timestep = self._env.reset(rng_key)

        # Create Stoa timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=navix_timestep.observation.astype(jnp.float32),
            extras={
                "step_count": navix_timestep.t.astype(jnp.int32),
                **navix_timestep.info,  # Include any additional info from Navix
            },
        )

        return navix_timestep, timestep

    def step(
        self,
        state: NavixTimestep,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[NavixTimestep, TimeStep]:
        """Step the environment.

        Args:
            state: Current Navix timestep state.
            action: Action to take.
            env_params: Optional environment parameters (unused for Navix).

        Returns:
            Tuple of (new_navix_timestep, stoa_timestep).
        """
        # Step the Navix environment
        new_navix_timestep = self._env.step(state, action)

        # Determine step type and discount based on Navix termination/truncation
        terminal = new_navix_timestep.is_termination()
        truncated = new_navix_timestep.is_truncation()

        # Determine step type
        step_type = jax.lax.select(
            terminal,
            StepType.TERMINATED,
            jax.lax.select(truncated, StepType.TRUNCATED, StepType.MID),
        )

        # Discount is 0 for termination, 1 for truncation or continuation
        discount = jax.lax.select(
            terminal,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )

        # Create Stoa timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=new_navix_timestep.reward.astype(jnp.float32),
            discount=discount,
            observation=new_navix_timestep.observation.astype(jnp.float32),
            extras={
                "step_count": new_navix_timestep.t.astype(jnp.int32),
                **new_navix_timestep.info,  # Include any additional info from Navix
            },
        )

        return new_navix_timestep, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space.

        Args:
            env_params: Optional environment parameters (unused for Navix).

        Returns:
            The observation space corresponding to Navix observation space.
        """
        navix_obs_space = self._env.observation_space

        return BoundedArraySpace(
            shape=navix_obs_space.shape,
            dtype=jnp.float32,
            minimum=navix_obs_space.minimum,
            maximum=navix_obs_space.maximum,
            name="observation",
        )

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the action space.

        Args:
            env_params: Optional environment parameters (unused for Navix).

        Returns:
            The action space (discrete) corresponding to Navix action set.
        """
        return DiscreteSpace(
            num_values=self._n_actions,
            dtype=jnp.int32,
            name="action",
        )

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the state space.

        Args:
            env_params: Optional environment parameters (unused for Navix).

        Returns:
            A generic space for the Navix state.
        """
        raise NotImplementedError(
            "Navix does not expose a state space. Use observation_space instead."
        )

    def render(self, state: NavixTimestep, env_params: Optional[EnvParams] = None) -> Any:
        """Render the environment.

        Args:
            state: Current Navix timestep state.
            env_params: Optional environment parameters (unused for Navix).

        Returns:
            Rendered environment (if supported).

        Raises:
            NotImplementedError: If rendering is not supported.
        """
        if hasattr(self._env, "render"):
            return self._env.render(state)
        else:
            raise NotImplementedError(f"Rendering not supported for {self._env.__class__.__name__}")
