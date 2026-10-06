from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.utility_wrappers.obs_transforms import StepCountState


class EpisodeStepLimitWrapper(Wrapper[StepCountState]):
    """
    Wrapper that enforces a maximum number of steps per episode.

    This wrapper tracks the number of steps taken in the current episode and
    automatically truncates episodes when they exceed the specified step limit.
    When truncation occurs, the step_type is set to StepType.TRUNCATED and the
    discount is maintained (typically 1.0) to indicate that the episode was
    truncated rather than naturally terminated.

    This is useful for environments that don't have natural episode boundaries
    or when you want to ensure episodes don't run indefinitely.
    """

    def __init__(self, env: Environment, max_episode_steps: int):
        """
        Initialize the episode step limit wrapper.

        Args:
            env: The environment to wrap.
            max_episode_steps: Maximum number of steps allowed per episode.
                               Episodes will be truncated after this many steps.

        Raises:
            ValueError: If max_episode_steps is not positive.
        """
        super().__init__(env)

        if max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {max_episode_steps}")

        self._max_episode_steps = max_episode_steps

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[StepCountState, TimeStep]:
        """
        Reset the environment and initialize the step counter.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (initial_state, initial_timestep) with step counter reset to 0.
        """
        # Reset the base environment
        base_state, timestep = self._env.reset(rng_key, env_params)

        # Create wrapper state with step counter initialized to 0
        state = StepCountState(
            base_env_state=base_state,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

        return state, timestep

    def step(
        self,
        state: StepCountState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[StepCountState, TimeStep]:
        """
        Step the environment and check for step limit truncation.

        Args:
            state: Current state including step counter.
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, new_timestep). If the step limit is reached,
            the timestep will have step_type set to StepType.TRUNCATED.
        """
        # Step the base environment
        new_base_state, timestep = self._env.step(state.base_env_state, action, env_params)

        # Increment step counter
        new_step_count = state.step_count + 1

        # Check if we've reached the step limit
        step_limit_reached = new_step_count >= self._max_episode_steps

        # If the episode is already done (terminated or truncated), don't override
        # the step type. Otherwise, check if we need to truncate due to step limit.
        should_truncate = jnp.logical_and(
            step_limit_reached, jnp.logical_not(timestep.terminated())
        )

        # Update step type if truncation is needed
        new_step_type = jax.lax.select(should_truncate, StepType.TRUNCATED, timestep.step_type)

        # Create updated timestep with potentially modified step type
        new_timestep = timestep.replace(step_type=new_step_type)  # type: ignore

        # Create new wrapper state
        new_state = StepCountState(
            base_env_state=new_base_state,
            step_count=new_step_count,
        )

        return new_state, new_timestep

    @property
    def max_episode_steps(self) -> int:
        """Get the maximum number of steps per episode."""
        return self._max_episode_steps
