from typing import Optional, Tuple

import jax
from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper, WrapperState, wrapper_state_replace
from stoa.env_types import Action, EnvParams, Observation, State, TimeStep
from stoa.environment import Environment
from stoa.stoa_struct import dataclass

NEXT_OBS_KEY_IN_EXTRAS = "next_obs"


def add_obs_to_extras(timestep: TimeStep) -> TimeStep:
    """Place the observation in timestep.extras[NEXT_OBS_KEY_IN_EXTRAS].

    Used when auto-resetting to store the observation from the terminal TimeStep.
    This is particularly useful for algorithms that need access to the true final
    observation in truncated episodes.

    Args:
        timestep: TimeStep object containing the timestep returned by the environment.

    Returns:
        TimeStep with observation stored in extras[NEXT_OBS_KEY_IN_EXTRAS].
    """
    extras = {**timestep.extras, NEXT_OBS_KEY_IN_EXTRAS: timestep.observation}
    return timestep.replace(extras=extras)  # type: ignore


class AutoResetWrapper(Wrapper[State]):
    """Automatically resets the environment when episodes terminate.

    This wrapper intercepts terminal timesteps and automatically calls reset(),
    replacing the terminal observation with the initial observation of the new episode.
    Optionally preserves the original terminal observation in timestep.extras.

    The wrapper expects the environment state to have a 'rng_key' attribute that provides
    the source of randomness for automatic resets.
    """

    def __init__(self, env: Environment, next_obs_in_extras: bool = False):
        """Initialize the AutoResetWrapper.

        Args:
            env: The environment to wrap.
            next_obs_in_extras: If True, stores the terminal observation in
                timestep.extras[NEXT_OBS_KEY_IN_EXTRAS]. Useful for algorithms
                that need access to the true final observation in truncated episodes.
        """
        super().__init__(env)
        self.next_obs_in_extras = next_obs_in_extras

        if next_obs_in_extras:
            self._maybe_add_obs_to_extras = add_obs_to_extras
        else:
            self._maybe_add_obs_to_extras = lambda timestep: timestep

    def _auto_reset(self, state: State, timestep: TimeStep) -> Tuple[State, TimeStep]:
        """Reset the environment and update the timestep with the new initial observation.

        Called when an episode terminates. Generates a new random rng_key, resets the
        environment, and replaces the terminal observation with the reset observation
        while preserving other timestep properties (reward, done flag, etc.).

        Args:
            state: Current environment state (terminal).
            timestep: Current timestep (terminal).

        Returns:
            Tuple of (new_state, updated_timestep) where:
            - new_state is the reset environment state
            - updated_timestep contains the reset observation but preserves
              the terminal reward, done flag, and optionally the terminal
              observation in extras.
        """
        # Generate new random key for reset
        rng_key, _ = jax.random.split(state.rng_key)  # type: ignore
        reset_state, reset_timestep = self._env.reset(rng_key)
        reset_observation = reset_timestep.observation

        # Preserve terminal observation in extras if requested
        updated_timestep = self._maybe_add_obs_to_extras(timestep)

        # Replace observation with the reset observation while preserving
        # other timestep properties (reward, done, etc.)
        updated_timestep = updated_timestep.replace(observation=reset_observation)  # type: ignore

        return reset_state, updated_timestep

    def _handle_non_terminal(self, state: State, timestep: TimeStep) -> Tuple[State, TimeStep]:
        """Handle non-terminal timesteps by optionally adding observation to extras.

        Args:
            state: Current environment state.
            timestep: Current timestep (non-terminal).

        Returns:
            Tuple of (state, timestep) with observation optionally added to extras.
        """
        return state, self._maybe_add_obs_to_extras(timestep)

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (initial_state, initial_timestep). The initial_timestep
            will have the observation added to extras if next_obs_in_extras is True.
        """
        # Reset the base environment
        state, timestep = self._env.reset(rng_key, env_params)

        # Add observation to extras if requested (for consistency with step behavior)
        timestep = self._maybe_add_obs_to_extras(timestep)
        return state, timestep

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment with automatic resetting on termination.

        If the episode terminates after the step, the environment is automatically
        reset and the timestep observation is replaced with the reset observation.
        The terminal reward and done flag are preserved in the timestep.

        Args:
            state: Current environment state.
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, new_timestep). If the episode terminated:
            - new_state corresponds to the automatically reset environment
            - new_timestep preserves the terminal reward and done flag but
              contains the reset observation
            If the episode didn't terminate:
            - new_state is the stepped environment state
            - new_timestep is the stepped timestep, optionally with observation
              added to extras
        """
        # Step the environment
        state, timestep = self._env.step(state, action, env_params)

        state, timestep = jax.lax.cond(
            timestep.done(),
            self._auto_reset,
            self._handle_non_terminal,
            state,
            timestep,
        )

        return state, timestep


@dataclass(custom_replace_fn=wrapper_state_replace)
class CachedAutoResetState(WrapperState):
    """State for cached auto-reset wrapper."""

    cached_state: State
    cached_obs: Observation


class CachedAutoResetWrapper(Wrapper[CachedAutoResetState]):
    """Auto-reset wrapper that caches the initial reset for repeated use."""

    def __init__(self, env: Environment, next_obs_in_extras: bool = False):
        super().__init__(env)
        self._maybe_add_obs_to_extras = add_obs_to_extras if next_obs_in_extras else lambda x: x

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[CachedAutoResetState, TimeStep]:
        """Reset and cache the initial state and observation."""
        base_state, timestep = self._env.reset(rng_key, env_params)

        state = CachedAutoResetState(
            base_env_state=base_state,
            cached_state=base_state,
            cached_obs=timestep.observation,
        )

        return state, self._maybe_add_obs_to_extras(timestep)

    def step(
        self,
        state: CachedAutoResetState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[CachedAutoResetState, TimeStep]:
        """Step with cached auto-reset on episode termination."""
        new_base_state, timestep = self._env.step(state.base_env_state, action, env_params)

        def auto_reset(
            s: CachedAutoResetState, ts: TimeStep
        ) -> Tuple[CachedAutoResetState, TimeStep]:
            # Use cached state and observation directly
            new_state = CachedAutoResetState(
                base_env_state=s.cached_state,
                cached_state=s.cached_state,
                cached_obs=s.cached_obs,
            )

            # Store terminal obs in extras, then replace base observation with cached reset obs
            updated_ts = self._maybe_add_obs_to_extras(ts).replace(observation=s.cached_obs)  # type: ignore
            return new_state, updated_ts

        def no_reset(
            s: CachedAutoResetState, ts: TimeStep
        ) -> Tuple[CachedAutoResetState, TimeStep]:
            new_state = CachedAutoResetState(
                base_env_state=new_base_state,
                cached_state=s.cached_state,
                cached_obs=s.cached_obs,
            )
            return new_state, self._maybe_add_obs_to_extras(ts)

        state, timestep = jax.lax.cond(timestep.done(), auto_reset, no_reset, state, timestep)
        return state, timestep
