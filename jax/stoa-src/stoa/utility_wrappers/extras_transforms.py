from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment


class ConsistentExtrasWrapper(Wrapper[State]):
    """Ensures TimeStep.extras has consistent structure for JAX scanning.

    This wrapper performs a single dummy step during initialization to discover
    all possible keys in extras, then ensures these keys are always present
    (with zero values when missing) in both reset and step. This can cause a
    significant memory overhead if extras contain large arrays and a start
    up cost due to the dummy step, but is necessary for JAX scanning.

    Alternatively, users can manually ensure consistent extras structure
    by modifying their environment's reset and step methods. If one does not need
    all the extras a base env provides, consider using SpecificExtrasWrapper or
    NoExtrasWrapper for much lower overhead and faster initialisation.
    """

    def __init__(
        self,
        env: Environment,
        rng_key: Optional[PRNGKey] = None,
        env_params: Optional[EnvParams] = None,
        do_dummy_step: bool = True,
    ):
        """Initialize by discovering extras structure via a dummy step.

        Args:
            env: The environment to wrap.
            rng_key: Random key for the dummy reset/step.
            env_params: Optional environment parameters.
            do_dummy_step: Whether to perform the dummy step to discover extras.
        """
        super().__init__(env)

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Do a dummy reset and step to discover extras structure
        reset_key, step_key = jax.random.split(rng_key)
        dummy_state, reset_timestep = env.reset(reset_key, env_params)
        # Merge extras from both reset and step (if step is done)
        all_extras = dict(reset_timestep.extras)

        if do_dummy_step:
            dummy_action = env.action_space(env_params).sample(step_key)
            _, step_timestep = env.step(dummy_state, dummy_action, env_params)
            all_extras.update(step_timestep.extras)

        # Create zero-filled template
        self._extras_template = jax.tree_map(lambda x: jnp.zeros_like(x), all_extras)

    def _fill_extras(self, extras: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing keys with zeros from template."""
        filled = dict(self._extras_template)
        filled.update(extras)
        return filled

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(rng_key, env_params)
        return state, timestep.replace(extras=self._fill_extras(timestep.extras))  # type: ignore

    def step(
        self, state: State, action: Action, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        new_state, timestep = self._env.step(state, action, env_params)
        return new_state, timestep.replace(extras=self._fill_extras(timestep.extras))  # type: ignore


class SpecificExtrasWrapper(Wrapper[State]):
    """Ensures TimeStep.extras has specific keys with None values when missing.

    This wrapper selects only specific keys from the base environment's extras
    and ensures these keys are always present (with None values when missing)
    in both reset and step. All other extras keys are discarded.
    """

    def __init__(self, env: Environment, extras_keys: List[str]):
        """Initializes the wrapper with a specific set of keys to keep.

        Args:
            env: The environment to wrap.
            extras_keys: A list of string keys to select from the extras.
                         If a key is not present in the original extras, its
                         value will be set to None.
        """
        super().__init__(env)
        self._extras_keys = extras_keys

    def _select_extras(self, extras: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filters the extras dictionary to include only the specified keys.

        Args:
            extras: The original extras dictionary from a TimeStep.

        Returns:
            A new dictionary containing only the keys specified during
            initialization. Missing keys are assigned a value of None.
        """
        # Create a new dictionary by iterating through the desired keys.
        # Use dict.get(key, None) to provide a default value of None if the
        # key is not found in the original extras dictionary.
        selected_extras = {key: extras.get(key) for key in self._extras_keys}
        return selected_extras

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """
        Resets the environment and filters the extras of the resulting TimeStep.

        Args:
            rng_key: A JAX random key.
            env_params: Optional parameters for the environment.

        Returns:
            A tuple containing the initial state and a TimeStep with filtered extras.
        """
        # Call the reset method of the wrapped environment.
        state, timestep = self._env.reset(rng_key, env_params)

        # Filter the extras from the returned timestep.
        filtered_extras = self._select_extras(timestep.extras)

        # Return the state and a new timestep with the filtered extras.
        return state, timestep.replace(extras=filtered_extras)  # type: ignore

    def step(
        self, state: State, action: Action, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """
        Steps the environment and filters the extras of the resulting TimeStep.

        Args:
            state: The current state of the environment.
            action: The action to take in the environment.
            env_params: Optional parameters for the environment.

        Returns:
            A tuple containing the new state and a TimeStep with filtered extras.
        """
        # Call the step method of the wrapped environment.
        new_state, timestep = self._env.step(state, action, env_params)

        # Filter the extras from the returned timestep.
        filtered_extras = self._select_extras(timestep.extras)

        # Return the new state and a new timestep with the filtered extras.
        return new_state, timestep.replace(extras=filtered_extras)  # type: ignore


class NoExtrasWrapper(Wrapper[State]):
    """Removes all base environment extras by setting TimeStep.extras to an empty dict."""

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """
        Resets the environment and returns a TimeStep with empty extras.

        Args:
            rng_key: A JAX random key.
            env_params: Optional parameters for the environment.

        Returns:
            A tuple containing the initial state and a TimeStep with empty extras.
        """
        # Call the reset method of the wrapped environment.
        state, timestep = self._env.reset(rng_key, env_params)
        # Return the state and a new timestep with empty extras.
        return state, timestep.replace(extras={})  # type: ignore

    def step(
        self, state: State, action: Action, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """
        Steps the environment and returns a TimeStep with empty extras.

        Args:
            state: The current state of the environment.
            action: The action to take in the environment.
            env_params: Optional parameters for the environment.

        Returns:
            A tuple containing the new state and a TimeStep with empty extras.
        """
        # Call the step method of the wrapped environment.
        new_state, timestep = self._env.step(state, action, env_params)
        # Return the new state and a new timestep with empty extras.
        return new_state, timestep.replace(extras={})  # type: ignore
