from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey

from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, DiscreteSpace, MultiDiscreteSpace, Space


class MultiDiscreteToDiscreteWrapper(Wrapper[State]):
    """Converts multi-discrete action spaces to single discrete action spaces.

    This wrapper takes environments with multi-discrete action spaces (where each
    action is a vector of discrete values) and converts them to single discrete
    action spaces by flattening the action space using a factorization scheme.

    For example, if the original action space has shape [3, 4, 2] (meaning 3 choices
    for the first action dimension, 4 for the second, and 2 for the third), this
    wrapper will convert it to a single discrete space with 3*4*2=24 possible actions.
    """

    def __init__(self, env: Environment):
        """Initialize the multi-discrete to discrete wrapper.

        Args:
            env: Environment to wrap.

        Raises:
            ValueError: If the environment's action space is not MultiDiscreteSpace.
        """
        super().__init__(env)

        action_space = self._env.action_space()
        if not isinstance(action_space, MultiDiscreteSpace):
            raise ValueError(
                f"MultiDiscreteToDiscreteWrapper requires MultiDiscreteSpace, "
                f"got {type(action_space)}"
            )

        self._action_spec_num_values = action_space.num_values
        self._total_actions = int(np.prod(np.asarray(self._action_spec_num_values)))

    def apply_factorization(self, flat_action: Array) -> Array:
        """Convert a flat discrete action to multi-discrete action components.

        This applies the factorization to convert from the flattened action space
        back to the original multi-discrete action space.

        Args:
            flat_action: Single discrete action value.

        Returns:
            Multi-discrete action vector.
        """
        action_components = []
        remaining_action = flat_action
        n = self._action_spec_num_values.shape[0]

        # Factor the flat action into components
        for i in range(n - 1, 0, -1):
            remaining_action, remainder = jnp.divmod(
                remaining_action, self._action_spec_num_values[i]
            )
            action_components.append(remainder)
        action_components.append(remaining_action)

        # Stack components in correct order
        action = jnp.stack(
            list(reversed(action_components)),
            axis=-1,
            dtype=self._action_spec_num_values.dtype,
        )
        return action

    def inverse_factorization(self, multi_action: Array) -> Array:
        """Convert multi-discrete action components to flat discrete action.

        This is the inverse of apply_factorization, converting from multi-discrete
        actions back to the flattened action space.

        Args:
            multi_action: Multi-discrete action vector.

        Returns:
            Single discrete action value.
        """
        n = self._action_spec_num_values.shape[0]
        action_components = jnp.split(multi_action, n, axis=-1)

        # Combine components into flat action
        flat_action = action_components[0].squeeze(-1)
        for i in range(1, n):
            flat_action = self._action_spec_num_values[i] * flat_action + action_components[
                i
            ].squeeze(-1)

        return flat_action

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (state, timestep). No action conversion needed for reset.
        """
        return self._env.reset(rng_key, env_params)

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment with action space conversion.

        Args:
            state: Current environment state.
            action: Flat discrete action to convert to multi-discrete.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, timestep).
        """
        # Convert flat action to multi-discrete action
        multi_discrete_action = self.apply_factorization(action)

        # Step the environment with the converted action
        new_state, timestep = self._env.step(state, multi_discrete_action, env_params)

        return new_state, timestep

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the converted action space.

        Args:
            env_params: Optional environment parameters.

        Returns:
            Single discrete action space with total number of possible actions.
        """
        return DiscreteSpace(
            num_values=self._total_actions,
            dtype=jnp.int32,
            name="discrete_action",
        )


class MultiBoundedToBoundedWrapper(Wrapper[State]):
    """Flattens multi-dimensional bounded action spaces to 1D bounded action spaces.

    This wrapper takes environments with multi-dimensional continuous action spaces
    and flattens them to 1D while preserving the bounds. This is useful for algorithms
    that work better with 1D action vectors or for compatibility with certain
    function approximators.

    For example, if the original action space has shape (3, 4) with bounds [-1, 1],
    this wrapper will convert it to shape (12,) with the same bounds.
    """

    def __init__(self, env: Environment):
        """Initialize the multi-bounded to bounded wrapper.

        Args:
            env: Environment to wrap.

        Raises:
            ValueError: If the environment's action space is not BoundedArraySpace.
        """
        super().__init__(env)

        action_space = self._env.action_space()
        if not isinstance(action_space, BoundedArraySpace):
            raise ValueError(
                f"MultiBoundedToBoundedWrapper requires BoundedArraySpace, "
                f"got {type(action_space)}"
            )

        if action_space.shape is None:
            raise ValueError(
                "MultiBoundedToBoundedWrapper requires action space with defined shape"
            )

        self._original_action_shape = action_space.shape
        self._flattened_size = int(np.prod(np.asarray(action_space.shape)))
        self._original_action_space = action_space

    def _reshape_action(self, flat_action: Array) -> Array:
        """Reshape a flattened action back to original multi-dimensional shape.

        Args:
            flat_action: 1D action array.

        Returns:
            Action reshaped to original multi-dimensional shape.
        """
        return flat_action.reshape(self._original_action_shape)

    def _flatten_bounds(self, bounds: Array) -> Array:
        """Flatten bounds from original shape to 1D.

        Args:
            bounds: Bounds array in original shape.

        Returns:
            Flattened bounds array.
        """
        # Check if bounds are scalar (shape == ())
        is_scalar = bounds.ndim == 0
        return jax.lax.cond(
            is_scalar,
            lambda b: jnp.full((self._flattened_size,), b, dtype=b.dtype),
            lambda b: jnp.reshape(b, (self._flattened_size,)),
            bounds,
        )

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (state, timestep). No action conversion needed for reset.
        """
        return self._env.reset(rng_key, env_params)

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment with action shape conversion.

        Args:
            state: Current environment state.
            action: Flattened 1D action to reshape to original dimensions.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, timestep).
        """
        # Reshape flat action to original multi-dimensional shape
        reshaped_action = self._reshape_action(action)

        # Step the environment with the reshaped action
        new_state, timestep = self._env.step(state, reshaped_action, env_params)

        return new_state, timestep

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the flattened action space.

        Args:
            env_params: Optional environment parameters.

        Returns:
            1D bounded action space with flattened bounds.
        """
        original_space = self._original_action_space

        # Flatten the bounds
        flattened_minimum = self._flatten_bounds(original_space.minimum)
        flattened_maximum = self._flatten_bounds(original_space.maximum)

        return BoundedArraySpace(
            shape=(self._flattened_size,),
            dtype=original_space.dtype,
            minimum=flattened_minimum,
            maximum=flattened_maximum,
            name=f"flattened_{original_space.name}" if original_space.name else "flattened_action",
        )
