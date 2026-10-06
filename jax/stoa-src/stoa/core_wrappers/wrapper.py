import dataclasses
from typing import Any, Generic, Optional, Tuple, TypeVar

import jax
from chex import PRNGKey

from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, EnvironmentSpace, Space
from stoa.stoa_struct import dataclass


def wrapper_state_replace(self: "WrapperState", **changes: Any) -> "WrapperState":
    """Replace that can update attributes in nested wrapper states."""
    # Separate changes into those for this level vs nested levels
    my_fields = {f.name for f in self.__dataclass_fields__.values()}
    local_changes = {k: v for k, v in changes.items() if k in my_fields}
    nested_changes = {k: v for k, v in changes.items() if k not in my_fields} or None

    if nested_changes is not None:
        # Only try to replace if base_env_state is a WrapperState
        if isinstance(self.base_env_state, WrapperState):
            # Recursively update the nested wrapper state
            new_base_state = self.base_env_state.replace(**nested_changes)
            local_changes["base_env_state"] = new_base_state
        else:
            # We've reached a non-wrapper state, can't go deeper
            raise AttributeError(
                f"Cannot update attributes {list(nested_changes.keys())} - "
                f"they are not accessible through wrapper states"
            )

    # Use dataclass replace for local changes
    return dataclasses.replace(self, **local_changes)


@dataclass(custom_replace_fn=wrapper_state_replace)
class WrapperState:
    """Base state class for environment wrappers."""

    base_env_state: Any

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the base environment state if not found."""
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.base_env_state, name)

    def get_inner_state_at(self, depth: int) -> "WrapperState":
        """
        Access the wrapped state at a specific depth.

        Args:
            depth: The desired depth. `depth=0` returns self, `depth=1` returns
                   the wrapped state below the top level, and so on.
        """
        if depth == 0:
            return self

        current_state = self
        # We iterate 'depth' times to go 'depth' levels down.
        for _ in range(depth):
            # Check if we can go deeper
            if not isinstance(current_state, WrapperState):
                raise IndexError(f"Cannot access depth {depth}. Maximum depth is {self.depth}.")
            current_state = current_state.base_env_state

        return current_state

    @property
    def unwrapped_state(self) -> Any:
        """Get the deepest non-wrapper state."""
        current = self.base_env_state
        while isinstance(current, WrapperState):
            current = current.base_env_state
        return current

    @property
    def depth(self) -> int:
        """How many wrappers deep are we?"""
        if isinstance(self.base_env_state, WrapperState):
            return 1 + self.base_env_state.depth
        return 1


S = TypeVar("S", bound="State")


class Wrapper(Environment, Generic[S]):
    """
    Base class for stoa environment wrappers.

    This class wraps an existing environment, allowing for additional
    functionality to be layered on top of the base environment. By default,
    it delegates all methods to the wrapped environment.
    """

    def __init__(self, env: Environment):
        """
        Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__()
        self._env = env

    def __repr__(self) -> str:
        """Return a string representation of the wrapper and its wrapped environment."""
        return f"{self.__class__.__name__}({repr(self._env)})"

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the wrapped environment.

        Args:
            name: The attribute name.

        Returns:
            The attribute value from the wrapped environment.

        Raises:
            AttributeError: If the attribute is not found.
        """
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> Environment:
        """
        Returns the base (unwrapped) environment.

        Returns:
            The innermost wrapped environment.
        """
        return self._env.unwrapped

    def reset(self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None) -> Tuple[S, TimeStep]:
        """
        Reset the environment.

        Args:
            rng_key: A JAX PRNG key for random number generation.
            env_params: Optional environment parameters.

        Returns:
            A tuple of the initial state and the first TimeStep.
        """
        return self._env.reset(rng_key, env_params)

    def step(
        self,
        state: S,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[S, TimeStep]:
        """
        Take a step in the environment.

        Args:
            state: The current environment state.
            action: The action to take.
            env_params: Optional environment parameters.

        Returns:
            A tuple of the new state and the resulting TimeStep.
        """
        return self._env.step(state.base_env_state, action, env_params)

    def reward_space(self, env_params: Optional[EnvParams] = None) -> BoundedArraySpace:
        return self._env.reward_space(env_params)

    def discount_space(self, env_params: Optional[EnvParams] = None) -> BoundedArraySpace:
        return self._env.discount_space(env_params)

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return self._env.observation_space(env_params)

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return self._env.action_space(env_params)

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return self._env.state_space(env_params)

    def environment_space(self, env_params: Optional[EnvParams] = None) -> EnvironmentSpace:
        return self._env.environment_space(env_params)

    def render(self, state: S, env_params: Optional[EnvParams] = None) -> Any:
        return self._env.render(state.base_env_state, env_params)

    def close(self) -> None:
        self._env.close()


@dataclass(custom_replace_fn=wrapper_state_replace)
class StateWithKey(WrapperState):
    """
    Wrapper state that includes a JAX PRNG key.
    """

    rng_key: PRNGKey


class AddRNGKey(Wrapper[StateWithKey]):
    """
    Wrapper that adds a JAX PRNG key to the environment state.

    This allows environments that do not natively manage RNG keys to be
    compatible with other stoa wrappers.
    """

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[StateWithKey, TimeStep]:
        """
        Reset the environment and initialize the RNG key in the state.

        Args:
            rng_key: A JAX PRNG key for random number generation.
            env_params: Optional environment parameters.

        Returns:
            A tuple of the initial state (with RNG key) and the first TimeStep.
        """
        rng_key, wrapped_state_key = jax.random.split(rng_key)
        base_env_state, timestep = self._env.reset(rng_key, env_params)
        wrapped_env_state = StateWithKey(base_env_state, wrapped_state_key)
        return wrapped_env_state, timestep

    def step(
        self,
        state: StateWithKey,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[StateWithKey, TimeStep]:
        """
        Take a step in the environment, updating the RNG key in the state.

        Args:
            state: The current state, including the RNG key.
            action: The action to take.
            env_params: Optional environment parameters.

        Returns:
            A tuple of the new state (with updated RNG key) and the resulting TimeStep.
        """
        rng_key = state.rng_key
        base_env_state, timestep = self._env.step(state.base_env_state, action, env_params)
        rng_key, wrapped_state_key = jax.random.split(rng_key)
        wrapped_env_state = StateWithKey(base_env_state, wrapped_state_key)
        return wrapped_env_state, timestep
