from typing import Any, Optional, Tuple

import jax.numpy as jnp
from chex import PRNGKey
from jumanji.env import Environment as JumanjiEnvironment
from jumanji.specs import Array, BoundedArray, DiscreteArray, MultiDiscreteArray, Spec

from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import (
    ArraySpace,
    BoundedArraySpace,
    DictSpace,
    DiscreteSpace,
    MultiDiscreteSpace,
    Space,
)


def jumanji_spec_to_stoa_space(spec: Spec) -> Space:
    """Converts Jumanji specs to Stoa spaces."""
    if isinstance(spec, DiscreteArray):
        return DiscreteSpace(num_values=spec.num_values, dtype=spec.dtype)
    elif isinstance(spec, MultiDiscreteArray):
        return MultiDiscreteSpace(
            num_values=spec.num_values,
            dtype=spec.dtype,
        )
    elif isinstance(spec, BoundedArray):
        return BoundedArraySpace(
            shape=spec.shape,
            dtype=spec.dtype,
            minimum=spec.minimum,
            maximum=spec.maximum,
        )
    elif isinstance(spec, Array):
        return ArraySpace(shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, Spec):
        # Handle nested specs (e.g., DictSpec)
        nested_spaces = {
            # Iterate over specs
            f"{key}": jumanji_spec_to_stoa_space(value)
            for key, value in vars(spec).items()
            if isinstance(value, Spec)
        }
        return DictSpace(spaces=nested_spaces)
    else:
        raise TypeError(f"Unsupported Jumanji spec type: {type(spec)}")


class JumanjiToStoa(Environment):
    """Jumanji environments in Stoa interface."""

    def __init__(self, env: JumanjiEnvironment):
        """Initialize the Jumanji Stoa Adapter.

        Args:
            env: The Jumanji environment to wrap.
        """
        self._env = env

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[Any, TimeStep]:
        """Reset the environment."""
        # Reset the Jumanji environment
        jumanji_state, jumanji_timestep = self._env.reset(rng_key)

        # Convert Jumanji timestep to Stoa timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=jumanji_timestep.observation,
            extras=jumanji_timestep.extras,
        )

        return jumanji_state, timestep

    def step(
        self,
        state: Any,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[Any, TimeStep]:
        """Step the environment."""
        # Take a Jumanji step
        next_state, jumanji_timestep = self._env.step(state, action)

        # Jumanji has the same step_type as Stoa except for truncation
        step_type = jumanji_timestep.step_type

        # Create the Stoa timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(jumanji_timestep.reward, dtype=jnp.float32),
            discount=jnp.asarray(jumanji_timestep.discount, dtype=jnp.float32),
            observation=jumanji_timestep.observation,
            extras=jumanji_timestep.extras,
        )

        return next_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space."""
        jumanji_obs_spec = self._env.observation_spec
        return jumanji_spec_to_stoa_space(jumanji_obs_spec)

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the action space."""
        jumanji_action_spec = self._env.action_spec
        return jumanji_spec_to_stoa_space(jumanji_action_spec)

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the state space."""
        raise NotImplementedError(
            "Jumanji does not expose a state space. Use observation_space instead."
        )

    def render(self, state: Any, env_params: Optional[EnvParams] = None) -> Any:
        """Render the environment."""
        if hasattr(self._env, "render"):
            return self._env.render(state)
        else:
            raise NotImplementedError(f"Rendering not supported for {self._env.__class__.__name__}")
