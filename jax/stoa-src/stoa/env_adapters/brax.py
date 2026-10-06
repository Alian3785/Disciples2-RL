from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from brax.envs.base import Env as BraxEnv
from brax.envs.base import State as BraxState
from chex import PRNGKey

from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, Space


class BraxToStoa(Environment):
    """Brax environments in stoa interface."""

    def __init__(self, env: BraxEnv):
        """Initialize the Brax Stoa Adapter.

        Args:
            env: The Brax environment to wrap.
        """
        self._env = env
        # Cache action and observation dimensions
        self._action_size = env.action_size
        self._obs_size = env.observation_size

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[BraxState, TimeStep]:
        """Reset the environment."""
        # Reset the Brax environment
        brax_state = self._env.reset(rng_key)

        # Create the initial timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=brax_state.obs,
            extras={**brax_state.info},
        )

        return brax_state, timestep

    def step(
        self,
        state: BraxState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[BraxState, TimeStep]:
        """Step the environment."""
        # Take a Brax step (no RNG needed for deterministic stepping)
        next_brax_state = self._env.step(state, action)

        # Determine step type and discount
        # Brax uses 'done' for termination and 'truncation' in info for truncation
        terminated = next_brax_state.done
        truncated = next_brax_state.info.get("truncation", jnp.array(False)).astype(jnp.bool_)

        # Determine step type based on termination/truncation
        step_type = jax.lax.select(
            jnp.logical_and(terminated, jnp.logical_not(truncated)),
            StepType.TERMINATED,
            jax.lax.select(truncated, StepType.TRUNCATED, StepType.MID),
        )

        # Discount is 0 for termination, 1 for truncation or continuation
        discount = jnp.where(jnp.logical_and(terminated, jnp.logical_not(truncated)), 0.0, 1.0)

        # Create the timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(next_brax_state.reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            observation=next_brax_state.obs,
            extras={
                "metrics": next_brax_state.metrics,
                **next_brax_state.info,
            },
        )

        return next_brax_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space."""
        return BoundedArraySpace(
            shape=(self._obs_size,),
            dtype=jnp.float32,
            minimum=-jnp.inf,
            maximum=jnp.inf,
            name="observation",
        )

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the action space."""
        # Brax environments typically have continuous actions in [-1, 1]
        return BoundedArraySpace(
            shape=(self._action_size,),
            dtype=jnp.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the state space."""
        raise NotImplementedError(
            "Brax does not expose a state space. Use observation_space instead."
        )

    def render(self, state: BraxState, env_params: Optional[EnvParams] = None) -> Any:
        """Render the environment."""
        if hasattr(self._env, "render"):
            # Some Brax environments support rendering
            return self._env.render(state)
        else:
            raise NotImplementedError(f"Rendering not supported for {self._env.__class__.__name__}")
