from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from pgx import Env as PGXEnv

from stoa.env_adapters.base import AdapterStateWithKey
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, DiscreteSpace, Space


class PGXToStoa(Environment):
    """PGX environments in Stoa interface.

    Minimal adapter for PGX game environments. Handles multi-player games
    by exposing the current player's view and legal actions.
    """

    def __init__(self, env: PGXEnv):
        """Initialize the PGX adapter.

        Args:
            env: The PGX environment to wrap.
        """
        self._env = env

        # Cache environment properties
        self._num_actions = env.num_actions
        self._observation_shape = env.observation_shape

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[AdapterStateWithKey, TimeStep]:
        """Reset the environment."""
        init_key, state_key = jax.random.split(rng_key)

        # Initialize PGX environment
        pgx_state = self._env.init(init_key)

        # Wrap state with RNG key
        state = AdapterStateWithKey(
            base_env_state=pgx_state,
            rng_key=state_key,
        )

        # Extract observation and legal actions
        observation = pgx_state.observation.astype(jnp.float32)
        legal_action_mask = pgx_state.legal_action_mask.astype(jnp.float32)

        # Create extras with PGX-specific information
        extras = {
            "legal_action_mask": legal_action_mask,
            "current_player": pgx_state.current_player.astype(jnp.int32),
            "step_count": pgx_state._step_count,
        }

        # Create initial timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=observation,
            extras=extras,
        )

        return state, timestep

    def step(
        self,
        state: AdapterStateWithKey,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[AdapterStateWithKey, TimeStep]:
        """Step the environment."""
        step_key, next_key = jax.random.split(state.rng_key)

        # Step PGX environment
        pgx_state = self._env.step(state.base_env_state, action, step_key)

        # Wrap new state with RNG key
        new_state = AdapterStateWithKey(
            base_env_state=pgx_state,
            rng_key=next_key,
        )

        # Extract observation and legal actions
        observation = pgx_state.observation.astype(jnp.float32)
        legal_action_mask = pgx_state.legal_action_mask.astype(jnp.float32)

        reward = jnp.squeeze(pgx_state.rewards).astype(jnp.float32)

        # Check for termination/truncation
        terminated = jnp.squeeze(pgx_state.terminated).astype(jnp.bool_)

        # Determine step type
        step_type = jax.lax.select(
            terminated,
            StepType.TERMINATED,
            StepType.MID,
        )

        # Discount is 0 for termination, 1 for truncation or continuation
        discount = jnp.where(terminated, 0.0, 1.0).astype(jnp.float32)

        # Create extras with PGX-specific information
        extras = {
            "legal_action_mask": legal_action_mask,
            "current_player": pgx_state.current_player,
            "step_count": pgx_state._step_count,
        }

        # Create timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation,
            extras=extras,
        )

        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space."""
        return ArraySpace(
            shape=self._observation_shape,
            dtype=jnp.float32,
            name="observation",
        )

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the action space."""
        return DiscreteSpace(
            num_values=self._num_actions,
            dtype=jnp.int32,
            name="action",
        )

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the state space."""
        raise NotImplementedError(
            "PGX does not expose a state space. Use observation_space instead."
        )

    def render(self, state: AdapterStateWithKey, env_params: Optional[EnvParams] = None) -> Any:
        """Render the environment."""
        if hasattr(self._env, "render"):
            return self._env.render(state.base_env_state)
        else:
            raise NotImplementedError(f"Rendering not supported for {self._env.__class__.__name__}")

    def get_legal_actions(self, state: AdapterStateWithKey) -> jnp.ndarray:
        """Get legal actions for the current state.

        Args:
            state: Current environment state.

        Returns:
            Boolean array indicating legal actions.
        """
        return state.base_env_state.legal_action_mask.astype(jnp.bool_)

    def get_current_player(self, state: AdapterStateWithKey) -> jnp.ndarray:
        """Get the current player.

        Args:
            state: Current environment state.

        Returns:
            Current player index.
        """
        return state.base_env_state.current_player.astype(jnp.int32)
