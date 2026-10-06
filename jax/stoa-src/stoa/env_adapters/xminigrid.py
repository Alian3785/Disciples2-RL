from typing import Any, Optional, Tuple

import jax.numpy as jnp
from chex import PRNGKey
from xminigrid.environment import Environment as XMiniGridEnvironment
from xminigrid.environment import EnvParams as XMiniGridEnvParams
from xminigrid.environment import TimeStep as XMiniGridState

from stoa.env_types import Action, EnvParams, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, DiscreteSpace, Space


class XMiniGridToStoa(Environment):
    """XMiniGrid environments in Stoa interface."""

    def __init__(self, env: XMiniGridEnvironment, env_params: Optional[XMiniGridEnvParams] = None):
        """Initialize the XMiniGrid adapter.

        Args:
            env: The XMiniGrid environment to adapt.
            env_params: Optional environment parameters. If None, uses default params.
        """
        self._env = env
        self._env_params = env_params or env.default_params()

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[XMiniGridState, TimeStep]:
        """Reset the environment.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters (uses XMiniGrid env_params if None).

        Returns:
            Tuple of (xminigrid_state, stoa_timestep).
        """
        # Use provided env_params or fall back to default
        xmg_env_params = env_params if env_params is not None else self._env_params

        # Reset the XMiniGrid environment
        xmg_timestep = self._env.reset(xmg_env_params, rng_key)

        # Convert XMiniGrid timestep to Stoa timestep
        # XMiniGrid timestep contains state, observation, reward, discount, step_type
        timestep = TimeStep(
            step_type=xmg_timestep.step_type,  # XMiniGrid uses same step types as Stoa
            reward=jnp.asarray(xmg_timestep.reward, dtype=jnp.float32),
            discount=jnp.asarray(xmg_timestep.discount, dtype=jnp.float32),
            observation=xmg_timestep.observation.astype(jnp.float32),
            extras={
                "step_count": xmg_timestep.state.step_num,
            },
        )

        return xmg_timestep, timestep

    def step(
        self,
        state: XMiniGridState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[XMiniGridState, TimeStep]:
        """Step the environment.

        Args:
            state: Current XMiniGrid state.
            action: Action to take.
            env_params: Optional environment parameters (uses XMiniGrid env_params if None).

        Returns:
            Tuple of (new_xminigrid_state, stoa_timestep).
        """
        # Use provided env_params or fall back to default
        xmg_env_params = env_params if env_params is not None else self._env_params

        # Step the XMiniGrid environment
        xmg_timestep = self._env.step(xmg_env_params, state, action)

        # Convert XMiniGrid timestep to Stoa timestep
        timestep = TimeStep(
            step_type=xmg_timestep.step_type,  # XMiniGrid uses same step types as Stoa
            reward=jnp.asarray(xmg_timestep.reward, dtype=jnp.float32),
            discount=jnp.asarray(xmg_timestep.discount, dtype=jnp.float32),
            observation=xmg_timestep.observation.astype(jnp.float32),
            extras={
                "step_count": xmg_timestep.state.step_num,
            },
        )

        return xmg_timestep, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space.

        Args:
            env_params: Optional environment parameters (uses XMiniGrid env_params if None).

        Returns:
            The observation space corresponding to XMiniGrid observation space.
        """
        xmg_env_params = env_params if env_params is not None else self._env_params
        obs_shape = self._env.observation_shape(xmg_env_params)

        return ArraySpace(
            shape=obs_shape,
            dtype=jnp.float32,
            name="observation",
        )

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the action space.

        Args:
            env_params: Optional environment parameters (uses XMiniGrid env_params if None).

        Returns:
            The action space corresponding to XMiniGrid action space.
        """
        xmg_env_params = env_params if env_params is not None else self._env_params
        num_actions = self._env.num_actions(xmg_env_params)

        return DiscreteSpace(
            num_values=num_actions,
            dtype=jnp.int32,
            name="action",
        )

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the state space.

        Args:
            env_params: Optional environment parameters (uses XMiniGrid env_params if None).

        Returns:
            A generic space for the XMiniGrid state.
        """
        raise NotImplementedError(
            "XMiniGrid does not expose a state space. Use observation_space instead."
        )

    def render(self, state: XMiniGridState, env_params: Optional[EnvParams] = None) -> Any:
        """Render the environment.

        Args:
            state: Current XMiniGrid state.
            env_params: Optional environment parameters (uses XMiniGrid env_params if None).

        Returns:
            Rendered environment (if supported).

        Raises:
            NotImplementedError: If rendering is not supported.
        """
        xmg_env_params = env_params if env_params is not None else self._env_params

        if hasattr(self._env, "render"):
            return self._env.render(state, xmg_env_params)
        else:
            raise NotImplementedError(f"Rendering not supported for {self._env.__class__.__name__}")
