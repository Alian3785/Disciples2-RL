import abc
from typing import Any, Optional, Tuple

import jax.numpy as jnp
from chex import PRNGKey, dataclass

from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.spaces import BoundedArraySpace, EnvironmentSpace, Space


@dataclass
class Environment:
    """Abstract base class for JAX-native RL environments."""

    @abc.abstractmethod
    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Resets the environment to an initial state."""
        pass

    @abc.abstractmethod
    def step(
        self, state: State, action: Action, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Updates the environment according to the agent's action."""
        pass

    def reward_space(self, env_params: Optional[EnvParams] = None) -> BoundedArraySpace:
        """Describes the reward returned by the environment."""
        return BoundedArraySpace(
            shape=(), dtype=jnp.float32, minimum=-jnp.inf, maximum=jnp.inf, name="reward"
        )

    def discount_space(self, env_params: Optional[EnvParams] = None) -> BoundedArraySpace:
        """Describes the discount returned by the environment."""
        return BoundedArraySpace(
            shape=(), dtype=jnp.float32, minimum=0.0, maximum=1.0, name="discount"
        )

    @abc.abstractmethod
    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Defines the structure and bounds of the observation space."""
        pass

    @abc.abstractmethod
    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Defines the structure and bounds of the action space."""
        pass

    @abc.abstractmethod
    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Defines the structure and bounds of the state space."""
        pass

    def environment_space(self, env_params: Optional[EnvParams] = None) -> EnvironmentSpace:
        """Defines the structure and bounds of the environment space."""
        return EnvironmentSpace(
            observations=self.observation_space(env_params),
            actions=self.action_space(env_params),
            rewards=self.reward_space(env_params),
            discounts=self.discount_space(env_params),
            state=self.state_space(env_params),
        )

    @property
    def unwrapped(self) -> "Environment":
        return self

    def render(self, state: State, env_params: Optional[EnvParams] = None) -> Any:
        """Render environment"""
        raise NotImplementedError("Render method not implemented for this environment.")

    def close(self) -> None:
        """Frees any resources used by the environment."""
        pass
