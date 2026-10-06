from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from mujoco import mjx
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv
from mujoco_playground import State as MjxState

from stoa.env_adapters.base import AdapterState, adapter_state_replace
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, DictSpace, Space
from stoa.stoa_struct import dataclass


@dataclass(custom_replace_fn=adapter_state_replace)
class PlaygroundState(AdapterState):
    """State for the MuJoCo Playground environments
    so domain randomization is natively supported."""

    model: MjxModel


class MuJoCoPlaygroundToStoa(Environment):
    """MuJoCo Playground environments in Stoa interface."""

    def __init__(
        self,
        env: MjxEnv,
        domain_randomizer_fn: Optional[
            Callable[[mjx.Model, PRNGKey], Tuple[mjx.Model, Any]]
        ] = None,
    ):
        """Initialize the MuJoCo Playground Stoa adapter.

        Args:
            env: The MuJoCo Playground environment to wrap.
        """
        self._env = env
        self._domain_randomizer_fn = domain_randomizer_fn

        # Cache action and observation dimensions
        self._action_size = env.action_size
        self._obs_size = env.observation_size

    def _env_fn(self, mjx_model: mjx.Model) -> MjxEnv:
        env = self._env
        env.unwrapped._mjx_model = mjx_model
        return env

    def _mjx_to_timestep(self, mjx_state: MjxState) -> TimeStep:
        """Converts an MjxState to a Stoa TimeStep."""
        # Terminated and truncate respond to done
        done = mjx_state.done
        # Assume "truncated" is in info if truncation is possible.
        truncated = mjx_state.info.get("truncated", jnp.array(False))

        # The episode has truly ended if the state is "done" but not "truncated".
        is_terminated = jnp.logical_and(done, jnp.logical_not(truncated))

        step_type = jax.lax.select(
            is_terminated,
            StepType.TERMINATED,
            jax.lax.select(truncated, StepType.TRUNCATED, StepType.MID),
        )

        # Discount is 0 only on termination, not truncation.
        discount = jnp.where(is_terminated, 0.0, 1.0)

        return TimeStep(
            step_type=step_type,
            reward=jnp.asarray(mjx_state.reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            observation=mjx_state.obs,
            extras={"metrics": mjx_state.metrics, **mjx_state.info},
        )

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[PlaygroundState, TimeStep]:
        """Reset the environment."""
        if self._domain_randomizer_fn is not None:
            dr_key, reset_key = jax.random.split(rng_key)

            # Get base model from the unwrapped environment.
            base_model = self._env.unwrapped.mjx_model

            # Apply domain randomization to get a new model for this episode.
            rand_keys_batch = jax.random.split(dr_key, 1)
            randomized_batched_model, in_axes = self._domain_randomizer_fn(
                base_model, rand_keys_batch
            )

            def _unbatch(leaf: Any, axis: Optional[int]) -> Any:
                return leaf[0] if axis == 0 else leaf

            # Unbatch the randomized model to match the expected structure.
            mjx_model = jax.tree_util.tree_map(_unbatch, randomized_batched_model, in_axes)
            # Create a temporary environment with the randomized model.
            temp_env = self._env_fn(mjx_model=mjx_model)
            # Reset the temporary environment.
            mjx_state = temp_env.reset(reset_key)
        else:
            # If no domain randomization, use the original environment.
            mjx_state = self._env.reset(rng_key)
            mjx_model = self._env.unwrapped.mjx_model

        # Create the initial timestep
        timestep = self._mjx_to_timestep(mjx_state)
        # Wrap the state in our custom PlaygroundState
        state = PlaygroundState(base_env_state=mjx_state, model=mjx_model)
        return state, timestep

    def step(
        self,
        state: PlaygroundState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[Any, TimeStep]:
        """Step the environment."""
        if self._domain_randomizer_fn is not None:
            # If domain randomization is used, we need to ensure the model is correct.
            mjx_model = state.model
            temp_env = self._env_fn(mjx_model=mjx_model)
            next_mjx_state = temp_env.step(state.base_env_state, action)
        else:
            # If no domain randomization, use the original environment.
            next_mjx_state = self._env.step(state.base_env_state, action)

        # Convert the next MjxState to a Stoa TimeStep
        timestep = self._mjx_to_timestep(next_mjx_state)
        next_state = state.replace(
            base_env_state=next_mjx_state,
        )
        return next_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space."""
        if isinstance(self._obs_size, dict):
            return DictSpace(
                spaces={
                    key: BoundedArraySpace(
                        shape=(shape,) if isinstance(shape, int) else shape,
                        dtype=jnp.float32,
                        minimum=-jnp.inf,
                        maximum=jnp.inf,
                    )
                    for key, shape in self._obs_size.items()
                }
            )
        else:
            return BoundedArraySpace(
                shape=(self._obs_size,) if isinstance(self._obs_size, int) else self._obs_size,
                dtype=jnp.float32,
                minimum=-jnp.inf,
                maximum=jnp.inf,
                name="observation",
            )

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the action space."""
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
            "MuJoCo Playground does not expose a state space. Use observation_space instead."
        )

    def render(self, state: Any, env_params: Optional[EnvParams] = None) -> Any:
        """Render the environment."""
        if hasattr(self._env, "render"):
            return self._env.render(state)
        else:
            raise NotImplementedError(f"Rendering not supported for {self._env.__class__.__name__}")
