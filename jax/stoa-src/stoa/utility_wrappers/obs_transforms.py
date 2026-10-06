from typing import Any, Dict, Optional, Tuple, Type

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jax import Array

from stoa.core_wrappers.wrapper import Wrapper, WrapperState, wrapper_state_replace
from stoa.env_types import Action, EnvParams, Observation, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, BoundedArraySpace, DictSpace, DiscreteSpace, Space
from stoa.stoa_struct import dataclass


class AddStartFlagAndPrevAction(Wrapper[State]):
    """Wrapper that adds a start flag and the previous action to the observation.

    This wrapper modifies the observation to include:
    1. A start flag (1.0 for the first step, 0.0 for subsequent steps)
    2. The previous action (zero-initialized for the first step)
    3. The original observation

    The observation must be a flat array (1D). For discrete actions, the previous
    action is one-hot encoded before concatenation.
    """

    def __init__(self, env: Environment):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.

        Raises:
            ValueError: If the observation space is not a flat array.
            ValueError: If the action space is not supported.
        """
        super().__init__(env)

        # Check if the original observation is flat (1D array)
        orig_obs_space = self._env.observation_space()
        if not isinstance(orig_obs_space, (ArraySpace, BoundedArraySpace)):
            raise ValueError("Observation space must be an ArraySpace or BoundedArraySpace.")

        if len(orig_obs_space.shape) != 1:
            raise ValueError("The observation must be a flat (1D) array.")

        self.orig_obs_dim = orig_obs_space.shape[0]

        # Get action space information
        orig_action_space = self._env.action_space()
        if isinstance(orig_action_space, DiscreteSpace):
            self.action_dim = orig_action_space.num_values
            self._discrete = True
            self._process_action = lambda a: jax.nn.one_hot(a, self.action_dim, dtype=jnp.float32)
        elif isinstance(orig_action_space, (ArraySpace, BoundedArraySpace)):
            if len(orig_action_space.shape) != 1:
                raise ValueError("Only 1D continuous action spaces are supported.")

            self.action_dim = orig_action_space.shape[0]
            self._discrete = False
            self._process_action = lambda a: a.astype(jnp.float32)

        else:
            raise ValueError(f"Unsupported action space type: {type(orig_action_space)}")

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment and initialize the wrapper state.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (initial_state, initial_timestep) with modified observation.
        """
        state, timestep = self._env.reset(rng_key, env_params)

        # Initialize previous action as zeros
        # Always store as float32 array for consistency - discrete actions will be one-hot encoded
        prev_action = jnp.zeros(self.action_dim, dtype=jnp.float32)

        # Modify observation: [start_flag=1.0, prev_action_zeros, original_obs]
        start_flag = jnp.array([1.0], dtype=jnp.float32)

        # For the first step, previous action is always zeros (already in correct format)
        prev_action_encoded = prev_action

        new_observation = jnp.concatenate([start_flag, prev_action_encoded, timestep.observation])

        modified_timestep = timestep.replace(observation=new_observation)  # type: ignore

        return state, modified_timestep

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment and update the wrapper state.

        Args:
            state: Current wrapper state.
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, new_timestep) with modified observation.
        """
        # Step the base environment
        new_state, timestep = self._env.step(state, action, env_params)

        # Process the action to ensure it is in the correct format
        processed_action = self._process_action(action)

        # Modify observation: [start_flag=0.0, prev_action, original_obs]
        start_flag = jnp.array([0.0], dtype=jnp.float32)
        new_observation = jnp.concatenate([start_flag, processed_action, timestep.observation])

        modified_timestep = timestep.replace(observation=new_observation)  # type: ignore

        return new_state, modified_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the modified observation space.

        The new observation space has dimension:
        1 (start flag) + action_dim + original_observation_dim

        Args:
            env_params: Optional environment parameters.

        Returns:
            Modified observation space.
        """
        orig_obs_space = self._env.observation_space(env_params)
        new_obs_dim = 1 + self.action_dim + self.orig_obs_dim
        dtype = orig_obs_space.dtype
        # Create a new space with the modified shape
        return type(orig_obs_space)(shape=(new_obs_dim,), dtype=dtype, name=orig_obs_space.name)  # type: ignore


class MakeChannelLast(Wrapper[State]):
    """Simple wrapper for observations that have the channel dim first.
    This makes the channel dim last.

    This wrapper transforms observations from channel-first (e.g., CHW) to
    channel-last (e.g., HWC) format. It only modifies observations and does not
    change the environment state, so no state wrapping is needed.
    """

    def __init__(self, env: Environment) -> None:
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.

        Raises:
            AssertionError: If observation is not > 2 dimensional.
            ValueError: If observation space is not supported.
        """
        super().__init__(env)

        # Get observation space and validate it's multi-dimensional
        obs_space = self.observation_space()
        if not isinstance(obs_space, (ArraySpace, BoundedArraySpace)):
            raise ValueError("Observation space must be an ArraySpace or BoundedArraySpace.")

        obs_shape = jnp.array(obs_space.shape)
        assert len(obs_shape) > 2, "MakeChannelLast requires > 2 dimensional observations"

        # Calculate new shape with channel moved to last position
        # Roll the first axis (channel) to the last position
        self._new_obs_shape = tuple(jnp.roll(obs_shape, len(obs_shape) - 1))

    def _make_channel_last(self, observation: Observation) -> Observation:
        """Transform observation from channel-first to channel-last format.

        Args:
            observation: The observation to transform.

        Returns:
            Transformed observation with channel dimension last.
        """
        # Move the first axis (channel) to the last position
        return jnp.moveaxis(observation, 0, -1)

    def _transform_timestep_observations(self, timestep: TimeStep) -> TimeStep:
        """Transform all observations in a timestep.

        Args:
            timestep: The timestep with observations to transform.

        Returns:
            Timestep with transformed observations.
        """
        # Transform main observation
        new_observation = self._make_channel_last(timestep.observation)

        return timestep.replace(  # type: ignore
            observation=new_observation,
        )

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment and transform observations.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (state, timestep) with transformed observations.
            State is passed through unchanged.
        """
        state, timestep = self._env.reset(rng_key, env_params)
        transformed_timestep = self._transform_timestep_observations(timestep)
        return state, transformed_timestep

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment and transform observations.

        Args:
            state: Current environment state (passed through unchanged).
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, new_timestep) with transformed observations.
            State is passed through unchanged.
        """
        new_state, timestep = self._env.step(state, action, env_params)
        transformed_timestep = self._transform_timestep_observations(timestep)
        return new_state, transformed_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the transformed observation space.

        Args:
            env_params: Optional environment parameters.

        Returns:
            Observation space with channel dimension moved to last position.
        """
        orig_obs_space = self._env.observation_space(env_params)
        kwargs = orig_obs_space.__dict__
        kwargs["shape"] = self._new_obs_shape
        # Create a new space with the modified shape
        return type(orig_obs_space)(**kwargs)


@dataclass(custom_replace_fn=wrapper_state_replace)
class StepCountState(WrapperState):
    """State for tracking episode step count."""

    step_count: Array


class AddStepCountWrapper(Wrapper[StepCountState]):
    """
    Wrapper that adds step count to observations or timestep extras.

    Can either:
    1. Convert observation to dict with step_count (default)
    2. Add step_count to timestep.extras and keep original observation unchanged
    """

    def __init__(self, env: Environment, obs_key: str = "observation", in_extras: bool = False):
        """
        Initialize the step count wrapper.

        Args:
            env: The environment to wrap.
            obs_key: Key name for the original observation in the dict (only used if in_extras=False).
            in_extras: If True, add step_count to timestep.extras instead of modifying observation.
        """
        super().__init__(env)
        self._obs_key = obs_key
        self._in_extras = in_extras

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[StepCountState, TimeStep]:
        """Reset the environment and initialize step counter."""
        base_state, timestep = self._env.reset(rng_key, env_params)

        # Initialize wrapper state
        state = StepCountState(
            base_env_state=base_state,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

        if self._in_extras:
            # Add step count to extras, keep original observation
            new_extras = {**timestep.extras, "step_count": state.step_count}
            new_timestep = timestep.replace(extras=new_extras)  # type: ignore
        else:
            # Create dict observation
            dict_obs = {
                self._obs_key: timestep.observation,
                "step_count": state.step_count,
            }
            new_timestep = timestep.replace(observation=dict_obs)  # type: ignore

        return state, new_timestep

    def step(
        self,
        state: StepCountState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[StepCountState, TimeStep]:
        """Step the environment and update step counter."""
        new_base_state, timestep = self._env.step(state.base_env_state, action, env_params)

        # Update step count
        new_step_count = state.step_count + 1
        new_state = StepCountState(
            base_env_state=new_base_state,
            step_count=new_step_count,
        )

        if self._in_extras:
            # Add step count to extras, keep original observation
            new_extras = {**timestep.extras, "step_count": new_step_count}
            new_timestep = timestep.replace(extras=new_extras)  # type: ignore
        else:
            # Create dict observation
            dict_obs = {
                self._obs_key: timestep.observation,
                "step_count": new_step_count,
            }
            new_timestep = timestep.replace(observation=dict_obs)  # type: ignore

        return new_state, new_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space."""
        original_space = self._env.observation_space(env_params)

        if self._in_extras:
            # Step count is in extras, observation space unchanged
            return original_space
        else:
            # Step count is in observation dict
            spaces = {
                self._obs_key: original_space,
                "step_count": ArraySpace(shape=(), dtype=jnp.int32, name="step_count"),
            }
            return DictSpace(spaces=spaces)


class AddActionMaskWrapper(Wrapper[State]):
    """
    Wrapper that adds action mask to observations as a dictionary.

    Converts the observation to a dict with:
    - "observation": original observation
    - "action_mask": legal action mask
    """

    def __init__(
        self, env: Environment, action_mask_key: str = "action_mask", obs_key: str = "observation"
    ):
        """
        Initialize the action mask wrapper.

        Args:
            env: The environment to wrap.
            action_mask_key: Key for the action mask found in the extras of the timestep.
            obs_key: Key name for the original observation in the new observation dict.
        """
        super().__init__(env)
        self._obs_key = obs_key
        self._action_mask_key = action_mask_key

        # Get action space info for mask shape
        action_space = self.action_space()
        if hasattr(action_space, "num_values"):
            self._mask_shape = (action_space.num_values,)
        else:
            raise ValueError(
                f"Unsupported action space type for action mask: {type(action_space)}. "
                "Action space must have a 'num_values' attribute."
            )

    def _get_action_mask(self, timestep: TimeStep) -> Array:
        """Extract action mask from timestep extras or create default."""
        return timestep.extras.get(
            self._action_mask_key, jnp.ones(self._mask_shape, dtype=jnp.bool_)
        )

    def _create_dict_obs(self, timestep: TimeStep) -> Dict[str, Any]:
        """Create dictionary observation with action mask."""
        action_mask = self._get_action_mask(timestep)
        return {
            self._obs_key: timestep.observation,
            "action_mask": action_mask,
        }

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment and add action mask."""
        state, timestep = self._env.reset(rng_key, env_params)
        dict_obs = self._create_dict_obs(timestep)
        new_timestep = timestep.replace(observation=dict_obs)  # type: ignore
        return state, new_timestep

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment and add action mask."""
        new_state, timestep = self._env.step(state, action, env_params)
        dict_obs = self._create_dict_obs(timestep)
        new_timestep = timestep.replace(observation=dict_obs)  # type: ignore
        return new_state, new_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the dict observation space."""
        original_space = self._env.observation_space(env_params)
        spaces = {
            self._obs_key: original_space,
            "action_mask": ArraySpace(shape=self._mask_shape, dtype=jnp.bool_, name="action_mask"),
        }
        return DictSpace(spaces=spaces)


class ObservationTypeWrapper(Wrapper[State]):
    """
    Wrapper that converts dict observations to a user-specified type.

    Takes a dict observation and converts it to any type that can be constructed
    from keyword arguments (NamedTuple, dataclass, etc.).
    """

    def __init__(self, env: Environment, observation_type: Type):
        """
        Initialize the observation type wrapper.

        Args:
            env: Environment with dict observations to wrap.
            observation_type: Type to convert observations to (e.g., NamedTuple, dataclass).
                              Must be constructible from keyword arguments.
        """
        super().__init__(env)
        self._observation_type = observation_type

        # Validate that the environment has dict observations
        obs_space = self._env.observation_space()
        if not isinstance(obs_space, DictSpace):
            raise ValueError(
                f"ObservationTypeWrapper requires DictSpace observations, " f"got {type(obs_space)}"
            )

        # Store the original dict space for reference
        self._original_dict_space = obs_space

    def _convert_observation(self, dict_obs: Dict[str, Any]) -> Any:
        """Convert dict observation to the target type."""
        return self._observation_type(**dict_obs)

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment and convert observation type."""
        state, timestep = self._env.reset(rng_key, env_params)
        typed_obs = self._convert_observation(timestep.observation)
        new_timestep = timestep.replace(observation=typed_obs)  # type: ignore
        return state, new_timestep

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment and convert observation type."""
        new_state, timestep = self._env.step(state, action, env_params)
        typed_obs = self._convert_observation(timestep.observation)
        new_timestep = timestep.replace(observation=typed_obs)  # type: ignore
        return new_state, new_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """
        Return the underlying dict observation space.

        Note: The actual observations returned by this environment are of type
        `self._observation_type`, not dictionaries. However, the space describes
        the structure and bounds of the data that gets converted to that type.

        This maintains compatibility with the Stoa space system while allowing
        type conversion. Users can introspect the actual observation structure
        through this space.
        """
        return self._original_dict_space
