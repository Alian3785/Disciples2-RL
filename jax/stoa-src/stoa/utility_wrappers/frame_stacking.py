from typing import Optional, Tuple

import jax.numpy as jnp
from chex import Array, PRNGKey, Shape

from stoa.core_wrappers.wrapper import Wrapper, WrapperState, wrapper_state_replace
from stoa.env_types import Action, EnvParams, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, BoundedArraySpace, Space
from stoa.stoa_struct import dataclass


@dataclass(custom_replace_fn=wrapper_state_replace)
class FrameStackState(WrapperState):
    """State for tracking stacked frames."""

    stacked_frames: Array


class FrameStacker:
    """Helper class for managing frame stacking logic."""

    def __init__(self, num_frames: int, frame_shape: Shape, flatten: bool = False):
        """Initialize the frame stacker.

        Args:
            num_frames: Number of frames to stack.
            frame_shape: Shape of individual frames.
            flatten: Whether to flatten the stacking dimension with the last dimension.
        """
        self._num_frames = num_frames
        self._flatten = flatten
        self._frame_shape = tuple(frame_shape)

    def reset(self) -> Array:
        """Create initial stacked frames (all zeros).

        Returns:
            Array of stacked frames initialized to zeros.
        """
        stacked_frames = jnp.zeros((*self._frame_shape, self._num_frames))
        return stacked_frames

    def step(self, stacked_frames: Array, new_frame: Array) -> Array:
        """Add a new frame to the stack.

        Args:
            stacked_frames: Current stacked frames.
            new_frame: New frame to add to the stack.

        Returns:
            Updated stacked frames with the new frame added.
        """
        # Shift frames and add new frame to the end
        stacked_frames = jnp.roll(stacked_frames, shift=-1, axis=-1)
        stacked_frames = stacked_frames.at[..., -1].set(new_frame)
        return stacked_frames

    def get_observation(self, stacked_frames: Array) -> Array:
        """Convert stacked frames to observation format.

        Args:
            stacked_frames: The stacked frames.

        Returns:
            Observation in the correct format (flattened or not).
        """
        if not self._flatten:
            return stacked_frames
        else:
            # Flatten the last two dimensions (channels and frames)
            new_shape = stacked_frames.shape[:-2] + (-1,)
            return stacked_frames.reshape(new_shape)


class FrameStackingWrapper(Wrapper[FrameStackState]):
    """Wrapper that stacks observations along a new final axis.

    This wrapper maintains a rolling buffer of the last N observations,
    which is useful for environments where temporal context is important
    (e.g., when dealing with velocity or partial observability).
    """

    def __init__(self, env: Environment, num_frames: int = 4, flatten: bool = True):
        """Initialize the frame stacking wrapper.

        Args:
            env: Environment to wrap.
            num_frames: Number of frames to stack.
            flatten: Whether to flatten the channel and stacking dimensions together.
                    e.g. (H, W, C, num_frames) -> (H, W, C * num_frames)
        """
        super().__init__(env)
        self._num_frames = num_frames
        self._flatten = flatten

        # Get observation space to determine frame shape
        obs_space = self._env.observation_space()
        self._frame_shape = obs_space.shape
        self._stacker = FrameStacker(
            num_frames=num_frames, frame_shape=self._frame_shape, flatten=flatten
        )

    def _update_space(self, space: Space) -> Space:
        """Update a space to account for frame stacking.

        Args:
            space: The original space.

        Returns:
            Updated space with stacking dimension.
        """
        if space.shape is None:
            raise ValueError(f"Cannot stack frames for space without shape: {space}")

        # Calculate new shape
        if not self._flatten:
            new_shape = space.shape + (self._num_frames,)
        else:
            # Flatten last dimension with stacking dimension
            new_shape = space.shape[:-1] + (self._num_frames * space.shape[-1],)

        if isinstance(space, BoundedArraySpace):
            # Handle bounded spaces
            if space.minimum.shape != ():
                # Repeat bounds for each frame
                new_minimum = jnp.repeat(space.minimum, self._num_frames, axis=-1)
                new_maximum = jnp.repeat(space.maximum, self._num_frames, axis=-1)
                if self._flatten:
                    new_minimum = new_minimum.reshape(new_shape)
                    new_maximum = new_maximum.reshape(new_shape)
            else:
                # Scalar bounds remain the same
                new_minimum = space.minimum
                new_maximum = space.maximum

            return BoundedArraySpace(
                shape=new_shape,
                dtype=space.dtype,
                minimum=new_minimum,
                maximum=new_maximum,
                name=space.name,
            )

        elif isinstance(space, ArraySpace):
            return ArraySpace(
                shape=new_shape,
                dtype=space.dtype,
                name=space.name,
            )

        else:
            raise ValueError(f"Unsupported space type for frame stacking: {type(space)}")

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[FrameStackState, TimeStep]:
        """Reset the environment and initialize frame stacking.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (initial_state, initial_timestep) with stacked observations.
        """
        # Reset base environment
        base_state, timestep = self._env.reset(rng_key, env_params)

        # Initialize stacked frames
        stacked_frames = self._stacker.reset()

        # Add the first observation to the stack
        stacked_frames = self._stacker.step(stacked_frames, timestep.observation)

        # Create state with stacked frames
        state = FrameStackState(
            base_env_state=base_state,
            stacked_frames=stacked_frames,
        )

        # Update timestep with stacked observation
        stacked_observation = self._stacker.get_observation(stacked_frames)
        new_timestep = timestep.replace(observation=stacked_observation)  # type: ignore

        return state, new_timestep

    def step(
        self,
        state: FrameStackState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[FrameStackState, TimeStep]:
        """Step the environment and update frame stacking.

        Args:
            state: Current state including stacked frames.
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, new_timestep) with updated stacked observations.
        """
        # Step base environment
        new_base_state, timestep = self._env.step(state.base_env_state, action, env_params)

        # Update stacked frames with new observation
        new_stacked_frames = self._stacker.step(state.stacked_frames, timestep.observation)

        # Create new state
        new_state = FrameStackState(
            base_env_state=new_base_state,
            stacked_frames=new_stacked_frames,
        )

        # Update timestep with stacked observation
        stacked_observation = self._stacker.get_observation(new_stacked_frames)
        new_timestep = timestep.replace(observation=stacked_observation)  # type: ignore

        return new_state, new_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space with frame stacking applied.

        Args:
            env_params: Optional environment parameters.

        Returns:
            Updated observation space that accounts for frame stacking.
        """
        original_space = self._env.observation_space(env_params)
        return self._update_space(original_space)
