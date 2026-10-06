from typing import Any, Optional, Tuple

from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import Space


class ObservationExtractWrapper(Wrapper[State]):
    """
    Extracts a specific attribute from dict or named tuple observations.

    This wrapper is useful when the base environment returns observations as
    dictionaries or named tuples, but you only care about a specific field
    for your agent's observation.

    Example:
        # If env returns obs = {"pixels": array, "state": array, "info": dict}
        # and you only want the "pixels" for your agent:
        wrapped_env = ObservationExtractWrapper(env, observation_attribute="pixels")
    """

    def __init__(self, env: Environment, observation_attribute: str):
        """
        Initialize the observation extraction wrapper.

        Args:
            env: The environment to wrap.
            observation_attribute: The key/attribute name to extract from the observation.
                For dicts, this is the key name. For named tuples, this is the field name.
        """
        super().__init__(env)
        self._observation_attribute = observation_attribute

    def _extract_observation(self, full_observation: Any) -> Tuple[Any, dict]:
        """
        Extract the desired attribute from the full observation.

        Args:
            full_observation: The complete observation from the base environment.

        Returns:
            The extracted observation attribute.

        Raises:
            ValueError: If the observation doesn't have the specified attribute.
        """
        if isinstance(full_observation, dict):
            if self._observation_attribute not in full_observation:
                available_keys = list(full_observation.keys())
                raise ValueError(
                    f"Observation attribute '{self._observation_attribute}' not found in "
                    f"observation dict. Available keys: {available_keys}"
                )
            # Extract from dict
            obs = full_observation[self._observation_attribute]
            # Return the rest of the dict as extras
            extras = {k: v for k, v in full_observation.items() if k != self._observation_attribute}
            return obs, extras

        elif hasattr(full_observation, "_asdict"):
            # Named tuple
            obs_dict = full_observation._asdict()
            if self._observation_attribute not in obs_dict:
                available_fields = list(obs_dict.keys())
                raise ValueError(
                    f"Observation attribute '{self._observation_attribute}' not found in "
                    f"named tuple observation. Available fields: {available_fields}"
                )
            # Extract from named tuple
            obs = obs_dict[self._observation_attribute]
            # Return the rest of the named tuple as extras
            extras = {k: v for k, v in obs_dict.items() if k != self._observation_attribute}
            return obs, extras

        elif hasattr(full_observation, self._observation_attribute):
            # Regular object with attribute
            obs = getattr(full_observation, self._observation_attribute)
            # Return the rest of the object as extras
            extras = {
                k: v
                for k, v in full_observation.__dict__.items()
                if k != self._observation_attribute
            }
            return obs, extras

        else:
            raise ValueError(
                f"Cannot extract attribute '{self._observation_attribute}' from observation "
                f"of type {type(full_observation)}. Observation must be a dict, named tuple, "
                f"or object with the specified attribute."
            )

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """
        Reset the environment and extract the desired observation attribute.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (state, timestep) where timestep.observation contains only
            the extracted attribute.
        """
        state, timestep = self._env.reset(rng_key, env_params)

        # Extract the desired observation attribute
        extracted_obs, obs_extras = self._extract_observation(timestep.observation)

        # Update timestep extras with the obs extras
        new_extras = {**obs_extras, **timestep.extras}

        # Create new timestep with extracted observation and updated extras
        new_timestep = timestep.replace(observation=extracted_obs, extras=new_extras)  # type: ignore

        return state, new_timestep

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """
        Step the environment and extract the desired observation attribute.

        Args:
            state: Current environment state.
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, timestep) where timestep.observation contains only
            the extracted attribute.
        """
        new_state, timestep = self._env.step(state, action, env_params)

        # Extract the desired observation attribute
        extracted_obs, obs_extras = self._extract_observation(timestep.observation)
        # Update timestep extras with the obs extras
        new_extras = {**obs_extras, **timestep.extras}

        # Create new timestep with extracted observation and updated extras
        new_timestep = timestep.replace(observation=extracted_obs, extras=new_extras)  # type: ignore

        return new_state, new_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """
        Get the observation space for the extracted attribute.

        Args:
            env_params: Optional environment parameters.

        Returns:
            The space corresponding to the extracted observation attribute.

        Raises:
            ValueError: If the base environment's observation space doesn't have
                       the specified attribute.
        """
        base_obs_space = self._env.observation_space(env_params)

        # Handle different space types
        if hasattr(base_obs_space, "spaces") and isinstance(base_obs_space.spaces, dict):
            # DictSpace
            if self._observation_attribute not in base_obs_space.spaces:
                available_keys = list(base_obs_space.spaces.keys())
                raise ValueError(
                    f"Observation attribute '{self._observation_attribute}' not found in "
                    f"observation space. Available keys: {available_keys}"
                )
            new_obs_space: Space = base_obs_space.spaces[self._observation_attribute]
            return new_obs_space

        else:
            raise ValueError(
                f"Cannot extract observation attribute from space of type "
                f"{type(base_obs_space)}. Expected DictSpace or TupleSpace."
            )
