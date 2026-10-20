import dataclasses
from typing import Any

from chex import PRNGKey

from stoa.stoa_struct import dataclass


def adapter_state_replace(self: "AdapterState", **changes: Any) -> "AdapterState":
    """Replace that can update attributes in nested adapter states."""
    # Separate changes into those for this level vs nested levels
    my_fields = {f.name for f in self.__dataclass_fields__.values()}
    local_changes = {k: v for k, v in changes.items() if k in my_fields}
    nested_changes = {k: v for k, v in changes.items() if k not in my_fields} or None

    if nested_changes is not None:
        # Only try to replace if base_env_state is a AdapterState
        if isinstance(self.base_env_state, AdapterState):
            # Recursively update the nested adapter state
            new_base_state = self.base_env_state.replace(**nested_changes)
            local_changes["base_env_state"] = new_base_state
        else:
            # We've reached a non-adapter state, can't go deeper
            raise AttributeError(
                f"Cannot update attributes {list(nested_changes.keys())} - "
                f"they are not accessible through adapter states"
            )

    # Use dataclass replace for local changes
    return dataclasses.replace(self, **local_changes)


@dataclass(custom_replace_fn=adapter_state_replace)
class AdapterState:
    """Base state for environment adapters."""

    base_env_state: Any

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the base environment state if not found."""
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.base_env_state, name)

    def __repr__(self) -> str:
        """Return a string representation of the adapter state."""
        return f"{self.__class__.__name__}(base_env_state={self.base_env_state!r})"

    @property
    def unwrapped(self) -> Any:
        """Get the deepest non-adapter state."""
        current = self.base_env_state
        while isinstance(current, AdapterState):
            current = current.base_env_state
        return current


@dataclass
class AdapterStateWithKey(AdapterState):
    """Adapter state that includes a JAX PRNG key."""

    rng_key: PRNGKey
