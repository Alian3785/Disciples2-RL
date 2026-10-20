import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Dict, TypeVar, overload

import jax
from flax.struct import field, serialization
from typing_extensions import dataclass_transform

_T = TypeVar("_T")

# This dataclass is based on the flax.struct dataclass, which is compatible with JAX transformations.
# The reason for creating a new custom dataclass is to allow for overriding the replace method
# to handle nested wrapper states correctly, which is not supported by the default flax.struct dataclass.
# custom_replace_fn should be a method with the signature: def(self, **updates) -> Self


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
@overload
def dataclass(clz: _T, **kwargs: Any) -> _T:
    ...


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
@overload
def dataclass(**kwargs: Any) -> Callable[[_T], _T]:
    ...


@dataclass_transform(field_specifiers=(field,))
def dataclass(
    clz: _T | None = None,
    **kwargs: Any,
) -> _T | Callable[[_T], _T]:

    # Support passing arguments to the decorator (e.g. @dataclass(kw_only=True))
    if clz is None:
        return functools.partial(dataclass, **kwargs)  # type: ignore[bad-return-type]

    # check if already a flax dataclass
    if "_flax_dataclass" in clz.__dict__:
        return clz

    if "frozen" not in kwargs.keys():
        kwargs["frozen"] = True
    custom_replace_fn = kwargs.pop("custom_replace_fn", None)
    data_clz = dataclasses.dataclass(**kwargs)(clz)  # type: ignore
    meta_fields = []
    data_fields = []
    for field_info in dataclasses.fields(data_clz):
        is_pytree_node = field_info.metadata.get("pytree_node", True)
        if is_pytree_node:
            data_fields.append(field_info.name)
        else:
            meta_fields.append(field_info.name)

    def default_replace(self: _T, **updates: Any) -> _T:
        """Returns a new object replacing the specified fields with new values."""
        return dataclasses.replace(self, **updates)  # type: ignore[type-var]

    if custom_replace_fn is not None:
        # custom_replace_fn should be a method with the signature: def(self, **updates) -> Self
        replace = custom_replace_fn
    else:
        replace = default_replace

    data_clz.replace = replace

    jax.tree_util.register_dataclass(data_clz, data_fields, meta_fields)

    def to_state_dict(x: _T) -> Dict[str, Any]:
        state_dict = {name: serialization.to_state_dict(getattr(x, name)) for name in data_fields}
        return state_dict

    def from_state_dict(x: _T, state: Dict[str, Any]) -> _T:
        """Restore the state of a data class."""
        state = state.copy()  # copy the state so we can pop the restored fields.
        updates = {}
        for name in data_fields:
            if name not in state:
                raise ValueError(
                    f"Missing field {name} in state dict while restoring"  # type: ignore[attr-defined, union-attr]
                    f" an instance of {clz.__name__},"
                    f" at path {serialization.current_path()}"
                )
            value = getattr(x, name)
            value_state = state.pop(name)
            updates[name] = serialization.from_state_dict(value, value_state, name=name)
        if state:
            names = ",".join(state.keys())
            raise ValueError(
                f'Unknown field(s) "{names}" in state dict while'  # type: ignore[attr-defined, union-attr]
                f" restoring an instance of {clz.__name__}"
                f" at path {serialization.current_path()}"
            )
        return x.replace(**updates)  # type: ignore[no-any-return, attr-defined]

    serialization.register_serialization_state(data_clz, to_state_dict, from_state_dict)

    # add a _flax_dataclass flag to distinguish from regular dataclasses
    data_clz._flax_dataclass = True  # type: ignore[attr-defined]
    # also add a _stoa_dataclass flag to distinguish from other dataclasses
    data_clz._stoa_dataclass = True  # type: ignore[attr-defined]

    return data_clz  # type: ignore
