from dataclasses import field
from typing import TYPE_CHECKING, Dict, TypeAlias

import jax.numpy as jnp
from chex import ArrayTree
from jax import Array

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass

State: TypeAlias = ArrayTree
Reward: TypeAlias = Array
Discount: TypeAlias = Array
Observation: TypeAlias = ArrayTree
Action: TypeAlias = ArrayTree
TimeStepExtras: TypeAlias = Dict[str, ArrayTree]
EnvParams: TypeAlias = ArrayTree
ActionMask: TypeAlias = ArrayTree
StepCount: TypeAlias = Array


class StepType(jnp.int8):
    """Defines the status of a `TimeStep` within a sequence."""

    # Denotes the first `TimeStep` in a sequence.
    FIRST: Array = jnp.array(0, dtype=jnp.int8)
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID: Array = jnp.array(1, dtype=jnp.int8)
    # Denotes the last `TimeStep` in a sequence that was terminated.
    TERMINATED: Array = jnp.array(2, dtype=jnp.int8)
    # Denotes the last `TimeStep` in a sequence that was truncated.
    TRUNCATED: Array = jnp.array(3, dtype=jnp.int8)


@dataclass
class TimeStep:
    step_type: StepType
    reward: Reward
    discount: Discount
    observation: Observation
    extras: TimeStepExtras = field(default_factory=dict)

    def first(self) -> Array:
        return self.step_type == StepType.FIRST

    def mid(self) -> Array:
        return self.step_type == StepType.MID

    def last(self) -> Array:
        return jnp.logical_or(
            self.step_type == StepType.TERMINATED, self.step_type == StepType.TRUNCATED
        )

    def terminated(self) -> Array:
        return self.step_type == StepType.TERMINATED

    def truncated(self) -> Array:
        return self.step_type == StepType.TRUNCATED

    def done(self) -> Array:
        return self.last()
