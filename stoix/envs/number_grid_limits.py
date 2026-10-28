"""Hard limits requested for the JAX game; see docs/COMBAT.md."""
import jax.numpy as jnp

MAX_LEVEL = 99
MAX_HP = 9999
MAX_INITIATIVE = 150
MAX_ACCURACY = 100
MAX_ARMOR = 90
MAX_RESOURCE = 9999


def cap_stats(values):
    """Cap final HP, accuracy, armor and initiative; damage has its own rules."""
    # Columns match number_grid_combat.STAT_NAMES; no import cycle with profiles.
    return jnp.minimum(values, jnp.array(
        [MAX_HP, jnp.inf, MAX_ACCURACY, MAX_ARMOR, MAX_INITIATIVE], values.dtype))


def add_resource(balance, gain):
    """Credit nonnegative income, discarding overflow before any subsequent spend."""
    balance = jnp.clip(balance, 0, MAX_RESOURCE)
    # Limit the increment before addition to avoid overflowing int32 balances.
    return balance + jnp.clip(gain, 0, MAX_RESOURCE-balance)
