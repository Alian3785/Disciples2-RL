"""Persistent exploration from the hero and capital. See docs/WORLD.md#fog."""

import jax.numpy as jnp
import numpy as np


def revealed_at(state, positions):
    """Look up tile visibility without treating clipped off-map coordinates as visible."""
    size = state.revealed.shape[0]
    inside = jnp.all((positions >= 0) & (positions < size), axis=-1)
    clipped = jnp.clip(positions, 0, size - 1)
    return inside & state.revealed[clipped[..., 0], clipped[..., 1]]


def visible_enemies(state):
    return state.alive & revealed_at(state, state.enemy_positions)


class FogRules:
    def __init__(self, game_map):
        size = int(game_map['size'])
        self.axis = jnp.arange(size, dtype=jnp.int32)
        interior = (self.axis > 0) & (self.axis < size - 1)
        self.playable = interior[:, None] & interior[None, :]
        self.observation_size = size * size
        capital = game_map.get('capital_position')
        footprint = game_map.get('capital_footprint')
        if capital is None and (footprint or game_map.get('territory') or game_map.get('capital_services')):
            capital = game_map['agent_position']
        self.capital = jnp.zeros((size, size), bool)
        if capital is not None:
            bounds = np.asarray(footprint or [capital], dtype=np.int32)
            self.capital = self.area(bounds.min(axis=0), bounds.max(axis=0), 6)

    def area(self, start, end, radius):
        """Expand a tile or footprint, omitting the four outer corner tiles."""
        row = jnp.maximum(start[0] - self.axis, self.axis - end[0])
        col = jnp.maximum(start[1] - self.axis, self.axis - end[1])
        return (self.playable & (row[:, None] <= radius) & (col[None, :] <= radius)
                & ~((row[:, None] == radius) & (col[None, :] == radius)))

    def reveal(self, state, radius=5):
        sight = self.area(state.position, state.position, radius)
        return state.replace(revealed=state.revealed | sight)

    def observation(self, state, terrain=0, blocked=False, owners=0):
        """One row-major value per tile: zero in fog, otherwise terrain, obstacle and owner."""
        codes = 1 + terrain + 4 * blocked + 8 * owners
        return jnp.where(state.revealed.reshape(-1), codes, 0).astype(jnp.float32) / 24.
