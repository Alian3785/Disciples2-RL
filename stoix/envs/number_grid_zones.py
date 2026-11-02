"""Original zone of control: a strategic step beside an enemy stack stops there and attacks it.

Static analysis of the installed Steam Discipl2.exe (2026-11-02); the original game was not
run. Rules, addresses and the remaining simplifications: docs/COMBAT.md#zones.
"""
import jax.numpy as jnp

# Discipl2.exe 0x7b2990: (dx, dy) neighbours from the top-left; JAX cells are [y+1, x+1].
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
NEUTRAL = 5  # LSubRace.dbf L_NEUTRAL: equal plain-neutral leaders still hold zones.


def kin_stacks(progression):
    """Stacks whose leader surely shares the hero leader's non-neutral subrace.

    Discipl2.exe 0x5df50d ignores such a neighbour unless it is the clicked tile. Maps keep
    no stack leader, so only a roster made entirely of that subrace (Gunits SUBRACE) counts.
    """
    rows = progression.rows
    subrace = [int(r.get('game_data', {}).get('SUBRACE') or 0) for r in rows]
    leader = next((i for i in progression.initial_ids[:6].tolist() if rows[i].get('hero')), 0)
    squads = [[i for i in squad if i] for squad in progression.enemy_ids.tolist()]
    own = subrace[leader]
    return [own not in (0, NEUTRAL) and bool(squad) and all(subrace[i] == own for i in squad)
            for squad in squads]


class ZoneRules:
    def __init__(self, game_map, progression=None, territory=None, ruins=None, homes=()):
        """Settlement tiles and the stacks that never hold a zone; homes are bot capitals."""
        count = len(game_map['opponent_positions'])
        forts, inside = [], [list(p) for p in homes]
        if territory is not None:
            forts = [list(territory.capital)]+[list(c['position']) for c in territory.cities]
            inside += forts+territory.enemy_capital_positions.tolist()
        if ruins is not None:
            inside += ruins.entrances.tolist()
        # Entering a fort resolves the fort, not neighbouring stacks (Discipl2.exe 0x5cde3e).
        self.forts = jnp.array(forts, jnp.int32).reshape(-1, 2)
        # Garrisons, capital guards and ruin keepers are inside an object, not map stacks.
        self.inside = jnp.array(inside, jnp.int32).reshape(-1, 2)
        kin = kin_stacks(progression) if progression is not None else [False]*count
        if len(kin) != count:
            raise ValueError('Zone rules need one enemy roster per army')
        self.kin = jnp.array(kin, bool).reshape(count)
        self.neighbours = jnp.array(NEIGHBOURS, jnp.int32)
        self.metadata = dict(neighbours=NEIGHBOURS, kin_armies=[i for i, k in enumerate(kin) if k])

    @staticmethod
    def _at(cells, tiles):
        return jnp.any(jnp.all(cells[..., None, :] == tiles, axis=-1), axis=-1)

    def holders(self, state):
        """Living enemy stacks on open ground that project a zone."""
        return state.alive & ~self.kin & ~self._at(state.enemy_positions, self.inside)

    def _beside(self, state, cell, holders):
        return holders & (jnp.max(jnp.abs(state.enemy_positions-cell), axis=-1) == 1)

    def _first(self, state, cell, holders):
        """The first holder around cell in the original neighbour order, or -1."""
        tiles = cell+self.neighbours
        present = jnp.all(tiles[:, None, :] == state.enemy_positions[None], axis=-1) & holders
        count = holders.shape[0]
        index = jnp.min(jnp.where(present, jnp.arange(count), count), axis=1, initial=count)
        found = index < count
        return jnp.where(jnp.any(found), index[jnp.argmax(found)], -1).astype(jnp.int32)

    def resolve(self, state, destination, moving):
        """One-tile move under Discipl2.exe 0x5ce51b (step) and 0x5cde3e (move result).

        Returns whether the hero enters destination and the stack it then fights (-1: none).
        A step beside a holder stops there and attacks the first holder around the new tile.
        A hero who started beside a holder that is also beside destination does not move and
        attacks the first holder around its own tile; stepping away from it is free.
        """
        holders = self.holders(state)
        checked = moving & ~self._at(destination, self.forts)
        started = self._beside(state, state.position, holders) & ~self._at(state.position, self.forts)
        held = checked & jnp.any(started & self._beside(state, destination, holders))
        enemy = jnp.where(held, self._first(state, state.position, holders),
                          jnp.where(checked, self._first(state, destination, holders), -1))
        return ~held, enemy
