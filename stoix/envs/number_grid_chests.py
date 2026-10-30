"""One-shot potion chests; Chebyshev pickup range from the Python reference.

The current scenario collects on a successful map move, per the user request.
See docs/ITEMS.md for sources and scenario choices.
"""
import jax.numpy as jnp

CHEST_LOOT = 21


class ChestRules:
    def __init__(self, game_map, potion_rules=None, item_rules=None):
        from stoix.envs.number_grid_potions import PotionRules
        rules = potion_rules if potion_rules is not None else PotionRules(game_map)
        chests = game_map['chests']
        if not isinstance(chests, list):
            raise ValueError('chests must be a list')
        positions, loot = [], []
        occupied = {tuple(p) for p in game_map['opponent_positions']}
        occupied.add(tuple(game_map['agent_position']))
        occupied.add(tuple(game_map.get('capital_position', game_map['agent_position'])))
        for chest in chests:
            if not isinstance(chest, dict):
                raise ValueError('Each chest must contain position and potions')
            position, contents = chest.get('position'), chest.get('potions',{})
            if (not isinstance(position, (list, tuple)) or len(position) != 2
                    or any(type(x) is not int or not 0 < x < game_map['size']-1 for x in position)
                    or tuple(position) in occupied):
                raise ValueError('Chest positions must be distinct, inside the map and unoccupied')
            occupied.add(tuple(position))
            positions.append(position)
            loot.append([contents.get(key,0) for key in rules.keys])
        self.count = len(chests)
        self.positions = jnp.asarray(positions, jnp.int32).reshape(-1, 2)
        self.loot = jnp.asarray(loot, jnp.int32).reshape(self.count, rules.count)
        self.features = jnp.concatenate((self.positions/(game_map['size']-1),
                                        self.loot.astype(jnp.float32)), axis=1)
        self.compact_observations = bool(game_map.get('compact_observations', False))
        self.observation_size = self.count if self.compact_observations else self.features.size+self.count
        self.items = item_rules

    def collect(self, state, moved):
        nearby = jnp.max(jnp.abs(self.positions-state.position), axis=1) <= 1
        collected = state.chest_alive & nearby & moved
        loot = jnp.sum(jnp.where(collected[:, None], self.loot, 0), axis=0)
        if self.items is not None:
            state=self.items.collect(state,collected)
        # last_loot is cleared per map step; a city captured on this move may already be in it.
        return state.replace(chest_alive=state.chest_alive & ~collected,
                             potions=state.potions+loot, last_loot=state.last_loot+loot,
                             last_event=jnp.where(jnp.any(collected), CHEST_LOOT, state.last_event))

    def observation(self, state):
        if self.compact_observations:
            return state.chest_alive.astype(jnp.float32)
        # Coordinates, scenario-specific item counts, then the live/collected flag.
        return jnp.concatenate((self.features, state.chest_alive[:, None]), axis=1).reshape(-1)
