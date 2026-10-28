"""Unit swaps: the player rearranges the hero's party on the map. See docs/UNITS.md.

Python reference campaign_env_economy.py: _build_blue_unit_swap_pairs,
_can_swap_blue_unit_positions/_blue_unit_swap_action_mask_values,
_step_swap_blue_unit_positions, _step_swap_big_blue_unit_positions,
_unit_swap_reward; MAX_GRID_UNIT_SWAPS_PER_TURN (campaign_env.py:54),
reward_unit_swap_penalty (campaign_env.py:164). Slots 0-2 are the front row
(reference positions 7-9), slots 3-5 the back row (10-12); slot s and s+3
form a column. A big unit stands in the front slot and covers its column.
"""
import jax.numpy as jnp
import numpy as np

MAX_PER_TURN = 3
PENALTY = .004
UNITS_SWAPPED = 55
# Every state field whose leading axis is the twelve battle slots; the hero's six come first.
SLOT_FIELDS = (
    'unit_ids', 'unit_levels', 'unit_xp', 'hp', 'native_ids', 'native_levels', 'native_xp', 'last_xp', 'mercenary_units',
    'effect_rounds', 'disable_source', 'transform_long', 'powerup_long', 'saved_powerup_long',
    'copied', 'copy_stats', 'copy_secondary', 'copy_traits', 'summon_owner', 'summon_saved_hp', 'preparation',
    'initiative_override', 'saved_initiative_override', 'slow_original', 'waited', 'bonus_turns',
    'battle_revived', 'battle_earned_xp', 'primary_override', 'preweak_damage', 'powerup', 'powerup_layered',
    'saved_primary_override', 'saved_preweak_damage', 'saved_powerup', 'saved_powerup_layered',
    'healer_wards', 'ward_native_used', 'saved_healer_wards', 'saved_ward_native_used', 'pending_healers',
    'fenrir', 'armor_shreds', 'imp', 'imp_wards', 'decay_form', 'decay_shreds', 'feared',
    'weakened', 'saved_weakened', 'item_weakened', 'saved_item_weakened', 'lycanthrope',
    'paralyzed', 'long_paralyzed', 'poison_mode', 'poison_damage', 'poison_source', 'last_poison_damage',
    'water_mode', 'water_damage', 'water_source', 'last_water_damage',
    'burn_mode', 'burn_damage', 'burn_source', 'last_burn_damage', 'activation_done',
    'potion_doses', 'potion_bonus', 'potion_temporary', 'potion_active', 'potion_wards',
    'spell_mods', 'spell_wards', 'equipment_bonus', 'wards_used', 'priority', 'turn_phase',
    'defended', 'retreating', 'escaped')


def enabled(game_map):
    value = game_map.get('unit_swaps', False)
    if not isinstance(value, bool):
        raise ValueError('unit_swaps must be true or false')
    return value


def swap_pairs(roster):
    """Reference layout: pairs of the first four occupied starting slots, then each
    remaining slot paired with every slot already included; 15 pairs over 6 slots."""
    occupied = [slot for slot, key in enumerate(roster[:6]) if key is not None][:4]
    pairs = [(a, b) for i, a in enumerate(occupied) for b in occupied[i+1:]]
    included = list(occupied)
    for extra in range(6):
        if extra not in included:
            pairs += [(min(other, extra), max(other, extra)) for other in sorted(included)]
            included.append(extra)
    return tuple(pairs)


def unit_swap_action_names(game_map):
    if not enabled(game_map):
        return ()
    return tuple(f'swap_{a}_{b}' for a, b in swap_pairs(game_map['hero_roster']))


class UnitSwapRules:
    def __init__(self, game_map, progression, start):
        self.pairs = swap_pairs(game_map['hero_roster'])
        self.count = len(self.pairs)
        self.start, self.end = start, start+self.count
        self.sizes = progression.sizes
        pairs = np.array(self.pairs)
        self.a, self.b = jnp.array(pairs[:, 0]), jnp.array(pairs[:, 1])
        small, big = np.tile(np.arange(6), (self.count, 1)), np.tile(np.arange(6), (self.count, 1))
        for i, (a, b) in enumerate(self.pairs):
            small[i, [a, b]] = b, a
            # A big unit moves its whole column: rows keep their units, the big one stays in front.
            ca, cb = a % 3, b % 3
            big[i, [ca, ca+3, cb, cb+3]] = cb, cb+3, ca, ca+3
        self.small, self.big = jnp.array(small), jnp.array(big)
        self.bits = jnp.left_shift(jnp.uint32(1), jnp.arange(self.count, dtype=jnp.uint32))
        self.observation_size = 1+self.count
        self.metadata = dict(start=start, pairs=[list(p) for p in self.pairs], limit=MAX_PER_TURN, penalty=PENALTY)

    def _pairs(self, state):
        """Per pair: can the two endpoints be exchanged, and does a big unit move a whole column."""
        ids = state.unit_ids[:6]
        slots = jnp.arange(6)
        column = slots % 3
        real = ids != 0  # a dead unit keeps its place and can be moved
        # An empty back slot behind a big front unit is that unit's endpoint.
        owner = jnp.where(real, slots, jnp.where(self.sizes[ids[column]] == 2, column, -1))
        big = (owner >= 0) & (self.sizes[ids[jnp.maximum(owner, 0)]] == 2)
        columns = big[self.a] | big[self.b]
        return jnp.where(columns, column[self.a] != column[self.b], real[self.a] | real[self.b]), columns

    def available(self, state):
        possible, _ = self._pairs(state)
        unused = (state.unit_swaps_used & self.bits) == 0
        return possible & unused & ~state.in_battle & ~state.done & (state.unit_swaps_today < MAX_PER_TURN)

    def valid(self, state, action):
        index = jnp.clip(action-self.start, 0, self.count-1)
        return (action >= self.start) & (action < self.end) & self.available(state)[index]

    def apply(self, state, action, resting):
        index = jnp.clip(action-self.start, 0, self.count-1)
        valid = self.valid(state, action)
        perm = jnp.where(self._pairs(state)[1][index], self.big[index], self.small[index])
        order = jnp.concatenate((jnp.where(valid, perm, jnp.arange(6)), jnp.arange(6, 12)))
        moved = {name: getattr(state, name)[order] for name in SLOT_FIELDS}
        promoted = jnp.sum(jnp.left_shift((state.last_promoted >> order.astype(jnp.uint32)) & 1,
                                          jnp.arange(12, dtype=jnp.uint32)), dtype=jnp.uint32)
        state = state.replace(
            **moved, last_promoted=promoted,
            unit_swaps_today=jnp.where(resting, 0, state.unit_swaps_today+valid.astype(jnp.int32)),
            unit_swaps_used=jnp.where(resting, jnp.uint32(0), state.unit_swaps_used | jnp.where(valid, self.bits[index], jnp.uint32(0))),
            last_swap=jnp.where(valid, index, -1).astype(jnp.int32),
            last_event=jnp.where(valid, UNITS_SWAPPED, state.last_event).astype(jnp.int32))
        return state, jnp.where(valid, -PENALTY, 0.).astype(jnp.float32)

    def observation(self, state):
        return jnp.concatenate((jnp.atleast_1d(state.unit_swaps_today/MAX_PER_TURN),
                                ((state.unit_swaps_used & self.bits) != 0).astype(jnp.float32)))
