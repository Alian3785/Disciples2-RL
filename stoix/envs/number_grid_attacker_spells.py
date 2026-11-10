"""Spells at the army that attacks the capital next turn (user rule for S143, 09.11.2026).

Map casts normally hit the nearest stack (number_grid_casting.py). With `attacker_spells`, every
damage, curse or summoning spell of the book, and every such scroll or staff, gets a second action
aimed at the army the thief would poison: one already on the map that stands on the capital
entrance after the next enemy turn (number_grid_thief.capital_attackers), the lowest stack index
first. So the action exists only one turn before that attack. Everything else is the ordinary cast
of the same spell: research or purchase, mana, the daily limit, scroll burning and staff access,
damage that ignores armour, a kill, a curse, or a summon battle against that army. Damage and
curses need a field army, as always. A curse on an army that attacks in the enemy turn lasts
through that battle (user rule 09.11.2026, number_grid_casting.end_curses); curses were left out
until then, when every curse ended on REST before the army moved.
"""
import jax.numpy as jnp

from stoix.envs.number_grid_casting import casting_enabled
from stoix.envs.number_grid_spell_items import scenario_spell_items
from stoix.envs.number_grid_spells import SPELL_BY_ID, spell_book
from stoix.envs.number_grid_thief import capital_attackers

STRIKE_KINDS = ('damage', 'debuff', 'summon_battle')


def enabled(game_map):
    flag = game_map.get('attacker_spells', False)
    if type(flag) is not bool:
        raise ValueError('attacker_spells must be a boolean')
    if flag and not casting_enabled(game_map):
        raise ValueError('attacker_spells needs spell_casting')
    return flag


def _sources(game_map):
    """Striking book rows and scenario scrolls/staves, as (index, name) pairs."""
    own, extras = spell_book(game_map)
    book = [(i, row['id']) for i, row in enumerate(own + extras) if row['cast']['kind'] in STRIKE_KINDS]
    items = [(k, item['key']) for k, item in enumerate(scenario_spell_items(game_map))
             if SPELL_BY_ID[item['spell']]['cast']['kind'] in STRIKE_KINDS]
    return book, items


def attacker_spell_action_names(game_map):
    if not enabled(game_map):
        return ()
    book, items = _sources(game_map)
    return tuple(f'cast_{name}_at_attacker' for _, name in book + items)


class AttackerSpellRules:
    def __init__(self, game_map, start, casting=None, territory=None):
        self.enabled = enabled(game_map)
        self.start = self.end = start
        self.count = self.observation_size = 0
        if not self.enabled:
            return
        if casting is None or territory is None:
            raise ValueError('attacker_spells requires spell casting and territory (the capital)')
        book, items = _sources(game_map)
        self.casting = casting
        self.capital = jnp.asarray(territory.capital, jnp.int32)
        # The ordinary cast action each one stands for: book rows, then scrolls and staves.
        bases = [casting.start + i for i, _ in book] + [casting.item_start + k for k, _ in items]
        self.count = len(bases)
        if not self.count:
            raise ValueError('attacker_spells: the map has no damage or summoning spell')
        self.end = start + self.count
        self.bases = jnp.array(bases, jnp.int32)
        self.observation_size = 3
        self.metadata = dict(start=start, casts=bases)

    def contains(self, action):
        return (action >= self.start) & (action < self.end)

    def base(self, action):
        return self.bases[jnp.clip(action - self.start, 0, self.count - 1)]

    def targets(self, env, state, attackers=None):
        """casting.targets layout: the attacker for every spell, for damage only a field army."""
        attackers = capital_attackers(env, state, self.capital) if attackers is None else attackers
        found = jnp.any(attackers)
        target = jnp.where(found, jnp.argmax(attackers), -1).astype(jnp.int32)
        field = jnp.where(found & self.casting.field[jnp.maximum(target, 0)], target, -1).astype(jnp.int32)
        distance = jnp.max(jnp.abs(state.enemy_positions - state.position), axis=1)
        return target, field, distance

    def available(self, env, state, max_hp, cap, attackers=None):
        mask = self.casting.available(state, max_hp, cap, self.targets(env, state, attackers))
        return mask[self.bases - self.casting.start]

    def observation(self, env, state, attackers=None):
        """An attacker exists, damage can reach it, and its remaining share of base HP."""
        target, field, _ = self.targets(env, state, attackers)
        t = jnp.maximum(target, 0)
        health = jnp.clip(1. - jnp.sum(state.enemy_wounds[t]) / self.casting.base_totals[t], 0., 1.)
        return jnp.stack((target >= 0, field >= 0, jnp.where(target >= 0, health, 0.))).astype(jnp.float32)
