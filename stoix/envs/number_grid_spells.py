"""Research only: compact faction spell actions, with shared mask/step rules.

The 101 supported reference spells are catalogued in data/spells.json. Original
English descriptions are preserved verbatim; descriptions shown in Russian are
translations. Casting is implemented separately in number_grid_casting.
"""
import json
from pathlib import Path

import jax.numpy as jnp

from stoix.envs.number_grid_buildings import DEFAULT_FACTION
from stoix.envs.number_grid_territory import MANA_KINDS

SPELL_LEARNED = 47
CATALOG = json.loads((Path(__file__).parent / 'data/spells.json').read_text(encoding='utf-8'))
SPELL_BY_ID = {row['id']: row for rows in CATALOG['factions'].values() for row in rows}


def shop_spell_ids(game_map):
    """Spells sold by the map's spell shops, in first-offer order and without duplicates."""
    shops = ([game_map['spell_shop']] if game_map.get('spell_shop') is not None
             else list(game_map.get('spell_shops') or []))
    ids = []
    for shop in shops:
        for offer in shop.get('spells', []) if isinstance(shop, dict) else []:
            spell = offer.get('spell') if isinstance(offer, dict) else None
            if spell not in SPELL_BY_ID:
                raise ValueError(f'Unknown spell shop spell: {spell}')
            if spell not in ids:
                ids.append(spell)
    return ids


def spell_book(game_map):
    """Own faction rows, then purchasable spells of other factions (one bit each)."""
    rows = CATALOG['factions'][game_map.get('faction', DEFAULT_FACTION)]
    own = {row['id'] for row in rows}
    extras = [SPELL_BY_ID[i] for i in shop_spell_ids(game_map) if i not in own]
    if len(rows) + len(extras) > 32:
        raise ValueError('At most 32 spells fit the learned-spell bitset')
    return rows, extras


def research_action_names(game_map):
    if not game_map.get('spell_research', False):
        return ()
    return tuple('learn_' + row['id'] for row in CATALOG['factions'][game_map.get('faction', DEFAULT_FACTION)])


class SpellResearchRules:
    def __init__(self, game_map, construction, start):
        self.rows = CATALOG['factions'][construction.faction]
        self.count = len(self.rows)
        self.start, self.end = start, start + self.count
        self.bits = jnp.left_shift(jnp.uint32(1), jnp.arange(self.count, dtype=jnp.uint32))
        self.levels = jnp.array([r['level'] for r in self.rows], jnp.int32)
        self.costs = jnp.array([r['research_mana'] for r in self.rows], jnp.int32) // construction.lord['research_divisor']
        self.scenario_max_level = game_map.get('max_spell_level', 5)
        if type(self.scenario_max_level) is not int or not 1 <= self.scenario_max_level <= 5:
            raise ValueError('max_spell_level must be an integer from 1 to 5')
        self.allowed = self.levels <= min(construction.lord['max_spell_level'], self.scenario_max_level)
        tower = next(i for i, r in enumerate(construction.rows) if r['name'] == 'Башня магии')
        self.tower_bit = jnp.uint32(1 << tower)
        self.observation_size = self.count + 1
        self.metadata = dict(faction=construction.faction, mana_kinds=MANA_KINDS,
            tower_action=18 + tower, reward=1., daily_limit=1, max_level=self.scenario_max_level,
            spells=[dict(row, action=start+i, cost=self.costs[i].tolist()) for i, row in enumerate(self.rows)])

    def available(self, state, index=None):
        bits, costs, allowed = self.bits, self.costs, self.allowed
        if index is not None:
            bits, costs, allowed = bits[index], costs[index], allowed[index]
        return (allowed & ((state.learned_spells & bits) == 0)
                & ((state.buildings & self.tower_bit) != 0)
                & jnp.all(state.mana >= costs, axis=-1) & ~state.spell_researched_today
                & ~state.in_battle & ~state.done)

    def apply(self, state, action):
        index = jnp.clip(action-self.start, 0, self.count-1)
        valid = (action >= self.start) & (action < self.end) & self.available(state, index)
        return state.replace(
            learned_spells=state.learned_spells | jnp.where(valid, self.bits[index], jnp.uint32(0)),
            mana=state.mana-jnp.where(valid, self.costs[index], 0),
            spell_researched_today=state.spell_researched_today | valid,
            last_researched_spell=jnp.where(valid, index, -1),
            last_event=jnp.where(valid, SPELL_LEARNED, state.last_event)), valid.astype(jnp.float32)

    def observation(self, state):
        return jnp.concatenate((((state.learned_spells & self.bits) != 0).astype(jnp.float32),
                                jnp.array([state.spell_researched_today], jnp.float32)))

    def status(self, state):
        # 0 ready, 1 learned, 2 ruler, 3 tower, 4 mana, 5 day, 6 battle, 7 done, 8 scenario.
        code = jnp.where(state.done, 7, jnp.where(state.in_battle, 6,
            jnp.where(state.spell_researched_today, 5,
                jnp.where(jnp.all(state.mana >= self.costs, axis=-1), 0, 4))))
        code = jnp.where((state.buildings & self.tower_bit) == 0, 3, code)
        code = jnp.where(self.allowed, code, 2)
        code = jnp.where(self.levels > self.scenario_max_level, 8, code)
        return jnp.where((state.learned_spells & self.bits) != 0, 1, code)
