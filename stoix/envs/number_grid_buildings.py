"""Capital construction data and vectorized rules shared by PPO and human play.

Prices/requirements were checked against the installed Rise of the Elves DBFs;
Russian names, ordering and exclusions come from the Python reference.
"""
import json
import math
from pathlib import Path

import jax.numpy as jnp

from stoix.envs.number_grid_combat import UNITS
from stoix.envs.number_grid_lords import DEFAULT_FACTION, LORDS, lord_settings

BUILD_START, BUILD_SLOTS = 18, 25
DAILY_GOLD = 100
CATALOG = json.loads((Path(__file__).parent / 'data/buildings.json').read_text(encoding='utf-8'))
FACTIONS = tuple(CATALOG['factions'])


class BuildingRules:
    def __init__(self, game_map):
        self.faction = game_map.get('faction', DEFAULT_FACTION)
        if self.faction not in FACTIONS:
            raise ValueError(f'Unknown faction: {self.faction}')
        faction = CATALOG['factions'][self.faction]
        self.name = faction['name']
        self.lord = lord_settings(game_map)
        unavailable = {u['name']: u['upgrade_unavailable_reason'] for u in UNITS.values()
                       if game_map.get('unit_progression') and u.get('upgrade_unavailable_reason')}
        self.rows = [{**row, 'unavailable_reason': unavailable.get(row['unit'], '')}
                     for row in faction['buildings']]
        count = len(self.rows)
        names = {row['name']: i for i, row in enumerate(self.rows)}
        ids = {row['id']: i for i, row in enumerate(self.rows)}
        if count > BUILD_SLOTS or len(names) != count or len(ids) != count:
            raise ValueError('Invalid building catalogue')
        requires, blocks, ancestors = [], [], []
        for row in self.rows:
            if type(row['gold']) is not int or row['gold'] <= 0:
                raise ValueError('Building prices must be positive integer gold amounts')
            try:
                requires.append(sum(1 << names[name] for name in row['requires']))
                blocks.append(sum(1 << names[name] for name in row['blocks']))
            except KeyError as exc:
                raise ValueError('Unknown building dependency') from exc
        # Compute dependency closure once, before JIT. An excluded prerequisite
        # makes all its descendants permanently unavailable too.
        for i in range(count):
            reached = requires[i]
            while True:
                expanded = reached
                for j in range(count):
                    if reached & (1 << j):
                        expanded |= requires[j]
                if expanded == reached:
                    break
                reached = expanded
            if reached & (1 << i):
                raise ValueError('Cyclic building dependencies')
            ancestors.append(reached)

        def descendants(bits):
            return bits | sum(1 << i for i, parents in enumerate(ancestors)
                              if parents & bits and not bits & (1 << i))

        forbidden = game_map.get('blocked_buildings', [])
        if not isinstance(forbidden, (list, tuple)):
            raise ValueError('blocked_buildings must be a list of building IDs or names')
        initial = sum(1 << i for i, row in enumerate(self.rows) if row['unavailable_reason'])
        for name in forbidden:
            index = ids.get(name, names.get(name))
            if index is None:
                raise ValueError(f'Unknown blocked building for {self.faction}: {name}')
            initial |= 1 << index
        level = game_map.get('max_building_level', 5)
        if type(level) is not int or not 1 <= level <= 5:
            raise ValueError('max_building_level must be an integer from 1 to 5')
        initial |= sum(1 << i for i, row in enumerate(self.rows) if row['level'] > level)
        self.initial_blocked = jnp.uint32(descendants(initial))
        starting = names.get(self.lord['starting_building'])
        self.initial_built = jnp.uint32(0 if starting is None else 1 << starting)
        built = 0 if starting is None else 1 << starting
        for name in game_map.get('starting_built_buildings', []):
            index = ids.get(name, names.get(name))
            if index is None:
                raise ValueError('Unknown starting building: '+str(name))
            built |= 1 << index
        if built & descendants(initial) or any(built & (1 << i) and
                (ancestors[i] & built != ancestors[i] or blocks[i] & built) for i in range(count)):
            raise ValueError('Starting buildings violate dependencies or exclusions')
        self.initial_built = jnp.uint32(built)
        blocked_by_start = 0
        for i in range(count):
            if built & (1 << i):
                blocked_by_start |= blocks[i]
        self.initial_blocked |= jnp.uint32(descendants(blocked_by_start))
        self.lord_observation = jnp.array([kind == self.lord['id'] for kind in LORDS], jnp.float32)
        self.bits = jnp.left_shift(jnp.uint32(1), jnp.arange(BUILD_SLOTS, dtype=jnp.uint32))
        self.present = jnp.arange(BUILD_SLOTS) < count
        self.costs = jnp.asarray([r['gold'] for r in self.rows] + [0] * (BUILD_SLOTS-count), jnp.int32)
        self.requires = jnp.asarray(requires + [0] * (BUILD_SLOTS-count), jnp.uint32)
        self.blocks = jnp.asarray([descendants(b) for b in blocks] + [0] * (BUILD_SLOTS-count), jnp.uint32)
        self.faction_observation = (jnp.arange(len(FACTIONS)) == FACTIONS.index(self.faction)).astype(jnp.float32)
        reward = game_map.get('building_reward', .05)
        if type(reward) not in (int, float) or not math.isfinite(reward) or reward < 0:
            raise ValueError('building_reward must be finite and nonnegative')
        self.reward = float(reward)

    def available(self, state, index=None):
        """The same predicate validates one selected action or the entire mask."""
        bits, present, costs, requires = self.bits, self.present, self.costs, self.requires
        if index is not None:
            bits, present, costs, requires = bits[index], present[index], costs[index], requires[index]
        return (present & ((state.buildings & bits) == 0)
                & ((state.blocked_buildings & bits) == 0)
                & ((state.buildings & requires) == requires)
                & (state.gold >= costs) & ~state.built_today & ~state.in_battle & ~state.done)

    def observation(self, state):
        # Static prices and tree edges are identified by faction; each slot is
        # 1=built, -1=permanently blocked/absent, 0=not yet built.
        status = jnp.where((state.buildings & self.bits) != 0, 1.,
                           jnp.where(((state.blocked_buildings & self.bits) != 0) | ~self.present, -1., 0.))
        values = (self.faction_observation, jnp.asarray([state.built_today], jnp.float32), status)
        return jnp.concatenate(values)

    def status(self, state):
        """Presentation codes: ready, built, blocked, prerequisite, gold, day,
        battle, episode over, absent. The browser only displays these results.
        """
        code = jnp.where(state.done, 7, jnp.where(state.in_battle, 6,
                         jnp.where(state.built_today, 5, jnp.where(state.gold < self.costs, 4, 0))))
        code = jnp.where((state.buildings & self.requires) != self.requires, 3, code)
        code = jnp.where((state.blocked_buildings & self.bits) != 0, 2, code)
        code = jnp.where((state.buildings & self.bits) != 0, 1, code)
        return jnp.where(self.present, code, 8)

    def metadata(self):
        return {'faction': self.faction, 'name': self.name, 'lord': self.lord,
                'lords': [dict(id=key, **value) for key, value in LORDS.items()],
                'factions': [{'id': key, 'name': CATALOG['factions'][key]['name']} for key in FACTIONS],
                'buildings': [{**row, 'action': BUILD_START+i} for i, row in enumerate(self.rows)]}
