"""Strategic terrain costs; Python reference campaign_env_map.py:134-200.

Road/plain/forest/water: 1/2/4/6. Living flying leader: always 2.
A dead leader doubles tile cost and disables flight/terrain boots.
"""
import jax.numpy as jnp

from stoix.envs.number_grid_sites import site_entries

PLAIN, ROAD, FOREST, WATER = range(4)
KINDS = ('plain', 'road', 'forest', 'water')
COSTS = (2, 1, 4, 6)
# Death Knight has the Pegasus Knight's movement abilities in installed Gmabi.dbf.
# Source rows and the game's flight description: docs/validation/death-knight-flight-20261020.json.
FLYING_HERO_IDS = frozenset(('g000uu0019', 'g000uu0022', 'g000uu0072', 'g000uu0096', 'g000uu0099'))


class TerrainRules:
    observation_size = 35  # Local 5x5 types, eight nominal entry costs, flight/death.

    def __init__(self, game_map, progression=None):
        self.size = int(game_map['size'])
        terrain = game_map.get('terrain', {})
        if not isinstance(terrain, dict) or set(terrain)-set(KINDS[1:]):
            raise ValueError('Terrain must map road, forest and water to cell lists')
        occupied = set()
        grid = [PLAIN]*(self.size*self.size)
        obstacles = {tuple(p) for p in game_map.get('obstacles', [])}
        for _, site in site_entries(game_map):
            r,c = site['position']
            obstacles.update((r+dr,c+dc) for dr in range(3) for dc in range(3))
        for ruin in game_map.get('ruins', []):
            r,c = ruin['position']
            obstacles.update((r+dr,c+dc) for dr in range(3) for dc in range(3))
        objects = {tuple(game_map['agent_position']), *map(tuple, game_map['opponent_positions'])}
        objects.add(tuple(game_map.get('capital_position', game_map['agent_position'])))
        objects.update(tuple(o['position']) for key in ('chests','cities','mines') for o in game_map.get(key, []))
        for kind, points in terrain.items():
            for point in points:
                if (not isinstance(point, (list, tuple)) or len(point)!=2
                        or any(type(v) is not int or not 0<v<self.size-1 for v in point)):
                    raise ValueError('Terrain cells must be interior integer coordinates')
                cell = tuple(point)
                if cell in occupied or cell in obstacles:
                    raise ValueError('Terrain cells overlap terrain or an obstacle')
                if kind == 'water' and cell in objects and not game_map.get('allow_water_objects', False):
                    raise ValueError('Water overlaps a map object or army')
                occupied.add(cell)
                grid[cell[0]*self.size+cell[1]] = KINDS.index(kind)
        self.grid = jnp.array(grid, jnp.int32)
        self.base_costs = jnp.array(COSTS, jnp.int32)
        self.local_offsets = jnp.array([(r,c) for r in range(-2,3) for c in range(-2,3)], jnp.int32)
        self.named = progression is not None
        if self.named:
            self.heroes = jnp.array([bool(r.get('hero')) for r in progression.rows])
            self.flying = jnp.array([bool(r.get('hero')) and r.get('game_id') in FLYING_HERO_IDS
                                     for r in progression.rows])

    def kinds(self, positions):
        positions = jnp.clip(positions, 0, self.size-1)
        return self.grid[positions[...,0]*self.size+positions[...,1]]

    def travel_flags(self, state):
        if not self.named:
            return jnp.bool_(False), jnp.bool_(False)
        # Native identity survives temporary combat forms. Summoned/copied heroes
        # cannot supply a travel ability or suppress the dead-leader penalty.
        ids = jnp.where(state.in_battle, state.native_ids, state.unit_ids)[:6]
        genuine = (state.summon_owner[:6]<0) & ~state.copied[:6]
        heroes = self.heroes[ids] & genuine
        hero = jnp.argmax(heroes)
        present = jnp.any(heroes)
        dead = present & (state.hp[hero]<=0)
        flying = present & ~dead & self.flying[ids[hero]]
        return flying, dead

    def costs(self, state, item_rules=None, spells=None):
        flying, dead = self.travel_flags(state)
        costs = self.base_costs
        if item_rules is not None:
            costs = costs.at[FOREST].set(jnp.where(~dead & item_rules.value(state,'forest',4),2,4))
            costs = costs.at[WATER].set(jnp.where(~dead & item_rules.value(state,'water',4),2,6))
        if spells is not None:
            # Reference _campaign_tile_move_cost: a terrain spell caps the tile at
            # the plain cost before the dead-leader doubling, unlike boots.
            forest, water = spells
            costs = costs.at[FOREST].set(jnp.where(forest, jnp.minimum(costs[FOREST], 2), costs[FOREST]))
            costs = costs.at[WATER].set(jnp.where(water, jnp.minimum(costs[WATER], 2), costs[WATER]))
        return jnp.where(flying, 2, costs*jnp.where(dead,2,1))

    def move_cost(self, state, positions, item_rules=None, spells=None):
        return self.costs(state, item_rules, spells)[self.kinds(positions)]

    def observation(self, state, directions, item_rules=None, spells=None):
        flying, dead = self.travel_flags(state)
        return jnp.concatenate((self.kinds(state.position+self.local_offsets)/3.,
            self.move_cost(state,state.position+directions,item_rules,spells)/12.,
            jnp.array([flying,dead], jnp.float32)))
