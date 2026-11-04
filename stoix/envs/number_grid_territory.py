"""Territory, mine income and settlements. Sources: docs/WORLD.md.

Each source's land front order is built once at load; a growth step reads a small
window of that order against the land owners grid, one source after another.
"""
from collections import deque
from functools import lru_cache
import jax
import jax.numpy as jnp
from stoix.envs.number_grid_limits import add_resource

from stoix.envs.number_grid_lords import lord_settings
from stoix.envs.number_grid_updates import set_rows, take_rows

MANA_KINDS = ('infernal', 'life', 'death', 'runes', 'elves')
RESOURCE_KINDS = ('gold',) + MANA_KINDS
RESOURCE_NAMES = ('Золото', 'Мана преисподней', 'Мана жизни', 'Мана смерти', 'Мана рун', 'Мана эльфов')
CITY_CAPTURED, CITY_UPGRADED, OBSTACLE = 43, 44, 45
CAPITAL_ARMOR, CAPITAL_REGENERATION = 50, 35  # GVars PROT_CAP and the capital REGEN bonus
CITY_GROWTH = (0, 15, 15, 20, 20, 25)
CITY_CAPTURE_PERCENT = 100  # Installed GVars T_CAPTURE; see docs/WORLD.md.
CITY_BONUS = (0, 10, 15, 20, 25, 30)
CITY_COST = (0, 150, 250, 500, 750, 0)
LAND_CHUNK = 64  # Tiles of a source's order read at once; a daily quota is at most 25.
# Discipl2.exe table 0x6ef018 as (dr, dc): the land front spreads in four directions.
LAND_STEPS = ((1,0),(-1,0),(0,1),(0,-1))
# Discipl2.exe table 0x6f1690 as (dr, dc): the ring a painted tile checks for a mine.
MINE_RING = ((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1))


def reachable_tiles(size, start, blocked):
    """Tiles a party can walk to from start: eight directions, as a step on the map."""
    blocked = set(blocked)
    seen, queue = {start}, deque([start])
    while queue:
        r, c = queue.popleft()
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                tile = (r+dr, c+dc)
                if 0 < tile[0] < size-1 and 0 < tile[1] < size-1 and tile not in blocked and tile not in seen:
                    seen.add(tile)
                    queue.append(tile)
    return seen


def _heap_push(heap, item):
    """MSVC push_heap of a minimum-depth heap; equal depths keep the original's order."""
    heap.append(item)
    hole = len(heap)-1
    while hole > 0 and item[0] < heap[(hole-1)//2][0]:
        heap[hole] = heap[(hole-1)//2]
        hole = (hole-1)//2
    heap[hole] = item


def _heap_pop(heap):
    """MSVC pop_heap: Discipl2.exe 0x5e1f2f, sift the hole down, then the last item up."""
    top, last = heap[0], heap.pop()
    if not heap:
        return top
    hole, child = 0, 2
    while child < len(heap):
        if heap[child-1][0] < heap[child][0]:
            child -= 1
        heap[hole] = heap[child]
        hole, child = child, 2*child+2
    if child == len(heap):
        heap[hole] = heap[child-1]
        hole = child-1
    while hole > 0 and last[0] < heap[(hole-1)//2][0]:
        heap[hole] = heap[(hole-1)//2]
        hole = (hole-1)//2
    heap[hole] = last
    return top


@lru_cache(maxsize=64)
def land_order(size, start, rejected):
    """Tiles one source's land front visits, in the original order (Discipl2.exe 0x5e1499).

    A rejected tile (0x5e119b) is neither painted nor spread through. Tiles the owner
    already has are spread through for free, so the order does not depend on ownership.
    The start tile itself, the middle of a fort, is never painted.
    """
    heap, seen, order = [], {start}, []

    def spread(tile, depth):
        for dr, dc in LAND_STEPS:
            r, c = tile[0]+dr, tile[1]+dc
            if 0 < r < size-1 and 0 < c < size-1:
                _heap_push(heap, (depth+1, r, c))

    spread(start, 1)
    while heap:
        depth, r, c = _heap_pop(heap)
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if (r, c) not in rejected:
            order.append(r*size+c)
            spread((r, c), depth)
    return tuple(order)


def fort_center(tiles):
    """Where a fort's front starts: corner + size/2, Discipl2.exe 0x5dac34."""
    rows, columns = [t[0] for t in tiles], [t[1] for t in tiles]
    return (min(rows)+(max(rows)-min(rows)+1)//2, min(columns)+(max(columns)-min(columns)+1)//2)


class TerritoryRules:
    def __init__(self, game_map, progression, start, site_rules=None, ruin_rules=None):
        """Prepare settlement, resource, obstacle, and faction growth tables from the map."""
        self.size = int(game_map['size'])
        self.moving_enemies = bool(game_map.get('scheduled_enemies') or game_map.get('scripted_bot')
                                   or game_map.get('scenario_events') or game_map.get('bot_attack'))
        self.capital = tuple(game_map.get('capital_position', game_map['agent_position']))
        self.cities = list(game_map.get('cities', []))
        self.count = len(self.cities)
        self.start, self.end = start, start+self.count
        self.positions = jnp.array([c['position'] for c in self.cities], jnp.int32).reshape(-1,2)
        self.initial_levels = jnp.array([c.get('level',1) for c in self.cities], jnp.int32)
        self.mines = list(game_map.get('mines', []))
        self.mine_positions = jnp.array([m['position'] for m in self.mines], jnp.int32).reshape(-1,2)
        self.mine_kinds = jnp.array([RESOURCE_KINDS.index(m['kind']) for m in self.mines], jnp.int32)
        self.mine_onehot = jax.nn.one_hot(self.mine_kinds, 6, dtype=jnp.int32)
        self.mine_cells = self.mine_positions[:,0]*self.size+self.mine_positions[:,1]
        lord = lord_settings(game_map)
        self.lord = lord['id']
        self.lord_regen = lord['regeneration']
        faction_mana = dict(legions=0, empire=1, undead_hordes=2, mountain_clans=3, elves=4)
        self.passive_mana_income = game_map.get('passive_mana_income_enabled', True)
        if type(self.passive_mana_income) is not bool:
            raise ValueError('passive_mana_income_enabled must be a boolean')
        self.base_income = jnp.array([100]+[25 if self.passive_mana_income and
            i == faction_mana[game_map['faction']] else 0 for i in range(5)], jnp.int32)
        self.heroes = jnp.array([bool(r.get('hero')) for r in progression.rows])
        self.regen = jnp.array([int(r.get('regeneration',r.get('game_data',{}).get('REGEN',0)) or 0) for r in progression.rows],jnp.int32)
        self.growth = jnp.array(CITY_GROWTH,jnp.int32)
        self.fort_bonus = jnp.array(CITY_BONUS,jnp.int32)
        self.costs = jnp.array(CITY_COST,jnp.int32)//lord['city_cost_divisor']
        obstacles = {tuple(p) for p in game_map.get('obstacles',[])}
        reserved = {self.capital, tuple(game_map['agent_position']), *(tuple(c['position']) for c in self.cities),
                    *(tuple(m['position']) for m in self.mines),
                    *(tuple(p) for p in game_map['opponent_positions']),
                    *(tuple(c['position']) for c in game_map.get('chests',[]))}
        if obstacles & reserved:
            raise ValueError('Obstacles overlap an army, city, chest, capital or mine')
        mine_tiles = {tuple(m['position']) for m in self.mines}
        if mine_tiles & {self.capital, tuple(game_map['agent_position']), *(tuple(c['position']) for c in self.cities),
                         *(tuple(p) for p in game_map['opponent_positions']),
                         *(tuple(c['position']) for c in game_map.get('chests',[]))}:
            raise ValueError('A mine cannot share its tile with an army, city, chest, capital or the start')
        points =[self.capital]+[tuple(c['position']) for c in self.cities]+[tuple(m['position']) for m in self.mines]+list(obstacles)
        if any(len(p)!=2 or any(type(v) is not int or not 0<v<self.size-1 for v in p) for p in points):
            raise ValueError('Territory objects must be interior cells')
        if len({tuple(c['position']) for c in self.cities}) != self.count or self.capital in {tuple(c['position']) for c in self.cities}:
            raise ValueError('Settlement entrances must be distinct')
        if any(type(c.get('level',1)) is not int or not 1<=c.get('level',1)<=5 for c in self.cities):
            raise ValueError('City level must be 1..5')
        # All 3x3 sites share the same impassable footprint and entrance rules.
        if site_rules is not None:
            obstacles.update(tuple(p) for site in site_rules.sites for p in site['footprint'])
        if ruin_rules is not None:
            obstacles.update(tuple(p) for ruin in ruin_rules.ruins for p in ruin['footprint'])
        self.obstacle_cells = tuple(sorted(obstacles))
        # The original blocks Crystal tiles (mines and mana sources) for movement:
        # impassable_cells is the movement rule. Land has its own, in _build_land.
        # Discipl2.exe table 0x6f0e90; docs/WORLD.md#territory-владения-и-доход.
        impassable = obstacles | mine_tiles
        self.impassable_cells = tuple(sorted(impassable))
        blocked = [False]*(self.size*self.size)
        for r,c in impassable:
            blocked[r*self.size+c] = True
        self.blocked_grid = jnp.array(blocked)
        directions = ((-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1))
        self.movement_grid = jnp.array([[0<r+dr<self.size-1 and 0<c+dc<self.size-1
            and (r+dr,c+dc) not in impassable for dr,dc in directions]
            for r in range(self.size) for c in range(self.size)],bool)
        walkable = reachable_tiles(self.size, self.capital, self.impassable_cells)
        open_ground = reachable_tiles(self.size, self.capital, obstacles)
        water = {tuple(p) for p in game_map.get('terrain',{}).get('water',[])}
        if ruin_rules is not None:
            reserved.difference_update(tuple(r['entrance']) for r in ruin_rules.ruins)
            for ruin in ruin_rules.ruins:
                if not any(tuple(tile) in walkable for tile in ruin['interaction_tiles']):
                    raise ValueError('Ruin has no reachable approach: '+ruin['name'])
        enforce_enemies = game_map.get('enforce_enemy_reachability', True)
        if type(enforce_enemies) is not bool:
            raise ValueError('enforce_enemy_reachability must be a boolean')
        # A focused magic scenario can enclose armies, while every service and
        # resource must still be reachable. Casting validates its own targets.
        exempt = set() if enforce_enemies else {tuple(p) for p in game_map['opponent_positions']}
        exempt -= {self.capital, tuple(game_map['agent_position']),
                   *(tuple(c['position']) for c in self.cities),
                   *(tuple(m['position']) for m in self.mines),
                   *(tuple(c['position']) for c in game_map.get('chests', []))}
        for p in reserved - exempt:
            # A mine is reached by land, every other object by the party.
            if p not in (open_ground if p in mine_tiles else walkable):
                raise ValueError('Map object is unreachable from capital: '+str(p))
        self.opponent_sources = list(game_map.get('enemy_territory_sources', []))
        self.contested = bool(self.opponent_sources)
        homes = tuple(tuple(s['position']) for s in self.opponent_sources)
        if (len(set(homes)) != len(homes) or self.capital in homes
                or any(len(p) != 2 or any(type(v) is not int or not 0 < v < self.size-1 for v in p)
                       or p in obstacles | water for p in homes)):
            raise ValueError('Opponent territory sources must be distinct passable land cells')
        if any(type(s.get('growth', 10)) is not int or s.get('growth', 10) < 0 for s in self.opponent_sources):
            raise ValueError('Opponent territory growth must be a nonnegative integer')
        self.opponent_growth = jnp.array([s.get('growth', 10) for s in self.opponent_sources], jnp.int32)
        self._build_land(game_map, mine_tiles, water)
        enemy_positions = game_map['opponent_positions']
        self.enemy_count = len(enemy_positions)
        armies = [[] for _ in range(self.size*self.size)]
        for i,(r,c) in enumerate(enemy_positions):
            armies[r*self.size+c].append(i)
        width = max(1,max(map(len,armies)))
        self.armies_by_cell = jnp.array([row+[self.enemy_count]*(width-len(row)) for row in armies],jnp.int32)
        self.city_defenders = self.armies_by_cell[self.positions[:,0]*self.size+self.positions[:,1]]
        self.city_enemies = jnp.array([[tuple(p)==tuple(c['position']) for p in enemy_positions] for c in self.cities],bool).reshape(self.count,len(enemy_positions))
        self.goal_cities = jnp.any(self.city_enemies,axis=1)
        # Reference FINAL_OBJECTIVE_CITIES ("cities" goal): when cities are marked,
        # capturing exactly those wins; no other stack or city is required.
        if any(type(c.get('objective',False)) is not bool for c in self.cities):
            raise ValueError('City objective must be true or false')
        self.objective_cities = any(c.get('objective',False) for c in self.cities)
        if self.objective_cities:
            self.goal_cities = jnp.array([c.get('objective',False) for c in self.cities])
        if game_map.get('objective_enemy') is not None or game_map.get('campaign_objective') in ('waves', 'reach_position', 'all_enemies'):
            # Army/location objectives leave cities usable without requiring capture.
            self.objective_cities = False
            self.goal_cities = jnp.zeros(self.count, bool)
        self.enemy_city = jnp.array([next((i for i,c in enumerate(self.cities) if tuple(p)==tuple(c['position'])),-1) for p in enemy_positions],jnp.int32)
        # Enemy capitals may contain a garrison and a visiting army on one entrance.
        # Guards are not ordinary army goals and have no scripted bot or land spread.
        self.enemy_capitals = list(game_map.get('enemy_capitals', []))
        guard_pairs = [(g, c) for c in self.enemy_capitals for g in c.get('guards', [c.get('guard')])]
        guards = [g for g, _ in guard_pairs]
        if (any(not c.get('guards', [c.get('guard')]) for c in self.enemy_capitals)
                or any(type(g) is not int or not 0 <= g < self.enemy_count
                       or list(enemy_positions[g]) != list(c.get('position', [])) for g, c in guard_pairs)):
            raise ValueError('Each enemy capital needs a guard stack on its position')
        if len(set(guards)) != len(guards) or any(self.enemy_city[g] >= 0 for g in guards):
            raise ValueError('Capital guards must be distinct stacks outside cities')
        self.has_capital_guards = bool(guards)
        self.capital_guards = jnp.array([i in guards for i in range(self.enemy_count)], bool)
        self.enemy_capital_positions = jnp.array(
            [c['position'] for c in self.enemy_capitals], jnp.int32).reshape(-1, 2)
        offsets = [(r,c) for r in range(-2,3) for c in range(-2,3)]
        self.local_offsets = jnp.array(offsets,jnp.int32)
        # Mana, six income values, independent source growth, city and mine rows,
        # local 5x5 ownership/obstacle windows, fort/lord context.
        self.observation_size = 5+6+(1+self.count)+7*self.count+5*len(self.mines)+50+3
        if self.contested:
            self.observation_size += 25+len(self.mines)+len(self.opponent_sources)
        self.metadata = dict(cities=[dict(c,action=start+i) for i,c in enumerate(self.cities)],
            mines=self.mines, resource_kinds=RESOURCE_KINDS, resource_names=RESOURCE_NAMES,
            mine_income=50,base_income=self.base_income.tolist(),capital_growth=10,
            city_growth=CITY_GROWTH,city_bonus=CITY_BONUS,city_costs=self.costs.tolist(),
            city_capture_percent=CITY_CAPTURE_PERCENT,
            capital_regeneration=35,capital_armor=50,own_land_regeneration=10,
            lord_type=self.lord,lord_regeneration=self.lord_regen,obstacles=game_map.get('obstacles',[]),
            enemy_capitals=self.enemy_capitals,capital_guard_armor=CAPITAL_ARMOR,
            capital_guard_regeneration=CAPITAL_REGENERATION,
            enemy_territory_sources=self.opponent_sources)

    def _build_land(self, game_map, mine_tiles, water):
        """Static land tables: start paint, each source's front order and mine painting.

        Sources: docs/WORLD.md#territory-владения-и-доход. A captured neutral city also stops
        other sources' fronts in the original; these orders treat it as neutral throughout.
        """
        tiles = lambda points: {tuple(p) for p in points}
        capital = tiles(game_map.get('capital_footprint', [self.capital]))
        cities = [tiles(c.get('footprint', [c['position']])) for c in self.cities]
        opponents = [tiles(s.get('footprint', [s['position']])) for s in self.opponent_sources]
        # Discipl2.exe 0x5e119b, 0x5e5c82: real players' forts stop a front except its own
        # source. Neutral cities, sites, ruins and ordinary landmarks do not stop it.
        real_forts = [capital, *(f for f, c in zip(cities, self.cities) if not c.get('neutral', True)), *opponents,
                      *(tiles(c.get('footprint', [c['position']])) for c in game_map.get('enemy_capitals', []))]
        ground = tiles(game_map.get('land_blocked', game_map.get('obstacles', []))) | water | mine_tiles
        forts = (capital, *cities, *opponents)
        orders = [land_order(self.size, fort_center(fort),
                             frozenset(ground.union(*(f for f in real_forts if f is not fort))-fort))
                  for fort in forts]
        owners = [0]*(self.size*self.size)
        for r, c in game_map.get('foreign_land', []):
            owners[r*self.size+c] = 2
        for fort in opponents:
            for r, c in fort:
                owners[r*self.size+c] = 2
        for r, c in game_map.get('starting_land', capital):
            owners[r*self.size+c] = 1
        self.initial_owners = jnp.array(owners, jnp.int8)
        self.source_owner = jnp.array([1]*(1+self.count)+[2]*len(opponents), jnp.int8)
        # Padded so that a window read at the end of an order needs no clamping; a tile outside
        # the order has a rank no pointer reaches.
        width = max(map(len, orders))+LAND_CHUNK
        self.land_order = jnp.array([list(o)+[0]*(width-len(o)) for o in orders], jnp.int32)
        ranks = [[width]*(self.size*self.size) for _ in orders]
        for row, order in zip(ranks, orders):
            for rank, cell in enumerate(order):
                row[cell] = rank
        self.land_rank = jnp.array(ranks, jnp.int32)
        self.land_lengths = jnp.array([len(o) for o in orders], jnp.int32)
        self.reachable_counts = self.land_lengths[:1+self.count]
        self.opponent_counts = self.land_lengths[1+self.count:]
        # Everything before a source's pointer is already the owner's: fronts skip it next time.
        me = [1]*(1+self.count)+[2]*len(opponents)
        pointers = [next((i for i, cell in enumerate(o) if owners[cell] != m), len(o)) for o, m in zip(orders, me)]
        self.initial_claims = jnp.array(pointers[:1+self.count], jnp.int32)
        self.initial_opponent_claims = jnp.array(pointers[1+self.count:], jnp.int32)
        # Discipl2.exe 0x60c3b5/0x60c411: a painted tile that touches a mine also paints the
        # mine and its ring, outside the quota; those tiles can touch further mines in turn.
        # Both tables hold one bit per mine: a tile's touch_bits name the mine it touches, its
        # paint_bits every mine whose touch paints it.
        mines = [tuple(m['position']) for m in self.mines]
        if len(mines) > 32:
            raise ValueError('A map holds at most 32 mines')
        index = {tile: i for i, tile in enumerate(mines)}

        def touched(r, c):
            return next((index[(r+dr, c+dc)] for dr, dc in MINE_RING if (r+dr, c+dc) in index),
                        index.get((r, c), -1))

        touch = [0]*(self.size*self.size)
        paint = [0]*(self.size*self.size)
        inside = lambda r, c: 0 < r < self.size-1 and 0 < c < self.size-1
        for i, (r, c) in enumerate(mines):
            for dr, dc in ((0,0), *MINE_RING):
                if inside(r+dr, c+dc):
                    touch[(r+dr)*self.size+c+dc] = 1 << touched(r+dr, c+dc)
            pending, done = [i], {i}
            while pending:
                mr, mc = mines[pending.pop()]
                for dr, dc in ((0,0), *MINE_RING):
                    tile = (mr+dr, mc+dc)
                    if inside(*tile) and tile not in water:
                        paint[tile[0]*self.size+tile[1]] |= 1 << i
                        chained = touched(*tile)
                        if chained >= 0 and chained not in done:
                            done.add(chained)
                            pending.append(chained)
        self.touch_bits = jnp.array(touch, jnp.uint32)
        self.paint_bits = jnp.array(paint, jnp.uint32)

    def _cells(self, positions):
        return positions[...,0]*self.size+positions[...,1]

    def owned(self, state, positions):
        """Query agent ownership at positions without changing the land."""
        return self.owner(state, positions) == 1

    def owner(self, state, positions):
        """Return neutral, agent, or foreign ownership; out-of-map positions are neutral."""
        inside = jnp.all((positions > 0) & (positions < self.size-1), axis=-1)
        cells = self._cells(jnp.clip(positions, 0, self.size-1))
        return jnp.where(inside, state.territory_owners[cells], 0)

    def _spread(self, state, amounts):
        """Paint each source's next amounts[source] tiles in turn: tiles it owns are free.

        Discipl2.exe 0x5e1499 collects that many new tiles, then 0x60c2ff adds touched mines.
        Under vmap every pass runs for the whole batch while any source still grows, so a pass
        only reads tables and selects: no scatter, whose kernel launch would cost far more.
        Measured cost: docs/WORLD.md#territory-владения-и-доход.
        """
        sources = self.land_order.shape[0]
        amounts = amounts.astype(jnp.int32)
        claims = jnp.concatenate((state.territory_claims, state.opponent_claims)).astype(jnp.int32)
        # Either side of a contested map can lose tiles again, so its fronts look for the first
        # tile they do not own instead of trusting their pointers. Elsewhere a front that has
        # read its whole order has nothing left to do.
        pointers = jnp.zeros_like(claims) if self.contested else claims
        growing = amounts > 0 if self.contested else (amounts > 0) & (claims < self.land_lengths)

        def next_source(after):
            # Selects rather than argmax: they fuse with the pass instead of launching a kernel.
            found = jnp.int32(sources)
            for candidate in range(sources-1, -1, -1):
                found = jnp.where(growing[candidate] & (candidate > after), candidate, found)
            return found

        first = next_source(-1)
        window = jnp.arange(LAND_CHUNK, dtype=jnp.int32)

        def body(carry):
            owners, pointers, source, left, touched = carry
            me = self.source_owner[source]
            length = self.land_lengths[source]
            ranks = self.land_rank[source]
            start = pointers[source]
            if self.contested:
                start = jnp.maximum(start, jnp.minimum(length, jnp.min(jnp.where(owners != me, ranks, length))))
            cells = take_rows(self.land_order, source, start+window)
            fresh = (start+window < length) & (owners[cells] != me)
            count = jnp.cumsum(fresh)
            taken = fresh & (count <= left)
            left = left-jnp.minimum(count[-1], left)
            ended = left == 0
            read = jnp.where(ended, LAND_CHUNK-jnp.argmax(taken[::-1]), LAND_CHUNK)
            pointer = jnp.minimum(start+read, length)
            # Everything the pointer passes is either taken now or the front's own already.
            owners = jnp.where((ranks >= start) & (ranks < pointer), me, owners)
            pointers = set_rows(pointers, source, pointer)
            finished = ended | (pointer >= length)
            if self.mines:
                touched = touched | jnp.bitwise_or.reduce(jnp.where(taken, self.touch_bits[cells], jnp.uint32(0)))
                owners = jnp.where(finished & ((self.paint_bits & touched) != 0), me, owners)
                touched = jnp.where(finished, jnp.uint32(0), touched)
            following = next_source(source)
            source = jnp.where(finished, following, source)
            left = jnp.where(finished, amounts[jnp.minimum(following, sources-1)], left)
            return owners, pointers, source, left, touched

        owners, pointers, _, _, _ = jax.lax.while_loop(lambda carry: carry[2] < sources, body,
            (state.territory_owners, pointers, first, amounts[jnp.minimum(first, sources-1)], jnp.uint32(0)))
        pointers = jnp.where(amounts > 0, pointers, claims)
        return state.replace(territory_owners=owners, territory_claims=pointers[:1+self.count],
                             opponent_claims=pointers[1+self.count:])

    def blocked(self, positions):
        outside = jnp.any((positions<=0)|(positions>=self.size-1),axis=-1)
        return outside | self.blocked_grid[self._cells(jnp.clip(positions,0,self.size-1))]

    def movement_mask(self, state):
        return self.movement_grid[self._cells(jnp.clip(state.position,0,self.size-1))]

    def _living(self, state, indices):
        return (indices<self.enemy_count) & state.alive[jnp.minimum(indices,self.enemy_count-1)]

    def enemy_at(self, state, positions):
        if self.moving_enemies:
            present = jnp.all(positions[..., None, :] == state.enemy_positions, axis=-1) & state.alive
            first = jnp.min(jnp.where(present, jnp.arange(self.enemy_count), self.enemy_count), axis=-1)
            return jnp.where(first < self.enemy_count, first, -1)
        indices = self.armies_by_cell[self._cells(jnp.clip(positions,0,self.size-1))]
        first = jnp.min(jnp.where(self._living(state,indices),indices,self.enemy_count),axis=-1)
        return jnp.where(first<self.enemy_count,first,-1)

    def remaining_defenders(self, state):
        return jnp.sum(self._living(state,self.city_defenders),axis=-1)

    def city_at(self, state):
        return jnp.all(self.positions==state.position,axis=1)

    def at_owned_city(self, state):
        return jnp.any(self.city_at(state)&state.city_owned)

    def service_at(self, state):
        return jnp.all(state.position==jnp.array(self.capital)) | self.at_owned_city(state)

    def fort_regeneration(self,state):
        city = jnp.max(jnp.where(self.city_at(state)&state.city_owned,self.fort_bonus[state.city_levels],0),initial=0)
        return jnp.where(jnp.all(state.position==jnp.array(self.capital)),35,city)

    def regeneration_percent(self,state,banner=0):
        place = jnp.where(self.service_at(state),self.fort_regeneration(state),10*self.owned(state,state.position))
        return jnp.where(state.unit_ids!=0,jnp.clip(self.regen[state.unit_ids]+self.lord_regen+place+10*banner,0,100),0)

    def mine_ownership(self,state):
        return self.owned(state,self.mine_positions)

    def income(self,state):
        income = self.base_income+50*jnp.sum(self.mine_onehot*self.mine_ownership(state)[:,None],axis=0)
        return income if self.passive_mana_income else income.at[1:].set(0)

    def next_turn(self,state,resting=True):
        """Where resting, grow the land source by source and credit resources from the resulting land.

        A flag rather than lax.cond: under vmap a cond would select over every field of the state.
        """
        state = self._spread(state, jnp.where(resting, jnp.concatenate((jnp.array([10],jnp.int32),
            jnp.where(state.city_owned,self.growth[state.city_levels],0), self.opponent_growth)), 0))
        income = self.income(state)
        return state.replace(gold=jnp.where(resting,add_resource(state.gold,income[0]),state.gold),
                             mana=jnp.where(resting,add_resource(state.mana,income[1:]),state.mana))

    def leader_alive(self,state):
        return jnp.any(self.heroes[state.unit_ids[:6]] & (state.hp[:6]>0) & (state.summon_owner[:6]<0) & ~state.copied[:6])

    def city_attack_refused(self,state,positions):
        """A party with a dead leader cannot attack a city's defenders; field armies stay open."""
        city = jnp.any(jnp.all(positions[...,None,:]==self.positions,axis=-1)&~state.city_owned,axis=-1)
        return city & (self.enemy_at(state,positions)>=0) & ~self.leader_alive(state)

    def conquered(self,state,fort_victory):
        """The city whose defender the party just beat in a fort battle; capture checks the rest."""
        return fort_victory & jnp.all(self.positions==state.enemy_positions[jnp.maximum(state.enemy,0)],axis=1)

    def capture(self,state,entered,conquest=False):
        """Take a cleared city; grow_captured spreads its land at the end of the step.

        Entry needs a living original hero. The battle that kills the last defender
        takes the city by itself, even if the leader fell in it (Discipl2.exe
        CBattleResultVillage 0x41bc3b -> CEffectCapture), and the party moves in.
        """
        cleared = self.remaining_defenders(state)==0
        captured = ~state.city_owned & cleared & ((entered & self.city_at(state) & self.leader_alive(state)) | conquest)
        state = state.replace(city_owned=state.city_owned|captured,last_captured_cities=captured,
            last_event=jnp.where(jnp.any(captured),CITY_CAPTURED,state.last_event))
        if self.count:
            state = state.replace(position=jnp.where(jnp.any(captured),self.positions[jnp.argmax(captured)],state.position))
        return state

    def grow_captured(self,state,captured):
        """Discipl2.exe sub_4343DA: trunc(T_CAPTURE/100) * T_CITY[level] new tiles at once.

        By the city's own front, without a day tick or the other sources' growth. Called once per
        public step, because capture also runs on every pass of the battle loop and nothing
        reads the land between the two.
        """
        if not self.count:
            return state
        amounts = jnp.where(captured,(CITY_CAPTURE_PERCENT//100)*self.growth[state.city_levels],0)
        return self._spread(state, jnp.concatenate((jnp.zeros(1,jnp.int32),amounts,jnp.zeros_like(self.opponent_growth))))

    def goals_captured(self,state):
        return jnp.all(state.city_owned | ~self.goal_cities)

    def available(self,state):
        return ~state.in_battle & ~state.done & state.city_owned & (state.city_levels<5) & (state.gold>=self.costs[state.city_levels])

    def upgrade(self,state,action):
        if not self.count:
            return state
        index = jnp.clip(action-self.start,0,self.count-1)
        valid = (action>=self.start)&(action<self.end)&self.available(state)[index]
        cost = jnp.where(valid,self.costs[state.city_levels[index]],0)
        return state.replace(city_levels=state.city_levels.at[index].add(valid.astype(jnp.int32)),gold=state.gold-cost,
            last_upgraded_city=jnp.where(valid,index,-1),last_service_cost=jnp.where(valid,cost,state.last_service_cost),
            last_event=jnp.where(valid,CITY_UPGRADED,state.last_event))

    def defender_in_fort(self,state):
        """A defending garrison or visiting army, independent of its armour bonus."""
        position = state.enemy_positions[jnp.maximum(state.enemy, 0)]
        city = jnp.any(jnp.all(self.positions == position, axis=1) & ~state.city_owned)
        capital = jnp.any(jnp.all(self.enemy_capital_positions == position, axis=1))
        return (state.enemy >= 0) & (city | capital)

    def defender_armor(self,state):
        armor = jnp.int32(0)
        if self.count:
            index = self.enemy_city[jnp.maximum(state.enemy,0)]
            armor = jnp.where((state.enemy>=0)&(index>=0)&~state.city_owned[jnp.maximum(index,0)],self.fort_bonus[state.city_levels[jnp.maximum(index,0)]],0)
        if self.has_capital_guards:
            # Reference _resolve_settlement_defender_armor_bonus: the capital tile's guard.
            armor = armor+jnp.where((state.enemy>=0)&self.capital_guards[jnp.maximum(state.enemy,0)],CAPITAL_ARMOR,0)
        return armor

    def observation(self,state):
        """Encode land, resource ownership, and settlement state without changing territory."""
        cities = jnp.concatenate((self.positions/(self.size-1),state.city_levels[:,None]/5.,state.city_owned[:,None],
            state.territory_claims[1:,None]/(self.size*self.size),
            self.remaining_defenders(state)[:,None]/2.,
            self.costs[state.city_levels,None]/1000.),axis=1).reshape(-1)
        mines = jnp.concatenate((self.mine_positions/(self.size-1),self.mine_kinds[:,None]/5.,
            self.mine_ownership(state)[:,None],jnp.full((len(self.mines),1),.05)),axis=1).reshape(-1)
        local = state.position+self.local_offsets
        observation = jnp.concatenate((state.mana/1000.,self.income(state)/100.,state.territory_claims/(self.size*self.size),
            cities,mines,self.owned(state,local),self.blocked(local),
            jnp.array([self.service_at(state),self.fort_regeneration(state)/100.,self.lord_regen/100.])))
        if self.contested:
            observation = jnp.concatenate((observation, self.owner(state, local) == 2,
                self.owner(state, self.mine_positions) == 2, state.opponent_claims/(self.size*self.size)))
        return observation

    def quotes(self,state,banner=0):
        """Expose land masks, mine owners, income, and settlement information for queries and UI."""
        territory = state.territory_owners
        return dict(income=self.income(state),mine_owned=self.mine_ownership(state),
            mine_owner=self.owner(state, self.mine_positions),
            city_available=self.available(state),city_costs=self.costs[state.city_levels],
            at_owned_fort=self.service_at(state),regeneration=self.regeneration_percent(state,banner)[:6],
            territory=(territory == 1).reshape(self.size,self.size),
            opponent_territory=(territory == 2).reshape(self.size,self.size),
            territory_owner=territory.reshape(self.size,self.size))
