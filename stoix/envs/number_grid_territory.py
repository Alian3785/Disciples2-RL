"""Territory, mine income and settlements. Sources: docs/TERRITORY_CITIES.md.

Static BFS ranks are built once; runtime queries only gather the ranks needed.
Each source grows independently, including overlap with another owned source.
"""
from collections import deque
from functools import lru_cache
import jax
import jax.numpy as jnp
from stoix.envs.number_grid_limits import add_resource

from stoix.envs.number_grid_lords import lord_settings

MANA_KINDS = ('infernal', 'life', 'death', 'runes', 'elves')
RESOURCE_KINDS = ('gold',) + MANA_KINDS
RESOURCE_NAMES = ('Золото', 'Мана преисподней', 'Мана жизни', 'Мана смерти', 'Мана рун', 'Мана эльфов')
CITY_CAPTURED, CITY_UPGRADED, OBSTACLE = 43, 44, 45
CAPITAL_ARMOR, CAPITAL_REGENERATION = 50, 35  # GVars PROT_CAP and the capital REGEN bonus
CITY_GROWTH = (0, 15, 15, 20, 20, 25)
CITY_CAPTURE_PERCENT = 100  # Installed GVars T_CAPTURE; see docs/TERRITORY_CITIES.md.
CITY_BONUS = (0, 10, 15, 20, 25, 30)
CITY_COST = (0, 150, 250, 500, 750, 0)


@lru_cache(maxsize=32)
def growth_ranks(size, sources, blocked):
    """Reference BFS + Chebyshev/Manhattan tie break, converted from x/y to r/c."""
    blocked = set(blocked)
    rows = []
    for origin in sources:
        distance, queue = {origin: 0}, deque([origin])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                nxt = (r+dr, c+dc)
                if (0 < nxt[0] < size-1 and 0 < nxt[1] < size-1
                        and nxt not in blocked and nxt not in distance):
                    distance[nxt] = distance[(r,c)]+1
                    queue.append(nxt)
        def order(tile, origin=origin, distance=distance):
            dy, dx = tile[0]-origin[0], tile[1]-origin[1]
            return (distance[tile], max(abs(dx),abs(dy)), abs(dx)+abs(dy),
                    abs(dy), abs(dx), tile[0], tile[1])
        ranks = [size*size]* (size*size)
        for rank, (r,c) in enumerate(sorted(distance, key=order)):
            ranks[r*size+c] = rank
        rows.append(tuple(ranks))
    return tuple(rows)


@lru_cache(maxsize=32)
def growth_distances(size, sources, blocked):
    """Unweighted eight-neighbour distances, independent of source reach/owner."""
    blocked = set(blocked)
    result = []
    for origin in sources:
        distances = [size*size]*(size*size)
        distances[origin[0]*size+origin[1]] = 0
        queue = deque([origin])
        while queue:
            r, c = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r+dr, c+dc
                    if (0 < nr < size-1 and 0 < nc < size-1 and (nr, nc) not in blocked
                            and distances[nr*size+nc] == size*size):
                        distances[nr*size+nc] = distances[r*size+c]+1
                        queue.append((nr, nc))
        result.append(tuple(distances))
    return tuple(result)


class TerritoryRules:
    def __init__(self, game_map, progression, start, site_rules=None, ruin_rules=None):
        """Prepare settlement, resource, obstacle, and faction growth tables from the map."""
        self.size = int(game_map['size'])
        self.moving_enemies = bool(game_map.get('scheduled_enemies') or game_map.get('scripted_bot')
                                   or game_map.get('scenario_events'))
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
        self.base_income = jnp.array([100]+[25 if i == faction_mana[game_map['faction']] else 0 for i in range(5)], jnp.int32)
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
        points = [self.capital]+[tuple(c['position']) for c in self.cities]+[tuple(m['position']) for m in self.mines]+list(obstacles)
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
        blocked = [False]*(self.size*self.size)
        for r,c in obstacles:
            blocked[r*self.size+c] = True
        self.blocked_grid = jnp.array(blocked)
        directions = ((-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1))
        self.movement_grid = jnp.array([[0<r+dr<self.size-1 and 0<c+dc<self.size-1
            and (r+dr,c+dc) not in obstacles for dr,dc in directions]
            for r in range(self.size) for c in range(self.size)],bool)
        sources = (self.capital,)+tuple(tuple(c['position']) for c in self.cities)
        ranks = growth_ranks(self.size, sources, tuple(sorted(obstacles)))
        # Water is passable to armies but cannot become faction land. Keep
        # physical reachability separate, including land accessible across water.
        water = {tuple(p) for p in game_map.get('terrain',{}).get('water',[])}
        land_ranks = growth_ranks(self.size, sources, tuple(sorted(obstacles | water))) if water else ranks
        self.ranks = jnp.array(land_ranks,jnp.int32)
        self.reachable_counts = jnp.sum(self.ranks < self.size*self.size,axis=1)
        if ruin_rules is not None:
            reserved.difference_update(tuple(r['entrance']) for r in ruin_rules.ruins)
            for ruin in ruin_rules.ruins:
                if not any(ranks[0][r*self.size+c] < self.size*self.size for r,c in ruin['interaction_tiles']):
                    raise ValueError('Ruin has no reachable approach: '+ruin['name'])
        for p in reserved:
            if ranks[0][p[0]*self.size+p[1]] == self.size*self.size:
                raise ValueError('Map object is unreachable from capital: '+str(p))
        self.initial_claims = jnp.array([1]+[0]*self.count,jnp.int32)
        self.opponent_sources = list(game_map.get('enemy_territory_sources', []))
        self.contested = bool(self.opponent_sources)
        self.initial_owners = jnp.zeros(self.size*self.size if self.contested else 0, jnp.int8)
        if self.contested:
            homes = tuple(tuple(s['position']) for s in self.opponent_sources)
            if (len(set(homes)) != len(homes) or self.capital in homes
                    or any(len(p) != 2 or any(type(v) is not int or not 0 < v < self.size-1 for v in p)
                           or p in obstacles | water for p in homes)):
                raise ValueError('Opponent territory sources must be distinct passable land cells')
            if any(type(s.get('growth', 10)) is not int or s.get('growth', 10) < 0 for s in self.opponent_sources):
                raise ValueError('Opponent territory growth must be a nonnegative integer')
            land_blocked = tuple(sorted(obstacles | water))
            self.distances = jnp.array(growth_distances(self.size, sources, land_blocked), jnp.int32)
            self.opponent_ranks = jnp.array(growth_ranks(self.size, homes, land_blocked), jnp.int32)
            self.opponent_distances = jnp.array(growth_distances(self.size, homes, land_blocked), jnp.int32)
            self.opponent_counts = jnp.sum(self.opponent_ranks < self.size*self.size, axis=1)
            self.opponent_growth = jnp.array([s.get('growth', 10) for s in self.opponent_sources], jnp.int32)
            self.initial_owners = self.initial_owners.at[self.capital[0]*self.size+self.capital[1]].set(1)
            for r, c in homes:
                self.initial_owners = self.initial_owners.at[r*self.size+c].set(2)
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
        if game_map.get('objective_enemy') is not None or game_map.get('campaign_objective') in ('waves', 'reach_position'):
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

    def _cells(self, positions):
        return positions[...,0]*self.size+positions[...,1]

    def owned(self, state, positions):
        """Query agent ownership at positions without advancing claims or repainting contested land."""
        if self.contested:
            return self.owner(state, positions) == 1
        cells = self._cells(jnp.clip(positions,0,self.size-1))
        return jnp.any(self.ranks[:,cells] < state.territory_claims.reshape((-1,)+(1,)*cells.ndim),axis=0)

    def owner(self, state, positions):
        """Return neutral, agent, or opponent ownership; out-of-map positions are neutral."""
        inside = jnp.all((positions > 0) & (positions < self.size-1), axis=-1)
        if self.contested:
            cells = self._cells(jnp.clip(positions, 0, self.size-1))
            return jnp.where(inside, state.territory_owners[cells], 0)
        return (inside & self.owned(state, positions)).astype(jnp.int8)

    def repaint(self, state):
        """Strictly nearer reaching source wins; ties retain the previous owner.

        Python reference campaign_env_territory._refresh_faction_territories.
        This transition is never called from income/observation queries.
        """
        agent = jnp.min(jnp.where(self.ranks < state.territory_claims[:, None],
                                  self.distances, self.size*self.size), axis=0)
        opponent = jnp.min(jnp.where(self.opponent_ranks < state.opponent_claims[:, None],
                                     self.opponent_distances, self.size*self.size), axis=0)
        owners = jnp.where(agent < opponent, 1,
                          jnp.where(opponent < agent, 2, state.territory_owners)).astype(jnp.int8)
        return state.replace(territory_owners=owners)

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
        return self.base_income+50*jnp.sum(self.mine_onehot*self.mine_ownership(state)[:,None],axis=0)

    def next_turn(self,state):
        """Grow both factions, resolve ownership, and credit resources from the resulting land."""
        rates = jnp.concatenate((jnp.array([10],jnp.int32),jnp.where(state.city_owned,self.growth[state.city_levels],0)))
        state = state.replace(territory_claims=jnp.minimum(state.territory_claims+rates,self.reachable_counts))
        if self.contested:
            state = state.replace(opponent_claims=jnp.minimum(
                state.opponent_claims+self.opponent_growth, self.opponent_counts))
            state = self.repaint(state)
        income = self.income(state)
        return state.replace(gold=add_resource(state.gold,income[0]),mana=add_resource(state.mana,income[1:]))

    def capture(self,state,entered):
        """Capture with a living original hero and immediately spread the city's land."""
        hero = jnp.any(self.heroes[state.unit_ids[:6]] & (state.hp[:6]>0) & (state.summon_owner[:6]<0) & ~state.copied[:6])
        cleared = self.remaining_defenders(state)==0
        captured = entered & ~state.city_owned & self.city_at(state) & cleared & hero
        # Discipl2.exe sub_4343DA divides T_CAPTURE by 100 before multiplying
        # by T_CITY[level]. In this grid model, add that one-off budget to the
        # city's own cell; later days keep using the normal independent growth.
        initial = jnp.minimum(1+(CITY_CAPTURE_PERCENT//100)*self.growth[state.city_levels],
                              self.reachable_counts[1:])
        claims = state.territory_claims.at[1:].set(jnp.where(captured,initial,state.territory_claims[1:]))
        state = state.replace(city_owned=state.city_owned|captured,territory_claims=claims,
            last_captured_cities=captured,last_event=jnp.where(jnp.any(captured),CITY_CAPTURED,state.last_event))
        if self.contested and self.count:
            cells = self._cells(self.positions)
            state = state.replace(territory_owners=state.territory_owners.at[cells].set(
                jnp.where(captured, 1, state.territory_owners[cells])))
            # Reconcile the newly reached border now, without a day tick or
            # advancing either faction's other sources. Failed entry is a no-op.
            state = jax.lax.cond(jnp.any(captured),self.repaint,lambda value:value,state)
        return state

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
        territory = (state.territory_owners if self.contested else
                     jnp.any(self.ranks < state.territory_claims[:, None], axis=0).astype(jnp.int8))
        return dict(income=self.income(state),mine_owned=self.mine_ownership(state),
            mine_owner=self.owner(state, self.mine_positions),
            city_available=self.available(state),city_costs=self.costs[state.city_levels],
            at_owned_fort=self.service_at(state),regeneration=self.regeneration_percent(state,banner)[:6],
            territory=(territory == 1).reshape(self.size,self.size),
            opponent_territory=(territory == 2).reshape(self.size,self.size),
            territory_owner=territory.reshape(self.size,self.size))
