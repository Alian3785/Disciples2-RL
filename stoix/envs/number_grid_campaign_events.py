"""Scripted events of a custom .sg campaign (S143 «Дикие горы»), on device with fixed army slots.

Source: the CMidEvent, CMidStackTemplate and CMidLocation objects of the .sg, compiled by
scripts/build_wild_mountains.py into `campaign_events`. User rules of 09.11.2026:

- Armies sent against a capital or city walk the shortest path to its entrance around
  impassable tiles (obstacles, mines and mana sources), spending their leader's movement points
  on terrain as the hero does. They pass other neutral stacks without interacting. An army
  that steps on the hero's party fights it; one that ends its move on an owned entrance
  attacks the garrison (number_grid_garrisons.py).
- Orders to go to or defend a location walk to it and stand; roam, berserk and stand orders
  stand where the army is. Only the Legions' side is played: the passive Undead's events and
  every army sent against Sentinel are not compiled.
- A repeating event recreates its template's army, with full HP at the template levels, only
  when the previous copy is dead; otherwise that creation is skipped.
- Conditions are conjunctions: day frequency, the hero's party entering a zone, a given army in a
  zone, owned cities, killed or absent armies, the agent's land share and looted ruins.
  Effects: create an army, change an army's order, enable or disable an event, a spell on the
  hero's party, and the scenario victory, which is a state check, not a fired event.
- Events with a frequency are checked after the enemy turn of a REST, at the start of the new
  day, so a created army first moves on the next REST. Location-based conditions are checked
  after every map decision. CHANCE is drawn from the battle key, the day and the map step.
"""
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from stoix.envs.number_grid_combat import ATTACK_TYPES
from stoix.envs.number_grid_spells import SPELL_BY_ID
from stoix.envs.number_grid_territory import reachable_tiles

CONDITIONS = ('FREQUENCY', 'HERO_IN_ZONE', 'STACK_IN_ZONE', 'CITY_OWNED', 'STACK_KILLED', 'STACK_ALIVE',
              'LAND_PERCENT', 'RUIN_LOOTED')
EFFECTS = ('CREATE_STACK', 'CHANGE_ORDER', 'ENABLE_EVENT', 'CAST_SPELL', 'WIN')
NO_DISTANCE = 65535


def bfs(size, target, blocked):
    """Steps from every tile to target, eight directions, through unblocked interior tiles."""
    distance = np.full(size*size, NO_DISTANCE, np.int32)
    distance[target[0]*size+target[1]] = 0
    queue = deque([tuple(target)])
    while queue:
        r, c = queue.popleft()
        step = distance[r*size+c]+1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r+dr, c+dc
                if (dr or dc) and 0 < nr < size-1 and 0 < nc < size-1 and (nr, nc) not in blocked \
                        and distance[nr*size+nc] > step:
                    distance[nr*size+nc] = step
                    queue.append((nr, nc))
    return distance


def nearest(cell, walkable):
    """The walkable tile closest to cell: Chebyshev rings, then row and column order."""
    best = min(walkable, key=lambda t: (max(abs(t[0]-cell[0]), abs(t[1]-cell[1])), t))
    return list(best)


class CampaignEventRules:
    def __init__(self, game_map, env):
        spec = game_map.get('campaign_events')
        self.enabled = spec is not None
        self.count = self.mover_count = self.observation_size = 0
        self.win_configured = False
        self.initial_enabled = jnp.zeros(0, bool)
        self.initial_targets = jnp.zeros(0, jnp.int32)
        if not self.enabled:
            return
        if not (env.territory_enabled and env.spell_casting_enabled and env.progression_enabled):
            raise ValueError('campaign_events require territory, persistent enemy wounds and unit progression')
        self.env = env
        territory = env.territory
        self.size = env.size
        events = spec['events']
        self.count = E = len(events)
        enemies = len(game_map['opponent_positions'])
        cities, ruins = territory.count, len(game_map.get('ruins', []))
        self.playable = (self.size-2)**2
        impassable = set(territory.impassable_cells)
        walkable = reachable_tiles(self.size, tuple(territory.capital), impassable)
        self.walkable_grid = jnp.array([(r, c) in walkable for r in range(self.size) for c in range(self.size)])

        # Targets: distinct entrance or location tiles, snapped to the nearest walkable tile.
        targets = []

        def target_index(cell):
            if cell is None:
                return -1
            snapped = nearest(tuple(cell), walkable)
            if snapped not in targets:
                targets.append(snapped)
            return targets.index(snapped)

        # Condition tables; unused columns hold neutral values.
        days = np.zeros(E, np.int32)
        hero_zone = np.zeros((E, 2), np.int32)
        hero_radius = np.full(E, -1, np.int32)
        stack_zone_enemy = np.full(E, -1, np.int32)
        stack_zone = np.zeros((E, 2), np.int32)
        stack_radius = np.full(E, -1, np.int32)
        city_needed = np.zeros((E, max(cities, 1)), bool)
        killed_needed = np.zeros((E, enemies), bool)
        alive_needed = np.zeros((E, enemies), bool)
        absent_needed = np.zeros((E, enemies), bool)
        land_percent = np.zeros(E, np.int32)
        ruin_needed = np.zeros((E, max(ruins, 1)), bool)
        win = np.zeros(E, bool)
        self.creates, self.orders, self.toggles, self.spells = [], [], [], []
        for index, event in enumerate(events):
            for condition in event['conditions']:
                kind = condition['kind']
                if kind == 'FREQUENCY':
                    days[index] = condition['days']
                elif kind == 'HERO_IN_ZONE':
                    if hero_radius[index] >= 0:
                        raise ValueError('One hero zone per event')
                    hero_zone[index], hero_radius[index] = condition['position'], condition['radius']
                elif kind == 'STACK_IN_ZONE':
                    if stack_radius[index] >= 0:
                        raise ValueError('One army zone per event')
                    stack_zone_enemy[index] = condition['enemy']
                    stack_zone[index], stack_radius[index] = condition['position'], condition['radius']
                elif kind == 'CITY_OWNED':
                    city_needed[index, condition['city']] = True
                elif kind == 'STACK_KILLED':
                    killed_needed[index, condition['enemy']] = True
                elif kind == 'STACK_ALIVE':
                    (alive_needed if condition['alive'] else absent_needed)[index, condition['enemy']] = True
                elif kind == 'LAND_PERCENT':
                    land_percent[index] = condition['percent']
                elif kind == 'RUIN_LOOTED':
                    ruin_needed[index, condition['ruin']] = True
                else:
                    raise ValueError('Unknown campaign condition: '+kind)
            for effect in event['effects']:
                kind = effect['kind']
                if kind == 'CREATE_STACK':
                    self.creates.append(dict(event=index, enemy=effect['enemy'], radius=effect['radius'],
                        position=nearest(tuple(effect['position']), walkable),
                        target=target_index(effect.get('target'))))
                elif kind == 'CHANGE_ORDER':
                    self.orders.append(dict(event=index, enemy=effect['enemy'],
                                            target=target_index(effect.get('target'))))
                elif kind == 'ENABLE_EVENT':
                    self.toggles.append(dict(event=index, target=effect['event'], enable=effect['enable']))
                elif kind == 'CAST_SPELL':
                    spell = SPELL_BY_ID[effect['spell']]['cast']
                    if spell['kind'] != 'damage':
                        raise ValueError('Campaign spell events hit the hero party with damage spells only')
                    self.spells.append(dict(event=index, amount=int(spell['amount']),
                                            bit=1 << ATTACK_TYPES.index(spell['element'])))
                elif kind == 'WIN':
                    win[index] = True
                else:
                    raise ValueError('Unknown campaign effect: '+kind)
            if win[index] and len(event['effects']) != 1:
                raise ValueError('A victory event has no other effects')
        self.win_configured = bool(win.any())
        armies = spec['armies']
        self.movers = jnp.array([a['enemy'] for a in armies], jnp.int32)
        self.mover_count = len(armies)
        self.slot = {a['enemy']: i for i, a in enumerate(armies)}
        if len(self.slot) != len(armies) or any(a['moves'] <= 0 for a in armies):
            raise ValueError('Campaign armies must be unique with positive movement points')
        for effect in self.creates+self.orders:
            if effect['enemy'] not in self.slot:
                raise ValueError('Every created or ordered army must be a campaign army')
        self.moves = jnp.array([a['moves'] for a in armies], jnp.int32)
        self.max_moves = max(a['moves'] for a in armies)
        self.initial_targets = jnp.array([target_index(a.get('target')) for a in armies], jnp.int32)
        self.targets = jnp.array(targets, jnp.int32).reshape(-1, 2)
        self.target_count = len(targets)
        self.distances = jnp.array(np.stack([bfs(self.size, t, impassable) for t in targets])
                                   if targets else np.zeros((1, self.size*self.size), np.int32), jnp.int32)
        self.enemy_ids = env.progression.enemy_ids[self.movers]
        self.days, self.hero_zone, self.hero_radius = jnp.array(days), jnp.array(hero_zone), jnp.array(hero_radius)
        self.stack_zone_enemy, self.stack_zone = jnp.array(stack_zone_enemy), jnp.array(stack_zone)
        self.stack_radius = jnp.array(stack_radius)
        self.city_needed, self.killed_needed = jnp.array(city_needed), jnp.array(killed_needed)
        self.alive_needed, self.absent_needed = jnp.array(alive_needed), jnp.array(absent_needed)
        self.land_percent, self.ruin_needed = jnp.array(land_percent), jnp.array(ruin_needed)
        self.win = jnp.array(win)
        self.once = jnp.array([bool(e['once']) for e in events])
        self.chance = jnp.array([e['chance'] for e in events], jnp.int32)
        self.initial_enabled = jnp.array([bool(e['enabled']) for e in events])
        # Centre-first placement within a location's radius (at most 2 in S143).
        radius = max((c['radius'] for c in self.creates), default=0)
        if radius > 2:
            raise ValueError('Campaign placement radius is at most 2')
        offsets = sorted(((r, c) for r in range(-2, 3) for c in range(-2, 3)), key=lambda o: (max(map(abs, o)), o))
        self.offsets = jnp.array(offsets, jnp.int32)
        self.rings = jnp.max(jnp.abs(self.offsets), axis=1)
        self.directions = jnp.array([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c], jnp.int32)
        self.observation_size = 2*E+self.mover_count
        self.metadata = dict(events=[dict(id=e['id'], name=e['name']) for e in events],
                             targets=targets, armies=[a['enemy'] for a in armies])

    # Conditions -------------------------------------------------------------
    def _state_conditions(self, state):
        """Conditions that hold in a state, independent of when they are checked."""
        env = self.env
        spawned_dead = state.scenario_spawned & ~state.alive
        cities = jnp.all(~self.city_needed[:, :state.city_owned.size] | state.city_owned, axis=1)
        killed = jnp.all(~self.killed_needed | spawned_dead, axis=1)
        alive = jnp.all(~self.alive_needed | state.alive, axis=1) & jnp.all(~self.absent_needed | ~state.alive, axis=1)
        land = jnp.sum(state.territory_owners == 1)*100 >= self.land_percent*self.playable
        ruins = (jnp.all(~self.ruin_needed[:, :state.ruin_looted.size] | state.ruin_looted, axis=1)
                 if env.ruins_enabled else ~jnp.any(self.ruin_needed, axis=1))
        safe = jnp.maximum(self.stack_zone_enemy, 0)
        near = jnp.max(jnp.abs(state.enemy_positions[safe]-self.stack_zone), axis=1) <= self.stack_radius
        stack_zone = (self.stack_zone_enemy < 0) | (state.alive[safe] & near)
        return cities & killed & alive & land & ruins & stack_zone

    def won(self, state):
        if not self.win_configured:
            return jnp.bool_(False)
        return jnp.any(self.win & state.campaign_enabled & self._state_conditions(state))

    def process(self, state, new_day):
        """Fire due events: new-day frequency checks and every decision's state checks."""
        if not self.enabled:
            return state
        env = self.env
        party = jnp.any(state.unit_ids[:6] != 0)
        inside = jnp.max(jnp.abs(state.position-self.hero_zone), axis=1) <= self.hero_radius
        before = jnp.max(jnp.abs(state.origin-self.hero_zone), axis=1) <= self.hero_radius
        hero_zone = (self.hero_radius < 0) | (party & inside & ~before)  # the party enters the zone
        frequency = (self.days == 0) | (new_day & (self.days > 0) & (state.day % jnp.maximum(self.days, 1) == 0))
        key = jax.random.fold_in(jax.random.fold_in(state.battle_key, state.day), state.map_steps)
        roll = jax.random.uniform(key, (self.count,))*100. < self.chance
        due = (state.campaign_enabled & ~self.win & (~self.once | ~state.campaign_fired) & frequency & hero_zone
               & self._state_conditions(state) & roll & ~state.done & ~state.in_battle)
        state = jax.lax.cond(jnp.any(due), lambda s: self._apply(env, s, due), lambda s: s, state)
        return state.replace(campaign_fired=state.campaign_fired | due)

    # Effects ------------------------------------------------------------------
    def _apply(self, env, state, due):
        enabled = state.campaign_enabled
        for toggle in self.toggles:
            enabled = enabled.at[toggle['target']].set(jnp.where(due[toggle['event']], bool(toggle['enable']),
                                                                 enabled[toggle['target']]))
        state = state.replace(campaign_enabled=enabled)
        for create in self.creates:
            state = self._create(env, state, due[create['event']], create)
        for order in self.orders:
            enemy, slot = order['enemy'], self.slot[order['enemy']]
            active = due[order['event']] & state.alive[enemy]
            state = state.replace(campaign_targets=state.campaign_targets.at[slot].set(
                jnp.where(active, order['target'], state.campaign_targets[slot])))
        for spell in self.spells:
            state = self._cast(env, state, due[spell['event']], spell)
        return state

    def _create(self, env, state, due, create):
        """A template's army appears at full strength unless its previous copy still lives."""
        enemy, slot = create['enemy'], self.slot[create['enemy']]
        candidates = jnp.array(create['position'], jnp.int32)+self.offsets
        clipped = jnp.clip(candidates, 0, self.size-1)
        others = state.alive & (jnp.arange(state.alive.size) != enemy)
        occupied = jnp.any(jnp.all(state.enemy_positions[:, None, :] == candidates[None], axis=-1) & others[:, None], axis=0)
        occupied |= jnp.all(candidates == state.position, axis=1)
        allowed = (self.walkable_grid[clipped[:, 0]*self.size+clipped[:, 1]] & ~occupied
                   & (self.rings <= create['radius']))
        spawn = due & ~state.alive[enemy] & jnp.any(allowed)
        cell = candidates[jnp.argmax(allowed)]

        def row(values, value):
            return values.at[enemy].set(jnp.where(spawn, jnp.asarray(value).astype(values.dtype), values[enemy]))
        return state.replace(alive=row(state.alive, True), scenario_spawned=row(state.scenario_spawned, True),
            enemy_positions=row(state.enemy_positions, cell), enemy_wounds=row(state.enemy_wounds, 0),
            enemy_progress=row(state.enemy_progress, env.progression.initial_enemies[enemy]),
            enemy_spell_effects=row(state.enemy_spell_effects, 0), summon_marks=row(state.summon_marks, False),
            pending_encounters=row(state.pending_encounters, False),
            garrison_pending=row(state.garrison_pending, False) if env.garrison_control else state.garrison_pending,
            campaign_targets=state.campaign_targets.at[slot].set(
                jnp.where(spawn, create['target'], state.campaign_targets[slot])))

    def _cast(self, env, state, due, spell):
        """A damage spell on every living unit of the hero's party, as ScriptEventRules CAST_SPELL."""
        traits = env.progression.traits[state.unit_ids[:6]]
        blocks = traits[:, 2] | traits[:, 3] | env._equipment_wards(state)[:6]
        hit = due & (state.hp[:6] > 0) & ((blocks & jnp.uint32(spell['bit'])) == 0)
        hp = jnp.where(hit, jnp.maximum(0, state.hp[:6]-spell['amount']), state.hp[:6])
        state = state.replace(hp=state.hp.at[:6].set(hp))
        wiped = due & ~state.done & jnp.any(state.unit_ids[:6] != 0) & ~jnp.any(hp > 0)
        if env.leader_hire.enabled:
            # Apply the map's party-loss rule, just as after a lost battle.
            return env.leader_hire.wipe(env, state, wiped)
        return state.replace(lost=state.lost | wiped, done=state.done | wiped)

    # Enemy turn ------------------------------------------------------------------
    def advance(self, state, resting=True):
        """Ordered armies walk the shortest path to their targets with their movement points."""
        if not self.enabled:
            return state
        env, indices = self.env, self.movers
        target = state.campaign_targets
        party = jnp.any(state.unit_ids[:6] != 0)
        active = (resting & ~state.done & state.alive[indices] & (target >= 0)
                  & ~state.pending_encounters[indices])
        if env.garrison_control:
            active &= ~state.garrison_pending[indices]
        distance = self.distances[jnp.maximum(target, 0)]
        rows = jnp.arange(indices.size)
        base = jnp.broadcast_to(env.terrain.base_costs, (indices.size, 4))
        heroes = env.terrain.heroes[self.enemy_ids]
        leader = jnp.argmax(heroes, axis=1)
        flying = jnp.any(heroes, axis=1) & env.terrain.flying[self.enemy_ids[rows, leader]]
        costs = jnp.where(flying[:, None], 2, base)

        def move(step, carry):
            positions, remaining, contact = carry
            candidates = positions[:, None, :]+self.directions
            clipped = jnp.clip(candidates, 0, self.size-1)
            options = distance[rows[:, None], clipped[..., 0]*self.size+clipped[..., 1]]
            current = distance[rows, positions[:, 0]*self.size+positions[:, 1]]
            allowed = jnp.all((candidates > 0) & (candidates < self.size-1), axis=-1) & (options < current[:, None])
            choice = candidates[rows, jnp.argmin(jnp.where(allowed, options, NO_DISTANCE), axis=1)]
            moving = active & ~contact & (remaining > 0) & jnp.any(allowed, axis=1)
            positions = jnp.where(moving[:, None], choice, positions)
            # As for the hero, the final step may be taken with a partial budget.
            remaining -= jnp.where(moving, jnp.minimum(remaining, costs[rows, env.terrain.kinds(choice)]), 0)
            contact |= active & party & jnp.all(positions == state.position, axis=1)
            return positions, remaining, contact

        positions = state.enemy_positions[indices]
        contact = active & party & jnp.all(positions == state.position, axis=1)
        positions, _, contact = jax.lax.fori_loop(0, self.max_moves, move, (positions, self.moves, contact))
        return state.replace(enemy_positions=state.enemy_positions.at[indices].set(positions),
            pending_encounters=state.pending_encounters.at[indices].set(state.pending_encounters[indices] | contact))

    def observation(self, state):
        return jnp.concatenate((state.campaign_fired, state.campaign_enabled,
                                (state.campaign_targets+1)/(self.target_count+1))).astype(jnp.float32)
