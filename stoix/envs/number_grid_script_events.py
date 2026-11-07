"""Functional campaign events on device, with fixed army/inventory capacity.

Source, supported effects and current JAX adaptations are documented in
docs/BETHREZENS_RETREAT.md and docs/MAPS.md. No host callbacks or source parsing during step.
"""
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_combat import ATTACK_TYPES, HP
from stoix.envs.number_grid_items import remove_items
from stoix.envs.number_grid_scenarios import path_distances
from stoix.envs.number_grid_spells import SPELL_BY_ID


class ScriptEventRules:
    def __init__(self, game_map, env):
        self.events = game_map.get('scenario_events', [])
        self.configured = bool(self.events)
        self.enabled = self.configured and game_map.get('scenario_events_enabled', True)
        self.count = len(self.events)
        required_event = game_map.get('objective_after_event')
        self.objective_event = -1
        if required_event is not None:
            matches = [i for i, event in enumerate(self.events) if event['id'] == required_event]
            if (game_map.get('campaign_objective') != 'reach_position' or len(matches) != 1
                    or not any(e['kind'] == 'CREATE_STACK' and e.get('order') == 4
                               for e in self.events[matches[0]]['effects'])):
                raise ValueError('objective_after_event requires one attacking spawn event and a position goal')
            self.objective_event = matches[0]
        count = len(game_map['opponent_positions'])
        reserved = game_map.get('scenario_event_enemies', [])
        self.reserved = jnp.array([i in reserved for i in range(count)], bool)
        self.observation_size = self.count + int(self.objective_event >= 0)
        self.initial_orders = jnp.zeros(0, jnp.int32)
        if not self.configured:
            return
        if not (env.territory_enabled and env.spell_casting_enabled and env.items_enabled):
            raise ValueError('Script events require territory, persistent spell wounds and inventory')
        hero = game_map.get('scenario_hero', game_map.get('objective_hero'))
        eligible = [bool(r.get('hero')) and (hero is None or r.get('key') == hero)
                    for r in env.progression.rows]
        if not any(eligible[int(i)] for i in env.progression.initial_ids[:6]):
            raise ValueError('Script events require their hero in the starting party')
        self.heroes = jnp.array(eligible, bool)
        self.normal_ai = game_map['scenario_normal_ai']
        if self.normal_ai not in ('stationary', 'pursuit'):
            raise ValueError('scenario_normal_ai must be stationary or pursuit')
        self.size = env.size
        self.blocked = env.territory.blocked_grid
        self.directions = jnp.array([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c], jnp.int32)
        self.offsets = jnp.concatenate((jnp.zeros((1, 2), jnp.int32), self.directions))
        # Source conditions are conjunctions. Unused columns carry neutral values.
        days, positions, radii, leaders, items, exists, killed, ends = [], [], [], [], [], [], [], []
        self.effects = {}
        for index, event in enumerate(self.events):
            day, position, radius, leader, item, enemy, end = 0, [0, 0], -1, False, -1, -1, 0
            killed_enemy = -1
            for condition in event['conditions']:
                kind = condition['kind']
                if kind == 'FREQUENCY':
                    day = condition['day']
                elif kind in ('LOCATION', 'LEADER_TO_ZONE', 'ITEM_TO_LOCATION'):
                    position, radius = condition['position'], condition['radius']
                    leader = condition.get('hero', False)
                    if 'item' in condition:
                        item = env.item_rules.keys.index(condition['item'])
                elif kind == 'STACK_EXISTS':
                    enemy = condition['enemy']
                elif kind == 'KILL_STACK':
                    if 'enemy' in condition:
                        killed_enemy = condition['enemy']
                    else:
                        end = -1  # Legacy native condition: the player's hero died.
                else:
                    raise ValueError('Unknown script condition: '+kind)
            for effect in event['effects']:
                if effect['kind'] == 'WIN':
                    end = 1 if effect['win'] else -1
                else:
                    self.effects.setdefault(effect['kind'], []).append(dict(event=index, **effect))
            days.append(day)
            positions.append(position)
            radii.append(radius)
            leaders.append(leader)
            items.append(item)
            exists.append(enemy)
            killed.append(killed_enemy)
            ends.append(end)
        self.days = jnp.array(days, jnp.int32)
        self.positions = jnp.array(positions, jnp.int32)
        self.radii = jnp.array(radii, jnp.int32)
        self.leaders = jnp.array(leaders, bool)
        self.items = jnp.array(items, jnp.int32)
        self.exists = jnp.array(exists, jnp.int32)
        self.ends = jnp.array(ends, jnp.int32)
        self.killed = jnp.array(killed, jnp.int32)
        self.has_kill_triggers = any(i >= 0 for i in killed)
        movers = list(game_map['scenario_order_armies'])
        movers += [dict(enemy=e['enemy'], order=0, moves=e['moves']) for e in self.effects.get('CREATE_STACK', [])]
        self.movers = jnp.array([m['enemy'] for m in movers], jnp.int32)
        self.initial_orders = jnp.array([m['order'] for m in movers], jnp.int32)
        self.moves = jnp.array([m['moves'] for m in movers], jnp.int32)
        self.max_moves = max((m['moves'] for m in movers), default=0)
        self.observation_size += len(movers)
        self.order_slot = {m['enemy']: i for i, m in enumerate(movers)}
        if len(self.order_slot) != len(movers) or any(m['moves'] <= 0 for m in movers):
            raise ValueError('Script movers must be unique with positive movement budgets')
        self.movement_points = game_map.get('scenario_movement_points', False)
        self.capital = jnp.array(env.territory.capital, jnp.int32)
        if self.movement_points:
            self.progression = env.progression
            self.terrain = env.terrain
            self.enemy_ids = env.progression.enemy_ids[self.movers]

        if self.enabled:
            self.distances = path_distances(self.size, env.territory.impassable_cells)
        for kind in ('MOVE_STACK', 'CREATE_STACK', 'MOVE_BESIDE'):
            rows = self.effects.get(kind, [])
            if rows:
                # Native S101 has only radius 0/1 effect destinations. Larger
                # radii are trigger zones, not placement zones in this scenario.
                if any(e.get('radius', 1) > 1 for e in rows):
                    raise ValueError('Unsupported script placement radius')
                setattr(self, kind.lower(), dict(
                    events=jnp.array([e['event'] for e in rows], jnp.int32),
                    enemies=jnp.array([e['enemy'] for e in rows], jnp.int32),
                    positions=jnp.array([e.get('position', [0, 0]) for e in rows], jnp.int32),
                    radii=jnp.array([e.get('radius', 1) for e in rows], jnp.int32)))

    def observation(self, state):
        values = (state.script_events_fired.astype(jnp.float32), state.script_orders/14.)
        if self.objective_event >= 0:
            values += (jnp.atleast_1d(self.objective_ready(state)).astype(jnp.float32),)
        return jnp.concatenate(values)

    def objective_ready(self, state):
        """The optional exit gate opens only after an actual enemy-turn advance."""
        if not self.enabled or self.objective_event < 0:
            return jnp.bool_(True)
        return state.script_events_fired[self.objective_event] & state.script_objective_ready

    def process(self, env, state, new_day=False, locations=True):
        """Snapshot triggers before effects: teleports never recursively trigger."""
        if not self.enabled:
            return state
        in_zone = jnp.max(jnp.abs(self.positions-state.position), axis=1) <= self.radii
        has_item = jnp.any(state.item_inventory[None, :] == self.items[:, None], axis=1)
        has_leader = jnp.any(self.heroes[state.unit_ids[:6]] & (state.hp[:6] > 0)
                            & ~state.copied[:6] & (state.summon_owner[:6] < 0))
        frequency = (self.days > 0) & new_day & (state.day % jnp.maximum(self.days, 1) == 0)
        spatial = (self.radii >= 0) & locations & in_zone
        killed = (self.killed >= 0) & ~state.alive[jnp.maximum(0, self.killed)]
        # Unlike end-of-campaign KILL_STACK, a named enemy's death activates
        # another army, including deaths caused by a strategic spell.
        due = ((frequency | spatial | killed)
               & ((self.days == 0) | frequency) & ((self.radii < 0) | spatial)
               & ((self.killed < 0) | killed) & ((self.items < 0) | has_item)
               & (~self.leaders | has_leader)
               & ((self.exists < 0) | state.alive[jnp.maximum(0, self.exists)])
               & (self.ends == 0) & ~state.script_events_fired & ~state.done & ~state.in_battle)
        state = jax.lax.cond(jnp.any(due), lambda s: self._apply(env, s, due), lambda s: s, state)
        return state.replace(script_events_fired=state.script_events_fired | due)

    def _place(self, state, due, kind):
        table = getattr(self, kind.lower(), None)
        if table is None:
            return state
        spawning, beside = kind == 'CREATE_STACK', kind == 'MOVE_BESIDE'

        def move(index, current):
            enemy = table['enemies'][index]
            origin = current.position if beside else table['positions'][index]
            candidates = origin+self.offsets
            clipped = jnp.clip(candidates, 0, self.size-1)
            cells = clipped[:, 0]*self.size+clipped[:, 1]
            occupied = jnp.any(jnp.all(current.enemy_positions[:, None, :] == candidates[None, :, :], axis=-1)
                               & current.alive[:, None] & (jnp.arange(current.alive.size) != enemy)[:, None], axis=0)
            occupied |= (enemy >= 0) & jnp.all(candidates == current.position, axis=1)
            allowed = (jnp.all((candidates > 0) & (candidates < self.size-1), axis=1)
                       & ~self.blocked[cells] & ~occupied
                       & (jnp.max(jnp.abs(self.offsets), axis=1) <= table['radii'][index]))
            if beside:
                allowed = allowed.at[0].set(False)
            # Stable centre-first free-cell placement. Native free-cell tie order
            # is not established; this is an explicit JAX placement convention.
            target = candidates[jnp.argmax(allowed)]
            active = due[table['events'][index]] & jnp.any(allowed)
            if not spawning:
                active &= (enemy < 0) | current.alive[jnp.maximum(enemy, 0)]
            positions = current.enemy_positions.at[jnp.maximum(enemy, 0)].set(jnp.where(
                active & (enemy >= 0), target, current.enemy_positions[jnp.maximum(enemy, 0)]))
            changes = dict(enemy_positions=positions,
                           position=jnp.where(active & (enemy < 0), target, current.position))
            if spawning:
                changes.update(alive=current.alive.at[enemy].set(current.alive[enemy] | active),
                    scenario_spawned=current.scenario_spawned.at[enemy].set(current.scenario_spawned[enemy] | active))
            return current.replace(**changes)

        return jax.lax.fori_loop(0, len(table['events']), move, state)

    def _apply(self, env, state, due):
        # S101 dependencies: move intro parties before its two map spells;
        # move-beside before removal/exchange. No inter-event effect dependencies.
        for kind in ('MOVE_STACK', 'MOVE_BESIDE', 'CREATE_STACK'):
            state = self._place(state, due, kind)
        for effect in self.effects.get('REMOVE_STACK', []):
            index = effect['enemy']
            active = due[effect['event']]
            state = state.replace(alive=state.alive.at[index].set(state.alive[index] & ~active),
                pending_encounters=state.pending_encounters.at[index].set(state.pending_encounters[index] & ~active))
        for kind in ('CHANGE_ORDER', 'CREATE_STACK'):
            for effect in self.effects.get(kind, []):
                slot = self.order_slot[effect['enemy']]
                active = due[effect['event']] & state.alive[effect['enemy']]
                state = state.replace(script_orders=state.script_orders.at[slot].set(jnp.where(
                    active, effect['order'], state.script_orders[slot])))
        for effect in self.effects.get('BATTLE', []):
            enemy = effect['enemy']
            active = due[effect['event']] & state.alive[enemy]
            # The world-step encounter queue begins this fight in the same
            # player action; it does not wait for an enemy movement turn.
            state = state.replace(pending_encounters=state.pending_encounters.at[enemy].set(
                state.pending_encounters[enemy] | active))
        for effect in self.effects.get('CAST_SPELL', []):
            spell = SPELL_BY_ID[effect['spell']]['cast']
            bit = jnp.uint32(1 << ATTACK_TYPES.index(spell['element']))
            traits = env.progression.traits[state.unit_ids[:6]]
            blocks = traits[:, 2] | traits[:, 3] | env._equipment_wards(state)[:6]
            hit = due[effect['event']] & (state.hp[:6] > 0) & ((blocks & bit) == 0)
            hp = jnp.where(hit, jnp.maximum(0, state.hp[:6]-spell['amount']), state.hp[:6])
            lost = ~jnp.any(hp > 0)
            state = state.replace(hp=state.hp.at[:6].set(hp), lost=state.lost | lost, done=state.done | lost)
        for effect in self.effects.get('CAST_SPELL_MAP', []):
            # Existing JAX spell rule: one party, no area falloff. Target is the
            # scripted coordinate, not the nearest party to the player's hero.
            spell = SPELL_BY_ID[effect['spell']]['cast']
            enemy = env.territory.enemy_at(state, jnp.array(effect['position'], jnp.int32))
            index = jnp.maximum(enemy, 0)
            ids = env.casting.current_enemy_ids(state)[index]
            maximum = env.casting.enemy_max_hp(state)[index]
            traits = env.progression.traits[ids]
            bit = jnp.uint32(1 << ATTACK_TYPES.index(spell['element']))
            hit = due[effect['event']] & (enemy >= 0) & (ids != 0) & (((traits[:, 2] | traits[:, 3]) & bit) == 0)
            wounds = jnp.minimum(maximum, state.enemy_wounds[index]+hit*spell['amount'])
            dead = ~jnp.any((ids != 0) & (wounds < maximum))
            state = state.replace(enemy_wounds=state.enemy_wounds.at[index].set(wounds),
                                  alive=state.alive.at[index].set(state.alive[index] & ~dead))
        for effect in self.effects.get('DESTROY_ITEM', []):
            item = env.item_rules.keys.index(effect['item'])
            # S101 has one quest totem. No other party inventory is simulated.
            remove = due[effect['event']] & (state.item_inventory == item)
            state = remove_items(state, remove)
        for effect in self.effects.get('GIVE_ITEM', []):
            active = due[effect['event']]
            item = env.item_rules.keys.index(effect['item'])
            slot = jnp.sum(state.item_inventory >= 0)
            state = state.replace(item_inventory=state.item_inventory.at[slot].set(
                jnp.where(active, item, state.item_inventory[jnp.minimum(slot, state.item_inventory.size-1)]), mode='drop'),
                last_item_loot=state.last_item_loot.at[item].add(active.astype(jnp.int32)))
        return state

    def advance(self, state):
        """Shortest-path pursuit, with optional terrain movement-point budgets.

        Order 6 targets the player capital. Other active orders target the hero.
        Armies on S103 spend road/plain/forest/water points like ordinary travel;
        existing S101 scripts retain their established cells-per-turn budgets.
        """
        indices = self.movers
        if not indices.size:
            return state
        capital_attack = state.script_orders == 6
        active = state.alive[indices] & ~state.pending_encounters[indices] & (
            capital_attack | (state.script_orders == 4)
            | ((state.script_orders == 1) & (self.normal_ai == 'pursuit')))
        targets = jnp.where(capital_attack[:, None], self.capital, state.position)
        distance = self.distances[targets[:, 0]*self.size+targets[:, 1]]
        rows = jnp.arange(indices.size)
        travel_costs = jnp.ones((indices.size, 4), jnp.int32)
        if self.movement_points:
            heroes = self.terrain.heroes[self.enemy_ids]
            leader = jnp.argmax(heroes, axis=1)
            maximum = self.progression.stats(self.enemy_ids, state.enemy_progress[indices, :, 0])[..., HP]
            dead = jnp.any(heroes, axis=1) & (state.enemy_wounds[indices][rows, leader] >= maximum[rows, leader])
            flying = jnp.any(heroes, axis=1) & ~dead & self.terrain.flying[self.enemy_ids[rows, leader]]
            base = jnp.broadcast_to(self.terrain.base_costs, (indices.size, 4))
            walking = jnp.where(dead[:, None], base.at[:, 1].set(base[:, 0])*2, base)
            travel_costs = jnp.where(flying[:, None], 2, walking)

        def move(step, carry):
            positions, remaining, contact = carry
            candidates = positions[:, None, :]+self.directions
            clipped = jnp.clip(candidates, 0, self.size-1)
            costs = distance[rows[:, None], clipped[..., 0]*self.size+clipped[..., 1]]
            current = distance[rows, positions[:, 0]*self.size+positions[:, 1]]
            allowed = jnp.all((candidates > 0) & (candidates < self.size-1), axis=-1) & (costs < current[:, None])
            choice = candidates[rows, jnp.argmin(jnp.where(allowed, costs, 65535), axis=1)]
            moving = active & ~contact & (remaining > 0) & jnp.any(allowed, axis=1)
            nominal = (travel_costs[rows, self.terrain.kinds(choice)] if self.movement_points
                       else jnp.ones_like(remaining))
            positions = jnp.where(moving[:, None], choice, positions)
            # Existing JAX travel permits the final step with a partial budget.
            remaining -= jnp.where(moving, jnp.minimum(remaining, nominal), 0)
            contact |= active & jnp.all(positions == state.position, axis=1)
            return positions, remaining, contact

        positions = state.enemy_positions[indices]
        contact = active & jnp.all(positions == state.position, axis=1)
        positions, _, contact = jax.lax.fori_loop(0, self.max_moves, move,
                                                  (positions, self.moves, contact))
        responded = (state.script_events_fired[self.objective_event]
                     if self.objective_event >= 0 else jnp.bool_(False))
        return state.replace(enemy_positions=state.enemy_positions.at[indices].set(positions),
            script_objective_ready=state.script_objective_ready | responded,
            pending_encounters=state.pending_encounters.at[indices].set(state.pending_encounters[indices] | contact))

    def outcome(self, state):
        ended = ((self.ends == 1) & state.won) | ((self.ends == -1) & state.lost)
        return state.replace(script_events_fired=state.script_events_fired | ended)
