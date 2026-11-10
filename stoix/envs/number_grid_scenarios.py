"""Fixed-size scheduled armies, from reference campaign_env_territory.py.

Distances are immutable map data, prepared once on the host. Spawning, pursuit,
encounter queues and objectives run in JAX on the learner's device.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp


@lru_cache(maxsize=4)
def path_distances(size, obstacles):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    blocked = set(obstacles)
    rows, cols = [], []
    for r in range(1, size-1):
        for c in range(1, size-1):
            if (r, c) in blocked:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 < r+dr < size-1 and 0 < c+dc < size-1 and (r+dr, c+dc) not in blocked:
                        rows.append(r*size+c)
                        cols.append((r+dr)*size+c+dc)
    graph = csr_matrix(([1]*len(rows), (rows, cols)), shape=(size*size, size*size))
    distances = shortest_path(graph, directed=False, unweighted=True)
    distances[distances == float('inf')] = 65535
    return jnp.asarray(distances.astype('uint16'))


class ScenarioRules:
    """Scheduled armies pursue the hero; a wave with a target marches to that cell instead.

    city_defence_train (Python reference campaign_env_city_defence.py): the first assault
    appears on the first REST (spawn_turn 0) and marches to the city; the second appears
    `after.days` RESTs after the first one is destroyed (a death is recorded on the REST that
    follows it), then marches.
    """

    def __init__(self, game_map, territory):
        self.events = game_map.get('scheduled_enemies', [])
        self.enabled = bool(self.events)
        self.size = game_map['size']
        count = len(game_map['opponent_positions'])
        enemies = [e['enemy'] for e in self.events]
        if len(set(enemies)) != len(enemies) or any(type(i) is not int or not 0 <= i < count for i in enemies):
            raise ValueError('Scheduled armies require unique enemy indices')
        if any(type(e['moves_per_turn']) is not int or e['moves_per_turn'] < 1 for e in self.events):
            raise ValueError('Scheduled movement must be a positive integer')
        if any(type(e.get('spawn_turn', 0)) is not int or e.get('spawn_turn', 0) < 0
               or ('after' in e) == ('spawn_turn' in e) for e in self.events):
            raise ValueError('A scheduled army needs either a nonnegative spawn_turn or after')
        follows = [e['after'] for e in self.events if 'after' in e]
        if any(set(a) != {'enemy', 'days'} or a['enemy'] not in enemies or a['enemy'] == e['enemy']
               or type(a['days']) is not int or a['days'] < 0 for e in self.events for a in [e.get('after')] if a):
            raise ValueError('after names another scheduled army and a nonnegative number of days')
        targets = [e.get('target') for e in self.events]
        if any(t is not None and (len(t) != 2 or any(type(v) is not int or not 0 < v < self.size-1 for v in t))
               for t in targets):
            raise ValueError('A scheduled target must be an interior cell')
        self.after_slots = jnp.array([enemies.index(e['after']['enemy']) if 'after' in e else -1
                                      for e in self.events], jnp.int32)
        self.after_days = jnp.array([e['after']['days'] if 'after' in e else 0 for e in self.events], jnp.int32)
        self.has_followers = bool(follows)
        self.targeted = jnp.array([t is not None for t in targets], bool)
        self.targets = jnp.array([t or [0, 0] for t in targets], jnp.int32).reshape(-1, 2)
        # User request 2026-10-26 (Default Capital): armies the Empire bot already
        # destroyed are absent at reset; only bot_attack may bring one of the slots.
        absent = game_map.get('absent_enemies', [])
        if (not isinstance(absent, list) or len(set(absent)) != len(absent) or set(absent) & set(enemies)
                or any(type(i) is not int or not 0 <= i < count for i in absent)):
            raise ValueError('absent_enemies must list distinct valid unscheduled enemy indices')
        # Armies missing at reset: not alive and not yet spawned.
        self.scheduled = jnp.array([i in enemies or i in game_map.get('scenario_event_enemies', []) or i in absent
                                    for i in range(count)])
        self.indices = jnp.array(enemies, jnp.int32)
        self.spawn_turns = jnp.array([e.get('spawn_turn', 0) for e in self.events], jnp.int32)
        self.moves = jnp.array([e['moves_per_turn'] for e in self.events], jnp.int32)
        self.wave_indices = jnp.array([e.get('wave_index', 0) for e in self.events], jnp.int32)
        self.wave_groups = jnp.array([[e.get('wave_index', 0) == w for e in self.events]
                                     for w in sorted({e.get('wave_index', 0) for e in self.events})], bool)
        self.wave_numbers = jnp.array(sorted({e.get('wave_index', 0) for e in self.events}), jnp.int32)
        self.max_turn = max(1, max((e.get('spawn_turn', 0) for e in self.events), default=1))
        if self.enabled:
            self.distances = path_distances(self.size, territory.impassable_cells)
            self.target_distances = self.distances[self.targets[:, 0]*self.size+self.targets[:, 1]]
            # Reference tie order is x first, y second, after three greedy choices.
            self.directions = jnp.array([(dy, dx) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                         if dx or dy], jnp.int32)
        self.observation_size = 4*len(self.events)

    def advance(self, state):
        indices = self.indices
        spawned = state.scenario_spawned[indices]
        cleared = jnp.where(spawned & ~state.alive[indices] & (state.scenario_cleared_day < 0),
                            state.day, state.scenario_cleared_day)
        due = state.day-1 >= self.spawn_turns
        if self.has_followers:
            before = cleared[jnp.maximum(self.after_slots, 0)]
            due = jnp.where(self.after_slots >= 0, (before >= 0) & (state.day-before >= self.after_days), due)
        active = due & (~spawned | state.alive[indices])
        alive = state.alive.at[indices].set(state.alive[indices] | (due & ~spawned))
        spawned = state.scenario_spawned.at[indices].set(spawned | due)
        hero = self.distances[state.position[0]*self.size+state.position[1]]
        distance = jnp.where(self.targeted[:, None], self.target_distances, hero[None])
        goals = jnp.where(self.targeted[:, None], self.targets, state.position[None])

        def move(step, positions):
            delta = jnp.sign(goals-positions)
            greedy = jnp.stack((delta, jnp.stack((jnp.zeros_like(delta[:, 0]), delta[:, 1]), axis=-1),
                                jnp.stack((delta[:, 0], jnp.zeros_like(delta[:, 1])), axis=-1)), axis=1)
            offsets = jnp.concatenate((greedy, jnp.broadcast_to(self.directions, (len(self.events), 8, 2))), axis=1)
            candidates = positions[:, None, :]+offsets
            inside = jnp.all((candidates > 0) & (candidates < self.size-1), axis=-1)
            clipped = jnp.clip(candidates, 0, self.size-1)
            costs = jnp.take_along_axis(distance, clipped[..., 0]*self.size+clipped[..., 1], axis=1)
            current = jnp.take_along_axis(distance, (positions[:, 0]*self.size+positions[:, 1])[:, None], axis=1)[:, 0]
            eligible = inside & (costs < current[:, None])
            choice = candidates[jnp.arange(len(self.events)), jnp.argmax(eligible, axis=1)]
            moving = active & (step < self.moves) & jnp.any(eligible, axis=1)
            return jnp.where(moving[:, None], choice, positions)

        positions = jax.lax.fori_loop(0, max(e['moves_per_turn'] for e in self.events), move,
                                      state.enemy_positions[indices])
        contact = active & jnp.all(positions == state.position, axis=1)
        return state.replace(alive=alive, scenario_spawned=spawned, scenario_cleared_day=cleared,
            enemy_positions=state.enemy_positions.at[indices].set(positions),
            pending_encounters=state.pending_encounters.at[indices].set(contact))

    def observation(self, state):
        indices = self.indices
        return jnp.stack((state.scenario_spawned[indices], state.pending_encounters[indices],
            jnp.maximum(0, self.spawn_turns-(state.day-1))/self.max_turn,
            self.wave_indices/11.), axis=-1).reshape(-1)

    def wave_bonus(self, before, after):
        complete_before = before.scenario_spawned[self.indices] & ~before.alive[self.indices]
        complete_after = after.scenario_spawned[self.indices] & ~after.alive[self.indices]
        old = jnp.all(complete_before | ~self.wave_groups, axis=1)
        new = jnp.all(complete_after | ~self.wave_groups, axis=1)
        return 4*jnp.sum(jnp.where(new & ~old, self.wave_numbers, 0))
