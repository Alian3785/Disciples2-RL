"""Compact semantic local tiles, tactical turn identity and navigation landmarks.

Observation-only experiment: exposes current map state, without selecting actions
or changing rewards, transitions, action masks, rosters or the PPO trainer.
"""
from collections import deque

import numpy as np
from gymnasium import spaces


class LocalObservation:
    ENCODING_VERSION = "local5-navigation-v2"
    RADIUS = 2
    TILE_FEATURES = (
        'in_bounds', 'walkable', 'capital', 'healing', 'chest', 'ruin',
        'merchant', 'spell_shop', 'mercenary', 'trainer', 'legions_territory',
        'empire_territory', 'gold_mine', 'mana_kind', 'enemy', 'objective_enemy',
    )

    def __init__(self, env):
        self.env = env
        self._distance_cache = {}
        self.enemy_width = 4  # alive count, total HP, total damage, max HP
        self.tile_width = len(self.TILE_FEATURES) + self.enemy_width
        sizes = [
            ('mode', 1), ('local_tiles', 25 * self.tile_width),
            ('party', env.grid_blue_obs_size),
            ('resources', env.grid_resource_obs_size - 1),  # No remaining-chest count.
            ('buildings', env.grid_building_obs_size),
            ('spells', env.grid_spell_obs_size - 5),  # No distant enemy debuffs.
            ('merchant_stock', env.grid_current_merchant_stock_obs_size),
            ('spell_shop_stock', env.grid_current_spell_shop_stock_obs_size),
            ('mercenary_roster', env.grid_mercenary_roster_obs_size),
            ('wave_progress', 2 if env.grid_wave_obs_size else 0),
            ('trainer', 3 if env.grid_trainer_obs_size else 0),
            ('battle', env.BATTLE_OBS_SIZE), ('battle_turn', 26),
            ('party_strength', 24), ('landmarks', 32), ('paths', 40),
            ('visited', 25), ('time_gold', 2),
        ]
        self.slices = {}
        offset = 0
        for name, size in sizes:
            self.slices[name] = (offset, offset + size)
            offset += size
        self.space = spaces.Box(0.0, 1.0, shape=(offset,), dtype=np.float32)

    def tiles(self):
        e, grid = self.env, self.env.grid_env
        ax, ay = grid.agent_pos
        result = np.zeros((25, self.tile_width), dtype=np.float32)
        # Only visible positions enter the tile dictionary. No global-ID slots.
        enemies = {}
        for enemy_id, pos in grid.enemy_positions.items():
            if max(abs(pos[0] - ax), abs(pos[1] - ay)) <= self.RADIUS and grid.enemies_alive.get(enemy_id, False):
                enemies.setdefault(tuple(pos), []).append(enemy_id)
        def interaction_tiles(attribute):
            return {tuple(p) for tiles in getattr(e, attribute, {}).values() for p in tiles}
        merchants = interaction_tiles('merchant_site_interaction_tiles')
        shops = interaction_tiles('spell_shop_site_interaction_tiles')
        mercenaries = interaction_tiles('mercenary_site_interaction_tiles')
        heals = set(e.castle_heal_tiles) | {tuple(e.CASTLE_POS)}
        ruins = set(e.grid_ruin_positions)
        mines = set(getattr(e, 'gold_mine_tiles', ()))
        for index, (dy, dx) in enumerate((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)):
            pos = (ax + dx, ay + dy)
            if not (0 <= pos[0] < grid.grid_size and 0 <= pos[1] < grid.grid_size):
                continue  # Explicit all-zero padding beyond the map boundary.
            mana = e.mana_sources.get(pos, {})
            ids = sorted(enemies.get(pos, ()))
            result[index, :len(self.TILE_FEATURES)] = [
                1, pos not in grid.obstacle_positions and pos not in grid.dynamic_blocked_positions,
                pos == tuple(e.CASTLE_POS), pos in heals, pos in e.chests, pos in ruins,
                pos in merchants, pos in shops, pos in mercenaries,
                pos in e.trainer_interaction_tiles, pos in e.legions_territory_tile_set,
                pos in e.empire_territory_tile_set, pos in mines,
                e.grid_mana_kind_to_norm.get(str(mana.get('kind', '')), 0),
                bool(ids), e._map.objective_enemy_id in ids,
            ]
            if ids:
                # Aggregate all living stacks sharing the visible tile.
                team = [u for enemy_id in ids for u in e.enemy_team_states.get(enemy_id, ())
                        if not e._is_empty_enemy_unit(u) and self.health(u) > 0]
                hp = sum(self.health(u) for u in team)
                damage = sum(float(u.get('damage', 0) or 0) for u in team)
                maximum = sum(self.max_health(u) for u in team)
                result[index, len(self.TILE_FEATURES):] = (
                    len(team) / 6.0, hp / (hp + 400.0),
                    damage / (damage + 200.0), maximum / (maximum + 400.0))
        return result.ravel()

    @staticmethod
    def health(unit):
        return max(0.0, float(unit.get('health', unit.get('hp', 0)) or 0))

    @staticmethod
    def max_health(unit):
        return max(0.0, float(unit.get('max_health', unit.get('maxhp', 0)) or 0))

    def battle_turn(self):
        result = np.zeros(26, dtype=np.float32)
        e = self.env
        battle = e.battle_env
        if e.mode != e.MODE_BATTLE or battle is None:
            return result
        active = getattr(battle, 'current_blue_attacker_pos', None)
        for u in battle.combined:
            pos = int(u.get('position', 0) or 0)
            if 1 <= pos <= 12 and self.max_health(u) > 0:
                result[pos - 1] = float(pos == active and self.health(u) > 0)
                maximum = self.max_health(u)
                result[12 + pos - 1] = maximum / (maximum + 200.0)
        attacks = max(0.0, float(getattr(battle, 'blue_attacks_left', 0) or 0))
        result[24] = attacks / (attacks + 1.0)
        result[25] = getattr(battle, '_post_victory_team', None) == 'blue'
        return result

    def party_strength(self):
        result = np.zeros((6, 4), dtype=np.float32)
        for u in self.env._get_blue_state() or ():
            slot = int(u.get('position', 0) or 0) - 7
            if not 0 <= slot < 6 or self.max_health(u) <= 0:
                continue
            hp = self.max_health(u)
            damage = max(0.0, float(u.get('damage', 0) or 0))
            armor = max(0.0, float(u.get('armor', 0) or 0))
            initiative = max(0.0, float(u.get('initiative', 0) or 0))
            result[slot] = (hp / (hp + 200.0), damage / (damage + 100.0),
                            armor / 165.0, initiative / (initiative + 50.0))
        return result.ravel()

    def landmarks(self):
        e, grid = self.env, self.env.grid_env
        ax, ay = grid.agent_pos
        scale = max(1, grid.grid_size - 1)
        result = np.zeros(32, dtype=np.float32)
        result[:2] = (ax / scale, ay / scale)
        living = [tuple(pos) for enemy_id, pos in grid.enemy_positions.items()
                  if grid.enemies_alive.get(enemy_id, False)]
        objective_id = e._map.objective_enemy_id
        objective = ([tuple(grid.enemy_positions[objective_id])]
                     if grid.enemies_alive.get(objective_id, False)
                     and objective_id in grid.enemy_positions else [])
        groups = (objective, living, [tuple(e.CASTLE_POS)],
                  list(e.castle_heal_tiles), list(e.chests), list(e.trainer_interaction_tiles))
        for index, positions in enumerate(groups):
            if not positions:
                continue
            x, y = min(positions, key=lambda p: (max(abs(p[0]-ax), abs(p[1]-ay)), p[0], p[1]))
            dx, dy = x - ax, y - ay
            result[2 + 5*index:7 + 5*index] = (
                1, 0.5 + 0.5*dx/scale, 0.5 + 0.5*dy/scale,
                max(abs(dx), abs(dy))/scale, float(dx == 0 and dy == 0))
        return result

    def visited(self):
        grid = self.env.grid_env
        ax, ay = grid.agent_pos
        return np.array([(ax + dx, ay + dy) in grid.visited_cells
                         for dy in range(-2, 3) for dx in range(-2, 3)], dtype=np.float32)

    def distance_field(self, group, targets, blocked):
        """Terrain-only geodesic distance; does not choose or execute actions."""
        grid = self.env.grid_env
        size = grid.grid_size
        targets = tuple(sorted(set(tuple(p) for p in targets)))
        signature = (size, targets, blocked)
        cached = self._distance_cache.get(group)
        if cached is not None and cached[0] == signature:
            return cached[1]
        distance = np.full((size, size), -1, dtype=np.int32)
        queue = deque()
        for x, y in targets:
            if 0 <= x < size and 0 <= y < size and (x, y) not in blocked:
                distance[y, x] = 0
                queue.append((x, y))
        deltas = grid.ACTION_DELTAS[:8]
        while queue:
            x, y = queue.popleft()
            next_distance = int(distance[y, x]) + 1
            for dx, dy in deltas:
                nx, ny = x + dx, y + dy
                if (0 <= nx < size and 0 <= ny < size and
                        distance[ny, nx] < 0 and (nx, ny) not in blocked):
                    distance[ny, nx] = next_distance
                    queue.append((nx, ny))
        distance.setflags(write=False)
        self._distance_cache[group] = (signature, distance)
        return distance

    def paths(self):
        e, grid = self.env, self.env.grid_env
        result = np.zeros((4, 10), dtype=np.float32)
        # These spatial fields have no use during a tactical battle.
        if e.mode == e.MODE_BATTLE:
            return result.ravel()
        ax, ay = grid.agent_pos
        if not (0 <= ax < grid.grid_size and 0 <= ay < grid.grid_size):
            return result.ravel()
        objective_id = e._map.objective_enemy_id
        objective = ([grid.enemy_positions[objective_id]]
                     if grid.enemies_alive.get(objective_id, False)
                     and objective_id in grid.enemy_positions else [])
        groups = (objective, [tuple(e.CASTLE_POS)], list(e.chests), list(e.trainer_interaction_tiles))
        blocked = frozenset(grid.obstacle_positions) | frozenset(grid.dynamic_blocked_positions)
        for index, targets in enumerate(groups):
            if not targets:
                continue
            field = self.distance_field(index, targets, blocked)
            current = int(field[ay, ax])
            if current < 0:
                continue
            result[index, 0] = 1.0
            result[index, 1] = current / (current + 20.0)
            for action, (dx, dy) in enumerate(grid.ACTION_DELTAS[:8]):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < grid.grid_size and 0 <= ny < grid.grid_size:
                    value = int(field[ny, nx])
                    if value >= 0:
                        result[index, 2 + action] = max(0.0, min(1.0, 0.5 + 0.5*(current-value)))
        return result.ravel()

    def build(self, battle_obs=None):
        e = self.env
        obs = np.zeros(self.space.shape, dtype=np.float32)
        def put(name, values):
            start, end = self.slices[name]
            obs[start:end] = values
        put('mode', float(e.mode))
        put('local_tiles', self.tiles())
        put('party', e._build_blue_team_grid_obs())
        put('party_strength', self.party_strength())
        put('landmarks', self.landmarks())
        put('paths', self.paths())
        put('visited', self.visited())
        put('battle_turn', self.battle_turn())
        resource = e._build_resource_grid_obs()
        put('resources', np.concatenate((resource[:9], resource[10:])))
        put('buildings', e._build_building_grid_obs())
        spells = e._build_spell_grid_obs()
        debuff_start = 3 + len(e.spell_keys) + e.grid_map_offensive_spell_action_count
        put('spells', np.concatenate((spells[:debuff_start], spells[debuff_start + 5:])))
        put('merchant_stock', e._build_current_merchant_stock_grid_obs())
        put('spell_shop_stock', e._build_current_spell_shop_stock_grid_obs())
        put('mercenary_roster', e._build_current_mercenary_roster_grid_obs())
        if e.grid_wave_obs_size:
            # Public event countdown and own completed waves, no live enemy count
            # or direction/distance to wave stacks outside the viewport.
            put('wave_progress', e._build_wave_grid_obs()[[0, 3]])
        if e.grid_trainer_obs_size:
            # Current-site services only; no pointer to an off-screen trainer.
            if e._trainer_sites_at_position(e.grid_env.agent_pos):
                put('trainer', e._build_trainer_grid_obs()[2:])
        if e.mode == e.MODE_BATTLE and battle_obs is not None:
            put('battle', battle_obs)
        turns, gold = max(0, float(e.turns)), max(0, float(e.gold))
        put('time_gold', [turns / (turns + e.turn_norm_k), gold / (gold + e.gold_norm_k)])
        np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(obs, 0.0, 1.0, out=obs)
        return obs
