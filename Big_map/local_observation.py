"""Egocentric 5x5 observations, without an off-screen map or map memory.

The unchanged baseline party, economy, inventory and battle encodings are kept.
Visible enemies retain baseline per-unit type/HP features, indexed by local tile
rather than global enemy ID. No absolute coordinates or distant target pointers.
"""
import numpy as np
from gymnasium import spaces


class LocalObservation:
    RADIUS = 2
    TILE_FEATURES = (
        'in_bounds', 'walkable', 'capital', 'healing', 'chest', 'ruin',
        'merchant', 'spell_shop', 'mercenary', 'trainer', 'legions_territory',
        'empire_territory', 'gold_mine', 'mana_kind', 'enemy', 'objective_enemy',
    )

    def __init__(self, env):
        self.env = env
        self.enemy_width = env.grid_enemy_unit_slots * env.grid_enemy_unit_feature_size
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
            ('battle', env.BATTLE_OBS_SIZE), ('time_gold', 2),
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
                # Standard maps have at most one active stack per tile. If a
                # scenario overlaps stacks, show the same first-ID encounter.
                team = e.enemy_team_states.get(ids[0], ())
                for slot, unit in enumerate(team[:e.grid_enemy_unit_slots]):
                    if e._is_empty_enemy_unit(unit):
                        continue
                    start = len(self.TILE_FEATURES) + slot * e.grid_enemy_unit_feature_size
                    end = start + e.grid_enemy_unit_type_feature_size
                    result[index, start:end] = e._encode_enemy_unit_type(unit.get('unit_type'))
                    result[index, end] = e._normalize_unit_health(unit)
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
