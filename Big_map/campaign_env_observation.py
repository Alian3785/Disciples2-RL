# campaign_env_observation.py
"""CampaignObservationMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignObservationMixin:
    def _init_enemy_obs_metadata(self) -> None:
        """Готовит метаданные для grid-наблюдения о врагах."""
        unit_types = set()
        max_units = 0
        for team in ENEMY_CONFIGS.values():
            if not team:
                continue
            max_units = max(max_units, len(team))
            for unit in team:
                if self._is_empty_enemy_unit(unit):
                    continue
                unit_type = unit.get("unit_type")
                if unit_type:
                    unit_types.add(unit_type)

        self.enemy_unit_types = sorted(unit_types)
        self.enemy_unit_type_to_index = {t: i for i, t in enumerate(self.enemy_unit_types)}
        self.grid_enemy_unit_type_feature_size = len(self.enemy_unit_types)
        self.grid_enemy_unit_feature_size = self.grid_enemy_unit_type_feature_size + 1
        self.grid_enemy_unit_slots = max_units
        self.grid_enemy_count = len(self.grid_env.enemy_positions or {})
        self.grid_enemy_obs_size = self.grid_enemy_count * (
            2 + self.grid_enemy_unit_slots * self.grid_enemy_unit_feature_size
        )
    def _init_campaign_grid_obs_metadata(self) -> None:
        """Готовит метаданные для grid-наблюдения о состоянии кампании."""
        self.grid_blue_positions = tuple(int(pos) for pos in self.GRID_BOTTLE_POSITIONS)
        self.grid_blue_features_per_unit = 8
        self.grid_blue_obs_size = len(self.grid_blue_positions) * self.grid_blue_features_per_unit

        self.grid_battle_equipment_item_names = tuple(
            str(item_name) for item_name in self.BATTLE_EQUIPPABLE_ITEM_NAMES
        )
        self.grid_battle_equipment_features_per_slot = (
            len(self.grid_battle_equipment_item_names) + 1
        )
        self.grid_battle_equipment_obs_size = (
            int(self.BATTLE_EQUIP_SLOTS) * self.grid_battle_equipment_features_per_slot
        )
        self.grid_resource_core_obs_size = 11 + self.grid_battle_equipment_obs_size
        self.grid_mana_total_kinds = tuple(str(kind) for kind in self.MANA_KIND_ORDER)
        self.grid_mana_total_obs_size = len(self.grid_mana_total_kinds)
        self.grid_resource_obs_size = (
            self.grid_resource_core_obs_size + self.grid_mana_total_obs_size
        )

        mana_kind_order = {str(kind): index for index, kind in enumerate(self.MANA_KIND_ORDER)}
        self.grid_mana_source_entries = tuple(
            {
                "pos": tuple(int(coord) for coord in tuple(tile_pos)[:2]),
                "kind": str(mana_meta.get("kind", "") or ""),
            }
            for tile_pos, mana_meta in sorted(
                getattr(self, "mana_sources", {}).items(),
                key=lambda item: (
                    mana_kind_order.get(str(item[1].get("kind", "") or ""), len(mana_kind_order)),
                    int(item[0][1]),
                    int(item[0][0]),
                ),
            )
        )
        self.grid_mana_kind_to_norm = {
            str(kind): float(index + 1) / float(max(1, len(self.MANA_KIND_ORDER)))
            for index, kind in enumerate(self.MANA_KIND_ORDER)
        }
        self.grid_mana_source_features_per_entry = 3
        self.grid_mana_source_obs_size = (
            len(self.grid_mana_source_entries) * self.grid_mana_source_features_per_entry
        )
        self.grid_mana_total_half_point_by_kind = {}
        for kind in self.MANA_KIND_ORDER:
            source_count = sum(
                1
                for mana_meta in getattr(self, "mana_sources", {}).values()
                if str(mana_meta.get("kind", "") or "") == str(kind)
            )
            max_income = float(source_count) * float(self.LEGIONS_MANA_SOURCE_MANA_PER_TURN)
            if str(kind) == "infernal":
                max_income += float(self.BASE_INFERNAL_MANA_PER_TURN)
            self.grid_mana_total_half_point_by_kind[str(kind)] = max(
                1.0,
                float(self.turn_norm_k) * max(50.0, max_income),
            )

        self.grid_building_features_per_building = 3
        self.grid_building_obs_size = (
            len(self.building_keys) * self.grid_building_features_per_building
        )
        self.grid_spell_core_obs_size = 3
        self.grid_spell_features_per_spell = 1
        self.grid_legion_spell_used_features_per_spell = 1
        self.grid_map_offensive_spell_action_count = self._map_offensive_spell_action_max_count()
        self.grid_legion_spell_used_obs_size = (
            self.grid_map_offensive_spell_action_count
            * self.grid_legion_spell_used_features_per_spell
        )
        self.grid_nearest_enemy_debuff_flag_names = (
            "any",
            "armor",
            "damage",
            "initiative",
            "accuracy",
        )
        self.grid_nearest_enemy_debuff_obs_size = len(self.grid_nearest_enemy_debuff_flag_names)
        self.grid_scroll_core_obs_size = 3
        self.grid_scroll_slot_count = int(self.MAX_SCROLL_CAST_ACTIONS)
        self.grid_scroll_slot_features_per_entry = 4
        self.grid_scroll_obs_size = (
            self.grid_scroll_core_obs_size
            + self.grid_scroll_slot_count * self.grid_scroll_slot_features_per_entry
        )
        self.grid_spell_obs_size = (
            self.grid_spell_core_obs_size
            + len(self.spell_keys) * self.grid_spell_features_per_spell
            + self.grid_legion_spell_used_obs_size
            + self.grid_nearest_enemy_debuff_obs_size
            + self.grid_scroll_obs_size
        )

        self.grid_chest_positions = tuple(sorted(tuple(pos) for pos in self._static_chests.keys()))
        self.grid_chest_features_per_entry = 3
        self.grid_chest_obs_size = (
            len(self.grid_chest_positions) * self.grid_chest_features_per_entry
        )

        self.grid_heal_tile_positions = tuple(
            sorted(tuple(pos) for pos in self._static_castle_heal_tiles)
        )
        self.grid_heal_tile_features_per_entry = 2
        self.grid_heal_tile_obs_size = (
            len(self.grid_heal_tile_positions) * self.grid_heal_tile_features_per_entry
        )

        self.grid_legions_territory_positions = tuple(
            (x, y)
            for y in range(int(self.grid_size))
            for x in range(int(self.grid_size))
        )
        self.grid_legions_territory_features_per_entry = 1
        self.grid_legions_territory_obs_size = (
            len(self.grid_legions_territory_positions)
            * self.grid_legions_territory_features_per_entry
        )
        self.grid_legions_territory_index_by_pos = {
            tuple(tile_pos): index
            for index, tile_pos in enumerate(self.grid_legions_territory_positions)
        }
        self.grid_empire_territory_positions = self.grid_legions_territory_positions
        self.grid_empire_territory_features_per_entry = 1
        self.grid_empire_territory_obs_size = (
            len(self.grid_empire_territory_positions)
            * self.grid_empire_territory_features_per_entry
        )

        self.grid_ruin_positions = tuple(
            sorted(
                tuple(int(coord) for coord in tuple(reward_data.get("ruin_pos", (0, 0)))[:2])
                for reward_data in self.RUIN_REWARD_BY_ENEMY_ID.values()
            )
        )
        self.grid_ruin_features_per_entry = 2
        self.grid_ruin_obs_size = (
            len(self.grid_ruin_positions) * self.grid_ruin_features_per_entry
        )

        self.grid_objective_city_names = tuple(self.FINAL_OBJECTIVE_CITIES.keys())
        self.grid_objective_city_features_per_entry = 4
        self.grid_objective_city_obs_size = (
            len(self.grid_objective_city_names) * self.grid_objective_city_features_per_entry
        )
        self.grid_merchant_site_names = tuple(
            str(site_name) for site_name in self.BASE_MERCHANT_SITE_DATA.keys()
        )
        self.grid_merchant_site_features_per_entry = 5
        self.grid_merchant_site_obs_size = (
            len(self.grid_merchant_site_names) * self.grid_merchant_site_features_per_entry
        )
        self.grid_merchant_item_names = tuple(
            str(item_data.get("name", "") or "") for item_data in self.MERCHANT_BUY_ITEMS
        )
        self.grid_merchant_item_features_per_entry = 1
        self.grid_current_merchant_stock_obs_size = (
            len(self.grid_merchant_item_names) * self.grid_merchant_item_features_per_entry
        )
        self.grid_spell_shop_site_names = tuple(
            str(site_name) for site_name in self.BASE_SPELL_SHOP_SITE_DATA.keys()
        )
        self.grid_spell_shop_site_features_per_entry = 5
        self.grid_spell_shop_site_obs_size = (
            len(self.grid_spell_shop_site_names) * self.grid_spell_shop_site_features_per_entry
        )
        self.grid_spell_shop_spell_names = tuple(
            str(spell_data.get("name", "") or "") for spell_data in self.SPELL_SHOP_BUY_SPELLS
        )
        self.grid_spell_shop_spell_features_per_entry = 1
        self.grid_current_spell_shop_stock_obs_size = (
            len(self.grid_spell_shop_spell_names) * self.grid_spell_shop_spell_features_per_entry
        )
        self.grid_mercenary_site_names = tuple(
            str(site_name) for site_name in self.BASE_MERCENARY_SITE_DATA.keys()
        )
        self.grid_mercenary_site_features_per_entry = 5
        self.grid_mercenary_site_obs_size = (
            len(self.grid_mercenary_site_names) * self.grid_mercenary_site_features_per_entry
        )
        self.grid_mercenary_slot_count = max(
            (len(roster) for roster in self.mercenary_site_rosters.values()),
            default=0,
        )
        self.grid_mercenary_slot_features_per_entry = 8
        self.grid_mercenary_roster_obs_size = (
            self.grid_mercenary_slot_count * self.grid_mercenary_slot_features_per_entry
        )
        self.grid_trainer_features_per_entry = 5
        self.grid_trainer_obs_size = self.grid_trainer_features_per_entry
        self.grid_trainer_total_affordable_xp_half_point = 20.0
        self.grid_merchant_initial_total_stock_by_site_name: Dict[str, int] = {}
        self.grid_merchant_initial_stock_by_site_item: Dict[Tuple[str, str], int] = {}
        self.grid_spell_shop_initial_total_stock_by_site_name: Dict[str, int] = {}
        self.grid_spell_shop_initial_stock_by_site_spell: Dict[Tuple[str, str], int] = {}
        self.grid_mercenary_initial_total_stock_by_site_name: Dict[str, int] = {}
        self.grid_mercenary_initial_stock_by_site_slot: Dict[Tuple[str, int], int] = {}
        self.grid_max_mercenary_price = 1.0
        self.grid_max_mercenary_unit_health = 1.0
        self.grid_max_mercenary_unit_damage = 1.0
        for site_name in self.grid_merchant_site_names:
            total_site_stock = 0
            initial_site_stock = self._create_initial_merchant_stocks().get(site_name, {})
            for item_name in self.grid_merchant_item_names:
                try:
                    initial_stock = max(
                        0,
                        int(initial_site_stock.get(item_name, 0) or 0),
                    )
                except (TypeError, ValueError):
                    initial_stock = 0
                total_site_stock += int(initial_stock)
                self.grid_merchant_initial_stock_by_site_item[(site_name, item_name)] = int(
                    initial_stock
                )
            self.grid_merchant_initial_total_stock_by_site_name[site_name] = int(total_site_stock)
        for site_name in self.grid_spell_shop_site_names:
            total_site_stock = 0
            initial_site_stock = self._create_initial_spell_shop_stocks().get(site_name, {})
            for spell_name in self.grid_spell_shop_spell_names:
                try:
                    initial_stock = max(
                        0,
                        int(initial_site_stock.get(spell_name, 0) or 0),
                    )
                except (TypeError, ValueError):
                    initial_stock = 0
                total_site_stock += int(initial_stock)
                self.grid_spell_shop_initial_stock_by_site_spell[(site_name, spell_name)] = int(
                    initial_stock
                )
            self.grid_spell_shop_initial_total_stock_by_site_name[site_name] = int(total_site_stock)
        for site_name in self.grid_mercenary_site_names:
            total_site_stock = 0
            for entry in self.mercenary_site_rosters.get(site_name, []):
                slot_raw = entry.get("slot", 0)
                try:
                    slot = int(slot_raw if slot_raw is not None else 0)
                except (TypeError, ValueError):
                    slot = 0
                initial_stock_raw = entry.get("initial_stock", entry.get("stock", 0))
                try:
                    initial_stock = max(0, int(initial_stock_raw or 0))
                except (TypeError, ValueError):
                    initial_stock = 0
                total_site_stock += int(initial_stock)
                self.grid_mercenary_initial_stock_by_site_slot[(site_name, slot)] = int(
                    initial_stock
                )
                try:
                    self.grid_max_mercenary_price = max(
                        float(self.grid_max_mercenary_price),
                        float(entry.get("gold", 0.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    self.grid_max_mercenary_unit_health = max(
                        float(self.grid_max_mercenary_unit_health),
                        float(entry.get("unit_max_health", 0.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    self.grid_max_mercenary_unit_damage = max(
                        float(self.grid_max_mercenary_unit_damage),
                        float(entry.get("unit_damage", 0.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    pass
            self.grid_mercenary_initial_total_stock_by_site_name[site_name] = int(
                total_site_stock
            )

        merchant_small_heal_stock = int(
            sum(
                int(item_data.get("stock", 0) or 0)
                for item_data in self.MERCHANT_BUY_ITEMS
                if str(item_data.get("grant", "") or "") == "small_heal_bonus"
            )
        )
        merchant_large_heal_stock = int(
            sum(
                int(item_data.get("stock", 0) or 0)
                for item_data in self.MERCHANT_BUY_ITEMS
                if str(item_data.get("grant", "") or "") == "large_heal_bonus"
            )
        )
        merchant_revive_stock = int(
            sum(
                int(item_data.get("stock", 0) or 0)
                for item_data in self.MERCHANT_BUY_ITEMS
                if str(item_data.get("grant", "") or "") == "revive_bonus"
            )
        )
        self.grid_max_heal_bottle_capacity = int(self.MAX_HEAL_BOTTLES) + merchant_large_heal_stock
        self.grid_max_healing_bottle_capacity = int(self.MAX_HEALING_BOTTLES) + merchant_small_heal_stock + sum(
            1
            for chest_items in self._static_chests.values()
            for item_name in chest_items
            if str(item_name or "") == str(self.BONUS_SMALL_HEAL_SOURCE_ITEM)
        )
        self.grid_max_revive_bottle_capacity = int(self.MAX_REVIVE_BOTTLES) + merchant_revive_stock + sum(
            1
            for chest_items in self._static_chests.values()
            for item_name in chest_items
            if str(item_name or "") == str(self.BONUS_REVIVE_SOURCE_ITEM)
        )
        self.grid_max_invulnerability_potions = sum(
            1
            for chest_items in self._static_chests.values()
            for item_name in chest_items
            if str(item_name or "") == str(self.INVULNERABILITY_POTION_ITEM_NAME)
        )
        self.grid_max_strength_potions = sum(
            1
            for chest_items in self._static_chests.values()
            for item_name in chest_items
            if str(item_name or "") == str(self.STRENGTH_POTION_ITEM_NAME)
        )
        self.grid_max_build_price = 1.0
        for build_key in self.building_keys:
            building = self.active_buildings.get(build_key)
            if not isinstance(building, dict):
                continue
            try:
                price = float(building.get("gold", 0) or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            self.grid_max_build_price = max(self.grid_max_build_price, price)

        self.grid_campaign_obs_size = (
            self.grid_blue_obs_size
            + self.grid_resource_obs_size
            + self.grid_mana_source_obs_size
            + self.grid_building_obs_size
            + self.grid_spell_obs_size
            + self.grid_chest_obs_size
            + self.grid_heal_tile_obs_size
            + self.grid_legions_territory_obs_size
            + self.grid_ruin_obs_size
            + self.grid_objective_city_obs_size
            + self.grid_merchant_site_obs_size
            + self.grid_current_merchant_stock_obs_size
            + self.grid_spell_shop_site_obs_size
            + self.grid_current_spell_shop_stock_obs_size
            + self.grid_mercenary_site_obs_size
            + self.grid_mercenary_roster_obs_size
            + self.grid_trainer_obs_size
        )
    def _encode_enemy_unit_type(self, unit_type: Optional[str]) -> List[float]:
        if self.grid_enemy_unit_type_feature_size <= 0:
            return []
        vec = [0.0] * self.grid_enemy_unit_type_feature_size
        idx = self.enemy_unit_type_to_index.get(unit_type or "")
        if idx is not None:
            vec[idx] = 1.0
        return vec
    @staticmethod
    def _normalize_unit_health(unit: Dict) -> float:
        hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0)
        max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or hp)
        if max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, hp / max_hp))
    @staticmethod
    def _normalize_ratio(value: object, max_value: object) -> float:
        try:
            value_f = float(value or 0.0)
            max_f = float(max_value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if max_f <= 0.0:
            return 0.0
        return float(np.clip(value_f / max_f, 0.0, 1.0))
    @staticmethod
    def _normalize_saturating_count(value: object, half_point: float = 1.0) -> float:
        try:
            value_f = max(0.0, float(value or 0.0))
        except (TypeError, ValueError):
            return 0.0
        k = max(1e-6, float(half_point))
        return float(np.clip(value_f / (value_f + k), 0.0, 1.0))
    def _build_enemy_grid_obs(self) -> np.ndarray:
        """Строит признаки врагов на карте для grid-наблюдения."""
        if self.grid_enemy_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        enemy_obs: List[float] = []
        zero_enemy_slot = [0.0] * self.grid_enemy_unit_feature_size

        for enemy_id in sorted(self.grid_env.enemy_positions.keys()):
            ex, ey = self.grid_env.enemy_positions.get(enemy_id, (0, 0))
            enemy_obs.extend([ex / grid_scale, ey / grid_scale])

            is_alive = self.grid_env.enemies_alive.get(enemy_id, False)
            team = self._get_enemy_team_state(enemy_id)
            team_sorted = sorted(team, key=lambda u: int(u.get("position", 0)))

            for idx in range(self.grid_enemy_unit_slots):
                if not is_alive or idx >= len(team_sorted):
                    enemy_obs.extend(zero_enemy_slot)
                    continue

                unit = team_sorted[idx]
                if self._is_empty_enemy_unit(unit):
                    enemy_obs.extend(zero_enemy_slot)
                    continue

                type_one_hot = self._encode_enemy_unit_type(unit.get("unit_type"))
                hp_norm = self._normalize_unit_health(unit)
                enemy_obs.extend([*type_one_hot, hp_norm])

        return np.array(enemy_obs, dtype=np.float32)
    def _build_blue_team_grid_obs(self) -> np.ndarray:
        state_by_pos = {
            int(unit.get("position", -1) or -1): unit for unit in (self._get_blue_state() or [])
        }
        blue_obs: List[float] = []

        for position in self.grid_blue_positions:
            unit = state_by_pos.get(int(position), {})
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0.0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
            alive_v = 1.0 if max_hp > 0.0 and hp > 0.0 else 0.0
            hp_ratio = self._normalize_unit_health(unit) if unit else 0.0
            wounded_v = 1.0 if max_hp > 0.0 and hp > 0.0 and hp < max_hp else 0.0
            revivable_v = 1.0 if max_hp > 0.0 and hp <= 0.0 else 0.0
            invulnerability_v = 1.0 if int(position) in self.active_invulnerability_potion_positions else 0.0
            strength_v = 1.0 if int(position) in self.active_strength_potion_positions else 0.0
            hero_v = 1.0 if unit and self._is_hero_unit(unit) else 0.0
            level_norm = self._normalize_saturating_count(unit.get("Level", 0) if unit else 0.0, half_point=3.0)

            blue_obs.extend(
                [
                    alive_v,
                    hp_ratio,
                    wounded_v,
                    revivable_v,
                    invulnerability_v,
                    strength_v,
                    hero_v,
                    level_norm,
                ]
            )

        return np.array(blue_obs, dtype=np.float32)
    def _build_resource_grid_obs(self) -> np.ndarray:
        inventory = self.get_bottle_inventory_counters()
        build_locked_flag = self.active_buildings.get("alredybuilt", 0)
        try:
            build_locked = 1.0 if int(build_locked_flag or 0) == 1 else 0.0
        except (TypeError, ValueError):
            build_locked = 0.0

        resource_obs = [
            self._normalize_ratio(self.moves, max(1, self.moves_per_turn)),
            self._normalize_saturating_count(self.moves_per_turn, half_point=float(self.MOVES_PER_TURN)),
            1.0 if self._is_at_castle() else 0.0,
            build_locked,
            self._normalize_ratio(
                inventory.get("heal_left", 0),
                max(1, int(self.grid_max_heal_bottle_capacity)),
            ),
            self._normalize_ratio(
                inventory.get("healing_left", 0),
                max(1, int(self.grid_max_healing_bottle_capacity)),
            ),
            self._normalize_ratio(
                inventory.get("revive_left", 0),
                max(1, int(self.grid_max_revive_bottle_capacity)),
            ),
            self._normalize_ratio(
                inventory.get("invulnerability_left", 0),
                max(1, int(self.grid_max_invulnerability_potions)),
            ),
            self._normalize_ratio(
                inventory.get("strength_left", 0),
                max(1, int(self.grid_max_strength_potions)),
            ),
            self._normalize_ratio(len(self.chests), max(1, len(self.grid_chest_positions))),
            self._normalize_ratio(
                len(self.captured_objective_cities),
                max(1, len(self.grid_objective_city_names)),
            ),
        ]
        for slot_index in range(int(self.BATTLE_EQUIP_SLOTS)):
            equipped_name = ""
            if slot_index < len(self.equipped_hero_items):
                equipped_name = str(self.equipped_hero_items[slot_index] or "")
            resource_obs.append(1.0 if not equipped_name else 0.0)
            for item_name in self.grid_battle_equipment_item_names:
                resource_obs.append(1.0 if equipped_name == item_name else 0.0)
        for mana_kind in self.grid_mana_total_kinds:
            mana_attr = str(self.MANA_ATTR_BY_KIND.get(str(mana_kind), "") or "")
            mana_value = float(getattr(self, mana_attr, 0.0) or 0.0) if mana_attr else 0.0
            resource_obs.append(
                self._normalize_saturating_count(
                    mana_value,
                    half_point=float(
                        self.grid_mana_total_half_point_by_kind.get(str(mana_kind), 1.0)
                    ),
                )
            )
        return np.array(resource_obs, dtype=np.float32)
    def _build_mana_source_grid_obs(self) -> np.ndarray:
        if self.grid_mana_source_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        mana_obs: List[float] = []
        for entry in self.grid_mana_source_entries:
            tile_pos = tuple(entry.get("pos", (0, 0)))
            mana_kind = str(entry.get("kind", "") or "")
            x = int(tile_pos[0]) if len(tile_pos) >= 1 else 0
            y = int(tile_pos[1]) if len(tile_pos) >= 2 else 0
            type_norm = float(self.grid_mana_kind_to_norm.get(mana_kind, 0.0))
            mana_obs.extend(
                [
                    x / grid_scale,
                    y / grid_scale,
                    type_norm,
                ]
            )
        return np.array(mana_obs, dtype=np.float32)
    def _build_building_grid_obs(self) -> np.ndarray:
        building_obs: List[float] = []
        max_build_price = self._current_max_building_gold_cost()
        for build_key in self.building_keys:
            building = self.active_buildings.get(build_key)
            if not isinstance(building, dict):
                building_obs.extend([0.0, 0.0, 0.0])
                continue

            built_flag = building.get("Build", building.get("built", 0))
            blocked_flag = building.get("blocked", 0)
            try:
                built_v = 1.0 if int(built_flag or 0) == 1 else 0.0
            except (TypeError, ValueError):
                built_v = 0.0
            try:
                blocked_v = 1.0 if int(blocked_flag or 0) == 1 else 0.0
            except (TypeError, ValueError):
                blocked_v = 0.0
            price_v = self._get_building_gold_cost(building)

            building_obs.extend(
                [
                    built_v,
                    blocked_v,
                    self._normalize_ratio(price_v, max_build_price),
                ]
            )

        return np.array(building_obs, dtype=np.float32)
    def _build_spell_grid_obs(self) -> np.ndarray:
        self._ensure_inventory_cache()
        spell_obs: List[float] = [
            1.0 if self._has_magic_tower_built() else 0.0,
            1.0 if self.spell_learning_locked else 0.0,
            self._normalize_ratio(
                len(self.get_learned_spell_descriptions()),
                max(1, len(self.spell_keys)),
            ),
        ]
        for spell_key in self.spell_keys:
            spell = self.active_spells.get(spell_key)
            if not isinstance(spell, dict):
                spell_obs.append(0.0)
                continue
            try:
                learned_v = 1.0 if int(spell.get("learned", 0) or 0) == 1 else 0.0
            except (TypeError, ValueError):
                learned_v = 0.0
            spell_obs.append(learned_v)

        current_specs = self._current_map_offensive_spell_action_specs()
        for idx in range(self.grid_map_offensive_spell_action_count):
            spell_key = ""
            if idx < len(current_specs):
                spell_key = str(current_specs[idx].get("id", "") or "")
            spell_obs.append(
                1.0 if spell_key and self._spell_cast_limit_reached_this_turn(spell_key) else 0.0
            )

        nearest_enemy = self._get_nearest_enemy_stack_for_mask(spell_targetable_only=True)
        nearest_enemy_summary = self._enemy_stack_spell_effect_summary(
            None if nearest_enemy is None else nearest_enemy.get("enemy_id")
        )
        nearest_enemy_effect_types = set(
            str(effect_type)
            for effect_type in nearest_enemy_summary.get("debuff_types", [])
            if str(effect_type)
        )
        spell_obs.extend(
            [
                1.0 if nearest_enemy_effect_types else 0.0,
                1.0 if "armor" in nearest_enemy_effect_types else 0.0,
                1.0 if "damage" in nearest_enemy_effect_types else 0.0,
                1.0 if "initiative" in nearest_enemy_effect_types else 0.0,
                1.0 if "accuracy" in nearest_enemy_effect_types else 0.0,
            ]
        )
        kind_index = {
            "damage": 1,
            "debuff": 2,
            "summon_battle": 3,
            "moves": 4,
            "heal": 5,
            "buff": 6,
            "ward": 7,
            "health_bonus": 8,
        }
        scroll_entries = self._scroll_cast_slot_entries_cache
        available_scroll_entry_count = len(self._available_scroll_spell_entries_cache)
        total_scroll_count = int(self._total_scroll_count_cache)
        spell_obs.extend(
            [
                1.0 if self.scroll_magic_unlocked else 0.0,
                self._normalize_saturating_count(
                    available_scroll_entry_count,
                    half_point=float(max(1, self.MAX_SCROLL_CAST_ACTIONS)),
                ),
                self._normalize_saturating_count(
                    total_scroll_count,
                    half_point=float(max(1, self.MAX_SCROLL_CAST_ACTIONS)),
                ),
            ]
        )
        for idx in range(int(self.MAX_SCROLL_CAST_ACTIONS)):
            scroll_entry: Dict[str, object] = {}
            if idx < len(scroll_entries):
                scroll_entry = dict(scroll_entries[idx])
            spell_kind = str(scroll_entry.get("spell_kind", "") or "")
            try:
                spell_level = int(scroll_entry.get("spell_level", 0) or 0)
            except (TypeError, ValueError):
                spell_level = 0
            spell_obs.extend(
                [
                    1.0 if scroll_entry else 0.0,
                    self._normalize_saturating_count(
                        int(scroll_entry.get("copies", 0) or 0),
                        half_point=2.0,
                    ),
                    self._normalize_ratio(kind_index.get(spell_kind, 0), max(1, len(kind_index))),
                    self._normalize_ratio(spell_level, 5.0),
                ]
            )
        return np.array(spell_obs, dtype=np.float32)
    def _build_chest_grid_obs(self) -> np.ndarray:
        if self.grid_chest_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        active_chests = {tuple(pos) for pos in self.chests.keys()}
        chest_obs: List[float] = []
        for chest_pos in self.grid_chest_positions:
            chest_obs.extend(
                [
                    chest_pos[0] / grid_scale,
                    chest_pos[1] / grid_scale,
                    1.0 if tuple(chest_pos) in active_chests else 0.0,
                ]
            )
        return np.array(chest_obs, dtype=np.float32)
    def _build_heal_tiles_grid_obs(self) -> np.ndarray:
        if self.grid_heal_tile_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        heal_obs: List[float] = []
        for tile_pos in self.grid_heal_tile_positions:
            heal_obs.extend([tile_pos[0] / grid_scale, tile_pos[1] / grid_scale])
        return np.array(heal_obs, dtype=np.float32)
    def _build_legions_territory_grid_obs(self) -> np.ndarray:
        if self.grid_legions_territory_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)
        return self._legions_territory_grid_obs_cache
    def _build_empire_territory_grid_obs(self) -> np.ndarray:
        if self.grid_empire_territory_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        territory_tiles = self.empire_territory_tile_set
        territory_obs = np.fromiter(
            (
                1.0 if tuple(tile_pos) in territory_tiles else 0.0
                for tile_pos in self.grid_empire_territory_positions
            ),
            dtype=np.float32,
            count=self.grid_empire_territory_obs_size,
        )
        return territory_obs
    def _build_ruin_grid_obs(self) -> np.ndarray:
        if self.grid_ruin_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        ruin_obs: List[float] = []
        for ruin_pos in self.grid_ruin_positions:
            ruin_obs.extend([ruin_pos[0] / grid_scale, ruin_pos[1] / grid_scale])
        return np.array(ruin_obs, dtype=np.float32)
    def _build_objective_city_grid_obs(self) -> np.ndarray:
        if self.grid_objective_city_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        objective_obs: List[float] = []
        enemy_positions = getattr(self.grid_env, "enemy_positions", {}) or {}
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}

        for city_name in self.grid_objective_city_names:
            city_enemy_ids = tuple(
                int(enemy_id)
                for enemy_id in self.FINAL_OBJECTIVE_CITIES.get(city_name, ())
            )
            city_positions = [
                tuple(enemy_positions[enemy_id])
                for enemy_id in city_enemy_ids
                if enemy_id in enemy_positions
            ]
            if city_positions:
                centroid_x = sum(pos[0] for pos in city_positions) / float(len(city_positions))
                centroid_y = sum(pos[1] for pos in city_positions) / float(len(city_positions))
            else:
                centroid_x = 0.0
                centroid_y = 0.0

            captured_v = 1.0 if city_name in self.captured_objective_cities else 0.0
            enemies_remaining = sum(
                1.0 for enemy_id in city_enemy_ids if bool(enemies_alive.get(enemy_id, False))
            )
            remaining_ratio = self._normalize_ratio(
                enemies_remaining,
                max(1, len(city_enemy_ids)),
            )

            objective_obs.extend(
                [
                    centroid_x / grid_scale,
                    centroid_y / grid_scale,
                    captured_v,
                    remaining_ratio,
                ]
            )

        return np.array(objective_obs, dtype=np.float32)
    def _build_merchant_site_grid_obs(self) -> np.ndarray:
        if self.grid_merchant_site_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        current_pos = tuple(int(coord) for coord in tuple(self.grid_env.agent_pos)[:2])
        site_obs: List[float] = []

        for site_name in self.grid_merchant_site_names:
            anchor = self.merchant_site_anchors.get(site_name, (0, 0))
            total_stock = sum(
                max(0, int(stock or 0))
                for stock in self.merchant_stocks.get(site_name, {}).values()
            )
            max_total_stock = int(
                self.grid_merchant_initial_total_stock_by_site_name.get(site_name, 0) or 0
            )
            site_obs.extend(
                [
                    int(anchor[0]) / grid_scale,
                    int(anchor[1]) / grid_scale,
                    1.0
                    if current_pos in self.merchant_site_interaction_tiles.get(site_name, ())
                    else 0.0,
                    1.0 if total_stock > 0 else 0.0,
                    (
                        self._normalize_ratio(total_stock, max_total_stock)
                        if max_total_stock > 0
                        else 0.0
                    ),
                ]
            )

        return np.array(site_obs, dtype=np.float32)
    def _build_current_merchant_stock_grid_obs(self) -> np.ndarray:
        if self.grid_current_merchant_stock_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        site_names = self._merchant_sites_at_position(self.grid_env.agent_pos)
        active_site_name = str(site_names[0]) if site_names else ""
        stock_obs: List[float] = []

        for item_name in self.grid_merchant_item_names:
            current_stock = self._merchant_item_stock(active_site_name, item_name) if active_site_name else 0
            initial_stock = int(
                self.grid_merchant_initial_stock_by_site_item.get(
                    (active_site_name, item_name),
                    0,
                )
                or 0
            )
            stock_obs.append(
                self._normalize_ratio(current_stock, initial_stock) if initial_stock > 0 else 0.0
            )

        return np.array(stock_obs, dtype=np.float32)
    def _build_spell_shop_site_grid_obs(self) -> np.ndarray:
        if self.grid_spell_shop_site_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        current_pos = tuple(int(coord) for coord in tuple(self.grid_env.agent_pos)[:2])
        site_obs: List[float] = []

        for site_name in self.grid_spell_shop_site_names:
            anchor = self.spell_shop_site_anchors.get(site_name, (0, 0))
            total_stock = sum(
                max(0, int(stock or 0))
                for stock in self.spell_shop_stocks.get(site_name, {}).values()
            )
            max_total_stock = int(
                self.grid_spell_shop_initial_total_stock_by_site_name.get(site_name, 0) or 0
            )
            site_obs.extend(
                [
                    int(anchor[0]) / grid_scale,
                    int(anchor[1]) / grid_scale,
                    1.0
                    if current_pos in self.spell_shop_site_interaction_tiles.get(site_name, ())
                    else 0.0,
                    1.0 if total_stock > 0 else 0.0,
                    (
                        self._normalize_ratio(total_stock, max_total_stock)
                        if max_total_stock > 0
                        else 0.0
                    ),
                ]
            )

        return np.array(site_obs, dtype=np.float32)
    def _build_current_spell_shop_stock_grid_obs(self) -> np.ndarray:
        if self.grid_current_spell_shop_stock_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        site_names = self._spell_shop_sites_at_position(self.grid_env.agent_pos)
        active_site_name = str(site_names[0]) if site_names else ""
        stock_obs: List[float] = []

        for spell_name in self.grid_spell_shop_spell_names:
            current_stock = (
                self._spell_shop_spell_stock(active_site_name, spell_name) if active_site_name else 0
            )
            initial_stock = int(
                self.grid_spell_shop_initial_stock_by_site_spell.get(
                    (active_site_name, spell_name),
                    0,
                )
                or 0
            )
            stock_obs.append(
                self._normalize_ratio(current_stock, initial_stock) if initial_stock > 0 else 0.0
            )

        return np.array(stock_obs, dtype=np.float32)
    def _build_mercenary_site_grid_obs(self) -> np.ndarray:
        if self.grid_mercenary_site_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        current_pos = tuple(int(coord) for coord in tuple(self.grid_env.agent_pos)[:2])
        site_obs: List[float] = []

        for site_name in self.grid_mercenary_site_names:
            anchor = self.mercenary_site_anchors.get(site_name, (0, 0))
            total_stock = sum(
                max(0, int(entry.get("stock", 0) or 0))
                for entry in self.mercenary_site_rosters.get(site_name, [])
            )
            max_total_stock = int(
                self.grid_mercenary_initial_total_stock_by_site_name.get(site_name, 0) or 0
            )
            site_obs.extend(
                [
                    int(anchor[0]) / grid_scale,
                    int(anchor[1]) / grid_scale,
                    1.0
                    if current_pos in self.mercenary_site_interaction_tiles.get(site_name, ())
                    else 0.0,
                    1.0 if total_stock > 0 else 0.0,
                    (
                        self._normalize_ratio(total_stock, max_total_stock)
                        if max_total_stock > 0
                        else 0.0
                    ),
                ]
            )

        return np.array(site_obs, dtype=np.float32)
    def _build_current_mercenary_roster_grid_obs(self) -> np.ndarray:
        if self.grid_mercenary_roster_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        site_names = self._mercenary_sites_at_position(self.grid_env.agent_pos)
        active_site_name = str(site_names[0]) if site_names else ""
        active_entries = self.mercenary_site_rosters.get(active_site_name, []) if active_site_name else []
        entries_by_slot: Dict[int, Dict[str, object]] = {}
        option_index_by_slot: Dict[int, int] = {}

        for entry in active_entries:
            slot_raw = entry.get("slot", 0)
            try:
                slot = int(slot_raw if slot_raw is not None else 0)
            except (TypeError, ValueError):
                continue
            entries_by_slot[slot] = entry

        if active_site_name:
            for idx, option in enumerate(self.active_mercenary_hire_options):
                if str(option.get("site_name", "") or "") != active_site_name:
                    continue
                slot_raw = option.get("slot", 0)
                try:
                    slot = int(slot_raw if slot_raw is not None else 0)
                except (TypeError, ValueError):
                    continue
                option_index_by_slot[slot] = int(idx)

        roster_obs: List[float] = []
        for slot in range(int(self.grid_mercenary_slot_count)):
            entry = entries_by_slot.get(int(slot))
            if not isinstance(entry, dict):
                roster_obs.extend([0.0] * self.grid_mercenary_slot_features_per_entry)
                continue

            stock = max(0, int(entry.get("stock", 0) or 0))
            initial_stock = max(
                0,
                int(
                    self.grid_mercenary_initial_stock_by_site_slot.get(
                        (active_site_name, int(slot)),
                        entry.get("initial_stock", stock),
                    )
                    or 0
                ),
            )
            stand = str(entry.get("stand", "") or "").strip().lower()
            option_idx = option_index_by_slot.get(int(slot))
            hireable_now = 0.0
            if option_idx is not None:
                hireable_now = 1.0 if self._can_hire_mercenary_unit_option(self._mercenary_option(option_idx)) else 0.0

            roster_obs.extend(
                [
                    1.0,
                    self._normalize_ratio(stock, initial_stock) if initial_stock > 0 else 0.0,
                    1.0 if stand == "ahead" else 0.0,
                    1.0 if stand == "behind" else 0.0,
                    self._normalize_ratio(
                        float(entry.get("gold", 0.0) or 0.0),
                        self.grid_max_mercenary_price,
                    ),
                    self._normalize_ratio(
                        float(entry.get("unit_max_health", 0.0) or 0.0),
                        self.grid_max_mercenary_unit_health,
                    ),
                    self._normalize_ratio(
                        float(entry.get("unit_damage", 0.0) or 0.0),
                        self.grid_max_mercenary_unit_damage,
                    ),
                    float(hireable_now),
                ]
            )

        return np.array(roster_obs, dtype=np.float32)
    def _build_trainer_grid_obs(self) -> np.ndarray:
        if self.grid_trainer_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        current_pos = tuple(int(coord) for coord in tuple(self.grid_env.agent_pos)[:2])
        grid_scale = max(1, self.grid_env.grid_size - 1)
        trainer_tiles = tuple(getattr(self, "trainer_interaction_tiles", ()) or ())

        dx_norm = 0.0
        dy_norm = 0.0
        if trainer_tiles:
            nearest_tile = min(
                trainer_tiles,
                key=lambda tile: (
                    abs(int(tile[0]) - int(current_pos[0])) + abs(int(tile[1]) - int(current_pos[1])),
                    int(tile[1]),
                    int(tile[0]),
                ),
            )
            dx = int(nearest_tile[0]) - int(current_pos[0])
            dy = int(nearest_tile[1]) - int(current_pos[1])
            dx_norm = float(np.clip((float(dx) / float(grid_scale) + 1.0) * 0.5, 0.0, 1.0))
            dy_norm = float(np.clip((float(dy) / float(grid_scale) + 1.0) * 0.5, 0.0, 1.0))

        previews = [self._trainer_preview_for_position(pos) for pos in self.TRAINER_POSITIONS]
        trainable_slots = sum(1 for preview in previews if bool(preview.get("trainable", False)))
        total_affordable_xp = sum(
            max(0, int(preview.get("affordable_xp", 0) or 0))
            for preview in previews
        )
        trainer_obs = [
            float(dx_norm),
            float(dy_norm),
            1.0 if self._trainer_sites_at_position(self.grid_env.agent_pos) else 0.0,
            self._normalize_ratio(trainable_slots, max(1, len(self.TRAINER_POSITIONS))),
            self._normalize_saturating_count(
                total_affordable_xp,
                half_point=float(self.grid_trainer_total_affordable_xp_half_point),
            ),
        ]
        return np.array(trainer_obs, dtype=np.float32)
    def _build_campaign_grid_obs(self) -> np.ndarray:
        campaign_obs = np.concatenate(
            [
                self._build_blue_team_grid_obs(),
                self._build_resource_grid_obs(),
                self._build_mana_source_grid_obs(),
                self._build_building_grid_obs(),
                self._build_spell_grid_obs(),
                self._build_chest_grid_obs(),
                self._build_heal_tiles_grid_obs(),
                self._build_legions_territory_grid_obs(),
                self._build_ruin_grid_obs(),
                self._build_objective_city_grid_obs(),
                self._build_merchant_site_grid_obs(),
                self._build_current_merchant_stock_grid_obs(),
                self._build_spell_shop_site_grid_obs(),
                self._build_current_spell_shop_stock_grid_obs(),
                self._build_mercenary_site_grid_obs(),
                self._build_current_mercenary_roster_grid_obs(),
                self._build_trainer_grid_obs(),
            ]
        )
        return campaign_obs.astype(np.float32, copy=False)
    def _augment_grid_obs(self, base_obs: np.ndarray) -> np.ndarray:
        enemy_obs = self._build_enemy_grid_obs()
        campaign_obs = self._build_campaign_grid_obs()
        if enemy_obs.size == 0 and campaign_obs.size == 0:
            return base_obs.astype(np.float32, copy=False)
        return np.concatenate(
            [
                base_obs.astype(np.float32, copy=False),
                enemy_obs,
                campaign_obs,
            ]
        )
    def _get_grid_obs(self) -> np.ndarray:
        """Возвращает grid-наблюдение с признаками врагов."""
        return self._augment_grid_obs(self.grid_env._get_obs())
    def _build_obs(
        self,
        grid_obs: Optional[np.ndarray] = None,
        battle_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Строит объединённое наблюдение."""
        mode_flag = np.array([float(self.mode)], dtype=np.float32)

        if grid_obs is None:
            grid_obs = np.zeros(self.GRID_OBS_SIZE, dtype=np.float32)

        if battle_obs is None:
            battle_obs = np.zeros(self.BATTLE_OBS_SIZE, dtype=np.float32)

        turns_raw = max(0.0, float(self.turns))
        gold_raw = max(0.0, float(self.gold))
        turns_norm = self._clip01(turns_raw / (turns_raw + self.turn_norm_k))
        gold_norm = self._clip01(gold_raw / (gold_raw + self.gold_norm_k))
        extra_obs = np.array([turns_norm, gold_norm], dtype=np.float32)

        obs = np.concatenate([mode_flag, grid_obs, battle_obs, extra_obs])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        obs = np.clip(obs, 0.0, 1.0)
        return obs
