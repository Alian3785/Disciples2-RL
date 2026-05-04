# campaign_env_masks.py
"""CampaignMaskMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignMaskMixin:
    def action_masks(self):
        """SB3-contrib MaskablePPO hook for invalid action masking."""
        return self.compute_action_mask()

    def compute_action_mask(self) -> np.ndarray:
        """Маска действий в зависимости от режима."""
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.mode == self.MODE_GRID:
            self._ensure_inventory_cache()
            # В grid режиме движение доступно только при moves > 0.
            # REST (8) доступен при ранении или когда очки перемещения закончились.
            grid_mask = self.grid_env.compute_action_mask()
            grid_move_cost = max(1, int(self.GRID_MOVE_COST))
            mask[:8] = bool(self.moves >= grid_move_cost) & grid_mask[:8]
            mask[8] = bool(grid_mask[8]) and (
                self._has_wounded_blue() or self.moves < grid_move_cost
            )
            # Общее действие лечения банками на позиции 7-12.
            if self._has_any_heal_bottle_left():
                for idx, pos in enumerate(self.GRID_BOTTLE_POSITIONS):
                    mask[self.GRID_BOTTLE_ACTION_START + idx] = self._can_heal_position(pos)
            # Бутыли воскрешения на позиции 7-12.
            if self.revive_bottles_used < self._max_revive_bottles_available():
                for idx, pos in enumerate(self.REVIVE_BOTTLE_POSITIONS):
                    mask[self.GRID_REVIVE_ACTION_START + idx] = self._can_revive_position(pos)
            if self._hero_item_available(self.INVULNERABILITY_POTION_ITEM_NAME):
                for idx, pos in enumerate(self.INVULNERABILITY_POTION_POSITIONS):
                    mask[self.GRID_INVULNERABILITY_ACTION_START + idx] = (
                        self._can_apply_invulnerability_potion_position(pos)
                    )
            if self._hero_item_available(self.STRENGTH_POTION_ITEM_NAME):
                for idx, pos in enumerate(self.STRENGTH_POTION_POSITIONS):
                    mask[self.GRID_STRENGTH_ACTION_START + idx] = (
                        self._can_apply_strength_potion_position(pos)
                    )
            if self._merchant_sites_at_position(self.grid_env.agent_pos):
                for idx, item_data in enumerate(self.MERCHANT_BUY_ITEMS):
                    item_name = str(item_data.get("name", "") or "")
                    item_price = float(item_data.get("price", 0.0) or 0.0)
                    mask[self.GRID_MERCHANT_BUY_ACTION_START + idx] = (
                        bool(item_name)
                        and float(self.gold or 0.0) >= item_price
                        and self._merchant_site_for_item(
                            item_name,
                            position=self.grid_env.agent_pos,
                        )
                        is not None
                    )
            if self._spell_shop_sites_at_position(self.grid_env.agent_pos):
                for idx, spell_data in enumerate(self.SPELL_SHOP_BUY_SPELLS):
                    spell_name = str(spell_data.get("name", "") or "")
                    spell_id = str(spell_data.get("spell_id", "") or "")
                    spell_price = float(spell_data.get("price", 0.0) or 0.0)
                    mask[self.GRID_SPELL_SHOP_BUY_ACTION_START + idx] = (
                        bool(spell_name)
                        and float(self.gold or 0.0) >= spell_price
                        and not self._spell_shop_spell_owned(spell_id)
                        and self._spell_shop_site_for_spell(
                            spell_name,
                            position=self.grid_env.agent_pos,
                        )
                        is not None
                    )
            if self._hero_item_available(self.ENERGY_ELIXIR_ITEM_NAME):
                for idx, pos in enumerate(self.ENERGY_ELIXIR_POSITIONS):
                    mask[self.GRID_ENERGY_ACTION_START + idx] = (
                        self._can_apply_energy_elixir_position(pos)
                    )
            if self._hero_item_available(self.HASTE_ELIXIR_ITEM_NAME):
                for idx, pos in enumerate(self.HASTE_ELIXIR_POSITIONS):
                    mask[self.GRID_HASTE_ACTION_START + idx] = (
                        self._can_apply_haste_elixir_position(pos)
                    )
            if self._hero_item_available(self.FIRE_WARD_ITEM_NAME):
                for idx, pos in enumerate(self.FIRE_WARD_POSITIONS):
                    mask[self.GRID_FIRE_WARD_ACTION_START + idx] = (
                        self._can_apply_fire_ward_position(pos)
                    )
            if self._hero_item_available(self.EARTH_WARD_ITEM_NAME):
                for idx, pos in enumerate(self.EARTH_WARD_POSITIONS):
                    mask[self.GRID_EARTH_WARD_ACTION_START + idx] = (
                        self._can_apply_earth_ward_position(pos)
                    )
            if self._hero_item_available(self.WATER_WARD_ITEM_NAME):
                for idx, pos in enumerate(self.WATER_WARD_POSITIONS):
                    mask[self.GRID_WATER_WARD_ACTION_START + idx] = (
                        self._can_apply_water_ward_position(pos)
                    )
            if self._hero_item_available(self.AIR_WARD_ITEM_NAME):
                for idx, pos in enumerate(self.AIR_WARD_POSITIONS):
                    mask[self.GRID_AIR_WARD_ACTION_START + idx] = (
                        self._can_apply_air_ward_position(pos)
                    )
            if self._hero_item_available(self.TITAN_ELIXIR_ITEM_NAME):
                for idx, pos in enumerate(self.TITAN_ELIXIR_POSITIONS):
                    mask[self.GRID_TITAN_ELIXIR_ACTION_START + idx] = (
                        self._can_apply_titan_elixir_position(pos)
                    )
            if self._hero_item_available(self.SUPREME_ELIXIR_ITEM_NAME):
                for idx, pos in enumerate(self.SUPREME_ELIXIR_POSITIONS):
                    mask[self.GRID_SUPREME_ELIXIR_ACTION_START + idx] = (
                        self._can_apply_supreme_elixir_position(pos)
                    )

            # Лечение и воскрешение за золото доступны только на специальных клетках лечения.
            if self._is_at_castle() and float(self.gold) > 0.0:
                if self._has_temple_built():
                    for idx, pos in enumerate(self.CASTLE_HEAL_POSITIONS):
                        mask[self.GRID_CASTLE_HEAL_ACTION_START + idx] = self._can_heal_position(pos)
                    for idx, pos in enumerate(self.CASTLE_REVIVE_POSITIONS):
                        mask[self.GRID_CASTLE_REVIVE_ACTION_START + idx] = self._can_castle_revive_position(pos)
            for idx in range(len(self.active_hire_options)):
                mask[self.GRID_HIRE_ACTION_START + idx] = self._can_hire_faction_unit_option(
                    self._hire_option(idx)
                )
            for idx in range(len(self.active_mercenary_hire_options)):
                mask[self.GRID_MERCENARY_HIRE_ACTION_START + idx] = (
                    self._can_hire_mercenary_unit_option(self._mercenary_option(idx))
                )
            if self._trainer_sites_at_position(self.grid_env.agent_pos):
                for idx, pos in enumerate(self.TRAINER_POSITIONS):
                    mask[self.GRID_TRAINER_ACTION_START + idx] = self._can_train_unit_position(
                        pos
                    )

            # Постройки: доступны при достаточном золоте и если не blocked/built
            alredybuilt_flag = self.active_buildings.get("alredybuilt", 0)
            try:
                has_build_lock = int(alredybuilt_flag or 0) == 1
            except (TypeError, ValueError):
                has_build_lock = False
            for idx, build_key in enumerate(self.building_keys):
                building = self.active_buildings.get(build_key)
                if not isinstance(building, dict):
                    continue
                price = self._get_building_gold_cost(building)
                built_flag = building.get("Build", building.get("built", 0))
                is_built = int(built_flag or 0) == 1
                blocked_flag = building.get("blocked", 0)
                is_blocked = int(blocked_flag or 0) == 1
                mask[self.GRID_BUILD_ACTION_START + idx] = (
                    not has_build_lock and not is_built and not is_blocked and self.gold >= price
                )
            if self._has_magic_tower_built() and not self.spell_learning_locked:
                for idx, spell_key in enumerate(self.spell_keys):
                    spell = self.active_spells.get(spell_key)
                    if not isinstance(spell, dict):
                        continue
                    try:
                        spell_level = int(spell.get("level", 0) or 0)
                    except (TypeError, ValueError):
                        spell_level = 0
                    if self._is_spell_level_masked_by_typeoflord(spell_level):
                        continue
                    try:
                        is_learned = int(spell.get("learned", 0) or 0) == 1
                    except (TypeError, ValueError):
                        is_learned = False
                    if is_learned:
                        continue
                    spell_costs = self._get_spell_learning_costs(spell)
                    mask[self.grid_spell_action_start + idx] = self._has_mana_for_costs(
                        spell_costs
                    )
            for idx, settlement_name in enumerate(self.settlement_upgrade_names):
                mask[self.grid_settlement_upgrade_action_start + idx] = self._can_upgrade_settlement(
                    settlement_name
                )
            for idx, item_name in enumerate(self.BATTLE_EQUIPPABLE_ITEM_NAMES):
                mask[self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + idx] = self._can_equip_battle_item(
                    item_name,
                )
            nearest_targetable_enemy = self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=True
            )
            nearest_any_enemy = self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=False
            )
            current_specs = self._current_map_offensive_spell_action_specs()
            for idx in range(self._map_offensive_spell_action_max_count()):
                spell_key = ""
                nearest_enemy = None
                if idx < len(current_specs):
                    spell_key = str(current_specs[idx].get("id", "") or "")
                    nearest_enemy = (
                        nearest_any_enemy
                        if self._map_offensive_spell_targets_untargetable_stacks(spell_key)
                        else nearest_targetable_enemy
                    )
                mask[self.grid_legion_damage_spell_action_start + idx] = (
                    bool(spell_key)
                    and self._can_cast_legion_damage_spell(
                        spell_key,
                        nearest_enemy=nearest_enemy,
                    )
                )
            current_support_specs = self._current_map_support_spell_action_specs()
            for idx in range(self._map_support_spell_action_max_count()):
                spell_key = ""
                if idx < len(current_support_specs):
                    spell_key = str(current_support_specs[idx].get("id", "") or "")
                mask[self.grid_map_support_spell_action_start + idx] = (
                    bool(spell_key) and self._can_cast_support_spell(spell_key)
                )
            current_spell_shop_specs = self._spell_shop_cast_spell_entries()
            for idx in range(int(self.MAX_SPELL_SHOP_CAST_ACTIONS)):
                spell_key = ""
                if idx < len(current_spell_shop_specs):
                    spell_key = str(current_spell_shop_specs[idx].get("spell_id", "") or "")
                mask[self.grid_spell_shop_cast_action_start + idx] = (
                    bool(spell_key)
                    and self._can_cast_spell_shop_spell(
                        spell_key,
                        nearest_targetable_enemy=nearest_targetable_enemy,
                        nearest_any_enemy=nearest_any_enemy,
                    )
                )
            mask[self.GRID_UNLOCK_SCROLL_MAGIC_ACTION] = (
                not bool(self.scroll_magic_unlocked)
                and float(self.gold or 0.0) >= float(self.SCROLL_MAGE_UNLOCK_GOLD_COST)
                and bool(self._available_scroll_spell_entries_cache)
            )
            current_scroll_specs = self._scroll_cast_slot_entries_cache
            for idx in range(int(self.MAX_SCROLL_CAST_ACTIONS)):
                scroll_entry: Dict[str, object] = {}
                if idx < len(current_scroll_specs):
                    scroll_entry = dict(current_scroll_specs[idx])
                mask[self.grid_scroll_cast_action_start + idx] = (
                    bool(scroll_entry)
                    and self._can_cast_scroll_spell(
                        scroll_entry,
                        nearest_targetable_enemy=nearest_targetable_enemy,
                        nearest_any_enemy=nearest_any_enemy,
                    )
                )
        else:
            # В battle режиме используем маску из BattleEnv
            if self.battle_env is not None:
                battle_mask = self.battle_env.compute_action_mask()
                mask[: len(battle_mask)] = battle_mask
            else:
                # Фолбэк: все действия доступны
                mask[:] = True

        return mask
    def _has_wounded_blue(self) -> bool:
        """Проверяет, есть ли раненые юниты BLUE (hp < max_hp, но живы)."""
        state = self._get_blue_state()
        for unit in state:
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            if hp > 0 and hp < max_hp:
                return True
        return False
    def _can_revive_position(self, position: int) -> bool:
        """Можно ли применить бутыль воскрешения к позиции (юнит мёртв: hp <= 0, есть max_hp)."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return max_hp > 0 and hp <= 0
        return False
    def _can_apply_invulnerability_potion_position(self, position: int) -> bool:
        if not self._hero_item_available(self.INVULNERABILITY_POTION_ITEM_NAME):
            return False
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return (
                max_hp > 0
                and hp > 0
                and int(position) not in self.active_invulnerability_potion_positions
            )
        return False
    def _can_apply_strength_potion_position(self, position: int) -> bool:
        if not self._hero_item_available(self.STRENGTH_POTION_ITEM_NAME):
            return False
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return (
                max_hp > 0
                and hp > 0
                and int(position) not in self.active_strength_potion_positions
            )
        return False
    def _can_castle_revive_position(self, position: int) -> bool:
        """Можно ли воскресить юнита за золото на клетке лечения."""
        revive_cost = self._castle_revive_cost_at_position(position)
        return revive_cost > 0.0 and float(self.gold or 0.0) >= revive_cost
    def _revive_unit_at_position(self, position: int) -> Tuple[bool, Optional[str]]:
        """
        Воскрешает юнита на позиции, устанавливая 1 HP, если он мёртв.
        Возвращает флаг успеха и имя юнита.
        """
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)

            if max_hp <= 0 or hp > 0:
                return False, None

            unit["hp"] = 1.0
            unit["health"] = 1.0
            name = unit.get("name", f"pos_{position}")
            self._log(f"Бутыль воскрешения: {name} (pos {position}) оживлён с 1 HP")
            return True, name

        return False, None
    def _can_heal_position(self, position: int) -> bool:
        """Можно ли применить бутыль лечения к указанной позиции (юнит жив и не на фулл HP)."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            return max_hp > 0 and hp > 0 and hp < max_hp
        return False
    def get_nearest_enemy_stack(
        self,
        from_pos: Optional[Tuple[int, int]] = None,
        *,
        only_alive: bool = True,
        spell_targetable_only: bool = False,
    ) -> Optional[Dict[str, object]]:
        """
        Возвращает один ближайший вражеский отряд на карте.

        Правило выбора детерминированное:
        1. минимальная длина пути по текущей карте с учётом препятствий;
        2. при равенстве меньший enemy_id;
        3. затем верхняя-левая координата (y, x).

        Если путь ни к одному врагу недоступен, используется fallback по
        свободной от препятствий дистанции хода (Chebyshev) с тем же tie-break.
        """
        enemy_positions = getattr(self.grid_env, "enemy_positions", {}) or {}
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}
        if not enemy_positions:
            return None

        start_pos = tuple(from_pos) if from_pos is not None else tuple(self.grid_env.agent_pos)
        start_tile = (
            int(start_pos[0]),
            int(start_pos[1]),
        )
        blocked_tiles = {
            (int(pos[0]), int(pos[1]))
            for pos in (getattr(self.grid_env, "obstacle_positions", set()) or set())
        }
        path_distances = self._build_territory_path_distances(
            source_tile=start_tile,
            blocked_tile_set=blocked_tiles,
        )

        reachable_candidates: List[Tuple[int, int, int, int]] = []
        fallback_candidates: List[Tuple[int, int, int, int]] = []

        for raw_enemy_id, raw_pos in enemy_positions.items():
            enemy_id = int(raw_enemy_id)
            if only_alive and not bool(enemies_alive.get(enemy_id, False)):
                continue
            if spell_targetable_only and not self._is_enemy_stack_spell_targetable(enemy_id):
                continue

            ex = int(raw_pos[0])
            ey = int(raw_pos[1])
            enemy_tile = (ex, ey)

            if enemy_tile in path_distances:
                reachable_candidates.append((int(path_distances[enemy_tile]), enemy_id, ey, ex))

            chebyshev_distance = max(abs(start_tile[0] - ex), abs(start_tile[1] - ey))
            fallback_candidates.append((int(chebyshev_distance), enemy_id, ey, ex))

        if not fallback_candidates:
            return None

        best_distance, best_enemy_id, best_y, best_x = min(
            reachable_candidates or fallback_candidates
        )
        best_pos = (int(best_x), int(best_y))
        is_reachable = bool(reachable_candidates)

        return {
            "enemy_id": int(best_enemy_id),
            "position": best_pos,
            "path_distance": int(best_distance) if is_reachable else None,
            "fallback_distance": None if is_reachable else int(best_distance),
            "reachable": is_reachable,
            "is_alive": bool(enemies_alive.get(int(best_enemy_id), False)),
            "description": ENEMY_DESCRIPTIONS.get(int(best_enemy_id), "Неизвестный враг"),
        }
    def _get_nearest_enemy_stack_for_mask(
        self,
        from_pos: Optional[Tuple[int, int]] = None,
        *,
        only_alive: bool = True,
        spell_targetable_only: bool = False,
    ) -> Optional[Dict[str, object]]:
        """
        Дешёвый кандидат ближайшего врага для лёгких grid-проверок.

        В отличие от get_nearest_enemy_stack() не строит BFS по карте, а использует
        только геометрическую близость (Chebyshev) и состояние живых отрядов.
        Этого достаточно для action mask и spell-observation: сама цель при реальном
        касте всё равно выбирается точным методом в _step_cast_legion_damage_spell().
        """
        enemy_positions = getattr(self.grid_env, "enemy_positions", {}) or {}
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}
        if not enemy_positions:
            return None

        start_pos = tuple(from_pos) if from_pos is not None else tuple(self.grid_env.agent_pos)
        start_x = int(start_pos[0])
        start_y = int(start_pos[1])
        candidates: List[Tuple[int, int, int, int]] = []

        for raw_enemy_id, raw_pos in enemy_positions.items():
            enemy_id = int(raw_enemy_id)
            if only_alive and not bool(enemies_alive.get(enemy_id, False)):
                continue
            if spell_targetable_only and not self._is_enemy_stack_spell_targetable(enemy_id):
                continue
            if not self._enemy_team_has_living_units(enemy_id):
                continue

            ex = int(raw_pos[0])
            ey = int(raw_pos[1])
            chebyshev_distance = max(abs(start_x - ex), abs(start_y - ey))
            candidates.append((int(chebyshev_distance), enemy_id, ey, ex))

        if not candidates:
            return None

        best_distance, best_enemy_id, best_y, best_x = min(candidates)
        best_pos = (int(best_x), int(best_y))
        return {
            "enemy_id": int(best_enemy_id),
            "position": best_pos,
            "path_distance": None,
            "fallback_distance": int(best_distance),
            "reachable": False,
            "is_alive": bool(enemies_alive.get(int(best_enemy_id), False)),
            "description": ENEMY_DESCRIPTIONS.get(int(best_enemy_id), "Неизвестный враг"),
        }
