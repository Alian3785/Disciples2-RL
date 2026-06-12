# campaign_env_masks.py
"""CampaignMaskMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignMaskMixin:
    def action_masks(self):
        """
        Возвращает маску допустимых действий для MaskablePPO.

        SB3-contrib вызывает этот hook перед выбором действия, чтобы агент
        не выбирал заведомо недопустимые action-индексы. Вся фактическая
        логика маскирования собрана в compute_action_mask(), а этот метод
        оставлен тонкой совместимой оберткой под API библиотеки.
        """
        return self.compute_action_mask()
    def _grid_potion_action_mask_values(self) -> Tuple[bool, ...]:
        """
        Строит плоскую маску применения зелий по всем боевым позициям BLUE.

        Возвращаемый tuple упорядочен блоками: сначала все целевые позиции
        для первого зелья из scenario_potion_item_names, затем те же позиции
        для следующего зелья и так далее. Для лечения требуется живой раненый
        юнит, для воскрешения - мертвый юнит с положительным max HP, для
        временных баффов - живой юнит без уже активного такого эффекта на
        позиции. Такая предварительная агрегация не меняет состояние среды и
        используется только при построении action mask.
        """
        positions = tuple(int(pos) for pos in self.GRID_POTION_USE_POSITIONS)
        state = self._get_blue_state()
        unit_by_pos: Dict[int, Dict[str, object]] = {}
        for unit in state:
            if not isinstance(unit, dict):
                continue
            try:
                position = int(unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                continue
            if position in positions:
                unit_by_pos[position] = unit

        can_heal_by_pos: Dict[int, bool] = {}
        can_revive_by_pos: Dict[int, bool] = {}
        is_living_by_pos: Dict[int, bool] = {}
        for position in positions:
            unit = unit_by_pos.get(position)
            if not isinstance(unit, dict):
                can_heal_by_pos[position] = False
                can_revive_by_pos[position] = False
                is_living_by_pos[position] = False
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            can_heal_by_pos[position] = max_hp > 0 and hp > 0 and hp < max_hp
            can_revive_by_pos[position] = max_hp > 0 and hp <= 0
            is_living_by_pos[position] = max_hp > 0 and hp > 0

        definitions = self._potion_definitions_by_canonical()
        values: List[bool] = []
        for item_name in self.scenario_potion_item_names:
            canonical_name = self._canonical_potion_item_name(item_name)
            definition = definitions.get(canonical_name, {})
            available = bool(definition) and self._potion_available_count(item_name) > 0
            effect = definition.get("effect", {}) if isinstance(definition, dict) else {}
            effect_kind = str((effect or {}).get("kind", "") or "")
            duration = str(definition.get("duration", "") or "")
            active_positions = (
                self._active_potion_positions(definition.get("name", ""))
                if available and duration == "temporary"
                else set()
            )
            for position in positions:
                allowed = False
                if available:
                    if effect_kind == "heal":
                        allowed = bool(can_heal_by_pos.get(position, False))
                    elif effect_kind == "revive":
                        allowed = bool(can_revive_by_pos.get(position, False))
                    elif bool(is_living_by_pos.get(position, False)):
                        allowed = not (
                            duration == "temporary" and int(position) in active_positions
                        )
                values.append(bool(allowed))
        return tuple(values)

    def compute_action_mask(self) -> np.ndarray:
        """
        Собирает полную маску допустимых действий для текущего режима среды.

        В grid-режиме метод комбинирует маску передвижения GridWorldEnv с
        кампаниями действиями: зельями, покупками, наймом, тренировкой,
        постройками, изучением и кастом заклинаний, экипировкой, свитками и
        перестановками отрядов. Каждый action-диапазон заполняется только
        если выполнены игровые условия: хватает ресурсов, есть подходящая
        цель, агент стоит у нужного сайта, предмет не использован повторно и
        т.п.

        В battle-режиме маска делегируется текущему BattleEnv, потому что
        допустимость действий в бою определяется уже боевым окружением.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.mode == self.MODE_GRID:
            self._ensure_inventory_cache()
            # В grid режиме движение доступно при любом положительном остатке moves:
            # дорогие клетки добирают все оставшиеся очки, если полной стоимости не хватает.
            # REST (8) доступен при ранении или когда очки перемещения закончились.
            grid_mask = self.grid_env.compute_action_mask()
            mask[:8] = bool(self.moves > 0) & grid_mask[:8]
            movement_available = bool(np.any(mask[:8]))
            mask[8] = bool(grid_mask[8]) and (
                self._has_wounded_blue() or self.moves <= 0 or not movement_available
            )
            potion_mask_values = self._grid_potion_action_mask_values()
            for potion_idx, item_name in enumerate(self.scenario_potion_item_names):
                action_start = self.GRID_POTION_USE_ACTION_START + potion_idx * len(
                    self.GRID_POTION_USE_POSITIONS
                )
                for target_idx, pos in enumerate(self.GRID_POTION_USE_POSITIONS):
                    mask_index = action_start + target_idx
                    value_index = potion_idx * len(self.GRID_POTION_USE_POSITIONS) + target_idx
                    mask[mask_index] = bool(
                        potion_mask_values[value_index]
                        if value_index < len(potion_mask_values)
                        else False
                    )
            if self._merchant_sites_at_position(self.grid_env.agent_pos):
                for idx, item_name in enumerate(self.scenario_merchant_potion_item_names):
                    item_data = self._merchant_item_definition(item_name) or {}
                    item_price = float(item_data.get("price", 0.0) or 0.0)
                    mask[self.GRID_MERCHANT_POTION_BUY_ACTION_START + idx] = (
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
            # Лечение и воскрешение за золото доступны только на специальных клетках лечения.
            if self._is_at_castle() and float(self.gold) > 0.0:
                if self._has_temple_built():
                    for idx, pos in enumerate(self.CASTLE_HEAL_POSITIONS):
                        mask[self.GRID_CASTLE_HEAL_ACTION_START + idx] = self._can_heal_position(pos)
                    for idx, pos in enumerate(self.CASTLE_REVIVE_POSITIONS):
                        mask[self.GRID_CASTLE_REVIVE_ACTION_START + idx] = self._can_castle_revive_position(pos)
            for idx, can_hire in enumerate(self._faction_hire_action_mask_values()):
                mask[self.GRID_HIRE_ACTION_START + idx] = bool(can_hire)
            for idx, can_hire in enumerate(self._mercenary_hire_action_mask_values()):
                mask[self.GRID_MERCENARY_HIRE_ACTION_START + idx] = bool(can_hire)
            if self._trainer_sites_at_position(self.grid_env.agent_pos):
                for idx, pos in enumerate(self.TRAINER_POSITIONS):
                    mask[self.GRID_TRAINER_ACTION_START + idx] = self._can_train_unit_position(
                        pos
                    )

            for idx, can_build in enumerate(self._building_action_mask_values()):
                mask[self.GRID_BUILD_ACTION_START + idx] = bool(can_build)
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
            for idx, can_equip in enumerate(self._battle_equip_action_mask_values()):
                mask[self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + idx] = bool(can_equip)
            for idx, item_name in enumerate(getattr(self, "scenario_book_item_names", ()) or ()):
                mask[self.GRID_EQUIP_BOOK_ACTION_START + idx] = self._can_equip_book_item(
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
            current_staff_specs = self.staff_spell_action_entries()
            for idx, staff_entry in enumerate(current_staff_specs):
                mask[self.grid_staff_spell_action_start + idx] = (
                    bool(staff_entry)
                    and self._can_cast_staff_spell(
                        staff_entry,
                        nearest_targetable_enemy=nearest_targetable_enemy,
                        nearest_any_enemy=nearest_any_enemy,
                    )
                )
            mask[self.GRID_UNLOCK_SCROLL_MAGIC_ACTION] = (
                not bool(self.scroll_magic_unlocked)
                and float(self.gold or 0.0) >= float(self.SCROLL_MAGE_UNLOCK_GOLD_COST)
                and self._has_scroll_or_staff_magic_unlock_source()
            )
            current_scroll_specs = self.scroll_cast_slot_entries()
            for idx in range(int(self.GRID_SCROLL_CAST_ACTION_COUNT)):
                scroll_entry: Dict[str, object] = {}
                if idx < len(current_scroll_specs):
                    scroll_entry = current_scroll_specs[idx]
                action_index = self.grid_scroll_cast_action_start + idx
                if action_index < len(mask):
                    mask[action_index] = (
                        bool(scroll_entry)
                        and self._can_cast_scroll_spell(
                            scroll_entry,
                            nearest_targetable_enemy=nearest_targetable_enemy,
                            nearest_any_enemy=nearest_any_enemy,
                        )
                    )
            for idx, can_swap in enumerate(self._blue_unit_swap_action_mask_values()):
                action_index = int(self.GRID_SWAP_UNIT_ACTION_START) + int(idx)
                if action_index < len(mask):
                    mask[action_index] = bool(can_swap)
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
        """
        Проверяет, есть ли у игрока живые, но не полностью здоровые юниты.

        Используется при построении маски отдыха и лечения: если у BLUE есть
        хотя бы один юнит с hp > 0 и hp < max HP, соответствующие действия
        могут быть разрешены. Метод читает текущее состояние отряда и не
        изменяет здоровье или другие поля юнитов.
        """
        state = self._get_blue_state()
        for unit in state:
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            if hp > 0 and hp < max_hp:
                return True
        return False
    def _can_revive_position(self, position: int) -> bool:
        """
        Определяет, можно ли применить воскрешение к конкретной позиции.

        Позиция считается подходящей, если на ней есть BLUE-юнит с
        положительным max HP, но текущим hp <= 0. Это проверка только
        пригодности цели; наличие самого предмета и его количество
        проверяются вызывающим кодом.
        """
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return max_hp > 0 and hp <= 0
        return False
    def _is_living_blue_position(self, position: int) -> bool:
        """
        Проверяет, стоит ли на указанной позиции живой BLUE-юнит.

        Возвращает True только когда найден юнит с max HP > 0 и текущим
        hp > 0. Helper нужен для зелий-баффов и других действий, которые
        можно применять только к живым союзным отрядам.
        """
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return max_hp > 0 and hp > 0
        return False
    def _can_use_potion_on_position(self, item_name: object, position: int) -> bool:
        """
        Проверяет применимость конкретного зелья к конкретной позиции.

        Метод нормализует имя предмета, находит его описание, убеждается, что
        зелье доступно в инвентаре, и затем выбирает правила по типу эффекта.
        Лечение требует раненого живого юнита, воскрешение - мертвого юнита,
        а баффы требуют живую цель. Для временных баффов дополнительно
        запрещается повторное применение на позицию, где такой эффект уже
        активен.
        """
        canonical_name = self._canonical_potion_item_name(item_name)
        definition = self._potion_definitions_by_canonical().get(canonical_name, {})
        if not definition or self._potion_available_count(item_name) <= 0:
            return False
        effect = definition.get("effect", {})
        effect_kind = str((effect or {}).get("kind", "") or "")
        if effect_kind == "heal":
            return self._can_heal_position(position)
        if effect_kind == "revive":
            return self._can_revive_position(position)
        if not self._is_living_blue_position(position):
            return False
        if str(definition.get("duration", "") or "") == "temporary":
            active_positions = self._active_potion_positions(definition.get("name", ""))
            if int(position) in active_positions:
                return False
        return True
    def _can_castle_revive_position(self, position: int) -> bool:
        """
        Проверяет доступность платного воскрешения в замке для позиции.

        Действие разрешено, если для юнита на позиции рассчитана
        положительная стоимость воскрешения и у игрока хватает золота для
        оплаты. Само списание золота и изменение hp выполняются не здесь, а в
        обработчике действия.
        """
        revive_cost = self._castle_revive_cost_at_position(position)
        return revive_cost > 0.0 and float(self.gold or 0.0) >= revive_cost
    def _revive_unit_at_position(self, position: int) -> Tuple[bool, Optional[str]]:
        """
        Воскрешает мертвого BLUE-юнита на указанной позиции.

        Если на позиции найден юнит с max HP > 0 и hp <= 0, метод выставляет
        ему hp и health в 1.0, пишет событие в лог и возвращает флаг успеха
        вместе с именем юнита. Если позиция пуста, юнит уже жив или не имеет
        корректного max HP, возвращается (False, None).
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
        """
        Проверяет, можно ли лечить BLUE-юнита на указанной позиции.

        Лечение разрешено только для живого юнита с положительным max HP,
        текущим hp > 0 и неполным здоровьем. Метод не учитывает наличие
        конкретной бутылки лечения в инвентаре: он отвечает только за
        пригодность цели.
        """
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
        Возвращает один ближайший вражеский отряд на карте кампании.

        Основной критерий - длина пути от from_pos или текущей позиции агента
        до вражеской клетки с учетом препятствий на карте. При равенстве
        кандидаты упорядочиваются детерминированно: сначала меньший enemy_id,
        затем более верхняя и левая координата. Флаг only_alive отбрасывает
        уничтоженные отряды, а spell_targetable_only оставляет только цели,
        которые могут быть выбраны заклинаниями.

        Если ни один враг недостижим по построенному BFS-пути, используется
        fallback по Chebyshev-дистанции без учета препятствий. Возвращаемый
        словарь содержит enemy_id, позицию, дистанцию, признак достижимости,
        состояние жизни и человекочитаемое описание врага.
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
            if only_alive and not self._enemy_team_has_living_units(enemy_id):
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
        Быстро выбирает ближайшего врага для легких grid-проверок.

        В отличие от get_nearest_enemy_stack(), этот helper не строит BFS и не
        оценивает реальную проходимость карты. Он фильтрует врагов по alive,
        spell_targetable_only и наличию живых юнитов в стеке, затем выбирает
        минимальную Chebyshev-дистанцию от from_pos или текущей позиции агента
        с тем же детерминированным tie-break по enemy_id и координатам.

        Такой приближенный результат достаточно точен для action mask и
        spell-observation, где важна быстрая проверка возможности действия.
        При реальном касте цель дополнительно выбирается более точным методом.
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
