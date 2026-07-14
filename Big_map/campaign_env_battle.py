# campaign_env_battle.py
"""Battle-related campaign logic.

Миксин связывает стратегическую карту CampaignEnv с тактическим BattleEnv:
запускает бой, передает команды, синхронизирует расходники и экипировку,
сохраняет урон между боями и возвращает управление обратно в grid-режим.
"""

from __future__ import annotations

from campaign_env_data import *


class CampaignBattleMixin:
    """Часть CampaignEnv, отвечающая за жизненный цикл боя и persistent-состояние отрядов."""

    def _step_battle(self, action: int):
        """Выполняет один action в текущем BattleEnv и переводит результат в формат CampaignEnv.

        Метод является главным мостом между двумя режимами игры. Пока бой идет,
        он просто прокидывает действие в BattleEnv, масштабирует награду и собирает
        диагностический info. Когда бой завершен, метод применяет кампанийные
        последствия: победные награды, захват целей, сохранение HP, отступление,
        провал кампании или возврат на карту.
        """
        if self.battle_env is None:
            self.mode = self.MODE_GRID
            self._clear_current_battle_context()
            grid_obs = self._get_grid_obs()
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"error": "no_battle_env", "agent_pos": self.grid_env.agent_pos},
            )

        battle_info = {}

        # Проверка: бой уже завершён (например, RED победил в свой ход)
        if self.battle_env.winner is not None:
            winner = self.battle_env.winner
            obs = self.battle_env._obs()
            reward = 0.0
            battle_reward_raw = 0.0
            battle_item_use_reward = 0.0
            terminated = True
            truncated = False
        else:
            action = int(action)
            if action < 0:
                action = 0
            if action >= self.battle_env.action_space.n:
                action = self.battle_env.action_space.n - 1
            obs, battle_reward_raw, terminated, truncated, battle_info = self.battle_env.step(action)
            if isinstance(battle_info, dict):
                consumed_item_name = str(
                    battle_info.get("battle_hero_item_name", "") or ""
                )
                hero_item_action = bool(battle_info.get("battle_hero_item_action"))
                hero_item_applied = bool(battle_info.get("battle_hero_item_applied"))
                hero_item_consumed = bool(battle_info.get("battle_hero_item_consumed"))
                hero_item_charge_spent = bool(
                    battle_info.get("battle_hero_item_charge_spent")
                )
                if (
                    hero_item_action
                    and hero_item_consumed
                    and consumed_item_name
                ):
                    self._consume_battle_equipable_item(consumed_item_name)
                else:
                    self._sync_equipped_hero_items()
                if hero_item_action and hero_item_applied and (
                    hero_item_charge_spent or hero_item_consumed
                ):
                    self.battle_items_used_total += 1
                    battle_item_use_reward = float(self.reward_battle_item_use)
                else:
                    battle_item_use_reward = 0.0
                equipped_items = battle_info.get("equipped_hero_items")
                equipped_item_uses_left = battle_info.get("equipped_hero_item_uses_left")
                if isinstance(equipped_items, list) and str(
                    self.current_battle_context.get("kind", "hero") or "hero"
                ) != "summon_spell":
                    self.equipped_hero_items = list(equipped_items)
                    if isinstance(equipped_item_uses_left, list):
                        self.equipped_hero_item_uses_left = list(equipped_item_uses_left)
                    self._sync_equipped_hero_items()
            else:
                battle_item_use_reward = 0.0
            reward = self.battle_reward_scale * float(battle_reward_raw) + float(
                battle_item_use_reward
            )

        battle_context_kind = str(self.current_battle_context.get("kind", "hero") or "hero")
        info = {
            "mode": "battle",
            "enemy_id": self.current_enemy_id,
            "battle_step": True,
            "battle_context_kind": battle_context_kind,
            "battle_context_spell_key": str(
                self.current_battle_context.get("spell_key", "") or ""
            ),
            "battle_summoned_unit_name": str(
                self.current_battle_context.get("summoned_unit_name", "") or ""
            ),
            "battle_reward_raw": float(battle_reward_raw),
            "battle_reward_scaled": float(self.battle_reward_scale * float(battle_reward_raw)),
            "battle_item_use_reward": float(battle_item_use_reward),
            "battle_items_equipped_total": int(self.battle_items_equipped_total),
            "battle_items_used_total": int(self.battle_items_used_total),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
            "equipped_hero_items": list(self.equipped_hero_items),
        }
        if isinstance(battle_info, dict):
            info.update(battle_info)
        info.update(self._artifact_state_info())
        info.update(self._banner_state_info())

        if terminated or truncated:
            winner = self.battle_env.winner
            info["battle_winner"] = winner

            if battle_context_kind == "summon_spell":
                return self._finalize_summon_spell_battle(
                    obs=obs,
                    reward=reward,
                    info=info,
                    winner=winner,
                )

            # Плотная награда за снятое HP дракона-цели (до разбора исхода боя).
            reward = self._apply_objective_dragon_damage_reward(reward, info)

            if truncated and winner is None:
                self._log(
                    f"=== БОЙ ПРОТИВ ВРАГА {self.current_enemy_id} ЗАВЕРШЁН ПО ЛИМИТУ БЕЗ ПОБЕДИТЕЛЯ ==="
                )
                self._save_enemy_state_from_battle(self.current_enemy_id)
                if self.persist_blue_hp:
                    self._save_blue_state()
                    self._log("Состояние BLUE сохранено после таймаута боя")

                self.mode = self.MODE_GRID
                self.battle_env = None
                self._clear_current_battle_context()
                info["battle_result"] = "timeout"
                info["battle_timeout_no_winner"] = True
                info["mode"] = "grid"
                info["agent_pos"] = self._restore_agent_to_battle_origin()
                info["enemies_alive"] = dict(self.grid_env.enemies_alive)
                info["enemy_state_saved"] = True
                info["blue_state_saved"] = bool(self.persist_blue_hp)

                grid_obs = self._get_grid_obs()
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=False,
                    truncated=False,
                    info=info,
                )

            if winner == "blue":
                exp_reward, exp_raw = self._compute_blue_exp_reward()
                survival_reward, blue_alive_ratio, blue_hp_ratio = self._compute_blue_survival_reward()
                ruin_clear_bonus_reward = (
                    float(self.reward_ruin_clear_bonus)
                    if int(self.current_enemy_id or -1) in self.RUIN_REWARD_BY_ENEMY_ID
                    else 0.0
                )
                enemy_reward = self._compute_enemy_defeat_reward(self.current_enemy_id)
                reward += enemy_reward + exp_reward + survival_reward
                info["enemy_defeat_reward"] = enemy_reward
                info["enemy_defeat_reward_base"] = float(self.reward_defeat_enemy)
                info["ruin_clear_bonus_reward"] = float(ruin_clear_bonus_reward)
                info["blue_exp_reward"] = exp_reward
                info["blue_exp_raw"] = exp_raw
                info["blue_survival_reward"] = survival_reward
                info["blue_alive_ratio"] = blue_alive_ratio
                info["blue_hp_ratio"] = blue_hp_ratio

                # Победа: помечаем врага побеждённым
                self._log(f"=== ПОБЕДА В БОЮ ПРОТИВ ВРАГА {self.current_enemy_id}! ===")
                self.grid_env.mark_enemy_defeated(self.current_enemy_id)
                self._clear_enemy_map_spell_effects_for_enemy(self.current_enemy_id)

                # Восстанавливаем и воскрешаем всех юнитов BLUE после победы
                if self.persist_blue_hp:
                    self._save_blue_state()
                    self._log("Состояние BLUE сохранено (без автолечения и воскрешений)")

                upgrade_count = 0
                if self.battle_env is not None:
                    self._log(f"Опыт за бой: {self.battle_env.last_battle_exp:g}")
                    for name in getattr(self.battle_env, "last_levelups", []) or []:
                        self._log(f"Уровень юнита {name} повышен")
                    upgrade_count = self._log_turns_into_levelups()
                tier_sum = int(getattr(self, "_last_upgrade_tier_sum", 0) or 0)
                upgrade_reward = float(getattr(self, "_last_upgrade_reward", 0.0) or 0.0)
                reward += upgrade_reward
                info["unit_upgrades"] = int(upgrade_count)
                info["unit_upgrade_tier_sum"] = int(tier_sum)
                info["unit_upgrade_max_tier"] = int(getattr(self, "_last_upgrade_max_tier", 0) or 0)
                info["unit_upgrade_tier3_count"] = int(
                    getattr(self, "_last_upgrade_tier3_count", 0) or 0
                )
                info["hero_levelups"] = int(getattr(self, "_last_hero_levelup_count", 0) or 0)
                info["hero_levelup_max_level"] = int(
                    getattr(self, "_last_hero_levelup_max_level", 0) or 0
                )
                info["hero_levelup_reward"] = float(
                    getattr(self, "_last_hero_levelup_reward", 0.0) or 0.0
                )
                info["unit_upgrade_reward"] = float(upgrade_reward)
                info.update(self._grant_ruin_reward(self.current_enemy_id))

                # Возвращаемся в grid режим
                self.mode = self.MODE_GRID
                self.battle_env = None
                self._clear_current_battle_context()
                info["battle_result"] = "victory"
                info["mode"] = "grid"
                info["agent_pos"] = self._restore_agent_to_battle_origin()
                info["enemies_alive"] = dict(self.grid_env.enemies_alive)

                objective_cities_captured_before = len(self.captured_objective_cities)
                newly_captured_objective_cities = self._capture_objective_city_if_cleared(
                    self.current_enemy_id
                )
                info["objective_cities_captured_total"] = sorted(self.captured_objective_cities)
                newly_activated_settlement_territories = (
                    self._activate_legions_settlement_territory_if_cleared(self.current_enemy_id)
                )
                if newly_activated_settlement_territories:
                    info["legions_settlement_territories_activated"] = list(
                        newly_activated_settlement_territories
                    )
                    for settlement_name in newly_activated_settlement_territories:
                        self._log(
                            f"=== ЗЕМЛЯ ЛЕГИОНОВ НАЧИНАЕТ РАСПРОСТРАНЯТЬСЯ ИЗ ПОСЕЛЕНИЯ {settlement_name} ==="
                        )

                if newly_captured_objective_cities and self._campaign_objective_is_cities():
                    final_objective_reward = self._compute_final_objective_reward(
                        len(newly_captured_objective_cities),
                        captured_before=objective_cities_captured_before,
                    )
                    reward += final_objective_reward
                    info["captured_objective_cities"] = list(newly_captured_objective_cities)
                    info["final_objective_reward"] = float(final_objective_reward)
                    for city_name in newly_captured_objective_cities:
                        self._log(f"=== ЗАХВАЧЕН ГОРОД {city_name}: НАГРАДА {final_objective_reward:g} ===")

                reward = self._apply_green_dragon_objective_reward_if_needed(
                    self.current_enemy_id,
                    reward,
                    info,
                )
                campaign_objective_reason = self._campaign_objective_completion_reason(
                    self.current_enemy_id
                )
                if campaign_objective_reason is not None:
                    self._log("=== ЗАХВАЧЕНЫ ВСЕ ЦЕЛЕВЫЕ ГОРОДА: ПОБЕДА В КАМПАНИИ ===")
                    grid_obs = self._get_grid_obs()
                    info["campaign_result"] = "victory"
                    info["campaign_victory_reason"] = campaign_objective_reason
                    return self._finalize_grid_step_result(
                        grid_obs=grid_obs,
                        reward=reward,
                        terminated=True,
                        truncated=False,
                        info=info,
                    )

                # Проверяем полную победу
                if self.grid_env.all_enemies_defeated():
                    self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
                    grid_obs = self._get_grid_obs()
                    info["campaign_result"] = "victory"
                    info["campaign_victory_reason"] = "all_enemies_defeated"
                    return self._finalize_grid_step_result(
                        grid_obs=grid_obs,
                        reward=reward,
                        terminated=True,
                        truncated=False,
                        info=info,
                    )

                grid_obs = self._get_grid_obs()
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=False,
                    truncated=False,
                    info=info,
                )

            else:
                escaped_blue_units = [
                    unit
                    for unit in getattr(self.battle_env, "escaped_units", []) or []
                    if isinstance(unit, dict)
                    and unit.get("team") == "blue"
                    and not bool(unit.get("Summoned"))
                    and self._unit_current_hp(unit) > 0.0
                ]
                if self.persist_blue_hp and escaped_blue_units:
                    self._log(
                        f"=== BLUE РћРўРЎРўРЈРџРђР•Рў РџРћРЎР›Р• РџРћР РђР–Р•РќРРЇ Р’ Р‘РћР® РџР РћРўРР’ Р’Р РђР“Рђ {self.current_enemy_id}: "
                        f"РЎРџРђРЎР›РћРЎР¬ {len(escaped_blue_units)} Р®РќРРў. ==="
                    )
                    self._save_enemy_state_from_battle(self.current_enemy_id)
                    self._save_blue_state()
                    self._log("РЎРѕСЃС‚РѕСЏРЅРёРµ BLUE СЃРѕС…СЂР°РЅРµРЅРѕ РїРѕСЃР»Рµ РѕС‚СЃС‚СѓРїР»РµРЅРёСЏ")

                    self.mode = self.MODE_GRID
                    self.battle_env = None
                    self._clear_current_battle_context()
                    info["battle_result"] = "defeat"
                    info["battle_retreat"] = True
                    info["battle_defeat_with_escaped_units"] = True
                    info["escaped_blue_units"] = int(len(escaped_blue_units))
                    info["mode"] = "grid"
                    info["agent_pos"] = self._restore_agent_to_battle_origin()
                    info["enemies_alive"] = dict(self.grid_env.enemies_alive)
                    info["enemy_state_saved"] = True
                    info["blue_state_saved"] = True

                    grid_obs = self._get_grid_obs()
                    return self._finalize_grid_step_result(
                        grid_obs=grid_obs,
                        reward=reward,
                        terminated=False,
                        truncated=False,
                        info=info,
                    )

                reward += self.reward_loss
                self._clear_battle_origin()
                self._clear_current_battle_context()
                self._log(f"=== ПОРАЖЕНИЕ В БОЮ ПРОТИВ ВРАГА {self.current_enemy_id}! ===")
                self._log("=== КАМПАНИЯ ПРОВАЛЕНА ===")
                info["battle_result"] = "defeat"
                info["campaign_result"] = "defeat"
                if self.battle_env is not None:
                    self._log(f"Опыт за бой: {self.battle_env.last_battle_exp:g}")
                    for name in getattr(self.battle_env, "last_levelups", []) or []:
                        self._log(f"Уровень юнита {name} повышен")
                    upgrade_count = self._log_turns_into_levelups()
                    tier_sum = int(getattr(self, "_last_upgrade_tier_sum", 0) or 0)
                    upgrade_reward = float(getattr(self, "_last_upgrade_reward", 0.0) or 0.0)
                    reward += upgrade_reward
                    info["unit_upgrades"] = int(upgrade_count)
                    info["unit_upgrade_tier_sum"] = int(tier_sum)
                    info["unit_upgrade_max_tier"] = int(getattr(self, "_last_upgrade_max_tier", 0) or 0)
                    info["unit_upgrade_tier3_count"] = int(
                        getattr(self, "_last_upgrade_tier3_count", 0) or 0
                    )
                    info["hero_levelups"] = int(getattr(self, "_last_hero_levelup_count", 0) or 0)
                    info["hero_levelup_max_level"] = int(
                        getattr(self, "_last_hero_levelup_max_level", 0) or 0
                    )
                    info["hero_levelup_reward"] = float(
                        getattr(self, "_last_hero_levelup_reward", 0.0) or 0.0
                    )
                    info["unit_upgrade_reward"] = float(upgrade_reward)
                # Поражение минует _finalize_grid_step_result — добавляем терминальный бонус здесь.
                reward += self._episode_enemies_defeated_bonus(info)
                return self._build_obs(battle_obs=obs), reward, True, False, info

        info["battle_ongoing"] = True
        return self._build_obs(battle_obs=obs), reward, terminated, truncated, info
    def _restore_agent_to_battle_origin(self) -> Tuple[int, int]:
        """Возвращает агента на клетку, с которой был инициирован текущий бой.

        Во время столкновения агент может технически находиться на клетке врага.
        После нефатального завершения боя нужно вернуть его на исходную клетку
        и очистить одноразовый маркер battle_origin_pos.
        """
        if self.battle_origin_pos is None:
            return tuple(self.grid_env.agent_pos)
        origin = (int(self.battle_origin_pos[0]), int(self.battle_origin_pos[1]))
        self.grid_env.agent_pos = origin
        self.battle_origin_pos = None
        return origin
    def _clear_battle_origin(self) -> None:
        """Очищает сохраненную стартовую клетку боя без перемещения агента."""
        self.battle_origin_pos = None
    @staticmethod
    def _normalize_armor_value(value: object) -> int:
        """Безопасно приводит значение брони к неотрицательному целому числу."""
        try:
            return max(0, int(round(float(value or 0))))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _unit_current_hp(unit: Dict) -> float:
        """Возвращает текущий HP: ``health`` основной, ``hp`` fallback для старых данных."""
        if not isinstance(unit, dict):
            return 0.0
        if "health" in unit and unit.get("health") is not None:
            raw_value = unit.get("health")
        else:
            raw_value = unit.get("hp", 0.0)
        try:
            return max(0.0, float(raw_value or 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _unit_max_hp(unit: Dict, fallback: object = 0.0) -> float:
        """Возвращает максимум HP: ``max_health`` основной, ``maxhp`` fallback для старых данных."""
        if not isinstance(unit, dict):
            return 0.0
        if "max_health" in unit and unit.get("max_health") is not None:
            raw_value = unit.get("max_health")
        elif "maxhp" in unit and unit.get("maxhp") is not None:
            raw_value = unit.get("maxhp")
        else:
            raw_value = fallback
        try:
            return max(0.0, float(raw_value or 0.0))
        except (TypeError, ValueError):
            return 0.0
    def _resolve_heal_tile_context(
        self,
        position: Optional[Tuple[int, int]],
    ) -> tuple[int, float, str, Optional[int]]:
        """Определяет бонусы клетки лечения для BLUE-отряда перед боем.

        Возвращает кортеж: бонус брони, бонус лечения от отдыха, имя поселения
        и уровень поселения. Бонус действует только на территории поселения,
        которое уже активировано фракцией Легионов.
        """
        if position is None:
            return 0, 0.0, "", None

        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return 0, 0.0, "", None

        settlement_name = self.legions_settlement_source_name_by_tile.get(normalized_position)
        if not settlement_name:
            return 0, 0.0, "", None
        if settlement_name not in self.legions_active_settlement_territory_capture_turn_by_name:
            return 0, 0.0, "", None

        settlement_level = self.legions_settlement_level_by_name.get(str(settlement_name))
        if settlement_level is None:
            return 0, 0.0, "", None
        normalized_level = max(0, int(settlement_level))
        return (
            int(SETTLEMENT_ARMOR_BONUS_BY_LEVEL.get(normalized_level, 0)),
            float(self.SETTLEMENT_REST_HEAL_BONUS_BY_LEVEL.get(normalized_level, 0.0)),
            str(settlement_name),
            normalized_level,
        )
    def _resolve_rest_regeneration_context(
        self,
        position: Optional[Tuple[int, int]],
    ) -> tuple[float, str, Optional[object]]:
        """Возвращает процентный бонус лечения от отдыха на клетке города/столицы.

        Второе и третье значения описывают источник бонуса: строковую метку
        и уровень поселения либо специальный маркер ``capital``.
        """
        if position is None:
            return 0.0, "", None

        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return 0.0, "", None

        if normalized_position == tuple(self.CASTLE_POS):
            return float(CAPITAL_HEAL_TILE_REST_BONUS) / 100.0, "capital", "capital"

        settlement_name = self.legions_settlement_source_name_by_tile.get(normalized_position)
        if not settlement_name:
            return 0.0, "", None
        if settlement_name not in self.legions_active_settlement_territory_capture_turn_by_name:
            return 0.0, "", None

        settlement_level = self.legions_settlement_level_by_name.get(str(settlement_name))
        if settlement_level is None:
            return 0.0, "", None
        normalized_level = max(0, int(settlement_level))
        bonus_percent = float(self.SETTLEMENT_REST_HEAL_BONUS_BY_LEVEL.get(normalized_level, 0.0))
        return bonus_percent / 100.0, str(settlement_name), normalized_level
    def _resolve_rest_heal_percent(
        self,
        position: Optional[Tuple[int, int]],
    ) -> float:
        """Возвращает базовый процент лечения от отдыха на текущей клетке.

        Своя территория лечит сильнее обычной клетки. Дополнительные бонусы
        поселений/лорда считаются отдельными helper-методами.
        """
        if self._is_player_controlled_territory_tile(position):
            return float(self.PLAYER_TERRITORY_REST_HEAL_PERCENT)
        return float(self.DEFAULT_REST_HEAL_PERCENT)
    def _resolve_typeoflord_rest_heal_bonus_percent(self) -> float:
        """Возвращает дополнительный бонус лечения от отдыха по типу лорда."""
        if int(self.typeoflord) == 1:
            return float(self.TYPEOFLORD_ONE_REST_HEAL_BONUS_PERCENT)
        return 0.0
    def _resolve_enemy_regeneration_bonus_percent(
        self,
        position: Optional[Tuple[int, int]],
    ) -> float:
        """Возвращает бонус регенерации врага на клетке города/столицы.

        Используется для RED-стеков, которые стоят на источнике территории:
        столице Империи или поселении Легионов.
        """
        if position is None:
            return 0.0

        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return 0.0

        if normalized_position == tuple(self.empire_territory_source_tile):
            return float(CAPITAL_HEAL_TILE_REST_BONUS) / 100.0

        settlement_name = self.legions_settlement_source_name_by_tile.get(normalized_position)
        if not settlement_name:
            return 0.0

        settlement_level = self.legions_settlement_level_by_name.get(str(settlement_name))
        if settlement_level is None:
            return 0.0

        normalized_level = max(0, int(settlement_level))
        return float(self.SETTLEMENT_REST_HEAL_BONUS_BY_LEVEL.get(normalized_level, 0.0)) / 100.0
    def _resolve_settlement_defender_armor_bonus(
        self,
        enemy_id: Optional[int],
    ) -> tuple[int, str]:
        """Возвращает бонус брони RED-защитника и строковую метку источника.

        Бонус может приходить от уровня поселения или от столицы. Если враг
        одновременно соответствует нескольким источникам, применяется больший.
        """
        if enemy_id is None:
            return 0, ""

        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return 0, ""

        bonus = 0
        source = ""

        settlement_level = SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID.get(normalized_enemy_id)
        if settlement_level is not None:
            bonus = int(SETTLEMENT_ARMOR_BONUS_BY_LEVEL.get(int(settlement_level), 0))
            source = f"settlement_level_{int(settlement_level)}"

        enemy_pos = self.grid_env.enemy_positions.get(normalized_enemy_id)
        capital_tiles = {tuple(self.CASTLE_POS)}
        empire_source_tile = tuple(getattr(self, "empire_territory_source_tile", ()))
        if len(empire_source_tile) == 2:
            capital_tiles.add(empire_source_tile)
        if enemy_pos is not None and tuple(enemy_pos) in capital_tiles:
            capital_bonus = int(CAPITAL_HEAL_TILE_ARMOR_BONUS)
            if capital_bonus >= bonus:
                bonus = capital_bonus
                source = "capital"

        return max(0, int(bonus)), source
    def _apply_settlement_defender_armor_bonus(
        self,
        enemy_id: Optional[int],
        red_team: List[Dict],
    ) -> None:
        """Добавляет RED-команде броню за оборону поселения или столицы."""
        bonus, source = self._resolve_settlement_defender_armor_bonus(enemy_id)
        if bonus <= 0:
            return

        affected_units = 0
        for unit in red_team:
            if self._is_empty_enemy_unit(unit):
                continue
            base_armor = self._normalize_armor_value(unit.get("armor", 0))
            unit["armor"] = base_armor + bonus
            unit["settlement_armor_bonus"] = int(bonus)
            if source.startswith("settlement_level_"):
                try:
                    unit["settlement_level"] = int(source.rsplit("_", 1)[-1])
                except (TypeError, ValueError):
                    pass
            elif source == "capital":
                unit["settlement_level"] = "capital"
            affected_units += 1

        if affected_units <= 0:
            return

        if source == "capital":
            self._log(
                f"Бонус столицы: RED-отряд {enemy_id} получает +{bonus} брони на каждого юнита "
                f"({affected_units} юнитов)."
            )
        else:
            self._log(
                f"Бонус города: RED-отряд {enemy_id} получает +{bonus} брони на каждого юнита "
                f"({affected_units} юнитов, {source})."
            )
    def _apply_hero_heal_tile_armor_bonus(self, blue_team: List[Dict]) -> None:
        """Добавляет BLUE-команде временную броню от клетки, где начался бой."""
        bonus, _rest_bonus, source, settlement_level = self._resolve_heal_tile_context(
            self.battle_origin_pos
        )
        if bonus <= 0:
            return

        affected_units = 0
        for unit in blue_team:
            max_hp = self._unit_max_hp(unit)
            if max_hp <= 0:
                continue

            base_armor = self._normalize_armor_value(unit.get("armor", 0))
            unit["campaign_heal_tile_base_armor"] = int(base_armor)
            unit["campaign_heal_tile_armor_bonus"] = int(bonus)
            unit["armor"] = base_armor + bonus
            unit["settlement_armor_bonus"] = int(bonus)
            if settlement_level is not None:
                unit["settlement_level"] = int(settlement_level)
            affected_units += 1

        if affected_units <= 0:
            return

        if settlement_level is not None:
            self._log(
                f"Бонус города {source} уровня {settlement_level}: BLUE-отряд героя получает "
                f"+{bonus} брони на каждого юнита ({affected_units} юнитов)."
            )
    def _init_battle(
        self,
        enemy_id: int,
        *,
        blue_team: Optional[List[Dict]] = None,
    ):
        """Инициализирует BattleEnv с RED-командой врага и текущей BLUE-командой.

        RED берется из сохраненного состояния enemy_team_states, если оно есть,
        иначе из enemy configs выбранной карты. BLUE может быть передан явно для
        специальных боев (например, summon), либо собирается из persistent-состояния героя.
        """
        red_team_state = self._get_enemy_team_state(enemy_id)
        if red_team_state:
            red_team = self._build_battle_team_with_placeholders("red", red_team_state)
        else:
            red_team = self._build_battle_team_with_placeholders(
                "red",
                self._enemy_configs.get(enemy_id)
                or next(iter(self._enemy_configs.values())),
            )
        self._apply_settlement_defender_armor_bonus(enemy_id, red_team)

        # Для рекордной награды по дракону-цели: запоминаем суммарный max-HP RED.
        self._objective_dragon_max_hp = 0.0
        if self._is_green_dragon_objective_enemy(enemy_id):
            self._objective_dragon_max_hp = float(
                sum(float(u.get("max_health", 0) or 0) for u in red_team)
            )

        if blue_team is not None:
            prepared_blue_team = self._build_battle_team_with_placeholders("blue", blue_team)
            self._apply_equipped_book_battle_effects(prepared_blue_team)
            equipped_hero_items: List[Optional[str]] = [None] * int(self.BATTLE_EQUIP_SLOTS)
            equipped_hero_item_uses_left: List[Optional[int]] = [None] * int(
                self.BATTLE_EQUIP_SLOTS
            )
            hero_item_effects: Dict[str, Dict[str, object]] = {}
            self._log("Используется кастомная BLUE команда")
        elif self.persist_blue_hp and self.blue_team_state is not None:
            prepared_blue_team = self._build_battle_team_with_placeholders(
                "blue",
                self.blue_team_state,
            )
            self._clear_equipped_banner_effects(prepared_blue_team)
            self._clear_equipped_artifact_effects(prepared_blue_team)
            self._apply_hero_heal_tile_armor_bonus(prepared_blue_team)
            self._apply_active_blue_potion_effects(prepared_blue_team)
            self._apply_active_blue_support_spell_effects(prepared_blue_team)
            self._apply_equipped_book_battle_effects(prepared_blue_team)
            self._apply_equipped_artifact_effects(prepared_blue_team)
            self._apply_equipped_banner_effects(prepared_blue_team)
            self._sync_equipped_hero_items()
            equipped_hero_items = list(self.equipped_hero_items)
            equipped_hero_item_uses_left = list(self.equipped_hero_item_uses_left)
            hero_item_effects = dict(self._battle_item_effect_definitions())
            self._log("Загружено сохранённое состояние BLUE команды")
        else:
            prepared_blue_team = self._build_battle_team_with_placeholders(
                "blue",
                self._create_starting_blue_team(),
            )
            self._clear_equipped_banner_effects(prepared_blue_team)
            self._clear_equipped_artifact_effects(prepared_blue_team)
            self._apply_hero_heal_tile_armor_bonus(prepared_blue_team)
            self._apply_active_blue_potion_effects(prepared_blue_team)
            self._apply_active_blue_support_spell_effects(prepared_blue_team)
            self._apply_equipped_book_battle_effects(prepared_blue_team)
            self._apply_equipped_artifact_effects(prepared_blue_team)
            self._apply_equipped_banner_effects(prepared_blue_team)
            self._sync_equipped_hero_items()
            equipped_hero_items = list(self.equipped_hero_items)
            equipped_hero_item_uses_left = list(self.equipped_hero_item_uses_left)
            hero_item_effects = dict(self._battle_item_effect_definitions())
            self._log("Используется дефолтная BLUE команда")

        self.battle_env = BattleEnv(
            reward_win=self.battle_reward_win,
            reward_loss=self.battle_reward_loss,
            reward_step=self.battle_reward_step,
            log_enabled=self.log_enabled,
        )
        self.battle_env.equipped_hero_items = list(equipped_hero_items)
        self.battle_env.equipped_hero_item_uses_left = list(equipped_hero_item_uses_left)
        self.battle_env.hero_item_effects = dict(hero_item_effects)
        self.battle_env.exp_multiplier = float(self._active_book_exp_multiplier())

        # Инициализируем бой с кастомными командами
        self.battle_env._init_with_custom_teams(red_team, prepared_blue_team)
    def _save_blue_state(self):
        """
        Сохраняет persistent-состояние базовой BLUE-команды после боя.

        HP НЕ восстанавливается, погибшие НЕ воскрешаются. Метод переносит
        только долгоживущие изменения: текущий HP, level-up/превращения юнитов
        и прогресс героя. Временные боевые эффекты, бонусы картовых заклинаний,
        зелий, баннеров, артефактов и клеток лечения очищаются, потому что они
        должны применяться заново при следующем входе в BattleEnv.
        """
        if self.battle_env is None:
            return

        ordered_base_positions = tuple(int(u["position"]) for u in UNITS_BLUE)
        base_positions = set(ordered_base_positions)
        original_blue_team = self._build_battle_team_with_placeholders(
            "blue",
            self.blue_team_state if self.blue_team_state is not None else UNITS_BLUE,
        )
        original_by_position = {
            int(unit.get("position", -1) or -1): deepcopy(unit)
            for unit in original_blue_team
        }
        start_empty_positions = {
            position
            for position, unit in original_by_position.items()
            if self._is_empty_blue_unit(unit)
        }
        escaped_by_position: Dict[int, Dict] = {}
        if getattr(self.battle_env, "winner", None) in ("blue", "red"):
            for escaped_unit in getattr(self.battle_env, "escaped_units", []) or []:
                if not isinstance(escaped_unit, dict):
                    continue
                if escaped_unit.get("team") != "blue" or bool(escaped_unit.get("Summoned")):
                    continue
                try:
                    escaped_pos = int(escaped_unit.get("position", -1) or -1)
                except (TypeError, ValueError):
                    continue
                if escaped_pos in base_positions and escaped_pos not in start_empty_positions:
                    restored_escape = deepcopy(escaped_unit)
                    restored_escape["running_away"] = 0
                    escaped_by_position.setdefault(escaped_pos, restored_escape)
        saved_by_position: Dict[int, Dict] = {}

        for u in self.battle_env.combined:
            if u.get("team") != "blue":
                continue
            
            try:
                pos = int(u.get("position", -1) or -1)
            except (TypeError, ValueError):
                continue
            if pos not in base_positions:
                # Юнит не относится к базовым позициям (призванный?) — пропускаем
                continue
            if pos in start_empty_positions:
                # Эта позиция была пустой в начале боя. Любой появившийся здесь юнит временный.
                continue
            escaped_unit = escaped_by_position.get(pos)
            if escaped_unit is not None:
                u = escaped_unit
            if bool(u.get("Summoned")):
                original_unit = deepcopy(
                    original_by_position.get(pos, placeholder_unit("blue", pos))
                )
                if not self._is_empty_blue_unit(original_unit) and pos not in saved_by_position:
                    original_unit["health"] = 0.0
                    original_unit["hp"] = 0.0
                    original_unit["initiative"] = 0
                    saved_by_position[pos] = original_unit
                continue

            # Сохраняем юнита в текущей форме (после улучшений)
            restored_unit = deepcopy(u)
            restore_transformed = getattr(
                self.battle_env, "_restore_transformed_unit", None
            )
            if callable(restore_transformed):
                restore_transformed(restored_unit)
            if escaped_unit is not None:
                cleanse_negative_effects = getattr(
                    self.battle_env,
                    "_cleanse_negative_effects",
                    None,
                )
                if callable(cleanse_negative_effects):
                    cleanse_negative_effects(restored_unit)

            original_unit_for_position = original_by_position.get(pos, {})
            original_damage = int(
                restored_unit.get(
                    "original_damage",
                    original_unit_for_position.get("damage", restored_unit.get("damage", 0)),
                )
                or 0
            )
            if (
                int(restored_unit.get("powerup", 0) or 0) != 0
                or int(restored_unit.get("teamated", 0) or 0) != 0
            ) and original_damage > 0:
                restored_unit["damage"] = original_damage
            if int(restored_unit.get("hermited", 0) or 0) != 0:
                base_ini = int(restored_unit.get("initiative_base", 0) or 0)
                restored_unit["initiative_base"] = base_ini * 2 if base_ini > 0 else base_ini
            
            # СОХРАНЯЕМ ТЕКУЩЕЕ HP (не восстанавливаем!)
            current_hp = self._unit_current_hp(restored_unit)
            restored_unit["health"] = current_hp
            restored_unit["hp"] = current_hp

            # Сбрасываем все временные боевые эффекты
            restored_unit["defense"] = 0
            restored_unit["waited"] = 0
            restored_unit["paralyzed"] = 0
            restored_unit["long_paralyzed"] = 0
            restored_unit["running_away"] = 0
            restored_unit["transformed"] = 0
            restored_unit["poison_turns_left"] = 0
            restored_unit["poison_damage_per_tick"] = 0
            restored_unit["burn_turns_left"] = 0
            restored_unit["burn_damage_per_tick"] = 0
            restored_unit["burn_source_lord_pos"] = None
            restored_unit["uran_turns_left"] = 0
            restored_unit["uran_damage_per_tick"] = 0
            restored_unit["powerup"] = 0
            restored_unit["bonusturn"] = 0
            restored_unit["resilience_used_types"] = []
            restored_unit.pop("teamated", None)
            restored_unit.pop("hermited", None)
            position = int(restored_unit.get("position", -1) or -1)
            if "campaign_banner_base_damage" in restored_unit:
                restored_unit["damage"] = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_banner_base_damage",
                        restored_unit.get("damage", 0),
                    )
                )
            if "campaign_banner_base_accuracy" in restored_unit:
                restored_unit["accuracy"] = self._normalize_accuracy_value(
                    restored_unit.get(
                        "campaign_banner_base_accuracy",
                        restored_unit.get("accuracy", 0),
                    )
                )
            if "campaign_banner_base_initiative" in restored_unit:
                restored_unit["initiative_base"] = int(
                    self._normalize_damage_value(
                        restored_unit.get(
                            "campaign_banner_base_initiative",
                            restored_unit.get(
                                "initiative_base",
                                restored_unit.get("initiative", 0),
                            ),
                        )
                    )
                )
                restored_unit["initiative"] = int(restored_unit.get("initiative_base", 0) or 0)
            if "campaign_banner_base_armor" in restored_unit:
                base_armor = self._normalize_armor_value(
                    restored_unit.get(
                        "campaign_banner_base_armor",
                        restored_unit.get("armor", 0),
                    )
                )
                restored_unit["armor"] = int(base_armor)
                restored_unit["base_armor"] = int(base_armor)
            if "campaign_artifact_base_damage" in restored_unit:
                restored_unit["damage"] = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_artifact_base_damage",
                        restored_unit.get("damage", 0),
                    )
                )
            if "campaign_artifact_base_initiative" in restored_unit:
                restored_unit["initiative_base"] = int(
                    self._normalize_damage_value(
                        restored_unit.get(
                            "campaign_artifact_base_initiative",
                            restored_unit.get(
                                "initiative_base",
                                restored_unit.get("initiative", 0),
                            ),
                        )
                    )
                )
                restored_unit["initiative"] = int(restored_unit.get("initiative_base", 0) or 0)
            if "campaign_artifact_base_armor" in restored_unit:
                base_armor = self._normalize_armor_value(
                    restored_unit.get(
                        "campaign_artifact_base_armor",
                        restored_unit.get("armor", 0),
                    )
                )
                restored_unit["armor"] = int(base_armor)
                restored_unit["base_armor"] = int(base_armor)
            if "campaign_map_spell_base_damage" in restored_unit:
                restored_unit["damage"] = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_map_spell_base_damage",
                        restored_unit.get("damage", 0),
                    )
                )
            if "campaign_map_spell_base_damage_secondary" in restored_unit:
                restored_unit["damage_secondary"] = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_map_spell_base_damage_secondary",
                        restored_unit.get("damage_secondary", 0),
                    )
                )
            if "campaign_map_spell_base_initiative" in restored_unit:
                restored_unit["initiative_base"] = int(
                    self._normalize_damage_value(
                        restored_unit.get(
                            "campaign_map_spell_base_initiative",
                            restored_unit.get("initiative_base", restored_unit.get("initiative", 0)),
                        )
                    )
                )
                restored_unit["initiative"] = int(restored_unit.get("initiative_base", 0) or 0)
            if "campaign_map_spell_base_accuracy" in restored_unit:
                restored_unit["accuracy"] = self._normalize_accuracy_value(
                    restored_unit.get(
                        "campaign_map_spell_base_accuracy",
                        restored_unit.get("accuracy", 0),
                    )
                )
            if "campaign_map_spell_base_accuracy_secondary" in restored_unit:
                restored_unit["accuracy_secondary"] = self._normalize_accuracy_value(
                    restored_unit.get(
                        "campaign_map_spell_base_accuracy_secondary",
                        restored_unit.get("accuracy_secondary", 0),
                    )
                )
            if "campaign_map_spell_base_max_health" in restored_unit:
                base_max_health = float(
                    restored_unit.get(
                        "campaign_map_spell_base_max_health",
                        self._unit_max_hp(restored_unit),
                    )
                    or 0.0
                )
                health_bonus = float(
                    restored_unit.get("campaign_map_spell_health_bonus", 0) or 0.0
                )
                current_hp = self._unit_current_hp(restored_unit)
                restored_health = max(0.0, min(base_max_health, current_hp - health_bonus))
                restored_unit["max_health"] = float(base_max_health)
                restored_unit["maxhp"] = float(base_max_health)
                restored_unit["health"] = float(restored_health)
                restored_unit["hp"] = float(restored_health)
            added_spell_resistance = [
                str(res)
                for res in (restored_unit.get("campaign_map_spell_added_resistance") or [])
                if str(res)
            ]
            if added_spell_resistance:
                added_spell_resistance_set = set(added_spell_resistance)
                restored_unit["resistance"] = [
                    res
                    for res in (restored_unit.get("resistance") or [])
                    if str(res) not in added_spell_resistance_set
                ]
            added_book_resistance = [
                str(res)
                for res in (restored_unit.get("campaign_book_added_resistance") or [])
                if str(res)
            ]
            if added_book_resistance:
                added_book_resistance_set = set(added_book_resistance)
                restored_unit["resistance"] = [
                    res
                    for res in (restored_unit.get("resistance") or [])
                    if str(res) not in added_book_resistance_set
                ]
            if "campaign_map_spell_base_armor" in restored_unit:
                restored_unit["armor"] = self._normalize_armor_value(
                    restored_unit.get(
                        "campaign_map_spell_base_armor",
                        restored_unit.get("armor", 0),
                    )
                )
                restored_unit["base_armor"] = int(restored_unit["armor"])
            restored_unit.pop("campaign_map_spell_base_damage", None)
            restored_unit.pop("campaign_map_spell_base_damage_secondary", None)
            restored_unit.pop("campaign_map_spell_base_initiative", None)
            restored_unit.pop("campaign_map_spell_base_accuracy", None)
            restored_unit.pop("campaign_map_spell_base_accuracy_secondary", None)
            restored_unit.pop("campaign_map_spell_base_max_health", None)
            restored_unit.pop("campaign_map_spell_health_bonus", None)
            restored_unit.pop("campaign_map_spell_added_resistance", None)
            restored_unit.pop("campaign_book_added_resistance", None)
            restored_unit.pop("campaign_active_book", None)
            restored_unit.pop("campaign_map_spell_base_armor", None)
            restored_unit.pop("campaign_banner_base_damage", None)
            restored_unit.pop("campaign_banner_base_accuracy", None)
            restored_unit.pop("campaign_banner_base_initiative", None)
            restored_unit.pop("campaign_banner_base_armor", None)
            restored_unit.pop("campaign_banner_damage_multiplier", None)
            restored_unit.pop("campaign_banner_accuracy_bonus", None)
            restored_unit.pop("campaign_banner_initiative_multiplier", None)
            restored_unit.pop("campaign_banner_armor_bonus", None)
            restored_unit.pop("campaign_active_banner", None)
            restored_unit.pop("campaign_artifact_base_damage", None)
            restored_unit.pop("campaign_artifact_base_damage_secondary", None)
            restored_unit.pop("campaign_artifact_base_initiative", None)
            restored_unit.pop("campaign_artifact_base_armor", None)
            restored_unit.pop("campaign_artifact_damage_multiplier", None)
            restored_unit.pop("campaign_artifact_initiative_multiplier", None)
            restored_unit.pop("campaign_artifact_armor_bonus", None)
            restored_unit.pop("campaign_active_artifacts", None)
            if "campaign_potion_base_damage" in restored_unit:
                base_damage = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_potion_base_damage",
                        restored_unit.get("damage", 0),
                    )
                )
                base_damage_secondary = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_potion_base_damage_secondary",
                        restored_unit.get("damage_secondary", 0),
                    )
                )
                restored_unit["damage"] = int(base_damage)
                restored_unit["damage_secondary"] = int(base_damage_secondary)
            restored_unit.pop("original_damage", None)
            restored_unit.pop("campaign_potion_base_damage", None)
            restored_unit.pop("campaign_potion_base_damage_secondary", None)
            restored_unit.pop("campaign_damage_potion_multiplier", None)
            restored_unit.pop("campaign_strength_potion_multiplier", None)
            restored_unit.pop("campaign_energy_elixir_multiplier", None)
            if "campaign_potion_base_initiative" in restored_unit:
                base_initiative = int(
                    restored_unit.get(
                        "campaign_potion_base_initiative",
                        restored_unit.get("initiative_base", 0),
                    )
                    or 0
                )
                restored_unit["initiative_base"] = int(base_initiative)
            restored_unit.pop("campaign_potion_base_initiative", None)
            restored_unit.pop("campaign_initiative_potion_multiplier", None)
            if "campaign_potion_base_accuracy" in restored_unit:
                restored_unit["accuracy"] = self._normalize_accuracy_value(
                    restored_unit.get(
                        "campaign_potion_base_accuracy",
                        restored_unit.get("accuracy", 0),
                    )
                )
            if "campaign_potion_base_accuracy_secondary" in restored_unit:
                restored_unit["accuracy_secondary"] = self._normalize_accuracy_value(
                    restored_unit.get(
                        "campaign_potion_base_accuracy_secondary",
                        restored_unit.get("accuracy_secondary", 0),
                    )
                )
            restored_unit.pop("campaign_potion_base_accuracy", None)
            restored_unit.pop("campaign_potion_base_accuracy_secondary", None)
            restored_unit.pop("campaign_accuracy_potion_multiplier", None)
            if "campaign_potion_base_incoming_damage_multiplier" in restored_unit:
                restored_unit["incoming_damage_multiplier"] = max(
                    0.0,
                    float(
                        restored_unit.get(
                            "campaign_potion_base_incoming_damage_multiplier",
                            restored_unit.get("incoming_damage_multiplier", 1.0),
                        )
                        or 1.0
                    ),
                )
            restored_unit.pop("campaign_potion_base_incoming_damage_multiplier", None)
            restored_unit.pop("campaign_incoming_damage_potion_multiplier", None)
            added_resistance = [
                str(res)
                for res in (restored_unit.get("campaign_potion_added_resistance") or [])
                if str(res)
            ]
            if added_resistance:
                added_resistance_set = set(added_resistance)
                restored_unit["resistance"] = [
                    res
                    for res in (restored_unit.get("resistance") or [])
                    if str(res) not in added_resistance_set
                ]
            restored_unit.pop("campaign_potion_added_resistance", None)
            if position in self.active_invulnerability_potion_positions:
                base_armor = self._normalize_armor_value(
                    restored_unit.get(
                        "campaign_potion_base_armor",
                        restored_unit.get("armor", 0),
                    )
                )
            else:
                # Armor must be restored to the unit's battle-start value.
                base_armor = int(restored_unit.get("base_armor", restored_unit.get("armor", 0)) or 0)
            if "campaign_heal_tile_base_armor" in restored_unit:
                base_armor = self._normalize_armor_value(
                    restored_unit.get("campaign_heal_tile_base_armor", base_armor)
                )
            restored_unit["armor"] = max(0, base_armor)
            restored_unit.pop("campaign_potion_base_armor", None)
            restored_unit.pop("campaign_invulnerability_potion_bonus", None)
            restored_unit.pop("campaign_heal_tile_base_armor", None)
            restored_unit.pop("campaign_heal_tile_armor_bonus", None)
            restored_unit.pop("settlement_armor_bonus", None)
            restored_unit.pop("settlement_level", None)
            
            # Восстанавливаем initiative к базовому значению
            restored_unit["initiative"] = restored_unit.get("initiative_base", 0)
            restored_unit["needaunit"] = self._resolve_hero_needaunit(restored_unit)
            if self._apply_lord_replacement_might_bonus(restored_unit):
                self._log(
                    f"Герой {restored_unit.get('name', 'Герой')} получает умение Мощь: "
                    f"primary damage -> {int(restored_unit.get('damage', 0) or 0)}."
                )
            
            saved_by_position[position] = restored_unit

        self.blue_team_state = [
            saved_by_position.get(position, placeholder_unit("blue", position))
            for position in ordered_base_positions
        ]
        self._mark_equipment_dirty()
        self._sync_hero_progression_flags(self.blue_team_state)
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state, grant_delta=True)
        self._refresh_campaign_equipment_effects(log=False)
        self._log("Состояние BLUE команды сохранено (HP не восстановлено)")
    def _find_unit_data_by_name(self, name: str) -> Optional[Dict]:
        """Ищет запись юнита в UNIT_DATA по отображаемому имени."""
        if name is None:
            return None
        return self._unit_data_by_name().get(str(name))

    @staticmethod
    @lru_cache(maxsize=1)
    def _unit_data_by_name() -> Dict[str, Dict]:
        """Строит быстрый индекс UNIT_DATA по имени юнита."""
        result: Dict[str, Dict] = {}
        for entry in UNIT_DATA:
            if not isinstance(entry, dict):
                continue
            # Prefer canonical Russian key, keep mojibake fallback for legacy data dumps.
            for key in ("\u043a\u0442\u043e", "\u0420\u0454\u0421\u201a\u0420\u0455"):
                entry_name = entry.get(key)
                if entry_name is not None:
                    result.setdefault(str(entry_name), entry)
        return result
    def _clear_current_battle_context(self) -> None:
        """Очищает metadata текущего боя после возврата на карту."""
        self.current_battle_context = {}
    def _build_battle_team_with_placeholders(
        self,
        team: str,
        units: Optional[List[Dict]],
    ) -> List[Dict]:
        """Нормализует команду до шести позиций, добавляя placeholder-юнитов.

        BattleEnv ожидает фиксированные слоты: RED занимает позиции 1-6,
        BLUE занимает позиции 7-12. Метод отбрасывает некорректные/дублирующиеся
        позиции и гарантирует, что результат всегда содержит полный набор слотов.
        """
        normalized_team = "red" if str(team).lower() == "red" else "blue"
        positions = range(1, 7) if normalized_team == "red" else range(7, 13)
        units_by_position: Dict[int, Dict] = {}

        for raw_unit in units or []:
            if not isinstance(raw_unit, dict):
                continue
            try:
                position = int(raw_unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                continue
            if position not in positions or position in units_by_position:
                continue
            normalized_unit = deepcopy(raw_unit)
            normalized_unit["team"] = normalized_team
            normalized_unit["position"] = int(position)
            units_by_position[position] = normalized_unit

        return [
            units_by_position.get(int(position), placeholder_unit(normalized_team, int(position)))
            for position in positions
        ]
    def _build_summoned_blue_team(
        self,
        summon_unit_name: str,
    ) -> Optional[Tuple[List[Dict], Dict]]:
        """Создает временную BLUE-команду из одного призванного юнита.

        Используется для summon_spell-контекста. Команда не должна влиять на
        persistent BLUE-состояние героя и его экипировку.
        """
        unit_data = self._find_unit_data_by_name(str(summon_unit_name or "").strip())
        if not isinstance(unit_data, dict):
            return None

        template_unit = self._build_unit_from_data(unit_data, team="blue", position=7)
        summon_position = 7 if template_unit.get("stand") == "ahead" else 10
        template_unit["team"] = "blue"
        template_unit["position"] = int(summon_position)
        template_unit["stand"] = "ahead" if summon_position in (7, 8, 9) else "behind"
        template_unit["hero"] = False
        template_unit["needaunit"] = 0
        template_unit["exp_current"] = 0
        template_unit["exp_required"] = 0
        template_unit["next_level_exp"] = 0

        blue_team = self._build_battle_team_with_placeholders("blue", [template_unit])
        summoned_unit = next(
            unit for unit in blue_team if int(unit.get("position", -1) or -1) == int(summon_position)
        )
        return blue_team, summoned_unit
    def _save_enemy_state_from_battle(self, enemy_id: Optional[int]) -> None:
        """Сохраняет persistent-состояние RED-команды после незавершенного боя.

        Метод нужен для таймаутов и отступлений: враг должен остаться на карте
        с текущим HP, опытом и level-up-прогрессом, но без временных статусов боя.
        """
        if self.battle_env is None:
            return

        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return

        original_team = self._build_battle_team_with_placeholders(
            "red",
            self._get_enemy_team_state(normalized_enemy_id),
        )
        original_by_position = {
            int(unit.get("position", -1) or -1): deepcopy(unit)
            for unit in original_team
        }
        battle_by_position = {
            int(unit.get("position", -1) or -1): deepcopy(unit)
            for unit in self.battle_env.combined
            if unit.get("team") == "red"
            and not bool(unit.get("Summoned"))
            and 1 <= int(unit.get("position", -1) or -1) <= 6
        }
        restore_transformed = getattr(
            self.battle_env, "_restore_transformed_unit", None
        )
        if callable(restore_transformed):
            for battle_unit in battle_by_position.values():
                restore_transformed(battle_unit)

        saved_team: List[Dict] = []
        for position in range(1, 7):
            original_unit = deepcopy(original_by_position.get(position, placeholder_unit("red", position)))
            battle_unit = battle_by_position.get(position)
            if battle_unit is not None and not self._is_empty_enemy_unit(original_unit):
                current_hp = self._unit_current_hp(battle_unit)
                original_unit["health"] = float(current_hp)
                original_unit["hp"] = float(current_hp)
                original_unit["Level"] = int(
                    battle_unit.get("Level", original_unit.get("Level", 0)) or 0
                )
                original_unit["exp_current"] = float(
                    battle_unit.get("exp_current", original_unit.get("exp_current", 0)) or 0.0
                )
                original_unit["exp_required"] = battle_unit.get(
                    "exp_required",
                    original_unit.get("exp_required", 0),
                )
                original_unit["next_level_exp"] = int(
                    battle_unit.get(
                        "next_level_exp",
                        original_unit.get("next_level_exp", 0),
                    )
                    or 0
                )
                original_unit["hero"] = bool(
                    battle_unit.get("hero", original_unit.get("hero", False))
                )
                original_unit["needaunit"] = int(
                    battle_unit.get("needaunit", original_unit.get("needaunit", 0)) or 0
                )
                original_unit["initiative"] = (
                    int(original_unit.get("initiative_base", original_unit.get("initiative", 0)) or 0)
                    if current_hp > 0.0
                    else 0
                )

            original_unit["defense"] = 0
            original_unit["waited"] = 0
            original_unit["paralyzed"] = 0
            original_unit["long_paralyzed"] = 0
            original_unit["running_away"] = 0
            original_unit["transformed"] = 0
            original_unit["poison_turns_left"] = 0
            original_unit["poison_damage_per_tick"] = 0
            original_unit["burn_turns_left"] = 0
            original_unit["burn_damage_per_tick"] = 0
            original_unit["burn_source_lord_pos"] = None
            original_unit["uran_turns_left"] = 0
            original_unit["uran_damage_per_tick"] = 0
            original_unit["resilience_used_types"] = []
            original_unit["powerup"] = 0
            original_unit["bonusturn"] = 0
            original_unit.pop("teamated", None)
            original_unit.pop("hermited", None)
            original_unit.pop("settlement_armor_bonus", None)
            original_unit.pop("settlement_level", None)
            saved_team.append(original_unit)

        self.enemy_team_states[normalized_enemy_id] = self._build_battle_team_with_placeholders(
            "red",
            saved_team,
        )
        if normalized_enemy_id in self.enemy_map_spell_effects:
            self._recompute_enemy_stack_spell_effects(normalized_enemy_id)
    def _build_unit_from_data(self, entry: Dict, team: str, position: int) -> Dict:
        """Преобразует запись UNIT_DATA в словарь юнита для BattleEnv."""
        def _entry_get(*keys, default=0):
            """Достает поле по одному из поддерживаемых ключей.

            В данных встречаются нормальные русские ключи, legacy mojibake-ключи
            и англоязычные fallback-и. Helper скрывает эту совместимость от
            остального метода.
            """
            for key in keys:
                if key in entry:
                    return entry.get(key)
            return default

        def _entry_number(*keys, default=0):
            value = _entry_get(*keys, default=default)
            try:
                return int(round(float(value or 0)))
            except (TypeError, ValueError):
                return int(default or 0)

        new_unit = map_unit_to_battle(entry, team, position)
        new_unit["exp_kill"] = _entry_get(
            "\u043e\u043f\u044b\u0442 \u0443\u0431\u0438\u0439\u0441\u0442\u0432\u0430",
            "\u0420\u0455\u0420\u0457\u0421\u2039\u0421\u201a \u0421\u0453\u0420\xb1\u0420\u0451\u0420\u2116\u0421\u0403\u0421\u201a\u0420\u0406\u0420\xb0",
            "exp_kill",
            default=0,
        )
        exp_required_raw = _entry_get(
            "\u043d\u0443\u0436\u043d\u044b\u0439 \u043e\u043f\u044b\u0442",
            "\u0420\u0405\u0421\u0453\u0420\xb6\u0420\u0405\u0421\u2039\u0420\u2116 \u0420\u0455\u0420\u0457\u0421\u2039\u0421\u201a",
            "exp_required",
            default=0,
        )
        if isinstance(exp_required_raw, str) and exp_required_raw.strip().lower() == "max":
            new_unit["exp_required"] = 0
        else:
            new_unit["exp_required"] = _entry_number(
                "\u043d\u0443\u0436\u043d\u044b\u0439 \u043e\u043f\u044b\u0442",
                "\u0420\u0405\u0421\u0453\u0420\xb6\u0420\u0405\u0421\u2039\u0420\u2116 \u0420\u0455\u0420\u0457\u0421\u2039\u0421\u201a",
                "exp_required",
                default=0,
            )
        new_unit["exp_current"] = _entry_get(
            "\u0442\u0435\u043a\u0443\u0449\u0438\u0439 \u043e\u043f\u044b\u0442",
            "\u0421\u201a\u0420\xb5\u0420\u0454\u0421\u0453\u0421\u2030\u0420\u0451\u0420\u2116 \u0420\u0455\u0420\u0457\u0421\u2039\u0421\u201a",
            "exp_current",
            default=0,
        )
        try:
            new_unit["next_level_exp"] = int(
                round(
                    float(
                        _entry_get(
                            "\u0441\u043b\u0435\u0434\u0443\u0440\u043e\u0432\u0435\u043d\u044c",
                            "next_level_exp",
                            default=0,
                        )
                    )
                )
            )
        except (TypeError, ValueError):
            new_unit["next_level_exp"] = 0
        turns_raw = _entry_get(
            "\u043f\u0440\u0435\u0432\u0440\u0430\u0449\u0430\u0435\u0442\u0441\u044f \u0432",
            "\u0420\u0457\u0421\u0402\u0420\xb5\u0420\u0406\u0421\u0402\u0420\xb0\u0421\u2030\u0420\xb0\u0420\xb5\u0421\u201a\u0421\u0403\u0421\u040f \u0420\u0406",
            "turns_into",
            default=[],
        )
        if isinstance(turns_raw, list):
            turns_into = list(turns_raw)
        elif turns_raw:
            turns_into = [turns_raw]
        else:
            turns_into = []
        new_unit["turns_into"] = turns_into
        new_unit["hp"] = new_unit.get("health", 0)
        new_unit["maxhp"] = new_unit.get("max_health", 0)
        return new_unit
    def _replace_blue_unit(self, upgraded: Dict) -> None:
        """Заменяет юнита в persistent BLUE-состоянии после апгрейда."""
        if self.blue_team_state is None:
            return
        pos = upgraded.get("position")
        if pos is None:
            return
        for idx, unit in enumerate(self.blue_team_state):
            if int(unit.get("position", -1)) == int(pos):
                self.blue_team_state[idx] = deepcopy(upgraded)
                self._mark_equipment_dirty()
                self._sync_moves_per_turn_with_hero(units=self.blue_team_state)
                self._refresh_campaign_equipment_effects(log=False)
                return
    def _reset_last_upgrade_reward_tracking(self) -> None:
        self._last_upgrade_tier_sum = 0
        self._last_upgrade_max_tier = 0
        self._last_upgrade_tier3_count = 0
        self._last_hero_levelup_count = 0
        self._last_hero_levelup_max_level = 0
        self._last_hero_levelup_reward = 0.0
        self._last_upgrade_reward = 0.0

    def _unit_upgrade_reward_for_tier(self, target_tier: int) -> float:
        target_tier = max(0, int(target_tier))
        reward = float(self.reward_unit_upgrade) + (
            float(target_tier) * float(getattr(self, "reward_unit_tier_bonus", 0.0) or 0.0)
        )
        if target_tier == 3:
            reward *= float(getattr(self, "reward_unit_tier3_multiplier", 1.0))
        return float(reward)

    def _record_hero_levelup_reward(self, unit: Dict) -> None:
        if not self._is_hero_unit(unit):
            return
        try:
            target_level = int(unit.get("Level", 0) or 0)
        except (TypeError, ValueError):
            return
        if not (2 <= target_level <= 4):
            return
        reward = self._unit_upgrade_reward_for_tier(target_level)
        self._last_hero_levelup_count += 1
        self._last_hero_levelup_max_level = max(
            self._last_hero_levelup_max_level, target_level
        )
        self._last_hero_levelup_reward += reward
        self._last_upgrade_reward += reward

    def _log_turns_into_levelups(self) -> int:
        """Применяет превращения BLUE-юнитов после level-up и возвращает их число.

        BattleEnv только сообщает, какие имена получили уровень. CampaignEnv
        дополнительно проверяет, построено ли здание, открывающее целевой юнит
        из turns_into, и уже после этого заменяет боевую и persistent-запись.
        """
        self._reset_last_upgrade_reward_tracking()
        if self.battle_env is None:
            return 0
        levelup_names = getattr(self.battle_env, "last_levelups", []) or []
        if not levelup_names:
            return 0

        name_to_units: Dict[str, List[Dict]] = {}
        for unit in getattr(self.battle_env, "combined", []) or []:
            if unit.get("team") != "blue":
                continue
            name = str(unit.get("name", "") or "").strip()
            if not name:
                continue
            name_to_units.setdefault(name, []).append(unit)

        upgraded_count = 0
        # Сумма и максимум тиров (уровней) юнитов, достигнутых апгрейдом — для бонуса и отчёта.
        for name in levelup_names:
            unit_data = self._find_unit_data_by_name(name)
            capital_value = unit_data.get("\u0441\u0442\u043e\u043b\u0438\u0446\u0430") if unit_data else None
            buildings = self._get_buildings_for_capital(capital_value)

            units = name_to_units.get(name)
            if not units:
                continue
            unit = units.pop(0)
            pos = unit.get("position")
            if pos is None:
                unit_label = name
            else:
                unit_label = f"{name} (pos {pos})"
            self._record_hero_levelup_reward(unit)
            turns_into = unit.get("turns_into", [])
            if not isinstance(turns_into, list):
                turns_into = [turns_into]

            for target in turns_into:
                target_name = str(target).strip()
                if not target_name:
                    continue
                building = None
                for entry in buildings.values():
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("unit") == target_name:
                        building = entry
                        break
                if building is None:
                    continue
                built_flag = building.get("Build", building.get("built", 0))
                if int(built_flag or 0) == 1:
                    self._log(f'Unit "{unit_label}" upgraded to "{target_name}"')
                    unit_data = self._find_unit_data_by_name(target_name)
                    if unit_data is not None:
                        upgraded_unit = self._build_unit_from_data(
                            unit_data,
                            unit.get("team", "blue"),
                            unit.get("position", pos),
                        )
                        self._reapply_persistent_elixir_bonuses_to_promoted_unit(
                            source_unit=unit,
                            upgraded_unit=upgraded_unit,
                        )
                        unit.clear()
                        unit.update(deepcopy(upgraded_unit))
                        self._replace_blue_unit(upgraded_unit)
                        upgraded_count += 1
                        target_tier = int(
                            float(
                                unit_data.get("уровень", 0)
                                or upgraded_unit.get("Level", 0)
                                or 0
                            )
                        )
                        self._last_upgrade_tier_sum += max(0, target_tier)
                        self._last_upgrade_max_tier = max(
                            self._last_upgrade_max_tier, target_tier
                        )
                        if target_tier == 3:
                            self._last_upgrade_tier3_count += 1
                        self._last_upgrade_reward += self._unit_upgrade_reward_for_tier(
                            target_tier
                        )
                    break

        return int(upgraded_count)
    def _heal_blue_team(self, heal_percent: float = 0.05, bonus_percent: float = 0.0) -> int:
        """
        Восстанавливает HP раненым живым юнитам BLUE-команды.
        
        Args:
            heal_percent: Доля от max_health для восстановления (0.05 = 5%)
            bonus_percent: Дополнительная доля от max_health для восстановления.
             
        Returns:
            Количество юнитов, которым восстановлено HP
        """
        state = self._get_blue_state()
        
        healed_count = 0
        for unit in state:
            hp = self._unit_current_hp(unit)
            max_hp = self._unit_max_hp(unit, fallback=hp)
            
            # Юнит жив и ранен
            if hp > 0 and hp < max_hp:
                heal_amount = max_hp * (
                    float(heal_percent or 0.0) + max(0.0, float(bonus_percent or 0.0))
                )
                new_hp = min(max_hp, hp + heal_amount)
                unit["hp"] = new_hp
                unit["health"] = new_hp
                healed_count += 1
                
        return healed_count
    def _heal_unit_at_position(
        self,
        position: int,
        heal_amount: float,
        source_label: str = "Бутыль лечения",
    ) -> Tuple[float, Optional[str]]:
        """
        Лечит конкретную позицию BLUE-команды.

        Возвращает фактический объем лечения и имя юнита. Если позиция пустая,
        юнит мертв или уже имеет полный HP, возвращает ``(0.0, None)``.
        """
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = self._unit_current_hp(unit)
            max_hp = self._unit_max_hp(unit, fallback=hp)

            if max_hp <= 0 or hp <= 0 or hp >= max_hp:
                return 0.0, None

            new_hp = min(max_hp, hp + heal_amount)
            healed = new_hp - hp
            unit["hp"] = new_hp
            unit["health"] = new_hp

            name = unit.get("name", f"pos_{position}")
            self._log(
                f"{source_label}: {name} (pos {position}) HP {hp:.1f} -> {new_hp:.1f} (+{healed:.1f})"
            )
            return healed, name

        return 0.0, None
    def _get_blue_state(self) -> List[Dict]:
        """Возвращает и при необходимости инициализирует persistent BLUE-состояние."""
        if self.blue_team_state is None:
            self.blue_team_state = self._create_starting_blue_team()
            self._mark_equipment_dirty()
            self._sync_hero_progression_flags(self.blue_team_state)
            self._sync_moves_per_turn_with_hero(units=self.blue_team_state, refill=True)
        else:
            self._sync_hero_progression_flags(self.blue_team_state)
        return self.blue_team_state
    def _battle_grid_move_cost(self) -> int:
        """Возвращает стоимость входа в бой в очках перемещения по лимиту героя."""
        move_cap = max(0, int(self.moves_per_turn or 0))
        return max(0, (move_cap + 1) // 2)

    def _apply_battle_grid_steps(self, extra_steps: Optional[int] = None) -> int:
        """Списывает очки перемещения за вход в бой и возвращает фактически списанное."""
        spent_moves = self._battle_grid_move_cost() if extra_steps is None else int(extra_steps)
        if spent_moves <= 0:
            return 0
        moves_before = int(self.moves)
        self._spend_moves(spent_moves)
        return max(0, moves_before - int(self.moves))
