# campaign_env_battle.py
"""CampaignBattleMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignBattleMixin:
    def _step_battle(self, action: int):
        """Шаг в режиме боя."""
        if self.battle_env is None:
            # Защита от некорректного состояния
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
            # safety: action space CampaignEnv > BattleEnv, clamp to valid range
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
                if (
                    battle_info.get("battle_hero_item_action")
                    and battle_info.get("battle_hero_item_consumed")
                    and consumed_item_name
                ):
                    self._consume_battle_equipable_item(consumed_item_name)
                    if battle_info.get("battle_hero_item_applied"):
                        self.battle_items_used_total += 1
                        battle_item_use_reward = float(self.reward_battle_item_use)
                    else:
                        battle_item_use_reward = 0.0
                else:
                    battle_item_use_reward = 0.0
                equipped_items = battle_info.get("equipped_hero_items")
                if isinstance(equipped_items, list) and str(
                    self.current_battle_context.get("kind", "hero") or "hero"
                ) != "summon_spell":
                    self.equipped_hero_items = list(equipped_items)
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
                upgrade_reward = float(upgrade_count) * self.reward_unit_upgrade
                reward += upgrade_reward
                info["unit_upgrades"] = int(upgrade_count)
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

                if newly_captured_objective_cities:
                    final_objective_reward = self._compute_final_objective_reward(
                        len(newly_captured_objective_cities),
                        captured_before=objective_cities_captured_before,
                    )
                    reward += final_objective_reward
                    info["captured_objective_cities"] = list(newly_captured_objective_cities)
                    info["final_objective_reward"] = float(final_objective_reward)
                    for city_name in newly_captured_objective_cities:
                        self._log(f"=== ЗАХВАЧЕН ГОРОД {city_name}: НАГРАДА {final_objective_reward:g} ===")

                if self._all_objective_cities_captured():
                    self._log("=== ЗАХВАЧЕНЫ ВСЕ ЦЕЛЕВЫЕ ГОРОДА: ПОБЕДА В КАМПАНИИ ===")
                    grid_obs = self._get_grid_obs()
                    info["campaign_result"] = "victory"
                    info["campaign_victory_reason"] = "objective_cities_cleared"
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
                # Поражение: конец эпизода
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
                    self._log_turns_into_levelups()
                return self._build_obs(battle_obs=obs), reward, True, False, info

        info["battle_ongoing"] = True
        return self._build_obs(battle_obs=obs), reward, terminated, truncated, info
    def _restore_agent_to_battle_origin(self) -> Tuple[int, int]:
        """Возвращает агента на клетку, с которой был инициирован текущий бой."""
        if self.battle_origin_pos is None:
            return tuple(self.grid_env.agent_pos)
        origin = (int(self.battle_origin_pos[0]), int(self.battle_origin_pos[1]))
        self.grid_env.agent_pos = origin
        self.battle_origin_pos = None
        return origin
    def _clear_battle_origin(self) -> None:
        self.battle_origin_pos = None
    @staticmethod
    def _normalize_armor_value(value: object) -> int:
        try:
            return max(0, int(round(float(value or 0))))
        except (TypeError, ValueError):
            return 0
    def _resolve_heal_tile_context(
        self,
        position: Optional[Tuple[int, int]],
    ) -> tuple[int, float, str, Optional[int]]:
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
        """Возвращает процентный бонус лечения от отдыха на клетке города/столицы."""
        if position is None:
            return 0.0, "", None

        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return 0.0, "", None

        if normalized_position == tuple(self.CASTLE_POS):
            return float(CAPITAL_HEAL_TILE_ARMOR_BONUS) / 100.0, "capital", "capital"

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
        """Возвращает базовый процент лечения от отдыха на текущей клетке."""
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
        """Возвращает бонус регенерации врага на клетке города/столицы."""
        if position is None:
            return 0.0

        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return 0.0

        if normalized_position == tuple(self.empire_territory_source_tile):
            return float(CAPITAL_HEAL_TILE_ARMOR_BONUS) / 100.0

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
        if enemy_pos is not None and tuple(enemy_pos) == tuple(self.CASTLE_POS):
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
        bonus, _rest_bonus, source, settlement_level = self._resolve_heal_tile_context(
            self.battle_origin_pos
        )
        if bonus <= 0:
            return

        affected_units = 0
        for unit in blue_team:
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
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
        """Инициализирует бой с конкретной RED командой."""
        red_team_state = self._get_enemy_team_state(enemy_id)
        if red_team_state:
            red_team = self._build_battle_team_with_placeholders("red", red_team_state)
        else:
            red_team = self._build_battle_team_with_placeholders(
                "red",
                ENEMY_CONFIGS.get(enemy_id, ENEMY_CONFIGS[1]),
            )
        self._apply_settlement_defender_armor_bonus(enemy_id, red_team)

        # Восстанавливаем состояние BLUE или берём дефолтное.
        if blue_team is not None:
            prepared_blue_team = self._build_battle_team_with_placeholders("blue", blue_team)
            equipped_hero_items: List[Optional[str]] = [None] * int(self.BATTLE_EQUIP_SLOTS)
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
            self._apply_equipped_artifact_effects(prepared_blue_team)
            self._apply_equipped_banner_effects(prepared_blue_team)
            self._sync_equipped_hero_items()
            equipped_hero_items = list(self.equipped_hero_items)
            hero_item_effects = dict(self._battle_item_effect_definitions())
            self._log("Загружено сохранённое состояние BLUE команды")
        else:
            prepared_blue_team = self._build_battle_team_with_placeholders("blue", UNITS_BLUE)
            self._clear_equipped_banner_effects(prepared_blue_team)
            self._clear_equipped_artifact_effects(prepared_blue_team)
            self._apply_hero_heal_tile_armor_bonus(prepared_blue_team)
            self._apply_active_blue_potion_effects(prepared_blue_team)
            self._apply_active_blue_support_spell_effects(prepared_blue_team)
            self._apply_equipped_artifact_effects(prepared_blue_team)
            self._apply_equipped_banner_effects(prepared_blue_team)
            self._sync_equipped_hero_items()
            equipped_hero_items = list(self.equipped_hero_items)
            hero_item_effects = dict(self._battle_item_effect_definitions())
            self._log("Используется дефолтная BLUE команда")

        self.battle_env = BattleEnv(
            reward_win=self.battle_reward_win,
            reward_loss=self.battle_reward_loss,
            reward_step=self.battle_reward_step,
            log_enabled=self.log_enabled,
        )
        self.battle_env.equipped_hero_items = list(equipped_hero_items)
        self.battle_env.hero_item_effects = dict(hero_item_effects)

        # Инициализируем бой с кастомными командами
        self.battle_env._init_with_custom_teams(red_team, prepared_blue_team)
    def _save_blue_state(self):
        """
        Сохраняет состояние BLUE команды после победы.
        HP НЕ восстанавливается, погибшие НЕ воскрешаются.
        Только сбрасываются временные эффекты (armor boost, paralysis, poison и т.д.).
        """
        if self.battle_env is None:
            return

        # Сохраняем только базовые позиции BLUE (без призванных юнитов)
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
            
            # СОХРАНЯЕМ ТЕКУЩЕЕ HP (не восстанавливаем!)
            current_hp = max(
                0.0,
                float(restored_unit.get("health", 0) or restored_unit.get("hp", 0)),
            )
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
            restored_unit["powerup"] = 0
            restored_unit["bonusturn"] = 0
            restored_unit.pop("resilience_used_types", None)
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
                        restored_unit.get("max_health", restored_unit.get("maxhp", 0)),
                    )
                    or 0.0
                )
                health_bonus = float(
                    restored_unit.get("campaign_map_spell_health_bonus", 0) or 0.0
                )
                current_hp = max(0.0, float(restored_unit.get("health", 0) or restored_unit.get("hp", 0) or 0.0))
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
                restored_unit["resistance"] = [
                    res
                    for res in (restored_unit.get("resistance") or [])
                    if str(res) not in set(added_spell_resistance)
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
            if (
                position in self.active_strength_potion_positions
                or position in self.active_energy_elixir_positions
            ):
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
            if position in self.active_haste_elixir_positions:
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
            added_resistance = [
                str(res)
                for res in (restored_unit.get("campaign_potion_added_resistance") or [])
                if str(res)
            ]
            if added_resistance:
                restored_unit["resistance"] = [
                    res
                    for res in (restored_unit.get("resistance") or [])
                    if str(res) not in set(added_resistance)
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
        self._sync_hero_progression_flags(self.blue_team_state)
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state, grant_delta=True)
        self._refresh_campaign_equipment_effects(log=False)
        self._log("Состояние BLUE команды сохранено (HP не восстановлено)")
    def _find_unit_data_by_name(self, name: str) -> Optional[Dict]:
        for entry in UNIT_DATA:
            # Prefer canonical Russian key, keep mojibake fallback for legacy data dumps.
            entry_name = entry.get("\u043a\u0442\u043e")
            if entry_name is None:
                entry_name = entry.get("\u0420\u0454\u0421\u201a\u0420\u0455")
            if entry_name == name:
                return entry
        return None
    def _clear_current_battle_context(self) -> None:
        self.current_battle_context = {}
    def _build_battle_team_with_placeholders(
        self,
        team: str,
        units: Optional[List[Dict]],
    ) -> List[Dict]:
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
                current_hp = max(
                    0.0,
                    float(battle_unit.get("health", 0) or battle_unit.get("hp", 0) or 0.0),
                )
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
        def _entry_get(*keys, default=0):
            for key in keys:
                if key in entry:
                    return entry.get(key)
            return default

        new_unit = map_unit_to_battle(entry, team, position)
        new_unit["exp_kill"] = _entry_get(
            "\u043e\u043f\u044b\u0442 \u0443\u0431\u0438\u0439\u0441\u0442\u0432\u0430",
            "\u0420\u0455\u0420\u0457\u0421\u2039\u0421\u201a \u0421\u0453\u0420\xb1\u0420\u0451\u0420\u2116\u0421\u0403\u0421\u201a\u0420\u0406\u0420\xb0",
            "exp_kill",
            default=0,
        )
        new_unit["exp_required"] = _entry_get(
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
        if self.blue_team_state is None:
            return
        pos = upgraded.get("position")
        if pos is None:
            return
        for idx, unit in enumerate(self.blue_team_state):
            if int(unit.get("position", -1)) == int(pos):
                self.blue_team_state[idx] = deepcopy(upgraded)
                self._sync_moves_per_turn_with_hero(units=self.blue_team_state)
                self._refresh_campaign_equipment_effects(log=False)
                return
    def _log_turns_into_levelups(self) -> int:
        """Applies BLUE unit upgrades after level-up checks and returns applied count."""
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
                    break

        return int(upgraded_count)
    def _heal_blue_team(self, heal_percent: float = 0.05, bonus_percent: float = 0.0) -> int:
        """
        Восстанавливает HP раненым юнитам BLUE команды.
        
        Args:
            heal_percent: Доля от max_health для восстановления (0.05 = 5%)
            bonus_percent: Дополнительная доля от max_health для восстановления.
             
        Returns:
            Количество юнитов, которым восстановлено HP
        """
        state = self._get_blue_state()
        
        healed_count = 0
        for unit in state:
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            
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
        Лечит конкретную позицию BLUE команды, возвращает фактический объём лечения и имя юнита.
        """
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)

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
    def _missing_hp_at_position(self, position: int) -> float:
        """Возвращает недостающее здоровье живого юнита BLUE на позиции или 0.0."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            if max_hp <= 0 or hp <= 0 or hp >= max_hp:
                return 0.0
            return max_hp - hp
        return 0.0
    def _get_blue_state(self) -> List[Dict]:
        """Возвращает (и при необходимости инициализирует) сохранённое состояние BLUE."""
        if self.blue_team_state is None:
            self.blue_team_state = deepcopy(UNITS_BLUE)
            self._sync_hero_progression_flags(self.blue_team_state)
            self._sync_moves_per_turn_with_hero(units=self.blue_team_state, refill=True)
        else:
            self._sync_hero_progression_flags(self.blue_team_state)
        return self.blue_team_state
    def _battle_grid_move_cost(self) -> int:
        """Returns the combat entry cost based on the hero's full movement cap."""
        move_cap = max(0, int(self.moves_per_turn or 0))
        return max(0, (move_cap + 1) // 2)

    def _apply_battle_grid_steps(self, extra_steps: Optional[int] = None) -> int:
        """Списывает дополнительные очки перемещения из-за входа в бой."""
        spent_moves = self._battle_grid_move_cost() if extra_steps is None else int(extra_steps)
        if spent_moves <= 0:
            return 0
        moves_before = int(self.moves)
        self._spend_moves(spent_moves)
        return max(0, moves_before - int(self.moves))
