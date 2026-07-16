# campaign_env_territory.py
"""Территориальная логика CampaignEnv.

Миксин отвечает за рост территорий фракций, захват поселений и финальных городов,
доходы от уже захваченных клеток, награды за руины и служебные кеши для grid observation.
"""

from __future__ import annotations

from campaign_env_data import *


class CampaignTerritoryMixin:
    """Методы CampaignEnv, связанные с владением клетками глобальной карты."""

    TERRITORY_PATH_DISTANCE_CACHE_SIZE = 64

    @classmethod
    def _normalize_campaign_objective(cls, value: object) -> str:
        if isinstance(value, bool):
            return cls.CAMPAIGN_OBJECTIVE_DRAGON if value else cls.CAMPAIGN_OBJECTIVE_CITIES
        raw = str(value or cls.CAMPAIGN_OBJECTIVE_CITIES).strip().lower()
        raw = raw.replace("-", "_")
        aliases = {
            "city": cls.CAMPAIGN_OBJECTIVE_CITIES,
            "cities": cls.CAMPAIGN_OBJECTIVE_CITIES,
            "objective_city": cls.CAMPAIGN_OBJECTIVE_CITIES,
            "objective_cities": cls.CAMPAIGN_OBJECTIVE_CITIES,
            "dragon": cls.CAMPAIGN_OBJECTIVE_DRAGON,
            "green_dragon": cls.CAMPAIGN_OBJECTIVE_DRAGON,
            "blue_dragon": cls.CAMPAIGN_OBJECTIVE_BLUE_DRAGON,
            "water_dragon": cls.CAMPAIGN_OBJECTIVE_BLUE_DRAGON,
            "orc": cls.CAMPAIGN_OBJECTIVE_ORC,
            "orc_duel": cls.CAMPAIGN_OBJECTIVE_ORC,
            "build_all": cls.CAMPAIGN_OBJECTIVE_BUILD_ALL,
            "builder": cls.CAMPAIGN_OBJECTIVE_BUILD_ALL,
            "buildings": cls.CAMPAIGN_OBJECTIVE_BUILD_ALL,
            "all_enemies": cls.CAMPAIGN_OBJECTIVE_ALL_ENEMIES,
            "enemies": cls.CAMPAIGN_OBJECTIVE_ALL_ENEMIES,
            "clear_map": cls.CAMPAIGN_OBJECTIVE_ALL_ENEMIES,
            "target_enemy": cls.CAMPAIGN_OBJECTIVE_TARGET_ENEMY,
            "siege": cls.CAMPAIGN_OBJECTIVE_TARGET_ENEMY,
            "siege_army": cls.CAMPAIGN_OBJECTIVE_TARGET_ENEMY,
            "waves": cls.CAMPAIGN_OBJECTIVE_WAVES,
            "wave": cls.CAMPAIGN_OBJECTIVE_WAVES,
            "last_stand": cls.CAMPAIGN_OBJECTIVE_WAVES,
            "survival_waves": cls.CAMPAIGN_OBJECTIVE_WAVES,
        }
        if raw in aliases:
            return aliases[raw]
        raise ValueError(
            f"Unsupported campaign_objective={value!r}; "
            "expected 'cities', 'dragon', 'blue_dragon', 'orc', 'build_all', "
            "'all_enemies', 'target_enemy' or 'waves'"
        )

    def _campaign_objective_is_cities(self) -> bool:
        return (
            self._normalize_campaign_objective(getattr(self, "campaign_objective", "cities"))
            == self.CAMPAIGN_OBJECTIVE_CITIES
        )

    def _campaign_objective_is_dragon(self) -> bool:
        # «Охотничьи» цели: blue_dragon и orc полностью повторяют логику dragon,
        # меняется только целевой враг (self.OBJECTIVE_ENEMY_ID из конфига карты).
        return self._normalize_campaign_objective(
            getattr(self, "campaign_objective", "cities")
        ) in (
            self.CAMPAIGN_OBJECTIVE_DRAGON,
            self.CAMPAIGN_OBJECTIVE_BLUE_DRAGON,
            self.CAMPAIGN_OBJECTIVE_ORC,
        )

    def _campaign_objective_is_blue_dragon(self) -> bool:
        return (
            self._normalize_campaign_objective(getattr(self, "campaign_objective", "cities"))
            == self.CAMPAIGN_OBJECTIVE_BLUE_DRAGON
        )

    def _campaign_objective_is_orc(self) -> bool:
        return (
            self._normalize_campaign_objective(getattr(self, "campaign_objective", "cities"))
            == self.CAMPAIGN_OBJECTIVE_ORC
        )

    def _campaign_objective_is_build_all(self) -> bool:
        return (
            self._normalize_campaign_objective(getattr(self, "campaign_objective", "cities"))
            == self.CAMPAIGN_OBJECTIVE_BUILD_ALL
        )

    def _campaign_objective_is_all_enemies(self) -> bool:
        return (
            self._normalize_campaign_objective(getattr(self, "campaign_objective", "cities"))
            == self.CAMPAIGN_OBJECTIVE_ALL_ENEMIES
        )

    def _campaign_objective_is_target_enemy(self) -> bool:
        return (
            self._normalize_campaign_objective(getattr(self, "campaign_objective", "cities"))
            == self.CAMPAIGN_OBJECTIVE_TARGET_ENEMY
        )

    def _campaign_objective_is_waves(self) -> bool:
        return (
            self._normalize_campaign_objective(getattr(self, "campaign_objective", "cities"))
            == self.CAMPAIGN_OBJECTIVE_WAVES
        )

    def _is_green_dragon_objective_enemy(self, enemy_id: Optional[int]) -> bool:
        if not self._campaign_objective_is_dragon():
            return False
        try:
            return int(enemy_id) == int(
                getattr(self, "OBJECTIVE_ENEMY_ID", self.GREEN_DRAGON_OBJECTIVE_ENEMY_ID)
            )
        except (TypeError, ValueError):
            return False

    def _is_target_objective_enemy(self, enemy_id: Optional[int]) -> bool:
        if not self._campaign_objective_is_target_enemy():
            return False
        try:
            return int(enemy_id) == int(getattr(self, "OBJECTIVE_ENEMY_ID"))
        except (TypeError, ValueError):
            return False

    def _validate_objective_for_map(self) -> None:
        """Проверяет, что выбранная цель кампании осуществима на выбранной карте."""
        map_config = getattr(self, "_map", None)
        map_enemy_ids = set(map_config.enemy_positions().keys()) if map_config else set()
        map_label = getattr(map_config, "name", "default")
        objective = self.campaign_objective
        if objective == self.CAMPAIGN_OBJECTIVE_CITIES:
            if not self.FINAL_OBJECTIVE_CITIES:
                raise ValueError(
                    f"campaign_objective='cities' is not supported on map '{map_label}': "
                    "the map has no objective cities"
                )
        elif objective in (self.CAMPAIGN_OBJECTIVE_DRAGON, self.CAMPAIGN_OBJECTIVE_BLUE_DRAGON):
            if int(self.GREEN_DRAGON_OBJECTIVE_ENEMY_ID) not in map_enemy_ids:
                raise ValueError(
                    f"campaign_objective={objective!r} is not supported on map '{map_label}': "
                    f"enemy {self.GREEN_DRAGON_OBJECTIVE_ENEMY_ID} (dragon) is not on the map"
                )
        elif objective == self.CAMPAIGN_OBJECTIVE_ORC:
            objective_enemy_id = getattr(map_config, "objective_enemy_id", None)
            if (
                map_config is None
                or map_config.default_objective != self.CAMPAIGN_OBJECTIVE_ORC
                or objective_enemy_id is None
                or int(objective_enemy_id) not in map_enemy_ids
            ):
                raise ValueError(
                    f"campaign_objective='orc' is not supported on map '{map_label}'"
                )
        elif objective == self.CAMPAIGN_OBJECTIVE_BUILD_ALL:
            # Строительная цель не зависит от врагов, но включается только на
            # картах, которые явно объявили её целью по умолчанию.
            if (
                map_config is None
                or map_config.default_objective != self.CAMPAIGN_OBJECTIVE_BUILD_ALL
            ):
                raise ValueError(
                    f"campaign_objective='build_all' is not supported on map '{map_label}'"
                )
        elif objective == self.CAMPAIGN_OBJECTIVE_ALL_ENEMIES:
            if (
                map_config is None
                or map_config.default_objective != self.CAMPAIGN_OBJECTIVE_ALL_ENEMIES
                or not map_enemy_ids
            ):
                raise ValueError(
                    f"campaign_objective='all_enemies' is not supported on map '{map_label}'"
                )
        elif objective == self.CAMPAIGN_OBJECTIVE_TARGET_ENEMY:
            objective_enemy_id = getattr(map_config, "objective_enemy_id", None)
            if (
                map_config is None
                or map_config.default_objective != self.CAMPAIGN_OBJECTIVE_TARGET_ENEMY
                or objective_enemy_id is None
                or int(objective_enemy_id) not in map_enemy_ids
            ):
                raise ValueError(
                    f"campaign_objective='target_enemy' is not supported on map '{map_label}'"
                )
        elif objective == self.CAMPAIGN_OBJECTIVE_WAVES:
            wave_configs = self._scheduled_enemy_wave_configs()
            wave_enemy_ids = {
                int(config["enemy_id"])
                for config in wave_configs
            }
            if (
                map_config is None
                or map_config.default_objective != self.CAMPAIGN_OBJECTIVE_WAVES
                or not wave_configs
                or not wave_enemy_ids.issubset(map_enemy_ids)
            ):
                raise ValueError(
                    f"campaign_objective='waves' is not supported on map '{map_label}'"
                )

    def _compute_green_dragon_objective_reward(self) -> float:
        # На картах без городов формула сохраняет «двухгородский» эквивалент,
        # чтобы награда за целевого врага совпадала с драконьей.
        city_count = len(self.FINAL_OBJECTIVE_CITIES) or int(
            getattr(self, "OBJECTIVE_ENEMY_REWARD_CITY_EQUIVALENT", 2)
        )
        base_reward = self._compute_final_objective_reward(
            city_count,
            captured_before=0,
        )
        return float(base_reward) * float(
            getattr(self, "reward_green_dragon_objective_multiplier", 1.0)
        )

    def _apply_green_dragon_reached_reward_if_needed(
        self,
        enemy_id: Optional[int],
        reward: float,
        info: Dict[str, object],
    ) -> float:
        if not self._is_green_dragon_objective_enemy(enemy_id):
            return float(reward)
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return float(reward)
        grid_env = getattr(self, "grid_env", None)
        enemies_alive = getattr(grid_env, "enemies_alive", {}) or {}
        if not bool(enemies_alive.get(normalized_enemy_id, False)):
            return float(reward)

        info["green_dragon_reached"] = True
        info["green_dragon_reached_enemy_id"] = int(
            getattr(self, "OBJECTIVE_ENEMY_ID", self.GREEN_DRAGON_OBJECTIVE_ENEMY_ID)
        )
        if bool(getattr(self, "green_dragon_reached_reward_granted", False)):
            info["green_dragon_reached_reward"] = 0.0
            info["green_dragon_reached_reward_repeat"] = True
            return float(reward)

        reached_reward = float(getattr(self, "reward_green_dragon_reached", 0.0) or 0.0)
        self.green_dragon_reached_reward_granted = True
        info["green_dragon_reached_reward"] = float(reached_reward)
        info["green_dragon_reached_reward_repeat"] = False
        info["campaign_objective"] = self.campaign_objective
        return float(reward) + float(reached_reward)

    def _apply_objective_city_reached_reward_if_needed(
        self,
        enemy_id: Optional[int],
        reward: float,
        info: Dict[str, object],
    ) -> float:
        """Разовая награда за то, что агент добрался до целевого города и вступил
        в бой с его защитниками. Начисляется один раз на каждый город при первой
        схватке с любым из его ещё живых отрядов-защитников (аналог reach-награды
        дракона). Активна только для цели «города»."""
        if not self._campaign_objective_is_cities():
            return float(reward)
        city_name = self._objective_city_for_enemy(enemy_id)
        if city_name is None:
            return float(reward)
        if city_name in self.captured_objective_cities:
            return float(reward)
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return float(reward)
        grid_env = getattr(self, "grid_env", None)
        enemies_alive = getattr(grid_env, "enemies_alive", {}) or {}
        if not bool(enemies_alive.get(normalized_enemy_id, False)):
            return float(reward)

        info["objective_city_reached"] = True
        info["objective_city_reached_name"] = str(city_name)
        info["objective_city_reached_enemy_id"] = int(normalized_enemy_id)
        granted = getattr(self, "objective_city_reached_reward_granted", None)
        if granted is None:
            granted = set()
            self.objective_city_reached_reward_granted = granted
        if city_name in granted:
            info["objective_city_reached_reward"] = 0.0
            info["objective_city_reached_reward_repeat"] = True
            return float(reward)

        reached_reward = float(getattr(self, "reward_objective_city_reached", 0.0) or 0.0)
        granted.add(city_name)
        info["objective_city_reached_reward"] = float(reached_reward)
        info["objective_city_reached_reward_repeat"] = False
        info["campaign_objective"] = self.CAMPAIGN_OBJECTIVE_CITIES
        return float(reward) + float(reached_reward)

    def _apply_green_dragon_objective_reward_if_needed(
        self,
        enemy_id: Optional[int],
        reward: float,
        info: Dict[str, object],
    ) -> float:
        if not self._is_green_dragon_objective_enemy(enemy_id):
            return float(reward)
        objective_reward = self._compute_green_dragon_objective_reward()
        city_count = len(self.FINAL_OBJECTIVE_CITIES) or int(
            getattr(self, "OBJECTIVE_ENEMY_REWARD_CITY_EQUIVALENT", 2)
        )
        base_objective_reward = self._compute_final_objective_reward(
            city_count,
            captured_before=0,
        )
        info["final_objective_reward"] = float(objective_reward)
        info["green_dragon_objective_reward"] = float(objective_reward)
        info["green_dragon_objective_base_reward"] = float(base_objective_reward)
        info["green_dragon_objective_multiplier"] = float(
            getattr(self, "reward_green_dragon_objective_multiplier", 1.0)
        )
        info["green_dragon_objective_enemy_id"] = int(
            getattr(self, "OBJECTIVE_ENEMY_ID", self.GREEN_DRAGON_OBJECTIVE_ENEMY_ID)
        )
        info["campaign_objective"] = self.campaign_objective
        return float(reward) + float(objective_reward)

    def _apply_objective_dragon_damage_reward(
        self,
        reward: float,
        info: Dict[str, object],
    ) -> float:
        """Награда за РЕКОРД в бою с драконом-целью за эпизод (не за каждый удар).

        Дракон полностью регенерирует HP между ходами, поэтому за повторные атаки
        одинаковой глубины ничего не платим — иначе агент фармит частичный урон.
        Платим только за прогресс относительно лучшего результата эпизода:
        - новый минимум HP дракона (продавил глубже, чем когда-либо в этом эпизоде);
        - новый рекорд числа раундов, что основной отряд продержался в бою с драконом.
        Оба бонуса монотонны и за эпизод ограничены, поэтому фарм регенерации не работает."""
        if not self._is_green_dragon_objective_enemy(self.current_enemy_id):
            return float(reward)
        max_hp = float(getattr(self, "_objective_dragon_max_hp", 0.0) or 0.0)
        battle_env = getattr(self, "battle_env", None)
        if max_hp <= 0.0 or battle_env is None:
            return float(reward)
        end_hp = float(
            sum(
                max(0.0, float(u.get("health", 0) or 0))
                for u in getattr(battle_env, "combined", [])
                if u.get("team") == "red"
            )
        )
        total = float(reward)

        # --- рекорд продавливания HP дракона за эпизод ---
        prev_min = getattr(self, "_objective_dragon_min_hp_this_episode", None)
        if prev_min is None:
            prev_min = max_hp
        if end_hp < float(prev_min):
            gain = (float(prev_min) - end_hp) / max_hp
            damage_reward = float(
                getattr(self, "reward_dragon_damage_fraction", 0.0) or 0.0
            ) * gain
            self._objective_dragon_min_hp_this_episode = end_hp
            info["dragon_hp_record_low"] = float(end_hp)
            info["dragon_damage_record_reward"] = float(damage_reward)
            total += damage_reward

        # --- рекорд числа раундов, что основной отряд продержался против дракона ---
        rounds = int(getattr(battle_env, "round_no", 0) or 0)
        prev_best_rounds = int(getattr(self, "_objective_dragon_best_rounds_this_episode", 0) or 0)
        if rounds > prev_best_rounds:
            rounds_reward = float(
                getattr(self, "reward_dragon_rounds_record", 0.0) or 0.0
            ) * float(rounds - prev_best_rounds)
            self._objective_dragon_best_rounds_this_episode = rounds
            info["dragon_rounds_record"] = int(rounds)
            info["dragon_rounds_record_reward"] = float(rounds_reward)
            total += rounds_reward

        return total

    def _episode_enemies_defeated_bonus(self, info: Dict[str, object]) -> float:
        """Терминальная награда за число побеждённых за эпизод врагов в нормализованном виде.

        Нормировка — доля побеждённых от всех врагов на карте [0..1], умноженная на вес.
        Вес подобран так, чтобы максимум был заметно ниже награды за убийство дракона."""
        grid_env = getattr(self, "grid_env", None)
        enemies_alive = getattr(grid_env, "enemies_alive", {}) or {}
        total = len(enemies_alive)
        if total <= 0:
            return 0.0
        defeated = sum(1 for alive in enemies_alive.values() if not bool(alive))
        fraction = float(defeated) / float(total)
        bonus = float(getattr(self, "reward_enemies_defeated_weight", 0.0) or 0.0) * fraction
        info["enemies_defeated_count"] = int(defeated)
        info["enemies_defeated_total"] = int(total)
        info["enemies_defeated_bonus"] = float(bonus)
        return float(bonus)

    def _compute_no_dragon_victory_late_step_penalty(self, info: Dict[str, object]) -> float:
        if not self._campaign_objective_is_dragon():
            return 0.0
        if str(info.get("campaign_result", "") or "") == "victory":
            return 0.0

        base_penalty = float(
            getattr(self, "reward_no_dragon_victory_late_step_penalty", 0.0) or 0.0
        )
        penalty_cap = float(
            getattr(self, "reward_no_dragon_victory_late_step_penalty_cap", 0.0) or 0.0
        )
        if base_penalty <= 0.0 or penalty_cap <= 0.0:
            return 0.0

        campaign_steps = max(0, int(getattr(self, "campaign_steps", 0) or 0))
        start_fraction = float(
            getattr(self, "reward_no_dragon_victory_late_step_start_fraction", 0.35)
            or 0.0
        )
        grace_steps = max(1, int(round(float(self.max_grid_steps) * start_fraction)))
        if campaign_steps <= grace_steps:
            return 0.0

        late_span = max(1, int(self.max_grid_steps) - grace_steps)
        late_progress = min(1.0, float(campaign_steps - grace_steps) / float(late_span))
        effective_cap = max(base_penalty, penalty_cap)
        penalty = base_penalty + (effective_cap - base_penalty) * late_progress
        return -min(max(base_penalty, penalty), effective_cap)

    def _campaign_objective_completion_reason(
        self,
        defeated_enemy_id: Optional[int] = None,
    ) -> Optional[str]:
        if self._is_green_dragon_objective_enemy(defeated_enemy_id):
            return (
                "orc_defeated"
                if self._campaign_objective_is_orc()
                else "green_dragon_defeated"
            )
        if self._is_target_objective_enemy(defeated_enemy_id):
            return "target_enemy_defeated"
        if self._campaign_objective_is_cities() and self._all_objective_cities_captured():
            return "objective_cities_cleared"
        if (
            self._campaign_objective_is_all_enemies()
            and self.grid_env.all_enemies_defeated()
        ):
            return "all_enemies_defeated"
        if (
            self._campaign_objective_is_waves()
            and self._all_scheduled_enemy_waves_defeated()
        ):
            return "all_waves_defeated"
        return None

    def _apply_target_enemy_objective_reward_if_needed(
        self,
        enemy_id: Optional[int],
        reward: float,
        info: Dict[str, object],
    ) -> float:
        if (
            not self._is_target_objective_enemy(enemy_id)
            or bool(getattr(self, "target_enemy_reward_granted", False))
        ):
            return float(reward)
        final_reward = max(0.0, float(self.reward_all_enemies))
        self.target_enemy_reward_granted = True
        info["target_enemy_objective_reward"] = float(final_reward)
        info["target_enemy_reward_granted"] = True
        return float(reward) + float(final_reward)

    def _apply_all_enemies_objective_reward_if_needed(
        self,
        reward: float,
        info: Dict[str, object],
    ) -> float:
        if (
            not self._campaign_objective_is_all_enemies()
            or bool(getattr(self, "all_enemies_reward_granted", False))
            or not self.grid_env.all_enemies_defeated()
        ):
            return float(reward)
        final_reward = max(0.0, float(self.reward_all_enemies))
        self.all_enemies_reward_granted = True
        info["all_enemies_reward"] = float(final_reward)
        info["all_enemies_reward_granted"] = True
        return float(reward) + float(final_reward)

    def _apply_wave_defeat_reward_if_needed(
        self,
        enemy_id: Optional[int],
        reward: float,
        info: Dict[str, object],
    ) -> float:
        """Record a scheduled-wave victory and add its escalating one-time reward."""
        config = self._scheduled_enemy_wave_config_by_id(enemy_id)
        if config is None:
            return float(reward)
        normalized_enemy_id = int(config["enemy_id"])
        self.scheduled_enemy_defeated_ids.add(normalized_enemy_id)
        self._remove_scheduled_enemy_from_pending_encounters(normalized_enemy_id)

        wave_index = int(config["wave_index"])
        wave_configs = tuple(
            wave_config
            for wave_config in self._scheduled_enemy_wave_configs()
            if int(wave_config["wave_index"]) == wave_index
        )
        wave_enemy_ids = {
            int(wave_config["enemy_id"])
            for wave_config in wave_configs
        }
        defeated_ids = set(self.scheduled_enemy_defeated_ids)
        wave_completed = wave_enemy_ids.issubset(defeated_ids)
        wave_groups = self._scheduled_enemy_wave_groups()
        completed_wave_count = sum(
            1
            for wave_group in wave_groups
            if set(wave_group["enemy_ids"]).issubset(defeated_ids)
        )

        info["wave_enemy_id"] = normalized_enemy_id
        info["wave_index"] = wave_index
        info["wave_enemy_ids"] = sorted(wave_enemy_ids)
        info["wave_completed"] = bool(wave_completed)
        info["wave_remaining_enemy_ids"] = sorted(wave_enemy_ids - defeated_ids)
        info["wave_stacks_defeated_count"] = int(len(defeated_ids))
        info["wave_stacks_total"] = int(len(self._scheduled_enemy_wave_configs()))
        info["waves_defeated_count"] = int(completed_wave_count)
        info["waves_total"] = int(len(wave_groups))

        if not self._campaign_objective_is_waves():
            return float(reward)

        if not wave_completed:
            info["wave_defeat_bonus"] = 0.0
            info["wave_defeat_reward_repeat"] = False
            return float(reward)

        if wave_index in self.wave_reward_granted_indices:
            info["wave_defeat_bonus"] = 0.0
            info["wave_defeat_reward_repeat"] = True
            return float(reward)

        self.wave_reward_granted_indices.add(wave_index)
        self.wave_reward_granted_enemy_ids.update(wave_enemy_ids)
        reward_scale = max(
            0.0,
            float(getattr(self._map, "wave_reward_scale", 1.0) or 0.0),
        )
        wave_bonus = (
            float(self.reward_defeat_enemy)
            * reward_scale
            * float(max(1, wave_index))
        )
        total_reward = float(reward) + float(wave_bonus)
        info["wave_defeat_bonus"] = float(wave_bonus)
        info["wave_defeat_reward_repeat"] = False

        if (
            self._campaign_objective_is_waves()
            and self._all_scheduled_enemy_waves_defeated()
            and not bool(getattr(self, "waves_final_reward_granted", False))
        ):
            final_reward = max(0.0, float(self.reward_all_enemies))
            self.waves_final_reward_granted = True
            total_reward += final_reward
            info["waves_final_objective_reward"] = float(final_reward)
            info["waves_final_reward_granted"] = True

        return float(total_reward)

    def _ruin_progress_info(self) -> Dict[str, object]:
        """Собирает компактный progress-блок по руинам для info после reset/step."""
        last_reward = None
        if isinstance(self.last_ruin_reward, dict):
            last_reward = dict(self.last_ruin_reward)
        return {
            "ruins_cleared": int(len(self.cleared_ruin_enemy_ids)),
            "ruins_total": int(len(self.RUIN_REWARD_BY_ENEMY_ID)),
            "cleared_ruin_enemy_ids": sorted(int(enemy_id) for enemy_id in self.cleared_ruin_enemy_ids),
            "last_ruin_reward": last_reward,
        }
    def _reset_legions_settlement_territory_state(self) -> None:
        """Сбрасывает активные поселения Легионов и их уровни перед новым episode."""
        self.legions_active_settlement_territory_capture_turn_by_name: Dict[str, int] = {}
        self.legions_settlement_territory_tiles_by_name: Dict[str, Tuple[Tuple[int, int], ...]] = {
            str(source_name): ()
            for source_name in self.legions_settlement_territory_source_by_name.keys()
        }
        self.legions_settlement_level_by_name: Dict[str, int] = {
            str(source_name): max(
                1,
                int(
                    source_data.get("settlement_level", source_data.get("territory_level", 1))
                    or 1
                ),
            )
            for source_name, source_data in self.legions_settlement_territory_source_by_name.items()
        }
    def _legions_settlement_territory_info(self) -> Dict[str, object]:
        """Возвращает диагностическое состояние всех поселений и их территорий."""
        return {
            "legions_active_settlement_territories": sorted(
                self.legions_active_settlement_territory_capture_turn_by_name.keys()
            ),
            "legions_settlement_territory_capture_turn_by_name": {
                str(name): int(turn)
                for name, turn in self.legions_active_settlement_territory_capture_turn_by_name.items()
            },
            "legions_settlement_territory_tiles_by_name": {
                str(name): tuple(tiles)
                for name, tiles in self.legions_settlement_territory_tiles_by_name.items()
            },
            "legions_settlement_level_by_name": {
                str(name): int(level)
                for name, level in self.legions_settlement_level_by_name.items()
            },
        }
    def _grant_ruin_reward(self, enemy_id: Optional[int]) -> Dict[str, object]:
        """Выдает награду за руины, связанные с побежденным enemy_id.

        Метод идемпотентный: повторная победа над тем же ruin enemy не начисляет золото
        или предмет повторно, а только возвращает текущее состояние прогресса.
        """
        reward_data = self.RUIN_REWARD_BY_ENEMY_ID.get(int(enemy_id or -1))
        if reward_data is None:
            return {
                "ruin_cleared": False,
                "ruin_reward_applied": False,
                **self._ruin_progress_info(),
            }

        ruin_enemy_id = int(enemy_id or -1)
        ruin_title = str(reward_data.get("title", "") or "")
        raw_item_names = tuple(reward_data.get("items", ()) or ())
        if raw_item_names:
            item_names = tuple(
                str(item_name)
                for item_name in raw_item_names
                if str(item_name)
            )
        else:
            legacy_item_name = str(reward_data.get("item", "") or "")
            item_names = (legacy_item_name,) if legacy_item_name else ()
        item_name = item_names[0] if item_names else ""
        gold_reward = max(0.0, float(reward_data.get("gold", 0.0) or 0.0))
        ruin_pos = tuple(int(v) for v in tuple(reward_data.get("ruin_pos", ())))
        marker_pos = tuple(int(v) for v in tuple(reward_data.get("marker_pos", ())))

        # Руины считаются одноразовой наградой, поэтому сначала проверяем историю зачисток.
        if ruin_enemy_id in self.cleared_ruin_enemy_ids:
            return {
                "ruin_cleared": False,
                "ruin_reward_applied": False,
                "ruin_reward_enemy_id": ruin_enemy_id,
                "ruin_reward_title": ruin_title,
                **self._ruin_progress_info(),
            }

        self.cleared_ruin_enemy_ids.add(ruin_enemy_id)
        self.gold = max(0.0, float(self.gold or 0.0)) + gold_reward
        artifact_grant_info: Dict[str, object] = {}
        item_grant_infos: List[Dict[str, object]] = []
        for reward_item_name in item_names:
            # Предметы выдаются через общий inventory helper, чтобы не обходить его побочные эффекты.
            grant_info = self._grant_hero_item_reward(reward_item_name)
            item_grant_infos.append(dict(grant_info))
            artifact_grant_info.update(grant_info)

        self.last_ruin_reward = {
            "enemy_id": ruin_enemy_id,
            "title": ruin_title,
            "item": item_name,
            "items": list(item_names),
            "gold": gold_reward,
            "ruin_position": ruin_pos,
            "marker_position": marker_pos,
        }

        ruin_label = ruin_title or f"Руины {ruin_pos}"
        reward_parts = [f"+{gold_reward:g} gold"]
        reward_parts.extend(item_names)
        self._log(f"{ruin_label}: получено {', '.join(reward_parts)}")

        return {
            "ruin_cleared": True,
            "ruin_reward_applied": True,
            "ruin_reward_enemy_id": ruin_enemy_id,
            "ruin_reward_title": ruin_title,
            "ruin_reward_item": item_name,
            "ruin_reward_items": list(item_names),
            "ruin_reward_item_grants": item_grant_infos,
            "ruin_reward_gold": gold_reward,
            "ruin_reward_ruin_pos": ruin_pos,
            "ruin_reward_marker_pos": marker_pos,
            **artifact_grant_info,
            **self._ruin_progress_info(),
        }
    def _is_player_controlled_territory_tile(self, position: Optional[Tuple[int, int]]) -> bool:
        """Возвращает True для клетки, относящейся к территории игрока.

        В текущей кампании земля игрока хранится в legions_territory_tile_set:
        сюда входят клетки от стартовой столицы и захваченных городов/поселений.
        Название legacy-поля историческое и не связано с текущим цветом отрисовки.
        """
        if position is None:
            return False
        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return False
        return normalized_position in self.legions_territory_tile_set
    @classmethod
    def _settlement_upgrade_cost_for_level(cls, current_level: int) -> float:
        """Стоимость апгрейда задается уровнем до улучшения, а не целевым уровнем."""
        try:
            normalized_level = int(current_level)
        except (TypeError, ValueError):
            normalized_level = 0
        return float(cls.SETTLEMENT_UPGRADE_GOLD_COST_BY_LEVEL.get(normalized_level, 0.0))
    def _can_upgrade_settlement(self, settlement_name: str) -> bool:
        """Проверяет, доступен ли апгрейд уже захваченного поселения за текущее золото."""
        if str(settlement_name) not in self.legions_active_settlement_territory_capture_turn_by_name:
            return False
        current_level = int(self.legions_settlement_level_by_name.get(str(settlement_name), 0) or 0)
        if current_level >= int(self.MAX_SETTLEMENT_LEVEL):
            return False
        upgrade_cost = self._settlement_upgrade_cost_for_level(current_level)
        return upgrade_cost > 0.0 and float(self.gold or 0.0) >= float(upgrade_cost)
    @classmethod
    def is_final_objective_enemy_id(cls, enemy_id: Optional[int]) -> bool:
        """True, если enemy_id относится к финальной цели кампании."""
        try:
            objective_enemy_ids = {
                int(objective_enemy_id)
                for enemy_ids in cls.FINAL_OBJECTIVE_CITIES.values()
                for objective_enemy_id in enemy_ids
            }
            return int(enemy_id) in objective_enemy_ids
        except (TypeError, ValueError):
            return False
    def _is_final_objective_enemy(self, enemy_id: Optional[int]) -> bool:
        return self.is_final_objective_enemy_id(enemy_id)
    def _objective_city_for_enemy(self, enemy_id: Optional[int]) -> Optional[str]:
        """Находит целевой город, к которому привязан данный отряд-защитник."""
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return None
        for city_name, enemy_ids in self.FINAL_OBJECTIVE_CITIES.items():
            if normalized_enemy_id in {int(value) for value in enemy_ids}:
                return city_name
        return None
    def _is_objective_city_cleared(self, city_name: str) -> bool:
        """True, если оба отряда указанного целевого города уже побеждены."""
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}
        city_enemy_ids = self.FINAL_OBJECTIVE_CITIES.get(city_name)
        if not city_enemy_ids:
            return False
        for enemy_id in city_enemy_ids:
            if enemy_id not in enemies_alive:
                return False
            if bool(enemies_alive.get(enemy_id, False)):
                return False
        return True
    def _capture_objective_city_if_cleared(self, enemy_id: Optional[int]) -> List[str]:
        """Помечает город как захваченный, если текущий бой зачистил его полностью."""
        city_name = self._objective_city_for_enemy(enemy_id)
        if city_name is None:
            return []
        if city_name in self.captured_objective_cities:
            return []
        if not self._is_objective_city_cleared(city_name):
            return []
        self.captured_objective_cities.add(city_name)
        return [city_name]
    def _all_objective_cities_captured(self) -> bool:
        """True, когда игрок захватил все города из FINAL_OBJECTIVE_CITIES."""
        return (
            self._campaign_objective_is_cities()
            and bool(self.FINAL_OBJECTIVE_CITIES)
            and len(self.captured_objective_cities) == len(self.FINAL_OBJECTIVE_CITIES)
        )
    def _is_legions_settlement_territory_cleared(self, settlement_name: str) -> bool:
        """Проверяет, побеждены ли все required enemies, открывающие поселение Легионов."""
        source_data = self.legions_settlement_territory_source_by_name.get(str(settlement_name), {})
        required_enemy_ids = tuple(int(enemy_id) for enemy_id in source_data.get("required_enemy_ids", ()))
        if not required_enemy_ids:
            return False
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}
        for enemy_id in required_enemy_ids:
            if enemy_id not in enemies_alive:
                return False
            if bool(enemies_alive.get(enemy_id, False)):
                return False
        return True
    def _activate_legions_settlement_territory_if_cleared(
        self,
        enemy_id: Optional[int],
    ) -> List[str]:
        """Активирует рост территории поселения после победы над последним защитником."""
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return []
        settlement_name = self.legions_settlement_territory_source_name_by_enemy_id.get(normalized_enemy_id)
        if not settlement_name:
            return []
        if settlement_name in self.legions_active_settlement_territory_capture_turn_by_name:
            return []
        if not self._is_legions_settlement_territory_cleared(settlement_name):
            return []
        self.legions_active_settlement_territory_capture_turn_by_name[str(settlement_name)] = int(self.turns)
        # Захват поселения сразу добавляет его источник роста территории в общий frontier.
        self._refresh_faction_territories()
        return [str(settlement_name)]
    def _compute_final_objective_reward(
        self,
        city_count: int = 1,
        *,
        captured_before: int = 0,
    ) -> float:
        """
        Награда за захват одного или нескольких целевых городов.

        Первый захваченный целевой город даёт уменьшенную награду (x1),
        последующие — полную (x5).
        """
        if city_count <= 0:
            return 0.0
        total_reward = 0.0
        captured_before = max(0, int(captured_before))
        base_reward = float(self.reward_all_enemies)
        full_multiplier = float(self.FINAL_OBJECTIVE_CITY_REWARD_MULTIPLIER)

        for offset in range(int(city_count)):
            capture_index = captured_before + offset
            multiplier = 1.0 if capture_index == 0 else full_multiplier
            total_reward += base_reward * multiplier

        return float(total_reward)
    def _compute_blue_exp_reward(self) -> Tuple[float, float]:
        """Возвращает (reward, raw_exp) за опыт BLUE в завершённом бою."""
        if self.battle_env is None or self.battle_env.winner != "blue":
            return 0.0, 0.0
        raw_exp = max(0.0, float(getattr(self.battle_env, "last_battle_exp", 0.0) or 0.0))
        exp_norm = raw_exp / (raw_exp + self.exp_reward_norm_k)
        reward = self.reward_exp_weight * exp_norm
        return float(reward), float(raw_exp)
    def _compute_blue_survival_reward(self) -> Tuple[float, float, float]:
        """
        Награда за сохранность базовой команды BLUE после боя.
        Возвращает (reward, alive_ratio, hp_ratio).
        """
        if self.battle_env is None:
            return 0.0, 0.0, 0.0

        base_roster = (
            self.blue_team_state
            if getattr(self, "blue_team_state", None) is not None
            else self._create_starting_blue_team()
        )
        base_units = [
            u
            for u in base_roster
            if int(u.get("position", -1)) >= 0
        ]
        if not base_units:
            return 0.0, 0.0, 0.0

        base_by_pos = {int(u["position"]): u for u in base_units}
        combined_blue = {
            int(u.get("position", -1)): u
            for u in self.battle_env.combined
            if u.get("team") == "blue" and int(u.get("position", -1)) in base_by_pos
        }

        alive_count = 0
        hp_ratio_sum = 0.0
        total = len(base_by_pos)

        for pos, base_unit in base_by_pos.items():
            cur_unit = combined_blue.get(pos)
            hp = float(cur_unit.get("health", 0) or cur_unit.get("hp", 0) or 0) if cur_unit else 0.0
            base_max = float(
                base_unit.get("max_health", 0)
                or base_unit.get("maxhp", 0)
                or base_unit.get("health", 0)
                or base_unit.get("hp", 0)
                or 1.0
            )
            cur_max = (
                float(cur_unit.get("max_health", 0) or cur_unit.get("maxhp", 0) or 0)
                if cur_unit
                else 0.0
            )
            max_hp = max(1.0, base_max, cur_max)
            if hp > 0:
                alive_count += 1
            hp_ratio_sum += float(np.clip(hp / max_hp, 0.0, 1.0))

        alive_ratio = float(alive_count) / float(total)
        hp_ratio = float(hp_ratio_sum) / float(total)
        reward = (
            self.reward_survival_alive_weight * alive_ratio
            + self.reward_survival_hp_weight * hp_ratio
        )
        return float(reward), float(alive_ratio), float(hp_ratio)
    def _legions_gold_income_per_turn(self) -> float:
        """Считает доход за turn: базовое золото плюс доход от захваченных шахт."""
        return float(self.GOLD_PER_TURN) + float(self.legions_captured_gold_mine_count) * float(
            self.LEGIONS_GOLD_MINE_GOLD_PER_TURN
        )

    def _scheduled_enemy_wave_configs(self) -> Tuple[Dict[str, int], ...]:
        raw_waves = tuple(getattr(self._map, "scheduled_enemy_waves", ()) or ())
        normalized: List[Dict[str, int]] = []
        for index, raw_wave in enumerate(raw_waves):
            if not isinstance(raw_wave, dict):
                continue
            raw_enemy_ids = raw_wave.get("enemy_ids")
            if raw_enemy_ids is None:
                raw_enemy_id = raw_wave.get("enemy_id")
                enemy_ids = () if raw_enemy_id is None else (raw_enemy_id,)
            elif isinstance(raw_enemy_ids, (list, tuple, set, frozenset)):
                enemy_ids = tuple(raw_enemy_ids)
            else:
                enemy_ids = (raw_enemy_ids,)
            member_count = len(enemy_ids)
            base_spawn_turn = max(
                1,
                int(raw_wave.get("spawn_turn", 5 * (index + 1)) or 1),
            )
            spawn_interval = max(
                0,
                int(raw_wave.get("spawn_interval", 0) or 0),
            )
            for member_index, raw_enemy_id in enumerate(enemy_ids):
                if raw_enemy_id is None:
                    continue
                normalized.append(
                    {
                        "enemy_id": int(raw_enemy_id),
                        "spawn_turn": int(
                            base_spawn_turn + member_index * spawn_interval
                        ),
                        "moves_per_turn": max(
                            0,
                            int(raw_wave.get("moves_per_turn", 0) or 0),
                        ),
                        "wave_index": max(
                            1,
                            int(raw_wave.get("wave_index", index + 1) or index + 1),
                        ),
                        "wave_member_index": int(member_index + 1),
                        "wave_member_count": int(member_count),
                    }
                )
        if normalized:
            return tuple(normalized)

        legacy_enemy_id = getattr(self._map, "scheduled_enemy_id", None)
        legacy_spawn_turn = getattr(self._map, "scheduled_enemy_spawn_turn", None)
        if legacy_enemy_id is None or legacy_spawn_turn is None:
            return ()
        return (
            {
                "enemy_id": int(legacy_enemy_id),
                "spawn_turn": max(1, int(legacy_spawn_turn)),
                "moves_per_turn": max(
                    0,
                    int(getattr(self._map, "scheduled_enemy_moves_per_turn", 0) or 0),
                ),
                "wave_index": 1,
                "wave_member_index": 1,
                "wave_member_count": 1,
            },
        )

    def _scheduled_enemy_wave_groups(self) -> Tuple[Dict[str, object], ...]:
        grouped: Dict[int, Dict[str, object]] = {}
        for config in self._scheduled_enemy_wave_configs():
            wave_index = int(config["wave_index"])
            group = grouped.setdefault(
                wave_index,
                {
                    "wave_index": wave_index,
                    "spawn_turn": int(config["spawn_turn"]),
                    "moves_per_turn": int(config["moves_per_turn"]),
                    "enemy_ids": [],
                    "last_spawn_turn": int(config["spawn_turn"]),
                },
            )
            group["enemy_ids"].append(int(config["enemy_id"]))
            group["spawn_turn"] = min(
                int(group["spawn_turn"]),
                int(config["spawn_turn"]),
            )
            group["last_spawn_turn"] = max(
                int(group["last_spawn_turn"]),
                int(config["spawn_turn"]),
            )
        return tuple(
            {
                **group,
                "enemy_ids": tuple(int(enemy_id) for enemy_id in group["enemy_ids"]),
            }
            for group in grouped.values()
        )

    def _scheduled_enemy_wave_config_by_id(
        self,
        enemy_id: Optional[int],
    ) -> Optional[Dict[str, int]]:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return None
        for config in self._scheduled_enemy_wave_configs():
            if int(config["enemy_id"]) == normalized_enemy_id:
                return config
        return None

    def _all_scheduled_enemy_waves_defeated(self) -> bool:
        configs = self._scheduled_enemy_wave_configs()
        if not configs:
            return False
        required_ids = {int(config["enemy_id"]) for config in configs}
        return required_ids.issubset(
            set(getattr(self, "scheduled_enemy_defeated_ids", set()) or set())
        )

    def _queue_scheduled_enemy_encounter(self, enemy_id: int) -> None:
        normalized_enemy_id = int(enemy_id)
        pending = self.scheduled_enemy_pending_encounter_ids
        if normalized_enemy_id not in pending:
            pending.append(normalized_enemy_id)
        self.scheduled_enemy_pending_encounter_id = (
            int(pending[0]) if pending else None
        )

    def _remove_scheduled_enemy_from_pending_encounters(self, enemy_id: int) -> None:
        normalized_enemy_id = int(enemy_id)
        self.scheduled_enemy_pending_encounter_ids = [
            int(pending_id)
            for pending_id in self.scheduled_enemy_pending_encounter_ids
            if int(pending_id) != normalized_enemy_id
        ]
        self.scheduled_enemy_pending_encounter_id = (
            int(self.scheduled_enemy_pending_encounter_ids[0])
            if self.scheduled_enemy_pending_encounter_ids
            else None
        )

    def _pop_scheduled_enemy_pending_encounter(self) -> Optional[int]:
        while self.scheduled_enemy_pending_encounter_ids:
            enemy_id = int(self.scheduled_enemy_pending_encounter_ids.pop(0))
            if self.grid_env.enemies_alive.get(enemy_id, False):
                self.scheduled_enemy_pending_encounter_id = (
                    int(self.scheduled_enemy_pending_encounter_ids[0])
                    if self.scheduled_enemy_pending_encounter_ids
                    else None
                )
                return enemy_id
        self.scheduled_enemy_pending_encounter_id = None
        return None

    def _scheduled_enemy_state_info(self) -> Dict[str, object]:
        configs = self._scheduled_enemy_wave_configs()
        if not configs:
            return {
                "scheduled_enemy_id": None,
                "scheduled_enemy_spawned": False,
                "scheduled_enemy_active": False,
                "scheduled_enemy_wave_count": 0,
                "scheduled_enemy_waves_spawned": 0,
                "scheduled_enemy_waves_defeated": 0,
                "scheduled_enemy_waves_active": 0,
                "scheduled_enemy_stack_count": 0,
                "scheduled_enemy_stacks_spawned": 0,
                "scheduled_enemy_stacks_defeated": 0,
                "scheduled_enemy_stacks_active": 0,
                "scheduled_enemy_waves": [],
                "scheduled_enemy_wave_groups": [],
            }

        wave_groups = self._scheduled_enemy_wave_groups()
        spawned_ids = set(
            getattr(self, "scheduled_enemy_spawned_ids", set()) or set()
        )
        defeated_ids = set(
            getattr(self, "scheduled_enemy_defeated_ids", set()) or set()
        )
        wave_states: List[Dict[str, object]] = []
        for config in configs:
            enemy_id = int(config["enemy_id"])
            spawned = enemy_id in spawned_ids
            alive = bool(self.grid_env.enemies_alive.get(enemy_id, False))
            wave_states.append(
                {
                    **dict(config),
                    "spawned": bool(spawned),
                    "defeated": bool(enemy_id in defeated_ids),
                    "active": bool(spawned and alive),
                    "position": tuple(
                        self.grid_env.enemy_positions.get(
                            enemy_id,
                            self._static_enemy_positions.get(enemy_id, self.CASTLE_POS),
                        )
                    ),
                }
            )

        group_states: List[Dict[str, object]] = []
        for group in wave_groups:
            enemy_ids = {
                int(enemy_id)
                for enemy_id in tuple(group["enemy_ids"])
            }
            group_spawned = enemy_ids.issubset(spawned_ids)
            group_defeated = enemy_ids.issubset(defeated_ids)
            group_active = any(
                enemy_id in spawned_ids
                and self.grid_env.enemies_alive.get(enemy_id, False)
                for enemy_id in enemy_ids
            )
            group_states.append(
                {
                    **dict(group),
                    "spawned": bool(group_spawned),
                    "defeated": bool(group_defeated),
                    "active": bool(group_active),
                }
            )

        first = configs[0]
        first_enemy_id = int(first["enemy_id"])
        active_stack_count = sum(1 for state in wave_states if bool(state["active"]))
        spawned_group_count = sum(1 for state in group_states if bool(state["spawned"]))
        defeated_group_count = sum(1 for state in group_states if bool(state["defeated"]))
        active_group_count = sum(1 for state in group_states if bool(state["active"]))
        return {
            "scheduled_enemy_id": first_enemy_id,
            "scheduled_enemy_spawn_turn": int(first["spawn_turn"]),
            "scheduled_enemy_moves_per_turn": int(first["moves_per_turn"]),
            "scheduled_enemy_spawned": bool(first_enemy_id in spawned_ids),
            "scheduled_enemy_active": bool(
                first_enemy_id in spawned_ids
                and self.grid_env.enemies_alive.get(first_enemy_id, False)
            ),
            "scheduled_enemy_position": tuple(
                self.grid_env.enemy_positions.get(
                    first_enemy_id,
                    self._static_enemy_positions.get(first_enemy_id, self.CASTLE_POS),
                )
            ),
            "scheduled_enemy_wave_count": int(len(wave_groups)),
            "scheduled_enemy_waves_spawned": int(spawned_group_count),
            "scheduled_enemy_waves_defeated": int(defeated_group_count),
            "scheduled_enemy_waves_active": int(active_group_count),
            "scheduled_enemy_stack_count": int(len(configs)),
            "scheduled_enemy_stacks_spawned": int(len(spawned_ids)),
            "scheduled_enemy_stacks_defeated": int(len(defeated_ids)),
            "scheduled_enemy_stacks_active": int(active_stack_count),
            "scheduled_enemy_pending_encounter_ids": list(
                getattr(self, "scheduled_enemy_pending_encounter_ids", ()) or ()
            ),
            "scheduled_enemy_waves": wave_states,
            "scheduled_enemy_wave_groups": group_states,
        }

    def _reset_scheduled_enemy_state(self) -> None:
        self.scheduled_enemy_spawned = False
        self.scheduled_enemy_spawned_ids: set[int] = set()
        self.scheduled_enemy_defeated_ids: set[int] = set()
        self.wave_reward_granted_enemy_ids: set[int] = set()
        self.wave_reward_granted_indices: set[int] = set()
        self.waves_final_reward_granted = False
        self.scheduled_enemy_pending_encounter_ids: List[int] = []
        self.scheduled_enemy_pending_encounter_id = None
        self.scheduled_enemy_turn_infos: List[Dict[str, object]] = []
        for config in self._scheduled_enemy_wave_configs():
            normalized_enemy_id = int(config["enemy_id"])
            if normalized_enemy_id in self._static_enemy_positions:
                self.grid_env.enemy_positions[normalized_enemy_id] = tuple(
                    self._static_enemy_positions[normalized_enemy_id]
                )
            if normalized_enemy_id in self.grid_env.enemies_alive:
                self.grid_env.enemies_alive[normalized_enemy_id] = False

    def _scheduled_enemy_next_step(
        self,
        origin: Tuple[int, int],
        target: Tuple[int, int],
    ) -> Tuple[int, int]:
        origin_x, origin_y = int(origin[0]), int(origin[1])
        target_x, target_y = int(target[0]), int(target[1])
        step_x = 0 if target_x == origin_x else (1 if target_x > origin_x else -1)
        step_y = 0 if target_y == origin_y else (1 if target_y > origin_y else -1)
        candidates = [
            (origin_x + step_x, origin_y + step_y),
            (origin_x + step_x, origin_y),
            (origin_x, origin_y + step_y),
        ]
        current_distance = max(abs(target_x - origin_x), abs(target_y - origin_y))
        seen: set[Tuple[int, int]] = set()
        for candidate in candidates:
            if candidate in seen or candidate == (origin_x, origin_y):
                continue
            seen.add(candidate)
            if not (0 <= candidate[0] < self.grid_size and 0 <= candidate[1] < self.grid_size):
                continue
            if candidate in self.grid_env.obstacle_positions:
                continue
            candidate_distance = max(
                abs(target_x - candidate[0]),
                abs(target_y - candidate[1]),
            )
            if candidate_distance < current_distance:
                return candidate
        return origin_x, origin_y

    def _advance_scheduled_enemy_wave_one_turn(
        self,
        config: Dict[str, int],
    ) -> Dict[str, object]:
        normalized_enemy_id = int(config["enemy_id"])
        normalized_spawn_turn = max(1, int(config["spawn_turn"]))
        moves_per_turn = max(0, int(config["moves_per_turn"]))
        info: Dict[str, object] = {
            "enemy_id": normalized_enemy_id,
            "wave_index": int(config["wave_index"]),
            "turn": int(self.turns),
            "spawned_this_turn": False,
            "moved_path": [],
            "contact_with_agent": False,
        }
        if int(self.turns) < normalized_spawn_turn:
            info["state"] = "dormant"
            return info
        if normalized_enemy_id not in self.scheduled_enemy_spawned_ids:
            self.scheduled_enemy_spawned_ids.add(normalized_enemy_id)
            self.scheduled_enemy_spawned = True
            self.grid_env.enemies_alive[normalized_enemy_id] = True
            self.grid_env.enemy_positions[normalized_enemy_id] = tuple(
                self._static_enemy_positions[normalized_enemy_id]
            )
            info["spawned_this_turn"] = True
        if not self.grid_env.enemies_alive.get(normalized_enemy_id, False):
            info["state"] = "defeated"
            return info

        position = tuple(self.grid_env.enemy_positions[normalized_enemy_id])
        target = tuple(self.grid_env.agent_pos)
        moved_path: List[Tuple[int, int]] = []
        for _ in range(moves_per_turn):
            if position == target:
                break
            next_position = self._scheduled_enemy_next_step(position, target)
            if next_position == position:
                break
            position = tuple(next_position)
            moved_path.append(position)
        self.grid_env.enemy_positions[normalized_enemy_id] = position
        contact = position == target
        if contact:
            self._queue_scheduled_enemy_encounter(normalized_enemy_id)
        info.update(
            {
                "state": "engaging" if contact else "pursuing",
                "position": position,
                "target": target,
                "moved_path": list(moved_path),
                "steps_moved": int(len(moved_path)),
                "contact_with_agent": bool(contact),
            }
        )
        return info

    def _advance_scheduled_enemy_one_turn(self) -> Dict[str, object]:
        """Backward-compatible single-pursuer wrapper used by older callers."""
        configs = self._scheduled_enemy_wave_configs()
        if not configs:
            return {}
        return self._advance_scheduled_enemy_wave_one_turn(configs[0])

    def _advance_scheduled_enemies_one_turn(self) -> List[Dict[str, object]]:
        return [
            self._advance_scheduled_enemy_wave_one_turn(config)
            for config in self._scheduled_enemy_wave_configs()
        ]

    def _advance_turns(self, delta_turns: int) -> float:
        """Продвигает campaign time и выполняет все эффекты начала нового turn.

        Здесь обновляются территории, доходы, мана, временные map effects, лечение врагов,
        scripted bot и одноразовые флаги действий. Возвращается штраф за прошедшие turn.
        """
        if delta_turns <= 0:
            return 0.0
        total_gold_income = 0.0
        scripted_bot_turn_infos = []
        scheduled_enemy_turn_infos = []
        for _ in range(int(delta_turns)):
            self.turns += 1
            # Территории растут в начале каждого turn, поэтому доход ниже уже учитывает новые шахты.
            self._refresh_faction_territories()
            self._clear_all_enemy_map_spell_effects()
            self._clear_all_blue_map_spell_effects()
            total_gold_income += self._legions_gold_income_per_turn()
            self._apply_mana_income_for_turn()
            self._heal_wounded_enemy_teams_for_turn()
            scheduled_enemy_turn_infos.extend(
                self._advance_scheduled_enemies_one_turn()
            )
            advance_bot = getattr(self, "_advance_scripted_capital_bot_one_turn", None)
            if callable(advance_bot) and bool(getattr(self, "scripted_capital_bot_enabled", False)):
                scripted_bot_turn_infos.append(advance_bot())
        # Эти флаги ограничивают действие в рамках одного turn и должны сбрасываться после advance.
        self._clear_expired_combat_potion_effects()
        self.combat_potion_battle_bonus_pending = False
        self.gold += float(total_gold_income)
        self.scripted_capital_bot_turn_infos = scripted_bot_turn_infos
        self.scheduled_enemy_turn_infos = scheduled_enemy_turn_infos
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state)
        self.moves = self.moves_per_turn
        self.active_buildings["alredybuilt"] = 0
        self.spell_learning_locked = False
        self.battle_item_equip_used_this_turn = False
        self.book_equip_used_this_turn = False
        self.spell_cast_counts_by_id_this_turn = {}
        self.grid_unit_swap_actions_used_this_turn = set()
        self.grid_unit_swaps_this_turn = 0
        self.summon_hero_battle_bonus_pending = False
        self.summon_hero_battle_bonus_enemy_ids_this_turn = set()
        return -float(delta_turns) * self.reward_turn_penalty
    @staticmethod
    def _territory_sort_key(
        origin_tile: Tuple[int, int],
        tile: Tuple[int, int],
    ) -> Tuple[int, int, int, int, int, int]:
        origin_x, origin_y = int(origin_tile[0]), int(origin_tile[1])
        tile_x, tile_y = int(tile[0]), int(tile[1])
        dx = tile_x - origin_x
        dy = tile_y - origin_y
        # В кампании есть диагональный ход, поэтому основная близость — по Chebyshev,
        # а дальше добиваем детерминированной развязкой по Manhattan и координатам.
        return (
            max(abs(dx), abs(dy)),
            abs(dx) + abs(dy),
            abs(dy),
            abs(dx),
            tile_y,
            tile_x,
        )
    def _legions_territory_sort_key(self, tile: Tuple[int, int]) -> Tuple[int, int, int, int, int, int]:
        return self._territory_sort_key(self.legions_territory_source_tile, tile)
    def _empire_territory_sort_key(self, tile: Tuple[int, int]) -> Tuple[int, int, int, int, int, int]:
        return self._territory_sort_key(self.empire_territory_source_tile, tile)
    def _build_territory_forbidden_tiles(
        self,
        *,
        source_tile: Tuple[int, int],
        additional_claimable_tiles: Tuple[Tuple[int, int], ...] = (),
    ) -> Tuple[Tuple[int, int], ...]:
        """Строит клетки, куда территория не может расширяться.

        Forbidden tiles ограничивают именно владение территорией. Вражеские стеки и явно
        разрешенные source/anchor tiles исключаются, чтобы территория могла добраться до целей.
        """
        forbidden_tiles = set(tuple(tile) for tile in self._static_obstacle_tiles)

        base_forbidden_tiles = self._map.territory_forbidden_tiles_provider()
        if base_forbidden_tiles:
            forbidden_tiles.update(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_forbidden_tiles)
            )

        enemy_tiles = {
            tuple(int(coord) for coord in pos)
            for pos in (self._static_enemy_positions or {}).values()
        }
        forbidden_tiles.difference_update(enemy_tiles)
        claimable_tiles = {
            tuple(int(coord) for coord in tuple(source_tile))
        }
        claimable_tiles.update(
            tuple(int(coord) for coord in tile)
            for tile in tuple(additional_claimable_tiles or ())
        )
        forbidden_tiles.difference_update(claimable_tiles)

        return tuple(sorted(forbidden_tiles))
    def _build_legions_territory_forbidden_tiles(self) -> Tuple[Tuple[int, int], ...]:
        """Forbidden mask для территории игрока/Легионов."""
        return self._build_territory_forbidden_tiles(
            source_tile=self.legions_territory_source_tile,
            additional_claimable_tiles=(
                self.legions_territory_source_tile,
                self.empire_territory_source_tile,
            ),
        )
    def _build_empire_territory_forbidden_tiles(self) -> Tuple[Tuple[int, int], ...]:
        """Forbidden mask для территории scripted Empire side."""
        return self._build_territory_forbidden_tiles(
            source_tile=self.empire_territory_source_tile,
            additional_claimable_tiles=(
                self.legions_territory_source_tile,
                self.empire_territory_source_tile,
            ),
        )
    def _build_territory_path_blocked_tiles(
        self,
        *,
        source_tile: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], ...]:
        """Строит препятствия для BFS-пути роста территории.

        Path blocked tiles используются только для достижимости: вода и статические препятствия
        не дают frontier перескочить через карту, но source tile всегда остается стартом.
        """
        blocked_tiles = set(tuple(tile) for tile in self._static_obstacle_tiles)
        base_water_tiles = self._map.water_tiles_provider()
        if base_water_tiles:
            blocked_tiles.update(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_water_tiles)
            )
        blocked_tiles.discard(tuple(source_tile))
        return tuple(sorted(blocked_tiles))
    def _build_legions_territory_path_blocked_tiles(self) -> Tuple[Tuple[int, int], ...]:
        """BFS-блокеры для роста территории Легионов."""
        return self._build_territory_path_blocked_tiles(
            source_tile=self.legions_territory_source_tile,
        )
    def _build_empire_territory_path_blocked_tiles(self) -> Tuple[Tuple[int, int], ...]:
        """BFS-блокеры для роста территории Empire side."""
        return self._build_territory_path_blocked_tiles(
            source_tile=self.empire_territory_source_tile,
        )
    def _build_territory_path_distances(
        self,
        *,
        source_tile: Tuple[int, int],
        blocked_tile_set: set[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], int]:
        """Считает BFS distance от source tile до всех достижимых клеток карты.

        Метод используется не только для роста территории, но и для поиска ближайших
        целей в action mask/spell логике. Поэтому результаты кешируются по стартовой
        клетке и набору блокеров: повторный запрос в том же состоянии карты не гоняет BFS.
        """
        source = (int(source_tile[0]), int(source_tile[1]))
        blocked_tiles = frozenset(
            (int(tile[0]), int(tile[1]))
            for tile in (blocked_tile_set or set())
        )
        cache_key = (int(self.grid_size), source, blocked_tiles)
        cache = getattr(self, "_territory_path_distances_cache", None)
        if cache is None:
            cache = {}
            self._territory_path_distances_cache = cache
        else:
            cached_distances = cache.get(cache_key)
            if cached_distances is not None:
                # dict сохраняет порядок вставки, поэтому pop+assign работает как простой LRU touch.
                cache.pop(cache_key, None)
                cache[cache_key] = cached_distances
                return cached_distances
        max_cache_size = max(
            0,
            int(getattr(self, "TERRITORY_PATH_DISTANCE_CACHE_SIZE", 64) or 0),
        )

        if source in blocked_tiles:
            distances: Dict[Tuple[int, int], int] = {}
            if max_cache_size > 0:
                cache[cache_key] = distances
                while len(cache) > max_cache_size:
                    cache.pop(next(iter(cache)))
            return distances

        distances: Dict[Tuple[int, int], int] = {source: 0}
        frontier = deque([source])
        neighbor_offsets = (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        )

        while frontier:
            x, y = frontier.popleft()
            current_distance = distances[(x, y)]
            for dx, dy in neighbor_offsets:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < int(self.grid_size) and 0 <= ny < int(self.grid_size)):
                    continue
                tile = (nx, ny)
                if tile in blocked_tiles or tile in distances:
                    continue
                distances[tile] = current_distance + 1
                frontier.append(tile)

        if max_cache_size > 0:
            cache[cache_key] = distances
            while len(cache) > max_cache_size:
                cache.pop(next(iter(cache)))
        return distances
    def _build_legions_territory_path_distances(self) -> Dict[Tuple[int, int], int]:
        return self._build_territory_path_distances(
            source_tile=self.legions_territory_source_tile,
            blocked_tile_set=self.legions_territory_path_blocked_tile_set,
        )
    def _build_empire_territory_path_distances(self) -> Dict[Tuple[int, int], int]:
        return self._build_territory_path_distances(
            source_tile=self.empire_territory_source_tile,
            blocked_tile_set=self.empire_territory_path_blocked_tile_set,
        )
    def _build_territory_order(
        self,
        *,
        forbidden_tile_set: set[Tuple[int, int]],
        path_distances: Dict[Tuple[int, int], int],
        sort_key,
    ) -> Tuple[Tuple[int, int], ...]:
        """Формирует стабильный порядок захвата клеток территории.

        Сначала идет BFS distance от источника, затем локальный sort_key. Это делает рост
        территории детерминированным и не зависящим от порядка обхода set/dict.
        """
        allowed_tiles = tuple(
            (x, y)
            for y in range(int(self.grid_size))
            for x in range(int(self.grid_size))
            if (x, y) not in forbidden_tile_set and (x, y) in path_distances
        )
        return tuple(
            sorted(
                allowed_tiles,
                key=lambda tile: (int(path_distances[tile]), *sort_key(tile)),
            )
        )
    def _build_legions_territory_order(self) -> Tuple[Tuple[int, int], ...]:
        """Порядок роста основной территории игрока от стартовой столицы."""
        return self._build_territory_order(
            forbidden_tile_set=self.legions_territory_forbidden_tile_set,
            path_distances=self.legions_territory_path_distances,
            sort_key=self._legions_territory_sort_key,
        )
    def _build_legions_settlement_territory_orders_by_name(self) -> Dict[str, Tuple[Tuple[int, int], ...]]:
        """Предрасчитывает порядок роста для каждого захватываемого поселения Легионов."""
        orders_by_name: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        for source_data in self._static_legions_settlement_territory_sources:
            settlement_name = str(source_data.get("name", "") or "")
            source_tile = tuple(source_data.get("source_tile", (0, 0)))
            path_blocked_tiles = self._build_territory_path_blocked_tiles(source_tile=source_tile)
            path_distances = self._build_territory_path_distances(
                source_tile=source_tile,
                blocked_tile_set=set(path_blocked_tiles),
            )
            orders_by_name[settlement_name] = self._build_territory_order(
                forbidden_tile_set=self.legions_territory_forbidden_tile_set,
                path_distances=path_distances,
                sort_key=lambda tile, origin_tile=source_tile: self._territory_sort_key(origin_tile, tile),
            )
        return orders_by_name
    def _build_empire_territory_order(self) -> Tuple[Tuple[int, int], ...]:
        """Порядок роста территории Empire side от ее столицы."""
        return self._build_territory_order(
            forbidden_tile_set=self.empire_territory_forbidden_tile_set,
            path_distances=self.empire_territory_path_distances,
            sort_key=self._empire_territory_sort_key,
        )
    @staticmethod
    def _territory_claim_count(total_tiles: int, expansion_per_turn: int, turns: int) -> int:
        """Количество клеток, которое источник территории должен контролировать к данному turn."""
        return min(
            int(total_tiles),
            1 + max(0, int(turns)) * int(expansion_per_turn),
        )
    @staticmethod
    def _claim_territory_tiles(
        order: Tuple[Tuple[int, int], ...],
        claim_count: int,
    ) -> Tuple[Tuple[int, int], ...]:
        """Берет первые claim_count клеток из заранее отсортированного порядка захвата."""
        if claim_count <= 0:
            return ()
        return tuple(order[: int(claim_count)])
    @staticmethod
    def _settlement_territory_claim_count(
        total_tiles: int,
        expansion_per_turn: int,
        turns: int,
        activation_turn: Optional[int],
    ) -> int:
        """Считает рост поселения только после turn, на котором оно было активировано."""
        if activation_turn is None:
            return 0
        elapsed_turns = max(0, int(turns) - int(activation_turn))
        return min(
            int(total_tiles),
            1 + elapsed_turns * max(0, int(expansion_per_turn)),
        )
    def _refresh_legions_territory_tiles(self) -> None:
        """Совместимый wrapper для старых вызовов: сейчас обновляются все фракционные территории."""
        self._refresh_faction_territories()
    def _refresh_legions_territory_grid_obs_cache(self) -> None:
        """Обновляет бинарный observation cache клеток территории Легионов."""
        obs_size = int(getattr(self, "grid_legions_territory_obs_size", 0) or 0)
        if obs_size <= 0:
            self._legions_territory_grid_obs_cache = np.zeros(0, dtype=np.float32)
            return

        territory_obs = np.zeros(obs_size, dtype=np.float32)
        index_by_pos = getattr(self, "grid_legions_territory_index_by_pos", {})
        for tile_pos in self.legions_territory_tiles:
            tile_index = index_by_pos.get(tuple(tile_pos))
            if tile_index is not None:
                territory_obs[int(tile_index)] = 1.0
        self._legions_territory_grid_obs_cache = territory_obs
    def _refresh_legions_captured_mana_source_state(self) -> None:
        """Пересчитывает источники маны, которые уже попали в территорию игрока."""
        captured_tiles_by_kind: Dict[str, List[Tuple[int, int]]] = {
            str(kind): []
            for kind in self.MANA_KIND_ORDER
        }
        for raw_tile, mana_meta in getattr(self, "mana_sources", {}).items():
            tile = tuple(int(coord) for coord in tuple(raw_tile)[:2])
            if tile not in self.legions_territory_tile_set:
                continue
            mana_kind = str(mana_meta.get("kind", "") or "")
            if mana_kind not in captured_tiles_by_kind:
                continue
            captured_tiles_by_kind[mana_kind].append(tile)

        self.legions_captured_mana_source_tiles_by_kind = {
            str(kind): tuple(sorted(captured_tiles_by_kind[str(kind)]))
            for kind in self.MANA_KIND_ORDER
        }
        self.legions_captured_mana_source_counts_by_kind = {
            str(kind): len(self.legions_captured_mana_source_tiles_by_kind[str(kind)])
            for kind in self.MANA_KIND_ORDER
        }
        self.mana_income_per_turn = self._compute_mana_income_per_turn()
    def _refresh_faction_territories(self) -> None:
        """Главный пересчет текущих территорий фракций по номеру turn.

        Метод объединяет рост от стартовой столицы и всех активированных поселений,
        обновляет быстрые set-представления, доходные шахты, источники маны и observation cache.
        """
        legions_claim_count = self._territory_claim_count(
            len(self.legions_territory_order),
            self.LEGIONS_TERRITORY_EXPANSION_PER_TURN,
            self.turns,
        )
        capital_legions_tiles = self._claim_territory_tiles(
            self.legions_territory_order,
            legions_claim_count,
        )
        combined_legions_tiles: List[Tuple[int, int]] = []
        combined_legions_seen: set[Tuple[int, int]] = set()
        # Сначала добавляем основную территорию столицы, сохраняя порядок для стабильного render/obs.
        for tile in capital_legions_tiles:
            normalized_tile = (int(tile[0]), int(tile[1]))
            if normalized_tile in combined_legions_seen:
                continue
            combined_legions_seen.add(normalized_tile)
            combined_legions_tiles.append(normalized_tile)

        # Затем добавляем территории поселений, которые были открыты победой над required enemies.
        for source_data in self._static_legions_settlement_territory_sources:
            settlement_name = str(source_data.get("name", "") or "")
            source_order = self.legions_settlement_territory_orders_by_name.get(settlement_name, ())
            capture_turn = self.legions_active_settlement_territory_capture_turn_by_name.get(settlement_name)
            settlement_claim_count = self._settlement_territory_claim_count(
                len(source_order),
                int(source_data.get("expansion_per_turn", 0) or 0),
                self.turns,
                capture_turn,
            )
            settlement_tiles = self._claim_territory_tiles(source_order, settlement_claim_count)
            self.legions_settlement_territory_tiles_by_name[settlement_name] = settlement_tiles
            for tile in settlement_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in combined_legions_seen:
                    continue
                combined_legions_seen.add(normalized_tile)
                combined_legions_tiles.append(normalized_tile)

        self.legions_territory_tiles = tuple(combined_legions_tiles)
        self.legions_territory_tile_set = set(self.legions_territory_tiles)
        # Доходные объекты считаются производными от актуального набора контролируемых клеток.
        self.legions_captured_gold_mine_tiles = tuple(
            tile
            for tile in self.gold_mine_tiles
            if tuple(tile) in self.legions_territory_tile_set
        )
        self.legions_captured_gold_mine_count = len(self.legions_captured_gold_mine_tiles)
        self._refresh_legions_captured_mana_source_state()
        # Empire territory is optional for speed-focused training runs.
        if bool(getattr(self, "empire_territory_enabled", True)):
            empire_claim_count = self._territory_claim_count(
                len(self.empire_territory_order),
                self.EMPIRE_TERRITORY_EXPANSION_PER_TURN,
                self.turns,
            )
            self.empire_territory_tiles = self._claim_territory_tiles(
                self.empire_territory_order,
                empire_claim_count,
            )
            self.empire_territory_tile_set = set(self.empire_territory_tiles)
        else:
            self.empire_territory_tiles = ()
            self.empire_territory_tile_set = set()
        if hasattr(self, "grid_legions_territory_positions"):
            self._refresh_legions_territory_grid_obs_cache()
