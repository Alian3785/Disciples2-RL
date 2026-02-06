# campaign_env.py
"""
Двухуровневая среда: Grid World + Battle.
ВЕРСИЯ БЕЗ ВОССТАНОВЛЕНИЯ HP.

Агент перемещается по карте 10×10, встречает врагов и сражается с ними.
Всего 4 битвы с врагами разной сложности.

После каждой победы HP юнитов BLUE НЕ восстанавливается,
погибшие юниты НЕ воскрешаются. Только сбрасываются временные эффекты.

Схема наград:
  - За участие в битве (нашёл врага): +0.1
  - За победу в каждой битве: +0.3
  - За победу над всеми 4 врагами: +4.0
  - За поражение: -1.0
  - За таймаут (лимит шагов): -6.0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, List, Tuple
from copy import deepcopy

from battle_env import BattleEnv, UNITS_BLUE, FEATURES_PER_UNIT
from grid_world_env import GridWorldEnv
from enemy_configs import ENEMY_CONFIGS, ENEMY_DESCRIPTIONS
from data_dicts_compact_lines import DATA as UNIT_DATA, map_unit_to_battle
from Buildings import (
    empire_buildings_d2,
    mountain_clans_buildings_d2,
    undead_hordes_buildings_d2,
    legions_buildings_d2,
    elves_buildings_d2,
)

BUILDINGS_D2 = {
    "empire": empire_buildings_d2,
    "mountain_clans": mountain_clans_buildings_d2,
    "undead_hordes": undead_hordes_buildings_d2,
    "legions": legions_buildings_d2,
    "elves": elves_buildings_d2,
}
BUILDINGS_D2_TEMPLATE = {key: deepcopy(value) for key, value in BUILDINGS_D2.items()}


class CampaignEnv(gym.Env):
    """
    Двухуровневая среда: Grid World + Battle (4 битвы).

    Режимы:
        - MODE_GRID: агент перемещается по карте
        - MODE_BATTLE: агент сражается (используется BattleEnv)

    Награды:
        - Награда за участие в битве (нашёл врага): +0.1
        - Награда за победу в каждой битве: +0.3
        - Финальная награда после победы во всех 4 битвах: +4.0
        - Штраф за поражение: -1.0
        - Штраф за таймаут (лимит шагов на карте): -6.0

    Observation Space:
        Объединённое наблюдение:
        - [0]: режим (0 = grid, 1 = battle)
        - [1:1+GRID_OBS_SIZE]: grid observation
          (базовые признаки + координаты вражеских отрядов + тип/HP юнитов)
        - [...]: battle observation (когда в бою) или нули
        - [...]: ход и золото (каждые 20 шагов по карте)

    Action Space:
        Discrete(52):
        - В режиме grid:
          - 0-7: движение в 8 направлениях
          - 8: отдых (REST) — восстановление 5% HP раненым юнитам
          - 9-14: применить бутыль лечения к позициям BLUE 7-12 (+150 HP, не выше максимума)
          - 15-20: применить бутыль воскрешения к позициям BLUE 7-12 (оживляет с 1 HP)
          - 21-26: лечение в замке (полный HP) по позициям BLUE 7-12
          - 27-51: ????????? ????????? ??????? (25 ????????)
        - В режиме battle: используются действия 0-14 (атака/защита/ожидание)

        Action masking применяется для блокировки недопустимых действий.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    MODE_GRID = 0
    MODE_BATTLE = 1

    GRID_OBS_SIZE = 0  # вычисляется в __init__
    BATTLE_OBS_SIZE = FEATURES_PER_UNIT * 12
    GRID_STEPS_PER_TURN = 20
    GOLD_PER_TURN = 1.5
    GRID_BOTTLE_ACTION_START = 9
    GRID_BOTTLE_POSITIONS = list(range(7, 13))  # 6 позиций BLUE
    HEAL_BOTTLE_AMOUNT = 150.0
    MAX_HEAL_BOTTLES = 3
    GRID_REVIVE_ACTION_START = GRID_BOTTLE_ACTION_START + len(GRID_BOTTLE_POSITIONS)  # 15
    REVIVE_BOTTLE_POSITIONS = GRID_BOTTLE_POSITIONS
    MAX_REVIVE_BOTTLES = 3
    GRID_CASTLE_HEAL_ACTION_START = GRID_REVIVE_ACTION_START + len(REVIVE_BOTTLE_POSITIONS)  # 21
    CASTLE_HEAL_POSITIONS = GRID_BOTTLE_POSITIONS
    CASTLE_POS = (1, 1)
    GRID_BUILD_ACTION_START = GRID_CASTLE_HEAL_ACTION_START + len(CASTLE_HEAL_POSITIONS)  # 27

    def __init__(
        self,
        grid_size: int = 10,
        reward_engage_battle: float = 0.1,  # Награда за участие в битве (нашёл врага)
        reward_defeat_enemy: float = 0.3,   # Награда за победу в каждой битве
        reward_all_enemies: float = 4.0,    # Финальная награда за победу над всеми врагами
        reward_loss: float = -1.0,          # Штраф за поражение
        reward_timeout: float = -6.0,       # Штраф за таймаут (лимит шагов)
        reward_turn_penalty: float = 0.05,  # Штраф за каждый прошедший ход
        persist_blue_hp: bool = True,
        log_enabled: bool = False,
        max_grid_steps: int = 200,
        realcapital: int = 1,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.reward_engage_battle = reward_engage_battle  # Награда за участие в битве
        self.reward_defeat_enemy = reward_defeat_enemy    # Награда за победу в битве
        self.reward_all_enemies = reward_all_enemies      # Финальная награда
        self.reward_loss = reward_loss
        self.reward_timeout = reward_timeout
        self.reward_turn_penalty = reward_turn_penalty
        self.persist_blue_hp = persist_blue_hp
        self.log_enabled = log_enabled
        self.max_grid_steps = max_grid_steps
        self.turn_norm_k = max(
            1.0,
            float(self.max_grid_steps) / float(self.GRID_STEPS_PER_TURN),
        )
        self.gold_norm_k = max(1.0, self.turn_norm_k * float(self.GOLD_PER_TURN))
        # Realcapital: 1 = empire, 2 = legions, 3 = mountain_clans, 4 = undead_hordes, 5 = elves
        self.Realcapital = self._normalize_realcapital(realcapital)

        # Подсреды — все 4 врага
        self.grid_env = GridWorldEnv(
            grid_size=grid_size,
            max_steps=max_grid_steps,
            enemy_positions={
                1: (3, 5),   # Враг 1: слабый (центр-лево)
                2: (7, 3),   # Враг 2: средний (право-верх)
                3: (5, 7),   # Враг 3: отряд с орком и гоблинами (центр-низ)
                4: (8, 8),   # Враг 4: 3 орка (верх-право)
            },
        )
        self.battle_env: Optional[BattleEnv] = None
        self.grid_obs_base_size = int(self.grid_env.observation_space.shape[0])
        self._init_enemy_obs_metadata()
        self.GRID_OBS_SIZE = self.grid_obs_base_size + self.grid_enemy_obs_size

        # Текущий режим
        self.mode = self.MODE_GRID
        self.current_enemy_id: Optional[int] = None

        # Сохраняем состояние BLUE команды между боями
        self.blue_team_state: Optional[List[Dict]] = None
        self.heal_bottles_used: int = 0
        self.revive_bottles_used: int = 0
        self.turns: int = 0
        self.gold: float = 0.0
        self.active_buildings = self._get_buildings_for_capital(self.Realcapital)
        self.building_keys = self._get_building_keys(self.active_buildings)
        total_actions = self.GRID_BUILD_ACTION_START + len(self.building_keys)
        # Action space: максимум из двух режимов (battle = 15 действий, grid расширен для бутылей и построек)
        self.action_space = spaces.Discrete(total_actions)

        # Observation space: режим + grid + battle + ход + золото
        total_obs = 1 + self.GRID_OBS_SIZE + self.BATTLE_OBS_SIZE + 2
        self.observation_space = spaces.Box(
            low=np.zeros(total_obs, dtype=np.float32),
            high=np.ones(total_obs, dtype=np.float32),
            shape=(total_obs,),
            dtype=np.float32,
        )

        # Логи
        self._campaign_logs: List[str] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._reset_buildings_state()
        self.active_buildings = self._get_buildings_for_capital(self.Realcapital)
        self.building_keys = self._get_building_keys(self.active_buildings)
        self.mode = self.MODE_GRID
        self.current_enemy_id = None
        # Стартовое состояние BLUE — копия базовых юнитов, чтобы
        # вне боя можно было применять лечение до первого боя.
        self.blue_team_state = deepcopy(UNITS_BLUE)
        self.heal_bottles_used = 0
        self.revive_bottles_used = 0
        self.turns = 0
        self.gold = 0.0
        self.battle_env = None
        self._campaign_logs = []

        grid_obs, grid_info = self.grid_env.reset(seed=seed)
        grid_obs = self._augment_grid_obs(grid_obs)

        self._log("=== НОВАЯ КАМПАНИЯ ===")
        self._log(f"Старт на позиции {self.grid_env.agent_pos}")
        self._log(f"Враги на карте: {list(self.grid_env.enemy_positions.items())}")

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "turns": self.turns,
            "gold": self.gold,
        }

        return self._build_obs(grid_obs=grid_obs), info

    def step(self, action: int):
        # Приводим action к int (может прийти как numpy.ndarray)
        action = int(action)

        if self.mode == self.MODE_GRID:
            return self._step_grid(action)
        else:
            return self._step_battle(action)

    def _step_grid(self, action: int):
        """Шаг в режиме перемещения по карте."""
        # В grid режиме:
        #   0-7 — движение, 8 — отдых,
        #   9-14 — применить бутыль лечения к позициям 7-12 соответственно.
        #   15-20 — применить бутыль воскрешения к позициям 7-12 соответственно.
        #   21-26 — лечение в замке по позициям 7-12.
        #   27-51 ? ????????? ?????? ????????? ??????? (25 ????????).
        if action >= self.GRID_BUILD_ACTION_START:
            return self._step_building(action)
        if action >= self.GRID_CASTLE_HEAL_ACTION_START:
            return self._step_heal_in_castle(action)
        if action >= self.GRID_REVIVE_ACTION_START:
            return self._step_revive_with_bottle(action)
        if action >= self.GRID_BOTTLE_ACTION_START:
            return self._step_heal_with_bottle(action)

        grid_action = min(action, 8)

        old_pos = self.grid_env.agent_pos
        grid_obs, reward, terminated, truncated, grid_info = self.grid_env.step(
            grid_action
        )
        grid_obs = self._augment_grid_obs(grid_obs)
        new_pos = self.grid_env.agent_pos

        if old_pos != new_pos:
            self._log(f"Перемещение: {old_pos} -> {new_pos}")

        if grid_info.get("battle_triggered"):
            self._apply_battle_grid_steps(extra_steps=10)

        reward += self._update_turns()

        info = {
            "mode": "grid",
            "agent_pos": new_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": grid_info.get("battle_triggered", False),
            "rest_action": grid_info.get("rest_action", False),
            "turns": self.turns,
            "gold": self.gold,
        }

        # Обработка действия REST — восстановление 5% HP
        if grid_info.get("rest_action"):
            healed = self._heal_blue_team(heal_percent=0.05)
            if healed > 0:
                self._log(f"Отдых: восстановлено HP у {healed} юнитов (5%)")
            else:
                self._log("Отдых: все юниты на полном здоровье")
            # Отдых завершает ход
            reward += self._advance_turns(1)
            info["turns"] = self.turns
            info["gold"] = self.gold
            info["healed_units"] = healed

        # Проверяем столкновение с врагом
        if grid_info.get("battle_triggered"):
            enemy_id = grid_info["enemy_id"]
            self.current_enemy_id = enemy_id
            self.mode = self.MODE_BATTLE

            self._log(f"!!! СТОЛКНОВЕНИЕ С ВРАГОМ {enemy_id} !!!")
            self._log(f"Описание: {ENEMY_DESCRIPTIONS.get(enemy_id, 'Неизвестный враг')}")

            # Награда за участие в битве (нашёл врага!)
            reward += self.reward_engage_battle
            self._log(f"Награда за участие в битве: +{self.reward_engage_battle}")

            # Создаём BattleEnv с нужной RED командой
            self._init_battle(enemy_id)

            # Возвращаем battle observation
            battle_obs = self.battle_env._obs()
            info["mode"] = "battle"
            info["enemy_id"] = enemy_id

            return self._build_obs(battle_obs=battle_obs), reward, False, False, info

        # Проверяем победу (все враги побеждены)
        if self.grid_env.all_enemies_defeated():
            reward += self.reward_all_enemies
            self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
            info["campaign_result"] = "victory"
            return self._build_obs(grid_obs=grid_obs), reward, True, False, info

        # Проверяем лимит шагов на карте
        if truncated:
            self._log("=== ЛИМИТ ШАГОВ НА КАРТЕ ИСЧЕРПАН ===")
            reward += self.reward_timeout
            self._log(f"Штраф за таймаут: {self.reward_timeout}")
            info["campaign_result"] = "timeout"

        return self._build_obs(grid_obs=grid_obs), reward, terminated, truncated, info

    def _is_at_castle(self) -> bool:
        """True если агент стоит на клетке замка (1, 1) в grid-режиме."""
        return tuple(self.grid_env.agent_pos) == tuple(self.CASTLE_POS)

    def _heal_unit_to_full_at_position(self, position: int) -> Tuple[float, Optional[str]]:
        """Лечит одного юнита BLUE на позиции до max_hp. Возвращает (сколько вылечили, имя)."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)

            if max_hp <= 0 or hp <= 0 or hp >= max_hp:
                return 0.0, None

            healed = max_hp - hp
            unit["hp"] = max_hp
            unit["health"] = max_hp

            name = unit.get("name", f"pos_{position}")
            self._log(
                f"Castle heal: {name} (pos {position}) HP {hp:.1f} -> {max_hp:.1f} (+{healed:.1f})"
            )
            return healed, name

        return 0.0, None

    def _step_heal_in_castle(self, action: int):
        """Лечит выбранного юнита BLUE до полного HP, только если агент стоит в замке (1,1)."""
        idx = action - self.GRID_CASTLE_HEAL_ACTION_START
        target_pos = (
            self.CASTLE_HEAL_POSITIONS[idx]
            if 0 <= idx < len(self.CASTLE_HEAL_POSITIONS)
            else None
        )

        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        healed_amount, unit_name = (0.0, None)
        if self._is_at_castle() and target_pos is not None:
            healed_amount, unit_name = self._heal_unit_to_full_at_position(position=target_pos)

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "castle_heal_action": True,
            "castle_pos_required": self.CASTLE_POS,
            "castle_heal_available": self._is_at_castle(),
            "castle_heal_target_pos": target_pos,
            "healed_amount": healed_amount,
            "healed_unit_name": unit_name,
            "turns": self.turns,
            "gold": self.gold,
        }

        return self._build_obs(grid_obs=grid_obs), reward, terminated, truncated, info

    def _normalize_realcapital(self, value: int) -> int:
        """??????????? Realcapital: ????????? ?????? ???????? 1-5, ????? ?????? 1 (???????)."""
        try:
            realcapital_value = int(value)
        except (TypeError, ValueError):
            return 1
        return realcapital_value if realcapital_value in (1, 2, 3, 4, 5) else 1

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _get_building_keys(self, buildings: Dict) -> List[str]:
        """?????????? ????? ?????? (??? ????????? ????-?????? ????? 'alredybuilt')."""
        keys: List[str] = []
        for key, entry in buildings.items():
            if isinstance(entry, dict) and entry.get("id"):
                keys.append(key)
        return keys

    def _reset_buildings_state(self) -> None:
        """Сбрасывает состояние зданий к эталону из Buildings.py."""
        for key, template in BUILDINGS_D2_TEMPLATE.items():
            target = BUILDINGS_D2.get(key)
            if not isinstance(target, dict) or not isinstance(template, dict):
                continue
            target.clear()
            target.update(deepcopy(template))

    def get_built_building_names(self) -> List[str]:
        """Возвращает названия построенных зданий активной фракции в порядке ключей."""
        built_names: List[str] = []
        for key in self.building_keys:
            building = self.active_buildings.get(key)
            if not isinstance(building, dict):
                continue
            built_flag = building.get("Build", building.get("built", 0))
            try:
                is_built = int(built_flag or 0) == 1
            except (TypeError, ValueError):
                is_built = False
            if not is_built:
                continue
            name = str(building.get("name", "") or "").strip()
            if name:
                built_names.append(name)
        return built_names

    def _step_building(self, action: int):
        """Выбирает здание активной фракции и отмечает его как построенное."""
        idx = action - self.GRID_BUILD_ACTION_START
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        build_key = None
        build_name = None
        built = False
        already_built = False
        blocked_names: List[str] = []
        build_cost = None
        insufficient_gold = False
        build_blocked = False
        build_locked = False

        if 0 <= idx < len(self.building_keys):
            build_key = self.building_keys[idx]
            building = self.active_buildings.get(build_key)
            if isinstance(building, dict):
                build_name = building.get("name")
                price_raw = building.get("gold", 0)
                try:
                    build_cost = float(price_raw)
                except (TypeError, ValueError):
                    build_cost = 0.0
                alredybuilt_flag = self.active_buildings.get("alredybuilt", 0)
                try:
                    build_locked = int(alredybuilt_flag or 0) == 1
                except (TypeError, ValueError):
                    build_locked = False
                built_flag = building.get("Build", building.get("built", 0))
                already_built = int(built_flag or 0) == 1
                blocked_flag = building.get("blocked", 0)
                build_blocked = int(blocked_flag or 0) == 1
                if build_locked:
                    pass
                elif build_blocked:
                    pass
                elif self.gold < (build_cost or 0):
                    insufficient_gold = True
                elif not already_built:
                    self.gold -= build_cost or 0
                    building["built"] = 1
                    built = True
                    self.active_buildings["alredybuilt"] = 1

                    blocks_raw = building.get("blocks", [])
                    if isinstance(blocks_raw, (list, tuple)):
                        block_names = [
                            str(name).strip() for name in blocks_raw if str(name).strip()
                        ]
                    elif blocks_raw:
                        block_names = [str(blocks_raw).strip()]
                    else:
                        block_names = []

                    if block_names:
                        for entry in self.active_buildings.values():
                            if not isinstance(entry, dict):
                                continue
                            entry_name = str(entry.get("name", "") or "").strip()
                            if entry_name and entry_name in block_names:
                                entry["blocked"] = 1
                                blocked_names.append(entry_name)

                    if build_name:
                        self._log("ЗДАНИЕ ПОСТРОЕНО")
                        self._log(f'Построено здание "{build_name}"')
                        if blocked_names:
                            self._log(f'Заблокированы здания: {", ".join(blocked_names)}')

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "build_action": True,
            "build_key": build_key,
            "build_name": build_name,
            "built": built,
            "already_built": already_built,
            "blocked_names": blocked_names,
            "build_cost": build_cost,
            "insufficient_gold": insufficient_gold,
            "build_blocked": build_blocked,
            "build_locked": build_locked,
            "turns": self.turns,
            "gold": self.gold,
        }

        return self._build_obs(grid_obs=grid_obs), reward, terminated, truncated, info

    def _step_heal_with_bottle(self, action: int):
        """Применяет бутыль лечения к указанной позиции BLUE в режиме grid."""
        idx = action - self.GRID_BOTTLE_ACTION_START
        target_pos = self.GRID_BOTTLE_POSITIONS[idx] if 0 <= idx < len(self.GRID_BOTTLE_POSITIONS) else None

        # Лечение не продвигает ход карты и не даёт штрафа/таймера.
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        # Считаем использование бутыли независимо от успешности лечения
        self.heal_bottles_used = min(self.heal_bottles_used + 1, self.MAX_HEAL_BOTTLES)

        healed_amount, unit_name = (0.0, None)
        if target_pos is not None:
            healed_amount, unit_name = self._heal_unit_at_position(
                position=target_pos, heal_amount=self.HEAL_BOTTLE_AMOUNT
            )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "heal_action": True,
            "heal_target_pos": target_pos,
            "healed_amount": healed_amount,
            "healed_unit_name": unit_name,
            "heal_bottles_used": self.heal_bottles_used,
            "heal_bottles_left": max(0, self.MAX_HEAL_BOTTLES - self.heal_bottles_used),
            "turns": self.turns,
            "gold": self.gold,
        }

        return self._build_obs(grid_obs=grid_obs), reward, terminated, truncated, info

    def _step_revive_with_bottle(self, action: int):
        """Применяет бутыль воскрешения к указанной позиции BLUE в режиме grid."""
        idx = action - self.GRID_REVIVE_ACTION_START
        target_pos = (
            self.REVIVE_BOTTLE_POSITIONS[idx]
            if 0 <= idx < len(self.REVIVE_BOTTLE_POSITIONS)
            else None
        )

        # Воскрешение не продвигает ход карты и не даёт штрафа/таймера.
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        self.revive_bottles_used = min(self.revive_bottles_used + 1, self.MAX_REVIVE_BOTTLES)

        revived, unit_name = (False, None)
        if target_pos is not None:
            revived, unit_name = self._revive_unit_at_position(position=target_pos)

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "revive_action": True,
            "revive_target_pos": target_pos,
            "revived": revived,
            "revived_unit_name": unit_name,
            "revive_bottles_used": self.revive_bottles_used,
            "revive_bottles_left": max(0, self.MAX_REVIVE_BOTTLES - self.revive_bottles_used),
            "turns": self.turns,
            "gold": self.gold,
        }

        return self._build_obs(grid_obs=grid_obs), reward, terminated, truncated, info

    def _step_battle(self, action: int):
        """Шаг в режиме боя."""
        if self.battle_env is None:
            # Защита от некорректного состояния
            self.mode = self.MODE_GRID
            grid_obs = self._get_grid_obs()
            return self._build_obs(grid_obs=grid_obs), 0.0, False, False, {"error": "no_battle_env"}

        # Проверка: бой уже завершён (например, RED победил в свой ход)
        if self.battle_env.winner is not None:
            winner = self.battle_env.winner
            obs = self.battle_env._obs()
            reward = 0.0
            terminated = True
            truncated = False
        else:
            # safety: action space CampaignEnv > BattleEnv, clamp to valid range
            action = int(action)
            if action < 0:
                action = 0
            if action >= self.battle_env.action_space.n:
                action = self.battle_env.action_space.n - 1
            obs, reward, terminated, truncated, battle_info = self.battle_env.step(action)

        info = {
            "mode": "battle",
            "enemy_id": self.current_enemy_id,
            "battle_step": True,
            "turns": self.turns,
            "gold": self.gold,
        }

        if terminated or truncated:
            # После окончания боя добавляем 1 ход
            reward += self._advance_turns(1)
            info["turns"] = self.turns
            info["gold"] = self.gold
            winner = self.battle_env.winner

            if winner == "blue":
                # Победа: помечаем врага побеждённым
                self._log(f"=== ПОБЕДА В БОЮ ПРОТИВ ВРАГА {self.current_enemy_id}! ===")
                self.grid_env.mark_enemy_defeated(self.current_enemy_id)
                reward += self.reward_defeat_enemy
                self._log(f"Награда за победу в битве: +{self.reward_defeat_enemy}")

                # Восстанавливаем и воскрешаем всех юнитов BLUE после победы
                if self.persist_blue_hp:
                    self._save_blue_state()
                    self._log("Состояние BLUE сохранено (без автолечения и воскрешений)")

                if self.battle_env is not None:
                    self._log(f"Опыт за бой: {self.battle_env.last_battle_exp:g}")
                    for name in getattr(self.battle_env, "last_levelups", []) or []:
                        self._log(f"Уровень юнита {name} повышен")
                    self._log_turns_into_levelups()

                # Возвращаемся в grid режим
                self.mode = self.MODE_GRID
                self.battle_env = None
                info["battle_result"] = "victory"
                info["mode"] = "grid"

                # Проверяем полную победу
                if self.grid_env.all_enemies_defeated():
                    reward += self.reward_all_enemies
                    self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
                    grid_obs = self._get_grid_obs()
                    info["campaign_result"] = "victory"
                    return self._build_obs(grid_obs=grid_obs), reward, True, False, info

                grid_obs = self._get_grid_obs()
                info["agent_pos"] = self.grid_env.agent_pos
                info["enemies_alive"] = dict(self.grid_env.enemies_alive)
                return self._build_obs(grid_obs=grid_obs), reward, False, False, info

            else:
                # Поражение: конец эпизода
                self._log(f"=== ПОРАЖЕНИЕ В БОЮ ПРОТИВ ВРАГА {self.current_enemy_id}! ===")
                self._log("=== КАМПАНИЯ ПРОВАЛЕНА ===")
                reward += self.reward_loss
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

    def _init_battle(self, enemy_id: int):
        """Инициализирует бой с конкретной RED командой."""
        red_team = deepcopy(ENEMY_CONFIGS.get(enemy_id, ENEMY_CONFIGS[1]))

        # Восстанавливаем состояние BLUE или берём дефолтное
        if self.persist_blue_hp and self.blue_team_state is not None:
            blue_team = deepcopy(self.blue_team_state)
            self._log("Загружено сохранённое состояние BLUE команды")
        else:
            blue_team = deepcopy(UNITS_BLUE)
            self._log("Используется дефолтная BLUE команда")

        self.battle_env = BattleEnv(log_enabled=self.log_enabled)

        # Инициализируем бой с кастомными командами
        self.battle_env._init_with_custom_teams(red_team, blue_team)

    def _save_blue_state(self):
        """
        Сохраняет состояние BLUE команды после победы.
        HP НЕ восстанавливается, погибшие НЕ воскрешаются.
        Только сбрасываются временные эффекты (armor boost, paralysis, poison и т.д.).
        """
        if self.battle_env is None:
            return

        # Сохраняем только базовые позиции BLUE (без призванных юнитов)
        base_positions = {u["position"] for u in UNITS_BLUE}

        self.blue_team_state = []

        for u in self.battle_env.combined:
            if u.get("team") != "blue":
                continue
            
            pos = u["position"]
            if pos not in base_positions:
                # Юнит не относится к базовым позициям (призванный?) — пропускаем
                continue

            # Сохраняем юнита в текущей форме (после улучшений)
            restored_unit = deepcopy(u)
            
            # СОХРАНЯЕМ ТЕКУЩЕЕ HP (не восстанавливаем!)
            current_hp = max(0.0, float(u.get("health", 0) or u.get("hp", 0)))
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
            restored_unit.pop("original_damage", None)
            # Armor must be restored to the unit's battle-start value.
            base_armor = int(restored_unit.get("base_armor", restored_unit.get("armor", 0)) or 0)
            restored_unit["armor"] = max(0, base_armor)
            
            # Восстанавливаем initiative к базовому значению
            restored_unit["initiative"] = restored_unit.get("initiative_base", 0)
            
            self.blue_team_state.append(restored_unit)
        
        self._log("Состояние BLUE команды сохранено (HP не восстановлено)")

    def _find_unit_data_by_name(self, name: str) -> Optional[Dict]:
        for entry in UNIT_DATA:
            if entry.get("кто") == name:
                return entry
        return None

    def _get_buildings_for_capital(self, capital: Optional[int]) -> Dict:
        """Возвращает словарь построек по значению 'столица'."""
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            capital_value = None

        if capital_value == 2:
            return legions_buildings_d2
        if capital_value == 3:
            return mountain_clans_buildings_d2
        if capital_value == 4:
            return undead_hordes_buildings_d2
        if capital_value == 5:
            return elves_buildings_d2
        return empire_buildings_d2

    def _build_unit_from_data(self, entry: Dict, team: str, position: int) -> Dict:
        new_unit = map_unit_to_battle(entry, team, position)
        new_unit["exp_kill"] = entry.get("опыт убийства", 0)
        new_unit["exp_required"] = entry.get("нужный опыт", 0)
        new_unit["exp_current"] = entry.get("текущий опыт", 0)
        turns_raw = entry.get("превращается в", [])
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
                return

    def _log_turns_into_levelups(self) -> None:
        """Логирует повышение до юнита, если есть построенное здание."""
        if self.battle_env is None:
            return
        levelup_names = getattr(self.battle_env, "last_levelups", []) or []
        if not levelup_names:
            return

        name_to_units: Dict[str, List[Dict]] = {}
        for unit in getattr(self.battle_env, "combined", []) or []:
            name = str(unit.get("name", "") or "").strip()
            if not name:
                continue
            name_to_units.setdefault(name, []).append(unit)

        upgraded_any = False
        for name in levelup_names:
            unit_data = self._find_unit_data_by_name(name)
            capital_value = unit_data.get("столица") if unit_data else None
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
                    self._log(f'Юнит "{unit_label}" повышен до "{target_name}"')
                    unit_data = self._find_unit_data_by_name(target_name)
                    if unit_data is not None:
                        upgraded_unit = self._build_unit_from_data(
                            unit_data,
                            unit.get("team", "blue"),
                            unit.get("position", pos),
                        )
                        unit.clear()
                        unit.update(deepcopy(upgraded_unit))
                        self._replace_blue_unit(upgraded_unit)
                        upgraded_any = True
                    break

        if upgraded_any:
            return

    def _heal_blue_team(self, heal_percent: float = 0.05) -> int:
        """
        Восстанавливает HP раненым юнитам BLUE команды.
        
        Args:
            heal_percent: Доля от max_health для восстановления (0.05 = 5%)
            
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
                heal_amount = max_hp * heal_percent
                new_hp = min(max_hp, hp + heal_amount)
                unit["hp"] = new_hp
                unit["health"] = new_hp
                healed_count += 1
                
        return healed_count

    def _heal_unit_at_position(self, position: int, heal_amount: float) -> Tuple[float, Optional[str]]:
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
                f"Бутыль лечения: {name} (pos {position}) HP {hp:.1f} -> {new_hp:.1f} (+{healed:.1f})"
            )
            return healed, name

        return 0.0, None

    def _get_blue_state(self) -> List[Dict]:
        """Возвращает (и при необходимости инициализирует) сохранённое состояние BLUE."""
        if self.blue_team_state is None:
            self.blue_team_state = deepcopy(UNITS_BLUE)
        return self.blue_team_state

    def _apply_battle_grid_steps(self, extra_steps: int = 10) -> None:
        """Увеличивает счётчик шагов по гриду из-за боя, не выше лимита хода."""
        if extra_steps <= 0:
            return
        steps_into_turn = int(self.grid_env.step_count) % self.GRID_STEPS_PER_TURN
        if steps_into_turn + extra_steps >= self.GRID_STEPS_PER_TURN:
            self.grid_env.step_count += self.GRID_STEPS_PER_TURN - steps_into_turn
        else:
            self.grid_env.step_count += extra_steps

    def _update_turns(self) -> float:
        """Обновляет количество ходов и золота по шагам в grid-режиме."""
        grid_steps = int(self.grid_env.step_count)
        new_turns = grid_steps // self.GRID_STEPS_PER_TURN
        if new_turns > self.turns:
            return self._advance_turns(new_turns - self.turns)
        return 0.0

    def _advance_turns(self, delta_turns: int) -> float:
        """Увеличивает счётчик ходов и золото на заданное количество."""
        if delta_turns <= 0:
            return 0.0
        self.turns += int(delta_turns)
        self.gold += float(delta_turns) * self.GOLD_PER_TURN
        self.active_buildings["alredybuilt"] = 0
        return -float(delta_turns) * self.reward_turn_penalty

    @staticmethod
    def _is_empty_enemy_unit(unit: Dict) -> bool:
        """True если слот вражеского отряда пустой."""
        name = (unit.get("name") or "").strip().lower()
        if name == "пусто":
            return True
        hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0)
        max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0)
        return hp <= 0 and max_hp <= 0

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
        self.enemy_unit_type_to_id = {t: i + 1 for i, t in enumerate(self.enemy_unit_types)}
        self.enemy_unit_type_scale = max(1, len(self.enemy_unit_types))
        self.grid_enemy_unit_slots = max_units
        self.grid_enemy_count = len(self.grid_env.enemy_positions or {})
        self.grid_enemy_obs_size = self.grid_enemy_count * (
            2 + self.grid_enemy_unit_slots * 2
        )

    def _normalize_enemy_unit_type(self, unit_type: Optional[str]) -> float:
        type_id = self.enemy_unit_type_to_id.get(unit_type or "", 0)
        return float(type_id) / float(self.enemy_unit_type_scale)

    @staticmethod
    def _normalize_unit_health(unit: Dict) -> float:
        hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0)
        max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or hp)
        if max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, hp / max_hp))

    def _build_enemy_grid_obs(self) -> np.ndarray:
        """Строит признаки врагов на карте для grid-наблюдения."""
        if self.grid_enemy_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        enemy_obs: List[float] = []

        for enemy_id in sorted(self.grid_env.enemy_positions.keys()):
            ex, ey = self.grid_env.enemy_positions.get(enemy_id, (0, 0))
            enemy_obs.extend([ex / grid_scale, ey / grid_scale])

            is_alive = self.grid_env.enemies_alive.get(enemy_id, False)
            team = ENEMY_CONFIGS.get(enemy_id, [])
            team_sorted = sorted(team, key=lambda u: int(u.get("position", 0)))

            for idx in range(self.grid_enemy_unit_slots):
                if not is_alive or idx >= len(team_sorted):
                    enemy_obs.extend([0.0, 0.0])
                    continue

                unit = team_sorted[idx]
                if self._is_empty_enemy_unit(unit):
                    enemy_obs.extend([0.0, 0.0])
                    continue

                type_norm = self._normalize_enemy_unit_type(unit.get("unit_type"))
                hp_norm = self._normalize_unit_health(unit)
                enemy_obs.extend([type_norm, hp_norm])

        return np.array(enemy_obs, dtype=np.float32)

    def _augment_grid_obs(self, base_obs: np.ndarray) -> np.ndarray:
        enemy_obs = self._build_enemy_grid_obs()
        if enemy_obs.size == 0:
            return base_obs.astype(np.float32, copy=False)
        return np.concatenate([base_obs.astype(np.float32, copy=False), enemy_obs])

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

    def compute_action_mask(self) -> np.ndarray:
        """Маска действий в зависимости от режима."""
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.mode == self.MODE_GRID:
            # В grid режиме доступны первые 8 действий (движение) всегда,
            # а REST (8) — только если есть раненые юниты BLUE.
            mask[:8] = True
            mask[8] = self._has_wounded_blue()
            # Бутыли лечения на позиции 7-12.
            if self.heal_bottles_used < self.MAX_HEAL_BOTTLES:
                for idx, pos in enumerate(self.GRID_BOTTLE_POSITIONS):
                    mask[self.GRID_BOTTLE_ACTION_START + idx] = self._can_heal_position(pos)
            # Бутыли воскрешения на позиции 7-12.
            if self.revive_bottles_used < self.MAX_REVIVE_BOTTLES:
                for idx, pos in enumerate(self.REVIVE_BOTTLE_POSITIONS):
                    mask[self.GRID_REVIVE_ACTION_START + idx] = self._can_revive_position(pos)

            # Лечение в замке: доступно только на клетке (1, 1)
            if self._is_at_castle():
                for idx, pos in enumerate(self.CASTLE_HEAL_POSITIONS):
                    mask[self.GRID_CASTLE_HEAL_ACTION_START + idx] = self._can_heal_position(pos)

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
                price_raw = building.get("gold", 0)
                try:
                    price = float(price_raw)
                except (TypeError, ValueError):
                    price = 0.0
                built_flag = building.get("Build", building.get("built", 0))
                is_built = int(built_flag or 0) == 1
                blocked_flag = building.get("blocked", 0)
                is_blocked = int(blocked_flag or 0) == 1
                mask[self.GRID_BUILD_ACTION_START + idx] = (
                    not has_build_lock and not is_built and not is_blocked and self.gold >= price
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

    def _log(self, message: str):
        """Добавляет сообщение в лог кампании."""
        self._campaign_logs.append(message)
        if self.log_enabled:
            print(f"[CAMPAIGN] {message}")

    def pop_campaign_logs(self) -> List[str]:
        """Возвращает и очищает логи кампании."""
        logs = self._campaign_logs[:]
        self._campaign_logs.clear()
        return logs

    def pop_battle_logs(self) -> List[str]:
        """Возвращает логи текущего боя (если есть)."""
        if self.battle_env is not None:
            return self.battle_env.pop_pretty_events()
        return []

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

    def render(self, mode: str = "ansi") -> Optional[str]:
        """Отрисовка текущего состояния."""
        if mode != "ansi":
            return None

        lines = []
        lines.append(f"=== CAMPAIGN MODE: {'GRID' if self.mode == self.MODE_GRID else 'BATTLE'} ===")

        if self.mode == self.MODE_GRID:
            lines.append("")
            grid_render = self.grid_env.render(mode="ansi")
            if grid_render:
                lines.append(grid_render)
            lines.append(f"Ход: {self.turns} | Золото: {self.gold:g}")
        else:
            lines.append(f"В бою с врагом {self.current_enemy_id}")
            lines.append(f"Описание: {ENEMY_DESCRIPTIONS.get(self.current_enemy_id, '?')}")
            lines.append(f"Ход: {self.turns} | Золото: {self.gold:g}")

        result = "\n".join(lines)
        print(result)
        return result

    def get_state_summary(self) -> Dict:
        """Возвращает сводку текущего состояния."""
        return {
            "mode": "grid" if self.mode == self.MODE_GRID else "battle",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "enemies_defeated": [
                eid for eid, alive in self.grid_env.enemies_alive.items() if not alive
            ],
            "visited_cells": len(self.grid_env.visited_cells),
            "grid_steps": self.grid_env.step_count,
            "current_enemy": self.current_enemy_id,
            "blue_hp_saved": self.blue_team_state is not None,
            "turns": self.turns,
            "gold": self.gold,
        }
