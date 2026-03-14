# campaign_env.py
"""
Двухуровневая среда: Grid World + Battle.
ВЕРСИЯ БЕЗ ВОССТАНОВЛЕНИЯ HP.

Агент перемещается по карте 48×48 (по умолчанию), встречает врагов и сражается с ними.
Список врагов и их позиции соответствуют совпавшим отрядам из сценарной карты.

После каждой победы HP юнитов BLUE НЕ восстанавливается,
погибшие юниты НЕ воскрешаются. Только сбрасываются временные эффекты.

Схема наград:
  - За участие в битве (нашёл врага): +0.1
  - За победу в каждой битве: +0.3
  - За полную зачистку карты: финал кампании
  - За поражение: -1.0
  - За таймаут (лимит шагов): -6.0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, List, Tuple
from copy import deepcopy

from battle_env import BattleEnv, UNITS_BLUE, FEATURES_PER_UNIT
from enemy_configs import ENEMY_CONFIGS, ENEMY_DESCRIPTIONS
from grid import (
    are_targets_reachable,
    DEFAULT_GRID_SIZE,
    DEFAULT_HERO_GRID_POSITION,
    HEAL_TILE_COUNT,
    resolve_obstacle_tiles,
    resolve_heal_tiles,
    scale_enemy_positions,
)
from grid_world_env import GridWorldEnv
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
    Двухуровневая среда: Grid World + Battle.

    Режимы:
        - MODE_GRID: агент перемещается по карте
        - MODE_BATTLE: агент сражается (используется BattleEnv)

    Награды:
        - Награда за участие в битве (нашёл врага): +0.1
        - Награда за победу в каждой битве: +0.3
        - Финальный бонус за победу над зелёным драконом: reward_all_enemies * 10
        - Штраф за поражение: -1.0
        - Штраф за таймаут (лимит шагов на карте): -6.0

    Observation Space:
        Объединённое наблюдение:
        - [0]: режим (0 = grid, 1 = battle)
        - [1:1+GRID_OBS_SIZE]: grid observation
          (базовые признаки + координаты вражеских отрядов + тип/HP юнитов)
        - [...]: battle observation (когда в бою) или нули
        - [...]: ход и золото

    Action Space:
        Discrete(58):
        - В режиме grid:
          - 0-7: движение в 8 направлениях
          - 8: отдых (REST) — восстановление 5% HP раненым юнитам
          - 9-14: применить подходящую банку лечения к позициям BLUE 7-12
            (автовыбор +50 HP или +100 HP в зависимости от недостающего здоровья и запасов)
          - 15-20: применить бутыль воскрешения к позициям BLUE 7-12 (оживляет с 1 HP)
          - 21-26: лечение на специальных клетках карты по позициям BLUE 7-12
            (стоимость за 1 HP зависит от Level)
          - 27-32: воскрешение на специальных клетках карты по позициям BLUE 7-12
            (стоимость зависит от Level)
          - 33-57: ????????? ????????? ??????? (25 ????????)
        - В режиме battle: используются действия 0-14 (атака/защита/ожидание)

        Action masking применяется для блокировки недопустимых действий.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    MODE_GRID = 0
    MODE_BATTLE = 1

    GRID_OBS_SIZE = 0  # вычисляется в __init__
    BATTLE_OBS_SIZE = FEATURES_PER_UNIT * 12
    MOVES_PER_TURN = 20
    GRID_STEPS_PER_TURN = MOVES_PER_TURN
    HERO_PATHFINDING_LEVEL = 2
    HERO_PATHFINDING_MOVE_BONUS_PCT = 0.20
    GOLD_PER_TURN = 1.5
    GRID_BOTTLE_ACTION_START = 9
    GRID_BOTTLE_POSITIONS = list(range(7, 13))  # 6 позиций BLUE
    HEAL_BOTTLE_AMOUNT = 100.0
    MAX_HEAL_BOTTLES = 3
    HEALING_BOTTLE_AMOUNT = 50.0
    MAX_HEALING_BOTTLES = 3
    GRID_REVIVE_ACTION_START = GRID_BOTTLE_ACTION_START + len(GRID_BOTTLE_POSITIONS)  # 15
    REVIVE_BOTTLE_POSITIONS = GRID_BOTTLE_POSITIONS
    MAX_REVIVE_BOTTLES = 3
    CASTLE_POS = DEFAULT_HERO_GRID_POSITION
    CASTLE_HEAL_TILE_COUNT = HEAL_TILE_COUNT
    GRID_CASTLE_HEAL_ACTION_START = GRID_REVIVE_ACTION_START + len(REVIVE_BOTTLE_POSITIONS)  # 21
    CASTLE_HEAL_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_CASTLE_REVIVE_ACTION_START = GRID_CASTLE_HEAL_ACTION_START + len(CASTLE_HEAL_POSITIONS)  # 27
    CASTLE_REVIVE_POSITIONS = REVIVE_BOTTLE_POSITIONS
    GRID_BUILD_ACTION_START = GRID_CASTLE_REVIVE_ACTION_START + len(CASTLE_REVIVE_POSITIONS)  # 33
    GREEN_DRAGON_ENEMY_ID = 31
    GREEN_DRAGON_REWARD_MULTIPLIER = 10.0
    ENEMY_REWARD_MULTIPLIERS = {}

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        reward_engage_battle: float = 0.2,  # Бонус за вход в бой с живым врагом.
        reward_defeat_enemy: float = 1.0,   # Награда за победу в бою против каждого отряда
        reward_all_enemies: float = 10.0,   # Базовая финальная награда (зелёный дракон получает x10)
        reward_loss: float = -5.0,          # Штраф за поражение кампании
        reward_timeout: float = -3.0,       # Штраф за таймаут (лимит шагов)
        reward_turn_penalty: float = 0.02,  # Штраф за завершение хода (REST)
        reward_repeat_position_penalty: float = 0.015,  # Штраф за повторное посещение клетки.
        reward_repeat_position_penalty_cap: float = 0.08,  # Кеп на повторный штраф за 1 шаг.
        reward_backtrack_penalty: float = 0.03,  # Штраф за микро-цикл A->B->A.
        reward_no_movement_penalty: float = 0.02,  # Штраф если движение не сменило позицию.
        exp_reward_norm_k: float = 100.0,   # Нормализация опыта: norm=exp/(exp+k)
        reward_exp_weight: float = 0.2,     # Вес exp-компоненты
        reward_survival_alive_weight: float = 0.5,  # Вес доли выживших BLUE
        reward_survival_hp_weight: float = 0.5,     # Вес средней доли HP BLUE
        reward_unit_upgrade: float = 1.0,
        persist_blue_hp: bool = True,
        log_enabled: bool = False,
        max_grid_steps: int = 1800,
        realcapital: int = 1,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.reward_engage_battle = reward_engage_battle  # Награда за участие в битве
        self.reward_defeat_enemy = reward_defeat_enemy    # Награда за победу в битве
        self.reward_all_enemies = reward_all_enemies      # База для финального бонуса за зелёного дракона
        self.reward_loss = reward_loss
        self.reward_timeout = reward_timeout
        self.reward_turn_penalty = reward_turn_penalty
        self.reward_repeat_position_penalty = max(0.0, float(reward_repeat_position_penalty))
        self.reward_repeat_position_penalty_cap = max(0.0, float(reward_repeat_position_penalty_cap))
        self.reward_backtrack_penalty = max(0.0, float(reward_backtrack_penalty))
        self.reward_no_movement_penalty = max(0.0, float(reward_no_movement_penalty))
        self.exp_reward_norm_k = max(1.0, float(exp_reward_norm_k))
        self.reward_exp_weight = max(0.0, float(reward_exp_weight))
        self.reward_survival_alive_weight = max(0.0, float(reward_survival_alive_weight))
        self.reward_survival_hp_weight = max(0.0, float(reward_survival_hp_weight))
        self.reward_unit_upgrade = max(0.0, float(reward_unit_upgrade))
        self.persist_blue_hp = persist_blue_hp
        self.log_enabled = log_enabled
        self.max_grid_steps = max_grid_steps
        # Базовый лимит шагов внутри одного хода кампании.
        self.moves_per_turn = int(self.MOVES_PER_TURN)
        self.turn_norm_k = max(
            1.0,
            float(self.max_grid_steps) / float(self.GRID_STEPS_PER_TURN),
        )
        self.gold_norm_k = max(1.0, self.turn_norm_k * float(self.GOLD_PER_TURN))
        # Realcapital: 1 = empire, 2 = legions, 3 = mountain_clans, 4 = undead_hordes, 5 = elves
        self.Realcapital = self._normalize_realcapital(realcapital)

        enemy_positions = scale_enemy_positions(self.grid_size)
        self.castle_heal_tiles: Tuple[Tuple[int, int], ...] = resolve_heal_tiles(
            self.grid_size,
            enemy_positions,
            start_position=self.CASTLE_POS,
            heal_tile_count=self.CASTLE_HEAL_TILE_COUNT,
        )
        obstacle_tiles = resolve_obstacle_tiles(
            self.grid_size,
            enemy_positions,
            heal_tiles=self.castle_heal_tiles,
            start_position=self.CASTLE_POS,
        )
        self._static_enemy_positions = dict(enemy_positions)
        self._static_castle_heal_tiles: Tuple[Tuple[int, int], ...] = tuple(self.castle_heal_tiles)
        self._static_obstacle_tiles: Tuple[Tuple[int, int], ...] = tuple(obstacle_tiles)

        # Подсреда grid: враги и препятствия автоматически масштабируются под размер карты.
        self.grid_env = GridWorldEnv(
            grid_size=grid_size,
            enemy_positions=self._static_enemy_positions,
            obstacle_positions=self._static_obstacle_tiles,
            start_position=self.CASTLE_POS,
            max_steps=max_grid_steps,
        )
        self._ensure_map_layout_consistency()
        self.battle_env: Optional[BattleEnv] = None
        self.grid_obs_base_size = int(self.grid_env.observation_space.shape[0])
        self._init_enemy_obs_metadata()
        self.GRID_OBS_SIZE = self.grid_obs_base_size + self.grid_enemy_obs_size

        # Текущий режим
        self.mode = self.MODE_GRID
        self.current_enemy_id: Optional[int] = None
        self.battle_origin_pos: Optional[Tuple[int, int]] = None

        # Сохраняем состояние BLUE команды между боями
        self.blue_team_state: Optional[List[Dict]] = None
        self.heal_bottles_used: int = 0
        self.healing_bottles_used: int = 0
        self.revive_bottles_used: int = 0
        self.turns: int = 0
        self.gold: float = 0.0
        self.moves: int = self.moves_per_turn
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
        self.grid_visit_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._reset_buildings_state()
        self.active_buildings = self._get_buildings_for_capital(self.Realcapital)
        self.building_keys = self._get_building_keys(self.active_buildings)
        self.mode = self.MODE_GRID
        self.current_enemy_id = None
        self.battle_origin_pos = None
        # Стартовое состояние BLUE — копия базовых юнитов, чтобы
        # вне боя можно было применять лечение до первого боя.
        self.blue_team_state = deepcopy(UNITS_BLUE)
        self.heal_bottles_used = 0
        self.healing_bottles_used = 0
        self.revive_bottles_used = 0
        self.turns = 0
        self.gold = 0.0
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state, refill=True)
        self.battle_env = None
        self._campaign_logs = []
        self.castle_heal_tiles = self._static_castle_heal_tiles
        self.grid_env.enemy_positions = dict(self._static_enemy_positions)
        self.grid_env.obstacle_positions = set(self._static_obstacle_tiles)

        grid_obs, grid_info = self.grid_env.reset(seed=seed)
        grid_obs = self._augment_grid_obs(grid_obs)
        self._reset_stagnation_tracking(self.grid_env.agent_pos)

        self._log("=== НОВАЯ КАМПАНИЯ ===")
        self._log(f"Старт на позиции {self.grid_env.agent_pos}")
        self._log(f"Клетки лечения на карте: {self.castle_heal_tiles}")
        self._log(f"Препятствия на карте: {sorted(self.grid_env.obstacle_positions)}")
        self._log(f"Враги на карте: {list(self.grid_env.enemy_positions.items())}")

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
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
        #   9-14 — применить подходящую банку лечения к позициям 7-12 соответственно.
        #   15-20 — применить бутыль воскрешения к позициям 7-12 соответственно.
        #   21-26 — лечение на специальных клетках карты по позициям 7-12.
        #   27-32 — воскрешение на специальных клетках карты по позициям 7-12.
        #   33-57 ? ????????? ?????? ????????? ??????? (25 ????????).
        if action >= self.GRID_BUILD_ACTION_START:
            return self._step_building(action)
        if action >= self.GRID_CASTLE_REVIVE_ACTION_START:
            return self._step_revive_in_castle(action)
        if action >= self.GRID_CASTLE_HEAL_ACTION_START:
            return self._step_heal_in_castle(action)
        if action >= self.GRID_REVIVE_ACTION_START:
            return self._step_revive_with_bottle(action)
        if action >= self.GRID_BOTTLE_ACTION_START:
            return self._step_heal_with_bottle(action)

        if 0 <= action <= 7 and self.moves <= 0:
            grid_obs = self._get_grid_obs()
            info = {
                "mode": "grid",
                "agent_pos": self.grid_env.agent_pos,
                "enemies_alive": dict(self.grid_env.enemies_alive),
                "battle_triggered": False,
                "blocked_by_moves": True,
                "turns": self.turns,
                "gold": self.gold,
                "moves": self.moves,
            }
            return self._build_obs(grid_obs=grid_obs), 0.0, False, False, info

        grid_action = min(action, 8)

        old_pos = self.grid_env.agent_pos
        grid_obs, _grid_reward, terminated, truncated, grid_info = self.grid_env.step(
            grid_action
        )
        # Add dense navigation signal from GridWorldEnv to reduce timeout-only learning.
        reward = 0.2 * float(_grid_reward)
        grid_obs = self._augment_grid_obs(grid_obs)
        new_pos = self.grid_env.agent_pos
        stagnation_penalty = 0.0
        battle_engage_bonus = 0.0

        if old_pos != new_pos:
            self._log(f"Перемещение: {old_pos} -> {new_pos}")
        elif grid_info.get("blocked_by_obstacle"):
            self._log(
                f"Препятствие: переход {old_pos} -> {grid_info.get('blocked_target')} заблокирован"
            )
        elif grid_info.get("blocked_by_boundary"):
            self._log(
                f"Граница карты: переход {old_pos} -> {grid_info.get('blocked_target')} заблокирован"
            )

        if 0 <= grid_action <= 7:
            if not grid_info.get("blocked_by_obstacle", False):
                stagnation_penalty = self._compute_stagnation_penalty(old_pos, new_pos)
            reward += stagnation_penalty
            self._spend_moves(1)

        if grid_info.get("battle_triggered"):
            enemy_id = grid_info.get("enemy_id")
            if enemy_id is not None and self.grid_env.enemies_alive.get(enemy_id, False):
                battle_engage_bonus = float(self.reward_engage_battle)
                reward += battle_engage_bonus
            self._apply_battle_grid_steps(extra_steps=10)

        info = {
            "mode": "grid",
            "agent_pos": new_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": grid_info.get("battle_triggered", False),
            "rest_action": grid_info.get("rest_action", False),
            "blocked_by_obstacle": grid_info.get("blocked_by_obstacle", False),
            "blocked_by_boundary": grid_info.get("blocked_by_boundary", False),
            "blocked_obstacle_pos": grid_info.get("blocked_target"),
            "grid_reward_raw": float(_grid_reward),
            "grid_reward_scaled": float(reward),
            "stagnation_penalty": float(stagnation_penalty),
            "battle_engage_bonus": float(battle_engage_bonus),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
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
            info["moves"] = self.moves
            info["healed_units"] = healed

        # Проверяем столкновение с врагом
        if grid_info.get("battle_triggered"):
            enemy_id = grid_info["enemy_id"]
            self.current_enemy_id = enemy_id
            self.battle_origin_pos = (int(old_pos[0]), int(old_pos[1]))
            self.mode = self.MODE_BATTLE

            self._log(f"!!! СТОЛКНОВЕНИЕ С ВРАГОМ {enemy_id} !!!")
            self._log(f"Описание: {ENEMY_DESCRIPTIONS.get(enemy_id, 'Неизвестный враг')}")

            # Создаём BattleEnv с нужной RED командой
            self._init_battle(enemy_id)

            # Возвращаем battle observation
            battle_obs = self.battle_env._obs()
            info["mode"] = "battle"
            info["enemy_id"] = enemy_id

            return self._build_obs(battle_obs=battle_obs), reward, False, False, info

        # Проверяем победу (все враги побеждены)
        if self.grid_env.all_enemies_defeated():
            self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
            info["campaign_result"] = "victory"
            info["campaign_victory_reason"] = "all_enemies_defeated"
            return self._build_obs(grid_obs=grid_obs), reward, True, False, info

        # Проверяем лимит шагов на карте
        if truncated:
            self._log("=== ЛИМИТ ШАГОВ НА КАРТЕ ИСЧЕРПАН ===")
            reward += self.reward_timeout
            info["campaign_result"] = "timeout"

        return self._build_obs(grid_obs=grid_obs), reward, terminated, truncated, info

    def _is_at_castle(self) -> bool:
        """True если агент стоит на одной из клеток лечения в grid-режиме."""
        return tuple(self.grid_env.agent_pos) in self.castle_heal_tiles

    @staticmethod
    def _castle_heal_gold_per_hp(unit: Dict) -> float:
        """Стоимость лечения на клетке лечения: 1/2/3 gold за 1 HP в зависимости от Level."""
        level_raw = unit.get("Level", 1)
        try:
            level = int(round(float(level_raw)))
        except (TypeError, ValueError):
            level = 1

        if level in (2, 3):
            return 2.0
        if level >= 4:
            return 3.0
        return 1.0

    @staticmethod
    def _castle_revive_gold_cost(unit: Dict) -> float:
        """Стоимость воскрешения на клетке лечения в зависимости от Level."""
        level_raw = unit.get("Level", 1)
        try:
            level = int(round(float(level_raw)))
        except (TypeError, ValueError):
            level = 1

        if level <= 1:
            return 50.0
        if level == 2:
            return 200.0
        if level == 3:
            return 400.0
        if level == 4:
            return 600.0
        return 800.0

    @staticmethod
    def _is_hero_unit(unit: Dict) -> bool:
        try:
            return int(round(float(unit.get("next_level_exp", 0) or 0))) > 0
        except (TypeError, ValueError):
            return False

    def _resolve_travel_hero(self, units: Optional[List[Dict]] = None) -> Optional[Dict]:
        roster = units if units is not None else self._get_blue_state()
        heroes = [unit for unit in roster if self._is_hero_unit(unit)]
        if not heroes:
            return None
        for unit in heroes:
            try:
                if int(unit.get("position", -1)) == 8:
                    return unit
            except (TypeError, ValueError):
                continue
        return min(heroes, key=lambda unit: int(unit.get("position", 999) or 999))

    def _hero_level(self, units: Optional[List[Dict]] = None) -> int:
        hero = self._resolve_travel_hero(units=units)
        if hero is None:
            return 0
        try:
            return max(0, int(round(float(hero.get("Level", 0) or 0))))
        except (TypeError, ValueError):
            return 0

    def _hero_has_pathfinding(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_PATHFINDING_LEVEL)

    def _hero_pathfinding_move_bonus(self, units: Optional[List[Dict]] = None) -> int:
        if not self._hero_has_pathfinding(units=units):
            return 0
        return max(
            0,
            int(round(float(self.MOVES_PER_TURN) * float(self.HERO_PATHFINDING_MOVE_BONUS_PCT))),
        )

    def _hero_move_bonus(self, units: Optional[List[Dict]] = None) -> int:
        return self._hero_level(units=units) + self._hero_pathfinding_move_bonus(units=units)

    def get_travel_hero_visual_info(self) -> Optional[Dict]:
        hero = self._resolve_travel_hero()
        if hero is None:
            return None

        abilities: List[str] = []
        if self._hero_has_pathfinding(units=[hero]):
            abilities.append("Нахождение пути (+20% шагов)")

        return {
            "name": str(hero.get("name", "") or "").strip(),
            "level": self._hero_level(units=[hero]),
            "abilities": abilities,
        }

    def _sync_moves_per_turn_with_hero(
        self,
        units: Optional[List[Dict]] = None,
        *,
        refill: bool = False,
        grant_delta: bool = False,
    ) -> None:
        previous_cap = int(self.moves_per_turn)
        new_cap = int(self.MOVES_PER_TURN) + self._hero_move_bonus(units=units)
        self.moves_per_turn = new_cap
        if refill:
            self.moves = new_cap
            return
        if grant_delta and new_cap > previous_cap:
            self.moves = min(new_cap, int(self.moves) + (new_cap - previous_cap))
        elif int(self.moves) > new_cap:
            self.moves = new_cap

    def _heal_unit_to_full_at_position(self, position: int) -> Tuple[float, Optional[str]]:
        """Лечит одного юнита BLUE на позиции в пределах золота с тарифом по Level."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)

            if max_hp <= 0 or hp <= 0 or hp >= max_hp:
                return 0.0, None

            missing_hp = max_hp - hp
            gold_available = max(0.0, float(self.gold or 0.0))
            if gold_available <= 0.0:
                return 0.0, None

            gold_per_hp = self._castle_heal_gold_per_hp(unit)
            if gold_per_hp <= 0.0:
                return 0.0, None

            healed = min(missing_hp, gold_available / gold_per_hp)
            gold_spent = healed * gold_per_hp
            new_hp = hp + healed
            unit["hp"] = new_hp
            unit["health"] = new_hp
            self.gold = max(0.0, gold_available - gold_spent)

            try:
                level_for_log = int(round(float(unit.get("Level", 1) or 1)))
            except (TypeError, ValueError):
                level_for_log = 1

            name = unit.get("name", f"pos_{position}")
            self._log(
                f"Heal tile: {name} (pos {position}) HP {hp:.1f} -> {new_hp:.1f} "
                f"(+{healed:.1f}, level={level_for_log}, "
                f"cost={gold_spent:.1f} gold, gold_left={self.gold:.1f})"
            )
            return healed, name

        return 0.0, None

    def _step_heal_in_castle(self, action: int):
        """Лечит выбранного юнита BLUE на клетках лечения с ценой за HP по Level."""
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
        gold_spent = 0.0
        gold_before = float(self.gold)
        if self._is_at_castle() and target_pos is not None:
            healed_amount, unit_name = self._heal_unit_to_full_at_position(position=target_pos)
            gold_spent = max(0.0, gold_before - float(self.gold))

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "castle_heal_action": True,
            "castle_pos_required": self.CASTLE_POS,
            "castle_heal_tiles": self.castle_heal_tiles,
            "castle_heal_available": self._is_at_castle(),
            "castle_heal_target_pos": target_pos,
            "healed_amount": healed_amount,
            "healed_unit_name": unit_name,
            "gold_spent": gold_spent,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._build_obs(grid_obs=grid_obs), reward, terminated, truncated, info

    def _ensure_map_layout_consistency(self) -> None:
        """Fail fast if generated map features overlap on forbidden tiles."""
        heal_tiles = {tuple(tile) for tile in self.castle_heal_tiles}
        enemy_tiles = {tuple(pos) for pos in self.grid_env.enemy_positions.values()}
        obstacle_tiles = {tuple(pos) for pos in self.grid_env.obstacle_positions}

        heal_enemy_overlap = heal_tiles & enemy_tiles
        if heal_enemy_overlap:
            raise ValueError(f"Heal tiles overlap enemy tiles: {sorted(heal_enemy_overlap)}")

        obstacle_enemy_overlap = obstacle_tiles & enemy_tiles
        if obstacle_enemy_overlap:
            raise ValueError(
                f"Obstacle tiles overlap enemy tiles: {sorted(obstacle_enemy_overlap)}"
            )

        obstacle_heal_overlap = obstacle_tiles & heal_tiles
        if obstacle_heal_overlap:
            raise ValueError(
                f"Obstacle tiles overlap heal tiles: {sorted(obstacle_heal_overlap)}"
            )

        reachable_targets = set(enemy_tiles)
        reachable_targets.update(heal_tiles)
        if not are_targets_reachable(
            self.grid_size,
            self.CASTLE_POS,
            obstacle_tiles,
            reachable_targets,
        ):
            raise ValueError("Obstacle tiles block path to at least one enemy or heal tile")

    def _castle_revive_cost_at_position(self, position: int) -> float:
        """Возвращает стоимость воскрешения мёртвого юнита на позиции или 0.0."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp > 0:
                return 0.0
            return float(self._castle_revive_gold_cost(unit))
        return 0.0

    def _revive_unit_with_gold_at_position(
        self,
        position: int,
    ) -> Tuple[bool, Optional[str], float]:
        """Воскрешает юнита за золото на клетке лечения, если хватает средств."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp > 0:
                return False, None, 0.0

            revive_cost = float(self._castle_revive_gold_cost(unit))
            gold_available = max(0.0, float(self.gold or 0.0))
            if revive_cost <= 0.0 or gold_available < revive_cost:
                return False, None, 0.0

            unit["hp"] = 1.0
            unit["health"] = 1.0
            self.gold = max(0.0, gold_available - revive_cost)
            name = unit.get("name", f"pos_{position}")
            self._log(
                f"Heal tile revive: {name} (pos {position}) revived with 1 HP "
                f"(cost={revive_cost:.1f} gold, gold_left={self.gold:.1f})"
            )
            return True, name, revive_cost

        return False, None, 0.0

    def _step_revive_in_castle(self, action: int):
        """Воскрешает выбранного юнита BLUE за золото на клетках лечения."""
        idx = action - self.GRID_CASTLE_REVIVE_ACTION_START
        target_pos = (
            self.CASTLE_REVIVE_POSITIONS[idx]
            if 0 <= idx < len(self.CASTLE_REVIVE_POSITIONS)
            else None
        )

        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        revive_cost = 0.0
        target_unit_name = None
        if target_pos is not None:
            revive_cost = self._castle_revive_cost_at_position(target_pos)
            for unit in self._get_blue_state():
                if int(unit.get("position", -1)) == target_pos:
                    target_unit_name = unit.get("name", f"pos_{target_pos}")
                    break

        revived, revived_unit_name, gold_spent = (False, None, 0.0)
        if self._is_at_castle() and target_pos is not None:
            revived, revived_unit_name, gold_spent = self._revive_unit_with_gold_at_position(
                position=target_pos
            )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "castle_revive_action": True,
            "castle_heal_tiles": self.castle_heal_tiles,
            "castle_revive_available": (
                self._is_at_castle()
                and target_pos is not None
                and self._can_castle_revive_position(target_pos)
            ),
            "castle_revive_target_pos": target_pos,
            "target_unit_name": target_unit_name,
            "revived": revived,
            "revived_unit_name": revived_unit_name,
            "revive_cost": revive_cost,
            "gold_spent": gold_spent,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
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
            "moves": self.moves,
        }

        return self._build_obs(grid_obs=grid_obs), reward, terminated, truncated, info

    def _step_heal_with_bottle(self, action: int):
        """Применяет подходящую банку лечения к указанной позиции BLUE в режиме grid."""
        idx = action - self.GRID_BOTTLE_ACTION_START
        target_pos = self.GRID_BOTTLE_POSITIONS[idx] if 0 <= idx < len(self.GRID_BOTTLE_POSITIONS) else None

        # Лечение не продвигает ход карты и не даёт штрафа/таймера.
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        healed_amount, unit_name = (0.0, None)
        selected_heal_amount = 0.0
        selected_bottle_kind = None
        if target_pos is not None:
            selected_heal_amount, selected_bottle_kind = self._select_heal_bottle_for_position(
                target_pos
            )
        if target_pos is not None and selected_bottle_kind is not None:
            healed_amount, unit_name = self._heal_unit_at_position(
                position=target_pos,
                heal_amount=selected_heal_amount,
                source_label=(
                    "Банка исцеления"
                    if selected_bottle_kind == "small"
                    else "Бутыль лечения"
                ),
            )
            if healed_amount > 0.0:
                if selected_bottle_kind == "large":
                    self.heal_bottles_used = min(self.heal_bottles_used + 1, self.MAX_HEAL_BOTTLES)
                else:
                    self.healing_bottles_used = min(
                        self.healing_bottles_used + 1,
                        self.MAX_HEALING_BOTTLES,
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
            "selected_heal_bottle_kind": selected_bottle_kind,
            "selected_heal_bottle_amount": selected_heal_amount,
            "heal_bottles_used": self.heal_bottles_used,
            "heal_bottles_left": max(0, self.MAX_HEAL_BOTTLES - self.heal_bottles_used),
            "healing_bottles_used": self.healing_bottles_used,
            "healing_bottles_left": max(
                0, self.MAX_HEALING_BOTTLES - self.healing_bottles_used
            ),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
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
            "moves": self.moves,
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
            obs, _battle_reward, terminated, truncated, battle_info = self.battle_env.step(action)
            reward = 0.0

        info = {
            "mode": "battle",
            "enemy_id": self.current_enemy_id,
            "battle_step": True,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        if terminated or truncated:
            winner = self.battle_env.winner
            info["battle_winner"] = winner

            if winner == "blue":
                exp_reward, exp_raw = self._compute_blue_exp_reward()
                survival_reward, blue_alive_ratio, blue_hp_ratio = self._compute_blue_survival_reward()
                enemy_reward = self._compute_enemy_defeat_reward(self.current_enemy_id)
                reward += enemy_reward + exp_reward + survival_reward
                info["enemy_defeat_reward"] = enemy_reward
                info["blue_exp_reward"] = exp_reward
                info["blue_exp_raw"] = exp_raw
                info["blue_survival_reward"] = survival_reward
                info["blue_alive_ratio"] = blue_alive_ratio
                info["blue_hp_ratio"] = blue_hp_ratio

                # Победа: помечаем врага побеждённым
                self._log(f"=== ПОБЕДА В БОЮ ПРОТИВ ВРАГА {self.current_enemy_id}! ===")
                self.grid_env.mark_enemy_defeated(self.current_enemy_id)

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

                # Возвращаемся в grid режим
                self.mode = self.MODE_GRID
                self.battle_env = None
                info["battle_result"] = "victory"
                info["mode"] = "grid"
                info["agent_pos"] = self._restore_agent_to_battle_origin()
                info["enemies_alive"] = dict(self.grid_env.enemies_alive)

                if self._is_green_dragon_enemy(self.current_enemy_id):
                    green_dragon_reward = self._compute_green_dragon_victory_reward(
                        self.current_enemy_id
                    )
                    reward += green_dragon_reward
                    self._log("=== ПОБЕЖДЁН ЗЕЛЁНЫЙ ДРАКОН: ДОСРОЧНАЯ ПОБЕДА В КАМПАНИИ ===")
                    grid_obs = self._get_grid_obs()
                    info["campaign_result"] = "victory"
                    info["campaign_victory_reason"] = "green_dragon_defeated"
                    info["green_dragon_reward"] = float(green_dragon_reward)
                    return self._build_obs(grid_obs=grid_obs), reward, True, False, info

                # Проверяем полную победу
                if self.grid_env.all_enemies_defeated():
                    self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
                    grid_obs = self._get_grid_obs()
                    info["campaign_result"] = "victory"
                    info["campaign_victory_reason"] = "all_enemies_defeated"
                    return self._build_obs(grid_obs=grid_obs), reward, True, False, info

                grid_obs = self._get_grid_obs()
                return self._build_obs(grid_obs=grid_obs), reward, False, False, info

            else:
                # Поражение: конец эпизода
                reward += self.reward_loss
                self._clear_battle_origin()
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

        self._sync_moves_per_turn_with_hero(units=self.blue_team_state, grant_delta=True)
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
                        unit.clear()
                        unit.update(deepcopy(upgraded_unit))
                        self._replace_blue_unit(upgraded_unit)
                        upgraded_count += 1
                    break

        return int(upgraded_count)

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

    def _has_any_heal_bottle_left(self) -> bool:
        """True, если доступна хотя бы одна банка лечения любого типа."""
        return (
            self.heal_bottles_used < self.MAX_HEAL_BOTTLES
            or self.healing_bottles_used < self.MAX_HEALING_BOTTLES
        )

    def _select_heal_bottle_for_position(self, position: int) -> Tuple[float, Optional[str]]:
        """
        Выбирает, какую банку применить к юниту:
        - при наличии обеих: +50 HP, если этого достаточно для полного лечения, иначе +100 HP;
        - если один тип закончился: использует другой;
        - если лечить нельзя: возвращает (0.0, None).
        """
        missing_hp = self._missing_hp_at_position(position)
        if missing_hp <= 0.0:
            return 0.0, None

        has_large = self.heal_bottles_used < self.MAX_HEAL_BOTTLES
        has_small = self.healing_bottles_used < self.MAX_HEALING_BOTTLES

        if has_large and has_small:
            if missing_hp <= self.HEALING_BOTTLE_AMOUNT:
                return self.HEALING_BOTTLE_AMOUNT, "small"
            return self.HEAL_BOTTLE_AMOUNT, "large"
        if has_large:
            return self.HEAL_BOTTLE_AMOUNT, "large"
        if has_small:
            return self.HEALING_BOTTLE_AMOUNT, "small"
        return 0.0, None

    def _get_blue_state(self) -> List[Dict]:
        """Возвращает (и при необходимости инициализирует) сохранённое состояние BLUE."""
        if self.blue_team_state is None:
            self.blue_team_state = deepcopy(UNITS_BLUE)
            self._sync_moves_per_turn_with_hero(units=self.blue_team_state, refill=True)
        return self.blue_team_state

    def _spend_moves(self, spent_moves: int) -> None:
        """Списывает очки перемещения в grid-режиме."""
        if spent_moves <= 0:
            return
        self.moves = max(0, int(self.moves) - int(spent_moves))

    def _reset_stagnation_tracking(self, start_pos: Tuple[int, int]) -> None:
        """Сбрасывает трекинг повторов и микро-циклов в grid-режиме."""
        pos = (int(start_pos[0]), int(start_pos[1]))
        self.grid_visit_counts = {pos: 1}
        self.recent_positions = [pos]

    def _compute_stagnation_penalty(
        self,
        old_pos: Tuple[int, int],
        new_pos: Tuple[int, int],
    ) -> float:
        """
        Штрафует стагнацию на карте:
        - повторные посещения клетки (с кепом на шаг),
        - движение "в стену" без смены позиции,
        - петля A->B->A.
        Возвращает отрицательное значение (или 0.0).
        """
        target_pos = (int(new_pos[0]), int(new_pos[1]))
        visit_count = int(self.grid_visit_counts.get(target_pos, 0)) + 1
        self.grid_visit_counts[target_pos] = visit_count

        self.recent_positions.append(target_pos)
        if len(self.recent_positions) > 3:
            self.recent_positions = self.recent_positions[-3:]

        penalty = 0.0
        if visit_count > 1 and self.reward_repeat_position_penalty > 0.0:
            repeat_penalty = self.reward_repeat_position_penalty * float(visit_count - 1)
            penalty += min(repeat_penalty, self.reward_repeat_position_penalty_cap)

        if old_pos == new_pos and self.reward_no_movement_penalty > 0.0:
            penalty += self.reward_no_movement_penalty

        if len(self.recent_positions) >= 3 and self.reward_backtrack_penalty > 0.0:
            pos_a, pos_b, pos_c = self.recent_positions[-3:]
            if pos_a == pos_c and pos_a != pos_b:
                penalty += self.reward_backtrack_penalty

        return -float(penalty)

    def _apply_battle_grid_steps(self, extra_steps: int = 10) -> None:
        """Списывает дополнительные очки перемещения из-за входа в бой."""
        if extra_steps <= 0:
            return
        self._spend_moves(extra_steps)

    def _update_turns(self) -> float:
        """Совместимость: ходы больше не рассчитываются по step_count."""
        # Ходы теперь продвигаются только явным завершением хода (REST).
        return 0.0

    def _compute_enemy_defeat_reward(self, enemy_id: Optional[int]) -> float:
        """Фиксированная награда за победу над каждым вражеским отрядом."""
        return float(self.reward_defeat_enemy)

    def _is_green_dragon_enemy(self, enemy_id: Optional[int]) -> bool:
        """True, если enemy_id соответствует отряду с зелёным драконом."""
        try:
            return int(enemy_id) == int(self.GREEN_DRAGON_ENEMY_ID)
        except (TypeError, ValueError):
            return False

    def _compute_green_dragon_victory_reward(self, enemy_id: Optional[int]) -> float:
        """Финальный бонус за победу над зелёным драконом (x10 от базовой финальной награды)."""
        if not self._is_green_dragon_enemy(enemy_id):
            return 0.0
        return float(self.reward_all_enemies) * float(self.GREEN_DRAGON_REWARD_MULTIPLIER)

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

        base_units = [u for u in UNITS_BLUE if int(u.get("position", -1)) >= 0]
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

    def _advance_turns(self, delta_turns: int) -> float:
        """Увеличивает счётчик ходов и золото на заданное количество."""
        if delta_turns <= 0:
            return 0.0
        self.turns += int(delta_turns)
        self.gold += float(delta_turns) * self.GOLD_PER_TURN
        self.moves = self.moves_per_turn
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
            # В grid режиме движение доступно только при moves > 0.
            # REST (8) доступен при ранении или когда очки перемещения закончились.
            grid_mask = self.grid_env.compute_action_mask()
            mask[:8] = bool(self.moves > 0) & grid_mask[:8]
            mask[8] = bool(grid_mask[8]) and (self._has_wounded_blue() or self.moves <= 0)
            # Общее действие лечения банками на позиции 7-12.
            if self._has_any_heal_bottle_left():
                for idx, pos in enumerate(self.GRID_BOTTLE_POSITIONS):
                    mask[self.GRID_BOTTLE_ACTION_START + idx] = self._can_heal_position(pos)
            # Бутыли воскрешения на позиции 7-12.
            if self.revive_bottles_used < self.MAX_REVIVE_BOTTLES:
                for idx, pos in enumerate(self.REVIVE_BOTTLE_POSITIONS):
                    mask[self.GRID_REVIVE_ACTION_START + idx] = self._can_revive_position(pos)

            # Лечение и воскрешение за золото доступны только на специальных клетках лечения.
            if self._is_at_castle() and float(self.gold) > 0.0:
                for idx, pos in enumerate(self.CASTLE_HEAL_POSITIONS):
                    mask[self.GRID_CASTLE_HEAL_ACTION_START + idx] = self._can_heal_position(pos)
                for idx, pos in enumerate(self.CASTLE_REVIVE_POSITIONS):
                    mask[self.GRID_CASTLE_REVIVE_ACTION_START + idx] = self._can_castle_revive_position(pos)

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
            lines.append(f"Ход: {self.turns} | Золото: {self.gold:g} | Шаги: {self.moves}")
        else:
            lines.append(f"В бою с врагом {self.current_enemy_id}")
            lines.append(f"Описание: {ENEMY_DESCRIPTIONS.get(self.current_enemy_id, '?')}")
            lines.append(f"Ход: {self.turns} | Золото: {self.gold:g} | Шаги: {self.moves}")

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
            "moves": self.moves,
        }
