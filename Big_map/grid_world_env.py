# grid_world_env.py
"""
Среда перемещения по сетке 40×40 (по умолчанию).
Агент может двигаться в 8 направлениях, стоять на месте или отдыхать.
На определённых клетках расположены враги (лагеря RED).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, Set


class GridWorldEnv(gym.Env):
    """
    Среда перемещения по сетке.

    Actions (9):
        0: UP          (y-1)
        1: DOWN        (y+1)
        2: LEFT        (x-1)
        3: RIGHT       (x+1)
        4: UP-LEFT     (y-1, x-1)
        5: UP-RIGHT    (y-1, x+1)
        6: DOWN-LEFT   (y+1, x-1)
        7: DOWN-RIGHT  (y+1, x+1)
        8: REST        (завершить ход — восстановление 5% HP)

    Observation:
        - Позиция агента (x, y) нормализованная [0, 1]
        - Флаги живых врагов
        - Расстояния до каждого врага (нормализованные)
        - Доля исследованных клеток
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Константы действий
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_UP_LEFT = 4
    ACTION_UP_RIGHT = 5
    ACTION_DOWN_LEFT = 6
    ACTION_DOWN_RIGHT = 7
    ACTION_REST = 8  # Завершить ход (восстановление HP)
    # Базовая раскладка лагерей RED на исторической карте 20x20.
    # При смене размера карты координаты масштабируются пропорционально.
    BASE_GRID_SIZE = 20
    BASE_ENEMY_POSITIONS = {
        1: (4, 2),
        2: (11, 2),
        3: (17, 2),
        4: (4, 6),
        5: (11, 6),
        6: (17, 6),
        7: (4, 11),
        8: (11, 11),
        9: (17, 11),
        10: (6, 15),
        11: (15, 15),
        12: (2, 1),
        13: (8, 1),
        14: (14, 1),
        15: (2, 4),
        16: (8, 4),
        17: (14, 4),
        18: (2, 8),
        19: (8, 8),
        20: (14, 8),
        21: (2, 13),
        22: (8, 13),
        23: (14, 13),
        24: (2, 17),
        25: (8, 17),
        26: (14, 17),
        27: (18, 4),
        28: (18, 8),
        29: (18, 13),
        30: (10, 17),
    }

    def __init__(
        self,
        grid_size: int = 40,
        enemy_positions: Optional[Dict[int, Tuple[int, int]]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        step_penalty: float = -0.01,
        exploration_bonus: float = 0.005,
        max_steps: int = 1000,
    ):
        super().__init__()

        self.grid_size = int(grid_size)
        if self.grid_size < 2:
            raise ValueError("grid_size must be >= 2")

        if start_position is None:
            self.start_position = (0, self.grid_size - 1)  # Левый нижний угол
        else:
            self.start_position = self._clamp_to_grid(start_position)

        self.step_penalty = step_penalty
        self.exploration_bonus = exploration_bonus
        self.max_steps = max_steps

        # Позиции врагов на карте (enemy_id -> (x, y)).
        # Если позиции не заданы, масштабируем "базовую" раскладку 20x20
        # на текущий размер карты.
        if enemy_positions is None:
            self.enemy_positions = self._scaled_default_enemy_positions(self.grid_size)
        else:
            self.enemy_positions = {
                int(enemy_id): self._clamp_to_grid(pos)
                for enemy_id, pos in enemy_positions.items()
            }
        self.num_enemies = len(self.enemy_positions)

        # 9 действий: 8 направлений + отдых
        self.action_space = spaces.Discrete(9)

        # Observation: [x, y, enemy_alive..., dist_to_enemy..., visited_ratio]
        obs_len = 2 + self.num_enemies + self.num_enemies + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

        # Состояние
        self.agent_pos: Tuple[int, int] = self.start_position
        self.enemies_alive: Dict[int, bool] = {eid: True for eid in self.enemy_positions.keys()}
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.current_enemy_encounter: Optional[int] = None
        self.step_count: int = 0

    def _clamp_to_grid(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Ограничивает координаты рамками карты."""
        x = int(pos[0])
        y = int(pos[1])
        return (
            max(0, min(self.grid_size - 1, x)),
            max(0, min(self.grid_size - 1, y)),
        )

    @classmethod
    def _scaled_default_enemy_positions(cls, target_grid_size: int) -> Dict[int, Tuple[int, int]]:
        """Масштабирует базовые позиции врагов 20x20 на произвольный размер карты."""
        if target_grid_size <= 1:
            raise ValueError("target_grid_size must be >= 2")
        if target_grid_size == cls.BASE_GRID_SIZE:
            return dict(cls.BASE_ENEMY_POSITIONS)

        base_span = cls.BASE_GRID_SIZE - 1
        target_span = target_grid_size - 1
        scale = float(target_span) / float(base_span)
        scaled_positions: Dict[int, Tuple[int, int]] = {}
        for enemy_id, (x, y) in cls.BASE_ENEMY_POSITIONS.items():
            sx = int(round(x * scale))
            sy = int(round(y * scale))
            scaled_positions[enemy_id] = (
                max(0, min(target_span, sx)),
                max(0, min(target_span, sy)),
            )
        return scaled_positions

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.agent_pos = self.start_position
        # Динамически инициализируем на основе enemy_positions
        self.enemies_alive = {eid: True for eid in self.enemy_positions.keys()}
        self.visited_cells = {self.start_position}
        self.current_enemy_encounter = None
        self.step_count = 0

        return self._get_obs(), {"agent_pos": self.agent_pos}

    def step(self, action: int):
        self.step_count += 1

        # Приводим action к int (может прийти как numpy.ndarray)
        action = int(action)

        reward = self.step_penalty
        terminated = False
        truncated = False
        info = {
            "battle_triggered": False,
            "rest_action": False,
            "enemy_id": None,
            "agent_pos": self.agent_pos,
            "step": self.step_count,
        }

        # Проверка действия REST (завершить ход — восстановление HP)
        if action == self.ACTION_REST:
            info["rest_action"] = True
            # Не двигаемся, просто отдыхаем
        else:
            # Движение
            dx, dy = self._action_to_delta(action)
            new_x = max(0, min(self.grid_size - 1, self.agent_pos[0] + dx))
            new_y = max(0, min(self.grid_size - 1, self.agent_pos[1] + dy))
            self.agent_pos = (new_x, new_y)
            info["agent_pos"] = self.agent_pos

            # Бонус за исследование новых клеток
            if self.agent_pos not in self.visited_cells:
                self.visited_cells.add(self.agent_pos)
                reward += self.exploration_bonus

            # Проверка столкновения с врагом
            for enemy_id, pos in self.enemy_positions.items():
                if pos == self.agent_pos and self.enemies_alive[enemy_id]:
                    self.current_enemy_encounter = enemy_id
                    info["battle_triggered"] = True
                    info["enemy_id"] = enemy_id
                    break

        # Проверка лимита шагов
        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info

    def _action_to_delta(self, action: int) -> Tuple[int, int]:
        """Преобразование действия в смещение (dx, dy)."""
        deltas = {
            self.ACTION_UP: (0, -1),
            self.ACTION_DOWN: (0, 1),
            self.ACTION_LEFT: (-1, 0),
            self.ACTION_RIGHT: (1, 0),
            self.ACTION_UP_LEFT: (-1, -1),
            self.ACTION_UP_RIGHT: (1, -1),
            self.ACTION_DOWN_LEFT: (-1, 1),
            self.ACTION_DOWN_RIGHT: (1, 1),
            self.ACTION_REST: (0, 0),  # Не двигаемся при отдыхе
        }
        return deltas.get(action, (0, 0))

    def _get_obs(self) -> np.ndarray:
        """Формирует вектор наблюдения."""
        # Нормализованная позиция агента
        x_norm = self.agent_pos[0] / max(1, self.grid_size - 1)
        y_norm = self.agent_pos[1] / max(1, self.grid_size - 1)

        # Флаги живых врагов
        enemy_ids = sorted(self.enemy_positions.keys())
        enemies_flags = [
            1.0 if self.enemies_alive.get(i, False) else 0.0
            for i in enemy_ids
        ]

        # Расстояния до врагов (нормализованные)
        max_dist = np.sqrt(2) * (self.grid_size - 1)
        distances = []
        for i in enemy_ids:
            if i in self.enemy_positions:
                ex, ey = self.enemy_positions[i]
                dist = np.sqrt(
                    (self.agent_pos[0] - ex) ** 2 + (self.agent_pos[1] - ey) ** 2
                )
                distances.append(dist / max(1e-9, max_dist))
            else:
                distances.append(1.0)  # Враг отсутствует — максимальное расстояние

        # Доля исследованных клеток
        total_cells = self.grid_size ** 2
        visited_ratio = len(self.visited_cells) / total_cells

        return np.array(
            [x_norm, y_norm] + enemies_flags + distances + [visited_ratio],
            dtype=np.float32,
        )

    def mark_enemy_defeated(self, enemy_id: int) -> None:
        """Помечает врага как побеждённого. Вызывается после победы в бою."""
        if enemy_id in self.enemies_alive:
            self.enemies_alive[enemy_id] = False
            self.current_enemy_encounter = None

    def all_enemies_defeated(self) -> bool:
        """Проверяет, все ли враги побеждены."""
        return not any(self.enemies_alive.values())

    def get_enemy_at_position(self, pos: Tuple[int, int]) -> Optional[int]:
        """Возвращает ID врага на заданной позиции (если есть и жив)."""
        for enemy_id, enemy_pos in self.enemy_positions.items():
            if enemy_pos == pos and self.enemies_alive.get(enemy_id, False):
                return enemy_id
        return None

    def render(self, mode: str = "ansi") -> Optional[str]:
        """Отрисовка карты в текстовом виде."""
        if mode != "ansi":
            return None

        enemy_ids = sorted(self.enemy_positions.keys())

        lines = []
        lines.append("+" + "-" * (self.grid_size * 2 + 1) + "+")

        for y in range(self.grid_size):
            row = "| "
            for x in range(self.grid_size):
                pos = (x, y)
                if pos == self.agent_pos:
                    row += "A "  # Agent
                elif pos in [self.enemy_positions.get(i) for i in enemy_ids]:
                    # Найти какой враг
                    enemy_id = self.get_enemy_at_position(pos)
                    if enemy_id is not None:
                        row += f"{enemy_id} "  # Живой враг
                    else:
                        row += "x "  # Побеждённый враг
                elif pos in self.visited_cells:
                    row += ". "  # Посещённая клетка
                else:
                    row += "  "  # Непосещённая клетка
            row += "|"
            lines.append(row)

        lines.append("+" + "-" * (self.grid_size * 2 + 1) + "+")
        lines.append(f"Agent: {self.agent_pos}")
        lines.append(f"Enemies alive: {[k for k, v in self.enemies_alive.items() if v]}")
        lines.append(f"Visited: {len(self.visited_cells)}/{self.grid_size**2}")

        result = "\n".join(lines)
        print(result)
        return result

    def compute_action_mask(self) -> np.ndarray:
        """
        Возвращает маску допустимых действий.
        Все 9 действий всегда доступны (стены обрабатываются clamp'ом).
        """
        return np.ones(9, dtype=bool)
