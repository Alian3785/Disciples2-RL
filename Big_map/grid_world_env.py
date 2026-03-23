# grid_world_env.py
"""
Grid navigation environment for the campaign map.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from grid import (
    BASE_ENEMY_POSITIONS,
    BASE_GRID_SIZE,
    DEFAULT_GRID_SIZE,
    DEFAULT_HERO_GRID_POSITION,
    clamp_grid_pos,
    resolve_obstacle_tiles,
    scale_enemy_positions,
)


class GridWorldEnv(gym.Env):
    """
    Grid movement layer used by CampaignEnv.

    Actions (9):
        0: UP
        1: DOWN
        2: LEFT
        3: RIGHT
        4: UP-LEFT
        5: UP-RIGHT
        6: DOWN-LEFT
        7: DOWN-RIGHT
        8: REST
    """

    metadata = {"render_modes": ["human", "ansi"]}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_UP_LEFT = 4
    ACTION_UP_RIGHT = 5
    ACTION_DOWN_LEFT = 6
    ACTION_DOWN_RIGHT = 7
    ACTION_REST = 8
    OBSTACLE_BUMP_PENALTY = -0.25

    BASE_GRID_SIZE = BASE_GRID_SIZE
    BASE_ENEMY_POSITIONS = BASE_ENEMY_POSITIONS

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        enemy_positions: Optional[Dict[int, Tuple[int, int]]] = None,
        obstacle_positions: Optional[Tuple[Tuple[int, int], ...] | Set[Tuple[int, int]]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        step_penalty: float = -0.01,
        exploration_bonus: float = 0.005,
        obstacle_penalty: float = OBSTACLE_BUMP_PENALTY,
        max_steps: int = 1000,
    ):
        super().__init__()

        self.grid_size = int(grid_size)
        if self.grid_size < 2:
            raise ValueError("grid_size must be >= 2")

        if start_position is None:
            self.start_position = clamp_grid_pos(
                DEFAULT_HERO_GRID_POSITION,
                self.grid_size,
            )
        else:
            self.start_position = self._clamp_to_grid(start_position)

        self.step_penalty = step_penalty
        self.exploration_bonus = exploration_bonus
        self.obstacle_penalty = float(obstacle_penalty)
        self.max_steps = max_steps

        if enemy_positions is None:
            self.enemy_positions = self._scaled_default_enemy_positions(self.grid_size)
        else:
            self.enemy_positions = {
                int(enemy_id): self._clamp_to_grid(pos)
                for enemy_id, pos in enemy_positions.items()
            }
        self.num_enemies = len(self.enemy_positions)
        self.obstacle_positions = self._resolve_obstacle_positions(obstacle_positions)

        self.action_space = spaces.Discrete(9)
        obs_len = 2 + self.num_enemies + self.num_enemies + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_len,),
            dtype=np.float32,
        )

        self.agent_pos: Tuple[int, int] = self.start_position
        self.enemies_alive: Dict[int, bool] = {eid: True for eid in self.enemy_positions.keys()}
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.current_enemy_encounter: Optional[int] = None
        self.chest_positions: Set[Tuple[int, int]] = set()
        self.merchant_positions: Set[Tuple[int, int]] = set()
        self.step_count = 0

    def _clamp_to_grid(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        return clamp_grid_pos(pos, self.grid_size)

    def _resolve_obstacle_positions(
        self,
        obstacle_positions: Optional[Tuple[Tuple[int, int], ...] | Set[Tuple[int, int]]],
    ) -> Set[Tuple[int, int]]:
        if obstacle_positions is None:
            return set(
                resolve_obstacle_tiles(
                    self.grid_size,
                    self.enemy_positions,
                    heal_tiles=(self.start_position,),
                    start_position=self.start_position,
                )
            )

        blocked_tiles = {tuple(pos) for pos in self.enemy_positions.values()}
        blocked_tiles.add(self.start_position)
        resolved: Set[Tuple[int, int]] = set()
        for raw_pos in obstacle_positions:
            tile = self._clamp_to_grid(raw_pos)
            if tile in blocked_tiles:
                continue
            resolved.add(tile)
        return resolved

    @classmethod
    def _scaled_default_enemy_positions(cls, target_grid_size: int) -> Dict[int, Tuple[int, int]]:
        return scale_enemy_positions(
            target_grid_size,
            base_grid_size=cls.BASE_GRID_SIZE,
            base_enemy_positions=cls.BASE_ENEMY_POSITIONS,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.agent_pos = self.start_position
        self.enemies_alive = {eid: True for eid in self.enemy_positions.keys()}
        self.visited_cells = {self.start_position}
        self.current_enemy_encounter = None
        self.step_count = 0

        return self._get_obs(), {"agent_pos": self.agent_pos}

    def step(self, action: int):
        self.step_count += 1
        action = int(action)

        reward = self.step_penalty
        terminated = False
        truncated = False
        info = {
            "battle_triggered": False,
            "rest_action": False,
            "blocked_by_obstacle": False,
            "blocked_by_boundary": False,
            "blocked_target": None,
            "enemy_id": None,
            "agent_pos": self.agent_pos,
            "step": self.step_count,
        }

        if action == self.ACTION_REST:
            info["rest_action"] = True
        else:
            target_pos = self._target_pos_for_action(action)
            if not self._is_within_grid(target_pos):
                info["blocked_by_boundary"] = True
                info["blocked_target"] = target_pos
            elif target_pos in self.obstacle_positions:
                reward += self.obstacle_penalty
                info["blocked_by_obstacle"] = True
                info["blocked_target"] = target_pos
            else:
                self.agent_pos = target_pos
                info["agent_pos"] = self.agent_pos

                if self.agent_pos not in self.visited_cells:
                    self.visited_cells.add(self.agent_pos)
                    reward += self.exploration_bonus

                enemy_id = self.get_enemy_at_position(self.agent_pos)
                if enemy_id is not None:
                    self.current_enemy_encounter = enemy_id
                    info["battle_triggered"] = True
                    info["enemy_id"] = enemy_id

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info

    def _action_to_delta(self, action: int) -> Tuple[int, int]:
        deltas = {
            self.ACTION_UP: (0, -1),
            self.ACTION_DOWN: (0, 1),
            self.ACTION_LEFT: (-1, 0),
            self.ACTION_RIGHT: (1, 0),
            self.ACTION_UP_LEFT: (-1, -1),
            self.ACTION_UP_RIGHT: (1, -1),
            self.ACTION_DOWN_LEFT: (-1, 1),
            self.ACTION_DOWN_RIGHT: (1, 1),
            self.ACTION_REST: (0, 0),
        }
        return deltas.get(action, (0, 0))

    def _target_pos_for_action(self, action: int) -> Tuple[int, int]:
        dx, dy = self._action_to_delta(action)
        return self.agent_pos[0] + dx, self.agent_pos[1] + dy

    def _is_within_grid(self, pos: Tuple[int, int]) -> bool:
        return 0 <= int(pos[0]) < self.grid_size and 0 <= int(pos[1]) < self.grid_size

    def _get_obs(self) -> np.ndarray:
        x_norm = self.agent_pos[0] / max(1, self.grid_size - 1)
        y_norm = self.agent_pos[1] / max(1, self.grid_size - 1)

        enemy_ids = sorted(self.enemy_positions.keys())
        enemies_flags = [1.0 if self.enemies_alive.get(i, False) else 0.0 for i in enemy_ids]

        max_dist = np.sqrt(2) * (self.grid_size - 1)
        distances = []
        for enemy_id in enemy_ids:
            if enemy_id in self.enemy_positions:
                ex, ey = self.enemy_positions[enemy_id]
                dist = np.sqrt((self.agent_pos[0] - ex) ** 2 + (self.agent_pos[1] - ey) ** 2)
                distances.append(dist / max(1e-9, max_dist))
            else:
                distances.append(1.0)

        total_cells = self.grid_size**2
        visited_ratio = len(self.visited_cells) / total_cells

        return np.array(
            [x_norm, y_norm] + enemies_flags + distances + [visited_ratio],
            dtype=np.float32,
        )

    def mark_enemy_defeated(self, enemy_id: int) -> None:
        if enemy_id in self.enemies_alive:
            self.enemies_alive[enemy_id] = False
            self.current_enemy_encounter = None

    def all_enemies_defeated(self) -> bool:
        return not any(self.enemies_alive.values())

    def get_enemy_at_position(self, pos: Tuple[int, int]) -> Optional[int]:
        for enemy_id, enemy_pos in self.enemy_positions.items():
            if enemy_pos == pos and self.enemies_alive.get(enemy_id, False):
                return enemy_id
        return None

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode != "ansi":
            return None

        enemy_ids = sorted(self.enemy_positions.keys())
        chest_positions = {
            self._clamp_to_grid(pos)
            for pos in getattr(self, "chest_positions", set()) or set()
        }
        merchant_positions = {
            self._clamp_to_grid(pos)
            for pos in getattr(self, "merchant_positions", set()) or set()
        }

        lines = []
        lines.append("+" + "-" * (self.grid_size * 2 + 1) + "+")
        for y in range(self.grid_size):
            row = "| "
            for x in range(self.grid_size):
                pos = (x, y)
                if pos == self.agent_pos:
                    row += "A "
                elif pos in self.obstacle_positions:
                    row += "# "
                elif pos in merchant_positions:
                    row += "M "
                elif pos in chest_positions:
                    row += "T "
                elif pos in [self.enemy_positions.get(i) for i in enemy_ids]:
                    enemy_id = self.get_enemy_at_position(pos)
                    row += f"{enemy_id} " if enemy_id is not None else "x "
                elif pos in self.visited_cells:
                    row += ". "
                else:
                    row += "  "
            row += "|"
            lines.append(row)

        lines.append("+" + "-" * (self.grid_size * 2 + 1) + "+")
        lines.append(f"Agent: {self.agent_pos}")
        lines.append(f"Enemies alive: {[k for k, v in self.enemies_alive.items() if v]}")
        lines.append(f"Merchants: {sorted(merchant_positions)}")
        lines.append(f"Chests: {sorted(chest_positions)}")
        lines.append(f"Visited: {len(self.visited_cells)}/{self.grid_size**2}")

        result = "\n".join(lines)
        print(result)
        return result

    def compute_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[self.ACTION_REST] = True

        for action in range(self.ACTION_REST):
            target_pos = self._target_pos_for_action(action)
            mask[action] = self._is_within_grid(target_pos) and (
                target_pos not in self.obstacle_positions
            )

        return mask
