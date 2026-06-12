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


_MISSING = object()


class _TrackedEnemyPositionDict(dict):
    def __init__(self, *args, on_change=None, **kwargs):
        self._on_change = on_change
        super().__init__(*args, **kwargs)

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self._changed()

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        self._changed()

    def clear(self) -> None:
        super().clear()
        self._changed()

    def pop(self, key, default=_MISSING):
        if default is _MISSING:
            value = super().pop(key)
        else:
            value = super().pop(key, default)
        self._changed()
        return value

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        value = super().setdefault(key, default)
        self._changed()
        return value

    def popitem(self):
        item = super().popitem()
        self._changed()
        return item

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._changed()


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

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_UP_LEFT = 4
    ACTION_UP_RIGHT = 5
    ACTION_DOWN_LEFT = 6
    ACTION_DOWN_RIGHT = 7
    ACTION_REST = 8
    ACTION_DELTAS: Tuple[Tuple[int, int], ...] = (
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
        (0, 0),
    )
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
        self._enemy_cache_dirty = True

        if enemy_positions is None:
            self.enemy_positions = self._scaled_default_enemy_positions(self.grid_size)
        else:
            self.enemy_positions = {
                int(enemy_id): self._clamp_to_grid(pos)
                for enemy_id, pos in enemy_positions.items()
            }
        self.num_enemies = len(self.enemy_positions)
        self.obstacle_positions = self._resolve_obstacle_positions(obstacle_positions)
        self._grid_span = max(1, self.grid_size - 1)
        self._grid_span_inv = 1.0 / float(self._grid_span)
        self._max_enemy_distance_inv = 1.0 / (np.sqrt(2.0) * float(self._grid_span))
        self._total_cells_inv = 1.0 / float(self.grid_size**2)
        self._enemy_cache_source_id: Optional[int] = None
        self._enemy_cache_len = -1
        self._enemy_ids: Tuple[int, ...] = ()
        self._enemy_coords = np.zeros((0, 2), dtype=np.float32)
        self._refresh_enemy_cache()

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
        self.spell_shop_positions: Set[Tuple[int, int]] = set()
        self.mercenary_positions: Set[Tuple[int, int]] = set()
        self.trainer_positions: Set[Tuple[int, int]] = set()
        self.mana_sources: Dict[Tuple[int, int], Dict[str, object]] = {}
        self.water_positions: Set[Tuple[int, int]] = set()
        self.forest_positions: Set[Tuple[int, int]] = set()
        self.road_positions: Set[Tuple[int, int]] = set()
        self.dynamic_blocked_positions: Set[Tuple[int, int]] = set()
        self.scripted_bot_position: Optional[Tuple[int, int]] = None
        self.scripted_bot_symbol: str = "B"
        self.step_count = 0

    @property
    def enemy_positions(self) -> Dict[int, Tuple[int, int]]:
        return self._enemy_positions

    @enemy_positions.setter
    def enemy_positions(self, positions: Dict[int, Tuple[int, int]]) -> None:
        self._enemy_positions = _TrackedEnemyPositionDict(
            positions,
            on_change=self._invalidate_enemy_cache,
        )
        self._invalidate_enemy_cache()

    def _invalidate_enemy_cache(self) -> None:
        self._enemy_cache_dirty = True

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

    def _refresh_enemy_cache(self) -> None:
        enemy_ids = tuple(sorted(int(enemy_id) for enemy_id in self.enemy_positions.keys()))
        self._enemy_ids = enemy_ids
        self._enemy_coords = np.array(
            [self.enemy_positions[enemy_id] for enemy_id in enemy_ids],
            dtype=np.float32,
        ).reshape((len(enemy_ids), 2))
        self._enemy_cache_source_id = id(self.enemy_positions)
        self._enemy_cache_len = len(self.enemy_positions)
        self._enemy_cache_dirty = False

    def _ensure_enemy_cache(self) -> None:
        if (
            self._enemy_cache_dirty
            or self._enemy_cache_source_id != id(self.enemy_positions)
            or self._enemy_cache_len != len(self.enemy_positions)
        ):
            self._refresh_enemy_cache()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._ensure_enemy_cache()

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
            elif target_pos in self.obstacle_positions or target_pos in self.dynamic_blocked_positions:
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
                battle_triggered_by = "same_tile" if enemy_id is not None else None
                if enemy_id is None:
                    enemy_id = self.get_adjacent_enemy_at_position(self.agent_pos)
                    battle_triggered_by = "adjacent" if enemy_id is not None else None
                if enemy_id is not None:
                    self.current_enemy_encounter = enemy_id
                    info["battle_triggered"] = True
                    info["enemy_id"] = enemy_id
                    info["enemy_pos"] = self.enemy_positions.get(enemy_id)
                    info["battle_triggered_by"] = battle_triggered_by

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info

    def _action_to_delta(self, action: int) -> Tuple[int, int]:
        action = int(action)
        if 0 <= action < len(self.ACTION_DELTAS):
            return self.ACTION_DELTAS[action]
        return (0, 0)

    def _target_pos_for_action(self, action: int) -> Tuple[int, int]:
        dx, dy = self._action_to_delta(action)
        return self.agent_pos[0] + dx, self.agent_pos[1] + dy

    def _is_within_grid(self, pos: Tuple[int, int]) -> bool:
        return 0 <= int(pos[0]) < self.grid_size and 0 <= int(pos[1]) < self.grid_size

    def _get_obs(self) -> np.ndarray:
        self._ensure_enemy_cache()

        enemy_ids = self._enemy_ids
        enemy_count = len(enemy_ids)
        obs = np.empty(2 + enemy_count + enemy_count + 1, dtype=np.float32)
        agent_x, agent_y = self.agent_pos

        obs[0] = float(agent_x) * self._grid_span_inv
        obs[1] = float(agent_y) * self._grid_span_inv

        flags_start = 2
        distances_start = flags_start + enemy_count
        for index, enemy_id in enumerate(enemy_ids):
            obs[flags_start + index] = (
                1.0 if self.enemies_alive.get(enemy_id, False) else 0.0
            )

        if enemy_count:
            dx = self._enemy_coords[:, 0] - float(agent_x)
            dy = self._enemy_coords[:, 1] - float(agent_y)
            obs[distances_start : distances_start + enemy_count] = (
                np.sqrt(dx * dx + dy * dy) * self._max_enemy_distance_inv
            )

        obs[-1] = float(len(self.visited_cells)) * self._total_cells_inv
        return obs

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

    def get_adjacent_enemy_at_position(self, pos: Tuple[int, int]) -> Optional[int]:
        px, py = int(pos[0]), int(pos[1])
        candidates = []
        for enemy_id, enemy_pos in self.enemy_positions.items():
            if not self.enemies_alive.get(enemy_id, False):
                continue
            ex, ey = int(enemy_pos[0]), int(enemy_pos[1])
            dx = abs(px - ex)
            dy = abs(py - ey)
            if max(dx, dy) == 1:
                candidates.append((int(enemy_id), ey, ex))
        if not candidates:
            return None
        return min(candidates)[0]

    def compute_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[self.ACTION_REST] = True
        agent_x, agent_y = self.agent_pos
        grid_size = self.grid_size
        obstacle_positions = self.obstacle_positions
        dynamic_blocked_positions = self.dynamic_blocked_positions

        for action, (dx, dy) in enumerate(self.ACTION_DELTAS[: self.ACTION_REST]):
            target_x = agent_x + dx
            target_y = agent_y + dy
            target_pos = (target_x, target_y)
            mask[action] = (
                0 <= target_x < grid_size
                and 0 <= target_y < grid_size
                and target_pos not in obstacle_positions
                and target_pos not in dynamic_blocked_positions
            )

        return mask
