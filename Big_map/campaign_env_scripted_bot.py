"""Scripted capital bot behavior for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignScriptedBotMixin:
    SCRIPTED_CAPITAL_BOT_MAX_STEPS_PER_TURN = 20
    SCRIPTED_CAPITAL_BOT_GRID_MOVE_COST = 2
    SCRIPTED_CAPITAL_BOT_BATTLE_STEP_LIMIT = 300
    SCRIPTED_CAPITAL_BOT_SYMBOL = "B"
    SCRIPTED_CAPITAL_BOT_COLOR = (0.6, 0.15, 0.85, 0.95)

    _SCRIPTED_BOT_RED_TO_BLUE_POSITION = {
        1: 7,
        2: 8,
        3: 9,
        4: 10,
        5: 11,
        6: 12,
    }

    def _init_scripted_capital_bot_state(self) -> None:
        self.scripted_capital_bot_enabled = True
        self.scripted_capital_bot_enemy_id = int(self.EMPIRE_TERRITORY_SOURCE_ENEMY_ID)
        self.scripted_capital_bot_home = tuple(
            int(coord) for coord in self.empire_territory_source_tile
        )
        self.scripted_capital_bot_position = tuple(self.scripted_capital_bot_home)
        self.scripted_capital_bot_state = "hunting"
        self.scripted_capital_bot_respawn_turns_left = 0
        self.scripted_capital_bot_rest_turns_left = 0
        self.scripted_capital_bot_team_state = self._create_scripted_capital_bot_team()
        self.scripted_capital_bot_enemies_defeated = 0
        self._scripted_bot_last_move_points_spent = 0
        self._scripted_bot_last_move_costs = []
        self._scripted_bot_last_move_terrains = []
        self.scripted_capital_bot_last_info: Dict[str, object] = {
            "enabled": True,
            "state": self.scripted_capital_bot_state,
            "position": self.scripted_capital_bot_position,
            "enemies_defeated": 0,
            "events": [],
        }
        self._sync_scripted_capital_bot_grid_state()

    def _reset_scripted_capital_bot_state(self) -> None:
        if not getattr(self, "scripted_capital_bot_enabled", True):
            return
        self.scripted_capital_bot_position = tuple(self.scripted_capital_bot_home)
        self.scripted_capital_bot_state = "hunting"
        self.scripted_capital_bot_respawn_turns_left = 0
        self.scripted_capital_bot_rest_turns_left = 0
        self.scripted_capital_bot_team_state = self._create_scripted_capital_bot_team()
        self.scripted_capital_bot_enemies_defeated = 0
        self._scripted_bot_last_move_points_spent = 0
        self._scripted_bot_last_move_costs = []
        self._scripted_bot_last_move_terrains = []
        self.scripted_capital_bot_last_info = {
            "enabled": True,
            "state": self.scripted_capital_bot_state,
            "position": self.scripted_capital_bot_position,
            "enemies_defeated": 0,
            "events": ["reset"],
        }
        self._sync_scripted_capital_bot_grid_state()

    def _create_scripted_capital_bot_team(self) -> List[Dict]:
        source_team = ENEMY_CONFIGS.get(int(self.EMPIRE_TERRITORY_SOURCE_ENEMY_ID), [])
        bot_units: List[Dict] = []
        for source_unit in source_team:
            if self._is_empty_enemy_unit(source_unit):
                continue
            source_position = int(source_unit.get("position", -1) or -1)
            bot_position = self._SCRIPTED_BOT_RED_TO_BLUE_POSITION.get(source_position)
            if bot_position is None:
                continue
            unit = deepcopy(source_unit)
            unit["team"] = "blue"
            unit["position"] = int(bot_position)
            unit["stand"] = "ahead" if int(bot_position) in (7, 8, 9) else "behind"
            unit["hp"] = float(unit.get("hp", unit.get("health", 0)) or 0)
            unit["maxhp"] = float(unit.get("maxhp", unit.get("max_health", 0)) or 0)
            bot_units.append(unit)
        team = self._build_battle_team_with_placeholders("blue", bot_units)
        self._sync_hero_progression_flags(team)
        return team

    def _scripted_capital_bot_info(self) -> Dict[str, object]:
        info = dict(getattr(self, "scripted_capital_bot_last_info", {}) or {})
        info.update(
            {
                "enabled": bool(getattr(self, "scripted_capital_bot_enabled", False)),
                "state": str(getattr(self, "scripted_capital_bot_state", "")),
                "position": tuple(getattr(self, "scripted_capital_bot_position", ())),
                "home": tuple(getattr(self, "scripted_capital_bot_home", ())),
                "symbol": self.SCRIPTED_CAPITAL_BOT_SYMBOL,
                "enemies_defeated": int(
                    getattr(self, "scripted_capital_bot_enemies_defeated", 0) or 0
                ),
            }
        )
        return {
            "scripted_capital_bot": info,
            "scripted_capital_bot_enemies_defeated": int(info["enemies_defeated"]),
        }

    def _sync_scripted_capital_bot_grid_state(self) -> None:
        if not hasattr(self, "grid_env"):
            return
        bot_enemy_id = int(getattr(self, "scripted_capital_bot_enemy_id", -1))
        if bot_enemy_id in getattr(self.grid_env, "enemies_alive", {}):
            self.grid_env.enemies_alive[bot_enemy_id] = False

        dynamic_blocked = set(getattr(self.grid_env, "dynamic_blocked_positions", set()) or set())
        previous = getattr(self, "_scripted_capital_bot_previous_blocked_pos", None)
        if previous is not None:
            dynamic_blocked.discard(tuple(previous))

        if (
            bool(getattr(self, "scripted_capital_bot_enabled", False))
            and str(getattr(self, "scripted_capital_bot_state", "")) != "defeated"
        ):
            position = tuple(getattr(self, "scripted_capital_bot_position", ()))
            if len(position) == 2 and position != tuple(getattr(self.grid_env, "agent_pos", ())):
                dynamic_blocked.add(position)
            self._scripted_capital_bot_previous_blocked_pos = position
            self.grid_env.scripted_bot_position = position
            self.grid_env.scripted_bot_symbol = self.SCRIPTED_CAPITAL_BOT_SYMBOL
        else:
            self._scripted_capital_bot_previous_blocked_pos = None
            self.grid_env.scripted_bot_position = None
        self.grid_env.dynamic_blocked_positions = dynamic_blocked

    def _advance_scripted_capital_bot_one_turn(self) -> Dict[str, object]:
        if not bool(getattr(self, "scripted_capital_bot_enabled", False)):
            return {"enabled": False, "events": []}

        events: List[str] = []
        state = str(getattr(self, "scripted_capital_bot_state", "hunting"))
        info: Dict[str, object] = {
            "enabled": True,
            "state_before": state,
            "position_before": tuple(self.scripted_capital_bot_position),
            "events": events,
        }

        if state == "defeated":
            self.scripted_capital_bot_respawn_turns_left = max(
                0,
                int(self.scripted_capital_bot_respawn_turns_left) - 1,
            )
            events.append("respawn_timer")
            if self.scripted_capital_bot_respawn_turns_left <= 0:
                self.scripted_capital_bot_team_state = self._create_scripted_capital_bot_team()
                self.scripted_capital_bot_position = tuple(self.scripted_capital_bot_home)
                self.scripted_capital_bot_state = "hunting"
                events.append("respawned")
                self._log("Бот столицы людей снова появился в столице.")
        elif state == "resting":
            self.scripted_capital_bot_rest_turns_left = max(
                0,
                int(self.scripted_capital_bot_rest_turns_left) - 1,
            )
            events.append("rested_at_capital")
            if self.scripted_capital_bot_rest_turns_left <= 0:
                revived, healed = self._fully_restore_scripted_capital_bot_team()
                self.scripted_capital_bot_state = "hunting"
                events.append("fully_restored")
                info["revived_units"] = int(revived)
                info["healed_units"] = int(healed)
                self._log(
                    "Бот столицы людей завершил отдых: "
                    f"воскрешено {revived}, вылечено {healed}."
                )
        elif state == "returning":
            path = self._scripted_bot_path_to(self.scripted_capital_bot_home)
            moved_path = self._move_scripted_bot_along_path(path)
            events.append("returning")
            info["path"] = moved_path
            info.update(self._scripted_bot_movement_info(moved_path))
            if tuple(self.scripted_capital_bot_position) == tuple(self.scripted_capital_bot_home):
                self.scripted_capital_bot_state = "resting"
                self.scripted_capital_bot_rest_turns_left = 1
                events.append("arrived_home")
                self._log("Бот столицы людей вернулся в столицу и отдыхает 1 ход.")
        else:
            came_from, enemy_distances = self._scripted_bot_explore_for_enemies()
            target = self._pick_scripted_bot_nearest_enemy(enemy_distances)
            info["target"] = target
            if target is None:
                events.append("no_target")
            else:
                target_pos = tuple(target["position"])
                path = self._scripted_bot_path_from_came_from(came_from, target_pos)
                if not path:
                    path = self._scripted_bot_path_to(target_pos)
                path = self._scripted_bot_path_to_engagement_tile(path, target_pos)
                moved_path = self._move_scripted_bot_along_path(path)
                events.append("hunting")
                info["path"] = moved_path
                info.update(self._scripted_bot_movement_info(moved_path))
                info["target_enemy_id"] = int(target["enemy_id"])
                if self._scripted_bot_can_engage_enemy_from_position(
                    self.scripted_capital_bot_position,
                    target_pos,
                ):
                    battle_info = self._run_scripted_capital_bot_battle(int(target["enemy_id"]))
                    info["battle"] = battle_info
                    events.append("battle")

        self._sync_scripted_capital_bot_grid_state()
        info["state"] = str(self.scripted_capital_bot_state)
        info["position"] = tuple(self.scripted_capital_bot_position)
        self.scripted_capital_bot_last_info = info
        return dict(info)

    def _scripted_bot_blocked_tiles(
        self,
        *,
        target_tile: Optional[Tuple[int, int]] = None,
    ) -> set[Tuple[int, int]]:
        blocked = {
            (int(pos[0]), int(pos[1]))
            for pos in (getattr(self.grid_env, "obstacle_positions", set()) or set())
        }
        agent_pos = tuple(getattr(self.grid_env, "agent_pos", ()))
        if len(agent_pos) == 2:
            blocked.add((int(agent_pos[0]), int(agent_pos[1])))
        for enemy_id, pos in (getattr(self.grid_env, "enemy_positions", {}) or {}).items():
            if not bool(getattr(self.grid_env, "enemies_alive", {}).get(int(enemy_id), False)):
                continue
            enemy_tile = (int(pos[0]), int(pos[1]))
            if target_tile is not None and enemy_tile == tuple(target_tile):
                continue
            blocked.add(enemy_tile)
        blocked.discard(tuple(self.scripted_capital_bot_position))
        if target_tile is not None:
            blocked.discard(tuple(target_tile))
        return blocked

    def _scripted_bot_path_to(self, target_tile: Tuple[int, int]) -> List[Tuple[int, int]]:
        start = tuple(int(coord) for coord in self.scripted_capital_bot_position)
        target = tuple(int(coord) for coord in target_tile)
        if start == target:
            return [start]

        blocked = self._scripted_bot_blocked_tiles(target_tile=target)
        if start in blocked:
            blocked.discard(start)

        frontier = deque([start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
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
            if (x, y) == target:
                break
            for dx, dy in neighbor_offsets:
                tile = (x + dx, y + dy)
                if not (0 <= tile[0] < int(self.grid_size) and 0 <= tile[1] < int(self.grid_size)):
                    continue
                if tile in blocked or tile in came_from:
                    continue
                came_from[tile] = (x, y)
                frontier.append(tile)

        if target not in came_from:
            return self._scripted_bot_greedy_path(target, blocked)

        path = [target]
        current = target
        while came_from[current] is not None:
            current = came_from[current]  # type: ignore[index]
            path.append(current)
        path.reverse()
        return path

    def _scripted_bot_greedy_path(
        self,
        target: Tuple[int, int],
        blocked: set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        current = tuple(int(coord) for coord in self.scripted_capital_bot_position)
        path = [current]
        for _ in range(self._scripted_bot_max_grid_moves_per_turn()):
            if current == target:
                break
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    tile = (current[0] + dx, current[1] + dy)
                    if not (0 <= tile[0] < int(self.grid_size) and 0 <= tile[1] < int(self.grid_size)):
                        continue
                    if tile in blocked and tile != target:
                        continue
                    distance = max(abs(tile[0] - target[0]), abs(tile[1] - target[1]))
                    candidates.append((distance, abs(tile[0] - target[0]) + abs(tile[1] - target[1]), tile))
            if not candidates:
                break
            _, _, next_tile = min(candidates)
            if next_tile == current:
                break
            path.append(next_tile)
            current = next_tile
        return path

    def _move_scripted_bot_along_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not path:
            self._scripted_bot_last_move_points_spent = 0
            self._scripted_bot_last_move_costs = []
            self._scripted_bot_last_move_terrains = []
            return []
        remaining_points = max(0, int(self.SCRIPTED_CAPITAL_BOT_MAX_STEPS_PER_TURN))
        movement: List[Tuple[int, int]] = []
        move_costs: List[int] = []
        move_terrains: List[str] = []
        for raw_tile in path[1:]:
            if remaining_points <= 0:
                break
            tile = (int(raw_tile[0]), int(raw_tile[1]))
            move_cost = int(
                self._campaign_tile_move_cost(
                    tile,
                    units=self.scripted_capital_bot_team_state,
                    allow_boots=False,
                )
            )
            terrain = self._campaign_tile_terrain(tile)
            spent = min(remaining_points, max(1, int(move_cost)))
            movement.append(tile)
            move_costs.append(int(move_cost))
            move_terrains.append(str(terrain))
            remaining_points -= int(spent)
        self._scripted_bot_last_move_points_spent = (
            max(0, int(self.SCRIPTED_CAPITAL_BOT_MAX_STEPS_PER_TURN)) - remaining_points
        )
        self._scripted_bot_last_move_costs = list(move_costs)
        self._scripted_bot_last_move_terrains = list(move_terrains)
        if movement:
            old_pos = tuple(self.scripted_capital_bot_position)
            self.scripted_capital_bot_position = tuple(movement[-1])
            self._log(
                f"Бот столицы людей: {old_pos} -> {self.scripted_capital_bot_position} "
                f"({len(movement)} шаг.)"
            )
        return movement

    def _scripted_bot_grid_move_cost(self) -> int:
        move_costs = list(getattr(self, "_scripted_bot_last_move_costs", []) or [])
        if move_costs:
            return max(1, int(move_costs[0]))
        return max(1, int(self.SCRIPTED_CAPITAL_BOT_GRID_MOVE_COST))

    def _scripted_bot_max_grid_moves_per_turn(self) -> int:
        movement_budget = max(0, int(self.SCRIPTED_CAPITAL_BOT_MAX_STEPS_PER_TURN))
        return movement_budget

    def _scripted_bot_move_points_spent(self, moved_path: List[Tuple[int, int]]) -> int:
        spent = getattr(self, "_scripted_bot_last_move_points_spent", None)
        if spent is not None:
            return int(spent)
        return int(len(moved_path)) * self._scripted_bot_grid_move_cost()

    def _scripted_bot_movement_info(self, moved_path: List[Tuple[int, int]]) -> Dict[str, object]:
        return {
            "move_cost": int(self._scripted_bot_grid_move_cost()),
            "move_points_budget": int(self.SCRIPTED_CAPITAL_BOT_MAX_STEPS_PER_TURN),
            "move_points_spent": int(self._scripted_bot_move_points_spent(moved_path)),
            "move_costs": list(getattr(self, "_scripted_bot_last_move_costs", []) or []),
            "move_terrains": list(getattr(self, "_scripted_bot_last_move_terrains", []) or []),
        }

    def _scripted_bot_path_to_engagement_tile(
        self,
        path: List[Tuple[int, int]],
        enemy_tile: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        if not path:
            return []
        normalized_enemy_tile = (int(enemy_tile[0]), int(enemy_tile[1]))
        if tuple(path[-1]) == normalized_enemy_tile and len(path) > 1:
            return list(path[:-1])
        return list(path)

    @staticmethod
    def _scripted_bot_can_engage_enemy_from_position(
        position: Tuple[int, int],
        enemy_tile: Tuple[int, int],
    ) -> bool:
        px, py = int(position[0]), int(position[1])
        ex, ey = int(enemy_tile[0]), int(enemy_tile[1])
        return max(abs(px - ex), abs(py - ey)) <= 1

    def _scripted_bot_alive_enemy_tiles(self) -> Dict[Tuple[int, int], int]:
        """Карта тайл -> enemy_id для всех живых врагов (кроме самого бота)."""
        bot_enemy_id = int(self.scripted_capital_bot_enemy_id)
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}
        result: Dict[Tuple[int, int], int] = {}
        for raw_enemy_id, raw_pos in (getattr(self.grid_env, "enemy_positions", {}) or {}).items():
            enemy_id = int(raw_enemy_id)
            if enemy_id == bot_enemy_id:
                continue
            if not bool(enemies_alive.get(enemy_id, False)):
                continue
            if not self._enemy_team_has_living_units(enemy_id):
                continue
            tile = (int(raw_pos[0]), int(raw_pos[1]))
            result[tile] = enemy_id
        return result

    def _scripted_bot_explore_for_enemies(
        self,
    ) -> Tuple[Dict[Tuple[int, int], Optional[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
        """Один BFS из позиции бота. Враги — терминальные клетки (через них не идём).

        Возвращает (came_from, distances_to_enemy_tiles).
        """
        start = tuple(int(coord) for coord in self.scripted_capital_bot_position)
        enemy_tiles = self._scripted_bot_alive_enemy_tiles()
        blocked = {
            (int(pos[0]), int(pos[1]))
            for pos in (getattr(self.grid_env, "obstacle_positions", set()) or set())
        }
        agent_pos = tuple(getattr(self.grid_env, "agent_pos", ()))
        if len(agent_pos) == 2:
            blocked.add((int(agent_pos[0]), int(agent_pos[1])))
        blocked.discard(start)

        grid_size = int(self.grid_size)
        neighbor_offsets = (
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        )

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        distances: Dict[Tuple[int, int], int] = {start: 0}
        enemy_distances: Dict[Tuple[int, int], int] = {}
        if start in enemy_tiles:
            enemy_distances[start] = 0

        frontier = deque([start])
        while frontier:
            current = frontier.popleft()
            if current in enemy_tiles and current != start:
                continue
            cx, cy = current
            current_dist = distances[current]
            for dx, dy in neighbor_offsets:
                tile = (cx + dx, cy + dy)
                if not (0 <= tile[0] < grid_size and 0 <= tile[1] < grid_size):
                    continue
                if tile in came_from:
                    continue
                if tile in blocked:
                    continue
                came_from[tile] = current
                distances[tile] = current_dist + 1
                if tile in enemy_tiles:
                    enemy_distances[tile] = current_dist + 1
                else:
                    frontier.append(tile)

        return came_from, enemy_distances

    def _pick_scripted_bot_nearest_enemy(
        self,
        enemy_distances: Dict[Tuple[int, int], int],
    ) -> Optional[Dict[str, object]]:
        enemy_tiles = self._scripted_bot_alive_enemy_tiles()
        if not enemy_tiles:
            return None

        bot_pos = tuple(self.scripted_capital_bot_position)
        candidates: List[Tuple[int, int, int, int, Tuple[int, int]]] = []
        for tile, enemy_id in enemy_tiles.items():
            if tile in enemy_distances:
                distance = enemy_distances[tile]
                candidates.append((distance, enemy_id, tile[1], tile[0], tile))
            else:
                fallback = max(abs(bot_pos[0] - tile[0]), abs(bot_pos[1] - tile[1]))
                candidates.append((fallback + 10000, enemy_id, tile[1], tile[0], tile))

        distance, enemy_id, _y, _x, position = min(candidates)
        return {
            "enemy_id": int(enemy_id),
            "position": tuple(position),
            "path_distance": int(distance) if distance < 10000 else None,
            "fallback_distance": None if distance < 10000 else int(distance - 10000),
            "description": ENEMY_DESCRIPTIONS.get(int(enemy_id), "Неизвестный враг"),
        }

    def _scripted_bot_path_from_came_from(
        self,
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        target: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        if target not in came_from:
            return []
        path: List[Tuple[int, int]] = [target]
        current: Optional[Tuple[int, int]] = target
        while came_from.get(current) is not None:
            current = came_from[current]
            path.append(current)  # type: ignore[arg-type]
        path.reverse()
        return path

    def _find_scripted_bot_nearest_enemy(self) -> Optional[Dict[str, object]]:
        _came_from, enemy_distances = self._scripted_bot_explore_for_enemies()
        return self._pick_scripted_bot_nearest_enemy(enemy_distances)

    def _run_scripted_capital_bot_battle(self, enemy_id: int) -> Dict[str, object]:
        self._log(f"Бот столицы людей вступает в бой с врагом {enemy_id}.")
        red_team_state = self._get_enemy_team_state(enemy_id)
        red_team = self._build_battle_team_with_placeholders(
            "red",
            red_team_state or ENEMY_CONFIGS.get(enemy_id, ENEMY_CONFIGS[1]),
        )
        self._apply_settlement_defender_armor_bonus(enemy_id, red_team)
        blue_team = self._build_battle_team_with_placeholders(
            "blue",
            self.scripted_capital_bot_team_state,
        )

        battle_env = BattleEnv(
            reward_win=self.battle_reward_win,
            reward_loss=self.battle_reward_loss,
            reward_step=self.battle_reward_step,
            log_enabled=self.log_enabled,
        )
        battle_env._init_with_custom_teams(red_team, blue_team)

        steps = 0
        while battle_env.winner is None and steps < int(self.SCRIPTED_CAPITAL_BOT_BATTLE_STEP_LIMIT):
            action = battle_env.scripted_action_for_current_blue()
            battle_env.step(action)
            steps += 1

        winner = battle_env.winner
        if winner == "blue":
            upgrade_count = self._apply_scripted_capital_bot_levelups(battle_env)
            self._save_scripted_capital_bot_state_from_battle(battle_env)
            self.grid_env.mark_enemy_defeated(enemy_id)
            self._clear_enemy_map_spell_effects_for_enemy(enemy_id)
            self._capture_objective_city_if_cleared(enemy_id)
            self._activate_legions_settlement_territory_if_cleared(enemy_id)
            self.scripted_capital_bot_enemies_defeated += 1
            self.scripted_capital_bot_state = "returning"
            self._log(
                f"Бот столицы людей победил врага {enemy_id}; "
                f"опыт {battle_env.last_battle_exp:g}, улучшений {upgrade_count}."
            )
        else:
            self._save_enemy_state_from_battle_env(enemy_id, battle_env)
            self.scripted_capital_bot_team_state = self._create_scripted_capital_bot_team()
            self.scripted_capital_bot_state = "defeated"
            self.scripted_capital_bot_respawn_turns_left = 1
            self._log(f"Бот столицы людей проиграл врагу {enemy_id}; респавн через 1 ход.")

        return {
            "enemy_id": int(enemy_id),
            "winner": winner or "timeout",
            "steps": int(steps),
            "exp": float(getattr(battle_env, "last_battle_exp", 0.0) or 0.0),
            "levelups": list(getattr(battle_env, "last_levelups", []) or []),
            "enemies_defeated": int(self.scripted_capital_bot_enemies_defeated),
        }

    def _save_enemy_state_from_battle_env(self, enemy_id: int, battle_env: BattleEnv) -> None:
        previous_battle_env = getattr(self, "battle_env", None)
        try:
            self.battle_env = battle_env
            self._save_enemy_state_from_battle(enemy_id)
        finally:
            self.battle_env = previous_battle_env

    def _save_scripted_capital_bot_state_from_battle(self, battle_env: BattleEnv) -> None:
        saved_team: List[Dict] = []
        for position in range(7, 13):
            battle_unit = next(
                (
                    deepcopy(unit)
                    for unit in battle_env.combined
                    if unit.get("team") == "blue"
                    and int(unit.get("position", -1) or -1) == int(position)
                    and not bool(unit.get("Summoned"))
                ),
                placeholder_unit("blue", position),
            )
            saved_team.append(self._normalize_scripted_bot_saved_unit(battle_unit))
        self.scripted_capital_bot_team_state = self._build_battle_team_with_placeholders(
            "blue",
            saved_team,
        )

    def _normalize_scripted_bot_saved_unit(self, unit: Dict) -> Dict:
        saved = deepcopy(unit)
        hp = max(0.0, float(saved.get("health", 0) or saved.get("hp", 0) or 0.0))
        saved["health"] = hp
        saved["hp"] = hp
        saved["team"] = "blue"
        saved["stand"] = "ahead" if int(saved.get("position", 0) or 0) in (7, 8, 9) else "behind"
        saved["initiative"] = int(saved.get("initiative_base", saved.get("initiative", 0)) or 0) if hp > 0 else 0
        for key, value in (
            ("defense", 0),
            ("waited", 0),
            ("paralyzed", 0),
            ("long_paralyzed", 0),
            ("running_away", 0),
            ("transformed", 0),
            ("poison_turns_left", 0),
            ("poison_damage_per_tick", 0),
            ("burn_turns_left", 0),
            ("burn_damage_per_tick", 0),
            ("uran_turns_left", 0),
            ("uran_damage_per_tick", 0),
            ("powerup", 0),
            ("bonusturn", 0),
        ):
            saved[key] = value
        saved["resilience_used_types"] = []
        return saved

    def _apply_scripted_capital_bot_levelups(self, battle_env: BattleEnv) -> int:
        levelup_names = list(getattr(battle_env, "last_levelups", []) or [])
        if not levelup_names:
            return 0
        blue_units_by_name: Dict[str, List[Dict]] = {}
        for unit in battle_env.combined:
            if unit.get("team") != "blue":
                continue
            name = str(unit.get("name", "") or "").strip()
            if name:
                blue_units_by_name.setdefault(name, []).append(unit)

        upgraded_count = 0
        for name in levelup_names:
            units = blue_units_by_name.get(str(name), [])
            if not units:
                continue
            unit = units.pop(0)
            turns_into = unit.get("turns_into", [])
            if not isinstance(turns_into, list):
                turns_into = [turns_into]
            target_name = next((str(target).strip() for target in turns_into if str(target).strip()), "")
            if not target_name:
                continue
            unit_data = self._find_unit_data_by_name(target_name)
            if unit_data is None:
                continue
            upgraded = self._build_unit_from_data(
                unit_data,
                team="blue",
                position=int(unit.get("position", 7) or 7),
            )
            unit.clear()
            unit.update(deepcopy(upgraded))
            upgraded_count += 1
            self._log(f'Бот столицы людей: "{name}" улучшен до "{target_name}" без построек.')
        return int(upgraded_count)

    def _fully_restore_scripted_capital_bot_team(self) -> Tuple[int, int]:
        revived = 0
        healed = 0
        for unit in self.scripted_capital_bot_team_state:
            if str(unit.get("name", "") or "") == "пусто":
                continue
            max_hp = float(unit.get("maxhp", unit.get("max_health", 0)) or 0)
            if max_hp <= 0:
                continue
            hp = float(unit.get("hp", unit.get("health", 0)) or 0)
            if hp <= 0:
                revived += 1
            elif hp < max_hp:
                healed += 1
            unit["hp"] = max_hp
            unit["health"] = max_hp
            unit["initiative"] = int(unit.get("initiative_base", unit.get("initiative", 0)) or 0)
        return int(revived), int(healed)
