# campaign_env_territory.py
"""CampaignTerritoryMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignTerritoryMixin:
    def _ruin_progress_info(self) -> Dict[str, object]:
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
        reward_data = self.RUIN_REWARD_BY_ENEMY_ID.get(int(enemy_id or -1))
        if reward_data is None:
            return {
                "ruin_cleared": False,
                "ruin_reward_applied": False,
                **self._ruin_progress_info(),
            }

        ruin_enemy_id = int(enemy_id or -1)
        ruin_title = str(reward_data.get("title", "") or "")
        item_name = str(reward_data.get("item", "") or "")
        gold_reward = max(0.0, float(reward_data.get("gold", 0.0) or 0.0))
        ruin_pos = tuple(int(v) for v in tuple(reward_data.get("ruin_pos", ())))
        marker_pos = tuple(int(v) for v in tuple(reward_data.get("marker_pos", ())))

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
        if item_name:
            artifact_grant_info = self._add_hero_item(item_name)

        self.last_ruin_reward = {
            "enemy_id": ruin_enemy_id,
            "title": ruin_title,
            "item": item_name,
            "gold": gold_reward,
            "ruin_position": ruin_pos,
            "marker_position": marker_pos,
        }

        ruin_label = ruin_title or f"Руины {ruin_pos}"
        reward_parts = [f"+{gold_reward:g} gold"]
        if item_name:
            reward_parts.append(item_name)
        self._log(f"{ruin_label}: получено {', '.join(reward_parts)}")

        return {
            "ruin_cleared": True,
            "ruin_reward_applied": True,
            "ruin_reward_enemy_id": ruin_enemy_id,
            "ruin_reward_title": ruin_title,
            "ruin_reward_item": item_name,
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
        try:
            normalized_level = int(current_level)
        except (TypeError, ValueError):
            normalized_level = 0
        return float(cls.SETTLEMENT_UPGRADE_GOLD_COST_BY_LEVEL.get(normalized_level, 0.0))
    def _can_upgrade_settlement(self, settlement_name: str) -> bool:
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
        return len(self.captured_objective_cities) == len(self.FINAL_OBJECTIVE_CITIES)
    def _is_legions_settlement_territory_cleared(self, settlement_name: str) -> bool:
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
    def _legions_gold_income_per_turn(self) -> float:
        return float(self.GOLD_PER_TURN) + float(self.legions_captured_gold_mine_count) * float(
            self.LEGIONS_GOLD_MINE_GOLD_PER_TURN
        )
    def _advance_turns(self, delta_turns: int) -> float:
        """Увеличивает счётчик ходов и золото на заданное количество."""
        if delta_turns <= 0:
            return 0.0
        pending_hire_penalty = self._pending_hire_turn_penalty(
            delta_turns,
            units=self.blue_team_state,
        )
        total_gold_income = 0.0
        scripted_bot_turn_infos = []
        for _ in range(int(delta_turns)):
            self.turns += 1
            self._refresh_faction_territories()
            self._clear_all_enemy_map_spell_effects()
            self._clear_all_blue_map_spell_effects()
            total_gold_income += self._legions_gold_income_per_turn()
            self._apply_mana_income_for_turn()
            self._heal_wounded_enemy_teams_for_turn()
            advance_bot = getattr(self, "_advance_scripted_capital_bot_one_turn", None)
            if callable(advance_bot):
                scripted_bot_turn_infos.append(advance_bot())
        self._clear_expired_combat_potion_effects()
        self.combat_potion_battle_bonus_pending = False
        self.gold += float(total_gold_income)
        if scripted_bot_turn_infos:
            self.scripted_capital_bot_turn_infos = scripted_bot_turn_infos
        self.moves = self.moves_per_turn
        self.active_buildings["alredybuilt"] = 0
        self.spell_learning_locked = False
        self.battle_item_equip_used_this_turn = False
        self.spell_cast_counts_by_id_this_turn = {}
        self.summon_hero_battle_bonus_pending = False
        self.summon_hero_battle_bonus_enemy_ids_this_turn = set()
        return -float(delta_turns) * self.reward_turn_penalty - float(pending_hire_penalty)
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
        forbidden_tiles = set(tuple(tile) for tile in self._static_obstacle_tiles)

        base_forbidden_tiles = _load_legions_territory_base_forbidden_tiles()
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
        return self._build_territory_forbidden_tiles(
            source_tile=self.legions_territory_source_tile,
            additional_claimable_tiles=(
                self.legions_territory_source_tile,
                self.empire_territory_source_tile,
            ),
        )
    def _build_empire_territory_forbidden_tiles(self) -> Tuple[Tuple[int, int], ...]:
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
        blocked_tiles = set(tuple(tile) for tile in self._static_obstacle_tiles)
        base_water_tiles = _load_legions_territory_base_water_tiles()
        if base_water_tiles:
            blocked_tiles.update(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_water_tiles)
            )
        blocked_tiles.discard(tuple(source_tile))
        return tuple(sorted(blocked_tiles))
    def _build_legions_territory_path_blocked_tiles(self) -> Tuple[Tuple[int, int], ...]:
        return self._build_territory_path_blocked_tiles(
            source_tile=self.legions_territory_source_tile,
        )
    def _build_empire_territory_path_blocked_tiles(self) -> Tuple[Tuple[int, int], ...]:
        return self._build_territory_path_blocked_tiles(
            source_tile=self.empire_territory_source_tile,
        )
    def _build_territory_path_distances(
        self,
        *,
        source_tile: Tuple[int, int],
        blocked_tile_set: set[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], int]:
        source = (int(source_tile[0]), int(source_tile[1]))
        if source in blocked_tile_set:
            return {}

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
                if tile in blocked_tile_set or tile in distances:
                    continue
                distances[tile] = current_distance + 1
                frontier.append(tile)

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
        return self._build_territory_order(
            forbidden_tile_set=self.legions_territory_forbidden_tile_set,
            path_distances=self.legions_territory_path_distances,
            sort_key=self._legions_territory_sort_key,
        )
    def _build_legions_settlement_territory_orders_by_name(self) -> Dict[str, Tuple[Tuple[int, int], ...]]:
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
        return self._build_territory_order(
            forbidden_tile_set=self.empire_territory_forbidden_tile_set,
            path_distances=self.empire_territory_path_distances,
            sort_key=self._empire_territory_sort_key,
        )
    @staticmethod
    def _territory_claim_count(total_tiles: int, expansion_per_turn: int, turns: int) -> int:
        return min(
            int(total_tiles),
            1 + max(0, int(turns)) * int(expansion_per_turn),
        )
    @staticmethod
    def _claim_territory_tiles(
        order: Tuple[Tuple[int, int], ...],
        claim_count: int,
    ) -> Tuple[Tuple[int, int], ...]:
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
        if activation_turn is None:
            return 0
        elapsed_turns = max(0, int(turns) - int(activation_turn))
        return min(
            int(total_tiles),
            1 + elapsed_turns * max(0, int(expansion_per_turn)),
        )
    def _refresh_legions_territory_tiles(self) -> None:
        self._refresh_faction_territories()
    def _refresh_legions_territory_grid_obs_cache(self) -> None:
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
        for tile in capital_legions_tiles:
            normalized_tile = (int(tile[0]), int(tile[1]))
            if normalized_tile in combined_legions_seen:
                continue
            combined_legions_seen.add(normalized_tile)
            combined_legions_tiles.append(normalized_tile)

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
        self.legions_captured_gold_mine_tiles = tuple(
            tile
            for tile in self.gold_mine_tiles
            if tuple(tile) in self.legions_territory_tile_set
        )
        self.legions_captured_gold_mine_count = len(self.legions_captured_gold_mine_tiles)
        self._refresh_legions_captured_mana_source_state()
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
        if hasattr(self, "grid_legions_territory_positions"):
            self._refresh_legions_territory_grid_obs_cache()
