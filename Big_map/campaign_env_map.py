# campaign_env_map.py
"""CampaignMapSitesMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignMapSitesMixin:
    @classmethod
    def _mercenary_stand_for_unit_data(cls, unit_data: Optional[Dict]) -> Optional[str]:
        if not isinstance(unit_data, dict):
            return None
        if cls._unit_data_is_big(unit_data):
            return None

        unit_type = str(unit_data.get("тип", unit_data.get("unit_type", "")) or "").strip()
        if unit_type in cls.MERCENARY_FRONTLINE_UNIT_TYPES:
            return "ahead"
        if unit_type in cls.MERCENARY_BACKLINE_UNIT_TYPES:
            return "behind"

        attack_type = str(
            unit_data.get("тип атаки1", unit_data.get("attack_type_primary", "")) or ""
        ).strip().lower()
        return "ahead" if attack_type == "weapon" else "behind"
    def _mercenary_option(self, action_idx: int) -> Optional[Dict[str, object]]:
        if not (0 <= int(action_idx) < len(self.active_mercenary_hire_options)):
            return None

        option = dict(self.active_mercenary_hire_options[int(action_idx)])
        roster_entry = self._mercenary_roster_entry(
            str(option.get("site_name", "") or ""),
            int(option.get("slot", 0) or 0),
        )
        option["stock"] = int(roster_entry.get("stock", 0) or 0) if roster_entry else 0
        if roster_entry is not None:
            option["stand"] = str(roster_entry.get("stand", option.get("stand", "")) or "")
            option["unit_type"] = str(
                roster_entry.get("unit_type", option.get("unit_type", "")) or ""
            )
        return option
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

        hero["needaunit"] = self._resolve_hero_needaunit(hero)
        abilities: List[str] = []
        if self._hero_has_pathfinding(units=[hero]):
            abilities.append("Нахождение пути (+20% шагов)")
        leadership_count = self._hero_leadership_count(units=[hero])
        if leadership_count == 1:
            abilities.append("Лидерство")
        elif leadership_count >= 2:
            abilities.append("Лидерство x2")
        if self._hero_has_endurance(units=[hero]):
            abilities.append("Выносливость")
        if self._hero_has_strength(units=[hero]):
            abilities.append("Сила (+25% основной урон)")

        return {
            "name": str(hero.get("name", "") or "").strip(),
            "level": self._hero_level(units=[hero]),
            "abilities": abilities,
        }
    def _sync_grid_chest_positions(self) -> None:
        self.grid_env.chest_positions = set(self.chests.keys())
    def _sync_grid_mana_sources(self) -> None:
        self.grid_env.mana_sources = {
            tuple(pos): dict(meta)
            for pos, meta in getattr(self, "mana_sources", {}).items()
        }
    def _sync_grid_merchant_positions(self) -> None:
        self.grid_env.merchant_positions = set(self.merchant_interaction_tiles)
    def _sync_grid_spell_shop_positions(self) -> None:
        self.grid_env.spell_shop_positions = set(self.spell_shop_interaction_tiles)
    def _sync_grid_mercenary_positions(self) -> None:
        self.grid_env.mercenary_positions = set(self.mercenary_interaction_tiles)
    def _sync_grid_trainer_positions(self) -> None:
        self.grid_env.trainer_positions = set(self.trainer_interaction_tiles)
    def _apply_chest_item_rewards(self, item_name: str) -> List[str]:
        granted_items: List[str] = []
        if item_name == self.BONUS_SMALL_HEAL_SOURCE_ITEM:
            self.extra_healing_bottles = max(0, int(self.extra_healing_bottles or 0)) + 1
            granted_items.append(self.BONUS_SMALL_HEAL_ITEM_NAME)
        if item_name == self.BONUS_REVIVE_SOURCE_ITEM:
            self.extra_revive_bottles = max(0, int(self.extra_revive_bottles or 0)) + 1
            granted_items.append(self.BONUS_REVIVE_ITEM_NAME)
        return granted_items
    @classmethod
    def _is_auto_consumed_chest_item(cls, item_name: str) -> bool:
        return str(item_name or "") in cls.AUTO_CONSUMED_CHEST_ITEMS
    def _collect_adjacent_chests(self) -> List[Dict]:
        if not self.chests:
            return []

        agent_pos = tuple(self.grid_env.agent_pos)
        collected: List[Dict] = []
        for chest_pos, item_names in sorted(self.chests.items(), key=lambda entry: entry[0]):
            if not self._is_within_chest_pickup_range(agent_pos, chest_pos):
                continue
            loot_items = [str(item_name) for item_name in item_names if str(item_name)]
            collected.append(
                {
                    "pos": tuple(chest_pos),
                    "items": loot_items,
                    "granted_items": [],
                }
            )

        if not collected:
            return []

        for chest_data in collected:
            chest_pos = tuple(chest_data["pos"])
            self.chests.pop(chest_pos, None)
            granted_items: List[str] = []
            for item_name in chest_data["items"]:
                if not self._is_auto_consumed_chest_item(item_name):
                    self._add_hero_item(item_name)
                granted_items.extend(self._apply_chest_item_rewards(item_name))
                self._log(f"Сундук на {chest_pos}: получен предмет '{item_name}'")
            for granted_item in granted_items:
                self._add_hero_item(granted_item)
                self._log(
                    f"Сундук на {chest_pos}: бонусом получен предмет '{granted_item}'"
                )
            chest_data["granted_items"] = granted_items

        self._sync_grid_chest_positions()
        return collected
    def _spell_shop_cast_limit_reached_this_turn(self, spell_key: str) -> bool:
        return self._spell_cast_count_this_turn(spell_key) >= 1
    def _spell_shop_spell_has_regular_action(self, spell_key: str) -> bool:
        return str(spell_key or "") in set(self._map_castable_spell_ids_for_capital(self.Realcapital))
    def _spell_shop_cast_spell_entries(self) -> Tuple[Dict[str, object], ...]:
        entries: List[Dict[str, object]] = []
        seen_spell_ids: set[str] = set()
        for spell_data in self.SPELL_SHOP_BUY_SPELLS:
            spell_id = str(spell_data.get("spell_id", "") or "")
            if (
                not spell_id
                or spell_id in seen_spell_ids
                or self._spell_shop_spell_has_regular_action(spell_id)
            ):
                continue
            spell_spec = self._map_offensive_spell_spec(spell_id)
            if not spell_spec:
                spell_spec = self._map_support_spell_spec(spell_id)
            if not spell_spec or not isinstance(self._spell_entry_by_id(spell_id), dict):
                continue
            entries.append(dict(spell_data))
            seen_spell_ids.add(spell_id)
            if len(entries) >= int(self.MAX_SPELL_SHOP_CAST_ACTIONS):
                break
        return tuple(entries)
    def _spell_shop_cast_spell_entry(self, idx: int) -> Dict[str, object]:
        entries = self._spell_shop_cast_spell_entries()
        if 0 <= int(idx) < len(entries):
            return dict(entries[int(idx)])
        return {}
    def _grant_merchant_item(self, item_name: str) -> List[str]:
        granted_items: List[str] = []
        item_data = self._merchant_item_definition(item_name)
        grant_kind = str((item_data or {}).get("grant", "") or "")

        if grant_kind == "small_heal_bonus":
            self.extra_healing_bottles = max(0, int(self.extra_healing_bottles or 0)) + 1
            granted_items.append(self.BONUS_SMALL_HEAL_ITEM_NAME)
        elif grant_kind == "large_heal_bonus":
            self.extra_heal_bottles = max(0, int(self.extra_heal_bottles or 0)) + 1
            granted_items.append(self.BONUS_LARGE_HEAL_ITEM_NAME)
        elif grant_kind == "revive_bonus":
            self.extra_revive_bottles = max(0, int(self.extra_revive_bottles or 0)) + 1
            granted_items.append(self.BONUS_REVIVE_ITEM_NAME)
        else:
            granted_items.append(str(item_name or ""))

        for granted_item in granted_items:
            self._add_hero_item(granted_item)
        return granted_items
    def _step_buy_merchant_item(self, action: int):
        idx = action - self.GRID_MERCHANT_BUY_ACTION_START
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        item_data = self.MERCHANT_BUY_ITEMS[idx] if 0 <= idx < len(self.MERCHANT_BUY_ITEMS) else {}
        item_name = str(item_data.get("name", "") or "")
        item_price = float(item_data.get("price", 0.0) or 0.0)
        site_names = self._merchant_sites_at_position(self.grid_env.agent_pos)
        site_name = self._merchant_site_for_item(item_name, position=self.grid_env.agent_pos)
        stock_before = self._merchant_item_stock(site_name, item_name) if site_name else 0

        purchased = False
        insufficient_gold = False
        not_at_merchant = not bool(site_names)
        out_of_stock = bool(site_names) and not bool(site_name)
        granted_items: List[str] = []

        if site_name is not None and item_name:
            if float(self.gold or 0.0) < item_price:
                insufficient_gold = True
            elif stock_before > 0:
                self.gold = max(0.0, float(self.gold or 0.0) - item_price)
                self.merchant_stocks[site_name][item_name] = max(0, stock_before - 1)
                granted_items = self._grant_merchant_item(item_name)
                purchased = True
                self._log(
                    f"Торговец {site_name}: куплен предмет '{item_name}' за {item_price:.1f} gold"
                )

        stock_after = self._merchant_item_stock(site_name, item_name) if site_name else 0
        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "merchant_buy_action": True,
            "merchant_buy_item": item_name,
            "merchant_buy_price": item_price,
            "merchant_buy_site": site_name,
            "merchant_buy_purchased": purchased,
            "merchant_buy_stock_before": int(stock_before),
            "merchant_buy_stock_after": int(stock_after),
            "merchant_buy_granted_items": list(granted_items),
            "merchant_buy_not_at_merchant": not_at_merchant,
            "merchant_buy_out_of_stock": out_of_stock,
            "merchant_buy_insufficient_gold": insufficient_gold,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_buy_spell_shop_spell(self, action: int):
        idx = action - self.GRID_SPELL_SHOP_BUY_ACTION_START
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        spell_data = (
            self.SPELL_SHOP_BUY_SPELLS[idx] if 0 <= idx < len(self.SPELL_SHOP_BUY_SPELLS) else {}
        )
        spell_name = str(spell_data.get("name", "") or "")
        spell_id = str(spell_data.get("spell_id", "") or "")
        spell_price = float(spell_data.get("price", 0.0) or 0.0)
        site_names = self._spell_shop_sites_at_position(self.grid_env.agent_pos)
        site_name = self._spell_shop_site_for_spell(spell_name, position=self.grid_env.agent_pos)
        stock_before = self._spell_shop_spell_stock(site_name, spell_name) if site_name else 0

        purchased = False
        insufficient_gold = False
        already_owned = self._spell_shop_spell_owned(spell_id)
        not_at_spell_shop = not bool(site_names)
        out_of_stock = bool(site_names) and not bool(site_name)
        added_to_spellbook = False

        if site_name is not None and spell_name:
            if already_owned:
                pass
            elif float(self.gold or 0.0) < spell_price:
                insufficient_gold = True
            elif stock_before > 0:
                self.gold = max(0.0, float(self.gold or 0.0) - spell_price)
                self.spell_shop_stocks[site_name][spell_name] = max(0, stock_before - 1)
                self.spell_shop_purchased_spell_ids.add(spell_id)
                spell_entry = self.active_spells.get(spell_id)
                if isinstance(spell_entry, dict):
                    spell_entry["learned"] = 1
                    added_to_spellbook = True
                purchased = True
                self._log(
                    f"Лавка заклинаний {site_name}: куплено заклинание '{spell_name}' за {spell_price:.1f} gold"
                )

        stock_after = self._spell_shop_spell_stock(site_name, spell_name) if site_name else 0
        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "spell_shop_buy_action": True,
            "spell_shop_buy_spell": spell_name,
            "spell_shop_buy_spell_id": spell_id,
            "spell_shop_buy_price": spell_price,
            "spell_shop_buy_site": site_name,
            "spell_shop_buy_purchased": purchased,
            "spell_shop_buy_added_to_spellbook": added_to_spellbook,
            "spell_shop_buy_already_owned": already_owned,
            "spell_shop_buy_stock_before": int(stock_before),
            "spell_shop_buy_stock_after": int(stock_after),
            "spell_shop_buy_not_at_spell_shop": not_at_spell_shop,
            "spell_shop_buy_out_of_stock": out_of_stock,
            "spell_shop_buy_insufficient_gold": insufficient_gold,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_hire_mercenary_unit(self, action: int):
        idx = action - self.GRID_MERCENARY_HIRE_ACTION_START
        option = self._mercenary_option(idx)

        grid_obs = self._get_grid_obs()
        terminated = False
        truncated = False

        site_names = self._mercenary_sites_at_position(self.grid_env.agent_pos)
        site_name = None if option is None else str(option.get("site_name", "") or "").strip()
        site_id = None if option is None else str(option.get("site_id", "") or "").strip()
        unit_name = None if option is None else str(
            option.get("unit_name", option.get("name", "")) or ""
        ).strip()
        unit_type = None if option is None else str(option.get("unit_type", "") or "").strip()
        hire_cost = 0.0
        hire_stand = None if option is None else str(option.get("stand", "") or "").strip().lower()
        hire_target_pos = None
        stock_before = 0
        if option is not None:
            try:
                hire_cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
            except (TypeError, ValueError):
                hire_cost = 0.0
            stock_before = max(0, int(option.get("stock", 0) or 0))
            positions = self._hire_positions_for_stand(hire_stand)
            if positions:
                hire_target_pos = self._find_first_empty_blue_position(positions)

        hire_available_before = self._can_hire_mercenary_unit_option(option)
        hired, hired_unit_name, hired_position, gold_spent, hired_site_name, stock_before_used, stock_after = (
            self._hire_mercenary_unit_option(option)
        )

        hero = self._resolve_travel_hero()
        hero_needaunit = self._resolve_hero_needaunit(hero) if hero is not None else 0
        reward = self._hire_reward_value() if hired else 0.0
        matching_site_in_range = bool(site_name) and site_name in site_names
        insufficient_gold = matching_site_in_range and bool(unit_name) and float(self.gold or 0.0) < hire_cost
        out_of_stock = matching_site_in_range and stock_before <= 0

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "hire_action": True,
            "hire_action_kind": "mercenary",
            "hire_action_idx": int(idx),
            "hire_unit_name": unit_name,
            "hire_role": hire_stand,
            "hire_cost": float(hire_cost),
            "hire_target_pos": hire_target_pos,
            "hire_available": bool(hire_available_before),
            "hired": bool(hired),
            "hired_units_count": int(1 if hired else 0),
            "hired_unit_name": hired_unit_name,
            "hired_position": hired_position,
            "gold_spent": float(gold_spent),
            "hire_reward": float(reward),
            "hero_needaunit": int(hero_needaunit),
            "mercenary_hire_action": True,
            "mercenary_hire_action_idx": int(idx),
            "mercenary_hire_site": hired_site_name or site_name,
            "mercenary_hire_site_id": site_id,
            "mercenary_hire_unit_name": unit_name,
            "mercenary_hire_unit_type": unit_type,
            "mercenary_hire_stand": hire_stand,
            "mercenary_hire_cost": float(hire_cost),
            "mercenary_hire_stock_before": int(stock_before_used if hired else stock_before),
            "mercenary_hire_stock_after": int(stock_after),
            "mercenary_hire_available": bool(hire_available_before),
            "mercenary_hire_not_at_camp": not matching_site_in_range,
            "mercenary_hire_out_of_stock": bool(out_of_stock),
            "mercenary_hire_insufficient_gold": bool(
                matching_site_in_range and not hired and bool(unit_name) and float(self.gold or 0.0) < hire_cost
            ),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_train_unit_at_trainer(self, action: int):
        idx = action - self.GRID_TRAINER_ACTION_START
        target_pos = (
            self.TRAINER_POSITIONS[idx] if 0 <= idx < len(self.TRAINER_POSITIONS) else None
        )
        terminated = False
        truncated = False
        reward = 0.0

        site_names = self._trainer_sites_at_position(self.grid_env.agent_pos)
        site_name = site_names[0] if site_names else None
        preview_before = (
            self._trainer_preview_for_position(int(target_pos))
            if target_pos is not None
            else {
                "site_names": list(site_names),
                "unit": None,
                "unit_name": None,
                "trainable": False,
                "missing_unit": True,
                "dead_or_empty": False,
                "already_capped": False,
                "insufficient_gold": False,
                "exp_cap": None,
                "exp_before": 0,
                "xp_missing": 0,
                "gold_per_xp": 0.0,
                "gold_needed_full": 0.0,
                "affordable_xp": 0,
                "gold_spent": 0.0,
                "exp_after": 0,
            }
        )

        trained = False
        xp_gained = 0
        gold_spent = 0.0
        exp_after = int(preview_before.get("exp_before", 0) or 0)
        unit_name = str(preview_before.get("unit_name") or "").strip() or None
        unit = preview_before.get("unit")

        if target_pos is not None and bool(preview_before.get("trainable", False)) and isinstance(unit, dict):
            xp_gained = max(0, int(preview_before.get("affordable_xp", 0) or 0))
            gold_spent = max(0.0, float(preview_before.get("gold_spent", 0.0) or 0.0))
            exp_cap = preview_before.get("exp_cap")
            exp_before = max(0, int(preview_before.get("exp_before", 0) or 0))
            if exp_cap is not None and xp_gained > 0:
                exp_after = min(int(exp_cap), exp_before + xp_gained)
                unit["exp_current"] = int(exp_after)
                self.gold = max(0.0, float(self.gold or 0.0) - gold_spent)
                trained = True
                self._sync_hero_progression_flags(self.blue_team_state)
                self._log(
                    f"Тренировочный лагерь: {unit.get('name', 'unit')}#{int(target_pos)} "
                    f"+{xp_gained} XP за {gold_spent:.1f} золота"
                )

        preview_after = (
            self._trainer_preview_for_position(int(target_pos))
            if target_pos is not None
            else preview_before
        )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "trainer_action": True,
            "trainer_action_idx": int(idx),
            "trainer_site": site_name,
            "trainer_position": target_pos,
            "trainer_unit_name": unit_name,
            "trainer_trainable": bool(preview_before.get("trainable", False)),
            "trainer_not_at_camp": not bool(site_names),
            "trainer_missing_unit": bool(preview_before.get("missing_unit", False)),
            "trainer_dead_or_empty": bool(preview_before.get("dead_or_empty", False)),
            "trainer_already_capped": bool(preview_before.get("already_capped", False)),
            "trainer_insufficient_gold": bool(preview_before.get("insufficient_gold", False)),
            "trainer_gold_per_xp": float(preview_before.get("gold_per_xp", 0.0) or 0.0),
            "trainer_gold_needed_full": float(preview_before.get("gold_needed_full", 0.0) or 0.0),
            "trainer_gold_spent": float(gold_spent),
            "trainer_exp_before": int(preview_before.get("exp_before", 0) or 0),
            "trainer_exp_after": int(exp_after),
            "trainer_exp_cap": preview_before.get("exp_cap"),
            "trainer_xp_missing_before": int(preview_before.get("xp_missing", 0) or 0),
            "trainer_xp_gained": int(xp_gained),
            "trainer_xp_affordable_before": int(preview_before.get("affordable_xp", 0) or 0),
            "trainer_xp_missing_after": int(preview_after.get("xp_missing", 0) or 0),
            "trainer_trained": bool(trained),
            "gold_spent": float(gold_spent),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        grid_obs = self._get_grid_obs()

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _init_merchant_site_metadata(self) -> None:
        obstacle_tiles = set(self._static_obstacle_tiles)
        self.merchant_site_anchors: Dict[str, Tuple[int, int]] = {}
        self.merchant_site_interaction_tiles: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        all_tiles: List[Tuple[int, int]] = []
        seen_tiles: set[Tuple[int, int]] = set()

        for site_name, site_data in self.BASE_MERCHANT_SITE_DATA.items():
            anchor = scale_static_tiles(
                self.grid_size,
                (tuple(site_data["anchor"]),),
            )[0]
            scaled_tiles = scale_static_tiles(
                self.grid_size,
                tuple(site_data["interaction_tiles"]),
            )

            site_tiles: List[Tuple[int, int]] = []
            site_seen: set[Tuple[int, int]] = set()
            for tile in scaled_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in obstacle_tiles or normalized_tile in site_seen:
                    continue
                site_seen.add(normalized_tile)
                site_tiles.append(normalized_tile)
                if normalized_tile not in seen_tiles:
                    seen_tiles.add(normalized_tile)
                    all_tiles.append(normalized_tile)

            self.merchant_site_anchors[str(site_name)] = (int(anchor[0]), int(anchor[1]))
            self.merchant_site_interaction_tiles[str(site_name)] = tuple(site_tiles)

        self.merchant_interaction_tiles = tuple(all_tiles)
    def _init_spell_shop_site_metadata(self) -> None:
        obstacle_tiles = set(self._static_obstacle_tiles)
        self.spell_shop_site_anchors: Dict[str, Tuple[int, int]] = {}
        self.spell_shop_site_interaction_tiles: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        all_tiles: List[Tuple[int, int]] = []
        seen_tiles: set[Tuple[int, int]] = set()

        for site_name, site_data in self.BASE_SPELL_SHOP_SITE_DATA.items():
            anchor = scale_static_tiles(
                self.grid_size,
                (tuple(site_data["anchor"]),),
            )[0]
            scaled_tiles = scale_static_tiles(
                self.grid_size,
                tuple(site_data["interaction_tiles"]),
            )

            site_tiles: List[Tuple[int, int]] = []
            site_seen: set[Tuple[int, int]] = set()
            for tile in scaled_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in obstacle_tiles or normalized_tile in site_seen:
                    continue
                site_seen.add(normalized_tile)
                site_tiles.append(normalized_tile)
                if normalized_tile not in seen_tiles:
                    seen_tiles.add(normalized_tile)
                    all_tiles.append(normalized_tile)

            self.spell_shop_site_anchors[str(site_name)] = (int(anchor[0]), int(anchor[1]))
            self.spell_shop_site_interaction_tiles[str(site_name)] = tuple(site_tiles)

        self.spell_shop_interaction_tiles = tuple(all_tiles)
    def _init_mercenary_site_metadata(self) -> None:
        obstacle_tiles = set(self._static_obstacle_tiles)
        self.mercenary_site_anchors: Dict[str, Tuple[int, int]] = {}
        self.mercenary_site_interaction_tiles: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        all_tiles: List[Tuple[int, int]] = []
        seen_tiles: set[Tuple[int, int]] = set()

        for site_name, site_data in self.BASE_MERCENARY_SITE_DATA.items():
            anchor = scale_static_tiles(
                self.grid_size,
                (tuple(site_data["anchor"]),),
            )[0]
            scaled_tiles = scale_static_tiles(
                self.grid_size,
                tuple(site_data["interaction_tiles"]),
            )

            site_tiles: List[Tuple[int, int]] = []
            site_seen: set[Tuple[int, int]] = set()
            for tile in scaled_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in obstacle_tiles or normalized_tile in site_seen:
                    continue
                site_seen.add(normalized_tile)
                site_tiles.append(normalized_tile)
                if normalized_tile not in seen_tiles:
                    seen_tiles.add(normalized_tile)
                    all_tiles.append(normalized_tile)

            self.mercenary_site_anchors[str(site_name)] = (int(anchor[0]), int(anchor[1]))
            self.mercenary_site_interaction_tiles[str(site_name)] = tuple(site_tiles)

        self.mercenary_interaction_tiles = tuple(all_tiles)
    def _init_trainer_site_metadata(self) -> None:
        obstacle_tiles = set(self._static_obstacle_tiles)
        self.trainer_site_anchors: Dict[str, Tuple[int, int]] = {}
        self.trainer_site_interaction_tiles: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        all_tiles: List[Tuple[int, int]] = []
        seen_tiles: set[Tuple[int, int]] = set()

        for site_name, site_data in self.BASE_TRAINER_SITE_DATA.items():
            anchor = scale_static_tiles(
                self.grid_size,
                (tuple(site_data["anchor"]),),
            )[0]
            scaled_tiles = scale_static_tiles(
                self.grid_size,
                tuple(site_data["interaction_tiles"]),
            )

            site_tiles: List[Tuple[int, int]] = []
            site_seen: set[Tuple[int, int]] = set()
            for tile in scaled_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in obstacle_tiles or normalized_tile in site_seen:
                    continue
                site_seen.add(normalized_tile)
                site_tiles.append(normalized_tile)
                if normalized_tile not in seen_tiles:
                    seen_tiles.add(normalized_tile)
                    all_tiles.append(normalized_tile)

            self.trainer_site_anchors[str(site_name)] = (int(anchor[0]), int(anchor[1]))
            self.trainer_site_interaction_tiles[str(site_name)] = tuple(site_tiles)

        self.trainer_interaction_tiles = tuple(all_tiles)
    def _create_initial_mercenary_site_rosters(self) -> Dict[str, List[Dict[str, object]]]:
        rosters: Dict[str, List[Dict[str, object]]] = {}
        for site_name, site_data in self.BASE_MERCENARY_SITE_DATA.items():
            site_roster: List[Dict[str, object]] = []
            for idx, entry in enumerate(tuple(site_data.get("roster", ()))):
                if not isinstance(entry, dict):
                    continue
                unit_name = str(entry.get("unit_name", entry.get("name", "")) or "").strip()
                if not unit_name:
                    continue
                unit_data = self._find_unit_data_by_name(unit_name)
                stand = self._mercenary_stand_for_unit_data(unit_data)
                try:
                    gold_cost = max(0.0, float(entry.get("gold", 0.0) or 0.0))
                except (TypeError, ValueError):
                    gold_cost = 0.0
                try:
                    stock = max(0, int(entry.get("stock", 0) or 0))
                except (TypeError, ValueError):
                    stock = 0
                site_roster.append(
                    {
                        "slot": int(entry.get("slot", idx) or idx),
                        "unit_id": str(entry.get("unit_id", "") or ""),
                        "unit_name": unit_name,
                        "gold": float(gold_cost),
                        "stock": int(stock),
                        "initial_stock": int(stock),
                        "stand": "" if stand is None else stand,
                        "unit_type": (
                            str(unit_data.get("тип", unit_data.get("unit_type", "")) or "").strip()
                            if isinstance(unit_data, dict)
                            else ""
                        ),
                        "unit_max_health": (
                            float(unit_data.get("здоровье", unit_data.get("max_health", 0)) or 0.0)
                            if isinstance(unit_data, dict)
                            else 0.0
                        ),
                        "unit_damage": (
                            float(unit_data.get("урон", unit_data.get("damage", 0)) or 0.0)
                            if isinstance(unit_data, dict)
                            else 0.0
                        ),
                        "big": bool(self._unit_data_is_big(unit_data)),
                    }
                )
            rosters[str(site_name)] = site_roster
        return rosters
    def _create_initial_merchant_stocks(self) -> Dict[str, Dict[str, int]]:
        stocks: Dict[str, Dict[str, int]] = {
            str(site_name): {} for site_name in self.BASE_MERCHANT_SITE_DATA.keys()
        }
        alara_stock: Dict[str, int] = {}
        for item_data in self.MERCHANT_BUY_ITEMS:
            item_name = str(item_data.get("name", "") or "")
            if not item_name:
                continue
            alara_stock[item_name] = max(0, int(item_data.get("stock", 0) or 0))
        stocks["Лавка Алара"] = alara_stock
        stocks.setdefault("Лавка Тралара", {})
        return stocks
    @classmethod
    def _merchant_item_definition(cls, item_name: str) -> Optional[Dict[str, object]]:
        normalized_name = str(item_name or "")
        for item_data in cls.MERCHANT_BUY_ITEMS:
            if str(item_data.get("name", "") or "") == normalized_name:
                return dict(item_data)
        return None
    def _merchant_item_stock(self, site_name: str, item_name: str) -> int:
        site_stock = self.merchant_stocks.get(str(site_name), {})
        try:
            return max(0, int(site_stock.get(str(item_name), 0) or 0))
        except (TypeError, ValueError):
            return 0
    def _merchant_site_for_item(
        self,
        item_name: str,
        position: Optional[Tuple[int, int]] = None,
    ) -> Optional[str]:
        for site_name in self._merchant_sites_at_position(position):
            if self._merchant_item_stock(site_name, item_name) > 0:
                return site_name
        return None
    def _merchant_stock_snapshot(self, site_name: str) -> Dict[str, int]:
        snapshot: Dict[str, int] = {}
        for item_data in self.MERCHANT_BUY_ITEMS:
            item_name = str(item_data.get("name", "") or "")
            if not item_name:
                continue
            stock_left = self._merchant_item_stock(site_name, item_name)
            if stock_left > 0:
                snapshot[item_name] = stock_left
        return snapshot
    def _merchant_sites_at_position(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        if position is None:
            position = tuple(self.grid_env.agent_pos)
        normalized_pos = (int(position[0]), int(position[1]))
        return [
            site_name
            for site_name, tiles in self.merchant_site_interaction_tiles.items()
            if normalized_pos in tiles
        ]
    def _merchant_context_info(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        site_names = self._merchant_sites_at_position(position)
        merchant_stock = {
            site_name: self._merchant_stock_snapshot(site_name)
            for site_name in site_names
        }
        return {
            "is_at_merchant": bool(site_names),
            "merchant_sites_in_range": list(site_names),
            "merchant_stock": merchant_stock,
        }
    def _create_initial_spell_shop_stocks(self) -> Dict[str, Dict[str, int]]:
        stocks: Dict[str, Dict[str, int]] = {
            str(site_name): {} for site_name in self.BASE_SPELL_SHOP_SITE_DATA.keys()
        }
        for site_name in stocks.keys():
            site_stock: Dict[str, int] = {}
            for spell_data in self.SPELL_SHOP_BUY_SPELLS:
                spell_name = str(spell_data.get("name", "") or "")
                if not spell_name:
                    continue
                site_stock[spell_name] = max(0, int(spell_data.get("stock", 0) or 0))
            stocks[site_name] = site_stock
        return stocks
    @classmethod
    def _spell_shop_spell_definition(cls, spell_name: str) -> Optional[Dict[str, object]]:
        normalized_name = str(spell_name or "")
        for spell_data in cls.SPELL_SHOP_BUY_SPELLS:
            if str(spell_data.get("name", "") or "") == normalized_name:
                return dict(spell_data)
        return None
    def _spell_shop_spell_stock(self, site_name: str, spell_name: str) -> int:
        site_stock = self.spell_shop_stocks.get(str(site_name), {})
        try:
            return max(0, int(site_stock.get(str(spell_name), 0) or 0))
        except (TypeError, ValueError):
            return 0
    def _spell_shop_spell_owned(self, spell_id: str) -> bool:
        normalized_spell_id = str(spell_id or "")
        if not normalized_spell_id:
            return False
        if normalized_spell_id in self.spell_shop_purchased_spell_ids:
            return True
        return self._is_spell_learned(normalized_spell_id)
    def _spell_shop_site_for_spell(
        self,
        spell_name: str,
        position: Optional[Tuple[int, int]] = None,
    ) -> Optional[str]:
        for site_name in self._spell_shop_sites_at_position(position):
            if self._spell_shop_spell_stock(site_name, spell_name) > 0:
                return site_name
        return None
    def _spell_shop_stock_snapshot(self, site_name: str) -> Dict[str, int]:
        snapshot: Dict[str, int] = {}
        for spell_data in self.SPELL_SHOP_BUY_SPELLS:
            spell_name = str(spell_data.get("name", "") or "")
            if not spell_name:
                continue
            stock_left = self._spell_shop_spell_stock(site_name, spell_name)
            if stock_left > 0:
                snapshot[spell_name] = stock_left
        return snapshot
    def _spell_shop_sites_at_position(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        if position is None:
            position = tuple(self.grid_env.agent_pos)
        normalized_pos = (int(position[0]), int(position[1]))
        return [
            site_name
            for site_name, tiles in self.spell_shop_site_interaction_tiles.items()
            if normalized_pos in tiles
        ]
    def _spell_shop_context_info(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        site_names = self._spell_shop_sites_at_position(position)
        spell_shop_stock = {
            site_name: self._spell_shop_stock_snapshot(site_name)
            for site_name in site_names
        }
        purchased_spell_names = [
            str(spell_data.get("name", "") or "")
            for spell_data in self.SPELL_SHOP_BUY_SPELLS
            if str(spell_data.get("spell_id", "") or "") in self.spell_shop_purchased_spell_ids
        ]
        return {
            "is_at_spell_shop": bool(site_names),
            "spell_shop_sites_in_range": list(site_names),
            "spell_shop_stock": spell_shop_stock,
            "spell_shop_purchased_spells": purchased_spell_names,
        }
    def _mercenary_roster_entry(
        self,
        site_name: str,
        slot: int,
    ) -> Optional[Dict[str, object]]:
        for entry in self.mercenary_site_rosters.get(str(site_name), []):
            raw_slot = entry.get("slot", -1)
            try:
                entry_slot = int(-1 if raw_slot is None else raw_slot)
            except (TypeError, ValueError):
                entry_slot = -1
            if entry_slot == int(slot):
                return entry
        return None
    def _mercenary_stock_snapshot(self, site_name: str) -> List[Dict[str, object]]:
        snapshot: List[Dict[str, object]] = []
        for entry in self.mercenary_site_rosters.get(str(site_name), []):
            snapshot.append(
                {
                    "slot": int(entry.get("slot", 0) or 0),
                    "unit_id": str(entry.get("unit_id", "") or ""),
                    "unit_name": str(entry.get("unit_name", "") or ""),
                    "gold": float(entry.get("gold", 0.0) or 0.0),
                    "stock": max(0, int(entry.get("stock", 0) or 0)),
                    "stand": str(entry.get("stand", "") or ""),
                    "unit_type": str(entry.get("unit_type", "") or ""),
                    "big": bool(entry.get("big", False)),
                }
            )
        return snapshot
    def _mercenary_sites_at_position(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        if position is None:
            position = tuple(self.grid_env.agent_pos)
        normalized_pos = (int(position[0]), int(position[1]))
        return [
            site_name
            for site_name, tiles in self.mercenary_site_interaction_tiles.items()
            if normalized_pos in tiles
        ]
    def _mercenary_context_info(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        site_names = self._mercenary_sites_at_position(position)
        mercenary_stock = {
            site_name: self._mercenary_stock_snapshot(site_name)
            for site_name in site_names
        }
        return {
            "is_at_mercenary_camp": bool(site_names),
            "mercenary_sites_in_range": list(site_names),
            "mercenary_stock": mercenary_stock,
        }
    def _trainer_sites_at_position(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        if position is None:
            position = tuple(self.grid_env.agent_pos)
        normalized_pos = (int(position[0]), int(position[1]))
        return [
            site_name
            for site_name, tiles in self.trainer_site_interaction_tiles.items()
            if normalized_pos in tiles
        ]
    def _trainer_context_info(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        site_names = self._trainer_sites_at_position(position)
        return {
            "is_at_trainer": bool(site_names),
            "trainer_sites_in_range": list(site_names),
        }
    def _trainer_preview_for_position(self, position: int) -> Dict[str, object]:
        normalized_position = int(position)
        preview: Dict[str, object] = {
            "position": normalized_position,
            "site_names": list(self._trainer_sites_at_position(self.grid_env.agent_pos)),
            "unit": None,
            "unit_name": None,
            "trainable": False,
            "missing_unit": True,
            "dead_or_empty": False,
            "already_capped": False,
            "insufficient_gold": False,
            "exp_cap": None,
            "exp_before": 0,
            "xp_missing": 0,
            "gold_per_xp": 0.0,
            "gold_needed_full": 0.0,
            "affordable_xp": 0,
            "gold_spent": 0.0,
            "exp_after": 0,
        }

        for unit in self._get_blue_state():
            if int(unit.get("position", -1) or -1) != normalized_position:
                continue
            preview["unit"] = unit
            preview["unit_name"] = str(unit.get("name", "") or "").strip()
            preview["missing_unit"] = False

            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0.0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
            if self._is_empty_blue_unit(unit) or max_hp <= 0.0 or hp <= 0.0:
                preview["dead_or_empty"] = True
                return preview

            exp_cap = self._trainer_exp_cap(unit)
            preview["exp_cap"] = exp_cap
            if exp_cap is None:
                preview["already_capped"] = True
                return preview

            try:
                exp_before = int(round(float(unit.get("exp_current", 0) or 0)))
            except (TypeError, ValueError):
                exp_before = 0
            exp_before = max(0, exp_before)
            preview["exp_before"] = exp_before
            xp_missing = max(0, int(exp_cap) - exp_before)
            preview["xp_missing"] = xp_missing
            if xp_missing <= 0:
                preview["already_capped"] = True
                preview["exp_after"] = exp_before
                return preview

            gold_per_xp = float(self._trainer_gold_per_xp(unit))
            preview["gold_per_xp"] = gold_per_xp
            preview["gold_needed_full"] = float(xp_missing) * gold_per_xp

            affordable_xp = 0
            if gold_per_xp > 0.0:
                affordable_xp = min(
                    xp_missing,
                    max(0, int(float(self.gold or 0.0) // gold_per_xp)),
                )
            preview["affordable_xp"] = int(affordable_xp)
            preview["gold_spent"] = float(affordable_xp) * gold_per_xp
            preview["exp_after"] = exp_before + int(affordable_xp)
            preview["insufficient_gold"] = affordable_xp <= 0
            preview["trainable"] = bool(affordable_xp > 0)
            return preview

        return preview
    def _can_train_unit_position(self, position: int) -> bool:
        if not self._trainer_sites_at_position(self.grid_env.agent_pos):
            return False
        preview = self._trainer_preview_for_position(position)
        return bool(preview.get("trainable", False))
    @staticmethod
    def _is_empty_enemy_unit(unit: Dict) -> bool:
        """True если слот вражеского отряда пустой."""
        name = (unit.get("name") or "").strip().lower()
        if name == "пусто":
            return True
        hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0)
        max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0)
        return hp <= 0 and max_hp <= 0
