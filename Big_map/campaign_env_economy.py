# campaign_env_economy.py
"""CampaignEconomyMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignEconomyMixin:
    @staticmethod
    def _castle_heal_gold_per_hp(unit: Dict) -> float:
        """Стоимость лечения на клетке лечения.

        Фракционные юниты лечатся по Level, а нейтралы — по bucket'ам max HP.
        """
        is_neutral_unit = bool(unit.get("is_neutral_unit", False))
        if not is_neutral_unit:
            capital_raw = unit.get("capital", None)
            try:
                is_neutral_unit = capital_raw is not None and int(round(float(capital_raw))) == 0
            except (TypeError, ValueError):
                is_neutral_unit = False

        if is_neutral_unit:
            max_hp_raw = unit.get("maxhp", unit.get("max_health", 0))
            try:
                max_hp = float(max_hp_raw or 0.0)
            except (TypeError, ValueError):
                max_hp = 0.0

            if max_hp <= 100.0:
                return 1.0
            if max_hp <= 250.0:
                return 2.0
            if max_hp <= 500.0:
                return 3.0
            return 4.0

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
        """Стоимость воскрешения на клетке лечения.

        Фракционные юниты воскрешаются по Level, а нейтралы — по bucket'ам max HP.
        """
        is_neutral_unit = bool(unit.get("is_neutral_unit", False))
        if not is_neutral_unit:
            capital_raw = unit.get("capital", None)
            try:
                is_neutral_unit = capital_raw is not None and int(round(float(capital_raw))) == 0
            except (TypeError, ValueError):
                is_neutral_unit = False

        if is_neutral_unit:
            max_hp_raw = unit.get("maxhp", unit.get("max_health", 0))
            try:
                max_hp = float(max_hp_raw or 0.0)
            except (TypeError, ValueError):
                max_hp = 0.0

            if max_hp <= 100.0:
                return 50.0
            if max_hp <= 150.0:
                return 200.0
            if max_hp <= 250.0:
                return 400.0
            if max_hp <= 350.0:
                return 600.0
            if max_hp <= 500.0:
                return 800.0
            if max_hp <= 2000.0:
                return 1000.0
            return 1500.0

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
    @classmethod
    def _trainer_gold_per_xp(cls, unit: Dict) -> float:
        if not isinstance(unit, dict):
            return 0.0

        is_neutral_unit = bool(unit.get("is_neutral_unit", False))
        if not is_neutral_unit:
            capital_raw = unit.get("capital", None)
            try:
                is_neutral_unit = capital_raw is not None and int(round(float(capital_raw))) == 0
            except (TypeError, ValueError):
                is_neutral_unit = False
        if is_neutral_unit or cls._is_hero_unit(unit):
            return 5.0

        level_raw = unit.get("Level", 0)
        try:
            level = int(round(float(level_raw or 0)))
        except (TypeError, ValueError):
            level = 0

        if level <= 0:
            return 5.0
        if level == 1:
            return 4.0
        if level == 2:
            return 5.0
        if level == 3:
            return 6.0
        return 7.0
    @staticmethod
    def _trainer_exp_cap(unit: Dict) -> Optional[int]:
        if not isinstance(unit, dict):
            return None
        exp_required_raw = unit.get("exp_required", 0)
        if isinstance(exp_required_raw, str) and exp_required_raw.strip().lower() == "max":
            return None
        try:
            exp_required = int(round(float(exp_required_raw or 0)))
        except (TypeError, ValueError):
            return None
        if exp_required <= 0:
            return None
        return max(0, exp_required - 1)
    @staticmethod
    def _is_hero_unit(unit: Dict) -> bool:
        if not isinstance(unit, dict):
            return False
        if "hero" in unit:
            value = unit.get("hero")
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return int(value) != 0
            return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}
        try:
            return int(round(float(unit.get("next_level_exp", 0) or 0))) > 0
        except (TypeError, ValueError):
            return False
    def _resolve_hero_needaunit(self, unit: Dict) -> int:
        try:
            return 1 if int(round(float(unit.get("needaunit", 0) or 0))) > 0 else 0
        except (TypeError, ValueError):
            return 0
    def _sync_hero_progression_flags(self, units: Optional[List[Dict]] = None) -> None:
        roster = units if units is not None else self.blue_team_state
        if not roster:
            return
        for unit in roster:
            unit["hero"] = self._is_hero_unit(unit)
            unit["needaunit"] = self._resolve_hero_needaunit(unit)
    def _resolve_travel_hero(self, units: Optional[List[Dict]] = None) -> Optional[Dict]:
        roster = units if units is not None else self._get_blue_state()
        heroes = [unit for unit in roster if self._is_hero_unit(unit)]
        if not heroes:
            return None
        return heroes[0]
    def _hero_level(self, units: Optional[List[Dict]] = None) -> int:
        hero = self._resolve_travel_hero(units=units)
        if hero is None:
            return 0
        try:
            return max(0, int(round(float(hero.get("Level", 0) or 0))))
        except (TypeError, ValueError):
            return 0
    @classmethod
    def _hero_ability_tokens(cls, hero: Optional[Dict]) -> set[str]:
        if not isinstance(hero, dict):
            return set()

        raw_tokens = hero.get("hero_abilities", ())
        if isinstance(raw_tokens, str):
            token_values = raw_tokens.replace(",", " ").replace(";", " ").split()
        elif isinstance(raw_tokens, dict):
            token_values = [
                key for key, value in raw_tokens.items() if bool(value)
            ]
        elif isinstance(raw_tokens, (list, tuple, set)):
            token_values = list(raw_tokens)
        else:
            token_values = []

        return {
            str(token or "").strip().lower()
            for token in token_values
            if str(token or "").strip()
        }
    def _hero_needs_faction_hire(self, units: Optional[List[Dict]] = None) -> bool:
        hero = self._resolve_travel_hero(units=units)
        if hero is None:
            return False
        return self._resolve_hero_needaunit(hero) > 0
    def _pending_hire_turn_penalty(
        self,
        delta_turns: int,
        units: Optional[List[Dict]] = None,
    ) -> float:
        if delta_turns <= 0:
            return 0.0
        if not self._hero_needs_faction_hire(units=units):
            return 0.0
        return float(delta_turns) * float(self.reward_needaunit_turn_penalty)
    def _hero_has_pathfinding(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_PATHFINDING_LEVEL)
    def _hero_has_leadership(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_LEADERSHIP_LEVEL)
    def _hero_has_second_leadership(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_SECOND_LEADERSHIP_LEVEL)
    def _hero_leadership_count(self, units: Optional[List[Dict]] = None) -> int:
        level = self._hero_level(units=units)
        count = 0
        if level >= int(self.HERO_LEADERSHIP_LEVEL):
            count += 1
        if level >= int(self.HERO_SECOND_LEADERSHIP_LEVEL):
            count += 1
        return count
    def _hero_has_endurance(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_ENDURANCE_LEVEL)
    def _hero_has_strength(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_STRENGTH_LEVEL)
    def _hero_has_banner_bearer(self, units: Optional[List[Dict]] = None) -> bool:
        if self._hero_level(units=units) >= int(self.HERO_BANNER_BEARER_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_BANNER_BEARER_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _hero_has_marching_lore(self, units: Optional[List[Dict]] = None) -> bool:
        if int(self.typeoflord) == 3:
            return True
        if self._hero_level(units=units) >= int(self.HERO_MARCHING_LORE_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_MARCHING_LORE_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _hero_has_artifact_knowledge(self, units: Optional[List[Dict]] = None) -> bool:
        if int(self.typeoflord) == 1:
            return True
        if self._hero_level(units=units) >= int(self.HERO_ARTIFACT_KNOWLEDGE_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_ARTIFACT_KNOWLEDGE_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _hero_has_sorcery_lore(self, units: Optional[List[Dict]] = None) -> bool:
        if int(self.typeoflord) == 2:
            return True
        if self._hero_level(units=units) >= int(self.HERO_SORCERY_LORE_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_SORCERY_LORE_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _hero_has_book_lore(self, units: Optional[List[Dict]] = None) -> bool:
        if self._hero_level(units=units) >= int(self.HERO_BOOK_LORE_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_BOOK_LORE_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _lord_replacement_might_level(self) -> Optional[int]:
        lord_type = int(self.typeoflord)
        if lord_type == 1:
            return int(self.HERO_ARTIFACT_KNOWLEDGE_LEVEL)
        if lord_type == 2:
            return int(self.HERO_SORCERY_LORE_LEVEL)
        if lord_type == 3:
            return int(self.HERO_MARCHING_LORE_LEVEL)
        return None
    def _hero_has_might(self, units: Optional[List[Dict]] = None) -> bool:
        hero = self._resolve_travel_hero(units=units)
        if str(self.HERO_MIGHT_ABILITY_KEY).lower() in self._hero_ability_tokens(hero):
            return True
        might_level = self._lord_replacement_might_level()
        if might_level is None:
            return False
        return self._hero_level(units=units) >= int(might_level)
    def _apply_lord_replacement_might_bonus(self, hero: Optional[Dict]) -> bool:
        if not isinstance(hero, dict) or not self._is_hero_unit(hero):
            return False

        might_level = self._lord_replacement_might_level()
        if might_level is None:
            return False

        try:
            hero_level = int(round(float(hero.get("Level", 0) or 0)))
        except (TypeError, ValueError):
            hero_level = 0
        if hero_level < int(might_level):
            return False

        applied_levels: set[int] = set()
        for level in hero.get("campaign_lord_might_bonus_levels") or []:
            try:
                applied_levels.add(int(level))
            except (TypeError, ValueError):
                continue
        if int(might_level) in applied_levels:
            return False

        base_damage = self._normalize_damage_value(
            hero.get("original_damage", hero.get("damage", 0))
        )
        boosted_damage = self._normalize_damage_value(
            float(base_damage) * float(self.HERO_MIGHT_DAMAGE_MULTIPLIER)
        )
        hero["damage"] = int(boosted_damage)
        hero["original_damage"] = int(boosted_damage)
        applied_levels.add(int(might_level))
        hero["campaign_lord_might_bonus_levels"] = sorted(applied_levels)
        return True
    @staticmethod
    def _is_empty_blue_unit(unit: Optional[Dict]) -> bool:
        if not isinstance(unit, dict):
            return True
        name = str(unit.get("name", "") or "").strip().lower()
        if name == "пусто":
            return True
        hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
        max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
        return hp <= 0 and max_hp <= 0
    def _hire_option(self, action_idx: int) -> Optional[Dict[str, object]]:
        if 0 <= int(action_idx) < len(self.active_hire_options):
            return dict(self.active_hire_options[int(action_idx)])
        return None
    def _hire_positions_for_role(self, role: object) -> Tuple[int, ...]:
        role_name = str(role or "").strip().lower()
        if role_name == "warrior":
            return tuple(int(pos) for pos in self.HIRE_FRONT_POSITIONS)
        if role_name in {"mage", "archer", "support"}:
            return tuple(int(pos) for pos in self.HIRE_BACK_POSITIONS)
        return tuple()
    def _hire_positions_for_stand(self, stand: object) -> Tuple[int, ...]:
        stand_name = str(stand or "").strip().lower()
        if stand_name == "ahead":
            return tuple(int(pos) for pos in self.HIRE_FRONT_POSITIONS)
        if stand_name == "behind":
            return tuple(int(pos) for pos in self.HIRE_BACK_POSITIONS)
        return tuple()
    def _find_first_empty_blue_position(self, positions: Tuple[int, ...]) -> Optional[int]:
        state = self._get_blue_state()
        units_by_position = {
            int(unit.get("position", -1) or -1): unit
            for unit in state
            if isinstance(unit, dict)
        }
        for position in positions:
            unit = units_by_position.get(int(position))
            if self._is_empty_blue_unit(unit):
                return int(position)
        return None
    def _can_hire_faction_unit_option(self, option: Optional[Dict[str, object]]) -> bool:
        if not isinstance(option, dict):
            return False
        if not self._is_at_castle():
            return False

        hero = self._resolve_travel_hero()
        if hero is None:
            return False
        if not self._hero_has_leadership(units=[hero]):
            return False
        if self._resolve_hero_needaunit(hero) <= 0:
            return False

        try:
            cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            return False
        if float(self.gold or 0.0) < cost:
            return False

        unit_name = str(option.get("name", "") or "").strip()
        positions = self._hire_positions_for_role(option.get("role"))
        if not unit_name or not positions:
            return False
        if self._find_unit_data_by_name(unit_name) is None:
            return False
        return self._find_first_empty_blue_position(positions) is not None
    def _hire_faction_unit_option(
        self,
        option: Optional[Dict[str, object]],
    ) -> Tuple[bool, Optional[str], Optional[int], float]:
        if not self._can_hire_faction_unit_option(option):
            return False, None, None, 0.0

        assert isinstance(option, dict)
        unit_name = str(option.get("name", "") or "").strip()
        positions = self._hire_positions_for_role(option.get("role"))
        target_pos = self._find_first_empty_blue_position(positions)
        if target_pos is None:
            return False, None, None, 0.0

        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None:
            return False, None, None, 0.0

        recruited_unit = self._build_unit_from_data(unit_data, "blue", target_pos)
        state = self._get_blue_state()
        replaced = False
        for idx, unit in enumerate(state):
            if int(unit.get("position", -1) or -1) != int(target_pos):
                continue
            state[idx] = deepcopy(recruited_unit)
            replaced = True
            break
        if not replaced:
            state.append(deepcopy(recruited_unit))
            state.sort(key=lambda unit: int(unit.get("position", 999) or 999))

        hero = self._resolve_travel_hero(units=state)
        if hero is not None:
            hero["needaunit"] = 0

        cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        self.gold = max(0.0, float(self.gold or 0.0) - cost)
        self._sync_hero_progression_flags(state)
        self._log(
            f'Найм в городе: "{unit_name}" нанят на позицию {int(target_pos)} '
            f"(стоимость={cost:.1f} gold, gold_left={float(self.gold):.1f})"
        )
        return True, unit_name, int(target_pos), cost
    @staticmethod
    def _unit_data_is_big(unit_data: Optional[Dict]) -> bool:
        if not isinstance(unit_data, dict):
            return False
        raw_value = unit_data.get("размер", unit_data.get("big", 0))
        try:
            return bool(int(raw_value or 0))
        except (TypeError, ValueError):
            return bool(raw_value)
    def _can_hire_mercenary_unit_option(self, option: Optional[Dict[str, object]]) -> bool:
        if not isinstance(option, dict):
            return False

        site_name = str(option.get("site_name", "") or "")
        if not site_name or site_name not in self._mercenary_sites_at_position(self.grid_env.agent_pos):
            return False
        if max(0, int(option.get("stock", 0) or 0)) <= 0:
            return False

        hero = self._resolve_travel_hero()
        if hero is None:
            return False
        if not self._hero_has_leadership(units=[hero]):
            return False

        unit_name = str(option.get("unit_name", option.get("name", "")) or "").strip()
        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None or self._unit_data_is_big(unit_data):
            return False

        stand = str(option.get("stand", "") or "").strip().lower()
        if not stand:
            resolved_stand = self._mercenary_stand_for_unit_data(unit_data)
            stand = "" if resolved_stand is None else resolved_stand
        positions = self._hire_positions_for_stand(stand)
        if not unit_name or not positions:
            return False

        try:
            cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            return False
        if float(self.gold or 0.0) < cost:
            return False

        return self._find_first_empty_blue_position(positions) is not None
    def _hire_mercenary_unit_option(
        self,
        option: Optional[Dict[str, object]],
    ) -> Tuple[bool, Optional[str], Optional[int], float, Optional[str], int, int]:
        if not self._can_hire_mercenary_unit_option(option):
            return False, None, None, 0.0, None, 0, 0

        assert isinstance(option, dict)
        site_name = str(option.get("site_name", "") or "")
        slot = int(option.get("slot", 0) or 0)
        unit_name = str(option.get("unit_name", option.get("name", "")) or "").strip()
        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None or self._unit_data_is_big(unit_data):
            return False, None, None, 0.0, site_name or None, 0, 0

        stand = str(option.get("stand", "") or "").strip().lower()
        if not stand:
            resolved_stand = self._mercenary_stand_for_unit_data(unit_data)
            stand = "" if resolved_stand is None else resolved_stand
        positions = self._hire_positions_for_stand(stand)
        target_pos = self._find_first_empty_blue_position(positions)
        if target_pos is None:
            return False, None, None, 0.0, site_name or None, 0, 0

        roster_entry = self._mercenary_roster_entry(site_name, slot)
        stock_before = max(0, int(roster_entry.get("stock", 0) or 0)) if roster_entry else 0
        if stock_before <= 0:
            return False, None, None, 0.0, site_name or None, stock_before, stock_before

        recruited_unit = self._build_unit_from_data(unit_data, "blue", target_pos)
        state = self._get_blue_state()
        replaced = False
        for idx, unit in enumerate(state):
            if int(unit.get("position", -1) or -1) != int(target_pos):
                continue
            state[idx] = deepcopy(recruited_unit)
            replaced = True
            break
        if not replaced:
            state.append(deepcopy(recruited_unit))
            state.sort(key=lambda unit: int(unit.get("position", 999) or 999))

        hero = self._resolve_travel_hero(units=state)
        if hero is not None:
            hero["needaunit"] = 0

        cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        self.gold = max(0.0, float(self.gold or 0.0) - cost)
        if roster_entry is not None:
            roster_entry["stock"] = max(0, stock_before - 1)
        stock_after = max(0, int(roster_entry.get("stock", 0) or 0)) if roster_entry else 0
        self._sync_hero_progression_flags(state)
        self._log(
            f'Лагерь наёмников {site_name}: "{unit_name}" нанят на позицию {int(target_pos)} '
            f"(стоимость={cost:.1f} gold, gold_left={float(self.gold):.1f}, stock_left={stock_after})"
        )
        return True, unit_name, int(target_pos), cost, site_name or None, stock_before, stock_after
    def _sync_moves_per_turn_with_hero(
        self,
        units: Optional[List[Dict]] = None,
        *,
        refill: bool = False,
        grant_delta: bool = False,
    ) -> None:
        previous_cap = int(self.moves_per_turn)
        self._sync_equipped_boot_items(units=units)
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
        terminated = False
        truncated = False

        healed_amount, unit_name = (0.0, None)
        gold_spent = 0.0
        gold_before = float(self.gold)
        temple_built = self._has_temple_built()
        missing_temple = not temple_built
        if self._is_at_castle() and temple_built and target_pos is not None:
            healed_amount, unit_name = self._heal_unit_to_full_at_position(position=target_pos)
            gold_spent = max(0.0, gold_before - float(self.gold))
        reward = max(0.0, float(healed_amount)) * self.reward_castle_heal_per_hp

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "castle_heal_action": True,
            "castle_pos_required": self.CASTLE_POS,
            "castle_heal_tiles": self.castle_heal_tiles,
            "castle_heal_available": self._is_at_castle() and temple_built,
            "castle_heal_target_pos": target_pos,
            "temple_built": temple_built,
            "missing_temple": missing_temple,
            "healed_amount": healed_amount,
            "healed_unit_name": unit_name,
            "gold_spent": gold_spent,
            "castle_heal_reward": reward,
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
    def _ensure_map_layout_consistency(self) -> None:
        """Fail fast if generated map features overlap on forbidden tiles."""
        heal_tiles = {tuple(tile) for tile in self.castle_heal_tiles}
        enemy_tiles = {tuple(pos) for pos in self.grid_env.enemy_positions.values()}
        obstacle_tiles = {tuple(pos) for pos in self.grid_env.obstacle_positions}
        chest_tiles = {tuple(pos) for pos in getattr(self, "_static_chests", {}).keys()}

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

        chest_enemy_overlap = chest_tiles & enemy_tiles
        if chest_enemy_overlap:
            raise ValueError(
                f"Chest tiles overlap enemy tiles: {sorted(chest_enemy_overlap)}"
            )

        chest_heal_overlap = chest_tiles & heal_tiles
        if chest_heal_overlap:
            raise ValueError(
                f"Chest tiles overlap heal tiles: {sorted(chest_heal_overlap)}"
            )

        chest_obstacle_overlap = chest_tiles & obstacle_tiles
        if chest_obstacle_overlap:
            raise ValueError(
                f"Chest tiles overlap obstacle tiles: {sorted(chest_obstacle_overlap)}"
            )

        reachable_targets = set(enemy_tiles)
        reachable_targets.update(heal_tiles)
        reachable_targets.update(chest_tiles)
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

        temple_built = self._has_temple_built()
        missing_temple = not temple_built
        revived, revived_unit_name, gold_spent = (False, None, 0.0)
        if self._is_at_castle() and temple_built and target_pos is not None:
            revived, revived_unit_name, gold_spent = self._revive_unit_with_gold_at_position(
                position=target_pos
            )
        reward = self.reward_castle_revive if revived else 0.0

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "castle_revive_action": True,
            "castle_heal_tiles": self.castle_heal_tiles,
            "temple_built": temple_built,
            "missing_temple": missing_temple,
            "castle_revive_available": (
                self._is_at_castle()
                and temple_built
                and target_pos is not None
                and self._can_castle_revive_position(target_pos)
            ),
            "castle_revive_target_pos": target_pos,
            "target_unit_name": target_unit_name,
            "revived": revived,
            "revived_unit_name": revived_unit_name,
            "revive_cost": revive_cost,
            "gold_spent": gold_spent,
            "castle_revive_reward": reward,
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
    def _step_hire_faction_unit(self, action: int):
        idx = action - self.GRID_HIRE_ACTION_START
        option = self._hire_option(idx)

        grid_obs = self._get_grid_obs()
        terminated = False
        truncated = False

        unit_name = None if option is None else str(option.get("name", "") or "").strip()
        hire_cost = 0.0
        hire_role = None if option is None else str(option.get("role", "") or "").strip().lower()
        hire_target_pos = None
        if option is not None:
            try:
                hire_cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
            except (TypeError, ValueError):
                hire_cost = 0.0
            positions = self._hire_positions_for_role(hire_role)
            if positions:
                hire_target_pos = self._find_first_empty_blue_position(positions)

        hire_available_before = self._can_hire_faction_unit_option(option)
        hired, hired_unit_name, hired_position, gold_spent = self._hire_faction_unit_option(option)

        hero = self._resolve_travel_hero()
        hero_needaunit = self._resolve_hero_needaunit(hero) if hero is not None else 0
        reward = self._hire_reward_value() if hired else 0.0

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "hire_action": True,
            "hire_action_idx": int(idx),
            "hire_unit_name": unit_name,
            "hire_role": hire_role,
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
    def _normalize_realcapital(self, value: int) -> int:
        """Нормализует realcapital: ожидает целое 1–5, иначе возвращает 1 (значение по умолчанию)."""
        try:
            realcapital_value = int(value)
        except (TypeError, ValueError):
            return 1
        return realcapital_value if realcapital_value in (1, 2, 3, 4, 5) else 1
    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))
    def _get_building_keys(self, buildings: Dict) -> List[str]:
        """Список ключей построек (без служебного ключа 'alredybuilt')."""
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
            self._scale_building_gold_costs_in_place(target)
    @classmethod
    def _scale_building_gold_costs_in_place(cls, buildings: Dict) -> None:
        cost_multiplier = max(0.0, float(cls.BUILDING_GOLD_COST_MULTIPLIER))
        if cost_multiplier == 1.0:
            return

        for entry in buildings.values():
            if not isinstance(entry, dict) or "gold" not in entry:
                continue
            try:
                base_gold = float(entry.get("gold", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            entry["gold"] = base_gold * cost_multiplier
    def _building_cost_multiplier(self) -> float:
        return (
            float(self.TYPEOFLORD_THREE_BUILDING_COST_MULTIPLIER)
            if int(self.typeoflord) == 3
            else 1.0
        )
    def _get_building_gold_cost(self, building: Optional[Dict[str, object]]) -> float:
        if not isinstance(building, dict):
            return 0.0
        try:
            base_cost = max(0.0, float(building.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            base_cost = 0.0
        return base_cost * float(self._building_cost_multiplier())
    def _current_max_building_gold_cost(self) -> float:
        max_price = 1.0
        for build_key in self.building_keys:
            max_price = max(
                max_price,
                self._get_building_gold_cost(self.active_buildings.get(build_key)),
            )
        return float(max_price)
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
                build_cost = self._get_building_gold_cost(building)
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

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_upgrade_settlement(self, action: int):
        idx = action - self.grid_settlement_upgrade_action_start
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        settlement_name = None
        source_tile = None
        captured = False
        current_level = None
        next_level = None
        upgrade_cost = 0.0
        upgraded = False
        insufficient_gold = False
        at_max_level = False

        if 0 <= idx < len(self.settlement_upgrade_names):
            settlement_name = str(self.settlement_upgrade_names[idx])
            source_tile = self.legions_settlement_source_tile_by_name.get(settlement_name)
            captured = (
                settlement_name in self.legions_active_settlement_territory_capture_turn_by_name
            )
            current_level = int(self.legions_settlement_level_by_name.get(settlement_name, 1) or 1)
            at_max_level = current_level >= int(self.MAX_SETTLEMENT_LEVEL)
            upgrade_cost = self._settlement_upgrade_cost_for_level(current_level)
            insufficient_gold = (
                captured
                and not at_max_level
                and upgrade_cost > 0.0
                and float(self.gold or 0.0) < float(upgrade_cost)
            )
            next_level = min(int(self.MAX_SETTLEMENT_LEVEL), current_level + 1)
            if captured and not at_max_level and upgrade_cost > 0.0 and not insufficient_gold:
                self.gold -= float(upgrade_cost)
                self.legions_settlement_level_by_name[settlement_name] = int(next_level)
                upgraded = True
                self._log(
                    f'Город "{settlement_name}" улучшен: уровень {current_level} -> {next_level} '
                    f"за {upgrade_cost:g} золота."
                )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "settlement_upgrade_action": True,
            "settlement_upgrade_name": settlement_name,
            "settlement_upgrade_source_tile": tuple(source_tile) if source_tile is not None else None,
            "settlement_upgrade_captured": bool(captured),
            "settlement_upgrade_old_level": current_level,
            "settlement_upgrade_new_level": next_level if upgraded else current_level,
            "settlement_upgrade_cost": float(upgrade_cost),
            "settlement_upgrade_applied": bool(upgraded),
            "settlement_upgrade_insufficient_gold": bool(insufficient_gold),
            "settlement_upgrade_at_max_level": bool(at_max_level),
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
    def _get_hire_options_for_capital(self, capital: Optional[int]) -> List[Dict[str, object]]:
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            capital_value = None

        if capital_value == 2:
            hire_data = HIRE_D2["legions"]
        elif capital_value == 3:
            hire_data = HIRE_D2["mountain_clans"]
        elif capital_value == 4:
            hire_data = HIRE_D2["undead_hordes"]
        elif capital_value == 5:
            hire_data = HIRE_D2["elves"]
        else:
            hire_data = HIRE_D2["empire"]

        if not isinstance(hire_data, dict):
            return []
        return [dict(entry) for entry in hire_data.values() if isinstance(entry, dict)]
    def _get_mercenary_hire_options(self) -> List[Dict[str, object]]:
        options: List[Dict[str, object]] = []
        for site_name, site_data in self.BASE_MERCENARY_SITE_DATA.items():
            site_id = str(site_data.get("site_id", "") or "")
            for idx, entry in enumerate(tuple(site_data.get("roster", ()))):
                if not isinstance(entry, dict):
                    continue
                unit_name = str(entry.get("unit_name", entry.get("name", "")) or "").strip()
                if not unit_name:
                    continue
                slot = int(entry.get("slot", idx) or idx)
                try:
                    gold_cost = max(0.0, float(entry.get("gold", 0.0) or 0.0))
                except (TypeError, ValueError):
                    gold_cost = 0.0
                unit_data = self._find_unit_data_by_name(unit_name)
                stand = self._mercenary_stand_for_unit_data(unit_data)
                options.append(
                    {
                        "site_name": str(site_name),
                        "site_id": site_id,
                        "slot": int(slot),
                        "unit_id": str(entry.get("unit_id", "") or ""),
                        "unit_name": unit_name,
                        "name": unit_name,
                        "gold": float(gold_cost),
                        "stand": "" if stand is None else stand,
                        "unit_type": (
                            str(unit_data.get("тип", unit_data.get("unit_type", "")) or "").strip()
                            if isinstance(unit_data, dict)
                            else ""
                        ),
                    }
                )
        return options
    def _refresh_dynamic_action_layout(self) -> None:
        self.active_hire_options = self._get_hire_options_for_capital(self.Realcapital)
        self.active_mercenary_hire_options = self._get_mercenary_hire_options()
        self._refresh_battle_equippable_item_names()
        self._refresh_book_item_names()
        self._refresh_staff_spell_action_entries()
        self.GRID_MERCENARY_HIRE_ACTION_START = (
            self.GRID_HIRE_ACTION_START + len(self.active_hire_options)
        )
        self.GRID_TRAINER_ACTION_START = (
            self.GRID_MERCENARY_HIRE_ACTION_START + len(self.active_mercenary_hire_options)
        )
        self.GRID_MERCHANT_BUY_ACTION_START = (
            self.GRID_TRAINER_ACTION_START + len(self.TRAINER_POSITIONS)
        )
        self.GRID_SPELL_SHOP_BUY_ACTION_START = self.GRID_MERCHANT_BUY_ACTION_START + len(
            self.MERCHANT_BUY_ITEMS
        )
        self.GRID_ENERGY_ACTION_START = self.GRID_SPELL_SHOP_BUY_ACTION_START + len(
            self.SPELL_SHOP_BUY_SPELLS
        )
        self.GRID_HASTE_ACTION_START = self.GRID_ENERGY_ACTION_START + len(
            self.ENERGY_ELIXIR_POSITIONS
        )
        self.GRID_FIRE_WARD_ACTION_START = self.GRID_HASTE_ACTION_START + len(
            self.HASTE_ELIXIR_POSITIONS
        )
        self.GRID_EARTH_WARD_ACTION_START = self.GRID_FIRE_WARD_ACTION_START + len(
            self.FIRE_WARD_POSITIONS
        )
        self.GRID_WATER_WARD_ACTION_START = self.GRID_EARTH_WARD_ACTION_START + len(
            self.EARTH_WARD_POSITIONS
        )
        self.GRID_AIR_WARD_ACTION_START = self.GRID_WATER_WARD_ACTION_START + len(
            self.WATER_WARD_POSITIONS
        )
        self.GRID_TITAN_ELIXIR_ACTION_START = self.GRID_AIR_WARD_ACTION_START + len(
            self.AIR_WARD_POSITIONS
        )
        self.GRID_SUPREME_ELIXIR_ACTION_START = self.GRID_TITAN_ELIXIR_ACTION_START + len(
            self.TITAN_ELIXIR_POSITIONS
        )
        self.GRID_BUILD_ACTION_START = self.GRID_SUPREME_ELIXIR_ACTION_START + len(
            self.SUPREME_ELIXIR_POSITIONS
        )
        self.grid_spell_action_start = self.GRID_BUILD_ACTION_START + len(self.building_keys)
        self.grid_settlement_upgrade_action_start = self.grid_spell_action_start + len(
            self.spell_keys
        )
        self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START = (
            self.grid_settlement_upgrade_action_start + len(self.settlement_upgrade_names)
        )
        self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START = (
            self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + len(self.BATTLE_EQUIPPABLE_ITEM_NAMES)
        )
        self.GRID_EQUIP_BOOK_ACTION_START = (
            self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + len(self.BATTLE_EQUIPPABLE_ITEM_NAMES)
        )
        self.grid_legion_damage_spell_action_start = (
            self.GRID_EQUIP_BOOK_ACTION_START + len(self.scenario_book_item_names)
        )
        self.grid_map_support_spell_action_start = (
            self.grid_legion_damage_spell_action_start + self._map_offensive_spell_action_max_count()
        )
        self.grid_spell_shop_cast_action_start = (
            self.grid_map_support_spell_action_start + self._map_support_spell_action_max_count()
        )
        self.grid_staff_spell_action_start = (
            self.grid_spell_shop_cast_action_start + int(self.MAX_SPELL_SHOP_CAST_ACTIONS)
        )
        self.GRID_UNLOCK_SCROLL_MAGIC_ACTION = (
            self.grid_staff_spell_action_start + len(self.staff_spell_action_entries())
        )
        self.grid_scroll_cast_action_start = self.GRID_UNLOCK_SCROLL_MAGIC_ACTION + 1
        self._refresh_book_observation_layout()
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
    def _update_turns(self) -> float:
        """Совместимость: ходы больше не рассчитываются по step_count."""
        # Ходы теперь продвигаются только явным завершением хода (REST).
        return 0.0
