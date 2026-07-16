# campaign_env_magic.py
"""CampaignMagicMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignMagicMixin:
    @classmethod
    def _empty_mana_dict(cls, value: float = 0.0) -> Dict[str, float]:
        return {
            str(kind): float(value)
            for kind in cls.MANA_KIND_ORDER
        }
    def _reset_mana_state(self) -> None:
        for kind, attr_name in self.MANA_ATTR_BY_KIND.items():
            setattr(self, str(attr_name), 0.0)
        self.legions_captured_mana_source_tiles_by_kind: Dict[str, Tuple[Tuple[int, int], ...]] = {
            str(kind): ()
            for kind in self.MANA_KIND_ORDER
        }
        self.legions_captured_mana_source_counts_by_kind: Dict[str, int] = {
            str(kind): 0
            for kind in self.MANA_KIND_ORDER
        }
        self.mana_income_per_turn: Dict[str, float] = self._empty_mana_dict()
    def _current_mana_totals(self) -> Dict[str, float]:
        return {
            str(kind): float(getattr(self, self.MANA_ATTR_BY_KIND[str(kind)], 0.0) or 0.0)
            for kind in self.MANA_KIND_ORDER
        }
    @classmethod
    def _base_mana_kind_for_capital(cls, capital: Optional[int]) -> str:
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            capital_value = 2
        return {
            1: "life",
            2: "infernal",
            3: "runes",
            4: "death",
            5: "elves",
        }.get(capital_value, "infernal")
    def _compute_mana_income_per_turn(self) -> Dict[str, float]:
        mana_income = self._empty_mana_dict()
        map_config = getattr(self, "_map", None)
        if map_config is not None and not bool(
            getattr(map_config, "passive_mana_income_enabled", True)
        ):
            return mana_income
        base_mana_kind = self._base_mana_kind_for_capital(getattr(self, "Realcapital", 2))
        mana_income[str(base_mana_kind)] += float(self.BASE_INFERNAL_MANA_PER_TURN)
        for kind in self.MANA_KIND_ORDER:
            captured_count = int(
                getattr(self, "legions_captured_mana_source_counts_by_kind", {}).get(str(kind), 0)
                or 0
            )
            mana_income[str(kind)] += float(captured_count) * float(
                self.LEGIONS_MANA_SOURCE_MANA_PER_TURN
            )
        return mana_income
    def _apply_mana_income_for_turn(self) -> Dict[str, float]:
        mana_income = self._compute_mana_income_per_turn()
        for kind, delta in mana_income.items():
            attr_name = str(self.MANA_ATTR_BY_KIND.get(str(kind), "") or "")
            if not attr_name:
                continue
            current_value = float(getattr(self, attr_name, 0.0) or 0.0)
            setattr(self, attr_name, current_value + float(delta))
        self.mana_income_per_turn = dict(mana_income)
        return dict(mana_income)
    def _mana_state_info(self) -> Dict[str, object]:
        mana_totals = self._current_mana_totals()
        mana_income = dict(self._compute_mana_income_per_turn())
        info: Dict[str, object] = {
            "mana_totals": dict(mana_totals),
            "mana_income_per_turn": dict(mana_income),
            "legions_captured_mana_source_tiles_by_kind": {
                str(kind): tuple(self.legions_captured_mana_source_tiles_by_kind.get(str(kind), ()))
                for kind in self.MANA_KIND_ORDER
            },
            "legions_captured_mana_source_counts_by_kind": {
                str(kind): int(self.legions_captured_mana_source_counts_by_kind.get(str(kind), 0) or 0)
                for kind in self.MANA_KIND_ORDER
            },
        }
        for kind in self.MANA_KIND_ORDER:
            attr_name = str(self.MANA_ATTR_BY_KIND[str(kind)])
            info[attr_name] = float(mana_totals[str(kind)])
            info[f"{attr_name}_income_per_turn"] = float(mana_income[str(kind)])
        return info
    def _get_spell_keys(self, spells: Dict) -> List[str]:
        keys: List[str] = []
        for key, entry in spells.items():
            if isinstance(entry, dict) and entry.get("id"):
                keys.append(key)
        return keys
    def _reset_spells_state(self) -> None:
        for key, template in SPELLS_D2_TEMPLATE.items():
            target = SPELLS_D2.get(key)
            if not isinstance(target, dict) or not isinstance(template, dict):
                continue
            target.clear()
            target.update(deepcopy(template))
            for entry in target.values():
                if isinstance(entry, dict):
                    entry["learned"] = int(entry.get("learned", 0) or 0)
    @classmethod
    def _spell_costs_from_entry(cls, spell: Dict, prefix: str) -> Dict[str, float]:
        costs: Dict[str, float] = {}
        for suffix, mana_kind in cls.SPELL_COST_MANA_KIND_BY_SUFFIX.items():
            raw_value = spell.get(f"{prefix}_{suffix}", 0.0) if isinstance(spell, dict) else 0.0
            try:
                value = max(0.0, float(raw_value or 0.0))
            except (TypeError, ValueError):
                value = 0.0
            costs[str(mana_kind)] = value
        return costs
    @classmethod
    @lru_cache(maxsize=256)
    def _spell_use_cost_items_from_values(
        cls,
        raw_values: Tuple[object, ...],
    ) -> Tuple[Tuple[str, float], ...]:
        items: List[Tuple[str, float]] = []
        for raw_value, mana_kind in zip(
            raw_values,
            cls.SPELL_COST_MANA_KIND_BY_SUFFIX.values(),
        ):
            try:
                value = max(0.0, float(raw_value or 0.0))
            except (TypeError, ValueError):
                value = 0.0
            items.append((str(mana_kind), value))
        return tuple(items)
    @staticmethod
    def _spell_cost_cache_key_value(raw_value: object) -> object:
        try:
            hash(raw_value)
        except TypeError:
            return repr(raw_value)
        return raw_value
    def _spell_learning_cost_multiplier(self) -> float:
        return (
            float(self.TYPEOFLORD_TWO_SPELL_LEARNING_COST_MULTIPLIER)
            if int(self.typeoflord) == 2
            else 1.0
        )
    def _get_spell_learning_costs(self, spell: Dict) -> Dict[str, float]:
        base_costs = self._spell_costs_from_entry(spell, prefix="learn")
        multiplier = float(self._spell_learning_cost_multiplier())
        if abs(multiplier - 1.0) <= 1e-9:
            return base_costs
        return {
            str(mana_kind): max(0.0, float(required_amount) * multiplier)
            for mana_kind, required_amount in base_costs.items()
        }
    def _get_spell_use_costs(self, spell: Dict) -> Dict[str, float]:
        if not isinstance(spell, dict):
            return self._spell_costs_from_entry(spell, prefix="use")
        raw_values = tuple(
            self._spell_cost_cache_key_value(spell.get(f"use_{suffix}", 0.0))
            for suffix in self.SPELL_COST_MANA_KIND_BY_SUFFIX.keys()
        )
        return dict(self._spell_use_cost_items_from_values(raw_values))
    def _same_spell_cast_limit_per_turn(self) -> int:
        if int(self.typeoflord) == 2:
            return int(self.TYPEOFLORD_TWO_SAME_SPELL_CASTS_PER_TURN)
        return 1
    def _spell_cast_count_this_turn(self, spell_key: str) -> int:
        normalized_spell_key = str(spell_key or "")
        if not normalized_spell_key:
            return 0
        return max(
            0,
            int(self.spell_cast_counts_by_id_this_turn.get(normalized_spell_key, 0) or 0),
        )
    def _spell_cast_limit_reached_this_turn(self, spell_key: str) -> bool:
        return self._spell_cast_count_this_turn(spell_key) >= int(
            self._same_spell_cast_limit_per_turn()
        )
    def _register_spell_cast_this_turn(self, spell_key: str) -> None:
        normalized_spell_key = str(spell_key or "")
        if not normalized_spell_key:
            return
        self.spell_cast_counts_by_id_this_turn[normalized_spell_key] = (
            self._spell_cast_count_this_turn(normalized_spell_key) + 1
        )
    @classmethod
    @lru_cache(maxsize=8)
    def _map_offensive_spell_action_specs_for_capital(
        cls,
        capital: Optional[int],
    ) -> Tuple[Dict[str, object], ...]:
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            return ()
        specs = cls.MAP_OFFENSIVE_SPELL_ACTION_SPECS_BY_CAPITAL.get(capital_value, ())
        return tuple(specs)
    @classmethod
    @lru_cache(maxsize=1)
    def _map_offensive_spell_action_max_count(cls) -> int:
        return max(
            [0]
            + [
                len(tuple(specs))
                for specs in cls.MAP_OFFENSIVE_SPELL_ACTION_SPECS_BY_CAPITAL.values()
            ]
        )
    def _current_map_offensive_spell_action_specs(self) -> Tuple[Dict[str, object], ...]:
        return self._map_offensive_spell_action_specs_for_capital(self.Realcapital)
    @classmethod
    @lru_cache(maxsize=8)
    def _map_offensive_spell_ids_for_capital(cls, capital: Optional[int]) -> Tuple[str, ...]:
        return tuple(
            str(spec.get("id", "") or "")
            for spec in cls._map_offensive_spell_action_specs_for_capital(capital)
            if str(spec.get("id", "") or "")
        )
    @classmethod
    @lru_cache(maxsize=1)
    def _map_offensive_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(spec.get("id", "") or ""): dict(spec)
            for specs in cls.MAP_OFFENSIVE_SPELL_ACTION_SPECS_BY_CAPITAL.values()
            for spec in tuple(specs)
            if str(spec.get("id", "") or "")
        }
    def _map_offensive_spell_spec(self, spell_key: str) -> Dict[str, object]:
        return dict(self._map_offensive_spell_specs_by_id().get(str(spell_key or ""), {}))
    def _map_offensive_spell_targets_untargetable_stacks(self, spell_key: str) -> bool:
        spell_spec = self._map_offensive_spell_spec(spell_key)
        return bool(spell_spec.get("targets_untargetable_stacks", False))
    def _get_map_offensive_spell_target(
        self,
        spell_key: str,
        *,
        for_mask: bool = False,
    ) -> Optional[Dict[str, object]]:
        spell_targetable_only = not self._map_offensive_spell_targets_untargetable_stacks(
            spell_key
        )
        if for_mask:
            return self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=spell_targetable_only
            )
        return self.get_nearest_enemy_stack(spell_targetable_only=spell_targetable_only)
    @classmethod
    @lru_cache(maxsize=8)
    def _map_support_spell_action_specs_for_capital(
        cls,
        capital: Optional[int],
    ) -> Tuple[Dict[str, object], ...]:
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            return ()
        specs = cls.MAP_SUPPORT_SPELL_ACTION_SPECS_BY_CAPITAL.get(capital_value, ())
        return tuple(specs)
    @classmethod
    @lru_cache(maxsize=1)
    def _map_support_spell_action_max_count(cls) -> int:
        return max(
            [0]
            + [
                len(tuple(specs))
                for specs in cls.MAP_SUPPORT_SPELL_ACTION_SPECS_BY_CAPITAL.values()
            ]
        )
    def _current_map_support_spell_action_specs(self) -> Tuple[Dict[str, object], ...]:
        return self._map_support_spell_action_specs_for_capital(self.Realcapital)
    @classmethod
    @lru_cache(maxsize=8)
    def _map_support_spell_ids_for_capital(cls, capital: Optional[int]) -> Tuple[str, ...]:
        return tuple(
            str(spec.get("id", "") or "")
            for spec in cls._map_support_spell_action_specs_for_capital(capital)
            if str(spec.get("id", "") or "")
        )
    @classmethod
    @lru_cache(maxsize=1)
    def _map_support_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(spec.get("id", "") or ""): dict(spec)
            for specs in cls.MAP_SUPPORT_SPELL_ACTION_SPECS_BY_CAPITAL.values()
            for spec in tuple(specs)
            if str(spec.get("id", "") or "")
        }
    def _map_support_spell_spec(self, spell_key: str) -> Dict[str, object]:
        return dict(self._map_support_spell_specs_by_id().get(str(spell_key or ""), {}))
    @classmethod
    @lru_cache(maxsize=8)
    def _map_castable_spell_ids_for_capital(cls, capital: Optional[int]) -> Tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [
                    *cls._map_offensive_spell_ids_for_capital(capital),
                    *cls._map_support_spell_ids_for_capital(capital),
                ]
            )
        )
    @classmethod
    @lru_cache(maxsize=1)
    def _all_spell_entries_by_id(cls) -> Dict[str, Dict[str, object]]:
        all_spells: Dict[str, Dict[str, object]] = {}
        for faction_spells in SPELLS_D2_TEMPLATE.values():
            if not isinstance(faction_spells, dict):
                continue
            for spell_key, spell_data in faction_spells.items():
                if not isinstance(spell_data, dict):
                    continue
                all_spells[str(spell_key)] = dict(spell_data)
        return all_spells
    def _spell_entry_by_id(self, spell_key: str) -> Optional[Dict[str, object]]:
        normalized_spell_key = str(spell_key or "")
        spell = self.active_spells.get(normalized_spell_key)
        if isinstance(spell, dict):
            return spell
        fallback_spell = self._all_spell_entries_by_id().get(normalized_spell_key)
        if isinstance(fallback_spell, dict):
            return dict(fallback_spell)
        return None
    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_supported_spell_definitions_by_id(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(entry.get("spell_id", "") or ""): dict(entry)
            for entry in SCROLL_ITEM_DEFINITIONS
            if bool(entry.get("supported", False))
            and str(entry.get("spell_id", "") or "")
        }
    def _scroll_spell_definition_by_id(self, spell_key: str) -> Dict[str, object]:
        return dict(
            self._scroll_supported_spell_definitions_by_id().get(str(spell_key or ""), {})
        )

    @classmethod
    @lru_cache(maxsize=1)
    def _staff_spell_definitions_by_item_name(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(entry.get("item_name", "") or ""): dict(entry)
            for entry in cls.STAFF_SPELL_ITEM_DEFINITIONS
            if str(entry.get("item_name", "") or "")
            and str(entry.get("spell_id", "") or "")
        }

    def _staff_spell_definition_by_item_name(self, item_name: str) -> Dict[str, object]:
        return dict(
            self._staff_spell_definitions_by_item_name().get(str(item_name or "").strip(), {})
        )

    def _staff_spell_action_entry_for_item(self, item_name: str) -> Dict[str, object]:
        definition = self._staff_spell_definition_by_item_name(item_name)
        spell_key = str(definition.get("spell_id", "") or "")
        if not spell_key:
            return {}

        spell = self._spell_entry_by_id(spell_key)
        if not isinstance(spell, dict):
            return {}

        spell_spec = self._map_offensive_spell_spec(spell_key)
        if not spell_spec:
            spell_spec = self._map_support_spell_spec(spell_key)
        if not spell_spec:
            return {}

        entry = dict(definition)
        entry.update(dict(spell_spec))
        entry["item_name"] = str(definition.get("item_name", "") or "")
        entry["spell_id"] = spell_key
        entry["id"] = spell_key
        entry["spell_kind"] = str(spell_spec.get("kind", "") or "")
        entry["spell_name"] = str(spell.get("name", "") or spell_key)
        entry["spell_description"] = str(spell.get("description", "") or spell_key)
        entry["spell_level"] = self._spell_level_from_entry(spell)
        return entry

    def _refresh_staff_spell_action_entries(self) -> Tuple[Dict[str, object], ...]:
        self.scenario_staff_spell_item_names = self._scenario_staff_spell_item_names()
        entries: List[Dict[str, object]] = []
        item_counts: Dict[str, int] = {}
        for item_name in self.scenario_staff_spell_item_names:
            item_counts[item_name] = int(item_counts.get(item_name, 0) or 0) + 1
            entry = self._staff_spell_action_entry_for_item(item_name)
            if entry:
                entry["required_item_count"] = int(item_counts[item_name])
                entries.append(entry)
        self._staff_spell_action_entries_cache = tuple(entries)
        return self._staff_spell_action_entries_cache

    def staff_spell_action_entries(self) -> Tuple[Dict[str, object], ...]:
        entries = getattr(self, "_staff_spell_action_entries_cache", None)
        if entries is None:
            entries = self._refresh_staff_spell_action_entries()
        return tuple(entries)

    def available_staff_spell_unlock_entries(self) -> Tuple[Dict[str, object], ...]:
        available_entries: List[Dict[str, object]] = []
        for entry in self.staff_spell_action_entries():
            item_name = str(entry.get("item_name", "") or "").strip()
            if not item_name:
                continue
            try:
                required_item_count = max(
                    1,
                    int(entry.get("required_item_count", 1) or 1),
                )
            except (TypeError, ValueError):
                required_item_count = 1
            if self._count_hero_item(item_name) >= required_item_count:
                available_entries.append(dict(entry))
        return tuple(available_entries)

    def _has_scroll_or_staff_magic_unlock_source(self) -> bool:
        return bool(
            self.available_scroll_spell_entries()
            or self.available_staff_spell_unlock_entries()
        )

    def _can_cast_staff_spell(
        self,
        slot_entry: Dict[str, object],
        *,
        nearest_targetable_enemy: Optional[Dict[str, object]] = None,
        nearest_any_enemy: Optional[Dict[str, object]] = None,
    ) -> bool:
        if not bool(self.scroll_magic_unlocked):
            return False
        if not isinstance(slot_entry, dict):
            return False
        item_name = str(slot_entry.get("item_name", "") or "").strip()
        spell_key = str(slot_entry.get("spell_id", slot_entry.get("id", "")) or "")
        spell_kind = str(slot_entry.get("spell_kind", slot_entry.get("kind", "")) or "")
        if not item_name or not spell_key or not self._is_staff_spell_item_name(item_name):
            return False
        try:
            required_item_count = max(1, int(slot_entry.get("required_item_count", 1) or 1))
        except (TypeError, ValueError):
            required_item_count = 1
        if self._count_hero_item(item_name) < required_item_count:
            return False
        if not (
            self._travel_hero_is_mage()
            or self._hero_has_sorcery_lore()
        ):
            return False

        spell = self._spell_entry_by_id(spell_key)
        if not isinstance(spell, dict):
            return False
        if self._is_spell_blocked_by_typeoflord(spell):
            return False
        if not self._has_mana_for_costs(self._get_spell_use_costs(spell)):
            return False

        if self._spell_kind_is_offensive(spell_kind):
            if spell_kind == "summon_battle":
                summon_unit_name = str(slot_entry.get("summon_unit_name", "") or "").strip()
                if not summon_unit_name or not isinstance(
                    self._find_unit_data_by_name(summon_unit_name),
                    dict,
                ):
                    return False
            nearest_enemy = self._resolve_spell_target_enemy(
                spell_key,
                slot_entry,
                nearest_targetable_enemy=nearest_targetable_enemy,
                nearest_any_enemy=nearest_any_enemy,
            )
            if nearest_enemy is None:
                return False
            return self._enemy_team_has_living_units(nearest_enemy.get("enemy_id"))

        if self._spell_kind_is_support(spell_kind):
            return self._support_spell_spec_would_apply(spell_key, slot_entry)

        return False
    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_offensive_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        offensive_kinds = {"damage", "debuff", "summon_battle"}
        return {
            str(entry.get("spell_id", "") or ""): dict(entry)
            for entry in SCROLL_ITEM_DEFINITIONS
            if bool(entry.get("supported", False))
            and str(entry.get("spell_kind", "") or "") in offensive_kinds
        }
    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_support_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(entry.get("spell_id", "") or ""): dict(entry)
            for entry in SCROLL_ITEM_DEFINITIONS
            if bool(entry.get("supported", False))
            and str(entry.get("spell_kind", "") or "")
            in {"moves", "heal", "buff", "ward", "health_bonus"}
        }
    @classmethod
    @lru_cache(maxsize=1)
    def _combined_map_offensive_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        combined = dict(cls._map_offensive_spell_specs_by_id())
        combined.update(cls._scroll_offensive_spell_specs_by_id())
        return combined
    @classmethod
    @lru_cache(maxsize=1)
    def _combined_map_support_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        combined = dict(cls._map_support_spell_specs_by_id())
        combined.update(cls._scroll_support_spell_specs_by_id())
        return combined
    def _is_legions_capital(self) -> bool:
        return int(self.Realcapital) == 2
    def _is_empire_capital(self) -> bool:
        return int(self.Realcapital) == 1
    def _is_undead_hordes_capital(self) -> bool:
        return int(self.Realcapital) == 4
    def _create_initial_enemy_team_states(self) -> Dict[int, List[Dict]]:
        # В режиме blue_dragon объект-дракон (enemy_31) — синий вместо зелёного.
        configs = (
            self._enemy_configs_blue_dragon
            if self._campaign_objective_is_blue_dragon()
            else self._enemy_configs
        )
        states = {
            int(enemy_id): deepcopy(team)
            for enemy_id, team in configs.items()
        }
        level_overrides = getattr(self._map, "enemy_unit_level_overrides", None) or {}
        for raw_enemy_id, unit_levels in level_overrides.items():
            try:
                enemy_id = int(raw_enemy_id)
            except (TypeError, ValueError):
                continue
            if not isinstance(unit_levels, dict):
                continue
            normalized_levels = {
                str(unit_name): max(1, int(level))
                for unit_name, level in unit_levels.items()
                if str(unit_name or "").strip()
            }
            for unit in states.get(enemy_id, ()):
                unit_name = str(unit.get("name", "") or "")
                if unit_name in normalized_levels:
                    unit["Level"] = int(normalized_levels[unit_name])
        return states
    def _get_enemy_team_state(self, enemy_id: Optional[int]) -> List[Dict]:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return []
        team = self.enemy_team_states.get(normalized_enemy_id)
        if team is None:
            team = deepcopy(self._enemy_configs.get(normalized_enemy_id, []))
            self.enemy_team_states[normalized_enemy_id] = team
        return team
    def _has_named_building_built(self, building_name: str) -> bool:
        normalized_name = str(building_name or "").strip()
        if not normalized_name:
            return False
        for build_key in self.building_keys:
            building = self.active_buildings.get(build_key)
            if not isinstance(building, dict):
                continue
            if str(building.get("name", "") or "").strip() != normalized_name:
                continue
            try:
                return int(building.get("Build", building.get("built", 0)) or 0) == 1
            except (TypeError, ValueError):
                return False
        return False
    def _has_magic_tower_built(self) -> bool:
        return self._has_named_building_built(self.MAGIC_TOWER_BUILDING_NAME)
    def _has_temple_built(self) -> bool:
        return self._has_named_building_built(self.TEMPLE_BUILDING_NAME)
    def _has_mana_for_costs(self, costs: Dict[str, float]) -> bool:
        for mana_kind, required_amount in costs.items():
            attr_name = str(self.MANA_ATTR_BY_KIND.get(str(mana_kind), "") or "")
            if not attr_name:
                continue
            current_amount = float(getattr(self, attr_name, 0.0) or 0.0)
            if current_amount + 1e-9 < float(required_amount):
                return False
        return True
    def _spend_mana_costs(self, costs: Dict[str, float]) -> None:
        for mana_kind, required_amount in costs.items():
            attr_name = str(self.MANA_ATTR_BY_KIND.get(str(mana_kind), "") or "")
            if not attr_name:
                continue
            current_amount = float(getattr(self, attr_name, 0.0) or 0.0)
            setattr(self, attr_name, max(0.0, current_amount - float(required_amount)))
    def get_learned_spell_descriptions(self) -> List[str]:
        learned_descriptions: List[str] = []
        for spell_key in self.spell_keys:
            spell = self.active_spells.get(spell_key)
            if not isinstance(spell, dict):
                continue
            try:
                is_learned = int(spell.get("learned", 0) or 0) == 1
            except (TypeError, ValueError):
                is_learned = False
            if not is_learned:
                continue
            description = str(spell.get("description", "") or "").strip()
            if description:
                learned_descriptions.append(description)
        return learned_descriptions
    def _spell_state_info(self) -> Dict[str, object]:
        learned_spells = self.get_learned_spell_descriptions()
        spell_shop_purchased_spells = [
            str(spell_data.get("name", "") or "")
            for spell_data in self.SPELL_SHOP_BUY_SPELLS
            if str(spell_data.get("spell_id", "") or "") in self.spell_shop_purchased_spell_ids
        ]
        info = {
            "learned_spells": list(learned_spells),
            "learned_spells_count": int(len(learned_spells)),
            "spells_total": int(len(self.spell_keys)),
            "spell_learning_locked": bool(self.spell_learning_locked),
            "magic_tower_built": bool(self._has_magic_tower_built()),
            "spell_shop_purchased_spells": spell_shop_purchased_spells,
            "spell_shop_purchased_spells_count": int(len(spell_shop_purchased_spells)),
        }
        info.update(self._staff_spell_state_info())
        info.update(self._scroll_state_info())
        return info
    def _staff_spell_state_info(self) -> Dict[str, object]:
        slot_entries = self.staff_spell_action_entries()
        return {
            "staff_spell_cast_slots": [
                {
                    "slot": int(idx),
                    "item_name": str(entry.get("item_name", "") or ""),
                    "spell_id": str(entry.get("spell_id", "") or ""),
                    "spell_name": str(entry.get("spell_name", "") or ""),
                    "spell_kind": str(entry.get("spell_kind", "") or ""),
                    "spell_level": int(entry.get("spell_level", 0) or 0),
                    "required_item_count": int(
                        entry.get("required_item_count", 1) or 1
                    ),
                    "owned": bool(
                        self._count_hero_item(str(entry.get("item_name", "") or ""))
                        >= int(entry.get("required_item_count", 1) or 1)
                    ),
                }
                for idx, entry in enumerate(slot_entries)
            ],
            "staff_spell_cast_slots_count": int(len(slot_entries)),
        }
    def _scroll_state_info(self) -> Dict[str, object]:
        self._ensure_inventory_cache()
        inventory_entries = self._scroll_inventory_entries_cache
        slot_entries = self._scroll_cast_slot_entries_cache
        return {
            "scroll_magic_unlocked": bool(self.scroll_magic_unlocked),
            "supported_scroll_types_count": int(len(self._available_scroll_spell_entries_cache)),
            "total_scroll_count": int(self._total_scroll_count_cache),
            "scroll_inventory": [
                {
                    "item_name": str(entry.get("item_name", "") or ""),
                    "spell_id": str(entry.get("spell_id", "") or ""),
                    "spell_name": str(entry.get("spell_name", "") or ""),
                    "copies": int(entry.get("copies", 0) or 0),
                    "supported": bool(entry.get("supported", False)),
                }
                for entry in inventory_entries
            ],
            "scroll_cast_slots": [
                {
                    "slot": int(idx),
                    "item_name": str(entry.get("item_name", "") or ""),
                    "spell_id": str(entry.get("spell_id", "") or ""),
                    "spell_name": str(entry.get("spell_name", "") or ""),
                    "copies": int(entry.get("copies", 0) or 0),
                    "spell_kind": str(entry.get("spell_kind", "") or ""),
                    "spell_level": int(entry.get("spell_level", 0) or 0),
                }
                for idx, entry in enumerate(slot_entries)
            ],
        }
    def _is_spell_learned(self, spell_key: str) -> bool:
        spell = self.active_spells.get(str(spell_key))
        if not isinstance(spell, dict):
            return False
        try:
            return int(spell.get("learned", 0) or 0) == 1
        except (TypeError, ValueError):
            return False
    @staticmethod
    def _spell_level_from_entry(spell: Optional[Dict[str, object]]) -> int:
        if not isinstance(spell, dict):
            return 0
        try:
            return int(spell.get("level", 0) or 0)
        except (TypeError, ValueError):
            return 0
    def _is_spell_level_masked_by_typeoflord(self, spell_level: Optional[int]) -> bool:
        try:
            normalized_level = int(spell_level or 0)
        except (TypeError, ValueError):
            normalized_level = 0
        return normalized_level >= 5 and int(self.typeoflord) != 2
    def _is_spell_blocked_by_typeoflord(self, spell: Optional[Dict[str, object]]) -> bool:
        return self._is_spell_level_masked_by_typeoflord(self._spell_level_from_entry(spell))
    def _can_cast_legion_damage_spell(
        self,
        spell_key: str,
        *,
        nearest_enemy: Optional[Dict[str, object]] = None,
    ) -> bool:
        normalized_spell_key = str(spell_key or "")
        current_spell_ids = self._map_offensive_spell_ids_for_capital(self.Realcapital)
        if not current_spell_ids:
            return False
        if normalized_spell_key not in current_spell_ids:
            return False
        if self._spell_cast_limit_reached_this_turn(normalized_spell_key):
            return False
        spell = self.active_spells.get(normalized_spell_key)
        if not isinstance(spell, dict):
            return False
        spell_spec = self._map_offensive_spell_spec(normalized_spell_key)
        if self._is_spell_blocked_by_typeoflord(spell):
            return False
        if not self._is_spell_learned(normalized_spell_key):
            return False
        if str(spell_spec.get("kind", "") or "") == "summon_battle":
            summon_unit_name = str(spell_spec.get("summon_unit_name", "") or "").strip()
            if not summon_unit_name or not isinstance(
                self._find_unit_data_by_name(summon_unit_name),
                dict,
            ):
                return False
        if nearest_enemy is None:
            nearest_enemy = self._get_map_offensive_spell_target(normalized_spell_key)
        if nearest_enemy is None:
            return False
        target_enemy_id = nearest_enemy.get("enemy_id")
        if not self._enemy_team_has_living_units(target_enemy_id):
            return False
        return self._has_mana_for_costs(self._get_spell_use_costs(spell))
    def _can_cast_support_spell(self, spell_key: str) -> bool:
        normalized_spell_key = str(spell_key or "")
        current_spell_ids = self._map_support_spell_ids_for_capital(self.Realcapital)
        if not current_spell_ids:
            return False
        if normalized_spell_key not in current_spell_ids:
            return False
        if self._spell_cast_limit_reached_this_turn(normalized_spell_key):
            return False
        spell = self.active_spells.get(normalized_spell_key)
        if not isinstance(spell, dict):
            return False
        if self._is_spell_blocked_by_typeoflord(spell):
            return False
        if not self._is_spell_learned(normalized_spell_key):
            return False
        spell_spec = self._map_support_spell_spec(normalized_spell_key)
        if not self._support_spell_spec_would_apply(normalized_spell_key, spell_spec):
            return False
        return self._has_mana_for_costs(self._get_spell_use_costs(spell))
    def _can_cast_spell_shop_spell(
        self,
        spell_key: str,
        *,
        nearest_targetable_enemy: Optional[Dict[str, object]] = None,
        nearest_any_enemy: Optional[Dict[str, object]] = None,
    ) -> bool:
        normalized_spell_key = str(spell_key or "")
        if not normalized_spell_key:
            return False
        if not self._spell_shop_spell_owned(normalized_spell_key):
            return False
        if self._spell_shop_spell_has_regular_action(normalized_spell_key):
            return False
        if self._spell_shop_cast_limit_reached_this_turn(normalized_spell_key):
            return False

        spell = self._spell_entry_by_id(normalized_spell_key)
        if not isinstance(spell, dict):
            return False

        offensive_spec = self._map_offensive_spell_spec(normalized_spell_key)
        if offensive_spec:
            if str(offensive_spec.get("kind", "") or "") == "summon_battle":
                summon_unit_name = str(offensive_spec.get("summon_unit_name", "") or "").strip()
                if not summon_unit_name or not isinstance(
                    self._find_unit_data_by_name(summon_unit_name),
                    dict,
                ):
                    return False
            nearest_enemy = (
                nearest_any_enemy
                if self._map_offensive_spell_targets_untargetable_stacks(normalized_spell_key)
                else nearest_targetable_enemy
            )
            if nearest_enemy is None:
                nearest_enemy = self._get_map_offensive_spell_target(normalized_spell_key)
            if nearest_enemy is None:
                return False
            target_enemy_id = nearest_enemy.get("enemy_id")
            if not self._enemy_team_has_living_units(target_enemy_id):
                return False
            return self._has_mana_for_costs(self._get_spell_use_costs(spell))

        support_spec = self._map_support_spell_spec(normalized_spell_key)
        if support_spec:
            if not self._support_spell_spec_would_apply(normalized_spell_key, support_spec):
                return False
            return self._has_mana_for_costs(self._get_spell_use_costs(spell))

        return False
    @staticmethod
    def _spell_kind_is_offensive(spell_kind: str) -> bool:
        return str(spell_kind or "") in {"damage", "debuff", "summon_battle"}
    @staticmethod
    def _spell_kind_is_support(spell_kind: str) -> bool:
        return str(spell_kind or "") in {"moves", "heal", "buff", "ward", "health_bonus"}
    def _resolve_spell_target_enemy(
        self,
        spell_key: str,
        spell_spec: Dict[str, object],
        *,
        nearest_targetable_enemy: Optional[Dict[str, object]] = None,
        nearest_any_enemy: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict[str, object]]:
        normalized_spell_key = str(spell_key or "")
        spell_kind = str(spell_spec.get("spell_kind", spell_spec.get("kind", "")) or "")
        if not self._spell_kind_is_offensive(spell_kind):
            return None
        targets_untargetable = bool(
            spell_spec.get(
                "targets_untargetable_stacks",
                spell_kind == "summon_battle"
                or self._map_offensive_spell_targets_untargetable_stacks(normalized_spell_key),
            )
        )
        if targets_untargetable:
            return nearest_any_enemy or self.get_nearest_enemy_stack(
                spell_targetable_only=False
            )
        return nearest_targetable_enemy or self.get_nearest_enemy_stack(
            spell_targetable_only=True
        )
    def _can_cast_scroll_spell(
        self,
        slot_entry: Dict[str, object],
        *,
        nearest_targetable_enemy: Optional[Dict[str, object]] = None,
        nearest_any_enemy: Optional[Dict[str, object]] = None,
    ) -> bool:
        if not bool(self.scroll_magic_unlocked):
            return False
        if not isinstance(slot_entry, dict):
            return False
        item_name = str(slot_entry.get("item_name", "") or "")
        spell_key = str(slot_entry.get("spell_id", "") or "")
        spell_kind = str(slot_entry.get("spell_kind", "") or "")
        if (
            not item_name
            or not spell_key
            or not bool(slot_entry.get("supported", False))
            or self.count_scroll_item(item_name) <= 0
        ):
            return False
        if self._spell_kind_is_offensive(spell_kind):
            if spell_kind == "summon_battle":
                summon_unit_name = str(slot_entry.get("summon_unit_name", "") or "").strip()
                if not summon_unit_name or not isinstance(
                    self._find_unit_data_by_name(summon_unit_name),
                    dict,
                ):
                    return False
            nearest_enemy = self._resolve_spell_target_enemy(
                spell_key,
                slot_entry,
                nearest_targetable_enemy=nearest_targetable_enemy,
                nearest_any_enemy=nearest_any_enemy,
            )
            if nearest_enemy is None:
                return False
            return self._enemy_team_has_living_units(nearest_enemy.get("enemy_id"))
        if self._spell_kind_is_support(spell_kind):
            return self._support_spell_spec_would_apply(spell_key, slot_entry)
        return False
    def _cast_from_spell_spec(
        self,
        *,
        source: str,
        spell_key: str,
        spell_description: Optional[str],
        spell_spec: Dict[str, object],
        mana_costs: Optional[Dict[str, float]] = None,
        spend_mana: bool = False,
        enforce_turn_limit: bool = False,
        consume_item_name: Optional[str] = None,
        nearest_targetable_enemy: Optional[Dict[str, object]] = None,
        nearest_any_enemy: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        normalized_spell_key = str(spell_key or "")
        spell_kind = str(spell_spec.get("spell_kind", spell_spec.get("kind", "")) or "")
        mana_costs = dict(mana_costs or {})
        result: Dict[str, object] = {
            "source": str(source or ""),
            "spell_cast_applied": False,
            "spell_cast_executed": False,
            "spell_cast_reward": 0.0,
            "spell_units_affected": 0,
            "spell_heal_total": 0.0,
            "spell_moves_restored": 0.0,
            "moves_before_cast": float(self.moves or 0.0),
            "moves_after_cast": float(self.moves or 0.0),
            "target_enemy_id": None,
            "target_enemy_position": None,
            "target_enemy_reachable": False,
            "target_enemy_description": None,
            "spell_damage_total": 0.0,
            "spell_damage_units_hit": 0,
            "spell_damage_units_defeated": 0,
            "spell_damage_units_blocked": 0,
            "spell_enemy_defeated": False,
            "enemy_defeat_reward": 0.0,
            "enemy_defeat_reward_base": 0.0,
            "ruin_clear_bonus_reward": 0.0,
            "final_objective_reward": 0.0,
            "newly_captured_objective_cities": [],
            "newly_activated_settlement_territories": [],
            "battle_triggered": False,
            "battle_context_kind": None,
            "spell_effect_summary": (
                self._blue_stack_spell_effect_summary()
                if self._spell_kind_is_support(spell_kind)
                else {
                    "active_spell_ids": [],
                    "debuff_types": [],
                    "armor_delta": 0,
                    "damage_multiplier": 1.0,
                    "initiative_multiplier": 1.0,
                    "accuracy_multiplier": 1.0,
                }
            ),
            "scroll_item_name": str(consume_item_name or ""),
            "scroll_item_consumed": False,
            "scroll_copies_before": (
                int(self.count_scroll_item(str(consume_item_name or "")))
                if consume_item_name
                else 0
            ),
            "scroll_copies_after": (
                int(self.count_scroll_item(str(consume_item_name or "")))
                if consume_item_name
                else 0
            ),
            "reward": 0.0,
        }
        if self._spell_kind_is_support(spell_kind) and not self._support_spell_spec_would_apply(
            normalized_spell_key,
            spell_spec,
        ):
            return result

        if spend_mana and mana_costs:
            self._spend_mana_costs(mana_costs)
        if enforce_turn_limit and normalized_spell_key:
            self._register_spell_cast_this_turn(normalized_spell_key)

        result["spell_cast_executed"] = True
        result["spell_cast_reward"] = float(self.reward_spell_cast)
        result["reward"] = float(result["reward"]) + float(self.reward_spell_cast)

        spell_damage = float(spell_spec.get("damage", 0.0) or 0.0)
        spell_damage_type = str(spell_spec.get("damage_type", "") or "") or None
        spell_debuff_type = str(spell_spec.get("debuff_type", "") or "") or None
        spell_buff_type = str(spell_spec.get("buff_type", "") or "") or None
        spell_resistance_type = str(spell_spec.get("resistance_type", "") or "") or None
        spell_heal_amount = float(spell_spec.get("heal_amount", 0.0) or 0.0)
        spell_health_delta = int(spell_spec.get("health_delta", 0) or 0)
        spell_moves_restore_fraction = float(
            spell_spec.get("moves_restore_fraction", 0.0) or 0.0
        )
        summon_unit_name = str(spell_spec.get("summon_unit_name", "") or "") or None

        nearest_enemy = self._resolve_spell_target_enemy(
            normalized_spell_key,
            spell_spec,
            nearest_targetable_enemy=nearest_targetable_enemy,
            nearest_any_enemy=nearest_any_enemy,
        )
        if nearest_enemy is not None:
            result["target_enemy_id"] = int(nearest_enemy.get("enemy_id", -1))
            result["target_enemy_position"] = tuple(nearest_enemy.get("position", ()))
            result["target_enemy_reachable"] = bool(nearest_enemy.get("reachable", False))
            result["target_enemy_description"] = str(
                nearest_enemy.get("description", "") or ""
            )

        target_enemy_id = result["target_enemy_id"]
        if spell_kind == "damage":
            (
                result["spell_damage_total"],
                result["spell_damage_units_hit"],
                result["spell_damage_units_defeated"],
                result["spell_damage_units_blocked"],
            ) = self._apply_damage_to_enemy_stack(
                target_enemy_id,
                spell_damage,
                spell_damage_type,
            )
            result["spell_cast_applied"] = int(result["spell_damage_units_hit"]) > 0
            result["spell_units_affected"] = int(result["spell_damage_units_hit"])
        elif spell_kind == "debuff":
            (
                result["spell_units_affected"],
                result["spell_effect_summary"],
            ) = self._apply_debuff_to_enemy_stack(target_enemy_id, normalized_spell_key)
            result["spell_cast_applied"] = int(result["spell_units_affected"]) > 0
        elif spell_kind == "summon_battle":
            summon_battle_team = self._build_summoned_blue_team(str(summon_unit_name or ""))
            if summon_battle_team is not None and target_enemy_id is not None:
                summon_blue_team, summoned_unit = summon_battle_team
                result["spell_cast_applied"] = True
                result["spell_units_affected"] = 1
                self.summon_hero_battle_bonus_enemy_ids_this_turn.add(int(target_enemy_id))
                self.summon_hero_battle_bonus_pending = True
                self.current_enemy_id = int(target_enemy_id)
                self.battle_origin_pos = tuple(self.grid_env.agent_pos)
                self.current_battle_context = {
                    "kind": "summon_spell",
                    "spell_key": normalized_spell_key,
                    "summoned_unit_name": str(summon_unit_name or ""),
                    "target_enemy_id": int(target_enemy_id),
                }
                self.mode = self.MODE_BATTLE
                self._log(
                    f'Применено заклинание "{spell_description}" по ближайшему вражескому отряду '
                    f"{target_enemy_id}: призван {summoned_unit['name']} для отдельного боя."
                )
                self._init_battle(int(target_enemy_id), blue_team=summon_blue_team)
                result["battle_triggered"] = True
                result["battle_context_kind"] = "summon_spell"
            else:
                result["spell_cast_applied"] = False
        elif spell_kind == "moves":
            result["spell_moves_restored"] = self._restore_grid_moves(
                spell_moves_restore_fraction
            )
            result["spell_units_affected"] = (
                1 if float(result["spell_moves_restored"]) > 0.0 else 0
            )
            result["spell_cast_applied"] = float(result["spell_moves_restored"]) > 0.0
            result["spell_effect_summary"] = self._blue_stack_spell_effect_summary()
        elif spell_kind == "heal":
            (
                result["spell_units_affected"],
                result["spell_heal_total"],
            ) = self._apply_heal_to_blue_stack(spell_heal_amount)
            result["spell_cast_applied"] = float(result["spell_heal_total"]) > 0.0
            result["spell_effect_summary"] = self._blue_stack_spell_effect_summary()
        elif spell_kind in {"buff", "ward", "health_bonus"}:
            (
                result["spell_units_affected"],
                result["spell_effect_summary"],
            ) = self._apply_support_spell_to_blue_stack(normalized_spell_key)
            result["spell_cast_applied"] = int(result["spell_units_affected"]) > 0

        if consume_item_name:
            if bool(result["spell_cast_applied"]) and self._consume_hero_item(consume_item_name):
                result["scroll_item_consumed"] = True
            result["scroll_copies_after"] = int(self.count_scroll_item(consume_item_name))

        if spell_description and spell_kind != "summon_battle":
            if spell_kind == "damage":
                self._log(
                    f'Применено заклинание "{spell_description}" по ближайшему вражескому отряду '
                    f"{target_enemy_id}: {float(result['spell_damage_total']):g} урона суммарно, "
                    f"задето {int(result['spell_damage_units_hit'])} юнитов, "
                    f"заблокировано {int(result['spell_damage_units_blocked'])}, "
                    f"добито {int(result['spell_damage_units_defeated'])}."
                )
            elif spell_kind == "debuff":
                self._log(
                    f'Применено заклинание "{spell_description}" по ближайшему вражескому отряду '
                    f"{target_enemy_id}: дебафф '{spell_debuff_type or 'unknown'}' на "
                    f"{int(result['spell_units_affected'])} живых юнитах до следующего игрового хода."
                )
            elif spell_kind == "moves":
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"восстановлено {float(result['spell_moves_restored']):g} ед. движения "
                    f"({float(result['moves_before_cast']):g} -> {float(self.moves or 0.0):g} "
                    f"из {float(self.moves_per_turn or 0.0):g})."
                )
            elif spell_kind == "heal":
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"исцелено {float(result['spell_heal_total']):g} HP суммарно на "
                    f"{int(result['spell_units_affected'])} юнитах."
                )
            elif spell_kind == "ward":
                ward_description = (
                    "защита от первой оружейной атаки"
                    if str(spell_resistance_type or "") == "Weapon"
                    else f"защита от магии {spell_resistance_type or 'unknown'}"
                )
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"{ward_description} до следующего игрового хода."
                )
            elif spell_kind == "health_bonus":
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"+{spell_health_delta} HP на {int(result['spell_units_affected'])} живых юнитах "
                    f"до следующего игрового хода."
                )
            elif spell_kind == "buff":
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"бафф '{spell_buff_type or 'unknown'}' на {int(result['spell_units_affected'])} "
                    f"живых юнитах до следующего игрового хода."
                )

        if (
            self._spell_kind_is_offensive(spell_kind)
            and spell_kind != "summon_battle"
            and target_enemy_id is not None
            and not self._enemy_team_has_living_units(target_enemy_id)
        ):
            result["spell_enemy_defeated"] = True
            result["enemy_defeat_reward"] = self._compute_enemy_defeat_reward(target_enemy_id)
            result["enemy_defeat_reward_base"] = float(self.reward_defeat_enemy)
            result["ruin_clear_bonus_reward"] = (
                float(self.reward_ruin_clear_bonus)
                if int(target_enemy_id or -1) in self.RUIN_REWARD_BY_ENEMY_ID
                else 0.0
            )
            result["reward"] = (
                float(result["reward"])
                + float(result["enemy_defeat_reward"])
                + float(result["ruin_clear_bonus_reward"])
            )
            self.grid_env.mark_enemy_defeated(target_enemy_id)
            self._clear_enemy_map_spell_effects_for_enemy(target_enemy_id)
            self._log(f"=== ВРАГ {target_enemy_id} УНИЧТОЖЕН ЗАКЛИНАНИЕМ НА КАРТЕ ===")

            objective_cities_captured_before = len(self.captured_objective_cities)
            result["newly_captured_objective_cities"] = self._capture_objective_city_if_cleared(
                target_enemy_id
            )
            result["newly_activated_settlement_territories"] = (
                self._activate_legions_settlement_territory_if_cleared(target_enemy_id)
            )
            if (
                result["newly_captured_objective_cities"]
                and self._campaign_objective_is_cities()
            ):
                result["final_objective_reward"] = self._compute_final_objective_reward(
                    len(result["newly_captured_objective_cities"]),
                    captured_before=objective_cities_captured_before,
                )
                result["reward"] = float(result["reward"]) + float(
                    result["final_objective_reward"]
                )
            result["reward"] = self._apply_green_dragon_objective_reward_if_needed(
                target_enemy_id,
                float(result["reward"]),
                result,
            )
            if result["newly_activated_settlement_territories"]:
                for settlement_name in result["newly_activated_settlement_territories"]:
                    self._log(
                        f"=== ЗЕМЛЯ ЛЕГИОНОВ НАЧИНАЕТ РАСПРОСТРАНЯТЬСЯ ИЗ ПОСЕЛЕНИЯ {settlement_name} ==="
                    )

        result["moves_after_cast"] = float(self.moves or 0.0)
        return result
    def _build_spell_cast_info(
        self,
        *,
        spell_key: Optional[str],
        spell_description: Optional[str],
        spell_level: Optional[int],
        spell_kind: str,
        spell_spec: Dict[str, object],
        cast_result: Dict[str, object],
        can_cast: bool,
        mana_costs: Optional[Dict[str, float]] = None,
        spell_cast_used_this_turn: bool = False,
        spell_is_learned: bool = False,
        blocked_by_typeoflord: bool = False,
        target_enemy_id: object = None,
        target_enemy_position: object = None,
        target_enemy_reachable: bool = False,
        target_enemy_description: object = None,
        spell_effect_summary: Optional[Dict[str, object]] = None,
        moves_before_cast: Optional[float] = None,
        minimal_extra: Optional[Dict[str, object]] = None,
        detailed_extra: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        cast_result = cast_result or {}
        mana_costs = mana_costs or {}
        spell_key = str(spell_key or "")
        spell_kind = str(spell_kind or "")

        spell_cast_applied = bool(cast_result.get("spell_cast_applied", False))
        spell_cast_executed = bool(cast_result.get("spell_cast_executed", False))
        spell_cast_reward = float(cast_result.get("spell_cast_reward", 0.0) or 0.0)
        spell_enemy_defeated = bool(cast_result.get("spell_enemy_defeated", False))
        target_enemy_id = cast_result.get("target_enemy_id", target_enemy_id)
        target_enemy_position = cast_result.get("target_enemy_position", target_enemy_position)
        target_enemy_reachable = bool(
            cast_result.get("target_enemy_reachable", target_enemy_reachable)
        )
        target_enemy_description = cast_result.get(
            "target_enemy_description",
            target_enemy_description,
        )

        insufficient_mana = bool(
            not can_cast and mana_costs and not self._has_mana_for_costs(mana_costs)
        )
        info: Dict[str, object] = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": bool(cast_result.get("battle_triggered", False)),
            "spell_cast_action": True,
            "spell_key": spell_key,
            "spell_kind": spell_kind,
            "spell_cast_applied": spell_cast_applied,
            "spell_cast_executed": spell_cast_executed,
            "spell_cast_reward": spell_cast_reward,
            "spell_enemy_defeated": spell_enemy_defeated,
            "target_enemy_id": target_enemy_id,
            "insufficient_mana": insufficient_mana,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }
        if minimal_extra:
            info.update(minimal_extra)
        if not self._include_detailed_step_info():
            return info

        spell_spec = dict(spell_spec or {})
        mana_costs = dict(mana_costs)
        detailed_extra = detailed_extra or {}
        raw_spell_effect_summary = (
            spell_effect_summary
            if spell_effect_summary is not None
            else cast_result.get("spell_effect_summary", {})
        )
        spell_effect_summary = (
            dict(raw_spell_effect_summary)
            if isinstance(raw_spell_effect_summary, dict)
            else {}
        )
        if not spell_effect_summary:
            spell_effect_summary = (
                self._blue_stack_spell_effect_summary()
                if self._spell_kind_is_support(spell_kind)
                else {
                    "active_spell_ids": [],
                    "debuff_types": [],
                    "armor_delta": 0,
                    "damage_multiplier": 1.0,
                    "initiative_multiplier": 1.0,
                    "accuracy_multiplier": 1.0,
                }
            )

        def _result_float(key: str, default: float = 0.0) -> float:
            return float(cast_result.get(key, default) or 0.0)

        def _result_int(key: str, default: int = 0) -> int:
            return int(cast_result.get(key, default) or 0)

        enemy_defeat_reward = _result_float("enemy_defeat_reward")
        ruin_clear_bonus_reward = _result_float("ruin_clear_bonus_reward")
        moves_before_value = (
            float(moves_before_cast)
            if moves_before_cast is not None
            else _result_float("moves_before_cast", float(self.moves or 0.0))
        )
        effect_type_key = (
            "debuff_types" if self._spell_kind_is_offensive(spell_kind) else "buff_types"
        )
        info.update(
            {
                "spell_description": spell_description,
                "spell_level": spell_level,
                "spell_damage": float(spell_spec.get("damage", 0.0) or 0.0),
                "spell_damage_type": str(spell_spec.get("damage_type", "") or "") or None,
                "spell_debuff_type": str(spell_spec.get("debuff_type", "") or "") or None,
                "spell_summon_unit_name": str(spell_spec.get("summon_unit_name", "") or "") or None,
                "spell_buff_type": str(spell_spec.get("buff_type", "") or "") or None,
                "spell_resistance_type": str(spell_spec.get("resistance_type", "") or "") or None,
                "spell_heal_amount": float(spell_spec.get("heal_amount", 0.0) or 0.0),
                "spell_health_delta": int(spell_spec.get("health_delta", 0) or 0),
                "spell_moves_restore_fraction": float(
                    spell_spec.get("moves_restore_fraction", 0.0) or 0.0
                ),
                "spell_heal_total": _result_float("spell_heal_total"),
                "spell_moves_restored": _result_float("spell_moves_restored"),
                "spell_cast_used_this_turn": bool(spell_cast_used_this_turn),
                "spell_is_learned": bool(spell_is_learned),
                "blocked_by_typeoflord": bool(blocked_by_typeoflord),
                "spell_use_costs": dict(mana_costs),
                "target_enemy_position": target_enemy_position,
                "target_enemy_reachable": bool(target_enemy_reachable),
                "target_enemy_description": target_enemy_description,
                "spell_damage_total": _result_float("spell_damage_total"),
                "spell_damage_units_hit": _result_int("spell_damage_units_hit"),
                "spell_damage_units_defeated": _result_int("spell_damage_units_defeated"),
                "spell_damage_units_blocked": _result_int("spell_damage_units_blocked"),
                "spell_units_affected": _result_int("spell_units_affected"),
                "spell_effect_active_ids": list(spell_effect_summary.get("active_spell_ids", [])),
                "spell_effect_types": list(spell_effect_summary.get(effect_type_key, [])),
                "spell_effect_resistance_types": list(
                    spell_effect_summary.get("resistance_types", [])
                ),
                "spell_effect_armor_delta": int(spell_effect_summary.get("armor_delta", 0) or 0),
                "spell_effect_damage_multiplier": float(
                    spell_effect_summary.get("damage_multiplier", 1.0) or 1.0
                ),
                "spell_effect_initiative_multiplier": float(
                    spell_effect_summary.get("initiative_multiplier", 1.0) or 1.0
                ),
                "spell_effect_accuracy_multiplier": float(
                    spell_effect_summary.get("accuracy_multiplier", 1.0) or 1.0
                ),
                "spell_effect_health_delta": int(
                    spell_effect_summary.get("health_delta", 0) or 0
                ),
                "enemy_defeat_reward": enemy_defeat_reward,
                "enemy_defeat_reward_base": (
                    float(self.reward_defeat_enemy) if spell_enemy_defeated else 0.0
                ),
                "blue_exp_reward": 0.0,
                "blue_exp_raw": 0.0,
                "ruin_clear_bonus_reward": ruin_clear_bonus_reward,
                "moves_before_cast": moves_before_value,
                "moves_after_cast": _result_float("moves_after_cast", float(self.moves or 0.0)),
            }
        )
        info.update(detailed_extra)
        return info
    def _finalize_map_spell_step(
        self,
        *,
        reward: float,
        info: Dict[str, object],
        cast_result: Dict[str, object],
    ):
        if bool(cast_result.get("battle_triggered", False)) and self.mode == self.MODE_BATTLE:
            battle_obs = (
                self.battle_env._obs()
                if self.battle_env is not None
                else np.zeros(0, dtype=np.float32)
            )
            reward = self._apply_green_dragon_reached_reward_if_needed(
                self.current_enemy_id,
                reward,
                info,
            )
            reward = self._apply_objective_city_reached_reward_if_needed(
                self.current_enemy_id,
                reward,
                info,
            )
            info["battle_triggered"] = True
            info["mode"] = "battle"
            info["enemy_id"] = self.current_enemy_id
            info["battle_context_kind"] = cast_result.get("battle_context_kind")
            return self._build_obs(battle_obs=battle_obs), reward, False, False, info

        if bool(cast_result.get("spell_enemy_defeated", False)):
            target_enemy_id = cast_result.get("target_enemy_id")
            reward = self._apply_wave_defeat_reward_if_needed(
                target_enemy_id,
                reward,
                info,
            )
            info.update(self._grant_ruin_reward(target_enemy_id))
            info["objective_cities_captured_total"] = sorted(self.captured_objective_cities)
            newly_activated = list(
                cast_result.get("newly_activated_settlement_territories", []) or []
            )
            if newly_activated:
                info["legions_settlement_territories_activated"] = newly_activated
            newly_captured = list(
                cast_result.get("newly_captured_objective_cities", []) or []
            )
            if newly_captured:
                info["captured_objective_cities"] = newly_captured
                if self._campaign_objective_is_cities():
                    info["final_objective_reward"] = float(
                        cast_result.get("final_objective_reward", 0.0) or 0.0
                    )

            green_dragon_objective_reward = float(
                cast_result.get("green_dragon_objective_reward", 0.0) or 0.0
            )
            if green_dragon_objective_reward:
                info["final_objective_reward"] = green_dragon_objective_reward
                info["green_dragon_objective_reward"] = green_dragon_objective_reward
                info["green_dragon_objective_enemy_id"] = int(
                    getattr(self, "OBJECTIVE_ENEMY_ID", self.GREEN_DRAGON_OBJECTIVE_ENEMY_ID)
                )
                info["campaign_objective"] = self.campaign_objective

            campaign_objective_reason = self._campaign_objective_completion_reason(
                target_enemy_id
            )
            if campaign_objective_reason is not None:
                self._log("=== ЗАХВАЧЕНЫ ВСЕ ЦЕЛЕВЫЕ ГОРОДА: ПОБЕДА В КАМПАНИИ ===")
                grid_obs = self._get_grid_obs()
                info["campaign_result"] = "victory"
                info["campaign_victory_reason"] = campaign_objective_reason
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                )

            if (
                self.grid_env.all_enemies_defeated()
                and not self._campaign_objective_is_waves()
            ):
                self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
                grid_obs = self._get_grid_obs()
                info["campaign_result"] = "victory"
                info["campaign_victory_reason"] = "all_enemies_defeated"
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                )

        grid_obs = self._get_grid_obs()
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=False,
            truncated=False,
            info=info,
        )
    @classmethod
    def _spell_untargetable_enemy_ids(cls) -> frozenset[int]:
        blocked_ids: set[int] = set()
        for override_group in (
            VILLAGE_LINKED_STACK_OVERRIDES,
            VILLAGE_INTERNAL_GARRISON_OVERRIDES,
            CAPITAL_INTERNAL_STACK_OVERRIDES,
            RUIN_STACK_OVERRIDES,
        ):
            for entry in override_group:
                try:
                    blocked_ids.add(int(entry.get("enemy_id", -1)))
                except (TypeError, ValueError, AttributeError):
                    continue
        return frozenset(enemy_id for enemy_id in blocked_ids if enemy_id >= 0)
    @classmethod
    def _is_enemy_stack_spell_targetable(cls, enemy_id: Optional[int]) -> bool:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return False
        return normalized_enemy_id not in cls._spell_untargetable_enemy_ids()
    def _enemy_team_has_living_units(self, enemy_id: Optional[int]) -> bool:
        for unit in self._get_enemy_team_state(enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue
            hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            if max_hp > 0.0 and hp > 0.0:
                return True
        return False
    @staticmethod
    def _normalize_accuracy_value(value: object) -> int:
        try:
            return max(0, min(100, int(round(float(value or 0)))))
        except (TypeError, ValueError):
            return 0
    @staticmethod
    def _enemy_map_spell_base_field(stat_name: str) -> str:
        return f"campaign_map_spell_base_{str(stat_name or '').strip()}"
    def _capture_enemy_stack_spell_bases(self, enemy_id: Optional[int]) -> None:
        for unit in self._get_enemy_team_state(enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue
            base_values = {
                "armor": self._normalize_armor_value(unit.get("armor", 0)),
                "damage": self._normalize_damage_value(unit.get("damage", 0)),
                "damage_secondary": self._normalize_damage_value(unit.get("damage_secondary", 0)),
                "initiative_base": self._normalize_damage_value(
                    unit.get("initiative_base", unit.get("initiative", 0))
                ),
                "accuracy": self._normalize_accuracy_value(unit.get("accuracy", 0)),
                "accuracy_secondary": self._normalize_accuracy_value(
                    unit.get("accuracy_secondary", 0)
                ),
            }
            for stat_name, base_value in base_values.items():
                base_field = self._enemy_map_spell_base_field(stat_name)
                if base_field not in unit:
                    unit[base_field] = base_value
    def _restore_enemy_stack_spell_bases(self, enemy_id: Optional[int]) -> None:
        for unit in self._get_enemy_team_state(enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue

            base_armor = self._normalize_armor_value(
                unit.pop(self._enemy_map_spell_base_field("armor"), unit.get("armor", 0))
            )
            base_damage = self._normalize_damage_value(
                unit.pop(self._enemy_map_spell_base_field("damage"), unit.get("damage", 0))
            )
            base_damage_secondary = self._normalize_damage_value(
                unit.pop(
                    self._enemy_map_spell_base_field("damage_secondary"),
                    unit.get("damage_secondary", 0),
                )
            )
            base_initiative = self._normalize_damage_value(
                unit.pop(
                    self._enemy_map_spell_base_field("initiative_base"),
                    unit.get("initiative_base", unit.get("initiative", 0)),
                )
            )
            base_accuracy = self._normalize_accuracy_value(
                unit.pop(self._enemy_map_spell_base_field("accuracy"), unit.get("accuracy", 0))
            )
            base_accuracy_secondary = self._normalize_accuracy_value(
                unit.pop(
                    self._enemy_map_spell_base_field("accuracy_secondary"),
                    unit.get("accuracy_secondary", 0),
                )
            )

            unit["armor"] = base_armor
            unit["damage"] = base_damage
            unit["damage_secondary"] = base_damage_secondary
            unit["initiative_base"] = base_initiative
            unit["accuracy"] = base_accuracy
            unit["accuracy_secondary"] = base_accuracy_secondary

            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            unit["initiative"] = base_initiative if max_hp > 0.0 and current_hp > 0.0 else 0
    def _enemy_stack_spell_effect_summary(self, enemy_id: Optional[int]) -> Dict[str, object]:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            normalized_enemy_id = -1

        active_spell_ids = sorted(
            str(spell_id)
            for spell_id in self.enemy_map_spell_effects.get(normalized_enemy_id, set())
            if str(spell_id)
        )
        specs_by_id = self._combined_map_offensive_spell_specs_by_id()

        armor_delta = 0
        damage_multiplier = 1.0
        initiative_multiplier = 1.0
        accuracy_multiplier = 1.0
        debuff_types: List[str] = []

        for spell_id in active_spell_ids:
            spec = specs_by_id.get(str(spell_id), {})
            if str(spec.get("kind", "") or "") != "debuff":
                continue
            debuff_type = str(spec.get("debuff_type", "") or "").strip()
            if debuff_type:
                debuff_types.append(debuff_type)
            armor_delta += int(spec.get("armor_delta", 0) or 0)
            damage_multiplier *= float(spec.get("damage_multiplier", 1.0) or 1.0)
            initiative_multiplier *= float(spec.get("initiative_multiplier", 1.0) or 1.0)
            accuracy_multiplier *= float(spec.get("accuracy_multiplier", 1.0) or 1.0)

        return {
            "active_spell_ids": active_spell_ids,
            "debuff_types": sorted(set(debuff_types)),
            "armor_delta": int(armor_delta),
            "damage_multiplier": float(damage_multiplier),
            "initiative_multiplier": float(initiative_multiplier),
            "accuracy_multiplier": float(accuracy_multiplier),
        }
    def _recompute_enemy_stack_spell_effects(self, enemy_id: Optional[int]) -> int:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return 0

        active_spell_ids = {
            str(spell_id)
            for spell_id in self.enemy_map_spell_effects.get(normalized_enemy_id, set())
            if str(spell_id)
        }
        if not active_spell_ids:
            self._restore_enemy_stack_spell_bases(normalized_enemy_id)
            return 0

        self._capture_enemy_stack_spell_bases(normalized_enemy_id)
        summary = self._enemy_stack_spell_effect_summary(normalized_enemy_id)

        affected_units = 0
        for unit in self._get_enemy_team_state(normalized_enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue

            base_armor = self._normalize_armor_value(
                unit.get(self._enemy_map_spell_base_field("armor"), unit.get("armor", 0))
            )
            base_damage = self._normalize_damage_value(
                unit.get(self._enemy_map_spell_base_field("damage"), unit.get("damage", 0))
            )
            base_damage_secondary = self._normalize_damage_value(
                unit.get(
                    self._enemy_map_spell_base_field("damage_secondary"),
                    unit.get("damage_secondary", 0),
                )
            )
            base_initiative = self._normalize_damage_value(
                unit.get(
                    self._enemy_map_spell_base_field("initiative_base"),
                    unit.get("initiative_base", unit.get("initiative", 0)),
                )
            )
            base_accuracy = self._normalize_accuracy_value(
                unit.get(self._enemy_map_spell_base_field("accuracy"), unit.get("accuracy", 0))
            )
            base_accuracy_secondary = self._normalize_accuracy_value(
                unit.get(
                    self._enemy_map_spell_base_field("accuracy_secondary"),
                    unit.get("accuracy_secondary", 0),
                )
            )

            unit["armor"] = self._normalize_armor_value(
                float(base_armor) + float(summary["armor_delta"])
            )
            unit["damage"] = self._normalize_damage_value(
                float(base_damage) * float(summary["damage_multiplier"])
            )
            unit["damage_secondary"] = self._normalize_damage_value(
                float(base_damage_secondary) * float(summary["damage_multiplier"])
            )

            new_initiative_base = self._normalize_damage_value(
                float(base_initiative) * float(summary["initiative_multiplier"])
            )
            unit["initiative_base"] = int(new_initiative_base)
            unit["accuracy"] = self._normalize_accuracy_value(
                float(base_accuracy) * float(summary["accuracy_multiplier"])
            )
            unit["accuracy_secondary"] = self._normalize_accuracy_value(
                float(base_accuracy_secondary) * float(summary["accuracy_multiplier"])
            )

            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            is_alive = max_hp > 0.0 and current_hp > 0.0
            unit["initiative"] = int(new_initiative_base) if is_alive else 0
            if is_alive:
                affected_units += 1

        return int(affected_units)
    def _apply_debuff_to_enemy_stack(
        self,
        enemy_id: Optional[int],
        spell_key: str,
    ) -> Tuple[int, Dict[str, object]]:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return 0, self._enemy_stack_spell_effect_summary(enemy_id)

        active_spell_ids = self.enemy_map_spell_effects.setdefault(normalized_enemy_id, set())
        active_spell_ids.add(str(spell_key))
        affected_units = self._recompute_enemy_stack_spell_effects(normalized_enemy_id)
        return int(affected_units), self._enemy_stack_spell_effect_summary(normalized_enemy_id)
    def _clear_all_enemy_map_spell_effects(self) -> int:
        expired_stacks = 0
        for enemy_id, active_spell_ids in list(self.enemy_map_spell_effects.items()):
            if not active_spell_ids:
                continue
            self._restore_enemy_stack_spell_bases(enemy_id)
            expired_stacks += 1
        self.enemy_map_spell_effects = {}
        if expired_stacks > 0:
            self._log(
                f"Истекли временные дебаффы боевых заклинаний на {expired_stacks} вражеских отрядах."
        )
        return int(expired_stacks)
    def _clear_enemy_map_spell_effects_for_enemy(self, enemy_id: Optional[int]) -> None:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return
        if normalized_enemy_id not in self.enemy_map_spell_effects:
            return
        self.enemy_map_spell_effects.pop(normalized_enemy_id, None)
        self._restore_enemy_stack_spell_bases(normalized_enemy_id)
    def _hero_stack_has_living_units(self) -> bool:
        for unit in self._get_blue_state():
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            if max_hp > 0.0 and current_hp > 0.0:
                return True
        return False
    def _blue_stack_has_healable_units(self, heal_amount: float) -> bool:
        if float(heal_amount or 0.0) <= 0.0:
            return False
        for unit in self._get_blue_state():
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            if max_hp > 0.0 and 0.0 < current_hp < max_hp:
                return True
        return False
    def _grid_moves_restore_amount(self, restore_fraction: float) -> float:
        current_moves = max(0.0, float(self.moves or 0.0))
        move_cap = max(0.0, float(self.moves_per_turn or 0.0))
        restore_basis = max(0.0, float(self.MOVES_PER_TURN or 0.0))
        if move_cap <= 0.0 or restore_basis <= 0.0 or float(restore_fraction or 0.0) <= 0.0:
            return 0.0
        restore_amount = max(0.0, round(restore_basis * float(restore_fraction)))
        next_moves = min(move_cap, current_moves + restore_amount)
        return float(max(0.0, next_moves - current_moves))
    def _support_spell_spec_would_apply(
        self,
        spell_key: str,
        spell_spec: Dict[str, object],
    ) -> bool:
        if not isinstance(spell_spec, dict):
            return False
        if not self._hero_stack_has_living_units():
            return False

        spell_kind = str(spell_spec.get("spell_kind", spell_spec.get("kind", "")) or "")
        if spell_kind == "moves":
            restore_fraction = float(spell_spec.get("moves_restore_fraction", 0.0) or 0.0)
            return self._grid_moves_restore_amount(restore_fraction) > 0.0
        if spell_kind == "heal":
            heal_amount = float(spell_spec.get("heal_amount", 0.0) or 0.0)
            return self._blue_stack_has_healable_units(heal_amount)
        if spell_kind in {"buff", "ward", "health_bonus"}:
            normalized_spell_key = str(
                spell_key
                or spell_spec.get("spell_id", "")
                or spell_spec.get("id", "")
                or ""
            )
            return bool(normalized_spell_key) and normalized_spell_key not in self.blue_map_spell_effects
        return False
    def _blue_stack_spell_effect_summary(self) -> Dict[str, object]:
        active_spell_ids = sorted(str(spell_id) for spell_id in self.blue_map_spell_effects if str(spell_id))
        specs_by_id = self._combined_map_support_spell_specs_by_id()

        armor_delta = 0
        damage_multiplier = 1.0
        initiative_multiplier = 1.0
        accuracy_multiplier = 1.0
        health_delta = 0
        buff_types: List[str] = []
        resistance_types: List[str] = []

        for spell_id in active_spell_ids:
            spec = specs_by_id.get(str(spell_id), {})
            spell_kind = str(spec.get("kind", "") or "")
            buff_type = str(spec.get("buff_type", "") or "").strip()
            if spell_kind == "buff":
                if buff_type:
                    buff_types.append(buff_type)
                armor_delta += int(spec.get("armor_delta", 0) or 0)
                damage_multiplier *= float(spec.get("damage_multiplier", 1.0) or 1.0)
                initiative_multiplier *= float(spec.get("initiative_multiplier", 1.0) or 1.0)
                accuracy_multiplier *= float(spec.get("accuracy_multiplier", 1.0) or 1.0)
            elif spell_kind == "health_bonus":
                if buff_type:
                    buff_types.append(buff_type)
                health_delta += int(spec.get("health_delta", 0) or 0)
            elif spell_kind == "ward":
                if buff_type:
                    buff_types.append(buff_type)
                resistance_type = str(spec.get("resistance_type", "") or "").strip()
                if resistance_type:
                    resistance_types.append(resistance_type)

        return {
            "active_spell_ids": active_spell_ids,
            "buff_types": sorted(set(buff_types)),
            "resistance_types": sorted(set(resistance_types)),
            "armor_delta": int(armor_delta),
            "damage_multiplier": float(damage_multiplier),
            "initiative_multiplier": float(initiative_multiplier),
            "accuracy_multiplier": float(accuracy_multiplier),
            "health_delta": int(health_delta),
        }
    def _apply_heal_to_blue_stack(self, heal_amount: float) -> Tuple[int, float]:
        if heal_amount <= 0.0:
            return 0, 0.0

        affected_units = 0
        total_healed = 0.0
        for unit in self._get_blue_state():
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            if max_hp <= 0.0 or current_hp <= 0.0:
                continue
            healed_hp = min(max_hp, current_hp + float(heal_amount))
            healed_amount = max(0.0, healed_hp - current_hp)
            unit["health"] = float(healed_hp)
            unit["hp"] = float(healed_hp)
            if healed_amount > 0.0:
                affected_units += 1
                total_healed += float(healed_amount)
        return int(affected_units), float(total_healed)
    def _restore_grid_moves(self, restore_fraction: float) -> float:
        current_moves = max(0.0, float(self.moves or 0.0))
        move_cap = max(0.0, float(self.moves_per_turn or 0.0))
        restore_basis = max(0.0, float(self.MOVES_PER_TURN or 0.0))
        if move_cap <= 0.0 or restore_basis <= 0.0 or restore_fraction <= 0.0:
            return 0.0
        restore_amount = max(0.0, round(restore_basis * float(restore_fraction)))
        next_moves = min(move_cap, current_moves + restore_amount)
        restored = max(0.0, next_moves - current_moves)
        self.moves = int(round(next_moves))
        return float(restored)
    def _apply_support_spell_to_blue_stack(
        self,
        spell_key: str,
    ) -> Tuple[int, Dict[str, object]]:
        normalized_spell_key = str(spell_key or "")
        if not normalized_spell_key:
            return 0, self._blue_stack_spell_effect_summary()
        self.blue_map_spell_effects.add(normalized_spell_key)
        living_units = sum(
            1
            for unit in self._get_blue_state()
            if float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0) > 0.0
            and float(unit.get("health", 0) or unit.get("hp", 0) or 0.0) > 0.0
        )
        return int(living_units), self._blue_stack_spell_effect_summary()
    def _clear_all_blue_map_spell_effects(self) -> int:
        expired_count = len(self.blue_map_spell_effects)
        self.blue_map_spell_effects = set()
        if expired_count > 0:
            self._log(
                f"Истекли временные эффекты боевых заклинаний на отряде героя ({expired_count} шт.)."
            )
        return int(expired_count)
    @classmethod
    def _unit_blocks_spell_damage(cls, unit: Dict, damage_type: Optional[str]) -> bool:
        normalized_type = str(damage_type or "").strip()
        if not normalized_type:
            return False

        normalized_type_lower = normalized_type.lower()
        immunities_lower = {
            str(value or "").strip().lower() for value in (unit.get("immunity") or [])
        }
        if normalized_type_lower in immunities_lower:
            return True

        resistances_lower = {
            str(value or "").strip().lower() for value in (unit.get("resistance") or [])
        }
        if normalized_type_lower in resistances_lower:
            return True
        return False
    def _apply_damage_to_enemy_stack(
        self,
        enemy_id: Optional[int],
        damage_amount: float,
        damage_type: Optional[str] = None,
    ) -> Tuple[float, int, int, int]:
        actual_damage = 0.0
        units_hit = 0
        defeated_units = 0
        blocked_units = 0

        for unit in self._get_enemy_team_state(enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue
            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            if max_hp <= 0.0 or current_hp <= 0.0:
                continue
            if self._unit_blocks_spell_damage(unit, damage_type):
                blocked_units += 1
                continue
            new_hp = max(0.0, current_hp - float(damage_amount))
            actual_damage += max(0.0, current_hp - new_hp)
            units_hit += 1
            if new_hp <= 0.0:
                defeated_units += 1
                unit["initiative"] = 0
            unit["health"] = float(new_hp)
            if "hp" in unit:
                unit["hp"] = float(new_hp)

        return (
            float(actual_damage),
            int(units_hit),
            int(defeated_units),
            int(blocked_units),
        )
    def _heal_wounded_enemy_teams_for_turn(self) -> Tuple[int, float]:
        healed_units = 0
        healed_total = 0.0

        # Полная регенерация — свойство именно Зелёного дракона (id 31), а не
        # целевого врага кампании: орк из orc_duel лечится обычной долей.
        green_dragon_enemy_id = int(self.GREEN_DRAGON_OBJECTIVE_ENEMY_ID)
        for enemy_id, is_alive in self.grid_env.enemies_alive.items():
            if not bool(is_alive):
                continue
            is_green_dragon = int(enemy_id) == green_dragon_enemy_id
            enemy_position = self.grid_env.enemy_positions.get(int(enemy_id))
            on_player_territory = self._is_player_controlled_territory_tile(enemy_position)
            heal_fraction = 0.05 if on_player_territory else 0.15
            heal_fraction += self._resolve_enemy_regeneration_bonus_percent(enemy_position)
            for unit in self._get_enemy_team_state(enemy_id):
                if self._is_empty_enemy_unit(unit):
                    continue
                current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
                max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
                if max_hp <= 0.0 or current_hp <= 0.0 or current_hp >= max_hp:
                    continue
                if is_green_dragon:
                    new_hp = max_hp
                else:
                    heal_amount = max(1.0, max_hp * float(heal_fraction))
                    new_hp = min(max_hp, current_hp + heal_amount)
                actual_heal = max(0.0, new_hp - current_hp)
                if actual_heal <= 0.0:
                    continue
                unit["health"] = float(new_hp)
                if "hp" in unit:
                    unit["hp"] = float(new_hp)
                healed_units += 1
                healed_total += actual_heal

        if healed_units > 0:
            self._log(
                f"Вражеские отряды восстановили {healed_total:g} HP на {healed_units} юнитах за ход."
            )

        return int(healed_units), float(healed_total)
    def _step_learn_spell(self, action: int):
        idx = action - self.grid_spell_action_start
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        spell_key = None
        spell_description = None
        spell_level = None
        spell_learned = False
        spell_already_learned = False
        spell_learning_locked = bool(self.spell_learning_locked)
        missing_magic_tower = not self._has_magic_tower_built()
        blocked_by_typeoflord = False
        insufficient_mana = False
        mana_costs: Dict[str, float] = {}
        mana_totals_before = self._current_mana_totals()

        if 0 <= idx < len(self.spell_keys):
            spell_key = self.spell_keys[idx]
            spell = self.active_spells.get(spell_key)
            if isinstance(spell, dict):
                spell_description = str(spell.get("description", "") or "").strip()
                try:
                    spell_level = int(spell.get("level", 0) or 0)
                except (TypeError, ValueError):
                    spell_level = 0
                blocked_by_typeoflord = self._is_spell_level_masked_by_typeoflord(spell_level)
                try:
                    spell_already_learned = int(spell.get("learned", 0) or 0) == 1
                except (TypeError, ValueError):
                    spell_already_learned = False
                mana_costs = self._get_spell_learning_costs(spell)
                insufficient_mana = not self._has_mana_for_costs(mana_costs)
                if (
                    not spell_learning_locked
                    and not missing_magic_tower
                    and not blocked_by_typeoflord
                    and not spell_already_learned
                    and not insufficient_mana
                ):
                    self._spend_mana_costs(mana_costs)
                    spell["learned"] = 1
                    self.spell_learning_locked = True
                    spell_learning_locked = True
                    spell_learned = True
                    reward += float(self.reward_spell_learn)
                    if spell_description:
                        self._log("ЗАКЛИНАНИЕ ИЗУЧЕНО")
                        self._log(f'Изучено заклинание: "{spell_description}"')

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "spell_action": True,
            "spell_key": spell_key,
            "spell_description": spell_description,
            "spell_level": spell_level,
            "spell_learned": spell_learned,
            "spell_already_learned": spell_already_learned,
            "spell_learning_locked": spell_learning_locked,
            "missing_magic_tower": missing_magic_tower,
            "blocked_by_typeoflord": blocked_by_typeoflord,
            "insufficient_mana": insufficient_mana,
            "spell_learning_costs": dict(mana_costs),
            "mana_totals_before_spell_learning": dict(mana_totals_before),
            "spell_learning_reward": float(reward if spell_learned else 0.0),
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
    def _step_cast_legion_damage_spell(self, action: int):
        idx = int(action) - int(self.grid_legion_damage_spell_action_start)
        reward = 0.0
        current_specs = self._current_map_offensive_spell_action_specs()

        spell_key = None
        spell_description = None
        spell_level = None
        spell = None
        spell_spec: Dict[str, object] = {}
        spell_kind = ""
        if 0 <= idx < len(current_specs):
            spell_spec = dict(current_specs[idx])
            spell_key = str(spell_spec.get("id", "") or "")
            spell = self.active_spells.get(str(spell_key))
            spell_kind = str(spell_spec.get("kind", "") or "")

        if isinstance(spell, dict):
            spell_description = str(spell.get("description", "") or "").strip()
            try:
                spell_level = int(spell.get("level", 0) or 0)
            except (TypeError, ValueError):
                spell_level = 0

        mana_costs = self._get_spell_use_costs(spell) if isinstance(spell, dict) else {}
        has_map_offensive_spell_actions = bool(current_specs)
        spell_known = bool(spell_key) and self._is_spell_learned(str(spell_key))
        spell_used_this_turn = bool(spell_key) and self._spell_cast_limit_reached_this_turn(
            str(spell_key)
        )
        blocked_by_typeoflord = self._is_spell_blocked_by_typeoflord(spell)
        nearest_enemy = (
            self._get_map_offensive_spell_target(str(spell_key))
            if spell_key
            else None
        )
        target_enemy_id = None if nearest_enemy is None else int(nearest_enemy.get("enemy_id", -1))
        target_position = None if nearest_enemy is None else tuple(nearest_enemy.get("position", ()))
        target_reachable = bool(nearest_enemy.get("reachable", False)) if nearest_enemy else False
        target_description = None if nearest_enemy is None else str(nearest_enemy.get("description", "") or "")

        can_cast = (
            bool(spell_key)
            and isinstance(spell, dict)
            and has_map_offensive_spell_actions
            and spell_known
            and not blocked_by_typeoflord
            and not spell_used_this_turn
            and nearest_enemy is not None
            and self._has_mana_for_costs(mana_costs)
        )

        if can_cast:
            cast_result = self._cast_from_spell_spec(
                source="learned_offensive_spell",
                spell_key=str(spell_key),
                spell_description=spell_description,
                spell_spec=spell_spec,
                mana_costs=mana_costs,
                spend_mana=True,
                enforce_turn_limit=True,
                nearest_targetable_enemy=nearest_enemy
                if not self._map_offensive_spell_targets_untargetable_stacks(str(spell_key))
                else None,
                nearest_any_enemy=nearest_enemy
                if self._map_offensive_spell_targets_untargetable_stacks(str(spell_key))
                else None,
            )
            reward += float(cast_result.get("reward", 0.0) or 0.0)
        else:
            cast_result = {}

        info = self._build_spell_cast_info(
            spell_key=spell_key,
            spell_description=spell_description,
            spell_level=spell_level,
            spell_kind=spell_kind,
            spell_spec=spell_spec,
            cast_result=cast_result,
            can_cast=can_cast,
            mana_costs=mana_costs,
            spell_cast_used_this_turn=spell_used_this_turn,
            spell_is_learned=spell_known,
            blocked_by_typeoflord=blocked_by_typeoflord,
            target_enemy_id=target_enemy_id,
            target_enemy_position=target_position,
            target_enemy_reachable=target_reachable,
            target_enemy_description=target_description,
            minimal_extra={
                "spell_has_map_offensive_actions": bool(has_map_offensive_spell_actions),
            },
            detailed_extra={
                "summon_hero_battle_bonus_pending": bool(
                    self.summon_hero_battle_bonus_pending
                ),
                "summon_hero_battle_bonus_enemy_ids": sorted(
                    int(enemy_id)
                    for enemy_id in self.summon_hero_battle_bonus_enemy_ids_this_turn
                ),
                "spell_is_legions_capital": bool(self._is_legions_capital()),
                "spell_is_undead_hordes_capital": bool(self._is_undead_hordes_capital()),
            },
        )

        return self._finalize_map_spell_step(
            reward=reward,
            info=info,
            cast_result=cast_result,
        )
    def _step_cast_support_spell(self, action: int):
        idx = int(action) - int(self.grid_map_support_spell_action_start)
        reward = 0.0
        current_specs = self._current_map_support_spell_action_specs()

        spell_key = None
        spell_description = None
        spell_level = None
        spell = None
        spell_spec: Dict[str, object] = {}
        spell_kind = ""

        if 0 <= idx < len(current_specs):
            spell_spec = dict(current_specs[idx])
            spell_key = str(spell_spec.get("id", "") or "")
            spell = self.active_spells.get(str(spell_key))
            spell_kind = str(spell_spec.get("kind", "") or "")

        if isinstance(spell, dict):
            spell_description = str(spell.get("description", "") or "").strip()
            try:
                spell_level = int(spell.get("level", 0) or 0)
            except (TypeError, ValueError):
                spell_level = 0

        mana_costs = self._get_spell_use_costs(spell) if isinstance(spell, dict) else {}
        has_map_support_spell_actions = bool(current_specs)
        spell_known = bool(spell_key) and self._is_spell_learned(str(spell_key))
        spell_used_this_turn = bool(spell_key) and self._spell_cast_limit_reached_this_turn(
            str(spell_key)
        )
        blocked_by_typeoflord = self._is_spell_blocked_by_typeoflord(spell)
        can_cast = (
            bool(spell_key)
            and isinstance(spell, dict)
            and has_map_support_spell_actions
            and spell_known
            and not blocked_by_typeoflord
            and not spell_used_this_turn
            and self._support_spell_spec_would_apply(str(spell_key), spell_spec)
            and self._has_mana_for_costs(mana_costs)
        )

        moves_before_cast = float(self.moves or 0.0)

        if can_cast:
            cast_result = self._cast_from_spell_spec(
                source="learned_support_spell",
                spell_key=str(spell_key),
                spell_description=spell_description,
                spell_spec=spell_spec,
                mana_costs=mana_costs,
                spend_mana=True,
                enforce_turn_limit=True,
            )
            reward += float(cast_result.get("reward", 0.0) or 0.0)
        else:
            cast_result = {}

        info = self._build_spell_cast_info(
            spell_key=spell_key,
            spell_description=spell_description,
            spell_level=spell_level,
            spell_kind=spell_kind,
            spell_spec=spell_spec,
            cast_result=cast_result,
            can_cast=can_cast,
            mana_costs=mana_costs,
            spell_cast_used_this_turn=spell_used_this_turn,
            spell_is_learned=spell_known,
            blocked_by_typeoflord=blocked_by_typeoflord,
            moves_before_cast=moves_before_cast,
            minimal_extra={
                "spell_has_map_support_actions": bool(has_map_support_spell_actions),
            },
            detailed_extra={
                "spell_is_empire_capital": bool(self._is_empire_capital()),
            },
        )

        return self._finalize_map_spell_step(
            reward=reward,
            info=info,
            cast_result=cast_result,
        )
    def _step_cast_spell_shop_spell(self, action: int):
        idx = int(action) - int(self.grid_spell_shop_cast_action_start)
        reward = 0.0

        spell_data = self._spell_shop_cast_spell_entry(idx)
        spell_key = str(spell_data.get("spell_id", "") or "")
        spell_name = str(spell_data.get("name", "") or "").strip()
        spell = self._spell_entry_by_id(spell_key)
        offensive_spec = self._map_offensive_spell_spec(spell_key) if spell_key else {}
        support_spec = {} if offensive_spec else self._map_support_spell_spec(spell_key)
        spell_spec: Dict[str, object] = dict(offensive_spec or support_spec)
        spell_kind = str(spell_spec.get("kind", "") or "")

        spell_description = spell_name or spell_key or None
        spell_level = None
        if isinstance(spell, dict):
            spell_description = str(spell.get("description", "") or spell_description or "").strip()
            try:
                spell_level = int(spell.get("level", 0) or 0)
            except (TypeError, ValueError):
                spell_level = 0

        mana_costs = self._get_spell_use_costs(spell) if isinstance(spell, dict) else {}
        spell_purchased = bool(spell_key) and self._spell_shop_spell_owned(spell_key)
        spell_regular_action = bool(spell_key) and self._spell_shop_spell_has_regular_action(spell_key)
        spell_used_this_turn = bool(spell_key) and self._spell_shop_cast_limit_reached_this_turn(
            spell_key
        )

        nearest_enemy = (
            self._get_map_offensive_spell_target(str(spell_key))
            if spell_key and offensive_spec
            else None
        )
        targets_untargetable = bool(
            spell_key
            and offensive_spec
            and self._map_offensive_spell_targets_untargetable_stacks(str(spell_key))
        )
        target_enemy_id = None if nearest_enemy is None else int(nearest_enemy.get("enemy_id", -1))
        target_position = None if nearest_enemy is None else tuple(nearest_enemy.get("position", ()))
        target_reachable = bool(nearest_enemy.get("reachable", False)) if nearest_enemy else False
        target_description = None if nearest_enemy is None else str(nearest_enemy.get("description", "") or "")
        can_cast = (
            bool(spell_key)
            and isinstance(spell, dict)
            and spell_purchased
            and not spell_regular_action
            and not spell_used_this_turn
            and self._can_cast_spell_shop_spell(
                str(spell_key),
                nearest_targetable_enemy=None if targets_untargetable else nearest_enemy,
                nearest_any_enemy=nearest_enemy if targets_untargetable else None,
            )
        )

        moves_before_cast = float(self.moves or 0.0)

        if can_cast:
            cast_result = self._cast_from_spell_spec(
                source="spell_shop",
                spell_key=str(spell_key),
                spell_description=spell_description,
                spell_spec=spell_spec,
                mana_costs=mana_costs,
                spend_mana=True,
                enforce_turn_limit=True,
                nearest_targetable_enemy=None if targets_untargetable else nearest_enemy,
                nearest_any_enemy=nearest_enemy if targets_untargetable else None,
            )
            reward += float(cast_result.get("reward", 0.0) or 0.0)
        else:
            cast_result = {}

        info = self._build_spell_cast_info(
            spell_key=spell_key,
            spell_description=spell_description,
            spell_level=spell_level,
            spell_kind=spell_kind,
            spell_spec=spell_spec,
            cast_result=cast_result,
            can_cast=can_cast,
            mana_costs=mana_costs,
            spell_cast_used_this_turn=spell_used_this_turn,
            spell_is_learned=spell_purchased,
            target_enemy_id=target_enemy_id,
            target_enemy_position=target_position,
            target_enemy_reachable=target_reachable,
            target_enemy_description=target_description,
            moves_before_cast=moves_before_cast,
            minimal_extra={
                "spell_shop_cast_action": True,
                "spell_shop_cast_slot": int(idx),
            },
            detailed_extra={
                "spell_shop_spell_purchased": bool(spell_purchased),
                "spell_shop_spell_has_regular_action": bool(spell_regular_action),
                "spell_has_map_offensive_actions": bool(offensive_spec),
                "spell_has_map_support_actions": bool(support_spec),
            },
        )

        return self._finalize_map_spell_step(
            reward=reward,
            info=info,
            cast_result=cast_result,
        )
    def _step_cast_staff_spell(self, action: int):
        idx = int(action) - int(self.grid_staff_spell_action_start)
        reward = 0.0
        slot_entries = self.staff_spell_action_entries()
        spell_data = dict(slot_entries[idx]) if 0 <= idx < len(slot_entries) else {}
        spell_key = str(spell_data.get("spell_id", "") or "")
        spell = self._spell_entry_by_id(spell_key)
        spell_description = str(
            spell_data.get("spell_name", "")
            or spell_data.get("spell_description", "")
            or spell_key
        ).strip()
        spell_level = int(spell_data.get("spell_level", 0) or 0) if spell_data else None
        spell_kind = str(spell_data.get("spell_kind", "") or "")
        item_name = str(spell_data.get("item_name", "") or "")
        required_item_count = int(spell_data.get("required_item_count", 1) or 1)
        item_owned = bool(item_name) and self._count_hero_item(item_name) >= required_item_count
        hero_is_mage = self._travel_hero_is_mage()
        has_sorcery_lore = self._hero_has_sorcery_lore()
        mana_costs = self._get_spell_use_costs(spell) if isinstance(spell, dict) else {}
        blocked_by_typeoflord = self._is_spell_blocked_by_typeoflord(spell)
        nearest_targetable_enemy = None
        nearest_any_enemy = None
        if self._spell_kind_is_offensive(spell_kind):
            nearest_targetable_enemy = self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=True
            )
            nearest_any_enemy = self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=False
            )
        can_cast = bool(spell_key) and self._can_cast_staff_spell(
            spell_data,
            nearest_targetable_enemy=nearest_targetable_enemy,
            nearest_any_enemy=nearest_any_enemy,
        )

        moves_before_cast = float(self.moves or 0.0)

        if can_cast:
            cast_result = self._cast_from_spell_spec(
                source="staff_spell",
                spell_key=spell_key,
                spell_description=spell_description,
                spell_spec=spell_data,
                mana_costs=mana_costs,
                spend_mana=True,
                enforce_turn_limit=False,
                nearest_targetable_enemy=nearest_targetable_enemy,
                nearest_any_enemy=nearest_any_enemy,
            )
            reward += float(cast_result.get("reward", 0.0) or 0.0)
        else:
            cast_result = {}

        info = self._build_spell_cast_info(
            spell_key=spell_key,
            spell_description=spell_description,
            spell_level=spell_level,
            spell_kind=spell_kind,
            spell_spec=spell_data,
            cast_result=cast_result,
            can_cast=can_cast,
            mana_costs=mana_costs,
            spell_is_learned=bool(spell_key and self._is_spell_learned(spell_key)),
            blocked_by_typeoflord=blocked_by_typeoflord,
            moves_before_cast=moves_before_cast,
            minimal_extra={
                "staff_spell_cast_action": True,
                "staff_spell_cast_slot": int(idx),
                "staff_item_name": item_name,
            },
            detailed_extra={
                "staff_item_owned": bool(item_owned),
                "staff_required_item_count": int(required_item_count),
                "staff_magic_unlocked": bool(self.scroll_magic_unlocked),
                "staff_requires_sorcery_lore": not bool(hero_is_mage),
                "staff_hero_is_mage": bool(hero_is_mage),
                "staff_has_sorcery_lore": bool(has_sorcery_lore),
                "staff_spell_key": spell_key,
                "staff_spell_kind": spell_kind,
            },
        )
        return self._finalize_map_spell_step(
            reward=reward,
            info=info,
            cast_result=cast_result,
        )
    def _step_unlock_scroll_magic(self, action: int):
        reward = 0.0
        scroll_magic_unlocked_before = bool(self.scroll_magic_unlocked)
        supported_entries = self.available_scroll_spell_entries()
        supported_staff_entries = self.available_staff_spell_unlock_entries()
        has_unlock_source = bool(supported_entries or supported_staff_entries)
        can_unlock = (
            int(action) == int(self.GRID_UNLOCK_SCROLL_MAGIC_ACTION)
            and self.mode == self.MODE_GRID
            and not scroll_magic_unlocked_before
            and float(self.gold or 0.0) >= float(self.SCROLL_MAGE_UNLOCK_GOLD_COST)
            and has_unlock_source
        )
        gold_spent = 0.0
        if can_unlock:
            gold_spent = float(self.SCROLL_MAGE_UNLOCK_GOLD_COST)
            self.gold = max(0.0, float(self.gold or 0.0) - gold_spent)
            self.scroll_magic_unlocked = True
            self._log(
                f"Маг для чтения свитков нанят за {gold_spent:g} gold. "
                "Свитки доступны до конца эпизода."
            )
        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "scroll_magic_unlock_action": True,
            "scroll_magic_unlocked_before": bool(scroll_magic_unlocked_before),
            "scroll_magic_unlocked_after": bool(self.scroll_magic_unlocked),
            "scroll_magic_unlock_gold_spent": float(gold_spent),
            "scroll_magic_unlock_available": bool(
                not scroll_magic_unlocked_before and has_unlock_source
            ),
            "scroll_magic_unlock_scroll_slots": int(len(supported_entries)),
            "scroll_magic_unlock_staff_slots": int(len(supported_staff_entries)),
            "scroll_magic_unlock_can_afford": bool(
                float(self.gold or 0.0) + gold_spent
                >= float(self.SCROLL_MAGE_UNLOCK_GOLD_COST)
            ),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }
        grid_obs = self._get_grid_obs()
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=False,
            truncated=False,
            info=info,
        )
    def _step_cast_scroll_spell(self, action: int):
        idx = int(action) - int(self.grid_scroll_cast_action_start)
        reward = 0.0
        slot_entries = self.scroll_cast_slot_entries()
        spell_data = dict(slot_entries[idx]) if 0 <= idx < len(slot_entries) else {}
        spell_key = str(spell_data.get("spell_id", "") or "")
        spell_description = str(
            spell_data.get("spell_name", "")
            or spell_data.get("spell_description", "")
            or spell_key
        ).strip()
        spell_level = int(spell_data.get("spell_level", 0) or 0) if spell_data else None
        spell_kind = str(spell_data.get("spell_kind", "") or "")
        item_name = str(spell_data.get("item_name", "") or "")
        scroll_copies_before = int(self.count_scroll_item(item_name))
        nearest_targetable_enemy = None
        nearest_any_enemy = None
        if self._spell_kind_is_offensive(spell_kind):
            nearest_targetable_enemy = self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=True
            )
            nearest_any_enemy = self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=False
            )
        can_cast = bool(spell_key) and self._can_cast_scroll_spell(
            spell_data,
            nearest_targetable_enemy=nearest_targetable_enemy,
            nearest_any_enemy=nearest_any_enemy,
        )

        if can_cast:
            cast_result = self._cast_from_spell_spec(
                source="scroll",
                spell_key=spell_key,
                spell_description=spell_description,
                spell_spec=spell_data,
                spend_mana=False,
                enforce_turn_limit=False,
                consume_item_name=item_name,
                nearest_targetable_enemy=nearest_targetable_enemy,
                nearest_any_enemy=nearest_any_enemy,
            )
            reward += float(cast_result.get("reward", 0.0) or 0.0)
            scroll_item_consumed = bool(cast_result.get("scroll_item_consumed", False))
            scroll_copies_after = int(cast_result.get("scroll_copies_after", 0) or 0)
        else:
            cast_result = {}
            scroll_item_consumed = False
            scroll_copies_after = scroll_copies_before

        info = self._build_spell_cast_info(
            spell_key=spell_key,
            spell_description=spell_description,
            spell_level=spell_level,
            spell_kind=spell_kind,
            spell_spec=spell_data,
            cast_result=cast_result,
            can_cast=can_cast,
            mana_costs={},
            minimal_extra={
                "scroll_cast_action": True,
                "scroll_cast_slot": int(idx),
                "scroll_item_name": item_name,
            },
            detailed_extra={
                "scroll_spell_key": spell_key,
                "scroll_spell_kind": spell_kind,
                "scroll_item_consumed": bool(scroll_item_consumed),
                "scroll_copies_before": int(scroll_copies_before),
                "scroll_copies_after": int(scroll_copies_after),
            },
        )
        return self._finalize_map_spell_step(
            reward=reward,
            info=info,
            cast_result=cast_result,
        )
    def _finalize_summon_spell_battle(
        self,
        *,
        obs: np.ndarray,
        reward: float,
        info: Dict[str, object],
        winner: Optional[str],
    ):
        enemy_id = self.current_enemy_id
        info["blue_exp_reward"] = 0.0
        info["blue_exp_raw"] = 0.0
        info["blue_survival_reward"] = 0.0
        info["blue_alive_ratio"] = 0.0
        info["blue_hp_ratio"] = 0.0
        info["enemy_defeat_reward"] = 0.0
        info["enemy_defeat_reward_base"] = 0.0
        info["ruin_clear_bonus_reward"] = 0.0

        if winner == "blue":
            try:
                normalized_enemy_id = int(enemy_id)
            except (TypeError, ValueError):
                normalized_enemy_id = None
            if normalized_enemy_id is not None:
                self.summon_hero_battle_bonus_enemy_ids_this_turn.discard(normalized_enemy_id)
                self.summon_hero_battle_bonus_pending = bool(
                    self.summon_hero_battle_bonus_enemy_ids_this_turn
                )
            enemy_reward = self._compute_enemy_defeat_reward(enemy_id)
            reward += enemy_reward
            info["enemy_defeat_reward"] = float(enemy_reward)
            info["enemy_defeat_reward_base"] = float(self.reward_defeat_enemy)
            if int(enemy_id or -1) in self.RUIN_REWARD_BY_ENEMY_ID:
                info["ruin_clear_bonus_reward"] = float(self.reward_ruin_clear_bonus)

            self._log(
                f"=== ПРИЗВАННЫЙ ЮНИТ ПОБЕДИЛ В БОЮ ПРОТИВ ВРАГА {enemy_id}! ==="
            )
            self.grid_env.mark_enemy_defeated(enemy_id)
            self._clear_enemy_map_spell_effects_for_enemy(enemy_id)
            reward = self._apply_wave_defeat_reward_if_needed(
                enemy_id,
                reward,
                info,
            )

            objective_cities_captured_before = len(self.captured_objective_cities)
            newly_captured_objective_cities = self._capture_objective_city_if_cleared(enemy_id)
            newly_activated_settlement_territories = (
                self._activate_legions_settlement_territory_if_cleared(enemy_id)
            )
            info.update(self._grant_ruin_reward(enemy_id))
            info["objective_cities_captured_total"] = sorted(self.captured_objective_cities)
            if newly_activated_settlement_territories:
                info["legions_settlement_territories_activated"] = list(
                    newly_activated_settlement_territories
                )
                for settlement_name in newly_activated_settlement_territories:
                    self._log(
                        f"=== ЗЕМЛЯ ЛЕГИОНОВ НАЧИНАЕТ РАСПРОСТРАНЯТЬСЯ ИЗ ПОСЕЛЕНИЯ {settlement_name} ==="
                    )

            if newly_captured_objective_cities and self._campaign_objective_is_cities():
                final_objective_reward = self._compute_final_objective_reward(
                    len(newly_captured_objective_cities),
                    captured_before=objective_cities_captured_before,
                )
                reward += final_objective_reward
                info["captured_objective_cities"] = list(newly_captured_objective_cities)
                info["final_objective_reward"] = float(final_objective_reward)

            reward = self._apply_green_dragon_objective_reward_if_needed(
                enemy_id,
                reward,
                info,
            )
            self.mode = self.MODE_GRID
            self.battle_env = None
            self._clear_current_battle_context()
            info["battle_result"] = "victory"
            info["mode"] = "grid"
            info["agent_pos"] = self._restore_agent_to_battle_origin()
            info["enemies_alive"] = dict(self.grid_env.enemies_alive)

            campaign_objective_reason = self._campaign_objective_completion_reason(enemy_id)
            if campaign_objective_reason is not None:
                self._log("=== ЗАХВАЧЕНЫ ВСЕ ЦЕЛЕВЫЕ ГОРОДА: ПОБЕДА В КАМПАНИИ ===")
                grid_obs = self._get_grid_obs()
                info["campaign_result"] = "victory"
                info["campaign_victory_reason"] = campaign_objective_reason
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                )

            if (
                self.grid_env.all_enemies_defeated()
                and not self._campaign_objective_is_waves()
            ):
                self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
                grid_obs = self._get_grid_obs()
                info["campaign_result"] = "victory"
                info["campaign_victory_reason"] = "all_enemies_defeated"
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                )

            grid_obs = self._get_grid_obs()
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=reward,
                terminated=False,
                truncated=False,
                info=info,
            )

        self._save_enemy_state_from_battle(enemy_id)
        if winner == "red":
            self._log(
                f"=== ПРИЗВАННЫЙ ЮНИТ ПОТЕРПЕЛ ПОРАЖЕНИЕ В БОЮ ПРОТИВ ВРАГА {enemy_id}! ==="
            )
            info["battle_result"] = "defeat"
        else:
            self._log(
                f"=== БОЙ ПРИЗЫВНОГО ЮНИТА ПРОТИВ ВРАГА {enemy_id} ЗАВЕРШЁН БЕЗ ПОБЕДИТЕЛЯ ==="
            )
            info["battle_result"] = "timeout"

        self.mode = self.MODE_GRID
        self.battle_env = None
        self._clear_current_battle_context()
        info["mode"] = "grid"
        info["agent_pos"] = self._restore_agent_to_battle_origin()
        info["enemies_alive"] = dict(self.grid_env.enemies_alive)
        info["summon_enemy_state_saved"] = True

        grid_obs = self._get_grid_obs()
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=False,
            truncated=False,
            info=info,
        )
    def _get_spells_for_capital(self, capital: Optional[int]) -> Dict:
        """Возвращает словарь заклинаний по значению 'столица'."""
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            capital_value = None

        if capital_value == 2:
            return deepcopy(legions_spells_d2)
        if capital_value == 3:
            return deepcopy(mountain_clans_spells_d2)
        if capital_value == 4:
            return deepcopy(undead_hordes_spells_d2)
        if capital_value == 5:
            return deepcopy(elves_spells_d2)
        return deepcopy(empire_spells_d2)
