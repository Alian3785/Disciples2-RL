# campaign_env_inventory.py
"""Inventory, consumable, and equipment helpers for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class _TrackedInventoryList(list):
    """Invalidates inventory caches on list mutation."""

    def __init__(self, owner: object, iterable=()):
        self._owner = owner
        super().__init__(iterable or ())

    def _changed(self) -> None:
        owner = getattr(self, "_owner", None)
        if owner is not None and hasattr(owner, "_mark_inventory_changed"):
            owner._mark_inventory_changed()

    def append(self, item) -> None:
        super().append(item)
        self._changed()

    def extend(self, iterable) -> None:
        super().extend(iterable)
        self._changed()

    def insert(self, index, item) -> None:
        super().insert(index, item)
        self._changed()

    def clear(self) -> None:
        super().clear()
        self._changed()

    def pop(self, index=-1):
        item = super().pop(index)
        self._changed()
        return item

    def remove(self, item) -> None:
        super().remove(item)
        self._changed()

    def reverse(self) -> None:
        super().reverse()
        self._changed()

    def sort(self, *args, **kwargs) -> None:
        super().sort(*args, **kwargs)
        self._changed()

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self._changed()

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        self._changed()

    def __iadd__(self, iterable):
        result = super().__iadd__(iterable)
        self._changed()
        return result


class CampaignInventoryMixin:
    @property
    def heroitems(self) -> List[object]:
        return getattr(self, "_heroitems", _TrackedInventoryList(self))

    @heroitems.setter
    def heroitems(self, value) -> None:
        if isinstance(value, _TrackedInventoryList) and getattr(value, "_owner", None) is self:
            self._heroitems = value
        else:
            self._heroitems = _TrackedInventoryList(self, value or ())
        self._mark_inventory_changed()

    def _mark_inventory_changed(self) -> None:
        self._inventory_version = int(getattr(self, "_inventory_version", 0) or 0) + 1
        self.inventory_dirty = True
        self._inventory_cache_valid = False
        self._mark_battle_items_dirty()
        self._mark_equipment_dirty()
        self._merchant_autosale_checked_inventory_version = None
        self._merchant_autosale_checked_position = None

    def _mark_battle_items_dirty(self) -> None:
        self.battle_items_dirty = True
        self._battle_items_sync_signature = None
        self._battle_equip_action_mask_signature = None
        self._battle_equip_action_mask_cache = ()
        self._equipped_hero_items_sync_signature = None

    def _mark_equipment_dirty(self) -> None:
        self.equipment_dirty = True
        self._equipment_dirty = True
        self._equipment_refresh_signature = None
        self._equipped_hero_items_sync_signature = None
        self._equipped_artifact_items_sync_signature = None
        self._equipped_banner_items_sync_signature = None
        self._equipped_book_items_sync_signature = None
        self._mark_boots_dirty()

    def _mark_boots_dirty(self) -> None:
        self.boots_dirty = True
        self._equipped_boot_items_sync_signature = None

    @classmethod
    def _is_battle_orb_item_name(cls, item_name: str) -> bool:
        normalized_name = str(item_name or "").strip()
        if not normalized_name:
            return False
        normalized_lower = normalized_name.lower()
        return normalized_lower.startswith("orb of ") or normalized_lower.endswith(" orb")

    @classmethod
    def _is_battle_talisman_item_name(cls, item_name: str) -> bool:
        normalized_name = str(item_name or "").strip()
        if not normalized_name:
            return False
        normalized_lower = normalized_name.lower()
        return normalized_lower.startswith("talisman of ") or normalized_lower.endswith(
            " talisman"
        )

    @classmethod
    def _is_battle_magic_item_name(cls, item_name: str) -> bool:
        return cls._is_battle_orb_item_name(item_name) or cls._is_battle_talisman_item_name(
            item_name
        )

    @classmethod
    @lru_cache(maxsize=None)
    def _potion_lookup_tables(
        cls,
    ) -> Tuple[
        Dict[str, Dict[str, object]],
        Dict[str, str],
        Dict[str, Tuple[str, ...]],
    ]:
        definitions: Dict[str, Dict[str, object]] = {}
        alias_to_canonical: Dict[str, str] = {}
        aliases_by_canonical: Dict[str, Tuple[str, ...]] = {}
        for definition in tuple(getattr(cls, "POTION_ITEM_DEFINITIONS", ()) or ()):
            if not isinstance(definition, dict):
                continue
            canonical_name = str(definition.get("name", "") or "").strip()
            if not canonical_name:
                continue
            normalized_definition = dict(definition)
            normalized_definition["name"] = canonical_name
            definitions[canonical_name] = normalized_definition

            alias_names = {canonical_name}
            alias_to_canonical[canonical_name] = canonical_name
            for alias_name in tuple(definition.get("aliases", ()) or ()):
                normalized_alias = str(alias_name or "").strip()
                if normalized_alias:
                    alias_to_canonical[normalized_alias] = canonical_name
                    alias_names.add(normalized_alias)
            aliases_by_canonical[canonical_name] = tuple(alias_names)

        return definitions, alias_to_canonical, aliases_by_canonical

    @classmethod
    def _potion_definitions_by_canonical(cls) -> Dict[str, Dict[str, object]]:
        return cls._potion_lookup_tables()[0]

    @classmethod
    def _potion_alias_to_canonical(cls) -> Dict[str, str]:
        return cls._potion_lookup_tables()[1]

    @classmethod
    def _canonical_potion_item_name(cls, item_name: object) -> str:
        normalized_name = str(item_name or "").strip()
        if not normalized_name:
            return ""
        return str(cls._potion_alias_to_canonical().get(normalized_name, "") or "")

    @classmethod
    def _potion_item_definition(cls, item_name: object) -> Dict[str, object]:
        canonical_name = cls._canonical_potion_item_name(item_name)
        if not canonical_name:
            return {}
        return dict(cls._potion_definitions_by_canonical().get(canonical_name, {}))

    @classmethod
    def _is_potion_item_name(cls, item_name: object) -> bool:
        return bool(cls._canonical_potion_item_name(item_name))

    @classmethod
    def _potion_item_aliases_for_name(cls, item_name: object) -> Tuple[str, ...]:
        canonical_name = cls._canonical_potion_item_name(item_name)
        if not canonical_name:
            return ()
        return cls._potion_lookup_tables()[2].get(canonical_name, ())

    def _scenario_merchant_potion_item_entries(self) -> Tuple[Dict[str, object], ...]:
        entries: List[Dict[str, object]] = []
        for item_data in tuple(getattr(self, "MERCHANT_BUY_ITEMS", ()) or ()):
            if not isinstance(item_data, dict):
                continue
            item_name = str(item_data.get("name", "") or "").strip()
            if not item_name or not self._is_potion_item_name(item_name):
                continue
            try:
                stock = int(item_data.get("stock", 0) or 0)
            except (TypeError, ValueError):
                stock = 0
            if stock <= 0:
                continue
            entries.append(dict(item_data))
        return tuple(entries)

    def _scenario_merchant_potion_item_names(self) -> Tuple[str, ...]:
        return tuple(
            str(item_data.get("name", "") or "")
            for item_data in self._scenario_merchant_potion_item_entries()
        )

    def _scenario_potion_item_names(self) -> Tuple[str, ...]:
        present_names: set[str] = set()

        def add_if_potion(raw_item_name: object) -> None:
            canonical_name = self._canonical_potion_item_name(raw_item_name)
            if canonical_name:
                present_names.add(canonical_name)

        for item_names in getattr(self, "_static_chests", {}).values():
            for item_name in tuple(item_names or ()):
                add_if_potion(item_name)

        for item_name in tuple(getattr(self._map, "starting_hero_items", ()) or ()):
            add_if_potion(item_name)

        for reward_data in tuple(getattr(self, "RUIN_REWARD_BY_ENEMY_ID", {}).values()):
            if isinstance(reward_data, dict):
                add_if_potion(reward_data.get("item", ""))

        for item_data in self._scenario_merchant_potion_item_entries():
            add_if_potion(item_data.get("name", ""))

        if int(getattr(self, "MAX_HEALING_BOTTLES", 0) or 0) > 0:
            add_if_potion(self.BONUS_SMALL_HEAL_ITEM_NAME)
        if int(getattr(self, "MAX_HEAL_BOTTLES", 0) or 0) > 0:
            add_if_potion(self.BONUS_LARGE_HEAL_ITEM_NAME)
        if int(getattr(self, "MAX_REVIVE_BOTTLES", 0) or 0) > 0:
            add_if_potion(self.BONUS_REVIVE_ITEM_NAME)

        ordered_names: List[str] = []
        for definition in tuple(getattr(self, "POTION_ITEM_DEFINITIONS", ()) or ()):
            canonical_name = str((definition or {}).get("name", "") or "").strip()
            if canonical_name and canonical_name in present_names:
                ordered_names.append(canonical_name)
        return tuple(ordered_names)

    def _scenario_battle_potion_equip_names(self) -> Tuple[str, ...]:
        scenario_potion_names = set(getattr(self, "scenario_potion_item_names", ()) or ())
        base_names = (
            self.BONUS_SMALL_HEAL_ITEM_NAME,
            self.BONUS_LARGE_HEAL_ITEM_NAME,
            self.HEALING_OINTMENT_ITEM_NAME,
            self.BONUS_REVIVE_ITEM_NAME,
        )
        return tuple(
            item_name
            for item_name in base_names
            if self._canonical_potion_item_name(item_name) in scenario_potion_names
        )

    def _refresh_potion_item_names(self) -> Tuple[str, ...]:
        self.scenario_potion_item_names = self._scenario_potion_item_names()
        self.scenario_merchant_potion_item_names = self._scenario_merchant_potion_item_names()
        self.scenario_battle_potion_equip_names = self._scenario_battle_potion_equip_names()
        self.GRID_POTION_USE_ACTION_COUNT = (
            len(self.scenario_potion_item_names) * len(self.GRID_POTION_USE_POSITIONS)
        )
        self.GRID_MERCHANT_POTION_BUY_ACTION_COUNT = len(
            self.scenario_merchant_potion_item_names
        )
        self.GRID_EQUIP_BATTLE_POTION_ACTION_COUNT = len(
            self.scenario_battle_potion_equip_names
        )
        return self.scenario_potion_item_names

    def _potion_action_start_for_item(self, item_name: object) -> int:
        canonical_name = self._canonical_potion_item_name(item_name)
        if not canonical_name:
            return int(getattr(self, "GRID_POTION_USE_ACTION_START", 9))
        try:
            potion_idx = list(self.scenario_potion_item_names).index(canonical_name)
        except ValueError:
            return int(getattr(self, "GRID_POTION_USE_ACTION_START", 9))
        return int(self.GRID_POTION_USE_ACTION_START) + int(potion_idx) * len(
            self.GRID_POTION_USE_POSITIONS
        )

    @classmethod
    def _is_staff_spell_item_name(cls, item_name: str) -> bool:
        normalized_name = str(item_name or "").strip()
        return bool(normalized_name) and normalized_name in set(cls.STAFF_SPELL_ITEM_NAMES)

    def _scenario_staff_spell_item_names(self) -> Tuple[str, ...]:
        staff_item_names: List[str] = []

        def add_if_staff(raw_item_name: object) -> None:
            normalized_name = str(raw_item_name or "").strip()
            if not self._is_staff_spell_item_name(normalized_name):
                return
            staff_item_names.append(normalized_name)

        for item_names in getattr(self, "_static_chests", {}).values():
            for item_name in tuple(item_names or ()):
                add_if_staff(item_name)

        for reward_data in tuple(getattr(self, "RUIN_REWARD_BY_ENEMY_ID", {}).values()):
            if not isinstance(reward_data, dict):
                continue
            add_if_staff(reward_data.get("item", ""))

        for item_data in tuple(getattr(self, "MERCHANT_BUY_ITEMS", ()) or ()):
            if not isinstance(item_data, dict):
                continue
            try:
                stock = int(item_data.get("stock", 0) or 0)
            except (TypeError, ValueError):
                stock = 0
            if stock <= 0:
                continue
            add_if_staff(item_data.get("name", ""))

        return tuple(staff_item_names)

    @classmethod
    def _battle_item_default_uses_left(cls, item_name: str) -> Optional[int]:
        if cls._is_battle_talisman_item_name(item_name):
            return int(cls.TALISMAN_USES_PER_ITEM)
        normalized_name = str(item_name or "")
        if normalized_name:
            return 1
        return None

    def _scenario_battle_magic_item_names(self) -> Tuple[str, ...]:
        magic_item_names: List[str] = []
        seen_names: set[str] = set()

        def add_if_battle_magic(raw_item_name: object) -> None:
            normalized_name = str(raw_item_name or "").strip()
            if not self._is_battle_magic_item_name(normalized_name):
                return
            if normalized_name in seen_names:
                return
            seen_names.add(normalized_name)
            magic_item_names.append(normalized_name)

        for item_names in getattr(self, "_static_chests", {}).values():
            for item_name in tuple(item_names or ()):
                add_if_battle_magic(item_name)

        for reward_data in tuple(getattr(self, "RUIN_REWARD_BY_ENEMY_ID", {}).values()):
            if isinstance(reward_data, dict):
                add_if_battle_magic(reward_data.get("item", ""))

        for item_data in tuple(getattr(self, "MERCHANT_BUY_ITEMS", ()) or ()):
            if not isinstance(item_data, dict):
                continue
            try:
                stock = int(item_data.get("stock", 0) or 0)
            except (TypeError, ValueError):
                stock = 0
            if stock <= 0:
                continue
            add_if_battle_magic(item_data.get("name", ""))

        return tuple(magic_item_names)


    @classmethod
    def _is_book_item_name(cls, item_name: str) -> bool:
        normalized_name = str(item_name or "").strip()
        return bool(normalized_name) and normalized_name in set(cls.BOOK_ITEM_NAMES)

    def _scenario_book_item_names(self) -> Tuple[str, ...]:
        book_item_names: List[str] = []
        seen_names: set[str] = set()

        def add_if_book(raw_item_name: object) -> None:
            normalized_name = str(raw_item_name or "").strip()
            if not self._is_book_item_name(normalized_name):
                return
            if normalized_name in seen_names:
                return
            seen_names.add(normalized_name)
            book_item_names.append(normalized_name)

        for item_names in getattr(self, "_static_chests", {}).values():
            for item_name in tuple(item_names or ()):
                add_if_book(item_name)

        for reward_data in tuple(getattr(self, "RUIN_REWARD_BY_ENEMY_ID", {}).values()):
            if isinstance(reward_data, dict):
                add_if_book(reward_data.get("item", ""))

        for item_data in tuple(getattr(self, "MERCHANT_BUY_ITEMS", ()) or ()):
            if not isinstance(item_data, dict):
                continue
            try:
                stock = int(item_data.get("stock", 0) or 0)
            except (TypeError, ValueError):
                stock = 0
            if stock <= 0:
                continue
            add_if_book(item_data.get("name", ""))

        return tuple(book_item_names)

    def _refresh_book_item_names(self) -> Tuple[str, ...]:
        self.scenario_book_item_names = self._scenario_book_item_names()
        return self.scenario_book_item_names

    def _refresh_battle_equippable_item_names(self) -> Tuple[str, ...]:
        base_names = tuple(
            str(item_name)
            for item_name in getattr(
                self,
                "scenario_battle_potion_equip_names",
                self.BASE_BATTLE_EQUIPPABLE_ITEM_NAMES,
            )
        )
        scenario_magic_item_names = self._scenario_battle_magic_item_names()
        battle_item_names: List[str] = []
        seen_names: set[str] = set()
        for item_name in (*base_names, *scenario_magic_item_names):
            normalized_name = str(item_name or "").strip()
            if not normalized_name or normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)
            battle_item_names.append(normalized_name)

        self.scenario_battle_magic_item_names = tuple(scenario_magic_item_names)
        self.scenario_battle_orb_item_names = tuple(scenario_magic_item_names)
        self.BATTLE_EQUIPPABLE_ITEM_NAMES = tuple(battle_item_names)
        return self.BATTLE_EQUIPPABLE_ITEM_NAMES

    def _hire_reward_value(self) -> float:
        return max(0.0, 3.0 * float(self.reward_defeat_enemy))
    def _max_heal_bottles_available(self) -> int:
        return int(self.MAX_HEAL_BOTTLES) + max(0, int(self.extra_heal_bottles or 0))
    def _heal_bottles_left(self) -> int:
        return max(0, self._max_heal_bottles_available() - int(self.heal_bottles_used or 0))
    def _max_healing_bottles_available(self) -> int:
        return int(self.MAX_HEALING_BOTTLES) + max(0, int(self.extra_healing_bottles or 0))
    def _healing_bottles_left(self) -> int:
        return max(0, self._max_healing_bottles_available() - int(self.healing_bottles_used or 0))
    def _max_revive_bottles_available(self) -> int:
        return int(self.MAX_REVIVE_BOTTLES) + max(0, int(self.extra_revive_bottles or 0))
    def _revive_bottles_left(self) -> int:
        return max(0, self._max_revive_bottles_available() - int(self.revive_bottles_used or 0))
    @classmethod
    def _battle_item_effect_definitions(cls) -> Dict[str, Dict[str, object]]:
        effects: Dict[str, Dict[str, object]] = {
            cls.BONUS_SMALL_HEAL_ITEM_NAME: {
                "kind": "heal",
                "amount": float(cls.HEALING_BOTTLE_AMOUNT),
            },
            cls.BONUS_LARGE_HEAL_ITEM_NAME: {
                "kind": "heal",
                "amount": float(cls.HEAL_BOTTLE_AMOUNT),
            },
            cls.HEALING_OINTMENT_ITEM_NAME: {
                "kind": "heal",
                "amount": float(cls.HEALING_OINTMENT_AMOUNT),
            },
            cls.BONUS_REVIVE_ITEM_NAME: {
                "kind": "revive",
                "amount": 1.0,
            },
        }

        summon_orbs = {
            "Squire Orb": "Скваер",
            "Angel Orb": "Ангел",
            "Venerable Warrior Orb": "Старый ветеран",
            "Imp Orb": "Бес",
            "Infernal Knight Orb": "Адский рыцарь",
            "Incubus Orb": "Инкуб",
            "Zombie Orb": "Зомби",
            "Skeleton Champion Orb": "Скелет рыцарь",
            "Vampire Orb": "Вампир",
            "Lich Orb": "Лич",
            "Elder Vampire Orb": "Высший вампир",
            "Elf Lord Orb": "Лесной лорд",
            "Goblin Orb": "Гоблин",
            "Orc Orb": "Орк",
            "Lizard Man Orb": "Человек ящер",
        }
        for item_name, unit_name in summon_orbs.items():
            effects[item_name] = {
                "kind": "summon",
                "target_team": "blue",
                "target_state": "empty_or_dead",
                "summon_unit_name": unit_name,
            }

        effects.update(
            {
                "Orb of Healing": {
                    "kind": "heal",
                    "amount": 100.0,
                    "target_team": "blue",
                    "target_state": "alive",
                },
                "Orb of Restoration": {
                    "kind": "heal",
                    "amount": 30.0,
                    "target_team": "blue",
                    "target_state": "alive",
                    "scope": "party",
                },
                "Orb of Regeneration": {
                    "kind": "heal",
                    "amount": 75.0,
                    "target_team": "blue",
                    "target_state": "alive",
                    "scope": "party",
                },
                "Orb of Life": {
                    "kind": "revive",
                    "amount": 1.0,
                    "target_team": "blue",
                    "target_state": "dead",
                },
                "Orb of Vigor": {
                    "kind": "damage_buff",
                    "damage_multiplier": 1.25,
                    "target_team": "blue",
                    "target_state": "alive",
                },
                "Orb of Strength": {
                    "kind": "damage_buff",
                    "damage_multiplier": 1.50,
                    "target_team": "blue",
                    "target_state": "alive",
                },
                "Orb of Rage": {
                    "kind": "extra_turn",
                    "target_team": "blue",
                    "target_state": "alive",
                },
                "Orb of Fire": {
                    "kind": "damage",
                    "amount": 25.0,
                    "damage_type": "Fire",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Inferno": {
                    "kind": "damage",
                    "amount": 75.0,
                    "damage_type": "Fire",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Lightning": {
                    "kind": "damage",
                    "amount": 50.0,
                    "damage_type": "Air",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Thunder": {
                    "kind": "damage",
                    "amount": 100.0,
                    "damage_type": "Air",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Earth": {
                    "kind": "damage",
                    "amount": 25.0,
                    "damage_type": "Earth",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Stone Rain": {
                    "kind": "damage",
                    "amount": 75.0,
                    "damage_type": "Earth",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Water": {
                    "kind": "damage",
                    "amount": 25.0,
                    "damage_type": "Water",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Icefall": {
                    "kind": "damage",
                    "amount": 75.0,
                    "damage_type": "Water",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Nosferat": {
                    "kind": "drain",
                    "amount": 25.0,
                    "damage_type": "Death",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Vampire": {
                    "kind": "drain",
                    "amount": 75.0,
                    "damage_type": "Death",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Elder Vampires": {
                    "kind": "drain",
                    "amount": 75.0,
                    "damage_type": "Death",
                    "target_team": "red",
                    "target_state": "alive",
                    "share_leftover": True,
                },
                "Orb of Poison": {
                    "kind": "dot",
                    "dot_type": "poison",
                    "amount": 20.0,
                    "damage_type": "Poison",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Venom": {
                    "kind": "dot",
                    "dot_type": "poison",
                    "amount": 30.0,
                    "damage_type": "Poison",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Bane": {
                    "kind": "dot",
                    "dot_type": "poison",
                    "amount": 10.0,
                    "damage_type": "Poison",
                    "target_team": "red",
                    "target_state": "alive",
                    "scope": "party",
                },
                "Orb of Thanatos": {
                    "kind": "dot",
                    "dot_type": "poison",
                    "amount": 30.0,
                    "damage_type": "Poison",
                    "target_team": "red",
                    "target_state": "alive",
                    "scope": "party",
                },
                "Orb of Frost": {
                    "kind": "dot",
                    "dot_type": "water",
                    "amount": 10.0,
                    "damage_type": "Water",
                    "target_team": "red",
                    "target_state": "alive",
                    "scope": "party",
                },
                "Orb of Freezing": {
                    "kind": "dot",
                    "dot_type": "water",
                    "amount": 30.0,
                    "damage_type": "Water",
                    "target_team": "red",
                    "target_state": "alive",
                    "scope": "party",
                },
                "Orb of Flame": {
                    "kind": "dot",
                    "dot_type": "burn",
                    "amount": 15.0,
                    "damage_type": "Fire",
                    "target_team": "red",
                    "target_state": "alive",
                    "scope": "party",
                },
                "Orb of Fear": {
                    "kind": "paralysis",
                    "damage_type": "Mind",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Witches": {
                    "kind": "transform",
                    "damage_type": "Mind",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Lycanthropy": {
                    "kind": "lycanthropy",
                    "damage_type": "Mind",
                    "transform_unit_name": "Оборотень",
                    "target_team": "red",
                    "target_state": "alive",
                },
                "Orb of Weakening": {
                    "kind": "debuff_damage",
                    "damage_multiplier": 0.67,
                    "damage_type": "Mind",
                    "target_team": "red",
                    "target_state": "alive",
                },
            }
        )
        summon_talismans = {
            "Squire Talisman": "Скваер",
            "Angel Talisman": "Ангел",
            "Venerable Warrior Talisman": "Старый ветеран",
            "Imp Talisman": "Бес",
            "Infernal Knight Talisman": "Адский рыцарь",
            "Incubus Talisman": "Инкуб",
            "Zombie Talisman": "Зомби",
            "Skeleton Champion Talisman": "Скелет рыцарь",
            "Vampire Talisman": "Вампир",
            "Lich Talisman": "Лич",
            "Elder Vampire Talisman": "Высший вампир",
            "Elf Lord Talisman": "Лесной лорд",
            "Goblin Talisman": "Гоблин",
            "Orc Talisman": "Орк",
            "Lizard Man Talisman": "Человек ящер",
        }
        for item_name, unit_name in summon_talismans.items():
            effects[item_name] = {
                "kind": "summon",
                "target_team": "blue",
                "target_state": "empty_or_dead",
                "summon_unit_name": unit_name,
                "item_category": "talisman",
                "uses_per_item": int(cls.TALISMAN_USES_PER_ITEM),
            }

        talisman_orb_pairs = {
            "Talisman of Healing": "Orb of Healing",
            "Talisman of Restoration": "Orb of Restoration",
            "Talisman of Regeneration": "Orb of Regeneration",
            "Talisman of Life": "Orb of Life",
            "Talisman of Vigor": "Orb of Vigor",
            "Talisman of Strength": "Orb of Strength",
            "Talisman of Fire": "Orb of Fire",
            "Talisman of Inferno": "Orb of Inferno",
            "Talisman of Nosferat": "Orb of Nosferat",
            "Talisman of Vampire": "Orb of Vampire",
            "Talisman of Elder Vampires": "Orb of Elder Vampires",
            "Talisman of Lightning": "Orb of Lightning",
            "Talisman of Thunder": "Orb of Thunder",
            "Talisman of Earth": "Orb of Earth",
            "Talisman of Stone Rain": "Orb of Stone Rain",
            "Talisman of Water": "Orb of Water",
            "Talisman of Icefall": "Orb of Icefall",
            "Talisman of Poison": "Orb of Poison",
            "Talisman of Venom": "Orb of Venom",
            "Talisman of Bane": "Orb of Bane",
            "Talisman of Thanatos": "Orb of Thanatos",
            "Talisman of Frost": "Orb of Frost",
            "Talisman of Freezing": "Orb of Freezing",
            "Talisman of Fear": "Orb of Fear",
            "Talisman of Witches": "Orb of Witches",
        }
        for talisman_name, orb_name in talisman_orb_pairs.items():
            if orb_name not in effects:
                continue
            talisman_effect = dict(effects[orb_name])
            talisman_effect["item_category"] = "talisman"
            talisman_effect["uses_per_item"] = int(cls.TALISMAN_USES_PER_ITEM)
            effects[talisman_name] = talisman_effect

        effects["Talisman of Nightmare"] = {
            "kind": "fear",
            "damage_type": "Mind",
            "target_team": "red",
            "target_state": "alive",
            "scope": "single",
            "item_category": "talisman",
            "uses_per_item": int(cls.TALISMAN_USES_PER_ITEM),
        }
        return effects
    def _battle_items_counter_signature(self) -> Tuple[int, int, int, int, int, int]:
        return (
            int(getattr(self, "healing_bottles_used", 0) or 0),
            int(getattr(self, "heal_bottles_used", 0) or 0),
            int(getattr(self, "revive_bottles_used", 0) or 0),
            int(getattr(self, "extra_healing_bottles", 0) or 0),
            int(getattr(self, "extra_heal_bottles", 0) or 0),
            int(getattr(self, "extra_revive_bottles", 0) or 0),
        )
    def _blue_roster_equipment_signature(
        self,
        units: Optional[List[Dict]] = None,
    ) -> Tuple[Tuple[object, ...], ...]:
        roster = units if units is not None else getattr(self, "blue_team_state", None)
        signature: List[Tuple[object, ...]] = []
        for unit in roster or []:
            if not isinstance(unit, dict):
                continue
            try:
                position = int(unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                position = -1
            try:
                level = int(round(float(unit.get("Level", 0) or 0)))
            except (TypeError, ValueError):
                level = 0
            signature.append(
                (
                    position,
                    str(unit.get("name", unit.get("\u043a\u0442\u043e", "")) or ""),
                    bool(self._is_hero_unit(unit)),
                    bool(self._is_travel_unit_alive(unit)),
                    level,
                    tuple(sorted(self._hero_ability_tokens(unit))),
                )
            )
        return tuple(signature)
    def _equipment_state_signature(
        self,
        units: Optional[List[Dict]] = None,
    ) -> Tuple[object, ...]:
        return (
            int(getattr(self, "_inventory_version", 0) or 0),
            self._battle_items_counter_signature(),
            int(getattr(self, "typeoflord", 0) or 0),
            tuple(getattr(self, "equipped_artifact_items", ()) or ()),
            tuple(getattr(self, "equipped_banner_items", ()) or ()),
            tuple(getattr(self, "equipped_book_items", ()) or ()),
            tuple(getattr(self, "equipped_boot_items", ()) or ()),
            self._blue_roster_equipment_signature(units=units),
        )
    def _equipped_hero_items_state_signature(self) -> Tuple[object, ...]:
        return (
            int(getattr(self, "_inventory_version", 0) or 0),
            self._battle_items_counter_signature(),
            int(getattr(self, "typeoflord", 0) or 0),
            tuple(getattr(self, "equipped_hero_items", ()) or ()),
            tuple(getattr(self, "equipped_hero_item_uses_left", ()) or ()),
            tuple(getattr(self, "equipped_hero_item_uses_left_item_names", ()) or ()),
            tuple(getattr(self, "equipped_book_items", ()) or ()),
            self._blue_roster_equipment_signature(),
        )
    def _equipped_book_items_state_signature(self) -> Tuple[object, ...]:
        return (
            int(getattr(self, "_inventory_version", 0) or 0),
            tuple(getattr(self, "equipped_book_items", ()) or ()),
            self._blue_roster_equipment_signature(),
        )
    def _equipped_artifact_items_state_signature(
        self,
        units: Optional[List[Dict]] = None,
    ) -> Tuple[object, ...]:
        return (
            int(getattr(self, "_inventory_version", 0) or 0),
            tuple(getattr(self, "equipped_artifact_items", ()) or ()),
            self._blue_roster_equipment_signature(units=units),
        )
    def _equipped_banner_items_state_signature(
        self,
        units: Optional[List[Dict]] = None,
    ) -> Tuple[object, ...]:
        return (
            int(getattr(self, "_inventory_version", 0) or 0),
            tuple(getattr(self, "equipped_banner_items", ()) or ()),
            self._blue_roster_equipment_signature(units=units),
        )
    def _equipped_boot_items_state_signature(
        self,
        units: Optional[List[Dict]] = None,
    ) -> Tuple[object, ...]:
        return (
            int(getattr(self, "_inventory_version", 0) or 0),
            tuple(getattr(self, "equipped_boot_items", ()) or ()),
            self._blue_roster_equipment_signature(units=units),
        )
    def _battle_equip_action_mask_state_signature(self) -> Tuple[object, ...]:
        return (
            int(getattr(self, "_inventory_version", 0) or 0),
            self._battle_items_counter_signature(),
            int(getattr(self, "typeoflord", 0) or 0),
            bool(getattr(self, "battle_item_equip_used_this_turn", False)),
            tuple(getattr(self, "equipped_hero_items", ()) or ()),
            tuple(getattr(self, "equipped_hero_item_uses_left", ()) or ()),
            tuple(getattr(self, "equipped_hero_item_uses_left_item_names", ()) or ()),
            tuple(getattr(self, "equipped_book_items", ()) or ()),
            self._blue_roster_equipment_signature(),
        )
    def _battle_equip_item_available_count(self, item_name: str) -> int:
        normalized_name = str(item_name or "")
        if normalized_name == self.BONUS_SMALL_HEAL_ITEM_NAME:
            return int(self._healing_bottles_left())
        if normalized_name == self.BONUS_LARGE_HEAL_ITEM_NAME:
            return int(self._heal_bottles_left())
        if normalized_name == self.BONUS_REVIVE_ITEM_NAME:
            return int(self._revive_bottles_left())
        if normalized_name == self.HEALING_OINTMENT_ITEM_NAME:
            return int(self._count_hero_item(self.HEALING_OINTMENT_ITEM_NAME))
        if normalized_name in self.BATTLE_EQUIPPABLE_ITEM_NAMES:
            return int(self._count_hero_item(normalized_name))
        return 0
    def _potion_available_count(self, item_name: object) -> int:
        canonical_name = self._canonical_potion_item_name(item_name)
        if not canonical_name:
            return 0
        if canonical_name == self.BONUS_SMALL_HEAL_ITEM_NAME:
            return int(self._healing_bottles_left())
        if canonical_name == self.BONUS_LARGE_HEAL_ITEM_NAME:
            return int(self._heal_bottles_left())
        if canonical_name == self.BONUS_REVIVE_ITEM_NAME:
            return int(self._revive_bottles_left())
        return int(self._count_hero_item(canonical_name))
    @classmethod
    def _counter_backed_battle_item_names(cls) -> Tuple[str, str, str]:
        return (
            cls.BONUS_SMALL_HEAL_ITEM_NAME,
            cls.BONUS_LARGE_HEAL_ITEM_NAME,
            cls.BONUS_REVIVE_ITEM_NAME,
        )
    def _counter_backed_battle_item_counts(self) -> Dict[str, int]:
        return {
            self.BONUS_SMALL_HEAL_ITEM_NAME: int(self._healing_bottles_left()),
            self.BONUS_LARGE_HEAL_ITEM_NAME: int(self._heal_bottles_left()),
            self.BONUS_REVIVE_ITEM_NAME: int(self._revive_bottles_left()),
        }
    def _sync_counter_backed_battle_items(self) -> None:
        sync_signature = self._battle_items_counter_signature()
        if (
            not bool(getattr(self, "battle_items_dirty", True))
            and getattr(self, "_battle_items_sync_signature", None) == sync_signature
        ):
            return
        target_counts = self._counter_backed_battle_item_counts()
        counter_item_names = set(target_counts)
        current_counts = {item_name: 0 for item_name in counter_item_names}
        synced_items: List[object] = []
        changed = False

        for entry in list(getattr(self, "heroitems", []) or []):
            item_name = self._hero_item_name(entry)
            if item_name not in counter_item_names:
                synced_items.append(entry)
                continue

            if current_counts[item_name] < int(target_counts.get(item_name, 0) or 0):
                synced_items.append(entry)
                current_counts[item_name] += 1
            else:
                changed = True

        for item_name in self._counter_backed_battle_item_names():
            target_count = int(target_counts.get(item_name, 0) or 0)
            while current_counts[item_name] < target_count:
                synced_items.append(self._make_hero_item_entry(item_name))
                current_counts[item_name] += 1
                changed = True

        if changed or len(synced_items) != len(getattr(self, "heroitems", []) or []):
            self.heroitems = synced_items
        self.battle_items_dirty = False
        self._battle_items_sync_signature = sync_signature
    def _count_equipped_battle_items(
        self,
        item_name: str,
        *,
        exclude_slot: Optional[int] = None,
    ) -> int:
        normalized_name = str(item_name or "")
        count = 0
        for slot_index, equipped_name in enumerate(self.equipped_hero_items):
            if exclude_slot is not None and int(slot_index) == int(exclude_slot):
                continue
            if str(equipped_name or "") == normalized_name:
                count += 1
        return count
    def _sync_equipped_hero_items(self) -> None:
        self._sync_equipped_book_items()
        sync_signature = self._equipped_hero_items_state_signature()
        if getattr(self, "_equipped_hero_items_sync_signature", None) == sync_signature:
            return

        normalized_slots: List[Optional[str]] = [None] * int(self.BATTLE_EQUIP_SLOTS)
        normalized_uses_left: List[Optional[int]] = [None] * int(self.BATTLE_EQUIP_SLOTS)
        normalized_uses_left_item_names: List[Optional[str]] = [None] * int(
            self.BATTLE_EQUIP_SLOTS
        )
        reserved: Dict[str, int] = {}
        current_slots = list(self.equipped_hero_items or [])
        current_uses_left = list(getattr(self, "equipped_hero_item_uses_left", []) or [])
        current_uses_left_item_names = list(
            getattr(self, "equipped_hero_item_uses_left_item_names", []) or []
        )
        for slot_index in range(int(self.BATTLE_EQUIP_SLOTS)):
            item_name = (
                str(current_slots[slot_index] or "")
                if slot_index < len(current_slots) and current_slots[slot_index] is not None
                else ""
            )
            if item_name not in self.BATTLE_EQUIPPABLE_ITEM_NAMES:
                continue
            if not self._battle_magic_item_access_allowed(item_name):
                continue
            available_count = int(self._battle_equip_item_available_count(item_name))
            already_reserved = int(reserved.get(item_name, 0))
            if already_reserved >= available_count:
                continue
            default_uses_left = self._battle_item_default_uses_left(item_name)
            uses_left = default_uses_left
            item_name_matches_saved_charge = (
                not current_uses_left_item_names
                or (
                    slot_index < len(current_uses_left_item_names)
                    and str(current_uses_left_item_names[slot_index] or "") == item_name
                )
            )
            if item_name_matches_saved_charge and slot_index < len(current_uses_left):
                try:
                    existing_uses_left = int(current_uses_left[slot_index])
                except (TypeError, ValueError):
                    existing_uses_left = 0
                if existing_uses_left > 0:
                    uses_left = existing_uses_left
            if self._is_battle_talisman_item_name(item_name):
                try:
                    uses_left = max(1, min(int(self.TALISMAN_USES_PER_ITEM), int(uses_left or 0)))
                except (TypeError, ValueError):
                    uses_left = int(self.TALISMAN_USES_PER_ITEM)
            normalized_slots[slot_index] = item_name
            normalized_uses_left[slot_index] = uses_left
            normalized_uses_left_item_names[slot_index] = item_name
            reserved[item_name] = already_reserved + 1
        self.equipped_hero_items = normalized_slots
        self.equipped_hero_item_uses_left = normalized_uses_left
        self.equipped_hero_item_uses_left_item_names = normalized_uses_left_item_names
        self._equipped_hero_items_sync_signature = (
            self._equipped_hero_items_state_signature()
        )
    def _battle_equip_slot_for_item(self, item_name: str) -> Optional[int]:
        normalized_name = str(item_name or "")
        if normalized_name not in self.BATTLE_EQUIPPABLE_ITEM_NAMES:
            return None
        if not self._battle_magic_item_access_allowed(normalized_name):
            return None
        if self.battle_item_equip_used_this_turn:
            return None
        self._sync_counter_backed_battle_items()
        self._sync_equipped_hero_items()
        slot_index = None
        for candidate_index in range(int(self.BATTLE_EQUIP_SLOTS)):
            equipped_name = ""
            if candidate_index < len(self.equipped_hero_items):
                equipped_name = str(self.equipped_hero_items[candidate_index] or "")
            if not equipped_name:
                slot_index = candidate_index
                break
        if slot_index is None:
            return None
        available_count = int(self._battle_equip_item_available_count(normalized_name))
        reserved = self._count_equipped_battle_items(normalized_name)
        if available_count <= reserved:
            return None
        return int(slot_index)
    @staticmethod
    def _normalize_battle_equip_item_arg(item_name: object, maybe_item_name: object = None) -> str:
        return str(maybe_item_name if maybe_item_name is not None else item_name or "")
    def _can_equip_battle_item(self, item_name: object, maybe_item_name: object = None) -> bool:
        normalized_name = self._normalize_battle_equip_item_arg(item_name, maybe_item_name)
        for candidate_name, allowed in zip(
            self.BATTLE_EQUIPPABLE_ITEM_NAMES,
            self._battle_equip_action_mask_values(),
        ):
            if str(candidate_name or "") == normalized_name:
                return bool(allowed)
        return False
    def _battle_equip_action_mask_values(self) -> Tuple[bool, ...]:
        cached_signature = getattr(self, "_battle_equip_action_mask_signature", None)
        current_signature = self._battle_equip_action_mask_state_signature()
        cached_values = tuple(getattr(self, "_battle_equip_action_mask_cache", ()) or ())
        if (
            cached_signature == current_signature
            and len(cached_values) == len(self.BATTLE_EQUIPPABLE_ITEM_NAMES)
        ):
            return cached_values

        if bool(getattr(self, "battle_item_equip_used_this_turn", False)):
            result = tuple(False for _ in self.BATTLE_EQUIPPABLE_ITEM_NAMES)
            self._battle_equip_action_mask_signature = current_signature
            self._battle_equip_action_mask_cache = result
            return result

        self._sync_counter_backed_battle_items()
        self._sync_equipped_hero_items()

        slot_available = any(
            not str(
                (
                    self.equipped_hero_items[idx]
                    if idx < len(getattr(self, "equipped_hero_items", []) or [])
                    else ""
                )
                or ""
            )
            for idx in range(int(self.BATTLE_EQUIP_SLOTS))
        )
        if not slot_available:
            result = tuple(False for _ in self.BATTLE_EQUIPPABLE_ITEM_NAMES)
            self._battle_equip_action_mask_signature = (
                self._battle_equip_action_mask_state_signature()
            )
            self._battle_equip_action_mask_cache = result
            return result

        reserved_counts: Dict[str, int] = {}
        for equipped_name in tuple(getattr(self, "equipped_hero_items", ()) or ()):
            normalized_equipped_name = str(equipped_name or "")
            if not normalized_equipped_name:
                continue
            reserved_counts[normalized_equipped_name] = (
                int(reserved_counts.get(normalized_equipped_name, 0) or 0) + 1
            )

        values: List[bool] = []
        for item_name in self.BATTLE_EQUIPPABLE_ITEM_NAMES:
            normalized_name = str(item_name or "")
            if not self._battle_magic_item_access_allowed(normalized_name):
                values.append(False)
                continue
            available_count = int(self._battle_equip_item_available_count(normalized_name))
            reserved_count = int(reserved_counts.get(normalized_name, 0) or 0)
            values.append(available_count > reserved_count)

        result = tuple(bool(value) for value in values)
        self._battle_equip_action_mask_signature = (
            self._battle_equip_action_mask_state_signature()
        )
        self._battle_equip_action_mask_cache = result
        return result
    def _equip_battle_item(self, item_name: object, maybe_item_name: object = None) -> Optional[int]:
        normalized_name = self._normalize_battle_equip_item_arg(item_name, maybe_item_name)
        slot_index = self._battle_equip_slot_for_item(normalized_name)
        if slot_index is None:
            return None
        if len(getattr(self, "equipped_hero_item_uses_left", []) or []) != int(
            self.BATTLE_EQUIP_SLOTS
        ):
            self.equipped_hero_item_uses_left = [None] * int(self.BATTLE_EQUIP_SLOTS)
        self.equipped_hero_items[int(slot_index)] = str(normalized_name)
        self.equipped_hero_item_uses_left[int(slot_index)] = self._battle_item_default_uses_left(
            normalized_name
        )
        self._sync_equipped_hero_items()
        return int(slot_index)
    def _consume_counter_backed_item(self, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        if normalized_name == self.BONUS_SMALL_HEAL_ITEM_NAME:
            self._sync_counter_backed_battle_items()
            if self._healing_bottles_left() <= 0:
                return False
            self.healing_bottles_used = min(
                self.healing_bottles_used + 1,
                self._max_healing_bottles_available(),
            )
            self._sync_counter_backed_battle_items()
            return True
        if normalized_name == self.BONUS_LARGE_HEAL_ITEM_NAME:
            self._sync_counter_backed_battle_items()
            if self._heal_bottles_left() <= 0:
                return False
            self.heal_bottles_used = min(
                self.heal_bottles_used + 1,
                self._max_heal_bottles_available(),
            )
            self._sync_counter_backed_battle_items()
            return True
        if normalized_name == self.BONUS_REVIVE_ITEM_NAME:
            self._sync_counter_backed_battle_items()
            if self._revive_bottles_left() <= 0:
                return False
            self.revive_bottles_used = min(
                self.revive_bottles_used + 1,
                self._max_revive_bottles_available(),
            )
            self._sync_counter_backed_battle_items()
            return True
        return False
    def _consume_battle_equipable_item(self, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        consumed = False
        if normalized_name == self.HEALING_OINTMENT_ITEM_NAME:
            consumed = self._consume_hero_item(self.HEALING_OINTMENT_ITEM_NAME)
        elif normalized_name in self._counter_backed_battle_item_names():
            consumed = self._consume_counter_backed_item(normalized_name)
        elif normalized_name in self.BATTLE_EQUIPPABLE_ITEM_NAMES:
            consumed = self._consume_hero_item(normalized_name)
        self._sync_equipped_hero_items()
        return bool(consumed)
    @staticmethod
    def _normalize_damage_value(value: object) -> int:
        try:
            return max(0, int(round(float(value or 0))))
        except (TypeError, ValueError):
            return 0
    @staticmethod
    def _normalize_health_value(value: object) -> int:
        try:
            return max(0, int(round(float(value or 0))))
        except (TypeError, ValueError):
            return 0
    @staticmethod
    def _hero_item_name(entry: object) -> str:
        if isinstance(entry, HeroInventoryItem):
            return str(entry)
        if isinstance(entry, dict):
            return str(entry.get("name", "") or "")
        return str(entry or "")
    @staticmethod
    def _hero_item_gold(entry: object) -> int:
        if isinstance(entry, HeroInventoryItem):
            try:
                return max(0, int(getattr(entry, "gold", 0) or 0))
            except (TypeError, ValueError):
                return 0
        if isinstance(entry, dict):
            try:
                return max(0, int(entry.get("gold", 0) or 0))
            except (TypeError, ValueError):
                return 0
        return 0
    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_item_definitions_by_name(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(entry.get("item_name", "") or ""): dict(entry)
            for entry in SCROLL_ITEM_DEFINITIONS
            if str(entry.get("item_name", "") or "")
        }
    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_item_alias_to_canonical(cls) -> Dict[str, str]:
        alias_map: Dict[str, str] = {
            str(item_name): str(item_name)
            for item_name in cls._scroll_item_definitions_by_name().keys()
        }
        for alias_name, canonical_name in SCROLL_ITEM_ALIASES.items():
            alias_map[str(alias_name or "")] = str(canonical_name or "")
        return alias_map
    @classmethod
    def _canonical_scroll_item_name(cls, item_name: str) -> str:
        return str(
            cls._scroll_item_alias_to_canonical().get(str(item_name or ""), "") or ""
        )
    def _scenario_scroll_item_names(self) -> Tuple[str, ...]:
        present_names: set[str] = set()

        def add_if_scroll(raw_item_name: object) -> None:
            canonical_name = self._canonical_scroll_item_name(str(raw_item_name or ""))
            if canonical_name:
                present_names.add(canonical_name)

        for item_names in getattr(self, "_static_chests", {}).values():
            for item_name in tuple(item_names or ()):
                add_if_scroll(item_name)

        for item_name in tuple(getattr(self._map, "starting_scroll_items", ()) or ()):
            add_if_scroll(item_name)

        for reward_data in tuple(getattr(self, "RUIN_REWARD_BY_ENEMY_ID", {}).values()):
            if isinstance(reward_data, dict):
                add_if_scroll(reward_data.get("item", ""))

        for item_data in tuple(getattr(self, "MERCHANT_BUY_ITEMS", ()) or ()):
            if not isinstance(item_data, dict):
                continue
            try:
                stock = int(item_data.get("stock", 0) or 0)
            except (TypeError, ValueError):
                stock = 0
            if stock <= 0:
                continue
            add_if_scroll(item_data.get("name", ""))

        ordered_names: List[str] = []
        seen_names: set[str] = set()
        for definition in SCROLL_ITEM_DEFINITIONS:
            if not bool(definition.get("supported", False)):
                continue
            canonical_name = self._canonical_scroll_item_name(
                str(definition.get("item_name", "") or "")
            )
            if not canonical_name or canonical_name not in present_names:
                continue
            if canonical_name in seen_names:
                continue
            seen_names.add(canonical_name)
            ordered_names.append(canonical_name)
        return tuple(ordered_names)
    def _scenario_scroll_spell_entries(self) -> Tuple[Dict[str, object], ...]:
        definition_by_name = self._scroll_item_definitions_by_name()
        entries: List[Dict[str, object]] = []
        for item_name in tuple(getattr(self, "scenario_scroll_item_names", ()) or ()):
            canonical_name = self._canonical_scroll_item_name(str(item_name or ""))
            definition = definition_by_name.get(canonical_name, {})
            if not isinstance(definition, dict) or not bool(
                definition.get("supported", False)
            ):
                continue
            entries.append(dict(definition))
        return tuple(entries)
    def _refresh_scroll_item_names(self) -> Tuple[str, ...]:
        self.scenario_scroll_item_names = self._scenario_scroll_item_names()
        self.scenario_scroll_spell_entries = self._scenario_scroll_spell_entries()
        self.GRID_SCROLL_CAST_ACTION_COUNT = len(self.scenario_scroll_spell_entries)
        if hasattr(self, "_inventory_cache_valid"):
            self._invalidate_inventory_cache()
        return self.scenario_scroll_item_names
    def _invalidate_inventory_cache(self) -> None:
        self._mark_inventory_changed()
        self._inventory_cache_valid = False
        self._inventory_cache_signature = ()
        self._inventory_cache_version = -1
        self._hero_item_counts_cache = {}
        self._scroll_item_counts_cache = {}
        self._scroll_inventory_entries_cache = ()
        self._available_scroll_spell_entries_cache = ()
        self._scroll_cast_slot_entries_cache = ()
        self._total_scroll_count_cache = 0
    def _ensure_inventory_cache(self) -> None:
        current_version = int(getattr(self, "_inventory_version", 0) or 0)
        if (
            self._inventory_cache_valid
            and int(getattr(self, "_inventory_cache_version", -1) or -1) == current_version
        ):
            return
        current_signature = tuple(self._hero_item_name(entry) for entry in self.heroitems)

        hero_item_counts: Dict[str, int] = {}
        scroll_item_counts: Dict[str, int] = {}
        for item_name in current_signature:
            hero_item_counts[item_name] = hero_item_counts.get(item_name, 0) + 1

            canonical_scroll_name = self._canonical_scroll_item_name(item_name)
            if canonical_scroll_name:
                scroll_item_counts[canonical_scroll_name] = (
                    scroll_item_counts.get(canonical_scroll_name, 0) + 1
                )

        definition_by_name = self._scroll_item_definitions_by_name()
        inventory_entries: List[Dict[str, object]] = []
        available_entries: List[Dict[str, object]] = []
        for definition in SCROLL_ITEM_DEFINITIONS:
            item_name = str(definition.get("item_name", "") or "")
            canonical_name = self._canonical_scroll_item_name(item_name)
            if not canonical_name:
                continue

            copies = int(scroll_item_counts.get(canonical_name, 0) or 0)
            if copies <= 0:
                continue

            base_definition = definition_by_name.get(canonical_name, {})
            if not isinstance(base_definition, dict):
                continue

            scroll_entry = dict(base_definition)
            scroll_entry["copies"] = copies
            inventory_entries.append(scroll_entry)
            if bool(scroll_entry.get("supported", False)):
                available_entries.append(scroll_entry)

        self._hero_item_counts_cache = hero_item_counts
        self._scroll_item_counts_cache = scroll_item_counts
        self._scroll_inventory_entries_cache = tuple(inventory_entries)
        self._available_scroll_spell_entries_cache = tuple(available_entries)
        scenario_cast_entries: List[Dict[str, object]] = []
        scenario_entries = tuple(getattr(self, "scenario_scroll_spell_entries", ()) or ())
        if not scenario_entries:
            scenario_entries = tuple(
                entry
                for entry in available_entries
                if bool(entry.get("supported", False))
            )
        for definition in scenario_entries:
            item_name = str(definition.get("item_name", "") or "")
            canonical_name = self._canonical_scroll_item_name(item_name)
            if not canonical_name:
                continue
            scroll_entry = dict(definition)
            scroll_entry["copies"] = int(scroll_item_counts.get(canonical_name, 0) or 0)
            scenario_cast_entries.append(scroll_entry)
        self._scroll_cast_slot_entries_cache = tuple(scenario_cast_entries)
        self._total_scroll_count_cache = int(sum(scroll_item_counts.values()))
        self._inventory_cache_signature = current_signature
        self._inventory_cache_version = current_version
        self._inventory_cache_valid = True
        self.inventory_dirty = False
    def scroll_item_definition(self, item_name: str) -> Dict[str, object]:
        canonical_name = self._canonical_scroll_item_name(item_name)
        if not canonical_name:
            return {}
        definition = self._scroll_item_definitions_by_name().get(canonical_name, {})
        if not isinstance(definition, dict):
            return {}
        return dict(definition)
    def count_scroll_item(self, item_name: str) -> int:
        canonical_name = self._canonical_scroll_item_name(item_name)
        if not canonical_name:
            return 0
        self._ensure_inventory_cache()
        return int(self._scroll_item_counts_cache.get(canonical_name, 0) or 0)
    def available_scroll_spell_entries(self) -> Tuple[Dict[str, object], ...]:
        self._ensure_inventory_cache()
        return self._available_scroll_spell_entries_cache
    def scroll_cast_slot_entries(self) -> Tuple[Dict[str, object], ...]:
        self._ensure_inventory_cache()
        return self._scroll_cast_slot_entries_cache
    @classmethod
    def _hero_item_gold_value_by_name(cls, item_name: str) -> int:
        canonical_scroll_name = cls._canonical_scroll_item_name(item_name)
        if canonical_scroll_name:
            try:
                return max(
                    0,
                    int(
                        cls._scroll_item_definitions_by_name()
                        .get(canonical_scroll_name, {})
                        .get("gold_value", 0)
                        or 0
                    ),
                )
            except (TypeError, ValueError):
                return 0
        potion_definition = cls._potion_item_definition(item_name)
        if potion_definition:
            try:
                return max(0, int(potion_definition.get("gold_value", 0) or 0))
            except (TypeError, ValueError):
                return 0
        try:
            return max(0, int(cls.CHEST_ITEM_GOLD_VALUES.get(str(item_name or ""), 0) or 0))
        except (TypeError, ValueError):
            return 0
    @classmethod
    def _make_hero_item_entry(cls, item_name: str) -> HeroInventoryItem:
        normalized_name = str(item_name or "")
        return HeroInventoryItem(
            normalized_name,
            cls._hero_item_gold_value_by_name(normalized_name),
        )
    def _hero_item_display_name(self, entry: object) -> str:
        return self._hero_item_name(entry)
    @classmethod
    def _is_artifact_item_name(cls, item_name: str) -> bool:
        normalized_name = str(item_name or "").strip().lower()
        if not normalized_name:
            return False
        return "(artifact)" in normalized_name or "(артефакт)" in normalized_name
    @classmethod
    def _is_banner_item_name(cls, item_name: str) -> bool:
        normalized_name = str(item_name or "").strip()
        if not normalized_name:
            return False
        return normalized_name in set(cls.BANNER_ITEM_NAMES)
    @classmethod
    def _is_boot_item_name(cls, item_name: str) -> bool:
        normalized_name = str(item_name or "").strip()
        if not normalized_name:
            return False
        return normalized_name in set(cls.BOOT_ITEM_NAMES)
    def _sync_equipped_banner_items(self, units: Optional[List[Dict]] = None) -> None:
        sync_signature = self._equipped_banner_items_state_signature(units=units)
        if (
            getattr(self, "_equipped_banner_items_sync_signature", None)
            == sync_signature
        ):
            return
        normalized_slots: List[Optional[str]] = [None] * int(self.BANNER_EQUIP_SLOTS)
        if not self._hero_has_banner_bearer(units=units):
            self.equipped_banner_items = normalized_slots
            self._equipped_banner_items_sync_signature = (
                self._equipped_banner_items_state_signature(units=units)
            )
            return

        available_banner_names = {
            self._hero_item_name(entry)
            for entry in self.heroitems
            if self._is_banner_item_name(self._hero_item_name(entry))
        }
        selected_banner = next(
            (
                item_name
                for item_name in self.BANNER_EQUIP_PRIORITY
                if item_name in available_banner_names
            ),
            None,
        )
        if selected_banner:
            normalized_slots[0] = str(selected_banner)
        self.equipped_banner_items = normalized_slots
        self._equipped_banner_items_sync_signature = (
            self._equipped_banner_items_state_signature(units=units)
        )
    def _sync_equipped_boot_items(self, units: Optional[List[Dict]] = None) -> None:
        sync_signature = self._equipped_boot_items_state_signature(units=units)
        if getattr(self, "_equipped_boot_items_sync_signature", None) == sync_signature:
            return
        normalized_slots: List[Optional[str]] = [None] * int(self.BOOTS_EQUIP_SLOTS)
        hero = self._resolve_travel_hero(units=units)
        if hero is None or not self._is_travel_unit_alive(hero):
            self.equipped_boot_items = normalized_slots
            self.boots_dirty = False
            self._equipped_boot_items_sync_signature = (
                self._equipped_boot_items_state_signature(units=units)
            )
            return
        if not self._hero_has_marching_lore(units=units):
            self.equipped_boot_items = normalized_slots
            self.boots_dirty = False
            self._equipped_boot_items_sync_signature = (
                self._equipped_boot_items_state_signature(units=units)
            )
            return

        ranked_boots: List[Tuple[int, int, int, str]] = []
        for inventory_index, entry in enumerate(getattr(self, "heroitems", []) or []):
            item_name = self._hero_item_name(entry)
            if not self._is_boot_item_name(item_name):
                continue
            effect = self._boot_effect_definition(item_name)
            ranked_boots.append(
                (
                    -int(effect.get("move_bonus", 0) or 0),
                    -int(self._hero_item_gold_value_by_name(item_name)),
                    int(inventory_index),
                    item_name,
                )
            )

        ranked_boots.sort()
        for slot_index, (_, _, _, item_name) in enumerate(
            ranked_boots[: int(self.BOOTS_EQUIP_SLOTS)]
        ):
            normalized_slots[slot_index] = item_name

        self.equipped_boot_items = normalized_slots
        self.boots_dirty = False
        self._equipped_boot_items_sync_signature = (
            self._equipped_boot_items_state_signature(units=units)
        )
    def _sync_equipped_book_items(self) -> None:
        sync_signature = self._equipped_book_items_state_signature()
        if getattr(self, "_equipped_book_items_sync_signature", None) == sync_signature:
            return
        normalized_slots: List[Optional[str]] = [None] * int(self.BOOK_EQUIP_SLOTS)
        if not self._hero_has_book_lore():
            self.equipped_book_items = normalized_slots
            self._equipped_book_items_sync_signature = (
                self._equipped_book_items_state_signature()
            )
            return

        current_slots = list(getattr(self, "equipped_book_items", []) or [])
        reserved: Dict[str, int] = {}

        for slot_index in range(int(self.BOOK_EQUIP_SLOTS)):
            item_name = (
                str(current_slots[slot_index] or "")
                if slot_index < len(current_slots) and current_slots[slot_index] is not None
                else ""
            )
            if not self._is_book_item_name(item_name):
                continue
            available_count = int(self._count_hero_item(item_name))
            already_reserved = int(reserved.get(item_name, 0))
            if already_reserved >= available_count:
                continue
            normalized_slots[slot_index] = item_name
            reserved[item_name] = already_reserved + 1

        self.equipped_book_items = normalized_slots
        self._equipped_book_items_sync_signature = (
            self._equipped_book_items_state_signature()
        )
    def _sync_equipped_artifact_items(self, units: Optional[List[Dict]] = None) -> None:
        sync_signature = self._equipped_artifact_items_state_signature(units=units)
        if (
            getattr(self, "_equipped_artifact_items_sync_signature", None)
            == sync_signature
        ):
            return
        normalized_slots: List[Optional[str]] = [None] * int(self.ARTIFACT_EQUIP_SLOTS)
        if not self._hero_has_artifact_knowledge(units=units):
            self.equipped_artifact_items = normalized_slots
            self.artifact_auto_equip_next_slot = 0
            self._equipped_artifact_items_sync_signature = (
                self._equipped_artifact_items_state_signature(units=units)
            )
            return

        ranked_artifacts: List[Tuple[int, int, str]] = []
        for inventory_index, entry in enumerate(self.heroitems):
            item_name = self._hero_item_name(entry)
            if not self._is_artifact_item_name(item_name):
                continue
            ranked_artifacts.append(
                (
                    -int(self._hero_item_gold_value_by_name(item_name)),
                    int(inventory_index),
                    item_name,
                )
            )

        ranked_artifacts.sort()
        for slot_index, (_, _, item_name) in enumerate(
            ranked_artifacts[: int(self.ARTIFACT_EQUIP_SLOTS)]
        ):
            normalized_slots[slot_index] = item_name

        self.equipped_artifact_items = normalized_slots
        self.artifact_auto_equip_next_slot = 0
        self._equipped_artifact_items_sync_signature = (
            self._equipped_artifact_items_state_signature(units=units)
        )
    def _auto_equip_artifact_item(self, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "")
        if not self._is_artifact_item_name(normalized_name):
            return {
                "artifact_auto_equipped": False,
                "artifact_auto_equip_slot": None,
                "artifact_auto_equip_previous": None,
            }

        previous_slots = list(getattr(self, "equipped_artifact_items", []) or [])
        self._sync_equipped_artifact_items()
        equipped_slots = list(self.equipped_artifact_items or [])
        equipped_slot_index = next(
            (
                slot_index
                for slot_index, item in enumerate(equipped_slots)
                if item == normalized_name
                and item
                != (previous_slots[slot_index] if slot_index < len(previous_slots) else None)
            ),
            None,
        )
        artifact_equipped = equipped_slot_index is not None
        previous_item = (
            previous_slots[equipped_slot_index]
            if equipped_slot_index is not None and equipped_slot_index < len(previous_slots)
            else None
        )
        if artifact_equipped:
            self._log(
                f"Артефакт '{normalized_name}' автоматически экипирован в слот "
                f"{equipped_slot_index + 1} (предыдущий: {previous_item or 'пусто'})."
            )
        return {
            "artifact_auto_equipped": bool(artifact_equipped),
            "artifact_auto_equip_slot": (
                int(equipped_slot_index + 1) if equipped_slot_index is not None else None
            ),
            "artifact_auto_equip_previous": previous_item,
        }
    def _auto_equip_book_item(self, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "")
        if not self._is_book_item_name(normalized_name):
            return {
                "book_auto_equipped": False,
                "book_auto_equip_slot": None,
                "book_auto_equip_previous": None,
            }
        if not self._hero_has_book_lore():
            self._sync_equipped_book_items()
            return {
                "book_auto_equipped": False,
                "book_auto_equip_slot": None,
                "book_auto_equip_previous": None,
            }

        previous_slots = list(getattr(self, "equipped_book_items", []) or [])
        self._sync_equipped_book_items()
        current_slots = list(getattr(self, "equipped_book_items", []) or [])
        if current_slots and current_slots[0]:
            return {
                "book_auto_equipped": False,
                "book_auto_equip_slot": None,
                "book_auto_equip_previous": current_slots[0],
            }

        self.equipped_book_items = [None] * int(self.BOOK_EQUIP_SLOTS)
        self.equipped_book_items[0] = normalized_name
        self._sync_equipped_book_items()
        equipped = self.equipped_book_items[0] == normalized_name
        previous_item = previous_slots[0] if previous_slots else None
        if equipped:
            self._sync_equipped_hero_items()
            self.book_equip_used_this_turn = True
            self.books_equipped_total += 1
            self._log(
                f"Book '{normalized_name}' auto-equipped in slot 1 "
                f"(previous: {previous_item or 'empty'})."
            )
        return {
            "book_auto_equipped": bool(equipped),
            "book_auto_equip_slot": 1 if equipped else None,
            "book_auto_equip_previous": previous_item,
        }
    def _append_hero_item(self, item_name: str) -> None:
        normalized_name = str(item_name or "")
        self.heroitems.append(self._make_hero_item_entry(normalized_name))
        self._invalidate_inventory_cache()
    def _counter_backed_potion_reward_item_name(self, item_name: object) -> Optional[str]:
        definition = self._potion_item_definition(item_name)
        grant_kind = str((definition or {}).get("merchant_grant", "") or "")
        if grant_kind == "small_heal_bonus":
            return self.BONUS_SMALL_HEAL_ITEM_NAME
        if grant_kind == "large_heal_bonus":
            return self.BONUS_LARGE_HEAL_ITEM_NAME
        if grant_kind == "revive_bonus":
            return self.BONUS_REVIVE_ITEM_NAME
        return None
    def _grant_counter_backed_potion_reward(self, item_name: object) -> List[str]:
        granted_item = self._counter_backed_potion_reward_item_name(item_name)
        if not granted_item:
            return []
        if granted_item == self.BONUS_SMALL_HEAL_ITEM_NAME:
            self.extra_healing_bottles = max(0, int(self.extra_healing_bottles or 0)) + 1
        elif granted_item == self.BONUS_LARGE_HEAL_ITEM_NAME:
            self.extra_heal_bottles = max(0, int(self.extra_heal_bottles or 0)) + 1
        elif granted_item == self.BONUS_REVIVE_ITEM_NAME:
            self.extra_revive_bottles = max(0, int(self.extra_revive_bottles or 0)) + 1
        self._append_hero_item(granted_item)
        self._sync_counter_backed_battle_items()
        return [granted_item]
    def _empty_hero_item_reward_info(self) -> Dict[str, object]:
        return {
            "artifact_auto_equipped": False,
            "artifact_auto_equip_slot": None,
            "artifact_auto_equip_previous": None,
            "book_auto_equipped": False,
            "book_auto_equip_slot": None,
            "book_auto_equip_previous": None,
            "boot_auto_equipped": False,
            "boot_auto_equip_slot": None,
            "boot_auto_equip_previous": None,
            "active_boot_move_bonus": int(self._active_boot_move_bonus()),
        }
    def _grant_hero_item_reward(self, item_name: str) -> Dict[str, object]:
        granted_items = self._grant_counter_backed_potion_reward(item_name)
        if granted_items:
            return {
                **self._empty_hero_item_reward_info(),
                "granted_items": list(granted_items),
            }
        return self._add_hero_item(item_name)
    def _add_hero_item(self, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "")
        previous_boot_slots = list(getattr(self, "equipped_boot_items", []) or [])
        self._append_hero_item(normalized_name)
        artifact_info = self._auto_equip_artifact_item(normalized_name)
        book_info = self._auto_equip_book_item(normalized_name)
        if (
            bool(artifact_info.get("artifact_auto_equipped"))
            or bool(book_info.get("book_auto_equipped"))
            or self._is_artifact_item_name(normalized_name)
            or self._is_banner_item_name(normalized_name)
            or self._is_book_item_name(normalized_name)
        ):
            self._refresh_campaign_equipment_effects(log=True)
        else:
            self._sync_equipped_banner_items()
        boot_info = {
            "boot_auto_equipped": False,
            "boot_auto_equip_slot": None,
            "boot_auto_equip_previous": None,
            "active_boot_move_bonus": int(self._active_boot_move_bonus()),
        }
        if self._is_boot_item_name(normalized_name):
            self._sync_moves_per_turn_with_hero(
                units=self.blue_team_state,
                grant_delta=True,
            )
            equipped_boot_slots = list(self.equipped_boot_items or [])
            equipped_slot_index = next(
                (
                    slot_index
                    for slot_index, item in enumerate(equipped_boot_slots)
                    if item == normalized_name
                    and item
                    != (
                        previous_boot_slots[slot_index]
                        if slot_index < len(previous_boot_slots)
                        else None
                    )
                ),
                None,
            )
            previous_item = (
                previous_boot_slots[equipped_slot_index]
                if equipped_slot_index is not None
                and equipped_slot_index < len(previous_boot_slots)
                else None
            )
            if equipped_slot_index is not None:
                self._log(
                    f"Boots '{normalized_name}' auto-equipped in slot "
                    f"{equipped_slot_index + 1} (previous: {previous_item or 'empty'})."
                )
            boot_info.update(
                {
                    "boot_auto_equipped": equipped_slot_index is not None,
                    "boot_auto_equip_slot": (
                        int(equipped_slot_index + 1)
                        if equipped_slot_index is not None
                        else None
                    ),
                    "boot_auto_equip_previous": previous_item,
                    "active_boot_move_bonus": int(self._active_boot_move_bonus()),
                }
            )
        else:
            self._sync_equipped_boot_items()
        return {**artifact_info, **book_info, **boot_info}
    def _artifact_state_info(self) -> Dict[str, object]:
        self._refresh_campaign_equipment_effects(log=False)
        return {
            "equipped_artifact_items": list(self.equipped_artifact_items),
            "artifact_auto_equip_next_slot": int(self.artifact_auto_equip_next_slot),
        }
    def _banner_state_info(self) -> Dict[str, object]:
        self._refresh_campaign_equipment_effects(log=False)
        return {
            "equipped_banner_items": list(self.equipped_banner_items),
        }
    def _boot_state_info(self) -> Dict[str, object]:
        self._sync_equipped_boot_items()
        return {
            "equipped_boot_items": list(self.equipped_boot_items),
            "active_boot_move_bonus": int(self._active_boot_move_bonus()),
        }
    def _book_state_info(self) -> Dict[str, object]:
        self._sync_equipped_book_items()
        return {
            "equipped_book_items": list(self.equipped_book_items),
            "book_equip_used_this_turn": bool(self.book_equip_used_this_turn),
            "books_equipped_total": int(self.books_equipped_total),
        }
    @classmethod
    def _artifact_effect_definition(cls, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "")
        damage_multipliers = {
            cls.RING_OF_AGES_ITEM_NAME: cls.RING_OF_AGES_DAMAGE_MULTIPLIER,
            cls.RING_OF_AGES_ENGLISH_ITEM_NAME: cls.RING_OF_AGES_DAMAGE_MULTIPLIER,
            cls.DWARVEN_BRACER_ITEM_NAME: cls.DWARVEN_BRACER_DAMAGE_MULTIPLIER,
            cls.UNHOLY_CHALICE_ARTIFACT_ITEM_NAME: (
                cls.UNHOLY_CHALICE_ARTIFACT_DAMAGE_MULTIPLIER
            ),
            cls.RING_OF_STRENGTH_ITEM_NAME: cls.RING_OF_STRENGTH_DAMAGE_MULTIPLIER,
            cls.RUNIC_BLADE_ITEM_NAME: cls.RUNIC_BLADE_DAMAGE_MULTIPLIER,
            cls.MJOLNIRS_CROWN_ITEM_NAME: cls.MJOLNIRS_CROWN_DAMAGE_MULTIPLIER,
            cls.BETHREZENS_CLAW_ITEM_NAME: cls.BETHREZENS_CLAW_DAMAGE_MULTIPLIER,
        }
        initiative_multipliers = {
            cls.RING_OF_AGES_ITEM_NAME: cls.RING_OF_AGES_INITIATIVE_MULTIPLIER,
            cls.RING_OF_AGES_ENGLISH_ITEM_NAME: cls.RING_OF_AGES_INITIATIVE_MULTIPLIER,
            cls.BETHREZENS_CLAW_ITEM_NAME: cls.BETHREZENS_CLAW_INITIATIVE_MULTIPLIER,
        }
        armor_bonuses = {
            cls.RUNESTONE_ARTIFACT_ITEM_NAME: cls.RUNESTONE_ARTIFACT_ARMOR_BONUS,
            cls.HOLY_CHALICE_ARTIFACT_ITEM_NAME: cls.HOLY_CHALICE_ARTIFACT_ARMOR_BONUS,
            cls.SKULL_BRACERS_ITEM_NAME: cls.SKULL_BRACERS_ARMOR_BONUS,
            cls.HORN_OF_AWARENESS_ITEM_NAME: cls.HORN_OF_AWARENESS_ARMOR_BONUS,
            cls.ETCHED_CIRCLET_ITEM_NAME: cls.ETCHED_CIRCLET_ARMOR_BONUS,
        }

        effect: Dict[str, object] = {}
        if normalized_name in damage_multipliers:
            effect["damage_multiplier"] = float(damage_multipliers[normalized_name])
        if normalized_name in initiative_multipliers:
            effect["initiative_multiplier"] = float(initiative_multipliers[normalized_name])
        if normalized_name in armor_bonuses:
            effect["armor_bonus"] = int(armor_bonuses[normalized_name])
        return effect
    @classmethod
    def _banner_effect_definition(cls, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "").strip()
        return dict(cls.BANNER_EFFECT_DEFINITIONS.get(normalized_name, {}))
    @classmethod
    def _boot_effect_definition(cls, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "").strip()
        return dict(cls.BOOT_EFFECT_DEFINITIONS.get(normalized_name, {}))
    @classmethod
    def _book_effect_definition(cls, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "").strip()
        return dict(cls.BOOK_EFFECT_DEFINITIONS.get(normalized_name, {}))
    def _active_book_item_name(self) -> Optional[str]:
        self._sync_equipped_book_items()
        return next(
            (
                str(item_name)
                for item_name in list(getattr(self, "equipped_book_items", []) or [])
                if self._is_book_item_name(str(item_name or ""))
            ),
            None,
        )
    def _active_book_effect_definition(self) -> Dict[str, object]:
        active_book = self._active_book_item_name()
        if not active_book:
            return {}
        return self._book_effect_definition(active_book)
    def _active_book_exp_multiplier(self) -> float:
        effect = self._active_book_effect_definition()
        try:
            return max(0.0, float(effect.get("exp_multiplier", 1.0) or 1.0))
        except (TypeError, ValueError):
            return 1.0
    def _battle_magic_item_access_allowed(self, item_name: str) -> bool:
        normalized_name = str(item_name or "").strip()
        if not self._is_battle_magic_item_name(normalized_name):
            return True
        if self._hero_has_sorcery_lore():
            return True
        effect = self._active_book_effect_definition()
        if self._is_battle_orb_item_name(normalized_name):
            return bool(effect.get("unlocks_orbs", False))
        if self._is_battle_talisman_item_name(normalized_name):
            return bool(effect.get("unlocks_talismans", False))
        return False
    def _active_boot_move_bonus(self, units: Optional[List[Dict]] = None) -> int:
        self._sync_equipped_boot_items(units=units)
        active_boot = next(
            (
                str(item_name)
                for item_name in list(self.equipped_boot_items or [])
                if self._is_boot_item_name(str(item_name or ""))
            ),
            None,
        )
        if not active_boot:
            return 0
        effect = self._boot_effect_definition(active_boot)
        return max(0, int(effect.get("move_bonus", 0) or 0))
    def _active_boot_terrain_move_cost(
        self,
        terrain: str,
        units: Optional[List[Dict]] = None,
    ) -> Optional[int]:
        self._sync_equipped_boot_items(units=units)
        active_boot = next(
            (
                str(item_name)
                for item_name in list(self.equipped_boot_items or [])
                if self._is_boot_item_name(str(item_name or ""))
            ),
            None,
        )
        if not active_boot:
            return None
        effect = self._boot_effect_definition(active_boot)
        terrain_bonuses = effect.get("terrain_move_bonuses", {})
        if not isinstance(terrain_bonuses, dict):
            return None
        raw_cost = terrain_bonuses.get(str(terrain or ""))
        if raw_cost is None:
            return None
        try:
            return max(1, int(round(float(raw_cost))))
        except (TypeError, ValueError):
            return None
    @classmethod
    @lru_cache(maxsize=None)
    def _hero_item_aliases(cls, item_name: str) -> Tuple[str, ...]:
        normalized_name = str(item_name or "")
        aliases = {normalized_name}
        potion_aliases = cls._potion_item_aliases_for_name(normalized_name)
        if potion_aliases:
            aliases.update(potion_aliases)
        canonical_scroll_name = cls._canonical_scroll_item_name(normalized_name)
        if canonical_scroll_name:
            aliases.add(canonical_scroll_name)
            for alias_name, alias_canonical_name in cls._scroll_item_alias_to_canonical().items():
                if str(alias_canonical_name or "") == canonical_scroll_name:
                    aliases.add(str(alias_name or ""))
        if normalized_name == cls.INVULNERABILITY_POTION_ITEM_NAME:
            aliases.add(cls.RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME)
        elif normalized_name == cls.RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME:
            aliases.add(cls.INVULNERABILITY_POTION_ITEM_NAME)
        elif normalized_name == cls.RING_OF_AGES_ITEM_NAME:
            aliases.add(cls.RING_OF_AGES_ENGLISH_ITEM_NAME)
        elif normalized_name == cls.RING_OF_AGES_ENGLISH_ITEM_NAME:
            aliases.add(cls.RING_OF_AGES_ITEM_NAME)
        return tuple(aliases)
    @classmethod
    @lru_cache(maxsize=None)
    def _hero_item_alias_set(cls, item_name: str) -> frozenset[str]:
        return frozenset(cls._hero_item_aliases(item_name))
    def _count_hero_item(self, item_name: str) -> int:
        if not item_name:
            return 0
        self._ensure_inventory_cache()
        normalized_names = self._hero_item_alias_set(item_name)
        return int(
            sum(int(self._hero_item_counts_cache.get(name, 0) or 0) for name in normalized_names)
        )
    def _consume_hero_item(self, item_name: str) -> bool:
        if not item_name:
            return False
        normalized_names = self._hero_item_alias_set(item_name)
        for idx, entry in enumerate(self.heroitems):
            if self._hero_item_name(entry) in normalized_names:
                del self.heroitems[idx]
                self._invalidate_inventory_cache()
                self._sync_equipped_book_items()
                self._sync_equipped_artifact_items()
                self._sync_equipped_boot_items()
                self._sync_equipped_hero_items()
                self._refresh_campaign_equipment_effects(log=False)
                self._sync_moves_per_turn_with_hero(units=self.blue_team_state)
                return True
        return False
    def _hero_item_sell_value(self, entry: object) -> int:
        gold_value = self._hero_item_gold(entry)
        if gold_value > 0:
            return gold_value
        return self._hero_item_gold_value_by_name(self._hero_item_name(entry))
    def _is_useful_hero_item_name(self, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        if self._is_artifact_item_name(normalized_name):
            return True
        if self._is_banner_item_name(normalized_name):
            return True
        if self._is_boot_item_name(normalized_name):
            return True
        if self._is_book_item_name(normalized_name):
            return True
        if self._is_staff_spell_item_name(normalized_name):
            return True
        if self._canonical_scroll_item_name(normalized_name):
            return True
        if self._is_potion_item_name(normalized_name):
            return True
        if normalized_name in set(getattr(self, "scenario_battle_magic_item_names", ())):
            return True
        return normalized_name in {
            self.BONUS_SMALL_HEAL_ITEM_NAME,
            self.BONUS_LARGE_HEAL_ITEM_NAME,
            self.HEALING_OINTMENT_ITEM_NAME,
            self.BONUS_REVIVE_ITEM_NAME,
            self.INVULNERABILITY_POTION_ITEM_NAME,
            self.RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME,
            self.STRENGTH_POTION_ITEM_NAME,
            self.TITAN_ELIXIR_ITEM_NAME,
            self.SUPREME_ELIXIR_ITEM_NAME,
            *self.RUIN_REWARD_ITEM_NAMES,
            *[str(item_data.get("name", "") or "") for item_data in self.MERCHANT_BUY_ITEMS],
        }
    def _is_sell_only_hero_item(self, entry: object) -> bool:
        item_name = self._hero_item_name(entry)
        if not item_name or self._is_useful_hero_item_name(item_name):
            return False
        return self._hero_item_sell_value(entry) > 0
    def _sell_sell_only_heroitems_at_merchant(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        sale_info: Dict[str, object] = {
            "merchant_auto_sale": False,
            "merchant_sold_items": [],
            "merchant_sold_items_count": 0,
            "merchant_sale_gold": 0.0,
            "merchant_sale_reward": 0.0,
        }
        if not self.heroitems:
            return sale_info

        normalized_position = (
            (int(position[0]), int(position[1]))
            if isinstance(position, (list, tuple)) and len(position) >= 2
            else tuple(self.grid_env.agent_pos)
        )
        site_names = self._merchant_sites_at_position(normalized_position)
        if not site_names:
            return sale_info

        current_version = int(getattr(self, "_inventory_version", 0) or 0)
        if (
            getattr(self, "_merchant_autosale_checked_inventory_version", None)
            == current_version
            and getattr(self, "_merchant_autosale_checked_position", None)
            == normalized_position
        ):
            return sale_info

        sold_items: List[Dict[str, object]] = []
        kept_items: List[object] = []
        total_gold = 0
        for entry in self.heroitems:
            if not self._is_sell_only_hero_item(entry):
                kept_items.append(entry)
                continue

            item_name = self._hero_item_name(entry)
            gold_value = self._hero_item_sell_value(entry)
            if gold_value <= 0:
                kept_items.append(entry)
                continue

            sold_items.append({"name": item_name, "gold": int(gold_value)})
            total_gold += int(gold_value)

        if not sold_items:
            self._merchant_autosale_checked_inventory_version = current_version
            self._merchant_autosale_checked_position = normalized_position
            return sale_info

        self.heroitems = kept_items
        self._invalidate_inventory_cache()
        self._sync_equipped_book_items()
        self._sync_equipped_artifact_items()
        self._sync_equipped_boot_items()
        self._sync_equipped_hero_items()
        self._refresh_campaign_equipment_effects(log=False)
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state)
        self.gold = max(0.0, float(self.gold or 0.0)) + float(total_gold)
        sale_reward = float(len(sold_items)) * float(self.reward_sell_junk_item)
        sale_info.update(
            {
                "merchant_auto_sale": True,
                "merchant_sold_items": sold_items,
                "merchant_sold_items_count": int(len(sold_items)),
                "merchant_sale_gold": float(total_gold),
                "merchant_sale_reward": float(sale_reward),
            }
        )

        sold_names = ", ".join(str(item["name"]) for item in sold_items)
        self._log(
            f"Торговец {', '.join(site_names)}: продано {len(sold_items)} предметов "
            f"на {total_gold} gold ({sold_names})"
        )
        self._merchant_autosale_checked_inventory_version = int(
            getattr(self, "_inventory_version", 0) or 0
        )
        self._merchant_autosale_checked_position = normalized_position
        return sale_info
    def _combat_potion_reward_value(self) -> float:
        return float(self.reward_defeat_enemy)
    def _count_collected_combat_potions(self, collected_chests: List[Dict]) -> int:
        count = 0
        for chest_data in collected_chests:
            for item_name in chest_data.get("items", []):
                definition = self._potion_item_definition(item_name)
                if definition and str(definition.get("duration", "") or "") == "temporary":
                    count += 1
        return int(count)
    def _clear_expired_combat_potion_effects(self) -> None:
        for item_name, positions in sorted(
            getattr(self, "active_potion_effect_positions", {}).items()
        ):
            if positions:
                self._log(
                    f"Эффект зелья {item_name} завершён на позициях {sorted(positions)}"
                )
        if self.active_invulnerability_potion_positions:
            self._log(
                "Эффект зелья неуязвимости завершён на позициях "
                f"{sorted(self.active_invulnerability_potion_positions)}"
            )
        if self.active_strength_potion_positions:
            self._log(
                "Эффект зелья силы завершён на позициях "
                f"{sorted(self.active_strength_potion_positions)}"
            )
        if self.active_energy_elixir_positions:
            self._log(
                "Эффект эликсира энергии завершён на позициях "
                f"{sorted(self.active_energy_elixir_positions)}"
            )
        if self.active_haste_elixir_positions:
            self._log(
                "Эффект эликсира быстроты завершён на позициях "
                f"{sorted(self.active_haste_elixir_positions)}"
            )
        if self.active_fire_ward_positions:
            self._log(
                "Эффект защиты от магии Огня завершён на позициях "
                f"{sorted(self.active_fire_ward_positions)}"
            )
        if self.active_earth_ward_positions:
            self._log(
                "Эффект защиты от магии Земли завершён на позициях "
                f"{sorted(self.active_earth_ward_positions)}"
            )
        if self.active_water_ward_positions:
            self._log(
                "Эффект защиты от магии Воды завершён на позициях "
                f"{sorted(self.active_water_ward_positions)}"
            )
        if self.active_air_ward_positions:
            self._log(
                "Эффект защиты от магии Воздуха завершён на позициях "
                f"{sorted(self.active_air_ward_positions)}"
            )
        self.active_invulnerability_potion_positions.clear()
        self.active_strength_potion_positions.clear()
        self.active_energy_elixir_positions.clear()
        self.active_haste_elixir_positions.clear()
        self.active_fire_ward_positions.clear()
        self.active_earth_ward_positions.clear()
        self.active_water_ward_positions.clear()
        self.active_air_ward_positions.clear()
        self.active_potion_effect_positions.clear()
    def get_bottle_inventory_counters(self) -> Dict[str, int]:
        max_heal = self._max_heal_bottles_available()
        max_healing = self._max_healing_bottles_available()
        max_revive = self._max_revive_bottles_available()
        heal_used = int(self.heal_bottles_used or 0)
        healing_used = int(self.healing_bottles_used or 0)
        revive_used = int(self.revive_bottles_used or 0)
        return {
            "max_heal": max_heal,
            "heal_left": max(0, max_heal - heal_used),
            "max_healing": max_healing,
            "healing_left": max(0, max_healing - healing_used),
            "ointment_left": self._count_hero_item(self.HEALING_OINTMENT_ITEM_NAME),
            "max_revive": max_revive,
            "revive_left": max(0, max_revive - revive_used),
            "invulnerability_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
        }
    @classmethod
    def _is_within_chest_pickup_range(
        cls,
        agent_pos: Tuple[int, int],
        chest_pos: Tuple[int, int],
    ) -> bool:
        dx = abs(int(agent_pos[0]) - int(chest_pos[0]))
        dy = abs(int(agent_pos[1]) - int(chest_pos[1]))
        return max(dx, dy) <= int(cls.CHEST_PICKUP_RADIUS)
    def _can_equip_book_item(self, item_name: object) -> bool:
        normalized_name = str(item_name or "").strip()
        if not self._is_book_item_name(normalized_name):
            return False
        if not self._hero_has_book_lore():
            return False
        if normalized_name not in tuple(getattr(self, "scenario_book_item_names", ()) or ()):
            return False
        if self.book_equip_used_this_turn:
            return False
        self._sync_equipped_book_items()
        if normalized_name in set(str(item or "") for item in self.equipped_book_items):
            return False
        return self._count_hero_item(normalized_name) > 0
    def _equip_book_item(self, item_name: object) -> Optional[int]:
        normalized_name = str(item_name or "").strip()
        if not self._can_equip_book_item(normalized_name):
            return None
        if len(getattr(self, "equipped_book_items", []) or []) != int(self.BOOK_EQUIP_SLOTS):
            self.equipped_book_items = [None] * int(self.BOOK_EQUIP_SLOTS)
        self.equipped_book_items[0] = normalized_name
        self._sync_equipped_book_items()
        self._sync_equipped_hero_items()
        if self.equipped_book_items and self.equipped_book_items[0] == normalized_name:
            return 0
        return None
    def _step_equip_battle_item(self, action: int):
        action_start = self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START
        idx = int(action) - int(action_start)
        item_name = (
            self.BATTLE_EQUIPPABLE_ITEM_NAMES[idx]
            if 0 <= idx < len(self.BATTLE_EQUIPPABLE_ITEM_NAMES)
            else None
        )

        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        previous_item = None
        equipped = False
        equipped_slot_index = None
        equip_turn_locked = bool(self.battle_item_equip_used_this_turn)
        if not equip_turn_locked and item_name is not None:
            target_slot_index = self._battle_equip_slot_for_item(str(item_name))
            if target_slot_index is not None:
                previous_item = self.equipped_hero_items[int(target_slot_index)]
                equipped_slot_index = self._equip_battle_item(str(item_name))
                equipped = equipped_slot_index is not None
            if equipped:
                self.battle_item_equip_used_this_turn = True
                self.battle_items_equipped_total += 1
                if not previous_item:
                    reward = float(self.reward_battle_item_equip)
                self._log(
                    f"Боевой слот {int(equipped_slot_index) + 1}: экипирован предмет '{str(item_name)}'"
                )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "equip_battle_item_action": True,
            "equip_battle_item_slot": (
                int(equipped_slot_index) + 1 if equipped_slot_index is not None else 0
            ),
            "equip_battle_item_name": str(item_name or ""),
            "equip_battle_item_previous": previous_item,
            "equip_battle_item_applied": bool(equipped),
            "equip_battle_item_turn_locked": bool(equip_turn_locked),
            "equip_battle_item_reward": float(reward),
            "battle_items_equipped_total": int(self.battle_items_equipped_total),
            "battle_items_used_total": int(self.battle_items_used_total),
            "equipped_hero_items": list(self.equipped_hero_items),
            "equipped_hero_item_uses_left": list(self.equipped_hero_item_uses_left),
            "equip_battle_item_uses_left": (
                self.equipped_hero_item_uses_left[int(equipped_slot_index)]
                if equipped_slot_index is not None
                and int(equipped_slot_index) < len(self.equipped_hero_item_uses_left)
                else None
            ),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_equip_book_item(self, action: int):
        action_start = self.GRID_EQUIP_BOOK_ACTION_START
        idx = int(action) - int(action_start)
        book_names = tuple(getattr(self, "scenario_book_item_names", ()) or ())
        item_name = book_names[idx] if 0 <= idx < len(book_names) else None

        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        previous_item = None
        equipped = False
        equipped_slot_index = None
        equip_turn_locked = bool(self.book_equip_used_this_turn)
        if not equip_turn_locked and item_name is not None:
            self._sync_equipped_book_items()
            previous_item = (
                self.equipped_book_items[0]
                if getattr(self, "equipped_book_items", None)
                else None
            )
            equipped_slot_index = self._equip_book_item(str(item_name))
            equipped = equipped_slot_index is not None
            if equipped:
                self.book_equip_used_this_turn = True
                self.books_equipped_total += 1
                self._log(
                    f"Book slot {int(equipped_slot_index) + 1}: equipped '{str(item_name)}'"
                )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "equip_book_action": True,
            "equip_book_slot": (
                int(equipped_slot_index) + 1 if equipped_slot_index is not None else 0
            ),
            "equip_book_name": str(item_name or ""),
            "equip_book_previous": previous_item,
            "equip_book_applied": bool(equipped),
            "equip_book_turn_locked": bool(equip_turn_locked),
            "equipped_book_items": list(self.equipped_book_items),
            "book_equip_used_this_turn": bool(self.book_equip_used_this_turn),
            "books_equipped_total": int(self.books_equipped_total),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _active_potion_positions(self, item_name: object) -> set[int]:
        canonical_name = self._canonical_potion_item_name(item_name)
        if not canonical_name:
            return set()
        definition = self._potion_item_definition(canonical_name)
        active_attr = str(definition.get("active_attr", "") or "")
        if active_attr:
            current_value = getattr(self, active_attr, None)
            if not isinstance(current_value, set):
                current_value = set()
                setattr(self, active_attr, current_value)
            return current_value
        if not hasattr(self, "active_potion_effect_positions"):
            self.active_potion_effect_positions = {}
        return self.active_potion_effect_positions.setdefault(canonical_name, set())

    def _consume_potion_item(self, item_name: object) -> bool:
        canonical_name = self._canonical_potion_item_name(item_name)
        if not canonical_name:
            return False
        if canonical_name in {
            self.BONUS_SMALL_HEAL_ITEM_NAME,
            self.BONUS_LARGE_HEAL_ITEM_NAME,
            self.HEALING_OINTMENT_ITEM_NAME,
            self.BONUS_REVIVE_ITEM_NAME,
        }:
            return self._consume_battle_equipable_item(canonical_name)
        return self._consume_hero_item(canonical_name)

    def _apply_temporary_potion_to_position(
        self,
        item_name: object,
        position: int,
    ) -> Tuple[bool, Optional[str], float]:
        state = self._get_blue_state()
        active_positions = self._active_potion_positions(item_name)
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0 or int(position) in active_positions:
                return False, None, 0.0
            active_positions.add(int(position))
            canonical_name = self._canonical_potion_item_name(item_name)
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"{canonical_name}: {name} (pos {position}) получает эффект до конца текущего хода"
            )
            return True, name, 0.0
        return False, None, 0.0

    def _apply_permanent_potion_bonus_to_unit(
        self,
        unit: Dict,
        definition: Dict[str, object],
        *,
        increment_uses: bool = True,
    ) -> None:
        effect = dict(definition.get("effect", {}) or {})
        effect_kind = str(effect.get("kind", "") or "")
        canonical_name = str(definition.get("name", "") or "")
        if effect_kind == "damage":
            multiplier = max(0.0, float(effect.get("multiplier", 1.0) or 1.0))
            base_damage = self._normalize_damage_value(unit.get("damage", 0))
            base_damage_secondary = self._normalize_damage_value(
                unit.get("damage_secondary", 0)
            )
            unit["damage"] = self._normalize_damage_value(float(base_damage) * multiplier)
            unit["damage_secondary"] = self._normalize_damage_value(
                float(base_damage_secondary) * multiplier
            )
            unit["original_damage"] = int(unit["damage"])
        elif effect_kind == "health":
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            multiplier = max(0.0, float(effect.get("multiplier", 1.0) or 1.0))
            new_max_hp = self._normalize_health_value(float(max_hp) * multiplier)
            if new_max_hp <= int(round(max_hp)):
                new_max_hp = int(round(max_hp)) + 1
            new_hp = min(
                new_max_hp,
                self._normalize_health_value(float(hp) * multiplier),
            )
            unit["maxhp"] = int(new_max_hp)
            unit["max_health"] = int(new_max_hp)
            unit["hp"] = int(new_hp)
            unit["health"] = int(new_hp)
        elif effect_kind == "damage_reduction":
            multiplier = max(
                0.0,
                float(effect.get("incoming_damage_multiplier", 1.0) or 1.0),
            )
            base_multiplier = float(unit.get("incoming_damage_multiplier", 1.0) or 1.0)
            unit["incoming_damage_multiplier"] = max(0.0, base_multiplier * multiplier)
        elif effect_kind == "accuracy":
            multiplier = max(0.0, float(effect.get("multiplier", 1.0) or 1.0))
            base_accuracy = self._normalize_accuracy_value(unit.get("accuracy", 0))
            base_accuracy_secondary = self._normalize_accuracy_value(
                unit.get("accuracy_secondary", 0)
            )
            unit["accuracy"] = self._normalize_accuracy_value(
                float(base_accuracy) * multiplier
            )
            unit["accuracy_secondary"] = self._normalize_accuracy_value(
                float(base_accuracy_secondary) * multiplier
            )
        elif effect_kind == "initiative":
            multiplier = max(0.0, float(effect.get("multiplier", 1.0) or 1.0))
            base_initiative = int(
                unit.get("initiative_base", unit.get("initiative", 0)) or 0
            )
            boosted_initiative = self._normalize_damage_value(
                float(base_initiative) * multiplier
            )
            unit["initiative_base"] = int(boosted_initiative)
            unit["initiative"] = int(boosted_initiative)

        if increment_uses:
            counter_key = str(definition.get("permanent_counter_key", "") or "")
            if counter_key:
                unit[counter_key] = max(0, int(unit.get(counter_key, 0) or 0)) + 1
            applied_names = [
                str(name)
                for name in (unit.get("campaign_permanent_potions", []) or [])
                if str(name)
            ]
            applied_names.append(canonical_name)
            unit["campaign_permanent_potions"] = applied_names

    def _apply_permanent_potion_to_position(
        self,
        item_name: object,
        position: int,
    ) -> Tuple[bool, Optional[str], float]:
        definition = self._potion_item_definition(item_name)
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0:
                return False, None, 0.0

            self._apply_permanent_potion_bonus_to_unit(
                unit,
                definition,
                increment_uses=True,
            )
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"{definition.get('name', item_name)}: {name} (pos {position}) "
                "получает постоянный эффект"
            )
            return True, name, 0.0
        return False, None, 0.0

    def _apply_potion_to_position(
        self,
        item_name: object,
        position: int,
    ) -> Tuple[bool, Optional[str], float]:
        definition = self._potion_item_definition(item_name)
        effect = dict(definition.get("effect", {}) or {})
        effect_kind = str(effect.get("kind", "") or "")
        if effect_kind == "heal":
            healed_amount, unit_name = self._heal_unit_at_position(
                position=position,
                heal_amount=float(effect.get("amount", 0.0) or 0.0),
                source_label=str(definition.get("name", item_name) or item_name),
            )
            return healed_amount > 0.0, unit_name, float(healed_amount)
        if effect_kind == "revive":
            revived, unit_name = self._revive_unit_at_position(position=position)
            return bool(revived), unit_name, float(effect.get("amount", 1.0) or 1.0)
        if str(definition.get("duration", "") or "") == "permanent":
            return self._apply_permanent_potion_to_position(item_name, position)
        return self._apply_temporary_potion_to_position(item_name, position)

    def _step_use_potion(self, action: int):
        idx = action - self.GRID_POTION_USE_ACTION_START
        target_count = len(self.GRID_POTION_USE_POSITIONS)
        potion_idx = int(idx // target_count) if target_count > 0 else -1
        target_idx = int(idx % target_count) if target_count > 0 else -1
        item_name = (
            self.scenario_potion_item_names[potion_idx]
            if 0 <= potion_idx < len(self.scenario_potion_item_names)
            else ""
        )
        target_pos = (
            self.GRID_POTION_USE_POSITIONS[target_idx]
            if 0 <= target_idx < len(self.GRID_POTION_USE_POSITIONS)
            else None
        )
        definition = self._potion_item_definition(item_name)
        effect = dict(definition.get("effect", {}) or {})
        effect_kind = str(effect.get("kind", "") or "")
        duration = str(definition.get("duration", "") or "")

        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        applied = False
        consumed = False
        unit_name = None
        applied_amount = 0.0
        if target_pos is not None and self._can_use_potion_on_position(item_name, target_pos):
            consumed = self._consume_potion_item(item_name)
            if consumed:
                applied, unit_name, applied_amount = self._apply_potion_to_position(
                    item_name,
                    target_pos,
                )
                if not applied and effect_kind not in {"heal", "revive"}:
                    self._append_hero_item(str(item_name or ""))
                    consumed = False
                if applied and duration == "temporary":
                    reward = self._combat_potion_reward_value()
                    self.combat_potion_battle_bonus_pending = True

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "potion_action": True,
            "potion_item": item_name,
            "potion_effect_kind": effect_kind,
            "potion_target_pos": target_pos,
            "potion_applied": applied,
            "potion_consumed": consumed,
            "potion_unit_name": unit_name,
            "potion_amount": float(applied_amount),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
            "heal_bottles_used": self.heal_bottles_used,
            "extra_heal_bottles": self.extra_heal_bottles,
            "heal_bottles_left": self._heal_bottles_left(),
            "max_heal_bottles": self._max_heal_bottles_available(),
            "healing_bottles_used": self.healing_bottles_used,
            "extra_healing_bottles": self.extra_healing_bottles,
            "healing_bottles_left": self._healing_bottles_left(),
            "max_healing_bottles": self._max_healing_bottles_available(),
            "revive_bottles_used": self.revive_bottles_used,
            "extra_revive_bottles": self.extra_revive_bottles,
            "revive_bottles_left": self._revive_bottles_left(),
            "max_revive_bottles": self._max_revive_bottles_available(),
            "healing_ointment_left": self._count_hero_item(self.HEALING_OINTMENT_ITEM_NAME),
        }
        if effect_kind == "heal":
            info.update(
                {
                    "heal_action": True,
                    "heal_target_pos": target_pos,
                    "healed_amount": float(applied_amount),
                    "healed_unit_name": unit_name,
                    "selected_heal_bottle_kind": item_name,
                    "selected_heal_bottle_amount": float(effect.get("amount", 0.0) or 0.0),
                }
            )
        if effect_kind == "revive":
            info.update(
                {
                    "revive_action": True,
                    "revive_target_pos": target_pos,
                    "revived": bool(applied),
                    "revived_unit_name": unit_name,
                }
            )
        if duration == "temporary":
            info.update(
                {
                    "combat_potion_action": True,
                    "combat_potion_item": item_name,
                    "combat_potion_target_pos": target_pos,
                    "combat_potion_applied": applied,
                    "combat_potion_consumed": consumed,
                    "combat_potion_unit_name": unit_name,
                    "combat_potion_reward": float(reward),
                    "combat_potion_battle_bonus_pending": bool(
                        self.combat_potion_battle_bonus_pending
                    ),
                }
            )
        if duration == "permanent":
            info.update(
                {
                    "persistent_elixir_action": True,
                    "persistent_elixir_item": item_name,
                    "persistent_elixir_target_pos": target_pos,
                    "persistent_elixir_applied": applied,
                    "persistent_elixir_consumed": consumed,
                    "persistent_elixir_unit_name": unit_name,
                }
            )
        info.update(self._merchant_context_info(self.grid_env.agent_pos))

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _reapply_persistent_elixir_bonuses_to_promoted_unit(
        self,
        source_unit: Dict,
        upgraded_unit: Dict,
    ) -> None:
        for definition in tuple(getattr(self, "POTION_ITEM_DEFINITIONS", ()) or ()):
            if not isinstance(definition, dict):
                continue
            if str(definition.get("duration", "") or "") != "permanent":
                continue
            counter_key = str(definition.get("permanent_counter_key", "") or "")
            if not counter_key:
                continue
            upgraded_unit.pop(counter_key, None)
            uses = max(0, int(source_unit.get(counter_key, 0) or 0))
            for _ in range(uses):
                self._apply_permanent_potion_bonus_to_unit(
                    upgraded_unit,
                    dict(definition),
                    increment_uses=True,
                )
    def _apply_active_blue_potion_effects(self, blue_team: List[Dict]) -> None:
        if not blue_team:
            return

        active_entries: List[Tuple[Dict[str, object], set[int]]] = []
        for definition in tuple(getattr(self, "POTION_ITEM_DEFINITIONS", ()) or ()):
            if not isinstance(definition, dict):
                continue
            if str(definition.get("duration", "") or "") != "temporary":
                continue
            positions = self._active_potion_positions(definition.get("name", ""))
            if positions:
                active_entries.append((dict(definition), positions))
        if not active_entries:
            return

        buffed_by_item: Dict[str, List[int]] = {}

        for unit in blue_team:
            position = int(unit.get("position", -1) or -1)
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if position <= 0 or max_hp <= 0 or hp <= 0:
                continue

            damage_multiplier = 1.0
            initiative_multiplier = 1.0
            accuracy_multiplier = 1.0
            incoming_damage_multiplier = 1.0
            added_resistance: List[str] = []
            resistance = [str(res) for res in (unit.get("resistance") or [])]

            for definition, active_positions in active_entries:
                if position not in active_positions:
                    continue
                effect = dict(definition.get("effect", {}) or {})
                effect_kind = str(effect.get("kind", "") or "")
                item_name = str(definition.get("name", "") or "")
                buffed_by_item.setdefault(item_name, []).append(position)
                if effect_kind == "damage":
                    damage_multiplier *= max(0.0, float(effect.get("multiplier", 1.0) or 1.0))
                elif effect_kind == "initiative":
                    initiative_multiplier *= max(
                        0.0,
                        float(effect.get("multiplier", 1.0) or 1.0),
                    )
                elif effect_kind == "accuracy":
                    accuracy_multiplier *= max(
                        0.0,
                        float(effect.get("multiplier", 1.0) or 1.0),
                    )
                elif effect_kind == "damage_reduction":
                    incoming_damage_multiplier *= max(
                        0.0,
                        float(effect.get("incoming_damage_multiplier", 1.0) or 1.0),
                    )
                elif effect_kind == "ward":
                    element = str(effect.get("resistance_type", "") or "")
                    if element and element not in resistance:
                        resistance.append(element)
                        added_resistance.append(element)

            if abs(damage_multiplier - 1.0) > 1e-9:
                base_damage = self._normalize_damage_value(unit.get("damage", 0))
                base_damage_secondary = self._normalize_damage_value(unit.get("damage_secondary", 0))
                unit["campaign_potion_base_damage"] = int(base_damage)
                unit["campaign_potion_base_damage_secondary"] = int(base_damage_secondary)
                unit["damage"] = self._normalize_damage_value(
                    float(base_damage) * float(damage_multiplier)
                )
                unit["damage_secondary"] = self._normalize_damage_value(
                    float(base_damage_secondary) * float(damage_multiplier)
                )
                unit["campaign_damage_potion_multiplier"] = float(damage_multiplier)

            if abs(initiative_multiplier - 1.0) > 1e-9:
                base_initiative = int(unit.get("initiative_base", unit.get("initiative", 0)) or 0)
                unit["campaign_potion_base_initiative"] = int(base_initiative)
                boosted_initiative = self._normalize_damage_value(
                    float(base_initiative) * initiative_multiplier
                )
                unit["initiative_base"] = int(boosted_initiative)
                unit["initiative"] = int(boosted_initiative)
                unit["campaign_initiative_potion_multiplier"] = float(initiative_multiplier)

            if abs(accuracy_multiplier - 1.0) > 1e-9:
                base_accuracy = self._normalize_accuracy_value(unit.get("accuracy", 0))
                base_accuracy_secondary = self._normalize_accuracy_value(
                    unit.get("accuracy_secondary", 0)
                )
                unit["campaign_potion_base_accuracy"] = int(base_accuracy)
                unit["campaign_potion_base_accuracy_secondary"] = int(base_accuracy_secondary)
                unit["accuracy"] = self._normalize_accuracy_value(
                    float(base_accuracy) * accuracy_multiplier
                )
                unit["accuracy_secondary"] = self._normalize_accuracy_value(
                    float(base_accuracy_secondary) * accuracy_multiplier
                )
                unit["campaign_accuracy_potion_multiplier"] = float(accuracy_multiplier)

            if abs(incoming_damage_multiplier - 1.0) > 1e-9:
                base_incoming_multiplier = float(
                    unit.get("incoming_damage_multiplier", 1.0) or 1.0
                )
                unit["campaign_potion_base_incoming_damage_multiplier"] = float(
                    base_incoming_multiplier
                )
                unit["incoming_damage_multiplier"] = max(
                    0.0,
                    base_incoming_multiplier * incoming_damage_multiplier,
                )
                unit["campaign_incoming_damage_potion_multiplier"] = float(
                    incoming_damage_multiplier
                )

            if added_resistance:
                unit["resistance"] = resistance
                unit["campaign_potion_added_resistance"] = added_resistance

        for item_name, positions in sorted(buffed_by_item.items()):
            self._log(
                f"Активно зелье {item_name} для позиций {sorted(set(positions))} "
                "перед началом боя"
            )
    def _apply_active_blue_support_spell_effects(self, blue_team: List[Dict]) -> None:
        if not blue_team:
            return

        summary = self._blue_stack_spell_effect_summary()
        active_spell_ids = list(summary.get("active_spell_ids", []))
        if not active_spell_ids:
            return

        buffed_armor: List[int] = []
        buffed_damage: List[int] = []
        buffed_initiative: List[int] = []
        buffed_accuracy: List[int] = []
        buffed_health: List[int] = []
        buffed_resistance: List[int] = []
        resistance_types = [
            str(value)
            for value in (summary.get("resistance_types", []) or [])
            if str(value)
        ]

        for unit in blue_team:
            position = int(unit.get("position", -1) or -1)
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0.0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
            if position <= 0 or max_hp <= 0.0 or hp <= 0.0:
                continue

            armor_delta = int(summary.get("armor_delta", 0) or 0)
            if armor_delta != 0:
                base_armor = self._normalize_armor_value(unit.get("armor", 0))
                unit["campaign_map_spell_base_armor"] = int(base_armor)
                unit["armor"] = self._normalize_armor_value(float(base_armor) + float(armor_delta))
                buffed_armor.append(position)

            damage_multiplier = float(summary.get("damage_multiplier", 1.0) or 1.0)
            if abs(damage_multiplier - 1.0) > 1e-9:
                base_damage = self._normalize_damage_value(unit.get("damage", 0))
                base_damage_secondary = self._normalize_damage_value(unit.get("damage_secondary", 0))
                unit["campaign_map_spell_base_damage"] = int(base_damage)
                unit["campaign_map_spell_base_damage_secondary"] = int(base_damage_secondary)
                unit["damage"] = self._normalize_damage_value(
                    float(base_damage) * float(damage_multiplier)
                )
                unit["damage_secondary"] = self._normalize_damage_value(
                    float(base_damage_secondary) * float(damage_multiplier)
                )
                buffed_damage.append(position)

            initiative_multiplier = float(summary.get("initiative_multiplier", 1.0) or 1.0)
            if abs(initiative_multiplier - 1.0) > 1e-9:
                base_initiative = int(unit.get("initiative_base", unit.get("initiative", 0)) or 0)
                unit["campaign_map_spell_base_initiative"] = int(base_initiative)
                boosted_initiative = self._normalize_damage_value(
                    float(base_initiative) * float(initiative_multiplier)
                )
                unit["initiative_base"] = int(boosted_initiative)
                unit["initiative"] = int(boosted_initiative)
                buffed_initiative.append(position)

            accuracy_multiplier = float(summary.get("accuracy_multiplier", 1.0) or 1.0)
            if abs(accuracy_multiplier - 1.0) > 1e-9:
                base_accuracy = self._normalize_accuracy_value(unit.get("accuracy", 0))
                base_accuracy_secondary = self._normalize_accuracy_value(
                    unit.get("accuracy_secondary", 0)
                )
                unit["campaign_map_spell_base_accuracy"] = int(base_accuracy)
                unit["campaign_map_spell_base_accuracy_secondary"] = int(base_accuracy_secondary)
                unit["accuracy"] = self._normalize_accuracy_value(
                    float(base_accuracy) * float(accuracy_multiplier)
                )
                unit["accuracy_secondary"] = self._normalize_accuracy_value(
                    float(base_accuracy_secondary) * float(accuracy_multiplier)
                )
                buffed_accuracy.append(position)

            health_delta = int(summary.get("health_delta", 0) or 0)
            if health_delta != 0:
                base_max_health = float(
                    unit.get("max_health", 0)
                    or unit.get("maxhp", 0)
                    or unit.get("health", 0)
                    or unit.get("hp", 0)
                    or 0.0
                )
                base_health = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
                unit["campaign_map_spell_base_max_health"] = float(base_max_health)
                unit["campaign_map_spell_health_bonus"] = int(health_delta)
                boosted_max_health = max(0.0, base_max_health + float(health_delta))
                boosted_health = max(0.0, base_health + float(health_delta))
                boosted_health = min(boosted_max_health, boosted_health)
                unit["max_health"] = float(boosted_max_health)
                unit["maxhp"] = float(boosted_max_health)
                unit["health"] = float(boosted_health)
                unit["hp"] = float(boosted_health)
                buffed_health.append(position)

            if resistance_types:
                resistance = [str(res) for res in (unit.get("resistance") or [])]
                added_resistance = [res for res in resistance_types if res not in resistance]
                if added_resistance:
                    unit["resistance"] = resistance + added_resistance
                    unit["campaign_map_spell_added_resistance"] = list(added_resistance)
                    buffed_resistance.append(position)

        if buffed_armor:
            self._log(
                f"Активен бафф брони от заклинаний героя для позиций {sorted(buffed_armor)} "
                f"перед началом боя (+{int(summary.get('armor_delta', 0) or 0)})."
            )
        if buffed_damage:
            self._log(
                "Активен бафф урона от заклинаний героя для позиций "
                f"{sorted(buffed_damage)} перед началом боя "
                f"(x{float(summary.get('damage_multiplier', 1.0) or 1.0):.3g})."
            )
        if buffed_initiative:
            self._log(
                "Активен бафф инициативы от заклинаний героя для позиций "
                f"{sorted(buffed_initiative)} перед началом боя "
                f"(x{float(summary.get('initiative_multiplier', 1.0) or 1.0):.3g})."
            )
        if buffed_accuracy:
            self._log(
                "Активен бафф точности от заклинаний героя для позиций "
                f"{sorted(buffed_accuracy)} перед началом боя "
                f"(x{float(summary.get('accuracy_multiplier', 1.0) or 1.0):.3g})."
            )
        if buffed_health:
            self._log(
                "Активен бафф здоровья от заклинаний героя для позиций "
                f"{sorted(buffed_health)} перед началом боя "
                f"(+{int(summary.get('health_delta', 0) or 0)} HP)."
            )
        if buffed_resistance:
            self._log(
                "Активна защита от магии заклинаниями героя для позиций "
                f"{sorted(buffed_resistance)} перед началом боя: {', '.join(resistance_types)}."
            )
    def _refresh_campaign_equipment_effects(self, *, log: bool = False) -> None:
        if self.blue_team_state is None:
            return
        refresh_signature = self._equipment_state_signature(units=self.blue_team_state)
        if (
            not bool(getattr(self, "equipment_dirty", True))
            and getattr(self, "_equipment_refresh_signature", None) == refresh_signature
        ):
            return
        self._clear_equipped_banner_effects(self.blue_team_state)
        self._apply_equipped_artifact_effects(self.blue_team_state, log=log)
        self._apply_equipped_banner_effects(self.blue_team_state, log=log)
        self._equipment_refresh_signature = self._equipment_state_signature(
            units=self.blue_team_state
        )
        self.equipment_dirty = False
        self._equipment_dirty = False
    def _clear_equipped_book_effects(self, blue_team: Optional[List[Dict]]) -> None:
        if not blue_team:
            return
        for unit in blue_team:
            if not isinstance(unit, dict):
                continue
            added_resistance = [
                str(res)
                for res in (unit.get("campaign_book_added_resistance") or [])
                if str(res)
            ]
            if added_resistance:
                unit["resistance"] = [
                    res
                    for res in (unit.get("resistance") or [])
                    if str(res) not in set(added_resistance)
                ]
            unit.pop("campaign_book_added_resistance", None)
            unit.pop("campaign_active_book", None)
    def _apply_equipped_book_battle_effects(
        self,
        blue_team: List[Dict],
        *,
        log: bool = True,
    ) -> None:
        self._clear_equipped_book_effects(blue_team)
        if not blue_team:
            return
        active_book = self._active_book_item_name()
        if not active_book:
            return
        hero = self._resolve_travel_hero(units=blue_team)
        if hero is None:
            return

        hero["campaign_active_book"] = str(active_book)
        effect = self._book_effect_definition(active_book)
        resistance_type = str(effect.get("resistance_type", "") or "").strip()
        if not resistance_type:
            return

        resistance = [str(res) for res in (hero.get("resistance") or [])]
        if resistance_type not in resistance:
            hero["resistance"] = resistance + [resistance_type]
            hero["campaign_book_added_resistance"] = [resistance_type]
            if log:
                self._log(
                    f"Book '{active_book}' grants hero resistance {resistance_type} for the next battle."
                )
    def _clear_equipped_banner_effects(self, blue_team: Optional[List[Dict]]) -> None:
        if not blue_team:
            return

        for unit in blue_team:
            if not isinstance(unit, dict):
                continue
            if "campaign_banner_base_damage" in unit:
                unit["damage"] = self._normalize_damage_value(
                    unit.get(
                        "campaign_banner_base_damage",
                        unit.get("damage", 0),
                    )
                )
            if "campaign_banner_base_accuracy" in unit:
                unit["accuracy"] = self._normalize_accuracy_value(
                    unit.get(
                        "campaign_banner_base_accuracy",
                        unit.get("accuracy", 0),
                    )
                )
            if "campaign_banner_base_initiative" in unit:
                base_initiative = int(
                    self._normalize_damage_value(
                        unit.get(
                            "campaign_banner_base_initiative",
                            unit.get("initiative_base", unit.get("initiative", 0)),
                        )
                    )
                )
                unit["initiative_base"] = int(base_initiative)
                unit["initiative"] = int(base_initiative)
            if "campaign_banner_base_armor" in unit:
                base_armor = self._normalize_armor_value(
                    unit.get(
                        "campaign_banner_base_armor",
                        unit.get("armor", 0),
                    )
                )
                unit["armor"] = int(base_armor)
                unit["base_armor"] = int(base_armor)
            unit.pop("campaign_banner_base_damage", None)
            unit.pop("campaign_banner_base_accuracy", None)
            unit.pop("campaign_banner_base_initiative", None)
            unit.pop("campaign_banner_base_armor", None)
            unit.pop("campaign_banner_damage_multiplier", None)
            unit.pop("campaign_banner_accuracy_bonus", None)
            unit.pop("campaign_banner_initiative_multiplier", None)
            unit.pop("campaign_banner_armor_bonus", None)
            unit.pop("campaign_active_banner", None)
    def _refresh_campaign_banner_effects(self, *, log: bool = False) -> None:
        if self.blue_team_state is None:
            return
        self._apply_equipped_banner_effects(self.blue_team_state, log=log)
    def _apply_equipped_banner_effects(
        self,
        blue_team: List[Dict],
        *,
        log: bool = True,
    ) -> None:
        self._clear_equipped_banner_effects(blue_team)
        if not blue_team:
            return

        if not self._hero_has_banner_bearer(units=blue_team):
            self.equipped_banner_items = [None] * int(self.BANNER_EQUIP_SLOTS)
            return

        self._sync_equipped_banner_items(units=blue_team)
        active_banner = next(
            (
                str(item_name)
                for item_name in list(self.equipped_banner_items or [])
                if self._is_banner_item_name(str(item_name or ""))
            ),
            None,
        )
        if not active_banner:
            return

        effect = self._banner_effect_definition(active_banner)
        damage_multiplier = float(effect.get("damage_multiplier", 1.0) or 1.0)
        accuracy_bonus = int(effect.get("accuracy_bonus", 0) or 0)
        initiative_multiplier = float(effect.get("initiative_multiplier", 1.0) or 1.0)
        armor_bonus = int(effect.get("armor_bonus", 0) or 0)
        affected_units = 0

        for unit in blue_team:
            if not isinstance(unit, dict) or self._is_empty_blue_unit(unit):
                continue

            unit["campaign_active_banner"] = str(active_banner)
            affected_units += 1
            if abs(damage_multiplier - 1.0) > 1e-9:
                base_damage = self._normalize_damage_value(unit.get("damage", 0))
                unit["campaign_banner_base_damage"] = int(base_damage)
                unit["campaign_banner_damage_multiplier"] = float(damage_multiplier)
                unit["damage"] = self._normalize_damage_value(
                    float(base_damage) * float(damage_multiplier)
                )

            if accuracy_bonus != 0:
                base_accuracy = self._normalize_accuracy_value(unit.get("accuracy", 0))
                unit["campaign_banner_base_accuracy"] = int(base_accuracy)
                unit["campaign_banner_accuracy_bonus"] = int(accuracy_bonus)
                unit["accuracy"] = self._normalize_accuracy_value(
                    int(base_accuracy) + int(accuracy_bonus)
                )

            if abs(initiative_multiplier - 1.0) > 1e-9:
                base_initiative = int(
                    self._normalize_damage_value(
                        unit.get("initiative_base", unit.get("initiative", 0))
                    )
                )
                unit["campaign_banner_base_initiative"] = int(base_initiative)
                unit["campaign_banner_initiative_multiplier"] = float(initiative_multiplier)
                boosted_initiative = self._normalize_damage_value(
                    float(base_initiative) * float(initiative_multiplier)
                )
                unit["initiative_base"] = int(boosted_initiative)
                unit["initiative"] = int(boosted_initiative)

            if armor_bonus != 0:
                base_armor = self._normalize_armor_value(unit.get("armor", 0))
                unit["campaign_banner_base_armor"] = int(base_armor)
                unit["campaign_banner_armor_bonus"] = int(armor_bonus)
                unit["armor"] = self._normalize_armor_value(
                    int(base_armor) + int(armor_bonus)
                )
                unit["base_armor"] = int(unit.get("armor", 0) or 0)

        if log and affected_units > 0:
            self._log(
                f"Active banner '{active_banner}' applied to {affected_units} BLUE units "
                f"(damage x{damage_multiplier:.3g}, accuracy +{accuracy_bonus}, "
                f"initiative x{initiative_multiplier:.3g}, armor +{armor_bonus})."
            )
    def _clear_equipped_artifact_effects(self, blue_team: Optional[List[Dict]]) -> None:
        if not blue_team:
            return

        for unit in blue_team:
            if "campaign_artifact_base_damage" in unit:
                unit["damage"] = self._normalize_damage_value(
                    unit.get(
                        "campaign_artifact_base_damage",
                        unit.get("damage", 0),
                    )
                )
            if "campaign_artifact_base_initiative" in unit:
                base_initiative = int(
                    self._normalize_damage_value(
                        unit.get(
                            "campaign_artifact_base_initiative",
                            unit.get("initiative_base", unit.get("initiative", 0)),
                        )
                    )
                )
                unit["initiative_base"] = int(base_initiative)
                unit["initiative"] = int(base_initiative)
            if "campaign_artifact_base_armor" in unit:
                base_armor = self._normalize_armor_value(
                    unit.get(
                        "campaign_artifact_base_armor",
                        unit.get("armor", 0),
                    )
                )
                unit["armor"] = int(base_armor)
                unit["base_armor"] = int(base_armor)
            unit.pop("campaign_artifact_base_damage", None)
            unit.pop("campaign_artifact_base_damage_secondary", None)
            unit.pop("campaign_artifact_base_initiative", None)
            unit.pop("campaign_artifact_base_armor", None)
            unit.pop("campaign_artifact_damage_multiplier", None)
            unit.pop("campaign_artifact_initiative_multiplier", None)
            unit.pop("campaign_artifact_armor_bonus", None)
            unit.pop("campaign_active_artifacts", None)
    def _apply_equipped_artifact_effects(
        self,
        blue_team: List[Dict],
        *,
        log: bool = True,
    ) -> None:
        self._clear_equipped_artifact_effects(blue_team)
        if not blue_team:
            return

        if not self._hero_has_artifact_knowledge(units=blue_team):
            self.equipped_artifact_items = [None] * int(self.ARTIFACT_EQUIP_SLOTS)
            self.artifact_auto_equip_next_slot = 0
            return

        self._sync_equipped_artifact_items(units=blue_team)
        active_artifacts = [
            str(item_name)
            for item_name in list(self.equipped_artifact_items or [])
            if self._is_artifact_item_name(str(item_name or ""))
        ]
        if not active_artifacts:
            return

        hero = self._resolve_travel_hero(units=blue_team)
        if hero is None:
            return

        damage_multiplier = 1.0
        initiative_multiplier = 1.0
        armor_bonus = 0
        for item_name in active_artifacts:
            effect = self._artifact_effect_definition(item_name)
            damage_multiplier *= float(effect.get("damage_multiplier", 1.0) or 1.0)
            initiative_multiplier *= float(effect.get("initiative_multiplier", 1.0) or 1.0)
            armor_bonus += int(effect.get("armor_bonus", 0) or 0)

        hero["campaign_active_artifacts"] = list(active_artifacts)
        if abs(damage_multiplier - 1.0) > 1e-9:
            base_damage = self._normalize_damage_value(hero.get("damage", 0))
            hero["campaign_artifact_base_damage"] = int(base_damage)
            hero["campaign_artifact_damage_multiplier"] = float(damage_multiplier)
            hero["damage"] = self._normalize_damage_value(
                float(base_damage) * float(damage_multiplier)
            )

        if abs(initiative_multiplier - 1.0) > 1e-9:
            base_initiative = int(
                hero.get("initiative_base", hero.get("initiative", 0)) or 0
            )
            hero["campaign_artifact_base_initiative"] = int(base_initiative)
            hero["campaign_artifact_initiative_multiplier"] = float(initiative_multiplier)
            boosted_initiative = self._normalize_damage_value(
                float(base_initiative) * float(initiative_multiplier)
            )
            hero["initiative_base"] = int(boosted_initiative)
            hero["initiative"] = int(boosted_initiative)

        if armor_bonus != 0:
            base_armor = self._normalize_armor_value(hero.get("armor", 0))
            hero["campaign_artifact_base_armor"] = int(base_armor)
            hero["campaign_artifact_armor_bonus"] = int(armor_bonus)
            hero["armor"] = self._normalize_armor_value(int(base_armor) + int(armor_bonus))
            hero["base_armor"] = int(hero.get("armor", 0) or 0)

        if log:
            hero_name = str(hero.get("name", "Герой") or "Герой")
            self._log(
                f"Активны экипированные артефакты героя {hero_name}: {', '.join(active_artifacts)} "
                f"(урон x{damage_multiplier:.3g}, инициатива x{initiative_multiplier:.3g}, "
                f"броня +{int(armor_bonus)})."
            )
    def _compute_enemy_defeat_reward(self, enemy_id: Optional[int]) -> float:
        """Enemy defeat reward including ruin bonus."""
        reward = float(self.reward_defeat_enemy)
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            normalized_enemy_id = -1
        if normalized_enemy_id in self.RUIN_REWARD_BY_ENEMY_ID:
            reward += float(self.reward_ruin_clear_bonus)
        return reward
