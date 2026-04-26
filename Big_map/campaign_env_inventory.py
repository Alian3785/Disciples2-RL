# campaign_env_inventory.py
"""CampaignInventoryMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignInventoryMixin:
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
        return {
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
        return 0
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
        normalized_slots: List[Optional[str]] = [None] * int(self.BATTLE_EQUIP_SLOTS)
        reserved: Dict[str, int] = {}
        current_slots = list(self.equipped_hero_items or [])
        for slot_index in range(int(self.BATTLE_EQUIP_SLOTS)):
            item_name = (
                str(current_slots[slot_index] or "")
                if slot_index < len(current_slots) and current_slots[slot_index] is not None
                else ""
            )
            if item_name not in self.BATTLE_EQUIPPABLE_ITEM_NAMES:
                continue
            available_count = int(self._battle_equip_item_available_count(item_name))
            already_reserved = int(reserved.get(item_name, 0))
            if already_reserved >= available_count:
                continue
            normalized_slots[slot_index] = item_name
            reserved[item_name] = already_reserved + 1
        self.equipped_hero_items = normalized_slots
    def _can_equip_battle_item(self, slot_index: int, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        if normalized_name not in self.BATTLE_EQUIPPABLE_ITEM_NAMES:
            return False
        if not (0 <= int(slot_index) < int(self.BATTLE_EQUIP_SLOTS)):
            return False
        if self.battle_item_equip_used_this_turn:
            return False
        self._sync_equipped_hero_items()
        if str(self.equipped_hero_items[int(slot_index)] or "") == normalized_name:
            return False
        available_count = int(self._battle_equip_item_available_count(normalized_name))
        reserved_elsewhere = self._count_equipped_battle_items(
            normalized_name,
            exclude_slot=int(slot_index),
        )
        return available_count > reserved_elsewhere
    def _equip_battle_item(self, slot_index: int, item_name: str) -> bool:
        if not self._can_equip_battle_item(slot_index, item_name):
            return False
        self.equipped_hero_items[int(slot_index)] = str(item_name)
        self._sync_equipped_hero_items()
        return True
    def _consume_counter_backed_item(self, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        if normalized_name == self.BONUS_SMALL_HEAL_ITEM_NAME:
            if self._healing_bottles_left() <= 0:
                return False
            self.healing_bottles_used = min(
                self.healing_bottles_used + 1,
                self._max_healing_bottles_available(),
            )
            self._consume_hero_item(self.BONUS_SMALL_HEAL_ITEM_NAME)
            return True
        if normalized_name == self.BONUS_LARGE_HEAL_ITEM_NAME:
            if self._heal_bottles_left() <= 0:
                return False
            self.heal_bottles_used = min(
                self.heal_bottles_used + 1,
                self._max_heal_bottles_available(),
            )
            self._consume_hero_item(self.BONUS_LARGE_HEAL_ITEM_NAME)
            return True
        if normalized_name == self.BONUS_REVIVE_ITEM_NAME:
            if self._revive_bottles_left() <= 0:
                return False
            self.revive_bottles_used = min(
                self.revive_bottles_used + 1,
                self._max_revive_bottles_available(),
            )
            self._consume_hero_item(self.BONUS_REVIVE_ITEM_NAME)
            return True
        return False
    def _consume_battle_equipable_item(self, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        consumed = False
        if normalized_name == self.HEALING_OINTMENT_ITEM_NAME:
            consumed = self._consume_hero_item(self.HEALING_OINTMENT_ITEM_NAME)
        else:
            consumed = self._consume_counter_backed_item(normalized_name)
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
    def _invalidate_inventory_cache(self) -> None:
        self._inventory_cache_valid = False
        self._inventory_cache_signature = ()
        self._hero_item_counts_cache = {}
        self._scroll_item_counts_cache = {}
        self._scroll_inventory_entries_cache = ()
        self._available_scroll_spell_entries_cache = ()
        self._scroll_cast_slot_entries_cache = ()
        self._total_scroll_count_cache = 0
    def _ensure_inventory_cache(self) -> None:
        current_signature = tuple(self._hero_item_name(entry) for entry in self.heroitems)
        if self._inventory_cache_valid and current_signature == self._inventory_cache_signature:
            return

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
        self._scroll_cast_slot_entries_cache = tuple(
            available_entries[: int(self.MAX_SCROLL_CAST_ACTIONS)]
        )
        self._total_scroll_count_cache = int(sum(scroll_item_counts.values()))
        self._inventory_cache_signature = current_signature
        self._inventory_cache_valid = True
    def is_scroll_item(self, item_name: str) -> bool:
        return bool(self.scroll_item_definition(item_name))
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
    def count_scroll_spell(self, spell_id: str) -> int:
        normalized_spell_id = str(spell_id or "")
        if not normalized_spell_id:
            return 0
        for definition in SCROLL_ITEM_DEFINITIONS:
            if str(definition.get("spell_id", "") or "") == normalized_spell_id:
                return self.count_scroll_item(str(definition.get("item_name", "") or ""))
        return 0
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
    def _sync_equipped_artifact_items(self) -> None:
        normalized_slots: List[Optional[str]] = [None] * int(self.ARTIFACT_EQUIP_SLOTS)
        reserved: Dict[str, int] = {}
        current_slots = list(getattr(self, "equipped_artifact_items", []) or [])
        available_counts: Dict[str, int] = {}

        for entry in self.heroitems:
            item_name = self._hero_item_name(entry)
            if not self._is_artifact_item_name(item_name):
                continue
            available_counts[item_name] = available_counts.get(item_name, 0) + 1

        for slot_index in range(int(self.ARTIFACT_EQUIP_SLOTS)):
            item_name = (
                str(current_slots[slot_index] or "")
                if slot_index < len(current_slots) and current_slots[slot_index] is not None
                else ""
            )
            if not self._is_artifact_item_name(item_name):
                continue
            available_count = int(available_counts.get(item_name, 0) or 0)
            already_reserved = int(reserved.get(item_name, 0) or 0)
            if already_reserved >= available_count:
                continue
            normalized_slots[slot_index] = item_name
            reserved[item_name] = already_reserved + 1

        self.equipped_artifact_items = normalized_slots
        self.artifact_auto_equip_next_slot = (
            int(getattr(self, "artifact_auto_equip_next_slot", 0) or 0)
            % max(1, int(self.ARTIFACT_EQUIP_SLOTS))
        )
    def _auto_equip_artifact_item(self, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "")
        if not self._is_artifact_item_name(normalized_name):
            return {
                "artifact_auto_equipped": False,
                "artifact_auto_equip_slot": None,
                "artifact_auto_equip_previous": None,
            }

        self._sync_equipped_artifact_items()
        slot_index = int(self.artifact_auto_equip_next_slot or 0) % max(
            1,
            int(self.ARTIFACT_EQUIP_SLOTS),
        )
        previous_item = self.equipped_artifact_items[slot_index]
        self.equipped_artifact_items[slot_index] = normalized_name
        self.artifact_auto_equip_next_slot = (slot_index + 1) % max(
            1,
            int(self.ARTIFACT_EQUIP_SLOTS),
        )
        self._log(
            f"Артефакт '{normalized_name}' автоматически экипирован в слот "
            f"{slot_index + 1} (предыдущий: {previous_item or 'пусто'})."
        )
        return {
            "artifact_auto_equipped": True,
            "artifact_auto_equip_slot": int(slot_index + 1),
            "artifact_auto_equip_previous": previous_item,
        }
    def _append_hero_item(self, item_name: str) -> None:
        normalized_name = str(item_name or "")
        self.heroitems.append(self._make_hero_item_entry(normalized_name))
        self._invalidate_inventory_cache()
    def _add_hero_item(self, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "")
        self._append_hero_item(normalized_name)
        artifact_info = self._auto_equip_artifact_item(normalized_name)
        if bool(artifact_info.get("artifact_auto_equipped")):
            self._refresh_campaign_artifact_effects(log=True)
        return artifact_info
    def _artifact_state_info(self) -> Dict[str, object]:
        self._refresh_campaign_artifact_effects(log=False)
        return {
            "equipped_artifact_items": list(self.equipped_artifact_items),
            "artifact_auto_equip_next_slot": int(self.artifact_auto_equip_next_slot),
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
    def _hero_item_aliases(cls, item_name: str) -> Tuple[str, ...]:
        normalized_name = str(item_name or "")
        aliases = {normalized_name}
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
    def _count_hero_item(self, item_name: str) -> int:
        if not item_name:
            return 0
        self._ensure_inventory_cache()
        normalized_names = set(self._hero_item_aliases(item_name))
        return int(
            sum(int(self._hero_item_counts_cache.get(name, 0) or 0) for name in normalized_names)
        )
    def _hero_item_available(self, item_name: str) -> bool:
        return self._count_hero_item(item_name) > 0
    def _consume_hero_item(self, item_name: str) -> bool:
        if not item_name:
            return False
        normalized_names = set(self._hero_item_aliases(item_name))
        for idx, entry in enumerate(self.heroitems):
            if self._hero_item_name(entry) in normalized_names:
                del self.heroitems[idx]
                self._invalidate_inventory_cache()
                self._sync_equipped_artifact_items()
                self._refresh_campaign_artifact_effects(log=False)
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
        site_names = self._merchant_sites_at_position(position)
        sale_info: Dict[str, object] = {
            "merchant_auto_sale": False,
            "merchant_sold_items": [],
            "merchant_sold_items_count": 0,
            "merchant_sale_gold": 0.0,
            "merchant_sale_reward": 0.0,
        }
        if not site_names or not self.heroitems:
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
            return sale_info

        self.heroitems = kept_items
        self._invalidate_inventory_cache()
        self._sync_equipped_artifact_items()
        self._refresh_campaign_artifact_effects(log=False)
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
        return sale_info
    def _combat_potion_reward_value(self) -> float:
        return float(self.reward_defeat_enemy)
    def _count_collected_combat_potions(self, collected_chests: List[Dict]) -> int:
        tracked_items = {
            self.INVULNERABILITY_POTION_ITEM_NAME,
            self.STRENGTH_POTION_ITEM_NAME,
        }
        count = 0
        for chest_data in collected_chests:
            for item_name in chest_data.get("items", []):
                if str(item_name or "") in tracked_items:
                    count += 1
        return int(count)
    def _active_invulnerability_bonus_for_position(self, position: int) -> int:
        return (
            int(self.INVULNERABILITY_POTION_ARMOR_BONUS)
            if int(position) in self.active_invulnerability_potion_positions
            else 0
        )
    def _active_strength_multiplier_for_position(self, position: int) -> float:
        return (
            float(self.STRENGTH_POTION_DAMAGE_MULTIPLIER)
            if int(position) in self.active_strength_potion_positions
            else 1.0
        )
    def _clear_expired_combat_potion_effects(self) -> None:
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
    def _step_equip_battle_item(self, action: int):
        slot_index = 0
        action_start = self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START
        if int(action) >= self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START:
            slot_index = 1
            action_start = self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START
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
        equip_turn_locked = bool(self.battle_item_equip_used_this_turn)
        if (
            not equip_turn_locked
            and item_name is not None
            and 0 <= int(slot_index) < len(self.equipped_hero_items)
        ):
            previous_item = self.equipped_hero_items[int(slot_index)]
            equipped = self._equip_battle_item(int(slot_index), str(item_name))
            if equipped:
                self.battle_item_equip_used_this_turn = True
                self.battle_items_equipped_total += 1
                if not previous_item:
                    reward = float(self.reward_battle_item_equip)
                self._log(
                    f"Боевой слот {int(slot_index) + 1}: экипирован предмет '{str(item_name)}'"
                )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "equip_battle_item_action": True,
            "equip_battle_item_slot": int(slot_index) + 1,
            "equip_battle_item_name": str(item_name or ""),
            "equip_battle_item_previous": previous_item,
            "equip_battle_item_applied": bool(equipped),
            "equip_battle_item_turn_locked": bool(equip_turn_locked),
            "equip_battle_item_reward": float(reward),
            "battle_items_equipped_total": int(self.battle_items_equipped_total),
            "battle_items_used_total": int(self.battle_items_used_total),
            "equipped_hero_items": list(self.equipped_hero_items),
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
            if selected_bottle_kind == "ointment":
                if self._consume_battle_equipable_item(self.HEALING_OINTMENT_ITEM_NAME):
                    healed_amount, unit_name = self._heal_unit_at_position(
                        position=target_pos,
                        heal_amount=selected_heal_amount,
                        source_label=self.HEALING_OINTMENT_ITEM_NAME,
                    )
                    if healed_amount <= 0.0:
                        self._append_hero_item(self.HEALING_OINTMENT_ITEM_NAME)
                else:
                    selected_bottle_kind = None
                    selected_heal_amount = 0.0
            else:
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
                    self._consume_battle_equipable_item(
                        self.BONUS_LARGE_HEAL_ITEM_NAME
                        if selected_bottle_kind == "large"
                        else self.BONUS_SMALL_HEAL_ITEM_NAME
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
            "extra_heal_bottles": self.extra_heal_bottles,
            "heal_bottles_left": self._heal_bottles_left(),
            "max_heal_bottles": self._max_heal_bottles_available(),
            "healing_bottles_used": self.healing_bottles_used,
            "extra_healing_bottles": self.extra_healing_bottles,
            "healing_bottles_left": max(
                0, self._max_healing_bottles_available() - self.healing_bottles_used
            ),
            "max_healing_bottles": self._max_healing_bottles_available(),
            "healing_ointment_left": self._count_hero_item(self.HEALING_OINTMENT_ITEM_NAME),
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

        revived, unit_name = (False, None)
        if target_pos is not None and self._consume_battle_equipable_item(
            self.BONUS_REVIVE_ITEM_NAME
        ):
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
            "extra_revive_bottles": self.extra_revive_bottles,
            "revive_bottles_left": self._revive_bottles_left(),
            "max_revive_bottles": self._max_revive_bottles_available(),
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
    def _apply_invulnerability_potion_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0:
                return False, None
            if int(position) in self.active_invulnerability_potion_positions:
                return False, None
            self.active_invulnerability_potion_positions.add(int(position))
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"Зелье неуязвимости: {name} (pos {position}) получает "
                f"+{self.INVULNERABILITY_POTION_ARMOR_BONUS} брони до конца текущего хода"
            )
            return True, name
        return False, None
    def _apply_strength_potion_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0:
                return False, None
            if int(position) in self.active_strength_potion_positions:
                return False, None
            self.active_strength_potion_positions.add(int(position))
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"Зелье силы: {name} (pos {position}) получает +30% урона "
                "до конца текущего хода"
            )
            return True, name
        return False, None
    def _can_apply_single_turn_effect_position(
        self,
        position: int,
        item_name: str,
        active_positions: set[int],
    ) -> bool:
        if not self._hero_item_available(item_name):
            return False
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return max_hp > 0 and hp > 0 and int(position) not in active_positions
        return False
    def _can_apply_energy_elixir_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.ENERGY_ELIXIR_ITEM_NAME,
            self.active_energy_elixir_positions,
        )
    def _can_apply_haste_elixir_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.HASTE_ELIXIR_ITEM_NAME,
            self.active_haste_elixir_positions,
        )
    def _can_apply_fire_ward_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.FIRE_WARD_ITEM_NAME,
            self.active_fire_ward_positions,
        )
    def _can_apply_earth_ward_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.EARTH_WARD_ITEM_NAME,
            self.active_earth_ward_positions,
        )
    def _can_apply_water_ward_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.WATER_WARD_ITEM_NAME,
            self.active_water_ward_positions,
        )
    def _can_apply_air_ward_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.AIR_WARD_ITEM_NAME,
            self.active_air_ward_positions,
        )
    def _can_apply_permanent_elixir_position(self, position: int, item_name: str) -> bool:
        if not self._hero_item_available(item_name):
            return False
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return max_hp > 0 and hp > 0
        return False
    def _can_apply_titan_elixir_position(self, position: int) -> bool:
        return self._can_apply_permanent_elixir_position(
            position,
            self.TITAN_ELIXIR_ITEM_NAME,
        )
    def _can_apply_supreme_elixir_position(self, position: int) -> bool:
        return self._can_apply_permanent_elixir_position(
            position,
            self.SUPREME_ELIXIR_ITEM_NAME,
        )
    def _apply_position_flag_effect(
        self,
        position: int,
        active_positions: set[int],
        log_message: str,
    ) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0 or int(position) in active_positions:
                return False, None
            active_positions.add(int(position))
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(log_message.format(name=name, position=int(position)))
            return True, name
        return False, None
    def _apply_energy_elixir_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_energy_elixir_positions,
            "Эликсир энергии: {name} (pos {position}) получает +50% урона до конца текущего хода",
        )
    def _apply_haste_elixir_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_haste_elixir_positions,
            "Эликсир быстроты: {name} (pos {position}) получает +60% инициативы до конца текущего хода",
        )
    def _apply_fire_ward_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_fire_ward_positions,
            "Защита от магии Огня: {name} (pos {position}) получает resistance Fire до конца текущего хода",
        )
    def _apply_earth_ward_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_earth_ward_positions,
            "Защита от магии Земли: {name} (pos {position}) получает resistance Earth до конца текущего хода",
        )
    def _apply_water_ward_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_water_ward_positions,
            "Защита от магии Воды: {name} (pos {position}) получает resistance Water до конца текущего хода",
        )
    def _apply_air_ward_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_air_ward_positions,
            "Защита от магии Воздуха: {name} (pos {position}) получает resistance Air до конца текущего хода",
        )
    def _apply_titan_elixir_bonus_to_unit(
        self,
        unit: Dict,
        *,
        increment_uses: bool = True,
    ) -> None:
        base_damage = self._normalize_damage_value(unit.get("damage", 0))
        base_damage_secondary = self._normalize_damage_value(unit.get("damage_secondary", 0))
        unit["damage"] = self._normalize_damage_value(
            float(base_damage) * float(self.TITAN_ELIXIR_DAMAGE_MULTIPLIER)
        )
        unit["damage_secondary"] = self._normalize_damage_value(
            float(base_damage_secondary) * float(self.TITAN_ELIXIR_DAMAGE_MULTIPLIER)
        )
        unit["original_damage"] = int(unit["damage"])
        if increment_uses:
            unit["campaign_titan_elixir_uses"] = (
                max(0, int(unit.get("campaign_titan_elixir_uses", 0) or 0)) + 1
            )
    def _apply_supreme_elixir_bonus_to_unit(
        self,
        unit: Dict,
        *,
        increment_uses: bool = True,
    ) -> None:
        hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
        max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
        new_max_hp = self._normalize_health_value(
            float(max_hp) * float(self.SUPREME_ELIXIR_HEALTH_MULTIPLIER)
        )
        if new_max_hp <= int(round(max_hp)):
            new_max_hp = int(round(max_hp)) + 1
        new_hp = min(
            new_max_hp,
            self._normalize_health_value(
                float(hp) * float(self.SUPREME_ELIXIR_HEALTH_MULTIPLIER)
            ),
        )
        unit["maxhp"] = int(new_max_hp)
        unit["max_health"] = int(new_max_hp)
        unit["hp"] = int(new_hp)
        unit["health"] = int(new_hp)
        if increment_uses:
            unit["campaign_supreme_elixir_uses"] = (
                max(0, int(unit.get("campaign_supreme_elixir_uses", 0) or 0)) + 1
            )
    def _reapply_persistent_elixir_bonuses_to_promoted_unit(
        self,
        source_unit: Dict,
        upgraded_unit: Dict,
    ) -> None:
        upgraded_unit.pop("campaign_titan_elixir_uses", None)
        upgraded_unit.pop("campaign_supreme_elixir_uses", None)

        titan_uses = max(0, int(source_unit.get("campaign_titan_elixir_uses", 0) or 0))
        supreme_uses = max(0, int(source_unit.get("campaign_supreme_elixir_uses", 0) or 0))

        for _ in range(titan_uses):
            self._apply_titan_elixir_bonus_to_unit(upgraded_unit, increment_uses=True)
        for _ in range(supreme_uses):
            self._apply_supreme_elixir_bonus_to_unit(upgraded_unit, increment_uses=True)
    def _apply_titan_elixir_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0:
                return False, None

            self._apply_titan_elixir_bonus_to_unit(unit, increment_uses=True)
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"Эликсир Силы титана: {name} (pos {position}) получает "
                f"+10% к урону навсегда на эпизод"
            )
            return True, name
        return False, None
    def _apply_supreme_elixir_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0:
                return False, None

            self._apply_supreme_elixir_bonus_to_unit(unit, increment_uses=True)
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"Эликсир Всевышнего: {name} (pos {position}) получает "
                f"+15% к max HP навсегда на эпизод"
            )
            return True, name
        return False, None
    def _step_apply_single_turn_position_item(
        self,
        *,
        action: int,
        action_start: int,
        positions: List[int],
        item_name: str,
        can_apply_fn,
        apply_fn,
        info_key: str,
    ):
        idx = action - action_start
        target_pos = positions[idx] if 0 <= idx < len(positions) else None
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        applied = False
        consumed = False
        unit_name = None
        if (
            target_pos is not None
            and can_apply_fn(target_pos)
            and self._consume_hero_item(item_name)
        ):
            consumed = True
            applied, unit_name = apply_fn(target_pos)
            if not applied:
                self._append_hero_item(item_name)
                consumed = False
            else:
                reward = self._combat_potion_reward_value()
                self.combat_potion_battle_bonus_pending = True

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            info_key: True,
            "combat_potion_action": True,
            "combat_potion_item": item_name,
            "combat_potion_target_pos": target_pos,
            "combat_potion_applied": applied,
            "combat_potion_consumed": consumed,
            "combat_potion_unit_name": unit_name,
            "combat_potion_reward": float(reward),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_apply_invulnerability_potion(self, action: int):
        idx = action - self.GRID_INVULNERABILITY_ACTION_START
        target_pos = (
            self.INVULNERABILITY_POTION_POSITIONS[idx]
            if 0 <= idx < len(self.INVULNERABILITY_POTION_POSITIONS)
            else None
        )
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        applied = False
        consumed = False
        unit_name = None
        if (
            target_pos is not None
            and self._can_apply_invulnerability_potion_position(target_pos)
            and self._consume_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME)
        ):
            consumed = True
            applied, unit_name = self._apply_invulnerability_potion_to_position(target_pos)
            if not applied:
                self._append_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME)
                consumed = False
            else:
                reward = self._combat_potion_reward_value()
                self.combat_potion_battle_bonus_pending = True

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "invulnerability_potion_action": True,
            "combat_potion_action": True,
            "combat_potion_item": self.INVULNERABILITY_POTION_ITEM_NAME,
            "combat_potion_target_pos": target_pos,
            "combat_potion_applied": applied,
            "combat_potion_consumed": consumed,
            "combat_potion_unit_name": unit_name,
            "combat_potion_reward": float(reward),
            "invulnerability_potions_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_potions_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
            "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
            "active_strength_positions": sorted(self.active_strength_potion_positions),
            "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_apply_strength_potion(self, action: int):
        idx = action - self.GRID_STRENGTH_ACTION_START
        target_pos = (
            self.STRENGTH_POTION_POSITIONS[idx]
            if 0 <= idx < len(self.STRENGTH_POTION_POSITIONS)
            else None
        )
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        applied = False
        consumed = False
        unit_name = None
        if (
            target_pos is not None
            and self._can_apply_strength_potion_position(target_pos)
            and self._consume_hero_item(self.STRENGTH_POTION_ITEM_NAME)
        ):
            consumed = True
            applied, unit_name = self._apply_strength_potion_to_position(target_pos)
            if not applied:
                self._append_hero_item(self.STRENGTH_POTION_ITEM_NAME)
                consumed = False
            else:
                reward = self._combat_potion_reward_value()
                self.combat_potion_battle_bonus_pending = True

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "strength_potion_action": True,
            "combat_potion_action": True,
            "combat_potion_item": self.STRENGTH_POTION_ITEM_NAME,
            "combat_potion_target_pos": target_pos,
            "combat_potion_applied": applied,
            "combat_potion_consumed": consumed,
            "combat_potion_unit_name": unit_name,
            "combat_potion_reward": float(reward),
            "invulnerability_potions_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_potions_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
            "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
            "active_strength_positions": sorted(self.active_strength_potion_positions),
            "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_apply_energy_elixir(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_ENERGY_ACTION_START,
            positions=self.ENERGY_ELIXIR_POSITIONS,
            item_name=self.ENERGY_ELIXIR_ITEM_NAME,
            can_apply_fn=self._can_apply_energy_elixir_position,
            apply_fn=self._apply_energy_elixir_to_position,
            info_key="energy_elixir_action",
        )
    def _step_apply_haste_elixir(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_HASTE_ACTION_START,
            positions=self.HASTE_ELIXIR_POSITIONS,
            item_name=self.HASTE_ELIXIR_ITEM_NAME,
            can_apply_fn=self._can_apply_haste_elixir_position,
            apply_fn=self._apply_haste_elixir_to_position,
            info_key="haste_elixir_action",
        )
    def _step_apply_fire_ward(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_FIRE_WARD_ACTION_START,
            positions=self.FIRE_WARD_POSITIONS,
            item_name=self.FIRE_WARD_ITEM_NAME,
            can_apply_fn=self._can_apply_fire_ward_position,
            apply_fn=self._apply_fire_ward_to_position,
            info_key="fire_ward_action",
        )
    def _step_apply_earth_ward(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_EARTH_WARD_ACTION_START,
            positions=self.EARTH_WARD_POSITIONS,
            item_name=self.EARTH_WARD_ITEM_NAME,
            can_apply_fn=self._can_apply_earth_ward_position,
            apply_fn=self._apply_earth_ward_to_position,
            info_key="earth_ward_action",
        )
    def _step_apply_water_ward(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_WATER_WARD_ACTION_START,
            positions=self.WATER_WARD_POSITIONS,
            item_name=self.WATER_WARD_ITEM_NAME,
            can_apply_fn=self._can_apply_water_ward_position,
            apply_fn=self._apply_water_ward_to_position,
            info_key="water_ward_action",
        )
    def _step_apply_air_ward(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_AIR_WARD_ACTION_START,
            positions=self.AIR_WARD_POSITIONS,
            item_name=self.AIR_WARD_ITEM_NAME,
            can_apply_fn=self._can_apply_air_ward_position,
            apply_fn=self._apply_air_ward_to_position,
            info_key="air_ward_action",
        )
    def _step_apply_permanent_position_item(
        self,
        *,
        action: int,
        action_start: int,
        positions: List[int],
        item_name: str,
        can_apply_fn,
        apply_fn,
        info_key: str,
    ):
        idx = action - action_start
        target_pos = positions[idx] if 0 <= idx < len(positions) else None
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        applied = False
        consumed = False
        unit_name = None
        if (
            target_pos is not None
            and can_apply_fn(target_pos)
            and self._consume_hero_item(item_name)
        ):
            consumed = True
            applied, unit_name = apply_fn(target_pos)
            if not applied:
                self._append_hero_item(item_name)
                consumed = False

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            info_key: True,
            "persistent_elixir_action": True,
            "persistent_elixir_item": item_name,
            "persistent_elixir_target_pos": target_pos,
            "persistent_elixir_applied": applied,
            "persistent_elixir_consumed": consumed,
            "persistent_elixir_unit_name": unit_name,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_apply_titan_elixir(self, action: int):
        return self._step_apply_permanent_position_item(
            action=action,
            action_start=self.GRID_TITAN_ELIXIR_ACTION_START,
            positions=self.TITAN_ELIXIR_POSITIONS,
            item_name=self.TITAN_ELIXIR_ITEM_NAME,
            can_apply_fn=self._can_apply_titan_elixir_position,
            apply_fn=self._apply_titan_elixir_to_position,
            info_key="titan_elixir_action",
        )
    def _step_apply_supreme_elixir(self, action: int):
        return self._step_apply_permanent_position_item(
            action=action,
            action_start=self.GRID_SUPREME_ELIXIR_ACTION_START,
            positions=self.SUPREME_ELIXIR_POSITIONS,
            item_name=self.SUPREME_ELIXIR_ITEM_NAME,
            can_apply_fn=self._can_apply_supreme_elixir_position,
            apply_fn=self._apply_supreme_elixir_to_position,
            info_key="supreme_elixir_action",
        )
    def _apply_active_blue_potion_effects(self, blue_team: List[Dict]) -> None:
        if not blue_team:
            return

        buffed_invulnerability: List[int] = []
        buffed_strength: List[int] = []
        buffed_energy: List[int] = []
        buffed_haste: List[int] = []
        buffed_fire: List[int] = []
        buffed_earth: List[int] = []
        buffed_water: List[int] = []
        buffed_air: List[int] = []

        for unit in blue_team:
            position = int(unit.get("position", -1) or -1)
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if position <= 0 or max_hp <= 0 or hp <= 0:
                continue

            if position in self.active_invulnerability_potion_positions:
                base_armor = self._normalize_armor_value(unit.get("armor", 0))
                unit["campaign_potion_base_armor"] = int(base_armor)
                unit["armor"] = int(base_armor) + int(self.INVULNERABILITY_POTION_ARMOR_BONUS)
                unit["campaign_invulnerability_potion_bonus"] = int(
                    self.INVULNERABILITY_POTION_ARMOR_BONUS
                )
                buffed_invulnerability.append(position)

            damage_multiplier = 1.0
            if position in self.active_strength_potion_positions:
                damage_multiplier *= float(self.STRENGTH_POTION_DAMAGE_MULTIPLIER)
            if position in self.active_energy_elixir_positions:
                damage_multiplier *= float(self.ENERGY_ELIXIR_DAMAGE_MULTIPLIER)
            if damage_multiplier > 1.0:
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
                if position in self.active_strength_potion_positions:
                    unit["campaign_strength_potion_multiplier"] = float(
                        self.STRENGTH_POTION_DAMAGE_MULTIPLIER
                    )
                    buffed_strength.append(position)
                if position in self.active_energy_elixir_positions:
                    unit["campaign_energy_elixir_multiplier"] = float(
                        self.ENERGY_ELIXIR_DAMAGE_MULTIPLIER
                    )
                    buffed_energy.append(position)

            initiative_multiplier = 1.0
            if position in self.active_haste_elixir_positions:
                initiative_multiplier = max(
                    initiative_multiplier,
                    float(self.HASTE_ELIXIR_INITIATIVE_MULTIPLIER),
                )
            if initiative_multiplier > 1.0:
                base_initiative = int(unit.get("initiative_base", unit.get("initiative", 0)) or 0)
                unit["campaign_potion_base_initiative"] = int(base_initiative)
                boosted_initiative = self._normalize_damage_value(
                    float(base_initiative) * initiative_multiplier
                )
                unit["initiative_base"] = int(boosted_initiative)
                unit["initiative"] = int(boosted_initiative)
                unit["campaign_initiative_potion_multiplier"] = float(initiative_multiplier)
                buffed_haste.append(position)

            added_resistance: List[str] = []
            resistance = [str(res) for res in (unit.get("resistance") or [])]
            for active_positions, element, bucket in (
                (self.active_fire_ward_positions, "Fire", buffed_fire),
                (self.active_earth_ward_positions, "Earth", buffed_earth),
                (self.active_water_ward_positions, "Water", buffed_water),
                (self.active_air_ward_positions, "Air", buffed_air),
            ):
                if position not in active_positions or element in resistance:
                    continue
                resistance.append(element)
                added_resistance.append(element)
                bucket.append(position)
            if added_resistance:
                unit["resistance"] = resistance
                unit["campaign_potion_added_resistance"] = added_resistance

        if buffed_invulnerability:
            self._log(
                "Активно зелье неуязвимости для позиций "
                f"{sorted(buffed_invulnerability)} перед началом боя"
            )
        if buffed_strength:
            self._log(
                "Активно зелье силы для позиций "
                f"{sorted(buffed_strength)} перед началом боя"
            )
        if buffed_energy:
            self._log(
                "Активен эликсир энергии для позиций "
                f"{sorted(buffed_energy)} перед началом боя"
            )
        if buffed_haste:
            self._log(
                "Активен эликсир быстроты для позиций "
                f"{sorted(buffed_haste)} перед началом боя"
            )
        if buffed_fire:
            self._log(
                "Активна защита от магии Огня для позиций "
                f"{sorted(buffed_fire)} перед началом боя"
            )
        if buffed_earth:
            self._log(
                "Активна защита от магии Земли для позиций "
                f"{sorted(buffed_earth)} перед началом боя"
            )
        if buffed_water:
            self._log(
                "Активна защита от магии Воды для позиций "
                f"{sorted(buffed_water)} перед началом боя"
            )
        if buffed_air:
            self._log(
                "Активна защита от магии Воздуха для позиций "
                f"{sorted(buffed_air)} перед началом боя"
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
    def _refresh_campaign_artifact_effects(self, *, log: bool = False) -> None:
        if self.blue_team_state is None:
            return
        self._apply_equipped_artifact_effects(self.blue_team_state, log=log)
    def _apply_equipped_artifact_effects(
        self,
        blue_team: List[Dict],
        *,
        log: bool = True,
    ) -> None:
        self._clear_equipped_artifact_effects(blue_team)
        if not blue_team:
            return

        self._sync_equipped_artifact_items()
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
    def _has_any_heal_bottle_left(self) -> bool:
        """True, если доступна хотя бы одна банка лечения любого типа."""
        return (
            self.heal_bottles_used < self._max_heal_bottles_available()
            or self.healing_bottles_used < self._max_healing_bottles_available()
            or self._hero_item_available(self.HEALING_OINTMENT_ITEM_NAME)
        )
    def _select_heal_bottle_for_position(self, position: int) -> Tuple[float, Optional[str]]:
        """
        Выбирает, какой тип лечения применить к юниту:
        - предпочитает наименьшее достаточное лечение среди +50 / +100 / +200;
        - если недостаёт более 200 HP и есть Целебная мазь, использует её;
        - если +50/+100 закончились, но есть Целебная мазь, использует её как фолбэк;
        - если лечить нельзя: возвращает (0.0, None).
        """
        missing_hp = self._missing_hp_at_position(position)
        if missing_hp <= 0.0:
            return 0.0, None

        has_large = self.heal_bottles_used < self._max_heal_bottles_available()
        has_small = self.healing_bottles_used < self._max_healing_bottles_available()
        has_ointment = self._hero_item_available(self.HEALING_OINTMENT_ITEM_NAME)

        if has_small and missing_hp <= self.HEALING_BOTTLE_AMOUNT:
            return self.HEALING_BOTTLE_AMOUNT, "small"
        if has_large and missing_hp <= self.HEAL_BOTTLE_AMOUNT:
            return self.HEAL_BOTTLE_AMOUNT, "large"
        if has_ointment and missing_hp > self.HEALING_OINTMENT_AMOUNT:
            return self.HEALING_OINTMENT_AMOUNT, "ointment"
        if has_ointment and not (has_small or has_large):
            return self.HEALING_OINTMENT_AMOUNT, "ointment"
        if has_ointment and not has_large and missing_hp > self.HEALING_BOTTLE_AMOUNT:
            return self.HEALING_OINTMENT_AMOUNT, "ointment"
        if has_large:
            return self.HEAL_BOTTLE_AMOUNT, "large"
        if has_small:
            return self.HEALING_BOTTLE_AMOUNT, "small"
        if has_ointment:
            return self.HEALING_OINTMENT_AMOUNT, "ointment"
        return 0.0, None
    def _compute_enemy_defeat_reward(self, enemy_id: Optional[int]) -> float:
        """Фиксированная награда за победу над каждым вражеским отрядом."""
        reward = float(self.reward_defeat_enemy)
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            normalized_enemy_id = -1
        if normalized_enemy_id in self.RUIN_REWARD_BY_ENEMY_ID:
            reward += float(self.reward_ruin_clear_bonus)
        return reward
