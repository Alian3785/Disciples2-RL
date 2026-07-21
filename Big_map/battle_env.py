# ============================================================
# КАСТОМНАЯ СРЕДА Gymnasium «RED vs BLUE»
# Version: action space = 39 (12 pos actions + defend + wait + run + 24 hero item actions)
# + big-юниты: Dead dragon (3-6), Asterot (7-10), Gargoyle (9-12)
# + Чекпоинты каждые 1,000,000 шагов
# + Тест с логами и улучшенной визуализацией (большие карточки)
# ---
# Изменение: RED (кроме Mage/Dead dragon) выбирают цель с минимальным HP.
# + Action Masking (sb3_contrib: MaskablePPO + ActionMasker)
# ============================================================

import gymnasium as gym

import os
import random
import numpy as np
import math
from copy import deepcopy
from operator import itemgetter
from typing import Dict, List, Optional, Tuple
from gymnasium import spaces
from data_dicts_compact_lines import DATA as UNIT_DATA, placeholder_unit
from grid import BLUE_TEAM_TEMPLATE

# Импорт действий юнитов (предполагается, что эти файлы лежат рядом)
from summon_actions import (
    handle_laclaan_action,
    handle_lyf_action,
    handle_occultmaster_action,
)


SAVE_UNITS_ARRAYS_FLAG = int(os.environ.get("SAVE_UNITS_ARRAYS_FLAG", "1"))

# --- словари для кодирования в наблюдении ---
TYPE_LIST = [
    "Archer",
    "Mage",
    "Wolf Lord",
    "Teurg",
    "Witch",
    "Warrior",
    "Centaur Savage",
    "Doppelganger",
    "Demon",
    "Death",
    "Wight",
    "Sentry",
    "Watcher",
    "Lord",
    "Bone Lord",
    "Dregazul",
    "Dead dragon",
    "Gumtic",
    "Ismir son",
    "Ghost",
    "Shadow",
    "Succub",
    "Betrezen",
    "Uter",
    "Uter Demon",
    "Tiamat",
    "Baroness",
    "Incub",
    "Abyss Devil",
    "Cliric",
    "Profit",
    "Sundancer",
    "Sylfid",
    "Travnitsa",
    "Deva roshi",
    "Patriach",
    "Novice",
    "Alchemist",
    "Dwarfdruid",
    "Arhidruid",
    "Hermit",
    "Vampire",
    "Highvampire",
    "Summoner",
    "Occultmaster",
    "Lyf",
    "Laclaan",
    "Spider",
    "Drulliaan",
    "Elfarcher",
    "Shamanka",
    "Aleman",
]
ATTACK_TYPES = [
    "Weapon",
    "Earth",
    "Fire",
    "Water",
    "Poison",
    "Death",
    "Mind",
    "Life",
    "Air",
]  # one-hot(9)

SOUL_CRYSTAL_ARTIFACT_ITEM_NAME = "Soul Crystal (Artifact)"
HORN_OF_INCUBUS_ARTIFACT_ITEM_NAME = "Horn of Incubus (Artifact)"
UNHOLY_DAGGER_ARTIFACT_ITEM_NAME = "Unholy Dagger (Artifact)"
THANATOS_BLADE_ARTIFACT_ITEM_NAME = "Thanatos Blade (Artifact)"
SKULL_OF_THANATOS_ARTIFACT_ITEM_NAME = "Skull of Thanatos (Artifact)"
HAGS_RING_ARTIFACT_ITEM_NAME = "Hag's Ring (Artifact)"

HERO_COMBAT_ARTIFACT_ITEMS = frozenset(
    {
        SOUL_CRYSTAL_ARTIFACT_ITEM_NAME,
        HORN_OF_INCUBUS_ARTIFACT_ITEM_NAME,
        UNHOLY_DAGGER_ARTIFACT_ITEM_NAME,
        THANATOS_BLADE_ARTIFACT_ITEM_NAME,
        SKULL_OF_THANATOS_ARTIFACT_ITEM_NAME,
        HAGS_RING_ARTIFACT_ITEM_NAME,
    }
)
UNHOLY_DAGGER_DRAIN_PERCENT = 25
THANATOS_BLADE_POISON_DAMAGE = 20
SKULL_OF_THANATOS_POISON_DAMAGE = 35
HERO_ARTIFACT_STATUS_CHANCES = {
    SOUL_CRYSTAL_ARTIFACT_ITEM_NAME: 80,
    HORN_OF_INCUBUS_ARTIFACT_ITEM_NAME: 80,
    THANATOS_BLADE_ARTIFACT_ITEM_NAME: 80,
    SKULL_OF_THANATOS_ARTIFACT_ITEM_NAME: 80,
    HAGS_RING_ARTIFACT_ITEM_NAME: 70,
}
HERO_ARTIFACT_POISON_DAMAGE = {
    THANATOS_BLADE_ARTIFACT_ITEM_NAME: THANATOS_BLADE_POISON_DAMAGE,
    SKULL_OF_THANATOS_ARTIFACT_ITEM_NAME: SKULL_OF_THANATOS_POISON_DAMAGE,
}

ZERO_DAMAGE_NO_HP_UNIT_TYPES = frozenset(
    {
        "Ghost",
        "Witch",
        "Shadow",
        "Incub",
        "Succub",
        "Summoner",
        "Occultmaster",
        "Laclaan",
        "Baroness",
        "Travnitsa",
        "Novice",
        "Alchemist",
        "Dwarfdruid",
        "Arhidruid",
    }
)

DOUBLE_STRIKE_TYPES = frozenset({"Demon", "Elfarcher"})

MELEE_TYPES = frozenset(
    {
        "Warrior",
        "Centaur Savage",
        "Demon",
        "Lord",
        "Bone Lord",
        "Dregazul",
        "Ismir son",
        "Betrezen",
        "Uter",
        "Abyss Devil",
        "Aleman",
        "Spider",
    }
)
SMART_MELEE_TARGET_TYPES = frozenset({"Betrezen", "Uter"})
LONG_PARALYSIS_TYPES = frozenset({"Betrezen", "Uter", "Abyss Devil"})

# Кого Двойнику нельзя копировать (стражи столиц и сюжетные боссы).
DOPPELGANGER_FORBIDDEN_COPY_NAMES = frozenset(
    {
        "Ашган",
        "Ашкаэль",
        "Видар",
        "Мизраэль",
        "Иллюмиэлль",
        "Драллиаан",
        "Лаклаан",
    }
)

AOE_TYPES = frozenset(
    {
        "Mage",
        "Dead dragon",
        "Wolf Lord",
        "Gumtic",
        "Drulliaan",
        "Uter Demon",
        "Tiamat",
        "Teurg",
        "Hermit",
        "Vampire",
        "Highvampire",
        "Shamanka",
    }
)
VAMPIRE_AOE_TYPES = frozenset({"Vampire", "Highvampire"})

POINT_HEAL_SUPPORT_TYPES = frozenset({"Cliric", "Deva roshi"})
PATRIACH_SUPPORT_TYPES = frozenset({"Patriach"})
POINT_BUFF_SUPPORT_TYPES = frozenset(
    {
        "Travnitsa",
        "Novice",
        "Alchemist",
        "Dwarfdruid",
        "Arhidruid",
    }
)
CLEANSING_SUPPORT_TYPES = frozenset({"Dwarfdruid", "Arhidruid"})
MASS_HEAL_TYPES = frozenset({"Profit", "Sundancer", "Sylfid"})
SUPPORT_TYPES = (
    POINT_HEAL_SUPPORT_TYPES
    | PATRIACH_SUPPORT_TYPES
    | POINT_BUFF_SUPPORT_TYPES
    | MASS_HEAL_TYPES
)

# Only these innate healers receive one healing-only turn after their team has
# defeated the opposition.  The name allowlist is intentional: a Doppelganger
# that copied a healer and units with healing items must not enter this phase.
POST_VICTORY_HEALER_NAMES = frozenset(
    {
        "Служка",
        "Жрец",
        "Священник",
        "Архангел",
        "Медиум",
        "Оракул",
        "Дева рощи",
        "Патриарх",
        "Клирик",
        "Аббатиса",
        "Прорицательница",
        "Нейтральный эльфийский оракул",
        "Солнечная танцовщица",
        "Сильфида",
    }
)
POST_VICTORY_HEALER_TYPES = (
    POINT_HEAL_SUPPORT_TYPES | PATRIACH_SUPPORT_TYPES | MASS_HEAL_TYPES
)

FEATURES_PER_UNIT = 8 + len(TYPE_LIST) + 5 * len(ATTACK_TYPES) + 14 + 3
OBSERVATION_SIZE = FEATURES_PER_UNIT * 12


def _one_hot(value: str, vocab: List[str]) -> List[float]:
    return [1.0 if value == v else 0.0 for v in vocab]


def _multi_hot(values: List[str], vocab: List[str]) -> List[float]:
    s = set(values or [])
    return [1.0 if v in s else 0.0 for v in vocab]


def _build_transforms_by_name(unit_data: List[Dict]) -> Dict[str, List[str]]:
    transforms_by_name: Dict[str, List[str]] = {}
    for entry in unit_data:
        if not isinstance(entry, dict):
            continue
        name_raw = entry.get("кто")
        if not name_raw:
            continue
        name = str(name_raw).strip()
        if not name:
            continue
        raw_transforms = entry.get("превращается в", [])
        if isinstance(raw_transforms, (list, tuple)):
            transforms = [str(v) for v in raw_transforms if v is not None]
        elif raw_transforms:
            transforms = [str(raw_transforms)]
        else:
            transforms = []
        transforms_by_name[name] = transforms
    return transforms_by_name


TRANSFORMS_BY_NAME = _build_transforms_by_name(UNIT_DATA)


_ATK_CANON = {
    "weapon": "Weapon",
    "earth": "Earth",
    "fire": "Fire",
    "water": "Water",
    "poison": "Poison",
    "death": "Death",
    "mind": "Mind",
    "life": "Life",
    "air": "Air",
    "": "",
}


def _canon_attack_type(value: object) -> str:
    raw = str(value or "")
    return _ATK_CANON.get(raw.lower(), raw)


def _canon_attack_list(values: object) -> List[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_canon_attack_type(value) for value in values]


def _unit_data_name(entry: Dict) -> str:
    return str(entry.get("кто", "") or "").strip()


def _legacy_name_aliases(name: str) -> List[str]:
    aliases = [name]
    for errors in ("strict", "replace", "ignore"):
        try:
            alias = name.encode("utf-8").decode("cp1251", errors=errors)
        except UnicodeError:
            continue
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def _entry_stand(entry: Dict) -> str:
    attack_type = str(entry.get("тип атаки1", "") or "").lower()
    if attack_type == "weapon" or bool(entry.get("размер", 0)):
        return "ahead"
    return "behind"


def _entry_armor(entry: Dict) -> int:
    try:
        return int(round(float(entry.get("броня", 0) or 0) * 100))
    except (TypeError, ValueError):
        return 0


def _entry_accuracy(entry: Dict, key: str) -> int:
    try:
        return int(round(float(entry.get(key, 0) or 0) * 100))
    except (TypeError, ValueError):
        return 0


def _build_source_by_name(unit_data: List[Dict]) -> Dict[str, str]:
    source_by_name: Dict[str, str] = {}
    for entry in unit_data:
        if not isinstance(entry, dict):
            continue
        name = _unit_data_name(entry)
        if not name:
            continue
        source = str(entry.get("происходитиз", "") or "").strip()
        for alias in _legacy_name_aliases(name):
            source_by_name[alias] = source
    return source_by_name


def _build_battle_template_by_name(unit_data: List[Dict]) -> Dict[str, Dict]:
    templates: Dict[str, Dict] = {}
    for entry in unit_data:
        if not isinstance(entry, dict):
            continue
        name = _unit_data_name(entry)
        if not name:
            continue
        template = {
            "form_name": name,
            "initiative_base": _to_int_or_default(entry.get("инит", 0), default=0),
            "stand": _entry_stand(entry),
            "unit_type": str(entry.get("тип", "") or ""),
            "Level": _to_int_or_default(entry.get("уровень", 0), default=0),
            "damage": _to_int_or_default(entry.get("урон", 0), default=0),
            "damage_secondary": _to_int_or_default(entry.get("урон2", 0), default=0),
            "max_health": _to_int_or_default(entry.get("здоровье", 0), default=0),
            "armor": _entry_armor(entry),
            "accuracy": _entry_accuracy(entry, "точн"),
            "accuracy_secondary": _entry_accuracy(entry, "точн2"),
            "immunity": _canon_attack_list(entry.get("иммунитет", [])),
            "resistance": _canon_attack_list(entry.get("защита", [])),
            "attack_type_primary": _canon_attack_type(entry.get("тип атаки1", "")),
            "attack_type_secondary": _canon_attack_type(entry.get("тип атаки2", "")),
            "big": bool(entry.get("размер", 0)),
        }
        for alias in _legacy_name_aliases(name):
            templates[alias] = template
    return templates


def _apply_transformations_to_unit(unit: Dict) -> None:
    name = str(unit.get("name", "") or "").strip()
    transforms = TRANSFORMS_BY_NAME.get(name)
    if transforms is None:
        unit.setdefault("превращается в", [])
    else:
        unit["превращается в"] = list(transforms)


def _apply_transformations_to_units(units: List[Dict]) -> None:
    for unit in units:
        _apply_transformations_to_unit(unit)


def _to_int_or_default(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


SOURCE_BY_NAME = _build_source_by_name(UNIT_DATA)
BATTLE_TEMPLATE_BY_NAME = _build_battle_template_by_name(UNIT_DATA)


def _build_levels_by_name(unit_data: List[Dict]) -> Dict[str, int]:
    levels: Dict[str, int] = {}
    for entry in unit_data:
        if not isinstance(entry, dict):
            continue
        # Prefer canonical Russian keys and keep legacy mojibake fallback.
        entry_name = entry.get("\u043a\u0442\u043e")
        if entry_name is None:
            entry_name = entry.get("РєС‚Рѕ")
        if entry_name is None:
            continue
        name = str(entry_name).strip()
        if not name:
            continue
        level_raw = entry.get("\u0443\u0440\u043e\u0432\u0435\u043d\u044c", entry.get("Level", 0))
        levels[name] = _to_int_or_default(level_raw, default=0)
    return levels


LEVEL_BY_NAME = _build_levels_by_name(UNIT_DATA)


def _resolve_unit_level(unit: Dict) -> int:
    if "Level" in unit:
        return _to_int_or_default(unit.get("Level", 0), default=0)
    unit_name = str(unit.get("name", "") or "").strip()
    if not unit_name:
        return 0
    return _to_int_or_default(LEVEL_BY_NAME.get(unit_name, 0), default=0)


def _build_next_level_exp_by_name(unit_data: List[Dict]) -> Dict[str, int]:
    next_level_by_name: Dict[str, int] = {}
    for entry in unit_data:
        if not isinstance(entry, dict):
            continue
        entry_name = entry.get("\u043a\u0442\u043e")
        if entry_name is None:
            entry_name = entry.get("Р С”РЎвЂљР С•")
        if entry_name is None:
            continue
        name = str(entry_name).strip()
        if not name:
            continue
        next_level_raw = entry.get(
            "\u0441\u043b\u0435\u0434\u0443\u0440\u043e\u0432\u0435\u043d\u044c",
            entry.get("next_level_exp", 0),
        )
        next_level_by_name[name] = _to_int_or_default(next_level_raw, default=0)
    return next_level_by_name


NEXT_LEVEL_EXP_BY_NAME = _build_next_level_exp_by_name(UNIT_DATA)
HERO_LEADERSHIP_LEVEL = 3
HERO_ENDURANCE_LEVEL = 4
HERO_STRENGTH_LEVEL = 5
HERO_SECOND_LEADERSHIP_LEVEL = 6
HERO_BANNER_BEARER_LEVEL = 7
HERO_MARCHING_LORE_LEVEL = 8
HERO_ARTIFACT_KNOWLEDGE_LEVEL = 9
HERO_SORCERY_LORE_LEVEL = 10
HERO_ENDURANCE_HEALTH_MULTIPLIER = 1.20
HERO_STRENGTH_DAMAGE_MULTIPLIER = 1.25


def _resolve_unit_next_level_exp(unit: Dict) -> int:
    if "next_level_exp" in unit:
        return _to_int_or_default(unit.get("next_level_exp", 0), default=0)
    if "\u0441\u043b\u0435\u0434\u0443\u0440\u043e\u0432\u0435\u043d\u044c" in unit:
        return _to_int_or_default(unit.get("\u0441\u043b\u0435\u0434\u0443\u0440\u043e\u0432\u0435\u043d\u044c", 0), default=0)
    unit_name = str(unit.get("name", "") or "").strip()
    if not unit_name:
        return 0
    return _to_int_or_default(NEXT_LEVEL_EXP_BY_NAME.get(unit_name, 0), default=0)


def _to_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


THIEF_UNIT_NAMES = frozenset(
    {
        "Вор империи",
        "Вор легионов",
        "Вор гномов",
        "Вор нежити",
        "Вор эльфов",
    }
)


def _is_thief_battle_unit(unit: Dict) -> bool:
    """Recognize faction thieves independently from hero/evolution metadata."""
    if not isinstance(unit, dict):
        return False
    if _to_bool_flag(unit.get("thief", unit.get("is_thief", False))):
        return True
    return str(unit.get("name", "") or "").strip() in THIEF_UNIT_NAMES


def _is_neutral_battle_unit(unit: Dict) -> bool:
    if not isinstance(unit, dict):
        return False
    if bool(unit.get("is_neutral_unit", False)):
        return True

    capital = unit.get("capital", unit.get("\u0441\u0442\u043e\u043b\u0438\u0446\u0430"))
    if capital is None:
        return False
    try:
        return int(round(float(capital))) == 0
    except (TypeError, ValueError):
        return str(capital).strip() == "0"


def _resolve_unit_hero_flag(unit: Dict) -> bool:
    if not isinstance(unit, dict):
        return False
    if "hero" in unit:
        return _to_bool_flag(unit.get("hero"))
    return _resolve_unit_next_level_exp(unit) > 0


def _is_travel_hero_unit(unit: Dict) -> bool:
    return _resolve_unit_hero_flag(unit)


def _is_combat_hero_unit(unit: Optional[Dict]) -> bool:
    if not isinstance(unit, dict):
        return False
    if str(unit.get("team", "") or "").lower() != "blue":
        return False
    return _is_travel_hero_unit(unit)


def _resolve_unit_needaunit(unit: Dict) -> int:
    return 1 if _to_int_or_default(unit.get("needaunit", 0), default=0) > 0 else 0


def _levelup_stat_with_ten_percent(value: object) -> int:
    """Increase a non-negative integer stat by 10%, rounding the gain upward."""
    current = max(0, _to_int_or_default(value, default=0))
    if current <= 0:
        return 0
    increase = max(1, int(math.ceil(float(current) * 0.10 - 1e-9)))
    return int(current + increase)


def _apply_ten_percent_levelup_stats(unit: Dict) -> None:
    """Apply the shared endless-level growth for dynamic units and heroes."""
    base_damage = unit.get("original_damage")
    if base_damage is None:
        base_damage = unit.get("damage", 0)
    new_damage = _levelup_stat_with_ten_percent(base_damage)
    unit["damage"] = new_damage
    unit["original_damage"] = new_damage

    current_health = unit.get("health", unit.get("hp", 0))
    max_health = unit.get("max_health", unit.get("maxhp", 0))
    unit["health"] = _levelup_stat_with_ten_percent(current_health)
    unit["max_health"] = _levelup_stat_with_ten_percent(max_health)
    if "hp" in unit:
        unit["hp"] = unit["health"]
    if "maxhp" in unit:
        unit["maxhp"] = unit["max_health"]

    current_accuracy = _to_int_or_default(unit.get("accuracy", 0), default=0)
    unit["accuracy"] = max(0, min(100, current_accuracy + 1))


def _apply_hero_level_milestone_bonuses(unit: Dict) -> None:
    """Apply leadership/endurance/strength bonuses tied to the new hero level."""
    unit["needaunit"] = _resolve_unit_needaunit(unit)
    if _is_travel_hero_unit(unit) and _resolve_unit_level(unit) in (
        HERO_LEADERSHIP_LEVEL,
        HERO_SECOND_LEADERSHIP_LEVEL,
    ):
        unit["needaunit"] = 1
    if _is_travel_hero_unit(unit) and _resolve_unit_level(unit) == HERO_ENDURANCE_LEVEL:
        current_health = float(unit.get("health", unit.get("hp", 0)) or 0)
        max_health = float(unit.get("max_health", unit.get("maxhp", 0)) or 0)
        unit["health"] = max(
            0,
            _to_int_or_default(
                current_health * HERO_ENDURANCE_HEALTH_MULTIPLIER,
                default=0,
            ),
        )
        unit["max_health"] = max(
            0,
            _to_int_or_default(
                max_health * HERO_ENDURANCE_HEALTH_MULTIPLIER,
                default=0,
            ),
        )
        if "hp" in unit:
            unit["hp"] = unit["health"]
        if "maxhp" in unit:
            unit["maxhp"] = unit["max_health"]
    if _is_travel_hero_unit(unit) and _resolve_unit_level(unit) == HERO_STRENGTH_LEVEL:
        base_damage = float(unit.get("original_damage", unit.get("damage", 0)) or 0)
        strength_damage = max(
            0,
            _to_int_or_default(base_damage * HERO_STRENGTH_DAMAGE_MULTIPLIER, default=0),
        )
        unit["damage"] = strength_damage
        unit["original_damage"] = strength_damage


def _apply_hero_levelup_bonuses(unit: Dict) -> None:
    _apply_ten_percent_levelup_stats(unit)
    current_exp_kill = float(unit.get("exp_kill", 0) or 0)
    unit["exp_kill"] = max(0, _to_int_or_default(current_exp_kill * 1.1, default=0))
    _apply_hero_level_milestone_bonuses(unit)


def _unit_has_available_evolution(unit: Dict) -> bool:
    turns_into = unit.get(
        "turns_into",
        unit.get("\u043f\u0440\u0435\u0432\u0440\u0430\u0449\u0430\u0435\u0442\u0441\u044f \u0432", ()),
    )
    if isinstance(turns_into, str):
        return bool(turns_into.strip())
    if isinstance(turns_into, (list, tuple, set, frozenset)):
        return any(str(target or "").strip() for target in turns_into)
    return bool(turns_into)


def _uses_dynamic_unit_levelup(unit: Dict) -> bool:
    """Return whether a unit gains stats instead of evolving into another form."""
    # Faction thieves do not gain experience or levels in the original game.
    # Keep this guard even though battle XP distribution normally excludes them.
    if _is_thief_battle_unit(unit):
        return False
    if _is_neutral_battle_unit(unit):
        return True
    if _to_bool_flag(unit.get("boss", unit.get("is_boss", False))):
        return True
    if _resolve_unit_hero_flag(unit):
        return False
    return not _unit_has_available_evolution(unit)


def _apply_dynamic_unit_levelup(unit: Dict, exp_required: int) -> None:
    """Level a dynamic unit, optionally applying a scenario-specific XP curve."""
    unit["Level"] = _to_int_or_default(unit.get("Level", 0), default=0) + 1
    exp_increment = max(
        0,
        _to_int_or_default(
            unit.get("dynamic_level_exp_increment", 0),
            default=0,
        ),
    )
    unit["exp_required"] = max(0, int(exp_required) + int(exp_increment))
    unit["exp_current"] = 0
    _apply_ten_percent_levelup_stats(unit)
    if _resolve_unit_hero_flag(unit):
        _apply_hero_level_milestone_bonuses(unit)


DEFAULT_RED_ROSTER: Tuple[Tuple[str, int], ...] = (
    ("Скелет рыцарь", 1),
    ("Рыцарь смерти", 2),
    ("Змей ужаса", 3),
    ("Архилич", 4),
    ("Смерть", 5),
    ("пусто", 6),
)


def _build_default_battle_unit(name: str, team: str, position: int) -> Dict:
    if name == "пусто":
        return placeholder_unit(team, position)

    template = BATTLE_TEMPLATE_BY_NAME.get(name)
    if not isinstance(template, dict):
        raise KeyError(f"Default battle unit template not found: {name!r}")

    unit = deepcopy(template)
    unit.update(
        {
            "name": name,
            "initiative": int(unit.get("initiative_base", 0) or 0),
            "team": team,
            "position": position,
            "health": int(unit.get("max_health", 0) or 0),
            "paralyzed": 0,
            "long_paralyzed": 0,
            "running_away": 0,
            "transformed": 0,
            "basestats": [],
        }
    )
    return unit


UNITS_RED = [
    _build_default_battle_unit(name, "red", position)
    for name, position in DEFAULT_RED_ROSTER
]

UNITS_BLUE = deepcopy(BLUE_TEAM_TEMPLATE)

_apply_transformations_to_units(UNITS_RED)
_apply_transformations_to_units(UNITS_BLUE)


def _apply_team_traits(
    units: List[Dict], enemy_units: Optional[List[Dict]] = None
) -> None:
    if any(unit.get("unit_type") == "Sundancer" for unit in units):
        for unit in units:
            unit["Firedefence"] = 0

    if any(unit.get("unit_type") == "Sylfid" for unit in units):
        for unit in units:
            unit["Airdefence"] = 0

    if any(
        unit.get("unit_type") in ("Travnitsa", "Novice", "Alchemist") for unit in units
    ):
        for unit in units:
            unit["powerup"] = 0

    if any(
        unit.get("unit_type") in ("Dwarfdruid", "Arhidruid", "Travnitsa", "Novice")
        for unit in units
    ):
        for unit in units:
            unit["original_damage"] = unit.get("damage", 0)

    if any(unit.get("unit_type") == "Deva roshi" for unit in units):
        for unit in units:
            unit["Firedefence"] = 0
            unit["Airdefence"] = 0
            unit["Waterdefence"] = 0
            unit["Earthdefence"] = 0

    if any(unit.get("unit_type") == "Alchemist" for unit in units):
        for unit in units:
            unit["bonusturn"] = 0

    if enemy_units is not None and any(
        unit.get("unit_type") == "Tiamat" for unit in units
    ):
        for enemy in enemy_units:
            enemy["teamated"] = 0

    if enemy_units is not None and any(
        unit.get("unit_type") == "Hermit" for unit in units
    ):
        for enemy in enemy_units:
            enemy["hermited"] = 0  # флаг: уже замедлён «hermit»


_apply_team_traits(UNITS_RED, UNITS_BLUE)
_apply_team_traits(UNITS_BLUE, UNITS_RED)

for unit in UNITS_RED + UNITS_BLUE:
    unit["Level"] = _resolve_unit_level(unit)
    unit["next_level_exp"] = _resolve_unit_next_level_exp(unit)
    unit["hero"] = _resolve_unit_hero_flag(unit)
    unit["needaunit"] = _resolve_unit_needaunit(unit)
    unit.setdefault("defense", 0)
    unit.setdefault("waited", 0)


# Диапазоны позиций
# Диапазоны позиций
RED_POSITIONS: List[int] = list(range(1, 7))
BLUE_POSITIONS: List[int] = list(range(7, 13))

# Доступные цели для действия агента — теперь 12 позиций (pos1..pos12)
# Логика боя и маски остаётся прежней: невалидные позиции будут заблокированы маской.
TARGET_POSITIONS: List[int] = (
    RED_POSITIONS + BLUE_POSITIONS
)  # 12 действий на выбор позиции
HERO_ITEM_TARGET_SLOTS: List[int] = list(range(1, 7))
DEFEND_ACTION_INDEX: int = len(TARGET_POSITIONS)
WAIT_ACTION_INDEX: int = len(TARGET_POSITIONS) + 1
RUN_AWAY_ACTION_INDEX: int = len(TARGET_POSITIONS) + 2
FIRST_HERO_ITEM_ACTION_START: int = len(TARGET_POSITIONS) + 3
SECOND_HERO_ITEM_ACTION_START: int = FIRST_HERO_ITEM_ACTION_START + len(
    HERO_ITEM_TARGET_SLOTS
)
FIRST_HERO_ITEM_ENEMY_ACTION_START: int = SECOND_HERO_ITEM_ACTION_START + len(
    HERO_ITEM_TARGET_SLOTS
)
SECOND_HERO_ITEM_ENEMY_ACTION_START: int = FIRST_HERO_ITEM_ENEMY_ACTION_START + len(
    HERO_ITEM_TARGET_SLOTS
)
DEFEND_ARMOR_BONUS: int = 50
WAIT_INIT_DIVISOR: int = 10
TOTAL_AGENT_ACTIONS: int = SECOND_HERO_ITEM_ENEMY_ACTION_START + len(
    HERO_ITEM_TARGET_SLOTS
)  # 39

RED_FRONT_POSITIONS: List[int] = [1, 2, 3]
RED_BACK_POSITIONS: List[int] = [4, 5, 6]
BLUE_FRONT_POSITIONS: List[int] = [7, 8, 9]
BLUE_BACK_POSITIONS: List[int] = [10, 11, 12]

# Константы для нормирования наблюдений
MAX_HP = max([u["health"] for u in (UNITS_RED + UNITS_BLUE)])  # 1020
MAX_HP = max(MAX_HP, 525)  # учитываем призыв Occultmaster (драконов до 525 HP)
MAX_INIT = max([u["initiative_base"] for u in (UNITS_RED + UNITS_BLUE)])
INIT_RANDOM_BONUS = 9  # initiative jitter applied at the start of each round
MAX_INIT_DYNAMIC = MAX_INIT + INIT_RANDOM_BONUS
# В базовых массивах стартовых армий урон равен 0, поэтому дополнительно учитываем
# потенциальный максимум с учётом призываемых существ (драконы Occultmaster/Lacflaan).
MAX_DMG = max([u["damage"] for u in (UNITS_RED + UNITS_BLUE)])
MAX_DMG = max(MAX_DMG, 125)  # чёрный дракон Laclaan наносит 125 урона
# Include arena summons (e.g., Lyf's spider) that are not present in the base rosters.
MAX_DMG2 = max(
    max([u["damage_secondary"] for u in (UNITS_RED + UNITS_BLUE)]),
    35,  # Гигантский чёрный паук
)
# Травница может баффать союзника на +75% урона, поэтому расширяем допустимый максимум.
TRAVNITSA_MULTIPLIER = 2
MAX_DMG_BUFFED = int(np.ceil(MAX_DMG * TRAVNITSA_MULTIPLIER))

POISON_TURNS = 6  # max duration; actual value rolled per application
BURN_DAMAGE = 10  # базовый фолбэк; Владыка задаёт per-tick из своего урон2
BURN_TURNS = 6
URAN_DAMAGE = 10  # базовый фолбэк; Сын Измира задаёт per-tick из своего урон2
URAN_TURNS = 6
DEFAULT_TRANSFORM_RECOVERY_CHANCE = 0.3
WIGHT_LEVEL_DOWN_RECOVERY_CHANCE = 0.01

# Observation normalization profile.
INIT_BASE_NORM_MAX = 80.0
INIT_CUR_NORM_MAX = 89.0
DMG_NORM_MAX = 250.0
DMG2_NORM_MAX = 35.0
# Armor can temporarily exceed 100 because of settlement/capital defense bonuses
# on battle start (+30/+50) and the in-battle DEFEND action (+50).
ARMOR_NORM_MAX = 165.0
ACC_NORM_MAX = 100.0
BONUS_NORM_MAX = 10.0
EXP_KILL_NORM_MAX = 120.0
EXP_REQ_NORM_MAX = 700.0
POS_NORM_MAX = 11.0
HP_FALLBACK_MAX = 800.0


def _clip01(x: float) -> float:
    x = float(x)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x != x:
        return 0.0
    return x


def _safe_div(x: float, d: float) -> float:
    d = float(d)
    if d <= 1e-12:
        return 0.0
    return float(x) / d


def _norm_positive(value: object, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 1e-12:
        return 0.0
    value = float(value)
    if value <= 0.0:
        return 0.0
    return _clip01(value / denominator)


def _to01_bool(v) -> float:
    try:
        return 1.0 if float(v) > 0.0 else 0.0
    except (TypeError, ValueError):
        return 1.0 if bool(v) else 0.0


# ------------------------ Кастомная среда ------------------------
class BattleEnv(gym.Env):
    """
    Observation (720): 12 слотов ? 60 признаков (добавлены типы Betrezen, Uter и Uter Demon).
    Action space: Discrete(39): 12 position actions + defend + wait + run
    + hero-only item slot 1/2 actions against blue and red positions 1..6.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_step: float = 0.0,
        penalty_invalid_target: float = -0.10,
        penalty_unreachable_warrior: float = -0.20,
        log_enabled: bool = False,
    ):
        super().__init__()
        self.reward_win = float(reward_win)
        self.reward_loss = float(reward_loss)
        self.reward_step = float(reward_step)

        self.penalty_invalid_target = float(penalty_invalid_target)
        self.penalty_unreachable_warrior = float(penalty_unreachable_warrior)

        self.log_enabled = bool(log_enabled)
        self._pretty_events: List[str] = []
        self.last_battle_exp: float = 0.0
        self.last_levelups: List[str] = []
        self.exp_multiplier: float = 1.0

        # Action space = 12 targets + defend/wait/run + 24 hero item actions.
        self.action_space = spaces.Discrete(TOTAL_AGENT_ACTIONS)  # 39

        # Observation space contract: each component is expected in [0, 1].
        low_unit = np.zeros(FEATURES_PER_UNIT, dtype=np.float32)
        high_unit = np.ones(FEATURES_PER_UNIT, dtype=np.float32)
        low = np.tile(low_unit, 12)
        high = np.tile(high_unit, 12)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Состояние боя
        self.rng = random.Random()  # без seed > случайность между эпизодами
        self.combined: List[Dict] = []
        self.round_no: int = 1
        self.winner: Optional[str] = None
        self._post_victory_team: Optional[str] = None
        self._post_victory_healer_positions: List[int] = []
        self.current_blue_attacker_pos: Optional[int] = None
        self.blue_attacks_left: int = 0
        self._lord_applied_burn: Dict[int, bool] = {}
        self._spider_applied_poison: Dict[int, bool] = {}
        self._dregazul_applied_poison: Dict[int, bool] = {}
        self._ismir_applied_uran: Dict[int, bool] = {}
        self.escaped_units: List[Dict] = []
        # Опыт трупов, затёртых призывом в их клетку, — учитывается при подсчёте опыта за бой.
        self.overwritten_exp_kill: Dict[str, float] = {"red": 0.0, "blue": 0.0}
        # XP is accumulated at each death because escaped and revived units are
        # entitled only to kills that happened while they were on the field.
        self._battle_exp_event_count: int = 0
        self._battle_defeated_exp: Dict[str, float] = {"red": 0.0, "blue": 0.0}
        self.step_count: int = 0
        self.equipped_hero_items: List[Optional[str]] = [None, None]
        self.equipped_hero_item_uses_left: List[Optional[int]] = [None, None]
        self.hero_item_slots_used_this_battle: List[bool] = [False, False]
        self.hero_item_effects: Dict[str, Dict[str, object]] = {}

    # ------------------ [MASKING] Маска допустимых действий ------------------
    def _hero_item_name_for_slot(self, item_no: int) -> str:
        slot_index = max(0, int(item_no) - 1)
        if 0 <= slot_index < len(self.equipped_hero_items):
            return str(self.equipped_hero_items[slot_index] or "")
        return ""

    def _hero_item_effect(self, item_name: str) -> Dict[str, object]:
        if not item_name:
            return {}
        effect = self.hero_item_effects.get(str(item_name), {})
        return dict(effect) if isinstance(effect, dict) else {}

    @staticmethod
    def _hero_item_is_talisman(effect: Dict[str, object]) -> bool:
        return str(effect.get("item_category", "") or "").strip().lower() == "talisman"

    def _hero_item_slot_uses_left(self, slot_index: int, effect: Dict[str, object]) -> Optional[int]:
        if not self._hero_item_is_talisman(effect):
            return 1
        default_uses = int(effect.get("uses_per_item", 5) or 5)
        if len(getattr(self, "equipped_hero_item_uses_left", []) or []) < len(
            self.equipped_hero_items
        ):
            self.equipped_hero_item_uses_left = [None] * len(self.equipped_hero_items)

        try:
            uses_left = int(self.equipped_hero_item_uses_left[int(slot_index)])
        except (TypeError, ValueError, IndexError):
            uses_left = default_uses
        return max(0, min(default_uses, int(uses_left)))

    def _hero_item_slot_available(self, slot_index: int, effect: Dict[str, object]) -> bool:
        if not self._hero_item_is_talisman(effect):
            return True
        if 0 <= int(slot_index) < len(getattr(self, "hero_item_slots_used_this_battle", [])):
            if bool(self.hero_item_slots_used_this_battle[int(slot_index)]):
                return False
        uses_left = self._hero_item_slot_uses_left(int(slot_index), effect)
        return bool(uses_left and int(uses_left) > 0)

    @staticmethod
    def _normalize_hero_item_target_team(value: object) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in ("ally", "allies", "own", "self", "blue"):
            return "blue"
        if normalized in ("enemy", "enemies", "red"):
            return "red"
        return normalized

    def _hero_item_effect_target_team(
        self,
        *,
        effect: Dict[str, object],
        effect_kind: str,
    ) -> str:
        explicit_team = self._normalize_hero_item_target_team(effect.get("target_team"))
        if explicit_team:
            return explicit_team
        if effect_kind in ("heal", "revive", "summon", "damage_buff", "extra_turn"):
            return "blue"
        if effect_kind in (
            "damage",
            "drain",
            "dot",
            "paralysis",
            "transform",
            "lycanthropy",
            "debuff_damage",
        ):
            return "red"
        return ""

    @staticmethod
    def _hero_item_effect_scope(effect: Dict[str, object]) -> str:
        normalized = str(effect.get("scope", "single") or "single").strip().lower()
        return "party" if normalized in ("party", "group", "all") else "single"

    @staticmethod
    def _hero_item_effect_amount(effect: Dict[str, object]) -> float:
        try:
            return float(effect.get("amount", effect.get("damage", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _hero_item_position_team(pos: int) -> str:
        if int(pos) in BLUE_POSITIONS:
            return "blue"
        if int(pos) in RED_POSITIONS:
            return "red"
        return ""

    @staticmethod
    def _is_empty_battle_unit(unit: Optional[Dict]) -> bool:
        if not isinstance(unit, dict):
            return True
        name = str(unit.get("name", "") or "").strip().lower()
        if name in ("пусто", "РїСѓСЃС‚Рѕ".lower()):
            return True
        health = float(unit.get("health", 0) or unit.get("hp", 0) or 0)
        max_health = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0)
        return health <= 0 and max_health <= 0

    def _hero_item_target_state(self, effect: Dict[str, object], effect_kind: str) -> str:
        explicit_state = str(effect.get("target_state", "") or "").strip().lower()
        if explicit_state:
            return explicit_state
        if effect_kind == "summon":
            return "empty_or_dead"
        if effect_kind == "revive":
            return "dead"
        return "alive"

    def _hero_item_target_matches_state(
        self,
        *,
        effect: Dict[str, object],
        effect_kind: str,
        target_team: str,
        target_pos: int,
        target_unit: Optional[Dict],
    ) -> bool:
        if self._hero_item_position_team(target_pos) != target_team:
            return False
        if isinstance(target_unit, dict) and target_unit.get("team") != target_team:
            return False

        target_state = self._hero_item_target_state(effect, effect_kind)
        is_empty = self._is_empty_battle_unit(target_unit)
        health = (
            float(target_unit.get("health", 0) or target_unit.get("hp", 0) or 0)
            if isinstance(target_unit, dict)
            else 0.0
        )
        max_health = (
            float(target_unit.get("max_health", 0) or target_unit.get("maxhp", 0) or 0)
            if isinstance(target_unit, dict)
            else 0.0
        )
        alive = bool(isinstance(target_unit, dict) and not is_empty and health > 0)

        if target_state == "alive":
            return alive
        if target_state == "dead":
            return bool(
                isinstance(target_unit, dict)
                and not is_empty
                and max_health > 0
                and health <= 0
            )
        if target_state == "empty_or_dead":
            if target_team != "blue":
                return False
            if target_pos in BLUE_BACK_POSITIONS and self._is_position_behind_big(target_pos):
                return False
            return bool(is_empty or (max_health > 0 and health <= 0))
        return False

    @staticmethod
    def _hero_item_dot_fields(dot_type: object) -> Tuple[str, str]:
        normalized = str(dot_type or "").strip().lower()
        if normalized == "burn":
            return "burn_turns_left", "burn_damage_per_tick"
        if normalized in ("water", "frost", "freezing", "uran"):
            return "uran_turns_left", "uran_damage_per_tick"
        return "poison_turns_left", "poison_damage_per_tick"

    def _hero_item_find_unit_template(self, unit_name: str) -> Dict[str, object]:
        normalized_name = str(unit_name or "").strip()
        if not normalized_name:
            return {}
        template = BATTLE_TEMPLATE_BY_NAME.get(normalized_name, {})
        return dict(template) if isinstance(template, dict) else {}

    def _hero_item_party_targets(
        self,
        *,
        target_team: str,
        target_pos: int,
        target_unit: Optional[Dict],
    ) -> Tuple[Dict, ...]:
        if self._hero_item_position_team(target_pos) != target_team:
            return ()
        if not isinstance(target_unit, dict) or target_unit.get("team") != target_team:
            return ()
        if self._is_empty_battle_unit(target_unit) or not self._alive(target_unit):
            return ()
        return tuple(
            unit
            for unit in self.combined
            if unit.get("team") == target_team
            and not self._is_empty_battle_unit(unit)
            and self._alive(unit)
        )

    def _hero_item_effect_targets(
        self,
        *,
        effect: Dict[str, object],
        effect_kind: str,
        target_team: str,
        target_pos: int,
        target_unit: Optional[Dict],
    ) -> Tuple[Dict, ...]:
        if not self._hero_item_target_matches_state(
            effect=effect,
            effect_kind=effect_kind,
            target_team=target_team,
            target_pos=target_pos,
            target_unit=target_unit,
        ):
            return ()
        if self._hero_item_effect_scope(effect) == "party":
            return self._hero_item_party_targets(
                target_team=target_team,
                target_pos=target_pos,
                target_unit=target_unit,
            )
        return (target_unit,) if isinstance(target_unit, dict) else ()

    def _hero_item_attacker_proxy(
        self,
        source_unit: Optional[Dict],
        effect: Dict[str, object],
        *,
        primary: bool = False,
        unit_type: Optional[str] = None,
    ) -> Dict:
        proxy = dict(source_unit) if isinstance(source_unit, dict) else {}
        proxy.setdefault("team", "blue")
        proxy.setdefault("name", "Hero item")
        proxy.setdefault("position", 0)
        damage_type = str(effect.get("damage_type", "") or "").strip()
        if damage_type:
            if primary:
                proxy["attack_type_primary"] = damage_type
            proxy["attack_type_secondary"] = damage_type
        if unit_type:
            proxy["unit_type"] = str(unit_type)
        return proxy

    def _hero_item_type_effect_blocked(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
        effect: Dict[str, object],
        primary_damage: bool = False,
        status: bool = False,
    ) -> bool:
        damage_type = str(effect.get("damage_type", "") or "").strip()
        if not damage_type:
            return False

        proxy = self._hero_item_attacker_proxy(
            source_unit,
            effect,
            primary=primary_damage,
        )
        if primary_damage and self._is_immune_damage(proxy, target_unit):
            self._log(
                f"Hero item '{damage_type}' damage is blocked by immunity on "
                f"{target_unit.get('team', '').upper()} {target_unit.get('name')}#{target_unit.get('position')}."
            )
            return True
        if status and self._is_immune_status(proxy, target_unit):
            self._log(
                f"Hero item '{damage_type}' status is blocked by immunity on "
                f"{target_unit.get('team', '').upper()} {target_unit.get('name')}#{target_unit.get('position')}."
            )
            return True
        if self._resilience_blocks(proxy, target_unit, custom_tag=damage_type):
            return True
        return False

    def _apply_hero_item_heal(self, target_unit: Dict, amount: float) -> float:
        current_health = float(target_unit.get("health", 0) or 0)
        max_health = float(
            target_unit.get("max_health", 0) or target_unit.get("health", 0) or 0
        )
        if max_health <= 0 or current_health <= 0 or current_health >= max_health:
            return 0.0
        new_health = min(max_health, current_health + float(amount or 0.0))
        healed = max(0.0, new_health - current_health)
        target_unit["health"] = int(round(new_health))
        if "hp" in target_unit:
            target_unit["hp"] = int(round(new_health))
        return healed

    def _apply_hero_item_revive(self, target_unit: Dict, amount: float) -> float:
        current_health = float(target_unit.get("health", 0) or 0)
        max_health = float(
            target_unit.get("max_health", 0) or target_unit.get("health", 0) or 0
        )
        if max_health <= 0 or current_health > 0:
            return 0.0
        revive_hp = min(max_health, max(1.0, float(amount or 0.0)))
        target_unit["health"] = int(round(revive_hp))
        if "hp" in target_unit:
            target_unit["hp"] = int(round(revive_hp))
        target_unit["paralyzed"] = 0
        target_unit["long_paralyzed"] = 0
        target_unit["running_away"] = 0
        return revive_hp

    def _apply_hero_item_damage(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        effect_amount = self._hero_item_effect_amount(effect)
        current_health = float(target_unit.get("health", 0) or 0)
        if effect_amount <= 0 or current_health <= 0:
            return 0.0
        if self._hero_item_type_effect_blocked(
            source_unit=source_unit,
            target_unit=target_unit,
            effect=effect,
            primary_damage=True,
        ):
            return 0.0
        damage = min(current_health, effect_amount)
        new_health = max(0.0, current_health - damage)
        target_unit["health"] = int(round(new_health))
        if "hp" in target_unit:
            target_unit["hp"] = int(round(new_health))
        if new_health <= 0:
            self._record_unit_defeat_for_exp(target_unit)
            target_unit["initiative"] = 0
            self._kill_linked_summons(target_unit)
        return damage

    def _apply_hero_item_drain(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        damage = self._apply_hero_item_damage(
            source_unit=source_unit,
            target_unit=target_unit,
            effect=effect,
        )
        if damage > 0 and isinstance(source_unit, dict):
            self._apply_vampiric_heal(
                source_unit,
                int(round(damage)),
                share_leftover=bool(effect.get("share_leftover", False)),
            )
        return damage

    def _apply_hero_item_dot(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        if not self._alive(target_unit):
            return 0.0
        amount = self._hero_item_effect_amount(effect)
        if amount <= 0:
            return 0.0
        if self._hero_item_type_effect_blocked(
            source_unit=source_unit,
            target_unit=target_unit,
            effect=effect,
            status=True,
        ):
            return 0.0

        turns_key, damage_key = self._hero_item_dot_fields(effect.get("dot_type"))
        max_turns = POISON_TURNS
        if turns_key == "burn_turns_left":
            max_turns = BURN_TURNS
        elif turns_key == "uran_turns_left":
            max_turns = URAN_TURNS
        existing_turns = int(target_unit.get(turns_key, 0) or 0)
        existing_damage = int(target_unit.get(damage_key, 0) or 0)
        if existing_turns > 0 and existing_damage >= int(round(amount)):
            return 0.0

        target_unit[turns_key] = self.rng.randint(1, max_turns)
        target_unit[damage_key] = int(round(amount))
        return float(amount)

    def _apply_hero_item_damage_buff(
        self,
        *,
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        try:
            multiplier = float(effect.get("damage_multiplier", 0.0) or 0.0)
        except (TypeError, ValueError):
            multiplier = 0.0
        if multiplier <= 0.0 or not self._alive(target_unit):
            return 0.0
        base_damage = int(target_unit.get("original_damage", 0) or 0)
        if base_damage <= 0:
            base_damage = int(target_unit.get("damage", 0) or 0)
        if base_damage <= 0:
            return 0.0
        target_unit["original_damage"] = base_damage
        new_damage = max(0, int(round(base_damage * multiplier)))
        if new_damage == int(target_unit.get("damage", 0) or 0):
            return 0.0
        target_unit["damage"] = new_damage
        target_unit["powerup"] = 1
        return float(new_damage - base_damage)

    def _apply_hero_item_extra_turn(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
    ) -> float:
        if not self._alive(target_unit):
            return 0.0
        base_ini = int(target_unit.get("initiative_base", 0) or 0)
        target_unit["initiative"] = base_ini
        target_unit.setdefault("bonusturn", 0)
        target_unit["bonusturn"] = int(target_unit.get("bonusturn", 0) or 0) + 1
        return 1.0

    def _apply_hero_item_paralysis(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        proxy = self._hero_item_attacker_proxy(source_unit, effect, primary=True)
        return 1.0 if self._apply_paralysis_effect(proxy, target_unit) else 0.0

    def _apply_hero_item_fear(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        proxy = self._hero_item_attacker_proxy(source_unit, effect, primary=True)
        return 1.0 if self._apply_fear_effect(proxy, target_unit) else 0.0

    def _apply_hero_item_transform(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        proxy = self._hero_item_attacker_proxy(source_unit, effect)
        if self._hero_item_type_effect_blocked(
            source_unit=proxy,
            target_unit=target_unit,
            effect=effect,
            status=True,
        ):
            return 0.0
        if target_unit.get("transformed", 0) == 1:
            return 0.0
        self._apply_witch_effect(proxy, target_unit)
        return 1.0

    def _apply_hero_item_lycanthropy(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        if self._hero_item_type_effect_blocked(
            source_unit=source_unit,
            target_unit=target_unit,
            effect=effect,
            status=True,
        ):
            return 0.0
        unit_name = str(effect.get("transform_unit_name", "") or "").strip()
        template = self._hero_item_find_unit_template(unit_name)
        if not template or target_unit.get("transformed", 0) == 1:
            return 0.0

        self._ensure_transform_snapshot(target_unit, include_health_fields=True)
        old_hp = target_unit.get("health", 0)
        old_max = target_unit.get("max_health", old_hp)
        new_max = int(template.get("max_health", 0) or 0)
        new_health = self._scale_health_to_new_max(old_hp, old_max, new_max)
        old_initiative = target_unit.get("initiative", 0)

        target_unit["name"] = template.get("form_name", unit_name)
        target_unit["accuracy"] = int(template.get("accuracy", 0) or 0)
        target_unit["accuracy_secondary"] = int(template.get("accuracy_secondary", 0) or 0)
        target_unit["damage"] = int(template.get("damage", 0) or 0)
        target_unit["damage_secondary"] = int(template.get("damage_secondary", 0) or 0)
        target_unit["original_damage"] = target_unit["damage"]
        target_unit["attack_type_primary"] = template.get("attack_type_primary", "Weapon")
        target_unit["attack_type_secondary"] = template.get("attack_type_secondary", "")
        target_unit["initiative_base"] = int(template.get("initiative_base", 0) or 0)
        target_unit["initiative"] = (
            old_initiative if int(old_initiative or 0) == 0 else target_unit["initiative_base"]
        )
        target_unit["unit_type"] = template.get("unit_type", target_unit.get("unit_type", ""))
        target_unit["stand"] = template.get("stand", target_unit.get("stand", "ahead"))
        target_unit["armor"] = int(template.get("armor", 0) or 0)
        target_unit["immunity"] = list(template.get("immunity", []))
        target_unit["resistance"] = list(template.get("resistance", []))
        target_unit["resilience_used_types"] = []
        target_unit["big"] = bool(template.get("big", False))
        target_unit["Level"] = int(template.get("Level", target_unit.get("Level", 0)) or 0)
        target_unit["max_health"] = new_max
        target_unit["health"] = new_health
        if "maxhp" in target_unit:
            target_unit["maxhp"] = new_max
        if "hp" in target_unit:
            target_unit["hp"] = new_health
        target_unit["transformed"] = 1
        target_unit["transform_effect"] = "lycanthropy"
        return 1.0

    def _apply_hero_item_damage_debuff(
        self,
        *,
        source_unit: Optional[Dict],
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        if self._hero_item_type_effect_blocked(
            source_unit=source_unit,
            target_unit=target_unit,
            effect=effect,
            status=True,
        ):
            return 0.0
        if target_unit.get("teamated", 0) == 1:
            return 0.0
        try:
            multiplier = float(effect.get("damage_multiplier", 1.0) or 1.0)
        except (TypeError, ValueError):
            multiplier = 1.0
        current_damage = int(target_unit.get("damage", 0) or 0)
        new_damage = max(0, int(round(current_damage * max(0.0, multiplier))))
        target_unit["teamated"] = 1
        target_unit["damage"] = new_damage
        return float(max(0, current_damage - new_damage))

    def _build_hero_item_summon_unit(
        self,
        *,
        source_unit: Optional[Dict],
        target_team: str,
        target_pos: int,
        effect: Dict[str, object],
    ) -> Optional[Dict]:
        unit_name = str(effect.get("summon_unit_name", "") or "").strip()
        template = self._hero_item_find_unit_template(unit_name)
        if not template:
            return None

        max_health = int(template.get("max_health", 0) or 0)
        summon_unit = {
            "name": template.get("form_name", unit_name),
            "initiative": int(template.get("initiative_base", 0) or 0),
            "initiative_base": int(template.get("initiative_base", 0) or 0),
            "team": str(target_team),
            "position": int(target_pos),
            "stand": "ahead" if int(target_pos) in (RED_FRONT_POSITIONS + BLUE_FRONT_POSITIONS) else "behind",
            "unit_type": template.get("unit_type", "Warrior"),
            "Level": int(template.get("Level", 0) or 0),
            "damage": int(template.get("damage", 0) or 0),
            "damage_secondary": int(template.get("damage_secondary", 0) or 0),
            "original_damage": int(template.get("damage", 0) or 0),
            "health": max_health,
            "max_health": max_health,
            "armor": int(template.get("armor", 0) or 0),
            "accuracy": int(template.get("accuracy", 0) or 0),
            "accuracy_secondary": int(template.get("accuracy_secondary", 0) or 0),
            "immunity": list(template.get("immunity", [])),
            "resistance": list(template.get("resistance", [])),
            "attack_type_primary": template.get("attack_type_primary", "Weapon"),
            "attack_type_secondary": template.get("attack_type_secondary", ""),
            "big": bool(template.get("big", False)),
            "paralyzed": 0,
            "long_paralyzed": 0,
            "running_away": 0,
            "transformed": 0,
            "hero": False,
            "basestats": {},
            "exp_kill": 0,
            "exp_required": 0,
            "exp_current": 0,
            "next_level_exp": 0,
            "needaunit": 0,
            "turns_into": [],
            "bonusturn": 0,
            "defense": 0,
            "waited": 0,
            "poison_turns_left": 0,
            "poison_damage_per_tick": 0,
            "burn_turns_left": 0,
            "burn_damage_per_tick": 0,
            "uran_turns_left": 0,
            "uran_damage_per_tick": 0,
            "resilience_used_types": [],
            "base_armor": int(template.get("armor", 0) or 0),
        }
        if isinstance(source_unit, dict):
            summon_unit["Summoned"] = int(source_unit.get("position", 0) or 0)
        _apply_transformations_to_unit(summon_unit)
        return summon_unit

    def _apply_hero_item_summon(
        self,
        *,
        source_unit: Optional[Dict],
        target_team: str,
        target_pos: int,
        target_unit: Dict,
        effect: Dict[str, object],
    ) -> float:
        summon_unit = self._build_hero_item_summon_unit(
            source_unit=source_unit,
            target_team=target_team,
            target_pos=target_pos,
            effect=effect,
        )
        if not summon_unit:
            return 0.0
        if bool(summon_unit.get("big", False)) and int(target_pos) in (RED_BACK_POSITIONS + BLUE_BACK_POSITIONS):
            return 0.0
        if int(target_pos) in (RED_BACK_POSITIONS + BLUE_BACK_POSITIONS) and self._is_position_behind_big(target_pos):
            return 0.0

        self._accumulate_overwritten_exp(target_unit)
        target_unit.clear()
        target_unit.update(summon_unit)
        return 1.0

    def _hero_item_effect_can_apply_to_target(
        self,
        *,
        effect: Dict[str, object],
        effect_kind: str,
        target_unit: Dict,
    ) -> bool:
        if effect_kind == "heal":
            return self._alive(target_unit)
        if effect_kind == "revive":
            current_health = float(target_unit.get("health", 0) or 0)
            max_health = float(
                target_unit.get("max_health", 0) or target_unit.get("health", 0) or 0
            )
            return max_health > 0 and current_health <= 0
        if effect_kind == "summon":
            target_pos = int(target_unit.get("position", 0) or 0)
            unit_name = str(effect.get("summon_unit_name", "") or "").strip()
            template = self._hero_item_find_unit_template(unit_name)
            if not template:
                return False
            if bool(template.get("big", False)) and target_pos in (
                RED_BACK_POSITIONS + BLUE_BACK_POSITIONS
            ):
                return False
            if (
                target_pos in (RED_BACK_POSITIONS + BLUE_BACK_POSITIONS)
                and self._is_position_behind_big(target_pos)
            ):
                return False
            return True
        if effect_kind in ("damage", "drain"):
            return self._hero_item_effect_amount(effect) > 0 and self._alive(target_unit)
        if effect_kind == "dot":
            amount = int(round(self._hero_item_effect_amount(effect)))
            return amount > 0 and self._alive(target_unit)
        if effect_kind == "damage_buff":
            if not self._alive(target_unit):
                return False
            try:
                multiplier = float(effect.get("damage_multiplier", 0.0) or 0.0)
            except (TypeError, ValueError):
                multiplier = 0.0
            base_damage = int(target_unit.get("original_damage", 0) or 0)
            if base_damage <= 0:
                base_damage = int(target_unit.get("damage", 0) or 0)
            return base_damage > 0 and int(round(base_damage * multiplier)) != int(
                target_unit.get("damage", 0) or 0
            )
        if effect_kind == "extra_turn":
            return self._alive(target_unit)
        if effect_kind == "paralysis":
            return self._alive(target_unit)
        if effect_kind == "fear":
            return self._alive(target_unit)
        if effect_kind == "transform":
            return self._alive(target_unit)
        if effect_kind == "lycanthropy":
            unit_name = str(effect.get("transform_unit_name", "") or "").strip()
            return (
                self._alive(target_unit)
                and bool(self._hero_item_find_unit_template(unit_name))
            )
        if effect_kind == "debuff_damage":
            return self._alive(target_unit)
        return False

    def _hero_item_target_allowed(
        self,
        *,
        item_name: str,
        target_pos: Optional[int],
        units_by_pos: Optional[Dict[int, Dict]] = None,
    ) -> bool:
        effect = self._hero_item_effect(item_name)
        if not effect or target_pos is None:
            return False
        effect_kind = str(effect.get("kind", "") or "").strip().lower()
        target_unit = (
            units_by_pos.get(int(target_pos))
            if units_by_pos is not None
            else self._unit_by_position(int(target_pos))
        )
        if not isinstance(target_unit, dict):
            return False
        target_team = self._hero_item_effect_target_team(
            effect=effect,
            effect_kind=effect_kind,
        )
        if not target_team or target_unit.get("team") != target_team:
            return False
        targets = self._hero_item_effect_targets(
            effect=effect,
            effect_kind=effect_kind,
            target_team=target_team,
            target_pos=int(target_pos),
            target_unit=target_unit,
        )
        if not targets:
            return False
        return any(
            self._hero_item_effect_can_apply_to_target(
                effect=effect,
                effect_kind=effect_kind,
                target_unit=target,
            )
            for target in targets
        )

    def _apply_hero_item_effect(
        self,
        *,
        item_name: str,
        target_pos: Optional[int],
    ) -> Tuple[bool, Optional[str], float]:
        effect = self._hero_item_effect(item_name)
        if not effect or target_pos is None:
            return False, None, 0.0
        units_by_pos = self._units_by_position()
        target_unit = units_by_pos.get(int(target_pos))
        if not isinstance(target_unit, dict):
            return False, None, 0.0
        effect_kind = str(effect.get("kind", "") or "").strip().lower()
        target_team = self._hero_item_effect_target_team(
            effect=effect,
            effect_kind=effect_kind,
        )
        if not target_team or target_unit.get("team") != target_team:
            return False, None, 0.0
        targets = self._hero_item_effect_targets(
            effect=effect,
            effect_kind=effect_kind,
            target_team=target_team,
            target_pos=int(target_pos),
            target_unit=target_unit,
        )
        if not targets:
            return False, None, 0.0

        source_unit = (
            units_by_pos.get(self.current_blue_attacker_pos)
            if self.current_blue_attacker_pos is not None
            else None
        )
        valid_attempt = any(
            self._hero_item_effect_can_apply_to_target(
                effect=effect,
                effect_kind=effect_kind,
                target_unit=target,
            )
            for target in targets
        )
        if not valid_attempt:
            return False, None, 0.0

        total_value = 0.0
        for target in targets:
            if effect_kind == "heal":
                value = self._apply_hero_item_heal(
                    target,
                    self._hero_item_effect_amount(effect),
                )
            elif effect_kind == "revive":
                value = self._apply_hero_item_revive(
                    target,
                    self._hero_item_effect_amount(effect),
                )
            elif effect_kind == "damage":
                value = self._apply_hero_item_damage(
                    source_unit=source_unit,
                    target_unit=target,
                    effect=effect,
                )
            elif effect_kind == "drain":
                value = self._apply_hero_item_drain(
                    source_unit=source_unit,
                    target_unit=target,
                    effect=effect,
                )
            elif effect_kind == "dot":
                value = self._apply_hero_item_dot(
                    source_unit=source_unit,
                    target_unit=target,
                    effect=effect,
                )
            elif effect_kind == "damage_buff":
                value = self._apply_hero_item_damage_buff(
                    target_unit=target,
                    effect=effect,
                )
            elif effect_kind == "extra_turn":
                value = self._apply_hero_item_extra_turn(
                    source_unit=source_unit,
                    target_unit=target,
                )
            elif effect_kind == "paralysis":
                value = self._apply_hero_item_paralysis(
                    source_unit=source_unit,
                    target_unit=target,
                    effect=effect,
                )
            elif effect_kind == "fear":
                value = self._apply_hero_item_fear(
                    source_unit=source_unit,
                    target_unit=target,
                    effect=effect,
                )
            elif effect_kind == "transform":
                value = self._apply_hero_item_transform(
                    source_unit=source_unit,
                    target_unit=target,
                    effect=effect,
                )
            elif effect_kind == "lycanthropy":
                value = self._apply_hero_item_lycanthropy(
                    source_unit=source_unit,
                    target_unit=target,
                    effect=effect,
                )
            elif effect_kind == "debuff_damage":
                value = self._apply_hero_item_damage_debuff(
                    source_unit=source_unit,
                    target_unit=target,
                    effect=effect,
                )
            elif effect_kind == "summon":
                value = self._apply_hero_item_summon(
                    source_unit=source_unit,
                    target_team=target_team,
                    target_pos=int(target_pos),
                    target_unit=target,
                    effect=effect,
                )
            else:
                value = 0.0
            total_value += max(0.0, float(value or 0.0))

        return True, effect_kind, total_value

    def compute_action_mask(self) -> np.ndarray:
        """
        Возвращает маску допустимых действий для ТЕКУЩЕГО ходящего BLUE атакера.
        True — действие разрешено (можно выбирать), False — запрещено.

        Для ближников пересекаем с досягаемыми целями (_warrior_allowed_targets).
        Для AoE (Mage, Dead dragon, Gumtic, Uter Demon, Tiamat, Teurg) — разрешаем любые живые цели.
        Для остальных (включая стрелков) — разрешаем любые живые цели.
        Hero-only действия предметов доступны только на ходу героя. Для каждого
        предмета есть отдельные окна действий по своим и вражеским позициям 1..6;
        конкретные разрешенные цели выбираются по target_team/target_state/scope
        эффекта предмета.
        """
        n_targets = len(TARGET_POSITIONS)
        mask = np.ones(TOTAL_AGENT_ACTIONS, dtype=bool)  # по умолчанию всё разрешено
        mask_targets = mask[:n_targets]
        first_item_mask = mask[
            FIRST_HERO_ITEM_ACTION_START : FIRST_HERO_ITEM_ACTION_START
            + len(HERO_ITEM_TARGET_SLOTS)
        ]
        second_item_mask = mask[
            SECOND_HERO_ITEM_ACTION_START : SECOND_HERO_ITEM_ACTION_START
            + len(HERO_ITEM_TARGET_SLOTS)
        ]
        first_enemy_item_mask = mask[
            FIRST_HERO_ITEM_ENEMY_ACTION_START : FIRST_HERO_ITEM_ENEMY_ACTION_START
            + len(HERO_ITEM_TARGET_SLOTS)
        ]
        second_enemy_item_mask = mask[
            SECOND_HERO_ITEM_ENEMY_ACTION_START : SECOND_HERO_ITEM_ENEMY_ACTION_START
            + len(HERO_ITEM_TARGET_SLOTS)
        ]
        mask[DEFEND_ACTION_INDEX] = True
        mask[WAIT_ACTION_INDEX] = True
        mask[RUN_AWAY_ACTION_INDEX] = True
        first_item_mask[:] = False
        second_item_mask[:] = False
        first_enemy_item_mask[:] = False
        second_enemy_item_mask[:] = False

        # Если не ход BLUE или нет текущего атакера — не ограничиваем
        if self.current_blue_attacker_pos is None:
            return mask

        units_by_pos = self._units_by_position()
        attacker = units_by_pos.get(self.current_blue_attacker_pos)
        if attacker is None or not self._alive(attacker):
            return mask
        if attacker.get("waited", 0) == 1:
            mask[WAIT_ACTION_INDEX] = False

        demon_second_strike = (
            attacker.get("unit_type") in DOUBLE_STRIKE_TYPES
            and self.blue_attacks_left == 1
        )
        if demon_second_strike:
            mask[DEFEND_ACTION_INDEX] = False
            mask[WAIT_ACTION_INDEX] = False
            mask[RUN_AWAY_ACTION_INDEX] = False

        if _is_combat_hero_unit(attacker):
            first_item_name = self._hero_item_name_for_slot(1)
            second_item_name = self._hero_item_name_for_slot(2)
            first_item_effect = self._hero_item_effect(first_item_name)
            second_item_effect = self._hero_item_effect(second_item_name)
            first_item_available = bool(first_item_effect) and self._hero_item_slot_available(
                0,
                first_item_effect,
            )
            second_item_available = bool(second_item_effect) and self._hero_item_slot_available(
                1,
                second_item_effect,
            )
            for idx, slot_no in enumerate(HERO_ITEM_TARGET_SLOTS):
                battle_pos = BLUE_POSITIONS[slot_no - 1]
                enemy_battle_pos = RED_POSITIONS[slot_no - 1]
                first_item_mask[idx] = first_item_available and self._hero_item_target_allowed(
                    item_name=first_item_name,
                    target_pos=battle_pos,
                    units_by_pos=units_by_pos,
                )
                second_item_mask[idx] = second_item_available and self._hero_item_target_allowed(
                    item_name=second_item_name,
                    target_pos=battle_pos,
                    units_by_pos=units_by_pos,
                )
                first_enemy_item_mask[idx] = first_item_available and self._hero_item_target_allowed(
                    item_name=first_item_name,
                    target_pos=enemy_battle_pos,
                    units_by_pos=units_by_pos,
                )
                second_enemy_item_mask[idx] = second_item_available and self._hero_item_target_allowed(
                    item_name=second_item_name,
                    target_pos=enemy_battle_pos,
                    units_by_pos=units_by_pos,
                )

        def is_alive_red(pos: int) -> bool:
            u = units_by_pos.get(pos)
            return (u is not None) and self._alive(u) and (u["team"] == "red")

        atype = attacker.get("unit_type")

        if atype == "Doppelganger":
            forbidden_names = DOPPELGANGER_FORBIDDEN_COPY_NAMES
            attacker_pos = attacker.get("position")
            for i, pos in enumerate(TARGET_POSITIONS):
                if pos == attacker_pos:
                    mask_targets[i] = False
                    continue

                target_unit = units_by_pos.get(pos)
                if target_unit is None or not self._alive(target_unit):
                    mask_targets[i] = False
                    continue

                if target_unit.get("big", False):
                    mask_targets[i] = False
                    continue

                if target_unit.get("name") in forbidden_names:
                    mask_targets[i] = False
                    continue

                mask_targets[i] = True

            return mask

        if attacker.get("team") == "blue" and self._is_summoner_type(atype):
            # посчитаем занятые синие позиции живыми союзниками (big учитывает пару)
            occupied: set[int] = set()
            for u in self.combined:
                if u.get("team") != "blue" or not self._alive(u):
                    continue
                p = int(u.get("position", 0) or 0)
                occupied.add(p)
                if u.get("big", False):
                    # big занимает партнёра в своём столбце (±3)
                    if p in BLUE_FRONT_POSITIONS:
                        occupied.add(p + 3)
                    elif p in BLUE_BACK_POSITIONS:
                        occupied.add(p - 3)

            empty_blue = {p for p in BLUE_POSITIONS if p not in occupied}

            # Маска по первым 6 действиям: разрешаем только те,
            # чья противоположная синяя клетка пустая
            for i, red_pos in enumerate(TARGET_POSITIONS):
                opp_blue = self._opposite_position(red_pos, "blue")
                mask_targets[i] = opp_blue in empty_blue

            # Ранний выход: ниже общая ветка «любые живые цели» затёрла бы клетки призыва.
            return mask

        # Ближники: только досягаемые цели
        if atype in MELEE_TYPES:
            allowed_reach = set(self._warrior_allowed_targets(attacker, units_by_pos=units_by_pos))
            for i, pos in enumerate(TARGET_POSITIONS):
                mask_targets[i] = is_alive_red(pos) and (pos in allowed_reach)

            # Страховка от пустой маски:
            if not mask_targets.any():
                live_idxs = [
                    i for i, pos in enumerate(TARGET_POSITIONS) if is_alive_red(pos)
                ]
                if live_idxs:
                    mask_targets[live_idxs] = True

        # AoE: формальная цель, запрещаем только мёртвые слоты
        elif atype in AOE_TYPES:
            attacker_pos = attacker.get("position")
            for i, pos in enumerate(TARGET_POSITIONS):
                mask_targets[i] = is_alive_red(pos) or (
                    atype == "Wolf Lord" and pos == attacker_pos
                )
            if not mask_targets.any():
                mask_targets[:] = True
        else:
            if atype in SUPPORT_TYPES:
                for i, pos in enumerate(TARGET_POSITIONS):
                    opposite_pos = self._opposite_position(pos, attacker["team"])
                    if opposite_pos is None:
                        mask_targets[i] = False
                        continue
                    recipient = units_by_pos.get(opposite_pos)
                    if atype in PATRIACH_SUPPORT_TYPES:
                        mask_targets[i] = (
                            recipient is not None
                            and recipient.get("team") == attacker.get("team")
                            and recipient.get("name") != "пусто"
                            and not bool(recipient.get("Summoned"))
                            and not self._is_position_behind_big(
                                int(recipient.get("position", 0) or 0)
                            )
                            and (
                                self._alive(recipient)
                                or not self._patriach_already_revived(recipient)
                            )
                        )
                    else:
                        mask_targets[i] = (
                            recipient is not None
                            and self._alive(recipient)
                            and recipient.get("team") == attacker.get("team")
                        )
                if not mask_targets.any():
                    mask_targets[:] = False
            else:
                # Прочие (включая стрелков): любые живые цели.
                for i, pos in enumerate(TARGET_POSITIONS):
                    mask_targets[i] = is_alive_red(pos)
                if not mask_targets.any():
                    mask_targets[:] = True

        if self._post_victory_team == "blue":
            # The battle has already been won. During the deferred healing
            # phase only native support targets (or an explicit skip) remain.
            mask[n_targets:] = False
            mask[WAIT_ACTION_INDEX] = True

        return mask

    # ------------------ Вспомогательные методы ------------------

    def seed(self, seed=None):
        # Метод оставлен для совместимости, но мы им не пользуемся, чтобы не фиксировать случайность
        self.rng = random.Random(seed)

    def _alive(self, u) -> bool:
        return u["health"] > 0

    @staticmethod
    def _is_summoner_type(unit_type: object) -> bool:
        # Тип в данных пишется как "Summoner" — сравниваем без учёта регистра.
        return str(unit_type or "").strip().lower() == "summoner"

    def _accumulate_overwritten_exp(self, unit) -> None:
        """Копит exp_kill трупов, затёртых призывом, чтобы победитель не терял опыт."""
        if not isinstance(unit, dict) or self._alive(unit):
            return
        team = str(unit.get("team", "") or "")
        bank = getattr(self, "overwritten_exp_kill", None)
        if isinstance(bank, dict) and team in bank:
            bank[team] += float(unit.get("exp_kill", 0) or 0)

    def _record_unit_defeat_for_exp(self, victim: Dict) -> None:
        """Record one death and award its raw XP to units present at that moment."""
        if not isinstance(victim, dict):
            return
        victim_team = str(victim.get("team", "") or "").lower()
        if victim_team not in {"red", "blue"}:
            return

        # A dead unit loses everything earned earlier in this battle.  If it is
        # revived, subsequent kills start filling this counter again from zero.
        victim["_battle_exp_earned"] = 0.0

        defeated_exp = max(0.0, float(victim.get("exp_kill", 0) or 0))
        defeated_by_team = getattr(self, "_battle_defeated_exp", None)
        if not isinstance(defeated_by_team, dict):
            defeated_by_team = {"red": 0.0, "blue": 0.0}
            self._battle_defeated_exp = defeated_by_team
        defeated_by_team[victim_team] = (
            float(defeated_by_team.get(victim_team, 0.0) or 0.0) + defeated_exp
        )
        self._battle_exp_event_count = int(
            getattr(self, "_battle_exp_event_count", 0) or 0
        ) + 1

        receiving_team = "red" if victim_team == "blue" else "blue"
        recipients = [
            unit
            for unit in self.combined
            if str(unit.get("team", "") or "").lower() == receiving_team
            and self._alive(unit)
            and not _is_thief_battle_unit(unit)
        ]
        if not recipients or defeated_exp <= 0.0:
            return

        share = defeated_exp / len(recipients)
        for unit in recipients:
            unit["_battle_exp_earned"] = max(
                0.0,
                float(unit.get("_battle_exp_earned", 0.0) or 0.0),
            ) + share

    def _team_alive(self, team: str) -> bool:
        return any(self._alive(u) and u["team"] == team for u in self.combined)

    def _has_transform_targets(self) -> bool:
        forbidden = {
            "Ашган",
            "Ашкаэль",
            "Видар",
            "Мизраэль",
            "Иллюмиэлль",
            "Драллиаан",
            "Лаклаан",
        }
        for unit in self.combined:
            if not self._alive(unit):
                continue
            if unit.get("big", False):
                continue
            if unit.get("name") in forbidden:
                continue
            return True
        return False

    def _restore_default_doppelgangers(self) -> None:
        defaults = {
            "initiative": 80,
            "initiative_base": 80,
            "unit_type": "Doppelganger",
            "damage": 30,
            "damage_secondary": 0,
            "max_health": 120,
            "armor": 0,
            "accuracy": 100,
            "accuracy_secondary": 0,
            "immunity": [],
            "resistance": [],
            "attack_type_primary": "Mind",
            "attack_type_secondary": "",
        }

        for unit in self.combined:
            if unit.get("name") != "Двойник":
                continue
            unit["doppel_copied"] = 0
            if not self._alive(unit):
                continue

            prev_max = float(unit.get("max_health", 0) or 0)
            current_health = float(unit.get("health", 0) or 0)
            ratio = (
                0.0 if prev_max <= 0 else max(0.0, min(1.0, current_health / prev_max))
            )

            for key, value in defaults.items():
                unit[key] = list(value) if isinstance(value, list) else value

            unit["health"] = int(round(defaults["max_health"] * ratio))

    def _apply_doppelganger_copy(
        self, attacker: Dict, target_unit: Optional[Dict]
    ) -> bool:
        """Двойник становится копией цели; текущее HP переносится долей."""
        if target_unit is None or not self._alive(target_unit):
            return False

        current_health = float(attacker.get("health", 0) or 0)
        current_max_health = float(attacker.get("max_health", 1) or 1)
        attacker_ratio = (
            0.0 if current_max_health <= 0 else current_health / current_max_health
        )

        attacker["unit_type"] = target_unit.get("unit_type", attacker["unit_type"])
        attacker["initiative"] = target_unit.get("initiative", attacker["initiative"])
        attacker["initiative_base"] = target_unit.get(
            "initiative_base", attacker["initiative_base"]
        )
        attacker["damage"] = target_unit.get("damage", attacker["damage"])
        attacker["damage_secondary"] = target_unit.get(
            "damage_secondary", attacker.get("damage_secondary")
        )
        attacker["max_health"] = target_unit.get("max_health", attacker["max_health"])
        attacker["armor"] = target_unit.get("armor", attacker["armor"])
        attacker["accuracy"] = target_unit.get("accuracy", attacker["accuracy"])
        attacker["accuracy_secondary"] = target_unit.get(
            "accuracy_secondary", attacker.get("accuracy_secondary")
        )
        attacker["immunity"] = list(target_unit.get("immunity", []))
        attacker["resistance"] = list(target_unit.get("resistance", []))
        attacker["attack_type_primary"] = target_unit.get(
            "attack_type_primary", attacker.get("attack_type_primary")
        )
        attacker["attack_type_secondary"] = target_unit.get(
            "attack_type_secondary", attacker.get("attack_type_secondary")
        )

        target_health = float(target_unit.get("health", 0) or 0)
        attacker["health"] = int(
            round(target_health * max(0.0, min(1.0, attacker_ratio)))
        )
        attacker["doppel_copied"] = 1
        self._log(
            f"Doppelganger выбор: pos{target_unit['position']} → "
            f"{target_unit['team'].upper()} {target_unit['name']}#{target_unit['position']}"
        )
        return True

    def _unit_by_position(self, pos: int):
        return next((u for u in self.combined if u["position"] == pos), None)

    def _units_by_position(self) -> Dict[int, Dict]:
        units_by_pos: Dict[int, Dict] = {}
        for unit in self.combined:
            units_by_pos.setdefault(unit["position"], unit)
        return units_by_pos

    def _live_positions_of(self, team: str):
        return [
            u["position"] for u in self.combined if u["team"] == team and self._alive(u)
        ]

    def _is_position_behind_big(self, pos: int) -> bool:
        if pos in RED_BACK_POSITIONS or pos in BLUE_BACK_POSITIONS:
            front_pos = pos - 3
            partner = self._unit_by_position(front_pos)
            if (
                partner is not None
                and self._alive(partner)
                and partner.get("big", False)
            ):
                return True
        return False

    def _log(self, s: str):
        if self.log_enabled:
            self._pretty_events.append(s)

    def pop_pretty_events(self) -> List[str]:
        out = self._pretty_events[:]
        self._pretty_events.clear()
        return out

    def _col_of(self, pos: int) -> int:
        return (pos - 1) % 3

    def _enemy_rows(self, team: str):
        if team == "red":
            return [7, 8, 9], [10, 11, 12]
        else:
            return [1, 2, 3], [4, 5, 6]

    def _warrior_near_far_cols(self, col: int):
        if col == 0:
            return [0, 1], [2]
        if col == 1:
            return [0, 1, 2], []
        return [1, 2], [0]

    def _opposite_position(self, pos: int, team: str) -> Optional[int]:
        col = self._col_of(pos)
        if team == "blue":
            if pos in RED_FRONT_POSITIONS:
                return BLUE_FRONT_POSITIONS[col]
            if pos in RED_BACK_POSITIONS:
                return BLUE_BACK_POSITIONS[col]
        else:
            if pos in BLUE_FRONT_POSITIONS:
                return RED_FRONT_POSITIONS[col]
            if pos in BLUE_BACK_POSITIONS:
                return RED_BACK_POSITIONS[col]
        return None

    def _transform_snapshot_keys(self, include_health_fields: bool = False) -> Tuple[str, ...]:
        keys = (
            "accuracy",
            "accuracy_secondary",
            "damage",
            "damage_secondary",
            "original_damage",
            "attack_type_primary",
            "attack_type_secondary",
            "initiative",
            "initiative_base",
            "unit_type",
            "stand",
            "armor",
            "immunity",
            "resistance",
            "resilience_used_types",
            "big",
            "Level",
            "next_level_exp",
            "needaunit",
            "transformed",
            "wight_form_name",
            "transform_effect",
            "transform_recover_chance",
        )
        if include_health_fields:
            keys += ("max_health", "maxhp")
        return keys

    def _build_transform_snapshot(
        self, unit: Dict, include_health_fields: bool = False
    ) -> Dict:
        snapshot = {}
        for key in self._transform_snapshot_keys(include_health_fields):
            if key in unit:
                snapshot[key] = deepcopy(unit[key])
        snapshot.setdefault(
            "initiative",
            deepcopy(unit.get("initiative_base", unit.get("initiative", 0))),
        )
        snapshot.setdefault("transformed", int(unit.get("transformed", 0) or 0))
        return snapshot

    def _ensure_transform_snapshot(
        self, unit: Dict, include_health_fields: bool = False
    ) -> Dict:
        pos_key = unit.get("position")
        basestats = unit.get("basestats")
        if isinstance(basestats, dict):
            snapshots = basestats.get(pos_key)
            if not isinstance(snapshots, list):
                snapshots = []
                basestats[pos_key] = snapshots
        else:
            snapshots = basestats if isinstance(basestats, list) else []
            basestats = {pos_key: snapshots}
            unit["basestats"] = basestats

        if not snapshots:
            snapshot = self._build_transform_snapshot(unit, include_health_fields)
            snapshots.append(snapshot)
            return snapshot

        snapshot = snapshots[0]
        if isinstance(snapshot, dict):
            for key in self._transform_snapshot_keys(include_health_fields):
                if key not in snapshot and key in unit:
                    snapshot[key] = deepcopy(unit[key])
            snapshot.setdefault(
                "initiative",
                deepcopy(unit.get("initiative_base", unit.get("initiative", 0))),
            )
            snapshot.setdefault("transformed", int(unit.get("transformed", 0) or 0))
            return snapshot

        snapshot = self._build_transform_snapshot(unit, include_health_fields)
        snapshots[0] = snapshot
        return snapshot

    def _transform_snapshots_for_unit(self, unit: Dict) -> Tuple[object, List[Dict]]:
        basestats = unit.get("basestats")
        pos_key = unit.get("position")
        if isinstance(basestats, dict):
            snapshots = basestats.get(pos_key) or []
            return basestats, snapshots if isinstance(snapshots, list) else []
        if isinstance(basestats, list):
            return basestats, basestats
        return basestats, []

    def _scale_health_to_new_max(
        self, current_health: object, current_max: object, new_max: object
    ) -> int:
        try:
            old_hp = max(0.0, float(current_health or 0))
        except (TypeError, ValueError):
            old_hp = 0.0
        try:
            old_max = max(0.0, float(current_max or 0))
        except (TypeError, ValueError):
            old_max = 0.0
        try:
            target_max = max(0, int(round(float(new_max or 0))))
        except (TypeError, ValueError):
            target_max = 0
        if target_max <= 0:
            return 0
        if old_max <= 0:
            return target_max if old_hp > 0 else 0
        scaled = int(round((old_hp / old_max) * target_max))
        if old_hp > 0 and scaled <= 0:
            return 1
        return max(0, min(target_max, scaled))

    def _restore_transformed_unit(self, unit: Optional[Dict]) -> bool:
        if not isinstance(unit, dict) or not unit.get("transformed", 0):
            return False
        basestats, snapshots = self._transform_snapshots_for_unit(unit)
        snapshot = snapshots[0] if snapshots else None
        if not isinstance(snapshot, dict):
            unit["transformed"] = 0
            unit.pop("wight_form_name", None)
            unit.pop("transform_effect", None)
            unit.pop("transform_recover_chance", None)
            return False

        current_health = unit.get("health", unit.get("hp", 0))
        current_max = unit.get("max_health", unit.get("maxhp", current_health))
        restored_max = snapshot.get("max_health", snapshot.get("maxhp"))

        for key in self._transform_snapshot_keys(include_health_fields=False):
            if key in snapshot:
                unit[key] = deepcopy(snapshot[key])

        if restored_max is not None:
            restored_health = self._scale_health_to_new_max(
                current_health, current_max, restored_max
            )
            unit["max_health"] = deepcopy(restored_max)
            unit["health"] = restored_health
            if "maxhp" in unit or "maxhp" in snapshot:
                unit["maxhp"] = deepcopy(restored_max)
            if "hp" in unit or "hp" in snapshot:
                unit["hp"] = restored_health

        if isinstance(basestats, dict):
            basestats.pop(unit.get("position"), None)
        elif isinstance(basestats, list):
            try:
                snapshots.pop(0)
            except IndexError:
                pass

        if not unit.get("transformed", 0):
            unit.pop("wight_form_name", None)
            unit.pop("transform_effect", None)
            unit.pop("transform_recover_chance", None)
        return True

    def _restore_all_transformed_units(self) -> None:
        for unit in self.combined:
            self._restore_transformed_unit(unit)

    def _warrior_allowed_targets(
        self,
        attacker: Dict,
        *,
        units_by_pos: Optional[Dict[int, Dict]] = None,
    ) -> List[int]:
        assert attacker.get("unit_type") in MELEE_TYPES
        if attacker["stand"] == "behind":
            if any(
                self._alive(u)
                and u["team"] == attacker["team"]
                and u.get("unit_type") in MELEE_TYPES
                and u["stand"] == "ahead"
                for u in self.combined
            ):
                return []
        ahead, behind = self._enemy_rows(attacker["team"])
        col = self._col_of(attacker["position"])
        near_cols, far_cols = self._warrior_near_far_cols(col)

        if units_by_pos is None:
            units_by_pos = self._units_by_position()

        def alive(pos):
            uu = units_by_pos.get(pos)
            return uu is not None and self._alive(uu)

        near = [ahead[c] for c in near_cols if alive(ahead[c])]
        if near:
            return near
        far = [ahead[c] for c in far_cols if alive(ahead[c])]
        if far:
            return far

        near2 = [behind[c] for c in near_cols if alive(behind[c])]
        if near2:
            return near2
        far2 = [behind[c] for c in far_cols if alive(behind[c])]
        return far2

    def _travnitsa_auto_target(self, healer: Dict) -> Optional[int]:
        allies = [
            u
            for u in self.combined
            if u["team"] == healer["team"] and self._alive(u) and u is not healer
        ]
        if not allies:
            return None
        if healer.get("team") == "red":

            def dmg_val(unit: Dict) -> int:
                return int(unit.get("damage", 0) or 0)

            max_damage = max(dmg_val(u) for u in allies)
            top_candidates = [u for u in allies if dmg_val(u) == max_damage]
            return self.rng.choice(top_candidates)["position"]
        return self.rng.choice(allies)["position"]

    def _summoner_auto_target(self, summoner: Dict) -> Optional[int]:
        def slot_empty(pos: int) -> bool:
            unit = self._unit_by_position(pos)
            return unit is None or not self._alive(unit)

        front_candidates = [pos for pos in RED_FRONT_POSITIONS if slot_empty(pos)]
        if front_candidates:
            chosen_pos = self.rng.choice(front_candidates)
        else:
            back_candidates = [
                pos
                for pos in RED_BACK_POSITIONS
                if slot_empty(pos) and not self._is_position_behind_big(pos)
            ]
            if not back_candidates:
                return None
            chosen_pos = self.rng.choice(back_candidates)

        if chosen_pos in RED_FRONT_POSITIONS:
            col = RED_FRONT_POSITIONS.index(chosen_pos)
            return BLUE_FRONT_POSITIONS[col]
        col = RED_BACK_POSITIONS.index(chosen_pos)
        return BLUE_BACK_POSITIONS[col]

    def _alchemist_auto_target(self, supporter: Dict) -> Optional[int]:
        allies = [
            u
            for u in self.combined
            if u["team"] == supporter["team"]
            and self._alive(u)
            and u is not supporter
            and u.get("unit_type") != "Alchemist"
            and u.get("running_away", 0) != 1
        ]
        if not allies:
            return None
        needs_help = [
            u
            for u in allies
            if int(u.get("initiative", 0) or 0) < int(u.get("initiative_base", 0) or 0)
        ]
        if supporter.get("team") == "red":
            if not needs_help:
                return None

            def dmg_value(unit: Dict) -> int:
                return int(unit.get("damage", 0) or 0)

            positive = [u for u in needs_help if dmg_value(u) > 0]
            if not positive:
                return None
            max_dmg = max(dmg_value(u) for u in positive)
            best = [u for u in positive if dmg_value(u) == max_dmg]
            return self.rng.choice(best)["position"]

        pool = needs_help if needs_help else allies
        chosen = self.rng.choice(pool)
        return chosen["position"]

    def _cliric_auto_target(self, healer: Dict) -> Optional[int]:
        allies = [
            u
            for u in self.combined
            if u["team"] == healer["team"]
            and self._alive(u)
            and u["health"] < u.get("max_health", 0)
        ]
        if not allies:
            return None
        min_hp = min(u["health"] for u in allies)
        candidates = [u for u in allies if u["health"] == min_hp]
        return self.rng.choice(candidates)["position"]

    def _patriach_targetable_allies(self, healer: Dict) -> List[Dict]:
        team = healer.get("team")
        return [
            unit
            for unit in self.combined
            if unit.get("team") == team
            and unit.get("name") != "пусто"
            and not bool(unit.get("Summoned"))
            and not self._is_position_behind_big(
                int(unit.get("position", 0) or 0)
            )
            and (
                self._alive(unit)
                or not self._patriach_already_revived(unit)
            )
        ]

    @staticmethod
    def _patriach_recipient_key(recipient: Dict) -> Tuple[str, int]:
        return (
            str(recipient.get("team", "") or ""),
            int(recipient.get("position", 0) or 0),
        )

    def _patriach_already_revived(self, recipient: Dict) -> bool:
        revived_recipients = getattr(self, "_patriach_revived_recipients", set())
        return self._patriach_recipient_key(recipient) in revived_recipients

    def _patriach_auto_target(self, healer: Dict) -> Optional[int]:
        allies = self._patriach_targetable_allies(healer)
        dead_allies = [
            unit
            for unit in allies
            if (unit.get("health", 0) or 0) <= 0
            and int(unit.get("initiative", 0) or 0) <= 0
        ]
        if dead_allies:
            return self.rng.choice(dead_allies)["position"]

        wounded = [
            unit
            for unit in allies
            if self._alive(unit) and unit.get("health", 0) < unit.get("max_health", 0)
        ]
        if not wounded:
            return None

        min_hp = min(u["health"] for u in wounded)
        candidates = [u for u in wounded if u["health"] == min_hp]
        return self.rng.choice(candidates)["position"]

    def _pick_lowest_hp(self, positions: List[int]) -> Optional[int]:
        """
        Вернуть позицию цели с наименьшим текущим HP среди заданных позиций.
        Берём только живых; при равенстве HP — случайная из минимальных.
        """
        candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is not None and self._alive(u):
                candidates.append((u["health"], p))
        if not candidates:
            return None
        min_hp = min(hp for hp, _ in candidates)
        best = [p for hp, p in candidates if hp == min_hp]
        return self.rng.choice(best)

    def _filter_non_immune_targets(self, attacker: Dict, positions: List[int]) -> List[int]:
        """Возвращает список позиций, цели на которых НЕ имеют иммунитета к основной атаке атакующего."""
        attack_type = attacker.get("attack_type_primary", "Weapon")
        attack_type_lower = str(attack_type).lower()
        valid_positions = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is None or not self._alive(u):
                continue
            immunities = [str(val).lower() for val in (u.get("immunity") or [])]
            if attack_type_lower not in immunities:
                valid_positions.append(p)
        return valid_positions

    def _pick_highest_hp(self, positions: List[int]) -> Optional[int]:
        candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is not None and self._alive(u):
                candidates.append((u["health"], p))
        if not candidates:
            return None
        max_hp = max(hp for hp, _ in candidates)
        best = [p for hp, p in candidates if hp == max_hp]
        return self.rng.choice(best)

    def _pick_highest_damage_non_paralyzed(self, positions: List[int]) -> Optional[int]:
        """
        Выбирает цель с наибольшим показателем damage, на которой нет паралича (обычного или долгого).
        Если таких целей несколько — выбирается случайная из них.
        Если подходящих целей нет, возвращается None.
        """
        candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is None or not self._alive(u):
                continue
            # Проверка на наличие паралича
            if u.get("paralyzed", 0) == 1 or u.get("long_paralyzed", 0) == 1:
                continue
            candidates.append((u.get("damage", 0), p))
        
        if not candidates:
            return None
            
        max_dmg = max(dmg for dmg, _ in candidates)
        best = [p for dmg, p in candidates if dmg == max_dmg]
        return self.rng.choice(best)

    def _pick_highest_hp_non_big(
        self, exclude_unit: Optional[Dict] = None
    ) -> Optional[int]:
        candidates: List[Tuple[int, int]] = []
        for unit in self.combined:
            if not self._alive(unit):
                continue
            if bool(unit.get("big", False)):
                continue
            if exclude_unit is not None and unit is exclude_unit:
                continue
            if unit.get("name") in DOPPELGANGER_FORBIDDEN_COPY_NAMES:
                continue
            candidates.append((int(unit.get("health", 0) or 0), unit["position"]))
        if not candidates:
            return None
        max_hp = max(hp for hp, _ in candidates)
        best = [pos for hp, pos in candidates if hp == max_hp]
        return self.rng.choice(best)

    def _pick_weapon_immune_or_highest_hp(self, positions: List[int]) -> Optional[int]:
        immune_candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is None or not self._alive(u):
                continue
            immunities = [str(val).lower() for val in (u.get("immunity") or [])]
            if "weapon" in immunities:
                immune_candidates.append(p)
        if immune_candidates:
            return self.rng.choice(immune_candidates)
        return self._pick_highest_hp(positions)

    def _apply_alchemist_support(self, alchemist: Dict, recipient: Dict) -> bool:
        if recipient is None or not self._alive(recipient):
            return False
        if recipient is alchemist:
            return False
        if (
            recipient.get("unit_type") == "Alchemist"
            or recipient.get("running_away", 0) == 1
        ):
            self._log(
                f"Дополнительный ход без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не может получить бонус от алхимика."
            )
            return False

        recipient.setdefault("bonusturn", 0)
        base_ini = int(recipient.get("initiative_base", 0) or 0)
        recipient["initiative"] = base_ini
        recipient["bonusturn"] += 1
        self._log(
            f"Дополнительный ход: {alchemist['team'].upper()} {alchemist['name']}#{alchemist['position']} восстанавливает инициативу "
            f"{recipient['team'].upper()} {recipient['name']}#{recipient['position']} до {base_ini} и увеличивает bonusturn до {recipient['bonusturn']}."
        )
        return True

    def _apply_cliric_heal(self, healer: Dict, recipient: Optional[Dict]) -> bool:
        if recipient is None or not self._alive(recipient):
            return False

        heal_amount = int(healer.get("damage", 0) or 0)
        if heal_amount <= 0:
            self._log(
                f"? Лечение без силы: {healer['team'].upper()} {healer['name']}#{healer['position']} имеет недостаточный урон (damage={heal_amount})."
            )
            return False

        if healer.get("unit_type") == "Sundancer":
            if "resistance" not in recipient:
                recipient["resistance"] = []
            if "Fire" not in recipient["resistance"]:
                recipient["resistance"].append("Fire")
            recipient.setdefault("resilience_used_types", [])
            if "Fire" in recipient["resilience_used_types"]:
                recipient["resilience_used_types"].remove("Fire")
            recipient["Firedefence"] = 1
            self._log(
                f"? Защита от огня: {healer['team'].upper()} {healer['name']}#{healer['position']} даёт огненную защиту "
                f"для {recipient['team'].upper()} {recipient['name']}#{recipient['position']}"
            )

        elif healer.get("unit_type") == "Sylfid":
            if "resistance" not in recipient:
                recipient["resistance"] = []
            if "Air" not in recipient["resistance"]:
                recipient["resistance"].append("Air")
            recipient.setdefault("resilience_used_types", [])
            if "Air" in recipient["resilience_used_types"]:
                recipient["resilience_used_types"].remove("Air")
            recipient["Airdefence"] = 1
            self._log(
                f"? Защита от воздуха: {healer['team'].upper()} {healer['name']}#{healer['position']} даёт воздушную защиту "
                f"для {recipient['team'].upper()} {recipient['name']}#{recipient['position']}"
            )

        elif healer.get("unit_type") == "Deva roshi":
            if "resistance" not in recipient:
                recipient["resistance"] = []
            recipient.setdefault("resilience_used_types", [])
            element_map = (
                ("Fire", "Firedefence"),
                ("Air", "Airdefence"),
                ("Water", "Waterdefence"),
                ("Earth", "Earthdefence"),
            )
            for element, field in element_map:
                if element not in recipient["resistance"]:
                    recipient["resistance"].append(element)
                if element in recipient["resilience_used_types"]:
                    recipient["resilience_used_types"].remove(element)
                recipient[field] = 1
            self._log(
                f"? Четверная защита: {healer['team'].upper()} {healer['name']}#{healer['position']} усиливает защиту "
                f"{recipient['team'].upper()} {recipient['name']}#{recipient['position']} от огня, воздуха, воды и земли."
            )

        before = recipient["health"]
        max_hp = recipient.get("max_health", before)
        if before >= max_hp:
            return False

        healed = min(heal_amount, max_hp - before)
        if healed <= 0:
            return False

        recipient["health"] += healed
        self._log(
            f"? Лечение: {healer['team'].upper()} {healer['name']}#{healer['position']} восстанавливает {healed} HP "
            f"для {recipient['team'].upper()} {recipient['name']}#{recipient['position']} ({before}>{recipient['health']})."
        )

        return True

    def _apply_patriach_support(self, healer: Dict, recipient: Optional[Dict]) -> str:
        if recipient is None:
            return "invalid"
        if recipient.get("team") != healer.get("team"):
            return "invalid"
        if recipient.get("name") == "пусто":
            return "invalid"
        if bool(recipient.get("Summoned")):
            return "invalid"
        if self._is_position_behind_big(
            int(recipient.get("position", 0) or 0)
        ):
            return "invalid"

        if self._alive(recipient):
            healed = self._apply_cliric_heal(healer, recipient)
            return "heal_success" if healed else "heal_no_effect"

        if self._patriach_already_revived(recipient):
            return "invalid"

        health = float(recipient.get("health", 0) or 0)
        initiative = int(recipient.get("initiative", 0) or 0)
        if health > 0 or initiative > 0:
            return "invalid"

        max_hp = int(recipient.get("max_health", 0) or 0)
        if max_hp <= 0:
            return "revive_failed"

        restored = max(1, int(round(max_hp * 0.5)))
        self._patriach_revived_recipients.add(
            self._patriach_recipient_key(recipient)
        )
        recipient["health"] = restored
        recipient["initiative"] = 0
        recipient["paralyzed"] = 0
        recipient["long_paralyzed"] = 0
        recipient["running_away"] = 0
        recipient.setdefault("poison_turns_left", 0)
        recipient.setdefault("burn_turns_left", 0)
        recipient.setdefault("uran_turns_left", 0)
        recipient.setdefault("poison_damage_per_tick", 0)
        recipient.setdefault("burn_damage_per_tick", 0)
        recipient.setdefault("uran_damage_per_tick", 0)
        recipient["poison_turns_left"] = 0
        recipient["burn_turns_left"] = 0
        recipient["uran_turns_left"] = 0
        recipient["poison_damage_per_tick"] = 0
        recipient["burn_damage_per_tick"] = 0
        recipient["uran_damage_per_tick"] = 0

        self._cleanse_negative_effects(recipient)
        self._log(
            f"? Воскрешение Патриарха: {healer['team'].upper()} {healer['name']}#{healer['position']} "
            f"возвращает {recipient['team'].upper()} {recipient['name']}#{recipient['position']} к жизни ({restored}/{max_hp})."
        )
        return "revive_success"

    def _cleanse_negative_effects(self, unit: Optional[Dict]) -> bool:
        """Снимает доты/контроль/статовые дебаффы, если юнит жив."""
        if unit is None or not self._alive(unit):
            return False

        cleared = False
        burn_cleared = False  # следим, что именно поджог снят
        poison_cleared = False  # отметка для яда
        uran_cleared = False  # отметка для «воды»

        # Сбрасываем периодический урон (яд/поджог/«вода») вместе с их счётчиками.
        for turns_key, dmg_key in (
            ("poison_turns_left", "poison_damage_per_tick"),
            ("burn_turns_left", "burn_damage_per_tick"),
            ("uran_turns_left", "uran_damage_per_tick"),
        ):
            had_effect = unit.get(turns_key, 0) > 0 or unit.get(dmg_key, 0) > 0
            if had_effect:
                unit[turns_key] = 0
                unit[dmg_key] = 0
                cleared = True
                if turns_key == "burn_turns_left":
                    burn_cleared = True
                elif turns_key == "poison_turns_left":
                    poison_cleared = True
                elif turns_key == "uran_turns_left":
                    uran_cleared = True

        # Снимаем паралич, долгий паралич и состояние «бегства».
        for flag in ("paralyzed", "long_paralyzed"):
            if unit.get(flag, 0):
                unit[flag] = 0
                cleared = True

        # Hermit-замедление: возвращаем базовую инициативу и убираем флаг.
        if unit.get("hermited", 0):
            base_ini = int(unit.get("initiative_base", 0) or 0)
            unit["initiative_base"] = base_ini * 2 if base_ini > 0 else base_ini
            unit["initiative"] = max(
                int(unit.get("initiative", 0) or 0), unit["initiative_base"]
            )
            unit["hermited"] = 0
            cleared = True

        # Если юнит превращён ведьмой/суккубом/Сущим — откатываем сохранённые характеристики.
        if unit.get("transformed", 0):
            if self._restore_transformed_unit(unit):
                cleared = True

        # Если убрали поджог – сбрасываем кэш Владыки так же, как poison/water-кэши.
        if burn_cleared:
            self._lord_applied_burn.clear()
        if poison_cleared:
            self._spider_applied_poison.clear()
            self._dregazul_applied_poison.clear()
        if uran_cleared:
            self._ismir_applied_uran.clear()

        if cleared:
            self._log(
                f"Очищение: {unit['team'].upper()} {unit['name']}#{unit['position']} избавляется от негативных эффектов."
            )
        return cleared

    def _kill_linked_summons(self, summoner: Optional[Dict]) -> None:
        """Kills all alive summons that are linked to the provided summoner."""
        if not summoner:
            return

        summoner_type = (summoner.get("unit_type") or "").lower()
        if summoner_type not in {"summoner", "occultmaster", "laclaan", "lyf"}:
            return

        summoner_team = summoner.get("team")
        summoner_pos = summoner.get("position")
        if summoner_team is None or summoner_pos is None:
            return

        for unit in self.combined:
            if unit is summoner:
                continue
            if unit.get("team") != summoner_team:
                continue
            if not self._alive(unit):
                continue
            if unit.get("Summoned") != summoner_pos:
                continue

            before_hp = int(unit.get("health", 0) or 0)
            unit["health"] = 0
            if "hp" in unit:
                unit["hp"] = 0
            self._record_unit_defeat_for_exp(unit)
            unit["initiative"] = 0
            unit_team = (unit.get("team") or "").upper()
            unit_name = unit.get("name") or unit.get("unit_type") or "unit"
            unit_pos = unit.get("position")
            summoner_name = summoner.get("name") or summoner.get("unit_type") or "unit"
            self._log(
                f"? {unit_team} {unit_name}#{unit_pos} гибнет вместе с призывателем "
                f"{summoner_name}#{summoner_pos} ({before_hp}>0)."
            )

    def _share_highvampire_leech(
        self, vampire: Dict, hp_pool: int
    ) -> Tuple[int, bool, bool]:
        leftover = int(hp_pool or 0)
        if leftover <= 0:
            return 0, False, False

        allies = [
            ally
            for ally in self.combined
            if ally is not vampire
            and ally.get("team") == vampire.get("team")
            and self._alive(ally)
            and int(ally.get("health", 0) or 0)
            < int(ally.get("max_health", ally.get("health", 0) or 0))
        ]
        if not allies:
            return leftover, False, False

        had_targets = True
        shared_any = False

        while leftover > 0:
            wounded = [
                ally
                for ally in allies
                if self._alive(ally)
                and int(ally.get("health", 0) or 0)
                < int(ally.get("max_health", ally.get("health", 0) or 0))
            ]
            if not wounded:
                break

            share = max(1, leftover // len(wounded))
            distributed_this_round = 0

            for ally in wounded:
                if leftover <= 0:
                    break
                max_hp = int(ally.get("max_health", ally.get("health", 0) or 0))
                before_hp = int(ally.get("health", 0) or 0)
                missing = max_hp - before_hp
                if missing <= 0:
                    continue

                heal = min(share, missing, leftover)
                if heal <= 0:
                    continue

                ally["health"] = before_hp + heal
                leftover -= heal
                distributed_this_round += heal
                shared_any = True
                self._log(
                    f"Делёж крови: {vampire['team'].upper()} {vampire['name']}#{vampire['position']} "
                    f"передаёт {heal} HP для {ally['team'].upper()} {ally['name']}#{ally['position']} ({before_hp}>{ally['health']})."
                )

            if distributed_this_round == 0:
                for ally in wounded:
                    if leftover <= 0:
                        break
                    max_hp = int(ally.get("max_health", ally.get("health", 0) or 0))
                    before_hp = int(ally.get("health", 0) or 0)
                    if before_hp >= max_hp:
                        continue
                    ally["health"] = before_hp + 1
                    leftover -= 1
                    distributed_this_round += 1
                    shared_any = True
                    self._log(
                        f"Делёж крови: {vampire['team'].upper()} {vampire['name']}#{vampire['position']} "
                        f"добавляет 1 HP для {ally['team'].upper()} {ally['name']}#{ally['position']} ({before_hp}>{ally['health']})."
                    )
                if distributed_this_round == 0:
                    break

        return leftover, shared_any, had_targets

    def _apply_vampiric_heal(
        self, attacker: Dict, leeched_total: int, share_leftover: bool = False
    ) -> None:
        leeched = int(leeched_total or 0)
        if leeched <= 0:
            return

        atk_team = (attacker.get("team") or "").upper()
        attacker_name = attacker.get("name") or attacker.get("unit_type") or "unit"
        attacker_pos = attacker.get("position")

        before_hp = int(attacker.get("health", 0) or 0)
        max_hp = int(attacker.get("max_health", before_hp) or before_hp)
        healed = 0
        if before_hp < max_hp:
            healed = min(leeched, max_hp - before_hp)
            if healed > 0:
                attacker["health"] = before_hp + healed
                self._log(
                    f"Кровопийство: {atk_team} {attacker_name}#{attacker_pos} восстанавливает {healed} HP "
                    f"({before_hp}>{attacker['health']}) из {leeched} похищенных."
                )

        leftover = leeched - healed
        if share_leftover and leftover > 0:
            leftover, shared_any, had_targets = self._share_highvampire_leech(
                attacker, leftover
            )
            if not had_targets:
                self._log(
                    f"Делёж крови: {atk_team} {attacker_name}#{attacker_pos} — раненых союзников нет."
                )
            elif leftover > 0 and shared_any:
                self._log(
                    f"Делёж крови: {atk_team} {attacker_name}#{attacker_pos} не смог распределить {leftover} HP — союзники уже восстановлены."
                )
            elif leftover > 0:
                self._log(
                    f"Делёж крови: {atk_team} {attacker_name}#{attacker_pos} не нашёл, кому передать {leftover} HP."
                )

    def _apply_cached_poison(
        self, attacker: Dict, victim: Dict, applied_cache: Dict[int, bool]
    ) -> None:
        if not self._alive(victim):
            return

        acc2 = float(attacker.get("accuracy_secondary", 0) or 0)

        pos = attacker.get("position")
        if applied_cache.get(pos, False):
            return

        roll = self._roll_status(acc2)
        atk_team = (attacker.get("team") or "").upper()
        self._log(
            f"Шанс отравления {int(acc2)}% - "
            + ("успех" if roll else "неудача")
            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
        )
        if not roll:
            return

        if self._is_immune_status(attacker, victim):
            self._log(
                f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — яд не действует на "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return

        status_tag = attacker.get("attack_type_secondary", "") or attacker.get(
            "attack_type_primary", ""
        )
        if self._resilience_blocks(attacker, victim, custom_tag=status_tag):
            return

        turns = self.rng.randint(1, POISON_TURNS)
        victim["poison_turns_left"] = turns
        victim["poison_damage_per_tick"] = int(attacker.get("damage_secondary", 0) or 0)
        applied_cache[pos] = True
        self._log(
            f"? {atk_team} {attacker['name']}#{pos} накладывает яд "
            f"({victim['poison_damage_per_tick']} урона/ход) на "
            f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
        )

    def _apply_travnitsa_buff(self, healer: Dict, recipient: Optional[Dict]) -> bool:
        if recipient is None or not self._alive(recipient) or recipient is healer:
            return False

        current_damage = recipient.get("damage", 0)
        stored_original = recipient.get("original_damage", 0)

        if stored_original <= 0:
            stored_original = current_damage
            if stored_original <= 0:
                return False

        if healer.get("unit_type") == "Travnitsa":
            buffed_damage = int(round(stored_original * 1.25))
            recipient["powerup"] = 1
        elif healer.get("unit_type") == "Novice":
            buffed_damage = int(round(stored_original * 1.5))
            recipient["powerup"] = 1
        elif healer.get("unit_type") == "Alchemist":
            return self._apply_alchemist_support(healer, recipient)
        elif healer.get("unit_type") == "Dwarfdruid":
            buffed_damage = int(round(stored_original * 1.75))
        elif healer.get("unit_type") == "Arhidruid":
            buffed_damage = int(round(stored_original * 2))

        recipient["damage"] = buffed_damage
        recipient["powerup"] = 1

        self._log(
            f"Усиление травницы: {healer['team'].upper()} {healer['name']}#{healer['position']} увеличивает урон "
            f"{recipient['team'].upper()} {recipient['name']}#{recipient['position']} до {buffed_damage}."
        )
        return True

    # ----------- эффекты начала хода (яд/поджог) -----------
    def _apply_start_of_turn_effects(self, unit: Dict) -> bool:
        # DOT-тики и откаты эффектов применяются один раз за раунд: после «ждать»
        # юнит активируется повторно. Раньше от двойного тика защищала проверка
        # initiative > 10 — она молча пропускала тик юнитам с инициативой <= 10
        # (Патриарх и жрецы при броске +0, замедленные Отшельником и т.п.).
        first_activation = not unit.get("round_effects_done", 0)
        unit["round_effects_done"] = 1

        # DEFEND: временная броня спадает только в начале следующего хода этого юнита.
        if unit.get("defense", 0) == 1:
            current_armor = int(unit.get("armor", 0) or 0)
            reduced_armor = max(0, current_armor - DEFEND_ARMOR_BONUS)
            unit["armor"] = reduced_armor
            unit["defense"] = 0
            self._log(
                f"DEFENCE EXPIRE: {unit['team'].upper()} {unit['name']}#{unit['position']} теряет бонус брони "
                f"-{DEFEND_ARMOR_BONUS} ({current_armor}->{unit['armor']})."
            )
        # Переключатель «нет целей — бей как Warrior» касается только
        # НЕскопировавшегося Двойника: копия воина не должна терять свой тип.
        if (
            unit.get("name") == "Двойник"
            and self._alive(unit)
            and not unit.get("doppel_copied", 0)
        ):
            has_targets = self._has_transform_targets()
            if unit.get("unit_type") == "Doppelganger" and not has_targets:
                unit["unit_type"] = "Warrior"
                self._log("Двойник: целей нет — тип переключён на Warrior.")
            elif unit.get("unit_type") == "Warrior" and has_targets:
                unit["unit_type"] = "Doppelganger"
                self._log(
                    "Двойник: цели появились — тип переключён на Doppelganger."
                )

        if unit.get("unit_type") == "Sundancer":
            for ally in self.combined:
                if ally.get("team") != unit.get("team"):
                    continue
                if ally.get("Firedefence", 0) == 1:
                    ally["Firedefence"] = 0
                    if "resistance" in ally:
                        ally["resistance"] = [
                            res for res in ally.get("resistance", []) if res != "Fire"
                        ]
                        if not ally["resistance"]:
                            ally["resistance"] = []
                    if (
                        "resilience_used_types" in ally
                        and ally["resilience_used_types"]
                    ):
                        ally["resilience_used_types"] = [
                            res
                            for res in ally["resilience_used_types"]
                            if res != "Fire"
                        ]
                    self._log(
                        f"? Солнечная защита рассеялась: {ally['team'].upper()} {ally['name']}#{ally['position']} лишается огненной стойкости."
                    )

        if unit.get("unit_type") == "Sylfid":
            for ally in self.combined:
                if ally.get("team") != unit.get("team"):
                    continue
                if ally.get("Airdefence", 0) == 1:
                    ally["Airdefence"] = 0
                    if "resistance" in ally:
                        ally["resistance"] = [
                            res for res in ally.get("resistance", []) if res != "Air"
                        ]
                        if not ally["resistance"]:
                            ally["resistance"] = []
                    if (
                        "resilience_used_types" in ally
                        and ally["resilience_used_types"]
                    ):
                        ally["resilience_used_types"] = [
                            res for res in ally["resilience_used_types"] if res != "Air"
                        ]
                    self._log(
                        f"? Воздушная защита рассеялась: {ally['team'].upper()} {ally['name']}#{ally['position']} лишается воздушной стойкости."
                    )

        if unit.get("unit_type") == "Deva roshi":
            for ally in self.combined:
                if ally.get("team") != unit.get("team"):
                    continue
                reset_map = (
                    ("Firedefence", "Fire"),
                    ("Airdefence", "Air"),
                    ("Waterdefence", "Water"),
                    ("Earthdefence", "Earth"),
                )
                cleared = []
                for field, element in reset_map:
                    if ally.get(field, 0) == 1:
                        ally[field] = 0
                        cleared.append(element)
                if not cleared:
                    continue
                if "resistance" in ally:
                    ally["resistance"] = [
                        res for res in ally.get("resistance", []) if res not in cleared
                    ]
                    if not ally["resistance"]:
                        ally["resistance"] = []
                if "resilience_used_types" in ally and ally["resilience_used_types"]:
                    ally["resilience_used_types"] = [
                        res
                        for res in ally["resilience_used_types"]
                        if res not in cleared
                    ]
                self._log(
                    f"? Защита Дэвы рассеялась: {ally['team'].upper()} {ally['name']}#{ally['position']} теряет стойкость против {', '.join(cleared)}."
                )

        if (
            unit.get("hermited", 0) == 1
            and self._alive(unit)
            and first_activation
        ):
            if self.rng.random() < 0.5:
                old_base = int(unit.get("initiative_base", 0) or 0)
                new_base = old_base * 2
                unit["hermited"] = 0
                unit["initiative_base"] = new_base
                current_ini = int(unit.get("initiative", 0) or 0)
                unit["initiative"] = max(current_ini, new_base)
                self._log(
                    f"Hermit slow fades: {unit['team'].upper()} {unit['name']}#{unit['position']} восстанавливает инициативу "
                    f"{old_base}->{new_base}."
                )

        # ЯД
        if (
            unit.get("poison_turns_left", 0) > 0
            and self._alive(unit)
            and first_activation
        ):
            immunities_lower = [str(i).lower() for i in (unit.get("immunity") or [])]
            if "poison" in immunities_lower:
                unit["poison_turns_left"] = 0
                unit["poison_damage_per_tick"] = 0
                self._log(
                    f"Иммунитет к эффекту 'poison' — яд не действует на {unit['team'].upper()} {unit['name']}#{unit['position']}."
                )
            else:
                tick = int(unit.get("poison_damage_per_tick", 0) or 0)
                if tick > 0:
                    before, after = self._subtract_health(unit, tick)
                    unit["poison_turns_left"] -= 1
                    self._log(
                        f"? Яд поражает {unit['team'].upper()} {unit['name']}#{unit['position']}: "
                        f"{tick} ({before}>{max(0, after)}); осталось ходов: {unit['poison_turns_left']}"
                    )
                    if unit["health"] <= 0:
                        unit["initiative"] = 0
                        self._kill_linked_summons(unit)
                        self._log(
                            f"? {unit['team'].upper()} {unit['name']}#{unit['position']} погибает от яда."
                        )
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["poison_turns_left"] -= 1
                if unit["poison_turns_left"] <= 0:
                    unit["poison_damage_per_tick"] = 0
                    # яд снят — пауки могут снова применить эффект
                    self._spider_applied_poison.clear()
                    self._dregazul_applied_poison.clear()

        # ПОДЖОГ — теперь урон за тик берётся из burn_damage_per_tick (для Владыки = его урон2)
        if (
            unit.get("burn_turns_left", 0) > 0
            and self._alive(unit)
            and first_activation
        ):
            # зеркальная проверка иммунитета к Fire (как для яда)
            immunities_lower = [str(i).lower() for i in (unit.get("immunity") or [])]
            if "fire" in immunities_lower:
                unit["burn_turns_left"] = 0
                unit["burn_damage_per_tick"] = 0
                self._log(
                    f"Иммунитет к эффекту 'Fire' — поджог не действует на {unit['team'].upper()} {unit['name']}#{unit['position']}."
                )
            else:
                tick = int(unit.get("burn_damage_per_tick", 0) or BURN_DAMAGE)
                if tick > 0:
                    before, after = self._subtract_health(unit, tick)
                    unit["burn_turns_left"] -= 1
                    self._log(
                        f"Поджог поражает {unit['team'].upper()} {unit['name']}#{unit['position']}: "
                        f"{tick} ({before}>{max(0, after)}); осталось ходов: {unit['burn_turns_left']}"
                    )
                    if unit["health"] <= 0:
                        unit["initiative"] = 0
                        self._kill_linked_summons(unit)
                        self._log(
                            f"? {unit['team'].upper()} {unit['name']}#{unit['position']} погибает от поджога."
                        )
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["burn_turns_left"] -= 1
                if unit["burn_turns_left"] <= 0:
                    unit["burn_damage_per_tick"] = 0
                    # поджог снят — Владыки могут снова применить эффект
                    self._lord_applied_burn.clear()

        # Вода — наносит периодический урон (для Сына Измира = его урон2)
        if (
            unit.get("uran_turns_left", 0) > 0
            and self._alive(unit)
            and first_activation
        ):
            immunities_lower = [str(i).lower() for i in (unit.get("immunity") or [])]
            if "water" in immunities_lower:
                unit["uran_turns_left"] = 0
                unit["uran_damage_per_tick"] = 0
                self._log(
                    f"Иммунитет к эффекту 'Water' — вода не действует на {unit['team'].upper()} {unit['name']}#{unit['position']}."
                )
            else:
                tick = int(unit.get("uran_damage_per_tick", 0) or URAN_DAMAGE)
                if tick > 0:
                    before, after = self._subtract_health(unit, tick)
                    unit["uran_turns_left"] -= 1
                    self._log(
                        f"? Вода поражает {unit['team'].upper()} {unit['name']}#{unit['position']}: "
                        f"{tick} ({before}>{max(0, after)}); осталось ходов: {unit['uran_turns_left']}"
                    )
                    if unit["health"] <= 0:
                        unit["initiative"] = 0
                        self._kill_linked_summons(unit)
                        self._log(
                            f"? {unit['team'].upper()} {unit['name']}#{unit['position']} погибает от воды."
                        )
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["uran_turns_left"] -= 1
                if unit["uran_turns_left"] <= 0:
                    unit["uran_damage_per_tick"] = 0
                    # вода снята — сын Имира может навесить повторно
                    self._ismir_applied_uran.clear()

        # Runaway check (unit flees before acting if running_away == 1)
        if unit.get("running_away", 0) == 1 and self._alive(unit):
            if unit.get("paralyzed", 0) == 0 and unit.get("long_paralyzed", 0) == 0:
                self._log(
                    f"{unit['team'].upper()} {unit['name']}#{unit['position']} в панике покидает бой."
                )
                self._mark_unit_escaped(unit)
                self._check_victory_after_hit()
                return False
            # если был паралич — снимаем флаг бегства, отложив на следующий ход
            else:
                self._log(
                    f"{unit['team'].upper()} {unit['name']}#{unit['position']} пока парализован и не может бежать."
                )

        basestats = unit.get("basestats")
        if unit.get("transformed", 0) == 1 and basestats:
            pos_key = unit.get("position")
            snapshots: List[Dict] = []
            if isinstance(basestats, dict):
                snapshots = basestats.get(pos_key) or []
            elif isinstance(basestats, list):
                snapshots = basestats

            if snapshots and first_activation:
                chance = unit.get("transform_recover_chance")
                if chance is None:
                    chance = DEFAULT_TRANSFORM_RECOVERY_CHANCE
                try:
                    recover_chance = float(chance)
                except (TypeError, ValueError):
                    recover_chance = DEFAULT_TRANSFORM_RECOVERY_CHANCE
                if self.rng.random() < recover_chance:
                    self._restore_transformed_unit(unit)

        return True

    # ------------------ Боевая логика ------------------

    def _reset_state(self):
        self._patriach_revived_recipients: set[Tuple[str, int]] = set()
        self.combined = deepcopy(UNITS_RED + UNITS_BLUE)
        _apply_transformations_to_units(self.combined)
        for u in self.combined:
            base_ini = int(u.get("initiative_base", 0) or 0)
            u["initiative"] = base_ini
            if self._alive(u):
                u["initiative"] += self.rng.randint(0, 9)
            u.setdefault("attack_type_primary", "Weapon")
            u.setdefault("attack_type_secondary", "")
            u.setdefault("immunity", [])
            u.setdefault("resistance", [])
            u.setdefault("armor", 0)
            # Keep battle-start armor so campaign can restore temporary armor changes.
            u["base_armor"] = int(u.get("armor", 0) or 0)
            u.setdefault("accuracy", 100)
            u.setdefault("accuracy_secondary", 0)
            u.setdefault("damage_secondary", 0)
            u.setdefault("big", False)
            u.setdefault("Level", _resolve_unit_level(u))
            u.setdefault("next_level_exp", _resolve_unit_next_level_exp(u))
            u["hero"] = _resolve_unit_hero_flag(u)
            u["needaunit"] = _resolve_unit_needaunit(u)
            u.setdefault("bonusturn", 0)
            u["defense"] = 0
            u["waited"] = 0
            u["poison_turns_left"] = 0
            u["poison_damage_per_tick"] = 0
            u["burn_turns_left"] = 0
            u["burn_damage_per_tick"] = 0  # per-tick для поджога
            u["uran_turns_left"] = 0
            u["uran_damage_per_tick"] = 0
            u["resilience_used_types"] = []
            u["round_effects_done"] = 0
            u["doppel_copied"] = 0
            u["transformed"] = 0
            u["basestats"] = {}
            u.pop("wight_form_name", None)
            u.pop("transform_effect", None)
            u.pop("transform_recover_chance", None)
            u["_battle_exp_earned"] = 0.0
        self.round_no = 1
        self.winner = None
        self._post_victory_team = None
        self._post_victory_healer_positions = []
        self.current_blue_attacker_pos = None
        self.blue_attacks_left = 0
        self._lord_applied_burn = {}
        self._spider_applied_poison = {}
        self._dregazul_applied_poison = {}
        self._ismir_applied_uran = {}
        self.escaped_units = []
        self.overwritten_exp_kill = {"red": 0.0, "blue": 0.0}
        self._battle_exp_event_count = 0
        self._battle_defeated_exp = {"red": 0.0, "blue": 0.0}
        self.step_count = 0
        self.hero_item_slots_used_this_battle = [False] * len(self.equipped_hero_items)
        if len(getattr(self, "equipped_hero_item_uses_left", []) or []) < len(
            self.equipped_hero_items
        ):
            self.equipped_hero_item_uses_left = [None] * len(self.equipped_hero_items)
        self._log(f"Эпизод начат. Раунд {self.round_no}.")

    def _candidates(self):
        return [u for u in self.combined if self._alive(u) and u["initiative"] > 0]

    def _pop_next(self):
        cand = self._candidates()
        if not cand:
            return None
        self.rng.shuffle(cand)
        cand.sort(key=itemgetter("initiative"), reverse=True)
        return cand[0]

    def _empty_battle_slot(self, team: str, position: int) -> Dict:
        slot = placeholder_unit(team, int(position))
        slot.update(
            {
                "hp": 0,
                "maxhp": 0,
                "base_armor": 0,
                "defense": 0,
                "waited": 0,
                "poison_turns_left": 0,
                "poison_damage_per_tick": 0,
                "burn_turns_left": 0,
                "burn_damage_per_tick": 0,
                "uran_turns_left": 0,
                "uran_damage_per_tick": 0,
                "resilience_used_types": [],
                "bonusturn": 0,
                "original_damage": 0,
                "exp_kill": 0,
                "exp_required": 0,
                "exp_current": 0,
                "next_level_exp": 0,
                "turns_into": [],
                "capital": 0,
                "is_neutral_unit": False,
                "_battle_exp_earned": 0.0,
            }
        )
        return slot

    def _mark_unit_escaped(self, unit: Dict) -> None:
        snapshot = deepcopy(unit)
        snapshot["running_away"] = 0
        self.escaped_units.append(snapshot)

        empty_slot = self._empty_battle_slot(
            str(unit.get("team", "")),
            int(unit.get("position", 0) or 0),
        )
        unit.clear()
        unit.update(empty_slot)

    def _end_round_restore(self):
        for u in self.combined:
            if self._alive(u):
                base_ini = int(u.get("initiative_base", 0) or 0)
                u["initiative"] = base_ini + self.rng.randint(0, 9)
            else:
                u["initiative"] = 0
            u["waited"] = 0
            u["round_effects_done"] = 0
        self._log("Восстановление инициативы. Новый раунд.")

    # --- иммунитеты/броня/точность/стойкость ---
    def _is_immune_damage(self, attacker: Dict, victim: Dict) -> bool:
        atk1 = (attacker.get("attack_type_primary", "Weapon") or "").lower()
        immunities = [str(i).lower() for i in (victim.get("immunity") or [])]
        return atk1 in immunities

    def _is_immune_status(self, attacker: Dict, victim: Dict) -> bool:
        atk2 = (attacker.get("attack_type_secondary", "") or "").lower()
        if not atk2:
            return False
        immunities = [str(i).lower() for i in (victim.get("immunity") or [])]
        return atk2 in immunities

    def _resilience_blocks(
        self, attacker: Dict, victim: Dict, custom_tag: Optional[str] = None
    ) -> bool:
        res_list_raw = victim.get("resistance") or []
        if not res_list_raw:
            return False
        # Создаём словарь: lower -> original для сохранения оригинального регистра при записи
        res_map = {str(r).lower(): r for r in res_list_raw}
        used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
        available_lower = set(res_map.keys()) - used_lower
        if not available_lower:
            return False

        atk1 = attacker.get("attack_type_primary", "")

        attack_tags = []
        if custom_tag:
            attack_tags.append(custom_tag)
        elif atk1:
            attack_tags.append(atk1)

        for tag in attack_tags:
            tag_lower = (tag or "").lower()
            if tag_lower and tag_lower in available_lower:
                # Сохраняем оригинальное значение для consistency
                original_tag = res_map.get(tag_lower, tag)
                victim.setdefault("resilience_used_types", []).append(original_tag)
                self._log(
                    f"Стойкость — первый удар типа '{tag}' по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} поглощён."
                )
                return True
        return False

    def _reset_powerup(self, unit: Optional[Dict]) -> None:
        if unit is None:
            return
        if unit.get("powerup", 0) != 0:
            unit["powerup"] = 0
            if unit.get("original_damage", 0) > 0:
                unit["damage"] = unit["original_damage"]

    def _apply_damage_with_armor(self, attacker: Dict, base_dmg: float, victim: Dict) -> int:
        attacker_type = str(attacker.get("unit_type", "") or "")
        if attacker_type in ZERO_DAMAGE_NO_HP_UNIT_TYPES:
            return 0

        raw_damage = float(base_dmg or 0)
        if raw_damage <= 0:
            return 0

        armor = float(victim.get("armor", 0) or 0)
        armor = min(armor, 90)  # Максимум 90% снижения урона
        factor = (1.0 - armor / 100.0) if armor > 0 else 1.0
        bonus = self.rng.randint(0, 5)
        boosted = raw_damage + bonus
        eff = boosted * max(0.0, factor)
        incoming_multiplier = float(victim.get("incoming_damage_multiplier", 1.0) or 1.0)
        eff *= max(0.0, incoming_multiplier)
        return int(round(eff))

    def _subtract_health(self, unit: Dict, damage: float) -> Tuple[float, float]:
        """Apply damage without ever leaving a unit with negative health."""
        before = unit.get("health", 0) or 0
        after = max(0, before - damage)
        unit["health"] = after
        if "hp" in unit:
            unit["hp"] = after
        if before > 0 and after <= 0:
            self._record_unit_defeat_for_exp(unit)
        return before, after

    def _roll_hit(self, attacker: Dict) -> bool:
        acc = float(attacker.get("accuracy", 100) or 0)
        return self.rng.random() < (acc / 100.0)

    def _roll_status(self, chance_percent: float) -> bool:
        cp = max(0.0, min(100.0, float(chance_percent or 0.0)))
        return self.rng.random() < (cp / 100.0)

    def _apply_fear_effect(self, attacker: Dict, victim: Dict) -> bool:
        if victim.get("running_away", 0) == 1:
            return False
        atk_type = attacker.get("attack_type_primary", "")
        immun = set(victim.get("immunity") or [])
        resist = set(victim.get("resistance") or [])
        if atk_type and atk_type in immun:
            self._log(
                f"Страх Баронессы не действует на {victim['team'].upper()} {victim['name']}#{victim['position']} — тип атаки '{atk_type}' поглощён."
            )
            return False
        if atk_type and atk_type in resist:
            used = set(victim.get("resilience_used_types") or [])
            if atk_type not in used:
                victim.setdefault("resilience_used_types", []).append(atk_type)
                self._log(
                    f"Страх Баронессы не действует на {victim['team'].upper()} {victim['name']}#{victim['position']} — тип атаки '{atk_type}' поглощён."
                )
                return False
        victim["running_away"] = 1
        self._log(
            f"{attacker['team'].upper()} {attacker['name']}#{attacker['position']} вселяет страх в "
            f"{victim['team'].upper()} {victim['name']}#{victim['position']}: бегство!"
        )
        return True

    def _apply_paralysis_effect(self, attacker: Dict, victim: Dict) -> bool:
        if victim.get("paralyzed", 0):
            return False

        effect_type = attacker.get("attack_type_primary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        if effect_type_lower in immunities_lower:
            self._log(
                f"Иммунитет к '{effect_type}' — паралич не действует на "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
            return False

        res_lower = {str(r).lower() for r in (victim.get("resistance") or [])}
        if effect_type_lower in res_lower:
            used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
            if effect_type_lower not in used_lower:
                victim.setdefault("resilience_used_types", []).append(effect_type)
                self._log(
                    f"Стойкость — первый эффект типа '{effect_type}' по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} поглощён."
                )
                return False

        victim["paralyzed"] = 1
        self._log(
            f"Паралич: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
            f"лишает хода {victim['team'].upper()} {victim['name']}#{victim['position']}"
        )
        return True

    def _apply_witch_effect(self, attacker: Dict, victim: Dict) -> None:
        was_transformed = bool(victim.get("transformed", 0))
        snapshot = self._ensure_transform_snapshot(victim)
        if not was_transformed:
            # Restore normal turn priority even if the unit had already acted and
            # therefore had initiative=0 at the moment of transformation.
            snapshot["initiative"] = deepcopy(
                victim.get("initiative_base", victim.get("initiative", 0))
            )
        is_big = bool(victim.get("big", False))
        old_initiative = victim.get("initiative", 0)
        form_name = "Толстый бес" if is_big else "Бес"
        form_template = BATTLE_TEMPLATE_BY_NAME[form_name]
        new_initiative = int(form_template["initiative_base"])

        victim["accuracy"] = int(form_template["accuracy"])
        victim["accuracy_secondary"] = int(form_template["accuracy_secondary"])
        victim["damage"] = int(form_template["damage"])
        victim["damage_secondary"] = int(form_template["damage_secondary"])
        victim["attack_type_primary"] = str(form_template["attack_type_primary"])
        victim["attack_type_secondary"] = str(form_template["attack_type_secondary"])
        victim["initiative_base"] = new_initiative
        if old_initiative == 0:
            victim["initiative"] = old_initiative
        else:
            victim["initiative"] = new_initiative
        victim["unit_type"] = str(form_template["unit_type"])
        victim["armor"] = int(form_template["armor"])
        victim["immunity"] = list(form_template["immunity"])
        victim["resistance"] = list(form_template["resistance"])
        victim["resilience_used_types"] = []
        victim["transformed"] = 1
        self._log(
            f"Ведьма меняет характеристики {victim['team'].upper()} {victim['name']}#{victim['position']}: "
            f"form={form_name}, accuracy={victim['accuracy']}, damage={victim['damage']}, "
            f"initiative={victim['initiative']}, armor={victim['armor']}"
        )

    def _apply_wight_decay_effect(self, attacker: Dict, victim: Dict) -> bool:
        if _is_neutral_battle_unit(victim):
            self._log(
                f"? Wight drain: {victim['team'].upper()} {victim['name']}#{victim['position']} "
                "is neutral and cannot be lowered."
            )
            return False

        current_form = str(
            victim.get("wight_form_name") or victim.get("name", "") or ""
        ).strip()
        lower_form = SOURCE_BY_NAME.get(current_form, "")
        if not lower_form:
            self._log(
                f"? Wight drain: {victim['team'].upper()} {victim['name']}#{victim['position']} "
                f"has no lower form from '{current_form}'."
            )
            return False

        lower_template = BATTLE_TEMPLATE_BY_NAME.get(lower_form)
        if not isinstance(lower_template, dict):
            self._log(
                f"? Wight drain: lower form '{lower_form}' is missing from unit data."
            )
            return False

        self._ensure_transform_snapshot(victim, include_health_fields=True)

        old_hp = victim.get("health", 0)
        old_max = victim.get("max_health", old_hp)
        old_damage = victim.get("damage", 0)
        old_unit_type = victim.get("unit_type", "")
        old_initiative = victim.get("initiative", 0)
        new_max = int(lower_template.get("max_health", 0) or 0)
        new_health = self._scale_health_to_new_max(old_hp, old_max, new_max)

        victim["wight_form_name"] = lower_template.get("form_name", lower_form)
        victim["accuracy"] = int(lower_template.get("accuracy", 0) or 0)
        victim["accuracy_secondary"] = int(
            lower_template.get("accuracy_secondary", 0) or 0
        )
        victim["damage"] = int(lower_template.get("damage", 0) or 0)
        victim["damage_secondary"] = int(
            lower_template.get("damage_secondary", 0) or 0
        )
        victim["original_damage"] = victim["damage"]
        victim["attack_type_primary"] = lower_template.get(
            "attack_type_primary", "Weapon"
        )
        victim["attack_type_secondary"] = lower_template.get(
            "attack_type_secondary", ""
        )
        victim["initiative_base"] = int(lower_template.get("initiative_base", 0) or 0)
        victim["initiative"] = old_initiative if old_initiative == 0 else victim["initiative_base"]
        victim["unit_type"] = lower_template.get("unit_type", victim.get("unit_type", ""))
        victim["stand"] = lower_template.get("stand", victim.get("stand", "ahead"))
        victim["armor"] = int(lower_template.get("armor", 0) or 0)
        victim["immunity"] = list(lower_template.get("immunity", []))
        victim["resistance"] = list(lower_template.get("resistance", []))
        victim["resilience_used_types"] = []
        victim["big"] = bool(lower_template.get("big", False))
        victim["Level"] = int(lower_template.get("Level", victim.get("Level", 0)) or 0)
        victim["max_health"] = new_max
        victim["health"] = new_health
        if "maxhp" in victim:
            victim["maxhp"] = new_max
        if "hp" in victim:
            victim["hp"] = new_health
        victim["transformed"] = 1
        victim["transform_effect"] = "wight_level_down"
        victim["transform_recover_chance"] = WIGHT_LEVEL_DOWN_RECOVERY_CHANCE

        self._log(
            f"? Wight drain: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
            f"lowers {victim['team'].upper()} {victim['name']}#{victim['position']} "
            f"from {current_form} to {victim['wight_form_name']}: "
            f"type {old_unit_type}->{victim['unit_type']}, damage {old_damage}->{victim['damage']}, "
            f"health {old_hp}/{old_max}->{victim['health']}/{victim['max_health']}."
        )
        return True

    def _apply_long_paralysis_effect(self, attacker: Dict, victim: Dict) -> bool:
        if victim.get("long_paralyzed", 0):
            return False

        effect_type = attacker.get("attack_type_secondary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        if effect_type_lower in immunities_lower:
            self._log(
                f"Иммунитет к '{effect_type}' — долгий паралич не действует на "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
            return False

        res_map = {str(r).lower(): r for r in (victim.get("resistance") or [])}
        if effect_type_lower in res_map:
            used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
            if effect_type_lower not in used_lower:
                victim.setdefault("resilience_used_types", []).append(res_map[effect_type_lower])
                self._log(
                    f"Стойкость — первый эффект типа '{effect_type}' по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} поглощён."
                )
                return False
        if attacker.get("unit_type") == "Abyss Devil":
            victim["paralyzed"] = 1
            self._log(
                f"? Окаменение: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
                f"накладывает окаменение на {victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
        else:
            victim["long_paralyzed"] = 1
            self._log(
                f"? Долгий паралич: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
                f"накладывает долгий паралич на {victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
        return True

    def _active_hero_combat_artifacts(self, attacker: Dict) -> List[str]:
        if not bool(attacker.get("hero", False)):
            return []

        active_artifacts: List[str] = []
        seen = set()
        for raw_name in list(attacker.get("campaign_active_artifacts") or []):
            item_name = str(raw_name or "")
            if item_name not in HERO_COMBAT_ARTIFACT_ITEMS or item_name in seen:
                continue
            active_artifacts.append(item_name)
            seen.add(item_name)
        return active_artifacts

    def _artifact_attacker(
        self,
        attacker: Dict,
        *,
        effect_type: str,
        primary: bool = False,
        unit_type: Optional[str] = None,
    ) -> Dict:
        artifact_attacker = dict(attacker)
        if primary:
            artifact_attacker["attack_type_primary"] = effect_type
        artifact_attacker["attack_type_secondary"] = effect_type
        if unit_type:
            artifact_attacker["unit_type"] = unit_type
        return artifact_attacker

    def _roll_hero_artifact_status(
        self, attacker: Dict, victim: Dict, artifact_name: str, chance: float
    ) -> bool:
        roll = self._roll_status(float(chance or 0))
        self._log(
            f"Artifact {artifact_name}: status chance {int(chance)}% - "
            + ("success" if roll else "fail")
            + f" vs {victim['team'].upper()} {victim['name']}#{victim['position']}."
        )
        return roll

    def _apply_hero_artifact_life_drain(
        self, attacker: Dict, inflicted_damage: int
    ) -> None:
        if UNHOLY_DAGGER_ARTIFACT_ITEM_NAME not in self._active_hero_combat_artifacts(
            attacker
        ):
            return

        inflicted = int(inflicted_damage or 0)
        if inflicted <= 0:
            return

        leeched_total = int(inflicted * UNHOLY_DAGGER_DRAIN_PERCENT // 100)
        if leeched_total <= 0:
            return

        self._log(
            f"Artifact {UNHOLY_DAGGER_ARTIFACT_ITEM_NAME}: drains {leeched_total} HP "
            f"from {inflicted} inflicted damage."
        )
        self._apply_vampiric_heal(attacker, leeched_total, share_leftover=False)

    def _apply_hero_artifact_poison(
        self, attacker: Dict, victim: Dict, artifact_name: str, damage_per_tick: int
    ) -> bool:
        if not self._alive(victim):
            return False

        chance = HERO_ARTIFACT_STATUS_CHANCES.get(artifact_name, 0)
        if not self._roll_hero_artifact_status(attacker, victim, artifact_name, chance):
            return False

        artifact_attacker = self._artifact_attacker(
            attacker, effect_type="Death", primary=False
        )
        if self._is_immune_status(artifact_attacker, victim):
            self._log(
                f"Artifact {artifact_name}: immunity to Death blocks poison on "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return False

        if self._resilience_blocks(artifact_attacker, victim, custom_tag="Death"):
            return False

        existing_turns = int(victim.get("poison_turns_left", 0) or 0)
        existing_damage = int(victim.get("poison_damage_per_tick", 0) or 0)
        if existing_turns > 0 and existing_damage >= int(damage_per_tick):
            self._log(
                f"Artifact {artifact_name}: existing poison "
                f"({existing_damage} damage/turn) is not weaker."
            )
            return False

        turns = self.rng.randint(1, POISON_TURNS)
        victim["poison_turns_left"] = turns
        victim["poison_damage_per_tick"] = int(damage_per_tick)
        self._log(
            f"Artifact {artifact_name}: applies poison "
            f"({int(damage_per_tick)} damage/turn) to "
            f"{victim['team'].upper()} {victim['name']}#{victim['position']} "
            f"for {turns} turns."
        )
        return True

    def _apply_hero_artifact_transform(
        self, attacker: Dict, victim: Dict, artifact_name: str
    ) -> bool:
        if not self._alive(victim):
            return False

        chance = HERO_ARTIFACT_STATUS_CHANCES.get(artifact_name, 0)
        if not self._roll_hero_artifact_status(attacker, victim, artifact_name, chance):
            return False

        artifact_attacker = self._artifact_attacker(
            attacker, effect_type="Mind", primary=False
        )
        if self._is_immune_status(artifact_attacker, victim):
            self._log(
                f"Artifact {artifact_name}: immunity to Mind blocks transform on "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return False

        forbiddenwitch_names = {
            "\u00c0\u00f8\u00e3\u00e0\u00ed",
            "\u00c0\u00f8\u00ea\u00e0\u00fd\u00eb\u00fc",
            "\u00c2\u00e8\u00e4\u00e0\u00f0",
            "\u00cc\u00e8\u00e7\u00f0\u00e0\u00fd\u00eb\u00fc",
            "\u00c8\u00eb\u00eb\u00fe\u00ec\u00e8\u00fd\u00eb\u00eb\u00fc",
        }
        if victim.get("name") in forbiddenwitch_names:
            self._log("Artifact Hag's Ring: capital guard cannot be transformed.")
            return False

        if self._resilience_blocks(artifact_attacker, victim, custom_tag="Mind"):
            return False

        self._apply_witch_effect(artifact_attacker, victim)
        return True

    def _apply_hero_artifact_post_hit_statuses(
        self, attacker: Dict, victim: Dict
    ) -> None:
        if not self._alive(victim):
            return

        for artifact_name in self._active_hero_combat_artifacts(attacker):
            if not self._alive(victim):
                return
            if artifact_name == UNHOLY_DAGGER_ARTIFACT_ITEM_NAME:
                continue

            chance = HERO_ARTIFACT_STATUS_CHANCES.get(artifact_name, 0)
            if artifact_name == SOUL_CRYSTAL_ARTIFACT_ITEM_NAME:
                if self._roll_hero_artifact_status(
                    attacker, victim, artifact_name, chance
                ):
                    artifact_attacker = self._artifact_attacker(
                        attacker, effect_type="Mind", primary=True
                    )
                    self._apply_paralysis_effect(artifact_attacker, victim)
            elif artifact_name == HORN_OF_INCUBUS_ARTIFACT_ITEM_NAME:
                if self._roll_hero_artifact_status(
                    attacker, victim, artifact_name, chance
                ):
                    artifact_attacker = self._artifact_attacker(
                        attacker,
                        effect_type="Earth",
                        primary=False,
                        unit_type="Abyss Devil",
                    )
                    self._apply_long_paralysis_effect(artifact_attacker, victim)
            elif artifact_name in HERO_ARTIFACT_POISON_DAMAGE:
                self._apply_hero_artifact_poison(
                    attacker,
                    victim,
                    artifact_name,
                    HERO_ARTIFACT_POISON_DAMAGE[artifact_name],
                )
            elif artifact_name == HAGS_RING_ARTIFACT_ITEM_NAME:
                self._apply_hero_artifact_transform(attacker, victim, artifact_name)

    def _apply_tiamat_damage_debuff(self, attacker: Dict, victim: Dict) -> bool:
        effect_type = attacker.get("attack_type_secondary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        if effect_type_lower and effect_type_lower in immunities_lower:
            self._log(
                f"IMMUNE: effect '{effect_type}' blocks Tiamat debuff on "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return False

        if effect_type_lower:
            res_map = {str(r).lower(): r for r in (victim.get("resistance") or [])}
            if effect_type_lower in res_map:
                used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
                if effect_type_lower not in used_lower:
                    victim.setdefault("resilience_used_types", []).append(res_map[effect_type_lower])
                    self._log(
                        f"RESIST: '{effect_type}' blocks Tiamat debuff on "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    return False

        if victim.get("teamated", 0) == 1:
            return False

        victim["teamated"] = 1
        base_primary = int(victim.get("damage", 0) or 0)
        new_primary = max(0, int(round(base_primary * 0.68)))

        if new_primary == base_primary:
            self._log(
                f"Tiamat debuff: damage of {victim['team'].upper()} {victim['name']}#{victim['position']} stays {base_primary}."
            )
            return True

        victim["damage"] = new_primary
        self._log(
            f"Tiamat debuff: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} lowers damage of "
            f"{victim['team'].upper()} {victim['name']}#{victim['position']} from {base_primary} to {new_primary}."
        )

        return True

    def _apply_hermit_initiative_slow(self, attacker: Dict, victim: Dict) -> bool:
        """
        Разово режет инициативу цели на 50% (и базовую, и текущую), не стакается.
        Защита/стойкость проверяются по attack_type_secondary атакующего (как у Tiamat).
        """
        effect_type = attacker.get("attack_type_secondary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        # Иммунитет к типу статуса
        if effect_type_lower and effect_type_lower in immunities_lower:
            self._log(
                f"IMMUNE: effect '{effect_type}' blocks Hermit slow on "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return False

        # Первая стойкость к типу статуса
        if effect_type_lower:
            res_map = {str(r).lower(): r for r in (victim.get("resistance") or [])}
            if effect_type_lower in res_map:
                used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
                if effect_type_lower not in used_lower:
                    victim.setdefault("resilience_used_types", []).append(res_map[effect_type_lower])
                    self._log(
                        f"RESIST: '{effect_type}' blocks Hermit slow on "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    return False

        # Нестакающийся эффект
        if victim.get("hermited", 0) == 1:
            return False

        victim["hermited"] = 1

        base_ini = int(victim.get("initiative_base", 0) or 0)
        cur_ini = int(victim.get("initiative", 0) or 0)

        new_base = max(0, int(round(base_ini * 0.5)))
        new_cur = min(cur_ini, new_base)

        victim["initiative_base"] = new_base
        victim["initiative"] = new_cur

        self._log(
            f"Hermit slow: lowers initiative of {victim['team'].upper()} {victim['name']}#{victim['position']} "
            f"from base {base_ini} to {new_base} (current {cur_ini}->{new_cur})."
        )
        return True

    def _apply_teurg_armor_shred(self, attacker: Dict, victim: Dict) -> bool:
        forbiddenteurg_names = {"Ашган", "Ашкаэль", "Видар", "Мизраэль", "Иллюмиэлль"}

        if victim.get("name") in forbiddenteurg_names:
            self._log("Это страж столицы")
            return False

        effect_type = attacker.get("attack_type_secondary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        if effect_type_lower and effect_type_lower in immunities_lower:
            self._log(
                f"IMMUNE: effect '{effect_type}' keeps armor intact on "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return False

        if effect_type_lower:
            res_map = {str(r).lower(): r for r in (victim.get("resistance") or [])}
            if effect_type_lower in res_map:
                used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
                if effect_type_lower not in used_lower:
                    victim.setdefault("resilience_used_types", []).append(res_map[effect_type_lower])
                    self._log(
                        f"RESIST: '{effect_type}' keeps armor intact on "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    return False

        current_armor = int(victim.get("armor", 0) or 0)
        new_armor = max(0, current_armor - 15)
        victim["armor"] = new_armor
        if new_armor == current_armor:
            self._log(
                f"Teurg armor shred: armor of {victim['team'].upper()} {victim['name']}#{victim['position']} already 0."
            )
        else:
            self._log(
                f"Teurg armor shred: armor of {victim['team'].upper()} {victim['name']}#{victim['position']} {current_armor}->{new_armor}."
            )
        return True

    def _attack(self, attacker, target_pos: Optional[int]):
        """
        Выполнить действие юнита attacker по типам:
        - Shadow / Succub / Incub: AOE-статусы без урона.
        - Cliric / Deva roshi / Patriach (живые цели): точечное лечение союзника (через противоположную клетку).
        - Patriach (мертвые цели): воскрешает союзника до 50% от максимального здоровья.
        - Travnitsa / Novice / Dwarfdruid / Arhidruid: точечный бафф союзника (через противоположную клетку).
        - Alchemist: доп. ход союзнику (через противоположную клетку).
        - Profit / Sundancer / Sylfid: массовое лечение (всем союзникам).
        - Mage / Dead dragon / Gumtic / Uter Demon / Tiamat / Teurg / Hermit / Vampire / Highvampire: AOE-урон + спецэффекты.
        * Vampire суммирует реальный нанесённый урон и лечится на эту сумму (не выше max_health).
        * Highvampire делится излишком: сперва лечит себя, затем равномерно распределяет остаток между ранеными союзниками.
        * Bone Lord атакует одиночную цель как Warrior, затем высасывает до половины нанесённого урона: сперва лечит себя, остаток раздаёт союзникам как Highvampire.
        * Dregazul бьёт как Bone Lord, но лечит только себя (как Vampire) и после высасывания крови пытается отравить цель, как Spider.
        - summoner: призывает союзного юнита на свободную клетку своей стороны (см. правила ниже).
        - Occultmaster: логирует позиции союзных юнитов без имени либо с `health == 0`, а также состояние своих рядов; при пустом ряду с шансом 50% призывает дракона смерти или драколича, при непустом — закрывает свободные слоты случайным существом (Вампир/Дух/Лич/Тень/Сущий, по 20%).
        - Остальные: одиночная атака по target_pos (+ их пост-эффекты).

        Возврат: (hit: bool, reason: str)
        reason ? {"ok","dead_or_absent","aoe","aoe_no_targets","immune_damage","miss","resilience_block"}
        """
        unit_type = attacker.get("unit_type")
        atk_team = attacker.get("team")
        atk_name = attacker.get("name")
        enemy_team = "blue" if atk_team == "red" else "red"
        units_by_pos = self._units_by_position()

        def _unit_at(pos: Optional[int]) -> Optional[Dict]:
            if pos is None:
                return None
            return units_by_pos.get(pos)

        # ---------- вспомогательные ----------
        def _live_enemies():
            return [
                u for u in self.combined if u["team"] == enemy_team and self._alive(u)
            ]

        def _aoe_status_all(action_desc: str, log_type: str, apply_effect):
            targets = _live_enemies()
            if not targets:
                self._log(
                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} ({log_type}) пытается {action_desc}: целей нет."
                )
                return False, "aoe_no_targets"
            self._log(
                f"{atk_team.upper()} {attacker['name']}#{attacker['position']} ({log_type}) пытается {action_desc} всю команду противника."
            )
            for victim in targets:
                if not self._roll_hit(attacker):
                    self._log(
                        f"Промах: {atk_team.upper()} {attacker['name']}#{attacker['position']} по "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    continue
                apply_effect(victim)
            return True, "aoe"

        def _aoe_damage_all():
            targets = _live_enemies()
            if not targets:
                self._log(
                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} ({unit_type}) применяет массовую атаку: целей нет."
                )
                return False, "aoe_no_targets"

            self._log(
                f"{atk_team.upper()} {attacker['name']}#{attacker['position']} ({unit_type}) применяет массовую атаку."
            )
            vamp_total = 0  # суммарный реальный урон для Vampire/Highvampire

            for victim in targets:
                if not self._roll_hit(attacker):
                    self._log(
                        f"Промах: {atk_team.upper()} {attacker['name']}#{attacker['position']} по "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    continue
                if self._is_immune_damage(attacker, victim):
                    self._log(
                        f"Иммунитет к урону '{attacker.get('attack_type_primary', '')}' — "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}: урон 0."
                    )
                    continue
                if self._resilience_blocks(attacker, victim):
                    continue

                dmg = self._apply_damage_with_armor(attacker, attacker["damage"], victim)
                before, after = self._subtract_health(victim, dmg)
                self._log(
                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} > "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']}: {dmg} "
                    f"({before}>{max(0, after)})"
                )

                # Суммируем фактический урон для вампиризма (без оверкилла)
                if unit_type in VAMPIRE_AOE_TYPES and dmg > 0:
                    inflicted = min(dmg, max(0, before))
                    vamp_total += max(0, inflicted)

                if victim["health"] <= 0:
                    victim["initiative"] = 0
                    self._kill_linked_summons(victim)
                    self._log(
                        f"? {victim['team'].upper()} {victim['name']}#{victim['position']} выведен из строя."
                    )
                    continue

                # --- тип-специфичные AOE эффекты ---
                if unit_type == "Teurg" and dmg > 0:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс снижения брони {int(acc2)}% - "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        self._apply_teurg_armor_shred(attacker, victim)

                if (
                    unit_type == "Dead dragon"
                    and victim.get("poison_turns_left", 0) <= 0
                ):
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс отравления {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — яд НЕ накладывается "
                                f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                        else:
                            status_tag = attacker.get(
                                "attack_type_secondary", ""
                            ) or attacker.get("attack_type_primary", "")
                            if not self._resilience_blocks(
                                attacker, victim, custom_tag=status_tag
                            ):
                                turns = self.rng.randint(1, POISON_TURNS)
                                victim["poison_turns_left"] = turns
                                victim["poison_damage_per_tick"] = int(
                                    attacker.get("damage_secondary", 0) or 0
                                )
                                self._log(
                                    f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} накладывает яд "
                                    f"({victim['poison_damage_per_tick']} урона/ход) на "
                                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                )

                if unit_type == "Gumtic" and victim.get("burn_turns_left", 0) <= 0:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс статуса {int(acc2)}% — "
                        + ("успешно:" if roll else "неудачно")
                        + f" против {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' у цели "
                                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                        else:
                            status_tag = attacker.get(
                                "attack_type_secondary", ""
                            ) or attacker.get("attack_type_primary", "")
                            if not self._resilience_blocks(
                                attacker, victim, custom_tag=status_tag
                            ):
                                turns = self.rng.randint(1, BURN_TURNS)
                                victim["burn_turns_left"] = turns
                                victim["burn_damage_per_tick"] = int(
                                    attacker.get("damage_secondary", 0) or 0
                                )
                                self._log(
                                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} поджигает цель "
                                    f"({victim['burn_damage_per_tick']} урона/ход) "
                                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                )

                if unit_type == "Drulliaan" and victim.get("uran_turns_left", 0) <= 0:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс водного проклятия {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — вода НЕ накладывается "
                                f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                        else:
                            status_tag = attacker.get(
                                "attack_type_secondary", ""
                            ) or attacker.get("attack_type_primary", "")
                            if not self._resilience_blocks(
                                attacker, victim, custom_tag=status_tag
                            ):
                                turns = self.rng.randint(1, URAN_TURNS)
                                victim["uran_turns_left"] = turns
                                victim["uran_damage_per_tick"] = int(
                                    attacker.get("damage_secondary", 0) or 0
                                )
                                self._log(
                                    f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} накладывает воду "
                                    f"({victim['uran_damage_per_tick']} урона/ход) на "
                                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                )

                if unit_type == "Hermit":
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс замедления {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        self._apply_hermit_initiative_slow(attacker, victim)

                if unit_type == "Uter Demon":
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    if acc2 > 0:
                        roll = self._roll_status(acc2)
                        self._log(
                            f"? Шанс долгого паралича {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            self._apply_long_paralysis_effect(attacker, victim)

                if unit_type == "Shamanka":
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    if acc2 > 0:
                        roll = self._roll_status(acc2)
                        self._log(
                            f"? Шанc напугать {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            self._apply_fear_effect(attacker, victim)

                if unit_type == "Tiamat":
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс ослабления урона {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        self._apply_tiamat_damage_debuff(attacker, victim)

            # После цикла — вампиризм
            if unit_type in VAMPIRE_AOE_TYPES and vamp_total > 0:
                leeched_total = vamp_total // 2
                self._apply_vampiric_heal(
                    attacker, leeched_total, share_leftover=(unit_type == "Highvampire")
                )

            return True, "aoe"

        def _single_target():
            victim = _unit_at(target_pos)
            if victim is None or not self._alive(victim):
                self._log(
                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} бьёт pos{target_pos}: цели нет/мертва/недоступна."
                )
                return False, "dead_or_absent"

            if not self._roll_hit(attacker):
                self._log(
                    f"Промах: {atk_team.upper()} {attacker['name']}#{attacker['position']} по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                )
                return True, "miss"

            if self._is_immune_damage(attacker, victim):
                self._log(
                    f"Иммунитет к урону '{attacker.get('attack_type_primary', '')}' — "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']}: атака наносит 0."
                )
                return True, "immune_damage"

            if self._resilience_blocks(attacker, victim):
                return True, "resilience_block"

            dmg = self._apply_damage_with_armor(attacker, attacker["damage"], victim)
            before, after = self._subtract_health(victim, dmg)
            self._log(
                f"{atk_team.upper()} {attacker['name']}#{attacker['position']} > "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}: {dmg} "
                f"({before}>{max(0, after)})"
            )

            inflicted = min(dmg, max(0, before))
            self._apply_hero_artifact_life_drain(attacker, inflicted)

            if unit_type == "Bone Lord":
                if inflicted > 0:
                    leeched_total = int(inflicted) // 2
                    self._apply_vampiric_heal(
                        attacker, leeched_total, share_leftover=True
                    )

            if unit_type == "Dregazul":
                if inflicted > 0:
                    leeched_total = int(inflicted) // 2
                    self._apply_vampiric_heal(
                        attacker, leeched_total, share_leftover=False
                    )
                    self._apply_cached_poison(
                        attacker, victim, self._dregazul_applied_poison
                    )

            if self._alive(victim):
                if unit_type == "Ghost":
                    self._apply_paralysis_effect(attacker, victim)

                if atk_name == "Ниддог":
                    self._apply_cached_poison(
                        attacker, victim, self._spider_applied_poison
                    )

                if unit_type == "Baroness":
                    self._apply_fear_effect(attacker, victim)
                    self._log("Baroness inspires fear.")

                if unit_type == "Aleman" and dmg > 0:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс снижения брони {int(acc2)}% - "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        self._apply_teurg_armor_shred(attacker, victim)

                if unit_type == "Witch":
                    forbiddenwitch_names = {
                        "Ашган",
                        "Ашкаэль",
                        "Видар",
                        "Мизраэль",
                        "Иллюмиэлль",
                    }
                    # Иммунитет/стойкость к «превращению»
                    if self._is_immune_status(attacker, victim):
                        self._log(
                            f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — превращение НЕ накладывается "
                            f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                    elif victim.get("name") in forbiddenwitch_names:
                        self._log("Это страж столицы")
                    else:
                        status_tag = attacker.get(
                            "attack_type_secondary", ""
                        ) or attacker.get("attack_type_primary", "")
                        if not self._resilience_blocks(
                            attacker, victim, custom_tag=status_tag
                        ):
                            self._apply_witch_effect(attacker, victim)

                if unit_type in LONG_PARALYSIS_TYPES:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    if acc2 > 0:
                        roll = self._roll_status(acc2)
                        self._log(
                            f"? Шанс долгого паралича {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            self._apply_long_paralysis_effect(attacker, victim)

                if unit_type in (
                    "Death",
                    "Wight",
                    "Sentry",
                    "Watcher",
                    "Lord",
                    "Ismir son",
                    "Centaur Savage",
                    "Spider",
                ):
                    wight_neutral_target = (
                        unit_type == "Wight" and _is_neutral_battle_unit(victim)
                    )
                    acc2 = (
                        0.0
                        if wight_neutral_target
                        else float(attacker.get("accuracy_secondary", 0) or 0)
                    )
                    roll = False if wight_neutral_target else self._roll_status(acc2)

                    if unit_type == "Death" and victim.get("poison_turns_left", 0) <= 0:
                        self._log(
                            f"Шанс отравления {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            if self._is_immune_status(attacker, victim):
                                self._log(
                                    f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — яд НЕ накладывается "
                                    f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                                )
                            else:
                                status_tag = attacker.get(
                                    "attack_type_secondary", ""
                                ) or attacker.get("attack_type_primary", "")
                                if not self._resilience_blocks(
                                    attacker, victim, custom_tag=status_tag
                                ):
                                    turns = self.rng.randint(1, POISON_TURNS)
                                    victim["poison_turns_left"] = turns
                                    victim["poison_damage_per_tick"] = int(
                                        attacker.get("damage_secondary", 0) or 0
                                    )
                                    self._log(
                                        f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} накладывает яд "
                                        f"({victim['poison_damage_per_tick']} урона/ход) на "
                                        f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                    )

                    elif unit_type == "Wight":
                        if wight_neutral_target:
                            self._log(
                                f"? Wight drain skipped: {victim['team'].upper()} {victim['name']}#{victim['position']} "
                                "is neutral."
                            )
                        else:
                            self._log(
                                f"Wight drain chance {int(acc2)}% - "
                                + ("success" if roll else "fail")
                                + f" vs {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                            if roll:
                                if self._is_immune_status(attacker, victim):
                                    self._log(
                                        f"Immunity blocks '{attacker.get('attack_type_secondary', '')}' - "
                                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                                    )
                                else:
                                    status_tag = attacker.get(
                                        "attack_type_secondary", ""
                                    ) or attacker.get("attack_type_primary", "")
                                    if not self._resilience_blocks(
                                        attacker, victim, custom_tag=status_tag
                                    ):
                                        self._apply_wight_decay_effect(attacker, victim)

                    elif (
                        unit_type == "Sentry" and victim.get("uran_turns_left", 0) <= 0
                    ):
                        self._log(
                            f"Шанс наложить воду {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            if self._is_immune_status(attacker, victim):
                                self._log(
                                    f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — вода НЕ накладывается "
                                    f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                                )
                            else:
                                status_tag = attacker.get(
                                    "attack_type_secondary", ""
                                ) or attacker.get("attack_type_primary", "")
                                if not self._resilience_blocks(
                                    attacker, victim, custom_tag=status_tag
                                ):
                                    turns = self.rng.randint(1, URAN_TURNS)
                                    victim["uran_turns_left"] = turns
                                    victim["uran_damage_per_tick"] = int(
                                        attacker.get("damage_secondary", 0) or 0
                                    )
                                    self._log(
                                        f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} накладывает воду "
                                        f"({victim['uran_damage_per_tick']} урона/ход) на "
                                        f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                    )

                    elif (
                        unit_type == "Watcher" and victim.get("burn_turns_left", 0) <= 0
                    ):
                        self._log(
                            f"Ignite chance {int(acc2)}% — "
                            + ("success" if roll else "fail")
                            + f" vs {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            if self._is_immune_status(attacker, victim):
                                self._log(
                                    f"Immunity blocks '{attacker.get('attack_type_secondary', '')}' on "
                                    f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                                )
                            else:
                                status_tag = attacker.get(
                                    "attack_type_secondary", ""
                                ) or attacker.get("attack_type_primary", "")
                                if not self._resilience_blocks(
                                    attacker, victim, custom_tag=status_tag
                                ):
                                    turns = self.rng.randint(1, BURN_TURNS)
                                    victim["burn_turns_left"] = turns
                                    victim["burn_damage_per_tick"] = int(
                                        attacker.get("damage_secondary", 0) or 0
                                    )
                                    self._log(
                                        f"{atk_team.upper()} {attacker['name']}#{attacker['position']} applies burn "
                                        f"({victim['burn_damage_per_tick']} dmg/turn) to "
                                        f"{victim['team'].upper()} {victim['name']}#{victim['position']} for {turns} turns."
                                    )

                    elif unit_type == "Lord":
                        pos = attacker["position"]
                        if not self._lord_applied_burn.get(pos, False):
                            self._log(
                                f"Шанс наложить поджог {int(acc2)}% — "
                                + ("успех" if roll else "неудача")
                                + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                            if roll:
                                if self._is_immune_status(attacker, victim):
                                    self._log(
                                        f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — поджог НЕ накладывается "
                                        f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                                    )
                                else:
                                    status_tag = attacker.get(
                                        "attack_type_secondary", ""
                                    ) or attacker.get("attack_type_primary", "")
                                    if not self._resilience_blocks(
                                        attacker, victim, custom_tag=status_tag
                                    ):
                                        turns = self.rng.randint(1, BURN_TURNS)
                                        victim["burn_turns_left"] = turns
                                        victim["burn_damage_per_tick"] = int(
                                            attacker.get("damage_secondary", 0) or 0
                                        )
                                        self._lord_applied_burn[pos] = True
                                        self._log(
                                            f"{atk_team.upper()} {attacker['name']}#{pos} накладывает поджог "
                                            f"({victim['burn_damage_per_tick']} урона/ход) на "
                                            f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                        )

                    elif unit_type == "Ismir son":
                        pos = attacker["position"]
                        if not self._ismir_applied_uran.get(pos, False):
                            self._log(
                                f"Шанс наложить воду {int(acc2)}% — "
                                + ("успех" if roll else "неудача")
                                + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                            if roll:
                                if self._is_immune_status(attacker, victim):
                                    self._log(
                                        f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — вода НЕ накладывается "
                                        f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                                    )
                                else:
                                    status_tag = attacker.get(
                                        "attack_type_secondary", ""
                                    ) or attacker.get("attack_type_primary", "")
                                    if not self._resilience_blocks(
                                        attacker, victim, custom_tag=status_tag
                                    ):
                                        turns = self.rng.randint(1, URAN_TURNS)
                                        victim["uran_turns_left"] = turns
                                        victim["uran_damage_per_tick"] = int(
                                            attacker.get("damage_secondary", 0) or 0
                                        )
                                        self._ismir_applied_uran[pos] = True
                                        self._log(
                                            f"? {atk_team.upper()} {attacker['name']}#{pos} накладывает воду "
                                            f"({victim['uran_damage_per_tick']} урона/ход) на "
                                            f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                        )

                    elif unit_type == "Spider":
                        self._apply_cached_poison(
                            attacker, victim, self._spider_applied_poison
                        )

                    elif unit_type == "Centaur Savage":
                        extra_damage = float(attacker.get("damage", 0) or 0) * 0.05
                        extra_damage = round(extra_damage, 2)
                        if extra_damage > 0 and victim.get("health", 0) > 0:
                            before, after = self._subtract_health(victim, extra_damage)
                            self._log(
                                f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} Критический удар "
                                f"({extra_damage:g}) to {victim['team'].upper()} {victim['name']}#{victim['position']} "
                                f"({before:g}>{max(0.0, after):g})."
                            )

                self._apply_hero_artifact_post_hit_statuses(attacker, victim)

            if victim["health"] <= 0:
                victim["initiative"] = 0
                self._kill_linked_summons(victim)
                self._log(
                    f"? {victim['team'].upper()} {victim['name']}#{victim['position']} выведен из строя."
                )
            return True, "ok"

        # ====================== блоки по типам ======================

        # --- AOE-статусы без урона ---
        if unit_type == "Shadow":
            return _aoe_status_all(
                "парализовать",
                "Shadow",
                lambda v: self._apply_paralysis_effect(attacker, v),
            )
        elif unit_type == "Incub":
            return _aoe_status_all(
                "Окаменить",
                "Incub",
                lambda v: self._apply_paralysis_effect(attacker, v),
            )
        elif unit_type == "Succub":
            # Учитывать иммунитет и resistance для превращения
            def _succub_apply(v):
                forbiddenwitch_names = {
                    "Ашган",
                    "Ашкаэль",
                    "Видар",
                    "Мизраэль",
                    "Иллюмиэлль",
                }
                # Иммунитет/стойкость к «превращению»
                if self._is_immune_status(attacker, v):
                    self._log(
                        f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — превращение НЕ накладывается "
                        f"на {v['team'].upper()} {v['name']}#{v['position']}."
                    )
                    return
                elif v.get("name") in forbiddenwitch_names:
                    self._log("Это страж столицы")
                    return
                status_tag = attacker.get("attack_type_secondary", "") or attacker.get(
                    "attack_type_primary", ""
                )
                if self._resilience_blocks(attacker, v, custom_tag=status_tag):
                    return
                self._apply_witch_effect(attacker, v)

            return _aoe_status_all("превратить", "Succub", _succub_apply)

        # --- Саппорты: точечное ЛЕЧЕНИЕ через противоположную клетку ---
        elif unit_type in POINT_HEAL_SUPPORT_TYPES:
            if target_pos is None:
                self._log(
                    f"Лечение не сработало: {atk_team.upper()} {attacker['name']}#{attacker['position']} — не указана цель."
                )
                return False, "dead_or_absent"
            opposite_pos = self._opposite_position(target_pos, attacker["team"])
            recipient = (
                _unit_at(opposite_pos)
                if opposite_pos is not None
                else None
            )
            if (
                recipient is None
                or recipient.get("team") != atk_team
                or not self._alive(recipient)
            ):
                self._log(
                    f"Лечение не сработало: {atk_team.upper()} {attacker['name']}#{attacker['position']} не находит союзника на противоположной позиции."
                )
                return False, "dead_or_absent"
            healed = self._apply_cliric_heal(attacker, recipient)
            if not healed:
                self._log(
                    f"? Лечение без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} уже на максимальном здоровье."
                )
            return True, "ok"

        elif unit_type == "Patriach":
            if target_pos is None:
                self._log(
                    f"Помощь без указания: {atk_team.upper()} {attacker['name']}#{attacker['position']} - не выбрана цель."
                )
                return False, "dead_or_absent"
            opposite_pos = self._opposite_position(target_pos, attacker["team"])
            recipient = (
                _unit_at(opposite_pos)
                if opposite_pos is not None
                else None
            )
            result = self._apply_patriach_support(attacker, recipient)
            if result == "invalid":
                self._log(
                    f"Помощь без цели: {atk_team.upper()} {attacker['name']}#{attacker['position']} не может выбрать союзника на противоположной клетке."
                )
                return False, "dead_or_absent"
            if result == "revive_failed" and recipient is not None:
                self._log(
                    f"? Воскрешение без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не может быть возвращён к жизни."
                )
            elif result == "heal_no_effect" and recipient is not None:
                self._log(
                    f"? Лечение без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} уже на максимальном здоровье."
                )
            return True, "ok"

        # --- Саппорты: точечный БАФФ / доп. ход через противоположную клетку ---
        elif unit_type in POINT_BUFF_SUPPORT_TYPES:
            if target_pos is None:
                self._log(
                    f"Усиление не сработало: {atk_team.upper()} {attacker['name']}#{attacker['position']} — не указана цель."
                )
                return False, "dead_or_absent"
            opposite_pos = self._opposite_position(target_pos, attacker["team"])
            recipient = (
                _unit_at(opposite_pos)
                if opposite_pos is not None
                else None
            )
            if (
                recipient is None
                or recipient.get("team") != atk_team
                or not self._alive(recipient)
            ):
                self._log(
                    f"Усиление не сработало: {atk_team.upper()} {attacker['name']}#{attacker['position']} не находит союзника на противоположной позиции."
                )
                return False, "dead_or_absent"

            if unit_type == "Alchemist":
                if self._apply_alchemist_support(attacker, recipient):
                    self._log(
                        f"Поддержка: {atk_team.upper()} {attacker['name']}#{attacker['position']} восстанавливает инициативу союзника на pos{recipient['position']}."
                    )
                else:
                    self._log(
                        f"Поддержка без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не получает бонус."
                    )
            else:
                cleanse_support = unit_type in CLEANSING_SUPPORT_TYPES
                cleansed = (
                    self._cleanse_negative_effects(recipient)
                    if cleanse_support
                    else False
                )
                buffed = self._apply_travnitsa_buff(attacker, recipient)

                if buffed:
                    if cleanse_support and cleansed:
                        self._log(
                            f"Усиление и очищение: {atk_team.upper()} {attacker['name']}#{attacker['position']} усиливает и снимает негатив с союзника на pos{recipient['position']}."
                        )
                    else:
                        self._log(
                            f"Усиление: {atk_team.upper()} {attacker['name']}#{attacker['position']} повышает урон союзника на pos{recipient['position']}"
                        )
                else:
                    self._log(
                        f"Усиление без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не может быть усилен."
                    )
                    if cleanse_support and cleansed:
                        self._log(
                            f"Очищение: {atk_team.upper()} {attacker['name']}#{attacker['position']} всё же снимает негативные эффекты с союзника на pos{recipient['position']}."
                        )
                    elif cleanse_support:
                        self._log(
                            f"Очищение без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не имел негативных эффектов."
                        )
            return True, "ok"

        # --- Саппорты: массовое ЛЕЧЕНИЕ (Profit, Sundancer, Sylfid) ---
        elif unit_type in MASS_HEAL_TYPES:
            healed_any = False
            cleansed_any = False
            profit_cleanses = unit_type == "Profit"

            for ally in self.combined:
                if (
                    ally.get("team") == atk_team
                    and self._alive(ally)
                    and ally is not attacker
                ):
                    if profit_cleanses and self._cleanse_negative_effects(ally):
                        cleansed_any = True
                    if self._apply_cliric_heal(attacker, ally):
                        healed_any = True

            if healed_any or (profit_cleanses and cleansed_any):
                if profit_cleanses and healed_any and cleansed_any:
                    self._log(
                        f"? Массовое лечение и очищение: {atk_team.upper()} {attacker['name']}#{attacker['position']} "
                        f"восстанавливает здоровье и снимает негатив с союзников."
                    )
                elif healed_any:
                    self._log(
                        f"? Массовое лечение: {atk_team.upper()} {attacker['name']}#{attacker['position']} "
                        f"восстанавливает союзникам здоровье."
                    )
                else:  # только Profit может очистить без лечения
                    self._log(
                        f"? Массовое очищение: {atk_team.upper()} {attacker['name']}#{attacker['position']} "
                        f"снимает негативные эффекты с союзников."
                    )
            else:
                self._log(
                    f"? Массовое лечение без эффекта: {atk_team.upper()} {attacker['name']}#{attacker['position']} — все союзники в порядке."
                )
            return True, "aoe"

        elif unit_type == "Occultmaster":
            spawned, reason = handle_occultmaster_action(
                self,
                attacker,
                RED_FRONT_POSITIONS,
                RED_BACK_POSITIONS,
                BLUE_FRONT_POSITIONS,
                BLUE_BACK_POSITIONS,
            )
            if spawned:
                _apply_transformations_to_units(self.combined)
            if not spawned and atk_team == "red":
                self._log(
                    f"{atk_team.upper()} ход: {atk_name}#{attacker['position']} ({unit_type}) не может призвать союзников и встаёт в защиту."
                )
                armor_before = int(attacker.get("armor", 0) or 0)
                attacker["armor"] = armor_before + DEFEND_ARMOR_BONUS
                attacker["defense"] = 1
                self._log(
                    f"{atk_team.upper()} DEFENCE: {atk_name}#{attacker['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{attacker['armor']})."
                )
            return spawned, reason

        elif unit_type == "Lyf":
            spawned, reason = handle_lyf_action(
                self,
                attacker,
                RED_FRONT_POSITIONS,
                RED_BACK_POSITIONS,
                BLUE_FRONT_POSITIONS,
                BLUE_BACK_POSITIONS,
            )
            if spawned:
                _apply_transformations_to_units(self.combined)
            if not spawned and atk_team == "red":
                self._log(
                    f"{atk_team.upper()} ход: {atk_name}#{attacker['position']} ({unit_type}) не может призвать союзников и встаёт в защиту."
                )
                armor_before = int(attacker.get("armor", 0) or 0)
                attacker["armor"] = armor_before + DEFEND_ARMOR_BONUS
                attacker["defense"] = 1
                self._log(
                    f"{atk_team.upper()} DEFENCE: {atk_name}#{attacker['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{attacker['armor']})."
                )
            return spawned, reason

        elif unit_type == "Laclaan":
            spawned, reason = handle_laclaan_action(
                self,
                attacker,
                RED_FRONT_POSITIONS,
                RED_BACK_POSITIONS,
                BLUE_FRONT_POSITIONS,
                BLUE_BACK_POSITIONS,
            )
            if spawned:
                _apply_transformations_to_units(self.combined)
            if not spawned and atk_team == "red":
                self._log(
                    f"{atk_team.upper()} ход: {atk_name}#{attacker['position']} ({unit_type}) не может призвать союзников и встаёт в защиту."
                )
                armor_before = int(attacker.get("armor", 0) or 0)
                attacker["armor"] = armor_before + DEFEND_ARMOR_BONUS
                attacker["defense"] = 1
                self._log(
                    f"{atk_team.upper()} DEFENCE: {atk_name}#{attacker['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{attacker['armor']})."
                )
            return spawned, reason

        # --- Призыватель: призывает юнита на свободную союзную клетку ---
        elif self._is_summoner_type(unit_type):
            if target_pos is None:
                self._log(
                    f"Призыв не сработал: {atk_team.upper()} {attacker['name']}#{attacker['position']} — не указана цель."
                )
                return False, "dead_or_absent"

            # Куда ставить призыв: противоположная клетка НА СВОЕЙ СТОРОНЕ
            dest = self._opposite_position(target_pos, attacker["team"])
            if dest is None:
                self._log(
                    f"Призыв не сработал: не удалось сопоставить pos{target_pos} для команды {atk_team}."
                )
                return False, "dead_or_absent"

            # Проверка занятости клетки живым юнитом
            slot = _unit_at(dest)
            occupied = slot is not None and self._alive(slot)

            # Если dest — задний ряд, запрещаем призыв, если в той же колонке есть живой большой союзник
            def _partner_cell(p: int) -> Optional[int]:
                if p in BLUE_FRONT_POSITIONS or p in RED_FRONT_POSITIONS:
                    return p + 3
                if p in BLUE_BACK_POSITIONS or p in RED_BACK_POSITIONS:
                    return p - 3
                return None

            partner_blocked = False
            if dest in BLUE_BACK_POSITIONS + RED_BACK_POSITIONS:
                partner = _partner_cell(dest)
                if partner is not None:
                    partner_unit = _unit_at(partner)
                    partner_blocked = (
                        partner_unit is not None
                        and self._alive(partner_unit)
                        and partner_unit.get("team") == atk_team
                        and bool(partner_unit.get("big", False))
                    )

            if occupied or partner_blocked:
                reason = "занята" if occupied else "заблокирована большим союзником"
                self._log(
                    f"Призыв отменён: клетка pos{dest} {reason} для {atk_team.upper()} {attacker['name']}#{attacker['position']}."
                )
                return False, "dead_or_absent"

            # Шаблон призываемого юнита (Зомби), с адаптацией под команду/позицию/ряд

            summoner_name = (attacker.get("name") or "").strip().lower()
            position_of_summoner = attacker.get("position")

            if "оккульт" in summoner_name:
                spawn_template = {
                    "name": "Зомби",
                    "initiative": 0,
                    "initiative_base": 50,
                    "unit_type": "Warrior",
                    "damage": 50,
                    "damage_secondary": 0,
                    "health": 170,
                    "max_health": 170,
                    "armor": 0,
                    "accuracy": 80,
                    "accuracy_secondary": 0,
                    "immunity": ["Death"],
                    "resistance": [],
                    "attack_type_primary": "Weapon",
                    "attack_type_secondary": "",
                    "big": False,
                    "Summoned": position_of_summoner,
                }
            elif "элементалист" in summoner_name:
                spawn_template = {
                    "name": "Элементаль Воздуха",
                    "initiative": 0,
                    "initiative_base": 40,
                    "unit_type": "Archer",
                    "damage": 30,
                    "damage_secondary": 0,
                    "health": 100,
                    "max_health": 100,
                    "armor": 0,
                    "accuracy": 80,
                    "accuracy_secondary": 0,
                    "immunity": ["Air"],
                    "resistance": [],
                    "attack_type_primary": "Air",
                    "attack_type_secondary": "",
                    "big": False,
                    "Summoned": position_of_summoner,
                }
            elif "мудрец" in summoner_name:
                spawn_template = {
                    "name": "Малый энт",
                    "initiative": 0,
                    "initiative_base": 50,
                    "unit_type": "Warrior",
                    "damage": 30,
                    "damage_secondary": 0,
                    "health": 80,
                    "max_health": 80,
                    "armor": 0,
                    "accuracy": 80,
                    "accuracy_secondary": 0,
                    "immunity": [],
                    "resistance": [],
                    "attack_type_primary": "Weapon",
                    "attack_type_secondary": "",
                    "big": False,
                    "Summoned": position_of_summoner,
                }
            else:
                self._log(
                    f"Призыв отменён: {atk_team.upper()} {attacker['name']}#{attacker['position']} не умеет призывать существ."
                )
                return False, "dead_or_absent"

            # Корректировка позиции и стойки с учётом выбранного шаблона
            spawn = {
                **spawn_template,
                "team": atk_team,
                "position": dest,
                "stand": "ahead"
                if dest in (RED_FRONT_POSITIONS + BLUE_FRONT_POSITIONS)
                else "behind",
                "paralyzed": 0,
                "long_paralyzed": 0,
                "running_away": 0,
                "transformed": 0,
                "basestats": [],
            }
            _apply_transformations_to_unit(spawn)

            # Инициализация полей как в _reset_state()
            # Призванные не дают опыта за убийство — иначе их можно фармить бесконечно.
            spawn.setdefault("exp_kill", 0)
            spawn.setdefault("defense", 0)
            spawn.setdefault("poison_damage_per_tick", 0)
            spawn.setdefault("burn_turns_left", 0)
            spawn.setdefault("burn_damage_per_tick", 0)
            spawn.setdefault("uran_turns_left", 0)
            spawn.setdefault("uran_damage_per_tick", 0)
            spawn.setdefault("resilience_used_types", [])
            spawn.setdefault("bonusturn", 0)
            spawn.setdefault("original_damage", spawn.get("damage", 0))

            # В позиции обычно есть «пустышка»; перезапишем её, чтобы не дублировать позицию
            if slot is None:
                self.combined.append(spawn)
            else:
                self._accumulate_overwritten_exp(slot)
                slot.clear()
                slot.update(spawn)

            self._log(
                f"Призыв: {atk_team.upper()} {attacker['name']}#{attacker['position']} "
                f"призывает {spawn['name']} на pos{dest} ({spawn['stand']})."
            )
            return True, "ok"

        # --- AOE-урон по врагам (включая Vampire/Highvampire) ---
        elif unit_type in AOE_TYPES:
            return _aoe_damage_all()

        # --- Обычная одиночная атака и пост-эффекты ---
        else:
            return _single_target()

    def _apply_exp_award_to_unit(self, unit: Dict, awarded_exp: int) -> None:
        if awarded_exp <= 0 or _is_thief_battle_unit(unit):
            return

        current = float(unit.get("exp_current", 0) or 0)
        new_value = current + int(awarded_exp)
        req_raw = unit.get("exp_required", 0)
        unit["exp_current"] = new_value

        if isinstance(req_raw, str) and req_raw.lower() == "max":
            return
        try:
            req_value = float(req_raw)
        except (TypeError, ValueError):
            return
        if req_value <= 0 or new_value < req_value:
            return

        unit_name = unit.get("name", "unknown")
        next_level_exp = _resolve_unit_next_level_exp(unit)
        if _uses_dynamic_unit_levelup(unit):
            _apply_dynamic_unit_levelup(
                unit,
                exp_required=_to_int_or_default(req_value, default=0),
            )
        elif _resolve_unit_hero_flag(unit) and next_level_exp > 0:
            unit["Level"] = _to_int_or_default(unit.get("Level", 0), default=0) + 1
            unit["exp_required"] = (
                _to_int_or_default(req_value, default=0) + next_level_exp
            )
            unit["exp_current"] = 0
            _apply_hero_levelup_bonuses(unit)
        else:
            unit["exp_current"] = max(0.0, req_value - 1)
        self._log(f"Уровень юнита {unit_name} повышен")
        self.last_levelups.append(unit_name)

    def _apply_battle_exp(self, losing_team: str) -> None:
        """Award XP with original-game timing for deaths, revivals and escapes."""
        escaped_units = [
            unit
            for unit in (getattr(self, "escaped_units", []) or [])
            if isinstance(unit, dict)
        ]
        escaped_keys = {
            (str(unit.get("team", "")), int(unit.get("position", 0) or 0))
            for unit in escaped_units
        }
        event_based = int(getattr(self, "_battle_exp_event_count", 0) or 0) > 0

        if event_based:
            defeated_exp = getattr(self, "_battle_defeated_exp", {})
            total_exp = float(defeated_exp.get(losing_team, 0.0) or 0.0)
        else:
            # Compatibility path for direct callers/tests that provide an
            # already-finished battle without running combat damage events.
            total_exp = 0.0
            for unit in self.combined:
                if unit.get("team") != losing_team:
                    continue
                unit_key = (
                    str(unit.get("team", "")),
                    int(unit.get("position", 0) or 0),
                )
                if unit_key in escaped_keys:
                    continue
                if int(unit.get("running_away", 0)) == 1 and self._alive(unit):
                    continue
                total_exp += float(unit.get("exp_kill", 0) or 0)

            bank = getattr(self, "overwritten_exp_kill", None)
            if isinstance(bank, dict):
                total_exp += float(bank.get(losing_team, 0.0) or 0.0)

        winning_team = "red" if losing_team == "blue" else "blue"
        try:
            exp_multiplier = max(
                0.0,
                float(getattr(self, "exp_multiplier", 1.0) or 1.0),
            )
        except (TypeError, ValueError):
            exp_multiplier = 1.0
        winner_multiplier = exp_multiplier if winning_team == "blue" else 1.0
        total_exp *= winner_multiplier

        self.last_battle_exp = total_exp
        self.last_levelups = []
        self._log(f"Опыт за бой: {total_exp:g}")

        current_winners = [
            unit
            for unit in self.combined
            if unit.get("team") == winning_team
            and self._alive(unit)
            and not _is_thief_battle_unit(unit)
        ]

        if event_based:
            # Escaped winners retain only XP earned before they actually left.
            recipients = [
                *current_winners,
                *[
                    unit
                    for unit in escaped_units
                    if unit.get("team") == winning_team
                    and not _is_thief_battle_unit(unit)
                ],
            ]
            awards = [
                (
                    unit,
                    int(
                        math.floor(
                            max(
                                0.0,
                                float(unit.get("_battle_exp_earned", 0.0) or 0.0),
                            )
                            * winner_multiplier
                            + 0.5
                        )
                    ),
                )
                for unit in recipients
            ]
        else:
            # Legacy/direct-state battles have no timing information available.
            recipients = [
                unit
                for unit in current_winners
                if int(unit.get("running_away", 0)) == 0
                and (
                    str(unit.get("team", "")),
                    int(unit.get("position", 0) or 0),
                )
                not in escaped_keys
            ]
            share = total_exp / len(recipients) if recipients else 0.0
            rounded_share = int(math.floor(share + 0.5))
            awards = [(unit, rounded_share) for unit in recipients]

        for unit, awarded_exp in awards:
            self._apply_exp_award_to_unit(unit, awarded_exp)

        # This field is battle-local and must never leak into campaign state.
        for unit in [*self.combined, *escaped_units]:
            unit.pop("_battle_exp_earned", None)

    def _clear_dead_running_away_flags(self) -> None:
        for unit in self.combined:
            if int(unit.get("running_away", 0) or 0) == 1 and not self._alive(unit):
                unit["running_away"] = 0

    def _is_post_victory_healer(self, unit: Optional[Dict], team: str) -> bool:
        return bool(
            isinstance(unit, dict)
            and unit.get("team") == team
            and unit.get("name") in POST_VICTORY_HEALER_NAMES
            and unit.get("unit_type") in POST_VICTORY_HEALER_TYPES
        )

    def _support_selector_for_recipient(
        self, healer: Dict, recipient_pos: int
    ) -> Optional[int]:
        """Return the mirrored action position used to select an allied unit."""
        if healer.get("team") == "blue":
            own_positions = BLUE_POSITIONS
            selector_positions = RED_POSITIONS
        else:
            own_positions = RED_POSITIONS
            selector_positions = BLUE_POSITIONS
        try:
            return selector_positions[own_positions.index(int(recipient_pos))]
        except (TypeError, ValueError):
            return None

    def _post_victory_auto_selector(self, healer: Dict) -> Optional[int]:
        unit_type = healer.get("unit_type")
        if unit_type in POINT_HEAL_SUPPORT_TYPES:
            recipient_pos = self._cliric_auto_target(healer)
        elif unit_type in PATRIACH_SUPPORT_TYPES:
            recipient_pos = self._patriach_auto_target(healer)
        elif unit_type in MASS_HEAL_TYPES:
            selector_positions = (
                RED_POSITIONS if healer.get("team") == "blue" else BLUE_POSITIONS
            )
            return selector_positions[0] if selector_positions else None
        else:
            return None
        if recipient_pos is None:
            return None
        return self._support_selector_for_recipient(healer, recipient_pos)

    def _execute_post_victory_heal_turn(
        self, healer: Dict, target_pos: Optional[int] = None
    ) -> float:
        """Execute exactly one native healing action and return restored HP."""
        if not self._alive(healer):
            return 0.0
        if target_pos is None:
            target_pos = self._post_victory_auto_selector(healer)
        if target_pos is None:
            self._log(
                f"Послепобедное лечение: {healer['team'].upper()} "
                f"{healer['name']}#{healer['position']} не находит подходящей цели."
            )
            return 0.0

        team = str(healer.get("team", "") or "")
        before_hp = sum(
            max(0.0, float(unit.get("health", 0) or 0))
            for unit in self.combined
            if unit.get("team") == team
        )
        self._attack(healer, int(target_pos))
        after_hp = sum(
            max(0.0, float(unit.get("health", 0) or 0))
            for unit in self.combined
            if unit.get("team") == team
        )
        return max(0.0, after_hp - before_hp)

    def _finalize_victory(self, winner_team: str) -> None:
        loser_team = "red" if winner_team == "blue" else "blue"
        self._post_victory_team = None
        self._post_victory_healer_positions = []
        self.current_blue_attacker_pos = None
        self.blue_attacks_left = 0
        self.winner = winner_team
        self._log(f"Победа {winner_team.upper()}!")
        self._apply_battle_exp(loser_team)

    def _activate_next_blue_post_victory_healer(self) -> None:
        while self._post_victory_healer_positions:
            position = int(self._post_victory_healer_positions.pop(0))
            healer = self._unit_by_position(position)
            if not self._is_post_victory_healer(healer, "blue"):
                continue
            if not self._alive(healer):
                continue
            self.current_blue_attacker_pos = position
            self.blue_attacks_left = 1
            self._log(
                f"Послепобедный ход лечения BLUE: "
                f"{healer['name']}#{healer['position']} получает одно действие."
            )
            return
        self._finalize_victory("blue")

    def _begin_post_victory_healing(self, winner_team: str) -> None:
        # Preserve the old victory cleanup ordering before selecting healers.
        # Restoring copied Doppelgangers also keeps them out of this phase.
        self._clear_dead_running_away_flags()
        self._revert_fenrir_survivors()
        self._restore_default_doppelgangers()
        self._restore_all_transformed_units()

        healer_positions = sorted(
            int(unit.get("position", 0) or 0)
            for unit in self.combined
            if self._is_post_victory_healer(unit, winner_team) and self._alive(unit)
        )
        if not healer_positions:
            self._finalize_victory(winner_team)
            return

        self._post_victory_team = winner_team
        self._post_victory_healer_positions = healer_positions
        self.current_blue_attacker_pos = None
        self.blue_attacks_left = 0
        self._log(
            f"Противник побеждён. Начинается послепобедное лечение "
            f"отряда {winner_team.upper()}: {len(healer_positions)} ход(а)."
        )

        if winner_team == "blue":
            self._activate_next_blue_post_victory_healer()
            return

        while self._post_victory_healer_positions:
            position = int(self._post_victory_healer_positions.pop(0))
            healer = self._unit_by_position(position)
            if not self._is_post_victory_healer(healer, "red"):
                continue
            if not self._alive(healer):
                continue
            self._log(
                f"Послепобедный ход лечения RED: "
                f"{healer['name']}#{healer['position']} получает одно действие."
            )
            self._execute_post_victory_heal_turn(healer)
        self._finalize_victory("red")

    def _check_victory_after_hit(self):
        if self.winner is not None or self._post_victory_team is not None:
            return
        if not self._team_alive("blue"):
            self._begin_post_victory_healing("red")
        elif not self._team_alive("red"):
            self._begin_post_victory_healing("blue")

    def _advance_until_blue_turn(self) -> bool:
        """
        Автопрокрутка до следующего хода BLUE или завершения боя.
        RED ходит сам.
        """
        while self._team_alive("red") and self._team_alive("blue"):
            nxt = self._pop_next()
            if nxt is None:
                self._log(f"— Конец раунда {self.round_no}.")
                self._end_round_restore()
                self.round_no += 1
                self._log(f"— Начало раунда {self.round_no}.")
                continue

            if not self._apply_start_of_turn_effects(nxt):
                if self.winner is not None:
                    return False
                continue

            units_by_pos = self._units_by_position()

            if nxt.get("paralyzed", 0) == 1 and self._alive(nxt):
                nxt["paralyzed"] = 0
                nxt["initiative"] = 0
                self._log(
                    f"? Паралич: {nxt['team'].upper()} {nxt['name']}#{nxt['position']} пропускает ход."
                )
                self._reset_powerup(nxt)
                continue

            if nxt.get("long_paralyzed", 0) == 1 and self._alive(nxt):
                nxt["initiative"] = 0
                # 33% шанс снять долгий паралич
                if self.rng.random() < 0.33:
                    nxt["long_paralyzed"] = 0
                    self._log(
                        f"? Долгий паралич: {nxt['team'].upper()} {nxt['name']}#{nxt['position']} пропускает ход и восстанавливается."
                    )
                else:
                    self._log(
                        f"? Пас: {nxt['team'].upper()} {nxt['name']}#{nxt['position']} остаётся парализован."
                    )
                self._reset_powerup(nxt)
                continue

            if nxt["team"] == "blue":
                self.current_blue_attacker_pos = nxt["position"]
                self.blue_attacks_left = (
                    2 if nxt.get("unit_type") in DOUBLE_STRIKE_TYPES else 1
                )
                self._log(
                    f"Ход BLUE: {nxt['name']}#{nxt['position']} (иниц {nxt['initiative']}). "
                    + (
                        "Ожидание действия (демон: 2 удара)."
                        if self.blue_attacks_left == 2
                        else "Ожидание действия."
                    )
                )
                return True

            # Ход RED (авто)
            nxt["initiative"] = 0
            nxt_type = nxt.get("unit_type")
            strikes = 2 if nxt_type in DOUBLE_STRIKE_TYPES else 1

            for hit_i in range(strikes):
                if self.winner is not None:
                    return False

                target_pos = None
                if nxt_type in SMART_MELEE_TARGET_TYPES:
                    # Особая логика для юнитов с параличом/окаменением:
                    # Атакуем цель с макс. уроном, без паралича и иммунитета
                    options = self._warrior_allowed_targets(nxt, units_by_pos=units_by_pos)
                    valid_options = (
                        self._filter_non_immune_targets(nxt, options) if options else []
                    )

                    target_pos = None
                    if valid_options:
                        # Пытаемся найти самую опасную непарализованную цель
                        target_pos = self._pick_highest_damage_non_paralyzed(valid_options)
                        # Если таких нет (все парализованы или нет целей), берем просто по мин. HP (добивание)
                        if target_pos is None:
                            target_pos = self._pick_lowest_hp(valid_options)
                            log_target = "цель с мин. HP (non-immune, т.к. все опасные парализованы)"
                        else:
                            log_target = "цель с макс. damage (без паралича)"

                        prefix = nxt_type
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({prefix}) > {log_target} pos{target_pos} (удар {hit_i + 1}/{strikes})."
                        )
                    else:
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не может атаковать: доступных целей нет."
                        )
                        armor_before = int(nxt.get("armor", 0) or 0)
                        nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                        nxt["defense"] = 1
                        self._log(
                            f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                        )
                        break
                elif nxt_type in MELEE_TYPES:
                    options = self._warrior_allowed_targets(nxt, units_by_pos=units_by_pos)
                    valid_options = (
                        self._filter_non_immune_targets(nxt, options) if options else []
                    )

                    if valid_options:
                        target_pos = self._pick_lowest_hp(valid_options)
                        prefix = nxt_type
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({prefix}) > цель с мин. HP (non-immune) pos{target_pos} (удар {hit_i + 1}/{strikes})."
                        )
                    else:
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не может атаковать: доступных целей нет."
                        )
                        armor_before = int(nxt.get("armor", 0) or 0)
                        nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                        nxt["defense"] = 1
                        self._log(
                            f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                        )
                        break
                elif nxt_type in AOE_TYPES:
                    if hit_i == 0:
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) выполняет массовую атаку."
                        )
                    else:
                        break
                elif nxt_type in SUPPORT_TYPES:
                    if nxt_type in POINT_HEAL_SUPPORT_TYPES:
                        ally_pos = self._cliric_auto_target(nxt)
                        if ally_pos is None:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не находит союзников для лечения."
                            )
                            armor_before = int(nxt.get("armor", 0) or 0)
                            nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                            nxt["defense"] = 1
                            self._log(
                                f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                            )
                            break
                        recipient = units_by_pos.get(ally_pos)
                        if recipient is None:
                            break
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) лечит союзника на pos{ally_pos}."
                        )
                        self._apply_cliric_heal(nxt, recipient)
                        break
                    if nxt_type in PATRIACH_SUPPORT_TYPES:
                        ally_pos = self._patriach_auto_target(nxt)
                        if ally_pos is None:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) не находит союзников для помощи."
                            )
                            armor_before = int(nxt.get("armor", 0) or 0)
                            nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                            nxt["defense"] = 1
                            self._log(
                                f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                            )
                            break
                        recipient = units_by_pos.get(ally_pos)
                        result = self._apply_patriach_support(nxt, recipient)
                        if result == "heal_success":
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) лечит союзника на pos{ally_pos}."
                            )
                        elif result == "heal_no_effect":
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) пытается лечить союзника на pos{ally_pos}, но тот полон сил."
                            )
                        elif result == "revive_success":
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) воскрешает союзника на pos{ally_pos}."
                            )
                        else:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) не смог помочь союзнику на pos{ally_pos}."
                            )
                        break

                    if nxt_type in MASS_HEAL_TYPES:
                        healed_any = False
                        for ally in self.combined:
                            if (
                                ally.get("team") == nxt.get("team")
                                and self._alive(ally)
                                and ally is not nxt
                            ):
                                if self._apply_cliric_heal(nxt, ally):
                                    healed_any = True
                        if healed_any:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) массово исцеляет союзников."
                            )
                        else:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не смог исцелить союзников."
                            )
                        break

                    if nxt_type == "Alchemist":
                        buff_pos = self._alchemist_auto_target(nxt)
                    else:
                        buff_pos = self._travnitsa_auto_target(nxt)

                    if buff_pos is None:
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не находит союзников для поддержки."
                        )
                        if nxt_type == "Alchemist" and nxt.get("team") == "red":
                            armor_before = int(nxt.get("armor", 0) or 0)
                            nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                            nxt["defense"] = 1
                            self._log(
                                f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                            )
                        break

                    recipient = units_by_pos.get(buff_pos)
                    if recipient is None:
                        break

                    if nxt_type == "Alchemist":
                        success = self._apply_alchemist_support(nxt, recipient)
                        action_log = (
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) восстанавливает инициативу союзнику на pos{buff_pos}."
                            if success
                            else f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не смог выполнить действие для союзника на pos{buff_pos}."
                        )
                    else:
                        success = self._apply_travnitsa_buff(nxt, recipient)
                        action_log = (
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) усиливает союзника на pos{buff_pos}."
                            if success
                            else f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не смог выполнить действие для союзника на pos{buff_pos}."
                        )

                    self._log(action_log)
                    break
                else:
                    if self._is_summoner_type(nxt_type):
                        target_pos = self._summoner_auto_target(nxt)
                        if target_pos is None:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не находит свободных клеток для призыва."
                            )
                            armor_before = int(nxt.get("armor", 0) or 0)
                            nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                            nxt["defense"] = 1
                            self._log(
                                f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                            )
                            break
                        log_target = "клетку для призыва"
                    else:
                        live_blue_positions = self._live_positions_of("blue")
                        if nxt_type in ("Witch", "Ghost"):
                            target_pos = self._pick_weapon_immune_or_highest_hp(
                                live_blue_positions
                            )
                            log_target = "цель с Weapon-иммунитетом или макс. HP"
                        elif nxt_type == "Doppelganger":
                            # Двойник копирует цель, а не атакует её (общая ветка
                            # _attack ниже била бы выбранного юнита — в т.ч. своего).
                            target_pos = self._pick_highest_hp_non_big(exclude_unit=nxt)
                            target_unit = (
                                self._unit_by_position(target_pos)
                                if target_pos is not None
                                else None
                            )
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Doppelganger) "
                                f"> копирование цели с макс. HP (non-big) pos{target_pos}."
                            )
                            if not self._apply_doppelganger_copy(nxt, target_unit):
                                self._log(
                                    f"RED ход: {nxt['name']}#{nxt['position']} (Doppelganger) "
                                    f"не находит целей для копирования и встаёт в защиту."
                                )
                                armor_before = int(nxt.get("armor", 0) or 0)
                                nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                                nxt["defense"] = 1
                                self._log(
                                    f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                                )
                            break
                        elif nxt_type == "Baroness":
                            target_pos = (
                                self._pick_highest_hp(live_blue_positions)
                                if live_blue_positions
                                else None
                            )
                            log_target = "цель с макс. HP"
                        else:
                            valid_targets = (
                                self._filter_non_immune_targets(nxt, live_blue_positions)
                                if live_blue_positions
                                else []
                            )
                            if valid_targets:
                                target_pos = self._pick_lowest_hp(valid_targets)
                                log_target = "цель с мин. HP (non-immune)"
                            else:
                                self._log(
                                    f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не находит уязвимых целей."
                                )
                                armor_before = int(nxt.get("armor", 0) or 0)
                                nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                                nxt["defense"] = 1
                                self._log(
                                    f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                                )
                                break
                    self._log(
                        f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) > {log_target} pos{target_pos} (удар {hit_i + 1}/{strikes})."
                    )

                if target_pos is not None or nxt_type in AOE_TYPES:
                    self._attack(nxt, target_pos)
                    self._check_victory_after_hit()
                    if self.winner is not None:
                        return False
            self._reset_powerup(nxt)
        return False

    def scripted_action_for_current_blue(self) -> int:
        """Choose a deterministic script action for the current BLUE unit."""
        mask = self.compute_action_mask()
        true_indexes = [int(index) for index, allowed in enumerate(mask) if bool(allowed)]
        if not true_indexes:
            return WAIT_ACTION_INDEX

        units_by_pos = self._units_by_position()
        attacker = units_by_pos.get(self.current_blue_attacker_pos)
        if attacker is None or not self._alive(attacker):
            return WAIT_ACTION_INDEX if bool(mask[WAIT_ACTION_INDEX]) else true_indexes[0]

        unit_type = str(attacker.get("unit_type", "") or "")

        if unit_type in POINT_HEAL_SUPPORT_TYPES:
            recipient_pos = self._cliric_auto_target(attacker)
            target_pos = (
                self._support_selector_for_recipient(attacker, recipient_pos)
                if recipient_pos is not None
                else None
            )
            if target_pos in TARGET_POSITIONS:
                action_index = TARGET_POSITIONS.index(target_pos)
                if bool(mask[action_index]):
                    return int(action_index)

        if unit_type in PATRIACH_SUPPORT_TYPES:
            recipient_pos = self._patriach_auto_target(attacker)
            target_pos = (
                self._support_selector_for_recipient(attacker, recipient_pos)
                if recipient_pos is not None
                else None
            )
            if target_pos in TARGET_POSITIONS:
                action_index = TARGET_POSITIONS.index(target_pos)
                if bool(mask[action_index]):
                    return int(action_index)

        if unit_type in MASS_HEAL_TYPES:
            for action_index, target_pos in enumerate(TARGET_POSITIONS):
                target = units_by_pos.get(target_pos)
                if (
                    bool(mask[action_index])
                    and target is not None
                    and target.get("team") == attacker.get("team")
                    and self._alive(target)
                ):
                    return int(action_index)

        target_actions = [idx for idx in range(len(TARGET_POSITIONS)) if bool(mask[idx])]
        if target_actions:
            enemy_targets = []
            for idx in target_actions:
                target = units_by_pos.get(TARGET_POSITIONS[idx])
                if (
                    target is not None
                    and target.get("team") != attacker.get("team")
                    and self._alive(target)
                ):
                    enemy_targets.append((idx, target))
            if enemy_targets:
                return int(
                    min(
                        enemy_targets,
                        key=lambda item: (
                            float(item[1].get("health", 0) or 0),
                            TARGET_POSITIONS[item[0]],
                        ),
                    )[0]
                )

            return int(target_actions[0])

        if bool(mask[DEFEND_ACTION_INDEX]):
            return DEFEND_ACTION_INDEX
        if bool(mask[WAIT_ACTION_INDEX]):
            return WAIT_ACTION_INDEX
        return true_indexes[0]

    # ------------------ Наблюдение для агента ------------------

    def _obs(self) -> np.ndarray:
        vec = []
        units_by_pos = self._units_by_position()

        for pos in RED_POSITIONS + BLUE_POSITIONS:
            u = units_by_pos.get(pos)
            hp_raw = float(u.get("health", 0) or 0)
            max_hp_raw = float(u.get("max_health", 0) or 0)
            hp = _norm_positive(
                hp_raw,
                max_hp_raw if max_hp_raw > 0 else HP_FALLBACK_MAX,
            )
            ini = _norm_positive(u.get("initiative", 0) or 0, INIT_CUR_NORM_MAX)
            ini_b = _norm_positive(u.get("initiative_base", 0) or 0, INIT_BASE_NORM_MAX)
            dmg = _norm_positive(u.get("damage", 0) or 0, DMG_NORM_MAX)
            dmg2 = _norm_positive(u.get("damage_secondary", 0) or 0, DMG2_NORM_MAX)
            team_v = 0.0 if u["team"] == "red" else 1.0
            pos_raw = float(u.get("position", pos) or pos)
            pos_v = _clip01((pos_raw - 1.0) / POS_NORM_MAX)
            stand_v = 0.0 if u["stand"] == "ahead" else 1.0

            t_onehot = _one_hot(u.get("unit_type", "Archer"), TYPE_LIST)
            imm_mhot = _multi_hot(u.get("immunity", []), ATTACK_TYPES)
            atk1_oh = (
                _one_hot(u.get("attack_type_primary", ""), ATTACK_TYPES)
                if u.get("attack_type_primary", "") in ATTACK_TYPES
                else [0.0] * len(ATTACK_TYPES)
            )
            atk2_oh = (
                _one_hot(u.get("attack_type_secondary", ""), ATTACK_TYPES)
                if u.get("attack_type_secondary", "") in ATTACK_TYPES
                else [0.0] * len(ATTACK_TYPES)
            )
            res_mhot = _multi_hot(u.get("resistance", []), ATTACK_TYPES)
            armor_v = _norm_positive(u.get("armor", 0) or 0, ARMOR_NORM_MAX)
            acc_v = _norm_positive(u.get("accuracy", 0) or 0, ACC_NORM_MAX)
            acc2_v = _norm_positive(u.get("accuracy_secondary", 0) or 0, ACC_NORM_MAX)
            run_v = _to01_bool(u.get("running_away", 0))
            par_v = _to01_bool(u.get("paralyzed", 0))
            long_par_v = _to01_bool(u.get("long_paralyzed", 0))
            trans_v = _to01_bool(u.get("transformed", 0))
            bonus_v = _norm_positive(u.get("bonusturn", 0) or 0, BONUS_NORM_MAX)

            waited_v = _to01_bool(u.get("waited", 0))
            def_v = _to01_bool(u.get("defense", 0))
            big_v = _to01_bool(u.get("big", False))
            res_used_mhot = _multi_hot(u.get("resilience_used_types", []), ATTACK_TYPES)
            pois_dmg_v = _norm_positive(u.get("poison_damage_per_tick", 0) or 0, DMG2_NORM_MAX)
            burn_dmg_v = _norm_positive(u.get("burn_damage_per_tick", 0) or 0, DMG2_NORM_MAX)
            uran_dmg_v = _norm_positive(u.get("uran_damage_per_tick", 0) or 0, DMG2_NORM_MAX)
            exp_kill_v = _norm_positive(u.get("exp_kill", 0) or 0, EXP_KILL_NORM_MAX)
            exp_required_raw = max(
                0.0,
                float(_to_int_or_default(u.get("exp_required", 0), default=0)),
            )
            exp_required_v = _norm_positive(exp_required_raw, EXP_REQ_NORM_MAX)
            exp_current_v = _norm_positive(
                u.get("exp_current", 0) or 0,
                max(1.0, exp_required_raw),
            )

            vec.extend(
                [
                    hp,
                    ini,
                    ini_b,
                    dmg,
                    dmg2,
                    team_v,
                    pos_v,
                    stand_v,
                    *t_onehot,
                    *imm_mhot,
                    *atk1_oh,
                    *atk2_oh,
                    *res_mhot,
                    *res_used_mhot,
                    armor_v,
                    acc_v,
                    acc2_v,
                    run_v,
                    par_v,
                    long_par_v,
                    trans_v,
                    bonus_v,
                    waited_v,
                    def_v,
                    big_v,
                    pois_dmg_v,
                    burn_dmg_v,
                    uran_dmg_v,
                    exp_kill_v,
                    exp_required_v,
                    exp_current_v,
                ]
            )
        obs = np.array(vec, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        obs = np.clip(obs, 0.0, 1.0)
        return obs

    # ------------------ API Gymnasium ------------------

    def reset(self, *, seed=None, options=None):
        # ВАЖНО: не фиксируем сид, чтобы промахи/порядок были случайными в каждом эпизоде
        self._reset_state()
        self._advance_until_blue_turn()
        return self._obs(), {}

    def _revert_fenrir_survivors(self) -> None:
        """Возвращает всех выживших "Дух Фенрира" обратно в "Повелитель волков".
        Здоровье переносится по той же формуле (доля HP) в шкалу 225.
        Если на юните есть снапшот "wolflord_base", восстанавливаем характеристики из него,
        иначе используем каноничные статы Повелителя волков.
        """
        for u in getattr(self, "combined", []):
            if not isinstance(u, dict):
                continue
            if u.get("name") != "Дух Фенрира":
                continue
            if not self._alive(u):
                continue

            cur_hp = float(u.get("health", 0) or 0)
            cur_max = float(u.get("max_health", 0) or 0)
            ratio = 0.0 if cur_max <= 0 else (cur_hp / cur_max)

            base = u.get("wolflord_base")
            if isinstance(base, dict) and base:
                # Восстанавливаем все сохранённые поля, кроме HP — его зададим по доле
                for k, v in base.items():
                    if k in ("health", "max_health"):
                        continue
                    # глубокая копия для списков/словарей
                    u[k] = deepcopy(v) if isinstance(v, (list, dict)) else v
                max_to = int(base.get("max_health", 225) or 225)
            else:
                # Фолбэк: каноничные характеристики Повелителя волков
                max_to = 225
                u["unit_type"] = "Wolf Lord"
                u["damage"] = 40
                u["damage_secondary"] = 0
                u["initiative_base"] = 40
                u["initiative"] = 40
                u["attack_type_primary"] = "Water"
                u["attack_type_secondary"] = ""
                u["armor"] = 0
                u["accuracy"] = 80
                u["accuracy_secondary"] = 0
                u["immunity"] = []
                u["resistance"] = []
                u["big"] = False

            u["max_health"] = max_to
            u["health"] = int(round(max_to * ratio))
            u["name"] = "Повелитель волков"
            # Инициативу приводим к базовой
            u["initiative"] = u.get("initiative_base", u.get("initiative", 0))
            # Чистим снапшот
            if "wolflord_base" in u:
                try:
                    del u["wolflord_base"]
                except Exception:
                    pass

    def _step_post_victory_blue_heal(self, action):
        """Consume one BLUE healer's single action after combat is decided."""
        assert self._post_victory_team == "blue"
        healer = self._unit_by_position(self.current_blue_attacker_pos)
        if not self._is_post_victory_healer(healer, "blue") or not self._alive(healer):
            self.current_blue_attacker_pos = None
            self.blue_attacks_left = 0
            self._activate_next_blue_post_victory_healer()
            if self.winner is not None:
                info = {
                    "post_victory_heal_phase": True,
                    "post_victory_heal_completed": True,
                }
                return self._obs(), self.reward_win, True, False, info
            healer = self._unit_by_position(self.current_blue_attacker_pos)

        action_idx = int(action)
        action_mask = self.compute_action_mask()
        valid_action = 0 <= action_idx < len(action_mask) and bool(action_mask[action_idx])
        skipped = action_idx == WAIT_ACTION_INDEX or not valid_action
        restored_hp = 0.0

        if not skipped and action_idx < len(TARGET_POSITIONS):
            restored_hp = self._execute_post_victory_heal_turn(
                healer, TARGET_POSITIONS[action_idx]
            )
        else:
            reason = "пропускает действие" if valid_action else "выбрал недопустимое действие"
            self._log(
                f"Послепобедное лечение BLUE: {healer['name']}#{healer['position']} "
                f"{reason}."
            )

        healer["initiative"] = 0
        healer_name = str(healer.get("name", "") or "")
        healer_position = int(healer.get("position", 0) or 0)
        self.current_blue_attacker_pos = None
        self.blue_attacks_left = 0
        self._activate_next_blue_post_victory_healer()

        terminated = self.winner is not None
        info = {
            "post_victory_heal_phase": True,
            "post_victory_healer_name": healer_name,
            "post_victory_healer_position": healer_position,
            "post_victory_healed_hp": float(restored_hp),
            "post_victory_heal_skipped": bool(skipped),
            "post_victory_heal_completed": bool(terminated),
            "post_victory_healers_remaining": len(
                self._post_victory_healer_positions
            ),
        }
        reward = self.reward_win if terminated else self.reward_step
        return self._obs(), reward, terminated, False, info

    def step(self, action):
        assert self.winner is None, "Эпизод завершён — вызовите reset()."

        if self._post_victory_team == "blue":
            return self._step_post_victory_blue_heal(action)

        if self.current_blue_attacker_pos is None:
            self._advance_until_blue_turn()
            if self._post_victory_team == "blue":
                return self._step_post_victory_blue_heal(action)
            if self.winner is not None:
                # Эпизод завершён в промежуточном апдейте — возвращаем Фенрира в Волка
                base = self.reward_win if self.winner == "blue" else self.reward_loss
                return self._obs(), base, True, False, {}

        step_shaping = 0.0
        action_idx = int(action)
        is_defend_action = action_idx == DEFEND_ACTION_INDEX
        is_wait_action = action_idx == WAIT_ACTION_INDEX
        is_run_away_action = action_idx == RUN_AWAY_ACTION_INDEX
        is_first_hero_item_action = FIRST_HERO_ITEM_ACTION_START <= action_idx < (
            FIRST_HERO_ITEM_ACTION_START + len(HERO_ITEM_TARGET_SLOTS)
        )
        is_second_hero_item_action = SECOND_HERO_ITEM_ACTION_START <= action_idx < (
            SECOND_HERO_ITEM_ACTION_START + len(HERO_ITEM_TARGET_SLOTS)
        )
        is_first_hero_item_enemy_action = (
            FIRST_HERO_ITEM_ENEMY_ACTION_START
            <= action_idx
            < (FIRST_HERO_ITEM_ENEMY_ACTION_START + len(HERO_ITEM_TARGET_SLOTS))
        )
        is_second_hero_item_enemy_action = (
            SECOND_HERO_ITEM_ENEMY_ACTION_START
            <= action_idx
            < (SECOND_HERO_ITEM_ENEMY_ACTION_START + len(HERO_ITEM_TARGET_SLOTS))
        )
        is_hero_item_action = (
            is_first_hero_item_action
            or is_second_hero_item_action
            or is_first_hero_item_enemy_action
            or is_second_hero_item_enemy_action
        )
        target_pos = None
        hero_item_slot = None
        hero_item_target_pos = None
        hero_item_no = None
        hero_item_name = ""
        hero_item_applied = False
        hero_item_effect_kind = None
        hero_item_effect_value = 0.0
        hero_item_consumed = False
        hero_item_charge_spent = False
        hero_item_uses_left = None
        hero_item_target_team = ""
        if action_idx < len(TARGET_POSITIONS):
            target_pos = TARGET_POSITIONS[action_idx]
        elif is_first_hero_item_action:
            hero_item_no = 1
            hero_item_slot = HERO_ITEM_TARGET_SLOTS[
                action_idx - FIRST_HERO_ITEM_ACTION_START
            ]
            hero_item_target_pos = BLUE_POSITIONS[hero_item_slot - 1]
            hero_item_target_team = "blue"
        elif is_second_hero_item_action:
            hero_item_no = 2
            hero_item_slot = HERO_ITEM_TARGET_SLOTS[
                action_idx - SECOND_HERO_ITEM_ACTION_START
            ]
            hero_item_target_pos = BLUE_POSITIONS[hero_item_slot - 1]
            hero_item_target_team = "blue"
        elif is_first_hero_item_enemy_action:
            hero_item_no = 1
            hero_item_slot = HERO_ITEM_TARGET_SLOTS[
                action_idx - FIRST_HERO_ITEM_ENEMY_ACTION_START
            ]
            hero_item_target_pos = RED_POSITIONS[hero_item_slot - 1]
            hero_item_target_team = "red"
        elif is_second_hero_item_enemy_action:
            hero_item_no = 2
            hero_item_slot = HERO_ITEM_TARGET_SLOTS[
                action_idx - SECOND_HERO_ITEM_ENEMY_ACTION_START
            ]
            hero_item_target_pos = RED_POSITIONS[hero_item_slot - 1]
            hero_item_target_team = "red"
        if hero_item_no is not None:
            hero_item_name = self._hero_item_name_for_slot(hero_item_no)

        units_by_pos = self._units_by_position()
        attacker = units_by_pos.get(self.current_blue_attacker_pos)
        can_strike = (
            attacker is not None
            and self._alive(attacker)
            and (
                attacker["initiative"] > 0
                or (
                    attacker.get("unit_type") in DOUBLE_STRIKE_TYPES
                    and self.blue_attacks_left > 0
                )
            )
        )
        if can_strike:
            if is_wait_action:
                current_ini = int(attacker.get("initiative", 0) or 0)
                if current_ini > 0:
                    reduced = max(1, int(math.ceil(current_ini / WAIT_INIT_DIVISOR)))
                else:
                    reduced = 0
                attacker["initiative"] = reduced
                attacker["waited"] = 1
                if (
                    attacker.get("unit_type") in DOUBLE_STRIKE_TYPES
                    and self.blue_attacks_left > 1
                ):
                    self.blue_attacks_left = 1
                self._log(
                    f"BLUE действие: {attacker['name']}#{attacker['position']} ожидает (инициатива {current_ini}->{reduced})."
                )
            else:
                attacker["initiative"] = 0

                if is_defend_action:
                    armor_before = int(attacker.get("armor", 0) or 0)
                    attacker["armor"] = armor_before + DEFEND_ARMOR_BONUS
                    attacker["defense"] = 1
                    if (
                        attacker.get("unit_type") in DOUBLE_STRIKE_TYPES
                        and self.blue_attacks_left > 1
                    ):
                        self.blue_attacks_left = 1
                    self._log(
                        f"BLUE действие: {attacker['name']}#{attacker['position']} усиливает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{attacker['armor']})."
                    )
                elif is_run_away_action:
                    attacker["running_away"] = 1
                    if (
                        attacker.get("unit_type") in DOUBLE_STRIKE_TYPES
                        and self.blue_attacks_left > 1
                    ):
                        self.blue_attacks_left = 1
                    self._log(
                        f"BLUE действие: {attacker['name']}#{attacker['position']} выбирает побег (running_away=1)."
                    )
                elif is_hero_item_action:
                    target_unit = (
                        units_by_pos.get(hero_item_target_pos)
                        if hero_item_target_pos is not None
                        else None
                    )
                    hero_item_effect = self._hero_item_effect(hero_item_name)
                    slot_index = int(hero_item_no) - 1 if hero_item_no is not None else -1
                    if (
                        _is_combat_hero_unit(attacker)
                        and hero_item_effect
                        and 0 <= slot_index < len(self.equipped_hero_items)
                        and self._hero_item_slot_available(slot_index, hero_item_effect)
                    ):
                        (
                            hero_item_applied,
                            hero_item_effect_kind,
                            hero_item_effect_value,
                        ) = self._apply_hero_item_effect(
                            item_name=hero_item_name,
                            target_pos=hero_item_target_pos,
                        )
                        if hero_item_applied and hero_item_no is not None:
                            if len(getattr(self, "equipped_hero_item_uses_left", []) or []) < len(
                                self.equipped_hero_items
                            ):
                                self.equipped_hero_item_uses_left = [None] * len(
                                    self.equipped_hero_items
                                )
                            if len(
                                getattr(self, "hero_item_slots_used_this_battle", []) or []
                            ) < len(self.equipped_hero_items):
                                self.hero_item_slots_used_this_battle = [False] * len(
                                    self.equipped_hero_items
                                )
                            hero_item_charge_spent = True
                            if self._hero_item_is_talisman(hero_item_effect):
                                uses_before = self._hero_item_slot_uses_left(
                                    slot_index,
                                    hero_item_effect,
                                )
                                uses_after = max(0, int(uses_before or 0) - 1)
                                self.hero_item_slots_used_this_battle[slot_index] = True
                                if uses_after > 0:
                                    self.equipped_hero_item_uses_left[slot_index] = uses_after
                                    hero_item_uses_left = uses_after
                                else:
                                    self.equipped_hero_item_uses_left[slot_index] = None
                                    self.equipped_hero_items[slot_index] = None
                                    hero_item_consumed = True
                                    hero_item_uses_left = 0
                            else:
                                self.equipped_hero_items[slot_index] = None
                                self.equipped_hero_item_uses_left[slot_index] = None
                                hero_item_consumed = True
                                hero_item_uses_left = 0
                            if hero_item_effect_kind in ("damage", "drain"):
                                self._check_victory_after_hit()

                        target_label = (
                            f"{target_unit['name']}#{target_unit['position']}"
                            if isinstance(target_unit, dict)
                            else f"pos{hero_item_target_pos}"
                        )
                        if hero_item_applied and hero_item_effect_kind == "heal":
                            self._log(
                                f"BLUE действие: {attacker['name']}#{attacker['position']} "
                                f"использует предмет {hero_item_no} '{hero_item_name}' на слот {hero_item_slot} "
                                f"({target_label}) и лечит на {int(round(hero_item_effect_value))} HP."
                            )
                        elif hero_item_applied and hero_item_effect_kind == "damage":
                            self._log(
                                f"BLUE action: {attacker['name']}#{attacker['position']} "
                                f"uses item {hero_item_no} '{hero_item_name}' on {target_label} "
                                f"for {int(round(hero_item_effect_value))} damage."
                            )
                        elif hero_item_applied and hero_item_effect_kind == "revive":
                            self._log(
                                f"BLUE действие: {attacker['name']}#{attacker['position']} "
                                f"использует предмет {hero_item_no} '{hero_item_name}' на слот {hero_item_slot} "
                                f"({target_label}) и воскрешает цель с {int(round(hero_item_effect_value))} HP."
                            )
                        else:
                            self._log(
                                f"BLUE действие: {attacker['name']}#{attacker['position']} "
                                f"пытается использовать предмет {hero_item_no} '{hero_item_name}' на слот {hero_item_slot} "
                                f"(pos{hero_item_target_pos}) без эффекта."
                            )
                    else:
                        self._log(
                            f"BLUE действие: {attacker['name']}#{attacker['position']} "
                            f"выбирает hero-only предмет {hero_item_no} на слот {hero_item_slot} "
                            f"без эффекта."
                        )
                else:
                    wolf_self_select = attacker.get(
                        "unit_type"
                    ) == "Wolf Lord" and target_pos == attacker.get("position")
                    if wolf_self_select:
                        self._log(
                            f"BLUE действие: {attacker['name']}#{attacker['position']} > pos{target_pos} (своя позиция выбрана)"
                        )
                        # Снимем снапшот исходных характеристик Повелителя волков, чтобы вернуть их по окончании боя
                        if not attacker.get("wolflord_base"):
                            attacker["wolflord_base"] = {
                                "name": attacker.get("name"),
                                "unit_type": attacker.get("unit_type"),
                                "damage": attacker.get("damage"),
                                "damage_secondary": attacker.get("damage_secondary", 0),
                                "initiative": attacker.get("initiative", 0),
                                "initiative_base": attacker.get("initiative_base", 0),
                                "attack_type_primary": attacker.get(
                                    "attack_type_primary", "Weapon"
                                ),
                                "attack_type_secondary": attacker.get(
                                    "attack_type_secondary", ""
                                ),
                                "max_health": attacker.get("max_health", 225),
                                "armor": attacker.get("armor", 0),
                                "accuracy": attacker.get("accuracy", 100),
                                "accuracy_secondary": attacker.get(
                                    "accuracy_secondary", 0
                                ),
                                "immunity": list(attacker.get("immunity", [])),
                                "resistance": list(attacker.get("resistance", [])),
                                "big": bool(attacker.get("big", False)),
                            }
                        current_health = float(attacker.get("health", 0) or 0)
                        current_max_health = float(attacker.get("max_health", 0) or 0)
                        health_ratio = (
                            0.0
                            if current_max_health <= 0
                            else current_health / current_max_health
                        )
                        attacker["damage"] = 90
                        attacker["unit_type"] = "Warrior"
                        attacker["initiative"] = 0
                        attacker["initiative_base"] = 65
                        attacker["attack_type_primary"] = "Weapon"
                        attacker["max_health"] = 275
                        attacker["health"] = int(round(275 * health_ratio))
                        attacker["name"] = "Дух Фенрира"
                    else:
                        self._log(
                            f"BLUE действие: {attacker['name']}#{attacker['position']} > pos{target_pos}"
                        )

                    # --- Саппорты и лечение/бафф теперь тоже через _attack() ---
                    if wolf_self_select:
                        pass
                    elif attacker.get("unit_type") in SUPPORT_TYPES:
                        hit, reason = self._attack(attacker, target_pos)
                        if not hit and reason == "dead_or_absent":
                            step_shaping += self.penalty_invalid_target
                        self._check_victory_after_hit()

                    elif attacker.get("unit_type") == "Doppelganger":
                        target_unit = units_by_pos.get(target_pos)
                        if not self._apply_doppelganger_copy(attacker, target_unit):
                            self._log(f"Doppelganger выбор: pos{target_pos} → пусто")
                        self._check_victory_after_hit()

                    elif attacker.get("unit_type") in MELEE_TYPES:
                        allowed = self._warrior_allowed_targets(attacker, units_by_pos=units_by_pos)
                        if target_pos not in allowed:
                            self._log(
                                f"BLUE {attacker['name']}#{attacker['position']} ({attacker.get('unit_type')}) не может достать pos{target_pos}. "
                                f"Доступные цели: {(', '.join('pos' + str(p) for p in allowed)) if allowed else 'нет'}"
                            )
                            step_shaping += self.penalty_unreachable_warrior
                        else:
                            hit, reason = self._attack(attacker, target_pos)
                            if not hit and reason == "dead_or_absent":
                                step_shaping += self.penalty_invalid_target
                            self._check_victory_after_hit()

                    elif attacker.get("unit_type") in AOE_TYPES:
                        hit, reason = self._attack(attacker, target_pos)
                        self._check_victory_after_hit()
                    else:
                        hit, reason = self._attack(attacker, target_pos)
                        if not hit and reason == "dead_or_absent":
                            step_shaping += self.penalty_invalid_target
                        self._check_victory_after_hit()

        step_info = {"battle_hero_item_action": bool(is_hero_item_action)}
        if is_hero_item_action or self.log_enabled or bool(getattr(self, "debug_info", False)):
            step_info.update(
                {
                    "battle_hero_item_slot": (
                        int(hero_item_no) if hero_item_no is not None else None
                    ),
                    "battle_hero_item_target_pos": (
                        int(hero_item_target_pos) if hero_item_target_pos is not None else None
                    ),
                    "battle_hero_item_target_team": str(hero_item_target_team or ""),
                    "battle_hero_item_name": str(hero_item_name or ""),
                    "battle_hero_item_effect_kind": (
                        str(hero_item_effect_kind) if hero_item_effect_kind else ""
                    ),
                    "battle_hero_item_effect_value": float(hero_item_effect_value or 0.0),
                    "battle_hero_item_applied": bool(hero_item_applied),
                    "battle_hero_item_consumed": bool(hero_item_consumed),
                    "battle_hero_item_charge_spent": bool(hero_item_charge_spent),
                    "battle_hero_item_uses_left": hero_item_uses_left,
                    "equipped_hero_items": list(self.equipped_hero_items),
                    "equipped_hero_item_uses_left": list(self.equipped_hero_item_uses_left),
                }
            )

        if self._post_victory_team == "blue":
            healer = self._unit_by_position(self.current_blue_attacker_pos)
            step_info.update(
                {
                    "post_victory_heal_phase": True,
                    "post_victory_heal_started": True,
                    "post_victory_healer_name": (
                        str(healer.get("name", "") or "")
                        if isinstance(healer, dict)
                        else ""
                    ),
                    "post_victory_healer_position": (
                        int(healer.get("position", 0) or 0)
                        if isinstance(healer, dict)
                        else None
                    ),
                    "post_victory_healers_remaining": len(
                        self._post_victory_healer_positions
                    ),
                }
            )
            return (
                self._obs(),
                self.reward_step + step_shaping,
                False,
                False,
                step_info,
            )

        if self.winner is None:
            if self.blue_attacks_left > 0:
                self.blue_attacks_left -= 1
            if self.blue_attacks_left > 0:
                reward, terminated = self.reward_step + step_shaping, False
                return self._obs(), reward, terminated, False, step_info
            self.current_blue_attacker_pos = None
            self._reset_powerup(attacker)
            self._advance_until_blue_turn()

        if self.winner is None:
            base = self.reward_step
            terminated = False
        else:
            base = self.reward_win if self.winner == "blue" else self.reward_loss
            terminated = True

        reward = base + step_shaping
        truncated = False
        if not terminated:
            self.step_count += 1
            if self.step_count >= 1000:
                truncated = True
                self._clear_dead_running_away_flags()
                self._log("? Лимит по шагам: бой остановлен на 1000 такте.")
        return self._obs(), reward, terminated, truncated, step_info

    # ==================== CAMPAIGN MODE ====================

    def _init_with_custom_teams(self, red_team: list, blue_team: list):
        """
        Инициализация боя с произвольными командами.
        Используется в CampaignEnv для запуска боя с разными врагами
        и сохранённым состоянием BLUE команды.

        Args:
            red_team: Список словарей с юнитами RED команды
            blue_team: Список словарей с юнитами BLUE команды
        """
        self._patriach_revived_recipients: set[Tuple[str, int]] = set()
        self.combined = deepcopy(red_team + blue_team)
        _apply_transformations_to_units(self.combined)

        # Применяем стандартную инициализацию к каждому юниту
        for u in self.combined:
            base_ini = int(u.get("initiative_base", 0) or 0)
            u["initiative"] = base_ini
            if self._alive(u):
                u["initiative"] += self.rng.randint(0, 9)

            # Дефолтные значения
            u.setdefault("attack_type_primary", "Weapon")
            u.setdefault("attack_type_secondary", "")
            u.setdefault("immunity", [])
            u.setdefault("resistance", [])
            u.setdefault("armor", 0)
            # Keep battle-start armor so campaign can restore temporary armor changes.
            u["base_armor"] = int(u.get("armor", 0) or 0)
            u.setdefault("accuracy", 100)
            u.setdefault("accuracy_secondary", 0)           
            u.setdefault("damage_secondary", 0)
            u.setdefault("big", False)
            u.setdefault("Level", _resolve_unit_level(u))
            u.setdefault("next_level_exp", _resolve_unit_next_level_exp(u))
            u["hero"] = _resolve_unit_hero_flag(u)
            u["needaunit"] = _resolve_unit_needaunit(u)
            u.setdefault("bonusturn", 0)
            u.setdefault("original_damage", u.get("damage", 0))

            # Сбрасываем состояния боя
            u["defense"] = 0
            u["waited"] = 0
            u["poison_turns_left"] = 0
            u["poison_damage_per_tick"] = 0
            u["burn_turns_left"] = 0
            u["burn_damage_per_tick"] = 0
            u["uran_turns_left"] = 0
            u["uran_damage_per_tick"] = 0
            u["resilience_used_types"] = []
            u["round_effects_done"] = 0
            u["doppel_copied"] = 0
            u["transformed"] = 0
            u["basestats"] = {}
            u.pop("wight_form_name", None)
            u.pop("transform_effect", None)
            u.pop("transform_recover_chance", None)
            u["_battle_exp_earned"] = 0.0

        # Сбрасываем глобальное состояние боя
        self.round_no = 1
        self.winner = None
        self._post_victory_team = None
        self._post_victory_healer_positions = []
        self.current_blue_attacker_pos = None
        self.blue_attacks_left = 0
        self._lord_applied_burn = {}
        self._spider_applied_poison = {}
        self._dregazul_applied_poison = {}
        self._ismir_applied_uran = {}
        self.escaped_units = []
        self.overwritten_exp_kill = {"red": 0.0, "blue": 0.0}
        self._battle_exp_event_count = 0
        self._battle_defeated_exp = {"red": 0.0, "blue": 0.0}
        self.step_count = 0
        self.hero_item_slots_used_this_battle = [False] * len(self.equipped_hero_items)
        if len(getattr(self, "equipped_hero_item_uses_left", []) or []) < len(
            self.equipped_hero_items
        ):
            self.equipped_hero_item_uses_left = [None] * len(self.equipped_hero_items)

        self._log(f"Бой начат с кастомными командами. Раунд {self.round_no}.")

        # Прокручиваем до первого хода BLUE
        self._advance_until_blue_turn()
