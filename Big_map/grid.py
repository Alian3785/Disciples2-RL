"""
Centralized grid/campaign configuration.

This module is the single source of truth for:
- campaign grid size and start position
- base enemy camp coordinates on the campaign map
- heal/revive tile placement on the campaign map
- obstacle tile placement on the campaign map
- blue hero squad template
- enemy squad composition specs used to build campaign battles
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Dict, Optional, Tuple
from data_dicts_compact_lines import DATA, is_hero_name

DEFAULT_GRID_SIZE = 48
DEFAULT_HERO_GRID_POSITION: Tuple[int, int] = (5, 27)

BASE_STATIC_OBSTACLE_BLOCKS: Tuple[Tuple[int, int, int, int], ...] = (
    # Mountains copied from the scenario occupancy map.
    (31, 0, 5, 5),
    (0, 13, 5, 5),
    (25, 19, 5, 5),
    (20, 43, 5, 5),
    (35, 43, 5, 5),
    (43, 43, 5, 5),
    (45, 0, 3, 3),
    (19, 5, 3, 3),
    (33, 5, 3, 3),
    (3, 6, 3, 3),
    (0, 7, 3, 3),
    (11, 7, 3, 3),
    (8, 8, 3, 3),
    (0, 10, 3, 3),
    (19, 10, 3, 3),
    (36, 10, 3, 3),
    (9, 12, 3, 3),
    (45, 13, 3, 3),
    (34, 15, 3, 3),
    (31, 16, 3, 3),
    (0, 18, 3, 3),
    (7, 18, 3, 3),
    (16, 18, 3, 3),
    (19, 18, 3, 3),
    (22, 18, 3, 3),
    (30, 19, 3, 3),
    (0, 21, 3, 3),
    (10, 21, 3, 3),
    (17, 23, 3, 3),
    (20, 24, 3, 3),
    (27, 24, 3, 3),
    (23, 26, 3, 3),
    (29, 27, 3, 3),
    (16, 28, 3, 3),
    (45, 28, 3, 3),
    (18, 33, 3, 3),
    (39, 37, 3, 3),
    (33, 40, 3, 3),
    (36, 40, 3, 3),
    (42, 40, 3, 3),
    (0, 0, 2, 2),
    (24, 0, 2, 2),
    (26, 0, 2, 2),
    (43, 0, 2, 2),
    (20, 3, 2, 2),
    (46, 3, 2, 2),
    (6, 7, 2, 2),
    (20, 8, 2, 2),
    (35, 8, 2, 2),
    (46, 10, 2, 2),
    (12, 12, 2, 2),
    (5, 13, 2, 2),
    (7, 13, 2, 2),
    (21, 13, 2, 2),
    (8, 15, 2, 2),
    (37, 15, 2, 2),
    (43, 15, 2, 2),
    (15, 16, 2, 2),
    (3, 18, 2, 2),
    (5, 18, 2, 2),
    (18, 21, 2, 2),
    (6, 24, 2, 2),
    (46, 25, 2, 2),
    (26, 28, 2, 2),
    (38, 28, 2, 2),
    (40, 28, 2, 2),
    (42, 30, 2, 2),
    (17, 31, 2, 2),
    (46, 31, 2, 2),
    (19, 36, 2, 2),
    (21, 38, 2, 2),
    (31, 39, 2, 2),
    (40, 40, 2, 2),
    (46, 40, 2, 2),
    (0, 41, 2, 2),
    (16, 46, 2, 2),
    (33, 46, 2, 2),
    (38, 0, 1, 1),
    (31, 12, 1, 1),
    (30, 13, 1, 1),
    (11, 18, 1, 1),
    (9, 21, 1, 1),
    (3, 23, 1, 1),
    (4, 24, 1, 1),
    (5, 24, 1, 1),
    (8, 47, 1, 1),
    # Everything else occupied on the match-view map is treated as solid.
    (22, 8, 5, 5),
    (1, 25, 5, 5),
    (3, 1, 4, 4),
    (1, 37, 4, 4),
    (37, 6, 4, 4),
    (25, 30, 4, 4),
    (37, 22, 4, 4),
    (30, 22, 3, 3),
    (3, 20, 3, 3),
    (15, 9, 3, 3),
    (43, 34, 3, 3),
    (19, 29, 3, 3),
    (21, 34, 3, 3),
    (36, 1, 3, 3),
    (9, 42, 3, 3),
    (15, 1, 3, 3),
    (38, 30, 3, 3),
    (42, 2, 3, 3),
)
BASE_STATIC_EMPTY_TILES: Tuple[Tuple[int, int], ...] = (
    # Neutral map stacks that do not match any campaign enemy template stay empty.
    (7, 2),
    (19, 2),
    (39, 3),
    (2, 4),
    (7, 4),
    (16, 5),
    (25, 5),
    (45, 5),
    (14, 7),
    (29, 9),
    (7, 10),
    (41, 10),
    (32, 11),
    (18, 12),
    (40, 13),
    (29, 14),
    (23, 16),
    (28, 16),
    (40, 20),
    (41, 25),
    (26, 26),
    (35, 29),
    (37, 31),
    (30, 33),
    (41, 34),
    (33, 37),
    (44, 37),
    (4, 41),
    (28, 41),
    (30, 41),
    (41, 44),
    # Empire capital eastern inner tile mirrors the open Legions capital slot.
    (26, 10),
)

BASE_VILLAGE_HEAL_TILES: Tuple[Tuple[int, int], ...] = (
    (6, 4),
    (4, 40),
    (40, 9),
    (28, 33),
    (40, 25),
)
BASE_VILLAGE_PREVIOUS_CORNER_TILES: Tuple[Tuple[int, int], ...] = (
    (3, 1),
    (1, 37),
    (37, 6),
    (25, 30),
    (37, 22),
)
BASE_STATIC_HEAL_TILES: Tuple[Tuple[int, int], ...] = (
    DEFAULT_HERO_GRID_POSITION,
    *BASE_VILLAGE_HEAL_TILES,
)
BASE_SETTLEMENT_LEVEL_BY_HEAL_TILE: Dict[Tuple[int, int], int] = {
    (6, 4): 1,
    (4, 40): 1,
    (40, 9): 2,
    (28, 33): 2,
    (40, 25): 2,
}

LEGIONS_SETTLEMENT_TERRITORY_EXPANSION_BY_LEVEL: Dict[int, int] = {
    1: 15,
    2: 15,
    3: 20,
    4: 20,
    5: 25,
}
BASE_LEGIONS_SETTLEMENT_TERRITORY_DATA: Tuple[Dict[str, object], ...] = (
    {
        "name": "Стагириас",
        "source_tile": (6, 4),
        "required_enemy_ids": (32, 67),
    },
    {
        "name": "Порт Бирмингем",
        "source_tile": (4, 40),
        "required_enemy_ids": (22,),
    },
    {
        "name": "Кордилия",
        "source_tile": (40, 9),
        "required_enemy_ids": (33,),
    },
    {
        "name": "Порт Полонис",
        "source_tile": (28, 33),
        "required_enemy_ids": (34, 68),
    },
    {
        "name": "Соругирилла",
        "source_tile": (40, 25),
        "required_enemy_ids": (35, 69),
    },
)

SETTLEMENT_ARMOR_BONUS_BY_LEVEL: Dict[int, int] = {
    1: 10,
    2: 15,
    3: 20,
    4: 25,
    5: 30,
}
CAPITAL_HEAL_TILE_ARMOR_BONUS = 40
CAPITAL_HEAL_TILE_REST_BONUS = 50

SETTLEMENT_DEFENDER_HEAL_TILE_BY_ENEMY_ID: Dict[int, Tuple[int, int]] = {
    22: (4, 40),
    32: (6, 4),
    33: (40, 9),
    34: (28, 33),
    35: (40, 25),
    67: (6, 4),
    68: (28, 33),
    69: (40, 25),
}
SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID: Dict[int, int] = {
    enemy_id: BASE_SETTLEMENT_LEVEL_BY_HEAL_TILE[heal_tile]
    for enemy_id, heal_tile in SETTLEMENT_DEFENDER_HEAL_TILE_BY_ENEMY_ID.items()
}
HEAL_TILE_COUNT = len(BASE_STATIC_HEAL_TILES)

BASE_STATIC_CHESTS: Tuple[Tuple[Tuple[int, int], Tuple[str, ...]], ...] = (
    ((28, 46), ("Banner of War",)),
    ((36, 38), ("Potion of Healing", "Life Potion")),
    ((19, 0), ("Potion of Healing", "Life Potion")),
    ((43, 17), ("Potion of Strength",)),
    ((22, 21), ("Potion of Invulnerability",)),
    ((6, 10), ("Banner of Might",)),
    ((5, 17), ("Tome of Arcanum",)),
    ((22, 27), ("Vampire Orb",)),
    ((46, 6), ("Orb of Life",)),
    ((45, 40), ("Lich Orb",)),
    ((37, 22), ("Zombie Talisman",)),
    ((11, 11), ("Ruby (Valuable)",)),
    ((13, 19), ("Ruby (Valuable)",)),
    ((32, 41), ("Potion of Healing", "Diamond (Valuable)")),
    ((1, 32), ("Mind ward scroll",)),
    ((2, 3), ("Mind ward scroll",)),
    ((34, 10), ("Potion of Healing", "Bronze Ring (Valuable)")),
    ((24, 30), ("Potion of Healing", "Bronze Ring (Valuable)")),
)
CHEST_COUNT = len(BASE_STATIC_CHESTS)

BASE_STATIC_MANA_SOURCES: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    ("infernal", (6, 16)),
    ("infernal", (44, 21)),
    ("infernal", (10, 29)),
    ("life", (6, 11)),
    ("life", (25, 16)),
    ("life", (29, 40)),
    ("death", (7, 41)),
    ("runes", (1, 4)),
)
MANA_SOURCE_DISPLAY_SPECS: Dict[str, Dict[str, object]] = {
    "infernal": {
        "name": "Мана преисподней",
        "letter": "П",
        "fill_color": (0.84, 0.18, 0.16, 0.84),
        "edge_color": (0.46, 0.08, 0.06, 0.98),
        "text_color": (1.0, 0.96, 0.96, 1.0),
        "ansi_color": "\x1b[31m",
    },
    "life": {
        "name": "Мана жизни",
        "letter": "Ж",
        "fill_color": (0.18, 0.38, 0.88, 0.84),
        "edge_color": (0.06, 0.18, 0.46, 0.98),
        "text_color": (1.0, 0.98, 0.98, 1.0),
        "ansi_color": "\x1b[34m",
    },
    "death": {
        "name": "Мана смерти",
        "letter": "С",
        "fill_color": (0.72, 0.72, 0.72, 0.88),
        "edge_color": (0.36, 0.36, 0.36, 0.98),
        "text_color": (0.10, 0.10, 0.10, 1.0),
        "ansi_color": "\x1b[90m",
    },
    "runes": {
        "name": "Мана рун",
        "letter": "Р",
        "fill_color": (0.40, 0.84, 0.96, 0.84),
        "edge_color": (0.08, 0.44, 0.60, 0.98),
        "text_color": (0.06, 0.18, 0.24, 1.0),
        "ansi_color": "\x1b[96m",
    },
}
MANA_SOURCE_COUNT = len(BASE_STATIC_MANA_SOURCES)

# Base campaign layout mirrors the 48x48 scenario coordinates directly.
BASE_GRID_SIZE = DEFAULT_GRID_SIZE
BASE_ENEMY_POSITIONS: Dict[int, Tuple[int, int]] = {}

_UNIT_NAME_ALIASES: Dict[str, str] = {
    "Выверна": "Виверна",
    "Гоблин-лучник": "Гоблин лучник",
    "Зеленый дракон": "Зелёный дракон",
    "Зелёный дракон": "Зелёный дракон",
    "Дракон Рока": "Дракон рока",
    "Лорд Тьмы": "Лорд тьмы",
}


def _canonical_enemy_name(name: str) -> str:
    """Возвращает нормализованное имя юнита для сравнений внутри grid-конфига."""
    raw = str(name or "").strip()
    return _UNIT_NAME_ALIASES.get(raw, raw)


def _build_big_unit_names() -> frozenset[str]:
    """Собирает имена больших юнитов из DATA.

    В UNIT_DATA размер 1 означает large-unit: такой юнит занимает целую колонку
    переднего и заднего ряда в тактической сетке. Этот список нужен при упаковке
    enemy-team specs, чтобы случайно не поставить другого юнита в ту же колонку.
    """
    names: set[str] = set()
    for entry in DATA:
        if not isinstance(entry, dict):
            continue
        entry_name = entry.get("\u043a\u0442\u043e")
        if entry_name is None:
            entry_name = entry.get("\u0420\u0454\u0421\u201a\u0420\u0455")
        if not entry_name:
            continue
        try:
            size = int(entry.get("\u0440\u0430\u0437\u043c\u0435\u0440", 0) or 0)
        except (TypeError, ValueError):
            size = 0
        if size != 1:
            continue
        name = str(entry_name).strip()
        if not name:
            continue
        names.add(name)
        names.add(_canonical_enemy_name(name))
    return frozenset(names)


BIG_UNIT_NAMES = _build_big_unit_names()


def _is_big_enemy_unit(name: str | None) -> bool:
    """Проверяет, считается ли юнит большим для раскладки 3x2."""
    if not name:
        return False
    return _canonical_enemy_name(str(name)) in BIG_UNIT_NAMES


def _normalize_big_unit_rows(
    front: list[str | None],
    back: list[str | None],
) -> Tuple[list[str | None], list[str | None]]:
    """Освобождает колонку большого юнита в enemy-team spec.

    В бою большой юнит визуально занимает переднюю и заднюю ячейку одной колонки.
    Если в исходном описании рядом с ним стоит обычный юнит, пытаемся аккуратно
    перенести обычного юнита в ближайший свободный слот. Если места нет или
    конфигурация неоднозначная, возвращаем исходную раскладку без изменений.
    """
    if len(front) != 3 or len(back) != 3:
        return list(front), list(back)

    original_slots: list[tuple[str, int, str]] = []
    for row_name, row_values in (("front", front), ("back", back)):
        for col, name in enumerate(row_values):
            if isinstance(name, str) and name:
                original_slots.append((row_name, int(col), str(name)))

    if not original_slots:
        return list(front), list(back)

    big_slots = [
        (row_name, col, name)
        for row_name, col, name in original_slots
        if _is_big_enemy_unit(name)
    ]
    if not big_slots:
        return list(front), list(back)

    blocked_cols = [col for _, col, _ in big_slots]
    if len(set(blocked_cols)) != len(blocked_cols):
        return list(front), list(back)

    result_front: list[str | None] = [None, None, None]
    result_back: list[str | None] = [None, None, None]
    blocked_col_set = set(blocked_cols)

    for _, col, name in big_slots:
        result_front[col] = name

    displaced_units: list[tuple[str, int, str]] = []
    for row_name, col, name in original_slots:
        if _is_big_enemy_unit(name):
            continue
        target_row = result_front if row_name == "front" else result_back
        if col not in blocked_col_set and target_row[col] is None:
            target_row[col] = name
            continue
        displaced_units.append((row_name, col, name))

    def _candidate_slots(preferred_row: str, original_col: int) -> list[tuple[str, int]]:
        """Возвращает свободные слоты от ближайшего к исходной колонке к дальнему."""
        row_order = [preferred_row, "back" if preferred_row == "front" else "front"]
        candidates: list[tuple[str, int]] = []
        for row_name in row_order:
            target_row = result_front if row_name == "front" else result_back
            free_cols = [
                col
                for col in range(3)
                if col not in blocked_col_set and target_row[col] is None
            ]
            free_cols.sort(key=lambda col: (abs(col - original_col), col))
            candidates.extend((row_name, col) for col in free_cols)
        return candidates

    for preferred_row, original_col, name in displaced_units:
        placed = False
        for row_name, col in _candidate_slots(preferred_row, original_col):
            target_row = result_front if row_name == "front" else result_back
            if target_row[col] is None:
                target_row[col] = name
                placed = True
                break
        if not placed:
            return list(front), list(back)

    return result_front, result_back


def _normalize_enemy_team_spec(spec: dict[str, object]) -> dict[str, object]:
    """Нормализует одну запись ENEMY_TEAM_SPECS перед передачей в CampaignEnv."""
    front, back = _normalize_big_unit_rows(
        list(spec["front"]),
        list(spec["back"]),
    )
    return {
        "description": str(spec["description"]),
        "front": front,
        "back": back,
    }

# Battle-ready template for the hero squad used in campaign battles.
BLUE_TEAM_TEMPLATE = [
    {
        "name": "Одержимый",
        "initiative": 50,
        "initiative_base": 50,
        "team": "blue",
        "position": 7,
        "stand": "ahead",
        "unit_type": "Warrior",
        "damage": 25,
        "damage_secondary": 0,
        "health": 120,
        "max_health": 120,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 25,
        "exp_required": 95,
        "exp_current": 0,
        "turns_into": ["Берсерк"],
    },
    {
        "name": "Герцог",
        "initiative": 50,
        "initiative_base": 50,
        "team": "blue",
        "position": 8,
        "stand": "ahead",
        "unit_type": "Warrior",
        "damage": 50,
        "damage_secondary": 0,
        "health": 150,
        "max_health": 150,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 60,
        "exp_required": 150,
        "exp_current": 0,
        "turns_into": [],
    },
    {
        "name": "Одержимый",
        "initiative": 50,
        "initiative_base": 50,
        "team": "blue",
        "position": 9,
        "stand": "ahead",
        "unit_type": "Warrior",
        "damage": 25,
        "damage_secondary": 0,
        "health": 120,
        "max_health": 120,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 25,
        "exp_required": 95,
        "exp_current": 0,
        "turns_into": ["Берсерк"],
    },
    {
        "name": "пусто",
        "initiative": 0,
        "initiative_base": 0,
        "team": "blue",
        "position": 10,
        "stand": "behind",
        "unit_type": "Archer",
        "damage": 0,
        "damage_secondary": 0,
        "health": 0,
        "max_health": 0,
        "armor": 0,
        "accuracy": 0,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 0,
        "exp_required": 0,
        "exp_current": 0,
        "turns_into": [],
    },
    {
        "name": "Сектант",
        "initiative": 40,
        "initiative_base": 40,
        "team": "blue",
        "position": 11,
        "stand": "behind",
        "unit_type": "Mage",
        "damage": 15,
        "damage_secondary": 0,
        "health": 45,
        "max_health": 45,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Fire",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 20,
        "exp_required": 70,
        "exp_current": 0,
        "turns_into": ["Колдун", "Ведьма"],
    },
    {
        "name": "пусто",
        "initiative": 0,
        "initiative_base": 0,
        "team": "blue",
        "position": 12,
        "stand": "behind",
        "unit_type": "Archer",
        "damage": 0,
        "damage_secondary": 0,
        "health": 0,
        "max_health": 0,
        "armor": 0,
        "accuracy": 0,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 0,
        "exp_required": 0,
        "exp_current": 0,
        "turns_into": [],
    },
]

for _unit in BLUE_TEAM_TEMPLATE:
    _unit["hero"] = is_hero_name(_unit.get("name"))

# Real stacks from the 48x48 scenario that match campaign enemy compositions.
MATCHED_MAP_STACKS: tuple[dict[str, object], ...] = (
    {"enemy_id": 1, "position": (9, 7), "description": "Отряд: волк на передней линии", "front": [None, "Волк", None], "back": [None, None, None]},
    {"enemy_id": 2, "position": (12, 11), "description": "Отряд: людоед на передней линии", "front": [None, "Людоед", None], "back": [None, None, None]},
    {"enemy_id": 3, "position": (14, 15), "description": "Отряд: орк чемпион на передней линии", "front": [None, "Орк чемпион", None], "back": [None, None, None]},
    {"enemy_id": 4, "position": (5, 16), "description": "Отряд: гигантский паук на передней линии", "front": [None, "Гигантский паук", None], "back": [None, None, None]},
    {"enemy_id": 5, "position": (18, 17), "description": "Отряд: орк чемпион на передней линии", "front": [None, "Орк чемпион", None], "back": [None, None, None]},
    {"enemy_id": 6, "position": (10, 18), "description": "Отряд: тролль на передней линии", "front": [None, "Тролль", None], "back": [None, None, None]},
    {"enemy_id": 7, "position": (6, 22), "description": "Отряд: воин + привидение спереди, колдун + призрак + привидение сзади", "front": ["Воин", None, "Привидение"], "back": ["Колдун", "Призрак", "Привидение"]},
    {"enemy_id": 8, "position": (14, 23), "description": "Отряд: людоед на передней линии", "front": [None, "Людоед", None], "back": [None, None, None]},
    {"enemy_id": 9, "position": (24, 23), "description": "Отряд: тамплиер спереди, колдун и призрак сзади", "front": [None, "Тамплиер", None], "back": ["Колдун", None, "Призрак"]},
    {"enemy_id": 10, "position": (34, 24), "description": "Отряд: волк на передней линии", "front": [None, "Волк", None], "back": [None, None, None]},
    {"enemy_id": 11, "position": (34, 25), "description": "Отряд: оборотень спереди и тень сзади", "front": [None, "Оборотень", None], "back": [None, "Тень", None]},
    {"enemy_id": 12, "position": (7, 26), "description": "Отряд: 2 гоблина и гоблин лучник в центре передней линии", "front": ["Гоблин", "Гоблин лучник", "Гоблин"], "back": [None, None, None]},
    {"enemy_id": 13, "position": (31, 26), "description": "Отряд: 2 зомби спереди и привидение сзади", "front": ["Зомби", None, "Зомби"], "back": [None, "Привидение", None]},
    {"enemy_id": 14, "position": (18, 27), "description": "Отряд: орк чемпион на передней линии", "front": [None, "Орк чемпион", None], "back": [None, None, None]},
    {"enemy_id": 15, "position": (22, 28), "description": "Отряд: 2 привидения спереди, вампир сзади", "front": ["Привидение", None, "Привидение"], "back": [None, "Вампир", None]},
    {"enemy_id": 16, "position": (4, 31), "description": "Отряд: 3 гоблина на передней линии", "front": ["Гоблин", "Гоблин", "Гоблин"], "back": [None, None, None]},
    {"enemy_id": 17, "position": (22, 31), "description": "Отряд: воин спереди, 2 призрака и привидение сзади", "front": [None, "Воин", None], "back": ["Призрак", "Привидение", "Призрак"]},
    {"enemy_id": 18, "position": (14, 32), "description": "Отряд: людоед на передней линии", "front": [None, "Людоед", None], "back": [None, None, None]},
    {"enemy_id": 19, "position": (45, 32), "description": "Отряд: зомби спереди, привидение сзади", "front": [None, "Зомби", None], "back": [None, "Привидение", None]},
    {"enemy_id": 20, "position": (29, 33), "description": "Отряд: 2 воина спереди, призрак и привидение сзади", "front": ["Воин", None, "Воин"], "back": ["Призрак", None, "Привидение"]},
    {"enemy_id": 21, "position": (25, 35), "description": "Отряд: зомби спереди, привидение сзади", "front": [None, "Зомби", None], "back": [None, "Привидение", None]},
    {"enemy_id": 22, "position": (1, 37), "description": "Отряд: титан и 2 скваера на передней линии", "front": ["Скваер", "Титан", "Скваер"], "back": [None, None, None]},
    {"enemy_id": 23, "position": (10, 37), "description": "Отряд: волк на передней линии", "front": [None, "Волк", None], "back": [None, None, None]},
    {"enemy_id": 24, "position": (14, 37), "description": "Отряд: волк спереди и волк сзади", "front": [None, "Волк", None], "back": [None, "Волк", None]},
    {"enemy_id": 25, "position": (10, 40), "description": "Отряд: копейщик спереди и служка сзади", "front": [None, "Копейщик", None], "back": [None, "Служка", None]},
    {"enemy_id": 26, "position": (20, 40), "description": "Отряд: орк в центре передней линии и гоблин лучник на задней линии", "front": [None, "Орк", None], "back": [None, "Гоблин лучник", None]},
    {"enemy_id": 27, "position": (9, 41), "description": "Отряд: скваер в центре и 2 крестьянина на передней линии", "front": ["Крестьянин", "Скваер", "Крестьянин"], "back": [None, None, None]},
    {"enemy_id": 28, "position": (14, 42), "description": "Отряд: орк в центре и 2 гоблина на передней линии", "front": ["Гоблин", "Орк", "Гоблин"], "back": [None, None, None]},
    {"enemy_id": 29, "position": (7, 43), "description": "Отряд: 3 скваера на передней линии", "front": ["Скваер", "Скваер", "Скваер"], "back": [None, None, None]},
    {"enemy_id": 30, "position": (18, 44), "description": "Отряд: орк чемпион на передней линии", "front": [None, "Орк чемпион", None], "back": [None, None, None]},
    {"enemy_id": 31, "position": (30, 46), "description": "Отряд: зеленый дракон", "front": [None, "Зелёный дракон", None], "back": [None, None, None]},
)
VILLAGE_LINKED_STACK_OVERRIDES: tuple[dict[str, object], ...] = (
    {
        "enemy_id": 22,
        "position": (4, 40),
        "description": "Отряд Порта Бирмингем: титан и 2 скваера",
        "front": ["Скваер", "Титан", "Скваер"],
        "back": [None, None, None],
    },
    {
        "enemy_id": 32,
        "position": (6, 4),
        "description": "Отряд Стагириаса: Сегерик (Королевский страж), Холмовой великан, Гном",
        "front": ["Холмовой гигант", "Королевский страж", "Гном"],
        "back": [None, None, None],
    },
    {
        "enemy_id": 33,
        "position": (40, 9),
        "description": "Отряд Кордилии: Тифас (Рыцарь смерти), Злой дух, Призрак, Привидение",
        "front": ["Дух", "Рыцарь смерти", "Призрак"],
        "back": [None, "Привидение", None],
    },
    {
        "enemy_id": 34,
        "position": (28, 33),
        "description": "Отряд Порта Полонис: Воин спереди, Колдун, Нигош (Носферату) и Привидение сзади",
        "front": [None, "Воин", None],
        "back": ["Колдун", "Носферату", "Привидение"],
    },
    {
        "enemy_id": 35,
        "position": (40, 25),
        "description": "Отряд Соругириллы: Унфегра (Королева личей), Виверна",
        "front": [None, "Виверна", None],
        "back": ["Королева лич", None, None],
    },
)


def _pack_stack_units(units: Tuple[str, ...]) -> Tuple[list[str | None], list[str | None]]:
    """Упаковывает список до 6 имен в front/back ряды формата BattleEnv.

    Правила намеренно простые и детерминированные: одиночный юнит идет в центр,
    два юнита - в центр переднего и заднего ряда, более крупные стеки занимают
    слоты слева направо. Затем вызывающий код отдельно нормализует large-unit.
    """
    unit_list = [str(name) for name in units if isinstance(name, str) and name][:6]
    slots: list[str | None] = [None, None, None, None, None, None]
    count = len(unit_list)

    if count <= 0:
        return list(slots[:3]), list(slots[3:])
    if count == 1:
        slots[1] = unit_list[0]
    elif count == 2:
        slots[1] = unit_list[0]
        slots[4] = unit_list[1]
    elif count == 3:
        slots[:3] = unit_list
    elif count == 4:
        slots[:3] = unit_list[:3]
        slots[4] = unit_list[3]
    elif count == 5:
        slots[:3] = unit_list[:3]
        slots[3] = unit_list[3]
        slots[5] = unit_list[4]
    else:
        slots = unit_list[:6]

    return list(slots[:3]), list(slots[3:])


def _make_stack_override(
    enemy_id: int,
    position: Tuple[int, int],
    units: Tuple[str, ...],
    *,
    label: Optional[str] = None,
) -> dict[str, object]:
    """Создает override-запись для карты по краткому списку юнитов."""
    front, back = _pack_stack_units(units)
    front, back = _normalize_big_unit_rows(front, back)
    composition = ", ".join(units) if units else "(empty)"
    return {
        "enemy_id": int(enemy_id),
        "position": tuple(position),
        "description": str(label) if label else f"Отряд: {composition}",
        "front": front,
        "back": back,
    }


VILLAGE_INTERNAL_GARRISON_OVERRIDES: tuple[dict[str, object], ...] = (
    {
        "enemy_id": 67,
        "position": (6, 4),
        "description": "Внутренний гарнизон Стагириаса: Хранитель кузни",
        "front": [None, "Хранитель кузни", None],
        "back": [None, None, None],
    },
    {
        "enemy_id": 68,
        "position": (28, 33),
        "description": "Внутренний гарнизон Порта Полонис: Зомби, Воин",
        "front": [None, "Зомби", None],
        "back": [None, "Воин", None],
    },
    {
        "enemy_id": 69,
        "position": (40, 25),
        "description": "Внутренний гарнизон Соругириллы: Зомби, Воин",
        "front": [None, "Зомби", None],
        "back": [None, "Воин", None],
    },
)

CAPITAL_INTERNAL_STACK_OVERRIDES: tuple[dict[str, object], ...] = (
    {
        "enemy_id": 75,
        "position": (26, 10),
        "description": "Страж столицы Империи: Мизраэль",
        "front": [None, None, None],
        "back": [None, "Мизраэль", None],
    },
)


RUIN_STACK_OVERRIDES: tuple[dict[str, object], ...] = (
    _make_stack_override(
        70,
        (17, 11),
        ("Орк", "Гоблин", "Гоблин"),
        label="Руины (15, 9): Орк, Гоблин, Гоблин",
    ),
    _make_stack_override(
        71,
        (5, 22),
        ("Привидение", "Привидение", "Виверна"),
        label="Руины (3, 20): Привидение, Привидение, Виверна",
    ),
    _make_stack_override(
        72,
        (32, 24),
        ("Привидение", "Привидение", "Некромант", "Зомби"),
        label="Руины (30, 22): Привидение, Привидение, Некромант, Зомби",
    ),
    _make_stack_override(
        73,
        (21, 31),
        ("Колдун", "Воин"),
        label="Руины (19, 29): Колдун, Воин",
    ),
    _make_stack_override(
        74,
        (45, 36),
        ("Лорд Тьмы", "Лич"),
        label="Руины (43, 34): Лорд Тьмы, Лич",
    ),
)


ADDITIONAL_MAP_STACK_OVERRIDES: tuple[dict[str, object], ...] = (
    _make_stack_override(36, (7, 2), ("Холмовой гигант",)),
    _make_stack_override(37, (19, 2), ("Гном", "Травница", "Алхимик")),
    _make_stack_override(38, (39, 3), ("Призрак", "Колдун", "Воин")),
    _make_stack_override(39, (2, 4), ("Волк", "Варвар")),
    _make_stack_override(40, (7, 4), ("Гном", "Метатель топоров", "Травница")),
    _make_stack_override(41, (16, 5), ("Гном", "Метатель топоров")),
    _make_stack_override(42, (25, 5), ("Разбойник",)),
    _make_stack_override(43, (45, 5), ("Призрак", "Тамплиер")),
    _make_stack_override(44, (14, 7), ("Гном", "Гном")),
    _make_stack_override(45, (29, 9), ("Головорез",)),
    _make_stack_override(46, (7, 10), ("Гигантский чёрный паук",)),
    _make_stack_override(47, (41, 10), ("Призрак", "Привидение", "Привидение", "Воин")),
    _make_stack_override(48, (32, 11), ("Жрец", "Копейщик")),
    _make_stack_override(49, (18, 12), ("Гном",)),
    _make_stack_override(50, (40, 13), ("Зомби",)),
    _make_stack_override(51, (29, 14), ("Крестьянин", "Крестьянин", "Крестьянин")),
    _make_stack_override(52, (23, 16), ("Орк",)),
    _make_stack_override(53, (28, 16), ("Скваер", "Ополченец")),
    _make_stack_override(54, (40, 20), ("Дракон Рока",)),
    _make_stack_override(
        55,
        (41, 25),
        ("Скелет рыцарь", "Привидение", "Привидение", "Привидение", "Привидение", "Привидение"),
    ),
    _make_stack_override(56, (26, 26), ("Призрак", "Воин")),
    _make_stack_override(57, (35, 29), ("Призрак", "Привидение", "Воин")),
    _make_stack_override(58, (37, 31), ("Зомби",)),
    _make_stack_override(59, (30, 33), ("Зомби",)),
    _make_stack_override(60, (41, 34), ("Зомби",)),
    _make_stack_override(61, (33, 37), ("Зомби", "Призрак")),
    _make_stack_override(62, (44, 37), ("Призрак", "Привидение", "Зомби", "Воин")),
    _make_stack_override(63, (4, 41), ("Разбойник",)),
    _make_stack_override(64, (28, 41), ("Лесной эльф", "Рейнджер")),
    _make_stack_override(65, (30, 41), ("Кентавр копейщик", "Рейнджер")),
    _make_stack_override(66, (41, 44), ("Демон", "Привидение", "Привидение")),
)

ALL_MAP_STACK_OVERRIDES: tuple[dict[str, object], ...] = (
    VILLAGE_LINKED_STACK_OVERRIDES
    + VILLAGE_INTERNAL_GARRISON_OVERRIDES
    + CAPITAL_INTERNAL_STACK_OVERRIDES
    + RUIN_STACK_OVERRIDES
    + ADDITIONAL_MAP_STACK_OVERRIDES
)

BASE_ENEMY_POSITIONS = {
    int(row["enemy_id"]): tuple(row["position"])
    for row in MATCHED_MAP_STACKS
}
BASE_ENEMY_POSITIONS.update(
    {
        int(row["enemy_id"]): tuple(row["position"])
        for row in ALL_MAP_STACK_OVERRIDES
    }
)

ENEMY_TEAM_SPECS: dict[int, dict[str, object]] = {
    int(row["enemy_id"]): {
        "description": str(row["description"]),
        "front": list(row["front"]),
        "back": list(row["back"]),
    }
    for row in MATCHED_MAP_STACKS
}
ENEMY_TEAM_SPECS.update(
    {
        int(row["enemy_id"]): {
            "description": str(row["description"]),
            "front": list(row["front"]),
            "back": list(row["back"]),
        }
        for row in ALL_MAP_STACK_OVERRIDES
    }
)
ENEMY_TEAM_SPECS = {
    enemy_id: _normalize_enemy_team_spec(spec)
    for enemy_id, spec in ENEMY_TEAM_SPECS.items()
}


def clamp_grid_pos(pos: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    """Clamp a coordinate to the current grid bounds."""
    x = int(pos[0])
    y = int(pos[1])
    return (
        max(0, min(int(grid_size) - 1, x)),
        max(0, min(int(grid_size) - 1, y)),
    )


def scale_enemy_positions(
    target_grid_size: int,
    *,
    base_grid_size: int = BASE_GRID_SIZE,
    base_enemy_positions: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, Tuple[int, int]]:
    """Scale the base campaign enemy layout to an arbitrary grid size.

    Масштабирование сохраняет относительное положение enemy stacks на карте.
    Для базового размера 48x48 координаты возвращаются как есть, чтобы не было
    дрейфа из-за округления.
    """
    if target_grid_size <= 1:
        raise ValueError("target_grid_size must be >= 2")

    base_positions = (
        dict(base_enemy_positions)
        if base_enemy_positions is not None
        else dict(BASE_ENEMY_POSITIONS)
    )
    if target_grid_size == base_grid_size:
        return dict(base_positions)

    base_span = int(base_grid_size) - 1
    target_span = int(target_grid_size) - 1
    scale = float(target_span) / float(base_span)
    scaled_positions: Dict[int, Tuple[int, int]] = {}

    for enemy_id, (x, y) in base_positions.items():
        sx = int(round(x * scale))
        sy = int(round(y * scale))
        scaled_positions[int(enemy_id)] = (
            max(0, min(target_span, sx)),
            max(0, min(target_span, sy)),
        )

    return scaled_positions


def scale_static_tiles(
    target_grid_size: int,
    base_tiles: Tuple[Tuple[int, int], ...],
    *,
    base_grid_size: int = BASE_GRID_SIZE,
) -> Tuple[Tuple[int, int], ...]:
    """Scale a static set of map tiles to the current grid size.

    Используется для heal tiles, сундуков, маны, пустых клеток и obstacle-blocks.
    Порядок входных координат сохраняется, потому что некоторые callers связывают
    масштабированные клетки с исходными записями через zip().
    """
    if target_grid_size <= 1:
        raise ValueError("target_grid_size must be >= 2")

    if target_grid_size == base_grid_size:
        return tuple(clamp_grid_pos(tile, target_grid_size) for tile in base_tiles)

    base_span = int(base_grid_size) - 1
    target_span = int(target_grid_size) - 1
    scale = float(target_span) / float(base_span)
    scaled_tiles: list[Tuple[int, int]] = []

    for x, y in base_tiles:
        sx = int(round(x * scale))
        sy = int(round(y * scale))
        scaled_tiles.append(
            (
                max(0, min(target_span, sx)),
                max(0, min(target_span, sy)),
            )
        )

    return tuple(scaled_tiles)


def _expand_static_blocks(
    blocks: Tuple[Tuple[int, int, int, int], ...],
) -> Tuple[Tuple[int, int], ...]:
    """Expand ordered rectangle footprints into ordered individual tiles.

    Прямоугольники могут пересекаться; seen_tiles убирает дубли, но сохраняет
    первый порядок обхода, чтобы obstacle generation оставалась стабильной.
    """
    expanded_tiles: list[Tuple[int, int]] = []
    seen_tiles: set[Tuple[int, int]] = set()

    for x, y, width, height in blocks:
        for dy in range(int(height)):
            for dx in range(int(width)):
                tile = (int(x) + dx, int(y) + dy)
                if tile in seen_tiles:
                    continue
                seen_tiles.add(tile)
                expanded_tiles.append(tile)

    return tuple(expanded_tiles)


def are_targets_reachable(
    grid_size: int,
    start_position: Tuple[int, int],
    blocked_tiles: set[Tuple[int, int]],
    targets: set[Tuple[int, int]],
) -> bool:
    """Check that the start can still reach every target tile.

    Обход 8-направленный, как и движение на campaign map. Функция используется
    при генерации препятствий: новый obstacle принимается только если он не
    отрезает старт, врагов, heal tiles и другие защищенные клетки.
    """
    start_tile = clamp_grid_pos(start_position, grid_size)
    if start_tile in blocked_tiles:
        return False

    pending_targets = {
        clamp_grid_pos(tile, grid_size)
        for tile in targets
        if clamp_grid_pos(tile, grid_size) not in blocked_tiles
    }
    if not pending_targets:
        return True

    visited = {start_tile}
    frontier = deque([start_tile])
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

    while frontier and pending_targets:
        x, y = frontier.popleft()
        pending_targets.discard((x, y))

        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < int(grid_size) and 0 <= ny < int(grid_size)):
                continue
            tile = (nx, ny)
            if tile in blocked_tiles or tile in visited:
                continue
            visited.add(tile)
            frontier.append(tile)

    return not pending_targets


def _nearest_free_tile(
    anchor: Tuple[int, int],
    blocked_tiles: set[Tuple[int, int]],
    grid_size: int,
) -> Optional[Tuple[int, int]]:
    """Находит ближайшую свободную клетку к anchor.

    Tie-break идет по координате tile, чтобы при одинаковой дистанции результат
    был стабильным между запусками.
    """
    best_tile: Optional[Tuple[int, int]] = None
    best_distance: Optional[int] = None
    ax, ay = clamp_grid_pos(anchor, grid_size)

    for y in range(grid_size):
        for x in range(grid_size):
            tile = (x, y)
            if tile in blocked_tiles:
                continue
            distance = abs(x - ax) + abs(y - ay)
            if (
                best_distance is None
                or distance < best_distance
                or (distance == best_distance and tile < best_tile)
            ):
                best_tile = tile
                best_distance = distance

    return best_tile


def _farthest_free_tile(
    blocked_tiles: set[Tuple[int, int]],
    reference_tiles: Tuple[Tuple[int, int], ...],
    grid_size: int,
) -> Optional[Tuple[int, int]]:
    """Находит свободную клетку, максимально удаленную от reference_tiles.

    Используется как fallback для дополнительных heal tiles: сначала максимизируем
    минимальную дистанцию до уже выбранных точек, затем суммарную дистанцию.
    """
    best_tile: Optional[Tuple[int, int]] = None
    best_score: Optional[Tuple[int, int, int, int]] = None

    for y in range(grid_size):
        for x in range(grid_size):
            tile = (x, y)
            if tile in blocked_tiles:
                continue

            if reference_tiles:
                distances = [
                    (x - rx) * (x - rx) + (y - ry) * (y - ry)
                    for rx, ry in reference_tiles
                ]
                min_distance = min(distances)
                total_distance = sum(distances)
            else:
                min_distance = 0
                total_distance = 0

            score = (min_distance, total_distance, -y, -x)
            if best_score is None or score > best_score:
                best_tile = tile
                best_score = score

    return best_tile


def resolve_heal_tiles(
    grid_size: int,
    enemy_positions: Dict[int, Tuple[int, int]],
    *,
    start_position: Tuple[int, int] = DEFAULT_HERO_GRID_POSITION,
    heal_tile_count: int = HEAL_TILE_COUNT,
) -> Tuple[Tuple[int, int], ...]:
    """Use the fixed scenario heal tiles first, then fill extra slots deterministically.

    Порядок важен: первые клетки соответствуют старту героя и поселениям. Если
    heal_tile_count больше статического списка, оставшиеся клетки добираются
    около углов, а затем максимально далеко от уже выбранных heal tiles.
    """
    desired_count = max(0, int(heal_tile_count))
    if desired_count == 0:
        return ()

    heal_tiles: list[Tuple[int, int]] = []
    heal_seen: set[Tuple[int, int]] = set()
    static_heal_tiles = scale_static_tiles(
        grid_size,
        BASE_STATIC_HEAL_TILES,
        base_grid_size=BASE_GRID_SIZE,
    )

    for raw_tile in static_heal_tiles:
        tile = clamp_grid_pos(raw_tile, grid_size)
        if tile in heal_seen:
            continue
        heal_tiles.append(tile)
        heal_seen.add(tile)
        if len(heal_tiles) >= desired_count:
            return tuple(heal_tiles)

    enemy_tiles = {tuple(pos) for pos in enemy_positions.values()}

    if grid_size >= 4:
        margin = 1
    else:
        margin = 0
    far_edge = max(0, int(grid_size) - 1 - margin)
    near_edge = margin

    preferred_anchors = (
        clamp_grid_pos(start_position, grid_size),
        (far_edge, near_edge),
        (near_edge, far_edge),
        (far_edge, far_edge),
    )

    blocked_tiles = set(enemy_tiles)
    blocked_tiles.update(heal_tiles)

    for anchor in preferred_anchors:
        tile = _nearest_free_tile(anchor, blocked_tiles, grid_size)
        if tile is None:
            continue
        heal_tiles.append(tile)
        blocked_tiles.add(tile)
        if len(heal_tiles) >= desired_count:
            break

    while len(heal_tiles) < desired_count:
        tile = _farthest_free_tile(blocked_tiles, tuple(heal_tiles), grid_size)
        if tile is None:
            break
        heal_tiles.append(tile)
        blocked_tiles.add(tile)

    return tuple(heal_tiles)


def resolve_legions_settlement_territory_sources(
    grid_size: int,
    enemy_positions: Dict[int, Tuple[int, int]],
    *,
    base_settlement_territory_data: Tuple[Dict[str, object], ...] = (
        BASE_LEGIONS_SETTLEMENT_TERRITORY_DATA
    ),
    base_grid_size: int = BASE_GRID_SIZE,
) -> Tuple[Dict[str, object], ...]:
    """Resolve static settlement territory sources to the current grid size.

    Источник территории поселения должен совпадать с живым защитным enemy stack,
    если такой стек есть на карте. Если enemy_id отсутствует, используем
    масштабированную статическую клетку поселения.
    """
    resolved_sources: list[Dict[str, object]] = []
    for raw_entry in base_settlement_territory_data:
        entry = dict(raw_entry or {})
        required_enemy_ids = tuple(
            int(enemy_id)
            for enemy_id in tuple(entry.get("required_enemy_ids", ()))
        )
        base_source_tile = tuple(entry.get("source_tile", (0, 0)))
        scenario_settlement_level = max(
            1,
            int(BASE_SETTLEMENT_LEVEL_BY_HEAL_TILE.get(base_source_tile, 1) or 1),
        )
        source_tile = None
        for enemy_id in required_enemy_ids:
            if enemy_id in enemy_positions:
                source_tile = clamp_grid_pos(enemy_positions[enemy_id], grid_size)
                break
        if source_tile is None:
            source_tile = scale_static_tiles(
                grid_size,
                (base_source_tile,),
                base_grid_size=base_grid_size,
            )[0]
        resolved_sources.append(
            {
                "name": str(entry.get("name", "") or ""),
                "source_tile": tuple(source_tile),
                "required_enemy_ids": required_enemy_ids,
                "territory_level": scenario_settlement_level,
                "expansion_per_turn": max(
                    0,
                    int(
                        LEGIONS_SETTLEMENT_TERRITORY_EXPANSION_BY_LEVEL.get(
                            scenario_settlement_level,
                            0,
                        )
                        or 0
                    ),
                ),
                "settlement_level": int(
                    BASE_SETTLEMENT_LEVEL_BY_HEAL_TILE.get(base_source_tile, 0)
                    or 0
                ),
            }
        )
    return tuple(resolved_sources)


def resolve_chests(
    grid_size: int,
    enemy_positions: Dict[int, Tuple[int, int]],
    *,
    heal_tiles: Tuple[Tuple[int, int], ...] | list[Tuple[int, int]] = (),
    obstacle_tiles: Tuple[Tuple[int, int], ...] | list[Tuple[int, int]] | set[Tuple[int, int]] = (),
    start_position: Tuple[int, int] = DEFAULT_HERO_GRID_POSITION,
    base_static_chests: Tuple[Tuple[Tuple[int, int], Tuple[str, ...]], ...] = BASE_STATIC_CHESTS,
) -> Dict[Tuple[int, int], Tuple[str, ...]]:
    """Resolve deterministic treasure chest tiles and the items stored in each chest.

    Сундук не должен появиться на старте, враге, heal tile или obstacle. Если
    статическая координата занята после масштабирования, переносим его на
    ближайшую свободную клетку.
    """
    if not base_static_chests:
        return {}

    base_tiles = tuple(pos for pos, _ in base_static_chests)
    scaled_tiles = scale_static_tiles(
        grid_size,
        base_tiles,
        base_grid_size=BASE_GRID_SIZE,
    )
    reserved_tiles = {tuple(pos) for pos in enemy_positions.values()}
    reserved_tiles.update(tuple(tile) for tile in heal_tiles)
    reserved_tiles.update(clamp_grid_pos(tile, grid_size) for tile in obstacle_tiles)
    reserved_tiles.add(clamp_grid_pos(start_position, grid_size))

    resolved: Dict[Tuple[int, int], Tuple[str, ...]] = {}
    for raw_entry, raw_tile in zip(base_static_chests, scaled_tiles):
        _, item_names = raw_entry
        tile = clamp_grid_pos(raw_tile, grid_size)
        if tile in reserved_tiles or tile in resolved:
            tile = _nearest_free_tile(
                tile,
                reserved_tiles | set(resolved.keys()),
                grid_size,
            )
            if tile is None:
                continue
        resolved[tile] = tuple(str(item_name) for item_name in item_names if str(item_name))
        reserved_tiles.add(tile)

    return resolved


def resolve_mana_sources(
    grid_size: int,
    *,
    reserved_tiles: (
        Tuple[Tuple[int, int], ...]
        | list[Tuple[int, int]]
        | set[Tuple[int, int]]
    ) = (),
    base_static_mana_sources: Tuple[Tuple[str, Tuple[int, int]], ...] = BASE_STATIC_MANA_SOURCES,
    base_grid_size: int = BASE_GRID_SIZE,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Scale the static mana-source layout to the current grid size.

    Возвращается уже UI-ready dict: CampaignEnv/renderer могут брать отсюда
    русское имя, букву, цвета и ANSI-код без повторного lookup. Если источник
    попадает на занятую клетку или поверх другого источника, переносим его на
    ближайшую свободную клетку.
    """
    if not base_static_mana_sources:
        return {}

    scaled_tiles = scale_static_tiles(
        grid_size,
        tuple(position for _, position in base_static_mana_sources),
        base_grid_size=base_grid_size,
    )
    blocked_tiles = {clamp_grid_pos(tile, grid_size) for tile in reserved_tiles}
    resolved: Dict[Tuple[int, int], Dict[str, object]] = {}
    for (kind, _), scaled_tile in zip(base_static_mana_sources, scaled_tiles):
        tile = clamp_grid_pos(scaled_tile, grid_size)
        if tile in blocked_tiles or tile in resolved:
            tile = _nearest_free_tile(
                tile,
                blocked_tiles | set(resolved.keys()),
                grid_size,
            )
            if tile is None:
                continue
        display = dict(MANA_SOURCE_DISPLAY_SPECS.get(kind, {}))
        resolved[tile] = {
            "kind": str(kind),
            "name": str(display.get("name", kind) or kind),
            "letter": str(display.get("letter", "?") or "?")[:1],
            "fill_color": tuple(display.get("fill_color", (0.8, 0.8, 0.8, 0.8))),
            "edge_color": tuple(display.get("edge_color", (0.3, 0.3, 0.3, 1.0))),
            "text_color": tuple(display.get("text_color", (0.0, 0.0, 0.0, 1.0))),
            "ansi_color": str(display.get("ansi_color", "") or ""),
        }
        blocked_tiles.add(tile)
    return resolved


@lru_cache(maxsize=64)
def _resolve_obstacle_tiles_cached(
    grid_size: int,
    enemy_tiles: Tuple[Tuple[int, int], ...],
    heal_tiles: Tuple[Tuple[int, int], ...],
    start_position: Tuple[int, int],
    base_static_obstacle_blocks: Tuple[Tuple[int, int, int, int], ...],
    base_static_empty_tiles: Tuple[Tuple[int, int], ...],
) -> Tuple[Tuple[int, int], ...]:
    """Cached implementation for immutable obstacle-layout inputs."""
    start_tile = clamp_grid_pos(start_position, grid_size)
    reserved_tiles = set(enemy_tiles)
    reserved_tiles.update(heal_tiles)
    reserved_tiles.add(start_tile)
    reserved_tiles.update(
        scale_static_tiles(
            grid_size,
            base_static_empty_tiles,
            base_grid_size=BASE_GRID_SIZE,
        )
    )
    protected_targets = set(reserved_tiles)

    static_obstacle_tiles = scale_static_tiles(
        grid_size,
        _expand_static_blocks(base_static_obstacle_blocks),
        base_grid_size=BASE_GRID_SIZE,
    )

    obstacle_tiles: list[Tuple[int, int]] = []
    obstacle_seen: set[Tuple[int, int]] = set()
    for raw_tile in static_obstacle_tiles:
        tile = clamp_grid_pos(raw_tile, grid_size)
        if tile in reserved_tiles or tile in obstacle_seen:
            continue

        candidate_blocked_tiles = set(obstacle_seen)
        candidate_blocked_tiles.add(tile)
        if not are_targets_reachable(
            grid_size,
            start_tile,
            candidate_blocked_tiles,
            protected_targets,
        ):
            continue
        obstacle_tiles.append(tile)
        obstacle_seen.add(tile)

    return tuple(obstacle_tiles)


def resolve_obstacle_tiles(
    grid_size: int,
    enemy_positions: Dict[int, Tuple[int, int]],
    *,
    heal_tiles: Tuple[Tuple[int, int], ...] | list[Tuple[int, int]] = (),
    start_position: Tuple[int, int] = DEFAULT_HERO_GRID_POSITION,
    base_static_obstacle_blocks: Tuple[Tuple[int, int, int, int], ...] = (
        BASE_STATIC_OBSTACLE_BLOCKS
    ),
    base_static_empty_tiles: Tuple[Tuple[int, int], ...] = BASE_STATIC_EMPTY_TILES,
) -> Tuple[Tuple[int, int], ...]:
    """Build deterministic obstacle tiles for the campaign map.

    Obstacles строятся только из статических rectangle blocks сценария. Каждый
    candidate проверяется на reachability, чтобы важные клетки не оказались
    отрезаны от стартовой позиции. Поскольку раскладка детерминирована,
    одинаковые immutable-входы между экземплярами CampaignEnv используют общий
    LRU-кэш.
    """
    normalized_enemy_tiles = tuple(
        sorted(
            {
                (int(pos[0]), int(pos[1]))
                for pos in enemy_positions.values()
            }
        )
    )
    normalized_heal_tiles = tuple(
        sorted(
            {
                (int(tile[0]), int(tile[1]))
                for tile in heal_tiles
            }
        )
    )
    normalized_start_position = (int(start_position[0]), int(start_position[1]))
    normalized_obstacle_blocks = tuple(
        (int(x), int(y), int(width), int(height))
        for x, y, width, height in base_static_obstacle_blocks
    )
    normalized_empty_tiles = tuple(
        (int(tile[0]), int(tile[1]))
        for tile in base_static_empty_tiles
    )
    return _resolve_obstacle_tiles_cached(
        int(grid_size),
        normalized_enemy_tiles,
        normalized_heal_tiles,
        normalized_start_position,
        normalized_obstacle_blocks,
        normalized_empty_tiles,
    )


# Preserve the familiar functools cache controls on the public wrapper. They are
# useful for deterministic tests and diagnostics without exposing the private helper.
resolve_obstacle_tiles.cache_info = _resolve_obstacle_tiles_cached.cache_info
resolve_obstacle_tiles.cache_clear = _resolve_obstacle_tiles_cached.cache_clear
