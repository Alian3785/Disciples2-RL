# maps/default.py
"""Полная кампанейская карта 48x48 «A Return To Simpler Times».

Данные перенесены из grid.py и campaign_env_data.py: этот модуль — единственный
источник содержимого карты по умолчанию. grid.py реэкспортирует эти имена для
обратной совместимости.
"""

from __future__ import annotations

from typing import Dict, Tuple

from campaign_env_scenario import (
    _load_campaign_base_forest_tiles,
    _load_campaign_base_road_tiles,
    _load_campaign_base_water_tiles,
    _load_legions_territory_base_forbidden_tiles,
    _load_legions_territory_base_gold_mine_tiles,
)
from maps.base import MapConfig, make_stack_override

GRID_SIZE = 48
HERO_START: Tuple[int, int] = (5, 27)

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
    HERO_START,
    *BASE_VILLAGE_HEAL_TILES,
)
BASE_SETTLEMENT_LEVEL_BY_HEAL_TILE: Dict[Tuple[int, int], int] = {
    (6, 4): 1,
    (4, 40): 1,
    (40, 9): 2,
    (28, 33): 2,
    (40, 25): 2,
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
MANA_SOURCE_COUNT = len(BASE_STATIC_MANA_SOURCES)

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
    make_stack_override(
        70,
        (17, 11),
        ("Орк", "Гоблин", "Гоблин"),
        label="Руины (15, 9): Орк, Гоблин, Гоблин",
    ),
    make_stack_override(
        71,
        (5, 22),
        ("Привидение", "Привидение", "Виверна"),
        label="Руины (3, 20): Привидение, Привидение, Виверна",
    ),
    make_stack_override(
        72,
        (32, 24),
        ("Привидение", "Привидение", "Некромант", "Зомби"),
        label="Руины (30, 22): Привидение, Привидение, Некромант, Зомби",
    ),
    make_stack_override(
        73,
        (21, 31),
        ("Колдун", "Воин"),
        label="Руины (19, 29): Колдун, Воин",
    ),
    make_stack_override(
        74,
        (45, 36),
        ("Лорд Тьмы", "Лич"),
        label="Руины (43, 34): Лорд Тьмы, Лич",
    ),
)


ADDITIONAL_MAP_STACK_OVERRIDES: tuple[dict[str, object], ...] = (
    make_stack_override(36, (7, 2), ("Холмовой гигант",)),
    make_stack_override(37, (19, 2), ("Гном", "Травница", "Алхимик")),
    make_stack_override(38, (39, 3), ("Призрак", "Колдун", "Воин")),
    make_stack_override(39, (2, 4), ("Волк", "Варвар")),
    make_stack_override(40, (7, 4), ("Гном", "Метатель топоров", "Травница")),
    make_stack_override(41, (16, 5), ("Гном", "Метатель топоров")),
    make_stack_override(42, (25, 5), ("Разбойник",)),
    make_stack_override(43, (45, 5), ("Призрак", "Тамплиер")),
    make_stack_override(44, (14, 7), ("Гном", "Гном")),
    make_stack_override(45, (29, 9), ("Головорез",)),
    make_stack_override(46, (7, 10), ("Гигантский чёрный паук",)),
    make_stack_override(47, (41, 10), ("Призрак", "Привидение", "Привидение", "Воин")),
    make_stack_override(48, (32, 11), ("Жрец", "Копейщик")),
    make_stack_override(49, (18, 12), ("Гном",)),
    make_stack_override(50, (40, 13), ("Зомби",)),
    make_stack_override(51, (29, 14), ("Крестьянин", "Крестьянин", "Крестьянин")),
    make_stack_override(52, (23, 16), ("Орк",)),
    make_stack_override(53, (28, 16), ("Скваер", "Ополченец")),
    make_stack_override(54, (40, 20), ("Дракон Рока",)),
    make_stack_override(
        55,
        (41, 25),
        ("Скелет рыцарь", "Привидение", "Привидение", "Привидение", "Привидение", "Привидение"),
    ),
    make_stack_override(56, (26, 26), ("Призрак", "Воин")),
    make_stack_override(57, (35, 29), ("Призрак", "Привидение", "Воин")),
    make_stack_override(58, (37, 31), ("Зомби",)),
    make_stack_override(59, (30, 33), ("Зомби",)),
    make_stack_override(60, (41, 34), ("Зомби",)),
    make_stack_override(61, (33, 37), ("Зомби", "Призрак")),
    make_stack_override(62, (44, 37), ("Призрак", "Привидение", "Зомби", "Воин")),
    make_stack_override(63, (4, 41), ("Разбойник",)),
    make_stack_override(64, (28, 41), ("Лесной эльф", "Рейнджер")),
    make_stack_override(65, (30, 41), ("Кентавр копейщик", "Рейнджер")),
    make_stack_override(66, (41, 44), ("Демон", "Привидение", "Привидение")),
)

ALL_MAP_STACK_OVERRIDES: tuple[dict[str, object], ...] = (
    VILLAGE_LINKED_STACK_OVERRIDES
    + VILLAGE_INTERNAL_GARRISON_OVERRIDES
    + CAPITAL_INTERNAL_STACK_OVERRIDES
    + RUIN_STACK_OVERRIDES
    + ADDITIONAL_MAP_STACK_OVERRIDES
)

# --- Точки взаимодействия (перенесены из campaign_env_data.py) ---

MERCHANT_SITES = {
    "Лавка Алара": {
        "anchor": (21, 34),
        "interaction_tiles": (
            (24, 35),
            (24, 36),
            (22, 37),
            (23, 37),
            (24, 37),
        ),
    },
    "Лавка Тралара": {
        "anchor": (36, 1),
        "interaction_tiles": (
            (39, 2),
            (39, 3),
            (37, 4),
            (38, 4),
            (39, 4),
        ),
    },
}
SPELL_SHOP_SITES = {
    "Лавка заклинаний Маллавиена": {
        "site_id": "S001SI0009",
        "anchor": (42, 2),
        "interaction_tiles": (
            (45, 3),
            (45, 4),
            (43, 5),
            (44, 5),
            (45, 5),
        ),
    },
}
MERCENARY_SITES = {
    "Лагерь северных варваров": {
        "site_id": "S001SI0002",
        "anchor": (9, 42),
        "interaction_tiles": (
            (12, 43),
            (12, 44),
            (10, 45),
            (11, 45),
            (12, 45),
        ),
        "roster": (
            {
                "slot": 0,
                "unit_id": "g000uu5040",
                "unit_name": "Варвар",
                "gold": 850.0,
                "stock": 1,
            },
        ),
    },
    "Пустой лагерь наёмников": {
        "site_id": "S001SI0007",
        "anchor": (15, 1),
        "interaction_tiles": (
            (18, 2),
            (18, 3),
            (16, 4),
            (17, 4),
            (18, 4),
        ),
        "roster": (
            {
                "slot": 0,
                "unit_id": "g000uu5017",
                "unit_name": "Гоблин",
                "gold": 50.0,
                "stock": 1,
            },
            {
                "slot": 1,
                "unit_id": "g000uu5018",
                "unit_name": "Гоблин лучник",
                "gold": 50.0,
                "stock": 1,
            },
        ),
    },
}
TRAINER_SITES = {
    "Тренировочный лагерь": {
        "site_id": "S001SI0006",
        "anchor": (38, 30),
        "interaction_tiles": (
            (41, 31),
            (41, 32),
            (39, 33),
            (40, 33),
            (41, 33),
        ),
    },
}

FINAL_OBJECTIVE_CITIES: Dict[str, Tuple[int, ...]] = {
    "Порт Полонис": (34, 68),
    "Соругирилла": (35, 69),
}

# Названия предметов должны совпадать с константами CampaignConstantsMixin
# (RUIN_LIFE_ELIXIR_ITEM_NAME и т.д.) — это проверяется тестом.
RUIN_REWARDS: Dict[int, Dict[str, object]] = {
    70: {
        "title": "Замок орка",
        "ruin_pos": (15, 9),
        "marker_pos": (17, 11),
        "item": "Эликсир Силы титана",
        "gold": 200,
    },
    71: {
        "title": "Храм Мараши",
        "ruin_pos": (3, 20),
        "marker_pos": (5, 22),
        "item": "Эликсир неуязвимости",
        "gold": 300,
    },
    72: {
        "title": "Руины (30, 22)",
        "ruin_pos": (30, 22),
        "marker_pos": (32, 24),
        "item": "Эликсир Жизни",
        "gold": 300,
    },
    73: {
        "title": "Руины Бритона",
        "ruin_pos": (19, 29),
        "marker_pos": (21, 31),
        "item": "Эликсир Всевышнего",
        "gold": 150,
    },
    74: {
        "title": "Замок Ху'Ларша",
        "ruin_pos": (43, 34),
        "marker_pos": (45, 36),
        "item": "Кольцо веков (Артефакт)",
        "gold": 400,
    },
}

MAP = MapConfig(
    name="default",
    grid_size=GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=BASE_STATIC_OBSTACLE_BLOCKS,
    empty_tiles=BASE_STATIC_EMPTY_TILES,
    village_heal_tiles=BASE_VILLAGE_HEAL_TILES,
    settlement_level_by_heal_tile=BASE_SETTLEMENT_LEVEL_BY_HEAL_TILE,
    legions_settlement_territory_data=BASE_LEGIONS_SETTLEMENT_TERRITORY_DATA,
    settlement_defender_heal_tile_by_enemy_id=SETTLEMENT_DEFENDER_HEAL_TILE_BY_ENEMY_ID,
    chests=BASE_STATIC_CHESTS,
    mana_sources=BASE_STATIC_MANA_SOURCES,
    enemy_stacks=MATCHED_MAP_STACKS + ALL_MAP_STACK_OVERRIDES,
    merchant_sites=MERCHANT_SITES,
    spell_shop_sites=SPELL_SHOP_SITES,
    mercenary_sites=MERCENARY_SITES,
    trainer_sites=TRAINER_SITES,
    final_objective_cities=FINAL_OBJECTIVE_CITIES,
    ruin_rewards=RUIN_REWARDS,
    default_objective="cities",
    objective_enemy_id=31,
    empire_territory_source_enemy_id=75,
    empire_territory_source_tile=(26, 10),
    scripted_capital_bot_supported=True,
    water_tiles_provider=_load_campaign_base_water_tiles,
    forest_tiles_provider=_load_campaign_base_forest_tiles,
    road_tiles_provider=_load_campaign_base_road_tiles,
    gold_mine_tiles_provider=_load_legions_territory_base_gold_mine_tiles,
    territory_forbidden_tiles_provider=_load_legions_territory_base_forbidden_tiles,
)

# Слитые словари в том же порядке и с той же семантикой .update(), что в grid.py.
BASE_ENEMY_POSITIONS: Dict[int, Tuple[int, int]] = MAP.enemy_positions()
ENEMY_TEAM_SPECS: dict[int, dict[str, object]] = MAP.enemy_team_specs()
