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
from typing import Dict, Optional, Tuple

DEFAULT_GRID_SIZE = 48
DEFAULT_HERO_GRID_POSITION: Tuple[int, int] = (5, 27)
OBSTACLE_COVERAGE_RATIO = 0.0
OBSTACLE_EDGE_MARGIN = 2
BASE_OBSTACLE_RIDGE_POLYLINES: Tuple[Tuple[Tuple[int, int], ...], ...] = ()
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
    ((11, 11), ("Ruby (Valuable)",)),
    ((13, 19), ("Ruby (Valuable)",)),
    ((32, 41), ("Potion of Healing", "Diamond (Valuable)")),
    ((1, 32), ("Mind ward scroll",)),
    ((2, 3), ("Mind ward scroll",)),
    ((34, 10), ("Potion of Healing", "Bronze Ring (Valuable)")),
    ((24, 30), ("Potion of Healing", "Bronze Ring (Valuable)")),
)
CHEST_COUNT = len(BASE_STATIC_CHESTS)

# Base campaign layout mirrors the 48x48 scenario coordinates directly.
BASE_GRID_SIZE = DEFAULT_GRID_SIZE
BASE_ENEMY_POSITIONS: Dict[int, Tuple[int, int]] = {}

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
        "description": "Отряд Порта Полонис: Нигош (Носферату), Колдун, Воин, Привидение",
        "front": ["Колдун", "Носферату", "Воин"],
        "back": [None, "Привидение", None],
    },
    {
        "enemy_id": 35,
        "position": (40, 25),
        "description": "Отряд Соругириллы: Унфегра (Королева личей), Виверна",
        "front": [None, "Королева лич", None],
        "back": [None, "Виверна", None],
    },
)
def _pack_stack_units(units: Tuple[str, ...]) -> Tuple[list[str | None], list[str | None]]:
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
) -> dict[str, object]:
    front, back = _pack_stack_units(units)
    composition = ", ".join(units) if units else "(empty)"
    return {
        "enemy_id": int(enemy_id),
        "position": tuple(position),
        "description": f"Отряд: {composition}",
        "front": front,
        "back": back,
    }


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
    VILLAGE_LINKED_STACK_OVERRIDES + ADDITIONAL_MAP_STACK_OVERRIDES
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
    """Scale the base campaign enemy layout to an arbitrary grid size."""
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
    """Scale a static set of map tiles to the current grid size."""
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
    """Expand ordered rectangle footprints into ordered individual tiles."""
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


def _bresenham_line(
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> Tuple[Tuple[int, int], ...]:
    """Rasterize a straight segment between two grid points."""
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    points: list[Tuple[int, int]] = []

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return tuple(points)


def _rasterize_polyline(
    points: Tuple[Tuple[int, int], ...],
) -> Tuple[Tuple[int, int], ...]:
    if not points:
        return ()
    if len(points) == 1:
        return points

    rasterized: list[Tuple[int, int]] = []
    for index in range(len(points) - 1):
        segment = _bresenham_line(points[index], points[index + 1])
        if index > 0:
            segment = segment[1:]
        rasterized.extend(segment)
    return tuple(rasterized)


def _tile_within_margin(
    tile: Tuple[int, int],
    grid_size: int,
    margin: int,
) -> bool:
    if margin <= 0:
        return True
    x, y = int(tile[0]), int(tile[1])
    return (
        margin <= x < int(grid_size) - margin
        and margin <= y < int(grid_size) - margin
    )


def _obstacle_ring_offsets(radius: int) -> Tuple[Tuple[int, int], ...]:
    if radius <= 0:
        return ((0, 0),)

    offsets: list[Tuple[int, int, int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if max(abs(dx), abs(dy)) != radius:
                continue
            offsets.append((abs(dx) + abs(dy), abs(dy), dy, dx))
    offsets.sort()
    return tuple((dx, dy) for _, _, dy, dx in offsets)


def are_targets_reachable(
    grid_size: int,
    start_position: Tuple[int, int],
    blocked_tiles: set[Tuple[int, int]],
    targets: set[Tuple[int, int]],
) -> bool:
    """Check that the start can still reach every target tile."""
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
    """Use the fixed scenario heal tiles first, then fill extra slots deterministically."""
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


def resolve_chests(
    grid_size: int,
    enemy_positions: Dict[int, Tuple[int, int]],
    *,
    heal_tiles: Tuple[Tuple[int, int], ...] | list[Tuple[int, int]] = (),
    obstacle_tiles: Tuple[Tuple[int, int], ...] | list[Tuple[int, int]] | set[Tuple[int, int]] = (),
    start_position: Tuple[int, int] = DEFAULT_HERO_GRID_POSITION,
    base_static_chests: Tuple[Tuple[Tuple[int, int], Tuple[str, ...]], ...] = BASE_STATIC_CHESTS,
) -> Dict[Tuple[int, int], Tuple[str, ...]]:
    """Resolve deterministic treasure chest tiles and the items stored in each chest."""
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
    obstacle_coverage_ratio: float = OBSTACLE_COVERAGE_RATIO,
    obstacle_edge_margin: int = OBSTACLE_EDGE_MARGIN,
    base_obstacle_ridge_polylines: Tuple[Tuple[Tuple[int, int], ...], ...] = (
        BASE_OBSTACLE_RIDGE_POLYLINES
    ),
) -> Tuple[Tuple[int, int], ...]:
    """Build deterministic obstacle tiles for the campaign map."""
    start_tile = clamp_grid_pos(start_position, grid_size)
    reserved_tiles = {tuple(pos) for pos in enemy_positions.values()}
    reserved_tiles.update(tuple(tile) for tile in heal_tiles)
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
    if static_obstacle_tiles:
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

    effective_margin = min(
        max(0, int(obstacle_edge_margin)),
        max(0, (int(grid_size) - 1) // 4),
    )
    target_count = max(0, int(round(float(grid_size) * float(grid_size) * obstacle_coverage_ratio)))

    free_capacity = 0
    for y in range(int(grid_size)):
        for x in range(int(grid_size)):
            tile = (x, y)
            if tile in reserved_tiles:
                continue
            if not _tile_within_margin(tile, grid_size, effective_margin):
                continue
            free_capacity += 1
    target_count = min(target_count, free_capacity)

    ridge_skeleton: list[Tuple[int, int]] = []
    skeleton_seen: set[Tuple[int, int]] = set()
    for ridge in base_obstacle_ridge_polylines:
        scaled_points = scale_static_tiles(
            grid_size,
            ridge,
            base_grid_size=BASE_GRID_SIZE,
        )
        for tile in _rasterize_polyline(scaled_points):
            tile = clamp_grid_pos(tile, grid_size)
            if tile in skeleton_seen:
                continue
            if not _tile_within_margin(tile, grid_size, effective_margin):
                continue
            skeleton_seen.add(tile)
            ridge_skeleton.append(tile)

    ordered_candidates: list[Tuple[int, int]] = []
    candidate_seen: set[Tuple[int, int]] = set()
    for tile in ridge_skeleton:
        if tile in reserved_tiles or tile in candidate_seen:
            continue
        candidate_seen.add(tile)
        ordered_candidates.append(tile)
    for radius in range(1, int(grid_size)):
        ring_offsets = _obstacle_ring_offsets(radius)
        for center in ridge_skeleton:
            cx, cy = center
            for dx, dy in ring_offsets:
                tile = (cx + dx, cy + dy)
                if tile in reserved_tiles or tile in candidate_seen:
                    continue
                if not (0 <= tile[0] < int(grid_size) and 0 <= tile[1] < int(grid_size)):
                    continue
                if not _tile_within_margin(tile, grid_size, effective_margin):
                    continue
                candidate_seen.add(tile)
                ordered_candidates.append(tile)

    obstacle_tiles: list[Tuple[int, int]] = []
    obstacle_seen: set[Tuple[int, int]] = set()
    for tile in ordered_candidates:
        if len(obstacle_tiles) >= target_count:
            break
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
