"""
RED команды для Campaign Mode.
Состав задан фиксированными отрядами (30 штук).
"""

from __future__ import annotations

from data_dicts_compact_lines import DATA, map_unit_to_battle


def _default_stand(position: int) -> str:
    return "ahead" if position in (1, 2, 3) else "behind"


def _build_name_index() -> dict[str, dict]:
    """Индекс юнитов по имени из DATA (берем первое вхождение)."""
    by_name: dict[str, dict] = {}
    for entry in DATA:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("кто", "") or "").strip()
        if name and name not in by_name:
            by_name[name] = entry
    return by_name


_UNITS_BY_NAME = _build_name_index()

# Нормализация вариантов названий из ТЗ к каноническим именам в DATA.
_NAME_ALIASES: dict[str, str] = {
    "Выверна": "Виверна",
    "Гоблин-лучник": "Гоблин лучник",
    "Зеленый дракон": "Зелёный дракон",
}


def _canonical_name(name: str) -> str:
    raw = str(name or "").strip()
    return _NAME_ALIASES.get(raw, raw)


def _empty_red_slot(position: int) -> dict:
    """Пустой слот для RED стороны."""
    return {
        "name": "пусто",
        "initiative": 0,
        "initiative_base": 0,
        "team": "red",
        "position": int(position),
        "stand": _default_stand(int(position)),
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
        "Level": 0,
        "exp_kill": 0,
        "exp_required": 0,
        "exp_current": 0,
        "turns_into": [],
    }


def _red_unit(name: str, position: int) -> dict:
    """Создает RED юнита по имени из DATA и ставит в указанную позицию."""
    canonical_name = _canonical_name(name)
    src = _UNITS_BY_NAME.get(canonical_name)
    if src is None:
        raise KeyError(f'Unit "{name}" not found in DATA (canonical: "{canonical_name}")')
    unit = map_unit_to_battle(src, "red", int(position))
    unit["position"] = int(position)
    unit["stand"] = _default_stand(int(position))
    unit.setdefault("exp_kill", 0)
    unit.setdefault("exp_required", 0)
    unit.setdefault("exp_current", 0)
    unit.setdefault("turns_into", [])
    unit.setdefault("Level", 0)
    return unit


def _build_team(front: list[str | None], back: list[str | None]) -> list[dict]:
    """
    Строит RED-отряд из 6 слотов:
    front -> позиции 1..3, back -> позиции 4..6.
    """
    if len(front) != 3 or len(back) != 3:
        raise ValueError("front/back must contain exactly 3 slots")

    units: list[dict] = []
    for idx, name in enumerate(front, start=1):
        units.append(_red_unit(name, idx) if name else _empty_red_slot(idx))
    for idx, name in enumerate(back, start=4):
        units.append(_red_unit(name, idx) if name else _empty_red_slot(idx))
    return units


ENEMY_TEAM_SPECS: dict[int, dict[str, object]] = {
    1: {
        "description": "Отряд: 3 гоблина на передней линии",
        "front": ["Гоблин", "Гоблин", "Гоблин"],
        "back": [None, None, None],
    },
    2: {
        "description": "Отряд: 2 гоблина и гоблин лучник в центре передней линии",
        "front": ["Гоблин", "Гоблин лучник", "Гоблин"],
        "back": [None, None, None],
    },
    3: {
        "description": "Отряд: воин + привидение спереди, колдун + призрак + привидение сзади",
        "front": ["Воин", None, "Привидение"],
        "back": ["Колдун", "Призрак", "Привидение"],
    },
    4: {
        "description": "Отряд: титан и 2 скваера на передней линии",
        "front": ["Скваер", "Титан", "Скваер"],
        "back": [None, None, None],
    },
    5: {
        "description": "Отряд: волк на передней линии",
        "front": [None, "Волк", None],
        "back": [None, None, None],
    },
    6: {
        "description": "Отряд: волк спереди и волк сзади",
        "front": [None, "Волк", None],
        "back": [None, "Волк", None],
    },
    7: {
        "description": "Отряд: людоед на передней линии",
        "front": [None, "Людоед", None],
        "back": [None, None, None],
    },
    8: {
        "description": "Отряд: копейщик спереди и служка сзади",
        "front": [None, "Копейщик", None],
        "back": [None, "Служка", None],
    },
    9: {
        "description": "Отряд: орк в центре и 2 гоблина на передней линии",
        "front": ["Гоблин", "Орк", "Гоблин"],
        "back": [None, None, None],
    },
    10: {
        "description": "Отряд: 3 скваера на передней линии",
        "front": ["Скваер", "Скваер", "Скваер"],
        "back": [None, None, None],
    },
    11: {
        "description": "Отряд: скваер в центре и 2 крестьянина на передней линии",
        "front": ["Крестьянин", "Скваер", "Крестьянин"],
        "back": [None, None, None],
    },
    12: {
        "description": "Отряд: орк в центре передней линии и гоблин лучник на задней линии",
        "front": [None, "Орк", None],
        "back": [None, "Гоблин лучник", None],
    },
    13: {
        "description": "Отряд: орк чемпион на передней линии",
        "front": [None, "Орк чемпион", None],
        "back": [None, None, None],
    },
    14: {
        "description": "Отряд: зомби спереди, привидение сзади",
        "front": [None, "Зомби", None],
        "back": [None, "Привидение", None],
    },
    15: {
        "description": "Отряд: воин спереди, 2 призрака и привидение сзади",
        "front": [None, "Воин", None],
        "back": ["Призрак", "Привидение", "Призрак"],
    },
    16: {
        "description": "Отряд: кентавр спереди, рейнджер сзади",
        "front": [None, "Кентавр", None],
        "back": [None, "Рейнджер", None],
    },
    17: {
        "description": "Отряд: 2 привидения спереди, вампир сзади",
        "front": ["Привидение", None, "Привидение"],
        "back": [None, "Вампир", None],
    },
    18: {
        "description": "Отряд: 2 воина спереди, призрак и привидение сзади",
        "front": ["Воин", None, "Воин"],
        "back": ["Призрак", None, "Привидение"],
    },
    19: {
        "description": "Отряд: воин спереди, колдун сзади",
        "front": [None, "Воин", None],
        "back": [None, "Колдун", None],
    },
    20: {
        "description": "Отряд: тамплиер спереди, колдун и призрак сзади",
        "front": [None, "Тамплиер", None],
        "back": ["Колдун", None, "Призрак"],
    },
    21: {
        "description": "Отряд: тролль на передней линии",
        "front": [None, "Тролль", None],
        "back": [None, None, None],
    },
    22: {
        "description": "Отряд: выверна в центре спереди, 2 привидения сзади (не в центре)",
        "front": [None, "Выверна", None],
        "back": ["Привидение", None, "Привидение"],
    },
    23: {
        "description": "Отряд: гигантский паук на передней линии",
        "front": [None, "Гигантский паук", None],
        "back": [None, None, None],
    },
    24: {
        "description": "Отряд: 2 зомби спереди и привидение сзади",
        "front": ["Зомби", None, "Зомби"],
        "back": [None, "Привидение", None],
    },
    25: {
        "description": "Отряд: выверна слева спереди, королева лич в центре сзади",
        "front": ["Выверна", None, None],
        "back": [None, "Королева лич", None],
    },
    26: {
        "description": "Отряд: оборотень спереди и тень сзади",
        "front": [None, "Оборотень", None],
        "back": [None, "Тень", None],
    },
    27: {
        "description": "Отряд: лорд тьмы спереди и лич сзади",
        "front": [None, "Лорд тьмы", None],
        "back": [None, "Лич", None],
    },
    28: {
        "description": "Отряд: выверна и оборотень спереди, королева лич сзади",
        "front": ["Выверна", None, "Оборотень"],
        "back": [None, "Королева лич", None],
    },
    29: {
        "description": "Отряд: рыцарь на пегасе и рыцарь спереди, 2 стрелка сзади",
        "front": ["Рыцарь на пегасе", None, "Рыцарь"],
        "back": ["Стрелок", None, "Стрелок"],
    },
    30: {
        "description": "Отряд: зеленый дракон",
        "front": [None, "Зеленый дракон", None],
        "back": [None, None, None],
    },
}


ENEMY_CONFIGS = {
    enemy_id: _build_team(
        front=list(spec["front"]),  # type: ignore[arg-type]
        back=list(spec["back"]),    # type: ignore[arg-type]
    )
    for enemy_id, spec in ENEMY_TEAM_SPECS.items()
}

ENEMY_DESCRIPTIONS = {
    enemy_id: str(spec["description"])
    for enemy_id, spec in ENEMY_TEAM_SPECS.items()
}
