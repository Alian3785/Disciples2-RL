"""
RED squads for Campaign Mode.

Team composition specs live in grid.py so campaign map generation and battle
payloads share the same source of truth.
"""

from __future__ import annotations

from data_dicts_compact_lines import DATA, map_unit_to_battle
from grid import ENEMY_TEAM_SPECS


def _default_stand(position: int) -> str:
    return "ahead" if position in (1, 2, 3) else "behind"


def _build_name_index() -> dict[str, dict]:
    """Index units by name from DATA using the first match."""
    by_name: dict[str, dict] = {}
    for entry in DATA:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("кто", "") or "").strip()
        if name and name not in by_name:
            by_name[name] = entry
    return by_name


_UNITS_BY_NAME = _build_name_index()

_NAME_ALIASES: dict[str, str] = {
    "Выверна": "Виверна",
    "Гоблин-лучник": "Гоблин лучник",
    "Зеленый дракон": "Зелёный дракон",
    "Дракон Рока": "Дракон рока",
    "Лорд Тьмы": "Лорд тьмы",
}


def _canonical_name(name: str) -> str:
    raw = str(name or "").strip()
    return _NAME_ALIASES.get(raw, raw)


def _empty_red_slot(position: int) -> dict:
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
    if len(front) != 3 or len(back) != 3:
        raise ValueError("front/back must contain exactly 3 slots")

    units: list[dict] = []
    for idx, name in enumerate(front, start=1):
        units.append(_red_unit(name, idx) if name else _empty_red_slot(idx))
    for idx, name in enumerate(back, start=4):
        units.append(_red_unit(name, idx) if name else _empty_red_slot(idx))
    return units


ENEMY_CONFIGS = {
    enemy_id: _build_team(
        front=list(spec["front"]),  # type: ignore[arg-type]
        back=list(spec["back"]),  # type: ignore[arg-type]
    )
    for enemy_id, spec in ENEMY_TEAM_SPECS.items()
}

ENEMY_DESCRIPTIONS = {
    enemy_id: str(spec["description"])
    for enemy_id, spec in ENEMY_TEAM_SPECS.items()
}
