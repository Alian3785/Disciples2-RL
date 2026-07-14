"""
RED squads for Campaign Mode.

Team composition specs live in grid.py so campaign map generation and battle
payloads share the same source of truth.
"""

from __future__ import annotations

from functools import lru_cache

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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value or default)))
    except (TypeError, ValueError):
        return int(default)


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
    unit["exp_kill"] = _safe_float(unit.get("exp_kill", 0))
    unit["exp_required"] = _safe_float(unit.get("exp_required", 0))
    unit["exp_current"] = _safe_float(unit.get("exp_current", 0))
    unit["Level"] = _safe_int(unit.get("Level", 0))
    turns_into = unit.get("turns_into", [])
    unit["turns_into"] = turns_into if isinstance(turns_into, list) else []
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


def build_enemy_configs(team_specs: dict[int, dict[str, object]]) -> dict[int, list[dict]]:
    """Строит боевые RED-команды по enemy-team specs выбранной карты."""
    return {
        int(enemy_id): _build_team(
            front=list(spec["front"]),  # type: ignore[arg-type]
            back=list(spec["back"]),  # type: ignore[arg-type]
        )
        for enemy_id, spec in team_specs.items()
    }


def build_enemy_descriptions(team_specs: dict[int, dict[str, object]]) -> dict[int, str]:
    """Собирает описания отрядов по enemy-team specs выбранной карты."""
    return {
        int(enemy_id): str(spec["description"])
        for enemy_id, spec in team_specs.items()
    }


ENEMY_CONFIGS = build_enemy_configs(ENEMY_TEAM_SPECS)

ENEMY_DESCRIPTIONS = build_enemy_descriptions(ENEMY_TEAM_SPECS)


def _swap_team_unit(team: list[dict], old_name: str, new_name: str) -> list[dict]:
    return [
        _red_unit(new_name, u["position"])
        if str(u.get("name", "")).strip() == old_name
        else u
        for u in team
    ]


def build_blue_dragon_enemy_configs(
    enemy_configs: dict[int, list[dict]],
) -> dict[int, list[dict]]:
    """Ростеры для режима "blue_dragon": всё как обычно, но Зелёный дракон -> Синий дракон."""
    return {
        enemy_id: _swap_team_unit(team, "Зелёный дракон", "Синий дракон")
        for enemy_id, team in enemy_configs.items()
    }


ENEMY_CONFIGS_BLUE_DRAGON = build_blue_dragon_enemy_configs(ENEMY_CONFIGS)


def get_enemy_configs_bundle(
    map_name: str = "default",
) -> tuple[dict[int, list[dict]], dict[int, str], dict[int, list[dict]]]:
    """Возвращает (configs, descriptions, blue_dragon_configs) для карты.

    Для карты по умолчанию отдаются те же объекты-глобалы: наблюдение кеширует
    сигнатуры отрядов по id(team), а тесты monkeypatch'ат модульные словари, —
    identity должен сохраняться. Для остальных карт результат кешируется.
    """
    normalized = str(map_name or "default").strip().lower()
    if normalized == "default":
        return ENEMY_CONFIGS, ENEMY_DESCRIPTIONS, ENEMY_CONFIGS_BLUE_DRAGON
    return _build_enemy_configs_bundle_cached(normalized)


@lru_cache(maxsize=8)
def _build_enemy_configs_bundle_cached(
    map_name: str,
) -> tuple[dict[int, list[dict]], dict[int, str], dict[int, list[dict]]]:
    from maps import get_map

    team_specs = get_map(map_name).enemy_team_specs()
    configs = build_enemy_configs(team_specs)
    return (
        configs,
        build_enemy_descriptions(team_specs),
        build_blue_dragon_enemy_configs(configs),
    )
