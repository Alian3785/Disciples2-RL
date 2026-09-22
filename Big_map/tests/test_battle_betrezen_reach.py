import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import (
    AOE_TYPES,
    LONG_PARALYSIS_TYPES,
    MELEE_TYPES,
    SMART_MELEE_TARGET_TYPES,
    BattleEnv,
)
from data_dicts_compact_lines import DATA, map_unit_to_battle


def _unit(name: str, team: str, position: int) -> dict:
    source = next(entry for entry in DATA if entry.get("кто") == name)
    unit = map_unit_to_battle(source, team=team, position=position)
    unit["health"] = unit["max_health"]
    return unit


def test_betrezen_is_ranged_single_target_with_paralysis():
    """Одержимый Утер бьёт одну любую цель и парализует, но не ближник и не AoE."""
    assert "Betrezen" not in MELEE_TYPES
    assert "Betrezen" not in AOE_TYPES
    assert "Betrezen" in LONG_PARALYSIS_TYPES
    assert "Betrezen" in SMART_MELEE_TARGET_TYPES
    # «Утер ребенок» остаётся обычным ближником.
    assert "Uter" in MELEE_TYPES


def test_betrezen_reaches_back_row_while_front_row_is_alive():
    env = BattleEnv(log_enabled=False)
    hero = _unit("Утер", team="red", position=1)
    front = _unit("Гоблин", team="blue", position=7)
    back = _unit("Гоблин", team="blue", position=10)
    env.combined = [hero, front, back]

    assert hero["unit_type"] == "Betrezen"
    options = env._smart_paralysis_target_options(hero)
    assert sorted(options) == [7, 10]


def test_melee_paralysis_unit_still_limited_to_front_row():
    env = BattleEnv(log_enabled=False)
    melee = _unit("Утер ребенок", team="red", position=1)
    front = _unit("Гоблин", team="blue", position=7)
    back = _unit("Гоблин", team="blue", position=10)
    env.combined = [melee, front, back]

    assert melee["unit_type"] in MELEE_TYPES
    options = env._smart_paralysis_target_options(melee)
    assert 10 not in options
    assert 7 in options
