import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_dicts_compact_lines import (  # noqa: E402
    DATA,
    map_unit_to_battle,
)


def _unit_entry(name: str) -> dict:
    return next(entry for entry in DATA if entry.get("кто") == name)


def test_map_unit_to_battle_uses_position_for_stand():
    archer = map_unit_to_battle(_unit_entry("Стрелок"), team="blue", position=10)
    warrior = map_unit_to_battle(_unit_entry("Скваер"), team="blue", position=7)

    assert archer["unit_type"] == "Archer"
    assert archer["attack_type_primary"] == "Weapon"
    assert archer["stand"] == "behind"
    assert warrior["stand"] == "ahead"
