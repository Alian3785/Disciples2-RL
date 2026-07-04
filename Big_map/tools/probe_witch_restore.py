"""Проверка: после отката превращения (Суккуб/Ведьма) initiative_base не восстанавливается."""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from battle_env import BattleEnv
from console_encoding import setup_utf8_console
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit

setup_utf8_console()


def _entry(name: str) -> dict:
    for e in DATA:
        if isinstance(e, dict) and str(e.get("кто", "")).strip() == name:
            return e
    raise KeyError(name)


def main() -> None:
    env = BattleEnv(log_enabled=True)
    env.rng = random.Random(0)
    red = [
        map_unit_to_battle(_entry("Суккуб"), "red", 5)
        if pos == 5
        else placeholder_unit("red", pos)
        for pos in range(1, 7)
    ]
    blue = [
        map_unit_to_battle(_entry("Мародёр"), "blue", 10)
        if pos == 10
        else placeholder_unit("blue", pos)
        for pos in range(7, 13)
    ]
    env._init_with_custom_teams(red, blue)
    env.pop_pretty_events()

    succub = next(u for u in env.combined if u.get("name") == "Суккуб")
    marauder = next(u for u in env.combined if u.get("name") == "Мародёр")

    fields = ("initiative_base", "damage", "accuracy", "unit_type", "attack_type_primary")
    print("До превращения: ", {k: marauder.get(k) for k in fields})

    env._apply_witch_effect(succub, marauder)
    print("После превращения:", {k: marauder.get(k) for k in fields})

    restored = env._restore_transformed_unit(marauder)
    print(f"Откат превращения (restored={restored}):")
    print("После отката:   ", {k: marauder.get(k) for k in fields})
    for line in env.pop_pretty_events():
        print(f"   {line}")


if __name__ == "__main__":
    main()
