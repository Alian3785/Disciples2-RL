"""Проверка: тик DOT (поджог/яд) пропускается при инициативе юнита <= 10.

Условие тика в _apply_start_of_turn_effects требует initiative > 10 (защита от
двойного тика после "ждать"), но юниты с базовой инициативой 10 (Патриарх,
жрецы, Дева рощи) при броске +0 активируются с инициативой ровно 10 — и DOT
по ним в этот раунд не тикает.
"""

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
        map_unit_to_battle(_entry("Владыка"), "red", 1)
        if pos == 1
        else placeholder_unit("red", pos)
        for pos in range(1, 7)
    ]
    blue = [
        map_unit_to_battle(_entry("Патриарх"), "blue", 10)
        if pos == 10
        else placeholder_unit("blue", pos)
        for pos in range(7, 13)
    ]
    env._init_with_custom_teams(red, blue)
    env.pop_pretty_events()

    patriarch = next(u for u in env.combined if u.get("name") == "Патриарх")
    patriarch["burn_turns_left"] = 6
    patriarch["burn_damage_per_tick"] = 30

    for initiative in (11, 10):
        patriarch["health"] = patriarch["max_health"]
        patriarch["initiative"] = initiative
        # Каждый замер — как первая активация нового раунда.
        patriarch["round_effects_done"] = 0
        hp_before = patriarch["health"]
        env._apply_start_of_turn_effects(patriarch)
        hp_after = patriarch["health"]
        ticked = hp_before != hp_after
        print(
            f"инициатива={initiative}: HP {hp_before} -> {hp_after} "
            f"({'тик сработал' if ticked else 'ТИК ПРОПУЩЕН'}), "
            f"осталось ходов поджога: {patriarch['burn_turns_left']}"
        )
        for line in env.pop_pretty_events():
            print(f"   {line}")


if __name__ == "__main__":
    main()
