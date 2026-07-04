"""Проверка: может ли BLUE-призыватель (Элементалист/Мудрец) призывать в бою."""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from battle_env import BattleEnv
from console_encoding import setup_utf8_console
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit

setup_utf8_console()


def _entry(name: str) -> dict:
    for e in DATA:
        if isinstance(e, dict) and str(e.get("кто", "")).strip() == name:
            return e
    raise KeyError(name)


def probe(summoner_name: str) -> None:
    env = BattleEnv(log_enabled=True)
    env.rng = random.Random(42)

    red = [
        map_unit_to_battle(_entry("Грифон"), "red", 3)
        if pos == 3
        else placeholder_unit("red", pos)
        for pos in range(1, 7)
    ]
    blue = [
        map_unit_to_battle(_entry(summoner_name), "blue", 8)
        if pos == 8
        else placeholder_unit("blue", pos)
        for pos in range(7, 13)
    ]
    env._init_with_custom_teams(red, blue)
    env.pop_pretty_events()

    unit = next(u for u in env.combined if u.get("name") == summoner_name)
    print(f"=== {summoner_name}: unit_type={unit.get('unit_type')!r} ===")

    mask = env.compute_action_mask()
    valid = np.flatnonzero(mask).tolist()
    print(f"Валидные действия (0-11 = pos1-12, 12=защита, 13=ждать, 14=бежать): {valid}")

    # Пробуем призвать на свою пустую клетку pos9 (action 8)
    for action, label in [(8, "своя пустая клетка pos9"), (2, "враг Грифон pos3")]:
        if action in valid:
            obs, r, term, trunc, info = env.step(action)
            print(f"-- действие {action} ({label}):")
            for line in env.pop_pretty_events():
                print(f"   {line}")
            summons = [
                u
                for u in env.combined
                if u.get("team") == "blue" and u.get("Summoned") is not None
            ]
            print(f"   Призванных юнитов у BLUE: {len(summons)}")
            if term or trunc:
                break
        else:
            print(f"-- действие {action} ({label}): ЗАБЛОКИРОВАНО маской")


if __name__ == "__main__":
    probe("Элементалист")
    print()
    probe("Мудрец")
