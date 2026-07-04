"""Регрессионные тесты бага №3: красный Двойник атаковал вместо копирования.

ИИ красных выбирал цель для копирования (макс. HP, non-big), но исполнял выбор
общей веткой _attack — Двойник бил выбранного юнита (в том числе СВОЕГО
союзника, т.к. выбор цели не фильтрует команды). Теперь обе стороны используют
общий метод _apply_doppelganger_copy.
"""

import random

from battle_env import BattleEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


def _entry(name: str) -> dict:
    for e in DATA:
        if isinstance(e, dict) and str(e.get("кто", "")).strip() == name:
            return e
    raise KeyError(name)


def _make_env(red_placement: dict, blue_placement: dict, seed: int = 0) -> BattleEnv:
    env = BattleEnv(log_enabled=True)
    env.rng = random.Random(seed)
    red = [
        map_unit_to_battle(_entry(red_placement[pos]), "red", pos)
        if pos in red_placement
        else placeholder_unit("red", pos)
        for pos in range(1, 7)
    ]
    blue = [
        map_unit_to_battle(_entry(blue_placement[pos]), "blue", pos)
        if pos in blue_placement
        else placeholder_unit("blue", pos)
        for pos in range(7, 13)
    ]
    env._init_with_custom_teams(red, blue)
    return env


def test_red_doppelganger_copies_instead_of_attacking_ally():
    # Самый жирный небольшой юнит на поле — свой Адский рыцарь (270 HP):
    # раньше красный Двойник бил его каждый раунд, теперь копирует.
    env = _make_env({2: "Двойник", 3: "Адский рыцарь"}, {8: "Мастер клинка"})
    dopp = next(u for u in env.combined if u.get("name") == "Двойник")
    ally = next(u for u in env.combined if u.get("name") == "Адский рыцарь")

    assert ally["health"] == ally["max_health"] == 270, "союзник не должен получать урон"
    assert dopp["unit_type"] == "Warrior"
    assert dopp["damage"] == 100
    assert dopp["max_health"] == 270
    assert dopp["name"] == "Двойник"


def test_forbidden_guards_excluded_from_copy_targeting():
    env = _make_env({2: "Двойник"}, {8: "Мизраэль", 9: "Скваер"})
    dopp = next(u for u in env.combined if u.get("name") == "Двойник")

    pos = env._pick_highest_hp_non_big(exclude_unit=dopp)

    # Мизраэль (страж столицы, 900 HP, non-big) — запрещённая цель копирования.
    assert pos != 8


def test_blue_doppelganger_copy_path_unchanged():
    env = _make_env({2: "Скваер"}, {8: "Двойник"})
    dopp = next(u for u in env.combined if u.get("name") == "Двойник")
    assert dopp["unit_type"] == "Doppelganger"

    mask = env.compute_action_mask()
    assert bool(mask[1])  # Скваер на pos2 — валидная цель копирования

    env.step(1)

    assert dopp["unit_type"] == "Warrior"
    assert dopp["damage"] == 25
    assert dopp["max_health"] == 100
    assert dopp["name"] == "Двойник"
