"""Регрессионные тесты бага №2: тик DOT пропускался при инициативе юнита <= 10.

Старая защита от двойного тика после «ждать» (проверка initiative > 10) молча
отключала яд/поджиг/воду и откаты эффектов юнитам с низкой инициативой
(Патриарх и жрецы при броске +0). Теперь применение отслеживается флагом
round_effects_done, который сбрасывается в конце раунда.
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


def _put_burn(unit: dict, *, turns: int = 6, tick: int = 30) -> None:
    unit["burn_turns_left"] = turns
    unit["burn_damage_per_tick"] = tick
    unit["health"] = unit["max_health"]


def test_dot_ticks_at_initiative_10():
    env = _make_env({1: "Владыка"}, {12: "Патриарх"})
    patriarch = next(u for u in env.combined if u.get("name") == "Патриарх")
    _put_burn(patriarch)
    patriarch["initiative"] = 10  # базовая иниц. Патриарха при броске +0
    patriarch["round_effects_done"] = 0

    env._apply_start_of_turn_effects(patriarch)

    assert patriarch["health"] == patriarch["max_health"] - 30
    assert patriarch["burn_turns_left"] == 5


def test_no_double_tick_after_wait_and_reset_next_round():
    env = _make_env({1: "Владыка"}, {12: "Патриарх"})
    patriarch = next(u for u in env.combined if u.get("name") == "Патриарх")
    _put_burn(patriarch)
    patriarch["round_effects_done"] = 0
    hp0 = patriarch["health"]

    # Первая активация в раунде — тик есть.
    env._apply_start_of_turn_effects(patriarch)
    assert patriarch["health"] == hp0 - 30

    # Повторная активация после «ждать» — тика нет.
    patriarch["initiative"] = 1
    env._apply_start_of_turn_effects(patriarch)
    assert patriarch["health"] == hp0 - 30
    assert patriarch["burn_turns_left"] == 5

    # Новый раунд сбрасывает флаг — тик снова есть.
    env._end_round_restore()
    env._apply_start_of_turn_effects(patriarch)
    assert patriarch["health"] == hp0 - 60
    assert patriarch["burn_turns_left"] == 4


class _AlwaysRecover(random.Random):
    def random(self):
        return 0.0


def test_transform_recovery_at_low_initiative():
    env = _make_env({5: "Суккуб"}, {10: "Мародёр"})
    succub = next(u for u in env.combined if u.get("name") == "Суккуб")
    marauder = next(u for u in env.combined if u.get("name") == "Мародёр")

    env._apply_witch_effect(succub, marauder)
    assert marauder["transformed"] == 1
    assert marauder["damage"] == 20

    env.rng = _AlwaysRecover()
    marauder["initiative"] = 5  # раньше при <= 10 откат не проверялся вовсе
    marauder["round_effects_done"] = 0

    env._apply_start_of_turn_effects(marauder)

    assert marauder["transformed"] == 0
    assert marauder["damage"] == 70
