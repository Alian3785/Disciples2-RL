"""Регрессионные тесты бага №4: после отката превращения (Суккуб/Ведьма/Колдунья,
предмет, артефакт — все идут через _apply_witch_effect) юнит навсегда оставался
с initiative_base 30/50: рукописный снапшот эффекта не содержал этого поля,
и порча утекала в persistent-состояние отряда кампании через _save_blue_state.
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


def test_initiative_base_restored_after_recovery():
    env = _make_env({5: "Суккуб"}, {10: "Мародёр"})
    succub = next(u for u in env.combined if u.get("name") == "Суккуб")
    marauder = next(u for u in env.combined if u.get("name") == "Мародёр")

    env._apply_witch_effect(succub, marauder)
    assert marauder["initiative_base"] == 30
    assert marauder["damage"] == 20
    assert marauder["unit_type"] == "Warrior"

    assert env._restore_transformed_unit(marauder) is True

    assert marauder["initiative_base"] == 60
    assert marauder["damage"] == 70
    assert marauder["unit_type"] == "Elfarcher"
    assert marauder["transformed"] == 0


def test_initiative_base_restored_at_battle_end():
    env = _make_env({5: "Суккуб"}, {10: "Мародёр"})
    succub = next(u for u in env.combined if u.get("name") == "Суккуб")
    marauder = next(u for u in env.combined if u.get("name") == "Мародёр")

    env._apply_witch_effect(succub, marauder)
    assert marauder["initiative_base"] == 30

    # Превращённый доживает до конца боя: победа синих откатывает всех превращённых.
    succub["health"] = 0
    env._check_victory_after_hit()

    assert env.winner == "blue"
    assert marauder["initiative_base"] == 60
    assert marauder["transformed"] == 0


def test_recovery_does_not_log_raw_basestats_dictionary():
    env = _make_env({5: "Суккуб"}, {10: "Мародёр"})
    succub = next(u for u in env.combined if u.get("name") == "Суккуб")
    marauder = next(u for u in env.combined if u.get("name") == "Мародёр")

    env._apply_witch_effect(succub, marauder)
    marauder["transform_recover_chance"] = 1.0
    marauder["round_effects_done"] = 0
    env.pop_pretty_events()

    env._apply_start_of_turn_effects(marauder)

    assert marauder["transformed"] == 0
    assert all(
        not event.strip().startswith("{")
        for event in env.pop_pretty_events()
    )
