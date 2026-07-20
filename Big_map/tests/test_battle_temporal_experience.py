import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv, UNITS_BLUE, UNITS_RED
from data_dicts_compact_lines import DATA, map_unit_to_battle


def _unit_at(units: list[dict], position: int) -> dict:
    return deepcopy(
        next(unit for unit in units if int(unit.get("position", -1)) == position)
    )


def _named_unit(name: str, position: int) -> dict:
    source = next(entry for entry in DATA if entry.get("кто") == name)
    return map_unit_to_battle(source, team="blue", position=position)


def _set_non_leveling_exp(unit: dict, current: int = 0) -> None:
    unit["exp_current"] = int(current)
    unit["exp_required"] = 100_000


def test_escaped_winner_keeps_only_exp_earned_before_leaving_battle():
    env = BattleEnv(log_enabled=False)
    runner = _unit_at(UNITS_BLUE, 7)
    ally = _unit_at(UNITS_BLUE, 8)
    first_enemy = _unit_at(UNITS_RED, 1)
    second_enemy = _unit_at(UNITS_RED, 2)
    _set_non_leveling_exp(runner, current=11)
    _set_non_leveling_exp(ally)
    first_enemy["exp_kill"] = 40
    second_enemy["exp_kill"] = 60
    env.combined = [first_enemy, second_enemy, runner, ally]

    env._subtract_health(first_enemy, first_enemy["health"])
    env._mark_unit_escaped(runner)
    escaped = env.escaped_units[0]
    env._subtract_health(second_enemy, second_enemy["health"])
    env._apply_battle_exp("red")

    assert escaped["exp_current"] == 31  # initial 11 + 40 / 2
    assert ally["exp_current"] == 80  # 40 / 2 before escape + all later 60
    assert env.last_battle_exp == 100
    assert "_battle_exp_earned" not in escaped
    assert all("_battle_exp_earned" not in unit for unit in env.combined)


def test_revived_winner_receives_only_exp_from_kills_after_resurrection():
    env = BattleEnv(log_enabled=False)
    env._patriach_revived_recipients = set()
    patriarch = _named_unit("Патриарх", position=7)
    revived = _named_unit("Защитник Веры", position=8)
    first_enemy = _unit_at(UNITS_RED, 1)
    second_enemy = _unit_at(UNITS_RED, 2)
    _set_non_leveling_exp(patriarch)
    _set_non_leveling_exp(revived)
    first_enemy["exp_kill"] = 40
    second_enemy["exp_kill"] = 60
    env.combined = [first_enemy, second_enemy, patriarch, revived]

    env._subtract_health(revived, revived["health"])
    revived["initiative"] = 0
    env._subtract_health(first_enemy, first_enemy["health"])
    assert env._apply_patriach_support(patriarch, revived) == "revive_success"
    env._subtract_health(second_enemy, second_enemy["health"])
    env._apply_battle_exp("red")

    assert patriarch["exp_current"] == 70  # all 40, then 60 / 2
    assert revived["exp_current"] == 30  # only the kill after resurrection
    assert env.last_battle_exp == 100


def test_unit_that_dies_again_loses_exp_earned_after_resurrection():
    env = BattleEnv(log_enabled=False)
    env._patriach_revived_recipients = set()
    patriarch = _named_unit("Патриарх", position=7)
    revived = _named_unit("Защитник Веры", position=8)
    first_enemy = _unit_at(UNITS_RED, 1)
    second_enemy = _unit_at(UNITS_RED, 2)
    _set_non_leveling_exp(patriarch)
    _set_non_leveling_exp(revived)
    first_enemy["exp_kill"] = 40
    second_enemy["exp_kill"] = 60
    env.combined = [first_enemy, second_enemy, patriarch, revived]

    env._subtract_health(revived, revived["health"])
    revived["initiative"] = 0
    assert env._apply_patriach_support(patriarch, revived) == "revive_success"
    env._subtract_health(first_enemy, first_enemy["health"])
    env._subtract_health(revived, revived["health"])
    env._subtract_health(second_enemy, second_enemy["health"])
    env._apply_battle_exp("red")

    assert revived["exp_current"] == 0
    assert patriarch["exp_current"] == 80  # 40 / 2, then all later 60


def test_repeated_kill_of_a_revived_enemy_counts_each_actual_defeat():
    env = BattleEnv(log_enabled=False)
    env._patriach_revived_recipients = set()
    winner = _unit_at(UNITS_BLUE, 7)
    healer = _named_unit("Патриарх", position=1)
    healer["team"] = "red"
    victim = _unit_at(UNITS_RED, 2)
    _set_non_leveling_exp(winner)
    victim["exp_kill"] = 25
    env.combined = [healer, victim, winner]

    env._subtract_health(victim, victim["health"])
    victim["initiative"] = 0
    assert env._apply_patriach_support(healer, victim) == "revive_success"
    env._subtract_health(victim, victim["health"])
    env._subtract_health(healer, healer["health"])
    env._apply_battle_exp("red")

    assert winner["exp_current"] == 2 * 25 + int(healer["exp_kill"])
    assert env.last_battle_exp == 2 * 25 + int(healer["exp_kill"])
