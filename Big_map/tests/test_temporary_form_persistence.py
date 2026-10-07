"""Battle endings and every campaign save must discard temporary identities."""
from copy import deepcopy

import pytest

from battle_env import (
    BattleEnv, DEFEND_ACTION_INDEX, RUN_AWAY_ACTION_INDEX, TARGET_POSITIONS,
)
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit
from permanent_unit_stats import add_permanent_effect


CATALOG = {entry["кто"]: entry for entry in DATA}
IDENTITY = (
    "name", "unit_id", "unit_type", "damage", "damage_secondary", "max_health",
    "initiative_base", "armor", "accuracy", "accuracy_secondary", "immunity",
    "resistance", "attack_type_primary", "attack_type_secondary", "Level",
    "exp_current", "exp_required", "exp_kill", "position", "stand",
)


def fighter(name, team="blue", position=7):
    return map_unit_to_battle(CATALOG[name], team, position)


def padded(units, team):
    occupied = {unit["position"] for unit in units}
    positions = range(7, 13) if team == "blue" else range(1, 7)
    return units + [placeholder_unit(team, pos) for pos in positions if pos not in occupied]


def temporary_form(battle, unit, monkeypatch):
    if unit["name"] == "Двойник":
        assert battle._apply_doppelganger_copy(unit, fighter("Охотник на ведьм", "red", 4))
    else:
        # Execute the real Wolf Lord action without advancing other fighters.
        with monkeypatch.context() as patch:
            patch.setattr(battle, "_advance_until_blue_turn", lambda: None)
            battle.current_blue_attacker_pos = unit["position"]
            battle.blue_attacks_left = 1
            battle.step(TARGET_POSITIONS.index(unit["position"]))
        assert unit["name"] == "Дух Фенрира"


def form_battle(name, monkeypatch):
    original = fighter(name)
    original["exp_current"] = 17
    add_permanent_effect(original, {"kind": "health", "multiplier": 1.1})
    add_permanent_effect(original, {"kind": "damage", "multiplier": 1.1})
    battle = BattleEnv(log_enabled=False)
    runtime = deepcopy(original)
    battle.combined = padded([runtime, fighter("Скваер", position=8)], "blue") + padded(
        [fighter("Скваер", "red", 4)], "red")
    temporary_form(battle, runtime, monkeypatch)
    return battle, runtime, original


def assert_restored(actual, original, health):
    for key in IDENTITY:
        assert actual.get(key) == original.get(key), key
    assert actual["health"] == actual["hp"] == health
    assert actual["maxhp"] == original["max_health"]
    assert actual.get("original_damage", actual["damage"]) == original["damage"]
    assert actual["campaign_stat_sources"] == original["campaign_stat_sources"]
    assert not actual.get("transformed")
    assert not actual.get("doppel_copied")
    assert not actual.get("wolflord_base")
    assert not actual.get("_attack_form_unit_id")
    assert not actual.get("basestats")


@pytest.mark.parametrize("name", ["Двойник", "Повелитель волков"])
@pytest.mark.parametrize("condition", ["dead", "critical", "wounded"])
@pytest.mark.parametrize("route", ["blue", "red", "city_blue", "city_red", "bot"])
def test_unfinished_battle_save_restores_forms_without_mutating_combat(
    campaign_factory, monkeypatch, name, condition, route,
):
    env = campaign_factory(map_name="trade_train")
    battle, runtime, original = form_battle(name, monkeypatch)
    runtime["health"] = {"dead": 0, "critical": 1, "wounded": runtime["max_health"] / 2}[condition]
    expected_hp = {"dead": 0, "critical": 1, "wounded": round(original["max_health"] / 2)}[condition]
    if route in ("red", "city_red"):
        runtime.update(team="red", position=4)
        original.update(team="red", position=4)
        battle.combined = padded([runtime], "red")
    env.battle_env = battle
    before = deepcopy(battle.combined)
    if route == "blue":
        env.blue_team_state = padded([deepcopy(original)], "blue")
        env._save_blue_state()
        roster = env.blue_team_state
    elif route == "red":
        env.enemy_team_states[7701] = padded([deepcopy(original)], "red")
        env._save_enemy_state_from_battle(7701)
        roster = env.enemy_team_states[7701]
    elif route.startswith("city_"):
        roster = env._saved_city_battle_team(runtime["team"])
        if route == "city_red":
            original.update(team="blue", position=10, stand="behind")
    else:
        env._save_scripted_capital_bot_state_from_battle(battle)
        roster = env.scripted_capital_bot_team_state
    saved = next(unit for unit in roster if unit["position"] == original["position"])
    assert_restored(saved, original, expected_hp)
    assert battle.combined == before


@pytest.mark.parametrize("name", ["Двойник", "Повелитель волков"])
@pytest.mark.parametrize("route", ["victory", "direct_victory", "timeout"])
@pytest.mark.parametrize("dead", [False, True])
def test_endings_unwind_nested_forms_for_living_and_dead(monkeypatch, name, route, dead):
    battle, runtime, original = form_battle(name, monkeypatch)
    runtime["health"] = runtime["max_health"] if not dead else 0
    battle._apply_witch_effect({}, runtime)
    expected_hp = original["max_health"] if not dead else 0
    if route == "timeout":
        monkeypatch.setattr(battle, "_advance_until_blue_turn", lambda: None)
        battle.current_blue_attacker_pos = 8
        battle.blue_attacks_left = 1
        battle.step_count = 999
        _, _, terminated, truncated, _ = battle.step(DEFEND_ACTION_INDEX)
        assert truncated and not terminated
    else:
        for unit in battle.combined:
            if unit["team"] == "red":
                unit["health"] = 0
                unit["exp_kill"] = 0
        if route == "victory":
            battle._check_victory_after_hit()
        else:
            battle._finalize_victory("blue")
        assert battle.winner == "blue"
    assert_restored(runtime, original, expected_hp)


def test_public_fenrir_retreat_restores_escaped_unit_before_campaign_save(campaign_factory):
    original = fighter("Повелитель волков")
    battle = BattleEnv(log_enabled=False)
    battle.seed(7)
    battle._init_with_custom_teams(padded([fighter("Скваер", "red", 4)], "red"),
                                  padded([original], "blue"))
    battle.step(TARGET_POSITIONS.index(7))
    assert battle._unit_by_position(7)["name"] == "Дух Фенрира"
    _, _, terminated, truncated, _ = battle.step(RUN_AWAY_ACTION_INDEX)
    assert terminated and not truncated and battle.winner == "red"
    escaped = battle.escaped_units[0]
    assert escaped["name"] == original["name"]
    assert escaped["unit_type"] == "Wolf Lord"
    assert 0 < escaped["health"] <= original["max_health"]
    assert escaped["max_health"] == original["max_health"]
    env = campaign_factory(map_name="trade_train")
    env.blue_team_state = padded([original], "blue")
    env.battle_env = battle
    env._save_blue_state()
    saved = next(unit for unit in env.blue_team_state if unit["position"] == 7)
    assert saved["name"] == original["name"]
    assert saved["health"] == saved["hp"] == escaped["health"]
    assert saved["damage"] == original["damage"]


@pytest.mark.parametrize("team", ["blue", "red"])
def test_partial_retreat_is_preserved_even_without_a_winner(campaign_factory, monkeypatch, team):
    battle, runtime, original = form_battle("Двойник", monkeypatch)
    if team == "red":
        runtime.update(team="red", position=4)
        original.update(team="red", position=4)
        battle.combined = padded([runtime], "red")
    runtime["health"] = 1
    battle._mark_unit_escaped(runtime)
    assert battle.winner is None
    env = campaign_factory(map_name="trade_train")
    env.battle_env = battle
    if team == "blue":
        env.blue_team_state = padded([original], team)
        env._save_blue_state()
        roster = env.blue_team_state
    else:
        env.enemy_team_states[7701] = padded([original], team)
        env._save_enemy_state_from_battle(7701)
        roster = env.enemy_team_states[7701]
    saved = next(unit for unit in roster if unit["position"] == original["position"])
    assert_restored(saved, original, 1)


def test_doppelganger_copying_wolf_lord_returns_to_its_own_identity(monkeypatch):
    battle, runtime, original = form_battle("Двойник", monkeypatch)
    assert battle._apply_doppelganger_copy(runtime, fighter("Повелитель волков"))
    with monkeypatch.context() as patch:
        patch.setattr(battle, "_advance_until_blue_turn", lambda: None)
        battle.current_blue_attacker_pos = 7
        battle.blue_attacks_left = 1
        battle.step(TARGET_POSITIONS.index(7))
    assert runtime["name"] == "Дух Фенрира"
    runtime["health"] = 0
    battle._finalize_victory("red")
    assert_restored(runtime, original, 0)
