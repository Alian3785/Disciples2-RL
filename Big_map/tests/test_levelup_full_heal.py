"""A new level fully heals the unit; stopping at an evolution cap does not."""

from copy import deepcopy

import pytest

from battle_env import BattleEnv
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


UNITS = {unit["кто"]: unit for unit in DATA}


def ready_unit(name, team="blue", position=7, health=None):
    unit = map_unit_to_battle(UNITS[name], team, position)
    unit["maxhp"] = unit["max_health"]
    unit["health"] = unit["hp"] = unit["max_health"] // 2 if health is None else health
    unit["exp_current"] = float(unit["exp_required"]) - 1
    return unit


@pytest.mark.parametrize("name", ["Ангел", "Герцог"])
@pytest.mark.parametrize("team", ["blue", "red"])
def test_levelup_restores_full_health(name, team):
    unit = ready_unit(name, team)
    old_max, level = unit["max_health"], unit["Level"]
    BattleEnv(log_enabled=False)._apply_exp_award_to_unit(unit, 1)
    assert unit["Level"] == level + 1
    assert unit["max_health"] > old_max
    assert unit["health"] == unit["hp"] == unit["max_health"] == unit["maxhp"]


def test_evolution_cap_without_levelup_keeps_wounds():
    squire = ready_unit("Скваер")
    BattleEnv(log_enabled=False)._apply_exp_award_to_unit(squire, 1)
    assert squire["Level"] == 1
    assert squire["exp_current"] == squire["exp_required"] - 1
    assert squire["health"] == squire["hp"] == squire["max_health"] // 2


def test_levelup_never_revives_a_dead_unit():
    angel = ready_unit("Ангел", health=0)
    BattleEnv(log_enabled=False)._apply_exp_award_to_unit(angel, 1)
    assert angel["health"] == angel["hp"] == 0


def test_battle_victory_levelup_heals_the_wounded_winner():
    # Wounded but able to survive the Squire's opening strike.
    hero = ready_unit("Герцог", health=60)
    enemy = map_unit_to_battle(UNITS["Скваер"], "red", 1)
    battle = BattleEnv(log_enabled=False)
    battle.rng.seed(1)
    battle._init_with_custom_teams(
        [enemy] + [placeholder_unit("red", pos) for pos in range(2, 7)],
        [hero] + [placeholder_unit("blue", pos) for pos in range(8, 13)],
    )
    hero = battle._unit_by_position(7)
    enemy = battle._unit_by_position(1)
    battle._subtract_health(enemy, enemy["health"])
    enemy["initiative"] = 0
    battle._check_victory_after_hit()
    assert battle.winner == "blue"
    assert battle.last_levelup_units == [hero]
    assert hero["health"] == hero["max_health"] == 165


def test_campaign_save_keeps_full_health_without_temporary_health_bonus():
    env = CampaignEnv(
        map_name="scroll_train", observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    try:
        env.reset(seed=42)
        angel = ready_unit("Ангел", position=8)
        expected = deepcopy(angel)
        BattleEnv(log_enabled=False)._apply_exp_award_to_unit(expected, 1)
        env.blue_team_state = [
            angel if u["position"] == 8 else u for u in env.blue_team_state
        ]
        env._mark_equipment_dirty()
        env._blue_stack_spell_effect_summary = lambda: {
            "active_spell_ids": ["test"], "health_delta": 50,
        }
        prepared = deepcopy(env.blue_team_state)
        env._apply_active_blue_support_spell_effects(prepared)
        target = next(u for u in prepared if u["position"] == 8)
        assert target["max_health"] == angel["max_health"] + 50
        BattleEnv(log_enabled=False)._apply_exp_award_to_unit(target, 1)
        assert target["health"] == target["max_health"] == expected["max_health"] + 50

        battle = BattleEnv(log_enabled=False)
        battle.combined = prepared
        env.battle_env = battle
        env._save_blue_state()
        saved = next(u for u in env.blue_team_state if u["position"] == 8)
        assert saved["Level"] == expected["Level"]
        assert saved["health"] == saved["max_health"] == expected["max_health"]
    finally:
        env.close()
