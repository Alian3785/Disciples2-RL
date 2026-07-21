import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv, UNITS_BLUE, UNITS_RED
from campaign_env import CampaignEnv


def _travel_hero(env: CampaignEnv) -> dict:
    hero = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero is not None
    return hero


def _unit_at(units: list[dict], position: int) -> dict:
    return deepcopy(
        next(unit for unit in units if int(unit.get("position", -1)) == position)
    )


def test_weapon_master_unlocks_by_level_or_explicit_ability_token():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    hero = _travel_hero(env)

    hero["Level"] = env.HERO_WEAPON_MASTER_LEVEL - 1
    hero["hero_abilities"] = []
    assert env._hero_has_weapon_master(units=[hero]) is False

    hero["hero_abilities"] = ["WEAPONS_MASTER"]
    assert env._hero_has_weapon_master(units=[hero]) is True
    assert "Мастер оружия (+25% опыта всему отряду за бой)" in (
        env.get_travel_hero_visual_info()["abilities"]
    )

    hero["hero_abilities"] = []
    hero["Level"] = env.HERO_WEAPON_MASTER_LEVEL
    assert env._hero_has_weapon_master(units=[hero]) is True


def test_weapon_master_sets_party_exp_bonus_without_stacking_with_tome_of_war():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    hero = _travel_hero(env)
    hero["Level"] = 1
    hero["hero_abilities"] = [env.HERO_WEAPON_MASTER_ABILITY_KEY]

    env._active_book_exp_multiplier = lambda: 1.25
    env._init_battle(enemy_id=1)

    assert env.battle_env is not None
    assert env.battle_env.exp_multiplier == pytest.approx(1.25)


def test_weapon_master_multiplier_preserves_escape_timing_for_party_exp():
    battle = BattleEnv(log_enabled=False)
    battle.exp_multiplier = 1.25
    runner = _unit_at(UNITS_BLUE, 7)
    ally = _unit_at(UNITS_BLUE, 8)
    first_enemy = _unit_at(UNITS_RED, 1)
    second_enemy = _unit_at(UNITS_RED, 2)
    runner["exp_current"] = 0
    runner["exp_required"] = 100_000
    ally["exp_current"] = 0
    ally["exp_required"] = 100_000
    first_enemy["exp_kill"] = 40
    second_enemy["exp_kill"] = 60
    battle.combined = [first_enemy, second_enemy, runner, ally]

    battle._subtract_health(first_enemy, first_enemy["health"])
    battle._mark_unit_escaped(runner)
    escaped = battle.escaped_units[0]
    battle._subtract_health(second_enemy, second_enemy["health"])
    battle._apply_battle_exp("red")

    assert escaped["exp_current"] == 25
    assert ally["exp_current"] == 100
    assert battle.last_battle_exp == pytest.approx(125)
