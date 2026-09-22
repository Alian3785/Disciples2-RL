import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from grid import CAPITAL_HEAL_TILE_ARMOR_BONUS
from maps.super_last_stand import HERO_START, LEFT_CITY_TILE


def _make_env() -> CampaignEnv:
    env = CampaignEnv(
        log_enabled=False,
        detailed_step_info=False,
        map_name="super_last_stand",
    )
    env.reset(seed=11)
    return env


def test_capital_tile_gives_blue_team_the_same_armor_as_red_defenders():
    env = _make_env()
    assert tuple(env.CASTLE_POS) == HERO_START

    bonus, _rest, source, settlement_level = env._resolve_heal_tile_context(env.CASTLE_POS)

    assert bonus == CAPITAL_HEAL_TILE_ARMOR_BONUS == 40
    assert source == "capital"
    assert settlement_level is None


def test_capital_armor_bonus_is_applied_to_every_living_unit():
    env = _make_env()
    env.battle_origin_pos = tuple(env.CASTLE_POS)
    blue_team = [unit for unit in env.blue_team_state if env._unit_max_hp(unit) > 0]
    base_armor = [env._normalize_armor_value(unit.get("armor", 0)) for unit in blue_team]

    env._apply_hero_heal_tile_armor_bonus(blue_team)

    for unit, base in zip(blue_team, base_armor):
        assert unit["armor"] == base + CAPITAL_HEAL_TILE_ARMOR_BONUS
        assert unit["settlement_armor_bonus"] == CAPITAL_HEAL_TILE_ARMOR_BONUS
        assert "settlement_level" not in unit


def test_uncaptured_settlement_tile_still_gives_no_armor():
    env = _make_env()
    bonus, _rest, source, _level = env._resolve_heal_tile_context(LEFT_CITY_TILE)

    assert bonus == 0
    assert source == ""
