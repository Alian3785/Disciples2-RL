from copy import deepcopy

import pytest

from battle_env import ARMOR_NORM_MAX, ATTACK_TYPES, FEATURES_PER_UNIT, TYPE_LIST, BattleEnv, UNITS_BLUE
from campaign_env import CampaignEnv
from enemy_configs import ENEMY_CONFIGS


ARMOR_FEATURE_OFFSET = 8 + len(TYPE_LIST) + 5 * len(ATTACK_TYPES)


def _battle_red_unit(env: CampaignEnv, position: int) -> dict:
    assert env.battle_env is not None
    return next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "red" and int(unit.get("position", -1)) == int(position)
    )


def _armor_obs_index(position: int) -> int:
    return (int(position) - 1) * FEATURES_PER_UNIT + ARMOR_FEATURE_OFFSET


def test_city_linked_stack_gets_level_1_armor_bonus():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    env._init_battle(enemy_id=32)

    for base_unit in ENEMY_CONFIGS[32]:
        battle_unit = _battle_red_unit(env, int(base_unit.get("position", -1)))
        base_armor = int(base_unit.get("armor", 0) or 0)
        if CampaignEnv._is_empty_enemy_unit(base_unit):
            assert int(battle_unit.get("armor", -1)) == base_armor
            continue

        assert int(battle_unit.get("base_armor", -1)) == base_armor + 10
        assert int(battle_unit.get("armor", -1)) >= base_armor + 10
        assert int(battle_unit.get("settlement_armor_bonus", -1)) == 10
        assert int(battle_unit.get("settlement_level", -1)) == 1


def test_city_internal_garrison_gets_level_2_armor_bonus():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    env._init_battle(enemy_id=68)

    for base_unit in ENEMY_CONFIGS[68]:
        battle_unit = _battle_red_unit(env, int(base_unit.get("position", -1)))
        base_armor = int(base_unit.get("armor", 0) or 0)
        if CampaignEnv._is_empty_enemy_unit(base_unit):
            assert int(battle_unit.get("armor", -1)) == base_armor
            continue

        assert int(battle_unit.get("base_armor", -1)) == base_armor + 15
        assert int(battle_unit.get("armor", -1)) >= base_armor + 15
        assert int(battle_unit.get("settlement_armor_bonus", -1)) == 15
        assert int(battle_unit.get("settlement_level", -1)) == 2


def test_enemy_on_capital_heal_tile_gets_capital_armor_bonus():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    env.grid_env.enemy_positions[1] = tuple(env.CASTLE_POS)

    env._init_battle(enemy_id=1)

    for base_unit in ENEMY_CONFIGS[1]:
        battle_unit = _battle_red_unit(env, int(base_unit.get("position", -1)))
        base_armor = int(base_unit.get("armor", 0) or 0)
        if CampaignEnv._is_empty_enemy_unit(base_unit):
            assert int(battle_unit.get("armor", -1)) == base_armor
            continue

        assert int(battle_unit.get("base_armor", -1)) == base_armor + 50
        assert int(battle_unit.get("armor", -1)) >= base_armor + 50
        assert int(battle_unit.get("settlement_armor_bonus", -1)) == 50
        assert battle_unit.get("settlement_level") == "capital"


def test_battle_observation_keeps_high_armor_below_clip_point():
    env = BattleEnv(log_enabled=False)
    red_team = deepcopy(ENEMY_CONFIGS[1])
    blue_team = deepcopy(UNITS_BLUE)

    target = next(unit for unit in red_team if not CampaignEnv._is_empty_enemy_unit(unit))
    target["armor"] = 115
    target_position = int(target.get("position", -1))

    env._init_with_custom_teams(red_team, blue_team)
    obs = env._obs()

    armor_obs = float(obs[_armor_obs_index(target_position)])
    assert armor_obs == pytest.approx(115.0 / ARMOR_NORM_MAX)
    assert armor_obs < 1.0
