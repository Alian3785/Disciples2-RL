from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import ARMOR_NORM_MAX, ATTACK_TYPES, FEATURES_PER_UNIT, TYPE_LIST, BattleEnv, UNITS_BLUE
from campaign_env import CampaignEnv
from enemy_configs import ENEMY_CONFIGS
from grid import CAPITAL_HEAL_TILE_ARMOR_BONUS


ARMOR_FEATURE_OFFSET = 8 + len(TYPE_LIST) + 5 * len(ATTACK_TYPES)


def _battle_red_unit(env: CampaignEnv, position: int) -> dict:
    assert env.battle_env is not None
    return next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "red" and int(unit.get("position", -1)) == int(position)
    )


def _battle_blue_unit(env: CampaignEnv, position: int) -> dict:
    assert env.battle_env is not None
    return next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1)) == int(position)
    )


def _armor_obs_index(position: int) -> int:
    return (int(position) - 1) * FEATURES_PER_UNIT + ARMOR_FEATURE_OFFSET


def _settlement_upgrade_action_index(env: CampaignEnv, settlement_name: str) -> int:
    return env.grid_settlement_upgrade_action_start + env.settlement_upgrade_names.index(
        settlement_name
    )


def _capture_settlement(env: CampaignEnv, source_data: dict) -> str:
    settlement_name = str(source_data.get("name", "") or "")
    env.legions_active_settlement_territory_capture_turn_by_name[settlement_name] = int(env.turns)
    env._refresh_faction_territories()
    return settlement_name


def test_city_linked_stack_gets_level_1_armor_bonus():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.grid_env.enemy_positions[1] = tuple(env.CASTLE_POS)

    env._init_battle(enemy_id=1)

    for base_unit in ENEMY_CONFIGS[1]:
        battle_unit = _battle_red_unit(env, int(base_unit.get("position", -1)))
        base_armor = int(base_unit.get("armor", 0) or 0)
        if CampaignEnv._is_empty_enemy_unit(base_unit):
            assert int(battle_unit.get("armor", -1)) == base_armor
            continue

        assert int(battle_unit.get("base_armor", -1)) == base_armor + CAPITAL_HEAL_TILE_ARMOR_BONUS
        assert int(battle_unit.get("armor", -1)) >= base_armor + CAPITAL_HEAL_TILE_ARMOR_BONUS
        assert int(battle_unit.get("settlement_armor_bonus", -1)) == CAPITAL_HEAL_TILE_ARMOR_BONUS
        assert battle_unit.get("settlement_level") == "capital"


def test_hero_stack_from_captured_settlement_tile_gets_matching_armor_bonus_and_restore_after_save():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    source_data = next(
        source
        for source in env._static_legions_settlement_territory_sources
        if int(source.get("settlement_level", 0) or 0) == 1
    )
    _capture_settlement(env, source_data)
    env.battle_origin_pos = tuple(source_data.get("source_tile", ()))

    grid_unit = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 7)
    base_armor = int(grid_unit.get("armor", 0) or 0)

    env._init_battle(enemy_id=1)

    battle_unit = _battle_blue_unit(env, 7)
    assert int(battle_unit.get("base_armor", -1)) == base_armor + 10
    assert int(battle_unit.get("armor", -1)) >= base_armor + 10
    assert int(battle_unit.get("settlement_armor_bonus", -1)) == 10
    assert int(battle_unit.get("settlement_level", -1)) == 1

    env._save_blue_state()
    saved_unit = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 7)
    assert int(saved_unit.get("armor", -1)) == base_armor
    assert "settlement_armor_bonus" not in saved_unit
    assert "settlement_level" not in saved_unit


def test_hero_stack_from_uncaptured_settlement_tile_gets_no_city_bonus():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    source_data = next(
        source
        for source in env._static_legions_settlement_territory_sources
        if int(source.get("settlement_level", 0) or 0) == 1
    )
    env.battle_origin_pos = tuple(source_data.get("source_tile", ()))

    grid_unit = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 7)
    base_armor = int(grid_unit.get("armor", 0) or 0)

    env._init_battle(enemy_id=1)

    battle_unit = _battle_blue_unit(env, 7)
    assert int(battle_unit.get("base_armor", -1)) == base_armor
    assert int(battle_unit.get("armor", -1)) >= base_armor
    assert "settlement_armor_bonus" not in battle_unit
    assert "settlement_level" not in battle_unit


def test_rest_on_settlement_tile_adds_percentage_bonus_regeneration():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    source_data = next(
        source
        for source in env._static_legions_settlement_territory_sources
        if int(source.get("settlement_level", 0) or 0) == 1
    )
    settlement_name = _capture_settlement(env, source_data)
    env.grid_env.agent_pos = tuple(source_data.get("source_tile", ()))

    unit = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 7)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
    start_hp = max(1.0, max_hp - 40.0)
    unit["hp"] = start_hp
    unit["health"] = start_hp

    _, _, terminated, truncated, info = env.step(8)

    assert terminated is False
    assert truncated is False
    assert info.get("rest_action") is True
    assert info.get("rest_heal_percent") == pytest.approx(0.15)
    assert info.get("rest_heal_typeoflord_bonus_percent") == pytest.approx(0.15)
    assert info.get("rest_heal_bonus_percent") == pytest.approx(0.10)
    assert info.get("rest_heal_total_percent") == pytest.approx(0.40)
    assert info.get("rest_heal_bonus_level") == 1
    assert info.get("rest_heal_bonus_source") == settlement_name
    expected_hp = min(max_hp, start_hp + max_hp * 0.40)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(expected_hp)


def test_rest_on_capital_tile_adds_capital_percentage_bonus_regeneration():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.grid_env.agent_pos = tuple(env.CASTLE_POS)

    unit = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 8)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
    start_hp = max(1.0, max_hp - 80.0)
    unit["hp"] = start_hp
    unit["health"] = start_hp

    _, _, terminated, truncated, info = env.step(8)

    assert terminated is False
    assert truncated is False
    assert info.get("rest_action") is True
    assert info.get("rest_heal_percent") == pytest.approx(0.15)
    assert info.get("rest_heal_typeoflord_bonus_percent") == pytest.approx(0.15)
    assert info.get("rest_heal_bonus_percent") == pytest.approx(0.50)
    assert info.get("rest_heal_total_percent") == pytest.approx(0.80)
    assert info.get("rest_heal_bonus_source") == "capital"
    assert info.get("rest_heal_bonus_level") == "capital"
    expected_hp = min(max_hp, start_hp + max_hp * 0.80)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(expected_hp)


def test_rest_on_player_controlled_territory_tile_uses_15_percent_base_regeneration():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env._advance_turns(1)

    territory_tile = next(
        tile
        for tile in env.legions_territory_tiles
        if tuple(tile) != tuple(env.CASTLE_POS)
    )
    env.grid_env.agent_pos = tuple(territory_tile)

    unit = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 7)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
    start_hp = max(1.0, max_hp - 40.0)
    unit["hp"] = start_hp
    unit["health"] = start_hp

    _, _, terminated, truncated, info = env.step(8)

    assert terminated is False
    assert truncated is False
    assert info.get("rest_action") is True
    assert info.get("rest_heal_percent") == pytest.approx(0.15)
    assert info.get("rest_heal_typeoflord_bonus_percent") == pytest.approx(0.15)
    assert info.get("rest_heal_bonus_percent") == pytest.approx(0.0)
    assert info.get("rest_heal_total_percent") == pytest.approx(0.30)
    expected_hp = min(max_hp, start_hp + max_hp * 0.30)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(expected_hp)


def test_rest_typeoflord_bonus_is_not_applied_for_non_warrior_lords():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.typeoflord = 2
    env._advance_turns(1)

    territory_tile = next(
        tile
        for tile in env.legions_territory_tiles
        if tuple(tile) != tuple(env.CASTLE_POS)
    )
    env.grid_env.agent_pos = tuple(territory_tile)

    unit = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 7)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
    start_hp = max(1.0, max_hp - 40.0)
    unit["hp"] = start_hp
    unit["health"] = start_hp

    _, _, terminated, truncated, info = env.step(8)

    assert terminated is False
    assert truncated is False
    assert info.get("rest_action") is True
    assert info.get("rest_heal_percent") == pytest.approx(0.15)
    assert info.get("rest_heal_typeoflord_bonus_percent") == pytest.approx(0.0)
    assert info.get("rest_heal_bonus_percent") == pytest.approx(0.0)
    assert info.get("rest_heal_total_percent") == pytest.approx(0.15)
    expected_hp = min(max_hp, start_hp + max_hp * 0.15)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(expected_hp)


def test_settlement_upgrade_action_requires_capture_and_enough_gold():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    source_data = next(
        source
        for source in env._static_legions_settlement_territory_sources
        if int(source.get("settlement_level", 0) or 0) == 1
    )
    settlement_name = str(source_data.get("name", "") or "")
    action = _settlement_upgrade_action_index(env, settlement_name)

    env.gold = 999.0
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    _capture_settlement(env, source_data)
    env.gold = 149.0
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    _, _, terminated, truncated, info = env.step(action)
    assert terminated is False
    assert truncated is False
    assert info.get("settlement_upgrade_action") is True
    assert info.get("settlement_upgrade_name") == settlement_name
    assert info.get("settlement_upgrade_applied") is False
    assert info.get("settlement_upgrade_insufficient_gold") is True
    assert int(env.legions_settlement_level_by_name[settlement_name]) == 1


def test_settlement_upgrade_action_advances_levels_costs_and_updates_city_bonuses():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    source_data = next(
        source
        for source in env._static_legions_settlement_territory_sources
        if int(source.get("settlement_level", 0) or 0) == 1
    )
    settlement_name = _capture_settlement(env, source_data)
    action = _settlement_upgrade_action_index(env, settlement_name)

    env.gold = 5000.0
    starting_gold = env.gold
    expected_costs = [150.0, 250.0, 500.0, 750.0]
    expected_levels = [2, 3, 4, 5]

    for expected_cost, expected_level in zip(expected_costs, expected_levels):
        _, _, terminated, truncated, info = env.step(action)
        assert terminated is False
        assert truncated is False
        assert info.get("settlement_upgrade_applied") is True
        assert info.get("settlement_upgrade_cost") == pytest.approx(expected_cost)
        assert int(env.legions_settlement_level_by_name[settlement_name]) == expected_level

    assert env.gold == pytest.approx(starting_gold - sum(expected_costs))
    assert bool(env.compute_action_mask()[action]) is False

    env.battle_origin_pos = tuple(source_data.get("source_tile", ()))
    grid_unit = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 7)
    base_armor = int(grid_unit.get("armor", 0) or 0)
    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, 7)
    assert int(battle_unit.get("base_armor", -1)) == base_armor + 30
    assert int(battle_unit.get("settlement_armor_bonus", -1)) == 30
    assert int(battle_unit.get("settlement_level", -1)) == 5

    env._save_blue_state()
    env.grid_env.agent_pos = tuple(source_data.get("source_tile", ()))
    unit = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 7)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
    start_hp = max(1.0, max_hp - 80.0)
    unit["hp"] = start_hp
    unit["health"] = start_hp

    _, _, _, _, rest_info = env.step(8)
    expected_hp = min(max_hp, start_hp + max_hp * 0.60)
    assert rest_info.get("rest_heal_percent") == pytest.approx(0.15)
    assert rest_info.get("rest_heal_typeoflord_bonus_percent") == pytest.approx(0.15)
    assert rest_info.get("rest_heal_bonus_percent") == pytest.approx(0.30)
    assert rest_info.get("rest_heal_total_percent") == pytest.approx(0.60)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(expected_hp)


def test_enemy_regeneration_on_city_and_enemy_capital_tiles_uses_percentage_bonus():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    env.enemy_team_states[10] = [
        {
            "position": 1,
            "health": 100.0,
            "hp": 100.0,
            "max_health": 200.0,
            "maxhp": 200.0,
            "name": "City enemy",
            "team": "red",
            "stand": "ahead",
            "unit_type": "Warrior",
        },
    ]
    env.enemy_team_states[20] = [
        {
            "position": 1,
            "health": 40.0,
            "hp": 40.0,
            "max_health": 200.0,
            "maxhp": 200.0,
            "name": "Capital enemy",
            "team": "red",
            "stand": "ahead",
            "unit_type": "Warrior",
        },
    ]
    env.grid_env.enemy_positions = {
        10: (28, 33),
        20: tuple(env.empire_territory_source_tile),
    }
    env.grid_env.enemies_alive = {10: True, 20: True}
    env.scripted_capital_bot_enabled = False

    env._advance_turns(1)

    assert env.enemy_team_states[10][0]["health"] == pytest.approx(160.0)
    assert env.enemy_team_states[20][0]["health"] == pytest.approx(170.0)


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
