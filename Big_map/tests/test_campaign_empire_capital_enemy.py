import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv, UNITS_BLUE
from campaign_env import CampaignEnv
from enemy_configs import ENEMY_CONFIGS
from grid import CAPITAL_HEAL_TILE_ARMOR_BONUS, ENEMY_TEAM_SPECS, are_targets_reachable


EMPIRE_CAPITAL_INNER_ENEMY_ID = 75
EMPIRE_CAPITAL_INNER_TILE = (26, 10)


def test_empire_capital_inner_tile_is_open_and_reachable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert env.grid_env.enemy_positions[EMPIRE_CAPITAL_INNER_ENEMY_ID] == EMPIRE_CAPITAL_INNER_TILE
    assert EMPIRE_CAPITAL_INNER_TILE not in set(env.grid_env.obstacle_positions)
    assert EMPIRE_CAPITAL_INNER_TILE not in set(env.castle_heal_tiles)
    assert are_targets_reachable(
        env.grid_size,
        env.grid_env.start_position,
        set(env.grid_env.obstacle_positions),
        {EMPIRE_CAPITAL_INNER_TILE},
    )


def test_empire_capital_inner_enemy_has_requested_composition():
    assert ENEMY_TEAM_SPECS[EMPIRE_CAPITAL_INNER_ENEMY_ID]["front"] == [
        None,
        None,
        None,
    ]
    assert ENEMY_TEAM_SPECS[EMPIRE_CAPITAL_INNER_ENEMY_ID]["back"] == [
        None,
        "Мизраэль",
        None,
    ]

    team_by_position = {
        int(unit.get("position", -1)): unit for unit in ENEMY_CONFIGS[EMPIRE_CAPITAL_INNER_ENEMY_ID]
    }
    assert team_by_position[5]["name"] == "Мизраэль"
    assert team_by_position[5]["stand"] == "behind"


def test_empire_capital_enemy_gets_only_capital_armor_bonus():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    env._init_battle(EMPIRE_CAPITAL_INNER_ENEMY_ID)

    base_unit = next(
        unit for unit in ENEMY_CONFIGS[EMPIRE_CAPITAL_INNER_ENEMY_ID]
        if unit["name"] == "Мизраэль"
    )
    battle_unit = next(
        unit for unit in env.battle_env.combined
        if unit.get("team") == "red" and unit.get("name") == "Мизраэль"
    )

    assert int(battle_unit["position"]) == 5
    assert int(battle_unit["armor"]) == int(base_unit["armor"]) + CAPITAL_HEAL_TILE_ARMOR_BONUS
    assert int(battle_unit["settlement_armor_bonus"]) == CAPITAL_HEAL_TILE_ARMOR_BONUS
    assert battle_unit["settlement_level"] == "capital"


def test_capital_named_enemy_max_exp_required_is_battle_observable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    entry = env._find_unit_data_by_name("Мизраэль")
    assert entry is not None

    red_unit = env._build_unit_from_data(entry, "red", 5)
    assert red_unit["exp_required"] == 0

    battle = BattleEnv(log_enabled=False)
    battle._init_with_custom_teams(
        env._build_battle_team_with_placeholders("red", [red_unit]),
        env._build_battle_team_with_placeholders("blue", UNITS_BLUE),
    )

    battle._obs()
