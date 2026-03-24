import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from enemy_configs import ENEMY_CONFIGS
from grid import ENEMY_TEAM_SPECS, are_targets_reachable


EMPIRE_CAPITAL_INNER_ENEMY_ID = 75
EMPIRE_CAPITAL_INNER_TILE = (26, 10)


def test_empire_capital_inner_tile_is_open_and_reachable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
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
        "Рыцарь",
        None,
        "Рыцарь на пегасе",
    ]
    assert ENEMY_TEAM_SPECS[EMPIRE_CAPITAL_INNER_ENEMY_ID]["back"] == [
        "Стрелок",
        None,
        "Стрелок",
    ]

    team_by_position = {
        int(unit.get("position", -1)): unit for unit in ENEMY_CONFIGS[EMPIRE_CAPITAL_INNER_ENEMY_ID]
    }
    assert team_by_position[1]["name"] == "Рыцарь"
    assert team_by_position[3]["name"] == "Рыцарь на пегасе"
    assert team_by_position[4]["name"] == "Стрелок"
    assert team_by_position[6]["name"] == "Стрелок"
