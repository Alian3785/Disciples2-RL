import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from grid_world_env import GridWorldEnv


def test_grid_world_battle_triggers_on_adjacent_enemy_tile():
    env = GridWorldEnv(
        grid_size=5,
        enemy_positions={1: (2, 0)},
        obstacle_positions=set(),
        start_position=(0, 0),
    )
    env.reset(seed=123)

    _, _, terminated, truncated, info = env.step(env.ACTION_RIGHT)

    assert terminated is False
    assert truncated is False
    assert env.agent_pos == (1, 0)
    assert info["battle_triggered"] is True
    assert info["enemy_id"] == 1
    assert info["enemy_pos"] == (2, 0)
    assert info["battle_triggered_by"] == "adjacent"


def test_campaign_adjacent_battle_spends_half_full_move_cap():
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=True,
        realcapital=2,
        reward_engage_battle=0.0,
    )
    env.reset(seed=123)

    enemy_id = 1
    for eid in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[eid] = False
    env.grid_env.enemies_alive[enemy_id] = True

    env.grid_env.agent_pos = (2, 2)
    env.grid_env.visited_cells = {(2, 2)}
    env.grid_env.obstacle_positions = set()
    env.grid_env.dynamic_blocked_positions = set()
    env.grid_env.enemy_positions[enemy_id] = (4, 2)
    env.moves_per_turn = 21
    env.moves = 8

    _, _, terminated, truncated, info = env.step(env.grid_env.ACTION_RIGHT)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_BATTLE
    assert env.current_enemy_id == enemy_id
    assert env.grid_env.agent_pos == (3, 2)
    assert env.battle_origin_pos == (3, 2)
    assert info["battle_triggered"] is True
    assert info["enemy_id"] == enemy_id
    assert info["move_cost"] == 2
    assert info["battle_move_cost"] == 11
    assert info["battle_move_spent"] == 6
    assert env.moves == 0


def test_campaign_adjacent_battle_uses_full_move_cap_even_with_many_moves_left():
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=True,
        realcapital=2,
        reward_engage_battle=0.0,
    )
    env.reset(seed=123)

    enemy_id = 1
    for eid in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[eid] = False
    env.grid_env.enemies_alive[enemy_id] = True

    env.grid_env.agent_pos = (2, 2)
    env.grid_env.visited_cells = {(2, 2)}
    env.grid_env.obstacle_positions = set()
    env.grid_env.dynamic_blocked_positions = set()
    env.grid_env.enemy_positions[enemy_id] = (4, 2)
    env.moves_per_turn = 26
    env.moves = 26

    env.step(env.grid_env.ACTION_RIGHT)

    assert env.moves == 11


def test_campaign_grid_move_cost_is_two_points_without_battle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    for eid in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[eid] = False

    env.grid_env.agent_pos = (2, 2)
    env.grid_env.visited_cells = {(2, 2)}
    env.grid_env.obstacle_positions = set()
    env.grid_env.dynamic_blocked_positions = set()
    env.moves = 5

    _, _, _, _, info = env.step(env.grid_env.ACTION_RIGHT)

    assert info["move_cost"] == 2
    assert env.grid_env.agent_pos == (3, 2)
    assert env.moves == 3


def test_campaign_grid_move_with_less_than_tile_cost_spends_remaining_point():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    for eid in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[eid] = False

    env.grid_env.agent_pos = (2, 2)
    env.grid_env.visited_cells = {(2, 2)}
    env.grid_env.obstacle_positions = set()
    env.grid_env.dynamic_blocked_positions = set()
    env.moves = 1

    _, _, _, _, info = env.step(env.grid_env.ACTION_RIGHT)

    assert info.get("blocked_by_moves", False) is False
    assert info["move_cost"] == 2
    assert info["move_points_spent"] == 1
    assert env.grid_env.agent_pos == (3, 2)
    assert env.moves == 0
