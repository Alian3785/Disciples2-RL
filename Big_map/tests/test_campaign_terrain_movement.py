import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _new_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    for enemy_id in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[enemy_id] = False
    env.grid_env.dynamic_blocked_positions = set()
    return env


def _set_travel_hero(env: CampaignEnv, name: str, *, flying: bool = False) -> dict:
    hero = env._resolve_travel_hero()
    assert hero is not None
    hero["name"] = name
    hero["Level"] = 8
    hero["hero_abilities"] = [env.HERO_FLYING_ABILITY_KEY] if flying else []
    hero["flying"] = bool(flying)
    env._sync_hero_progression_flags(env.blue_team_state)
    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, refill=True)
    return hero


def _entry_move_for_terrain(env: CampaignEnv, terrain: str):
    terrain_tiles = sorted(getattr(env, f"{terrain}_tiles"))
    obstacle_tiles = set(env.grid_env.obstacle_positions)
    enemy_tiles = set(env.grid_env.enemy_positions.values())
    candidates = [
        (-1, 0, env.grid_env.ACTION_RIGHT),
        (1, 0, env.grid_env.ACTION_LEFT),
        (0, -1, env.grid_env.ACTION_DOWN),
        (0, 1, env.grid_env.ACTION_UP),
    ]

    for target in terrain_tiles:
        if target in obstacle_tiles or target in enemy_tiles:
            continue
        tx, ty = target
        for dx, dy, action in candidates:
            start = (tx + dx, ty + dy)
            if not (0 <= start[0] < env.grid_size and 0 <= start[1] < env.grid_size):
                continue
            if start in obstacle_tiles or start in enemy_tiles:
                continue
            return start, action, target
    raise AssertionError(f"No passable {terrain} move target found")


def _entry_move_for_plain(env: CampaignEnv):
    obstacle_tiles = set(env.grid_env.obstacle_positions)
    enemy_tiles = set(env.grid_env.enemy_positions.values())
    terrain_tiles = set(env.water_tiles) | set(env.forest_tiles) | set(env.road_tiles)
    candidates = [
        (-1, 0, env.grid_env.ACTION_RIGHT),
        (1, 0, env.grid_env.ACTION_LEFT),
        (0, -1, env.grid_env.ACTION_DOWN),
        (0, 1, env.grid_env.ACTION_UP),
    ]

    for y in range(env.grid_size):
        for x in range(env.grid_size):
            target = (x, y)
            if target in obstacle_tiles or target in enemy_tiles or target in terrain_tiles:
                continue
            for dx, dy, action in candidates:
                start = (x + dx, y + dy)
                if not (0 <= start[0] < env.grid_size and 0 <= start[1] < env.grid_size):
                    continue
                if start in obstacle_tiles or start in enemy_tiles:
                    continue
                return start, action, target
    raise AssertionError("No passable plain move target found")


def _entry_move(env: CampaignEnv, terrain: str):
    if terrain == "plain":
        return _entry_move_for_plain(env)
    return _entry_move_for_terrain(env, terrain)


@pytest.mark.parametrize(
    ("terrain", "expected_cost"),
    [
        ("water", 6),
        ("forest", 4),
        ("road", 1),
    ],
)
def test_non_flying_hero_spends_terrain_move_cost(terrain: str, expected_cost: int):
    env = _new_env()
    _set_travel_hero(env, "Герцог", flying=False)
    start, action, target = _entry_move_for_terrain(env, terrain)
    env.grid_env.agent_pos = start
    env.grid_env.visited_cells = {start}
    env.moves = 20

    _, _, _, _, info = env.step(action)

    assert env.grid_env.agent_pos == target
    assert info["target_terrain"] == terrain
    assert info["move_cost"] == expected_cost
    assert info["move_points_spent"] == expected_cost
    assert env.moves == 20 - expected_cost


@pytest.mark.parametrize(
    ("terrain", "expected_cost"),
    [
        ("plain", 4),
        ("water", 12),
        ("forest", 8),
        ("road", 2),
    ],
)
def test_dead_hero_keeps_move_cap_but_doubles_step_costs(
    terrain: str,
    expected_cost: int,
):
    env = _new_env()
    hero = _set_travel_hero(env, "Герцог", flying=False)
    alive_move_cap = int(env.moves_per_turn)
    hero["hp"] = 0
    hero["health"] = 0
    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, refill=True)
    start, action, target = _entry_move(env, terrain)
    env.grid_env.agent_pos = start
    env.grid_env.visited_cells = {start}
    env.moves = 20

    _, _, _, _, info = env.step(action)

    assert env.moves_per_turn == alive_move_cap
    assert env.grid_env.agent_pos == target
    assert info["target_terrain"] == terrain
    assert info["move_cost"] == expected_cost
    assert info["move_points_spent"] == expected_cost
    assert env.moves == 20 - expected_cost


def test_dead_flying_hero_does_not_ignore_terrain():
    env = _new_env()
    hero = _set_travel_hero(env, "Рыцарь на пегасе", flying=True)
    hero["hp"] = 0
    hero["health"] = 0
    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, refill=True)
    start, action, target = _entry_move_for_terrain(env, "water")
    env.grid_env.agent_pos = start
    env.grid_env.visited_cells = {start}
    env.moves = 20

    _, _, _, _, info = env.step(action)

    assert env.grid_env.agent_pos == target
    assert info["target_terrain"] == "water"
    assert info["move_cost"] == 12
    assert info["move_points_spent"] == 12


def test_dead_hero_disables_terrain_boot_effects():
    env = _new_env()
    hero = _set_travel_hero(env, "Герцог", flying=False)
    env._add_hero_item(env.ELVEN_BOOTS_ITEM_NAME)
    assert env.equipped_boot_items == [env.ELVEN_BOOTS_ITEM_NAME]

    hero["hp"] = 0
    hero["health"] = 0
    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, refill=True)
    start, action, target = _entry_move_for_terrain(env, "forest")
    env.grid_env.agent_pos = start
    env.grid_env.visited_cells = {start}
    env.moves = 20

    _, _, _, _, info = env.step(action)

    assert env.equipped_boot_items == [None]
    assert env.grid_env.agent_pos == target
    assert info["target_terrain"] == "forest"
    assert info["move_cost"] == 8
    assert info["move_points_spent"] == 8


def test_expensive_terrain_move_with_short_budget_spends_all_remaining_moves():
    env = _new_env()
    _set_travel_hero(env, "Герцог", flying=False)
    start, action, target = _entry_move_for_terrain(env, "water")
    env.grid_env.agent_pos = start
    env.grid_env.visited_cells = {start}
    env.moves = 3

    _, _, _, _, info = env.step(action)

    assert env.grid_env.agent_pos == target
    assert info["move_cost"] == 6
    assert info["move_points_spent"] == 3
    assert env.moves == 0


@pytest.mark.parametrize("terrain", ["water", "forest", "road"])
def test_flying_hero_ignores_terrain_costs(terrain: str):
    env = _new_env()
    _set_travel_hero(env, "Рыцарь на пегасе", flying=True)
    start, action, target = _entry_move_for_terrain(env, terrain)
    env.grid_env.agent_pos = start
    env.grid_env.visited_cells = {start}
    env.moves = 20

    _, _, _, _, info = env.step(action)

    assert env.grid_env.agent_pos == target
    assert info["target_terrain"] == terrain
    assert info["move_cost"] == 2
    assert info["move_points_spent"] == 2
    assert env.moves == 18


@pytest.mark.parametrize(
    ("terrain", "boot_name"),
    [
        ("forest", CampaignEnv.ELVEN_BOOTS_ITEM_NAME),
        ("water", CampaignEnv.BOOTS_OF_THE_ELEMENTS_ITEM_NAME),
    ],
)
def test_terrain_boots_reduce_matching_terrain_cost_to_plain_cost(
    terrain: str,
    boot_name: str,
):
    env = _new_env()
    _set_travel_hero(env, "Герцог", flying=False)
    env._add_hero_item(boot_name)
    start, action, target = _entry_move_for_terrain(env, terrain)
    env.grid_env.agent_pos = start
    env.grid_env.visited_cells = {start}
    env.moves = 20

    _, _, _, _, info = env.step(action)

    assert env.equipped_boot_items == [boot_name]
    assert env.grid_env.agent_pos == target
    assert info["target_terrain"] == terrain
    assert info["move_cost"] == 2
    assert info["move_points_spent"] == 2
    assert env.moves == 18


def test_scripted_bot_uses_same_non_flying_terrain_costs():
    env = _new_env()
    start, _action, target = _entry_move_for_terrain(env, "water")
    env.scripted_capital_bot_position = start
    for unit in env.scripted_capital_bot_team_state:
        if env._is_hero_unit(unit):
            unit["name"] = "Герцог"
            unit["hero_abilities"] = []
            unit["flying"] = False
            break
    env._sync_hero_progression_flags(env.scripted_capital_bot_team_state)

    moved_path = env._move_scripted_bot_along_path([start, target])
    info = env._scripted_bot_movement_info(moved_path)

    assert moved_path == [target]
    assert env.scripted_capital_bot_position == target
    assert info["move_cost"] == 6
    assert info["move_points_spent"] == 6
    assert info["move_terrains"] == ["water"]
