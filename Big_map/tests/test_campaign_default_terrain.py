"""Regression checks for real default/small terrain and gold-mine mechanics."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import campaign_env_scenario as scenario
from campaign_env import CampaignEnv
from maps import get_map
from scenario_map_data import parse_map_size, parse_water_cells


GOLD_MINES = {(2, 31), (24, 4), (29, 46), (41, 46)}


@pytest.mark.parametrize("map_name", ["default", "small"])
def test_maps_load_the_original_scenario_layers(map_name):
    config = get_map(map_name)
    assert len(config.water_tiles_provider()) == 355
    assert len(config.forest_tiles_provider()) == 251
    assert len(config.road_tiles_provider()) == 24
    assert set(config.gold_mine_tiles_provider()) == GOLD_MINES
    forbidden = set(config.territory_forbidden_tiles_provider())
    assert len(forbidden) == 641
    # Resource/chest interaction cells are deliberately exempt from this mask.
    assert {(0, 32), (0, 33), (0, 34)} <= forbidden
    assert not GOLD_MINES & forbidden


def test_scenario_loads_in_a_snapshot_without_tools_or_site_packages(tmp_path):
    for name in ("campaign_env_scenario.py", "scenario_map_data.py", scenario.DEFAULT_SCENARIO_FILENAME):
        shutil.copy2(ROOT / name, tmp_path / name)
    script = f"""
import sys
sys.path.insert(0, {str(tmp_path)!r})
from campaign_env_scenario import (
    _load_campaign_base_water_tiles, _load_campaign_base_forest_tiles,
    _load_campaign_base_road_tiles, _load_legions_territory_base_gold_mine_tiles,
    _load_legions_territory_base_forbidden_tiles,
)
assert tuple(len(f()) for f in (
    _load_campaign_base_water_tiles, _load_campaign_base_forest_tiles,
    _load_campaign_base_road_tiles, _load_legions_territory_base_gold_mine_tiles,
    _load_legions_territory_base_forbidden_tiles,
)) == (355, 251, 24, 4, 641)
assert 'tools' not in sys.modules and 'PIL' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", script], check=True, cwd=tmp_path)


def test_missing_scenario_is_reported_instead_of_an_empty_map(monkeypatch):
    monkeypatch.setattr(scenario, "DEFAULT_SCENARIO_FILENAME", "missing_scenario.sg")
    scenario._load_base_scenario_data.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            scenario._load_base_scenario_data()
    finally:
        scenario._load_base_scenario_data.cache_clear()


def test_missing_terrain_blocks_are_reported_instead_of_empty_water():
    data = (ROOT / scenario.DEFAULT_SCENARIO_FILENAME).read_bytes()
    damaged = data.replace(b".?AVCMidgardMapBlock@@", b".?AVMissingTerrain@@")
    with pytest.raises(ValueError, match="terrain blocks do not cover"):
        parse_water_cells(damaged, parse_map_size(damaged))


@pytest.fixture(params=["default", "small"])
def env(request, monkeypatch):
    monkeypatch.setenv("CAMPAIGN_OBSERVATION_VERSION", "local5")
    environment = CampaignEnv(map_name=request.param, Realcapital=2, log_enabled=False,
                              scripted_capital_bot_enabled=False, use_boss_starting_roster=False)
    environment.reset(seed=42)
    try:
        yield environment
    finally:
        environment.close()


def test_environment_scales_terrain_and_gold_mines(env):
    expected_size = 24 if env.map_name == "small" else 48
    assert env.grid_size == expected_size
    assert not env.scripted_capital_bot_config_enabled
    assert not env.use_boss_starting_roster
    expected_mines = {(1, 15), (12, 2), (14, 23), (20, 23)} if expected_size == 24 else GOLD_MINES
    assert set(env.gold_mine_tiles) == expected_mines
    for tiles in (env.water_tiles, env.forest_tiles, env.road_tiles):
        assert tiles
        assert all(0 <= x < expected_size and 0 <= y < expected_size for x, y in tiles)


@pytest.mark.parametrize("terrain,cost", [("water", 6), ("forest", 4), ("road", 1)])
def test_real_terrain_changes_movement_cost(env, terrain, cost):
    # Adjacent encounters spend additional battle moves; isolate travel here.
    for enemy_id in env.grid_env.enemies_alive:
        env.grid_env.enemies_alive[enemy_id] = False
    env.grid_env.dynamic_blocked_positions = set()
    hero = env._resolve_travel_hero()
    hero["name"] = "Герцог"
    hero["hero_abilities"] = []
    hero["flying"] = False
    env._sync_hero_progression_flags(env.blue_team_state)
    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, refill=True)
    blocked = set(env.grid_env.obstacle_positions) | set(env.grid_env.enemy_positions.values())
    directions = ((-1, 0, env.grid_env.ACTION_RIGHT), (1, 0, env.grid_env.ACTION_LEFT),
                  (0, -1, env.grid_env.ACTION_DOWN), (0, 1, env.grid_env.ACTION_UP))
    for target in sorted(set(getattr(env, f"{terrain}_tiles"))):
        if target in blocked or env._campaign_tile_terrain(target) != terrain:
            continue
        for dx, dy, action in directions:
            start = (target[0] + dx, target[1] + dy)
            if start in blocked or not all(0 <= c < env.grid_size for c in start):
                continue
            env.grid_env.agent_pos = start
            env.grid_env.visited_cells = {start}
            starting_moves = env.moves_per_turn
            env.moves = starting_moves
            observation, _, _, _, info = env.step(action)
            assert env.grid_env.agent_pos == target
            assert info["target_terrain"] == terrain
            assert info["move_points_spent"] == cost
            assert env.moves == starting_moves - cost
            assert env.observation_space.contains(observation)
            return
    pytest.fail(f"No passable {terrain} movement found on {env.map_name}")


def test_captured_gold_mine_increases_actual_turn_income(env):
    assert env.legions_captured_gold_mine_count == 0
    env._advance_turns(10)
    assert env.legions_captured_gold_mine_count >= 1
    assert env._legions_gold_income_per_turn() == (
        env.GOLD_PER_TURN + env.legions_captured_gold_mine_count * env.LEGIONS_GOLD_MINE_GOLD_PER_TURN
    )
    before, expected_income = env.gold, env._legions_gold_income_per_turn()
    env._advance_turns(1)
    assert env.gold - before == expected_income
