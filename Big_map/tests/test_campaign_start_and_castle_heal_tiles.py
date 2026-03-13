import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from grid import DEFAULT_HERO_GRID_POSITION, are_targets_reachable
from grid_world_env import GridWorldEnv


def _castle_heal_action_slice(env: CampaignEnv):
    start = env.GRID_CASTLE_HEAL_ACTION_START
    stop = start + len(env.CASTLE_HEAL_POSITIONS)
    return slice(start, stop)


def _castle_revive_action_slice(env: CampaignEnv):
    start = env.GRID_CASTLE_REVIVE_ACTION_START
    stop = start + len(env.CASTLE_REVIVE_POSITIONS)
    return slice(start, stop)


def _pairwise_manhattan_distances(tiles):
    distances = []
    for index, (x1, y1) in enumerate(tiles):
        for x2, y2 in tiles[index + 1:]:
            distances.append(abs(x1 - x2) + abs(y1 - y2))
    return distances


def _largest_component_size(tiles):
    remaining = set(tiles)
    best = 0

    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 0
        while stack:
            x, y = stack.pop()
            size += 1
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                neighbor = (x + dx, y + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        best = max(best, size)

    return best


def test_hero_starts_at_legions_capital_and_returns_there_on_reset():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    assert env.grid_env.agent_pos == DEFAULT_HERO_GRID_POSITION

    _, info = env.reset(seed=123)
    assert env.grid_env.agent_pos == DEFAULT_HERO_GRID_POSITION
    assert tuple(info["agent_pos"]) == DEFAULT_HERO_GRID_POSITION


def test_only_capital_heal_tile_exists_and_does_not_overlap_enemies():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    heal_tiles = tuple(env.castle_heal_tiles)
    assert len(heal_tiles) == 1
    assert heal_tiles[0] == DEFAULT_HERO_GRID_POSITION

    enemy_tiles = set(env.grid_env.enemy_positions.values())
    for tile in heal_tiles:
        assert tile not in enemy_tiles


def test_default_layout_uses_static_obstacles_and_keeps_targets_reachable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    obstacle_tiles = set(env.grid_env.obstacle_positions)
    assert obstacle_tiles
    assert len(obstacle_tiles) == 836
    assert obstacle_tiles.isdisjoint(set(env.castle_heal_tiles))
    assert obstacle_tiles.isdisjoint(set(env.grid_env.enemy_positions.values()))
    assert env.grid_env.start_position not in obstacle_tiles

    reachable_targets = set(env.castle_heal_tiles) | set(env.grid_env.enemy_positions.values())
    assert are_targets_reachable(
        env.grid_size,
        env.grid_env.start_position,
        obstacle_tiles,
        reachable_targets,
    )


def test_hero_start_is_empty_and_inside_legions_capital_area():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    start_x, start_y = env.grid_env.start_position
    capital_tiles = {
        (x, y)
        for x in range(1, 6)
        for y in range(25, 30)
    }

    assert env.grid_env.start_position == DEFAULT_HERO_GRID_POSITION
    assert env.grid_env.start_position not in env.grid_env.obstacle_positions
    assert (start_x, start_y) in capital_tiles


def test_obstacle_blocks_movement_with_configured_penalty():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    origin_tile = env.grid_env.agent_pos
    obstacle_tile = (origin_tile[0] + 1, origin_tile[1])
    move_action = env.grid_env.ACTION_RIGHT
    env.grid_env.obstacle_positions = {obstacle_tile}

    env.grid_env.agent_pos = origin_tile
    env.grid_env.visited_cells = {origin_tile}
    env.grid_visit_counts = {}
    env.recent_positions = []

    mask = env.compute_action_mask()
    assert bool(mask[move_action]) is False

    _, reward, _, _, info = env.step(move_action)

    expected_reward = 0.2 * (
        float(env.grid_env.step_penalty) + float(env.grid_env.obstacle_penalty)
    )
    assert env.grid_env.agent_pos == origin_tile
    assert reward == pytest.approx(expected_reward)
    assert info["blocked_by_obstacle"] is True
    assert tuple(info["blocked_obstacle_pos"]) == obstacle_tile


def test_grid_world_render_marks_obstacles():
    env = GridWorldEnv(
        grid_size=5,
        enemy_positions={},
        obstacle_positions={(2, 2)},
        start_position=(1, 2),
    )
    env.reset(seed=123)

    rendered = env.render(mode="ansi")
    assert rendered is not None
    assert "# " in rendered


def test_grid_world_action_mask_blocks_edges_and_obstacles():
    env = GridWorldEnv(
        grid_size=3,
        enemy_positions={},
        obstacle_positions={(1, 0)},
        start_position=(0, 0),
    )
    env.reset(seed=123)

    mask = env.compute_action_mask()

    assert mask.tolist() == [False, True, False, False, False, False, False, True, True]


def test_campaign_grid_mask_respects_edges_and_obstacles():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    env.grid_env.agent_pos = (0, 0)
    env.grid_env.visited_cells = {(0, 0)}
    env.grid_env.obstacle_positions = {(1, 0)}
    env.moves = env.moves_per_turn

    mask = env.compute_action_mask()

    assert mask[:8].tolist() == [False, True, False, False, False, False, False, True]
    assert bool(mask[8]) is False


def test_castle_heal_actions_available_on_all_heal_tiles():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    env.gold = 10.0

    # Wound one unit so castle-heal actions become valid in the mask.
    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 7)
    current_hp = float(unit.get("health", unit.get("hp", 0)))
    unit["hp"] = max(1.0, current_hp - 1.0)
    unit["health"] = unit["hp"]

    castle_slice = _castle_heal_action_slice(env)

    for tile in env.castle_heal_tiles:
        env.grid_env.agent_pos = tile
        mask = env.compute_action_mask()
        assert mask[castle_slice].any()


def test_castle_revive_actions_available_on_all_heal_tiles_when_gold_is_enough():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    env.gold = 50.0

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 7)
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["Level"] = 1

    castle_revive_slice = _castle_revive_action_slice(env)

    for tile in env.castle_heal_tiles:
        env.grid_env.agent_pos = tile
        mask = env.compute_action_mask()
        assert mask[castle_revive_slice].any()


def test_castle_heal_spends_only_required_gold_when_enough():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    unit["hp"] = max_hp - 12.0
    unit["health"] = unit["hp"]
    unit["Level"] = 1

    env.gold = 50.0
    _, _, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(max_hp)
    assert float(unit.get("health", 0) or 0) == pytest.approx(max_hp)
    assert env.gold == pytest.approx(38.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(12.0)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(12.0)


def test_castle_heal_level_2_or_3_costs_two_gold_per_hp():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    unit["hp"] = max_hp - 20.0
    unit["health"] = unit["hp"]
    unit["Level"] = 3

    env.gold = 50.0
    _, _, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(max_hp)
    assert float(unit.get("health", 0) or 0) == pytest.approx(max_hp)
    assert env.gold == pytest.approx(10.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(20.0)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(40.0)


def test_castle_heal_level_4_or_5_costs_three_gold_per_hp_and_can_be_partial():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    unit["hp"] = max_hp - 20.0
    unit["health"] = unit["hp"]
    unit["Level"] = 5

    env.gold = 5.5
    _, _, _, _, info = env.step(action)

    expected_healed = 5.5 / 3.0
    assert float(unit.get("hp", 0) or 0) == pytest.approx((max_hp - 20.0) + expected_healed)
    assert float(unit.get("health", 0) or 0) == pytest.approx((max_hp - 20.0) + expected_healed)
    assert env.gold == pytest.approx(0.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(expected_healed)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(5.5)


@pytest.mark.parametrize(
    ("level", "revive_cost"),
    [
        (1, 50.0),
        (2, 200.0),
        (3, 400.0),
        (4, 600.0),
        (5, 800.0),
    ],
)
def test_castle_revive_spends_gold_by_unit_level(level, revive_cost):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_REVIVE_ACTION_START + env.CASTLE_REVIVE_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["Level"] = level

    env.gold = revive_cost + 25.0
    _, _, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(1.0)
    assert float(unit.get("health", 0) or 0) == pytest.approx(1.0)
    assert env.gold == pytest.approx(25.0)
    assert info.get("revived") is True
    assert float(info.get("revive_cost", 0.0)) == pytest.approx(revive_cost)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(revive_cost)


def test_castle_revive_action_is_blocked_when_gold_is_below_required_cost():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_REVIVE_ACTION_START + env.CASTLE_REVIVE_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["Level"] = 4

    env.gold = 599.0
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    _, _, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(0.0)
    assert float(unit.get("health", 0) or 0) == pytest.approx(0.0)
    assert env.gold == pytest.approx(599.0)
    assert info.get("revived") is False
    assert float(info.get("revive_cost", 0.0)) == pytest.approx(600.0)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(0.0)
