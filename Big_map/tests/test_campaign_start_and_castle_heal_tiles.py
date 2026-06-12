import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from grid import (
    BASE_STATIC_HEAL_TILES,
    BASE_VILLAGE_HEAL_TILES,
    BASE_VILLAGE_PREVIOUS_CORNER_TILES,
    DEFAULT_HERO_GRID_POSITION,
    MANA_SOURCE_COUNT,
    are_targets_reachable,
    resolve_mana_sources,
)
from grid_world_env import GridWorldEnv


def _castle_heal_action_slice(env: CampaignEnv):
    start = env.GRID_CASTLE_HEAL_ACTION_START
    stop = start + len(env.CASTLE_HEAL_POSITIONS)
    return slice(start, stop)


def _castle_revive_action_slice(env: CampaignEnv):
    start = env.GRID_CASTLE_REVIVE_ACTION_START
    stop = start + len(env.CASTLE_REVIVE_POSITIONS)
    return slice(start, stop)


def _heal_bottle_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_BOTTLE_ACTION_START + env.GRID_BOTTLE_POSITIONS.index(target_pos)


def _potion_use_action_index(env: CampaignEnv, item_name: str, target_pos: int) -> int:
    return (
        env._potion_action_start_for_item(item_name)
        + env.GRID_POTION_USE_POSITIONS.index(target_pos)
    )


def _mark_temple_built(env: CampaignEnv) -> None:
    for build_key in env.building_keys:
        building = env.active_buildings.get(build_key)
        if not isinstance(building, dict):
            continue
        if str(building.get("name", "") or "").strip() == env.TEMPLE_BUILDING_NAME:
            building["built"] = 1
            return
    raise AssertionError("temple building not found")


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


VILLAGE_FOOTPRINTS = (
    ((3, 1), 4, 4),
    ((1, 37), 4, 4),
    ((37, 6), 4, 4),
    ((25, 30), 4, 4),
    ((37, 22), 4, 4),
)


def test_hero_starts_at_legions_capital_and_returns_there_on_reset():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    assert env.grid_env.agent_pos == DEFAULT_HERO_GRID_POSITION

    _, info = env.reset(seed=123)
    assert env.grid_env.agent_pos == DEFAULT_HERO_GRID_POSITION
    assert tuple(info["agent_pos"]) == DEFAULT_HERO_GRID_POSITION


def test_capital_and_village_heal_tiles_match_expected_layout():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    heal_tiles = tuple(env.castle_heal_tiles)
    assert heal_tiles == tuple(BASE_STATIC_HEAL_TILES)

    enemy_tiles = set(env.grid_env.enemy_positions.values())
    assert DEFAULT_HERO_GRID_POSITION not in enemy_tiles
    assert heal_tiles[0] == DEFAULT_HERO_GRID_POSITION
    assert set(heal_tiles) & enemy_tiles == set(BASE_VILLAGE_HEAL_TILES)


def test_village_only_configured_heal_tiles_stay_open():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    obstacle_tiles = set(env.grid_env.obstacle_positions)
    heal_tiles = set(env.castle_heal_tiles)

    for tile in BASE_VILLAGE_PREVIOUS_CORNER_TILES:
        assert tile in obstacle_tiles
    for tile in BASE_VILLAGE_HEAL_TILES:
        assert tile not in obstacle_tiles
        assert tile in heal_tiles

    for (origin_x, origin_y), width, height in VILLAGE_FOOTPRINTS:
        footprint_tiles = {
            (x, y)
            for x in range(origin_x, origin_x + width)
            for y in range(origin_y, origin_y + height)
        }
        open_tiles = footprint_tiles - obstacle_tiles
        assert open_tiles == {tile for tile in BASE_VILLAGE_HEAL_TILES if tile in footprint_tiles}


def test_default_layout_uses_static_obstacles_and_keeps_targets_reachable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    obstacle_tiles = set(env.grid_env.obstacle_positions)
    assert obstacle_tiles
    assert obstacle_tiles == set(env._static_obstacle_tiles)
    assert len(obstacle_tiles) > 800
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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


def test_grid_world_tracks_obstacles():
    env = GridWorldEnv(
        grid_size=5,
        enemy_positions={},
        obstacle_positions={(2, 2)},
        start_position=(1, 2),
    )
    env.reset(seed=123)

    assert (2, 2) in env.obstacle_positions


def test_grid_world_tracks_chests():
    env = GridWorldEnv(
        grid_size=5,
        enemy_positions={},
        obstacle_positions=set(),
        start_position=(1, 2),
    )
    env.chest_positions = {(3, 2)}
    env.reset(seed=123)

    assert (3, 2) in env.chest_positions


def test_default_campaign_uses_all_real_map_chests():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert len(env.chests) == 18
    assert env.chests[(36, 38)] == ("Potion of Healing", "Life Potion")
    assert env.chests[(24, 30)] == ("Potion of Healing", "Bronze Ring (Valuable)")


def test_default_campaign_mana_sources_do_not_overlap_map_objects():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    mana_tiles = set(env.mana_sources)

    assert len(mana_tiles) == MANA_SOURCE_COUNT
    assert mana_tiles.isdisjoint(set(env.grid_env.obstacle_positions))
    assert mana_tiles.isdisjoint(set(env.grid_env.enemy_positions.values()))
    assert mana_tiles.isdisjoint(set(env.castle_heal_tiles))
    assert mana_tiles.isdisjoint(set(env.chests))


def test_resolve_mana_sources_relocates_reserved_and_duplicate_tiles():
    resolved = resolve_mana_sources(
        3,
        reserved_tiles={(0, 0)},
        base_static_mana_sources=(
            ("infernal", (0, 0)),
            ("life", (0, 0)),
            ("death", (1, 0)),
        ),
        base_grid_size=3,
    )

    assert len(resolved) == 3
    assert (0, 0) not in resolved
    assert {meta["kind"] for meta in resolved.values()} == {"infernal", "life", "death"}


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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    env.grid_env.agent_pos = (0, 0)
    env.grid_env.visited_cells = {(0, 0)}
    env.grid_env.obstacle_positions = {(1, 0)}
    env.moves = env.moves_per_turn

    mask = env.compute_action_mask()

    assert mask[:8].tolist() == [False, True, False, False, False, False, False, True]
    assert bool(mask[8]) is False


def test_castle_heal_actions_available_on_all_heal_tiles():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _mark_temple_built(env)
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


def test_castle_heal_actions_require_temple_building():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.gold = 10.0
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
    unit["hp"] = max_hp - 12.0
    unit["health"] = unit["hp"]
    hp_before = float(unit.get("hp", 0) or 0.0)

    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(hp_before)
    assert env.gold == pytest.approx(10.0)
    assert info.get("castle_heal_action") is True
    assert info.get("temple_built") is False
    assert info.get("missing_temple") is True
    assert info.get("castle_heal_available") is False
    assert float(info.get("healed_amount", -1.0)) == pytest.approx(0.0)


def test_castle_revive_actions_available_on_all_heal_tiles_when_gold_is_enough():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _mark_temple_built(env)
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


def test_castle_revive_actions_require_temple_building():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.gold = 50.0
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    target_pos = 7
    action = env.GRID_CASTLE_REVIVE_ACTION_START + env.CASTLE_REVIVE_POSITIONS.index(target_pos)
    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["Level"] = 1

    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(0.0)
    assert env.gold == pytest.approx(50.0)
    assert info.get("castle_revive_action") is True
    assert info.get("temple_built") is False
    assert info.get("missing_temple") is True
    assert info.get("castle_revive_available") is False
    assert info.get("revived") is False


def _legacy_test_campaign_collects_chest_from_adjacent_tile_and_updates_heroitems():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    start_x, start_y = env.grid_env.agent_pos
    chest_tile = (start_x + 2, start_y)
    move_target = (start_x + 1, start_y)
    env.grid_env.obstacle_positions.discard(move_target)
    env.grid_env.obstacle_positions.discard(chest_tile)
    env.chests = {chest_tile: ("Potion of Healing", "Life Potion", "Test Relic")}
    env.heroitems = []
    env.extra_healing_bottles = 0
    env.extra_revive_bottles = 0
    env._sync_grid_chest_positions()

    _, _, terminated, truncated, info = env.step(env.grid_env.ACTION_RIGHT)

    assert terminated is False
    assert truncated is False
    assert env.grid_env.agent_pos == move_target
    assert env.heroitems == [
        "Potion of Healing",
        "Life Potion",
        "Test Relic",
        "Банка исцеления (+50 HP)",
        "Зелье воскрешения (+1)",
    ]
    assert env.chests == {}
    assert env.grid_env.chest_positions == set()
    assert env.extra_healing_bottles == 1
    assert env.extra_revive_bottles == 1
    assert info["collected_chests"] == [
        {
            "pos": chest_tile,
            "items": ["Potion of Healing", "Life Potion", "Test Relic"],
            "granted_items": ["Банка исцеления (+50 HP)", "Зелье воскрешения (+1)"],
        }
    ]
    assert info["heroitems"] == env.heroitems
    assert info["extra_healing_bottles"] == 1
    assert info["extra_revive_bottles"] == 1
    assert info["chests_remaining"] == 0

    counters = env.get_bottle_inventory_counters()
    assert counters["max_healing"] == env.MAX_HEALING_BOTTLES + 1
    assert counters["healing_left"] == env.MAX_HEALING_BOTTLES + 1
    assert counters["max_revive"] == env.MAX_REVIVE_BOTTLES + 1
    assert counters["revive_left"] == env.MAX_REVIVE_BOTTLES + 1

    rendered = env.render(mode="ansi")
    assert rendered is not None
    assert (
        "Предметы героя: Potion of Healing, Life Potion, Test Relic, "
        "Банка исцеления (+50 HP), Зелье воскрешения (+1)"
    ) in rendered
    assert "Сундуки: нет" in rendered


def test_campaign_collects_chest_from_adjacent_tile_and_hides_auto_consumed_loot_in_heroitems():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    start_x, start_y = env.grid_env.agent_pos
    chest_tile = (start_x + 2, start_y)
    move_target = (start_x + 1, start_y)
    env.grid_env.obstacle_positions.discard(move_target)
    env.grid_env.obstacle_positions.discard(chest_tile)
    env.chests = {chest_tile: ("Potion of Healing", "Life Potion", "Test Relic")}
    env.heroitems = []
    env.extra_healing_bottles = 0
    env.extra_revive_bottles = 0
    env._sync_grid_chest_positions()

    _, _, terminated, truncated, info = env.step(env.grid_env.ACTION_RIGHT)

    assert terminated is False
    assert truncated is False
    assert env.grid_env.agent_pos == move_target
    assert env.heroitems == [
        "Test Relic",
        env.BONUS_SMALL_HEAL_ITEM_NAME,
        env.BONUS_REVIVE_ITEM_NAME,
    ]
    assert env.chests == {}
    assert env.grid_env.chest_positions == set()
    assert env.extra_healing_bottles == 1
    assert env.extra_revive_bottles == 1
    assert info["collected_chests"] == [
        {
            "pos": chest_tile,
            "items": ["Potion of Healing", "Life Potion", "Test Relic"],
            "granted_items": [env.BONUS_SMALL_HEAL_ITEM_NAME, env.BONUS_REVIVE_ITEM_NAME],
        }
    ]
    assert info["heroitems"] == env.heroitems
    assert info["extra_healing_bottles"] == 1
    assert info["extra_revive_bottles"] == 1
    assert info["chests_remaining"] == 0

    counters = env.get_bottle_inventory_counters()
    assert counters["max_healing"] == env.MAX_HEALING_BOTTLES + 1
    assert counters["healing_left"] == env.MAX_HEALING_BOTTLES + 1
    assert counters["max_revive"] == env.MAX_REVIVE_BOTTLES + 1
    assert counters["revive_left"] == env.MAX_REVIVE_BOTTLES + 1

    rendered = env.render(mode="ansi")
    assert rendered is not None
    assert (
        f"Предметы героя: Test Relic, {env.BONUS_SMALL_HEAL_ITEM_NAME}, "
        f"{env.BONUS_REVIVE_ITEM_NAME}"
    ) in rendered
    assert "Сундуки: нет" in rendered


def test_campaign_minimal_step_info_reports_collected_chest_count():
    env = CampaignEnv(
        log_enabled=False,
        detailed_step_info=False,
        persist_blue_hp=True,
        Realcapital=2,
    )
    env.reset(seed=123)

    start_x, start_y = env.grid_env.agent_pos
    chest_tile = (start_x + 2, start_y)
    move_target = (start_x + 1, start_y)
    env.grid_env.obstacle_positions.discard(move_target)
    env.grid_env.obstacle_positions.discard(chest_tile)
    env.chests = {chest_tile: ("Potion of Healing",)}
    env._sync_grid_chest_positions()

    _, _, terminated, truncated, info = env.step(env.grid_env.ACTION_RIGHT)

    assert terminated is False
    assert truncated is False
    assert info["collected_chests_count"] == 1


def test_castle_heal_spends_only_required_gold_when_enough():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _mark_temple_built(env)

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    unit["hp"] = max_hp - 12.0
    unit["health"] = unit["hp"]
    unit["Level"] = 1

    env.gold = 50.0
    _, reward, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(max_hp)
    assert float(unit.get("health", 0) or 0) == pytest.approx(max_hp)
    assert env.gold == pytest.approx(38.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(12.0)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(12.0)
    assert reward == pytest.approx(12.0 * env.reward_castle_heal_per_hp)
    assert float(info.get("castle_heal_reward", -1.0)) == pytest.approx(reward)


def test_castle_heal_level_2_or_3_costs_two_gold_per_hp():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _mark_temple_built(env)

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    unit["hp"] = max_hp - 20.0
    unit["health"] = unit["hp"]
    unit["Level"] = 3

    env.gold = 50.0
    _, reward, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(max_hp)
    assert float(unit.get("health", 0) or 0) == pytest.approx(max_hp)
    assert env.gold == pytest.approx(10.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(20.0)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(40.0)
    assert reward == pytest.approx(20.0 * env.reward_castle_heal_per_hp)
    assert float(info.get("castle_heal_reward", -1.0)) == pytest.approx(reward)


def test_castle_heal_level_4_or_5_costs_three_gold_per_hp_and_can_be_partial():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _mark_temple_built(env)

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    unit["hp"] = max_hp - 20.0
    unit["health"] = unit["hp"]
    unit["Level"] = 5

    env.gold = 5.5
    _, reward, _, _, info = env.step(action)

    expected_healed = 5.5 / 3.0
    assert float(unit.get("hp", 0) or 0) == pytest.approx((max_hp - 20.0) + expected_healed)
    assert float(unit.get("health", 0) or 0) == pytest.approx((max_hp - 20.0) + expected_healed)
    assert env.gold == pytest.approx(0.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(expected_healed)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(5.5)
    assert reward == pytest.approx(expected_healed * env.reward_castle_heal_per_hp)
    assert float(info.get("castle_heal_reward", -1.0)) == pytest.approx(reward)


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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _mark_temple_built(env)

    target_pos = 7
    action = env.GRID_CASTLE_REVIVE_ACTION_START + env.CASTLE_REVIVE_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["Level"] = level

    env.gold = revive_cost + 25.0
    _, reward, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(1.0)
    assert float(unit.get("health", 0) or 0) == pytest.approx(1.0)
    assert env.gold == pytest.approx(25.0)
    assert info.get("revived") is True
    assert float(info.get("revive_cost", 0.0)) == pytest.approx(revive_cost)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(revive_cost)
    assert reward == pytest.approx(env.reward_castle_revive)
    assert float(info.get("castle_revive_reward", -1.0)) == pytest.approx(reward)


def test_castle_revive_action_is_blocked_when_gold_is_below_required_cost():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _mark_temple_built(env)

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

    _, reward, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(0.0)
    assert float(unit.get("health", 0) or 0) == pytest.approx(0.0)
    assert env.gold == pytest.approx(599.0)
    assert info.get("revived") is False
    assert float(info.get("revive_cost", 0.0)) == pytest.approx(600.0)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(0.0)
    assert reward == pytest.approx(0.0)
    assert float(info.get("castle_revive_reward", -1.0)) == pytest.approx(0.0)


def test_small_heal_potion_action_uses_50_hp_bottle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = _potion_use_action_index(env, env.BONUS_SMALL_HEAL_ITEM_NAME, target_pos)
    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))

    unit["hp"] = max_hp - 40.0
    unit["health"] = unit["hp"]

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info.get("heal_action") is True
    assert info.get("heal_target_pos") == target_pos
    assert info.get("potion_item") == env.BONUS_SMALL_HEAL_ITEM_NAME
    assert info.get("selected_heal_bottle_kind") == env.BONUS_SMALL_HEAL_ITEM_NAME
    assert float(info.get("selected_heal_bottle_amount", 0.0)) == pytest.approx(50.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(40.0)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(max_hp)
    assert float(unit.get("health", 0) or 0.0) == pytest.approx(max_hp)
    assert env.heal_bottles_used == 0
    assert env.healing_bottles_used == 1
    assert int(info.get("healing_bottles_used", -1)) == 1
    assert int(info.get("healing_bottles_left", -1)) == 2
    assert int(info.get("heal_bottles_left", -1)) == 3


def test_large_heal_potion_action_uses_100_hp_bottle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = _potion_use_action_index(env, env.BONUS_LARGE_HEAL_ITEM_NAME, target_pos)
    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))

    unit["hp"] = max_hp - 60.0
    unit["health"] = unit["hp"]

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info.get("heal_action") is True
    assert info.get("heal_target_pos") == target_pos
    assert info.get("potion_item") == env.BONUS_LARGE_HEAL_ITEM_NAME
    assert info.get("selected_heal_bottle_kind") == env.BONUS_LARGE_HEAL_ITEM_NAME
    assert float(info.get("selected_heal_bottle_amount", 0.0)) == pytest.approx(100.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(60.0)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(max_hp)
    assert float(unit.get("health", 0) or 0.0) == pytest.approx(max_hp)
    assert env.heal_bottles_used == 1
    assert env.healing_bottles_used == 0
    assert int(info.get("heal_bottles_left", -1)) == 2
    assert int(info.get("healing_bottles_left", -1)) == 3


@pytest.mark.parametrize(
    ("large_used", "small_used", "missing_hp", "expected_kind", "expected_healed"),
    [
        (3, 0, 40.0, "small", 40.0),
        (0, 3, 80.0, "large", 80.0),
    ],
)
def test_exact_heal_potion_action_uses_requested_available_bottle(
    large_used,
    small_used,
    missing_hp,
    expected_kind,
    expected_healed,
):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    item_name = (
        env.BONUS_SMALL_HEAL_ITEM_NAME
        if expected_kind == "small"
        else env.BONUS_LARGE_HEAL_ITEM_NAME
    )
    action = _potion_use_action_index(env, item_name, target_pos)
    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))

    env.heal_bottles_used = large_used
    env.healing_bottles_used = small_used
    unit["hp"] = max_hp - missing_hp
    unit["health"] = unit["hp"]

    _, _, _, _, info = env.step(action)

    assert info.get("potion_item") == item_name
    assert info.get("selected_heal_bottle_kind") == item_name
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(expected_healed)
    assert float(unit.get("hp", 0) or 0.0) == pytest.approx(max_hp)


def test_exact_heal_potion_masks_require_alive_wounded_unit_and_matching_stock():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = _potion_use_action_index(env, env.BONUS_SMALL_HEAL_ITEM_NAME, target_pos)
    large_action = _potion_use_action_index(env, env.BONUS_LARGE_HEAL_ITEM_NAME, target_pos)
    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    empty_action = _heal_bottle_action_index(env, 10)

    mask = env.compute_action_mask()
    assert bool(mask[action]) is False
    assert bool(mask[empty_action]) is False

    unit["hp"] = max_hp - 1.0
    unit["health"] = unit["hp"]
    mask = env.compute_action_mask()
    assert bool(mask[action]) is True

    unit["hp"] = 0.0
    unit["health"] = 0.0
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    unit["hp"] = max_hp
    unit["health"] = max_hp
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    unit["hp"] = max_hp - 5.0
    unit["health"] = unit["hp"]
    env.healing_bottles_used = env.MAX_HEALING_BOTTLES
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False
    assert bool(mask[large_action]) is True

    env.heal_bottles_used = env.MAX_HEAL_BOTTLES
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False
    assert bool(mask[large_action]) is False
