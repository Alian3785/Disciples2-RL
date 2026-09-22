"""Столицы и города занимают место на карте на всех картах, а не одну клетку."""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from grid import are_targets_reachable, scale_static_tiles
from maps import available_maps, get_map
from maps.base import (
    CAPITAL_ENTRANCE_OFFSET,
    CAPITAL_FOOTPRINT_SIZE,
    CITY_ENTRANCE_OFFSET,
    CITY_FOOTPRINT_SIZE,
    capital_footprint_block,
    city_footprint_block,
    footprint_tiles,
    settlement_footprint_blocks,
)


def _play_tiles(env: CampaignEnv, block) -> set[tuple[int, int]]:
    return set(scale_static_tiles(env.grid_size, tuple(sorted(footprint_tiles(block)))))


def _settlement_entrances(map_config):
    capitals = [tuple(map_config.hero_start)]
    if map_config.empire_territory_source_tile is not None:
        capitals.append(tuple(map_config.empire_territory_source_tile))
    cities = sorted(
        {tuple(tile) for tile in map_config.village_heal_tiles}
        | {tuple(tile) for tile in map_config.settlement_level_by_heal_tile}
    )
    return capitals, cities


def test_footprint_helpers_follow_the_simpler_times_layout():
    # Столица Легионов на базовой карте: блок (1, 25, 5, 5), вход (5, 27).
    assert capital_footprint_block((5, 27)) == (1, 25, 5, 5)
    # Деревня на базовой карте: блок (3, 1, 4, 4), heal tile (6, 4).
    assert city_footprint_block((6, 4)) == (3, 1, 4, 4)
    assert CAPITAL_FOOTPRINT_SIZE == 5 and CAPITAL_ENTRANCE_OFFSET == (4, 2)
    assert CITY_FOOTPRINT_SIZE == 4 and CITY_ENTRANCE_OFFSET == (3, 3)
    assert (5, 27) in footprint_tiles(capital_footprint_block((5, 27)))
    assert len(footprint_tiles(capital_footprint_block((5, 27)))) == 25
    assert settlement_footprint_blocks(capitals=((5, 27),), cities=((6, 4),)) == (
        (1, 25, 5, 5),
        (3, 1, 4, 4),
    )
    # Обрезка по краю карты не двигает вход.
    assert capital_footprint_block((1, 0), grid_size=48) == (0, 0, 2, 3)


def test_reference_map_footprints_are_covered_by_its_own_obstacle_blocks():
    # На сконвертированных картах стены заданы прямоугольниками occupancy-карты
    # сценария, поэтому сравниваем покрытие клеток, а не сами блоки.
    for map_name in ("default", "wotans_retribution"):
        map_config = get_map(map_name)
        capitals, cities = _settlement_entrances(map_config)
        covered = set()
        for block in map_config.obstacle_blocks:
            covered |= footprint_tiles(block)
        for entrance in capitals:
            walls = footprint_tiles(capital_footprint_block(entrance)) - {entrance}
            assert walls <= covered, (map_name, "capital", entrance)
        for entrance in cities:
            walls = footprint_tiles(city_footprint_block(entrance)) - {entrance}
            assert walls <= covered, (map_name, "city", entrance)


@pytest.mark.parametrize("map_name", available_maps())
def test_every_map_walls_in_its_capitals_and_cities(map_name: str):
    map_config = get_map(map_name)
    capitals, cities = _settlement_entrances(map_config)
    assert capitals, map_name

    env = CampaignEnv(map_name=map_name, log_enabled=False, Realcapital=2)
    env.reset(seed=7)
    obstacle_tiles = set(env.grid_env.obstacle_positions)
    enemy_tiles = set(env.grid_env.enemy_positions.values())
    open_tiles = set(env.castle_heal_tiles) | enemy_tiles | {tuple(env.CASTLE_POS)}

    for entrance in capitals:
        walls = _play_tiles(env, capital_footprint_block(entrance)) - open_tiles
        assert walls <= obstacle_tiles, (map_name, "capital", entrance, walls - obstacle_tiles)
    for entrance in cities:
        walls = _play_tiles(env, city_footprint_block(entrance)) - open_tiles
        assert walls <= obstacle_tiles, (map_name, "city", entrance, walls - obstacle_tiles)

    # Входы проходимы, и стены никого не отрезают: все heal tiles и отряды достижимы.
    assert tuple(env.CASTLE_POS) not in obstacle_tiles
    assert obstacle_tiles.isdisjoint(env.castle_heal_tiles)
    reachable_targets = set(env.castle_heal_tiles)
    if bool(getattr(map_config, "enforce_enemy_reachability", True)):
        reachable_targets |= enemy_tiles
    assert are_targets_reachable(
        env.grid_size,
        env.grid_env.start_position,
        obstacle_tiles,
        reachable_targets,
    ), map_name
