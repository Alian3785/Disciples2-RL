# campaign_env_scenario.py
"""Scenario file helpers for CampaignEnv.

This module is intentionally side-effect-light and contains only cached readers
for static data extracted from the local .sg scenario file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

DEFAULT_SCENARIO_FILENAME = "A Return To Simpler Times.sg"

LEGIONS_TERRITORY_PASSABLE_POSITIONED_CLASSES = frozenset(
    {
        ".?AVCMidLandmark@@",
        ".?AVCMidLocation@@",
        ".?AVCMidRoad@@",
    }
)


def _load_base_scenario_cells(parser_name: str) -> Tuple[Tuple[int, int], ...]:
    scenario_path = Path(__file__).resolve().with_name(DEFAULT_SCENARIO_FILENAME)
    if not scenario_path.exists():
        return ()

    try:
        from tools import inspect_sg_map
    except Exception:
        return ()

    parser = getattr(inspect_sg_map, parser_name, None)
    if parser is None:
        return ()

    try:
        data = scenario_path.read_bytes()
        map_size = int(inspect_sg_map.parse_map_size(data))
        cells = parser(data, map_size)
    except Exception:
        return ()

    return tuple(
        sorted(
            (int(x), int(y))
            for x, y in cells
            if 0 <= int(x) < map_size and 0 <= int(y) < map_size
        )
    )


@lru_cache(maxsize=1)
def _load_campaign_base_water_tiles() -> Tuple[Tuple[int, int], ...]:
    return _load_base_scenario_cells("parse_water_cells")


@lru_cache(maxsize=1)
def _load_campaign_base_forest_tiles() -> Tuple[Tuple[int, int], ...]:
    return _load_base_scenario_cells("parse_forest_cells")


@lru_cache(maxsize=1)
def _load_campaign_base_road_tiles() -> Tuple[Tuple[int, int], ...]:
    return _load_base_scenario_cells("parse_road_cells")


@lru_cache(maxsize=1)
def _load_legions_territory_base_forbidden_tiles() -> Tuple[Tuple[int, int], ...]:
    scenario_path = Path(__file__).resolve().with_name(DEFAULT_SCENARIO_FILENAME)
    if not scenario_path.exists():
        return ()

    try:
        from tools.inspect_sg_map import (
            RENDER_OBJECT_SPECS,
            extract_render_objects,
            parse_map_size,
            parse_positioned_objects,
            parse_resource_objects,
            parse_water_cells,
        )
    except Exception:
        return ()

    try:
        data = scenario_path.read_bytes()
        map_size = int(parse_map_size(data))
        water_cells = parse_water_cells(data, map_size)
        render_objects = extract_render_objects(data)
        positioned_objects = parse_positioned_objects(data)
        resource_objects = parse_resource_objects(data)
    except Exception:
        return ()

    forbidden_tiles: set[Tuple[int, int]] = {
        (int(x), int(y))
        for x, y in water_cells
        if 0 <= int(x) < map_size and 0 <= int(y) < map_size
    }

    for class_name, objects in render_objects.items():
        spec = RENDER_OBJECT_SPECS.get(class_name, {})
        width, height = spec.get("footprint", (1, 1))
        width = max(1, int(width))
        height = max(1, int(height))
        for obj in objects:
            ox = int(obj.get("x", 0) or 0)
            oy = int(obj.get("y", 0) or 0)
            for tx in range(ox, min(map_size, ox + width)):
                for ty in range(oy, min(map_size, oy + height)):
                    forbidden_tiles.add((tx, ty))

    for obj in positioned_objects:
        if str(obj.get("class", "") or "") in LEGIONS_TERRITORY_PASSABLE_POSITIONED_CLASSES:
            continue
        ox = int(obj.get("x", 0) or 0)
        oy = int(obj.get("y", 0) or 0)
        if 0 <= ox < map_size and 0 <= oy < map_size:
            forbidden_tiles.add((ox, oy))

    resource_tiles = {
        (int(resource.get("x", 0) or 0), int(resource.get("y", 0) or 0))
        for resource in resource_objects
        if str(resource.get("class", "") or "").startswith("resource:")
    }
    chest_tiles = {
        (int(obj.get("x", 0) or 0), int(obj.get("y", 0) or 0))
        for obj in render_objects.get(".?AVCMidBag@@", ())
    }
    forbidden_tiles.difference_update(resource_tiles | chest_tiles)

    return tuple(sorted(forbidden_tiles))


@lru_cache(maxsize=1)
def _load_legions_territory_base_water_tiles() -> Tuple[Tuple[int, int], ...]:
    return _load_campaign_base_water_tiles()


@lru_cache(maxsize=1)
def _load_legions_territory_base_gold_mine_tiles() -> Tuple[Tuple[int, int], ...]:
    scenario_path = Path(__file__).resolve().with_name(DEFAULT_SCENARIO_FILENAME)
    if not scenario_path.exists():
        return ()

    try:
        from tools.inspect_sg_map import parse_map_size, parse_resource_objects
    except Exception:
        return ()

    try:
        data = scenario_path.read_bytes()
        map_size = int(parse_map_size(data))
        resource_objects = parse_resource_objects(data)
    except Exception:
        return ()

    return tuple(
        sorted(
            {
                (int(resource.get("x", 0) or 0), int(resource.get("y", 0) or 0))
                for resource in resource_objects
                if str(resource.get("class", "") or "") == "resource:0"
                and 0 <= int(resource.get("x", 0) or 0) < map_size
                and 0 <= int(resource.get("y", 0) or 0) < map_size
            }
        )
    )


__all__ = [name for name in globals() if not name.startswith("__")]
