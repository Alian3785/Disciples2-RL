"""Read campaign terrain and resource positions without rendering dependencies.

These readers use the same .sg fields and terrain codes as tools/inspect_sg_map,
but are part of the environment itself so training snapshots need no tools package.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import re
import struct


_MAP_SIZE_RE = re.compile(rb"MAP_SIZE(.{4})", re.S)
_CLASS_RE = re.compile(rb"WHAT(.{4})(\.\?AVC[^\x00]+)\x00")
_SKIP_POSITION_CLASSES = (
    "MapBlock", "Mountains", "MidUnit", "MidItem", "MidPlayer", "MapFog",
    "PlayerKnownSpells", "PlayerBuildings", "ScenarioInfo", "ScenVariables",
    "TurnSummary", "QuestLog", "Diplomacy", "SpellEffects", "SpellCast",
    "MidgardPlan", "TalismanCharges",
)
RENDER_OBJECT_SPECS = {
    ".?AVCCapital@@": {"footprint": (5, 5)},
    ".?AVCMidVillage@@": {"footprint": (4, 4)},
    ".?AVCMidRuin@@": {"footprint": (3, 3)},
    ".?AVCMidBag@@": {"footprint": (1, 1)},
    ".?AVCMidSiteMerchant@@": {"footprint": (3, 3)},
    ".?AVCMidSiteMercs@@": {"footprint": (3, 3)},
    ".?AVCMidSiteTrainer@@": {"footprint": (3, 3)},
    ".?AVCMidSiteMage@@": {"footprint": (3, 3)},
    **{f"resource:{i}": {"footprint": (1, 1)} for i in range(6)},
}


def parse_map_size(data: bytes) -> int:
    match = _MAP_SIZE_RE.search(data)
    if match is None:
        raise ValueError("Scenario has no MAP_SIZE field")
    size = struct.unpack("<I", match.group(1))[0]
    if not 0 < size <= 256:
        raise ValueError(f"Invalid scenario map size: {size}")
    return size


def _int_field(chunk: bytes, field: bytes) -> int | None:
    index = chunk.find(field)
    if index < 0:
        return None
    return struct.unpack_from("<I", chunk, index + len(field))[0]


def _object_chunks(data: bytes, class_name: bytes):
    marker = class_name + b"\x00"
    start = 0
    while True:
        index = data.find(marker, start)
        if index < 0:
            return
        end = data.find(b"ENDOBJECT", index)
        if end < 0:
            raise ValueError(f"Unterminated scenario object: {class_name!r}")
        yield data[index:end]
        start = end + len(b"ENDOBJECT")


@lru_cache(maxsize=1)
def _terrain_grid(data: bytes, map_size: int) -> tuple[tuple[int, ...], ...]:
    grid = [[0] * map_size for _ in range(map_size)]
    covered = set()
    for chunk in _object_chunks(data, b".?AVCMidgardMapBlock@@"):
        block_index = chunk.find(b"BLOCKID")
        if block_index < 0:
            raise ValueError("Terrain block has no BLOCKID")
        id_size = _int_field(chunk, b"BLOCKID")
        id_start = block_index + len(b"BLOCKID") + 4
        block_id = chunk[id_start:id_start + id_size].rstrip(b"\x00")
        block_y = int(block_id[-4:-2], 16)
        block_x = int(block_id[-2:], 16)
        payload_size = _int_field(chunk, b"BLOCKDATA")
        if payload_size != 8 * 4 * 4:
            raise ValueError(f"Invalid terrain block payload size: {payload_size}")
        payload_start = chunk.index(b"BLOCKDATA") + len(b"BLOCKDATA") + 4
        payload = chunk[payload_start:payload_start + payload_size]
        for index, value in enumerate(struct.unpack("<32I", payload)):
            x = block_x + index % 8
            y = block_y + index // 8
            if 0 <= x < map_size and 0 <= y < map_size:
                grid[y][x] = value & 0xFF
                covered.add((x, y))
    if len(covered) != map_size * map_size:
        raise ValueError("Scenario terrain blocks do not cover the map")
    return tuple(tuple(row) for row in grid)


def parse_water_cells(data: bytes, map_size: int) -> set[tuple[int, int]]:
    # Code 29 is water; shoreline transition codes must remain walkable land.
    grid = _terrain_grid(data, map_size)
    return {(x, y) for y in range(map_size) for x in range(map_size) if grid[y][x] == 29}


def parse_forest_cells(data: bytes, map_size: int) -> set[tuple[int, int]]:
    grid = _terrain_grid(data, map_size)
    return {(x, y) for y in range(map_size) for x in range(map_size)
            if (grid[y][x] >> 3) & 7 == 1}


def parse_positioned_objects(data: bytes) -> list[dict]:
    objects = []
    for match in _CLASS_RE.finditer(data):
        class_name = match.group(2).decode("ascii")
        if any(skip in class_name for skip in _SKIP_POSITION_CLASSES):
            continue
        end = data.find(b"ENDOBJECT", match.start())
        if end < 0:
            raise ValueError(f"Unterminated scenario object: {class_name}")
        chunk = data[match.start():end]
        x, y = _int_field(chunk, b"POS_X"), _int_field(chunk, b"POS_Y")
        if x is not None and y is not None:
            objects.append({"class": class_name, "x": x, "y": y})
    return objects


def parse_road_cells(data: bytes, map_size: int) -> set[tuple[int, int]]:
    return {(obj["x"], obj["y"]) for obj in parse_positioned_objects(data)
            if obj["class"] == ".?AVCMidRoad@@"
            and 0 <= obj["x"] < map_size and 0 <= obj["y"] < map_size}


def parse_resource_objects(data: bytes) -> list[dict]:
    resources = []
    for chunk in _object_chunks(data, b".?AVCMidCrystal@@"):
        resource_id = _int_field(chunk, b"RESOURCE")
        x, y = _int_field(chunk, b"POS_X"), _int_field(chunk, b"POS_Y")
        if resource_id is None or x is None or y is None:
            raise ValueError("Resource has no RESOURCE/POS_X/POS_Y field")
        if resource_id in range(6):
            resources.append({"class": f"resource:{resource_id}", "x": x, "y": y,
                              "resource_id": resource_id})
    return resources


def extract_render_objects(data: bytes) -> dict[str, list[dict]]:
    objects = defaultdict(list)
    for obj in parse_positioned_objects(data):
        if obj["class"] in RENDER_OBJECT_SPECS:
            objects[obj["class"]].append(obj)
    for obj in parse_resource_objects(data):
        objects[obj["class"]].append(obj)
    return dict(objects)
