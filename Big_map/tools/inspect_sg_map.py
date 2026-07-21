#!/usr/bin/env python3
"""
Inspect a Disciples II .sg scenario file and extract coarse map structure.

Current parser focuses on:
- map size
- mountain obstacle blocks (AVCMidMountains)
- stack positions and owners (AVCMidStack)
- anchor positions of other map objects with POS_X/POS_Y fields

This is a reverse-engineering helper, not a complete format implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MAP_SIZE_RE = re.compile(rb"MAP_SIZE(.{4})", re.S)
MOUNTAIN_RE = re.compile(
    rb"ID_MOUNT(.{4})SIZE_X(.{4})SIZE_Y(.{4})POS_X(.{4})POS_Y(.{4})IMAGE(.{4})RACE(.{4})",
    re.S,
)
CLASS_RE = re.compile(rb"WHAT(.{4})(\.\?AVC[^\x00]+)\x00")
STACK_RE = re.compile(
    rb"\.\?AVCMidStack@@.*?OBJ_ID(.{4})([^\x00]+)\x00.*?POS_X(.{4})POS_Y(.{4}).*?OWNER(.{4})([^\x00]+)\x00",
    re.S,
)

SKIP_POSITION_CLASSES = (
    "MapBlock",
    "Mountains",
    "MidUnit",
    "MidItem",
    "MidPlayer",
    "MapFog",
    "PlayerKnownSpells",
    "PlayerBuildings",
    "ScenarioInfo",
    "ScenVariables",
    "TurnSummary",
    "QuestLog",
    "Diplomacy",
    "SpellEffects",
    "SpellCast",
    "MidgardPlan",
    "TalismanCharges",
)

RENDER_OBJECT_SPECS = {
    ".?AVCCapital@@": {
        "label": "Capital",
        "code": "C",
        "color": (45, 85, 185),
        "footprint": (5, 5),
    },
    ".?AVCMidVillage@@": {
        "label": "Village",
        "code": "V",
        "color": (90, 150, 235),
        "footprint": (4, 4),
    },
    ".?AVCMidRuin@@": {
        "label": "Ruin",
        "code": "R",
        "color": (150, 95, 35),
        "footprint": (3, 3),
    },
    ".?AVCMidBag@@": {
        "label": "Chest",
        "code": "$",
        "color": (235, 195, 35),
        "footprint": (1, 1),
    },
    ".?AVCMidSiteMerchant@@": {
        "label": "Merchant",
        "code": "ME",
        "color": (45, 165, 85),
        "footprint": (3, 3),
    },
    ".?AVCMidSiteMercs@@": {
        "label": "Mercs",
        "code": "MC",
        "color": (160, 85, 200),
        "footprint": (3, 3),
    },
    ".?AVCMidSiteTrainer@@": {
        "label": "Trainer",
        "code": "TR",
        "color": (245, 140, 35),
        "footprint": (3, 3),
    },
    ".?AVCMidSiteMage@@": {
        "label": "Mage Tower",
        "code": "MT",
        "color": (85, 115, 235),
        "footprint": (3, 3),
    },
    "resource:0": {
        "label": "Gold Mine",
        "code": "g",
        "color": (218, 179, 33),
        "footprint": (1, 1),
    },
    "resource:1": {
        "label": "Red mana",
        "code": "r",
        "color": (219, 68, 55),
        "footprint": (1, 1),
    },
    "resource:2": {
        "label": "Yellow mana",
        "code": "y",
        "color": (241, 196, 15),
        "footprint": (1, 1),
    },
    "resource:3": {
        "label": "Orange mana",
        "code": "o",
        "color": (230, 126, 34),
        "footprint": (1, 1),
    },
    "resource:4": {
        "label": "White mana",
        "code": "w",
        "color": (236, 240, 241),
        "footprint": (1, 1),
    },
    "resource:5": {
        "label": "Blue mana",
        "code": "b",
        "color": (52, 152, 219),
        "footprint": (1, 1),
    },
}

RENDER_ORDER = (
    ".?AVCCapital@@",
    ".?AVCMidVillage@@",
    ".?AVCMidRuin@@",
    ".?AVCMidBag@@",
    "resource:0",
    "resource:1",
    "resource:2",
    "resource:3",
    "resource:4",
    "resource:5",
    ".?AVCMidSiteMerchant@@",
    ".?AVCMidSiteMercs@@",
    ".?AVCMidSiteTrainer@@",
    ".?AVCMidSiteMage@@",
)

SETTLEMENT_ENTRY_COLOR = (255, 64, 160)
RUIN_INTERACTION_COLOR = (0, 210, 180)

MAPBLOCK_WIDTH = 8
MAPBLOCK_HEIGHT = 4
GROUND_PLAIN_ID = 0
GROUND_FOREST_ID = 1
GROUND_WATER_ID = 3
GROUND_MOUNTAIN_ID = 4
ROAD_CLASS_NAME = ".?AVCMidRoad@@"
# Keep water detection conservative: transitional shoreline codes were
# overcounting land as water on exported maps.
WATER_TILE_CODES = {29}
NEUTRAL_ENEMY_OWNER = "S001PL0000"

DATA_NAME_ALIASES = {
    "Сквайр": "Скваер",
    "Послушница": "Служка",
    "Холмовой великан": "Холмовой гигант",
    "Злой дух": "Дух",
    "Королева личей": "Королева лич",
    "Эльф-следопыт": "Рейнджер",
    "Гоблин-лучник": "Гоблин лучник",
    "Орк-багатур": "Орк чемпион",
    "Зеленый дракон": "Зелёный дракон",
    "Гигантский черный паук": "Гигантский чёрный паук",
    "Скелет-рыцарь": "Скелет рыцарь",
}

# Localized names are not unique in Gunits.dbf: the neutral and Elven Alliance
# variants of these units share English display names. Prefer the stable type ID
# when a scenario is matched against DATA.
DATA_TYPE_ID_ALIASES = {
    "g000uu5008": "Нейтральный лорд эльфов",
    "g000uu5009": "Нейтральный эльф-рейнджер",
    "g000uu5004": "Нейтральный грифон",
    "g000uu5010": "Нейтральный повелитель небес",
    "g000uu5006": "Нейтральный эльфийский оракул",
    "g000uu5011": "Нейтральный кентавр-копейщик",
}

CAMPAIGN_STACK_NAME_ALIAS_PAIRS = (
    ("Гоблин-лучник", "Гоблин лучник"),
    ("Зеленый дракон", "Зелёный дракон"),
    ("Зeлeный дракон", "Зелёный дракон"),
    ("Орк-багатур", "Орк чемпион"),
    ("Эльф-следопыт", "Рейнджер"),
    ("Послушница", "Служка"),
    ("Оборотнь", "Оборотень"),
    ("Сквайр", "Скваер"),
    ("Cквайр", "Скваер"),
    ("Squire", "Скваер"),
    ("Peasant", "Крестьянин"),
    ("Wolf", "Волк"),
    ("Titan", "Титан"),
    ("Dwarf", "Гном"),
    ("Гиганткий паук", "Гигантский паук"),
    ("Копeйщик", "Копейщик"),
)


def u32(blob: bytes) -> int:
    return struct.unpack("<I", blob)[0]


def parse_map_size(data: bytes) -> int:
    match = MAP_SIZE_RE.search(data)
    if match is None:
        raise ValueError("MAP_SIZE field not found")
    return u32(match.group(1))


def parse_mountains(data: bytes, map_size: int) -> tuple[list[dict], set[tuple[int, int]]]:
    mount_start = data.find(b".?AVCMidMountains@@")
    if mount_start < 0:
        return [], set()
    mount_end = data.find(b"ENDOBJECT", mount_start)
    if mount_end < 0:
        raise ValueError("AVCMidMountains object is not terminated")

    chunk = data[mount_start:mount_end]
    entries: list[dict] = []
    obstacle_cells: set[tuple[int, int]] = set()

    for match in MOUNTAIN_RE.finditer(chunk):
        id_mount, size_x, size_y, pos_x, pos_y, image, race = (
            u32(match.group(i)) for i in range(1, 8)
        )
        entry = {
            "id": id_mount,
            "x": pos_x,
            "y": pos_y,
            "w": size_x,
            "h": size_y,
            "image": image,
            "race": race,
        }
        entries.append(entry)
        for x in range(pos_x, min(map_size, pos_x + size_x)):
            for y in range(pos_y, min(map_size, pos_y + size_y)):
                obstacle_cells.add((x, y))

    return entries, obstacle_cells


def parse_stacks(data: bytes) -> list[dict]:
    stacks: list[dict] = []
    for match in STACK_RE.finditer(data):
        stacks.append(
            {
                "obj_id": match.group(2).decode("ascii"),
                "owner": match.group(6).decode("ascii"),
                "x": u32(match.group(3)),
                "y": u32(match.group(4)),
            }
        )
    return stacks


def parse_positioned_objects(data: bytes) -> list[dict]:
    objects: list[dict] = []
    for match in CLASS_RE.finditer(data):
        start = match.start()
        end = data.find(b"ENDOBJECT", start)
        if end < 0:
            continue
        chunk = data[start:end]
        cls = match.group(2).decode("ascii", "ignore")
        if any(skip in cls for skip in SKIP_POSITION_CLASSES):
            continue

        pos_x_idx = chunk.find(b"POS_X")
        pos_y_idx = chunk.find(b"POS_Y")
        if pos_x_idx < 0 or pos_y_idx < 0:
            continue

        objects.append(
            {
                "class": cls,
                "x": struct.unpack_from("<I", chunk, pos_x_idx + 5)[0],
                "y": struct.unpack_from("<I", chunk, pos_y_idx + 5)[0],
            }
        )
    return objects


def parse_dbf_rows(path: Path, *, encoding: str = "cp866") -> list[dict[str, str]]:
    data = path.read_bytes()
    num_records = struct.unpack("<I", data[4:8])[0]
    header_len = struct.unpack("<H", data[8:10])[0]
    record_len = struct.unpack("<H", data[10:12])[0]

    fields: list[tuple[str, int]] = []
    offset = 32
    while offset < header_len and data[offset] != 0x0D:
        name = data[offset : offset + 11].split(b"\x00", 1)[0].decode("ascii", "ignore")
        field_len = data[offset + 16]
        fields.append((name, field_len))
        offset += 32

    rows: list[dict[str, str]] = []
    start = header_len
    for index in range(num_records):
        rec = data[start + index * record_len : start + (index + 1) * record_len]
        if not rec or rec[0] in (0x1A, 0x2A):
            continue
        pos = 1
        row: dict[str, str] = {}
        for name, field_len in fields:
            raw = rec[pos : pos + field_len]
            pos += field_len
            row[name] = raw.decode(encoding, "ignore").strip()
        rows.append(row)
    return rows


def find_string_field(chunk: bytes, field_name: bytes, *, encoding: str = "ascii") -> str | None:
    index = chunk.find(field_name)
    if index < 0:
        return None
    size_start = index + len(field_name)
    if size_start + 4 > len(chunk):
        return None
    size = u32(chunk[size_start : size_start + 4])
    value_start = size_start + 4
    value_end = min(len(chunk), value_start + size)
    raw = chunk[value_start:value_end].rstrip(b"\x00")
    return raw.decode(encoding, "ignore").strip()


def find_int_field(chunk: bytes, field_name: bytes) -> int | None:
    index = chunk.find(field_name)
    if index < 0:
        return None
    value_start = index + len(field_name)
    if value_start + 4 > len(chunk):
        return None
    return u32(chunk[value_start : value_start + 4])


def iter_object_chunks(data: bytes, class_name: bytes) -> list[bytes]:
    chunks: list[bytes] = []
    start = 0
    marker = class_name + b"\x00"
    while True:
        index = data.find(marker, start)
        if index < 0:
            break
        end = data.find(b"ENDOBJECT", index)
        if end < 0:
            break
        chunks.append(data[index:end])
        start = end + len(b"ENDOBJECT")
    return chunks


def parse_mapblock_tile_grid(data: bytes, map_size: int) -> list[list[int]]:
    grid = [[0 for _ in range(map_size)] for _ in range(map_size)]
    for chunk in iter_object_chunks(data, b".?AVCMidgardMapBlock@@"):
        block_id = find_string_field(chunk, b"BLOCKID")
        if not block_id or len(block_id) < 4:
            continue
        suffix = block_id[-4:]
        block_y = int(suffix[:2], 16)
        block_x = int(suffix[2:], 16)
        data_index = chunk.find(b"BLOCKDATA")
        if data_index < 0:
            continue
        payload_size = u32(chunk[data_index + 9 : data_index + 13])
        payload = chunk[data_index + 13 : data_index + 13 + payload_size]
        values = [
            u32(payload[offset : offset + 4]) & 0xFF
            for offset in range(0, len(payload), 4)
            if offset + 4 <= len(payload)
        ]
        for index, value in enumerate(values):
            local_y = index // MAPBLOCK_WIDTH
            local_x = index % MAPBLOCK_WIDTH
            x = block_x + local_x
            y = block_y + local_y
            if 0 <= x < map_size and 0 <= y < map_size:
                grid[y][x] = value
    return grid


def tile_ground_id(tile_code: int) -> int:
    return (int(tile_code) >> 3) & 7


def parse_ground_cells(data: bytes, map_size: int, ground_id: int) -> set[tuple[int, int]]:
    tile_grid = parse_mapblock_tile_grid(data, map_size)
    normalized_ground_id = int(ground_id)
    return {
        (x, y)
        for y in range(map_size)
        for x in range(map_size)
        if tile_ground_id(tile_grid[y][x]) == normalized_ground_id
    }


def parse_water_cells(data: bytes, map_size: int) -> set[tuple[int, int]]:
    tile_grid = parse_mapblock_tile_grid(data, map_size)
    return {
        (x, y)
        for y in range(map_size)
        for x in range(map_size)
        if tile_grid[y][x] in WATER_TILE_CODES
    }


def parse_forest_cells(data: bytes, map_size: int) -> set[tuple[int, int]]:
    return parse_ground_cells(data, map_size, GROUND_FOREST_ID)


def parse_road_cells(data: bytes, map_size: int | None = None) -> set[tuple[int, int]]:
    road_cells: set[tuple[int, int]] = set()
    for obj in parse_positioned_objects(data):
        if str(obj.get("class", "") or "") != ROAD_CLASS_NAME:
            continue
        x = int(obj.get("x", 0) or 0)
        y = int(obj.get("y", 0) or 0)
        if map_size is not None and not (0 <= x < int(map_size) and 0 <= y < int(map_size)):
            continue
        road_cells.add((x, y))
    return road_cells


def parse_unit_objects(data: bytes) -> dict[str, dict]:
    units: dict[str, dict] = {}
    for chunk in iter_object_chunks(data, b".?AVCMidUnit@@"):
        unit_id = find_string_field(chunk, b"UNIT_ID")
        if not unit_id:
            continue
        units[unit_id] = {
            "unit_id": unit_id,
            "type_id": find_string_field(chunk, b"TYPE"),
            "level": find_int_field(chunk, b"LEVEL"),
            "name_override": find_string_field(chunk, b"NAME_TXT", encoding="cp1251") or "",
            "hp": find_int_field(chunk, b"HP"),
            "xp": find_int_field(chunk, b"XP"),
        }
    return units


def parse_stack_objects(data: bytes) -> list[dict]:
    stacks: list[dict] = []
    for chunk in iter_object_chunks(data, b".?AVCMidStack@@"):
        stack_id = find_string_field(chunk, b"STACK_ID")
        if not stack_id:
            continue
        unit_ids = []
        positions = []
        for slot in range(6):
            unit_ids.append(find_string_field(chunk, f"UNIT_{slot}".encode("ascii")) or "")
            positions.append(find_int_field(chunk, f"POS_{slot}".encode("ascii")))
        stacks.append(
            {
                "stack_id": stack_id,
                "group_id": find_string_field(chunk, b"GROUP_ID"),
                "src_template_id": find_string_field(chunk, b"SRCTMPL_ID"),
                "leader_id": find_string_field(chunk, b"LEADER_ID"),
                "owner": find_string_field(chunk, b"OWNER"),
                "inside": find_string_field(chunk, b"INSIDE"),
                "subrace": find_string_field(chunk, b"SUBRACE"),
                "x": find_int_field(chunk, b"POS_X"),
                "y": find_int_field(chunk, b"POS_Y"),
                "move": find_int_field(chunk, b"MOVE"),
                "create_level": find_int_field(chunk, b"CREAT_LVL"),
                "unit_ids": unit_ids,
                "positions": positions,
            }
        )
    return stacks


def load_global_unit_name_map(scenario_path: Path) -> dict[str, dict]:
    root = scenario_path.parent.parent
    gunits_rows = parse_dbf_rows(root / "Globals" / "Gunits.dbf")
    text_rows = parse_dbf_rows(root / "Globals" / "Tglobal.dbf")

    text_by_id = {row["TXT_ID"].lower(): row["TEXT"] for row in text_rows if row.get("TXT_ID")}
    units: dict[str, dict] = {}
    for row in gunits_rows:
        unit_id = row.get("UNIT_ID", "").lower()
        if not unit_id:
            continue
        text_id = row.get("NAME_TXT", "").lower()
        units[unit_id] = {
            "unit_id": row.get("UNIT_ID", ""),
            "name": text_by_id.get(text_id, row.get("NAME_TXT", "")),
            "global_level": row.get("LEVEL", ""),
            "unit_cat": row.get("UNIT_CAT", ""),
        }
    return units


def load_data_unit_names(data_file: Path) -> set[str]:
    import ast

    text = data_file.read_text(encoding="utf-8-sig")
    module = ast.parse(text)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != "DATA":
                continue
            data_value = ast.literal_eval(node.value)
            if not isinstance(data_value, list) or not data_value:
                return set()
            name_key = next(
                (
                    key
                    for key in data_value[0].keys()
                    if isinstance(key, str) and key.lower() == "кто"
                ),
                None,
            )
            if name_key is None:
                return set()
            return {
                row[name_key]
                for row in data_value
                if isinstance(row, dict) and isinstance(row.get(name_key), str)
            }
    return set()


def normalize_name(name: str) -> str:
    normalized = (name or "").replace("ё", "е").replace("Ё", "Е")
    normalized = "".join(
        ch for ch in normalized if ch.isalnum() or ch.isspace() or ch == "-"
    )
    return " ".join(normalized.lower().split())


CAMPAIGN_STACK_NAME_ALIASES = {
    " ".join(normalize_name(source).replace("-", " ").split()):
    " ".join(normalize_name(target).replace("-", " ").split())
    for source, target in CAMPAIGN_STACK_NAME_ALIAS_PAIRS
}


def canonicalize_campaign_stack_name(name: str) -> str:
    normalized = normalize_name(unicodedata.normalize("NFKC", name or ""))
    normalized = " ".join(normalized.replace("-", " ").split())
    return CAMPAIGN_STACK_NAME_ALIASES.get(normalized, normalized)


def build_data_name_comparison(
    scenario_path: Path,
    data_file: Path,
    *,
    stack_report: list[dict] | None = None,
) -> dict:
    global_units = load_global_unit_name_map(scenario_path)
    report_rows = stack_report if stack_report is not None else build_stack_composition_report(scenario_path)
    data_names = load_data_unit_names(data_file)
    data_norm = {normalize_name(name): name for name in data_names}

    results: list[dict] = []
    seen_type_ids: set[str] = set()
    for row in report_rows:
        for unit in row["composition"]:
            type_id = (unit.get("type_id") or "").lower()
            if not type_id or type_id in seen_type_ids:
                continue
            seen_type_ids.add(type_id)
            canonical_name = global_units.get(type_id, {}).get("name", unit.get("name", ""))
            mapped_name = DATA_TYPE_ID_ALIASES.get(
                type_id,
                DATA_NAME_ALIASES.get(canonical_name, canonical_name),
            )
            exact_match = data_norm.get(normalize_name(mapped_name))
            results.append(
                {
                    "type_id": unit.get("type_id"),
                    "canonical_name": canonical_name,
                    "mapped_name": mapped_name,
                    "matched_data_name": exact_match,
                }
            )

    results.sort(key=lambda row: (row["matched_data_name"] is None, row["canonical_name"], row["type_id"]))
    return {
        "matched": [row for row in results if row["matched_data_name"] is not None],
        "unmatched": [row for row in results if row["matched_data_name"] is None],
    }


def build_stack_composition_report(scenario_path: Path) -> list[dict]:
    data = scenario_path.read_bytes()
    unit_objects = parse_unit_objects(data)
    global_units = load_global_unit_name_map(scenario_path)
    stacks = parse_stack_objects(data)

    report_rows: list[dict] = []
    for stack in stacks:
        composition: list[dict] = []
        for slot, unit_id in enumerate(stack["unit_ids"]):
            if not unit_id or unit_id == "G000000000":
                continue
            unit_info = unit_objects.get(unit_id, {})
            type_id = unit_info.get("type_id") or unit_id
            global_info = global_units.get(type_id.lower(), {})
            name = unit_info.get("name_override") or global_info.get("name") or type_id
            composition.append(
                {
                    "slot": slot,
                    "battle_pos": stack["positions"][slot],
                    "unit_id": unit_id,
                    "type_id": type_id,
                    "name": name,
                    "level": unit_info.get("level"),
                    "hp": unit_info.get("hp"),
                    "is_leader": unit_id == stack.get("leader_id"),
                }
            )
        report_rows.append(
            {
                "stack_id": stack["stack_id"],
                "owner": stack.get("owner"),
                "inside": stack.get("inside"),
                "x": stack.get("x"),
                "y": stack.get("y"),
                "leader_id": stack.get("leader_id"),
                "composition": composition,
            }
        )

    report_rows.sort(
        key=lambda row: (
            row.get("owner") or "",
            int(row.get("y") or 0),
            int(row.get("x") or 0),
            row.get("stack_id") or "",
        )
    )
    return report_rows


def format_stack_report(report_rows: list[dict]) -> str:
    lines: list[str] = []
    for row in report_rows:
        inside_suffix = f", inside={row['inside']}" if row.get("inside") and row["inside"] != "G000000000" else ""
        units = []
        for unit in row["composition"]:
            leader_prefix = "*" if unit["is_leader"] else ""
            level_suffix = f" Lv{unit['level']}" if unit.get("level") is not None else ""
            hp_suffix = f" HP{unit['hp']}" if unit.get("hp") is not None else ""
            units.append(
                f"{leader_prefix}{unit['name']} [{unit['unit_id']} -> {unit['type_id']}{level_suffix}{hp_suffix}]"
            )
        units_str = "; ".join(units) if units else "(empty)"
        lines.append(
            f"{row['stack_id']} owner={row.get('owner')} pos=({row.get('x')},{row.get('y')}){inside_suffix}: {units_str}"
        )
    return "\n".join(lines)


def build_campaign_stack_match_report(
    scenario_path: Path,
    *,
    stack_report: list[dict] | None = None,
) -> dict:
    from grid import ENEMY_TEAM_SPECS

    report_rows = stack_report if stack_report is not None else build_stack_composition_report(scenario_path)
    campaign_by_signature: dict[tuple[str, ...], dict] = {}
    for enemy_id, spec in ENEMY_TEAM_SPECS.items():
        composition_names = [
            name
            for name in list(spec["front"]) + list(spec["back"])
            if isinstance(name, str) and name
        ]
        signature = tuple(sorted(canonicalize_campaign_stack_name(name) for name in composition_names))
        campaign_by_signature[signature] = {
            "enemy_id": int(enemy_id),
            "description": str(spec["description"]),
            "composition_names": composition_names,
            "signature": list(signature),
        }

    matched_map_stacks: list[dict] = []
    unmatched_map_stacks: list[dict] = []
    other_owner_stacks: list[dict] = []

    for row in report_rows:
        stack_entry = {
            "stack_id": row.get("stack_id"),
            "owner": row.get("owner"),
            "inside": row.get("inside"),
            "x": row.get("x"),
            "y": row.get("y"),
            "composition_names": [str(unit.get("name") or "") for unit in row["composition"]],
        }
        if row.get("owner") != NEUTRAL_ENEMY_OWNER:
            other_owner_stacks.append(stack_entry)
            continue

        signature = tuple(
            sorted(canonicalize_campaign_stack_name(name) for name in stack_entry["composition_names"])
        )
        stack_entry["signature"] = list(signature)
        campaign_match = campaign_by_signature.get(signature)
        if campaign_match is None:
            unmatched_map_stacks.append(stack_entry)
            continue
        matched_map_stacks.append(
            {
                **stack_entry,
                "campaign_enemy_id": campaign_match["enemy_id"],
                "campaign_description": campaign_match["description"],
                "campaign_composition_names": campaign_match["composition_names"],
            }
        )

    matched_enemy_ids = {row["campaign_enemy_id"] for row in matched_map_stacks}
    unmatched_campaign_templates = [
        campaign_by_signature[signature]
        for signature in sorted(campaign_by_signature, key=lambda item: campaign_by_signature[item]["enemy_id"])
        if campaign_by_signature[signature]["enemy_id"] not in matched_enemy_ids
    ]

    return {
        "neutral_enemy_owner": NEUTRAL_ENEMY_OWNER,
        "matched_map_stacks": matched_map_stacks,
        "unmatched_map_stacks": unmatched_map_stacks,
        "other_owner_stacks": other_owner_stacks,
        "matched_campaign_enemy_ids": sorted(matched_enemy_ids),
        "unmatched_campaign_templates": unmatched_campaign_templates,
    }


def format_campaign_stack_match_report(report: dict) -> str:
    lines: list[str] = []
    matched_rows = report["matched_map_stacks"]
    unmatched_rows = report["unmatched_map_stacks"]
    other_owner_rows = report["other_owner_stacks"]

    lines.append(f"Matched neutral map stacks: {len(matched_rows)}")
    for row in matched_rows:
        inside_suffix = f", inside={row['inside']}" if row.get("inside") and row["inside"] != "G000000000" else ""
        composition = ", ".join(row["composition_names"]) if row["composition_names"] else "(empty)"
        lines.append(
            f"- {row['stack_id']} pos=({row['x']},{row['y']}){inside_suffix} -> "
            f"campaign #{row['campaign_enemy_id']}: {composition}"
        )

    lines.append("")
    lines.append(f"Unmatched neutral map stacks: {len(unmatched_rows)}")
    for row in unmatched_rows:
        inside_suffix = f", inside={row['inside']}" if row.get("inside") and row["inside"] != "G000000000" else ""
        composition = ", ".join(row["composition_names"]) if row["composition_names"] else "(empty)"
        lines.append(f"- {row['stack_id']} pos=({row['x']},{row['y']}){inside_suffix}: {composition}")

    lines.append("")
    lines.append(f"Other-owner stacks ignored: {len(other_owner_rows)}")
    for row in other_owner_rows:
        inside_suffix = f", inside={row['inside']}" if row.get("inside") and row["inside"] != "G000000000" else ""
        composition = ", ".join(row["composition_names"]) if row["composition_names"] else "(empty)"
        lines.append(
            f"- {row['stack_id']} owner={row['owner']} pos=({row['x']},{row['y']}){inside_suffix}: {composition}"
        )

    lines.append("")
    lines.append("Campaign enemy templates missing on the map:")
    for template in report["unmatched_campaign_templates"]:
        composition = ", ".join(template["composition_names"]) if template["composition_names"] else "(empty)"
        lines.append(f"- #{template['enemy_id']}: {template['description']} [{composition}]")

    return "\n".join(lines)


def extract_render_objects(data: bytes) -> dict[str, list[dict]]:
    objects_by_class: dict[str, list[dict]] = defaultdict(list)
    for obj in parse_positioned_objects(data):
        if obj["class"] in RENDER_OBJECT_SPECS:
            objects_by_class[obj["class"]].append(obj)
    for obj in parse_resource_objects(data):
        objects_by_class[obj["class"]].append(obj)
    return dict(objects_by_class)


def parse_settlement_objects(data: bytes) -> dict[tuple[str, int, int], dict]:
    settlements: dict[tuple[str, int, int], dict] = {}
    for class_name in (b".?AVCCapital@@", b".?AVCMidVillage@@"):
        class_str = class_name.decode("ascii")
        for chunk in iter_object_chunks(data, class_name):
            x = find_int_field(chunk, b"POS_X")
            y = find_int_field(chunk, b"POS_Y")
            if x is None or y is None:
                continue
            stack_id = find_string_field(chunk, b"STACK") or ""
            garrison_units = []
            for slot in range(6):
                unit_id = find_string_field(chunk, f"UNIT_{slot}".encode("ascii")) or ""
                if unit_id and unit_id != "G000000000":
                    garrison_units.append(unit_id)
            settlements[(class_str, x, y)] = {
                "class": class_str,
                "x": x,
                "y": y,
                "stack_id": stack_id,
                "has_stack": bool(stack_id and stack_id != "G000000000"),
                "garrison_units": garrison_units,
                "has_garrison": bool(garrison_units),
            }
    return settlements


def build_settlement_entry_cells(
    settlement_objects: dict[tuple[str, int, int], dict],
) -> dict[tuple[int, int], dict]:
    entry_cells: dict[tuple[int, int], dict] = {}
    for settlement in settlement_objects.values():
        spec = RENDER_OBJECT_SPECS.get(settlement["class"], {})
        footprint_w, footprint_h = spec.get("footprint", (1, 1))
        corner_tiles = [
            (
                int(settlement["x"]),
                int(settlement["y"]) + max(0, int(footprint_h) - 1),
            ),
            (
                int(settlement["x"]) + max(0, int(footprint_w) - 1),
                int(settlement["y"]),
            ),
        ]
        for tile in corner_tiles:
            entry_cells[tile] = {
                "class": settlement["class"],
                "label": spec.get("label", "Settlement"),
                "x": tile[0],
                "y": tile[1],
            }
    return entry_cells


def parse_resource_objects(data: bytes) -> list[dict]:
    resources: list[dict] = []
    for chunk in iter_object_chunks(data, b".?AVCMidCrystal@@"):
        resource_id = find_int_field(chunk, b"RESOURCE")
        x = find_int_field(chunk, b"POS_X")
        y = find_int_field(chunk, b"POS_Y")
        if resource_id is None or x is None or y is None:
            continue
        render_key = f"resource:{resource_id}"
        if render_key not in RENDER_OBJECT_SPECS:
            continue
        resources.append(
            {
                "class": render_key,
                "x": x,
                "y": y,
                "resource_id": resource_id,
            }
        )
    return resources


def settlement_overlay_code(spec_code: str, settlement: dict | None) -> str:
    if not settlement:
        return spec_code
    if settlement["has_garrison"] and settlement["has_stack"]:
        return "G+S"
    if settlement["has_garrison"]:
        return "G"
    if settlement["has_stack"]:
        return "S"
    return spec_code


def build_summary(data: bytes, *, include_cells: bool = False) -> dict:
    map_size = parse_map_size(data)
    mountains, obstacle_cells = parse_mountains(data, map_size)
    water_cells = parse_water_cells(data, map_size)
    stacks = parse_stacks(data)
    positioned_objects = parse_positioned_objects(data)
    render_objects = extract_render_objects(data)
    settlement_objects = parse_settlement_objects(data)
    settlement_entry_cells = build_settlement_entry_cells(settlement_objects)

    stack_cells = {(stack["x"], stack["y"]) for stack in stacks}
    other_object_cells = {
        (obj["x"], obj["y"])
        for obj in positioned_objects
        if obj["class"] != ".?AVCMidStack@@"
    }
    non_empty_cells = set(obstacle_cells) | stack_cells | other_object_cells
    schematic_non_empty_cells = set(obstacle_cells) | stack_cells

    stacks_by_owner: dict[str, list[dict]] = defaultdict(list)
    for stack in stacks:
        stacks_by_owner[stack["owner"]].append(
            {"obj_id": stack["obj_id"], "x": stack["x"], "y": stack["y"]}
        )

    summary = {
        "map_size": map_size,
        "map_area": map_size * map_size,
        "mountain_entries": len(mountains),
        "obstacle_cells": len(obstacle_cells),
        "water_cells": len(water_cells),
        "stack_count": len(stacks),
        "stack_cells": len(stack_cells),
        "other_positioned_object_count": len(positioned_objects) - len(stacks),
        "other_positioned_object_cells": len(other_object_cells),
        "non_empty_cells": len(non_empty_cells),
        "empty_cells": map_size * map_size - len(non_empty_cells),
        "schematic_non_empty_cells": len(schematic_non_empty_cells),
        "schematic_empty_cells": map_size * map_size - len(schematic_non_empty_cells),
        "stack_owner_counts": dict(Counter(stack["owner"] for stack in stacks)),
        "object_type_counts": dict(Counter(obj["class"] for obj in positioned_objects)),
        "render_object_counts": {
            RENDER_OBJECT_SPECS[cls]["label"]: len(rows)
            for cls, rows in sorted(render_objects.items())
        },
        "settlement_entry_count": len(settlement_entry_cells),
        "stack_samples_by_owner": {
            owner: rows[:20] for owner, rows in sorted(stacks_by_owner.items())
        },
        "mountain_samples": mountains[:20],
    }

    if include_cells:
        all_cells = {
            (x, y)
            for x in range(map_size)
            for y in range(map_size)
        }
        summary["obstacle_cell_coords"] = sorted([list(cell) for cell in obstacle_cells])
        summary["stack_cell_coords"] = sorted([list(cell) for cell in stack_cells])
        summary["other_object_cell_coords"] = sorted([list(cell) for cell in other_object_cells])
        summary["settlement_entry_cell_coords"] = sorted([list(cell) for cell in settlement_entry_cells])
        summary["empty_cell_coords"] = sorted(
            [list(cell) for cell in (all_cells - non_empty_cells)]
        )

    return summary


def render_ascii_map(data: bytes) -> str:
    summary = build_summary(data)
    map_size = int(summary["map_size"])
    mountains, obstacle_cells = parse_mountains(data, map_size)
    _ = mountains
    stacks = parse_stacks(data)
    stack_cells = {(stack["x"], stack["y"]) for stack in stacks}

    lines: list[str] = []
    for y in range(map_size):
        row_chars: list[str] = []
        for x in range(map_size):
            tile = (x, y)
            if tile in stack_cells:
                row_chars.append("E")
            elif tile in obstacle_cells:
                row_chars.append("#")
            else:
                row_chars.append(".")
        lines.append("".join(row_chars))
    return "\n".join(lines)


def transform_tile(x: int, y: int, map_size: int, *, rotate_clockwise: bool) -> tuple[int, int]:
    if not rotate_clockwise:
        return x, y
    return map_size - 1 - y, x


def transform_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    map_size: int,
    *,
    rotate_clockwise: bool,
) -> tuple[int, int, int, int]:
    if not rotate_clockwise:
        return x, y, width, height
    return map_size - y - height, x, height, width


def text_fill_for_background(color: tuple[int, int, int]) -> tuple[int, int, int]:
    brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
    return (0, 0, 0) if brightness >= 160 else (255, 255, 255)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    if not text:
        return
    left, top, right, bottom = bounds
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = left + max(0, (right - left - text_width) // 2)
    y = top + max(0, (bottom - top - text_height) // 2)
    draw.text((x, y), text, fill=fill, font=font)


def render_png_map(
    data: bytes,
    output_path: Path,
    *,
    cell_size: int = 12,
    rotate_clockwise: bool = True,
    stack_marker_colors: dict[tuple[int, int], tuple[int, int, int]] | None = None,
    stack_legend_items: list[tuple[str, str, tuple[int, int, int], int]] | None = None,
    fill_stack_cells: bool = True,
) -> None:
    summary = build_summary(data)
    map_size = int(summary["map_size"])
    mountains, obstacle_cells = parse_mountains(data, map_size)
    water_cells = parse_water_cells(data, map_size)
    _ = mountains
    stacks = parse_stacks(data)
    stack_cells = {(stack["x"], stack["y"]) for stack in stacks}
    render_objects = extract_render_objects(data)
    settlement_objects = parse_settlement_objects(data)
    rendered_ruin_interaction_tiles: set[tuple[int, int]] = set()
    suppressed_stack_cells = {
        (info["x"], info["y"])
        for info in settlement_objects.values()
        if info["has_stack"]
    }

    legend_items: list[tuple[str, str, tuple[int, int, int], int]] = [
        ("", "Empty land", (255, 255, 255), summary["schematic_empty_cells"] - summary["water_cells"]),
        ("~", "Water", (151, 203, 255), summary["water_cells"]),
        ("#", "Obstacle", (140, 140, 140), summary["obstacle_cells"]),
    ]
    for cls in RENDER_ORDER:
        rows = render_objects.get(cls, [])
        if rows:
            legend_items.append(
                (
                    RENDER_OBJECT_SPECS[cls]["code"],
                    RENDER_OBJECT_SPECS[cls]["label"],
                    RENDER_OBJECT_SPECS[cls]["color"],
                    len(rows),
                )
            )
    ruin_count = len(render_objects.get(".?AVCMidRuin@@", []))
    if ruin_count:
        legend_items.append(("RI", "Ruin interaction", RUIN_INTERACTION_COLOR, ruin_count * 3))
    if stack_legend_items is None:
        legend_items.append(("E", "Stack / enemy", (210, 45, 45), summary["stack_count"]))
    else:
        legend_items.extend(stack_legend_items)

    legend_width = 240
    width = map_size * cell_size + legend_width
    height = max(map_size * cell_size, 28 + 22 * len(legend_items) + 48)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    obstacle_color = (140, 140, 140)
    water_color = (151, 203, 255)
    stack_color = (210, 45, 45)
    grid_color = (225, 225, 225)
    outline_color = (35, 35, 35)
    map_pixel_width = map_size * cell_size
    map_pixel_height = map_size * cell_size

    for y in range(map_size):
        for x in range(map_size):
            tile = (x, y)
            if fill_stack_cells and tile in stack_cells:
                fill = stack_color
            elif tile in obstacle_cells:
                fill = obstacle_color
            elif tile in water_cells:
                fill = water_color
            else:
                fill = (255, 255, 255)

            draw_x, draw_y = transform_tile(x, y, map_size, rotate_clockwise=rotate_clockwise)
            x0 = draw_x * cell_size
            y0 = draw_y * cell_size
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1
            draw.rectangle((x0, y0, x1, y1), fill=fill)

    object_label_bounds: list[tuple[str, tuple[int, int, int], tuple[int, int, int, int]]] = []
    for cls in RENDER_ORDER:
        spec = RENDER_OBJECT_SPECS[cls]
        for obj in render_objects.get(cls, []):
            overlay_code = settlement_overlay_code(spec["code"], settlement_objects.get((cls, obj["x"], obj["y"])))
            footprint_w, footprint_h = spec["footprint"]
            draw_x, draw_y, draw_w, draw_h = transform_rect(
                obj["x"],
                obj["y"],
                footprint_w,
                footprint_h,
                map_size,
                rotate_clockwise=rotate_clockwise,
            )
            x0 = draw_x * cell_size
            y0 = draw_y * cell_size
            x1 = min(map_pixel_width - 1, x0 + draw_w * cell_size - 1)
            y1 = min(map_pixel_height - 1, y0 + draw_h * cell_size - 1)
            inset = max(1, cell_size // 8)
            draw.rectangle(
                (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
                fill=spec["color"],
                outline=outline_color,
                width=1,
            )
            object_label_bounds.append(
                (
                    overlay_code,
                    spec["color"],
                    (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
                )
            )
            if cls == ".?AVCMidRuin@@":
                rendered_ruin_interaction_tiles.add((draw_x, draw_y + draw_h - 1))
                if draw_h >= 2:
                    rendered_ruin_interaction_tiles.add((draw_x, draw_y + draw_h - 2))
                if draw_w >= 2:
                    rendered_ruin_interaction_tiles.add((draw_x + 1, draw_y + draw_h - 1))

    stack_inset = max(1, cell_size // 5)
    for x, y in stack_cells:
        if (x, y) in suppressed_stack_cells:
            continue
        draw_x, draw_y = transform_tile(x, y, map_size, rotate_clockwise=rotate_clockwise)
        x0 = draw_x * cell_size + stack_inset
        y0 = draw_y * cell_size + stack_inset
        x1 = x0 + cell_size - 2 * stack_inset - 1
        y1 = y0 + cell_size - 2 * stack_inset - 1
        marker_fill = (
            stack_marker_colors.get((x, y), stack_color)
            if stack_marker_colors is not None
            else stack_color
        )
        draw.ellipse((x0, y0, x1, y1), fill=marker_fill, outline=outline_color, width=1)

    if cell_size >= 6:
        for x in range(0, map_pixel_width, cell_size):
            draw.line((x, 0, x, map_pixel_height - 1), fill=grid_color, width=1)
        for y in range(0, map_pixel_height, cell_size):
            draw.line((0, y, map_pixel_width - 1, y), fill=grid_color, width=1)

    entry_inset = max(1, cell_size // 8)
    for draw_x, draw_y in rendered_ruin_interaction_tiles:
        x0 = draw_x * cell_size + entry_inset
        y0 = draw_y * cell_size + entry_inset
        x1 = x0 + cell_size - 2 * entry_inset - 1
        y1 = y0 + cell_size - 2 * entry_inset - 1
        draw.rectangle((x0, y0, x1, y1), fill=RUIN_INTERACTION_COLOR, outline=outline_color, width=1)

    for code, color, bounds in object_label_bounds:
        draw_centered_text(
            draw,
            bounds,
            code,
            font=font,
            fill=text_fill_for_background(color),
        )

    legend_x = map_pixel_width + 16
    draw.text((legend_x, 10), "Legend", fill=(0, 0, 0), font=font)
    cursor_y = 34
    for code, label, color, count in legend_items:
        swatch = (legend_x, cursor_y, legend_x + 14, cursor_y + 14)
        draw.rectangle(swatch, fill=color, outline=outline_color, width=1)
        legend_label = f"[{code}] {label}: {count}" if code else f"{label}: {count}"
        draw.text((legend_x + 22, cursor_y + 1), legend_label, fill=(0, 0, 0), font=font)
        cursor_y += 22
    draw.text(
        (legend_x, cursor_y + 4),
        "On C/V tiles: G = garrison, S = linked stack, G+S = both",
        fill=(0, 0, 0),
        font=font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_campaign_stack_match_png(
    data: bytes,
    output_path: Path,
    scenario_path: Path,
    *,
    cell_size: int = 12,
    rotate_clockwise: bool = True,
    stack_report: list[dict] | None = None,
) -> dict:
    match_report = build_campaign_stack_match_report(scenario_path, stack_report=stack_report)
    matched_color = (45, 165, 85)
    unmatched_color = (210, 45, 45)
    other_owner_color = (85, 115, 235)

    stack_marker_colors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for row in match_report["matched_map_stacks"]:
        stack_marker_colors[(int(row["x"]), int(row["y"]))] = matched_color
    for row in match_report["unmatched_map_stacks"]:
        stack_marker_colors[(int(row["x"]), int(row["y"]))] = unmatched_color
    for row in match_report["other_owner_stacks"]:
        stack_marker_colors[(int(row["x"]), int(row["y"]))] = other_owner_color

    legend_items = [
        ("M", "Campaign match", matched_color, len(match_report["matched_map_stacks"])),
        ("U", "Enemy stack: no campaign match", unmatched_color, len(match_report["unmatched_map_stacks"])),
    ]
    if match_report["other_owner_stacks"]:
        legend_items.append(("P", "Player / other stack", other_owner_color, len(match_report["other_owner_stacks"])))

    render_png_map(
        data,
        output_path,
        cell_size=cell_size,
        rotate_clockwise=rotate_clockwise,
        stack_marker_colors=stack_marker_colors,
        stack_legend_items=legend_items,
        fill_stack_cells=False,
    )
    return match_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Disciples II .sg map structure")
    parser.add_argument("path", nargs="?", type=Path, help="Path to .sg file")
    parser.add_argument(
        "--game-root",
        type=Path,
        help="Read a campaign scenario from the installed Disciples II DBF tables",
    )
    parser.add_argument(
        "--scenario-id",
        help="Campaign scenario id, for example S117SC0000",
    )
    parser.add_argument(
        "--map-data-output",
        type=Path,
        help="Write a deterministic static campaign-map JSON snapshot",
    )
    parser.add_argument(
        "--conversion-report-output",
        type=Path,
        help="Write the campaign conversion counts and source hashes",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full summary as JSON",
    )
    parser.add_argument(
        "--include-cells",
        action="store_true",
        help="Include full coordinate lists for obstacle, object and empty cells",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a simplified ASCII map (#=obstacle, E=stack, .=other/empty)",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        help="Write a PNG schematic map to this path",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=12,
        help="Cell size in pixels for PNG rendering",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Do not rotate the PNG map clockwise",
    )
    parser.add_argument(
        "--dump-stacks",
        action="store_true",
        help="Print all stack coordinates and compositions",
    )
    parser.add_argument(
        "--stacks-json-output",
        type=Path,
        help="Write stack coordinates and compositions to a JSON file",
    )
    parser.add_argument(
        "--stacks-text-output",
        type=Path,
        help="Write stack coordinates and compositions to a text report",
    )
    parser.add_argument(
        "--dump-campaign-matches",
        action="store_true",
        help="Print neutral map stacks that do or do not match campaign enemy templates",
    )
    parser.add_argument(
        "--campaign-match-json-output",
        type=Path,
        help="Write neutral stack/campaign match report to a JSON file",
    )
    parser.add_argument(
        "--campaign-match-text-output",
        type=Path,
        help="Write neutral stack/campaign match report to a text file",
    )
    parser.add_argument(
        "--campaign-match-png-output",
        type=Path,
        help="Write a PNG view that highlights stacks matching campaign enemy templates",
    )
    args = parser.parse_args()

    campaign_mode = bool(args.game_root or args.scenario_id or args.map_data_output)
    if campaign_mode:
        if not args.game_root or not args.scenario_id or not args.map_data_output:
            parser.error("--game-root, --scenario-id and --map-data-output are required together")
        from tools.d2_campaign_converter import convert_campaign

        _snapshot, report = convert_campaign(
            args.game_root,
            str(args.scenario_id),
            args.map_data_output,
            args.conversion_report_output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.path is None:
        parser.error("path is required unless campaign DBF mode is selected")

    data = args.path.read_bytes()
    summary = build_summary(data, include_cells=args.include_cells)
    stack_report = None
    campaign_match_report = None

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"path: {args.path}")
    print(f"map_size: {summary['map_size']}x{summary['map_size']}")
    print(f"mountain_entries: {summary['mountain_entries']}")
    print(f"obstacle_cells: {summary['obstacle_cells']}")
    print(f"water_cells: {summary['water_cells']}")
    print(f"stack_count: {summary['stack_count']}")
    print(f"empty_cells: {summary['empty_cells']}")
    if summary["render_object_counts"]:
        print("render_object_counts:")
        for label, count in summary["render_object_counts"].items():
            print(f"  {label}: {count}")
    print("stack_owner_counts:")
    for owner, count in summary["stack_owner_counts"].items():
        print(f"  {owner}: {count}")
    print("top_object_types:")
    for cls, count in Counter(summary["object_type_counts"]).most_common(10):
        print(f"  {cls}: {count}")
    if args.render:
        print("legend: # obstacle, E stack, . empty/other")
        print(render_ascii_map(data))
    if args.png_output:
        render_png_map(
            data,
            args.png_output,
            cell_size=max(1, args.cell_size),
            rotate_clockwise=not args.no_rotate,
        )
        print(f"png_output: {args.png_output.resolve()}")
    if (
        args.dump_stacks
        or args.stacks_json_output
        or args.stacks_text_output
        or args.dump_campaign_matches
        or args.campaign_match_json_output
        or args.campaign_match_text_output
        or args.campaign_match_png_output
    ):
        stack_report = build_stack_composition_report(args.path)
    if args.dump_stacks and stack_report is not None:
        print("stacks:")
        print(format_stack_report(stack_report))
    if args.stacks_json_output and stack_report is not None:
        args.stacks_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.stacks_json_output.write_text(
            json.dumps(stack_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"stacks_json_output: {args.stacks_json_output.resolve()}")
    if args.stacks_text_output and stack_report is not None:
        args.stacks_text_output.parent.mkdir(parents=True, exist_ok=True)
        args.stacks_text_output.write_text(format_stack_report(stack_report), encoding="utf-8")
        print(f"stacks_text_output: {args.stacks_text_output.resolve()}")
    if (
        args.dump_campaign_matches
        or args.campaign_match_json_output
        or args.campaign_match_text_output
        or args.campaign_match_png_output
    ) and stack_report is not None:
        campaign_match_report = build_campaign_stack_match_report(args.path, stack_report=stack_report)
    if args.dump_campaign_matches and campaign_match_report is not None:
        print("campaign_matches:")
        print(format_campaign_stack_match_report(campaign_match_report))
    if args.campaign_match_json_output and campaign_match_report is not None:
        args.campaign_match_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.campaign_match_json_output.write_text(
            json.dumps(campaign_match_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"campaign_match_json_output: {args.campaign_match_json_output.resolve()}")
    if args.campaign_match_text_output and campaign_match_report is not None:
        args.campaign_match_text_output.parent.mkdir(parents=True, exist_ok=True)
        args.campaign_match_text_output.write_text(
            format_campaign_stack_match_report(campaign_match_report),
            encoding="utf-8",
        )
        print(f"campaign_match_text_output: {args.campaign_match_text_output.resolve()}")
    if args.campaign_match_png_output and stack_report is not None:
        render_campaign_stack_match_png(
            data,
            args.campaign_match_png_output,
            args.path,
            cell_size=max(1, args.cell_size),
            rotate_clockwise=not args.no_rotate,
            stack_report=stack_report,
        )
        print(f"campaign_match_png_output: {args.campaign_match_png_output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
