#!/usr/bin/env python3
"""Read-only converter for Disciples II campaign scenarios stored in DBF tables."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from tools.inspect_sg_map import parse_dbf_rows, tile_ground_id


FORMAT_VERSION = 1
EVENT_TABLES = (
    "SEvent.dbf",
    "SEvCond.DBF",
    "SEvEffec.dbf",
    "SEvUnfog.dbf",
    "SLoc.dbf",
    "SStkTemp.dbf",
)
STATIC_TABLES = (
    "Sinfo.DBF",
    "SMapBlck.DBF",
    "SMounts.DBF",
    "SRoad.dbf",
    "Splayer.DBF",
    "SFort.DBF",
    "Sstack.dbf",
    "SUGroup.dbf",
    "SUnit.DBF",
    "Tscen.DBF",
    "SBag.dbf",
    "SInventr.DBF",
    "SItem.DBF",
    "SCrys.DBF",
    "SRuin.DBF",
    "SSite.DBF",
    "SMerch.DBF",
)
GLOBAL_TABLES = ("Gunits.dbf", "GItem.DBF", "Tglobal.dbf")

# Stable D2 type ids are deliberately used instead of localized display names.
# Every type present in S117 is mapped; conversion fails for any missing type.
UNIT_TYPE_TO_DATA_NAME = {
    "g000uu0001": "Скваер",
    "g000uu0002": "Рыцарь",
    "g000uu0004": "Охотник на ведьм",
    "g000uu0006": "Лучник",
    "g000uu0007": "Стрелок",
    "g000uu0008": "Ученик",
    "g000uu0011": "Служка",
    "g000uu0016": "Жрец",
    "g000uu0019": "Рыцарь на пегасе",
    "g000uu0027": "Арбалетчик",
    "g000uu0029": "Холмовой гигант",
    "g000uu0031": "Штормовой гигант",
    "g000uu0033": "Метатель топоров",
    "g000uu0036": "Гном",
    "g000uu0044": "Королевский страж",
    "g000uu0054": "Темный паладин",
    "g000uu0060": "Тварь",
    "g000uu0062": "Сектант",
    "g000uu0152": "Белый маг",
    "g000uu0154": "Имперский ассасин",
    "g000uu0155": "Защитник Веры",
    "g000uu0172": "Модеус",
    "g000uu3001": "Мизраэль",
    "g000uu3002": "Королевский страж",
    "g000uu5001": "Крестьянин",
    "g000uu5002": "Ополченец",
    "g000uu5003": "Копейщик",
    "g000uu5028": "Тритон",
    "g000uu5030": "Разбойник",
    "g000uu5031": "Головорез",
    "g000uu5032": "Вождь варваров",
    "g000uu5034": "Бес",
    "g000uu5040": "Варвар",
    "g000uu5103": "Копейщик",
    "g000uu5112": "Орк чемпион",
    "g000uu5113": "Орк",
    "g000uu5116": "Тролль",
    "g000uu5126": "Русалка",
    "g000uu5130": "Разбойник",
    "g000uu5131": "Головорез",
    "g000uu5132": "Вождь варваров",
    "g000uu5137": "Белый медведь",
    "g000uu5139": "Волк",
    "g000uu5201": "Скваер",
    "g000uu5210": "Волшебник",
    "g000uu5218": "Титан",
    "g000uu5229": "Холмовой гигант",
    "g000uu5230": "Скальной гигант",
    "g000uu5253": "Берсерк",
    "g000uu5354": "Имперский ассасин",
    "g000uu6012": "Дикий гигант",
    "g000uu6112": "Дикий гигант",
}

RESOURCE_NAMES = {0: "gold", 1: "infernal", 2: "life", 3: "death", 4: "runes", 5: "grove"}


def _int(value: object, default: int = 0) -> int:
    try:
        return int(str(value or default))
    except (TypeError, ValueError):
        return int(default)


def _gold(value: str) -> int:
    match = re.search(r"G(\d+)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _rows(path: Path, scenario_id: str | None = None) -> list[dict[str, str]]:
    rows = parse_dbf_rows(path)
    if scenario_id is None:
        return rows
    return [row for row in rows if row.get("NO_SCEN") == scenario_id]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_block_data(encoded: str) -> bytes:
    """Decode the game's printable six-bit DBF representation."""
    values = [ord(char) - 33 for char in str(encoded)]
    if any(value < 0 or value > 63 for value in values):
        raise ValueError("SMapBlck.BLOCKDATA contains an invalid six-bit character")
    output = bytearray()
    for offset in range(0, len(values), 4):
        chunk = values[offset : offset + 4]
        if len(chunk) >= 2:
            output.append((chunk[0] << 2) | (chunk[1] >> 4))
        if len(chunk) >= 3:
            output.append(((chunk[1] & 15) << 4) | (chunk[2] >> 2))
        if len(chunk) >= 4:
            output.append(((chunk[2] & 3) << 6) | chunk[3])
    if len(output) != 128:
        raise ValueError(f"Expected 128 decoded map-block bytes, got {len(output)}")
    return bytes(output)


def _terrain_grid(rows: Iterable[dict[str, str]], map_size: int) -> list[list[int]]:
    grid: list[list[int | None]] = [[None] * map_size for _ in range(map_size)]
    count = 0
    for row in rows:
        suffix = str(row["BLOCKID"])[-4:]
        block_y = int(suffix[:2], 16)
        block_x = int(suffix[2:], 16)
        decoded = _decode_block_data(row["BLOCKDATA"])
        values = [decoded[offset] for offset in range(0, len(decoded), 4)]
        if len(values) != 32:
            raise ValueError(f"Map block {row['BLOCKID']} does not contain 32 cells")
        for index, value in enumerate(values):
            x = block_x + index % 8
            y = block_y + index // 8
            if 0 <= x < map_size and 0 <= y < map_size:
                grid[y][x] = int(value)
        count += 1
    missing = [(x, y) for y in range(map_size) for x in range(map_size) if grid[y][x] is None]
    if missing:
        raise ValueError(f"Terrain has {len(missing)} missing cells; first={missing[0]}")
    expected_blocks = (map_size // 8) * (map_size // 4)
    if count != expected_blocks:
        raise ValueError(f"Expected {expected_blocks} map blocks, got {count}")
    return [[int(value) for value in row] for row in grid]


def _interaction_tile(kind: str, x: int, y: int) -> list[int]:
    if kind == "capital":
        return [x + 4, y + 2]
    if kind == "settlement":
        return [x + 3, y + 3]
    if kind == "ruin":
        return [x + 2, y + 2]
    return [x, y]


def _site_interaction_tiles(x: int, y: int) -> list[list[int]]:
    return [[x + 3, y + 1], [x + 3, y + 2], [x + 1, y + 3], [x + 2, y + 3], [x + 3, y + 3]]


def _assert_output_is_safe(game_root: Path, output: Path) -> None:
    resolved_game = game_root.resolve()
    resolved_output = output.resolve()
    protected = [resolved_game / name for name in ("Scens", "Campaign", "Globals", "Exports")]
    for root in protected:
        try:
            resolved_output.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"Refusing to write generated data inside Disciples II directory: {output}")


def build_campaign_snapshot(game_root: Path, scenario_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    game_root = game_root.resolve()
    scens = game_root / "Scens"
    globals_dir = game_root / "Globals"
    source_paths = [scens / name for name in STATIC_TABLES] + [globals_dir / name for name in GLOBAL_TABLES]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Disciples II tables: {missing}")
    hashes_before = {str(path.relative_to(game_root)): _sha256(path) for path in source_paths}

    info_rows = _rows(scens / "Sinfo.DBF", scenario_id)
    if len(info_rows) != 1:
        raise ValueError(f"Expected one Sinfo row for {scenario_id}, got {len(info_rows)}")
    info = info_rows[0]
    map_size = _int(info["MAP_SIZE"])
    texts = {
        (row["ID_OBJECT"], row["ID_STRING"]): row["TEXT"]
        for row in _rows(scens / "Tscen.DBF", scenario_id)
    }
    global_text = {
        row["TXT_ID"].lower(): row["TEXT"]
        for row in _rows(globals_dir / "Tglobal.dbf")
        if row.get("TXT_ID")
    }
    global_units = {
        row["UNIT_ID"].lower(): row for row in _rows(globals_dir / "Gunits.dbf")
    }
    global_items = {
        row["ITEM_ID"].lower(): row for row in _rows(globals_dir / "GItem.DBF")
    }

    unit_rows = {row["UNIT_ID"]: row for row in _rows(scens / "SUnit.DBF", scenario_id)}
    unknown_types = sorted(
        {row["TYPE"].lower() for row in unit_rows.values()} - set(UNIT_TYPE_TO_DATA_NAME)
    )
    if unknown_types:
        raise ValueError(f"Unmapped Disciples II unit type ids: {unknown_types}")

    def unit_entry(unit_id: str) -> dict[str, Any]:
        row = unit_rows.get(unit_id)
        if row is None:
            raise ValueError(f"Unknown scenario unit id {unit_id}")
        type_id = row["TYPE"].lower()
        global_row = global_units.get(type_id)
        if global_row is None:
            raise ValueError(f"Unknown global unit type {type_id}")
        return {
            "source_unit_id": unit_id,
            "type_id": type_id,
            "source_name": texts.get((unit_id, "NAME_TXT"), ""),
            "global_name": global_text.get(global_row.get("NAME_TXT", "").lower(), type_id),
            "unit_name": UNIT_TYPE_TO_DATA_NAME[type_id],
            "level": _int(row.get("LEVEL"), 1),
            "hp": _int(row.get("HP")),
            "xp": _int(row.get("XP")),
        }

    group_rows = {row["GROUP_ID"]: row for row in _rows(scens / "SUGroup.dbf", scenario_id)}

    def group_entry(group_id: str) -> dict[str, Any]:
        row = group_rows.get(group_id)
        if row is None:
            return {"source_group_id": group_id, "front": [None] * 3, "back": [None] * 3, "units": []}
        slots: list[dict[str, Any] | None] = [None] * 6
        units: list[dict[str, Any]] = []
        for battle_slot in range(6):
            unit_index = _int(row.get(f"POS_{battle_slot}"), -1)
            if unit_index < 0:
                continue
            unit_id = row.get(f"UNIT_{unit_index}", "")
            if not unit_id or unit_id == "G000000000":
                continue
            entry = unit_entry(unit_id)
            entry["battle_slot"] = battle_slot
            slots[battle_slot] = entry
            units.append(entry)
        return {
            "source_group_id": group_id,
            "front": [slot["unit_name"] if slot else None for slot in slots[:3]],
            "back": [slot["unit_name"] if slot else None for slot in slots[3:]],
            "units": units,
        }

    players = sorted(_rows(scens / "Splayer.DBF", scenario_id), key=lambda row: row["PLAYER_ID"])
    player_rows = [
        {"source_id": row["PLAYER_ID"], "race_id": row["RACE_ID"].lower(), "lord_id": row["LORD_ID"].lower()}
        for row in players
    ]

    forts = sorted(_rows(scens / "SFort.DBF", scenario_id), key=lambda row: row["CITY_ID"])
    settlements: list[dict[str, Any]] = []
    settlement_by_id: dict[str, dict[str, Any]] = {}
    for row in forts:
        kind = "capital" if not row.get("SIZE") else "settlement"
        x, y = _int(row["POS_X"]), _int(row["POS_Y"])
        entry = {
            "source_id": row["CITY_ID"],
            "name": texts.get((row["CITY_ID"], "NAME_TXT"), row["CITY_ID"]),
            "kind": kind,
            "position": [x, y],
            "interaction_tile": _interaction_tile(kind, x, y),
            "owner": row["OWNER"],
            "level": 5 if kind == "capital" else _int(row["SIZE"], 1),
            "inside_stack_id": row.get("STACK", ""),
            "garrison": group_entry(row["CITY_ID"]),
        }
        settlements.append(entry)
        settlement_by_id[row["CITY_ID"]] = entry

    stack_rows = sorted(_rows(scens / "Sstack.dbf", scenario_id), key=lambda row: row["STACK_ID"])
    stacks: list[dict[str, Any]] = []
    for row in stack_rows:
        inside = row.get("INSIDE", "")
        x, y = _int(row["POS_X"]), _int(row["POS_Y"])
        if inside in settlement_by_id:
            position = list(settlement_by_id[inside]["interaction_tile"])
        else:
            position = [x, y]
        stacks.append(
            {
                "source_id": row["STACK_ID"],
                "source_position": [x, y],
                "position": position,
                "owner": row["OWNER"],
                "inside": inside,
                "leader_id": row.get("LEADER_ID", ""),
                "group": group_entry(row["STACK_ID"]),
            }
        )

    scenario_items = {
        row["ITEM_ID"]: row["ITEM_TYPE"].lower()
        for row in _rows(scens / "SItem.DBF", scenario_id)
    }

    def item_entry(type_id: str) -> dict[str, Any]:
        global_row = global_items.get(type_id.lower())
        if global_row is None:
            raise ValueError(f"Unknown global item type {type_id}")
        return {
            "type_id": type_id.lower(),
            "name": global_text.get(global_row.get("NAME_TXT", "").lower(), type_id),
            "category": _int(global_row.get("ITEM_CAT")),
            "gold_value": _gold(global_row.get("VALUE", "")),
        }

    inventory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(scens / "SInventr.DBF", scenario_id):
        item_type = scenario_items.get(row["ITEM_ID"])
        if item_type is None:
            raise ValueError(f"Inventory references unknown scenario item {row['ITEM_ID']}")
        inventory[row["INV_ID"]].append(item_entry(item_type))

    chests = [
        {
            "source_id": row["BAG_ID"],
            "position": [_int(row["POS_X"]), _int(row["POS_Y"])],
            "items": inventory.get(row["BAG_ID"], []),
        }
        for row in sorted(_rows(scens / "SBag.dbf", scenario_id), key=lambda row: row["BAG_ID"])
    ]
    resources = [
        {
            "source_id": row["CRYSTAL_ID"],
            "resource_id": _int(row["RESOURCE"]),
            "resource_name": RESOURCE_NAMES[_int(row["RESOURCE"])],
            "position": [_int(row["POS_X"]), _int(row["POS_Y"])],
        }
        for row in sorted(_rows(scens / "SCrys.DBF", scenario_id), key=lambda row: row["CRYSTAL_ID"])
    ]
    ruins = []
    for row in sorted(_rows(scens / "SRuin.DBF", scenario_id), key=lambda row: row["RUIN_ID"]):
        x, y = _int(row["POS_X"]), _int(row["POS_Y"])
        ruins.append(
            {
                "source_id": row["RUIN_ID"],
                "name": texts.get((row["RUIN_ID"], "NAME_TXT"), row["RUIN_ID"]),
                "position": [x, y],
                "interaction_tile": _interaction_tile("ruin", x, y),
                "gold": _gold(row.get("CASH", "")),
                "items": [] if row.get("ITEM") in ("", "G000000000") else [item_entry(row["ITEM"])],
                "garrison": group_entry(row["RUIN_ID"]),
            }
        )

    merchant_stock: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(scens / "SMerch.DBF", scenario_id):
        entry = item_entry(row["ITEM_ID"])
        entry["stock"] = _int(row["ITEM_COUNT"])
        merchant_stock[row["SITE_ID"]].append(entry)
    sites = []
    for row in sorted(_rows(scens / "SSite.DBF", scenario_id), key=lambda row: row["SITE_ID"]):
        x, y = _int(row["POS_X"]), _int(row["POS_Y"])
        site_type = "merchant" if row["SITE_ID"] in merchant_stock else "trainer"
        sites.append(
            {
                "source_id": row["SITE_ID"],
                "name": texts.get((row["SITE_ID"], "NAME_TXT"), row["SITE_ID"]),
                "site_type": site_type,
                "position": [x, y],
                "interaction_tiles": _site_interaction_tiles(x, y),
                "stock": merchant_stock.get(row["SITE_ID"], []),
            }
        )

    terrain = _terrain_grid(_rows(scens / "SMapBlck.DBF", scenario_id), map_size)
    mountains = [
        {
            "source_id": _int(row["ID_MOUNT"]),
            "position": [_int(row["POS_X"]), _int(row["POS_Y"])],
            "size": [_int(row["SIZE_X"]), _int(row["SIZE_Y"])],
            "image": _int(row["IMAGE"]),
            "race": _int(row["RACE"]),
        }
        for row in _rows(scens / "SMounts.DBF", scenario_id)
    ]
    roads = sorted(
        [[_int(row["POS_X"]), _int(row["POS_Y"])] for row in _rows(scens / "SRoad.dbf", scenario_id)],
        key=lambda tile: (tile[1], tile[0]),
    )

    agent_player_id = "S117PL000a" if scenario_id == "S117SC0000" else ""
    enemy_candidates: list[dict[str, Any]] = []
    for stack in stacks:
        if stack["owner"] == agent_player_id or not stack["group"]["units"]:
            continue
        enemy_candidates.append({"kind": "stack", **stack})
    for settlement in settlements:
        if settlement["owner"] == agent_player_id or not settlement["garrison"]["units"]:
            continue
        enemy_candidates.append(
            {
                "kind": "settlement_garrison",
                "source_id": settlement["source_id"],
                "position": settlement["interaction_tile"],
                "inside": settlement["source_id"],
                "owner": settlement["owner"],
                "group": settlement["garrison"],
            }
        )
    for ruin in ruins:
        if not ruin["garrison"]["units"]:
            continue
        enemy_candidates.append(
            {
                "kind": "ruin_guardian",
                "source_id": ruin["source_id"],
                "position": ruin["interaction_tile"],
                "inside": ruin["source_id"],
                "owner": "neutral",
                "group": ruin["garrison"],
            }
        )
    enemy_candidates.sort(key=lambda row: (row["kind"], row["source_id"]))
    enemies = []
    for enemy_id, candidate in enumerate(enemy_candidates, start=1):
        group = candidate["group"]
        enemies.append(
            {
                "enemy_id": enemy_id,
                "source_id": candidate["source_id"],
                "kind": candidate["kind"],
                "position": candidate["position"],
                "inside": candidate.get("inside", ""),
                "owner": candidate.get("owner", ""),
                "description": f"{candidate['kind']} {candidate['source_id']}: "
                + ", ".join(unit["unit_name"] for unit in group["units"]),
                "front": group["front"],
                "back": group["back"],
            }
        )

    snapshot = {
        "format_version": FORMAT_VERSION,
        "scenario": {
            "source_id": scenario_id,
            "name": info["NAME"],
            "briefing": info["BRIEFING"],
            "map_size": map_size,
            "campaign_id": info["CAMPAIGN"],
            "excluded_event_tables": list(EVENT_TABLES),
        },
        "terrain": {
            "tile_codes": terrain,
            "water_tiles": [[x, y] for y in range(map_size) for x in range(map_size) if terrain[y][x] == 29],
            "forest_tiles": [[x, y] for y in range(map_size) for x in range(map_size) if tile_ground_id(terrain[y][x]) == 1],
            "mountains": mountains,
            "roads": roads,
        },
        "players": player_rows,
        "settlements": settlements,
        "stacks": stacks,
        "enemies": enemies,
        "chests": chests,
        "resources": resources,
        "ruins": ruins,
        "sites": sites,
    }
    hashes_after = {str(path.relative_to(game_root)): _sha256(path) for path in source_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("Disciples II source DBFs changed during read-only conversion")
    report = {
        "scenario_id": scenario_id,
        "source_hashes": hashes_before,
        "counts": {
            "map_cells": map_size * map_size,
            "map_blocks": len(_rows(scens / "SMapBlck.DBF", scenario_id)),
            "roads": len(roads),
            "mountain_entries": len(mountains),
            "capitals": sum(row["kind"] == "capital" for row in settlements),
            "settlements": sum(row["kind"] == "settlement" for row in settlements),
            "source_stacks": len(stacks),
            "runtime_enemies": len(enemies),
            "chests": len(chests),
            "resources": len(resources),
            "ruins": len(ruins),
            "merchants": sum(site["site_type"] == "merchant" for site in sites),
            "trainers": sum(site["site_type"] == "trainer" for site in sites),
            "water_tiles": len(snapshot["terrain"]["water_tiles"]),
            "forest_tiles": len(snapshot["terrain"]["forest_tiles"]),
        },
        "event_tables_read": [],
    }
    return snapshot, report


def convert_campaign(
    game_root: Path,
    scenario_id: str,
    output: Path,
    report_output: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _assert_output_is_safe(game_root, output)
    if report_output is not None:
        _assert_output_is_safe(game_root, report_output)
    snapshot, report = build_campaign_snapshot(game_root, scenario_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report_output is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot, report


__all__ = [
    "EVENT_TABLES",
    "FORMAT_VERSION",
    "UNIT_TYPE_TO_DATA_NAME",
    "build_campaign_snapshot",
    "convert_campaign",
]
