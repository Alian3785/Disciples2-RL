"""Static conversion of Disciples II scenario S117SC0000 (The Charmed Child)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from maps.base import MapConfig


SCENARIO_ID = "S117SC0000"
GRID_SIZE = 48
AGENT_CAPITAL_SOURCE_ID = "S117FT0001"
AGENT_CAPITAL_ANCHOR = (36, 4)
HERO_START = (40, 6)
BOT_CAPITAL_SOURCE_ID = "S117FT0000"
BOT_CAPITAL_ANCHOR = (9, 31)
BOT_HOME = (13, 33)
TARGET_CITY_SOURCE_IDS = (
    "S117FT0004",
    "S117FT0005",
    "S117FT0006",
)

_SNAPSHOT_PATH = Path(__file__).with_name("data") / "wotans_retribution.json"
SOURCE_SNAPSHOT = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _tile(raw: Iterable[int]) -> tuple[int, int]:
    x, y = raw
    return int(x), int(y)


SETTLEMENTS_BY_ID = {
    str(row["source_id"]): row for row in SOURCE_SNAPSHOT["settlements"]
}


def _enemy_ids_inside(source_id: str) -> tuple[int, ...]:
    return tuple(
        int(row["enemy_id"])
        for row in SOURCE_SNAPSHOT["enemies"]
        if str(row.get("inside", "")) == source_id
    )


TARGET_CITY_ENEMY_IDS = {
    source_id: _enemy_ids_inside(source_id) for source_id in TARGET_CITY_SOURCE_IDS
}
BOT_CAPITAL_DEFENDER_IDS = _enemy_ids_inside(BOT_CAPITAL_SOURCE_ID)
PROTECTED_ENEMY_IDS = tuple(
    int(row["enemy_id"])
    for row in SOURCE_SNAPSHOT["enemies"]
    if row["kind"] in {"settlement_garrison", "ruin_guardian"}
    or str(row.get("inside", "")).startswith("S117FT")
    or str(row.get("inside", "")).startswith("S117RU")
)


def _object_footprint_blocks() -> tuple[tuple[int, int, int, int], ...]:
    """Turn the non-walkable artwork footprint of every static site into cells."""
    blocks: list[tuple[int, int, int, int]] = []
    for settlement in SOURCE_SNAPSHOT["settlements"]:
        x, y = _tile(settlement["position"])
        size = 5 if settlement["kind"] == "capital" else 4
        interaction = _tile(settlement["interaction_tile"])
        for px in range(x, x + size):
            for py in range(y, y + size):
                if (px, py) != interaction:
                    blocks.append((px, py, 1, 1))
    for ruin in SOURCE_SNAPSHOT["ruins"]:
        x, y = _tile(ruin["position"])
        interaction = _tile(ruin["interaction_tile"])
        for px in range(x, x + 3):
            for py in range(y, y + 3):
                if (px, py) != interaction:
                    blocks.append((px, py, 1, 1))
    for site in SOURCE_SNAPSHOT["sites"]:
        x, y = _tile(site["position"])
        for px in range(x, x + 3):
            for py in range(y, y + 3):
                blocks.append((px, py, 1, 1))
    return tuple(dict.fromkeys(blocks))


MOUNTAIN_BLOCKS = tuple(
    (
        int(row["position"][0]),
        int(row["position"][1]),
        int(row["size"][0]),
        int(row["size"][1]),
    )
    for row in SOURCE_SNAPSHOT["terrain"]["mountains"]
)
STATIC_OBJECT_BLOCKS = _object_footprint_blocks()
OBSTACLE_BLOCKS = MOUNTAIN_BLOCKS + STATIC_OBJECT_BLOCKS

WATER_TILES = tuple(_tile(row) for row in SOURCE_SNAPSHOT["terrain"]["water_tiles"])
FOREST_TILES = tuple(_tile(row) for row in SOURCE_SNAPSHOT["terrain"]["forest_tiles"])
ROAD_TILES = tuple(_tile(row) for row in SOURCE_SNAPSHOT["terrain"]["roads"])
GOLD_MINE_TILES = tuple(
    _tile(row["position"])
    for row in SOURCE_SNAPSHOT["resources"]
    if row["resource_name"] == "gold"
)


def _water_tiles() -> tuple[tuple[int, int], ...]:
    return WATER_TILES


def _forest_tiles() -> tuple[tuple[int, int], ...]:
    return FOREST_TILES


def _road_tiles() -> tuple[tuple[int, int], ...]:
    return ROAD_TILES


def _gold_mine_tiles() -> tuple[tuple[int, int], ...]:
    return GOLD_MINE_TILES


def _territory_forbidden_tiles() -> tuple[tuple[int, int], ...]:
    tiles = set(WATER_TILES)
    for x, y, width, height in OBSTACLE_BLOCKS:
        tiles.update(
            (px, py)
            for px in range(x, x + width)
            for py in range(y, y + height)
        )
    return tuple(sorted(tiles))


ENEMY_STACKS = tuple(
    {
        "enemy_id": int(row["enemy_id"]),
        "position": _tile(row["position"]),
        "description": str(row["description"]),
        "front": list(row["front"]),
        "back": list(row["back"]),
    }
    for row in SOURCE_SNAPSHOT["enemies"]
)

CHESTS = tuple(
    (
        _tile(row["position"]),
        tuple(str(item["name"]) for item in row["items"]),
    )
    for row in SOURCE_SNAPSHOT["chests"]
)

MANA_SOURCES = tuple(
    (str(row["resource_name"]), _tile(row["position"]))
    for row in SOURCE_SNAPSHOT["resources"]
    if row["resource_name"] != "gold"
)

VILLAGE_HEAL_TILES = tuple(
    _tile(row["interaction_tile"])
    for row in SOURCE_SNAPSHOT["settlements"]
    if row["kind"] == "settlement"
)
SETTLEMENT_LEVEL_BY_HEAL_TILE = {
    _tile(row["interaction_tile"]): int(row["level"])
    for row in SOURCE_SNAPSHOT["settlements"]
    if row["kind"] == "settlement"
}
SETTLEMENT_TERRITORY_DATA = tuple(
    {
        "name": str(row["name"]),
        "source_tile": _tile(row["interaction_tile"]),
        "required_enemy_ids": _enemy_ids_inside(str(row["source_id"])),
        "settlement_level": int(row["level"]),
    }
    for row in SOURCE_SNAPSHOT["settlements"]
    if row["kind"] == "settlement"
)
SETTLEMENT_DEFENDER_HEAL_TILES = {
    enemy_id: _tile(row["interaction_tile"])
    for row in SOURCE_SNAPSHOT["settlements"]
    if row["kind"] == "settlement"
    for enemy_id in _enemy_ids_inside(str(row["source_id"]))
}

FINAL_OBJECTIVE_CITIES = {
    str(SETTLEMENTS_BY_ID[source_id]["name"]): TARGET_CITY_ENEMY_IDS[source_id]
    for source_id in TARGET_CITY_SOURCE_IDS
}


def _site_data(site_type: str) -> dict[str, dict[str, object]]:
    return {
        str(row["source_id"]): {
            "anchor": _tile(row["position"]),
            "interaction_tiles": tuple(_tile(tile) for tile in row["interaction_tiles"]),
        }
        for row in SOURCE_SNAPSHOT["sites"]
        if row["site_type"] == site_type
    }


MERCHANT_SITES = _site_data("merchant")
TRAINER_SITES = _site_data("trainer")


def _merchant_grant(item_name: str) -> str:
    if item_name == "Life Potion":
        return "revive_bonus"
    if item_name == "Potion of Healing":
        return "small_heal_bonus"
    if item_name == "Potion of Restoration":
        return "large_heal_bonus"
    return "inventory"


MERCHANT_ITEMS_BY_SITE = {
    str(row["source_id"]): tuple(
        {
            "name": str(item["name"]),
            "price": float(item["gold_value"]),
            "stock": int(item["stock"]),
            "grant": _merchant_grant(str(item["name"])),
        }
        for item in row["stock"]
    )
    for row in SOURCE_SNAPSHOT["sites"]
    if row["site_type"] == "merchant"
}
MERCHANT_ITEMS = tuple(
    item
    for item_name, item in {
        str(item["name"]): dict(item)
        for site_items in MERCHANT_ITEMS_BY_SITE.values()
        for item in site_items
    }.items()
)

RUIN_BY_ID = {str(row["source_id"]): row for row in SOURCE_SNAPSHOT["ruins"]}
RUIN_REWARDS = {
    int(enemy["enemy_id"]): {
        "title": str(ruin["name"]),
        "ruin_pos": _tile(ruin["position"]),
        "marker_pos": _tile(ruin["interaction_tile"]),
        "items": tuple(str(item["name"]) for item in ruin["items"]),
        "gold": int(ruin["gold"]),
    }
    for enemy in SOURCE_SNAPSHOT["enemies"]
    if enemy["kind"] == "ruin_guardian"
    for ruin in (RUIN_BY_ID[str(enemy["source_id"])],)
}


MAP = MapConfig(
    name="wotans_retribution",
    grid_size=GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=OBSTACLE_BLOCKS,
    empty_tiles=(),
    village_heal_tiles=VILLAGE_HEAL_TILES,
    settlement_level_by_heal_tile=SETTLEMENT_LEVEL_BY_HEAL_TILE,
    legions_settlement_territory_data=SETTLEMENT_TERRITORY_DATA,
    settlement_defender_heal_tile_by_enemy_id=SETTLEMENT_DEFENDER_HEAL_TILES,
    chests=CHESTS,
    mana_sources=MANA_SOURCES,
    enemy_stacks=ENEMY_STACKS,
    merchant_sites=MERCHANT_SITES,
    spell_shop_sites={},
    mercenary_sites={},
    trainer_sites=TRAINER_SITES,
    final_objective_cities=FINAL_OBJECTIVE_CITIES,
    ruin_rewards=RUIN_REWARDS,
    default_objective="cities",
    objective_enemy_id=None,
    empire_territory_source_enemy_id=BOT_CAPITAL_DEFENDER_IDS[0],
    empire_territory_source_tile=BOT_HOME,
    scripted_capital_bot_supported=True,
    scripted_capital_bot_home_tile=BOT_HOME,
    scripted_capital_bot_faction="Империя",
    scripted_capital_bot_roster={
        7: "Скваер",
        9: "Скваер",
        10: "Рейнджер людей",
        12: "Служка",
    },
    scripted_capital_bot_home_enemy_ids=BOT_CAPITAL_DEFENDER_IDS,
    scripted_capital_bot_protected_enemy_ids=PROTECTED_ENEMY_IDS,
    supported_objectives=("cities",),
    boss_starting_roster_default=False,
    starting_roster={
        7: "Гном",
        8: "Королевский страж",
        10: "Метатель топоров",
        11: "Травница",
    },
    starting_unit_overrides={8: {"hero": True}},
    starting_gold=100.0,
    merchant_buy_items=MERCHANT_ITEMS,
    merchant_buy_items_by_site=MERCHANT_ITEMS_BY_SITE,
    starting_capital_id=3,
    starting_lord_type=1,
    water_tiles_provider=_water_tiles,
    forest_tiles_provider=_forest_tiles,
    road_tiles_provider=_road_tiles,
    gold_mine_tiles_provider=_gold_mine_tiles,
    territory_forbidden_tiles_provider=_territory_forbidden_tiles,
)
