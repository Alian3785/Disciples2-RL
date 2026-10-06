"""Mirrow match: symmetric corner economies and a shared central Tiamat.

The two capital entrances, neutral regions and terrain are rotated by 180
degrees. The bot capital therefore opens to the west, towards the board,
rather than inheriting the standard east-facing capital entrance.
"""

from __future__ import annotations

from maps.base import MapConfig, capital_footprint_block

GRID_SIZE = 48
DISPLAY_NAME = "Mirrow match"
HERO_START = (5, 5)
BOT_HOME = (42, 42)
TIAMAT_TILE = (24, 24)
TIAMAT_ENEMY_ID = 250
SCRIPTED_BOT_ENEMY_ID = -75


def mirror_tile(tile: tuple[int, int]) -> tuple[int, int]:
    """Rotate one tile by 180 degrees about the board's geometric center."""
    return (GRID_SIZE - 1 - tile[0], GRID_SIZE - 1 - tile[1])


def _mirror_block(block: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = block
    return (GRID_SIZE - x - width, GRID_SIZE - y - height, width, height)


AGENT_CAPITAL_BLOCK = capital_footprint_block(HERO_START, grid_size=GRID_SIZE)
BOT_CAPITAL_BLOCK = _mirror_block(AGENT_CAPITAL_BLOCK)
OBSTACLE_BLOCKS = (AGENT_CAPITAL_BLOCK, BOT_CAPITAL_BLOCK)
AGENT_GOLD_MINE_TILE = (7, 5)
BOT_GOLD_MINE_TILE = mirror_tile(AGENT_GOLD_MINE_TILE)
AGENT_MANA_TILE = (7, 6)
BOT_MANA_TILE = mirror_tile(AGENT_MANA_TILE)
GOLD_MINE_TILES = (AGENT_GOLD_MINE_TILE, BOT_GOLD_MINE_TILE)
MANA_SOURCES = (("infernal", AGENT_MANA_TILE), ("life", BOT_MANA_TILE))

# Small lakes and woods provide landmarks without enclosing an enemy or a
# resource. Mermaids and Krakens stand on dry shores, reachable by every party.
_AGENT_WATER = {(x, y) for x in range(3, 6) for y in range(19, 22)} | {
    (16, 21), (16, 22), (17, 21), (17, 22),
}
_AGENT_FOREST = {(x, y) for x in range(1, 4) for y in range(15, 18)} | {
    (15, 2), (16, 2), (17, 2), (15, 3), (16, 3),
}
# The clear diagonal road reaches both home gates. All encounters, including
# Tiamat, have room to be bypassed; no fight gates the opposing capital.
_ROAD_BASE = {(n, n) for n in range(5, 43)} | {
    (6, 5), (6, 6), (7, 5), (7, 6),
}
WATER_TILES = tuple(sorted(_AGENT_WATER | {mirror_tile(t) for t in _AGENT_WATER}))
FOREST_TILES = tuple(sorted(_AGENT_FOREST | {mirror_tile(t) for t in _AGENT_FOREST}))
ROAD_TILES = tuple(sorted(_ROAD_BASE | {mirror_tile(t) for t in _ROAD_BASE}))


def _water_tiles() -> tuple[tuple[int, int], ...]:
    return WATER_TILES


def _forest_tiles() -> tuple[tuple[int, int], ...]:
    return FOREST_TILES


def _road_tiles() -> tuple[tuple[int, int], ...]:
    return ROAD_TILES


def _gold_mine_tiles() -> tuple[tuple[int, int], ...]:
    return GOLD_MINE_TILES


# Each row is one requested formation, not a count of occupied tactical cells.
# A large creature appears once and its rear cell stays empty. In particular,
# the giant's Alchemist uses the neighboring rear column, never the giant's.
# Position choices are independent of roster order so the fourteen requested
# formations stay readily auditable. Difficulty tiers increase away from home.
BASE_FORMATIONS = (
    ("goblins", (8, 8), 1, ("Гоблин", "Гоблин", "Гоблин"), (None, None, None)),
    ("orc", (14, 3), 2, (None, "Орк", None), (None, None, None)),
    ("orc_archers", (16, 9), 3, (None, "Орк", None), ("Гоблин лучник", None, "Гоблин лучник")),
    ("spearman_acolyte", (13, 7), 2, (None, "Копейщик", None), (None, "Служка", None)),
    ("militia", (5, 10), 1, ("Ополченец", None, "Ополченец"), (None, None, None)),
    ("bandits", (11, 9), 1, ("Разбойник", "Головорез", "Разбойник"), (None, None, None)),
    ("squires", (11, 4), 1, ("Скваер", "Скваер", "Скваер"), (None, None, None)),
    ("occultist", (4, 15), 2, (None, "Оккультист", None), (None, None, None)),
    ("barbarian", (17, 4), 3, (None, "Варвар", None), (None, None, None)),
    ("mermaid", (6, 18), 3, (None, "Русалка", None), (None, None, None)),
    ("kraken", (18, 20), 4, (None, "Кракен", None), (None, None, None)),
    ("giant_alchemist", (20, 7), 4, (None, "Скальной гигант", None), (None, None, "Алхимик")),
    ("angels", (21, 14), 4, ("Ангел", None, "Ангел"), (None, None, None)),
    ("manticore", (22, 21), 5, (None, "Мантикора", None), (None, None, None)),
)
ADDITIONAL_FORMATIONS = (
    ("peasants", (8, 3), 1, ("Крестьянин", None, "Крестьянин"), (None, None, None)),
    ("skeletons", (8, 13), 2, ("Скелет", None, "Скелет"), (None, None, None)),
    ("wolf", (11, 15), 2, (None, "Волк", None), (None, None, None)),
    ("dwarf_crossbowman", (18, 13), 3, (None, "Гном воин", None), (None, "Арбалетчик", None)),
    ("yeti", (13, 20), 4, (None, "Йети", None), (None, None, None)),
)
FORMATIONS = (*BASE_FORMATIONS, *ADDITIONAL_FORMATIONS)


def _region_stacks(*, mirrored: bool) -> tuple[dict[str, object], ...]:
    region = "bot" if mirrored else "agent"
    # Use IDs without legacy city, ruin or dragon semantics (22, 31..35, 67..75).
    id_base = 200 if mirrored else 100
    stacks = []
    for offset, (key, tile, tier, front, back) in enumerate(FORMATIONS, start=1):
        names = [str(name) for name in (*front, *back) if name]
        stacks.append({
            "enemy_id": id_base + offset,
            "position": mirror_tile(tile) if mirrored else tile,
            "description": f"Mirrow match / {region} / {', '.join(names)}",
            "front": list(front),
            "back": list(back),
            "formation_key": key,
            "region": region,
            "difficulty_tier": tier,
        })
    return tuple(stacks)


TIAMAT_STACK = {
    "enemy_id": TIAMAT_ENEMY_ID,
    "position": TIAMAT_TILE,
    "description": "Центр Mirrow match: один Тиамат",
    "front": [None, "Тиамат", None],
    "back": [None, None, None],
    "formation_key": "tiamat",
    "region": "center",
    "difficulty_tier": 6,
}

AGENT_ENEMY_STACKS = _region_stacks(mirrored=False)
BOT_ENEMY_STACKS = _region_stacks(mirrored=True)
ENEMY_STACKS = (*AGENT_ENEMY_STACKS, *BOT_ENEMY_STACKS, TIAMAT_STACK)

MAP = MapConfig(
    name="mirrow_match",
    grid_size=GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=OBSTACLE_BLOCKS,
    # Reserve the west-facing entrance without adding a dummy hostile stack.
    empty_tiles=(BOT_HOME,),
    village_heal_tiles=(),
    settlement_level_by_heal_tile={},
    legions_settlement_territory_data=(),
    settlement_defender_heal_tile_by_enemy_id={},
    chests=(),
    mana_sources=MANA_SOURCES,
    enemy_stacks=ENEMY_STACKS,
    merchant_sites={},
    spell_shop_sites={},
    mercenary_sites={},
    trainer_sites={},
    final_objective_cities={},
    ruin_rewards={},
    default_objective="scripted_bot",
    objective_enemy_id=SCRIPTED_BOT_ENEMY_ID,
    empire_territory_source_enemy_id=None,
    empire_territory_source_tile=BOT_HOME,
    scripted_capital_bot_supported=True,
    scripted_capital_bot_home_tile=BOT_HOME,
    scripted_capital_bot_faction="Империя",
    scripted_capital_bot_roster={
        7: "Скваер", 8: "Рыцарь на пегасе", 9: "Скваер", 11: "Служка",
    },
    supported_objectives=("scripted_bot",),
    boss_starting_roster_default=False,
    # This is a default, not a fixed scenario faction. Explicit Realcapital
    # and lord settings still select the existing ordinary faction templates.
    default_capital_id=2,
    water_tiles_provider=_water_tiles,
    forest_tiles_provider=_forest_tiles,
    road_tiles_provider=_road_tiles,
    gold_mine_tiles_provider=_gold_mine_tiles,
)
