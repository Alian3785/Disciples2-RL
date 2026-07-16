"""Magic-training map with two enemy stacks sealed behind obstacles."""

from __future__ import annotations

from maps.base import MapConfig

GRID_SIZE = 48
HERO_START = (5, 27)

FIRST_ENEMY_ID = 1
SECOND_ENEMY_ID = 2
FIRST_ENEMY_TILE = (30, 14)
SECOND_ENEMY_TILE = (38, 35)

MANA_SOURCE_TILES = {
    "life": (4, 27),
    "runes": (6, 27),
    "infernal": (5, 26),
    "death": (5, 28),
}


def _enclosure_blocks(position: tuple[int, int]) -> tuple[tuple[int, int, int, int], ...]:
    x, y = position
    return (
        (x - 1, y - 1, 3, 1),
        (x - 1, y + 1, 3, 1),
        (x - 1, y, 1, 1),
        (x + 1, y, 1, 1),
    )


ENEMY_ENCLOSURE_BLOCKS = (
    *_enclosure_blocks(FIRST_ENEMY_TILE),
    *_enclosure_blocks(SECOND_ENEMY_TILE),
)

MAP = MapConfig(
    name="magic_train",
    grid_size=GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=ENEMY_ENCLOSURE_BLOCKS,
    empty_tiles=(),
    village_heal_tiles=(),
    settlement_level_by_heal_tile={},
    legions_settlement_territory_data=(),
    settlement_defender_heal_tile_by_enemy_id={},
    chests=(),
    mana_sources=tuple(
        (mana_kind, position)
        for mana_kind, position in MANA_SOURCE_TILES.items()
    ),
    enemy_stacks=(
        {
            "enemy_id": FIRST_ENEMY_ID,
            "position": FIRST_ENEMY_TILE,
            "description": "Орк и Ополченец впереди; Гоблин лучник позади",
            "front": ["Орк", None, "Ополченец"],
            "back": [None, "Гоблин лучник", None],
        },
        {
            "enemy_id": SECOND_ENEMY_ID,
            "position": SECOND_ENEMY_TILE,
            "description": "Скваер и Разбойник впереди; Ученик позади",
            "front": ["Скваер", None, "Разбойник"],
            "back": [None, "Ученик", None],
        },
    ),
    merchant_sites={},
    spell_shop_sites={},
    mercenary_sites={},
    trainer_sites={},
    final_objective_cities={},
    ruin_rewards={},
    default_objective="all_enemies",
    objective_enemy_id=None,
    empire_territory_source_enemy_id=None,
    empire_territory_source_tile=None,
    scripted_capital_bot_supported=False,
    boss_starting_roster_default=False,
    starting_roster={
        7: "Одержимый",
        9: "Одержимый",
        10: "Архидьявол",
        12: "Сектант",
    },
    starting_gold=200.0,
    starting_capital_id=2,
    starting_lord_type=2,
    enforce_enemy_reachability=False,
)
