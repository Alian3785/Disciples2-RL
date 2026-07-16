"""Scroll-training map with one copy of every supported scroll type."""

from __future__ import annotations

from maps.base import MapConfig
from scroll_spell_data import SCROLL_ITEM_DEFINITIONS

GRID_SIZE = 48
PLAY_GRID_SIZE = 24
HERO_START = (4, 24)

OCCULT_MASTER_ENEMY_ID = 1
GREEN_DRAGON_ENEMY_ID = 2
UNDEAD_STACK_ENEMY_ID = 3

OCCULT_MASTER_TILE = (40, 16)
GREEN_DRAGON_TILE = (42, 24)
UNDEAD_STACK_TILE = (40, 32)

STARTING_SCROLL_ITEMS = tuple(
    str(definition.get("item_name", "") or "")
    for definition in SCROLL_ITEM_DEFINITIONS
    if bool(definition.get("supported", False))
    and str(definition.get("item_name", "") or "")
)

if len(STARTING_SCROLL_ITEMS) != 99 or len(set(STARTING_SCROLL_ITEMS)) != 99:
    raise RuntimeError(
        "scroll_train requires exactly 99 unique supported scroll definitions"
    )

MAP = MapConfig(
    name="scroll_train",
    grid_size=GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=(),
    empty_tiles=(),
    village_heal_tiles=(),
    settlement_level_by_heal_tile={},
    legions_settlement_territory_data=(),
    settlement_defender_heal_tile_by_enemy_id={},
    chests=(),
    mana_sources=(),
    enemy_stacks=(
        {
            "enemy_id": OCCULT_MASTER_ENEMY_ID,
            "position": OCCULT_MASTER_TILE,
            "description": "Мастер оккультист",
            "front": [None, "Мастер оккультист", None],
            "back": [None, None, None],
        },
        {
            "enemy_id": GREEN_DRAGON_ENEMY_ID,
            "position": GREEN_DRAGON_TILE,
            "description": "Зелёный дракон",
            "front": [None, "Зелёный дракон", None],
            "back": [None, None, None],
        },
        {
            "enemy_id": UNDEAD_STACK_ENEMY_ID,
            "position": UNDEAD_STACK_TILE,
            "description": (
                "Два Скелета-призрака впереди; Архилич и Сущий позади"
            ),
            "front": ["Скелет призрак", None, "Скелет призрак"],
            "back": ["Архилич", None, "Сущий"],
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
    play_grid_size=PLAY_GRID_SIZE,
    boss_starting_roster_default=False,
    starting_roster={
        7: "Одержимый",
        9: "Одержимый",
        10: "Архидьявол",
        12: "Сектант",
    },
    starting_capital_id=2,
    starting_lord_type=2,
    starting_scroll_items=STARTING_SCROLL_ITEMS,
    turn_penalty=0.03,
    passive_mana_income_enabled=False,
)
