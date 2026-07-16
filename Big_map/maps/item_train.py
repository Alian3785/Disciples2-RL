"""Focused item-training map with two magic items and three compact enemy stacks."""

from __future__ import annotations

from maps.base import MapConfig

GRID_SIZE = 48
PLAY_GRID_SIZE = 24
HERO_START = (4, 24)

ANGEL_ORB_CHEST = (18, 14)
ELDER_VAMPIRE_TALISMAN_CHEST = (18, 34)

DWARF_STACK_TILE = (40, 20)
ARMOR_STACK_TILE = (42, 24)
CENTAUR_STACK_TILE = (40, 28)

DWARF_STACK_ENEMY_ID = 1
ARMOR_STACK_ENEMY_ID = 2
CENTAUR_STACK_ENEMY_ID = 3

ANGEL_ORB_ITEM = "Angel Orb"
ELDER_VAMPIRE_TALISMAN_ITEM = "Elder Vampire Talisman"

MAP = MapConfig(
    name="item_train",
    grid_size=GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=(),
    empty_tiles=(),
    village_heal_tiles=(),
    settlement_level_by_heal_tile={},
    legions_settlement_territory_data=(),
    settlement_defender_heal_tile_by_enemy_id={},
    chests=(
        (ANGEL_ORB_CHEST, (ANGEL_ORB_ITEM,)),
        (ELDER_VAMPIRE_TALISMAN_CHEST, (ELDER_VAMPIRE_TALISMAN_ITEM,)),
    ),
    mana_sources=(),
    enemy_stacks=(
        {
            "enemy_id": DWARF_STACK_ENEMY_ID,
            "position": DWARF_STACK_TILE,
            "description": "Король гномов впереди и Арбалетчик позади",
            "front": [None, "Король гномов", None],
            "back": [None, "Арбалетчик", None],
        },
        {
            "enemy_id": ARMOR_STACK_ENEMY_ID,
            "position": ARMOR_STACK_TILE,
            "description": "Оживший доспех",
            "front": [None, "Оживший доспех", None],
            "back": [None, None, None],
        },
        {
            "enemy_id": CENTAUR_STACK_ENEMY_ID,
            "position": CENTAUR_STACK_TILE,
            "description": "Два диких кентавра",
            "front": ["Кентавр дикарь", None, "Кентавр дикарь"],
            "back": [None, None, None],
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
    item_pickup_rewards={
        ANGEL_ORB_ITEM: 1.0,
        ELDER_VAMPIRE_TALISMAN_ITEM: 1.0,
    },
    battle_item_equip_reward=1.0,
    battle_item_use_reward=2.0,
    starting_hero_abilities=("sorcery_lore",),
    starting_capital_id=2,
    starting_lord_type=2,
    passive_mana_income_enabled=False,
)
