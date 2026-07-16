# maps/hire_train.py
"""Карта обучения найму: один Герцог, один город и один целевой отряд."""

from __future__ import annotations

from maps.base import MapConfig

GRID_SIZE = 48
PLAY_GRID_SIZE = 24
HERO_START = (5, 27)
TRAINING_CITY_TILE = (24, 12)
TARGET_ARMY_TILE = (46, 46)

TARGET_ARMY_ENEMY_ID = 1
TRAINING_CITY_GARRISON_ENEMY_ID = 2
TRAINING_CITY_NAME = "Учебный город"

MAP = MapConfig(
    name="hire_train",
    grid_size=GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=(),
    empty_tiles=(),
    village_heal_tiles=(TRAINING_CITY_TILE,),
    settlement_level_by_heal_tile={TRAINING_CITY_TILE: 1},
    legions_settlement_territory_data=(
        {
            "name": TRAINING_CITY_NAME,
            "source_tile": TRAINING_CITY_TILE,
            "required_enemy_ids": (TRAINING_CITY_GARRISON_ENEMY_ID,),
        },
    ),
    settlement_defender_heal_tile_by_enemy_id={
        TRAINING_CITY_GARRISON_ENEMY_ID: TRAINING_CITY_TILE,
    },
    chests=(),
    mana_sources=(),
    enemy_stacks=(
        {
            "enemy_id": TARGET_ARMY_ENEMY_ID,
            "position": TARGET_ARMY_TILE,
            "description": (
                "Целевой отряд: Орк слева, два Гоблина впереди "
                "и Гоблин лучник сзади"
            ),
            "front": ["Орк", "Гоблин", "Гоблин"],
            "back": [None, "Гоблин лучник", None],
        },
        {
            "enemy_id": TRAINING_CITY_GARRISON_ENEMY_ID,
            "position": TRAINING_CITY_TILE,
            "description": f"Гарнизон {TRAINING_CITY_NAME}: Гоблин лучник",
            "front": [None, None, None],
            "back": [None, "Гоблин лучник", None],
        },
    ),
    merchant_sites={},
    spell_shop_sites={},
    mercenary_sites={},
    trainer_sites={},
    final_objective_cities={},
    ruin_rewards={},
    default_objective="orc",
    objective_enemy_id=TARGET_ARMY_ENEMY_ID,
    empire_territory_source_enemy_id=None,
    empire_territory_source_tile=None,
    scripted_capital_bot_supported=False,
    play_grid_size=PLAY_GRID_SIZE,
    boss_starting_roster_default=False,
    starting_roster={8: "Герцог"},
    starting_gold=300.0,
)
