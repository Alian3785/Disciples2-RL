# maps/orc_duel.py
"""Минимальная тренировочная карта: чистое поле, столица и один Орк.

Старт героя и размер сетки совпадают с картой по умолчанию, но все объекты
(горы, вода/лес/дороги, сундуки, мана, поселения, торговцы, руины) убраны.
Единственный противник — отряд из одного Орка на месте Зелёного дракона;
цель кампании — победить его, награда как за дракона.
"""

from __future__ import annotations

from maps.base import MapConfig

# id 1 свободен от особой семантики: 31 — полная регенерация дракона,
# 22/32-35/67-69 — поселения, 70-74 — руины, 75 — столица Империи.
ORC_ENEMY_ID = 1

MAP = MapConfig(
    name="orc_duel",
    grid_size=48,
    hero_start=(5, 27),
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
            "enemy_id": ORC_ENEMY_ID,
            "position": (30, 46),
            "description": "Отряд: Орк",
            "front": [None, "Орк", None],
            "back": [None, None, None],
        },
    ),
    merchant_sites={},
    spell_shop_sites={},
    mercenary_sites={},
    trainer_sites={},
    final_objective_cities={},
    ruin_rewards={},
    default_objective="orc",
    objective_enemy_id=ORC_ENEMY_ID,
    empire_territory_source_enemy_id=None,
    empire_territory_source_tile=None,
    scripted_capital_bot_supported=False,
    # Terrain-провайдеры по умолчанию пустые: чистая равнина.
)
