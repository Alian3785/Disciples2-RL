# maps/builder.py
"""Тренировочная карта строительства: чистое поле, столица и один Орк-статист.

Раскладка повторяет orc_duel (тот же старт героя, размер сетки, пустая карта),
но цель кампании другая: построить все доступные здания столицы. За каждое
построенное здание начисляется промежуточная награда, а когда строить больше
нечего (все оставшиеся здания построены или заблокированы взаимоисключающими
ветками), эпизод завершается победой с наградой как за дракона.

Орк на карте не является целью: его победа не завершает эпизод. Он оставлен,
чтобы карта не была полностью пустой от врагов (парность с orc_duel и
стабильность подсистем, ожидающих хотя бы один вражеский отряд).
"""

from __future__ import annotations

from maps.base import MapConfig

# id 1 свободен от особой семантики: 31 — полная регенерация дракона,
# 22/32-35/67-69 — поселения, 70-74 — руины, 75 — столица Империи.
BUILDER_BYSTANDER_ENEMY_ID = 1

MAP = MapConfig(
    name="builder",
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
            "enemy_id": BUILDER_BYSTANDER_ENEMY_ID,
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
    default_objective="build_all",
    objective_enemy_id=None,
    empire_territory_source_enemy_id=None,
    empire_territory_source_tile=None,
    scripted_capital_bot_supported=False,
    # Terrain-провайдеры по умолчанию пустые: чистая равнина.
)
