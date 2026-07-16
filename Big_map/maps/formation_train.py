"""Formation-training duel for an intentionally incomplete Undead Hordes party."""

from __future__ import annotations

from maps.base import MapConfig

FORMATION_ENEMY_ID = 1
FORMATION_ENEMY_TILE = (30, 46)

MAP = MapConfig(
    name="formation_train",
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
            "enemy_id": FORMATION_ENEMY_ID,
            "position": FORMATION_ENEMY_TILE,
            "description": (
                "Ангел, Мастер клинка и Защитник Веры впереди; "
                "Ученик позади"
            ),
            "front": ["Ангел", "Мастер клинка", "Защитник Веры"],
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
        7: "Воин",
        8: "Королева лич",
        11: "Оборотень",
    },
    starting_capital_id=4,
    starting_lord_type=2,
    leadership_step_penalty=0.06,
)
