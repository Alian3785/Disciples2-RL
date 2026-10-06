"""Two scripted assaults on an initially owned, empty level-five city."""
from maps.base import MapConfig, settlement_footprint_blocks

CITY_NAME = "Учебный город"
CITY_TILE = (24, 24)
CAPITAL_TILE = (4, 2)
BOT_SPAWN_TILE = (24, 44)
ASSAULT_ROSTERS = ({7: "Воин", 9: "Воин", 11: "Колдун"}, {8: "Гном"})
ASSAULT_FACTIONS = ("Нежить", "Горные Кланы")

MAP = MapConfig(
    name="city_defence_train",
    grid_size=48,
    hero_start=CAPITAL_TILE,
    obstacle_blocks=settlement_footprint_blocks(
        capitals=(CAPITAL_TILE,), cities=(CITY_TILE,), grid_size=48),
    empty_tiles=(),
    village_heal_tiles=(CITY_TILE,),
    settlement_level_by_heal_tile={CITY_TILE: 5},
    legions_settlement_territory_data=({
        "name": CITY_NAME, "source_tile": CITY_TILE,
        "required_enemy_ids": (), "settlement_level": 5,
    },),
    settlement_defender_heal_tile_by_enemy_id={},
    chests=(), mana_sources=(), enemy_stacks=(),
    merchant_sites={}, spell_shop_sites={}, mercenary_sites={}, trainer_sites={},
    final_objective_cities={}, ruin_rewards={},
    default_objective="city_defence", supported_objectives=("city_defence",),
    objective_enemy_id=None,
    empire_territory_source_enemy_id=None, empire_territory_source_tile=None,
    scripted_capital_bot_supported=True,
    scripted_capital_bot_home_tile=BOT_SPAWN_TILE,
    scripted_capital_bot_faction=ASSAULT_FACTIONS[0],
    scripted_capital_bot_roster=ASSAULT_ROSTERS[0],
    boss_starting_roster_default=False,
    # Enough to recruit a reserve remotely; no fixed faction or hero roster.
    starting_gold=1000.0,
)
