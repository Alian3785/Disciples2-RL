"""Trade-training map with elite shops, mercenaries, trainer and four racial armies."""

from __future__ import annotations

from maps.base import MapConfig

GRID_SIZE = 48
PLAY_GRID_SIZE = 24
HERO_START = (5, 27)
STARTING_GOLD = 9000.0

MERCHANT_NAME = "Лавка Алара"
MERCENARY_CAMP_NAME = "Элитный лагерь наёмников"
TRAINER_NAME = "Тренировочный лагерь"

POWERFUL_MERCHANT_ITEMS = (
    {
        "name": "Bethrezen's Claw (Artifact)",
        "price": 5000.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Кольцо веков (Артефакт)",
        "price": 3750.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Banner of War",
        "price": 5000.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Boots of Seven Leagues",
        "price": 2000.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Tome of Sorcery",
        "price": 1200.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Angel Orb",
        "price": 2500.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Elder Vampire Talisman",
        "price": 4000.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Эликсир Всевышнего",
        "price": 800.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Эликсир неуязвимости",
        "price": 700.0,
        "stock": 2,
        "grant": "inventory",
    },
    {
        "name": "Эликсир быстроты",
        "price": 600.0,
        "stock": 2,
        "grant": "inventory",
    },
    {
        "name": "Elder Vampire Orb",
        "price": 3000.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Venerable Warrior Orb",
        "price": 2500.0,
        "stock": 1,
        "grant": "inventory",
    },
)

ELITE_MERCENARIES = (
    {"slot": 0, "unit_name": "Защитник Веры", "gold": 2500.0, "stock": 1},
    {"slot": 1, "unit_name": "Белый маг", "gold": 2000.0, "stock": 1},
    {"slot": 2, "unit_name": "Король гномов", "gold": 2500.0, "stock": 1},
    {"slot": 3, "unit_name": "Архидруид", "gold": 1800.0, "stock": 1},
    {"slot": 4, "unit_name": "Скелет рыцарь", "gold": 2000.0, "stock": 1},
    {"slot": 5, "unit_name": "Архилич", "gold": 2500.0, "stock": 1},
    {"slot": 6, "unit_name": "Кентавр дикарь", "gold": 1800.0, "stock": 1},
    {"slot": 7, "unit_name": "Смотритель", "gold": 1800.0, "stock": 1},
)

MERCHANT_SITES = {
    MERCHANT_NAME: {
        "anchor": (34, 10),
        "interaction_tiles": (
            (37, 11),
            (37, 12),
            (35, 13),
            (36, 13),
            (37, 13),
        ),
    },
}

MERCENARY_SITES = {
    MERCENARY_CAMP_NAME: {
        "site_id": "TRADE_MERCENARY_CAMP",
        "anchor": (34, 22),
        "interaction_tiles": (
            (37, 23),
            (37, 24),
            (35, 25),
            (36, 25),
            (37, 25),
        ),
        "roster": ELITE_MERCENARIES,
    },
}

TRAINER_SITES = {
    TRAINER_NAME: {
        "site_id": "TRADE_TRAINER_CAMP",
        "anchor": (34, 34),
        "interaction_tiles": (
            (37, 35),
            (37, 36),
            (35, 37),
            (36, 37),
            (37, 37),
        ),
    },
}

ENEMY_STACKS = (
    {
        "enemy_id": 1,
        "position": (45, 6),
        "description": (
            "Имперский рыцарь и Рыцарь на пегасе впереди; "
            "Патриарх и Волшебник позади"
        ),
        "front": ["Имперский рыцарь", "Рыцарь на пегасе", None],
        "back": ["Патриарх", "Волшебник", None],
    },
    {
        "enemy_id": 2,
        "position": (45, 18),
        "description": (
            "Кентавр дикарь и Кентавр латник впереди; "
            "Солнечная танцовщица и Смотритель позади"
        ),
        "front": ["Кентавр дикарь", "Кентавр латник", None],
        "back": ["Солнечная танцовщица", "Смотритель", None],
    },
    {
        "enemy_id": 3,
        "position": (45, 30),
        "description": (
            "Скелет рыцарь и Лорд тьмы впереди; Архилич и Сущий позади"
        ),
        "front": ["Скелет рыцарь", "Лорд тьмы", None],
        "back": ["Архилич", "Сущий", None],
    },
    {
        "enemy_id": 4,
        "position": (45, 42),
        "description": (
            "Король гномов и Старый ветеран впереди; "
            "Архидруид и Огнеметчик позади"
        ),
        "front": ["Король гномов", "Старый ветеран", None],
        "back": ["Архидруид", "Огнеметчик", None],
    },
    {
        "enemy_id": 5,
        "position": (39, 6),
        "description": "Орочья королевская стража: Король орков",
        "front": [None, "Король орков", None],
        "back": [None, None, None],
    },
    {
        "enemy_id": 6,
        "position": (39, 42),
        "description": "Орочий дозор: Орк впереди и Гоблин лучник позади",
        "front": [None, "Орк", None],
        "back": [None, "Гоблин лучник", None],
    },
)

MAP = MapConfig(
    name="trade_train",
    grid_size=GRID_SIZE,
    play_grid_size=PLAY_GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=(),
    empty_tiles=(),
    village_heal_tiles=(),
    settlement_level_by_heal_tile={},
    legions_settlement_territory_data=(),
    settlement_defender_heal_tile_by_enemy_id={},
    chests=(),
    mana_sources=(),
    enemy_stacks=ENEMY_STACKS,
    merchant_sites=MERCHANT_SITES,
    spell_shop_sites={},
    mercenary_sites=MERCENARY_SITES,
    trainer_sites=TRAINER_SITES,
    final_objective_cities={},
    ruin_rewards={},
    default_objective="all_enemies",
    objective_enemy_id=None,
    empire_territory_source_enemy_id=None,
    empire_territory_source_tile=None,
    scripted_capital_bot_supported=False,
    boss_starting_roster_default=False,
    starting_roster={8: "Герцог"},
    starting_gold=STARTING_GOLD,
    merchant_buy_items=POWERFUL_MERCHANT_ITEMS,
    starting_capital_id=2,
    starting_lord_type=1,
)
