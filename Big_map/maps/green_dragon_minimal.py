"""Minimal Legions scenario played on 32x32 with a 48x48 coordinate base."""

from __future__ import annotations

from maps.base import MapConfig

GRID_SIZE = 48
PLAY_GRID_SIZE = 32
HERO_START = (4, 24)

GREEN_DRAGON_ENEMY_ID = 31
GREEN_DRAGON_NAME = "Зелёный дракон"
BLUE_DRAGON_NAME = "Синий дракон"
LEVEL_TWO_CITY_ENEMY_ID = 40
LEVEL_THREE_CITY_ENEMY_ID = 41
RUIN_ENEMY_ID = 42

LEVEL_TWO_CITY_TILE = (30, 4)
LEVEL_THREE_CITY_TILE = (35, 44)
RUIN_TILE = (26, 3)
RUIN_MARKER_TILE = (27, 4)
KRAKEN_TILE = (29, 42)
WILD_CENTAUR_TILE = (21, 23)

LIFE_MANA_TILE = (29, 4)
DEATH_MANA_TILE = (34, 44)
GOLD_MINE_TILE = (35, 43)

MERCHANT_NAME = "Лавка странствующего алхимика"


def _stack(
    enemy_id: int,
    position: tuple[int, int],
    description: str,
    front: list[str | None],
    back: list[str | None],
) -> dict[str, object]:
    return {
        "enemy_id": int(enemy_id),
        "position": tuple(position),
        "description": str(description),
        "front": list(front),
        "back": list(back),
    }


# The roaming armies are placed in broad strength bands.  Within each band the
# exact lane varies so that the map remains open instead of becoming a corridor.
ROAMING_ENEMY_STACKS = (
    _stack(
        1,
        (14, 25),
        "Один Орк",
        [None, "Орк", None],
        [None, None, None],
    ),
    _stack(
        2,
        (8, 23),
        "Два Гоблина и Гоблин лучник",
        ["Гоблин", None, "Гоблин"],
        [None, "Гоблин лучник", None],
    ),
    _stack(
        3,
        (9, 27),
        "Два Ополченца и Лучник",
        ["Ополченец", None, "Ополченец"],
        [None, "Лучник", None],
    ),
    _stack(
        4,
        (11, 29),
        "Копейщик и Служка",
        [None, "Копейщик", None],
        [None, "Служка", None],
    ),
    _stack(
        5,
        (10, 20),
        "Три Скваера",
        ["Скваер", "Скваер", "Скваер"],
        [None, None, None],
    ),
    _stack(
        6,
        (24, 17),
        "Демон и Одержимый",
        [None, "Демон", None],
        ["Одержимый", None, None],
    ),
    _stack(
        7,
        (12, 17),
        "Скелет и Призрак",
        [None, "Скелет", None],
        [None, "Призрак", None],
    ),
    _stack(
        8,
        (15, 15),
        "Два Крестьянина и Инкуб",
        ["Крестьянин", None, "Крестьянин"],
        [None, "Инкуб", None],
    ),
    _stack(
        9,
        (18, 34),
        "Два Одержимых и Колдунья",
        ["Одержимый", None, "Одержимый"],
        [None, "Колдунья", None],
    ),
    _stack(
        10,
        (13, 31),
        "Один Оккультист",
        [None, None, None],
        [None, "Оккультист", None],
    ),
    _stack(
        11,
        KRAKEN_TILE,
        "Кракен в озере",
        [None, "Кракен", None],
        [None, None, None],
    ),
    _stack(
        12,
        (17, 13),
        "Рыцарь смерти и Зомби",
        ["Рыцарь смерти", None, "Зомби"],
        [None, None, None],
    ),
    _stack(
        13,
        (22, 38),
        "Два Орка",
        ["Орк", None, "Орк"],
        [None, None, None],
    ),
    _stack(
        14,
        (31, 13),
        "Два Тёмных эльфа мясника и Призрак",
        ["Тёмный эльф мясник", None, "Тёмный эльф мясник"],
        [None, "Призрак", None],
    ),
    _stack(
        15,
        (26, 32),
        "Два Гнома воина и Алхимик",
        ["Гном воин", None, "Гном воин"],
        [None, "Алхимик", None],
    ),
    _stack(
        16,
        (19, 9),
        "Два Берсеркера и Сектант",
        ["Берсерк", None, "Берсерк"],
        [None, "Сектант", None],
    ),
    _stack(
        17,
        (16, 35),
        "Оборотень и Зомби",
        ["Оборотень", None, "Зомби"],
        [None, None, None],
    ),
    _stack(
        18,
        WILD_CENTAUR_TILE,
        "Дикий кентавр",
        [None, "Кентавр дикарь", None],
        [None, None, None],
    ),
    _stack(
        19,
        (39, 8),
        "Мантикора",
        [None, "Мантикора", None],
        [None, None, None],
    ),
    _stack(
        20,
        (42, 37),
        "Тиамат",
        [None, "Тиамат", None],
        [None, None, None],
    ),
    _stack(
        GREEN_DRAGON_ENEMY_ID,
        (47, 24),
        "Зелёный дракон — цель карты",
        [None, GREEN_DRAGON_NAME, None],
        [None, None, None],
    ),
)

SETTLEMENT_AND_RUIN_STACKS = (
    _stack(
        LEVEL_TWO_CITY_ENEMY_ID,
        LEVEL_TWO_CITY_TILE,
        "Гарнизон города II уровня: Молох и Сектант",
        [None, "Молох", None],
        ["Сектант", None, None],
    ),
    _stack(
        LEVEL_THREE_CITY_ENEMY_ID,
        LEVEL_THREE_CITY_TILE,
        "Гарнизон города III уровня: Имперский рыцарь и Патриарх",
        [None, "Имперский рыцарь", None],
        [None, "Патриарх", None],
    ),
    _stack(
        RUIN_ENEMY_ID,
        RUIN_MARKER_TILE,
        "Руины: Людоед",
        [None, "Людоед", None],
        [None, None, None],
    ),
)

MERCHANT_ITEMS = (
    {
        "name": "Бутыль лечения (+100 HP)",
        "price": 300.0,
        "stock": 7,
        "grant": "large_heal_bonus",
    },
    {
        "name": "Целебная мазь",
        "price": 600.0,
        "stock": 7,
        "grant": "inventory",
    },
    {
        "name": "Зелье воскрешения (+1)",
        "price": 400.0,
        "stock": 7,
        "grant": "revive_bonus",
    },
    {
        "name": "Эликсир неуязвимости",
        "price": 700.0,
        "stock": 2,
        "grant": "inventory",
    },
)

MERCHANT_SITES = {
    MERCHANT_NAME: {
        "anchor": (14, 41),
        "interaction_tiles": (
            (12, 39),
            (14, 39),
            (15, 39),
            (12, 41),
            (15, 41),
        ),
    },
}

WATER_TILES = (
    KRAKEN_TILE,
    (28, 42),
    (30, 42),
    (29, 41),
    (29, 44),
)

# These deliberately non-contiguous base rows become a contiguous 3x3 forest
# around the centaur after the official 48 -> 32 coordinate scaling.
FOREST_TILES = tuple((x, y) for x in (20, 21, 22) for y in (21, 23, 24))


def _water_tiles() -> tuple[tuple[int, int], ...]:
    return WATER_TILES


def _forest_tiles() -> tuple[tuple[int, int], ...]:
    return FOREST_TILES


def _gold_mine_tiles() -> tuple[tuple[int, int], ...]:
    return (GOLD_MINE_TILE,)


MAP = MapConfig(
    name="green_dragon_minimal",
    grid_size=GRID_SIZE,
    play_grid_size=PLAY_GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=(
        (16, 25, 1, 1),
        (23, 27, 1, 1),
        (32, 20, 1, 1),
        (37, 30, 1, 1),
        (44, 20, 1, 1),
    ),
    empty_tiles=(),
    village_heal_tiles=(LEVEL_TWO_CITY_TILE, LEVEL_THREE_CITY_TILE),
    settlement_level_by_heal_tile={
        LEVEL_TWO_CITY_TILE: 2,
        LEVEL_THREE_CITY_TILE: 3,
    },
    legions_settlement_territory_data=(
        {
            "name": "Форпост Пепла",
            "source_tile": LEVEL_TWO_CITY_TILE,
            "required_enemy_ids": (LEVEL_TWO_CITY_ENEMY_ID,),
            "settlement_level": 2,
        },
        {
            "name": "Цитадель Рассвета",
            "source_tile": LEVEL_THREE_CITY_TILE,
            "required_enemy_ids": (LEVEL_THREE_CITY_ENEMY_ID,),
            "settlement_level": 3,
        },
    ),
    settlement_defender_heal_tile_by_enemy_id={
        LEVEL_TWO_CITY_ENEMY_ID: LEVEL_TWO_CITY_TILE,
        LEVEL_THREE_CITY_ENEMY_ID: LEVEL_THREE_CITY_TILE,
    },
    chests=(
        ((7, 34), ("Эликсир инициативы (+10%)",)),
        ((18, 3), ("Зелье точности (+30%)",)),
        ((28, 27), ("Runestone (Artifact)",)),
        ((40, 18), ("Angel Orb",)),
    ),
    mana_sources=(
        ("life", LIFE_MANA_TILE),
        ("death", DEATH_MANA_TILE),
    ),
    enemy_stacks=ROAMING_ENEMY_STACKS + SETTLEMENT_AND_RUIN_STACKS,
    merchant_sites=MERCHANT_SITES,
    spell_shop_sites={},
    mercenary_sites={},
    trainer_sites={},
    final_objective_cities={},
    ruin_rewards={
        RUIN_ENEMY_ID: {
            "title": "Руины старого тракта",
            "ruin_pos": RUIN_TILE,
            "marker_pos": RUIN_MARKER_TILE,
            "item": "Emerald (Valuable)",
            "gold": 0,
        },
    },
    default_objective="dragon",
    objective_enemy_id=GREEN_DRAGON_ENEMY_ID,
    empire_territory_source_enemy_id=None,
    empire_territory_source_tile=None,
    scripted_capital_bot_supported=False,
    boss_starting_roster_default=False,
    starting_roster={
        7: "Одержимый",
        8: "Герцог",
        9: "Одержимый",
        11: "Сектант",
    },
    starting_capital_id=2,
    starting_lord_type=1,
    water_tiles_provider=_water_tiles,
    forest_tiles_provider=_forest_tiles,
    gold_mine_tiles_provider=_gold_mine_tiles,
    merchant_buy_items=MERCHANT_ITEMS,
)
