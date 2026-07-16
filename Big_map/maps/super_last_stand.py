"""Super Last Stand: an exploration/economy map ending in eleven invasion waves."""

from __future__ import annotations

from maps.base import MapConfig

GRID_SIZE = 48
HERO_START = (24, 42)
STARTING_GOLD = 3000.0

LEFT_CITY_TILE = (5, 24)
RIGHT_CITY_TILE = (42, 24)
LEFT_CITY_ENEMY_ID = 10
RIGHT_CITY_ENEMY_ID = 11

ORC_ENEMY_ID = 1
OCCULT_MASTER_ENEMY_ID = 2
LAKE_ENEMY_ID = 3
RUIN_ENEMY_ID = 70
ELVEN_RUIN_ENEMY_ID = 71

WAVE_ENEMY_ID_BASE = 100
WAVE_RACE_NAMES = ("Люди", "Эльфы", "Гномы")
WAVE_SPAWN_POSITIONS = ((22, 0), (24, 0), (26, 0))
WAVE_MOVES_PER_TURN = 30


def wave_enemy_id(wave_index: int, race_index: int) -> int:
    return int(WAVE_ENEMY_ID_BASE + (int(wave_index) - 1) * 3 + int(race_index) + 1)


HUMAN_WAVE_ENEMY_ID = wave_enemy_id(1, 0)
ELF_WAVE_ENEMY_ID = wave_enemy_id(1, 1)
DWARF_WAVE_ENEMY_ID = wave_enemy_id(1, 2)

MERCHANT_NAME = "Лавка Последнего рубежа"
SPELL_SHOP_NAME = "Башня дальней магии"

CAPITAL_GOLD_MINE_TILE = (23, 42)
CAPITAL_INFERNAL_MANA_TILE = (25, 42)
LEFT_GOLD_MINE_TILE = (3, 22)
LEFT_DEATH_MANA_TILE = (3, 27)
RIGHT_GOLD_MINE_TILE = (44, 22)
RIGHT_LIFE_MANA_TILE = (44, 27)

CENTER_INVULNERABILITY_CHEST = (24, 24)
LAKE_CHEST_TILE = (36, 7)
LAKE_ENEMY_TILE = (36, 10)
RUIN_TILE = (15, 14)
RUIN_MARKER_TILE = (17, 16)
ELVEN_RUIN_TILE = (29, 14)
ELVEN_RUIN_MARKER_TILE = (30, 16)
ELVEN_RUIN_GOLD = 600
ELVEN_RUIN_BANNER_ITEM = "Banner of War"
ELVEN_RUIN_ORB_ITEM = "Orb of Witches"


def _rectangle_border(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> set[tuple[int, int]]:
    tiles = {
        (x, y_min)
        for x in range(int(x_min), int(x_max) + 1)
    }
    tiles.update(
        (x, y_max)
        for x in range(int(x_min), int(x_max) + 1)
    )
    tiles.update(
        (x_min, y)
        for y in range(int(y_min), int(y_max) + 1)
    )
    tiles.update(
        (x_max, y)
        for y in range(int(y_min), int(y_max) + 1)
    )
    return tiles


LEFT_CITY_WATER = _rectangle_border(1, 18, 8, 30)
RIGHT_CITY_WATER = _rectangle_border(39, 18, 46, 30)
LAKE_WATER = {
    (x, y)
    for x in range(32, 41)
    for y in range(3, 13)
}
WATER_TILES = tuple(sorted(LEFT_CITY_WATER | RIGHT_CITY_WATER | LAKE_WATER))

GOLD_MINE_TILES = (
    CAPITAL_GOLD_MINE_TILE,
    LEFT_GOLD_MINE_TILE,
    RIGHT_GOLD_MINE_TILE,
)


def _water_tiles() -> tuple[tuple[int, int], ...]:
    return WATER_TILES


def _gold_mine_tiles() -> tuple[tuple[int, int], ...]:
    return GOLD_MINE_TILES


def _row_slots(names: tuple[str, ...]) -> list[str | None]:
    if len(names) == 0:
        return [None, None, None]
    if len(names) == 1:
        return [None, names[0], None]
    if len(names) == 2:
        return [names[0], None, names[1]]
    if len(names) == 3:
        return list(names)
    raise ValueError(f"Too many units for one battle row: {names!r}")


def _fit_wave_rows(
    front_names: tuple[str, ...],
    back_names: tuple[str, ...],
) -> tuple[list[str | None], list[str | None]]:
    """Fit the requested army into the engine's 3x2 battle formation.

    Three elf armies list four rear units and only two front units. The engine
    has three cells per row, so the first overflow rear unit occupies the free
    front cell; no requested unit is discarded. Large-unit column normalization
    is applied later by MapConfig.enemy_team_specs().
    """
    front = list(front_names)
    back = list(back_names)
    while len(back) > 3 and len(front) < 3:
        front.append(back.pop(0))
    if len(front) > 3 or len(back) > 3:
        raise ValueError(
            f"Wave formation does not fit the 3x2 battle grid: front={front!r}, back={back!r}"
        )
    return _row_slots(tuple(front)), _row_slots(tuple(back))


# title, approximate strength, then human/elf/dwarf (front, back) formations.
WAVE_DEFINITIONS = (
    (
        "Первая волна",
        "",
        (
            (("Скваер",), ("Лучник", "Служка", "Ученик")),
            (("Малый энт",), ("Рейнджер", "Медиум", "Адепт эльфов")),
            (("Гном",), ("Метатель топоров", "Травница")),
        ),
    ),
    (
        "Регулярные роты",
        "~770",
        (
            (("Рыцарь",), ("Стрелок", "Жрец", "Маг")),
            (("Кентавр копейщик",), ("Охотник", "Оракул", "Дозорный")),
            (("Гном воин",), ("Арбалетчик", "Посвященная", "Метатель топоров")),
        ),
    ),
    (
        "Разведка боем",
        "~1050",
        (
            (
                ("Охотник на ведьм", "Рыцарь", "Вор империи"),
                ("Стрелок", "Маг"),
            ),
            (
                ("Энт", "Вор эльфов"),
                ("Охотник", "Дозорный", "Проводник"),
            ),
            (
                ("Вор гномов", "Гном воин", "Гном"),
                ("Арбалетчик", "Посвященная"),
            ),
        ),
    ),
    (
        "Первая сталь",
        "~1300",
        (
            (
                ("Имперский рыцарь", "Инквизитор"),
                ("Стрелок", "Жрец", "Волшебник"),
            ),
            (
                ("Грифон", "Кентавр странник"),
                ("Охотник", "Архонт", "Оракул"),
            ),
            (
                ("Холмовой гигант", "Ветеран"),
                ("Арбалетчик", "Огнеметчик", "Посвященная"),
            ),
        ),
    ),
    (
        "Гвардия героев",
        "~1650",
        (
            (
                ("Рыцарь на пегасе", "Имперский рыцарь", "Инквизитор"),
                ("Рейнджер людей", "Священник", "Архимаг"),
            ),
            (
                ("Лесной лорд", "Кентавр дикарь"),
                ("Хранитель леса", "Стингер", "Дева рощи", "Теург"),
            ),
            (
                ("Королевский страж", "Скальной гигант"),
                ("Инженер", "Хранитель кузни", "Друид"),
            ),
        ),
    ),
    (
        "Стальной прилив",
        "~1850",
        (
            (
                ("Паладин", "Инквизитор", "Имперский рыцарь"),
                ("Священник", "Волшебник", "Аббатиса"),
            ),
            (
                ("Повелитель небес", "Кентавр дикарь"),
                ("Смотритель", "Часовой", "Теург"),
            ),
            (
                ("Ледяной гигант", "Ветеран"),
                ("Хранитель кузни", "Огнеметчик", "Друид"),
            ),
        ),
    ),
    (
        "Цвет трёх народов",
        "~2100",
        (
            (
                ("Мастер клинка", "Паладин", "Инквизитор"),
                ("Священник", "Белый маг", "Прорицательница"),
            ),
            (
                ("Кентавр дикарь", "Кентавр дикарь"),
                ("Стингер", "Смотритель", "Разбойник эльф", "Сильфида"),
            ),
            (
                ("Сын Имира", "Старый ветеран"),
                ("Хранитель кузни", "Огнеметчик", "Архидруид"),
            ),
        ),
    ),
    (
        "Ярость ветеранов",
        "~2350",
        (
            (
                ("Ангел", "Великий инквизитор", "Паладин"),
                ("Патриарх", "Священник", "Белый маг"),
            ),
            (
                ("Кентавр дикарь", "Кентавр дикарь"),
                ("Мародёр", "Разбойник эльф", "Смотритель", "Часовой"),
            ),
            (
                ("Сын Имира", "Старый ветеран", "Ветеран"),
                ("Хранитель кузни", "Архидруид"),
            ),
        ),
    ),
    (
        "Предвестники бури",
        "~2500",
        (
            (
                ("Защитник Веры", "Ангел", "Великий инквизитор"),
                ("Патриарх", "Прорицательница", "Белый маг"),
            ),
            (
                ("Кентавр дикарь", "Кентавр дикарь", "Кентавр дикарь"),
                ("Мародёр", "Смотритель", "Часовой"),
            ),
            (
                ("Сын Имира", "Ледяной гигант", "Старый ветеран"),
                ("Хранитель кузни",),
            ),
        ),
    ),
    (
        "Канун бури",
        "~2650",
        (
            (
                ("Защитник Веры", "Ангел", "Ангел"),
                ("Патриарх", "Патриарх", "Белый маг"),
            ),
            (
                ("Кентавр дикарь", "Кентавр дикарь", "Кентавр дикарь"),
                ("Мародёр", "Мародёр", "Мародёр"),
            ),
            (
                ("Сын Имира", "Ледяной гигант", "Король гномов"),
                ("Владыка рун",),
            ),
        ),
    ),
    (
        "Судный час",
        "",
        (
            (
                ("Защитник Веры", "Мастер клинка", "Ангел"),
                ("Сэр Аллемон", "Патриарх", "Белый маг"),
            ),
            (
                ("Кентавр дикарь", "Кентавр дикарь", "Кентавр дикарь"),
                ("Иллюмиэлль", "Мародёр", "Сильфида"),
            ),
            (
                ("Сын Имира", "Король гномов", "Владыка рун"),
                ("Маг хугин", "Архидруид"),
            ),
        ),
    ),
)


def _build_wave_enemy_stacks() -> tuple[dict[str, object], ...]:
    stacks: list[dict[str, object]] = []
    for wave_index, (title, strength, armies) in enumerate(
        WAVE_DEFINITIONS,
        start=1,
    ):
        for race_index, (front_names, back_names) in enumerate(armies):
            front, back = _fit_wave_rows(front_names, back_names)
            strength_suffix = f" ({strength})" if strength else ""
            stacks.append(
                {
                    "enemy_id": wave_enemy_id(wave_index, race_index),
                    "position": WAVE_SPAWN_POSITIONS[race_index],
                    "description": (
                        f"Волна {wave_index} — «{title}»{strength_suffix}; "
                        f"{WAVE_RACE_NAMES[race_index]}"
                    ),
                    "front": front,
                    "back": back,
                }
            )
    return tuple(stacks)


WAVE_ENEMY_STACKS = _build_wave_enemy_stacks()
SCHEDULED_WAVES = tuple(
    {
        "enemy_ids": tuple(
            wave_enemy_id(wave_index, race_index)
            for race_index in range(len(WAVE_RACE_NAMES))
        ),
        "spawn_turn": (wave_index - 1) * 15 + 5,
        "spawn_interval": 5,
        "moves_per_turn": WAVE_MOVES_PER_TURN,
        "wave_index": wave_index,
    }
    for wave_index in range(1, len(WAVE_DEFINITIONS) + 1)
)


MERCHANT_ITEMS = (
    {"name": "Angel Orb", "price": 2500.0, "stock": 1, "grant": "inventory"},
    {"name": "Zombie Orb", "price": 800.0, "stock": 1, "grant": "inventory"},
    {
        "name": "Зелье силы (+30%)",
        "price": 450.0,
        "stock": 3,
        "grant": "inventory",
    },
    {
        "name": "Эликсир Жизни",
        "price": 400.0,
        "stock": 5,
        "grant": "revive_bonus",
    },
    {
        "name": "Эликсир восстановления",
        "price": 300.0,
        "stock": 5,
        "grant": "large_heal_bonus",
    },
    {
        "name": "Зелье точности (+30%)",
        "price": 450.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Эликсир быстроты",
        "price": 600.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Эликсир неуязвимости",
        "price": 700.0,
        "stock": 1,
        "grant": "inventory",
    },
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
        "name": "Skull of Thanatos (Artifact)",
        "price": 3750.0,
        "stock": 1,
        "grant": "inventory",
    },
    {
        "name": "Soul Crystal (Artifact)",
        "price": 2000.0,
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
        "name": 'Свиток "Призывание II: Белиарх"',
        "price": 1000.0,
        "stock": 1,
        "grant": "inventory",
    },
)

MERCHANT_SITES = {
    MERCHANT_NAME: {
        "anchor": (28, 39),
        "interaction_tiles": (
            (27, 41),
            (28, 41),
            (29, 41),
            (27, 42),
            (28, 42),
        ),
    },
}

SPELL_SHOP_SITES = {
    SPELL_SHOP_NAME: {
        "anchor": (2, 2),
        "interaction_tiles": (
            (3, 3),
            (4, 3),
            (3, 4),
            (4, 4),
        ),
    },
}

STATIC_ENEMY_STACKS = (
    {
        "enemy_id": ORC_ENEMY_ID,
        "position": (15, 35),
        "description": "Один Орк",
        "front": [None, "Орк", None],
        "back": [None, None, None],
    },
    {
        "enemy_id": OCCULT_MASTER_ENEMY_ID,
        "position": (33, 35),
        "description": "Один Мастер оккультист",
        "front": [None, None, None],
        "back": [None, "Мастер оккультист", None],
    },
    {
        "enemy_id": LAKE_ENEMY_ID,
        "position": LAKE_ENEMY_TILE,
        "description": "Морской змей и Кракен охраняют драгоценность в озере",
        "front": ["Морской змей", None, "Кракен"],
        "back": [None, None, None],
    },
    {
        "enemy_id": LEFT_CITY_ENEMY_ID,
        "position": LEFT_CITY_TILE,
        "description": "Гарнизон города тёмных эльфов",
        "front": ["Тёмный эльф мясник", None, "Тёмный эльф мясник"],
        "back": [
            "Тёмный эльф жнец",
            "Тёмный эльф маг",
            "Тёмный эльф призрак",
        ],
    },
    {
        "enemy_id": RIGHT_CITY_ENEMY_ID,
        "position": RIGHT_CITY_TILE,
        "description": "Гарнизон города сильных варваров",
        "front": ["Варвар", "Вождь варваров", "Варвар"],
        "back": [None, "Шаманка", None],
    },
    {
        "enemy_id": RUIN_ENEMY_ID,
        "position": RUIN_MARKER_TILE,
        "description": "Руины: Тролль, Людоед и Гоблин старейшина",
        "front": ["Тролль", None, "Людоед"],
        "back": [None, "Гоблин старейшина", None],
    },
    {
        "enemy_id": ELVEN_RUIN_ENEMY_ID,
        "position": ELVEN_RUIN_MARKER_TILE,
        "description": (
            "Забытые эльфийские руины: нейтральный грифон, "
            "эльфийский оракул, лорд эльфов и эльф-рейнджер"
        ),
        "front": [
            "Нейтральный грифон",
            None,
            "Нейтральный лорд эльфов",
        ],
        "back": [
            None,
            "Нейтральный эльфийский оракул",
            "Нейтральный эльф-рейнджер",
        ],
    },
)

ENEMY_STACKS = STATIC_ENEMY_STACKS + WAVE_ENEMY_STACKS

MAP = MapConfig(
    name="super_last_stand",
    grid_size=GRID_SIZE,
    hero_start=HERO_START,
    obstacle_blocks=(),
    empty_tiles=(),
    village_heal_tiles=(LEFT_CITY_TILE, RIGHT_CITY_TILE),
    settlement_level_by_heal_tile={
        LEFT_CITY_TILE: 2,
        RIGHT_CITY_TILE: 3,
    },
    legions_settlement_territory_data=(
        {
            "name": "Ноктурна",
            "source_tile": LEFT_CITY_TILE,
            "required_enemy_ids": (LEFT_CITY_ENEMY_ID,),
            "settlement_level": 2,
        },
        {
            "name": "Крепость Бурь",
            "source_tile": RIGHT_CITY_TILE,
            "required_enemy_ids": (RIGHT_CITY_ENEMY_ID,),
            "settlement_level": 3,
        },
    ),
    settlement_defender_heal_tile_by_enemy_id={
        LEFT_CITY_ENEMY_ID: LEFT_CITY_TILE,
        RIGHT_CITY_ENEMY_ID: RIGHT_CITY_TILE,
    },
    chests=(
        (CENTER_INVULNERABILITY_CHEST, ("Эликсир неуязвимости",)),
        (LAKE_CHEST_TILE, ("Imperial Crown (Valuable)",)),
    ),
    mana_sources=(
        ("infernal", CAPITAL_INFERNAL_MANA_TILE),
        ("death", LEFT_DEATH_MANA_TILE),
        ("life", RIGHT_LIFE_MANA_TILE),
    ),
    enemy_stacks=ENEMY_STACKS,
    merchant_sites=MERCHANT_SITES,
    spell_shop_sites=SPELL_SHOP_SITES,
    mercenary_sites={},
    trainer_sites={},
    final_objective_cities={},
    ruin_rewards={
        RUIN_ENEMY_ID: {
            "title": "Руины павшего каравана",
            "ruin_pos": RUIN_TILE,
            "marker_pos": RUIN_MARKER_TILE,
            "items": (
                "Runestone (Artifact)",
                "Unholy Chalice (Artifact)",
            ),
            "gold": 900,
        },
        ELVEN_RUIN_ENEMY_ID: {
            "title": "Забытые эльфийские руины",
            "ruin_pos": ELVEN_RUIN_TILE,
            "marker_pos": ELVEN_RUIN_MARKER_TILE,
            "items": (
                ELVEN_RUIN_BANNER_ITEM,
                ELVEN_RUIN_ORB_ITEM,
            ),
            "gold": ELVEN_RUIN_GOLD,
        },
    },
    default_objective="waves",
    objective_enemy_id=None,
    empire_territory_source_enemy_id=None,
    empire_territory_source_tile=None,
    scripted_capital_bot_supported=False,
    boss_starting_roster_default=False,
    starting_roster={11: "Утер"},
    starting_unit_overrides={
        11: {
            "name": "Бетрезен",
            "unit_type": "Mage",
            "hero": True,
            "exp_current": 0,
            "exp_required": 150,
            "next_level_exp": 500,
            "dynamic_level_exp_increment": 500,
        },
    },
    unit_name_aliases={"Бетрезен": "Утер"},
    starting_gold=STARTING_GOLD,
    merchant_buy_items=MERCHANT_ITEMS,
    starting_hero_abilities=("sorcery_lore", "artifact_knowledge"),
    starting_capital_id=2,
    starting_lord_type=2,
    scheduled_enemy_waves=SCHEDULED_WAVES,
    wave_reward_scale=1.0,
    water_tiles_provider=_water_tiles,
    gold_mine_tiles_provider=_gold_mine_tiles,
)
