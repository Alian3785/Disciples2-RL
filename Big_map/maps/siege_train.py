"""Siege scenario with twenty roaming stacks and a turn-25 pursuing army."""

from __future__ import annotations

from maps.base import MapConfig

GRID_SIZE = 48
HERO_START = (5, 27)

SIEGE_ENEMY_ID = 100
SIEGE_SPAWN_TILE = (47, 47)
SIEGE_SPAWN_TURN = 25
SIEGE_MOVES_PER_TURN = 25

STARTING_ELIXIR_ITEMS = (
    "Эликсир неуязвимости",
    "Зелье силы (+30%)",
    "Эликсир энергии",
    "Эликсир быстроты",
    "Эликсир защиты от магии Огня",
    "Эликсир защиты от магии Земли",
    "Эликсир защиты от магии Воды",
    "Эликсир защиты от магии Воздуха",
    "Эликсир Силы титана",
    "Эликсир Всевышнего",
)


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


STATIC_ENEMY_STACKS = (
    _stack(1, (10, 25), "Дозор I: один Гоблин", [None, "Гоблин", None], [None, None, None]),
    _stack(
        2,
        (11, 33),
        "Дозор II: два Гоблина и Гоблин-лучник",
        ["Гоблин", None, "Гоблин"],
        [None, "Гоблин лучник", None],
    ),
    _stack(
        21,
        (12, 22),
        "Бедное ополчение: два Крестьянина; Гоблин лучник позади",
        ["Крестьянин", None, "Крестьянин"],
        [None, "Гоблин лучник", None],
    ),
    _stack(
        22,
        (12, 25),
        "Смешанный дозор: Гоблин и Бес; Гоблин лучник позади",
        ["Гоблин", None, "Бес"],
        [None, "Гоблин лучник", None],
    ),
    _stack(
        23,
        (12, 28),
        "Шайка: два Разбойника; Гоблин лучник позади",
        ["Разбойник", None, "Разбойник"],
        [None, "Гоблин лучник", None],
    ),
    _stack(
        24,
        (12, 31),
        (
            "Наёмный патруль: Разбойник и Ополченец впереди; "
            "Гоблин лучник позади"
        ),
        ["Разбойник", None, "Ополченец"],
        [None, "Гоблин лучник", None],
    ),
    _stack(
        25,
        (8, 34),
        "Ополчение с магом: Крестьянин и Ополченец впереди; Адепт позади",
        ["Крестьянин", None, "Ополченец"],
        [None, "Адепт", None],
    ),
    _stack(
        3,
        (13, 20),
        "Имперское ополчение: два Скваера и Служка",
        ["Скваер", None, "Скваер"],
        [None, "Служка", None],
    ),
    _stack(
        4,
        (14, 36),
        "Мёртвый патруль: два Воина и Адепт",
        ["Воин", None, "Воин"],
        [None, "Адепт", None],
    ),
    _stack(
        5,
        (17, 27),
        "Орочий заслон: два Орка и Гоблин-старейшина",
        ["Орк", None, "Орк"],
        [None, "Гоблин старейшина", None],
    ),
    _stack(
        6,
        (18, 40),
        "Горный дозор: два Гнома-воина и Арбалетчик",
        ["Гном воин", None, "Гном воин"],
        [None, "Арбалетчик", None],
    ),
    _stack(
        7,
        (21, 13),
        "Имперский патруль: два Рыцаря и Маг",
        ["Рыцарь", None, "Рыцарь"],
        [None, "Маг", None],
    ),
    _stack(
        8,
        (23, 34),
        "Чумной отряд: два Зомби, Призрак и Чернокнижник",
        ["Зомби", None, "Зомби"],
        ["Призрак", None, "Чернокнижник"],
    ),
    _stack(
        9,
        (26, 6),
        "Берсерки: два Берсерка, Колдун и Ведьма",
        ["Берсерк", None, "Берсерк"],
        ["Колдун", None, "Ведьма"],
    ),
    _stack(
        10,
        (27, 28),
        "Орден охотников: Имперский рыцарь, Охотник на ведьм и Священник",
        ["Имперский рыцарь", None, "Охотник на ведьм"],
        [None, "Священник", None],
    ),
    _stack(
        11,
        (30, 43),
        "Костяная гвардия: два Скелета-воина, Некромант и Дух",
        ["Скелет воин", None, "Скелет воин"],
        ["Некромант", None, "Дух"],
    ),
    _stack(
        12,
        (33, 15),
        "Гномьи ветераны: два Ветерана, Друид и Алхимик",
        ["Ветеран", None, "Ветеран"],
        ["Друид", None, "Алхимик"],
    ),
    _stack(
        13,
        (35, 32),
        "Тёмный конклав: два Тёмных паладина, Демонолог и Колдунья",
        ["Темный паладин", None, "Темный паладин"],
        ["Демонолог", None, "Колдунья"],
    ),
    _stack(
        14,
        (37, 5),
        "Дикая стража: два Кентавра-дикаря, Архонт и Теург",
        ["Кентавр дикарь", None, "Кентавр дикарь"],
        ["Архонт", None, "Теург"],
    ),
    _stack(
        15,
        (39, 40),
        "Старшая гвардия: Ветеран, Старый ветеран, Архидруид и Отшельник",
        ["Ветеран", None, "Старый ветеран"],
        ["Архидруид", None, "Отшельник"],
    ),
    _stack(
        16,
        (41, 19),
        "Небесный орден: Ангел, Паладин, Белый маг и Аббатиса",
        ["Ангел", None, "Паладин"],
        ["Белый маг", None, "Аббатиса"],
    ),
    _stack(
        17,
        (43, 34),
        "Адская гвардия: два Адских рыцаря, Инкуб и Пандемонеус",
        ["Адский рыцарь", None, "Адский рыцарь"],
        ["Инкуб", None, "Пандемонеус"],
    ),
    _stack(
        18,
        (45, 10),
        "Храмовая элита: Защитник Веры, Мастер клинка, Белый маг и Прорицательница",
        ["Защитник Веры", None, "Мастер клинка"],
        ["Белый маг", None, "Прорицательница"],
    ),
    _stack(
        19,
        (46, 43),
        "Высшая нежить: два Скелета-призрака, Архилич и Сущий",
        ["Скелет призрак", None, "Скелет призрак"],
        ["Архилич", None, "Сущий"],
    ),
    _stack(
        20,
        (47, 2),
        "Последний рубеж: Король гномов, Владыка рун, Защитник Веры, Архидруид, Отшельник и Архилич",
        ["Король гномов", "Владыка рун", "Защитник Веры"],
        ["Архидруид", "Отшельник", "Архилич"],
    ),
)

SIEGE_STACK = _stack(
    SIEGE_ENEMY_ID,
    SIEGE_SPAWN_TILE,
    "Осадная армия Империи",
    ["Имперский рыцарь", "Рыцарь на пегасе", None],
    ["Рейнджер людей", "Белый маг", None],
)

MAP = MapConfig(
    name="siege_train",
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
    enemy_stacks=(*STATIC_ENEMY_STACKS, SIEGE_STACK),
    merchant_sites={},
    spell_shop_sites={},
    mercenary_sites={},
    trainer_sites={},
    final_objective_cities={},
    ruin_rewards={},
    default_objective="target_enemy",
    objective_enemy_id=SIEGE_ENEMY_ID,
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
    starting_hero_items=tuple(
        item_name
        for item_name in STARTING_ELIXIR_ITEMS
        for _ in range(2)
    ),
    starting_capital_id=2,
    starting_lord_type=1,
    enemy_unit_level_overrides={
        SIEGE_ENEMY_ID: {
            "Рыцарь на пегасе": 3,
            "Рейнджер людей": 2,
        },
    },
    scheduled_enemy_id=SIEGE_ENEMY_ID,
    scheduled_enemy_spawn_turn=SIEGE_SPAWN_TURN,
    scheduled_enemy_moves_per_turn=SIEGE_MOVES_PER_TURN,
)
