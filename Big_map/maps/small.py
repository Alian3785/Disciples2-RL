# maps/small.py
"""Уменьшенная вдвое копия карты по умолчанию: игра на сетке 24x24.

Аналог уменьшения доски Go до 9x9: та же карта, но игровая сетка 24x24 и
примерно вдвое меньше объектов каждого типа (отряды, руины, торговцы, лагеря,
сундуки, источники маны, препятствия, поселения).

Не тронуты (по требованиям задачи):
- оба целевых города — Порт Полонис (34+68) и Соругирилла (35+69);
- вражеская столица Империи (75, Мизраэль) и её скриптовый бот;
- дракон-цель id 31 (синий вариант через objective blue_dragon работает
  автоматически — это подмена юнита, а не отдельный стек);
- стартовый отряд агента: базовый шаблон (Одержимые, Герцог, Сектант),
  боссовый ростер по умолчанию выключен.

Данные хранятся в координатах базовой 48x48 карты (как в maps/default.py);
уменьшение до 24x24 выполняет штатный механизм масштабирования среды
(play_grid_size=24 -> scale_*/resolve_* в grid.py).
"""

from __future__ import annotations

from collections import Counter
from functools import cache

from maps import default as _default
from maps.base import MapConfig, settlement_footprint_blocks

PLAY_GRID_SIZE = 24

_BASE_TILES_PER_PLAY_TILE = (_default.GRID_SIZE - 1) / (PLAY_GRID_SIZE - 1)


def _base_tile(play_tile):
    """Клетка базовой карты, которую среда сжимает ровно в play_tile сетки 24x24."""
    return tuple(round(coord * _BASE_TILES_PER_PLAY_TILE) for coord in play_tile)


def _play_tile(base_tile):
    """Клетка сетки 24x24, в которую среда сжимает клетку базовой карты."""
    return tuple(
        max(0, min(PLAY_GRID_SIZE - 1, round(coord / _BASE_TILES_PER_PLAY_TILE)))
        for coord in base_tile
    )


_BASE_TILE_COUNT_BY_PLAY_TILE = Counter(
    _play_tile((x, y)) for x in range(_default.GRID_SIZE) for y in range(_default.GRID_SIZE)
)


@cache
def _majority_tiles(provider):
    """Оставляет слой местности только на клетках 24x24, которые он покрывает хотя бы наполовину.

    Штатное сжатие отдаёт клетку слою, если тот есть хотя бы на одной из четырёх
    исходных клеток. Из-за этого лес занимал 29% small вместо 11% на default,
    вода 24% вместо 15%, а запрет роста территории 47% вместо 28%.
    """
    base_tiles = tuple(provider())
    covered = Counter(_play_tile(tile) for tile in base_tiles)
    return tuple(
        tile
        for tile in base_tiles
        if 2 * covered[_play_tile(tile)] >= _BASE_TILE_COUNT_BY_PLAY_TILE[_play_tile(tile)]
    )


# Провайдеры объявлены на уровне модуля, чтобы среду можно было сериализовать.
def _water_tiles():
    return _majority_tiles(_default.MAP.water_tiles_provider)


def _forest_tiles():
    return _majority_tiles(_default.MAP.forest_tiles_provider)


def _territory_forbidden_tiles():
    return _majority_tiles(_default.MAP.territory_forbidden_tiles_provider)


def _block_contains(block, tile) -> bool:
    x, y, width, height = block
    return x <= tile[0] < x + width and y <= tile[1] < y + height

# Свободные отряды: каждый второй из соответствующих списков default-карты.
# Стек 55 исключен: при сжатии 48->24 его клетка (41,25) совпадает с клеткой
# целевого города Соругирилла (40,25).
_KEPT_MATCHED_IDS = frozenset(range(1, 31, 2))       # 1, 3, ..., 29
_KEPT_ADDITIONAL_IDS = frozenset(range(37, 66, 2)) - {55}   # 37, 39, ..., 65 без 55
# Неприкасаемые: дракон, столица Империи, оба целевых города с гарнизонами.
_UNTOUCHABLE_IDS = frozenset({31, 75, 34, 35, 68, 69})
# Руины: 5 -> 2 (лёгкая ранняя охрана и тяжёлая поздняя).
_KEPT_RUIN_IDS = frozenset({70, 74})

_KEPT_STACK_IDS = (
    _KEPT_MATCHED_IDS | _KEPT_ADDITIONAL_IDS | _UNTOUCHABLE_IDS | _KEPT_RUIN_IDS
)

# Охрана лавки заклинаний стоит на угловом входе, как на default. Без переноса
# сжатие ставит её внутрь здания лавки.
# Отряд 21 стоит справа от входов Лавки Алара, как на default; без переноса
# сжатие ставит его на вход (12, 17). Отряды 9, 13 и 49 на default стоят вплотную
# к горам, а при сжатии попадали внутрь них; они остаются с той же стороны гор.
_MOVED_STACK_POSITIONS = {
    9: _base_tile((11, 11)),
    13: _base_tile((15, 12)),
    21: _base_tile((13, 17)),
    43: _base_tile((23, 3)),
    49: _base_tile((8, 6)),
}

ENEMY_STACKS = tuple(
    dict(row, position=_MOVED_STACK_POSITIONS.get(int(row["enemy_id"]), row["position"]))
    for row in (_default.MATCHED_MAP_STACKS + _default.ALL_MAP_STACK_OVERRIDES)
    if int(row["enemy_id"]) in _KEPT_STACK_IDS
)

# Поселения: только два целевых города (остальные деревни убраны).
_KEPT_SETTLEMENT_NAMES = ("Порт Полонис", "Соругирилла")
VILLAGE_HEAL_TILES = ((28, 33), (40, 25))
SETTLEMENT_LEVEL_BY_HEAL_TILE = {
    tile: _default.BASE_SETTLEMENT_LEVEL_BY_HEAL_TILE[tile]
    for tile in VILLAGE_HEAL_TILES
}
LEGIONS_SETTLEMENT_TERRITORY_DATA = tuple(
    entry
    for entry in _default.BASE_LEGIONS_SETTLEMENT_TERRITORY_DATA
    if str(entry.get("name", "")) in _KEPT_SETTLEMENT_NAMES
)
SETTLEMENT_DEFENDER_HEAL_TILE_BY_ENEMY_ID = {
    enemy_id: tile
    for enemy_id, tile in _default.SETTLEMENT_DEFENDER_HEAL_TILE_BY_ENEMY_ID.items()
    if enemy_id in (34, 35, 68, 69)
}

# Торговцы 2 -> 1, лагеря наёмников 2 -> 1; лавка заклинаний и тренер
# существуют в единственном экземпляре — остаются.
MERCHANT_SITES = {"Лавка Алара": _default.MERCHANT_SITES["Лавка Алара"]}
SPELL_SHOP_SITES = dict(_default.SPELL_SHOP_SITES)
MERCENARY_SITES = {
    "Лагерь северных варваров": _default.MERCENARY_SITES["Лагерь северных варваров"]
}
TRAINER_SITES = dict(_default.TRAINER_SITES)

# Здания default: столицы, деревни, руины и объекты на карте. Прореживается
# (каждый второй блок) только рельеф. Здание остаётся, только если small
# сохранила его объект, поэтому у убранных деревень и руин нет пустых зданий.
_DEFAULT_STACK_POSITIONS = {
    int(row["enemy_id"]): tuple(row["position"])
    for row in (_default.MATCHED_MAP_STACKS + _default.ALL_MAP_STACK_OVERRIDES)
}
_DEFAULT_SITES = (
    _default.MERCHANT_SITES, _default.SPELL_SHOP_SITES,
    _default.MERCENARY_SITES, _default.TRAINER_SITES,
)
_SMALL_SITES = (MERCHANT_SITES, SPELL_SHOP_SITES, MERCENARY_SITES, TRAINER_SITES)
_CAPITAL_TILES = (_default.HERO_START, _default.MAP.empire_territory_source_tile)
_DEFAULT_BUILDING_TILES = (
    *_CAPITAL_TILES,
    *_default.BASE_VILLAGE_HEAL_TILES,
    *(_DEFAULT_STACK_POSITIONS[enemy_id] for enemy_id in _default.RUIN_REWARDS),
    *(site["anchor"] for sites in _DEFAULT_SITES for site in sites.values()),
)
_KEPT_BUILDING_TILES = (
    *_CAPITAL_TILES,
    *VILLAGE_HEAL_TILES,
    *(_DEFAULT_STACK_POSITIONS[enemy_id] for enemy_id in sorted(_KEPT_RUIN_IDS)),
    *(site["anchor"] for sites in _SMALL_SITES for site in sites.values()),
)
_TERRAIN_BLOCKS = tuple(
    block
    for block in _default.BASE_STATIC_OBSTACLE_BLOCKS
    if not any(_block_contains(block, tile) for tile in _DEFAULT_BUILDING_TILES)
)
_BUILDING_BLOCKS = tuple(
    block
    for block in _default.BASE_STATIC_OBSTACLE_BLOCKS
    if any(_block_contains(block, tile) for tile in _KEPT_BUILDING_TILES)
)
# Габариты столиц и оставленных городов возвращаются явно (блоки, уже
# попавшие в список зданий, не дублируются).
_SETTLEMENT_BLOCKS = settlement_footprint_blocks(
    capitals=_CAPITAL_TILES,
    cities=VILLAGE_HEAL_TILES,
)
OBSTACLE_BLOCKS = (
    *_TERRAIN_BLOCKS[::2],
    *_BUILDING_BLOCKS,
    *(block for block in _SETTLEMENT_BLOCKS if block not in _BUILDING_BLOCKS),
)
# Пустые клетки default, попавшие при сжатии внутрь гор или зданий (например,
# клетка охраны лавки заклинаний), иначе оставляли в них проходимые дыры.
_OBSTACLE_PLAY_TILES = frozenset(
    _play_tile((x + dx, y + dy))
    for x, y, width, height in OBSTACLE_BLOCKS
    for dx in range(width)
    for dy in range(height)
)
EMPTY_TILES = tuple(
    tile
    for tile in _default.BASE_STATIC_EMPTY_TILES
    if _play_tile(tile) not in _OBSTACLE_PLAY_TILES
)

# При сжатии Banner of War попадал на шахту (14, 23), а Orb of Life на угловой
# вход лавки заклинаний (23, 3). Оба сундука лежат на соседних свободных клетках.
_MOVED_CHESTS = {
    (28, 46): _base_tile((13, 23)),
    (46, 6): _base_tile((23, 4)),
}
CHESTS = tuple(
    (_MOVED_CHESTS.get(position, position), items)
    for position, items in _default.BASE_STATIC_CHESTS[::2]
)
# Источники маны: 8 -> 4, по одному каждого вида.
MANA_SOURCES = (
    ("infernal", (6, 16)),
    ("life", (6, 11)),
    ("death", (7, 41)),
    ("runes", (1, 4)),
)
RUIN_REWARDS = {
    enemy_id: reward
    for enemy_id, reward in _default.RUIN_REWARDS.items()
    if enemy_id in _KEPT_RUIN_IDS
}

MAP = MapConfig(
    name="small",
    grid_size=_default.GRID_SIZE,          # координатная база данных: 48
    hero_start=_default.HERO_START,        # (5, 27) -> (2, 13) на сетке 24
    obstacle_blocks=OBSTACLE_BLOCKS,
    empty_tiles=EMPTY_TILES,
    village_heal_tiles=VILLAGE_HEAL_TILES,
    settlement_level_by_heal_tile=SETTLEMENT_LEVEL_BY_HEAL_TILE,
    legions_settlement_territory_data=LEGIONS_SETTLEMENT_TERRITORY_DATA,
    settlement_defender_heal_tile_by_enemy_id=SETTLEMENT_DEFENDER_HEAL_TILE_BY_ENEMY_ID,
    chests=CHESTS,
    mana_sources=MANA_SOURCES,
    enemy_stacks=ENEMY_STACKS,
    merchant_sites=MERCHANT_SITES,
    spell_shop_sites=SPELL_SHOP_SITES,
    mercenary_sites=MERCENARY_SITES,
    trainer_sites=TRAINER_SITES,
    final_objective_cities=dict(_default.FINAL_OBJECTIVE_CITIES),
    ruin_rewards=RUIN_REWARDS,
    default_objective="cities",
    objective_enemy_id=31,
    empire_territory_source_enemy_id=75,
    empire_territory_source_tile=(26, 10),
    scripted_capital_bot_supported=True,
    play_grid_size=PLAY_GRID_SIZE,
    boss_starting_roster_default=False,    # Одержимые, Герцог, Сектант
    water_tiles_provider=_water_tiles,
    forest_tiles_provider=_forest_tiles,
    # Дороги узкие: по большинству они бы рвались, поэтому сжимаются штатно.
    road_tiles_provider=_default.MAP.road_tiles_provider,
    gold_mine_tiles_provider=_default.MAP.gold_mine_tiles_provider,
    territory_forbidden_tiles_provider=_territory_forbidden_tiles,
)
