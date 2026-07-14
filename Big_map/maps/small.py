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

from maps import default as _default
from maps.base import MapConfig

PLAY_GRID_SIZE = 24

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

ENEMY_STACKS = tuple(
    row
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

# Половина объектов каждого списка (каждый второй элемент).
OBSTACLE_BLOCKS = _default.BASE_STATIC_OBSTACLE_BLOCKS[::2]
CHESTS = _default.BASE_STATIC_CHESTS[::2]
# Источники маны: 8 -> 4, по одному каждого вида.
MANA_SOURCES = (
    ("infernal", (6, 16)),
    ("life", (6, 11)),
    ("death", (7, 41)),
    ("runes", (1, 4)),
)
# Торговцы 2 -> 1, лагеря наёмников 2 -> 1; лавка заклинаний и тренер
# существуют в единственном экземпляре — остаются.
MERCHANT_SITES = {"Лавка Алара": _default.MERCHANT_SITES["Лавка Алара"]}
SPELL_SHOP_SITES = dict(_default.SPELL_SHOP_SITES)
MERCENARY_SITES = {
    "Лагерь северных варваров": _default.MERCENARY_SITES["Лагерь северных варваров"]
}
TRAINER_SITES = dict(_default.TRAINER_SITES)

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
    empty_tiles=_default.BASE_STATIC_EMPTY_TILES,
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
    water_tiles_provider=_default.MAP.water_tiles_provider,
    forest_tiles_provider=_default.MAP.forest_tiles_provider,
    road_tiles_provider=_default.MAP.road_tiles_provider,
    gold_mine_tiles_provider=_default.MAP.gold_mine_tiles_provider,
    territory_forbidden_tiles_provider=_default.MAP.territory_forbidden_tiles_provider,
)
