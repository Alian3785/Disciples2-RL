"""
Centralized grid/campaign geometry, rules and backward-compat facade.

Содержимое карты (препятствия, сундуки, мана, поселения, вражеские стеки,
точки взаимодействия) живет в пакете maps/ — этот модуль реэкспортирует
данные карты по умолчанию (maps/default.py) под историческими именами,
чтобы существующие импортеры работали без правок, и оставляет у себя:
- геометрию и resolve_*-функции (масштабирование, препятствия, heal tiles)
- шаблон отряда синего героя
- игровые правила (бонусы поселений/столицы, спеки отображения маны)
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Dict, Mapping, Optional, Tuple

from data_dicts_compact_lines import is_hero_name

# --- Реэкспорт хелперов сборки стеков (канонические копии в maps/base.py) ---
from maps.base import (
    BIG_UNIT_NAMES,
    _canonical_enemy_name,
    _is_big_enemy_unit,
    _normalize_big_unit_rows,
    _normalize_enemy_team_spec,
    _pack_stack_units,
    make_stack_override as _make_stack_override,
)

# --- Реэкспорт данных карты по умолчанию (канонические копии в maps/default.py) ---
from maps.default import (
    GRID_SIZE as DEFAULT_GRID_SIZE,
    HERO_START as DEFAULT_HERO_GRID_POSITION,
    ADDITIONAL_MAP_STACK_OVERRIDES,
    ALL_MAP_STACK_OVERRIDES,
    BASE_ENEMY_POSITIONS,
    BASE_LEGIONS_SETTLEMENT_TERRITORY_DATA,
    BASE_SETTLEMENT_LEVEL_BY_HEAL_TILE,
    BASE_STATIC_CHESTS,
    BASE_STATIC_EMPTY_TILES,
    BASE_STATIC_HEAL_TILES,
    BASE_STATIC_MANA_SOURCES,
    BASE_STATIC_OBSTACLE_BLOCKS,
    BASE_VILLAGE_HEAL_TILES,
    BASE_VILLAGE_PREVIOUS_CORNER_TILES,
    CAPITAL_INTERNAL_STACK_OVERRIDES,
    CHEST_COUNT,
    ENEMY_TEAM_SPECS,
    HEAL_TILE_COUNT,
    MANA_SOURCE_COUNT,
    MATCHED_MAP_STACKS,
    RUIN_STACK_OVERRIDES,
    SETTLEMENT_DEFENDER_HEAL_TILE_BY_ENEMY_ID,
    SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID,
    VILLAGE_INTERNAL_GARRISON_OVERRIDES,
    VILLAGE_LINKED_STACK_OVERRIDES,
)

LEGIONS_SETTLEMENT_TERRITORY_EXPANSION_BY_LEVEL: Dict[int, int] = {
    1: 15,
    2: 15,
    3: 20,
    4: 20,
    5: 25,
}

SETTLEMENT_ARMOR_BONUS_BY_LEVEL: Dict[int, int] = {
    1: 10,
    2: 15,
    3: 20,
    4: 25,
    5: 30,
}
CAPITAL_HEAL_TILE_ARMOR_BONUS = 40
CAPITAL_HEAL_TILE_REST_BONUS = 50

MANA_SOURCE_DISPLAY_SPECS: Dict[str, Dict[str, object]] = {
    "infernal": {
        "name": "Мана преисподней",
        "letter": "П",
        "fill_color": (0.84, 0.18, 0.16, 0.84),
        "edge_color": (0.46, 0.08, 0.06, 0.98),
        "text_color": (1.0, 0.96, 0.96, 1.0),
        "ansi_color": "\x1b[31m",
    },
    "life": {
        "name": "Мана жизни",
        "letter": "Ж",
        "fill_color": (0.18, 0.38, 0.88, 0.84),
        "edge_color": (0.06, 0.18, 0.46, 0.98),
        "text_color": (1.0, 0.98, 0.98, 1.0),
        "ansi_color": "\x1b[34m",
    },
    "death": {
        "name": "Мана смерти",
        "letter": "С",
        "fill_color": (0.72, 0.72, 0.72, 0.88),
        "edge_color": (0.36, 0.36, 0.36, 0.98),
        "text_color": (0.10, 0.10, 0.10, 1.0),
        "ansi_color": "\x1b[90m",
    },
    "runes": {
        "name": "Мана рун",
        "letter": "Р",
        "fill_color": (0.40, 0.84, 0.96, 0.84),
        "edge_color": (0.08, 0.44, 0.60, 0.98),
        "text_color": (0.06, 0.18, 0.24, 1.0),
        "ansi_color": "\x1b[96m",
    },
}

# Base campaign layout mirrors the 48x48 scenario coordinates directly.
BASE_GRID_SIZE = DEFAULT_GRID_SIZE

# Battle-ready template for the hero squad used in campaign battles.
BLUE_TEAM_TEMPLATE = [
    {
        "name": "Одержимый",
        "initiative": 50,
        "initiative_base": 50,
        "team": "blue",
        "position": 7,
        "stand": "ahead",
        "unit_type": "Warrior",
        "damage": 25,
        "damage_secondary": 0,
        "health": 120,
        "max_health": 120,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 25,
        "exp_required": 95,
        "exp_current": 0,
        "turns_into": ["Берсерк"],
    },
    {
        "name": "Герцог",
        "initiative": 50,
        "initiative_base": 50,
        "team": "blue",
        "position": 8,
        "stand": "ahead",
        "unit_type": "Warrior",
        "damage": 50,
        "damage_secondary": 0,
        "health": 150,
        "max_health": 150,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 60,
        "exp_required": 150,
        "exp_current": 0,
        "turns_into": [],
    },
    {
        "name": "Одержимый",
        "initiative": 50,
        "initiative_base": 50,
        "team": "blue",
        "position": 9,
        "stand": "ahead",
        "unit_type": "Warrior",
        "damage": 25,
        "damage_secondary": 0,
        "health": 120,
        "max_health": 120,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 25,
        "exp_required": 95,
        "exp_current": 0,
        "turns_into": ["Берсерк"],
    },
    {
        "name": "пусто",
        "initiative": 0,
        "initiative_base": 0,
        "team": "blue",
        "position": 10,
        "stand": "behind",
        "unit_type": "Archer",
        "damage": 0,
        "damage_secondary": 0,
        "health": 0,
        "max_health": 0,
        "armor": 0,
        "accuracy": 0,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 0,
        "exp_required": 0,
        "exp_current": 0,
        "turns_into": [],
    },
    {
        "name": "Сектант",
        "initiative": 40,
        "initiative_base": 40,
        "team": "blue",
        "position": 11,
        "stand": "behind",
        "unit_type": "Mage",
        "damage": 15,
        "damage_secondary": 0,
        "health": 45,
        "max_health": 45,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Fire",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 20,
        "exp_required": 70,
        "exp_current": 0,
        "turns_into": ["Колдун", "Ведьма"],
    },
    {
        "name": "пусто",
        "initiative": 0,
        "initiative_base": 0,
        "team": "blue",
        "position": 12,
        "stand": "behind",
        "unit_type": "Archer",
        "damage": 0,
        "damage_secondary": 0,
        "health": 0,
        "max_health": 0,
        "armor": 0,
        "accuracy": 0,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "basestats": [],
        "exp_kill": 0,
        "exp_required": 0,
        "exp_current": 0,
        "turns_into": [],
    },
]

for _unit in BLUE_TEAM_TEMPLATE:
    _unit["hero"] = is_hero_name(_unit.get("name"))


def clamp_grid_pos(pos: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    """Clamp a coordinate to the current grid bounds."""
    x = int(pos[0])
    y = int(pos[1])
    return (
        max(0, min(int(grid_size) - 1, x)),
        max(0, min(int(grid_size) - 1, y)),
    )


def scale_enemy_positions(
    target_grid_size: int,
    *,
    base_grid_size: int = BASE_GRID_SIZE,
    base_enemy_positions: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, Tuple[int, int]]:
    """Scale the base campaign enemy layout to an arbitrary grid size.

    Масштабирование сохраняет относительное положение enemy stacks на карте.
    Для базового размера 48x48 координаты возвращаются как есть, чтобы не было
    дрейфа из-за округления.
    """
    if target_grid_size <= 1:
        raise ValueError("target_grid_size must be >= 2")

    base_positions = (
        dict(base_enemy_positions)
        if base_enemy_positions is not None
        else dict(BASE_ENEMY_POSITIONS)
    )
    if target_grid_size == base_grid_size:
        return dict(base_positions)

    base_span = int(base_grid_size) - 1
    target_span = int(target_grid_size) - 1
    scale = float(target_span) / float(base_span)
    scaled_positions: Dict[int, Tuple[int, int]] = {}

    for enemy_id, (x, y) in base_positions.items():
        sx = int(round(x * scale))
        sy = int(round(y * scale))
        scaled_positions[int(enemy_id)] = (
            max(0, min(target_span, sx)),
            max(0, min(target_span, sy)),
        )

    return scaled_positions


def scale_static_tiles(
    target_grid_size: int,
    base_tiles: Tuple[Tuple[int, int], ...],
    *,
    base_grid_size: int = BASE_GRID_SIZE,
) -> Tuple[Tuple[int, int], ...]:
    """Scale a static set of map tiles to the current grid size.

    Используется для heal tiles, сундуков, маны, пустых клеток и obstacle-blocks.
    Порядок входных координат сохраняется, потому что некоторые callers связывают
    масштабированные клетки с исходными записями через zip().
    """
    if target_grid_size <= 1:
        raise ValueError("target_grid_size must be >= 2")

    if target_grid_size == base_grid_size:
        return tuple(clamp_grid_pos(tile, target_grid_size) for tile in base_tiles)

    base_span = int(base_grid_size) - 1
    target_span = int(target_grid_size) - 1
    scale = float(target_span) / float(base_span)
    scaled_tiles: list[Tuple[int, int]] = []

    for x, y in base_tiles:
        sx = int(round(x * scale))
        sy = int(round(y * scale))
        scaled_tiles.append(
            (
                max(0, min(target_span, sx)),
                max(0, min(target_span, sy)),
            )
        )

    return tuple(scaled_tiles)


def _expand_static_blocks(
    blocks: Tuple[Tuple[int, int, int, int], ...],
) -> Tuple[Tuple[int, int], ...]:
    """Expand ordered rectangle footprints into ordered individual tiles.

    Прямоугольники могут пересекаться; seen_tiles убирает дубли, но сохраняет
    первый порядок обхода, чтобы obstacle generation оставалась стабильной.
    """
    expanded_tiles: list[Tuple[int, int]] = []
    seen_tiles: set[Tuple[int, int]] = set()

    for x, y, width, height in blocks:
        for dy in range(int(height)):
            for dx in range(int(width)):
                tile = (int(x) + dx, int(y) + dy)
                if tile in seen_tiles:
                    continue
                seen_tiles.add(tile)
                expanded_tiles.append(tile)

    return tuple(expanded_tiles)


def are_targets_reachable(
    grid_size: int,
    start_position: Tuple[int, int],
    blocked_tiles: set[Tuple[int, int]],
    targets: set[Tuple[int, int]],
) -> bool:
    """Check that the start can still reach every target tile.

    Обход 8-направленный, как и движение на campaign map. Функция используется
    при генерации препятствий: новый obstacle принимается только если он не
    отрезает старт, врагов, heal tiles и другие защищенные клетки.
    """
    start_tile = clamp_grid_pos(start_position, grid_size)
    if start_tile in blocked_tiles:
        return False

    pending_targets = {
        clamp_grid_pos(tile, grid_size)
        for tile in targets
        if clamp_grid_pos(tile, grid_size) not in blocked_tiles
    }
    if not pending_targets:
        return True

    visited = {start_tile}
    frontier = deque([start_tile])
    neighbor_offsets = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    while frontier and pending_targets:
        x, y = frontier.popleft()
        pending_targets.discard((x, y))

        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < int(grid_size) and 0 <= ny < int(grid_size)):
                continue
            tile = (nx, ny)
            if tile in blocked_tiles or tile in visited:
                continue
            visited.add(tile)
            frontier.append(tile)

    return not pending_targets


def _nearest_free_tile(
    anchor: Tuple[int, int],
    blocked_tiles: set[Tuple[int, int]],
    grid_size: int,
) -> Optional[Tuple[int, int]]:
    """Находит ближайшую свободную клетку к anchor.

    Tie-break идет по координате tile, чтобы при одинаковой дистанции результат
    был стабильным между запусками.
    """
    best_tile: Optional[Tuple[int, int]] = None
    best_distance: Optional[int] = None
    ax, ay = clamp_grid_pos(anchor, grid_size)

    for y in range(grid_size):
        for x in range(grid_size):
            tile = (x, y)
            if tile in blocked_tiles:
                continue
            distance = abs(x - ax) + abs(y - ay)
            if (
                best_distance is None
                or distance < best_distance
                or (distance == best_distance and tile < best_tile)
            ):
                best_tile = tile
                best_distance = distance

    return best_tile


def _farthest_free_tile(
    blocked_tiles: set[Tuple[int, int]],
    reference_tiles: Tuple[Tuple[int, int], ...],
    grid_size: int,
) -> Optional[Tuple[int, int]]:
    """Находит свободную клетку, максимально удаленную от reference_tiles.

    Используется как fallback для дополнительных heal tiles: сначала максимизируем
    минимальную дистанцию до уже выбранных точек, затем суммарную дистанцию.
    """
    best_tile: Optional[Tuple[int, int]] = None
    best_score: Optional[Tuple[int, int, int, int]] = None

    for y in range(grid_size):
        for x in range(grid_size):
            tile = (x, y)
            if tile in blocked_tiles:
                continue

            if reference_tiles:
                distances = [
                    (x - rx) * (x - rx) + (y - ry) * (y - ry)
                    for rx, ry in reference_tiles
                ]
                min_distance = min(distances)
                total_distance = sum(distances)
            else:
                min_distance = 0
                total_distance = 0

            score = (min_distance, total_distance, -y, -x)
            if best_score is None or score > best_score:
                best_tile = tile
                best_score = score

    return best_tile


def resolve_heal_tiles(
    grid_size: int,
    enemy_positions: Dict[int, Tuple[int, int]],
    *,
    start_position: Tuple[int, int] = DEFAULT_HERO_GRID_POSITION,
    heal_tile_count: int = HEAL_TILE_COUNT,
    base_static_heal_tiles: Tuple[Tuple[int, int], ...] = BASE_STATIC_HEAL_TILES,
) -> Tuple[Tuple[int, int], ...]:
    """Use the fixed scenario heal tiles first, then fill extra slots deterministically.

    Порядок важен: первые клетки соответствуют старту героя и поселениям. Если
    heal_tile_count больше статического списка, оставшиеся клетки добираются
    около углов, а затем максимально далеко от уже выбранных heal tiles.
    """
    desired_count = max(0, int(heal_tile_count))
    if desired_count == 0:
        return ()

    heal_tiles: list[Tuple[int, int]] = []
    heal_seen: set[Tuple[int, int]] = set()
    static_heal_tiles = scale_static_tiles(
        grid_size,
        base_static_heal_tiles,
        base_grid_size=BASE_GRID_SIZE,
    )

    for raw_tile in static_heal_tiles:
        tile = clamp_grid_pos(raw_tile, grid_size)
        if tile in heal_seen:
            continue
        heal_tiles.append(tile)
        heal_seen.add(tile)
        if len(heal_tiles) >= desired_count:
            return tuple(heal_tiles)

    enemy_tiles = {tuple(pos) for pos in enemy_positions.values()}

    if grid_size >= 4:
        margin = 1
    else:
        margin = 0
    far_edge = max(0, int(grid_size) - 1 - margin)
    near_edge = margin

    preferred_anchors = (
        clamp_grid_pos(start_position, grid_size),
        (far_edge, near_edge),
        (near_edge, far_edge),
        (far_edge, far_edge),
    )

    blocked_tiles = set(enemy_tiles)
    blocked_tiles.update(heal_tiles)

    for anchor in preferred_anchors:
        tile = _nearest_free_tile(anchor, blocked_tiles, grid_size)
        if tile is None:
            continue
        heal_tiles.append(tile)
        blocked_tiles.add(tile)
        if len(heal_tiles) >= desired_count:
            break

    while len(heal_tiles) < desired_count:
        tile = _farthest_free_tile(blocked_tiles, tuple(heal_tiles), grid_size)
        if tile is None:
            break
        heal_tiles.append(tile)
        blocked_tiles.add(tile)

    return tuple(heal_tiles)


def resolve_legions_settlement_territory_sources(
    grid_size: int,
    enemy_positions: Dict[int, Tuple[int, int]],
    *,
    base_settlement_territory_data: Tuple[Dict[str, object], ...] = (
        BASE_LEGIONS_SETTLEMENT_TERRITORY_DATA
    ),
    base_grid_size: int = BASE_GRID_SIZE,
    base_settlement_level_by_heal_tile: Mapping[Tuple[int, int], int] = (
        BASE_SETTLEMENT_LEVEL_BY_HEAL_TILE
    ),
) -> Tuple[Dict[str, object], ...]:
    """Resolve static settlement territory sources to the current grid size.

    Источник территории поселения должен совпадать с живым защитным enemy stack,
    если такой стек есть на карте. Если enemy_id отсутствует, используем
    масштабированную статическую клетку поселения.
    """
    resolved_sources: list[Dict[str, object]] = []
    for raw_entry in base_settlement_territory_data:
        entry = dict(raw_entry or {})
        required_enemy_ids = tuple(
            int(enemy_id)
            for enemy_id in tuple(entry.get("required_enemy_ids", ()))
        )
        base_source_tile = tuple(entry.get("source_tile", (0, 0)))
        scenario_settlement_level = max(
            1,
            int(base_settlement_level_by_heal_tile.get(base_source_tile, 1) or 1),
        )
        source_tile = None
        for enemy_id in required_enemy_ids:
            if enemy_id in enemy_positions:
                source_tile = clamp_grid_pos(enemy_positions[enemy_id], grid_size)
                break
        if source_tile is None:
            source_tile = scale_static_tiles(
                grid_size,
                (base_source_tile,),
                base_grid_size=base_grid_size,
            )[0]
        resolved_sources.append(
            {
                "name": str(entry.get("name", "") or ""),
                "source_tile": tuple(source_tile),
                "required_enemy_ids": required_enemy_ids,
                "territory_level": scenario_settlement_level,
                "expansion_per_turn": max(
                    0,
                    int(
                        LEGIONS_SETTLEMENT_TERRITORY_EXPANSION_BY_LEVEL.get(
                            scenario_settlement_level,
                            0,
                        )
                        or 0
                    ),
                ),
                "settlement_level": int(
                    base_settlement_level_by_heal_tile.get(base_source_tile, 0)
                    or 0
                ),
            }
        )
    return tuple(resolved_sources)


def resolve_chests(
    grid_size: int,
    enemy_positions: Dict[int, Tuple[int, int]],
    *,
    heal_tiles: Tuple[Tuple[int, int], ...] | list[Tuple[int, int]] = (),
    obstacle_tiles: Tuple[Tuple[int, int], ...] | list[Tuple[int, int]] | set[Tuple[int, int]] = (),
    start_position: Tuple[int, int] = DEFAULT_HERO_GRID_POSITION,
    base_static_chests: Tuple[Tuple[Tuple[int, int], Tuple[str, ...]], ...] = BASE_STATIC_CHESTS,
) -> Dict[Tuple[int, int], Tuple[str, ...]]:
    """Resolve deterministic treasure chest tiles and the items stored in each chest.

    Сундук не должен появиться на старте, враге, heal tile или obstacle. Если
    статическая координата занята после масштабирования, переносим его на
    ближайшую свободную клетку.
    """
    if not base_static_chests:
        return {}

    base_tiles = tuple(pos for pos, _ in base_static_chests)
    scaled_tiles = scale_static_tiles(
        grid_size,
        base_tiles,
        base_grid_size=BASE_GRID_SIZE,
    )
    reserved_tiles = {tuple(pos) for pos in enemy_positions.values()}
    reserved_tiles.update(tuple(tile) for tile in heal_tiles)
    reserved_tiles.update(clamp_grid_pos(tile, grid_size) for tile in obstacle_tiles)
    reserved_tiles.add(clamp_grid_pos(start_position, grid_size))

    resolved: Dict[Tuple[int, int], Tuple[str, ...]] = {}
    for raw_entry, raw_tile in zip(base_static_chests, scaled_tiles):
        _, item_names = raw_entry
        tile = clamp_grid_pos(raw_tile, grid_size)
        if tile in reserved_tiles or tile in resolved:
            tile = _nearest_free_tile(
                tile,
                reserved_tiles | set(resolved.keys()),
                grid_size,
            )
            if tile is None:
                continue
        resolved[tile] = tuple(str(item_name) for item_name in item_names if str(item_name))
        reserved_tiles.add(tile)

    return resolved


def resolve_mana_sources(
    grid_size: int,
    *,
    reserved_tiles: (
        Tuple[Tuple[int, int], ...]
        | list[Tuple[int, int]]
        | set[Tuple[int, int]]
    ) = (),
    base_static_mana_sources: Tuple[Tuple[str, Tuple[int, int]], ...] = BASE_STATIC_MANA_SOURCES,
    base_grid_size: int = BASE_GRID_SIZE,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Scale the static mana-source layout to the current grid size.

    Возвращается уже UI-ready dict: CampaignEnv/renderer могут брать отсюда
    русское имя, букву, цвета и ANSI-код без повторного lookup. Если источник
    попадает на занятую клетку или поверх другого источника, переносим его на
    ближайшую свободную клетку.
    """
    if not base_static_mana_sources:
        return {}

    scaled_tiles = scale_static_tiles(
        grid_size,
        tuple(position for _, position in base_static_mana_sources),
        base_grid_size=base_grid_size,
    )
    blocked_tiles = {clamp_grid_pos(tile, grid_size) for tile in reserved_tiles}
    resolved: Dict[Tuple[int, int], Dict[str, object]] = {}
    for (kind, _), scaled_tile in zip(base_static_mana_sources, scaled_tiles):
        tile = clamp_grid_pos(scaled_tile, grid_size)
        if tile in blocked_tiles or tile in resolved:
            tile = _nearest_free_tile(
                tile,
                blocked_tiles | set(resolved.keys()),
                grid_size,
            )
            if tile is None:
                continue
        display = dict(MANA_SOURCE_DISPLAY_SPECS.get(kind, {}))
        resolved[tile] = {
            "kind": str(kind),
            "name": str(display.get("name", kind) or kind),
            "letter": str(display.get("letter", "?") or "?")[:1],
            "fill_color": tuple(display.get("fill_color", (0.8, 0.8, 0.8, 0.8))),
            "edge_color": tuple(display.get("edge_color", (0.3, 0.3, 0.3, 1.0))),
            "text_color": tuple(display.get("text_color", (0.0, 0.0, 0.0, 1.0))),
            "ansi_color": str(display.get("ansi_color", "") or ""),
        }
        blocked_tiles.add(tile)
    return resolved


@lru_cache(maxsize=64)
def _resolve_obstacle_tiles_cached(
    grid_size: int,
    enemy_tiles: Tuple[Tuple[int, int], ...],
    heal_tiles: Tuple[Tuple[int, int], ...],
    start_position: Tuple[int, int],
    base_static_obstacle_blocks: Tuple[Tuple[int, int, int, int], ...],
    base_static_empty_tiles: Tuple[Tuple[int, int], ...],
) -> Tuple[Tuple[int, int], ...]:
    """Cached implementation for immutable obstacle-layout inputs."""
    start_tile = clamp_grid_pos(start_position, grid_size)
    reserved_tiles = set(enemy_tiles)
    reserved_tiles.update(heal_tiles)
    reserved_tiles.add(start_tile)
    reserved_tiles.update(
        scale_static_tiles(
            grid_size,
            base_static_empty_tiles,
            base_grid_size=BASE_GRID_SIZE,
        )
    )
    protected_targets = set(reserved_tiles)

    static_obstacle_tiles = scale_static_tiles(
        grid_size,
        _expand_static_blocks(base_static_obstacle_blocks),
        base_grid_size=BASE_GRID_SIZE,
    )

    obstacle_tiles: list[Tuple[int, int]] = []
    obstacle_seen: set[Tuple[int, int]] = set()
    for raw_tile in static_obstacle_tiles:
        tile = clamp_grid_pos(raw_tile, grid_size)
        if tile in reserved_tiles or tile in obstacle_seen:
            continue

        candidate_blocked_tiles = set(obstacle_seen)
        candidate_blocked_tiles.add(tile)
        if not are_targets_reachable(
            grid_size,
            start_tile,
            candidate_blocked_tiles,
            protected_targets,
        ):
            continue
        obstacle_tiles.append(tile)
        obstacle_seen.add(tile)

    return tuple(obstacle_tiles)


def resolve_obstacle_tiles(
    grid_size: int,
    enemy_positions: Dict[int, Tuple[int, int]],
    *,
    heal_tiles: Tuple[Tuple[int, int], ...] | list[Tuple[int, int]] = (),
    start_position: Tuple[int, int] = DEFAULT_HERO_GRID_POSITION,
    base_static_obstacle_blocks: Tuple[Tuple[int, int, int, int], ...] = (
        BASE_STATIC_OBSTACLE_BLOCKS
    ),
    base_static_empty_tiles: Tuple[Tuple[int, int], ...] = BASE_STATIC_EMPTY_TILES,
) -> Tuple[Tuple[int, int], ...]:
    """Build deterministic obstacle tiles for the campaign map.

    Obstacles строятся только из статических rectangle blocks сценария. Каждый
    candidate проверяется на reachability, чтобы важные клетки не оказались
    отрезаны от стартовой позиции. Поскольку раскладка детерминирована,
    одинаковые immutable-входы между экземплярами CampaignEnv используют общий
    LRU-кэш.
    """
    normalized_enemy_tiles = tuple(
        sorted(
            {
                (int(pos[0]), int(pos[1]))
                for pos in enemy_positions.values()
            }
        )
    )
    normalized_heal_tiles = tuple(
        sorted(
            {
                (int(tile[0]), int(tile[1]))
                for tile in heal_tiles
            }
        )
    )
    normalized_start_position = (int(start_position[0]), int(start_position[1]))
    normalized_obstacle_blocks = tuple(
        (int(x), int(y), int(width), int(height))
        for x, y, width, height in base_static_obstacle_blocks
    )
    normalized_empty_tiles = tuple(
        (int(tile[0]), int(tile[1]))
        for tile in base_static_empty_tiles
    )
    return _resolve_obstacle_tiles_cached(
        int(grid_size),
        normalized_enemy_tiles,
        normalized_heal_tiles,
        normalized_start_position,
        normalized_obstacle_blocks,
        normalized_empty_tiles,
    )


# Preserve the familiar functools cache controls on the public wrapper. They are
# useful for deterministic tests and diagnostics without exposing the private helper.
resolve_obstacle_tiles.cache_info = _resolve_obstacle_tiles_cached.cache_info
resolve_obstacle_tiles.cache_clear = _resolve_obstacle_tiles_cached.cache_clear
