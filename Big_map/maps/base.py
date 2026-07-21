# maps/base.py
"""Shared MapConfig structure and stack-building helpers for campaign maps.

Инвариант импортов: модули maps/* могут импортировать только стандартную
библиотеку, data_dicts_compact_lines и campaign_env_scenario. Импорт grid,
enemy_configs или campaign_env_data отсюда создаст цикл.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

from data_dicts_compact_lines import DATA

_UNIT_NAME_ALIASES: Dict[str, str] = {
    "Выверна": "Виверна",
    "Гоблин-лучник": "Гоблин лучник",
    "Зеленый дракон": "Зелёный дракон",
    "Зелёный дракон": "Зелёный дракон",
    "Дракон Рока": "Дракон рока",
    "Лорд Тьмы": "Лорд тьмы",
}


def _canonical_enemy_name(name: str) -> str:
    """Возвращает нормализованное имя юнита для сравнений внутри map-конфига."""
    raw = str(name or "").strip()
    return _UNIT_NAME_ALIASES.get(raw, raw)


def _build_big_unit_names() -> frozenset[str]:
    """Собирает имена больших юнитов из DATA.

    В UNIT_DATA размер 1 означает large-unit: такой юнит занимает целую колонку
    переднего и заднего ряда в тактической сетке. Этот список нужен при упаковке
    enemy-team specs, чтобы случайно не поставить другого юнита в ту же колонку.
    """
    names: set[str] = set()
    for entry in DATA:
        if not isinstance(entry, dict):
            continue
        entry_name = entry.get("кто")
        if entry_name is None:
            entry_name = entry.get("РєС‚Рѕ")
        if not entry_name:
            continue
        try:
            size = int(entry.get("размер", 0) or 0)
        except (TypeError, ValueError):
            size = 0
        if size != 1:
            continue
        name = str(entry_name).strip()
        if not name:
            continue
        names.add(name)
        names.add(_canonical_enemy_name(name))
    return frozenset(names)


BIG_UNIT_NAMES = _build_big_unit_names()


def _is_big_enemy_unit(name: str | None) -> bool:
    """Проверяет, считается ли юнит большим для раскладки 3x2."""
    if not name:
        return False
    return _canonical_enemy_name(str(name)) in BIG_UNIT_NAMES


def _normalize_big_unit_rows(
    front: list[str | None],
    back: list[str | None],
) -> Tuple[list[str | None], list[str | None]]:
    """Освобождает колонку большого юнита в enemy-team spec.

    В бою большой юнит визуально занимает переднюю и заднюю ячейку одной колонки.
    Если в исходном описании рядом с ним стоит обычный юнит, пытаемся аккуратно
    перенести обычного юнита в ближайший свободный слот. Если места нет или
    конфигурация неоднозначная, возвращаем исходную раскладку без изменений.
    """
    if len(front) != 3 or len(back) != 3:
        return list(front), list(back)

    original_slots: list[tuple[str, int, str]] = []
    for row_name, row_values in (("front", front), ("back", back)):
        for col, name in enumerate(row_values):
            if isinstance(name, str) and name:
                original_slots.append((row_name, int(col), str(name)))

    if not original_slots:
        return list(front), list(back)

    big_slots = [
        (row_name, col, name)
        for row_name, col, name in original_slots
        if _is_big_enemy_unit(name)
    ]
    if not big_slots:
        return list(front), list(back)

    blocked_cols = [col for _, col, _ in big_slots]
    if len(set(blocked_cols)) != len(blocked_cols):
        return list(front), list(back)

    result_front: list[str | None] = [None, None, None]
    result_back: list[str | None] = [None, None, None]
    blocked_col_set = set(blocked_cols)

    for _, col, name in big_slots:
        result_front[col] = name

    displaced_units: list[tuple[str, int, str]] = []
    for row_name, col, name in original_slots:
        if _is_big_enemy_unit(name):
            continue
        target_row = result_front if row_name == "front" else result_back
        if col not in blocked_col_set and target_row[col] is None:
            target_row[col] = name
            continue
        displaced_units.append((row_name, col, name))

    def _candidate_slots(preferred_row: str, original_col: int) -> list[tuple[str, int]]:
        """Возвращает свободные слоты от ближайшего к исходной колонке к дальнему."""
        row_order = [preferred_row, "back" if preferred_row == "front" else "front"]
        candidates: list[tuple[str, int]] = []
        for row_name in row_order:
            target_row = result_front if row_name == "front" else result_back
            free_cols = [
                col
                for col in range(3)
                if col not in blocked_col_set and target_row[col] is None
            ]
            free_cols.sort(key=lambda col: (abs(col - original_col), col))
            candidates.extend((row_name, col) for col in free_cols)
        return candidates

    for preferred_row, original_col, name in displaced_units:
        placed = False
        for row_name, col in _candidate_slots(preferred_row, original_col):
            target_row = result_front if row_name == "front" else result_back
            if target_row[col] is None:
                target_row[col] = name
                placed = True
                break
        if not placed:
            return list(front), list(back)

    return result_front, result_back


def _normalize_enemy_team_spec(spec: dict[str, object]) -> dict[str, object]:
    """Нормализует одну запись enemy-team specs перед передачей в CampaignEnv."""
    front, back = _normalize_big_unit_rows(
        list(spec["front"]),
        list(spec["back"]),
    )
    return {
        "description": str(spec["description"]),
        "front": front,
        "back": back,
    }


def _pack_stack_units(units: Tuple[str, ...]) -> Tuple[list[str | None], list[str | None]]:
    """Упаковывает список до 6 имен в front/back ряды формата BattleEnv.

    Правила намеренно простые и детерминированные: одиночный юнит идет в центр,
    два юнита - в центр переднего и заднего ряда, более крупные стеки занимают
    слоты слева направо. Затем вызывающий код отдельно нормализует large-unit.
    """
    unit_list = [str(name) for name in units if isinstance(name, str) and name][:6]
    slots: list[str | None] = [None, None, None, None, None, None]
    count = len(unit_list)

    if count <= 0:
        return list(slots[:3]), list(slots[3:])
    if count == 1:
        slots[1] = unit_list[0]
    elif count == 2:
        slots[1] = unit_list[0]
        slots[4] = unit_list[1]
    elif count == 3:
        slots[:3] = unit_list
    elif count == 4:
        slots[:3] = unit_list[:3]
        slots[4] = unit_list[3]
    elif count == 5:
        slots[:3] = unit_list[:3]
        slots[3] = unit_list[3]
        slots[5] = unit_list[4]
    else:
        slots = unit_list[:6]

    return list(slots[:3]), list(slots[3:])


def make_stack_override(
    enemy_id: int,
    position: Tuple[int, int],
    units: Tuple[str, ...],
    *,
    label: Optional[str] = None,
) -> dict[str, object]:
    """Создает override-запись для карты по краткому списку юнитов."""
    front, back = _pack_stack_units(units)
    front, back = _normalize_big_unit_rows(front, back)
    composition = ", ".join(units) if units else "(empty)"
    return {
        "enemy_id": int(enemy_id),
        "position": tuple(position),
        "description": str(label) if label else f"Отряд: {composition}",
        "front": front,
        "back": back,
    }


def _no_tiles() -> Tuple[Tuple[int, int], ...]:
    """Пустой terrain-провайдер для карт без воды/леса/дорог."""
    return ()


@dataclass(frozen=True)
class MapConfig:
    """Полное описание содержимого одной кампанейской карты.

    enemy_stacks хранит записи единого формата (как MATCHED_MAP_STACKS):
    {"enemy_id", "position", "description", "front", "back"}. При дубликатах
    enemy_id побеждает более поздняя запись — так же, как .update() в grid.py.
    """

    name: str
    # Координатная база данных карты. Все клетки в этом конфиге заданы в сетке
    # grid_size x grid_size; среда масштабирует их к игровому размеру.
    grid_size: int
    hero_start: Tuple[int, int]
    # Статическая раскладка.
    obstacle_blocks: Tuple[Tuple[int, int, int, int], ...]
    empty_tiles: Tuple[Tuple[int, int], ...]
    village_heal_tiles: Tuple[Tuple[int, int], ...]
    settlement_level_by_heal_tile: Mapping[Tuple[int, int], int]
    legions_settlement_territory_data: Tuple[Dict[str, object], ...]
    settlement_defender_heal_tile_by_enemy_id: Mapping[int, Tuple[int, int]]
    chests: Tuple[Tuple[Tuple[int, int], Tuple[str, ...]], ...]
    mana_sources: Tuple[Tuple[str, Tuple[int, int]], ...]
    # Вражеские отряды.
    enemy_stacks: Tuple[Dict[str, object], ...]
    # Точки взаимодействия.
    merchant_sites: Mapping[str, Dict]
    spell_shop_sites: Mapping[str, Dict]
    mercenary_sites: Mapping[str, Dict]
    trainer_sites: Mapping[str, Dict]
    # Цель кампании и фракционные привязки.
    final_objective_cities: Mapping[str, Tuple[int, ...]]
    ruin_rewards: Mapping[int, Dict[str, object]]
    default_objective: str
    objective_enemy_id: Optional[int]
    empire_territory_source_enemy_id: Optional[int]
    empire_territory_source_tile: Optional[Tuple[int, int]]
    scripted_capital_bot_supported: bool
    # Optional map-specific scripted-bot identity. When omitted, the legacy
    # Empire-capital tile and hard-coded party remain in effect.
    scripted_capital_bot_home_tile: Optional[Tuple[int, int]] = None
    scripted_capital_bot_faction: str = "Империя"
    scripted_capital_bot_roster: Optional[Mapping[int, str]] = None
    scripted_capital_bot_home_enemy_ids: Tuple[int, ...] = ()
    scripted_capital_bot_protected_enemy_ids: Tuple[int, ...] = ()
    # Optional explicit objective allow-list. Empty keeps legacy capability-
    # based validation for existing maps.
    supported_objectives: Tuple[str, ...] = ()
    # Игровой размер сетки по умолчанию (None = grid_size). Позволяет играть
    # уменьшенную копию карты: данные в 48-координатах, игра на 24x24.
    play_grid_size: Optional[int] = None
    # Стартовый ростер агента по умолчанию: True = боссовый (демоны),
    # False = базовый шаблон (Одержимые, Герцог, Сектант). Параметр
    # use_boss_starting_roster у CampaignEnv переопределяет это значение.
    boss_starting_roster_default: bool = True
    # Optional fixed map roster: battle position 7-12 -> unit name.
    # When set, it overrides the usual faction/lord and boss rosters.
    starting_roster: Optional[Mapping[int, str]] = None
    # Optional per-position changes applied after the roster unit is built.
    # Useful for scenario aliases and explicit hero/type flags without
    # duplicating the canonical unit database entry.
    starting_unit_overrides: Optional[Mapping[int, Mapping[str, object]]] = None
    # Optional display-name aliases resolved back to canonical UNIT_DATA names.
    unit_name_aliases: Optional[Mapping[str, str]] = None
    # Gold restored at the beginning of every episode on this map.
    starting_gold: float = 0.0
    # Optional map-specific replacement for the global merchant assortment.
    merchant_buy_items: Optional[Tuple[Dict[str, object], ...]] = None
    # Optional exact inventory per merchant site. Item definitions still come
    # from merchant_buy_items, while these mappings control per-site stocks.
    merchant_buy_items_by_site: Optional[
        Mapping[str, Tuple[Dict[str, object], ...]]
    ] = None
    # Arbitrary items placed directly into the hero inventory at episode start.
    starting_hero_items: Tuple[str, ...] = ()
    # Scrolls placed directly into the hero inventory at episode start.
    starting_scroll_items: Tuple[str, ...] = ()
    starting_scroll_magic_unlocked: bool = False
    # Optional map-specific training rewards for items collected from chests.
    item_pickup_rewards: Optional[Mapping[str, float]] = None
    # Optional overrides for the generic battle-item shaping rewards.
    battle_item_equip_reward: Optional[float] = None
    battle_item_use_reward: Optional[float] = None
    # Abilities granted to the map's travel hero after normal level-1 progression sync.
    starting_hero_abilities: Tuple[str, ...] = ()
    # Optional fixed faction/lord identity for scenario-specific starting parties.
    starting_capital_id: Optional[int] = None
    starting_lord_type: Optional[int] = None
    # Optional per-free-leadership-point step penalty override for focused maps.
    leadership_step_penalty: Optional[float] = None
    # Optional override for the REST/end-turn campaign penalty.
    turn_penalty: Optional[float] = None
    # Some focused training maps intentionally disable all passive turn mana income.
    passive_mana_income_enabled: bool = True
    # Optional per-enemy unit level overrides: enemy id -> unit name -> level.
    enemy_unit_level_overrides: Optional[Mapping[int, Mapping[str, int]]] = None
    # Optional dormant pursuer that spawns on a campaign turn and moves toward the hero.
    scheduled_enemy_id: Optional[int] = None
    scheduled_enemy_spawn_turn: Optional[int] = None
    scheduled_enemy_moves_per_turn: int = 0
    # Optional ordered list of pursuing waves. Each entry references either one
    # enemy_id or a simultaneous enemy_ids group from enemy_stacks and may define
    # spawn_turn, optional spawn_interval between group members, moves_per_turn
    # and wave_index. Legacy single-pursuer fields above remain supported.
    scheduled_enemy_waves: Tuple[Dict[str, object], ...] = ()
    # Extra reward for completing wave N is
    # reward_defeat_enemy * wave_reward_scale * N.
    wave_reward_scale: float = 1.0
    # Keep enemy tiles path-reachable unless a scenario intentionally requires magic.
    enforce_enemy_reachability: bool = True
    # Местность: ленивые провайдеры (например, парсинг .sg-файла).
    water_tiles_provider: Callable[[], Tuple[Tuple[int, int], ...]] = _no_tiles
    forest_tiles_provider: Callable[[], Tuple[Tuple[int, int], ...]] = _no_tiles
    road_tiles_provider: Callable[[], Tuple[Tuple[int, int], ...]] = _no_tiles
    gold_mine_tiles_provider: Callable[[], Tuple[Tuple[int, int], ...]] = _no_tiles
    territory_forbidden_tiles_provider: Callable[[], Tuple[Tuple[int, int], ...]] = _no_tiles

    @property
    def static_heal_tiles(self) -> Tuple[Tuple[int, int], ...]:
        """Первая клетка — столица/старт героя, дальше поселения."""
        return (
            (int(self.hero_start[0]), int(self.hero_start[1])),
            *((int(tile[0]), int(tile[1])) for tile in self.village_heal_tiles),
        )

    @property
    def heal_tile_count(self) -> int:
        return len(self.static_heal_tiles)

    def enemy_positions(self) -> Dict[int, Tuple[int, int]]:
        """enemy_id -> клетка; при дубликатах id побеждает поздняя запись."""
        positions: Dict[int, Tuple[int, int]] = {}
        for row in self.enemy_stacks:
            positions[int(row["enemy_id"])] = tuple(row["position"])
        return positions

    def enemy_team_specs(self) -> Dict[int, Dict[str, object]]:
        """enemy_id -> нормализованный состав отряда (как grid.ENEMY_TEAM_SPECS)."""
        specs: Dict[int, Dict[str, object]] = {}
        for row in self.enemy_stacks:
            specs[int(row["enemy_id"])] = {
                "description": str(row["description"]),
                "front": list(row["front"]),
                "back": list(row["back"]),
            }
        return {
            enemy_id: _normalize_enemy_team_spec(spec)
            for enemy_id, spec in specs.items()
        }
