# campaign_env.py
"""
Двухуровневая среда: Grid World + Battle.
ВЕРСИЯ БЕЗ ВОССТАНОВЛЕНИЯ HP.

Агент перемещается по карте 48×48 (по умолчанию), встречает врагов и сражается с ними.
Список врагов и их позиции соответствуют совпавшим отрядам из сценарной карты.

После каждой победы HP юнитов BLUE НЕ восстанавливается,
погибшие юниты НЕ воскрешаются. Только сбрасываются временные эффекты.

Схема наград:
  - За участие в битве (нашёл врага): +0.1
  - За победу в каждой битве: +0.3
  - За полную зачистку карты: финал кампании
  - За поражение: -1.0
  - За таймаут (лимит шагов): -6.0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from functools import lru_cache
from typing import Optional, Dict, List, Tuple
from copy import deepcopy
from pathlib import Path

from battle_env import BattleEnv, UNITS_BLUE, FEATURES_PER_UNIT
from enemy_configs import ENEMY_CONFIGS, ENEMY_DESCRIPTIONS
from grid import (
    CAPITAL_HEAL_TILE_ARMOR_BONUS,
    CAPITAL_INTERNAL_STACK_OVERRIDES,
    RUIN_STACK_OVERRIDES,
    are_targets_reachable,
    DEFAULT_GRID_SIZE,
    DEFAULT_HERO_GRID_POSITION,
    HEAL_TILE_COUNT,
    SETTLEMENT_ARMOR_BONUS_BY_LEVEL,
    SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID,
    resolve_chests,
    resolve_legions_settlement_territory_sources,
    resolve_mana_sources,
    resolve_obstacle_tiles,
    resolve_heal_tiles,
    scale_static_tiles,
    scale_enemy_positions,
    VILLAGE_INTERNAL_GARRISON_OVERRIDES,
    VILLAGE_LINKED_STACK_OVERRIDES,
)
from grid_world_env import GridWorldEnv
from data_dicts_compact_lines import DATA as UNIT_DATA, map_unit_to_battle, placeholder_unit
from Buildings import (
    empire_buildings_d2,
    mountain_clans_buildings_d2,
    undead_hordes_buildings_d2,
    legions_buildings_d2,
    elves_buildings_d2,
)
from spells import (
    empire_spells_d2,
    mountain_clans_spells_d2,
    undead_hordes_spells_d2,
    legions_spells_d2,
    elves_spells_d2,
)
from hire import (
    empire_hire_d2,
    mountain_clans_hire_d2,
    undead_hordes_hire_d2,
    legions_hire_d2,
    elves_hire_d2,
)
from scroll_spell_data import SCROLL_ITEM_ALIASES, SCROLL_ITEM_DEFINITIONS

BUILDINGS_D2 = {
    "empire": empire_buildings_d2,
    "mountain_clans": mountain_clans_buildings_d2,
    "undead_hordes": undead_hordes_buildings_d2,
    "legions": legions_buildings_d2,
    "elves": elves_buildings_d2,
}
BUILDINGS_D2_TEMPLATE = {key: deepcopy(value) for key, value in BUILDINGS_D2.items()}
SPELLS_D2 = {
    "empire": empire_spells_d2,
    "mountain_clans": mountain_clans_spells_d2,
    "undead_hordes": undead_hordes_spells_d2,
    "legions": legions_spells_d2,
    "elves": elves_spells_d2,
}
SPELLS_D2_TEMPLATE = {key: deepcopy(value) for key, value in SPELLS_D2.items()}
HIRE_D2 = {
    "empire": empire_hire_d2,
    "mountain_clans": mountain_clans_hire_d2,
    "undead_hordes": undead_hordes_hire_d2,
    "legions": legions_hire_d2,
    "elves": elves_hire_d2,
}

DEFAULT_SCENARIO_FILENAME = "A Return To Simpler Times.sg"
LEGIONS_TERRITORY_PASSABLE_POSITIONED_CLASSES = frozenset(
    {
        ".?AVCMidLandmark@@",
        ".?AVCMidLocation@@",
        ".?AVCMidRoad@@",
    }
)


@lru_cache(maxsize=1)
def _load_legions_territory_base_forbidden_tiles() -> Tuple[Tuple[int, int], ...]:
    scenario_path = Path(__file__).resolve().with_name(DEFAULT_SCENARIO_FILENAME)
    if not scenario_path.exists():
        return ()

    try:
        from tools.inspect_sg_map import (
            RENDER_OBJECT_SPECS,
            extract_render_objects,
            parse_map_size,
            parse_positioned_objects,
            parse_resource_objects,
            parse_water_cells,
        )
    except Exception:
        return ()

    try:
        data = scenario_path.read_bytes()
        map_size = int(parse_map_size(data))
        water_cells = parse_water_cells(data, map_size)
        render_objects = extract_render_objects(data)
        positioned_objects = parse_positioned_objects(data)
        resource_objects = parse_resource_objects(data)
    except Exception:
        return ()

    forbidden_tiles: set[Tuple[int, int]] = {
        (int(x), int(y))
        for x, y in water_cells
        if 0 <= int(x) < map_size and 0 <= int(y) < map_size
    }

    for class_name, objects in render_objects.items():
        spec = RENDER_OBJECT_SPECS.get(class_name, {})
        width, height = spec.get("footprint", (1, 1))
        width = max(1, int(width))
        height = max(1, int(height))
        for obj in objects:
            ox = int(obj.get("x", 0) or 0)
            oy = int(obj.get("y", 0) or 0)
            for tx in range(ox, min(map_size, ox + width)):
                for ty in range(oy, min(map_size, oy + height)):
                    forbidden_tiles.add((tx, ty))

    for obj in positioned_objects:
        if str(obj.get("class", "") or "") in LEGIONS_TERRITORY_PASSABLE_POSITIONED_CLASSES:
            continue
        ox = int(obj.get("x", 0) or 0)
        oy = int(obj.get("y", 0) or 0)
        if 0 <= ox < map_size and 0 <= oy < map_size:
            forbidden_tiles.add((ox, oy))

    resource_tiles = {
        (int(resource.get("x", 0) or 0), int(resource.get("y", 0) or 0))
        for resource in resource_objects
        if str(resource.get("class", "") or "").startswith("resource:")
    }
    chest_tiles = {
        (int(obj.get("x", 0) or 0), int(obj.get("y", 0) or 0))
        for obj in render_objects.get(".?AVCMidBag@@", ())
    }
    forbidden_tiles.difference_update(resource_tiles | chest_tiles)

    return tuple(sorted(forbidden_tiles))


@lru_cache(maxsize=1)
def _load_legions_territory_base_water_tiles() -> Tuple[Tuple[int, int], ...]:
    scenario_path = Path(__file__).resolve().with_name(DEFAULT_SCENARIO_FILENAME)
    if not scenario_path.exists():
        return ()

    try:
        from tools.inspect_sg_map import parse_map_size, parse_water_cells
    except Exception:
        return ()

    try:
        data = scenario_path.read_bytes()
        map_size = int(parse_map_size(data))
        water_cells = parse_water_cells(data, map_size)
    except Exception:
        return ()

    return tuple(
        sorted(
            (int(x), int(y))
            for x, y in water_cells
            if 0 <= int(x) < map_size and 0 <= int(y) < map_size
        )
    )


@lru_cache(maxsize=1)
def _load_legions_territory_base_gold_mine_tiles() -> Tuple[Tuple[int, int], ...]:
    scenario_path = Path(__file__).resolve().with_name(DEFAULT_SCENARIO_FILENAME)
    if not scenario_path.exists():
        return ()

    try:
        from tools.inspect_sg_map import parse_map_size, parse_resource_objects
    except Exception:
        return ()

    try:
        data = scenario_path.read_bytes()
        map_size = int(parse_map_size(data))
        resource_objects = parse_resource_objects(data)
    except Exception:
        return ()

    return tuple(
        sorted(
            {
                (int(resource.get("x", 0) or 0), int(resource.get("y", 0) or 0))
                for resource in resource_objects
                if str(resource.get("class", "") or "") == "resource:0"
                and 0 <= int(resource.get("x", 0) or 0) < map_size
                and 0 <= int(resource.get("y", 0) or 0) < map_size
            }
        )
    )


class HeroInventoryItem(str):
    """String-compatible inventory entry that also stores gold value."""

    def __new__(cls, name: str, gold: int = 0):
        obj = str.__new__(cls, str(name or ""))
        obj.gold = max(0, int(gold or 0))
        return obj

    def __repr__(self) -> str:
        return f"HeroInventoryItem(name={str.__repr__(self)}, gold={int(self.gold)})"


class CampaignEnv(gym.Env):
    """
    Двухуровневая среда: Grid World + Battle.

    Режимы:
        - MODE_GRID: агент перемещается по карте
        - MODE_BATTLE: агент сражается (используется BattleEnv)

    Награды:
        - Награда за участие в битве (нашёл врага): +0.1
        - Награда за победу в каждой битве: +0.3
        - Награда за успешный найм юнита после лидерства: 3 * reward_defeat_enemy
        - Награда за первый захваченный целевой город: reward_all_enemies
        - Награда за второй захваченный целевой город: reward_all_enemies * 5
        - Победа в кампании после захвата Порта Полонис и Соругириллы
        - Масштабированная награда из BattleEnv: battle_reward_scale * battle_reward
- Награда за лечение на специальных клетках: reward_castle_heal_per_hp * restored_hp
- Награда за успешное воскрешение на специальных клетках: reward_castle_revive
- Небольшая награда за бой основного отряда героя в тот же ход после призыва юнита
        - Минимальная награда за продажу бесполезного предмета у торговца
        - Дополнительный штраф за каждый завершённый ход, пока герою нужен юнит
        - Штраф за поражение: -1.0
        - Штраф за таймаут (лимит шагов на карте): -6.0

    Observation Space:
        Объединённое наблюдение:
        - [0]: режим (0 = grid, 1 = battle)
        - [1:1+GRID_OBS_SIZE]: grid observation
          (базовые признаки + все enemy stacks + BLUE roster + ресурсы/инвентарь
           + постройки + сундуки + клетки лечения + руины + прогресс по целевым городам)
        - [...]: battle observation (когда в бою) или нули
        - [...]: ход и золото

    Action Space:
        Discrete(...):
        - В режиме grid:
          - 0-7: движение в 8 направлениях
          - 8: отдых (REST) — восстановление 5% HP раненым юнитам
            (+ бонус лечения на клетках столицы и поселений)
          - 9-14: применить подходящую банку лечения к позициям BLUE 7-12
            (автовыбор +50 HP или +100 HP в зависимости от недостающего здоровья и запасов)
          - 15-20: применить бутыль воскрешения к позициям BLUE 7-12 (оживляет с 1 HP)
          - 21-26: применить зелье неуязвимости к позициям BLUE 7-12 (+50 брони на 1 ход)
          - 27-32: применить зелье силы к позициям BLUE 7-12 (+30% урона на 1 ход)
          - 33-38: лечение на специальных клетках карты по позициям BLUE 7-12
            (стоимость за 1 HP зависит от Level)
          - 39-44: воскрешение на специальных клетках карты по позициям BLUE 7-12
            (стоимость зависит от Level)
          - 45-54: покупка товаров у торговца
          - 55-60: применить Эликсир энергии к позициям 7-12
          - 61-66: применить Эликсир быстроты к позициям 7-12
          - 67-72: применить защиту от магии Огня к позициям 7-12
          - 73-78: применить защиту от магии Земли к позициям 7-12
          - 79-84: применить защиту от магии Воды к позициям 7-12
          - 85-90: применить защиту от магии Воздуха к позициям 7-12
          - 91-96: применить Эликсир Силы титана к позициям 7-12
          - 97-102: применить Эликсир Всевышнего к позициям 7-12
          - 103-...: постройка зданий активной фракции
        - В режиме battle: используются действия 0-14 (атака/защита/ожидание)

        Action masking применяется для блокировки недопустимых действий.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    MODE_GRID = 0
    MODE_BATTLE = 1

    GRID_OBS_SIZE = 0  # вычисляется в __init__
    BATTLE_OBS_SIZE = FEATURES_PER_UNIT * 12
    MOVES_PER_TURN = 20
    GRID_STEPS_PER_TURN = MOVES_PER_TURN
    HERO_PATHFINDING_LEVEL = 2
    HERO_LEADERSHIP_LEVEL = 3
    HERO_ENDURANCE_LEVEL = 4
    HERO_STRENGTH_LEVEL = 5
    HERO_SECOND_LEADERSHIP_LEVEL = 6
    HERO_PATHFINDING_MOVE_BONUS_PCT = 0.20
    TYPEOFLORD_TWO_SPELL_LEARNING_COST_MULTIPLIER = 0.5
    TYPEOFLORD_TWO_SAME_SPELL_CASTS_PER_TURN = 2
    TYPEOFLORD_THREE_BUILDING_COST_MULTIPLIER = 0.5
    TYPEOFLORD_ONE_REST_HEAL_BONUS_PERCENT = 0.15
    DEFAULT_REST_HEAL_PERCENT = 0.05
    PLAYER_TERRITORY_REST_HEAL_PERCENT = 0.15
    GOLD_PER_TURN = 100.0
    BUILDING_GOLD_COST_MULTIPLIER = 100.0
    LEGIONS_GOLD_MINE_GOLD_PER_TURN = 50.0
    BASE_INFERNAL_MANA_PER_TURN = 25.0
    LEGIONS_MANA_SOURCE_MANA_PER_TURN = 50.0
    MANA_KIND_ORDER = ("infernal", "life", "death", "runes", "elves")
    MANA_ATTR_BY_KIND = {
        "infernal": "infernal_mana",
        "life": "life_mana",
        "death": "death_mana",
        "runes": "runes_mana",
        "elves": "elven_mana",
    }
    MANA_LABEL_BY_KIND = {
        "infernal": "Мана преисподней",
        "life": "Мана жизни",
        "death": "Мана смерти",
        "runes": "Мана рун",
        "elves": "Мана эльфов",
    }
    SPELL_COST_MANA_KIND_BY_SUFFIX = {
        "hell": "infernal",
        "life": "life",
        "death": "death",
        "rune": "runes",
        "nature": "elves",
    }
    MAGIC_TOWER_BUILDING_NAME = "Башня магии"
    TEMPLE_BUILDING_NAME = "Храм"
    GRID_BOTTLE_ACTION_START = 9
    GRID_BOTTLE_POSITIONS = list(range(7, 13))  # 6 позиций BLUE
    HEAL_BOTTLE_AMOUNT = 100.0
    MAX_HEAL_BOTTLES = 3
    HEALING_BOTTLE_AMOUNT = 50.0
    MAX_HEALING_BOTTLES = 3
    HEALING_OINTMENT_AMOUNT = 200.0
    GRID_REVIVE_ACTION_START = GRID_BOTTLE_ACTION_START + len(GRID_BOTTLE_POSITIONS)  # 15
    REVIVE_BOTTLE_POSITIONS = GRID_BOTTLE_POSITIONS
    MAX_REVIVE_BOTTLES = 3
    GRID_INVULNERABILITY_ACTION_START = GRID_REVIVE_ACTION_START + len(REVIVE_BOTTLE_POSITIONS)  # 21
    INVULNERABILITY_POTION_POSITIONS = GRID_BOTTLE_POSITIONS
    INVULNERABILITY_POTION_ITEM_NAME = "Potion of Invulnerability"
    INVULNERABILITY_POTION_ARMOR_BONUS = 50
    GRID_STRENGTH_ACTION_START = (
        GRID_INVULNERABILITY_ACTION_START + len(INVULNERABILITY_POTION_POSITIONS)
    )  # 27
    STRENGTH_POTION_POSITIONS = GRID_BOTTLE_POSITIONS
    STRENGTH_POTION_ITEM_NAME = "Potion of Strength"
    STRENGTH_POTION_DAMAGE_MULTIPLIER = 1.30
    CASTLE_POS = DEFAULT_HERO_GRID_POSITION
    CASTLE_HEAL_TILE_COUNT = HEAL_TILE_COUNT
    SETTLEMENT_UPGRADE_GOLD_COST_BY_LEVEL = {
        1: 150.0,
        2: 250.0,
        3: 500.0,
        4: 750.0,
    }
    MAX_SETTLEMENT_LEVEL = 5
    SETTLEMENT_REST_HEAL_BONUS_BY_LEVEL = {
        1: 10.0,
        2: 15.0,
        3: 20.0,
        4: 25.0,
        5: 30.0,
    }
    LEGIONS_TERRITORY_EXPANSION_PER_TURN = 10
    EMPIRE_TERRITORY_SOURCE_ENEMY_ID = 75
    EMPIRE_TERRITORY_SOURCE_TILE = (26, 10)
    EMPIRE_TERRITORY_EXPANSION_PER_TURN = LEGIONS_TERRITORY_EXPANSION_PER_TURN
    GRID_CASTLE_HEAL_ACTION_START = GRID_STRENGTH_ACTION_START + len(STRENGTH_POTION_POSITIONS)  # 33
    CASTLE_HEAL_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_CASTLE_REVIVE_ACTION_START = GRID_CASTLE_HEAL_ACTION_START + len(CASTLE_HEAL_POSITIONS)  # 39
    CASTLE_REVIVE_POSITIONS = REVIVE_BOTTLE_POSITIONS
    HIRE_FRONT_POSITIONS = (7, 9)
    HIRE_BACK_POSITIONS = (10, 11, 12)
    GRID_HIRE_ACTION_START = GRID_CASTLE_REVIVE_ACTION_START + len(CASTLE_REVIVE_POSITIONS)  # 45
    GRID_MERCENARY_HIRE_ACTION_START = GRID_HIRE_ACTION_START
    GRID_TRAINER_ACTION_START = GRID_MERCENARY_HIRE_ACTION_START
    TRAINER_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_MERCHANT_BUY_ACTION_START = GRID_TRAINER_ACTION_START
    GRID_SPELL_SHOP_BUY_ACTION_START = GRID_MERCHANT_BUY_ACTION_START + 10
    ENERGY_ELIXIR_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_ENERGY_ACTION_START = GRID_SPELL_SHOP_BUY_ACTION_START + 2
    HASTE_ELIXIR_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_HASTE_ACTION_START = GRID_ENERGY_ACTION_START + len(ENERGY_ELIXIR_POSITIONS)
    FIRE_WARD_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_FIRE_WARD_ACTION_START = GRID_HASTE_ACTION_START + len(HASTE_ELIXIR_POSITIONS)
    EARTH_WARD_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_EARTH_WARD_ACTION_START = GRID_FIRE_WARD_ACTION_START + len(FIRE_WARD_POSITIONS)
    WATER_WARD_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_WATER_WARD_ACTION_START = GRID_EARTH_WARD_ACTION_START + len(EARTH_WARD_POSITIONS)
    AIR_WARD_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_AIR_WARD_ACTION_START = GRID_WATER_WARD_ACTION_START + len(WATER_WARD_POSITIONS)
    TITAN_ELIXIR_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_TITAN_ELIXIR_ACTION_START = GRID_AIR_WARD_ACTION_START + len(AIR_WARD_POSITIONS)
    SUPREME_ELIXIR_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_SUPREME_ELIXIR_ACTION_START = (
        GRID_TITAN_ELIXIR_ACTION_START + len(TITAN_ELIXIR_POSITIONS)
    )
    FINAL_OBJECTIVE_CITIES = {
        "Порт Полонис": (34, 68),
        "Соругирилла": (35, 69),
    }
    BASE_MERCHANT_SITE_DATA = {
        "Лавка Алара": {
            "anchor": (21, 34),
            "interaction_tiles": (
                (24, 35),
                (24, 36),
                (22, 37),
                (23, 37),
                (24, 37),
            ),
        },
        "Лавка Тралара": {
            "anchor": (36, 1),
            "interaction_tiles": (
                (39, 2),
                (39, 3),
                (37, 4),
                (38, 4),
                (39, 4),
            ),
        },
    }
    BASE_SPELL_SHOP_SITE_DATA = {
        "Лавка заклинаний Маллавиена": {
            "site_id": "S001SI0009",
            "anchor": (42, 2),
            "interaction_tiles": (
                (45, 3),
                (45, 4),
                (43, 5),
                (44, 5),
                (45, 5),
            ),
        },
    }
    BASE_MERCENARY_SITE_DATA = {
        "Лагерь северных варваров": {
            "site_id": "S001SI0002",
            "anchor": (9, 42),
            "interaction_tiles": (
                (12, 43),
                (12, 44),
                (10, 45),
                (11, 45),
                (12, 45),
            ),
            "roster": (
                {
                    "slot": 0,
                    "unit_id": "g000uu5040",
                    "unit_name": "Варвар",
                    "gold": 850.0,
                    "stock": 1,
                },
            ),
        },
        "Пустой лагерь наёмников": {
            "site_id": "S001SI0007",
            "anchor": (15, 1),
            "interaction_tiles": (
                (18, 2),
                (18, 3),
                (16, 4),
                (17, 4),
                (18, 4),
            ),
            "roster": (
                {
                    "slot": 0,
                    "unit_id": "g000uu5017",
                    "unit_name": "Гоблин",
                    "gold": 50.0,
                    "stock": 1,
                },
                {
                    "slot": 1,
                    "unit_id": "g000uu5018",
                    "unit_name": "Гоблин лучник",
                    "gold": 50.0,
                    "stock": 1,
                },
            ),
        },
    }
    BASE_TRAINER_SITE_DATA = {
        "Тренировочный лагерь": {
            "site_id": "S001SI0006",
            "anchor": (38, 30),
            "interaction_tiles": (
                (41, 31),
                (41, 32),
                (39, 33),
                (40, 33),
                (41, 33),
            ),
        },
    }
    MERCENARY_FRONTLINE_UNIT_TYPES = frozenset(
        {
            "Warrior",
            "Demon",
            "Lord",
            "Bone Lord",
            "Dregazul",
            "Spider",
            "Centaur Savage",
            "Aleman",
            "Uter",
            "Abyss Devil",
            "Ismir son",
            "Gumtic",
            "Uter Demon",
        }
    )
    MERCENARY_BACKLINE_UNIT_TYPES = frozenset(
        {
            "Archer",
            "Mage",
            "Summoner",
            "Cliric",
            "Profit",
            "Patriach",
            "Death",
            "Ghost",
            "Shadow",
            "Vampire",
            "Highvampire",
            "Wight",
            "Incub",
            "Succub",
            "Doppelganger",
            "Witch",
            "Baroness",
            "Sentry",
            "Watcher",
            "Hermit",
            "Teurg",
            "Shamanka",
            "Drulliaan",
        }
    )
    FINAL_OBJECTIVE_CITY_REWARD_MULTIPLIER = 5.0
    CHEST_PICKUP_RADIUS = 1
    BONUS_SMALL_HEAL_SOURCE_ITEM = "Potion of Healing"
    BONUS_SMALL_HEAL_ITEM_NAME = "Банка исцеления (+50 HP)"
    BONUS_LARGE_HEAL_ITEM_NAME = "Бутыль лечения (+100 HP)"
    HEALING_OINTMENT_ITEM_NAME = "Целебная мазь"
    BONUS_REVIVE_SOURCE_ITEM = "Life Potion"
    BONUS_REVIVE_ITEM_NAME = "Зелье воскрешения (+1)"
    BATTLE_EQUIP_SLOTS = 2
    BATTLE_EQUIPPABLE_ITEM_NAMES = (
        BONUS_SMALL_HEAL_ITEM_NAME,
        BONUS_LARGE_HEAL_ITEM_NAME,
        HEALING_OINTMENT_ITEM_NAME,
        BONUS_REVIVE_ITEM_NAME,
    )
    # Map-targeted offensive spells cast on the nearest enemy stack.
    # Keep existing Legion damage-spell order stable for trained policies, then append debuffs.
    LEGION_DAMAGE_SPELL_ACTION_SPECS = (
        {"id": "lod_d2_s003", "kind": "damage", "damage": 15.0, "damage_type": "Fire"},
        {"id": "lod_d2_s004", "kind": "damage", "damage": 15.0, "damage_type": "Mind"},
        {"id": "lod_d2_s008", "kind": "damage", "damage": 30.0, "damage_type": "Fire"},
        {"id": "lod_d2_s014", "kind": "damage", "damage": 50.0, "damage_type": "Mind"},
        {"id": "lod_d2_s017", "kind": "damage", "damage": 75.0, "damage_type": "Fire"},
        {"id": "lod_d2_s018", "kind": "damage", "damage": 75.0, "damage_type": "Earth"},
        {"id": "lod_d2_s019", "kind": "damage", "damage": 60.0, "damage_type": "Fire"},
        {"id": "lod_d2_s023", "kind": "damage", "damage": 125.0, "damage_type": "Fire"},
        {"id": "lod_d2_s005", "kind": "debuff", "debuff_type": "armor", "armor_delta": -50},
        {"id": "lod_d2_s009", "kind": "debuff", "debuff_type": "damage", "damage_multiplier": 0.85},
        {"id": "lod_d2_s010", "kind": "debuff", "debuff_type": "initiative", "initiative_multiplier": 0.85},
        {"id": "lod_d2_s015", "kind": "debuff", "debuff_type": "accuracy", "accuracy_multiplier": 0.75},
        {"id": "lod_d2_s016", "kind": "debuff", "debuff_type": "accuracy", "accuracy_multiplier": 0.67},
        {"id": "lod_d2_s020", "kind": "debuff", "debuff_type": "initiative", "initiative_multiplier": 0.67},
        {
            "id": "lod_d2_s001",
            "kind": "summon_battle",
            "summon_unit_name": "Адская гончая",
            "targets_untargetable_stacks": True,
        },
        {
            "id": "lod_d2_s006",
            "kind": "summon_battle",
            "summon_unit_name": "Белиарх",
            "targets_untargetable_stacks": True,
        },
        {
            "id": "lod_d2_s021",
            "kind": "summon_battle",
            "summon_unit_name": "Мститель",
            "targets_untargetable_stacks": True,
        },
    )
    # Map-targeted Undead Horde spells cast on the nearest enemy stack.
    # AoE damage spells are temporarily mapped to the nearest enemy stack only.
    UNDEAD_DAMAGE_SPELL_ACTION_SPECS = (
        {
            "id": "und_d2_s001",
            "kind": "summon_battle",
            "summon_unit_name": "Скелет",
            "targets_untargetable_stacks": True,
        },
        {"id": "und_d2_s002", "kind": "damage", "damage": 15.0, "damage_type": "Death"},
        {"id": "und_d2_s003", "kind": "damage", "damage": 15.0, "damage_type": "Water"},
        {"id": "und_d2_s004", "kind": "debuff", "debuff_type": "accuracy", "accuracy_multiplier": 0.90},
        {"id": "und_d2_s005", "kind": "debuff", "debuff_type": "armor", "armor_delta": -50},
        {
            "id": "und_d2_s006",
            "kind": "summon_battle",
            "summon_unit_name": "Темный энт",
            "targets_untargetable_stacks": True,
        },
        {"id": "und_d2_s007", "kind": "damage", "damage": 30.0, "damage_type": "Death"},
        {"id": "und_d2_s009", "kind": "debuff", "debuff_type": "damage", "damage_multiplier": 0.85},
        {"id": "und_d2_s010", "kind": "damage", "damage": 30.0, "damage_type": "Earth"},
        {
            "id": "und_d2_s011",
            "kind": "summon_battle",
            "summon_unit_name": "Кошмар",
            "targets_untargetable_stacks": True,
        },
        {"id": "und_d2_s012", "kind": "damage", "damage": 50.0, "damage_type": "Death"},
        {"id": "und_d2_s013", "kind": "debuff", "debuff_type": "accuracy", "accuracy_multiplier": 0.80},
        {"id": "und_d2_s015", "kind": "damage", "damage": 40.0, "damage_type": "Earth"},
        {"id": "und_d2_s016", "kind": "debuff", "debuff_type": "initiative", "initiative_multiplier": 0.67},
        {"id": "und_d2_s017", "kind": "debuff", "debuff_type": "damage", "damage_multiplier": 0.67},
        {"id": "und_d2_s018", "kind": "damage", "damage": 75.0, "damage_type": "Fire"},
        {
            "id": "und_d2_s021",
            "kind": "summon_battle",
            "summon_unit_name": "Танатос",
            "targets_untargetable_stacks": True,
        },
        {"id": "und_d2_s023", "kind": "damage", "damage": 125.0, "damage_type": "Death"},
        {"id": "und_d2_s024", "kind": "damage", "damage": 80.0, "damage_type": "Death"},
    )
    EMPIRE_DAMAGE_SPELL_ACTION_SPECS = (
        {"id": "emp_d2_s004", "kind": "damage", "damage": 15.0, "damage_type": "Air"},
        {
            "id": "emp_d2_s008",
            "kind": "summon_battle",
            "summon_unit_name": "Оживший доспех",
            "targets_untargetable_stacks": True,
        },
        {"id": "emp_d2_s014", "kind": "damage", "damage": 50.0, "damage_type": "Air"},
        {
            "id": "emp_d2_s016",
            "kind": "summon_battle",
            "summon_unit_name": "Голем",
            "targets_untargetable_stacks": True,
        },
        {"id": "emp_d2_s015", "kind": "damage", "damage": 40.0, "damage_type": "Air"},
        {"id": "emp_d2_s022", "kind": "damage", "damage": 125.0, "damage_type": "Air"},
    )
    EMPIRE_SUPPORT_SPELL_ACTION_SPECS = (
        {"id": "emp_d2_s001", "kind": "ward", "buff_type": "resistance", "resistance_type": "Air"},
        {"id": "emp_d2_s002", "kind": "buff", "buff_type": "initiative", "initiative_multiplier": 1.10},
        {"id": "emp_d2_s003", "kind": "buff", "buff_type": "damage", "damage_multiplier": 1.10},
        {"id": "emp_d2_s005", "kind": "ward", "buff_type": "resistance", "resistance_type": "Water"},
        {"id": "emp_d2_s006", "kind": "moves", "moves_restore_fraction": 0.50},
        {"id": "emp_d2_s007", "kind": "heal", "heal_amount": 30.0},
        {"id": "emp_d2_s010", "kind": "ward", "buff_type": "resistance", "resistance_type": "Earth"},
        {"id": "emp_d2_s011", "kind": "ward", "buff_type": "resistance", "resistance_type": "Mind"},
        {"id": "emp_d2_s012", "kind": "buff", "buff_type": "armor", "armor_delta": 20},
        {"id": "emp_d2_s013", "kind": "buff", "buff_type": "accuracy", "accuracy_multiplier": 1.20},
        {"id": "emp_d2_s017", "kind": "ward", "buff_type": "resistance", "resistance_type": "Fire"},
        {"id": "emp_d2_s018", "kind": "buff", "buff_type": "damage", "damage_multiplier": 1.33},
        {"id": "emp_d2_s019", "kind": "heal", "heal_amount": 60.0},
        {"id": "emp_d2_s021", "kind": "ward", "buff_type": "resistance", "resistance_type": "Death"},
        {"id": "emp_d2_s023", "kind": "heal", "heal_amount": 150.0},
        {"id": "emp_d2_s024", "kind": "heal", "heal_amount": 80.0},
    )
    MOUNTAIN_SUPPORT_SPELL_ACTION_SPECS = (
        {"id": "mcl_d2_s001", "kind": "buff", "buff_type": "armor", "armor_delta": 10},
        {"id": "mcl_d2_s003", "kind": "buff", "buff_type": "damage", "damage_multiplier": 1.10},
        {"id": "mcl_d2_s006", "kind": "buff", "buff_type": "initiative", "initiative_multiplier": 1.15},
        {"id": "mcl_d2_s009", "kind": "heal", "heal_amount": 30.0},
        {"id": "mcl_d2_s012", "kind": "moves", "moves_restore_fraction": 1.00},
        {"id": "mcl_d2_s014", "kind": "buff", "buff_type": "accuracy", "accuracy_multiplier": 1.25},
        {"id": "mcl_d2_s015", "kind": "buff", "buff_type": "damage", "damage_multiplier": 1.20},
        {"id": "mcl_d2_s016", "kind": "buff", "buff_type": "armor", "armor_delta": 33},
        {"id": "mcl_d2_s017", "kind": "buff", "buff_type": "accuracy", "accuracy_multiplier": 1.33},
        {"id": "mcl_d2_s020", "kind": "moves", "moves_restore_fraction": 1.00},
        {"id": "mcl_d2_s023", "kind": "buff", "buff_type": "damage", "damage_multiplier": 1.50},
        {"id": "mcl_d2_s024", "kind": "buff", "buff_type": "armor", "armor_delta": 33},
    )
    MOUNTAIN_DAMAGE_SPELL_ACTION_SPECS = (
        {
            "id": "mcl_d2_s005",
            "kind": "summon_battle",
            "summon_unit_name": "Рух",
            "targets_untargetable_stacks": True,
        },
        {"id": "mcl_d2_s004", "kind": "damage", "damage": 15.0, "damage_type": "Water"},
        {"id": "mcl_d2_s008", "kind": "damage", "damage": 30.0, "damage_type": "Water"},
        {
            "id": "mcl_d2_s011",
            "kind": "summon_battle",
            "summon_unit_name": "Валькирия",
            "targets_untargetable_stacks": True,
        },
        {"id": "mcl_d2_s013", "kind": "damage", "damage": 60.0, "damage_type": "Earth"},
        {"id": "mcl_d2_s018", "kind": "damage", "damage": 75.0, "damage_type": "Water"},
        {"id": "mcl_d2_s019", "kind": "damage", "damage": 60.0, "damage_type": "Water"},
        {
            "id": "mcl_d2_s021",
            "kind": "summon_battle",
            "summon_unit_name": "Каменный предок",
            "targets_untargetable_stacks": True,
        },
    )
    ELVES_DAMAGE_SPELL_ACTION_SPECS = (
        {"id": "elf_d2_s001", "kind": "damage", "damage": 15.0, "damage_type": "Earth"},
        {
            "id": "elf_d2_s002",
            "kind": "summon_battle",
            "summon_unit_name": "Малый энт",
            "targets_untargetable_stacks": True,
        },
        {
            "id": "elf_d2_s007",
            "kind": "summon_battle",
            "summon_unit_name": "Энт",
            "targets_untargetable_stacks": True,
        },
        {"id": "elf_d2_s008", "kind": "damage", "damage": 30.0, "damage_type": "Fire"},
        {"id": "elf_d2_s009", "kind": "damage", "damage": 30.0, "damage_type": "Water"},
        {"id": "elf_d2_s010", "kind": "debuff", "debuff_type": "initiative", "initiative_multiplier": 0.85},
        {
            "id": "elf_d2_s012",
            "kind": "summon_battle",
            "summon_unit_name": "Великий энт",
            "targets_untargetable_stacks": True,
        },
        {"id": "elf_d2_s013", "kind": "damage", "damage": 50.0, "damage_type": "Mind"},
        {"id": "elf_d2_s015", "kind": "debuff", "debuff_type": "accuracy", "accuracy_multiplier": 0.80},
        {"id": "elf_d2_s016", "kind": "damage", "damage": 60.0, "damage_type": "Earth"},
        {"id": "elf_d2_s017", "kind": "damage", "damage": 75.0, "damage_type": "Air"},
        {"id": "elf_d2_s018", "kind": "debuff", "debuff_type": "armor", "armor_delta": -80},
        {"id": "elf_d2_s019", "kind": "debuff", "debuff_type": "damage", "damage_multiplier": 0.67},
        {
            "id": "elf_d2_s021",
            "kind": "summon_battle",
            "summon_unit_name": "Буйный энт",
            "targets_untargetable_stacks": True,
        },
        {"id": "elf_d2_s022", "kind": "damage", "damage": 125.0, "damage_type": "Earth"},
    )
    ELVES_SUPPORT_SPELL_ACTION_SPECS = (
        {"id": "elf_d2_s004", "kind": "moves", "moves_restore_fraction": 0.30},
        {"id": "elf_d2_s005", "kind": "buff", "buff_type": "armor", "armor_delta": 10},
        {"id": "elf_d2_s006", "kind": "heal", "heal_amount": 30.0},
        {"id": "elf_d2_s014", "kind": "heal", "heal_amount": 50.0},
        {"id": "elf_d2_s020", "kind": "health_bonus", "buff_type": "health", "health_delta": 50},
    )
    UNDEAD_SUPPORT_SPELL_ACTION_SPECS = (
        {"id": "und_d2_s022", "kind": "ward", "buff_type": "resistance", "resistance_type": "Weapon"},
    )
    MAP_OFFENSIVE_SPELL_ACTION_SPECS_BY_CAPITAL = {
        1: EMPIRE_DAMAGE_SPELL_ACTION_SPECS,
        2: LEGION_DAMAGE_SPELL_ACTION_SPECS,
        3: MOUNTAIN_DAMAGE_SPELL_ACTION_SPECS,
        4: UNDEAD_DAMAGE_SPELL_ACTION_SPECS,
        5: ELVES_DAMAGE_SPELL_ACTION_SPECS,
    }
    MAP_SUPPORT_SPELL_ACTION_SPECS_BY_CAPITAL = {
        1: EMPIRE_SUPPORT_SPELL_ACTION_SPECS,
        3: MOUNTAIN_SUPPORT_SPELL_ACTION_SPECS,
        4: UNDEAD_SUPPORT_SPELL_ACTION_SPECS,
        5: ELVES_SUPPORT_SPELL_ACTION_SPECS,
    }
    ENERGY_ELIXIR_ITEM_NAME = "Эликсир энергии"
    FIRE_WARD_ITEM_NAME = "Эликсир защиты от магии Огня"
    EARTH_WARD_ITEM_NAME = "Эликсир защиты от магии Земли"
    WATER_WARD_ITEM_NAME = "Эликсир защиты от магии Воды"
    AIR_WARD_ITEM_NAME = "Эликсир защиты от магии Воздуха"
    HASTE_ELIXIR_ITEM_NAME = "Эликсир быстроты"
    RUIN_LIFE_ELIXIR_ITEM_NAME = "Эликсир Жизни"
    RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME = "Эликсир неуязвимости"
    TITAN_ELIXIR_ITEM_NAME = "Эликсир Силы титана"
    SUPREME_ELIXIR_ITEM_NAME = "Эликсир Всевышнего"
    RING_OF_AGES_ITEM_NAME = "Кольцо веков (Артефакт)"
    RING_OF_AGES_ENGLISH_ITEM_NAME = "Ring of the Ages (Artifact)"
    DWARVEN_BRACER_ITEM_NAME = "Dwarven Bracer (Artifact)"
    UNHOLY_CHALICE_ARTIFACT_ITEM_NAME = "Unholy Chalice (Artifact)"
    RING_OF_STRENGTH_ITEM_NAME = "Ring of Strength (Artifact)"
    RUNIC_BLADE_ITEM_NAME = "Runic Blade (Artifact)"
    MJOLNIRS_CROWN_ITEM_NAME = "Mjolnir's Crown (Artifact)"
    BETHREZENS_CLAW_ITEM_NAME = "Bethrezen's Claw (Artifact)"
    RUNESTONE_ARTIFACT_ITEM_NAME = "Runestone (Artifact)"
    HOLY_CHALICE_ARTIFACT_ITEM_NAME = "Holy Chalice (Artifact)"
    SKULL_BRACERS_ITEM_NAME = "Skull Bracers (Artifact)"
    HORN_OF_AWARENESS_ITEM_NAME = "Horn of Awareness (Artifact)"
    ETCHED_CIRCLET_ITEM_NAME = "Etched Circlet (Artifact)"
    SOUL_CRYSTAL_ARTIFACT_ITEM_NAME = "Soul Crystal (Artifact)"
    HORN_OF_INCUBUS_ARTIFACT_ITEM_NAME = "Horn of Incubus (Artifact)"
    UNHOLY_DAGGER_ARTIFACT_ITEM_NAME = "Unholy Dagger (Artifact)"
    THANATOS_BLADE_ARTIFACT_ITEM_NAME = "Thanatos Blade (Artifact)"
    SKULL_OF_THANATOS_ARTIFACT_ITEM_NAME = "Skull of Thanatos (Artifact)"
    HAGS_RING_ARTIFACT_ITEM_NAME = "Hag's Ring (Artifact)"
    ARTIFACT_EQUIP_SLOTS = 2
    ENERGY_ELIXIR_DAMAGE_MULTIPLIER = 1.50
    HASTE_ELIXIR_INITIATIVE_MULTIPLIER = 1.60
    TITAN_ELIXIR_DAMAGE_MULTIPLIER = 1.10
    SUPREME_ELIXIR_HEALTH_MULTIPLIER = 1.15
    RING_OF_AGES_DAMAGE_MULTIPLIER = 1.40
    RING_OF_AGES_INITIATIVE_MULTIPLIER = 1.25
    DWARVEN_BRACER_DAMAGE_MULTIPLIER = 1.10
    UNHOLY_CHALICE_ARTIFACT_DAMAGE_MULTIPLIER = 1.15
    RING_OF_STRENGTH_DAMAGE_MULTIPLIER = 1.20
    RUNIC_BLADE_DAMAGE_MULTIPLIER = 1.25
    MJOLNIRS_CROWN_DAMAGE_MULTIPLIER = 1.35
    BETHREZENS_CLAW_DAMAGE_MULTIPLIER = 1.40
    BETHREZENS_CLAW_INITIATIVE_MULTIPLIER = 1.50
    RUNESTONE_ARTIFACT_ARMOR_BONUS = 10
    HOLY_CHALICE_ARTIFACT_ARMOR_BONUS = 15
    SKULL_BRACERS_ARMOR_BONUS = 20
    HORN_OF_AWARENESS_ARMOR_BONUS = 25
    ETCHED_CIRCLET_ARMOR_BONUS = 35
    MERCHANT_BUY_ITEMS = (
        {"name": RUIN_LIFE_ELIXIR_ITEM_NAME, "price": 400.0, "stock": 10, "grant": "revive_bonus"},
        {"name": "Эликсир исцеления", "price": 150.0, "stock": 10, "grant": "small_heal_bonus"},
        {"name": "Эликсир восстановления", "price": 300.0, "stock": 10, "grant": "large_heal_bonus"},
        {"name": "Целебная мазь", "price": 600.0, "stock": 10, "grant": "inventory"},
        {"name": "Эликсир энергии", "price": 200.0, "stock": 1, "grant": "inventory"},
        {"name": "Эликсир защиты от магии Огня", "price": 400.0, "stock": 1, "grant": "inventory"},
        {"name": "Эликсир защиты от магии Земли", "price": 400.0, "stock": 1, "grant": "inventory"},
        {"name": "Эликсир защиты от магии Воды", "price": 400.0, "stock": 1, "grant": "inventory"},
        {"name": "Эликсир защиты от магии Воздуха", "price": 400.0, "stock": 1, "grant": "inventory"},
        {"name": "Эликсир быстроты", "price": 600.0, "stock": 1, "grant": "inventory"},
    )
    SPELL_SHOP_BUY_SPELLS = (
        {"name": "Армагеддон", "spell_id": "emp_d2_s022", "price": 1000.0, "stock": 1},
        {"name": "Вызов Мстителя", "spell_id": "lod_d2_s021", "price": 1000.0, "stock": 1},
    )
    MAX_SPELL_SHOP_CAST_ACTIONS = 5
    SCROLL_MAGE_UNLOCK_GOLD_COST = 500.0
    MAX_SCROLL_CAST_ACTIONS = 5
    GRID_BUILD_ACTION_START = (
        GRID_SUPREME_ELIXIR_ACTION_START + len(SUPREME_ELIXIR_POSITIONS)
    )
    CHEST_ITEM_GOLD_VALUES = {
        "Potion of Healing": 150,
        "Life Potion": 400,
        "Potion of Strength": 450,
        "Potion of Invulnerability": 700,
        "Banner of Might": 1000,
        "Banner of War": 5000,
        "Tome of Arcanum": 900,
        "Vampire Orb": 800,
        "Orb of Life": 800,
        "Lich Orb": 800,
        "Mind ward scroll": 600,
        "Bronze Ring (Valuable)": 250,
        "Ruby (Valuable)": 1250,
        "Diamond (Valuable)": 1750,
        BONUS_SMALL_HEAL_ITEM_NAME: 150,
        BONUS_LARGE_HEAL_ITEM_NAME: 300,
        BONUS_REVIVE_ITEM_NAME: 400,
        RUIN_LIFE_ELIXIR_ITEM_NAME: 400,
        RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME: 700,
        TITAN_ELIXIR_ITEM_NAME: 800,
        SUPREME_ELIXIR_ITEM_NAME: 800,
        RING_OF_AGES_ITEM_NAME: 3750,
        RING_OF_AGES_ENGLISH_ITEM_NAME: 3750,
        SOUL_CRYSTAL_ARTIFACT_ITEM_NAME: 2000,
        HORN_OF_INCUBUS_ARTIFACT_ITEM_NAME: 2000,
        UNHOLY_DAGGER_ARTIFACT_ITEM_NAME: 1500,
        THANATOS_BLADE_ARTIFACT_ITEM_NAME: 1500,
        SKULL_OF_THANATOS_ARTIFACT_ITEM_NAME: 3750,
        HAGS_RING_ARTIFACT_ITEM_NAME: 2500,
    }
    AUTO_CONSUMED_CHEST_ITEMS = (
        BONUS_SMALL_HEAL_SOURCE_ITEM,
        BONUS_REVIVE_SOURCE_ITEM,
    )
    ENEMY_REWARD_MULTIPLIERS = {}
    RUIN_REWARD_BY_ENEMY_ID: Dict[int, Dict[str, object]] = {
        70: {
            "title": "Замок орка",
            "ruin_pos": (15, 9),
            "marker_pos": (17, 11),
            "item": TITAN_ELIXIR_ITEM_NAME,
            "gold": 200,
        },
        71: {
            "title": "Храм Мараши",
            "ruin_pos": (3, 20),
            "marker_pos": (5, 22),
            "item": RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME,
            "gold": 300,
        },
        72: {
            "title": "Руины (30, 22)",
            "ruin_pos": (30, 22),
            "marker_pos": (32, 24),
            "item": RUIN_LIFE_ELIXIR_ITEM_NAME,
            "gold": 300,
        },
        73: {
            "title": "Руины Бритона",
            "ruin_pos": (19, 29),
            "marker_pos": (21, 31),
            "item": SUPREME_ELIXIR_ITEM_NAME,
            "gold": 150,
        },
        74: {
            "title": "Замок Ху'Ларша",
            "ruin_pos": (43, 34),
            "marker_pos": (45, 36),
            "item": RING_OF_AGES_ITEM_NAME,
            "gold": 400,
        },
    }
    RUIN_REWARD_ITEM_NAMES = frozenset(
        {
            str(reward_data.get("item", "") or "")
            for reward_data in RUIN_REWARD_BY_ENEMY_ID.values()
            if str(reward_data.get("item", "") or "")
        }
    )

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        reward_engage_battle: float = 0.2,  # Бонус за вход в бой с живым врагом.
        reward_defeat_enemy: float = 1.0,   # Награда за победу в бою против каждого отряда
        reward_ruin_clear_bonus: Optional[float] = None,  # Небольшой бонус сверху за зачистку руин
        reward_all_enemies: float = 10.0,   # База: 1-й целевой город = x1, 2-й = x5
        reward_loss: float = -5.0,          # Штраф за поражение кампании
        reward_timeout: float = -3.0,       # Штраф за таймаут (лимит шагов)
        reward_turn_penalty: float = 0.02,  # Штраф за завершение хода (REST)
        reward_needaunit_turn_penalty: float = 0.05,  # Штраф за ход, пока герою нужен найм.
        reward_repeat_position_penalty: float = 0.015,  # Штраф за повторное посещение клетки.
        reward_repeat_position_penalty_cap: float = 0.08,  # Кеп на повторный штраф за 1 шаг.
        reward_backtrack_penalty: float = 0.03,  # Штраф за микро-цикл A->B->A.
        reward_no_movement_penalty: float = 0.02,  # Штраф если движение не сменило позицию.
        exp_reward_norm_k: float = 100.0,   # Нормализация опыта: norm=exp/(exp+k)
        reward_exp_weight: float = 0.2,     # Вес exp-компоненты
        reward_survival_alive_weight: float = 0.5,  # Вес доли выживших BLUE
        reward_survival_hp_weight: float = 0.5,     # Вес средней доли HP BLUE
        reward_unit_upgrade: float = 1.0,
        battle_reward_scale: float = 0.25,
        battle_reward_win: float = 2.0,
        battle_reward_loss: float = -1.0,
        battle_reward_step: float = 0.01,
        reward_battle_item_equip: float = 0.02,
        reward_battle_item_use: float = 0.05,
        reward_combat_potion_battle_participation: float = 0.05,
        reward_sell_junk_item: float = 0.05,
        reward_spell_learn: float = 2.0,
        reward_spell_cast: float = 0.005,
        reward_summon_hero_battle_engage: float = 0.05,
        reward_castle_heal_per_hp: float = 0.01,
        reward_castle_revive: float = 0.5,
        persist_blue_hp: bool = True,
        log_enabled: bool = False,
        max_grid_steps: int = 1800,
        realcapital: int = 1,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.reward_engage_battle = reward_engage_battle  # Награда за участие в битве
        self.reward_defeat_enemy = reward_defeat_enemy    # Награда за победу в битве
        if reward_ruin_clear_bonus is None:
            reward_ruin_clear_bonus = max(0.1, float(reward_defeat_enemy) * 0.25)
        self.reward_ruin_clear_bonus = max(0.0, float(reward_ruin_clear_bonus))
        self.reward_all_enemies = reward_all_enemies      # База награды за захват целевого города
        self.reward_loss = reward_loss
        self.reward_timeout = reward_timeout
        self.reward_turn_penalty = reward_turn_penalty
        self.reward_needaunit_turn_penalty = max(0.0, float(reward_needaunit_turn_penalty))
        self.reward_repeat_position_penalty = max(0.0, float(reward_repeat_position_penalty))
        self.reward_repeat_position_penalty_cap = max(0.0, float(reward_repeat_position_penalty_cap))
        self.reward_backtrack_penalty = max(0.0, float(reward_backtrack_penalty))
        self.reward_no_movement_penalty = max(0.0, float(reward_no_movement_penalty))
        self.exp_reward_norm_k = max(1.0, float(exp_reward_norm_k))
        self.reward_exp_weight = max(0.0, float(reward_exp_weight))
        self.reward_survival_alive_weight = max(0.0, float(reward_survival_alive_weight))
        self.reward_survival_hp_weight = max(0.0, float(reward_survival_hp_weight))
        self.reward_unit_upgrade = max(0.0, float(reward_unit_upgrade))
        self.battle_reward_scale = max(0.0, float(battle_reward_scale))
        self.battle_reward_win = float(battle_reward_win)
        self.battle_reward_loss = float(battle_reward_loss)
        self.battle_reward_step = float(battle_reward_step)
        self.reward_battle_item_equip = max(0.0, float(reward_battle_item_equip))
        self.reward_battle_item_use = max(0.0, float(reward_battle_item_use))
        self.reward_combat_potion_battle_participation = max(
            0.0,
            float(reward_combat_potion_battle_participation),
        )
        self.reward_sell_junk_item = max(0.0, float(reward_sell_junk_item))
        self.reward_spell_learn = max(0.0, float(reward_spell_learn))
        self.reward_spell_cast = max(0.0, float(reward_spell_cast))
        self.reward_summon_hero_battle_engage = max(
            0.0,
            float(reward_summon_hero_battle_engage),
        )
        self.reward_castle_heal_per_hp = max(0.0, float(reward_castle_heal_per_hp))
        self.reward_castle_revive = max(0.0, float(reward_castle_revive))
        self.persist_blue_hp = persist_blue_hp
        self.log_enabled = log_enabled
        self.max_grid_steps = max(1, int(max_grid_steps))
        # Базовый лимит шагов внутри одного хода кампании.
        self.moves_per_turn = int(self.MOVES_PER_TURN)
        self.turn_norm_k = max(
            1.0,
            float(self.max_grid_steps) / float(self.GRID_STEPS_PER_TURN),
        )
        self.gold_norm_k = max(1.0, self.turn_norm_k * float(self.GOLD_PER_TURN))
        # Realcapital: 1 = empire, 2 = legions, 3 = mountain_clans, 4 = undead_hordes, 5 = elves
        self.Realcapital = self._normalize_realcapital(realcapital)
        # Typeoflord is currently fixed for the campaign layout/state.
        # Значения typeoflord для всех фракций трактуются так:
        #   1 = повелитель воинов
        #   2 = повелитель магов
        #   3 = повелитель воров
        self.typeoflord = 1
        self.Typeoflord = int(self.typeoflord)
        self._reset_buildings_state()
        self._reset_spells_state()

        enemy_positions = scale_enemy_positions(self.grid_size)
        self.castle_heal_tiles: Tuple[Tuple[int, int], ...] = resolve_heal_tiles(
            self.grid_size,
            enemy_positions,
            start_position=self.CASTLE_POS,
            heal_tile_count=self.CASTLE_HEAL_TILE_COUNT,
        )
        obstacle_tiles = resolve_obstacle_tiles(
            self.grid_size,
            enemy_positions,
            heal_tiles=self.castle_heal_tiles,
            start_position=self.CASTLE_POS,
        )
        chest_contents = resolve_chests(
            self.grid_size,
            enemy_positions,
            heal_tiles=self.castle_heal_tiles,
            obstacle_tiles=obstacle_tiles,
            start_position=self.CASTLE_POS,
        )
        mana_sources = resolve_mana_sources(self.grid_size)
        self._static_enemy_positions = dict(enemy_positions)
        self._static_castle_heal_tiles: Tuple[Tuple[int, int], ...] = tuple(self.castle_heal_tiles)
        self._static_obstacle_tiles: Tuple[Tuple[int, int], ...] = tuple(obstacle_tiles)
        self._static_chests: Dict[Tuple[int, int], Tuple[str, ...]] = dict(chest_contents)
        self._static_mana_sources: Dict[Tuple[int, int], Dict[str, object]] = {
            tuple(pos): dict(meta)
            for pos, meta in mana_sources.items()
        }
        self._static_legions_settlement_territory_sources: Tuple[Dict[str, object], ...] = (
            resolve_legions_settlement_territory_sources(
                self.grid_size,
                enemy_positions,
            )
        )
        self.legions_settlement_territory_source_by_name: Dict[str, Dict[str, object]] = {
            str(source_data.get("name", "") or ""): dict(source_data)
            for source_data in self._static_legions_settlement_territory_sources
        }
        self.legions_settlement_source_tile_by_name: Dict[str, Tuple[int, int]] = {
            str(source_data.get("name", "") or ""): tuple(source_data.get("source_tile", (0, 0)))
            for source_data in self._static_legions_settlement_territory_sources
        }
        self.legions_settlement_source_name_by_tile: Dict[Tuple[int, int], str] = {
            tuple(source_data.get("source_tile", ())): str(source_data.get("name", "") or "")
            for source_data in self._static_legions_settlement_territory_sources
            if len(tuple(source_data.get("source_tile", ()))) == 2
        }
        self.legions_settlement_territory_source_name_by_enemy_id: Dict[int, str] = {
            int(enemy_id): str(source_data.get("name", "") or "")
            for source_data in self._static_legions_settlement_territory_sources
            for enemy_id in tuple(source_data.get("required_enemy_ids", ()))
        }
        self._init_merchant_site_metadata()
        self._init_spell_shop_site_metadata()
        self._init_mercenary_site_metadata()
        self._init_trainer_site_metadata()

        # Подсреда grid: враги и препятствия автоматически масштабируются под размер карты.
        self.grid_env = GridWorldEnv(
            grid_size=grid_size,
            enemy_positions=self._static_enemy_positions,
            obstacle_positions=self._static_obstacle_tiles,
            start_position=self.CASTLE_POS,
            max_steps=max_grid_steps,
        )
        self.legions_territory_source_tile: Tuple[int, int] = tuple(
            int(coord) for coord in self.grid_env.start_position
        )
        base_gold_mine_tiles = _load_legions_territory_base_gold_mine_tiles()
        self.gold_mine_tiles: Tuple[Tuple[int, int], ...] = tuple(
            sorted(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_gold_mine_tiles)
            )
        ) if base_gold_mine_tiles else ()
        empire_source_tile = self._static_enemy_positions.get(
            int(self.EMPIRE_TERRITORY_SOURCE_ENEMY_ID),
            scale_static_tiles(self.grid_size, (tuple(self.EMPIRE_TERRITORY_SOURCE_TILE),))[0],
        )
        self.empire_territory_source_tile: Tuple[int, int] = tuple(
            int(coord) for coord in empire_source_tile
        )
        self.legions_territory_forbidden_tiles: Tuple[Tuple[int, int], ...] = (
            self._build_legions_territory_forbidden_tiles()
        )
        self.legions_territory_forbidden_tile_set: set[Tuple[int, int]] = set(
            self.legions_territory_forbidden_tiles
        )
        self.legions_territory_path_blocked_tiles: Tuple[Tuple[int, int], ...] = (
            self._build_legions_territory_path_blocked_tiles()
        )
        self.legions_territory_path_blocked_tile_set: set[Tuple[int, int]] = set(
            self.legions_territory_path_blocked_tiles
        )
        self.legions_territory_path_distances: Dict[Tuple[int, int], int] = (
            self._build_legions_territory_path_distances()
        )
        self.legions_territory_order: Tuple[Tuple[int, int], ...] = (
            self._build_legions_territory_order()
        )
        self.legions_settlement_territory_orders_by_name: Dict[str, Tuple[Tuple[int, int], ...]] = (
            self._build_legions_settlement_territory_orders_by_name()
        )
        self.legions_territory_tiles: Tuple[Tuple[int, int], ...] = ()
        self.legions_territory_tile_set: set[Tuple[int, int]] = set()
        self._reset_legions_settlement_territory_state()
        self.legions_captured_gold_mine_tiles: Tuple[Tuple[int, int], ...] = ()
        self.legions_captured_gold_mine_count: int = 0
        self.empire_territory_forbidden_tiles: Tuple[Tuple[int, int], ...] = (
            self._build_empire_territory_forbidden_tiles()
        )
        self.empire_territory_forbidden_tile_set: set[Tuple[int, int]] = set(
            self.empire_territory_forbidden_tiles
        )
        self.empire_territory_path_blocked_tiles: Tuple[Tuple[int, int], ...] = (
            self._build_empire_territory_path_blocked_tiles()
        )
        self.empire_territory_path_blocked_tile_set: set[Tuple[int, int]] = set(
            self.empire_territory_path_blocked_tiles
        )
        self.empire_territory_path_distances: Dict[Tuple[int, int], int] = (
            self._build_empire_territory_path_distances()
        )
        self.empire_territory_order: Tuple[Tuple[int, int], ...] = (
            self._build_empire_territory_order()
        )
        self.empire_territory_tiles: Tuple[Tuple[int, int], ...] = ()
        self.empire_territory_tile_set: set[Tuple[int, int]] = set()
        self._legions_territory_grid_obs_cache = np.zeros(0, dtype=np.float32)
        self._sync_grid_merchant_positions()
        self._sync_grid_spell_shop_positions()
        self._sync_grid_mercenary_positions()
        self._sync_grid_trainer_positions()
        self._ensure_map_layout_consistency()
        self.battle_env: Optional[BattleEnv] = None
        self.grid_obs_base_size = int(self.grid_env.observation_space.shape[0])
        self._init_enemy_obs_metadata()

        # Текущий режим
        self.mode = self.MODE_GRID
        self.current_enemy_id: Optional[int] = None
        self.battle_origin_pos: Optional[Tuple[int, int]] = None
        self.current_battle_context: Dict[str, object] = {}

        # Сохраняем состояние BLUE команды между боями
        self.blue_team_state: Optional[List[Dict]] = None
        self.heal_bottles_used: int = 0
        self.healing_bottles_used: int = 0
        self.revive_bottles_used: int = 0
        self.turns: int = 0
        self.gold: float = 0.0
        self.moves: int = self.moves_per_turn
        self.active_buildings = self._get_buildings_for_capital(self.Realcapital)
        self.building_keys = self._get_building_keys(self.active_buildings)
        self.active_spells = self._get_spells_for_capital(self.Realcapital)
        self.spell_keys = self._get_spell_keys(self.active_spells)
        self.active_hire_options: List[Dict[str, object]] = []
        self.active_mercenary_hire_options: List[Dict[str, object]] = []
        self.settlement_upgrade_names = list(self.legions_settlement_territory_source_by_name.keys())
        self._refresh_dynamic_action_layout()
        self.chests: Dict[Tuple[int, int], Tuple[str, ...]] = dict(self._static_chests)
        self.mana_sources: Dict[Tuple[int, int], Dict[str, object]] = {
            tuple(pos): dict(meta)
            for pos, meta in self._static_mana_sources.items()
        }
        self._reset_mana_state()
        self.spell_learning_locked = False
        self.heroitems: List[object] = []
        self._inventory_cache_valid: bool = False
        self._inventory_cache_signature: Tuple[str, ...] = ()
        self._hero_item_counts_cache: Dict[str, int] = {}
        self._scroll_item_counts_cache: Dict[str, int] = {}
        self._scroll_inventory_entries_cache: Tuple[Dict[str, object], ...] = ()
        self._available_scroll_spell_entries_cache: Tuple[Dict[str, object], ...] = ()
        self._scroll_cast_slot_entries_cache: Tuple[Dict[str, object], ...] = ()
        self._total_scroll_count_cache: int = 0
        self.equipped_hero_items: List[Optional[str]] = [None] * int(self.BATTLE_EQUIP_SLOTS)
        self.equipped_artifact_items: List[Optional[str]] = [None] * int(
            self.ARTIFACT_EQUIP_SLOTS
        )
        self.artifact_auto_equip_next_slot: int = 0
        self.battle_item_equip_used_this_turn: bool = False
        self.enemy_team_states: Dict[int, List[Dict]] = self._create_initial_enemy_team_states()
        self.enemy_map_spell_effects: Dict[int, set[str]] = {}
        self.blue_map_spell_effects: set[str] = set()
        self.spell_cast_counts_by_id_this_turn: Dict[str, int] = {}
        self.summon_hero_battle_bonus_pending: bool = False
        self.summon_hero_battle_bonus_enemy_ids_this_turn: set[int] = set()
        self.extra_heal_bottles: int = 0
        self.extra_healing_bottles: int = 0
        self.extra_revive_bottles: int = 0
        self.merchant_stocks: Dict[str, Dict[str, int]] = self._create_initial_merchant_stocks()
        self.spell_shop_stocks: Dict[str, Dict[str, int]] = self._create_initial_spell_shop_stocks()
        self.spell_shop_purchased_spell_ids: set[str] = set()
        self.scroll_magic_unlocked: bool = False
        self.mercenary_site_rosters: Dict[str, List[Dict[str, object]]] = (
            self._create_initial_mercenary_site_rosters()
        )
        self.active_invulnerability_potion_positions: set[int] = set()
        self.active_strength_potion_positions: set[int] = set()
        self.active_energy_elixir_positions: set[int] = set()
        self.active_haste_elixir_positions: set[int] = set()
        self.active_fire_ward_positions: set[int] = set()
        self.active_earth_ward_positions: set[int] = set()
        self.active_water_ward_positions: set[int] = set()
        self.active_air_ward_positions: set[int] = set()
        self.battle_items_equipped_total: int = 0
        self.battle_items_used_total: int = 0
        self.captured_objective_cities: set[str] = set()
        self.cleared_ruin_enemy_ids: set[int] = set()
        self.last_ruin_reward: Optional[Dict[str, object]] = None
        self._refresh_faction_territories()
        self._sync_grid_chest_positions()
        self._sync_grid_mana_sources()
        self._sync_grid_merchant_positions()
        self._sync_grid_spell_shop_positions()
        self._sync_grid_mercenary_positions()
        self._sync_grid_trainer_positions()
        self._init_campaign_grid_obs_metadata()
        self._refresh_legions_territory_grid_obs_cache()
        self.GRID_OBS_SIZE = (
            self.grid_obs_base_size
            + self.grid_enemy_obs_size
            + self.grid_campaign_obs_size
        )
        total_actions = (
            self.grid_scroll_cast_action_start
            + int(self.MAX_SCROLL_CAST_ACTIONS)
        )
        # Action space: максимум из двух режимов (battle = 27 действий, grid расширен для экономики и заклинаний)
        self.action_space = spaces.Discrete(total_actions)

        # Observation space: режим + grid + battle + ход + золото
        total_obs = 1 + self.GRID_OBS_SIZE + self.BATTLE_OBS_SIZE + 2
        self.observation_space = spaces.Box(
            low=np.zeros(total_obs, dtype=np.float32),
            high=np.ones(total_obs, dtype=np.float32),
            shape=(total_obs,),
            dtype=np.float32,
        )

        # Логи
        self._campaign_logs: List[str] = []
        self.grid_visit_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._reset_buildings_state()
        self.active_buildings = self._get_buildings_for_capital(self.Realcapital)
        self.building_keys = self._get_building_keys(self.active_buildings)
        self._reset_spells_state()
        self.active_spells = self._get_spells_for_capital(self.Realcapital)
        self.spell_keys = self._get_spell_keys(self.active_spells)
        self.settlement_upgrade_names = list(self.legions_settlement_territory_source_by_name.keys())
        self._refresh_dynamic_action_layout()
        self.mode = self.MODE_GRID
        self.current_enemy_id = None
        self.battle_origin_pos = None
        self.current_battle_context = {}
        # Стартовое состояние BLUE — копия базовых юнитов, чтобы
        # вне боя можно было применять лечение до первого боя.
        self.blue_team_state = deepcopy(UNITS_BLUE)
        self._sync_hero_progression_flags(self.blue_team_state)
        self.heal_bottles_used = 0
        self.healing_bottles_used = 0
        self.revive_bottles_used = 0
        self.turns = 0
        self.gold = 0.0
        self.spell_learning_locked = False
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state, refill=True)
        self.battle_env = None
        self._campaign_logs = []
        self.heroitems = []
        self._invalidate_inventory_cache()
        self.equipped_hero_items = [None] * int(self.BATTLE_EQUIP_SLOTS)
        self.equipped_artifact_items = [None] * int(self.ARTIFACT_EQUIP_SLOTS)
        self.artifact_auto_equip_next_slot = 0
        self.battle_item_equip_used_this_turn = False
        self.enemy_team_states = self._create_initial_enemy_team_states()
        self.enemy_map_spell_effects = {}
        self.blue_map_spell_effects = set()
        self.spell_cast_counts_by_id_this_turn = {}
        self.summon_hero_battle_bonus_pending = False
        self.summon_hero_battle_bonus_enemy_ids_this_turn = set()
        self.extra_heal_bottles = 0
        self.extra_healing_bottles = 0
        self.extra_revive_bottles = 0
        self.merchant_stocks = self._create_initial_merchant_stocks()
        self.spell_shop_stocks = self._create_initial_spell_shop_stocks()
        self.spell_shop_purchased_spell_ids = set()
        self.scroll_magic_unlocked = False
        self.mercenary_site_rosters = self._create_initial_mercenary_site_rosters()
        self.active_invulnerability_potion_positions = set()
        self.active_strength_potion_positions = set()
        self.active_energy_elixir_positions = set()
        self.active_haste_elixir_positions = set()
        self.active_fire_ward_positions = set()
        self.active_earth_ward_positions = set()
        self.active_water_ward_positions = set()
        self.active_air_ward_positions = set()
        self.battle_items_equipped_total = 0
        self.battle_items_used_total = 0
        self.combat_potion_battle_bonus_pending = False
        self.captured_objective_cities = set()
        self.cleared_ruin_enemy_ids = set()
        self.last_ruin_reward = None
        self.chests = dict(self._static_chests)
        self.mana_sources = {
            tuple(pos): dict(meta)
            for pos, meta in self._static_mana_sources.items()
        }
        self._reset_legions_settlement_territory_state()
        self._reset_mana_state()
        self.castle_heal_tiles = self._static_castle_heal_tiles
        self._refresh_faction_territories()
        self.grid_env.enemy_positions = dict(self._static_enemy_positions)
        self.grid_env.obstacle_positions = set(self._static_obstacle_tiles)
        self._sync_grid_chest_positions()
        self._sync_grid_mana_sources()
        self._sync_grid_merchant_positions()
        self._sync_grid_spell_shop_positions()
        self._sync_grid_mercenary_positions()
        self._sync_grid_trainer_positions()
        total_actions = (
            self.grid_scroll_cast_action_start
            + int(self.MAX_SCROLL_CAST_ACTIONS)
        )
        self.action_space = spaces.Discrete(total_actions)

        grid_obs, grid_info = self.grid_env.reset(seed=seed)
        grid_obs = self._augment_grid_obs(grid_obs)
        self._reset_stagnation_tracking(self.grid_env.agent_pos)

        self._log("=== НОВАЯ КАМПАНИЯ ===")
        self._log(f"Старт на позиции {self.grid_env.agent_pos}")
        self._log(f"Клетки лечения на карте: {self.castle_heal_tiles}")
        self._log(f"Препятствия на карте: {sorted(self.grid_env.obstacle_positions)}")
        self._log(f"Враги на карте: {list(self.grid_env.enemy_positions.items())}")
        self._log(f"Сундуки на карте: {list(self.chests.items())}")

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "typeoflord": int(self.typeoflord),
            "heroitems": list(self.heroitems),
            "equipped_hero_items": list(self.equipped_hero_items),
            "battle_items_equipped_total": int(self.battle_items_equipped_total),
            "battle_items_used_total": int(self.battle_items_used_total),
            "extra_heal_bottles": int(self.extra_heal_bottles or 0),
            "extra_healing_bottles": int(self.extra_healing_bottles or 0),
            "extra_revive_bottles": int(self.extra_revive_bottles or 0),
            "invulnerability_potions_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_potions_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
            "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
            "active_strength_positions": sorted(self.active_strength_potion_positions),
            "active_energy_positions": sorted(self.active_energy_elixir_positions),
            "active_haste_positions": sorted(self.active_haste_elixir_positions),
            "active_fire_ward_positions": sorted(self.active_fire_ward_positions),
            "active_earth_ward_positions": sorted(self.active_earth_ward_positions),
            "active_water_ward_positions": sorted(self.active_water_ward_positions),
            "active_air_ward_positions": sorted(self.active_air_ward_positions),
            "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
            "chests_remaining": len(self.chests),
            "legions_territory_tiles": tuple(self.legions_territory_tiles),
            "legions_territory_count": len(self.legions_territory_tiles),
            "legions_captured_gold_mine_tiles": tuple(self.legions_captured_gold_mine_tiles),
            "legions_captured_gold_mine_count": int(self.legions_captured_gold_mine_count),
            "legions_gold_mine_income_per_turn": (
                int(self.legions_captured_gold_mine_count)
                * float(self.LEGIONS_GOLD_MINE_GOLD_PER_TURN)
            ),
            "empire_territory_tiles": tuple(self.empire_territory_tiles),
            "empire_territory_count": len(self.empire_territory_tiles),
        }
        info.update(self._legions_settlement_territory_info())
        info.update(self._mana_state_info())
        info.update(self._spell_state_info())
        info.update(self._ruin_progress_info())
        info.update(self._artifact_state_info())
        info.update(self._merchant_context_info(self.grid_env.agent_pos))
        info.update(self._spell_shop_context_info(self.grid_env.agent_pos))
        info.update(self._mercenary_context_info(self.grid_env.agent_pos))
        info.update(self._trainer_context_info(self.grid_env.agent_pos))

        return self._build_obs(grid_obs=grid_obs), info

    def step(self, action: int):
        # Приводим action к int (может прийти как numpy.ndarray)
        action = int(action)

        if self.mode == self.MODE_GRID:
            return self._step_grid(action)
        else:
            return self._step_battle(action)

    def _step_grid(self, action: int):
        """Шаг в режиме перемещения по карте."""
        # В grid режиме:
        #   0-7 — движение, 8 — отдых,
        #   9-14 — применить подходящую банку лечения к позициям 7-12 соответственно.
        #   15-20 — применить бутыль воскрешения к позициям 7-12 соответственно.
        #   21-26 — применить зелье неуязвимости к позициям 7-12 соответственно.
        #   27-32 — применить зелье силы к позициям 7-12 соответственно.
        #   33-38 — лечение на специальных клетках карты по позициям 7-12.
        #   39-44 — воскрешение на специальных клетках карты по позициям 7-12.
        #   после 44 — найм фракционных юнитов в городе/столице;
        #   число action в этом блоке зависит от массива найма выбранной расы в hire.py.
        #   после него — отдельный блок найма в лагерях наёмников на карте.
        #   после блока наёмников — покупки у торговца, затем боевые эликсиры и постройки.
        #   после построек — изучение заклинаний активной фракции.
        #   после заклинаний — улучшение захваченных городов.
        if action >= self.grid_scroll_cast_action_start:
            return self._step_cast_scroll_spell(action)
        if action >= self.GRID_UNLOCK_SCROLL_MAGIC_ACTION:
            return self._step_unlock_scroll_magic(action)
        if action >= self.grid_spell_shop_cast_action_start:
            return self._step_cast_spell_shop_spell(action)
        if action >= self.grid_map_support_spell_action_start:
            return self._step_cast_support_spell(action)
        if action >= self.grid_legion_damage_spell_action_start:
            return self._step_cast_legion_damage_spell(action)
        if action >= self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START:
            return self._step_equip_battle_item(action)
        if action >= self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START:
            return self._step_equip_battle_item(action)
        if action >= self.grid_settlement_upgrade_action_start:
            return self._step_upgrade_settlement(action)
        if action >= self.grid_spell_action_start:
            return self._step_learn_spell(action)
        if action >= self.GRID_BUILD_ACTION_START:
            return self._step_building(action)
        if action >= self.GRID_SUPREME_ELIXIR_ACTION_START:
            return self._step_apply_supreme_elixir(action)
        if action >= self.GRID_TITAN_ELIXIR_ACTION_START:
            return self._step_apply_titan_elixir(action)
        if action >= self.GRID_AIR_WARD_ACTION_START:
            return self._step_apply_air_ward(action)
        if action >= self.GRID_WATER_WARD_ACTION_START:
            return self._step_apply_water_ward(action)
        if action >= self.GRID_EARTH_WARD_ACTION_START:
            return self._step_apply_earth_ward(action)
        if action >= self.GRID_FIRE_WARD_ACTION_START:
            return self._step_apply_fire_ward(action)
        if action >= self.GRID_HASTE_ACTION_START:
            return self._step_apply_haste_elixir(action)
        if action >= self.GRID_ENERGY_ACTION_START:
            return self._step_apply_energy_elixir(action)
        if action >= self.GRID_SPELL_SHOP_BUY_ACTION_START:
            return self._step_buy_spell_shop_spell(action)
        if action >= self.GRID_MERCHANT_BUY_ACTION_START:
            return self._step_buy_merchant_item(action)
        if action >= self.GRID_TRAINER_ACTION_START:
            return self._step_train_unit_at_trainer(action)
        if action >= self.GRID_MERCENARY_HIRE_ACTION_START:
            return self._step_hire_mercenary_unit(action)
        if action >= self.GRID_HIRE_ACTION_START:
            return self._step_hire_faction_unit(action)
        if action >= self.GRID_CASTLE_REVIVE_ACTION_START:
            return self._step_revive_in_castle(action)
        if action >= self.GRID_CASTLE_HEAL_ACTION_START:
            return self._step_heal_in_castle(action)
        if action >= self.GRID_STRENGTH_ACTION_START:
            return self._step_apply_strength_potion(action)
        if action >= self.GRID_INVULNERABILITY_ACTION_START:
            return self._step_apply_invulnerability_potion(action)
        if action >= self.GRID_REVIVE_ACTION_START:
            return self._step_revive_with_bottle(action)
        if action >= self.GRID_BOTTLE_ACTION_START:
            return self._step_heal_with_bottle(action)

        if 0 <= action <= 7 and self.moves <= 0:
            grid_obs = self._get_grid_obs()
            info = {
                "mode": "grid",
                "agent_pos": self.grid_env.agent_pos,
                "enemies_alive": dict(self.grid_env.enemies_alive),
                "battle_triggered": False,
                "blocked_by_moves": True,
                "turns": self.turns,
                "gold": self.gold,
                "moves": self.moves,
                "heroitems": list(self.heroitems),
                "equipped_hero_items": list(self.equipped_hero_items),
                "extra_healing_bottles": int(self.extra_healing_bottles or 0),
                "extra_revive_bottles": int(self.extra_revive_bottles or 0),
                "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
                "active_strength_positions": sorted(self.active_strength_potion_positions),
                "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
                "collected_chests": [],
                "chests_remaining": len(self.chests),
            }
            info.update(self._merchant_context_info(self.grid_env.agent_pos))
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )

        grid_action = min(action, 8)

        old_pos = self.grid_env.agent_pos
        grid_obs, _grid_reward, terminated, truncated, grid_info = self.grid_env.step(
            grid_action
        )
        # Add dense navigation signal from GridWorldEnv to reduce timeout-only learning.
        reward = 0.2 * float(_grid_reward)
        grid_obs = self._augment_grid_obs(grid_obs)
        new_pos = self.grid_env.agent_pos
        stagnation_penalty = 0.0
        battle_engage_bonus = 0.0
        summon_hero_battle_bonus = 0.0
        combat_potion_battle_bonus = 0.0

        if old_pos != new_pos:
            self._log(f"Перемещение: {old_pos} -> {new_pos}")
        elif grid_info.get("blocked_by_obstacle"):
            self._log(
                f"Препятствие: переход {old_pos} -> {grid_info.get('blocked_target')} заблокирован"
            )
        elif grid_info.get("blocked_by_boundary"):
            self._log(
                f"Граница карты: переход {old_pos} -> {grid_info.get('blocked_target')} заблокирован"
            )

        if 0 <= grid_action <= 7:
            if not grid_info.get("blocked_by_obstacle", False):
                stagnation_penalty = self._compute_stagnation_penalty(old_pos, new_pos)
            reward += stagnation_penalty
            self._spend_moves(1)

        if grid_info.get("battle_triggered"):
            enemy_id = grid_info.get("enemy_id")
            if enemy_id is not None and self.grid_env.enemies_alive.get(enemy_id, False):
                battle_engage_bonus = float(self.reward_engage_battle)
                reward += battle_engage_bonus
                if int(enemy_id) in self.summon_hero_battle_bonus_enemy_ids_this_turn:
                    summon_hero_battle_bonus = float(self.reward_summon_hero_battle_engage)
                    reward += summon_hero_battle_bonus
                    self.summon_hero_battle_bonus_enemy_ids_this_turn.discard(int(enemy_id))
                self.summon_hero_battle_bonus_pending = bool(
                    self.summon_hero_battle_bonus_enemy_ids_this_turn
                )
            if self.combat_potion_battle_bonus_pending:
                combat_potion_battle_bonus = float(
                    self.reward_combat_potion_battle_participation
                )
                reward += combat_potion_battle_bonus
                self.combat_potion_battle_bonus_pending = False
            self._apply_battle_grid_steps(extra_steps=10)
        collected_chests = self._collect_adjacent_chests()
        collected_combat_potions = self._count_collected_combat_potions(collected_chests)
        combat_potion_pickup_reward = (
            float(collected_combat_potions) * self._combat_potion_reward_value()
        )
        reward += combat_potion_pickup_reward

        info = {
            "mode": "grid",
            "agent_pos": new_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": grid_info.get("battle_triggered", False),
            "rest_action": grid_info.get("rest_action", False),
            "blocked_by_obstacle": grid_info.get("blocked_by_obstacle", False),
            "blocked_by_boundary": grid_info.get("blocked_by_boundary", False),
            "blocked_obstacle_pos": grid_info.get("blocked_target"),
            "grid_reward_raw": float(_grid_reward),
            "grid_reward_scaled": float(reward),
            "stagnation_penalty": float(stagnation_penalty),
            "battle_engage_bonus": float(battle_engage_bonus),
            "summon_hero_battle_bonus": float(summon_hero_battle_bonus),
            "summon_hero_battle_bonus_applied": bool(summon_hero_battle_bonus > 0.0),
            "summon_hero_battle_bonus_pending": bool(self.summon_hero_battle_bonus_pending),
            "summon_hero_battle_bonus_enemy_ids": sorted(
                int(enemy_id) for enemy_id in self.summon_hero_battle_bonus_enemy_ids_this_turn
            ),
            "combat_potion_battle_bonus": float(combat_potion_battle_bonus),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
            "extra_healing_bottles": int(self.extra_healing_bottles or 0),
            "extra_revive_bottles": int(self.extra_revive_bottles or 0),
            "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
            "active_strength_positions": sorted(self.active_strength_potion_positions),
            "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
            "collected_combat_potions": int(collected_combat_potions),
            "combat_potion_pickup_reward": float(combat_potion_pickup_reward),
            "collected_chests": [
                {
                    "pos": tuple(chest_data["pos"]),
                    "items": list(chest_data["items"]),
                    "granted_items": list(chest_data.get("granted_items", [])),
                }
                for chest_data in collected_chests
            ],
            "chests_remaining": len(self.chests),
        }
        info.update(self._merchant_context_info(new_pos))

        # Обработка действия REST — восстановление базового процента HP и бонусного лечения на своей столице/городе.
        if grid_info.get("rest_action"):
            pending_hire_turn_penalty = self._pending_hire_turn_penalty(
                1,
                units=self.blue_team_state,
            )
            rest_heal_percent = self._resolve_rest_heal_percent(self.grid_env.agent_pos)
            rest_heal_typeoflord_bonus_percent = self._resolve_typeoflord_rest_heal_bonus_percent()
            (
                rest_heal_bonus_percent,
                rest_heal_bonus_source,
                rest_heal_bonus_level,
            ) = self._resolve_rest_regeneration_context(self.grid_env.agent_pos)
            rest_total_percent = float(
                rest_heal_percent
                + rest_heal_typeoflord_bonus_percent
                + rest_heal_bonus_percent
            )
            healed = self._heal_blue_team(
                heal_percent=rest_heal_percent,
                bonus_percent=rest_heal_typeoflord_bonus_percent + rest_heal_bonus_percent,
            )
            if healed > 0:
                rest_heal_formula_parts = [f"{rest_heal_percent * 100.0:g}%"]
                if rest_heal_typeoflord_bonus_percent > 0.0:
                    rest_heal_formula_parts.append(
                        f"{rest_heal_typeoflord_bonus_percent * 100.0:g}%"
                    )
                if rest_heal_bonus_percent > 0.0:
                    rest_heal_formula_parts.append(f"{rest_heal_bonus_percent * 100.0:g}%")
                rest_heal_formula_text = " + ".join(rest_heal_formula_parts)
                if rest_heal_bonus_percent > 0.0:
                    if rest_heal_bonus_source == "capital":
                        self._log(
                            f"Отдых в столице: восстановлено HP у {healed} юнитов "
                            f"({rest_heal_formula_text})"
                        )
                    else:
                        self._log(
                            f"Отдых в городе {rest_heal_bonus_source} уровня {rest_heal_bonus_level}: "
                            f"восстановлено HP у {healed} юнитов "
                            f"({rest_heal_formula_text})"
                        )
                else:
                    self._log(
                        f"Отдых: восстановлено HP у {healed} юнитов "
                        f"({rest_heal_formula_text})"
                    )
            else:
                self._log("Отдых: все юниты на полном здоровье")
            # Отдых завершает ход
            reward += self._advance_turns(1)
            info["turns"] = self.turns
            info["gold"] = self.gold
            info["moves"] = self.moves
            info["healed_units"] = healed
            info["pending_hire_turn_penalty"] = float(pending_hire_turn_penalty)
            info["rest_heal_percent"] = float(rest_heal_percent)
            info["rest_heal_typeoflord_bonus_percent"] = float(
                rest_heal_typeoflord_bonus_percent
            )
            info["rest_heal_bonus_percent"] = float(rest_heal_bonus_percent)
            info["rest_heal_total_percent"] = float(rest_total_percent)
            info["rest_heal_bonus_source"] = rest_heal_bonus_source
            if rest_heal_bonus_level is not None:
                info["rest_heal_bonus_level"] = rest_heal_bonus_level

        # Проверяем столкновение с врагом
        if grid_info.get("battle_triggered"):
            enemy_id = grid_info["enemy_id"]
            self.current_enemy_id = enemy_id
            self.battle_origin_pos = (int(old_pos[0]), int(old_pos[1]))
            self.current_battle_context = {"kind": "hero"}
            self.mode = self.MODE_BATTLE

            self._log(f"!!! СТОЛКНОВЕНИЕ С ВРАГОМ {enemy_id} !!!")
            self._log(f"Описание: {ENEMY_DESCRIPTIONS.get(enemy_id, 'Неизвестный враг')}")

            # Создаём BattleEnv с нужной RED командой
            self._init_battle(enemy_id)

            # Возвращаем battle observation
            battle_obs = self.battle_env._obs()
            info["mode"] = "battle"
            info["enemy_id"] = enemy_id

            return self._build_obs(battle_obs=battle_obs), reward, False, False, info

        # Проверяем победу (все враги побеждены)
        if self.grid_env.all_enemies_defeated():
            self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
            info["campaign_result"] = "victory"
            info["campaign_victory_reason"] = "all_enemies_defeated"
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=reward,
                terminated=True,
                truncated=False,
                info=info,
            )

        # Проверяем лимит шагов на карте
        if truncated:
            self._log("=== ЛИМИТ ШАГОВ НА КАРТЕ ИСЧЕРПАН ===")
            reward += self.reward_timeout
            info["campaign_result"] = "timeout"

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _is_at_castle(self) -> bool:
        """True если агент стоит на одной из клеток лечения в grid-режиме."""
        return tuple(self.grid_env.agent_pos) in self.castle_heal_tiles

    def _finalize_grid_step_result(
        self,
        *,
        grid_obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, object],
    ):
        position_raw = info.get("agent_pos", self.grid_env.agent_pos)
        if isinstance(position_raw, (list, tuple)) and len(position_raw) >= 2:
            position = (int(position_raw[0]), int(position_raw[1]))
        else:
            position = tuple(self.grid_env.agent_pos)

        finalized_info: Dict[str, object] = dict(info)
        finalized_info["mode"] = "grid"

        sale_info = self._sell_sell_only_heroitems_at_merchant(position=position)
        finalized_reward = float(reward) + float(sale_info["merchant_sale_reward"])
        finalized_info.update(sale_info)
        self._sync_equipped_hero_items()
        self._refresh_campaign_artifact_effects(log=False)

        if "grid_reward_scaled" in finalized_info:
            try:
                finalized_info["grid_reward_scaled"] = (
                    float(finalized_info.get("grid_reward_scaled", 0.0))
                    + float(sale_info["merchant_sale_reward"])
                )
            except (TypeError, ValueError):
                finalized_info["grid_reward_scaled"] = float(finalized_reward)

        finalized_info["agent_pos"] = position
        finalized_info["turns"] = self.turns
        finalized_info["gold"] = self.gold
        finalized_info["moves"] = self.moves
        finalized_info["typeoflord"] = int(self.typeoflord)
        finalized_info["heroitems"] = list(self.heroitems)
        finalized_info["equipped_hero_items"] = list(self.equipped_hero_items)
        finalized_info.update(self._artifact_state_info())
        finalized_info["battle_items_equipped_total"] = int(self.battle_items_equipped_total)
        finalized_info["battle_items_used_total"] = int(self.battle_items_used_total)
        finalized_info["extra_heal_bottles"] = int(self.extra_heal_bottles or 0)
        finalized_info["extra_healing_bottles"] = int(self.extra_healing_bottles or 0)
        finalized_info["extra_revive_bottles"] = int(self.extra_revive_bottles or 0)
        finalized_info["invulnerability_potions_left"] = self._count_hero_item(
            self.INVULNERABILITY_POTION_ITEM_NAME
        )
        finalized_info["strength_potions_left"] = self._count_hero_item(
            self.STRENGTH_POTION_ITEM_NAME
        )
        finalized_info["energy_elixirs_left"] = self._count_hero_item(
            self.ENERGY_ELIXIR_ITEM_NAME
        )
        finalized_info["haste_elixirs_left"] = self._count_hero_item(
            self.HASTE_ELIXIR_ITEM_NAME
        )
        finalized_info["titan_elixirs_left"] = self._count_hero_item(
            self.TITAN_ELIXIR_ITEM_NAME
        )
        finalized_info["supreme_elixirs_left"] = self._count_hero_item(
            self.SUPREME_ELIXIR_ITEM_NAME
        )
        finalized_info["fire_wards_left"] = self._count_hero_item(self.FIRE_WARD_ITEM_NAME)
        finalized_info["earth_wards_left"] = self._count_hero_item(self.EARTH_WARD_ITEM_NAME)
        finalized_info["water_wards_left"] = self._count_hero_item(self.WATER_WARD_ITEM_NAME)
        finalized_info["air_wards_left"] = self._count_hero_item(self.AIR_WARD_ITEM_NAME)
        finalized_info["active_invulnerability_positions"] = sorted(
            self.active_invulnerability_potion_positions
        )
        finalized_info["active_strength_positions"] = sorted(
            self.active_strength_potion_positions
        )
        finalized_info["active_energy_positions"] = sorted(self.active_energy_elixir_positions)
        finalized_info["active_haste_positions"] = sorted(self.active_haste_elixir_positions)
        finalized_info["active_fire_ward_positions"] = sorted(self.active_fire_ward_positions)
        finalized_info["active_earth_ward_positions"] = sorted(self.active_earth_ward_positions)
        finalized_info["active_water_ward_positions"] = sorted(self.active_water_ward_positions)
        finalized_info["active_air_ward_positions"] = sorted(self.active_air_ward_positions)
        finalized_info["combat_potion_battle_bonus_pending"] = bool(
            self.combat_potion_battle_bonus_pending
        )
        finalized_info.setdefault("collected_chests", [])
        finalized_info["chests_remaining"] = len(self.chests)
        finalized_info["legions_territory_tiles"] = tuple(self.legions_territory_tiles)
        finalized_info["legions_territory_count"] = len(self.legions_territory_tiles)
        finalized_info["legions_captured_gold_mine_tiles"] = tuple(
            self.legions_captured_gold_mine_tiles
        )
        finalized_info["legions_captured_gold_mine_count"] = int(
            self.legions_captured_gold_mine_count
        )
        finalized_info["legions_gold_mine_income_per_turn"] = (
            int(self.legions_captured_gold_mine_count)
            * float(self.LEGIONS_GOLD_MINE_GOLD_PER_TURN)
        )
        finalized_info["empire_territory_tiles"] = tuple(self.empire_territory_tiles)
        finalized_info["empire_territory_count"] = len(self.empire_territory_tiles)
        finalized_info.update(self._legions_settlement_territory_info())
        finalized_info.update(self._mana_state_info())
        finalized_info.update(self._spell_state_info())
        finalized_info.update(self._ruin_progress_info())
        finalized_info.update(self._merchant_context_info(position))
        finalized_info.update(self._spell_shop_context_info(position))
        finalized_info.update(self._mercenary_context_info(position))
        finalized_info.update(self._trainer_context_info(position))

        return (
            self._build_obs(grid_obs=grid_obs),
            float(finalized_reward),
            bool(terminated),
            bool(truncated),
            finalized_info,
        )

    @staticmethod
    def _castle_heal_gold_per_hp(unit: Dict) -> float:
        """Стоимость лечения на клетке лечения.

        Фракционные юниты лечатся по Level, а нейтралы — по bucket'ам max HP.
        """
        is_neutral_unit = bool(unit.get("is_neutral_unit", False))
        if not is_neutral_unit:
            capital_raw = unit.get("capital", None)
            try:
                is_neutral_unit = capital_raw is not None and int(round(float(capital_raw))) == 0
            except (TypeError, ValueError):
                is_neutral_unit = False

        if is_neutral_unit:
            max_hp_raw = unit.get("maxhp", unit.get("max_health", 0))
            try:
                max_hp = float(max_hp_raw or 0.0)
            except (TypeError, ValueError):
                max_hp = 0.0

            if max_hp <= 100.0:
                return 1.0
            if max_hp <= 250.0:
                return 2.0
            if max_hp <= 500.0:
                return 3.0
            return 4.0

        level_raw = unit.get("Level", 1)
        try:
            level = int(round(float(level_raw)))
        except (TypeError, ValueError):
            level = 1

        if level in (2, 3):
            return 2.0
        if level >= 4:
            return 3.0
        return 1.0

    @staticmethod
    def _castle_revive_gold_cost(unit: Dict) -> float:
        """Стоимость воскрешения на клетке лечения.

        Фракционные юниты воскрешаются по Level, а нейтралы — по bucket'ам max HP.
        """
        is_neutral_unit = bool(unit.get("is_neutral_unit", False))
        if not is_neutral_unit:
            capital_raw = unit.get("capital", None)
            try:
                is_neutral_unit = capital_raw is not None and int(round(float(capital_raw))) == 0
            except (TypeError, ValueError):
                is_neutral_unit = False

        if is_neutral_unit:
            max_hp_raw = unit.get("maxhp", unit.get("max_health", 0))
            try:
                max_hp = float(max_hp_raw or 0.0)
            except (TypeError, ValueError):
                max_hp = 0.0

            if max_hp <= 100.0:
                return 50.0
            if max_hp <= 150.0:
                return 200.0
            if max_hp <= 250.0:
                return 400.0
            if max_hp <= 350.0:
                return 600.0
            if max_hp <= 500.0:
                return 800.0
            if max_hp <= 2000.0:
                return 1000.0
            return 1500.0

        level_raw = unit.get("Level", 1)
        try:
            level = int(round(float(level_raw)))
        except (TypeError, ValueError):
            level = 1

        if level <= 1:
            return 50.0
        if level == 2:
            return 200.0
        if level == 3:
            return 400.0
        if level == 4:
            return 600.0
        return 800.0

    @classmethod
    def _trainer_gold_per_xp(cls, unit: Dict) -> float:
        if not isinstance(unit, dict):
            return 0.0

        is_neutral_unit = bool(unit.get("is_neutral_unit", False))
        if not is_neutral_unit:
            capital_raw = unit.get("capital", None)
            try:
                is_neutral_unit = capital_raw is not None and int(round(float(capital_raw))) == 0
            except (TypeError, ValueError):
                is_neutral_unit = False
        if is_neutral_unit or cls._is_hero_unit(unit):
            return 5.0

        level_raw = unit.get("Level", 0)
        try:
            level = int(round(float(level_raw or 0)))
        except (TypeError, ValueError):
            level = 0

        if level <= 0:
            return 5.0
        if level == 1:
            return 4.0
        if level == 2:
            return 5.0
        if level == 3:
            return 6.0
        return 7.0

    @staticmethod
    def _trainer_exp_cap(unit: Dict) -> Optional[int]:
        if not isinstance(unit, dict):
            return None
        exp_required_raw = unit.get("exp_required", 0)
        if isinstance(exp_required_raw, str) and exp_required_raw.strip().lower() == "max":
            return None
        try:
            exp_required = int(round(float(exp_required_raw or 0)))
        except (TypeError, ValueError):
            return None
        if exp_required <= 0:
            return None
        return max(0, exp_required - 1)

    @staticmethod
    def _is_hero_unit(unit: Dict) -> bool:
        if not isinstance(unit, dict):
            return False
        if "hero" in unit:
            value = unit.get("hero")
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return int(value) != 0
            return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}
        try:
            return int(round(float(unit.get("next_level_exp", 0) or 0))) > 0
        except (TypeError, ValueError):
            return False

    def _resolve_hero_needaunit(self, unit: Dict) -> int:
        try:
            return 1 if int(round(float(unit.get("needaunit", 0) or 0))) > 0 else 0
        except (TypeError, ValueError):
            return 0

    def _sync_hero_progression_flags(self, units: Optional[List[Dict]] = None) -> None:
        roster = units if units is not None else self.blue_team_state
        if not roster:
            return
        for unit in roster:
            unit["hero"] = self._is_hero_unit(unit)
            unit["needaunit"] = self._resolve_hero_needaunit(unit)

    def _resolve_travel_hero(self, units: Optional[List[Dict]] = None) -> Optional[Dict]:
        roster = units if units is not None else self._get_blue_state()
        heroes = [unit for unit in roster if self._is_hero_unit(unit)]
        if not heroes:
            return None
        return heroes[0]

    def _hero_level(self, units: Optional[List[Dict]] = None) -> int:
        hero = self._resolve_travel_hero(units=units)
        if hero is None:
            return 0
        try:
            return max(0, int(round(float(hero.get("Level", 0) or 0))))
        except (TypeError, ValueError):
            return 0

    def _hero_needs_faction_hire(self, units: Optional[List[Dict]] = None) -> bool:
        hero = self._resolve_travel_hero(units=units)
        if hero is None:
            return False
        return self._resolve_hero_needaunit(hero) > 0

    def _hire_reward_value(self) -> float:
        return max(0.0, 3.0 * float(self.reward_defeat_enemy))

    def _pending_hire_turn_penalty(
        self,
        delta_turns: int,
        units: Optional[List[Dict]] = None,
    ) -> float:
        if delta_turns <= 0:
            return 0.0
        if not self._hero_needs_faction_hire(units=units):
            return 0.0
        return float(delta_turns) * float(self.reward_needaunit_turn_penalty)

    def _hero_has_pathfinding(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_PATHFINDING_LEVEL)

    def _hero_has_leadership(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_LEADERSHIP_LEVEL)

    def _hero_has_second_leadership(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_SECOND_LEADERSHIP_LEVEL)

    def _hero_leadership_count(self, units: Optional[List[Dict]] = None) -> int:
        level = self._hero_level(units=units)
        count = 0
        if level >= int(self.HERO_LEADERSHIP_LEVEL):
            count += 1
        if level >= int(self.HERO_SECOND_LEADERSHIP_LEVEL):
            count += 1
        return count

    def _hero_has_endurance(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_ENDURANCE_LEVEL)

    def _hero_has_strength(self, units: Optional[List[Dict]] = None) -> bool:
        return self._hero_level(units=units) >= int(self.HERO_STRENGTH_LEVEL)

    @staticmethod
    def _is_empty_blue_unit(unit: Optional[Dict]) -> bool:
        if not isinstance(unit, dict):
            return True
        name = str(unit.get("name", "") or "").strip().lower()
        if name == "пусто":
            return True
        hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
        max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
        return hp <= 0 and max_hp <= 0

    def _hire_option(self, action_idx: int) -> Optional[Dict[str, object]]:
        if 0 <= int(action_idx) < len(self.active_hire_options):
            return dict(self.active_hire_options[int(action_idx)])
        return None

    def _hire_positions_for_role(self, role: object) -> Tuple[int, ...]:
        role_name = str(role or "").strip().lower()
        if role_name == "warrior":
            return tuple(int(pos) for pos in self.HIRE_FRONT_POSITIONS)
        if role_name in {"mage", "archer", "support"}:
            return tuple(int(pos) for pos in self.HIRE_BACK_POSITIONS)
        return tuple()

    def _hire_positions_for_stand(self, stand: object) -> Tuple[int, ...]:
        stand_name = str(stand or "").strip().lower()
        if stand_name == "ahead":
            return tuple(int(pos) for pos in self.HIRE_FRONT_POSITIONS)
        if stand_name == "behind":
            return tuple(int(pos) for pos in self.HIRE_BACK_POSITIONS)
        return tuple()

    def _find_first_empty_blue_position(self, positions: Tuple[int, ...]) -> Optional[int]:
        state = self._get_blue_state()
        units_by_position = {
            int(unit.get("position", -1) or -1): unit
            for unit in state
            if isinstance(unit, dict)
        }
        for position in positions:
            unit = units_by_position.get(int(position))
            if self._is_empty_blue_unit(unit):
                return int(position)
        return None

    def _can_hire_faction_unit_option(self, option: Optional[Dict[str, object]]) -> bool:
        if not isinstance(option, dict):
            return False
        if not self._is_at_castle():
            return False

        hero = self._resolve_travel_hero()
        if hero is None:
            return False
        if not self._hero_has_leadership(units=[hero]):
            return False
        if self._resolve_hero_needaunit(hero) <= 0:
            return False

        try:
            cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            return False
        if float(self.gold or 0.0) < cost:
            return False

        unit_name = str(option.get("name", "") or "").strip()
        positions = self._hire_positions_for_role(option.get("role"))
        if not unit_name or not positions:
            return False
        if self._find_unit_data_by_name(unit_name) is None:
            return False
        return self._find_first_empty_blue_position(positions) is not None

    def _hire_faction_unit_option(
        self,
        option: Optional[Dict[str, object]],
    ) -> Tuple[bool, Optional[str], Optional[int], float]:
        if not self._can_hire_faction_unit_option(option):
            return False, None, None, 0.0

        assert isinstance(option, dict)
        unit_name = str(option.get("name", "") or "").strip()
        positions = self._hire_positions_for_role(option.get("role"))
        target_pos = self._find_first_empty_blue_position(positions)
        if target_pos is None:
            return False, None, None, 0.0

        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None:
            return False, None, None, 0.0

        recruited_unit = self._build_unit_from_data(unit_data, "blue", target_pos)
        state = self._get_blue_state()
        replaced = False
        for idx, unit in enumerate(state):
            if int(unit.get("position", -1) or -1) != int(target_pos):
                continue
            state[idx] = deepcopy(recruited_unit)
            replaced = True
            break
        if not replaced:
            state.append(deepcopy(recruited_unit))
            state.sort(key=lambda unit: int(unit.get("position", 999) or 999))

        hero = self._resolve_travel_hero(units=state)
        if hero is not None:
            hero["needaunit"] = 0

        cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        self.gold = max(0.0, float(self.gold or 0.0) - cost)
        self._sync_hero_progression_flags(state)
        self._log(
            f'Найм в городе: "{unit_name}" нанят на позицию {int(target_pos)} '
            f"(стоимость={cost:.1f} gold, gold_left={float(self.gold):.1f})"
        )
        return True, unit_name, int(target_pos), cost

    @staticmethod
    def _unit_data_is_big(unit_data: Optional[Dict]) -> bool:
        if not isinstance(unit_data, dict):
            return False
        raw_value = unit_data.get("размер", unit_data.get("big", 0))
        try:
            return bool(int(raw_value or 0))
        except (TypeError, ValueError):
            return bool(raw_value)

    @classmethod
    def _mercenary_stand_for_unit_data(cls, unit_data: Optional[Dict]) -> Optional[str]:
        if not isinstance(unit_data, dict):
            return None
        if cls._unit_data_is_big(unit_data):
            return None

        unit_type = str(unit_data.get("тип", unit_data.get("unit_type", "")) or "").strip()
        if unit_type in cls.MERCENARY_FRONTLINE_UNIT_TYPES:
            return "ahead"
        if unit_type in cls.MERCENARY_BACKLINE_UNIT_TYPES:
            return "behind"

        attack_type = str(
            unit_data.get("тип атаки1", unit_data.get("attack_type_primary", "")) or ""
        ).strip().lower()
        return "ahead" if attack_type == "weapon" else "behind"

    def _mercenary_option(self, action_idx: int) -> Optional[Dict[str, object]]:
        if not (0 <= int(action_idx) < len(self.active_mercenary_hire_options)):
            return None

        option = dict(self.active_mercenary_hire_options[int(action_idx)])
        roster_entry = self._mercenary_roster_entry(
            str(option.get("site_name", "") or ""),
            int(option.get("slot", 0) or 0),
        )
        option["stock"] = int(roster_entry.get("stock", 0) or 0) if roster_entry else 0
        if roster_entry is not None:
            option["stand"] = str(roster_entry.get("stand", option.get("stand", "")) or "")
            option["unit_type"] = str(
                roster_entry.get("unit_type", option.get("unit_type", "")) or ""
            )
        return option

    def _can_hire_mercenary_unit_option(self, option: Optional[Dict[str, object]]) -> bool:
        if not isinstance(option, dict):
            return False

        site_name = str(option.get("site_name", "") or "")
        if not site_name or site_name not in self._mercenary_sites_at_position(self.grid_env.agent_pos):
            return False
        if max(0, int(option.get("stock", 0) or 0)) <= 0:
            return False

        hero = self._resolve_travel_hero()
        if hero is None:
            return False
        if not self._hero_has_leadership(units=[hero]):
            return False

        unit_name = str(option.get("unit_name", option.get("name", "")) or "").strip()
        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None or self._unit_data_is_big(unit_data):
            return False

        stand = str(option.get("stand", "") or "").strip().lower()
        if not stand:
            resolved_stand = self._mercenary_stand_for_unit_data(unit_data)
            stand = "" if resolved_stand is None else resolved_stand
        positions = self._hire_positions_for_stand(stand)
        if not unit_name or not positions:
            return False

        try:
            cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            return False
        if float(self.gold or 0.0) < cost:
            return False

        return self._find_first_empty_blue_position(positions) is not None

    def _hire_mercenary_unit_option(
        self,
        option: Optional[Dict[str, object]],
    ) -> Tuple[bool, Optional[str], Optional[int], float, Optional[str], int, int]:
        if not self._can_hire_mercenary_unit_option(option):
            return False, None, None, 0.0, None, 0, 0

        assert isinstance(option, dict)
        site_name = str(option.get("site_name", "") or "")
        slot = int(option.get("slot", 0) or 0)
        unit_name = str(option.get("unit_name", option.get("name", "")) or "").strip()
        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None or self._unit_data_is_big(unit_data):
            return False, None, None, 0.0, site_name or None, 0, 0

        stand = str(option.get("stand", "") or "").strip().lower()
        if not stand:
            resolved_stand = self._mercenary_stand_for_unit_data(unit_data)
            stand = "" if resolved_stand is None else resolved_stand
        positions = self._hire_positions_for_stand(stand)
        target_pos = self._find_first_empty_blue_position(positions)
        if target_pos is None:
            return False, None, None, 0.0, site_name or None, 0, 0

        roster_entry = self._mercenary_roster_entry(site_name, slot)
        stock_before = max(0, int(roster_entry.get("stock", 0) or 0)) if roster_entry else 0
        if stock_before <= 0:
            return False, None, None, 0.0, site_name or None, stock_before, stock_before

        recruited_unit = self._build_unit_from_data(unit_data, "blue", target_pos)
        state = self._get_blue_state()
        replaced = False
        for idx, unit in enumerate(state):
            if int(unit.get("position", -1) or -1) != int(target_pos):
                continue
            state[idx] = deepcopy(recruited_unit)
            replaced = True
            break
        if not replaced:
            state.append(deepcopy(recruited_unit))
            state.sort(key=lambda unit: int(unit.get("position", 999) or 999))

        hero = self._resolve_travel_hero(units=state)
        if hero is not None:
            hero["needaunit"] = 0

        cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        self.gold = max(0.0, float(self.gold or 0.0) - cost)
        if roster_entry is not None:
            roster_entry["stock"] = max(0, stock_before - 1)
        stock_after = max(0, int(roster_entry.get("stock", 0) or 0)) if roster_entry else 0
        self._sync_hero_progression_flags(state)
        self._log(
            f'Лагерь наёмников {site_name}: "{unit_name}" нанят на позицию {int(target_pos)} '
            f"(стоимость={cost:.1f} gold, gold_left={float(self.gold):.1f}, stock_left={stock_after})"
        )
        return True, unit_name, int(target_pos), cost, site_name or None, stock_before, stock_after

    def _hero_pathfinding_move_bonus(self, units: Optional[List[Dict]] = None) -> int:
        if not self._hero_has_pathfinding(units=units):
            return 0
        return max(
            0,
            int(round(float(self.MOVES_PER_TURN) * float(self.HERO_PATHFINDING_MOVE_BONUS_PCT))),
        )

    def _hero_move_bonus(self, units: Optional[List[Dict]] = None) -> int:
        return self._hero_level(units=units) + self._hero_pathfinding_move_bonus(units=units)

    def get_travel_hero_visual_info(self) -> Optional[Dict]:
        hero = self._resolve_travel_hero()
        if hero is None:
            return None

        hero["needaunit"] = self._resolve_hero_needaunit(hero)
        abilities: List[str] = []
        if self._hero_has_pathfinding(units=[hero]):
            abilities.append("Нахождение пути (+20% шагов)")
        leadership_count = self._hero_leadership_count(units=[hero])
        if leadership_count == 1:
            abilities.append("Лидерство")
        elif leadership_count >= 2:
            abilities.append("Лидерство x2")
        if self._hero_has_endurance(units=[hero]):
            abilities.append("Выносливость")
        if self._hero_has_strength(units=[hero]):
            abilities.append("Сила (+25% основной урон)")

        return {
            "name": str(hero.get("name", "") or "").strip(),
            "level": self._hero_level(units=[hero]),
            "abilities": abilities,
        }

    def _sync_grid_chest_positions(self) -> None:
        self.grid_env.chest_positions = set(self.chests.keys())

    def _sync_grid_mana_sources(self) -> None:
        self.grid_env.mana_sources = {
            tuple(pos): dict(meta)
            for pos, meta in getattr(self, "mana_sources", {}).items()
        }

    def _sync_grid_merchant_positions(self) -> None:
        self.grid_env.merchant_positions = set(self.merchant_interaction_tiles)

    def _sync_grid_spell_shop_positions(self) -> None:
        self.grid_env.spell_shop_positions = set(self.spell_shop_interaction_tiles)

    def _sync_grid_mercenary_positions(self) -> None:
        self.grid_env.mercenary_positions = set(self.mercenary_interaction_tiles)

    def _sync_grid_trainer_positions(self) -> None:
        self.grid_env.trainer_positions = set(self.trainer_interaction_tiles)

    def _max_heal_bottles_available(self) -> int:
        return int(self.MAX_HEAL_BOTTLES) + max(0, int(self.extra_heal_bottles or 0))

    def _heal_bottles_left(self) -> int:
        return max(0, self._max_heal_bottles_available() - int(self.heal_bottles_used or 0))

    def _max_healing_bottles_available(self) -> int:
        return int(self.MAX_HEALING_BOTTLES) + max(0, int(self.extra_healing_bottles or 0))

    def _healing_bottles_left(self) -> int:
        return max(0, self._max_healing_bottles_available() - int(self.healing_bottles_used or 0))

    def _max_revive_bottles_available(self) -> int:
        return int(self.MAX_REVIVE_BOTTLES) + max(0, int(self.extra_revive_bottles or 0))

    def _revive_bottles_left(self) -> int:
        return max(0, self._max_revive_bottles_available() - int(self.revive_bottles_used or 0))

    @classmethod
    def _battle_item_effect_definitions(cls) -> Dict[str, Dict[str, object]]:
        return {
            cls.BONUS_SMALL_HEAL_ITEM_NAME: {
                "kind": "heal",
                "amount": float(cls.HEALING_BOTTLE_AMOUNT),
            },
            cls.BONUS_LARGE_HEAL_ITEM_NAME: {
                "kind": "heal",
                "amount": float(cls.HEAL_BOTTLE_AMOUNT),
            },
            cls.HEALING_OINTMENT_ITEM_NAME: {
                "kind": "heal",
                "amount": float(cls.HEALING_OINTMENT_AMOUNT),
            },
            cls.BONUS_REVIVE_ITEM_NAME: {
                "kind": "revive",
                "amount": 1.0,
            },
        }

    def _battle_equip_item_available_count(self, item_name: str) -> int:
        normalized_name = str(item_name or "")
        if normalized_name == self.BONUS_SMALL_HEAL_ITEM_NAME:
            return int(self._healing_bottles_left())
        if normalized_name == self.BONUS_LARGE_HEAL_ITEM_NAME:
            return int(self._heal_bottles_left())
        if normalized_name == self.BONUS_REVIVE_ITEM_NAME:
            return int(self._revive_bottles_left())
        if normalized_name == self.HEALING_OINTMENT_ITEM_NAME:
            return int(self._count_hero_item(self.HEALING_OINTMENT_ITEM_NAME))
        return 0

    def _count_equipped_battle_items(
        self,
        item_name: str,
        *,
        exclude_slot: Optional[int] = None,
    ) -> int:
        normalized_name = str(item_name or "")
        count = 0
        for slot_index, equipped_name in enumerate(self.equipped_hero_items):
            if exclude_slot is not None and int(slot_index) == int(exclude_slot):
                continue
            if str(equipped_name or "") == normalized_name:
                count += 1
        return count

    def _sync_equipped_hero_items(self) -> None:
        normalized_slots: List[Optional[str]] = [None] * int(self.BATTLE_EQUIP_SLOTS)
        reserved: Dict[str, int] = {}
        current_slots = list(self.equipped_hero_items or [])
        for slot_index in range(int(self.BATTLE_EQUIP_SLOTS)):
            item_name = (
                str(current_slots[slot_index] or "")
                if slot_index < len(current_slots) and current_slots[slot_index] is not None
                else ""
            )
            if item_name not in self.BATTLE_EQUIPPABLE_ITEM_NAMES:
                continue
            available_count = int(self._battle_equip_item_available_count(item_name))
            already_reserved = int(reserved.get(item_name, 0))
            if already_reserved >= available_count:
                continue
            normalized_slots[slot_index] = item_name
            reserved[item_name] = already_reserved + 1
        self.equipped_hero_items = normalized_slots

    def _can_equip_battle_item(self, slot_index: int, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        if normalized_name not in self.BATTLE_EQUIPPABLE_ITEM_NAMES:
            return False
        if not (0 <= int(slot_index) < int(self.BATTLE_EQUIP_SLOTS)):
            return False
        if self.battle_item_equip_used_this_turn:
            return False
        self._sync_equipped_hero_items()
        if str(self.equipped_hero_items[int(slot_index)] or "") == normalized_name:
            return False
        available_count = int(self._battle_equip_item_available_count(normalized_name))
        reserved_elsewhere = self._count_equipped_battle_items(
            normalized_name,
            exclude_slot=int(slot_index),
        )
        return available_count > reserved_elsewhere

    def _equip_battle_item(self, slot_index: int, item_name: str) -> bool:
        if not self._can_equip_battle_item(slot_index, item_name):
            return False
        self.equipped_hero_items[int(slot_index)] = str(item_name)
        self._sync_equipped_hero_items()
        return True

    def _consume_counter_backed_item(self, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        if normalized_name == self.BONUS_SMALL_HEAL_ITEM_NAME:
            if self._healing_bottles_left() <= 0:
                return False
            self.healing_bottles_used = min(
                self.healing_bottles_used + 1,
                self._max_healing_bottles_available(),
            )
            self._consume_hero_item(self.BONUS_SMALL_HEAL_ITEM_NAME)
            return True
        if normalized_name == self.BONUS_LARGE_HEAL_ITEM_NAME:
            if self._heal_bottles_left() <= 0:
                return False
            self.heal_bottles_used = min(
                self.heal_bottles_used + 1,
                self._max_heal_bottles_available(),
            )
            self._consume_hero_item(self.BONUS_LARGE_HEAL_ITEM_NAME)
            return True
        if normalized_name == self.BONUS_REVIVE_ITEM_NAME:
            if self._revive_bottles_left() <= 0:
                return False
            self.revive_bottles_used = min(
                self.revive_bottles_used + 1,
                self._max_revive_bottles_available(),
            )
            self._consume_hero_item(self.BONUS_REVIVE_ITEM_NAME)
            return True
        return False

    def _consume_battle_equipable_item(self, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        consumed = False
        if normalized_name == self.HEALING_OINTMENT_ITEM_NAME:
            consumed = self._consume_hero_item(self.HEALING_OINTMENT_ITEM_NAME)
        else:
            consumed = self._consume_counter_backed_item(normalized_name)
        self._sync_equipped_hero_items()
        return bool(consumed)

    @staticmethod
    def _normalize_damage_value(value: object) -> int:
        try:
            return max(0, int(round(float(value or 0))))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _normalize_health_value(value: object) -> int:
        try:
            return max(0, int(round(float(value or 0))))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _hero_item_name(entry: object) -> str:
        if isinstance(entry, HeroInventoryItem):
            return str(entry)
        if isinstance(entry, dict):
            return str(entry.get("name", "") or "")
        return str(entry or "")

    @staticmethod
    def _hero_item_gold(entry: object) -> int:
        if isinstance(entry, HeroInventoryItem):
            try:
                return max(0, int(getattr(entry, "gold", 0) or 0))
            except (TypeError, ValueError):
                return 0
        if isinstance(entry, dict):
            try:
                return max(0, int(entry.get("gold", 0) or 0))
            except (TypeError, ValueError):
                return 0
        return 0

    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_item_definitions_by_name(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(entry.get("item_name", "") or ""): dict(entry)
            for entry in SCROLL_ITEM_DEFINITIONS
            if str(entry.get("item_name", "") or "")
        }

    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_item_alias_to_canonical(cls) -> Dict[str, str]:
        alias_map: Dict[str, str] = {
            str(item_name): str(item_name)
            for item_name in cls._scroll_item_definitions_by_name().keys()
        }
        for alias_name, canonical_name in SCROLL_ITEM_ALIASES.items():
            alias_map[str(alias_name or "")] = str(canonical_name or "")
        return alias_map

    @classmethod
    def _canonical_scroll_item_name(cls, item_name: str) -> str:
        return str(
            cls._scroll_item_alias_to_canonical().get(str(item_name or ""), "") or ""
        )

    def _invalidate_inventory_cache(self) -> None:
        self._inventory_cache_valid = False
        self._inventory_cache_signature = ()
        self._hero_item_counts_cache = {}
        self._scroll_item_counts_cache = {}
        self._scroll_inventory_entries_cache = ()
        self._available_scroll_spell_entries_cache = ()
        self._scroll_cast_slot_entries_cache = ()
        self._total_scroll_count_cache = 0

    def _ensure_inventory_cache(self) -> None:
        current_signature = tuple(self._hero_item_name(entry) for entry in self.heroitems)
        if self._inventory_cache_valid and current_signature == self._inventory_cache_signature:
            return

        hero_item_counts: Dict[str, int] = {}
        scroll_item_counts: Dict[str, int] = {}
        for item_name in current_signature:
            hero_item_counts[item_name] = hero_item_counts.get(item_name, 0) + 1

            canonical_scroll_name = self._canonical_scroll_item_name(item_name)
            if canonical_scroll_name:
                scroll_item_counts[canonical_scroll_name] = (
                    scroll_item_counts.get(canonical_scroll_name, 0) + 1
                )

        definition_by_name = self._scroll_item_definitions_by_name()
        inventory_entries: List[Dict[str, object]] = []
        available_entries: List[Dict[str, object]] = []
        for definition in SCROLL_ITEM_DEFINITIONS:
            item_name = str(definition.get("item_name", "") or "")
            canonical_name = self._canonical_scroll_item_name(item_name)
            if not canonical_name:
                continue

            copies = int(scroll_item_counts.get(canonical_name, 0) or 0)
            if copies <= 0:
                continue

            base_definition = definition_by_name.get(canonical_name, {})
            if not isinstance(base_definition, dict):
                continue

            scroll_entry = dict(base_definition)
            scroll_entry["copies"] = copies
            inventory_entries.append(scroll_entry)
            if bool(scroll_entry.get("supported", False)):
                available_entries.append(scroll_entry)

        self._hero_item_counts_cache = hero_item_counts
        self._scroll_item_counts_cache = scroll_item_counts
        self._scroll_inventory_entries_cache = tuple(inventory_entries)
        self._available_scroll_spell_entries_cache = tuple(available_entries)
        self._scroll_cast_slot_entries_cache = tuple(
            available_entries[: int(self.MAX_SCROLL_CAST_ACTIONS)]
        )
        self._total_scroll_count_cache = int(sum(scroll_item_counts.values()))
        self._inventory_cache_signature = current_signature
        self._inventory_cache_valid = True

    def is_scroll_item(self, item_name: str) -> bool:
        return bool(self.scroll_item_definition(item_name))

    def scroll_item_definition(self, item_name: str) -> Dict[str, object]:
        canonical_name = self._canonical_scroll_item_name(item_name)
        if not canonical_name:
            return {}
        definition = self._scroll_item_definitions_by_name().get(canonical_name, {})
        if not isinstance(definition, dict):
            return {}
        return dict(definition)

    def count_scroll_item(self, item_name: str) -> int:
        canonical_name = self._canonical_scroll_item_name(item_name)
        if not canonical_name:
            return 0
        self._ensure_inventory_cache()
        return int(self._scroll_item_counts_cache.get(canonical_name, 0) or 0)

    def count_scroll_spell(self, spell_id: str) -> int:
        normalized_spell_id = str(spell_id or "")
        if not normalized_spell_id:
            return 0
        for definition in SCROLL_ITEM_DEFINITIONS:
            if str(definition.get("spell_id", "") or "") == normalized_spell_id:
                return self.count_scroll_item(str(definition.get("item_name", "") or ""))
        return 0

    def available_scroll_spell_entries(self) -> Tuple[Dict[str, object], ...]:
        self._ensure_inventory_cache()
        return self._available_scroll_spell_entries_cache

    def scroll_cast_slot_entries(self) -> Tuple[Dict[str, object], ...]:
        self._ensure_inventory_cache()
        return self._scroll_cast_slot_entries_cache

    @classmethod
    def _hero_item_gold_value_by_name(cls, item_name: str) -> int:
        canonical_scroll_name = cls._canonical_scroll_item_name(item_name)
        if canonical_scroll_name:
            try:
                return max(
                    0,
                    int(
                        cls._scroll_item_definitions_by_name()
                        .get(canonical_scroll_name, {})
                        .get("gold_value", 0)
                        or 0
                    ),
                )
            except (TypeError, ValueError):
                return 0
        try:
            return max(0, int(cls.CHEST_ITEM_GOLD_VALUES.get(str(item_name or ""), 0) or 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _make_hero_item_entry(cls, item_name: str) -> HeroInventoryItem:
        normalized_name = str(item_name or "")
        return HeroInventoryItem(
            normalized_name,
            cls._hero_item_gold_value_by_name(normalized_name),
        )

    def _hero_item_display_name(self, entry: object) -> str:
        return self._hero_item_name(entry)

    @classmethod
    def _is_artifact_item_name(cls, item_name: str) -> bool:
        normalized_name = str(item_name or "").strip().lower()
        if not normalized_name:
            return False
        return "(artifact)" in normalized_name or "(артефакт)" in normalized_name

    def _sync_equipped_artifact_items(self) -> None:
        normalized_slots: List[Optional[str]] = [None] * int(self.ARTIFACT_EQUIP_SLOTS)
        reserved: Dict[str, int] = {}
        current_slots = list(getattr(self, "equipped_artifact_items", []) or [])
        available_counts: Dict[str, int] = {}

        for entry in self.heroitems:
            item_name = self._hero_item_name(entry)
            if not self._is_artifact_item_name(item_name):
                continue
            available_counts[item_name] = available_counts.get(item_name, 0) + 1

        for slot_index in range(int(self.ARTIFACT_EQUIP_SLOTS)):
            item_name = (
                str(current_slots[slot_index] or "")
                if slot_index < len(current_slots) and current_slots[slot_index] is not None
                else ""
            )
            if not self._is_artifact_item_name(item_name):
                continue
            available_count = int(available_counts.get(item_name, 0) or 0)
            already_reserved = int(reserved.get(item_name, 0) or 0)
            if already_reserved >= available_count:
                continue
            normalized_slots[slot_index] = item_name
            reserved[item_name] = already_reserved + 1

        self.equipped_artifact_items = normalized_slots
        self.artifact_auto_equip_next_slot = (
            int(getattr(self, "artifact_auto_equip_next_slot", 0) or 0)
            % max(1, int(self.ARTIFACT_EQUIP_SLOTS))
        )

    def _auto_equip_artifact_item(self, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "")
        if not self._is_artifact_item_name(normalized_name):
            return {
                "artifact_auto_equipped": False,
                "artifact_auto_equip_slot": None,
                "artifact_auto_equip_previous": None,
            }

        self._sync_equipped_artifact_items()
        slot_index = int(self.artifact_auto_equip_next_slot or 0) % max(
            1,
            int(self.ARTIFACT_EQUIP_SLOTS),
        )
        previous_item = self.equipped_artifact_items[slot_index]
        self.equipped_artifact_items[slot_index] = normalized_name
        self.artifact_auto_equip_next_slot = (slot_index + 1) % max(
            1,
            int(self.ARTIFACT_EQUIP_SLOTS),
        )
        self._log(
            f"Артефакт '{normalized_name}' автоматически экипирован в слот "
            f"{slot_index + 1} (предыдущий: {previous_item or 'пусто'})."
        )
        return {
            "artifact_auto_equipped": True,
            "artifact_auto_equip_slot": int(slot_index + 1),
            "artifact_auto_equip_previous": previous_item,
        }

    def _append_hero_item(self, item_name: str) -> None:
        normalized_name = str(item_name or "")
        self.heroitems.append(self._make_hero_item_entry(normalized_name))
        self._invalidate_inventory_cache()

    def _add_hero_item(self, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "")
        self._append_hero_item(normalized_name)
        artifact_info = self._auto_equip_artifact_item(normalized_name)
        if bool(artifact_info.get("artifact_auto_equipped")):
            self._refresh_campaign_artifact_effects(log=True)
        return artifact_info

    def _artifact_state_info(self) -> Dict[str, object]:
        self._refresh_campaign_artifact_effects(log=False)
        return {
            "equipped_artifact_items": list(self.equipped_artifact_items),
            "artifact_auto_equip_next_slot": int(self.artifact_auto_equip_next_slot),
        }

    @classmethod
    def _artifact_effect_definition(cls, item_name: str) -> Dict[str, object]:
        normalized_name = str(item_name or "")
        damage_multipliers = {
            cls.RING_OF_AGES_ITEM_NAME: cls.RING_OF_AGES_DAMAGE_MULTIPLIER,
            cls.RING_OF_AGES_ENGLISH_ITEM_NAME: cls.RING_OF_AGES_DAMAGE_MULTIPLIER,
            cls.DWARVEN_BRACER_ITEM_NAME: cls.DWARVEN_BRACER_DAMAGE_MULTIPLIER,
            cls.UNHOLY_CHALICE_ARTIFACT_ITEM_NAME: (
                cls.UNHOLY_CHALICE_ARTIFACT_DAMAGE_MULTIPLIER
            ),
            cls.RING_OF_STRENGTH_ITEM_NAME: cls.RING_OF_STRENGTH_DAMAGE_MULTIPLIER,
            cls.RUNIC_BLADE_ITEM_NAME: cls.RUNIC_BLADE_DAMAGE_MULTIPLIER,
            cls.MJOLNIRS_CROWN_ITEM_NAME: cls.MJOLNIRS_CROWN_DAMAGE_MULTIPLIER,
            cls.BETHREZENS_CLAW_ITEM_NAME: cls.BETHREZENS_CLAW_DAMAGE_MULTIPLIER,
        }
        initiative_multipliers = {
            cls.RING_OF_AGES_ITEM_NAME: cls.RING_OF_AGES_INITIATIVE_MULTIPLIER,
            cls.RING_OF_AGES_ENGLISH_ITEM_NAME: cls.RING_OF_AGES_INITIATIVE_MULTIPLIER,
            cls.BETHREZENS_CLAW_ITEM_NAME: cls.BETHREZENS_CLAW_INITIATIVE_MULTIPLIER,
        }
        armor_bonuses = {
            cls.RUNESTONE_ARTIFACT_ITEM_NAME: cls.RUNESTONE_ARTIFACT_ARMOR_BONUS,
            cls.HOLY_CHALICE_ARTIFACT_ITEM_NAME: cls.HOLY_CHALICE_ARTIFACT_ARMOR_BONUS,
            cls.SKULL_BRACERS_ITEM_NAME: cls.SKULL_BRACERS_ARMOR_BONUS,
            cls.HORN_OF_AWARENESS_ITEM_NAME: cls.HORN_OF_AWARENESS_ARMOR_BONUS,
            cls.ETCHED_CIRCLET_ITEM_NAME: cls.ETCHED_CIRCLET_ARMOR_BONUS,
        }

        effect: Dict[str, object] = {}
        if normalized_name in damage_multipliers:
            effect["damage_multiplier"] = float(damage_multipliers[normalized_name])
        if normalized_name in initiative_multipliers:
            effect["initiative_multiplier"] = float(initiative_multipliers[normalized_name])
        if normalized_name in armor_bonuses:
            effect["armor_bonus"] = int(armor_bonuses[normalized_name])
        return effect

    @classmethod
    def _hero_item_aliases(cls, item_name: str) -> Tuple[str, ...]:
        normalized_name = str(item_name or "")
        aliases = {normalized_name}
        canonical_scroll_name = cls._canonical_scroll_item_name(normalized_name)
        if canonical_scroll_name:
            aliases.add(canonical_scroll_name)
            for alias_name, alias_canonical_name in cls._scroll_item_alias_to_canonical().items():
                if str(alias_canonical_name or "") == canonical_scroll_name:
                    aliases.add(str(alias_name or ""))
        if normalized_name == cls.INVULNERABILITY_POTION_ITEM_NAME:
            aliases.add(cls.RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME)
        elif normalized_name == cls.RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME:
            aliases.add(cls.INVULNERABILITY_POTION_ITEM_NAME)
        elif normalized_name == cls.RING_OF_AGES_ITEM_NAME:
            aliases.add(cls.RING_OF_AGES_ENGLISH_ITEM_NAME)
        elif normalized_name == cls.RING_OF_AGES_ENGLISH_ITEM_NAME:
            aliases.add(cls.RING_OF_AGES_ITEM_NAME)
        return tuple(aliases)

    def _count_hero_item(self, item_name: str) -> int:
        if not item_name:
            return 0
        self._ensure_inventory_cache()
        normalized_names = set(self._hero_item_aliases(item_name))
        return int(
            sum(int(self._hero_item_counts_cache.get(name, 0) or 0) for name in normalized_names)
        )

    def _hero_item_available(self, item_name: str) -> bool:
        return self._count_hero_item(item_name) > 0

    def _consume_hero_item(self, item_name: str) -> bool:
        if not item_name:
            return False
        normalized_names = set(self._hero_item_aliases(item_name))
        for idx, entry in enumerate(self.heroitems):
            if self._hero_item_name(entry) in normalized_names:
                del self.heroitems[idx]
                self._invalidate_inventory_cache()
                self._sync_equipped_artifact_items()
                self._refresh_campaign_artifact_effects(log=False)
                return True
        return False

    def _hero_item_sell_value(self, entry: object) -> int:
        gold_value = self._hero_item_gold(entry)
        if gold_value > 0:
            return gold_value
        return self._hero_item_gold_value_by_name(self._hero_item_name(entry))

    def _is_useful_hero_item_name(self, item_name: str) -> bool:
        normalized_name = str(item_name or "")
        if self._is_artifact_item_name(normalized_name):
            return True
        return normalized_name in {
            self.BONUS_SMALL_HEAL_ITEM_NAME,
            self.BONUS_LARGE_HEAL_ITEM_NAME,
            self.HEALING_OINTMENT_ITEM_NAME,
            self.BONUS_REVIVE_ITEM_NAME,
            self.INVULNERABILITY_POTION_ITEM_NAME,
            self.RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME,
            self.STRENGTH_POTION_ITEM_NAME,
            self.TITAN_ELIXIR_ITEM_NAME,
            self.SUPREME_ELIXIR_ITEM_NAME,
            *self.RUIN_REWARD_ITEM_NAMES,
            *[str(item_data.get("name", "") or "") for item_data in self.MERCHANT_BUY_ITEMS],
        }

    def _ruin_progress_info(self) -> Dict[str, object]:
        last_reward = None
        if isinstance(self.last_ruin_reward, dict):
            last_reward = dict(self.last_ruin_reward)
        return {
            "ruins_cleared": int(len(self.cleared_ruin_enemy_ids)),
            "ruins_total": int(len(self.RUIN_REWARD_BY_ENEMY_ID)),
            "cleared_ruin_enemy_ids": sorted(int(enemy_id) for enemy_id in self.cleared_ruin_enemy_ids),
            "last_ruin_reward": last_reward,
        }

    @classmethod
    def _empty_mana_dict(cls, value: float = 0.0) -> Dict[str, float]:
        return {
            str(kind): float(value)
            for kind in cls.MANA_KIND_ORDER
        }

    def _reset_mana_state(self) -> None:
        for kind, attr_name in self.MANA_ATTR_BY_KIND.items():
            setattr(self, str(attr_name), 0.0)
        self.legions_captured_mana_source_tiles_by_kind: Dict[str, Tuple[Tuple[int, int], ...]] = {
            str(kind): ()
            for kind in self.MANA_KIND_ORDER
        }
        self.legions_captured_mana_source_counts_by_kind: Dict[str, int] = {
            str(kind): 0
            for kind in self.MANA_KIND_ORDER
        }
        self.mana_income_per_turn: Dict[str, float] = self._empty_mana_dict()

    def _current_mana_totals(self) -> Dict[str, float]:
        return {
            str(kind): float(getattr(self, self.MANA_ATTR_BY_KIND[str(kind)], 0.0) or 0.0)
            for kind in self.MANA_KIND_ORDER
        }

    def _compute_mana_income_per_turn(self) -> Dict[str, float]:
        mana_income = self._empty_mana_dict()
        mana_income["infernal"] = float(self.BASE_INFERNAL_MANA_PER_TURN)
        for kind in self.MANA_KIND_ORDER:
            captured_count = int(
                getattr(self, "legions_captured_mana_source_counts_by_kind", {}).get(str(kind), 0)
                or 0
            )
            mana_income[str(kind)] += float(captured_count) * float(
                self.LEGIONS_MANA_SOURCE_MANA_PER_TURN
            )
        return mana_income

    def _apply_mana_income_for_turn(self) -> Dict[str, float]:
        mana_income = self._compute_mana_income_per_turn()
        for kind, delta in mana_income.items():
            attr_name = str(self.MANA_ATTR_BY_KIND.get(str(kind), "") or "")
            if not attr_name:
                continue
            current_value = float(getattr(self, attr_name, 0.0) or 0.0)
            setattr(self, attr_name, current_value + float(delta))
        self.mana_income_per_turn = dict(mana_income)
        return dict(mana_income)

    def _mana_state_info(self) -> Dict[str, object]:
        mana_totals = self._current_mana_totals()
        mana_income = dict(self._compute_mana_income_per_turn())
        info: Dict[str, object] = {
            "mana_totals": dict(mana_totals),
            "mana_income_per_turn": dict(mana_income),
            "legions_captured_mana_source_tiles_by_kind": {
                str(kind): tuple(self.legions_captured_mana_source_tiles_by_kind.get(str(kind), ()))
                for kind in self.MANA_KIND_ORDER
            },
            "legions_captured_mana_source_counts_by_kind": {
                str(kind): int(self.legions_captured_mana_source_counts_by_kind.get(str(kind), 0) or 0)
                for kind in self.MANA_KIND_ORDER
            },
        }
        for kind in self.MANA_KIND_ORDER:
            attr_name = str(self.MANA_ATTR_BY_KIND[str(kind)])
            info[attr_name] = float(mana_totals[str(kind)])
            info[f"{attr_name}_income_per_turn"] = float(mana_income[str(kind)])
        return info

    def _reset_legions_settlement_territory_state(self) -> None:
        self.legions_active_settlement_territory_capture_turn_by_name: Dict[str, int] = {}
        self.legions_settlement_territory_tiles_by_name: Dict[str, Tuple[Tuple[int, int], ...]] = {
            str(source_name): ()
            for source_name in self.legions_settlement_territory_source_by_name.keys()
        }
        self.legions_settlement_level_by_name: Dict[str, int] = {
            str(source_name): max(
                1,
                int(
                    source_data.get("settlement_level", source_data.get("territory_level", 1))
                    or 1
                ),
            )
            for source_name, source_data in self.legions_settlement_territory_source_by_name.items()
        }

    def _legions_settlement_territory_info(self) -> Dict[str, object]:
        return {
            "legions_active_settlement_territories": sorted(
                self.legions_active_settlement_territory_capture_turn_by_name.keys()
            ),
            "legions_settlement_territory_capture_turn_by_name": {
                str(name): int(turn)
                for name, turn in self.legions_active_settlement_territory_capture_turn_by_name.items()
            },
            "legions_settlement_territory_tiles_by_name": {
                str(name): tuple(tiles)
                for name, tiles in self.legions_settlement_territory_tiles_by_name.items()
            },
            "legions_settlement_level_by_name": {
                str(name): int(level)
                for name, level in self.legions_settlement_level_by_name.items()
            },
        }

    def _grant_ruin_reward(self, enemy_id: Optional[int]) -> Dict[str, object]:
        reward_data = self.RUIN_REWARD_BY_ENEMY_ID.get(int(enemy_id or -1))
        if reward_data is None:
            return {
                "ruin_cleared": False,
                "ruin_reward_applied": False,
                **self._ruin_progress_info(),
            }

        ruin_enemy_id = int(enemy_id or -1)
        ruin_title = str(reward_data.get("title", "") or "")
        item_name = str(reward_data.get("item", "") or "")
        gold_reward = max(0.0, float(reward_data.get("gold", 0.0) or 0.0))
        ruin_pos = tuple(int(v) for v in tuple(reward_data.get("ruin_pos", ())))
        marker_pos = tuple(int(v) for v in tuple(reward_data.get("marker_pos", ())))

        if ruin_enemy_id in self.cleared_ruin_enemy_ids:
            return {
                "ruin_cleared": False,
                "ruin_reward_applied": False,
                "ruin_reward_enemy_id": ruin_enemy_id,
                "ruin_reward_title": ruin_title,
                **self._ruin_progress_info(),
            }

        self.cleared_ruin_enemy_ids.add(ruin_enemy_id)
        self.gold = max(0.0, float(self.gold or 0.0)) + gold_reward
        artifact_grant_info: Dict[str, object] = {}
        if item_name:
            artifact_grant_info = self._add_hero_item(item_name)

        self.last_ruin_reward = {
            "enemy_id": ruin_enemy_id,
            "title": ruin_title,
            "item": item_name,
            "gold": gold_reward,
            "ruin_position": ruin_pos,
            "marker_position": marker_pos,
        }

        ruin_label = ruin_title or f"Руины {ruin_pos}"
        reward_parts = [f"+{gold_reward:g} gold"]
        if item_name:
            reward_parts.append(item_name)
        self._log(f"{ruin_label}: получено {', '.join(reward_parts)}")

        return {
            "ruin_cleared": True,
            "ruin_reward_applied": True,
            "ruin_reward_enemy_id": ruin_enemy_id,
            "ruin_reward_title": ruin_title,
            "ruin_reward_item": item_name,
            "ruin_reward_gold": gold_reward,
            "ruin_reward_ruin_pos": ruin_pos,
            "ruin_reward_marker_pos": marker_pos,
            **artifact_grant_info,
            **self._ruin_progress_info(),
        }

    def _is_sell_only_hero_item(self, entry: object) -> bool:
        item_name = self._hero_item_name(entry)
        if not item_name or self._is_useful_hero_item_name(item_name):
            return False
        return self._hero_item_sell_value(entry) > 0

    def _sell_sell_only_heroitems_at_merchant(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        site_names = self._merchant_sites_at_position(position)
        sale_info: Dict[str, object] = {
            "merchant_auto_sale": False,
            "merchant_sold_items": [],
            "merchant_sold_items_count": 0,
            "merchant_sale_gold": 0.0,
            "merchant_sale_reward": 0.0,
        }
        if not site_names or not self.heroitems:
            return sale_info

        sold_items: List[Dict[str, object]] = []
        kept_items: List[object] = []
        total_gold = 0
        for entry in self.heroitems:
            if not self._is_sell_only_hero_item(entry):
                kept_items.append(entry)
                continue

            item_name = self._hero_item_name(entry)
            gold_value = self._hero_item_sell_value(entry)
            if gold_value <= 0:
                kept_items.append(entry)
                continue

            sold_items.append({"name": item_name, "gold": int(gold_value)})
            total_gold += int(gold_value)

        if not sold_items:
            return sale_info

        self.heroitems = kept_items
        self._invalidate_inventory_cache()
        self._sync_equipped_artifact_items()
        self._refresh_campaign_artifact_effects(log=False)
        self.gold = max(0.0, float(self.gold or 0.0)) + float(total_gold)
        sale_reward = float(len(sold_items)) * float(self.reward_sell_junk_item)
        sale_info.update(
            {
                "merchant_auto_sale": True,
                "merchant_sold_items": sold_items,
                "merchant_sold_items_count": int(len(sold_items)),
                "merchant_sale_gold": float(total_gold),
                "merchant_sale_reward": float(sale_reward),
            }
        )

        sold_names = ", ".join(str(item["name"]) for item in sold_items)
        self._log(
            f"Торговец {', '.join(site_names)}: продано {len(sold_items)} предметов "
            f"на {total_gold} gold ({sold_names})"
        )
        return sale_info

    def _combat_potion_reward_value(self) -> float:
        return float(self.reward_defeat_enemy)

    def _count_collected_combat_potions(self, collected_chests: List[Dict]) -> int:
        tracked_items = {
            self.INVULNERABILITY_POTION_ITEM_NAME,
            self.STRENGTH_POTION_ITEM_NAME,
        }
        count = 0
        for chest_data in collected_chests:
            for item_name in chest_data.get("items", []):
                if str(item_name or "") in tracked_items:
                    count += 1
        return int(count)

    def _active_invulnerability_bonus_for_position(self, position: int) -> int:
        return (
            int(self.INVULNERABILITY_POTION_ARMOR_BONUS)
            if int(position) in self.active_invulnerability_potion_positions
            else 0
        )

    def _active_strength_multiplier_for_position(self, position: int) -> float:
        return (
            float(self.STRENGTH_POTION_DAMAGE_MULTIPLIER)
            if int(position) in self.active_strength_potion_positions
            else 1.0
        )

    def _clear_expired_combat_potion_effects(self) -> None:
        if self.active_invulnerability_potion_positions:
            self._log(
                "Эффект зелья неуязвимости завершён на позициях "
                f"{sorted(self.active_invulnerability_potion_positions)}"
            )
        if self.active_strength_potion_positions:
            self._log(
                "Эффект зелья силы завершён на позициях "
                f"{sorted(self.active_strength_potion_positions)}"
            )
        if self.active_energy_elixir_positions:
            self._log(
                "Эффект эликсира энергии завершён на позициях "
                f"{sorted(self.active_energy_elixir_positions)}"
            )
        if self.active_haste_elixir_positions:
            self._log(
                "Эффект эликсира быстроты завершён на позициях "
                f"{sorted(self.active_haste_elixir_positions)}"
            )
        if self.active_fire_ward_positions:
            self._log(
                "Эффект защиты от магии Огня завершён на позициях "
                f"{sorted(self.active_fire_ward_positions)}"
            )
        if self.active_earth_ward_positions:
            self._log(
                "Эффект защиты от магии Земли завершён на позициях "
                f"{sorted(self.active_earth_ward_positions)}"
            )
        if self.active_water_ward_positions:
            self._log(
                "Эффект защиты от магии Воды завершён на позициях "
                f"{sorted(self.active_water_ward_positions)}"
            )
        if self.active_air_ward_positions:
            self._log(
                "Эффект защиты от магии Воздуха завершён на позициях "
                f"{sorted(self.active_air_ward_positions)}"
            )
        self.active_invulnerability_potion_positions.clear()
        self.active_strength_potion_positions.clear()
        self.active_energy_elixir_positions.clear()
        self.active_haste_elixir_positions.clear()
        self.active_fire_ward_positions.clear()
        self.active_earth_ward_positions.clear()
        self.active_water_ward_positions.clear()
        self.active_air_ward_positions.clear()

    def get_bottle_inventory_counters(self) -> Dict[str, int]:
        max_heal = self._max_heal_bottles_available()
        max_healing = self._max_healing_bottles_available()
        max_revive = self._max_revive_bottles_available()
        heal_used = int(self.heal_bottles_used or 0)
        healing_used = int(self.healing_bottles_used or 0)
        revive_used = int(self.revive_bottles_used or 0)
        return {
            "max_heal": max_heal,
            "heal_left": max(0, max_heal - heal_used),
            "max_healing": max_healing,
            "healing_left": max(0, max_healing - healing_used),
            "ointment_left": self._count_hero_item(self.HEALING_OINTMENT_ITEM_NAME),
            "max_revive": max_revive,
            "revive_left": max(0, max_revive - revive_used),
            "invulnerability_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
        }

    @classmethod
    def _is_within_chest_pickup_range(
        cls,
        agent_pos: Tuple[int, int],
        chest_pos: Tuple[int, int],
    ) -> bool:
        dx = abs(int(agent_pos[0]) - int(chest_pos[0]))
        dy = abs(int(agent_pos[1]) - int(chest_pos[1]))
        return max(dx, dy) <= int(cls.CHEST_PICKUP_RADIUS)

    def _apply_chest_item_rewards(self, item_name: str) -> List[str]:
        granted_items: List[str] = []
        if item_name == self.BONUS_SMALL_HEAL_SOURCE_ITEM:
            self.extra_healing_bottles = max(0, int(self.extra_healing_bottles or 0)) + 1
            granted_items.append(self.BONUS_SMALL_HEAL_ITEM_NAME)
        if item_name == self.BONUS_REVIVE_SOURCE_ITEM:
            self.extra_revive_bottles = max(0, int(self.extra_revive_bottles or 0)) + 1
            granted_items.append(self.BONUS_REVIVE_ITEM_NAME)
        return granted_items

    @classmethod
    def _is_auto_consumed_chest_item(cls, item_name: str) -> bool:
        return str(item_name or "") in cls.AUTO_CONSUMED_CHEST_ITEMS

    def _collect_adjacent_chests(self) -> List[Dict]:
        if not self.chests:
            return []

        agent_pos = tuple(self.grid_env.agent_pos)
        collected: List[Dict] = []
        for chest_pos, item_names in sorted(self.chests.items(), key=lambda entry: entry[0]):
            if not self._is_within_chest_pickup_range(agent_pos, chest_pos):
                continue
            loot_items = [str(item_name) for item_name in item_names if str(item_name)]
            collected.append(
                {
                    "pos": tuple(chest_pos),
                    "items": loot_items,
                    "granted_items": [],
                }
            )

        if not collected:
            return []

        for chest_data in collected:
            chest_pos = tuple(chest_data["pos"])
            self.chests.pop(chest_pos, None)
            granted_items: List[str] = []
            for item_name in chest_data["items"]:
                if not self._is_auto_consumed_chest_item(item_name):
                    self._add_hero_item(item_name)
                granted_items.extend(self._apply_chest_item_rewards(item_name))
                self._log(f"Сундук на {chest_pos}: получен предмет '{item_name}'")
            for granted_item in granted_items:
                self._add_hero_item(granted_item)
                self._log(
                    f"Сундук на {chest_pos}: бонусом получен предмет '{granted_item}'"
                )
            chest_data["granted_items"] = granted_items

        self._sync_grid_chest_positions()
        return collected

    def _sync_moves_per_turn_with_hero(
        self,
        units: Optional[List[Dict]] = None,
        *,
        refill: bool = False,
        grant_delta: bool = False,
    ) -> None:
        previous_cap = int(self.moves_per_turn)
        new_cap = int(self.MOVES_PER_TURN) + self._hero_move_bonus(units=units)
        self.moves_per_turn = new_cap
        if refill:
            self.moves = new_cap
            return
        if grant_delta and new_cap > previous_cap:
            self.moves = min(new_cap, int(self.moves) + (new_cap - previous_cap))
        elif int(self.moves) > new_cap:
            self.moves = new_cap

    def _heal_unit_to_full_at_position(self, position: int) -> Tuple[float, Optional[str]]:
        """Лечит одного юнита BLUE на позиции в пределах золота с тарифом по Level."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)

            if max_hp <= 0 or hp <= 0 or hp >= max_hp:
                return 0.0, None

            missing_hp = max_hp - hp
            gold_available = max(0.0, float(self.gold or 0.0))
            if gold_available <= 0.0:
                return 0.0, None

            gold_per_hp = self._castle_heal_gold_per_hp(unit)
            if gold_per_hp <= 0.0:
                return 0.0, None

            healed = min(missing_hp, gold_available / gold_per_hp)
            gold_spent = healed * gold_per_hp
            new_hp = hp + healed
            unit["hp"] = new_hp
            unit["health"] = new_hp
            self.gold = max(0.0, gold_available - gold_spent)

            try:
                level_for_log = int(round(float(unit.get("Level", 1) or 1)))
            except (TypeError, ValueError):
                level_for_log = 1

            name = unit.get("name", f"pos_{position}")
            self._log(
                f"Heal tile: {name} (pos {position}) HP {hp:.1f} -> {new_hp:.1f} "
                f"(+{healed:.1f}, level={level_for_log}, "
                f"cost={gold_spent:.1f} gold, gold_left={self.gold:.1f})"
            )
            return healed, name

        return 0.0, None

    def _step_heal_in_castle(self, action: int):
        """Лечит выбранного юнита BLUE на клетках лечения с ценой за HP по Level."""
        idx = action - self.GRID_CASTLE_HEAL_ACTION_START
        target_pos = (
            self.CASTLE_HEAL_POSITIONS[idx]
            if 0 <= idx < len(self.CASTLE_HEAL_POSITIONS)
            else None
        )

        grid_obs = self._get_grid_obs()
        terminated = False
        truncated = False

        healed_amount, unit_name = (0.0, None)
        gold_spent = 0.0
        gold_before = float(self.gold)
        temple_built = self._has_temple_built()
        missing_temple = not temple_built
        if self._is_at_castle() and temple_built and target_pos is not None:
            healed_amount, unit_name = self._heal_unit_to_full_at_position(position=target_pos)
            gold_spent = max(0.0, gold_before - float(self.gold))
        reward = max(0.0, float(healed_amount)) * self.reward_castle_heal_per_hp

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "castle_heal_action": True,
            "castle_pos_required": self.CASTLE_POS,
            "castle_heal_tiles": self.castle_heal_tiles,
            "castle_heal_available": self._is_at_castle() and temple_built,
            "castle_heal_target_pos": target_pos,
            "temple_built": temple_built,
            "missing_temple": missing_temple,
            "healed_amount": healed_amount,
            "healed_unit_name": unit_name,
            "gold_spent": gold_spent,
            "castle_heal_reward": reward,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _ensure_map_layout_consistency(self) -> None:
        """Fail fast if generated map features overlap on forbidden tiles."""
        heal_tiles = {tuple(tile) for tile in self.castle_heal_tiles}
        enemy_tiles = {tuple(pos) for pos in self.grid_env.enemy_positions.values()}
        obstacle_tiles = {tuple(pos) for pos in self.grid_env.obstacle_positions}
        chest_tiles = {tuple(pos) for pos in getattr(self, "_static_chests", {}).keys()}

        obstacle_enemy_overlap = obstacle_tiles & enemy_tiles
        if obstacle_enemy_overlap:
            raise ValueError(
                f"Obstacle tiles overlap enemy tiles: {sorted(obstacle_enemy_overlap)}"
            )

        obstacle_heal_overlap = obstacle_tiles & heal_tiles
        if obstacle_heal_overlap:
            raise ValueError(
                f"Obstacle tiles overlap heal tiles: {sorted(obstacle_heal_overlap)}"
            )

        chest_enemy_overlap = chest_tiles & enemy_tiles
        if chest_enemy_overlap:
            raise ValueError(
                f"Chest tiles overlap enemy tiles: {sorted(chest_enemy_overlap)}"
            )

        chest_heal_overlap = chest_tiles & heal_tiles
        if chest_heal_overlap:
            raise ValueError(
                f"Chest tiles overlap heal tiles: {sorted(chest_heal_overlap)}"
            )

        chest_obstacle_overlap = chest_tiles & obstacle_tiles
        if chest_obstacle_overlap:
            raise ValueError(
                f"Chest tiles overlap obstacle tiles: {sorted(chest_obstacle_overlap)}"
            )

        reachable_targets = set(enemy_tiles)
        reachable_targets.update(heal_tiles)
        reachable_targets.update(chest_tiles)
        if not are_targets_reachable(
            self.grid_size,
            self.CASTLE_POS,
            obstacle_tiles,
            reachable_targets,
        ):
            raise ValueError("Obstacle tiles block path to at least one enemy or heal tile")

    def _castle_revive_cost_at_position(self, position: int) -> float:
        """Возвращает стоимость воскрешения мёртвого юнита на позиции или 0.0."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp > 0:
                return 0.0
            return float(self._castle_revive_gold_cost(unit))
        return 0.0

    def _revive_unit_with_gold_at_position(
        self,
        position: int,
    ) -> Tuple[bool, Optional[str], float]:
        """Воскрешает юнита за золото на клетке лечения, если хватает средств."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp > 0:
                return False, None, 0.0

            revive_cost = float(self._castle_revive_gold_cost(unit))
            gold_available = max(0.0, float(self.gold or 0.0))
            if revive_cost <= 0.0 or gold_available < revive_cost:
                return False, None, 0.0

            unit["hp"] = 1.0
            unit["health"] = 1.0
            self.gold = max(0.0, gold_available - revive_cost)
            name = unit.get("name", f"pos_{position}")
            self._log(
                f"Heal tile revive: {name} (pos {position}) revived with 1 HP "
                f"(cost={revive_cost:.1f} gold, gold_left={self.gold:.1f})"
            )
            return True, name, revive_cost

        return False, None, 0.0

    def _step_revive_in_castle(self, action: int):
        """Воскрешает выбранного юнита BLUE за золото на клетках лечения."""
        idx = action - self.GRID_CASTLE_REVIVE_ACTION_START
        target_pos = (
            self.CASTLE_REVIVE_POSITIONS[idx]
            if 0 <= idx < len(self.CASTLE_REVIVE_POSITIONS)
            else None
        )

        grid_obs = self._get_grid_obs()
        terminated = False
        truncated = False

        revive_cost = 0.0
        target_unit_name = None
        if target_pos is not None:
            revive_cost = self._castle_revive_cost_at_position(target_pos)
            for unit in self._get_blue_state():
                if int(unit.get("position", -1)) == target_pos:
                    target_unit_name = unit.get("name", f"pos_{target_pos}")
                    break

        temple_built = self._has_temple_built()
        missing_temple = not temple_built
        revived, revived_unit_name, gold_spent = (False, None, 0.0)
        if self._is_at_castle() and temple_built and target_pos is not None:
            revived, revived_unit_name, gold_spent = self._revive_unit_with_gold_at_position(
                position=target_pos
            )
        reward = self.reward_castle_revive if revived else 0.0

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "castle_revive_action": True,
            "castle_heal_tiles": self.castle_heal_tiles,
            "temple_built": temple_built,
            "missing_temple": missing_temple,
            "castle_revive_available": (
                self._is_at_castle()
                and temple_built
                and target_pos is not None
                and self._can_castle_revive_position(target_pos)
            ),
            "castle_revive_target_pos": target_pos,
            "target_unit_name": target_unit_name,
            "revived": revived,
            "revived_unit_name": revived_unit_name,
            "revive_cost": revive_cost,
            "gold_spent": gold_spent,
            "castle_revive_reward": reward,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_hire_faction_unit(self, action: int):
        idx = action - self.GRID_HIRE_ACTION_START
        option = self._hire_option(idx)

        grid_obs = self._get_grid_obs()
        terminated = False
        truncated = False

        unit_name = None if option is None else str(option.get("name", "") or "").strip()
        hire_cost = 0.0
        hire_role = None if option is None else str(option.get("role", "") or "").strip().lower()
        hire_target_pos = None
        if option is not None:
            try:
                hire_cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
            except (TypeError, ValueError):
                hire_cost = 0.0
            positions = self._hire_positions_for_role(hire_role)
            if positions:
                hire_target_pos = self._find_first_empty_blue_position(positions)

        hire_available_before = self._can_hire_faction_unit_option(option)
        hired, hired_unit_name, hired_position, gold_spent = self._hire_faction_unit_option(option)

        hero = self._resolve_travel_hero()
        hero_needaunit = self._resolve_hero_needaunit(hero) if hero is not None else 0
        reward = self._hire_reward_value() if hired else 0.0

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "hire_action": True,
            "hire_action_idx": int(idx),
            "hire_unit_name": unit_name,
            "hire_role": hire_role,
            "hire_cost": float(hire_cost),
            "hire_target_pos": hire_target_pos,
            "hire_available": bool(hire_available_before),
            "hired": bool(hired),
            "hired_units_count": int(1 if hired else 0),
            "hired_unit_name": hired_unit_name,
            "hired_position": hired_position,
            "gold_spent": float(gold_spent),
            "hire_reward": float(reward),
            "hero_needaunit": int(hero_needaunit),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _normalize_realcapital(self, value: int) -> int:
        """??????????? Realcapital: ????????? ?????? ???????? 1-5, ????? ?????? 1 (???????)."""
        try:
            realcapital_value = int(value)
        except (TypeError, ValueError):
            return 1
        return realcapital_value if realcapital_value in (1, 2, 3, 4, 5) else 1

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _get_building_keys(self, buildings: Dict) -> List[str]:
        """?????????? ????? ?????? (??? ????????? ????-?????? ????? 'alredybuilt')."""
        keys: List[str] = []
        for key, entry in buildings.items():
            if isinstance(entry, dict) and entry.get("id"):
                keys.append(key)
        return keys

    def _get_spell_keys(self, spells: Dict) -> List[str]:
        keys: List[str] = []
        for key, entry in spells.items():
            if isinstance(entry, dict) and entry.get("id"):
                keys.append(key)
        return keys

    def _reset_buildings_state(self) -> None:
        """Сбрасывает состояние зданий к эталону из Buildings.py."""
        for key, template in BUILDINGS_D2_TEMPLATE.items():
            target = BUILDINGS_D2.get(key)
            if not isinstance(target, dict) or not isinstance(template, dict):
                continue
            target.clear()
            target.update(deepcopy(template))
            self._scale_building_gold_costs_in_place(target)

    def _reset_spells_state(self) -> None:
        for key, template in SPELLS_D2_TEMPLATE.items():
            target = SPELLS_D2.get(key)
            if not isinstance(target, dict) or not isinstance(template, dict):
                continue
            target.clear()
            target.update(deepcopy(template))
            for entry in target.values():
                if isinstance(entry, dict):
                    entry["learned"] = int(entry.get("learned", 0) or 0)

    @classmethod
    def _scale_building_gold_costs_in_place(cls, buildings: Dict) -> None:
        cost_multiplier = max(0.0, float(cls.BUILDING_GOLD_COST_MULTIPLIER))
        if cost_multiplier == 1.0:
            return

        for entry in buildings.values():
            if not isinstance(entry, dict) or "gold" not in entry:
                continue
            try:
                base_gold = float(entry.get("gold", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            entry["gold"] = base_gold * cost_multiplier

    def _building_cost_multiplier(self) -> float:
        return (
            float(self.TYPEOFLORD_THREE_BUILDING_COST_MULTIPLIER)
            if int(self.typeoflord) == 3
            else 1.0
        )

    def _get_building_gold_cost(self, building: Optional[Dict[str, object]]) -> float:
        if not isinstance(building, dict):
            return 0.0
        try:
            base_cost = max(0.0, float(building.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            base_cost = 0.0
        return base_cost * float(self._building_cost_multiplier())

    def _current_max_building_gold_cost(self) -> float:
        max_price = 1.0
        for build_key in self.building_keys:
            max_price = max(
                max_price,
                self._get_building_gold_cost(self.active_buildings.get(build_key)),
            )
        return float(max_price)

    def get_built_building_names(self) -> List[str]:
        """Возвращает названия построенных зданий активной фракции в порядке ключей."""
        built_names: List[str] = []
        for key in self.building_keys:
            building = self.active_buildings.get(key)
            if not isinstance(building, dict):
                continue
            built_flag = building.get("Build", building.get("built", 0))
            try:
                is_built = int(built_flag or 0) == 1
            except (TypeError, ValueError):
                is_built = False
            if not is_built:
                continue
            name = str(building.get("name", "") or "").strip()
            if name:
                built_names.append(name)
        return built_names

    @classmethod
    def _spell_costs_from_entry(cls, spell: Dict, prefix: str) -> Dict[str, float]:
        costs: Dict[str, float] = {}
        for suffix, mana_kind in cls.SPELL_COST_MANA_KIND_BY_SUFFIX.items():
            raw_value = spell.get(f"{prefix}_{suffix}", 0.0) if isinstance(spell, dict) else 0.0
            try:
                value = max(0.0, float(raw_value or 0.0))
            except (TypeError, ValueError):
                value = 0.0
            costs[str(mana_kind)] = value
        return costs

    def _spell_learning_cost_multiplier(self) -> float:
        return (
            float(self.TYPEOFLORD_TWO_SPELL_LEARNING_COST_MULTIPLIER)
            if int(self.typeoflord) == 2
            else 1.0
        )

    def _get_spell_learning_costs(self, spell: Dict) -> Dict[str, float]:
        base_costs = self._spell_costs_from_entry(spell, prefix="learn")
        multiplier = float(self._spell_learning_cost_multiplier())
        if abs(multiplier - 1.0) <= 1e-9:
            return base_costs
        return {
            str(mana_kind): max(0.0, float(required_amount) * multiplier)
            for mana_kind, required_amount in base_costs.items()
        }

    def _get_spell_use_costs(self, spell: Dict) -> Dict[str, float]:
        return self._spell_costs_from_entry(spell, prefix="use")

    def _same_spell_cast_limit_per_turn(self) -> int:
        if int(self.typeoflord) == 2:
            return int(self.TYPEOFLORD_TWO_SAME_SPELL_CASTS_PER_TURN)
        return 1

    def _spell_cast_count_this_turn(self, spell_key: str) -> int:
        normalized_spell_key = str(spell_key or "")
        if not normalized_spell_key:
            return 0
        return max(
            0,
            int(self.spell_cast_counts_by_id_this_turn.get(normalized_spell_key, 0) or 0),
        )

    def _spell_cast_limit_reached_this_turn(self, spell_key: str) -> bool:
        return self._spell_cast_count_this_turn(spell_key) >= int(
            self._same_spell_cast_limit_per_turn()
        )

    def _spell_shop_cast_limit_reached_this_turn(self, spell_key: str) -> bool:
        return self._spell_cast_count_this_turn(spell_key) >= 1

    def _register_spell_cast_this_turn(self, spell_key: str) -> None:
        normalized_spell_key = str(spell_key or "")
        if not normalized_spell_key:
            return
        self.spell_cast_counts_by_id_this_turn[normalized_spell_key] = (
            self._spell_cast_count_this_turn(normalized_spell_key) + 1
        )

    @classmethod
    def _legion_damage_spell_ids(cls) -> Tuple[str, ...]:
        return tuple(
            str(spec.get("id", "") or "")
            for spec in cls.LEGION_DAMAGE_SPELL_ACTION_SPECS
            if str(spec.get("id", "") or "")
        )

    @classmethod
    def _map_offensive_spell_action_specs_for_capital(
        cls,
        capital: Optional[int],
    ) -> Tuple[Dict[str, object], ...]:
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            return ()
        specs = cls.MAP_OFFENSIVE_SPELL_ACTION_SPECS_BY_CAPITAL.get(capital_value, ())
        return tuple(specs)

    @classmethod
    def _map_offensive_spell_action_max_count(cls) -> int:
        return max(
            [0]
            + [
                len(tuple(specs))
                for specs in cls.MAP_OFFENSIVE_SPELL_ACTION_SPECS_BY_CAPITAL.values()
            ]
        )

    def _current_map_offensive_spell_action_specs(self) -> Tuple[Dict[str, object], ...]:
        return self._map_offensive_spell_action_specs_for_capital(self.Realcapital)

    @classmethod
    def _map_offensive_spell_ids_for_capital(cls, capital: Optional[int]) -> Tuple[str, ...]:
        return tuple(
            str(spec.get("id", "") or "")
            for spec in cls._map_offensive_spell_action_specs_for_capital(capital)
            if str(spec.get("id", "") or "")
        )

    @classmethod
    def _map_offensive_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(spec.get("id", "") or ""): dict(spec)
            for specs in cls.MAP_OFFENSIVE_SPELL_ACTION_SPECS_BY_CAPITAL.values()
            for spec in tuple(specs)
            if str(spec.get("id", "") or "")
        }

    def _map_offensive_spell_spec(self, spell_key: str) -> Dict[str, object]:
        return dict(self._map_offensive_spell_specs_by_id().get(str(spell_key or ""), {}))

    def _map_offensive_spell_targets_untargetable_stacks(self, spell_key: str) -> bool:
        spell_spec = self._map_offensive_spell_spec(spell_key)
        return bool(spell_spec.get("targets_untargetable_stacks", False))

    def _get_map_offensive_spell_target(
        self,
        spell_key: str,
        *,
        for_mask: bool = False,
    ) -> Optional[Dict[str, object]]:
        spell_targetable_only = not self._map_offensive_spell_targets_untargetable_stacks(
            spell_key
        )
        if for_mask:
            return self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=spell_targetable_only
            )
        return self.get_nearest_enemy_stack(spell_targetable_only=spell_targetable_only)

    @classmethod
    def _map_support_spell_action_specs_for_capital(
        cls,
        capital: Optional[int],
    ) -> Tuple[Dict[str, object], ...]:
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            return ()
        specs = cls.MAP_SUPPORT_SPELL_ACTION_SPECS_BY_CAPITAL.get(capital_value, ())
        return tuple(specs)

    @classmethod
    def _map_support_spell_action_max_count(cls) -> int:
        return max(
            [0]
            + [
                len(tuple(specs))
                for specs in cls.MAP_SUPPORT_SPELL_ACTION_SPECS_BY_CAPITAL.values()
            ]
        )

    def _current_map_support_spell_action_specs(self) -> Tuple[Dict[str, object], ...]:
        return self._map_support_spell_action_specs_for_capital(self.Realcapital)

    @classmethod
    def _map_support_spell_ids_for_capital(cls, capital: Optional[int]) -> Tuple[str, ...]:
        return tuple(
            str(spec.get("id", "") or "")
            for spec in cls._map_support_spell_action_specs_for_capital(capital)
            if str(spec.get("id", "") or "")
        )

    @classmethod
    def _map_support_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(spec.get("id", "") or ""): dict(spec)
            for specs in cls.MAP_SUPPORT_SPELL_ACTION_SPECS_BY_CAPITAL.values()
            for spec in tuple(specs)
            if str(spec.get("id", "") or "")
        }

    def _map_support_spell_spec(self, spell_key: str) -> Dict[str, object]:
        return dict(self._map_support_spell_specs_by_id().get(str(spell_key or ""), {}))

    @classmethod
    def _map_castable_spell_ids_for_capital(cls, capital: Optional[int]) -> Tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [
                    *cls._map_offensive_spell_ids_for_capital(capital),
                    *cls._map_support_spell_ids_for_capital(capital),
                ]
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def _all_spell_entries_by_id(cls) -> Dict[str, Dict[str, object]]:
        all_spells: Dict[str, Dict[str, object]] = {}
        for faction_spells in SPELLS_D2_TEMPLATE.values():
            if not isinstance(faction_spells, dict):
                continue
            for spell_key, spell_data in faction_spells.items():
                if not isinstance(spell_data, dict):
                    continue
                all_spells[str(spell_key)] = dict(spell_data)
        return all_spells

    def _spell_entry_by_id(self, spell_key: str) -> Optional[Dict[str, object]]:
        normalized_spell_key = str(spell_key or "")
        spell = self.active_spells.get(normalized_spell_key)
        if isinstance(spell, dict):
            return spell
        fallback_spell = self._all_spell_entries_by_id().get(normalized_spell_key)
        if isinstance(fallback_spell, dict):
            return dict(fallback_spell)
        return None

    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_supported_spell_definitions_by_id(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(entry.get("spell_id", "") or ""): dict(entry)
            for entry in SCROLL_ITEM_DEFINITIONS
            if bool(entry.get("supported", False))
            and str(entry.get("spell_id", "") or "")
        }

    def _scroll_spell_definition_by_id(self, spell_key: str) -> Dict[str, object]:
        return dict(
            self._scroll_supported_spell_definitions_by_id().get(str(spell_key or ""), {})
        )

    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_offensive_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        offensive_kinds = {"damage", "debuff", "summon_battle"}
        return {
            str(entry.get("spell_id", "") or ""): dict(entry)
            for entry in SCROLL_ITEM_DEFINITIONS
            if bool(entry.get("supported", False))
            and str(entry.get("spell_kind", "") or "") in offensive_kinds
        }

    @classmethod
    @lru_cache(maxsize=1)
    def _scroll_support_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(entry.get("spell_id", "") or ""): dict(entry)
            for entry in SCROLL_ITEM_DEFINITIONS
            if bool(entry.get("supported", False))
            and str(entry.get("spell_kind", "") or "")
            in {"moves", "heal", "buff", "ward", "health_bonus"}
        }

    @classmethod
    @lru_cache(maxsize=1)
    def _combined_map_offensive_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        combined = dict(cls._map_offensive_spell_specs_by_id())
        combined.update(cls._scroll_offensive_spell_specs_by_id())
        return combined

    @classmethod
    @lru_cache(maxsize=1)
    def _combined_map_support_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        combined = dict(cls._map_support_spell_specs_by_id())
        combined.update(cls._scroll_support_spell_specs_by_id())
        return combined

    def _spell_shop_spell_has_regular_action(self, spell_key: str) -> bool:
        return str(spell_key or "") in set(self._map_castable_spell_ids_for_capital(self.Realcapital))

    def _spell_shop_cast_spell_entries(self) -> Tuple[Dict[str, object], ...]:
        entries: List[Dict[str, object]] = []
        seen_spell_ids: set[str] = set()
        for spell_data in self.SPELL_SHOP_BUY_SPELLS:
            spell_id = str(spell_data.get("spell_id", "") or "")
            if (
                not spell_id
                or spell_id in seen_spell_ids
                or self._spell_shop_spell_has_regular_action(spell_id)
            ):
                continue
            spell_spec = self._map_offensive_spell_spec(spell_id)
            if not spell_spec:
                spell_spec = self._map_support_spell_spec(spell_id)
            if not spell_spec or not isinstance(self._spell_entry_by_id(spell_id), dict):
                continue
            entries.append(dict(spell_data))
            seen_spell_ids.add(spell_id)
            if len(entries) >= int(self.MAX_SPELL_SHOP_CAST_ACTIONS):
                break
        return tuple(entries)

    def _spell_shop_cast_spell_entry(self, idx: int) -> Dict[str, object]:
        entries = self._spell_shop_cast_spell_entries()
        if 0 <= int(idx) < len(entries):
            return dict(entries[int(idx)])
        return {}

    @classmethod
    def _legion_damage_spell_specs_by_id(cls) -> Dict[str, Dict[str, object]]:
        return {
            str(spec.get("id", "") or ""): dict(spec)
            for spec in cls.LEGION_DAMAGE_SPELL_ACTION_SPECS
            if str(spec.get("id", "") or "")
        }

    @classmethod
    def _legion_damage_spell_damage_by_id(cls) -> Dict[str, float]:
        return {
            str(spec.get("id", "") or ""): float(spec.get("damage", 0.0) or 0.0)
            for spec in cls.LEGION_DAMAGE_SPELL_ACTION_SPECS
            if str(spec.get("kind", "") or "") == "damage"
        }

    @classmethod
    def _legion_damage_spell_element_by_id(cls) -> Dict[str, str]:
        return {
            str(spec.get("id", "") or ""): str(spec.get("damage_type", "") or "")
            for spec in cls.LEGION_DAMAGE_SPELL_ACTION_SPECS
            if str(spec.get("kind", "") or "") == "damage"
        }

    def _is_legions_capital(self) -> bool:
        return int(self.Realcapital) == 2

    def _is_empire_capital(self) -> bool:
        return int(self.Realcapital) == 1

    def _is_undead_hordes_capital(self) -> bool:
        return int(self.Realcapital) == 4

    def _create_initial_enemy_team_states(self) -> Dict[int, List[Dict]]:
        return {
            int(enemy_id): deepcopy(team)
            for enemy_id, team in ENEMY_CONFIGS.items()
        }

    def _get_enemy_team_state(self, enemy_id: Optional[int]) -> List[Dict]:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return []
        team = self.enemy_team_states.get(normalized_enemy_id)
        if team is None:
            team = deepcopy(ENEMY_CONFIGS.get(normalized_enemy_id, []))
            self.enemy_team_states[normalized_enemy_id] = team
        return team

    def _has_named_building_built(self, building_name: str) -> bool:
        normalized_name = str(building_name or "").strip()
        if not normalized_name:
            return False
        for build_key in self.building_keys:
            building = self.active_buildings.get(build_key)
            if not isinstance(building, dict):
                continue
            if str(building.get("name", "") or "").strip() != normalized_name:
                continue
            try:
                return int(building.get("Build", building.get("built", 0)) or 0) == 1
            except (TypeError, ValueError):
                return False
        return False

    def _has_magic_tower_built(self) -> bool:
        return self._has_named_building_built(self.MAGIC_TOWER_BUILDING_NAME)

    def _has_temple_built(self) -> bool:
        return self._has_named_building_built(self.TEMPLE_BUILDING_NAME)

    def _has_mana_for_costs(self, costs: Dict[str, float]) -> bool:
        for mana_kind, required_amount in costs.items():
            attr_name = str(self.MANA_ATTR_BY_KIND.get(str(mana_kind), "") or "")
            if not attr_name:
                continue
            current_amount = float(getattr(self, attr_name, 0.0) or 0.0)
            if current_amount + 1e-9 < float(required_amount):
                return False
        return True

    def _spend_mana_costs(self, costs: Dict[str, float]) -> None:
        for mana_kind, required_amount in costs.items():
            attr_name = str(self.MANA_ATTR_BY_KIND.get(str(mana_kind), "") or "")
            if not attr_name:
                continue
            current_amount = float(getattr(self, attr_name, 0.0) or 0.0)
            setattr(self, attr_name, max(0.0, current_amount - float(required_amount)))

    def get_learned_spell_descriptions(self) -> List[str]:
        learned_descriptions: List[str] = []
        for spell_key in self.spell_keys:
            spell = self.active_spells.get(spell_key)
            if not isinstance(spell, dict):
                continue
            try:
                is_learned = int(spell.get("learned", 0) or 0) == 1
            except (TypeError, ValueError):
                is_learned = False
            if not is_learned:
                continue
            description = str(spell.get("description", "") or "").strip()
            if description:
                learned_descriptions.append(description)
        return learned_descriptions

    def _spell_state_info(self) -> Dict[str, object]:
        learned_spells = self.get_learned_spell_descriptions()
        spell_shop_purchased_spells = [
            str(spell_data.get("name", "") or "")
            for spell_data in self.SPELL_SHOP_BUY_SPELLS
            if str(spell_data.get("spell_id", "") or "") in self.spell_shop_purchased_spell_ids
        ]
        info = {
            "learned_spells": list(learned_spells),
            "learned_spells_count": int(len(learned_spells)),
            "spells_total": int(len(self.spell_keys)),
            "spell_learning_locked": bool(self.spell_learning_locked),
            "magic_tower_built": bool(self._has_magic_tower_built()),
            "spell_shop_purchased_spells": spell_shop_purchased_spells,
            "spell_shop_purchased_spells_count": int(len(spell_shop_purchased_spells)),
        }
        info.update(self._scroll_state_info())
        return info

    def _total_scroll_count(self) -> int:
        self._ensure_inventory_cache()
        return int(self._total_scroll_count_cache)

    def _scroll_state_info(self) -> Dict[str, object]:
        self._ensure_inventory_cache()
        inventory_entries = self._scroll_inventory_entries_cache
        slot_entries = self._scroll_cast_slot_entries_cache
        return {
            "scroll_magic_unlocked": bool(self.scroll_magic_unlocked),
            "supported_scroll_types_count": int(len(self._available_scroll_spell_entries_cache)),
            "total_scroll_count": int(self._total_scroll_count_cache),
            "scroll_inventory": [
                {
                    "item_name": str(entry.get("item_name", "") or ""),
                    "spell_id": str(entry.get("spell_id", "") or ""),
                    "spell_name": str(entry.get("spell_name", "") or ""),
                    "copies": int(entry.get("copies", 0) or 0),
                    "supported": bool(entry.get("supported", False)),
                }
                for entry in inventory_entries
            ],
            "scroll_cast_slots": [
                {
                    "slot": int(idx),
                    "item_name": str(entry.get("item_name", "") or ""),
                    "spell_id": str(entry.get("spell_id", "") or ""),
                    "spell_name": str(entry.get("spell_name", "") or ""),
                    "copies": int(entry.get("copies", 0) or 0),
                    "spell_kind": str(entry.get("spell_kind", "") or ""),
                    "spell_level": int(entry.get("spell_level", 0) or 0),
                }
                for idx, entry in enumerate(slot_entries)
            ],
        }

    def _is_spell_learned(self, spell_key: str) -> bool:
        spell = self.active_spells.get(str(spell_key))
        if not isinstance(spell, dict):
            return False
        try:
            return int(spell.get("learned", 0) or 0) == 1
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _spell_level_from_entry(spell: Optional[Dict[str, object]]) -> int:
        if not isinstance(spell, dict):
            return 0
        try:
            return int(spell.get("level", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _is_spell_level_masked_by_typeoflord(self, spell_level: Optional[int]) -> bool:
        try:
            normalized_level = int(spell_level or 0)
        except (TypeError, ValueError):
            normalized_level = 0
        return normalized_level >= 5 and int(self.typeoflord) != 2

    def _is_spell_blocked_by_typeoflord(self, spell: Optional[Dict[str, object]]) -> bool:
        return self._is_spell_level_masked_by_typeoflord(self._spell_level_from_entry(spell))

    def _can_cast_legion_damage_spell(
        self,
        spell_key: str,
        *,
        nearest_enemy: Optional[Dict[str, object]] = None,
    ) -> bool:
        normalized_spell_key = str(spell_key or "")
        current_spell_ids = self._map_offensive_spell_ids_for_capital(self.Realcapital)
        if not current_spell_ids:
            return False
        if normalized_spell_key not in current_spell_ids:
            return False
        if self._spell_cast_limit_reached_this_turn(normalized_spell_key):
            return False
        spell = self.active_spells.get(normalized_spell_key)
        if not isinstance(spell, dict):
            return False
        spell_spec = self._map_offensive_spell_spec(normalized_spell_key)
        if self._is_spell_blocked_by_typeoflord(spell):
            return False
        if not self._is_spell_learned(normalized_spell_key):
            return False
        if str(spell_spec.get("kind", "") or "") == "summon_battle":
            summon_unit_name = str(spell_spec.get("summon_unit_name", "") or "").strip()
            if not summon_unit_name or not isinstance(
                self._find_unit_data_by_name(summon_unit_name),
                dict,
            ):
                return False
        if nearest_enemy is None:
            nearest_enemy = self._get_map_offensive_spell_target(normalized_spell_key)
        if nearest_enemy is None:
            return False
        target_enemy_id = nearest_enemy.get("enemy_id")
        if not self._enemy_team_has_living_units(target_enemy_id):
            return False
        return self._has_mana_for_costs(self._get_spell_use_costs(spell))

    def _can_cast_support_spell(self, spell_key: str) -> bool:
        normalized_spell_key = str(spell_key or "")
        current_spell_ids = self._map_support_spell_ids_for_capital(self.Realcapital)
        if not current_spell_ids:
            return False
        if normalized_spell_key not in current_spell_ids:
            return False
        if self._spell_cast_limit_reached_this_turn(normalized_spell_key):
            return False
        spell = self.active_spells.get(normalized_spell_key)
        if not isinstance(spell, dict):
            return False
        if self._is_spell_blocked_by_typeoflord(spell):
            return False
        if not self._is_spell_learned(normalized_spell_key):
            return False
        if not self._hero_stack_has_living_units():
            return False
        return self._has_mana_for_costs(self._get_spell_use_costs(spell))

    def _can_cast_spell_shop_spell(
        self,
        spell_key: str,
        *,
        nearest_targetable_enemy: Optional[Dict[str, object]] = None,
        nearest_any_enemy: Optional[Dict[str, object]] = None,
    ) -> bool:
        normalized_spell_key = str(spell_key or "")
        if not normalized_spell_key:
            return False
        if not self._spell_shop_spell_owned(normalized_spell_key):
            return False
        if self._spell_shop_spell_has_regular_action(normalized_spell_key):
            return False
        if self._spell_shop_cast_limit_reached_this_turn(normalized_spell_key):
            return False

        spell = self._spell_entry_by_id(normalized_spell_key)
        if not isinstance(spell, dict):
            return False

        offensive_spec = self._map_offensive_spell_spec(normalized_spell_key)
        if offensive_spec:
            if str(offensive_spec.get("kind", "") or "") == "summon_battle":
                summon_unit_name = str(offensive_spec.get("summon_unit_name", "") or "").strip()
                if not summon_unit_name or not isinstance(
                    self._find_unit_data_by_name(summon_unit_name),
                    dict,
                ):
                    return False
            nearest_enemy = (
                nearest_any_enemy
                if self._map_offensive_spell_targets_untargetable_stacks(normalized_spell_key)
                else nearest_targetable_enemy
            )
            if nearest_enemy is None:
                nearest_enemy = self._get_map_offensive_spell_target(normalized_spell_key)
            if nearest_enemy is None:
                return False
            target_enemy_id = nearest_enemy.get("enemy_id")
            if not self._enemy_team_has_living_units(target_enemy_id):
                return False
            return self._has_mana_for_costs(self._get_spell_use_costs(spell))

        support_spec = self._map_support_spell_spec(normalized_spell_key)
        if support_spec:
            if not self._hero_stack_has_living_units():
                return False
            return self._has_mana_for_costs(self._get_spell_use_costs(spell))

        return False

    @staticmethod
    def _spell_kind_is_offensive(spell_kind: str) -> bool:
        return str(spell_kind or "") in {"damage", "debuff", "summon_battle"}

    @staticmethod
    def _spell_kind_is_support(spell_kind: str) -> bool:
        return str(spell_kind or "") in {"moves", "heal", "buff", "ward", "health_bonus"}

    def _resolve_spell_target_enemy(
        self,
        spell_key: str,
        spell_spec: Dict[str, object],
        *,
        nearest_targetable_enemy: Optional[Dict[str, object]] = None,
        nearest_any_enemy: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict[str, object]]:
        normalized_spell_key = str(spell_key or "")
        spell_kind = str(spell_spec.get("spell_kind", spell_spec.get("kind", "")) or "")
        if not self._spell_kind_is_offensive(spell_kind):
            return None
        targets_untargetable = bool(
            spell_spec.get(
                "targets_untargetable_stacks",
                spell_kind == "summon_battle"
                or self._map_offensive_spell_targets_untargetable_stacks(normalized_spell_key),
            )
        )
        if targets_untargetable:
            return nearest_any_enemy or self.get_nearest_enemy_stack(
                spell_targetable_only=False
            )
        return nearest_targetable_enemy or self.get_nearest_enemy_stack(
            spell_targetable_only=True
        )

    def _can_cast_scroll_spell(
        self,
        slot_entry: Dict[str, object],
        *,
        nearest_targetable_enemy: Optional[Dict[str, object]] = None,
        nearest_any_enemy: Optional[Dict[str, object]] = None,
    ) -> bool:
        if not bool(self.scroll_magic_unlocked):
            return False
        if not isinstance(slot_entry, dict):
            return False
        item_name = str(slot_entry.get("item_name", "") or "")
        spell_key = str(slot_entry.get("spell_id", "") or "")
        spell_kind = str(slot_entry.get("spell_kind", "") or "")
        if (
            not item_name
            or not spell_key
            or not bool(slot_entry.get("supported", False))
            or self.count_scroll_item(item_name) <= 0
        ):
            return False
        if self._spell_kind_is_offensive(spell_kind):
            if spell_kind == "summon_battle":
                summon_unit_name = str(slot_entry.get("summon_unit_name", "") or "").strip()
                if not summon_unit_name or not isinstance(
                    self._find_unit_data_by_name(summon_unit_name),
                    dict,
                ):
                    return False
            nearest_enemy = self._resolve_spell_target_enemy(
                spell_key,
                slot_entry,
                nearest_targetable_enemy=nearest_targetable_enemy,
                nearest_any_enemy=nearest_any_enemy,
            )
            if nearest_enemy is None:
                return False
            return self._enemy_team_has_living_units(nearest_enemy.get("enemy_id"))
        if self._spell_kind_is_support(spell_kind):
            return self._hero_stack_has_living_units()
        return False

    def _cast_from_spell_spec(
        self,
        *,
        source: str,
        spell_key: str,
        spell_description: Optional[str],
        spell_spec: Dict[str, object],
        mana_costs: Optional[Dict[str, float]] = None,
        spend_mana: bool = False,
        enforce_turn_limit: bool = False,
        consume_item_name: Optional[str] = None,
        nearest_targetable_enemy: Optional[Dict[str, object]] = None,
        nearest_any_enemy: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        normalized_spell_key = str(spell_key or "")
        spell_kind = str(spell_spec.get("spell_kind", spell_spec.get("kind", "")) or "")
        mana_costs = dict(mana_costs or {})
        result: Dict[str, object] = {
            "source": str(source or ""),
            "spell_cast_applied": False,
            "spell_cast_executed": False,
            "spell_cast_reward": 0.0,
            "spell_units_affected": 0,
            "spell_heal_total": 0.0,
            "spell_moves_restored": 0.0,
            "moves_before_cast": float(self.moves or 0.0),
            "moves_after_cast": float(self.moves or 0.0),
            "target_enemy_id": None,
            "target_enemy_position": None,
            "target_enemy_reachable": False,
            "target_enemy_description": None,
            "spell_damage_total": 0.0,
            "spell_damage_units_hit": 0,
            "spell_damage_units_defeated": 0,
            "spell_damage_units_blocked": 0,
            "spell_enemy_defeated": False,
            "enemy_defeat_reward": 0.0,
            "enemy_defeat_reward_base": 0.0,
            "ruin_clear_bonus_reward": 0.0,
            "final_objective_reward": 0.0,
            "newly_captured_objective_cities": [],
            "newly_activated_settlement_territories": [],
            "battle_triggered": False,
            "battle_context_kind": None,
            "spell_effect_summary": (
                self._blue_stack_spell_effect_summary()
                if self._spell_kind_is_support(spell_kind)
                else {
                    "active_spell_ids": [],
                    "debuff_types": [],
                    "armor_delta": 0,
                    "damage_multiplier": 1.0,
                    "initiative_multiplier": 1.0,
                    "accuracy_multiplier": 1.0,
                }
            ),
            "scroll_item_name": str(consume_item_name or ""),
            "scroll_item_consumed": False,
            "scroll_copies_before": (
                int(self.count_scroll_item(str(consume_item_name or "")))
                if consume_item_name
                else 0
            ),
            "scroll_copies_after": (
                int(self.count_scroll_item(str(consume_item_name or "")))
                if consume_item_name
                else 0
            ),
            "reward": 0.0,
        }
        if spend_mana and mana_costs:
            self._spend_mana_costs(mana_costs)
        if enforce_turn_limit and normalized_spell_key:
            self._register_spell_cast_this_turn(normalized_spell_key)

        result["spell_cast_executed"] = True
        result["spell_cast_reward"] = float(self.reward_spell_cast)
        result["reward"] = float(result["reward"]) + float(self.reward_spell_cast)

        spell_damage = float(spell_spec.get("damage", 0.0) or 0.0)
        spell_damage_type = str(spell_spec.get("damage_type", "") or "") or None
        spell_debuff_type = str(spell_spec.get("debuff_type", "") or "") or None
        spell_buff_type = str(spell_spec.get("buff_type", "") or "") or None
        spell_resistance_type = str(spell_spec.get("resistance_type", "") or "") or None
        spell_heal_amount = float(spell_spec.get("heal_amount", 0.0) or 0.0)
        spell_health_delta = int(spell_spec.get("health_delta", 0) or 0)
        spell_moves_restore_fraction = float(
            spell_spec.get("moves_restore_fraction", 0.0) or 0.0
        )
        summon_unit_name = str(spell_spec.get("summon_unit_name", "") or "") or None

        nearest_enemy = self._resolve_spell_target_enemy(
            normalized_spell_key,
            spell_spec,
            nearest_targetable_enemy=nearest_targetable_enemy,
            nearest_any_enemy=nearest_any_enemy,
        )
        if nearest_enemy is not None:
            result["target_enemy_id"] = int(nearest_enemy.get("enemy_id", -1))
            result["target_enemy_position"] = tuple(nearest_enemy.get("position", ()))
            result["target_enemy_reachable"] = bool(nearest_enemy.get("reachable", False))
            result["target_enemy_description"] = str(
                nearest_enemy.get("description", "") or ""
            )

        target_enemy_id = result["target_enemy_id"]
        if spell_kind == "damage":
            (
                result["spell_damage_total"],
                result["spell_damage_units_hit"],
                result["spell_damage_units_defeated"],
                result["spell_damage_units_blocked"],
            ) = self._apply_damage_to_enemy_stack(
                target_enemy_id,
                spell_damage,
                spell_damage_type,
            )
            result["spell_cast_applied"] = int(result["spell_damage_units_hit"]) > 0
            result["spell_units_affected"] = int(result["spell_damage_units_hit"])
        elif spell_kind == "debuff":
            (
                result["spell_units_affected"],
                result["spell_effect_summary"],
            ) = self._apply_debuff_to_enemy_stack(target_enemy_id, normalized_spell_key)
            result["spell_cast_applied"] = int(result["spell_units_affected"]) > 0
        elif spell_kind == "summon_battle":
            summon_battle_team = self._build_summoned_blue_team(str(summon_unit_name or ""))
            if summon_battle_team is not None and target_enemy_id is not None:
                summon_blue_team, summoned_unit = summon_battle_team
                result["spell_cast_applied"] = True
                result["spell_units_affected"] = 1
                self.summon_hero_battle_bonus_enemy_ids_this_turn.add(int(target_enemy_id))
                self.summon_hero_battle_bonus_pending = True
                self.current_enemy_id = int(target_enemy_id)
                self.battle_origin_pos = tuple(self.grid_env.agent_pos)
                self.current_battle_context = {
                    "kind": "summon_spell",
                    "spell_key": normalized_spell_key,
                    "summoned_unit_name": str(summon_unit_name or ""),
                    "target_enemy_id": int(target_enemy_id),
                }
                self.mode = self.MODE_BATTLE
                self._log(
                    f'Применено заклинание "{spell_description}" по ближайшему вражескому отряду '
                    f"{target_enemy_id}: призван {summoned_unit['name']} для отдельного боя."
                )
                self._init_battle(int(target_enemy_id), blue_team=summon_blue_team)
                result["battle_triggered"] = True
                result["battle_context_kind"] = "summon_spell"
            else:
                result["spell_cast_applied"] = False
        elif spell_kind == "moves":
            result["spell_moves_restored"] = self._restore_grid_moves(
                spell_moves_restore_fraction
            )
            result["spell_units_affected"] = (
                1 if float(result["spell_moves_restored"]) > 0.0 else 0
            )
            result["spell_cast_applied"] = float(result["spell_moves_restored"]) > 0.0
            result["spell_effect_summary"] = self._blue_stack_spell_effect_summary()
        elif spell_kind == "heal":
            (
                result["spell_units_affected"],
                result["spell_heal_total"],
            ) = self._apply_heal_to_blue_stack(spell_heal_amount)
            result["spell_cast_applied"] = float(result["spell_heal_total"]) > 0.0
            result["spell_effect_summary"] = self._blue_stack_spell_effect_summary()
        elif spell_kind in {"buff", "ward", "health_bonus"}:
            (
                result["spell_units_affected"],
                result["spell_effect_summary"],
            ) = self._apply_support_spell_to_blue_stack(normalized_spell_key)
            result["spell_cast_applied"] = int(result["spell_units_affected"]) > 0

        if consume_item_name:
            if bool(result["spell_cast_applied"]) and self._consume_hero_item(consume_item_name):
                result["scroll_item_consumed"] = True
            result["scroll_copies_after"] = int(self.count_scroll_item(consume_item_name))

        if spell_description and spell_kind != "summon_battle":
            if spell_kind == "damage":
                self._log(
                    f'Применено заклинание "{spell_description}" по ближайшему вражескому отряду '
                    f"{target_enemy_id}: {float(result['spell_damage_total']):g} урона суммарно, "
                    f"задето {int(result['spell_damage_units_hit'])} юнитов, "
                    f"заблокировано {int(result['spell_damage_units_blocked'])}, "
                    f"добито {int(result['spell_damage_units_defeated'])}."
                )
            elif spell_kind == "debuff":
                self._log(
                    f'Применено заклинание "{spell_description}" по ближайшему вражескому отряду '
                    f"{target_enemy_id}: дебафф '{spell_debuff_type or 'unknown'}' на "
                    f"{int(result['spell_units_affected'])} живых юнитах до следующего игрового хода."
                )
            elif spell_kind == "moves":
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"восстановлено {float(result['spell_moves_restored']):g} ед. движения "
                    f"({float(result['moves_before_cast']):g} -> {float(self.moves or 0.0):g} "
                    f"из {float(self.moves_per_turn or 0.0):g})."
                )
            elif spell_kind == "heal":
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"исцелено {float(result['spell_heal_total']):g} HP суммарно на "
                    f"{int(result['spell_units_affected'])} юнитах."
                )
            elif spell_kind == "ward":
                ward_description = (
                    "защита от первой оружейной атаки"
                    if str(spell_resistance_type or "") == "Weapon"
                    else f"защита от магии {spell_resistance_type or 'unknown'}"
                )
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"{ward_description} до следующего игрового хода."
                )
            elif spell_kind == "health_bonus":
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"+{spell_health_delta} HP на {int(result['spell_units_affected'])} живых юнитах "
                    f"до следующего игрового хода."
                )
            elif spell_kind == "buff":
                self._log(
                    f'Применено заклинание "{spell_description}" на отряд героя: '
                    f"бафф '{spell_buff_type or 'unknown'}' на {int(result['spell_units_affected'])} "
                    f"живых юнитах до следующего игрового хода."
                )

        if (
            self._spell_kind_is_offensive(spell_kind)
            and spell_kind != "summon_battle"
            and target_enemy_id is not None
            and not self._enemy_team_has_living_units(target_enemy_id)
        ):
            result["spell_enemy_defeated"] = True
            result["enemy_defeat_reward"] = self._compute_enemy_defeat_reward(target_enemy_id)
            result["enemy_defeat_reward_base"] = float(self.reward_defeat_enemy)
            result["ruin_clear_bonus_reward"] = (
                float(self.reward_ruin_clear_bonus)
                if int(target_enemy_id or -1) in self.RUIN_REWARD_BY_ENEMY_ID
                else 0.0
            )
            result["reward"] = (
                float(result["reward"])
                + float(result["enemy_defeat_reward"])
                + float(result["ruin_clear_bonus_reward"])
            )
            self.grid_env.mark_enemy_defeated(target_enemy_id)
            self._clear_enemy_map_spell_effects_for_enemy(target_enemy_id)
            self._log(f"=== ВРАГ {target_enemy_id} УНИЧТОЖЕН ЗАКЛИНАНИЕМ НА КАРТЕ ===")

            objective_cities_captured_before = len(self.captured_objective_cities)
            result["newly_captured_objective_cities"] = self._capture_objective_city_if_cleared(
                target_enemy_id
            )
            result["newly_activated_settlement_territories"] = (
                self._activate_legions_settlement_territory_if_cleared(target_enemy_id)
            )
            if result["newly_captured_objective_cities"]:
                result["final_objective_reward"] = self._compute_final_objective_reward(
                    len(result["newly_captured_objective_cities"]),
                    captured_before=objective_cities_captured_before,
                )
                result["reward"] = float(result["reward"]) + float(
                    result["final_objective_reward"]
                )
            if result["newly_activated_settlement_territories"]:
                for settlement_name in result["newly_activated_settlement_territories"]:
                    self._log(
                        f"=== ЗЕМЛЯ ЛЕГИОНОВ НАЧИНАЕТ РАСПРОСТРАНЯТЬСЯ ИЗ ПОСЕЛЕНИЯ {settlement_name} ==="
                    )

        result["moves_after_cast"] = float(self.moves or 0.0)
        return result

    def _finalize_map_spell_step(
        self,
        *,
        reward: float,
        info: Dict[str, object],
        cast_result: Dict[str, object],
    ):
        if bool(cast_result.get("battle_triggered", False)) and self.mode == self.MODE_BATTLE:
            battle_obs = (
                self.battle_env._obs()
                if self.battle_env is not None
                else np.zeros(0, dtype=np.float32)
            )
            info["battle_triggered"] = True
            info["mode"] = "battle"
            info["enemy_id"] = self.current_enemy_id
            info["battle_context_kind"] = cast_result.get("battle_context_kind")
            return self._build_obs(battle_obs=battle_obs), reward, False, False, info

        if bool(cast_result.get("spell_enemy_defeated", False)):
            target_enemy_id = cast_result.get("target_enemy_id")
            info.update(self._grant_ruin_reward(target_enemy_id))
            info["objective_cities_captured_total"] = sorted(self.captured_objective_cities)
            newly_activated = list(
                cast_result.get("newly_activated_settlement_territories", []) or []
            )
            if newly_activated:
                info["legions_settlement_territories_activated"] = newly_activated
            newly_captured = list(
                cast_result.get("newly_captured_objective_cities", []) or []
            )
            if newly_captured:
                info["captured_objective_cities"] = newly_captured
                info["final_objective_reward"] = float(
                    cast_result.get("final_objective_reward", 0.0) or 0.0
                )

            if self._all_objective_cities_captured():
                self._log("=== ЗАХВАЧЕНЫ ВСЕ ЦЕЛЕВЫЕ ГОРОДА: ПОБЕДА В КАМПАНИИ ===")
                grid_obs = self._get_grid_obs()
                info["campaign_result"] = "victory"
                info["campaign_victory_reason"] = "objective_cities_cleared"
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                )

            if self.grid_env.all_enemies_defeated():
                self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
                grid_obs = self._get_grid_obs()
                info["campaign_result"] = "victory"
                info["campaign_victory_reason"] = "all_enemies_defeated"
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                )

        grid_obs = self._get_grid_obs()
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=False,
            truncated=False,
            info=info,
        )

    @classmethod
    def _spell_untargetable_enemy_ids(cls) -> frozenset[int]:
        blocked_ids: set[int] = set()
        for override_group in (
            VILLAGE_LINKED_STACK_OVERRIDES,
            VILLAGE_INTERNAL_GARRISON_OVERRIDES,
            CAPITAL_INTERNAL_STACK_OVERRIDES,
            RUIN_STACK_OVERRIDES,
        ):
            for entry in override_group:
                try:
                    blocked_ids.add(int(entry.get("enemy_id", -1)))
                except (TypeError, ValueError, AttributeError):
                    continue
        return frozenset(enemy_id for enemy_id in blocked_ids if enemy_id >= 0)

    @classmethod
    def _is_enemy_stack_spell_targetable(cls, enemy_id: Optional[int]) -> bool:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return False
        return normalized_enemy_id not in cls._spell_untargetable_enemy_ids()

    def _enemy_team_has_living_units(self, enemy_id: Optional[int]) -> bool:
        for unit in self._get_enemy_team_state(enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue
            hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            if max_hp > 0.0 and hp > 0.0:
                return True
        return False

    @staticmethod
    def _normalize_accuracy_value(value: object) -> int:
        try:
            return max(0, min(100, int(round(float(value or 0)))))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _enemy_map_spell_base_field(stat_name: str) -> str:
        return f"campaign_map_spell_base_{str(stat_name or '').strip()}"

    def _capture_enemy_stack_spell_bases(self, enemy_id: Optional[int]) -> None:
        for unit in self._get_enemy_team_state(enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue
            base_values = {
                "armor": self._normalize_armor_value(unit.get("armor", 0)),
                "damage": self._normalize_damage_value(unit.get("damage", 0)),
                "damage_secondary": self._normalize_damage_value(unit.get("damage_secondary", 0)),
                "initiative_base": self._normalize_damage_value(
                    unit.get("initiative_base", unit.get("initiative", 0))
                ),
                "accuracy": self._normalize_accuracy_value(unit.get("accuracy", 0)),
                "accuracy_secondary": self._normalize_accuracy_value(
                    unit.get("accuracy_secondary", 0)
                ),
            }
            for stat_name, base_value in base_values.items():
                base_field = self._enemy_map_spell_base_field(stat_name)
                if base_field not in unit:
                    unit[base_field] = base_value

    def _restore_enemy_stack_spell_bases(self, enemy_id: Optional[int]) -> None:
        for unit in self._get_enemy_team_state(enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue

            base_armor = self._normalize_armor_value(
                unit.pop(self._enemy_map_spell_base_field("armor"), unit.get("armor", 0))
            )
            base_damage = self._normalize_damage_value(
                unit.pop(self._enemy_map_spell_base_field("damage"), unit.get("damage", 0))
            )
            base_damage_secondary = self._normalize_damage_value(
                unit.pop(
                    self._enemy_map_spell_base_field("damage_secondary"),
                    unit.get("damage_secondary", 0),
                )
            )
            base_initiative = self._normalize_damage_value(
                unit.pop(
                    self._enemy_map_spell_base_field("initiative_base"),
                    unit.get("initiative_base", unit.get("initiative", 0)),
                )
            )
            base_accuracy = self._normalize_accuracy_value(
                unit.pop(self._enemy_map_spell_base_field("accuracy"), unit.get("accuracy", 0))
            )
            base_accuracy_secondary = self._normalize_accuracy_value(
                unit.pop(
                    self._enemy_map_spell_base_field("accuracy_secondary"),
                    unit.get("accuracy_secondary", 0),
                )
            )

            unit["armor"] = base_armor
            unit["damage"] = base_damage
            unit["damage_secondary"] = base_damage_secondary
            unit["initiative_base"] = base_initiative
            unit["accuracy"] = base_accuracy
            unit["accuracy_secondary"] = base_accuracy_secondary

            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            unit["initiative"] = base_initiative if max_hp > 0.0 and current_hp > 0.0 else 0

    def _enemy_stack_spell_effect_summary(self, enemy_id: Optional[int]) -> Dict[str, object]:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            normalized_enemy_id = -1

        active_spell_ids = sorted(
            str(spell_id)
            for spell_id in self.enemy_map_spell_effects.get(normalized_enemy_id, set())
            if str(spell_id)
        )
        specs_by_id = self._combined_map_offensive_spell_specs_by_id()

        armor_delta = 0
        damage_multiplier = 1.0
        initiative_multiplier = 1.0
        accuracy_multiplier = 1.0
        debuff_types: List[str] = []

        for spell_id in active_spell_ids:
            spec = specs_by_id.get(str(spell_id), {})
            if str(spec.get("kind", "") or "") != "debuff":
                continue
            debuff_type = str(spec.get("debuff_type", "") or "").strip()
            if debuff_type:
                debuff_types.append(debuff_type)
            armor_delta += int(spec.get("armor_delta", 0) or 0)
            damage_multiplier *= float(spec.get("damage_multiplier", 1.0) or 1.0)
            initiative_multiplier *= float(spec.get("initiative_multiplier", 1.0) or 1.0)
            accuracy_multiplier *= float(spec.get("accuracy_multiplier", 1.0) or 1.0)

        return {
            "active_spell_ids": active_spell_ids,
            "debuff_types": sorted(set(debuff_types)),
            "armor_delta": int(armor_delta),
            "damage_multiplier": float(damage_multiplier),
            "initiative_multiplier": float(initiative_multiplier),
            "accuracy_multiplier": float(accuracy_multiplier),
        }

    def _recompute_enemy_stack_spell_effects(self, enemy_id: Optional[int]) -> int:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return 0

        active_spell_ids = {
            str(spell_id)
            for spell_id in self.enemy_map_spell_effects.get(normalized_enemy_id, set())
            if str(spell_id)
        }
        if not active_spell_ids:
            self._restore_enemy_stack_spell_bases(normalized_enemy_id)
            return 0

        self._capture_enemy_stack_spell_bases(normalized_enemy_id)
        summary = self._enemy_stack_spell_effect_summary(normalized_enemy_id)

        affected_units = 0
        for unit in self._get_enemy_team_state(normalized_enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue

            base_armor = self._normalize_armor_value(
                unit.get(self._enemy_map_spell_base_field("armor"), unit.get("armor", 0))
            )
            base_damage = self._normalize_damage_value(
                unit.get(self._enemy_map_spell_base_field("damage"), unit.get("damage", 0))
            )
            base_damage_secondary = self._normalize_damage_value(
                unit.get(
                    self._enemy_map_spell_base_field("damage_secondary"),
                    unit.get("damage_secondary", 0),
                )
            )
            base_initiative = self._normalize_damage_value(
                unit.get(
                    self._enemy_map_spell_base_field("initiative_base"),
                    unit.get("initiative_base", unit.get("initiative", 0)),
                )
            )
            base_accuracy = self._normalize_accuracy_value(
                unit.get(self._enemy_map_spell_base_field("accuracy"), unit.get("accuracy", 0))
            )
            base_accuracy_secondary = self._normalize_accuracy_value(
                unit.get(
                    self._enemy_map_spell_base_field("accuracy_secondary"),
                    unit.get("accuracy_secondary", 0),
                )
            )

            unit["armor"] = self._normalize_armor_value(
                float(base_armor) + float(summary["armor_delta"])
            )
            unit["damage"] = self._normalize_damage_value(
                float(base_damage) * float(summary["damage_multiplier"])
            )
            unit["damage_secondary"] = self._normalize_damage_value(
                float(base_damage_secondary) * float(summary["damage_multiplier"])
            )

            new_initiative_base = self._normalize_damage_value(
                float(base_initiative) * float(summary["initiative_multiplier"])
            )
            unit["initiative_base"] = int(new_initiative_base)
            unit["accuracy"] = self._normalize_accuracy_value(
                float(base_accuracy) * float(summary["accuracy_multiplier"])
            )
            unit["accuracy_secondary"] = self._normalize_accuracy_value(
                float(base_accuracy_secondary) * float(summary["accuracy_multiplier"])
            )

            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            is_alive = max_hp > 0.0 and current_hp > 0.0
            unit["initiative"] = int(new_initiative_base) if is_alive else 0
            if is_alive:
                affected_units += 1

        return int(affected_units)

    def _apply_debuff_to_enemy_stack(
        self,
        enemy_id: Optional[int],
        spell_key: str,
    ) -> Tuple[int, Dict[str, object]]:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return 0, self._enemy_stack_spell_effect_summary(enemy_id)

        active_spell_ids = self.enemy_map_spell_effects.setdefault(normalized_enemy_id, set())
        active_spell_ids.add(str(spell_key))
        affected_units = self._recompute_enemy_stack_spell_effects(normalized_enemy_id)
        return int(affected_units), self._enemy_stack_spell_effect_summary(normalized_enemy_id)

    def _clear_all_enemy_map_spell_effects(self) -> int:
        expired_stacks = 0
        for enemy_id, active_spell_ids in list(self.enemy_map_spell_effects.items()):
            if not active_spell_ids:
                continue
            self._restore_enemy_stack_spell_bases(enemy_id)
            expired_stacks += 1
        self.enemy_map_spell_effects = {}
        if expired_stacks > 0:
            self._log(
                f"Истекли временные дебаффы боевых заклинаний на {expired_stacks} вражеских отрядах."
            )
        return int(expired_stacks)

    def _clear_enemy_map_spell_effects_for_enemy(self, enemy_id: Optional[int]) -> None:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return
        if normalized_enemy_id not in self.enemy_map_spell_effects:
            return
        self.enemy_map_spell_effects.pop(normalized_enemy_id, None)
        self._restore_enemy_stack_spell_bases(normalized_enemy_id)

    def _hero_stack_has_living_units(self) -> bool:
        for unit in self._get_blue_state():
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            if max_hp > 0.0 and current_hp > 0.0:
                return True
        return False

    def _blue_stack_spell_effect_summary(self) -> Dict[str, object]:
        active_spell_ids = sorted(str(spell_id) for spell_id in self.blue_map_spell_effects if str(spell_id))
        specs_by_id = self._combined_map_support_spell_specs_by_id()

        armor_delta = 0
        damage_multiplier = 1.0
        initiative_multiplier = 1.0
        accuracy_multiplier = 1.0
        health_delta = 0
        buff_types: List[str] = []
        resistance_types: List[str] = []

        for spell_id in active_spell_ids:
            spec = specs_by_id.get(str(spell_id), {})
            spell_kind = str(spec.get("kind", "") or "")
            buff_type = str(spec.get("buff_type", "") or "").strip()
            if spell_kind == "buff":
                if buff_type:
                    buff_types.append(buff_type)
                armor_delta += int(spec.get("armor_delta", 0) or 0)
                damage_multiplier *= float(spec.get("damage_multiplier", 1.0) or 1.0)
                initiative_multiplier *= float(spec.get("initiative_multiplier", 1.0) or 1.0)
                accuracy_multiplier *= float(spec.get("accuracy_multiplier", 1.0) or 1.0)
            elif spell_kind == "health_bonus":
                if buff_type:
                    buff_types.append(buff_type)
                health_delta += int(spec.get("health_delta", 0) or 0)
            elif spell_kind == "ward":
                if buff_type:
                    buff_types.append(buff_type)
                resistance_type = str(spec.get("resistance_type", "") or "").strip()
                if resistance_type:
                    resistance_types.append(resistance_type)

        return {
            "active_spell_ids": active_spell_ids,
            "buff_types": sorted(set(buff_types)),
            "resistance_types": sorted(set(resistance_types)),
            "armor_delta": int(armor_delta),
            "damage_multiplier": float(damage_multiplier),
            "initiative_multiplier": float(initiative_multiplier),
            "accuracy_multiplier": float(accuracy_multiplier),
            "health_delta": int(health_delta),
        }

    def _apply_heal_to_blue_stack(self, heal_amount: float) -> Tuple[int, float]:
        if heal_amount <= 0.0:
            return 0, 0.0

        affected_units = 0
        total_healed = 0.0
        for unit in self._get_blue_state():
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            if max_hp <= 0.0 or current_hp <= 0.0:
                continue
            healed_hp = min(max_hp, current_hp + float(heal_amount))
            healed_amount = max(0.0, healed_hp - current_hp)
            unit["health"] = float(healed_hp)
            unit["hp"] = float(healed_hp)
            if healed_amount > 0.0:
                affected_units += 1
                total_healed += float(healed_amount)
        return int(affected_units), float(total_healed)

    def _restore_grid_moves(self, restore_fraction: float) -> float:
        current_moves = max(0.0, float(self.moves or 0.0))
        move_cap = max(0.0, float(self.moves_per_turn or 0.0))
        if move_cap <= 0.0 or restore_fraction <= 0.0:
            return 0.0
        restore_amount = max(0.0, round(move_cap * float(restore_fraction)))
        next_moves = min(move_cap, current_moves + restore_amount)
        restored = max(0.0, next_moves - current_moves)
        self.moves = int(round(next_moves))
        return float(restored)

    def _apply_support_spell_to_blue_stack(
        self,
        spell_key: str,
    ) -> Tuple[int, Dict[str, object]]:
        normalized_spell_key = str(spell_key or "")
        if not normalized_spell_key:
            return 0, self._blue_stack_spell_effect_summary()
        self.blue_map_spell_effects.add(normalized_spell_key)
        living_units = sum(
            1
            for unit in self._get_blue_state()
            if float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0) > 0.0
            and float(unit.get("health", 0) or unit.get("hp", 0) or 0.0) > 0.0
        )
        return int(living_units), self._blue_stack_spell_effect_summary()

    def _clear_all_blue_map_spell_effects(self) -> int:
        expired_count = len(self.blue_map_spell_effects)
        self.blue_map_spell_effects = set()
        if expired_count > 0:
            self._log(
                f"Истекли временные эффекты боевых заклинаний на отряде героя ({expired_count} шт.)."
            )
        return int(expired_count)

    @classmethod
    def _unit_blocks_spell_damage(cls, unit: Dict, damage_type: Optional[str]) -> bool:
        normalized_type = str(damage_type or "").strip()
        if not normalized_type:
            return False

        normalized_type_lower = normalized_type.lower()
        immunities_lower = {
            str(value or "").strip().lower() for value in (unit.get("immunity") or [])
        }
        if normalized_type_lower in immunities_lower:
            return True

        resistances_lower = {
            str(value or "").strip().lower() for value in (unit.get("resistance") or [])
        }
        if normalized_type_lower in resistances_lower:
            return True

        defence_flag_names = (
            f"{normalized_type}defence",
            f"{normalized_type}Defense",
            f"{normalized_type.lower()}defence",
            f"{normalized_type.lower()}_defence",
        )
        for flag_name in defence_flag_names:
            try:
                if int(unit.get(flag_name, 0) or 0) == 1:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _apply_damage_to_enemy_stack(
        self,
        enemy_id: Optional[int],
        damage_amount: float,
        damage_type: Optional[str] = None,
    ) -> Tuple[float, int, int, int]:
        actual_damage = 0.0
        units_hit = 0
        defeated_units = 0
        blocked_units = 0

        for unit in self._get_enemy_team_state(enemy_id):
            if self._is_empty_enemy_unit(unit):
                continue
            current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
            max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
            if max_hp <= 0.0 or current_hp <= 0.0:
                continue
            if self._unit_blocks_spell_damage(unit, damage_type):
                blocked_units += 1
                continue
            new_hp = max(0.0, current_hp - float(damage_amount))
            actual_damage += max(0.0, current_hp - new_hp)
            units_hit += 1
            if new_hp <= 0.0:
                defeated_units += 1
                unit["initiative"] = 0
            unit["health"] = float(new_hp)
            if "hp" in unit:
                unit["hp"] = float(new_hp)

        return (
            float(actual_damage),
            int(units_hit),
            int(defeated_units),
            int(blocked_units),
        )

    def _heal_wounded_enemy_teams_for_turn(self) -> Tuple[int, float]:
        healed_units = 0
        healed_total = 0.0

        for enemy_id, is_alive in self.grid_env.enemies_alive.items():
            if not bool(is_alive):
                continue
            enemy_position = self.grid_env.enemy_positions.get(int(enemy_id))
            on_player_territory = self._is_player_controlled_territory_tile(enemy_position)
            heal_fraction = 0.05 if on_player_territory else 0.15
            heal_fraction += self._resolve_enemy_regeneration_bonus_percent(enemy_position)
            for unit in self._get_enemy_team_state(enemy_id):
                if self._is_empty_enemy_unit(unit):
                    continue
                current_hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
                max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0.0)
                if max_hp <= 0.0 or current_hp <= 0.0 or current_hp >= max_hp:
                    continue
                heal_amount = max(1.0, max_hp * float(heal_fraction))
                new_hp = min(max_hp, current_hp + heal_amount)
                actual_heal = max(0.0, new_hp - current_hp)
                if actual_heal <= 0.0:
                    continue
                unit["health"] = float(new_hp)
                if "hp" in unit:
                    unit["hp"] = float(new_hp)
                healed_units += 1
                healed_total += actual_heal

        if healed_units > 0:
            self._log(
                f"Вражеские отряды восстановили {healed_total:g} HP на {healed_units} юнитах за ход."
            )

        return int(healed_units), float(healed_total)

    def _is_player_controlled_territory_tile(self, position: Optional[Tuple[int, int]]) -> bool:
        """Возвращает True для клетки, относящейся к территории игрока.

        В текущей кампании земля игрока хранится в legions_territory_tile_set:
        сюда входят клетки от стартовой столицы и захваченных городов/поселений.
        Название legacy-поля историческое и не связано с текущим цветом отрисовки.
        """
        if position is None:
            return False
        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return False
        return normalized_position in self.legions_territory_tile_set

    def _grant_merchant_item(self, item_name: str) -> List[str]:
        granted_items: List[str] = []
        item_data = self._merchant_item_definition(item_name)
        grant_kind = str((item_data or {}).get("grant", "") or "")

        if grant_kind == "small_heal_bonus":
            self.extra_healing_bottles = max(0, int(self.extra_healing_bottles or 0)) + 1
            granted_items.append(self.BONUS_SMALL_HEAL_ITEM_NAME)
        elif grant_kind == "large_heal_bonus":
            self.extra_heal_bottles = max(0, int(self.extra_heal_bottles or 0)) + 1
            granted_items.append(self.BONUS_LARGE_HEAL_ITEM_NAME)
        elif grant_kind == "revive_bonus":
            self.extra_revive_bottles = max(0, int(self.extra_revive_bottles or 0)) + 1
            granted_items.append(self.BONUS_REVIVE_ITEM_NAME)
        else:
            granted_items.append(str(item_name or ""))

        for granted_item in granted_items:
            self._add_hero_item(granted_item)
        return granted_items

    def _step_buy_merchant_item(self, action: int):
        idx = action - self.GRID_MERCHANT_BUY_ACTION_START
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        item_data = self.MERCHANT_BUY_ITEMS[idx] if 0 <= idx < len(self.MERCHANT_BUY_ITEMS) else {}
        item_name = str(item_data.get("name", "") or "")
        item_price = float(item_data.get("price", 0.0) or 0.0)
        site_names = self._merchant_sites_at_position(self.grid_env.agent_pos)
        site_name = self._merchant_site_for_item(item_name, position=self.grid_env.agent_pos)
        stock_before = self._merchant_item_stock(site_name, item_name) if site_name else 0

        purchased = False
        insufficient_gold = False
        not_at_merchant = not bool(site_names)
        out_of_stock = bool(site_names) and not bool(site_name)
        granted_items: List[str] = []

        if site_name is not None and item_name:
            if float(self.gold or 0.0) < item_price:
                insufficient_gold = True
            elif stock_before > 0:
                self.gold = max(0.0, float(self.gold or 0.0) - item_price)
                self.merchant_stocks[site_name][item_name] = max(0, stock_before - 1)
                granted_items = self._grant_merchant_item(item_name)
                purchased = True
                self._log(
                    f"Торговец {site_name}: куплен предмет '{item_name}' за {item_price:.1f} gold"
                )

        stock_after = self._merchant_item_stock(site_name, item_name) if site_name else 0
        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "merchant_buy_action": True,
            "merchant_buy_item": item_name,
            "merchant_buy_price": item_price,
            "merchant_buy_site": site_name,
            "merchant_buy_purchased": purchased,
            "merchant_buy_stock_before": int(stock_before),
            "merchant_buy_stock_after": int(stock_after),
            "merchant_buy_granted_items": list(granted_items),
            "merchant_buy_not_at_merchant": not_at_merchant,
            "merchant_buy_out_of_stock": out_of_stock,
            "merchant_buy_insufficient_gold": insufficient_gold,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_buy_spell_shop_spell(self, action: int):
        idx = action - self.GRID_SPELL_SHOP_BUY_ACTION_START
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        spell_data = (
            self.SPELL_SHOP_BUY_SPELLS[idx] if 0 <= idx < len(self.SPELL_SHOP_BUY_SPELLS) else {}
        )
        spell_name = str(spell_data.get("name", "") or "")
        spell_id = str(spell_data.get("spell_id", "") or "")
        spell_price = float(spell_data.get("price", 0.0) or 0.0)
        site_names = self._spell_shop_sites_at_position(self.grid_env.agent_pos)
        site_name = self._spell_shop_site_for_spell(spell_name, position=self.grid_env.agent_pos)
        stock_before = self._spell_shop_spell_stock(site_name, spell_name) if site_name else 0

        purchased = False
        insufficient_gold = False
        already_owned = self._spell_shop_spell_owned(spell_id)
        not_at_spell_shop = not bool(site_names)
        out_of_stock = bool(site_names) and not bool(site_name)
        added_to_spellbook = False

        if site_name is not None and spell_name:
            if already_owned:
                pass
            elif float(self.gold or 0.0) < spell_price:
                insufficient_gold = True
            elif stock_before > 0:
                self.gold = max(0.0, float(self.gold or 0.0) - spell_price)
                self.spell_shop_stocks[site_name][spell_name] = max(0, stock_before - 1)
                self.spell_shop_purchased_spell_ids.add(spell_id)
                spell_entry = self.active_spells.get(spell_id)
                if isinstance(spell_entry, dict):
                    spell_entry["learned"] = 1
                    added_to_spellbook = True
                purchased = True
                self._log(
                    f"Лавка заклинаний {site_name}: куплено заклинание '{spell_name}' за {spell_price:.1f} gold"
                )

        stock_after = self._spell_shop_spell_stock(site_name, spell_name) if site_name else 0
        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "spell_shop_buy_action": True,
            "spell_shop_buy_spell": spell_name,
            "spell_shop_buy_spell_id": spell_id,
            "spell_shop_buy_price": spell_price,
            "spell_shop_buy_site": site_name,
            "spell_shop_buy_purchased": purchased,
            "spell_shop_buy_added_to_spellbook": added_to_spellbook,
            "spell_shop_buy_already_owned": already_owned,
            "spell_shop_buy_stock_before": int(stock_before),
            "spell_shop_buy_stock_after": int(stock_after),
            "spell_shop_buy_not_at_spell_shop": not_at_spell_shop,
            "spell_shop_buy_out_of_stock": out_of_stock,
            "spell_shop_buy_insufficient_gold": insufficient_gold,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_hire_mercenary_unit(self, action: int):
        idx = action - self.GRID_MERCENARY_HIRE_ACTION_START
        option = self._mercenary_option(idx)

        grid_obs = self._get_grid_obs()
        terminated = False
        truncated = False

        site_names = self._mercenary_sites_at_position(self.grid_env.agent_pos)
        site_name = None if option is None else str(option.get("site_name", "") or "").strip()
        site_id = None if option is None else str(option.get("site_id", "") or "").strip()
        unit_name = None if option is None else str(
            option.get("unit_name", option.get("name", "")) or ""
        ).strip()
        unit_type = None if option is None else str(option.get("unit_type", "") or "").strip()
        hire_cost = 0.0
        hire_stand = None if option is None else str(option.get("stand", "") or "").strip().lower()
        hire_target_pos = None
        stock_before = 0
        if option is not None:
            try:
                hire_cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
            except (TypeError, ValueError):
                hire_cost = 0.0
            stock_before = max(0, int(option.get("stock", 0) or 0))
            positions = self._hire_positions_for_stand(hire_stand)
            if positions:
                hire_target_pos = self._find_first_empty_blue_position(positions)

        hire_available_before = self._can_hire_mercenary_unit_option(option)
        hired, hired_unit_name, hired_position, gold_spent, hired_site_name, stock_before_used, stock_after = (
            self._hire_mercenary_unit_option(option)
        )

        hero = self._resolve_travel_hero()
        hero_needaunit = self._resolve_hero_needaunit(hero) if hero is not None else 0
        reward = self._hire_reward_value() if hired else 0.0
        matching_site_in_range = bool(site_name) and site_name in site_names
        insufficient_gold = matching_site_in_range and bool(unit_name) and float(self.gold or 0.0) < hire_cost
        out_of_stock = matching_site_in_range and stock_before <= 0

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "hire_action": True,
            "hire_action_kind": "mercenary",
            "hire_action_idx": int(idx),
            "hire_unit_name": unit_name,
            "hire_role": hire_stand,
            "hire_cost": float(hire_cost),
            "hire_target_pos": hire_target_pos,
            "hire_available": bool(hire_available_before),
            "hired": bool(hired),
            "hired_units_count": int(1 if hired else 0),
            "hired_unit_name": hired_unit_name,
            "hired_position": hired_position,
            "gold_spent": float(gold_spent),
            "hire_reward": float(reward),
            "hero_needaunit": int(hero_needaunit),
            "mercenary_hire_action": True,
            "mercenary_hire_action_idx": int(idx),
            "mercenary_hire_site": hired_site_name or site_name,
            "mercenary_hire_site_id": site_id,
            "mercenary_hire_unit_name": unit_name,
            "mercenary_hire_unit_type": unit_type,
            "mercenary_hire_stand": hire_stand,
            "mercenary_hire_cost": float(hire_cost),
            "mercenary_hire_stock_before": int(stock_before_used if hired else stock_before),
            "mercenary_hire_stock_after": int(stock_after),
            "mercenary_hire_available": bool(hire_available_before),
            "mercenary_hire_not_at_camp": not matching_site_in_range,
            "mercenary_hire_out_of_stock": bool(out_of_stock),
            "mercenary_hire_insufficient_gold": bool(
                matching_site_in_range and not hired and bool(unit_name) and float(self.gold or 0.0) < hire_cost
            ),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_train_unit_at_trainer(self, action: int):
        idx = action - self.GRID_TRAINER_ACTION_START
        target_pos = (
            self.TRAINER_POSITIONS[idx] if 0 <= idx < len(self.TRAINER_POSITIONS) else None
        )
        terminated = False
        truncated = False
        reward = 0.0

        site_names = self._trainer_sites_at_position(self.grid_env.agent_pos)
        site_name = site_names[0] if site_names else None
        preview_before = (
            self._trainer_preview_for_position(int(target_pos))
            if target_pos is not None
            else {
                "site_names": list(site_names),
                "unit": None,
                "unit_name": None,
                "trainable": False,
                "missing_unit": True,
                "dead_or_empty": False,
                "already_capped": False,
                "insufficient_gold": False,
                "exp_cap": None,
                "exp_before": 0,
                "xp_missing": 0,
                "gold_per_xp": 0.0,
                "gold_needed_full": 0.0,
                "affordable_xp": 0,
                "gold_spent": 0.0,
                "exp_after": 0,
            }
        )

        trained = False
        xp_gained = 0
        gold_spent = 0.0
        exp_after = int(preview_before.get("exp_before", 0) or 0)
        unit_name = str(preview_before.get("unit_name") or "").strip() or None
        unit = preview_before.get("unit")

        if target_pos is not None and bool(preview_before.get("trainable", False)) and isinstance(unit, dict):
            xp_gained = max(0, int(preview_before.get("affordable_xp", 0) or 0))
            gold_spent = max(0.0, float(preview_before.get("gold_spent", 0.0) or 0.0))
            exp_cap = preview_before.get("exp_cap")
            exp_before = max(0, int(preview_before.get("exp_before", 0) or 0))
            if exp_cap is not None and xp_gained > 0:
                exp_after = min(int(exp_cap), exp_before + xp_gained)
                unit["exp_current"] = int(exp_after)
                self.gold = max(0.0, float(self.gold or 0.0) - gold_spent)
                trained = True
                self._sync_hero_progression_flags(self.blue_team_state)
                self._log(
                    f"Тренировочный лагерь: {unit.get('name', 'unit')}#{int(target_pos)} "
                    f"+{xp_gained} XP за {gold_spent:.1f} золота"
                )

        preview_after = (
            self._trainer_preview_for_position(int(target_pos))
            if target_pos is not None
            else preview_before
        )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "trainer_action": True,
            "trainer_action_idx": int(idx),
            "trainer_site": site_name,
            "trainer_position": target_pos,
            "trainer_unit_name": unit_name,
            "trainer_trainable": bool(preview_before.get("trainable", False)),
            "trainer_not_at_camp": not bool(site_names),
            "trainer_missing_unit": bool(preview_before.get("missing_unit", False)),
            "trainer_dead_or_empty": bool(preview_before.get("dead_or_empty", False)),
            "trainer_already_capped": bool(preview_before.get("already_capped", False)),
            "trainer_insufficient_gold": bool(preview_before.get("insufficient_gold", False)),
            "trainer_gold_per_xp": float(preview_before.get("gold_per_xp", 0.0) or 0.0),
            "trainer_gold_needed_full": float(preview_before.get("gold_needed_full", 0.0) or 0.0),
            "trainer_gold_spent": float(gold_spent),
            "trainer_exp_before": int(preview_before.get("exp_before", 0) or 0),
            "trainer_exp_after": int(exp_after),
            "trainer_exp_cap": preview_before.get("exp_cap"),
            "trainer_xp_missing_before": int(preview_before.get("xp_missing", 0) or 0),
            "trainer_xp_gained": int(xp_gained),
            "trainer_xp_affordable_before": int(preview_before.get("affordable_xp", 0) or 0),
            "trainer_xp_missing_after": int(preview_after.get("xp_missing", 0) or 0),
            "trainer_trained": bool(trained),
            "gold_spent": float(gold_spent),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        grid_obs = self._get_grid_obs()

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_building(self, action: int):
        """Выбирает здание активной фракции и отмечает его как построенное."""
        idx = action - self.GRID_BUILD_ACTION_START
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        build_key = None
        build_name = None
        built = False
        already_built = False
        blocked_names: List[str] = []
        build_cost = None
        insufficient_gold = False
        build_blocked = False
        build_locked = False

        if 0 <= idx < len(self.building_keys):
            build_key = self.building_keys[idx]
            building = self.active_buildings.get(build_key)
            if isinstance(building, dict):
                build_name = building.get("name")
                build_cost = self._get_building_gold_cost(building)
                alredybuilt_flag = self.active_buildings.get("alredybuilt", 0)
                try:
                    build_locked = int(alredybuilt_flag or 0) == 1
                except (TypeError, ValueError):
                    build_locked = False
                built_flag = building.get("Build", building.get("built", 0))
                already_built = int(built_flag or 0) == 1
                blocked_flag = building.get("blocked", 0)
                build_blocked = int(blocked_flag or 0) == 1
                if build_locked:
                    pass
                elif build_blocked:
                    pass
                elif self.gold < (build_cost or 0):
                    insufficient_gold = True
                elif not already_built:
                    self.gold -= build_cost or 0
                    building["built"] = 1
                    built = True
                    self.active_buildings["alredybuilt"] = 1

                    blocks_raw = building.get("blocks", [])
                    if isinstance(blocks_raw, (list, tuple)):
                        block_names = [
                            str(name).strip() for name in blocks_raw if str(name).strip()
                        ]
                    elif blocks_raw:
                        block_names = [str(blocks_raw).strip()]
                    else:
                        block_names = []

                    if block_names:
                        for entry in self.active_buildings.values():
                            if not isinstance(entry, dict):
                                continue
                            entry_name = str(entry.get("name", "") or "").strip()
                            if entry_name and entry_name in block_names:
                                entry["blocked"] = 1
                                blocked_names.append(entry_name)

                    if build_name:
                        self._log("ЗДАНИЕ ПОСТРОЕНО")
                        self._log(f'Построено здание "{build_name}"')
                        if blocked_names:
                            self._log(f'Заблокированы здания: {", ".join(blocked_names)}')

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "build_action": True,
            "build_key": build_key,
            "build_name": build_name,
            "built": built,
            "already_built": already_built,
            "blocked_names": blocked_names,
            "build_cost": build_cost,
            "insufficient_gold": insufficient_gold,
            "build_blocked": build_blocked,
            "build_locked": build_locked,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_learn_spell(self, action: int):
        idx = action - self.grid_spell_action_start
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        spell_key = None
        spell_description = None
        spell_level = None
        spell_learned = False
        spell_already_learned = False
        spell_learning_locked = bool(self.spell_learning_locked)
        missing_magic_tower = not self._has_magic_tower_built()
        blocked_by_typeoflord = False
        insufficient_mana = False
        mana_costs: Dict[str, float] = {}
        mana_totals_before = self._current_mana_totals()

        if 0 <= idx < len(self.spell_keys):
            spell_key = self.spell_keys[idx]
            spell = self.active_spells.get(spell_key)
            if isinstance(spell, dict):
                spell_description = str(spell.get("description", "") or "").strip()
                try:
                    spell_level = int(spell.get("level", 0) or 0)
                except (TypeError, ValueError):
                    spell_level = 0
                blocked_by_typeoflord = self._is_spell_level_masked_by_typeoflord(spell_level)
                try:
                    spell_already_learned = int(spell.get("learned", 0) or 0) == 1
                except (TypeError, ValueError):
                    spell_already_learned = False
                mana_costs = self._get_spell_learning_costs(spell)
                insufficient_mana = not self._has_mana_for_costs(mana_costs)
                if (
                    not spell_learning_locked
                    and not missing_magic_tower
                    and not blocked_by_typeoflord
                    and not spell_already_learned
                    and not insufficient_mana
                ):
                    self._spend_mana_costs(mana_costs)
                    spell["learned"] = 1
                    self.spell_learning_locked = True
                    spell_learning_locked = True
                    spell_learned = True
                    reward += float(self.reward_spell_learn)
                    if spell_description:
                        self._log("ЗАКЛИНАНИЕ ИЗУЧЕНО")
                        self._log(f'Изучено заклинание: "{spell_description}"')

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "spell_action": True,
            "spell_key": spell_key,
            "spell_description": spell_description,
            "spell_level": spell_level,
            "spell_learned": spell_learned,
            "spell_already_learned": spell_already_learned,
            "spell_learning_locked": spell_learning_locked,
            "missing_magic_tower": missing_magic_tower,
            "blocked_by_typeoflord": blocked_by_typeoflord,
            "insufficient_mana": insufficient_mana,
            "spell_learning_costs": dict(mana_costs),
            "mana_totals_before_spell_learning": dict(mana_totals_before),
            "spell_learning_reward": float(reward if spell_learned else 0.0),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_cast_legion_damage_spell(self, action: int):
        idx = int(action) - int(self.grid_legion_damage_spell_action_start)
        reward = 0.0
        current_specs = self._current_map_offensive_spell_action_specs()

        spell_key = None
        spell_description = None
        spell_level = None
        spell_damage = 0.0
        spell_damage_type = None
        spell = None
        spell_spec: Dict[str, object] = {}
        spell_kind = ""
        spell_debuff_type = None
        summon_unit_name = None
        spell_effect_summary: Dict[str, object] = {
            "active_spell_ids": [],
            "debuff_types": [],
            "armor_delta": 0,
            "damage_multiplier": 1.0,
            "initiative_multiplier": 1.0,
            "accuracy_multiplier": 1.0,
        }
        spell_units_affected = 0
        if 0 <= idx < len(current_specs):
            spell_spec = dict(current_specs[idx])
            spell_key = str(spell_spec.get("id", "") or "")
            spell = self.active_spells.get(str(spell_key))
            spell_kind = str(spell_spec.get("kind", "") or "")
            spell_damage = float(spell_spec.get("damage", 0.0) or 0.0)
            spell_damage_type = str(spell_spec.get("damage_type", "") or "") or None
            spell_debuff_type = str(spell_spec.get("debuff_type", "") or "") or None
            summon_unit_name = str(spell_spec.get("summon_unit_name", "") or "") or None

        if isinstance(spell, dict):
            spell_description = str(spell.get("description", "") or "").strip()
            try:
                spell_level = int(spell.get("level", 0) or 0)
            except (TypeError, ValueError):
                spell_level = 0

        mana_costs = self._get_spell_use_costs(spell) if isinstance(spell, dict) else {}
        has_map_offensive_spell_actions = bool(current_specs)
        spell_known = bool(spell_key) and self._is_spell_learned(str(spell_key))
        spell_used_this_turn = bool(spell_key) and self._spell_cast_limit_reached_this_turn(
            str(spell_key)
        )
        blocked_by_typeoflord = self._is_spell_blocked_by_typeoflord(spell)
        insufficient_mana = not self._has_mana_for_costs(mana_costs) if mana_costs else True
        nearest_enemy = (
            self._get_map_offensive_spell_target(str(spell_key))
            if spell_key
            else None
        )
        target_enemy_id = None if nearest_enemy is None else int(nearest_enemy.get("enemy_id", -1))
        target_position = None if nearest_enemy is None else tuple(nearest_enemy.get("position", ()))
        target_reachable = bool(nearest_enemy.get("reachable", False)) if nearest_enemy else False
        target_description = None if nearest_enemy is None else str(nearest_enemy.get("description", "") or "")

        spell_cast_applied = False
        spell_cast_executed = False
        actual_damage = 0.0
        units_hit = 0
        defeated_units = 0
        blocked_units = 0
        spell_enemy_defeated = False
        enemy_reward = 0.0
        ruin_clear_bonus_reward = 0.0
        newly_captured_objective_cities: List[str] = []
        newly_activated_settlement_territories: List[str] = []
        final_objective_reward = 0.0
        spell_cast_reward = 0.0

        can_cast = (
            bool(spell_key)
            and isinstance(spell, dict)
            and has_map_offensive_spell_actions
            and spell_known
            and not blocked_by_typeoflord
            and not spell_used_this_turn
            and nearest_enemy is not None
            and self._has_mana_for_costs(mana_costs)
        )

        if can_cast:
            cast_result = self._cast_from_spell_spec(
                source="learned_offensive_spell",
                spell_key=str(spell_key),
                spell_description=spell_description,
                spell_spec=spell_spec,
                mana_costs=mana_costs,
                spend_mana=True,
                enforce_turn_limit=True,
                nearest_targetable_enemy=nearest_enemy
                if not self._map_offensive_spell_targets_untargetable_stacks(str(spell_key))
                else None,
                nearest_any_enemy=nearest_enemy
                if self._map_offensive_spell_targets_untargetable_stacks(str(spell_key))
                else None,
            )
            spell_cast_applied = bool(cast_result.get("spell_cast_applied", False))
            spell_cast_executed = bool(cast_result.get("spell_cast_executed", False))
            spell_cast_reward = float(cast_result.get("spell_cast_reward", 0.0) or 0.0)
            reward += float(cast_result.get("reward", 0.0) or 0.0)
            actual_damage = float(cast_result.get("spell_damage_total", 0.0) or 0.0)
            units_hit = int(cast_result.get("spell_damage_units_hit", 0) or 0)
            defeated_units = int(
                cast_result.get("spell_damage_units_defeated", 0) or 0
            )
            blocked_units = int(cast_result.get("spell_damage_units_blocked", 0) or 0)
            spell_units_affected = int(cast_result.get("spell_units_affected", 0) or 0)
            spell_effect_summary = dict(cast_result.get("spell_effect_summary", {}))
            target_enemy_id = cast_result.get("target_enemy_id")
            target_position = cast_result.get("target_enemy_position")
            target_reachable = bool(cast_result.get("target_enemy_reachable", False))
            target_description = cast_result.get("target_enemy_description")
            spell_enemy_defeated = bool(cast_result.get("spell_enemy_defeated", False))
            enemy_reward = float(cast_result.get("enemy_defeat_reward", 0.0) or 0.0)
            ruin_clear_bonus_reward = float(
                cast_result.get("ruin_clear_bonus_reward", 0.0) or 0.0
            )
            newly_captured_objective_cities = list(
                cast_result.get("newly_captured_objective_cities", []) or []
            )
            newly_activated_settlement_territories = list(
                cast_result.get("newly_activated_settlement_territories", []) or []
            )
            final_objective_reward = float(
                cast_result.get("final_objective_reward", 0.0) or 0.0
            )
        else:
            cast_result = {}

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "spell_cast_action": True,
            "spell_key": spell_key,
            "spell_description": spell_description,
            "spell_level": spell_level,
            "spell_kind": spell_kind,
            "spell_damage": float(spell_damage),
            "spell_damage_type": spell_damage_type,
            "spell_debuff_type": spell_debuff_type,
            "spell_summon_unit_name": summon_unit_name,
            "spell_cast_applied": bool(spell_cast_applied),
            "spell_cast_executed": bool(spell_cast_executed),
            "spell_cast_reward": float(spell_cast_reward),
            "spell_cast_used_this_turn": bool(spell_used_this_turn),
            "summon_hero_battle_bonus_pending": bool(self.summon_hero_battle_bonus_pending),
            "summon_hero_battle_bonus_enemy_ids": sorted(
                int(enemy_id) for enemy_id in self.summon_hero_battle_bonus_enemy_ids_this_turn
            ),
            "spell_is_learned": bool(spell_known),
            "blocked_by_typeoflord": bool(blocked_by_typeoflord),
            "spell_is_legions_capital": bool(self._is_legions_capital()),
            "spell_is_undead_hordes_capital": bool(self._is_undead_hordes_capital()),
            "spell_has_map_offensive_actions": bool(has_map_offensive_spell_actions),
            "spell_use_costs": dict(mana_costs),
            "insufficient_mana": bool(not can_cast and mana_costs and not self._has_mana_for_costs(mana_costs)),
            "target_enemy_id": target_enemy_id,
            "target_enemy_position": target_position,
            "target_enemy_reachable": bool(target_reachable),
            "target_enemy_description": target_description,
            "spell_damage_total": float(actual_damage),
            "spell_damage_units_hit": int(units_hit),
            "spell_damage_units_defeated": int(defeated_units),
            "spell_damage_units_blocked": int(blocked_units),
            "spell_units_affected": int(spell_units_affected),
            "spell_effect_active_ids": list(spell_effect_summary.get("active_spell_ids", [])),
            "spell_effect_types": list(spell_effect_summary.get("debuff_types", [])),
            "spell_effect_armor_delta": int(spell_effect_summary.get("armor_delta", 0) or 0),
            "spell_effect_damage_multiplier": float(
                spell_effect_summary.get("damage_multiplier", 1.0) or 1.0
            ),
            "spell_effect_initiative_multiplier": float(
                spell_effect_summary.get("initiative_multiplier", 1.0) or 1.0
            ),
            "spell_effect_accuracy_multiplier": float(
                spell_effect_summary.get("accuracy_multiplier", 1.0) or 1.0
            ),
            "spell_enemy_defeated": bool(spell_enemy_defeated),
            "enemy_defeat_reward": float(enemy_reward),
            "enemy_defeat_reward_base": float(self.reward_defeat_enemy) if spell_enemy_defeated else 0.0,
            "blue_exp_reward": 0.0,
            "blue_exp_raw": 0.0,
            "ruin_clear_bonus_reward": float(ruin_clear_bonus_reward),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_map_spell_step(
            reward=reward,
            info=info,
            cast_result=cast_result,
        )

    def _step_cast_support_spell(self, action: int):
        idx = int(action) - int(self.grid_map_support_spell_action_start)
        reward = 0.0
        current_specs = self._current_map_support_spell_action_specs()

        spell_key = None
        spell_description = None
        spell_level = None
        spell = None
        spell_spec: Dict[str, object] = {}
        spell_kind = ""
        spell_buff_type = None
        spell_resistance_type = None
        spell_heal_amount = 0.0
        spell_health_delta = 0
        spell_moves_restore_fraction = 0.0
        spell_effect_summary: Dict[str, object] = self._blue_stack_spell_effect_summary()
        spell_units_affected = 0
        spell_heal_total = 0.0
        spell_moves_restored = 0.0

        if 0 <= idx < len(current_specs):
            spell_spec = dict(current_specs[idx])
            spell_key = str(spell_spec.get("id", "") or "")
            spell = self.active_spells.get(str(spell_key))
            spell_kind = str(spell_spec.get("kind", "") or "")
            spell_buff_type = str(spell_spec.get("buff_type", "") or "") or None
            spell_resistance_type = str(spell_spec.get("resistance_type", "") or "") or None
            spell_heal_amount = float(spell_spec.get("heal_amount", 0.0) or 0.0)
            spell_health_delta = int(spell_spec.get("health_delta", 0) or 0)
            spell_moves_restore_fraction = float(
                spell_spec.get("moves_restore_fraction", 0.0) or 0.0
            )

        if isinstance(spell, dict):
            spell_description = str(spell.get("description", "") or "").strip()
            try:
                spell_level = int(spell.get("level", 0) or 0)
            except (TypeError, ValueError):
                spell_level = 0

        mana_costs = self._get_spell_use_costs(spell) if isinstance(spell, dict) else {}
        has_map_support_spell_actions = bool(current_specs)
        spell_known = bool(spell_key) and self._is_spell_learned(str(spell_key))
        spell_used_this_turn = bool(spell_key) and self._spell_cast_limit_reached_this_turn(
            str(spell_key)
        )
        blocked_by_typeoflord = self._is_spell_blocked_by_typeoflord(spell)
        can_cast = (
            bool(spell_key)
            and isinstance(spell, dict)
            and has_map_support_spell_actions
            and spell_known
            and not blocked_by_typeoflord
            and not spell_used_this_turn
            and self._hero_stack_has_living_units()
            and self._has_mana_for_costs(mana_costs)
        )

        spell_cast_applied = False
        spell_cast_executed = False
        spell_cast_reward = 0.0
        moves_before_cast = float(self.moves or 0.0)

        if can_cast:
            cast_result = self._cast_from_spell_spec(
                source="learned_support_spell",
                spell_key=str(spell_key),
                spell_description=spell_description,
                spell_spec=spell_spec,
                mana_costs=mana_costs,
                spend_mana=True,
                enforce_turn_limit=True,
            )
            spell_cast_applied = bool(cast_result.get("spell_cast_applied", False))
            spell_cast_executed = bool(cast_result.get("spell_cast_executed", False))
            spell_cast_reward = float(cast_result.get("spell_cast_reward", 0.0) or 0.0)
            reward += float(cast_result.get("reward", 0.0) or 0.0)
            spell_units_affected = int(cast_result.get("spell_units_affected", 0) or 0)
            spell_heal_total = float(cast_result.get("spell_heal_total", 0.0) or 0.0)
            spell_moves_restored = float(
                cast_result.get("spell_moves_restored", 0.0) or 0.0
            )
            spell_effect_summary = dict(cast_result.get("spell_effect_summary", {}))
        else:
            cast_result = {}

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "spell_cast_action": True,
            "spell_key": spell_key,
            "spell_description": spell_description,
            "spell_level": spell_level,
            "spell_kind": spell_kind,
            "spell_buff_type": spell_buff_type,
            "spell_resistance_type": spell_resistance_type,
            "spell_heal_amount": float(spell_heal_amount),
            "spell_health_delta": int(spell_health_delta),
            "spell_moves_restore_fraction": float(spell_moves_restore_fraction),
            "spell_heal_total": float(spell_heal_total),
            "spell_moves_restored": float(spell_moves_restored),
            "spell_cast_applied": bool(spell_cast_applied),
            "spell_cast_executed": bool(spell_cast_executed),
            "spell_cast_reward": float(spell_cast_reward),
            "spell_cast_used_this_turn": bool(spell_used_this_turn),
            "spell_is_learned": bool(spell_known),
            "blocked_by_typeoflord": bool(blocked_by_typeoflord),
            "spell_is_empire_capital": bool(self._is_empire_capital()),
            "spell_has_map_support_actions": bool(has_map_support_spell_actions),
            "spell_use_costs": dict(mana_costs),
            "insufficient_mana": bool(
                not can_cast and mana_costs and not self._has_mana_for_costs(mana_costs)
            ),
            "target_enemy_id": None,
            "target_enemy_position": None,
            "target_enemy_reachable": False,
            "target_enemy_description": None,
            "spell_damage_total": 0.0,
            "spell_damage_units_hit": 0,
            "spell_damage_units_defeated": 0,
            "spell_damage_units_blocked": 0,
            "spell_units_affected": int(spell_units_affected),
            "spell_effect_active_ids": list(spell_effect_summary.get("active_spell_ids", [])),
            "spell_effect_types": list(spell_effect_summary.get("buff_types", [])),
            "spell_effect_resistance_types": list(
                spell_effect_summary.get("resistance_types", [])
            ),
            "spell_effect_armor_delta": int(spell_effect_summary.get("armor_delta", 0) or 0),
            "spell_effect_damage_multiplier": float(
                spell_effect_summary.get("damage_multiplier", 1.0) or 1.0
            ),
            "spell_effect_initiative_multiplier": float(
                spell_effect_summary.get("initiative_multiplier", 1.0) or 1.0
            ),
            "spell_effect_accuracy_multiplier": float(
                spell_effect_summary.get("accuracy_multiplier", 1.0) or 1.0
            ),
            "spell_effect_health_delta": int(spell_effect_summary.get("health_delta", 0) or 0),
            "spell_enemy_defeated": False,
            "enemy_defeat_reward": 0.0,
            "enemy_defeat_reward_base": 0.0,
            "blue_exp_reward": 0.0,
            "blue_exp_raw": 0.0,
            "ruin_clear_bonus_reward": 0.0,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "moves_before_cast": float(moves_before_cast),
            "moves_after_cast": float(self.moves or 0.0),
        }

        return self._finalize_map_spell_step(
            reward=reward,
            info=info,
            cast_result=cast_result,
        )

    def _step_cast_spell_shop_spell(self, action: int):
        idx = int(action) - int(self.grid_spell_shop_cast_action_start)
        reward = 0.0

        spell_data = self._spell_shop_cast_spell_entry(idx)
        spell_key = str(spell_data.get("spell_id", "") or "")
        spell_name = str(spell_data.get("name", "") or "").strip()
        spell = self._spell_entry_by_id(spell_key)
        offensive_spec = self._map_offensive_spell_spec(spell_key) if spell_key else {}
        support_spec = {} if offensive_spec else self._map_support_spell_spec(spell_key)
        spell_spec: Dict[str, object] = dict(offensive_spec or support_spec)
        spell_kind = str(spell_spec.get("kind", "") or "")

        spell_description = spell_name or spell_key or None
        spell_level = None
        if isinstance(spell, dict):
            spell_description = str(spell.get("description", "") or spell_description or "").strip()
            try:
                spell_level = int(spell.get("level", 0) or 0)
            except (TypeError, ValueError):
                spell_level = 0

        spell_damage = float(spell_spec.get("damage", 0.0) or 0.0)
        spell_damage_type = str(spell_spec.get("damage_type", "") or "") or None
        spell_debuff_type = str(spell_spec.get("debuff_type", "") or "") or None
        summon_unit_name = str(spell_spec.get("summon_unit_name", "") or "") or None
        spell_buff_type = str(spell_spec.get("buff_type", "") or "") or None
        spell_resistance_type = str(spell_spec.get("resistance_type", "") or "") or None
        spell_heal_amount = float(spell_spec.get("heal_amount", 0.0) or 0.0)
        spell_health_delta = int(spell_spec.get("health_delta", 0) or 0)
        spell_moves_restore_fraction = float(spell_spec.get("moves_restore_fraction", 0.0) or 0.0)

        mana_costs = self._get_spell_use_costs(spell) if isinstance(spell, dict) else {}
        spell_purchased = bool(spell_key) and self._spell_shop_spell_owned(spell_key)
        spell_regular_action = bool(spell_key) and self._spell_shop_spell_has_regular_action(spell_key)
        spell_used_this_turn = bool(spell_key) and self._spell_shop_cast_limit_reached_this_turn(
            spell_key
        )

        nearest_enemy = (
            self._get_map_offensive_spell_target(str(spell_key))
            if spell_key and offensive_spec
            else None
        )
        target_enemy_id = None if nearest_enemy is None else int(nearest_enemy.get("enemy_id", -1))
        target_position = None if nearest_enemy is None else tuple(nearest_enemy.get("position", ()))
        target_reachable = bool(nearest_enemy.get("reachable", False)) if nearest_enemy else False
        target_description = None if nearest_enemy is None else str(nearest_enemy.get("description", "") or "")
        can_cast = (
            bool(spell_key)
            and isinstance(spell, dict)
            and spell_purchased
            and not spell_regular_action
            and not spell_used_this_turn
            and self._can_cast_spell_shop_spell(str(spell_key))
        )

        spell_cast_applied = False
        spell_cast_executed = False
        spell_cast_reward = 0.0
        spell_units_affected = 0
        spell_heal_total = 0.0
        spell_moves_restored = 0.0
        moves_before_cast = float(self.moves or 0.0)
        actual_damage = 0.0
        units_hit = 0
        defeated_units = 0
        blocked_units = 0
        spell_enemy_defeated = False
        enemy_reward = 0.0
        ruin_clear_bonus_reward = 0.0
        newly_captured_objective_cities: List[str] = []
        newly_activated_settlement_territories: List[str] = []
        final_objective_reward = 0.0
        spell_effect_summary: Dict[str, object]
        if support_spec:
            spell_effect_summary = self._blue_stack_spell_effect_summary()
        else:
            spell_effect_summary = {
                "active_spell_ids": [],
                "debuff_types": [],
                "armor_delta": 0,
                "damage_multiplier": 1.0,
                "initiative_multiplier": 1.0,
                "accuracy_multiplier": 1.0,
            }

        if can_cast:
            cast_result = self._cast_from_spell_spec(
                source="spell_shop",
                spell_key=str(spell_key),
                spell_description=spell_description,
                spell_spec=spell_spec,
                mana_costs=mana_costs,
                spend_mana=True,
                enforce_turn_limit=True,
            )
            spell_cast_applied = bool(cast_result.get("spell_cast_applied", False))
            spell_cast_executed = bool(cast_result.get("spell_cast_executed", False))
            spell_cast_reward = float(cast_result.get("spell_cast_reward", 0.0) or 0.0)
            reward += float(cast_result.get("reward", 0.0) or 0.0)
            spell_units_affected = int(cast_result.get("spell_units_affected", 0) or 0)
            spell_heal_total = float(cast_result.get("spell_heal_total", 0.0) or 0.0)
            spell_moves_restored = float(
                cast_result.get("spell_moves_restored", 0.0) or 0.0
            )
            actual_damage = float(cast_result.get("spell_damage_total", 0.0) or 0.0)
            units_hit = int(cast_result.get("spell_damage_units_hit", 0) or 0)
            defeated_units = int(
                cast_result.get("spell_damage_units_defeated", 0) or 0
            )
            blocked_units = int(cast_result.get("spell_damage_units_blocked", 0) or 0)
            spell_effect_summary = dict(cast_result.get("spell_effect_summary", {}))
            target_enemy_id = cast_result.get("target_enemy_id")
            target_position = cast_result.get("target_enemy_position")
            target_reachable = bool(cast_result.get("target_enemy_reachable", False))
            target_description = cast_result.get("target_enemy_description")
            spell_enemy_defeated = bool(cast_result.get("spell_enemy_defeated", False))
            enemy_reward = float(cast_result.get("enemy_defeat_reward", 0.0) or 0.0)
            ruin_clear_bonus_reward = float(
                cast_result.get("ruin_clear_bonus_reward", 0.0) or 0.0
            )
            newly_captured_objective_cities = list(
                cast_result.get("newly_captured_objective_cities", []) or []
            )
            newly_activated_settlement_territories = list(
                cast_result.get("newly_activated_settlement_territories", []) or []
            )
            final_objective_reward = float(
                cast_result.get("final_objective_reward", 0.0) or 0.0
            )
        else:
            cast_result = {}

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "spell_cast_action": True,
            "spell_shop_cast_action": True,
            "spell_shop_cast_slot": int(idx),
            "spell_key": spell_key,
            "spell_description": spell_description,
            "spell_level": spell_level,
            "spell_kind": spell_kind,
            "spell_damage": float(spell_damage),
            "spell_damage_type": spell_damage_type,
            "spell_debuff_type": spell_debuff_type,
            "spell_summon_unit_name": summon_unit_name,
            "spell_buff_type": spell_buff_type,
            "spell_resistance_type": spell_resistance_type,
            "spell_heal_amount": float(spell_heal_amount),
            "spell_health_delta": int(spell_health_delta),
            "spell_moves_restore_fraction": float(spell_moves_restore_fraction),
            "spell_heal_total": float(spell_heal_total),
            "spell_moves_restored": float(spell_moves_restored),
            "spell_cast_applied": bool(spell_cast_applied),
            "spell_cast_executed": bool(spell_cast_executed),
            "spell_cast_reward": float(spell_cast_reward),
            "spell_cast_used_this_turn": bool(spell_used_this_turn),
            "spell_is_learned": bool(spell_purchased),
            "spell_shop_spell_purchased": bool(spell_purchased),
            "spell_shop_spell_has_regular_action": bool(spell_regular_action),
            "spell_has_map_offensive_actions": bool(offensive_spec),
            "spell_has_map_support_actions": bool(support_spec),
            "spell_use_costs": dict(mana_costs),
            "insufficient_mana": bool(
                not can_cast and mana_costs and not self._has_mana_for_costs(mana_costs)
            ),
            "target_enemy_id": target_enemy_id,
            "target_enemy_position": target_position,
            "target_enemy_reachable": bool(target_reachable),
            "target_enemy_description": target_description,
            "spell_damage_total": float(actual_damage),
            "spell_damage_units_hit": int(units_hit),
            "spell_damage_units_defeated": int(defeated_units),
            "spell_damage_units_blocked": int(blocked_units),
            "spell_units_affected": int(spell_units_affected),
            "spell_effect_active_ids": list(spell_effect_summary.get("active_spell_ids", [])),
            "spell_effect_types": list(
                spell_effect_summary.get(
                    "debuff_types" if offensive_spec else "buff_types",
                    [],
                )
            ),
            "spell_effect_resistance_types": list(
                spell_effect_summary.get("resistance_types", [])
            ),
            "spell_effect_armor_delta": int(spell_effect_summary.get("armor_delta", 0) or 0),
            "spell_effect_damage_multiplier": float(
                spell_effect_summary.get("damage_multiplier", 1.0) or 1.0
            ),
            "spell_effect_initiative_multiplier": float(
                spell_effect_summary.get("initiative_multiplier", 1.0) or 1.0
            ),
            "spell_effect_accuracy_multiplier": float(
                spell_effect_summary.get("accuracy_multiplier", 1.0) or 1.0
            ),
            "spell_effect_health_delta": int(spell_effect_summary.get("health_delta", 0) or 0),
            "spell_enemy_defeated": bool(spell_enemy_defeated),
            "enemy_defeat_reward": float(enemy_reward),
            "enemy_defeat_reward_base": float(self.reward_defeat_enemy) if spell_enemy_defeated else 0.0,
            "blue_exp_reward": 0.0,
            "blue_exp_raw": 0.0,
            "ruin_clear_bonus_reward": float(ruin_clear_bonus_reward),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "moves_before_cast": float(moves_before_cast),
            "moves_after_cast": float(self.moves or 0.0),
        }

        return self._finalize_map_spell_step(
            reward=reward,
            info=info,
            cast_result=cast_result,
        )

    def _step_unlock_scroll_magic(self, action: int):
        reward = 0.0
        scroll_magic_unlocked_before = bool(self.scroll_magic_unlocked)
        supported_entries = self.available_scroll_spell_entries()
        can_unlock = (
            int(action) == int(self.GRID_UNLOCK_SCROLL_MAGIC_ACTION)
            and self.mode == self.MODE_GRID
            and not scroll_magic_unlocked_before
            and float(self.gold or 0.0) >= float(self.SCROLL_MAGE_UNLOCK_GOLD_COST)
            and bool(supported_entries)
        )
        gold_spent = 0.0
        if can_unlock:
            gold_spent = float(self.SCROLL_MAGE_UNLOCK_GOLD_COST)
            self.gold = max(0.0, float(self.gold or 0.0) - gold_spent)
            self.scroll_magic_unlocked = True
            self._log(
                f"Маг для чтения свитков нанят за {gold_spent:g} gold. "
                "Свитки доступны до конца эпизода."
            )
        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "scroll_magic_unlock_action": True,
            "scroll_magic_unlocked_before": bool(scroll_magic_unlocked_before),
            "scroll_magic_unlocked_after": bool(self.scroll_magic_unlocked),
            "scroll_magic_unlock_gold_spent": float(gold_spent),
            "scroll_magic_unlock_available": bool(
                not scroll_magic_unlocked_before and bool(supported_entries)
            ),
            "scroll_magic_unlock_can_afford": bool(
                float(self.gold or 0.0) + gold_spent
                >= float(self.SCROLL_MAGE_UNLOCK_GOLD_COST)
            ),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }
        grid_obs = self._get_grid_obs()
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=False,
            truncated=False,
            info=info,
        )

    def _step_cast_scroll_spell(self, action: int):
        idx = int(action) - int(self.grid_scroll_cast_action_start)
        reward = 0.0
        slot_entries = self.scroll_cast_slot_entries()
        spell_data = dict(slot_entries[idx]) if 0 <= idx < len(slot_entries) else {}
        spell_key = str(spell_data.get("spell_id", "") or "")
        spell_description = str(
            spell_data.get("spell_name", "")
            or spell_data.get("spell_description", "")
            or spell_key
        ).strip()
        spell_level = int(spell_data.get("spell_level", 0) or 0) if spell_data else None
        spell_kind = str(spell_data.get("spell_kind", "") or "")
        spell_damage = float(spell_data.get("damage", 0.0) or 0.0)
        spell_damage_type = str(spell_data.get("damage_type", "") or "") or None
        spell_debuff_type = str(spell_data.get("debuff_type", "") or "") or None
        summon_unit_name = str(spell_data.get("summon_unit_name", "") or "") or None
        spell_buff_type = str(spell_data.get("buff_type", "") or "") or None
        spell_resistance_type = str(spell_data.get("resistance_type", "") or "") or None
        spell_heal_amount = float(spell_data.get("heal_amount", 0.0) or 0.0)
        spell_health_delta = int(spell_data.get("health_delta", 0) or 0)
        spell_moves_restore_fraction = float(
            spell_data.get("moves_restore_fraction", 0.0) or 0.0
        )
        item_name = str(spell_data.get("item_name", "") or "")
        scroll_copies_before = int(self.count_scroll_item(item_name))
        nearest_targetable_enemy = self._get_nearest_enemy_stack_for_mask(
            spell_targetable_only=True
        )
        nearest_any_enemy = self._get_nearest_enemy_stack_for_mask(
            spell_targetable_only=False
        )
        can_cast = bool(spell_key) and self._can_cast_scroll_spell(
            spell_data,
            nearest_targetable_enemy=nearest_targetable_enemy,
            nearest_any_enemy=nearest_any_enemy,
        )

        spell_cast_applied = False
        spell_cast_executed = False
        spell_cast_reward = 0.0
        spell_units_affected = 0
        spell_heal_total = 0.0
        spell_moves_restored = 0.0
        actual_damage = 0.0
        units_hit = 0
        defeated_units = 0
        blocked_units = 0
        spell_enemy_defeated = False
        enemy_reward = 0.0
        ruin_clear_bonus_reward = 0.0
        newly_captured_objective_cities: List[str] = []
        newly_activated_settlement_territories: List[str] = []
        final_objective_reward = 0.0
        spell_effect_summary: Dict[str, object] = (
            self._blue_stack_spell_effect_summary()
            if self._spell_kind_is_support(spell_kind)
            else {
                "active_spell_ids": [],
                "debuff_types": [],
                "armor_delta": 0,
                "damage_multiplier": 1.0,
                "initiative_multiplier": 1.0,
                "accuracy_multiplier": 1.0,
            }
        )
        target_enemy_id = None
        target_position = None
        target_reachable = False
        target_description = None

        if can_cast:
            cast_result = self._cast_from_spell_spec(
                source="scroll",
                spell_key=spell_key,
                spell_description=spell_description,
                spell_spec=spell_data,
                spend_mana=False,
                enforce_turn_limit=False,
                consume_item_name=item_name,
                nearest_targetable_enemy=nearest_targetable_enemy,
                nearest_any_enemy=nearest_any_enemy,
            )
            spell_cast_applied = bool(cast_result.get("spell_cast_applied", False))
            spell_cast_executed = bool(cast_result.get("spell_cast_executed", False))
            spell_cast_reward = float(cast_result.get("spell_cast_reward", 0.0) or 0.0)
            reward += float(cast_result.get("reward", 0.0) or 0.0)
            spell_units_affected = int(cast_result.get("spell_units_affected", 0) or 0)
            spell_heal_total = float(cast_result.get("spell_heal_total", 0.0) or 0.0)
            spell_moves_restored = float(
                cast_result.get("spell_moves_restored", 0.0) or 0.0
            )
            actual_damage = float(cast_result.get("spell_damage_total", 0.0) or 0.0)
            units_hit = int(cast_result.get("spell_damage_units_hit", 0) or 0)
            defeated_units = int(
                cast_result.get("spell_damage_units_defeated", 0) or 0
            )
            blocked_units = int(cast_result.get("spell_damage_units_blocked", 0) or 0)
            spell_effect_summary = dict(cast_result.get("spell_effect_summary", {}))
            target_enemy_id = cast_result.get("target_enemy_id")
            target_position = cast_result.get("target_enemy_position")
            target_reachable = bool(cast_result.get("target_enemy_reachable", False))
            target_description = cast_result.get("target_enemy_description")
            spell_enemy_defeated = bool(cast_result.get("spell_enemy_defeated", False))
            enemy_reward = float(cast_result.get("enemy_defeat_reward", 0.0) or 0.0)
            ruin_clear_bonus_reward = float(
                cast_result.get("ruin_clear_bonus_reward", 0.0) or 0.0
            )
            newly_captured_objective_cities = list(
                cast_result.get("newly_captured_objective_cities", []) or []
            )
            newly_activated_settlement_territories = list(
                cast_result.get("newly_activated_settlement_territories", []) or []
            )
            final_objective_reward = float(
                cast_result.get("final_objective_reward", 0.0) or 0.0
            )
            scroll_item_consumed = bool(cast_result.get("scroll_item_consumed", False))
            scroll_copies_after = int(cast_result.get("scroll_copies_after", 0) or 0)
        else:
            cast_result = {}
            scroll_item_consumed = False
            scroll_copies_after = scroll_copies_before

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "spell_cast_action": True,
            "scroll_cast_action": True,
            "scroll_cast_slot": int(idx),
            "scroll_item_name": item_name,
            "scroll_spell_key": spell_key,
            "scroll_spell_kind": spell_kind,
            "scroll_item_consumed": bool(scroll_item_consumed),
            "scroll_copies_before": int(scroll_copies_before),
            "scroll_copies_after": int(scroll_copies_after),
            "spell_key": spell_key,
            "spell_description": spell_description,
            "spell_level": spell_level,
            "spell_kind": spell_kind,
            "spell_damage": float(spell_damage),
            "spell_damage_type": spell_damage_type,
            "spell_debuff_type": spell_debuff_type,
            "spell_summon_unit_name": summon_unit_name,
            "spell_buff_type": spell_buff_type,
            "spell_resistance_type": spell_resistance_type,
            "spell_heal_amount": float(spell_heal_amount),
            "spell_health_delta": int(spell_health_delta),
            "spell_moves_restore_fraction": float(spell_moves_restore_fraction),
            "spell_heal_total": float(spell_heal_total),
            "spell_moves_restored": float(spell_moves_restored),
            "spell_cast_applied": bool(spell_cast_applied),
            "spell_cast_executed": bool(spell_cast_executed),
            "spell_cast_reward": float(spell_cast_reward),
            "spell_cast_used_this_turn": False,
            "spell_is_learned": False,
            "spell_use_costs": {},
            "insufficient_mana": False,
            "target_enemy_id": target_enemy_id,
            "target_enemy_position": target_position,
            "target_enemy_reachable": bool(target_reachable),
            "target_enemy_description": target_description,
            "spell_damage_total": float(actual_damage),
            "spell_damage_units_hit": int(units_hit),
            "spell_damage_units_defeated": int(defeated_units),
            "spell_damage_units_blocked": int(blocked_units),
            "spell_units_affected": int(spell_units_affected),
            "spell_effect_active_ids": list(
                spell_effect_summary.get("active_spell_ids", [])
            ),
            "spell_effect_types": list(
                spell_effect_summary.get(
                    "debuff_types"
                    if self._spell_kind_is_offensive(spell_kind)
                    else "buff_types",
                    [],
                )
            ),
            "spell_effect_resistance_types": list(
                spell_effect_summary.get("resistance_types", [])
            ),
            "spell_effect_armor_delta": int(
                spell_effect_summary.get("armor_delta", 0) or 0
            ),
            "spell_effect_damage_multiplier": float(
                spell_effect_summary.get("damage_multiplier", 1.0) or 1.0
            ),
            "spell_effect_initiative_multiplier": float(
                spell_effect_summary.get("initiative_multiplier", 1.0) or 1.0
            ),
            "spell_effect_accuracy_multiplier": float(
                spell_effect_summary.get("accuracy_multiplier", 1.0) or 1.0
            ),
            "spell_effect_health_delta": int(
                spell_effect_summary.get("health_delta", 0) or 0
            ),
            "spell_enemy_defeated": bool(spell_enemy_defeated),
            "enemy_defeat_reward": float(enemy_reward),
            "enemy_defeat_reward_base": float(self.reward_defeat_enemy)
            if spell_enemy_defeated
            else 0.0,
            "blue_exp_reward": 0.0,
            "blue_exp_raw": 0.0,
            "ruin_clear_bonus_reward": float(ruin_clear_bonus_reward),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }
        return self._finalize_map_spell_step(
            reward=reward,
            info=info,
            cast_result=cast_result,
        )

    def _step_upgrade_settlement(self, action: int):
        idx = action - self.grid_settlement_upgrade_action_start
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        settlement_name = None
        source_tile = None
        captured = False
        current_level = None
        next_level = None
        upgrade_cost = 0.0
        upgraded = False
        insufficient_gold = False
        at_max_level = False

        if 0 <= idx < len(self.settlement_upgrade_names):
            settlement_name = str(self.settlement_upgrade_names[idx])
            source_tile = self.legions_settlement_source_tile_by_name.get(settlement_name)
            captured = (
                settlement_name in self.legions_active_settlement_territory_capture_turn_by_name
            )
            current_level = int(self.legions_settlement_level_by_name.get(settlement_name, 1) or 1)
            at_max_level = current_level >= int(self.MAX_SETTLEMENT_LEVEL)
            upgrade_cost = self._settlement_upgrade_cost_for_level(current_level)
            insufficient_gold = (
                captured
                and not at_max_level
                and upgrade_cost > 0.0
                and float(self.gold or 0.0) < float(upgrade_cost)
            )
            next_level = min(int(self.MAX_SETTLEMENT_LEVEL), current_level + 1)
            if captured and not at_max_level and upgrade_cost > 0.0 and not insufficient_gold:
                self.gold -= float(upgrade_cost)
                self.legions_settlement_level_by_name[settlement_name] = int(next_level)
                upgraded = True
                self._log(
                    f'Город "{settlement_name}" улучшен: уровень {current_level} -> {next_level} '
                    f"за {upgrade_cost:g} золота."
                )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "settlement_upgrade_action": True,
            "settlement_upgrade_name": settlement_name,
            "settlement_upgrade_source_tile": tuple(source_tile) if source_tile is not None else None,
            "settlement_upgrade_captured": bool(captured),
            "settlement_upgrade_old_level": current_level,
            "settlement_upgrade_new_level": next_level if upgraded else current_level,
            "settlement_upgrade_cost": float(upgrade_cost),
            "settlement_upgrade_applied": bool(upgraded),
            "settlement_upgrade_insufficient_gold": bool(insufficient_gold),
            "settlement_upgrade_at_max_level": bool(at_max_level),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_equip_battle_item(self, action: int):
        slot_index = 0
        action_start = self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START
        if int(action) >= self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START:
            slot_index = 1
            action_start = self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START
        idx = int(action) - int(action_start)
        item_name = (
            self.BATTLE_EQUIPPABLE_ITEM_NAMES[idx]
            if 0 <= idx < len(self.BATTLE_EQUIPPABLE_ITEM_NAMES)
            else None
        )

        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        previous_item = None
        equipped = False
        equip_turn_locked = bool(self.battle_item_equip_used_this_turn)
        if (
            not equip_turn_locked
            and item_name is not None
            and 0 <= int(slot_index) < len(self.equipped_hero_items)
        ):
            previous_item = self.equipped_hero_items[int(slot_index)]
            equipped = self._equip_battle_item(int(slot_index), str(item_name))
            if equipped:
                self.battle_item_equip_used_this_turn = True
                self.battle_items_equipped_total += 1
                if not previous_item:
                    reward = float(self.reward_battle_item_equip)
                self._log(
                    f"Боевой слот {int(slot_index) + 1}: экипирован предмет '{str(item_name)}'"
                )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "equip_battle_item_action": True,
            "equip_battle_item_slot": int(slot_index) + 1,
            "equip_battle_item_name": str(item_name or ""),
            "equip_battle_item_previous": previous_item,
            "equip_battle_item_applied": bool(equipped),
            "equip_battle_item_turn_locked": bool(equip_turn_locked),
            "equip_battle_item_reward": float(reward),
            "battle_items_equipped_total": int(self.battle_items_equipped_total),
            "battle_items_used_total": int(self.battle_items_used_total),
            "equipped_hero_items": list(self.equipped_hero_items),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_heal_with_bottle(self, action: int):
        """Применяет подходящую банку лечения к указанной позиции BLUE в режиме grid."""
        idx = action - self.GRID_BOTTLE_ACTION_START
        target_pos = self.GRID_BOTTLE_POSITIONS[idx] if 0 <= idx < len(self.GRID_BOTTLE_POSITIONS) else None

        # Лечение не продвигает ход карты и не даёт штрафа/таймера.
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        healed_amount, unit_name = (0.0, None)
        selected_heal_amount = 0.0
        selected_bottle_kind = None
        if target_pos is not None:
            selected_heal_amount, selected_bottle_kind = self._select_heal_bottle_for_position(
                target_pos
            )
        if target_pos is not None and selected_bottle_kind is not None:
            if selected_bottle_kind == "ointment":
                if self._consume_battle_equipable_item(self.HEALING_OINTMENT_ITEM_NAME):
                    healed_amount, unit_name = self._heal_unit_at_position(
                        position=target_pos,
                        heal_amount=selected_heal_amount,
                        source_label=self.HEALING_OINTMENT_ITEM_NAME,
                    )
                    if healed_amount <= 0.0:
                        self._append_hero_item(self.HEALING_OINTMENT_ITEM_NAME)
                else:
                    selected_bottle_kind = None
                    selected_heal_amount = 0.0
            else:
                healed_amount, unit_name = self._heal_unit_at_position(
                    position=target_pos,
                    heal_amount=selected_heal_amount,
                    source_label=(
                        "Банка исцеления"
                        if selected_bottle_kind == "small"
                        else "Бутыль лечения"
                    ),
                )
                if healed_amount > 0.0:
                    self._consume_battle_equipable_item(
                        self.BONUS_LARGE_HEAL_ITEM_NAME
                        if selected_bottle_kind == "large"
                        else self.BONUS_SMALL_HEAL_ITEM_NAME
                    )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "heal_action": True,
            "heal_target_pos": target_pos,
            "healed_amount": healed_amount,
            "healed_unit_name": unit_name,
            "selected_heal_bottle_kind": selected_bottle_kind,
            "selected_heal_bottle_amount": selected_heal_amount,
            "heal_bottles_used": self.heal_bottles_used,
            "extra_heal_bottles": self.extra_heal_bottles,
            "heal_bottles_left": self._heal_bottles_left(),
            "max_heal_bottles": self._max_heal_bottles_available(),
            "healing_bottles_used": self.healing_bottles_used,
            "extra_healing_bottles": self.extra_healing_bottles,
            "healing_bottles_left": max(
                0, self._max_healing_bottles_available() - self.healing_bottles_used
            ),
            "max_healing_bottles": self._max_healing_bottles_available(),
            "healing_ointment_left": self._count_hero_item(self.HEALING_OINTMENT_ITEM_NAME),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_revive_with_bottle(self, action: int):
        """Применяет бутыль воскрешения к указанной позиции BLUE в режиме grid."""
        idx = action - self.GRID_REVIVE_ACTION_START
        target_pos = (
            self.REVIVE_BOTTLE_POSITIONS[idx]
            if 0 <= idx < len(self.REVIVE_BOTTLE_POSITIONS)
            else None
        )

        # Воскрешение не продвигает ход карты и не даёт штрафа/таймера.
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        revived, unit_name = (False, None)
        if target_pos is not None and self._consume_battle_equipable_item(
            self.BONUS_REVIVE_ITEM_NAME
        ):
            revived, unit_name = self._revive_unit_at_position(position=target_pos)

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "revive_action": True,
            "revive_target_pos": target_pos,
            "revived": revived,
            "revived_unit_name": unit_name,
            "revive_bottles_used": self.revive_bottles_used,
            "extra_revive_bottles": self.extra_revive_bottles,
            "revive_bottles_left": self._revive_bottles_left(),
            "max_revive_bottles": self._max_revive_bottles_available(),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _apply_invulnerability_potion_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0:
                return False, None
            if int(position) in self.active_invulnerability_potion_positions:
                return False, None
            self.active_invulnerability_potion_positions.add(int(position))
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"Зелье неуязвимости: {name} (pos {position}) получает "
                f"+{self.INVULNERABILITY_POTION_ARMOR_BONUS} брони до конца текущего хода"
            )
            return True, name
        return False, None

    def _apply_strength_potion_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0:
                return False, None
            if int(position) in self.active_strength_potion_positions:
                return False, None
            self.active_strength_potion_positions.add(int(position))
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"Зелье силы: {name} (pos {position}) получает +30% урона "
                "до конца текущего хода"
            )
            return True, name
        return False, None

    def _can_apply_single_turn_effect_position(
        self,
        position: int,
        item_name: str,
        active_positions: set[int],
    ) -> bool:
        if not self._hero_item_available(item_name):
            return False
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return max_hp > 0 and hp > 0 and int(position) not in active_positions
        return False

    def _can_apply_energy_elixir_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.ENERGY_ELIXIR_ITEM_NAME,
            self.active_energy_elixir_positions,
        )

    def _can_apply_haste_elixir_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.HASTE_ELIXIR_ITEM_NAME,
            self.active_haste_elixir_positions,
        )

    def _can_apply_fire_ward_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.FIRE_WARD_ITEM_NAME,
            self.active_fire_ward_positions,
        )

    def _can_apply_earth_ward_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.EARTH_WARD_ITEM_NAME,
            self.active_earth_ward_positions,
        )

    def _can_apply_water_ward_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.WATER_WARD_ITEM_NAME,
            self.active_water_ward_positions,
        )

    def _can_apply_air_ward_position(self, position: int) -> bool:
        return self._can_apply_single_turn_effect_position(
            position,
            self.AIR_WARD_ITEM_NAME,
            self.active_air_ward_positions,
        )

    def _can_apply_permanent_elixir_position(self, position: int, item_name: str) -> bool:
        if not self._hero_item_available(item_name):
            return False
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return max_hp > 0 and hp > 0
        return False

    def _can_apply_titan_elixir_position(self, position: int) -> bool:
        return self._can_apply_permanent_elixir_position(
            position,
            self.TITAN_ELIXIR_ITEM_NAME,
        )

    def _can_apply_supreme_elixir_position(self, position: int) -> bool:
        return self._can_apply_permanent_elixir_position(
            position,
            self.SUPREME_ELIXIR_ITEM_NAME,
        )

    def _apply_position_flag_effect(
        self,
        position: int,
        active_positions: set[int],
        log_message: str,
    ) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0 or int(position) in active_positions:
                return False, None
            active_positions.add(int(position))
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(log_message.format(name=name, position=int(position)))
            return True, name
        return False, None

    def _apply_energy_elixir_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_energy_elixir_positions,
            "Эликсир энергии: {name} (pos {position}) получает +50% урона до конца текущего хода",
        )

    def _apply_haste_elixir_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_haste_elixir_positions,
            "Эликсир быстроты: {name} (pos {position}) получает +60% инициативы до конца текущего хода",
        )

    def _apply_fire_ward_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_fire_ward_positions,
            "Защита от магии Огня: {name} (pos {position}) получает resistance Fire до конца текущего хода",
        )

    def _apply_earth_ward_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_earth_ward_positions,
            "Защита от магии Земли: {name} (pos {position}) получает resistance Earth до конца текущего хода",
        )

    def _apply_water_ward_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_water_ward_positions,
            "Защита от магии Воды: {name} (pos {position}) получает resistance Water до конца текущего хода",
        )

    def _apply_air_ward_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        return self._apply_position_flag_effect(
            position,
            self.active_air_ward_positions,
            "Защита от магии Воздуха: {name} (pos {position}) получает resistance Air до конца текущего хода",
        )

    def _apply_titan_elixir_bonus_to_unit(
        self,
        unit: Dict,
        *,
        increment_uses: bool = True,
    ) -> None:
        base_damage = self._normalize_damage_value(unit.get("damage", 0))
        base_damage_secondary = self._normalize_damage_value(unit.get("damage_secondary", 0))
        unit["damage"] = self._normalize_damage_value(
            float(base_damage) * float(self.TITAN_ELIXIR_DAMAGE_MULTIPLIER)
        )
        unit["damage_secondary"] = self._normalize_damage_value(
            float(base_damage_secondary) * float(self.TITAN_ELIXIR_DAMAGE_MULTIPLIER)
        )
        unit["original_damage"] = int(unit["damage"])
        if increment_uses:
            unit["campaign_titan_elixir_uses"] = (
                max(0, int(unit.get("campaign_titan_elixir_uses", 0) or 0)) + 1
            )

    def _apply_supreme_elixir_bonus_to_unit(
        self,
        unit: Dict,
        *,
        increment_uses: bool = True,
    ) -> None:
        hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
        max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
        new_max_hp = self._normalize_health_value(
            float(max_hp) * float(self.SUPREME_ELIXIR_HEALTH_MULTIPLIER)
        )
        if new_max_hp <= int(round(max_hp)):
            new_max_hp = int(round(max_hp)) + 1
        new_hp = min(
            new_max_hp,
            self._normalize_health_value(
                float(hp) * float(self.SUPREME_ELIXIR_HEALTH_MULTIPLIER)
            ),
        )
        unit["maxhp"] = int(new_max_hp)
        unit["max_health"] = int(new_max_hp)
        unit["hp"] = int(new_hp)
        unit["health"] = int(new_hp)
        if increment_uses:
            unit["campaign_supreme_elixir_uses"] = (
                max(0, int(unit.get("campaign_supreme_elixir_uses", 0) or 0)) + 1
            )

    def _reapply_persistent_elixir_bonuses_to_promoted_unit(
        self,
        source_unit: Dict,
        upgraded_unit: Dict,
    ) -> None:
        upgraded_unit.pop("campaign_titan_elixir_uses", None)
        upgraded_unit.pop("campaign_supreme_elixir_uses", None)

        titan_uses = max(0, int(source_unit.get("campaign_titan_elixir_uses", 0) or 0))
        supreme_uses = max(0, int(source_unit.get("campaign_supreme_elixir_uses", 0) or 0))

        for _ in range(titan_uses):
            self._apply_titan_elixir_bonus_to_unit(upgraded_unit, increment_uses=True)
        for _ in range(supreme_uses):
            self._apply_supreme_elixir_bonus_to_unit(upgraded_unit, increment_uses=True)

    def _apply_titan_elixir_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0:
                return False, None

            self._apply_titan_elixir_bonus_to_unit(unit, increment_uses=True)
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"Эликсир Силы титана: {name} (pos {position}) получает "
                f"+10% к урону навсегда на эпизод"
            )
            return True, name
        return False, None

    def _apply_supreme_elixir_to_position(self, position: int) -> Tuple[bool, Optional[str]]:
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp <= 0:
                return False, None

            self._apply_supreme_elixir_bonus_to_unit(unit, increment_uses=True)
            name = str(unit.get("name", f"pos_{position}") or f"pos_{position}")
            self._log(
                f"Эликсир Всевышнего: {name} (pos {position}) получает "
                f"+15% к max HP навсегда на эпизод"
            )
            return True, name
        return False, None

    def _step_apply_single_turn_position_item(
        self,
        *,
        action: int,
        action_start: int,
        positions: List[int],
        item_name: str,
        can_apply_fn,
        apply_fn,
        info_key: str,
    ):
        idx = action - action_start
        target_pos = positions[idx] if 0 <= idx < len(positions) else None
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        applied = False
        consumed = False
        unit_name = None
        if (
            target_pos is not None
            and can_apply_fn(target_pos)
            and self._consume_hero_item(item_name)
        ):
            consumed = True
            applied, unit_name = apply_fn(target_pos)
            if not applied:
                self._append_hero_item(item_name)
                consumed = False
            else:
                reward = self._combat_potion_reward_value()
                self.combat_potion_battle_bonus_pending = True

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            info_key: True,
            "combat_potion_action": True,
            "combat_potion_item": item_name,
            "combat_potion_target_pos": target_pos,
            "combat_potion_applied": applied,
            "combat_potion_consumed": consumed,
            "combat_potion_unit_name": unit_name,
            "combat_potion_reward": float(reward),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_apply_invulnerability_potion(self, action: int):
        idx = action - self.GRID_INVULNERABILITY_ACTION_START
        target_pos = (
            self.INVULNERABILITY_POTION_POSITIONS[idx]
            if 0 <= idx < len(self.INVULNERABILITY_POTION_POSITIONS)
            else None
        )
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        applied = False
        consumed = False
        unit_name = None
        if (
            target_pos is not None
            and self._can_apply_invulnerability_potion_position(target_pos)
            and self._consume_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME)
        ):
            consumed = True
            applied, unit_name = self._apply_invulnerability_potion_to_position(target_pos)
            if not applied:
                self._append_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME)
                consumed = False
            else:
                reward = self._combat_potion_reward_value()
                self.combat_potion_battle_bonus_pending = True

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "invulnerability_potion_action": True,
            "combat_potion_action": True,
            "combat_potion_item": self.INVULNERABILITY_POTION_ITEM_NAME,
            "combat_potion_target_pos": target_pos,
            "combat_potion_applied": applied,
            "combat_potion_consumed": consumed,
            "combat_potion_unit_name": unit_name,
            "combat_potion_reward": float(reward),
            "invulnerability_potions_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_potions_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
            "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
            "active_strength_positions": sorted(self.active_strength_potion_positions),
            "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_apply_strength_potion(self, action: int):
        idx = action - self.GRID_STRENGTH_ACTION_START
        target_pos = (
            self.STRENGTH_POTION_POSITIONS[idx]
            if 0 <= idx < len(self.STRENGTH_POTION_POSITIONS)
            else None
        )
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        applied = False
        consumed = False
        unit_name = None
        if (
            target_pos is not None
            and self._can_apply_strength_potion_position(target_pos)
            and self._consume_hero_item(self.STRENGTH_POTION_ITEM_NAME)
        ):
            consumed = True
            applied, unit_name = self._apply_strength_potion_to_position(target_pos)
            if not applied:
                self._append_hero_item(self.STRENGTH_POTION_ITEM_NAME)
                consumed = False
            else:
                reward = self._combat_potion_reward_value()
                self.combat_potion_battle_bonus_pending = True

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "strength_potion_action": True,
            "combat_potion_action": True,
            "combat_potion_item": self.STRENGTH_POTION_ITEM_NAME,
            "combat_potion_target_pos": target_pos,
            "combat_potion_applied": applied,
            "combat_potion_consumed": consumed,
            "combat_potion_unit_name": unit_name,
            "combat_potion_reward": float(reward),
            "invulnerability_potions_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_potions_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
            "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
            "active_strength_positions": sorted(self.active_strength_potion_positions),
            "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_apply_energy_elixir(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_ENERGY_ACTION_START,
            positions=self.ENERGY_ELIXIR_POSITIONS,
            item_name=self.ENERGY_ELIXIR_ITEM_NAME,
            can_apply_fn=self._can_apply_energy_elixir_position,
            apply_fn=self._apply_energy_elixir_to_position,
            info_key="energy_elixir_action",
        )

    def _step_apply_haste_elixir(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_HASTE_ACTION_START,
            positions=self.HASTE_ELIXIR_POSITIONS,
            item_name=self.HASTE_ELIXIR_ITEM_NAME,
            can_apply_fn=self._can_apply_haste_elixir_position,
            apply_fn=self._apply_haste_elixir_to_position,
            info_key="haste_elixir_action",
        )

    def _step_apply_fire_ward(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_FIRE_WARD_ACTION_START,
            positions=self.FIRE_WARD_POSITIONS,
            item_name=self.FIRE_WARD_ITEM_NAME,
            can_apply_fn=self._can_apply_fire_ward_position,
            apply_fn=self._apply_fire_ward_to_position,
            info_key="fire_ward_action",
        )

    def _step_apply_earth_ward(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_EARTH_WARD_ACTION_START,
            positions=self.EARTH_WARD_POSITIONS,
            item_name=self.EARTH_WARD_ITEM_NAME,
            can_apply_fn=self._can_apply_earth_ward_position,
            apply_fn=self._apply_earth_ward_to_position,
            info_key="earth_ward_action",
        )

    def _step_apply_water_ward(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_WATER_WARD_ACTION_START,
            positions=self.WATER_WARD_POSITIONS,
            item_name=self.WATER_WARD_ITEM_NAME,
            can_apply_fn=self._can_apply_water_ward_position,
            apply_fn=self._apply_water_ward_to_position,
            info_key="water_ward_action",
        )

    def _step_apply_air_ward(self, action: int):
        return self._step_apply_single_turn_position_item(
            action=action,
            action_start=self.GRID_AIR_WARD_ACTION_START,
            positions=self.AIR_WARD_POSITIONS,
            item_name=self.AIR_WARD_ITEM_NAME,
            can_apply_fn=self._can_apply_air_ward_position,
            apply_fn=self._apply_air_ward_to_position,
            info_key="air_ward_action",
        )

    def _step_apply_permanent_position_item(
        self,
        *,
        action: int,
        action_start: int,
        positions: List[int],
        item_name: str,
        can_apply_fn,
        apply_fn,
        info_key: str,
    ):
        idx = action - action_start
        target_pos = positions[idx] if 0 <= idx < len(positions) else None
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        applied = False
        consumed = False
        unit_name = None
        if (
            target_pos is not None
            and can_apply_fn(target_pos)
            and self._consume_hero_item(item_name)
        ):
            consumed = True
            applied, unit_name = apply_fn(target_pos)
            if not applied:
                self._append_hero_item(item_name)
                consumed = False

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            info_key: True,
            "persistent_elixir_action": True,
            "persistent_elixir_item": item_name,
            "persistent_elixir_target_pos": target_pos,
            "persistent_elixir_applied": applied,
            "persistent_elixir_consumed": consumed,
            "persistent_elixir_unit_name": unit_name,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
        }
        info.update(self._merchant_context_info(self.grid_env.agent_pos))

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_apply_titan_elixir(self, action: int):
        return self._step_apply_permanent_position_item(
            action=action,
            action_start=self.GRID_TITAN_ELIXIR_ACTION_START,
            positions=self.TITAN_ELIXIR_POSITIONS,
            item_name=self.TITAN_ELIXIR_ITEM_NAME,
            can_apply_fn=self._can_apply_titan_elixir_position,
            apply_fn=self._apply_titan_elixir_to_position,
            info_key="titan_elixir_action",
        )

    def _step_apply_supreme_elixir(self, action: int):
        return self._step_apply_permanent_position_item(
            action=action,
            action_start=self.GRID_SUPREME_ELIXIR_ACTION_START,
            positions=self.SUPREME_ELIXIR_POSITIONS,
            item_name=self.SUPREME_ELIXIR_ITEM_NAME,
            can_apply_fn=self._can_apply_supreme_elixir_position,
            apply_fn=self._apply_supreme_elixir_to_position,
            info_key="supreme_elixir_action",
        )

    def _step_battle(self, action: int):
        """Шаг в режиме боя."""
        if self.battle_env is None:
            # Защита от некорректного состояния
            self.mode = self.MODE_GRID
            self._clear_current_battle_context()
            grid_obs = self._get_grid_obs()
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"error": "no_battle_env", "agent_pos": self.grid_env.agent_pos},
            )

        battle_info = {}

        # Проверка: бой уже завершён (например, RED победил в свой ход)
        if self.battle_env.winner is not None:
            winner = self.battle_env.winner
            obs = self.battle_env._obs()
            reward = 0.0
            battle_reward_raw = 0.0
            battle_item_use_reward = 0.0
            terminated = True
            truncated = False
        else:
            # safety: action space CampaignEnv > BattleEnv, clamp to valid range
            action = int(action)
            if action < 0:
                action = 0
            if action >= self.battle_env.action_space.n:
                action = self.battle_env.action_space.n - 1
            obs, battle_reward_raw, terminated, truncated, battle_info = self.battle_env.step(action)
            if isinstance(battle_info, dict):
                consumed_item_name = str(
                    battle_info.get("battle_hero_item_name", "") or ""
                )
                if (
                    battle_info.get("battle_hero_item_action")
                    and battle_info.get("battle_hero_item_consumed")
                    and consumed_item_name
                ):
                    self._consume_battle_equipable_item(consumed_item_name)
                    if battle_info.get("battle_hero_item_applied"):
                        self.battle_items_used_total += 1
                        battle_item_use_reward = float(self.reward_battle_item_use)
                    else:
                        battle_item_use_reward = 0.0
                else:
                    battle_item_use_reward = 0.0
                equipped_items = battle_info.get("equipped_hero_items")
                if isinstance(equipped_items, list) and str(
                    self.current_battle_context.get("kind", "hero") or "hero"
                ) != "summon_spell":
                    self.equipped_hero_items = list(equipped_items)
                    self._sync_equipped_hero_items()
            else:
                battle_item_use_reward = 0.0
            reward = self.battle_reward_scale * float(battle_reward_raw) + float(
                battle_item_use_reward
            )

        battle_context_kind = str(self.current_battle_context.get("kind", "hero") or "hero")
        info = {
            "mode": "battle",
            "enemy_id": self.current_enemy_id,
            "battle_step": True,
            "battle_context_kind": battle_context_kind,
            "battle_context_spell_key": str(
                self.current_battle_context.get("spell_key", "") or ""
            ),
            "battle_summoned_unit_name": str(
                self.current_battle_context.get("summoned_unit_name", "") or ""
            ),
            "battle_reward_raw": float(battle_reward_raw),
            "battle_reward_scaled": float(self.battle_reward_scale * float(battle_reward_raw)),
            "battle_item_use_reward": float(battle_item_use_reward),
            "battle_items_equipped_total": int(self.battle_items_equipped_total),
            "battle_items_used_total": int(self.battle_items_used_total),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "heroitems": list(self.heroitems),
            "equipped_hero_items": list(self.equipped_hero_items),
        }
        if isinstance(battle_info, dict):
            info.update(battle_info)
        info.update(self._artifact_state_info())

        if terminated or truncated:
            winner = self.battle_env.winner
            info["battle_winner"] = winner

            if battle_context_kind == "summon_spell":
                return self._finalize_summon_spell_battle(
                    obs=obs,
                    reward=reward,
                    info=info,
                    winner=winner,
                )

            if winner == "blue":
                exp_reward, exp_raw = self._compute_blue_exp_reward()
                survival_reward, blue_alive_ratio, blue_hp_ratio = self._compute_blue_survival_reward()
                ruin_clear_bonus_reward = (
                    float(self.reward_ruin_clear_bonus)
                    if int(self.current_enemy_id or -1) in self.RUIN_REWARD_BY_ENEMY_ID
                    else 0.0
                )
                enemy_reward = self._compute_enemy_defeat_reward(self.current_enemy_id)
                reward += enemy_reward + exp_reward + survival_reward
                info["enemy_defeat_reward"] = enemy_reward
                info["enemy_defeat_reward_base"] = float(self.reward_defeat_enemy)
                info["ruin_clear_bonus_reward"] = float(ruin_clear_bonus_reward)
                info["blue_exp_reward"] = exp_reward
                info["blue_exp_raw"] = exp_raw
                info["blue_survival_reward"] = survival_reward
                info["blue_alive_ratio"] = blue_alive_ratio
                info["blue_hp_ratio"] = blue_hp_ratio

                # Победа: помечаем врага побеждённым
                self._log(f"=== ПОБЕДА В БОЮ ПРОТИВ ВРАГА {self.current_enemy_id}! ===")
                self.grid_env.mark_enemy_defeated(self.current_enemy_id)
                self._clear_enemy_map_spell_effects_for_enemy(self.current_enemy_id)

                # Восстанавливаем и воскрешаем всех юнитов BLUE после победы
                if self.persist_blue_hp:
                    self._save_blue_state()
                    self._log("Состояние BLUE сохранено (без автолечения и воскрешений)")

                upgrade_count = 0
                if self.battle_env is not None:
                    self._log(f"Опыт за бой: {self.battle_env.last_battle_exp:g}")
                    for name in getattr(self.battle_env, "last_levelups", []) or []:
                        self._log(f"Уровень юнита {name} повышен")
                    upgrade_count = self._log_turns_into_levelups()
                upgrade_reward = float(upgrade_count) * self.reward_unit_upgrade
                reward += upgrade_reward
                info["unit_upgrades"] = int(upgrade_count)
                info["unit_upgrade_reward"] = float(upgrade_reward)
                info.update(self._grant_ruin_reward(self.current_enemy_id))

                # Возвращаемся в grid режим
                self.mode = self.MODE_GRID
                self.battle_env = None
                self._clear_current_battle_context()
                info["battle_result"] = "victory"
                info["mode"] = "grid"
                info["agent_pos"] = self._restore_agent_to_battle_origin()
                info["enemies_alive"] = dict(self.grid_env.enemies_alive)

                objective_cities_captured_before = len(self.captured_objective_cities)
                newly_captured_objective_cities = self._capture_objective_city_if_cleared(
                    self.current_enemy_id
                )
                info["objective_cities_captured_total"] = sorted(self.captured_objective_cities)
                newly_activated_settlement_territories = (
                    self._activate_legions_settlement_territory_if_cleared(self.current_enemy_id)
                )
                if newly_activated_settlement_territories:
                    info["legions_settlement_territories_activated"] = list(
                        newly_activated_settlement_territories
                    )
                    for settlement_name in newly_activated_settlement_territories:
                        self._log(
                            f"=== ЗЕМЛЯ ЛЕГИОНОВ НАЧИНАЕТ РАСПРОСТРАНЯТЬСЯ ИЗ ПОСЕЛЕНИЯ {settlement_name} ==="
                        )

                if newly_captured_objective_cities:
                    final_objective_reward = self._compute_final_objective_reward(
                        len(newly_captured_objective_cities),
                        captured_before=objective_cities_captured_before,
                    )
                    reward += final_objective_reward
                    info["captured_objective_cities"] = list(newly_captured_objective_cities)
                    info["final_objective_reward"] = float(final_objective_reward)
                    for city_name in newly_captured_objective_cities:
                        self._log(f"=== ЗАХВАЧЕН ГОРОД {city_name}: НАГРАДА {final_objective_reward:g} ===")

                if self._all_objective_cities_captured():
                    self._log("=== ЗАХВАЧЕНЫ ВСЕ ЦЕЛЕВЫЕ ГОРОДА: ПОБЕДА В КАМПАНИИ ===")
                    grid_obs = self._get_grid_obs()
                    info["campaign_result"] = "victory"
                    info["campaign_victory_reason"] = "objective_cities_cleared"
                    return self._finalize_grid_step_result(
                        grid_obs=grid_obs,
                        reward=reward,
                        terminated=True,
                        truncated=False,
                        info=info,
                    )

                # Проверяем полную победу
                if self.grid_env.all_enemies_defeated():
                    self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
                    grid_obs = self._get_grid_obs()
                    info["campaign_result"] = "victory"
                    info["campaign_victory_reason"] = "all_enemies_defeated"
                    return self._finalize_grid_step_result(
                        grid_obs=grid_obs,
                        reward=reward,
                        terminated=True,
                        truncated=False,
                        info=info,
                    )

                grid_obs = self._get_grid_obs()
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=False,
                    truncated=False,
                    info=info,
                )

            else:
                # Поражение: конец эпизода
                reward += self.reward_loss
                self._clear_battle_origin()
                self._clear_current_battle_context()
                self._log(f"=== ПОРАЖЕНИЕ В БОЮ ПРОТИВ ВРАГА {self.current_enemy_id}! ===")
                self._log("=== КАМПАНИЯ ПРОВАЛЕНА ===")
                info["battle_result"] = "defeat"
                info["campaign_result"] = "defeat"
                if self.battle_env is not None:
                    self._log(f"Опыт за бой: {self.battle_env.last_battle_exp:g}")
                    for name in getattr(self.battle_env, "last_levelups", []) or []:
                        self._log(f"Уровень юнита {name} повышен")
                    self._log_turns_into_levelups()
                return self._build_obs(battle_obs=obs), reward, True, False, info

        info["battle_ongoing"] = True
        return self._build_obs(battle_obs=obs), reward, terminated, truncated, info

    def _finalize_summon_spell_battle(
        self,
        *,
        obs: np.ndarray,
        reward: float,
        info: Dict[str, object],
        winner: Optional[str],
    ):
        enemy_id = self.current_enemy_id
        info["blue_exp_reward"] = 0.0
        info["blue_exp_raw"] = 0.0
        info["blue_survival_reward"] = 0.0
        info["blue_alive_ratio"] = 0.0
        info["blue_hp_ratio"] = 0.0
        info["enemy_defeat_reward"] = 0.0
        info["enemy_defeat_reward_base"] = 0.0
        info["ruin_clear_bonus_reward"] = 0.0

        if winner == "blue":
            try:
                normalized_enemy_id = int(enemy_id)
            except (TypeError, ValueError):
                normalized_enemy_id = None
            if normalized_enemy_id is not None:
                self.summon_hero_battle_bonus_enemy_ids_this_turn.discard(normalized_enemy_id)
                self.summon_hero_battle_bonus_pending = bool(
                    self.summon_hero_battle_bonus_enemy_ids_this_turn
                )
            enemy_reward = self._compute_enemy_defeat_reward(enemy_id)
            reward += enemy_reward
            info["enemy_defeat_reward"] = float(enemy_reward)
            info["enemy_defeat_reward_base"] = float(self.reward_defeat_enemy)
            if int(enemy_id or -1) in self.RUIN_REWARD_BY_ENEMY_ID:
                info["ruin_clear_bonus_reward"] = float(self.reward_ruin_clear_bonus)

            self._log(
                f"=== ПРИЗВАННЫЙ ЮНИТ ПОБЕДИЛ В БОЮ ПРОТИВ ВРАГА {enemy_id}! ==="
            )
            self.grid_env.mark_enemy_defeated(enemy_id)
            self._clear_enemy_map_spell_effects_for_enemy(enemy_id)

            objective_cities_captured_before = len(self.captured_objective_cities)
            newly_captured_objective_cities = self._capture_objective_city_if_cleared(enemy_id)
            newly_activated_settlement_territories = (
                self._activate_legions_settlement_territory_if_cleared(enemy_id)
            )
            info.update(self._grant_ruin_reward(enemy_id))
            info["objective_cities_captured_total"] = sorted(self.captured_objective_cities)
            if newly_activated_settlement_territories:
                info["legions_settlement_territories_activated"] = list(
                    newly_activated_settlement_territories
                )
                for settlement_name in newly_activated_settlement_territories:
                    self._log(
                        f"=== ЗЕМЛЯ ЛЕГИОНОВ НАЧИНАЕТ РАСПРОСТРАНЯТЬСЯ ИЗ ПОСЕЛЕНИЯ {settlement_name} ==="
                    )

            if newly_captured_objective_cities:
                final_objective_reward = self._compute_final_objective_reward(
                    len(newly_captured_objective_cities),
                    captured_before=objective_cities_captured_before,
                )
                reward += final_objective_reward
                info["captured_objective_cities"] = list(newly_captured_objective_cities)
                info["final_objective_reward"] = float(final_objective_reward)

            self.mode = self.MODE_GRID
            self.battle_env = None
            self._clear_current_battle_context()
            info["battle_result"] = "victory"
            info["mode"] = "grid"
            info["agent_pos"] = self._restore_agent_to_battle_origin()
            info["enemies_alive"] = dict(self.grid_env.enemies_alive)

            if self._all_objective_cities_captured():
                self._log("=== ЗАХВАЧЕНЫ ВСЕ ЦЕЛЕВЫЕ ГОРОДА: ПОБЕДА В КАМПАНИИ ===")
                grid_obs = self._get_grid_obs()
                info["campaign_result"] = "victory"
                info["campaign_victory_reason"] = "objective_cities_cleared"
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                )

            if self.grid_env.all_enemies_defeated():
                self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
                grid_obs = self._get_grid_obs()
                info["campaign_result"] = "victory"
                info["campaign_victory_reason"] = "all_enemies_defeated"
                return self._finalize_grid_step_result(
                    grid_obs=grid_obs,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                )

            grid_obs = self._get_grid_obs()
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=reward,
                terminated=False,
                truncated=False,
                info=info,
            )

        self._save_enemy_state_from_battle(enemy_id)
        if winner == "red":
            self._log(
                f"=== ПРИЗВАННЫЙ ЮНИТ ПОТЕРПЕЛ ПОРАЖЕНИЕ В БОЮ ПРОТИВ ВРАГА {enemy_id}! ==="
            )
            info["battle_result"] = "defeat"
        else:
            self._log(
                f"=== БОЙ ПРИЗЫВНОГО ЮНИТА ПРОТИВ ВРАГА {enemy_id} ЗАВЕРШЁН БЕЗ ПОБЕДИТЕЛЯ ==="
            )
            info["battle_result"] = "timeout"

        self.mode = self.MODE_GRID
        self.battle_env = None
        self._clear_current_battle_context()
        info["mode"] = "grid"
        info["agent_pos"] = self._restore_agent_to_battle_origin()
        info["enemies_alive"] = dict(self.grid_env.enemies_alive)
        info["summon_enemy_state_saved"] = True

        grid_obs = self._get_grid_obs()
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=False,
            truncated=False,
            info=info,
        )

    def _restore_agent_to_battle_origin(self) -> Tuple[int, int]:
        """Возвращает агента на клетку, с которой был инициирован текущий бой."""
        if self.battle_origin_pos is None:
            return tuple(self.grid_env.agent_pos)
        origin = (int(self.battle_origin_pos[0]), int(self.battle_origin_pos[1]))
        self.grid_env.agent_pos = origin
        self.battle_origin_pos = None
        return origin

    def _clear_battle_origin(self) -> None:
        self.battle_origin_pos = None

    @staticmethod
    def _normalize_armor_value(value: object) -> int:
        try:
            return max(0, int(round(float(value or 0))))
        except (TypeError, ValueError):
            return 0

    def _resolve_heal_tile_context(
        self,
        position: Optional[Tuple[int, int]],
    ) -> tuple[int, float, str, Optional[int]]:
        if position is None:
            return 0, 0.0, "", None

        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return 0, 0.0, "", None

        settlement_name = self.legions_settlement_source_name_by_tile.get(normalized_position)
        if not settlement_name:
            return 0, 0.0, "", None
        if settlement_name not in self.legions_active_settlement_territory_capture_turn_by_name:
            return 0, 0.0, "", None

        settlement_level = self.legions_settlement_level_by_name.get(str(settlement_name))
        if settlement_level is None:
            return 0, 0.0, "", None
        normalized_level = max(0, int(settlement_level))
        return (
            int(SETTLEMENT_ARMOR_BONUS_BY_LEVEL.get(normalized_level, 0)),
            float(self.SETTLEMENT_REST_HEAL_BONUS_BY_LEVEL.get(normalized_level, 0.0)),
            str(settlement_name),
            normalized_level,
        )

    def _resolve_rest_regeneration_context(
        self,
        position: Optional[Tuple[int, int]],
    ) -> tuple[float, str, Optional[object]]:
        """Возвращает процентный бонус лечения от отдыха на клетке города/столицы."""
        if position is None:
            return 0.0, "", None

        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return 0.0, "", None

        if normalized_position == tuple(self.CASTLE_POS):
            return float(CAPITAL_HEAL_TILE_ARMOR_BONUS) / 100.0, "capital", "capital"

        settlement_name = self.legions_settlement_source_name_by_tile.get(normalized_position)
        if not settlement_name:
            return 0.0, "", None
        if settlement_name not in self.legions_active_settlement_territory_capture_turn_by_name:
            return 0.0, "", None

        settlement_level = self.legions_settlement_level_by_name.get(str(settlement_name))
        if settlement_level is None:
            return 0.0, "", None
        normalized_level = max(0, int(settlement_level))
        bonus_percent = float(self.SETTLEMENT_REST_HEAL_BONUS_BY_LEVEL.get(normalized_level, 0.0))
        return bonus_percent / 100.0, str(settlement_name), normalized_level

    def _resolve_rest_heal_percent(
        self,
        position: Optional[Tuple[int, int]],
    ) -> float:
        """Возвращает базовый процент лечения от отдыха на текущей клетке."""
        if self._is_player_controlled_territory_tile(position):
            return float(self.PLAYER_TERRITORY_REST_HEAL_PERCENT)
        return float(self.DEFAULT_REST_HEAL_PERCENT)

    def _resolve_typeoflord_rest_heal_bonus_percent(self) -> float:
        """Возвращает дополнительный бонус лечения от отдыха по типу лорда."""
        if int(self.typeoflord) == 1:
            return float(self.TYPEOFLORD_ONE_REST_HEAL_BONUS_PERCENT)
        return 0.0

    def _resolve_enemy_regeneration_bonus_percent(
        self,
        position: Optional[Tuple[int, int]],
    ) -> float:
        """Возвращает бонус регенерации врага на клетке города/столицы."""
        if position is None:
            return 0.0

        try:
            normalized_position = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, IndexError):
            return 0.0

        if normalized_position == tuple(self.empire_territory_source_tile):
            return float(CAPITAL_HEAL_TILE_ARMOR_BONUS) / 100.0

        settlement_name = self.legions_settlement_source_name_by_tile.get(normalized_position)
        if not settlement_name:
            return 0.0

        settlement_level = self.legions_settlement_level_by_name.get(str(settlement_name))
        if settlement_level is None:
            return 0.0

        normalized_level = max(0, int(settlement_level))
        return float(self.SETTLEMENT_REST_HEAL_BONUS_BY_LEVEL.get(normalized_level, 0.0)) / 100.0

    @classmethod
    def _settlement_upgrade_cost_for_level(cls, current_level: int) -> float:
        try:
            normalized_level = int(current_level)
        except (TypeError, ValueError):
            normalized_level = 0
        return float(cls.SETTLEMENT_UPGRADE_GOLD_COST_BY_LEVEL.get(normalized_level, 0.0))

    def _can_upgrade_settlement(self, settlement_name: str) -> bool:
        if str(settlement_name) not in self.legions_active_settlement_territory_capture_turn_by_name:
            return False
        current_level = int(self.legions_settlement_level_by_name.get(str(settlement_name), 0) or 0)
        if current_level >= int(self.MAX_SETTLEMENT_LEVEL):
            return False
        upgrade_cost = self._settlement_upgrade_cost_for_level(current_level)
        return upgrade_cost > 0.0 and float(self.gold or 0.0) >= float(upgrade_cost)

    def _resolve_settlement_defender_armor_bonus(
        self,
        enemy_id: Optional[int],
    ) -> tuple[int, str]:
        if enemy_id is None:
            return 0, ""

        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return 0, ""

        bonus = 0
        source = ""

        settlement_level = SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID.get(normalized_enemy_id)
        if settlement_level is not None:
            bonus = int(SETTLEMENT_ARMOR_BONUS_BY_LEVEL.get(int(settlement_level), 0))
            source = f"settlement_level_{int(settlement_level)}"

        enemy_pos = self.grid_env.enemy_positions.get(normalized_enemy_id)
        if enemy_pos is not None and tuple(enemy_pos) == tuple(self.CASTLE_POS):
            capital_bonus = int(CAPITAL_HEAL_TILE_ARMOR_BONUS)
            if capital_bonus >= bonus:
                bonus = capital_bonus
                source = "capital"

        return max(0, int(bonus)), source

    def _apply_settlement_defender_armor_bonus(
        self,
        enemy_id: Optional[int],
        red_team: List[Dict],
    ) -> None:
        bonus, source = self._resolve_settlement_defender_armor_bonus(enemy_id)
        if bonus <= 0:
            return

        affected_units = 0
        for unit in red_team:
            if self._is_empty_enemy_unit(unit):
                continue
            base_armor = self._normalize_armor_value(unit.get("armor", 0))
            unit["armor"] = base_armor + bonus
            unit["settlement_armor_bonus"] = int(bonus)
            if source.startswith("settlement_level_"):
                try:
                    unit["settlement_level"] = int(source.rsplit("_", 1)[-1])
                except (TypeError, ValueError):
                    pass
            elif source == "capital":
                unit["settlement_level"] = "capital"
            affected_units += 1

        if affected_units <= 0:
            return

        if source == "capital":
            self._log(
                f"Бонус столицы: RED-отряд {enemy_id} получает +{bonus} брони на каждого юнита "
                f"({affected_units} юнитов)."
            )
        else:
            self._log(
                f"Бонус города: RED-отряд {enemy_id} получает +{bonus} брони на каждого юнита "
                f"({affected_units} юнитов, {source})."
            )

    def _apply_hero_heal_tile_armor_bonus(self, blue_team: List[Dict]) -> None:
        bonus, _rest_bonus, source, settlement_level = self._resolve_heal_tile_context(
            self.battle_origin_pos
        )
        if bonus <= 0:
            return

        affected_units = 0
        for unit in blue_team:
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0:
                continue

            base_armor = self._normalize_armor_value(unit.get("armor", 0))
            unit["campaign_heal_tile_base_armor"] = int(base_armor)
            unit["campaign_heal_tile_armor_bonus"] = int(bonus)
            unit["armor"] = base_armor + bonus
            unit["settlement_armor_bonus"] = int(bonus)
            if settlement_level is not None:
                unit["settlement_level"] = int(settlement_level)
            affected_units += 1

        if affected_units <= 0:
            return

        if settlement_level is not None:
            self._log(
                f"Бонус города {source} уровня {settlement_level}: BLUE-отряд героя получает "
                f"+{bonus} брони на каждого юнита ({affected_units} юнитов)."
            )

    def _apply_active_blue_potion_effects(self, blue_team: List[Dict]) -> None:
        if not blue_team:
            return

        buffed_invulnerability: List[int] = []
        buffed_strength: List[int] = []
        buffed_energy: List[int] = []
        buffed_haste: List[int] = []
        buffed_fire: List[int] = []
        buffed_earth: List[int] = []
        buffed_water: List[int] = []
        buffed_air: List[int] = []

        for unit in blue_team:
            position = int(unit.get("position", -1) or -1)
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if position <= 0 or max_hp <= 0 or hp <= 0:
                continue

            if position in self.active_invulnerability_potion_positions:
                base_armor = self._normalize_armor_value(unit.get("armor", 0))
                unit["campaign_potion_base_armor"] = int(base_armor)
                unit["armor"] = int(base_armor) + int(self.INVULNERABILITY_POTION_ARMOR_BONUS)
                unit["campaign_invulnerability_potion_bonus"] = int(
                    self.INVULNERABILITY_POTION_ARMOR_BONUS
                )
                buffed_invulnerability.append(position)

            damage_multiplier = 1.0
            if position in self.active_strength_potion_positions:
                damage_multiplier *= float(self.STRENGTH_POTION_DAMAGE_MULTIPLIER)
            if position in self.active_energy_elixir_positions:
                damage_multiplier *= float(self.ENERGY_ELIXIR_DAMAGE_MULTIPLIER)
            if damage_multiplier > 1.0:
                base_damage = self._normalize_damage_value(unit.get("damage", 0))
                base_damage_secondary = self._normalize_damage_value(unit.get("damage_secondary", 0))
                unit["campaign_potion_base_damage"] = int(base_damage)
                unit["campaign_potion_base_damage_secondary"] = int(base_damage_secondary)
                unit["damage"] = self._normalize_damage_value(
                    float(base_damage) * float(damage_multiplier)
                )
                unit["damage_secondary"] = self._normalize_damage_value(
                    float(base_damage_secondary) * float(damage_multiplier)
                )
                unit["campaign_damage_potion_multiplier"] = float(damage_multiplier)
                if position in self.active_strength_potion_positions:
                    unit["campaign_strength_potion_multiplier"] = float(
                        self.STRENGTH_POTION_DAMAGE_MULTIPLIER
                    )
                    buffed_strength.append(position)
                if position in self.active_energy_elixir_positions:
                    unit["campaign_energy_elixir_multiplier"] = float(
                        self.ENERGY_ELIXIR_DAMAGE_MULTIPLIER
                    )
                    buffed_energy.append(position)

            initiative_multiplier = 1.0
            if position in self.active_haste_elixir_positions:
                initiative_multiplier = max(
                    initiative_multiplier,
                    float(self.HASTE_ELIXIR_INITIATIVE_MULTIPLIER),
                )
            if initiative_multiplier > 1.0:
                base_initiative = int(unit.get("initiative_base", unit.get("initiative", 0)) or 0)
                unit["campaign_potion_base_initiative"] = int(base_initiative)
                boosted_initiative = self._normalize_damage_value(
                    float(base_initiative) * initiative_multiplier
                )
                unit["initiative_base"] = int(boosted_initiative)
                unit["initiative"] = int(boosted_initiative)
                unit["campaign_initiative_potion_multiplier"] = float(initiative_multiplier)
                buffed_haste.append(position)

            added_resistance: List[str] = []
            resistance = [str(res) for res in (unit.get("resistance") or [])]
            for active_positions, element, bucket in (
                (self.active_fire_ward_positions, "Fire", buffed_fire),
                (self.active_earth_ward_positions, "Earth", buffed_earth),
                (self.active_water_ward_positions, "Water", buffed_water),
                (self.active_air_ward_positions, "Air", buffed_air),
            ):
                if position not in active_positions or element in resistance:
                    continue
                resistance.append(element)
                added_resistance.append(element)
                bucket.append(position)
            if added_resistance:
                unit["resistance"] = resistance
                unit["campaign_potion_added_resistance"] = added_resistance

        if buffed_invulnerability:
            self._log(
                "Активно зелье неуязвимости для позиций "
                f"{sorted(buffed_invulnerability)} перед началом боя"
            )
        if buffed_strength:
            self._log(
                "Активно зелье силы для позиций "
                f"{sorted(buffed_strength)} перед началом боя"
            )
        if buffed_energy:
            self._log(
                "Активен эликсир энергии для позиций "
                f"{sorted(buffed_energy)} перед началом боя"
            )
        if buffed_haste:
            self._log(
                "Активен эликсир быстроты для позиций "
                f"{sorted(buffed_haste)} перед началом боя"
            )
        if buffed_fire:
            self._log(
                "Активна защита от магии Огня для позиций "
                f"{sorted(buffed_fire)} перед началом боя"
            )
        if buffed_earth:
            self._log(
                "Активна защита от магии Земли для позиций "
                f"{sorted(buffed_earth)} перед началом боя"
            )
        if buffed_water:
            self._log(
                "Активна защита от магии Воды для позиций "
                f"{sorted(buffed_water)} перед началом боя"
            )
        if buffed_air:
            self._log(
                "Активна защита от магии Воздуха для позиций "
                f"{sorted(buffed_air)} перед началом боя"
            )

    def _apply_active_blue_support_spell_effects(self, blue_team: List[Dict]) -> None:
        if not blue_team:
            return

        summary = self._blue_stack_spell_effect_summary()
        active_spell_ids = list(summary.get("active_spell_ids", []))
        if not active_spell_ids:
            return

        buffed_armor: List[int] = []
        buffed_damage: List[int] = []
        buffed_initiative: List[int] = []
        buffed_accuracy: List[int] = []
        buffed_health: List[int] = []
        buffed_resistance: List[int] = []
        resistance_types = [
            str(value)
            for value in (summary.get("resistance_types", []) or [])
            if str(value)
        ]

        for unit in blue_team:
            position = int(unit.get("position", -1) or -1)
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0.0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
            if position <= 0 or max_hp <= 0.0 or hp <= 0.0:
                continue

            armor_delta = int(summary.get("armor_delta", 0) or 0)
            if armor_delta != 0:
                base_armor = self._normalize_armor_value(unit.get("armor", 0))
                unit["campaign_map_spell_base_armor"] = int(base_armor)
                unit["armor"] = self._normalize_armor_value(float(base_armor) + float(armor_delta))
                buffed_armor.append(position)

            damage_multiplier = float(summary.get("damage_multiplier", 1.0) or 1.0)
            if abs(damage_multiplier - 1.0) > 1e-9:
                base_damage = self._normalize_damage_value(unit.get("damage", 0))
                base_damage_secondary = self._normalize_damage_value(unit.get("damage_secondary", 0))
                unit["campaign_map_spell_base_damage"] = int(base_damage)
                unit["campaign_map_spell_base_damage_secondary"] = int(base_damage_secondary)
                unit["damage"] = self._normalize_damage_value(
                    float(base_damage) * float(damage_multiplier)
                )
                unit["damage_secondary"] = self._normalize_damage_value(
                    float(base_damage_secondary) * float(damage_multiplier)
                )
                buffed_damage.append(position)

            initiative_multiplier = float(summary.get("initiative_multiplier", 1.0) or 1.0)
            if abs(initiative_multiplier - 1.0) > 1e-9:
                base_initiative = int(unit.get("initiative_base", unit.get("initiative", 0)) or 0)
                unit["campaign_map_spell_base_initiative"] = int(base_initiative)
                boosted_initiative = self._normalize_damage_value(
                    float(base_initiative) * float(initiative_multiplier)
                )
                unit["initiative_base"] = int(boosted_initiative)
                unit["initiative"] = int(boosted_initiative)
                buffed_initiative.append(position)

            accuracy_multiplier = float(summary.get("accuracy_multiplier", 1.0) or 1.0)
            if abs(accuracy_multiplier - 1.0) > 1e-9:
                base_accuracy = self._normalize_accuracy_value(unit.get("accuracy", 0))
                base_accuracy_secondary = self._normalize_accuracy_value(
                    unit.get("accuracy_secondary", 0)
                )
                unit["campaign_map_spell_base_accuracy"] = int(base_accuracy)
                unit["campaign_map_spell_base_accuracy_secondary"] = int(base_accuracy_secondary)
                unit["accuracy"] = self._normalize_accuracy_value(
                    float(base_accuracy) * float(accuracy_multiplier)
                )
                unit["accuracy_secondary"] = self._normalize_accuracy_value(
                    float(base_accuracy_secondary) * float(accuracy_multiplier)
                )
                buffed_accuracy.append(position)

            health_delta = int(summary.get("health_delta", 0) or 0)
            if health_delta != 0:
                base_max_health = float(
                    unit.get("max_health", 0)
                    or unit.get("maxhp", 0)
                    or unit.get("health", 0)
                    or unit.get("hp", 0)
                    or 0.0
                )
                base_health = float(unit.get("health", 0) or unit.get("hp", 0) or 0.0)
                unit["campaign_map_spell_base_max_health"] = float(base_max_health)
                unit["campaign_map_spell_health_bonus"] = int(health_delta)
                boosted_max_health = max(0.0, base_max_health + float(health_delta))
                boosted_health = max(0.0, base_health + float(health_delta))
                boosted_health = min(boosted_max_health, boosted_health)
                unit["max_health"] = float(boosted_max_health)
                unit["maxhp"] = float(boosted_max_health)
                unit["health"] = float(boosted_health)
                unit["hp"] = float(boosted_health)
                buffed_health.append(position)

            if resistance_types:
                resistance = [str(res) for res in (unit.get("resistance") or [])]
                added_resistance = [res for res in resistance_types if res not in resistance]
                if added_resistance:
                    unit["resistance"] = resistance + added_resistance
                    unit["campaign_map_spell_added_resistance"] = list(added_resistance)
                    buffed_resistance.append(position)

        if buffed_armor:
            self._log(
                f"Активен бафф брони от заклинаний героя для позиций {sorted(buffed_armor)} "
                f"перед началом боя (+{int(summary.get('armor_delta', 0) or 0)})."
            )
        if buffed_damage:
            self._log(
                "Активен бафф урона от заклинаний героя для позиций "
                f"{sorted(buffed_damage)} перед началом боя "
                f"(x{float(summary.get('damage_multiplier', 1.0) or 1.0):.3g})."
            )
        if buffed_initiative:
            self._log(
                "Активен бафф инициативы от заклинаний героя для позиций "
                f"{sorted(buffed_initiative)} перед началом боя "
                f"(x{float(summary.get('initiative_multiplier', 1.0) or 1.0):.3g})."
            )
        if buffed_accuracy:
            self._log(
                "Активен бафф точности от заклинаний героя для позиций "
                f"{sorted(buffed_accuracy)} перед началом боя "
                f"(x{float(summary.get('accuracy_multiplier', 1.0) or 1.0):.3g})."
            )
        if buffed_health:
            self._log(
                "Активен бафф здоровья от заклинаний героя для позиций "
                f"{sorted(buffed_health)} перед началом боя "
                f"(+{int(summary.get('health_delta', 0) or 0)} HP)."
            )
        if buffed_resistance:
            self._log(
                "Активна защита от магии заклинаниями героя для позиций "
                f"{sorted(buffed_resistance)} перед началом боя: {', '.join(resistance_types)}."
            )

    def _clear_equipped_artifact_effects(self, blue_team: Optional[List[Dict]]) -> None:
        if not blue_team:
            return

        for unit in blue_team:
            if "campaign_artifact_base_damage" in unit:
                unit["damage"] = self._normalize_damage_value(
                    unit.get(
                        "campaign_artifact_base_damage",
                        unit.get("damage", 0),
                    )
                )
            if "campaign_artifact_base_initiative" in unit:
                base_initiative = int(
                    self._normalize_damage_value(
                        unit.get(
                            "campaign_artifact_base_initiative",
                            unit.get("initiative_base", unit.get("initiative", 0)),
                        )
                    )
                )
                unit["initiative_base"] = int(base_initiative)
                unit["initiative"] = int(base_initiative)
            if "campaign_artifact_base_armor" in unit:
                base_armor = self._normalize_armor_value(
                    unit.get(
                        "campaign_artifact_base_armor",
                        unit.get("armor", 0),
                    )
                )
                unit["armor"] = int(base_armor)
                unit["base_armor"] = int(base_armor)
            unit.pop("campaign_artifact_base_damage", None)
            unit.pop("campaign_artifact_base_damage_secondary", None)
            unit.pop("campaign_artifact_base_initiative", None)
            unit.pop("campaign_artifact_base_armor", None)
            unit.pop("campaign_artifact_damage_multiplier", None)
            unit.pop("campaign_artifact_initiative_multiplier", None)
            unit.pop("campaign_artifact_armor_bonus", None)
            unit.pop("campaign_active_artifacts", None)

    def _refresh_campaign_artifact_effects(self, *, log: bool = False) -> None:
        if self.blue_team_state is None:
            return
        self._apply_equipped_artifact_effects(self.blue_team_state, log=log)

    def _apply_equipped_artifact_effects(
        self,
        blue_team: List[Dict],
        *,
        log: bool = True,
    ) -> None:
        self._clear_equipped_artifact_effects(blue_team)
        if not blue_team:
            return

        self._sync_equipped_artifact_items()
        active_artifacts = [
            str(item_name)
            for item_name in list(self.equipped_artifact_items or [])
            if self._is_artifact_item_name(str(item_name or ""))
        ]
        if not active_artifacts:
            return

        hero = self._resolve_travel_hero(units=blue_team)
        if hero is None:
            return

        damage_multiplier = 1.0
        initiative_multiplier = 1.0
        armor_bonus = 0
        for item_name in active_artifacts:
            effect = self._artifact_effect_definition(item_name)
            damage_multiplier *= float(effect.get("damage_multiplier", 1.0) or 1.0)
            initiative_multiplier *= float(effect.get("initiative_multiplier", 1.0) or 1.0)
            armor_bonus += int(effect.get("armor_bonus", 0) or 0)

        hero["campaign_active_artifacts"] = list(active_artifacts)
        if abs(damage_multiplier - 1.0) > 1e-9:
            base_damage = self._normalize_damage_value(hero.get("damage", 0))
            hero["campaign_artifact_base_damage"] = int(base_damage)
            hero["campaign_artifact_damage_multiplier"] = float(damage_multiplier)
            hero["damage"] = self._normalize_damage_value(
                float(base_damage) * float(damage_multiplier)
            )

        if abs(initiative_multiplier - 1.0) > 1e-9:
            base_initiative = int(
                hero.get("initiative_base", hero.get("initiative", 0)) or 0
            )
            hero["campaign_artifact_base_initiative"] = int(base_initiative)
            hero["campaign_artifact_initiative_multiplier"] = float(initiative_multiplier)
            boosted_initiative = self._normalize_damage_value(
                float(base_initiative) * float(initiative_multiplier)
            )
            hero["initiative_base"] = int(boosted_initiative)
            hero["initiative"] = int(boosted_initiative)

        if armor_bonus != 0:
            base_armor = self._normalize_armor_value(hero.get("armor", 0))
            hero["campaign_artifact_base_armor"] = int(base_armor)
            hero["campaign_artifact_armor_bonus"] = int(armor_bonus)
            hero["armor"] = self._normalize_armor_value(int(base_armor) + int(armor_bonus))
            hero["base_armor"] = int(hero.get("armor", 0) or 0)

        if log:
            hero_name = str(hero.get("name", "Герой") or "Герой")
            self._log(
                f"Активны экипированные артефакты героя {hero_name}: {', '.join(active_artifacts)} "
                f"(урон x{damage_multiplier:.3g}, инициатива x{initiative_multiplier:.3g}, "
                f"броня +{int(armor_bonus)})."
            )

    def _init_battle(
        self,
        enemy_id: int,
        *,
        blue_team: Optional[List[Dict]] = None,
    ):
        """Инициализирует бой с конкретной RED командой."""
        red_team_state = self._get_enemy_team_state(enemy_id)
        if red_team_state:
            red_team = self._build_battle_team_with_placeholders("red", red_team_state)
        else:
            red_team = self._build_battle_team_with_placeholders(
                "red",
                ENEMY_CONFIGS.get(enemy_id, ENEMY_CONFIGS[1]),
            )
        self._apply_settlement_defender_armor_bonus(enemy_id, red_team)

        # Восстанавливаем состояние BLUE или берём дефолтное.
        if blue_team is not None:
            prepared_blue_team = self._build_battle_team_with_placeholders("blue", blue_team)
            equipped_hero_items: List[Optional[str]] = [None] * int(self.BATTLE_EQUIP_SLOTS)
            hero_item_effects: Dict[str, Dict[str, object]] = {}
            self._log("Используется кастомная BLUE команда")
        elif self.persist_blue_hp and self.blue_team_state is not None:
            prepared_blue_team = self._build_battle_team_with_placeholders(
                "blue",
                self.blue_team_state,
            )
            self._clear_equipped_artifact_effects(prepared_blue_team)
            self._apply_hero_heal_tile_armor_bonus(prepared_blue_team)
            self._apply_active_blue_potion_effects(prepared_blue_team)
            self._apply_active_blue_support_spell_effects(prepared_blue_team)
            self._apply_equipped_artifact_effects(prepared_blue_team)
            self._sync_equipped_hero_items()
            equipped_hero_items = list(self.equipped_hero_items)
            hero_item_effects = dict(self._battle_item_effect_definitions())
            self._log("Загружено сохранённое состояние BLUE команды")
        else:
            prepared_blue_team = self._build_battle_team_with_placeholders("blue", UNITS_BLUE)
            self._clear_equipped_artifact_effects(prepared_blue_team)
            self._apply_hero_heal_tile_armor_bonus(prepared_blue_team)
            self._apply_active_blue_potion_effects(prepared_blue_team)
            self._apply_active_blue_support_spell_effects(prepared_blue_team)
            self._apply_equipped_artifact_effects(prepared_blue_team)
            self._sync_equipped_hero_items()
            equipped_hero_items = list(self.equipped_hero_items)
            hero_item_effects = dict(self._battle_item_effect_definitions())
            self._log("Используется дефолтная BLUE команда")

        self.battle_env = BattleEnv(
            reward_win=self.battle_reward_win,
            reward_loss=self.battle_reward_loss,
            reward_step=self.battle_reward_step,
            log_enabled=self.log_enabled,
        )
        self.battle_env.equipped_hero_items = list(equipped_hero_items)
        self.battle_env.hero_item_effects = dict(hero_item_effects)

        # Инициализируем бой с кастомными командами
        self.battle_env._init_with_custom_teams(red_team, prepared_blue_team)

    def _save_blue_state(self):
        """
        Сохраняет состояние BLUE команды после победы.
        HP НЕ восстанавливается, погибшие НЕ воскрешаются.
        Только сбрасываются временные эффекты (armor boost, paralysis, poison и т.д.).
        """
        if self.battle_env is None:
            return

        # Сохраняем только базовые позиции BLUE (без призванных юнитов)
        base_positions = {u["position"] for u in UNITS_BLUE}

        self.blue_team_state = []

        for u in self.battle_env.combined:
            if u.get("team") != "blue":
                continue
            
            pos = u["position"]
            if pos not in base_positions:
                # Юнит не относится к базовым позициям (призванный?) — пропускаем
                continue

            # Сохраняем юнита в текущей форме (после улучшений)
            restored_unit = deepcopy(u)
            restore_transformed = getattr(
                self.battle_env, "_restore_transformed_unit", None
            )
            if callable(restore_transformed):
                restore_transformed(restored_unit)
            
            # СОХРАНЯЕМ ТЕКУЩЕЕ HP (не восстанавливаем!)
            current_hp = max(
                0.0,
                float(restored_unit.get("health", 0) or restored_unit.get("hp", 0)),
            )
            restored_unit["health"] = current_hp
            restored_unit["hp"] = current_hp

            # Сбрасываем все временные боевые эффекты
            restored_unit["defense"] = 0
            restored_unit["waited"] = 0
            restored_unit["paralyzed"] = 0
            restored_unit["long_paralyzed"] = 0
            restored_unit["running_away"] = 0
            restored_unit["transformed"] = 0
            restored_unit["poison_turns_left"] = 0
            restored_unit["poison_damage_per_tick"] = 0
            restored_unit["burn_turns_left"] = 0
            restored_unit["burn_damage_per_tick"] = 0
            restored_unit["powerup"] = 0
            restored_unit["bonusturn"] = 0
            restored_unit.pop("resilience_used_types", None)
            position = int(restored_unit.get("position", -1) or -1)
            if "campaign_artifact_base_damage" in restored_unit:
                restored_unit["damage"] = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_artifact_base_damage",
                        restored_unit.get("damage", 0),
                    )
                )
            if "campaign_artifact_base_initiative" in restored_unit:
                restored_unit["initiative_base"] = int(
                    self._normalize_damage_value(
                        restored_unit.get(
                            "campaign_artifact_base_initiative",
                            restored_unit.get(
                                "initiative_base",
                                restored_unit.get("initiative", 0),
                            ),
                        )
                    )
                )
                restored_unit["initiative"] = int(restored_unit.get("initiative_base", 0) or 0)
            if "campaign_artifact_base_armor" in restored_unit:
                base_armor = self._normalize_armor_value(
                    restored_unit.get(
                        "campaign_artifact_base_armor",
                        restored_unit.get("armor", 0),
                    )
                )
                restored_unit["armor"] = int(base_armor)
                restored_unit["base_armor"] = int(base_armor)
            if "campaign_map_spell_base_damage" in restored_unit:
                restored_unit["damage"] = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_map_spell_base_damage",
                        restored_unit.get("damage", 0),
                    )
                )
            if "campaign_map_spell_base_damage_secondary" in restored_unit:
                restored_unit["damage_secondary"] = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_map_spell_base_damage_secondary",
                        restored_unit.get("damage_secondary", 0),
                    )
                )
            if "campaign_map_spell_base_initiative" in restored_unit:
                restored_unit["initiative_base"] = int(
                    self._normalize_damage_value(
                        restored_unit.get(
                            "campaign_map_spell_base_initiative",
                            restored_unit.get("initiative_base", restored_unit.get("initiative", 0)),
                        )
                    )
                )
                restored_unit["initiative"] = int(restored_unit.get("initiative_base", 0) or 0)
            if "campaign_map_spell_base_accuracy" in restored_unit:
                restored_unit["accuracy"] = self._normalize_accuracy_value(
                    restored_unit.get(
                        "campaign_map_spell_base_accuracy",
                        restored_unit.get("accuracy", 0),
                    )
                )
            if "campaign_map_spell_base_accuracy_secondary" in restored_unit:
                restored_unit["accuracy_secondary"] = self._normalize_accuracy_value(
                    restored_unit.get(
                        "campaign_map_spell_base_accuracy_secondary",
                        restored_unit.get("accuracy_secondary", 0),
                    )
                )
            if "campaign_map_spell_base_max_health" in restored_unit:
                base_max_health = float(
                    restored_unit.get(
                        "campaign_map_spell_base_max_health",
                        restored_unit.get("max_health", restored_unit.get("maxhp", 0)),
                    )
                    or 0.0
                )
                health_bonus = float(
                    restored_unit.get("campaign_map_spell_health_bonus", 0) or 0.0
                )
                current_hp = max(0.0, float(restored_unit.get("health", 0) or restored_unit.get("hp", 0) or 0.0))
                restored_health = max(0.0, min(base_max_health, current_hp - health_bonus))
                restored_unit["max_health"] = float(base_max_health)
                restored_unit["maxhp"] = float(base_max_health)
                restored_unit["health"] = float(restored_health)
                restored_unit["hp"] = float(restored_health)
            added_spell_resistance = [
                str(res)
                for res in (restored_unit.get("campaign_map_spell_added_resistance") or [])
                if str(res)
            ]
            if added_spell_resistance:
                restored_unit["resistance"] = [
                    res
                    for res in (restored_unit.get("resistance") or [])
                    if str(res) not in set(added_spell_resistance)
                ]
            if "campaign_map_spell_base_armor" in restored_unit:
                restored_unit["armor"] = self._normalize_armor_value(
                    restored_unit.get(
                        "campaign_map_spell_base_armor",
                        restored_unit.get("armor", 0),
                    )
                )
                restored_unit["base_armor"] = int(restored_unit["armor"])
            restored_unit.pop("campaign_map_spell_base_damage", None)
            restored_unit.pop("campaign_map_spell_base_damage_secondary", None)
            restored_unit.pop("campaign_map_spell_base_initiative", None)
            restored_unit.pop("campaign_map_spell_base_accuracy", None)
            restored_unit.pop("campaign_map_spell_base_accuracy_secondary", None)
            restored_unit.pop("campaign_map_spell_base_max_health", None)
            restored_unit.pop("campaign_map_spell_health_bonus", None)
            restored_unit.pop("campaign_map_spell_added_resistance", None)
            restored_unit.pop("campaign_map_spell_base_armor", None)
            restored_unit.pop("campaign_artifact_base_damage", None)
            restored_unit.pop("campaign_artifact_base_damage_secondary", None)
            restored_unit.pop("campaign_artifact_base_initiative", None)
            restored_unit.pop("campaign_artifact_base_armor", None)
            restored_unit.pop("campaign_artifact_damage_multiplier", None)
            restored_unit.pop("campaign_artifact_initiative_multiplier", None)
            restored_unit.pop("campaign_artifact_armor_bonus", None)
            restored_unit.pop("campaign_active_artifacts", None)
            if (
                position in self.active_strength_potion_positions
                or position in self.active_energy_elixir_positions
            ):
                base_damage = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_potion_base_damage",
                        restored_unit.get("damage", 0),
                    )
                )
                base_damage_secondary = self._normalize_damage_value(
                    restored_unit.get(
                        "campaign_potion_base_damage_secondary",
                        restored_unit.get("damage_secondary", 0),
                    )
                )
                restored_unit["damage"] = int(base_damage)
                restored_unit["damage_secondary"] = int(base_damage_secondary)
            restored_unit.pop("original_damage", None)
            restored_unit.pop("campaign_potion_base_damage", None)
            restored_unit.pop("campaign_potion_base_damage_secondary", None)
            restored_unit.pop("campaign_damage_potion_multiplier", None)
            restored_unit.pop("campaign_strength_potion_multiplier", None)
            restored_unit.pop("campaign_energy_elixir_multiplier", None)
            if position in self.active_haste_elixir_positions:
                base_initiative = int(
                    restored_unit.get(
                        "campaign_potion_base_initiative",
                        restored_unit.get("initiative_base", 0),
                    )
                    or 0
                )
                restored_unit["initiative_base"] = int(base_initiative)
            restored_unit.pop("campaign_potion_base_initiative", None)
            restored_unit.pop("campaign_initiative_potion_multiplier", None)
            added_resistance = [
                str(res)
                for res in (restored_unit.get("campaign_potion_added_resistance") or [])
                if str(res)
            ]
            if added_resistance:
                restored_unit["resistance"] = [
                    res
                    for res in (restored_unit.get("resistance") or [])
                    if str(res) not in set(added_resistance)
                ]
            restored_unit.pop("campaign_potion_added_resistance", None)
            if position in self.active_invulnerability_potion_positions:
                base_armor = self._normalize_armor_value(
                    restored_unit.get(
                        "campaign_potion_base_armor",
                        restored_unit.get("armor", 0),
                    )
                )
            else:
                # Armor must be restored to the unit's battle-start value.
                base_armor = int(restored_unit.get("base_armor", restored_unit.get("armor", 0)) or 0)
            if "campaign_heal_tile_base_armor" in restored_unit:
                base_armor = self._normalize_armor_value(
                    restored_unit.get("campaign_heal_tile_base_armor", base_armor)
                )
            restored_unit["armor"] = max(0, base_armor)
            restored_unit.pop("campaign_potion_base_armor", None)
            restored_unit.pop("campaign_invulnerability_potion_bonus", None)
            restored_unit.pop("campaign_heal_tile_base_armor", None)
            restored_unit.pop("campaign_heal_tile_armor_bonus", None)
            restored_unit.pop("settlement_armor_bonus", None)
            restored_unit.pop("settlement_level", None)
            
            # Восстанавливаем initiative к базовому значению
            restored_unit["initiative"] = restored_unit.get("initiative_base", 0)
            restored_unit["needaunit"] = self._resolve_hero_needaunit(restored_unit)
            
            self.blue_team_state.append(restored_unit)

        self._sync_hero_progression_flags(self.blue_team_state)
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state, grant_delta=True)
        self._refresh_campaign_artifact_effects(log=False)
        self._log("Состояние BLUE команды сохранено (HP не восстановлено)")

    def _find_unit_data_by_name(self, name: str) -> Optional[Dict]:
        for entry in UNIT_DATA:
            # Prefer canonical Russian key, keep mojibake fallback for legacy data dumps.
            entry_name = entry.get("\u043a\u0442\u043e")
            if entry_name is None:
                entry_name = entry.get("\u0420\u0454\u0421\u201a\u0420\u0455")
            if entry_name == name:
                return entry
        return None

    def _clear_current_battle_context(self) -> None:
        self.current_battle_context = {}

    def _build_battle_team_with_placeholders(
        self,
        team: str,
        units: Optional[List[Dict]],
    ) -> List[Dict]:
        normalized_team = "red" if str(team).lower() == "red" else "blue"
        positions = range(1, 7) if normalized_team == "red" else range(7, 13)
        units_by_position: Dict[int, Dict] = {}

        for raw_unit in units or []:
            if not isinstance(raw_unit, dict):
                continue
            try:
                position = int(raw_unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                continue
            if position not in positions or position in units_by_position:
                continue
            normalized_unit = deepcopy(raw_unit)
            normalized_unit["team"] = normalized_team
            normalized_unit["position"] = int(position)
            units_by_position[position] = normalized_unit

        return [
            units_by_position.get(int(position), placeholder_unit(normalized_team, int(position)))
            for position in positions
        ]

    def _build_summoned_blue_team(
        self,
        summon_unit_name: str,
    ) -> Optional[Tuple[List[Dict], Dict]]:
        unit_data = self._find_unit_data_by_name(str(summon_unit_name or "").strip())
        if not isinstance(unit_data, dict):
            return None

        template_unit = self._build_unit_from_data(unit_data, team="blue", position=7)
        summon_position = 7 if template_unit.get("stand") == "ahead" else 10
        template_unit["team"] = "blue"
        template_unit["position"] = int(summon_position)
        template_unit["stand"] = "ahead" if summon_position in (7, 8, 9) else "behind"
        template_unit["hero"] = False
        template_unit["needaunit"] = 0
        template_unit["exp_current"] = 0
        template_unit["exp_required"] = 0
        template_unit["next_level_exp"] = 0

        blue_team = self._build_battle_team_with_placeholders("blue", [template_unit])
        summoned_unit = next(
            unit for unit in blue_team if int(unit.get("position", -1) or -1) == int(summon_position)
        )
        return blue_team, summoned_unit

    def _save_enemy_state_from_battle(self, enemy_id: Optional[int]) -> None:
        if self.battle_env is None:
            return

        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return

        original_team = self._build_battle_team_with_placeholders(
            "red",
            self._get_enemy_team_state(normalized_enemy_id),
        )
        original_by_position = {
            int(unit.get("position", -1) or -1): deepcopy(unit)
            for unit in original_team
        }
        battle_by_position = {
            int(unit.get("position", -1) or -1): deepcopy(unit)
            for unit in self.battle_env.combined
            if unit.get("team") == "red"
            and not bool(unit.get("Summoned"))
            and 1 <= int(unit.get("position", -1) or -1) <= 6
        }
        restore_transformed = getattr(
            self.battle_env, "_restore_transformed_unit", None
        )
        if callable(restore_transformed):
            for battle_unit in battle_by_position.values():
                restore_transformed(battle_unit)

        saved_team: List[Dict] = []
        for position in range(1, 7):
            original_unit = deepcopy(original_by_position.get(position, placeholder_unit("red", position)))
            battle_unit = battle_by_position.get(position)
            if battle_unit is not None and not self._is_empty_enemy_unit(original_unit):
                current_hp = max(
                    0.0,
                    float(battle_unit.get("health", 0) or battle_unit.get("hp", 0) or 0.0),
                )
                original_unit["health"] = float(current_hp)
                original_unit["hp"] = float(current_hp)
                original_unit["Level"] = int(
                    battle_unit.get("Level", original_unit.get("Level", 0)) or 0
                )
                original_unit["exp_current"] = float(
                    battle_unit.get("exp_current", original_unit.get("exp_current", 0)) or 0.0
                )
                original_unit["exp_required"] = battle_unit.get(
                    "exp_required",
                    original_unit.get("exp_required", 0),
                )
                original_unit["next_level_exp"] = int(
                    battle_unit.get(
                        "next_level_exp",
                        original_unit.get("next_level_exp", 0),
                    )
                    or 0
                )
                original_unit["hero"] = bool(
                    battle_unit.get("hero", original_unit.get("hero", False))
                )
                original_unit["needaunit"] = int(
                    battle_unit.get("needaunit", original_unit.get("needaunit", 0)) or 0
                )
                original_unit["initiative"] = (
                    int(original_unit.get("initiative_base", original_unit.get("initiative", 0)) or 0)
                    if current_hp > 0.0
                    else 0
                )

            original_unit["defense"] = 0
            original_unit["waited"] = 0
            original_unit["paralyzed"] = 0
            original_unit["long_paralyzed"] = 0
            original_unit["running_away"] = 0
            original_unit["transformed"] = 0
            original_unit["poison_turns_left"] = 0
            original_unit["poison_damage_per_tick"] = 0
            original_unit["burn_turns_left"] = 0
            original_unit["burn_damage_per_tick"] = 0
            original_unit["burn_source_lord_pos"] = None
            original_unit["uran_turns_left"] = 0
            original_unit["uran_damage_per_tick"] = 0
            original_unit["resilience_used_types"] = []
            original_unit["powerup"] = 0
            original_unit["bonusturn"] = 0
            original_unit.pop("teamated", None)
            original_unit.pop("hermited", None)
            original_unit.pop("settlement_armor_bonus", None)
            original_unit.pop("settlement_level", None)
            saved_team.append(original_unit)

        self.enemy_team_states[normalized_enemy_id] = self._build_battle_team_with_placeholders(
            "red",
            saved_team,
        )
        if normalized_enemy_id in self.enemy_map_spell_effects:
            self._recompute_enemy_stack_spell_effects(normalized_enemy_id)

    def _get_buildings_for_capital(self, capital: Optional[int]) -> Dict:
        """Возвращает словарь построек по значению 'столица'."""
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            capital_value = None

        if capital_value == 2:
            return legions_buildings_d2
        if capital_value == 3:
            return mountain_clans_buildings_d2
        if capital_value == 4:
            return undead_hordes_buildings_d2
        if capital_value == 5:
            return elves_buildings_d2
        return empire_buildings_d2

    def _get_spells_for_capital(self, capital: Optional[int]) -> Dict:
        """Возвращает словарь заклинаний по значению 'столица'."""
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            capital_value = None

        if capital_value == 2:
            return legions_spells_d2
        if capital_value == 3:
            return mountain_clans_spells_d2
        if capital_value == 4:
            return undead_hordes_spells_d2
        if capital_value == 5:
            return elves_spells_d2
        return empire_spells_d2

    def _get_hire_options_for_capital(self, capital: Optional[int]) -> List[Dict[str, object]]:
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            capital_value = None

        if capital_value == 2:
            hire_data = HIRE_D2["legions"]
        elif capital_value == 3:
            hire_data = HIRE_D2["mountain_clans"]
        elif capital_value == 4:
            hire_data = HIRE_D2["undead_hordes"]
        elif capital_value == 5:
            hire_data = HIRE_D2["elves"]
        else:
            hire_data = HIRE_D2["empire"]

        if not isinstance(hire_data, dict):
            return []
        return [dict(entry) for entry in hire_data.values() if isinstance(entry, dict)]

    def _get_mercenary_hire_options(self) -> List[Dict[str, object]]:
        options: List[Dict[str, object]] = []
        for site_name, site_data in self.BASE_MERCENARY_SITE_DATA.items():
            site_id = str(site_data.get("site_id", "") or "")
            for idx, entry in enumerate(tuple(site_data.get("roster", ()))):
                if not isinstance(entry, dict):
                    continue
                unit_name = str(entry.get("unit_name", entry.get("name", "")) or "").strip()
                if not unit_name:
                    continue
                slot = int(entry.get("slot", idx) or idx)
                try:
                    gold_cost = max(0.0, float(entry.get("gold", 0.0) or 0.0))
                except (TypeError, ValueError):
                    gold_cost = 0.0
                unit_data = self._find_unit_data_by_name(unit_name)
                stand = self._mercenary_stand_for_unit_data(unit_data)
                options.append(
                    {
                        "site_name": str(site_name),
                        "site_id": site_id,
                        "slot": int(slot),
                        "unit_id": str(entry.get("unit_id", "") or ""),
                        "unit_name": unit_name,
                        "name": unit_name,
                        "gold": float(gold_cost),
                        "stand": "" if stand is None else stand,
                        "unit_type": (
                            str(unit_data.get("тип", unit_data.get("unit_type", "")) or "").strip()
                            if isinstance(unit_data, dict)
                            else ""
                        ),
                    }
                )
        return options

    def _refresh_dynamic_action_layout(self) -> None:
        self.active_hire_options = self._get_hire_options_for_capital(self.Realcapital)
        self.active_mercenary_hire_options = self._get_mercenary_hire_options()
        self.GRID_MERCENARY_HIRE_ACTION_START = (
            self.GRID_HIRE_ACTION_START + len(self.active_hire_options)
        )
        self.GRID_TRAINER_ACTION_START = (
            self.GRID_MERCENARY_HIRE_ACTION_START + len(self.active_mercenary_hire_options)
        )
        self.GRID_MERCHANT_BUY_ACTION_START = (
            self.GRID_TRAINER_ACTION_START + len(self.TRAINER_POSITIONS)
        )
        self.GRID_SPELL_SHOP_BUY_ACTION_START = self.GRID_MERCHANT_BUY_ACTION_START + len(
            self.MERCHANT_BUY_ITEMS
        )
        self.GRID_ENERGY_ACTION_START = self.GRID_SPELL_SHOP_BUY_ACTION_START + len(
            self.SPELL_SHOP_BUY_SPELLS
        )
        self.GRID_HASTE_ACTION_START = self.GRID_ENERGY_ACTION_START + len(
            self.ENERGY_ELIXIR_POSITIONS
        )
        self.GRID_FIRE_WARD_ACTION_START = self.GRID_HASTE_ACTION_START + len(
            self.HASTE_ELIXIR_POSITIONS
        )
        self.GRID_EARTH_WARD_ACTION_START = self.GRID_FIRE_WARD_ACTION_START + len(
            self.FIRE_WARD_POSITIONS
        )
        self.GRID_WATER_WARD_ACTION_START = self.GRID_EARTH_WARD_ACTION_START + len(
            self.EARTH_WARD_POSITIONS
        )
        self.GRID_AIR_WARD_ACTION_START = self.GRID_WATER_WARD_ACTION_START + len(
            self.WATER_WARD_POSITIONS
        )
        self.GRID_TITAN_ELIXIR_ACTION_START = self.GRID_AIR_WARD_ACTION_START + len(
            self.AIR_WARD_POSITIONS
        )
        self.GRID_SUPREME_ELIXIR_ACTION_START = self.GRID_TITAN_ELIXIR_ACTION_START + len(
            self.TITAN_ELIXIR_POSITIONS
        )
        self.GRID_BUILD_ACTION_START = self.GRID_SUPREME_ELIXIR_ACTION_START + len(
            self.SUPREME_ELIXIR_POSITIONS
        )
        self.grid_spell_action_start = self.GRID_BUILD_ACTION_START + len(self.building_keys)
        self.grid_settlement_upgrade_action_start = self.grid_spell_action_start + len(
            self.spell_keys
        )
        self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START = (
            self.grid_settlement_upgrade_action_start + len(self.settlement_upgrade_names)
        )
        self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START = (
            self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + len(self.BATTLE_EQUIPPABLE_ITEM_NAMES)
        )
        self.grid_legion_damage_spell_action_start = (
            self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START + len(self.BATTLE_EQUIPPABLE_ITEM_NAMES)
        )
        self.grid_map_support_spell_action_start = (
            self.grid_legion_damage_spell_action_start + self._map_offensive_spell_action_max_count()
        )
        self.grid_spell_shop_cast_action_start = (
            self.grid_map_support_spell_action_start + self._map_support_spell_action_max_count()
        )
        self.GRID_UNLOCK_SCROLL_MAGIC_ACTION = (
            self.grid_spell_shop_cast_action_start + int(self.MAX_SPELL_SHOP_CAST_ACTIONS)
        )
        self.grid_scroll_cast_action_start = self.GRID_UNLOCK_SCROLL_MAGIC_ACTION + 1

    def _build_unit_from_data(self, entry: Dict, team: str, position: int) -> Dict:
        def _entry_get(*keys, default=0):
            for key in keys:
                if key in entry:
                    return entry.get(key)
            return default

        new_unit = map_unit_to_battle(entry, team, position)
        new_unit["exp_kill"] = _entry_get(
            "\u043e\u043f\u044b\u0442 \u0443\u0431\u0438\u0439\u0441\u0442\u0432\u0430",
            "\u0420\u0455\u0420\u0457\u0421\u2039\u0421\u201a \u0421\u0453\u0420\xb1\u0420\u0451\u0420\u2116\u0421\u0403\u0421\u201a\u0420\u0406\u0420\xb0",
            "exp_kill",
            default=0,
        )
        new_unit["exp_required"] = _entry_get(
            "\u043d\u0443\u0436\u043d\u044b\u0439 \u043e\u043f\u044b\u0442",
            "\u0420\u0405\u0421\u0453\u0420\xb6\u0420\u0405\u0421\u2039\u0420\u2116 \u0420\u0455\u0420\u0457\u0421\u2039\u0421\u201a",
            "exp_required",
            default=0,
        )
        new_unit["exp_current"] = _entry_get(
            "\u0442\u0435\u043a\u0443\u0449\u0438\u0439 \u043e\u043f\u044b\u0442",
            "\u0421\u201a\u0420\xb5\u0420\u0454\u0421\u0453\u0421\u2030\u0420\u0451\u0420\u2116 \u0420\u0455\u0420\u0457\u0421\u2039\u0421\u201a",
            "exp_current",
            default=0,
        )
        try:
            new_unit["next_level_exp"] = int(
                round(
                    float(
                        _entry_get(
                            "\u0441\u043b\u0435\u0434\u0443\u0440\u043e\u0432\u0435\u043d\u044c",
                            "next_level_exp",
                            default=0,
                        )
                    )
                )
            )
        except (TypeError, ValueError):
            new_unit["next_level_exp"] = 0
        turns_raw = _entry_get(
            "\u043f\u0440\u0435\u0432\u0440\u0430\u0449\u0430\u0435\u0442\u0441\u044f \u0432",
            "\u0420\u0457\u0421\u0402\u0420\xb5\u0420\u0406\u0421\u0402\u0420\xb0\u0421\u2030\u0420\xb0\u0420\xb5\u0421\u201a\u0421\u0403\u0421\u040f \u0420\u0406",
            "turns_into",
            default=[],
        )
        if isinstance(turns_raw, list):
            turns_into = list(turns_raw)
        elif turns_raw:
            turns_into = [turns_raw]
        else:
            turns_into = []
        new_unit["turns_into"] = turns_into
        new_unit["hp"] = new_unit.get("health", 0)
        new_unit["maxhp"] = new_unit.get("max_health", 0)
        return new_unit

    def _replace_blue_unit(self, upgraded: Dict) -> None:
        if self.blue_team_state is None:
            return
        pos = upgraded.get("position")
        if pos is None:
            return
        for idx, unit in enumerate(self.blue_team_state):
            if int(unit.get("position", -1)) == int(pos):
                self.blue_team_state[idx] = deepcopy(upgraded)
                self._sync_moves_per_turn_with_hero(units=self.blue_team_state)
                self._refresh_campaign_artifact_effects(log=False)
                return

    def _log_turns_into_levelups(self) -> int:
        """Applies BLUE unit upgrades after level-up checks and returns applied count."""
        if self.battle_env is None:
            return 0
        levelup_names = getattr(self.battle_env, "last_levelups", []) or []
        if not levelup_names:
            return 0

        name_to_units: Dict[str, List[Dict]] = {}
        for unit in getattr(self.battle_env, "combined", []) or []:
            if unit.get("team") != "blue":
                continue
            name = str(unit.get("name", "") or "").strip()
            if not name:
                continue
            name_to_units.setdefault(name, []).append(unit)

        upgraded_count = 0
        for name in levelup_names:
            unit_data = self._find_unit_data_by_name(name)
            capital_value = unit_data.get("\u0441\u0442\u043e\u043b\u0438\u0446\u0430") if unit_data else None
            buildings = self._get_buildings_for_capital(capital_value)

            units = name_to_units.get(name)
            if not units:
                continue
            unit = units.pop(0)
            pos = unit.get("position")
            if pos is None:
                unit_label = name
            else:
                unit_label = f"{name} (pos {pos})"
            turns_into = unit.get("turns_into", [])
            if not isinstance(turns_into, list):
                turns_into = [turns_into]

            for target in turns_into:
                target_name = str(target).strip()
                if not target_name:
                    continue
                building = None
                for entry in buildings.values():
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("unit") == target_name:
                        building = entry
                        break
                if building is None:
                    continue
                built_flag = building.get("Build", building.get("built", 0))
                if int(built_flag or 0) == 1:
                    self._log(f'Unit "{unit_label}" upgraded to "{target_name}"')
                    unit_data = self._find_unit_data_by_name(target_name)
                    if unit_data is not None:
                        upgraded_unit = self._build_unit_from_data(
                            unit_data,
                            unit.get("team", "blue"),
                            unit.get("position", pos),
                        )
                        self._reapply_persistent_elixir_bonuses_to_promoted_unit(
                            source_unit=unit,
                            upgraded_unit=upgraded_unit,
                        )
                        unit.clear()
                        unit.update(deepcopy(upgraded_unit))
                        self._replace_blue_unit(upgraded_unit)
                        upgraded_count += 1
                    break

        return int(upgraded_count)

    def _heal_blue_team(self, heal_percent: float = 0.05, bonus_percent: float = 0.0) -> int:
        """
        Восстанавливает HP раненым юнитам BLUE команды.
        
        Args:
            heal_percent: Доля от max_health для восстановления (0.05 = 5%)
            bonus_percent: Дополнительная доля от max_health для восстановления.
             
        Returns:
            Количество юнитов, которым восстановлено HP
        """
        state = self._get_blue_state()
        
        healed_count = 0
        for unit in state:
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            
            # Юнит жив и ранен
            if hp > 0 and hp < max_hp:
                heal_amount = max_hp * (
                    float(heal_percent or 0.0) + max(0.0, float(bonus_percent or 0.0))
                )
                new_hp = min(max_hp, hp + heal_amount)
                unit["hp"] = new_hp
                unit["health"] = new_hp
                healed_count += 1
                
        return healed_count

    def _heal_unit_at_position(
        self,
        position: int,
        heal_amount: float,
        source_label: str = "Бутыль лечения",
    ) -> Tuple[float, Optional[str]]:
        """
        Лечит конкретную позицию BLUE команды, возвращает фактический объём лечения и имя юнита.
        """
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)

            if max_hp <= 0 or hp <= 0 or hp >= max_hp:
                return 0.0, None

            new_hp = min(max_hp, hp + heal_amount)
            healed = new_hp - hp
            unit["hp"] = new_hp
            unit["health"] = new_hp

            name = unit.get("name", f"pos_{position}")
            self._log(
                f"{source_label}: {name} (pos {position}) HP {hp:.1f} -> {new_hp:.1f} (+{healed:.1f})"
            )
            return healed, name

        return 0.0, None

    def _missing_hp_at_position(self, position: int) -> float:
        """Возвращает недостающее здоровье живого юнита BLUE на позиции или 0.0."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            if max_hp <= 0 or hp <= 0 or hp >= max_hp:
                return 0.0
            return max_hp - hp
        return 0.0

    def _has_any_heal_bottle_left(self) -> bool:
        """True, если доступна хотя бы одна банка лечения любого типа."""
        return (
            self.heal_bottles_used < self._max_heal_bottles_available()
            or self.healing_bottles_used < self._max_healing_bottles_available()
            or self._hero_item_available(self.HEALING_OINTMENT_ITEM_NAME)
        )

    def _select_heal_bottle_for_position(self, position: int) -> Tuple[float, Optional[str]]:
        """
        Выбирает, какой тип лечения применить к юниту:
        - предпочитает наименьшее достаточное лечение среди +50 / +100 / +200;
        - если недостаёт более 200 HP и есть Целебная мазь, использует её;
        - если +50/+100 закончились, но есть Целебная мазь, использует её как фолбэк;
        - если лечить нельзя: возвращает (0.0, None).
        """
        missing_hp = self._missing_hp_at_position(position)
        if missing_hp <= 0.0:
            return 0.0, None

        has_large = self.heal_bottles_used < self._max_heal_bottles_available()
        has_small = self.healing_bottles_used < self._max_healing_bottles_available()
        has_ointment = self._hero_item_available(self.HEALING_OINTMENT_ITEM_NAME)

        if has_small and missing_hp <= self.HEALING_BOTTLE_AMOUNT:
            return self.HEALING_BOTTLE_AMOUNT, "small"
        if has_large and missing_hp <= self.HEAL_BOTTLE_AMOUNT:
            return self.HEAL_BOTTLE_AMOUNT, "large"
        if has_ointment and missing_hp > self.HEALING_OINTMENT_AMOUNT:
            return self.HEALING_OINTMENT_AMOUNT, "ointment"
        if has_ointment and not (has_small or has_large):
            return self.HEALING_OINTMENT_AMOUNT, "ointment"
        if has_ointment and not has_large and missing_hp > self.HEALING_BOTTLE_AMOUNT:
            return self.HEALING_OINTMENT_AMOUNT, "ointment"
        if has_large:
            return self.HEAL_BOTTLE_AMOUNT, "large"
        if has_small:
            return self.HEALING_BOTTLE_AMOUNT, "small"
        if has_ointment:
            return self.HEALING_OINTMENT_AMOUNT, "ointment"
        return 0.0, None

    def _get_blue_state(self) -> List[Dict]:
        """Возвращает (и при необходимости инициализирует) сохранённое состояние BLUE."""
        if self.blue_team_state is None:
            self.blue_team_state = deepcopy(UNITS_BLUE)
            self._sync_hero_progression_flags(self.blue_team_state)
            self._sync_moves_per_turn_with_hero(units=self.blue_team_state, refill=True)
        else:
            self._sync_hero_progression_flags(self.blue_team_state)
        return self.blue_team_state

    def _spend_moves(self, spent_moves: int) -> None:
        """Списывает очки перемещения в grid-режиме."""
        if spent_moves <= 0:
            return
        self.moves = max(0, int(self.moves) - int(spent_moves))

    def _reset_stagnation_tracking(self, start_pos: Tuple[int, int]) -> None:
        """Сбрасывает трекинг повторов и микро-циклов в grid-режиме."""
        pos = (int(start_pos[0]), int(start_pos[1]))
        self.grid_visit_counts = {pos: 1}
        self.recent_positions = [pos]

    def _compute_stagnation_penalty(
        self,
        old_pos: Tuple[int, int],
        new_pos: Tuple[int, int],
    ) -> float:
        """
        Штрафует стагнацию на карте:
        - повторные посещения клетки (с кепом на шаг),
        - движение "в стену" без смены позиции,
        - петля A->B->A.
        Возвращает отрицательное значение (или 0.0).
        """
        target_pos = (int(new_pos[0]), int(new_pos[1]))
        visit_count = int(self.grid_visit_counts.get(target_pos, 0)) + 1
        self.grid_visit_counts[target_pos] = visit_count

        self.recent_positions.append(target_pos)
        if len(self.recent_positions) > 3:
            self.recent_positions = self.recent_positions[-3:]

        penalty = 0.0
        if visit_count > 1 and self.reward_repeat_position_penalty > 0.0:
            repeat_penalty = self.reward_repeat_position_penalty * float(visit_count - 1)
            penalty += min(repeat_penalty, self.reward_repeat_position_penalty_cap)

        if old_pos == new_pos and self.reward_no_movement_penalty > 0.0:
            penalty += self.reward_no_movement_penalty

        if len(self.recent_positions) >= 3 and self.reward_backtrack_penalty > 0.0:
            pos_a, pos_b, pos_c = self.recent_positions[-3:]
            if pos_a == pos_c and pos_a != pos_b:
                penalty += self.reward_backtrack_penalty

        return -float(penalty)

    def _apply_battle_grid_steps(self, extra_steps: int = 10) -> None:
        """Списывает дополнительные очки перемещения из-за входа в бой."""
        if extra_steps <= 0:
            return
        self._spend_moves(extra_steps)

    def _update_turns(self) -> float:
        """Совместимость: ходы больше не рассчитываются по step_count."""
        # Ходы теперь продвигаются только явным завершением хода (REST).
        return 0.0

    def _compute_enemy_defeat_reward(self, enemy_id: Optional[int]) -> float:
        """Фиксированная награда за победу над каждым вражеским отрядом."""
        reward = float(self.reward_defeat_enemy)
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            normalized_enemy_id = -1
        if normalized_enemy_id in self.RUIN_REWARD_BY_ENEMY_ID:
            reward += float(self.reward_ruin_clear_bonus)
        return reward

    @classmethod
    def is_final_objective_enemy_id(cls, enemy_id: Optional[int]) -> bool:
        """True, если enemy_id относится к финальной цели кампании."""
        try:
            objective_enemy_ids = {
                int(objective_enemy_id)
                for enemy_ids in cls.FINAL_OBJECTIVE_CITIES.values()
                for objective_enemy_id in enemy_ids
            }
            return int(enemy_id) in objective_enemy_ids
        except (TypeError, ValueError):
            return False

    def _is_final_objective_enemy(self, enemy_id: Optional[int]) -> bool:
        return self.is_final_objective_enemy_id(enemy_id)

    def _objective_city_for_enemy(self, enemy_id: Optional[int]) -> Optional[str]:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return None
        for city_name, enemy_ids in self.FINAL_OBJECTIVE_CITIES.items():
            if normalized_enemy_id in {int(value) for value in enemy_ids}:
                return city_name
        return None

    def _is_objective_city_cleared(self, city_name: str) -> bool:
        """True, если оба отряда указанного целевого города уже побеждены."""
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}
        city_enemy_ids = self.FINAL_OBJECTIVE_CITIES.get(city_name)
        if not city_enemy_ids:
            return False
        for enemy_id in city_enemy_ids:
            if enemy_id not in enemies_alive:
                return False
            if bool(enemies_alive.get(enemy_id, False)):
                return False
        return True

    def _capture_objective_city_if_cleared(self, enemy_id: Optional[int]) -> List[str]:
        """Помечает город как захваченный, если текущий бой зачистил его полностью."""
        city_name = self._objective_city_for_enemy(enemy_id)
        if city_name is None:
            return []
        if city_name in self.captured_objective_cities:
            return []
        if not self._is_objective_city_cleared(city_name):
            return []
        self.captured_objective_cities.add(city_name)
        return [city_name]

    def _all_objective_cities_captured(self) -> bool:
        return len(self.captured_objective_cities) == len(self.FINAL_OBJECTIVE_CITIES)

    def _is_legions_settlement_territory_cleared(self, settlement_name: str) -> bool:
        source_data = self.legions_settlement_territory_source_by_name.get(str(settlement_name), {})
        required_enemy_ids = tuple(int(enemy_id) for enemy_id in source_data.get("required_enemy_ids", ()))
        if not required_enemy_ids:
            return False
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}
        for enemy_id in required_enemy_ids:
            if enemy_id not in enemies_alive:
                return False
            if bool(enemies_alive.get(enemy_id, False)):
                return False
        return True

    def _activate_legions_settlement_territory_if_cleared(
        self,
        enemy_id: Optional[int],
    ) -> List[str]:
        try:
            normalized_enemy_id = int(enemy_id)
        except (TypeError, ValueError):
            return []
        settlement_name = self.legions_settlement_territory_source_name_by_enemy_id.get(normalized_enemy_id)
        if not settlement_name:
            return []
        if settlement_name in self.legions_active_settlement_territory_capture_turn_by_name:
            return []
        if not self._is_legions_settlement_territory_cleared(settlement_name):
            return []
        self.legions_active_settlement_territory_capture_turn_by_name[str(settlement_name)] = int(self.turns)
        self._refresh_faction_territories()
        return [str(settlement_name)]

    def _compute_final_objective_reward(
        self,
        city_count: int = 1,
        *,
        captured_before: int = 0,
    ) -> float:
        """
        Награда за захват одного или нескольких целевых городов.

        Первый захваченный целевой город даёт уменьшенную награду (x1),
        последующие — полную (x5).
        """
        if city_count <= 0:
            return 0.0
        total_reward = 0.0
        captured_before = max(0, int(captured_before))
        base_reward = float(self.reward_all_enemies)
        full_multiplier = float(self.FINAL_OBJECTIVE_CITY_REWARD_MULTIPLIER)

        for offset in range(int(city_count)):
            capture_index = captured_before + offset
            multiplier = 1.0 if capture_index == 0 else full_multiplier
            total_reward += base_reward * multiplier

        return float(total_reward)

    def _compute_blue_exp_reward(self) -> Tuple[float, float]:
        """Возвращает (reward, raw_exp) за опыт BLUE в завершённом бою."""
        if self.battle_env is None or self.battle_env.winner != "blue":
            return 0.0, 0.0
        raw_exp = max(0.0, float(getattr(self.battle_env, "last_battle_exp", 0.0) or 0.0))
        exp_norm = raw_exp / (raw_exp + self.exp_reward_norm_k)
        reward = self.reward_exp_weight * exp_norm
        return float(reward), float(raw_exp)

    def _compute_blue_survival_reward(self) -> Tuple[float, float, float]:
        """
        Награда за сохранность базовой команды BLUE после боя.
        Возвращает (reward, alive_ratio, hp_ratio).
        """
        if self.battle_env is None:
            return 0.0, 0.0, 0.0

        base_units = [u for u in UNITS_BLUE if int(u.get("position", -1)) >= 0]
        if not base_units:
            return 0.0, 0.0, 0.0

        base_by_pos = {int(u["position"]): u for u in base_units}
        combined_blue = {
            int(u.get("position", -1)): u
            for u in self.battle_env.combined
            if u.get("team") == "blue" and int(u.get("position", -1)) in base_by_pos
        }

        alive_count = 0
        hp_ratio_sum = 0.0
        total = len(base_by_pos)

        for pos, base_unit in base_by_pos.items():
            cur_unit = combined_blue.get(pos)
            hp = float(cur_unit.get("health", 0) or cur_unit.get("hp", 0) or 0) if cur_unit else 0.0
            base_max = float(
                base_unit.get("max_health", 0)
                or base_unit.get("maxhp", 0)
                or base_unit.get("health", 0)
                or base_unit.get("hp", 0)
                or 1.0
            )
            cur_max = (
                float(cur_unit.get("max_health", 0) or cur_unit.get("maxhp", 0) or 0)
                if cur_unit
                else 0.0
            )
            max_hp = max(1.0, base_max, cur_max)
            if hp > 0:
                alive_count += 1
            hp_ratio_sum += float(np.clip(hp / max_hp, 0.0, 1.0))

        alive_ratio = float(alive_count) / float(total)
        hp_ratio = float(hp_ratio_sum) / float(total)
        reward = (
            self.reward_survival_alive_weight * alive_ratio
            + self.reward_survival_hp_weight * hp_ratio
        )
        return float(reward), float(alive_ratio), float(hp_ratio)

    def _legions_gold_income_per_turn(self) -> float:
        return float(self.GOLD_PER_TURN) + float(self.legions_captured_gold_mine_count) * float(
            self.LEGIONS_GOLD_MINE_GOLD_PER_TURN
        )

    def _advance_turns(self, delta_turns: int) -> float:
        """Увеличивает счётчик ходов и золото на заданное количество."""
        if delta_turns <= 0:
            return 0.0
        pending_hire_penalty = self._pending_hire_turn_penalty(
            delta_turns,
            units=self.blue_team_state,
        )
        total_gold_income = 0.0
        for _ in range(int(delta_turns)):
            self.turns += 1
            self._refresh_faction_territories()
            self._clear_all_enemy_map_spell_effects()
            self._clear_all_blue_map_spell_effects()
            total_gold_income += self._legions_gold_income_per_turn()
            self._apply_mana_income_for_turn()
            self._heal_wounded_enemy_teams_for_turn()
        self._clear_expired_combat_potion_effects()
        self.combat_potion_battle_bonus_pending = False
        self.gold += float(total_gold_income)
        self.moves = self.moves_per_turn
        self.active_buildings["alredybuilt"] = 0
        self.spell_learning_locked = False
        self.battle_item_equip_used_this_turn = False
        self.spell_cast_counts_by_id_this_turn = {}
        self.summon_hero_battle_bonus_pending = False
        self.summon_hero_battle_bonus_enemy_ids_this_turn = set()
        return -float(delta_turns) * self.reward_turn_penalty - float(pending_hire_penalty)

    @staticmethod
    def _territory_sort_key(
        origin_tile: Tuple[int, int],
        tile: Tuple[int, int],
    ) -> Tuple[int, int, int, int, int, int]:
        origin_x, origin_y = int(origin_tile[0]), int(origin_tile[1])
        tile_x, tile_y = int(tile[0]), int(tile[1])
        dx = tile_x - origin_x
        dy = tile_y - origin_y
        # В кампании есть диагональный ход, поэтому основная близость — по Chebyshev,
        # а дальше добиваем детерминированной развязкой по Manhattan и координатам.
        return (
            max(abs(dx), abs(dy)),
            abs(dx) + abs(dy),
            abs(dy),
            abs(dx),
            tile_y,
            tile_x,
        )

    def _legions_territory_sort_key(self, tile: Tuple[int, int]) -> Tuple[int, int, int, int, int, int]:
        return self._territory_sort_key(self.legions_territory_source_tile, tile)

    def _empire_territory_sort_key(self, tile: Tuple[int, int]) -> Tuple[int, int, int, int, int, int]:
        return self._territory_sort_key(self.empire_territory_source_tile, tile)

    def _build_territory_forbidden_tiles(
        self,
        *,
        source_tile: Tuple[int, int],
        additional_claimable_tiles: Tuple[Tuple[int, int], ...] = (),
    ) -> Tuple[Tuple[int, int], ...]:
        forbidden_tiles = set(tuple(tile) for tile in self._static_obstacle_tiles)

        base_forbidden_tiles = _load_legions_territory_base_forbidden_tiles()
        if base_forbidden_tiles:
            forbidden_tiles.update(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_forbidden_tiles)
            )

        enemy_tiles = {
            tuple(int(coord) for coord in pos)
            for pos in (self._static_enemy_positions or {}).values()
        }
        forbidden_tiles.difference_update(enemy_tiles)
        claimable_tiles = {
            tuple(int(coord) for coord in tuple(source_tile))
        }
        claimable_tiles.update(
            tuple(int(coord) for coord in tile)
            for tile in tuple(additional_claimable_tiles or ())
        )
        forbidden_tiles.difference_update(claimable_tiles)

        return tuple(sorted(forbidden_tiles))

    def _build_legions_territory_forbidden_tiles(self) -> Tuple[Tuple[int, int], ...]:
        return self._build_territory_forbidden_tiles(
            source_tile=self.legions_territory_source_tile,
            additional_claimable_tiles=(
                self.legions_territory_source_tile,
                self.empire_territory_source_tile,
            ),
        )

    def _build_empire_territory_forbidden_tiles(self) -> Tuple[Tuple[int, int], ...]:
        return self._build_territory_forbidden_tiles(
            source_tile=self.empire_territory_source_tile,
            additional_claimable_tiles=(
                self.legions_territory_source_tile,
                self.empire_territory_source_tile,
            ),
        )

    def _build_territory_path_blocked_tiles(
        self,
        *,
        source_tile: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], ...]:
        blocked_tiles = set(tuple(tile) for tile in self._static_obstacle_tiles)
        base_water_tiles = _load_legions_territory_base_water_tiles()
        if base_water_tiles:
            blocked_tiles.update(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_water_tiles)
            )
        blocked_tiles.discard(tuple(source_tile))
        return tuple(sorted(blocked_tiles))

    def _build_legions_territory_path_blocked_tiles(self) -> Tuple[Tuple[int, int], ...]:
        return self._build_territory_path_blocked_tiles(
            source_tile=self.legions_territory_source_tile,
        )

    def _build_empire_territory_path_blocked_tiles(self) -> Tuple[Tuple[int, int], ...]:
        return self._build_territory_path_blocked_tiles(
            source_tile=self.empire_territory_source_tile,
        )

    def _build_territory_path_distances(
        self,
        *,
        source_tile: Tuple[int, int],
        blocked_tile_set: set[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], int]:
        source = (int(source_tile[0]), int(source_tile[1]))
        if source in blocked_tile_set:
            return {}

        distances: Dict[Tuple[int, int], int] = {source: 0}
        frontier = deque([source])
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

        while frontier:
            x, y = frontier.popleft()
            current_distance = distances[(x, y)]
            for dx, dy in neighbor_offsets:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < int(self.grid_size) and 0 <= ny < int(self.grid_size)):
                    continue
                tile = (nx, ny)
                if tile in blocked_tile_set or tile in distances:
                    continue
                distances[tile] = current_distance + 1
                frontier.append(tile)

        return distances

    def _build_legions_territory_path_distances(self) -> Dict[Tuple[int, int], int]:
        return self._build_territory_path_distances(
            source_tile=self.legions_territory_source_tile,
            blocked_tile_set=self.legions_territory_path_blocked_tile_set,
        )

    def _build_empire_territory_path_distances(self) -> Dict[Tuple[int, int], int]:
        return self._build_territory_path_distances(
            source_tile=self.empire_territory_source_tile,
            blocked_tile_set=self.empire_territory_path_blocked_tile_set,
        )

    def _build_territory_order(
        self,
        *,
        forbidden_tile_set: set[Tuple[int, int]],
        path_distances: Dict[Tuple[int, int], int],
        sort_key,
    ) -> Tuple[Tuple[int, int], ...]:
        allowed_tiles = tuple(
            (x, y)
            for y in range(int(self.grid_size))
            for x in range(int(self.grid_size))
            if (x, y) not in forbidden_tile_set and (x, y) in path_distances
        )
        return tuple(
            sorted(
                allowed_tiles,
                key=lambda tile: (int(path_distances[tile]), *sort_key(tile)),
            )
        )

    def _build_legions_territory_order(self) -> Tuple[Tuple[int, int], ...]:
        return self._build_territory_order(
            forbidden_tile_set=self.legions_territory_forbidden_tile_set,
            path_distances=self.legions_territory_path_distances,
            sort_key=self._legions_territory_sort_key,
        )

    def _build_legions_settlement_territory_orders_by_name(self) -> Dict[str, Tuple[Tuple[int, int], ...]]:
        orders_by_name: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        for source_data in self._static_legions_settlement_territory_sources:
            settlement_name = str(source_data.get("name", "") or "")
            source_tile = tuple(source_data.get("source_tile", (0, 0)))
            path_blocked_tiles = self._build_territory_path_blocked_tiles(source_tile=source_tile)
            path_distances = self._build_territory_path_distances(
                source_tile=source_tile,
                blocked_tile_set=set(path_blocked_tiles),
            )
            orders_by_name[settlement_name] = self._build_territory_order(
                forbidden_tile_set=self.legions_territory_forbidden_tile_set,
                path_distances=path_distances,
                sort_key=lambda tile, origin_tile=source_tile: self._territory_sort_key(origin_tile, tile),
            )
        return orders_by_name

    def _build_empire_territory_order(self) -> Tuple[Tuple[int, int], ...]:
        return self._build_territory_order(
            forbidden_tile_set=self.empire_territory_forbidden_tile_set,
            path_distances=self.empire_territory_path_distances,
            sort_key=self._empire_territory_sort_key,
        )

    @staticmethod
    def _territory_claim_count(total_tiles: int, expansion_per_turn: int, turns: int) -> int:
        return min(
            int(total_tiles),
            1 + max(0, int(turns)) * int(expansion_per_turn),
        )

    @staticmethod
    def _claim_territory_tiles(
        order: Tuple[Tuple[int, int], ...],
        claim_count: int,
    ) -> Tuple[Tuple[int, int], ...]:
        if claim_count <= 0:
            return ()
        return tuple(order[: int(claim_count)])

    @staticmethod
    def _settlement_territory_claim_count(
        total_tiles: int,
        expansion_per_turn: int,
        turns: int,
        activation_turn: Optional[int],
    ) -> int:
        if activation_turn is None:
            return 0
        elapsed_turns = max(0, int(turns) - int(activation_turn))
        return min(
            int(total_tiles),
            1 + elapsed_turns * max(0, int(expansion_per_turn)),
        )

    def _refresh_legions_territory_tiles(self) -> None:
        self._refresh_faction_territories()

    def _refresh_legions_territory_grid_obs_cache(self) -> None:
        obs_size = int(getattr(self, "grid_legions_territory_obs_size", 0) or 0)
        if obs_size <= 0:
            self._legions_territory_grid_obs_cache = np.zeros(0, dtype=np.float32)
            return

        territory_obs = np.zeros(obs_size, dtype=np.float32)
        index_by_pos = getattr(self, "grid_legions_territory_index_by_pos", {})
        for tile_pos in self.legions_territory_tiles:
            tile_index = index_by_pos.get(tuple(tile_pos))
            if tile_index is not None:
                territory_obs[int(tile_index)] = 1.0
        self._legions_territory_grid_obs_cache = territory_obs

    def _refresh_legions_captured_mana_source_state(self) -> None:
        captured_tiles_by_kind: Dict[str, List[Tuple[int, int]]] = {
            str(kind): []
            for kind in self.MANA_KIND_ORDER
        }
        for raw_tile, mana_meta in getattr(self, "mana_sources", {}).items():
            tile = tuple(int(coord) for coord in tuple(raw_tile)[:2])
            if tile not in self.legions_territory_tile_set:
                continue
            mana_kind = str(mana_meta.get("kind", "") or "")
            if mana_kind not in captured_tiles_by_kind:
                continue
            captured_tiles_by_kind[mana_kind].append(tile)

        self.legions_captured_mana_source_tiles_by_kind = {
            str(kind): tuple(sorted(captured_tiles_by_kind[str(kind)]))
            for kind in self.MANA_KIND_ORDER
        }
        self.legions_captured_mana_source_counts_by_kind = {
            str(kind): len(self.legions_captured_mana_source_tiles_by_kind[str(kind)])
            for kind in self.MANA_KIND_ORDER
        }
        self.mana_income_per_turn = self._compute_mana_income_per_turn()

    def _refresh_faction_territories(self) -> None:
        legions_claim_count = self._territory_claim_count(
            len(self.legions_territory_order),
            self.LEGIONS_TERRITORY_EXPANSION_PER_TURN,
            self.turns,
        )
        capital_legions_tiles = self._claim_territory_tiles(
            self.legions_territory_order,
            legions_claim_count,
        )
        combined_legions_tiles: List[Tuple[int, int]] = []
        combined_legions_seen: set[Tuple[int, int]] = set()
        for tile in capital_legions_tiles:
            normalized_tile = (int(tile[0]), int(tile[1]))
            if normalized_tile in combined_legions_seen:
                continue
            combined_legions_seen.add(normalized_tile)
            combined_legions_tiles.append(normalized_tile)

        for source_data in self._static_legions_settlement_territory_sources:
            settlement_name = str(source_data.get("name", "") or "")
            source_order = self.legions_settlement_territory_orders_by_name.get(settlement_name, ())
            capture_turn = self.legions_active_settlement_territory_capture_turn_by_name.get(settlement_name)
            settlement_claim_count = self._settlement_territory_claim_count(
                len(source_order),
                int(source_data.get("expansion_per_turn", 0) or 0),
                self.turns,
                capture_turn,
            )
            settlement_tiles = self._claim_territory_tiles(source_order, settlement_claim_count)
            self.legions_settlement_territory_tiles_by_name[settlement_name] = settlement_tiles
            for tile in settlement_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in combined_legions_seen:
                    continue
                combined_legions_seen.add(normalized_tile)
                combined_legions_tiles.append(normalized_tile)

        self.legions_territory_tiles = tuple(combined_legions_tiles)
        self.legions_territory_tile_set = set(self.legions_territory_tiles)
        self.legions_captured_gold_mine_tiles = tuple(
            tile
            for tile in self.gold_mine_tiles
            if tuple(tile) in self.legions_territory_tile_set
        )
        self.legions_captured_gold_mine_count = len(self.legions_captured_gold_mine_tiles)
        self._refresh_legions_captured_mana_source_state()
        empire_claim_count = self._territory_claim_count(
            len(self.empire_territory_order),
            self.EMPIRE_TERRITORY_EXPANSION_PER_TURN,
            self.turns,
        )
        self.empire_territory_tiles = self._claim_territory_tiles(
            self.empire_territory_order,
            empire_claim_count,
        )
        self.empire_territory_tile_set = set(self.empire_territory_tiles)
        if hasattr(self, "grid_legions_territory_positions"):
            self._refresh_legions_territory_grid_obs_cache()

    def _init_merchant_site_metadata(self) -> None:
        obstacle_tiles = set(self._static_obstacle_tiles)
        self.merchant_site_anchors: Dict[str, Tuple[int, int]] = {}
        self.merchant_site_interaction_tiles: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        all_tiles: List[Tuple[int, int]] = []
        seen_tiles: set[Tuple[int, int]] = set()

        for site_name, site_data in self.BASE_MERCHANT_SITE_DATA.items():
            anchor = scale_static_tiles(
                self.grid_size,
                (tuple(site_data["anchor"]),),
            )[0]
            scaled_tiles = scale_static_tiles(
                self.grid_size,
                tuple(site_data["interaction_tiles"]),
            )

            site_tiles: List[Tuple[int, int]] = []
            site_seen: set[Tuple[int, int]] = set()
            for tile in scaled_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in obstacle_tiles or normalized_tile in site_seen:
                    continue
                site_seen.add(normalized_tile)
                site_tiles.append(normalized_tile)
                if normalized_tile not in seen_tiles:
                    seen_tiles.add(normalized_tile)
                    all_tiles.append(normalized_tile)

            self.merchant_site_anchors[str(site_name)] = (int(anchor[0]), int(anchor[1]))
            self.merchant_site_interaction_tiles[str(site_name)] = tuple(site_tiles)

        self.merchant_interaction_tiles = tuple(all_tiles)

    def _init_spell_shop_site_metadata(self) -> None:
        obstacle_tiles = set(self._static_obstacle_tiles)
        self.spell_shop_site_anchors: Dict[str, Tuple[int, int]] = {}
        self.spell_shop_site_interaction_tiles: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        all_tiles: List[Tuple[int, int]] = []
        seen_tiles: set[Tuple[int, int]] = set()

        for site_name, site_data in self.BASE_SPELL_SHOP_SITE_DATA.items():
            anchor = scale_static_tiles(
                self.grid_size,
                (tuple(site_data["anchor"]),),
            )[0]
            scaled_tiles = scale_static_tiles(
                self.grid_size,
                tuple(site_data["interaction_tiles"]),
            )

            site_tiles: List[Tuple[int, int]] = []
            site_seen: set[Tuple[int, int]] = set()
            for tile in scaled_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in obstacle_tiles or normalized_tile in site_seen:
                    continue
                site_seen.add(normalized_tile)
                site_tiles.append(normalized_tile)
                if normalized_tile not in seen_tiles:
                    seen_tiles.add(normalized_tile)
                    all_tiles.append(normalized_tile)

            self.spell_shop_site_anchors[str(site_name)] = (int(anchor[0]), int(anchor[1]))
            self.spell_shop_site_interaction_tiles[str(site_name)] = tuple(site_tiles)

        self.spell_shop_interaction_tiles = tuple(all_tiles)

    def _init_mercenary_site_metadata(self) -> None:
        obstacle_tiles = set(self._static_obstacle_tiles)
        self.mercenary_site_anchors: Dict[str, Tuple[int, int]] = {}
        self.mercenary_site_interaction_tiles: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        all_tiles: List[Tuple[int, int]] = []
        seen_tiles: set[Tuple[int, int]] = set()

        for site_name, site_data in self.BASE_MERCENARY_SITE_DATA.items():
            anchor = scale_static_tiles(
                self.grid_size,
                (tuple(site_data["anchor"]),),
            )[0]
            scaled_tiles = scale_static_tiles(
                self.grid_size,
                tuple(site_data["interaction_tiles"]),
            )

            site_tiles: List[Tuple[int, int]] = []
            site_seen: set[Tuple[int, int]] = set()
            for tile in scaled_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in obstacle_tiles or normalized_tile in site_seen:
                    continue
                site_seen.add(normalized_tile)
                site_tiles.append(normalized_tile)
                if normalized_tile not in seen_tiles:
                    seen_tiles.add(normalized_tile)
                    all_tiles.append(normalized_tile)

            self.mercenary_site_anchors[str(site_name)] = (int(anchor[0]), int(anchor[1]))
            self.mercenary_site_interaction_tiles[str(site_name)] = tuple(site_tiles)

        self.mercenary_interaction_tiles = tuple(all_tiles)

    def _init_trainer_site_metadata(self) -> None:
        obstacle_tiles = set(self._static_obstacle_tiles)
        self.trainer_site_anchors: Dict[str, Tuple[int, int]] = {}
        self.trainer_site_interaction_tiles: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        all_tiles: List[Tuple[int, int]] = []
        seen_tiles: set[Tuple[int, int]] = set()

        for site_name, site_data in self.BASE_TRAINER_SITE_DATA.items():
            anchor = scale_static_tiles(
                self.grid_size,
                (tuple(site_data["anchor"]),),
            )[0]
            scaled_tiles = scale_static_tiles(
                self.grid_size,
                tuple(site_data["interaction_tiles"]),
            )

            site_tiles: List[Tuple[int, int]] = []
            site_seen: set[Tuple[int, int]] = set()
            for tile in scaled_tiles:
                normalized_tile = (int(tile[0]), int(tile[1]))
                if normalized_tile in obstacle_tiles or normalized_tile in site_seen:
                    continue
                site_seen.add(normalized_tile)
                site_tiles.append(normalized_tile)
                if normalized_tile not in seen_tiles:
                    seen_tiles.add(normalized_tile)
                    all_tiles.append(normalized_tile)

            self.trainer_site_anchors[str(site_name)] = (int(anchor[0]), int(anchor[1]))
            self.trainer_site_interaction_tiles[str(site_name)] = tuple(site_tiles)

        self.trainer_interaction_tiles = tuple(all_tiles)

    def _create_initial_mercenary_site_rosters(self) -> Dict[str, List[Dict[str, object]]]:
        rosters: Dict[str, List[Dict[str, object]]] = {}
        for site_name, site_data in self.BASE_MERCENARY_SITE_DATA.items():
            site_roster: List[Dict[str, object]] = []
            for idx, entry in enumerate(tuple(site_data.get("roster", ()))):
                if not isinstance(entry, dict):
                    continue
                unit_name = str(entry.get("unit_name", entry.get("name", "")) or "").strip()
                if not unit_name:
                    continue
                unit_data = self._find_unit_data_by_name(unit_name)
                stand = self._mercenary_stand_for_unit_data(unit_data)
                try:
                    gold_cost = max(0.0, float(entry.get("gold", 0.0) or 0.0))
                except (TypeError, ValueError):
                    gold_cost = 0.0
                try:
                    stock = max(0, int(entry.get("stock", 0) or 0))
                except (TypeError, ValueError):
                    stock = 0
                site_roster.append(
                    {
                        "slot": int(entry.get("slot", idx) or idx),
                        "unit_id": str(entry.get("unit_id", "") or ""),
                        "unit_name": unit_name,
                        "gold": float(gold_cost),
                        "stock": int(stock),
                        "initial_stock": int(stock),
                        "stand": "" if stand is None else stand,
                        "unit_type": (
                            str(unit_data.get("тип", unit_data.get("unit_type", "")) or "").strip()
                            if isinstance(unit_data, dict)
                            else ""
                        ),
                        "unit_max_health": (
                            float(unit_data.get("здоровье", unit_data.get("max_health", 0)) or 0.0)
                            if isinstance(unit_data, dict)
                            else 0.0
                        ),
                        "unit_damage": (
                            float(unit_data.get("урон", unit_data.get("damage", 0)) or 0.0)
                            if isinstance(unit_data, dict)
                            else 0.0
                        ),
                        "big": bool(self._unit_data_is_big(unit_data)),
                    }
                )
            rosters[str(site_name)] = site_roster
        return rosters

    def _create_initial_merchant_stocks(self) -> Dict[str, Dict[str, int]]:
        stocks: Dict[str, Dict[str, int]] = {
            str(site_name): {} for site_name in self.BASE_MERCHANT_SITE_DATA.keys()
        }
        alara_stock: Dict[str, int] = {}
        for item_data in self.MERCHANT_BUY_ITEMS:
            item_name = str(item_data.get("name", "") or "")
            if not item_name:
                continue
            alara_stock[item_name] = max(0, int(item_data.get("stock", 0) or 0))
        stocks["Лавка Алара"] = alara_stock
        stocks.setdefault("Лавка Тралара", {})
        return stocks

    @classmethod
    def _merchant_item_definition(cls, item_name: str) -> Optional[Dict[str, object]]:
        normalized_name = str(item_name or "")
        for item_data in cls.MERCHANT_BUY_ITEMS:
            if str(item_data.get("name", "") or "") == normalized_name:
                return dict(item_data)
        return None

    def _merchant_item_stock(self, site_name: str, item_name: str) -> int:
        site_stock = self.merchant_stocks.get(str(site_name), {})
        try:
            return max(0, int(site_stock.get(str(item_name), 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _merchant_site_for_item(
        self,
        item_name: str,
        position: Optional[Tuple[int, int]] = None,
    ) -> Optional[str]:
        for site_name in self._merchant_sites_at_position(position):
            if self._merchant_item_stock(site_name, item_name) > 0:
                return site_name
        return None

    def _merchant_stock_snapshot(self, site_name: str) -> Dict[str, int]:
        snapshot: Dict[str, int] = {}
        for item_data in self.MERCHANT_BUY_ITEMS:
            item_name = str(item_data.get("name", "") or "")
            if not item_name:
                continue
            stock_left = self._merchant_item_stock(site_name, item_name)
            if stock_left > 0:
                snapshot[item_name] = stock_left
        return snapshot

    def _merchant_sites_at_position(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        if position is None:
            position = tuple(self.grid_env.agent_pos)
        normalized_pos = (int(position[0]), int(position[1]))
        return [
            site_name
            for site_name, tiles in self.merchant_site_interaction_tiles.items()
            if normalized_pos in tiles
        ]

    def _merchant_context_info(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        site_names = self._merchant_sites_at_position(position)
        merchant_stock = {
            site_name: self._merchant_stock_snapshot(site_name)
            for site_name in site_names
        }
        return {
            "is_at_merchant": bool(site_names),
            "merchant_sites_in_range": list(site_names),
            "merchant_stock": merchant_stock,
        }

    def _create_initial_spell_shop_stocks(self) -> Dict[str, Dict[str, int]]:
        stocks: Dict[str, Dict[str, int]] = {
            str(site_name): {} for site_name in self.BASE_SPELL_SHOP_SITE_DATA.keys()
        }
        for site_name in stocks.keys():
            site_stock: Dict[str, int] = {}
            for spell_data in self.SPELL_SHOP_BUY_SPELLS:
                spell_name = str(spell_data.get("name", "") or "")
                if not spell_name:
                    continue
                site_stock[spell_name] = max(0, int(spell_data.get("stock", 0) or 0))
            stocks[site_name] = site_stock
        return stocks

    @classmethod
    def _spell_shop_spell_definition(cls, spell_name: str) -> Optional[Dict[str, object]]:
        normalized_name = str(spell_name or "")
        for spell_data in cls.SPELL_SHOP_BUY_SPELLS:
            if str(spell_data.get("name", "") or "") == normalized_name:
                return dict(spell_data)
        return None

    def _spell_shop_spell_stock(self, site_name: str, spell_name: str) -> int:
        site_stock = self.spell_shop_stocks.get(str(site_name), {})
        try:
            return max(0, int(site_stock.get(str(spell_name), 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _spell_shop_spell_owned(self, spell_id: str) -> bool:
        normalized_spell_id = str(spell_id or "")
        if not normalized_spell_id:
            return False
        if normalized_spell_id in self.spell_shop_purchased_spell_ids:
            return True
        return self._is_spell_learned(normalized_spell_id)

    def _spell_shop_site_for_spell(
        self,
        spell_name: str,
        position: Optional[Tuple[int, int]] = None,
    ) -> Optional[str]:
        for site_name in self._spell_shop_sites_at_position(position):
            if self._spell_shop_spell_stock(site_name, spell_name) > 0:
                return site_name
        return None

    def _spell_shop_stock_snapshot(self, site_name: str) -> Dict[str, int]:
        snapshot: Dict[str, int] = {}
        for spell_data in self.SPELL_SHOP_BUY_SPELLS:
            spell_name = str(spell_data.get("name", "") or "")
            if not spell_name:
                continue
            stock_left = self._spell_shop_spell_stock(site_name, spell_name)
            if stock_left > 0:
                snapshot[spell_name] = stock_left
        return snapshot

    def _spell_shop_sites_at_position(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        if position is None:
            position = tuple(self.grid_env.agent_pos)
        normalized_pos = (int(position[0]), int(position[1]))
        return [
            site_name
            for site_name, tiles in self.spell_shop_site_interaction_tiles.items()
            if normalized_pos in tiles
        ]

    def _spell_shop_context_info(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        site_names = self._spell_shop_sites_at_position(position)
        spell_shop_stock = {
            site_name: self._spell_shop_stock_snapshot(site_name)
            for site_name in site_names
        }
        purchased_spell_names = [
            str(spell_data.get("name", "") or "")
            for spell_data in self.SPELL_SHOP_BUY_SPELLS
            if str(spell_data.get("spell_id", "") or "") in self.spell_shop_purchased_spell_ids
        ]
        return {
            "is_at_spell_shop": bool(site_names),
            "spell_shop_sites_in_range": list(site_names),
            "spell_shop_stock": spell_shop_stock,
            "spell_shop_purchased_spells": purchased_spell_names,
        }

    def _mercenary_roster_entry(
        self,
        site_name: str,
        slot: int,
    ) -> Optional[Dict[str, object]]:
        for entry in self.mercenary_site_rosters.get(str(site_name), []):
            raw_slot = entry.get("slot", -1)
            try:
                entry_slot = int(-1 if raw_slot is None else raw_slot)
            except (TypeError, ValueError):
                entry_slot = -1
            if entry_slot == int(slot):
                return entry
        return None

    def _mercenary_stock_snapshot(self, site_name: str) -> List[Dict[str, object]]:
        snapshot: List[Dict[str, object]] = []
        for entry in self.mercenary_site_rosters.get(str(site_name), []):
            snapshot.append(
                {
                    "slot": int(entry.get("slot", 0) or 0),
                    "unit_id": str(entry.get("unit_id", "") or ""),
                    "unit_name": str(entry.get("unit_name", "") or ""),
                    "gold": float(entry.get("gold", 0.0) or 0.0),
                    "stock": max(0, int(entry.get("stock", 0) or 0)),
                    "stand": str(entry.get("stand", "") or ""),
                    "unit_type": str(entry.get("unit_type", "") or ""),
                    "big": bool(entry.get("big", False)),
                }
            )
        return snapshot

    def _mercenary_sites_at_position(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        if position is None:
            position = tuple(self.grid_env.agent_pos)
        normalized_pos = (int(position[0]), int(position[1]))
        return [
            site_name
            for site_name, tiles in self.mercenary_site_interaction_tiles.items()
            if normalized_pos in tiles
        ]

    def _mercenary_context_info(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        site_names = self._mercenary_sites_at_position(position)
        mercenary_stock = {
            site_name: self._mercenary_stock_snapshot(site_name)
            for site_name in site_names
        }
        return {
            "is_at_mercenary_camp": bool(site_names),
            "mercenary_sites_in_range": list(site_names),
            "mercenary_stock": mercenary_stock,
        }

    def _trainer_sites_at_position(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        if position is None:
            position = tuple(self.grid_env.agent_pos)
        normalized_pos = (int(position[0]), int(position[1]))
        return [
            site_name
            for site_name, tiles in self.trainer_site_interaction_tiles.items()
            if normalized_pos in tiles
        ]

    def _trainer_context_info(
        self,
        position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        site_names = self._trainer_sites_at_position(position)
        return {
            "is_at_trainer": bool(site_names),
            "trainer_sites_in_range": list(site_names),
        }

    def _trainer_preview_for_position(self, position: int) -> Dict[str, object]:
        normalized_position = int(position)
        preview: Dict[str, object] = {
            "position": normalized_position,
            "site_names": list(self._trainer_sites_at_position(self.grid_env.agent_pos)),
            "unit": None,
            "unit_name": None,
            "trainable": False,
            "missing_unit": True,
            "dead_or_empty": False,
            "already_capped": False,
            "insufficient_gold": False,
            "exp_cap": None,
            "exp_before": 0,
            "xp_missing": 0,
            "gold_per_xp": 0.0,
            "gold_needed_full": 0.0,
            "affordable_xp": 0,
            "gold_spent": 0.0,
            "exp_after": 0,
        }

        for unit in self._get_blue_state():
            if int(unit.get("position", -1) or -1) != normalized_position:
                continue
            preview["unit"] = unit
            preview["unit_name"] = str(unit.get("name", "") or "").strip()
            preview["missing_unit"] = False

            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0.0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
            if self._is_empty_blue_unit(unit) or max_hp <= 0.0 or hp <= 0.0:
                preview["dead_or_empty"] = True
                return preview

            exp_cap = self._trainer_exp_cap(unit)
            preview["exp_cap"] = exp_cap
            if exp_cap is None:
                preview["already_capped"] = True
                return preview

            try:
                exp_before = int(round(float(unit.get("exp_current", 0) or 0)))
            except (TypeError, ValueError):
                exp_before = 0
            exp_before = max(0, exp_before)
            preview["exp_before"] = exp_before
            xp_missing = max(0, int(exp_cap) - exp_before)
            preview["xp_missing"] = xp_missing
            if xp_missing <= 0:
                preview["already_capped"] = True
                preview["exp_after"] = exp_before
                return preview

            gold_per_xp = float(self._trainer_gold_per_xp(unit))
            preview["gold_per_xp"] = gold_per_xp
            preview["gold_needed_full"] = float(xp_missing) * gold_per_xp

            affordable_xp = 0
            if gold_per_xp > 0.0:
                affordable_xp = min(
                    xp_missing,
                    max(0, int(float(self.gold or 0.0) // gold_per_xp)),
                )
            preview["affordable_xp"] = int(affordable_xp)
            preview["gold_spent"] = float(affordable_xp) * gold_per_xp
            preview["exp_after"] = exp_before + int(affordable_xp)
            preview["insufficient_gold"] = affordable_xp <= 0
            preview["trainable"] = bool(affordable_xp > 0)
            return preview

        return preview

    def _can_train_unit_position(self, position: int) -> bool:
        if not self._trainer_sites_at_position(self.grid_env.agent_pos):
            return False
        preview = self._trainer_preview_for_position(position)
        return bool(preview.get("trainable", False))

    @staticmethod
    def _is_empty_enemy_unit(unit: Dict) -> bool:
        """True если слот вражеского отряда пустой."""
        name = (unit.get("name") or "").strip().lower()
        if name == "пусто":
            return True
        hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0)
        max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or 0)
        return hp <= 0 and max_hp <= 0

    def _init_enemy_obs_metadata(self) -> None:
        """Готовит метаданные для grid-наблюдения о врагах."""
        unit_types = set()
        max_units = 0
        for team in ENEMY_CONFIGS.values():
            if not team:
                continue
            max_units = max(max_units, len(team))
            for unit in team:
                if self._is_empty_enemy_unit(unit):
                    continue
                unit_type = unit.get("unit_type")
                if unit_type:
                    unit_types.add(unit_type)

        self.enemy_unit_types = sorted(unit_types)
        self.enemy_unit_type_to_index = {t: i for i, t in enumerate(self.enemy_unit_types)}
        self.grid_enemy_unit_type_feature_size = len(self.enemy_unit_types)
        self.grid_enemy_unit_feature_size = self.grid_enemy_unit_type_feature_size + 1
        self.grid_enemy_unit_slots = max_units
        self.grid_enemy_count = len(self.grid_env.enemy_positions or {})
        self.grid_enemy_obs_size = self.grid_enemy_count * (
            2 + self.grid_enemy_unit_slots * self.grid_enemy_unit_feature_size
        )

    def _init_campaign_grid_obs_metadata(self) -> None:
        """Готовит метаданные для grid-наблюдения о состоянии кампании."""
        self.grid_blue_positions = tuple(int(pos) for pos in self.GRID_BOTTLE_POSITIONS)
        self.grid_blue_features_per_unit = 8
        self.grid_blue_obs_size = len(self.grid_blue_positions) * self.grid_blue_features_per_unit

        self.grid_battle_equipment_item_names = tuple(
            str(item_name) for item_name in self.BATTLE_EQUIPPABLE_ITEM_NAMES
        )
        self.grid_battle_equipment_features_per_slot = (
            len(self.grid_battle_equipment_item_names) + 1
        )
        self.grid_battle_equipment_obs_size = (
            int(self.BATTLE_EQUIP_SLOTS) * self.grid_battle_equipment_features_per_slot
        )
        self.grid_resource_core_obs_size = 11 + self.grid_battle_equipment_obs_size
        self.grid_mana_total_kinds = tuple(str(kind) for kind in self.MANA_KIND_ORDER)
        self.grid_mana_total_obs_size = len(self.grid_mana_total_kinds)
        self.grid_resource_obs_size = (
            self.grid_resource_core_obs_size + self.grid_mana_total_obs_size
        )

        mana_kind_order = {str(kind): index for index, kind in enumerate(self.MANA_KIND_ORDER)}
        self.grid_mana_source_entries = tuple(
            {
                "pos": tuple(int(coord) for coord in tuple(tile_pos)[:2]),
                "kind": str(mana_meta.get("kind", "") or ""),
            }
            for tile_pos, mana_meta in sorted(
                getattr(self, "mana_sources", {}).items(),
                key=lambda item: (
                    mana_kind_order.get(str(item[1].get("kind", "") or ""), len(mana_kind_order)),
                    int(item[0][1]),
                    int(item[0][0]),
                ),
            )
        )
        self.grid_mana_kind_to_norm = {
            str(kind): float(index + 1) / float(max(1, len(self.MANA_KIND_ORDER)))
            for index, kind in enumerate(self.MANA_KIND_ORDER)
        }
        self.grid_mana_source_features_per_entry = 3
        self.grid_mana_source_obs_size = (
            len(self.grid_mana_source_entries) * self.grid_mana_source_features_per_entry
        )
        self.grid_mana_total_half_point_by_kind = {}
        for kind in self.MANA_KIND_ORDER:
            source_count = sum(
                1
                for mana_meta in getattr(self, "mana_sources", {}).values()
                if str(mana_meta.get("kind", "") or "") == str(kind)
            )
            max_income = float(source_count) * float(self.LEGIONS_MANA_SOURCE_MANA_PER_TURN)
            if str(kind) == "infernal":
                max_income += float(self.BASE_INFERNAL_MANA_PER_TURN)
            self.grid_mana_total_half_point_by_kind[str(kind)] = max(
                1.0,
                float(self.turn_norm_k) * max(50.0, max_income),
            )

        self.grid_building_features_per_building = 3
        self.grid_building_obs_size = (
            len(self.building_keys) * self.grid_building_features_per_building
        )
        self.grid_spell_core_obs_size = 3
        self.grid_spell_features_per_spell = 1
        self.grid_legion_spell_used_features_per_spell = 1
        self.grid_map_offensive_spell_action_count = self._map_offensive_spell_action_max_count()
        self.grid_legion_spell_used_obs_size = (
            self.grid_map_offensive_spell_action_count
            * self.grid_legion_spell_used_features_per_spell
        )
        self.grid_nearest_enemy_debuff_flag_names = (
            "any",
            "armor",
            "damage",
            "initiative",
            "accuracy",
        )
        self.grid_nearest_enemy_debuff_obs_size = len(self.grid_nearest_enemy_debuff_flag_names)
        self.grid_scroll_core_obs_size = 3
        self.grid_scroll_slot_count = int(self.MAX_SCROLL_CAST_ACTIONS)
        self.grid_scroll_slot_features_per_entry = 4
        self.grid_scroll_obs_size = (
            self.grid_scroll_core_obs_size
            + self.grid_scroll_slot_count * self.grid_scroll_slot_features_per_entry
        )
        self.grid_spell_obs_size = (
            self.grid_spell_core_obs_size
            + len(self.spell_keys) * self.grid_spell_features_per_spell
            + self.grid_legion_spell_used_obs_size
            + self.grid_nearest_enemy_debuff_obs_size
            + self.grid_scroll_obs_size
        )

        self.grid_chest_positions = tuple(sorted(tuple(pos) for pos in self._static_chests.keys()))
        self.grid_chest_features_per_entry = 3
        self.grid_chest_obs_size = (
            len(self.grid_chest_positions) * self.grid_chest_features_per_entry
        )

        self.grid_heal_tile_positions = tuple(
            sorted(tuple(pos) for pos in self._static_castle_heal_tiles)
        )
        self.grid_heal_tile_features_per_entry = 2
        self.grid_heal_tile_obs_size = (
            len(self.grid_heal_tile_positions) * self.grid_heal_tile_features_per_entry
        )

        self.grid_legions_territory_positions = tuple(
            (x, y)
            for y in range(int(self.grid_size))
            for x in range(int(self.grid_size))
        )
        self.grid_legions_territory_features_per_entry = 1
        self.grid_legions_territory_obs_size = (
            len(self.grid_legions_territory_positions)
            * self.grid_legions_territory_features_per_entry
        )
        self.grid_legions_territory_index_by_pos = {
            tuple(tile_pos): index
            for index, tile_pos in enumerate(self.grid_legions_territory_positions)
        }
        self.grid_empire_territory_positions = self.grid_legions_territory_positions
        self.grid_empire_territory_features_per_entry = 1
        self.grid_empire_territory_obs_size = (
            len(self.grid_empire_territory_positions)
            * self.grid_empire_territory_features_per_entry
        )

        self.grid_ruin_positions = tuple(
            sorted(
                tuple(int(coord) for coord in tuple(reward_data.get("ruin_pos", (0, 0)))[:2])
                for reward_data in self.RUIN_REWARD_BY_ENEMY_ID.values()
            )
        )
        self.grid_ruin_features_per_entry = 2
        self.grid_ruin_obs_size = (
            len(self.grid_ruin_positions) * self.grid_ruin_features_per_entry
        )

        self.grid_objective_city_names = tuple(self.FINAL_OBJECTIVE_CITIES.keys())
        self.grid_objective_city_features_per_entry = 4
        self.grid_objective_city_obs_size = (
            len(self.grid_objective_city_names) * self.grid_objective_city_features_per_entry
        )
        self.grid_merchant_site_names = tuple(
            str(site_name) for site_name in self.BASE_MERCHANT_SITE_DATA.keys()
        )
        self.grid_merchant_site_features_per_entry = 5
        self.grid_merchant_site_obs_size = (
            len(self.grid_merchant_site_names) * self.grid_merchant_site_features_per_entry
        )
        self.grid_merchant_item_names = tuple(
            str(item_data.get("name", "") or "") for item_data in self.MERCHANT_BUY_ITEMS
        )
        self.grid_merchant_item_features_per_entry = 1
        self.grid_current_merchant_stock_obs_size = (
            len(self.grid_merchant_item_names) * self.grid_merchant_item_features_per_entry
        )
        self.grid_spell_shop_site_names = tuple(
            str(site_name) for site_name in self.BASE_SPELL_SHOP_SITE_DATA.keys()
        )
        self.grid_spell_shop_site_features_per_entry = 5
        self.grid_spell_shop_site_obs_size = (
            len(self.grid_spell_shop_site_names) * self.grid_spell_shop_site_features_per_entry
        )
        self.grid_spell_shop_spell_names = tuple(
            str(spell_data.get("name", "") or "") for spell_data in self.SPELL_SHOP_BUY_SPELLS
        )
        self.grid_spell_shop_spell_features_per_entry = 1
        self.grid_current_spell_shop_stock_obs_size = (
            len(self.grid_spell_shop_spell_names) * self.grid_spell_shop_spell_features_per_entry
        )
        self.grid_mercenary_site_names = tuple(
            str(site_name) for site_name in self.BASE_MERCENARY_SITE_DATA.keys()
        )
        self.grid_mercenary_site_features_per_entry = 5
        self.grid_mercenary_site_obs_size = (
            len(self.grid_mercenary_site_names) * self.grid_mercenary_site_features_per_entry
        )
        self.grid_mercenary_slot_count = max(
            (len(roster) for roster in self.mercenary_site_rosters.values()),
            default=0,
        )
        self.grid_mercenary_slot_features_per_entry = 8
        self.grid_mercenary_roster_obs_size = (
            self.grid_mercenary_slot_count * self.grid_mercenary_slot_features_per_entry
        )
        self.grid_trainer_features_per_entry = 5
        self.grid_trainer_obs_size = self.grid_trainer_features_per_entry
        self.grid_trainer_total_affordable_xp_half_point = 20.0
        self.grid_merchant_initial_total_stock_by_site_name: Dict[str, int] = {}
        self.grid_merchant_initial_stock_by_site_item: Dict[Tuple[str, str], int] = {}
        self.grid_spell_shop_initial_total_stock_by_site_name: Dict[str, int] = {}
        self.grid_spell_shop_initial_stock_by_site_spell: Dict[Tuple[str, str], int] = {}
        self.grid_mercenary_initial_total_stock_by_site_name: Dict[str, int] = {}
        self.grid_mercenary_initial_stock_by_site_slot: Dict[Tuple[str, int], int] = {}
        self.grid_max_mercenary_price = 1.0
        self.grid_max_mercenary_unit_health = 1.0
        self.grid_max_mercenary_unit_damage = 1.0
        for site_name in self.grid_merchant_site_names:
            total_site_stock = 0
            initial_site_stock = self._create_initial_merchant_stocks().get(site_name, {})
            for item_name in self.grid_merchant_item_names:
                try:
                    initial_stock = max(
                        0,
                        int(initial_site_stock.get(item_name, 0) or 0),
                    )
                except (TypeError, ValueError):
                    initial_stock = 0
                total_site_stock += int(initial_stock)
                self.grid_merchant_initial_stock_by_site_item[(site_name, item_name)] = int(
                    initial_stock
                )
            self.grid_merchant_initial_total_stock_by_site_name[site_name] = int(total_site_stock)
        for site_name in self.grid_spell_shop_site_names:
            total_site_stock = 0
            initial_site_stock = self._create_initial_spell_shop_stocks().get(site_name, {})
            for spell_name in self.grid_spell_shop_spell_names:
                try:
                    initial_stock = max(
                        0,
                        int(initial_site_stock.get(spell_name, 0) or 0),
                    )
                except (TypeError, ValueError):
                    initial_stock = 0
                total_site_stock += int(initial_stock)
                self.grid_spell_shop_initial_stock_by_site_spell[(site_name, spell_name)] = int(
                    initial_stock
                )
            self.grid_spell_shop_initial_total_stock_by_site_name[site_name] = int(total_site_stock)
        for site_name in self.grid_mercenary_site_names:
            total_site_stock = 0
            for entry in self.mercenary_site_rosters.get(site_name, []):
                slot_raw = entry.get("slot", 0)
                try:
                    slot = int(slot_raw if slot_raw is not None else 0)
                except (TypeError, ValueError):
                    slot = 0
                initial_stock_raw = entry.get("initial_stock", entry.get("stock", 0))
                try:
                    initial_stock = max(0, int(initial_stock_raw or 0))
                except (TypeError, ValueError):
                    initial_stock = 0
                total_site_stock += int(initial_stock)
                self.grid_mercenary_initial_stock_by_site_slot[(site_name, slot)] = int(
                    initial_stock
                )
                try:
                    self.grid_max_mercenary_price = max(
                        float(self.grid_max_mercenary_price),
                        float(entry.get("gold", 0.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    self.grid_max_mercenary_unit_health = max(
                        float(self.grid_max_mercenary_unit_health),
                        float(entry.get("unit_max_health", 0.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    self.grid_max_mercenary_unit_damage = max(
                        float(self.grid_max_mercenary_unit_damage),
                        float(entry.get("unit_damage", 0.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    pass
            self.grid_mercenary_initial_total_stock_by_site_name[site_name] = int(
                total_site_stock
            )

        merchant_small_heal_stock = int(
            sum(
                int(item_data.get("stock", 0) or 0)
                for item_data in self.MERCHANT_BUY_ITEMS
                if str(item_data.get("grant", "") or "") == "small_heal_bonus"
            )
        )
        merchant_large_heal_stock = int(
            sum(
                int(item_data.get("stock", 0) or 0)
                for item_data in self.MERCHANT_BUY_ITEMS
                if str(item_data.get("grant", "") or "") == "large_heal_bonus"
            )
        )
        merchant_revive_stock = int(
            sum(
                int(item_data.get("stock", 0) or 0)
                for item_data in self.MERCHANT_BUY_ITEMS
                if str(item_data.get("grant", "") or "") == "revive_bonus"
            )
        )
        self.grid_max_heal_bottle_capacity = int(self.MAX_HEAL_BOTTLES) + merchant_large_heal_stock
        self.grid_max_healing_bottle_capacity = int(self.MAX_HEALING_BOTTLES) + merchant_small_heal_stock + sum(
            1
            for chest_items in self._static_chests.values()
            for item_name in chest_items
            if str(item_name or "") == str(self.BONUS_SMALL_HEAL_SOURCE_ITEM)
        )
        self.grid_max_revive_bottle_capacity = int(self.MAX_REVIVE_BOTTLES) + merchant_revive_stock + sum(
            1
            for chest_items in self._static_chests.values()
            for item_name in chest_items
            if str(item_name or "") == str(self.BONUS_REVIVE_SOURCE_ITEM)
        )
        self.grid_max_invulnerability_potions = sum(
            1
            for chest_items in self._static_chests.values()
            for item_name in chest_items
            if str(item_name or "") == str(self.INVULNERABILITY_POTION_ITEM_NAME)
        )
        self.grid_max_strength_potions = sum(
            1
            for chest_items in self._static_chests.values()
            for item_name in chest_items
            if str(item_name or "") == str(self.STRENGTH_POTION_ITEM_NAME)
        )
        self.grid_max_build_price = 1.0
        for build_key in self.building_keys:
            building = self.active_buildings.get(build_key)
            if not isinstance(building, dict):
                continue
            try:
                price = float(building.get("gold", 0) or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            self.grid_max_build_price = max(self.grid_max_build_price, price)

        self.grid_campaign_obs_size = (
            self.grid_blue_obs_size
            + self.grid_resource_obs_size
            + self.grid_mana_source_obs_size
            + self.grid_building_obs_size
            + self.grid_spell_obs_size
            + self.grid_chest_obs_size
            + self.grid_heal_tile_obs_size
            + self.grid_legions_territory_obs_size
            + self.grid_ruin_obs_size
            + self.grid_objective_city_obs_size
            + self.grid_merchant_site_obs_size
            + self.grid_current_merchant_stock_obs_size
            + self.grid_spell_shop_site_obs_size
            + self.grid_current_spell_shop_stock_obs_size
            + self.grid_mercenary_site_obs_size
            + self.grid_mercenary_roster_obs_size
            + self.grid_trainer_obs_size
        )

    def _encode_enemy_unit_type(self, unit_type: Optional[str]) -> List[float]:
        if self.grid_enemy_unit_type_feature_size <= 0:
            return []
        vec = [0.0] * self.grid_enemy_unit_type_feature_size
        idx = self.enemy_unit_type_to_index.get(unit_type or "")
        if idx is not None:
            vec[idx] = 1.0
        return vec

    @staticmethod
    def _normalize_unit_health(unit: Dict) -> float:
        hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0)
        max_hp = float(unit.get("max_health", 0) or unit.get("maxhp", 0) or hp)
        if max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, hp / max_hp))

    @staticmethod
    def _normalize_ratio(value: object, max_value: object) -> float:
        try:
            value_f = float(value or 0.0)
            max_f = float(max_value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if max_f <= 0.0:
            return 0.0
        return float(np.clip(value_f / max_f, 0.0, 1.0))

    @staticmethod
    def _normalize_saturating_count(value: object, half_point: float = 1.0) -> float:
        try:
            value_f = max(0.0, float(value or 0.0))
        except (TypeError, ValueError):
            return 0.0
        k = max(1e-6, float(half_point))
        return float(np.clip(value_f / (value_f + k), 0.0, 1.0))

    def _build_enemy_grid_obs(self) -> np.ndarray:
        """Строит признаки врагов на карте для grid-наблюдения."""
        if self.grid_enemy_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        enemy_obs: List[float] = []
        zero_enemy_slot = [0.0] * self.grid_enemy_unit_feature_size

        for enemy_id in sorted(self.grid_env.enemy_positions.keys()):
            ex, ey = self.grid_env.enemy_positions.get(enemy_id, (0, 0))
            enemy_obs.extend([ex / grid_scale, ey / grid_scale])

            is_alive = self.grid_env.enemies_alive.get(enemy_id, False)
            team = self._get_enemy_team_state(enemy_id)
            team_sorted = sorted(team, key=lambda u: int(u.get("position", 0)))

            for idx in range(self.grid_enemy_unit_slots):
                if not is_alive or idx >= len(team_sorted):
                    enemy_obs.extend(zero_enemy_slot)
                    continue

                unit = team_sorted[idx]
                if self._is_empty_enemy_unit(unit):
                    enemy_obs.extend(zero_enemy_slot)
                    continue

                type_one_hot = self._encode_enemy_unit_type(unit.get("unit_type"))
                hp_norm = self._normalize_unit_health(unit)
                enemy_obs.extend([*type_one_hot, hp_norm])

        return np.array(enemy_obs, dtype=np.float32)

    def _build_blue_team_grid_obs(self) -> np.ndarray:
        state_by_pos = {
            int(unit.get("position", -1) or -1): unit for unit in (self._get_blue_state() or [])
        }
        blue_obs: List[float] = []

        for position in self.grid_blue_positions:
            unit = state_by_pos.get(int(position), {})
            hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0.0)
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0)
            alive_v = 1.0 if max_hp > 0.0 and hp > 0.0 else 0.0
            hp_ratio = self._normalize_unit_health(unit) if unit else 0.0
            wounded_v = 1.0 if max_hp > 0.0 and hp > 0.0 and hp < max_hp else 0.0
            revivable_v = 1.0 if max_hp > 0.0 and hp <= 0.0 else 0.0
            invulnerability_v = 1.0 if int(position) in self.active_invulnerability_potion_positions else 0.0
            strength_v = 1.0 if int(position) in self.active_strength_potion_positions else 0.0
            hero_v = 1.0 if unit and self._is_hero_unit(unit) else 0.0
            level_norm = self._normalize_saturating_count(unit.get("Level", 0) if unit else 0.0, half_point=3.0)

            blue_obs.extend(
                [
                    alive_v,
                    hp_ratio,
                    wounded_v,
                    revivable_v,
                    invulnerability_v,
                    strength_v,
                    hero_v,
                    level_norm,
                ]
            )

        return np.array(blue_obs, dtype=np.float32)

    def _build_resource_grid_obs(self) -> np.ndarray:
        inventory = self.get_bottle_inventory_counters()
        build_locked_flag = self.active_buildings.get("alredybuilt", 0)
        try:
            build_locked = 1.0 if int(build_locked_flag or 0) == 1 else 0.0
        except (TypeError, ValueError):
            build_locked = 0.0

        resource_obs = [
            self._normalize_ratio(self.moves, max(1, self.moves_per_turn)),
            self._normalize_saturating_count(self.moves_per_turn, half_point=float(self.MOVES_PER_TURN)),
            1.0 if self._is_at_castle() else 0.0,
            build_locked,
            self._normalize_ratio(
                inventory.get("heal_left", 0),
                max(1, int(self.grid_max_heal_bottle_capacity)),
            ),
            self._normalize_ratio(
                inventory.get("healing_left", 0),
                max(1, int(self.grid_max_healing_bottle_capacity)),
            ),
            self._normalize_ratio(
                inventory.get("revive_left", 0),
                max(1, int(self.grid_max_revive_bottle_capacity)),
            ),
            self._normalize_ratio(
                inventory.get("invulnerability_left", 0),
                max(1, int(self.grid_max_invulnerability_potions)),
            ),
            self._normalize_ratio(
                inventory.get("strength_left", 0),
                max(1, int(self.grid_max_strength_potions)),
            ),
            self._normalize_ratio(len(self.chests), max(1, len(self.grid_chest_positions))),
            self._normalize_ratio(
                len(self.captured_objective_cities),
                max(1, len(self.grid_objective_city_names)),
            ),
        ]
        for slot_index in range(int(self.BATTLE_EQUIP_SLOTS)):
            equipped_name = ""
            if slot_index < len(self.equipped_hero_items):
                equipped_name = str(self.equipped_hero_items[slot_index] or "")
            resource_obs.append(1.0 if not equipped_name else 0.0)
            for item_name in self.grid_battle_equipment_item_names:
                resource_obs.append(1.0 if equipped_name == item_name else 0.0)
        for mana_kind in self.grid_mana_total_kinds:
            mana_attr = str(self.MANA_ATTR_BY_KIND.get(str(mana_kind), "") or "")
            mana_value = float(getattr(self, mana_attr, 0.0) or 0.0) if mana_attr else 0.0
            resource_obs.append(
                self._normalize_saturating_count(
                    mana_value,
                    half_point=float(
                        self.grid_mana_total_half_point_by_kind.get(str(mana_kind), 1.0)
                    ),
                )
            )
        return np.array(resource_obs, dtype=np.float32)

    def _build_mana_source_grid_obs(self) -> np.ndarray:
        if self.grid_mana_source_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        mana_obs: List[float] = []
        for entry in self.grid_mana_source_entries:
            tile_pos = tuple(entry.get("pos", (0, 0)))
            mana_kind = str(entry.get("kind", "") or "")
            x = int(tile_pos[0]) if len(tile_pos) >= 1 else 0
            y = int(tile_pos[1]) if len(tile_pos) >= 2 else 0
            type_norm = float(self.grid_mana_kind_to_norm.get(mana_kind, 0.0))
            mana_obs.extend(
                [
                    x / grid_scale,
                    y / grid_scale,
                    type_norm,
                ]
            )
        return np.array(mana_obs, dtype=np.float32)

    def _build_building_grid_obs(self) -> np.ndarray:
        building_obs: List[float] = []
        max_build_price = self._current_max_building_gold_cost()
        for build_key in self.building_keys:
            building = self.active_buildings.get(build_key)
            if not isinstance(building, dict):
                building_obs.extend([0.0, 0.0, 0.0])
                continue

            built_flag = building.get("Build", building.get("built", 0))
            blocked_flag = building.get("blocked", 0)
            try:
                built_v = 1.0 if int(built_flag or 0) == 1 else 0.0
            except (TypeError, ValueError):
                built_v = 0.0
            try:
                blocked_v = 1.0 if int(blocked_flag or 0) == 1 else 0.0
            except (TypeError, ValueError):
                blocked_v = 0.0
            price_v = self._get_building_gold_cost(building)

            building_obs.extend(
                [
                    built_v,
                    blocked_v,
                    self._normalize_ratio(price_v, max_build_price),
                ]
            )

        return np.array(building_obs, dtype=np.float32)

    def _build_spell_grid_obs(self) -> np.ndarray:
        self._ensure_inventory_cache()
        spell_obs: List[float] = [
            1.0 if self._has_magic_tower_built() else 0.0,
            1.0 if self.spell_learning_locked else 0.0,
            self._normalize_ratio(
                len(self.get_learned_spell_descriptions()),
                max(1, len(self.spell_keys)),
            ),
        ]
        for spell_key in self.spell_keys:
            spell = self.active_spells.get(spell_key)
            if not isinstance(spell, dict):
                spell_obs.append(0.0)
                continue
            try:
                learned_v = 1.0 if int(spell.get("learned", 0) or 0) == 1 else 0.0
            except (TypeError, ValueError):
                learned_v = 0.0
            spell_obs.append(learned_v)

        current_specs = self._current_map_offensive_spell_action_specs()
        for idx in range(self.grid_map_offensive_spell_action_count):
            spell_key = ""
            if idx < len(current_specs):
                spell_key = str(current_specs[idx].get("id", "") or "")
            spell_obs.append(
                1.0 if spell_key and self._spell_cast_limit_reached_this_turn(spell_key) else 0.0
            )

        nearest_enemy = self._get_nearest_enemy_stack_for_mask(spell_targetable_only=True)
        nearest_enemy_summary = self._enemy_stack_spell_effect_summary(
            None if nearest_enemy is None else nearest_enemy.get("enemy_id")
        )
        nearest_enemy_effect_types = set(
            str(effect_type)
            for effect_type in nearest_enemy_summary.get("debuff_types", [])
            if str(effect_type)
        )
        spell_obs.extend(
            [
                1.0 if nearest_enemy_effect_types else 0.0,
                1.0 if "armor" in nearest_enemy_effect_types else 0.0,
                1.0 if "damage" in nearest_enemy_effect_types else 0.0,
                1.0 if "initiative" in nearest_enemy_effect_types else 0.0,
                1.0 if "accuracy" in nearest_enemy_effect_types else 0.0,
            ]
        )
        kind_index = {
            "damage": 1,
            "debuff": 2,
            "summon_battle": 3,
            "moves": 4,
            "heal": 5,
            "buff": 6,
            "ward": 7,
            "health_bonus": 8,
        }
        scroll_entries = self._scroll_cast_slot_entries_cache
        available_scroll_entry_count = len(self._available_scroll_spell_entries_cache)
        total_scroll_count = int(self._total_scroll_count_cache)
        spell_obs.extend(
            [
                1.0 if self.scroll_magic_unlocked else 0.0,
                self._normalize_saturating_count(
                    available_scroll_entry_count,
                    half_point=float(max(1, self.MAX_SCROLL_CAST_ACTIONS)),
                ),
                self._normalize_saturating_count(
                    total_scroll_count,
                    half_point=float(max(1, self.MAX_SCROLL_CAST_ACTIONS)),
                ),
            ]
        )
        for idx in range(int(self.MAX_SCROLL_CAST_ACTIONS)):
            scroll_entry: Dict[str, object] = {}
            if idx < len(scroll_entries):
                scroll_entry = dict(scroll_entries[idx])
            spell_kind = str(scroll_entry.get("spell_kind", "") or "")
            try:
                spell_level = int(scroll_entry.get("spell_level", 0) or 0)
            except (TypeError, ValueError):
                spell_level = 0
            spell_obs.extend(
                [
                    1.0 if scroll_entry else 0.0,
                    self._normalize_saturating_count(
                        int(scroll_entry.get("copies", 0) or 0),
                        half_point=2.0,
                    ),
                    self._normalize_ratio(kind_index.get(spell_kind, 0), max(1, len(kind_index))),
                    self._normalize_ratio(spell_level, 5.0),
                ]
            )
        return np.array(spell_obs, dtype=np.float32)

    def _build_chest_grid_obs(self) -> np.ndarray:
        if self.grid_chest_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        active_chests = {tuple(pos) for pos in self.chests.keys()}
        chest_obs: List[float] = []
        for chest_pos in self.grid_chest_positions:
            chest_obs.extend(
                [
                    chest_pos[0] / grid_scale,
                    chest_pos[1] / grid_scale,
                    1.0 if tuple(chest_pos) in active_chests else 0.0,
                ]
            )
        return np.array(chest_obs, dtype=np.float32)

    def _build_heal_tiles_grid_obs(self) -> np.ndarray:
        if self.grid_heal_tile_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        heal_obs: List[float] = []
        for tile_pos in self.grid_heal_tile_positions:
            heal_obs.extend([tile_pos[0] / grid_scale, tile_pos[1] / grid_scale])
        return np.array(heal_obs, dtype=np.float32)

    def _build_legions_territory_grid_obs(self) -> np.ndarray:
        if self.grid_legions_territory_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)
        return self._legions_territory_grid_obs_cache

    def _build_empire_territory_grid_obs(self) -> np.ndarray:
        if self.grid_empire_territory_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        territory_tiles = self.empire_territory_tile_set
        territory_obs = np.fromiter(
            (
                1.0 if tuple(tile_pos) in territory_tiles else 0.0
                for tile_pos in self.grid_empire_territory_positions
            ),
            dtype=np.float32,
            count=self.grid_empire_territory_obs_size,
        )
        return territory_obs

    def _build_ruin_grid_obs(self) -> np.ndarray:
        if self.grid_ruin_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        ruin_obs: List[float] = []
        for ruin_pos in self.grid_ruin_positions:
            ruin_obs.extend([ruin_pos[0] / grid_scale, ruin_pos[1] / grid_scale])
        return np.array(ruin_obs, dtype=np.float32)

    def _build_objective_city_grid_obs(self) -> np.ndarray:
        if self.grid_objective_city_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        objective_obs: List[float] = []
        enemy_positions = getattr(self.grid_env, "enemy_positions", {}) or {}
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}

        for city_name in self.grid_objective_city_names:
            city_enemy_ids = tuple(
                int(enemy_id)
                for enemy_id in self.FINAL_OBJECTIVE_CITIES.get(city_name, ())
            )
            city_positions = [
                tuple(enemy_positions[enemy_id])
                for enemy_id in city_enemy_ids
                if enemy_id in enemy_positions
            ]
            if city_positions:
                centroid_x = sum(pos[0] for pos in city_positions) / float(len(city_positions))
                centroid_y = sum(pos[1] for pos in city_positions) / float(len(city_positions))
            else:
                centroid_x = 0.0
                centroid_y = 0.0

            captured_v = 1.0 if city_name in self.captured_objective_cities else 0.0
            enemies_remaining = sum(
                1.0 for enemy_id in city_enemy_ids if bool(enemies_alive.get(enemy_id, False))
            )
            remaining_ratio = self._normalize_ratio(
                enemies_remaining,
                max(1, len(city_enemy_ids)),
            )

            objective_obs.extend(
                [
                    centroid_x / grid_scale,
                    centroid_y / grid_scale,
                    captured_v,
                    remaining_ratio,
                ]
            )

        return np.array(objective_obs, dtype=np.float32)

    def _build_merchant_site_grid_obs(self) -> np.ndarray:
        if self.grid_merchant_site_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        current_pos = tuple(int(coord) for coord in tuple(self.grid_env.agent_pos)[:2])
        site_obs: List[float] = []

        for site_name in self.grid_merchant_site_names:
            anchor = self.merchant_site_anchors.get(site_name, (0, 0))
            total_stock = sum(
                max(0, int(stock or 0))
                for stock in self.merchant_stocks.get(site_name, {}).values()
            )
            max_total_stock = int(
                self.grid_merchant_initial_total_stock_by_site_name.get(site_name, 0) or 0
            )
            site_obs.extend(
                [
                    int(anchor[0]) / grid_scale,
                    int(anchor[1]) / grid_scale,
                    1.0
                    if current_pos in self.merchant_site_interaction_tiles.get(site_name, ())
                    else 0.0,
                    1.0 if total_stock > 0 else 0.0,
                    (
                        self._normalize_ratio(total_stock, max_total_stock)
                        if max_total_stock > 0
                        else 0.0
                    ),
                ]
            )

        return np.array(site_obs, dtype=np.float32)

    def _build_current_merchant_stock_grid_obs(self) -> np.ndarray:
        if self.grid_current_merchant_stock_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        site_names = self._merchant_sites_at_position(self.grid_env.agent_pos)
        active_site_name = str(site_names[0]) if site_names else ""
        stock_obs: List[float] = []

        for item_name in self.grid_merchant_item_names:
            current_stock = self._merchant_item_stock(active_site_name, item_name) if active_site_name else 0
            initial_stock = int(
                self.grid_merchant_initial_stock_by_site_item.get(
                    (active_site_name, item_name),
                    0,
                )
                or 0
            )
            stock_obs.append(
                self._normalize_ratio(current_stock, initial_stock) if initial_stock > 0 else 0.0
            )

        return np.array(stock_obs, dtype=np.float32)

    def _build_spell_shop_site_grid_obs(self) -> np.ndarray:
        if self.grid_spell_shop_site_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        current_pos = tuple(int(coord) for coord in tuple(self.grid_env.agent_pos)[:2])
        site_obs: List[float] = []

        for site_name in self.grid_spell_shop_site_names:
            anchor = self.spell_shop_site_anchors.get(site_name, (0, 0))
            total_stock = sum(
                max(0, int(stock or 0))
                for stock in self.spell_shop_stocks.get(site_name, {}).values()
            )
            max_total_stock = int(
                self.grid_spell_shop_initial_total_stock_by_site_name.get(site_name, 0) or 0
            )
            site_obs.extend(
                [
                    int(anchor[0]) / grid_scale,
                    int(anchor[1]) / grid_scale,
                    1.0
                    if current_pos in self.spell_shop_site_interaction_tiles.get(site_name, ())
                    else 0.0,
                    1.0 if total_stock > 0 else 0.0,
                    (
                        self._normalize_ratio(total_stock, max_total_stock)
                        if max_total_stock > 0
                        else 0.0
                    ),
                ]
            )

        return np.array(site_obs, dtype=np.float32)

    def _build_current_spell_shop_stock_grid_obs(self) -> np.ndarray:
        if self.grid_current_spell_shop_stock_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        site_names = self._spell_shop_sites_at_position(self.grid_env.agent_pos)
        active_site_name = str(site_names[0]) if site_names else ""
        stock_obs: List[float] = []

        for spell_name in self.grid_spell_shop_spell_names:
            current_stock = (
                self._spell_shop_spell_stock(active_site_name, spell_name) if active_site_name else 0
            )
            initial_stock = int(
                self.grid_spell_shop_initial_stock_by_site_spell.get(
                    (active_site_name, spell_name),
                    0,
                )
                or 0
            )
            stock_obs.append(
                self._normalize_ratio(current_stock, initial_stock) if initial_stock > 0 else 0.0
            )

        return np.array(stock_obs, dtype=np.float32)

    def _build_mercenary_site_grid_obs(self) -> np.ndarray:
        if self.grid_mercenary_site_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        grid_scale = max(1, self.grid_env.grid_size - 1)
        current_pos = tuple(int(coord) for coord in tuple(self.grid_env.agent_pos)[:2])
        site_obs: List[float] = []

        for site_name in self.grid_mercenary_site_names:
            anchor = self.mercenary_site_anchors.get(site_name, (0, 0))
            total_stock = sum(
                max(0, int(entry.get("stock", 0) or 0))
                for entry in self.mercenary_site_rosters.get(site_name, [])
            )
            max_total_stock = int(
                self.grid_mercenary_initial_total_stock_by_site_name.get(site_name, 0) or 0
            )
            site_obs.extend(
                [
                    int(anchor[0]) / grid_scale,
                    int(anchor[1]) / grid_scale,
                    1.0
                    if current_pos in self.mercenary_site_interaction_tiles.get(site_name, ())
                    else 0.0,
                    1.0 if total_stock > 0 else 0.0,
                    (
                        self._normalize_ratio(total_stock, max_total_stock)
                        if max_total_stock > 0
                        else 0.0
                    ),
                ]
            )

        return np.array(site_obs, dtype=np.float32)

    def _build_current_mercenary_roster_grid_obs(self) -> np.ndarray:
        if self.grid_mercenary_roster_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        site_names = self._mercenary_sites_at_position(self.grid_env.agent_pos)
        active_site_name = str(site_names[0]) if site_names else ""
        active_entries = self.mercenary_site_rosters.get(active_site_name, []) if active_site_name else []
        entries_by_slot: Dict[int, Dict[str, object]] = {}
        option_index_by_slot: Dict[int, int] = {}

        for entry in active_entries:
            slot_raw = entry.get("slot", 0)
            try:
                slot = int(slot_raw if slot_raw is not None else 0)
            except (TypeError, ValueError):
                continue
            entries_by_slot[slot] = entry

        if active_site_name:
            for idx, option in enumerate(self.active_mercenary_hire_options):
                if str(option.get("site_name", "") or "") != active_site_name:
                    continue
                slot_raw = option.get("slot", 0)
                try:
                    slot = int(slot_raw if slot_raw is not None else 0)
                except (TypeError, ValueError):
                    continue
                option_index_by_slot[slot] = int(idx)

        roster_obs: List[float] = []
        for slot in range(int(self.grid_mercenary_slot_count)):
            entry = entries_by_slot.get(int(slot))
            if not isinstance(entry, dict):
                roster_obs.extend([0.0] * self.grid_mercenary_slot_features_per_entry)
                continue

            stock = max(0, int(entry.get("stock", 0) or 0))
            initial_stock = max(
                0,
                int(
                    self.grid_mercenary_initial_stock_by_site_slot.get(
                        (active_site_name, int(slot)),
                        entry.get("initial_stock", stock),
                    )
                    or 0
                ),
            )
            stand = str(entry.get("stand", "") or "").strip().lower()
            option_idx = option_index_by_slot.get(int(slot))
            hireable_now = 0.0
            if option_idx is not None:
                hireable_now = 1.0 if self._can_hire_mercenary_unit_option(self._mercenary_option(option_idx)) else 0.0

            roster_obs.extend(
                [
                    1.0,
                    self._normalize_ratio(stock, initial_stock) if initial_stock > 0 else 0.0,
                    1.0 if stand == "ahead" else 0.0,
                    1.0 if stand == "behind" else 0.0,
                    self._normalize_ratio(
                        float(entry.get("gold", 0.0) or 0.0),
                        self.grid_max_mercenary_price,
                    ),
                    self._normalize_ratio(
                        float(entry.get("unit_max_health", 0.0) or 0.0),
                        self.grid_max_mercenary_unit_health,
                    ),
                    self._normalize_ratio(
                        float(entry.get("unit_damage", 0.0) or 0.0),
                        self.grid_max_mercenary_unit_damage,
                    ),
                    float(hireable_now),
                ]
            )

        return np.array(roster_obs, dtype=np.float32)

    def _build_trainer_grid_obs(self) -> np.ndarray:
        if self.grid_trainer_obs_size <= 0:
            return np.zeros(0, dtype=np.float32)

        current_pos = tuple(int(coord) for coord in tuple(self.grid_env.agent_pos)[:2])
        grid_scale = max(1, self.grid_env.grid_size - 1)
        trainer_tiles = tuple(getattr(self, "trainer_interaction_tiles", ()) or ())

        dx_norm = 0.0
        dy_norm = 0.0
        if trainer_tiles:
            nearest_tile = min(
                trainer_tiles,
                key=lambda tile: (
                    abs(int(tile[0]) - int(current_pos[0])) + abs(int(tile[1]) - int(current_pos[1])),
                    int(tile[1]),
                    int(tile[0]),
                ),
            )
            dx = int(nearest_tile[0]) - int(current_pos[0])
            dy = int(nearest_tile[1]) - int(current_pos[1])
            dx_norm = float(np.clip((float(dx) / float(grid_scale) + 1.0) * 0.5, 0.0, 1.0))
            dy_norm = float(np.clip((float(dy) / float(grid_scale) + 1.0) * 0.5, 0.0, 1.0))

        previews = [self._trainer_preview_for_position(pos) for pos in self.TRAINER_POSITIONS]
        trainable_slots = sum(1 for preview in previews if bool(preview.get("trainable", False)))
        total_affordable_xp = sum(
            max(0, int(preview.get("affordable_xp", 0) or 0))
            for preview in previews
        )
        trainer_obs = [
            float(dx_norm),
            float(dy_norm),
            1.0 if self._trainer_sites_at_position(self.grid_env.agent_pos) else 0.0,
            self._normalize_ratio(trainable_slots, max(1, len(self.TRAINER_POSITIONS))),
            self._normalize_saturating_count(
                total_affordable_xp,
                half_point=float(self.grid_trainer_total_affordable_xp_half_point),
            ),
        ]
        return np.array(trainer_obs, dtype=np.float32)

    def _build_campaign_grid_obs(self) -> np.ndarray:
        campaign_obs = np.concatenate(
            [
                self._build_blue_team_grid_obs(),
                self._build_resource_grid_obs(),
                self._build_mana_source_grid_obs(),
                self._build_building_grid_obs(),
                self._build_spell_grid_obs(),
                self._build_chest_grid_obs(),
                self._build_heal_tiles_grid_obs(),
                self._build_legions_territory_grid_obs(),
                self._build_ruin_grid_obs(),
                self._build_objective_city_grid_obs(),
                self._build_merchant_site_grid_obs(),
                self._build_current_merchant_stock_grid_obs(),
                self._build_spell_shop_site_grid_obs(),
                self._build_current_spell_shop_stock_grid_obs(),
                self._build_mercenary_site_grid_obs(),
                self._build_current_mercenary_roster_grid_obs(),
                self._build_trainer_grid_obs(),
            ]
        )
        return campaign_obs.astype(np.float32, copy=False)

    def _augment_grid_obs(self, base_obs: np.ndarray) -> np.ndarray:
        enemy_obs = self._build_enemy_grid_obs()
        campaign_obs = self._build_campaign_grid_obs()
        if enemy_obs.size == 0 and campaign_obs.size == 0:
            return base_obs.astype(np.float32, copy=False)
        return np.concatenate(
            [
                base_obs.astype(np.float32, copy=False),
                enemy_obs,
                campaign_obs,
            ]
        )

    def _get_grid_obs(self) -> np.ndarray:
        """Возвращает grid-наблюдение с признаками врагов."""
        return self._augment_grid_obs(self.grid_env._get_obs())

    def _build_obs(
        self,
        grid_obs: Optional[np.ndarray] = None,
        battle_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Строит объединённое наблюдение."""
        mode_flag = np.array([float(self.mode)], dtype=np.float32)

        if grid_obs is None:
            grid_obs = np.zeros(self.GRID_OBS_SIZE, dtype=np.float32)

        if battle_obs is None:
            battle_obs = np.zeros(self.BATTLE_OBS_SIZE, dtype=np.float32)

        turns_raw = max(0.0, float(self.turns))
        gold_raw = max(0.0, float(self.gold))
        turns_norm = self._clip01(turns_raw / (turns_raw + self.turn_norm_k))
        gold_norm = self._clip01(gold_raw / (gold_raw + self.gold_norm_k))
        extra_obs = np.array([turns_norm, gold_norm], dtype=np.float32)

        obs = np.concatenate([mode_flag, grid_obs, battle_obs, extra_obs])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        obs = np.clip(obs, 0.0, 1.0)
        return obs

    def compute_action_mask(self) -> np.ndarray:
        """Маска действий в зависимости от режима."""
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.mode == self.MODE_GRID:
            self._ensure_inventory_cache()
            # В grid режиме движение доступно только при moves > 0.
            # REST (8) доступен при ранении или когда очки перемещения закончились.
            grid_mask = self.grid_env.compute_action_mask()
            mask[:8] = bool(self.moves > 0) & grid_mask[:8]
            mask[8] = bool(grid_mask[8]) and (self._has_wounded_blue() or self.moves <= 0)
            # Общее действие лечения банками на позиции 7-12.
            if self._has_any_heal_bottle_left():
                for idx, pos in enumerate(self.GRID_BOTTLE_POSITIONS):
                    mask[self.GRID_BOTTLE_ACTION_START + idx] = self._can_heal_position(pos)
            # Бутыли воскрешения на позиции 7-12.
            if self.revive_bottles_used < self._max_revive_bottles_available():
                for idx, pos in enumerate(self.REVIVE_BOTTLE_POSITIONS):
                    mask[self.GRID_REVIVE_ACTION_START + idx] = self._can_revive_position(pos)
            if self._hero_item_available(self.INVULNERABILITY_POTION_ITEM_NAME):
                for idx, pos in enumerate(self.INVULNERABILITY_POTION_POSITIONS):
                    mask[self.GRID_INVULNERABILITY_ACTION_START + idx] = (
                        self._can_apply_invulnerability_potion_position(pos)
                    )
            if self._hero_item_available(self.STRENGTH_POTION_ITEM_NAME):
                for idx, pos in enumerate(self.STRENGTH_POTION_POSITIONS):
                    mask[self.GRID_STRENGTH_ACTION_START + idx] = (
                        self._can_apply_strength_potion_position(pos)
                    )
            if self._merchant_sites_at_position(self.grid_env.agent_pos):
                for idx, item_data in enumerate(self.MERCHANT_BUY_ITEMS):
                    item_name = str(item_data.get("name", "") or "")
                    item_price = float(item_data.get("price", 0.0) or 0.0)
                    mask[self.GRID_MERCHANT_BUY_ACTION_START + idx] = (
                        bool(item_name)
                        and float(self.gold or 0.0) >= item_price
                        and self._merchant_site_for_item(
                            item_name,
                            position=self.grid_env.agent_pos,
                        )
                        is not None
                    )
            if self._spell_shop_sites_at_position(self.grid_env.agent_pos):
                for idx, spell_data in enumerate(self.SPELL_SHOP_BUY_SPELLS):
                    spell_name = str(spell_data.get("name", "") or "")
                    spell_id = str(spell_data.get("spell_id", "") or "")
                    spell_price = float(spell_data.get("price", 0.0) or 0.0)
                    mask[self.GRID_SPELL_SHOP_BUY_ACTION_START + idx] = (
                        bool(spell_name)
                        and float(self.gold or 0.0) >= spell_price
                        and not self._spell_shop_spell_owned(spell_id)
                        and self._spell_shop_site_for_spell(
                            spell_name,
                            position=self.grid_env.agent_pos,
                        )
                        is not None
                    )
            if self._hero_item_available(self.ENERGY_ELIXIR_ITEM_NAME):
                for idx, pos in enumerate(self.ENERGY_ELIXIR_POSITIONS):
                    mask[self.GRID_ENERGY_ACTION_START + idx] = (
                        self._can_apply_energy_elixir_position(pos)
                    )
            if self._hero_item_available(self.HASTE_ELIXIR_ITEM_NAME):
                for idx, pos in enumerate(self.HASTE_ELIXIR_POSITIONS):
                    mask[self.GRID_HASTE_ACTION_START + idx] = (
                        self._can_apply_haste_elixir_position(pos)
                    )
            if self._hero_item_available(self.FIRE_WARD_ITEM_NAME):
                for idx, pos in enumerate(self.FIRE_WARD_POSITIONS):
                    mask[self.GRID_FIRE_WARD_ACTION_START + idx] = (
                        self._can_apply_fire_ward_position(pos)
                    )
            if self._hero_item_available(self.EARTH_WARD_ITEM_NAME):
                for idx, pos in enumerate(self.EARTH_WARD_POSITIONS):
                    mask[self.GRID_EARTH_WARD_ACTION_START + idx] = (
                        self._can_apply_earth_ward_position(pos)
                    )
            if self._hero_item_available(self.WATER_WARD_ITEM_NAME):
                for idx, pos in enumerate(self.WATER_WARD_POSITIONS):
                    mask[self.GRID_WATER_WARD_ACTION_START + idx] = (
                        self._can_apply_water_ward_position(pos)
                    )
            if self._hero_item_available(self.AIR_WARD_ITEM_NAME):
                for idx, pos in enumerate(self.AIR_WARD_POSITIONS):
                    mask[self.GRID_AIR_WARD_ACTION_START + idx] = (
                        self._can_apply_air_ward_position(pos)
                    )
            if self._hero_item_available(self.TITAN_ELIXIR_ITEM_NAME):
                for idx, pos in enumerate(self.TITAN_ELIXIR_POSITIONS):
                    mask[self.GRID_TITAN_ELIXIR_ACTION_START + idx] = (
                        self._can_apply_titan_elixir_position(pos)
                    )
            if self._hero_item_available(self.SUPREME_ELIXIR_ITEM_NAME):
                for idx, pos in enumerate(self.SUPREME_ELIXIR_POSITIONS):
                    mask[self.GRID_SUPREME_ELIXIR_ACTION_START + idx] = (
                        self._can_apply_supreme_elixir_position(pos)
                    )

            # Лечение и воскрешение за золото доступны только на специальных клетках лечения.
            if self._is_at_castle() and float(self.gold) > 0.0:
                if self._has_temple_built():
                    for idx, pos in enumerate(self.CASTLE_HEAL_POSITIONS):
                        mask[self.GRID_CASTLE_HEAL_ACTION_START + idx] = self._can_heal_position(pos)
                    for idx, pos in enumerate(self.CASTLE_REVIVE_POSITIONS):
                        mask[self.GRID_CASTLE_REVIVE_ACTION_START + idx] = self._can_castle_revive_position(pos)
            for idx in range(len(self.active_hire_options)):
                mask[self.GRID_HIRE_ACTION_START + idx] = self._can_hire_faction_unit_option(
                    self._hire_option(idx)
                )
            for idx in range(len(self.active_mercenary_hire_options)):
                mask[self.GRID_MERCENARY_HIRE_ACTION_START + idx] = (
                    self._can_hire_mercenary_unit_option(self._mercenary_option(idx))
                )
            if self._trainer_sites_at_position(self.grid_env.agent_pos):
                for idx, pos in enumerate(self.TRAINER_POSITIONS):
                    mask[self.GRID_TRAINER_ACTION_START + idx] = self._can_train_unit_position(
                        pos
                    )

            # Постройки: доступны при достаточном золоте и если не blocked/built
            alredybuilt_flag = self.active_buildings.get("alredybuilt", 0)
            try:
                has_build_lock = int(alredybuilt_flag or 0) == 1
            except (TypeError, ValueError):
                has_build_lock = False
            for idx, build_key in enumerate(self.building_keys):
                building = self.active_buildings.get(build_key)
                if not isinstance(building, dict):
                    continue
                price = self._get_building_gold_cost(building)
                built_flag = building.get("Build", building.get("built", 0))
                is_built = int(built_flag or 0) == 1
                blocked_flag = building.get("blocked", 0)
                is_blocked = int(blocked_flag or 0) == 1
                mask[self.GRID_BUILD_ACTION_START + idx] = (
                    not has_build_lock and not is_built and not is_blocked and self.gold >= price
                )
            if self._has_magic_tower_built() and not self.spell_learning_locked:
                for idx, spell_key in enumerate(self.spell_keys):
                    spell = self.active_spells.get(spell_key)
                    if not isinstance(spell, dict):
                        continue
                    try:
                        spell_level = int(spell.get("level", 0) or 0)
                    except (TypeError, ValueError):
                        spell_level = 0
                    if self._is_spell_level_masked_by_typeoflord(spell_level):
                        continue
                    try:
                        is_learned = int(spell.get("learned", 0) or 0) == 1
                    except (TypeError, ValueError):
                        is_learned = False
                    if is_learned:
                        continue
                    spell_costs = self._get_spell_learning_costs(spell)
                    mask[self.grid_spell_action_start + idx] = self._has_mana_for_costs(
                        spell_costs
                    )
            for idx, settlement_name in enumerate(self.settlement_upgrade_names):
                mask[self.grid_settlement_upgrade_action_start + idx] = self._can_upgrade_settlement(
                    settlement_name
                )
            for idx, item_name in enumerate(self.BATTLE_EQUIPPABLE_ITEM_NAMES):
                mask[self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + idx] = self._can_equip_battle_item(
                    0,
                    item_name,
                )
                mask[self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START + idx] = self._can_equip_battle_item(
                    1,
                    item_name,
                )
            nearest_targetable_enemy = self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=True
            )
            nearest_any_enemy = self._get_nearest_enemy_stack_for_mask(
                spell_targetable_only=False
            )
            current_specs = self._current_map_offensive_spell_action_specs()
            for idx in range(self._map_offensive_spell_action_max_count()):
                spell_key = ""
                nearest_enemy = None
                if idx < len(current_specs):
                    spell_key = str(current_specs[idx].get("id", "") or "")
                    nearest_enemy = (
                        nearest_any_enemy
                        if self._map_offensive_spell_targets_untargetable_stacks(spell_key)
                        else nearest_targetable_enemy
                    )
                mask[self.grid_legion_damage_spell_action_start + idx] = (
                    bool(spell_key)
                    and self._can_cast_legion_damage_spell(
                        spell_key,
                        nearest_enemy=nearest_enemy,
                    )
                )
            current_support_specs = self._current_map_support_spell_action_specs()
            for idx in range(self._map_support_spell_action_max_count()):
                spell_key = ""
                if idx < len(current_support_specs):
                    spell_key = str(current_support_specs[idx].get("id", "") or "")
                mask[self.grid_map_support_spell_action_start + idx] = (
                    bool(spell_key) and self._can_cast_support_spell(spell_key)
                )
            current_spell_shop_specs = self._spell_shop_cast_spell_entries()
            for idx in range(int(self.MAX_SPELL_SHOP_CAST_ACTIONS)):
                spell_key = ""
                if idx < len(current_spell_shop_specs):
                    spell_key = str(current_spell_shop_specs[idx].get("spell_id", "") or "")
                mask[self.grid_spell_shop_cast_action_start + idx] = (
                    bool(spell_key)
                    and self._can_cast_spell_shop_spell(
                        spell_key,
                        nearest_targetable_enemy=nearest_targetable_enemy,
                        nearest_any_enemy=nearest_any_enemy,
                    )
                )
            mask[self.GRID_UNLOCK_SCROLL_MAGIC_ACTION] = (
                not bool(self.scroll_magic_unlocked)
                and float(self.gold or 0.0) >= float(self.SCROLL_MAGE_UNLOCK_GOLD_COST)
                and bool(self._available_scroll_spell_entries_cache)
            )
            current_scroll_specs = self._scroll_cast_slot_entries_cache
            for idx in range(int(self.MAX_SCROLL_CAST_ACTIONS)):
                scroll_entry: Dict[str, object] = {}
                if idx < len(current_scroll_specs):
                    scroll_entry = dict(current_scroll_specs[idx])
                mask[self.grid_scroll_cast_action_start + idx] = (
                    bool(scroll_entry)
                    and self._can_cast_scroll_spell(
                        scroll_entry,
                        nearest_targetable_enemy=nearest_targetable_enemy,
                        nearest_any_enemy=nearest_any_enemy,
                    )
                )
        else:
            # В battle режиме используем маску из BattleEnv
            if self.battle_env is not None:
                battle_mask = self.battle_env.compute_action_mask()
                mask[: len(battle_mask)] = battle_mask
            else:
                # Фолбэк: все действия доступны
                mask[:] = True

        return mask

    def _log(self, message: str):
        """Добавляет сообщение в лог кампании."""
        self._campaign_logs.append(message)
        if self.log_enabled:
            print(f"[CAMPAIGN] {message}")

    def pop_campaign_logs(self) -> List[str]:
        """Возвращает и очищает логи кампании."""
        logs = self._campaign_logs[:]
        self._campaign_logs.clear()
        return logs

    def pop_battle_logs(self) -> List[str]:
        """Возвращает логи текущего боя (если есть)."""
        if self.battle_env is not None:
            return self.battle_env.pop_pretty_events()
        return []

    def _has_wounded_blue(self) -> bool:
        """Проверяет, есть ли раненые юниты BLUE (hp < max_hp, но живы)."""
        state = self._get_blue_state()
        for unit in state:
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            if hp > 0 and hp < max_hp:
                return True
        return False

    def _can_revive_position(self, position: int) -> bool:
        """Можно ли применить бутыль воскрешения к позиции (юнит мёртв: hp <= 0, есть max_hp)."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return max_hp > 0 and hp <= 0
        return False

    def _can_apply_invulnerability_potion_position(self, position: int) -> bool:
        if not self._hero_item_available(self.INVULNERABILITY_POTION_ITEM_NAME):
            return False
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return (
                max_hp > 0
                and hp > 0
                and int(position) not in self.active_invulnerability_potion_positions
            )
        return False

    def _can_apply_strength_potion_position(self, position: int) -> bool:
        if not self._hero_item_available(self.STRENGTH_POTION_ITEM_NAME):
            return False
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != int(position):
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            return (
                max_hp > 0
                and hp > 0
                and int(position) not in self.active_strength_potion_positions
            )
        return False

    def _can_castle_revive_position(self, position: int) -> bool:
        """Можно ли воскресить юнита за золото на клетке лечения."""
        revive_cost = self._castle_revive_cost_at_position(position)
        return revive_cost > 0.0 and float(self.gold or 0.0) >= revive_cost

    def _revive_unit_at_position(self, position: int) -> Tuple[bool, Optional[str]]:
        """
        Воскрешает юнита на позиции, устанавливая 1 HP, если он мёртв.
        Возвращает флаг успеха и имя юнита.
        """
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)

            if max_hp <= 0 or hp > 0:
                return False, None

            unit["hp"] = 1.0
            unit["health"] = 1.0
            name = unit.get("name", f"pos_{position}")
            self._log(f"Бутыль воскрешения: {name} (pos {position}) оживлён с 1 HP")
            return True, name

        return False, None

    def _can_heal_position(self, position: int) -> bool:
        """Можно ли применить бутыль лечения к указанной позиции (юнит жив и не на фулл HP)."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)
            return max_hp > 0 and hp > 0 and hp < max_hp
        return False

    def get_nearest_enemy_stack(
        self,
        from_pos: Optional[Tuple[int, int]] = None,
        *,
        only_alive: bool = True,
        spell_targetable_only: bool = False,
    ) -> Optional[Dict[str, object]]:
        """
        Возвращает один ближайший вражеский отряд на карте.

        Правило выбора детерминированное:
        1. минимальная длина пути по текущей карте с учётом препятствий;
        2. при равенстве меньший enemy_id;
        3. затем верхняя-левая координата (y, x).

        Если путь ни к одному врагу недоступен, используется fallback по
        свободной от препятствий дистанции хода (Chebyshev) с тем же tie-break.
        """
        enemy_positions = getattr(self.grid_env, "enemy_positions", {}) or {}
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}
        if not enemy_positions:
            return None

        start_pos = tuple(from_pos) if from_pos is not None else tuple(self.grid_env.agent_pos)
        start_tile = (
            int(start_pos[0]),
            int(start_pos[1]),
        )
        blocked_tiles = {
            (int(pos[0]), int(pos[1]))
            for pos in (getattr(self.grid_env, "obstacle_positions", set()) or set())
        }
        path_distances = self._build_territory_path_distances(
            source_tile=start_tile,
            blocked_tile_set=blocked_tiles,
        )

        reachable_candidates: List[Tuple[int, int, int, int]] = []
        fallback_candidates: List[Tuple[int, int, int, int]] = []

        for raw_enemy_id, raw_pos in enemy_positions.items():
            enemy_id = int(raw_enemy_id)
            if only_alive and not bool(enemies_alive.get(enemy_id, False)):
                continue
            if spell_targetable_only and not self._is_enemy_stack_spell_targetable(enemy_id):
                continue

            ex = int(raw_pos[0])
            ey = int(raw_pos[1])
            enemy_tile = (ex, ey)

            if enemy_tile in path_distances:
                reachable_candidates.append((int(path_distances[enemy_tile]), enemy_id, ey, ex))

            chebyshev_distance = max(abs(start_tile[0] - ex), abs(start_tile[1] - ey))
            fallback_candidates.append((int(chebyshev_distance), enemy_id, ey, ex))

        if not fallback_candidates:
            return None

        best_distance, best_enemy_id, best_y, best_x = min(
            reachable_candidates or fallback_candidates
        )
        best_pos = (int(best_x), int(best_y))
        is_reachable = bool(reachable_candidates)

        return {
            "enemy_id": int(best_enemy_id),
            "position": best_pos,
            "path_distance": int(best_distance) if is_reachable else None,
            "fallback_distance": None if is_reachable else int(best_distance),
            "reachable": is_reachable,
            "is_alive": bool(enemies_alive.get(int(best_enemy_id), False)),
            "description": ENEMY_DESCRIPTIONS.get(int(best_enemy_id), "Неизвестный враг"),
        }

    def _get_nearest_enemy_stack_for_mask(
        self,
        from_pos: Optional[Tuple[int, int]] = None,
        *,
        only_alive: bool = True,
        spell_targetable_only: bool = False,
    ) -> Optional[Dict[str, object]]:
        """
        Дешёвый кандидат ближайшего врага для лёгких grid-проверок.

        В отличие от get_nearest_enemy_stack() не строит BFS по карте, а использует
        только геометрическую близость (Chebyshev) и состояние живых отрядов.
        Этого достаточно для action mask и spell-observation: сама цель при реальном
        касте всё равно выбирается точным методом в _step_cast_legion_damage_spell().
        """
        enemy_positions = getattr(self.grid_env, "enemy_positions", {}) or {}
        enemies_alive = getattr(self.grid_env, "enemies_alive", {}) or {}
        if not enemy_positions:
            return None

        start_pos = tuple(from_pos) if from_pos is not None else tuple(self.grid_env.agent_pos)
        start_x = int(start_pos[0])
        start_y = int(start_pos[1])
        candidates: List[Tuple[int, int, int, int]] = []

        for raw_enemy_id, raw_pos in enemy_positions.items():
            enemy_id = int(raw_enemy_id)
            if only_alive and not bool(enemies_alive.get(enemy_id, False)):
                continue
            if spell_targetable_only and not self._is_enemy_stack_spell_targetable(enemy_id):
                continue
            if not self._enemy_team_has_living_units(enemy_id):
                continue

            ex = int(raw_pos[0])
            ey = int(raw_pos[1])
            chebyshev_distance = max(abs(start_x - ex), abs(start_y - ey))
            candidates.append((int(chebyshev_distance), enemy_id, ey, ex))

        if not candidates:
            return None

        best_distance, best_enemy_id, best_y, best_x = min(candidates)
        best_pos = (int(best_x), int(best_y))
        return {
            "enemy_id": int(best_enemy_id),
            "position": best_pos,
            "path_distance": None,
            "fallback_distance": int(best_distance),
            "reachable": False,
            "is_alive": bool(enemies_alive.get(int(best_enemy_id), False)),
            "description": ENEMY_DESCRIPTIONS.get(int(best_enemy_id), "Неизвестный враг"),
        }

    def render(self, mode: str = "ansi") -> Optional[str]:
        """Отрисовка текущего состояния."""
        if mode != "ansi":
            return None

        self._refresh_campaign_artifact_effects(log=False)
        lines = []
        lines.append(f"=== CAMPAIGN MODE: {'GRID' if self.mode == self.MODE_GRID else 'BATTLE'} ===")

        if self.mode == self.MODE_GRID:
            lines.append("")
            grid_render = self.grid_env.render(mode="ansi")
            if grid_render:
                lines.append(grid_render)
            lines.append(f"Ход: {self.turns} | Золото: {self.gold:g} | Шаги: {self.moves}")
        else:
            lines.append(f"В бою с врагом {self.current_enemy_id}")
            lines.append(f"Описание: {ENEMY_DESCRIPTIONS.get(self.current_enemy_id, '?')}")
            lines.append(f"Ход: {self.turns} | Золото: {self.gold:g} | Шаги: {self.moves}")
        mana_totals = self._current_mana_totals()
        mana_income = self._compute_mana_income_per_turn()
        mana_text = ", ".join(
            f"{self.MANA_LABEL_BY_KIND[str(kind)]}: {mana_totals[str(kind)]:g} (+{mana_income[str(kind)]:g}/ход)"
            for kind in self.MANA_KIND_ORDER
        )
        lines.append(f"Мана: {mana_text}")
        lines.append(
            "Башня магии: "
            + ("построена" if self._has_magic_tower_built() else "не построена")
        )
        lines.append(
            f"Изучение заклинаний в этом ходу: "
            + ("заблокировано" if self.spell_learning_locked else "доступно")
        )

        if self.chests:
            chest_text = ", ".join(
                f"{pos}:{list(item_names)}" for pos, item_names in sorted(self.chests.items())
            )
            lines.append(f"Сундуки: {chest_text}")
        else:
            lines.append("Сундуки: нет")
        if self.heroitems:
            lines.append(
                "Предметы героя: "
                + ", ".join(self._hero_item_display_name(entry) for entry in self.heroitems)
            )
        else:
            lines.append("Предметы героя: нет")
        lines.append(
            "Маг для свитков: "
            + ("нанят" if self.scroll_magic_unlocked else "не нанят")
        )
        scroll_state = self._scroll_state_info()
        scroll_inventory = list(scroll_state.get("scroll_inventory", []))
        if scroll_inventory:
            lines.append(
                "Свитки в инвентаре: "
                + ", ".join(
                    f"{entry['spell_name']} x{int(entry['copies'])}"
                    + ("" if entry.get("supported", False) else " (unsupported)")
                    for entry in scroll_inventory
                )
            )
        else:
            lines.append("Свитки в инвентаре: нет")
        scroll_slot_entries = self.scroll_cast_slot_entries()
        if scroll_slot_entries:
            lines.append("Слоты scroll-action:")
            lines.extend(
                f"- slot {idx}: {entry['spell_name']} x{int(entry['copies'])}"
                for idx, entry in enumerate(scroll_slot_entries)
            )
        equipped_labels = [
            str(item_name or "пусто") for item_name in list(self.equipped_hero_items or [])
        ]
        lines.append(f"Экипированные предметы: [{', '.join(equipped_labels)}]")
        artifact_labels = [
            str(item_name or "пусто") for item_name in list(self.equipped_artifact_items or [])
        ]
        lines.append(f"Экипированные артефакты: [{', '.join(artifact_labels)}]")
        lines.append(
            f"Боевые предметы: экипировано {int(self.battle_items_equipped_total)}, "
            f"использовано {int(self.battle_items_used_total)}"
        )
        lines.append(f"Доп. бутылей лечения: {max(0, int(self.extra_heal_bottles or 0))}")
        lines.append(f"Доп. банок исцеления: {max(0, int(self.extra_healing_bottles or 0))}")
        lines.append(f"Доп. зелий воскрешения: {max(0, int(self.extra_revive_bottles or 0))}")
        lines.append(
            "Зелий неуязвимости: "
            f"{self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME)}"
        )
        lines.append(
            "Зелий силы: "
            f"{self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME)}"
        )
        if self.active_invulnerability_potion_positions:
            lines.append(
                "Активное зелье неуязвимости на позициях: "
                f"{sorted(self.active_invulnerability_potion_positions)}"
            )
        if self.active_strength_potion_positions:
            lines.append(
                "Активное зелье силы на позициях: "
                f"{sorted(self.active_strength_potion_positions)}"
            )
        learned_spells = self.get_learned_spell_descriptions()
        if learned_spells:
            lines.append(
                f"Изученные заклинания ({len(learned_spells)}/{len(self.spell_keys)}):"
            )
            lines.extend(f"- {description}" for description in learned_spells)
        else:
            lines.append(f"Изученные заклинания: 0/{len(self.spell_keys)}")

        result = "\n".join(lines)
        print(result)
        return result

    def get_state_summary(self) -> Dict:
        """Возвращает сводку текущего состояния."""
        self._refresh_campaign_artifact_effects(log=False)
        scroll_state = self._scroll_state_info()
        return {
            "mode": "grid" if self.mode == self.MODE_GRID else "battle",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "enemies_defeated": [
                eid for eid, alive in self.grid_env.enemies_alive.items() if not alive
            ],
            "visited_cells": len(self.grid_env.visited_cells),
            "grid_steps": self.grid_env.step_count,
            "current_enemy": self.current_enemy_id,
            "blue_hp_saved": self.blue_team_state is not None,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "typeoflord": int(self.typeoflord),
            "heroitems": list(self.heroitems),
            "equipped_hero_items": list(self.equipped_hero_items),
            "equipped_artifact_items": list(self.equipped_artifact_items),
            "artifact_auto_equip_next_slot": int(self.artifact_auto_equip_next_slot),
            "battle_items_equipped_total": int(self.battle_items_equipped_total),
            "battle_items_used_total": int(self.battle_items_used_total),
            "learned_spells": list(self.get_learned_spell_descriptions()),
            "learned_spells_count": len(self.get_learned_spell_descriptions()),
            "spells_total": len(self.spell_keys),
            "spell_learning_locked": bool(self.spell_learning_locked),
            "magic_tower_built": bool(self._has_magic_tower_built()),
            "scroll_magic_unlocked": bool(self.scroll_magic_unlocked),
            "scroll_inventory": list(scroll_state.get("scroll_inventory", [])),
            "scroll_cast_slots": list(scroll_state.get("scroll_cast_slots", [])),
            "supported_scroll_types_count": int(scroll_state.get("supported_scroll_types_count", 0) or 0),
            "total_scroll_count": int(scroll_state.get("total_scroll_count", 0) or 0),
            "extra_heal_bottles": int(self.extra_heal_bottles or 0),
            "extra_healing_bottles": int(self.extra_healing_bottles or 0),
            "extra_revive_bottles": int(self.extra_revive_bottles or 0),
            "invulnerability_potions_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_potions_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
            "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
            "active_strength_positions": sorted(self.active_strength_potion_positions),
            "merchant_stocks": {
                site_name: self._merchant_stock_snapshot(site_name)
                for site_name in self.merchant_stocks.keys()
            },
            "spell_shop_stocks": {
                site_name: self._spell_shop_stock_snapshot(site_name)
                for site_name in self.spell_shop_stocks.keys()
            },
            "spell_shop_sites_in_range": list(self._spell_shop_sites_at_position()),
            "mercenary_stocks": {
                site_name: self._mercenary_stock_snapshot(site_name)
                for site_name in self.mercenary_site_rosters.keys()
            },
            "trainer_sites_in_range": list(self._trainer_sites_at_position()),
            **self._mana_state_info(),
            "chests_remaining": [
                {"pos": pos, "items": list(item_names)}
                for pos, item_names in sorted(self.chests.items())
            ],
        }
