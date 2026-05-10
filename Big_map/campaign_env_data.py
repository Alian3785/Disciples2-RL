# campaign_env_data.py
"""Shared imports, data mappings, and constants for CampaignEnv modules."""

from __future__ import annotations

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

from campaign_env_scenario import *

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


class HeroInventoryItem(str):
    """String-compatible inventory entry that also stores gold value."""

    def __new__(cls, name: str, gold: int = 0):
        obj = str.__new__(cls, str(name or ""))
        obj.gold = max(0, int(gold or 0))
        return obj

    def __repr__(self) -> str:
        return f"HeroInventoryItem(name={str.__repr__(self)}, gold={int(self.gold)})"


class CampaignConstantsMixin:

    metadata = {"render_modes": ["human", "ansi"]}

    MODE_GRID = 0
    MODE_BATTLE = 1

    GRID_OBS_SIZE = 0  # вычисляется в __init__
    BATTLE_OBS_SIZE = FEATURES_PER_UNIT * 12
    MOVES_PER_TURN = 20
    HERO_BASE_MOVES_BY_NAME = {
        "Рыцарь на пегасе": 20,
        "Рыцарь на Пегасе": 20,
        "Рейнджер людей": 35,
        "Рейнджер": 35,
        "Следопыт": 35,
        "Архимаг": 20,
        "Архангел": 20,
        "Герцог": 20,
        "Советник": 35,
        "Архидьявол": 20,
        "Баронесса": 20,
        "Королевский страж": 20,
        "Инженер": 30,
        "Хранитель знаний": 20,
        "Ученый": 20,
        "Жезловик гномов": 25,
        "Старейшина": 25,
        "Рыцарь смерти": 20,
        "Рыцарь Смерти": 20,
        "Носферату": 35,
        "Королева лич": 20,
        "Королева личей": 20,
        "Баньши": 20,
        "Лесной лорд": 25,
        "Кентавр Дикарь": 25,
        "Хранитель леса": 35,
        "Страж": 35,
        "Дриада": 25,
        "Мудрец": 20,
    }
    GRID_MOVE_COST = 2
    GRID_TERRAIN_PLAIN = "plain"
    GRID_TERRAIN_FOREST = "forest"
    GRID_TERRAIN_WATER = "water"
    GRID_TERRAIN_ROAD = "road"
    GRID_TERRAIN_MOVE_COSTS = {
        GRID_TERRAIN_PLAIN: 2,
        GRID_TERRAIN_FOREST: 4,
        GRID_TERRAIN_WATER: 6,
        GRID_TERRAIN_ROAD: 1,
    }
    GRID_TERRAIN_BOOT_MOVE_COST = 2
    GRID_STEPS_PER_TURN = MOVES_PER_TURN
    HERO_FLYING_ABILITY_KEY = "flying"
    HERO_FLYING_NAMES = frozenset(
        {
            "Рыцарь на пегасе",
            "Рыцарь на Пегасе",
            "Архангел",
            "Архидьявол",
            "Баньши",
        }
    )
    HERO_PATHFINDING_LEVEL = 2
    HERO_LEADERSHIP_LEVEL = 3
    HERO_ENDURANCE_LEVEL = 4
    HERO_STRENGTH_LEVEL = 5
    HERO_SECOND_LEADERSHIP_LEVEL = 6
    HERO_BANNER_BEARER_LEVEL = 7
    HERO_BANNER_BEARER_DESCRIPTION = "Позволяет предводителю использовать знамёна"
    HERO_BANNER_BEARER_ABILITY_KEY = "banner_bearer"
    HERO_MARCHING_LORE_LEVEL = 8
    HERO_MARCHING_LORE_DESCRIPTION = "Позволяет предводителю носить магические ботинки"
    HERO_MARCHING_LORE_ABILITY_KEY = "marching_lore"
    HERO_ARTIFACT_KNOWLEDGE_LEVEL = 9
    HERO_ARTIFACT_KNOWLEDGE_DESCRIPTION = "Позволяет предводителю использовать артефакты."
    HERO_ARTIFACT_KNOWLEDGE_ABILITY_KEY = "artifact_knowledge"
    HERO_SORCERY_LORE_LEVEL = 10
    HERO_SORCERY_LORE_DESCRIPTION = "Позволяет предводителю экипировать и использовать сферы в бою"
    HERO_SORCERY_LORE_ABILITY_KEY = "sorcery_lore"
    HERO_BOOK_LORE_LEVEL = 11
    HERO_BOOK_LORE_DESCRIPTION = "Позволяет предводителю читать магические книги"
    HERO_BOOK_LORE_ABILITY_KEY = "book_lore"
    HERO_MIGHT_DESCRIPTION = "+25% основной урон героя"
    HERO_MIGHT_ABILITY_KEY = "might"
    HERO_MIGHT_DAMAGE_MULTIPLIER = 1.25
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
    GRID_POTION_USE_ACTION_START = GRID_BOTTLE_ACTION_START
    GRID_POTION_USE_POSITIONS = GRID_BOTTLE_POSITIONS
    GRID_POTION_USE_ACTION_COUNT = 0
    GRID_MERCHANT_POTION_BUY_ACTION_START = GRID_BOTTLE_ACTION_START
    GRID_MERCHANT_POTION_BUY_ACTION_COUNT = 0
    GRID_EQUIP_BATTLE_POTION_ACTION_START = GRID_BOTTLE_ACTION_START
    GRID_EQUIP_BATTLE_POTION_ACTION_COUNT = 0
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
    INVULNERABILITY_POTION_ITEM_NAME = "Эликсир неуязвимости"
    INVULNERABILITY_POTION_ARMOR_BONUS = 50
    GRID_STRENGTH_ACTION_START = (
        GRID_INVULNERABILITY_ACTION_START + len(INVULNERABILITY_POTION_POSITIONS)
    )  # 27
    STRENGTH_POTION_POSITIONS = GRID_BOTTLE_POSITIONS
    STRENGTH_POTION_ITEM_NAME = "Зелье силы (+30%)"
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
    TALISMAN_USES_PER_ITEM = 5
    BASE_BATTLE_EQUIPPABLE_ITEM_NAMES = (
        BONUS_SMALL_HEAL_ITEM_NAME,
        BONUS_LARGE_HEAL_ITEM_NAME,
        HEALING_OINTMENT_ITEM_NAME,
        BONUS_REVIVE_ITEM_NAME,
    )
    BATTLE_EQUIPPABLE_ITEM_NAMES = BASE_BATTLE_EQUIPPABLE_ITEM_NAMES
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
    ARTIFACT_ITEM_GOLD_VALUES = {
        RUNESTONE_ARTIFACT_ITEM_NAME: 500,
        HOLY_CHALICE_ARTIFACT_ITEM_NAME: 1000,
        SKULL_BRACERS_ITEM_NAME: 1500,
        HORN_OF_AWARENESS_ITEM_NAME: 2000,
        ETCHED_CIRCLET_ITEM_NAME: 2500,
        DWARVEN_BRACER_ITEM_NAME: 500,
        UNHOLY_CHALICE_ARTIFACT_ITEM_NAME: 1000,
        RING_OF_STRENGTH_ITEM_NAME: 1500,
        RUNIC_BLADE_ITEM_NAME: 2000,
        MJOLNIRS_CROWN_ITEM_NAME: 2500,
        RING_OF_AGES_ITEM_NAME: 3750,
        RING_OF_AGES_ENGLISH_ITEM_NAME: 3750,
        BETHREZENS_CLAW_ITEM_NAME: 5000,
        SOUL_CRYSTAL_ARTIFACT_ITEM_NAME: 2000,
        HORN_OF_INCUBUS_ARTIFACT_ITEM_NAME: 2000,
        UNHOLY_DAGGER_ARTIFACT_ITEM_NAME: 1500,
        THANATOS_BLADE_ARTIFACT_ITEM_NAME: 1500,
        SKULL_OF_THANATOS_ARTIFACT_ITEM_NAME: 3750,
        HAGS_RING_ARTIFACT_ITEM_NAME: 2500,
    }
    BANNER_OF_PROTECTION_ITEM_NAME = "Banner of Protection"
    BANNER_OF_RESISTANCE_ITEM_NAME = "Banner of Resistance"
    BANNER_OF_FORTITUDE_ITEM_NAME = "Banner of Fortitude"
    BANNER_OF_STRIKING_ITEM_NAME = "Banner of Striking"
    BANNER_OF_BATTLE_ITEM_NAME = "Banner of Battle"
    BANNER_OF_SPEED_ITEM_NAME = "Banner of Speed"
    BANNER_OF_CELERITY_ITEM_NAME = "Banner of Celerity"
    BANNER_OF_STRENGTH_ITEM_NAME = "Banner of Strength"
    BANNER_OF_MIGHT_ITEM_NAME = "Banner of Might"
    BANNER_OF_WAR_ITEM_NAME = "Banner of War"
    BANNER_EQUIP_SLOTS = 1
    BANNER_ITEM_NAMES = (
        BANNER_OF_PROTECTION_ITEM_NAME,
        BANNER_OF_RESISTANCE_ITEM_NAME,
        BANNER_OF_FORTITUDE_ITEM_NAME,
        BANNER_OF_STRIKING_ITEM_NAME,
        BANNER_OF_BATTLE_ITEM_NAME,
        BANNER_OF_SPEED_ITEM_NAME,
        BANNER_OF_CELERITY_ITEM_NAME,
        BANNER_OF_STRENGTH_ITEM_NAME,
        BANNER_OF_MIGHT_ITEM_NAME,
        BANNER_OF_WAR_ITEM_NAME,
    )
    BANNER_EQUIP_PRIORITY = (
        BANNER_OF_WAR_ITEM_NAME,
        BANNER_OF_FORTITUDE_ITEM_NAME,
        BANNER_OF_MIGHT_ITEM_NAME,
        BANNER_OF_RESISTANCE_ITEM_NAME,
        BANNER_OF_BATTLE_ITEM_NAME,
        BANNER_OF_CELERITY_ITEM_NAME,
        BANNER_OF_STRENGTH_ITEM_NAME,
        BANNER_OF_PROTECTION_ITEM_NAME,
        BANNER_OF_STRIKING_ITEM_NAME,
        BANNER_OF_SPEED_ITEM_NAME,
    )
    BANNER_ITEM_GOLD_VALUES = {
        BANNER_OF_PROTECTION_ITEM_NAME: 1000,
        BANNER_OF_STRIKING_ITEM_NAME: 1000,
        BANNER_OF_SPEED_ITEM_NAME: 1000,
        BANNER_OF_STRENGTH_ITEM_NAME: 1000,
        BANNER_OF_RESISTANCE_ITEM_NAME: 3000,
        BANNER_OF_BATTLE_ITEM_NAME: 3000,
        BANNER_OF_CELERITY_ITEM_NAME: 3000,
        BANNER_OF_MIGHT_ITEM_NAME: 3000,
        BANNER_OF_FORTITUDE_ITEM_NAME: 5000,
        BANNER_OF_WAR_ITEM_NAME: 5000,
    }
    BANNER_EFFECT_DEFINITIONS = {
        BANNER_OF_PROTECTION_ITEM_NAME: {"armor_bonus": 10},
        BANNER_OF_RESISTANCE_ITEM_NAME: {"armor_bonus": 15},
        BANNER_OF_FORTITUDE_ITEM_NAME: {"armor_bonus": 20},
        BANNER_OF_STRIKING_ITEM_NAME: {"accuracy_bonus": 10},
        BANNER_OF_BATTLE_ITEM_NAME: {"accuracy_bonus": 15},
        BANNER_OF_SPEED_ITEM_NAME: {"initiative_multiplier": 1.10},
        BANNER_OF_CELERITY_ITEM_NAME: {"initiative_multiplier": 1.15},
        BANNER_OF_STRENGTH_ITEM_NAME: {"damage_multiplier": 1.10},
        BANNER_OF_MIGHT_ITEM_NAME: {"damage_multiplier": 1.15},
        BANNER_OF_WAR_ITEM_NAME: {"damage_multiplier": 1.20},
    }
    TOME_OF_AIR_ITEM_NAME = "Tome of Air"
    TOME_OF_WATER_ITEM_NAME = "Tome of Water"
    TOME_OF_EARTH_ITEM_NAME = "Tome of Earth"
    TOME_OF_FIRE_ITEM_NAME = "Tome of Fire"
    TOME_OF_WAR_ITEM_NAME = "Tome of War"
    TOME_OF_ARCANUM_ITEM_NAME = "Tome of Arcanum"
    TOME_OF_SORCERY_ITEM_NAME = "Tome of Sorcery"
    TOME_OF_MIND_ITEM_NAME = "Tome of Mind"
    BOOK_EQUIP_SLOTS = 1
    BOOK_ITEM_NAMES = (
        TOME_OF_AIR_ITEM_NAME,
        TOME_OF_WATER_ITEM_NAME,
        TOME_OF_EARTH_ITEM_NAME,
        TOME_OF_FIRE_ITEM_NAME,
        TOME_OF_WAR_ITEM_NAME,
        TOME_OF_ARCANUM_ITEM_NAME,
        TOME_OF_SORCERY_ITEM_NAME,
        TOME_OF_MIND_ITEM_NAME,
    )
    BOOK_ITEM_GOLD_VALUES = {
        TOME_OF_AIR_ITEM_NAME: 400,
        TOME_OF_WATER_ITEM_NAME: 400,
        TOME_OF_EARTH_ITEM_NAME: 400,
        TOME_OF_FIRE_ITEM_NAME: 400,
        TOME_OF_WAR_ITEM_NAME: 600,
        TOME_OF_ARCANUM_ITEM_NAME: 900,
        TOME_OF_SORCERY_ITEM_NAME: 1200,
        TOME_OF_MIND_ITEM_NAME: 900,
    }
    BOOK_EFFECT_DEFINITIONS = {
        TOME_OF_AIR_ITEM_NAME: {"resistance_type": "Air"},
        TOME_OF_WATER_ITEM_NAME: {"resistance_type": "Water"},
        TOME_OF_EARTH_ITEM_NAME: {"resistance_type": "Earth"},
        TOME_OF_FIRE_ITEM_NAME: {"resistance_type": "Fire"},
        TOME_OF_MIND_ITEM_NAME: {"resistance_type": "Mind"},
        TOME_OF_WAR_ITEM_NAME: {"exp_multiplier": 1.25},
        TOME_OF_ARCANUM_ITEM_NAME: {"unlocks_orbs": True},
        TOME_OF_SORCERY_ITEM_NAME: {"unlocks_talismans": True},
    }
    ELVEN_BOOTS_ITEM_NAME = "Elven Boots"
    BOOTS_OF_THE_ELEMENTS_ITEM_NAME = "Boots of the Elements"
    BOOTS_OF_SPEED_ITEM_NAME = "Boots of Speed"
    BOOTS_OF_TRAVELING_ITEM_NAME = "Boots of Traveling"
    BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME = "Boots of Seven Leagues"
    BOOTS_EQUIP_SLOTS = 1
    BOOT_ITEM_NAMES = (
        ELVEN_BOOTS_ITEM_NAME,
        BOOTS_OF_THE_ELEMENTS_ITEM_NAME,
        BOOTS_OF_SPEED_ITEM_NAME,
        BOOTS_OF_TRAVELING_ITEM_NAME,
        BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME,
    )
    BOOT_ITEM_GOLD_VALUES = {
        ELVEN_BOOTS_ITEM_NAME: 1200,
        BOOTS_OF_THE_ELEMENTS_ITEM_NAME: 1600,
        BOOTS_OF_SPEED_ITEM_NAME: 400,
        BOOTS_OF_TRAVELING_ITEM_NAME: 1200,
        BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME: 2000,
    }
    BOOT_EFFECT_DEFINITIONS = {
        ELVEN_BOOTS_ITEM_NAME: {"move_bonus": 0, "terrain_move_bonuses": {"forest": 2}},
        BOOTS_OF_THE_ELEMENTS_ITEM_NAME: {"move_bonus": 0, "terrain_move_bonuses": {"water": 2}},
        BOOTS_OF_SPEED_ITEM_NAME: {"move_bonus": 4},
        BOOTS_OF_TRAVELING_ITEM_NAME: {"move_bonus": 8},
        BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME: {"move_bonus": 12},
    }
    ENERGY_ELIXIR_DAMAGE_MULTIPLIER = 1.50
    HASTE_ELIXIR_INITIATIVE_MULTIPLIER = 1.60
    TITAN_ELIXIR_DAMAGE_MULTIPLIER = 1.10
    SUPREME_ELIXIR_HEALTH_MULTIPLIER = 1.15
    PROTECTION_POTION_ITEM_NAME = "Зелье защиты (-15%)"
    BARK_POTION_ITEM_NAME = "Зелье коры дерева (-30%)"
    IRON_SKIN_POTION_ITEM_NAME = "Зелье железной кожи (-10%)"
    HIT_POTION_ITEM_NAME = "Зелье меткости (+15%)"
    ACCURACY_POTION_ITEM_NAME = "Зелье точности (+30%)"
    LUCK_POTION_ITEM_NAME = "Эликсир удачи (+10%)"
    SWIFTNESS_POTION_ITEM_NAME = "Зелье проворства (+15%)"
    SPEED_POTION_ITEM_NAME = "Зелье скорости (+30%)"
    MERCURY_ELIXIR_ITEM_NAME = "Эликсир Меркурия (+60%)"
    INITIATIVE_ELIXIR_ITEM_NAME = "Эликсир инициативы (+10%)"
    VIGOR_POTION_ITEM_NAME = "Зелье бодрости (+15%)"
    MIGHT_POTION_ITEM_NAME = "Зелье мощи (+30%)"
    STRENGTH_POTION_CANONICAL_ITEM_NAME = "Зелье силы (+30%)"
    POTION_DAMAGE_REDUCTION_INVULNERABILITY_MULTIPLIER = 0.50
    POTION_DAMAGE_REDUCTION_BARK_MULTIPLIER = 0.70
    POTION_DAMAGE_REDUCTION_PROTECTION_MULTIPLIER = 0.85
    POTION_DAMAGE_REDUCTION_IRON_SKIN_MULTIPLIER = 0.90
    POTION_ACCURACY_HIT_MULTIPLIER = 1.15
    POTION_ACCURACY_ACCURACY_MULTIPLIER = 1.30
    POTION_ACCURACY_LUCK_MULTIPLIER = 1.10
    POTION_INITIATIVE_SWIFTNESS_MULTIPLIER = 1.15
    POTION_INITIATIVE_SPEED_MULTIPLIER = 1.30
    POTION_INITIATIVE_PERMANENT_MULTIPLIER = 1.10
    POTION_DAMAGE_VIGOR_MULTIPLIER = 1.15
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
    POTION_ITEM_DEFINITIONS = (
        {
            "name": BONUS_SMALL_HEAL_ITEM_NAME,
            "aliases": (
                BONUS_SMALL_HEAL_SOURCE_ITEM,
                "Эликсир исцеления",
                "Potion of Healing",
                "Healing Potion",
                "Small Healing Potion",
            ),
            "category": "healing",
            "effect": {"kind": "heal", "amount": HEALING_BOTTLE_AMOUNT},
            "duration": "instant",
            "merchant_grant": "small_heal_bonus",
            "battle_equip": True,
            "gold_value": 150,
        },
        {
            "name": BONUS_LARGE_HEAL_ITEM_NAME,
            "aliases": (
                "Эликсир восстановления",
                "Potion of Restoration",
                "Large Healing Potion",
                "Healing Bottle",
            ),
            "category": "healing",
            "effect": {"kind": "heal", "amount": HEAL_BOTTLE_AMOUNT},
            "duration": "instant",
            "merchant_grant": "large_heal_bonus",
            "battle_equip": True,
            "gold_value": 300,
        },
        {
            "name": HEALING_OINTMENT_ITEM_NAME,
            "aliases": (
                "Healing Ointment",
                "Healing Salve",
                "Ointment of Healing",
            ),
            "category": "healing",
            "effect": {"kind": "heal", "amount": HEALING_OINTMENT_AMOUNT},
            "duration": "instant",
            "merchant_grant": "inventory",
            "battle_equip": True,
            "gold_value": 600,
        },
        {
            "name": BONUS_REVIVE_ITEM_NAME,
            "aliases": (
                BONUS_REVIVE_SOURCE_ITEM,
                RUIN_LIFE_ELIXIR_ITEM_NAME,
                "Life Potion",
                "Potion of Life",
                "Revive Potion",
                "Эликсир Жизни",
            ),
            "category": "revive",
            "effect": {"kind": "revive", "amount": 1.0},
            "duration": "instant",
            "merchant_grant": "revive_bonus",
            "battle_equip": True,
            "gold_value": 400,
        },
        {
            "name": SUPREME_ELIXIR_ITEM_NAME,
            "aliases": (
                "Highfather Potion",
                "Highfather Elixir",
                "Elixir of the Highfather",
                "Supreme Elixir",
            ),
            "category": "permanent_health",
            "effect": {"kind": "health", "multiplier": SUPREME_ELIXIR_HEALTH_MULTIPLIER},
            "duration": "permanent",
            "battle_equip": False,
            "permanent_counter_key": "campaign_supreme_elixir_uses",
            "gold_value": 800,
        },
        {
            "name": PROTECTION_POTION_ITEM_NAME,
            "aliases": (
                "Potion of Protection",
                "Protection Potion",
                "Зелье защиты",
            ),
            "category": "temporary_damage_reduction",
            "effect": {
                "kind": "damage_reduction",
                "incoming_damage_multiplier": POTION_DAMAGE_REDUCTION_PROTECTION_MULTIPLIER,
            },
            "duration": "temporary",
            "battle_equip": False,
            "gold_value": 350,
        },
        {
            "name": BARK_POTION_ITEM_NAME,
            "aliases": (
                "Potion of Bark",
                "Potion of Barkskin",
                "Bark Potion",
                "Barkskin Potion",
                "Зелье коры дерева",
            ),
            "category": "temporary_damage_reduction",
            "effect": {
                "kind": "damage_reduction",
                "incoming_damage_multiplier": POTION_DAMAGE_REDUCTION_BARK_MULTIPLIER,
            },
            "duration": "temporary",
            "battle_equip": False,
            "gold_value": 500,
        },
        {
            "name": RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME,
            "aliases": (
                INVULNERABILITY_POTION_ITEM_NAME,
                "Potion of Invulnerability",
                "Invulnerability Potion",
                "Elixir of Invulnerability",
            ),
            "category": "temporary_damage_reduction",
            "effect": {
                "kind": "damage_reduction",
                "incoming_damage_multiplier": POTION_DAMAGE_REDUCTION_INVULNERABILITY_MULTIPLIER,
            },
            "duration": "temporary",
            "battle_equip": False,
            "active_attr": "active_invulnerability_potion_positions",
            "gold_value": 700,
        },
        {
            "name": IRON_SKIN_POTION_ITEM_NAME,
            "aliases": (
                "Potion of Iron Skin",
                "Iron Skin Potion",
                "Зелье железной кожи",
            ),
            "category": "permanent_damage_reduction",
            "effect": {
                "kind": "damage_reduction",
                "incoming_damage_multiplier": POTION_DAMAGE_REDUCTION_IRON_SKIN_MULTIPLIER,
            },
            "duration": "permanent",
            "battle_equip": False,
            "permanent_counter_key": "campaign_iron_skin_potion_uses",
            "gold_value": 800,
        },
        {
            "name": HIT_POTION_ITEM_NAME,
            "aliases": ("Potion of Hit", "Hit Potion", "Зелье меткости"),
            "category": "temporary_accuracy",
            "effect": {"kind": "accuracy", "multiplier": POTION_ACCURACY_HIT_MULTIPLIER},
            "duration": "temporary",
            "battle_equip": False,
            "gold_value": 250,
        },
        {
            "name": ACCURACY_POTION_ITEM_NAME,
            "aliases": (
                "Potion of Accuracy",
                "Accuracy Potion",
                "Potion of Precision",
                "Зелье точности",
            ),
            "category": "temporary_accuracy",
            "effect": {"kind": "accuracy", "multiplier": POTION_ACCURACY_ACCURACY_MULTIPLIER},
            "duration": "temporary",
            "battle_equip": False,
            "gold_value": 400,
        },
        {
            "name": LUCK_POTION_ITEM_NAME,
            "aliases": ("Potion of Luck", "Luck Potion", "Эликсир удачи"),
            "category": "permanent_accuracy",
            "effect": {"kind": "accuracy", "multiplier": POTION_ACCURACY_LUCK_MULTIPLIER},
            "duration": "permanent",
            "battle_equip": False,
            "permanent_counter_key": "campaign_luck_potion_uses",
            "gold_value": 800,
        },
        {
            "name": SWIFTNESS_POTION_ITEM_NAME,
            "aliases": ("Potion of Swiftness", "Swiftness Potion", "Зелье проворства"),
            "category": "temporary_initiative",
            "effect": {
                "kind": "initiative",
                "multiplier": POTION_INITIATIVE_SWIFTNESS_MULTIPLIER,
            },
            "duration": "temporary",
            "battle_equip": False,
            "gold_value": 250,
        },
        {
            "name": SPEED_POTION_ITEM_NAME,
            "aliases": ("Potion of Speed", "Speed Potion", "Зелье скорости"),
            "category": "temporary_initiative",
            "effect": {"kind": "initiative", "multiplier": POTION_INITIATIVE_SPEED_MULTIPLIER},
            "duration": "temporary",
            "battle_equip": False,
            "gold_value": 400,
        },
        {
            "name": HASTE_ELIXIR_ITEM_NAME,
            "aliases": (
                MERCURY_ELIXIR_ITEM_NAME,
                "Potion of Mercury",
                "Mercury Potion",
                "Elixir of Mercury",
                "Potion of Haste",
                "Haste Potion",
                "Эликсир Меркурия",
                "Эликсир быстроты",
            ),
            "category": "temporary_initiative",
            "effect": {"kind": "initiative", "multiplier": HASTE_ELIXIR_INITIATIVE_MULTIPLIER},
            "duration": "temporary",
            "battle_equip": False,
            "active_attr": "active_haste_elixir_positions",
            "gold_value": 600,
        },
        {
            "name": INITIATIVE_ELIXIR_ITEM_NAME,
            "aliases": (
                "Potion of Initiative",
                "Initiative Potion",
                "Эликсир инициативы",
            ),
            "category": "permanent_initiative",
            "effect": {
                "kind": "initiative",
                "multiplier": POTION_INITIATIVE_PERMANENT_MULTIPLIER,
            },
            "duration": "permanent",
            "battle_equip": False,
            "permanent_counter_key": "campaign_initiative_elixir_uses",
            "gold_value": 800,
        },
        {
            "name": VIGOR_POTION_ITEM_NAME,
            "aliases": ("Potion of Vigor", "Vigor Potion", "Зелье бодрости"),
            "category": "temporary_damage",
            "effect": {"kind": "damage", "multiplier": POTION_DAMAGE_VIGOR_MULTIPLIER},
            "duration": "temporary",
            "battle_equip": False,
            "gold_value": 250,
        },
        {
            "name": STRENGTH_POTION_CANONICAL_ITEM_NAME,
            "aliases": (
                STRENGTH_POTION_ITEM_NAME,
                MIGHT_POTION_ITEM_NAME,
                "Potion of Strength",
                "Strength Potion",
                "Potion of Might",
                "Might Potion",
                "Зелье силы",
                "Зелье мощи",
            ),
            "category": "temporary_damage",
            "effect": {"kind": "damage", "multiplier": STRENGTH_POTION_DAMAGE_MULTIPLIER},
            "duration": "temporary",
            "battle_equip": False,
            "active_attr": "active_strength_potion_positions",
            "gold_value": 450,
        },
        {
            "name": ENERGY_ELIXIR_ITEM_NAME,
            "aliases": (
                "Potion of Energy",
                "Energy Potion",
                "Elixir of Energy",
                "Эликсир энергии",
            ),
            "category": "temporary_damage",
            "effect": {"kind": "damage", "multiplier": ENERGY_ELIXIR_DAMAGE_MULTIPLIER},
            "duration": "temporary",
            "battle_equip": False,
            "active_attr": "active_energy_elixir_positions",
            "gold_value": 200,
        },
        {
            "name": TITAN_ELIXIR_ITEM_NAME,
            "aliases": (
                "Potion of Titan Strength",
                "Titan Strength Potion",
                "Titan Elixir",
                "Elixir of Titan Strength",
            ),
            "category": "permanent_damage",
            "effect": {"kind": "damage", "multiplier": TITAN_ELIXIR_DAMAGE_MULTIPLIER},
            "duration": "permanent",
            "battle_equip": False,
            "permanent_counter_key": "campaign_titan_elixir_uses",
            "gold_value": 800,
        },
        {
            "name": FIRE_WARD_ITEM_NAME,
            "aliases": (
                "Potion of Fire Ward",
                "Fire Ward Potion",
                "Fire Ward",
                "Elixir of Fire Protection",
            ),
            "category": "ward",
            "effect": {"kind": "ward", "resistance_type": "Fire"},
            "duration": "temporary",
            "battle_equip": False,
            "active_attr": "active_fire_ward_positions",
            "gold_value": 400,
        },
        {
            "name": WATER_WARD_ITEM_NAME,
            "aliases": (
                "Potion of Water Ward",
                "Water Ward Potion",
                "Water Ward",
                "Elixir of Water Protection",
            ),
            "category": "ward",
            "effect": {"kind": "ward", "resistance_type": "Water"},
            "duration": "temporary",
            "battle_equip": False,
            "active_attr": "active_water_ward_positions",
            "gold_value": 400,
        },
        {
            "name": EARTH_WARD_ITEM_NAME,
            "aliases": (
                "Potion of Earth Ward",
                "Earth Ward Potion",
                "Earth Ward",
                "Elixir of Earth Protection",
            ),
            "category": "ward",
            "effect": {"kind": "ward", "resistance_type": "Earth"},
            "duration": "temporary",
            "battle_equip": False,
            "active_attr": "active_earth_ward_positions",
            "gold_value": 400,
        },
        {
            "name": AIR_WARD_ITEM_NAME,
            "aliases": (
                "Potion of Air Ward",
                "Air Ward Potion",
                "Air Ward",
                "Elixir of Air Protection",
            ),
            "category": "ward",
            "effect": {"kind": "ward", "resistance_type": "Air"},
            "duration": "temporary",
            "battle_equip": False,
            "active_attr": "active_air_ward_positions",
            "gold_value": 400,
        },
    )
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
    STAFF_SPELL_ITEM_DEFINITIONS = (
        {"item_name": "Staff of Celerity", "spell_id": "emp_d2_s002"},
        {"item_name": "Staff of Necromancy", "spell_id": "und_d2_s001"},
        {"item_name": "Staff of Thunder", "spell_id": "emp_d2_s004"},
        {"item_name": "Staff of Holiness", "spell_id": "emp_d2_s007"},
        {"item_name": "Staff of Traveling", "spell_id": "emp_d2_s006"},
        {"item_name": "Staff of Tree Summoning", "spell_id": "elf_d2_s012"},
        {"item_name": "Spirit Staff", "spell_id": "emp_d2_s011"},
        {"item_name": "Staff of Dragon Mastering", "spell_id": "und_d2_s012"},
        {"item_name": "Staff of Clumsiness", "spell_id": "und_d2_s013"},
        {"item_name": "Staff of Protection", "spell_id": "emp_d2_s012"},
        {"item_name": "Highfather Staff", "spell_id": "emp_d2_s019"},
        {"item_name": "Staff of Demonology", "spell_id": "lod_d2_s017"},
        {"item_name": "Staff of Earth Elemental Control", "spell_id": "emp_d2_s016"},
        {"item_name": "Staff of Ice Spirits", "spell_id": "mcl_d2_s019"},
    )
    STAFF_SPELL_ITEM_NAMES = frozenset(
        str(definition.get("item_name", "") or "")
        for definition in STAFF_SPELL_ITEM_DEFINITIONS
        if str(definition.get("item_name", "") or "")
    )
    STAFF_SPELL_ITEM_GOLD_VALUES = {
        str(definition.get("item_name", "") or ""): 1200
        for definition in STAFF_SPELL_ITEM_DEFINITIONS
        if str(definition.get("item_name", "") or "")
    }
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
        BANNER_OF_PROTECTION_ITEM_NAME: 1000,
        BANNER_OF_STRIKING_ITEM_NAME: 1000,
        BANNER_OF_SPEED_ITEM_NAME: 1000,
        BANNER_OF_STRENGTH_ITEM_NAME: 1000,
        BANNER_OF_RESISTANCE_ITEM_NAME: 3000,
        BANNER_OF_BATTLE_ITEM_NAME: 3000,
        BANNER_OF_CELERITY_ITEM_NAME: 3000,
        BANNER_OF_MIGHT_ITEM_NAME: 3000,
        BANNER_OF_FORTITUDE_ITEM_NAME: 5000,
        BANNER_OF_WAR_ITEM_NAME: 5000,
        **BOOK_ITEM_GOLD_VALUES,
        "Vampire Orb": 800,
        "Orb of Life": 800,
        "Lich Orb": 800,
        "Zombie Talisman": 4000,
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
        **STAFF_SPELL_ITEM_GOLD_VALUES,
        **BOOT_ITEM_GOLD_VALUES,
        **ARTIFACT_ITEM_GOLD_VALUES,
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



__all__ = [name for name in globals() if not name.startswith("__")]
