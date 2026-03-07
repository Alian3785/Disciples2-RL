# ============================================================
# КАСТОМНАЯ СРЕДА Gymnasium «RED vs BLUE»
# Версия: action space = 8 (6 атак по врагам pos1..pos6 + защита + ожидание)
# + big-юниты: Dead dragon (3-6), Asterot (7-10), Gargoyle (9-12)
# + Чекпоинты каждые 1,000,000 шагов
# + Тест с логами и улучшенной визуализацией (большие карточки)
# ---
# Изменение: RED (кроме Mage/Dead dragon) выбирают цель с минимальным HP.
# + Action Masking (sb3_contrib: MaskablePPO + ActionMasker)
# ============================================================

try:
    import gymnasium as gym
except Exception:
    # noqa
    import gymnasium as gym

import os
import random
import numpy as np
import math
from copy import deepcopy
from operator import itemgetter
from typing import Dict, List, Optional, Tuple
from gymnasium import spaces
from data_dicts_compact_lines import DATA as UNIT_DATA

# Импорт действий юнитов (предполагается, что эти файлы лежат рядом)
from occultmaster_actions import handle_occultmaster_action
from lyf_actions import handle_lyf_action
from laclaan_actions import handle_laclaan_action

SAVE_UNITS_ARRAYS_FLAG = int(os.environ.get("SAVE_UNITS_ARRAYS_FLAG", "1"))

# --- словари для кодирования в наблюдении ---
TYPE_LIST = [
    "Archer",
    "Mage",
    "Wolf Lord",
    "Teurg",
    "Witch",
    "Warrior",
    "Centaur Savage",
    "Doppelganger",
    "Demon",
    "Death",
    "Wight",
    "Sentry",
    "Watcher",
    "Lord",
    "Bone Lord",
    "Dregazul",
    "Dead dragon",
    "Gumtic",
    "Ismir son",
    "Ghost",
    "Shadow",
    "Succub",
    "Betrezen",
    "Uter",
    "Uter Demon",
    "Tiamat",
    "Baroness",
    "Incub",
    "Abyss Devil",
    "Cliric",
    "Profit",
    "Sundancer",
    "Sylfid",
    "Travnitsa",
    "Deva roshi",
    "Patriach",
    "Novice",
    "Alchemist",
    "Dwarfdruid",
    "Arhidruid",
    "Hermit",
    "Vampire",
    "Highvampire",
    "Summoner",
    "Occultmaster",
    "Lyf",
    "Laclaan",
    "Spider",
    "Drulliaan",
    "Elfarcher",
    "Shamanka",
    "Aleman",
]
ATTACK_TYPES = [
    "Weapon",
    "Earth",
    "Fire",
    "Water",
    "Poison",
    "Death",
    "Mind",
    "Life",
    "Air",
]  # one-hot(9)

FEATURES_PER_UNIT = 8 + len(TYPE_LIST) + 5 * len(ATTACK_TYPES) + 14 + 3
OBSERVATION_SIZE = FEATURES_PER_UNIT * 12


def _one_hot(value: str, vocab: List[str]) -> List[float]:
    return [1.0 if value == v else 0.0 for v in vocab]


def _multi_hot(values: List[str], vocab: List[str]) -> List[float]:
    s = set(values or [])
    return [1.0 if v in s else 0.0 for v in vocab]


def _build_transforms_by_name(unit_data: List[Dict]) -> Dict[str, List[str]]:
    transforms_by_name: Dict[str, List[str]] = {}
    for entry in unit_data:
        if not isinstance(entry, dict):
            continue
        name_raw = entry.get("кто")
        if not name_raw:
            continue
        name = str(name_raw).strip()
        if not name:
            continue
        raw_transforms = entry.get("превращается в", [])
        if isinstance(raw_transforms, (list, tuple)):
            transforms = [str(v) for v in raw_transforms if v is not None]
        elif raw_transforms:
            transforms = [str(raw_transforms)]
        else:
            transforms = []
        transforms_by_name[name] = transforms
    return transforms_by_name


TRANSFORMS_BY_NAME = _build_transforms_by_name(UNIT_DATA)


def _apply_transformations_to_unit(unit: Dict) -> None:
    name = str(unit.get("name", "") or "").strip()
    transforms = TRANSFORMS_BY_NAME.get(name)
    if transforms is None:
        unit.setdefault("превращается в", [])
    else:
        unit["превращается в"] = list(transforms)


def _apply_transformations_to_units(units: List[Dict]) -> None:
    for unit in units:
        _apply_transformations_to_unit(unit)


def _to_int_or_default(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _build_levels_by_name(unit_data: List[Dict]) -> Dict[str, int]:
    levels: Dict[str, int] = {}
    for entry in unit_data:
        if not isinstance(entry, dict):
            continue
        # Prefer canonical Russian keys and keep legacy mojibake fallback.
        entry_name = entry.get("\u043a\u0442\u043e")
        if entry_name is None:
            entry_name = entry.get("РєС‚Рѕ")
        if entry_name is None:
            continue
        name = str(entry_name).strip()
        if not name:
            continue
        level_raw = entry.get("\u0443\u0440\u043e\u0432\u0435\u043d\u044c", entry.get("Level", 0))
        levels[name] = _to_int_or_default(level_raw, default=0)
    return levels


LEVEL_BY_NAME = _build_levels_by_name(UNIT_DATA)


def _resolve_unit_level(unit: Dict) -> int:
    if "Level" in unit:
        return _to_int_or_default(unit.get("Level", 0), default=0)
    unit_name = str(unit.get("name", "") or "").strip()
    if not unit_name:
        return 0
    return _to_int_or_default(LEVEL_BY_NAME.get(unit_name, 0), default=0)


def _build_next_level_exp_by_name(unit_data: List[Dict]) -> Dict[str, int]:
    next_level_by_name: Dict[str, int] = {}
    for entry in unit_data:
        if not isinstance(entry, dict):
            continue
        entry_name = entry.get("\u043a\u0442\u043e")
        if entry_name is None:
            entry_name = entry.get("Р С”РЎвЂљР С•")
        if entry_name is None:
            continue
        name = str(entry_name).strip()
        if not name:
            continue
        next_level_raw = entry.get(
            "\u0441\u043b\u0435\u0434\u0443\u0440\u043e\u0432\u0435\u043d\u044c",
            entry.get("next_level_exp", 0),
        )
        next_level_by_name[name] = _to_int_or_default(next_level_raw, default=0)
    return next_level_by_name


NEXT_LEVEL_EXP_BY_NAME = _build_next_level_exp_by_name(UNIT_DATA)


def _resolve_unit_next_level_exp(unit: Dict) -> int:
    if "next_level_exp" in unit:
        return _to_int_or_default(unit.get("next_level_exp", 0), default=0)
    if "\u0441\u043b\u0435\u0434\u0443\u0440\u043e\u0432\u0435\u043d\u044c" in unit:
        return _to_int_or_default(unit.get("\u0441\u043b\u0435\u0434\u0443\u0440\u043e\u0432\u0435\u043d\u044c", 0), default=0)
    unit_name = str(unit.get("name", "") or "").strip()
    if not unit_name:
        return 0
    return _to_int_or_default(NEXT_LEVEL_EXP_BY_NAME.get(unit_name, 0), default=0)


def _apply_hero_levelup_bonuses(unit: Dict) -> None:
    base_damage = float(unit.get("original_damage", unit.get("damage", 0)) or 0)
    new_damage = max(0, _to_int_or_default(base_damage * 1.1, default=0))
    unit["damage"] = new_damage
    unit["original_damage"] = new_damage

    current_health = float(unit.get("health", unit.get("hp", 0)) or 0)
    max_health = float(unit.get("max_health", unit.get("maxhp", 0)) or 0)
    unit["health"] = max(0, _to_int_or_default(current_health * 1.1, default=0))
    unit["max_health"] = max(0, _to_int_or_default(max_health * 1.1, default=0))
    if "hp" in unit:
        unit["hp"] = unit["health"]
    if "maxhp" in unit:
        unit["maxhp"] = unit["max_health"]

    current_accuracy = _to_int_or_default(unit.get("accuracy", 0), default=0)
    unit["accuracy"] = max(0, min(100, current_accuracy + 1))

    current_exp_kill = float(unit.get("exp_kill", 0) or 0)
    unit["exp_kill"] = max(0, _to_int_or_default(current_exp_kill * 1.1, default=0))

UNITS_RED = [{'name': 'Скелет рыцарь',
  'initiative': 50,
  'initiative_base': 50,
  'team': 'red',
  'position': 1,
  'stand': 'ahead',
  'unit_type': 'Warrior',
  'damage': 100,
  'damage_secondary': 0,
  'health': 270,
  'max_health': 270,
  'armor': 0,
  'accuracy': 80,
  'accuracy_secondary': 0,
  'immunity': ['death'],
  'resistance': [],
  'attack_type_primary': 'Weapon',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
  'basestats': []},
 {'name': 'Рыцарь смерти',
  'initiative': 50,
  'initiative_base': 50,
  'team': 'red',
  'position': 2,
  'stand': 'ahead',
  'unit_type': 'Warrior',
  'damage': 50,
  'damage_secondary': 0,
  'health': 150,
  'max_health': 150,
  'armor': 0,
  'accuracy': 80,
  'accuracy_secondary': 0,
  'immunity': ['death'],
  'resistance': [],
  'attack_type_primary': 'Weapon',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
  'basestats': []},
 {'name': 'Змей ужаса',
  'initiative': 35,
  'initiative_base': 35,
  'team': 'red',
  'position': 3,
  'stand': 'ahead',
  'unit_type': 'Dead dragon',
  'damage': 65,
  'damage_secondary': 20,
  'health': 450,
  'max_health': 450,
  'armor': 0,
  'accuracy': 80,
  'accuracy_secondary': 40,
  'immunity': ['death'],
  'resistance': [],
  'attack_type_primary': 'death',
  'attack_type_secondary': 'death',
  'big': True,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
  'basestats': []},
 {'name': 'Архилич',
  'initiative': 40,
  'initiative_base': 40,
  'team': 'red',
  'position': 4,
  'stand': 'behind',
  'unit_type': 'Mage',
  'damage': 90,
  'damage_secondary': 0,
  'health': 170,
  'max_health': 170,
  'armor': 0,
  'accuracy': 80,
  'accuracy_secondary': 0,
  'immunity': ['death'],
  'resistance': [],
  'attack_type_primary': 'death',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
  'basestats': []},
 {'name': 'Смерть',
  'initiative': 40,
  'initiative_base': 40,
  'team': 'red',
  'position': 5,
  'stand': 'behind',
  'unit_type': 'Death',
  'damage': 100,
  'damage_secondary': 20,
  'health': 125,
  'max_health': 125,
  'armor': 0,
  'accuracy': 80,
  'accuracy_secondary': 50,
  'immunity': ['death', 'Weapon'],
  'resistance': [],
  'attack_type_primary': 'death',
  'attack_type_secondary': 'death',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
  'basestats': []},
 {'name': 'пусто',
  'initiative': 0,
  'initiative_base': 0,
  'team': 'red',
  'position': 6,
  'stand': 'behind',
  'unit_type': 'Archer',
  'damage': 0,
  'damage_secondary': 0,
  'health': 0,
  'max_health': 0,
  'armor': 0,
  'accuracy': 0,
  'accuracy_secondary': 0,
  'immunity': [],
  'resistance': [],
  'attack_type_primary': 'Weapon',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
  'basestats': []}]

UNITS_BLUE = [{'name': 'Одержимый',
  'initiative': 50,
  'initiative_base': 50,
  'team': 'blue',
  'position': 7,
  'stand': 'ahead',
  'unit_type': 'Warrior',
  'damage': 25,
  'damage_secondary': 0,
  'health': 120,
  'max_health': 120,
  'armor': 0,
  'accuracy': 80,
  'accuracy_secondary': 0,
  'immunity': [],
  'resistance': [],
  'attack_type_primary': 'Weapon',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
 'basestats': [],
 'exp_kill': 25,
'exp_required': 95,
'exp_current': 0,
'turns_into': ['Берсерк']},
 {'name': 'Герцог',
  'initiative': 50,
  'initiative_base': 50,
  'team': 'blue',
  'position': 8,
  'stand': 'ahead',
  'unit_type': 'Warrior',
  'damage': 50,
  'damage_secondary': 0,
  'health': 150,
  'max_health': 150,
  'armor': 0,
  'accuracy': 80,
  'accuracy_secondary': 0,
  'immunity': [],
  'resistance': [],
  'attack_type_primary': 'Weapon',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
 'basestats': [],
 'exp_kill': 60,
'exp_required': 150,
'exp_current': 0,
'turns_into': []},
 {'name': 'Одержимый',
  'initiative': 50,
  'initiative_base': 50,
  'team': 'blue',
  'position': 9,
  'stand': 'ahead',
  'unit_type': 'Warrior',
  'damage': 25,
  'damage_secondary': 0,
  'health': 120,
  'max_health': 120,
  'armor': 0,
  'accuracy': 80,
  'accuracy_secondary': 0,
  'immunity': [],
  'resistance': [],
  'attack_type_primary': 'Weapon',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
 'basestats': [],
 'exp_kill': 25,
'exp_required': 95,
'exp_current': 0,
'turns_into': ['Берсерк']},
 {'name': 'пусто',
  'initiative': 0,
  'initiative_base': 0,
  'team': 'blue',
  'position': 10,
  'stand': 'behind',
  'unit_type': 'Archer',
  'damage': 0,
  'damage_secondary': 0,
  'health': 0,
  'max_health': 0,
  'armor': 0,
  'accuracy': 0,
  'accuracy_secondary': 0,
  'immunity': [],
  'resistance': [],
  'attack_type_primary': 'Weapon',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
 'basestats': [],
 'exp_kill': 0,
'exp_required': 0,
'exp_current': 0,
'turns_into': []},
 {'name': 'Сектант',
  'initiative': 40,
  'initiative_base': 40,
  'team': 'blue',
  'position': 11,
  'stand': 'behind',
  'unit_type': 'Mage',
  'damage': 15,
  'damage_secondary': 0,
  'health': 45,
  'max_health': 45,
  'armor': 0,
  'accuracy': 80,
  'accuracy_secondary': 0,
  'immunity': [],
  'resistance': [],
  'attack_type_primary': 'Fire',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
 'basestats': [],
 'exp_kill': 20,
'exp_required': 70,
'exp_current': 0,
'turns_into': ['Колдун', 'Ведьма']},
 {'name': 'пусто',
  'initiative': 0,
  'initiative_base': 0,
  'team': 'blue',
  'position': 12,
  'stand': 'behind',
  'unit_type': 'Archer',
  'damage': 0,
  'damage_secondary': 0,
  'health': 0,
  'max_health': 0,
  'armor': 0,
  'accuracy': 0,
  'accuracy_secondary': 0,
  'immunity': [],
  'resistance': [],
  'attack_type_primary': 'Weapon',
  'attack_type_secondary': '',
  'big': False,
  'paralyzed': 0,
  'long_paralyzed': 0,
  'running_away': 0,
  'transformed': 0,
 'basestats': [],
 'exp_kill': 0,
'exp_required': 0,
'exp_current': 0,
'turns_into': []}]

_apply_transformations_to_units(UNITS_RED)
_apply_transformations_to_units(UNITS_BLUE)


def _apply_team_traits(
    units: List[Dict], enemy_units: Optional[List[Dict]] = None
) -> None:
    if any(unit.get("unit_type") == "Sundancer" for unit in units):
        for unit in units:
            unit["Firedefence"] = 0

    if any(unit.get("unit_type") == "Sylfid" for unit in units):
        for unit in units:
            unit["Airdefence"] = 0

    if any(
        unit.get("unit_type") in ("Travnitsa", "Novice", "Alchemist") for unit in units
    ):
        for unit in units:
            unit["powerup"] = 0

    if any(
        unit.get("unit_type") in ("Dwarfdruid", "Arhidruid", "Travnitsa", "Novice")
        for unit in units
    ):
        for unit in units:
            unit["original_damage"] = unit.get("damage", 0)

    if any(unit.get("unit_type") == "Deva roshi" for unit in units):
        for unit in units:
            unit["Firedefence"] = 0
            unit["Airdefence"] = 0
            unit["Waterdefence"] = 0
            unit["Earthdefence"] = 0

    if any(unit.get("unit_type") == "Alchemist" for unit in units):
        for unit in units:
            unit["bonusturn"] = 0

    if enemy_units is not None and any(
        unit.get("unit_type") == "Tiamat" for unit in units
    ):
        for enemy in enemy_units:
            enemy["teamated"] = 0

    if enemy_units is not None and any(
        unit.get("unit_type") == "Hermit" for unit in units
    ):
        for enemy in enemy_units:
            enemy["hermited"] = 0  # флаг: уже замедлён «hermit»


_apply_team_traits(UNITS_RED, UNITS_BLUE)
_apply_team_traits(UNITS_BLUE, UNITS_RED)

for unit in UNITS_RED + UNITS_BLUE:
    unit["Level"] = _resolve_unit_level(unit)
    unit["next_level_exp"] = _resolve_unit_next_level_exp(unit)
    unit.setdefault("defense", 0)
    unit.setdefault("waited", 0)


# Диапазоны позиций
# Диапазоны позиций
RED_POSITIONS: List[int] = list(range(1, 7))
BLUE_POSITIONS: List[int] = list(range(7, 13))

# Доступные цели для действия агента — теперь 12 позиций (pos1..pos12)
# Логика боя и маски остаётся прежней: невалидные позиции будут заблокированы маской.
TARGET_POSITIONS: List[int] = (
    RED_POSITIONS + BLUE_POSITIONS
)  # 12 действий на выбор позиции
DEFEND_ACTION_INDEX: int = len(TARGET_POSITIONS)
WAIT_ACTION_INDEX: int = len(TARGET_POSITIONS) + 1
RUN_AWAY_ACTION_INDEX: int = len(TARGET_POSITIONS) + 2
DEFEND_ARMOR_BONUS: int = 50
WAIT_INIT_DIVISOR: int = 10
TOTAL_AGENT_ACTIONS: int = len(TARGET_POSITIONS) + 3  # 15

RED_FRONT_POSITIONS: List[int] = [1, 2, 3]
RED_BACK_POSITIONS: List[int] = [4, 5, 6]
BLUE_FRONT_POSITIONS: List[int] = [7, 8, 9]
BLUE_BACK_POSITIONS: List[int] = [10, 11, 12]

# Константы для нормирования наблюдений
MAX_HP = max([u["health"] for u in (UNITS_RED + UNITS_BLUE)])  # 1020
MAX_HP = max(MAX_HP, 525)  # учитываем призыв Occultmaster (драконов до 525 HP)
MAX_INIT = max([u["initiative_base"] for u in (UNITS_RED + UNITS_BLUE)])
INIT_RANDOM_BONUS = 9  # initiative jitter applied at the start of each round
MAX_INIT_DYNAMIC = MAX_INIT + INIT_RANDOM_BONUS
# В базовых массивах стартовых армий урон равен 0, поэтому дополнительно учитываем
# потенциальный максимум с учётом призываемых существ (драконы Occultmaster/Lacflaan).
MAX_DMG = max([u["damage"] for u in (UNITS_RED + UNITS_BLUE)])
MAX_DMG = max(MAX_DMG, 125)  # чёрный дракон Laclaan наносит 125 урона
# Include arena summons (e.g., Lyf's spider) that are not present in the base rosters.
MAX_DMG2 = max(
    max([u["damage_secondary"] for u in (UNITS_RED + UNITS_BLUE)]),
    35,  # Гигантский чёрный паук
)
# Травница может баффать союзника на +75% урона, поэтому расширяем допустимый максимум.
TRAVNITSA_MULTIPLIER = 2
MAX_DMG_BUFFED = int(np.ceil(MAX_DMG * TRAVNITSA_MULTIPLIER))

POISON_TURNS = 6  # max duration; actual value rolled per application
BURN_DAMAGE = 10  # базовый фолбэк; Владыка задаёт per-tick из своего урон2
BURN_TURNS = 6
URAN_DAMAGE = 10  # базовый фолбэк; Сын Измира задаёт per-tick из своего урон2
URAN_TURNS = 6
WIGHT_DECAY_PERCENT = 0.33  # share of stats removed by wight effect

# Observation normalization profile.
INIT_BASE_NORM_MAX = 80.0
INIT_CUR_NORM_MAX = 89.0
DMG_NORM_MAX = 250.0
DMG2_NORM_MAX = 35.0
ARMOR_NORM_MAX = 100.0
ACC_NORM_MAX = 100.0
BONUS_NORM_MAX = 10.0
EXP_KILL_NORM_MAX = 120.0
EXP_REQ_NORM_MAX = 700.0
POS_NORM_MAX = 11.0
HP_FALLBACK_MAX = 800.0


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _safe_div(x: float, d: float) -> float:
    d = float(d)
    if d <= 1e-12:
        return 0.0
    return float(x) / d


def _to01_bool(v) -> float:
    try:
        return 1.0 if float(v) > 0.0 else 0.0
    except (TypeError, ValueError):
        return 1.0 if bool(v) else 0.0


# ------------------------ Кастомная среда ------------------------
class BattleEnv(gym.Env):
    """
    Observation (720): 12 слотов ? 60 признаков (добавлены типы Betrezen, Uter и Uter Demon).
    Action space: Discrete(8) — 6 атак по врагам (pos1..pos6) + защита + ожидание.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_step: float = 0.0,
        penalty_invalid_target: float = -0.10,
        penalty_unreachable_warrior: float = -0.20,
        log_enabled: bool = False,
    ):
        super().__init__()
        self.reward_win = float(reward_win)
        self.reward_loss = float(reward_loss)
        self.reward_step = float(reward_step)

        self.penalty_invalid_target = float(penalty_invalid_target)
        self.penalty_unreachable_warrior = float(penalty_unreachable_warrior)

        self.log_enabled = bool(log_enabled)
        self._pretty_events: List[str] = []
        self.last_battle_exp: float = 0.0
        self.last_levelups: List[str] = []

        # Action space = 6 целей + защита + ожидание
        self.action_space = spaces.Discrete(TOTAL_AGENT_ACTIONS)  # 15

        # Observation space contract: each component is expected in [0, 1].
        low_unit = np.zeros(FEATURES_PER_UNIT, dtype=np.float32)
        high_unit = np.ones(FEATURES_PER_UNIT, dtype=np.float32)
        low = np.tile(low_unit, 12)
        high = np.tile(high_unit, 12)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Состояние боя
        self.rng = random.Random()  # без seed > случайность между эпизодами
        self.combined: List[Dict] = []
        self.round_no: int = 1
        self.winner: Optional[str] = None
        self.current_blue_attacker_pos: Optional[int] = None
        self.blue_attacks_left: int = 0
        self._lord_applied_burn: Dict[int, bool] = {}
        self._spider_applied_poison: Dict[int, bool] = {}
        self._dregazul_applied_poison: Dict[int, bool] = {}
        self._ismir_applied_uran: Dict[int, bool] = {}
        self.survived_inits: List[Dict] = []
        self.step_count: int = 0

    # ------------------ [MASKING] Маска допустимых действий ------------------
    def compute_action_mask(self) -> np.ndarray:
        """
        Возвращает маску допустимых действий для ТЕКУЩЕГО ходящего BLUE атакера.
        True — действие разрешено (можно выбирать), False — запрещено.

        Для ближников пересекаем с досягаемыми целями (_warrior_allowed_targets).
        Для AoE (Mage, Dead dragon, Gumtic, Uter Demon, Tiamat, Teurg) — разрешаем любые живые цели.
        Для остальных (включая стрелков) — разрешаем любые живые цели.
        """
        n_targets = len(TARGET_POSITIONS)
        mask = np.ones(TOTAL_AGENT_ACTIONS, dtype=bool)  # по умолчанию всё разрешено
        mask_targets = mask[:n_targets]
        mask[DEFEND_ACTION_INDEX] = True
        mask[WAIT_ACTION_INDEX] = True
        mask[RUN_AWAY_ACTION_INDEX] = True

        # Если не ход BLUE или нет текущего атакера — не ограничиваем
        if self.current_blue_attacker_pos is None:
            return mask

        attacker = self._unit_by_position(self.current_blue_attacker_pos)
        if attacker is None or not self._alive(attacker):
            return mask
        if attacker.get("waited", 0) == 1:
            mask[WAIT_ACTION_INDEX] = False

        demon_second_strike = (
            attacker.get("unit_type") in ("Demon", "Elfarcher")
            and self.blue_attacks_left == 1
        )
        if demon_second_strike:
            mask[DEFEND_ACTION_INDEX] = False
            mask[WAIT_ACTION_INDEX] = False
            mask[RUN_AWAY_ACTION_INDEX] = False

        def is_alive_red(pos: int) -> bool:
            u = self._unit_by_position(pos)
            return (u is not None) and self._alive(u) and (u["team"] == "red")

        atype = attacker.get("unit_type")

        if atype == "Doppelganger":
            forbidden_names = {
                "Ашган",
                "Ашкаэль",
                "Видар",
                "Мизраэль",
                "Иллюмиэлль",
                "Драллиаан",
                "Лаклаан",
            }
            attacker_pos = attacker.get("position")
            for i, pos in enumerate(TARGET_POSITIONS):
                if pos == attacker_pos:
                    mask_targets[i] = False
                    continue

                target_unit = self._unit_by_position(pos)
                if target_unit is None or not self._alive(target_unit):
                    mask_targets[i] = False
                    continue

                if target_unit.get("big", False):
                    mask_targets[i] = False
                    continue

                if target_unit.get("name") in forbidden_names:
                    mask_targets[i] = False
                    continue

                mask_targets[i] = True

            return mask

        if attacker.get("team") == "blue" and atype == "summoner":
            # посчитаем занятые синие позиции живыми союзниками (big учитывает пару)
            occupied: set[int] = set()
            for u in self.combined:
                if u.get("team") != "blue" or not self._alive(u):
                    continue
                p = int(u.get("position", 0) or 0)
                occupied.add(p)
                if u.get("big", False):
                    # big занимает партнёра в своём столбце (±3)
                    if p in BLUE_FRONT_POSITIONS:
                        occupied.add(p + 3)
                    elif p in BLUE_BACK_POSITIONS:
                        occupied.add(p - 3)

            empty_blue = {p for p in BLUE_POSITIONS if p not in occupied}

            # Маска по первым 6 действиям: разрешаем только те,
            # чья противоположная синяя клетка пустая
            for i, red_pos in enumerate(TARGET_POSITIONS):
                opp_blue = self._opposite_position(red_pos, "blue")
                mask_targets[i] = opp_blue in empty_blue

        support_types = (
            "Cliric",
            "Profit",
            "Travnitsa",
            "Sylfid",
            "Sundancer",
            "Deva roshi",
            "Patriach",
            "Novice",
            "Alchemist",
            "Dwarfdruid",
            "Arhidruid",
        )

        # Ближники: только досягаемые цели
        if atype in (
            "Warrior",
            "Centaur Savage",
            "Demon",
            "Lord",
            "Bone Lord",
            "Dregazul",
            "Ismir son",
            "Uter",
            "Abyss Devil",
            "Aleman",
            "Spider",
        ):
            allowed_reach = set(self._warrior_allowed_targets(attacker))
            for i, pos in enumerate(TARGET_POSITIONS):
                mask_targets[i] = is_alive_red(pos) and (pos in allowed_reach)

            # Страховка от пустой маски:
            if not mask_targets.any():
                live_idxs = [
                    i for i, pos in enumerate(TARGET_POSITIONS) if is_alive_red(pos)
                ]
                if live_idxs:
                    mask_targets[live_idxs] = True

        # AoE: формальная цель, запрещаем только мёртвые слоты
        elif atype in (
            "Mage",
            "Dead dragon",
            "Wolf Lord",
            "Gumtic",
            "Uter Demon",
            "Tiamat",
            "Teurg",
            "Hermit",
            "Vampire",
            "Highvampire",
            "Shamanka",
        ):
            attacker_pos = attacker.get("position")
            for i, pos in enumerate(TARGET_POSITIONS):
                mask_targets[i] = is_alive_red(pos) or (
                    atype == "Wolf Lord" and pos == attacker_pos
                )
            if not mask_targets.any():
                mask_targets[:] = True
        else:
            if atype in support_types:
                for i, pos in enumerate(TARGET_POSITIONS):
                    opposite_pos = self._opposite_position(pos, attacker["team"])
                    if opposite_pos is None:
                        mask_targets[i] = False
                        continue
                    recipient = self._unit_by_position(opposite_pos)
                    if atype == "Patriach":
                        mask_targets[i] = (
                            recipient is not None
                            and recipient.get("team") == attacker.get("team")
                            and recipient.get("name") != "пусто"
                            and not bool(recipient.get("Summoned"))
                        )
                    else:
                        mask_targets[i] = (
                            recipient is not None
                            and self._alive(recipient)
                            and recipient.get("team") == attacker.get("team")
                        )
                if not mask_targets.any():
                    mask_targets[:] = False
            else:
                # Прочие (включая стрелков): любые живые цели.
                for i, pos in enumerate(TARGET_POSITIONS):
                    mask_targets[i] = is_alive_red(pos)
                if not mask_targets.any():
                    mask_targets[:] = True

        return mask

    # ------------------ Вспомогательные методы ------------------

    def seed(self, seed=None):
        # Метод оставлен для совместимости, но мы им не пользуемся, чтобы не фиксировать случайность
        self.rng = random.Random(seed)

    def _alive(self, u) -> bool:
        return u["health"] > 0

    def _team_alive(self, team: str) -> bool:
        return any(self._alive(u) and u["team"] == team for u in self.combined)

    def _has_transform_targets(self) -> bool:
        forbidden = {
            "Ашган",
            "Ашкаэль",
            "Видар",
            "Мизраэль",
            "Иллюмиэлль",
            "Драллиаан",
            "Лаклаан",
        }
        for unit in self.combined:
            if not self._alive(unit):
                continue
            if unit.get("big", False):
                continue
            if unit.get("name") in forbidden:
                continue
            return True
        return False

    def _restore_default_doppelgangers(self) -> None:
        defaults = {
            "initiative": 80,
            "initiative_base": 80,
            "unit_type": "Doppelganger",
            "damage": 30,
            "damage_secondary": 0,
            "max_health": 120,
            "armor": 0,
            "accuracy": 100,
            "accuracy_secondary": 0,
            "immunity": [],
            "resistance": [],
            "attack_type_primary": "Mind",
            "attack_type_secondary": "",
        }

        for unit in self.combined:
            if not self._alive(unit):
                continue
            if unit.get("name") != "Двойник":
                continue

            prev_max = float(unit.get("max_health", 0) or 0)
            current_health = float(unit.get("health", 0) or 0)
            ratio = (
                0.0 if prev_max <= 0 else max(0.0, min(1.0, current_health / prev_max))
            )

            for key, value in defaults.items():
                unit[key] = list(value) if isinstance(value, list) else value

            unit["health"] = int(round(defaults["max_health"] * ratio))

    def _unit_by_position(self, pos: int):
        return next((u for u in self.combined if u["position"] == pos), None)

    def _live_positions_of(self, team: str):
        return [
            u["position"] for u in self.combined if u["team"] == team and self._alive(u)
        ]

    def _is_position_behind_big(self, pos: int) -> bool:
        if pos in RED_BACK_POSITIONS or pos in BLUE_BACK_POSITIONS:
            front_pos = pos - 3
            partner = self._unit_by_position(front_pos)
            if (
                partner is not None
                and self._alive(partner)
                and partner.get("big", False)
            ):
                return True
        return False

    def _log(self, s: str):
        if self.log_enabled:
            self._pretty_events.append(s)

    def pop_pretty_events(self) -> List[str]:
        out = self._pretty_events[:]
        self._pretty_events.clear()
        return out

    def _col_of(self, pos: int) -> int:
        return (pos - 1) % 3

    def _enemy_rows(self, team: str):
        if team == "red":
            return [7, 8, 9], [10, 11, 12]
        else:
            return [1, 2, 3], [4, 5, 6]

    def _warrior_near_far_cols(self, col: int):
        if col == 0:
            return [0, 1], [2]
        if col == 1:
            return [0, 1, 2], []
        return [1, 2], [0]

    def _warrior_allowed_targets(self, attacker: Dict) -> List[int]:
        assert attacker.get("unit_type") in (
            "Warrior",
            "Centaur Savage",
            "Demon",
            "Lord",
            "Bone Lord",
            "Dregazul",
            "Ismir son",
            "Uter",
            "Abyss Devil",
            "Aleman",
            "Spider",
        )
        if attacker["stand"] == "behind":
            if any(
                self._alive(u)
                and u["team"] == attacker["team"]
                and u.get("unit_type")
                in (
                    "Warrior",
                    "Centaur Savage",
                    "Demon",
                    "Lord",
                    "Bone Lord",
                    "Dregazul",
                    "Ismir son",
                    "Uter",
                    "Abyss Devil",
                    "Aleman",
                    "Spider",
                )
                and u["stand"] == "ahead"
                for u in self.combined
            ):
                return []
        ahead, behind = self._enemy_rows(attacker["team"])
        col = self._col_of(attacker["position"])
        near_cols, far_cols = self._warrior_near_far_cols(col)

        def alive(pos):
            uu = self._unit_by_position(pos)
            return uu is not None and self._alive(uu)

        near = [ahead[c] for c in near_cols if alive(ahead[c])]
        if near:
            return near
        far = [ahead[c] for c in far_cols if alive(ahead[c])]
        if far:
            return far

        near2 = [behind[c] for c in near_cols if alive(behind[c])]
        if near2:
            return near2
        far2 = [behind[c] for c in far_cols if alive(behind[c])]
        return far2

    def _travnitsa_auto_target(self, healer: Dict) -> Optional[int]:
        allies = [
            u
            for u in self.combined
            if u["team"] == healer["team"] and self._alive(u) and u is not healer
        ]
        if not allies:
            return None
        if healer.get("team") == "red":

            def dmg_val(unit: Dict) -> int:
                return int(unit.get("damage", 0) or 0)

            max_damage = max(dmg_val(u) for u in allies)
            top_candidates = [u for u in allies if dmg_val(u) == max_damage]
            return self.rng.choice(top_candidates)["position"]
        return self.rng.choice(allies)["position"]

    def _summoner_auto_target(self, summoner: Dict) -> Optional[int]:
        def slot_empty(pos: int) -> bool:
            unit = self._unit_by_position(pos)
            return unit is None or not self._alive(unit)

        front_candidates = [pos for pos in RED_FRONT_POSITIONS if slot_empty(pos)]
        if front_candidates:
            chosen_pos = self.rng.choice(front_candidates)
        else:
            back_candidates = [
                pos
                for pos in RED_BACK_POSITIONS
                if slot_empty(pos) and not self._is_position_behind_big(pos)
            ]
            if not back_candidates:
                return None
            chosen_pos = self.rng.choice(back_candidates)

        if chosen_pos in RED_FRONT_POSITIONS:
            col = RED_FRONT_POSITIONS.index(chosen_pos)
            return BLUE_FRONT_POSITIONS[col]
        col = RED_BACK_POSITIONS.index(chosen_pos)
        return BLUE_BACK_POSITIONS[col]

    def _alchemist_auto_target(self, supporter: Dict) -> Optional[int]:
        allies = [
            u
            for u in self.combined
            if u["team"] == supporter["team"]
            and self._alive(u)
            and u is not supporter
            and u.get("unit_type") != "Alchemist"
            and u.get("running_away", 0) != 1
        ]
        if not allies:
            return None
        needs_help = [
            u
            for u in allies
            if int(u.get("initiative", 0) or 0) < int(u.get("initiative_base", 0) or 0)
        ]
        if supporter.get("team") == "red":
            if not needs_help:
                return None

            def dmg_value(unit: Dict) -> int:
                return int(unit.get("damage", 0) or 0)

            positive = [u for u in needs_help if dmg_value(u) > 0]
            if not positive:
                return None
            max_dmg = max(dmg_value(u) for u in positive)
            best = [u for u in positive if dmg_value(u) == max_dmg]
            return self.rng.choice(best)["position"]

        pool = needs_help if needs_help else allies
        chosen = self.rng.choice(pool)
        return chosen["position"]

    def _apply_alchemist_support(self, alchemist: Dict, recipient: Dict) -> bool:
        if recipient is None or not self._alive(recipient):
            return False
        if recipient is alchemist:
            return False
        if (
            recipient.get("unit_type") == "Alchemist"
            or recipient.get("running_away", 0) == 1
        ):
            self._log(
                f"?? Дополнительный ход без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не может получить бонус от алхимика."
            )
            return False

        recipient.setdefault("bonusturn", 0)
        base_ini = int(recipient.get("initiative_base", 0) or 0)
        recipient["initiative"] = base_ini
        recipient["bonusturn"] += 1
        self._log(
            f"?? Дополнительный ход: {alchemist['team'].upper()} {alchemist['name']}#{alchemist['position']} восстанавливает инициативу "
            f"{recipient['team'].upper()} {recipient['name']}#{recipient['position']} до {base_ini} и увеличивает bonusturn до {recipient['bonusturn']}."
        )
        return True

    def _opposite_position(self, pos: int, team: str) -> Optional[int]:
        col = self._col_of(pos)
        if team == "blue":
            if pos in RED_FRONT_POSITIONS:
                return BLUE_FRONT_POSITIONS[col]
            if pos in RED_BACK_POSITIONS:
                return BLUE_BACK_POSITIONS[col]
        else:
            if pos in BLUE_FRONT_POSITIONS:
                return RED_FRONT_POSITIONS[col]
            if pos in BLUE_BACK_POSITIONS:
                return RED_BACK_POSITIONS[col]
        return None

    def _cliric_auto_target(self, healer: Dict) -> Optional[int]:
        allies = [
            u
            for u in self.combined
            if u["team"] == healer["team"]
            and self._alive(u)
            and u["health"] < u.get("max_health", 0)
        ]
        if not allies:
            return None
        min_hp = min(u["health"] for u in allies)
        candidates = [u for u in allies if u["health"] == min_hp]
        return self.rng.choice(candidates)["position"]

    def _apply_cliric_heal(self, healer: Dict, recipient: Optional[Dict]) -> bool:
        if recipient is None or not self._alive(recipient):
            return False

        heal_amount = int(healer.get("damage", 0) or 0)
        if heal_amount <= 0:
            self._log(
                f"? Лечение без силы: {healer['team'].upper()} {healer['name']}#{healer['position']} имеет недостаточный урон (damage={heal_amount})."
            )
            return False

        if healer.get("unit_type") == "Sundancer":
            if "resistance" not in recipient:
                recipient["resistance"] = []
            if "Fire" not in recipient["resistance"]:
                recipient["resistance"].append("Fire")
            recipient.setdefault("resilience_used_types", [])
            if "Fire" in recipient["resilience_used_types"]:
                recipient["resilience_used_types"].remove("Fire")
            recipient["Firedefence"] = 1
            self._log(
                f"? Защита от огня: {healer['team'].upper()} {healer['name']}#{healer['position']} даёт огненную защиту "
                f"для {recipient['team'].upper()} {recipient['name']}#{recipient['position']}"
            )

        elif healer.get("unit_type") == "Sylfid":
            if "resistance" not in recipient:
                recipient["resistance"] = []
            if "Air" not in recipient["resistance"]:
                recipient["resistance"].append("Air")
            recipient.setdefault("resilience_used_types", [])
            if "Air" in recipient["resilience_used_types"]:
                recipient["resilience_used_types"].remove("Air")
            recipient["Airdefence"] = 1
            self._log(
                f"? Защита от воздуха: {healer['team'].upper()} {healer['name']}#{healer['position']} даёт воздушную защиту "
                f"для {recipient['team'].upper()} {recipient['name']}#{recipient['position']}"
            )

        elif healer.get("unit_type") == "Deva roshi":
            if "resistance" not in recipient:
                recipient["resistance"] = []
            recipient.setdefault("resilience_used_types", [])
            element_map = (
                ("Fire", "Firedefence"),
                ("Air", "Airdefence"),
                ("Water", "Waterdefence"),
                ("Earth", "Earthdefence"),
            )
            for element, field in element_map:
                if element not in recipient["resistance"]:
                    recipient["resistance"].append(element)
                if element in recipient["resilience_used_types"]:
                    recipient["resilience_used_types"].remove(element)
                recipient[field] = 1
            self._log(
                f"? Четверная защита: {healer['team'].upper()} {healer['name']}#{healer['position']} усиливает защиту "
                f"{recipient['team'].upper()} {recipient['name']}#{recipient['position']} от огня, воздуха, воды и земли."
            )

        before = recipient["health"]
        max_hp = recipient.get("max_health", before)
        if before >= max_hp:
            return False

        healed = min(heal_amount, max_hp - before)
        if healed <= 0:
            return False

        recipient["health"] += healed
        self._log(
            f"? Лечение: {healer['team'].upper()} {healer['name']}#{healer['position']} восстанавливает {healed} HP "
            f"для {recipient['team'].upper()} {recipient['name']}#{recipient['position']} ({before}>{recipient['health']})."
        )

        return True

    def _patriach_targetable_allies(self, healer: Dict) -> List[Dict]:
        team = healer.get("team")
        return [
            unit
            for unit in self.combined
            if unit.get("team") == team
            and unit.get("name") != "пусто"
            and not bool(unit.get("Summoned"))
        ]

    def _patriach_auto_target(self, healer: Dict) -> Optional[int]:
        allies = self._patriach_targetable_allies(healer)
        dead_allies = [
            unit
            for unit in allies
            if (unit.get("health", 0) or 0) <= 0
            and int(unit.get("initiative", 0) or 0) <= 0
        ]
        if dead_allies:
            return self.rng.choice(dead_allies)["position"]

        wounded = [
            unit
            for unit in allies
            if self._alive(unit) and unit.get("health", 0) < unit.get("max_health", 0)
        ]
        if not wounded:
            return None

        min_hp = min(u["health"] for u in wounded)
        candidates = [u for u in wounded if u["health"] == min_hp]
        return self.rng.choice(candidates)["position"]

    def _apply_patriach_support(self, healer: Dict, recipient: Optional[Dict]) -> str:
        if recipient is None:
            return "invalid"
        if recipient.get("team") != healer.get("team"):
            return "invalid"
        if recipient.get("name") == "пусто":
            return "invalid"
        if bool(recipient.get("Summoned")):
            return "invalid"

        if self._alive(recipient):
            healed = self._apply_cliric_heal(healer, recipient)
            return "heal_success" if healed else "heal_no_effect"

        health = float(recipient.get("health", 0) or 0)
        initiative = int(recipient.get("initiative", 0) or 0)
        if health > 0 or initiative > 0:
            return "invalid"

        max_hp = int(recipient.get("max_health", 0) or 0)
        if max_hp <= 0:
            return "revive_failed"

        restored = max(1, int(round(max_hp * 0.5)))
        recipient["health"] = restored
        recipient["initiative"] = 0
        recipient["paralyzed"] = 0
        recipient["long_paralyzed"] = 0
        recipient["running_away"] = 0
        recipient.setdefault("poison_turns_left", 0)
        recipient.setdefault("burn_turns_left", 0)
        recipient.setdefault("uran_turns_left", 0)
        recipient.setdefault("poison_damage_per_tick", 0)
        recipient.setdefault("burn_damage_per_tick", 0)
        recipient.setdefault("uran_damage_per_tick", 0)
        recipient["poison_turns_left"] = 0
        recipient["burn_turns_left"] = 0
        recipient["uran_turns_left"] = 0
        recipient["poison_damage_per_tick"] = 0
        recipient["burn_damage_per_tick"] = 0
        recipient["uran_damage_per_tick"] = 0

        self._cleanse_negative_effects(recipient)
        self._log(
            f"? Воскрешение Патриарха: {healer['team'].upper()} {healer['name']}#{healer['position']} "
            f"возвращает {recipient['team'].upper()} {recipient['name']}#{recipient['position']} к жизни ({restored}/{max_hp})."
        )
        return "revive_success"

    def _cleanse_negative_effects(self, unit: Optional[Dict]) -> bool:
        """Снимает доты/контроль/статовые дебаффы, если юнит жив."""
        if unit is None or not self._alive(unit):
            return False

        cleared = False
        burn_cleared = False  # следим, что именно поджог снят
        poison_cleared = False  # отметка для яда
        uran_cleared = False  # отметка для «воды»

        # Сбрасываем периодический урон (яд/поджог/«вода») вместе с их счётчиками.
        for turns_key, dmg_key in (
            ("poison_turns_left", "poison_damage_per_tick"),
            ("burn_turns_left", "burn_damage_per_tick"),
            ("uran_turns_left", "uran_damage_per_tick"),
        ):
            had_effect = unit.get(turns_key, 0) > 0 or unit.get(dmg_key, 0) > 0
            if had_effect:
                unit[turns_key] = 0
                unit[dmg_key] = 0
                cleared = True
                if turns_key == "burn_turns_left":
                    burn_cleared = True
                elif turns_key == "poison_turns_left":
                    poison_cleared = True
                elif turns_key == "uran_turns_left":
                    uran_cleared = True

        # Снимаем паралич, долгий паралич и состояние «бегства».
        for flag in ("paralyzed", "long_paralyzed"):
            if unit.get(flag, 0):
                unit[flag] = 0
                cleared = True

        # Hermit-замедление: возвращаем базовую инициативу и убираем флаг.
        if unit.get("hermited", 0):
            base_ini = int(unit.get("initiative_base", 0) or 0)
            unit["initiative_base"] = base_ini * 2 if base_ini > 0 else base_ini
            unit["initiative"] = max(
                int(unit.get("initiative", 0) or 0), unit["initiative_base"]
            )
            unit["hermited"] = 0
            cleared = True

        # Если юнит превращён ведьмой/суккубом — откатываем сохранённые характеристики.
        if unit.get("transformed", 0):
            basestats = unit.get("basestats")
            pos_key = unit.get("position")
            snapshots: List[Dict] = []
            if isinstance(basestats, dict):
                snapshots = basestats.get(pos_key) or []
            elif isinstance(basestats, list):
                snapshots = basestats
            snapshot = snapshots[0] if snapshots else None
            if isinstance(snapshot, dict):
                for key in (
                    "accuracy",
                    "accuracy_secondary",
                    "damage",
                    "damage_secondary",
                    "attack_type_primary",
                    "attack_type_secondary",
                    "initiative",
                    "initiative_base",
                    "unit_type",
                    "resistance",
                    "resilience_used_types",
                    "transformed",
                ):
                    if key in snapshot:
                        unit[key] = deepcopy(snapshot[key])
                if isinstance(basestats, dict):
                    basestats.pop(pos_key, None)
                elif isinstance(basestats, list):
                    snapshots.pop(0)
                cleared = True
            else:
                unit["transformed"] = 0

        # Если убрали поджог – сбрасываем флажок Владыки, чтобы он мог снова наложить burn.
        if burn_cleared:
            self._release_lord_burn_from_unit(unit)
        if poison_cleared:
            self._spider_applied_poison.clear()
            self._dregazul_applied_poison.clear()
        if uran_cleared:
            self._ismir_applied_uran.clear()

        if cleared:
            self._log(
                f"?? Очищение: {unit['team'].upper()} {unit['name']}#{unit['position']} избавляется от негативных эффектов."
            )
        return cleared

    def _release_lord_burn_from_unit(self, unit: Optional[Dict]) -> None:
        """Освобождает слот Владыки, если burn был снят с юнита."""
        if not unit:
            return
        source_pos = unit.pop("burn_source_lord_pos", None)
        if source_pos is not None:
            self._lord_applied_burn.pop(source_pos, None)

    def _kill_linked_summons(self, summoner: Optional[Dict]) -> None:
        """Kills all alive summons that are linked to the provided summoner."""
        if not summoner:
            return

        summoner_type = (summoner.get("unit_type") or "").lower()
        if summoner_type not in {"summoner", "occultmaster", "laclaan", "lyf"}:
            return

        summoner_team = summoner.get("team")
        summoner_pos = summoner.get("position")
        if summoner_team is None or summoner_pos is None:
            return

        for unit in self.combined:
            if unit is summoner:
                continue
            if unit.get("team") != summoner_team:
                continue
            if not self._alive(unit):
                continue
            if unit.get("Summoned") != summoner_pos:
                continue

            before_hp = int(unit.get("health", 0) or 0)
            unit["health"] = 0
            unit["initiative"] = 0
            self._release_lord_burn_from_unit(unit)
            unit_team = (unit.get("team") or "").upper()
            unit_name = unit.get("name") or unit.get("unit_type") or "unit"
            unit_pos = unit.get("position")
            summoner_name = summoner.get("name") or summoner.get("unit_type") or "unit"
            self._log(
                f"? {unit_team} {unit_name}#{unit_pos} гибнет вместе с призывателем "
                f"{summoner_name}#{summoner_pos} ({before_hp}>0)."
            )

    def _share_highvampire_leech(
        self, vampire: Dict, hp_pool: int
    ) -> Tuple[int, bool, bool]:
        leftover = int(hp_pool or 0)
        if leftover <= 0:
            return 0, False, False

        allies = [
            ally
            for ally in self.combined
            if ally is not vampire
            and ally.get("team") == vampire.get("team")
            and self._alive(ally)
            and int(ally.get("health", 0) or 0)
            < int(ally.get("max_health", ally.get("health", 0) or 0))
        ]
        if not allies:
            return leftover, False, False

        had_targets = True
        shared_any = False

        while leftover > 0:
            wounded = [
                ally
                for ally in allies
                if self._alive(ally)
                and int(ally.get("health", 0) or 0)
                < int(ally.get("max_health", ally.get("health", 0) or 0))
            ]
            if not wounded:
                break

            share = max(1, leftover // len(wounded))
            distributed_this_round = 0

            for ally in wounded:
                if leftover <= 0:
                    break
                max_hp = int(ally.get("max_health", ally.get("health", 0) or 0))
                before_hp = int(ally.get("health", 0) or 0)
                missing = max_hp - before_hp
                if missing <= 0:
                    continue

                heal = min(share, missing, leftover)
                if heal <= 0:
                    continue

                ally["health"] = before_hp + heal
                leftover -= heal
                distributed_this_round += heal
                shared_any = True
                self._log(
                    f"?? Делёж крови: {vampire['team'].upper()} {vampire['name']}#{vampire['position']} "
                    f"передаёт {heal} HP для {ally['team'].upper()} {ally['name']}#{ally['position']} ({before_hp}>{ally['health']})."
                )

            if distributed_this_round == 0:
                for ally in wounded:
                    if leftover <= 0:
                        break
                    max_hp = int(ally.get("max_health", ally.get("health", 0) or 0))
                    before_hp = int(ally.get("health", 0) or 0)
                    if before_hp >= max_hp:
                        continue
                    ally["health"] = before_hp + 1
                    leftover -= 1
                    distributed_this_round += 1
                    shared_any = True
                    self._log(
                        f"?? Делёж крови: {vampire['team'].upper()} {vampire['name']}#{vampire['position']} "
                        f"добавляет 1 HP для {ally['team'].upper()} {ally['name']}#{ally['position']} ({before_hp}>{ally['health']})."
                    )
                if distributed_this_round == 0:
                    break

        return leftover, shared_any, had_targets

    def _apply_vampiric_heal(
        self, attacker: Dict, leeched_total: int, share_leftover: bool = False
    ) -> None:
        leeched = int(leeched_total or 0)
        if leeched <= 0:
            return

        atk_team = (attacker.get("team") or "").upper()
        attacker_name = attacker.get("name") or attacker.get("unit_type") or "unit"
        attacker_pos = attacker.get("position")

        before_hp = int(attacker.get("health", 0) or 0)
        max_hp = int(attacker.get("max_health", before_hp) or before_hp)
        healed = 0
        if before_hp < max_hp:
            healed = min(leeched, max_hp - before_hp)
            if healed > 0:
                attacker["health"] = before_hp + healed
                self._log(
                    f"?? Кровопийство: {atk_team} {attacker_name}#{attacker_pos} восстанавливает {healed} HP "
                    f"({before_hp}>{attacker['health']}) из {leeched} похищенных."
                )

        leftover = leeched - healed
        if share_leftover and leftover > 0:
            leftover, shared_any, had_targets = self._share_highvampire_leech(
                attacker, leftover
            )
            if not had_targets:
                self._log(
                    f"?? Делёж крови: {atk_team} {attacker_name}#{attacker_pos} — раненых союзников нет."
                )
            elif leftover > 0 and shared_any:
                self._log(
                    f"?? Делёж крови: {atk_team} {attacker_name}#{attacker_pos} не смог распределить {leftover} HP — союзники уже восстановлены."
                )
            elif leftover > 0:
                self._log(
                    f"?? Делёж крови: {atk_team} {attacker_name}#{attacker_pos} не нашёл, кому передать {leftover} HP."
                )

    def _apply_cached_poison(
        self, attacker: Dict, victim: Dict, applied_cache: Dict[int, bool]
    ) -> None:
        if not self._alive(victim):
            return

        acc2 = float(attacker.get("accuracy_secondary", 0) or 0)

        pos = attacker.get("position")
        if applied_cache.get(pos, False):
            return

        roll = self._roll_status(acc2)
        atk_team = (attacker.get("team") or "").upper()
        self._log(
            f"?? Шанс отравления {int(acc2)}% - "
            + ("успех" if roll else "неудача")
            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
        )
        if not roll:
            return

        if self._is_immune_status(attacker, victim):
            self._log(
                f"?? Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — яд не действует на "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return

        status_tag = attacker.get("attack_type_secondary", "") or attacker.get(
            "attack_type_primary", ""
        )
        if self._resilience_blocks(attacker, victim, custom_tag=status_tag):
            return

        turns = self.rng.randint(1, POISON_TURNS)
        victim["poison_turns_left"] = turns
        victim["poison_damage_per_tick"] = int(attacker.get("damage_secondary", 0) or 0)
        applied_cache[pos] = True
        self._log(
            f"? {atk_team} {attacker['name']}#{pos} накладывает яд "
            f"({victim['poison_damage_per_tick']} урона/ход) на "
            f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
        )

    def _apply_travnitsa_buff(self, healer: Dict, recipient: Optional[Dict]) -> bool:
        if recipient is None or not self._alive(recipient) or recipient is healer:
            return False

        current_damage = recipient.get("damage", 0)
        stored_original = recipient.get("original_damage", 0)

        if stored_original <= 0:
            stored_original = current_damage
            if stored_original <= 0:
                return False

        if healer.get("unit_type") == "Travnitsa":
            buffed_damage = int(round(stored_original * 1.25))
            recipient["powerup"] = 1
        elif healer.get("unit_type") == "Novice":
            buffed_damage = int(round(stored_original * 1.5))
            recipient["powerup"] = 1
        elif healer.get("unit_type") == "Alchemist":
            return self._apply_alchemist_support(healer, recipient)
        elif healer.get("unit_type") == "Dwarfdruid":
            buffed_damage = int(round(stored_original * 1.75))
        elif healer.get("unit_type") == "Arhidruid":
            buffed_damage = int(round(stored_original * 2))

        recipient["damage"] = buffed_damage
        recipient["powerup"] = 1

        self._log(
            f"?? Усиление травницы: {healer['team'].upper()} {healer['name']}#{healer['position']} увеличивает урон "
            f"{recipient['team'].upper()} {recipient['name']}#{recipient['position']} до {buffed_damage}."
        )
        return True

    # --- НОВОЕ: выбор позиции с минимальным HP из списка позиций ---
    def _pick_lowest_hp(self, positions: List[int]) -> Optional[int]:
        """
        Вернуть позицию цели с наименьшим текущим HP среди заданных позиций.
        Берём только живых; при равенстве HP — случайная из минимальных.
        """
        candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is not None and self._alive(u):
                candidates.append((u["health"], p))
        if not candidates:
            return None
        min_hp = min(hp for hp, _ in candidates)
        best = [p for hp, p in candidates if hp == min_hp]
        return self.rng.choice(best)

    def _filter_non_immune_targets(self, attacker: Dict, positions: List[int]) -> List[int]:
        """Возвращает список позиций, цели на которых НЕ имеют иммунитета к основной атаке атакующего."""
        attack_type = attacker.get("attack_type_primary", "Weapon")
        attack_type_lower = str(attack_type).lower()
        valid_positions = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is None or not self._alive(u):
                continue
            immunities = [str(val).lower() for val in (u.get("immunity") or [])]
            if attack_type_lower not in immunities:
                valid_positions.append(p)
        return valid_positions

    def _pick_highest_hp(self, positions: List[int]) -> Optional[int]:
        candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is not None and self._alive(u):
                candidates.append((u["health"], p))
        if not candidates:
            return None
        max_hp = max(hp for hp, _ in candidates)
        best = [p for hp, p in candidates if hp == max_hp]
        return self.rng.choice(best)

    def _pick_highest_damage_non_paralyzed(self, positions: List[int]) -> Optional[int]:
        """
        Выбирает цель с наибольшим показателем damage, на которой нет паралича (обычного или долгого).
        Если таких целей несколько — выбирается случайная из них.
        Если подходящих целей нет, возвращается None.
        """
        candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is None or not self._alive(u):
                continue
            # Проверка на наличие паралича
            if u.get("paralyzed", 0) == 1 or u.get("long_paralyzed", 0) == 1:
                continue
            candidates.append((u.get("damage", 0), p))
        
        if not candidates:
            return None
            
        max_dmg = max(dmg for dmg, _ in candidates)
        best = [p for dmg, p in candidates if dmg == max_dmg]
        return self.rng.choice(best)

    def _pick_highest_hp_non_big(
        self, exclude_unit: Optional[Dict] = None
    ) -> Optional[int]:
        candidates: List[Tuple[int, int]] = []
        for unit in self.combined:
            if not self._alive(unit):
                continue
            if bool(unit.get("big", False)):
                continue
            if exclude_unit is not None and unit is exclude_unit:
                continue
            candidates.append((int(unit.get("health", 0) or 0), unit["position"]))
        if not candidates:
            return None
        max_hp = max(hp for hp, _ in candidates)
        best = [pos for hp, pos in candidates if hp == max_hp]
        return self.rng.choice(best)

    def _pick_weapon_immune_or_highest_hp(self, positions: List[int]) -> Optional[int]:
        immune_candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is None or not self._alive(u):
                continue
            immunities = [str(val).lower() for val in (u.get("immunity") or [])]
            if "weapon" in immunities:
                immune_candidates.append(p)
        if immune_candidates:
            return self.rng.choice(immune_candidates)
        return self._pick_highest_hp(positions)

    # ----------- эффекты начала хода (яд/поджог) -----------
    def _apply_start_of_turn_effects(self, unit: Dict) -> bool:
        # DEFEND: временная броня спадает только в начале следующего хода этого юнита.
        if unit.get("defense", 0) == 1:
            current_armor = int(unit.get("armor", 0) or 0)
            reduced_armor = max(0, current_armor - DEFEND_ARMOR_BONUS)
            unit["armor"] = reduced_armor
            unit["defense"] = 0
            self._log(
                f"DEFENCE EXPIRE: {unit['team'].upper()} {unit['name']}#{unit['position']} теряет бонус брони "
                f"-{DEFEND_ARMOR_BONUS} ({current_armor}->{unit['armor']})."
            )
        if unit.get("name") == "Двойник" and self._alive(unit):
            has_targets = self._has_transform_targets()
            if unit.get("unit_type") == "Doppelganger" and not has_targets:
                unit["unit_type"] = "Warrior"
                self._log("?? Двойник: целей нет — тип переключён на Warrior.")
            elif unit.get("unit_type") == "Warrior" and has_targets:
                unit["unit_type"] = "Doppelganger"
                self._log(
                    "?? Двойник: цели появились — тип переключён на Doppelganger."
                )

        if unit.get("unit_type") == "Sundancer":
            for ally in self.combined:
                if ally.get("team") != unit.get("team"):
                    continue
                if ally.get("Firedefence", 0) == 1:
                    ally["Firedefence"] = 0
                    if "resistance" in ally:
                        ally["resistance"] = [
                            res for res in ally.get("resistance", []) if res != "Fire"
                        ]
                        if not ally["resistance"]:
                            ally["resistance"] = []
                    if (
                        "resilience_used_types" in ally
                        and ally["resilience_used_types"]
                    ):
                        ally["resilience_used_types"] = [
                            res
                            for res in ally["resilience_used_types"]
                            if res != "Fire"
                        ]
                    self._log(
                        f"? Солнечная защита рассеялась: {ally['team'].upper()} {ally['name']}#{ally['position']} лишается огненной стойкости."
                    )

        if unit.get("unit_type") == "Sylfid":
            for ally in self.combined:
                if ally.get("team") != unit.get("team"):
                    continue
                if ally.get("Airdefence", 0) == 1:
                    ally["Airdefence"] = 0
                    if "resistance" in ally:
                        ally["resistance"] = [
                            res for res in ally.get("resistance", []) if res != "Air"
                        ]
                        if not ally["resistance"]:
                            ally["resistance"] = []
                    if (
                        "resilience_used_types" in ally
                        and ally["resilience_used_types"]
                    ):
                        ally["resilience_used_types"] = [
                            res for res in ally["resilience_used_types"] if res != "Air"
                        ]
                    self._log(
                        f"? Воздушная защита рассеялась: {ally['team'].upper()} {ally['name']}#{ally['position']} лишается воздушной стойкости."
                    )

        if unit.get("unit_type") == "Deva roshi":
            for ally in self.combined:
                if ally.get("team") != unit.get("team"):
                    continue
                reset_map = (
                    ("Firedefence", "Fire"),
                    ("Airdefence", "Air"),
                    ("Waterdefence", "Water"),
                    ("Earthdefence", "Earth"),
                )
                cleared = []
                for field, element in reset_map:
                    if ally.get(field, 0) == 1:
                        ally[field] = 0
                        cleared.append(element)
                if not cleared:
                    continue
                if "resistance" in ally:
                    ally["resistance"] = [
                        res for res in ally.get("resistance", []) if res not in cleared
                    ]
                    if not ally["resistance"]:
                        ally["resistance"] = []
                if "resilience_used_types" in ally and ally["resilience_used_types"]:
                    ally["resilience_used_types"] = [
                        res
                        for res in ally["resilience_used_types"]
                        if res not in cleared
                    ]
                self._log(
                    f"? Защита Дэвы рассеялась: {ally['team'].upper()} {ally['name']}#{ally['position']} теряет стойкость против {', '.join(cleared)}."
                )

        if (
            unit.get("hermited", 0) == 1
            and self._alive(unit)
            and unit.get("initiative") > 10
        ):
            if self.rng.random() < 0.5:
                old_base = int(unit.get("initiative_base", 0) or 0)
                new_base = old_base * 2
                unit["hermited"] = 0
                unit["initiative_base"] = new_base
                current_ini = int(unit.get("initiative", 0) or 0)
                unit["initiative"] = max(current_ini, new_base)
                self._log(
                    f"?? Hermit slow fades: {unit['team'].upper()} {unit['name']}#{unit['position']} восстанавливает инициативу "
                    f"{old_base}->{new_base}."
                )

        # ЯД
        if (
            unit.get("poison_turns_left", 0) > 0
            and self._alive(unit)
            and unit.get("initiative") > 10
        ):
            immunities_lower = [str(i).lower() for i in (unit.get("immunity") or [])]
            if "poison" in immunities_lower:
                unit["poison_turns_left"] = 0
                unit["poison_damage_per_tick"] = 0
                self._log(
                    f"?? Иммунитет к эффекту 'poison' — яд не действует на {unit['team'].upper()} {unit['name']}#{unit['position']}."
                )
            else:
                tick = int(unit.get("poison_damage_per_tick", 0) or 0)
                if tick > 0:
                    before = unit["health"]
                    unit["health"] -= tick
                    unit["poison_turns_left"] -= 1
                    after = unit["health"]
                    self._log(
                        f"? Яд поражает {unit['team'].upper()} {unit['name']}#{unit['position']}: "
                        f"{tick} ({before}>{max(0, after)}); осталось ходов: {unit['poison_turns_left']}"
                    )
                    if unit["health"] <= 0:
                        unit["initiative"] = 0
                        self._kill_linked_summons(unit)
                        self._log(
                            f"? {unit['team'].upper()} {unit['name']}#{unit['position']} погибает от яда."
                        )
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["poison_turns_left"] -= 1
                if unit["poison_turns_left"] <= 0:
                    unit["poison_damage_per_tick"] = 0
                    # яд снят — пауки могут снова применить эффект
                    self._spider_applied_poison.clear()
                    self._dregazul_applied_poison.clear()

        # ПОДЖОГ — теперь урон за тик берётся из burn_damage_per_tick (для Владыки = его урон2)
        if (
            unit.get("burn_turns_left", 0) > 0
            and self._alive(unit)
            and unit.get("initiative") > 10
        ):
            # зеркальная проверка иммунитета к Fire (как для яда)
            immunities_lower = [str(i).lower() for i in (unit.get("immunity") or [])]
            if "fire" in immunities_lower:
                unit["burn_turns_left"] = 0
                unit["burn_damage_per_tick"] = 0
                self._release_lord_burn_from_unit(unit)
                self._log(
                    f"?? Иммунитет к эффекту 'Fire' — поджог не действует на {unit['team'].upper()} {unit['name']}#{unit['position']}."
                )
            else:
                tick = int(unit.get("burn_damage_per_tick", 0) or BURN_DAMAGE)
                if tick > 0:
                    before = unit["health"]
                    unit["health"] -= tick
                    unit["burn_turns_left"] -= 1
                    after = unit["health"]
                    self._log(
                        f"?? Поджог поражает {unit['team'].upper()} {unit['name']}#{unit['position']}: "
                        f"{tick} ({before}>{max(0, after)}); осталось ходов: {unit['burn_turns_left']}"
                    )
                    if unit["health"] <= 0:
                        unit["initiative"] = 0
                        self._kill_linked_summons(unit)
                        self._log(
                            f"? {unit['team'].upper()} {unit['name']}#{unit['position']} погибает от поджога."
                        )
                        self._check_victory_after_hit()
                        self._release_lord_burn_from_unit(unit)
                        return False
                else:
                    unit["burn_turns_left"] -= 1
                if unit["burn_turns_left"] <= 0:
                    unit["burn_damage_per_tick"] = 0
                    self._release_lord_burn_from_unit(unit)

        # Вода — наносит периодический урон (для Сына Измира = его урон2)
        if (
            unit.get("uran_turns_left", 0) > 0
            and self._alive(unit)
            and unit.get("initiative") > 10
        ):
            immunities_lower = [str(i).lower() for i in (unit.get("immunity") or [])]
            if "water" in immunities_lower:
                unit["uran_turns_left"] = 0
                unit["uran_damage_per_tick"] = 0
                self._log(
                    f"?? Иммунитет к эффекту 'Water' — вода не действует на {unit['team'].upper()} {unit['name']}#{unit['position']}."
                )
            else:
                tick = int(unit.get("uran_damage_per_tick", 0) or URAN_DAMAGE)
                if tick > 0:
                    before = unit["health"]
                    unit["health"] -= tick
                    unit["uran_turns_left"] -= 1
                    after = unit["health"]
                    self._log(
                        f"? Вода поражает {unit['team'].upper()} {unit['name']}#{unit['position']}: "
                        f"{tick} ({before}>{max(0, after)}); осталось ходов: {unit['uran_turns_left']}"
                    )
                    if unit["health"] <= 0:
                        unit["initiative"] = 0
                        self._kill_linked_summons(unit)
                        self._log(
                            f"? {unit['team'].upper()} {unit['name']}#{unit['position']} погибает от воды."
                        )
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["uran_turns_left"] -= 1
                if unit["uran_turns_left"] <= 0:
                    unit["uran_damage_per_tick"] = 0
                    # вода снята — сын Имира может навесить повторно
                    self._ismir_applied_uran.clear()

        # Runaway check (unit flees before acting if running_away == 1)
        if unit.get("running_away", 0) == 1 and self._alive(unit):
            if unit.get("paralyzed", 0) == 0 and unit.get("long_paralyzed", 0) == 0:
                self._log(
                    f"?? {unit['team'].upper()} {unit['name']}#{unit['position']} в панике покидает бой."
                )
                snapshot = deepcopy(unit)
                self.survived_inits.append(snapshot)
                for key in (
                    "health",
                    "max_health",
                    "initiative",
                    "damage",
                    "damage_secondary",
                    "accuracy",
                    "accuracy_secondary",
                    "armor",
                ):
                    unit[key] = 0
                for key in (
                    "poison_turns_left",
                    "poison_damage_per_tick",
                    "burn_turns_left",
                    "burn_damage_per_tick",
                    "uran_turns_left",
                    "uran_damage_per_tick",
                ):
                    unit[key] = 0
                unit["paralyzed"] = 0
                unit["long_paralyzed"] = 0
                unit["running_away"] = 0
                self._check_victory_after_hit()
                return False
            # если был паралич — снимаем флаг бегства, отложив на следующий ход
            else:
                self._log(
                    f"?? {unit['team'].upper()} {unit['name']}#{unit['position']} пока парализован и не может бежать."
                )

        basestats = unit.get("basestats")
        if unit.get("transformed", 0) == 1 and basestats:
            pos_key = unit.get("position")
            snapshots: List[Dict] = []
            if isinstance(basestats, dict):
                snapshots = basestats.get(pos_key) or []
            elif isinstance(basestats, list):
                snapshots = basestats

            if snapshots and self.rng.random() < 0.3 and unit.get("initiative") > 10:
                base_snapshot = snapshots[0]
                if isinstance(base_snapshot, dict):
                    unit["accuracy"] = base_snapshot.get(
                        "accuracy", unit.get("accuracy", 0)
                    )
                    unit["accuracy_secondary"] = base_snapshot.get(
                        "accuracy_secondary", unit.get("accuracy_secondary", 0)
                    )
                    unit["damage"] = base_snapshot.get("damage", unit.get("damage", 0))
                    unit["damage_secondary"] = base_snapshot.get(
                        "damage_secondary", unit.get("damage_secondary", 0)
                    )
                    unit["attack_type_primary"] = base_snapshot.get(
                        "attack_type_primary", unit.get("attack_type_primary", "Weapon")
                    )
                    unit["attack_type_secondary"] = base_snapshot.get(
                        "attack_type_secondary", unit.get("attack_type_secondary", "")
                    )
                    ini_val = base_snapshot.get("initiative", unit.get("initiative", 0))
                    unit["initiative_base"] = ini_val
                    unit["initiative"] = ini_val
                    unit["unit_type"] = base_snapshot.get(
                        "unit_type", unit.get("unit_type", "Warrior")
                    )
                    unit["resistance"] = list(base_snapshot.get("resistance", []))
                    unit["resilience_used_types"] = list(
                        base_snapshot.get("resilience_used_types", [])
                    )
                    unit["transformed"] = base_snapshot.get("transformed", 0)
                if isinstance(basestats, dict):
                    basestats.pop(pos_key, None)
                else:
                    try:
                        snapshots.pop(0)
                    except IndexError:
                        pass
                self._log(f"?? {unit['basestats']}")

        return True

    # ------------------ Боевая логика ------------------

    def _reset_state(self):
        self.combined = deepcopy(UNITS_RED + UNITS_BLUE)
        _apply_transformations_to_units(self.combined)
        for u in self.combined:
            base_ini = int(u.get("initiative_base", 0) or 0)
            u["initiative"] = base_ini
            if self._alive(u):
                u["initiative"] += self.rng.randint(0, 9)
            u.setdefault("attack_type_primary", "Weapon")
            u.setdefault("attack_type_secondary", "")
            u.setdefault("immunity", [])
            u.setdefault("resistance", [])
            u.setdefault("armor", 0)
            # Keep battle-start armor so campaign can restore temporary armor changes.
            u["base_armor"] = int(u.get("armor", 0) or 0)
            u.setdefault("accuracy", 100)
            u.setdefault("accuracy_secondary", 0)
            u.setdefault("damage_secondary", 0)
            u.setdefault("big", False)
            u.setdefault("Level", _resolve_unit_level(u))
            u.setdefault("next_level_exp", _resolve_unit_next_level_exp(u))
            u.setdefault("bonusturn", 0)
            u["defense"] = 0
            u["waited"] = 0
            u["poison_turns_left"] = 0
            u["poison_damage_per_tick"] = 0
            u["burn_turns_left"] = 0
            u["burn_damage_per_tick"] = 0  # per-tick для поджога
            u["burn_source_lord_pos"] = None
            u["uran_turns_left"] = 0
            u["uran_damage_per_tick"] = 0
            u["resilience_used_types"] = []
        self.round_no = 1
        self.winner = None
        self.current_blue_attacker_pos = None
        self.blue_attacks_left = 0
        self._lord_applied_burn = {}
        self._spider_applied_poison = {}
        self._dregazul_applied_poison = {}
        self._ismir_applied_uran = {}
        self.survived_inits = []
        self.step_count = 0
        self._log(f"Эпизод начат. Раунд {self.round_no}.")

    def _candidates(self):
        return [u for u in self.combined if self._alive(u) and u["initiative"] > 0]

    def _pop_next(self):
        cand = self._candidates()
        if not cand:
            return None
        self.rng.shuffle(cand)
        cand.sort(key=itemgetter("initiative"), reverse=True)
        return cand[0]

    def _end_round_restore(self):
        for u in self.combined:
            if self._alive(u):
                base_ini = int(u.get("initiative_base", 0) or 0)
                u["initiative"] = base_ini + self.rng.randint(0, 9)
            else:
                u["initiative"] = 0
            u["waited"] = 0
        self._log("Восстановление инициативы. Новый раунд.")

    # --- иммунитеты/броня/точность/стойкость ---
    def _is_immune_damage(self, attacker: Dict, victim: Dict) -> bool:
        atk1 = (attacker.get("attack_type_primary", "Weapon") or "").lower()
        immunities = [str(i).lower() for i in (victim.get("immunity") or [])]
        return atk1 in immunities

    def _is_immune_status(self, attacker: Dict, victim: Dict) -> bool:
        atk2 = (attacker.get("attack_type_secondary", "") or "").lower()
        if not atk2:
            return False
        immunities = [str(i).lower() for i in (victim.get("immunity") or [])]
        return atk2 in immunities

    def _resilience_blocks(
        self, attacker: Dict, victim: Dict, custom_tag: Optional[str] = None
    ) -> bool:
        res_list_raw = victim.get("resistance") or []
        if not res_list_raw:
            return False
        # Создаём словарь: lower -> original для сохранения оригинального регистра при записи
        res_map = {str(r).lower(): r for r in res_list_raw}
        used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
        available_lower = set(res_map.keys()) - used_lower
        if not available_lower:
            return False

        atk1 = attacker.get("attack_type_primary", "")

        attack_tags = []
        if custom_tag:
            attack_tags.append(custom_tag)
        elif atk1:
            attack_tags.append(atk1)

        for tag in attack_tags:
            tag_lower = (tag or "").lower()
            if tag_lower and tag_lower in available_lower:
                # Сохраняем оригинальное значение для consistency
                original_tag = res_map.get(tag_lower, tag)
                victim.setdefault("resilience_used_types", []).append(original_tag)
                self._log(
                    f"?? Стойкость — первый удар типа '{tag}' по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} поглощён."
                )
                return True
        return False

    def _reset_powerup(self, unit: Optional[Dict]) -> None:
        if unit is None:
            return
        if unit.get("powerup", 0) != 0:
            unit["powerup"] = 0
            if unit.get("original_damage", 0) > 0:
                unit["damage"] = unit["original_damage"]

    def _apply_damage_with_armor(self, base_dmg: float, victim: Dict) -> int:
        armor = float(victim.get("armor", 0) or 0)
        armor = min(armor, 90)  # Максимум 90% снижения урона
        factor = (1.0 - armor / 100.0) if armor > 0 else 1.0
        bonus = self.rng.randint(0, 5)
        boosted = base_dmg + bonus
        eff = boosted * max(0.0, factor)
        return int(round(eff))

    def _roll_hit(self, attacker: Dict) -> bool:
        acc = float(attacker.get("accuracy", 100) or 0)
        return self.rng.random() < (acc / 100.0)

    def _roll_status(self, chance_percent: float) -> bool:
        cp = max(0.0, min(100.0, float(chance_percent or 0.0)))
        return self.rng.random() < (cp / 100.0)

    def _apply_fear_effect(self, attacker: Dict, victim: Dict) -> bool:
        if victim.get("running_away", 0) == 1:
            return False
        atk_type = attacker.get("attack_type_primary", "")
        immun = set(victim.get("immunity") or [])
        resist = set(victim.get("resistance") or [])
        if atk_type and atk_type in immun:
            self._log(
                f"?? Страх Баронессы не действует на {victim['team'].upper()} {victim['name']}#{victim['position']} — тип атаки '{atk_type}' поглощён."
            )
            return False
        if atk_type and atk_type in resist:
            used = set(victim.get("resilience_used_types") or [])
            if atk_type not in used:
                victim.setdefault("resilience_used_types", []).append(atk_type)
                self._log(
                    f"?? Страх Баронессы не действует на {victim['team'].upper()} {victim['name']}#{victim['position']} — тип атаки '{atk_type}' поглощён."
                )
                return False
        victim["running_away"] = 1
        self._log(
            f"?? {attacker['team'].upper()} {attacker['name']}#{attacker['position']} вселяет страх в "
            f"{victim['team'].upper()} {victim['name']}#{victim['position']}: бегство!"
        )
        return True

    def _apply_paralysis_effect(self, attacker: Dict, victim: Dict) -> bool:
        if victim.get("paralyzed", 0):
            return False

        effect_type = attacker.get("attack_type_primary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        if effect_type_lower in immunities_lower:
            self._log(
                f"Иммунитет к '{effect_type}' — паралич не действует на "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
            return False

        res_lower = {str(r).lower() for r in (victim.get("resistance") or [])}
        if effect_type_lower in res_lower:
            used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
            if effect_type_lower not in used_lower:
                victim.setdefault("resilience_used_types", []).append(effect_type)
                self._log(
                    f"?? Стойкость — первый эффект типа '{effect_type}' по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} поглощён."
                )
                return False

        victim["paralyzed"] = 1
        self._log(
            f"Паралич: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
            f"лишает хода {victim['team'].upper()} {victim['name']}#{victim['position']}"
        )
        return True

    def _apply_witch_effect(self, attacker: Dict, victim: Dict) -> None:
        updated_entry = {
            "accuracy": victim["accuracy"],
            "accuracy_secondary": victim["accuracy_secondary"],
            "damage": victim["damage"],
            "damage_secondary": victim["damage_secondary"],
            "attack_type_primary": victim["attack_type_primary"],
            "attack_type_secondary": victim["attack_type_secondary"],
            "initiative": victim["initiative_base"],
            "unit_type": victim["unit_type"],
            "resistance": list(victim["resistance"]),
            "resilience_used_types": list(victim["resilience_used_types"]),
            "transformed": victim["transformed"],
        }
        pos_key = victim.get("position")
        basestats = victim.get("basestats")
        if not isinstance(basestats, dict):
            basestats = {}
            victim["basestats"] = basestats
        if pos_key not in basestats:
            basestats[pos_key] = [updated_entry]
        is_big = bool(victim.get("big", False))
        old_initiative = victim.get("initiative", 0)
        new_initiative = 50 if is_big else 30

        victim["accuracy"] = 80
        victim["accuracy_secondary"] = 0
        victim["damage"] = 30 if is_big else 20
        victim["damage_secondary"] = 0
        victim["attack_type_primary"] = "Weapon"
        victim["attack_type_secondary"] = ""
        victim["initiative_base"] = new_initiative
        if old_initiative == 0:
            victim["initiative"] = old_initiative
        else:
            victim["initiative"] = new_initiative
        victim["unit_type"] = "Warrior"
        victim["resistance"] = []
        victim["resilience_used_types"] = []
        victim["transformed"] = 1
        self._log(
            f"?? Ведьма меняет характеристики {victim['team'].upper()} {victim['name']}#{victim['position']}: "
            f"accuracy=80, damage={victim['damage']}, initiative={victim['initiative']}"
        )

    def _apply_wight_decay_effect(self, attacker: Dict, victim: Dict) -> None:
        dmg_before = int(victim.get("damage", 0) or 0)
        hp_before = int(victim.get("health", 0) or 0)
        max_before = int(victim.get("max_health", 0) or 0)

        dmg_reduction = int(round(dmg_before * WIGHT_DECAY_PERCENT))
        hp_reduction = int(round(hp_before * WIGHT_DECAY_PERCENT))
        max_reduction = int(round(max_before * WIGHT_DECAY_PERCENT))

        new_damage = max(0, dmg_before - dmg_reduction)
        new_max = max(0, max_before - max_reduction)
        new_hp = max(0, hp_before - hp_reduction)

        victim["damage"] = new_damage
        victim["max_health"] = new_max
        victim["health"] = min(new_hp, victim["max_health"])

        self._log(
            f"? Wight drain: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
            f"reduces stats of {victim['team'].upper()} {victim['name']}#{victim['position']}: "
            f"damage {dmg_before}->{victim['damage']}, health {hp_before}->{victim['health']}, "
            f"max_health {max_before}->{victim['max_health']}."
        )

    def _apply_long_paralysis_effect(self, attacker: Dict, victim: Dict) -> bool:
        if victim.get("long_paralyzed", 0):
            return False

        effect_type = attacker.get("attack_type_secondary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        if effect_type_lower in immunities_lower:
            self._log(
                f"Иммунитет к '{effect_type}' — долгий паралич не действует на "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
            return False

        res_map = {str(r).lower(): r for r in (victim.get("resistance") or [])}
        if effect_type_lower in res_map:
            used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
            if effect_type_lower not in used_lower:
                victim.setdefault("resilience_used_types", []).append(res_map[effect_type_lower])
                self._log(
                    f"?? Стойкость — первый эффект типа '{effect_type}' по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} поглощён."
                )
                return False
        if attacker.get("unit_type") == "Abyss Devil":
            victim["paralyzed"] = 1
            self._log(
                f"? Окаменение: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
                f"накладывает окаменение на {victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
        else:
            victim["long_paralyzed"] = 1
            self._log(
                f"? Долгий паралич: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
                f"накладывает долгий паралич на {victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
        return True

    def _apply_tiamat_damage_debuff(self, attacker: Dict, victim: Dict) -> bool:
        effect_type = attacker.get("attack_type_secondary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        if effect_type_lower and effect_type_lower in immunities_lower:
            self._log(
                f"IMMUNE: effect '{effect_type}' blocks Tiamat debuff on "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return False

        if effect_type_lower:
            res_map = {str(r).lower(): r for r in (victim.get("resistance") or [])}
            if effect_type_lower in res_map:
                used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
                if effect_type_lower not in used_lower:
                    victim.setdefault("resilience_used_types", []).append(res_map[effect_type_lower])
                    self._log(
                        f"RESIST: '{effect_type}' blocks Tiamat debuff on "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    return False

        if victim.get("teamated", 0) == 1:
            return False

        victim["teamated"] = 1
        base_primary = int(victim.get("damage", 0) or 0)
        new_primary = max(0, int(round(base_primary * 0.68)))

        if new_primary == base_primary:
            self._log(
                f"Tiamat debuff: damage of {victim['team'].upper()} {victim['name']}#{victim['position']} stays {base_primary}."
            )
            return True

        victim["damage"] = new_primary
        self._log(
            f"Tiamat debuff: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} lowers damage of "
            f"{victim['team'].upper()} {victim['name']}#{victim['position']} from {base_primary} to {new_primary}."
        )

        return True

    def _apply_hermit_initiative_slow(self, attacker: Dict, victim: Dict) -> bool:
        """
        Разово режет инициативу цели на 50% (и базовую, и текущую), не стакается.
        Защита/стойкость проверяются по attack_type_secondary атакующего (как у Tiamat).
        """
        effect_type = attacker.get("attack_type_secondary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        # Иммунитет к типу статуса
        if effect_type_lower and effect_type_lower in immunities_lower:
            self._log(
                f"IMMUNE: effect '{effect_type}' blocks Hermit slow on "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return False

        # Первая стойкость к типу статуса
        if effect_type_lower:
            res_map = {str(r).lower(): r for r in (victim.get("resistance") or [])}
            if effect_type_lower in res_map:
                used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
                if effect_type_lower not in used_lower:
                    victim.setdefault("resilience_used_types", []).append(res_map[effect_type_lower])
                    self._log(
                        f"RESIST: '{effect_type}' blocks Hermit slow on "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    return False

        # Нестакающийся эффект
        if victim.get("hermited", 0) == 1:
            return False

        victim["hermited"] = 1

        base_ini = int(victim.get("initiative_base", 0) or 0)
        cur_ini = int(victim.get("initiative", 0) or 0)

        new_base = max(0, int(round(base_ini * 0.5)))
        new_cur = min(cur_ini, new_base)

        victim["initiative_base"] = new_base
        victim["initiative"] = new_cur

        self._log(
            f"Hermit slow: lowers initiative of {victim['team'].upper()} {victim['name']}#{victim['position']} "
            f"from base {base_ini} to {new_base} (current {cur_ini}->{new_cur})."
        )
        return True

    def _apply_teurg_armor_shred(self, attacker: Dict, victim: Dict) -> bool:
        forbiddenteurg_names = {"Ашган", "Ашкаэль", "Видар", "Мизраэль", "Иллюмиэлль"}

        if victim.get("name") in forbiddenteurg_names:
            self._log("Это страж столицы")
            return False

        effect_type = attacker.get("attack_type_secondary", "")
        effect_type_lower = (effect_type or "").lower()
        immunities_lower = [str(i).lower() for i in (victim.get("immunity") or [])]

        if effect_type_lower and effect_type_lower in immunities_lower:
            self._log(
                f"IMMUNE: effect '{effect_type}' keeps armor intact on "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            return False

        if effect_type_lower:
            res_map = {str(r).lower(): r for r in (victim.get("resistance") or [])}
            if effect_type_lower in res_map:
                used_lower = {str(u).lower() for u in (victim.get("resilience_used_types") or [])}
                if effect_type_lower not in used_lower:
                    victim.setdefault("resilience_used_types", []).append(res_map[effect_type_lower])
                    self._log(
                        f"RESIST: '{effect_type}' keeps armor intact on "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    return False

        current_armor = int(victim.get("armor", 0) or 0)
        new_armor = max(0, current_armor - 15)
        victim["armor"] = new_armor
        if new_armor == current_armor:
            self._log(
                f"Teurg armor shred: armor of {victim['team'].upper()} {victim['name']}#{victim['position']} already 0."
            )
        else:
            self._log(
                f"Teurg armor shred: armor of {victim['team'].upper()} {victim['name']}#{victim['position']} {current_armor}->{new_armor}."
            )
        return True

    def _attack(self, attacker, target_pos: Optional[int]):
        """
        Выполнить действие юнита attacker по типам:
        - Shadow / Succub / Incub: AOE-статусы без урона.
        - Cliric / Deva roshi / Patriach (живые цели): точечное лечение союзника (через противоположную клетку).
        - Patriach (мертвые цели): воскрешает союзника до 50% от максимального здоровья.
        - Travnitsa / Novice / Dwarfdruid / Arhidruid: точечный бафф союзника (через противоположную клетку).
        - Alchemist: доп. ход союзнику (через противоположную клетку).
        - Profit / Sundancer / Sylfid: массовое лечение (всем союзникам).
        - Mage / Dead dragon / Gumtic / Uter Demon / Tiamat / Teurg / Hermit / Vampire / Highvampire: AOE-урон + спецэффекты.
        * Vampire суммирует реальный нанесённый урон и лечится на эту сумму (не выше max_health).
        * Highvampire делится излишком: сперва лечит себя, затем равномерно распределяет остаток между ранеными союзниками.
        * Bone Lord атакует одиночную цель как Warrior, затем высасывает до половины нанесённого урона: сперва лечит себя, остаток раздаёт союзникам как Highvampire.
        * Dregazul бьёт как Bone Lord, но лечит только себя (как Vampire) и после высасывания крови пытается отравить цель, как Spider.
        - summoner: призывает союзного юнита на свободную клетку своей стороны (см. правила ниже).
        - Occultmaster: логирует позиции союзных юнитов без имени либо с `health == 0`, а также состояние своих рядов; при пустом ряду с шансом 50% призывает дракона смерти или драколича, при непустом — закрывает свободные слоты случайным существом (Вампир/Дух/Лич/Тень/Сущий, по 20%).
        - Остальные: одиночная атака по target_pos (+ их пост-эффекты).

        Возврат: (hit: bool, reason: str)
        reason ? {"ok","dead_or_absent","aoe","aoe_no_targets","immune_damage","miss","resilience_block"}
        """
        unit_type = attacker.get("unit_type")
        atk_team = attacker.get("team")
        atk_name = attacker.get("name")
        enemy_team = "blue" if atk_team == "red" else "red"

        # ---------- вспомогательные ----------
        def _live_enemies():
            return [
                u for u in self.combined if u["team"] == enemy_team and self._alive(u)
            ]

        def _aoe_status_all(action_desc: str, log_type: str, apply_effect):
            targets = _live_enemies()
            if not targets:
                self._log(
                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} ({log_type}) пытается {action_desc}: целей нет."
                )
                return False, "aoe_no_targets"
            self._log(
                f"{atk_team.upper()} {attacker['name']}#{attacker['position']} ({log_type}) пытается {action_desc} всю команду противника."
            )
            for victim in targets:
                if not self._roll_hit(attacker):
                    self._log(
                        f"?? Промах: {atk_team.upper()} {attacker['name']}#{attacker['position']} по "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    continue
                apply_effect(victim)
            return True, "aoe"

        def _aoe_damage_all():
            targets = _live_enemies()
            if not targets:
                self._log(
                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} ({unit_type}) применяет массовую атаку: целей нет."
                )
                return False, "aoe_no_targets"

            self._log(
                f"{atk_team.upper()} {attacker['name']}#{attacker['position']} ({unit_type}) применяет массовую атаку."
            )
            vamp_total = 0  # суммарный реальный урон для Vampire/Highvampire

            for victim in targets:
                if not self._roll_hit(attacker):
                    self._log(
                        f"?? Промах: {atk_team.upper()} {attacker['name']}#{attacker['position']} по "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    continue
                if self._is_immune_damage(attacker, victim):
                    self._log(
                        f"?? Иммунитет к урону '{attacker.get('attack_type_primary', '')}' — "
                        f"{victim['team'].upper()} {victim['name']}#{victim['position']}: урон 0."
                    )
                    continue
                if self._resilience_blocks(attacker, victim):
                    continue

                before = victim["health"]
                dmg = self._apply_damage_with_armor(attacker["damage"], victim)
                victim["health"] -= dmg
                after = victim["health"]
                self._log(
                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} > "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']}: {dmg} "
                    f"({before}>{max(0, after)})"
                )

                # Суммируем фактический урон для вампиризма (без оверкилла)
                if unit_type in ("Vampire", "Highvampire") and dmg > 0:
                    inflicted = min(dmg, max(0, before))
                    vamp_total += max(0, inflicted)

                if victim["health"] <= 0:
                    victim["initiative"] = 0
                    self._kill_linked_summons(victim)
                    self._log(
                        f"? {victim['team'].upper()} {victim['name']}#{victim['position']} выведен из строя."
                    )
                    self._release_lord_burn_from_unit(victim)
                    continue

                # --- тип-специфичные AOE эффекты ---
                if unit_type == "Teurg" and dmg > 0:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс снижения брони {int(acc2)}% - "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        self._apply_teurg_armor_shred(attacker, victim)

                if (
                    unit_type == "Dead dragon"
                    and victim.get("poison_turns_left", 0) <= 0
                ):
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"?? Шанс отравления {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"?? Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — яд НЕ накладывается "
                                f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                        else:
                            status_tag = attacker.get(
                                "attack_type_secondary", ""
                            ) or attacker.get("attack_type_primary", "")
                            if not self._resilience_blocks(
                                attacker, victim, custom_tag=status_tag
                            ):
                                turns = self.rng.randint(1, POISON_TURNS)
                                victim["poison_turns_left"] = turns
                                victim["poison_damage_per_tick"] = int(
                                    attacker.get("damage_secondary", 0) or 0
                                )
                                self._log(
                                    f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} накладывает яд "
                                    f"({victim['poison_damage_per_tick']} урона/ход) на "
                                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                )

                if unit_type == "Gumtic" and victim.get("burn_turns_left", 0) <= 0:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс статуса {int(acc2)}% — "
                        + ("успешно:" if roll else "неудачно")
                        + f" против {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' у цели "
                                f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                        else:
                            status_tag = attacker.get(
                                "attack_type_secondary", ""
                            ) or attacker.get("attack_type_primary", "")
                            if not self._resilience_blocks(
                                attacker, victim, custom_tag=status_tag
                            ):
                                victim.pop("burn_source_lord_pos", None)
                                turns = self.rng.randint(1, BURN_TURNS)
                                victim["burn_turns_left"] = turns
                                victim["burn_damage_per_tick"] = int(
                                    attacker.get("damage_secondary", 0) or 0
                                )
                                self._log(
                                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} поджигает цель "
                                    f"({victim['burn_damage_per_tick']} урона/ход) "
                                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                )

                if unit_type == "Drulliaan" and victim.get("uran_turns_left", 0) <= 0:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"?? Шанс водного проклятия {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"?? Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — вода НЕ накладывается "
                                f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                        else:
                            status_tag = attacker.get(
                                "attack_type_secondary", ""
                            ) or attacker.get("attack_type_primary", "")
                            if not self._resilience_blocks(
                                attacker, victim, custom_tag=status_tag
                            ):
                                turns = self.rng.randint(1, URAN_TURNS)
                                victim["uran_turns_left"] = turns
                                victim["uran_damage_per_tick"] = int(
                                    attacker.get("damage_secondary", 0) or 0
                                )
                                self._log(
                                    f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} накладывает воду "
                                    f"({victim['uran_damage_per_tick']} урона/ход) на "
                                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                )

                if unit_type == "Hermit":
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"?? Шанс замедления {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        self._apply_hermit_initiative_slow(attacker, victim)

                if unit_type == "Uter Demon":
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    if acc2 > 0:
                        roll = self._roll_status(acc2)
                        self._log(
                            f"? Шанс долгого паралича {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            self._apply_long_paralysis_effect(attacker, victim)

                if unit_type == "Shamanka":
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    if acc2 > 0:
                        roll = self._roll_status(acc2)
                        self._log(
                            f"? Шанc напугать {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            self._apply_fear_effect(attacker, victim)

                if unit_type == "Tiamat":
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"?? Шанс ослабления урона {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        self._apply_tiamat_damage_debuff(attacker, victim)

            # После цикла — вампиризм
            if unit_type in ("Vampire", "Highvampire") and vamp_total > 0:
                leeched_total = vamp_total // 2
                self._apply_vampiric_heal(
                    attacker, leeched_total, share_leftover=(unit_type == "Highvampire")
                )

            return True, "aoe"

        def _single_target():
            victim = (
                self._unit_by_position(target_pos) if target_pos is not None else None
            )
            if victim is None or not self._alive(victim):
                self._log(
                    f"{atk_team.upper()} {attacker['name']}#{attacker['position']} бьёт pos{target_pos}: цели нет/мертва/недоступна."
                )
                return False, "dead_or_absent"

            if not self._roll_hit(attacker):
                self._log(
                    f"?? Промах: {atk_team.upper()} {attacker['name']}#{attacker['position']} по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                )
                return True, "miss"

            if self._is_immune_damage(attacker, victim):
                self._log(
                    f"?? Иммунитет к урону '{attacker.get('attack_type_primary', '')}' — "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']}: атака наносит 0."
                )
                return True, "immune_damage"

            if self._resilience_blocks(attacker, victim):
                return True, "resilience_block"

            before = victim["health"]
            dmg = self._apply_damage_with_armor(attacker["damage"], victim)
            victim["health"] -= dmg
            after = victim["health"]
            self._log(
                f"{atk_team.upper()} {attacker['name']}#{attacker['position']} > "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}: {dmg} "
                f"({before}>{max(0, after)})"
            )

            if unit_type == "Bone Lord":
                inflicted = min(dmg, max(0, before))
                if inflicted > 0:
                    leeched_total = int(inflicted) // 2
                    self._apply_vampiric_heal(
                        attacker, leeched_total, share_leftover=True
                    )

            if unit_type == "Dregazul":
                inflicted = min(dmg, max(0, before))
                if inflicted > 0:
                    leeched_total = int(inflicted) // 2
                    self._apply_vampiric_heal(
                        attacker, leeched_total, share_leftover=False
                    )
                    self._apply_cached_poison(
                        attacker, victim, self._dregazul_applied_poison
                    )

            if self._alive(victim):
                if unit_type == "Ghost":
                    self._apply_paralysis_effect(attacker, victim)

                if atk_name == "Ниддог":
                    self._apply_cached_poison(
                        attacker, victim, self._spider_applied_poison
                    )

                if unit_type == "Baroness":
                    self._apply_fear_effect(attacker, victim)
                    self._log("?? Baroness inspires fear.")

                if unit_type == "Aleman" and dmg > 0:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"Шанс снижения брони {int(acc2)}% - "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                    )
                    if roll:
                        self._apply_teurg_armor_shred(attacker, victim)

                if unit_type == "Witch":
                    forbiddenwitch_names = {
                        "Ашган",
                        "Ашкаэль",
                        "Видар",
                        "Мизраэль",
                        "Иллюмиэлль",
                    }
                    # Иммунитет/стойкость к «превращению»
                    if self._is_immune_status(attacker, victim):
                        self._log(
                            f"?? Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — превращение НЕ накладывается "
                            f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                    elif victim.get("name") in forbiddenwitch_names:
                        self._log("Это страж столицы")
                    else:
                        status_tag = attacker.get(
                            "attack_type_secondary", ""
                        ) or attacker.get("attack_type_primary", "")
                        if not self._resilience_blocks(
                            attacker, victim, custom_tag=status_tag
                        ):
                            self._apply_witch_effect(attacker, victim)

                if unit_type in ("Betrezen", "Uter", "Abyss Devil"):
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    if acc2 > 0:
                        roll = self._roll_status(acc2)
                        self._log(
                            f"? Шанс долгого паралича {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            self._apply_long_paralysis_effect(attacker, victim)

                if unit_type in (
                    "Death",
                    "Wight",
                    "Sentry",
                    "Watcher",
                    "Lord",
                    "Ismir son",
                    "Centaur Savage",
                    "Spider",
                ):
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)

                    if unit_type == "Death" and victim.get("poison_turns_left", 0) <= 0:
                        self._log(
                            f"?? Шанс отравления {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            if self._is_immune_status(attacker, victim):
                                self._log(
                                    f"?? Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — яд НЕ накладывается "
                                    f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                                )
                            else:
                                status_tag = attacker.get(
                                    "attack_type_secondary", ""
                                ) or attacker.get("attack_type_primary", "")
                                if not self._resilience_blocks(
                                    attacker, victim, custom_tag=status_tag
                                ):
                                    turns = self.rng.randint(1, POISON_TURNS)
                                    victim["poison_turns_left"] = turns
                                    victim["poison_damage_per_tick"] = int(
                                        attacker.get("damage_secondary", 0) or 0
                                    )
                                    self._log(
                                        f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} накладывает яд "
                                        f"({victim['poison_damage_per_tick']} урона/ход) на "
                                        f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                    )

                    elif unit_type == "Wight":
                        self._log(
                            f"?? Wight drain chance {int(acc2)}% - "
                            + ("success" if roll else "fail")
                            + f" vs {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            if self._is_immune_status(attacker, victim):
                                self._log(
                                    f"?? Immunity blocks '{attacker.get('attack_type_secondary', '')}' - "
                                    f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                                )
                            else:
                                status_tag = attacker.get(
                                    "attack_type_secondary", ""
                                ) or attacker.get("attack_type_primary", "")
                                if not self._resilience_blocks(
                                    attacker, victim, custom_tag=status_tag
                                ):
                                    self._apply_wight_decay_effect(attacker, victim)

                    elif (
                        unit_type == "Sentry" and victim.get("uran_turns_left", 0) <= 0
                    ):
                        self._log(
                            f"?? Шанс наложить воду {int(acc2)}% — "
                            + ("успех" if roll else "неудача")
                            + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            if self._is_immune_status(attacker, victim):
                                self._log(
                                    f"?? Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — вода НЕ накладывается "
                                    f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                                )
                            else:
                                status_tag = attacker.get(
                                    "attack_type_secondary", ""
                                ) or attacker.get("attack_type_primary", "")
                                if not self._resilience_blocks(
                                    attacker, victim, custom_tag=status_tag
                                ):
                                    turns = self.rng.randint(1, URAN_TURNS)
                                    victim["uran_turns_left"] = turns
                                    victim["uran_damage_per_tick"] = int(
                                        attacker.get("damage_secondary", 0) or 0
                                    )
                                    self._log(
                                        f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} накладывает воду "
                                        f"({victim['uran_damage_per_tick']} урона/ход) на "
                                        f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                    )

                    elif (
                        unit_type == "Watcher" and victim.get("burn_turns_left", 0) <= 0
                    ):
                        self._log(
                            f"?? Ignite chance {int(acc2)}% — "
                            + ("success" if roll else "fail")
                            + f" vs {victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        if roll:
                            if self._is_immune_status(attacker, victim):
                                self._log(
                                    f"Immunity blocks '{attacker.get('attack_type_secondary', '')}' on "
                                    f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                                )
                            else:
                                status_tag = attacker.get(
                                    "attack_type_secondary", ""
                                ) or attacker.get("attack_type_primary", "")
                                if not self._resilience_blocks(
                                    attacker, victim, custom_tag=status_tag
                                ):
                                    victim.pop("burn_source_lord_pos", None)
                                    turns = self.rng.randint(1, BURN_TURNS)
                                    victim["burn_turns_left"] = turns
                                    victim["burn_damage_per_tick"] = int(
                                        attacker.get("damage_secondary", 0) or 0
                                    )
                                    self._log(
                                        f"{atk_team.upper()} {attacker['name']}#{attacker['position']} applies burn "
                                        f"({victim['burn_damage_per_tick']} dmg/turn) to "
                                        f"{victim['team'].upper()} {victim['name']}#{victim['position']} for {turns} turns."
                                    )

                    elif unit_type == "Lord":
                        pos = attacker["position"]
                        if not self._lord_applied_burn.get(pos, False):
                            self._log(
                                f"?? Шанс наложить поджог {int(acc2)}% — "
                                + ("успех" if roll else "неудача")
                                + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                            if roll:
                                if self._is_immune_status(attacker, victim):
                                    self._log(
                                        f"?? Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — поджог НЕ накладывается "
                                        f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                                    )
                                else:
                                    status_tag = attacker.get(
                                        "attack_type_secondary", ""
                                    ) or attacker.get("attack_type_primary", "")
                                    if not self._resilience_blocks(
                                        attacker, victim, custom_tag=status_tag
                                    ):
                                        turns = self.rng.randint(1, BURN_TURNS)
                                        victim["burn_turns_left"] = turns
                                        victim["burn_damage_per_tick"] = int(
                                            attacker.get("damage_secondary", 0) or 0
                                        )
                                        victim["burn_source_lord_pos"] = pos
                                        self._lord_applied_burn[pos] = True
                                        self._log(
                                            f"?? {atk_team.upper()} {attacker['name']}#{pos} накладывает поджог "
                                            f"({victim['burn_damage_per_tick']} урона/ход) на "
                                            f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                        )

                    elif unit_type == "Ismir son":
                        pos = attacker["position"]
                        if not self._ismir_applied_uran.get(pos, False):
                            self._log(
                                f"?? Шанс наложить воду {int(acc2)}% — "
                                + ("успех" if roll else "неудача")
                                + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                            if roll:
                                if self._is_immune_status(attacker, victim):
                                    self._log(
                                        f"?? Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — вода НЕ накладывается "
                                        f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                                    )
                                else:
                                    status_tag = attacker.get(
                                        "attack_type_secondary", ""
                                    ) or attacker.get("attack_type_primary", "")
                                    if not self._resilience_blocks(
                                        attacker, victim, custom_tag=status_tag
                                    ):
                                        turns = self.rng.randint(1, URAN_TURNS)
                                        victim["uran_turns_left"] = turns
                                        victim["uran_damage_per_tick"] = int(
                                            attacker.get("damage_secondary", 0) or 0
                                        )
                                        self._ismir_applied_uran[pos] = True
                                        self._log(
                                            f"? {atk_team.upper()} {attacker['name']}#{pos} накладывает воду "
                                            f"({victim['uran_damage_per_tick']} урона/ход) на "
                                            f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {turns} хода."
                                        )

                    elif unit_type == "Spider":
                        self._apply_cached_poison(
                            attacker, victim, self._spider_applied_poison
                        )

                    elif unit_type == "Centaur Savage":
                        extra_damage = float(attacker.get("damage", 0) or 0) * 0.05
                        extra_damage = round(extra_damage, 2)
                        if extra_damage > 0 and victim.get("health", 0) > 0:
                            before = victim["health"]
                            victim["health"] -= extra_damage
                            after = victim["health"]
                            self._log(
                                f"? {atk_team.upper()} {attacker['name']}#{attacker['position']} Критический удар "
                                f"({extra_damage:g}) to {victim['team'].upper()} {victim['name']}#{victim['position']} "
                                f"({before:g}>{max(0.0, after):g})."
                            )

            if victim["health"] <= 0:
                victim["initiative"] = 0
                self._kill_linked_summons(victim)
                self._log(
                    f"? {victim['team'].upper()} {victim['name']}#{victim['position']} выведен из строя."
                )
                self._release_lord_burn_from_unit(victim)
            return True, "ok"

        # ====================== блоки по типам ======================

        # --- AOE-статусы без урона ---
        if unit_type == "Shadow":
            return _aoe_status_all(
                "парализовать",
                "Shadow",
                lambda v: self._apply_paralysis_effect(attacker, v),
            )
        elif unit_type == "Incub":
            return _aoe_status_all(
                "Окаменить",
                "Incub",
                lambda v: self._apply_paralysis_effect(attacker, v),
            )
        elif unit_type == "Succub":
            # Учитывать иммунитет и resistance для превращения
            def _succub_apply(v):
                forbiddenwitch_names = {
                    "Ашган",
                    "Ашкаэль",
                    "Видар",
                    "Мизраэль",
                    "Иллюмиэлль",
                }
                # Иммунитет/стойкость к «превращению»
                if self._is_immune_status(attacker, v):
                    self._log(
                        f"?? Иммунитет к эффекту '{attacker.get('attack_type_secondary', '')}' — превращение НЕ накладывается "
                        f"на {v['team'].upper()} {v['name']}#{v['position']}."
                    )
                elif v.get("name") in forbiddenwitch_names:
                    self._log("Это страж столицы")
                    return
                status_tag = attacker.get("attack_type_secondary", "") or attacker.get(
                    "attack_type_primary", ""
                )
                if self._resilience_blocks(attacker, v, custom_tag=status_tag):
                    return
                self._apply_witch_effect(attacker, v)

            return _aoe_status_all("превратить", "Succub", _succub_apply)

        # --- Саппорты: точечное ЛЕЧЕНИЕ через противоположную клетку ---
        elif unit_type in ("Cliric", "Deva roshi"):
            if target_pos is None:
                self._log(
                    f"?? Лечение не сработало: {atk_team.upper()} {attacker['name']}#{attacker['position']} — не указана цель."
                )
                return False, "dead_or_absent"
            opposite_pos = self._opposite_position(target_pos, attacker["team"])
            recipient = (
                self._unit_by_position(opposite_pos)
                if opposite_pos is not None
                else None
            )
            if (
                recipient is None
                or recipient.get("team") != atk_team
                or not self._alive(recipient)
            ):
                self._log(
                    f"?? Лечение не сработало: {atk_team.upper()} {attacker['name']}#{attacker['position']} не находит союзника на противоположной позиции."
                )
                return False, "dead_or_absent"
            healed = self._apply_cliric_heal(attacker, recipient)
            if not healed:
                self._log(
                    f"? Лечение без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} уже на максимальном здоровье."
                )
            return True, "ok"

        elif unit_type == "Patriach":
            if target_pos is None:
                self._log(
                    f"?? Помощь без указания: {atk_team.upper()} {attacker['name']}#{attacker['position']} - не выбрана цель."
                )
                return False, "dead_or_absent"
            opposite_pos = self._opposite_position(target_pos, attacker["team"])
            recipient = (
                self._unit_by_position(opposite_pos)
                if opposite_pos is not None
                else None
            )
            result = self._apply_patriach_support(attacker, recipient)
            if result == "invalid":
                self._log(
                    f"?? Помощь без цели: {atk_team.upper()} {attacker['name']}#{attacker['position']} не может выбрать союзника на противоположной клетке."
                )
                return False, "dead_or_absent"
            if result == "revive_failed" and recipient is not None:
                self._log(
                    f"? Воскрешение без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не может быть возвращён к жизни."
                )
            elif result == "heal_no_effect" and recipient is not None:
                self._log(
                    f"? Лечение без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} уже на максимальном здоровье."
                )
            return True, "ok"

        # --- Саппорты: точечный БАФФ / доп. ход через противоположную клетку ---
        elif unit_type in (
            "Travnitsa",
            "Novice",
            "Alchemist",
            "Dwarfdruid",
            "Arhidruid",
        ):
            if target_pos is None:
                self._log(
                    f"?? Усиление не сработало: {atk_team.upper()} {attacker['name']}#{attacker['position']} — не указана цель."
                )
                return False, "dead_or_absent"
            opposite_pos = self._opposite_position(target_pos, attacker["team"])
            recipient = (
                self._unit_by_position(opposite_pos)
                if opposite_pos is not None
                else None
            )
            if (
                recipient is None
                or recipient.get("team") != atk_team
                or not self._alive(recipient)
            ):
                self._log(
                    f"?? Усиление не сработало: {atk_team.upper()} {attacker['name']}#{attacker['position']} не находит союзника на противоположной позиции."
                )
                return False, "dead_or_absent"

            if unit_type == "Alchemist":
                if self._apply_alchemist_support(attacker, recipient):
                    self._log(
                        f"?? Поддержка: {atk_team.upper()} {attacker['name']}#{attacker['position']} восстанавливает инициативу союзника на pos{recipient['position']}."
                    )
                else:
                    self._log(
                        f"?? Поддержка без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не получает бонус."
                    )
            else:
                cleanse_support = unit_type in ("Dwarfdruid", "Arhidruid")
                cleansed = (
                    self._cleanse_negative_effects(recipient)
                    if cleanse_support
                    else False
                )
                buffed = self._apply_travnitsa_buff(attacker, recipient)

                if buffed:
                    if cleanse_support and cleansed:
                        self._log(
                            f"?? Усиление и очищение: {atk_team.upper()} {attacker['name']}#{attacker['position']} усиливает и снимает негатив с союзника на pos{recipient['position']}."
                        )
                    else:
                        self._log(
                            f"?? Усиление: {atk_team.upper()} {attacker['name']}#{attacker['position']} повышает урон союзника на pos{recipient['position']}"
                        )
                else:
                    self._log(
                        f"?? Усиление без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не может быть усилен."
                    )
                    if cleanse_support and cleansed:
                        self._log(
                            f"?? Очищение: {atk_team.upper()} {attacker['name']}#{attacker['position']} всё же снимает негативные эффекты с союзника на pos{recipient['position']}."
                        )
                    elif cleanse_support:
                        self._log(
                            f"?? Очищение без эффекта: {recipient['team'].upper()} {recipient['name']}#{recipient['position']} не имел негативных эффектов."
                        )
            return True, "ok"

        # --- Саппорты: массовое ЛЕЧЕНИЕ (Profit, Sundancer, Sylfid) ---
        elif unit_type in ("Profit", "Sundancer", "Sylfid"):
            healed_any = False
            cleansed_any = False
            profit_cleanses = unit_type == "Profit"

            for ally in self.combined:
                if (
                    ally.get("team") == atk_team
                    and self._alive(ally)
                    and ally is not attacker
                ):
                    if profit_cleanses and self._cleanse_negative_effects(ally):
                        cleansed_any = True
                    if self._apply_cliric_heal(attacker, ally):
                        healed_any = True

            if healed_any or (profit_cleanses and cleansed_any):
                if profit_cleanses and healed_any and cleansed_any:
                    self._log(
                        f"? Массовое лечение и очищение: {atk_team.upper()} {attacker['name']}#{attacker['position']} "
                        f"восстанавливает здоровье и снимает негатив с союзников."
                    )
                elif healed_any:
                    self._log(
                        f"? Массовое лечение: {atk_team.upper()} {attacker['name']}#{attacker['position']} "
                        f"восстанавливает союзникам здоровье."
                    )
                else:  # только Profit может очистить без лечения
                    self._log(
                        f"? Массовое очищение: {atk_team.upper()} {attacker['name']}#{attacker['position']} "
                        f"снимает негативные эффекты с союзников."
                    )
            else:
                self._log(
                    f"? Массовое лечение без эффекта: {atk_team.upper()} {attacker['name']}#{attacker['position']} — все союзники в порядке."
                )
            return True, "aoe"

        elif unit_type == "Occultmaster":
            spawned, reason = handle_occultmaster_action(
                self,
                attacker,
                RED_FRONT_POSITIONS,
                RED_BACK_POSITIONS,
                BLUE_FRONT_POSITIONS,
                BLUE_BACK_POSITIONS,
            )
            if spawned:
                _apply_transformations_to_units(self.combined)
            if not spawned and atk_team == "red":
                self._log(
                    f"{atk_team.upper()} ход: {atk_name}#{attacker['position']} ({unit_type}) не может призвать союзников и встаёт в защиту."
                )
                armor_before = int(attacker.get("armor", 0) or 0)
                attacker["armor"] = armor_before + DEFEND_ARMOR_BONUS
                attacker["defense"] = 1
                self._log(
                    f"{atk_team.upper()} DEFENCE: {atk_name}#{attacker['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{attacker['armor']})."
                )
            return spawned, reason

        elif unit_type == "Lyf":
            spawned, reason = handle_lyf_action(
                self,
                attacker,
                RED_FRONT_POSITIONS,
                RED_BACK_POSITIONS,
                BLUE_FRONT_POSITIONS,
                BLUE_BACK_POSITIONS,
            )
            if spawned:
                _apply_transformations_to_units(self.combined)
            if not spawned and atk_team == "red":
                self._log(
                    f"{atk_team.upper()} ход: {atk_name}#{attacker['position']} ({unit_type}) не может призвать союзников и встаёт в защиту."
                )
                armor_before = int(attacker.get("armor", 0) or 0)
                attacker["armor"] = armor_before + DEFEND_ARMOR_BONUS
                attacker["defense"] = 1
                self._log(
                    f"{atk_team.upper()} DEFENCE: {atk_name}#{attacker['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{attacker['armor']})."
                )
            return spawned, reason

        elif unit_type == "Laclaan":
            spawned, reason = handle_laclaan_action(
                self,
                attacker,
                RED_FRONT_POSITIONS,
                RED_BACK_POSITIONS,
                BLUE_FRONT_POSITIONS,
                BLUE_BACK_POSITIONS,
            )
            if spawned:
                _apply_transformations_to_units(self.combined)
            if not spawned and atk_team == "red":
                self._log(
                    f"{atk_team.upper()} ход: {atk_name}#{attacker['position']} ({unit_type}) не может призвать союзников и встаёт в защиту."
                )
                armor_before = int(attacker.get("armor", 0) or 0)
                attacker["armor"] = armor_before + DEFEND_ARMOR_BONUS
                attacker["defense"] = 1
                self._log(
                    f"{atk_team.upper()} DEFENCE: {atk_name}#{attacker['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{attacker['armor']})."
                )
            return spawned, reason

        # --- Призыватель: призывает юнита на свободную союзную клетку ---
        elif unit_type == "summoner":
            if target_pos is None:
                self._log(
                    f"?? Призыв не сработал: {atk_team.upper()} {attacker['name']}#{attacker['position']} — не указана цель."
                )
                return False, "dead_or_absent"

            # Куда ставить призыв: противоположная клетка НА СВОЕЙ СТОРОНЕ
            dest = self._opposite_position(target_pos, attacker["team"])
            if dest is None:
                self._log(
                    f"?? Призыв не сработал: не удалось сопоставить pos{target_pos} для команды {atk_team}."
                )
                return False, "dead_or_absent"

            # Проверка занятости клетки живым юнитом
            slot = self._unit_by_position(dest)
            occupied = slot is not None and self._alive(slot)

            # Если dest — задний ряд, запрещаем призыв, если в той же колонке есть живой большой союзник
            def _partner_cell(p: int) -> Optional[int]:
                if p in BLUE_FRONT_POSITIONS or p in RED_FRONT_POSITIONS:
                    return p + 3
                if p in BLUE_BACK_POSITIONS or p in RED_BACK_POSITIONS:
                    return p - 3
                return None

            partner_blocked = False
            if dest in BLUE_BACK_POSITIONS + RED_BACK_POSITIONS:
                partner = _partner_cell(dest)
                if partner is not None:
                    partner_unit = self._unit_by_position(partner)
                    partner_blocked = (
                        partner_unit is not None
                        and self._alive(partner_unit)
                        and partner_unit.get("team") == atk_team
                        and bool(partner_unit.get("big", False))
                    )

            if occupied or partner_blocked:
                reason = "занята" if occupied else "заблокирована большим союзником"
                self._log(
                    f"?? Призыв отменён: клетка pos{dest} {reason} для {atk_team.upper()} {attacker['name']}#{attacker['position']}."
                )
                return False, "dead_or_absent"

            # Шаблон призываемого юнита (Зомби), с адаптацией под команду/позицию/ряд

            summoner_name = (attacker.get("name") or "").strip().lower()
            position_of_summoner = attacker.get("position")

            if "оккульт" in summoner_name:
                spawn_template = {
                    "name": "Зомби",
                    "initiative": 0,
                    "initiative_base": 50,
                    "unit_type": "Warrior",
                    "damage": 50,
                    "damage_secondary": 0,
                    "health": 170,
                    "max_health": 170,
                    "armor": 0,
                    "accuracy": 80,
                    "accuracy_secondary": 0,
                    "immunity": ["Death"],
                    "resistance": [],
                    "attack_type_primary": "Weapon",
                    "attack_type_secondary": "",
                    "big": False,
                    "Summoned": position_of_summoner,
                }
            elif "элементалист" in summoner_name:
                spawn_template = {
                    "name": "Элементаль Воздуха",
                    "initiative": 0,
                    "initiative_base": 40,
                    "unit_type": "Archer",
                    "damage": 30,
                    "damage_secondary": 0,
                    "health": 100,
                    "max_health": 100,
                    "armor": 0,
                    "accuracy": 80,
                    "accuracy_secondary": 0,
                    "immunity": ["Air"],
                    "resistance": [],
                    "attack_type_primary": "Air",
                    "attack_type_secondary": "",
                    "big": False,
                    "Summoned": position_of_summoner,
                }
            elif "мудрец" in summoner_name:
                spawn_template = {
                    "name": "Малый энт",
                    "initiative": 0,
                    "initiative_base": 50,
                    "unit_type": "Warrior",
                    "damage": 30,
                    "damage_secondary": 0,
                    "health": 80,
                    "max_health": 80,
                    "armor": 0,
                    "accuracy": 80,
                    "accuracy_secondary": 0,
                    "immunity": [],
                    "resistance": [],
                    "attack_type_primary": "Weapon",
                    "attack_type_secondary": "",
                    "big": False,
                    "Summoned": position_of_summoner,
                }
            else:
                self._log(
                    f"?? Призыв отменён: {atk_team.upper()} {attacker['name']}#{attacker['position']} не умеет призывать существ."
                )
                return False, "dead_or_absent"

            # Корректировка позиции и стойки с учётом выбранного шаблона
            spawn = {
                **spawn_template,
                "team": atk_team,
                "position": dest,
                "stand": "ahead"
                if dest in (RED_FRONT_POSITIONS + BLUE_FRONT_POSITIONS)
                else "behind",
                "paralyzed": 0,
                "long_paralyzed": 0,
                "running_away": 0,
                "transformed": 0,
                "basestats": [],
            }
            _apply_transformations_to_unit(spawn)

            # Инициализация полей как в _reset_state()
            spawn.setdefault("defense", 0)
            spawn.setdefault("poison_damage_per_tick", 0)
            spawn.setdefault("burn_turns_left", 0)
            spawn.setdefault("burn_damage_per_tick", 0)
            spawn.setdefault("burn_source_lord_pos", None)
            spawn.setdefault("uran_turns_left", 0)
            spawn.setdefault("uran_damage_per_tick", 0)
            spawn.setdefault("resilience_used_types", [])
            spawn.setdefault("bonusturn", 0)
            spawn.setdefault("original_damage", spawn.get("damage", 0))

            # В позиции обычно есть «пустышка»; перезапишем её, чтобы не дублировать позицию
            if slot is None:
                self.combined.append(spawn)
            else:
                slot.clear()
                slot.update(spawn)

            self._log(
                f"?? Призыв: {atk_team.upper()} {attacker['name']}#{attacker['position']} "
                f"призывает {spawn['name']} на pos{dest} ({spawn['stand']})."
            )
            return True, "ok"

        # --- AOE-урон по врагам (включая Vampire/Highvampire) ---
        elif unit_type in (
            "Mage",
            "Dead dragon",
            "Wolf Lord",
            "Gumtic",
            "Drulliaan",
            "Uter Demon",
            "Tiamat",
            "Teurg",
            "Hermit",
            "Vampire",
            "Highvampire",
            "Shamanka",
        ):
            return _aoe_damage_all()

        # --- Обычная одиночная атака и пост-эффекты ---
        else:
            return _single_target()

    def _apply_battle_exp(self, losing_team: str) -> None:
        """Начисляет опыт победившей команде и пишет лог."""
        total_exp = 0.0
        for u in self.combined:
            if u.get("team") != losing_team:
                continue
            if int(u.get("running_away", 0)) == 1:
                continue
            total_exp += float(u.get("exp_kill", 0) or 0)

        self.last_battle_exp = total_exp
        self._log(f"?? Опыт за бой: {total_exp:g}")

        winning_team = "red" if losing_team == "blue" else "blue"
        winners = [
            u
            for u in self.combined
            if u.get("team") == winning_team
            and self._alive(u)
            and int(u.get("running_away", 0)) == 0
        ]
        if not winners:
            return

        share = total_exp / len(winners)
        share_rounded = int(math.floor(share + 0.5))

        self.last_levelups = []
        for u in winners:
            current = float(u.get("exp_current", 0) or 0)
            new_value = current + share_rounded
            req_raw = u.get("exp_required", 0)

            u["exp_current"] = new_value

            if isinstance(req_raw, str) and req_raw.lower() == "max":
                continue
            try:
                req_value = float(req_raw)
            except (TypeError, ValueError):
                continue
            if req_value <= 0:
                continue
            if new_value >= req_value:
                unit_name = u.get("name", "unknown")
                next_level_exp = _resolve_unit_next_level_exp(u)
                if next_level_exp > 0:
                    u["Level"] = _to_int_or_default(u.get("Level", 0), default=0) + 1
                    u["exp_required"] = _to_int_or_default(req_value, default=0) + next_level_exp
                    u["exp_current"] = 0
                    _apply_hero_levelup_bonuses(u)
                else:
                    u["exp_current"] = max(0.0, req_value - 1)
                self._log(f"Уровень юнита {unit_name} повышен")
                self.last_levelups.append(unit_name)

    def _check_victory_after_hit(self):
        if self.winner is not None:
            return
        if not self._team_alive("blue"):
            self.winner = "red"
            self._revert_fenrir_survivors()
            self._restore_default_doppelgangers()
            self._log("?? Победа RED!")
            self._apply_battle_exp("blue")
        elif not self._team_alive("red"):
            self.winner = "blue"
            self._revert_fenrir_survivors()
            self._restore_default_doppelgangers()
            self._log("?? Победа BLUE!")
            self._apply_battle_exp("red")

    def _advance_until_blue_turn(self) -> bool:
        """
        Автопрокрутка до следующего хода BLUE или завершения боя.
        RED ходит сам.
        """
        while self._team_alive("red") and self._team_alive("blue"):
            nxt = self._pop_next()
            if nxt is None:
                self._log(f"— Конец раунда {self.round_no}.")
                self._end_round_restore()
                self.round_no += 1
                self._log(f"— Начало раунда {self.round_no}.")
                continue

            if not self._apply_start_of_turn_effects(nxt):
                if self.winner is not None:
                    return False
                continue

            if nxt.get("paralyzed", 0) == 1 and self._alive(nxt):
                nxt["paralyzed"] = 0
                nxt["initiative"] = 0
                self._log(
                    f"? Паралич: {nxt['team'].upper()} {nxt['name']}#{nxt['position']} пропускает ход."
                )
                self._reset_powerup(nxt)
                continue

            if nxt.get("long_paralyzed", 0) == 1 and self._alive(nxt):
                nxt["initiative"] = 0
                # 33% шанс снять долгий паралич
                if self.rng.random() < 0.33:
                    nxt["long_paralyzed"] = 0
                    self._log(
                        f"? Долгий паралич: {nxt['team'].upper()} {nxt['name']}#{nxt['position']} пропускает ход и восстанавливается."
                    )
                else:
                    self._log(
                        f"? Пас: {nxt['team'].upper()} {nxt['name']}#{nxt['position']} остаётся парализован."
                    )
                self._reset_powerup(nxt)
                continue

            if nxt["team"] == "blue":
                self.current_blue_attacker_pos = nxt["position"]
                self.blue_attacks_left = (
                    2 if nxt.get("unit_type") in ("Demon", "Elfarcher") else 1
                )
                self._log(
                    f"Ход BLUE: {nxt['name']}#{nxt['position']} (иниц {nxt['initiative']}). "
                    + (
                        "Ожидание действия (демон: 2 удара)."
                        if self.blue_attacks_left == 2
                        else "Ожидание действия."
                    )
                )
                return True

            # Ход RED (авто)
            nxt["initiative"] = 0
            strikes = 2 if nxt.get("unit_type") in ("Demon", "Elfarcher") else 1

            for hit_i in range(strikes):
                if self.winner is not None:
                    return False

                target_pos = None
                if nxt.get("unit_type") in (
                    "Warrior",
                    "Centaur Savage",
                    "Demon",
                    "Lord",
                    "Bone Lord",
                    "Dregazul",
                    "Ismir son",
                    "Uter",
                    "Abyss Devil",
                    "Aleman",
                    "Spider",
                ):
                    options = self._warrior_allowed_targets(nxt)
                    valid_options = self._filter_non_immune_targets(nxt, options) if options else []

                    if valid_options:
                        target_pos = self._pick_lowest_hp(valid_options)
                        prefix = nxt.get("unit_type")
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({prefix}) > цель с мин. HP (non-immune) pos{target_pos} (удар {hit_i + 1}/{strikes})."
                        )
                    else:
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt.get('unit_type')}) не может атаковать: доступных целей нет."
                        )
                        armor_before = int(nxt.get("armor", 0) or 0)
                        nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                        nxt["defense"] = 1
                        self._log(
                            f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                        )
                        break
                elif nxt.get("unit_type") in ("Betrezen", "Uter"):
                    # Особая логика для юнитов с параличом/окаменением:
                    # Атакуем цель с макс. уроном, без паралича и иммунитета
                    options = self._warrior_allowed_targets(nxt)
                    valid_options = self._filter_non_immune_targets(nxt, options) if options else []
                    
                    target_pos = None
                    if valid_options:
                        # Пытаемся найти самую опасную непарализованную цель
                        target_pos = self._pick_highest_damage_non_paralyzed(valid_options)
                        # Если таких нет (все парализованы или нет целей), берем просто по мин. HP (добивание)
                        if target_pos is None:
                            target_pos = self._pick_lowest_hp(valid_options)
                            log_target = "цель с мин. HP (non-immune, т.к. все опасные парализованы)"
                        else:
                            log_target = "цель с макс. damage (без паралича)"

                        prefix = nxt.get("unit_type")
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({prefix}) > {log_target} pos{target_pos} (удар {hit_i + 1}/{strikes})."
                        )
                    else:
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt.get('unit_type')}) не может атаковать: доступных целей нет."
                        )
                        armor_before = int(nxt.get("armor", 0) or 0)
                        nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                        nxt["defense"] = 1
                        self._log(
                            f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                        )
                        break

                elif nxt.get("unit_type") in (
                    "Mage",
                    "Dead dragon",
                    "Wolf Lord",
                    "Gumtic",
                    "Drulliaan",
                    "Uter Demon",
                    "Tiamat",
                    "Teurg",
                    "Vampire",
                    "Highvampire",
                    "Shamanka",
                ):
                    if hit_i == 0:
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt.get('unit_type')}) выполняет массовую атаку."
                        )
                    else:
                        break
                elif nxt.get("unit_type") in (
                    "Cliric",
                    "Profit",
                    "Travnitsa",
                    "Deva roshi",
                    "Patriach",
                    "Novice",
                    "Alchemist",
                    "Dwarfdruid",
                    "Arhidruid",
                ):
                    nxt_type = nxt.get("unit_type")
                    if nxt_type == "Cliric":
                        ally_pos = self._cliric_auto_target(nxt)
                        if ally_pos is None:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Cliric) не находит союзников для лечения."
                            )
                            armor_before = int(nxt.get("armor", 0) or 0)
                            nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                            nxt["defense"] = 1
                            self._log(
                                f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                            )
                            break
                        recipient = self._unit_by_position(ally_pos)
                        if recipient is None:
                            break
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} (Cliric) лечит союзника на pos{ally_pos}."
                        )
                        self._apply_cliric_heal(nxt, recipient)
                        break
                    if nxt_type == "Patriach":
                        ally_pos = self._patriach_auto_target(nxt)
                        if ally_pos is None:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) не находит союзников для помощи."
                            )
                            armor_before = int(nxt.get("armor", 0) or 0)
                            nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                            nxt["defense"] = 1
                            self._log(
                                f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                            )
                            break
                        recipient = self._unit_by_position(ally_pos)
                        result = self._apply_patriach_support(nxt, recipient)
                        if result == "heal_success":
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) лечит союзника на pos{ally_pos}."
                            )
                        elif result == "heal_no_effect":
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) пытается лечить союзника на pos{ally_pos}, но тот полон сил."
                            )
                        elif result == "revive_success":
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) воскрешает союзника на pos{ally_pos}."
                            )
                        else:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Patriach) не смог помочь союзнику на pos{ally_pos}."
                            )
                        break

                    if nxt_type == "Profit":
                        healed_any = False
                        for ally in self.combined:
                            if (
                                ally.get("team") == nxt.get("team")
                                and self._alive(ally)
                                and ally is not nxt
                            ):
                                if self._apply_cliric_heal(nxt, ally):
                                    healed_any = True
                        if healed_any:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Profit) массово исцеляет союзников."
                            )
                        else:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} (Profit) не смог исцелить союзников."
                            )
                        break

                    if nxt_type == "Alchemist":
                        buff_pos = self._alchemist_auto_target(nxt)
                    else:
                        buff_pos = self._travnitsa_auto_target(nxt)

                    if buff_pos is None:
                        self._log(
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не находит союзников для поддержки."
                        )
                        if nxt_type == "Alchemist" and nxt.get("team") == "red":
                            armor_before = int(nxt.get("armor", 0) or 0)
                            nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                            nxt["defense"] = 1
                            self._log(
                                f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                            )
                        break

                    recipient = self._unit_by_position(buff_pos)
                    if recipient is None:
                        break

                    if nxt_type == "Alchemist":
                        success = self._apply_alchemist_support(nxt, recipient)
                        action_log = (
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) восстанавливает инициативу союзнику на pos{buff_pos}."
                            if success
                            else f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не смог выполнить действие для союзника на pos{buff_pos}."
                        )
                    else:
                        success = self._apply_travnitsa_buff(nxt, recipient)
                        action_log = (
                            f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) усиливает союзника на pos{buff_pos}."
                            if success
                            else f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не смог выполнить действие для союзника на pos{buff_pos}."
                        )

                    self._log(action_log)
                    break
                else:
                    nxt_type = nxt.get("unit_type")
                    if nxt_type == "summoner":
                        target_pos = self._summoner_auto_target(nxt)
                        if target_pos is None:
                            self._log(
                                f"RED ход: {nxt['name']}#{nxt['position']} ({nxt_type}) не находит свободных клеток для призыва."
                            )
                            armor_before = int(nxt.get("armor", 0) or 0)
                            nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                            nxt["defense"] = 1
                            self._log(
                                f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                            )
                            break
                        log_target = "клетку для призыва"
                    else:
                        live_blue_positions = self._live_positions_of("blue")
                        if nxt_type in ("Witch", "Ghost"):
                            target_pos = self._pick_weapon_immune_or_highest_hp(
                                live_blue_positions
                            )
                            log_target = "цель с Weapon-иммунитетом или макс. HP"
                        elif nxt_type == "Doppelganger":
                            target_pos = self._pick_highest_hp_non_big(exclude_unit=nxt)
                            log_target = "цель с макс. HP (non-big)"
                        elif nxt_type == "Baroness":
                            target_pos = (
                                self._pick_highest_hp(live_blue_positions)
                                if live_blue_positions
                                else None
                            )
                            log_target = "цель с макс. HP"
                        else:
                            valid_targets = (
                                self._filter_non_immune_targets(nxt, live_blue_positions)
                                if live_blue_positions
                                else []
                            )
                            if valid_targets:
                                target_pos = self._pick_lowest_hp(valid_targets)
                                log_target = "цель с мин. HP (non-immune)"
                            else:
                                self._log(
                                    f"RED ход: {nxt['name']}#{nxt['position']} ({nxt.get('unit_type')}) не находит уязвимых целей."
                                )
                                armor_before = int(nxt.get("armor", 0) or 0)
                                nxt["armor"] = armor_before + DEFEND_ARMOR_BONUS
                                nxt["defense"] = 1
                                self._log(
                                    f"RED DEFENCE: {nxt['name']}#{nxt['position']} получает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{nxt['armor']})."
                                )
                                break
                    ut = nxt.get("unit_type")
                    self._log(
                        f"RED ход: {nxt['name']}#{nxt['position']} ({ut}) > {log_target} pos{target_pos} (удар {hit_i + 1}/{strikes})."
                    )

                if target_pos is not None or nxt.get("unit_type") in (
                    "Mage",
                    "Dead dragon",
                    "Wolf Lord",
                    "Gumtic",
                    "Drulliaan",
                    "Uter Demon",
                    "Tiamat",
                    "Teurg",
                    "Vampire",
                    "Highvampire",
                    "Shamanka",
                ):
                    self._attack(nxt, target_pos)
                    self._check_victory_after_hit()
                    if self.winner is not None:
                        return False
            self._reset_powerup(nxt)
        return False

    # ------------------ Наблюдение для агента ------------------

    def _obs(self) -> np.ndarray:
        vec = []
        for pos in RED_POSITIONS + BLUE_POSITIONS:
            u = self._unit_by_position(pos)
            hp_raw = float(u.get("health", 0) or 0)
            max_hp_raw = float(u.get("max_health", 0) or 0)
            hp = _clip01(
                _safe_div(max(0.0, hp_raw), max_hp_raw if max_hp_raw > 0 else HP_FALLBACK_MAX)
            )
            ini = _clip01(_safe_div(max(0.0, float(u.get("initiative", 0) or 0)), INIT_CUR_NORM_MAX))
            ini_b = _clip01(_safe_div(max(0.0, float(u.get("initiative_base", 0) or 0)), INIT_BASE_NORM_MAX))
            dmg = _clip01(_safe_div(max(0.0, float(u.get("damage", 0) or 0)), DMG_NORM_MAX))
            dmg2 = _clip01(_safe_div(max(0.0, float(u.get("damage_secondary", 0) or 0)), DMG2_NORM_MAX))
            team_v = 0.0 if u["team"] == "red" else 1.0
            pos_raw = float(u.get("position", pos) or pos)
            pos_v = _clip01(_safe_div(pos_raw - 1.0, POS_NORM_MAX))
            stand_v = 0.0 if u["stand"] == "ahead" else 1.0

            t_onehot = _one_hot(u.get("unit_type", "Archer"), TYPE_LIST)
            imm_mhot = _multi_hot(u.get("immunity", []), ATTACK_TYPES)
            atk1_oh = (
                _one_hot(u.get("attack_type_primary", ""), ATTACK_TYPES)
                if u.get("attack_type_primary", "") in ATTACK_TYPES
                else [0.0] * len(ATTACK_TYPES)
            )
            atk2_oh = (
                _one_hot(u.get("attack_type_secondary", ""), ATTACK_TYPES)
                if u.get("attack_type_secondary", "") in ATTACK_TYPES
                else [0.0] * len(ATTACK_TYPES)
            )
            res_mhot = _multi_hot(u.get("resistance", []), ATTACK_TYPES)
            armor_v = _clip01(_safe_div(max(0.0, float(u.get("armor", 0) or 0)), ARMOR_NORM_MAX))
            acc_v = _clip01(_safe_div(max(0.0, float(u.get("accuracy", 0) or 0)), ACC_NORM_MAX))
            acc2_v = _clip01(_safe_div(max(0.0, float(u.get("accuracy_secondary", 0) or 0)), ACC_NORM_MAX))
            run_v = _to01_bool(u.get("running_away", 0))
            par_v = _to01_bool(u.get("paralyzed", 0))
            long_par_v = _to01_bool(u.get("long_paralyzed", 0))
            trans_v = _to01_bool(u.get("transformed", 0))
            bonus_v = _clip01(_safe_div(max(0.0, float(u.get("bonusturn", 0) or 0)), BONUS_NORM_MAX))

            waited_v = _to01_bool(u.get("waited", 0))
            def_v = _to01_bool(u.get("defense", 0))
            big_v = _to01_bool(u.get("big", False))
            res_used_mhot = _multi_hot(u.get("resilience_used_types", []), ATTACK_TYPES)
            pois_dmg_v = _clip01(
                _safe_div(max(0.0, float(u.get("poison_damage_per_tick", 0) or 0)), DMG2_NORM_MAX)
            )
            burn_dmg_v = _clip01(
                _safe_div(max(0.0, float(u.get("burn_damage_per_tick", 0) or 0)), DMG2_NORM_MAX)
            )
            uran_dmg_v = _clip01(
                _safe_div(max(0.0, float(u.get("uran_damage_per_tick", 0) or 0)), DMG2_NORM_MAX)
            )
            exp_kill_v = _clip01(
                _safe_div(max(0.0, float(u.get("exp_kill", 0) or 0)), EXP_KILL_NORM_MAX)
            )
            exp_required_raw = max(0.0, float(u.get("exp_required", 0) or 0))
            exp_required_v = _clip01(_safe_div(exp_required_raw, EXP_REQ_NORM_MAX))
            exp_current_v = _clip01(
                _safe_div(max(0.0, float(u.get("exp_current", 0) or 0)), max(1.0, exp_required_raw))
            )

            vec.extend(
                [
                    hp,
                    ini,
                    ini_b,
                    dmg,
                    dmg2,
                    team_v,
                    pos_v,
                    stand_v,
                    *t_onehot,
                    *imm_mhot,
                    *atk1_oh,
                    *atk2_oh,
                    *res_mhot,
                    *res_used_mhot,
                    armor_v,
                    acc_v,
                    acc2_v,
                    run_v,
                    par_v,
                    long_par_v,
                    trans_v,
                    bonus_v,
                    waited_v,
                    def_v,
                    big_v,
                    pois_dmg_v,
                    burn_dmg_v,
                    uran_dmg_v,
                    exp_kill_v,
                    exp_required_v,
                    exp_current_v,
                ]
            )
        obs = np.array(vec, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        obs = np.clip(obs, 0.0, 1.0)
        return obs

    # ------------------ API Gymnasium ------------------

    def reset(self, *, seed=None, options=None):
        # ВАЖНО: не фиксируем сид, чтобы промахи/порядок были случайными в каждом эпизоде
        self._reset_state()
        self._advance_until_blue_turn()
        return self._obs(), {}

    def _revert_fenrir_survivors(self) -> None:
        """Возвращает всех выживших "Дух Фенрира" обратно в "Повелитель волков".
        Здоровье переносится по той же формуле (доля HP) в шкалу 225.
        Если на юните есть снапшот "wolflord_base", восстанавливаем характеристики из него,
        иначе используем каноничные статы Повелителя волков.
        """
        for u in getattr(self, "combined", []):
            if not isinstance(u, dict):
                continue
            if u.get("name") != "Дух Фенрира":
                continue
            if not self._alive(u):
                continue

            cur_hp = float(u.get("health", 0) or 0)
            cur_max = float(u.get("max_health", 0) or 0)
            ratio = 0.0 if cur_max <= 0 else (cur_hp / cur_max)

            base = u.get("wolflord_base")
            if isinstance(base, dict) and base:
                # Восстанавливаем все сохранённые поля, кроме HP — его зададим по доле
                for k, v in base.items():
                    if k in ("health", "max_health"):
                        continue
                    # глубокая копия для списков/словарей
                    u[k] = deepcopy(v) if isinstance(v, (list, dict)) else v
                max_to = int(base.get("max_health", 225) or 225)
            else:
                # Фолбэк: каноничные характеристики Повелителя волков
                max_to = 225
                u["unit_type"] = "Wolf Lord"
                u["damage"] = 40
                u["damage_secondary"] = 0
                u["initiative_base"] = 40
                u["initiative"] = 40
                u["attack_type_primary"] = "Water"
                u["attack_type_secondary"] = ""
                u["armor"] = 0
                u["accuracy"] = 80
                u["accuracy_secondary"] = 0
                u["immunity"] = []
                u["resistance"] = []
                u["big"] = False

            u["max_health"] = max_to
            u["health"] = int(round(max_to * ratio))
            u["name"] = "Повелитель волков"
            # Инициативу приводим к базовой
            u["initiative"] = u.get("initiative_base", u.get("initiative", 0))
            # Чистим снапшот
            if "wolflord_base" in u:
                try:
                    del u["wolflord_base"]
                except Exception:
                    pass

    def step(self, action):
        assert self.winner is None, "Эпизод завершён — вызовите reset()."

        if self.current_blue_attacker_pos is None:
            self._advance_until_blue_turn()
            if self.winner is not None:
                # Эпизод завершён в промежуточном апдейте — возвращаем Фенрира в Волка
                base = self.reward_win if self.winner == "blue" else self.reward_loss
                return self._obs(), base, True, False, {}

        step_shaping = 0.0
        action_idx = int(action)
        is_defend_action = action_idx == DEFEND_ACTION_INDEX
        is_wait_action = action_idx == WAIT_ACTION_INDEX
        is_run_away_action = action_idx == RUN_AWAY_ACTION_INDEX
        target_pos = None
        if action_idx < len(TARGET_POSITIONS):
            target_pos = TARGET_POSITIONS[action_idx]

        attacker = self._unit_by_position(self.current_blue_attacker_pos)
        can_strike = (
            attacker is not None
            and self._alive(attacker)
            and (
                attacker["initiative"] > 0
                or (
                    attacker.get("unit_type") in ("Demon", "Elfarcher")
                    and self.blue_attacks_left > 0
                )
            )
        )
        if can_strike:
            if is_wait_action:
                current_ini = int(attacker.get("initiative", 0) or 0)
                if current_ini > 0:
                    reduced = max(1, int(math.ceil(current_ini / WAIT_INIT_DIVISOR)))
                else:
                    reduced = 0
                attacker["initiative"] = reduced
                attacker["waited"] = 1
                if (
                    attacker.get("unit_type") in ("Demon", "Elfarcher")
                    and self.blue_attacks_left > 1
                ):
                    self.blue_attacks_left = 1
                self._log(
                    f"BLUE действие: {attacker['name']}#{attacker['position']} ожидает (инициатива {current_ini}->{reduced})."
                )
            else:
                attacker["initiative"] = 0

                if is_defend_action:
                    armor_before = int(attacker.get("armor", 0) or 0)
                    attacker["armor"] = armor_before + DEFEND_ARMOR_BONUS
                    attacker["defense"] = 1
                    if (
                        attacker.get("unit_type") in ("Demon", "Elfarcher")
                        and self.blue_attacks_left > 1
                    ):
                        self.blue_attacks_left = 1
                    self._log(
                        f"BLUE действие: {attacker['name']}#{attacker['position']} усиливает броню +{DEFEND_ARMOR_BONUS} ({armor_before}->{attacker['armor']})."
                    )
                elif is_run_away_action:
                    attacker["running_away"] = 1
                    if (
                        attacker.get("unit_type") in ("Demon", "Elfarcher")
                        and self.blue_attacks_left > 1
                    ):
                        self.blue_attacks_left = 1
                    self._log(
                        f"BLUE действие: {attacker['name']}#{attacker['position']} выбирает побег (running_away=1)."
                    )
                else:
                    wolf_self_select = attacker.get(
                        "unit_type"
                    ) == "Wolf Lord" and target_pos == attacker.get("position")
                    if wolf_self_select:
                        self._log(
                            f"BLUE действие: {attacker['name']}#{attacker['position']} > pos{target_pos} (своя позиция выбрана)"
                        )
                        # Снимем снапшот исходных характеристик Повелителя волков, чтобы вернуть их по окончании боя
                        if not attacker.get("wolflord_base"):
                            attacker["wolflord_base"] = {
                                "name": attacker.get("name"),
                                "unit_type": attacker.get("unit_type"),
                                "damage": attacker.get("damage"),
                                "damage_secondary": attacker.get("damage_secondary", 0),
                                "initiative": attacker.get("initiative", 0),
                                "initiative_base": attacker.get("initiative_base", 0),
                                "attack_type_primary": attacker.get(
                                    "attack_type_primary", "Weapon"
                                ),
                                "attack_type_secondary": attacker.get(
                                    "attack_type_secondary", ""
                                ),
                                "max_health": attacker.get("max_health", 225),
                                "armor": attacker.get("armor", 0),
                                "accuracy": attacker.get("accuracy", 100),
                                "accuracy_secondary": attacker.get(
                                    "accuracy_secondary", 0
                                ),
                                "immunity": list(attacker.get("immunity", [])),
                                "resistance": list(attacker.get("resistance", [])),
                                "big": bool(attacker.get("big", False)),
                            }
                        current_health = float(attacker.get("health", 0) or 0)
                        current_max_health = float(attacker.get("max_health", 0) or 0)
                        health_ratio = (
                            0.0
                            if current_max_health <= 0
                            else current_health / current_max_health
                        )
                        attacker["damage"] = 90
                        attacker["unit_type"] = "Warrior"
                        attacker["initiative"] = 0
                        attacker["initiative_base"] = 65
                        attacker["attack_type_primary"] = "Weapon"
                        attacker["max_health"] = 275
                        attacker["health"] = int(round(275 * health_ratio))
                        attacker["name"] = "Дух Фенрира"
                    else:
                        self._log(
                            f"BLUE действие: {attacker['name']}#{attacker['position']} > pos{target_pos}"
                        )

                    # --- Саппорты и лечение/бафф теперь тоже через _attack() ---
                    if wolf_self_select:
                        pass
                    elif attacker.get("unit_type") in (
                        "Cliric",
                        "Profit",
                        "Travnitsa",
                        "Sylfid",
                        "Sundancer",
                        "Deva roshi",
                        "Patriach",
                        "Novice",
                        "Alchemist",
                        "Dwarfdruid",
                        "Arhidruid",
                    ):
                        hit, reason = self._attack(attacker, target_pos)
                        if not hit and reason == "dead_or_absent":
                            step_shaping += self.penalty_invalid_target
                        self._check_victory_after_hit()

                    elif attacker.get("unit_type") == "Doppelganger":
                        target_unit = self._unit_by_position(target_pos)
                        if target_unit is not None and self._alive(target_unit):
                            current_health = float(attacker.get("health", 0) or 0)
                            current_max_health = float(
                                attacker.get("max_health", 1) or 1
                            )
                            attacker_ratio = (
                                0.0
                                if current_max_health <= 0
                                else current_health / current_max_health
                            )

                            attacker["unit_type"] = target_unit.get(
                                "unit_type", attacker["unit_type"]
                            )
                            attacker["initiative"] = target_unit.get(
                                "initiative", attacker["initiative"]
                            )
                            attacker["initiative_base"] = target_unit.get(
                                "initiative_base", attacker["initiative_base"]
                            )
                            attacker["damage"] = target_unit.get(
                                "damage", attacker["damage"]
                            )
                            attacker["damage_secondary"] = target_unit.get(
                                "damage_secondary", attacker.get("damage_secondary")
                            )
                            attacker["max_health"] = target_unit.get(
                                "max_health", attacker["max_health"]
                            )
                            attacker["armor"] = target_unit.get(
                                "armor", attacker["armor"]
                            )
                            attacker["accuracy"] = target_unit.get(
                                "accuracy", attacker["accuracy"]
                            )
                            attacker["accuracy_secondary"] = target_unit.get(
                                "accuracy_secondary", attacker.get("accuracy_secondary")
                            )
                            attacker["immunity"] = list(target_unit.get("immunity", []))
                            attacker["resistance"] = list(
                                target_unit.get("resistance", [])
                            )
                            attacker["attack_type_primary"] = target_unit.get(
                                "attack_type_primary",
                                attacker.get("attack_type_primary"),
                            )
                            attacker["attack_type_secondary"] = target_unit.get(
                                "attack_type_secondary",
                                attacker.get("attack_type_secondary"),
                            )

                            target_health = float(target_unit.get("health", 0) or 0)
                            attacker["health"] = int(
                                round(
                                    target_health * max(0.0, min(1.0, attacker_ratio))
                                )
                            )
                            self._log(
                                f"?? Doppelganger выбор: pos{target_pos} → "
                                f"{target_unit['team'].upper()} {target_unit['name']}#{target_unit['position']}"
                            )
                        else:
                            self._log(f"?? Doppelganger выбор: pos{target_pos} → пусто")
                        self._check_victory_after_hit()

                    elif attacker.get("unit_type") in (
                        "Warrior",
                        "Centaur Savage",
                        "Demon",
                        "Lord",
                        "Bone Lord",
                        "Dregazul",
                        "Ismir son",
                        "Uter",
                        "Abyss Devil",
                        "Spider",
                    ):
                        allowed = self._warrior_allowed_targets(attacker)
                        if target_pos not in allowed:
                            self._log(
                                f"BLUE {attacker['name']}#{attacker['position']} ({attacker.get('unit_type')}) не может достать pos{target_pos}. "
                                f"Доступные цели: {(', '.join('pos' + str(p) for p in allowed)) if allowed else 'нет'}"
                            )
                            step_shaping += self.penalty_unreachable_warrior
                        else:
                            hit, reason = self._attack(attacker, target_pos)
                            if not hit and reason == "dead_or_absent":
                                step_shaping += self.penalty_invalid_target
                            self._check_victory_after_hit()

                    elif attacker.get("unit_type") in (
                        "Mage",
                        "Dead dragon",
                        "Wolf Lord",
                        "Gumtic",
                        "Drulliaan",
                        "Uter Demon",
                        "Tiamat",
                        "Teurg",
                        "Vampire",
                        "Highvampire",
                        "Shamanka",
                    ):
                        hit, reason = self._attack(attacker, target_pos)
                        self._check_victory_after_hit()
                    else:
                        hit, reason = self._attack(attacker, target_pos)
                        if not hit and reason == "dead_or_absent":
                            step_shaping += self.penalty_invalid_target
                        self._check_victory_after_hit()

        if self.winner is None:
            if self.blue_attacks_left > 0:
                self.blue_attacks_left -= 1
            if self.blue_attacks_left > 0:
                reward, terminated = self.reward_step + step_shaping, False
                return self._obs(), reward, terminated, False, {}
            self.current_blue_attacker_pos = None
            self._reset_powerup(attacker)
            self._advance_until_blue_turn()

        if self.winner is None:
            base = self.reward_step
            terminated = False
        else:
            base = self.reward_win if self.winner == "blue" else self.reward_loss
            terminated = True

        reward = base + step_shaping
        truncated = False
        if not terminated:
            self.step_count += 1
            if self.step_count >= 1000:
                truncated = True
                self._log("? Лимит по шагам: бой остановлен на 1000 такте.")
        return self._obs(), reward, terminated, truncated, {}

    # ==================== CAMPAIGN MODE ====================

    def _init_with_custom_teams(self, red_team: list, blue_team: list):
        """
        Инициализация боя с произвольными командами.
        Используется в CampaignEnv для запуска боя с разными врагами
        и сохранённым состоянием BLUE команды.

        Args:
            red_team: Список словарей с юнитами RED команды
            blue_team: Список словарей с юнитами BLUE команды
        """
        self.combined = deepcopy(red_team + blue_team)
        _apply_transformations_to_units(self.combined)

        # Применяем стандартную инициализацию к каждому юниту
        for u in self.combined:
            base_ini = int(u.get("initiative_base", 0) or 0)
            u["initiative"] = base_ini
            if self._alive(u):
                u["initiative"] += self.rng.randint(0, 9)

            # Дефолтные значения
            u.setdefault("attack_type_primary", "Weapon")
            u.setdefault("attack_type_secondary", "")
            u.setdefault("immunity", [])
            u.setdefault("resistance", [])
            u.setdefault("armor", 0)
            # Keep battle-start armor so campaign can restore temporary armor changes.
            u["base_armor"] = int(u.get("armor", 0) or 0)
            u.setdefault("accuracy", 100)
            u.setdefault("accuracy_secondary", 0)           
            u.setdefault("damage_secondary", 0)
            u.setdefault("big", False)
            u.setdefault("Level", _resolve_unit_level(u))
            u.setdefault("next_level_exp", _resolve_unit_next_level_exp(u))
            u.setdefault("bonusturn", 0)
            u.setdefault("original_damage", u.get("damage", 0))

            # Сбрасываем состояния боя
            u["defense"] = 0
            u["waited"] = 0
            u["poison_turns_left"] = 0
            u["poison_damage_per_tick"] = 0
            u["burn_turns_left"] = 0
            u["burn_damage_per_tick"] = 0
            u["burn_source_lord_pos"] = None
            u["uran_turns_left"] = 0
            u["uran_damage_per_tick"] = 0
            u["resilience_used_types"] = []

        # Сбрасываем глобальное состояние боя
        self.round_no = 1
        self.winner = None
        self.current_blue_attacker_pos = None
        self.blue_attacks_left = 0
        self._lord_applied_burn = {}
        self._spider_applied_poison = {}
        self._dregazul_applied_poison = {}
        self._ismir_applied_uran = {}
        self.survived_inits = []
        self.step_count = 0

        self._log(f"Бой начат с кастомными командами. Раунд {self.round_no}.")

        # Прокручиваем до первого хода BLUE
        self._advance_until_blue_turn()
