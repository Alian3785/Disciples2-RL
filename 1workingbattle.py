# ============================================================
# КАСТОМНАЯ СРЕДА Gymnasium «RED vs BLUE»
# Версия: action space = 6 (атакуем только врагов pos1..pos6)
# + big-юниты: Dead dragon (3↔6), Asterot (7↔10), Gargoyle (9↔12)
# + Чекпоинты каждые 1,000,000 шагов
# + Тест с логами и улучшенной визуализацией (большие карточки)
# ---
# Изменение: RED (кроме Mage/Dead dragon) выбирают цель с минимальным HP.
# ============================================================

#to do argparse добавить!!

# (Колаб) Мягко убедимся, что Gymnasium установлен.
try:
    import gymnasium as gym
except Exception:
    # noqa
    import gymnasium as gym

import json
import os
import random
from datetime import datetime
from operator import itemgetter
from typing import List, Dict, Optional

import numpy as np
from gymnasium import spaces

# --- словари для кодирования в наблюдении ---
TYPE_LIST = ["Archer", "gargoil", "Mage", "Warrior", "Demon", "Death", "lord", "Dead dragon", "Ismir son", "Ghost", "Shadow", "Betrezen"]  # one-hot(12)
ATTACK_TYPES = ["Weapon", "earth", "Fire", "Water", "poison", "death", "Mind"]                        # one-hot(7)

def _one_hot(value: str, vocab: List[str]) -> List[float]:
    return [1.0 if value == v else 0.0 for v in vocab]

def _multi_hot(values: List[str], vocab: List[str]) -> List[float]:
    s = set(values or [])
    return [1.0 if v in s else 0.0 for v in vocab]

# ------------------------ Исходные юниты ------------------------
UNITS_RED = [
    {"name": "скелет-рыцарь",  "initiative": 50, "initiative_base": 50, "team": "red",  "position": 1, "stand": "ahead",
     "unit_type": "Warrior", "damage": 100, "damage_secondary": 0, "health": 0, "max_health": 270, "armor": 0, "accuracy": 80, "accuracy_secondary": 0,
     "immunity": ["death", "Poison"], "resistance": [], "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "рыцарь смерти",  "initiative": 50, "initiative_base": 50, "team": "red",  "position": 2, "stand": "ahead",
     "unit_type": "Warrior", "damage": 120, "damage_secondary": 0, "health": 0, "max_health": 255, "armor": 0, "accuracy": 87, "accuracy_secondary": 0,
     "immunity": ["death"], "resistance": [], "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "Мертвый дракон", "initiative": 35, "initiative_base": 35, "team": "red", "position": 3, "stand": "ahead",
     "unit_type": "Dead dragon", "damage": 65, "damage_secondary": 20, "health": 450, "max_health": 450, "armor": 0, "accuracy": 80, "accuracy_secondary": 40,
     "immunity": ["death"], "resistance": [], "attack_type_primary": "death", "attack_type_secondary": "poison", "big": True, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "Архилич", "initiative": 40, "initiative_base": 40, "team": "red", "position": 4, "stand": "behind",
     "unit_type": "Mage", "damage": 90, "damage_secondary": 0, "health": 0, "max_health": 170, "armor": 0, "accuracy": 80, "accuracy_secondary": 0,
     "immunity": ["death"], "resistance": [], "attack_type_primary": "earth", "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "Смерть1", "initiative": 40, "initiative_base": 40, "team": "red", "position": 5, "stand": "behind",
     "unit_type": "Death", "damage": 100, "damage_secondary": 20, "health": 0, "max_health": 125, "armor": 0, "accuracy": 80, "accuracy_secondary": 50,
     "immunity": ["Weapon", "death"], "resistance": [], "attack_type_primary": "Weapon", "attack_type_secondary": "poison", "big": False, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "red", "position": 6, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0, "accuracy": 0, "accuracy_secondary": 0,
     "immunity": ["death"], "resistance": [], "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},
]

UNITS_BLUE = [
    {"name": "Демон Утер",   "initiative": 50, "initiative_base": 50, "team": "blue", "position": 7,  "stand": "ahead",
     "unit_type": "Betrezen","damage": 150, "damage_secondary": 0, "health": 300, "max_health": 300, "armor": 0, "accuracy": 80, "accuracy_secondary": 75,
     "immunity": [], "resistance": ["Mind", "Fire"], "attack_type_primary": "Fire", "attack_type_secondary": "Mind", "big": True, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "Герцог",  "initiative": 50, "initiative_base": 50, "team": "blue", "position": 8,  "stand": "ahead",
     "unit_type": "Warrior", "damage": 120, "damage_secondary": 0, "health": 0, "max_health": 306, "armor": 0, "accuracy": 100, "accuracy_secondary": 0,
     "immunity": [], "resistance": [], "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "Гаргулья", "initiative": 60, "initiative_base": 60, "team": "blue", "position": 9,  "stand": "ahead",
     "unit_type": "gargoil","damage": 85, "damage_secondary": 0, "health": 0, "max_health": 170, "armor": 65, "accuracy": 80, "accuracy_secondary": 0,
     "immunity": ["poison"], "resistance": ["Mind"], "attack_type_primary": "earth", "attack_type_secondary": "", "big": True, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "пусто",    "initiative": 0,  "initiative_base": 0,  "team": "blue", "position": 10, "stand": "behind",
     "unit_type": "Archer","damage": 0,  "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,  "accuracy": 0, "accuracy_secondary": 0,
     "immunity": [], "resistance": [], "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "пусто",    "initiative": 0,  "initiative_base": 0,  "team": "blue", "position": 11, "stand": "behind",
     "unit_type": "Archer","damage": 0,  "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,  "accuracy": 0, "accuracy_secondary": 0,
     "immunity": [], "resistance": [], "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},

    {"name": "пусто",    "initiative": 0,  "initiative_base": 0,  "team": "blue", "position": 12, "stand": "behind",
     "unit_type": "Archer","damage": 0,  "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,  "accuracy": 0, "accuracy_secondary": 0,
     "immunity": [], "resistance": [], "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0,
     "running_away": 0, "transformed": 0},
]   

# Удалить это потом когда не надо будет сохранять юнитов в файл
def _units_to_markdown_section(title: str, units: List[Dict]) -> str:
    lines: List[str] = [f"## {title}", ""]
    for idx, unit in enumerate(units, 1):
        display_name = unit.get("name", f"unit_{idx}")
        lines.append(f"- **{display_name}**")
        for key, value in unit.items():
            if key == "name":
                continue
            formatted = json.dumps(value, ensure_ascii=False)
            lines.append(f"  - `{key}`: {formatted}")
        lines.append("")
    return "\n".join(lines).rstrip()


def save_units_markdown(path: str = "units.md") -> None:
    sections = [
        _units_to_markdown_section("UNITS_RED", UNITS_RED),
        _units_to_markdown_section("UNITS_BLUE", UNITS_BLUE),
    ]
    content = "# Battle Units Snapshot\n\n" + "\n\n".join(sections) + "\n"
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content)


def _format_value_for_py(value) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value_for_py(v) for v in value) + "]"
    return str(value)


def _dict_to_python_literal(unit: Dict) -> List[str]:
    key_order = [
        "name",
        "initiative",
        "initiative_base",
        "team",
        "position",
        "stand",
        "unit_type",
        "damage",
        "damage_secondary",
        "health",
        "max_health",
        "armor",
        "accuracy",
        "accuracy_secondary",
        "immunity",
        "resistance",
        "attack_type_primary",
        "attack_type_secondary",
        "big",
        "paralyzed",
        "long_paralyzed",
        "running_away",
        "transformed",
    ]
    lines = ["    {"]
    ordered_keys = [k for k in key_order if k in unit] + [k for k in unit.keys() if k not in key_order]
    for k in ordered_keys:
        lines.append(f'        "{k}": {_format_value_for_py(unit[k])},')
    lines.append("    },")
    return lines


def _units_to_python_block(name: str, units: List[Dict]) -> str:
    lines: List[str] = [f"{name} = ["]
    for idx, unit in enumerate(units):
        lines.extend(_dict_to_python_literal(unit))
        if idx != len(units) - 1:
            lines.append("")
    lines.append("]")
    return "\n".join(lines)


def append_units_arrays_snapshot(path: str = "units_arrays.py") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "",
        "",
        f"# Snapshot at {timestamp}",
        "# ------------------------ Исходные юниты ------------------------",
        _units_to_python_block("UNITS_RED", UNITS_RED),
        "",
        _units_to_python_block("UNITS_BLUE", UNITS_BLUE),
    ]
    with open(path, "a", encoding="utf-8") as fp:
        fp.write("\n".join(header) + "\n")


save_units_markdown()
append_units_arrays_snapshot()


# Диапазоны позиций
RED_POSITIONS:  List[int] = list(range(1, 7))
BLUE_POSITIONS: List[int] = list(range(7, 13))

# Доступные цели для действия агента — только враги (pos1..pos6)
TARGET_POSITIONS: List[int] = RED_POSITIONS  # 6 действий

# Константы для нормирования наблюдений
MAX_HP   = max([u["health"] for u in (UNITS_RED + UNITS_BLUE)])   # 1020
MAX_INIT = max([u["initiative_base"] for u in (UNITS_RED + UNITS_BLUE)])
MAX_DMG  = max([u["damage"] for u in (UNITS_RED + UNITS_BLUE)])
MAX_DMG2 = max([u["damage_secondary"] for u in (UNITS_RED + UNITS_BLUE)])

POISON_TURNS  = 3
BURN_DAMAGE = 10     # базовый фолбэк; Владыка задаёт per-tick из своего урон2
BURN_TURNS  = 3
URAN_DAMAGE = 10     # базовый фолбэк; Сын Измира задаёт per-tick из своего урон2
URAN_TURNS  = 3

# ------------------------ Кастомная среда ------------------------
class BattleEnv(gym.Env):
    """
    Observation (696): 12 слотов × 58 признаков (добавлены Точность2, таймеры эффектов и другие состояния).
    Action space: Discrete(6) — выбор цели среди врагов (pos1..pos6).
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 reward_win: float = 1.0,
                 reward_loss: float = -1.0,
                 reward_step: float = 0.0,
                 penalty_invalid_target: float = -0.10,
                 penalty_unreachable_warrior: float = -0.20,
                 log_enabled: bool = False):
        super().__init__()
        self.reward_win  = float(reward_win)
        self.reward_loss = float(reward_loss)
        self.reward_step = float(reward_step)

        self.penalty_invalid_target = float(penalty_invalid_target)
        self.penalty_unreachable_warrior = float(penalty_unreachable_warrior)

        self.log_enabled = bool(log_enabled)
        self._pretty_events: List[str] = []

        # Action space = 6
        self.action_space = spaces.Discrete(len(TARGET_POSITIONS))  # 6

        # Observation space
        low_unit = np.array(
            [0, 0, 0, 0, 0, 0, 1, 0]
            + [0] * len(TYPE_LIST)
            + [0] * len(ATTACK_TYPES)
            + [0] * len(ATTACK_TYPES)
            + [0] * len(ATTACK_TYPES)
            + [0] * len(ATTACK_TYPES)
            + [0]
            + [0]
            + [0]
            + [0, 0, 0]
            + [0]
            + [0]
            + [0]
            + [0],
            dtype=np.float32,
        )
        high_unit = np.array(
            [MAX_HP, MAX_INIT, MAX_INIT, MAX_DMG, MAX_DMG2, 1, 12, 1]
            + [1] * len(TYPE_LIST)
            + [1] * len(ATTACK_TYPES)
            + [1] * len(ATTACK_TYPES)
            + [1] * len(ATTACK_TYPES)
            + [1] * len(ATTACK_TYPES)
            + [100]
            + [100]
            + [100]
            + [POISON_TURNS, BURN_TURNS, URAN_TURNS]
            + [1]
            + [1]
            + [1]
            + [1],
            dtype=np.float32,
        )
        low  = np.tile(low_unit, 12)
        high = np.tile(high_unit, 12)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Состояние боя
        self.rng = random.Random()  # без seed → случайность между эпизодами
        self.combined: List[Dict] = []
        self.round_no: int = 1
        self.winner: Optional[str] = None
        self.current_blue_attacker_pos: Optional[int] = None
        self.blue_attacks_left: int = 0
        self._lord_applied_burn: Dict[int, bool] = {}
        self._ismir_applied_uran: Dict[int, bool] = {}

    # ------------------ Вспомогательные методы ------------------

    def seed(self, seed=None):
        # Метод оставлен для совместимости, но мы им не пользуемся, чтобы не фиксировать случайность
        self.rng = random.Random(seed)

    def _alive(self, u) -> bool:
        return u["health"] > 0

    def _team_alive(self, team: str) -> bool:
        return any(self._alive(u) and u["team"] == team for u in self.combined)

    def _unit_by_position(self, pos: int):
        return next((u for u in self.combined if u["position"] == pos), None)

    def _live_positions_of(self, team: str):
        return [u["position"] for u in self.combined if u["team"] == team and self._alive(u)]

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
        if col == 0:   return [0, 1], [2]
        if col == 1:   return [0, 1, 2], []
        return [1, 2], [0]

    def _warrior_allowed_targets(self, attacker: Dict) -> List[int]:
        assert attacker.get("unit_type") in ("Warrior", "Demon", "lord", "Ismir son")
        if attacker["stand"] == "behind":
            if any(self._alive(u) and u["team"] == attacker["team"]
                   and u.get("unit_type") in ("Warrior", "Demon", "lord", "Ismir son") and u["stand"] == "ahead"
                   for u in self.combined):
                return []
        ahead, behind = self._enemy_rows(attacker["team"])
        col = self._col_of(attacker["position"])
        near_cols, far_cols = self._warrior_near_far_cols(col)

        def alive(pos):
            uu = self._unit_by_position(pos)
            return uu is not None and self._alive(uu)

        near = [ahead[c] for c in near_cols if alive(ahead[c])]
        if near: return near
        far = [ahead[c] for c in far_cols if alive(ahead[c])]
        if far: return far

        near2 = [behind[c] for c in near_cols if alive(behind[c])]
        if near2: return near2
        far2 = [behind[c] for c in far_cols if alive(behind[c])]
        return far2

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

    # ----------- эффекты начала хода (яд/поджог) -----------
    def _apply_start_of_turn_effects(self, unit: Dict) -> bool:
        # ЯД
        if unit.get("poison_turns_left", 0) > 0 and self._alive(unit):
            if "poison" in (unit.get("immunity") or []):
                unit["poison_turns_left"] = 0
                unit["poison_damage_per_tick"] = 0
                self._log(f"🛡 Иммунитет к эффекту 'poison' — яд не действует на {unit['team'].upper()} {unit['name']}#{unit['position']}.")
            else:
                tick = int(unit.get("poison_damage_per_tick", 0) or 0)
                if tick > 0:
                    before = unit["health"]
                    unit["health"] -= tick
                    unit["poison_turns_left"] -= 1
                    after = unit["health"]
                    self._log(f"☠ Яд поражает {unit['team'].upper()} {unit['name']}#{unit['position']}: "
                              f"{tick} ({before}→{max(0, after)}); осталось ходов: {unit['poison_turns_left']}")
                    if unit["health"] <= 0:
                        unit["initiative"] = 0
                        self._log(f"✖ {unit['team'].upper()} {unit['name']}#{unit['position']} погибает от яда.")
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["poison_turns_left"] -= 1
                if unit["poison_turns_left"] <= 0:
                    unit["poison_damage_per_tick"] = 0

        # ПОДЖОГ — теперь урон за тик берётся из burn_damage_per_tick (для Владыки = его урон2)
        if unit.get("burn_turns_left", 0) > 0 and self._alive(unit):
            # зеркальная проверка иммунитета к Fire (как для яда)
            if "Fire" in (unit.get("immunity") or []):
                unit["burn_turns_left"] = 0
                unit["burn_damage_per_tick"] = 0
                self._log(f"🛡 Иммунитет к эффекту 'Fire' — поджог не действует на {unit['team'].upper()} {unit['name']}#{unit['position']}.")
            else:
                tick = int(unit.get("burn_damage_per_tick", 0) or BURN_DAMAGE)
                if tick > 0:
                    before = unit["health"]
                    unit["health"] -= tick
                    unit["burn_turns_left"] -= 1
                    after = unit["health"]
                    self._log(f"🔥 Поджог поражает {unit['team'].upper()} {unit['name']}#{unit['position']}: "
                              f"{tick} ({before}→{max(0, after)}); осталось ходов: {unit['burn_turns_left']}")
                    if unit["health"] <= 0:
                        unit["initiative"] = 0
                        self._log(f"✖ {unit['team'].upper()} {unit['name']}#{unit['position']} погибает от поджога.")
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["burn_turns_left"] -= 1
                if unit["burn_turns_left"] <= 0:
                    unit["burn_damage_per_tick"] = 0

        # Вода — наносит периодический урон (для Сына Измира = его урон2)
        if unit.get("uran_turns_left", 0) > 0 and self._alive(unit):
            if "Water" in (unit.get("immunity") or []):
                unit["uran_turns_left"] = 0
                unit["uran_damage_per_tick"] = 0
                self._log(
                    f"🛡 Иммунитет к эффекту 'Water' — вода не действует на {unit['team'].upper()} {unit['name']}#{unit['position']}."
                )
            else:
                tick = int(unit.get("uran_damage_per_tick", 0) or URAN_DAMAGE)
                if tick > 0:
                    before = unit["health"]
                    unit["health"] -= tick
                    unit["uran_turns_left"] -= 1
                    after = unit["health"]
                    self._log(
                        f"☢ Вода поражает {unit['team'].upper()} {unit['name']}#{unit['position']}: "
                        f"{tick} ({before}→{max(0, after)}); осталось ходов: {unit['uran_turns_left']}"
                    )
                    if unit["health"] <= 0:
                        unit["initiative"] = 0
                        self._log(
                            f"✖ {unit['team'].upper()} {unit['name']}#{unit['position']} погибает от воды."
                        )
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["uran_turns_left"] -= 1
                if unit["uran_turns_left"] <= 0:
                    unit["uran_damage_per_tick"] = 0

        return True

    # ------------------ Боевая логика ------------------

    def _reset_state(self):
        self.combined = [u.copy() for u in (UNITS_RED + UNITS_BLUE)]
        for u in self.combined:
            u["initiative"] = u.get("initiative_base", 0)
            u.setdefault("attack_type_primary", "Weapon")
            u.setdefault("attack_type_secondary", "")
            u.setdefault("immunity", [])
            u.setdefault("resistance", [])
            u.setdefault("armor", 0)
            u.setdefault("accuracy", 100)
            u.setdefault("accuracy_secondary", 0)
            u.setdefault("damage_secondary", 0)
            u.setdefault("big", False)
            u["poison_turns_left"] = 0
            u["poison_damage_per_tick"] = 0
            u["burn_turns_left"] = 0
            u["burn_damage_per_tick"] = 0  # per-tick для поджога
            u["uran_turns_left"] = 0
            u["uran_damage_per_tick"] = 0
            u["resilience_used_types"] = []
        self.round_no = 1
        self.winner = None
        self.current_blue_attacker_pos = None
        self.blue_attacks_left = 0
        self._lord_applied_burn = {}
        self._ismir_applied_uran = {}
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
            u["initiative"] = u.get("initiative_base", 0) if self._alive(u) else 0
        self._log("Восстановление инициативы. Новый раунд.")

    # --- иммунитеты/броня/точность/стойкость ---
    def _is_immune_damage(self, attacker: Dict, victim: Dict) -> bool:
        atk1 = attacker.get("attack_type_primary", "Weapon")
        return atk1 in (victim.get("immunity") or [])

    def _is_immune_status(self, attacker: Dict, victim: Dict) -> bool:
        atk2 = attacker.get("attack_type_secondary", "")
        return bool(atk2) and (atk2 in (victim.get("immunity") or []))

    def _resilience_blocks(self, attacker: Dict, victim: Dict) -> bool:
        res_list = set(victim.get("resistance") or [])
        if not res_list:
            return False
        used = set(victim.get("resilience_used_types") or [])
        available = res_list - used
        if not available:
            return False

        atk1 = attacker.get("attack_type_primary", "")
        atk2 = attacker.get("attack_type_secondary", "")

        for tag in (atk1, atk2):
            if tag and tag in available:
                victim.setdefault("resilience_used_types", []).append(tag)
                self._log(f"🧿 Стойкость — первый удар типа '{tag}' по "
                          f"{victim['team'].upper()} {victim['name']}#{victim['position']} поглощён.")
                return True
        return False

    def _apply_damage_with_armor(self, base_dmg: float, victim: Dict) -> int:
        armor = float(victim.get("armor", 0) or 0)
        factor = (1.0 - armor / 100.0) if armor > 0 else 1.0
        eff = base_dmg * max(0.0, factor)
        return int(round(eff))

    def _roll_hit(self, attacker: Dict) -> bool:
        acc = float(attacker.get("accuracy", 100) or 0)
        return self.rng.random() < (acc / 100.0)

    def _roll_status(self, chance_percent: float) -> bool:
        cp = max(0.0, min(100.0, float(chance_percent or 0.0)))
        return self.rng.random() < (cp / 100.0)

    def _apply_paralysis_effect(self, attacker: Dict, victim: Dict) -> bool:
        if victim.get("paralyzed", 0):
            return False
        if "Mind" in (victim.get("immunity") or []):
            self._log(
                f"Иммунитет к 'Mind' — паралич не действует на "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
            return False

        res_list = set(victim.get("resistance") or [])
        if "Mind" in res_list:
            used = set(victim.get("resilience_used_types") or [])
            if "Mind" not in used:
                victim.setdefault("resilience_used_types", []).append("Mind")
                self._log(
                    f"🧿 Стойкость — первый эффект типа 'Mind' по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} поглощён."
                )
                return False

        victim["paralyzed"] = 1
        self._log(
            f"Паралич: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
            f"лишает хода {victim['team'].upper()} {victim['name']}#{victim['position']}"
        )
        return True

    def _apply_long_paralysis_effect(self, attacker: Dict, victim: Dict) -> bool:
        if victim.get("long_paralyzed", 0):
            return False
        if "Mind" in (victim.get("immunity") or []):
            self._log(
                f"Иммунитет к 'Mind' — долгий паралич не действует на "
                f"{victim['team'].upper()} {victim['name']}#{victim['position']}"
            )
            return False

        res_list = set(victim.get("resistance") or [])
        if "Mind" in res_list:
            used = set(victim.get("resilience_used_types") or [])
            if "Mind" not in used:
                victim.setdefault("resilience_used_types", []).append("Mind")
                self._log(
                    f"🧿 Стойкость — первый эффект типа 'Mind' по "
                    f"{victim['team'].upper()} {victim['name']}#{victim['position']} поглощён."
                )
                return False

        victim["long_paralyzed"] = 1
        self._log(
            f"⚡ Долгий паралич: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} "
            f"накладывает долгий паралич на {victim['team'].upper()} {victim['name']}#{victim['position']}"
        )
        return True

    def _apply_shadow_aoe(self, attacker: Dict, primary_victim: Optional[Dict]) -> None:
        enemy_team = "blue" if attacker.get("team") == "red" else "red"
        accuracy = float(attacker.get("accuracy", 0) or 0)
        for victim in self.combined:
            if victim is primary_victim:
                continue
            if victim.get("team") != enemy_team or not self._alive(victim):
                continue
            if victim.get("paralyzed", 0):
                continue
            roll = self._roll_status(accuracy)
            self._log(
                f"🕳 Шанс паралича тенью {int(accuracy)}% — "
                + ("успех" if roll else "неудача")
                + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}."
            )
            if roll:
                self._apply_paralysis_effect(attacker, victim)

    def _attack(self, attacker, target_pos: Optional[int]):
        """
        Выполнить урон.
        Возврат: (hit: bool, reason: str)
          reason ∈ {"ok","dead_or_absent","aoe","aoe_no_targets","immune_damage","immune_status","miss","resilience_block"}
        """
        # AoE для Shadow — массовый паралич
        if attacker is not None and attacker.get("unit_type") == "Shadow":
            enemy_team = "blue" if attacker["team"] == "red" else "red"
            targets = [u for u in self.combined if u["team"] == enemy_team and self._alive(u)]
            if targets:
                self._log(
                    f"{attacker['team'].upper()} {attacker['name']}#{attacker['position']} (Shadow) пытается парализовать всю команду противника."
                )
                for victim in targets:
                    if not self._roll_hit(attacker):
                        self._log(
                            f"💨 Промах: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} по "
                            f"{victim['team'].upper()} {victim['name']}#{victim['position']}."
                        )
                        continue
                    self._apply_paralysis_effect(attacker, victim)
                return True, "aoe"
            else:
                self._log(
                    f"{attacker['team'].upper()} {attacker['name']}#{attacker['position']} (Shadow) пытается парализовать: целей нет."
                )
                return False, "aoe_no_targets"

        # AoE для Mage и Dead dragon
        if attacker is not None and attacker.get("unit_type") in ("Mage", "Dead dragon"):
            enemy_team = "blue" if attacker["team"] == "red" else "red"
            targets = [u for u in self.combined if u["team"] == enemy_team and self._alive(u)]
            if targets:
                atype = attacker.get("unit_type")
                self._log(f"{attacker['team'].upper()} {attacker['name']}#{attacker['position']} ({atype}) применяет массовую атаку.")
                for victim in targets:
                    if not self._roll_hit(attacker):
                        self._log(f"💨 Промах: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} по "
                                  f"{victim['team'].upper()} {victim['name']}#{victim['position']}.")
                        continue
                    if self._is_immune_damage(attacker, victim):
                        self._log(f"🛡 Иммунитет к урону '{attacker.get('attack_type_primary','')}' — "
                                  f"{victim['team'].upper()} {victim['name']}#{victim['position']}: урон 0.")
                        continue
                    if self._resilience_blocks(attacker, victim):
                        continue
                    before = victim["health"]
                    dmg = self._apply_damage_with_armor(attacker["damage"], victim)
                    victim["health"] -= dmg
                    after = victim["health"]
                    self._log(f"{attacker['team'].upper()} {attacker['name']}#{attacker['position']} → "
                              f"{victim['team'].upper()} {victim['name']}#{victim['position']}: {dmg} "
                              f"({before}→{max(0, after)})")
                    if victim["health"] <= 0:
                        victim["initiative"] = 0
                        self._log(f"✖ {victim['team'].upper()} {victim['name']}#{victim['position']} выведен из строя.")
                    else:
                        # ЯД от Dead dragon по шансу Точность2
                        if atype == "Dead dragon" and attacker.get("attack_type_secondary", "") == "poison":
                            acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                            roll = self._roll_status(acc2)
                            self._log(f"🧪 Шанс отравления {int(acc2)}% — " +
                                      ("успех" if roll else "неудача") + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}.")
                            if roll:
                                if self._is_immune_status(attacker, victim):
                                    self._log(f"🛡 Иммунитет к эффекту '{attacker.get('attack_type_secondary','')}' — яд НЕ накладывается "
                                              f"на {victim['team'].upper()} {victim['name']}#{victim['position']}.")
                                else:
                                    victim["poison_turns_left"] = POISON_TURNS
                                    victim["poison_damage_per_tick"] = int(attacker.get("damage_secondary", 0) or 0)
                                    self._log(f"☠ {attacker['team'].upper()} {attacker['name']}#{attacker['position']} накладывает яд "
                                              f"({victim['poison_damage_per_tick']} урона/ход) на "
                                              f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {POISON_TURNS} хода.")
                return True, "aoe"
            else:
                self._log(f"{attacker['team'].upper()} {attacker['name']}#{attacker['position']} ({attacker.get('unit_type')}) применяет массовую атаку: целей нет.")
                return False, "aoe_no_targets"

        # Обычная одиночная атака
        victim = self._unit_by_position(target_pos) if target_pos is not None else None
        if victim is not None and self._alive(victim):
            if not self._roll_hit(attacker):
                self._log(f"💨 Промах: {attacker['team'].upper()} {attacker['name']}#{attacker['position']} по "
                          f"{victim['team'].upper()} {victim['name']}#{victim['position']}.")
                return True, "miss"

            if self._is_immune_damage(attacker, victim):
                self._log(f"🛡 Иммунитет к урону '{attacker.get('attack_type_primary','')}' — "
                          f"{victim['team'].upper()} {victim['name']}#{victim['position']}: атака наносит 0.")
                return True, "immune_damage"

            if self._resilience_blocks(attacker, victim):
                return True, "resilience_block"

            before = victim["health"]
            dmg = self._apply_damage_with_armor(attacker["damage"], victim)
            victim["health"] -= dmg
            after = victim["health"]
            self._log(f"{attacker['team'].upper()} {attacker['name']}#{attacker['position']} → "
                      f"{victim['team'].upper()} {victim['name']}#{victim['position']}: {dmg} "
                      f"({before}→{max(0, after)})")

            if self._alive(victim):
                if attacker.get("unit_type") == "Ghost":
                    self._apply_paralysis_effect(attacker, victim)

                # Долгий паралич от Betrezen по шансу Точность2
                if attacker.get("unit_type") == "Betrezen":
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    if acc2 > 0:
                        roll = self._roll_status(acc2)
                        self._log(f"⚡ Шанс долгого паралича {int(acc2)}% — " +
                                  ("успех" if roll else "неудача") + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}.")
                        if roll:
                            self._apply_long_paralysis_effect(attacker, victim)

                # ЯД от Death по шансу Точность2 (если ещё не отравлен)
                if attacker.get("unit_type") == "Death" and attacker.get("attack_type_secondary", "") == "poison" and victim.get("poison_turns_left", 0) <= 0:
                    acc2 = float(attacker.get("accuracy_secondary", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(f"🧪 Шанс отравления {int(acc2)}% — " +
                              ("успех" if roll else "неудача") + f" по {victim['team'].upper()} {victim['name']}#{victim['position']}.")
                    if roll:
                        if self._is_immune_status(attacker, victim):
                            self._log(f"🛡 Иммунитет к эффекту '{attacker.get('attack_type_secondary','')}' — яд НЕ накладывается "
                                      f"на {victim['team'].upper()} {victim['name']}#{victim['position']}.")
                        else:
                            victim["poison_turns_left"] = POISON_TURNS
                            victim["poison_damage_per_tick"] = int(attacker.get("damage_secondary", 0) or 0)
                            self._log(f"☠ {attacker['team'].upper()} {attacker['name']}#{attacker['position']} накладывает яд "
                                      f"({victim['poison_damage_per_tick']} урона/ход) на "
                                      f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {POISON_TURNS} хода.")

                # Поджог от Владыки
                if attacker.get("unit_type") == "lord":
                    pos = attacker["position"]
                    if not self._lord_applied_burn.get(pos, False):
                        if self._is_immune_status(attacker, victim):
                            self._log(f"🛡 Иммунитет к эффекту '{attacker.get('attack_type_secondary','')}' — поджог НЕ накладывается "
                                      f"на {victim['team'].upper()} {victim['name']}#{victim['position']}.")
                        else:
                            victim["burn_turns_left"] = BURN_TURNS
                            # сила тика поджога = урон2 Владыки
                            victim["burn_damage_per_tick"] = int(attacker.get("damage_secondary", 0) or 0)
                            self._lord_applied_burn[pos] = True
                            self._log(f"🔥 {attacker['team'].upper()} {attacker['name']}#{pos} накладывает поджог "
                                      f"({victim['burn_damage_per_tick']} урона/ход) на "
                                      f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {BURN_TURNS} хода.")

                # Вода от Сына Измира
                if attacker.get("unit_type") == "Ismir son":
                    pos = attacker["position"]
                    if not self._ismir_applied_uran.get(pos, False):
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"🛡 Иммунитет к эффекту '{attacker.get('attack_type_secondary','')}' — вода НЕ накладывается "
                                f"на {victim['team'].upper()} {victim['name']}#{victim['position']}."
                            )
                        else:
                            victim["uran_turns_left"] = URAN_TURNS
                            victim["uran_damage_per_tick"] = int(attacker.get("damage_secondary", 0) or 0)
                            self._ismir_applied_uran[pos] = True
                            self._log(
                                f"☢ {attacker['team'].upper()} {attacker['name']}#{pos} накладывает воду "
                                f"({victim['uran_damage_per_tick']} урона/ход) на "
                                f"{victim['team'].upper()} {victim['name']}#{victim['position']} на {URAN_TURNS} хода."
                            )

            if victim["health"] <= 0:
                victim["initiative"] = 0
                self._log(f"✖ {victim['team'].upper()} {victim['name']}#{victim['position']} выведен из строя.")
            return True, "ok"
        else:
            self._log(f"{attacker['team'].upper()} {attacker['name']}#{attacker['position']} бьёт pos{target_pos}: цели нет/мертва/недоступна.")
            return False, "dead_or_absent"

    def _check_victory_after_hit(self):
        if not self._team_alive("blue"):
            self.winner = "red";  self._log("🏆 Победа RED!")
        elif not self._team_alive("red"):
            self.winner = "blue"; self._log("🏆 Победа BLUE!")

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
                    f"⛔ Паралич: {nxt['team'].upper()} {nxt['name']}#{nxt['position']} пропускает ход."
                )
                continue

            if nxt.get("long_paralyzed", 0) == 1 and self._alive(nxt):
                nxt["initiative"] = 0
                # 33% шанс снять долгий паралич
                if self.rng.random() < 0.33:
                    nxt["long_paralyzed"] = 0
                    self._log(
                        f"⛔ Долгий паралич: {nxt['team'].upper()} {nxt['name']}#{nxt['position']} пропускает ход и восстанавливается."
                    )
                else:
                    self._log(
                        f"⛔ Пас: {nxt['team'].upper()} {nxt['name']}#{nxt['position']} остаётся парализован."
                    )
                continue

            if nxt["team"] == "blue":
                self.current_blue_attacker_pos = nxt["position"]
                self.blue_attacks_left = 2 if (nxt.get("unit_type") == "Demon") else 1
                self._log(
                    f"Ход BLUE: {nxt['name']}#{nxt['position']} (иниц {nxt['initiative']}). "
                    + ("Ожидание действия (демон: 2 удара)." if self.blue_attacks_left == 2 else "Ожидание действия.")
                )
                return True

            # Ход RED (авто)
            nxt["initiative"] = 0
            strikes = 2 if nxt.get("unit_type") == "Demon" else 1

            for hit_i in range(strikes):
                if self.winner is not None:
                    return False

                target_pos = None
                if nxt.get("unit_type") in ("Warrior", "Demon", "lord", "Ismir son"):
                    options = self._warrior_allowed_targets(nxt)
                    if options:
                        target_pos = self._pick_lowest_hp(options)
                        prefix = nxt.get("unit_type")
                        self._log(f"RED ход: {nxt['name']}#{nxt['position']} ({prefix}) → цель с мин. HP pos{target_pos} (удар {hit_i+1}/{strikes}).")
                    else:
                        self._log(f"RED ход: {nxt['name']}#{nxt['position']} ({nxt.get('unit_type')}) не может атаковать: доступных целей нет.")
                elif nxt.get("unit_type") in ("Mage", "Dead dragon"):
                    if hit_i == 0:
                        self._log(f"RED ход: {nxt['name']}#{nxt['position']} ({nxt.get('unit_type')}) выполняет массовую атаку.")
                    else:
                        break
                else:
                    live_blue_positions = self._live_positions_of("blue")
                    target_pos = self._pick_lowest_hp(live_blue_positions) if live_blue_positions else None
                    ut = nxt.get("unit_type")
                    self._log(f"RED ход: {nxt['name']}#{nxt['position']} ({ut}) → цель с мин. HP pos{target_pos} (удар {hit_i+1}/{strikes}).")

                if target_pos is not None or nxt.get("unit_type") in ("Mage", "Dead dragon"):
                    self._attack(nxt, target_pos)
                    self._check_victory_after_hit()
                    if self.winner is not None:
                        return False
        return False

    # ------------------ Наблюдение для агента ------------------

    def _obs(self) -> np.ndarray:
        vec = []
        for pos in RED_POSITIONS + BLUE_POSITIONS:
            u = self._unit_by_position(pos)
            hp      = float(max(0, u["health"]))
            ini     = float(max(0, u["initiative"]))
            ini_b   = float(u.get("initiative_base", 0))
            dmg     = float(u["damage"])
            dmg2    = float(u.get("damage_secondary", 0))
            team_v  = 0.0 if u["team"] == "red" else 1.0
            pos_v   = float(u["position"])
            stand_v = 0.0 if u["stand"] == "ahead" else 1.0

            t_onehot   = _one_hot(u.get("unit_type","Archer"), TYPE_LIST)
            imm_mhot   = _multi_hot(u.get("immunity", []), ATTACK_TYPES)
            atk1_oh    = _one_hot(u.get("attack_type_primary",""), ATTACK_TYPES) if u.get("attack_type_primary","") in ATTACK_TYPES else [0.0]*len(ATTACK_TYPES)
            atk2_oh    = _one_hot(u.get("attack_type_secondary",""), ATTACK_TYPES) if u.get("attack_type_secondary","") in ATTACK_TYPES else [0.0]*len(ATTACK_TYPES)
            res_mhot   = _multi_hot(u.get("resistance", []), ATTACK_TYPES)
            armor_v    = float(u.get("armor", 0))
            acc_v      = float(u.get("accuracy", 100))
            acc2_v     = float(u.get("accuracy_secondary", 0))
            poison_v   = float(u.get("poison_turns_left", 0))
            burn_v     = float(u.get("burn_turns_left", 0))
            uran_v     = float(u.get("uran_turns_left", 0))
            run_v      = float(u.get("running_away", 0))
            par_v      = float(u.get("paralyzed", 0))
            long_par_v = float(u.get("long_paralyzed", 0))
            trans_v    = float(u.get("transformed", 0))

            vec.extend([hp, ini, ini_b, dmg, dmg2, team_v, pos_v, stand_v,
                        *t_onehot, *imm_mhot, *atk1_oh, *atk2_oh, *res_mhot,
                        armor_v, acc_v, acc2_v, poison_v, burn_v, uran_v, run_v, par_v, long_par_v, trans_v])
        return np.array(vec, dtype=np.float32)

    # ------------------ API Gymnasium ------------------

    def reset(self, *, seed=None, options=None):
        # ВАЖНО: не фиксируем сид, чтобы промахи/порядок были случайными в каждом эпизоде
        self._reset_state()
        self._advance_until_blue_turn()
        return self._obs(), {}

    def step(self, action):
        assert self.winner is None, "Эпизод завершён — вызовите reset()."

        if self.current_blue_attacker_pos is None:
            self._advance_until_blue_turn()
            if self.winner is not None:
                base = self.reward_win if self.winner == "blue" else self.reward_loss
                return self._obs(), base, True, False, {}

        step_shaping = 0.0
        target_pos = TARGET_POSITIONS[int(action)]

        attacker = self._unit_by_position(self.current_blue_attacker_pos)
        can_strike = (
            attacker is not None
            and self._alive(attacker)
            and (attacker["initiative"] > 0 or (attacker.get("unit_type") == "Demon" and self.blue_attacks_left > 0))
        )
        if can_strike:
            self._log(f"BLUE действие: {attacker['name']}#{attacker['position']} → pos{target_pos}")
            attacker["initiative"] = 0

            if attacker.get("unit_type") in ("Warrior", "Demon", "lord", "Ismir son"):
                allowed = self._warrior_allowed_targets(attacker)
                if target_pos not in allowed:
                    self._log(
                        f"BLUE {attacker['name']}#{attacker['position']} ({attacker.get('unit_type')}) не может достать pos{target_pos}. "
                        f"Доступные цели: {(', '.join('pos'+str(p) for p in allowed)) if allowed else 'нет'}"
                    )
                    step_shaping += self.penalty_unreachable_warrior
                else:
                    hit, reason = self._attack(attacker, target_pos)
                    if not hit and reason == "dead_or_absent":
                        step_shaping += self.penalty_invalid_target
                    self._check_victory_after_hit()
            elif attacker.get("unit_type") in ("Mage", "Dead dragon"):
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
            self._advance_until_blue_turn()

        if self.winner is None:
            base = self.reward_step
            terminated = False
        else:
            base = self.reward_win if self.winner == "blue" else self.reward_loss
            terminated = True

        reward = base + step_shaping
        return self._obs(), reward, terminated, False, {}

# ============================================================
# ОБУЧЕНИЕ PPO + W&B, ЧЕКПОИНТЫ, ТЕСТ И УЛУЧШЕННАЯ ВИЗУАЛИЗАЦИЯ
# (включая big-юниты)
# ============================================================

# (Колаб) Установим пакеты, если нужно
try:
    import wandb  # noqa
    from stable_baselines3 import PPO  # noqa
except Exception:
    # noqa
    from stable_baselines3 import PPO
    import wandb

import re
import time
import math
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
# mypy: disable-error-code=attr-defined

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_checker import check_env

check_env(BattleEnv(log_enabled=False), warn=True)

# -------------------- Параметры обучения и теста --------------------
TOTAL_STEPS     = 100000
N_ENVS          = 8
MODEL_SAVE_FREQ = 1000000
EVAL_FREQ       = 1000000

# Чекпоинты каждые 1,000,000 шагов
CHECKPOINT_DIR        = None
CHECKPOINT_EVERY      = 1000000

VISUALIZE_TEST = True
FRAME_DELAY    = 0.28

# -------------------- Инициализация W&B --------------------
WANDB_AVAILABLE = True

try:
    run = wandb.init(
        project="red-blue-battle",
        name=f"ppo-{TOTAL_STEPS//1000}k-actions6-big",
        config={
            "algo": "PPO",
            "total_timesteps": TOTAL_STEPS,
            "n_envs": N_ENVS,
            "n_steps": 1024,
            "batch_size": 2048,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "model_save_freq": MODEL_SAVE_FREQ,
            "eval_freq": EVAL_FREQ,
            "obs_dim": 12*58,  # <— 12 юнитов × 58 признаков (включая Betrezen)
            "types": TYPE_LIST,
            "attack_types": ATTACK_TYPES,
            "action_space": len(TARGET_POSITIONS),  # 6
            "checkpoint_every": CHECKPOINT_EVERY,
            "big_enabled": True,
        },
        sync_tensorboard=True,
        save_code=True,
    )
except Exception as exc:
    WANDB_AVAILABLE = False
    fallback_run_id = f"offline-{int(time.time())}"
    print("[WARN] W&B init failed, switching to offline mode:", exc)

    class _OfflineRun:
        def __init__(self, offline_id: str):
            self.id = offline_id

        def finish(self):
            return None

    run = _OfflineRun(fallback_run_id)

CHECKPOINT_DIR = f"./checkpoints/{run.id}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------- Фабрика сред --------------------
def make_env():
    return Monitor(BattleEnv(
        reward_win=1.0, reward_loss=-1.0, reward_step=0.0,
        log_enabled=False
    ))

# ВАЖНО: без seed → каждая среда инициализируется случайно
vec_env  = make_vec_env(make_env, n_envs=N_ENVS)
eval_env = Monitor(BattleEnv(log_enabled=False))

# -------------------- Настройка модели PPO --------------------
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    n_steps=1024,
    batch_size=2048,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    # seed НЕ задаём, чтобы случайность модели/политики не фиксировалась
    tensorboard_log=f"./tb_logs/{run.id}",
)

# -------------------- Колбэк чекпоинтов на 1M шагов --------------------
class MillionStepCheckpointCallback(BaseCallback):
    def __init__(self, save_dir: str, milestone_steps: int = 1_000_000, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.milestone_steps = int(milestone_steps)
        self.next_milestone = self.milestone_steps
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_milestone:
            path = os.path.join(self.save_dir, f"ppo_step_{self.next_milestone}.zip")
            self.model.save(path)
            if self.verbose:
                print(f"[Checkpoint] Saved at {self.next_milestone} steps → {path}")
            if WANDB_AVAILABLE:
                wandb.log({"checkpoint_saved_at": self.next_milestone})
            self.next_milestone += self.milestone_steps
        return True

checkpoint_cb = MillionStepCheckpointCallback(
    save_dir=CHECKPOINT_DIR,
    milestone_steps=CHECKPOINT_EVERY,
    verbose=1
)

# -------------------- Колбэки W&B и оценки --------------------
if WANDB_AVAILABLE:
    wandb_cb = WandbCallback(
        gradient_save_freq=10000,
        model_save_path=f"./models/{run.id}",
        model_save_freq=MODEL_SAVE_FREQ,
        verbose=2,
    )
else:
    class _NoOpCallback(BaseCallback):
        def _on_step(self) -> bool:  # type: ignore[override]
            return True

    wandb_cb = _NoOpCallback()
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=f"./models/{run.id}/best",
    log_path=f"./eval/{run.id}",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

# -------------------- Обучение --------------------
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=CallbackList([wandb_cb, eval_cb, checkpoint_cb])
)

# Сохранить финальную модель
final_model_path = f"./models/{run.id}/final_model"
os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
model.save(final_model_path)
if WANDB_AVAILABLE:
    run.finish()

# -------------------- ТЕСТОВЫЙ ПРОГОН С ЛОГАМИ (финальная модель) --------------------
print("\n=== ТЕСТОВЫЙ ПРОГОН С ЛОГАМИ (финальная модель) ===")
test_env = BattleEnv(log_enabled=True)
obs, info = test_env.reset()  # без seed — промахи/события будут отличаться между прогонами

all_logs = []
all_logs += test_env.pop_pretty_events()

done = False
total_reward = 0.0
step_i = 0
while not done:
    step_i += 1
    action, _ = model.predict(obs, deterministic=True)

    target_pos = TARGET_POSITIONS[int(action)]
    chosen_line = f"[STEP {step_i}] Агент выбирает action={int(action)} → атака RED pos{target_pos}"
    print("\n" + chosen_line)
    all_logs.append(chosen_line)

    obs, reward, terminated, truncated, info = test_env.step(action)
    total_reward += reward
    done = terminated or truncated

    new_lines = test_env.pop_pretty_events()
    all_logs += new_lines
    for line in new_lines:
        print(line)

print("\n=== ЭПИЗОД ЗАВЕРШЁН ===")
print("Победитель:", test_env.winner.upper(), "| Суммарная награда:", total_reward)

# -------------------- ВИЗУАЛИЗАЦИЯ ЛОГОВ (HP-полоска + big-юниты) --------------------
if VISUALIZE_TEST:
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.patches import FancyArrowPatch
    from IPython.display import display

    VISUAL_SPEED_MULT = 8.0  # множитель задержки (больше = медленнее)

    # --- геометрия поля
    RED_FRONT, RED_BACK   = [1,2,3],   [4,5,6]
    BLUE_FRONT, BLUE_BACK = [7,8,9],   [10,11,12]
    COL_X = {0: 0.18, 1: 0.50, 2: 0.82}
    Y_BLUE_BACK, Y_BLUE_FRONT = 0.88, 0.70
    Y_RED_FRONT,  Y_RED_BACK  = 0.30, 0.12
    SLOT_W, SLOT_H = 0.28, 0.13
    HP_H = 0.028

    # --- состояние для отрисовки (maxhp = стартовое HP)
    state = {}
    def _suffix_by_type(t: str) -> str:
        return {
            "Mage": " (Mage)",
            "gargoil": " (Gargoil)",
            "Warrior": " (Warrior)",
            "Demon": " (Demon)",
            "Death": " (Death)",
            "Archer": " (лучник)",
            "lord": " (Lord)",
            "Dead dragon": " (Dead dragon)",
            "Ismir son": " (Ismir son)",
            "Ghost": " (Ghost)",
            "Shadow": " (Shadow)",
            "Betrezen": " (Betrezen)",
        }.get(t, f" ({t})")

    for u in (UNITS_RED + UNITS_BLUE):
        t = u.get("unit_type", "Archer")
        start_hp = float(u["health"])
        state[u["position"]] = {
            "team": u["team"],
            "name": u["name"] + _suffix_by_type(t),
            "hp": start_hp,
            "maxhp": start_hp,
            "stand": u["stand"],
            "type": t,
            "big": bool(u.get("big", False)),
            "acc2": float(u.get("accuracy_secondary", 0)),
            "paralyzed": int(u.get("paralyzed", 0)),
        }

    fig, ax = plt.subplots(figsize=(12, 8))
    handle = None
    try:
        handle = display(fig, display_id=True)
    except Exception:
        handle = None

    if handle is not None:
        plt.close(fig)
    else:
        plt.ion()
        try:
            fig.show()
        except Exception:
            pass

    def pos_to_xy(pos: int):
        col = (pos - 1) % 3
        x = COL_X[col] - SLOT_W/2
        if pos in BLUE_BACK:   y = Y_BLUE_BACK  - SLOT_H/2
        elif pos in BLUE_FRONT:y = Y_BLUE_FRONT - SLOT_H/2
        elif pos in RED_FRONT: y = Y_RED_FRONT  - SLOT_H/2
        else:                  y = Y_RED_BACK   - SLOT_H/2
        return x, y

    def pos_center(pos: int):
        x, y = pos_to_xy(pos)
        return x + SLOT_W/2, y + SLOT_H/2

    def _set_hp(pos: int, hp_value: float):
        clamped = max(0.0, hp_value)
        if pos in state:
            state[pos]["hp"] = clamped
        partner = None
        if pos in state and state[pos].get("big", False):
            partner = pos + 3
        elif (pos - 3) in state and state[pos - 3].get("big", False):
            partner = pos - 3
        if partner in state:
            state[partner]["hp"] = clamped

    def _set_paralysis(pos: int, value: int):
        if pos in state:
            state[pos]["paralyzed"] = value
        partner = None
        if pos in state and state[pos].get("big", False):
            partner = pos + 3
        elif (pos - 3) in state and state[pos - 3].get("big", False):
            partner = pos - 3
        if partner in state:
            state[partner]["paralyzed"] = value

    # ---- рисование ячейки обычного размера
    def draw_unit(ax, pos, is_active: bool = False):
        u = state[pos]; x, y = pos_to_xy(pos)

        # пропуск задних слотов с 0 HP
        if pos in (RED_BACK + BLUE_BACK) and u["hp"] <= 0:
            return

        card_color = (0.90, 0.30, 0.30, 0.16) if u["team"] == "red" else (0.30, 0.45, 0.90, 0.16)
        ax.add_patch(patches.FancyBboxPatch((x, y), SLOT_W, SLOT_H,
                                            boxstyle="round,pad=0.010,rounding_size=0.016",
                                            linewidth=1.3, edgecolor="black", facecolor=card_color))
        if is_active:
            ax.add_patch(patches.FancyBboxPatch(
                (x-0.01, y-0.01), SLOT_W+0.02, SLOT_H+0.02,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=3.0, edgecolor=(1.0, 0.82, 0.10), facecolor='none', alpha=0.95
            ))

        name = u["name"][:22]
        ax.text(x + 0.012, y + SLOT_H*0.78, name,
                fontsize=10, fontweight="bold", ha="left", va="center", color="black")

        # метка шанса отравления, если есть
        if u.get("acc2", 0) > 0:
            ax.text(x + SLOT_W - 0.012, y + SLOT_H*0.78, f"☠{int(u['acc2'])}%",
                    fontsize=9, ha="right", va="center", color="black")

        hp_x, hp_y = x + 0.012, y + SLOT_H*0.10
        hp_w = SLOT_W - 0.024
        ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w, HP_H, facecolor=(0.88,0.88,0.88), edgecolor='none'))
        frac = max(0.0, min(1.0, u["hp"]/u["maxhp"])) if u["maxhp"] > 1e-9 else 0.0
        ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w*frac, HP_H, facecolor=(0.15,0.70,0.25), edgecolor='none'))
        ax.text(hp_x + hp_w/2, hp_y + HP_H/2, f"{int(max(0,u['hp']))}/{int(u['maxhp'])}",
                fontsize=9, ha="center", va="center", color="black")

        if u.get("paralyzed", 0):
            ax.text(x + SLOT_W/2, y + SLOT_H*0.38, "PAR", fontsize=9, fontweight="bold",
                    ha="center", va="center", color=(0.75, 0.05, 0.05))

        ax.text(x + SLOT_W/2, y - 0.008, f"pos{pos}", fontsize=8, ha="center", va="top", color=(0.2,0.2,0.2))

    # ---- рисование «большой» карточки, занимающей front+back одного столбца
    def draw_big_unit(ax, front_pos: int, is_active: bool = False):
        back_pos = front_pos + 3  # пара в том же столбце
        uf = state[front_pos]
        x_front, y_front = pos_to_xy(front_pos)
        x_back,  y_back  = pos_to_xy(back_pos)
        x = x_front
        y = min(y_front, y_back)
        height = (max(y_front, y_back) + SLOT_H) - y
        width = SLOT_W

        if uf["hp"] <= 0:
            return

        card_color = (0.90, 0.30, 0.30, 0.20) if uf["team"] == "red" else (0.30, 0.45, 0.90, 0.20)
        ax.add_patch(patches.FancyBboxPatch((x, y), width, height,
                                            boxstyle="round,pad=0.012,rounding_size=0.020",
                                            linewidth=1.6, edgecolor="black", facecolor=card_color))
        if is_active:
            ax.add_patch(patches.FancyBboxPatch(
                (x-0.012, y-0.012), width+0.024, height+0.024,
                boxstyle="round,pad=0.012,rounding_size=0.022",
                linewidth=3.2, edgecolor=(1.0, 0.82, 0.10), facecolor='none', alpha=0.95
            ))

        name = uf["name"][:22]
        ax.text(x + 0.012, y + height*0.82, name,
                fontsize=11, fontweight="bold", ha="left", va="center", color="black")
        if uf.get("acc2", 0) > 0:
            ax.text(x + width - 0.012, y + height*0.82, f"☠{int(uf['acc2'])}%",
                    fontsize=10, ha="right", va="center", color="black")

        # HP-бар
        hp_w = width - 0.024
        hp_x, hp_y = x + 0.012, y + height*0.10
        ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w, HP_H, facecolor=(0.88,0.88,0.88), edgecolor='none'))
        frac = max(0.0, min(1.0, uf["hp"]/uf["maxhp"])) if uf["maxhp"] > 1e-9 else 0.0
        ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w*frac, HP_H, facecolor=(0.15,0.70,0.25), edgecolor='none'))
        ax.text(hp_x + hp_w/2, hp_y + HP_H/2, f"{int(max(0,uf['hp']))}/{int(uf['maxhp'])}",
                fontsize=9, ha="center", va="center", color="black")

        ax.text(x + width/2, y - 0.010, f"pos{front_pos}+{back_pos}", fontsize=8, ha="center", va="top", color=(0.2,0.2,0.2))

        if uf.get("paralyzed", 0):
            ax.text(x + width/2, y + height*0.46, "PAR", fontsize=10, fontweight="bold",
                    ha="center", va="center", color=(0.75, 0.05, 0.05))

    def draw_attack_arrow(ax, src_pos:int, dst_pos:int, team:str,
                          text: str|None = None,
                          style: str = "solid",
                          color: tuple|None = None,
                          alpha: float = 0.95,
                          curve: float = 0.18,
                          lw: float = 2.6):
        sx, sy = pos_center(src_pos)
        dx, dy = pos_center(dst_pos)

        vx, vy = dx - sx, dy - sy
        dist = math.hypot(vx, vy) + 1e-9
        pad = 0.06
        sx += vx/dist * pad
        sy += vy/dist * pad
        dx -= vx/dist * pad
        dy -= vy/dist * pad

        if color is None:
            color = (0.20, 0.35, 0.85) if team == "BLUE" else (0.85, 0.25, 0.25)

        con_style = f"arc3,rad={curve}" if abs(vx) > 0.01 and abs(vy) > 0.01 else "arc3,rad=0.0"
        arrow = FancyArrowPatch((sx, sy), (dx, dy),
                                arrowstyle="-|>",
                                mutation_scale=14,
                                linewidth=lw,
                                linestyle="--" if style == "dashed" else "solid",
                                color=color,
                                alpha=alpha,
                                connectionstyle=con_style)
        ax.add_patch(arrow)

        if text:
            mx, my = (sx + dx)/2, (sy + dy)/2
            ax.text(mx, my + 0.03, text, fontsize=10, fontweight="bold",
                ha="center", va="center", color=color)

    # --- регексы событий
    atk_re           = re.compile(r'^(RED|BLUE)\s+[^#]+#(\d+)\s+→\s+(RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)')
    kill_re          = re.compile(r'^✖\s+(RED|BLUE)\s+[^#]+#(\d+)\s+выведен из строя\.')
    vict_re          = re.compile(r'^🏆 Победа (RED|BLUE)!')
    blue_turn_re     = re.compile(r'^Ход BLUE:\s+[^#]+#(\d+)')
    red_turn_re      = re.compile(r'^RED ход:\s+[^#]+#(\d+)')
    blue_action_re   = re.compile(r'^BLUE действие:\s+[^#]+#(\d+)\s+→\s+pos(\d+)')
    # обновлён: допускаем "цель ..." или "цель с мин. HP ..."
    red_target_re    = re.compile(r'^RED ход:\s+[^#]+#(\d+).+цель.*pos(\d+)')
    mage_banner_re   = re.compile(r'^\w+\s+[^#]+#(\d+)\s+\((?:Mage|Dead dragon)\).+массов')
    blue_cant_re     = re.compile(r'^BLUE\s+[^#]+#(\d+).+не может достать pos(\d+)')
    poison_tick_re   = re.compile(r'^☠ Яд поражает (RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)')
    burn_tick_re     = re.compile(r'^🔥 Поджог поражает (RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)')
    uran_tick_re     = re.compile(r'^☢ Вода поражает (RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)')
    immune_dmg_re    = re.compile(r'^🛡 Иммунитет к урону')
    immune_stat_re   = re.compile(r'^🛡 Иммунитет к эффекту')
    resist_re        = re.compile(r'^🧿 Стойкость')
    miss_re          = re.compile(r'^💨 Промах:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+по\s+(RED|BLUE)\s+[^#]+#(\d+)\.')
    paralysis_apply_re = re.compile(r'^Паралич: (RED|BLUE)\s+[^#]+#(\d+)\s+лишает хода\s+(RED|BLUE)\s+[^#]+#(\d+)')
    paralysis_skip_re  = re.compile(r'^⛔ Паралич: (RED|BLUE)\s+[^#]+#(\d+)\s+пропускает ход\.')
    long_paralysis_pass_re = re.compile(r'^⛔ Пас: (RED|BLUE)\s+[^#]+#(\d+)\s+остаётся парализован\.')
    long_paralysis_recover_re = re.compile(r'^⛔ Долгий паралич: (RED|BLUE)\s+[^#]+#(\d+)\s+пропускает ход и восстанавливается\.')
    long_paralysis_chance_re = re.compile(r'^⚡ Шанс долгого паралича')
    long_paralysis_apply_re = re.compile(r'^⚡ Долгий паралич: (RED|BLUE)\s+[^#]+#(\d+)\s+накладывает долгий паралич на\s+(RED|BLUE)\s+[^#]+#(\d+)')

    def draw_board(arrows: list[dict] = None, headline: str = "", active_pos: int | None = None):
        ax.cla()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        ax.plot([0.02, 0.98], [0.50, 0.50], color=(0.6,0.6,0.6), lw=1.2, ls="--", alpha=0.7)
        ax.text(0.01, 0.96, "BLUE (top)", color=(0.2,0.35,0.8), fontsize=12, fontweight="bold", ha="left")
        ax.text(0.01, 0.04, "RED (bottom)", color=(0.8,0.25,0.25), fontsize=12, fontweight="bold", ha="left")

        arrows = arrows or []
        drawn_big_pairs = set()
        for front_pos in [3, 7, 9]:
            if front_pos in state and state[front_pos].get("big", False):
                draw_big_unit(ax, front_pos, is_active=(active_pos == front_pos))
                drawn_big_pairs.add((front_pos, front_pos+3))

        for pos in (BLUE_BACK + BLUE_FRONT + RED_FRONT + RED_BACK):
            if (pos-3, pos) in drawn_big_pairs or (pos, pos+3) in drawn_big_pairs:
                continue
            draw_unit(ax, pos, is_active=(active_pos == pos))

        for a in arrows:
            draw_attack_arrow(ax, a["src"], a["dst"], a["team"],
                              text=a.get("text"),
                              style=a.get("style","solid"),
                              color=a.get("color"),
                              alpha=a.get("alpha",0.95),
                              curve=a.get("curve",0.18),
                              lw=a.get("lw",2.6))

        if headline:
            ax.text(0.5, 0.52, headline[:140], fontsize=12, ha="center", va="bottom", color=(0.15,0.15,0.15))

        fig.canvas.draw()
        if handle is not None and hasattr(handle, "update"):
            handle.update(fig)
        else:
            try:
                fig.canvas.flush_events()
            except Exception:
                pass
            plt.pause(0.001)

    # --- начальный кадр
    current_actor_pos = None
    draw_board(headline="Начало боя", active_pos=current_actor_pos)

    last_selection = None  # dict: {"src":pos, "dst":pos, "team":"BLUE"/"RED"}

    for line in all_logs:
        arrows_now = []
        headline = ""

        m = blue_turn_re.match(line)
        if m:
            current_actor_pos = int(m.group(1))
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.05)
            continue

        m = red_turn_re.match(line)
        if m:
            current_actor_pos = int(m.group(1))
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.05)
            continue

        m = atk_re.match(line)
        if m:
            atk_team = m.group(1)
            atk_pos  = int(m.group(2))
            vic_pos  = int(m.group(4))
            dmg      = int(m.group(5))
            after    = int(m.group(7))
            _set_hp(vic_pos, after)
            current_actor_pos = atk_pos
            arrows_now.append({"src": atk_pos, "dst": vic_pos, "team": atk_team, "text": f"-{dmg}"})
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep(FRAME_DELAY * VISUAL_SPEED_MULT)
            continue

        m = poison_tick_re.match(line)
        if m:
            vic_pos = int(m.group(2)); after = int(m.group(5))
            _set_hp(vic_pos, after)
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
            continue

        m = burn_tick_re.match(line)
        if m:
            vic_pos = int(m.group(2)); after = int(m.group(5))
            _set_hp(vic_pos, after)
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
            continue

        m = uran_tick_re.match(line)
        if m:
            vic_pos = int(m.group(2)); after = int(m.group(5))
            _set_hp(vic_pos, after)
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
            continue

        m = blue_action_re.match(line)
        if m:
            src = int(m.group(1)); dst = int(m.group(2))
            last_selection = {"src": src, "dst": dst, "team": "BLUE"}
            current_actor_pos = src
            arrows_now.append({"src": src, "dst": dst, "team": "BLUE",
                               "style":"dashed", "alpha":0.65, "lw":2.0})
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        m = red_target_re.match(line)
        if m:
            src = int(m.group(1)); dst = int(m.group(2))
            last_selection = {"src": src, "dst": dst, "team": "RED"}
            current_actor_pos = src
            arrows_now.append({"src": src, "dst": dst, "team": "RED",
                               "style":"dashed", "alpha":0.65, "lw":2.0})
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        if immune_dmg_re.match(line) or immune_stat_re.match(line) or resist_re.match(line):
            if last_selection:
                arrows_now.append({"src": last_selection["src"], "dst": last_selection["dst"],
                                   "team": last_selection["team"], "color": (0.45,0.45,0.45),
                                   "style":"solid", "alpha":0.9, "text":"RESIST"})
                current_actor_pos = last_selection["src"]
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
            continue

        m = blue_cant_re.match(line)
        if m:
            src = int(m.group(1)); dst = int(m.group(2))
            arrows_now.append({"src": src, "dst": dst, "team": "BLUE",
                               "color": (0.5,0.5,0.5), "style":"dashed", "alpha":0.9, "text":"недосягаемо"})
            current_actor_pos = src
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
            continue

        m = miss_re.match(line)
        if m:
            atk_team = m.group(1)
            atk_pos  = int(m.group(2))
            vic_pos  = int(m.group(4))
            current_actor_pos = atk_pos
            arrows_now.append({"src": atk_pos, "dst": vic_pos, "team": atk_team,
                               "color": (0.5,0.5,0.5), "style":"solid", "alpha":0.9, "text":"MISS"})
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.15)
            continue

        m = kill_re.match(line)
        if m:
            pos = int(m.group(2))
            _set_hp(pos, 0)
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        if mage_banner_re.match(line):
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        m = long_paralysis_pass_re.match(line)
        if m:
            pos = int(m.group(2))
            current_actor_pos = pos
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        m = long_paralysis_recover_re.match(line)
        if m:
            pos = int(m.group(2))
            current_actor_pos = pos
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        if vict_re.match(line):
            headline = line
            current_actor_pos = None
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep(FRAME_DELAY * VISUAL_SPEED_MULT)
            continue

        # любые прочие события (в т.ч. 🧪 Шанс отравления ...)
        draw_board([], line, active_pos=current_actor_pos)
        time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)

    draw_board([], "Конец эпизода", active_pos=None)
    time.sleep(0.6 * VISUAL_SPEED_MULT)