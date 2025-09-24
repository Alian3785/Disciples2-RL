# ======================== ИМПОРТЫ ===========================
# Библиотеки стандартной библиотеки Python
import os
import re
import time
import shutil
import random
from operator import itemgetter

# Научный стек
import numpy as np

# Gymnasium (современная ветка OpenAI Gym)
import gymnasium as gym
from gymnasium import spaces

# ====================== ИСХОДНЫЕ ДАННЫЕ =====================
# В этом блоке определяем состав двух команд и константы боя.
# По условию 12 юнитов: RED (позиции 1..6), BLUE (позиции 7..12).
# Все базовые характеристики исходно: Damage=20, Health=60 (у шаблонов — больше).
# Initiative определяет порядок хода в раунде.

UNITS_RED = [
    {"Name": "рыцарь1",  "Initiative": 88, "BaseInitiative": 88, "team": "red", "position": 1,  "stand": "ahead",
     "Type": "Mage", "Damage": 10, "Health": 60, "Damage2": 0, "maxhealth": 1020, "Armor": 0, "Accuracy": 80, "Accuracy2": 0,
     "Immunity": [], "Resilience": ["Mind", "Fire"], "AttackType1": "Mind", "AttackType2": "", "big": True,
     "Twohits": 0, "Burn": 5, "Freeze": 0, "Poison": 3, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": ["Mind", "Fire"]},

    {"Name": "рыцарь2",  "Initiative": 55, "BaseInitiative": 55, "team": "red", "position": 2,  "stand": "ahead",
     "Type": "Archer", "Damage": 20, "Health": 60, "Damage2": 18, "maxhealth": 306, "Armor": 0, "Accuracy": 100, "Accuracy2": 45,
     "Immunity": [], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "poison", "big": False,
     "Twohits": 0, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": []},

    {"Name": "рыцарь3",  "Initiative": 22, "BaseInitiative": 22, "team": "red", "position": 3,  "stand": "ahead",
     "Type": "Asterot", "Damage": 30, "Health": 60, "Damage2": 0, "maxhealth": 170, "Armor": 65, "Accuracy": 80, "Accuracy2": 50,
     "Immunity": ["Poison"], "Resilience": ["Mind"], "AttackType1": "Weapon", "AttackType2": "", "big": True,
     "Twohits": 1, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": ["Mind"]},

    {"Name": "рыцарь4", "Initiative": 60, "BaseInitiative": 60, "team": "red", "position": 4, "stand": "behind",
     "Type": "Archer", "Damage": 20, "Health": 60, "Damage2": 12, "maxhealth": 60, "Armor": 0, "Accuracy": 80, "Accuracy2": 30,
     "Immunity": [], "Resilience": [], "AttackType1": "Earth", "AttackType2": "poison", "big": False,
     "Twohits": 0, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": []},

    {"Name": "рыцарь5", "Initiative": 47, "BaseInitiative": 47, "team": "red", "position": 5, "stand": "behind",
     "Type": "Archer", "Damage": 20, "Health": 60, "Damage2": 0, "maxhealth": 0, "Armor": 0, "Accuracy": 80, "Accuracy2": 50,
     "Immunity": [], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "", "big": False,
     "Twohits": 0, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": []},

    {"Name": "рыцарь6", "Initiative": 75, "BaseInitiative": 75, "team": "red", "position": 6, "stand": "behind",
     "Type": "PoisonDragon", "Damage": 12, "Health": 60, "Damage2": 8, "maxhealth": 0, "Armor": 0, "Accuracy": 90, "Accuracy2": 75,
     "Immunity": [], "Resilience": [], "AttackType1": "Poison", "AttackType2": "poison", "big": False,
     "Twohits": 0, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": []},
]

UNITS_BLUE = [
    {"Name": "рыцарь4",  "Initiative": 88, "BaseInitiative": 88, "team": "blue", "position": 7,  "stand": "ahead",
     "Type": "Mage", "Damage": 10, "Health": 60, "Damage2": 0, "maxhealth": 1020, "Armor": 0, "Accuracy": 80, "Accuracy2": 0,
     "Immunity": [], "Resilience": ["Mind", "Fire"], "AttackType1": "Mind", "AttackType2": "", "big": True,
     "Twohits": 0, "Burn": 2, "Freeze": 4, "Poison": 1, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": ["Mind", "Fire"]},

    {"Name": "рыцарь5",  "Initiative": 55, "BaseInitiative": 55, "team": "blue", "position": 8,  "stand": "ahead",
     "Type": "Archer", "Damage": 20, "Health": 60, "Damage2": 18, "maxhealth": 306, "Armor": 0, "Accuracy": 100, "Accuracy2": 45,
     "Immunity": [], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "poison", "big": False,
     "Twohits": 0, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": []},

    {"Name": "рыцарь6",  "Initiative": 22, "BaseInitiative": 22, "team": "blue", "position": 9,  "stand": "ahead",
     "Type": "Warrior", "Damage": 30, "Health": 60, "Damage2": 0, "maxhealth": 170, "Armor": 65, "Accuracy": 80, "Accuracy2": 50,
     "Immunity": ["Poison"], "Resilience": ["Mind"], "AttackType1": "Weapon", "AttackType2": "", "big": True,
     "Twohits": 0, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": ["Mind"]},

    {"Name": "рыцарь10", "Initiative": 60, "BaseInitiative": 60, "team": "blue", "position": 10, "stand": "behind",
     "Type": "Archer", "Damage": 20, "Health": 60, "Damage2": 12, "maxhealth": 60, "Armor": 0, "Accuracy": 80, "Accuracy2": 30,
     "Immunity": [], "Resilience": [], "AttackType1": "Earth", "AttackType2": "poison", "big": False,
     "Twohits": 0, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": []},

    {"Name": "рыцарь11", "Initiative": 47, "BaseInitiative": 47, "team": "blue", "position": 11, "stand": "behind",
     "Type": "Archer", "Damage": 20, "Health": 60, "Damage2": 0, "maxhealth": 0, "Armor": 0, "Accuracy": 80, "Accuracy2": 50,
     "Immunity": [], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "", "big": False,
     "Twohits": 0, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": []},

    {"Name": "рыцарь12", "Initiative": 75, "BaseInitiative": 75, "team": "blue", "position": 12, "stand": "behind",
     "Type": "Death", "Damage": 18, "Health": 60, "Damage2": 12, "maxhealth": 0, "Armor": 0, "Accuracy": 90, "Accuracy2": 70,
     "Immunity": [], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "poison", "big": False,
     "Twohits": 0, "Burn": 0, "Freeze": 0, "Poison": 0, "Paralized": 0, "Longparalized": 0, "Stoned": 0, "Runningaway": 0, "original_resilience": []},
]

# Два диапазона позиций: RED (1..6) и BLUE (7..12)
RED_POSITIONS  = list(range(1, 7))
BLUE_POSITIONS = list(range(7, 13))

# ----------------- СЛОВАРИ ДЛЯ КОДИРОВАНИЯ ПОЛЕЙ -----------------
ALL_UNITS = UNITS_RED + UNITS_BLUE

# team/stand как категориальные
TEAM2ID  = {"red": 0, "blue": 1}
STAND2ID = {"ahead": 0, "behind": 1}

# Типы юнитов (Type)
TYPE_VOCAB = sorted({u.get("Type", "") for u in ALL_UNITS})
TYPE2ID = {t: i for i, t in enumerate(TYPE_VOCAB)}

# Типы атак (AttackType1/AttackType2). Пустую строку считаем отдельной категорией.
ATTACK_TYPES = sorted({(u.get("AttackType1") or "") for u in ALL_UNITS} |
                      {(u.get("AttackType2") or "") for u in ALL_UNITS})
ATK2ID = {t: i for i, t in enumerate(ATTACK_TYPES)}


# Иммунитеты/Резисты — мультихоты. Нормализуем к нижнему регистру.
IMM_TOKENS = sorted({x.lower() for u in ALL_UNITS for x in (u.get("Immunity") or [])} |
                    {"death", "poison", "weapon", "mind", "fire"})  # гарантия стабильного порядка
RES_TOKENS = sorted({x.lower() for u in ALL_UNITS for x in (u.get("Resilience") or [])} |
                    {"mind", "fire"})
IMM2IDX = {t: i for i, t in enumerate(IMM_TOKENS)}
RES2IDX = {t: i for i, t in enumerate(RES_TOKENS)}

# Имя юнита (Name) — как ID (чтобы «добавить» свойство Name в obs).
NAME_VOCAB = sorted({u.get("Name", "") for u in ALL_UNITS})
NAME2ID = {n: i for i, n in enumerate(NAME_VOCAB)}

# Числовые максимумы для верхних границ Box
MAX_BASE_INIT = max(u.get("BaseInitiative", 0) for u in ALL_UNITS)
MAX_HEALTH    = max(u.get("Health", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_MAXHP     = max(u.get("maxhealth", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_DAMAGE    = max(u.get("Damage", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_DAMAGE2   = max(u.get("Damage2", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_ARMOR     = max(u.get("Armor", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_ACC       = max(u.get("Accuracy", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_ACC2      = max(u.get("Accuracy2", 0) for u in ALL_UNITS) if ALL_UNITS else 0
# Дополнительные показатели состояний
MAX_TWOHITS   = max(u.get("Twohits", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_BURN      = max(u.get("Burn", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_FREEZE    = max(u.get("Freeze", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_POISON    = max(u.get("Poison", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_PARALIZED = max(u.get("Paralized", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_LONGPARALIZED = max(u.get("Longparalized", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_STONED    = max(u.get("Stoned", 0) for u in ALL_UNITS) if ALL_UNITS else 0
MAX_RUNNINGAWAY = max(u.get("Runningaway", 0) for u in ALL_UNITS) if ALL_UNITS else 0

# ----------------- СХЕМА ВЕКТОРА ПРИЗНАКОВ НА ПОЗИЦИЮ -----------------
# Порядок признаков на каждую позицию (все float32):
# [Health, Initiative, BaseInitiative, Damage, Damage2, maxhealth, Armor,
#  Accuracy, Accuracy2, team_id, position, stand_id, type_id, attack1_id, attack2_id,
#  big_flag, name_id,
#  Twohits, Burn, Freeze, Poison, Paralized, Longparalized, Stoned, Runningaway,
#  imm_* (len(IMM_TOKENS)), res_* (len(RES_TOKENS))]
FEATURES_STATIC = 25  # до мультихотов (добавлены 8 статусных показателей)
FEATURES_IMM    = len(IMM_TOKENS)
FEATURES_RES    = len(RES_TOKENS)
FEATURES_PER_POS = FEATURES_STATIC + FEATURES_IMM + FEATURES_RES
TOTAL_OBS_SIZE   = 12 * FEATURES_PER_POS

# ==================== КЛАСС СРЕДЫ GYMNASIUM ==================
class BattleEnv(gym.Env):
    """
    КАСТОМНАЯ СРЕДА «RED vs BLUE» ДЛЯ RL

    Наблюдение (obs):
        Вектор длины TOTAL_OBS_SIZE = 12 * FEATURES_PER_POS (float32).
        Для каждой позиции 1..12 конкатенируем признаки:
            Health, Initiative, BaseInitiative, Damage, Damage2, maxhealth, Armor,
            Accuracy, Accuracy2, team_id, position, stand_id, type_id,
            attack1_id, attack2_id, big_flag, name_id,
            imm_* (мультихот по IMM_TOKENS), res_* (мультихот по RES_TOKENS)

    Действие (action):
        Discrete(6) — индекс 0..5 -> удар по позиции RED_POSITIONS[action], т.е. по позициям 1..6.

    Порядок ходов:
        — Внутри раунда ходят по инициативе (чем выше, тем раньше).
        — Как только юнит сходил в текущем раунде, его Initiative обнуляется (в этом раунде он больше не ходит).
        — Когда у всех живых Initiative = 0, начинается новый раунд, и живым Initiative восстанавливается до BaseInitiative.

    Завершение эпизода:
        — Сразу после каждого удара проверяется, остались ли живые у обеих сторон.
        — Как только у стороны не осталось живых — эпизод терминален (terminated=True).

    Награды:
        reward_step — базовая награда за шаг;
        reward_win / reward_loss — за завершение эпизода;
        reward_illegal — штраф за удар по пустой/мертвой цели.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_step: float = 0.0,
        log_enabled: bool = False,
        reward_illegal: float = -0.1,   # <--- новый параметр штрафа
    ):
        super().__init__()
        # Параметры наград
        self.reward_win     = float(reward_win)
        self.reward_loss    = float(reward_loss)
        self.reward_step    = float(reward_step)
        self.reward_illegal = float(reward_illegal)

        # Включаем человекочитаемые логи по флагу конструктора
        self.log_enabled = bool(log_enabled)

        # ГСЧ: без фиксированного seed, каждое выполнение случайно
        self.rng = random.Random()

        # Пространство действий: выбор целевой позиции среди 6 красных слотов
        self.action_space = spaces.Discrete(6)

        # Пространство наблюдений: TOTAL_OBS_SIZE float32
        low_per_pos = np.zeros(FEATURES_PER_POS, dtype=np.float32)
        # Верхние границы для статической части:
        high_static = np.array([
            max(1, MAX_HEALTH),           # Health
            max(1, MAX_BASE_INIT),        # Initiative
            max(1, MAX_BASE_INIT),        # BaseInitiative
            max(1, MAX_DAMAGE),           # Damage
            max(1, MAX_DAMAGE2),          # Damage2
            max(1, MAX_MAXHP),            # maxhealth
            max(1, MAX_ARMOR),            # Armor
            max(1, MAX_ACC),              # Accuracy
            max(1, MAX_ACC2),             # Accuracy2
            1,                            # team_id (0..1)
            12,                           # position
            1,                            # stand_id (0..1)
            max(1, len(TYPE2ID)-1),       # type_id
            max(1, len(ATK2ID)-1),        # attack1_id
            max(1, len(ATK2ID)-1),        # attack2_id
            1,                            # big_flag
            max(1, len(NAME2ID)-1),       # name_id
            max(1, MAX_TWOHITS),          # Twohits
            max(1, MAX_BURN),             # Burn
            max(1, MAX_FREEZE),           # Freeze
            max(1, MAX_POISON),           # Poison
            max(1, MAX_PARALIZED),        # Paralized
            max(1, MAX_LONGPARALIZED),    # Longparalized
            max(1, MAX_STONED),           # Stoned
            max(1, MAX_RUNNINGAWAY),      # Runningaway
        ], dtype=np.float32)

        high_per_pos = np.concatenate([
            high_static,
            np.ones(FEATURES_IMM, dtype=np.float32),  # мультихот иммунитетов
            np.ones(FEATURES_RES, dtype=np.float32),  # мультихот резистов
        ]).astype(np.float32)

        low   = np.tile(low_per_pos, 12)
        high  = np.tile(high_per_pos, 12)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Текущее состояние боя
        self.combined = None                   # список из 12 dict-юнитов (копии исходных)
        self.round_no = None                   # номер текущего раунда
        self.winner = None                     # "red" | "blue" | None
        self.current_blue_attacker_pos = None  # позиция синего, который ждёт действие агента

        # Буфер человекочитаемых логов (строки), используем только если log_enabled=True
        self._pretty_events = []

    # ------------------ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ------------------

    def _alive(self, u) -> bool:
        """Вернуть True, если юнит жив (HP > 0)."""
        return u["Health"] > 0

    def _team_alive(self, team: str) -> bool:
        """Есть ли у команды team хотя бы один живой юнит."""
        return any(self._alive(u) and u["team"] == team for u in self.combined)

    def _unit_by_position(self, pos: int):
        """Вернуть словарь-юнит по позиции pos (1..12), либо None, если пусто."""
        return next((u for u in self.combined if u["position"] == pos), None)

    def _live_positions_of(self, team: str):
        """Вернуть список позиций живых юнитов указанной команды."""
        return [u["position"] for u in self.combined if u["team"] == team and self._alive(u)]

    def _log(self, s: str):
        """Добавить строку в человекочитаемый лог (используется при тестовом прогоне)."""
        if self.log_enabled:
            self._pretty_events.append(s)

    def pop_pretty_events(self):
        """
        Забрать накопленные логи и очистить буфер.
        Удобно вызывать сразу после reset()/step(), чтобы получить «кадр» логов.
        """
        out = self._pretty_events[:]
        self._pretty_events.clear()
        return out

    def _reset_state(self):
        """
        Полная инициализация состояния эпизода:
        — Клонируем исходные словари юнитов (чтобы не мутировать глобальные).
        — Устанавливаем текущую Initiative равной BaseInitiative.
        — Сбрасываем флаги: номер раунда, победитель, «ожидающий BLUE».
        """
        self.combined = [u.copy() for u in (UNITS_RED + UNITS_BLUE)]
        for u in self.combined:
            u["Initiative"] = u["BaseInitiative"]
            # Initialize original_resilience for each unit
            u["original_resilience"] = u.get("Resilience", []).copy()
        self.round_no = 1
        self.winner = None
        self.current_blue_attacker_pos = None
        self._log(f"Эпизод начат. Раунд {self.round_no}.")

    def _apply_status_effects(self, unit):
        """
        Применить эффекты состояния перед ходом юнита.
        Вычитает урон из здоровья и с вероятностью 30% обнуляет эффекты.
        """
        if not self._alive(unit):
            return

        damage_taken = 0
        effects_applied = []

        # Применяем урон от эффектов
        if unit.get("Burn", 0) > 0:
            damage_taken += unit["Burn"]
            effects_applied.append(f"Burn({unit['Burn']})")

        if unit.get("Freeze", 0) > 0:
            damage_taken += unit["Freeze"]
            effects_applied.append(f"Freeze({unit['Freeze']})")

        if unit.get("Poison", 0) > 0:
            damage_taken += unit["Poison"]
            effects_applied.append(f"Poison({unit['Poison']})")

        # Вычитаем урон из здоровья
        if damage_taken > 0:
            unit["Health"] -= damage_taken
            effects_str = ", ".join(effects_applied)
            self._log(f"💀 {unit['team'].upper()} {unit['Name']}#{unit['position']}: {effects_str} наносит {damage_taken} урона")

            # Если юнит умер от эффектов, сбрасываем его инициативу
            if unit["Health"] <= 0:
                unit["Initiative"] = 0
                self._log(f"✖ {unit['team'].upper()} {unit['Name']}#{unit['position']} погибает от эффектов состояния!")

        # Каждый эффект отдельно обнуляется с вероятностью 30%
        effects_cleared = []
        if unit.get("Burn", 0) > 0 and self.rng.random() < 0.3:
            unit["Burn"] = 0
            effects_cleared.append("Burn")
        if unit.get("Freeze", 0) > 0 and self.rng.random() < 0.3:
            unit["Freeze"] = 0
            effects_cleared.append("Freeze")
        if unit.get("Poison", 0) > 0 and self.rng.random() < 0.3:
            unit["Poison"] = 0
            effects_cleared.append("Poison")

        if effects_cleared:
            effects_str = ", ".join(effects_cleared)
            self._log(f"✨ {unit['team'].upper()} {unit['Name']}#{unit['position']}: эффекты {effects_str} исчезают!")

    def _candidates(self):
        """
        Вернуть список юнитов, которые могут ходить в текущем раунде:
        — юнит должен быть жив, и
        — его текущая Initiative должна быть > 0.
        """
        return [u for u in self.combined if self._alive(u) and u["Initiative"] > 0]

    def _pop_next(self):
        """
        Выбор следующего ходящего:
        1) Берём кандидатов (живые, INI>0).
        2) Перемешиваем (решаем ничьи по инициативе случайно).
        3) Сортируем по инициативе по убыванию.
        4) Возвращаем верхнего.
        Если кандидатов нет — возвращаем None (конец раунда).
        """
        cand = self._candidates()
        if not cand:
            return None
        self.rng.shuffle(cand)
        cand.sort(key=itemgetter("Initiative"), reverse=True)
        return cand[0]

    def _end_round_restore(self):
        """
        Конец раунда: всем ЖИВЫМ возвращаем Initiative к BaseInitiative,
        мёртвым оставляем 0. Увеличение номера раунда происходит снаружи.
        Также сбрасываем Twohits для всех Asterot.
        """
        for u in self.combined:
            u["Initiative"] = u["BaseInitiative"] if self._alive(u) else 0
            # Сброс Twohits для Asterot
            if u.get("Type") == "Asterot" and self._alive(u):
                u["Twohits"] = 1
        self._log("Восстановление инициативы. Новый раунд.")

    # ------------------- УДАР И ПРОВЕРКА ПОБЕДЫ -------------------

    def _enemy_rows(self, team):
        """Вернуть списки позиций фронта и тыла врага."""
        if team == "red":
            ahead = {0: 7, 1: 8, 2: 9}   # фронт синих
            behind = {0: 10, 1: 11, 2: 12} # тыл синих
        else:
            ahead = {0: 1, 1: 2, 2: 3}   # фронт красных
            behind = {0: 4, 1: 5, 2: 6}  # тыл красных
        return ahead, behind

    def _col_of(self, position):
        """Вернуть колонку (0,1,2) для позиции."""
        return (position - 1) % 3

    def _warrior_near_far_cols(self, col):
        """Вернуть (ближние_колонки, дальние_колонки) для воина в колонке col."""
        if col == 0:  # Левая колонка
            return [0, 1], [2]  # Ближние: 0,1; дальняя: 2
        elif col == 2:  # Правая колонка
            return [1, 2], [0]  # Ближние: 1,2; дальняя: 0
        else:  # Центральная колонка (col == 1)
            return [0, 1, 2], []  # Все три, нет "дальней"

    def _warrior_allowed_targets(self, attacker):
        """
        Вернуть позиции, по которым воин МОЖЕТ ударить сейчас.
        - В тылу не атакует, пока жив любой союзный воин во фронте.
        - Во фронт врага сначала бьёт смежных; «дальний» доступен, когда оба смежных мертвы.
        - Если фронт врага выбит — правила применяются к тылу врага.
        """
        assert attacker.get("Type") in ["Warrior", "Asterot"]
        # Блок для воинов в тылу
        if attacker.get("stand") == "behind":
            if any(self._alive(u) and u["team"] == attacker["team"] and u.get("Type") in ["Warrior", "Asterot"] and u.get("stand") == "ahead"
                   for u in self.combined):
                return []

        enemy_team = "blue" if attacker["team"] == "red" else "red"
        ahead, behind = self._enemy_rows(attacker["team"])
        col = self._col_of(attacker["position"])
        near_cols, far_cols = self._warrior_near_far_cols(col)

        def alive(pos):
            uu = self._unit_by_position(pos)
            return uu is not None and self._alive(uu)

        # Сначала фронт
        near = [ahead[c] for c in near_cols if alive(ahead[c])]
        if near:
            return near
        far = [ahead[c] for c in far_cols if alive(ahead[c])]
        if far:
            return far

        # Если фронт врага выбит — тыл
        near2 = [behind[c] for c in near_cols if alive(behind[c])]
        if near2:
            return near2
        far2 = [behind[c] for c in far_cols if alive(behind[c])]
        return far2

    def _pick_warrior_target_for_red(self, attacker):
        """RED-воин выбирает цель с минимальным HP из доступных, при равенстве — случайная."""
        options = self._warrior_allowed_targets(attacker)
        if not options:
            return None
        pairs = [(self._unit_by_position(p)["Health"], p) for p in options]
        min_hp = min(h for h, _ in pairs)
        pool = [p for h, p in pairs if h == min_hp]
        return self.rng.choice(pool)

    def _attack(self, attacker, target_pos: int):
        """
        Выполнить атаку «attacker» по позиции target_pos:
        — Если атакующий — маг, то атака наносится всем вражеским юнитам с индивидуальной проверкой промаха, иммунитета и resistance.
        — Если атакующий — воин, то атака наносится только на разрешенные позиции.
        — Если атакующий — обычный юнит, то атака наносится только по указанной позиции.
        — Если после удара HP <= 0, сбрасываем Initiative жертвы (в этом раунде она уже не сходится).
        — Пишем человекочитаемую запись (если включён лог).
        """
        # Определяем цели атаки
        if attacker.get("Type") == "Mage":
            # Маг атакует всех вражеских юнитов
            enemy_team = "blue" if attacker["team"] == "red" else "red"
            targets = [u for u in self.combined if u["team"] == enemy_team and self._alive(u)]
            if not targets:
                self._log(f"{attacker['team'].upper()} {attacker['Name']}#{attacker['position']} (Маг): нет целей для атаки.")
                return

            self._log(f"🔮 {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} (Маг) атакует всех вражеских юнитов!")

            # Атакуем каждого вражеского юнита индивидуально
            for victim in targets:
                self._attack_single_target(attacker, victim["position"])
        elif attacker.get("Type") == "PoisonDragon":
            # PoisonDragon атакует всех вражеских юнитов как Mage
            enemy_team = "blue" if attacker["team"] == "red" else "red"
            targets = [u for u in self.combined if u["team"] == enemy_team and self._alive(u)]
            if not targets:
                self._log(f"{attacker['team'].upper()} {attacker['Name']}#{attacker['position']} (Ядовитый Дракон): нет целей для атаки.")
                return

            self._log(f"🐉 {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} (Ядовитый Дракон) атакует всех вражеских юнитов!")

            # Атакуем каждого вражеского юнита индивидуально
            for victim in targets:
                self._attack_single_target(attacker, victim["position"])
        elif attacker.get("Type") in ["Warrior", "Asterot"]:
            # Воин и Астерот могут атаковать только разрешенные цели
            allowed_targets = self._warrior_allowed_targets(attacker)

            # Проверяем, является ли выбранная агентом цель доступной
            if target_pos not in allowed_targets:
                # Цель недоступна - штраф за неправильный выбор
                unit_type = "Астерот" if attacker.get("Type") == "Asterot" else "Воин"
                self._log(f"❌ {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} ({unit_type}): цель pos{target_pos} недоступна!")
                return True  # Флаг, что был применен штраф

            # Цель доступна - выполняем атаку
            unit_type = "Астерот" if attacker.get("Type") == "Asterot" else "Воин"
            self._log(f"⚔️ {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} ({unit_type}) атакует pos{target_pos}!")
            self._attack_single_target(attacker, target_pos)
            return False  # Флаг, что штраф не применялся
        else:
            # Обычная атака по одной цели
            self._attack_single_target(attacker, target_pos)

    def _attack_single_target(self, attacker, target_pos: int):
        """
        Выполнить атаку по одной цели с проверкой промаха, иммунитета и resistance.
        """
        victim = self._unit_by_position(target_pos) if target_pos is not None else None
        if victim is not None and self._alive(victim):
            # Miss check: attacker misses with (100 - Accuracy)% probability
            attacker_accuracy = attacker.get("Accuracy", 0)
            if self.rng.random() > (attacker_accuracy / 100):
                self._log(f"❌ {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} → "
                          f"{victim['team'].upper()} {victim['Name']}#{victim['position']}: промах!")
                return

            # Immunity check: if victim is immune to attacker's AttackType1, no damage is applied
            atk1 = (attacker.get("AttackType1") or "Weapon")
            victim_imm = victim.get("Immunity") or []
            if atk1 in victim_imm:
                self._log(
                    f"🛡 Иммунитет к урону '{atk1}' — {victim['team'].upper()} {victim['Name']}#{victim['position']}: урон 0."
                )
                return

            # Resilience check: if victim has resilience to attacker's AttackType1, no damage is applied (only for first attack)
            victim_res = victim.get("Resilience") or []
            if atk1 in victim_res:
                self._log(
                    f"🛡 Резистентность к урону '{atk1}' — {victim['team'].upper()} {victim['Name']}#{victim['position']}: урон 0 (первая атака этого типа)."
                )
                # Remove this attack type from victim's Resilience (temporary)
                victim["Resilience"] = [r for r in victim_res if r != atk1]
                return

            before = victim["Health"]
            victim["Health"] -= attacker["Damage"]
            after = victim["Health"]
            self._log(f"{attacker['team'].upper()} {attacker['Name']}#{attacker['position']} → "
                      f"{victim['team'].upper()} {victim['Name']}#{victim['position']}: {attacker['Damage']} "
                      f"({before}→{max(0, after)})")
            if victim["Health"] <= 0:
                victim["Initiative"] = 0
                self._log(f"✖ {victim['team'].upper()} {victim['Name']}#{victim['position']} выведен из строя.")

            # Логика отравления для типов Death и PoisonDragon - срабатывает ТОЛЬКО ПОСЛЕ успешной основной атаки и если цель выжила
            if attacker.get("Type") in ["Death", "PoisonDragon"] and attacker.get("Damage2", 0) > 0 and victim["Health"] > 0:
                # Проверка Accuracy2 для отравления (аналогично проверке на промах)
                attacker_accuracy2 = attacker.get("Accuracy2", 0)
                if self.rng.random() > (attacker_accuracy2 / 100):
                    self._log(f"❌ {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} → "
                              f"{victim['team'].upper()} {victim['Name']}#{victim['position']}: отравление не удалось!")
                else:
                    # Проверки для AttackType2
                    atk2 = (attacker.get("AttackType2") or "")
                    victim_imm = victim.get("Immunity") or []
                    victim_res = victim.get("Resilience") or []

                    # Immunity check for AttackType2
                    if atk2 in victim_imm:
                        self._log(f"🛡 Иммунитет к урону '{atk2}' — {victim['team'].upper()} {victim['Name']}#{victim['position']}: отравление не сработало.")
                    elif atk2 in victim_res:
                        self._log(f"🛡 Резистентность к урону '{atk2}' — {victim['team'].upper()} {victim['Name']}#{victim['position']}: отравление не сработало.")
                        # Remove this attack type from victim's Resilience (temporary)
                        victim["Resilience"] = [r for r in victim_res if r != atk2]
                    else:
                        # Проверка на существующий Poison
                        if victim.get("Poison", 0) == 0:
                            victim["Poison"] += attacker["Damage2"]
                            self._log(f"☠ {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} → "
                                      f"{victim['team'].upper()} {victim['Name']}#{victim['position']}: отравление на {attacker['Damage2']}!")
                        else:
                            self._log(f"❌ {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} → "
                                      f"{victim['team'].upper()} {victim['Name']}#{victim['position']}: цель уже отравлена!")

        else:
            self._log(f"{attacker['team'].upper()} {attacker['Name']}#{attacker['position']} бьёт pos{target_pos}: цели нет/мертва.")

    def _check_victory_after_hit(self):
        """
        Сразу после любого удара проверяем, остались ли живые у обеих сторон.
        При отсутствии живых у одной из сторон — фиксируем победителя и логируем событие.
        """
        if not self._team_alive("blue"):
            self.winner = "red"
            self._log("🏆 Победа RED!")
        elif not self._team_alive("red"):
            self.winner = "blue"
            self._log("🏆 Победа BLUE!")

    # --------------- АВТОДОКРУТКА ДО ХОДА BLUE ----------------

    def _advance_until_blue_turn(self) -> bool:
        """
        Прокручиваем бой автоматически, пока не наступит очередь живого BLUE,
        или пока бой не завершится.
        Возвращает:
            True  — если остановились на ходу BLUE (среда ждёт действие агента),
            False — если бой завершился до этого момента.
        """
        while self._team_alive("red") and self._team_alive("blue"):
            nxt = self._pop_next()
            if nxt is None:
                # Все живые уже сходили — начинаем новый раунд
                self._log(f"— Конец раунда {self.round_no}.")
                self._end_round_restore()
                self.round_no += 1
                self._log(f"— Начало раунда {self.round_no}.")
                continue

            if nxt["team"] == "blue":
                # Пришёл ход живого BLUE — выходим и ждём action от агента
                self.current_blue_attacker_pos = nxt["position"]
                self._log(f"Ход BLUE: {nxt['Name']}#{nxt['position']} (иниц {nxt['Initiative']}). Ожидание действия.")
                return True

            # === Ход RED (противник): случайный выбор живой цели среди BLUE ===
            # Применяем эффекты состояния перед ходом
            self._apply_status_effects(nxt)

            nxt["Initiative"] = 0  # в рамках текущего раунда этот юнит сходил
            live_blue_positions = self._live_positions_of("blue")
            target_pos = self.rng.choice(live_blue_positions) if live_blue_positions else None
            self._log(f"RED ход: {nxt['Name']}#{nxt['position']} → случайная цель pos{target_pos}.")
            penalty_applied = self._attack(nxt, target_pos)
            if penalty_applied:
                # Штраф уже применен в _attack, пропускаем ход
                continue
            self._check_victory_after_hit()
            if self.winner is not None:
                return False

            # === Логика двойного хода для Asterot ===
            if nxt.get("Type") == "Asterot" and nxt.get("Twohits", 0) == 1:
                # Астерот делает двойной ход без эффектов состояния
                self._log(f"⭐ {nxt['team'].upper()} {nxt['Name']}#{nxt['position']} (Астерот): активирует двойной ход!")
                nxt["Twohits"] = 0  # сбрасываем Twohits

                # Делаем второй ход БЕЗ применения эффектов состояния
                live_blue_positions = self._live_positions_of("blue")
                target_pos = self.rng.choice(live_blue_positions) if live_blue_positions else None
                self._log(f"RED ход (двойной): {nxt['Name']}#{nxt['position']} → случайная цель pos{target_pos}.")
                penalty_applied = self._attack(nxt, target_pos)
                if penalty_applied:
                    continue
                self._check_victory_after_hit()
                if self.winner is not None:
                    return False

        # Боевые действия закончились до очереди BLUE
        return False

    # -------------------- НАБЛЮДЕНИЕ ДЛЯ АГЕНТА --------------------

    def _unit_obs(self, u: dict) -> np.ndarray:
        """Собрать вектор признаков для одного юнита согласно схеме FEATURES_PER_POS."""
        health          = float(max(0, u.get("Health", 0)))
        initiative      = float(max(0, u.get("Initiative", 0)))
        base_initiative = float(max(0, u.get("BaseInitiative", 0)))
        damage          = float(max(0, u.get("Damage", 0)))
        damage2         = float(max(0, u.get("Damage2", 0)))
        maxhp           = float(max(0, u.get("maxhealth", 0)))
        armor           = float(max(0, u.get("Armor", 0)))
        acc1            = float(max(0, u.get("Accuracy", 0)))
        acc2            = float(max(0, u.get("Accuracy2", 0)))
        team_id         = float(TEAM2ID.get(u.get("team", "red"), 0))
        position        = float(u.get("position", 0))
        stand_id        = float(STAND2ID.get(u.get("stand", "ahead"), 0))
        type_id         = float(TYPE2ID.get(u.get("Type", ""), 0))
        atk1_id         = float(ATK2ID.get(u.get("AttackType1", "") or "", 0))
        atk2_id         = float(ATK2ID.get(u.get("AttackType2", "") or "", 0))
        big_flag        = float(1.0 if u.get("big", False) else 0.0)
        name_id         = float(NAME2ID.get(u.get("Name", ""), 0))

        imm_vec = np.zeros(FEATURES_IMM, dtype=np.float32)
        for tok in (u.get("Immunity") or []):
            t = str(tok).lower()
            if t in IMM2IDX:
                imm_vec[IMM2IDX[t]] = 1.0

        res_vec = np.zeros(FEATURES_RES, dtype=np.float32)
        for tok in (u.get("Resilience") or []):
            t = str(tok).lower()
            if t in RES2IDX:
                res_vec[RES2IDX[t]] = 1.0

        head = np.array([
            health, initiative, base_initiative, damage, damage2,
            maxhp, armor, acc1, acc2, team_id, position, stand_id,
            type_id, atk1_id, atk2_id, big_flag, name_id,
            float(u.get("Twohits", 0)), float(u.get("Burn", 0)), float(u.get("Freeze", 0)),
            float(u.get("Poison", 0)), float(u.get("Paralized", 0)), float(u.get("Longparalized", 0)),
            float(u.get("Stoned", 0)), float(u.get("Runningaway", 0))
        ], dtype=np.float32)

        return np.concatenate([head, imm_vec, res_vec]).astype(np.float32)

    def _obs(self) -> np.ndarray:
        """Построить наблюдение: плоский вектор (TOTAL_OBS_SIZE,)."""
        vec = []
        for pos in RED_POSITIONS + BLUE_POSITIONS:
            u = self._unit_by_position(pos)
            vec.append(self._unit_obs(u))
        return np.concatenate(vec).astype(np.float32)

    # ------------------------ API GYMNASIUM ------------------------

    def reset(self, *, seed=None, options=None):
        """
        Сброс эпизода:
          1) Инициализируем состояние боя.
          2) Автоматически докручиваем до первого хода BLUE (или терминала).
          3) Возвращаем начальное наблюдение и пустой info (по стандарту Gymnasium).
        """
        # seed параметр намеренно игнорируется, чтобы сохранить случайность
        self._reset_state()
        self._advance_until_blue_turn()
        return self._obs(), {}

    def step(self, action):
        """
        Один шаг среды (ход BLUE):
          — Докрутка до BLUE при необходимости.
          — Применение действия агента (цель среди 1..6).
          — Явный штраф reward_illegal за удар по пустой/мертвой цели.
          — Докрутка до следующего BLUE или завершение боя.
        """
        assert self.winner is None, "Эпизод завершён — вызовите reset()."

        # Базовая награда за шаг
        step_reward = self.reward_step

        # Подстраховка: если не очередь BLUE, докрутим до его хода
        if self.current_blue_attacker_pos is None:
            self._advance_until_blue_turn()
            if self.winner is not None:
                # Бой завершился ещё до хода BLUE
                return self._obs(), (self.reward_win if self.winner == "blue" else self.reward_loss), True, False, {}

        # === ХОД BLUE (агент) ===
        target_pos = RED_POSITIONS[int(action)]          # перевести индекс действия в реальную позицию (1..6)
        attacker   = self._unit_by_position(self.current_blue_attacker_pos)

        # заранее проверим «незаконность» цели
        victim = self._unit_by_position(target_pos)
        illegal = (victim is None) or (victim.get("Health", 0) <= 0)

        if attacker is not None and self._alive(attacker) and attacker["Initiative"] > 0:
            # Применяем эффекты состояния перед ходом
            self._apply_status_effects(attacker)

            self._log(f"BLUE действие: {attacker['Name']}#{attacker['position']} → pos{target_pos}")
            attacker["Initiative"] = 0                   # в этом раунде синий уже сходил

            # Проверяем, был ли применен штраф для воина
            penalty_applied = self._attack(attacker, target_pos)  # применяем урон (или фиксируем штраф в логах)

            if not penalty_applied:
                self._check_victory_after_hit()             # проверяем победителя

                # Явный штраф за недопустимое действие, если бой продолжается
                if illegal and self.winner is None:
                    step_reward += self.reward_illegal

        # Если бой не завершён — докрутка до следующего BLUE (RED ходит автоматически)
        if self.winner is None:
            self.current_blue_attacker_pos = None
            self._advance_until_blue_turn()

        # Финальная сборка перехода
        if self.winner is None:
            reward, terminated = step_reward, False
        else:
            # Бой завершён - восстанавливаем оригинальные значения Resilience
            for u in self.combined:
                if "original_resilience" in u:
                    u["Resilience"] = u["original_resilience"].copy()
            reward  = self.reward_win if self.winner == "blue" else self.reward_loss
            terminated = True

        return self._obs(), reward, terminated, False, {}

# =================== ОБУЧЕНИЕ PPO + W&B =====================
if __name__ == "__main__":
    # Импортируем Stable-Baselines3 и колбэки прямо внутри main-блока (чтобы при импорте модуля они не требовались)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback

    # Weights & Biases: опционально и безопасно
    from stable_baselines3.common.callbacks import BaseCallback
    try:
        import wandb  # type: ignore
        WANDB_AVAILABLE = True
    except Exception:
        wandb = None
        WANDB_AVAILABLE = False

    try:
        if WANDB_AVAILABLE:
            from wandb.integration.sb3 import WandbCallback  # type: ignore
        else:
            raise ImportError
    except Exception:
        class WandbCallback(BaseCallback):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def _on_step(self) -> bool:
                return True

    # ---------------- ПАРАМЕТРЫ ОБУЧЕНИЯ ----------------
    TOTAL_STEPS    = 1_000_000  # сколько шагов среды сделает PPO (суммарно по всем векторным копиям)
    N_ENVS         = 8          # сколько параллельных копий среды использовать
    VISUALIZE_TEST = True       # включить ли опциональную визуализацию после теста
    FRAME_DELAY    = 0.28       # задержка между «кадрами» визуализации (сек)
    USE_COLOR      = True       # цветные ANSI в консоли (если терминал поддерживает)

    # ------------- ИНИЦИАЛИЗАЦИЯ W&B ЛОГА --------------
    USE_WANDB = bool(WANDB_AVAILABLE) and os.environ.get("WANDB_DISABLED", "false").lower() not in ("1", "true", "yes")
    run = None
    if USE_WANDB:
        try:
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "red-blue-battle"),
                name=os.environ.get("RUN_NAME", f"ppo-{TOTAL_STEPS//1000}k"),
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
                    "reward_illegal": -0.1,
                },
                sync_tensorboard=True,
                save_code=True,
            )
            run_id = run.id
        except Exception as e:
            print(f"[W&B] Disabled: {e}")
            USE_WANDB = False
            run = None
            run_id = time.strftime("%Y%m%d-%H%M%S")
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")

    # -------- ФАБРИКА СРЕД ДЛЯ ВЕКТОРИЗАЦИИ --------
    def make_env():
        # Оборачиваем среду в Monitor, чтобы SB3 считал эпизодические метрики (ep_rew_mean и т.п.)
        return Monitor(BattleEnv(
            reward_win=1.0,
            reward_loss=-1.0,
            reward_step=0.0,
            reward_illegal=-0.1,   # <--- передаём штраф
            log_enabled=False
        ))

    # Векторная среда: N_ENVS параллельных копий для ускоренного сбора опыта
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)  # seed убран

    # Отдельная среда для периодической оценки (EvalCallback)
    eval_env = Monitor(BattleEnv(log_enabled=False, reward_illegal=-0.1))

    # --------------- НАСТРОЙКА МОДЕЛИ PPO ---------------
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,                         # печать прогресса обучения SB3
        n_steps=1024,                      # длина rollout на КАЖДУЮ среду (итого N_ENVS * n_steps шагов на обновление)
        batch_size=2048,                   # размер мини-батча SGD (должен делиться на N_ENVS*n_steps / n_epochs)
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        tensorboard_log=f"./tb_logs/{run_id}",  # куда писать TB-логи (W&B их подхватит)
    )  # seed параметр удалён

    # Колбэк W&B: сохранение градиентов/модели и синк метрик
    callbacks = []
    if USE_WANDB:
        wandb_cb = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=f"./models/{run_id}",
            model_save_freq=10_000,
            verbose=2,
        )
        callbacks.append(wandb_cb)

    # Колбэк оценки: периодически замеряем среднюю награду на eval_env
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{run_id}/best",
        log_path=f"./eval/{run_id}",
        eval_freq=10_000,               # каждые 10k шагов
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_cb)

    # Запуск обучения с колбэками (W&B + eval)
    # model.learn(total_timesteps=TOTAL_STEPS, callback=CallbackList(callbacks))

    # Сохранение финальной модели
    model.save("ppo_blue_vs_red")
    if USE_WANDB and run is not None:
        run.finish()

    # ================= ТЕСТОВЫЙ ПРОГОН С ЛОГАМИ =================
    print("\n=== ТЕСТОВЫЙ ПРОГОН С ЛОГАМИ ===")
    test_env = BattleEnv(log_enabled=True, reward_illegal=-0.1)  # включаем логи; штраф тот же
    obs, info = test_env.reset()  # seed убран

    # Собираем все строки логов (будут использованы и для опциональной визуализации)
    all_logs = []
    all_logs += test_env.pop_pretty_events()    # логи, накопленные во время reset()/автодокрутки

    done = False
    total_reward = 0.0
    step_i = 0
    while not done:
        step_i += 1
        action, _ = model.predict(obs, deterministic=True)  # в тесте используем детерминированную политику
        target_pos = RED_POSITIONS[int(action)]
        chosen_line = f"[STEP {step_i}] Агент выбирает action={int(action)} → атака RED pos{target_pos}"
        print("\n" + chosen_line)
        all_logs.append(chosen_line)

        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Печатаем логи текущего шага и добавляем их в общий список
        new_lines = test_env.pop_pretty_events()
        all_logs += new_lines
        for line in new_lines:
            print(line)

    print("\n=== ЭПИЗОД ЗАВЕРШЁН ===")
    print("Победитель:", test_env.winner.upper(), "| Суммарная награда:", total_reward)

    # ============== ВИЗУАЛИЗАЦИЯ ПО ЛОГАМ (matplotlib, как в checkpointchek) ==============
    # Более наглядная графическая анимация, разбирающая строки логов и рисующая поле боя.
    if VISUALIZE_TEST:
        import math
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib.patches import FancyArrowPatch
        try:
            from IPython.display import display
        except Exception:
            display = None

        VISUAL_SPEED_MULT = 8.0

        # --- сетка поля
        RED_FRONT, RED_BACK   = [1,2,3],   [4,5,6]
        BLUE_FRONT, BLUE_BACK = [7,8,9],   [10,11,12]
        COL_X = {0: 0.18, 1: 0.50, 2: 0.82}
        Y_BLUE_BACK, Y_BLUE_FRONT = 0.88, 0.70
        Y_RED_FRONT,  Y_RED_BACK  = 0.30, 0.12
        SLOT_W, SLOT_H = 0.28, 0.13
        HP_H = 0.028

        # --- состояние (имена и стартовые HP для визуализации)
        state = {u["position"]: {"team": u["team"], "name": u["Name"], "hp": float(u["Health"]), "maxhp": float(u["Health"]),
                                "burn": float(u["Burn"]), "freeze": float(u["Freeze"]), "poison": float(u["Poison"]) }
                 for u in (UNITS_RED + UNITS_BLUE)}

        fig, ax = plt.subplots(figsize=(12, 8))
        handle = None
        IS_NOTEBOOK = False
        if display is not None:
            try:
                handle = display(fig, display_id=True)
                IS_NOTEBOOK = handle is not None
            except Exception:
                handle = None
                IS_NOTEBOOK = False
        if IS_NOTEBOOK:
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

        def draw_unit(ax, pos, is_active: bool = False):
            u = state[pos]; x, y = pos_to_xy(pos)
            # Скрываем задний ряд мёртвых (чтобы не загромождать)
            if pos in (RED_BACK + BLUE_BACK) and u["hp"] <= 0:
                return
            # Цвет карточки в зависимости от команды и типа
            if u.get("Type") == "Death":
                card_color = (0.70, 0.30, 0.90, 0.16)  # Фиолетовый для Death
            elif u.get("Type") == "PoisonDragon":
                card_color = (0.50, 0.90, 0.30, 0.16)  # Зеленый для PoisonDragon
            elif u.get("Type") == "Asterot":
                card_color = (0.90, 0.70, 0.30, 0.16)  # Оранжевый для Asterot
            elif u["team"] == "red":
                card_color = (0.90, 0.30, 0.30, 0.16)
            else:
                card_color = (0.30, 0.45, 0.90, 0.16)
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

            # Отображаем эффекты состояния и Twohits в правом верхнем углу
            effects_text = ""
            if u["burn"] > 0 or u["freeze"] > 0 or u["poison"] > 0 or u.get("Twohits", 0) > 0:
                effects = []
                if u.get("Twohits", 0) > 0:
                    effects.append(f"⭐{int(u['Twohits'])}")
                if u["burn"] > 0:
                    effects.append(f"B:{int(u['burn'])}")
                if u["freeze"] > 0:
                    effects.append(f"F:{int(u['freeze'])}")
                if u["poison"] > 0:
                    effects.append(f"P:{int(u['poison'])}")
                effects_text = " ".join(effects)
                ax.text(x + SLOT_W - 0.012, y + SLOT_H*0.78, effects_text,
                        fontsize=8, fontweight="bold", ha="right", va="center", color="red")

            hp_x, hp_y = x + 0.012, y + SLOT_H*0.10
            hp_w = SLOT_W - 0.024
            ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w, HP_H, facecolor=(0.88,0.88,0.88), edgecolor='none'))
            frac = max(0.0, min(1.0, u["hp"]/u["maxhp"])) if u["maxhp"] > 1e-9 else 0.0
            ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w*frac, HP_H, facecolor=(0.15,0.70,0.25), edgecolor='none'))
            ax.text(hp_x + hp_w/2, hp_y + HP_H/2, f"{int(max(0,u['hp']))}/{int(u['maxhp'])}",
                    fontsize=9, ha="center", va="center", color="black")

            ax.text(x + SLOT_W/2, y - 0.008, f"pos{pos}", fontsize=8, ha="center", va="top", color=(0.2,0.2,0.2))

        def draw_attack_arrow(ax, src_pos:int, dst_pos:int, team:str,
                              text: str|None = None, style: str = "solid",
                              color: tuple|None = None, alpha: float = 0.95,
                              curve: float = 0.18, lw: float = 2.6):
            sx, sy = pos_center(src_pos)
            dx, dy = pos_center(dst_pos)
            vx, vy = dx - sx, dy - sy
            dist = math.hypot(vx, vy) + 1e-9
            pad = 0.06
            sx += vx/dist * pad; sy += vy/dist * pad
            dx -= vx/dist * pad; dy -= vy/dist * pad
            if color is None:
                color = (0.20, 0.35, 0.85) if team == "BLUE" else (0.85, 0.25, 0.25)
            con_style = f"arc3,rad={curve}" if abs(vx) > 0.01 and abs(vy) > 0.01 else "arc3,rad=0.0"
            arrow = FancyArrowPatch((sx, sy), (dx, dy), arrowstyle="-|>",
                                    mutation_scale=14, linewidth=lw,
                                    linestyle="--" if style=="dashed" else "solid",
                                    color=color, alpha=alpha, connectionstyle=con_style)
            ax.add_patch(arrow)
            if text:
                mx, my = (sx + dx)/2, (sy + dy)/2
                ax.text(mx, my + 0.03, text, fontsize=10, fontweight="bold",
                        ha="center", va="center", color=color)

        # --- регексы (совместимы с текущими логами simpleenv)
        atk_re         = re.compile(r'^(RED|BLUE)\s+[^#]+#(\d+)\s+→\s+(RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)')
        kill_re        = re.compile(r'^✖\s+(RED|BLUE)\s+[^#]+#(\d+)\s+выведен из строя\.')
        vict_re        = re.compile(r'^🏆 Победа (RED|BLUE)!')
        blue_turn_re   = re.compile(r'^Ход BLUE:\s+[^#]+#(\d+)')
        red_turn_re    = re.compile(r'^RED ход:\s+[^#]+#(\d+)')
        blue_action_re = re.compile(r'^BLUE действие:\s+[^#]+#(\d+)\s+→\s+pos(\d+)')
        # Регексы для эффектов состояния
        status_damage_re = re.compile(r'^💀\s+(RED|BLUE)\s+[^#]+#(\d+):\s+(.+?)\s+наносит\s+(\d+)\s+урона')
        status_clear_re  = re.compile(r'^✨\s+(RED|BLUE)\s+[^#]+#(\d+):\s+эффекты\s+(.+?)\s+исчезают!')

        def draw_board(arrows: list[dict] = None, headline: str = "", active_pos: int | None = None):
            ax.cla()
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            ax.plot([0.02, 0.98], [0.50, 0.50], color=(0.6,0.6,0.6), lw=1.2, ls="--", alpha=0.7)
            ax.text(0.01, 0.96, "BLUE (top)", color=(0.2,0.35,0.8), fontsize=12, fontweight="bold", ha="left")
            ax.text(0.01, 0.04, "RED (bottom)", color=(0.8,0.25,0.25), fontsize=12, fontweight="bold", ha="left")

            arrows = arrows or []
            for pos in (BLUE_BACK + BLUE_FRONT + RED_FRONT + RED_BACK):
                draw_unit(ax, pos, is_active=(active_pos == pos))

            for a in arrows:
                draw_attack_arrow(ax, a["src"], a["dst"], a["team"],
                                  text=a.get("text"), style=a.get("style","solid"),
                                  color=a.get("color"), alpha=a.get("alpha",0.95),
                                  curve=a.get("curve",0.18), lw=a.get("lw",2.6))

            if headline:
                ax.text(0.5, 0.52, headline[:140], fontsize=12, ha="center", va="bottom", color=(0.15,0.15,0.15))

            fig.canvas.draw()
            if handle is not None:
                handle.update(fig)
            else:
                plt.draw(); plt.pause(0.001)

        # --- проигрываем логи покадрово
        current_actor_pos = None
        draw_board(headline="Начало боя", active_pos=current_actor_pos)

        last_selection = None
        for line in all_logs:
            arrows_now = []
            headline = ""

            m = blue_turn_re.match(line)
            if m:
                current_actor_pos = int(m.group(1))
                headline = line
                draw_board(arrows_now, headline, active_pos=current_actor_pos); time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.05); continue

            m = red_turn_re.match(line)
            if m:
                current_actor_pos = int(m.group(1))
                headline = line
                draw_board(arrows_now, headline, active_pos=current_actor_pos); time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.05); continue

            m = atk_re.match(line)
            if m:
                atk_team = m.group(1); atk_pos = int(m.group(2))
                vic_pos = int(m.group(4)); dmg = int(m.group(5)); after = int(m.group(7))
                if vic_pos in state: state[vic_pos]["hp"] = after
                current_actor_pos = atk_pos
                arrows_now.append({"src": atk_pos, "dst": vic_pos, "team": atk_team, "text": f"-{dmg}"})
                headline = line
                draw_board(arrows_now, headline, active_pos=current_actor_pos); time.sleep(FRAME_DELAY * VISUAL_SPEED_MULT); continue

            m = status_damage_re.match(line)
            if m:
                team = m.group(1); pos = int(m.group(2)); dmg = int(m.group(4))
                if pos in state:
                    # Обновляем HP юнита после получения урона от эффектов
                    state[pos]["hp"] = max(0, state[pos]["hp"] - dmg)
                headline = line
                draw_board([], headline, active_pos=current_actor_pos); time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2); continue

            m = blue_action_re.match(line)
            if m:
                src = int(m.group(1)); dst = int(m.group(2))
                last_selection = {"src": src, "dst": dst, "team": "BLUE"}
                current_actor_pos = src
                arrows_now.append({"src": src, "dst": dst, "team": "BLUE",
                                   "style":"dashed", "alpha":0.65, "lw":2.0})
                headline = line
                draw_board(arrows_now, headline, active_pos=current_actor_pos); time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1); continue

            m = kill_re.match(line)
            if m:
                pos = int(m.group(2))
                if pos in state: state[pos]["hp"] = 0
                headline = line
                draw_board([], headline, active_pos=current_actor_pos); time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1); continue

            m = status_damage_re.match(line)
            if m:
                pos = int(m.group(2))
                if pos in state:
                    # Обновляем HP (это уже делается в других местах, но на всякий случай)
                    pass
                headline = line
                draw_board([], headline, active_pos=current_actor_pos); time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2); continue

            m = status_clear_re.match(line)
            if m:
                pos = int(m.group(2))
                effects_str = m.group(3)
                if pos in state:
                    # Обнуляем эффекты в визуализации (теперь эффекты могут исчезать по отдельности)
                    if "Burn" in effects_str:
                        state[pos]["burn"] = 0
                    if "Freeze" in effects_str:
                        state[pos]["freeze"] = 0
                    if "Poison" in effects_str:
                        state[pos]["poison"] = 0
                headline = line
                draw_board([], headline, active_pos=current_actor_pos); time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2); continue

            if vict_re.match(line) or line.startswith("— ") or "RED ход" in line:
                headline = line
                draw_board([], headline, active_pos=current_actor_pos); time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1); continue

            # Прочие строки
            draw_board([], line, active_pos=current_actor_pos); time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)

        draw_board([], "Конец эпизода", active_pos=None)